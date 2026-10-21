/*
 * cholla_nif.c - Erlang NIF bindings to the Cactus Compute Needle engine.
 *
 * The engine (libneedle) is a prebuilt, self-contained shared library that
 * keeps one process-global, non-thread-safe model per kind (text, speech)
 * and a pool of worker threads sized to every online CPU. To run several
 * engines in one VM, each instance dlopens a private copy of the library
 * (an anonymous memfd on Linux, else an unlinked temp file), so every copy
 * has its own globals. An instance is a NIF resource with its own mutex.
 *
 * The workers spin while they wait, so instances that together run more
 * threads than there are cores slow each other down by orders of magnitude.
 * On Linux an instance can therefore be given a thread count and a CPU set:
 * its copy's GOT entries for sysconf() and sched_setaffinity() are redirected
 * so the engine sizes its pool to the thread count and its workers inherit
 * the CPU set instead of pinning themselves to cores 1..N-1.
 *
 * Anything that runs the model is scheduled on a dirty CPU scheduler.
 *
 * Conventions on the Erlang side:
 *   - strings are binaries or iodata; optional strings may be the atom nil
 *   - PCM audio is a binary of native-endian 32-bit floats, 16 kHz mono
 *   - results are {ok, Value} or {error, Reason}
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifdef __linux__
#include <elf.h>
#include <link.h>
#include <pthread.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#define THREAD_CONTROL 1
#else
#define THREAD_CONTROL 0
#endif

#include <erl_nif.h>

#include "needle.h"

typedef int (*load_fn)(const unsigned char *, unsigned long long);
typedef int (*models_fn)(void);
typedef const char *(*last_error_fn)(void);
typedef int (*init_fn)(const char *, const char *, const char *);
typedef int (*complete_fn)(const char *, const float *, int, int, char *, int);
typedef void (*set_audio_fn)(const char *, const char *, int);
typedef void (*reset_fn)(void);
typedef int (*transcribe_fn)(const float *, int, const char *, const char *, int, char *, int);
typedef int (*stream_process_fn)(const float *, int, const char *, const char *, char *, int);
typedef int (*stream_stop_fn)(char *, int);
typedef int (*embed_fn)(const char *, const float *, int, float *, int);

typedef struct {
    ErlNifMutex *lock;
    void *handle;
    char *path;  /* the library this instance was copied from */
    int threads; /* engine threads, counting the caller; 0 = engine default */
    int counted; /* threads charged to the VM-wide total */
#if THREAD_CONTROL
    int has_cpus;
    cpu_set_t cpus;
#endif
    load_fn load;
    models_fn models;
    last_error_fn last_error;
    init_fn init;
    complete_fn complete;
    set_audio_fn set_audio;
    reset_fn reset;
    transcribe_fn transcribe;
    stream_process_fn stream_process;
    stream_stop_fn stream_stop;
    embed_fn embed;
} instance_t;

static ErlNifResourceType *instance_type;

/* Guards the counters below and serialises dlopen of new copies. */
static ErlNifMutex *global_lock;
static int open_instances;
static int total_threads;

static ERL_NIF_TERM atom_ok, atom_error, atom_nil, atom_true, atom_false, atom_buffer_too_small,
    atom_path, atom_threads, atom_cpus, atom_thread_control, atom_vm_threads, atom_instances;

/* Engine outputs larger than this are treated as an engine fault. */
#define MAX_OUTPUT (64 * 1024 * 1024)

/* ------------------------------------------------------------------------ */
/* thread control                                                            */

/* Set on the calling thread for the duration of an engine call, so the
   sysconf hook answers for the instance being called. */
static __thread int current_threads;

#if THREAD_CONTROL
static long hook_sysconf(int name) {
    if ((name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF) && current_threads > 0)
        return current_threads;
    return sysconf(name);
}

/* Workers keep the CPU set they inherit from the thread that created them. */
static int hook_sched_setaffinity(pid_t pid, size_t size, const cpu_set_t *mask) {
    return 0;
}

typedef struct {
    ElfW(Addr) base;
    ElfW(Addr) relro_start, relro_end;
} relro_t;

static int find_relro(struct dl_phdr_info *info, size_t size, void *data) {
    relro_t *r = data;
    if (info->dlpi_addr != r->base) return 0;
    for (int i = 0; i < info->dlpi_phnum; i++)
        if (info->dlpi_phdr[i].p_type == PT_GNU_RELRO) {
            r->relro_start = r->base + info->dlpi_phdr[i].p_vaddr;
            r->relro_end = r->relro_start + info->dlpi_phdr[i].p_memsz;
        }
    return 1;
}

static int is_data_reloc(unsigned long type) {
#if defined(__x86_64__)
    return type == R_X86_64_JUMP_SLOT || type == R_X86_64_GLOB_DAT || type == R_X86_64_64;
#elif defined(__aarch64__)
    return type == R_AARCH64_JUMP_SLOT || type == R_AARCH64_GLOB_DAT || type == R_AARCH64_ABS64;
#else
    return 0;
#endif
}

/* Points every GOT slot of `handle` that refers to `symbol` at `target`.
   Returns the number of slots rewritten, or -1 on error. */
static int patch_got(void *handle, const char *symbol, void *target) {
    struct link_map *map;
    if (dlinfo(handle, RTLD_DI_LINKMAP, &map) != 0) return -1;
    ElfW(Addr) base = map->l_addr;

    const ElfW(Sym) *symtab = NULL;
    const char *strtab = NULL;
    const ElfW(Rela) *tables[2] = {NULL, NULL};
    size_t sizes[2] = {0, 0};
    long pltrel = DT_RELA;
    /* glibc relocates these entries in place; musl leaves them as offsets. */
#define ADDR(d) ((d)->d_un.d_ptr < base ? (d)->d_un.d_ptr + base : (d)->d_un.d_ptr)
    for (const ElfW(Dyn) *d = map->l_ld; d->d_tag != DT_NULL; d++) {
        switch (d->d_tag) {
        case DT_SYMTAB: symtab = (const void *)ADDR(d); break;
        case DT_STRTAB: strtab = (const void *)ADDR(d); break;
        case DT_RELA: tables[0] = (const void *)ADDR(d); break;
        case DT_RELASZ: sizes[0] = d->d_un.d_val; break;
        case DT_JMPREL: tables[1] = (const void *)ADDR(d); break;
        case DT_PLTRELSZ: sizes[1] = d->d_un.d_val; break;
        case DT_PLTREL: pltrel = (long)d->d_un.d_val; break;
        }
    }
#undef ADDR
    if (!symtab || !strtab || pltrel != DT_RELA) return -1;

    relro_t relro = {base, 0, 0};
    dl_iterate_phdr(find_relro, &relro);
    long page = sysconf(_SC_PAGESIZE);

    int patched = 0;
    for (int t = 0; t < 2; t++) {
        for (size_t i = 0; tables[t] && i < sizes[t] / sizeof(ElfW(Rela)); i++) {
            const ElfW(Rela) *rel = &tables[t][i];
            unsigned long sym = ELF64_R_SYM(rel->r_info);
            if (!sym || !is_data_reloc(ELF64_R_TYPE(rel->r_info))) continue;
            if (strcmp(strtab + symtab[sym].st_name, symbol) != 0) continue;

            ElfW(Addr) slot = base + rel->r_offset;
            void *start = (void *)(slot & ~(ElfW(Addr))(page - 1));
            size_t len = (slot + sizeof(void *)) - (ElfW(Addr))start;
            if (mprotect(start, len, PROT_READ | PROT_WRITE) != 0) return -1;
            *(void **)slot = target;
            if (slot >= relro.relro_start && slot < relro.relro_end) mprotect(start, len, PROT_READ);
            patched++;
        }
    }
    return patched;
}
#endif

/* ------------------------------------------------------------------------ */
/* term helpers                                                              */

static ERL_NIF_TERM make_binary(ErlNifEnv *env, const void *data, size_t len) {
    ERL_NIF_TERM term;
    unsigned char *buf = enif_make_new_binary(env, len, &term);
    if (len) memcpy(buf, data, len);
    return term;
}

static ERL_NIF_TERM ok(ErlNifEnv *env, ERL_NIF_TERM value) {
    return enif_make_tuple2(env, atom_ok, value);
}

static ERL_NIF_TERM error_str(ErlNifEnv *env, const char *message) {
    return enif_make_tuple2(env, atom_error, make_binary(env, message, strlen(message)));
}

/* "output buffer is too small: 384 bytes needed" -> 385, else 0. */
static int needed_bytes(instance_t *in) {
    const char *detail = in->last_error();
    if (!detail || !strstr(detail, "too small")) return 0;
    const char *colon = strrchr(detail, ':');
    long n = colon ? strtol(colon + 1, NULL, 10) : 0;
    return (n > 0 && n < MAX_OUTPUT) ? (int)n + 1 : 0;
}

/* Error from the engine's own last_error, or a generic message with the code.
   A too-small output buffer becomes {error, {buffer_too_small, Needed}}. */
static ERL_NIF_TERM engine_error(ErlNifEnv *env, instance_t *in, const char *what, int code) {
    int needed = needed_bytes(in);
    if (needed)
        return enif_make_tuple2(env, atom_error,
                                enif_make_tuple2(env, atom_buffer_too_small, enif_make_int(env, needed)));
    const char *detail = in->last_error();
    if (detail && *detail) return error_str(env, detail);
    char message[128];
    snprintf(message, sizeof message, "%s failed (code %d)", what, code);
    return error_str(env, message);
}

/* Copies a binary or iolist argument into a NUL-terminated C string.
   nil yields NULL when allow_nil is set. Returns 0 on a bad argument. */
static int get_cstring(ErlNifEnv *env, ERL_NIF_TERM term, int allow_nil, char **out) {
    ErlNifBinary bin;
    *out = NULL;
    if (allow_nil && enif_is_identical(term, atom_nil)) return 1;
    if (!enif_inspect_iolist_as_binary(env, term, &bin)) return 0;
    if (memchr(bin.data, '\0', bin.size)) return 0;
    char *s = enif_alloc(bin.size + 1);
    if (!s) return 0;
    memcpy(s, bin.data, bin.size);
    s[bin.size] = '\0';
    *out = s;
    return 1;
}

static void free_cstring(char *s) {
    if (s) enif_free(s);
}

/* Reads PCM (native f32 binary) or nil. Returns 0 on a bad argument. */
static int get_pcm(ErlNifEnv *env, ERL_NIF_TERM term, int allow_nil, const float **pcm, int *samples) {
    ErlNifBinary bin;
    *pcm = NULL;
    *samples = 0;
    if (allow_nil && enif_is_identical(term, atom_nil)) return 1;
    if (!enif_inspect_binary(env, term, &bin)) return 0;
    if (bin.size % sizeof(float) != 0) return 0;
    if (bin.size / sizeof(float) > 0x7fffffff) return 0;
    *pcm = (const float *)bin.data;
    *samples = (int)(bin.size / sizeof(float));
    return 1;
}

static int get_bool(ErlNifEnv *env, ERL_NIF_TERM term, int *out) {
    if (enif_is_identical(term, atom_true)) { *out = 1; return 1; }
    if (enif_is_identical(term, atom_false)) { *out = 0; return 1; }
    return 0;
}

static int get_capacity(ErlNifEnv *env, ERL_NIF_TERM term, int *out) {
    return enif_get_int(env, term, out) && *out > 0 && *out <= MAX_OUTPUT;
}

static int get_instance(ErlNifEnv *env, ERL_NIF_TERM term, instance_t **out) {
    return enif_get_resource(env, term, instance_type, (void **)out);
}

/* ------------------------------------------------------------------------ */
/* calling into an instance                                                  */

typedef struct {
#if THREAD_CONTROL
    int restore;
    cpu_set_t saved;
#endif
    int unused;
} call_scope;

/* Locks the instance and gives the calling thread its thread count and CPU
   set, which worker threads created during the call inherit. */
static void enter(instance_t *in, call_scope *scope) {
    enif_mutex_lock(in->lock);
    current_threads = in->threads;
#if THREAD_CONTROL
    scope->restore = 0;
    if (in->has_cpus &&
        pthread_getaffinity_np(pthread_self(), sizeof scope->saved, &scope->saved) == 0 &&
        pthread_setaffinity_np(pthread_self(), sizeof in->cpus, &in->cpus) == 0)
        scope->restore = 1;
#endif
}

static void leave(instance_t *in, call_scope *scope) {
#if THREAD_CONTROL
    if (scope->restore) pthread_setaffinity_np(pthread_self(), sizeof scope->saved, &scope->saved);
#endif
    current_threads = 0;
    enif_mutex_unlock(in->lock);
}

/* ------------------------------------------------------------------------ */
/* open_engine(Path, Threads, Cpus | nil) -> {ok, Instance} | {error, Reason} */

/* Makes a private copy of the library and returns a path dlopen can map.
   *fd stays open for the life of the instance when the copy is a memfd. */
static int private_copy(const char *path, char *out, size_t out_size, int *fd, char *err, size_t err_size) {
    FILE *src = fopen(path, "rb");
    if (!src) {
        snprintf(err, err_size, "cannot read %s: %s", path, strerror(errno));
        return 0;
    }

    *fd = -1;
    int temp = 0;
#ifdef __linux__
    *fd = (int)syscall(SYS_memfd_create, "libneedle", 0);
    if (*fd >= 0) snprintf(out, out_size, "/proc/self/fd/%d", *fd);
#endif
    if (*fd < 0) {
        /* Next to the original, which is on a filesystem that allows mapping code. */
        snprintf(out, out_size, "%s.XXXXXX", path);
        *fd = mkstemp(out);
        temp = 1;
    }
    if (*fd < 0) {
        snprintf(err, err_size, "cannot copy %s: %s", path, strerror(errno));
        fclose(src);
        return 0;
    }

    char buf[65536];
    size_t n;
    int good = 1;
    while (good && (n = fread(buf, 1, sizeof buf, src)) > 0)
        for (size_t off = 0; good && off < n;) {
            ssize_t w = write(*fd, buf + off, n - off);
            if (w <= 0) good = 0;
            else off += (size_t)w;
        }
    good = good && !ferror(src);
    fclose(src);

    if (temp) {
        close(*fd);
        *fd = -1;
    }
    if (!good) {
        snprintf(err, err_size, "cannot copy %s", path);
        if (temp) unlink(out);
        return 0;
    }
    return temp ? 2 : 1;
}

static ERL_NIF_TERM nif_open_engine(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *path;
    int threads;
    if (!enif_get_int(env, argv[1], &threads) || threads < 0 || threads > 4096)
        return enif_make_badarg(env);

#if THREAD_CONTROL
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
#endif
    int has_cpus = 0;
    if (!enif_is_identical(argv[2], atom_nil)) {
        ERL_NIF_TERM list = argv[2], head;
        int cpu;
        if (!enif_is_list(env, list)) return enif_make_badarg(env);
        while (enif_get_list_cell(env, list, &head, &list)) {
            if (!enif_get_int(env, head, &cpu) || cpu < 0 || cpu >= 1024) return enif_make_badarg(env);
#if THREAD_CONTROL
            if (cpu >= CPU_SETSIZE) return enif_make_badarg(env);
            CPU_SET(cpu, &cpus);
#endif
            has_cpus = 1;
        }
    }
    if (!get_cstring(env, argv[0], 0, &path)) return enif_make_badarg(env);

    char copy[4096], err[512];
    int fd = -1;
    enif_mutex_lock(global_lock);
    int kind = private_copy(path, copy, sizeof copy, &fd, err, sizeof err);
    if (!kind) {
        enif_mutex_unlock(global_lock);
        free_cstring(path);
        return error_str(env, err);
    }

    void *handle = dlopen(copy, RTLD_NOW | RTLD_LOCAL);
    if (kind == 2) unlink(copy);
    if (!handle) {
        const char *detail = dlerror();
        ERL_NIF_TERM result = error_str(env, detail ? detail : "dlopen failed");
        if (fd >= 0) close(fd);
        enif_mutex_unlock(global_lock);
        free_cstring(path);
        return result;
    }

    instance_t *in = enif_alloc_resource(instance_type, sizeof(instance_t));
    memset(in, 0, sizeof *in);
    in->handle = handle;
    in->path = path;
    in->threads = THREAD_CONTROL ? threads : 0;
#if THREAD_CONTROL
    in->has_cpus = has_cpus;
    in->cpus = cpus;
#endif
    in->lock = enif_mutex_create("cholla_instance");

    const char *missing = NULL;
#define RESOLVE(field, name)                                \
    do {                                                    \
        *(void **)(&in->field) = dlsym(handle, name);       \
        if (!in->field && !missing) missing = name;         \
    } while (0)
    RESOLVE(load, "needle_load");
    RESOLVE(models, "needle_models");
    RESOLVE(last_error, "needle_last_error");
    RESOLVE(init, "needle_init");
    RESOLVE(complete, "needle_complete");
    RESOLVE(set_audio, "needle_set_audio");
    RESOLVE(reset, "needle_reset");
    RESOLVE(transcribe, "needle_transcribe");
    RESOLVE(stream_process, "needle_stream_transcribe_process");
    RESOLVE(stream_stop, "needle_stream_transcribe_stop");
    RESOLVE(embed, "needle_embed");
#undef RESOLVE

    if (missing) snprintf(err, sizeof err, "%s is not a Needle 3 engine: missing symbol %s", path, missing);

#if THREAD_CONTROL
    if (!missing && (threads > 0 || has_cpus) &&
        (patch_got(handle, "sysconf", (void *)hook_sysconf) <= 0 ||
         patch_got(handle, "sched_setaffinity", (void *)hook_sched_setaffinity) <= 0)) {
        snprintf(err, sizeof err, "cannot limit the threads of %s: its imports are not as expected", path);
        missing = err;
    }
#endif

    ERL_NIF_TERM result;
    if (missing) {
        /* Nothing has run in the copy, so it can go. */
        dlclose(handle);
        if (fd >= 0) close(fd);
        in->handle = NULL;
        result = error_str(env, err);
    } else {
        /* The handle (and memfd) live for the rest of the VM: the engine's worker
           threads cannot be stopped, so the library is never unloaded. */
        in->counted = in->threads > 0 ? in->threads : (int)sysconf(_SC_NPROCESSORS_ONLN);
        total_threads += in->counted;
        open_instances++;
        result = ok(env, enif_make_resource(env, in));
    }
    enif_mutex_unlock(global_lock);
    enif_release_resource(in);
    return result;
}

static void instance_dtor(ErlNifEnv *env, void *obj) {
    instance_t *in = obj;
    if (in->handle) {
        /* Idle workers block rather than spin, so they stop counting. */
        enif_mutex_lock(global_lock);
        total_threads -= in->counted;
        open_instances--;
        enif_mutex_unlock(global_lock);
    }
    if (in->lock) enif_mutex_destroy(in->lock);
    free_cstring(in->path);
}

/* info(Instance) -> #{path, threads, cpus, thread_control, vm_threads, instances} */
static ERL_NIF_TERM nif_info(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);

    ERL_NIF_TERM cpus = atom_nil;
#if THREAD_CONTROL
    if (in->has_cpus) {
        cpus = enif_make_list(env, 0);
        for (int i = CPU_SETSIZE - 1; i >= 0; i--)
            if (CPU_ISSET(i, &in->cpus)) cpus = enif_make_list_cell(env, enif_make_int(env, i), cpus);
    }
#endif
    enif_mutex_lock(global_lock);
    int vm_threads = total_threads, instances = open_instances;
    enif_mutex_unlock(global_lock);

    ERL_NIF_TERM keys[] = {atom_path, atom_threads, atom_cpus, atom_thread_control, atom_vm_threads, atom_instances};
    ERL_NIF_TERM values[] = {
        make_binary(env, in->path, strlen(in->path)),
        in->threads > 0 ? enif_make_int(env, in->threads) : atom_nil,
        cpus,
        THREAD_CONTROL ? atom_true : atom_false,
        enif_make_int(env, vm_threads),
        enif_make_int(env, instances),
    };
    ERL_NIF_TERM map;
    enif_make_map_from_arrays(env, keys, values, 6, &map);
    return map;
}

/* ------------------------------------------------------------------------ */
/* load(Instance, Cact) -> {ok, ModelsMask} | {error, Reason}                */

static ERL_NIF_TERM nif_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    ErlNifBinary cact;
    if (!get_instance(env, argv[0], &in) || !enif_inspect_binary(env, argv[1], &cact))
        return enif_make_badarg(env);

    call_scope scope;
    enter(in, &scope);
    /* The engine copies the archive, so the binary need not outlive the call. */
    int rc = in->load(cact.data, (unsigned long long)cact.size);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, in, "needle_load", rc) : ok(env, enif_make_int(env, in->models()));
    leave(in, &scope);
    return result;
}

/* models(Instance) -> ModelsMask (NEEDLE_TEXT | NEEDLE_SPEECH) */
static ERL_NIF_TERM nif_models(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    enif_mutex_lock(in->lock);
    int mask = in->models();
    enif_mutex_unlock(in->lock);
    return enif_make_int(env, mask);
}

/* ------------------------------------------------------------------------ */
/* init(Instance, System, ToolsJson, ToolIndexPath | nil) -> {ok, PrefixTokens} | {error, R} */

static ERL_NIF_TERM nif_engine_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    char *system = NULL, *tools = NULL, *index = NULL;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    if (!get_cstring(env, argv[1], 0, &system) || !get_cstring(env, argv[2], 0, &tools) ||
        !get_cstring(env, argv[3], 1, &index)) {
        free_cstring(system); free_cstring(tools); free_cstring(index);
        return enif_make_badarg(env);
    }

    call_scope scope;
    enter(in, &scope);
    int rc = in->init(system, tools, index);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, in, "needle_init", rc) : ok(env, enif_make_int(env, rc));
    leave(in, &scope);
    free_cstring(system); free_cstring(tools); free_cstring(index);
    return result;
}

/* ------------------------------------------------------------------------ */
/* complete(Instance, Text | nil, Pcm | nil, MaxNewTokens, Capacity) -> {ok, Json} | {error, R}
   Not retried on a small buffer: the engine may already have advanced the
   conversation, so the caller decides. */

static ERL_NIF_TERM nif_complete(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    char *text = NULL;
    const float *pcm;
    int samples, max_new_tokens, capacity;
    if (!get_instance(env, argv[0], &in) || !get_pcm(env, argv[2], 1, &pcm, &samples) ||
        !enif_get_int(env, argv[3], &max_new_tokens) || max_new_tokens <= 0 ||
        !get_capacity(env, argv[4], &capacity) || !get_cstring(env, argv[1], 1, &text))
        return enif_make_badarg(env);
    if ((text == NULL) == (pcm == NULL)) {
        free_cstring(text);
        return enif_make_badarg(env); /* exactly one of text and pcm */
    }

    char *out = enif_alloc(capacity);
    if (!out) { free_cstring(text); return error_str(env, "out of memory"); }
    out[0] = '\0';

    call_scope scope;
    enter(in, &scope);
    int rc = in->complete(text, pcm, samples, max_new_tokens, out, capacity);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, in, "needle_complete", rc)
                                 : ok(env, make_binary(env, out, strnlen(out, capacity)));
    leave(in, &scope);
    enif_free(out);
    free_cstring(text);
    return result;
}

/* set_audio(Instance, Language | nil, Keywords | nil, WordTimestamps) -> ok */
static ERL_NIF_TERM nif_set_audio(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    char *language = NULL, *keywords = NULL;
    int word_timestamps;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    if (!get_bool(env, argv[3], &word_timestamps) || !get_cstring(env, argv[1], 1, &language) ||
        !get_cstring(env, argv[2], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }
    call_scope scope;
    enter(in, &scope);
    in->set_audio(language, keywords, word_timestamps); /* strings are copied */
    leave(in, &scope);
    free_cstring(language); free_cstring(keywords);
    return atom_ok;
}

/* reset(Instance) -> ok */
static ERL_NIF_TERM nif_reset(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    call_scope scope;
    enter(in, &scope);
    in->reset();
    leave(in, &scope);
    return atom_ok;
}

/* ------------------------------------------------------------------------ */
/* transcribe(Instance, Pcm, Language | nil, Keywords | nil, WordTimestamps, Capacity)
     -> {ok, Json} | {error, R}
   Stateless, so a too-small buffer is retried once at the size the engine asks for. */

static ERL_NIF_TERM nif_transcribe(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    const float *pcm;
    int samples, word_timestamps, capacity;
    char *language = NULL, *keywords = NULL;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    if (!get_pcm(env, argv[1], 0, &pcm, &samples) || !get_bool(env, argv[4], &word_timestamps) ||
        !get_capacity(env, argv[5], &capacity) || !get_cstring(env, argv[2], 1, &language) ||
        !get_cstring(env, argv[3], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM result;
    char *out = NULL;
    call_scope scope;
    enter(in, &scope);
    int rc = -1;
    for (int attempt = 0; attempt < 2; attempt++) {
        enif_free(out);
        out = enif_alloc(capacity);
        if (!out) break;
        out[0] = '\0';
        rc = in->transcribe(pcm, samples, language, keywords, word_timestamps, out, capacity);
        if (rc >= 0) break;
        int needed = needed_bytes(in);
        if (!needed || needed <= capacity) break;
        capacity = needed;
    }
    if (!out) result = error_str(env, "out of memory");
    else if (rc < 0) result = engine_error(env, in, "needle_transcribe", rc);
    else result = ok(env, make_binary(env, out, strnlen(out, capacity)));
    leave(in, &scope);
    enif_free(out);
    free_cstring(language); free_cstring(keywords);
    return result;
}

/* stream_process(Instance, Pcm, Language | nil, Keywords | nil, Capacity) -> {ok, Json} | {error, R} */
static ERL_NIF_TERM nif_stream_process(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    const float *pcm;
    int samples, capacity;
    char *language = NULL, *keywords = NULL;
    if (!get_instance(env, argv[0], &in)) return enif_make_badarg(env);
    if (!get_pcm(env, argv[1], 0, &pcm, &samples) || !get_capacity(env, argv[4], &capacity) ||
        !get_cstring(env, argv[2], 1, &language) || !get_cstring(env, argv[3], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }
    char *out = enif_alloc(capacity);
    if (!out) { free_cstring(language); free_cstring(keywords); return error_str(env, "out of memory"); }
    out[0] = '\0';

    call_scope scope;
    enter(in, &scope);
    int rc = in->stream_process(pcm, samples, language, keywords, out, capacity);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, in, "needle_stream_transcribe_process", rc)
                                 : ok(env, make_binary(env, out, strnlen(out, capacity)));
    leave(in, &scope);
    enif_free(out);
    free_cstring(language); free_cstring(keywords);
    return result;
}

/* stream_stop(Instance, Capacity) -> {ok, Json} | {error, R} */
static ERL_NIF_TERM nif_stream_stop(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    int capacity;
    if (!get_instance(env, argv[0], &in) || !get_capacity(env, argv[1], &capacity))
        return enif_make_badarg(env);
    char *out = enif_alloc(capacity);
    if (!out) return error_str(env, "out of memory");
    out[0] = '\0';

    call_scope scope;
    enter(in, &scope);
    int rc = in->stream_stop(out, capacity);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, in, "needle_stream_transcribe_stop", rc)
                                 : ok(env, make_binary(env, out, strnlen(out, capacity)));
    leave(in, &scope);
    enif_free(out);
    return result;
}

/* ------------------------------------------------------------------------ */
/* embed(Instance, Text | nil, Pcm | nil) -> {ok, Float32NativeBinary} | {error, R} */

static ERL_NIF_TERM nif_embed(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    instance_t *in;
    char *text = NULL;
    const float *pcm;
    int samples;
    if (!get_instance(env, argv[0], &in) || !get_pcm(env, argv[2], 1, &pcm, &samples) ||
        !get_cstring(env, argv[1], 1, &text))
        return enif_make_badarg(env);
    if ((text == NULL) == (pcm == NULL)) {
        free_cstring(text);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM result;
    call_scope scope;
    enter(in, &scope);
    int count = in->embed(text, pcm, samples, NULL, 0);
    if (count <= 0) {
        result = engine_error(env, in, "needle_embed", count);
    } else {
        ERL_NIF_TERM bin;
        float *floats = (float *)enif_make_new_binary(env, (size_t)count * sizeof(float), &bin);
        int rc = in->embed(text, pcm, samples, floats, count);
        result = rc != count ? engine_error(env, in, "needle_embed", rc) : ok(env, bin);
    }
    leave(in, &scope);
    free_cstring(text);
    return result;
}

/* ------------------------------------------------------------------------ */

static int on_load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    instance_type = enif_open_resource_type(env, NULL, "cholla_instance", instance_dtor,
                                            ERL_NIF_RT_CREATE, NULL);
    global_lock = enif_mutex_create("cholla_global");
    if (!instance_type || !global_lock) return 1;
    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_nil = enif_make_atom(env, "nil");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_buffer_too_small = enif_make_atom(env, "buffer_too_small");
    atom_path = enif_make_atom(env, "path");
    atom_threads = enif_make_atom(env, "threads");
    atom_cpus = enif_make_atom(env, "cpus");
    atom_thread_control = enif_make_atom(env, "thread_control");
    atom_vm_threads = enif_make_atom(env, "vm_threads");
    atom_instances = enif_make_atom(env, "instances");
    return 0;
}

/* Every function that takes an instance mutex, which a running inference can
   hold for seconds, runs on a dirty scheduler. */
static ErlNifFunc funcs[] = {
    {"open_engine", 3, nif_open_engine, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"info", 1, nif_info, 0},
    {"load", 2, nif_load, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"models", 1, nif_models, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"init", 4, nif_engine_init, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"complete", 5, nif_complete, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"set_audio", 4, nif_set_audio, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"reset", 1, nif_reset, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"transcribe", 6, nif_transcribe, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"stream_process", 5, nif_stream_process, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"stream_stop", 2, nif_stream_stop, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"embed", 3, nif_embed, ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.Cholla.NIF, funcs, on_load, NULL, NULL, NULL)
