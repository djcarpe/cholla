/*
 * cholla_nif.c - Erlang NIF bindings to the Cactus Compute Needle engine.
 *
 * The engine (libneedle) is a prebuilt, self-contained shared library that
 * keeps one process-global, non-thread-safe model per kind (text, speech).
 * This NIF dlopens it at runtime, resolves the C API declared in needle.h and
 * serialises every call behind one mutex. Anything that runs the model is
 * scheduled on a dirty CPU scheduler.
 *
 * Conventions on the Erlang side:
 *   - strings are binaries or iodata; optional strings may be the atom nil
 *   - PCM audio is a binary of native-endian 32-bit floats, 16 kHz mono
 *   - results are {ok, Value} or {error, ReasonBinary}
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

static struct {
    void *handle;
    char *path;
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
} engine;

static ErlNifMutex *lock;

static ERL_NIF_TERM atom_ok, atom_error, atom_nil, atom_true, atom_false,
    atom_already_open, atom_buffer_too_small;

/* Engine outputs larger than this are treated as an engine fault. */
#define MAX_OUTPUT (64 * 1024 * 1024)

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
static int needed_bytes(void) {
    const char *detail = engine.last_error ? engine.last_error() : NULL;
    if (!detail || !strstr(detail, "too small")) return 0;
    const char *colon = strrchr(detail, ':');
    long n = colon ? strtol(colon + 1, NULL, 10) : 0;
    return (n > 0 && n < MAX_OUTPUT) ? (int)n + 1 : 0;
}

/* Error from the engine's own last_error, or a generic message with the code.
   A too-small output buffer becomes {error, {buffer_too_small, Needed}}. */
static ERL_NIF_TERM engine_error(ErlNifEnv *env, const char *what, int code) {
    int needed = needed_bytes();
    if (needed)
        return enif_make_tuple2(env, atom_error,
                                enif_make_tuple2(env, atom_buffer_too_small, enif_make_int(env, needed)));
    const char *detail = engine.last_error ? engine.last_error() : NULL;
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

static ERL_NIF_TERM not_open(ErlNifEnv *env) {
    return error_str(env, "needle engine is not open; call Cholla.Engine.start or Cholla.NIF.open_engine first");
}

/* ------------------------------------------------------------------------ */
/* open_engine(Path) -> ok | {error, {already_open, Path}} | {error, Reason} */

static ERL_NIF_TERM nif_open_engine(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *path;
    if (!get_cstring(env, argv[0], 0, &path)) return enif_make_badarg(env);

    ERL_NIF_TERM result = atom_ok;
    enif_mutex_lock(lock);
    if (engine.handle) {
        /* The engine holds global state, so one library per VM. */
        if (strcmp(engine.path, path) != 0)
            result = enif_make_tuple2(env, atom_error,
                                      enif_make_tuple2(env, atom_already_open,
                                                       make_binary(env, engine.path, strlen(engine.path))));
        enif_mutex_unlock(lock);
        free_cstring(path);
        return result;
    }

    void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char *detail = dlerror();
        result = error_str(env, detail ? detail : "dlopen failed");
        enif_mutex_unlock(lock);
        free_cstring(path);
        return result;
    }

    const char *missing = NULL;
#define RESOLVE(field, name)                                       \
    do {                                                           \
        *(void **)(&engine.field) = dlsym(handle, name);           \
        if (!engine.field && !missing) missing = name;             \
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

    if (missing) {
        char message[256];
        snprintf(message, sizeof message, "%s is not a Needle 3 engine: missing symbol %s", path, missing);
        dlclose(handle);
        memset(&engine, 0, sizeof engine);
        result = error_str(env, message);
        free_cstring(path);
    } else {
        engine.handle = handle;
        engine.path = path; /* kept for the life of the VM */
    }
    enif_mutex_unlock(lock);
    return result;
}

/* engine_path() -> {ok, Path} | nil */
static ERL_NIF_TERM nif_engine_path(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    enif_mutex_lock(lock);
    ERL_NIF_TERM result = engine.handle ? ok(env, make_binary(env, engine.path, strlen(engine.path))) : atom_nil;
    enif_mutex_unlock(lock);
    return result;
}

/* ------------------------------------------------------------------------ */
/* load(Cact) -> {ok, ModelsMask} | {error, Reason}                          */

static ERL_NIF_TERM nif_load(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ErlNifBinary cact;
    if (!enif_inspect_binary(env, argv[0], &cact)) return enif_make_badarg(env);

    enif_mutex_lock(lock);
    if (!engine.handle) { enif_mutex_unlock(lock); return not_open(env); }
    /* The engine copies the archive, so the binary need not outlive the call. */
    int rc = engine.load(cact.data, (unsigned long long)cact.size);
    ERL_NIF_TERM result = rc < 0 ? engine_error(env, "needle_load", rc) : ok(env, enif_make_int(env, engine.models()));
    enif_mutex_unlock(lock);
    return result;
}

/* models() -> ModelsMask (NEEDLE_TEXT | NEEDLE_SPEECH) */
static ERL_NIF_TERM nif_models(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    enif_mutex_lock(lock);
    int mask = engine.handle ? engine.models() : 0;
    enif_mutex_unlock(lock);
    return enif_make_int(env, mask);
}

/* ------------------------------------------------------------------------ */
/* init(System, ToolsJson, ToolIndexPath | nil) -> {ok, PrefixTokens} | {error, R} */

static ERL_NIF_TERM nif_engine_init(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *system = NULL, *tools = NULL, *index = NULL;
    if (!get_cstring(env, argv[0], 0, &system) || !get_cstring(env, argv[1], 0, &tools) ||
        !get_cstring(env, argv[2], 1, &index)) {
        free_cstring(system); free_cstring(tools); free_cstring(index);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM result;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int rc = engine.init(system, tools, index);
        result = rc < 0 ? engine_error(env, "needle_init", rc) : ok(env, enif_make_int(env, rc));
    }
    enif_mutex_unlock(lock);
    free_cstring(system); free_cstring(tools); free_cstring(index);
    return result;
}

/* ------------------------------------------------------------------------ */
/* complete(Text | nil, Pcm | nil, MaxNewTokens, Capacity) -> {ok, Json} | {error, R}
   Not retried on a small buffer: the engine may already have advanced the
   conversation, so the caller decides. */

static ERL_NIF_TERM nif_complete(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *text = NULL;
    const float *pcm;
    int samples, max_new_tokens, capacity;
    if (!get_pcm(env, argv[1], 1, &pcm, &samples) || !enif_get_int(env, argv[2], &max_new_tokens) ||
        max_new_tokens <= 0 || !get_capacity(env, argv[3], &capacity) || !get_cstring(env, argv[0], 1, &text))
        return enif_make_badarg(env);
    if ((text == NULL) == (pcm == NULL)) {
        free_cstring(text);
        return enif_make_badarg(env); /* exactly one of text and pcm */
    }

    char *out = enif_alloc(capacity);
    if (!out) { free_cstring(text); return error_str(env, "out of memory"); }
    out[0] = '\0';

    ERL_NIF_TERM result;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int rc = engine.complete(text, pcm, samples, max_new_tokens, out, capacity);
        result = rc < 0 ? engine_error(env, "needle_complete", rc) : ok(env, make_binary(env, out, strnlen(out, capacity)));
    }
    enif_mutex_unlock(lock);
    enif_free(out);
    free_cstring(text);
    return result;
}

/* set_audio(Language | nil, Keywords | nil, WordTimestamps) -> ok */
static ERL_NIF_TERM nif_set_audio(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *language = NULL, *keywords = NULL;
    int word_timestamps;
    if (!get_bool(env, argv[2], &word_timestamps) || !get_cstring(env, argv[0], 1, &language) ||
        !get_cstring(env, argv[1], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }
    ERL_NIF_TERM result = atom_ok;
    enif_mutex_lock(lock);
    if (!engine.handle) result = not_open(env);
    else engine.set_audio(language, keywords, word_timestamps); /* strings are copied */
    enif_mutex_unlock(lock);
    free_cstring(language); free_cstring(keywords);
    return result;
}

/* reset() -> ok */
static ERL_NIF_TERM nif_reset(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    ERL_NIF_TERM result = atom_ok;
    enif_mutex_lock(lock);
    if (!engine.handle) result = not_open(env);
    else engine.reset();
    enif_mutex_unlock(lock);
    return result;
}

/* ------------------------------------------------------------------------ */
/* transcribe(Pcm, Language | nil, Keywords | nil, WordTimestamps, Capacity)
     -> {ok, Json} | {error, R}
   Stateless, so a too-small buffer is retried once at the size the engine asks for. */

static ERL_NIF_TERM nif_transcribe(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    const float *pcm;
    int samples, word_timestamps, capacity;
    char *language = NULL, *keywords = NULL;
    if (!get_pcm(env, argv[0], 0, &pcm, &samples) || !get_bool(env, argv[3], &word_timestamps) ||
        !get_capacity(env, argv[4], &capacity) || !get_cstring(env, argv[1], 1, &language) ||
        !get_cstring(env, argv[2], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM result;
    char *out = NULL;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int rc = -1;
        for (int attempt = 0; attempt < 2; attempt++) {
            enif_free(out);
            out = enif_alloc(capacity);
            if (!out) break;
            out[0] = '\0';
            rc = engine.transcribe(pcm, samples, language, keywords, word_timestamps, out, capacity);
            if (rc >= 0) break;
            int needed = needed_bytes();
            if (!needed || needed <= capacity) break;
            capacity = needed;
        }
        if (!out) result = error_str(env, "out of memory");
        else if (rc < 0) result = engine_error(env, "needle_transcribe", rc);
        else result = ok(env, make_binary(env, out, strnlen(out, capacity)));
    }
    enif_mutex_unlock(lock);
    enif_free(out);
    free_cstring(language); free_cstring(keywords);
    return result;
}

/* stream_process(Pcm, Language | nil, Keywords | nil, Capacity) -> {ok, Json} | {error, R} */
static ERL_NIF_TERM nif_stream_process(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    const float *pcm;
    int samples, capacity;
    char *language = NULL, *keywords = NULL;
    if (!get_pcm(env, argv[0], 0, &pcm, &samples) || !get_capacity(env, argv[3], &capacity) ||
        !get_cstring(env, argv[1], 1, &language) || !get_cstring(env, argv[2], 1, &keywords)) {
        free_cstring(language); free_cstring(keywords);
        return enif_make_badarg(env);
    }
    char *out = enif_alloc(capacity);
    if (!out) { free_cstring(language); free_cstring(keywords); return error_str(env, "out of memory"); }
    out[0] = '\0';

    ERL_NIF_TERM result;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int rc = engine.stream_process(pcm, samples, language, keywords, out, capacity);
        result = rc < 0 ? engine_error(env, "needle_stream_transcribe_process", rc)
                        : ok(env, make_binary(env, out, strnlen(out, capacity)));
    }
    enif_mutex_unlock(lock);
    enif_free(out);
    free_cstring(language); free_cstring(keywords);
    return result;
}

/* stream_stop(Capacity) -> {ok, Json} | {error, R} */
static ERL_NIF_TERM nif_stream_stop(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    int capacity;
    if (!get_capacity(env, argv[0], &capacity)) return enif_make_badarg(env);
    char *out = enif_alloc(capacity);
    if (!out) return error_str(env, "out of memory");
    out[0] = '\0';

    ERL_NIF_TERM result;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int rc = engine.stream_stop(out, capacity);
        result = rc < 0 ? engine_error(env, "needle_stream_transcribe_stop", rc)
                        : ok(env, make_binary(env, out, strnlen(out, capacity)));
    }
    enif_mutex_unlock(lock);
    enif_free(out);
    return result;
}

/* ------------------------------------------------------------------------ */
/* embed(Text | nil, Pcm | nil) -> {ok, Float32NativeBinary} | {error, R}   */

static ERL_NIF_TERM nif_embed(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
    char *text = NULL;
    const float *pcm;
    int samples;
    if (!get_pcm(env, argv[1], 1, &pcm, &samples) || !get_cstring(env, argv[0], 1, &text))
        return enif_make_badarg(env);
    if ((text == NULL) == (pcm == NULL)) {
        free_cstring(text);
        return enif_make_badarg(env);
    }

    ERL_NIF_TERM result;
    enif_mutex_lock(lock);
    if (!engine.handle) {
        result = not_open(env);
    } else {
        int count = engine.embed(text, pcm, samples, NULL, 0);
        if (count <= 0) {
            result = engine_error(env, "needle_embed", count);
        } else {
            ERL_NIF_TERM bin;
            float *floats = (float *)enif_make_new_binary(env, (size_t)count * sizeof(float), &bin);
            int rc = engine.embed(text, pcm, samples, floats, count);
            result = rc != count ? engine_error(env, "needle_embed", rc) : ok(env, bin);
        }
    }
    enif_mutex_unlock(lock);
    free_cstring(text);
    return result;
}

/* ------------------------------------------------------------------------ */

static int on_load(ErlNifEnv *env, void **priv, ERL_NIF_TERM info) {
    lock = enif_mutex_create("cholla_needle");
    if (!lock) return 1;
    atom_ok = enif_make_atom(env, "ok");
    atom_error = enif_make_atom(env, "error");
    atom_nil = enif_make_atom(env, "nil");
    atom_true = enif_make_atom(env, "true");
    atom_false = enif_make_atom(env, "false");
    atom_already_open = enif_make_atom(env, "already_open");
    atom_buffer_too_small = enif_make_atom(env, "buffer_too_small");
    return 0;
}

static int on_upgrade(ErlNifEnv *env, void **priv, void **old_priv, ERL_NIF_TERM info) {
    return 0; /* statics carry over within the same .so */
}

/* Every function takes the engine mutex, which a running inference can hold
   for seconds, so all of them run on dirty schedulers. */
static ErlNifFunc funcs[] = {
    {"open_engine", 1, nif_open_engine, ERL_NIF_DIRTY_JOB_IO_BOUND},
    {"engine_path", 0, nif_engine_path, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"load", 1, nif_load, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"models", 0, nif_models, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"init", 3, nif_engine_init, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"complete", 4, nif_complete, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"set_audio", 3, nif_set_audio, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"reset", 0, nif_reset, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"transcribe", 5, nif_transcribe, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"stream_process", 4, nif_stream_process, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"stream_stop", 1, nif_stream_stop, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"embed", 2, nif_embed, ERL_NIF_DIRTY_JOB_CPU_BOUND},
};

ERL_NIF_INIT(Elixir.Cholla.NIF, funcs, on_load, NULL, on_upgrade, NULL)
