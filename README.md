# Cholla

Elixir bindings to [Needle](https://cactuscompute.com/needle), Cactus Compute's
small on-device model for tool calling, structured extraction and embeddings.
They also cover Whistle, the companion speech model, for transcription and
spoken queries.

Cholla is a NIF over the engine's C API (`needle.h`). The engine library and
weights are downloaded from Hugging Face on first use and cached, just as the
`cactus-needle` Python package does.

## Installation

```elixir
def deps do
  [{:cholla, path: "../cholla"}]
end
```

Building needs a C compiler and `make`. Supported platforms are Linux (glibc or
musl) and macOS, on x86_64 or arm64. To skip the download on first query:

```sh
mix cholla.download            # engine + needle3.cact (~37 MB)
mix cholla.download --speech   # also whistle.cact (~17 MB)
```

## Tool calling

```elixir
control = Cholla.Tool.new("control_device",
  description: "Switch or toggle any named smart-home device.",
  params: [device: :string, action: {:enum, ["on", "off", "toggle"]}],
  triggers: [~S"\b(turn|switch|power|flip)\b.*\b(on|off)\b", ~S"\btoggle\b"],
  function: fn %{"device" => device, "action" => action} ->
    MyHome.set(device, action)
  end
)

agent = Cholla.agent(tools: [control])

# One turn: the model's decision, nothing executed.
{:ok, response} = Cholla.complete(agent, "toggle the garage door")
response["function_calls"]
#=> [%{"name" => "control_device",
#      "arguments" => %{"action" => "toggle", "device" => "garage door"}}]

# Full loop: run the tools, feed results back, return the final turn.
{:ok, response} = Cholla.run(agent, "toggle the garage door")
response["results"]
```

The loop stops when the model stops calling tools, repeats the previous
round's calls exactly, or reaches `:max_steps`. Pass `min_confidence: 0.7`
to hold a low-confidence turn: nothing runs, and the response carries
`"held" => true` with the planned calls.

Every response is the engine's JSON envelope as a map. It carries `"type"`,
`"function_calls"`, `"suppressed_calls"`, `"confidence"`, `"reasoning"`,
`"validation"` and timing fields. The engine withholds calls below a confidence
floor of 0.1 in `"suppressed_calls"`. You can apply your own threshold too:

```elixir
case response do
  %{"confidence" => c, "function_calls" => calls} when c >= 0.7 -> execute(calls)
  %{"function_calls" => calls} -> ask_to_confirm(calls)
end
```

Tool parameters take a keyword shorthand or a raw JSON Schema map. See
`Cholla.Tool` for the shorthand.

## Extraction and embeddings

```elixir
invoice = Cholla.Tool.new("Invoice",
  params: [vendor: :string, total: :number, due: :date])

Cholla.extract("Invoice from Acme Corp for $1,250.00, due March 3 2027.", invoice)
#=> {:ok, %{"vendor" => "Acme Corp", "total" => 1250.0, "due" => "2027-03-03"}}

{:ok, vector} = Cholla.embed("hello world")   # 3072 floats
```

By default, `extract/3` returns `{:error, {:ungrounded, paths}}` when the engine
flags a value as missing from the text. Pass `strict: false` to accept it anyway.

## Speech

```elixir
{:ok, pcm} = Cholla.Speech.read_wav("clip.wav")   # 16 kHz WAV
{:ok, %{"text" => text}} = Cholla.Speech.transcribe(pcm, word_timestamps: true)

# Live audio, about one second per chunk.
{:ok, %{"text" => text, "words" => words}} = Cholla.Speech.stream(chunks)

# Spoken query to an agent.
{:ok, response} = Cholla.complete(agent, {:pcm, pcm})
```

## Engine pools

The engine library keeps one model, one conversation and one live stream as
process-global state, is not thread-safe, and runs one call at a time. Cholla
gets around that by loading a private copy of the library per **instance**,
each with its own state and its own `Cholla.Engine` process. A `Cholla.Pool`
runs several instances and routes calls between them.

By default the application starts one pool, `:default`, with one instance
using every core: lowest latency, one query at a time. For throughput,
configure more instances with fewer threads each:

```elixir
config :cholla,
  pools: [
    default: [size: 3, threads: 2, preload: [:text]],
    speech: [size: 1, threads: 2, preload: [:speech]]
  ],
  speech_pool: :speech
```

`Cholla` calls use the default pool and `Cholla.Speech` calls use the speech
pool. Override either per agent or per call with `pool: name`, or start pools
in your own supervision tree with `{Cholla.Pool, name: :mine, size: 2}`.

Routing:

- **Stateless work** (stateless agents, `extract/3`, embeddings,
  transcription) goes to the instance with the least work queued.
- **A stateful agent** always routes to the same instance, so its
  conversation continues there. A different agent landing on that instance
  re-initialises it and restarts the conversation, as with one instance.
- **A live speech stream** routes by calling process, so keep a stream within
  one process.
- **`Cholla.run/3` reserves one instance for the whole tool loop.** Tool
  functions that call Cholla on the same pool run on that instance.
- **Hold an instance yourself** with `Cholla.Pool.transaction/3`.

### Threads

Each instance runs worker threads that **spin** while they wait. When the
instances together run more threads than there are cores, they slow each
other down by one or two orders of magnitude: two 8-thread instances on 8
cores took 61 s for work that one instance did in 1.4 s. Keep the sum of
`size * threads` over all pools within the core count. Cholla logs a warning
when it is exceeded. A sized pool without `:threads` divides the cores by
its size.

Measured on a 4-core/8-thread i7-6700K with stateless tool-call queries:

| Pool            | Throughput | Single query |
| --------------- | ---------- | ------------ |
| 1 × 8 threads   | 4.7/s      | 183 ms       |
| 2 × 4 threads   | 6.5/s      | 170 ms       |
| 4 × 2 threads   | 9.5/s      | 217 ms       |
| 8 × 1 thread    | 10.4/s     | 335 ms       |

Each instance loads its own weights (about 35 MB text, 17 MB speech). An
instance's library and threads stay resident until the VM exits, because the
engine's threads cannot be stopped.

How it works: each instance dlopens its own copy of the library, as an
anonymous memfd on Linux and an unlinked temp file elsewhere, so the copies
share no globals. To bound the threads, the NIF redirects the copy's
`sysconf` and `sched_setaffinity` imports. The engine then sizes its pool to
`threads`, and its workers keep the CPU set of the thread that started them
(`cpus:` per instance, or unrestricted) instead of pinning themselves to
cores 1..N-1. Thread and CPU limits apply on Linux only; elsewhere keep pools
at size 1.

Inference runs on dirty CPU schedulers.

## Configuration

```elixir
config :cholla,
  engine_path: "/opt/needle/libneedle.so",     # skip the engine download
  weights_path: "/opt/needle/needle3.cact",    # base text model
  speech_weights_path: "/opt/needle/whistle.cact",
  cache_dir: "/var/cache/cholla",
  engine_version: "3.3.3"
```

The environment variables `CHOLLA_ENGINE_PATH` (or `NEEDLE3_LIB_PATH`),
`CHOLLA_WEIGHTS_PATH` and `CHOLLA_SPEECH_WEIGHTS_PATH` take precedence. For a
fine-tuned `.cact`, pass `weights: path` to `Cholla.agent/1`.

## Raw API

`Cholla.NIF` maps one-to-one onto `needle.h`. That covers `needle_load`,
`needle_init`, `needle_complete`, `needle_embed`, `needle_reset`,
`needle_transcribe`, the streaming transcriber and the others. Use it when you
need full control: `Cholla.NIF.open_engine/3` opens an instance, and every
other function takes one.

## Tests

```sh
mix test                                   # offline unit tests
mix test --include engine --include speech # runs the real model
```

## Not ported from the Python package

- **Client-side year grounding.** The Python package checks dates against the
  years written in the query. Cholla relies on the engine's own `"validation"`
  flags, plus the same exemption for numbers written in the query.
- **Confidence-head detection.** Python reads a fine-tuned archive to see if it
  carries a confidence head. Cholla passes `"confidence"` through unchanged.
- **Engine builds and fine-tuning.** `needle build` and LoRA tuning stay with
  the Python tooling.
- **Windows.** The NIF loads the engine with `dlopen`.

The vendored `c_src/needle.h` is © Cactus Compute under the Apache License 2.0.
