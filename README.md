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

## Conversation state and concurrency

The engine keeps one model and one conversation for the whole VM, and it is
not thread-safe. `Cholla.Engine` serialises every call, so any number of
processes can use Cholla at once. Keep these rules in mind:

- **Turns accumulate.** Calls on the same agent continue its conversation
  until `Cholla.reset/1`. Build the agent with `stateless: true` for
  independent queries.
- **Switching agents restarts the conversation.** A call on another agent
  re-initialises the engine with that agent's prompt and tools.
- **Tool loops are atomic.** `Cholla.run/3` reserves the engine for the whole
  loop. Tool functions may call Cholla themselves.
- **You can hold the engine yourself.** Wrap related calls in
  `Cholla.Engine.transaction/1`.

Inference runs on dirty CPU schedulers. The engine starts its own worker
threads and pins them to cores.

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
need full control and can manage the global state yourself.

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
