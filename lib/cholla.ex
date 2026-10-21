defmodule Cholla do
  @moduledoc """
  Elixir bindings to [Needle](https://cactuscompute.com/needle), Cactus
  Compute's on-device model for tool calling, structured extraction, text
  embeddings and, with the Whistle model, speech.

      control = Cholla.Tool.new("control_device",
        description: "Switch or toggle any named smart-home device.",
        params: [device: :string, action: {:enum, ["on", "off", "toggle"]}],
        function: fn args -> %{ok: true, args: args} end
      )

      agent = Cholla.agent(tools: [control])

      {:ok, response} = Cholla.complete(agent, "toggle the garage door")
      response["function_calls"]
      #=> [%{"name" => "control_device",
      #      "arguments" => %{"device" => "garage door", "action" => "toggle"}}]

  The engine library and weights are downloaded from Hugging Face on first
  use and cached; see `Cholla.Download`.

  ## Responses

  Each turn returns the engine's JSON envelope decoded into a map with string
  keys: `"type"`, `"success"`, `"error"`, `"error_code"`, `"function_calls"`,
  `"suppressed_calls"` (calls withheld below the engine's confidence floor),
  `"reasoning"`, `"confidence"`, `"prefill_tps"`, `"decode_tps"`,
  `"peak_ram_mb"` and `"validation"` (`"ungrounded"` argument paths and
  `"negation"`).

  ## Conversation state

  Each engine instance holds a single conversation. Calls on the same agent
  continue it until `reset/1`; a call on another agent that lands on the
  same instance starts a new one. Build an agent with `stateless: true` to
  reset before every query.

  ## Pools

  Calls run on a `Cholla.Pool` of engine instances: the default pool, or the
  one named by the agent's or the call's `:pool` option. Stateless agents,
  extraction and embeddings go to the least busy instance; a stateful agent
  always uses the same one. See `Cholla.Pool` for sizing.
  """

  alias Cholla.{Agent, Engine, NIF, Pool, Tool}

  @type input :: String.t() | {:pcm, binary()}
  @type response :: %{String.t() => term()}

  @doc "Builds an agent. See `Cholla.Agent.new/1` for options."
  @spec agent(keyword()) :: Agent.t()
  defdelegate agent(opts \\ []), to: Agent, as: :new

  @doc """
  Runs one turn and returns the model's response, without executing tools.

  `input` is text or `{:pcm, binary}` with 16 kHz mono native-endian f32
  audio; speech input needs the Whistle model, which is fetched on first use.

  ## Options

    * `:max_new_tokens` - defaults to the agent's setting
    * `:language`, `:keywords`, `:word_timestamps` - transcription settings
      for speech input (see `Cholla.Speech.transcribe/2`)
  """
  @spec complete(Agent.t(), input(), keyword()) :: {:ok, response()} | {:error, term()}
  def complete(%Agent{} = agent, input, opts \\ []) do
    Pool.op(pool(agent, opts), affinity(agent), fn state ->
      case prepare(state, agent, input, opts) do
        {:ok, state} ->
          state = if agent.stateless, do: state, else: Engine.count_turn(state)
          {turn(state.instance, agent, input, opts), state}

        error ->
          Engine.reply(error)
      end
    end)
  end

  defp prepare(state, agent, input, opts) do
    with {:ok, state} <- Engine.bind(state, agent),
         {:ok, state} <- prepare_audio(state, input, opts) do
      if agent.stateless do
        :ok = NIF.reset(state.instance)
        {:ok, Engine.reset_turns(state)}
      else
        {:ok, state}
      end
    end
  end

  defp prepare_audio(state, text, _opts) when is_binary(text), do: {:ok, state}

  defp prepare_audio(state, {:pcm, pcm}, opts) when is_binary(pcm) do
    with {:ok, state} <- Engine.ensure_speech(state) do
      keywords = Cholla.Speech.keywords(opts[:keywords])
      word_timestamps = Keyword.get(opts, :word_timestamps, false)
      :ok = NIF.set_audio(state.instance, opts[:language], keywords, word_timestamps)
      {:ok, state}
    end
  end

  defp pool(agent, opts), do: opts[:pool] || agent.pool || Pool.default()

  # A stateful agent's conversation lives in one instance, so it always routes there.
  defp affinity(%Agent{stateless: true}), do: nil
  defp affinity(%Agent{id: id}), do: id

  defp turn(instance, agent, input, opts) do
    max = Keyword.get(opts, :max_new_tokens, agent.max_new_tokens)

    {text, pcm} =
      case input do
        {:pcm, pcm} -> {nil, pcm}
        text -> {text, nil}
      end

    with {:ok, json} <- NIF.complete(instance, text, pcm, max, agent.buffer_size) do
      case JSON.decode(json) do
        {:ok, response} -> {:ok, response}
        {:error, reason} -> {:error, {:bad_envelope, reason, json}}
      end
    end
  end

  @doc """
  Runs the full tool loop: the model picks calls, each call's tool
  `:function` runs with the arguments map, the results go back to the model,
  and this repeats until it stops calling tools, repeats the previous
  round's calls exactly, or reaches `:max_steps`.

  Returns the final response with `"results"` holding every tool result in
  order. A tool that raises yields `%{"error" => message}` for the model to
  see. One engine instance is reserved for the calling process for the whole
  loop, and tool functions that call `Cholla` on the same pool run on it.

  ## Options

    * `:pool` - the pool to run on, overriding the agent's
    * `:max_steps` - maximum tool rounds (default 8)
    * `:max_new_tokens` - per turn
    * `:strict` - refuse to run a call whose arguments the engine flags as
      ungrounded in the query, unless the value is a number written in the
      query or system prompt (default `true`)
    * `:min_confidence` - hold a turn whose `"confidence"` is below this value
      (or missing) instead of running its calls. The response comes back with
      `"held" => true` and the unexecuted `"function_calls"`, so the caller can
      ask for confirmation. Off by default
  """
  @spec run(Agent.t(), String.t(), keyword()) :: {:ok, response()} | {:error, term()}
  def run(%Agent{} = agent, query, opts \\ []) when is_binary(query) do
    max_steps = Keyword.get(opts, :max_steps, 8)
    strict = Keyword.get(opts, :strict, true)

    Pool.transaction(pool(agent, opts), affinity(agent), fn ->
      with {:ok, response} <- complete(agent, query, opts) do
        loop(agent, response, query, strict, max_steps, opts, [], nil)
      end
    end)
  end

  # `previous` holds the last round's calls: a model that repeats them exactly
  # (typically retrying a call that failed) is making no progress, so stop.
  defp loop(agent, response, query, strict, steps, opts, executed, previous) do
    calls = response["function_calls"] || []

    cond do
      steps == 0 or response["type"] != "call" or calls == [] or calls == previous ->
        {:ok, Map.put(response, "results", executed)}

      below_confidence?(response, opts[:min_confidence]) ->
        {:ok, response |> Map.put("results", executed) |> Map.put("held", true)}

      true ->
        run_calls(agent, response, calls, query, strict, steps, opts, executed)
    end
  end

  defp below_confidence?(_response, nil), do: false

  defp below_confidence?(response, min) do
    case response["confidence"] do
      confidence when is_number(confidence) -> confidence < min
      _ -> true
    end
  end

  defp run_calls(agent, response, calls, query, strict, steps, opts, executed) do
    ungrounded = ungrounded_paths(response)
    results = Enum.map(calls, &execute(agent, &1, ungrounded, query, strict))

    with {:ok, next} <- continue(agent, JSON.encode!(Enum.map(results, &jsonable/1)), opts) do
      loop(agent, next, query, strict, steps - 1, opts, executed ++ results, calls)
    end
  end

  # A follow-up turn that feeds tool results back; not counted as a new query.
  defp continue(agent, text, opts) do
    Pool.op(pool(agent, opts), affinity(agent), fn state ->
      case Engine.bind(state, agent) do
        {:ok, state} -> {turn(state.instance, agent, text, opts), state}
        error -> Engine.reply(error)
      end
    end)
  end

  defp execute(agent, call, ungrounded, query, strict) do
    name = to_string(call["name"])
    arguments = call["arguments"] || %{}

    fabricated =
      if strict do
        grounded = grounded_number_paths(arguments, [query, agent.system])
        ungrounded |> Map.get(name, []) |> Enum.reject(&(&1 in grounded)) |> Enum.sort()
      else
        []
      end

    case {fabricated, Map.fetch(agent.functions, name)} do
      {[_ | _], _} ->
        %{"error" => "ungrounded " <> Enum.join(fabricated, ", ")}

      {[], :error} ->
        %{"error" => "unknown tool: " <> name}

      {[], {:ok, fun}} ->
        try do
          fun.(arguments)
        rescue
          exception -> %{"error" => Exception.message(exception)}
        catch
          kind, value -> %{"error" => "#{kind}: #{inspect(value)}"}
        end
    end
  end

  @doc """
  One-shot structured extraction: fills `schema` from `text`.

  `schema` is a `Cholla.Tool` or a raw tool schema map; its parameters
  describe the fields. Returns `{:ok, arguments}` with string keys, or
  `{:ok, nil}` when the model extracted nothing.

  ## Options

    * `:system`, `:weights`, `:max_new_tokens`, `:pool`
    * `:strict` - return `{:error, {:ungrounded, paths}}` when the engine
      flags values not grounded in the text, or a negated request
      (default `true`)
  """
  @spec extract(String.t(), Tool.t() | map(), keyword()) ::
          {:ok, map() | nil} | {:error, term()}
  def extract(text, schema, opts \\ []) when is_binary(text) do
    agent =
      Agent.new(
        tools: [schema],
        system: opts[:system] || "",
        weights: opts[:weights],
        pool: opts[:pool],
        stateless: true
      )

    with {:ok, response} <- complete(agent, text, Keyword.take(opts, [:max_new_tokens])) do
      case (response["function_calls"] || []) ++ (response["suppressed_calls"] || []) do
        [] ->
          {:ok, nil}

        [call | _] ->
          arguments = call["arguments"] || %{}

          if Keyword.get(opts, :strict, true),
            do: validate_extraction(arguments, response, text, agent.system),
            else: {:ok, arguments}
      end
    end
  end

  defp validate_extraction(arguments, response, text, system) do
    validation = response["validation"] || %{}
    grounded = grounded_number_paths(arguments, [text, system])

    failures =
      (validation["ungrounded"] || [])
      |> Enum.map(&(&1 |> to_string() |> String.split(".", parts: 2) |> List.last()))
      |> Enum.reject(&(&1 in grounded))

    failures = if validation["negation"], do: ["negated request" | failures], else: failures

    case Enum.uniq(failures) do
      [] -> {:ok, arguments}
      paths -> {:error, {:ungrounded, Enum.sort(paths)}}
    end
  end

  @doc """
  Embeds text with the model's embedding head.

  Returns a list of floats, or the raw native-endian f32 binary with
  `format: :binary`. Pass `weights:` to embed with a fine-tuned model and
  `pool:` to run on a pool other than the default.
  """
  @spec embed(String.t(), keyword()) :: {:ok, [float()] | binary()} | {:error, term()}
  def embed(text, opts \\ []) when is_binary(text) do
    Pool.op(opts[:pool] || Pool.default(), nil, fn state ->
      case Engine.ensure_text(state, opts[:weights] && Path.expand(opts[:weights])) do
        {:ok, state} ->
          {NIF.embed(state.instance, text, nil) |> format_floats(opts[:format]), state}

        error ->
          Engine.reply(error)
      end
    end)
  end

  @doc false
  def format_floats({:ok, bin}, :binary), do: {:ok, bin}
  def format_floats({:ok, bin}, _), do: {:ok, for(<<f::float-32-native <- bin>>, do: f)}
  def format_floats(error, _), do: error

  @doc "Clears the agent's conversation, if it is the one its instance holds."
  @spec reset(Agent.t()) :: :ok | {:error, term()}
  def reset(%Agent{id: id} = agent) do
    Pool.op(pool(agent, []), id, fn
      %Engine{active: ^id} = state -> {NIF.reset(state.instance), Engine.reset_turns(state)}
      state -> {:ok, state}
    end)
  end

  @doc """
  Opens every instance of a pool and loads the base model ahead of the first
  query. Pass `speech: true` to load the Whistle speech model too, and
  `pool:` for a pool other than the default.
  """
  @spec preload(keyword()) :: :ok | {:error, term()}
  def preload(opts \\ []) do
    models = if opts[:speech], do: [:text, :speech], else: [:text]
    Pool.preload(opts[:pool] || Pool.default(), models)
  end

  ## Grounding helpers

  defp ungrounded_paths(response) do
    ((response["validation"] || %{})["ungrounded"] || [])
    |> Enum.reduce(%{}, fn name, acc ->
      {tool, path} =
        case String.split(to_string(name), ".", parts: 2) do
          [tool, path] -> {tool, path}
          [tool] -> {tool, tool}
        end

      Map.update(acc, tool, [path], &[path | &1])
    end)
  end

  @date_fact ~r/date:\s*\d{4}-\d{2}-\d{2}(?:\s+[A-Za-z]{3})?(?:\s+\d{2}:\d{2})?|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?/
  @number ~r/(?<![\w.,])(?:[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?|[-+]?\d+(?:\.\d+)?)(?!\d)/

  @doc false
  # Argument paths whose numeric value is written literally in a source text.
  def grounded_number_paths(arguments, sources) do
    numbers =
      sources
      |> Enum.reject(&is_nil/1)
      |> Enum.map(&Regex.replace(@date_fact, &1, " "))
      |> Enum.flat_map(&Regex.scan(@number, &1))
      |> Enum.map(fn [n] -> n |> String.replace(",", "") |> parse_number() end)

    for {path, value} <- numeric_paths(arguments, ""),
        Enum.any?(numbers, &(&1 == value * 1.0)),
        do: path
  end

  defp parse_number(string) do
    {number, ""} = Float.parse(String.trim_leading(string, "+"))
    number
  end

  defp numeric_paths(value, _path) when is_boolean(value), do: []
  defp numeric_paths(value, path) when is_number(value), do: [{path, value}]

  defp numeric_paths(%{} = map, path) do
    Enum.flat_map(map, fn {k, v} ->
      numeric_paths(v, if(path == "", do: "#{k}", else: "#{path}.#{k}"))
    end)
  end

  defp numeric_paths(list, path) when is_list(list) do
    list |> Enum.with_index() |> Enum.flat_map(fn {v, i} -> numeric_paths(v, "#{path}[#{i}]") end)
  end

  defp numeric_paths(_, _), do: []

  # Tool results become JSON for the model; anything JSON cannot hold is inspected.
  defp jsonable(%{__struct__: _} = struct), do: struct |> Map.from_struct() |> jsonable()
  defp jsonable(%{} = map), do: Map.new(map, fn {k, v} -> {json_key(k), jsonable(v)} end)
  defp jsonable(list) when is_list(list), do: Enum.map(list, &jsonable/1)
  defp jsonable(tuple) when is_tuple(tuple), do: tuple |> Tuple.to_list() |> jsonable()
  defp jsonable(v) when is_binary(v) or is_number(v) or is_boolean(v) or is_nil(v), do: v
  defp jsonable(v) when is_atom(v), do: Atom.to_string(v)
  defp jsonable(v), do: inspect(v)

  defp json_key(k) when is_binary(k), do: k
  defp json_key(k) when is_atom(k), do: Atom.to_string(k)
  defp json_key(k), do: inspect(k)
end
