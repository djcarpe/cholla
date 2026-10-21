defmodule Cholla.Train.Data do
  @moduledoc """
  Training examples: reading JSONL, rendering the chat template the model
  was trained on, tokenising, and the engram hashes the model looks up.

  An example is a map with `"query"`, `"tools"` (the schemas offered),
  `"answers"` (the calls to make, `[]` for none) and optionally `"system"`
  and `"reasoning"`. Lines in the platform chat format (`"messages"` plus
  OpenAI-form `"tools"`) are converted when they hold one user turn answered
  by one assistant turn.
  """

  import Bitwise
  alias Cholla.Train.Tokenizer

  @im_start "<|im_start|>"
  @im_end "<|im_end|>"
  @think_start "<think>"
  @think_end "</think>"
  @tools_start "<tools>"
  @tools_end "</tools>"
  @tool_call_start "<tool_call>"
  @tool_call_end "</tool_call>"

  @engram_seed 0x9E3779B9
  @engram_prime 0x01000193

  @doc "Reads a JSONL file of examples, skipping lines in neither format."
  @spec read_jsonl(Path.t()) :: {[map()], non_neg_integer()}
  def read_jsonl(path) do
    path
    |> File.stream!()
    |> Enum.map(&String.trim/1)
    |> Enum.reject(&(&1 == ""))
    |> Enum.map(&JSON.decode!/1)
    |> Enum.map(&normalize/1)
    |> Enum.split_with(&(&1 != nil))
    |> then(fn {kept, skipped} -> {kept, length(skipped)} end)
  end

  @doc "An example as the trainer wants it, or nil when the line fits neither format."
  def normalize(%{"query" => _} = example), do: example

  def normalize(%{"messages" => messages} = example) when is_list(messages) do
    turns = Enum.reject(messages, &(&1["role"] == "system"))

    case turns do
      [%{"role" => "user"} = user, %{"role" => "assistant"} = assistant] ->
        answers =
          for call <- assistant["tool_calls"] || [] do
            function = call["function"] || call
            arguments = function["arguments"] || %{}

            arguments =
              if is_binary(arguments),
                do: if(String.trim(arguments) == "", do: %{}, else: JSON.decode!(arguments)),
                else: arguments

            %{"name" => function["name"], "arguments" => arguments}
          end

        tools =
          for t <- example["tools"] || [], do: if(is_map(t), do: t["function"] || t, else: t)

        out = %{"query" => user["content"] || "", "tools" => tools, "answers" => answers}

        case Enum.find(messages, &(&1["role"] == "system")) do
          %{"content" => system} when is_binary(system) and system != "" ->
            Map.put(out, "system", system)

          _ ->
            out
        end

      _ ->
        nil
    end
  end

  def normalize(_), do: nil

  @doc """
  Renders an example as `{prompt, target}` text, the chat template of the
  model. Objects are written with `name`, `description`, `parameters` and
  `arguments` first (the order the model was trained on), then the other
  keys sorted, since Elixir maps keep no order.
  """
  @spec render(map()) :: {String.t(), String.t()}
  def render(example) do
    tools = example["tools"] || []
    tools_json = if is_binary(tools), do: tools, else: encode_json(tools)
    answers = example["answers"] || example["function_calls"] || []
    answers_json = if is_binary(answers), do: answers, else: encode_json(answers)
    reasoning = String.trim(example["reasoning"] || "")
    system = String.trim(example["system"] || "")

    prefix = if system != "", do: @im_start <> "system\n" <> system <> @im_end <> "\n", else: ""

    prompt =
      prefix <>
        @im_start <>
        "user\n" <>
        @tools_start <>
        tools_json <>
        @tools_end <> "\n" <> example["query"] <> @im_end <> "\n" <> @im_start <> "assistant\n"

    think =
      if reasoning != "",
        do: @think_start <> "\n" <> reasoning <> "\n" <> @think_end <> "\n",
        else: ""

    target = think <> @tool_call_start <> answers_json <> @tool_call_end <> @im_end
    {prompt, target}
  end

  @doc "Compact JSON in the model's key order; see `Cholla.Tool.encode_json/1`."
  @spec encode_json(term()) :: String.t()
  defdelegate encode_json(term), to: Cholla.Tool

  @doc """
  Tokenises an example to `max_len`: `{ids, mask}` lists, where the mask is
  1.0 on the target tokens (and the closing eos) and 0.0 on the prompt and
  padding.
  """
  @spec encode(Tokenizer.t(), map(), pos_integer()) :: {[integer()], [float()]}
  def encode(tok, example, max_len) do
    {prompt, target} = render(example)
    prompt_ids = Tokenizer.encode(tok, prompt)
    target_ids = Tokenizer.encode(tok, target)
    ids = [Tokenizer.bos_id()] ++ prompt_ids ++ target_ids ++ [Tokenizer.eos_id()]

    mask =
      List.duplicate(0.0, 1 + length(prompt_ids)) ++ List.duplicate(1.0, length(target_ids) + 1)

    ids = Enum.take(ids, max_len)
    mask = Enum.take(mask, max_len)
    pad = max_len - length(ids)
    {ids ++ List.duplicate(Tokenizer.pad_id(), pad), mask ++ List.duplicate(0.0, pad)}
  end

  @doc "The number of tokens an example takes, with bos and eos."
  def token_length(tok, example) do
    {prompt, target} = render(example)
    length(Tokenizer.encode(tok, prompt)) + length(Tokenizer.encode(tok, target)) + 2
  end

  @doc """
  The sequence length to train at: the smallest power-of-two bucket from 128
  that holds the longest example, capped at `cap`.
  """
  def fit_max_len(tok, examples, cap) do
    longest = examples |> Enum.map(&token_length(tok, &1)) |> Enum.max(fn -> 0 end)
    bucket(128, min(longest, cap)) |> min(cap)
  end

  defp bucket(b, longest) when b >= longest, do: b
  defp bucket(b, longest), do: bucket(b * 2, longest)

  @doc """
  The engram table rows a token sequence looks up: for each position and
  each `(order, head)` table, a hash of the token and its `order - 1`
  predecessors, offset into the flattened `[tables * slots, sub_dim]` table.
  Returns a `[T, tables]` list of lists.
  """
  @spec engram_indices([integer()], Cholla.Train.Config.t()) :: [[integer()]]
  def engram_indices(ids, config) do
    {orders, heads, _} = Cholla.Train.Config.engram_geometry(config)
    slots = config.engram_slots
    stride = if config.engram_seed_heads > 0, do: config.engram_seed_heads, else: heads
    tokens = List.to_tuple(ids)
    n = tuple_size(tokens)

    tables =
      for {order, oi} <- Enum.with_index(orders), h <- 0..(heads - 1) do
        {order, @engram_seed * (oi * stride + h + 1) &&& 0xFFFFFFFF}
      end

    for t <- 0..(n - 1) do
      for {{order, seed}, ti} <- Enum.with_index(tables) do
        acc =
          Enum.reduce(0..(order - 1), seed, fn j, acc ->
            u = if t - j >= 0, do: elem(tokens, t - j), else: 0
            bxor(acc, u) * @engram_prime &&& 0xFFFFFFFF
          end)

        acc = bxor(acc, acc >>> 15)
        ti * slots + rem(acc, slots)
      end
    end
  end

  @doc """
  The tool calls in a rendered completion as a comparable list, `nil` when
  the call block does not parse. Used to score exact matches.
  """
  def calls_of(text) when is_binary(text) do
    case String.split(text, @tool_call_start, parts: 2) do
      [_] ->
        []

      [_, rest] ->
        body = rest |> String.split(@tool_call_end, parts: 2) |> hd()

        case JSON.decode(body) do
          {:ok, calls} when is_list(calls) -> canonical_calls(calls)
          _ -> nil
        end
    end
  end

  @doc "A list of `%{\"name\", \"arguments\"}` calls in a canonical, comparable form."
  def canonical_calls(calls) when is_list(calls) do
    for %{} = c <- calls, do: {to_string(c["name"]), canonical(c["arguments"] || %{})}
  end

  defp canonical(%{} = map),
    do: map |> Enum.map(fn {k, v} -> {to_string(k), canonical(v)} end) |> Enum.sort()

  defp canonical(list) when is_list(list), do: Enum.map(list, &canonical/1)
  defp canonical(n) when is_integer(n), do: n * 1.0
  defp canonical(v), do: v
end
