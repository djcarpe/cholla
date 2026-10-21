defmodule Cholla.Train do
  @moduledoc """
  Fine-tunes Needle on your own tool-calling examples and builds the `.cact`
  the engine loads, in Elixir: no Python, no `needle` CLI.

      {:ok, adapter} = Cholla.Train.finetune("examples.jsonl", out: "my.safetensors")
      {:ok, cact} = Cholla.Train.build(lora: adapter, out: "my.cact")
      agent = Cholla.agent(tools: tools, weights: cact)

  ## Data

  One example per JSONL line (or a list of maps): `"query"`, `"tools"` (the
  schemas the model is offered, as `Cholla.Tool.to_schema/1` returns them),
  `"answers"` (`[%{"name", "arguments"}]`, `[]` when no tool fits) and
  optionally `"system"` and `"reasoning"`. The OpenAI chat format
  (`"messages"` + `"tools"`) is accepted for single-turn lines.

  ## How it trains

  The float checkpoint (`needle3.safetensors`, 242 MB, fetched from Hugging
  Face on first use) runs in Nx with the same straight-through
  quantisation as the exported archive (W4 weights, A8 activations), so the
  adapter learns against the numerics that ship. A LoRA adapter on the
  attention projections of every block is trained with AdamW, warmup and
  cosine decay, as `needle finetune` does, and saved in the same
  `.safetensors` layout, so adapters move between the two trainers.
  `build/1` merges it and writes a W4A8 `.cact` (about 63 MB).

  Training needs the EXLA compiler (`{:exla, "~> 1.0"}` in your deps). It
  runs on the CPU by default; a 100-example set takes some minutes. For a
  GPU, compile EXLA with `XLA_TARGET=cuda12` and set
  `config :exla, default_client: :cuda`.
  """

  require Logger
  alias Cholla.Train.{Checkpoint, Data, Export, LoRA, Model, Tokenizer}

  @defaults [
    epochs: 3,
    batch_size: 16,
    lr: 1.0e-4,
    lora_rank: 16,
    lora_alpha: 32.0,
    max_len: 1024,
    val_split: 0.1,
    seed: 0
  ]

  @doc """
  Trains a LoRA adapter on `examples` (a JSONL path or a list of maps) and
  writes it to `:out` (default `needle_lora.safetensors`). Returns
  `{:ok, path, stats}` with the final training loss, validation loss and
  the held-out examples.

  ## Options

    * `:out` - adapter path
    * `:checkpoint`, `:base_cact` - the float checkpoint and the base
      archive; both are fetched into the cache when omitted
    * `:epochs` (3), `:batch_size` (16), `:lr` (1.0e-4)
    * `:lora_rank` (16), `:lora_alpha` (32)
    * `:max_len` (1024) - cap on the sequence length; training uses the
      smallest power-of-two bucket from 128 that fits the data
    * `:val_split` (0.1) - fraction held out for validation, 0 to disable
    * `:seed` (0) - adapter init, split and shuffling
    * `:progress` - a function called with each progress line
  """
  @spec finetune(Path.t() | [map()], keyword()) :: {:ok, Path.t(), map()} | {:error, term()}
  def finetune(examples, opts \\ []) do
    opts = Keyword.merge(@defaults, opts)
    emit = emitter(opts)

    with {:ok, examples} <- load_examples(examples, emit),
         :ok <- ensure_compiler(),
         {:ok, checkpoint_path} <- checkpoint_path(opts),
         {:ok, base} <- base_cact(opts),
         {:ok, ckpt} <- Checkpoint.load(checkpoint_path),
         {:ok, consts} <- Checkpoint.constants(base) do
      run(examples, ckpt, consts, checkpoint_path, opts, emit)
    end
  end

  defp run(examples, ckpt, consts, checkpoint_path, opts, emit) do
    cfg = ckpt.config
    tok = consts.tokenizer
    tc = Map.take(consts, [:cb4, :hadamard, :p1, :p2])
    emit.("  #{pad("backend")} #{backend()}  float32")
    emit.("  #{pad("depth")} #{cfg.num_layers} layers")

    max_len = Data.fit_max_len(tok, examples, opts[:max_len])

    emit.(
      "  #{pad("data")} #{length(examples)} examples  seq_len #{max_len}  cap #{opts[:max_len]}"
    )

    seed = opts[:seed]
    rng = :rand.seed_s(:exsss, {seed, seed + 1, seed + 2})
    {shuffled, rng} = shuffle(examples, rng)
    n_val = min(trunc(length(examples) * opts[:val_split]), length(examples) - 1) |> max(0)
    {val, train} = Enum.split(shuffled, n_val)
    if n_val > 0, do: emit.("  #{pad("holdout")} #{n_val} examples for validation")

    encode = fn ex ->
      {ids, mask} = Data.encode(tok, ex, max_len)
      {ids, mask, Data.engram_indices(ids, cfg)}
    end

    train_enc = Enum.map(train, encode)
    val_enc = Enum.map(val, encode)

    frozen = Model.prepare(ckpt.params, cfg, tc)
    targets = Model.targets(ckpt.params)
    emit.("  #{pad("numerics")} CQ W4 STE + A8 (matches export)")

    rank = opts[:lora_rank]
    scale = opts[:lora_alpha] / rank
    lora = LoRA.init(targets, rank, seed)
    opt = LoRA.init_optimizer(lora)

    emit.(
      "  #{pad("lora")} rank #{rank}  alpha #{opts[:lora_alpha]}  #{map_size(lora)} weight groups"
    )

    batch = opts[:batch_size]
    steps_per_epoch = div(length(train) + batch - 1, batch)
    total = opts[:epochs] * steps_per_epoch
    warmup = min(max(1, div(total, 20)), total - 1) |> max(1)

    emit.(
      "  #{pad("schedule")} #{total} steps  warmup #{warmup}  cosine decay  clip 1.0  (compiling...)"
    )

    model_opts = [config: cfg, scale: scale]
    step_fn = Nx.Defn.jit(&LoRA.step(&1, &2, &3, &4, &5, &6, &7, &8, &9, model_opts))

    eval_fn =
      Nx.Defn.jit(fn lora, frozen, targets, consts, ids, mask, eidx ->
        Model.loss(Model.merge(frozen, targets, lora, scale, consts), consts, ids, mask, eidx,
          config: cfg
        )
      end)

    every = max(1, div(total, 50))

    {lora, _opt, _rng, _step, last, val_loss} =
      Enum.reduce(1..opts[:epochs], {lora, opt, rng, 0, 0.0, nil}, fn epoch,
                                                                      {lora, opt, rng, step,
                                                                       _last, _} ->
        {order, rng} = shuffle(train_enc, rng)

        {lora, opt, step, last} =
          order
          |> Enum.chunk_every(batch)
          |> Enum.reduce({lora, opt, step, 0.0}, fn chunk, {lora, opt, step, _} ->
            {ids, mask, eidx} = tensors(chunk, batch, max_len)
            lr = Nx.tensor(LoRA.learning_rate(step, opts[:lr], warmup, total), type: :f32)
            {lora, opt, loss} = step_fn.(lora, opt, lr, frozen, targets, tc, ids, mask, eidx)
            loss = Nx.to_number(loss)
            step = step + 1

            if rem(step, every) == 0,
              do: emit.("  #{pad("step")} #{step}/#{total}  loss #{fmt(loss)}")

            {lora, opt, step, loss}
          end)

        val_loss =
          if n_val > 0 do
            losses =
              val_enc
              |> Enum.chunk_every(batch)
              |> Enum.map(fn chunk ->
                {ids, mask, eidx} = tensors(chunk, batch, max_len)

                {Nx.to_number(eval_fn.(lora, frozen, targets, tc, ids, mask, eidx)),
                 length(chunk)}
              end)

            Enum.sum(for {l, n} <- losses, do: l * n) / n_val
          end

        if val_loss,
          do:
            emit.(
              "  #{pad("epoch")} #{epoch}/#{opts[:epochs]}  loss #{fmt(last)}  val #{fmt(val_loss)}"
            ),
          else: emit.("  #{pad("epoch")} #{epoch}/#{opts[:epochs]}  loss #{fmt(last)}")

        {lora, opt, rng, step, last, val_loss}
      end)

    out = Path.expand(opts[:out] || "needle_lora.safetensors")
    File.mkdir_p!(Path.dirname(out))

    :ok =
      LoRA.write(out, lora, %{
        scale: scale,
        base: checkpoint_path,
        rank: rank,
        seed: seed,
        pieces: nil,
        languages: nil
      })

    emit.("  #{pad("adapter")} #{out}")

    emit.(
      "  #{pad("note")} local tuning leaves the confidence head untrained; build drops it and confidence reports nil"
    )

    {:ok, out,
     %{loss: last, val_loss: val_loss, holdout: val, train: length(train), steps: total}}
  end

  @doc """
  Writes a `.cact` from the base checkpoint, merging a LoRA adapter
  (`lora: path`) and/or taking a shallower rung (`layers: n`, 2 up to the
  base depth). Without either, the published base archive is copied.
  Returns `{:ok, path, info}`.

  ## Options

    * `:out` - archive path (default `<checkpoint>.cact` beside the adapter)
    * `:lora` - adapter to merge; its confidence head is dropped, as the
      adapter does not train it
    * `:layers` - depth of the rung to export
    * `:checkpoint`, `:base_cact` - as in `finetune/2`; the adapter's
      recorded base checkpoint is used when it exists
    * `:progress` - a function called with each progress line
  """
  @spec build(keyword()) :: {:ok, Path.t(), map()} | {:error, term()}
  def build(opts \\ []) do
    emit = emitter(opts)

    with {:ok, base} <- base_cact(opts) do
      {:ok, archive} = Cholla.Train.Cact.read(base)
      base_layers = archive.header.num_layers
      layers = opts[:layers]

      cond do
        opts[:lora] == nil and (layers == nil or layers == base_layers) ->
          out = opts[:out] || raise(ArgumentError, "pass :out for the archive path")
          File.cp!(base, out)
          emit.("  #{pad("wrote")} #{out}  #{mb(out)} MB  the published base archive")
          {:ok, out, %{bytes: File.stat!(out).size, layers: base_layers}}

        true ->
          build_merged(opts, base, layers, emit)
      end
    end
  end

  defp build_merged(opts, base, layers, emit) do
    adapter =
      case opts[:lora] do
        nil -> {:ok, nil, %{}}
        path -> LoRA.read(path)
      end

    with {:ok, lora, meta} <- adapter,
         :ok <- not_whistle(meta, opts[:lora]),
         :ok <- ensure_compiler(),
         {:ok, checkpoint_path} <- checkpoint_path(opts, meta[:base]),
         {:ok, ckpt} <- Checkpoint.load(checkpoint_path),
         {:ok, consts} <- Checkpoint.constants(base) do
      params = ckpt.params

      params =
        if lora do
          scale = meta[:scale] || 2.0

          merged =
            Map.new(Model.targets(params), fn {name, w} ->
              %{a: a, b: b} = Map.fetch!(lora, name)
              {name, Nx.add(w, Nx.multiply(scale, Nx.dot(a, [2], [0], b, [1], [0])))}
            end)

          emit.("  #{pad("merged")} #{map_size(lora)} weight groups  #{opts[:lora]}")
          params = Map.merge(params, merged)

          if Map.has_key?(params, "confidence_head/probes") do
            emit.(
              "  #{pad("dropped")} the confidence head; it is not trained locally, so confidence reports nil"
            )

            Export.drop_head(params, "confidence_head")
          else
            params
          end
        else
          params
        end

      ckpt = %{ckpt | params: params}

      ckpt =
        if layers && layers != ckpt.config.num_layers,
          do: Checkpoint.ladder_slice(ckpt, layers),
          else: ckpt

      emit.("  #{pad("depth")} #{ckpt.config.num_layers} layers")

      out =
        Path.expand(
          opts[:out] ||
            Path.rootname(Path.basename(checkpoint_path)) <> ".cact"
        )

      with {:ok, info} <- Export.write(out, ckpt.params, ckpt.config, consts) do
        emit.("  #{pad("wrote")} #{out}  #{mb(out)} MB  #{info.tensors} tensors  W4A8")
        {:ok, out, Map.put(info, :layers, ckpt.config.num_layers)}
      end
    end
  end

  defp not_whistle(%{languages: langs}, path) when is_list(langs),
    do: {:error, "#{path} is a Whistle adapter; Cholla builds Needle text adapters"}

  defp not_whistle(_, _), do: :ok

  @doc """
  Scores a `.cact` on examples through the engine: the fraction whose tool
  calls (names and arguments) the model reproduces exactly. Returns
  `{:ok, %{correct, total, accuracy, misses}}`. Options: `:pool`,
  `:max_new_tokens` (256).
  """
  @spec evaluate(Path.t(), [map()] | Path.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def evaluate(cact, examples, opts \\ []) do
    with {:ok, examples} <- load_examples(examples, fn _ -> :ok end) do
      results =
        for example <- examples do
          agent =
            Cholla.agent(
              tools: example["tools"] || [],
              system: example["system"] || "",
              auto_date: false,
              weights: cact,
              stateless: true,
              pool: opts[:pool],
              max_new_tokens: Keyword.get(opts, :max_new_tokens, 256)
            )

          want = Data.canonical_calls(example["answers"] || [])

          case Cholla.complete(agent, example["query"]) do
            {:ok, %{"function_calls" => calls}} when is_list(calls) ->
              {Data.canonical_calls(calls) == want, example, calls}

            {:ok, response} ->
              {want == [], example, response}

            {:error, reason} ->
              {false, example, {:error, reason}}
          end
        end

      correct = Enum.count(results, &elem(&1, 0))
      total = length(results)

      misses =
        for {false, ex, got} <- results, do: %{query: ex["query"], want: ex["answers"], got: got}

      {:ok,
       %{
         correct: correct,
         total: total,
         accuracy: if(total > 0, do: correct / total, else: 0.0),
         misses: misses
       }}
    end
  end

  @doc "The chat-template text of an example, as the model sees it."
  defdelegate render(example), to: Data

  @doc "The tokenizer of the base archive."
  def tokenizer(opts \\ []) do
    with {:ok, base} <- base_cact(opts), do: Tokenizer.from_cact(base)
  end

  defp load_examples(path, emit) when is_binary(path) do
    if File.exists?(path) do
      {examples, skipped} = Data.read_jsonl(path)

      if skipped > 0,
        do:
          emit.(
            "  skipped #{skipped} line(s) that are neither query/answers nor single-turn chat format"
          )

      if examples == [], do: {:error, "no usable examples in #{path}"}, else: {:ok, examples}
    else
      {:error, {:enoent, path}}
    end
  end

  defp load_examples(examples, _emit) when is_list(examples) do
    examples =
      examples |> Enum.map(&(&1 |> stringify() |> Data.normalize())) |> Enum.reject(&is_nil/1)

    if examples == [], do: {:error, "no usable examples"}, else: {:ok, examples}
  end

  defp stringify(%{} = map), do: Map.new(map, fn {k, v} -> {to_string(k), stringify(v)} end)
  defp stringify(list) when is_list(list), do: Enum.map(list, &stringify/1)
  defp stringify(v), do: v

  defp checkpoint_path(opts, recorded \\ nil) do
    cond do
      opts[:checkpoint] -> existing(Path.expand(opts[:checkpoint]))
      recorded && File.exists?(recorded) -> {:ok, recorded}
      true -> Cholla.Download.checkpoint_path()
    end
  end

  defp base_cact(opts) do
    case opts[:base_cact] do
      nil -> Cholla.Download.weights_path()
      path -> existing(Path.expand(path))
    end
  end

  defp existing(path),
    do: if(File.exists?(path), do: {:ok, path}, else: {:error, {:enoent, path}})

  defp ensure_compiler do
    if Code.ensure_loaded?(EXLA) do
      if Nx.Defn.default_options()[:compiler] in [nil, Nx.Defn.Evaluator] do
        Nx.Defn.global_default_options(compiler: EXLA)
      end

      if Nx.default_backend() == {Nx.BinaryBackend, []},
        do: Nx.global_default_backend(EXLA.Backend)

      :ok
    else
      case Nx.Defn.default_options()[:compiler] do
        nil ->
          {:error, "training needs a compiled Nx backend: add {:exla, \"~> 1.0\"} to your deps"}

        _ ->
          :ok
      end
    end
  end

  defp backend do
    case Nx.default_backend() do
      {EXLA.Backend, opts} ->
        "exla #{opts[:client] || Application.get_env(:exla, :default_client, :host)}"

      {mod, _} ->
        inspect(mod)
    end
  end

  defp shuffle(list, rng) do
    {keyed, rng} =
      Enum.map_reduce(list, rng, fn x, rng ->
        {r, rng} = :rand.uniform_s(rng)
        {{r, x}, rng}
      end)

    {keyed |> Enum.sort_by(&elem(&1, 0)) |> Enum.map(&elem(&1, 1)), rng}
  end

  # a batch as tensors, padded to the batch size with masked-out rows
  defp tensors(chunk, batch, max_len) do
    pad = batch - length(chunk)
    n_tables = chunk |> hd() |> elem(2) |> hd() |> length()
    blank_eidx = List.duplicate(List.duplicate(0, n_tables), max_len)

    rows =
      chunk ++
        List.duplicate(
          {List.duplicate(0, max_len), List.duplicate(0.0, max_len), blank_eidx},
          pad
        )

    {
      Nx.tensor(Enum.map(rows, &elem(&1, 0)), type: :s32),
      Nx.tensor(Enum.map(rows, &elem(&1, 1)), type: :f32),
      Nx.tensor(Enum.map(rows, &elem(&1, 2)), type: :s32)
    }
  end

  defp emitter(opts) do
    progress = opts[:progress]

    fn msg ->
      Logger.info("cholla.train:" <> msg)
      if is_function(progress, 1), do: progress.(msg)
      :ok
    end
  end

  defp pad(label), do: String.pad_trailing(label, 9)
  defp fmt(x) when is_float(x), do: :erlang.float_to_binary(x, decimals: 4)
  defp fmt(x), do: to_string(x)
  defp mb(path), do: :erlang.float_to_binary(File.stat!(path).size / 1.0e6, decimals: 2)
end
