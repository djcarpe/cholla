defmodule Cholla.Train.LoRA do
  @moduledoc """
  LoRA adapters for the five attention projections of every block, their
  `.safetensors` format (shared with the Python trainer's `needle finetune`
  and `needle build`), and the AdamW training step.

  An adapter is `%{name => %{a: [L, in, rank], b: [L, rank, out]}}`; the
  merged weight is `W + scale * A @ B` with `scale = alpha / rank`.
  """

  alias Cholla.Train.{Model, Safetensors}

  @adapter_fields ~w(scale base rank seed pieces languages)

  @doc "A fresh adapter: `A` normal with std `1 / rank`, `B` zero."
  def init(targets, rank, seed) do
    key = Nx.Random.key(seed)

    {lora, _} =
      Enum.map_reduce(Enum.sort(Map.keys(targets)), key, fn name, key ->
        {l, in_dim, out_dim} = Nx.shape(targets[name])
        {a, key} = Nx.Random.normal(key, 0.0, 1.0, shape: {l, in_dim, rank}, type: :f32)
        {{name, %{a: Nx.divide(a, rank), b: Nx.broadcast(0.0, {l, rank, out_dim})}}, key}
      end)

    Map.new(lora)
  end

  @doc "Writes an adapter in the Python trainer's layout."
  def write(path, lora, meta) do
    tensors =
      for {name, %{a: a, b: b}} <- lora, {suffix, t} <- [{"A", a}, {"B", b}], into: %{} do
        {"lora/#{name}/#{suffix}", Nx.backend_copy(t, Nx.BinaryBackend)}
      end

    metadata =
      for key <- @adapter_fields, into: %{}, do: {key, JSON.encode!(meta[String.to_atom(key)])}

    Safetensors.write(path, tensors, metadata)
  end

  @doc "Reads an adapter: `{:ok, lora, meta}` with `meta` holding scale, base, rank, seed, languages."
  def read(path) do
    with {:ok, tensors, metadata} <- Safetensors.read(path, type: {:f, 32}) do
      lora =
        tensors
        |> Enum.filter(fn {name, _} -> String.starts_with?(name, "lora/") end)
        |> Enum.group_by(
          fn {name, _} ->
            name
            |> String.replace_prefix("lora/", "")
            |> String.replace_suffix("/A", "")
            |> String.replace_suffix("/B", "")
          end,
          fn {name, t} -> {if(String.ends_with?(name, "/A"), do: :a, else: :b), t} end
        )
        |> Map.new(fn {name, pairs} -> {name, Map.new(pairs)} end)

      meta =
        for key <- @adapter_fields, into: %{} do
          value =
            case metadata[key] do
              nil -> nil
              json -> JSON.decode!(json)
            end

          {String.to_atom(key), value}
        end

      {:ok, lora, meta}
    end
  end

  @doc "Zeroed AdamW moments for an adapter."
  def init_optimizer(lora) do
    zeros =
      Map.new(lora, fn {name, %{a: a, b: b}} ->
        {name, %{a: Nx.broadcast(0.0, Nx.shape(a)), b: Nx.broadcast(0.0, Nx.shape(b))}}
      end)

    %{mu: zeros, nu: zeros, count: Nx.tensor(0, type: :s32)}
  end

  @doc """
  The learning rate at `step` (from 0): linear warmup to `peak` over
  `warmup` steps, then cosine decay to 0 at `total`.
  """
  def learning_rate(step, peak, warmup, total) do
    cond do
      step < warmup ->
        peak * step / warmup

      true ->
        decay = max(total - warmup, 1)
        count = min(step - warmup, decay)
        peak * 0.5 * (1 + :math.cos(:math.pi() * count / decay))
    end
  end

  @b1 0.9
  @b2 0.999
  @eps 1.0e-8
  @weight_decay 1.0e-4
  @max_norm 1.0

  @doc """
  One training step: loss and gradient of the adapter on a batch, gradient
  clipped by global norm, AdamW with decoupled weight decay. Returns
  `{lora, opt_state, loss}`. Run under `Nx.Defn.jit/2`.
  """
  def step(lora, opt, lr, frozen, targets, consts, ids, mask, eidx, opts) do
    scale = Keyword.fetch!(opts, :scale)

    {loss, grads} =
      Nx.Defn.value_and_grad(lora, fn lora ->
        weights = Model.merge(frozen, targets, lora, scale, consts)
        Model.loss(weights, consts, ids, mask, eidx, opts)
      end)

    g_norm = global_norm(grads)
    clip = Nx.select(Nx.less(g_norm, @max_norm), 1.0, Nx.divide(@max_norm, g_norm))
    count = Nx.add(opt.count, 1)
    bc1 = Nx.subtract(1, Nx.pow(@b1, count))
    bc2 = Nx.subtract(1, Nx.pow(@b2, count))

    {lora, mu, nu} =
      Enum.reduce(Map.keys(lora), {%{}, %{}, %{}}, fn name, {l_acc, mu_acc, nu_acc} ->
        {p, m, v} =
          Enum.reduce([:a, :b], {%{}, %{}, %{}}, fn k, {p, m, v} ->
            g = Nx.multiply(grads[name][k], clip)
            param = lora[name][k]
            mu = Nx.add(Nx.multiply(@b1, opt.mu[name][k]), Nx.multiply(1 - @b1, g))

            nu =
              Nx.add(Nx.multiply(@b2, opt.nu[name][k]), Nx.multiply(1 - @b2, Nx.multiply(g, g)))

            update = Nx.divide(Nx.divide(mu, bc1), Nx.add(Nx.sqrt(Nx.divide(nu, bc2)), @eps))
            update = Nx.add(update, Nx.multiply(@weight_decay, param))
            param = Nx.subtract(param, Nx.multiply(lr, update))
            {Map.put(p, k, param), Map.put(m, k, mu), Map.put(v, k, nu)}
          end)

        {Map.put(l_acc, name, p), Map.put(mu_acc, name, m), Map.put(nu_acc, name, v)}
      end)

    {lora, %{mu: mu, nu: nu, count: count}, loss}
  end

  defp global_norm(grads) do
    grads
    |> Enum.flat_map(fn {_, %{a: a, b: b}} -> [a, b] end)
    |> Enum.map(&Nx.sum(Nx.multiply(&1, &1)))
    |> Enum.reduce(&Nx.add/2)
    |> Nx.sqrt()
  end
end
