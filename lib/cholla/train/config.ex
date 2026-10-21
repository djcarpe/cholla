defmodule Cholla.Train.Config do
  @moduledoc """
  The architecture geometry of a Needle 3 checkpoint, as saved in the
  checkpoint's metadata. Defaults are those of the Python trainer.
  """

  defstruct vocab_size: 16384,
            d_model: 768,
            num_heads: 12,
            num_kv_heads: 2,
            num_layers: 20,
            qk_head_dim: 48,
            v_head_dim: 64,
            max_seq_len: 4096,
            pad_token_id: 0,
            confidence_probes: 4,
            confidence_queries: 4,
            rope_theta: 100_000.0,
            engram_orders: [2, 3],
            engram_heads: 0,
            engram_slots: 18432,
            engram_seed_heads: 0,
            engram_layers: [3, 7, 11, 15, 19],
            global_layers: [4, 9, 14, 19],
            sliding_window: 1024,
            ladder_order: [],
            mhc_lanes: 4,
            qkv_conv_taps: 3,
            out_vocab: 0,
            kv_window: 0,
            kv_bits: 8,
            act_bits: 8

  @type t :: %__MODULE__{}

  @engram_sub_dim 128
  @engram_conv_taps 4

  @doc "Builds a config from the checkpoint's saved config map (string keys)."
  @spec from_saved(map()) :: t()
  def from_saved(saved) do
    fields = Map.keys(%__MODULE__{}) -- [:__struct__]

    Enum.reduce(fields, %__MODULE__{}, fn field, acc ->
      case Map.fetch(saved, Atom.to_string(field)) do
        {:ok, value} -> Map.put(acc, field, normalize(field, value))
        :error -> acc
      end
    end)
  end

  defp normalize(:rope_theta, v), do: v * 1.0
  defp normalize(_, v), do: v

  @doc "Engram geometry: `{orders, heads, sub_dim}`."
  def engram_geometry(%__MODULE__{} = c) do
    orders = c.engram_orders

    heads =
      if c.engram_heads > 0,
        do: c.engram_heads,
        else: max(1, div(c.d_model, length(orders) * @engram_sub_dim))

    {orders, heads, div(c.d_model, length(orders) * heads)}
  end

  def engram_conv_taps, do: @engram_conv_taps

  @doc "The padded Hadamard width: the next power of two at or above `d_model`."
  def hada_n(%__MODULE__{d_model: d}), do: next_pow2(d)

  @doc "The `{a, b}` Kronecker block sizes of the Hadamard MLP."
  def hada_blocks(n) do
    bits = bit_length(n - 1)
    b = Bitwise.bsl(1, div(bits, 2))
    {b, div(n, b)}
  end

  @doc """
  The KV window written into an archive: the smaller of the model's trained
  window and the window the engine's fixed KV budget allows at this depth.
  """
  def effective_kv_window(%__MODULE__{} = c) do
    budget = kv_budget_window(c)
    if c.kv_window > 0, do: min(budget, c.kv_window), else: budget
  end

  @kv_budget_bytes 11 * 1024 * 1024 + 512 * 1024
  @kv_group 32
  @kv_window_min 160

  defp kv_budget_window(c) do
    kd = c.num_kv_heads * c.qk_head_dim
    vd = c.num_kv_heads * c.v_head_dim
    sites = length(c.engram_layers)
    per_layer = kd + vd + (div(kd, @kv_group) + div(vd, @kv_group)) * 4
    per_site = c.d_model + div(c.d_model, @kv_group) * 4
    l = c.num_layers

    if c.sliding_window > 0 do
      n_global = length(c.global_layers)
      local = min(c.sliding_window, c.max_seq_len)
      fixed = ((l - n_global) * per_layer + sites * per_site) * local

      if n_global == 0 do
        if fixed <= @kv_budget_bytes, do: c.max_seq_len, else: max(@kv_window_min, local)
      else
        ctx = div(div(@kv_budget_bytes - fixed, n_global * per_layer), @kv_group) * @kv_group
        max(@kv_window_min, min(ctx, c.max_seq_len))
      end
    else
      per_pos = l * per_layer + sites * per_site
      window = div(div(@kv_budget_bytes, per_pos), @kv_group) * @kv_group
      max(@kv_window_min, min(window, c.max_seq_len))
    end
  end

  @doc """
  The nested block order of the depth ladder: both endpoints first, then the
  block bisecting the largest remaining gap, until every block is listed.
  """
  def ladder_order(%__MODULE__{ladder_order: order, num_layers: n}) when order != [] do
    if Enum.sort(order) != Enum.to_list(0..(n - 1)),
      do:
        raise(ArgumentError, "ladder_order #{inspect(order)} is not a permutation of #{n} blocks")

    order
  end

  def ladder_order(%__MODULE__{num_layers: n}), do: bisection_order(n)

  @doc "The original block indices kept by a rung of `depth` layers, sorted."
  def ladder_layer_indices(%__MODULE__{num_layers: n} = c, depth) do
    if depth < 2 or depth > n,
      do: raise(ArgumentError, "ladder depth must be in [2, #{n}], got #{depth}")

    c |> ladder_order() |> Enum.take(depth) |> Enum.sort()
  end

  @doc "The config of a `depth`-layer rung of this model."
  def ladder_config(%__MODULE__{} = c, depth) do
    selected = ladder_layer_indices(c, depth)
    remap = selected |> Enum.with_index() |> Map.new()

    %{
      c
      | num_layers: depth,
        ladder_order: for(l <- ladder_order(c), Map.has_key?(remap, l), do: remap[l]),
        global_layers: for(l <- c.global_layers, Map.has_key?(remap, l), do: remap[l]),
        engram_layers: for(l <- c.engram_layers, Map.has_key?(remap, l), do: remap[l])
    }
  end

  defp bisection_order(1), do: [0]

  defp bisection_order(n) when n > 1 do
    grow([0, n - 1], [0, n - 1], n)
  end

  defp grow(_selected, order, n) when length(order) == n, do: order

  defp grow(selected, order, n) do
    sorted = Enum.sort(selected)

    {_, left, right} =
      sorted
      |> Enum.zip(tl(sorted))
      |> Enum.filter(fn {l, r} -> r - l > 1 end)
      |> Enum.map(fn {l, r} -> {r - l, l, r} end)
      |> Enum.max_by(fn {gap, l, _} -> {gap, -l} end)

    candidate = div(left + right, 2)
    grow([candidate | selected], order ++ [candidate], n)
  end

  defp next_pow2(n), do: Bitwise.bsl(1, bit_length(n - 1))

  defp bit_length(0), do: 0
  defp bit_length(n), do: 1 + bit_length(Bitwise.bsr(n, 1))
end
