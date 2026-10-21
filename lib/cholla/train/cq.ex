defmodule Cholla.Train.CQ do
  @moduledoc """
  Cactus quantisation (CQ), the weight format the engine runs.

  Each group of 128 consecutive inputs is rotated by the normalised
  Walsh-Hadamard transform, its L2 norm stored in FP16, and each rotated
  weight replaced by the nearest entry of a shared 16-entry codebook on the
  unit sphere. The dequantised weight is `(codebook[idx] * norm) @ H`.

  Training applies this with a straight-through estimator so the LoRA
  adapter learns against the numerics the archive ships, as the Python
  trainer does. Activations are fake-quantised to 8 bits per vector.
  """

  import Nx.Defn

  @group 128
  @act_bits 8

  def group, do: @group

  @doc "The normalised Walsh-Hadamard matrix of order `n` (a power of two)."
  @spec hadamard(pos_integer()) :: Nx.Tensor.t()
  def hadamard(n) do
    rows = walsh(n)
    Nx.tensor(rows, type: :f32) |> Nx.divide(:math.sqrt(n))
  end

  defp walsh(1), do: [[1.0]]

  defp walsh(n) do
    h = walsh(div(n, 2))
    Enum.map(h, &(&1 ++ &1)) ++ Enum.map(h, &(&1 ++ Enum.map(&1, fn x -> -x end)))
  end

  @doc """
  Quantises `w` along its last axis (a multiple of 128) and dequantises it
  again: the weights the engine will see.
  """
  defn quantize(w, cb, h) do
    {idx, norm} = encode(w, cb, h)
    decode(idx, norm, cb, h, Nx.shape(w))
  end

  @doc "Codebook indices (u8, `[rows, in]`) and FP16-rounded group norms (`[rows, groups]`)."
  defn encode(w, cb, h) do
    d = Nx.axis_size(w, -1)
    rows = div(Nx.size(w), d)
    groups = Nx.reshape(w, {:auto, @group})
    rot = Nx.dot(groups, h)
    norm = Nx.sqrt(Nx.sum(rot * rot, axes: [1], keep_axes: true))
    unit = rot / Nx.max(norm, 1.0e-12)
    norm = norm |> Nx.as_type(:f16) |> Nx.as_type(:f32)
    # nearest codebook entry; ties go to the lower index, as searchsorted does
    idx = Nx.argmin(Nx.abs(Nx.new_axis(unit, 2) - cb), axis: 2) |> Nx.as_type(:u8)
    {Nx.reshape(idx, {rows, d}), Nx.reshape(norm, {rows, div(d, @group)})}
  end

  @doc "Dequantises `encode/3`'s output back to `shape`."
  defn decode(idx, norm, cb, h, shape) do
    values = Nx.take(cb, Nx.as_type(idx, :s32)) |> Nx.reshape({:auto, @group})
    rot = values * Nx.reshape(norm, {:auto, 1})
    Nx.reshape(Nx.dot(rot, h), shape)
  end

  @doc "Straight-through quantisation: quantised in the forward pass, identity in the backward."
  defn ste(w, cb, h) do
    w + stop_grad(quantize(w, cb, h) - w)
  end

  @doc """
  Symmetric 8-bit fake quantisation of each vector along the last axis,
  straight-through for the gradient. Matches the engine's per-token
  activation and per-head KV cache quantisation.
  """
  defn fake_quant_act(x) do
    qmax = 2 ** (@act_bits - 1) - 1
    absmax = Nx.reduce_max(Nx.abs(x), axes: [-1], keep_axes: true)
    scale = Nx.select(absmax > 0, absmax / qmax, 1.0)
    q = Nx.clip(round_half_even(x / scale), -qmax - 1, qmax) * scale
    x + stop_grad(q - x)
  end

  # numpy rounds half to even; Nx.round rounds half away from zero
  defnp round_half_even(x) do
    r = Nx.floor(x + 0.5)
    tie = x + 0.5 == r
    odd = Nx.remainder(r, 2) != 0
    Nx.select(tie and odd, r - 1, r)
  end

  @doc """
  Packs the indices of a `[out, in]` matrix at 4 bits, LSB-first (low nibble
  = even column), followed by the FP16 norms: one CQ blob.
  """
  @spec pack4(Nx.Tensor.t(), Nx.Tensor.t()) :: binary()
  def pack4(idx, norm) do
    {rows, d} = Nx.shape(idx)
    pairs = Nx.reshape(idx, {rows, div(d, 2), 2})
    lo = pairs[[.., .., 0]]
    hi = pairs[[.., .., 1]]
    packed = Nx.add(lo, Nx.multiply(hi, 16)) |> Nx.as_type(:u8)
    Nx.to_binary(packed) <> Nx.to_binary(Nx.as_type(norm, :f16))
  end
end
