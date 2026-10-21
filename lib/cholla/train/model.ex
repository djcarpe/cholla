defmodule Cholla.Train.Model do
  @moduledoc """
  The Needle 3 network in Nx, for training: a stack of blocks joined by
  manifold hyper-connections (four residual lanes mixed per block), grouped
  query attention with causal convolution taps, Hadamard MLPs, and engram
  n-gram tables merged in at five blocks. The forward pass fake-quantises
  weights (CQ, 4-bit) and activations (8-bit) so the adapter trains against
  the numerics of the exported archive.

  The functions here take and return `Nx` tensors and are meant to run
  under `Nx.Defn.jit/2`, which traces them once per shape. Weights are a
  flat map keyed by checkpoint name. `prepare/4` quantises the frozen
  weights once; `merge/5` applies a LoRA adapter to the five attention
  projections and quantises them straight-through for each step.
  """

  alias Cholla.Train.{Config, CQ}

  @lora_targets ~w(q_proj k_proj v_proj gate_proj out_proj)
  @sa "stack/layers/block/self_attn/"
  @hm "stack/layers/block/hadamard_mlp/"

  @doc "The names of the five projections a LoRA adapter trains, per block."
  def lora_targets, do: for(t <- @lora_targets, do: @sa <> t <> "/kernel")

  @doc """
  Prepares the frozen weights: everything except the LoRA targets and the
  heads, with the matmul weights CQ-quantised (unless `quant: false`) and
  projections transposed to `[out, in]`.
  """
  def prepare(params, %Config{} = config, consts, opts \\ []) do
    quant = Keyword.get(opts, :quant, true)
    q = fn t -> if quant, do: CQ.quantize(t, consts.cb4, consts.hadamard), else: t end
    sites = length(config.engram_layers)

    frozen =
      params
      |> Enum.reject(fn {name, _} ->
        name in lora_targets() or String.contains?(name, "_head/")
      end)
      |> Map.new()
      |> Map.merge(%{
        "embedding/embedding" => q.(params["embedding/embedding"]),
        "stack/mhc_phi_pre" => phi(params["stack/mhc_phi_pre"], q),
        "stack/mhc_phi_post" => phi(params["stack/mhc_phi_post"], q),
        "stack/mhc_phi_res" => phi(params["stack/mhc_phi_res"], q)
      })

    Enum.reduce(0..(sites - 1)//1, frozen, fn s, acc ->
      tables = params["engrams_#{s}/embedding"]
      {n_tables, slots, sub} = Nx.shape(tables)

      flat =
        Nx.concatenate(for i <- 0..(n_tables - 1), do: q.(Nx.reshape(tables[i], {slots, sub})))

      acc
      |> Map.put("engrams_#{s}/embedding", flat)
      |> Map.put(
        "engrams_#{s}/key_proj/kernel",
        q.(Nx.transpose(params["engrams_#{s}/key_proj/kernel"]))
      )
      |> Map.put(
        "engrams_#{s}/value_proj/kernel",
        q.(Nx.transpose(params["engrams_#{s}/value_proj/kernel"]))
      )
    end)
  end

  # [L, nC, n] quantised along nC, kept as [L, n, nC]
  defp phi(t, q), do: q.(Nx.transpose(t, axes: [0, 2, 1]))

  @doc "The base values of the LoRA targets, as `%{name => [L, in, out]}`."
  def targets(params), do: Map.new(lora_targets(), &{&1, params[&1]})

  @doc """
  Merges a LoRA adapter (`%{name => %{a: [L, in, r], b: [L, r, out]}}`) into
  the base targets, quantises the result straight-through, and adds them to
  the frozen weights as `[L, out, in]`. With `quant: false`, no quantisation.
  """
  def merge(frozen, targets, lora, scale, consts, opts \\ []) do
    quant = Keyword.get(opts, :quant, true)

    merged =
      Map.new(targets, fn {name, w} ->
        w = Nx.add(w, Nx.multiply(scale, Nx.dot(lora[name].a, [2], [0], lora[name].b, [1], [0])))
        wt = Nx.transpose(w, axes: [0, 2, 1])
        {name, if(quant, do: CQ.ste(wt, consts.cb4, consts.hadamard), else: wt)}
      end)

    Map.merge(frozen, merged)
  end

  @doc "Transposes and quantises the base targets without an adapter."
  def merge_base(frozen, targets, consts, opts \\ []) do
    quant = Keyword.get(opts, :quant, true)

    Map.merge(
      frozen,
      Map.new(targets, fn {name, w} ->
        wt = Nx.transpose(w, axes: [0, 2, 1])
        {name, if(quant, do: CQ.quantize(wt, consts.cb4, consts.hadamard), else: wt)}
      end)
    )
  end

  @doc """
  Mean token cross-entropy of the next-token prediction over the masked
  positions. `ids` `[B, T]`, `mask` `[B, T]` floats, `eidx` `[B, T, tables]`.
  """
  def loss(weights, consts, ids, mask, eidx, opts) do
    logits = forward(weights, consts, ids, eidx, opts)
    t = Nx.axis_size(ids, 1)
    logits = Nx.slice_along_axis(logits, 0, t - 1, axis: 1)
    targets = Nx.slice_along_axis(ids, 1, t - 1, axis: 1)
    mask = Nx.slice_along_axis(mask, 1, t - 1, axis: 1)
    lse = Nx.logsumexp(logits, axes: [-1])

    picked =
      Nx.take_along_axis(logits, Nx.new_axis(targets, -1), axis: 2) |> Nx.squeeze(axes: [-1])

    ce = Nx.subtract(lse, picked)
    Nx.divide(Nx.sum(Nx.multiply(ce, mask)), Nx.max(Nx.sum(mask), 1.0))
  end

  @doc """
  Logits `[B, T, vocab]` for token ids `[B, T]`. `eidx` holds the engram rows
  (see `Cholla.Train.Data.engram_indices/2`). Options: `config` (required),
  `quant` (default true), `hidden: true` also returns the per-block hidden
  states `[L, B, T, d]`.
  """
  def forward(w, c, ids, eidx, opts) do
    cfg = Keyword.fetch!(opts, :config)
    quant = Keyword.get(opts, :quant, true)
    {b, t} = Nx.shape(ids)
    d = cfg.d_model
    n = cfg.mhc_lanes

    x = Nx.multiply(Nx.take(w["embedding/embedding"], ids), :math.sqrt(d))
    {cos, sin} = rope(t, cfg.qk_head_dim, cfg.rope_theta)
    mask = causal_mask(t)

    local_mask =
      if cfg.sliding_window > 0 do
        pos = Nx.subtract(Nx.iota({t, 1}), Nx.iota({1, t}))
        Nx.logical_and(mask, Nx.less(pos, cfg.sliding_window))
      else
        mask
      end

    engram = engram_kv(w, cfg, eidx, quant, t)
    stream = Nx.broadcast(Nx.new_axis(x, 2), {b, t, n, d})

    {stream, hidden} =
      Enum.reduce(0..(cfg.num_layers - 1)//1, {stream, []}, fn l, {stream, hidden} ->
        layer_mask = if l in cfg.global_layers, do: mask, else: local_mask

        kv =
          case Enum.find_index(cfg.engram_layers, &(&1 == l)) do
            nil -> nil
            site -> Enum.at(engram, site)
          end

        lane = for i <- 0..(n - 1), do: if(rem(l, n) == i, do: 1.0, else: 0.0)
        pre_off = Nx.tensor(Enum.map(lane, &(8 * &1 - 4)), type: :f32)
        post_off = Nx.tensor(Enum.map(lane, &(-4 * (1 - &1))), type: :f32)

        stream =
          hyper_block(stream, w, c, cfg, l, pre_off, post_off, layer_mask, cos, sin, kv, quant)

        {stream, [Nx.mean(stream, axes: [2]) | hidden]}
      end)

    x = Nx.mean(stream, axes: [2])
    x = zcrms(x, w["stack/final_norm/scale"])
    head = w["embedding/embedding"]

    head =
      if cfg.out_vocab > 0, do: Nx.slice_along_axis(head, 0, cfg.out_vocab, axis: 0), else: head

    logits = Nx.dot(aq(x, quant), [2], head, [1])

    if opts[:hidden], do: {logits, Nx.stack(Enum.reverse(hidden))}, else: logits
  end

  defp hyper_block(stream, w, c, cfg, l, pre_off, post_off, mask, cos, sin, kv, quant) do
    {b, t, n, d} = Nx.shape(stream)
    nx = aq(rms_unit(Nx.reshape(stream, {b, t, n * d})), quant)

    hpre =
      Nx.dot(nx, [2], w["stack/mhc_phi_pre"][l], [1])
      |> Nx.multiply(w["stack/mhc_a_pre"][l])
      |> Nx.add(w["stack/mhc_b_pre"][l])
      |> Nx.add(pre_off)
      |> sigmoid()

    u = Nx.sum(Nx.multiply(Nx.new_axis(hpre, 3), stream), axes: [2])
    y = Nx.subtract(block(u, w, c, cfg, l, mask, cos, sin, kv, quant), u)

    hpost =
      Nx.dot(nx, [2], w["stack/mhc_phi_post"][l], [1])
      |> Nx.multiply(w["stack/mhc_a_post"][l])
      |> Nx.add(w["stack/mhc_b_post"][l])
      |> Nx.add(post_off)
      |> sigmoid()
      |> Nx.multiply(2)

    res = Nx.dot(nx, [2], w["stack/mhc_phi_res"][l], [1]) |> Nx.reshape({b, t, n, n})
    hres = sinkhorn(Nx.add(Nx.multiply(w["stack/mhc_a_res"][l], res), w["stack/mhc_b_res"][l]))

    Nx.dot(hres, [3], [0, 1], stream, [2], [0, 1])
    |> Nx.add(Nx.multiply(Nx.new_axis(hpost, 3), Nx.new_axis(y, 2)))
  end

  defp sinkhorn(logits) do
    Enum.reduce(1..20, logits, fn _, log_k ->
      log_k = Nx.subtract(log_k, Nx.logsumexp(log_k, axes: [-1], keep_axes: true))
      Nx.subtract(log_k, Nx.logsumexp(log_k, axes: [-2], keep_axes: true))
    end)
    |> Nx.exp()
  end

  defp block(x, w, c, cfg, l, mask, cos, sin, kv, quant) do
    x =
      case kv do
        nil ->
          x

        {ek, ev} ->
          alpha =
            Nx.sum(Nx.multiply(rms_unit(x), rms_unit(ek)), axes: [-1], keep_axes: true)
            |> Nx.divide(:math.sqrt(cfg.d_model))
            |> sigmoid()

          Nx.add(x, Nx.multiply(alpha, ev))
      end

    skip = x
    h = zcrms(x, w["stack/layers/block/ZCRMSNorm_0/scale"][l])
    a = attention(h, w, cfg, l, mask, cos, sin, quant)
    a = zcrms(a, w["stack/layers/block/post_attn_norm/scale"][l])
    x = Nx.add(skip, Nx.multiply(sigmoid(w["stack/layers/block/attn_gate"][l]), a))

    skip = x
    h = zcrms(x, w["stack/layers/block/pre_hada_norm/scale"][l])
    Nx.add(skip, hadamard_mlp(h, w, c, cfg, l))
  end

  defp attention(x, w, cfg, l, mask, cos, sin, quant) do
    {b, t, _} = Nx.shape(x)
    nh = cfg.num_heads
    nkv = cfg.num_kv_heads
    qk = cfg.qk_head_dim
    vh = cfg.v_head_dim

    xq = aq(x, quant)
    q = Nx.dot(xq, [2], w[@sa <> "q_proj/kernel"][l], [1])
    k = Nx.dot(xq, [2], w[@sa <> "k_proj/kernel"][l], [1])
    v = Nx.dot(xq, [2], w[@sa <> "v_proj/kernel"][l], [1])

    {q, k, v} =
      if cfg.qkv_conv_taps > 0 do
        {taps(q, w[@sa <> "q_taps"][l]), taps(k, w[@sa <> "k_taps"][l]),
         taps(v, w[@sa <> "v_taps"][l])}
      else
        {q, k, v}
      end

    q = q |> Nx.reshape({b, t, nh, qk}) |> Nx.transpose(axes: [0, 2, 1, 3])
    k = k |> Nx.reshape({b, t, nkv, qk}) |> Nx.transpose(axes: [0, 2, 1, 3])
    v = v |> Nx.reshape({b, t, nkv, vh}) |> Nx.transpose(axes: [0, 2, 1, 3])

    q = zcrms(q, w[@sa <> "q_norm/scale"][l]) |> rope_apply(cos, sin) |> aq(quant)
    k = zcrms(k, w[@sa <> "k_norm/scale"][l]) |> rope_apply(cos, sin) |> aq(quant)
    v = aq(v, quant)

    repeats = div(nh, nkv)

    k =
      k |> Nx.new_axis(2) |> Nx.broadcast({b, nkv, repeats, t, qk}) |> Nx.reshape({b, nh, t, qk})

    v =
      v |> Nx.new_axis(2) |> Nx.broadcast({b, nkv, repeats, t, vh}) |> Nx.reshape({b, nh, t, vh})

    scores = Nx.dot(q, [3], [0, 1], k, [3], [0, 1]) |> Nx.divide(:math.sqrt(qk))
    mask = mask |> Nx.new_axis(0) |> Nx.new_axis(0) |> Nx.broadcast(Nx.shape(scores))
    scores = Nx.select(mask, scores, Nx.Constants.min_finite({:f, 32}))
    out = Nx.dot(softmax(scores), [3], [0, 1], v, [2], [0, 1])
    out = out |> Nx.transpose(axes: [0, 2, 1, 3]) |> Nx.reshape({b, t, nh * vh})

    out = Nx.multiply(out, sigmoid(Nx.dot(xq, [2], w[@sa <> "gate_proj/kernel"][l], [1])))
    Nx.dot(aq(out, quant), [2], w[@sa <> "out_proj/kernel"][l], [1])
  end

  # causal depthwise taps along time: sum_j taps[j] * z shifted right by j
  defp taps(z, taps) do
    {n_taps, _} = Nx.shape(taps)

    Enum.reduce(1..(n_taps - 1)//1, Nx.multiply(taps[0], z), fn j, acc ->
      Nx.add(acc, Nx.multiply(taps[j], shift_right(z, j)))
    end)
  end

  defp shift_right(z, j), do: Nx.pad(z, 0.0, [{0, 0, 0}, {j, -j, 0}, {0, 0, 0}])

  defp hadamard_mlp(x, w, c, cfg, l) do
    {b, t, d} = Nx.shape(x)
    n = Config.hada_n(cfg)
    {ba, bb} = Config.hada_blocks(n)

    cond =
      Nx.add(
        1,
        Nx.dot(softmax(Nx.dot(x, [2], w[@hm <> "cond_v"][l], [0])), [2], w[@hm <> "cond_u"][l], [
          0
        ])
      )

    z = Nx.pad(x, 0.0, [{0, 0, 0}, {0, 0, 0}, {0, n - d, 0}])

    z =
      kron(Nx.multiply(w[@hm <> "d1"][l], z), w[@hm <> "w1a"][l], w[@hm <> "w1b"][l], ba, bb)
      |> Nx.take(c.p1, axis: 2)

    z =
      Nx.multiply(w[@hm <> "d2"][l], Nx.multiply(cond, z))
      |> Nx.add(w[@hm <> "b2"][l])
      |> silu()
      |> kron(w[@hm <> "w2a"][l], w[@hm <> "w2b"][l], ba, bb)
      |> Nx.take(c.p2, axis: 2)

    z = kron(Nx.multiply(w[@hm <> "d3"][l], z), w[@hm <> "w3a"][l], w[@hm <> "w3b"][l], ba, bb)

    Nx.multiply(w[@hm <> "d4"][l], z)
    |> Nx.slice_along_axis(0, d, axis: 2)
    |> Nx.reshape({b, t, d})
  end

  # z[..., i*bb + j] -> sum_ij z[i, j] a[i, k] b[j, l]
  defp kron(z, a, bm, ba, bb) do
    {b, t, _} = Nx.shape(z)
    z = Nx.reshape(z, {b, t, ba, bb})
    z = Nx.dot(z, [2], a, [0])
    z = Nx.dot(z, [2], bm, [0])
    Nx.reshape(z, {b, t, ba * bb})
  end

  defp engram_kv(w, cfg, eidx, quant, t) do
    sites = length(cfg.engram_layers)
    {orders, heads, _} = Config.engram_geometry(cfg)
    n_tables = length(orders) * heads
    {b, _, _} = Nx.shape(eidx)
    d = cfg.d_model
    # a table of order o only has an n-gram from position o - 1 on
    order_of = Nx.tensor(for(o <- orders, _ <- 1..heads, do: o - 1), type: :s32)

    ngram_ok =
      Nx.greater_equal(Nx.iota({t, 1}), Nx.reshape(order_of, {1, n_tables})) |> Nx.as_type(:f32)

    ngram_ok = ngram_ok |> Nx.new_axis(0) |> Nx.new_axis(3)
    dilation = Enum.max(orders)

    for s <- 0..(sites - 1)//1 do
      fetched = Nx.multiply(Nx.take(w["engrams_#{s}/embedding"], eidx), ngram_ok)
      e = aq(Nx.reshape(fetched, {b, t, d}), quant)
      k = Nx.dot(e, [2], w["engrams_#{s}/key_proj/kernel"], [1])
      v = Nx.dot(e, [2], w["engrams_#{s}/value_proj/kernel"], [1])
      taps = w["engrams_#{s}/taps"]

      v =
        Enum.reduce(1..(Config.engram_conv_taps() - 1), Nx.multiply(taps[0], v), fn j, acc ->
          Nx.add(acc, Nx.multiply(taps[j], shift_right(v, j * dilation)))
        end)

      {k, v}
    end
  end

  defp rope(t, head_dim, theta) do
    half = div(head_dim, 2)
    exponents = Nx.divide(Nx.multiply(Nx.iota({half}, type: :f32), 2), head_dim)
    freqs = Nx.divide(1.0, Nx.pow(theta, exponents))
    angles = Nx.multiply(Nx.new_axis(Nx.iota({t}, type: :f32), 1), Nx.new_axis(freqs, 0))
    {Nx.cos(angles), Nx.sin(angles)}
  end

  defp rope_apply(x, cos, sin) do
    half = div(Nx.axis_size(x, -1), 2)
    x1 = Nx.slice_along_axis(x, 0, half, axis: 3)
    x2 = Nx.slice_along_axis(x, half, half, axis: 3)

    Nx.concatenate(
      [
        Nx.subtract(Nx.multiply(x1, cos), Nx.multiply(x2, sin)),
        Nx.add(Nx.multiply(x2, cos), Nx.multiply(x1, sin))
      ],
      axis: -1
    )
  end

  defp causal_mask(t), do: Nx.greater_equal(Nx.iota({t, 1}), Nx.iota({1, t}))

  defp zcrms(x, scale) do
    rms = Nx.sqrt(Nx.add(Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true), 1.0e-6))
    Nx.multiply(Nx.add(1, scale), Nx.divide(x, rms))
  end

  defp rms_unit(x) do
    Nx.multiply(
      x,
      Nx.rsqrt(Nx.add(Nx.mean(Nx.multiply(x, x), axes: [-1], keep_axes: true), 1.0e-6))
    )
  end

  defp aq(x, true), do: CQ.fake_quant_act(x)
  defp aq(x, false), do: x

  # tanh form: exp(-x) overflows for x < -88 and its backward turns 0 * inf into NaN
  defp sigmoid(x), do: Nx.multiply(0.5, Nx.add(1, Nx.tanh(Nx.multiply(0.5, x))))
  defp silu(x), do: Nx.multiply(x, sigmoid(x))

  defp softmax(x) do
    m = Nx.reduce_max(x, axes: [-1], keep_axes: true)
    e = Nx.exp(Nx.subtract(x, m))
    Nx.divide(e, Nx.sum(e, axes: [-1], keep_axes: true))
  end
end
