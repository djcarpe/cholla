defmodule Cholla.Train.Export do
  @moduledoc """
  Lays a checkpoint out as a `.cact` archive: W4A8, the format `needle build`
  writes. Projections are transposed to `[out, in]` and CQ-quantised in
  groups of 128 along the input; norms, taps, gates and the Hadamard MLP's
  diagonals and factors stay FP16; the two Hadamard permutations are FP32;
  the tokenizer rides along as a RAW attachment.
  """

  alias Cholla.Train.{Cact, Config, CQ}

  @head_bits 4
  @bits 4
  @group 128
  @hadamard ~w(d1 d2 b2 d3 d4 w1a w1b w2a w2b w3a w3b cond_v cond_u)
  @mhc_fp16 ~w(mhc_a_pre mhc_a_post mhc_a_res mhc_b_pre mhc_b_post mhc_b_res)
  @mhc_phi ~w(mhc_phi_pre mhc_phi_post mhc_phi_res)
  # heads in canonical order: embedding 1, confidence 2, router 3
  @heads [{"embedding_head", 1}, {"confidence_head", 2}, {"router_head", 3}]

  @doc """
  Writes `params` (float tensors keyed by checkpoint name) and `config` to
  `path`. `consts` comes from `Cholla.Train.Checkpoint.constants/1`.
  Returns `{:ok, %{path, bytes, tensors}}`.
  """
  def write(path, params, %Config{} = config, consts) do
    tensors = tensors(params, config, consts) ++ [{:raw, {}, consts.tokenizer_blob, 0, 0}]
    header = header(config)
    binary = Cact.encode(header, consts.codebook, tensors)

    with :ok <- File.write(path, binary) do
      {:ok, %{path: path, bytes: byte_size(binary), tensors: length(tensors)}}
    end
  end

  @doc "The archive header for a config."
  def header(%Config{} = c) do
    {orders, heads, sub_dim} = Config.engram_geometry(c)
    if c.num_layers > 64, do: raise(ArgumentError, "cact global_mask holds 64 layers")
    if length(c.engram_layers) > 16, do: raise(ArgumentError, "cact header holds 16 engram sites")

    %{
      kv_window: Config.effective_kv_window(c),
      kv_bits: c.kv_bits,
      vocab_size: c.vocab_size,
      out_vocab: c.out_vocab,
      d_model: c.d_model,
      num_heads: c.num_heads,
      num_kv_heads: c.num_kv_heads,
      num_layers: c.num_layers,
      qk_head_dim: c.qk_head_dim,
      v_head_dim: c.v_head_dim,
      max_seq_len: c.max_seq_len,
      hada_n: Config.hada_n(c),
      mhc_lanes: c.mhc_lanes,
      sliding_window: c.sliding_window,
      global_layers: c.global_layers,
      qkv_conv_taps: c.qkv_conv_taps,
      engram_slots: c.engram_slots,
      engram_sub_dim: sub_dim,
      num_engram_tables: length(orders) * heads,
      engram_conv_taps: Config.engram_conv_taps(),
      engram_conv_dilation: Enum.max(orders),
      engram_seed_heads: c.engram_seed_heads,
      engram_orders: orders,
      engram_layers: c.engram_layers,
      rope_theta: c.rope_theta
    }
  end

  @doc "The model tensors in archive order, as `{dtype, shape, blob, group, bits}`."
  def tensors(params, %Config{} = c, consts) do
    q = fn t, bits -> cq(t, consts, bits) end
    sa = "stack/layers/block/self_attn/"
    blk = "stack/layers/block/"
    hm = "stack/layers/block/hadamard_mlp/"

    layers =
      Enum.flat_map(0..(c.num_layers - 1)//1, fn i ->
        taps =
          if c.qkv_conv_taps > 0,
            do: for(n <- ~w(q_taps k_taps v_taps), do: fp16(params[sa <> n][i])),
            else: []

        [
          fp16(params[blk <> "ZCRMSNorm_0/scale"][i]),
          q.(Nx.transpose(params[sa <> "q_proj/kernel"][i]), @bits),
          q.(Nx.transpose(params[sa <> "k_proj/kernel"][i]), @bits),
          q.(Nx.transpose(params[sa <> "v_proj/kernel"][i]), @bits)
        ] ++
          taps ++
          [
            fp16(params[sa <> "q_norm/scale"][i]),
            fp16(params[sa <> "k_norm/scale"][i]),
            q.(Nx.transpose(params[sa <> "gate_proj/kernel"][i]), @bits),
            q.(Nx.transpose(params[sa <> "out_proj/kernel"][i]), @bits),
            fp16(params[blk <> "post_attn_norm/scale"][i]),
            fp16(Nx.reshape(params[blk <> "attn_gate"][i], {1})),
            fp16(params[blk <> "pre_hada_norm/scale"][i])
          ] ++
          for(n <- @hadamard, do: fp16(params[hm <> n][i]))
      end)

    mhc =
      for(n <- @mhc_fp16, do: fp16(params["stack/" <> n])) ++
        for n <- @mhc_phi do
          phi = params["stack/" <> n]
          {l, n_c, lanes} = Nx.shape(phi)
          q.(phi |> Nx.transpose(axes: [0, 2, 1]) |> Nx.reshape({l * lanes, n_c}), @bits)
        end

    hada_n = Config.hada_n(c)

    perms = [
      fp32(Nx.as_type(consts.p1, :f32), {hada_n}),
      fp32(Nx.as_type(consts.p2, :f32), {hada_n})
    ]

    {orders, heads, sub_dim} = Config.engram_geometry(c)
    num_tables = length(orders) * heads

    engrams =
      Enum.flat_map(0..(length(c.engram_layers) - 1)//1, fn s ->
        tables =
          Nx.reshape(params["engrams_#{s}/embedding"], {num_tables * c.engram_slots, sub_dim})

        [
          q.(tables, @bits),
          q.(Nx.transpose(params["engrams_#{s}/key_proj/kernel"]), @bits),
          q.(Nx.transpose(params["engrams_#{s}/value_proj/kernel"]), @bits),
          fp16(params["engrams_#{s}/taps"])
        ]
      end)

    [q.(params["embedding/embedding"], @bits)] ++
      layers ++
      mhc ++
      perms ++ engrams ++ [fp16(params["stack/final_norm/scale"])] ++ head_tensors(params, consts)
  end

  defp head_tensors(params, consts) do
    present = for {key, code} <- @heads, Map.has_key?(params, key <> "/probes"), do: {key, code}

    if present == [] do
      []
    else
      manifest = fp16(Nx.tensor(Enum.map(present, &elem(&1, 1)), type: :f32))

      per_head =
        Enum.flat_map(present, fn {key, _} ->
          probes = params[key <> "/probes"]
          {_, _, d} = Nx.shape(probes)
          kernel = Nx.transpose(params[key <> "/proj/kernel"])
          {out, _} = Nx.shape(kernel)
          bias = params[key <> "/proj/bias"] || Nx.broadcast(0.0, {out})

          [
            cq(Nx.reshape(probes, {:auto, d}), consts, @head_bits),
            fp16(params[key <> "/gain"]),
            cq(params[key <> "/query"], consts, @head_bits),
            fp16(params[key <> "/row_bias"]),
            cq(kernel, consts, @head_bits),
            fp16(bias)
          ] ++
            if key == "router_head",
              do: [fp16(params[key <> "/calibration"] || Nx.tensor([0.9, 0.0, 0.6]))],
              else: []
        end)

      [manifest | per_head]
    end
  end

  defp cq(t, consts, bits) when bits == @bits do
    {idx, norm} = CQ.encode(t, consts.cb4, consts.hadamard)
    {:cq, Nx.shape(t), CQ.pack4(idx, norm), @group, @bits}
  end

  defp fp16(t), do: {:fp16, Nx.shape(t), Nx.to_binary(Nx.as_type(t, :f16)), 0, 0}
  defp fp32(t, shape), do: {:fp32, shape, Nx.to_binary(Nx.as_type(t, :f32)), 0, 0}

  @doc "Removes a head from the params (local tuning leaves it untrained)."
  def drop_head(params, key) do
    params |> Enum.reject(fn {name, _} -> String.starts_with?(name, key <> "/") end) |> Map.new()
  end
end
