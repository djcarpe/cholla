defmodule Cholla.Train.Cact do
  @moduledoc """
  Reads and writes Needle 3 `.cact` archives, the self-contained weights the
  engine loads.

  An archive is a 196-byte header with the architecture geometry, the shared
  CQ codebooks, a nameless tensor directory, then 64-byte aligned blobs in a
  fixed order. Matmul weights are CQ-quantised: per group of 128 inputs, a
  Walsh-Hadamard rotation, a per-group L2 norm in FP16, and one codebook
  index per weight packed LSB-first. Norms, gates and taps stay FP16. The
  layout is documented in the `needle.model.export` module of the
  `cactus-needle` Python package, which this module mirrors.

  `Cholla.Train.Export` lays out a checkpoint's tensors in archive order;
  this module only handles bytes.
  """

  import Bitwise

  @tag 0x05E12A84
  @align 64
  @header_fields 49
  @header_size @header_fields * 4
  @record_size 44
  @codebook_bits [2, 3, 4]

  @fp16 1
  @fp32 2
  @cq 3
  @raw 4

  @type dtype :: :fp16 | :fp32 | :cq | :raw
  @type record :: %{
          dtype: dtype(),
          shape: tuple(),
          offset: non_neg_integer(),
          nbytes: non_neg_integer(),
          group: non_neg_integer(),
          bits: non_neg_integer()
        }

  @doc "The archive tag that opens every `.cact`."
  def tag, do: @tag

  @doc "Reads the header, codebooks and tensor directory of an archive."
  @spec read(Path.t()) :: {:ok, map()} | {:error, term()}
  def read(path) do
    with {:ok, raw} <- File.read(path) do
      parse(raw)
    end
  end

  @doc "Parses an archive held in memory. See `read/1`."
  def parse(<<@tag::little-32, _::binary>> = raw) do
    <<words::binary-size(@header_size), rest::binary>> = raw
    fields = for <<w::little-32 <- words>>, do: w
    [_tag, num_tensors, codebook_len | _] = fields
    header = header_map(fields)

    <<cb::binary-size(codebook_len * 4), dir::binary-size(num_tensors * @record_size), _::binary>> =
      rest

    codebook = for <<f::little-float-32 <- cb>>, do: f
    records = for <<rec::binary-size(@record_size) <- dir>>, do: parse_record(rec)
    {:ok, %{header: header, codebook: codebook, tensors: records, raw: raw}}
  end

  def parse(_), do: {:error, :not_a_cact}

  @doc "The tokenizer blob of an archive: its RAW attachment."
  @spec tokenizer_blob(Path.t()) :: {:ok, binary()} | {:error, term()}
  def tokenizer_blob(path) do
    with {:ok, archive} <- read(path) do
      case Enum.find(archive.tensors, &(&1.dtype == :raw)) do
        nil -> {:error, :no_tokenizer}
        rec -> {:ok, blob(archive, rec)}
      end
    end
  end

  @doc """
  The two Hadamard permutations an archive carries, as lists of indices.

  They are the only FP32 vectors of the model's padded width (`hada_n`),
  written right after the mHC blocks.
  """
  @spec hada_perms(map()) :: {[non_neg_integer()], [non_neg_integer()]}
  def hada_perms(%{header: %{hada_n: n}} = archive) do
    [p1, p2] =
      archive.tensors
      |> Enum.filter(&(&1.dtype == :fp32 and &1.shape == {n}))
      |> Enum.take(2)
      |> Enum.map(fn rec -> for <<f::little-float-32 <- blob(archive, rec)>>, do: trunc(f) end)

    {p1, p2}
  end

  @doc "The codebook of one width (2, 3 or 4 bits) from an archive's shared table."
  @spec codebook(map(), 2 | 3 | 4) :: [float()]
  def codebook(%{codebook: cb}, bits) when bits in @codebook_bits do
    offset =
      @codebook_bits |> Enum.take_while(&(&1 < bits)) |> Enum.map(&(1 <<< &1)) |> Enum.sum()

    Enum.slice(cb, offset, 1 <<< bits)
  end

  @doc "The bytes of one tensor record."
  def blob(%{raw: raw}, %{offset: offset, nbytes: n}), do: binary_part(raw, offset, n)

  @doc "Decodes an FP16 or FP32 tensor record to a list of floats."
  def floats(archive, %{dtype: :fp16} = rec) do
    for <<f::little-float-16 <- blob(archive, rec)>>, do: f
  end

  def floats(archive, %{dtype: :fp32} = rec) do
    for <<f::little-float-32 <- blob(archive, rec)>>, do: f
  end

  @doc """
  Encodes an archive.

  `header` is the geometry map (see `header_map/1` for its keys), `codebook`
  the 28 shared codebook floats, and `tensors` a list of
  `{dtype, shape, blob, group, bits}` in archive order.
  """
  @spec encode(map(), [float()], [{dtype(), tuple(), binary(), integer(), integer()}]) ::
          binary()
  def encode(header, codebook, tensors) do
    hdr = header_binary(header, length(tensors), length(codebook))
    cb = for f <- codebook, into: <<>>, do: <<f::little-float-32>>
    start = byte_size(hdr) + byte_size(cb) + length(tensors) * @record_size

    {placed, _} =
      Enum.map_reduce(tensors, start, fn {_, _, blob, _, _} = t, pos ->
        pos = align(pos)
        {{t, pos}, pos + byte_size(blob)}
      end)

    directory =
      for {{dtype, shape, blob, group, bits}, offset} <- placed, into: <<>> do
        dims = Tuple.to_list(shape)
        padded = (dims ++ [0, 0, 0, 0]) |> Enum.take(4)

        <<dtype_code(dtype)::8, length(dims)::8, 0::16, padded_dims(padded)::binary,
          offset::little-64, byte_size(blob)::little-64, group::little-32, bits::little-32>>
      end

    body =
      Enum.reduce(placed, hdr <> cb <> directory, fn {{_, _, blob, _, _}, offset}, acc ->
        acc <> :binary.copy(<<0>>, offset - byte_size(acc)) <> blob
      end)

    body
  end

  @doc "Writes `encode/3`'s output to `path`."
  def write(path, header, codebook, tensors) do
    File.write(path, encode(header, codebook, tensors))
  end

  @doc false
  def header_map(fields) do
    [
      _tag,
      num_tensors,
      codebook_len,
      kv_window,
      kv_bits,
      vocab,
      out_vocab,
      d_model,
      num_heads,
      num_kv_heads,
      num_layers,
      qk_head_dim,
      v_head_dim,
      max_seq_len,
      hada_n,
      mhc_lanes,
      sliding_window,
      gmask_lo,
      gmask_hi,
      qkv_conv_taps,
      engram_slots,
      engram_sub_dim,
      num_engram_tables,
      engram_conv_taps,
      engram_conv_dilation,
      engram_seed_heads,
      num_orders
      | tail
    ] = fields

    {orders4, tail} = Enum.split(tail, 4)
    [num_sites | tail] = tail
    {sites16, [rope_bits]} = Enum.split(tail, 16)
    <<rope_theta::little-float-32>> = <<rope_bits::little-32>>
    gmask = gmask_lo ||| gmask_hi <<< 32

    %{
      num_tensors: num_tensors,
      codebook_len: codebook_len,
      kv_window: kv_window,
      kv_bits: kv_bits,
      vocab_size: vocab,
      out_vocab: out_vocab,
      d_model: d_model,
      num_heads: num_heads,
      num_kv_heads: num_kv_heads,
      num_layers: num_layers,
      qk_head_dim: qk_head_dim,
      v_head_dim: v_head_dim,
      max_seq_len: max_seq_len,
      hada_n: hada_n,
      mhc_lanes: mhc_lanes,
      sliding_window: sliding_window,
      global_layers: for(i <- 0..(num_layers - 1), (gmask >>> i &&& 1) == 1, do: i),
      qkv_conv_taps: qkv_conv_taps,
      engram_slots: engram_slots,
      engram_sub_dim: engram_sub_dim,
      num_engram_tables: num_engram_tables,
      engram_conv_taps: engram_conv_taps,
      engram_conv_dilation: engram_conv_dilation,
      engram_seed_heads: engram_seed_heads,
      engram_orders: Enum.take(orders4, num_orders),
      engram_layers: Enum.take(sites16, num_sites),
      rope_theta: rope_theta
    }
  end

  defp header_binary(h, num_tensors, codebook_len) do
    gmask = Enum.reduce(h.global_layers, 0, fn l, acc -> acc ||| 1 <<< l end)
    orders = h.engram_orders
    sites = h.engram_layers

    words =
      [
        @tag,
        num_tensors,
        codebook_len,
        h.kv_window,
        h.kv_bits,
        h.vocab_size,
        h.out_vocab,
        h.d_model,
        h.num_heads,
        h.num_kv_heads,
        h.num_layers,
        h.qk_head_dim,
        h.v_head_dim,
        h.max_seq_len,
        h.hada_n,
        h.mhc_lanes,
        h.sliding_window,
        gmask &&& 0xFFFFFFFF,
        gmask >>> 32 &&& 0xFFFFFFFF,
        h.qkv_conv_taps,
        h.engram_slots,
        h.engram_sub_dim,
        h.num_engram_tables,
        h.engram_conv_taps,
        h.engram_conv_dilation,
        h.engram_seed_heads,
        length(orders)
      ] ++
        pad_list(orders, 4) ++ [length(sites)] ++ pad_list(sites, 16)

    48 = length(words)
    for(w <- words, into: <<>>, do: <<w::little-32>>) <> <<h.rope_theta::little-float-32>>
  end

  defp pad_list(list, n), do: Enum.take(list ++ List.duplicate(0, n), n)
  defp padded_dims(dims), do: for(d <- dims, into: <<>>, do: <<d::little-32>>)

  defp parse_record(
         <<dtype::8, ndim::8, _::16, d0::little-32, d1::little-32, d2::little-32, d3::little-32,
           offset::little-64, nbytes::little-64, group::little-32, bits::little-32>>
       ) do
    %{
      dtype: dtype_atom(dtype),
      shape: [d0, d1, d2, d3] |> Enum.take(ndim) |> List.to_tuple(),
      offset: offset,
      nbytes: nbytes,
      group: group,
      bits: bits
    }
  end

  defp dtype_atom(@fp16), do: :fp16
  defp dtype_atom(@fp32), do: :fp32
  defp dtype_atom(@cq), do: :cq
  defp dtype_atom(@raw), do: :raw

  defp dtype_code(:fp16), do: @fp16
  defp dtype_code(:fp32), do: @fp32
  defp dtype_code(:cq), do: @cq
  defp dtype_code(:raw), do: @raw

  defp align(n), do: div(n + @align - 1, @align) * @align
end
