defmodule Cholla.Train.Tokenizer do
  @moduledoc """
  The SentencePiece BPE tokenizer packaged in a `.cact` archive.

  The archive carries the vocabulary as a self-contained dump (pieces with
  scores and types), and the engine encodes with the reference algorithm this
  module ports: spaces become `▁`, a dummy prefix is added, chat markers such
  as `<tool_call>` are matched whole, the rest is merged pair by pair taking
  the highest-scoring piece each time, and anything left unmatched falls back
  to byte pieces.

  The special ids are fixed: pad 0, eos 1, bos 2, unk 3.
  """

  defstruct pieces: %{},
            ids: %{},
            scores: %{},
            types: %{},
            byte_ids: %{},
            markers: [],
            add_dummy_prefix: true,
            byte_fallback: true,
            unk_id: 3,
            size: 0

  @type t :: %__MODULE__{}

  @pad_id 0
  @eos_id 1
  @bos_id 2
  @unk_id 3

  @meta_space "▁"

  @tk_normal 0
  @tk_unknown 1
  @tk_control 2
  @tk_user_defined 3
  @tk_byte 4

  def pad_id, do: @pad_id
  def eos_id, do: @eos_id
  def bos_id, do: @bos_id
  def unk_id, do: @unk_id

  @doc "Loads the tokenizer packaged in a `.cact` archive."
  @spec from_cact(Path.t()) :: {:ok, t()} | {:error, term()}
  def from_cact(path) do
    with {:ok, blob} <- Cholla.Train.Cact.tokenizer_blob(path) do
      {:ok, from_blob(blob)}
    end
  end

  @doc "Parses the RAW tokenizer blob of an archive."
  @spec from_blob(binary()) :: t()
  def from_blob(
        <<n::little-32, _pad::little-32, _eos::little-32, _bos::little-32, unk::little-32,
          dummy::8, byte_fallback::8, _::16, rest::binary>>
      ) do
    {records, _} =
      Enum.map_reduce(0..(n - 1)//1, rest, fn id, acc ->
        <<score::little-float-32, type::8, len::little-16, piece::binary-size(len), acc::binary>> =
          acc

        {{id, piece, score, type}, acc}
      end)

    pieces = Map.new(records, fn {id, piece, _, _} -> {id, piece} end)
    ids = Map.new(records, fn {id, piece, _, _} -> {piece, id} end)
    scores = Map.new(records, fn {id, _, score, _} -> {id, score} end)
    types = Map.new(records, fn {id, _, _, type} -> {id, type} end)

    byte_ids =
      for {id, <<"<0x", hex::binary-size(2), ">">>, _, @tk_byte} <- records, into: %{} do
        {String.to_integer(hex, 16), id}
      end

    markers =
      for {_, piece, _, @tk_user_defined} <- records, do: piece

    %__MODULE__{
      pieces: pieces,
      ids: ids,
      scores: scores,
      types: types,
      byte_ids: byte_ids,
      markers: Enum.sort_by(markers, &(-byte_size(&1))),
      add_dummy_prefix: dummy == 1,
      byte_fallback: byte_fallback == 1,
      unk_id: unk,
      size: n
    }
  end

  @doc "Encodes text to piece ids (no bos/eos)."
  @spec encode(t(), String.t()) :: [non_neg_integer()]
  def encode(_tok, ""), do: []

  def encode(%__MODULE__{} = tok, text) when is_binary(text) do
    escaped = String.replace(text, " ", @meta_space)
    escaped = if tok.add_dummy_prefix, do: @meta_space <> escaped, else: escaped
    segments(tok, escaped, [], [])
  end

  # Splits around chat markers; each run of plain text is one BPE segment.
  defp segments(tok, "", buf, acc), do: Enum.reverse(acc) ++ bpe(tok, flush(buf))

  defp segments(tok, text, buf, acc) do
    case Enum.find(tok.markers, &String.starts_with?(text, &1)) do
      nil ->
        {grapheme, rest} = String.next_grapheme(text)
        segments(tok, rest, [grapheme | buf], acc)

      marker ->
        rest = binary_part(text, byte_size(marker), byte_size(text) - byte_size(marker))
        ids = Enum.reverse(acc) |> Enum.concat(bpe(tok, flush(buf))) |> Enum.reverse()
        segments(tok, rest, [], [tok.ids[marker] | ids])
    end
  end

  defp flush(buf), do: buf |> Enum.reverse() |> IO.iodata_to_binary()

  defp bpe(_tok, ""), do: []

  defp bpe(tok, segment) do
    symbols = String.codepoints(segment) |> merge(tok)

    Enum.flat_map(symbols, fn sym ->
      case tok.ids[sym] do
        nil when tok.byte_fallback -> for <<b <- sym>>, do: tok.byte_ids[b]
        nil -> [tok.unk_id]
        id -> [id]
      end
    end)
  end

  # Repeatedly merges the adjacent pair whose joined piece scores highest,
  # earliest position on ties, until no adjacent pair is a piece.
  defp merge(symbols, tok) do
    case best_pair(symbols, tok) do
      nil ->
        symbols

      j ->
        symbols
        |> List.update_at(j, &(&1 <> Enum.at(symbols, j + 1)))
        |> List.delete_at(j + 1)
        |> merge(tok)
    end
  end

  defp best_pair(symbols, tok) do
    symbols
    |> Enum.chunk_every(2, 1, :discard)
    |> Enum.with_index()
    |> Enum.reduce({nil, nil}, fn {[a, b], j}, {best_score, best_j} ->
      case tok.ids[a <> b] do
        nil ->
          {best_score, best_j}

        id ->
          score = tok.scores[id]
          if best_score == nil or score > best_score, do: {score, j}, else: {best_score, best_j}
      end
    end)
    |> elem(1)
  end

  @doc "Decodes piece ids to text, dropping control and unknown pieces."
  @spec decode(t(), [non_neg_integer()]) :: String.t()
  def decode(%__MODULE__{} = tok, ids) do
    bytes =
      Enum.reduce(ids, [], fn id, acc ->
        case tok.types[id] do
          @tk_byte ->
            <<"<0x", hex::binary-size(2), ">">> = tok.pieces[id]
            [acc, String.to_integer(hex, 16)]

          t when t in [@tk_control, @tk_unknown] ->
            acc

          _ ->
            [acc, tok.pieces[id]]
        end
      end)
      |> IO.iodata_to_binary()

    text =
      bytes
      |> :unicode.characters_to_binary(:utf8, :utf8)
      |> scrub()
      |> String.replace(@meta_space, " ")

    if tok.add_dummy_prefix and String.starts_with?(text, " "),
      do: binary_part(text, 1, byte_size(text) - 1),
      else: text
  end

  defp scrub({:error, good, _}), do: good <> "�"
  defp scrub({:incomplete, good, _}), do: good <> "�"
  defp scrub(text), do: text

  @doc "The id of a piece, or nil."
  def piece_id(%__MODULE__{ids: ids}, piece), do: ids[piece]

  @doc "Normal pieces for the archive's piece-type codes."
  def type_codes,
    do: %{
      normal: @tk_normal,
      unknown: @tk_unknown,
      control: @tk_control,
      user_defined: @tk_user_defined,
      byte: @tk_byte
    }
end
