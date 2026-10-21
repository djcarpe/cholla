defmodule Cholla.Train.Safetensors do
  @moduledoc """
  Reads and writes `.safetensors` files: a little-endian u64 header length,
  a JSON header mapping tensor names to dtype, shape and byte offsets, then
  the tensor bytes. Tensors come back as `Nx` tensors; the `__metadata__`
  entry, when present, as a map of strings.
  """

  @dtypes %{
    "F16" => {:f, 16},
    "BF16" => {:bf, 16},
    "F32" => {:f, 32},
    "F64" => {:f, 64},
    "I32" => {:s, 32},
    "I64" => {:s, 64},
    "U8" => {:u, 8}
  }

  @doc """
  Reads a file. Returns `{:ok, tensors, metadata}` with the tensors keyed by
  name. Pass `type: {:f, 32}` to convert every tensor on load, or
  `only: ["name", ...]` to read a subset.
  """
  @spec read(Path.t(), keyword()) ::
          {:ok, %{String.t() => Nx.Tensor.t()}, map()} | {:error, term()}
  def read(path, opts \\ []) do
    with {:ok, data} <- File.read(path) do
      {:ok, header, body} = split(data)
      {metadata, entries} = Map.pop(header, "__metadata__", %{})
      only = opts[:only]

      tensors =
        for {name, %{"dtype" => dtype, "shape" => shape, "data_offsets" => [from, to]}} <- entries,
            only == nil or name in only,
            into: %{} do
          type = Map.fetch!(@dtypes, dtype)
          bytes = binary_part(body, from, to - from)
          tensor = Nx.from_binary(bytes, type) |> Nx.reshape(List.to_tuple(shape))
          tensor = if opts[:type], do: Nx.as_type(tensor, opts[:type]), else: tensor
          {name, tensor}
        end

      {:ok, tensors, metadata}
    end
  end

  @doc "Reads only the header: `{:ok, entries, metadata}`."
  def read_header(path) do
    with {:ok, file} <- File.open(path, [:read, :binary]),
         <<n::little-64>> <- IO.binread(file, 8),
         json when is_binary(json) <- IO.binread(file, n) do
      File.close(file)
      header = JSON.decode!(json)
      {metadata, entries} = Map.pop(header, "__metadata__", %{})
      {:ok, entries, metadata}
    end
  end

  defp split(<<n::little-64, json::binary-size(n), body::binary>>) do
    {:ok, JSON.decode!(json), body}
  end

  @doc """
  Writes `tensors` (name to `Nx` tensor) with string `metadata`. Tensors are
  written in their own type; pass `type:` to convert first.
  """
  @spec write(Path.t(), %{String.t() => Nx.Tensor.t()}, map(), keyword()) ::
          :ok | {:error, term()}
  def write(path, tensors, metadata \\ %{}, opts \\ []) do
    File.write(path, encode(tensors, metadata, opts))
  end

  @doc false
  def encode(tensors, metadata, opts \\ []) do
    names = tensors |> Map.keys() |> Enum.sort()

    {entries, blobs, _} =
      Enum.reduce(names, {[], [], 0}, fn name, {entries, blobs, pos} ->
        t = tensors[name]
        t = if opts[:type], do: Nx.as_type(t, opts[:type]), else: t
        bytes = Nx.to_binary(t)
        n = byte_size(bytes)

        entry =
          {name,
           %{
             "dtype" => dtype_name(Nx.type(t)),
             "shape" => Tuple.to_list(Nx.shape(t)),
             "data_offsets" => [pos, pos + n]
           }}

        {[entry | entries], [bytes | blobs], pos + n}
      end)

    header = Map.new(entries)

    header =
      if map_size(metadata) > 0, do: Map.put(header, "__metadata__", metadata), else: header

    json = JSON.encode!(header)
    pad = rem(8 - rem(byte_size(json), 8), 8)
    json = json <> String.duplicate(" ", pad)

    IO.iodata_to_binary([<<byte_size(json)::little-64>>, json, Enum.reverse(blobs)])
  end

  defp dtype_name(type) do
    case Enum.find(@dtypes, fn {_, t} -> t == type end) do
      {name, _} -> name
      nil -> raise ArgumentError, "cannot write #{inspect(type)} tensors to safetensors"
    end
  end
end
