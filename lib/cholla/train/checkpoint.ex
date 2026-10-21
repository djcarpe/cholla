defmodule Cholla.Train.Checkpoint do
  @moduledoc """
  Loads the float checkpoint a fine-tune starts from (`needle3.safetensors`)
  and the pieces of the base archive that training needs: the CQ codebook,
  the Hadamard permutations and the tokenizer.
  """

  alias Cholla.Train.{Cact, Config, Safetensors, Tokenizer}

  @format_version 2

  defstruct params: %{}, config: nil, run: %{}, path: nil

  @type t :: %__MODULE__{}

  @doc """
  Reads a checkpoint into float32 tensors keyed by their safetensors names,
  dropping the multi-token-prediction extras. Head tensors are kept.
  """
  @spec load(Path.t()) :: {:ok, t()} | {:error, term()}
  def load(path) do
    with {:ok, tensors, metadata} <- Safetensors.read(path, type: {:f, 32}),
         :ok <- check_version(metadata, path) do
      saved = JSON.decode!(metadata["config"] || "{}")

      if Map.has_key?(saved, "attn_dim") and not Map.has_key?(saved, "qk_head_dim") do
        {:error, "#{path} is a Needle 2 checkpoint; Cholla fine-tunes Needle 3"}
      else
        params =
          tensors |> Enum.reject(fn {k, _} -> String.starts_with?(k, "mtp_") end) |> Map.new()

        run = JSON.decode!(metadata["run"] || "{}")
        {:ok, %__MODULE__{params: params, config: Config.from_saved(saved), run: run, path: path}}
      end
    end
  end

  defp check_version(metadata, path) do
    case metadata["format_version"] do
      "2" ->
        :ok

      other ->
        {:error, "#{path} is not a format-v#{@format_version} checkpoint (got #{inspect(other)})"}
    end
  end

  @doc """
  The constants a forward pass and an export share, read from a base
  archive: `%{codebook: [28 floats], cb4: tensor, hadamard: tensor,
  p1: tensor, p2: tensor, tokenizer: Tokenizer.t(), tokenizer_blob: binary}`.
  """
  @spec constants(Path.t()) :: {:ok, map()} | {:error, term()}
  def constants(base_cact) do
    with {:ok, archive} <- Cact.read(base_cact),
         {:ok, blob} <- Cact.tokenizer_blob(base_cact) do
      {p1, p2} = Cact.hada_perms(archive)

      {:ok,
       %{
         codebook: archive.codebook,
         cb4: Nx.tensor(Cact.codebook(archive, 4), type: :f32),
         hadamard: Cholla.Train.CQ.hadamard(128),
         p1: Nx.tensor(p1, type: :s32),
         p2: Nx.tensor(p2, type: :s32),
         tokenizer: Tokenizer.from_blob(blob),
         tokenizer_blob: blob,
         header: archive.header
       }}
    end
  end

  @doc "Keeps the blocks of a `depth`-layer rung; see `Config.ladder_config/2`."
  @spec ladder_slice(t(), pos_integer()) :: t()
  def ladder_slice(%__MODULE__{config: config, params: params} = ckpt, depth)
      when depth == config.num_layers,
      do: %{ckpt | params: params}

  def ladder_slice(%__MODULE__{config: config, params: params} = ckpt, depth) do
    selected = Config.ladder_layer_indices(config, depth)
    idx = Nx.tensor(selected, type: :s32)
    rows = Nx.tensor([0 | Enum.map(selected, &(&1 + 1))], type: :s32)
    kept_sites = for {layer, s} <- Enum.with_index(config.engram_layers), layer in selected, do: s

    params =
      params
      |> Enum.flat_map(fn {name, t} ->
        cond do
          String.starts_with?(name, "stack/") and name != "stack/final_norm/scale" ->
            [{name, Nx.take(t, idx, axis: 0)}]

          String.starts_with?(name, "engrams_") ->
            [prefix, rest] = String.split(name, "/", parts: 2)
            site = prefix |> String.replace_prefix("engrams_", "") |> String.to_integer()

            case Enum.find_index(kept_sites, &(&1 == site)) do
              nil -> []
              new_site -> [{"engrams_#{new_site}/#{rest}", t}]
            end

          String.ends_with?(name, "_head/probes") or String.ends_with?(name, "_head/gain") ->
            [{name, Nx.take(t, rows, axis: 0)}]

          String.ends_with?(name, "_head/row_bias") ->
            [{name, Nx.take(t, rows, axis: 1)}]

          true ->
            [{name, t}]
        end
      end)
      |> Map.new()

    %{ckpt | params: params, config: Config.ladder_config(config, depth)}
  end
end
