defmodule Cholla.Download do
  @moduledoc """
  Fetches and caches the Cactus engine and weights from Hugging Face.

  The engine is the self-contained `libcactus` shared library of Cactus 4,
  published in the `cactus_runtime` Python wheels of `Cactus-Compute/cactus`
  for each platform (`engine_version: "3.3.3"` still fetches the older
  `libneedle3` of the `cactus-needle` wheels). Weights are the base
  `needle3.cact` text model, the optional `whistle.cact` speech model and
  the optional `mirage.cact` vision model (Cactus 4 only).

  Everything is cached under `cache_dir/0`. Each path can be overridden:

      config :cholla,
        engine_path: "/opt/needle/libneedle.so",
        weights_path: "/opt/needle/needle3.cact",
        speech_weights_path: "/opt/needle/whistle.cact",
        vision_weights_path: "/opt/needle/mirage.cact"

  The environment variables `CHOLLA_ENGINE_PATH` (or `NEEDLE3_LIB_PATH`),
  `CHOLLA_WEIGHTS_PATH`, `CHOLLA_SPEECH_WEIGHTS_PATH` and
  `CHOLLA_VISION_WEIGHTS_PATH` take precedence over
  the application config.
  """

  require Logger

  @hub "https://huggingface.co"
  @engine_repo "Cactus-Compute/needle3"
  @speech_repo "Cactus-Compute/whistle"
  @vision_repo "Cactus-Compute/mirage"
  @cactus_repo "Cactus-Compute/cactus"
  @default_engine_version "4.1.1"

  @doc "The engine release to download (`config :cholla, engine_version: ...`)."
  def engine_version, do: Application.get_env(:cholla, :engine_version, @default_engine_version)

  @doc """
  Cache root: `config :cholla, cache_dir: ...`, else `$XDG_CACHE_HOME/cholla`,
  else `~/.cache/cholla`.
  """
  def cache_dir do
    Application.get_env(:cholla, :cache_dir) ||
      Path.join(System.get_env("XDG_CACHE_HOME") || Path.expand("~/.cache"), "cholla")
  end

  @doc "Path to the engine library, downloading it if needed."
  @spec engine_path() :: {:ok, Path.t()} | {:error, term()}
  def engine_path do
    override(["CHOLLA_ENGINE_PATH", "NEEDLE3_LIB_PATH"], :engine_path) || fetch_engine()
  end

  @doc "Path to the base text weights (`needle3.cact`), downloading them if needed."
  @spec weights_path() :: {:ok, Path.t()} | {:error, term()}
  def weights_path do
    override(["CHOLLA_WEIGHTS_PATH"], :weights_path) ||
      fetch_file(@engine_repo, "needle3.cact", Path.join(cache_dir(), "weights"))
  end

  @doc """
  Path to the float checkpoint fine-tuning starts from
  (`checkpoints/needle3.safetensors`, 242 MB), downloading it if needed.
  """
  @spec checkpoint_path() :: {:ok, Path.t()} | {:error, term()}
  def checkpoint_path do
    override(["CHOLLA_CHECKPOINT_PATH"], :checkpoint_path) ||
      fetch_file(
        @engine_repo,
        "checkpoints/needle3.safetensors",
        Path.join(cache_dir(), "weights")
      )
  end

  @doc "Path to the speech weights (`whistle.cact`), downloading them if needed."
  @spec speech_weights_path() :: {:ok, Path.t()} | {:error, term()}
  def speech_weights_path do
    override(["CHOLLA_SPEECH_WEIGHTS_PATH"], :speech_weights_path) ||
      fetch_file(@speech_repo, "whistle.cact", Path.join(cache_dir(), "weights"))
  end

  @doc "Path to the vision weights (`mirage.cact`), downloading them if needed."
  @spec vision_weights_path() :: {:ok, Path.t()} | {:error, term()}
  def vision_weights_path do
    override(["CHOLLA_VISION_WEIGHTS_PATH"], :vision_weights_path) ||
      fetch_file(@vision_repo, "mirage.cact", Path.join(cache_dir(), "weights"))
  end

  defp override(envs, key) do
    case Enum.find_value(envs, &System.get_env/1) || Application.get_env(:cholla, key) do
      nil ->
        nil

      path ->
        path = Path.expand(path)
        if File.exists?(path), do: {:ok, path}, else: {:error, {:enoent, path}}
    end
  end

  @doc """
  The wheel platform tag for this machine, e.g. `"manylinux2014_x86_64"`.
  """
  @spec platform_tag() :: {:ok, String.t()} | {:error, term()}
  def platform_tag do
    arch = :erlang.system_info(:system_architecture) |> List.to_string()

    machine =
      cond do
        String.starts_with?(arch, ["x86_64", "amd64"]) -> "x86_64"
        String.starts_with?(arch, ["aarch64", "arm64"]) -> "aarch64"
        true -> nil
      end

    case {:os.type(), machine} do
      {_, nil} ->
        {:error, "no Cactus engine library is published for #{arch}; set CHOLLA_ENGINE_PATH"}

      {{:unix, :darwin}, "aarch64"} ->
        {:ok, "macosx_11_0_arm64"}

      {{:unix, :darwin}, "x86_64"} ->
        {:ok, "macosx_11_0_x86_64"}

      {{:unix, :linux}, machine} ->
        family = if musl?(arch), do: "musllinux_1_2_", else: "manylinux2014_"
        {:ok, family <> machine}

      {os, _} ->
        {:error, "Cholla does not support #{inspect(os)}; the NIF loads the engine with dlopen"}
    end
  end

  defp musl?(arch) do
    String.contains?(arch, "musl") or
      (Path.wildcard("/lib/ld-musl-*.so.1") != [] and Path.wildcard("/lib*/libc.so.6") == [] and
         Path.wildcard("/lib/*-linux-gnu/libc.so.6") == [] and
         Path.wildcard("/usr/lib/libc.so.6") == [])
  end

  # Needle 3's engine is libneedle3 in the cactus-needle wheels of
  # Cactus-Compute/needle3; Cactus 4's is libcactus in the cactus_runtime
  # wheels of Cactus-Compute/cactus.
  defp needle3?(version), do: String.starts_with?(version, "3.")

  defp lib_member(version, tag) do
    {dir, name} =
      if needle3?(version), do: {"needle", "libneedle3"}, else: {"cactus", "libcactus"}

    ext = if String.starts_with?(tag, "macosx"), do: ".dylib", else: ".so"
    "#{dir}/#{name}#{ext}"
  end

  defp wheel(version, tag) do
    if needle3?(version),
      do: {@engine_repo, "python/cactus_needle-#{version}-py3-none-#{tag}.whl"},
      else:
        {@cactus_repo, "releases/#{version}/python/cactus_runtime-#{version}-py3-none-#{tag}.whl"}
  end

  defp fetch_engine do
    with {:ok, tag} <- platform_tag() do
      version = engine_version()
      member = lib_member(version, tag)
      dir = Path.join([cache_dir(), "engine", version, tag])
      target = Path.join(dir, Path.basename(member))

      if File.exists?(target) do
        {:ok, target}
      else
        {repo, wheel} = wheel(version, tag)

        with {:ok, wheel_path} <- download(repo, wheel, Path.join(dir, "engine.whl")),
             {:ok, data} <- unzip_member(wheel_path, member),
             :ok <- File.mkdir_p(dir),
             :ok <- atomic_write(target, data) do
          File.rm(wheel_path)
          {:ok, target}
        end
      end
    end
  end

  defp unzip_member(zip, member) do
    case :zip.extract(String.to_charlist(zip), [
           :memory,
           {:file_list, [String.to_charlist(member)]}
         ]) do
      {:ok, [{_, data}]} -> {:ok, data}
      {:ok, []} -> {:error, "#{member} not found in #{zip}"}
      {:error, reason} -> {:error, {:unzip, reason}}
    end
  end

  defp fetch_file(repo, file, dir) do
    target = Path.join(dir, Path.basename(file))
    if File.exists?(target), do: {:ok, target}, else: download(repo, file, target)
  end

  @doc """
  Downloads `file` from the Hugging Face model `repo` to `target`.
  Set `HF_TOKEN` for gated or private repositories.
  """
  @spec download(String.t(), String.t(), Path.t()) :: {:ok, Path.t()} | {:error, term()}
  def download(repo, file, target) do
    url = "#{@hub}/#{repo}/resolve/main/#{file}"
    tmp = target <> ".part"
    File.mkdir_p!(Path.dirname(target))
    File.rm(tmp)
    Logger.info("cholla: downloading #{url}")

    headers =
      case System.get_env("HF_TOKEN") do
        nil -> []
        token -> [{~c"authorization", String.to_charlist("Bearer " <> token)}]
      end

    http_opts = [
      ssl: [
        verify: :verify_peer,
        cacerts: :public_key.cacerts_get(),
        customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]
      ],
      autoredirect: true,
      timeout: :timer.minutes(10),
      connect_timeout: :timer.seconds(30)
    ]

    case :httpc.request(:get, {String.to_charlist(url), headers}, http_opts,
           stream: String.to_charlist(tmp)
         ) do
      {:ok, :saved_to_file} ->
        File.rename(tmp, target)
        |> then(fn
          :ok -> {:ok, target}
          err -> err
        end)

      {:ok, {{_, status, _}, _, _}} ->
        File.rm(tmp)
        {:error, {:http, status, url}}

      {:error, reason} ->
        File.rm(tmp)
        {:error, {:http, reason, url}}
    end
  end

  defp atomic_write(target, data) do
    tmp = target <> ".part"

    with :ok <- File.write(tmp, data) do
      File.rename(tmp, target)
    end
  end
end
