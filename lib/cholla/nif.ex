defmodule Cholla.NIF do
  @moduledoc """
  Raw bindings to the Needle C API (`needle.h`).

  Every function here maps one-to-one onto an engine call on an instance
  opened with `open_engine/3`. An instance holds one model per kind and one
  conversation, so prefer the pooled API in `Cholla` and `Cholla.Speech`;
  calling these directly from several processes is memory-safe (each instance
  has a mutex) but interleaves conversation state.

  Conventions:

    * strings are binaries or iodata, and optional ones may be `nil`
    * PCM is a binary of native-endian 32-bit floats, 16 kHz mono, in [-1, 1]
    * results are `{:ok, value}` or `{:error, reason}`, where `reason` is the
      engine's `needle_last_error()` text or `{:buffer_too_small, bytes}`
  """

  @on_load :load_nif
  @compile {:autoload, false}

  @text 1
  @speech 2

  @doc "Bit set in `models/0` when a text model is loaded (`NEEDLE_TEXT`)."
  def text_model, do: @text

  @doc "Bit set in `models/0` when a speech model is loaded (`NEEDLE_SPEECH`)."
  def speech_model, do: @speech

  @doc false
  def load_nif do
    path = :filename.join(:code.priv_dir(:cholla), ~c"cholla_nif")
    :erlang.load_nif(path, 0)
  end

  @typedoc "An engine instance: a private copy of the engine library with its own state."
  @type instance :: reference()

  @doc """
  Opens an engine instance from the library at `path`. Each instance dlopens
  its own copy of the library, so instances share no state.

  `threads` sizes the instance's thread pool, counting the calling thread
  (0 keeps the engine's default of every core); `cpus` restricts it to the
  given CPU numbers, or is `nil`. Both apply on Linux only.
  """
  @spec open_engine(Path.t(), non_neg_integer(), [non_neg_integer()] | nil) ::
          {:ok, instance()} | {:error, binary()}
  def open_engine(_path, _threads, _cpus), do: :erlang.nif_error(:nif_not_loaded)

  @doc """
  The instance's library path, `:threads` (`nil` for the default), `:cpus`,
  whether `:thread_control` is supported, and the VM-wide `:vm_threads` and
  `:instances` totals.
  """
  @spec info(instance()) :: map()
  def info(_instance), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_load`: loads a `.cact` archive. Returns the `models/1` mask."
  @spec load(instance(), binary()) :: {:ok, non_neg_integer()} | {:error, term()}
  def load(_instance, _cact), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_models`: `NEEDLE_TEXT` and/or `NEEDLE_SPEECH` bits."
  @spec models(instance()) :: non_neg_integer()
  def models(_instance), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_init`: sets the system prompt and tools; returns the static prefix token count."
  @spec init(instance(), iodata(), iodata(), iodata() | nil) ::
          {:ok, non_neg_integer()} | {:error, term()}
  def init(_instance, _system, _tools_json, _tool_index_path),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_complete`: exactly one of `text` and `pcm` is non-nil. Returns the JSON envelope."
  @spec complete(instance(), iodata() | nil, binary() | nil, pos_integer(), pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def complete(_instance, _text, _pcm, _max_new_tokens, _capacity),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_set_audio`: transcription settings used by `complete/5` with PCM."
  @spec set_audio(instance(), iodata() | nil, iodata() | nil, boolean()) :: :ok
  def set_audio(_instance, _language, _keywords, _word_timestamps),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_reset`: clears the conversation."
  @spec reset(instance()) :: :ok
  def reset(_instance), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_transcribe`: returns the JSON transcript."
  @spec transcribe(instance(), binary(), iodata() | nil, iodata() | nil, boolean(), pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def transcribe(_instance, _pcm, _language, _keywords, _word_timestamps, _capacity),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_stream_transcribe_process`: feeds live audio; returns committed words as JSON."
  @spec stream_process(instance(), binary(), iodata() | nil, iodata() | nil, pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def stream_process(_instance, _pcm, _language, _keywords, _capacity),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_stream_transcribe_stop`: ends the live stream; returns the tail as JSON."
  @spec stream_stop(instance(), pos_integer()) :: {:ok, binary()} | {:error, term()}
  def stream_stop(_instance, _capacity), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_embed`: exactly one of `text` and `pcm` is non-nil. Returns native f32 floats."
  @spec embed(instance(), iodata() | nil, binary() | nil) :: {:ok, binary()} | {:error, term()}
  def embed(_instance, _text, _pcm), do: :erlang.nif_error(:nif_not_loaded)
end
