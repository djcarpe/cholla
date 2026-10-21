defmodule Cholla.NIF do
  @moduledoc """
  Raw bindings to the Needle C API (`needle.h`).

  Every function here maps one-to-one onto an engine call. The engine holds a
  single process-global model per kind, so prefer the serialised API in
  `Cholla` and `Cholla.Speech`; calling these directly from several processes
  is memory-safe (the NIF holds a mutex) but interleaves conversation state.

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

  @doc "dlopens the engine shared library. Only one engine can be open per VM."
  @spec open_engine(Path.t()) :: :ok | {:error, {:already_open, binary()} | binary()}
  def open_engine(_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc "Path of the open engine, or `nil`."
  @spec engine_path() :: {:ok, binary()} | nil
  def engine_path, do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_load`: loads a `.cact` archive. Returns the `models/0` mask."
  @spec load(binary()) :: {:ok, non_neg_integer()} | {:error, term()}
  def load(_cact), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_models`: `NEEDLE_TEXT` and/or `NEEDLE_SPEECH` bits."
  @spec models() :: non_neg_integer()
  def models, do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_init`: sets the system prompt and tools; returns the static prefix token count."
  @spec init(iodata(), iodata(), iodata() | nil) :: {:ok, non_neg_integer()} | {:error, term()}
  def init(_system, _tools_json, _tool_index_path), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_complete`: exactly one of `text` and `pcm` is non-nil. Returns the JSON envelope."
  @spec complete(iodata() | nil, binary() | nil, pos_integer(), pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def complete(_text, _pcm, _max_new_tokens, _capacity), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_set_audio`: transcription settings used by `complete/4` with PCM."
  @spec set_audio(iodata() | nil, iodata() | nil, boolean()) :: :ok | {:error, term()}
  def set_audio(_language, _keywords, _word_timestamps), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_reset`: clears the conversation."
  @spec reset() :: :ok | {:error, term()}
  def reset, do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_transcribe`: returns the JSON transcript."
  @spec transcribe(binary(), iodata() | nil, iodata() | nil, boolean(), pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def transcribe(_pcm, _language, _keywords, _word_timestamps, _capacity),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_stream_transcribe_process`: feeds live audio; returns committed words as JSON."
  @spec stream_process(binary(), iodata() | nil, iodata() | nil, pos_integer()) ::
          {:ok, binary()} | {:error, term()}
  def stream_process(_pcm, _language, _keywords, _capacity),
    do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_stream_transcribe_stop`: ends the live stream; returns the tail as JSON."
  @spec stream_stop(pos_integer()) :: {:ok, binary()} | {:error, term()}
  def stream_stop(_capacity), do: :erlang.nif_error(:nif_not_loaded)

  @doc "`needle_embed`: exactly one of `text` and `pcm` is non-nil. Returns native f32 floats."
  @spec embed(iodata() | nil, binary() | nil) :: {:ok, binary()} | {:error, term()}
  def embed(_text, _pcm), do: :erlang.nif_error(:nif_not_loaded)
end
