defmodule Cholla.Speech do
  @moduledoc """
  Speech with the Whistle model: transcription, live streaming
  transcription and audio embeddings. The Whistle weights (`whistle.cact`)
  are downloaded on first use.

  Audio is 16 kHz mono PCM as a binary of native-endian 32-bit floats in
  [-1, 1]. Build it with `from_floats/1` or `read_wav/1`. To answer spoken
  queries with an agent, pass `{:pcm, audio}` to `Cholla.complete/3`.
  """

  alias Cholla.{Engine, NIF}

  @sample_rate 16_000
  @capacity 262_144

  @doc "The sample rate the engine expects."
  def sample_rate, do: @sample_rate

  @doc """
  Transcribes audio of any length.

  Returns the decoded JSON, e.g.
  `%{"text" => "...", "language" => "en", "ttft_ms" => ..., "decode_tps" => ...}`,
  plus `"words"` with per-word times when `word_timestamps: true`.

  ## Options

    * `:language` - `"en"`, `"de"`, `"fr"`, `"es"`, `"it"`, `"nl"` or `"pl"`;
      detected when omitted
    * `:keywords` - a list of words or phrases to bias recognition towards
    * `:word_timestamps` - include word timings (default `false`)
  """
  @spec transcribe(binary(), keyword()) :: {:ok, map()} | {:error, term()}
  def transcribe(pcm, opts \\ []) when is_binary(pcm) do
    with_speech(fn ->
      NIF.transcribe(
        pcm,
        opts[:language],
        keywords(opts[:keywords]),
        Keyword.get(opts, :word_timestamps, false),
        @capacity
      )
    end)
  end

  @doc """
  Feeds about a second of live audio to the stream transcriber.

  Returns the words committed by this call (two consecutive passes agree on
  them) and the unconfirmed `"pending"` tail. The engine holds one stream per
  VM, so keep a stream within one process, for example inside
  `Cholla.Engine.transaction/1`.

  Takes the `:language` and `:keywords` options of `transcribe/2`.
  """
  @spec feed(binary(), keyword()) :: {:ok, map()} | {:error, term()}
  def feed(pcm, opts \\ []) when is_binary(pcm) do
    with_speech(fn ->
      NIF.stream_process(pcm, opts[:language], keywords(opts[:keywords]), @capacity)
    end)
  end

  @doc "Ends the live stream, committing its tail. The next `feed/2` starts a new stream."
  @spec stop() :: {:ok, map()} | {:error, term()}
  def stop, do: with_speech(fn -> NIF.stream_stop(@capacity) end)

  @doc """
  Transcribes an enumerable of PCM chunks as a live stream and returns every
  committed word, ending the stream at the end. Holds the engine for the
  duration.
  """
  @spec stream(Enumerable.t(), keyword()) :: {:ok, map()} | {:error, term()}
  def stream(chunks, opts \\ []) do
    Engine.transaction(fn ->
      result =
        Enum.reduce_while(chunks, {:ok, []}, fn chunk, {:ok, acc} ->
          case feed(chunk, opts) do
            {:ok, part} -> {:cont, {:ok, [part | acc]}}
            error -> {:halt, error}
          end
        end)

      with {:ok, parts} <- result, {:ok, tail} <- stop() do
        parts = Enum.reverse([tail | parts])
        words = Enum.flat_map(parts, &(&1["words"] || []))

        text =
          parts |> Enum.map(&(&1["text"] || "")) |> Enum.reject(&(&1 == "")) |> Enum.join(" ")

        {:ok, %{"text" => text, "words" => words, "parts" => parts}}
      else
        error ->
          stop()
          error
      end
    end)
  end

  @doc """
  Embeds audio with the speech model: one row per 80 ms frame (1280
  samples). Returns a flat list of floats by default, a list of per-frame
  rows with `format: :rows`, or the raw f32 binary with `format: :binary`.
  """
  @spec embed(binary(), keyword()) :: {:ok, [float()] | binary()} | {:error, term()}
  def embed(pcm, opts \\ []) when is_binary(pcm) do
    Engine.op(fn state ->
      case Engine.ensure_speech(state) do
        {:ok, state} -> {NIF.embed(nil, pcm) |> embedding(pcm, opts[:format]), state}
        error -> Engine.reply(error)
      end
    end)
  end

  @frame_samples 1280

  defp embedding({:ok, bin}, pcm, :rows) do
    frames = div(div(byte_size(pcm), 4) + @frame_samples - 1, @frame_samples)
    count = div(byte_size(bin), 4)

    if frames > 0 and rem(count, frames) == 0 do
      {:ok, floats} = Cholla.format_floats({:ok, bin}, :list)
      {:ok, Enum.chunk_every(floats, div(count, frames))}
    else
      {:error, {:unexpected_embedding_size, count, frames}}
    end
  end

  defp embedding(result, _pcm, format), do: Cholla.format_floats(result, format)

  defp with_speech(fun) do
    Engine.op(fn state ->
      case Engine.ensure_speech(state) do
        {:ok, state} -> {decode(fun.()), state}
        error -> Engine.reply(error)
      end
    end)
  end

  defp decode({:ok, json}), do: JSON.decode(json)
  defp decode(error), do: error

  @doc false
  def keywords(nil), do: nil
  def keywords(words) when is_list(words), do: Enum.join(words, "\n")
  def keywords(words) when is_binary(words), do: words

  @doc "Packs a list of floats in [-1, 1] into engine PCM."
  @spec from_floats([number()]) :: binary()
  def from_floats(samples), do: for(s <- samples, into: <<>>, do: <<s * 1.0::float-32-native>>)

  @doc """
  Reads a 16 kHz WAV file (16-bit integer or 32-bit float samples) into engine
  PCM, mixing multiple channels down to mono. Other sample rates are refused,
  since the engine does not resample.
  """
  @spec read_wav(Path.t()) :: {:ok, binary()} | {:error, term()}
  def read_wav(path) do
    with {:ok, data} <- File.read(path), do: parse_wav(data)
  end

  @doc false
  def parse_wav(<<"RIFF", _size::little-32, "WAVE", chunks::binary>>), do: wav_chunks(chunks, nil)
  def parse_wav(_), do: {:error, :not_a_wav}

  defp wav_chunks(<<"fmt ", size::little-32, body::binary-size(size), rest::binary>>, _fmt) do
    <<format::little-16, channels::little-16, rate::little-32, _::little-32, _::little-16,
      bits::little-16, _::binary>> = body

    wav_chunks(skip_pad(rest, size), {format, channels, rate, bits})
  end

  defp wav_chunks(<<"data", size::little-32, rest::binary>>, {format, channels, rate, bits}) do
    data = binary_part(rest, 0, min(size, byte_size(rest)))

    cond do
      rate != @sample_rate ->
        {:error, {:sample_rate, rate}}

      format == 1 and bits == 16 ->
        {:ok, mix(for(<<s::little-signed-16 <- data>>, do: s / 32768), channels)}

      format in [3, 0xFFFE] and bits == 32 ->
        {:ok, mix(for(<<s::little-float-32 <- data>>, do: s), channels)}

      true ->
        {:error, {:unsupported_wav, format, bits}}
    end
  end

  defp wav_chunks(<<_id::binary-4, size::little-32, rest::binary>>, fmt)
       when byte_size(rest) >= size,
       do: wav_chunks(skip_pad(binary_part(rest, size, byte_size(rest) - size), size), fmt)

  defp wav_chunks(_, _), do: {:error, :no_audio_data}

  defp skip_pad(rest, size) when rem(size, 2) == 1 and byte_size(rest) > 0,
    do: binary_part(rest, 1, byte_size(rest) - 1)

  defp skip_pad(rest, _), do: rest

  defp mix(samples, 1), do: from_floats(samples)

  defp mix(samples, channels) do
    samples
    |> Enum.chunk_every(channels, channels, :discard)
    |> Enum.map(&(Enum.sum(&1) / channels))
    |> from_floats()
  end
end
