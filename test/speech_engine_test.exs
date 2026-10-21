defmodule Cholla.SpeechEngineTest do
  use ExUnit.Case, async: false
  @moduletag :speech
  @moduletag timeout: 600_000

  alias Cholla.Speech

  # A clip of speech is needed; set CHOLLA_TEST_WAV to a 16 kHz WAV, or the
  # JFK sample from whisper.cpp is downloaded into the cache.
  setup_all do
    path =
      System.get_env("CHOLLA_TEST_WAV") ||
        Path.join(Cholla.Download.cache_dir(), "test/jfk.wav")

    unless File.exists?(path) do
      File.mkdir_p!(Path.dirname(path))
      {:ok, _} = Application.ensure_all_started([:inets, :ssl])
      url = ~c"https://github.com/ggml-org/whisper.cpp/raw/master/samples/jfk.wav"

      ssl = [
        verify: :verify_peer,
        cacerts: :public_key.cacerts_get(),
        customize_hostname_check: [match_fun: :public_key.pkix_verify_hostname_match_fun(:https)]
      ]

      {:ok, :saved_to_file} =
        :httpc.request(:get, {url, []}, [ssl: ssl], stream: String.to_charlist(path))
    end

    {:ok, pcm} = Speech.read_wav(path)
    %{pcm: pcm}
  end

  test "transcribe", %{pcm: pcm} do
    assert {:ok, %{"text" => text, "language" => "en", "words" => [_ | _]}} =
             Speech.transcribe(pcm, word_timestamps: true)

    assert text =~ ~r/ask not what your country/i
  end

  test "silence gives an empty transcript" do
    assert {:ok, %{"text" => ""}} =
             Speech.transcribe(Speech.from_floats(List.duplicate(0.0, 16_000)))
  end

  test "live stream", %{pcm: pcm} do
    chunks = for <<chunk::binary-size(64_000) <- pcm>>, do: chunk
    assert {:ok, %{"text" => text, "words" => [_ | _]}} = Speech.stream(chunks, language: "en")
    assert text =~ ~r/fellow Americans/i
  end

  test "audio embedding rows", %{pcm: pcm} do
    assert {:ok, [row | _] = rows} = Speech.embed(pcm, format: :rows)
    assert length(rows) == div(div(byte_size(pcm), 4) + 1279, 1280)
    assert Enum.all?(rows, &(length(&1) == length(row)))
  end

  test "complete answers spoken input", %{pcm: pcm} do
    agent = Cholla.agent(stateless: true)
    assert {:ok, %{"audio_text" => text}} = Cholla.complete(agent, {:pcm, pcm}, language: "en")
    assert text =~ ~r/your country/i
  end
end
