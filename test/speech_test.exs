defmodule Cholla.SpeechTest do
  use ExUnit.Case, async: true
  alias Cholla.Speech

  defp wav(format, channels, rate, bits, data) do
    fmt =
      <<format::little-16, channels::little-16, rate::little-32,
        rate * channels * div(bits, 8)::little-32, channels * div(bits, 8)::little-16,
        bits::little-16>>

    body =
      <<"WAVE", "fmt ", byte_size(fmt)::little-32, fmt::binary, "LIST", 3::little-32, "abc", 0,
        "data", byte_size(data)::little-32, data::binary>>

    <<"RIFF", byte_size(body)::little-32, body::binary>>
  end

  defp floats(pcm), do: for(<<f::float-32-native <- pcm>>, do: f)

  test "16-bit stereo is mixed down to mono floats" do
    data =
      <<16384::little-signed-16, 0::little-signed-16, -32768::little-signed-16,
        -32768::little-signed-16>>

    assert {:ok, pcm} = Speech.parse_wav(wav(1, 2, 16_000, 16, data))
    assert floats(pcm) == [0.25, -1.0]
  end

  test "32-bit float wav is read as is" do
    assert {:ok, pcm} =
             Speech.parse_wav(
               wav(3, 1, 16_000, 32, <<0.5::little-float-32, -0.25::little-float-32>>)
             )

    assert floats(pcm) == [0.5, -0.25]
  end

  test "other sample rates and non-wav input are refused" do
    assert {:error, {:sample_rate, 44_100}} = Speech.parse_wav(wav(1, 1, 44_100, 16, <<0, 0>>))
    assert {:error, :not_a_wav} = Speech.parse_wav("nope")
  end

  test "from_floats packs native f32" do
    assert Speech.from_floats([0, 0.5]) == <<0.0::float-32-native, 0.5::float-32-native>>
    assert Speech.keywords(["a", "b c"]) == "a\nb c"
  end
end
