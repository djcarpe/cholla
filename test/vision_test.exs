defmodule Cholla.VisionTest do
  use ExUnit.Case, async: true

  test "the engine's nan and inf scores become null, strings are kept" do
    raw = ~S({"score":-nan,"b":nan,"c":inf,"d":-inf,"name":"nan \"inf\" \\","e":[1,-nan],"f":0.5})

    assert {:ok, v} = raw |> Cholla.Vision.finite() |> JSON.decode()
    assert %{"score" => nil, "b" => nil, "c" => nil, "d" => nil, "e" => [1, nil], "f" => 0.5} = v
    assert v["name"] == ~S(nan "inf" \\) |> String.replace("\\\\", "\\")
  end

  test "JSON without them is unchanged" do
    json = ~S({"objects":[{"name":"info","score":0.91}],"text":"no nan"})
    assert Cholla.Vision.finite(json) == json
  end
end
