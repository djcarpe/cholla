defmodule Cholla.VisionEngineTest do
  use ExUnit.Case, async: false
  @moduletag :vision
  @moduletag timeout: 600_000

  alias Cholla.Vision

  # A 64x48 picture: a white page with a black bar, enough for one pass.
  # Real images come from an image decoder; Mirage takes packed RGB.
  defp picture(w \\ 64, h \\ 48) do
    for y <- 0..(h - 1), x <- 0..(w - 1), into: <<>> do
      if y in 20..27 and x in 8..55, do: <<0, 0, 0>>, else: <<255, 255, 255>>
    end
  end

  test "describe returns Mirage's structured perception" do
    assert {:ok, %{"objects" => objects, "text" => text, "scene" => scene}} =
             Vision.describe(picture(), 64, 48)

    assert is_list(objects) and is_list(text) and is_map(scene)
  end

  test "sections narrow the result" do
    assert {:ok, result} =
             Vision.describe(picture(), 64, 48, options: %{"sections" => ["objects"]})

    assert Map.keys(result) -- ["objects"] == []
  end

  test "a form answers with values and probabilities" do
    form = %{"page" => %{"enum" => ["white", "black"]}}

    assert {:ok, %{"form" => %{"page" => %{"probabilities" => probs} = field}}} =
             Vision.extract(picture(), 64, 48, form)

    assert Map.has_key?(field, "value")
    assert Map.has_key?(probs, "white")
  end

  test "a wrong size is refused before the engine" do
    assert {:error, {:bad_rgb_size, 3, _}} = Vision.describe(<<0, 0, 0>>, 64, 48)
  end

  test "the engine's own error comes back" do
    assert {:error, message} =
             Vision.describe(picture(), 64, 48, options: %{"form" => %{"tag" => "string"}})

    assert is_binary(message)
  end

  test "text and vision share an engine" do
    tool =
      Cholla.Tool.new("get_weather",
        description: "Get the current weather for a city.",
        params: [city: {:string, description: "City name"}]
      )

    agent = Cholla.agent(tools: [tool], stateless: true)

    assert {:ok, %{"function_calls" => [%{"name" => "get_weather"}]}} =
             Cholla.complete(agent, "what's the weather in Paris?")

    assert {:ok, %{"objects" => _}} = Vision.describe(picture(), 64, 48)

    assert {:ok, %{"function_calls" => [%{"name" => "get_weather"}]}} =
             Cholla.complete(agent, "weather in Berlin please")
  end
end

defmodule Cholla.CactRoundTripTest do
  # Tuned builds (Cholla.Train) are written by Cholla.Train.Cact; the Cactus 4
  # engine must load them as it loads Cactus Compute's own archives.
  use ExUnit.Case, async: false
  @moduletag :engine
  @moduletag timeout: 600_000

  alias Cholla.NIF
  alias Cholla.Train.Cact

  test "an archive written by Cholla.Train.Cact loads and answers in the engine" do
    {:ok, base} = Cholla.Download.weights_path()
    {:ok, archive} = Cact.read(base)

    tensors =
      for rec <- archive.tensors,
          do: {rec.dtype, rec.shape, Cact.blob(archive, rec), rec.group, rec.bits}

    rewritten = Cact.encode(archive.header, archive.codebook, tensors)
    assert <<tag::little-32, _::binary>> = rewritten
    assert tag == Cact.tag()

    {:ok, lib} = Cholla.Download.engine_path()
    {:ok, instance} = NIF.open_engine(lib, 2, nil)
    assert {:ok, models} = NIF.load(instance, rewritten)
    assert Bitwise.band(models, NIF.text_model()) != 0

    tools =
      JSON.encode!([
        %{
          "name" => "get_weather",
          "description" => "Get the current weather for a city.",
          "parameters" => %{
            "type" => "object",
            "properties" => %{"city" => %{"type" => "string"}},
            "required" => ["city"]
          }
        }
      ])

    assert {:ok, _} = NIF.init(instance, "", tools, nil)
    assert {:ok, json} = NIF.complete(instance, "weather in Paris", nil, 64, 65_536)
    assert %{"function_calls" => [%{"name" => "get_weather"}]} = JSON.decode!(json)
  end
end
