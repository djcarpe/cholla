defmodule Cholla.EngineTest do
  # These share the default pool, so they run serially.
  use ExUnit.Case, async: false
  @moduletag :engine
  @moduletag timeout: 600_000

  alias Cholla.{NIF, Tool}

  setup_all do
    :ok = Cholla.preload()
  end

  defp control(fun \\ nil) do
    Tool.new("control_device",
      description: "Switch or toggle any named smart-home device.",
      params: [device: :string, action: {:enum, ["on", "off", "toggle"]}],
      triggers: [~S"\b(turn|switch|power|flip)\b.*\b(on|off)\b", ~S"\btoggle\b"],
      function: fun
    )
  end

  defp weather(fun) do
    Tool.new("get_weather",
      description: "Get the current weather for a city.",
      params: [city: {:string, description: "City name"}],
      function: fun
    )
  end

  test "complete returns a tool call" do
    agent = Cholla.agent(tools: [control()], stateless: true)
    assert {:ok, response} = Cholla.complete(agent, "toggle the garage door")
    assert response["success"] == true
    assert [%{"name" => "control_device", "arguments" => args}] = response["function_calls"]
    assert args == %{"device" => "garage door", "action" => "toggle"}
    assert is_float(response["confidence"])
  end

  test "run executes tools and feeds results back" do
    test_pid = self()

    agent =
      Cholla.agent(
        tools: [
          weather(fn %{"city" => city} ->
            send(test_pid, {:called, city})
            %{city: city, temp_c: 18}
          end)
        ],
        stateless: true
      )

    assert {:ok, response} = Cholla.run(agent, "what's the weather in Lisbon?")
    assert_received {:called, "Lisbon"}
    assert [%{city: "Lisbon", temp_c: 18}] = response["results"]
    assert response["function_calls"] == []
  end

  test "a tool that raises reports the error to the model" do
    agent = Cholla.agent(tools: [weather(fn _ -> raise "service down" end)], stateless: true)
    assert {:ok, response} = Cholla.run(agent, "weather in Paris please")
    # The model may retry a failing tool, up to :max_steps rounds.
    assert [_ | _] = results = response["results"]
    assert Enum.all?(results, &(&1 == %{"error" => "service down"}))
  end

  test "run holds calls below min_confidence" do
    agent = Cholla.agent(tools: [control(fn _ -> flunk("must not run") end)], stateless: true)

    assert {:ok, %{"held" => true, "results" => [], "function_calls" => [_]}} =
             Cholla.run(agent, "toggle the garage door", min_confidence: 0.999)
  end

  test "extract fills a schema" do
    invoice =
      Tool.new("Invoice", description: "An invoice.", params: [vendor: :string, total: :number])

    assert {:ok, %{"vendor" => "Acme Corp", "total" => total}} =
             Cholla.extract("Invoice from Acme Corp for $1,250.00.", invoice)

    assert total == 1250
  end

  test "embeddings are deterministic and fixed width" do
    assert {:ok, a} = Cholla.embed("hello world")
    assert {:ok, ^a} = Cholla.embed("hello world")
    assert {:ok, bin} = Cholla.embed("goodbye", format: :binary)
    assert byte_size(bin) == length(a) * 4
  end

  test "agents can share the engine; switching rebinds" do
    lights = Cholla.agent(tools: [control()])
    weather = Cholla.agent(tools: [weather(nil)])

    for _ <- 1..2 do
      assert {:ok, %{"function_calls" => [%{"name" => "control_device"}]}} =
               Cholla.complete(lights, "turn on the porch light")

      assert {:ok, %{"function_calls" => [%{"name" => "get_weather"}]}} =
               Cholla.complete(weather, "weather in Oslo")
    end

    assert :ok = Cholla.reset(lights)
    assert :ok = Cholla.reset(weather)
  end

  test "concurrent callers are serialised" do
    agent = Cholla.agent(tools: [control()], stateless: true)
    rooms = ~w(kitchen bedroom hallway garage office attic)

    ask = fn room ->
      {:ok, response} = Cholla.complete(agent, "turn off the #{room} lights")
      response["function_calls"]
    end

    # The model is deterministic, so interleaving must not change any answer.
    serial = Enum.map(rooms, ask)

    concurrent =
      rooms
      |> Task.async_stream(ask, timeout: :infinity, max_concurrency: 6)
      |> Enum.map(fn {:ok, calls} -> calls end)

    assert concurrent == serial
    assert Enum.count(serial, &match?([_ | _], &1)) >= 4
  end

  test "a transaction holds the engine for its owner" do
    agent = Cholla.agent(tools: [control()], stateless: true)
    parent = self()

    Cholla.Pool.transaction(fn ->
      task =
        Task.async(fn ->
          send(parent, :other_done)
          Cholla.complete(agent, "toggle the fan")
        end)

      assert {:ok, _} = Cholla.complete(agent, "toggle the lamp")
      refute Task.yield(task, 300)
      send(parent, {:released, task})
    end)

    assert_received {:released, task}
    assert {:ok, {:ok, _}} = Task.yield(task, 60_000)
  end

  test "a dead transaction owner releases the engine" do
    {pid, ref} = spawn_monitor(fn -> Cholla.Pool.transaction(fn -> exit(:boom) end) end)
    assert_receive {:DOWN, ^ref, :process, ^pid, :boom}
    assert {:ok, _} = Cholla.embed("still works")
  end

  test "a too-small buffer reports the size needed" do
    agent = Cholla.agent(tools: [control()], stateless: true, buffer_size: 16)

    assert {:error, {:buffer_too_small, needed}} =
             Cholla.complete(agent, "toggle the garage door")

    assert needed > 16
  end

  test "bad arguments are rejected before reaching the engine" do
    %{instance: instance} = :sys.get_state(Cholla.Pool.engine(:default))
    pcm = <<0.0::float-32-native>>

    assert_raise ArgumentError, fn -> NIF.complete(instance, "text", pcm, 16, 1024) end
    assert_raise ArgumentError, fn -> NIF.complete(instance, nil, nil, 16, 1024) end
    assert_raise ArgumentError, fn -> NIF.complete(instance, nil, <<1, 2, 3>>, 16, 1024) end
    assert_raise ArgumentError, fn -> NIF.embed(instance, "a\0b", nil) end
    assert_raise ArgumentError, fn -> NIF.embed(make_ref(), "text", nil) end

    assert {:error, "cannot read /nonexistent/libneedle.so" <> _} =
             NIF.open_engine("/nonexistent/libneedle.so", 0, nil)
  end

  test "missing weights surface as an error" do
    agent = Cholla.agent(weights: "/nonexistent/tuned.cact")

    assert {:error, {:load_failed, "/nonexistent/tuned.cact", :enoent}} =
             Cholla.complete(agent, "hi")

    # The base model still works afterwards.
    assert {:ok, _} =
             Cholla.complete(Cholla.agent(tools: [control()], stateless: true), "toggle the fan")
  end
end
