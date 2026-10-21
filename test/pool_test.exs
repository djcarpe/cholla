defmodule Cholla.PoolTest do
  use ExUnit.Case, async: false
  @moduletag :engine
  @moduletag :capture_log
  @moduletag timeout: 600_000

  alias Cholla.{Engine, Pool, Tool}

  defp control do
    Tool.new("control_device",
      description: "Switch or toggle any named smart-home device.",
      params: [device: :string, action: {:enum, ["on", "off", "toggle"]}]
    )
  end

  defp calls({:ok, response}), do: response["function_calls"]

  setup do
    start_supervised!({Pool, name: :pair, size: 2, threads: 2, preload: [:text]})
    :ok = Pool.preload(:pair)
  end

  test "each instance is a separate engine with its own thread pool" do
    [a, b] = Pool.status(:pair)
    assert %{threads: 2, path: path} = a.instance
    assert %{threads: 2, path: ^path} = b.instance

    refute :sys.get_state(Pool.engine(:pair, 0)).instance ==
             :sys.get_state(Pool.engine(:pair, 1)).instance

    if a.instance.thread_control do
      # Opening a 3-thread instance adds 2 workers; the caller is the third.
      before = File.ls!("/proc/self/task") |> length()
      start_supervised!({Pool, name: :one, threads: 3, preload: [:text]}, id: :one)
      :ok = Pool.preload(:one)
      assert length(File.ls!("/proc/self/task")) - before == 2
    end
  end

  test "stateful agents keep their own instance and conversation" do
    lamp = Cholla.agent(tools: [control()], system: "You control the lamp.", pool: :pair)

    # Stateful agents route by identity; find one that lands on the other instance.
    fan =
      Stream.repeatedly(fn ->
        Cholla.agent(tools: [control()], system: "You control the fan.", pool: :pair)
      end)
      |> Enum.find(&(Pool.engine(:pair, &1.id) != Pool.engine(:pair, lamp.id)))

    {:ok, _} = Cholla.complete(lamp, "toggle the lamp")
    {:ok, _} = Cholla.complete(fan, "toggle the fan")
    {:ok, _} = Cholla.complete(lamp, "and again")

    # Neither agent evicted the other, so the lamp conversation has both turns.
    assert %{active: id, turns: 2} = :sys.get_state(Pool.engine(:pair, lamp.id))
    assert id == lamp.id
    assert %{active: id, turns: 1} = :sys.get_state(Pool.engine(:pair, fan.id))
    assert id == fan.id
  end

  test "stateless work goes to the idle instance while one is reserved" do
    agent = Cholla.agent(tools: [control()], stateless: true, pool: :pair)
    parent = self()

    holder =
      spawn(fn ->
        Pool.transaction(:pair, fn ->
          send(parent, {:holding, Pool.engine(:pair)})
          receive do: (:release -> :ok)
        end)
      end)

    assert_receive {:holding, held}, 60_000

    for _ <- 1..3 do
      assert Pool.engine(:pair) != held
      assert [%{"name" => "control_device"}] = calls(Cholla.complete(agent, "toggle the fan"))
    end

    send(holder, :release)
  end

  test "calls inside a transaction stay on the reserved instance" do
    Pool.transaction(:pair, fn ->
      engine = Pool.engine(:pair)
      assert Enum.all?(1..20, fn _ -> Pool.engine(:pair) == engine end)
      assert Pool.engine(:pair, :some_key) == engine
      assert Pool.transaction(:pair, fn -> Pool.engine(:pair) end) == engine
    end)
  end

  test "concurrent queries across instances match serial ones" do
    agent = Cholla.agent(tools: [control()], stateless: true, pool: :pair)
    rooms = ~w(lamp fan heater kettle porch\ light garage\ door)
    ask = fn room -> calls(Cholla.complete(agent, "toggle the #{room}")) end
    serial = Enum.map(rooms, ask)

    concurrent =
      rooms
      |> Task.async_stream(ask, timeout: :infinity, max_concurrency: 6)
      |> Enum.map(fn {:ok, calls} -> calls end)

    assert concurrent == serial

    busy =
      Enum.map(1..2, &:sys.get_state({:via, Registry, {Cholla.Registry, {:engine, :pair, &1}}}))

    assert Enum.all?(busy, &(&1.active != nil))
  end

  test "oversubscribing the cores logs a warning" do
    log =
      ExUnit.CaptureLog.capture_log(fn ->
        cores = System.schedulers_online()
        start_supervised!({Pool, name: :greedy, threads: cores}, id: :greedy)
        :ok = Pool.preload(:greedy)
      end)

    assert log =~ "slow each other"
  end

  test "a missing pool raises" do
    assert_raise ArgumentError, ~r/no Cholla pool named :nope/, fn ->
      Cholla.embed("hi", pool: :nope)
    end
  end

  test "engines report their instance" do
    assert %{thread_control: _, vm_threads: total} = Engine.info(Pool.engine(:pair))
    assert total >= 4
  end
end
