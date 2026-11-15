defmodule Cholla.AgentTest do
  use ExUnit.Case, async: true
  alias Cholla.Agent

  @now {{2026, 10, 20}, {9, 5, 0}}

  test "the local date is prefixed as a fact" do
    assert Agent.with_date_fact("", @now) == "date: 2026-10-20 Tue 09:05"
    assert Agent.with_date_fact("Be brief.", @now) == "date: 2026-10-20 Tue 09:05; Be brief."
  end

  test "an existing date fact or JSON prompt is left alone" do
    assert Agent.with_date_fact("date: 2020-01-01", @now) == "date: 2020-01-01"
    assert Agent.with_date_fact("today is 2020-01-01", @now) == "today is 2020-01-01"
    assert Agent.with_date_fact(~s({"k": 1}), @now) == ~s({"k": 1})
    assert Agent.with_date_fact("update: soon", @now) =~ "date: 2026-10-20"
  end

  test "tools are encoded and functions collected" do
    fun = fn _ -> :ok end
    tool = Cholla.Tool.new("t", params: [a: :string], function: fun)
    agent = Agent.new(tools: [tool, %{"name" => "raw", "parameters" => %{}}], auto_date: false)

    assert agent.system == ""
    assert agent.functions == %{"t" => fun}
    assert [%{"name" => "t"}, %{"name" => "raw"}] = JSON.decode!(agent.tools_json)
    refute Agent.new().id == Agent.new().id
  end
end
