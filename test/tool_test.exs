defmodule Cholla.ToolTest do
  use ExUnit.Case, async: true
  alias Cholla.Tool

  test "shorthand params become a JSON Schema object" do
    tool =
      Tool.new(:control_device,
        description: "Switch a device.",
        params: [
          device: {:string, description: "Device name"},
          action: {:enum, [:on, :off, :toggle]},
          level: {:integer, minimum: 0, maximum: 100, required: false},
          when: :date,
          tags: {:array, :string, max_items: 3},
          rooms: {:array, [name: :string, floor: {:integer, default: 0}]},
          mode: {:enum, ["eco", "boost"], default: "eco"}
        ],
        triggers: [~r/\btoggle\b/, ~S"\bswitch\b"]
      )

    assert Tool.to_schema(tool) == %{
             "name" => "control_device",
             "description" => "Switch a device.",
             "triggers" => [~S"\btoggle\b", ~S"\bswitch\b"],
             "parameters" => %{
               "type" => "object",
               "required" => ["device", "action", "when", "tags", "rooms"],
               "properties" => %{
                 "device" => %{"type" => "string", "description" => "Device name"},
                 "action" => %{"type" => "string", "enum" => ["on", "off", "toggle"]},
                 "level" => %{"type" => "integer", "minimum" => 0, "maximum" => 100},
                 "when" => %{"type" => "string", "format" => "date"},
                 "tags" => %{"type" => "array", "items" => %{"type" => "string"}, "maxItems" => 3},
                 "rooms" => %{
                   "type" => "array",
                   "items" => %{
                     "type" => "object",
                     "required" => ["name"],
                     "properties" => %{
                       "name" => %{"type" => "string"},
                       "floor" => %{"type" => "integer", "default" => 0}
                     }
                   }
                 },
                 "mode" => %{"type" => "string", "enum" => ["eco", "boost"], "default" => "eco"}
               }
             }
           }
  end

  test "raw schema maps pass through with string keys" do
    params = %{type: :object, properties: %{city: %{type: "string"}}, required: [:city]}

    assert Tool.new("w", params: params).parameters == %{
             "type" => "object",
             "properties" => %{"city" => %{"type" => "string"}},
             "required" => ["city"]
           }

    assert Tool.to_schema(%{name: "x", parameters: %{}}) == %{"name" => "x", "parameters" => %{}}
  end

  test "enum types follow their values" do
    assert Tool.parameters(n: {:enum, [1, 2]})["properties"]["n"]["type"] == "integer"
  end

  test "unknown types raise" do
    assert_raise ArgumentError, fn -> Tool.parameters(x: :uuid) end
  end
end
