defmodule Cholla.Tool do
  @moduledoc """
  A tool the model can call: a name, a description, a JSON Schema for its
  arguments, optional trigger regexes and, for `Cholla.run/3`, the Elixir
  function that executes it.

      Cholla.Tool.new("control_device",
        description: "Switch or toggle any named smart-home device.",
        params: [device: :string, action: {:enum, ["on", "off", "toggle"]}],
        triggers: [~S"\\b(turn|switch|power|flip)\\b.*\\b(on|off)\\b", ~S"\\btoggle\\b"],
        function: fn %{"device" => device, "action" => action} ->
          %{device: device, action: action}
        end
      )

  ## Parameter shorthand

  `:params` is either a JSON Schema map (`%{"type" => "object", ...}`) used
  as is, or a keyword list of `name: type` / `name: {type, opts}` where type is

    * `:string`, `:integer`, `:number`, `:boolean`, `:object`
    * `{:array, type}` or `{:array, type, opts}`
    * `{:enum, values}` or `{:enum, values, opts}`
    * `:date` or `:datetime` (strings with a JSON Schema `format`)
    * a nested keyword list, for an object with its own properties

  and opts are `:description`, `:required` (default `true`), `:default`,
  `:format`, `:minimum`, `:maximum`, `:pattern`, `:min_length`,
  `:max_length`, `:min_items` and `:max_items`. A parameter with a
  `:default` is not required.
  """

  @enforce_keys [:name]
  defstruct [
    :name,
    :description,
    :function,
    parameters: %{"type" => "object", "properties" => %{}},
    triggers: []
  ]

  @type t :: %__MODULE__{
          name: String.t(),
          description: String.t() | nil,
          parameters: map(),
          triggers: [String.t()],
          function: (map() -> term()) | nil
        }

  @doc "Builds a tool. See the module doc for options."
  @spec new(String.t() | atom(), keyword()) :: t()
  def new(name, opts \\ []) do
    %__MODULE__{
      name: to_string(name),
      description: opts[:description],
      parameters: parameters(Keyword.get(opts, :params, [])),
      triggers: Enum.map(Keyword.get(opts, :triggers, []), &regex_source/1),
      function: opts[:function]
    }
  end

  defp regex_source(%Regex{source: source}), do: source
  defp regex_source(source) when is_binary(source), do: source

  @doc "The JSON-ready schema map the engine receives."
  @spec to_schema(t() | map()) :: map()
  def to_schema(%__MODULE__{} = tool) do
    %{"name" => tool.name, "parameters" => tool.parameters}
    |> put_present("description", tool.description)
    |> put_present("triggers", if(tool.triggers == [], do: nil, else: tool.triggers))
  end

  def to_schema(%{} = schema), do: stringify(schema)

  defp put_present(map, _key, nil), do: map
  defp put_present(map, key, value), do: Map.put(map, key, value)

  @doc "Converts the `:params` shorthand into a JSON Schema object."
  @spec parameters(keyword() | map()) :: map()
  def parameters(%{} = schema), do: stringify(schema)

  def parameters(params) when is_list(params) do
    {properties, required} =
      Enum.reduce(params, {%{}, []}, fn {name, spec}, {props, required} ->
        {type, opts} = split(spec)
        name = to_string(name)
        props = Map.put(props, name, property(type, opts))

        required =
          if Keyword.get(opts, :required, not Keyword.has_key?(opts, :default)),
            do: [name | required],
            else: required

        {props, required}
      end)

    schema = %{"type" => "object", "properties" => properties}
    if required == [], do: schema, else: Map.put(schema, "required", Enum.reverse(required))
  end

  defp split({:enum, values}), do: {{:enum, values}, []}
  defp split({:enum, values, opts}), do: {{:enum, values}, opts}
  defp split({:array, item}), do: {{:array, item}, []}
  defp split({:array, item, opts}), do: {{:array, item}, opts}
  defp split({type, opts}) when is_list(opts), do: {type, opts}
  defp split(type), do: {type, []}

  @option_keys [
    description: "description",
    default: "default",
    format: "format",
    minimum: "minimum",
    maximum: "maximum",
    pattern: "pattern",
    min_length: "minLength",
    max_length: "maxLength",
    min_items: "minItems",
    max_items: "maxItems"
  ]

  defp property(type, opts) do
    Enum.reduce(@option_keys, type_schema(type), fn {key, json}, schema ->
      case Keyword.fetch(opts, key) do
        {:ok, value} -> Map.put(schema, json, value)
        :error -> schema
      end
    end)
  end

  defp type_schema(type) when type in [:string, :integer, :number, :boolean, :object],
    do: %{"type" => Atom.to_string(type)}

  defp type_schema(:date), do: %{"type" => "string", "format" => "date"}
  defp type_schema(:datetime), do: %{"type" => "string", "format" => "date-time"}

  defp type_schema({:array, item}) do
    {type, opts} = split(item)
    %{"type" => "array", "items" => property(type, opts)}
  end

  defp type_schema({:enum, [first | _] = values}) do
    type =
      cond do
        is_integer(first) -> "integer"
        is_number(first) -> "number"
        is_boolean(first) -> "boolean"
        true -> "string"
      end

    %{"type" => type, "enum" => Enum.map(values, &enum_value/1)}
  end

  defp type_schema(nested) when is_list(nested), do: parameters(nested)
  defp type_schema(%{} = schema), do: stringify(schema)

  defp type_schema(other),
    do: raise(ArgumentError, "unsupported Cholla.Tool parameter type: #{inspect(other)}")

  defp enum_value(value) when is_atom(value) and not is_boolean(value), do: Atom.to_string(value)
  defp enum_value(value), do: value

  @key_order ~w(name description parameters arguments type properties required enum items)

  @doc """
  Compact JSON with the key order the model was trained on: `name`,
  `description`, `parameters`, `arguments`, `type`, `properties`, `required`,
  `enum`, `items`, then any other keys sorted.

  The engine places the tools JSON in the prompt as given, and the model is
  sensitive to the spelling: Elixir's sorted map keys (`description` before
  `name`) cost the base model two thirds of its exact matches on a 60-example
  set against this order, so `Cholla.agent/1` and `Cholla.Train` both use it.
  """
  @spec encode_json(term()) :: String.t()
  def encode_json(term), do: term |> ordered_iodata() |> IO.iodata_to_binary()

  defp ordered_iodata(%{} = map) do
    {first, rest} = Enum.split_with(Map.keys(map), &(&1 in @key_order))

    keys =
      Enum.sort_by(first, &Enum.find_index(@key_order, fn k -> k == &1 end)) ++ Enum.sort(rest)

    pairs =
      for key <- keys do
        [JSON.encode!(to_string(key)), ?:, ordered_iodata(map[key])]
      end

    [?{, Enum.intersperse(pairs, ?,), ?}]
  end

  defp ordered_iodata(list) when is_list(list),
    do: [?[, list |> Enum.map(&ordered_iodata/1) |> Enum.intersperse(?,), ?]]

  defp ordered_iodata(other), do: JSON.encode!(other)

  defp stringify(%{} = map), do: Map.new(map, fn {k, v} -> {to_string(k), stringify(v)} end)
  defp stringify(list) when is_list(list), do: Enum.map(list, &stringify/1)

  defp stringify(atom) when is_atom(atom) and not is_boolean(atom) and not is_nil(atom),
    do: Atom.to_string(atom)

  defp stringify(other), do: other
end
