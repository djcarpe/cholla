defmodule Cholla.Agent do
  @moduledoc """
  An agent configuration: system prompt, tools and options.

  Agents are plain data. An engine instance runs one conversation at a time,
  owned by whichever agent last used it; using a different agent
  re-initialises the instance with that agent's prompt and tools, which
  discards the previous agent's conversation. Build agents with `Cholla.agent/1`.
  """

  alias Cholla.Tool

  @enforce_keys [:id]
  defstruct [
    :id,
    :weights,
    :tool_index_path,
    :pool,
    system: "",
    tools: [],
    tools_json: "[]",
    functions: %{},
    stateless: false,
    buffer_size: 262_144,
    max_new_tokens: 512
  ]

  @type t :: %__MODULE__{
          id: reference(),
          system: String.t(),
          tools: [map()],
          tools_json: String.t(),
          functions: %{String.t() => (map() -> term())},
          tool_index_path: String.t() | nil,
          weights: String.t() | nil,
          pool: atom() | nil,
          stateless: boolean(),
          buffer_size: pos_integer(),
          max_new_tokens: pos_integer()
        }

  @doc """
  Builds an agent.

  ## Options

    * `:tools` - a list of `Cholla.Tool` structs or raw tool schema maps
    * `:system` - system prompt (default `""`)
    * `:auto_date` - prefix the local date and time as a `date:` fact, as the
      Python package does, unless the prompt already has one (default `true`)
    * `:weights` - path to a fine-tuned `.cact`; the base model is used otherwise
    * `:tool_index_path` - path to a tool index for large tool sets
    * `:stateless` - reset the conversation before every query (default `false`).
      Stateless agents can run on any instance of their pool; a stateful
      agent's conversation stays on one instance
    * `:pool` - the `Cholla.Pool` to run on (default: the default pool)
    * `:buffer_size` - response buffer in bytes (default 256 KiB)
    * `:max_new_tokens` - default generation limit (default 512)
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    tools = Keyword.get(opts, :tools, [])
    schemas = Enum.map(tools, &Tool.to_schema/1)

    functions =
      for %Tool{name: name, function: fun} <- tools,
          is_function(fun, 1),
          into: %{},
          do: {name, fun}

    system = Keyword.get(opts, :system, "") || ""
    system = if Keyword.get(opts, :auto_date, true), do: with_date_fact(system), else: system

    %__MODULE__{
      id: make_ref(),
      system: system,
      tools: schemas,
      tools_json: Tool.encode_json(schemas),
      functions: functions,
      tool_index_path: opts[:tool_index_path] && Path.expand(opts[:tool_index_path]),
      weights: opts[:weights] && Path.expand(opts[:weights]),
      pool: opts[:pool],
      stateless: Keyword.get(opts, :stateless, false),
      buffer_size: Keyword.get(opts, :buffer_size, 262_144),
      max_new_tokens: Keyword.get(opts, :max_new_tokens, 512)
    }
  end

  @date_key ~r/(?<![A-Za-z])date:/
  @iso_stamp ~r/\d{4}-\d{2}-\d{2}/

  @doc false
  def with_date_fact(system, now \\ :calendar.local_time()) do
    cond do
      Regex.match?(@date_key, system) or Regex.match?(@iso_stamp, system) -> system
      String.starts_with?(String.trim_leading(system), "{") -> system
      String.trim(system) == "" -> date_fact(now)
      true -> date_fact(now) <> "; " <> system
    end
  end

  @days ~w(Mon Tue Wed Thu Fri Sat Sun)

  defp date_fact({{y, m, d} = date, {h, min, _}}) do
    day = Enum.at(@days, :calendar.day_of_the_week(date) - 1)
    pad = &String.pad_leading(Integer.to_string(&1), 2, "0")
    "date: #{y}-#{pad.(m)}-#{pad.(d)} #{day} #{pad.(h)}:#{pad.(min)}"
  end
end
