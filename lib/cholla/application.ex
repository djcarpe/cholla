defmodule Cholla.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    pools =
      for {name, opts} <- Application.get_env(:cholla, :pools, default: []),
          do: {Cholla.Pool, Keyword.put(opts, :name, name)}

    children = [{Registry, keys: :unique, name: Cholla.Registry} | pools]
    Supervisor.start_link(children, strategy: :one_for_one, name: Cholla.Supervisor)
  end
end
