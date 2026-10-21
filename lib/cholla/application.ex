defmodule Cholla.Application do
  @moduledoc false
  use Application

  @impl true
  def start(_type, _args) do
    Supervisor.start_link([Cholla.Engine], strategy: :one_for_one, name: Cholla.Supervisor)
  end
end
