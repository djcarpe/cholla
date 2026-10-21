defmodule Cholla.Engine do
  @moduledoc """
  Owns the process-global Needle engine.

  The engine holds one text model, one speech model and one conversation for
  the whole VM, and it is not thread-safe. This server serialises access,
  opens the engine library and loads weights on first use, and re-initialises
  the engine when a different agent takes over.

  `transaction/1` reserves the engine for the calling process across several
  operations, so a multi-step tool loop cannot be interleaved with another
  process's queries. Other callers wait until the transaction ends.
  """

  use GenServer
  require Logger

  alias Cholla.{Download, NIF}

  @unreset_turns 4

  defstruct open: false,
            text_weights: nil,
            speech_loaded: false,
            active: nil,
            turns: 0,
            warned: false,
            owner: nil,
            owner_ref: nil,
            depth: 0,
            queue: :queue.new()

  @doc false
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, name: __MODULE__)

  @doc """
  Runs `fun.(state)` inside the engine server, where `fun` returns
  `{reply, state}`. Used by `Cholla` and `Cholla.Speech`.
  """
  def op(fun) when is_function(fun, 1), do: GenServer.call(__MODULE__, {:op, fun}, :infinity)

  @doc "Runs `fun` with the engine reserved for the calling process."
  def transaction(fun) when is_function(fun, 0) do
    :ok = GenServer.call(__MODULE__, :acquire, :infinity)

    try do
      fun.()
    after
      GenServer.call(__MODULE__, :release, :infinity)
    end
  end

  @doc "Opens the engine library and loads the base text model now instead of on first use."
  def preload(opts \\ []) do
    op(fn state ->
      with {:ok, state} <- ensure_text(state, nil),
           {:ok, state} <- if(opts[:speech], do: ensure_speech(state), else: {:ok, state}) do
        {:ok, state}
      end
      |> reply()
    end)
  end

  ## Helpers for op functions. Each returns {:ok, state} or {:error, reason, state},
  ## where state always reflects what the engine actually holds.

  @doc false
  def ensure_open(%__MODULE__{open: true} = state), do: {:ok, state}

  def ensure_open(state) do
    with {:ok, path} <- Download.engine_path(),
         :ok <- open(path) do
      {:ok, %{state | open: true}}
    else
      {:error, reason} -> {:error, {:engine_unavailable, reason}, state}
    end
  end

  defp open(path) do
    case NIF.open_engine(path) do
      {:error, {:already_open, _}} -> :ok
      other -> other
    end
  end

  @doc false
  def ensure_text(state, weights) do
    with {:ok, state} <- ensure_open(state) do
      case if(weights, do: {:ok, weights}, else: Download.weights_path()) do
        {:ok, path} when path == state.text_weights -> {:ok, state}
        {:ok, path} -> load_text(state, path)
        {:error, reason} -> {:error, {:weights_unavailable, reason}, state}
      end
    end
  end

  defp load_text(state, path) do
    with {:ok, data} <- File.read(path),
         {:ok, _models} <- NIF.load(data) do
      # New text weights invalidate whatever prompt and tools were bound.
      {:ok, %{state | text_weights: path, active: nil}}
    else
      # A failed load may have replaced the model, so trust nothing loaded before.
      {:error, reason} ->
        {:error, {:load_failed, path, reason}, %{state | text_weights: nil, active: nil}}
    end
  end

  @doc false
  def ensure_speech(%__MODULE__{speech_loaded: true} = state), do: {:ok, state}

  def ensure_speech(state) do
    with {:ok, state} <- ensure_open(state) do
      with {:ok, path} <- Download.speech_weights_path(),
           {:ok, data} <- File.read(path),
           {:ok, models} <- NIF.load(data) do
        if Bitwise.band(models, NIF.speech_model()) != 0,
          do: {:ok, %{state | speech_loaded: true}},
          else: {:error, {:load_failed, path, "archive holds no speech model"}, state}
      else
        {:error, reason} -> {:error, {:speech_unavailable, reason}, state}
      end
    end
  end

  @doc false
  def bind(state, agent) do
    with {:ok, state} <- ensure_text(state, agent.weights) do
      if state.active == agent.id do
        {:ok, state}
      else
        case NIF.init(agent.system, agent.tools_json, agent.tool_index_path) do
          {:ok, _prefix} -> {:ok, %{state | active: agent.id, turns: 0, warned: false}}
          {:error, reason} -> {:error, {:init_failed, reason}, %{state | active: nil}}
        end
      end
    end
  end

  @doc false
  # Turns a helper result into an op reply.
  def reply({:ok, state}), do: {:ok, state}
  def reply({:error, reason, state}), do: {{:error, reason}, state}
  @doc false
  def count_turn(state) do
    turns = state.turns + 1

    if turns > @unreset_turns and not state.warned do
      Logger.warning(
        "cholla: #{@unreset_turns} queries on this agent without reset: every turn stays in " <>
          "the conversation, so unrelated queries lose accuracy and confidence. Call " <>
          "Cholla.reset/1 between independent queries, or build the agent with stateless: true"
      )

      %{state | turns: turns, warned: true}
    else
      %{state | turns: turns}
    end
  end

  @doc false
  def reset_turns(state), do: %{state | turns: 0}

  ## Server

  @impl true
  def init(_opts), do: {:ok, %__MODULE__{}}

  @impl true
  def handle_call(msg, {pid, _} = from, state) do
    if state.owner == nil or state.owner == pid do
      dispatch(msg, from, state)
    else
      {:noreply, %{state | queue: :queue.in({msg, from}, state.queue)}}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _, _}, %{owner_ref: ref} = state) do
    {:noreply, drain(%{state | owner: nil, owner_ref: nil, depth: 0})}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  defp dispatch(:acquire, {pid, _}, %{owner: nil} = state) do
    {:reply, :ok, %{state | owner: pid, owner_ref: Process.monitor(pid), depth: 1}}
  end

  defp dispatch(:acquire, _from, state), do: {:reply, :ok, %{state | depth: state.depth + 1}}

  defp dispatch(:release, _from, %{owner: nil} = state), do: {:reply, :ok, state}

  defp dispatch(:release, _from, %{depth: 1} = state) do
    Process.demonitor(state.owner_ref, [:flush])
    {:reply, :ok, drain(%{state | owner: nil, owner_ref: nil, depth: 0})}
  end

  defp dispatch(:release, _from, state), do: {:reply, :ok, %{state | depth: state.depth - 1}}

  defp dispatch({:op, fun}, _from, state) do
    {reply, state} =
      try do
        fun.(state)
      rescue
        exception -> {{:error, exception}, state}
      end

    {:reply, reply, state}
  end

  # Serves queued calls in order until one of them takes the lock.
  defp drain(%{owner: nil} = state) do
    case :queue.out(state.queue) do
      {:empty, _} ->
        state

      {{:value, {msg, from}}, queue} ->
        state = %{state | queue: queue}

        case dispatch(msg, from, state) do
          {:reply, reply, state} ->
            GenServer.reply(from, reply)
            drain(state)
        end
    end
  end

  defp drain(state), do: state
end
