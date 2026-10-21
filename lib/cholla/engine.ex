defmodule Cholla.Engine do
  @moduledoc """
  Owns one Needle engine instance.

  An instance holds one text model, one speech model and one conversation,
  and it is not thread-safe. This server serialises access to it, opens the
  instance and loads weights on first use (or at start with `:preload`), and
  re-initialises it when a different agent takes over.

  Engines run inside a `Cholla.Pool`, which routes calls between them; use
  the pool rather than this module directly.

  `transaction/2` reserves the engine for the calling process across several
  operations, so a multi-step tool loop cannot be interleaved with another
  process's queries. Other callers wait until the transaction ends.
  """

  use GenServer
  require Logger

  alias Cholla.{Download, NIF}

  @unreset_turns 4

  defstruct instance: nil,
            threads: nil,
            cpus: nil,
            load: nil,
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
  def start_link(opts), do: GenServer.start_link(__MODULE__, opts, Keyword.take(opts, [:name]))

  @doc false
  def child_spec(opts) do
    %{id: Keyword.get(opts, :id, __MODULE__), start: {__MODULE__, :start_link, [opts]}}
  end

  @doc """
  Runs `fun.(state)` inside the engine server, where `fun` returns
  `{reply, state}`. Used by `Cholla` and `Cholla.Speech` through the pool.
  """
  def op(engine, fun) when is_function(fun, 1),
    do: GenServer.call(engine, {:op, fun}, :infinity)

  @doc "Runs `fun` with the engine reserved for the calling process."
  def transaction(engine, fun) when is_function(fun, 0) do
    :ok = GenServer.call(engine, :acquire, :infinity)

    try do
      fun.()
    after
      GenServer.call(engine, :release, :infinity)
    end
  end

  @doc """
  Opens the instance and loads models now instead of on first use.
  `models` lists `:text` and/or `:speech`.
  """
  def preload(engine, models \\ [:text]) do
    op(engine, fn state -> state |> load_models(models) |> reply() end)
  end

  @doc "The instance's settings and the VM-wide thread total, or `nil` before it opens."
  def info(engine) do
    op(engine, fn
      %{instance: nil} = state -> {nil, state}
      state -> {NIF.info(state.instance), state}
    end)
  end

  ## Helpers for op functions. Each returns {:ok, state} or {:error, reason, state},
  ## where state always reflects what the engine actually holds.

  @doc false
  def ensure_open(%__MODULE__{instance: nil} = state) do
    with {:ok, path} <- Download.engine_path(),
         {:ok, instance} <- NIF.open_engine(path, state.threads || 0, state.cpus) do
      warn_oversubscribed(NIF.info(instance))
      {:ok, %{state | instance: instance}}
    else
      {:error, reason} -> {:error, {:engine_unavailable, reason}, state}
    end
  end

  def ensure_open(state), do: {:ok, state}

  defp warn_oversubscribed(%{vm_threads: total, instances: instances}) do
    cores = System.schedulers_online()

    if total > cores do
      Logger.warning(
        "cholla: #{instances} engine instances run #{total} threads on #{cores} cores. " <>
          "Engine threads spin while they wait, so oversubscribed instances slow each other " <>
          "down badly; give each pool `threads:` so the total stays within #{cores}"
      )
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
         {:ok, _models} <- NIF.load(state.instance, data) do
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
           {:ok, models} <- NIF.load(state.instance, data) do
        if Bitwise.band(models, NIF.speech_model()) != 0,
          do: {:ok, %{state | speech_loaded: true}},
          else: {:error, {:load_failed, path, "archive holds no speech model"}, state}
      else
        {:error, reason} -> {:error, {:speech_unavailable, reason}, state}
      end
    end
  end

  defp load_models(state, models) do
    Enum.reduce_while(models, {:ok, state}, fn model, {:ok, state} ->
      case load_model(state, model) do
        {:ok, state} -> {:cont, {:ok, state}}
        error -> {:halt, error}
      end
    end)
  end

  defp load_model(state, :text), do: ensure_text(state, nil)
  defp load_model(state, :speech), do: ensure_speech(state)

  @doc false
  def bind(state, agent) do
    with {:ok, state} <- ensure_text(state, agent.weights) do
      if state.active == agent.id do
        {:ok, state}
      else
        case NIF.init(state.instance, agent.system, agent.tools_json, agent.tool_index_path) do
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
  def init(opts) do
    state = %__MODULE__{
      threads: opts[:threads],
      cpus: opts[:cpus],
      load: opts[:load]
    }

    case Keyword.get(opts, :preload, []) do
      [] -> {:ok, state}
      models -> {:ok, state, {:continue, {:preload, models}}}
    end
  end

  @impl true
  def handle_continue({:preload, models}, state) do
    state =
      case load_models(state, models) do
        {:ok, state} ->
          state

        {:error, reason, state} ->
          Logger.warning("cholla: preloading #{inspect(models)} failed: #{inspect(reason)}")
          state
      end

    {:noreply, state}
  end

  @impl true
  def handle_call(msg, {pid, _} = from, state) do
    if state.owner == nil or state.owner == pid do
      dispatch(msg, from, state)
    else
      {:noreply, publish(%{state | queue: :queue.in({msg, from}, state.queue)})}
    end
  end

  @impl true
  def handle_info({:DOWN, ref, :process, _, _}, %{owner_ref: ref} = state) do
    {:noreply, drain(%{state | owner: nil, owner_ref: nil, depth: 0})}
  end

  def handle_info(_msg, state), do: {:noreply, state}

  defp dispatch(:acquire, {pid, _}, %{owner: nil} = state) do
    state = %{state | owner: pid, owner_ref: Process.monitor(pid), depth: 1}
    {:reply, :ok, publish(state)}
  end

  defp dispatch(:acquire, _from, state), do: {:reply, :ok, %{state | depth: state.depth + 1}}

  defp dispatch(:release, _from, %{owner: nil} = state), do: {:reply, :ok, state}

  defp dispatch(:release, _from, %{depth: 1} = state) do
    Process.demonitor(state.owner_ref, [:flush])
    {:reply, :ok, drain(%{state | owner: nil, owner_ref: nil, depth: 0})}
  end

  defp dispatch(:release, _from, state), do: {:reply, :ok, %{state | depth: state.depth - 1}}

  defp dispatch({:op, fun}, _from, state) do
    publish(state, 1)

    {reply, state} =
      try do
        fun.(state)
      rescue
        exception -> {{:error, exception}, state}
      end

    {:reply, reply, publish(state)}
  end

  # Serves queued calls in order until one of them takes the lock.
  defp drain(%{owner: nil} = state) do
    case :queue.out(state.queue) do
      {:empty, _} ->
        publish(state)

      {{:value, {msg, from}}, queue} ->
        state = %{state | queue: queue}

        case dispatch(msg, from, state) do
          {:reply, reply, state} ->
            GenServer.reply(from, reply)
            drain(state)
        end
    end
  end

  defp drain(state), do: publish(state)

  # Publishes this engine's backlog for the pool's router: work held back by a
  # transaction plus one while it runs or is reserved. The mailbox is counted
  # by the router itself.
  defp publish(state, running \\ 0)
  defp publish(%{load: nil} = state, _running), do: state

  defp publish(%{load: {ref, index}} = state, running) do
    busy = if state.owner, do: 1, else: running
    :atomics.put(ref, index, busy + :queue.len(state.queue))
    state
  end
end
