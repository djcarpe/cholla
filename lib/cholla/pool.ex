defmodule Cholla.Pool do
  @moduledoc """
  A pool of Needle engine instances and the router in front of them.

  The engine keeps one model, one conversation and one live stream per
  instance and runs one call at a time, so a single instance caps throughput
  at one query at a time. A pool runs several instances side by side, each a
  private copy of the engine library with its own weights and state, managed
  by its own `Cholla.Engine` process.

  ## Starting pools

  The application starts the pools in `config :cholla, :pools`, a keyword
  list of pool name to options. The default is one pool, `:default`, with a
  single instance using every core:

      config :cholla,
        pools: [
          default: [size: 3, threads: 2, preload: [:text]],
          speech: [size: 1, threads: 2, preload: [:speech]]
        ],
        speech_pool: :speech

  `Cholla` calls use `:default` (or `config :cholla, default_pool: name`),
  `Cholla.Speech` calls use `config :cholla, speech_pool: name`, falling back
  to the default pool, and either can be overridden per call or per agent
  with `pool: name`. Pools can also be started in your own supervision tree
  with `{Cholla.Pool, name: :mine, size: 2}`.

  ## Options

    * `:name` - the pool's name, an atom (required)
    * `:size` - number of instances (default 1)
    * `:threads` - threads per instance, counting the calling thread. Defaults
      to every core for a single instance and to the cores divided by `:size`
      otherwise
    * `:cpus` - a list with one list of CPU numbers per instance; each
      instance then runs only on its CPUs
    * `:preload` - models to load at start, `[:text]` and/or `[:speech]`;
      loading is otherwise deferred to first use

  ## Threads

  Every instance runs a pool of worker threads that spin while they wait for
  work. Instances whose threads add up to more than the machine's cores slow
  each other down by one or two orders of magnitude, so keep the sum of
  `size * threads` over all pools within the core count; a warning is logged
  when it is exceeded. Fewer threads per instance trade latency for
  throughput: on a 4-core/8-thread machine, one 8-thread instance answered
  about 5 queries a second and eight 1-thread instances about 14.

  Thread and CPU limits are applied on Linux only; elsewhere every instance
  sizes itself to all cores, so keep pools at size 1 there.

  ## Routing

  Stateless work (stateless agents, `Cholla.extract/3`, embeddings,
  transcription) goes to the instance with the least work queued. Work that
  depends on state held in an instance has affinity: a stateful agent's
  conversation always goes to the same instance (chosen by hashing the agent),
  and a live speech stream to the instance chosen for the calling process.
  Inside `transaction/3` every call the process makes to the pool goes to the
  reserved instance.

  Each instance loads its own copy of the weights (about 35 MB for text and
  17 MB for speech), and an instance's library and threads stay in memory
  until the VM exits.
  """

  use Supervisor

  alias Cholla.Engine

  @registry Cholla.Registry

  @doc false
  def child_spec(opts) do
    name = Keyword.fetch!(opts, :name)
    %{id: {__MODULE__, name}, start: {__MODULE__, :start_link, [opts]}, type: :supervisor}
  end

  @doc "Starts a pool. See the module docs for options."
  def start_link(opts) do
    name = Keyword.fetch!(opts, :name)
    Supervisor.start_link(__MODULE__, opts, name: {:via, Registry, {@registry, {:pool, name}}})
  end

  @impl true
  def init(opts) do
    name = Keyword.fetch!(opts, :name)
    size = Keyword.get(opts, :size, 1)
    cpus = Keyword.get(opts, :cpus)

    unless is_integer(size) and size > 0, do: raise(ArgumentError, "pool size must be positive")

    if cpus && length(cpus) != size,
      do: raise(ArgumentError, ":cpus needs one list of CPUs per instance")

    threads =
      Keyword.get_lazy(opts, :threads, fn ->
        if size == 1, do: nil, else: max(1, div(System.schedulers_online(), size))
      end)

    load = :atomics.new(size, signed: false)
    :persistent_term.put({__MODULE__, name}, {size, load})

    children =
      for index <- 1..size do
        Engine.child_spec(
          id: {Engine, index},
          name: engine_name(name, index),
          threads: threads,
          cpus: cpus && Enum.at(cpus, index - 1),
          preload: Keyword.get(opts, :preload, []),
          load: {load, index}
        )
      end

    Supervisor.init(children, strategy: :one_for_one)
  end

  defp engine_name(pool, index), do: {:via, Registry, {@registry, {:engine, pool, index}}}

  @doc false
  def default, do: Application.get_env(:cholla, :default_pool, :default)

  @doc false
  def speech_default, do: Application.get_env(:cholla, :speech_pool, default())

  @doc """
  Picks the engine for a call. With a `key`, the same key always gets the
  same engine; with `nil`, the engine with the least work queued. Inside a
  transaction on the pool, always the reserved engine.
  """
  @spec engine(atom(), term()) :: GenServer.server()
  def engine(pool, key \\ nil) do
    case Process.get({__MODULE__, pool}) do
      nil -> engine_name(pool, pick(pool, key))
      engine -> engine
    end
  end

  defp pick(pool, key) do
    {size, load} = config(pool)

    cond do
      size == 1 -> 1
      key != nil -> :erlang.phash2(key, size) + 1
      true -> least_loaded(pool, size, load)
    end
  end

  # Scans from a random start so ties spread across idle engines.
  defp least_loaded(pool, size, load) do
    start = :rand.uniform(size) - 1

    Enum.min_by(0..(size - 1), fn offset ->
      index = rem(start + offset, size) + 1
      :atomics.get(load, index) + mailbox(pool, index)
    end)
    |> then(&(rem(start + &1, size) + 1))
  end

  defp mailbox(pool, index) do
    with [{pid, _}] <- Registry.lookup(@registry, {:engine, pool, index}),
         {:message_queue_len, n} <- Process.info(pid, :message_queue_len) do
      n
    else
      _ -> 0
    end
  end

  defp config(pool) do
    :persistent_term.get({__MODULE__, pool}, nil) ||
      raise ArgumentError, "no Cholla pool named #{inspect(pool)} is running"
  end

  @doc "Runs an engine op on the engine `engine/2` picks."
  def op(pool, key, fun), do: pool |> engine(key) |> Engine.op(fun)

  @doc """
  Reserves one engine of `pool` for the calling process while `fun` runs.
  Every call the process makes to the pool meanwhile goes to that engine.
  Pass a `key` to reserve the engine that key routes to.
  """
  @spec transaction(atom(), term(), (-> result)) :: result when result: term()
  def transaction(pool \\ default(), key \\ nil, fun) when is_function(fun, 0) do
    dict_key = {__MODULE__, pool}

    case Process.get(dict_key) do
      nil ->
        engine = engine(pool, key)

        Engine.transaction(engine, fn ->
          Process.put(dict_key, engine)

          try do
            fun.()
          after
            Process.delete(dict_key)
          end
        end)

      engine ->
        Engine.transaction(engine, fun)
    end
  end

  @doc "Loads `models` (`:text`, `:speech`) into every engine of the pool."
  @spec preload(atom(), [:text | :speech]) :: :ok | {:error, term()}
  def preload(pool \\ default(), models \\ [:text]) do
    {size, _load} = config(pool)

    1..size
    |> Task.async_stream(&Engine.preload(engine_name(pool, &1), models), timeout: :infinity)
    |> Enum.find_value(:ok, fn
      {:ok, :ok} -> nil
      {:ok, error} -> error
    end)
  end

  @doc """
  Per-engine status: queued work and, once opened, the instance's threads,
  CPUs and the VM-wide thread total.
  """
  @spec status(atom()) :: [map()]
  def status(pool \\ default()) do
    {size, load} = config(pool)

    for index <- 1..size do
      %{
        index: index,
        queued: :atomics.get(load, index) + mailbox(pool, index),
        instance: Engine.info(engine_name(pool, index))
      }
    end
  end
end
