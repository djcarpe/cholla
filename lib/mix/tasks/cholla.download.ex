defmodule Mix.Tasks.Cholla.Download do
  @shortdoc "Downloads the Needle engine and weights into the cache"
  @moduledoc """
  Downloads the Needle engine library and base weights ahead of time, so the
  first query does not wait on the network.

      mix cholla.download            # engine + needle3.cact
      mix cholla.download --speech   # also whistle.cact
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, _} = OptionParser.parse!(args, strict: [speech: :boolean])
    Application.ensure_all_started([:inets, :ssl])

    steps = [engine: &Cholla.Download.engine_path/0, weights: &Cholla.Download.weights_path/0]

    steps =
      if opts[:speech], do: steps ++ [speech: &Cholla.Download.speech_weights_path/0], else: steps

    for {name, fetch} <- steps do
      case fetch.() do
        {:ok, path} -> Mix.shell().info("#{name}: #{path}")
        {:error, reason} -> Mix.raise("could not fetch #{name}: #{inspect(reason)}")
      end
    end
  end
end
