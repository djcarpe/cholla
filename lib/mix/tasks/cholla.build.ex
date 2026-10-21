defmodule Mix.Tasks.Cholla.Build do
  @shortdoc "Builds a .cact archive from the base checkpoint and a LoRA adapter"
  @moduledoc """
  Writes a `.cact` the engine loads, merging a LoRA adapter and/or taking a
  shallower rung of the base model.

      mix cholla.build --lora adapter.safetensors --out my.cact
      mix cholla.build --layers 10 --out needle3-10.cact

  Options: `--lora`, `--layers`, `--out`, `--checkpoint`, `--base-cact`.
  See `Cholla.Train.build/1`.
  """
  use Mix.Task

  @switches [
    lora: :string,
    layers: :integer,
    out: :string,
    checkpoint: :string,
    base_cact: :string
  ]

  @impl true
  def run(args) do
    {opts, _} = OptionParser.parse!(args, strict: @switches)
    Mix.Task.run("app.start")
    opts = Keyword.put(opts, :progress, fn msg -> Mix.shell().info(msg) end)

    case Cholla.Train.build(opts) do
      {:ok, out, _info} ->
        Mix.shell().info("  next      Cholla.agent(tools: tools, weights: #{inspect(out)})")

      {:error, reason} ->
        Mix.raise("build failed: #{inspect(reason)}")
    end
  end
end
