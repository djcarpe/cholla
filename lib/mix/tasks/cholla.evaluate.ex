defmodule Mix.Tasks.Cholla.Evaluate do
  @shortdoc "Scores a .cact on JSONL examples through the engine"
  @moduledoc """
  Runs every example through the engine with the given weights and counts
  exact tool-call matches.

      mix cholla.evaluate my.cact holdout.jsonl

  See `Cholla.Train.evaluate/3`.
  """
  use Mix.Task

  @impl true
  def run(args) do
    {opts, rest} = OptionParser.parse!(args, strict: [max_new_tokens: :integer, misses: :boolean])
    Mix.Task.run("app.start")

    case rest do
      [cact, data] ->
        case Cholla.Train.evaluate(cact, data, Keyword.take(opts, [:max_new_tokens])) do
          {:ok, %{correct: c, total: t, accuracy: a, misses: misses}} ->
            Mix.shell().info("  accuracy  #{c}/#{t} exact (#{Float.round(a * 100, 1)}%)")

            if opts[:misses] do
              for %{query: q, want: want, got: got} <- misses do
                Mix.shell().info(
                  "  miss      #{inspect(q)}\n    want #{inspect(want)}\n    got  #{inspect(got)}"
                )
              end
            end

          {:error, reason} ->
            Mix.raise("evaluation failed: #{inspect(reason)}")
        end

      _ ->
        Mix.raise("usage: mix cholla.evaluate model.cact data.jsonl [--misses]")
    end
  end
end
