defmodule Mix.Tasks.Cholla.Finetune do
  @shortdoc "Trains a LoRA adapter on JSONL examples"
  @moduledoc """
  Trains a LoRA adapter for Needle on a JSONL file of examples, in Elixir.

      mix cholla.finetune data.jsonl --out adapter.safetensors
      mix cholla.finetune data.jsonl --epochs 5 --lr 2e-4 --lora-rank 8

  Options: `--out`, `--checkpoint`, `--base-cact`, `--epochs`, `--batch-size`,
  `--lr`, `--lora-rank`, `--lora-alpha`, `--max-len`, `--val-split`, `--seed`.
  See `Cholla.Train.finetune/2`.
  """
  use Mix.Task

  @switches [
    out: :string,
    checkpoint: :string,
    base_cact: :string,
    epochs: :integer,
    batch_size: :integer,
    lr: :float,
    lora_rank: :integer,
    lora_alpha: :float,
    max_len: :integer,
    val_split: :float,
    seed: :integer
  ]

  @impl true
  def run(args) do
    {opts, rest} = OptionParser.parse!(args, strict: @switches)
    Mix.Task.run("app.start")

    case rest do
      [path] ->
        opts = Keyword.put(opts, :progress, fn msg -> Mix.shell().info(msg) end)

        case Cholla.Train.finetune(path, opts) do
          {:ok, out, _stats} -> Mix.shell().info("  next      mix cholla.build --lora #{out}")
          {:error, reason} -> Mix.raise("fine-tuning failed: #{inspect(reason)}")
        end

      _ ->
        Mix.raise("usage: mix cholla.finetune data.jsonl [options]")
    end
  end
end
