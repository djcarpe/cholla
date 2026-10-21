defmodule Cholla.TrainEngineTest do
  # Needs the downloaded base archive: mix test --include engine
  use ExUnit.Case, async: false
  @moduletag :engine

  alias Cholla.Train.{Checkpoint, Tokenizer}

  setup_all do
    {:ok, base} = Cholla.Download.weights_path()
    {:ok, tok} = Tokenizer.from_cact(base)
    %{base: base, tok: tok}
  end

  test "tokenizer matches sentencepiece", %{tok: tok} do
    assert Tokenizer.encode(tok, "hello world") == [3833, 662, 1711]
    assert Tokenizer.encode(tok, "a\nb\tc") == [8096, 38, 8115, 37, 8105]
    assert Tokenizer.decode(tok, [3833, 662, 1711]) == "hello world"
    assert Tokenizer.encode(tok, "<tool_call>[]</tool_call>") == [10, 3331, 11]
    text = "Invoice from Acme Corp for $1,250.00, due March 3 2027. émigré"
    assert Tokenizer.decode(tok, Tokenizer.encode(tok, text)) == text
  end

  test "constants of the base archive", %{base: base} do
    {:ok, c} = Checkpoint.constants(base)
    assert length(c.codebook) == 28
    assert Nx.shape(c.cb4) == {16}
    assert Nx.shape(c.p1) == {1024} and Nx.shape(c.p2) == {1024}
    assert c.p1 |> Nx.to_flat_list() |> Enum.sort() == Enum.to_list(0..1023)
    assert c.header.num_layers == 20
  end

  test "build copies the base archive when nothing changes", %{base: base} do
    out =
      Path.join(System.tmp_dir!(), "cholla-base-copy-#{System.unique_integer([:positive])}.cact")

    assert {:ok, ^out, %{layers: 20}} = Cholla.Train.build(out: out, base_cact: base)
    assert File.stat!(out).size == File.stat!(base).size
    File.rm!(out)
  end
end

defmodule Cholla.TrainRunTest do
  # A real (tiny) fine-tune: mix test --include train. Compiles the training
  # graph once (about a minute on a CPU) and downloads the checkpoint.
  use ExUnit.Case, async: false
  @moduletag :train
  @moduletag timeout: :timer.minutes(30)

  @tools [
    %{
      "name" => "set_power",
      "description" => "Set the transmitter power level.",
      "parameters" => %{
        "type" => "object",
        "properties" => %{"level" => %{"type" => "string", "enum" => ["low", "high"]}},
        "required" => ["level"]
      }
    }
  ]

  test "finetune, build and evaluate" do
    dir = Path.join(System.tmp_dir!(), "cholla-train-#{System.unique_integer([:positive])}")
    File.mkdir_p!(dir)

    examples =
      for {q, level} <- [
            {"whisper", "low"},
            {"shout", "high"},
            {"go to a whisper", "low"},
            {"time to shout", "high"}
          ] do
        %{
          "query" => q,
          "tools" => @tools,
          "answers" => [%{"name" => "set_power", "arguments" => %{"level" => level}}]
        }
      end

    lines = []

    progress = fn line ->
      send(self(), {:progress, line})
      lines
    end

    assert {:ok, adapter, stats} =
             Cholla.Train.finetune(examples,
               out: Path.join(dir, "a.safetensors"),
               epochs: 1,
               batch_size: 2,
               max_len: 128,
               val_split: 0.0,
               progress: progress
             )

    assert File.exists?(adapter)
    assert is_float(stats.loss) and stats.steps == 2
    assert_received {:progress, "  data      4 examples  seq_len 128  cap 128"}

    {:ok, lora, meta} = Cholla.Train.LoRA.read(adapter)
    assert map_size(lora) == 5 and meta.rank == 16 and meta.scale == 2.0
    assert Nx.shape(lora["stack/layers/block/self_attn/q_proj/kernel"].a) == {20, 768, 16}

    assert {:ok, cact, %{layers: 20, tensors: 574}} =
             Cholla.Train.build(lora: adapter, out: Path.join(dir, "a.cact"))

    {:ok, archive} = Cholla.Train.Cact.read(cact)
    assert archive.header.num_layers == 20 and archive.header.kv_window == 256
    # the confidence head is dropped: no heads.manifest after final_norm, only the tokenizer
    assert Enum.count(archive.tensors, &(&1.dtype == :raw)) == 1

    assert {:ok, rung, %{layers: 4}} =
             Cholla.Train.build(lora: adapter, layers: 4, out: Path.join(dir, "a4.cact"))

    {:ok, archive4} = Cholla.Train.Cact.read(rung)
    assert archive4.header.num_layers == 4 and archive4.header.engram_layers == [3]
    assert archive4.header.global_layers == [1, 2, 3]

    assert {:ok, %{total: 4, correct: correct}} = Cholla.Train.evaluate(cact, examples)
    assert correct in 0..4
    File.rm_rf!(dir)
  end
end
