defmodule Cholla.TrainTest do
  use ExUnit.Case, async: true

  alias Cholla.Train.{Cact, Config, CQ, Data, LoRA, Safetensors}

  @tools [
    %{
      "name" => "control_device",
      "description" => "Switch or toggle any named smart-home device.",
      "parameters" => %{
        "type" => "object",
        "properties" => %{
          "device" => %{"type" => "string"},
          "action" => %{"type" => "string", "enum" => ["on", "off", "toggle"]}
        },
        "required" => ["device", "action"]
      }
    },
    %{
      "name" => "set_temperature",
      "description" => "Set the thermostat.",
      "parameters" => %{
        "type" => "object",
        "properties" => %{"degrees" => %{"type" => "number"}},
        "required" => ["degrees"]
      }
    }
  ]

  @example %{
    "query" => "turn the kitchen lights off and set the thermostat to 68",
    "tools" => @tools,
    "answers" => [
      %{
        "name" => "control_device",
        "arguments" => %{"device" => "kitchen lights", "action" => "off"}
      },
      %{"name" => "set_temperature", "arguments" => %{"degrees" => 68}}
    ]
  }

  defp base_config do
    %Config{
      vocab_size: 8192,
      out_vocab: 8192,
      max_seq_len: 8192,
      kv_window: 256,
      engram_layers: [3, 7, 11, 15, 19],
      global_layers: [4, 9, 14, 19]
    }
  end

  describe "Data.render/1" do
    test "renders the chat template with the model's key order" do
      {prompt, target} = Data.render(@example)

      assert prompt ==
               "<|im_start|>user\n<tools>" <>
                 Data.encode_json(@tools) <>
                 "</tools>\nturn the kitchen lights off and set the thermostat to 68<|im_end|>\n<|im_start|>assistant\n"

      assert String.starts_with?(
               prompt,
               ~s(<|im_start|>user\n<tools>[{"name":"control_device","description":"Switch or toggle any named smart-home device.","parameters":{"type":"object","properties":{"action":)
             )

      assert target ==
               ~s(<tool_call>[{"name":"control_device","arguments":{"action":"off","device":"kitchen lights"}},{"name":"set_temperature","arguments":{"degrees":68}}]</tool_call><|im_end|>)
    end

    test "system and reasoning" do
      {prompt, target} =
        Data.render(%{
          "query" => "hi",
          "tools" => [],
          "answers" => [],
          "system" => "be terse",
          "reasoning" => "nothing fits"
        })

      assert String.starts_with?(
               prompt,
               "<|im_start|>system\nbe terse<|im_end|>\n<|im_start|>user\n<tools>[]</tools>\nhi<|im_end|>\n"
             )

      assert target == "<think>\nnothing fits\n</think>\n<tool_call>[]</tool_call><|im_end|>"
    end
  end

  test "Data.normalize/1 converts single-turn chat lines" do
    line = %{
      "messages" => [
        %{"role" => "system", "content" => "sys"},
        %{"role" => "user", "content" => "lights off"},
        %{
          "role" => "assistant",
          "tool_calls" => [
            %{
              "function" => %{
                "name" => "control_device",
                "arguments" => ~s({"device":"lights","action":"off"})
              }
            }
          ]
        }
      ],
      "tools" => [%{"type" => "function", "function" => hd(@tools)}]
    }

    assert %{"query" => "lights off", "system" => "sys", "tools" => [tool], "answers" => [call]} =
             Data.normalize(line)

    assert tool["name"] == "control_device"

    assert call == %{
             "name" => "control_device",
             "arguments" => %{"device" => "lights", "action" => "off"}
           }

    assert Data.normalize(%{"messages" => [%{"role" => "user", "content" => "x"}]}) == nil
  end

  test "Data.engram_indices/2 matches the engine's hash" do
    rows = Data.engram_indices([2, 4, 17, 300, 8191, 5, 0, 0], base_config())
    slots = 18432
    bare = for row <- rows, do: Enum.map(Enum.with_index(row), fn {v, ti} -> v - ti * slots end)

    assert bare == [
             [16327, 58, 741, 5968, 9933, 16853],
             [5410, 2216, 10303, 17694, 6527, 144],
             [13345, 15213, 13302, 12641, 15744, 6340],
             [12921, 6541, 8683, 12928, 6349, 15594],
             [573, 10853, 15864, 3302, 7500, 3380],
             [3738, 9128, 13411, 6165, 15202, 13092],
             [16880, 15325, 10850, 5762, 11964, 15835],
             [10891, 7446, 16797, 14987, 10910, 11421]
           ]
  end

  test "Data.calls_of/1 and canonical calls" do
    text = ~s(<tool_call>[{"name":"a","arguments":{"x":1,"y":"z"}}]</tool_call><|im_end|>)
    assert Data.calls_of(text) == [{"a", [{"x", 1.0}, {"y", "z"}]}]
    assert Data.calls_of("no calls here") == []
    assert Data.calls_of("<tool_call>[oops</tool_call>") == nil

    assert Data.canonical_calls([%{"name" => :a, "arguments" => %{x: 1}}]) == [
             {"a", [{"x", 1.0}]}
           ]
  end

  describe "Config" do
    test "ladder order and rungs" do
      cfg = base_config()

      assert Config.ladder_order(cfg) == [
               0,
               19,
               9,
               14,
               4,
               6,
               11,
               16,
               2,
               7,
               12,
               17,
               1,
               3,
               5,
               8,
               10,
               13,
               15,
               18
             ]

      assert Config.ladder_layer_indices(cfg, 10) == [0, 2, 4, 6, 7, 9, 11, 14, 16, 19]

      rung = Config.ladder_config(cfg, 10)
      assert rung.num_layers == 10
      assert rung.global_layers == [2, 5, 7, 9]
      assert rung.engram_layers == [4, 6, 9]
      assert rung.ladder_order == [0, 9, 5, 7, 2, 3, 6, 8, 1, 4]
      assert Config.effective_kv_window(rung) == 256

      rung2 = Config.ladder_config(cfg, 2)
      assert rung2.global_layers == [1] and rung2.engram_layers == [1]
      assert_raise ArgumentError, fn -> Config.ladder_layer_indices(cfg, 1) end
    end

    test "geometry" do
      cfg = base_config()
      assert Config.effective_kv_window(cfg) == 256
      assert Config.engram_geometry(cfg) == {[2, 3], 3, 128}
      assert Config.hada_n(cfg) == 1024
      assert Config.hada_blocks(1024) == {32, 32}
    end
  end

  test "LoRA.learning_rate/4 is optax's warmup cosine schedule" do
    got = for step <- 0..8, do: LoRA.learning_rate(step, 1.0e-4, 2, 9)

    want = [
      0.0,
      5.0e-5,
      1.0e-4,
      9.504843910690397e-5,
      8.117449033306912e-5,
      6.112604751251638e-5,
      3.887394996127114e-5,
      1.8825499864760786e-5,
      4.951557457388844e-6
    ]

    for {g, w} <- Enum.zip(got, want), do: assert_in_delta(g, w, 1.0e-9)
  end

  test "Cact encode/parse round trip" do
    header = %{
      kv_window: 256,
      kv_bits: 8,
      vocab_size: 8192,
      out_vocab: 8192,
      d_model: 768,
      num_heads: 12,
      num_kv_heads: 2,
      num_layers: 20,
      qk_head_dim: 48,
      v_head_dim: 64,
      max_seq_len: 8192,
      hada_n: 1024,
      mhc_lanes: 4,
      sliding_window: 1024,
      global_layers: [4, 9, 14, 19],
      qkv_conv_taps: 3,
      engram_slots: 18432,
      engram_sub_dim: 128,
      num_engram_tables: 6,
      engram_conv_taps: 4,
      engram_conv_dilation: 3,
      engram_seed_heads: 0,
      engram_orders: [2, 3],
      engram_layers: [3, 7, 11, 15, 19],
      rope_theta: 100_000.0
    }

    codebook = Enum.map(1..28, &(&1 / 100))
    half = <<1.5::little-float-16, -2.0::little-float-16>>

    tensors = [
      {:fp16, {2}, half, 0, 0},
      {:cq, {4, 128}, :binary.copy(<<7>>, 4 * 64 + 4 * 2), 128, 4},
      {:raw, {}, "tok", 0, 0}
    ]

    {:ok, archive} = Cact.parse(Cact.encode(header, codebook, tensors))
    assert Map.drop(archive.header, [:num_tensors, :codebook_len]) == header
    assert archive.header.num_tensors == 3
    assert Enum.map(archive.codebook, &Float.round(&1, 6)) == codebook

    assert [
             %{dtype: :fp16, shape: {2}},
             %{dtype: :cq, shape: {4, 128}, group: 128, bits: 4},
             %{dtype: :raw}
           ] = archive.tensors

    assert Enum.all?(archive.tensors, &(rem(&1.offset, 64) == 0))
    assert Cact.floats(archive, hd(archive.tensors)) == [1.5, -2.0]
    assert Cact.blob(archive, List.last(archive.tensors)) == "tok"
    assert Enum.map(Cact.codebook(archive, 2), &Float.round(&1, 6)) == Enum.take(codebook, 4)

    assert Enum.map(Cact.codebook(archive, 4), &Float.round(&1, 6)) ==
             Enum.slice(codebook, 12, 16)

    assert Cact.parse("nope") == {:error, :not_a_cact}
  end

  test "Safetensors round trip with metadata" do
    path =
      Path.join(System.tmp_dir!(), "cholla-#{System.unique_integer([:positive])}.safetensors")

    tensors = %{
      "a/b" => Nx.tensor([[1.0, 2.0], [3.0, 4.0]], type: :f32),
      "c" => Nx.tensor([1.0], type: :f16)
    }

    :ok = Safetensors.write(path, tensors, %{"scale" => "2.0"})
    {:ok, read, meta} = Safetensors.read(path)
    assert meta == %{"scale" => "2.0"}
    assert Nx.to_flat_list(read["a/b"]) == [1.0, 2.0, 3.0, 4.0] and Nx.type(read["c"]) == {:f, 16}
    {:ok, entries, _} = Safetensors.read_header(path)
    assert entries["a/b"]["shape"] == [2, 2]
    File.rm!(path)
  end

  describe "CQ" do
    test "hadamard is orthonormal and pack4 packs nibbles LSB-first" do
      h = CQ.hadamard(8)
      assert Nx.to_flat_list(Nx.round(Nx.dot(h, h))) == Nx.to_flat_list(Nx.eye(8, type: :f32))
      idx = Nx.tensor([[1, 2, 3, 4, 15, 0, 7, 8] ++ List.duplicate(0, 120)], type: :u8)
      norm = Nx.tensor([[2.0]], type: :f32)
      packed = CQ.pack4(idx, norm)
      assert binary_part(packed, 0, 4) == <<0x21, 0x43, 0x0F, 0x87>>
      assert byte_size(packed) == 64 + 2
      assert binary_part(packed, 64, 2) == <<2.0::little-float-16>>
    end

    test "quantize reconstructs within W4 error and fake_quant_act rounds half to even" do
      key = Nx.Random.key(1)
      {w, _} = Nx.Random.normal(key, shape: {8, 256}, type: :f32)

      cb =
        Nx.tensor([
          -0.2395,
          -0.1807,
          -0.1410,
          -0.1090,
          -0.0815,
          -0.0565,
          -0.0329,
          -0.0101,
          0.0127,
          0.0357,
          0.0595,
          0.0850,
          0.1127,
          0.1447,
          0.1842,
          0.2423
        ])

      q = CQ.quantize(w, cb, CQ.hadamard(128))

      rel =
        Nx.to_number(
          Nx.divide(Nx.sqrt(Nx.sum(Nx.pow(Nx.subtract(q, w), 2))), Nx.sqrt(Nx.sum(Nx.pow(w, 2))))
        )

      assert rel < 0.3
      x = Nx.tensor([[127.0, 2.5, 3.5, -2.5, -3.5, 0.4]])
      assert Nx.to_flat_list(CQ.fake_quant_act(x)) == [127.0, 2.0, 4.0, -2.0, -4.0, 0.0]
    end
  end

  test "Data.encode/3 masks the prompt and pads" do
    tok = %Cholla.Train.Tokenizer{
      ids: %{"a" => 10},
      pieces: %{10 => "a"},
      scores: %{10 => 0.0},
      types: %{10 => 0},
      markers: [],
      add_dummy_prefix: false,
      byte_fallback: false,
      unk_id: 3
    }

    {ids, mask} = Data.encode(tok, %{"query" => "a", "tools" => [], "answers" => []}, 8)
    assert length(ids) == 8 and length(mask) == 8
    assert hd(ids) == 2 and hd(mask) == 0.0
    assert Enum.at(mask, -1) == 0.0
  end
end
