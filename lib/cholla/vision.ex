defmodule Cholla.Vision do
  @moduledoc """
  Vision with Cactus Compute's Mirage model (Cactus 4): one local pass over
  an image returns structured perception as JSON — a scene description,
  objects with boxes and masks, OCR text lines, spatial relations, a short
  reasoning trace and, when asked, a constrained form.

  Images are packed 8-bit RGB, row-major, `width * height * 3` bytes. Decode
  them with an image library first (Mirage resizes to its own grid, so
  bigger images only cost memory). The Mirage weights (`mirage.cact`) are
  downloaded on first use, or taken from `config :cholla,
  vision_weights_path:`.

  Calls run on the vision pool (`config :cholla, vision_pool: name`, else the
  default pool) or the one given as `pool:`, on the least busy instance. A
  pass takes seconds on a server CPU, so give vision its own pool to keep it
  from queueing in front of text calls. See `Cholla.Pool`.
  """

  alias Cholla.{Engine, NIF, Pool}

  @capacity 1_048_576

  @doc """
  Describes an image.

  ## Options

    * `:options` - a map of Mirage options, sent as JSON: `sections`
      (`["scene", "objects", "text", "relations", "reasoning", …]`),
      `fields` (`["bbox", "mask", …]`), `conf`, `mask_format`
      (`"largest" | "merged" | "polygons" | "bitmap"`), `form`. `%{}` (the
      default) returns everything
    * `:language` - the OCR language, e.g. `"en"`; detected when omitted
    * `:keywords` - words to bias OCR towards
    * `:capacity` - the first output buffer size in bytes (default 1 MB;
      retried once at the size the engine asks for)
    * `:pool` - the `Cholla.Pool` to run on
  """
  @spec describe(binary(), pos_integer(), pos_integer(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def describe(rgb, width, height, opts \\ [])
      when is_binary(rgb) and is_integer(width) and is_integer(height) do
    cond do
      width <= 0 or height <= 0 ->
        {:error, :bad_dimensions}

      byte_size(rgb) != width * height * 3 ->
        {:error, {:bad_rgb_size, byte_size(rgb), width * height * 3}}

      true ->
        with {:ok, options} <- encode(Keyword.get(opts, :options, %{})) do
          run(rgb, width, height, options, opts)
        end
    end
  end

  @doc """
  Fills a constrained form: `form` maps each field to a JSON-Schema object,
  `%{"enum" => [...]}`, `%{"type" => "integer", "minimum" => a, "maximum" => b}`,
  `%{"type" => "string"}` or a box reference. Takes the options of
  `describe/4`; the form is merged into `:options`.
  """
  @spec extract(binary(), pos_integer(), pos_integer(), map(), keyword()) ::
          {:ok, map()} | {:error, term()}
  def extract(rgb, width, height, form, opts \\ []) when is_map(form) and map_size(form) > 0 do
    options = opts |> Keyword.get(:options, %{}) |> Map.put("form", form)
    describe(rgb, width, height, Keyword.put(opts, :options, options))
  end

  @doc "Loads Mirage into every instance of the vision pool now instead of on first use."
  def preload(opts \\ []), do: Pool.preload(vision_pool(opts), [:vision])

  defp run(rgb, width, height, options, opts) do
    capacity = Keyword.get(opts, :capacity, @capacity)
    keywords = Cholla.Speech.keywords(opts[:keywords])

    Pool.op(vision_pool(opts), nil, fn state ->
      case Engine.ensure_vision(state) do
        {:ok, state} ->
          result =
            NIF.describe(
              state.instance,
              rgb,
              width,
              height,
              options,
              opts[:language],
              keywords,
              capacity
            )

          {decode(result), state}

        error ->
          Engine.reply(error)
      end
    end)
  end

  defp encode(options) when is_map(options) do
    {:ok, JSON.encode!(options)}
  rescue
    e -> {:error, {:bad_options, Exception.message(e)}}
  end

  defp encode(options) when is_binary(options), do: {:ok, options}
  defp encode(other), do: {:error, {:bad_options, other}}

  defp decode({:ok, json}), do: JSON.decode(json)
  defp decode(error), do: error

  defp vision_pool(opts), do: opts[:pool] || Pool.vision_default()
end
