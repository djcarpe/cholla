defmodule Cholla.MixProject do
  use Mix.Project

  @version "0.1.0"

  def project do
    [
      app: :cholla,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_clean: ["clean"],
      deps: deps(),
      description: "Elixir bindings to Cactus Compute's Needle on-device model",
      package: package(),
      docs: [main: "readme", extras: ["README.md"]]
    ]
  end

  def application do
    [
      extra_applications: [:logger, :inets, :ssl, :public_key, :crypto],
      mod: {Cholla.Application, []}
    ]
  end

  defp deps do
    [
      {:elixir_make, "~> 0.9", runtime: false},
      {:ex_doc, "~> 0.34", only: :dev, runtime: false}
    ]
  end

  defp package do
    [
      licenses: ["Apache-2.0"],
      files: ~w(lib c_src Makefile mix.exs README.md .formatter.exs),
      links: %{"Needle" => "https://cactuscompute.com/needle"}
    ]
  end
end
