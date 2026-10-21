# Engine and speech tests download the Needle engine and weights (~55 MB) on
# first run. Enable them with: mix test --include engine --include speech
ExUnit.start(exclude: [:engine, :speech])
