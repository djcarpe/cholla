# Engine, speech and vision tests download the Cactus engine and weights (~75 MB) on
# first run. Enable them with: mix test --include engine --include speech --include vision
# Training tests (--include train) also download the 242 MB checkpoint.
ExUnit.start(exclude: [:engine, :speech, :vision, :train])
