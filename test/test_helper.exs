# Engine and speech tests download the Needle engine and weights (~55 MB) on
# first run. Enable them with: mix test --include engine --include speech
# Training tests (--include train) also download the 242 MB checkpoint.
ExUnit.start(exclude: [:engine, :speech, :train])
