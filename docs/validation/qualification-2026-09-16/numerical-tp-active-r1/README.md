# Compact active CTA experiment

All 36 original-oracle exact gate/up and strict full-MoE comparisons pass. All 13 scratch-growth, injected failure/recovery and graph reuse cases pass. Peak owned scratch falls from 45,651,456 to 33,723,904 bytes. The active interval skips unowned CUDA thread blocks while preserving the original full-shape K partitions and reduction ordering.

The paired warmed benchmark still regresses narrow strips by 14–32%; two-rank geometry is approximately 0.96–1.00 times baseline. Each case retains 60 alternating measurements. Other VM campaigns were active. This does not qualify performance or a full checkpoint. This experiment includes an isolated graph cache; the selected production candidate omits that graph cache and therefore requires separate exact-source tests and timings. Rejected 32-row and 64-row subdivisions are not integrated.
