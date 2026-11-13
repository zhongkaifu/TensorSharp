# Qwen35Model goldens

Rows the legacy `Qwen35Model` produced on `Qwen35RuntimeTests`' synthetic fixtures, captured at commit
`6a361f8c` (the last commit with the legacy implementation) by running the tests' own `Script`,
`DraftTrace` and `MediaScript` on it, F32 KV cache. `*_cpu.json` ran on the managed CPU backend
(deterministic across machines); `*_ggmlcpu.json` on `ggml_cpu`, where the legacy model ran Bonsai2's
PRISM rotations (the managed backend refuses them). JSON arrays of float rows, shortest round-trip form.

The runtime tests compare against these instead of constructing the deleted implementation. Regenerate
only by checking out `6a361f8c` and rerunning the capture; the fixture builders must not change in ways
that alter these fixtures' bytes.
