# Qwen4ExpModel goldens

Rows the legacy `Qwen4ExpModel` produced on `Qwen4ExpRuntimeTests`' synthetic fixtures, captured from the
last tree with the legacy implementation (`c56b23a4` plus the runtime work up to its removal) by running the tests'
own `Run`, `DraftTrace` and `FeatureTrace` on it, F32 KV cache:

- `script_cpu`, `script_chunked_cpu`: the managed CPU backend, QSA-free fixture (the per-layer path has no QSA
  layers), one prompt and the same prompt in two chunks, then four decode steps. Deterministic across machines.
- `script_qsa_ggmlcpu`, `script_qsa_chunked_ggmlcpu`: `ggml_cpu`, the QSA fixture (dense below a top-k of 64), the
  native token span. The GGML runtime backends compare against these.
- `mtp_drafts_ggmlcpu`: the shared MTP head's prompt catch-up and two chained draft steps (`ggml_cpu`).
- `dflash_features_ggmlcpu`: a prompt's and a 3-row verify's DFlash feature rows and logits (`ggml_cpu`).

JSON arrays of float rows, shortest round-trip form. The runtime tests compare against these instead of
constructing the deleted implementation. The fixture builders must not change in ways that alter these
fixtures' bytes.
