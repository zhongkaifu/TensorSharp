# Bonsai-8B Q1_0 — actual Apple Metal lifecycle checks

**4 passed,0 failed,0 skipped** on Apple M5Pro Metal, using r4 native `b1ed8e429e1408cb5372807f09c21cf6b46c9bca41a5259ea4715d8b92b3c1f4`. Each test verifies the actual mapped native hash. Checkpoint SHA256 `284a335aa3fb2ced3b1b01fcb40b08aa783e3b70832767f0dd2e3fdfa134bd54`; managed assembly/test source identities are in [inputs.json](inputs.json).

- Single and two-sequence paged batching match their scalar top1 results; these assertions do not compare every logit.
- Truncate and device-residency release/reload preserve the full vocabulary, observed maximum difference0 under the existing1e-4 limit.
- A2,050-token request grows its holder, releases it, and returns to a usable primary holder.

The first attempt failed4/4 before model load because the test module defaulted to the CPU backend. Its TRX remains. The corrected invocation sets `TS_TEST_GGML_BACKEND=metal`, `TS_TEST_MODEL_DIR` to the exact Bonsai file, `MAX_CONTEXT=4096`, and `TS_KV_INITIAL_TOKENS=64`.

These are real trained-model state checks, not quality or speed measurements. They do not clear the original Bonsai8 four-request TTFT regression or the Bonsai27 concurrent-output failure.

The fixture previously returned successfully when weights were missing. It now uses a discovery-time model gate, accepts an explicit model file, and fails an explicitly configured directory without a matching model. Missing weights therefore no longer masquerade as executed cases. Production source is frozen r4; only the fixture is changed in the isolated managed tree. Upstream ggml remains clean at `456172ec733a135778adcd32d00e576a58232e45`.
