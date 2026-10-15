# Qwen4Exp layer trace localization

`compare-qwen4exp-layer-traces.py` compares the labelled serial residuals and
pointer-sorted native arena rows at one history position. It finds a unique row
mapping by the longest identical hash/L2 prefix, then reports the first differing
layer and half. Layer numbers start at zero: half 0 is after attention/recurrent
mixing, and half 1 is after the feed-forward/expert block.

The real-model teacher fixture has a default-off diagnostic environment variable,
`TS_TEST_QWEN4EXP_BATCH_REBUILD_STEP`. Set it to the desired zero-based decode step,
along with `TS_Q4E_SPAN_TRACE=1`, to retire only that serial holder's cached graphs
and rebuild at its existing history position. The fixture preserves its teacher
token and rebinds the holder before forwarding. It records the rebuild step in
its report. Model/backend gating and numerical tolerances remain unchanged.

Example, with an already captured trace:

```powershell
python eng/validation/compare-qwen4exp-layer-traces.py `
  --log D:/validation/run/execution/process.log `
  --position 1501 --layers 48 --rows 2 --serial-id-prefix serial-2-0- `
  --output artifacts/qwen-layer-analysis.json
```

The parser requires all layer/half observations for every selected serial and
native row. Missing traces, repeated observations, malformed/nonfinite values,
and ambiguous row mappings cannot produce a localization. A log with multiple
native executions at the same position must first be separated into independent
runs; the parser does not guess which execution belongs to a serial cohort.
The input is bounded to 128 MiB. The output must resolve inside the workspace's
`artifacts/` or `docs/validation/` directory; external junction targets are not
implicitly authorized by a relative output name.

Exit 0 means the diagnostic produced a unique complete comparison, including
when it found a mismatch. Exit 2 means incomplete/ambiguous/invalid evidence.
Neither means a model passed validation. Keep the original process status and
numerical assertions separate. Tracing and graph rebuilding can alter allocation
layout and execution cost, so preserve untraced controls and do not use trace
timings for performance acceptance. Equal hashes are not a replacement for full
logit, semantic, or EOS checks.

CPU parser tests:

```powershell
python -m unittest discover -s eng/validation/tests -p test_compare_qwen4exp_layer_traces.py
```
# Attention tensor follow-up

`compare-qwen4exp-attention-dumps.py` compares the already-live tensors exported
by the optional native attention diagnostic. Supply the complete same-run log
and its dump directory. It requires unique residual-based row correspondence,
uses only the first solo invocation after each explicit serial rebuild label,
and rejects missing tensors, malformed shapes, changed payload hashes, path
escapes and invalid nonfinite values. Negative infinity is valid only in the
attention mask. Intermediate tensors marked `readable=0` are never compared.

```powershell
python eng/validation/compare-qwen4exp-attention-dumps.py `
  --log D:/validation/run/execution/process.log `
  --dump-root D:/validation/attention-dump `
  --output artifacts/qwen-attention-comparison.json
```

`freeze-qwen-teacher-variant.py` prepares a fresh copied teacher application
without rebuilding or executing it. Its required native SHA and explicit cache
root constrain the new copy; all other application bytes must match the frozen
baseline. Environment changes are explicit `--set-environment NAME=value`
arguments recorded in `preparation.json`. The baseline and previous execution
evidence remain intact. An enabled diagnostic may affect timing or memory
layout; compare its original numerical assertion outcome as well as the tensor
observations. None of these tools establishes semantic or performance acceptance.

For an explicitly separate serial snapshot, supply `--serial-log`,
`--serial-dump-root` and the requested `--serial-row` to the attention comparator.
Both input logs and payloads retain independent hashes. This mode additionally
requires matching first-prefill traces and all labelled serial layer/half
residuals, including the same teacher token. Missing rows remain missing; an
unlabelled solo or round-robin dump cannot substitute for a serial snapshot.
These checks support the stated cross-process diagnostic comparison and do not
prove that every unobserved state component is equal.
