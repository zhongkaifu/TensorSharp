# Maya exact-input baseline

`maya-token-baseline.py` runs an independently built Project Maya executable on
the exact prompt token IDs from a completed `DirectCudaModelProbe` quality row.
It does not build, download, or modify Maya/ggml. Only the owned process is ended.

Prepare a JSON plan containing:

```json
{
  "source_path": "C:/Works/project-maya",
  "source_revision": "<full clean revision>",
  "exe": "D:/isolated-maya-build/strata.exe",
  "exe_sha256": "<sha256>",
  "pack": "E:/models/model/maya-pack",
  "pack_evidence": "C:/Works/TensorSharp/artifacts/run/maya-pack.json",
  "pack_evidence_sha256": "<sha256>",
  "direct_report": "C:/Works/TensorSharp/artifacts/run/direct.json",
  "direct_report_sha256": "<sha256>",
  "case": "math",
  "repetitions": 2,
  "max_new": 128,
  "max_context": 1024,
  "prefill": "auto",
  "vocab_size": 154880,
  "eos_ids": [154820, 154827, 154829],
  "environment": {
    "CUDA_VISIBLE_DEVICES": "0",
    "STRATA_GLM_CPU_LANE": "8",
    "STRATA_GLM_SPLIT": "0",
    "STRATA_GLM_NO_MTP": "1",
    "STRATA_GLM_NO_SPEC": "1",
    "STRATA_GLM_NO_REUSE": "1",
    "STRATA_GLM_SLOTS": "0",
    "STRATA_GLM_USAGE": "0",
    "STRATA_GLM_LOOP_THINK": "0",
    "STRATA_GLM_LOOP_ANSWER": "0",
    "STRATA_GLM_PREFILL_TAIL_SKIP": "0"
  }
}
```

The pack evidence must record successful pack creation (`exit_code: 0`),
`original_checkpoint_stat_unchanged` and `source_unchanged`, every derived file's
`path`/`bytes`/`sha256` in `generated_files`, original shards'
`path`/`bytes`/`mtime_ns` in `model_stat_after`, and the prior whole-checkpoint
manifest's `path`/`sha256` in `original_checkpoint_identity`. The original model
files are stat-checked; their complete payloads are not rehashed on each run.

```powershell
python eng/validation/maya-token-baseline.py --plan artifacts/run/maya-plan.json --output artifacts/run/maya-results --timeout 1800
python -m unittest discover -s eng/validation/tests -p test_maya_token_baseline.py -v
```

Use a new output directory. The plan must identify a successful reference row
using the production parser. Maya's production tokenizer must reproduce the
reference prompt IDs exactly. Its production output parser checks final answer,
reasoning closure and absence of tool calls. Reasoning text and full output token
parity are separate fields; different correct wording in reasoning does not
automatically fail the answer-only task. Every passing row must end with an
observed checkpoint EOS, normal completion and matching wire token counts.

`prefill` defaults to `"auto"`, preserving the original runner behavior. A
positive integer sets Maya's explicit prefill workspace token capacity. For the
pinned GLM implementation, a 32-token prompt sends 31 tokens to the optional
batch path and handles the final token separately. Its default minimum batch
size is 32, so this prompt falls back to token-by-token prefill even though auto
has reserved a large batch workspace. To test the batch path explicitly, use
`"prefill": 32` together with `"STRATA_GLM_PREFILL_MIN": "1"`. Record both
changes and inspect the native log; `"STRATA_GLM_PREFILL_VERBOSE": "1"` prints
the completed batch token count and existing expert staging counters. Its output
cost remains in request timing. Do not treat a setting alone as proof that the
batch path executed. Keep the original run and use a new output directory.

`process_success`, `quality_passed`, and `performance_observation_complete` are
separate. An exit code of zero from Maya alone proves none of the latter two.
The runner returns 0 for qualified quality, 2 for completed processes with a
quality failure, and 1 for execution/identity failures. Complete failed rows
are retained and the remaining repetitions continue.

Maya's compatibility pack rounds some small projection weights to BF16. This is
the same source checkpoint with different execution arithmetic, not a claim of
byte-identical derived weights. Preserve `compat-bf16.json`. The optional
terminal-prefill optimization skips outputs without readers; the conservative
example explicitly disables it. MTP and its approximate draft expert selection
are disabled. RAM uses Maya's automatic headroom unless the plan explicitly
sets its RAM-tier cap; that cap is not a process-wide physical hard limit.

The one-second sampler records RSS/commit, process I/O, host availability and
board VRAM. It can miss peaks, includes other processes in board VRAM, and has
unquantified overhead. The actual executable is checked at READY; shared CUDA
module identities are not enumerated. Hashes of derived weights are checked
outside timings and can warm filesystem cache. Sequential first/second requests
are not an independent balanced A/B experiment. Request one is not a cold-disk
measurement. `post_prefill_forward_tokens_per_second` uses `generated - 1`:
Maya's first sample consumes prefill logits and does not perform a decode
forward, while its DONE decode timing includes all sampling and output overhead.
