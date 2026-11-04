# Original NInfer target: HTTP quality and performance evaluation

`qwen38-ninfer-body-eval.py` reuses the unchanged `qwen38-engine-bench/3`
protocol (SHA `4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c`)
and the complete-program protocol (SHA
`4093b80c7401454344b67b2a4e421314b4a263024d79cd30eefe09f67bfae3cb`). It refuses
changed protocol bytes. It adds no questions, repairs, relaxed scoring or implicit
server configuration. Default PLAN writes command/provenance data and sends no
HTTP requests; `--execute` invokes the selected original protocol against one
already-running server. It never starts/stops a server or obtains a GPU lease.

Use the public model factory and normal server/runtime to evaluate chat template,
tokenization, sampling, scheduling, prefix state, streaming and tool parsing
together. The standalone full-logits body probe is a different diagnostic; it
does not establish these API behaviors.

Build a coherent runtime from the tested source. This is the TensorSharp launch
argument array; provide actual absolute module paths through
`TS_Q35_NINFER_PTX` and `TS_Q35_W8_HEAD_PTX`, and place the compiled NInfer
attention PTX in the normal CUDA-kernel search path:

```json
[
  "dotnet", "TensorSharp.Server.Host.dll",
  "--model", "/workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer",
  "--backend", "cuda", "--host", "127.0.0.1", "--port", "5088",
  "--no-webui", "--no-multi-agent", "--no-skills",
  "--temperature", "0", "--repeat-penalty", "1"
]
```

For the short matrix, explicitly set `MAX_CONTEXT=40960`,
`TS_Q35_UBATCH=2048`, `TS_SCHED_MAX_RUNNING_SEQS=4`; record all CUDA experiment
flags and leave incompatible GGUF options OFF. Use no `--draft-model`, GGUF
`--mmproj`, or W8 **companion** artifact settings. This is a complete text target,
with no NInfer draft or vision binding. The existing generic
`qwen38-engine-servers.sh` TensorSharp branch always supplies a GGUF mmproj, so
its unmodified launch arguments are unsuitable for this new format.

A known supported Ignis v0.9.4 target-only reference argument array is:

```json
[
  "/root/ignis/ignis-v0.9.4-x86_64-unknown-linux-gnu/ignis-server",
  "--profile", "none",
  "--model-artifact", "/workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer",
  "--server-bind", "127.0.0.1:8013", "--model-max-context", "40960",
  "--spec-decode-lanes", "4", "--model-kv-format", "bf16",
  "--model-rope-scaling", "none", "--model-prefill-chunk", "2048",
  "--model-prefill-attention", "fp16", "--model-exception-weights", "bf16",
  "--model-enable-thinking", "false", "--download-enabled", "false",
  "--spec-backend", "off"
]
```

These are launch configurations, not new execution evidence. Use them only in
separate exclusive windows after checking the actual executable's help and
capabilities. Pin its binary SHA and retain startup logs. The `fp16` prefill
attention option and TensorSharp's BF16 attention arithmetic can still differ;
the same weight file does not prove identical complete-model arithmetic.

Export the artifact's original frontend once, without a GPU:

```sh
dotnet NinferQwen38BodyProbe.dll --export-frontend \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --expect-directory-sha ACTUAL_DIRECTORY_SHA256 \
  --out /absolute/artifacts/ninfer-original-frontend
```

The export preserves all six original files, checks their SHA values and records
the actual directory SHA. Point `--tokenizer` at its `frontend/tokenizer.json`
for both engines. Do not substitute a differently exported HF tokenizer merely
because its model name matches. Preserve actual server usage as well; generated
visible-text token counts and server tokens have different scopes.

Prepare/run one arm (omit `--execute` to inspect the plan first):

```sh
python3 eng/validation/qwen38-ninfer-body-eval.py --execute \
  --engine tensorsharp --suite text --url http://127.0.0.1:5088 \
  --model ACTUAL_MODEL_ID --label ninfer-target-ts-text \
  --artifact /workspace/hf/ignis/qwen3_8_27b_nvfp4full-v2.ninfer \
  --known-artifact-sha abb1e120d5f1f32d61689604d238227ff579ab76cbd9319628f3b3904fffd9af \
  --tokenizer /absolute/artifacts/ninfer-original-frontend/frontend/tokenizer.json \
  --manifest /absolute/artifacts/actual-ts-server-manifest.json \
  --out /absolute/artifacts/ninfer-target-ts-text
```

Run the same suite in a separate Ignis window with `--engine ignis`, its actual
model ID, port 8013, corresponding actual manifest and a fresh output. The
wrapper hashes the supplied tokenizer/manifest, preserves stdout/stderr and raw
protocol results, records child exit status, and refuses changing inputs. The
supplied full artifact SHA is **operator evidence, not recomputed by this
wrapper**. An external before/after guard must prove source/runtime/artifact and
exclusive GPU identity, including unchanged upstream ggml. Adapter-specific
thinking/skills fields differ across engines; never claim identical complete
wire bodies from equal questions alone.

Suites retain their original behavior:

| Suite | Original scenarios and scope |
| --- | --- |
| `text` | 15 scenarios × 3 = 45 cases; ordinarily 141 requests, with actual agent turns/errors recorded. Includes math, GSM8K, JSON/schema, tools, multi-turn, prefill, long output and concurrency. |
| `short` | All 16 scenarios × 3 = 48 cases; ordinarily 156 requests. This includes five image requests per repeat. Current NInfer text-only refusals must remain errors/unavailable; they are not passing vision validation. |
| `long` | Exact 128K/250K targets, 131072/256000, three repeats each and original order/neighbor checks. Use explicit CTX262144 and cap1 on both servers. No shortened inputs or JSON-format repair is added. |
| `program` | Original ten complete Python programs × three repeats, 4096-token output cap; genuine complete-module execution with the original closed AST/import/builtin policy and limits. |

For complete-program comparisons, model IDs in the real requests may differ.
The original `--baseline` implementation requires identical model/body identity;
this wrapper does not bypass that requirement. Compare raw requests and original
program/functional-case results explicitly, retaining unexecuted inputs and
protocol refusals. A small finite program set does not prove broad coding quality.

For long inputs, separate capacity, response formatting, semantic correctness and
latency. BF16 KV can limit 250K on a 32GB device even when another compressed-KV
deployment fits. Preserve actual HTTP/CUDA allocation failures and give them no
answer-quality or speed credit. Never count an engine that could not run as a
passing or zero-second result. Only scenarios passing every original repeated
quality check may contribute performance measurements to a comparison.

CPU wrapper tests (no model/device/HTTP work):

```sh
python3 -m unittest eng/validation/test_qwen38_ninfer_body_eval.py -v
```
