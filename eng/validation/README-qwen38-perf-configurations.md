# Reproducing the Qwen3.8-27B comparison configurations

This page records an earlier **ggml CUDA** campaign with the revisions and
configuration templates below. The current **native CUDA** engine evaluation
uses `--backend cuda` and the `Qwen35CudaModel` direct engine; see
[its reproduction guide](README-qwen38-engine-evaluation.md). These are separate
campaigns. Keep their results, source revisions and dependency evidence separate.

Use [the replay harness](README-qwen38-agent-replay.md) for fixture generation,
quality checks, paired comparisons and timing definitions. This page describes
configuration inputs; the existence of a configuration does not mean its run
completed or passed. Keep run results, configuration copies and model/source
manifests in ignored `artifacts/` or `docs/validation/`.

## Sources and model inputs

The baseline source is `a0341c6787ad24792fd7c02b024ac7f0d28ac185`. Candidate
changes are based on that revision; record the candidate commit or a manifest
of modified **and untracked** source files in addition to the base revision.
A base revision alone does not identify a candidate build. The native dependency
used for validation is unchanged upstream ggml
`da8e2a54b919557b816b1c2beb432ccb766f7ce0`. Build baseline and candidate in
separate directories and record the actual DLL/native-library hashes. Do not
patch ggml or reuse a baseline native binary as candidate evidence.

The model files used by the configurations are:

| File | Repository and revision | SHA-256 |
| --- | --- | --- |
| `Qwen3.8-27B-NVFP4-MTP-N4_0.gguf` | `akopytko/Qwen3.8-27B-NVFP4-GGUF` at `a5fcd490e745749e3cbf6ee1729bb0ceb28a5800` | `d5659ecb29adaf75f313fdae03eb7ff17baf3f986f0c3b1348e8afda1670c9a8` |
| `mmproj-BF16.gguf` | Same repository/revision | `de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8` |
| `Qwen3.8-27B-DFlash2-Q4_K_M.gguf` | `z-lab/Qwen3.8-27B-DFlash2-GGUF` at `2d9571f8ce46e151f61c6499c99dee6079e1d610` | `1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd` |
| `qwen3_8_27b_nvfp4.ninfer` | `neroued/Qwen3.8-27B-nvfp4-NInfer` at `a107ba1b5b0609d9ca90d5ed61439f5f5cc7d64d` | `74d2c57145e6ff11d1d2faa79594477f9bc903a611af1fb20218189fbbb77d82` |

The public NInfer artifact differs from the private artifact named in the
screenshots. Its main weights mix FP8/NVFP4 and its DFlash2 weights use Q8;
five draft MLP projections differ from the screenshot artifact's documented
NVFP4 draft MLPs. The GGUF main projections are NVFP4, with Q6_K output and Q4_0
embedding weights, and the separate draft uses Q4_K_M. The tested main GGUF has
unit sidecar scales; synthetic native scale fixtures cover nonunit sidecars.
These assets do not establish identical logits, acceptance rates or output
lengths across engines. Compare actual output quality separately from latency.

## Immutable workload

Generate once, then reuse the exact JSONL file and its manifest for every run:

```sh
python3 eng/validation/qwen38-agent-replay.py prepare \
  --output artifacts/qwen38-perf/replay-1008.jsonl \
  --steps 30 --lanes 4 --max-tokens 128 \
  --image-width 1344 --image-height 768
```

The seven default cases contain 47 requests per lane: 3 short, 3 prefill8k,
3 repeat19k, 4 incremental, 30 agent34, 3 image_replace and 1 arithmetic.
Concurrency waves `1,2,4` therefore plan 329 requests; `1` plans 47 and `1,2`
plans 141. A missing wave is missing coverage. The image geometry targets 1008
visual tokens; inspect actual prompt usage and image-token expansion for each
engine. Prompt padding is not the full context length.

The fixture sets temperature 0, repeat penalty 1 and thinking disabled, including
the template setting. The output cap is 128. Assistant/tool history is fixed;
the last six prior screenshot turns are retained. At step 7 and beyond, removal
of old history can move a cache hit back to the public prefix. This is a
synthetic, prescribed-history replay, not the unavailable private sessions from
the screenshots. Steps are sequential within each lane. The server remains
running between cases and waves, so step 0 is not necessarily a cold request.

## TensorSharp configuration

Adapt absolute paths to the host. The following is a **configuration template**
for the Q8/THP candidate. Put it in an ignored directory, add verified
`source_revision`, source-manifest identity and `model_manifest`, and use a new
output directory for every run:

```json
{
  "engine": "tensorsharp",
  "label": "candidate-q8-hugepages",
  "cwd": "/workspace/tensorsharp",
  "fixture": "/workspace/tensorsharp/artifacts/qwen38-perf/replay-1008.jsonl",
  "url": "http://127.0.0.1:5088",
  "model": "Qwen3.8-27B-NVFP4-MTP-N4_0",
  "concurrency": "1,2,4",
  "ggml": "/workspace/tensorsharp/ExternalProjects/ggml",
  "environment": {
    "MAX_CONTEXT": "32768",
    "TS_KV_INITIAL_TOKENS": "128",
    "TS_RETAINED_FUSED_CACHE_MAX": "2",
    "TS_SCHED_NUM_BLOCKS": "512",
    "TS_PREFIX_CACHE_ISOLATION": "shared",
    "TS_GGML_HOST_HUGEPAGES": "1"
  },
  "binaries": [
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll",
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/TensorSharp.Models.dll",
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/TensorSharp.Runtime.dll",
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/TensorSharp.Chat.dll",
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/libGgmlOps.so"
  ],
  "argv": [
    "/workspace/dotnet/dotnet",
    "/workspace/tensorsharp/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll",
    "--model", "/workspace/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf",
    "--mmproj", "/workspace/models/mmproj-BF16.gguf",
    "--draft-model", "/workspace/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf",
    "--backend", "ggml_cuda", "--host", "127.0.0.1", "--port", "5088",
    "--no-multi-agent", "--no-skills", "--temperature", "0",
    "--repeat-penalty", "1", "--prefill-chunk-size", "2048",
    "--spec-draft", "3", "--kv-cache-dtype", "q8_0"
  ]
}
```

Use these explicit overrides when constructing other configurations:

| Label | KV | Initial tokens | Isolation | Host huge pages | Draft window | Wave selection |
| --- | --- | ---: | --- | --- | ---: | --- |
| `baseline-tuned` | f16 | 24576 | baseline's existing lineage behavior | off | 3 | 1,2,4 planned |
| `candidate-lineage` | f16 | 24576 | `lineage` | `0` | 3 | 1,2,4 planned |
| `candidate-shared` | f16 | 24576 | `shared` | `0` | 3 | 1,2,4 planned |
| `candidate-f16-hugepages` | f16 | 128 | `shared` | `1` | 3 | 1,2 |
| `candidate-q8-hugepages` | q8_0 | 128 | `shared` | `1` | 3 | 1,2,4 |
| `candidate-q8-draft7` | q8_0 | 128 | `shared` | `1` | 7, plus `--spec-pmin 0` | 1 |

Baseline paths must point to the separate baseline build. Its original code
does not implement the new isolation or THP options. For current-code default
controls, explicitly set `TS_PREFIX_CACHE_ISOLATION=lineage` and
`TS_GGML_HOST_HUGEPAGES=0`; omission can inherit a shell setting.
The draft-7 case also disables the confidence cutoff, so it is a separate
configuration, not a one-variable window experiment. A configured maximum of
3 does not mean every verify call drafts exactly 3 tokens. Compare accepted
tokens per verification as well as the confidence cutoff and proposed count.

`lineage` is the default. `shared` is an explicit opt-in for a **trusted
single-tenant server**: matching private prefixes share retention and cache-hit
timing across stateless requests using the same API session. Named sessions
remain separate. Do not present shared-mode measurements as default behavior.
See [cache scope and checkpoint policy](README-qwen-chat-cache-benchmark.md).

THP is independently opt-in, Linux CUDA only, for owned anonymous mappings at
least 32 MiB. It does not modify kernel-wide policy. Accepted advice does not
guarantee physical huge pages, and reclaim/compaction under pressure can worsen
tail latency. Record both off/on settings and kernel policy; consult the
[THP probe and limitations](README-host-huge-pages.md).

On visible SM12 devices, the candidate defaults `GGML_CUDA_PDL=0` when it is
unset, retaining CUDA graphs. Explicit environment values are honored. Record
the launch value **and** the native startup diagnostic; unset at launch does
not mean PDL remains enabled. This is an owned integration workaround, with
the dependency unchanged. See [native validation](README-qwen-native-scales.md).

## Velox configuration

The comparison build used Velox revision
`ccec38c0fadafe4e0d2d4b753044847dcaa30bbe`. Supply the same fixture, wave selection
and fresh output directory to the runner. Set `engine` to `velox`, `model` to
`qwen3.8-27b`, `url` to `http://127.0.0.1:8010`, and list the built executable in
`binaries`. Its command array corresponds to:

```sh
/workspace/velox-inferer/target/release/velox serve \
  --artifact /workspace/models/qwen3_8_27b_nvfp4.ninfer \
  --host 127.0.0.1 --port 8010 --model-id qwen3.8-27b \
  --max-context 32768 --max-concurrency 4 --kv-format nvfp4 \
  --prefill-chunk 2048 --host-context-mib 16384 --ctx-checkpoints 1 \
  --force-tool-call on --vram-budget 30720 --enable-thinking false \
  --dflash-drafts 7 --mtp-drafts 0
```

The validated host used `LD_LIBRARY_PATH=/usr/local/cuda-13.1/lib64`; adapt this
to the installed runtime. Velox's forced-tool policy, NVFP4 KV and draft settings
differ from TensorSharp's. This is an operational comparison of the stated
configurations, not isolation of engine implementation cost with identical
weights, quantization, prompt tokens and output tokens.

## Launch and interpretation

```sh
python3 eng/validation/run-qwen38-perf.py \
  --config artifacts/qwen38-perf/configs/candidate-q8-hugepages.json \
  --output-dir artifacts/qwen38-perf/candidate-q8-new-run
```

Use a controlled launch environment and run one GPU workload at a time. The
runner records an explicit allowlist of relevant inherited and overridden
settings under `benchmark_launch_environment`, including unset entries; it
does not dump arbitrary environment variables or credentials. This snapshot
describes launch inputs, not values subsequently defaulted inside native code.
Older reports without it cannot prove the absence of inherited overrides;
retain that limitation rather than reconstructing the past from today's shell.

Check `provenance.json` for lifecycle and cleanup status and `replay.json` for
all planned requests. A completed report can still contain OCR/tool failures.
Interrupted, OOM or missing requests are not passes. Strict paired comparison
requires identical fixture hashes and request-key sets; a partial baseline
cannot qualify a full candidate run. Paired passing subsets may be shown as
exploratory only, with exclusions counted for both engines. Tool-call TTFT can
include frontend buffering, and prose transport checks do not establish prose
quality. No speed claim is implied by the commands in this document.
