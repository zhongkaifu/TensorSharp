# Local Flash Next, image and agent validation

These commands reuse the production CLI/server and the existing quality probes.
They are a validation plan, not a record that model inference passed. Keep all
outputs under ignored `artifacts/` or `docs/validation/`, and run GPU jobs serially.

## Check the actual checkpoints first

```powershell
python eng/validation/local-model-preflight.py --model-root C:/Works/models --expected eng/validation/local-multimodal-models.json --hash --require-verified --output artifacts/multimodal/integrity.json
python eng/validation/tests/test_local_model_preflight.py
```

The preflight reads bounded headers, validates tensor extents and split metadata,
then hashes complete files sequentially. It does not mmap or execute a model.
Without `--hash` the result is structural evidence only. A SHA without an
independent expected value is `identity_only`; `--require-verified` refuses it.
File length, sparse-file flags and download progress records alone never pass
the integrity gate. The historical `.ranges.json` interpretation uses explicit
`chunk_bytes`, or labels its legacy 64 MiB assumption in the report.

The Flash Next first shard legitimately contains metadata and **zero tensors**.
The other two shards contain 568 and 656 tensors, totaling the declared 1,224.
Point the loader at `Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf`; all three
files must be adjacent. Its projector is the same directory's `mmproj-BF16.gguf`.
These four files total 75,446,298,720 bytes. The four Qwen-Image Q4 files total
11,051,668,216 bytes. Neither total is a runtime memory estimate.

The supplied alternative `qwen-image-2.1-Q8_0.gguf` is excluded from the expected
manifest until an independent source SHA is established. Use explicit local
Q4 companion paths for validation: a config whose Q8 filename points to a Q4
download URL must not choose the benchmark's quantization implicitly.

## Single GPU scope and memory limits

The local RTX 3080 Laptop has 16 GiB VRAM and the host has approximately 32 GiB
RAM. Flash Next's 74.5 GB trunk uses existing mapped GGUF weights, CPU-offloaded
experts and optionally the selected-expert CUDA cache. This is distinct from
the dense Q8/F16 `WeightStreamingExecutor` and does **not** establish a hard
shared `MemoryBudget` ceiling for mmap/OS page-cache memory. Start with one
request, short prompts, `--cpu-moe` and a conservative expert cache, then record
placement logs, process memory, VRAM and time before increasing concurrency.
Its existing real-cache probe supports final-vocabulary comparisons and exact
prompt/generated token tracking; see [the probe guide](Qwen4ExpExpertCacheProbe/README.md).

Qwen-Image releases native resident weights/scratch between text conditioning,
denoising and VAE stages. Its Q4 DiT is 4.19 GB, text encoder 5.03 GB, VAE 0.68 GB
and editing projector 1.16 GB. Activations depend on image geometry and number
of references. This makes a single-GPU small-image run a reasonable first test,
not a guarantee that the 2048-square default fits. Neither architecture uses
the newly validated dense streaming path. Multi-GPU is unavailable locally and
must remain untested until a usable second device is supplied.

## Same-hardware Flash Next reference

Verify the reference binary's version before loading. The local historical
llama.cpp binary at revision `9558fa44c` does not contain the `qwen4exp`
architecture; its presence and CUDA support cannot qualify it for Flash Next.
An unchanged source checkout at `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`
does contain that architecture. Build a separate output directory and retain
the binary/DLL hashes, compiler version, architecture and CUDA Flash Attention
settings. Do not replace another experiment's reference binary.

The reference's `--cpu-moe`/`--n-cpu-moe` flags place expert tensors on CPU.
They do not enable TensorSharp's compact selected-expert GPU cache. Its Flash
Next PLE table uses lazy mapped row access. A useful baseline keeps the same
GGUF, CPU-thread count, F16 KV cache, context, no speculation and exact prompt
tokens, while reporting the placement/cache policy difference explicitly.

For a TensorSharp trained sample, run the existing expert-cache probe with
`--generation greedy --decode-tokens 100 --warmup 0 --iterations 1`, using:

> Explain how a computer works to a curious beginner. Describe the CPU, memory,
> storage, input, and output in five connected paragraphs with a concrete example.

After starting the qualified reference server on port 5099 with the same first
shard, `--cpu-moe --n-gpu-layers all --ctx-size 512 --parallel 1` and explicit
thread counts, replay the measured TensorSharp report:

```powershell
python eng/validation/qwen38-llama-reference.py --prompt-report artifacts/multimodal/flash-long/model.json --server http://127.0.0.1:5099 --max-new 100 --output artifacts/multimodal/llama-flash-long
```

`--prepare-only` creates the exact request without inference. For the short
arithmetic case use `--max-new 32 --expected-text 42`; this requires EOS and
the exact answer. A 100-token length-limited continuation can measure a longer
decode interval, but remains an incomplete answer and never automatically
passes semantic quality. Retain full text, token IDs and server timings, and
repeat in rotated order on idle hardware after all builds complete. OS page
cache history remains uncontrolled; do not call those repetitions cold reads.

For a text-only selected-expert cache comparison, prepare and inspect the
2/6/8 GiB sequence, then run it on idle hardware. The script alternates cap order
across rounds, uses eight CPU threads explicitly, and keeps every process's full
model report, sampled board VRAM, working set, generated text and termination:

```powershell
python eng/validation/flash-expert-cache-matrix.py --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --output artifacts/multimodal/cache-matrix-plan --prepare-only
python eng/validation/flash-expert-cache-matrix.py --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --output artifacts/multimodal/cache-matrix
```

An oversized cap may fail; preserve that evidence. This text-only experiment
does not qualify those caps for vision or large-prefill workspaces. A future
automatic policy must admit those phases before their allocations, rather than
waiting until decode to notice reduced free VRAM.

## Text and image understanding

Start a production server with the first Flash Next shard, matching projector,
`--backend ggml_cuda --cpu-moe`, and a port such as 5098. For example, set
`TS_HOST_MOE_EXPERT_CACHE_MB=2048` before launching to try the existing optional
cache. Preserve its actual engagement and placement log; requested settings
alone are not proof that a cache or device was used.

Use existing `imgs/banner_1.png` for real-image OCR (largest text: `TensorSharp`).
The CLI image probe accepts `--tp 1` and either an apphost executable or a DLL
in `--cli`; DLLs use `dotnet` (override its location with `--dotnet`):

```powershell
$env:TS_CPU_MOE = '1'
$env:TS_HOST_MOE_EXPERT_CACHE_MB = '2048'
python eng/validation/qwen38-cuda-vision.py --cli TensorSharp.Cli/bin/TensorSharp.Cli.dll --model C:/Works/models/Qwen3.8-Flash-Next-GGUF/Qwen3.8-Flash-Next-UD-IQ1_M-00001-of-00003.gguf --mmproj C:/Works/models/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf --image imgs/banner_1.png --mode ocr --tp 1 --ocr-max-tokens 128 --report-dir artifacts/multimodal/banner-ocr
python eng/validation/qwen38-parallel-vision.py --prepare --fixtures artifacts/multimodal/vision-fixtures
python eng/validation/qwen38-parallel-vision.py --url http://127.0.0.1:5098 --profile ggml_cuda --fixtures artifacts/multimodal/vision-fixtures --concurrency 1 --scenarios image_ocr,multi_image,image_follow_up --output artifacts/multimodal/vision-c1.json
```

The two generated cards test exact digits/color, attachment ordering and image
history. Preserve their manifest hashes. Once C=1 passes, repeat identical
requests at C=2 with `--baseline artifacts/multimodal/vision-c1.json`. Exact card
checks are narrow visual correctness tests, not broad visual-quality scores.
The CLI image probe hashes the platform-native deployment file beside the CLI
(`GgmlOps.dll` on Windows). This identifies the deployment, not a sampled process
module. Preserve loader logs if a search-path override can select another file.

## Image generation and editing

First run a 256-square, one-step execution smoke, then separately evaluate
512-square, 40-step outputs. The smoke cannot pass the quality case. Repeat the
quality run with the same seed/settings for a same-build repeatability control.

```powershell
python eng/validation/qwen-image21-bench.py --engine tensorsharp --models-dir C:/Works/models/qwen-image-2.1 --cli TensorSharp.Cli/bin/TensorSharp.Cli.dll --backend ggml_cuda --width 256 --height 256 --steps 1 --cfg 1 --seed 42 --output artifacts/multimodal/image-smoke
python eng/validation/qwen-image21-bench.py --engine tensorsharp --models-dir C:/Works/models/qwen-image-2.1 --cli TensorSharp.Cli/bin/TensorSharp.Cli.dll --backend ggml_cuda --width 512 --height 512 --steps 40 --cfg 1 --seed 42 --repeat 2 --prompt "A red ceramic teapot beside a blue cup on a wooden table, soft daylight." --output artifacts/multimodal/image-quality
python eng/validation/qwen-image21-bench.py --engine tensorsharp --models-dir C:/Works/models/qwen-image-2.1 --cli TensorSharp.Cli/bin/TensorSharp.Cli.dll --backend ggml_cuda --mode edit --image imgs/banner_1.png --width 512 --height 512 --steps 40 --cfg 1 --seed 42 --prompt "Change the starry background to a sunny blue sky. Preserve the person and all written text." --output artifacts/multimodal/image-edit
```

Inspect output images against the prompt: object identity/count, requested color,
spatial relation, visible artifacts, and preservation during edits. Pixel
statistics and finite output alone cannot establish semantic quality. The mask
probe `qwen-image21-mask-bench.py` can additionally enforce **zero changed
protected RGBA pixels** on a real edit, using the same source image and an explicit
mask. That exact invariant does not prove that the selected area was edited well.
No external reference-engine parity is claimed by `--engine tensorsharp`.

## Agent actions

Use the running Flash Next server. Generic declared tool calls are tested with
streaming and thinking on/off, including actual tool-result round trips:

```powershell
python eng/validation/validate-qwen38-tool-calls.py --url http://127.0.0.1:5098 --thinking off,on --output artifacts/multimodal/tool-calls.json
python eng/validation/validate-release-agent-workflows.py --url http://127.0.0.1:5098 --concurrency 1 --scenarios skill_selection,skill_run --output artifacts/multimodal/agent-skills.json
```

Start the server with `--skills-dir eng/validation/fixtures/skills` for the skill
cases. For actual file/code/shell work, enable the production execution features
and use the remaining workflow scenarios. Preserve tool events, tool return
codes, written files and independent execution checks; a model's final claim
that code ran is not evidence.

For a Windows server, explicitly configure `--code-exec --code-exec-unconfined`
and the fixture skills directory. The production Windows job object does not
provide filesystem/network confinement. The following commands test functional
execution and record that limitation; they do not claim sandbox isolation.
`--target-shell` describes the server, so select it explicitly even when the
validation client runs on another OS. PowerShell uses `Write-Output`; `cmd` is
also supported. Default POSIX prompts are unchanged.

```powershell
python eng/validation/validate-release-agent-workflows.py --url http://127.0.0.1:5098 --target-shell powershell --sandbox-off --concurrency 1 --scenarios skill_script_run,shell_run,code_generation_run,code_edit_run --output artifacts/multimodal/agent-actions.json
python eng/validation/verify-agent-code-artifacts.py --workflow-report artifacts/multimodal/agent-actions.json --artifact-store <server-retained-artifact-directory> --sandbox-off --output artifacts/multimodal/agent-independent.json
```

The same serial skill/action/independent-code sequence is available as one
reusable client command. It connects to an already running server, retains each
phase report even when an earlier phase fails, and does not change server flags:

```powershell
python eng/validation/run-flash-agent-quality.py --url http://127.0.0.1:5098 --output artifacts/multimodal/agent-suite --artifact-store TensorSharp.Server.Host/bin/code-artifacts --unconfined --target-shell powershell
```

Use the actual configured artifact directory. Add `--tool-calls` for the separate
declared-client-tool suite with thinking on/off. The default timeout is 900
seconds per workflow request; a timeout is a failure, not a quality result.

The independent verifier reads the last advertised source snapshot, rejects
paths outside the server's retained artifact directory, and executes additional
function inputs and exact result-type checks. Use the server's configured local
artifact store, not its request workspace or an unrelated download directory.
On Linux, omit `--sandbox-off` to use its default `bwrap` confinement. Missing
confinement never causes an automatic unconfined fallback. After C=1 passes,
repeat with `--concurrency 2 --distinct-inputs` to expose crossed request state;
thinking mode is a separate `--thinking` run with the same oracles.

CPU checks for the adapters and evidence rejection rules:

```powershell
python -m unittest discover -s eng/validation/tests -p test_windows_validation_adapters.py
python -m unittest discover -s eng/validation/tests -p test_qwen38_cuda_vision.py
python -m unittest discover -s eng/validation/tests -p test_remote_agent_fixtures.py
```

Report prefill, decode and end-to-end timings separately, with exact generated
token counts and termination reason. Truncated answers and incomplete agent
round trips fail their quality cases. Retain failures/OOMs/timeouts, actual
native/managed hashes and the unchanged ggml revision; never turn missing
hardware, models, reference engines or sandbox support into passing scenarios.
