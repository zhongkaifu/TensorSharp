# Gemma repetition diagnosis

This fresh-process probe records the checkpoint's complete chat template, final
prompt IDs, BOS/EOG configuration, tensor-type inventory, output IDs and decoded
text. It uses the production `KVCachePromptRenderer` with `GgufPromptRenderer`.
The default question is `请详细介绍最终幻想7`. It never adds repetition penalties,
silently changes the prompt or stops at a detected loop. A 1,024-token generation
can cross the checkpoint's sliding window (E4B: 512; 12B: 1,024, including
prompt positions); context shifting is disallowed.

For a hard wall-clock bound, invoke the compiled DLL through
`eng/validation/run-bounded-probe.py --timeout 600 --output FRESH_PROCESS_DIR --
dotnet PATH_TO_PROBE.dll ...`. The wrapper records process exit/timeout and kills
only the process it started. A timed-out run is incomplete, even if partial
tokens look correct. Both engine and direct modes persist token progress every
32 tokens; engine mode also writes partial decoded text. Native/managed binary
identities are persisted immediately after model loading, before generation.
`llama-reference.py --timeout 600` bounds its HTTP request budget; use the same
external process wrapper for a strict deadline and shut down the separately
owned llama server afterward.

Modes are `metadata` (no model execution), `direct` (public `ModelBase.Forward`)
and `engine` (a single request through `InferenceEngine`, default scheduler
settings, explicitly disabled speculation and repetition termination).
The direct mode records finite full-row checks, top-20 logits/probabilities,
entropy, token positions/timings and exact repeated suffixes. `--logits true`
writes all little-endian F32 vocabulary rows; 1,024 rows of a 262,144-token
vocabulary take 1 GiB. `--teacher FILE` replays raw token IDs from a JSON array,
this probe's report or llama.cpp's `completion.json`.

The default `--sampling production` applies the checkpoint's
`tokenizer.ggml.suppress_tokens` generation contract. Raw logits are still
recorded before masking, and each row reports both raw and production argmax.
`--sampling raw` explicitly bypasses that contract in direct mode for arithmetic
diagnosis; it is not a production quality baseline. Engine mode always applies
the model contract. Only declared IDs are suppressed: tool, channel and EOG
tokens remain available unless the checkpoint explicitly excludes them.
Direct-mode top probabilities, entropy and chosen-token probability describe
the raw, unmasked logits. They are labelled separately from production token
selection; suppressing IDs changes that sampling distribution.
`--temperature`, `--top-k`, `--top-p` and `--seed` are explicit options;
their defaults are `0`, `0`, `1`, and `17`. Repetition, frequency and presence
penalties remain disabled. A separate run with the model card's recommended
sampling settings must be reported separately from greedy reproduction; better
sampled prose alone does not establish a kernel correction.

Exit zero means the requested execution finished without a runtime error. It
does **not** certify language quality or agreement with an independent model
implementation. An exact-period repeated suffix is diagnostic evidence, not
proof of its cause. Read the entire answer, including whether it develops the
requested topic, maintains factual consistency and reaches a sensible ending.
For this question, verify actual FFVII content such as Cloud, Shinra/Avalanche,
Midgar, Sephiroth and the original-versus-remake distinction; merely mentioning
the title is insufficient. Short output, maximum-token termination, or a diverse
opening followed by a loop must not be reported as a quality pass.

Build in the coordinated managed build window, after independently building
current native sources. Keep upstream ggml unchanged and archive its revision
and clean status, model hash, loaded native hash and build logs with each run.

```powershell
dotnet build eng/validation/GemmaRepetitionProbe -c Release --no-incremental `
  -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true
$probe = 'eng/validation/GemmaRepetitionProbe/bin/Release/net10.0/GemmaRepetitionProbe.dll'
$model = 'artifacts/unified-memory-adaptive/models/gemma-4-12b-it-UD-IQ2_M.gguf'
dotnet $probe --model $model --output artifacts/gemma-iq2/metadata --mode metadata
dotnet $probe --model $model --output artifacts/gemma-iq2/direct --mode direct `
  --backend ggml_cuda --max-new 1024 --context 2048 --logits true
dotnet $probe --model $model --output artifacts/gemma-iq2/engine --mode engine `
  --backend ggml_cuda --max-new 1024 --context 2048
```

Run an independently built llama.cpp server with **the same GGUF**, one slot,
no draft/MTP model and sufficient context, only after TensorSharp releases the
GPU. Record the server's binary hash and startup log. A local example is below;
use `Start-Process -WindowStyle Hidden` if launching it in the background.

```powershell
& C:/Works/llama.cpp/build-ninja-cuda/bin/llama-server.exe `
  --model $model --host 127.0.0.1 --port 8087 --ctx-size 2048 --parallel 1 `
  --n-gpu-layers 99 --flash-attn on --cache-type-k f16 --cache-type-v f16
python eng/validation/GemmaRepetitionProbe/llama-reference.py `
  --prompt-record artifacts/gemma-iq2/metadata/prompt.json `
  --output artifacts/gemma-iq2/llama --max-new 1024
# Stop the reference server before this replay:
dotnet $probe --model $model --output artifacts/gemma-iq2/teacher --mode direct `
  --teacher artifacts/gemma-iq2/llama/completion.json --logits true
```

The helper independently renders/tokenizes the question, preserves any prompt
mismatch, then captures a completion from the exact TensorSharp prompt IDs with
greedy sampling and all penalties disabled. Prompt agreement, early/late
teacher-conditioned probability differences, nonfinite values and where a loop
begins distinguish tokenizer, arithmetic, state/cache and sampler hypotheses.
Top-20 agreement is not a full-logit oracle. If necessary, use independent
llama.cpp layer/projection evidence and a dequantized mathematical oracle.

`compare-teacher.py` compares allowed argmax and pairwise logit gaps from matched
histories, excluding model-suppressed IDs from both top lists. Softmax log
probabilities and logits share these pairwise gaps even when a suppressed raw
token dominates the denominator. `llama-teacher.py` independently captures one
prediction for each exact teacher prefix, which also permits CPU-versus-CPU
triangulation without allowing a different sampled token to change the history.

The requested Unsloth `gemma-4-12b-it-UD-IQ2_M.gguf` reproduction uses revision
`fc034cfff751157913579611efad8462ac1be606` and SHA256
`4bd2461d35398dbcf5f3d5f0c9ad91cac78ae35b556e3a81f315a0cc0815ae8c`.
A QAT Q4 checkpoint is a separate quality control, not a reproduction of that
file. Keep all generated evidence ignored. The independent llama sampler also
honors GGUF suppression even with all user penalties disabled; compare its
sampled tokens to production argmax and its raw top-logprobs to raw logits.

`TS_GEMMA4_DIAGNOSTIC_PAD_LOCAL_KV=1` is an arithmetic diagnostic, disabled by
default. It rounds the local CUDA prefill cache view to 256 rows (bounded by the
physical cache) while keeping the original valid-token count and causal mask.
Padding positions remain masked with negative infinity. This can change the
CUDA grouped-query flash dispatch without changing the logical context. Run
fresh default and diagnostic processes against the same teacher prefixes and
record both; a diagnostic result does not replace the normal quality baseline.
Persistent single-token decode already uses padded windows. The switch does
not affect the separate fresh/extended-window path used after SWA overflow.

## Local IQ2 and same-source Q4 observations

The IQ2 case remains a quality failure. Production sampling with the model's
suppression contract fixed still repeated during the 1024-token greedy run;
recommended sampling at temperature 1, top-k 64 and top-p 0.95 also failed
semantic review for seeds 1/17/42. One seed reached EOS with invented facts,
while the other two reached 1536 tokens with repeated material. Neither EOS
alone nor avoiding an exact token-period loop certifies an answer.

A fresh, independently built llama.cpp `4ebdf2c74acce30883d8e34b7c70b3eb8146f2fe`
also produced a degraded 1024-token IQ2 answer. This was checked separately from
the older `9558fa44c` executable. Twenty teacher-conditioned allowed argmax
positions agreed with TensorSharp, but that narrow comparison does not establish
full numerical correctness or fix long generation. Padding local KV changed
logits without resolving the quality issue and remains diagnostic-only.

The same Unsloth revision's `gemma-4-12b-it-Q4_K_M.gguf` provides a closer
control than the earlier QAT file. Its SHA256 is
`0a270ec9fe6b34f4a0d33992b6135117b484ebc4766ab76b51d4ae8c457e4c42`
(7,121,861,440 bytes). Tensor names/shapes and metadata match the IQ2 file except
for quantization/file type. On the same 19 final prompt IDs, context 2048,
F16 KV, no speculation, greedy sampling and no user penalties, both current
llama and the TensorSharp production engine completed 1024-token executions.
Both outputs were substantially more coherent than IQ2 and did not develop
the observed repetitive tail. Both stopped mid-answer at the length cap,
however, and contained factual/wording problems; neither is recorded as a
complete-answer quality pass. The Q4 improvement narrows the investigation but
does not prove the IQ2 checkpoint is solely responsible or exclude a shared
low-bit arithmetic problem.

These are language-quality observations, not speed measurements. The actual
TensorSharp Q4 run used native SHA `6b1cedc432f651854bba3ad802bd84cc87bd29290a01d06d4056e0ecc6d9e46d`;
the independent server executable SHA was
`752d225eefbebd366d9a0a1a491b3dc03261795f637d7052cb2c43cee0525e70`.
Full prompts, token histories, text, termination, actual loaded identities and
bounded process outcomes remain ignored under
`artifacts/unified-memory-adaptive/gemma-{iq2,q4-k-m}/`.
