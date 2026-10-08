# Gemma repetition diagnosis

This fresh-process probe records the checkpoint's complete chat template, final
prompt IDs, BOS/EOG configuration, tensor-type inventory, output IDs and decoded
text. It uses the production `KVCachePromptRenderer` with `GgufPromptRenderer`.
The default question is `请详细介绍最终幻想7`. It never adds repetition penalties,
silently changes the prompt or stops at a detected loop. A 1,024-token generation
can cross the checkpoint's sliding window (E4B: 512; 12B: 1,024, including
prompt positions); context shifting is disallowed.

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
