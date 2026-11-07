# DeepSeek V4.1 Flash (`deepseek41`)

> **Multi-GPU selection:** use `--layer-split N` for whole-layer placement or a supported `--tp N` tensor-parallel mode. With neither mode configured, the default is one device. Add `--tp-node-id` and `--tp-peers` to span nodes on `ggml_cuda` (layer split or TP), or with `--layer-split N` on `cuda` (the direct-CUDA engine's stages; see [Backends](#backends)); N is the local GPU count. See [multi-node setup and limitations](../multi-node-inference.md). Older measurements below retain their original topology and do not validate the new distributed layer path.

[← back to model index](README.md) | [中文](deepseek41_zh-cn.md)

TensorSharp has a dedicated **V4.1 inference graph on `ggml_cuda`**, with an
optional native vision encoder. It uses the DeepSeek whole-model loader and
scheduler, with V4.1-specific attention, Engram, residual connections, and chat handling. This card describes
the implemented path and its limits. Model-quality and performance claims need
the measured artifacts tracked in the [validation report](../deepseek41_validation.md).
The same graph also loads on `--backend ggml_cpu`, which exists to run and check
the architecture without a GPU rather than to serve it — see
[Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend). A second
GPU-free option carries no ggml and no native library at all: `--backend cpu`
runs V4.1 on the pure-C# `DeepSeek4CpuExecutor`, which implements this same
graph in managed code and is held to the PyTorch oracle. Both are correctness
and portability paths, not serving paths — see [Backends](#backends).

The [official model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
declares `DeepseekV41ForCausalLM`. Its text network has 40 layers, hidden size
5120, 64 query heads with 512 components, 384 routed experts with top-6 routing,
one shared expert, and a 2304-wide expert intermediate. It declares a
1,048,576-token context. V4.1 differs from V4 in ways that affect every forward
pass; changing the GGUF architecture name to `deepseek4` is invalid.

## Download the repaired Q2_K/Q5_K checkpoint

For new downloads, use `Q2_K-Q5/` from
[smalinin/DeepSeek-V4.1-Flash-GGUF](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/tree/d1de55c19f95172c882906cc83c0e55932d26a63/Q2_K-Q5),
pinned to `d1de55c19f95172c882906cc83c0e55932d26a63`. Keep all ten shards
together and give TensorSharp the first shard. The files total **335,382,014,624
bytes (312.349 GiB)**: mixed Q2_K/Q3_K backbone and experts, Q5_K Engram tables,
80 F32 mHC matrices and four BF16 Engram gates.

The original seven-shard `vcruz305` Q2_K artifact quantized those 84 sensitive
tensors to Q2_K. It is no longer recommended for quality validation; the
[publisher's repair report](https://huggingface.co/smalinin/DeepSeek-V4.1-Flash-GGUF/blob/2c525d63b9ba5319185c93637f00d70fea55b44f/Q2_K/Q2_REPAIR_REPORT.md)
explains the affected tensors. Adding Engram metadata to an old shard does not
repair its tensor precision.

The repaired package's headers have been checked for all 1,046 tensor names and
shapes, the sensitive tensor types, and matching tokenizer/Engram metadata.
All ten downloaded shards passed full-file SHA-256 verification. Bounded plain
and DSpark HTTP probes passed with both `--layer-split 2` and experimental
routed-expert TP (`--tp 2`) on `ggml_cuda`. Each of the four
processes passed three text checks and one image OCR/color check through EOS,
then shut down cleanly. Separate plain/DSpark text and image pairs matched all 24
token IDs and `max_tokens` finishes in both modes, with active DSpark and clean
exit. These bounded continuations are separate from the HTTP EOS checks. Image
DSpark was slower in both measured pairs. Short repeated warm text controls
(one warmup and three measured 69-prompt/24-output-token pairs per mode) also
completed with exact token IDs and finishes, active DSpark, and clean exit.
**These small-sample, paging-constrained checks do not establish broad quality or
performance qualification, a speedup, multi-node execution, or full-model TP.**
The historical measurements below do not qualify this quantization. Local evidence
is under `docs/validation/model-matrix-20260927/deepseek41/`
(not committed). Verify the hashes below for each new download.

**The GGUF already includes Engram.** TensorSharp reads its token map, hash
multipliers, bucket primes and offsets, and padding ID directly from the GGUF
metadata, alongside the learned Engram weight tensors. No Engram generation,
separate Engram file, or tokenizer/config download is needed for text inference.

The loader validates the embedded layout against the Engram tensor dimensions
and rejects missing or malformed constants. There is no legacy file fallback or
Engram path override. Engram table placement and page warming still apply to
the learned weights; they do not generate hash constants.

From the repository root, on the machine storing the model:

```bash
python3 -m venv /workspace/dsv41-tools
/workspace/dsv41-tools/bin/python -m pip install huggingface_hub
/workspace/dsv41-tools/bin/hf download smalinin/DeepSeek-V4.1-Flash-GGUF \
  --revision d1de55c19f95172c882906cc83c0e55932d26a63 \
  --include "Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-*.gguf" \
  --local-dir /workspace/models/deepseek41-q2-q5
```

`hf download` preserves the `Q2_K-Q5/` subdirectory. Verify every complete file
before inference; the SHA-256 values below are the pinned publisher LFS identities.
Run this outside timed benchmarks, since it reads all 312.349 GiB:

```bash
(cd /workspace/models/deepseek41-q2-q5/Q2_K-Q5 && sha256sum --check - <<'SHA256'
8126b49dfcfde02cb3db24b6f98d56b031f0a34eaca2509fc7b2b9d362cf0ac8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf
22bb293aee509a348ce32a739e006fa41f2348c6bcfafa3be76a3ee079eadf96  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00002-of-00010.gguf
db894848b4f14d42c39e18faa907c737cd4850f2fcb9deff9d5ebd86f580aaf7  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00003-of-00010.gguf
655a3400f2c092d6e3c11b8b18bf319b29563e59b960337115266574321953e3  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00004-of-00010.gguf
4b9378c6819d1130517e8719026b34e5257f1f73bd1d50cf6f30243982100bac  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00005-of-00010.gguf
04f161084d82032c65247c02e6169784a757be7db9e068b0baba6833125b6bb8  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00006-of-00010.gguf
b1b1bf3cfbbc7388ce49b5ced69c42a13c7c5c5d3d609ac86902897a08d3c88a  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00007-of-00010.gguf
fae7f35123ae3557034a541507bb9fc24fb62c2e16ff8441c32e8e0477743d5d  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00008-of-00010.gguf
781fd69e9dd17c09676865830523a2d077a97544d6a004429aab95d2570f8534  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00009-of-00010.gguf
fa5affb1f971cd6e7effad5684b73780472b486a46330cd7c5776f87c38fea97  DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00010-of-00010.gguf
SHA256
)
```

Stop if any hash fails. `eng/dsv41-verify-download.py` is a historical verifier
for the original seven-shard layout; it does not verify this ten-shard package.
The two Q5_K Engram tables total about 125.889 GiB. They cannot stay resident in
a 57.74 GiB RAM allowance, even before CPU experts; use
`TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=0` on such a host and report
paging costs. Choose CPU expert offload from actual device capacity, including
the optional drafter and vision encoder.

**Historical benchmark provenance:** earlier results in this card and the complete-file SHA-256 record
`docs/validation/deepseek41/checkpoint-sha256.json` (local validation evidence, not committed)
refer to revision `8e0c4de3cb6519bfc11ed69dc87184b457a57bb5` and its older
first shard in the original `vcruz305` release. Later seven-shard examples used
`58d8ac86298fdf85a2440defee08b1abcad32e45`; the old Q4_K_M placement evidence is
also separate. Retained historical launch commands identify those files, not
the repaired package. Do not attribute their hashes, 246.35 GiB size, approximately
60 GiB Engram warming, quality results or throughput to Q2_K-Q5. Record the
revision and all shard hashes with every new validation result.

## Unsloth's DeepSeek-V4.1-Flash-1017 GGUFs

[unsloth/DeepSeek-V4.1-Flash-1017-GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF/tree/60ad3651641b5c97679c807e064ab1d7a3ca1455)
packages the 1017 checkpoint for llama.cpp. TensorSharp loads the tested
`UD-IQ2_XXS` files as published on `--backend ggml_cuda` (and `ggml_cpu`). Their
layout differs from the one above in four ways, handled by the native loader:

- The Engram tables are one tensor, `per_layer_token_embd.weight` (Q5_K, in its
  own shard), described by `deepseek41.ple.*` metadata. The loader slices it
  into the per-layer tables and compresses the raw `engram.pad_token_id`
  through the token map, as `engram.py` does.
- `blk.N.engram_gate.weight` stores Engram's `q_weight * k_weight` product, the
  only form in which the reference uses the two.
- The DSpark drafter `dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf` is a `dflash`
  file. Its noise token, Markov rank, block size, expert counts and target
  layers come from the target's `deepseek41.dspark.*` metadata, and a drafter
  that disagrees with them is refused.
- The vision tower ships as llama.cpp's `mmproj-BF16.gguf` / `mmproj-F16.gguf`
  (`clip`, projector `deepseek41v`), which `--mmproj` loads directly; the image
  tokens' routing biases come from the text shards (`blk.N.exp_probs_b_vl.bias`).

The managed executors, `--backend cuda` and `--backend cpu`, read the Engram
part of this layout the same way: each layer's slice of the combined table
becomes a view of the shard that holds it (nothing is copied), and
`engram_gate` scales the query while the key keeps its plain RMS norm. The
1017's routed experts still keep these files off `cpu`. `cuda` decodes all of
them while they stay in VRAM: a 10-layer truncation of the `UD-IQ2_XXS` file
has run on it under the
[per-stage oracle](#per-stage-oracle-against-ggml_cuda), and the whole file has
run on two nodes of two A40s each against `ggml_cuda`
([Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40)); see
[Backends](#backends). The completed loading and distributed controls cover
`UD-IQ2_XXS` on `ggml_cuda` and `cuda`; other quantizations need separate validation. At revision `60ad3651…`, `UD-Q4_K_XL` contains Engram type 43,
which the tested unchanged ggml `3612c0af…` rejects. Its load failed; see the
[validation report](../deepseek41_validation.md).

Download one `--include` pattern per call (several patterns in one call matched
only the root files with `hf` 2.2.0):

```bash
REPO=unsloth/DeepSeek-V4.1-Flash-1017-GGUF
REV=60ad3651641b5c97679c807e064ab1d7a3ca1455
DIR=/workspace/DeepSeek-V4.1-Flash-1017-GGUF
for pattern in "UD-IQ2_XXS/*" dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf mmproj-BF16.gguf; do
  hf download $REPO --revision $REV --include "$pattern" --local-dir $DIR
done
(cd $DIR && sha256sum --check - <<'SHA256'
22ad51c224e499883c1a753ad35719af44c296bbc8bcb9c931de6bdf950fa3aa  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf
6880d0483490504a530b72686d79c92daa46e58ca5b71763bbfa3778d8a6131a  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00002-of-00006.gguf
04b3ec7947255efaf789da77557da9180e93128948644828798d07b8a6e7a60a  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00003-of-00006.gguf
318762f39cd7ebf03e3d7b30feaf938fab0464a9c2d6c0e0e99fc030bbfcd820  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00004-of-00006.gguf
34fa48fbe2237b1b71a4f18a04bbe10946f6b1c76b7bbff21c76da630da677b6  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00005-of-00006.gguf
406103485a7a7193337adee38b7178d3b636940aaa3ca033bc91e0d3687e3031  UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00006-of-00006.gguf
6ba6ac1b331c540691a209e7fc97dec2a24e295acb1e755c84c7d24a4373927a  dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf
9a1ef7c40d0d702c7ed8c5f863f60abae64091387ba3fce0ad19ce9beeda3bf9  mmproj-BF16.gguf
SHA256
)
```

`mmproj-F16.gguf` (`dad8ad13f1d9a25757e121592d7b4f8ca83620d1172048004998b7c0b964b18e`)
also loads; its matrices are converted back to BF16, which is exact except for
weights that fell into F16's subnormal range. The UD-IQ2_XXS shards total
300.5 GB, of which the Q5_K Engram table is 125.9 GiB. On five 46 GB A40s that
table cannot be device-resident: it stays host-mapped, so the host (or container)
memory limit must hold it, and `TS_DSV41_ENGRAM_WARM=1` reads it in during load
instead of paging it in during the first requests.

```bash
MAX_CONTEXT=32768 TS_DSV41_ENGRAM_WARM=1 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model $DIR/UD-IQ2_XXS/DeepSeek-V4.1-Flash-1017-UD-IQ2_XXS-00001-of-00006.gguf \
  --draft-model $DIR/dspark-DeepSeek-V4.1-Flash-1017-Q8_0.gguf \
  --mmproj $DIR/mmproj-BF16.gguf \
  --backend ggml_cuda --layer-split 5 --host 127.0.0.1 --port 5000
```

### Measurements from October 22–23, 2026 UTC

The campaign used five 46 GB A40s (CUDA 12.8 toolkit, driver 595.91), a
96-core host with 503 GB RAM and a 233.8 GiB container limit including page
cache, and NFS model storage. It compared against unsloth's
llama.cpp (`be8365722ea432809818fa21a4ecebfac2e7d298`, `-ngl 99 -sm layer -lm dio`).
TensorSharp used unmodified ggml `da8e2a54b919557b816b1c2beb432ccb766f7ce0`;
the reference fork uses its own in-tree ggml. Page cache was evicted before
every load. TensorSharp measurements used `ae60abf6` plus the
then-uncommitted campaign changes and native library SHA-256 prefix
`04bc9dd748378492`. The implementation was subsequently committed as
`7f26e2d1`; these are not measurements rerun at that commit or a published
release. After review fixes, native build `8b910b29e88d6891` confirmed the
2,048-random-token default TP cell at 1,046.3 prefill / 36.2 decode tok/s,
not a rerun of the complete server matrix.

TensorSharp runs its
defaults, which include YOCO bounded replay (approximate prefill, see below);
the "exact" rows set `TS_DSV41_YOCO_KEEP=0`. TensorSharp's synthetic figures
are the best of three runs and llama.cpp's are llama-bench means (TensorSharp's
own means are 0.4-5% lower at 2,048 and 8,192 tokens). Replacing `--layer-split 5` with
`--tp 5` in the command selects routed-MoE tensor parallelism, the faster
placement on this machine. Synthetic rows use random tokens;
server rows send wikitext-2 summarization prompts with a unique first line,
greedy, thinking off, to both servers at a 32,768-token context (llama.cpp's
four-request row: `-c 65536 -np 4`):

| Measurement | TensorSharp `--layer-split 5` | TensorSharp `--tp 5` | llama.cpp |
| --- | ---: | ---: | ---: |
| Prefill, 512 / 2,048 / 8,192 random tokens (tok/s) | 500.3 / 844.3 / 906.2 | 779.8 / 1,046.7 / 1,082.7 | 227.6 / 284.8 / 310.8 |
| Same, exact prefill (tok/s) | 352.7 / 463.5 / 444.3 | 516.4 / 562.9 / 540.8 | |
| Decode at depth 2,048 / 8,192 (tok/s) | 34.5 / 34.2 | 36.4 / 36.1 | 32.6 / 31.9 |
| Server prefill, ~1.8K / ~7.3K-token prompts (median tok/s) | 697.4 / 762.8 | 960.5 / 1,000.7 | 174.6 / 225.7 |
| Same, exact prefill (median tok/s) | 405.5 / 412.7 | 563.9 / 549.7 | |
| Server decode after those prompts (median tok/s) | 33.8 / 34.0 | 35.5 / 35.7 | 29.6 / 33.0 |
| Server decode with the DSpark drafter (mean tok/s) | 44.1 / 43.5 | 55.0 / 51.2 | 41.3 / 40.8 |
| Four concurrent ~1.8K-token requests, 256 tokens each (batch wall s) | 27.3-33.1 | 21.2-25.5 | 43.3-46.8 |

For each server prompt size, one warmup was discarded before three measured
requests. The two longer workloads contain 1,747–1,812 and 7,181–7,375 input
tokens, with the same source-text slices and token counts in both engines and
256 generated tokens per request. Each request has a unique first line to
prevent prefix reuse. Prefill rates use each engine's internal prompt-evaluation
timer (`prompt_eval_duration` for TensorSharp, `timings.prompt_ms` for
llama.cpp), not streamed HTTP TTFT. The exact and approximate TP results are:

| Single-request workload | TensorSharp TP, exact prefill | TensorSharp TP, default YOCO | llama.cpp layer split |
|---|---:|---:|---:|
| ~1.8K-token prompt, prefill median (tok/s) | 563.9 | 960.5 | 174.6 |
| ~7.3K-token prompt, prefill median (tok/s) | 549.7 | 1,000.7 | 225.7 |
| Decode after ~1.8K tokens, median (tok/s) | 35.4 | 35.5 | 29.6 |
| Decode after ~7.3K tokens, median (tok/s) | 35.6 | 35.7 | 33.0 |

Here “exact” means YOCO truncation is disabled, not bitwise agreement between
engines. Exact TP prefill is 3.23× / 2.44× the reference rate on these two
workloads; default YOCO is 5.50× / 4.43× and changes the output distribution.
The llama.cpp server observations came from an earlier session. Its three
~1.8K prefill rates were 69.8, 174.6 and 183.8 tok/s; the stall was not
investigated. These small-sample medians are not a universal speedup.

The four-request row is default YOCO only. Its three whole-batch wall times
have medians 28.26 s (layer split), 22.61 s (TP), and 44.52 s (llama.cpp),
for 1,024 output tokens per batch. TensorSharp's concurrent per-request
decode timers apportion shared batch time, so they are not used for a
cross-engine throughput comparison. The DSpark row uses means, and acceptance
depends on the generated text, which YOCO can change.

In an earlier 1017 CLI run on a short code prompt (58 tokens, 512 generated), the drafter reached 80%
acceptance and 60.2 tok/s with layer split and exact prefill (llama.cpp `--spec-type draft-dspark`:
51.6 tok/s at 3 drafted tokens). Teacher-forced wikitext-2 perplexity on
identical tokens, 4,608-token windows with the first 2,304 tokens prefilled
(past the sparse-prefill gate) and the rest scored one token at a time, 10
chunks: llama.cpp 2.4697; TensorSharp 2.4970 with layer split and 2.4978 with
`--tp 5` under the default, 2.4820 and 2.4811 with exact prefill. Mean KL
divergence from llama.cpp's distributions is 0.0172 under the default and 0.0160
exact (top-1 agreement 96.2% / 96.3%). TP and layer split differ from each
other by a mean KL divergence of 0.0005 with exact prefill and 0.0004 under the
default. On this same 23,030-scored-token sample, TP's default perplexity is
0.67% above exact TP and 1.14% above llama.cpp; exact TP is 0.46% above
llama.cpp. This is a wikitext-2 quality sample, not a general quality guarantee.
Before the YOCO and TP work, a build
with only the layout support and none of the speed changes measured 2.4752 and
0.0167 (exact prefill); per chunk, the final exact build's perplexity
difference from it is not significant (t = 1.02) and its KL divergence is lower
(t = -3.01). Raw records are local validation evidence under
`docs/validation/deepseek41-1017/` (not committed).

## Prepare the optional vision companion

With unsloth's 1017 GGUFs no preparation is needed: `--mmproj mmproj-BF16.gguf`
encodes images bit for bit like a companion prepared from the official 1017
checkpoint (`mmproj-F16.gguf` differs by F16's rounding of about 0.02% of the
weights). The companion below is for conversions that carry no vision tower.
The supplied smalinin GGUF conversion omits the vision tower, aligner, learned image
delimiters and per-layer visual routing biases. The official release places
the approximately 970 MB vision/aligner weights in an isolated shard, so these
can be prepared without downloading the original text weights:

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2 gguf
/workspace/dsv41-tools/bin/python eng/dsv41-prepare-vision.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5 \
  --parent-model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --repository deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277
```

This creates `deepseek41.vision.gguf` and `deepseek41.vision.json` beside the text
shards. The `--parent-model` argument supplies the GGUF tokenizer fingerprint;
existing companions remain compatible when the tokenizer matches.
The preparer downloads the isolated vision shard and small byte ranges for
the delimiter/router tensors and preserves BF16/F32 storage. It verifies the
complete isolated vision shard against its LFS SHA-256 and records individual
range hashes for delimiter/router weights; it does not download or verify the
entire original text shards. The companion provenance record
`docs/validation/deepseek41/vision-companion.json` (local validation evidence, not committed)
contains all 306 tensors, source revisions, ranges, and the output digest.
The native loader checks the parent tokenizer fingerprint and model
dimensions before attachment. To enable images, add
`--mmproj /workspace/models/deepseek41-q2-q5/Q2_K-Q5/deepseek41.vision.gguf` to the server
command below. Image capability remains disabled without an attached companion.

Vision uses dense F32 attention by default, matching the official tower's
attention arithmetic. On Ampere-or-newer NVIDIA CUDA, BF16 matrix inputs retain F32
accumulation and output until bias addition and BF16 activation rounding.
`TS_DSV41_VISION_BF16_GEMM=0` selects the diagnostic F32-promoted matrix path.
`TS_DSV41_VISION_FA=1` selects faster flash attention with F16 intermediates;
the real-image reference comparison showed larger feature differences on
that path. These options affect the image encoder, independently of text
attention. Exact bounds and measured tradeoffs are in the
[validation report](../deepseek41_validation.md#independent-vision-and-mixed-modality-reference).

For numerical investigations, `TS_DSV41_VISION_TRACE_DIR=/absolute/directory`
writes F32 patch, block, norm, and projector outputs. Tracing retains
intermediate tensors and adds device transfers, so leave it unset for normal
inference and benchmarks.

The encoder uses 32 bidirectional transformer layers, two-dimensional rotary
positions, a padded 3-by-3 spatial merge, and a two-layer projector. It emits
the complete image span, including learned start/end and per-row newline
embeddings. Image tokens use the visual MoE bias, suppress Engram injection,
and break Engram n-grams across image spans. Mixed image/text spans can cross
prefill microbatch boundaries. Existing video frame extraction uses the same
image path. CPU/CUDA numerical fixtures and exact preprocessing checks pass.
The complete-checkpoint CUDA layer-split run passed 25 image/video requests
across concurrency 1/4 and a separate image request after long text. The final
routed-TP profile also passed all 25 image/video requests across concurrency 1/4.
Strict encoder parity is tracked separately.
The official configuration does not provide an audio decoder. V4.1 rejects
audio-bearing requests, including mixed image/audio requests, instead of
ignoring the audio. Chat Completions, Responses and Web UI return HTTP 400.
The OpenAI parsers reject audio parts before reading their payloads, including
missing or malformed audio, so a later audio attachment cannot leave earlier
image uploads behind.

The OpenAI chat endpoint accepts `image_url` parts containing base64 image
data URIs, including multiple images and images in earlier turns. Its V4.1
`video_url` extension samples a base64 MP4, WebM, or MOV through the existing
video decoder. For example, a message's content array can contain:

```json
{"type":"video_url","video_url":{"url":"data:video/mp4;base64,...","fps":1,"max_frames":3}}
```

Each sampled frame becomes an image span with its source time in seconds,
computed from frame index divided by the probed frame rate. Time labels are
approximate for variable-frame-rate clips.
Frames remain in source order; this is frame sampling rather than a native
temporal encoder. `fps` must be greater than zero and at most 60, and
`max_frames` must be 1–64. The defaults use `VIDEO_SAMPLE_FPS` and a positive
`VIDEO_MAX_FRAMES`, otherwise 1 fps and 16 frames. Frames above the cap are
sampled across the clip. Remote HTTP image/video URLs are not fetched; send
data URIs. For V4.1, both `/v1/chat/completions` (`image_url`) and
`/v1/responses` (`input_image`) reject remote or malformed image URLs and
invalid/empty image base64 with HTTP 400 before streaming or writing any images
from that request.
Valid image data URIs retain their existing decoding and upload path. The text
context budget still applies after expanding every image.
The final routed-TP host (managed stage 3,651, native `6b3b5ab3…`) passed all
eight image rejection checks
`docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-image-input-rejections.json`
and four Responses audio rejection checks
`docs/validation/deepseek41/full-checkpoint/tp8-context65536-ubatch1024-cpumoe0-cputhreads32-sparse1-compact1-chunk1024-6b3-final-audio-input-rejections.json`
(local validation evidence, not committed).
Streaming and non-streaming requests returned JSON 400 without SSE for remote
image URLs, malformed image base64, and valid-shaped or malformed audio parts.
The earlier CPU-offload host's eight image checks remain preserved separately.

## Prepare the optional DSpark companion

V4.1 needs a V4.1 drafter: the `deepseek41-dspark` artifact prepared here, or the `dflash`
file unsloth publishes with the 1017 GGUFs (see
[Unsloth's DeepSeek-V4.1-Flash-1017 GGUFs](#unsloths-deepseek-v41-flash-1017-ggufs)).
V4 drafters are incompatible.
The official revision below isolates DSpark in shards 44–46 (7,933,129,808 bytes),
so the original text model need not be downloaded:

```bash
/workspace/dsv41-tools/bin/hf download deepseek-ai/DeepSeek-V4.1-Flash \
  --revision dba1be0a40aa45a94ad051997016db3960a90277 \
  --include config.json model.safetensors.index.json \
    model-00044-of-00048.safetensors model-00045-of-00048.safetensors \
    model-00046-of-00048.safetensors \
  --local-dir /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash

(cd /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash && sha256sum --check - <<'SHA256'
8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879  config.json
74b0686a3d2891980d5e303251b075a3bccae2c2ff650747db2620a649b98fa8  model.safetensors.index.json
9a6b39fb88a2510487a8efaef77aa7864e8061f6b62c95a0f010e9dd538f3b05  model-00044-of-00048.safetensors
0cc9d5f6ca3a2158ccc63ce2c70c76aeda8177d54913340481af566680329eb5  model-00045-of-00048.safetensors
e625902027b9d23d416f8818c665fab4704e0b96dc1bc778321601b700475a9d  model-00046-of-00048.safetensors
SHA256
)
```

Only after all five checks pass, convert the companion:

```bash
/workspace/dsv41-tools/bin/python -m pip install numpy==2.0.2
mkdir -p /workspace/models/deepseek41-dspark
/workspace/dsv41-tools/bin/python eng/dsv4-dspark-to-gguf.py \
  --checkpoint /workspace/models/deepseek41-source/DeepSeek-V4.1-Flash \
  --expert-type mxfp4 \
  --out /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf
```

The audited conversion has 81 tensors, including all three visual routing biases,
and occupies 7,940,628,416 bytes. Source identity and conversion checks are
complete. Initial trained DSpark text/image HTTP probes passed with two-GPU
layer split and experimental routed-expert TP on `ggml_cuda`, including complete
image answers and clean shutdown. Broad quality and throughput remain unqualified
under heavy paging. Separate 24-token text/image pairs matched plain token IDs
and `max_tokens` finishes in both modes with active DSpark and clean exit; these
are bounded continuations. No speedup, multi-node execution, or full-model TP is
claimed.
Add `--draft-model /workspace/models/deepseek41-dspark/DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf --spec`
to a launch using the repaired ten-shard text checkpoint. Keep the same loaded
companion and use `--no-spec` for a plain-decode comparison. The head occupies
the output-head GPU and affects placement; vision needs additional space.

## Run the implemented path

Install the .NET 10 SDK, CMake, a C++ compiler, and the CUDA toolkit with `nvcc`
on `PATH`. Build from the repository root. This example uses A40 CUDA architecture
8.6; adjust both architecture values for other GPUs. It references the repaired
download, whose capacity and inference behavior must be checked on your hardware:

```bash
TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON \
  TENSORSHARP_GGML_NATIVE_CUDA_ARCHITECTURES=86 \
  bash TensorSharp.GGML.Native/build-linux.sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj -c Release \
  -p:CudaArch=compute_86 -p:TensorSharpSkipGgmlNative=true

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MAX_CONTEXT=65536 \
  TS_CPU_MOE_THREADS=32 TS_DSV4_UBATCH=256 \
  TS_DSV41_ENGRAM_WARM=0 \
  TS_DSV41_COMPACT_RAW_GATHER=0 KV_CACHE_DTYPE=f16 \
  TS_SCHED_MAX_RUNNING_SEQS=4 TS_SCHED_MAX_BATCHED_TOKENS=4096 \
  TS_SCHED_PREFILL_CHUNK=256 TS_SCHED_SOLO_PREFILL_CHUNK=8192 \
  dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cuda --layer-split 8 --port 5000
```

The host build copies the native library beside the server DLL. This launch
uses explicit microbatch and scheduler settings. It is a launch example, not a
measured repaired-checkpoint profile. The historical profiles in the validation
report used different weights and sometimes different settings.
Sparse prefill attention needs no flag: it is the default on this path, and
`TS_DSV41_SPARSE_FA=0` turns it off. `TS_DSV4_UBATCH=256` pins the example's width;
leave it unset to let the loader choose (see
[Backends](#backends)). Choose `TS_CPU_MOE_THREADS` for the available CPU quota
and record it for each run.
Set it in the launch environment, including for GPU-only placements: native
CPU graph work and host reduction can still affect latency. The current CLI
also accepts `--cpu-moe-threads N`; use the same value if supplying both, since
the positive environment value takes precedence in the native loader.

Without an explicit thread setting, a GPU-only V4.1 load uses the caller's
`TS_DSV4_THREADS` value, defaulting to at most 32; CPU expert offload instead
uses the detected available CPU parallelism. Earlier native builds did not
receive the CLI thread override. In particular, the first measured TP run's
`--cpu-moe-threads 48` did **not** configure 48 native threads: its native pool
is inferred to have used the default 32 from the loader source, absent native
environment override, and the VM override probe. That original host did not
expose a pool-width getter, so 32 was not measured directly there. Keep that
baseline distinct from later explicit-thread experiments.

`--layer-split 8` requests **eight GPUs using layer split**. The
startup diagnostic states the placement mode. TensorSharp distributes whole
layers according to available VRAM by default. Set `CUDA_VISIBLE_DEVICES` to the exact
devices intended for the run.

`--tp 8 --backend ggml_cuda` enables **routed-MoE tensor parallelism** on eight
GPUs; `--tp` accepts every degree from `2` through `8`, including `3`, `5`, and
`6`. The degree reaches the native loader as a per-model argument and must equal
the number of GPUs it selects. `--tp` and `--layer-split` cannot be combined.

In this mode, routed-expert gate/up matrices are partitioned along the FFN
intermediate dimension, and down matrices along their output rows; every
selected GPU computes its share concurrently. Each rank receives the full F32
SwiGLU activation before its down projection; the disjoint
output rows are then gathered without summing partial dot products. This keeps
the down projection's reduction order, avoiding rounding changes that later
activation quantization can amplify. Quantization blocks are never split, and
uneven strips rotate across layers to balance storage. Attention,
shared experts, and caches retain their local layer placement. The same routed
tensor-parallel path can span nodes on `ggml_cuda`; attention remains unsharded.

### Distributed routed-MoE tensor parallelism

For DeepSeek V4.1 Flash 1017, the existing `--tp-node-id` and `--tp-peers`
options extend routed-MoE sharding across nodes. `--tp` is the GPU count **per
node**: two nodes with `--tp 2` use four global ranks. Each node must use the
same local degree (1–8), with 2–16 global ranks, identical checkpoint shards,
and matching context, precision and CPU-offload settings. Keep both nodes'
GGUF sources and upstream ggml unchanged.
Quantization-block geometry can impose a smaller rank limit; layouts that
would leave an empty strip on any rank are refused.
The 2–16 range is the executor's capacity, not a guarantee for every GGUF.
A 2304-wide K-quantized down row contains nine 256-value blocks, so more than
nine global ranks cannot form nonempty strips. In particular, two nodes with
eight GPUs each are refused for that layout. The completed cross-node fixture
coverage is two nodes with one or two GPUs per node; see the validation report.
Set the same explicit `--n-cpu-moe N` on both nodes when host expert offload is
needed; automatic per-node CPU placement is refused because it could change
the collective sequence. The loader still refuses an explicit count that does
not fit the available memory.

```bash
# Node 1: existing CLI worker (the first endpoint is node 0).
dotnet TensorSharp.Cli/bin/TensorSharp.Cli.dll --model /models/FIRST-SHARD.gguf \
  --backend ggml_cuda --tp 2 --tp-node-id 1 --tp-peers NODE0:9500,NODE1:9500
# Node 0: HTTP driver, started while the worker waits for its peer.
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll --model /models/FIRST-SHARD.gguf \
  --backend ggml_cuda --tp 2 --tp-node-id 0 --tp-peers NODE0:9500,NODE1:9500 \
  --no-multi-agent --no-skills
```

Each GPU holds its global routed-expert gate/up and down-output strips. The
existing cross-node F32 AllReduce assembles disjoint activations and output
rows twice per GPU-resident MoE layer. Across nodes this exchange is
host-driven: the device-resident exchange described below is local only, so
each rank's strips go through host F32 buffers and two blocking collectives per
layer (see [Current limits](#current-limits-and-tensor-parallel-work) for their
size). Attention, shared experts, Engram and
CPU-offloaded expert layers are computed on each node. This preserves the
existing tensor-parallel computation and makes network latency material to
decode speed; use the measured transport and matched controls when assessing
performance.

Distributed serving uses a single mirrored sequence and serializes overlapping
requests. Sequence slots, retained-prefix/truncation reuse, vision and DSpark
are unavailable in this mode; load-time/capability checks prevent unsupported
execution. Start the driver with `--model`; loading or switching models,
backends or projectors through a running distributed server is refused. Restart
the driver and matching CLI workers together to change that configuration.
Requests for the already loaded model remain available. Local multi-GPU
behavior retains its existing paths. Distributed
automatic prefill uses microbatches of 256; explicit positive
`TS_DSV4_UBATCH` remains configurable. For distributed whole-layer placement,
replace `--tp N` with `--layer-split N` on every node and retain the peer
arguments; see the [multi-node guide](../multi-node-inference.md).
Use the [fixed-token validation procedure](../../eng/validation/DeepSeek41DistributedProbe/README.md)
for numerical controls and the HTTP suites for actual checkpoint quality.

The following device-resident exchange description applies to local TP.

On `ggml_cuda` the routed-expert exchange is device-resident
(`ggml_ops_deepseek41_tp_cuda.cu`). Each TP layer is a launch node and a gather
node on the layer's own GPU: the launch stages the layer's activations and
issues every rank, the GPU then computes the shared expert while the ranks run,
and the gather waits for the ranks' output rows. Every transfer is staged
through portable pinned host memory by each GPU's own copy engine and ordered
only by CUDA events, so the host never waits inside a layer, and one thread per
rank issues the ranks' work concurrently. Peer-to-peer access is not needed; it
is advertised but broken on some PCIe hosts, including the five-A40 VM below.
Outputs are bit-identical to the host-driven node this replaced and, per layer,
to layer split's MoE (`GgmlOpsDsv41TpTest`, including `--model-dir` on the
real checkpoint). A rank that fails, for example on a scratch allocation, turns
that layer's output into NaNs and fails the request: the ranks that did issue
work are drained, nothing waits forever, and the next request runs normally
(`deepseek41-tp-resident-failure-cuda`).

On five A40s with unsloth's UD-IQ2_XXS file, `--tp 5` prefills 2,048 random
tokens at 1,046.7 tok/s with the default YOCO replay (562.9 with
`TS_DSV41_YOCO_KEEP=0`) and decodes at 36.4 tok/s, against 844.3 (463.5) and
34.5 for `--layer-split 5`. Before the device-resident exchange `--tp 5` ran
402.3 tok/s prefill (exact) and 15.2 tok/s decode. On real-text server prompts
of ~1.8K and ~7.3K tokens it prefills at 960.5 and 1,000.7 tok/s (median). Over
teacher-forced wikitext-2 (4,608-token windows), TP's next-token distributions
differ from layer split's by a mean KL divergence of 0.0005 with exact prefill
and 0.0004 under the default (median 0, top-1 agreement 99.8%), against 0.016
and 0.017 between either one and llama.cpp. unsloth's
llama.cpp build has no working split of its own for this model: `-sm row`
refuses to load (`device CUDA0 does not support split buffers`) and
`-sm tensor` aborts while allocating the graph for its first prompt batch, so
the comparison above is against its layer split. Decode stays close to layer
split: in an nsys decode profile the Q8_0/Q6_K dense projections (attention,
shared expert and output head), which stay on one GPU per layer, took 31% of
GPU kernel time, and a token's six routed experts, at this quantization, 21%
summed over all five GPUs. The host-driven measurements further below predate this change; see
also the [measured placement profiles](../deepseek41_validation.md#full-checkpoint-routed-moe-tp).

Before uploading a TP layer, its three routed-expert source ranges are read
sequentially into the page cache with the existing `TS_DSV4_LOAD_THREADS` policy
(default 16, 64 MiB scratch per reader). Already resident ranges are skipped.
Only the current layer is prepared, avoiding small, strided network-file page
faults during rank uploads. `TS_DSV4_WARM_PREAD=0` keeps direct mapped reads for
this stage. This preparation does not warm the complete checkpoint.

A September 29, 2026 same-build TP6 comparison on the six-A40 VM used
`TS_DSV4_WARM_PREAD=0,1,1,0` in model-start order. Load times excluding kernel
warmup were **546.26, 149.61, 157.35, and 510.88 seconds**: disabled/enabled
medians **528.57/153.48 s**, a descriptive **3.444×** ratio. Each launch required
zero client-kernel residency across 183.25 GiB of complete pages in all 120
routed-expert tensors. Settings, checkpoint, source, native library and managed
runtime stayed fixed; all 129,280 first-prefill logits and the one-token greedy
check matched exactly, and every process exited cleanly. The fourth cache
precondition initially left 135 pages resident and prevented launch. A separately
recorded continuation passed the same zero-page gate on its first attempt and
supplied the fourth observation; the failed attempt remains preserved. This is
an interrupted comparison with two starts per setting, not an uninterrupted
ABBA trial. Partial boundary pages and MooseFS userspace/network/server caches
were uncontrolled, so the ratio is not a cold-storage or universal speedup.

Without TensorSharp's fused CUDA backend (`TS_DSV4_FUSED=0`), the TP node runs on
the CPU backend and drives the ranks from the host, synchronizing each phase.
On that path, when NCCL selects `NCCL_P2P_DISABLE=1`, batches of at most 16 tokens use the
pinned host activation gather; larger batches use the private F32 NCCL gather,
which is created on first use.
This threshold follows paired measurements on six PCIe A40s and is not applied
to P2P-enabled configurations. `TS_DSV41_TP_HOST_TOKENS=0` forces the available
device gather for comparison; values from 0 through 4096 set the host threshold.
`TS_GGML_TP_F32_NCCL=0` selects the host fallback for every batch. Both transports
preserve F32 bits. Device gathering remains ordered on the rank CUDA streams,
without a separate gate/up completion fence; errors drain every rank before
returning control to the caller. Full-model throughput must still be measured
for the chosen placement and transport.

An earlier September 29, 2026 UTC host-driven transport check used six PCIe A40s, the repaired ten-shard
EngramQ5/Q2_K checkpoint, context 4096, F16 KV, host Engram tables with warming
disabled, and `NCCL_P2P_DISABLE=1`. Each placement had one process start and five
identical fixed-input rows: 512 prefill tokens and 128 decode tokens. Candidate native
`473ee64d…` passed the runtime-file integrity checks; the upstream ggml checkout
remained unchanged at `353b63b4…`. All three runs produced the same complete
128-token untimed greedy chain as the original layer-split baseline.

That host-driven experiment predates the device-resident exchange above and
does not select the fastest current placement. Its raw timing rows remain in
the [validation report](../deepseek41_validation.md#full-checkpoint-routed-moe-tp).

In that experiment, container memory was about 255 GiB for TP versus
70 GiB for layer split. File cache accounted for about 252 versus 68 GiB;
anonymous memory was about 1.6–1.7 versus 0.8–0.9 GiB. TP upload scratch is
bounded, but uploaded routed-weight pages remained in reclaimable file cache:
`TS_DSV4_LOAD_DROP_CACHE` then applied to ordinary layer uploads only. Since
2026-10-23 the TP uploader follows the same rule, releasing each layer's source
pages after its strips are uploaded when the sources and the host mappings would
not fit the host allowance together (`[dsv41-tp] expert source page cache: ...`). GPU capacity alone therefore does not establish that
a host-memory limit is sufficient; the observed 255 GiB is not a minimum RAM
requirement. These sequential loads had different
source-page residency and are not a controlled loading-latency comparison.

The current output-row implementation passed numerical fixtures on two and six
A40s, including F32, BF16, F16, Q2_K, Q3_K, Q4_K and Q6_K on two GPUs and real
repaired-checkpoint expert weights on six. Output-row strips now align to 128,
64 or 32 rows: the earlier 2-row fallback sent the 256-wide F16 fixture's down
projections at three and five ranks through ggml's F16-accumulating fallback
(relative L2 `2.8e-3`), which is why those two CUDA cases failed. On the five-A40
VM the CUDA fixtures pass at 2, 3, 4 and 5 ranks, with the device-resident node
and the launch/gather pair bit-identical to the host-driven node. CPU fixtures cover degrees 2 through
8; seven- and eight-GPU runs were unavailable on the six-GPU validation VM.
The final `473ee64d…` runtime passed six-GPU HTTP qualification with the repaired
ten-shard Q2_K/Q5 checkpoint: 16 text responses matched the unchanged layer-split
baseline exactly, all four strict tool cases and three image cases passed, and
all 129,280 first-prefill logits were bitwise identical. DSpark and ngram each
preserved all 96 greedy tokens and the finish reason on text and image inputs.
Recorded drafted/accepted/verify counts were 49/49/10 for each ngram scenario,
83/73/19 for text DSpark and 40/22/13 for image DSpark; DSpark exercised seven
and nine rollbacks respectively. Both HTTP and speculation exited cleanly with
no runtime-file changes. These counters establish active speculative coverage,
not a speculative throughput improvement.
These checks do not establish that the full checkpoint fits on two or four A40s.
When the historical seven-shard Q2_K Engram tables use host mappings, synchronous warming consumes
approximately 60 GiB of host page cache before readiness. GPU-resident tables
skip this warm. Record cold-load and warming time separately from warm throughput.

On CUDA, quantized gate/up and down output strips run through TensorSharp's owned
quantized strip kernel (`ggml_ops_matmul_quant_strip.cuh`,
`tsg_matmul_id_quant_pair`): it reads only the rank's weight strip but keeps
the unsplit launch's stream-k partitions and reduction order, so each strip's
gate/up rows equal the full tensor's bit for bit. Before it, ggml's batched
MMQ path grouped a strip's F32 sums differently from the unsplit launch and
the down projection's Q8 activation requantization amplified that into a
checkpoint-shaped Q2_K/Q3_K failure at 16 tokens (relative L2 `3.9e-5`
against the `1e-5` full-weight tolerance). `GgmlOpsDsv41TpTest` keeps the
strict full-weight reference and its original tolerances as the pass
criterion, records the same-device partitioned evaluation beside it, and
`--cuda 1 --quant-strip-only` checks bitwise gate/up equality plus scratch
growth/failure recovery. Nonaligned strip shapes stay on ggml's route. The
recorded 15-36% narrow-strip MoE slowdown describes the earlier split-down
implementation, not the current two-gather
implementation; see
`docs/validation/qualification-2026-09-16/numerical-tp-chosen-r1/README.md` (local validation evidence, not committed).

If the weights and context do not fit, add `--n-cpu-moe N` to keep the routed
experts of the first N layers on the host, or `--cpu-moe` for all routed
experts. Attention, routing, and the shared expert remain on the GPU.
[Engram table placement](#where-the-engram-tables-live) is selected separately;
when the tables use host mappings, only selected embedding rows are read and
transferred for each input batch. CPU MoE offload and layer split are implemented,
but their throughput must be measured for the chosen hardware
and context. When combined with `--tp N`, CPU-offloaded leading layers
retain whole CPU experts; the remaining layers use the routed-expert shards.

Native `6b3b5ab3…` explicitly assigns shared gate/up/down projections to the
layer device. This corrects earlier scheduler placement that could send shared
gate/up work to CPU after a CPU-offloaded or TP routed branch. The
placement and numerical checks
(`docs/validation/deepseek41/shared-expert-placement/README.md`, local validation
evidence, not committed) passed 597/597 on two GPUs. Earlier full-checkpoint placement benchmarks retain
their original binaries and results. Final CPU-offload, routed-TP and layer-split
profiles using the corrected placement have completed. See the placement
records in `docs/validation/deepseek41/final-placements/README.md` (local validation evidence, not committed).

For a warm CPU-offload benchmark, read the offloaded expert pages separately
after model loading and before starting the timed requests:

```bash
/workspace/dsv41-tools/bin/python eng/dsv41-warm-experts.py \
  /workspace/models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --layers 4 \
  --report cpu-expert-warming.json
```

This requires the `gguf` Python package and reads all three routed matrices
in each selected layer. Record its I/O time separately. Pages remain evictable;
Engram warming alone does not warm CPU expert weights.

### Where the Engram tables live

By default the two Engram tables are **GPU-resident**: each is loaded onto the
device that owns its layer, and the graph gathers rows with `get_rows` over the
quantized table. The host never reads a row, so nothing is dequantized on the
CPU and only the row ids cross the link. Startup prints
`Engram lookup: GPU-resident tables, gathered in-graph`.

This is possible because TensorSharp keeps the tables in the checkpoint's
quantization. A Q2_K row of 256 values is 84 bytes, so both 384M-row tables
together are 60.1 GiB. vLLM and SGLang store the same rows as FP8 values plus
per-32 block scales, 264 bytes a row, which is why their default is a host
table with an FP8 gather kernel.

Placement is automatic and conservative: the tables are priced into the
layer-split packing, and if putting them on GPUs would force any routed-expert
CPU offload, they stay host mappings instead and startup says so. Set
`TS_DSV41_ENGRAM_DEVICE=0` to force host mappings or `=1` to require GPU
residency and fail if it does not fit. `TS_DSV41_ENGRAM_THREADS`,
`TS_DSV41_ENGRAM_WARM` and `TS_DSV41_ENGRAM_RANDOM` only affect host mappings;
startup notes when one is set on the GPU-resident path.

GPU-resident tables avoid a host-table warm, but their capacity cost depends
on the checkpoint. The 1017 five-A40 measurements above use host-mapped
Q5_K tables; the older eight-A40 Q2_K throughput figures are not a prediction
for that layout. Earlier placement checks produced identical text at
temperature 0 across host mapping with and without warming and GPU residency.

That is argmax stability rather than bitwise agreement: ggml's CPU and CUDA
Q2_K dequantizers evaluate the same expression, but the device one may contract
a multiply-subtract into an FMA and differ by up to one ulp. Use
`TS_DSV41_ENGRAM_DEVICE=0` for a run that must match the CPU oracle bit for
bit.

### Host-mapped Engram tables

These options apply only when the tables stay on the host, which happens when
the GPUs cannot hold them or `TS_DSV41_ENGRAM_DEVICE=0` is set.

On network-backed storage, first access to sparse Engram rows dominates prefill
and decode latency. Each token hashes 24 row ids per table, and a row that is
not already page cache is one storage round trip: on the eight-A40 VM's
network-mounted Q4_K_M checkpoint that is **1,435-2,565 ms of input preparation
per 1024-token prefill chunk**, against 56-81 ms once the pages are resident.
Decode input preparation is 5-15 ms a token cold and 0.9 ms warm.

So **warming is the default** whenever the tables are host-mapped and the pages
can stay resident. It runs on its own thread after the model is serving, not
during load, so startup is unchanged and requests work (more slowly) while it
proceeds; startup prints how much it is warming and prints again when it
finishes. A model load reads the whole checkpoint through the page cache and
evicts the previous run's Engram pages, which is why this has to happen after
every load rather than once per machine.

Warming reads only those tables, creates no private copy or pinned allocation,
and leaves the pages evictable. It is skipped with a diagnostic when the mapped
host weights plus 8 GiB do not fit the detected host/cgroup memory allowance, or
when `MemAvailable` could not keep the tables cached anyway.

Warming reads each table with `pread` in 64 MiB blocks, one contiguous run of
the table per reader thread (`TS_DSV4_LOAD_THREADS`, default 16), and skips a
block whose pages `mincore` already reports resident, so a reload whose tables
are still cached reads almost nothing. It used to touch the mapping one byte per
4 KiB page, and on a network filesystem every such fault is a synchronous read
capped at the mount's readahead (128 KiB on the A40 VMs' MooseFS mount): the
seven-A40 lane's synchronous warm of the 103 GiB Q4_K_M tables took 311.3 s.
Measured on that VM, 8 GiB of an evicted table reads at 2.38-2.54 GiB/s with
`pread` and 0.63-0.69 GiB/s with the page walk; see [Load time](#load-time) for
the method. `TS_DSV4_WARM_PREAD=0` restores the page walk for both forms below.
A sync-mode read error fails the load with the file and offset; in the
background it is one log line and the model keeps serving.

| `TS_DSV41_ENGRAM_WARM` | behaviour |
|---|---|
| unset (default) | warm in the background once the model is serving; the finish line reads `[dsv41] warmed ... Engram pages in ...s (background, ...)` |
| `1` | warm synchronously during load, as before; startup takes as long as reading the tables (311.3 s for 103 GiB with the page walk on the seven-A40 lane; at the measured 2.24-2.54 GiB/s `pread` rate that is 41-46 s of reads) |
| `0` | never warm |

Sparse-read mapping advice (`MADV_RANDOM`) is applied only after warming
finishes, whichever form it took: the advice turns off the readahead the warm
pass depends on. The `pread` warm leaves the pages in the page cache without
mapping them into the process; a lookup's first touch of a row is then a minor
fault, not a storage read. 2,000 random 144-byte rows through a `MADV_RANDOM`
mapping of a table warmed that way averaged 0.0037-0.0056 ms, with no major
faults.

`TS_DSV41_ENGRAM_THREADS=1..32` controls the persistent lookup workers;
the default is the smaller of 16 and the hardware thread count. Both prefill
and decode use parallel row lookup: a single token selects 24 independent
rows per Engram table. Setting one worker keeps reads serial. This avoids
serialized page faults during decode without changing the embedding values.
The executor interleaves reads from both tables in one worker-pool job and
uploads each table's rows after the workers finish. Staging is grouped within
64 MiB; a single larger table retains the previous one-table allocation bound.

On Linux, parallel lookup automatically requests random-access advice for
the mapped Engram ranges to reduce unnecessary readahead. The hint is applied
after optional whole-table warming and leaves other tensor ranges at their
existing policy, except for shared boundary pages. Set
`TS_DSV41_ENGRAM_RANDOM=0` to disable it or `=1` to force it. When unset, a
one-worker configuration retains the default mapping policy. Unsupported
platforms and rejected OS hints remain nonfatal. The source comparison and
paired scratch-file results are in the Engram investigation,
`docs/validation/deepseek41/cli-gpu-execution/README.md` (local validation evidence, not committed).

### Backends

`--backend ggml_cuda` is the serving path: it is the only one with kernels for
this architecture's fused ops.

`--backend ggml_cpu` runs the whole model on one CPU device, with the scalar
implementations those fused ops fall back to. It exists so the architecture can
be run and checked without a GPU. It is not a serving path: this checkpoint
reads six of 384 routed experts per layer per token out of 246 GiB.

`TS_DSV41_ALLOW_NON_CUDA_GPU=1` additionally permits `ggml_vulkan` and
`ggml_metal`. There the ordinary graph runs on the GPU and only the
architecture-specific ops fall to the CPU backend, at a host round trip per
occurrence. It is opt-in rather than automatic because what the refusal
originally closed was a silent fallback onto whichever GPU enumerated first, not
an explicit request; startup names each device it applies to. Treat it as a
portability and correctness path until it has been measured on your hardware.

`--backend cpu` is not a ggml backend at all: it is the pure-C#
`DeepSeek4CpuExecutor`, with no native library and no GPU, and it implements the
whole V4.1 graph — the ratio-1 and ratio-2 block compressors, the shared
compressed and indexer caches with the lightning indexer's top-k, candidate
block pruning, the Engram tables, the delayed hyper-connection gates, the shared
expert, and the checkpoint's trained cache quantization (FP8 E4M3 raw rows,
MXFP4 indexer, NVFP4 compressed). It is held to the independent PyTorch oracle
`eng/dsv41-reference.py` at atol=rtol=2e-5, plus exact greedy-argmax agreement,
across one-shot prefill, chunk sizes 1/3/5/8 and reset — at fixture scale
rather than on the released weights. `--backend cuda`, the direct-CUDA engine,
also runs V4.1 through its own kernels with no ggml. It is held to `ggml_cuda`
and to `ggml_cpu`, both with exact prefill (`TS_DSV41_YOCO_KEEP=0`), on a
C#-built checkpoint shaped like the 1017 (`InferenceWeb.Tests`
`Dsv41SyntheticModelBuilder`): llama.cpp's combined Engram layout, the 1017's
IQ1_S/IQ2_S/IQ2_XXS gate+up and IQ2_S/IQ3_XXS/IQ2_XS/IQ2_XXS/Q2_K down experts
at a 768-wide expert (rows only 2-byte aligned), Q6_K/Q8_0 attention, Q5_K/Q8_0
shared experts and BF16 compressor and indexer, and a prompt long enough for the
sparse selection and the candidate mask to engage. The script is a 150-token
prefill (three 64-token ubatches), 15 decode steps, a 21-token append at the odd
position 165 (a multi-row call that completes the compressed block a decode step
began), 2 more decode steps, then a reset, the prompt and the first 3 decode steps
again, which must reproduce the first answers to the bit. With each routed-expert
route forced in turn (tensor cores, register-staged, warp-per-row, per-slot),
every logit vector is within 2e-2 relative L2 of the native one with the same
argmax: measured at most 8.2e-3 against `ggml_cuda` and 9.7e-3 against
`ggml_cpu`. That checkpoint is conditioned so two correct engines can meet such a
bound: on random weights with full-size updates the two ggml backends themselves
differed by 9e-2, because the routing, the sparse selection and the
FP8/NVFP4/MXFP4 cache rounding turn rounding differences into large ones. It
writes small attention and shared-expert updates, so its logits see the attention
path, and every row but a call's last, only coarsely: removing the sparse
selection altogether moves them by 1.6e-2. The stages are therefore also held to
the native executor's own trace (`TS_DSV41_TRACE_DIR` against
`TS_DSV4_CUDA_TRACE_DIR`), row by row:

- Every routed-expert block's output, on every row of every call and layer, on
  each forced route: within 0.1 relative L2 beyond twice the error of the stream
  it was given (measured at most 4.5e-2 against `ggml_cuda`). A row past that has
  to be a routing flip, where the direct engine's own fourth and fifth expert
  choices stood within 2e-2 (one row in 1128 against either backend, at a margin
  of 3e-3 to 5e-3). The experts it chose must be the top four of its own router
  scores plus the selection bias, which decides 183 of those rows.
- The sparse attention. The indexer's queries and keys are rounded to the trained
  4-bit cache format on both sides, so the two engines' scores differ by about 6%
  per token and their selections by a row or two on about half of the tokens;
  what each engine must do exactly is the arithmetic on its own scores. The rows
  every index source scores (all visible rows; after the candidate layer, its
  four best 8-row blocks with the query's own block first) and the 16 it selects
  are checked on both engines against those rules, which ggml's trace shows are
  its arithmetic. The scores are held to ggml's per token and on average, the
  compressor's latents per row (at most 4.3e-3), and the attention output on
  every row whose selection is ggml's (at most 3.4e-2).
- The Engram stage: the projected rows per token (2e-7 at decode, 5.4e-3 where the
  direct engine's prefill GEMM reads F16 activations) and the gated update each
  engine applied, relative to the stream (2.2e-5 at decode, 4.8e-4 at a
  multi-row call).

Each stage check catches a seeded bug that leaves every logit vector within
2e-2: a grouped expert route that drops a busy expert's members past the
sixteenth in layers 3-5, a consumer layer that attends over another source's
compressed cache, a candidate layer that keeps one block too many, an Engram gate
that scores the next token's key in a multi-row call, and a compressor that drops
the block an odd-position call completes. An IQ1_S decoder with its delta sign
inverted fails the logits themselves (3.9e-2 to 7.7e-2 on every vector, measured
on the script's earlier 16-step form). `eng/validation/compare-dsv4-traces.py`
diffs a `TS_DSV4_CUDA_TRACE_DIR` trace against a `TS_DSV41_TRACE_DIR` one stage
by stage (it maps the native `v41.NN.<stage>` names, the compressor's latents
included, and derives the Engram gate the direct engine folds into its kernel)
to find where a model first departs. Like `ggml_cpu`, both are
correctness and portability paths, not serving ones; the end of
[Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend) has the
detail. `--backend mlx` remains refused.

| `--backend` | Engram GGUF layouts | One node, several GPUs | Across nodes (`--tp-node-id`/`--tp-peers`) | Role |
|---|---|---|---|---|
| `ggml_cuda` | TensorSharp's and llama.cpp's (unsloth 1017) | `--layer-split N`, routed-MoE `--tp 2..8` | layer split (1-8 GPUs per node) and routed-MoE TP (2-16 global ranks) | serving |
| `ggml_cpu` | both | n/a (one CPU device) | refused | correctness, portability |
| `cuda` | both | `--layer-split N`; `--tp` refused | layer split (1-8 GPUs per node, the driver runs the last stage); TP refused | correctness, portability |
| `cpu` | both | n/a | refused | correctness, portability |
| `ggml_vulkan`, `ggml_metal` (`TS_DSV41_ALLOW_NON_CUDA_GPU=1`) | both | one GPU | refused | portability |

The two managed executors, `cuda` and `cpu`, read Engram in either layout and
pick it exactly as the native loader does (`deepseek41.ple.layers` present and
`deepseek41.engram.layer_ids` absent means combined). TensorSharp's per-layer
layout is `deepseek41.engram.*` metadata (layer ids, multipliers, primes,
offsets, the compressed pad id) with `blk.N.engram_embd`, `engram_wkv`,
`engram_q` and `engram_k` tensors, which is what `eng/dsv41-fixture.py` writes.
llama.cpp's combined layout, which unsloth's DeepSeek-V4.1-Flash-1017 GGUFs use,
is `deepseek41.ple.*` metadata, one `per_layer_token_embd.weight` table,
`engram_kv` and the fused `engram_gate` (q*k). For it the per-layer offsets are
rebased onto each layer's slice of the table and the raw `engram.pad_token_id`
is compressed through the token map, as `engram.py` does. Each slice is then
exposed as that layer's `blk.N.engram_embd.weight`: a row range of the shard
that holds the table, mapped or streamed in place. The combined entry itself is
dropped, so `TS_DSV4_MMAP=0` never buffers 126 GiB twice (on the 1017-shaped
fixture the pure C# executor buffers the table once, as its two row views, and
answers exactly as from the mapping). `--backend cuda` places the tables as
`ggml_cuda` does: in VRAM when the layer split holds them without offloading more
experts, host mappings otherwise; `TS_DSV41_ENGRAM_DEVICE=0` forces the host
mappings, which is what the 1017's 62.9 GiB slices take on 46 GB GPUs, and `=1`
requires VRAM. The direct engine used to ignore the variable. Forced to the host
mappings, the fixture's logits are those of the VRAM tables to the bit.
`engram_gate` scales
the normalized query and the key gets a gain of ones, which is the native
graph's arithmetic. On a fixture rewritten into the combined layout, both
executors give logits identical to the bit to a per-layer copy carrying the
same folded product. The 1017's `deepseek41.attention.indexer.candidate_top_k_blocks`
is read as the candidate budget by these executors and the native loader alike,
so ggml and the managed executors prune to the same budget (a file that also
has `candidate_top_k` must agree with it).

What the managed executors still refuse, from the header and before any weight
is mapped, is a file that mixes the layouts or carries only part of the
combined one: a `per_layer_token_embd.weight` table or `deepseek41.ple.*` keys
without `deepseek41.ple.layers`, a combined table beside TensorSharp's
`deepseek41.engram.layer_ids`, or `ple` metadata beside a per-layer table. The
message is `This DeepSeek V4.1 GGUF has ... never a mix or a part of one, so it
cannot tell which rows belong to which Engram layer.` Before, the partial case
failed inside the executor with `Missing or invalid DeepSeek V4.1 GGUF metadata:
deepseek41.engram.pad_id`.

The direct-CUDA engine also checks every routed-expert tensor before it uploads
anything, against the code that will run it. Its GPU expert kernels are the ones
the GLM-5.3-Flash and Qwen3.8 direct engines run, and implement Q8_0, Q6_K, Q5_K,
Q5_1, Q4_K, Q3_K, Q2_K, IQ4_XS, IQ4_NL, IQ3_S, IQ3_XXS, IQ2_S, IQ2_XS, IQ2_XXS,
IQ1_S and MXFP4 on every route: the per-token decode kernels, and for prefill the
tensor-core, register-staged and warp-per-row ones. That covers every routed-expert
type of the 1017 `UD-IQ2_XXS` file: gate/up IQ1_S (10 layers), IQ2_S (15) and
IQ2_XXS (15); down IQ2_S (15), IQ3_XXS (15), IQ2_XS (6), IQ2_XXS (3) and Q2_K (1).
Layers offloaded with `--n-cpu-moe` run on the managed host matmul, which
implements a different set (IQ2_XS, IQ3_XXS and Q2_K, but not IQ1_S, IQ2_S,
IQ2_XXS or Q3_K), so that file's experts have to stay in VRAM on `cuda`. A tensor
that cannot run where the layer split puts it is refused by name, for example
`blk.0.ffn_gate_exps.weight is IQ1_S (offloaded by --n-cpu-moe)`. Before, the
whole checkpoint streamed into VRAM and the first token through that layer failed
with a bare type number. When the GPUs cannot hold such a file, the VRAM refusal
says that no offload can help and names the first tensor in the way (`... and
blk.0.ffn_gate_exps.weight is IQ1_S, which the host matmul behind --n-cpu-moe does
not implement`); it used to suggest an `--n-cpu-moe` value that the engine then
refused. The IQ1_S and IQ2_S decoders, and IQ2_XXS's tensor-core and
register-staged ones, are recent kernel code. On a kernel module built from older
source (a committed PTX baseline not yet regenerated, which a machine without nvcc
loads) the engine refuses IQ1_S and IQ2_S experts by name, `blk.N.ffn_gate_exps.weight
is IQ1_S, and the loaded tensorsharp_dsv4_kernels module was built from kernel source
older than its IQ1_S expert decoder`, rather than reading them as Q8_0, and runs
IQ2_XXS experts on the per-token and warp-per-row kernels such a module already has.
It checks the module its devices loaded, before uploading anything. Until this change the engine
kept a narrower list of its own and refused IQ1_S, IQ2_S, IQ2_XS and IQ3_XXS. The
kernels are checked against ggml's dequantization of the same bytes, the engine
against `ggml_cuda` on the 1017-shaped fixture above, and the 1017's own weights
stage by stage on a 10-layer truncation
([Per-stage oracle](#per-stage-oracle-against-ggml_cuda)) and as a whole model on
2 nodes x 2 A40, where the final logits pass a gate against `ggml_cuda`
([Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40)). Every expert stayed in
VRAM there; no other quantization of these experts has run.

Distributed routed-MoE TP is `ggml_cuda` only (on `cuda` it is refused before
any peer is contacted). Distributed layer split also runs on the direct-CUDA
engine: `--backend cuda --layer-split N --tp-node-id K --tp-peers ...`, the same
flags on the CLI, the server (driver) and
`eng/validation/DeepSeek41DistributedProbe`. Each node is one pipeline stage:

- The cut comes from the shards' header bytes, never from free VRAM, so every
  node computes the same one: `LayerSplitPlan.CreateWeighted` over each layer's
  weights, its caches and the cache replicas the cut needs, the embedding on the
  first GPU and the head on the last. For the 1017 on 2 nodes x 2 GPUs at 64K
  context it is (12, 22, 31), as for `ggml_cuda`. A node then checks that its
  part fits (weights, caches, the cache replicas it holds, the embedding or head);
  when one does not, it refuses naming the GPU and the bytes, and every other node
  refuses with it before any weight is uploaded. It never moves layers.
- A node describes, maps, reads ahead and uploads only its own layers (the
  embedding when it owns layer 0, the head when it owns the last).
- A node of several GPUs receives the boundary state on its first GPU and hands
  on, through its own in-node boundaries, whatever its later GPUs read: on the
  1017 at 2 x 2, node 0 receives group 20's cache rows and the candidate mask on
  its first GPU, and layers 31-39 on its second read them. Every in-process
  boundary checks that it hands over exactly the blocks the ubatch closed in
  each group read past it (whether committed locally or received), and an active
  candidate mask when the layers past it prune with it, as the network receiver
  checks a payload.
- Per ubatch the stage boundary carries the hidden streams, the delayed
  hyper-connection gates, the compressed and indexer cache rows the stage
  committed for every group read past the boundary, the published sparse
  selection and the candidate mask (its visible columns): what the in-process
  split moves between two GPUs. The payload is self-describing: a header with a
  layout hash (dimensions, ratios, sources, the placement, ubatch, context), the
  position and a sequence number, which the receiver checks, and the committed
  ranges, which it recomputes from the position, before anything reaches its GPU.
  On the 1017's 21|22 boundary that is about 87 KB per decoded token at depth 2K
  (151 KB at 64K) and 21-37 MB per 256-token prefill ubatch. A payload must fit
  one transport frame (2 GiB): a prefill width and context that would exceed it
  (2048 rows at 1M context: the candidate mask alone is 2 GiB) is refused on
  every node at startup.
- Every node digests the checkpoint header and each setting that changes the
  engine's numbers (`TS_DSV4_UBATCH`, `MAX_CONTEXT`, `TS_CUDA_QMM_F16GEMM_*`,
  `TS_CUDA_Q80_*`, `TENSORSHARP_CUDA_MOE_FUSION`, `TS_DSV41_ENGRAM_DEVICE`, ...),
  then the placement and the kernel module (the SHA-256 of the image the GPUs
  loaded, the build's release and source revision, the boundary wire version and
  the decoders), and all refuse to load on any difference.
- The driver (node 0) runs the last stage, so logits never cross the network;
  `TENSORSHARP_LAYER_SPLIT_DRIVER_STAGE=first` is refused on `cuda`. So are
  `--n-cpu-moe`/`--cpu-moe` and a DSpark drafter. As on `ggml_cuda`, the run
  serves one mirrored sequence: no slots, truncation or prefix reuse. Engram
  tables stay host mappings unless `TS_DSV41_ENGRAM_DEVICE=1`, and each
  host-mapped table is kept by one node (see
  [Engram reads on `cuda`](#engram-reads-on-cuda)).
- A node that fails mid-forward aborts the mesh: its peers fail the request at
  once instead of waiting out the transport deadline, and every node must then be
  restarted.
- With `TS_DSV4_ASYNC_SEND=1` a sending node does not wait for one prefill
  ubatch's payload to leave before it starts the next ubatch of the same
  forward. It queues the payload's copies to pinned memory and sends it during
  the next ubatch, once that ubatch's layers on the node's first GPU are queued,
  so the node's GPUs work on consecutive ubatches at the same time instead of
  taking turns. `=2` sends at the next ubatch's own send instead of its first
  in-node hand-over. The last ubatch of every forward is sent before the forward
  returns, so nothing is in flight between requests, and a decode step (one
  ubatch) is unchanged. Unset or `0` (the default) sends each payload
  synchronously, as before. The deferred send is opt-in until it has run with
  each stage on GPUs of its own: on one shared GPU its gain cannot show, and the
  wait on its copy event from another GPU's CUDA context has never run (every
  test device shares one physical GPU's context). Payloads, their order and
  every kernel are the same in every mode, so the logits are bit-identical and
  the nodes need not agree on the setting.

Coverage: a two-node loopback on one GPU (each node a process-local stage over
real TCP) on the 1017-shaped synthetic fixture, with the cut forced past a KV
source, past the candidate source and past an Engram layer, is bit-identical to
the single-process engine at every logit of a multi-ubatch prefill, 18 decode
steps, an append and a reset plus re-prefill; so is `DeepSeek41DistributedProbe`
run as two processes, and so is the server's startup path driving a worker.
Nodes of several GPUs run in the tests as several logical devices (each its
own allocator, stream, events and kernel module; logical device d runs on
physical GPU d mod the visible count): on a synthetic topology whose ratio-1
group is read by four layers, two nodes of two GPUs each (the received rows and
mask handed on through node 0's in-node boundary) and three nodes of one GPU
(the middle node receiving and sending them on) are bit-identical as well, and
so is the single-process split over two to four GPUs. The deferred send is
held to the same standard: with `TS_DSV4_ASYNC_SEND` at 1, 2 and 0, a script of
prompts of 1 to 5 ubatches (odd and even counts, exact multiples and
remainders), a prompt fed in three chunks, decode steps, refused truncations
and resets answers byte-identically at three cuts of two nodes of two GPUs, on
three nodes of two GPUs (a middle node that both receives and sends) and three
of one GPU in process, and on three nodes of one GPU over TCP; so does a
15-ubatch prompt. These runs share one GPU and measure no speed. On one GPU every logical
device shares it. On a host with two A40s the two-nodes-of-two-GPUs run put
each node's two devices on the two physical GPUs (both nodes, in one process,
sharing them). All of these runs use the synthetic fixture. The one real
checkpoint run distributed on `cuda` so far is the whole 1017 on two pods of two
A40s each, compared with `ggml_cuda` rather than with a single-process `cuda` run
(no single host here holds it); see
[Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40).

### Engram reads on `cuda`

On the direct engine the executor reads the host-mapped Engram tables itself
(`Dsv41EngramReader`, `GatherEngramRows`). None of the settings below changes
a result: every row goes through the same host dequantizer into the same place.

- **Parallel rows** (on by default). A ubatch's rows of one table are
  dequantized in parallel over (token, column), `TS_DSV41_ENGRAM_THREADS` (1-32,
  default min(16, cores)) at a time, so a decode step's 24 rows per table are
  read together instead of one after another. S9 parallelized over tokens only,
  so a decoded token read its 48 rows serially (83.5-94.5 ms of node 1's time per
  token cold). The rows are read by the reader's own threads
  (`TS_DSV41_ENGRAM_THREADS` - 1 of them) and the thread that asks, never by the
  .NET thread pool, which a busy server's request threads could starve.
  `TS_DSV41_ENGRAM_THREADS=1` restores the S9 gather exactly (tokens in parallel
  on the .NET pool, each token's columns one after another) and turns the advice
  off unless it is asked for.
- **Prefetch** (opt-in, `TS_DSV41_ENGRAM_PREFETCH=1`; unset or `0`: off). When a
  forward starts, the reader's threads read one byte of each page of every row
  the forward will gather from this node's tables, in the order the layers ask
  for them (both tables of ubatch 0, then ubatch 1, ...), while the host issues
  the layers. It hashes a copy of the history (`Dsv41EngramData.PreviewHashes`)
  and only reads ahead. A gather always goes first: a thread that is reading
  ahead joins it before its next row, so the reader never has more than
  `TS_DSV41_ENGRAM_THREADS` rows in flight. A reset and a failed forward stop the
  read-ahead; a page-cache eviction stops it and waits until no thread still
  reads. It is off by default because what it gains on cold network storage, and
  what reading a long prompt's rows ahead costs a cgroup that cannot hold the
  tables, have not been measured on the full model.
- **`MADV_RANDOM`.** The tables are mapped for random reads, so a fault reads
  one page instead of a read-around window (S9 measured 55-60 KB of page cache
  per row read without it, against ggml's 3-4 KB). `TS_DSV41_ENGRAM_RANDOM=0`/`1`
  (unset: on when the threads are more than 1); Linux only. The warm below reads
  with `pread`, which the mapping advice does not slow.
- **Ownership** (distributed layer split, host-mapped tables). Each table is kept
  by one node, which maps it, warms it and gathers its rows
  (`Dsv4StagePlacement.EngramOwners`). `TS_DSV41_ENGRAM_OWNER` unset or `local`
  (the default) keeps each table with the node that runs its layer, as S9 did;
  `balance` hands the tables out in index order, each to the node holding the
  fewest table bytes so far and to its reader on a tie. On the 1017 at
  (12, 22, 31) both tables are read on node 1; with `balance` node 0 keeps
  layer 14's 63 GiB slice and each node holds one, the only placement in which
  each node's tables fit its 93.1 GiB cgroup. A table kept by another node
  crosses at the start of every forward, before any boundary: the keeping node
  hashes the forward with its own copy of the history (every node sees every
  forward's tokens), gathers the rows and sends their ids and raw quantized
  bytes, 24 x (4 + 176) = 4,320 B per token (8.8 MB for a 2,048-token prompt, in
  16 MiB pieces). The reading node checks the ids against its own hashes, then
  dequantizes the rows with the same function as a local gather, so the result
  is bit-identical to `local`. `balance` is opt-in until a two-node run has
  measured it: the reading node cannot start its first layer until the whole
  forward's rows have arrived, and the keeping node does not read that table
  ahead. The owners are part of the startup agreement, and the configuration
  digest compares the mode by meaning (unset and `local` agree; `balance` on one
  node only is refused). On the server a prompt arrives in chunks of at most
  4,096 tokens, so a message is at most ~18 MB; a CLI prompt crosses whole. A
  message buffer above 32 MiB (a long CLI prompt's) is released after its
  exchange, and the reading node's copy of the rows shrinks back on the next
  forward that needs at most half of it.
- **Warm budget.** `TS_DSV41_ENGRAM_WARM` keeps its meaning (unset: background,
  `0`: never, positive: before the model is handed out). A node warms only the
  tables it keeps, after the weight upload, with `TS_DSV41_ENGRAM_WARM_THREADS`
  (1-64, default 16) threads, each reading 64 MiB blocks with `pread` on its own
  file handle. Before reading it checks the budget: the cgroup's limit
  (`memory.limit_in_bytes` on v1, `memory.max` on v2) minus its anonymous and
  shmem memory minus 8 GiB, and the host's `MemAvailable` minus 8 GiB. Tables
  that do not fit are not warmed, and the log line gives the numbers. S9 sized
  the warm against `MemAvailable` alone, which on the VMs is host-wide and let a
  warm fill the 93.1 GiB cgroup. Each block is first checked with `mincore`, and
  only the pages that are not resident are read, so a re-warm after an eviction
  costs only what was lost. `mincore` reports every page of a file resident to a
  process that may not write it (Linux 5.2 and later); such a process (not root,
  read-only file) reads every block. With the default ownership, node 1 of the
  1017 keeps both tables (126 GiB), which its cgroup cannot hold, so it refuses
  the warm and its rows stay cold reads, made 16 at a time. A page-cache
  eviction (a `cold` benchmark row) stops a background warm that is still
  running.
- **Weight page cache.** `TS_DSV4_LOAD_DROP_CACHE` has the native loader's
  meaning (see [Load time](#load-time)): unset drops each weight chunk's page
  cache once the loader has read it exactly when the bytes the node uploads plus
  the tables it keeps plus 8 GiB exceed its cgroup limit (else `MemTotal`); `0`
  never, `1` always; Linux only, and never for shards on tmpfs. A 1017 node
  uploads ~76 GiB in a 93.1 GiB cgroup. With `balance` each node keeps 63 GiB, so
  both drop, and the next process's weight read no longer evicts the warm slice;
  with the default ownership node 1 keeps 126 GiB and drops, and node 0 keeps no
  table and does not. The kernel drops whole folios only, so a folio that
  straddles two chunks can stay.

None of these settings has been run on the full 1017 checkpoint: the S9 figures
above describe the engine before them, and their effect there has not been
measured. The defaults turn on only what is bit-identical and low-risk (the
parallel rows, `MADV_RANDOM`, the warm budget and the load's drop rule); the
prefetch, `balance` and the deferred send stay opt-in until a two-node run has
measured them.

Tests: `Dsv41EngramReaderTests` (the parallel gather against the S9 gather, byte
for byte at 1, 7 and 256 tokens and 2-32 threads; the reads on the reader's own
threads, never on the .NET pool; a gather ahead of a busy read-ahead, with at
most `TS_DSV41_ENGRAM_THREADS` rows in flight; the read-ahead stopped and waited
for; a failed gather; the preview against the forward's hashing; the fixture
engine bit-identical with the S9 gather, the default reader and two other
settings), `Dsv41CudaDistributedTests.Engram` (the 1017 owners from its header;
a two-node loopback with the table kept by the other node bit-identical to one
process, to `local` and to the unset default, with the message buffers released;
the digest reading unset as `local`; a different owner setting refused at
startup; an owner failing before its rows),
`Dsv41CudaDistributedTests.PerfSwitches` (each switch alone, all together and
the defaults, byte-identical to every switch off in one process, on two nodes
over TCP and on two nodes of two GPUs), `Dsv41CudaDistributedTests.Eviction`
(an eviction stops the read-ahead first) and
`MappedTableWarmTests` (cgroup v1/v2 parsing, the budget, and on Linux the
`mincore` re-warm, the drop and the advice;
`eng/validation/wsl-host-page-cache-tests.sh` runs them under WSL).

### Per-stage oracle against `ggml_cuda`

`eng/validation/dsv41-stage-oracle.py` checks `--backend cuda` on real weights
one stage at a time. It reads two traces of the same tokens: one from the direct
engine (`TS_DSV4_CUDA_TRACE_DIR`) and one from `ggml_cuda`
(`TS_DSV41_TRACE_DIR`, with `TS_DSV41_YOCO_KEEP=0`). It recomputes every stage
in float64 from the GGUF weights and from each engine's own traced inputs to
that stage, so each engine is judged on the rounding it adds there, not on what
it inherited.

Its gate requires:

- the direct engine's error at each stage to be at most 1.25 times ggml's, per
  call and in aggregate (F32 stages at most 1e-6);
- committed cache rows to equal the trained-format replay, up to ties;
- every sparse selection and routing pick to be an exact top-k of the engine's
  own scores, within derived F32 bounds;
- ggml's own error to match a precision model of ggml-cuda's arithmetic, and
  both engines to fit the oracle's attention and MoE semantics better than
  perturbed variants. These two-sided checks are there so that an error the
  oracle shares with the direct engine fails too.

[README-dsv41-stage-oracle.md](../../eng/validation/README-dsv41-stage-oracle.md)
has the rules, the limits, and how to capture and judge a trace pair. Long
prompts are captured in layer windows through `dsv41-trace-sink.py`.

On October 30, 2026 it was run on 2x A40 against a 10-layer truncation of the
1017 `UD-IQ2_XXS` file. The truncation keeps blocks 0-3, 20-24 and 36, with the
candidate budget cut to 65 blocks so that pruning engages. The prompts were 68,
773, 2,003 and 4,204 tokens long, each including 4 decode steps.

- Every check on the direct engine passed at every length.
- The gate passed at the first three lengths.
- At 4,204 tokens the gate failed one check of ggml's own arithmetic. At layer 4
  the precision model replays ggml's `attn_out` at 0.405 of its error, against a
  limit of 0.40. The cause is that ggml rotates the compressed rows it commits
  with float32 angles, while the oracle rebuilds those rows with exact angles.

The run covers ten layers in one process with `--layer-split 2`. It is not a
full-model gate; the next section is.

### Full 1017 on `cuda`, 2 nodes x 2 A40

On October 30, 2026 the whole 1017 `UD-IQ2_XXS` file (unsloth revision
`60ad3651…`, 300.5 GB) ran on the direct engine across two pods with two 46 GB
A40s each: `--backend cuda --layer-split 2 --tp-node-id {0|1} --tp-peers ...`,
the driver running the last stage, `--n-cpu-moe 0`. The build was TensorSharp
`501add66` (the same tree as `main` at `9e0d9926`) plus the change that adds this
section, measured before it was committed. Apart from documentation and the
validation harness `eng/validation/run-multinode-case.py` and its test, that
commit's code is identical to the measured build. ggml was unmodified
`facdc44a`; CUDA 13.2, driver 595.91. The pods compiled the direct engine's
kernels with nvcc 13.2 for sm_86, so the committed PTX that a machine without
nvcc loads was not used. Both nodes ran the same native library. Both engines
used F16 KV, `TS_DSV4_UBATCH=256` (the direct engine's default for V4.1),
`MAX_CONTEXT=65536`, `TS_DSV41_ENGRAM_DEVICE=0` (both Engram slices host-mapped
on node 1), `TS_DSV41_ENGRAM_WARM=0` and `TS_DSV41_YOCO_KEEP=0`, so `ggml_cuda`
ran exact prefill, the only prefill the direct engine has. Every run logged the
cut (12, 22, 31) on both nodes: node 1 holds layers 0-21 and the embedding,
node 0 layers 22-39 and the head. The nodes talk over an SSH tunnel, as in the
`ggml_cuda` measurements under [Current limits](#current-limits-and-tensor-parallel-work).

**Numerics.** `DeepSeek41DistributedProbe` ran three texts teacher-forced
(chunk 256, 128 decode steps): two excerpts of this repository's English
documentation and one of its C# sources. It saved every logit vector: whole-prompt
prefill, decode, chunked prefill, refill after the chunks, and reset. Each
vector was compared with the same step of a `ggml_cuda` reference recorded on
the same pods with the per-micro-batch sparse-attention fix. The limits are three
times ggml's own floor, ggml at ubatch 128 against ggml at 256 on the 2K text:

| Text (prompt tokens) | Vectors | Mean KL | p95 KL | Top-1 agreement | Decode PPL ratio (127 targets) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Limit | | <= 0.02626 | <= 0.1020 | >= 0.8422 | 0.9623-1.0377 |
| ggml's floor (2K prose) | 266 | 0.00876 | 0.0340 | 0.9474 | 0.9874 |
| 2K prose (1,875) | 266 | 0.00935 | 0.0346 | 0.9774 | 0.9939 |
| 8K prose (8,534) | 292 | 0.01957 | 0.0670 | 0.9418 | 1.0136 |
| C# code (1,507) | 264 | 0.00918 | 0.0429 | 0.9545 | 0.9868 |

All three texts pass. The 8K text uses 74.5% of the mean-KL allowance, and
there is no 8K floor to compare it with. A repeat of the 2K probe is bit-identical
(266 of 266 vectors). Within every probe the last chunk equals the whole-prompt
prefill, each refill step equals its decode step and the reset equals the first
prefill, bit for bit.

**HTTP suites.** The server ran on `cuda` and then on `ggml_cuda`, from the same
build with the same settings. Each ran the suite of the 2 x 2 `ggml_cuda` runs
below (`eng/validation/run-http-quality-suite.sh`: answers, JSON, multi-turn and
tool calls), a Paris/42 smoke on `/v1/chat/completions` with and without
streaming, and the ten V4.1 tool-policy scenarios of
`benchmarks/engine_comparison/validate_deepseek41_tools.py` (DSML tool calls,
one request at a time). Both passed the same 25 of 26 checks. On both, the
failure was `tool_round_trip`: the right JSON inside a code fence, which the
strict check rejects.

**Throughput.** Both engines ran the same benchmarks (random tokens, 128
decoded tokens, four runs per process) in one session, interleaved per workload
as `cuda`, `ggml_cuda`, `ggml_cuda`, `cuda`. Before every process except the
warm ones, the Engram page cache was evicted on both nodes. A warm process
repeats the token sets of the cold process before it. The table gives medians of
runs 2-4 over both processes of each engine (six runs), in tok/s:

| Workload | `cuda` prefill / decode | `ggml_cuda` prefill / decode | `cuda` / `ggml_cuda` |
| --- | ---: | ---: | ---: |
| 512 tokens, cold | 98.7 / 8.0 | 152.9 / 27.1 | 0.65 / 0.30 |
| 2,048 tokens, cold | 137.6 / 9.7 | 207.9 / 27.8 | 0.66 / 0.35 |
| 2,048 tokens, warm | 547.5 / 28.6 | 445.2 / 33.2 | 1.23 / 0.86 |
| 8,192 tokens, cold | 306.1 / 17.7 | 218.5 / 27.7 | 1.40 / 0.64 |
| 20,480 tokens, cold | 335.8 / 16.6 | 206.0 / 27.2 | 1.63 / 0.61 |

With the Engram rows in the page cache, the direct engine prefills 23% faster
than `ggml_cuda`'s exact prefill and decodes at 86% of its speed. With them on
storage it decodes at 8-18 tok/s.

**Throughput floor.** This first full-model run had one throughput target, a
floor: in the same Engram state, prefill of 2,048 tokens at least 0.6x
`ggml_cuda`, and decode at 2K and 8K tokens of context at least 0.8x. Prefill
meets it (0.66x cold, 1.23x warm). Decode misses it whenever the Engram rows
start cold: 0.35x at 2,048 tokens and 0.64x at 8,192. Warm, it meets it at 2,048
tokens (0.86x); there is no warm 8K row.

Profiling points to the Engram row gather as the most likely cause. On the node
that owns the Engram layers, the direct engine gathers rows from the mapped table
in parallel only across tokens (`DeepSeek4CudaExecutor.GatherEngramRows`). At
decode there is one token, so it reads 24 rows per table, from two tables, on one
thread, and every uncached row is a storage round trip in series. `ggml_cuda`
reads them with 16 I/O threads. With `TS_DSV4_PERF=2` at 2,048 tokens, node 1
took 84-95 ms per decoded token cold and 20 ms warm on the direct engine; on
`ggml_cuda` its input preparation took 5-7 ms per token cold and its compute
15 ms. The direct engine does not time node 1's work stage by stage, so the cause
is inferred from the code and these timings, not measured directly. The planned
fix, and the next performance change, gathers the rows in parallel across rows as
well as tokens.

The cold medians also favour the direct engine at 8K and 20K tokens, for a reason
other than speed. The direct engine leaves the rows it read mapped and cached:
node 1's process grew to 89 GB resident at 20,480 tokens, with the 100 GB
container at its limit. Later runs on fresh tokens then find more rows already
cached, and its 8K prefill climbs from about 140 tok/s in run 1 to up to 405 in
run 4. `ggml_cuda` stayed under 19 GB in the container, and its later runs gain
less (8K: 148-181 tok/s in run 1, 188-228 in runs 2-4). Run 1 of every process
uses the same tokens. On run 1 alone,
`cuda` / `ggml_cuda` is 0.61, 0.63, 0.86 and 0.95 for prefill at 512, 2,048,
8,192 and 20,480 tokens, and 0.29-0.53 for decode.

These figures need the same Engram state on both engines, and they are not a
serving claim: every row is one sequence through an SSH tunnel. The
`TS_DSV41_YOCO_KEEP=0` rows on `ggml_cuda` are exact prefill. Its default YOCO
replay (approximate prefill) was not run here.

**Memory.** The direct engine allocates its working set at load and holds it,
at a 65,536-token context. On node 0's two GPUs and node 1's two GPUs it used
40,539 / 41,111 / 40,433 / 39,791 MiB in every probe and benchmark, from 512 to
20,480 tokens, and at most 31 MiB more in the server. The highest is node 0's
second GPU, which holds layers 31-39 and the head. In the same session
`ggml_cuda` peaked there at 41,611 (512 tokens) to 43,149 MiB (20,480 tokens).
A 267 MiB allocation from outside TensorSharp appears on every GPU at each
quarter hour. With it, the measured worst case is 41,384 MiB for the direct
engine (in the server) and 43,416 MiB for `ggml_cuda`: 3,672 and 1,640 MiB under
44 GiB.

**Not covered:** YOCO replay (the direct engine has none), and `ggml_cuda`'s
throughput with its default YOCO replay as a baseline in the same session;
routed-expert TP; more than one sequence (distributed runs serve one mirrored
sequence, so no slots, batched decode, prefix reuse or rewind); other KV types,
contexts, ubatch widths or chunk sizes; Engram on the GPU or warmed;
`TS_DSV4_UBATCH` left unset (the runs set the default, 256, explicitly); numerics
on prompts longer than 8,534 tokens (the benchmarks reach 20,480 but check no
logits); warm throughput at 8,192 and 20,480 tokens; server requests with
real-text prompts of about 1.8K and 7.3K tokens (the throughput rows use random
tokens); the committed PTX on the full model (only the fixture and kernel tests
have run on it, on A40); `--n-cpu-moe`; and any other quantization of the 1017.

### One backend per GPU

The architecture-specific DeepSeek ops (compressors, attention prologue and
epilogue, MoE routing and reduction, clamped SwiGLU, hyper-connection gates,
top-k masks) are emitted as `GGML_OP_CUSTOM` nodes and run by a TensorSharp
backend. That backend **wraps** its GPU's CUDA backend and takes its place in
`ggml_backend_sched`: it claims the CUDA device's ops and buffer types as well
as its own, forwards ordinary nodes to CUDA as graph views, and launches the
fused kernels on the same stream.

Registering the two backends side by side instead splits the graph wherever the
backend changes, which a V4.1 layer does about fourteen times. A decode graph
was cut into 565 splits of roughly six nodes each, and the scheduler performs a
blocking host synchronization at every boundary. Wrapping brings that to one
split per GPU. The earlier dispatch diagnostic below isolates this mechanism;
it is not a 1017 full-model benchmark:

| | Splits per decode graph | Decode compute |
|---|---:|---:|
| Side-by-side backends | 565 / 577 | 26.6-27.1 ms |
| Wrapping backend | 8 | 23.2-23.5 ms |

Prefill is unchanged by this, as expected: a prefill split already does
milliseconds of work, so a per-boundary synchronization was noise there. Set
`TS_DSV4_FUSED=0` to fall back to stock CUDA kernels for these ops entirely.

Startup reports each initialized compute device and the routed-expert CPU
offload count. With `--backend ggml_cuda` and no CPU-offload option, all 40
layers run on CUDA devices. The `auxiliary CPU worker pool` message describes
the scheduler's host pool; it does not indicate CPU-only inference. Engram
lookups use host memory only when the tables are host-mapped. Layer split executes
successive layers on successive GPUs, so low per-device utilization alone does not establish a
CPU fallback. `TS_DSV4_PERF=2` reports input preparation and graph-compute times;
`TS_DSV4_PERF=3` additionally logs actual scheduler backend transitions. These
are diagnostic modes whose logging overhead affects throughput. See the CLI
execution investigation, `docs/validation/deepseek41/cli-gpu-execution/README.md`
(local validation evidence, not committed).

On `ggml_cuda`, V4.1 attention runs TensorSharp's owned F32 kernels (ggml's CUDA
flash attention narrows Q and the softmax weights to F16, and the cache
quantization can amplify those lost bits). **Prefill is sparse by default** once
a launch is wide and long: a chunk of more than 8 queries over at least 2,048
keys (the raw window ring plus the visible compressed rows) attends only to each
query's sliding window and indexer selection, at most 128 + 512 keys, through a
mask-compacted kernel. Everything else keeps the dense kernels: single-token
decode and every DSpark verify (6 rows) the split-key kernel, so a verify still
commits exactly the cache rows decode would; shorter prefill the tiled one, so a
prompt whose attention stays below 2,048 keys is unchanged bit for bit. A row
with more visible keys than the bound falls back to a full scan, so the bound
never drops a key.

Measured on one A40 (`GgmlOpsCudaAttentionPrecisionTest --benchmark-dsv41-prefill
512 33536 64 5`, which selects the kernel through the production gate and the
variable): 512 queries over 33,536 keys with 64 heads took **34.1-34.3 ms** per
launch sparse against **1,547-1,549 ms** tiled. Against a decomposed F32
reference the sparse kernel's maximum absolute error was 1.1e-7 (relative L2
7.4e-7) and the tiled kernel's 8.9e-8 (4.8e-7). A sparse query is also
independent of the other queries in its launch -- query 0 alone, in 9 queries
and in 512 is bit-identical -- so a prompt's result does not depend on how
prefill chunked it into micro-batches. (Splitting a prompt into separate prefill
CALLS is different under YOCO bounded replay, which keeps each call's own last
140 positions; see below.) With bounded replay off, splitting a prompt into
calls that end on micro-batch boundaries (every call but the last a multiple of
the prefill micro-batch, `TS_DSV4_UBATCH`) does not change the result either: a
micro-batch's key extent, and so whether it runs sparse or tiled, depends only
on its own positions. A call that ends inside a micro-batch -- calls shorter
than the micro-batch, or a prefix reuse or rewind that resumes at an arbitrary
position -- groups the following positions into different micro-batches, and
the projections and MoE are not invariant to the micro-batch width, so such a
split agrees with the whole prompt only to F32 rounding as the model amplifies
it. Earlier builds padded every micro-batch to the bucket of the whole call's
end, so a prompt prefilled in one call and in 256-token calls ran some
micro-batches on different kernels and left different caches
(`Dsv41PrefillSplitTests`). The fix kept the chunked result: a whole prompt's
early micro-batches now run tiled where earlier builds ran them sparse, so its
caches and logits can differ from those builds' once a micro-batch's padded
width used to reach 2,048 keys (from 1,025 prompt tokens at micro-batch 256).
`TS_DSV41_SPARSE_FA=0` restores tiled prefill.

CUDA runs these compacted launches on an MQA kernel: V4.1's 64 heads share one
K=V head, so a block loads each key tile once for four heads instead of once
per head. That made the sparse cost constant at 16.1-16.3 ms for 1,024 queries
(`--benchmark-sparse 1024 KEYS 64 5 640`; 512 x 8,960 keys: 8.0 ms against the
previous 34.4 ms, relative L2 5.2e-7), while tiled grows about 9 us per key
(15.9 / 22.5 / 63.8 ms at 1,792 / 2,560 / 6,656 keys). The crossover is near
1,800 keys, so the gate moved from 8,192 to 2,048 keys.

This owned gate (more than 8 queries, at least 2,048 keys, F32 compacted kernel)
is not the gate of ggml's flash-attention kernel, which the non-owned attention
path uses (non-CUDA GPUs, the CPU backend). There `TS_DSV41_SPARSE_FA=1` still
opts into ggml's mask-compacted flash attention for single-token batches or at
least 16,384 keys, whose F16 operands measured up to 7.8e-4 relative L2 against
the CPU oracle; that hint stays opt-in. Sparse attention reduces attention work;
it does not remove the cross-GPU copies of the shared compressed cache. On a
layer split those copies are now only each step's new rows, written into a
mirror on every GPU whose layers read the cache (`TS_DSV41_KV_MIRRORS`, on by
default), instead of the selected rows or the whole visible cache on every read.
Logits are bit-identical either way; five A40s decoded 32.6 -> 34.5 tok/s at a
2K context.

YOCO bounded replay, DeepInfer's YOCO prefill truncation, is on by default
with DeepInfer's keep of 140 (`TS_DSV41_YOCO_KEEP`). From the first ratio-1 layer
(layer 20 of 40) on, every layer reads the compressed cache that layer 20 writes
from its attention input, so those layers run only over a prefill call's last 140
positions, and each replayed position's sliding window is cut to the replay. That
is an approximation. With unsloth's UD-IQ2_XXS file on five A40s
(`--layer-split 5`, random tokens) it raised prefill from 352.7 to 500.3 tok/s at
512 tokens, from 463.5 to 844.3 at 2,048 and from 444.3 to 906.2 at 8,192
(`--tp 5`: 562.9 -> 1,046.7 and 540.8 -> 1,082.7). Teacher-forced wikitext-2 at
4,608-token windows (2,304 tokens prefilled, the rest scored one token at a
time, 10 chunks) moved perplexity from 2.4820 to 2.4970 (+0.6%; per chunk
t = 4.85) and the mean KL divergence from llama.cpp from 0.0160 to 0.0172. The
cost sits in the first tokens generated after a prompt: KL divergence 0.026 over
the first 128 against 0.015 exact, 0.018 against 0.015 up to 512, and 0.016
against 0.016 after that. With 2,560-token windows (1,280 prefilled) perplexity
moved from 2.4867 to 2.5193. A passcode sentence planted at 10%, 50% and 90%
depth of ~6.3K- and ~19K-token prompts was retrieved in all six cases with and
without it. `TS_DSV41_YOCO_KEEP=0` runs every layer over every position. Decode
steps, DSpark verifies and calls with image tokens always run every layer, and
so does a one-token prefill micro-batch: with `TS_DSV4_UBATCH=1` the default
notes that replay is inactive and an explicit value is refused. A rewind into
the positions an earlier call skipped (more than 13 tokens before that call's
end, and after its start) re-prefills instead of reusing the cache, and says so
once in the log. The server prefills a long prompt in calls of up to 4,096
tokens (fewer under concurrency), and each call keeps its own last 140
positions. A checkpoint whose layout cannot replay runs exact prefill with one
note under the default, and an explicit value is refused there; one with a
sliding window wider than 132 tokens keeps at least the window + 8 by default.
Only the ggml executor implements it: `--backend cuda` and `--backend cpu` note
once that their prefill is the exact one and refuse an explicit non-zero value.

`TS_DSV41_COMPACT_RAW_GATHER=1` opts into raw-window compaction for sparse
single-token decode. It gathers the 128 visible raw rows on their owning GPU
before moving them to the shared compressed-cache GPU. Masked duplicate rows
pad the raw prefix to 256 rows, so the combined 768-row K tensor satisfies
CUDA's 512-component attention alignment. The physical ring, prefill path,
and non-gathered decode stay unchanged. This option defaults off. The qualified
Q2_K comparison at an approximately 8k prompt improved sustained decode by
13.7% at concurrency 1 and 4; strict flash-arithmetic differences and the
complete measurement settings remain in the validation report.

The default context allocation is capped at 65,536 tokens unless `MAX_CONTEXT`
is supplied. `TS_DSV4_UBATCH` controls the forward microbatch. Unset, V4.1 on a
ggml GPU backend lets the loader choose it: 1024, 512 or 256, the widest that
needs no more routed-expert CPU layers than 256 would, logged as
`[dsv4] prefill ubatch: N (auto; ...)` (see
[Device memory held back for the graph](#device-memory-held-back-for-the-graph)).
The CPU executors and the direct-CUDA engine keep 256. Any explicit value is used
verbatim; `TS_DSV4_UBATCH=256` restores the previous fixed default. A larger
advertised model window does not establish that a particular GPU configuration
can allocate or efficiently serve it.

### Token-batched decode

When several sequences decode at once, all of their tokens run in **one graph**
instead of one graph each. Batching shares weight reads and kernel launches:
only the parts that touch a sequence's own state fork
per slot, which for V4.1 means the sliding-window ring, the compressor state,
the lightning indexer's selection and the attention itself. The Engram lookup
needs no fork at all -- its staged rows are already one column per token, so a
slot is just a column, hashed against that slot's own history.

The saving is bounded by routing: each token picks its own 6 of 384 experts, so
the routed-expert reads do not overlap between slots and only the dense
projections, the shared expert and the output head are shared. Use the 1017
whole-batch wall times above for the measured concurrent workload; the older
Q4_K_M throughput sweep is not a current-model concurrency estimate.

Earlier batching correctness diagnostics also show why output checks matter.
Batching changes GEMM shapes, so a batched step and a solo step are not
bit-identical and a near-tie in the logits can pick a different token. Solo
decode repeated its own output on 6 of 6 greedy prompts; a batched step matched
the solo text on 2-3 of 6. The same prompt run at batch width 2 and at batch
width 4 — the same code path, only wider GEMMs — disagrees at the same rate, so
what changes the output is which requests share a step, not the per-slot wiring.
Set `TS_BATCHED_FUSED_DECODE=0` for a serial path and its determinism.

Four concurrent 10,836-token documents, each hiding a different secret, were
answered with 4/4 correct secrets and no answer containing another slot's
secret, which is the check that the per-slot rings, compressed caches and sparse
selections really are separate.

### Device memory held back for the graph

`TS_DSV4_VRAM_RESERVE_MB` overrides the per-device headroom the layer-split
packer leaves unspent. The default prices the indexer's top-k transients, one
microbatch of activations and a 2 GiB floor. Excess headroom can force more
routed-expert layers onto the host; inspect the placement log for the chosen
checkpoint instead of reusing an older Q4_K_M throughput estimate.

Both that reserve and the raw sliding-window ring grow with the prefill
micro-batch, so with `TS_DSV4_UBATCH` unset the loader prices the split once per
candidate width. At 65,536 context the default reserve is 2,318 / 2,588 / 3,128
MiB per device for 256 / 512 / 1024, and the ring 512 / 768 / 1,280 rows. It
takes the widest candidate that needs no more routed-expert CPU layers than 256
would -- or than an explicit `--n-cpu-moe` the run pays anyway -- and that does
not move GPU-resident Engram tables to the host. A wider chunk is cheaper per
prefill token: one Q4_K_M-shaped routed-expert layer (`GgmlOpsDsv4MoeWidthBench`,
uniform top-6 routing, one A40) took 35.3-35.7 / 36.9-37.2 / 38.9-39.1 ms a chunk
at 256 / 512 / 1024 tokens resident on the GPU, 3.6x cheaper per token at 1024,
and 112.7-117.9 / 185.0-189.1 / 349.9-353.3 ms on 32 host threads (0.44-0.46
against 0.34 ms a token). But an extra host layer is paid on every decoded
token, so that trade is never made. The log line names the width and, when a
wider one was declined, why. A `--n-cpu-moe` below what 256 needs is refused with that number
(`Re-run with --n-cpu-moe N`). 2048 is not a candidate: the reserve was
validated at 1024 (a 57,424-token prefill peaked with 1,522 MiB free against a
3,072 MiB reserve) and a ggml device OOM ends the process. The choice is exported
(`TSGgml_Dsv4UBatch`) so speculative prefill chunks to the same width.

Priced with the Q4_K_M release's tensor sizes and the per-device budgets its
seven-A40 run implies (free memory after load plus what the load placed, 45,091-
45,123 MiB per device), all three widths need 6 routed-expert CPU layers at 65,536
context, so the loader picks 1024; at 131,072 context the same budgets also give
6 at every width. These are computed plans (`GgmlOpsDsv4UbatchPlanTest`), not
measured loads.

The graph cache is bounded by bytes as well as by entry count. An entry's
compute buffers scale with its shape, and concurrent sequences at different
positions produce many distinct shapes: four concurrent 10.8k-token prefills
used to fill all twelve entries and run a device out of memory, which is not a
survivable error -- ggml's allocator frees a buffer before reallocating it, so a
failed reserve leaves a null buffer behind and the process dies rather than the
request failing. Least-recently-used entries are now freed before a new one is
built, until every device has room for another entry as large as the largest one
cached plus a floor. `TS_DSV4_GRAPH_CACHE_HEADROOM_MB` sets that floor (default
1024); `0` restores the pure count cap.

### Load time

The weights are streamed to the GPUs by a pool of reader threads
(`TS_DSV4_LOAD_THREADS`, default 16) in `TS_DSV4_LOAD_CHUNK_MB` chunks (default
64). Each thread walks ONE CONTIGUOUS RUN of the job list. That matters more than
anything else about the loader on a network filesystem: the jobs are sorted by
(shard, file offset), so handing them out from a shared cursor - which is what
this loader used to do - makes every file descriptor read one chunk and then jump
`threads x chunk`, 1 GiB at the defaults. Readahead is per descriptor, so none of
the sixteen streams is sequential.

The following earlier loader I/O diagnostic used the Q4_K_M release on eight
A40s with a MooseFS mount; it is not a 1017 startup-time measurement. It was cold (every
run preceded by evicting all 414 GiB of shards from the page cache, alternating
the two orders twice each):

| job order | weight upload, 294.8 GiB | total model load |
|---|---:|---:|
| one contiguous run per thread | **141-147 s** (2.0-2.1 GiB/s) | **144-155 s** |
| shared cursor (`TS_DSV4_LOAD_CONTIGUOUS=0`) | 360-377 s (0.78-0.82 GiB/s) | 363-382 s |

`TensorSharp.Runtime/GgufReader.cs:330` records the same mechanism for the
managed GGUF prefault ("~3x slower on MooseFS"). Ranges are split by BYTES rather
than job count, because a tensor's last chunk is a partial one; a thread that runs
out steals from the BACK of the furthest-behind range so its victim keeps reading
forwards.

Three things that look like the fix and are not, each measured on this box:

* **Page-locking the staging buffers.** The host-to-device copies are 87 s of
  thread time against 5,539 s in `fread` - 1.6% of the loader's work. Pinning is
  worth several times that on the copy itself and almost nothing on the load.
* **More reader threads.** Throughput is not monotonic in thread count on this
  filesystem; `TensorSharp.Backends.Cuda/Dsv4/Dsv4CudaEngine.cs:744` records
  2.4 GB/s at 16 threads against 1.0 GB/s at 96.
* **`MADV_WILLNEED` on the host-expert prefault.** 29.7 s and 31.3 s against a
  27.7 s mean for the plain fault-in walk, i.e. no better.

The passes after the upload read with `pread` too. The prefault of the
host-resident experts (`--n-cpu-moe`) and the Engram warm (see
[Host-mapped Engram tables](#host-mapped-engram-tables)) merge their tensors
into file ranges, split the bytes into one contiguous run per thread
(`TS_DSV4_LOAD_THREADS`), and read 64 MiB blocks on a descriptor per thread,
skipping a block whose pages `mincore` already reports resident. They used to
touch the mapping one byte per 4 KiB page. On a network filesystem each of
those faults is a synchronous read capped at the mount's readahead
(`read_ahead_kb`, 128 KiB on the A40 VMs), which is why the seven-A40 lane
(`--n-cpu-moe 6`, Q4_K_M) logged 129.7 s to prefault 48.2 GiB of experts and
311.3 s to warm 103 GiB of Engram tables.

Measured on that VM with `GgmlOpsDsv4FileWarmBench` (built on Linux with the
native tests), 16 threads, 8 GiB ranges evicted before every arm with
`mincore` = 0 checked, three to five repeats, arms alternated:

| pass over 8 GiB | Engram table, shard 00002 @ 20 GiB | experts, shard 00003 @ 9,002,135,936 |
|---|---:|---:|
| `pread` (default) | 2.38-2.54 GiB/s | 2.24-2.47 GiB/s |
| prefault page walk, 256 MiB spans (`TS_DSV4_WARM_PREAD=0`) | 0.62-0.66 GiB/s | 0.68-0.74 GiB/s |
| Engram page walk, 8 MiB chunks (`TS_DSV4_WARM_PREAD=0`) | 0.63-0.69 GiB/s | 0.65-0.68 GiB/s |
| the same range again, already resident (all blocks skipped) | 64-159 GiB/s | 139-187 GiB/s |

`pread` fills the page cache but not the process's page tables, which the walk
also did, and the first prefill reads the experts densely. So the prefault also
reads one byte per page of each block once it is cached: on resident pages that
walk costs 0.004-0.006 s/GiB at 16 threads, against 0.019-0.023 s/GiB for
`madvise(MADV_POPULATE_READ)` on the same ranges. The host-mapped Engram tables
are not prefaulted into the process's page tables; their rows are read a few at a time.

Read-ahead hints are no substitute on this mount. `MADV_WILLNEED`,
`POSIX_FADV_WILLNEED` and `readahead(2)` over an evicted 8 GiB range each left
128 KiB of it (0.0015%) resident ten seconds later. Do not retry them.

A read error in the prefault or the synchronous Engram warm fails the load and
names the shard and offset. `TS_DSV4_WARM_PREAD=0` restores both page walks
exactly.

The offloaded experts are not page-locked unless `TS_HOST_MOE_PIN=1`. Every node
of an offloaded layer's routed experts is assigned to the CPU backend
(`build_moe_host`), and `ggml_backend_sched` never overrides that assignment, so
its op-offload rule never streams those weights to a GPU: only the
`[n_embd, n_tokens]` activations cross the bus, and a pinned expert is never a
DMA source. On the seven-A40 lane (`--n-cpu-moe 6`) the page-lock added 20.4 s
to the load for 48.2 GiB and made those pages unevictable in the same cgroup
that holds the page cache. The load now prints one line saying the offloaded
experts stay pageable. `TS_HOST_MOE_PIN=1` restores the page-lock and its
`page-locked ... GiB of host experts` line; `TS_HOST_MOE_PIN=0` still disables
pinning for every architecture. The other MoE architectures, whose prefill does
stream offloaded experts, keep pinning by default.

The loader never reads an uploaded chunk again, but it reads the host-mapped
weights right after the upload and serves them from the page cache for the rest
of the run, and page cache is charged to the cgroup. So by default each uploaded
chunk's page cache is released once the chunk is on the device exactly when the
upload plus the host-mapped weights plus 8 GiB exceed the host allowance (the
cgroup limit), and kept otherwise or when the allowance is unknown. The load
prints the decision with its numbers:

```text
[dsv4] load page cache: dropping each uploaded chunk's page cache (automatic: 263.0 GiB upload + 151.2 GiB host-mapped + 8.0 GiB headroom exceeds the 326.9 GiB allowance; TS_DSV4_LOAD_DROP_CACHE=0 overrides)
```

Those are the seven-A40 lane's numbers: 414 GiB of reads into a 326.9 GiB
cgroup. Its load logged the expert prefault at 0.37 GiB/s and the Engram warm at
0.33 GiB/s, about half the rate the same page walks measured on that VM with the
cgroup roughly half full (table above). Dropping cannot speed the upload itself,
which already reads at the storage rate, and costs 5.9-7.3 ms per resident
64 MiB chunk on that mount (`GgmlOpsDsv4FileWarmBench --drop-cost`, about 25-30 s
of thread time for 263 GiB). Its gain is expected on the stages after the upload
and has not yet been measured on a full load; the check is a cold load with the
default against one with `TS_DSV4_LOAD_DROP_CACHE=0`. The rule stays conditional
because dropping every time would make each reload of a checkpoint that lives
entirely on the GPUs cold. `TS_DSV4_LOAD_DROP_CACHE=0` never drops (the previous
default) and `=1` always drops. On the eight-A40 box, `=1` did not change the
upload's read thread-time (5,374 s against 5,539 s, inside the run-to-run spread)
and ended the load with ~39 GiB of page cache instead of ~330 GiB.

One caveat when timing this yourself: on a box whose page cache is already full of
the checkpoint, a load can be SLOWER than one that starts with an empty cache,
because the cgroup is at its limit before the first read and every subsequent read
contends with reclaim. Compare like with like - evict the shards first.

### Multi-turn KV reuse

A second turn's rendered prompt is not a continuation of the first turn's cache.
Ordinary chat drops the previous assistant turn's reasoning (see the history
policy above), so the render diverges from the cache exactly one token after the
previous turn's `<｜Assistant｜>`: the cache holds `<think>` there, the render
`</think>`. Everything before that point still matches, and from the third turn
on that is the WHOLE of the previous turn's prompt, because that prompt already
rendered the earlier turns with their reasoning removed.

The native executor therefore supports partial reuse: `TSGgml_Dsv4Truncate`
moves a slot's head back so the matching prefix is kept and only the new suffix
is forwarded. Per-turn prefill is then a function of the newest answer rather
than of the whole conversation. Two conditions bound it, both stated in
`dsv41_truncate.h`:

* **Alignment.** The target must be a multiple of the widest compression ratio
  (2 for the released checkpoint), so no compression block straddles the new
  head. Callers align their reuse length down; it costs at most one token.
* **Depth.** The raw sliding window lives in a ring of
  `pad64(n_swa + n_ubatch, 256)` positions - 512 for the released checkpoint,
  whose window is 128 - so a rewind reaches 385 positions back from the state it
  rewinds from. Generating an answer moves the head thousands of positions past
  the prompt boundary the next turn wants, which is why every slot keeps a
  **rewind checkpoint**: a shadow copy of the two modularly-addressed rings (the
  raw window and the compressor state), taken at the end of every multi-token
  forward, i.e. at a prompt boundary. Decode steps deliberately do not move it.
  The shadow costs `n_embd_head x ring_raw x 2` bytes per layer per slot - about
  21 MiB for the released checkpoint - and `TS_DSV41_REWIND_CHECKPOINT=0` turns
  it off, after which a rewind deeper than the live ring is declined.

An earlier cache-correctness diagnostic used eight A40s with the Q4_K_M release (`--n-cpu-moe 2`, greedy, the
reported prompt then two `continue` turns, `TS_KV_DEBUG=1`), on 2026-09-11, when
the CLI still planned its own reuse (`KVCache.PlanReuse`, which is what
`TS_KV_DEBUG` prints). Since 2026-09-17 the CLI and the server both go through the
engine's radix prefix cache, described below, and `TS_KV_DEBUG` prints nothing
there. The divergence lands exactly where the policy puts it - in both turns the
cache holds token 128821 (`<think>`) where the render holds 128822 (`</think>`),
one token past `<｜Assistant｜>`:

| turn | prompt tokens | matching prefix | plan |
|---:|---:|---:|---|
| 1 | 38 | 0 (cold) | Reset |
| 2 | 2,056 | 37, aligned to 36 | PartialReuse |
| 3 | 5,841 | 2,055, aligned to 2,054 | PartialReuse |

Turn 2 recovers only a constant - the system block, the question and the
assistant header - because the cache past that point is the first answer WITH its
reasoning, which the prompt no longer contains. Turn 3 recovers the whole of turn
2's prompt, and the share grows from there: the matching prefix grows with the
conversation while the re-forwarded suffix stays the size of one answer. Both
rewinds are far outside the live ring (6,529 positions for turn 3) and are served
by the checkpoint.

A decline is a normal outcome, not an error: the caller resets and re-prefills,
which is what happened for every turn before this existed. Plain `deepseek4`
does not truncate at all - its compressor overlaps blocks, so a boundary still
reads the previous block's state rows and aligning the head is not sufficient -
and neither do V4.1's direct-CUDA and pure-C# executors, which have no
checkpoint. `--think` off needs none of this: without the reasoning drop the
render is a pure extension of the cache and reuse needs no rewind.

Across requests this reuse is driven by the radix prefix cache, the default mode
and the path both the CLI and the server take. A finished turn stays resident as
the model's primary cache. The next thinking turn keeps it by rewinding it past
the whole previous answer, and the tree asks the model at admission whether the
slot reaches that far (`CanRewindPrimary`, which reads `TSGgml_Dsv4SlotCanReuse`:
the live ring or the prompt-boundary checkpoint). A refusal means a full prefill,
and the admission line says why, for example `Radix prompt reuse for …: 0/2056
tokens; 2056 token(s) to prefill (rewinding the cached conversation is declined by
the model).` Placement plays no part: `--tp`, `--layer-split` and a single GPU plan
alike. A turn that forwards at least two prompt tokens after its reuse leaves a new
checkpoint behind (the capability's `MinTailPrefillTokens`), so a regenerated turn
does not cost the turn after it its reuse.

That holds for a turn that ran alone and is followed by its own conversation's next
turn, which is how the CLI runs. A turn that overlapped another request ran on a
per-request slot, which is released when it finishes, and the first step the
engine runs for any other request discards a resident primary. So on a server whose
conversations overlap, a thinking turn re-prefills its prompt unless its previous
turn finished with nothing else running and nothing else was admitted before it.

Until 2026-09-29 the tree's donation rule refused every rewind longer than 16
tokens, this one included, so every thinking turn after the first reused nothing
(`kvPlan=Prefill` in the CLI; first reported with `--tp 6`). The rule keeps a deep
cached state for a later request rather than handing it to a request that shares
only its beginning. Declining does not keep a primary cache - the next step the
engine runs discards it either way - so the rule no longer binds one.
`DeepSeek41ThinkingTurnReuseTests` drives such a conversation through the engine
with the real V4.1 chat template. The matching prefix is the same as in the table
above by construction, but the table has not been re-measured through the engine
on the real model.

Finished requests' native slots are retained as well, so more than one
conversation can continue without a full re-prefill, including
conversations whose turns overlap. A retained slot serves at most one later
request, and the tree lets its own conversation rewind it past the 16-token rule
wherever the slot can (`CanMaterialize`, the same checkpoint check): a thinking
turn always has to rewind past the previous answer, so under the rule a retained
slot could serve no thinking turn at all. Another conversation never reaches a
scoped slot, so it cannot take one. If the native side refuses to retain a slot (its budget or
device headroom), the finished turn is not reused and is not advertised either -
before 2026-09-29 the failed attempt left an emptied primary registered, which an
exact continuation then decoded from.
Retention is always on for the native executor, the one that can rewind, except
while a DSpark drafter is loaded. `TS_DSV41_RETAINED_CACHE_MB` (default 2048)
sizes the budget of retained slots; a value that is not a positive number keeps
the default. Measured 2026-09-29 on 6x A40 (`--tp 6`, Q2_K) with four overlapping
three-turn Web UI conversations (`eng/validation/parallel-multiturn-webui.py`):
without retention every turn reused 0 tokens, because each finished slot was
released before the conversation's next turn; with it every conversation reused its own
previous prompt on every turn - thinking on, turn 2 reused 54-62 tokens of 167-210
(the answer before the dropped reasoning is re-rendered) and turn 3 the whole
turn-2 prompt; thinking off, 171-196 of 175-200 and 287-417 of 307-437. All eight
conversations passed their answer checks in both modes.

Native V4.1 requests own independent KV slots. The scheduler therefore sizes
its metadata-only block pool for one context per allowed running request.
The scheduler default permits 16 running requests; the example explicitly
sets `TS_SCHED_MAX_RUNNING_SEQS=4` for four live slots. At context 65,536 and
block size 256, their automatic accounting capacity is 1,024 blocks. This
does not preallocate extra GPU caches or enlarge a request's context limit.
Native slots still require enough device memory when allocated. An explicit
positive `TS_SCHED_NUM_BLOCKS` remains a hard aggregate accounting limit;
setting it below the live requests' combined needs can force prompt
recomputation.

For four concurrent prefills, `TS_SCHED_MAX_BATCHED_TOKENS=4096` permits
1,024 tokens per request in an all-prefill scheduler step. Once any request
starts decoding, `TS_SCHED_PREFILL_CHUNK` caps each remaining prefill; its
default is 256. `TS_SCHED_SOLO_PREFILL_CHUNK` controls a lone request; its
configured default is 8,192, bounded by the total batched-token limit (4,096
by default).
Explicitly setting both chunk limits to 1,024 can avoid a short first request
changing the other requests' prefill sizes, but longer mixed steps also delay
active decoders. Record these scheduler settings alongside
`TS_DSV4_UBATCH` when comparing performance.

The [benchmark matrix](../../benchmarks/engine_comparison/benchmark_config_deepseek41.json)
uses this conservative profile and inherits settings omitted from its per-backend
environment. Set `TS_CPU_MOE_THREADS` explicitly and
`TS_DSV41_COMPACT_RAW_GATHER=0` before launching the matrix to reproduce that
baseline. Set `BENCH_DSV41_GGUF` to the first shard if using the model directory
in this card; the matrix's default directory name differs. Its text scenarios
do not substitute for the separate strict tool, JSON, image/video, and reasoning
checks in the validation report.

### Running on the ggml CPU backend

`--backend ggml_cpu` selects the loader's CPU-only branch: one CPU compute
device instead of enumerated accelerators, every layer on it, and every
V4.1-specific op running the scalar CPU implementation in
`ggml_ops_dsv4_fused_cpu.cpp` rather than a CUDA kernel. Startup prints
`compute devices initialized: 1 CPU device(s)` and
`routed-expert placement: all 40 layer(s) on the explicitly selected CPU device`.

**This is a correctness and portability path, not a serving path.** Every
decoded token reads six of 384 routed experts in each of 40 layers, out of a
selected checkpoint, on general-purpose cores. Those reference kernels run
one worker per node (the V4.1 quantize, candidate-score and candidate-mask
kernels are the exceptions), and the wrapping backend of
[One backend per GPU](#one-backend-per-gpu) is a CUDA object that is not built
here at all, so none of that section's numbers carry over. Treat this backend
as a way to run the architecture where there is no CUDA device — to check a
change, to compare a CUDA result against a host one, or to bring the model up on
a machine that cannot host it otherwise. Do not put it behind a serving
endpoint and do not quote it as a throughput number: this card reports no CPU
throughput because none has been measured.

The server's default backend is `ggml_cpu` on everything but macOS, so omitting
`--backend` on a GPU box selects this path. That used to be a refusal naming
`ggml_cuda`; it now loads, and the load says so once on stderr before any weight
is read (`[dsv41] --backend ggml_cpu: DeepSeek V4.1 will run on ONE CPU
device...`). If you see that line on a machine with GPUs, you wanted
`--backend ggml_cuda`.

What the CPU path *does* have is agreement evidence, and it is per-op rather
than per-checkpoint. The validation report's
[fixture comparison](../deepseek41_validation.md#independent-numerical-reference) records 41/41 elementwise
checks at `atol=rtol=2e-5` against the independent oracle on a local CPU, with
maximum absolute error 5.1633e-6 and 41/41 greedy-token agreement. That covers
the fused ops and the fixture-sized graph. A full-checkpoint CPU run has not
been measured, so CPU/CUDA parity on the real weights is not established here.

```bash
dotnet TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll \
  --model /models/deepseek41-q2-q5/Q2_K-Q5/DeepSeek-V4.1-Flash-EngramQ5-Q2_K-00001-of-00010.gguf \
  --backend ggml_cpu --port 5000
```

The CPU backend reads the same embedded Engram metadata as CUDA. Download the
current GGUF shards; no separate Engram preparation is needed.

`TS_DSV4_THREADS` sets the compute thread count, defaulting to at most 32. That
cap was chosen for GPU runs, where those threads only do auxiliary host work; on
a CPU-only run they are the whole engine, so set it to the cores the run may
actually use. `TS_DSV4_UBATCH` (fixed at 256 here unless set, where `ggml_cuda`
picks the width automatically) and `MAX_CONTEXT` (default 65,536) both cost host
memory rather than VRAM here.

The options that name GPUs behave as follows:

- `--tp N` shards routed-expert dimensions across GPUs. Combined with
  `ggml_cpu` it is refused before the checkpoint is opened, not ignored.
- Multi-GPU `--tp N` and `--layer-split N` requests are refused on CPU backends.
- The Engram tables stay host-mapped: the GPU-resident placement described
  above needs a device to place them on. `TS_DSV41_ENGRAM_DEVICE=1` is
  therefore refused here before the checkpoint is opened rather than accepted
  and ignored, and so is any value other than `0` or `1`; `=0` names the host
  mappings this path already uses and is accepted. Because the tables are
  host-mapped, the [host-mapped Engram options](#host-mapped-engram-tables) —
  `TS_DSV41_ENGRAM_WARM`, `TS_DSV41_ENGRAM_THREADS`, `TS_DSV41_ENGRAM_RANDOM` —
  all apply on this backend.
- `TS_DSV4_VRAM_RESERVE_MB` is subtracted from the one device's free memory
  before layers are packed onto it. That device is the host here, so its
  default reserve (at least 2 GiB; about 2,318 MiB at the 256-token ubatch and
  65,536 context this path uses, see
  [Device memory held back for the graph](#device-memory-held-back-for-the-graph))
  is held back from system RAM, and the loader's refusal
  when the model does not fit is worded for VRAM ("not enough VRAM ... Re-run
  with `--n-cpu-moe N`"). The advice is still the right advice — see below.
- `TS_DSV41_COMPACT_RAW_GATHER` was measured on CUDA and defaults off; it
  changes the graph rather than the kernel, so it is reachable here, but
  nothing on this path has been measured with it enabled. Leave it off.
- `TS_DSV41_SPARSE_FA` does nothing here. It sets flash attention's `n_kv_max`
  bound, and ggml's CPU flash-attention kernel reads only the first three op
  parameters (`ggml-cpu/ops.cpp`), never that one. Setting it is silently
  inert rather than slow or wrong; there is no mask-compacted CPU kernel to
  opt into.

Read from the loader rather than measured: layer weights are allocated on the
selected compute device, which on this path is the host, so they are a private
anonymous allocation rather than an evictable GGUF mapping. That is the one
operational decision worth making before a long load — `--cpu-moe` (or
`--n-cpu-moe N`) moves the routed experts of those layers into the loader's
host context, which *is* served from the GGUF mapping, so the kernel can evict
those pages instead of the process being killed for them. It is also what the
loader tells you to do if the weights plus this context's caches do not fit,
even though it says "VRAM" while doing so. Neither the resident footprint nor
the load time of a CPU-only full-checkpoint load has been measured.

The vision companion follows the text model onto the same backend. It loaded
with a hardcoded `CUDA` until this path existed, which would have pulled a GPU
into an explicitly CPU-only run; a backend with no ggml registry name is now
refused by name instead of attempted.

`--backend cpu` is the pure C# executor (`DeepSeek4CpuExecutor`), which now
implements V4.1's graph as well as V4's: the compressors at ratios 1 and 2, the
shared compressed and indexer caches, candidate pruning, the Engram tables, the
delayed hyper-connection gates and the trained cache quantization. It is held to
`eng/dsv41-reference.py` at atol=rtol=2e-5 by
`InferenceWeb.Tests.Dsv41CpuExecutorTests`, across one-shot prefill, chunk sizes
1/3/5/8 and reset. That gate is fixture-scale — a five-layer, 256-hidden,
16-token F32 synthetic model — so it establishes architectural agreement with
the oracle rather than parity on the released 246 GiB Q2_K weights, and the
tests skip, naming the variable, unless `TS_DSV41_FIXTURE_DIR` names the fixture
directory. Unlike `--backend ggml_cpu` it takes no vision companion: that
encoder is a native ggml component, and `LoadVisionEncoder` throws here, so
image and video input are not available. The native loader's Engram and
attention knobs — `TS_DSV41_ENGRAM_WARM`, `_THREADS`, `_RANDOM`,
`TS_DSV41_SPARSE_FA`, `TS_DSV41_COMPACT_RAW_GATHER` — are inert on it.
Engram configuration comes directly from the GGUF, in either Engram layout
(see [Backends](#backends));
`TS_DSV4_THREADS` defaults to `ProcessorCount` here rather than
min(cores, 32); and `TS_DSV4_CPU_TRACE_DIR` writes the same per-tensor files
that `eng/dsv41-reference.py --output` writes, so the two directories diff
tensor by tensor. Like `--backend ggml_cpu` it is a correctness and portability
path, not a serving one: no throughput, load time or resident footprint has been
measured for a full checkpoint on it.

`--backend cuda`, the direct-CUDA engine, also runs V4.1 with its own kernels and
no ggml, on both Engram layouts (see [Backends](#backends)). It has per-sequence
slots, batched decode, retained conversation state and whole-layer placement
through `--layer-split N`, on one node or across nodes; routed-expert `--tp N`
remains a `ggml_cuda` mode. Its numerical checks:
[`Dsv41CudaEngineParityTests`](../../InferenceWeb.Tests/Dsv41CudaEngineParityTests.cs)
holds the whole engine to `ggml_cuda` and `ggml_cpu` on the 1017-shaped fixture
(see [Backends](#backends)), `eng/validation/dsv41-stage-oracle.py` holds each
stage to a float64 oracle on real weights
([Per-stage oracle](#per-stage-oracle-against-ggml_cuda)),
[`Dsv4ExpertKernelTests`](../../InferenceWeb.Tests/Dsv4ExpertKernelTests.cs)
checks synthetic quantized expert projections against upstream dequantization,
while [`Dsv41CudaSlotTests`](../../InferenceWeb.Tests/Dsv41CudaSlotTests.cs)
checks slot isolation, batched decode, ring rewind and retained state against a
fresh run of the same direct engine, at 1e-5 of the largest logit. A batched
decode step computes each sequence's dense projections bit for bit as that
sequence's own step does: up to 16 rows, Q2_K to Q6_K, Q8_0, IQ2_XXS, IQ3_XXS,
IQ3_S, IQ4_NL, IQ4_XS, MXFP4 and BF16 weights take kernels that compute every row
as one row, which covers all of the 1017's dense layouts: Q5_K (its shared experts
in 13 layers), Q6_K, Q8_0 and BF16 (its compressor and indexer)
([`Dsv4DenseRowInvarianceTests`](../../InferenceWeb.Tests/Dsv4DenseRowInvarianceTests.cs)).
F16 and F32 weights (cuBLAS picks its kernel by row count), the legacy Q4_0 to
Q5_1 layouts, IQ2_S and NVFP4 are not held to that. A BF16 projection runs such a
step one row at a time, reading its weight once per row: at 16 rows that is 6 to
14 times the time of one cuBLAS call on an RTX 3080 Laptop GPU.
These are fixture-scale, per-stage, kernel and state-consistency checks. The
full-checkpoint gate is separate: the whole 1017 on 2 nodes x 2 A40 against
`ggml_cuda`, with its HTTP suites
([Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40)). Model/device-gated tests require
their fixture and CUDA hardware; unavailable cases are skips. The CUDA backend notes,
`docs/validation/deepseek41-cuda-backend/README.md` (local validation evidence, not committed),
record what has been verified and what blocks the rest. `--backend mlx` remains refused.

## Forward graph and state

The native graph uses four residual streams and V4.1's delayed
hyper-connection mixing. Layers 1 and 14 add Engram features selected by
deterministic token n-gram hashes. Token normalization and bucket layouts come
from the GGUF metadata; sequence slots retain separate token histories.

Each attention block includes a 128-token raw sliding window. The first two
layers have no compressed attention; the next 18 use compression ratio 2 and
the final 20 use ratio 1. Compressed KV sources and indexer selections are
shared according to the official causal encoder-decoder topology. Query
projections, cache quantization, inverse RoPE, and the grouped output LoRA
follow the V4.1 graph. Index selection uses the lightning indexer and candidate
block filtering. The MoE uses the shared expert plus normalized selected
routed-expert outputs.

The implementation reuses native quantized matrix multiplication,
`mul_mat_id`, attention, hyper-connection kernels, per-sequence slots, and
graph caching. V4.1 activation quantization and candidate filtering have
dedicated operations. It does not call the pure C# or direct-CUDA V4 executor.

Useful source locations:

- [Architecture gate](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek41Architecture.cs)
  and [managed driver](../../TensorSharp.Models/Models/DeepSeek4/DeepSeek4Model.cs).
- [Native loader and scheduler](../../TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp)
  and [V4.1 graph](../../TensorSharp.GGML.Native/ggml_ops_deepseek41.inc).
- [Engram metadata loader](../../TensorSharp.GGML.Native/dsv41_engram_gguf.h)
  and [hashing and configuration validation](../../TensorSharp.GGML.Native/dsv41_engram.h).
- [Chat renderer](../../TensorSharp.Runtime/ChatTemplate.DeepSeek41.cs)
  and [output parser](../../TensorSharp.Runtime/DeepSeek41OutputParser.cs).

## Chat, tools, and JSON

V4.1 uses explicit BOS and `<｜System｜>` framing, and spaced DSML tags such as
`<｜DSML｜ calls>` and `<｜DSML｜ invoke name="tool">`. V4's unspaced DSML format
is incompatible. The renderer handles system, user, developer, assistant, and
tool history; parallel tool results are reordered by their source call IDs.
String tool arguments preserve whitespace, and incomplete invocations are not
dispatched. Inside complete DSML invokes, the parser also accepts the plain
`<parameter name="...">` and `</parameter>` variants observed from Q2_K.
Unrecognised or malformed parameter markup is rejected rather than converted
to empty or partial arguments. The OpenAI chat endpoint retains incoming tool calls, reasoning,
and tool-result IDs when rendering the next turn.

The V4.1 OpenAI chat endpoint constrains declared tool calls with a
request-local DSML grammar. `tool_choice: "auto"` leaves ordinary answers
unconstrained and activates only after the model opens a calls block. With
thinking enabled, activation also waits for `</think>`, so quoted tool syntax
inside reasoning does not start a call. `required` requires a call to a
client-declared function; a named choice restricts that call to the named
function. `none` prevents tool-call output, and `parallel_tool_calls: false`
limits a calls block to one invocation. Internal skill rounds receive fresh
grammar state. Other model families keep their existing policy behavior.

The tool grammar enforces declared names, required parameters, optional
omissions, primitive types, primitive enums/constants, and recursively typed
objects/arrays. It emits parameters and object properties in schema order.
Nested open objects with no declared properties accept arbitrary JSON maps.
For objects with declared properties, generation emits only those properties,
even when the schema permits additional keys. Function parameter schemas with
no declared properties retain the no-argument function convention. Typed
`additionalProperties` schemas are unsupported and rejected explicitly.
Declared integer `enum`/`const` values must be JSON integer literals in the
signed 64-bit range; other encodings or values are rejected before generation.
Ordinary numeric arguments retain the existing Int64-or-double parser behavior;
the lossless argument guarantee below concerns strings, not arbitrary-precision
JSON numbers.
Unsupported assertions, including type unions, schema combinators, patterns,
numeric bounds and string/array length bounds, return HTTP 400 before
generation. This supported subset applies to tool parameters; JSON response
schemas use the existing separate compiler.

Ordinary strings can use the trained raw `string="true"` representation.
Strings containing reserved DSML delimiters remain representable through
`string="false"` with a JSON string: JSON escapes such as `\u003c` preserve
the exact decoded value without closing the surrounding tool markup. The
same protection is retained when parsed calls are rendered into later tool
history, including strings and keys inside nested JSON. Ordinary history
formatting remains unchanged. Raw strings also reserve the `<param`, `</param`,
`<invoke` and `</invoke` tag families so these mistyped tool tags cannot absorb the
rest of the response as argument text. Literal strings containing those prefixes
use the same lossless JSON alternative; ordinary XML such as `<x>` and comparison
signs remain valid raw text. The strict parser is unchanged. Grammar tests establish syntax and argument
round trips; full-checkpoint tool selection and accuracy are measured
separately, with unconstrained baseline failures retained.

The reasoning renderer uses the reference default effort of 50. Ordinary chat
drops past reasoning; tool-enabled chat preserves it. Cached raw assistant
tokens cannot override this history policy. With thinking enabled, JSON grammar
enforcement starts after `</think>`; otherwise it starts at the first output
token. Prompt and parser tests establish format compatibility, not model-level
tool selection, reasoning quality, or JSON task accuracy.

The Chat Completions endpoint accepts `response_format` with thinking enabled for V4.1
because its protocol declares that delayed grammar trigger. This combination
requires JSON grammar enforcement. To request a
JSON final answer after a tool round trip, retain the tool history and catalog
and send `tool_choice: "none"`. Active tool generation and `response_format`
remain mutually exclusive. Validation checks the assistant content channel;
reasoning-only text never counts as a final answer.
These tool-policy and thinking/JSON guarantees apply to `/v1/chat/completions`.
The existing `/v1/responses` surface does not support the same V4.1 tool-history
round trips or reasoning-plus-JSON combination.

Because V4.1 renders tool declarations and parses DSML calls, it is also
eligible for skills, the code tools (`--code-exec`) and, on the server,
[sub-agent delegation](../multi_agent.md), which is on by default on the chat
paths. No delegation results are published for V4.1.

For V4.1, reaching `TS_THINKING_BUDGET` emits the trained `</think>` token and
continues the final answer within the original `max_tokens` limit. The default
budget is 75% when the requested output allowance is at least 512 tokens.
Smaller allowances have no automatic thinking budget; an explicit positive
`TS_THINKING_BUDGET` still applies. `0` disables that budget.
The closing token also consumes one output token. This transition has managed
test coverage; full-checkpoint thinking workflows are measured separately below.
While this V4.1 policy is active, the repetition guard also requests the same
normal closing-token transition if reasoning enters a detected loop. Repetition
in the final answer still stops generation. Disabling the request or scheduler
repetition guard disables this early transition; cancellation, EOS and the
original output limit retain precedence.

## Current limits and tensor-parallel work

- `ggml_cuda` is the serving backend for V4.1. `ggml_cpu` loads the same
  native graph on its scalar CPU implementations, as a correctness and
  portability path with no measured throughput; see
  [Running on the ggml CPU backend](#running-on-the-ggml-cpu-backend). `cpu`
  runs a pure-C# V4.1 executor checked against the PyTorch reference at
  2e-5, and `cuda` runs V4.1 through the direct-CUDA engine's own kernels,
  which is held to `ggml_cuda` on a 1017-shaped fixture, stage by stage on
  a 10-layer truncation of the 1017
  ([Per-stage oracle](#per-stage-oracle-against-ggml_cuda)) and on the whole
  1017 across 2 nodes x 2 A40
  ([Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40)); both are
  correctness and portability paths rather than serving ones. `mlx` fails before the weights are read, rather
  than loading V4.1 weights into a graph that does not implement it.
- `cuda` and `cpu` read both Engram layouts, including llama.cpp's combined
  one in unsloth's DeepSeek-V4.1-Flash-1017 GGUFs. That is checked against the
  real 1017 header and, end to end, on a fixture rewritten into the combined
  layout. No 1017 weights have run on `cpu`. On `cuda`, a 10-layer truncation of
  the `UD-IQ2_XXS` file has run on 2x A40 with `--layer-split 2`, under the
  [per-stage oracle](#per-stage-oracle-against-ggml_cuda). The 1017 `UD-IQ2_XXS`
  routed experts pass the load-time check when they stay in VRAM (the GPU expert
  kernels decode all of them; a kernel module older than the IQ1_S and IQ2_S
  decoders refuses them by name), while the host matmul behind `--n-cpu-moe` runs
  neither IQ1_S, IQ2_S nor IQ2_XXS. A `cuda` run of the whole file therefore needs
  GPUs that hold all of its experts, on one host or across nodes with the
  distributed layer split. One such run has been made, on 2 nodes x 2 A40 with
  one sequence and the settings in the table below; see
  [Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40) for what it covers
  and what it does not.
- Distributed routed-MoE TP runs only on `ggml_cuda`; distributed layer split on
  `ggml_cuda` and `cuda`. `cpu`, `ggml_cpu` and the opt-in non-CUDA ggml backends
  refuse `--tp-node-id`/`--tp-peers` before contacting a peer, and so does `cuda`
  with `--tp`.
- Distributed routed-MoE TP uses the host-driven exchange, not local TP's
  device-resident one. Per GPU-resident MoE layer and per token, each node
  enters two blocking F32 AllReduce collectives over the cross-node transport:
  the gathered expert activations, 2304 x 6 values (54 KiB), and the
  disjoint output rows, 5120 values (20 KiB). With every layer's experts on
  the GPUs that is 80 collectives and about 3 MiB of payload per decoded
  token (`--n-cpu-moe` layers run on every node instead), so decode
  speed is bounded by the network round trip rather than by the GPUs, and the
  local TP figures in this card do not apply. Attention, shared experts,
  Engram and CPU-offloaded expert layers run in full on every node.
- When the host-mapped Engram tables are larger than the host or container
  memory limit, they can never become resident. The 1017 `UD-IQ2_XXS` table is
  125.9 GiB, larger than a 100 GB cgroup, and on 46 GB A40s each layer's
  roughly 63 GiB slice cannot be placed on a GPU either. Background warming is
  then skipped (the mapped host weights plus 8 GiB exceed the allowance), and every
  token reads 24 rows per table, two tables, as random page faults. On network
  storage each uncached row is a storage round trip: the measurements under
  [Host-mapped Engram tables](#host-mapped-engram-tables) are 5-15 ms of input
  preparation per decoded token and 1.4-2.6 s per 1,024-token prefill chunk
  cold, against 0.9 ms and 56-81 ms warm. Page cache counts against the
  container limit, so rows keep being evicted and the cost does not go away
  over a session, and random-token benchmarks never warm at all. Under
  distributed layer split only the stage that owns the two Engram layers pays
  it; under distributed TP every node does.
- Measured on two RunPod pods with two 46 GB A40s each (sm_86, driver 595.91,
  CUDA 12.8), a 100 GB container limit and per-pod network storage, over an SSH
  tunnel (`eng/validation/multinode-ssh-tunnel.sh`; chained all-reduce 0.61 ms
  at 10 KiB, 2.6 ms at 1 MiB), F16 KV, `TS_DSV4_UBATCH=256`,
  `TS_DSV41_ENGRAM_DEVICE=0` and `TS_DSV41_ENGRAM_WARM=0` (the table never fits
  the cgroup). Both placements fit with `--n-cpu-moe 0`: layer split holds
  77.9 GiB per node (layers 0-21 with Engram on node 1, 22-39 on node 0), TP
  80.8 GiB per node. Medians of runs 2-4, random tokens (Engram cold):

  | 2 nodes x 2 A40, `ggml_cuda` | Prefill tok/s | Decode tok/s |
  | --- | ---: | ---: |
  | Layer split, 512 / 2,048 tokens | 148 / 163-179 | 24.2 / 25.2-25.9 |
  | Layer split, decode at 20K context | | 26.4 |
  | TP, exact prefill, 2,048 tokens | 56-58 | 9.4-9.7 |
  | TP, default YOCO, 2,048 tokens | 139 | 9.2 |

  The first run after a warm page cache reached 406 / 31.7 tok/s, close to the
  local five-A40 figures: on this storage, cold Engram reads dominate layer
  split. TP is network-bound (above). Distributed TP and layer split gave
  bit-identical logits on a 3,400-token teacher-forced probe (82 vectors) and,
  with the selection fix below, on a 23,357-token probe (126 vectors), where
  each placement is also bit-identical from run to run. The HTTP suites
  (answers, JSON, multi-turn, tool calls) passed in both placements except
  one strict check that rejects a correct JSON answer inside a code fence.
- On October 30, 2026 the same two pods, now with CUDA 13.2, ran the direct
  engine (`--backend cuda --layer-split 2`, distributed) against `ggml_cuda`
  with exact prefill, one build, one session, Engram evicted before each cold
  process. Medians of runs 2-4, tok/s, prefill / decode:

  | 2 nodes x 2 A40, layer split, exact prefill | `cuda` | `ggml_cuda` |
  | --- | ---: | ---: |
  | 2,048 tokens, Engram cold | 137.6 / 9.7 | 207.9 / 27.8 |
  | 2,048 tokens, Engram warm | 547.5 / 28.6 | 445.2 / 33.2 |
  | 20,480 tokens, Engram cold at the start | 335.8 / 16.6 | 206.0 / 27.2 |

  Decode on `cuda` misses this run's floor of 0.8x `ggml_cuda` whenever the
  Engram rows start cold: 0.35x at 2,048 tokens and 0.64x at 8,192 (0.86x warm at
  2,048). It is most likely bound by those rows, which the node that owns them
  reads one after another on one thread at decode; that node took 84-95 ms per
  decoded token cold against 20 ms warm. A parallel row gather is the planned
  fix. The long-prompt figures benefit from the rows the direct engine keeps
  cached. The numerics gate, the HTTP suites, the full matrix and the coverage
  are in [Full 1017 on `cuda`](#full-1017-on-cuda-2-nodes-x-2-a40).
- Index and candidate selections use `ggml_argsort_top_k`. ggml-cuda's radix
  top-k (`ggml_top_k` on wide rows) resolves ties and orders its output by
  atomic arrival, so above 4,096 index rows gathered decode summed its keys
  in a different order on every token, and above 16,384 rows the candidate
  selection could keep a different tied block: long contexts differed from run
  to run, and under distributed TP the two nodes, which each run every
  selection, drifted apart. The sort is deterministic, with no measurable
  decode cost at a 20K context.
- Execution uses one GPU by default. `--layer-split N` selects whole-layer
  placement; `--tp N` selects routed-MoE tensor parallelism with F32 activation/output gathers.
  The existing distributed groups extend routed-MoE TP across nodes on
  `ggml_cuda`; attention remains unsharded. Distributed requests use one
  mirrored sequence and omit slots, prefix retention/truncation, vision and DSpark.
- Concurrent requests have isolated sequence slots. On the native executor
  with the CUDA fused backend (`--backend ggml_cuda`), their decode steps run
  as one token-batched graph (see
  [Token-batched decode](#token-batched-decode)). `--backend ggml_cpu`,
  `TS_DSV4_FUSED=0`, a loaded DSpark drafter or `TS_BATCHED_FUSED_DECODE=0`
  keeps them on per-slot forward calls.
- V4.1 DSpark speculative decoding is experimental. The loader accepts a
  `deepseek41-dspark` or unsloth `dflash` drafter (`--draft-model`) on
  `ggml_cuda` and `ggml_cpu` only, refuses it on every other executor, and
  rejects V4 drafters. Synthetic integration tests
  (`DeepSeek41DsparkIntegrationTests`) and initial trained text/image HTTP
  probes with two-GPU layer split and experimental routed-expert TP on
  `ggml_cuda` passed. Broad quality and throughput remain unqualified under
  heavy paging. Separate 24-token text/image pairs matched plain token IDs and
  `max_tokens` finishes in both modes with active DSpark and clean exit. While a
  drafter is loaded, token-batched decode and the retained cache are off.
  Without a drafter, `--spec` (including `--spec-type ngram`) serves standard
  decode.
- The K/V cache is F16 on every executor and `KV_CACHE_DTYPE=q8_0` / `q4_0`
  is **refused at load** (`NotSupportedException`, before the checkpoint is
  opened, from `DeepSeek41Architecture.ValidateLoad`); an explicit `f32` is
  announced on stderr and reported as `f16`. It used to be accepted silently:
  the native graph allocated F16 caches regardless and `KvCacheDtype`
  reported `q8_0`. The caches are not the per-layer K/V tensors the shared
  families hand to ggml flash attention (which reads q8_0/q4_0 K/V at its
  64/128-wide vector-kernel head sizes): they are the MLA latent rows (K
  doubles as V at the latent width, F16-only in the CUDA flash-attention
  kernels) in the sliding-window ring, the compressed and indexer rows, the
  rewind-checkpoint shadows and the DSpark draft rings, written by
  `TSG_DSV4_FUSED_ATTN_PREP` / `TSG_DSV4_FUSED_COMPRESS` and read by
  `TSG_DSV4_FUSED_KGATHER`, the compact and TP gathers and the checkpoint
  copies as F16 rows with no dequantize step, on the ggml CUDA, ggml CPU,
  direct-CUDA and pure-C# executors alike. Quantizing them would mean
  re-typing every one of those kernels; and the checkpoint's own trained
  cache quantization (FP8 E4M3 raw rows, MXFP4 indexer rows, NVFP4
  compressed rows) is already applied before each F16 store, so a q8_0
  block would not shrink the cache's information content. The launch
  examples above pass `KV_CACHE_DTYPE=f16`, which is the only value that
  changes nothing.
- Image/video input requires the separately prepared vision companion.
  The encoder and image/text graph have CPU/CUDA fixture coverage and
  complete-checkpoint media checks under layer placement and routed TP.
  Real-image BF16 feature comparisons exceed
  the small-fixture elementwise tolerance; see the validation report. There is no
  validated audio inference path.
- A compatible llama.cpp V4.1 inference runtime is needed for a same-weight
  comparison. The linked GGUF repository's patch adds conversion support only.
  An unavailable reference does not establish quality or performance parity.
- The full-checkpoint numerical smoke produces the expected tokens but fails
  the strict F32-input oracle comparison (relative L2 0.146216, maximum absolute
  error 2.708920). Quantized activation arithmetic differs from that reference;
  the retained stage analysis
  (`docs/validation/deepseek41/smoke18-reference/README.md`, local validation
  evidence, not committed) does not fully attribute the final discrepancy. Greedy agreement is not
  strict numerical parity.

Extending tensor parallelism to attention requires rank-local graphs, weight
shards, and cache state, with a reduction after attention output before the
next nonlinear residual operation. Existing GLM support in
[ggml_ops_glm_dsa.cpp](../../TensorSharp.GGML.Native/ggml_ops_glm_dsa.cpp) supplies
reusable block-aligned weight slicing and rank-local graph execution with
device collectives. V4.1's eight output groups should remain intact when
sharding query heads and grouped output projections; the single KV head and
shared indexer state can initially be replicated.

The implemented expert sharding uses unequal block-aligned partitions. The
2304-wide dimension contains nine 256-element K-quant blocks: two ranks can
use widths 1280 and 1024, and four ranks can use 768, 512, 512, and 512.
Gate/up tensors split along that intermediate dimension for every expert;
down tensors split along their output rows, each rank reading the complete
hidden activations, so the rows are concatenated rather than summed.
Shared-expert and CPU-offload outputs are counted once. Current
ggml CUDA does not expose the old split-buffer interface; this path builds
and executes the rank-local expert graphs explicitly.

See the [validation protocol](../deepseek41_validation.md) for short and long
prompts, JSON, tool round trips, agent workflows, concurrency, placement, and
existing-model regression checks. Unsupported scenarios remain explicitly
unverified.

An earlier managed validation stage passed **3,651/3,651 tests locally and on the requested
VM**, with no skips, including 84 focused audio/image/API checks on both hosts.
Coverage includes tool-history delimiter round trips, valid partial Unicode
tokens, malformed UTF-8 rejection and image/audio request validation before
streaming. The preceding image-validation stage passed 3,635/3,635 tests on
both hosts, including 91 local focused checks.
Each Runtime DLL is unchanged from its verified 3,597-test raw tag-family
serialization stage. The initial matched placement baseline used the earlier
3,566-stage host; those measurements and the intervening 3,582-stage checks
remain preserved. The inference harness separately passed 33 unit tests locally
and on the VM; its current scope is listed in the
[validation report](../deepseek41_validation.md).
The first image-validation VM attempt
exposed a test admission race
(`docs/validation/deepseek41/retained-cache-admission/README.md`, local validation
evidence, not committed);
the synchronized test class and full lane pass on both hosts with the original
assertions preserved.
Exact commands, exclusions, counters, hashes and the retained intermittent test
failure (`docs/validation/deepseek41/managed-correctness/README.md`, local
validation evidence, not committed) are separate from full-checkpoint quality and
performance results.

The earlier layer and routed-TP correctness profiles use native `6b3b5ab3…` and managed stage 3,651.
Layer split passed **138/138 inference cases**; routed TP passed **129/130**.
The layer plan also includes eight concurrent long-context cases:

| Scenario group | Layer split | Routed TP |
|---|---:|---:|
| Short, JSON, schema, history and default-parallel tool workflows | 30/30 | 29/30 |
| Required, named, none, serial and parallel tool policies | 30/30 | 30/30 |
| Thinking workflows | 4/4 | 4/4 |
| Non-streaming workflows | 4/4 | 4/4 |
| Images and sampled video frames | 25/25 | 25/25 |
| Chinese and Unicode JSON | 10/10 | 10/10 |
| Separately configured serial tool workflows | 10/10 | 10/10 |
| Sustained decode | 15/15 | 15/15 |
| Long retrieval at approximately 8k and 32k input tokens | 2/2 | 2/2 |
| Four concurrent requests at each long-context size | 8/8 | Not in this profile |

These are retained correctness records for older weights and binaries, not
performance estimates for the 1017 checkpoint. The original timing and
concurrent-long accounting remain in `docs/validation/deepseek41/full-checkpoint/`
(local validation evidence, not committed); use the 1017 measurements above
for this card's performance comparison.

The previously failing named-thinking and thinking-agent cases now pass in both
final profiles. In routed TP, one default-parallel agent request still issued
`calculate_total` with zero placeholder arguments alongside `read_invoice`,
before receiving the invoice result. The strict checker rejected it; this
group changed from the first TP profile's 30/30 to 29/30. Separate serial-policy
success does not remove that failure. A local prompt/grammar/parser diagnosis
(`docs/validation/deepseek41/parallel-tool-dependency/README.md`, local validation
evidence, not committed) preserves the model-emitted calls and found no defect forcing the extra call or
zero values. The final four-layer CPU-offload profile separately passed 28/30
default-parallel quality cases, retaining two premature dependent-call failures.
The layer profile's 30/30 result does not remove either placement's failures.
Native code and launch settings also
changed, so these results do not isolate the grammar change or establish a
blanket quality gain. The twelve HTTP input-rejection checks above are separate
from the inference plans. Exact reports and remaining comparisons are in
the final placement records, `docs/validation/deepseek41/final-placements/README.md`
(local validation evidence, not committed).

Existing-model checks also retain regressions. The final 75-case comparison
(`docs/validation/deepseek41/existing-model-regressions/final3651-native6b3/README.md`,
local validation evidence, not committed) passed 39/75 cases and introduced no failures relative to its paired references
in that run; separate Unicode JSON coverage passed 15/15. The subsequent
repeated JSON comparison
(`docs/validation/deepseek41/json-performance/completed-r2/README.md`, local
validation evidence, not committed) exposed an additional failure for an identical request and recorded slower
Qwen3.5 first-token latency despite faster short-answer decode. Those results
remain separate from the earlier run's zero-introduced-failure observation.
They do not establish a blanket absence of regressions.

Qwen3.5's shorter alternating control
(`docs/validation/deepseek41/json-performance/qwen35-alternating/README.md`)
also showed slower final latency. A later 72-request control
(`docs/validation/deepseek41/json-performance/qwen35-solo72/README.md`; both
are local validation evidence, not committed) held the native library fixed, passed every answer and did not reproduce the
slowdown. No production fix was made from these diagnostics; the differing
results and their limits remain in the validation report.
