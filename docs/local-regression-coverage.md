# Local regression coverage inventory

The repository had several useful test suites, but no single local entry point
that combined config-driven provisioning, hardware budgets, CLI and HTTP
scenarios, component tests, and diagnostic reports. The local runner in
[`eng/run-local-regression.py`](../eng/run-local-regression.py) supplies that
entry point. It does **not** make every supported feature a qualified feature
on every machine. Its report must retain unsupported, skipped and untested
cells separately from passes.

The machine-readable inventory is
[`config/regression/catalog.json`](../config/regression/catalog.json), checked
against [`catalog.schema.json`](../config/regression/catalog.schema.json).
It inventories all 67 host presets, overlays and LoRA documents under `config/`
(excluding its own `config/regression/` metadata), 23 architecture/embedding/
special-quantization family records and seven backend names. The audit report
records the current scenario count and execution classification: **44 scenario
definitions = 25 default adapters + 14 typed profile executors + five suite
categories**. `automated` means an execution adapter exists; those with
`execution: "profile"` require an explicit `--scenario-profiles` document and
their approved fixtures. `suite` identifies a globally integrated component/native
lane, which must not be repeated or counted as per-model coverage. None of these
classifications is execution evidence. Each result requires its own
model hash, source identity, backend, input and actual outcome.

## Existing infrastructure reused or indexed

| Entry point | Coverage and limits |
|---|---|
| `TensorSharp.TestMatrix` | Existing subprocess CLI matrix: synthetic prefill/decode, text, uploaded text, multi-turn, tools, thinking, image/audio/video. Its filename family discovery and OS-based CUDA heuristic do not prove device availability or config completeness. Its default model directory was a specific Mac path. |
| `InferenceWeb.Tests` | xUnit runtime/model/backend/server contracts, config parsing and download groups, cache/reset/cancel, structured output, tools, agents, embeddings and native parity. `Category=Bench` is separate. `Requires=Cuda/Mlx/Models/Ggml*` and `GatedFacts` preserve missing dependencies as skips. Environment-independent tests do not qualify native kernels. |
| `TensorSharp.Knowledge.Tests`, `TensorSharp.Studio.Tests` | Independent retrieval/document and studio workflow contracts. Studio tests are not desktop visual end-to-end tests. |
| `TensorSharp.GGML.Native` CTest | The runner can discover and execute an explicitly supplied prebuilt native test directory, sequentially, parsing individual JUnit outcomes. `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON` supplies attention, MoE, quantization, cache, scheduler, TP and model-kernel tests. A separate native build and the relevant device are required; the runner does not infer a build directory or qualify a GPU from CPU tests. Upstream ggml stays unchanged. |
| `eng/ForcedLogitProbe` | Complete-marker and fixed-token-history logit evidence. Only `cpu`, `cuda`, `ggml_cpu`, `ggml_cuda`, `ggml_metal` are accepted. Repeating one backend tests consistency; independent oracle/cross-backend accuracy requires a separate controlled reference. |
| `eng/tests/dotnet-local-chat.py` | Model-free Microsoft `IChatClient` HTTP contract fixture. |
| `eng/tests/dotnet-local-chat-e2e.py` | Live server arithmetic, maintained-history, streaming and cancellation checks; SIGINT subcase is POSIX-only. |
| `eng/tests/NativeChatCompletion` | Model-free native `IChatClient` boundary and adapter checks. Its current `--real-model` check hardcodes `ggml_metal`; it is not live Windows/CUDA native-client coverage. |
| `eng/validation/run-release-profile.py` and `validate-release-*` | Existing explicit server/application identity and family-specific embedding, diffusion, generation and agent profiles. These require their own fixtures/profiles; arbitrary commands are not inferred from a model name. |
| `eng/validation/run-parallel-case.py`, `run-multinode-case.py` | Explicit local multi-GPU and distributed profiles. A one-GPU machine cannot qualify them. Remote machines are never discovered or contacted automatically. |

## Model/config matrix

Backend sets below are code admission or documented routes, **not** a claim of
all-device numerical qualification. The catalog records the source files and
per-family restrictions. A checkpoint's quantization, native library build,
companions and requested scenario may narrow these sets further.

| Family | Config coverage | Backend restrictions and remaining scope |
|---|---|---|
| Gemma 4 | E4B, 12B, 26B-A4B; dense and MoE | Seven routes; matching media/MTP companions needed. Quantized KV is declined. Other sizes/quants require separate qualification. |
| Qwen 3.5/3.6/3.8 27B | Dense 0.8B/9B/27B, MoE 35B-A3B; portable small 0.8B and non-agent 3.8 presets added | Seven routes; direct CUDA whole-model fast path depends on quantization. Image coverage requires matching projector. |
| Qwen3.8 Flash Next | Separate complete three-shard UD-Q2_K_XL and UD-IQ1_M presets, vision; MTP overlay added | Six family routes, MLX refused; IQ1_M preset conservatively omits direct CUDA until its mixed tensor layouts are qualified. Shared MTP head only on GGML. Sparse attention requires f16/f32 KV. No automatic assumption that SSD offload makes the model fit. |
| Bonsai2 | PTQ1_0 + optional projector added | Single-device four GGML routes only. PTQ1_0 repacks about 29% larger; download size is not resident weight size. PQ2_0 and other variants remain separate coverage. |
| GPT-OSS | 20B Q8_0 | Seven routes; MoE, always reasons, text only. |
| Nemotron | H8B/H47B, 3 Nano Omni, 3.5 Lightning added | Seven generic routes; Omni image needs projector, audio needs a separately converted Parakeet companion. Speculation is refused. |
| Mistral3 | Small 3.1 24B + optional projector added | Seven generic routes; no supported thinking/tool protocol. |
| Hunyuan Dense | Hy-MT2 1.8B added | Translation checkpoint; generic arithmetic/chat fixtures do not establish translation quality. Requires reference translation fixture. |
| DeepSeek V4 / V4.1 | Complete pinned shard groups added | Explicit CPU/GGML/direct-CUDA routes; V4.1 non-CUDA GPU paths are gated and excluded. V4.1 vision requires a prepared companion and GGML; the new preset is text-only. |
| GLM5.2 / 5.3 / 5.3 Flash | Complete pinned shards; Flash optional projector added | Documented CPU/GGML/direct-CUDA routes, with separate TP/layer placement gates. 5.2/5.3 are text-only; only Flash accepts vision. |
| Muse-Glimmer | 30B UD-Q4_K_XL + optional Q8 projector added | Immutable publisher metadata confirms the same weight/projector hashes already in TensorAgent. A separate matching DFlash is not automatically enabled; n-gram and learned-draft coverage stay distinct. |
| Muse-Spark | Complete eight-shard UD-IQ3_XXS + optional Q8 projector added | Publisher architecture is `muse-spark`; full checkpoint requires a large-memory explicit plan. Tensor parallelism and expert offload remain separately qualified configurations. |
| MiniMax-M3 | Complete seven-shard UD-Q4_K_XL text baseline added | Only `ggml_cuda` and correctness-only `ggml_cpu`. New upstream projector/drafter files postdate the documented converted companions and are not automatically enabled; vision/learned-draft qualification remains pending. |
| DiffusionGemma / Jev | Q3/Q4 and Jev presets | Separate diffusion generation. No ordinary tools/audio/video_url path. Jev video uses sampled image frames; speech uses an external ASR service. |
| BERT / Nomic-BERT | MiniLM, Snowflake, Nomic | `cpu`, `ggml_cpu`, `ggml_cuda`, `ggml_metal`; no direct CUDA, MLX or Vulkan. |
| EmbeddingGemma2 | Text preset | Six routes, no Vulkan. Media embeddings require the optional matching projector; the text preset alone does not cover media. |
| Qwen-Image-2.1 | Base/Turbo, 14 LoRA plugins | Managed CPU and four GGML routes only; direct CUDA/MLX refused. Requires dedicated VAE and Qwen3-VL encoder. Image editing needs projector and explicit fixture/mask. |
| MiniMax-H3 | FL2VA + Ref2VA and two Turbo LoRA plugins | Six documented routes; Vulkan unqualified. Single device. Audio VAE needed for 32 kHz stereo output. |
| LTX2.5 | Distilled native pipeline and optional Ingredients adapter | Experimental; six accepted names map to three owned executors: `ggml_cuda`→`cuda`, `ggml_metal`→`mlx`, `ggml_cpu`→`cpu`. These aliases do not qualify GGML kernels. Vulkan refused. Ingredients access is separately gated. |
| Wan2.2 | TI2V base/Turbo, two-expert I2V A14B | Six routes, MLX refused; single device. Correct VAE and both A14B expert files required; no audio output. |

The first 12 new family presets copy immutable URLs, SHA-256 and file identities from
`TensorAgent/src/TensorAgent.Core/Catalog/ModelCatalog.Extended.cs`,
`ModelCatalog.cs` and the Bonsai2 card. These are repository-source provenance
checks; they do not claim fresh network validation or model execution.

Five additional presets use fresh metadata-only verification on
2026-11-01. Each publisher manifest was reread at its immutable revision, with
an exact repository architecture, ungated status and complete LFS file
SHA-256/size entries: [Muse-Glimmer](https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/tree/faa5b025c584459c13febfa5c59883516710ae39),
[Muse-Spark](https://huggingface.co/unsloth/Muse-Spark-Open-816B-A42B-GGUF/tree/c00670568b6ce5e56a247f67091247984df2a01e),
[MiniMax-M3.1](https://huggingface.co/unsloth/MiniMax-M3.1-Flash-GGUF/tree/b17b7f5459c0b8f127ea12ee774409b282521d6a),
[Qwen3.5-0.8B](https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF/tree/6ab461498e2023f6e3c1baea90a8f0fe38ab64d0),
and [Qwen3.8 Flash Next IQ1_M](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/766911a6b7369840a91dbcd95f9f997acaab6cd6/UD-IQ1_M).
The 24 selected file identities and metadata response hashes are preserved in
[`source-pins.json`](../config/regression/source-pins.json). No model weights
were downloaded for this verification. Every registered family now has at
least a representative config, which does not cover every size, quantization,
optional companion, backend or scenario.

The small Qwen preset uses `Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-Q8_0.gguf`
(811,843,840 bytes) and optional `mmproj-F16.gguf` (204,987,232 bytes).
Set the runner's `--model-root` to their parent folder to reuse this layout.
These names and sizes match the supplied local inventory; the actual local
SHA-256 values still require verification before execution. The publisher's
metadata identifies the official `Qwen/Qwen3.5-0.8B` base and the implemented
`qwen35` architecture. `DirectCudaRealModelTests` already contains a 0.8B
regression target; that source evidence does not mean it ran in this session.

The IQ1_M preset uses the separate `Qwen3.8-Flash-Next-GGUF` directory layout.
Its three trunk files total 74,538,755,776 bytes, and the optional matching
BF16 projector is 907,542,944 bytes. Their immutable publisher hashes match the
supplied inventory identities; this source audit did not reread the local
weights. The existing Q2_K_XL preset and its file identities remain distinct.
Direct CUDA's expert decoder does not admit GGUF IQ1_M tensors; the quantization
filename alone does not establish the individual tensor types of a mixed
checkpoint. The new preset therefore retains the managed CPU and four GGML
routes, with direct CUDA pending separate tensor-layout qualification. This
is a conservative preset restriction, not a claim that every IQ1_M-named
checkpoint is incompatible with direct CUDA.

No speculative model URL, revision or checksum was invented. Known byte sizes
are copied into catalog `file_sizes` by matching SHA-256; absent sizes remain
unknown until a bounded metadata lookup or explicit budget is supplied.

## Scenarios and assertions

The executable CLI/HTTP cells check semantic output and protocol state, not
just exit codes: prompt text cannot satisfy output assertions; the CLI tool
check validates formatted presentation while the server tool check validates
the typed function envelope and JSON arguments. Structured replies validate
required keys/types/values, and streamed output must end correctly. Reset/cache
uses two synthetic sessions for marker isolation, `newChat` answer equivalence,
disposal with subsequent 404 and positive cached-token evidence. This bounded
fixture is not a general privacy proof.

Cancellation only passes with a unique echoed request ID linked to the stream
ID and a matching request-scoped `ChatAborted` diagnostic with a positive token
count below the 512-token limit. A drained queue and a healthy follow-up request
alone remain untested cancellation coverage, since a request may have completed
normally after the client disconnected.

Embeddings require finite vectors, dimensions, repeat consistency and a labeled
relevance ordering. The adapter also checks generation-route rejection and
empty-input/zero-dimension errors (400), plus an unknown model (404). Media CLI
cells require explicit fixtures, expected answers and positive
encoder/injection evidence.

`retrieval_vector_cpu` adds an owned embedding-only MiniLM host with a separate
CPU knowledge encoder loading the same pinned file. Three synthetic TXT files
exercise Ready indexed metadata and exact weight/prefix/source hashes, a
zero-lexical-overlap dense paraphrase top result, finite RRF scores, scoped
search/read, exact original downloads, malformed/atomic-upload rejection and
document/collection deletion. Default RAM accounts for both encoder instances;
download/disk bytes count the unique file once. This CPU-only adapter is limited
qualification, not an engine backend restriction. The sidecar has no public
backend telemetry, so its CPU configuration/source admission is disclosed
separately from the primary host's API backend evidence. The paired `retrieval`
profile below adds grounded-chat and source-URL assertions. Reranking, OCR/media,
persistence, concurrency and tenant authorization remain outside these bounded
adapters. Collection scope is not a tenant ACL.

Fixed-history consistency captures every compared logit row and the first
divergent case/step/token. Nonfinite values, incomplete output, a mismatched
model/history or backend fallback cannot pass. A fresh same-backend repeat is
not an independent numerical oracle. The current numerical gates require
relative L2 at most 0.001, cosine at least 0.999999 and identical top-1 token.
Maximum absolute error, KL, top-five overlap and top-token margins are retained
as diagnostics; they are not additional passing thresholds.

The sequence consistency scenario also covers short-short, long-long,
short-long-short and long-short orderings with 32/512 prompt tokens and 32
compared logit rows (one prefill row plus 31 teacher-forced decode forwards).
It compares model reuse after `ResetKVCache` with
fresh-process controls and records the preceding case/length at the first
divergence. This evidence can isolate order-sensitive behavior; it does not
itself establish the cause of an MoE failure.

The Jev adapter uses the existing fixed decision fixtures and validates
normalized finite probability distributions, expected choices, score/legend
and question ownership. The distinct `speculation_ngram` adapter compares
an uncontaminated plain baseline with an explicitly selected n-gram candidate,
requiring identical token IDs and finish reason plus positive draft/accept/
verify counters. It does not establish learned MTP/DFlash/DSpark coverage.

`image_generation_smoke` validates four malformed requests and two bounded
128px Qwen-Image outputs, including strict same-origin retrieval, bounded PNG
decoding and visible nonconstant pixels. RGB/RGBA8 and noninterlaced gray/palette
outputs normalize to RGBA8; legal 16-bit/Adam7 encodings are explicitly untested
with their local artifact and header retained. Seeded output hashes are observations;
this adapter does not qualify semantic image quality or universal bitwise
determinism. Invalid bounded payloads remain local diagnostic artifacts.
`jev_text_attachments` executes the 14 manifest cases requiring no extra media
service: text/CSV/JSON/HTML, PDF text layers, DOCX/XLSX/PPTX, a plain-text control,
ordered/reversed source pairs, cross-file decisions and eight attachments. It
checks pinned fixture bytes, all gold decisions, attachment order and extraction
diagnostics, a positive extraction-cache replay and eight exact typed validation
errors. Fixture `.gitattributes` preserves the pinned bytes, including PDF offsets,
on Windows; legacy text CRLF normalization is accepted only when it recovers the
exact pinned bytes, with original and submitted identities recorded. No binary
normalization is accepted. Inline fixtures stay in scenario-local upload storage.
Bounded vision/video fixtures use the separate `jev_media` profile executor.
Audio/ASR and uploaded-reference transport remain untested; office formatting,
embedded figures and formula recalculation are not qualified by text extraction.

`client_http_live` adds four prebuilt HTTP SDK invocations (OpenAI/Extensions.AI,
stream/response) with strict two-turn answers `4` and `12`, exact client identity
headers, clean process completion and raw HTTP/SSE controls. No SDK wire-history
capture, SDK cancellation/deadline or native-client qualification is claimed.
The `client_live` profile adds explicit native-client and HTTP SDK execution.
The native CLI source admits GGML CPU/CUDA/Metal/Vulkan, while its historical
recorded `LiveChecks` fixture is specifically Qwen3.8 Flash Next on Metal; that
existing evidence does not qualify the new profile on another device.

`server_agents` exercises two named read-only arithmetic reviewers using only
host coordination tools, with skills/code execution disabled. Successful spawn
and wait events, completed child snapshots, exact child/final JSON arithmetic,
distinct workspace identities and a separate request-level opt-out must all
pass. Identity separation is not a filesystem-confinement test. Dependency
scheduling, mutable worker tools and cancellation propagation remain untested.
The host uses its normal per-user `TensorSharp/agent-workspaces` storage; the
runner does not redirect or inspect unrelated user workspaces.

## Profile-driven execution matrix

The following 14 scenario IDs have typed executors rather than manual-only
recipes. They require [explicit profiles](local-regression.md#typed-scenario-profiles)
and independent approved inputs/oracles. Their integration evidence is source
review and synthetic harness tests, with the separately recorded local
`native_unit` direct-CUDA and `client_live` GGML-CUDA results below. Every other
row still requires a real model/device run report. No result qualifies Linux,
macOS, remote nodes or other unrun platforms.
Earlier model reports retain their original narrower scope.

| Scenario | Executed assertion scope and requirements | Qualification |
|---|---|---|
| `embedding_media` | Verified EmbeddingGemma2 projector; bounded image/audio/video; finite vector, exact dimension and independent SHA-pinned vector oracle with numerical gates | Harness only; actual model/device run required |
| `image_generation` | Bounded Qwen-Image generation; decode dimensions and nonconstant pixels; compare independent decoded RGBA8 pixel hash | Harness only; actual model/device run required |
| `image_editing` | Approved input/mask and output oracle; decoded pixel identity, changed-region evidence and masked preservation metric; metric dependencies required | Harness only; actual model/device run required |
| `video_generation` | Source-admitted family pipeline; actual returned diffusion seed, complete bounded decode, shape/rate/frame count, per-channel content variance and approved RGB reference error | Harness only; actual model/device run required |
| `video_audio_generation` | Supported audio-producing family plus the video gates; complete decoded PCM sample-rate/channel/length checks and independent audio error oracle | Harness only; actual model/device run required |
| `video_reference` | Family-admitted start/end/reference conditioning with SHA-pinned PNG/JPEG fixtures (<=32 MiB and each dimension <=1024 before loading), pipeline evidence, actual returned seed and complete output oracle; LTX Ingredients execution remains untested | Harness only; actual model/device run required |
| `speculation` | Explicit supported learned head/drafter and benchmark; exact greedy token parity against plain baseline and positive draft/verify/accept evidence | Harness only; actual model/device run required |
| `lora` | Pinned text LoRA, prompt and distinct approved base/adapted answers; load, use, unload and restored-baseline assertions | Harness only; actual model/device run required |
| `multi_gpu` | Explicit physical devices, tensor/layer placement receipts and per-device budgets; teacher-forced token history and numerical parity | Harness only; multiple actual GPUs required |
| `distributed` | Explicit authorized Linux SSH nodes, unique mandatory raw `/etc/machine-id` SHA-256 pins checked before run-directory creation or model launch, pinned preinstalled model/runtime/source manifests, per-node budgets, rank/logit evidence and verified owned-process teardown | Harness only; authorized multi-node run required |
| `retrieval` | Primary chat model plus CPU embedding role; pinned document, Ready/index provenance, exact source bytes, grounded answer and matching source URL, empty/unrelated-scope controls and cleanup | Actual Windows Qwen3.5-0.8B `ggml_cuda` + MiniLM CPU run failed the normal-completion contract with fixed 384 output tokens; document/source checks ran, full profile not qualified |
| `jev_media` | Bounded PNG/MP4/WebM plus typed gold decisions with/without media; positive visual preparation, deterministic replay/cache evidence and typed negative responses; no positive audio/ASR path | Harness only; actual vision-model run required |
| `client_live` | Pinned prebuilt native and HTTP clients; two native modes plus four HTTP SDK modes, exact two-turn answers, requested backend receipts and normal completion | Local Windows `ggml_cuda`, Qwen3.5-0.8B Q8_0: six modes and all 12 semantic answers passed with clean harness `d93f4aa`; retained binary build sources and limits below |
| `native_unit` | Pinned prebuilt `InferenceWeb.Tests.dll`; two fixed direct-CUDA synthetic graph/cache/quantized-weight tests, positive execution/counter evidence and zero fallback; no model load/download | Local Windows direct CUDA: both tests passed, graph capture/replay observed, zero CPU fallback; device-only scope and deployment provenance below |

The local `native-cuda-profile-d93f4aa` report on 2026-11-01 records two
executed/passing tests, no skipped tests, positive graph capture/replay and two
empty CPU-fallback counter blocks. All seven profile assertions passed, the run
exited zero, and Windows Job teardown verified no owned processes remained. It
used clean harness commit `d93f4aaa5fede1a5cb084b14c01e7e49703fd921`, a prebuilt
test assembly with declared build source
`b764dec2d149aabd725c067a47c6f4c5dfedd010`, and assembly SHA-256
`0b7a9e3bbdd4f8e39e0a1d39c7ce544335991f043f4ac501866ab73907816053`.
Its deployment inventory also records the `GgmlOps.dll` candidate SHA-256
`a0f0674576b5193222846db339456d86d089fffdcb9f105f64b6a85426d98128`; that inventory
is not mapped-module attestation or evidence of GGML execution. This result
qualifies only the two direct-CUDA synthetic tests, not model inference,
backend-wide correctness or a rebuild of every assembly from the harness commit.

The local `client-live-profile-d93f4aa` report on 2026-11-01 records
`profile::qwen08-native-http-cuda` passing on the RTX 3080 Laptop GPU. Native
stream/response and OpenAI/Extensions.AI HTTP stream/response each returned the
two required answers, `4` and `12`: six client invocations and 12 answers.
The host and native receipts identify `ggml_cuda`; raw HTTP/SSE controls and
verified owned-server teardown also passed. The Qwen3.5-0.8B Q8_0 weight SHA-256
was `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c`.
Harness commit `d93f4aaa5fede1a5cb084b14c01e7e49703fd921` was clean. All 22
profile assertions passed and the overall run exited zero. Both native and all
four HTTP client processes exited zero; every client and the owned server had
verified Windows Job teardown. The server was deliberately stopped after the
assertions, so its recorded termination exit is not claimed as a natural exit.
The profile's shared source declaration remains
`f6e6e506206d62755efa08755c6d83ed19e249f8`; the native example was built from it,
while the HTTP example reused the `b764dec` deployment.
That shared declaration does not independently establish both binaries' compiler
inputs. Their recorded SHA-256 values are
`220757da421387082df93fdd732369975165c253ff094ecd4bee1a06e5959d98` (native) and
`5b2b8e673b91a78dd7b655c07e536720d907329a1ea4c2c74d245e78b9e45c6c` (HTTP).
This is bounded semantic parity with normal completion. Native and HTTP sampling
seeds differ, so it does not establish exact token/logit parity, complete SDK
wire history, cancellation/deadline recovery, cache-hit behavior or all client
features. Both clean-harness reports record a caller-declared engine checkout at
`1783d3bc5ea133995f98e4dd02ad9c35a15ee506`; its current revision does not establish
that every retained assembly was rebuilt from it. These reports supersede the
earlier `*-profile-f6e6e50-dirty` runs for harness qualification, while those
historical reports and their actual binary build identities remain intact.

## Current checkpoint qualification

The `1783d3b` integration registers all 14 typed profile executors; the two bounded
profiles below were rerun using clean harness `d93f4aa`. This is not a
claim that all model/device combinations pass. The completed runs below retain
their original source/binary identities; targeted reruns do not rewrite earlier
full-run counts. Evidence names refer to local ignored report directories under
`D:/TensorSharp-regression-validation/artifacts/`, not published artifacts.

| Completed run | Recorded outcome | Remaining qualification boundary |
|---|---|---|
| `native-cuda-f6e6e50/report.json` | 61 selected CTest tests: 47 pass, 1 fail, 13 skip | `minimax-h3-lora-cuda` fails the f32-base/f32-factors, guard-off reference assertion; maximum difference `0.000628856`. Skips remain unqualified. |
| `native-cpu-5fa7ab0/report.json` | Original 41 selected CTest tests: 37 pass, 2 fail, 2 skip | `cpu-quantized-ffn-strip` still fails strict equality for Q2_K; reported maximum absolute difference `2.38418579e-07`. This assertion failure remains open. |
| `minimax-cpu-5b78c99/report.json` | Targeted MiniMax executor rerun: 1 pass | The allocation-error message assertion was made portable in a test-only change. This resolves that targeted test, not the other CPU failure or the original full report. |
| `native-cuda-profile-d93f4aa/report.json` | Clean-harness rerun: both fixed direct-CUDA sentinels and all 7 profile assertions pass; exit 0 | Positive graph capture/replay, zero CPU fallback and verified teardown; retained `b764dec` test assembly, not a rebuild from the harness commit. |
| `client-live-profile-d93f4aa/report.json` | Clean-harness rerun: two native and four HTTP SDK modes, 12 answers and all 22 profile assertions pass; exit 0 | Qwen3.5-0.8B Q8_0 with Windows `ggml_cuda`; native `f6e6e50` / HTTP `b764dec` deployment identities retained. |
| `grounded-retrieval-profile-d93f4aa/report.json` | Clean harness; profile fail, exit 1 | Qwen3.5-0.8B `ggml_cuda` + MiniLM CPU completed document/source checks, then failed normal completion under the fixed 384-token output budget. Owned Windows Job teardown verified zero remaining processes. |
| `minilm-resumed-run-d93f4aa/report.json` | CPU embedding case skipped, exit 3; no additional transfer | The 2 GiB RAM budget was below the declared estimate of `2178743728` bytes. This is a resource gate, not an inference failure. |
| `minilm-resumed-run-d93f4aa-ram4/report.json` | CPU embedding case and all 19 assertions pass, exit 0 | A valid 4 GiB RAM budget admitted the same workload; a real retained partial download resumed and its final identity verified. Harness code is `d93f4aa`, with documentation-only working-tree edits disclosed. |

The grounded retrieval failure is retained without increasing its output budget
or changing its oracle. The receipt establishes failure to finish normally; it
does not, by itself, classify an engine defect versus model/fixture behavior.
Passing import, indexing and source-byte checks do not make the whole retrieval
profile pass, and later scope-control assertions are not claimed as executed.

The MiniLM cancellation receipt is
`minilm-resume-d93f4aa/controlled-cancellation.json`: it records a real
`2097152`-byte partial, no installed final file, and cancellation in model
preparation. The subsequent accepted run resumed the remaining `22910912` bytes
under a `0.022` GiB network cap, which is below the full file size of `25008064`
bytes. The final SHA-256 was
`263215c3cadd6e16740741a7624ab4cbb6c8e777688bd5331ecfbf5681c2f8ed`.
The download manager was unchanged; both MiniLM reports disclose documentation
edits in the working tree and are not labeled clean-checkout runs. The final
case verified CPU model identity, embedding assertions, typed negative responses
and owned-server teardown. This is one checkpoint's successful resume and CPU
qualification, not evidence that every download or model combination passes.

The original `components-b764dec/report.json` records 110 Knowledge tests passing
and both model-free client contract lanes passing. Its server lane failed/timed
out: 36 failures were traced to model-host environment settings incorrectly
applied to model-free suites. `1783d3b` changes that environment policy, but the
rebuilt suite rerun was pending at this snapshot. A separate FileLogger `Flush`
failure and the overall timeout are not proven fixed. Studio's two missing-ffmpeg
tests now check the skip prerequisite before directory operations; their rebuilt
execution was also pending. Source corrections alone are not passing evidence.

This checkpoint therefore does not establish an all-green full-regression or
merge-ready result. Remaining numeric and grounded-completion failures require
resolution or explicit acceptance of their exact scope; component reruns, the
full Python suite and final CI results were still pending at this snapshot. Linux/WSL GPU,
macOS/Metal/MLX, Vulkan, multi-GPU and authorized remote-node runs remain unrun in
this qualification. The other typed profile classes still require their approved
fixtures, budgets and actual model/device reports.

Exact allowed keys, family/backend admission, fixtures and thresholds are enforced
by the executors' `validate_profile`/`availability` functions:

- [Image and embedding profiles](../eng/regression/image_profiles.py)
- [Video profiles](../eng/regression/video_profiles.py)
- [Scaling, speculation and LoRA profiles](../eng/regression/scaling_profiles.py),
  with the [fixed remote supervisor](../eng/regression/scaling_remote.py)
- [Retrieval and Jev media profiles](../eng/regression/knowledge_profiles.py)
- [Native unit and client profiles](../eng/regression/extended_profiles.py)

The [common schema](../config/regression/scenario-profiles.schema.json) validates
only the envelope; it cannot prove file hashes, catalog membership, source
provenance, resource fit or executor-specific oracle correctness. The primary
model is implicit in `preset`; companion roles use exact catalog IDs. Optional
weights are selected only for the executor's required options. RAM/VRAM budgets
must cover all resident roles/instances, while downloads count unique files once.
Artifact scratch and model-cache demand are charged to their respective volumes;
eligibility and fixture hashes are rechecked after preparation. Per-GPU free VRAM
is checked separately and never pooled. See the deployment guide for the actual
selectors and a runnable synthetic retrieval-profile generator.
Distributed machine pins identify OS instances, not physical separation between
containers or VMs; independent resource allocations and budgets remain explicit
operator requirements. A pin mismatch fails before run-directory creation,
resource inspection, model hashing or probe/anchor launch, rather than relying only
on post-run identity checks.

A selected profile-backed scenario bound to a preset retains a per-preset/backend
`untested` row when that combination lacks a matching selected profile. One model
or device's profile does not cover another combination; unresolved automatic
backend requests remain labeled `auto`. Unbound profile classes use
`catalog::<scenario-id>`, `scope=catalog`, `status=untested` only when no profile
is supplied. Missing binaries, fixtures or budgets cannot turn into passes. General image/video quality,
unimplemented submodes such as LTX Ingredients or owned local ASR, broader agent
workflows, OCR/reranking/tenant ACLs, SDK cancellation and arbitrary size/context/
quantization combinations remain outside the implemented bounded assertions.
Source support, successful preparation and a process exit code alone do not
establish those claims.

## Config repairs and maintenance

- `qwen-image-2.1.json` now uses the portable `../models/qwen-image-2.1` default
  and filenames matching its pinned Q4_K_M/encoder/projector sources. It had
  labeled Q4_K_M bytes as Q8_0 and used a specific Mac directory. Existing model
  files are neither renamed nor deleted; supply the existing model path
  explicitly when retaining an older layout.
- `qwen3.8-27b.json` reuses the agent preset's actual IQ3_XXS identity without
  enabling shell execution, network or package installs.
- `qwen3.5-0.8b.json` adds a small dense representative with a matching optional
  image projector, immutable hashes and exact file sizes for local planning.
  Its CPU default is overridden by the runner's explicit backend selection;
  the Q8_0 checkpoint does not use the NVFP4-only whole-model CUDA executor.
- `qwen3.8-flash-next-mtp.json` is an overlay, requiring the base Flash Next
  config. It must not be treated as a standalone model.
- `qwen3.8-flash-next-iq1_m.json` separately pins the supplied three IQ1_M
  shards. The projector remains optional for text, and no offload or speculative
  decoding policy is silently enabled. Explicit CPU-MoE runs require their
  own resident RAM/VRAM peak requirements; download size is not a fit guarantee.
- LTX README text now reflects its runnable ungated required assets; optional
  gated Ingredients remains a separately reportable blocker.
- [`host-config.schema.json`](../config/regression/host-config.schema.json)
  checks parsed JSONC value shapes. Runtime option names and applicability
  still come from CLI/server source. LoRA plugin documents are not host configs.

When adding an architecture, update its descriptor, catalog family/backend
restrictions, immutable model preset or explicit gap, scenario requirements,
and meaningful assertion. Include all required shards and companions. Never
turn a missing file/device/fixture into a successful result. Keep generated
reports in ignored `artifacts/` or `docs/validation/`; reusable tools and
fixtures belong outside generated evidence directories.

The config/source additions above were checked with source metadata, schemas
and lightweight harness tests. Those checks did not read model payloads or
execute models. Actual model, backend and platform qualification belongs to
the separately recorded execution reports; it cannot be inferred from this
inventory. No remote Mac was accessed during this config audit.
