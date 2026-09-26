# Jev decision inference

[← back to model index](README.md) | [中文](jev_zh-cn.md)

TensorSharp serves a Jev-compatible `POST /v1/systemone` endpoint with the
DiffusionGemma GGUF. A request supplies state and typed questions; the response
contains boolean probabilities (`noul`), categorical choices, and expected scores.
State may be text, images, or both: images travel inline in the request body and
are read as part of the state, so a typed decision can be made about a picture
(see [Image input](#image-input)).
`jev-latest` and `jev-preview` are API aliases for the loaded DiffusionGemma model,
not separate checkpoints or the proprietary hosted Jev model.

The implementation follows the seeded structured-read approach in
[vLLM PR #57250](https://github.com/vllm-project/vllm/pull/57250). It builds an answer
canvas with one token slot per question and reads the requested label logits after
one denoising step. It performs no output sampling, self-conditioning, JSON
generation, or commit forward. This differs from
[LocalJev](https://github.com/githubnext/localjev), which asks a chat endpoint to
generate probability values as JSON and retries malformed output. The
[Jev developer article](https://huggingface.co/blog/sora-2/how-to-use-the-jev-ai-model-a-step-by-step-develop)
describes the typed decision use cases and hosted API; it is not a model-weight or
numerical parity specification.

The adapted compiler prompt/template logic is attributed in
[`TensorSharp.Chat/Jev/NOTICE.md`](../../TensorSharp.Chat/Jev/NOTICE.md), with the
Apache-2.0 license included in build, publish and package outputs. Reference
revisions reviewed: LocalJev `3f23e36e1a3bff46c7e83e8e3781d3512bc82021`;
vLLM checkout `0eb42dbbfce96477f4ca174980f4d68336fb1971`, including the
[`structured_server.py` prototype at commit
`1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8`](https://github.com/vllm-project/vllm/blob/1b3b88ec2b7457aa030db4d0e7d8aaf04f6d0fb8/examples/features/structured_diffusion/structured_server.py).

## Start the server

The supplied Q4_K_M checkpoint needs no tokenizer files. Image input needs the
vision tower, which the configuration downloads once (2.8 GB); text-only decisions
do not. `--mmproj none` on the command line makes the server run text-only and
skips the download too, because a command-line `--mmproj` drops the
configuration's entry before it is resolved. From the repository root, in
PowerShell:

```powershell
$env:TENSORSHARP_MODELS = 'C:/Works/models'
$env:DIFFUSION_VRAM_HEADROOM_MB = '4096'
$env:MAX_CONTEXT = '4096'
dotnet run --project TensorSharp.Server.Host -c Release -- --config config/jev-diffusiongemma-q4.json
```

The [configuration](../../config/jev-diffusiongemma-q4.json) binds loopback port
5000 and uses `ggml_cuda`. Override `--backend ggml_cpu` for CPU execution or
`--backend ggml_metal` on a supported Mac. These are execution options, not claims
that every backend has been benchmarked. The configuration downloads
`diffusiongemma-26B-A4B-it-Q4_K_M.gguf` from
`unsloth/diffusiongemma-26B-A4B-it-GGUF` on Hugging Face when the local file is
missing and reuses it on subsequent launches. It defaults to the repository's
`models` directory; set `TENSORSHARP_MODELS` to an absolute directory to use a
different location. Ordinary chat endpoints remain available on the same server.

Every published GGUF of this checkpoint is text-only and no mmproj was released
for it, so the configuration loads the vision tower straight from the upstream
`model-00011-of-00011.safetensors` shard, which carries all 356 of its tensors.
It is fetched once and reused. Without it the server still answers text-only
requests and refuses image ones with HTTP 503 rather than answering from filler
rows. The same tower serves ordinary DiffusionGemma chat on this server.

The recipe reserves 4 GiB of VRAM for activations on a 16 GiB CUDA GPU. A larger
reserve leaves fewer weights resident but can avoid severe paging on longer
prompts. Tune `DIFFUSION_VRAM_HEADROOM_MB` for the device and workload; its model
default is 2048 MiB. The 4096-token admission ceiling is not a guarantee that
every schema and prompt of that size fits in device memory.

Set `MAX_CONTEXT` before starting the process to limit the combined tokenized
prompt and answer canvas. Without this
environment variable, the model's GGUF context limit applies. This setting is
environment-only; `max-tokens: 256` in the config caps ordinary chat generation,
not Jev input length or canvas width. Choose the context limit for the available
memory: a checkpoint's declared context does not guarantee that a particular
device can execute it. Oversized Jev requests are rejected, not truncated.

```powershell
Invoke-RestMethod http://127.0.0.1:5000/v1/systemone -Method Post `
  -ContentType 'application/json' -InFile docs/examples/jev-ticket.json |
  ConvertTo-Json -Depth 12
```

Equivalent curl request from the repository root:

```bash
curl http://127.0.0.1:5000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/jev-ticket.json
```

If your terminal is already in `docs/examples`, use `--data-binary @jev-ticket.json`
instead. The `@` tells curl to read the file contents. Without it, curl sends the
literal text `jev-ticket.json`, and the server returns `Request body must be valid
JSON.` before running inference.

## Request examples

The [ticket example](../examples/jev-ticket.json) asks three questions in one
shared canvas forward after prompt prefill: route the ticket to a department,
identify an outage, and score its
severity. A smaller Python example uses only the standard library:

```python
import json
from urllib.request import Request, urlopen

body = {
    "model": "jev-latest",
    "state": "I was charged twice for the same subscription this month.",
    "questions": {
        "billing": {
            "type": "noul",
            "instructions": "Is this a billing issue?"
        }
    },
    "samples": 1,
    "seed": 42
}
request = Request("http://127.0.0.1:5000/v1/systemone",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=180) as response:
    answer = json.load(response)
print(answer["answers"]["billing"]["noul"])
```

For an in-process .NET application, reference `TensorSharp.Chat` and use the
same validated request and execution gate:

```csharp
using System.Text.Json;
using TensorSharp.Server;
using TensorSharp.Server.Jev;

using var service = new ModelService();
// Pass the vision shard as mmProjPath to accept requests that carry images;
// null loads the text-only path, which refuses them.
service.LoadModel("C:/Works/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf",
                  mmProjPath: null, backendStr: "ggml_cuda");
using var json = JsonDocument.Parse(File.ReadAllText("docs/examples/jev-ticket.json"));
object response = await service.JevAsync(JevRequest.Parse(json.RootElement));
Console.WriteLine(JsonSerializer.Serialize(response));
```

Low-level callers can use `DiffusionGemmaModel.ReadStructured(promptTokens,
seedCanvas, positions, tokenIds)`, where positions are zero-based canvas indices
and each `tokenIds` row lists that question's labels. Hold `GpuComputeLock`
through any model access. At that level images are your responsibility: expand the
prompt's placeholders, install the encoded spans with
`DiffusionGemmaModel.SetVisionEmbeddings` before the read, and call
`ClearVisionEmbeddings` when the request ends. The service handles tokenization,
prompt rendering, image spans, context checks, canvas construction, serialization
and lifecycle safety for you.

`state` may be a string, JSON object, or array. Question names are returned
unchanged. Use `criteria` as a mapping from choice keys to descriptions for
`choice`, an ordered list of descriptions for `score`, or optional `true` and
`false` descriptions for `noul`. Scores use zero-based indices; for example,
three levels produce an expected score between 0 and 2.

| Request field | Default | Meaning |
|---|---|---|
| `instructions` | none | Optional request-wide instructions, added to the system text ahead of the questions |
| `images` | `[]` | Up to 8 inline images, each base64 or a `data:` URL |
| `samples` | `"auto"` | `1` to `32` independent seeded reads, or adaptive reads |
| `auto_max` | `4` | Total reads when adaptive uncertainty triggers, at most `32` |
| `auto_threshold` | `0.1` | Conditional-label entropy threshold, in nats |
| `seed` | `42` | Repeatable canvas noise on the same .NET runtime/backend |
| `steps` | `1` | Only one-step structured reads are supported |
| `think` | `0` | Thought generation is not part of this endpoint |
| `chunk_rows` | server canvas limit | Optional maximum canvas width per question chunk, `8` to `4096` |
| `chunk_prompt` | `"own"` | Include each chunk's questions, or `"shared"` to repeat all questions |

Adaptive mode starts with one read and uses `auto_max` total reads if any
question's conditional entropy exceeds the threshold. Fixed reads are averaged
as distributions. Use `samples: 1` for a fixed minimum-work request; the example
sets this explicitly. Multiple reads reuse the same prompt K/V on GGML CUDA and
Metal when prompt caching is enabled. Only one prompt cache and label projection
are retained per model; switching state or schema replaces the cache.

Requests support up to 64 questions and 2 to 26 alternatives per question.
Question IDs must be 1 to 128 characters without colons, control characters or
surrounding whitespace. Choice names and score descriptions can contain multiple
tokens: the compiler maps them to short labels and verifies that each label
occupies one token in the full answer template. Invalid templates and context
overflow return validation errors. Conditional question dependencies, sequential
question chaining, additional denoising steps, thought generation, and audio or
video input are rejected explicitly, each naming its own reason.

The server caps request bodies at 8 MiB, because image bytes travel base64-encoded
inside the body; `TS_JEV_MAX_BODY_MB` (1 to 64) sets another limit at startup.
`TS_JEV_MAX_CANVAS` defaults to 64
tokens (also bounded by the checkpoint's canvas width); schemas exceeding this
are split into chunks. `TS_JEV_MAX_PENDING` defaults to 32 admitted requests,
including the active request. Excess requests receive HTTP 529 with
`Retry-After: 1`. Invalid
requests receive HTTP 422, unknown model names receive HTTP 404, and
unavailable/non-diffusion models receive HTTP 503. Malformed JSON receives HTTP
400, oversized bodies receive HTTP 413, and non-JSON media types receive HTTP
415. Model reload and shutdown wait for active inference before disposing
weights. Cancellation is checked between managed layers and native dispatches;
it cannot interrupt a GPU kernel already executing.

## Image input

A request may carry images in `images`, an ordered array of at most 8 entries.
Each entry is a `data:` URL or bare base64:

```json
{
  "model": "jev-latest",
  "state": "Dashcam frame captured as the vehicle approaches the intersection.",
  "images": ["data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAA..."],
  "questions": {
    "signal": {
      "type": "choice",
      "instructions": "Which lamp of the traffic light is lit?",
      "criteria": { "red": "The top lamp is lit", "yellow": "The middle lamp is lit", "green": "The bottom lamp is lit" }
    },
    "stop": { "type": "noul", "instructions": "Must the vehicle stop before entering the intersection?" }
  },
  "samples": 1,
  "seed": 42
}
```

The [traffic-light example](../examples/jev-traffic-light.json) is that request with
a small synthetic picture already embedded, so it can be posted as-is. Its light is
green while the state text says only that the vehicle is approaching, so the answers
are reachable only by reading the pixels: on this checkpoint the identical request
without the image answers `red` and "must stop" with high confidence.

```bash
curl http://127.0.0.1:5000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/jev-traffic-light.json
```

Attaching your own file is base64 and nothing else:

```python
import base64, json
from pathlib import Path
from urllib.request import Request, urlopen

body = {
    "model": "jev-latest",
    "state": "Photo submitted with an expense report.",
    "images": ["data:image/jpeg;base64," + base64.b64encode(Path("receipt.jpg").read_bytes()).decode()],
    "questions": {
        "legible": {"type": "noul", "instructions": "Is the total amount legible?"},
        "kind": {"type": "choice", "instructions": "What kind of document is this?",
                 "criteria": {"receipt": "A purchase receipt", "invoice": "An invoice", "other": "Anything else"}},
    },
    "samples": 1,
    "seed": 42,
}
request = Request("http://127.0.0.1:5000/v1/systemone",
                  data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
with urlopen(request, timeout=300) as response:
    answer = json.load(response)
print(answer["answers"]["legible"]["noul"], answer["answers"]["kind"]["choice"])
```

The same request works in-process: `JevRequest.Parse` decodes the images and
`ModelService.JevAsync` runs them. The server host points
`ModelService.MediaStorage` at the upload directory it already governs; an
in-process caller that leaves it unset gets a directory under the system
temporary path, named once in the log.

### How images reach the read

Each image renders as one `<|image>` marker on the user turn, ahead of the state
text, and is expanded into `[BOI]` + soft rows + `[EOI]` before the context check
runs. The row count is the encoder's own output for that picture, not a constant:
every picture is resized, up or down, to the largest canvas within the 280-row
budget whose sides are multiples of 48 px, so the count depends on its aspect
ratio, not its size (the 192x384 traffic-light example is upscaled to 528x1104
and produces 253). Budget roughly 282 prompt tokens of
`MAX_CONTEXT` per image, plus one sentence of system text naming the attachments.
The encoded rows are installed on the model immediately before the forward that
prefills that prompt. A schema too wide for the canvas is
split into chunks, and every chunk prompt carries the images again at its own
offsets, so `n` chunks cost `n` prefills over the image rows; repeated reads of one
chunk (fixed `samples` or adaptive extension) reuse its prompt K/V as they do for
text. `diagnostics.images` reports how many images the answered request received,
and `usage.input_tokens` counts the expanded prompt including the soft rows.

Decoded bytes are written content-addressed into the server's upload directory, so
`--upload-max-mb`, `--upload-quota-mb` and `--upload-ttl-hours` govern them, the
same picture resent across requests stores one file, and its embeddings are encoded
once and cached (`TS_MM_EMBEDDING_CACHE_MB`, default 512). Image attention inside
each soft-token span is bidirectional on sliding layers and causal on global ones;
`DIFFUSION_IMAGE_BIDIRECTIONAL=0` makes it causal everywhere.

### What is refused

Pixels must arrive inside the request body. Remote URLs are never fetched and
filesystem paths are never read (HTTP 422), because either would let a client of
the inference server reach its network or its files; the chat endpoints refuse both
for the same reason. Multipart bodies are refused with HTTP 415, audio and video
with HTTP 422 naming the checkpoint's missing towers, and an entry that is not
valid base64, not a recognized image container (PNG, JPEG, GIF, BMP, WebP, TIFF,
HEIC) or larger than 16 MiB decoded with HTTP 422. Bytes that carry a container's
magic but do not decode are refused the same way, with the decoder's reason
(`images: PNG does not start with IHDR`), rather than as a server error. A request with images on a
server started without the vision tower receives HTTP 503, not an answer read from
filler rows. Image bytes count against the 8 MiB body cap; a body over it receives
HTTP 413, and storage limits answer with the status the upload policy declares
(413 over the per-file cap, 507 over the quota).

Video frames can only be sent as individual images: upstream's video feature path
for this checkpoint raises `NotImplementedError` and the vocabulary carries no
video begin/end pair. Audio is not supported at all — the checkpoint has no audio
weights, so the `<|audio>` ids its tokenizer inherits have nothing behind them.

## Probability semantics

For each question, the model computes logits `z` for its allowed labels and
returns `softmax(z)` at temperature 1 after the model's final logit softcap.
These are probabilities conditional on the listed answers, and — when the request
carries images — on the encoded image rows in the same prompt. A `noul` value is
the probability of the true label. A choice is the highest-probability option;
a score is the probability-weighted mean of its level indices.
Choice and score `confidence` is the largest conditional probability, matching
the vLLM prototype. LocalJev instead uses one minus normalized entropy, so its
confidence values should not be compared directly. Diagnostics name the
conditional entropy semantics and expose read counts, canvas widths, and
repeated-read agreement/standard error. A single read has no empirical error
estimate.

The adaptive entropy rule differs from the vLLM prototype's entropy over its
reported vocabulary entries. TensorSharp does not invent full-vocabulary mass
or argmax diagnostics from the smaller conditional distribution. The seed
controls reproducibility within TensorSharp; it does not promise Python/NumPy
and .NET generate identical noise for the same numeric seed.

The sparse output projection computes only the answer positions and requested
vocabulary rows. Its conditional probabilities are mathematically equivalent
to selecting the same labels from the full vocabulary distribution and
renormalizing. Small floating-point differences can result from different
matrix multiplication shapes. It does not compute the total probability mass
assigned to all allowed labels, or full-vocabulary entropy.

Quantization, question wording, answer ordering, and the number of reads can
affect results. Conditional confidence is not an empirically calibrated
probability of correctness. Evaluate on held-out examples from the intended
workload before choosing confidence thresholds.

## Execution and validation

The answer canvas is sized to the schema, rounded to a 16-token boundary within
the model's maximum canvas width. Multiple questions share the transformer
forward. The output head operates on requested label rows instead of allocating
the full canvas-by-vocabulary logits tensor. `ggml_cuda` and `ggml_metal` use the
existing DiffusionGemma prompt K/V and fused decode paths; other backends,
`ggml_vulkan`, `mlx` and `cuda` included, use the unified prompt-plus-canvas
forward. The model execution lock serializes access to shared
GPU state with ordinary diffusion chat requests.

GGML CUDA uses fused prompt attention by default, keeping attention operations
inside one native graph to reduce intermediate transfers and explicit KV-head
expansion. The DiffusionGemma path preserves exact sequence extents and the
existing matrix-multiplication precision policy. It uses materialized attention
scores, with no flash attention or persistent attention-graph cache, so prompt
memory still grows quadratically. Set `DIFFUSION_FUSED_PREFILL_ATTN=0` before
launch to restore the per-operation reference path. Other GGML GPU backends can
opt in with `1`; this does not imply that they have been validated.

Reusable validation tools live in `eng/`; generated evidence belongs in ignored
`artifacts/jev/`. See the benchmark tool's `--help` for comparing independently
running `/v1/systemone` endpoints with identical requests. A small smoke set
checks integration and obvious prompt conditioning; it does not establish
production calibration or broad model quality.

The image path is covered by protocol tests that need no model — what an `images`
entry may be, what it is refused for, content-addressed storage, and the rule that
each chunk prompt reads its own installed spans — plus the `--image` case of the
extended smoke, which requires a running server with the vision tower and screens
one unambiguous synthetic picture. That screen proves the pixels reached the read;
it is not an image-understanding benchmark and says nothing about calibration on
visual decisions. Record an independent evaluation before setting confidence
thresholds on image input.

```powershell
# Pure protocol/math tests (no model or GPU required).
dotnet test InferenceWeb.Tests -c Release --filter 'FullyQualifiedName~Jev&Requires!=Models&Requires!=Cuda&Requires!=Mlx'

# Python harness checks use mock responses, without model inference.
python eng/tests/jev-benchmark-tests.py

# Real GGUF sparse/full projection comparison, including deterministic reuse.
$env:TS_TEST_MODEL_DIR = 'C:/Works/models'
$env:TS_TEST_BACKEND = 'ggmlcuda'
$env:TS_TEST_GGML_BACKEND = 'cuda'
dotnet test InferenceWeb.Tests -c Release --filter 'FullyQualifiedName~JevStructuredReadTests'

# End-to-end quality smoke, HTTP contract and concurrent isolation.
python eng/jev-benchmark.py --endpoint tensorsharp=http://127.0.0.1:5000 --samples 1 --concurrency 1,2 --out artifacts/jev/http

# Adaptive/multiple reads, chunking, and concurrent ordinary-chat isolation.
python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --chat --chat-seed 0 --chat-prompt 'What is the capital of France? Answer in one short sentence.' --chat-expected Paris --output artifacts/jev/extended

# Image input, against a server started WITH the vision tower. --image posts the
# traffic-light example and requires its two unambiguous answers; the refusal of
# non-inline image input is checked with or without the flag.
python eng/jev-extended-smoke.py --endpoint http://127.0.0.1:5000 --image --output artifacts/jev/image

# Stop the server first: this probe loads its own copy of the weights.
dotnet run --project eng/JevProbe -c Release -- --model C:/Works/models/diffusiongemma-26B-A4B-it-Q4_K_M.gguf --backend ggmlcuda --iterations 5 --warmup 1 --widths 16,64,256 --output artifacts/jev/projection.json
```

The benchmark's original labeled examples are checked in under
[`TensorSharp.TestMatrix/Inputs/jev`](../../TensorSharp.TestMatrix/Inputs/jev/decisions.json).
Record an independent evaluation for a meaningful quality/calibration claim.
Latency percentiles include successful requests only; failures are reported
separately and count as incorrect decisions. Add `--max-p95-ms` and/or
`--min-requests-per-second` to enforce explicit budgets for every measured group.
Without these options, a smoke-test pass does not constitute a performance pass.
Use `--background-words 0,512` to include a longer-context group, and
`--state-bust` to change state on repeated measurements. Keep these settings,
context limits and warmup counts identical across compared endpoints.

To compare the unchanged LocalJev engine without launching a second server,
install Bun, check out the pinned LocalJev revision, and run the foreground
comparator against the running TensorSharp server:

```powershell
git clone https://github.com/githubnext/localjev artifacts/jev-reference/localjev
git -C artifacts/jev-reference/localjev checkout 3f23e36e1a3bff46c7e83e8e3781d3512bc82021
bun eng/jev-localjev-compare.ts --url http://127.0.0.1:5000 --model diffusiongemma-26B-A4B-it-Q4_K_M.gguf --localjev artifacts/jev-reference/localjev --limit 3 --repeats 1 --server-max-tokens 256 --out artifacts/jev/localjev-comparison
```

This imports the original `Engine` and includes prompt construction, inference,
JSON validation, and retries in its timing. It omits LocalJev's additional HTTP
bridge hop. Both paths receive identical state/questions but use different
internal prompts. Record the server's actual token limit; the supplied config
caps normal chat at 256 tokens. The report preserves upstream finish reasons,
retries and usage so a truncated or failed JSON response is visible.

Comparisons must report hardware, weights and quantization, input lengths,
read count, concurrency, warmup, failures, and latency distribution. LocalJev
over TensorSharp's chat endpoint measures the cost of the generated-JSON
approach on this backend; it is not a comparison against oMLX. The vLLM
prototype's published DGX Spark results and LocalJev's M5 Max results cannot be
used as same-hardware speed comparisons for a Windows RTX GPU.
