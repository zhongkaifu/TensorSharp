# Jev decision inference

[← back to model index](README.md) | [中文](jev_zh-cn.md)

TensorSharp serves a Jev-compatible `POST /v1/systemone` endpoint with the
DiffusionGemma GGUF. A request supplies state and typed questions; the response
contains boolean probabilities (`noul`), categorical choices, and expected scores.
State may combine text with images, uploaded files, documents, sampled video
frames, and audio transcripts. Attachments can travel inline or reference a file
returned by `/api/upload` (see [Files, documents, video and audio](#files-documents-video-and-audio)).
Images use the vision tower; speech uses an explicitly configured transcription
service, because DiffusionGemma has no audio tower.
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

The supplied Q4_K_M checkpoint needs no tokenizer files. Images, scanned documents
and video frames need the vision tower, which the configuration downloads once
(2.8 GB); text-only decisions
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
requests and refuses requests needing image rows with HTTP 503 rather than
answering from filler rows. The same tower serves ordinary DiffusionGemma chat on
this server.

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
| `images` | `[]` | Existing image input: base64 strings or `data:` URLs |
| `files` | `[]` | Attachments classified by filename extension; inline data or upload references |
| `documents` | `[]` | Text/code, PDF, DOCX, XLSX or PPTX attachments |
| `videos` | `[]` | Videos converted to a bounded set of image frames |
| `audios` | `[]` | Speech transcribed by the configured companion service |
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
question chaining, additional denoising steps, and thought generation are rejected
explicitly, each naming its own reason.

The server caps Jev request bodies at 8 MiB, including inline base64 data;
`TS_JEV_MAX_BODY_MB` (1 to 64) sets another limit at startup.
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

To attach your own image using the existing inline image field:

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
text. `diagnostics.images` reports the effective number of image spans, including
scanned pages and sampled video frames,
and `usage.input_tokens` counts the expanded prompt including the soft rows.

Decoded bytes are written content-addressed into the server's upload directory, so
`--upload-max-mb`, `--upload-quota-mb` and `--upload-ttl-hours` govern them, the
same picture resent across requests stores one file, and its embeddings are encoded
once and cached (`TS_MM_EMBEDDING_CACHE_MB`, default 512). Image attention inside
each soft-token span is bidirectional on sliding layers and causal on global ones;
`DIFFUSION_IMAGE_BIDIRECTIONAL=0` makes it causal everywhere.

## Files, documents, video and audio

`files`, `documents`, `videos` and `audios` are arrays with the same entry format.
Use `files` for mixed inputs; the other arrays require their named media kind.
Each entry has exactly one source:

```json
{"name": "incident.txt", "data": "VGhlIHNlcnZpY2UgaXMgZG93bi4="}
```

or, after uploading to the same server:

```json
{"file": "SERVER_FILENAME_FROM_UPLOAD.txt", "name": "incident.txt"}
```

`data` accepts bare base64 or a base64 `data:` URL. Inline entries require `name`
with a supported extension. `file` is the **bare filename from the upload
response's `file` property**, not its `url`, a local path or a remote URL. `name`
is optional for upload references and supplies a display name. The server reads
only files in its governed upload directory, rejecting paths, traversal and
symlinks. References remain usable only while the underlying upload exists;
the storage TTL and quota apply.

The [document request](../examples/jev-document.json) embeds a short incident
report and can be posted directly:

```bash
curl http://127.0.0.1:5000/v1/systemone \
  -H 'Content-Type: application/json' \
  --data-binary @docs/examples/jev-document.json
```

The standard-library [attachment client](../examples/jev-attachments.py) reads
local files on the client and either embeds them or uploads them first. It prints
the complete decision response, including preprocessing diagnostics:

```bash
# Inline a text document.
python docs/examples/jev-attachments.py docs/examples/jev-incident.txt --field documents

# Upload first; only the returned filename is sent in the Jev request.
python docs/examples/jev-attachments.py report.pdf --upload --field documents
python docs/examples/jev-attachments.py crossing.mp4 --upload --field videos --question "Is a green traffic light visible?"
python docs/examples/jev-attachments.py incident.wav --upload --field audios --question "Does the speaker report an active service outage?"
```

For a manual upload, use `curl -F "file=@report.pdf" http://127.0.0.1:5000/api/upload`,
then put the returned `file` value in the JSON
request. `/api/upload` accepts multipart; `/v1/systemone` continues to accept JSON
only. Uploading avoids base64 overhead and the Jev body limit, but does not bypass
attachment, context or storage limits.

### What the model receives

| Input | Processing and limits |
|---|---|
| Plain-text/code files | UTF-8 text, including CSV, JSON, Markdown and source code, is inserted into the state with its filename. No file tools are executed. |
| PDF | Text layer only, or the largest embedded image per page for wholly image-only documents, with the vision tower. At most 32 pages; failed page extraction is rejected. This is not a general PDF renderer or OCR engine: figures in text PDFs, vector graphics and composited scan layouts are not rendered. Mixed text/scanned PDFs may need to be supplied as page images. |
| DOCX / XLSX / PPTX | Text from document paragraphs, spreadsheet cells or slide text. XLSX uses stored values; it does not recalculate formulas. Formatting, charts and embedded pictures are not rendered. Legacy `.doc`, `.xls` and `.ppt` files are unsupported. Office archives are limited to 2,048 entries and 8 MiB expanded data. |
| Video | MP4, MOV, AVI, MKV or WebM, at most 600 seconds and 16 megapixels per frame; sample at 1 fps and uniformly downselect to at most 4 frames per video within the remaining image budget. Frames enter through the image tower with approximate timestamps. The soundtrack is not transcribed. This cannot guarantee detection of brief events or continuous motion. |
| Audio | MP3, WAV, OGG, FLAC or M4A sent to the configured ASR service; its speech transcript is inserted into the state. Sounds, music, speaker identity, timing and vocal emotion are not direct model inputs. Actual decoding/language support depends on the transcription service. |
| Image via `files` | Same vision path as `images`; supported upload image extensions are classified by the file name and checked when decoded. |

There may be at most 8 attachments across the four new arrays, at most 32 MiB per
attachment, and at most 64 MiB in total decoded/referenced media, including legacy
inline images. Images in either `images` or `files` must also meet the 16 MiB
decoded image limit. At most 8 image inputs may reach the model in total, including explicit
images, scanned pages and extracted video frames. Document/transcript text is
limited to 32,768 characters per attachment and 65,536 across attachments.
Byte, document text and PDF page limits reject excess data; they do not silently
truncate it. Videos are deliberately sampled within the remaining image budget.
The final expanded prompt plus answer canvas must also fit `MAX_CONTEXT`.
Video frames and scanned PDF images are limited to 16,777,216 pixels each before
decoding.
Video decoding requires the platform's media provider and a supported codec;
an accepted container extension does not guarantee that every codec inside it
can be decoded.

The server's `--upload-max-mb`, `--upload-quota-mb` and `--upload-ttl-hours` govern
stored attachments and derived images. Inline content is stored by content hash;
repeating the same content reuses its stored file. Prepared attachments and image
embeddings are cached within bounded budgets. `diagnostics.attachments` records
what was actually prepared: `name`, `kind`, `textCharacters`, `imageCount`,
`sampled`, `cacheHit` and `warning`. `cacheHit` describes non-audio extraction;
it does not expose the companion transcriber's separate cache. Inspect warnings for transformations that
limit fidelity. `diagnostics.timing.preprocessing_ms` is separate from
`inference_ms`; `total_ms` is their sum and excludes queuing and the prior upload.

### Configure audio transcription

DiffusionGemma has no audio weights. Configure an operator-controlled HTTP speech
recognizer before starting TensorSharp. A local companion keeps audio on the
machine when that service itself runs locally:

```powershell
# Example: an already running whisper-server service.
$env:TS_JEV_TRANSCRIPTION_URL = 'http://127.0.0.1:8178/inference'
$env:TS_JEV_TRANSCRIPTION_TIMEOUT_SECONDS = '120'
dotnet run --project TensorSharp.Server.Host -c Release -- --config config/jev-diffusiongemma-q4.json
```

The endpoint must accept multipart fields `file` and `response_format=json`, and
return a JSON object of at most 1 MiB containing a nonempty `text` string. For an OpenAI-compatible
speech server, set the full URL ending in `/v1/audio/transcriptions`.
`TS_JEV_TRANSCRIPTION_MODEL` adds the optional `model` field;
`TS_JEV_TRANSCRIPTION_API_KEY` adds an optional bearer token. The timeout defaults
to 120 seconds and accepts 1–600 seconds. Only the operator chooses this URL;
requests cannot choose where audio is sent. An in-process application can supply
`ModelService.JevAudioTranscriber` instead. Duplicate audio can reuse the bounded
transcript cache (16 entries per service, keyed by content hash and extension;
failed transcriptions are not cached).

Without a configured transcriber, audio requests receive HTTP 503, as do service
failures or timeouts. Empty speech receives HTTP 422. Neither failure falls back
to text-only guesses. ASR accuracy is part of the
end-to-end decision quality: validate transcription and decisions in the intended
languages and acoustic conditions. A transcript does not establish native audio
understanding.

### Validation failures

Remote URLs and arbitrary filesystem paths are never fetched or read. Unsupported
extensions, invalid base64, malformed documents/media, or exceeded attachment
limits receive HTTP 422. Missing model capabilities, such as a required vision
tower or audio transcriber, receive HTTP 503. Invalid legacy images also receive
HTTP 422 with the decoder's reason. Non-JSON Jev requests receive HTTP 415 and
requests above the body limit receive HTTP 413. Storage policy failures retain
their declared status (413 for the per-file limit, 507 for the quota).

## Probability semantics

For each question, the model computes logits `z` for its allowed labels and
returns `softmax(z)` at temperature 1 after the model's final logit softcap.
These are probabilities conditional on the listed answers and the prepared state:
text, extracted document content, transcripts and any encoded image rows. Text
extraction, video sampling and transcription can discard evidence before the
model reads it. A `noul` value is
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

The attachment suite uses original fixtures in
[`InferenceWeb.Tests/Fixtures/JevAttachments`](../../InferenceWeb.Tests/Fixtures/JevAttachments/manifest.json)
for text/code, PDF, DOCX, XLSX, PPTX, images, video, audio and mixed evidence. It exercises
inline data and the `/api/upload` → file-reference flow; `--modes
inline,data-url,upload` includes all three transports. Fixture hashes and
provenance are recorded in the manifest. The video repeats one still and the audio
is synthetic speech: these check ingestion, not temporal understanding or ASR
word-error rate. Contrasting documents, concurrent reuse of upload references and
a subsequent text-only request check evidence isolation.

The benchmark reports upload, client request/end-to-end, server preprocessing and
inference times separately. Its server must have the vision tower and an actual
transcription service for full coverage. Unavailable services and skipped cases
do not count as passing coverage. `--cases txt,pdf,docx` can isolate document
work, but does not validate all modalities. Set an explicit `--max-p95-ms` budget
for the hardware and workload before claiming a performance pass. Warm repeated
fixtures can hit extraction, transcript and embedding caches; measure fresh,
representative content separately before drawing production throughput claims.

```powershell
# Pure protocol/math tests (no model or GPU required).
dotnet test InferenceWeb.Tests -c Release --filter 'FullyQualifiedName~Jev&Requires!=Models&Requires!=Cuda&Requires!=Mlx'

# Python harness checks use mock responses, without model inference.
python eng/tests/jev-benchmark-tests.py
python eng/tests/jev-attachments-benchmark-tests.py

# Full attachment quality/transport screen: server needs vision AND real ASR.
# Add --max-p95-ms with the latency budget chosen for your deployment.
python eng/jev-attachments-benchmark.py --endpoint http://127.0.0.1:5000 --modes inline,data-url,upload --repeats 3 --concurrency 1,2 --description 'Record hardware, model, quantization, backend and ASR service' --output artifacts/jev/attachments

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
