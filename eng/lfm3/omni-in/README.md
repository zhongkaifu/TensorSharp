# LFM3-Omni media input: end-to-end tools

TensorSharp's LFM3-Omni image, video and audio input against llama.cpp on the same GGUF and mmproj, through the
servers' OpenAI endpoints with identical request bodies. Outputs belong in the ignored `artifacts/lfm3/omni-in/`.
Build TensorSharp first (`TENSORSHARP_GGML_NO_UPDATE=1 dotnet build TensorSharp.Server.Host -c Release`); the
llama.cpp side is a Metal build of `llama-server` / `llama-mtmd-cli` (`LLAMA`, default `~/work/llama.cpp-lfm3`).

- `make-media.py --out DIR --llama LLAMA [--signal MP4] [--ticket WAV]` builds the media set and its `manifest.json`
  (macOS: `sips`, `say`, ffmpeg, PIL): three animal photos and a newspaper page (llama.cpp's `test-1.jpeg`), a
  flowchart and a receipt drawn with PIL, a 3840 x 2160 photo (2048 rows), speech with known text (the IEEE Harvard
  sentences in four voices, a short story, `ticket.wav`, a 47 s concatenation), llama.cpp's `test-2.mp3` and
  `test-3.mp4` (a clip with a soundtrack), a narrated slideshow (`recipe_narrated.mp4`: steps on slides, the oven
  temperature only in the speech) and a short clip without sound.
- `run-e2e.sh BACKEND OUT [args]` starts `llama-server` (Metal, `-np $LLAMA_NP`) and `TensorSharp.Server.Host`
  (`--backend BACKEND --no-skills --no-multi-agent`) on the same files, runs `$TOOL.py` (default `compare`) and stops
  both.
  - `compare.py` sends 22 greedy chat requests (7 image, 3 video, 8 ASR, 2 audio questions, 2 image multi-turn) to
    both, media parts first (TensorSharp renders a message's media before its text), and writes `results.json` /
    `results.md`: both replies, both prompt token counts, the token-identical prefix (both replies tokenized by
    llama-server), and for ASR the word error rate against the spoken text and between the engines.
    `--reuse-llama OUT/llama.json` compares another TensorSharp backend with an earlier run's llama.cpp replies;
    `--only PREFIXES` runs some cases.
  - `ttft.py` measures cold time to first token (decode + encode + prefill) per media kind: each repetition sends
    media neither engine has seen (a JPEG comment, a PNG text chunk, one PCM sample's LSB, or container metadata
    changed, the decoded pixels and samples unchanged), llama.cpp with `cache_prompt: false`, the two servers
    alternating A/B then B/A. `--concurrent [--concurrent-llama]` instead measures a short text request's TTFT alone
    and while a media request is being prepared (sent `--delay` seconds after it); give llama-server two slots
    (`LLAMA_NP=2`) for its side.
  - `surfaces.py` (TensorSharp only) sends the same image, audio and video prompts through `/v1/chat/completions`,
    `/v1/responses` (`input_image`, `input_audio`, `input_video`) and `/api/upload` + `/api/chat`
    (`imagePaths`, `audioPaths`, `videoFilePaths`), and checks a follow-up turn in one `/api/chat` session reuses the
    cached prompt through the image (`kvReusedTokens`) while another image does not.
- `OmniInProbe` (`dotnet build eng/lfm3/omni-in/OmniInProbe -c Release -o OUT`; copy the GGML / MLX native libraries
  beside it):
  - `plan VIDEO`: the clip as the model reads it (probe, 2 fps plan, labels, soundtrack span and its split);
  - `layout MODEL MMPROJ BACKEND SYSTEM TEXT KIND=PATH...`: one user turn rendered and expanded (prompt tokens, every
    media span and its content id, preparation time), then the prompt's prefill with its media rows against a text
    prompt of the same length;
  - `split MODEL MMPROJ BACKEND IMAGE`: an image's encode as one program and split for GPU-lock yields (2 and 4
    programs), interleaved, with the rows' distance;
  - `stages MODEL MMPROJ BACKEND VIDEO`: a video's preparation stage by stage (the host frame sampling a frame family
    would pay, decode + resize, soundtrack decode, features, pair encodes, soundtrack encode).

The model-level tests are `InferenceWeb.Tests/Lfm3OmniInputTests.cs` (no weights) and `Lfm3OmniModelTests.cs`
(`TS_TEST_LFM3_MODEL` = a directory with LFM3-Omni-2B and its mmproj; lanes by `TS_TEST_GGML_BACKEND`, MLX, managed CPU).
