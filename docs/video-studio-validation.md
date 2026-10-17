# Video Studio validation and capability matrix

This records implementation-branch validation on 2026-10-15, not a production
readiness certification. See [design and reference audit](video-studio-design.md)
for source commits, licensing, architecture and dependency boundaries, and the
[user guide](video-studio.md) for operation.

## Environment and dependency revision

- macOS 27.0, Apple Silicon arm64, 48 GiB unified memory.
- .NET SDK 10.0.302 / runtime 10.0.10; Apple MAUI workload 10.0.20.
- Operator-installed Homebrew FFmpeg/ffprobe 9.0.2. The tested build enables GPL,
  version 3 components and libx264; Studio uses available H.264/AAC encoders.
- Upstream `ExternalProjects/ggml`:
  `a05c9922613c6a2230dbb1d0de679519ecafb4f0`, unchanged working tree. No ggml patch
  or native kernel change was introduced for Studio.
- Implementation baseline: TensorSharp `main`
  `6bdc5c7b799fdc63e061b8dbd3ec9a64bb46cf48`; branch `codex/video-studio`.

Generated logs, screenshots, reports, projects and sample exports are ignored
under `artifacts/studio/`. They are not source-controlled fixtures. Reusable
drivers are in `eng/studio/`; small colors/tones are generated deterministically
with FFmpeg. No model downloads or remote services are part of these drivers.

## Recorded validation

| Check | Actual result and scope |
|---|---|
| Shared core tests | 53 passed, 0 failed, 0 skipped. Real FFmpeg output tests include pixels, caption timing, audio RMS/mixing/fades/dissolve, 30000/1001 timing, true variable-rate source trimming, decode, captured revisions, packages and derivatives. Generation lifecycle tests explicitly use doubles; missing-tool/mobile tests prove inference is never called. Startup cleanup and package disk-reserve rejection are covered. |
| Host regression subset | 187 passed, 0 skipped: Studio tools/routes plus existing video turns, host, upload media and loopback range/auth tests. Additional host/localization subset: 88 passed. Model routing tests use doubles. |
| Full TensorAgent test project | At the full-suite checkpoint: 1702 total, 1588 passed, 2 failed, 112 skipped in about 2 min 8 s. Later loopback security cases passed in the focused suite below. The two failures are existing, unchanged bundled-skill description assertions: ResearchSkillTests requires a description longer than 200 characters; MarketDataSkillTests expects “gainers”. Two Studio integration regressions found by the first run were fixed before this result. The full project is not reported as passing. |
| Shared UI tests | 13 passed using a simulated DOM, including bounded restored messages, command contracts, immediate history restoration and all eight locale catalogs. These are not native-device tests. |
| Final focused host/security suite | 200 passed, 0 skipped, including IPv4/IPv6/mapped-loopback, remote peer, untrusted Host, explicit remote opt-in, image-turn routing and localization checks. |
| Server and TensorAgent builds | Server.Host, TensorAgent.Core and the real TensorAgentHost harness build successfully. |
| Mac Catalyst build | Final `net10.0-maccatalyst`, arm64 Debug: exit 0, 64 warnings, 0 errors; build only, no launched MAUI application claim. Final server build: 5 warnings/0 errors; TensorAgent harness: 11 warnings/0 errors. Existing warnings remain visible in build logs. |
| Running WebUX browser | Created project, imported real files, added clips on two video tracks, edited placement/trim/fade/caption, rendered playable MP4 and reopened revision 8. Browser video reached readyState 4, 320×180, 2 s. |
| Running TensorAgent shared page | Real AgentAppHost loopback page: imported/added/trimmed/undid, pointer-dragged 24 frames and undid, rendered playable MP4, reopened saved state after process restart. Linked saved conversation appeared immediately without another send or revision change. Desktop and 390×844 browser viewport screenshots; viewport simulation is not an iPhone test. |
| HTTP vertical slice, both hosts | Existing chat upload → Studio import → multiple tracks → atomic proposal/stale rejection/undo/redo → captured-revision render → range response → complete probe/decode. Server and Agent projects re-rendered after actual host restarts. |
| Native Server Studio generation | Real LTX-2.5 Distilled Q4_K_M, MLX, seed 42, 512×512, 17 frames at 24 fps, audio: completed, managed asset/provenance saved, inserted alongside imported clips, exported 4.708333 s composition. Admission-to-completion about 125.6 s, including identity hashing and cold setup. |
| Native TensorAgent Studio generation | Same actual model/settings through AgentAppHost: completed and exported 4.708333 s composition; about 114.0 s. External outbound networking denied using macOS sandbox-exec; localhost allowed. Hash/provenance, thumbnails, waveform, package round trip, scoped media and CSRF checks passed. |
| Native cancellation | TensorAgent LTX job cancelled after entering denoise: running → cancelling → cancelled, no new asset, following render completed. This validates this short MLX workload, not every model/backend or a mid-generation resume checkpoint. |
| Real conversational planner, both hosts | Gemma-4-E4B-it-Q8_0 on ggml_metal inspected the selected clip and produced a validated atomic trim-to-24-frames + volume-0.25 proposal. Explicit apply, one-step undo, manual movement and stale proposal rejection passed through both real HTTP hosts. No inference doubles. |
| Final restart/export/security smoke | Both original projects containing real generated footage reopened and exported again after the final VFR/recovery/security build, each producing 113 decoded video frames. Both fresh HTTP vertical slices passed. Actual server spoofed-Host and cross-origin requests to `/api/studio` and Studio-bound `/api/chat` returned 403. |

The two real generated clips were 512×512, 17 decoded frames, 0.708333 s, H.264
with audio, 187858 bytes, SHA-256
`57e6af40bd12f7195acf1bfcc258f732b0d0312340104a4dd260dd2a3bcfc561`.
The model file SHA-256 is
`0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0`.
This is a very short functional sample, not a long-duration quality benchmark,
character-consistency evaluation, throughput comparison or VRAM benchmark.

The first actual planner attempt produced nested operation-name JSON and failed
strict validation without modifying the project. Tool documentation and repair
feedback were clarified with an explicit flat command example. The same natural
instruction then passed on both hosts. This records the observed failure and
correction; it is not a claim that every text model reliably follows the schema.
An ordinary browser request initially asked for a clip ID. A trusted, Studio-only
workflow instruction now directs the existing agent to inspect the selected
project first; the browser request “Lower the selected clip volume to 0.5” then
produced a valid proposal without user-supplied IDs or tool names.

The 112 full-suite skips were explicit: 54 need an unstaged embedded CPython
runtime; 16 need network opt-in (nine also need Python); 33 need text/multimodal
model opt-in; six need image/video weights; three are platform-restricted. These
are not passes. The separately recorded real Studio LTX/Gemma runs do not turn
those other skipped scenarios into successful coverage.

Output checking includes full decoding, expected dimensions, at most one-frame
duration/count tolerance, red/blue pixel discrimination and nonzero PCM audio.
Core tests additionally check caption-on/off pixels and relative audio amplitude.
Rendered preview and downloaded export are the same captured-revision output;
there is no separate browser composition engine to claim parity with.

A true variable-rate fixture (four source frames at 0, 0.4, 1.0 and 2.0 s)
initially exposed an incorrect trim: the held frame covering an in-point could
be discarded. Sampling source presentation timestamps onto the project clock
before integer-frame trimming fixed it. The resulting 48-frame export now checks
red/blue/green boundaries and audio onset within one frame. The metadata VFR flag
is still a rate-comparison heuristic, not exhaustive classification of every
container's timestamp behavior.

### Asset deletion follow-up

`StudioHostTests` now includes the same deletion scenario over the actual
ASP.NET Core WebUX routes and authenticated TensorAgent loopback routes. The
focused run passed **32 tests, 0 failed, 0 skipped**. It verifies the default
`asset_in_use` refusal, explicit removal of affected timeline clips, unchanged
linked clips using other assets, stale revisions, cross-project asset rejection,
the required request header, retry idempotency, undo/redo and retained media bytes.
Deleted assets are unavailable through the current asset URL until undo restores
them. Fixtures use seeded PNG metadata with media processes disabled; this test
does not claim real generation, rendering or native-device coverage.
A broader focused run including `LocalizationTests` and `WebUiRoutesTests`
passed **135 tests, 0 failed, 0 skipped** after the shared UI/catalog changes.

```sh
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter FullyQualifiedName~StudioHostTests --nologo -v:q
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter 'FullyQualifiedName~StudioHostTests|FullyQualifiedName~LocalizationTests|FullyQualifiedName~WebUiRoutesTests' --nologo -v:q
```

Logs are ignored at `artifacts/studio/delete-asset-host-tests.log` and
`artifacts/studio/delete-asset-focused-host-tests.log`. The Server project reference
is test-only; TensorAgent's application dependency graph is unchanged.

The deletion follow-up also passed **67 core tests** (14 new cases) and **24
shared UI tests**, with no skips. Core coverage includes durable targeted asset
undo, legacy history, later imports, atomic rollback, missing media, package
exclusion and captured jobs. A real FFmpeg render completes after its captured
source is removed from the current library; the conditioning-generation lifecycle
test uses an explicit adapter double, not real model inference.

Running browser checks on both hosts confirmed the two-clip usage warning,
deletion, Undo and Redo; WebUX also verified Cancel. After restarting both hosts,
the deleted state persisted, Undo restored the asset and both clips, and each
restored project exported a fully decoded 320×180, 48-frame MP4. Screenshots,
sample exports and the token-free report are in
`artifacts/studio/delete-asset-browser/`. These are desktop browser checks against
the actual hosts, not native-device tests.

Final Server.Host and TensorAgentHost builds passed (5 and 11 warnings,
respectively). The additional Mac Catalyst build was **blocked**: the installed
.NET Mac Catalyst 26.5.10315 workload requires Xcode 26.6, while the selected Xcode
is 27.0. This follow-up does not claim a successful native application build or
launch. No toolchain settings were changed to bypass the compatibility check.

```sh
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj --no-restore --logger 'console;verbosity=normal'
node --test eng/validation/tests/studio-ui.test.mjs
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj --nologo -v:q
dotnet build eng/validation/TensorAgentHost/TensorAgentHost.csproj --nologo -v:q
# Attempted; blocked by the Xcode/workload mismatch described above:
dotnet build TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj -f net10.0-maccatalyst -p:TensorSharpAppleTargets=true --nologo -v:q
```

Additional ignored evidence uses the `artifacts/studio/delete-asset-` prefix:
`core-tests.log`, `ui-tests.log`, `server-build.log`, `agent-build.log`, and
`maccatalyst-build.log`.

### Generation modes follow-up (2026-10-16)

Studio requests now carry an explicit mode: `t2v`, `i2v`, `fl2v`, `keyframes` or
`ref`. Each loaded checkpoint advertises the modes it runs, plus a reason for each
family mode it cannot run. The same environment and unchanged ggml revision as
above were used, built from branch `codex/video-studio` at `eb7d19a8` plus this
change.

| Check | Actual result and scope |
|---|---|
| Shared core tests | 106 passed, 0 failed, 0 skipped. Includes 36 new mode cases: each mode accepts exactly its inputs and refuses the others, legacy requests are inferred and keep their fingerprint, capability mode derivation, plan validation, and admission refusing an unsupported mode or a short reference clip before the adapter runs. Admission uses real FFmpeg/ffprobe and an adapter double. |
| Model contract tests | 36 Studio/LTX capability cases passed. They cover modes and refusal reasons per checkpoint: H3 FL2VA/Ref2VA, LTX with/without the image encoder and the Ingredients adapter, and three Wan variants. They also check that each Studio mode reaches `MiniMaxH3ModeResolver` as the same mode chat sends, and reaches LTX as the intended frame guides or Ingredients sources. Metadata only; no weights load. The wider `Studio`/`Ltx25`/`MiniMaxH3`/`VideoGeneration` filter passed 578, failed 0 and skipped 80. The skips are existing fixture- or CUDA-gated tests (H3 oracle fixtures kept outside the repository, LTX CUDA kernels), not passes. |
| Shared UI tests | 35 passed with a simulated DOM. They check that the mode list follows the checkpoint, disabled modes show their reason, each mode submits only its own inputs, choices persist across mode switches, required images are enforced, reference order is preserved, ref raises frames to 121, and old takes reopen in their inferred mode. They are not a browser or device test. |
| TensorAgent focused suite | 176 passed, 0 skipped: Studio host routes, localization (all eight catalogs carry the new keys and placeholders), video turns, routes and LTX page tests. |

Real generation through Studio's HTTP API, using `eng/studio/validate_modes.py`
with synthetic stills (`testsrc2` first, `smptebars` last, `rgbtestsrc` guide,
`testsrc` reference), seed 42 and audio off. Each take was fully decoded at the
requested geometry, and its provenance recorded the requested mode. "Distance" is
the mean absolute difference of 8×8 RGB thumbnails (0–255); lower means closer.

| Host / checkpoint | Mode | Take | Result |
|---|---|---|---|
| WebUX Server, LTX-2.5 Distilled Q4_K_M, `mlx` | t2v | 512×512×17, 44 s | First frame unrelated to the stills (107 / 79) |
| | i2v | 512×512×17, 79 s | First frame 42.6 from the first image vs 102.8 from the last |
| | fl2v | 512×512×17, 86 s | First 42.6 vs 102.9; last frame 30.1 from the last image vs 112.6 |
| | keyframes | 512×512×17, guide at frame 8, 83 s | Frame 8 is 4.6 from its guide vs 98.8 |
| | ref | 512×512×121, one full-size Ingredients pass, 507 s | The server log shows a 1-image Ingredients reference sheet; the first frame is a new scene, not the reference |
| WebUX Server, MiniMax-H3 FL2VA Q4_K, `ggml_metal`, 20 steps | ref refused | — | 400 `unsupported_mode` with the advertised Ref2VA reason; no job created |
| | i2v | 640×384×22, 195 s (includes first checkpoint hash) | First 42.4 vs 102.2 |
| | fl2v | 640×384×22, 235 s | First 42.2 vs 102.2; last 29.4 vs 111.2 |
| | t2v | 640×384×22, 141 s | Unrelated to the stills |
| TensorAgent `AgentAppHost` harness, MiniMax-H3 Ref2VA Q4_K, `ggml_metal`, 20 steps | i2v, fl2v refused | — | 400 `unsupported_mode` with the advertised FL2VA reason; no job created |
| | ref | 640×384×22, 194 s | The model log shows 1 reference becoming 240 visual tokens. The reference test card appears printed on a waving flag in a new scene, as the prompt asked; it is not the first frame |
| | t2v | 640×384×22, 142 s | Unrelated to the stills |

Model SHA-256 values, as recorded in each take's provenance:
LTX `0f51eb0d82b19bddbfb3b0371a65217844ea03750f27dd733528f22152e0e0d0`, H3 FL2VA
`dd948e08ad0ba3c71bd42f368e283dd82e790f5122a63b276e22a3e0283d0c10`, H3 Ref2VA
`2fa5840021cf6967843eaeefde9aaa277e540de02986d5ee3d5b0e6a7a8c9dec`.

Limits of this evidence:
- Pinned-frame checks are relative and use synthetic patterns. They show where a
  frame lands, not perceptual or anatomical quality.
- LTX ref used a test card, which cannot show identity transfer. The H3 ref take
  shows appearance transfer for one synthetic image. Neither is a
  character-consistency evaluation.
- LTX ran at 512×512, which the user guide warns is weaker for faces and hands
  than the 1024×1024 preset.
- Wan has no local checkpoint here, so its modes have contract tests only.
- The UI was exercised through simulated-DOM tests, not a browser or device, in
  this follow-up.
- Each mode ran once with one seed.

Evidence (ignored): `artifacts/studio-modes/{ltx,h3-fl2va,h3-ref2va-agent}/`
(stills, takes, `modes.json`), the server and harness logs beside them, and
`artifacts/studio-modes/tests/`.

```sh
# One video model loaded in either host; repeat per checkpoint.
python3 eng/studio/validate_modes.py --url http://127.0.0.1:8187 \
  --width 512 --height 512 --frames 17 --output artifacts/studio-modes/ltx
python3 eng/studio/validate_modes.py --connection artifacts/studio-modes/agent-root/connection.json \
  --modes ref,t2v --output artifacts/studio-modes/h3-ref2va-agent
```

### Sampling-recipe follow-up for MiniMax-H3 Turbo (2026-10-17)

Studio now binds each job to the loaded model's sampling recipe, so a MiniMax-H3
FL2VA model with a LightX2V Turbo plug-in fixes the step count and refuses a shot
planned for another recipe with 409 `sampling_recipe_conflict` (see
[sampling recipes](video-studio.md#sampling-recipes)). The environment and the
unchanged ggml revision are those above; the build is branch `codex/video-studio` at
`eb7d19a8` plus the uncommitted Turbo change. Only automated tests ran.

| Check | Actual result and scope |
|---|---|
| Shared core tests | `TensorSharp.Studio.Tests`: 123 passed, 0 failed, 0 skipped, with real FFmpeg/ffprobe. The 10 new recipe cases cover the fixed-step default and the binding after the fingerprint, an idempotent resubmit, native binding, the named fixed-steps refusal (a Turbo recipe and a step-distilled checkpoint), the 409 with no job created (3 cases), legacy serialization and the draft path. |
| Adapter and host tests | `InferenceWeb.Tests` on the CPU lane, 15 classes: 626 passed, 0 failed, 0 skipped. The 16 new Studio Turbo cases cover the capability with and without a plug-in, the 3-argument `CreateCapability` still giving `native`, a contradicting `--video-steps` or a Ref2VA plug-in leaving the model unavailable, validation against the plug-in's recipe, and the provenance helpers. They use synthetic plug-in configs and CPU direct LoRA sets; no model weights load. |
| TensorAgent focused suite | 203 passed, 0 skipped: localization (all eight catalogs carry the two new keys), Studio host routes, `AgentAppHost` and video turns. A later run of 11 classes after the TensorAgent change passed 664 with 0 skipped, including Studio's plug-in provider: capability queries do not log, and each clip logs once. |
| Shared UI tests | `studio-ui.test.mjs`: 52 passed with a simulated DOM, 4 of them new (the recipe hint, the reopen note after a recipe change, steps from the model default, and a request that carries the model's recipe rather than the take's). Not a browser or device test. |
| Mutation check | 18 deliberate mutations of the admission code, the adapter and the page each failed at least one test; all were reverted and the binaries rebuilt before the final runs. |

Limits of this evidence:
- No Turbo take was generated through Studio, in either host. The run path was
  exercised only up to the bound-recipe check before generation.
  A real Turbo take through Studio, with its provenance (plug-in identity, recipe,
  sigma grid) checked on the finished clip, has not been run. The Turbo clips measured
  in the [MiniMax-H3 card](models/minimax-h3.md#turbo-validation-status) came from the CLI.
- The recipe hint and the reopen note were not checked in a browser or on a device.
- Not counted: `webui-video-upload.test.mjs` failed 9 of 9 with
  `currentArchitecture is not defined` in `TensorSharp.Server.Host/wwwroot/index.html`,
  a page this change does not touch.

### Job status detail follow-up (2026-10-17)

The Jobs tab now shows each job's stage, denoising steps, elapsed time, time in the
current stage and an estimate of the time left, with a per-stage breakdown once the job
ends (see [portable projects and recovery](video-studio.md#portable-projects-and-recovery)). The build is branch
`codex/video-studio` at `eb7d19a8` plus the uncommitted change. The change is managed
code only; ggml and the native kernels are unchanged.

| Check | Actual result and scope |
|---|---|
| Shared core tests | `TensorSharp.Studio.Tests`: 134 passed, 0 failed, 0 skipped, with real FFmpeg/ffprobe. 9 new estimator cases run on a fixed clock: the denoising step rate, a slow step, the first step alone, before denoising, after it, a previous run of the same settings, queued, ended and cancelling jobs, and a render. 2 new lifecycle cases drive the real worker with a test double: stages, steps and times while running, the breakdown after completion, the next run's whole-job estimate, cancellation, and the stage recorded after an abrupt restart. |
| Adapter and host tests | `InferenceWeb.Tests`: 9,322 passed, 534 skipped, 2 failed. The 2 new adapter cases check that a job counts only finished steps (Wan names the running step), that the model's `done` becomes `encoding`, and that a heartbeat on a timer thread cannot throw. The 2 failures are in `NoAnswerPlaceholderTests`. They read `TensorSharp.Chat/WebUiChatService.cs`, which this change does not touch, and one of their anchors is already missing at `eb7d19a8`. |
| TensorAgent focused suite | 214 passed, 0 skipped: localization (all eight catalogs carry the 51 new keys), Studio host routes, `AgentAppHost`, video turns and LoRA UI. |
| Shared UI tests | `studio-ui.test.mjs`: 55 passed with a simulated DOM, 3 of them new: a running card, the estimate wordings, and queued, completed, failed and render cards with the tab count. |
| Real generation, Server | Apple M5 Pro, macOS 27.0.1, `ggml_metal`, MiniMax-H3 FL2VA Q4_K, t2v 640×384, 22 frames, 4 steps, two identical requests through `/api/studio`, polled once a second. Run 1 (40 s, no history) reported `model-identity` (the checkpoint SHA-256), `text-encode`, `denoise` from the start of the loop, VAE tiles 1/6 to 6/6, `audio-decode`, `encoding` and `importing`. Its denoising estimate read 18 s, 11 s and 5 s after steps 1 to 3; decoding began 5 s after step 3. Run 2 (36 s) had a whole-job estimate from its first second: 15 s left after step 3, finishing 14.7 s later. Headless Chrome screenshots of the Jobs tab matched the cards above. |

Limits of this evidence:
- Only MiniMax-H3 ran for real, and only one short clip. Wan's immediate phase
  reports and LTX's progress went through the adapter tests, not a model run.
- The estimate before the first step uses the previous run's stages. When that run
  included the one-time checkpoint hash, it over-estimates by the hash time (4 s here).
- The page was checked in desktop Chrome only, not in the TensorAgent WebView or on a
  device.

Evidence (ignored): `artifacts/minimax-h3/turbo-wp6/` and
`artifacts/minimax-h3/turbo-wp7/`.

### Optional first/last frames and the Generate redesign (2026-10-17)

First and last frames are now each optional: none is t2v, a first frame alone i2v,
and a last frame, alone or with a first, fl2v (see [generation modes](video-studio.md#generation-modes)).
The Generate tab, library, guide and Assistant were redesigned
([make a take](video-studio.md#make-a-take), [the assistant](video-studio.md#the-assistant)).
The build is branch `codex/video-studio` at `eb7d19a8` plus the uncommitted change. The
change is managed code and the shared page; ggml and the native kernels are unchanged.

| Check | Actual result and scope |
|---|---|
| Reference behaviour | Read, not run, at sglang `ed40c7fc66`, vllm-omni `43e507117`, stable-diffusion.cpp `d637cfb` and MiniMax-H3 `d21241f0` (model card and prompt guide). All take zero, one or two FL2VA keyframes, and none substitutes a blank image for a missing frame. sglang and vllm-omni place a lone last frame at the end of the timeline; MiniMax publishes no last-only quality data. |
| Shared core tests | `TensorSharp.Studio.Tests`: 139 passed, 0 failed, with real FFmpeg/ffprobe. New: a last frame alone resolves to fl2v and is accepted; fl2v with only a first frame, or none, is refused; prompt-named takes (whole words and whole surrogate pairs, numbered after the highest take of the shot); job summaries carry the frame asset ids. |
| Adapter, native and host tests | `InferenceWeb.Tests` Studio/LTX/MiniMax-H3/video/Wan/CLI/Web UI filter: 1,294 passed, 0 failed, 92 skipped. The skips are existing fixture- and CUDA-gated tests, not passes. New: H3 and LTX receive last-only Studio requests exactly as chat sends them; the canvas comes from a lone last frame (a 119×178 still gives 768×1152); LTX last-only is one guide at the final frame and LTX i2v needs a frame-0 image; a lone end keyframe is one H3 condition block at the end time; Wan refuses an end image; a T2V-only Wan no longer advertises image conditioning; H3 advertises its 4–15 s range. |
| TensorAgent suite | 1,623 passed, 112 skipped, 2 failed. The 2 failures (`ResearchSkillTests`, `MarketDataSkillTests`) compare skill descriptions under `TensorAgent/skills`, which this change does not touch. The focused localization/Studio/route/video/LoRA filter: 296 passed. All eight `studio.json` catalogs have the same 458 keys; the seven non-English ones were translated for this change, including about 350 Studio strings per language that were still English. |
| Shared UI tests | `studio-ui.test.mjs`: 77 passed with a simulated DOM, including picker import, Escape layering, keyframe edits surviving a redraw, planned shots, take reopening, model-refusal refresh and the Assistant tab. |
| Code review | A five-dimension review with a verifier per dimension confirmed 45 findings and rejected 12; all 45 were fixed and most have a test above. |
| Real generation, Server, MiniMax-H3 FL2VA Q4_K, `ggml_metal`, 20 steps, seed 42 | Last frame only, 640×384×22: last frame 29.2 from the last image vs 111.4 from the first; the log shows "FirstLastFrame: 1 keyframe(s) -> 240 conditioning tokens". First and last at the same seed: first 42.2 vs 102.2, last 29.4 vs 111.3, 2 keyframes, 480 tokens. Last only, 640×384×107 (4.5 s): last 29.3 vs 111.4. |
| Real generation, Server, LTX-2.5 Distilled Q4_K_M, `mlx` | Last frame only, 512×512×17: last 29.9 vs 112.3; the log shows one temporal guide, "frame 16=1". First and last: first 42.6 vs 102.9, last 30.1 vs 112.6, guides at frames 0 and 16. |
| Real generation, TensorAgent `AgentAppHost` harness, MiniMax-H3 FL2VA | Last frame only through TensorAgent's Studio routes: the same distances as the Server run at the same seed (29.2 vs 111.4), 1 keyframe, 240 tokens. |
| Browser, Server (headless Chrome over CDP) | `eng/studio/validate_ui_browser.mjs` against the real hosts: the guide with no project; import through the page's file input; "As last frame" on a library photo opens Generate with the slot filled and focused; the summary reads "Ends on studio-last.png …"; the picker opens inline and Escape closes only it; the Assistant disables Send while a video model is loaded; at 390×844 the smallest button is 44 px. With `--submit` the form generated a last-only H3 take, named after its prompt and announced as ready. Checked on H3 and on LTX. |
| Browser, Assistant | With Gemma-4-E4B Q8_0 loaded: the request "Trim the selected clip to its first 2 seconds and lower its volume to 25 percent" gave "Trim … to 2 s" and "volume 25%"; Apply made the 4.00 s clip 2.00 s as one undoable edit. A second run summarized "first 2 seconds" while its command trimmed to 1 s, and the change list showed "to 1 s". |
| Browser, TensorAgent page | The harness's own page (TensorAgent `wwwroot`) with Studio in Simplified Chinese: guide, import, "作为尾帧" (As last frame) and Generate render translated. Messages that come from the host (unavailable-mode reasons, refusals) remain English. |

Limits of this evidence:
- The stills are synthetic. With a last photo unlike the described scene (a room
  ending on colour bars), both models reach it by a cut or dissolve, at 0.9 s and at
  4.5 s alike; the form says so. Last-only quality on real photos that the scene can
  plausibly arrive at was not measured, and MiniMax publishes none.
- Each case ran once with one seed. No Wan checkpoint exists here, so Wan's
  refusal of an end image has contract tests only.
- The TensorAgent Mac (Catalyst) and iOS apps were not run: they disable media
  subprocesses, so Studio there reviews and edits projects but cannot import,
  generate or export. The harness is the macOS .NET host.
- Touch, VoiceOver and real devices were not tested. The phone layout was checked in
  an emulated 390×844 viewport.

Evidence (ignored): `artifacts/studio-optional-frames/` (stills, takes, `modes.json`,
`ui.json`, screenshots and the host logs).

```sh
# A video model loaded in either host; fl2v-last is fl2v with an empty first frame.
python3 eng/studio/validate_modes.py --url http://127.0.0.1:8187 --modes fl2v-last,fl2v \
  --width 640 --height 384 --frames 22 --output artifacts/studio-optional-frames/h3-fl2va
python3 eng/studio/validate_modes.py --connection artifacts/studio-optional-frames/agent-root/connection.json \
  --modes fl2v-last --output artifacts/studio-optional-frames/h3-fl2va-agent
# The real page in headless Chrome (no packages): the default flow, a take from the form,
# the Assistant (a tool-capable chat model loaded), and TensorAgent's page.
node eng/studio/validate_ui_browser.mjs --url http://127.0.0.1:8187 --output artifacts/studio-optional-frames/ui
node eng/studio/validate_ui_browser.mjs --url http://127.0.0.1:8187 --submit --size 640x384 --output artifacts/studio-optional-frames/ui-submit
node eng/studio/validate_ui_browser.mjs --url http://127.0.0.1:8187 --assistant --output artifacts/studio-optional-frames/assistant
node eng/studio/validate_ui_browser.mjs --entry "<entryUrl from connection.json>" --screens-only --output artifacts/studio-optional-frames/agent-ui-zh
```

## Feature and platform status

| Capability | Implemented | Validation / limits |
|---|---|---|
| One shared project/command/render format | Both hosts | Core and HTTP tests; one exclusive service writer per storage root. TensorAgent token authentication; Server Studio loopback by default, explicit protected remote deployment opt-in. Not a multi-user authorization database. |
| Library, storyboard, multiple video/audio tracks | Both shared pages | Browser checks, simulated interaction tests, core tests. Generated and imported assets use the same clips. |
| Placement, drag, exact trim, split, duplicate, remove, snapping/zoom/scrub | Shared page/commands | Core and browser subset; not every touch gesture tested on hardware. Source media survives deletion/undo. |
| Delete library assets | Shared page/command in both hosts | Warns about timeline use; confirmed deletion removes that asset's clips atomically. Both HTTP transports verify revision checks, isolation, undo/redo and retained media. Does not reclaim disk space. |
| Linked edits, volume/fades/dissolve, contain/cover, track mute/visibility | Shared commands/render | Automated command/render tests. Dissolve requires an existing overlap and changes no clip positions. Embedded clip audio travels with its video. |
| Captions | Manual timed editable captions | In-house portable uppercase Basic Latin bitmap glyphs; unsupported Unicode refused. No automatic transcription/word-based cutting adapter. |
| Preview/export | Desktop/server FFmpeg render job; H.264/AAC MP4 | Real macOS rendering and full decode in both hosts. Browser/WebView must support the codec. Live composited preview and proxy editing deferred. |
| Versioned persistence, autosave, undo/redo | Shared core | Atomic writes, revision conflicts, idempotency, reopen/restart, hash relink, package limits. Schema 1 is first format; future schema rejected. No invented historic migration. |
| Agent edits | Existing runtime, scoped tools, reviewed atomic proposals | Structured-tool tests plus actual Gemma planner on both hosts; manual continuation and stale rejection use same command service. TensorAgent uses its saved-conversation recorder; server stores bounded Studio history in the current browser, not a new durable server chat store. |
| Generate alternate takes | Existing native video pipeline | Real LTX text-to-video/audio in both hosts. Original take remains an asset; replacement preserves placement. Side-by-side comparison viewer deferred. |
| Generation admission | Model capabilities plus complete local output pipeline | Missing ffprobe, FFmpeg, H.264/AAC or disabled subprocess platform marks model unavailable with a reason and refuses before adapter validation/inference. |
| Generation/render jobs | Durable bounded queue, cancellation, restart interruption | Doubles for deterministic races; native MLX cancellation separately validated. Interrupted stages retry from the beginning. |
| Import, thumbnails, waveforms | Async probe, bounded cached 320×180 thumbnail and 512 peaks | Real FFmpeg checks. These operations are cancellable request work, not durable queued jobs. |
| Offline operation | No cloud account/upload path | Manual editing without model checked; native TensorAgent run with external networking denied. Installed local tools/models required. |
| Responsive layouts/accessibility/localization | Shared page, semantic controls, keyboard editing scoped to the timeline, Escape layers, focus kept across redraws, one announcer, 44 px touch targets, all eight languages for the page's own text | Headless Chrome desktop and 390×844 emulation; TensorAgent's page in Simplified Chinese. Host-side messages remain English. Screen-reader and native touch validation outstanding. |
| Windows/Linux desktop | Same managed implementation and executable discovery | Not built/run on Windows or Linux for this task. Encoder/runtime availability remains platform-specific. |
| iOS/Android | Project-package review/edit UI and capability gating | No desktop subprocess import/export; mobile generation disabled by Studio host policy. No physical device or mobile native-media export validation. |
| Explicit remote host | Uses the host serving the current interface | No automatic remote fallback, remote media upload or paid provider added. A separate TensorAgent remote Studio transport is deferred. |

### Native model inventory

These are all current implementations of `IVideoGenerationModel`. Controls use
the loaded checkpoint's capabilities, not just the family name.

| Family | Exposed Studio operations | Validation in this task |
|---|---|---|
| Wan (2.1 / 2.2 variants) | t2v where permitted; i2v for TI2V and I2V-A14B checkpoints; a last frame (alone or with a first) and ref disabled with a reason, and an end image refused by the pipeline rather than dropped; 4n+1 temporal grid, variant sampling defaults | Capability/mode contract tests only; no local Wan checkpoint and no real Wan Studio inference. |
| MiniMax-H3 partitions | Modes t2v/i2v/fl2v on FL2VA (fl2v with a last frame alone or with a first) and t2v/ref on Ref2VA (up to 9 reference images); the other partition's modes are disabled with a reason; available audio output; 17n+5 frames at 24 fps; with a Turbo plug-in on FL2VA, the plug-in's fixed steps and jobs bound to its sampling recipe | 2026-10-16: real FL2VA t2v/i2v/fl2v through the Server and Ref2VA t2v/ref through the TensorAgent harness on ggml_metal; see the generation modes follow-up. 2026-10-17: Turbo recipe binding by automated tests only; no Turbo take generated. 2026-10-17: last-frame-only FL2VA through the Server and the TensorAgent harness; see the optional frames follow-up. H3 model license has separate territory restrictions; no license changes or downloads authorized. |
| LTX 2.5 Distilled | Modes t2v/i2v/fl2v/keyframes (fl2v with a last frame alone or with a first); ref (up to 16 images, at least 121 frames, one full-size pass) only with the installed Ingredients LoRA; optional audio output; 8n+1 frames and fixed distilled recipe | Real T2V/audio at 512×512×17 on MLX in both hosts (2026-10-15). 2026-10-16: all five modes through the Server on MLX; see the generation modes follow-up. 2026-10-17: last frame only on MLX; see the optional frames follow-up. |

Audio/video reference inputs (H3 Ref2VA clips and soundtracks), arbitrary LoRA selection, native
retake/extension and masked generative edits are not exposed by this Studio
adapter. Existing model-specific APIs remain separate. Spatial/temporal grids,
fixed steps/CFG, reference counts and incompatible options are rejected before
expensive work. Unsupported options are not silently ignored.

## Reproduce

### Chat/Studio generation parity

The generation-quality regression was caused by Studio's fixed 512×512 default
overriding the LTX preset's 1024×1024 output. Both interfaces already called the
same native model and encoder. Studio now resolves its omitted settings from
the same host defaults as chat and exposes them to the generation form. Reused
takes keep their recorded settings; **Use model defaults** upgrades an old take
before regeneration.

Full-resolution validation also exposed excessive LTX decoder memory overlap.
TensorSharp releases denoiser device residency before decoding and releases
completed convolutional residual intermediates earlier. MLX materializes the
activation and residual boundaries so lazy graphs do not retain completed
layers. This preserves the decoder's convolution math and full clip geometry;
it does not introduce temporal tiling or lower the resolution. Numerical checks
include untraced CPU/MLX decoder oracles and denoiser release/reload consistency.

For a real fixed-seed comparison, start a host with the desired model/config and
run the following probe. It imports the same original image into both surfaces,
records the requests and effective settings, and compares every decoded frame.
The report distinguishes media equality from a subjective anatomy review; a
shorter preview does not validate a six-second generation.

```sh
python3 eng/studio/validate_generation_parity.py \
  --url http://127.0.0.1:5002 --image imgs/banner_1.png \
  --prompt-file /path/to/prompt.txt --seed 42 --frames 145 \
  --output artifacts/studio-quality/parity
```

Keep the resulting videos, frame hashes, reports and visual review snapshots in
the ignored output directory. Record the actual backend, ggml revision and any
unavailable devices or model scenarios with the run; passing request/unit tests
alone is not evidence of native generation quality.

### Editor and host checks

```sh
dotnet build TensorSharp.Server.Host/TensorSharp.Server.Host.csproj
dotnet build eng/validation/TensorAgentHost/TensorAgentHost.csproj
dotnet test TensorSharp.Studio.Tests/TensorSharp.Studio.Tests.csproj --nologo -v:q
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj --filter 'FullyQualifiedName~StudioHostTests|FullyQualifiedName~VideoTurnsTests|FullyQualifiedName~LoopbackServerTests|FullyQualifiedName~LoopbackRangeTests|FullyQualifiedName~MediaRoutesTests|FullyQualifiedName~AgentAppHostTests' --nologo -v:q
node --test eng/validation/tests/studio-ui.test.mjs
dotnet build TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj -f net10.0-maccatalyst -p:TensorSharpAppleTargets=true
```

Start a host as described in the [guide](video-studio.md), then:

```sh
python3 eng/studio/validate.py --url http://127.0.0.1:8187 --output artifacts/studio/server-check
python3 eng/studio/validate.py --connection artifacts/studio-agent/connection.json --output artifacts/studio/agent-check
# Add --generate only with an explicitly loaded real video model.
# After stopping/restarting the host with the same storage:
python3 eng/studio/validate.py --url http://127.0.0.1:8187 --reopen PROJECT_ID --output artifacts/studio/reopen-check
# Explicitly load a tool-capable text model before testing the actual planner:
python3 eng/studio/validate_agent.py --url http://127.0.0.1:8187 --output artifacts/studio/planner-check
python3 eng/studio/validate_cancel.py --connection artifacts/studio-agent/connection.json --project PROJECT_ID --output artifacts/studio/cancel-check
```

The harness needs native library discovery configured, just like the application.
For the recorded MLX run, `TENSORSHARP_MLX_LIBRARY_DIR` pointed to the built
`TensorSharp.Server.Host/bin` containing `libmlxc.dylib` and `libmlx.dylib`. The
first harness attempt without that setting failed model loading and was not
counted as successful inference. The network-denied run prepended:

```sh
/usr/bin/sandbox-exec -p '(version 1)(allow default)(deny network-outbound)(allow network-outbound (remote ip "localhost:*"))'
```

Do not publish harness `connection.json` or raw startup logs: they contain the
loopback authentication credential. Reports and project packages omit it.
