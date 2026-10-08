# TensorAgent

<p align="center"><a href="https://buymeacoffee.com/zhongkaifu"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>TensorSharp/TensorAgent is free. If you like it, a coffee keeps the work on it going.</sub></p>

A local AI app for **iPhone, iPad, Mac and Windows**. TensorAgent runs TensorSharp
in the app process for text and reasoning, multimodal questions, agentic code and
document work, **Qwen-Image 2.1 image generation and editing**, and MiniMax-H3 short
video with audio. Choose a model for the task and use the same composer, saved chats
and artifact links. Image edits support painted selections, protected pixels,
original/result comparison and twelve optional LoRA plug-ins. The interface is
available in [eight languages](#languages).

The .NET MAUI project shares its chat page and loopback API across iOS/iPadOS,
Mac Catalyst and Windows WinUI heads. Physical iPhones and iPads use GGML Metal
(`ggml_metal`), Macs use Metal or CPU, and Windows offers CUDA/Vulkan when available,
then CPU. See [On the desktop](#on-the-desktop-macos-and-windows) for platform differences.
The Apple device target is iOS/iPadOS 17.0 or later, arm64; build it with
`TensorSharpAppleTargets=true`.

**Download and install:** follow the **[TensorAgent Desktop user guide](../docs/tensoragent_desktop.md)**
([中文](../docs/tensoragent_desktop_zh-cn.md)) for macOS and Windows packages, model
setup and your first chat. The updated Release Binaries workflow publishes Desktop
DMG/PKG/ZIP and MSI/ZIP assets; historical releases may lack them. iPhone/iPad still
use the [source-build instructions](#build-and-run).

**Model and validation scope.** The 37 catalog entries start at the 12 GB system
RAM tier, with larger models requiring higher tiers. Qwen3.8 Flash Next has an
experimental UD-IQ1_M entry from 32 GB and a UD-Q2_K_XL entry from 48 GB,
both with SSD-backed weights and optional vision and draft companions. These eligibility tiers describe physical system RAM,
including unified memory on Apple devices; storage size and GPU VRAM are separate.
Image/video generation has been measured in the Mac app, with no recorded
iOS media generation or Windows image/audio/video generation. The physical-phone run recorded
below is on iOS 26.6.1; iOS 27 has not been device-verified.

Inference runs locally. The code sandbox has no network unless the user grants it.
Apple dictation requests on-device recognition where the selected language supports it;
otherwise Apple's recogniser may process the audio off the device. Windows code execution
requires the explicit **Run without a sandbox** setting.

<p align="center"><img src="../website/assets/screenshots/tensoragent-iphone.png" alt="TensorAgent on an iPhone: Gemma 4 E2B scaled a recipe from 4 to 10 people by running a Python script in the app's built-in Python, and answered with a table" width="300"></p>

<sub>Gemma 4 E2B (Q8_0) in the iPhone 17 Pro simulator, captured on 2026-09-30. The simulator
has no GPU, so the engine runs on `ggml_cpu` there; on an iPhone it uses Metal. The model
wrote a short script, ran it in the app's built-in Python, and answered with the table. The
Mac app is shown under [On the desktop](#on-the-desktop-macos-and-windows).</sub>

## Install TensorAgent Desktop

Open the **[latest release](https://github.com/zhongkaifu/TensorSharp/releases/latest)**
and expand **Assets**. The desktop filenames start with `tensoragent-desktop-`,
followed by the release version and platform:

| Computer | Install package |
|---|---|
| Apple Silicon Mac, macOS 14+ | `tensoragent-desktop-<version>-osx-arm64.dmg`: drag TensorAgent to Applications; PKG and ZIP alternatives are available. |
| Windows x64, CPU | `tensoragent-desktop-<version>-win-x64-cpu.msi`: install for your user and open TensorAgent from Start; ZIP is available. |
| Windows x64, compatible NVIDIA GPU/driver | `tensoragent-desktop-<version>-win-x64-cuda.msi`: includes CUDA runtime libraries; ZIP is available. |

Desktop packages carry their .NET runtime and native engine. Windows WebView2 is
a separate prerequisite; Python/Node are optional system tools for skills, rather
than bundled desktop interpreters. Earlier Mac packages used ad hoc signing without
Apple notarization, which can produce the “Apple could not verify TensorAgent is free
of malware” launch warning. The updated Mac release pipeline requires Developer ID
signing and notarization; a corrected download still needs a successful release run
with the maintainer's Apple credentials. Existing downloads remain unchanged.
Windows installers are unsigned. See the [installation guide](../docs/tensoragent_desktop.md#2-install-and-open)
for verification and the per-app macOS opening procedure. Old release pages may
contain only CLI/server archives; workflow changes do not add Desktop files to past releases.

After opening the app, choose **☰ → Models → Download → Use**, wait for loading,
and type your first message. Model weights are separate downloads, and catalog
eligibility starts at 12 GB physical RAM. Use the [full user guide](../docs/tensoragent_desktop.md)
for storage setup, attachments, permissions, generated files, updates and troubleshooting.

### Release packaging for maintainers

[Release Binaries](../.github/workflows/release-binaries.yml) publishes on a `v<version>`
tag push; a manual run accepts the version without `v` and creates a draft by
default. Packaging accepts three-part numeric versions such as `2.8.6` or
`2026.10.03`, with optional prerelease/build labels, validated by
[the version resolver](../eng/resolve-release-version.py). Calendar tags map to
Windows Installer's limits (`2026.10.03` becomes MSI version `26.10.3`); the complete
release version remains in asset names. The workflow pins .NET SDK 10.0.302,
workload set 10.0.303.1 and Xcode 26.6 for its hosted Mac build, installs both Apple
MAUI workloads for restore, and builds the native macOS engine with a 14.0
deployment floor. Local builds still need a workload/Xcode combination that matches
their installed Xcode.

The [Mac packager](../eng/package-tensoragent-macos.sh) requires a Developer ID
Application signature and Hardened Runtime by default. It notarizes and staples
the app before creating ZIP/DMG/PKG packages, signs and notarizes the DMG and PKG,
staples their tickets, and checks Gatekeeper acceptance before writing checksums.
The ZIP contains the stapled app. Missing credentials or rejected notarization
fail packaging; there is no automatic ad hoc fallback. Follow Apple's
[notarization requirements](https://developer.apple.com/documentation/security/notarizing-macos-software-before-distribution)
when obtaining Developer ID certificates and a provisioning profile matching the
app's bundle identifier.

Configure these **GitHub Actions repository secrets** before running the Mac release job:

| Secret | Value |
|---|---|
| `TENSORAGENT_MACOS_APPLICATION_P12_BASE64` | Base64-encoded Developer ID Application certificate and private key export (`.p12`). |
| `TENSORAGENT_MACOS_APPLICATION_P12_PASSWORD` | Password for that application certificate export. |
| `TENSORAGENT_MACOS_INSTALLER_P12_BASE64` | Base64-encoded Developer ID Installer certificate and private key export (`.p12`). |
| `TENSORAGENT_MACOS_INSTALLER_P12_PASSWORD` | Password for that installer certificate export. |
| `TENSORAGENT_MACOS_PROVISION_PROFILE_BASE64` | Base64-encoded Developer ID provisioning profile for TensorAgent. |
| `TENSORAGENT_MACOS_NOTARY_APPLE_ID` | Apple Account used for notarization. |
| `TENSORAGENT_MACOS_NOTARY_APP_PASSWORD` | App-specific password for that account. |
| `TENSORAGENT_MACOS_TEAM_ID` | Developer Program team ID matching both certificates and the provisioning profile. |

The workflow imports certificates into a temporary keychain, derives signing
identities from them, installs the provisioning profile, stores notarization
credentials, and removes the temporary keychain and profile after the job.
Notarization diagnostics are workflow artifacts, separate from release assets.
These settings apply to future successful release runs; this source change does
not establish that a signed replacement DMG has already been published.

For a local distribution build, install the same certificates and profile, then
publish with their identity and profile UUID. Replace the sample version and build
number with the values for your release:

```bash
export MACOSX_DEPLOYMENT_TARGET=14.0
export TENSORAGENT_APPLICATION_IDENTITY="Developer ID Application: Your Name (TEAMID)"
dotnet publish TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj \
  -c Release -f net10.0-maccatalyst -r maccatalyst-arm64 --self-contained true \
  -p:Version=2.8.6 -p:ApplicationDisplayVersion=2.8.6 -p:ApplicationVersion=1 \
  -p:TensorSharpAppleTargets=true -p:CreatePackage=false \
  -p:UseHardenedRuntime=true \
  -p:CodesignKey="$TENSORAGENT_APPLICATION_IDENTITY" \
  -p:CodesignProvision="DEVELOPER_ID_PROFILE_UUID" \
  -p:CodesignEntitlements=Platforms/MacCatalyst/Entitlements.plist

# Store the notarization credentials in Keychain; enter the app-specific password when prompted.
xcrun notarytool store-credentials TensorAgent \
  --apple-id "your-apple-account@example.com" --team-id TEAMID
export TENSORAGENT_INSTALLER_IDENTITY="Developer ID Installer: Your Name (TEAMID)"
export TENSORAGENT_NOTARY_PROFILE=TensorAgent
bash eng/package-tensoragent-macos.sh \
  TensorAgent/src/TensorAgent.Maui/bin/Release/net10.0-maccatalyst/maccatalyst-arm64/TensorAgent.app \
  2.8.6 artifacts
```

The packager requires `TENSORAGENT_APPLICATION_IDENTITY`,
`TENSORAGENT_INSTALLER_IDENTITY` and `TENSORAGENT_NOTARY_PROFILE`. For a custom
keychain, also set `TENSORAGENT_SIGNING_KEYCHAIN`, store the notary credentials in
that keychain with `notarytool store-credentials --keychain`, and pass its path as
`CodesignKeychain` when publishing.

To produce only a DMG while resuming an existing app ZIP notarization submission,
use the [DMG finalizer](../eng/finalize-tensoragent-dmg.py). Supply the matching
Developer ID Application-signed app and stored notary profile; no Developer ID
Installer certificate is needed:

```bash
python3 eng/finalize-tensoragent-dmg.py \
  --app "/path/to/TensorAgent.app" --version "<version>" \
  --identity "Developer ID Application: <name> (<team-id>)" \
  --notary-profile "<stored-profile>" \
  --app-submission-id "<existing-submission-uuid>" \
  --output artifacts/tensoragent-dmg-final --timeout 48h
```

Keep the output directory after a timeout: its receipts and checkpoints allow
you to rerun the same command to resume. The finalizer promotes the DMG and writes
its checksum only after notarization, stapled-ticket validation and Gatekeeper
checks pass for both the app and disk image.

Use `--ad-hoc` only for local packaging tests, as the fourth argument after the
output directory: `bash eng/package-tensoragent-macos.sh APP VERSION artifacts --ad-hoc`.
It accepts a local ad hoc bundle, skips distribution signing, notarization and
Gatekeeper acceptance, and emits a warning. These test packages can trigger the
same macOS warning when copied to another machine; do not publish them as a
Gatekeeper-ready release.

The [Windows packager](../eng/package-tensoragent-windows.ps1) and
[WiX generator](../eng/generate-tensoragent-wix.py) validate a self-contained
publish folder and create unsigned MSI/ZIP packages, including per-user
installation and a Start menu shortcut.
Each release includes `SHA256SUMS-tensoragent-desktop-<version>-<variant>.txt` and
`tensoragent-desktop-<version>-<variant>-BUILD.txt`. BUILD records the unchanged
upstream ggml revision and actual packaging checks: Mac signatures/architecture,
notarization/stapling and Gatekeeper checks for distribution packages, archive
structure, Windows ZIP/MSI payload hashes, and package checksums. Ad hoc packaging
skips notarization and Gatekeeper acceptance; those checks must be recorded as skipped.
Those checks do not establish clean-machine installation, GUI launch, model
inference, GPU coverage or benchmark results. Keep any additional generated
validation evidence under ignored `docs/validation/` or `artifacts/`.

## What it does

**Chat with the desktop's capabilities, on a page built for a thumb.** The app
ships its own page — `src/TensorAgent.Maui/wwwroot/index.html`, bundled as
`webui/` and served from the loopback host. It is not the desktop page: that one
is laid out for a mouse and a wide window, and no amount of injected CSS turns it
into a phone UI. One row of chrome, everything reachable at the bottom next to
the keyboard, a layout that follows `visualViewport`, a single `+` sheet for
Photo / Camera / Video / File, and reasoning collapsed behind a disclosure.

Press **Enter** to send and **Shift+Enter** for a new line. Enter used to confirm an
IME composition does not send, and pressing it during generation does not stop the turn.
After a text turn, the answer shows its token count, elapsed time and tokens per second,
plus prompt/KV reuse when reported and a stopped/truncated marker when applicable.
Those statistics are saved with the conversation and return when it is reopened;
image/video turns do not show the placeholder zero-token counters.

What is shared is the API, not the document. The routes under the page —
`/api/chat`, `/api/models`, `/api/sessions`, `/api/upload`, `/api/skills`,
`/api/image-edit`, `/api/video-generate` — are bound to the same
`WebUiChatService` and `SkillsService` the desktop server binds, so streaming,
tool progress, reasoning blocks, skill steps and artifact links behave
identically. The main things the desktop page has that are not here: the phone
page draws no sub-agent progress panel (it ignores the `agents` field of
tool-progress frames, and only names the sub-agent tools in words — "Starting
sub-agent", "Waiting for sub-agents", "Messaging sub-agent", "Stopping sub-agent",
"Checking sub-agents", or "Preparing …" while the call is being written), and it has
no image or video controls of its own: with Qwen-Image 2.1 or MiniMax-H3 loaded, the
message itself asks for a picture or a clip, through `/api/chat` (`ImageTurns`,
`VideoTurns`; see "Pictures" and "Videos" below). So `/api/image-edit` and
`/api/video-generate` are bound, but nothing on the page calls them, and
`/api/image-generate`, the route the desktop page uses for text-to-image, is not bound
at all. The app's own client is appended as one script tag at request time; the page
file itself is never forked.

**A built-in model catalog.** Downloadable checkpoints cover TensorSharp’s supported
chat, image and video generation families, with the exact byte size and SHA-256 of
every file. Models are offered in phone/tablet and desktop/workstation memory tiers.
Downloads
resume from a kept `.part` after an interruption, are verified before use, and
belong to the APP rather than to the screen that started one — see "Downloads"
below. Bonsai 2 27B is the one entry that needs the 16 GB tier (iPads and Macs)
rather than a 12 GB phone: TensorSharp repacks its PTQ1_0 weights losslessly to
GGML Q2_0 at load (about 29% more payload), so the 5.95 GB download occupies about
7.7 GB of anonymous memory, which with the K/V cache and compute buffers exceeds
the roughly 8.5 GB a 12 GB iPhone grants. The PQ2_0 file (7.21 GB) holds the same
ternary weights and repacks to about the same size, so the entry downloads the
smaller PTQ1_0 file. See the [Bonsai2 card](../docs/models/bonsai2.md).
Qwen3.8 27B and Muse-Glimmer 30B are models a phone could run only at one or two bits,
offered at four where the memory exists; Qwen-Image 2.1 makes and edits pictures; and the
two MiniMax-H3 entries, one model in two checkpoints, make short videos with their own
soundtrack, one from a description or a starting photo, the other around the photos,
clips and recordings it is given. Qwen3.8 Flash Next, a 125B mixture of experts (about 6B
active per token), has two catalog quantizations. UD-Q2_K_XL starts at the
48 GB system RAM tier; in the recorded Mac run the engine kept 18.3 GB of its 78.9 GB
package resident and read the rest from SSD as tokens needed it. UD-IQ1_M starts at an
experimental 32 GB system RAM tier. Both variants offer an optional BF16 vision
projector for images/video frames and a shared Q8_0 MTP draft head. Its three weight shards need 74,538,755,776 bytes of storage (74.54 GB), or
75,446,298,720 bytes (75.45 GB) with the projector. SSD paging allows these packages to
exceed RAM; the engine plans expert placement for the available accelerator memory.
See [The desktop's larger models](#the-desktops-larger-models) for the historical Mac measurements.

| Model | Modalities | Required artifact(s) | Minimum system RAM tier | Source |
| --- | --- | --- | --- | --- |
| Gemma 4 E2B (Q8_0) | text, image, audio, video | download: 4,967,497,152-byte main GGUF + 557,368,064-byte projector | 12 GB | `ggml-org/gemma-4-E2B-it-GGUF` |
| Gemma 4 E4B (IQ4_XS) | text, image, audio, video | download: 4,715,416,704-byte main GGUF + 559,874,816-byte projector; 98,653,280-byte draft optional | 12 GB | `unsloth/gemma-4-E4B-it-GGUF` + `ggml-org/gemma-4-E4B-it-GGUF` (projector and draft) |
| Gemma 4 12B (UD-IQ2_M) | text; image and video with optional projector | download: 4,213,353,280-byte main GGUF; 175,115,840-byte projector and 465,109,248-byte draft optional | 12 GB | `unsloth/gemma-4-12b-it-GGUF` |
| Bonsai 2 27B (PTQ1_0) | text; image with optional projector | download: 5,946,648,928-byte main GGUF; 629,246,976-byte projector optional | 16 GB | `prism-ml/Ternary-Bonsai-2-27B-gguf` |
| Qwen3.5 9B (IQ4_XS) | text; image and video with optional projector | download: 5,168,653,536-byte main GGUF; 918,166,080-byte projector optional | 12 GB | `unsloth/Qwen3.5-9B-GGUF` |
| Qwen3.8 27B (UD-Q4_K_XL) | text; image and video with optional projector | download: 17,559,178,144-byte main GGUF; 927,607,488-byte projector optional | 32 GB | `unsloth/Qwen3.8-27B-GGUF` |
| Muse-Glimmer 30B (UD-Q4_K_XL) | text; image with optional projector | download: 15,878,222,368-byte main GGUF; 2,051,685,088-byte projector optional | 32 GB | `unsloth/Muse-Glimmer-30B-GGUF` |
| Qwen3.8 Flash Next (UD-Q2_K_XL) | text; images/video with optional projector | download: three shards of 10,946,624, 49,979,779,296 and 28,878,402,944 bytes (78.9 GB); 907,542,944-byte projector and 2,786,568,256-byte shared MTP head optional | 48 GB | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Qwen3.8 Flash Next (UD-IQ1_M) | text; images/video with optional projector | download: three shards of 10,946,624, 49,988,981,792 and 24,538,827,360 bytes (74.54 GB total); 907,542,944-byte projector and 2,786,568,256-byte shared MTP head optional | 32 GB, experimental | `unsloth/Qwen3.8-Flash-Next-GGUF` |
| Qwen-Image 2.1 (Q4_K_M) | text or a photo in, a picture out | download: 4,189,343,904-byte DiT + 5,027,784,800-byte Qwen3-VL-8B text encoder + 675,509,688-byte VAE + 1,159,029,824-byte vision projector | 24 GB | `Abiray/Qwen-Image-2.1-GGUF` + `Qwen/Qwen3-VL-8B-Instruct-GGUF` + `Comfy-Org/Qwen-Image-2.1` |
| MiniMax-H3 (Q4_K) | text, or one or two photos as the first and last frames, in; a video with its soundtrack out | download: 11,420,663,904-byte denoiser + 18,218,065,024-byte Qwen3-VL-32B text encoder + 5,207,808,496-byte video VAE + 605,254,808-byte audio VAE + 2,776,833-byte `vocab.json`, 1,671,839-byte `merges.txt` and 11,003-byte `tokenizer_config.json` | 32 GB | `unsloth/MiniMax-H3-GGUF` + `MiniMaxAI/MiniMax-H3` (tokenizer files) |
| MiniMax-H3 References (Q4_K) | text with up to nine photos, clips and recordings to feature in; a video with its soundtrack out | download: 11,381,096,544-byte denoiser, plus the same four companions as MiniMax-H3 (six files, 24.0 GB), linked from it when it is installed rather than downloaded again | 32 GB | `unsloth/MiniMax-H3-GGUF` + `MiniMaxAI/MiniMax-H3` (tokenizer files) |

Additional entries in [ModelCatalog.Extended.cs](src/TensorAgent.Core/Catalog/ModelCatalog.Extended.cs)
cover Gemma 4 26B/31B, Qwen 3.5/3.6, GPT-OSS, DeepSeek V4/V4.1 Flash,
GLM 5.2/5.3/5.3 Flash, Nemotron, Mistral 3, Hunyuan translation, DiffusionGemma,
and Wan 2.1/2.2 (including Turbo and dual-expert variants). Each includes every
required shard and network; supported vision and speculative companions are optional.
Qwen3.8 27B offers DFlash2 and Muse-Glimmer offers DFlash. Wan downloads its UMT5
encoder, matching VAE and, for A14B, both denoisers. DiffusionGemma’s optional
vision tower is a safetensors shard rather than an mmproj GGUF.

These additions are **experimental**: published file identities and catalog wiring
have been checked, but their RAM tiers are conservative estimates, not new app/device
benchmarks. See the [model cards](../docs/models/README.md) for supported backends and
actual inference coverage. DeepSeek V4.1’s public `clip` projector and `dflash` DSpark
exports do not match TensorSharp’s required formats; its catalog entry is text-only.
Its converted vision/DSpark companions remain available through the model card’s
CLI/server instructions. Nemotron audio likewise needs a separately converted tower.
Embedding encoders are served through TensorSharp’s embedding API/configs, not selected
as conversation models in TensorAgent. The catalog chooses representative quantizations
for the supported model families, rather than listing every publisher’s quantization.

Each chat entry also carries the context window the app loads it with (8,192 tokens for
Gemma 4 E2B and E4B; other entries carry their own limits), a K/V cache
precision that the "KV cache precision" setting overrides, and its model card's
sampling values (for Bonsai 2 27B, the publisher's thinking-mode recommendation:
temperature 1.0, top-k 20, top-p 0.95, min-p 0.05; for Qwen3.8 Flash Next, the
`general.sampling` values its GGUF carries: temperature 1.0, top-k 20, top-p 0.95,
min-p 0). The Bonsai 2 card is marked Experimental: Bonsai2 has not been validated on
iOS. Both Qwen3.8 Flash Next entries are Experimental and read most of their weights
from SSD: how fast they answer depends on the available page cache and the SSD. Both
support thinking, load a 32,768-token context with FP16 K/V, and use `LeanCaches` to
limit the app's cache budget. Their license is
[Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE).
The `qwen3.8-flash-next-iq1m` entry was validated through TensorAgent's catalog load
path (`AgentAppHost.UseModel`) and Web UI HTTP/SSE on Windows with an RTX 3080 Laptop
GPU (16 GB VRAM) and 32 GB system RAM. Text, image description and continuations
completed; the 496-token image answer exactly matched the prior server answer, and
the complete 1,115-token image follow-up reused 2,494 of 2,508 prompt tokens. This run
used FP16 K/V, a 32,768-token context, lean caches and speculation disabled.
The default n-gram speculation setting was also enabled through the real settings
route for two short arithmetic turns: both returned `4` at EOS, with 69 of 89
prompt tokens reused on the follow-up.
The Windows Release app was built and its eight embedded catalog translations verified
against source. These checks do not cover WinUI clicks, Mac/Metal, CPU-only devices,
or performance comparisons between quantizations. Generated evidence is kept in
ignored `docs/validation/tensoragent-qwen-iq1m/`.

The Models page groups entries into expandable model families. Search shows matching
models directly, ranked by relevance, and accepts names, sizes, quantizations,
capabilities, and companion names such as `mmproj` or `dflash`. **Can run here**
filters by the device's memory tier; **Downloaded** shows installed weights. Clear
search to return to the previous expanded families, or use the selected-model shortcut
to find the current model. Downloads keep progressing while their family is collapsed
or a different search is shown. A hidden download finishing does not interrupt browsing.

The list includes every entry, but only one that fits the device's memory tier
can be loaded: an entry that needs more is shown greyed, marked "Too big" with both
numbers, rather than hidden. Bonsai 2 27B needs the 16 GB tier, Qwen-Image 2.1 the 24 GB
tier, Qwen3.8 27B, Muse-Glimmer 30B, both MiniMax-H3 entries and Qwen3.8 Flash Next
UD-IQ1_M the 32 GB tier, and Flash Next UD-Q2_K_XL the 48 GB tier. Larger entries
use workstation tiers up to 512 GB; the card shows the requirement. Eligibility uses system RAM on both Mac and Windows;
it does not certify every backend or device at that tier. For an installed model
whose optional projector is missing, **Add vision** downloads just the projector.
**Add draft** independently downloads its optional speculative model. Adding a companion
to the selected model reloads it so the new file is used; speculation still follows the
Speculative decoding setting and the engine’s backend/sampler capability checks.

**Pictures.** With Qwen-Image 2.1 loaded, a message is a description of a picture, or,
with a photo attached, what to change about it. It is the same `/api/chat` turn, under
the same turn manager and transcript, as an answer: the page shows the denoising steps,
refreshing one picture in place from the small previews the engine sends, and the finished
picture stays in the saved chat. The app asks for 1024x1024 (an edit keeps the photo's
shape at the same area) at the model's own 40 steps, rather than its native 2048x2048,
which has four times the image tokens.

For a local edit, choose **+ → Photo**, attach photos, and choose **Select area**
beside any thumbnail above the message box, before sending. Saving a selection makes
that photo the **Editing target** and moves it first; the other photos remain attached
as references. Each photo remembers its own selection while you prepare the draft.
Cancelling the editor or a failed save keeps the previous target. Removing the target
leaves other saved selections inactive; choose **Adjust** and save to apply one. You can
paint and save the selection before loading a model. To apply the edit, load
**Qwen-Image 2.1**, describe what to change, and send. If another model is selected,
the saved selection shows **Open Models** and keeps the draft until Qwen is loaded.
The shared desktop and mobile editor supports mouse, touch and stylus,
brush and eraser, undo/redo, invert, pan and zoom. Pink marks editable pixels; the
exported grayscale mask uses white for edits and black for protected pixels. Edge
softness fades inward, so it never expands the selected area. **Process selected
region only** reduces model work for small selections, with less surrounding context.
The result keeps the source dimensions and exact protected RGBA pixels. Each message
edits one target; only its selection is sent and saved with the conversation.
**Compare original** toggles the result, and **Edit again** restores the target,
selection, references, other attachments and instruction. The browser editor has no
fixed megapixel or per-side limit; available browser memory and canvas support
determine the practical maximum image size.
HEIC/HEIF photos use a full-resolution PNG for editing and comparison while their
small preview remains in the attachment list. Reattach older HEIC/HEIF uploads if
the editor asks for a full-resolution source.

**LoRA plug-ins.** With Qwen-Image 2.1 loaded, Model > LoRA plug-ins lists twelve LoRAs
made for it. Each is pinned to a commit and a SHA-256 in `LoraCatalog` and downloaded on its
own, 80-680 MB, into `Library/Caches/TensorAgent/loras/<id>/`. A switch turns one on, a
slider sets the strength of a style or an edit (10-150%), and Remove deletes the files and
turns it off. Speed plug-ins (Viggle Turbo, 6 steps; Pruna 8-Step; Pruna 5-Step; Fun-Acc
4-Step) replace the model's 40 steps with their own schedule, so only one can be on and
turning on another turns it off. Styles (Film Stills, Grainscape, Quality Fix) apply to every
picture. Edits (Detail Enhancer, Natural Exposure, Object Remover, Object Mover, Anime
Consistency) apply only when a photo is attached, and the sheet shows the phrase a request
needs for the ones trained on one. The choice is saved (`imageLoras` in the settings) and
applies from the next picture, never to one being drawn, and the progress line names what
the picture is drawn with ("Drawing with Viggle Turbo + Film Stills… step 3 of 6"). Object
Remover works only at the model's own 40 steps, so a speed plug-in sits out the edits it
applies to; on its own example photos it removed both marked cats but none of the marked
cars, which is why its row says to check the result. A plug-in that is on but whose files
have gone refuses the picture with the reason rather than drawing without it, and its row
keeps the switch that turns it off. Most are under the Qwen Research License
(non-commercial), Anime Consistency is Apache-2.0, and the authors of Quality Fix and Detail
Enhancer state none; each row says which.

Measured in the Release app on an M5 Pro, 1024x1024 (`TENSORAGENT_IMAGE_BENCH`): a picture
from words took 321.5 s at the model's 40 steps and 58.3 s with Viggle Turbo, applying it
included; 58.8 s with Viggle Turbo and Film Stills, and 60.4 s with Viggle Turbo, Quality Fix
and Grainscape. Applying a new set took 0.3 s for one or two plug-ins and 1.3 s for those
three, and a set that does not change costs nothing. In the Debug build Pruna 8-Step took
79.0 s, Pruna 5-Step 53.6 s and Fun-Acc 4-Step 44.7 s, and an edit of a 1253x836 photo (made
at 1248x832) 406.2 s, 86.5 s with Viggle Turbo. A plug-in makes each step 5-6% dearer,
whatever its rank: by the engine's own step timer, a step took 7.8 s without plug-ins, 8.2-8.3 s
with any one of them, 8.3 s with two and 8.4 s with three. (A few-step schedule's whole
picture costs a little more a step than that, because most of its steps also decode a
preview.) Before this was fixed, applying plug-ins took 19-31 s in the Release app and 0.6 s
in the CLI: on Mono, TensorPrimitives' generic vector operators ran hundreds of times slower
than on CoreCLR, so `QwenImage21LoraSet` now uses plain loops. Without plug-ins the Debug
app's picture is bit-identical to the one it made before plug-ins existed, and with them it
is pixel-identical to the CLI's `--lora` run on the same files. The Release build's pictures
differ from those by at most 2 levels in 3% of pixels without plug-ins and 5 levels in 5% of
pixels with Viggle Turbo and Film Stills: Mono's LLVM code generation rounds some arithmetic
differently, plug-ins or not.

**Videos.** With MiniMax-H3 loaded, a message describes a short clip, and an attached
photo is its first frame (two photos, its first and last); with MiniMax-H3 References,
the photos, clips and recordings attached, up to nine together, are the people, things,
places and sounds the new scene features. It is the same `/api/chat` turn again
(`VideoTurns`): the page names each stage ("Reading the description…", "Filming… step N
of 20 · about X min left", "Developing the frames…", "Adding the sound…", "Saving the
video…"), then plays the clip inline and looped, with its sound (an AAC track inside the
MP4), and the clip stays in the saved chat. What a checkpoint cannot use is refused before
the model starts, in the app's words: a clip or a recording on the keyframes entry names
MiniMax-H3 References, and a document is refused rather than read as the script. There is
no size, length or step setting; the app asks for 640x384 (or that area at the photo's
shape, 608x416 for a 3:2 photo), 22 frames (0.92 s at 24 fps) and the model's own 20
steps, as the shipped `config/minimax-h3-*.json` do. On an M5 Pro the app made a clip
from words in 150.0-152.9 s, one from a photo in 216.5-223 s and one around a reference
photo in 199-210 s. Longer clips cost more than in proportion, because every
step attends over the whole clip: in the CLI 22 frames took 147.7 s, 39 frames 278.5 s
and 56 frames 424.4 s. A reference gives a subject's look and the description puts them
in the scene, so say who or what is in the shot: a description that said only "the
subject of the picture" produced a garden with nobody in it. The denoisers' license, the
MiniMax H3 Community License, excludes the EU, the UK, the Republic of Korea and the US
from its territory, as both entries' notes say; the Qwen3-VL text encoder is Apache-2.0.

**Many chats, kept, and one tap away.** The Web UI holds its history in the page and
nowhere else, which is fine for a desktop tab and useless on a phone that is suspended
and killed constantly. Transcripts are written on the host side instead, indexed, and
resumable: opening a saved chat re-renders it through the page's own bubble builders and
continues it, in place, without reloading the page.

The menu is a drawer from the LEFT edge and the saved chats are IN it, newest first —
not a row called "Chats" leading to a second screen. A chat someone has already had is
the thing the menu is opened for; a bottom sheet fits five rows and could only ever
offer the word. "All chats" is still there for renaming and deleting.

**Multi-modal input.** Photos, camera capture, video and files reach the same chat
service the paperclip's `/api/upload` reaches, so a native attachment and an in-page one
are the same thing by the time a message is sent — but the native ones are handed to it
DIRECTLY rather than posted over the loopback socket. The round trip was moving a file
this process already had, to a server inside this same process, through a hand-written
multipart parser; when the parser dropped a body the answer was "no file was uploaded"
and it named none of the five places that could have happened. HEIC works, which matters
because it is the iPhone camera's default format — and so does a photo that arrives with
no file extension at all, which is what iOS's picker actually hands over
(`UploadNaming`).

**Share into TensorAgent from other apps.** “Ask TensorAgent” is an iOS Share
extension for Safari, Photos, Mail, Messages, Files, Reddit, and any other app that
offers text, links, webpages, images, movies, audio, PDFs, or supported text/code documents. It copies the
share into a private App Group envelope, preserving large files with file-to-file I/O,
and TensorAgent imports those files through the same `/api/upload` service as every
other attachment. Safari also runs the bundled preprocessing script so an on-device
model receives the visible article text and selection, not only a URL it cannot fetch.

The normal result is an unsent draft in the chat composer:
`What can you tell me about this?` followed by the shared content, with shared files
shown as attachment chips. Existing draft text and attachments are preserved. The
share sheet offers prompt presets. Every separate share action is queued as its own
durable envelope and opens its own fresh chat after the preceding share is sent or
removed; independent shares are never combined into one composer or conversation.
Sharing never sends a model turn automatically: the user can review/edit the draft,
press Send, or remove the visible shared-item chip to discard the durable handoff and
its staged files.

The envelope is atomic and durable across a cold launch, app suspension, WebView
reload, or import failure. Merely showing the draft does not delete it; it is atomically
acknowledged only after the user sends and the accepted turn is written to the
conversation store (or explicitly discards it). iOS does not permit a general Share
extension to launch or foreground its containing app. If notification permission was
already granted, the extension posts a content-free “Shared item ready” notification
that opens TensorAgent with one tap; otherwise it confirms the save and asks the user
to open TensorAgent. The app consumes the inbox at launch and on every foreground.
Likewise, pressing Copy alone does not address an app and cannot wake TensorAgent;
use the source app's Share action and choose “Ask TensorAgent.” No private responder
chain or sensitive custom-URL fallback is used.

**Voice, by gesture.** Hold the message box for half a second and the composer
becomes one large hold-to-talk button; hold it, speak, release, and the transcription
lands in the message box for you to read before sending. A keyboard button beside it
goes back to typing. Recognition is Apple's, asked for on-device wherever the
language supports it.

iOS recognises ONE language per session and cannot detect which is being spoken, so
the chips beside the button choose it — and "Auto" means the first of *your* preferred
languages this device can recognise, not the region your phone formats dates in.
Those are different things, and the difference is not subtle: a phone set to the United
States reports `en_US` however many languages its owner has added, and an English
recogniser does not fail on Mandarin — it succeeds, and hands back the sounds
romanised.

**A live sign that it is working, and a trace of what it did.** A turn can spend a
minute between the question and the first word of the answer — reading a skill, writing
a program, running it — and for all of that the reply bubble is empty. Two things fill
it, both on by default:

- *What it is doing now.* A strip pinned directly ABOVE the message box names the step
  ("Running code… 12s", the seconds ticking) and shows the last three lines of whatever
  the model is producing: its reasoning, the command it is typing, or that command's
  output as it prints. Pinned, because a turn's activity starts at the top of the turn
  and by the time a program has been run the answer is streaming several screens below
  it — so the one question the user has ("is it stuck?") was the one thing they had to
  scroll away from the answer to find out. Above rather than below, because the
  composer is anchored to the bottom of the screen: growing it upwards leaves the box
  the thumb aims at exactly where it was.
- *What it has done.* One line per finished step, kept — `Reading skill ·
  documents · SKILL.md`, `Running code · python3 -c "from datetime…" · 3s`,
  with a red dot when a step failed — and a link for every file a script produced,
  rendered from the frame that reports it rather than from the model remembering to
  mention it. The desktop page deletes its activity block and keeps no history; on a
  phone that trace IS the answer to "what did it just spend a minute on".

The whole of the reasoning stays one tap away in the collapsed box above.

Tapping a link to a file the model's code produced opens it natively: Quick Look
(`QLPreviewController`) for anything it can render, the share sheet for anything
else (`FilePresenter`). The page hands the tap to the app because the WebView cannot
open it — the artifact route serves program-written files as attachments, and a
WKWebView with no download delegate drops those.

**A generation that starts repeating itself is stopped, and the stop is named.** A
4-bit model writing a long block of XML inside a script fell into `","+","+","+"`
and produced it 230 times over five and a half minutes, ending only when Stop was
tapped: the repetition penalties are deliberately off for code (they corrupt
legitimately repetitive structure), the reply limit was hundreds of thousands of
tokens, and a loop never reaches end-of-sequence. The engine now watches every
generation for an exact loop — a unit of at most 64 tokens, repeated at least eight
times and over at least 128 tokens — and ends it with the finish reason
`repetition` (`RepetitionGuard`). A tool-calling turn then tells the model what
repeated and how often, runs nothing from that round, and lets it try once more,
differently; a plain answer ends with a one-line note instead of a wall of the
same phrase.

**One row of chrome, and everything else in the menu.** The composer is a "+", the
message box and Send — nothing else. Reasoning is a Settings choice ("Show reasoning by
default"), Skills is a ☰ menu item beside Chats and Models, and the dictation language
appears only in voice mode. Each was a permanent control for something decided rarely,
on the one row a phone composer has.

**In the user's language.** The interface (the chat page, the native screens, the
messages the host sends and the share extension) is in English, Simplified Chinese,
Traditional Chinese, Japanese, Korean, Spanish, French and German. A first launch uses
the first of the system's preferred languages that the app has, or English if it has none
of them. Settings > Language changes it at once, without a restart. The choice is saved
(`uiLanguage` in `settings.json`) and used from then on, and "System" goes back to
following the system. The model is not told: it answers in whatever language you write
to it; system prompts, tool descriptions and tool results stay English. Share presets
and their draft messages use the interface language. The chat retains unsent text,
attachments and saved image selections while refreshing its language; an upload or a
send awaiting acceptance settles before that refresh.
iOS and macOS show their own texts for the app, such as the permission prompts and the
share sheet's "Ask TensorAgent", in the language the system picks for the app, not the
in-app choice. See [Languages](#languages).

**The answer keeps being written while you are somewhere else in the app.** A
generation belongs to the app, not to the HTTP request that asked for it
(`ChatTurnManager`). This is not a refinement: iOS suspends a WKWebView's content
process the moment its view leaves the window, which is what opening the model list does, so the page stops reading — and a
server that took that as "nobody wants this any more" threw away the minute the user had
just waited. Now the turn runs on, buffers what it produces, and the page ATTACHES to it
again when it comes back, replaying from the first frame; the display is held awake
(unless `keepAwakeWhileGenerating` is off, see Settings below) and a background-task
assertion is taken for as long as the model is working, both driven by
the host's own answer rather than by whichever page happens to be watching. Stopping is
something the Stop button asks for explicitly. The transcript is written by the turn, so
an answer that finishes with nobody reading is still there. Leaving the app itself is
different: iOS forbids GPU work from the background, so the turn pauses while
TensorAgent is not in front and carries on from the same token when it returns (see
"Leaving the APP mid-answer" below).

**The model you last used is loaded at launch.** The app has always remembered the
choice and then done nothing with it until you went back to the Models list and tapped
"Use" again, so every launch began at "No model yet" with a send button that refuses.
The weights are read on a background thread while the page paints, and the header says
"Loading Gemma 4 E2B…" until they are in — which is a different sentence from "no model
has ever been chosen", and asks the user for something different.

**A tool description is read on every turn, so nothing task-specific belongs in one.**
The shell tool's description carried a complete Yahoo Finance screener program for a
while: 3,204 characters of URL, response fields and a ten-row table, added to make one
stock-gainers request come out right. It did, and it also made the model reach for
finance APIs on requests that had nothing to do with finance, because a whole worked
program in a declaration does not read as guidance, it reads as what code here looks
like. What was left was 999 characters that are true of any request: prefer the standard
library for a lookup, write multi-line programs as a quoted heredoc rather than
`python3 -c`, and when a command's output already answers the question, copy it exactly
and invent nothing. The only sentences added since say what this host cannot do — `node`
is a JavaScriptCore layer rather than Node.js, and npm/npx, native executables and
child processes are unavailable — which is just as true of every request here.
Task-specific help belongs in a skill, which is injected only when it
is selected. `AgentAppHostTests` fails if any tool description names a vendor or product
again.

**Every bundled skill reaches the model's catalog.** On 2026-09-08, six of the thirteen
then bundled did not. The catalog was filled in id order under a budget of about a
thousand tokens, the thirteen descriptions came to half again as much, and the alphabet
decided who was cut: `documents` and `research` — the two this app's own router
depends on — were both below the line, while two entries of nearly a thousand
characters each at the head of the alphabet (both unbundled since) took half the
budget between them. A model asked to look something up
listed the skills it could see, found nothing that fetches a page, and refused. The
catalog now SHORTENS entries rather than dropping them, at the longest of a few fixed
lengths where everything fits, and charges for the text it actually emits; a catalog
that already fits is left exactly as it was.

**One compound request is routed, on any subject.** When the skills picker is
untouched (the request sends no `skills` field) and the latest turn has no attachments,
a message that asks for research where it is asked for, not merely named (search,
research, look up, 搜索, 检索, 查找, 调研 …; "our research results" or 搜索引擎 do not
count), and asks to create a deck (create, make, generate … / 生成, 制作 … followed
within 96 characters by pptx, PowerPoint, slides, slide deck, presentation, 幻灯片,
演示文稿 or 演示报告) is routed to the `research` and `documents` skills together. The
search uses the user's own words minus the deck request, at most 384 characters, with
each http(s) URL in them passed as its own argument; the turn is complete only when a
real `.pptx` of at least four slides exists that shows an http(s) URL and cites a
source from `notes.md`. The route needs the network, so with the Network switch off it
is refused before anything runs, with the `network_disabled` notice and a "Turn on
Network" button, and without a usable `skills_run` with `routed_workflow_unavailable`
and "Open Settings". Everything else goes through ordinary skill discovery.

**The sandbox switches take effect now.** "Run code" and "Allow network access" used to
apply "the next time TensorAgent starts", which is honest and useless: leaving an iPhone
app does not restart it, so the real instruction was "force-quit from the app switcher"
and the switch read as one that did nothing — network turned on, `curl` still answering
"network access is disabled by the user". `AgentAppHost.ApplySettings` moves all four
holders together (the runner's options, the installer's standing policy, the shell's
host list, and the terms a skill's scripts are planned against). A command already
running keeps the terms it started with.

**Skills.** Bundled and installed skill directories are discovered automatically — see
"Skills" below for platform requirements. Users can install more from a `.zip` or from
a link — one archive, or a plain-text list of links, one per line — and can remove any
skill, bundled ones included: removing a bundled skill is recorded in
`removed-skills.txt` in the backed-up data directory, so an app update does not bring it
back.

**Code, generated and run.** The agent host's shell tool works here, backed by an
in-process POSIX shell, an embedded CPython 3.13 and JavaScriptCore, because iOS
allows no child processes at all. With "Run code" on, a chat is declared `shell`,
`read_file`, `apply_patch` and `write_file` (which creates a file and refuses to
overwrite one); with skills on, `skills_list`, `skills_read` and `skills_run` as well.

A missing command never ends in "command not found" and nothing else. Installing a
native program is available to nobody here — iOS runs no child processes and will not
execute a binary that was not signed into the bundle — so the shell names what does
work instead: `$(( ))` and `python3` for `bc`, the interpreters for another language,
the fact that `apt`/`brew`/`sudo` have no meaning on this device, and that pure-Python
`none-any` wheels can be installed with `pip` when network access is on. JavaScriptCore's
`node` command is a compatibility layer; it cannot install npm packages or launch
native child processes. The shell also catches a transposed name. This is not
politeness: a dead end is where a model stops using the shell and starts inventing
the answer, which is exactly what one did on a
phone — reaching for `bc` to subtract two dates, being told 127, and finishing the
arithmetic in its head with the wrong number and a formula underneath.

**Sub-agents, on by default, with a switch.** Settings > Sandbox > "Sub-agents"
(`multiAgentEnabled` in `settings.json` and `POST /api/agent/settings`, on by default)
decides whether delegation is offered; every limit stays at the shared defaults
(`MultiAgentOptions`: at most three children running at once, eight per request, two
levels deep, 180 s per child). While it is on, every catalog model renders tool
declarations, so every chat is also offered `spawn_agent`, `wait_agent`,
`send_input`, `close_agent` and `list_agents` — even with Skills and "Run code" off.
Off, those five tools and the coordination prompt are not declared at all, exactly as
on a server started with `--no-multi-agent`. The change applies to the running app
(`ServerHostingOptions.RepointMultiAgent`) from the next message, with no restart; a
turn already delegating finishes under its old terms. Children are read-only: the app
never sets `AllowWorkerTools`, so a worker gets no mutable tools either. Children
use private workspaces with explicitly selected input files. Independent tasks
and permitted tools can overlap; declared dependencies wait for successful
prerequisites, and extra tasks queue within the shared limits. Each child starts
a fresh conversation (the parent's system instructions and its task, not the
parent's transcript) and restores
its own copy of any matching shared-prefix checkpoint rather than sharing KV pages.
The app does not read `TS_NO_MULTI_AGENT`, which only `TensorSharp.Server` reads,
and the phone page never sends `multi_agent: false`; the switch is the way to turn
delegation off. Nor does the page show a sub-agent progress panel (see above). No
latency, memory or quality measurement of delegation on a phone exists; the design and
its limits are in [Multiple agents](../docs/multi_agent.md).

**The first message is as fast as the second, and so is a new chat.** A
conversation's first turn used to forward several thousand tokens — the system prompt,
the tool schemas, the skill descriptions — before the model wrote a character: 0% KV
reuse, and twenty to forty seconds on the phone before the first token. Every later
turn reused 99% of that, so the prompt was never slow; it was paid for once, by the
user. Two things now pay it instead. As soon as the weights are in, the app forwards
that shared prompt on a throwaway one-token request (`AgentAppHost.WarmThePrefixCache`)
while the user is still reading the screen; a real message cancels it and waits for it
to be gone, so nobody ever shares the engine with it — and loses little by doing so,
because the cancelled warm-up's cache stays resident and the message continues from it
at the last chunk boundary (measured with `--delay`: letting the warm-up finish instead
was a wash on both Qwen 3.5 and Gemma 4). And the engine keeps a
**checkpoint** of the model's complete state at the end of that shared prefix — a deep
copy, kept apart from the per-conversation caches and never consumed — so every NEW
chat starts from a clone of it and re-prefills only its own message. That copy is what
makes new chats fast on the two families that could not be served any other way: Gemma
4's sliding-window layers physically hold only the last 512 positions, so the pooled
block cache could restore at most one window, and Qwen 3.5's recurrent state cannot be
rewound at all. Measured with `benchmarks/TensorAgentTtftBench` (below): on Gemma 4 E2B
a new chat went from 1.5 s / 0% reuse to 0.11 s / 99.6% on a Mac; the phone is
expected to follow the same shape, but no per-shape measurement on the phone is
recorded (see "What has not been verified"). Two more turns that used to
re-prefill everything no longer do: a turn after the user tapped Stop (the transcript
now records the tokens the engine forwarded past the last one streamed), and, on Qwen
3.5, a turn after the thinking toggle changed (the two thinking modes rendered through
different code and disagreed from the first tool declaration on; they now share one
renderer, and each answer remembers which mode its prompt ended in).

**A sandbox the user controls.** Two switches, both in Settings, both defaulting to
the safe answer: code execution on, because an agent that cannot act is not an
agent, and network off, because a model that can reach the internet from inside a
sandbox is a different risk entirely. The same section also holds the "Sub-agents"
switch (on by default; see above), and its note reads "Changes here take effect
straight away: the sandbox on the next command the model runs, sub-agents on the next
message." Every setting on that page does something; one that could not be enforced
was removed rather than left there implying it was.

The rest of Settings, with its defaults: the reply output limit (256 to 262,144 new
tokens, 2,048 by default); KV cache precision (FP16, Q8 or Q4, Q4 by default — it
overrides the catalog entry's precision and applies at the next model load, and Gemma 4,
whose attention cannot read a block-quantized cache, and Qwen3.8 Flash Next, whose engine
reads only F16/F32 K/V, use FP16 whatever it says); the
tool timeout (10 to 600 s in steps of 10, 120 by default); "Show reasoning by default"
(off); "Speculative decoding" (on, applied to the running engine from the next
reply — see "Speculative decoding" below); "Download over cellular" (off); and "Include optional files", the
projector and draft head (on). The reply limit and tool timeout apply at once, like the
sandbox switches. Four settings have no control and can be set only in `settings.json`
or through `POST /api/agent/settings`: `networkHosts`, a host allow-list for the
network switch (empty means any host); `contextLength`, an override of the catalog
entry's window (0 keeps it); `keepAwakeWhileGenerating`, which holds the display awake
while the model works (on by default); and `defaultSkills`, the skills preselected for
a new chat (none by default). The Skills master switch is in the page's Skills sheet,
and `imageLoras`, the LoRA plug-ins every picture is made with, is set from the page's
LoRA sheet (see "LoRA plug-ins" above).

**Model download and cache folder.** Settings > Storage shows the full folder path
and lets you save a different absolute path or choose **Use default**. Downloads,
imports, the model catalog and subsequent model loads use this folder, including
after a restart. Files are kept in `<folder>/<catalog-id>/`; existing files stay in
the previous folder, so move those model subfolders yourself if you want to reuse
them at the new location. A loaded model keeps running until you load another one.
Folder changes wait for model loading and are refused while a download or import
is active. The folder must be writable. `GET /api/agent/settings` reports the
effective `modelCacheDirectory`; change it through
`POST /api/agent/settings/model-cache-directory` with
`{"modelCacheDirectory":"<absolute-path>"}` (an empty string restores the default).

## Build and run

The user-local SDK is the one with the MAUI workloads:

```
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
```

One-time preparation; all three produce files that are not in git:

```
TensorSharp.GGML.Native/build-ios.sh        # GgmlOps.xcframework (device + simulator)
eng/fetch-python-ios.sh                     # CPython 3.13 for iOS
TensorAgent/scripts/prepare-python.sh       # stage the interpreter and its packages
```

`prepare-python.sh` also runs `TensorAgent/scripts/build-lxml-ios.sh` the first
time, because lxml — which python-docx and python-pptx import at module scope — is
a C extension no index publishes for iOS. That script cross-compiles libxml2,
libxslt and lxml against the embedded CPython for both slices (a few minutes;
needs a host `python3.13`, CMake and Ninja) and drops the wheels into the same
cache the BeeWare wheels come from.

`eng/fetch-python-ios.sh` fetches BeeWare's Python-Apple-support 3.13-b14
(`TENSORAGENT_PYTHON_VERSION`, `TENSORAGENT_PYTHON_BUILD`). The packages
`prepare-python.sh` stages are listed in the script: numpy 2.5.2.post1 and Pillow
10.4.0 from BeeWare's index, lxml 6.1.3 built as above, and the pure-Python pypdf,
openpyxl, et_xmlfile, reportlab, imageio, python-pptx, python-docx, XlsxWriter,
typing_extensions, charset-normalizer and defusedxml, all pinned to a version, plus
certifi and PyYAML, which are not pinned, so each run stages whatever version the
index currently has.
`TENSORAGENT_PYTHON_PACKAGES` adds more pure-Python packages;
`TENSORAGENT_PYTHON_NO_WHEELS=1` stages the standard library only.

Then:

```
TensorAgent/scripts/build-sim.sh            # simulator build
TensorAgent/scripts/run-sim.sh              # install, launch, stream stdout
TensorAgent/scripts/verify-sim.sh           # drive the running app's API from the Mac
                                            # (also takes a DEVICE log: a phone's 127.0.0.1
                                            #  is the phone's, so it skips the API half and
                                            #  checks everything the app logged about itself)
TensorAgent/scripts/build-device.sh          # device build only (SKIP_SIGNING=1: compile/link check)
TensorAgent/scripts/deploy-device.sh         # Debug by default: auto-sign, install, and launch
TensorAgent/scripts/verify-background.sh     # send the app away mid-answer and read what happened
TensorAgent/scripts/verify-share-rule.sh     # check the share extension's activation rule
TensorAgent/scripts/bench-spec-device.sh     # plain vs speculative decoding on the phone
```

`dotnet build TensorSharp.slnx` from the repository root builds the app as well. Once
the rest of the solution has built, `Directory.Solution.targets` builds the Mac app and,
on Apple silicon, the simulator app in a separate `dotnet build`, as `build-mac.sh` and
`build-sim.sh` do (on Windows, the Windows app, as `build-windows.ps1` does). The
simulator app is built for the simulator the SDK picks rather than `build-sim.sh`'s
explicit RuntimeIdentifier; it and its share extension are ad-hoc signed, where
`build-sim.sh`'s extension takes the Mac's team profile if one is installed. On a Mac,
the app build and its workload check use `DOTNET_ROOT/dotnet` when available, then
`~/.dotnet/dotnet`, matching the scripts. Otherwise they use the solution's SDK. This
lets the app use Xcode-compatible workloads even when the system SDK has older ones.
`-p:TensorAgentDotnet=/path/to/dotnet` selects another SDK explicitly; its architecture
must match the SDK running the solution. The selected SDK needs the app workloads
and must support the installed Xcode (for Xcode 27.0, workload set 10.0.401.1 supplies
the matching Apple SDKs). The simulator app also needs the files above. A missing workload
skips the app and a missing file skips that head, each with a warning; any other failure
fails the solution build. `-p:TensorSharpSkipTensorAgentApp=true` leaves the app out.
Running, deploying and verifying still go through the scripts.

`deploy-device.sh` selects the only connected physical iPhone, an installed
`Apple Development` identity, and a compatible provisioning profile. If more
than one phone or identity is available, set `DEVICE_ID` or `CODESIGN_KEY`;
`CODESIGN_PROVISION` can likewise override profile selection. The app's App ID needs
the Increased Memory Limit and Extended Virtual Addressing capabilities, which every
device build requests (`Platforms/iOS/Entitlements.plist` and `Entitlements.Share.plist`);
without them profile validation fails. Share-enabled builds
also require the App Group `group.ai.tensorsharp.tensoragent` on both App IDs and an
independent profile for `ai.tensorsharp.tensoragent.share`; override its selection with
`CODESIGN_SHARE_PROVISION`. The containing app's exact profile must never be reused
for the extension. Set `TENSORAGENT_SHARE_EXTENSION=false` only for an intentional
app-only regression build. The install is an update in place, so existing models,
conversations, and settings are retained.
`deploy-device.sh` builds Debug unless `CONFIGURATION=Release` is set, and either way
rebuilds the native iOS xcframework from the current checkout;
set `TENSORAGENT_REBUILD_XCFRAMEWORK=0` only when intentionally reusing it. Before
installing, it checks that the built executable still exports
`TSGgml_IsMetalAvailable`, a sentinel for the engine's `TSGgml_` entry points.

These environment variables drive a Debug build from a script, because neither
`simctl` nor `devicectl` can tap or type:

| | |
| --- | --- |
| `TENSORAGENT_START_PAGE=models` | open a page other than the chat |
| `TENSORAGENT_USE_MODEL=<catalog id>` | load a model, as tapping "Use" would |
| `TENSORAGENT_DEMO_PROMPT=<text>` | type a prompt into the composer and send it |
| `TENSORAGENT_UI_CHECK=1` | drive the composer's gestures and the menu, one line per check |
| `TENSORAGENT_SHARE_CHECK=1` | verify App Group import into an unsent draft, then explicit discard of its durable envelope and staged PNG |
| `TENSORAGENT_OPEN_MENU=1` | leave the menu open, so it can appear in a screenshot |
| `TENSORAGENT_NAV_CHECK=1` | leave the chat mid-answer for `TENSORAGENT_NAV_SECONDS` (15) and report whether the answer carried on |
| `TENSORAGENT_NETWORK_CHECK=1` | flip the network switch both ways and run `curl` after each, then put it back |
| `TENSORAGENT_DOWNLOAD=<catalog id>` | start a download and log it, stopping after `TENSORAGENT_DOWNLOAD_SECONDS` (60) |
| `TENSORAGENT_TTFT_CHECK=1` | four turns through `/api/chat` — first chat, follow-up, new chat, follow-up — with one `ttft` line each (also in `logs/ttft.log`) |
| `TENSORAGENT_BACKGROUND_CHECK=1` | ask the host's own model for a long answer (`TENSORAGENT_BACKGROUND_PROMPT`, `TENSORAGENT_BACKGROUND_TOKENS`, 4096) and trace what happens while the app is away; driven by `verify-background.sh` |
| `TENSORAGENT_PAGE_BACKGROUND_CHECK=1` | the same, through the page and the `TENSORAGENT_DEMO_PROMPT` it sends (`verify-background.sh` with `CHECK=page`) |
| `TENSORAGENT_SPEC_BENCH=1` | the plain-vs-speculative benchmark (`TENSORAGENT_SPEC_BENCH_MODES`, `TENSORAGENT_SPEC_BENCH_TOKENS`, 160); a Release build honours this one, and `TENSORAGENT_USE_MODEL` with it |
| `TENSORAGENT_IMAGE_BENCH=1` | the picture benchmark: `TENSORAGENT_IMAGE_BENCH_RUNS` (2) pictures of `TENSORAGENT_IMAGE_BENCH_PROMPT` through the app's own image turn, with the LoRA plug-ins turned on, one `imagebench` line each (also in `logs/imagebench.log`); a Release build honours this one too, and `TENSORAGENT_USE_MODEL` with it |
| `TENSORAGENT_SKIP_UPLOAD_CHECK=1` | skip the large-upload probe described below |

Two of those exist because the claim they check has no other witness. `TENSORAGENT_NAV_CHECK`
is the only way to see that a generation survives the chat leaving the screen: iOS suspends
a WKWebView's content process the moment its view leaves the window, and nothing off-device
reproduces that. `TENSORAGENT_NETWORK_CHECK` is the only way to see that flipping the
network switch changes what the very next command can do, in one running process — which is
the whole of the bug it guards. A large upload is posted to the app's own `/api/upload` on
every Debug launch for the same reason: the multipart parser's fault only appeared when a
single read filled its buffer, which is what iOS's HTTP client does and no test host did.

A whole round trip on a real iOS runtime is therefore scriptable: link a GGUF into
the app's model directory, launch with `TENSORAGENT_USE_MODEL` and
`TENSORAGENT_DEMO_PROMPT`, and `verify-sim.sh` reads the result out of the log.

A device build additionally needs a signing identity and provisioning profile:

```
dotnet build TensorAgent/src/TensorAgent.Maui/TensorAgent.Maui.csproj \
    -f net10.0-ios -r ios-arm64 -c Release -m:1 \
    -p:TensorSharpAppleTargets=true -p:CodesignKey="Apple Development: ..."
```

`-m:1` keeps the build on one MSBuild node: several referenced projects share one
output directory, and parallel nodes race on its `deps.json`. `build-device.sh`
passes it too, and besides `SKIP_SIGNING` reads, among others listed in its header,
`CODESIGN_KEY`, `CODESIGN_PROVISION`, `CLEAN=1` (a targeted clean first),
`NO_INCREMENTAL=1` and `TENSORAGENT_DOTNET_ARGS` (extra `dotnet build` arguments).

`TensorSharpAppleTargets=true` must be on the command line rather than only in the
csproj: it decides whether `TensorSharp.Models` builds its `net10.0-ios` and
`net10.0-maccatalyst` slices at all, and restore resolves a referenced project's
target frameworks before a `ProjectReference`'s `AdditionalProperties` are applied.
That restore goes to `obj/<host>/apple/` (`Directory.Build.props`), apart from the
desktop build's, so the two do not keep rewriting each other's `project.assets.json`
and recompiling `TensorSharp.Models`. Leave the property off and the referenced projects
fail with NETSDK1004, or quietly use the restore of an earlier build.

Release device builds keep the engine. It is linked statically and reached through
`dlsym`, and the Release build's strip step keeps only the symbols on its list, so
`GgmlExportedSymbols.targets` names each `TSGgml_` export as a `ReferenceNativeSymbol`.
A new native export has to be added there too; `TensorAgentMauiProjectTests` fails when
that list and the native export list disagree.

The app and its share extension target iOS 17.0 (`SupportedOSPlatformVersion` in
their project files), matching the xcframework's `TENSORSHARP_IOS_DEPLOYMENT_TARGET`
(default 17.0) in `build-ios.sh`. The app uses the UIKit scene
lifecycle with a single window (`UIApplicationSceneManifest` in `Info.plist` and
`Platforms/iOS/SceneDelegate.cs`), because a build linked against the iOS 27 SDK crashed
at launch without it. The device run recorded below is on iOS 26.6.1; no run on iOS 27
is recorded.

## On the desktop: macOS and Windows

The same project builds the desktop app. On a Mac it is the Mac Catalyst head
(`net10.0-maccatalyst`): the phone's code, running as a Mac app. On Windows it is the
WinUI head (`net10.0-windows10.0.19041.0`), which only a Windows machine builds. All
three serve the same page from the same loopback host, and share the catalog, the
settings, the conversations and the skills. What differs is everything the phone does
because it is a phone:

<p align="center"><img src="../website/assets/screenshots/tensoragent-mac.png" alt="TensorAgent on a Mac: a saved Qwen-Image 2.1 edit changes the TensorSharp banner background to a starry blue night sky, with Compare original and Edit again controls" width="880"></p>

<sub>A saved Qwen-Image 2.1 (Q4_K_M) edit in the current Mac Catalyst Release app,
captured on 2026-10-02 on an M5 Pro with GGML Metal: “Make the background a deep blue
night sky with stars, keep the text unchanged.” The result is from an earlier run;
Compare original and Edit again remain available. Switch models for text, multimodal
questions, code and document work, or short video with audio.</sub>

| | iPhone and iPad | Mac | Windows |
| --- | --- | --- | --- |
| Engine | `GgmlOps.xcframework`, linked statically | `libGgmlOps.dylib` from `build-macos.sh`, in `Contents/MonoBundle` | `GgmlOps.dll` from `build-windows.ps1`, beside the executable |
| Backends offered | Metal (CPU in the simulator) | Metal, then CPU | CUDA or Vulkan when the engine has it and the machine can run it, then CPU |
| Code execution | in-process shell, embedded CPython 3.13, JavaScriptCore | real `bash`, `python3`, `node` and `npm` processes, each confined by Seatbelt to the chat's folder | real processes, which Windows cannot confine; offered only after **Run without a sandbox** is turned on in Settings |
| Skills | the ten `verdicts.json` passes | all twelve | all twelve |
| Engine budget | measured against jetsam (`EngineMemoryPolicy`) | the engine's defaults | the engine's defaults |
| First-launch settings | K/V cache Q4, 2,048-token replies, 120 s per command | K/V cache Q8, 8,192-token replies, 300 s (`AppSettings.DesktopDefaults`) | as the Mac |
| Leaving the screen | the GPU is handed back and the turn waits | the turn carries on; App Nap is held off while the model works | the turn carries on; sleep and power throttling are held off |
| Files | `Library/Application Support`, `Library/Caches` | `~/Library/Application Support/TensorAgent`, `~/Library/Caches/TensorAgent` | `%LOCALAPPDATA%\TensorAgent\Data`, `...\Cache` |

`DeviceClass.Desktop` on `AgentPaths` is the one switch behind the budget and the
first-launch settings; every other host, the validation launcher and the benchmarks
included, stays on `DeviceClass.Phone` unless it asks.

### On a Mac

The Mac head needs the `maui-maccatalyst` workload in the same user-local SDK, and
`maui-ios` beside it, because the project restores both Apple target frameworks
(`dotnet workload install maui-maccatalyst maui-ios`), plus CMake and the Xcode
command-line tools.
Measured here with workload set 10.0.401.1 (MAUI 10.0.110, Mac Catalyst SDK 27.0.10722)
and Xcode 27.0. Then:

```
TensorAgent/scripts/build-mac.sh     # CONFIGURATION=Release for an LLVM build (about three minutes)
TensorAgent/scripts/run-mac.sh       # launch from this terminal; stdout also goes to artifacts/tensoragent-mac/app.log
TensorAgent/scripts/verify-sim.sh artifacts/tensoragent-mac/app.log   # recognises the Mac app by its engine line
TensorAgent/scripts/chat-e2e.py artifacts/tensoragent-mac/app.log     # answers, tools, image and audio, with TTFT and decode rate;
                                                                      # pictures and clips with --scenarios (below)
```

The Debug hooks in the table above work the same way: `run-mac.sh` passes the
environment straight through. A model is installed as on the phone, from the Models page,
or by placing the catalog's files under `~/Library/Caches/TensorAgent/models/<id>/`.

With an image or video model loaded, `chat-e2e.py` checks what the model makes instead:
`--scenarios draw,edit` with Qwen-Image 2.1, `film,animate` with MiniMax-H3 (from words,
and from the attached photo as the first frame) and `reference` with MiniMax-H3
References (the same photo as the subject of a new scene). A picture must stream its
steps and end as one PNG of the size it reported; a clip must stream its stages in order
and end as one MP4 the app serves with Range and HEAD, whose video track has the frame
count, rate and size the turn reported and whose sound, inside the MP4 or in a WAV beside
it, is 32 kHz stereo as long as the clip. Either must be in the saved chat.
`--loras id[:strength],...` turns LoRA plug-ins on for `draw` and `edit` through the app's
own route and puts the previous choice back afterwards; each picture must then name the
plug-ins it was drawn with (an edit-only one is not applied to a picture made from words)
and, with a speed plug-in, run its step count. `--draw-prompt`, `--edit-prompt` and
`--edit-photo` replace the default request and photo, for a plug-in that needs its own
phrase or a photo with red boxes on it. An unknown scenario name is an error rather than
a run of nothing that reports success.
`scripts/chat-e2e-selftest.py` runs the clip checks against files it makes itself, with
no app and no model, and reports what this machine cannot run (no cv2, no `afconvert`) as
skipped, never as passed.

- **The engine library is built with the app.** The referenced projects skip their native
  builds for every head, so `TensorAgentBuildDesktopEngine` runs the backend project's own
  incremental `build-macos.sh` before the app is compiled; an up-to-date library costs
  nothing, and a library older than the native sources beside it is never shipped.
- **No App Sandbox.** The app runs the model's code as real processes and confines each
  with TensorSharp's Seatbelt profile, which macOS will not apply inside the App Sandbox.
  The build is for distribution outside the Mac App Store.
  `Platforms/MacCatalyst/Entitlements.plist` supplies the runtime entitlements for
  Developer ID distribution while keeping App Sandbox off. Local Debug/Release
  builds use ad hoc signing unless you supply a signing identity; release packaging
  requires Developer ID signing and notarization as described above.
- **Debug and Release are two apps with one set of data.** `bin/Debug/.../TensorAgent.app`
  and `bin/Release/.../TensorAgent.app` share `~/Library/Application Support/TensorAgent`
  and `~/Library/Caches/TensorAgent`, but each offers only the catalog it was compiled
  with, so rebuild the one you open after pulling (`CONFIGURATION=Release build-mac.sh`
  for the Release one). A launch reclaims the models of entries the catalog has retired
  (`ModelCatalog.Retired`) and keeps a folder whose id it does not know, which a newer
  build may have installed; for the same reason it keeps, without loading it, a selected
  model it does not know.
- **The oldest Mac a source build runs on** is decided by the engine library, which `build-macos.sh`
  builds for the building Mac's own macOS unless `MACOSX_DEPLOYMENT_TARGET` says otherwise,
  not by the app's `SupportedOSPlatformVersion` (Mac Catalyst 17.0, macOS 14).
  The Desktop release workflow sets `MACOSX_DEPLOYMENT_TARGET=14.0` for its Apple
  Silicon packages; this build target does not establish runtime testing on every
  supported macOS version.
- **PATH.** An app started from the Finder or the Dock gets launchd's
  `/usr/bin:/bin:/usr/sbin:/sbin`, where Homebrew's `node`, `npm` and `python3.13` are not.
  At startup the app asks the login shell for its PATH and puts it first
  (`DesktopEnvironment`).
- **Mono, not CoreCLR.** .NET ships no CoreCLR for Mac Catalyst, so the app's managed code
  runs on Mono, as the phone's does. Release builds therefore use LLVM, which Mac Catalyst
  does not get by default (the csproj's `MtouchUseLlvm` explains the measurement).

### On Windows

Build on a Windows machine with the `maui-windows` workload, and CUDA or Vulkan tooling
if the engine should have them (`TensorSharp.GGML.Native/build-windows.ps1` reads
`TENSORSHARP_GGML_NATIVE_ENABLE_CUDA` / `_VULKAN` as the desktop hosts do):

```
powershell -NoProfile -ExecutionPolicy Bypass -File TensorAgent\scripts\build-windows.ps1 -Configuration Release -Run
```

The script stops if the build fails, verifies that the bundled page and embedded
image editor match the current source, and then runs the app. Without `-Run`, it
prints the verified executable path:
`TensorAgent\src\TensorAgent.Maui\bin\Release\net10.0-windows10.0.19041.0\win-x64\TensorAgent.Maui.exe`.
Close a running instance before rebuilding. Building `TensorSharp.Server.Host`
updates the server, while TensorAgent needs its own build to update the embedded
UI in `TensorAgent.Core.dll` beside its executable.
For environments that use NuGet mirrors, `-PackageSource` accepts one or more
feed URLs for restore; the default uses the repository's NuGet configuration.

If restore reports `NU1301` with a TLS `HandshakeFailure` for `api.nuget.org`,
the installed workload is not the problem: NuGet cannot download packages such
as the Windows runtime packs. An optional configuration uses Microsoft's
`dotnet-public` feed and keeps vulnerability auditing enabled through NuGet's
separate `data.nuget.org` endpoint:

```powershell
Copy-Item eng/NuGet.dotnet-public.config NuGet.Config
dotnet build TensorSharp.slnx -c Release
```

Run these commands from the repository root. Keep this machine-specific
`NuGet.Config` local (add `/NuGet.Config` to `.git/info/exclude`); remove it to
return to your usual feeds. The mirror does not carry every third-party package
(including NLayer and NVorbis), so those packages need cached copies or another
reachable feed. Missing packages remain build errors. For a one-off override,
use `-p:RestoreConfigFile=<absolute-path-to-config>` on the solution build; its
separate TensorAgent build receives the same restore settings.

The app is unpackaged and carries the Windows App SDK runtime with it
(`WindowsPackageType=None`, `WindowsAppSDKSelfContained`). The model's code runs only
after **Run without a sandbox** is turned on: a job object bounds a process tree but
confines neither its files nor its network, so the switch is the same explicit choice as
the server's `--code-exec-unconfined`. The app does not dictate on Windows; Windows' own
voice typing (Windows+H) writes into the message box.

**Windows validation (2026-10-01).** Debug and Release built and ran on Windows x64
with .NET SDK 10.0.204, an i7-11800H, 32 GiB RAM and an RTX 3080 Laptop GPU (16 GiB).
The managed suite passed 895 tests, with 177 explicit skips; 21 MAUI project checks
and 12 focused native CPU/CUDA checks passed. The upstream ggml checkout was unchanged
at `353b63b439f27ab2cc19dac97ab1681ba6d2d084`.

The actual Debug app passed 12 WebView checks, a 1.55 MB upload, navigation away from
a streaming answer and back, and transcript recovery after a forced process restart.
The actual Release app passed 15 HTTP/SSE turns covering chat, saved conversations,
cancellation and recovery, PowerShell, Python-generated downloadable files, and a
synthetic vision question. With local **Gemma 4 12B QAT UD-Q4_K_XL** and its BF16
projector, CUDA, 8,192 context tokens and speculation disabled, three 128-token samples
gave **37.1–38.6 tok/s** (median **38.2**); warm first-token latency was **0.27–0.49 s**,
and the first chat took **3.78 s** to its first token. This quantization differs from
the built-in catalog entry. Peak process working set was **18.2 GB**, with **24.7 GB**
private bytes at the end: these measurements do not establish low-memory suitability.

To repeat with local weights, build the desired configuration, then run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File eng/validation/run-tensoragent-windows.ps1 -Configuration Release -Root C:/Works/TensorSharp/docs/validation/my-windows-run -Weights C:/models/model.gguf -Projector C:/models/mmproj.gguf -Tools
```

Use a fresh evidence directory and omit `-Projector` for text-only testing. `-Tools`
temporarily enables unconfined execution in this isolated test installation and restores
the setting afterwards. Debug additionally runs the WebView and navigation probes.
`TENSORAGENT_VALIDATION_ROOT`, `TENSORAGENT_VALIDATION_WEIGHTS` and
`TENSORAGENT_VALIDATION_MMPROJ` opt the app into isolated local
validation; ordinary launches use their normal data and catalog. Evidence is kept in
ignored `docs/validation/` or `artifacts/`.

This is bounded validation, not full coverage: manual visual review, physical camera
and native file dialogs, live catalog downloads, audio/video generation, other models
and GPU backends, and a long-duration soak were not verified. The picker exception
handler was compiled but its device/permission failure path was not exercised on hardware.

### Measured on a Mac

M5 Pro, 51.5 GB, macOS 27, Gemma 4 E2B Q8_0 on `ggml_metal`, the in-process
`SpeculationBench` (`TENSORAGENT_SPEC_BENCH=1`), second pass, medians. The CoreCLR column
is the same host code in `benchmarks/TensorAgentTtftBench --desktop --scenarios specbench`,
which is what the desktop server's runtime gives it; the Mac column is the Release app.

| Turn | CoreCLR first token | CoreCLR tok/s | Mac app first token | Mac app tok/s |
| --- | --- | --- | --- | --- |
| one word | 87 ms | | 150 ms | |
| prose, 160 tokens | 81 ms | 77.1 | 144 ms | 74.8 |
| quote the prompt, 219 tokens | 151 ms | 76.5 | 216 ms | 74.3 |
| quote its own answer | 81 ms | 76.3 | 144 ms | 74.2 |
| the same, speculative | 95 ms | 301.5 | 160 ms | 301.1 |

Decode is within 3% of CoreCLR and speculative decoding reaches the same rate. What Mono
costs is about 60 ms of managed work per turn before the first token. Without LLVM the Mac
app's steady-state decode was 72.5 tok/s and the first token 20-30% later.

**App Nap.** With the screen locked, five of eight runs of the Mac app had stretches of
47-58 tok/s where the same code on CoreCLR, in a terminal that macOS never naps, never
dropped below 73. The app now holds a user-initiated activity while a turn runs or the
engine has work (`DesktopActivity`); four runs of four afterwards held 74-75 tok/s, and
`pmset -g assertions` shows "TensorAgent is generating a reply" while it does.

`chat-e2e.py` against the Mac app with the same model: a question, a follow-up at 99.7%
cache reuse, a new chat from the shared-prefix checkpoint, a 398-token story, a thinking
turn, a shell command whose output only a real run can know, and the title on an image
all pass. An audio clip does not, on the phone, the Mac or the desktop server alike: with
the agent's tools declared (the shell, or the sub-agent tools alone), Gemma 4 E2B and E4B
answer that they cannot hear it, or reach for `whisper` and `ffmpeg` through the shell. The
clip does reach the model: the audio embeddings the chat injects are identical to the ones
`EngineParallelInferenceTests.Gemma4_AudioPrompt` transcribes, and the same request is
transcribed with no tools, with a single unrelated tool, or after 1,600 tokens of other
conversation. Telling the model in the message that the recording is attached did not
change the answer (0 of 3 for each wording tried), so this is recorded as a model
limitation rather than prompted around: turn off **Run code** and **Sub-agents** for a
turn that is about an audio clip.

The browser workflow also runs in the Mac app: with Qwen3.5 9B and the network switch on,
`eng/validation/validate-browser-skill.py --connection` (a file holding the app's
`tensoragent_token` cookie) had the Playwright skill drive a real headless Chrome through
a local form, submit the value it read off the page, and return a screenshot of the
result, in 66 s.

<a id="the-macs-own-models"></a>

### The desktop's larger models

These entries are offered by system RAM tier on Mac and Windows. The measurements
below are historical Mac app results; the separate IQ1_M Windows catalog-host
validation above does not validate that variant on Mac or compare its performance
to these runs.

Measured on the same M5 Pro (48 GB) on 2026-09-30, `ggml_metal`, the Debug app, with each
entry's files downloaded and verified by the app (Qwen3.8 27B's by `curl`, checked against
the same pins):

| | Qwen3.8 27B (UD-Q4_K_XL) | Muse-Glimmer 30B (UD-Q4_K_XL) |
| --- | --- | --- |
| `chat-e2e.py`: fact, follow-up, new chat, story, thinking, shell, image | 7 of 7 | 7 of 7 |
| plain decode in the app, ~7k-token prompts (the CLI's greedy benchmark at 6,800) | 14.0-14.1 tok/s (14.8) | 15.9-16.0 tok/s (17.2) |
| speculative decoding in the app | n-gram: prose 0.93x, quoting 1.76-1.90x | declined by the engine on Metal |
| prompt reuse, follow-up / new chat | 99.7% / 99.6% | 99.8% / 66% |
| app footprint at its highest, beside the weights | 8.8 GB beside 17.6 GB | 10.9 GB beside 15.9 GB |

The 5-7% plain-decode gap to the CLI was not broken down further; the two differ in
more than the runtime (the app samples with the card's temperature, top-k and top-p and
streams through the chat pipeline, where the CLI benchmark takes the argmax). Three
things were changed for these two models:

- **The phone's cache budget** (`LeanCaches` on both entries). With the engine's desktop
  defaults the app grew to 18.8 GB beside Qwen3.8 27B's weights over those seven turns,
  and the machine fell to 0.1 GB free with 10.6 GB compressed; the phone's budget held it
  at 8.8 GB with the same reuse. The tiers are the weights, that footprint and about 5 GB
  for macOS, so 32 GB rather than 24.
- **Only a catalog draft head speculates.** Qwen3.8 27B's GGUF carries its own NextN/MTP
  layer, which the engine attaches; with it the app decoded slower in every turn shape
  (prose 0.83x, quoting 0.89-0.93x), because a four-token verify of the dense trunk costs
  about twice a plain token on Metal. n-gram takes its place (the table).
- **Muse-Glimmer reasons before it answers, even with thinking off**, and sometimes long:
  asked for a 250-word story it wrote the story to a file and counted its words with
  `wc -w` before answering. Its q8_0 cache stayed clean through a 1,542-word answer that
  grew the full-attention layers past 8,192 rows mid-reply. A new chat reuses only the
  first 4,352 tokens of the shared prompt, the size of its sliding-window ring (a longer
  saved prefix cannot be restored into the ring), so its first token takes about 7 s.
  Its vision tower takes 16 s on a 1260x840 image in the CLI and 24 s in the app.

Qwen-Image 2.1, the same day: a 1024x1024 picture at 40 steps took 324.5 s in the app
against 318.7 s for the CLI on the same files (7.84 and 7.83 s a step; the rest is the
eight previews the page shows), and the two pictures are pixel-identical. An edit of a
1253x836 photo at 1248x832 took 412 s against 399 s. Two engine fixes made that so; before
them the edit took 495 s in the app:

- **The VAE encoder runs fused on Metal.** Its average-down step front-pads time, which
  upstream ggml-metal cannot do, and that one node had refused the whole encoder graph, so
  every Metal edit encoded its photo per convolution. The leading zeros are now a concat
  where the pad is refused: 3.6 to 1.9 s in the CLI and 18 to 1.6 s in the app, with the
  edited picture 63 dB from the old one.
- **Weights are not transposed on the host.** The vision encoder's linear layers built a
  transposed copy of every weight on first use through a one-float-at-a-time loop (and
  the native multiply packed it back on every call). They now pass the weight as a
  transposed view, bit-identical, and that copy is tiled for the encoders that still make
  one: the reference photo's encode went from 7.3 to 3.7 s in the CLI and from 77 to 13 s
  in the app, and Gemma 4 E2B's first audio turn from 57 to 9 s.

MiniMax-H3, on 2026-09-30 and 2026-10-01 (ggml `353b63b4`, unmodified): the Debug app
(Mono) against the CLI (CoreCLR), 22 frames at 20 steps.

| | App | CLI |
| --- | --- | --- |
| from words, 640x384 | 150.0-152.9 s over three clips (6.84-6.86 s a step) | 147.7 s (~6.5 s a step) |
| from a photo, 608x416 | 216.5-223 s (9.5 s a step) | 209.3 s |
| around one reference photo, 640x384 | 199-210 s (9.2 s a step) | not run |

The app's footprint peaked at 2.24 GB on a photo turn (1.05 GB from words alone), and the
machine's wired memory at about 20 GB: the denoiser and the video VAE during the decode,
on top of the ~3.5 GB the system wires anyway. The largest file a stage maps is the
18.22 GB text encoder, so the tier is 18.22 + 2.24 + 5 GB for macOS = 25.5 GB: 32 GB
rather than 24 (`CatalogTests.EachDesktopEntryFitsItsTierBesideMacOS`). That fits only
because the engine now hands the denoiser and both VAEs back before the next clip's text
encoder runs; it used to keep all three device-resident between clips. Measured on a
quiet machine, two identical clips from words per arm: peak wired memory 19.1 GB by
default against 33.3 GB with `TS_H3_KEEP_RESIDENT=1` (the old behaviour, reached in the
second clip's text phase), for 1.3 s more text conditioning on the second clip (2.8 s
against 1.5 s). The clips took 151.4 and 148.9 s against 148.4 and 147.4 s, about 1%
apart and inside the 3 s by which the identical first clips differed between the arms.
The soundtrack is byte-identical to the CLI's for the same seed; the H.264 frames are
39.3 dB PSNR at the worst and 39.9 dB on average from the CLI's lossless PNG frames, and
the AAC track correlates 0.9986 / 0.9988 (left / right) with the WAV. A `WKWebView`
plays the clip from the app's loopback server: readyState 4, the full duration, one video
and one audio track (`eng/validation/webview-media-check.swift`).

Fixed along the way, besides the hard links in "Downloads" below:

- **Quitting no longer aborts.** Every quit after any GPU work aborted on
  `GGML_ASSERT([rsets->data count] == 0)` in ggml-metal's static destructor: Mono raises
  ProcessExit only for a managed shutdown, and AppKit's quit ends in `exit()`. The Mac
  app's `WillTerminate` now stops the turns, waits, releases the model and frees the
  engine. Catalyst gives that callback about 5 s before its watchdog calls `exit()`, and
  one step of a clip is about 7 s, so a quit mid-clip waits 3 s and then leaves with
  `_exit`, without destructors, rather than release the model under the GPU. Switching
  models during a picture or a clip now waits for it too, for up to 5 minutes.
- **The soundtrack is inside the MP4,** as AAC (two channels, 32 kHz, 128 kbit/s), index
  first, with the 32 kHz WAV still written beside it. A first version handed AVFoundation
  a managed array that the encoder read after its wrapper was disposed, and the track
  carried a ~5 kHz whine at a fifth of the level, different on every run; the memory is
  now CoreMedia's. The media probe's `mp4-soundtrack` check decodes a tone back.
- **A transparent PNG reaches the model as stored.** Core Graphics decoded it
  premultiplied, which keeps none of a pixel's colour where alpha is 0, and a mostly
  transparent picture reached MiniMax-H3 as black blotches (27% near-black pixels in the
  first frame, against 0.1% from the CLI). Such a PNG, if it embeds no colour profile, is
  now read by the managed PNG codec; media-probe check `png-transparent-colour`.
- **Companions follow every load**, keyed by family, not just the one at startup:
  switching between Qwen-Image and MiniMax-H3 left the other's paths, and MiniMax-H3's
  fallback scan of the model store would have taken Qwen-Image's 8B text encoder for its
  own. A diffusion entry loads only when it is completely downloaded, a video entry is
  never tried on the CPU backend (a clip that takes minutes on the GPU would take hours),
  and the prefix-cache warm-up skips video models.
- **Files answer Range and HEAD.** The loopback server answered every request with 200
  and the whole file, and Apple's Safari Web Content Guide requires a server that hosts
  media for iOS to support byte-range requests. It now answers 206 or 416 with
  Accept-Ranges, and HEAD as RFC 9110 says (a ranged HEAD as an unranged GET, as ASP.NET
  does). WebKit on macOS played the short clip without ranges as well, so what this buys
  (seeking, and iOS's media loader) has not been observed in a run.

Qwen3.8 Flash Next (UD-Q2_K_XL), on 2026-10-01 (ggml `353b63b`, unmodified): the Debug app, the three
shards downloaded, linked into its store and verified, 78.9 GB of weights on a Mac with
51.5 GB of RAM. The engine planned the placement at load: the first 33 layers' routed
experts (31.7 GB) run on the host straight from the file mapping, the GPU holds the other
15 layers' (14.4 GB), and of the 28.8 GB n-gram table only the 16 rows a token needs are
ever read (see the [model card](../docs/models/qwen38-flash-next.md#larger-than-memory)).

| | Qwen3.8 Flash Next (UD-Q2_K_XL) |
| --- | --- |
| `chat-e2e.py`: fact, follow-up, new chat, story, thinking, shell | 5 of 6 (thinking: see below) |
| plain decode in the app, a 7.2k-token prompt (the CLI, greedy, a 1.8k-token prompt) | 9.9 tok/s (12.8-12.9) |
| prompt reuse, follow-up / new chat / story / shell | 99.7% / 99.65% / 99.5% / 98.7% |
| first token, follow-up / new chat / shell | 3.1 s / 3.0 s / 6.9 s |
| warm-up after a load (the 7.2k-token shared prompt) | 85.2 s |
| app footprint at its highest, beside the weights | 7.31 GB beside 18.3 GB resident |

- **The thinking scenario failed on the model's own choice.** Asked with thinking on whether
  91 is prime, it answered "No. 91 = 7 × 13" and wrote no reasoning first. Given harder
  questions it reasoned in 3 of 3 tries.
- **The other thinking mode starts cold.** The warm-up follows the "Show reasoning by
  default" setting, and this template puts its thinking instructions near the top of the
  system turn, so the two modes share no prefix: the first message in the other mode read
  its whole 7,243-token prompt, 108 s. That mode is cached from then on.
- **It depends on the page cache.** In the CLI the same 1.8k-token run decoded at
  12.8-12.9 tok/s, and at 10.3-11.3 tok/s with the machine at 174 MB free and 5 GB
  compressed: experts the page cache has dropped come back from the SSD.
- **The tier.** 18.34 GB of resident weights, the 7.31 GB footprint and 5 GB for macOS make
  30.7 GB, and the 17.3 GB left must cache at least a third of the 31.7 GB of experts read
  from the SSD (`CatalogTests.EachDesktopEntryFitsItsTierBesideMacOS`). At 32 GB the
  30.7 GB would leave too little for that cache, so the Q2 catalog entry retains its
  48 GB minimum system RAM tier.

## Layout

```
TensorAgent/
  scripts/          build, run and verify; prepare-python.sh; verify-skills.py
  skills/           the twelve skills (ten go into the app bundle), plus verdicts.json saying which can run on iOS and why
  python-runtime/   staged CPython (not in git; produced by prepare-python.sh)
  src/TensorAgent.Core/
    Catalog/        the model list, the store, install state
    Downloads/      resumable, verified downloads, and the manager that owns them
    Sessions/       conversations, and the recorder that keeps them in step with the page
    Settings/       the two sandbox switches and the rest
    Hosting/        the loopback server, the route table, and AgentAppHost
    Interop/        the one DllImport resolver CPython and JavaScriptCore share, and hard links
    Shell/          the in-process POSIX shell and the agent host's backends (in-process, desktop)
    Sandbox/        ExecutionPolicy and ConfinedPaths, shared by all three runtimes
    Python/         embedded CPython and the wheel installer
    JavaScript/     JavaScriptCore over its C API, with Node-shaped globals
    WebUi/          the script appended to the app's own page, and its strings runtime (i18n.js)
    Localization/   the interface's string tables, one folder per language, and Loc
    Sharing/        durable-inbox import, bounded composer handoff, ACK lifecycle
  src/TensorAgent.Sharing/
                    dependency-free envelope format and prompt composition contract;
                    the languages, the string engine and the share extension's tables
  src/TensorAgent.ShareExtension/
                    iOS share sheet, NSItemProvider readers, Safari preprocessing
  src/TensorAgent.Maui/
    MainPage        the WebView, the attachment row, dictation
    Pages/          models, chats, settings, about
    Hosting/        where the files live on this device; the engine and media probes
    Services/Apple/ device memory, Quick Look and dictation, for iOS and the Mac app
    Platforms/iOS/  app and scene lifecycle, background downloads and generation,
                    loopback probe, share inbox
    Platforms/MacCatalyst/
                    the Mac app's delegates and Info.plist (no App Sandbox; see below)
    Platforms/Windows/
                    the WinUI entry point, and Windows' device memory, file opening
                    and (absent) dictation
  tests/TensorAgent.Tests/
```

`AgentAppHost` is deliberately in the platform-neutral project. An iOS app cannot
be unit-tested from a terminal, so the wiring is only ever checked if it can be
started, driven over real HTTP and torn down on a development machine.

## How it differs from the desktop, and why

**No ASP.NET Core.** There is no iOS runtime pack for it, so the transport is
`System.Net.HttpListener` and the routes are bound by hand. The payloads and the
event-stream framing are byte-compatible with the Server's.

**One interpreter, started once, and everyone waits for it.** CPython is a
process-wide singleton here, and `EmbeddedPython` starts it on first use. Publishing
"already tried" before the work rather than after it made the fast path a window into a
half-started interpreter — and the app opens that window on every launch, because the
page fetches `/api/agent/engine` as it loads (which asks for the version, which starts
CPython) while the self-test runs `python3` on another thread. The visible cost was a
model being told "no Python interpreter is embedded in this build" by a build that has
one, on the first command of a session, after which it stops reaching for the shell.

**No child processes.** `Process.Start` is unsupported on iOS, so the agent host's
`IShellBackend` seam is filled by an in-process interpreter. Confinement moves from
the kernel to the runtimes: every path goes through `ConfinedPaths`, every network
call consults the policy first. The backend reports that honestly, including the
one thing it genuinely cannot do — preempt a builtin already inside a long call.

**Native execution when hosted on desktop.** `AgentPaths.ExecutionMode` defaults to
`Auto`: an `AgentAppHost` running on macOS, Linux or Windows uses `DesktopShellBackend`
over TensorSharp's shared `ProcessShellBackend`, OS sandbox, shell sessions and package
installer. Real Node.js, npm/npx, Python and other programs on the host can then run
inside the session workspace. On macOS and Linux the process backend requires an
available sandbox; Windows requires the explicit **Run without a sandbox** setting.
It never silently falls back to unrestricted execution. Set `AgentExecutionMode.InProcess` when a desktop test is intended to emulate
the iOS runtimes. Embedded runtime injection is not supported in process mode.

The Playwright skill (`skills/playwright`, which drives a browser with `playwright-cli`
from `@playwright/cli`, run through `npx`) requires actual Node.js/npm and browser child
processes. These capabilities are available on the desktop execution path when their
dependencies are present, but cannot be supplied to the iOS app by widening its sandbox.
The full model-driven browser workflow is recorded on macOS arm64. Windows supports
the native JavaScript skill runner with explicit unconfined execution; the scripted
`WindowsBrowserSkillProbe` checks a local synthetic login form, window state, session
reuse and cleanup. Those checks do not establish a model-driven workflow or physical
user visibility. On Linux each command runs in its own PID namespace, so a detached
`playwright-cli` session cannot be assumed to survive between calls. See the
[Windows setup and validation scope](../docs/playwright_agent.md#windows-desktop).
The skill is left out of the iOS app bundle (see "Skills" below), because the app can
start neither a Node package manager nor a browser; it stays in `TensorAgent/skills` for
the desktop hosts. Its requirements
and desktop setup are in
[Running browser and native-runtime skills](../docs/playwright_agent.md#tensoragent).
No browser-specific bridge is used. A nonempty `networkHosts` restriction is enforced by
the embedded backend; the desktop backend refuses network-enabled launches under that
restriction because its general process sandbox cannot enforce DNS host allow-lists.

For HTTP/WebUI validation of the desktop host, use the reusable launcher:

```sh
dotnet run --project eng/validation/TensorAgentHost -- \
  --root artifacts/tensoragent-browser --skills TensorAgent/skills \
  --weights /path/to/model.gguf --network --port 5038
```

`--backend` defaults to `ggml_metal` on macOS and `ggml_cpu` elsewhere, `--context` to
32768 and `--max-tokens` to 4096, `--port 0` picks a free port, and `--web-root` serves
another copy of the phone page (by default the checkout's
`TensorAgent/src/TensorAgent.Maui/wwwroot`). The launcher turns speculative decoding off.
It writes its loopback URL and authentication cookie to
`artifacts/tensoragent-browser/connection.json`. The browser-workflow validator
(`eng/validation/validate-browser-skill.py`) accepts that file via `--connection`.
Desktop results do not establish browser support on iOS.

**The transcript is the host's, and it carries the attachments.** The Web UI keeps
its history in the page and nowhere else; on a phone the app is suspended and killed
constantly, so it is written on this side instead. What that costs is that the page
has to send everything worth keeping, and for a while it did not: a message's file
paths went to the model and nothing about them went to the transcript, so a chat with
a photo in it reopened as the words with a blank where the picture had been. The page
now sends an `attachments` array — the stored name, the name the user knows it by,
what kind of thing it is — and every URL in a saved chat is derived from that stored
name rather than remembered, so a transcript cannot point at an address that has
moved. The same array is what the host stages into the working directory of anything
the model runs, which is how "make a PDF of this photo" became a thing that works:
before it, only TEXT uploads were staged, so the model could see the picture and had
no file to open.

**Weights are not backed up.** Models go under `Library/Caches`, excluded from
iCloud; conversations, settings and installed skills go under
`Library/Application Support`, which is backed up. A five-gigabyte byte-identical
copy of a public file has no business in a user's iCloud quota. The app's own folder
in Files and Finder (its `Documents` directory) stays empty, because nothing is read
from or written to it: a model for a sideload-only catalog entry (the catalog supports
them, though no built-in entry is one) comes in through the Files picker on the
Models page, which copies it into the store, and a generated file goes out through
QuickLook or the share sheet.

**Downloads outlive the screen that started them.** `ModelDownloadManager` owns every
transfer for the life of the app: the model list attaches to a running job when it
opens and detaches when it closes, `POST /api/agent/catalog/{id}/download` is a window
on the job rather than its owner, and only `…/download/cancel` stops one. Leaving for
the chat, opening any other page, or dropping the progress stream now costs nothing.

Leaving the APP is the part iOS decides. A background-task assertion is held while
bytes are moving, which buys a while rather than an exemption — long enough to glance
at a message, not long enough for five gigabytes. What makes that survivable is that
nothing is ever lost: every file is written through its `.part`, so a transfer the
system does eventually stop resumes from the byte it reached, and the app restarts it
by itself when it comes back to the foreground. The user never taps twice.

A file that another installed entry already holds byte for byte (the same pinned size and
SHA-256) is hard-linked from there rather than downloaded, before the first transfer
starts, and the Models page counts only what will be fetched: the two MiniMax-H3 entries
share six files (24.0 GB), so whichever is installed second fetches only its 11.4 GB
denoiser; measured, its six shared files arrived in 0.1 s with 0 MB transferred and no
more disk used. Deleting either entry removes only its own names for those files.

An entry published as several shards of one model (Qwen3.8 Flash Next's three gguf-split
files) counts as installed only when every shard is complete. The app refuses to load one
with a shard missing or short ("Qwen3.8 Flash Next is not completely downloaded yet.")
rather than hand the engine a first shard whose neighbours are not there, and a launch
does not restore such an entry as the selected model.

**Leaving the APP mid-answer no longer costs the answer.** Leaving the chat is
`ChatTurnManager`'s problem; leaving TensorAgent altogether is a different problem with a
harder rule behind it. iOS does not let an app that is not frontmost submit work to the
GPU — there is no entitlement for it and no background mode that grants it on an iPhone
— and ggml-metal's reaction to a refused command buffer is not to retry but to latch:
`ggml_metal_synchronize` reports `command buffer 0 failed with status 5 | error:
Insufficient Permission (to submit GPU work from background)`, sets a sticky `has_error`,
and every `graph_compute` after it returns `GGML_STATUS_FAILED` "until the backend is
recreated". One badly-timed submission therefore did not cost a token. It cost the model
for the rest of the process, so the answer died AND every message after it, until the app
was force-quit. Holding a background-task assertion made it worse rather than better: it
guaranteed thirty seconds of submissions the GPU was never going to accept.

Three things now stand between the user and that. A `ComputeGate`
(`TensorSharp.Runtime.Scheduling`) is closed on `willResignActive` — several hundred
milliseconds before `didEnterBackground`, which is the difference between stopping in
time and not — and opened on `didBecomeActive`. The **engine's own step loop** parks on
it between two steps (`InferenceEngine.ComputeGate`), which is what actually stops the
GPU: the engine decodes on its own thread into an unbounded channel, so a page or a
wrapper that merely stops reading stops nothing. The host's stream wrapper waits on the
same gate before every pull, so no new request — and no cache warm-up — is submitted
from the background either. The turn does not fail, it pauses, and carries on from the
same token when the user comes back. A locked screen is the same event and takes the
same path.

The gate cannot be perfect, because a step already in flight when the user swipes away
is already doomed — iOS offers no barrier to wait behind, and a prefill step can take
seconds. So the second thing is that the fault is recognised when it happens, and the
third is that it is repaired.

Recognition had to be taught the shape the fault actually arrives in. The chat service
catches the failure and ends the stream with a `done` frame carrying the message, so a
wrapper watching only for exceptions watches the wrong thing — which is exactly what the
first device run showed: the turn dead, the engine still marked healthy, and every
message afterwards dying too. A frame whose error names a refused command buffer, the
background-execution refusal, or the backend needing to be recreated now marks the
engine (`AgentAppHost.ReadsLikeAPoisonedEngine`).

The repair took a device to get right, twice. Reloading the weights does nothing: the
ggml backend is a process global that a model load never touches, so the "repaired"
engine was the same poisoned Metal context with fresh weights in it and the answer failed
again with the identical sentence. The native layer said as much in a comment — a
`std::once_flag` made the backend a one-shot and the honest advice was to restart the
host, which on a phone means the app. `TSGgml_RecreateBackend` is the missing half: it
tears the backend down the way shutdown does, un-shoots that one-shot, clears the latched
failure, and builds a new one. The model is released FIRST, because its tensors live in
the buffers being freed (`ModelService.UnloadModelAndRecreateBackend`). And the rebuild
itself waits for the gate, which is the second thing the device taught: loading a model
is GPU work too, so a repair attempted at the moment of backgrounding produces a backend
that is poisoned before its first token.

The one place a dead backend is met with nothing running is the warm-up itself — a GPU
reset caused by the previous process being killed mid-compute lands there — and the
warm-up then rebuilds the backend and warms the new one immediately
(`RebuildAfterAPoisonedWarmUp`: now, then after 15 s, then after 60 s, because the reset
that causes the fault discards every command buffer for the next minute or so — three
fresh backends faulted in 53 s, observed — and never more than three times per launch),
because leaving it to the
next message cost that message the rebuild AND the whole prompt: a 40 s first token on
a new chat, measured, where the user had done nothing wrong.

What the user sees is a sentence saying the GPU was interrupted, and then their answer
carrying on. The half-written text is handed back to the model as its own words with an
instruction to continue from exactly where it stopped — the KV cache went with the
backend so the prompt is re-read either way, but the READER loses nothing. Those two
extra messages are marked so they stay out of the transcript. A fragment too short to be
worth continuing, or one that stops inside a tool call, is started cleanly instead, and
says so.

Warnings and errors are also written to `Library/Caches/TensorAgent/logs/errors.log`,
with their stacks, and every lifecycle event and gate wait to `logs/background.log`:
`devicectl --console` detaches the moment the app is backgrounded, which is when the
failures worth reading about happen, and the files come back with `devicectl device
copy from`.

**Coming back after minutes away no longer needs a force-quit.** The gate above keeps
the *model* alive across an absence; the thing that died instead was the page's way of
reaching it. iOS reclaims — "defuncts" — the sockets of a suspended app, the listening
socket included and 127.0.0.1 no exception, and the app is suspended about thirty
seconds after it leaves the screen. The managed `HttpListener` the loopback server is
built on hides that completely: the pending accept stays parked, `IsListening` stays
true, nothing is thrown and nothing is logged, while every connection the WebView
opens is refused. The page saw that as WebKit's one-size-fits-all `TypeError: Load
failed` on every request; the only request whose failure it displayed was the share
claim it makes on becoming visible — hence "Could not open the shared item: Load
failed" — and the lookups that would have re-attached the answer failed silently
around it. The host, meanwhile, was fine: the trace shows the turn resuming after a
27-minute absence in the same second the user saw the error, and the process being
force-quit thirty seconds later.

Two layers fix it, because two things were wrong. The host now PROBES its listener on
every return to the foreground (`LoopbackLifecycle` on `willEnterForeground` →
`AgentAppHost.OnForegroundAsync` → `LoopbackServer.EnsureListeningAsync`: a real TCP
connect and a `GET /health`, the one thing a defunct socket cannot fake) and rebuilds
it when the probe fails — `Stop()` + `Start()` on the same instance keeps the same
port, so the page's origin, its token cookie and its composer are untouched; only if
the port cannot be had again does the server move, and then the WebView is
navigated to the new entry URL and comes back to the same chat and the same running
turn. The result is one line in `background.log` either way (`foreground: loopback
listener alive (...)` or `... DEAD; ...; rebound on port N`), and the page is nudged
again once the transport is known good.

The page, for its part, had four habits that turned any lost stream into a chat that
never recovered, and a lost stream needs no reclaimed socket — a suspended content
process, a replaced WebKit networking process, or a keep-alive connection the host
closed after 15 s idle will all do it. It trusted a stream *object* as proof of a
stream (`resumeTurn` returned early while one existed, so a dead one was never
replaced); it retried a failed lookup exactly once, 800 ms later, and then stopped
forever; it took a stream that ended without the host's `done` frame for a finished
answer, pushed the fragment into the history, and then, when it re-attached, rendered
the whole answer under it in a second bubble and saved both; and a POST that failed
at the transport — which CFNetwork never replays, unlike a GET — surfaced as an error.
Now a stream is trusted only while it is *delivering* (the host writes a keep-alive
every 5 s, so eight seconds of silence means it is dead and it is superseded — checked
on becoming visible, on the app's nudge, and by a watchdog, because a connection that
dies without a FIN raises no event at all), a failed lookup is retried with backoff for
about forty seconds before the page gives the screen back and says so, an early end
re-reads the turn — running or just finished — rather than mistaking a fragment for the
answer, a send that lost its stream during the prefill attaches to the turn the host
started (or puts the words back in the composer if it never did), and every `post()` is
retried once on a transport failure. Three things it will not do: supersede a request
that has not been answered yet (its headers carry the turn id, and the host consumes a
shared draft when it accepts it — aborting one made every later send a 409); take a
turn it has already read for the answer to a new question (a finished turn is retained
for an hour, so "what is this conversation generating" is very often the *previous*
question's); or leave a recovery armed after Stop or a chat switch. A reader that was
superseded, or stopped, is ignored when its failure finally arrives; the replayed
answer is painted once per chunk rather than once per frame, and a turn's errors and
restart notices are said once however often it is replayed. What the page
saw is kept in a small ring buffer (`window.TensorAgent.diagnostics()`) that the app
writes into `background.log` on every return, because nothing else ever records what
happened on that side. The whole transition is driven by
`scripts/verify-background.sh` with `CHECK=page`; the page's state machine is pinned
by `WebUiPageTests` against a fake transport that can refuse, hang, drop and abort.

**Metal.** On a device `ggml_metal` is the default and the first backend offered.
The simulator slice has no Metal at all — the simulator GPU is Apple1/Apple2 and
has no `simdgroup_matrix` — so there it is not offered, and CPU is the default.
The page is never shown a backend the build cannot initialise: a default that does
not exist puts the user one tap from a load that fails.

## Skills

`scripts/verify-skills.py` judges which skills can run on the phone, by parsing every
Python script and resolving each import against the staged interpreter, and by checking
every `.sh` / `.bash` script for the Node package managers and bundlers (npm, npx, pnpm,
yarn, parcel, vite) the in-app shell does not have. It refuses anything reaching for a
capability iOS does not have, and `skills/verdicts.json` records a verdict for each of
the twelve directories under `skills/`. Ten pass. Two fail, both on shell scripts:
`playwright`, whose only script execs `npx` to drive a Chromium (it was added for the
desktop-hosted agent; see "Native execution when hosted on desktop" above), and
`web-artifacts-builder`, whose two scripts install and run pnpm, npm and parcel.

The MAUI project excludes both by name, so **the app bundle carries ten skills**;
`TensorAgentMauiProjectTests` keeps that exclusion list equal to the failing verdicts
in both directions. The repository's `skills/` directory keeps all twelve, because it
is also the skill root of the desktop hosts — the server's
`--skills-dir TensorAgent/skills` and the desktop TensorAgent host's
`--skills TensorAgent/skills` — where both can run.

The skills that cannot run on the phone, and what blocks each (three more upstream
ones — `academy-guide`, `discernment-nudge`, `brand-guidelines` — pass the checker and
were unbundled anyway, because they instruct the model on behalf of another product in
every turn):

| Skill | Blocked by |
| --- | --- |
| docx, pptx, xlsx | the validators shell out to LibreOffice (`lxml` and `defusedxml` are now bundled; those were the other blocker) |
| pdf | `pdfplumber` is missing, and `pdf2image` shells out to poppler |
| skill-creator | `subprocess`, `webbrowser` |
| webapp-testing | `playwright` needs a browser engine |
| playwright | Node.js/npm and native browser processes; supported by the desktop process backend, left out of the app bundle |
| web-artifacts-builder | pnpm, npm and parcel; left out of the app bundle (the desktop hosts still get it) |
| mcp-builder | an MCP server needs a process and a socket |

Importable is not the same as usable: `subprocess` is in the standard library and
still cannot work here, so the checker tests unavailability before availability.

**Three upstream skills were unbundled, and one was written.** `academy-guide`,
`discernment-nudge` and `brand-guidelines` came from another product and said so in
every turn: the first told the model to recommend courses from Claude Academy on any
"how do I" question, the second to append follow-up questions to every substantive
reply, the third to apply Anthropic's brand colours. A description is read on every
turn whether or not the skill is used, and those three were imperatives aimed at the
model, in an app that is not a Claude product; two of them also sat at the head of the
alphabet and took half the catalog budget. In their place there is `market-data`,
which asks a structured JSON endpoint for the day's gainers, losers and most-traded
shares, or a quote for named symbols, and can write the rows straight into a
`make_pptx.py` spec. That is where a task-specific recipe belongs: a skill is injected
only when the request matches it, so it cannot bias the turns that do not.

**The switch.** The skills sheet carries a master toggle above the list. Off is not
"nothing is ticked": `ServerHostingOptions.SkillsEnabled` makes the request planner
build no skill plan at all, so no skill is declared to the model and none is reachable.
That is what the switch is for — every bundled skill (ten) announces itself in every
prompt, which on a phone is thousands of tokens on every turn of every chat, and a user
who wants a plain assistant should be able to have one. It removes skills only: the code
tools stay while "Run code" is on, and the sub-agent tools follow the separate
"Sub-agents" switch, whatever this one says. It
applies to the next message, not the next launch.

**`research` was rewritten.** The old one had a search script that was a client for
an endpoint the user was expected to configure, plus one unauthenticated fallback
that answers a phone with a challenge page more often than with results — so every
research request began by asking the person who wanted research to supply the URLs.
The new one asks nine keyless, machine-readable services at once (Wikipedia,
DuckDuckGo, Marginalia, Hacker News, arXiv, Crossref, GitHub, Stack Overflow, Google
News' RSS), merges what they say, reads the best pages, and writes a dossier with
every source cited. Results are ranked by how much of the question the title and
snippet cover, then by how many independent indexes named the same page: agreement
between indexes that share no crawler is the only quality signal available without a
ranker, and relevance is what stops a keyword index's confidence outranking it —
asked "what is the Kessler syndrome and is it happening", MediaWiki's first answer is
the article on mental disorders. `analyze.py` then sorts the collected sources into
those that state a claim, those that state it with a denial or a hedge nearby, and
those that never mention it, quoting the sentence and the source for each.

Providers fail individually and often; that is designed for rather than hidden. A
challenge page is detected and refused by name rather than parsed, because the
alternative is reporting an engine's own navigation as the user's sources.

## Languages

There are eight: `en`, `zh-Hans`, `zh-Hant`, `ja`, `ko`, `es`, `fr` and `de`
(`UiLanguages.Supported` in TensorAgent.Sharing). A system language tag maps to one of
them by its language subtag. Chinese is mapped by script and region: `zh-Hant`, `zh-TW`,
`zh-HK`, `zh-MO` and Cantonese map to Traditional Chinese, and any other Chinese tag maps
to Simplified.

| Text | Where it is |
| --- | --- |
| Native screens and the host's messages | `src/TensorAgent.Core/Localization/<tag>/<area>.json`, read with `Loc.T` and `Loc.Plural` |
| The chat page | The same tables, served as `/i18n.js?lang=<tag>`. The markup carries `data-i18n`, `data-i18n-placeholder`, `data-i18n-title` and `data-i18n-aria-label`; the script calls `t()` and `tn()` |
| The share extension, and the drafts the app composes from what was shared | `src/TensorAgent.Sharing/Localization/<tag>/share.json`, read with `ShareStrings` |
| Permission prompts | `src/TensorAgent.Maui/Platforms/{iOS,MacCatalyst}/Resources/<tag>.lproj/InfoPlist.strings` (the two files are identical) |
| The share extension's name in the share sheet | `src/TensorAgent.ShareExtension/Resources/<tag>.lproj/InfoPlist.strings` |

How the tables work:

- A table is a flat JSON object, and a key's first segment is its file's name
  (`settings.sandbox.runCode.title` is in `settings.json`). A key a language lacks falls
  back to English.
- Placeholders are named (`{model}`, `{count}`).
- Counted text has `.one` and `.other` keys, chosen by each language's plural rule
  (`StringCatalog.PluralCategory`, mirrored in `WebUi/i18n.js`).
- Numbers and dates the interface formats itself use `Loc.Culture`, which is never set as
  the thread's culture.

The share extension cannot read the app's settings. So the app writes the choice to
`ui-language.txt` at the root of the App Group container, and the extension resolves it
against the same system languages.

Some text stays English whatever the user picks:

- the logs;
- everything the model reads: system prompts, tool descriptions and results, and refusals
  it is meant to act on;
- text the page compares with the model's or a tool's output;
- protocol values.

**Adding a string.**

1. Add the key to the English table, in the order the text appears on screen.
2. Use it from the code as a literal: `Loc.T("area.key")`, `t('area.key')` or
   `data-i18n="area.key"`.
3. Add it to every other language.

**Adding a language.**

1. Add it to `UiLanguages.Supported`. Extend `Match` too if its tags need more than the
   first subtag.
2. Give it a plural rule in `StringCatalog.PluralCategory` and in `i18n.js`.
3. Write every table and the three `InfoPlist.strings` files.
4. Add the tag to `CFBundleLocalizations` in the three `Info.plist` files.

`LocalizationTests` fails, naming each gap, until all of that agrees. It checks that:

- every language has every English key, with the same placeholders;
- every key the code asks for exists, and every key is asked for by some code;
- text the page repaints, and a menu row and the screen it opens, read the same;
- the system's files and the bundles' language lists match the tables;
- the running host follows the system on first launch, keeps each explicit choice after
  a restart, and follows the system again after choosing "System";
- refreshing the chat keeps unsent drafts and waits for a shared send's acceptance;
- page string responses keep their language and text consistent during concurrent switches.

For simulator checks of system detection, the Settings picker and relaunch persistence,
use `python3 ../eng/validation/verify-tensoragent-languages.py --help` from this directory.
The tool creates an isolated simulator and keeps logs and screenshots in ignored
`docs/validation/`; it does not run model inference or validate a physical device.

## Tests

```
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj
```

The project is part of `TensorSharp.slnx`, so `dotnet test TensorSharp.slnx` runs it
together with `InferenceWeb.Tests`.

Hermetic by default. These groups need something the machine may not have and say
so rather than passing silently:

| Set | Enable with |
| --- | --- |
| Live CPython | `TENSORAGENT_PYTHON_ROOT=<a staged slice or a CPython 3.13 prefix>` |
| End-to-end chat | `TENSORAGENT_TEST_MODEL_DIR=<a directory of catalog GGUFs>` (and `TENSORAGENT_TEST_MODEL_FILE` for a differently named copy) |
| Metal lifetime | the same weights, plus a Mac whose GgmlOps was built with ggml_metal |
| Image, audio and video input | `TENSORAGENT_TEST_MODEL_DIR` holding a multimodal catalog entry and its projector (`TENSORAGENT_TEST_MMPROJ_FILE` for a differently named projector) |
| Image editing | `TENSORAGENT_TEST_IMAGE_MODEL_DIR=<a Qwen-Image-2.1 DiT GGUF, its VAE, a Qwen3-VL-8B text encoder and its mmproj>` |
| Video generation | `TENSORAGENT_TEST_VIDEO_MODEL_DIR=<a Wan DiT GGUF, its VAE and a umt5-xxl encoder>` (`TENSORAGENT_TEST_VIDEO_MODEL_FILE` picks one DiT) |
| The open web | `TENSORAGENT_ALLOW_NETWORK_TESTS=1` — these ask the real internet a real question |
| Desktop process backend | macOS, Linux or Windows with a working OS sandbox and shell, plus Node.js for three of the four `DesktopAgentHostTests` |

`TENSORAGENT_TEST_BACKEND` chooses the backend for the three media sets: the input
tests default to the CPU, the image-editing and video-generation ones to Metal.

The LoRA plug-in tests (`LoraCatalogTests`, and the sheet's in `AgentAppHostTests` and
`WebUiPageTests`) are hermetic: an installed plug-in is files of its pinned sizes. Whether
the real files load is `InferenceWeb.Tests`' `RealArticleLoras_LoadCompletelyAgainstTheCheckpoint`,
with `TENSORSHARP_QWEN21_LORA_DIR` pointing at the app's `Library/Caches/TensorAgent/loras`
and `TENSORSHARP_QWEN21_DIT` at its Qwen-Image transformer GGUF. Its rank-256 Viggle Turbo
row needs a file the app does not ship, so against that folder it fails with
FileNotFoundException and the twelve the app offers pass.

None of this runs in CI. `.github/workflows/pr-unit-tests.yml` runs `InferenceWeb.Tests`,
whose `TensorAgentMauiProjectTests` read the MAUI head's project file, `Info.plist`,
entitlements, share extension and native export manifest; `TensorAgent.Tests` is not run
there, and no workflow builds the iOS app.

The live-CPython classes share one queue (`LivePythonCollection`). There is exactly
one interpreter per process and one sandbox policy inside it, so running those
classes in parallel had each overwriting the others' permissions: twenty-one tests
failed with messages naming a different test's temporary directory, and a machine
that had staged an interpreter looked broken while one that had not passed the whole
suite.

`WebUiPageTests` RUNS the page. `tensoragent.js` is loaded into JavaScriptCore on
top of `PageDom.js` — a DOM the size of what the script touches, plus a fetch that
answers from a table and records every request — and then driven the way a person
drives it: attach something, send it, reopen the chat. Everything else that guards
that file reads it as text, which cannot answer whether a photo comes back when a
saved chat is opened again. It did not, and six of those seven tests fail against
the version before the fix.

The Metal set is about teardown rather than answers. ggml-metal's device is a C++
static whose destructor asserts that every residency set has been handed back, so a
buffer our side forgets to release does not fail a test — it aborts the process at
exit, after the run reported success. These load, generate, unload and switch on
Metal and measure the device allocation directly, which is both the mechanism behind
that assert and what a phone runs out of. They will not fall back to the CPU, where
none of it exists; without Metal they skip and say so.

The end-to-end set loads a real model and drives the real API: a question answered,
a four-turn conversation, cache reuse and its invalidation, a conversation that
survives a restart, an aborted generation, and a throughput floor. It runs on the
CPU, so budget half an hour for it and do not rebuild the test project while it is
running — that overwrites the assembly under the running host and the failure looks
exactly like a native crash.

Two sets are worth naming because of what they are written against rather than what
they need. `ModelDownloadManagerTests` drives real HTTP transfers through a loopback
range server and asserts the property the whole download rework exists for: a watcher
that walks away does not take the transfer with it. `UploadNamingTests` and the two
upload tests in `MediaRoutesTests` pin the other end of "Upload failed (400)" — a
photo whose name has no extension is placed from its own bytes, and something nobody
can identify is still refused with a sentence a person can read.

No test here makes a MiniMax-H3 clip. `VideoTurnsTests` pin which message becomes which
request for which checkpoint and how the video service's frames reach the page and the
saved chat; `MiniMaxH3CatalogTests` that the two entries share every file but the
denoiser, under the names the engine looks for; `SharedFileLinkTests` that a shared file
is linked rather than fetched; and `LoopbackRangeTests` the ranges and HEAD, through the
real routes. The clip itself is checked against the running Mac app by `chat-e2e.py`'s
`film`, `animate` and `reference` scenarios (see "On a Mac").

### Measured

Gemma 4 E4B Q8_0, CPU backend, on a development Mac. The point of these numbers is
the shape, not the absolute value — a phone with Metal is a different machine.

| Turn | Prompt tokens | Reused | Reuse |
| --- | --- | --- | --- |
| 1 | 2812 | 0 | 0% |
| 2 | 2932 | 2908 | 99.2% |
| 3 | 3019 | 2996 | 99.2% |
| 4 | 3105 | 3082 | 99.3% |

Only the new message and the previous answer are processed on each turn. Rewriting
an earlier turn invalidates from the point the histories diverge, and the model then
answers from the rewritten history. (Starting a new chat used to drop reuse to zero
as well; see the next section for what changed.)

### Memory on the phone

A jetsam kill leaves no stack and no message, so the app now writes the two numbers
the kill is decided on: what the process is charged (`phys_footprint`) and what the
device has wired and free (`host_statistics64`), after a load, after the warm-up,
every half minute of a turn, after it, and at a memory warning (`ProcessMemory`;
`/api/agent/engine` carries the same line). The second number is the one that
matters. The weights are a file mapping that Metal wires while the model is loaded,
and wired file pages are charged to the machine rather than to the process, so a
5 GB model reads as nothing in the footprint and as +5 GB in the wired total. Every
jetsam report the phone kept showed the app at 2-5 GB with the device at 8-10 GB of
its 12 GB wired.

What the engine holds beyond the cache a turn is using is set in `EngineMemoryPolicy`
on every load, and each value is measured: caches start at 2,048 tokens and grow
(`TS_KV_INITIAL_TOKENS`); a request pre-reserves at most 1,024 tokens of reply beyond
its prompt, in 2,048-token steps (`TS_KV_GENERATION_RESERVE_MAX`); one finished
conversation stays resident (`TS_RETAINED_FUSED_CACHE_MAX=1`), plus the shared-prefix
checkpoint (`TS_PREFIX_CHECKPOINTS_MAX=1`, set at launch; the engine's default is two);
nothing is parked (`TS_KV_HOLDER_POOL_MAX=0`). The same call sets `MAX_CONTEXT` and
`KV_CACHE_DTYPE` from the catalog entry or the user's settings. (The reply length
setting used to decide the reservation: at its
top rung, 262,144 tokens, every request reserved the whole 32k window, host copy and
Metal mirror both.) ggml-metal's residency set is off on the phone
(`GGML_METAL_NO_RESIDENCY=1`, set at launch unless the launch environment already sets
it; `0` keeps the set), so the weights can be reclaimed while a tool runs, and a solo
prompt is prefilled in 1,024-token chunks (`TS_SCHED_SOLO_PREFILL_CHUNK`, also set at
launch). TensorSharp shares complete flash-attention
workspaces after their final consumers finish, reducing the persistent graph's
allocation while building unchanged upstream ggml (see
[allocation and benchmark details](../docs/perf/ggml-without-patches.md)). The memory warning now asks the engine to release what only
serves the next request's speed, on the engine's own thread between steps.

Measured on the Mac with the phone's settings and the research-then-slides prompt
that was killing the app (`--scenarios agentic --network`), footprint at the end of
the turn: Qwen3.5 9B IQ4_XS 3.7 GB before, 1.7-1.8 GB after. On the
iPhone 17 Pro Max with the user's own settings, the same prompt runs past 24,000
tokens of context at a 1.7 GB footprint where it used to die.

### The first message of a launch

TensorAgent uses the radix KV prefix cache by default, through the same engine as
TensorSharp.Server and TensorSharp.Cli. Prefix lookup respects the model's cache
capabilities, conversation scope, and media boundaries. The phone's existing
retention limits still apply, and memory warnings release idle cache payloads.
`TS_SCHED_PREFIX_CACHE=0` disables runtime prefix reuse.

Every chat starts from a copy of the model's state at the end of the prompt they all
share, but in a fresh process that state has to be made first: the warm-up after a
load prefills it, and on the phone that is 36-48 s for Qwen3.5 9B, which a first
message sent sooner pays in full. The checkpoint is now
written to `Library/Caches/TensorAgent/prefix-cache/<model id>/` the first time it is
taken (`PrefixCheckpointFileStore`) and read back by the next load
(`IPrefixCheckpointStore`, consulted by the engine at admission), so the first
message of every later launch clones it like any other new chat. A file is named
and checked by the model's K/V identity and the exact prefix tokens, written under a
temporary name and renamed, and at most two are kept per model, the least recently
used evicted first; one that no
longer describes its model is deleted and the prefix is prefilled and saved again.
Deleting a model deletes its checkpoints. Checkpoints, in memory and on disk, exist
for the Gemma 4 and `qwen35`-architecture entries (Gemma 4 E2B, E4B and 12B, Qwen3.5
9B, Bonsai 2 27B). The same idea as
llama.cpp's prompt-cache files, scoped to the one prefix the app cares about.
`benchmarks/TensorAgentTtftBench --scenarios restore`, run twice against the same `--root`, measures the difference;
on the Mac (M5 Pro), first message of a launch, no warm-up waited for:

| Model | Cold launch | Next launch | Checkpoint file |
| --- | ---: | ---: | ---: |
| Qwen3.5 9B IQ4_XS | 5.51 s | 0.41 s | 102 MB, restored in 39 ms |

On the iPhone 17 Pro Max, Qwen3.5 9B: the first launch wrote 117 MB in 386 ms after a
54 s cold first message; the next launch restored it in 44-284 ms and the warm-up (a
full first request, one-time graph builds included) took 1.2 s on the Release build
where it took ~40 s before. A message sent after that warm-up starts in ~0.6 s.

`PrefixCheckpointExactnessTests` proves the restored copy is the model: a chat started
from it produces the same tokens as a cold prefill, on Metal, for Qwen 3.5 and Gemma 4.

### Speculative decoding

Every turn is decoded speculatively unless the "Speculative decoding" switch in
Settings is off: a drafter guesses a few tokens ahead and the model
verifies them in one batched forward, so the answer is exactly what plain decoding would have
produced and it arrives in fewer forwards. The drafter is the model's own draft head
when the catalog lists one and it is downloaded with the optional files (Gemma 4 E4B
and 12B; a model installed before this change gets it with its next optional
download, and the head attaches at the next load), and otherwise a lookup over the
conversation's own tokens (n-gram), which
needs no weights and pays where an agent turn quotes a file, a tool result or an
earlier answer. `SpeculationPolicy` hands both to the engine at load time, through
the same environment the CLI's `--draft-model` and `--spec` use, and the engine's cost
governor parks drafting while it measures as a loss. Both catalog families
speculate on the app's cached-holder path: Gemma 4 with its draft head when the
optional file is downloaded (n-gram otherwise), Qwen 3.5 with n-gram. Measured on
the Mac host with the phone's settings, quoting or echoing text runs 1.6-2.5x plain
decoding and free prose stays within about 5% (Qwen) to 15% (E4B with the draft
head) of it. On an iPhone 17 Pro Max (`scripts/bench-spec-device.sh`) quoting runs
1.2-1.9x, prose 0.8-1.0x, and each turn's first token costs 0.1-0.6 s more with the
setting on; leave it off for chats that are mostly free prose.

The switch applies to the running engine at once, from the next reply: the Settings
page applies it through `AgentAppHost.ApplySettings`, which hands the engine the new
choice through `UpdateSpeculation`, instead of waiting for the next model load. A
`TS_SPEC` or `TS_SPEC_TYPE` set in the launch environment still wins over the switch.
To measure it on the phone,
`scripts/bench-spec-device.sh` deploys the app, launches it with
`TENSORAGENT_SPEC_BENCH=1`, and pulls back `Library/Caches/TensorAgent/logs/specbench.log`:
the same four turns under plain and speculative decoding, twice each, with prefill and
decode rates per turn (`SpeculationBench`). The Mac host benchmark runs the same turns
as `TensorAgentTtftBench --scenarios spec`, and `--no-spec` gives the other half.

Measured on the Mac (ggml_metal, greedy, plain → speculative, streams identical):
Gemma 4 E4B with its draft head 46 → 92 tok/s; Qwen 3.5-9B with n-gram 31 → 86 tok/s
on an answer that quotes a file and 31 → 27 on prose; Gemma 4 E2B with n-gram 81.5 →
80.9 on prose. Two engine faults this depended on are fixed in the same change: the
executor re-armed speculation on every prefill chunk (so any prompt longer than the
phone's 1024-token chunk lost its draft head), and a rejected verify window left
stale rows in Gemma 4's sliding-window cache
(`Gemma4SwaRollbackExactnessTests`). `benchmarks/AgentTurnBench` measures all of it.

### Every conversation shape, on Metal

`benchmarks/TensorAgentTtftBench` starts the real app host on the Mac with the phone's
settings (catalog context and K/V budget, 1024-token solo prefill chunks, every
bundled skill — thirteen when the tables below were measured, a different set from
the twelve in `TensorAgent/skills` today (the bench reads that directory; the app
bundle carries ten of them): three were unbundled and `market-data` added the next
day, and `playwright` later; the tool block has changed as well: the five sub-agent tools
(2026-09-24) and `apply_patch` in place of `edit_file` (2026-09-18) are declared now
and were not then), loads a catalog model on Metal the way tapping "Use" does, and
drives `/api/chat` exactly as the page does through every shape a conversation takes. It
prints, for each turn, the first-token time, the prompt size, how much of it the KV
cache served, and — next to any turn that reused nothing — the engine's own line
saying why. Run it with `--model <catalog id> --source <dir with the entry's files>`
(or `--weights <gguf>`), and `--warm` to let the prefix warm-up finish first, as a
user who takes a few seconds to type does.

Gemma 4 E2B Q8_0, ggml_metal, M5 Pro, 2026-09-07, first token / prompt reused:

| Turn | Before | After |
| --- | --- | --- |
| First turn of the first chat | 2.03 s / 0% | 0.16 s / 99.7% (warm-up) |
| Follow-up in the same chat | 0.10 s / 99.6% | 0.11 s / 99.6% |
| First turn of a NEW chat | 1.50 s / 0% | 0.11 s / 99.6% (checkpoint) |
| Turn after the user tapped Stop | 0.08 s / 99.4% | 0.08 s / 99.0% |
| Turn after a tool round | 1.52 s / 0% | 0.14 s / 95.9% |
| Thinking toggled on, same chat | 1.52 s / 0% | 1.55 s / 0% — Gemma 4's template puts the thinking marker at the top of the system turn, so that prompt shares nothing with the other mode |

Qwen 3.5 9B Q8_0, same machine:

| Turn | Before | After |
| --- | --- | --- |
| First turn of the first chat | 5.76 s / 0% | 0.29 s / 99.7% |
| First turn of a NEW chat | 4.95 s / 0% | 0.25 s / 99.6% |
| Thinking toggled on, same chat | 4.99 s / 0% | 0.23 s / 99.5% |
| Turn after the user tapped Stop | 5.17 s / 0% | 0.36 s / 99.2% |

The checkpoint is a copy, so it had to be proved a faithful one:
`InferenceWeb.Tests/PrefixCheckpointExactnessTests` generates greedily from a chat
started on the clone and from a cold prefill of the same prompt and requires the two
token sequences to be identical. Both families pass on Metal. Its first version
failed on Gemma 4 for a reason worth knowing: the engine's older trick of continuing
the live cache by rewinding up to sixteen trailing tokens is not exact on a
sliding-window model — the rewound tokens' keys stay in the ring where the window's
oldest positions should be, and a 15-token rewind changed the answer from its fifth
token. The engine now prefers a retained state or a checkpoint whenever one covers
the prompt exactly, and keeps the rewind only as the fallback.

### In the simulator, with a real model

The same Gemma 4 E4B Q8_0, linked into the simulator's model directory and loaded
through the app's own routes, answering through the app's own chat stream:

| | Prompt tokens | Reused |
| --- | --- | --- |
| Turn 1 | 5189 | 0 |
| Turn 2 | 5238 | 5221 (99.7%) |

The transcript was written to the app's container and listed by
`/api/agent/conversations`. Throughput there is not worth quoting: the simulator
has no Metal, and the prompt is large because every bundled skill (twelve at the time,
a different set from today's) declares itself; the declared tools differed too (the
five sub-agent tools came later).

## What has not been verified

Stated plainly, because the rest of this file is written as though everything was
checked and these were not:

- **Downloading in the background.** The manager, the routes and the resume are
  covered by tests that move real bytes. The background-task assertion has run on a
  device — a download kept arriving after the app was sent to the background (see "On
  a physical iPhone" below) — but a transfer that iOS actually stops, and the resume on
  `willEnterForeground` that follows, have not: neither can be exercised in a simulator
  that is never suspended, and how long iOS actually grants is a property of a real
  device under real memory pressure.
- **The native picker's own file names.** `UploadNaming` is tested against the shapes
  iOS produces (a stem with no extension, no content type, HEIC and MP4 bytes behind
  the same absent name), but the picker itself has only been run by hand.
- **Pictures on a phone.** The page makes and edits pictures through `/api/chat` with
  Qwen-Image 2.1, which needs the 24 GB memory tier and was measured in the Mac app;
  no image has been generated on iOS. `/api/image-edit` remains bound to the same
  service the desktop uses, with the LoRA plug-ins the user chose, but nothing on the page
  calls it, and `/api/image-generate`, the desktop page's text-to-image route, is not bound
  in the app.
- **LoRA plug-ins beyond the prompts tried.** Each of the twelve made pictures in the app
  from the prompts and photos `chat-e2e.py` sends, and the two box-driven edits were also run
  on their authors' example photos. Object Remover failed on one of its two. How each fares
  on other subjects is up to the plug-in, and nothing here measures it.
- **Qwen3.8 Flash Next beyond the recorded configurations.** UD-Q2_K_XL was measured
  in the Debug app on one 48 GB M5 Pro. A Mac with more memory gets a different expert
  placement plan, which nothing here has measured; the Q2 entry retains its 48 GB tier.
  UD-IQ1_M has Windows server validation at 32 GB system RAM and 16 GB CUDA VRAM,
  but no recorded Mac/Metal or TensorAgent app end-to-end validation. Its 32 GB tier
  remains experimental. Nor has the app's own downloader fetched the Q2 package's
  78.9 GB: the three shards were downloaded
  with `curl` and hard-linked into the app's store. How the store counts a split's shards
  is covered by `Qwen38FlashNextCatalogTests`.
- **Clips on a phone.** The two MiniMax-H3 entries need the 32 GB memory tier and
  were measured in the Mac app. No video model fits a phone, so none is offered there, and
  no clip has been played by iOS's media loader (WebKit's playback was checked in the
  Mac app). `/api/video-generate` stays bound as part of the shared surface, but nothing
  on the page calls it.
- **Sub-agents on the phone.** Delegation is offered in every chat while the
  "Sub-agents" switch is on (see "Sub-agents, on by default, with a switch"), but no delegated turn on a phone or in the
  simulator is recorded, and nothing has measured what up to three concurrent
  children cost in memory or time on a 12 GB device.
- **iOS 27.** The scene-lifecycle change that fixed the launch crash with the iOS 27
  SDK is in the build, but the only device run recorded is on iOS 26.6.1.
- **Package installation.** `WheelInstaller` refuses without the network switch and
  accepts only pure-Python wheels; the accepting path has not run on iOS. A request
  for a package the bundle ships (numpy, Pillow, lxml, python-pptx, python-docx, …)
  never reaches it: the installer answers for the bundle first.

- **The first-token numbers ON THE PHONE after the 2026-09-07 cache work.** Every
  figure in "Every conversation shape, on Metal" is from a Mac driving the real app
  host; the phone was not reachable that day. The phone numbers recorded since are
  the warm-up and persisted-checkpoint times in "The first message of a launch"
  (Qwen3.5 9B) and the speculation benchmark in
  "Speculative decoding"; none covers the per-shape table. The Debug build carries a probe for
  exactly this: launch with `TENSORAGENT_TTFT_CHECK=1` (and `TENSORAGENT_USE_MODEL`)
  and read the four `ttft` lines off `devicectl device process launch --console` —
  first chat, follow-up, new chat, follow-up. The shape to expect is the Mac's; the
  absolute times are the phone's.

### On a physical iPhone

An iPhone 17 Pro Max (A19 Pro, 12.26 GB, iOS 26.6.1), Debug build, installed with
`devicectl`:

```
engine probe   backend=GgmlMetal  ggmlMetalAvailable=true  gpu="Apple A19 Pro GPU"
               reason: ggml-metal on Apple A19 Pro GPU (MTLGPUFamilyApple7 present)
memory tier    physical memory 12.26 GB -> catalog tier 12 GB
self-test      all fourteen pass, including all seven CPython checks
gestures       all seven uicheck lines pass in the phone's own WKWebView
model load     gemma-4-E2B-it-Q8_0 (4.63 GB) + projector on ggml_metal in 22 s
a whole turn   prompt -> shell tool -> in-process CPython -> answer, recorded to the
               conversation store on the device
downloads      157 MB fetched in 40 s, still arriving after the app was sent to the
               background, cancelled cleanly, and the app was not terminated by iOS
away mid-decode
               Qwen3.5-9B: the engine parked on the gate twice (21.9 s and 12.2 s away),
               resumed each time, finished a 4,093-token answer with no fault, no
               restart and no rebuild; the next answer worked
away mid-prefill
               left 1 s after the answer was asked for: the GPU refused the step
               ("cannot recover in this process"), the turn was marked, the retry
               waited behind the gate, the backend was rebuilt and Qwen3.5-9B reloaded
               in 2.2 s, the 519-token answer finished (1 restart, 1 rebuild); the
               next answer worked
```

Both of the last two are `scripts/verify-background.sh device` (and `sim`, where the
gate is proved but no refusal can occur): it launches a Debug build with
`TENSORAGENT_BACKGROUND_CHECK=1`, brings Settings to the front once tokens are flowing
(`LEAVE_DURING=prefill` leaves the moment the answer is asked for, with a long prompt),
brings the app back, and reads `logs/background.log`. On the simulator pass
`TENSORAGENT_BACKGROUND_TOKENS=120`: a 4,096-token answer takes hours on its CPU.

`CHECK=page` runs the same transition through the real WebView instead of the host's
own HTTP client: the prompt is typed into the page (`TENSORAGENT_DEMO_PROMPT`), the app
is sent away for `AWAY_SECONDS` (720 by default on a device — long enough for iOS to
suspend the app and reclaim its sockets; the simulator never reclaims them and proves
only the page's side), and what is asserted is what the user sees: `pagecheck ok N
chars on screen in 1 bubble after the turn ended, page idle`, plus the host's own
`foreground: loopback listener alive|DEAD; ...; rebound on port N` line and the absence
of `Could not open the shared item`. The page's transport diary (`page (foreground):
{...}`) lands in the same trace.

Two things that run only here and nowhere else: the device slice of the engine (the
simulator's has no Metal at all) and the device staging of CPython, where every
compiled extension module is a signed framework rather than a `.so`.

### On-device self-test

Debug builds run a self-test at launch and log one line per check, because the
failures that matter here are not compile errors — an interpreter that links but
cannot find its standard library produces an app that starts perfectly and fails on
first use. On the simulator all fourteen pass:

```
ok shell: HELLO                 ok python:lxml: 3.0 <o>2</o>
ok shell:files: ab              ok python:pptx: 1
ok shell:awk: 6                 ok python:docx: 1
ok python: {"v": [3, 13]}       ok node: 2,4,6
ok python:stdlib: stdlib ok     ok node:print: 2
ok python:numpy: 3              ok sandbox:write: Permission denied
ok python:pillow: (2, 2)        ok sandbox:network: network access is disabled by the user
```

`python:lxml` parses, evaluates an XPath and runs an XSLT transform, which touches
all of etree's static libxml2/libxslt linkage at once; `python:pptx` and
`python:docx` each write a document and read it back.

### The composer, in real WebKit

The page's behaviour is the half of this app no unit test can reach: it is JavaScript
in WKWebView, and "is the hold-to-talk button visible" is a layout question with no
answer anywhere else. `TENSORAGENT_UI_CHECK=1` synthesises the gestures in the running
app and logs one line per assertion, which `verify-sim.sh` asserts on:

```
uicheck voice-switch-gone ok
uicheck reasoning-is-a-setting ok
uicheck skills-moved-to-the-menu ok
uicheck activity-above-the-box ok
uicheck a-tap-still-types ok
uicheck holding-the-box-gives-hold-to-talk ok
uicheck the-keyboard-button-returns ok
uicheck the-menu-comes-from-the-left ok
uicheck the-menu-lists-the-saved-chats ok
uicheck a-turn-can-be-taken-back-up ok
uicheck skills-have-a-master-switch ok
```

The menu is measured rather than asserted about — flush with the left edge, narrower
than the window, as tall as it — because a bottom sheet that had merely been renamed
would pass every check that only looked at class names.

### An answer that outlives the screen it was on

```
navcheck leaving the chat with a turn running · <id>=Running 1 frames
navcheck ok away for 20s: frames 1 -> 43, still generating=True
navcheck ok back in the chat, the answer on screen is 190 characters
```

The frame count rising while the chat is not on screen is the whole claim, and it is
the one thing a unit test cannot make: dropping a reader in a test proves the server
keeps going, and says nothing about what iOS does to the WebView. The last line is the
other half — the page found its way back to the turn rather than showing the half
sentence it walked away from.
