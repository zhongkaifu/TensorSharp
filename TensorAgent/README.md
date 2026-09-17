# TensorAgent

An iPhone and iPad app with the full capability of TensorSharp.Server's Web UI
chat, running entirely on the device: a .NET MAUI (`net10.0-ios`) head that links
the TensorSharp engine statically, serves its own phone-shaped page to a WKWebView
from an in-process loopback HTTP server, and answers that page's API with the same
chat pipeline the desktop uses.

This is the current source implementation of TensorSharp's iOS/iPadOS target.
Physical devices use the GGML Metal (`ggml_metal`) backend; build it with
`TensorSharpIosTargets=true`. It is not a remote client or a separate inference
engine. The latest tagged desktop release may not include TensorAgent yet, so
follow the source-build instructions below.

Nothing leaves the phone. The model runs locally, the sandbox has no network unless
the user grants it, and dictation asks for on-device speech recognition.

## What it does

**Chat with the desktop's capabilities, on a page built for a thumb.** The app
ships its own page — `src/TensorAgent.Maui/wwwroot/index.html`, bundled as
`webui/` and served from the loopback host. It is not the desktop page: that one
is laid out for a mouse and a wide window, and no amount of injected CSS turns it
into a phone UI. One row of chrome, everything reachable at the bottom next to
the keyboard, a layout that follows `visualViewport`, a single `+` sheet for
Photo / Camera / Video / File, and reasoning collapsed behind a disclosure.

What is shared is the API, not the document. The routes under the page —
`/api/chat`, `/api/models`, `/api/sessions`, `/api/upload`, `/api/skills`,
`/api/image-edit`, `/api/video-generate` — are bound to the same
`WebUiChatService` and `SkillsService` the desktop server binds, so streaming,
tool progress, reasoning blocks, skill steps and artifact links behave
identically. The app's own client is appended as one script tag at request time;
the page file itself is never forked.

**A built-in model catalog.** Six dense entries chosen to fit a phone, with the
exact byte size and SHA-256 of every file. Four are downloadable; those downloads
resume from a kept `.part` after an interruption, are verified before use, and
belong to the APP rather than to the screen that started one — see "Downloads"
below. The two Bonsai cards are text-only, local-import entries: their GGUFs embed
no publisher repository or license, so the app offers a file picker instead of
inventing a download URL and accepts only the exact hash-pinned artifact.

| Model | Modalities | Required artifact(s) | Needs | Source |
| --- | --- | --- | --- | --- |
| Gemma 4 E2B (Q8_0) | text, image, audio, video | download: 4,967,497,152-byte main GGUF + 557,368,064-byte projector | 12 GB | `ggml-org/gemma-4-E2B-it-GGUF` |
| Gemma 4 E4B (IQ4_XS) | text, image, audio, video | download: 4,715,416,704-byte main GGUF + 559,874,816-byte projector; 98,653,280-byte draft optional | 12 GB | `unsloth/gemma-4-E4B-it-GGUF` + `ggml-org/gemma-4-E4B-it-GGUF` projector |
| Gemma 4 12B (UD-IQ2_M) | text; image and video with optional projector | download: 4,213,353,280-byte main GGUF; 175,115,840-byte projector and 465,109,248-byte draft optional | 12 GB | `unsloth/gemma-4-12b-it-GGUF` |
| Bonsai 8B (Q1_0) | text only | local import: `Bonsai-8B-Q1_0.gguf`, exactly 1,158,654,496 bytes | 12 GB | no publisher repo embedded |
| Bonsai 27B (Q1_0) | text only | local import: `Bonsai-27B-Q1_0.gguf`, exactly 3,803,452,480 bytes | 12 GB | no publisher repo embedded |
| Qwen3.5 9B (IQ4_XS) | text; image and video with optional projector | download: 5,168,653,536-byte main GGUF; 918,166,080-byte projector optional | 12 GB | `unsloth/Qwen3.5-9B-GGUF` |

The catalog is filtered by the device's own memory, so a phone is never offered a
model it cannot load.

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
goes back to typing. Recognition is Apple's, asked for on-device.

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

**The answer keeps being written while you are somewhere else.** A generation belongs to
the app, not to the HTTP request that asked for it (`ChatTurnManager`). This is not a
refinement: iOS suspends a WKWebView's content process the moment its view leaves the
window, which is what opening the model list does, so the page stops reading — and a
server that took that as "nobody wants this any more" threw away the minute the user had
just waited. Now the turn runs on, buffers what it produces, and the page ATTACHES to it
again when it comes back, replaying from the first frame; the display is held awake and
a background-task assertion is taken for as long as the model is working, both driven by
the host's own answer rather than by whichever page happens to be watching. Stopping is
something the Stop button asks for explicitly. The transcript is written by the turn, so
an answer that finishes with nobody reading is still there.

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
like. What is left is 999 characters that are true of any request: prefer the standard
library for a lookup, write multi-line programs as a quoted heredoc rather than
`python3 -c`, and when a command's output already answers the question, copy it exactly
and invent nothing. Task-specific help belongs in a skill, which is injected only when it
is selected. `AgentAppHostTests` fails if any tool description names a vendor or product
again.

**Every bundled skill reaches the model's catalog.** Six of the thirteen did not. The
catalog is filled in id order under a budget of about a thousand tokens, the thirteen
descriptions come to half again as much, and the alphabet decided who was cut:
`documents` and `research` — the two this app's own router depends on — were both
below the line, while two entries of nearly a thousand characters each at the head of
the alphabet took half the budget between them. A model asked to look something up
listed the skills it could see, found nothing that fetches a page, and refused. The
catalog now SHORTENS entries rather than dropping them, at the longest of a few fixed
lengths where everything fits, and charges for the text it actually emits; a catalog
that already fits is left exactly as it was.

**The sandbox switches take effect now.** "Run code" and "Allow network access" used to
apply "the next time TensorAgent starts", which is honest and useless: leaving an iPhone
app does not restart it, so the real instruction was "force-quit from the app switcher"
and the switch read as one that did nothing — network turned on, `curl` still answering
"network access is disabled by the user". `AgentAppHost.ApplySettings` moves all four
holders together (the runner's options, the installer's standing policy, the shell's
host list, and the terms a skill's scripts are planned against). A command already
running keeps the terms it started with.

**Skills.** Eleven are bundled, chosen by inspection rather than by hope — see
"Skills" below. Users can install more from a zip.

**Code, generated and run.** The agent host's shell tool works here, backed by an
in-process POSIX shell, an embedded CPython 3.13 and JavaScriptCore, because iOS
allows no child processes at all.

A missing command never ends in "command not found" and nothing else. Installing a
native program is available to nobody here — iOS runs no child processes and will not
execute a binary that was not signed into the bundle — so the shell names what does
work instead: `$(( ))` and `python3` for `bc`, the interpreters for another language,
the fact that `apt`/`brew`/`sudo` have no meaning on this device *and* that Python and
JavaScript packages do install with `pip`/`npm` when network access is on. It also
catches a transposed name. This is not politeness: a dead end is where a model stops
using the shell and starts inventing the answer, which is exactly what one did on a
phone — reaching for `bc` to subtract two dates, being told 127, and finishing the
arithmetic in its head with the wrong number and a formula underneath.

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
a new chat went from 1.5 s / 0% reuse to 0.11 s / 99.6% on a Mac, and the same shape
holds on the phone, ten times slower in absolute terms. Two more turns that used to
re-prefill everything no longer do: a turn after the user tapped Stop (the transcript
now records the tokens the engine forwarded past the last one streamed), and, on Qwen
3.5, a turn after the thinking toggle changed (the two thinking modes rendered through
different code and disagreed from the first tool declaration on; they now share one
renderer, and each answer remembers which mode its prompt ended in).

**A sandbox the user controls.** Two switches, both in Settings, both defaulting to
the safe answer: code execution on, because an agent that cannot act is not an
agent, and network off, because a model that can reach the internet from inside a
sandbox is a different risk entirely. Every setting on that page does something;
one that could not be enforced was removed rather than left there implying it was.

## Build and run

The user-local SDK is the one with the MAUI workloads:

```
export DOTNET_ROOT="$HOME/.dotnet"
export PATH="$HOME/.dotnet:$PATH"
```

One-time preparation, both of which produce files that are not in git:

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

Then:

```
TensorAgent/scripts/build-sim.sh            # simulator build
TensorAgent/scripts/run-sim.sh              # install, launch, stream stdout
TensorAgent/scripts/verify-sim.sh           # drive the running app's API from the Mac
                                            # (also takes a DEVICE log: a phone's 127.0.0.1
                                            #  is the phone's, so it skips the API half and
                                            #  checks everything the app logged about itself)
TensorAgent/scripts/deploy-device.sh         # Debug build: auto-sign, install, and launch
TensorAgent/scripts/verify-background.sh     # send the app away mid-answer and read what happened
```

`deploy-device.sh` selects the only connected physical iPhone, an installed
`Apple Development` identity, and a compatible provisioning profile. If more
than one phone or identity is available, set `DEVICE_ID` or `CODESIGN_KEY`;
`CODESIGN_PROVISION` can likewise override profile selection. Share-enabled builds
also require the App Group `group.ai.tensorsharp.tensoragent` on both App IDs and an
independent profile for `ai.tensorsharp.tensoragent.share`; override its selection with
`CODESIGN_SHARE_PROVISION`. The containing app's exact profile must never be reused
for the extension. Set `TENSORAGENT_SHARE_EXTENSION=false` only for an intentional
app-only regression build. The install is an update in place, so existing models,
conversations, and settings are retained.
Release deployment rebuilds the native iOS xcframework from the current checkout;
set `TENSORAGENT_REBUILD_XCFRAMEWORK=0` only when intentionally reusing it.

Five environment variables drive a Debug build from a script, because neither
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
    -f net10.0-ios -r ios-arm64 -c Release \
    -p:TensorSharpIosTargets=true -p:CodesignKey="Apple Development: ..."
```

`TensorSharpIosTargets=true` must be on the command line rather than only in the
csproj: it decides whether `TensorSharp.Models` builds a `net10.0-ios` slice at
all, and restore resolves a referenced project's target frameworks before a
`ProjectReference`'s `AdditionalProperties` are applied.

## Layout

```
TensorAgent/
  scripts/          build, run and verify; prepare-python.sh; verify-skills.py
  skills/           the bundled skills, plus verdicts.json saying why each one is here
  python-runtime/   staged CPython (not in git; produced by prepare-python.sh)
  src/TensorAgent.Core/
    Catalog/        the model list, the store, install state
    Downloads/      resumable, verified downloads, and the manager that owns them
    Sessions/       conversations, and the recorder that keeps them in step with the page
    Settings/       the two sandbox switches and the rest
    Hosting/        the loopback server, the route table, and AgentAppHost
    Shell/          the in-process POSIX shell and the agent host's backend over it
    Sandbox/        ExecutionPolicy and ConfinedPaths, shared by all three runtimes
    Python/         embedded CPython and the wheel installer
    JavaScript/     JavaScriptCore over its C API, with Node-shaped globals
    WebUi/          the script appended to the Server's page
    Sharing/        durable-inbox import, bounded composer handoff, ACK lifecycle
  src/TensorAgent.Sharing/
                    dependency-free envelope format and prompt composition contract
  src/TensorAgent.ShareExtension/
                    iOS share sheet, NSItemProvider readers, Safari preprocessing
  src/TensorAgent.Maui/
    MainPage        the WebView, the attachment row, dictation
    Pages/          models, chats, settings
    Hosting/        where the files live on this device; the engine and media probes
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
copy of a public file has no business in a user's iCloud quota.

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

`scripts/verify-skills.py` decides what is bundled, by parsing every script and
resolving each import against the staged interpreter. It refuses anything reaching
for a capability iOS does not have. Eleven pass and are bundled; `skills/verdicts.json`
records every verdict.

The upstream ones that do not, and what blocks each (three more — `academy-guide`,
`discernment-nudge`, `brand-guidelines` — pass the checker and were unbundled anyway,
because they instruct the model on behalf of another product in every turn):

| Skill | Blocked by |
| --- | --- |
| docx, pptx, xlsx | the validators shell out to LibreOffice (`lxml` and `defusedxml` are now bundled; those were the other blocker) |
| pdf | `pdfplumber` is missing, and `pdf2image` shells out to poppler |
| skill-creator | `subprocess`, `webbrowser` |
| webapp-testing | `playwright` needs a browser engine |
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
build no plan at all, so no skill is declared to the model and none is reachable.
That is what the switch is for — eleven skills announce themselves in every prompt,
which on a phone is thousands of tokens on every turn of every chat, and a user who
wants a plain assistant should be able to have one. It applies to the next message,
not the next launch.

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

## Tests

```
dotnet test TensorAgent/tests/TensorAgent.Tests/TensorAgent.Tests.csproj
```

Hermetic by default. Four groups need something the machine may not have and say
so rather than passing silently:

| Set | Enable with |
| --- | --- |
| Live CPython | `TENSORAGENT_PYTHON_ROOT=<a staged slice or a CPython 3.13 prefix>` |
| End-to-end chat | `TENSORAGENT_TEST_MODEL_DIR=<a directory of catalog GGUFs>` (and `TENSORAGENT_TEST_MODEL_FILE` for a differently named copy) |
| Metal lifetime | the same weights, plus a Mac whose GgmlOps was built with ggml_metal |
| Media parity | the desktop provider's packages |
| The open web | `TENSORAGENT_ALLOW_NETWORK_TESTS=1` — these ask the real internet a real question |

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
on every load, and each value is measured: caches start at 2,048 tokens and grow;
a request pre-reserves at most 1,024 tokens of reply beyond its prompt, in 2,048-token
steps; one finished conversation stays resident, plus the shared-prefix checkpoint;
nothing is parked. (The reply length setting used to decide the reservation: at its
top rung, 262,144 tokens, every request reserved the whole 32k window, host copy and
Metal mirror both.) ggml-metal's residency set is off on the phone, so the weights can
be reclaimed while a tool runs. TensorSharp shares complete flash-attention
workspaces after their final consumers finish, reducing the persistent graph's
allocation while building unchanged upstream ggml (see
[allocation and benchmark details](../docs/perf/ggml-without-patches.md)). The memory warning now asks the engine to release what only
serves the next request's speed, on the engine's own thread between steps.

Measured on the Mac with the phone's settings and the research-then-slides prompt
that was killing the app (`--scenarios agentic --network`), footprint at the end of
the turn: Qwen3.5 9B IQ4_XS 3.7 GB before, 1.7-1.8 GB after; Bonsai 27B Q1_0 9.7 GB
before, 4.0-5.8 GB after (its recurrent state is 216 MB per resident copy). On the
iPhone 17 Pro Max with the user's own settings, the same prompt runs past 24,000
tokens of context at a 1.7 GB footprint where it used to die.

### The first message of a launch

TensorAgent uses the radix KV prefix cache by default, through the same engine as
TensorSharp.Server and TensorSharp.Cli. Prefix lookup respects the model's cache
capabilities, conversation scope, and media boundaries. The phone's existing
retention limits still apply, and memory warnings release idle cache payloads.
`TS_PREFIX_CACHE_MODE=legacy` selects the compatibility path for diagnosis;
`TS_SCHED_PREFIX_CACHE=0` disables runtime prefix reuse.

Every chat starts from a copy of the model's state at the end of the prompt they all
share, but in a fresh process that state has to be made first: the warm-up after a
load prefills it, and on the phone that is 36-48 s for Qwen3.5 9B and 152 s for
Bonsai 27B, which a first message sent sooner pays in full. The checkpoint is now
written to `Library/Caches/TensorAgent/prefix-cache/<model id>/` the first time it is
taken (`PrefixCheckpointFileStore`) and read back by the next load
(`IPrefixCheckpointStore`, consulted by the engine at admission), so the first
message of every later launch clones it like any other new chat. A file is named
and checked by the model's K/V identity and the exact prefix tokens, written under a
temporary name and renamed, and at most three are kept per model; one that no
longer describes its model is deleted and the prefix is prefilled and saved again.
Deleting a model deletes its checkpoints. The same idea as llama.cpp's prompt-cache
files, scoped to the one prefix the app cares about. `benchmarks/TensorAgentTtftBench
--scenarios restore`, run twice against the same `--root`, measures the difference;
on the Mac (M5 Pro), first message of a launch, no warm-up waited for:

| Model | Cold launch | Next launch | Checkpoint file |
| --- | ---: | ---: | ---: |
| Qwen3.5 9B IQ4_XS | 5.51 s | 0.41 s | 102 MB, restored in 39 ms |
| Bonsai 27B Q1_0 | 16.43 s | 0.84 s | 253 MB, restored in 127 ms |

On the iPhone 17 Pro Max, Qwen3.5 9B: the first launch wrote 117 MB in 386 ms after a
54 s cold first message; the next launch restored it in 44-284 ms and the warm-up (a
full first request, one-time graph builds included) took 1.2 s on the Release build
where it took ~40 s before. A message sent after that warm-up starts in ~0.6 s.

`PrefixCheckpointExactnessTests` proves the restored copy is the model: a chat started
from it produces the same tokens as a cold prefill, on Metal, for Qwen 3.5 and Gemma 4.

### Speculative decoding

Every turn is decoded speculatively unless the "Speculative decoding" switch in
Settings is off: a drafter guesses a few tokens ahead and the model verifies them in
one batched forward, so the answer is exactly what plain decoding would have
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

The switch applies at once: the engine keeps running and follows the new policy on
the next turn (`InferenceEngine.UpdateSpeculation`). To measure it on the phone,
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
settings (catalog context and K/V budget, 1024-token solo prefill chunks, all eleven
skills), loads a catalog model on Metal the way tapping "Use" does, and drives
`/api/chat` exactly as the page does through every shape a conversation takes. It
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
has no Metal, and the prompt is large because all eleven skills declare themselves.

## What has not been verified

Stated plainly, because the rest of this file is written as though everything was
checked and these were not:

- **Downloading in the background.** The manager, the routes and the resume are
  covered by tests that move real bytes. The iOS half — the background-task assertion
  and the resume on `willEnterForeground` — is compile-verified only: neither can be
  exercised in a simulator that is never suspended, and how long iOS actually grants
  is a property of a real device under real memory pressure.
- **The native picker's own file names.** `UploadNaming` is tested against the shapes
  iOS produces (a stem with no extension, no content type, HEIC and MP4 bytes behind
  the same absent name), but the picker itself has only been run by hand.
- **Image editing.** `/api/image-edit` remains bound to the same service the desktop
  uses, but the built-in catalog no longer offers a Qwen-Image-Edit checkpoint and
  no image has been generated on iOS.
- **Video generation.** The routes exist because they are part of the shared
  surface. No video model is small enough for the catalog, so nothing offers one.
- **Package installation.** `WheelInstaller` refuses without the network switch and
  accepts only pure-Python wheels; the accepting path has not run on iOS. A request
  for a package the bundle ships (numpy, Pillow, lxml, python-pptx, python-docx, …)
  never reaches it: the installer answers for the bundle first.

- **The first-token numbers ON THE PHONE after the 2026-09-07 cache work.** Every
  figure in "Every conversation shape, on Metal" is from a Mac driving the real app
  host; the phone was not reachable that day. The Debug build carries a probe for
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
ok voice-switch-gone                     ok holding-the-box-gives-hold-to-talk
ok reasoning-is-a-setting                ok the-keyboard-button-returns
ok skills-moved-to-the-menu              ok the-menu-comes-from-the-left
ok activity-above-the-box                ok the-menu-lists-the-saved-chats
ok a-tap-still-types                     ok a-turn-can-be-taken-back-up
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
