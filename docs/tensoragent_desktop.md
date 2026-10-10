# TensorAgent Desktop user guide

[English](tensoragent_desktop.md) | [中文](tensoragent_desktop_zh-cn.md)

TensorAgent runs AI models on your computer. This guide takes you from downloading
the Mac or Windows application to your first chat, attachments and optional code
tools. You do not need to build TensorSharp or install the .NET SDK to use the
desktop packages. Model weights are downloaded separately inside the app.

**[Download TensorAgent Desktop — latest release](https://github.com/zhongkaifu/TensorSharp/releases/latest)**
· [All releases](https://github.com/zhongkaifu/TensorSharp/releases)
· [Source builds and measured coverage](../TensorAgent/README.md)

Expand **Assets** on the release page and choose a file starting with
`tensoragent-desktop-`. Desktop packages are published by the updated **Release
Binaries** workflow; releases made before that workflow change may only contain
CLI/server archives. If the latest release has no Desktop assets, there is no
Desktop download for that release yet: use the [source-build instructions](../TensorAgent/README.md#on-the-desktop-macos-and-windows)
or return after the next packaged release. A CLI or server archive is a different
application.

## 1. Choose your download

`<version>` below is the release tag without its leading `v`. For example, a tag
`v2026.10.03` would use `2026.10.03` in the filenames; this example does not assert
that such a release exists.

| Your computer | File to choose | Other formats |
|---|---|---|
| Apple Silicon Mac (M1 or later), macOS 14 or later | `tensoragent-desktop-<version>-osx-arm64.dmg` | `.pkg` installer or `.zip` app bundle |
| Windows x64, CPU / AMD / Intel graphics | `tensoragent-desktop-<version>-win-x64-cpu.msi` | `.zip` portable folder |
| Windows x64 with a compatible NVIDIA GPU and driver | `tensoragent-desktop-<version>-win-x64-cuda.msi` | `.zip` portable folder |

The Windows CPU package runs inference on the CPU; it does not include a Vulkan
backend. The CUDA package includes CUDA runtime libraries, but the NVIDIA driver
must be installed separately. Use the CPU package if you do not have a compatible
NVIDIA GPU. The Mac package uses Metal with CPU fallback. Intel Mac and Windows
ARM64 Desktop packages are not produced by this workflow.

Windows builds declare Windows 10 version 1809 (build 17763) as their minimum.
Use a Windows version that still receives updates, such as a supported Windows 11
installation, and install the [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)
if it is missing. The app carries its .NET, Windows App SDK and Visual C++ runtime; WebView2
is a separate prerequisite. See Microsoft's [WebView2 requirements](https://learn.microsoft.com/en-us/windows/apps/develop/ui/controls/webview2).

**Memory and storage:** the current catalog starts at **12 GB physical system RAM**
(unified memory on a Mac). An 8 GB machine has no eligible built-in model.
Individual entries need 16, 24, 32 or 48 GB; the Models page explains each limit.
GPU VRAM and free disk space are separate requirements. Allow enough free space
for the application, the download size shown on the model card, and generated
files. The smaller chat models still download roughly 5–6 GB with their companion
files; image and video models need much more. Start with an eligible small chat
model before trying media generation.

Each platform also publishes
`SHA256SUMS-tensoragent-desktop-<version>-<variant>.txt`, where `<variant>` is
`osx-arm64`, `win-x64-cpu` or `win-x64-cuda`. Download the matching checksum file.
To check a package, replace `VERSION` in one of these examples with your download's
version and compare its hash with that file:

```bash
# macOS Terminal, in the folder containing your download
shasum -a 256 "tensoragent-desktop-VERSION-osx-arm64.dmg"
```

```powershell
# Windows PowerShell, in the folder containing your download
Get-FileHash ".\tensoragent-desktop-VERSION-win-x64-cpu.msi" -Algorithm SHA256
```

Earlier Mac packages used ad hoc signing without Apple notarization. The updated
Mac release pipeline requires Developer ID signing and notarization, but publishing
a corrected download requires a successful run with the maintainer's Apple
credentials. Existing release assets are unchanged; check the release notes and
`BUILD.txt` for the package you downloaded. Windows installers remain unsigned.
A matching hash checks the download against the release's checksum; it does not
replace a trusted publisher signature.

## 2. Install and open

### macOS

1. Download the **DMG**, open it and drag **TensorAgent.app** onto **Applications**.
2. Eject the disk image. Open **TensorAgent** from Applications or Spotlight.
3. If an older, unnotarized download shows **“Apple could not verify TensorAgent is
   free of malware”**, verify its source and checksum. For a copy you trust, try
   opening it once, then go to **System Settings → Privacy & Security**, scroll
   down and choose **Open Anyway**. Choose **Open** in the confirmation. This is
   Apple's [procedure for an app you trust](https://support.apple.com/en-us/102445)
   and creates an exception for this app. A future signed and notarized release
   should use the normal downloaded-app confirmation instead.

This procedure lets you open an existing trusted copy; it does not notarize it or
replace the download. If **Open Anyway** is unavailable on a managed Mac, ask your
administrator. If the alert instead says the app **will damage your computer** or
is **damaged**, use Apple's linked guidance and obtain a fresh trusted download.

The **PKG** is an alternative: open it and follow Installer to install
`/Applications/TensorAgent.app`; macOS may request administrator authorization.
For the **ZIP**, extract it and move **TensorAgent.app** into Applications. Choose
one format; they contain the same application. Open the installed copy, rather
than running it from the mounted DMG.

### Windows

1. Download the **CPU** or **CUDA MSI** from the table above. Open it and complete
   the installation for your user account.
2. Launch **TensorAgent** from the Start menu. The installer puts the app under
   `%LOCALAPPDATA%\Programs\TensorAgent`.
3. If Windows displays an unknown-publisher warning, verify the source and checksum
   before continuing. On a managed PC, follow your organization's installation
   policy if it blocks unsigned applications.
4. If the chat window is blank or reports a WebView2 error, install the
   [WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/)
   and reopen TensorAgent. Choose the **x64** standalone installer for an offline PC.

For the **ZIP**, use **Extract All**, put the complete folder somewhere writable,
and open **TensorAgent.Maui.exe**. Keep its DLLs, runtime files, `webui` and
`skills` beside it. Running only a copied EXE, or opening it inside the ZIP, will
not work. The ZIP uses the same per-user data folders as the installed app.

## 3. Set up a model and send your first message

1. Open **☰ → Settings → App language** if you want to change the interface
   language. The model responds in the language you use in your message.
2. If your system drive is short on space, set **Settings → Storage → Model
   download and cache folder** to an absolute, writable folder, then choose
   **Save folder**. Do this before downloading. Changing the path leaves existing
   files in place; see [Storage and backups](#storage-and-backups).
3. On the empty chat, choose **Choose a model**, or open **☰ → Models**. Read the
   model's input capabilities, RAM requirement, license and download size.
4. Choose **Download** beside an eligible chat model, such as **Gemma 4 E2B** or
   **Gemma 4 E4B (IQ4_XS)**. Keep **Settings → Downloads → Include optional files**
   on if you want the model to understand images. Model downloads need internet
   access regardless of the separate code-tool network setting.
5. Wait for the download and verification to finish, then choose **Use**. Wait
   until loading finishes and the model name appears above the composer. The first
   load and first reply can take longer while the engine prepares its caches.
6. Type `Explain what you can help me with in three short bullet points.` and
   press **Enter** or the send button. **Shift+Enter** inserts a new line. Use
   **Stop** to interrupt a reply and **New chat** to start a separate conversation.

Interrupted model downloads retain partial files. Return to **Models** and choose
**Resume** or **Retry**. The app verifies downloaded model files before use; you
do not need to download GGUFs manually for the built-in catalog.

## 4. Use attachments, skills and generated files

**Ask about a file or picture.** Select **+ → File** or **Photo**, choose an item,
wait for the upload to finish, then ask a specific question. For an image, the
loaded model must list images among its inputs and have its vision file installed.
If the card says **Text ready · vision file not downloaded**, choose **Add vision**
or **Enable vision**, finish that download, then choose **Use** again. Audio and
video input support also depends on the model and platform; check its card.

**Find earlier work.** Use **☰ → Chats** to reopen a saved conversation. Finished
answers and artifact links are saved automatically. Open a generated file's link
to view or save it outside the app. Save documents, pictures and videos you want
to keep; artifact files live in the app's cache, which may be reclaimed.

**Dictate a message.** On Mac, hold the composer to switch to **Hold to talk** and
grant microphone/speech permission when requested. Apple dictation requests
on-device recognition where supported; otherwise Apple's recognizer may process
audio off the device. On Windows, use the operating system's voice typing
(**Windows+H**) in the message box.

**Use skills.** Open **☰ → Skills** and keep **Use skills** on for document or code
work. You can inspect a skill and enable **Use in this chat**; turning the master
switch off gives you plain chat without the skills roster. A request such as
`Create a short Word document with a packing checklist and give me the file`
needs code execution and the required tools described below.

**Make or edit an image.** On a machine with the required memory, download and
**Use Qwen-Image 2.1**. Describe the picture in the composer, or attach a photo and
describe the change. Select a region with the image editor when you want a masked
edit. Finished edits offer **Compare original** and **Edit again**. A follow-up such
as "make it brighter" changes the picture above, while a new description draws a new
picture; each picture says which, and the newest offers the other reading. Optional
**LoRA plug-ins** are under the model-name menu and have separate downloads.
**Qwen-Image 2.1 Turbo** does the same in 8 steps instead of 40, about a fifth of the
time, and shares the text encoder, VAE and vision files with Qwen-Image 2.1, so having
both downloads them once; it offers only the style plug-ins validated on it.

**Make a video.** Download and **Use MiniMax-H3**, then describe a short clip or
attach a starting photo. **MiniMax-H3 References** accepts reference photos, clips
and recordings. These are large desktop-memory workloads. Mac media generation
has recorded app measurements; Windows image/audio/video generation has not been
verified. See the [model and validation details](../TensorAgent/README.md#what-has-not-been-verified).

## 5. Configure optional code tools

Local chat and inference work without Python or Node. Desktop skills run programs
installed on your computer; the desktop packages do **not** bundle Python, Node,
npm or a browser. Install only what the task needs, using the official
[Python](https://www.python.org/downloads/) and [Node.js](https://nodejs.org/en/download)
installers. Make the programs available on `PATH`, then restart TensorAgent. The
Mac app imports your login shell's `PATH`; Windows uses the process environment.
Python-based document skills may also need packages such as `python-docx`,
`python-pptx` or `openpyxl`. A skill's instructions and any missing-tool error tell
you what is required; not every bundled skill is ready just from installing the app.

Open **☰ → Settings → Sandbox**:

| Setting | What it does |
|---|---|
| **Run code** | Allows shell commands and skill scripts. On by default; turn it off for chat without program execution. |
| **Run without a sandbox** (Windows) | Required for Windows program execution, off by default. Programs can access files and network resources available to your Windows account. Enable it only if you accept that access. |
| **Allow network access** | Allows network access for code tools, off by default. Enable for research or package downloads when needed. It does not control model downloads. Windows unconfined programs have no enforced file or socket boundary. |
| **Sub-agents** | Lets the model delegate parts of a request to helper conversations on the same model. On by default; turn it off to reduce concurrent work and memory use. |

On macOS, TensorSharp confines child processes with its Seatbelt profile: they
write within the chat's workspace and allowed temporary folders. Windows has no
equivalent confinement in this app. You can leave Windows **Run without a sandbox**
off and still chat, use supported attachments, and run local model inference.
Settings shows **Now:** with the detected execution environment, including missing
Python/Node tools. Network and execution settings apply to subsequent commands;
sub-agent changes apply to the next message.

To try document work, install its tools, choose a text model with tool support,
enable the relevant skill and permissions, and ask for a small output first.
Watch the activity above the composer, then open the resulting file link. For
browser tasks, follow the separate [Playwright guide](playwright_agent.md).

## Storage and backups

The installed program and your personal data live in different locations:

| Content | macOS | Windows |
|---|---|---|
| Settings, saved conversations, installed skills | `~/Library/Application Support/TensorAgent` | `%LOCALAPPDATA%\TensorAgent\Data` |
| Default models, LoRAs, uploads, workspaces, artifacts and logs | `~/Library/Caches/TensorAgent` | `%LOCALAPPDATA%\TensorAgent\Cache` |
| Warning/error log | `~/Library/Caches/TensorAgent/logs/errors.log` | `%LOCALAPPDATA%\TensorAgent\Cache\logs\errors.log` |

Use **Settings → Storage** to see the effective model folder and counts. To move
models, wait for downloads/imports to finish, close the app, copy the existing
`<catalog-id>` subfolders into the new folder, reopen the app and save that full
folder path. The app does not move them for you. Use **Use default folder** to
restore the default location. LoRAs and other caches remain in their normal cache
folder.

For a backup, close TensorAgent and copy its data folder plus any generated files
you want to retain. Back up the cache as well if you need the uploaded files and
artifact links in saved chats to keep working. A configured model folder is
separate and needs its own copy if you want to avoid downloading weights again.
**Models → Delete** removes a model; **Settings → Delete all chats** removes saved
conversations and their associated work. Deletion cannot be undone.

## Update or uninstall

There is no in-app updater. Download the next release's matching package and
close TensorAgent before updating. On Mac, replace the app in Applications or run
the new PKG. On Windows, run the newer MSI; CPU and CUDA installers update the
same app installation. For ZIP deployments, extract into a fresh folder and use
that whole folder to avoid mixing versions. Only run one instance at a time.

Replacing the application does not delete its separate data and model folders.
Back them up before an update; new catalogs may retire old model entries and
reclaim their downloads. Read the release notes for compatibility changes.

To uninstall, move **TensorAgent.app** to Trash on Mac, use **Settings → Apps →
Installed apps → TensorAgent → Uninstall** on Windows (or **Apps & features** on
Windows 10), or delete a ZIP deployment's folder. Personal data and caches are
retained. To remove them too, close the app and delete the TensorAgent folders
listed above and any custom model folder only after saving what you need.

## Troubleshooting

| Problem | What to try |
|---|---|
| No `tensoragent-desktop-` files in Assets | This release predates Desktop packaging. Check all releases or build from source until a packaged release is available. |
| App will not open on Mac | Confirm Apple Silicon and macOS 14+, check the checksum, and follow Apple's trusted-app procedure above. A source build's OS floor may differ from the release package. |
| Blank Windows chat window | Install/update WebView2 Evergreen Runtime, reopen the app, and confirm you extracted or installed all files. |
| CUDA package fails to start or load a model | Check the NVIDIA driver and GPU compatibility. Try the CPU package to separate a GPU problem from an application problem. Do not mix files from the two variants. |
| Model says **Too big** | Its physical RAM tier exceeds this computer's RAM. Disk space or a larger page file does not change catalog eligibility. Pick an eligible model. |
| Download stops or verification fails | Check connectivity, writable folder and free space; use **Resume/Retry**. If it keeps failing, delete that model from Models and download again. |
| An attached picture cannot be read | Check image input support, add the vision file and choose **Use** again. Wait for all uploads before sending. |
| Commands or document skills do not run | Check **Run code**, Windows **Run without a sandbox**, required Python/Node/packages and the **Now:** line; restart after installing tools. |
| A research or package task cannot reach the internet | Check **Allow network access** and any system/organization restrictions. Model download access is separate. |
| Replies are slow or run out of memory | Close other large apps, select a smaller model, try a new chat and disable Sub-agents. CPU inference and first-load warm-up can be slow. KV cache precision changes require a model reload and are not supported by every model. |

For an unresolved failure, open a [GitHub issue](https://github.com/zhongkaifu/TensorSharp/issues)
with the release version, package variant, OS, RAM/GPU, selected model, exact error
and relevant lines from `errors.log`. Review logs before sharing: they may contain
your prompt, filenames or tool output. Missing models/devices and the known media
gaps are not passing validation; the existing [coverage notes](../TensorAgent/README.md#what-has-not-been-verified)
describe what has actually been exercised.
