# Running browser and native-runtime skills

TensorSharp runs skill scripts through the same execution backend used for shell
commands. The Playwright bundle in `TensorAgent/skills/playwright` includes
TensorSharp setup and account-handoff guidance. Its wrapper pins
`@playwright/cli@0.1.21` and prefers npm's cache to avoid repeated registry checks.
There is no browser bridge, special tool, or skill-name dispatch in the runtime.

## Desktop server

Run a tool-capable local model with skill scripts, shell commands, package
installation, and network access enabled explicitly:

```sh
dotnet run --project TensorSharp.Server.Host -c Release -- \
  --model /path/to/model.gguf --backend ggml_metal \
  --host 127.0.0.1 --port 5037 \
  --skills-dir "$PWD/TensorAgent/skills" \
  --skills-allow-exec --skills-allow-network --skills-max-rounds 32 \
  --code-exec --code-exec-allow-install --code-exec-allow-network \
  --code-exec-timeout 180
```

Use the appropriate model backend for the machine. Open WebUI Chat, select the
skill, and ask for a concrete browser task. Discovery can also select a skill by
its description. Without an explicit skills directory, conventional repository
`.agents/skills` roots are searched from the current directory to the nearest Git
boundary, nearest first. On the server, the `skills` directory next to the
binary is also the upload directory. It is always scanned first, so it wins a
name clash, and it is kept even with an explicit `--skills-dir` or
`TS_SKILLS_DIR`, which replaces only the `.agents/skills` defaults. Personal skills are not implicitly imported into a server.

The model reads `SKILL.md`, then calls `skills_run` with the actual skill id and
bundle-relative script path. For example:

```json
{"skill":"playwright","path":"scripts/playwright_cli.sh","args":["snapshot"]}
```

Paths such as `$CODEX_HOME/skills/...` in third-party documentation describe that
agent's installation layout. The host resolves bundled scripts without copying
or rewriting them. Arguments are passed separately from the executable.

## Runtime setup and macOS

Node.js, npm, and npx must be real executables. Check them using the shell tool.
Desktop hosts use the installed runtime when available. Tools can also be
installed under the session's `$HOME/.local/bin`, which is on the shared shell
and skill-script PATH. This avoids global writes and allows a later tool call to
reuse the installation. Network access must be enabled for direct downloads;
package-install permission alone does not enable general command networking.
The host package installer accepts npm registry specs including scoped names and
versions, and continues to reject URL, Git, local-path, and alias requirements.

macOS does not allow a child already confined by Seatbelt to initialize a second
Seatbelt sandbox. Applications that do this need their documented compatibility
setting. The bundled skill now instructs the model to create the following
**workspace project config** through `write_file` before opening the browser on
macOS under `sandbox-exec`. The shell tool exposes the actual host platform and
configured sandbox so the model can make that decision:

`.playwright/cli.config.json`

```json
{"browser":{"launchOptions":{"chromiumSandbox":false}}}
```

This config disables Chromium's inner sandbox. TensorSharp's required outer
Seatbelt sandbox remains enabled. It is ordinary CLI configuration, not an
automatic host override. An existing config is patched rather than overwritten.
Close the browser session when done, except during a pending login handoff.

The macOS policy supports native event loops, child-process signalling, local
IPC, and the exact `IOSurfaceRootUserClient` IOKit client needed to present
desktop windows. Blocking that client can leave a headed Chrome window completely
transparent even though navigation and screenshots work. This allowance applies
with networking enabled or disabled; it grants no general IOKit/GPU access.
After updating the host, restart it and open a new browser session: a running
browser retains the sandbox policy inherited when it launched.

A short session temporary path avoids Unix socket address-length limits.
macOS native applications also use the OS-selected per-user temporary directory;
that exact directory and `/private/tmp` are shared writable compatibility areas.
Home access remains restricted. Seatbelt is inherited by descendants, but a
deliberately detached process can outlive a tool call; this is not a guarantee of
per-command process-tree termination or isolation between hostile users.

## Using your account

The server's workspace HOME has no desktop browser cookies. `--persistent`
saves the automation browser's own profile; it does not attach to normal Chrome
or import the user's account. The skill verifies the visible account and reuses
the running session with `goto`. For any browser interaction, the agent asks for
missing information or choices, uses the supplied answer to fill/select the
relevant controls, verifies the result, and continues the task. This covers
account details as well as other forms and decisions; supplying values should
not lead to another instruction to fill the same form manually.

When direct user participation is still needed, keep a headed persistent browser
open and identify the page title/site. Replies in the **same chat** reuse its
workspace. A new chat does not inherit that profile. `--headed` does not prove
the window is frontmost or visible: `tab-list` and `tab-select` can select the
existing tab without restarting its profile. A remote server needs an accessible
browser or an explicitly supplied endpoint for direct user interaction.

Ordinary chat, tool arguments, SSE traces, and validation reports are not a
protected secret-input channel; the host currently records their contents.
Use synthetic credentials in validation. For a real account, prefer direct
password entry or a supported protected input path, while the agent fills the
other provided values and performs the authorized steps. Do not claim that
passwords supplied through ordinary chat are excluded from logs.

Reddit can display a "Prove your humanity" challenge even after browser launch
succeeds. This requires a user handoff; an unauthenticated page or CAPTCHA is not
a successful account search. The model should report the observed blocker,
avoid speculative summaries, and resume after the user completes the handoff.

## TensorAgent

`AgentAppHost` uses the shared process backend on macOS/Linux/Windows and its
embedded backend on mobile. `AgentPaths.ExecutionMode` can explicitly select
`InProcess` for mobile emulation. Native process execution refuses a configured
network host allow-list it cannot enforce; it does not silently broaden it.

The shipped MAUI UI is iOS-only. Its embedded JavaScript engine is not Node.js,
and iOS cannot spawn npm, Chromium, or arbitrary native child processes. This
skill therefore cannot run locally on physical iOS under the no-bridge
constraint. The app bundle leaves it out: `TensorAgent.Maui.csproj` excludes
`playwright` (and `web-artifacts-builder`, whose scripts need pnpm, npm and
parcel) by name, so the app carries ten of the twelve skills while the repository's
`TensorAgent/skills` directory keeps all twelve for the desktop hosts.
Desktop-host validation is not an iOS-device pass.

Linux's required bubblewrap backend gives each command its own PID namespace.
Detached CLI daemons cannot be assumed to survive across calls. A persistent
session namespace would be needed for that workflow; changing process lifetime
or disabling the sandbox implicitly is not an acceptable substitute. Windows
currently lacks required filesystem/network confinement in the shared process
backend. Neither platform is covered by the macOS browser validation.

## Repeatable validation

The integration runner serves a local page with JavaScript-rendered data, asks a
real model through `/api/chat` to read it, edit and submit form fields, navigate,
and save a screenshot. An independent HTTP oracle verifies the submitted data,
browser navigation, successful tool events, and downloadable PNG bytes. Every
SSE event is saved immediately so failures remain reviewable.

```sh
python3 eng/validation/validate-browser-skill.py \
  --base-url http://127.0.0.1:5037 \
  --output artifacts/browser-validation/server
```

Use a fresh output directory for every run. The default run must discover setup
from the skill, without a configuration hint in the user's task.
`--macos-browser-config` remains available to reproduce earlier configured
tests; those tests do not establish unassisted setup.
`--discover` omits explicit skill
selection. `--think` enables model reasoning. For the desktop TensorAgent host:

```sh
dotnet run --project eng/validation/TensorAgentHost -- \
  --root artifacts/browser-validation/app \
  --skills "$PWD/TensorAgent/skills" --weights /path/to/model.gguf \
  --network --port 5038

python3 eng/validation/validate-browser-skill.py \
  --base-url http://127.0.0.1:5038 \
  --macos-browser-config \
  --connection artifacts/browser-validation/app/connection.json \
  --output artifacts/browser-validation/tensoragent
```

Keep generated logs, JSON reports, and screenshots in ignored `artifacts/` or
`docs/validation/`. A screenshot or successful CLI command alone does not prove
model-driven task completion. A failed or unavailable device scenario is not a
pass. Timings from one local page are workflow observations, not a broad browser
or model throughput benchmark.

## Codex comparison

Reviewed upstream [`openai/codex`](https://github.com/openai/codex/tree/5c5308fc9a9ee789049d646ef11e5400384b9c6f)
at revision `5c5308fc9a9ee789049d646ef11e5400384b9c6f`, alongside the official
[skill documentation](https://learn.chatgpt.com/docs/build-skills) and
[shell/skills guidance](https://developers.openai.com/blog/skills-shell-tips).
The applicable patterns are bounded metadata discovery, reading selected
instructions on demand, resolving bundled resources through their host,
reusing session setup, deterministic file edits, inspecting actual results, and
reporting execution limitations accurately. TensorSharp already supplied
progressive disclosure and structured file/patch tools; these changes improve
their portability and the execution environment rather than adding a parallel
browser integration.

## Validation observed on September 20, 2026

On macOS arm64 with Qwen3.5-9B-IQ4_XS and `ggml_metal`, the configured
Server.Host WebUI chat workflow passed in 75.827 seconds (16 tool steps).
TensorAgent desktop passed explicit skill selection in 59.395 seconds and
catalog discovery without a skills array in 81.217 seconds (16 steps).
The later server/discovery runs overlapped; these are not isolated throughput
measurements. Validation exercised the real `/api/chat` backend, not automated
clicking of the chat UI itself. The form oracle and downloaded screenshots
confirmed browser interactions independently of the model's final prose.

Unconfigured model runs did not reliably recover the macOS nested sandbox
requirement and are retained as failures. Successful runs were supplied the
normal project configuration described above. Model recovery from invalid CLI
flags cost extra rounds; this is not a claim of uniformly optimal tool use.

Node v22.22.0 was downloaded and unpacked inside a required sandbox with the
bootstrap PATH excluding host Node. A later call selected this local runtime,
ran npm/npx 10.9.4, installed `is-number@7.0.0` through the host installer, and
imported it. Host Node was also installed; different version output verified
that the session runtime was selected.

The ggml checkout was clean at
`456172ec733a135778adcd32d00e576a58232e45`. At that earlier validation, Playwright skill file hashes
matched their initial values. Reusable validation lives in `eng/validation`;
per-run evidence and test results are under ignored `artifacts/playwright-validation`
and `artifacts/browser-skill`. Final shared regressions passed 881 tests with
zero failures or skips. The final TensorAgent suite passed 786 tests, skipped
104 unavailable/opt-in scenarios, and failed none. Platform/device scenarios outside the stated coverage
are not counted as passing.

## Follow-up: Qwen3.8-27B Reddit failure

The September 20 request `c7ca6ddab687` selected and read the skill correctly,
then failed on Chrome launch with `Session closed`. Seventeen generations spent
403.7 seconds retrying and investigating CLI source before cancellation. Median
KV reuse was 96.1%; a second long request was running concurrently, so this is
not an isolated model throughput measurement.

A direct reproduction with required Seatbelt confirmed the cause: without the
project config, the CLI daemon's `DEBUG=pw:browser` log reports sandbox
initialization failures. With `chromiumSandbox:false`, launch, later commands,
and headed persistent profiles succeed. Reddit then returns a verification
challenge. Runtime success does not establish authenticated Reddit task success.

The follow-up changes move the required setup into the skill, correct the stale
configuration filename in its reference, document login/verification handoffs,
and instruct the agent to ask for missing information and wait before dependent
actions. The shell declaration supplies stable environment facts. Generic
recovery guidance discourages unchanged retries and speculative dependency
debugging. The CLI wrapper is version-pinned and prefers its cache.

Live testing also found a separate argument transport defect: Qwen can emit a
JSON array inside a string-typed XML tool parameter. `skills_run` now decodes
that array before launching the script, preserving argument boundaries and empty
strings without shell evaluation. Invalid encoded vectors return an actionable
error. A regression passes the exact streamed Qwen XML through parsing and
dispatch. Screenshot guidance names the supported `--filename` option.

The authenticated fixture uses two Chinese requests and a synthetic local
account. The first signs in; the second reuses the same browser session, searches
for `codex reset`, opens the relevant posts, and returns their links and fresh
facts. Its independent HTTP oracle checks authentication and browser navigation;
a plausible final answer alone cannot pass. Run it against the configured server:

```sh
python3 eng/validation/validate-browser-skill.py \
  --base-url http://127.0.0.1:5001 --scenario authenticated-search \
  --discover --temperature 0.3 --think \
  --output artifacts/browser-validation/authenticated-search
```

No browser configuration is injected into this task. Credentials belong only to
the synthetic fixture; this test does not establish access to a user's desktop
profile or bypass Reddit verification. Generated follow-up evidence is under
`artifacts/playwright-fix-20260920/` and `artifacts/playwright-e2e-fix/`.

Use `--scenario account-handoff` to test the same kind of account request without
providing credentials. The oracle requires reaching the login page without
submitting guessed credentials or reading private posts. Its expected exit is
2 with `handoff_requires_review`, not an authenticated-task pass. Review the
final question and recorded successful CLI commands to confirm a headed,
persistent browser was left open for the user. The fixture stops after recording
the response; this mode validates the handoff, not interactive human sign-in.

Follow-up checks used Qwen3.8-27B-UD-Q4_K_XL, `ggml_metal`, q8_0 KV cache,
temperature 0.3 and a per-generation validation cap of 4096 tokens. No setup
hint or explicit skill selection was supplied. On the Release build used for
these runs:

| Local workflow | Thinking | Outcome | Seconds | Tool calls |
| --- | --- | --- | ---: | ---: |
| Form, navigation, screenshot download | Off | Passed, no failed calls | 106.671 | 14 |
| Login followed by authenticated search and cited summary | Off | Passed, no failed calls | 164.387 | 18 |

An earlier thinking-enabled login/search run also passed (594.265 seconds,
25 successful calls), before the final argument and prompt refinements. Thinking
and those refinements both differ between runs; this is not a controlled
comparison or a claim of a general speedup. The initial form run took 137.953
seconds and needed two failed-call recoveries; the final run accepted the model's
JSON argument vectors directly and used the correct screenshot option. For
routine browsing, the WebUI's Thinking toggle can avoid extended planning;
thinking remains under the caller's control and is not silently disabled.

The shared regression run at this stage passed 985 tests, zero failed/skipped;
it preceded the later continuation and clarification refinements. TensorAgent
compatibility coverage: 215 passed, zero failed, 43 skipped (unavailable
embedded/staged Python, network opt-ins, or unconfigured live-model tests).
The validation oracle's 19 tests passed. Skipped scenarios are not passes.
No native code changed; upstream ggml remains clean at
`456172ec733a135778adcd32d00e576a58232e45`. The CLI used Node 26.8.1,
npm 11.19.0, `@playwright/cli` 0.1.21 and its pinned Playwright dependency
`1.64.0-alpha-1789764292000`. The CLI's
[session behavior](https://github.com/microsoft/playwright-cli#sessions) is also
documented upstream. These synthetic local macOS checks do not establish Linux,
Windows, iOS, or authenticated Reddit retrieval and summarization.

The corrected real-Reddit handoff used the original Chinese request with Thinking off.
It completed five successful tool calls in 49.592 seconds, opened a headed
persistent browser, observed Reddit's reCAPTCHA page, and asked in Chinese for
manual verification, sign-in, and confirmation before continuing. It left the
browser open and did not click CAPTCHA controls or invent a post summary. This
validated the request for missing user input; at that stage, authenticated Reddit
retrieval and summarization were pending the user handoff. The
recorded request, response, command trace, and page snapshot are in
`artifacts/playwright-fix-20260920/reddit-handoff-final/`.

The first continuation after the user replied ready exposed another behavioral
failure: fresh subreddit and search snapshots still displayed `Sign Up` / `Log In`,
but the model proceeded to public results. The validation client cancelled that
attempt after 165.374 seconds and retained the browser for actual sign-in; this
is not authenticated-task success. The skill now requires a fresh account check
after handoff confirmation and stops an account-required search when visible
state still indicates sign-out. A confirmation is not proof of browser state.

That attempt also identified a concrete prompt-processing cost: reading the
subreddit snapshot expanded the prompt from 9,932 to 26,317 tokens (76.841 seconds
to first token); the search snapshot expanded it from 26,725 to 33,901 tokens
(42.898 seconds). The skill now starts with bounded snapshot sections and locates
relevant text before reading more. These observations motivate the change; they
are not a measured speedup for the revised guidance.

A later 115 focused tests passed with zero failures/skips after fixing canonical
argument precedence and making handoff replies resume the pending task using
conversation context. Of their test names, 102 overlap the earlier shared run
and 13 are new; the two pass counts must not be added as distinct test coverage.
The actual Reddit continuation retained the original tool transcript; abbreviated
`chat.start` logs describe the incoming messages, not the augmented model history.


Use `--scenario account-handoff-continuation` for a second literal `ready` reply
while the synthetic browser remains logged out. Its oracle rejects credential
submission and premature search, and requires a fresh browser observation in the
second turn. The result still requires review of the observed page, actionable
question, and retained browser. Continuation requests replay the complete streamed
assistant answer for tool-history matching; final prose alone is used for summary
quality checks.


The revised continuation fixture completed in 74.036 seconds with nine successful
tool calls (41.456 seconds/five calls for initial handoff, then 32.512 seconds/four
calls after `ready`). Review confirmed that the browser still displayed its login
form, the model asked in Chinese for sign-in in the correct window, and it neither
submitted credentials nor attempted search. It left its browser open; the harness
subsequently removed that synthetic session. A tab check and screenshot remain
in the continuation, so this does not establish optimal tool use. Evidence is in
`artifacts/playwright-fix-20260920/handoff-continuation-final/`.

An isolated managed Release build succeeded with zero warnings/errors in
`artifacts/playwright-fix-20260920/continuation-host/` output. The original server
and Reddit browser were retained to preserve the pending real-account handoff;
that running process predates the final prompt/argument-precedence refinements.
The latest skill was loaded in fresh fixture sessions. At the time of that run,
authenticated Reddit retrieval still awaited sign-in and was not marked complete.


The general clarification workflow is also exercised with
`--scenario missing-form-info`: first omit the form details, then supply synthetic
values in a second request. The model must fill and submit them through the
browser, verify the account, and finish the original search and cited summary.
The observed run passed the independent browser/fact checks and human review in
192.434 seconds with 22 successful calls, no failed calls. The first response
asked for the missing values; the second used them and continued the task.
Evidence is under `artifacts/playwright-fix-20260920/missing-form-info/`.

The generic skill prompt now applies this ask/use/verify/continue pattern to
missing choices and form values beyond login. Its 23 focused tests passed; all
23 test names overlap the earlier shared run. These are targeted reruns, not
additional distinct tests. The latest isolated managed build
(`clarification-host/`) succeeded without
warnings or errors. The synthetic workflow establishes functional form handling;
it does not establish a private channel for real passwords.


For an owned local validation session whose window is inaccessible,
`eng/validation/browser-secret-handoff.py --help` describes a one-use loopback
password-entry bridge. It exposes a random local form URL, checks the intended
site/path/account, then fills and submits the existing browser form. The real
password is fetched once into the CLI process rather than embedded in its
arguments or routed through model chat. The helper suppresses subprocess output,
does not write the value to artifacts, rejects reuse, and expires after ten
minutes. Submission is reported separately from authenticated login.

Use only an inspected, trusted local browser session with tracing/debug/HAR/video
capture disabled. Browser, website, and OS recording are outside the helper's
control. After submission, inspect URL/title and visible text without reading
password values; resume full snapshots only after the password form is gone.
This is a validation utility, not a new protected-input facility in the chat UI.


The bridge's 19 focused tests passed. Independent actual-browser checks also
passed for ordinary labels and Reddit-style accessible-role fields with a submit
button that starts disabled. Wrong site/path/account stopped before secret
retrieval; the expected credentials produced exactly one authenticated fixture
submission. No password appeared in the inspected helper arguments, output, or
artifacts. These checks used only synthetic credentials and isolated sessions;
evidence is in `artifacts/playwright-fix-20260920/secret-bridge-independent-st9sr95f/`.

Native submission of the local entry page exposed a separate frontend defect:
`no-referrer` caused the browser to send `Origin: null`, so the helper correctly
rejected its own form. Both the response header and HTML meta policy now use
`same-origin`; exact Origin validation remains enforced. This follows the
[Fetch Origin-header algorithm](https://fetch.spec.whatwg.org/#append-a-request-origin-header).
An isolated browser reproduced the 403 and verified the corrected native form
with one authenticated fixture submission. The earlier browser checks supplied
the local POST programmatically and did not cover this frontend behavior.
Evidence is in `artifacts/playwright-fix-20260920/secret-native-form-f1qntbwg/`.

The reusable native-form regression passed against the fixed helper without
policy overrides (one test, 4.405 seconds). It uses two isolated browser sessions,
synthetic credentials, and the actual HTML submit control. Run it with an already
installed pinned CLI; ordinary test discovery reports this opt-in test as skipped:

```sh
TENSORSHARP_BROWSER_HANDOFF_TEST=1 \
TENSORSHARP_PLAYWRIGHT_CLI=/absolute/path/to/node_modules/.bin/playwright-cli \
python3 -m unittest discover -s eng/validation/tests \
  -p test_browser_secret_handoff_browser.py -v
```

The successful run's evidence is in
`artifacts/validation/native-secret-handoff-9q6hj7wc/report.json`.

A fresh audit run of the validation oracle passed all 19 tests with no skips;
its log is `artifacts/playwright-fix-20260920/oracle-tests-audit.log`. The older
`oracle-tests-final.log` contains the earlier 17-test run.

After the corrected local entry form was submitted, independent browser inspection
confirmed account `u/fuzhongkai` and the absence of password fields. The model then
independently verified the visible account, searched r/OpenAI, visited seven posts,
read their captured content, returned seven matching links in Chinese, and closed
its browser. All 20 tool calls succeeded in 523.909 seconds. Metadata and numerical
anecdotes were checked against captured pages. Evidence is under
`artifacts/playwright-fix-20260920/reddit-signed-in-final/`. The initial snapshot
in this run still used inline output, before the wrapper fix described below.

Retrieval passed, but the first summary overstated some author/commenter claims
as policy or consensus and conflated a week-to-week usage comparison with a
model-to-model comparison. Source-attribution guidance was tightened. A separate
guided text revision took 103.145 seconds and improved attribution, but still
claimed an unsupported ordering and absence of substantive comments where the
snapshot had not captured their content. These are recorded quality limitations,
not an unassisted summary-quality pass. The delivered
`artifacts/playwright-fix-20260920/reviewed-summary.md` corrects those claims through
editorial review; raw model outputs remain in the evidence directories.

At that stage the retained server used the earlier runtime build; the latest
managed refinements had separate build and focused-test evidence. It was retained
to preserve the authenticated test session, since restart loses in-memory chat
state and startup orphan cleanup removes the prior owner's workspace/profile.
That process had exited before the subsequent window-visibility validation.

The signed-in continuation exposed an additional snapshot-output issue: unlike
navigation commands, the pinned CLI's bare `snapshot` returns its entire tree
inline. The wrapper now supplies a unique snapshot filename when none is given,
preserving explicit filenames. The model can read bounded sections of the saved
file without receiving an entire feed first. Account checks also distinguish
site account controls from `Sign Up` text inside advertisements or posts, and
the final guardrail separates ordinary form filling from CAPTCHA handoff.

Eight wrapper argument tests passed. On the same isolated 600-row browser page,
bare CLI output was 171,341 bytes and wrapped output was 147 bytes (99.914% less).
The complete 171,231-byte snapshot retained all 2,403 element references, and
clicking its last-row reference succeeded. Repeated snapshots used unique paths;
explicit filenames were preserved. Evidence is under
`artifacts/playwright-fix-20260920/snapshot-output-final-a4_7_er5/`. This measures output
size and snapshot integrity, not model-token savings or total task latency.

## Follow-up: invisible headed Chrome on macOS

A controlled comparison through the same `ShellRunner`, CLI version, scrubbed
environment, and local test page isolated another sandbox defect. With required
Seatbelt, Chrome opened a native window on an active display but CoreGraphics
reported its alpha as **0**. Browser automation still succeeded. With sandboxing
off, the corresponding window had alpha **1**. The confined browser log reported
`Failed to allocate IOSurface`; changing window position did not fix it.

Allowing only `IOSurfaceRootUserClient` in the required Seatbelt profile removed
the allocation failures and produced alpha **1** at the same window bounds. The
user then confirmed that the sandboxed test window was visible on this Mac's
local desktop. Filesystem and network restrictions remain in place. The browser
still logged EGL initialization failures before software rendering fallback;
this establishes desktop visibility, not hardware GPU acceleration or a browser
performance improvement.

The native regression compiles a small trusted fixture, then creates, locks,
writes, unlocks, and releases a 16×16 IOSurface inside the required sandbox with
networking both disabled and enabled. Together with existing confinement,
native-runtime, and profile tests, **80 tests passed, none failed or skipped**.
The normal Release server build succeeded with zero warnings/errors. These are
targeted checks for the graphics change; earlier model workflow counts above
were not rerun or added to this total.

To reproduce with a harmless, separately owned browser session:

```sh
dotnet run --project eng/validation/DesktopBrowserProbe -c Release -- \
  --sandbox required --output artifacts/browser-visibility/new-run --hold
```

The output directory must be fresh. The probe prints its local URL and session
identity, retains the window, and closes only its own session when the printed
`stop` file is created or Ctrl+C is received. Keep this debug-enabled probe on its
synthetic page. To inspect the browser PID from its launch log without capturing
other applications or screenshots:

```sh
swift eng/validation/macos-window-visibility.swift --pid <browser-pid>
```

The helper reports application state, window opacity, bounds and display
intersection. Window titles are omitted unless `--include-titles` is requested.
OS metadata alone cannot establish what a user sees; pair it with actual desktop
confirmation. Comparison evidence and the Release build log are under
`artifacts/playwright-window-visibility/`; the 80-test result is
`artifacts/browser-visibility-iosurface-tests/results/confinement-and-surface.trx`.
