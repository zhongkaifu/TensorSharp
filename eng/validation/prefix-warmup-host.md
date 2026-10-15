# Owned Host prefix and startup observation

`prefix-warmup-host.py --plan PLAN --activation EXACT_ROOT_ACTIVATION` launches
one private staged Windows/CUDA Host. It never builds, stages, downloads or
stops an existing server. A wrong activation or incomplete SDK/native delivery
cannot launch a model. Plans and generated reports belong in ignored evidence
directories; preserve earlier plans when source pins change.

The launcher requires the actual complete SDK delivery, Host/Probe publication
and published Windows/WSL self-test receipts. It verifies their receipt pins,
both published closures, the native build's unchanged clean upstream revision,
the stage's complete Host closure and its production `GgmlOps.dll`. A CPU test
summary alone is insufficient. SDK22 plans additionally require the delivery's
`publication_complete`, distinct from each publish receipt's `publish_complete`.
The executing launcher and all imported helpers must equal the plan's source
pins. Receipt/source/static-binary bookends are checked again after cleanup.

The text-only launch has no mmproj, uses `--unified-memory --max-tokens 20000`,
and sets child-only `MAX_CONTEXT=32768`. Both arms use fresh per-run prefix disk
and log directories. These deployment paths do not disable or change the
default tools/skills catalog. Inherited inference overrides fail explicitly.
An existing Host/model process or occupied port prevents launch; the harness
does not interrupt it. On WDDM, `nvidia-smi` also lists normal desktop clients.
A fresh plan must explicitly bind their exact PID/birth/name/executable
identities. Unknown, new or reused identities fail before launch and at the
ready/end checkpoints. These preexisting desktop clients remain active and
their traffic is unquantified; the harness makes no idle/exclusive GPU claim.

Process birth/executable, exclusive listener and the actual loaded native DLL
are observed before calling the separately frozen `prefix-warmup-http.py`.
The workload is three new chats FF7/Time/FF7, greedy think-off max16. A baseline
may observe cache state without treating it as passing reuse; a candidate can
require positive cached tokens in its first new chat as well as the HTTP
repeat-cache and literal output/finish/usage gates. Optional cancellation and
`client_visible_follow_up` are explicit in a fresh plan. The latter requires
the separately qualified HTTP mode described in `prefix-warmup-http.md`; no
history provider, manufactured raw IDs or private state identity is assumed.

Startup wall covers process launch through model readiness and completed
startup warmup. New-chat raw/content TTFT, full HTTP wall and actual attributed
prefill/decode remain separate in the HTTP report. The existing resource
sampler records process CPU times/RSS/private, available host RAM and board
VRAM across startup and requests. Sampling and descendant observation add
unquantified overhead, equally present in both arms; sampled maxima are not
allocator peaks, physical ownership or hard-limit proof.

Cleanup continuously records only proven owned descendants. It suspends the
identified live Host and its observed descendants, requires a stable final
tree scan, and terminates/reaps descendants only after PID/birth/executable
checks. The direct Host is then reaped by the established cleanup helper.
Changed identities are never stopped. A crashed or expired root cannot
establish exhaustive descendant cleanup and fails qualification. This is
ownership-checked cleanup, not OS process-tree containment.
It cannot prove discovery of a child reparented before observation.

Checkpoint identity records all split GGUF paths, sizes and modification times
before/after. It does not hash the large model again or establish a cold SSD
cache. Bounded visible parity does not establish full-answer semantic quality,
raw token/state identity, private KV identity, true GPU parallelism, media or
reload correctness, equivalent platform hard limits, or default readiness.

Controller tests never start a Host/model or invoke GPU commands:

```powershell
python -X utf8 -B -m unittest discover -s eng/validation/tests -p test_prefix_warmup_host.py
```
Thinking variants use explicit per-plan controls. `thinking` is `on` or `off`,
and `output_gate` is `visible` (strict default) or `stream` (thinking-on bounded
emitted-frame observation). An OFF warmup arm sets `startup_warmup_thinking` to
the exact string `0`; a default-ON arm omits it and records the actual Host
default. The launcher sets the environment only for its owned child and refuses
inherited inference overrides. `require_single_preferred_variant` verifies the
actual startup status contains exactly one completed preferred stage and its
generic one-slot reason. `require_all_new_chats_cached` checks actual positive
cached usage for each of the first three new sessions. ON thinking-only success
never upgrades the separate visible-output quality result.
An explicit client-visible continuation arm may set `follow_up_max_tokens` to
64 or128 while the initial new-chat cap remains16. The plan records that
different continuation scope; longer output does not relax literal visible
frames, finish, usage, actual private reuse or cleanup gates. A cap without a
requested continuation, or an unsupported/unbounded value, fails before launch.

`validation_token_trace: true` explicitly enables `TS_VALIDATION_TOKEN_TRACE=1`
only in the owned Host child. The default leaves it absent; inherited inference
overrides and nonboolean values are refused. Trace-enabled timings are diagnostic
observations, not performance comparisons. Actual token records still require
request/session linkage and an independent identity comparison.

A separate ON stream plan may set `long_new_chat_max_tokens: 256` and
`long_new_chat_expected_cached_tokens` to the exact observed public checkpoint
boundary (for example1332). This adds one fresh FF7 session after the unchanged
three16-token requests. It refuses followup/cancellation scope mixing. The probe
fully drains its bounded response and reports first raw, thinking and visible
latency separately. No visible output by256 remains inconclusive; it never
automatically raises the cap or establishes semantic/repeated-output quality.
The optional `http_timeout_seconds` setting is a finite1..1200-second per-request
deadline (default240). A256-token probe can explicitly use600 while keeping the
separate whole-Host lifetime watchdog. An absent trace does not by itself qualify
timing comparability; its control report uses only a negative trace exclusion.

The fresh structured-log directory may contain normal UTC-day or numeric size
rotation. The launcher selects the first canonical file, and HTTP capture binds
the complete bounded family with per-file pins. The existing whole-run lifetime
bounds the two-adjacent-date scope. Unknown names, rotation gaps or excessive
files/bytes fail; this does not retrospectively change older run verdicts.
