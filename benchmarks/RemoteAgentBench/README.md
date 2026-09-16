# Remote inference with local TensorSharp agent tools

This model-free client runs `SkillAgentLoop`, `SkillScriptRunner`, `ShellRunner`, `CodeRunnerAdapter`, file tools, workspace management and artifact storage locally in WSL. The model runs at a remote TensorSharp OpenAI chat endpoint reached through an already-established loopback tunnel. **This does not establish VM-local sandbox support or Web UI SSE behavior.** No model is loaded by this project, and the project is intentionally outside the solution to avoid interfering with active builds.

## Prepare and schedule

Export fixtures from the existing release workflow validator:

```sh
python eng/validation/export-remote-agent-fixtures.py --output /path/fixtures.json
```

The default export is the matched **46-case campaign**: all six original scenarios at concurrency 1 and 4 (30 cases), followed by the four execution scenarios with distinct inputs at concurrency 4 (16 cases). Trial IDs are exactly `c1-i0` and `c4-i0` through `c4-i3`, as in the original validator. A separate `variant` field/subdirectory distinguishes repeated original/distinct trial names. Original c4 repeats deliberately retain the same prompt; distinct c4 waves use the existing unique input markers. The export reuses `CASES` and `case_spec` directly, preserving prompt bytes and expected values. `--smoke` prepares a separate 12-case serial smoke window and does not replace the required 46-case campaign.

Build this project with `-p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true`. It depends on the AgentHost tool-ID preservation fix: the transport fails if the loop loses an explicit assistant/result ID instead of reassociating it itself. The Windows Release build passed with zero warnings/errors, and the ID fix passed its focused and full portable suites. The real WSL/tunneled-model campaign remains unexecuted at this preparation checkpoint.

The owning root stage establishes the tunnel and verifies the remote process immediately before/after the profile. Supply an identity JSON containing actual `remote_pid`, `remote_start_ticks`, and `mapped_native_libraries` (path-to-SHA256 map), plus the model/checkpoint identities, managed/native hashes, endpoint mapping and capture timestamp. The client checks the supplied mapping against `--expected-native-sha256` and records the original identity file/hash; it cannot observe the remote PID itself. A post-run identity check remains required before release qualification.

Run the already-built client and independent source checks:

```sh
python eng/validation/run-remote-agent-profile.py \
  --dotnet /path/dotnet --assembly /path/RemoteAgentBench.dll \
  --endpoint http://127.0.0.1:18080/v1/chat/completions --model deepseek \
  --fixtures /path/fixtures.json --identity /path/remote-identity.json \
  --expected-native-sha256 ACTUAL_PINNED_SHA256 \
  --skills-root /path/original-release-skills --artifact-root /path/new-artifacts \
  --output /path/new-results --bwrap /usr/local/bin/bwrap
```

Use `--thinking` to preserve an explicitly scheduled thinking-mode run. The bounds are configurable (`--timeout`, `--max-rounds`, `--max-tokens`); defaults match the scheduled campaign: 1200 seconds per case, 24 rounds and 4096 tokens per generation. Shell/script limits match the existing host defaults (shell 120 seconds, maximum requested shell duration 600 seconds, skill script 60 seconds). The runner never starts a model, tunnel, SSH session or build.

## Model-free plumbing check and scheduled SSH owner

`smoke-remote-agent-loopback.py` runs the two original serial code scenarios against a clearly labeled scripted loopback endpoint. It drives the real write/read/edit/shell implementations, required bubblewrap, artifact HTTP download, and the existing independent source verifier. The WSL smoke passed both workflows and both independent checks across eight scripted HTTP turns. Its `--scripted-fixture` mode requires a synthetic fixture identity and the model label `scripted-loopback-fixture`; every report explicitly excludes model quality and native identity proof. The first fixture attempt failed because its HTTP reader did not support chunked requests; that failed trace is retained separately.

For an explicitly scheduled remote lane, `run-remote-agent-over-ssh.py` owns an ephemeral SSH agent and a loopback-only tunnel. Supply `--assembly`, `--fixtures`, `--skills-root`, `--ready-identity`, `--output`, `--model`, `--expected-native-sha256`, `--remote-results`, and `--nonce`. It requires the 46-case campaign. Defaults use the existing Windows private-key and known-hosts files through their WSL mount, strict host-key checking, remote port 22050, local tunnel port 18080, and server port 5100. The key is streamed directly into `ssh-add` and never copied to disk or logged. The server must already be ready; this script does not start a model.

The owner checks the supplied PID/start time, actual mapped native hashes, and ownership of the remote listening socket before running. It records mapped managed hashes and checks them again afterward, verifies the requested model through `/v1/models`, then publishes a nonce-bound completion with the client report hash through an atomic remote rename. The remote `wait-external-suite.py` stage consumes that completion. The owner always stops its own tunnel and ephemeral agent. A failed/partial campaign or changed binding remains failed; the scripted smoke and the prepared real-model campaign have separate output directories.

The ready identity must contain real values obtained by the owning server stage. This is its required schema, with illustrative values that must be replaced:

```json
{
  "remote_pid": 12345,
  "remote_start_ticks": 987654321,
  "nonce": "ACTUAL_CAMPAIGN_NONCE",
  "mapped_native_libraries": {
    "/actual/server/directory/libGgmlOps.so": "ACTUAL_64_CHARACTER_LOWERCASE_SHA256"
  },
  "remote_boot_id": "ACTUAL_REMOTE_BOOT_ID",
  "captured_at_unix": 0,
  "model_checkpoint": {
    "path": "/actual/verified/model/first-shard.gguf",
    "inventory_sha256": "ACTUAL_VERIFIED_INVENTORY_SHA256"
  }
}
```

`remote_start_ticks` is field 22 from `/proc/PID/stat`, parsed after the closing parenthesis of the process name. The PID comes from the active server profile, and mapped library paths come from that PID's `/proc/PID/maps`. Include every mapped `GgmlOps` or `libggml` library with its actual hash. Optional boot/model/inventory metadata is retained verbatim. The owner independently recaptures the process and its endpoint socket; its preflight also pins all sibling `TensorSharp.*.dll` files so a later legitimate lazy load must match that application manifest.

The VM server stage can capture this identity after readiness using the shared observer:

```sh
python3 eng/validation/capture-release-server-identity.py \
  --profile /path/to/active-VM-profile-output/profile.json \
  --output /path/to/active-VM-profile-output/ready-identity.json \
  --expected-native-sha256 ACTUAL_NATIVE_SHA256 --nonce ACTUAL_CAMPAIGN_NONCE
```

The capture verifies the profile's native and managed snapshots, owned listening socket and model ID, observes process identity twice, and atomically publishes a fresh file. It preserves supplied model inventory metadata without hashing the model again. The owner rejects a missing or different ready-file nonce before starting SSH. The scheduling stage copies this ready file to WSL; a prior run's PID, hash or nonce must not be reused.

For the prepared WSL layout, the exact owner invocation is:

```sh
prepared=/home/zhongkaifu/tensorsharp-no-patch-20260915/results/remote-agent-prepared
python3 "$prepared/harness/eng/validation/run-remote-agent-over-ssh.py" \
  --assembly "$prepared/app/RemoteAgentBench.dll" \
  --fixtures "$prepared/fixtures-release46.json" --skills-root "$prepared/skills" \
  --ready-identity /path/to/fresh-actual-ready-identity.json \
  --output /path/to/fresh-local-campaign-output \
  --model ACTUAL_MODEL_ID --expected-native-sha256 ACTUAL_NATIVE_SHA256 \
  --remote-results /path/to/active-VM-profile-output --nonce ACTUAL_CAMPAIGN_NONCE \
  --max-rounds 24 --max-tokens 4096 --timeout 1200
```

The VM wait stage must use the same nonce and completion path `<remote-results>/external-client-complete.json`. `--remote-completion-name` can align an explicitly chosen alternative filename. Add `--thinking` only for a separately scheduled thinking-mode campaign. This command is a prepared interface, not evidence that the campaign has run.

For two campaigns in one held-server lifetime, capture a fresh identity and choose a separate nonce, local output and completion filename for each. For example, use `external-client-nonthinking-complete.json` first, then `external-client-thinking-complete.json` with `--thinking`. Each corresponding VM wait must consume that exact filename/nonce before the server proceeds to its remaining release suites.

## Evidence and gates

- WSL must provide actual bubblewrap with write, home-read and network confinement. Both script and shell runners use `Required`; install/network capabilities stay off. Tool success is observed from the real AgentHost invocation results.
- All local tool declarations are sent as client tools to the remote server, with `skills=[]` and `skills_discovery=false`. The local loop owns those same tools through `SkillToolContext`, with an empty local `ClientTools` list. This prevents remote automatic execution from hiding a local failure.
- Every raw HTTP request/response, finish reason, token usage, round boundary, local tool invocation, code result, history and artifact version is retained. The event shape permits reuse of existing artifact verification, but the report labels these as local callbacks, not SSE events.
- Original exact answer/checksum and required successful-tool gates remain intact. Artifact JSON is fetched over a loopback-only HTTP endpoint backed by `CodeArtifactStore.TryResolve`, using the last advertised immutable version. The independent Python verifier then runs the final source inside a separate required bubblewrap sandbox on additional inputs. Model-written `tests_passed` does not qualify source correctness on its own.
- Unknown or failed tool calls remain in the trace; a recovery can satisfy the original oracle. A round-limit stop, pending client call, unfinished tool reply, truncated completion, missing ID or incorrect final output fails.
- OpenAI responses do not contain exact raw model token arrays or generation-boundary tokens. This topology can qualify remote-model/local-tool functionality; its timing includes HTTP and repeated prompt rendering and does not qualify the local raw-token KV splice path.
- `release_qualified` stays false until the external remote post-run identity gate and the full release matrix are reviewed. Every output directory is new; original VM sandbox refusal evidence remains unchanged.
