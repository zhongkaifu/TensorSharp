#!/usr/bin/env bash
# End-to-end check: Pi, driven through TensorSharp, fixes the sample project's failing tests. Twice: once with the pi
# command (scripts/pi.sh) and once with the SDK (sdk/run-task.mjs), each on a fresh copy of sample-project/.
# Expects a running server (scripts/start-server.sh); START_SERVER=1 starts one and stops it afterwards.
#
# Environment:
#   TENSORSHARP_URL  server base URL (default http://127.0.0.1:5000/v1)
#   PI_BIN           the pi command (default: pi on PATH)
#   WORK_DIR         where the copies, logs and events go (default: <example>/.work)
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
URL="${TENSORSHARP_URL:-http://127.0.0.1:5000/v1}"
WORK="${WORK_DIR:-$EXAMPLE_DIR/.work}"
TASK="Run the test suite with \`npm test\`. Fix the bugs in src/stats.js so that every test passes. Do not change the tests. Run the tests again to confirm, then summarize the fixes in two sentences."
rm -rf "$WORK" && mkdir -p "$WORK"
export PI_CODING_AGENT_DIR="$WORK/pi-agent"
mkdir -p "$PI_CODING_AGENT_DIR" && cp "$EXAMPLE_DIR"/pi-agent/*.json "$PI_CODING_AGENT_DIR/"

server_pid=""
cleanup() { [[ -n "$server_pid" ]] && kill "$server_pid" 2>/dev/null || true; }
trap cleanup EXIT
if ! curl -sf -m 5 "$URL/models" >/dev/null; then
    if [[ "${START_SERVER:-0}" == 1 ]]; then
        "$EXAMPLE_DIR/scripts/start-server.sh" > "$WORK/server.log" 2>&1 &
        server_pid=$!
        echo "==> starting TensorSharp (log: $WORK/server.log)"
        for _ in $(seq 1 240); do curl -sf -m 5 "$URL/models" >/dev/null && break; sleep 5; done
    fi
    curl -sf -m 5 "$URL/models" >/dev/null || { echo "No TensorSharp server at $URL (start scripts/start-server.sh, or set START_SERVER=1)."; exit 1; }
fi
echo "==> server: $(curl -sf "$URL/models")"

check_project() {   # $1 = project copy, $2 = label
    local dir="$1" label="$2"
    if ! cmp -s "$EXAMPLE_DIR/sample-project/test/stats.test.js" "$dir/test/stats.test.js"; then
        echo "FAIL [$label]: the agent changed the tests"; return 1
    fi
    if (cd "$dir" && node --test --test-reporter=tap > "$WORK/$label-tests.tap" 2>&1); then
        echo "PASS [$label]: $(grep -E '^# (tests|pass|fail)' "$WORK/$label-tests.tap" | tr '\n' ' ')"
    else
        echo "FAIL [$label]: tests still fail (see $WORK/$label-tests.tap)"; return 1
    fi
}

status=0

# 1. The pi command, non-interactive, JSON events.
cp -R "$EXAMPLE_DIR/sample-project" "$WORK/cli-project"
(cd "$WORK/cli-project" && node --test >/dev/null 2>&1) && { echo "sample-project already passes; nothing to fix"; exit 1; }
echo "==> pi (CLI) on $WORK/cli-project"
start=$(date +%s)
(cd "$WORK/cli-project" && "$EXAMPLE_DIR/scripts/pi.sh" --mode json --no-session "$TASK" < /dev/null > "$WORK/cli-events.jsonl") || status=1
echo "    $(( $(date +%s) - start )) s; tool calls: $(grep -c '"type":"tool_execution_start"' "$WORK/cli-events.jsonl" || true)"
check_project "$WORK/cli-project" cli || status=1

# 2. The SDK.
cp -R "$EXAMPLE_DIR/sample-project" "$WORK/sdk-project"
cp -R "$EXAMPLE_DIR/sdk" "$WORK/sdk"
(cd "$WORK/sdk" && npm install --no-audit --no-fund --loglevel=error)
echo "==> pi (SDK) on $WORK/sdk-project"
TENSORSHARP_URL="$URL" PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 \
    node "$WORK/sdk/run-task.mjs" "$WORK/sdk-project" "$TASK" < /dev/null | tee "$WORK/sdk-output.txt" || status=1
check_project "$WORK/sdk-project" sdk || status=1

exit $status
