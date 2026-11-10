#!/usr/bin/env bash
# Run the Pi coding agent with TensorSharp as its model provider. All arguments go to pi, for example:
#   scripts/pi.sh                                   # interactive terminal UI
#   scripts/pi.sh -p "Explain src/stats.js" </dev/null
#
# Pi reads its provider and defaults from an agent directory. This script uses its own, so ~/.pi/agent is left
# alone; on the first run it is created from ../pi-agent/models.json and settings.json.
#
# Environment:
#   PI_CODING_AGENT_DIR  the agent directory (default: ~/.pi/tensorsharp-agent)
#   PI_BIN               the pi command (default: pi on PATH)
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PI_CODING_AGENT_DIR="${PI_CODING_AGENT_DIR:-$HOME/.pi/tensorsharp-agent}"
mkdir -p "$PI_CODING_AGENT_DIR"
for file in models.json settings.json; do
    [[ -f "$PI_CODING_AGENT_DIR/$file" ]] || cp "$EXAMPLE_DIR/pi-agent/$file" "$PI_CODING_AGENT_DIR/$file"
done

# A local model: no model-catalog refreshes, version checks or install telemetry.
export PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0
exec "${PI_BIN:-pi}" "$@"
