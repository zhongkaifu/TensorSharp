#!/usr/bin/env bash
# Start TensorSharp.Server.Host for the Pi coding agent: Qwen3.8-27B's fastest Apple Silicon preset
# (config/qwen3.8-27b-metal-dflash2.json) plus this example's overlay (tensorsharp-pi.json). Extra arguments are
# passed to the server and win over both files, for example --port 5001.
#
# Environment:
#   TENSORSHARP_MODELS   model root; the preset reads <root>/Qwen/... and downloads missing files (default: <repo>/models)
#   TENSORSHARP_PRESET   another preset to start from (default: config/qwen3.8-27b-metal-dflash2.json)
#   TENSORSHARP_SERVER   the server command, e.g. a release build's TensorSharp.Server.Host (default: the repo's build)
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/../.." && pwd)"
export TENSORSHARP_MODELS="${TENSORSHARP_MODELS:-$REPO_ROOT/models}"
PRESET="${TENSORSHARP_PRESET:-$REPO_ROOT/config/qwen3.8-27b-metal-dflash2.json}"
OVERLAY="$EXAMPLE_DIR/tensorsharp-pi.json"

if [[ -n "${TENSORSHARP_SERVER:-}" ]]; then
    read -r -a SERVER <<< "$TENSORSHARP_SERVER"
elif [[ -f "$REPO_ROOT/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll" ]]; then
    SERVER=(dotnet "$REPO_ROOT/TensorSharp.Server.Host/bin/TensorSharp.Server.Host.dll")
else
    SERVER=(dotnet run --project "$REPO_ROOT/TensorSharp.Server.Host" -c Release -p:TensorSharpSkipMlxNative=true --)
fi

echo "==> models: $TENSORSHARP_MODELS"
echo "==> ${SERVER[*]} --config $PRESET --config $OVERLAY $*"
exec "${SERVER[@]}" --config "$PRESET" --config "$OVERLAY" "$@"
