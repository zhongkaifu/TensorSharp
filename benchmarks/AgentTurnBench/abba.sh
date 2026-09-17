#!/usr/bin/env bash
# Interleaved (ABBA) AgentTurnBench runs of several builds on ONE device, with a
# base-vs-base control arm, so a regression can be told from machine noise.
#
#   benchmarks/AgentTurnBench/abba.sh OUT_DIR ROUNDS "BENCH ARGS" ARM [ARM...]
#
# ARM is name=MANAGED_REPO[:NATIVE_REPO]: the AgentTurnBench build under
# MANAGED_REPO/benchmarks/AgentTurnBench/bin/Release/net10.0 is copied into an
# isolated arm directory with the library from NATIVE_REPO/TensorSharp.GGML.Native/build
# (NATIVE_REPO defaults to MANAGED_REPO). Give the baseline twice under two names
# (b=... and c=...) to measure the noise floor. Rounds come in pairs: the arm list
# rotated by one more place per pair, forward then reversed, so use a multiple of
# 2 x (number of arms) rounds to balance positions. Every process start and end is logged to
# OUT_DIR/runs.txt with the load average (and the GPU's use when nvidia-smi exists).
# Summarize with abba_summary.py; compare.py applies the token-identity gate.
#
# Example (CUDA, GPU 6, 6 rounds, 3 measured passes per process):
#   GPU=6 benchmarks/AgentTurnBench/abba.sh results/abba 6 \
#     "--model gemma-4-E4B-it-Q8_0.gguf --backend ggml_cuda --chunk 512 --warmup 1 \
#      --scenarios newchat,tool,conc --conc 1,2,4 --measure-passes 3" \
#     b=/work/base a=/work/candidate c=/work/base
set -uo pipefail
if [ $# -lt 4 ]; then
  sed -n '2,21p' "$0" >&2
  exit 2
fi
OUT=$1; ROUNDS=$2; BENCH_ARGS=$3; shift 3
ARMS=("$@")
mkdir -p "$OUT" || exit 2
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU

state() {
  local s="load=$(cut -d' ' -f1-3 /proc/loadavg 2>/dev/null || uptime | sed 's/.*averages*: //')"
  if [ -n "${GPU:-}" ] && command -v nvidia-smi >/dev/null; then
    s="$s gpu$GPU=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i "$GPU")"
  fi
  echo "$s"
}

# Every arm runs exactly once per round, so the round number is its process number
# (no associative array: macOS still ships bash 3.2).
seen=" "
for spec in "${ARMS[@]}"; do
  name=${spec%%=*}
  if [[ "$spec" != *=* || ! "$name" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "abba.sh: invalid arm '$spec' (expected name=repo[:native-repo])" >&2; exit 2
  fi
  case "$seen" in *" $name "*) echo "abba.sh: arm name '$name' is given twice" >&2; exit 2 ;; esac
  seen="$seen$name "
done
# Record the requested workload before any child runs. A truncated process can
# leave valid-looking pass JSON files; the summary must know what is missing.
python3 - "$OUT" "$ROUNDS" "$BENCH_ARGS" "${ARMS[@]}" <<'PY' || exit 2
import json, pathlib, shlex, sys
out, rounds, args, *arms = sys.argv[1:]
out = pathlib.Path(out)
if (out / 'runs.txt').exists() or list(out.glob('*.json*')):
    sys.exit('abba.sh: output directory contains prior results; use a fresh directory')
try:
    rounds = int(rounds)
    words = shlex.split(args)
    passes = 1
    for i, word in enumerate(words):
        if word == '--measure-passes': passes = int(words[i + 1])
    if rounds < 1 or passes < 1: raise ValueError()
except (ValueError, IndexError):
    sys.exit('abba.sh: rounds and --measure-passes must be positive integers')
(out / 'run-plan.json').write_text(json.dumps(dict(rounds=rounds, measure_passes=passes,
    arms=[a.split('=', 1)[0] for a in arms]), indent=2) + '\n')
PY
# The managed resolver searches beside its DLL before consulting loader paths.
# Stage each arm so an explicit native override really is the library it loads,
# without replacing a user's build output (and without depending on LD vs DYLD).
case "$(uname -s)" in
  Darwin) native_name=libGgmlOps.dylib ;;
  Linux) native_name=libGgmlOps.so ;;
  *) echo "abba.sh: only macOS and Linux are supported" >&2; exit 2 ;;
esac
for spec in "${ARMS[@]}"; do
  name=${spec%%=*}; repos=${spec#*=}
  managed=${repos%%:*}; native=${repos#*:}
  source="$managed/benchmarks/AgentTurnBench/bin/Release/net10.0"
  stage="$OUT/runtime/$name"
  [ -f "$source/AgentTurnBench.dll" ] || { echo "abba.sh: missing $source/AgentTurnBench.dll" >&2; exit 2; }
  [ -f "$native/TensorSharp.GGML.Native/build/$native_name" ] || { echo "abba.sh: missing native library for $name" >&2; exit 2; }
  mkdir -p "$stage" || exit 2
  cp -R "$source/." "$stage/" || exit 2
  cp "$native/TensorSharp.GGML.Native/build/$native_name" "$stage/$native_name" || exit 2
done
python3 - "$OUT" "$native_name" "${ARMS[@]}" <<'PY' || exit 2
import hashlib, json, pathlib, sys
out, library, *arms = sys.argv[1:]
out = pathlib.Path(out)
identities = {}
for spec in arms:
    name, repos = spec.split('=', 1)
    managed, _, native = repos.partition(':')
    stage = out / 'runtime' / name
    identities[name] = dict(managed_repository=str(pathlib.Path(managed).resolve()),
        native_repository=str(pathlib.Path(native or managed).resolve()),
        sha256={f: hashlib.sha256((stage / f).read_bytes()).hexdigest()
                for f in ('AgentTurnBench.dll', library)})
(out / 'arm-identities.json').write_text(json.dumps(identities, indent=2) + '\n')
PY
# Plain reversal (ABC CBA) pins the middle arm of three or more to the middle
# position in every round, and position alone moved a cold pass by 4-9% on a
# shared A40. Rounds 2k+1 and 2k+2 therefore run the arm list rotated by k,
# forward then reversed: over 2N rounds every arm holds every position equally.
N=${#ARMS[@]}
failures=0
for round in $(seq 1 "$ROUNDS"); do
  shift_by=$(( ((round - 1) / 2) % N ))
  rotated=()
  for ((i = 0; i < N; i++)); do rotated+=("${ARMS[$(( (i + shift_by) % N ))]}"); done
  order=()
  if [ $((round % 2)) = 1 ]; then
    order=("${rotated[@]}")
  else
    for ((i = N - 1; i >= 0; i--)); do order+=("${rotated[$i]}"); done
  fi
  for spec in "${order[@]}"; do
    name=${spec%%=*}; repos=${spec#*=}
    n=$round
    echo "$name-$n start $(date -u +%T) $(state)" >> "$OUT/runs.txt"
    # shellcheck disable=SC2086 # BENCH_ARGS is a list of arguments
    dotnet "$OUT/runtime/$name/AgentTurnBench.dll" \
      $BENCH_ARGS --out "$OUT/$name-$n.json" > "$OUT/$name-$n.log" 2>&1
    rc=$?
    echo "$name-$n rc=$rc end $(date -u +%T) $(state)" >> "$OUT/runs.txt"
    [ "$rc" -eq 0 ] || failures=$((failures + 1))
  done
done
if [ "$failures" -ne 0 ]; then
  echo "FAILED $failures process(es)" >> "$OUT/runs.txt"
  echo "abba.sh: $failures process(es) failed; see $OUT/runs.txt" >&2
  exit 1
fi
echo DONE >> "$OUT/runs.txt"
