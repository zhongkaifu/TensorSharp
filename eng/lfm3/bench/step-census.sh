#!/usr/bin/env bash
# The census of batched decode steps on a Linux CUDA machine (stage B0 of the batched-serving design): for each N, one
# untraced `Lfm3Probe batch --sequences N` (the probe's own figures: step time, the host's share outside the executor's runs,
# graph builds / replays / recordings, input leaves and bytes, results read back, executor nodes and kernels), then the
# same cell under nsys with only its timed steps captured (`--profile-range` marks them with the CUDA profiler API), reduced
# by step-census.py: kernels a step and those inside CUDA graphs, GPU busy time and the idle gaps the device waits for the
# host, kernel / graph launch, memcpy and synchronize API calls, H2D / D2H copies and bytes, the busiest kernels.
#
# usage: step-census.sh MODEL BACKEND [N ...]        (default N: 1 2 4 8 16 32 64; BACKEND ggml_cuda or cuda)
#   OUT      result directory (default artifacts/lfm3/batch/census-<backend>-<stamp>)
#   PROMPT   prompt tokens a sequence (128), WARM untimed steps a cell (32), STEPS timed steps a run (32), REPS runs (1);
#            the plain run uses PLAIN_REPS (3) runs of STEPS
#   GREEDY   --greedy: each sequence decodes its argmax (device ids where the executor returns them, as the server)
#   GRAPH_TRACE  nsys --cuda-graph-trace (node: every kernel inside a CUDA graph is its own record; graph: one record a launch)
#   NSYS     nsys binary (default nsys on PATH, else /usr/local/bin/nsys); NSYS=none skips the traced runs
#   WRAP     a command prefix for every probe run (a lock: WRAP="/root/gpu-lo.sh flock /root/build.lock")
#   PROBE    Lfm3Probe.dll (default this checkout's Release build); MAX_CONTEXT and the KV dtype as for any probe run
# Several calls may share one OUT (e.g. plain-only sizes with NSYS=none, then traced sizes): conditions.txt is appended to.
# Build first (Release): TENSORSHARP_GGML_NO_UPDATE=1 dotnet build eng/lfm3/Lfm3Probe -c Release. The GPU must be idle.
# The traced step times include nsys's per-kernel overhead: read times from the plain rows, counts from the traced ones.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../../.." && pwd)
model=${1:?usage: step-census.sh MODEL BACKEND [N ...]}
backend=${2:?usage: step-census.sh MODEL BACKEND [N ...]}
shift 2
sizes=${*:-1 2 4 8 16 32 64}
probe=${PROBE:-$root/eng/lfm3/Lfm3Probe/bin/Release/net10.0/Lfm3Probe.dll}
[[ -f $probe ]] || { echo "build it first: TENSORSHARP_GGML_NO_UPDATE=1 dotnet build $root/eng/lfm3/Lfm3Probe -c Release" >&2; exit 2; }
PROMPT=${PROMPT:-128}
WARM=${WARM:-32}
STEPS=${STEPS:-32}
REPS=${REPS:-1}
PLAIN_REPS=${PLAIN_REPS:-3}
GRAPH_TRACE=${GRAPH_TRACE:-node}
nsys=${NSYS:-$(command -v nsys || echo /usr/local/bin/nsys)}
read -r -a wrap <<< "${WRAP:-}"
out=${OUT:-$root/artifacts/lfm3/batch/census-$backend-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$out"
if command -v nvidia-smi > /dev/null && [[ -n $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null) ]]; then
  echo "the GPU is busy:" >&2; nvidia-smi --query-compute-apps=pid,name --format=csv,noheader >&2; exit 3
fi
{ echo "== start $(date -u +%FT%TZ)"; uptime; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
  git -C "$root" log --oneline -1 2>/dev/null || echo "commit: $root is not a git checkout (an exported tree)"
  git -C "$root/ExternalProjects/ggml" log --format='ggml %H' -1 2>/dev/null || true
  [[ $nsys != none ]] && "$nsys" --version 2>/dev/null || true
  echo "model=$model backend=$backend sizes=$sizes PROMPT=$PROMPT WARM=$WARM STEPS=$STEPS REPS=$REPS PLAIN_REPS=$PLAIN_REPS GREEDY=${GREEDY:-} GRAPH_TRACE=$GRAPH_TRACE"
  echo "env: MAX_CONTEXT=${MAX_CONTEXT:-} TS_KV_CACHE_DTYPE=${TS_KV_CACHE_DTYPE:-}"; } >> "$out/conditions.txt"
echo "results: $out" >&2
common=(batch "$model" "$backend" --prompt "$PROMPT" --warm "$WARM" --steps "$STEPS" ${GREEDY:+--greedy})
for n in $sizes; do
  { echo "== n=$n plain $(date +%T)"; uptime; } >> "$out/conditions.txt"
  ${wrap[@]+"${wrap[@]}"} dotnet "$probe" "${common[@]}" --sequences "$n" --reps "$PLAIN_REPS" --ps > "$out/n$n.plain.txt" 2> "$out/n$n.plain.err" \
    || { echo "n=$n plain run failed: $out/n$n.plain.err" >&2; continue; }
  grep -E '^(CENSUS|RESULT)' "$out/n$n.plain.txt" >&2 || true
  [[ $nsys == none ]] && continue
  { echo "== n=$n traced $(date +%T)"; uptime; } >> "$out/conditions.txt"
  ${wrap[@]+"${wrap[@]}"} "$nsys" profile -o "$out/n$n" --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop \
    --cuda-graph-trace="$GRAPH_TRACE" -t cuda --sample=none --cpuctxsw=none \
    dotnet "$probe" "${common[@]}" --sequences "$n" --reps "$REPS" --profile-range > "$out/n$n.traced.txt" 2> "$out/n$n.traced.err" \
    || { echo "n=$n traced run failed: $out/n$n.traced.err" >&2; continue; }
  grep -E 'INFO profiler|WARN' "$out/n$n.traced.err" >&2 || true
  "$nsys" export --type sqlite --force-overwrite true -o "$out/n$n.sqlite" "$out/n$n.nsys-rep" > /dev/null
done
{ echo "== end $(date +%T)"; uptime; } >> "$out/conditions.txt"
# census.md: the traced reports (with the plain runs' CENSUS lines joined); census-plain.md: every plain run's CENSUS line
# (a plain-only call into an OUT that already holds traced reports leaves census.md as it is).
grep -h '^CENSUS' "$out"/n*.plain.txt | sort -t= -k2 -n | tee "$out/census-plain.md" || true
if [[ $nsys != none ]] && compgen -G "$out/n*.sqlite" > /dev/null; then
  python3 "$here/step-census.py" --steps $((STEPS * REPS)) --json "$out/census.json" "$out"/n*.sqlite | tee "$out/census.md"
fi
