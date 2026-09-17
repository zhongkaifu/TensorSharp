#!/usr/bin/env bash
set -uo pipefail
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export PYTHONPATH=/workspace/tensorsharp-no-patch-20260915/fixture-python-deps
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=3 GGML_CUDA_P2P=0
R=/workspace/ts-codex-20260917-dsv41-review/repo
B=/tmp/ts-codex-20260917-dsv41-native-make
OUT=/workspace/ts-codex-20260917-dsv41-review/results
FX=/workspace/ts-int4/repo/fixtures/text-f32-small
LIB=$B/libGgmlOps.so
exec 9>/workspace/locks/gpu3.lock
flock -n 9 || { echo 'GPU3 lock unavailable'; exit 9; }
cd "$R"
failed=0
step() {
  local name=$1; shift
  "$@" > "$OUT/$name.log" 2>&1
  local rc=$?
  echo "$name exit=$rc" | tee -a "$OUT/summary.txt"
  if [ "$rc" -ne 0 ]; then failed=$((failed + 1)); fi
}
date -u > "$OUT/summary.txt"
git -C "$R/ExternalProjects/ggml" rev-parse HEAD > "$OUT/upstream-revision.txt"
git -C "$R/ExternalProjects/ggml" status --porcelain > "$OUT/upstream-status.txt"
sha256sum "$LIB" > "$OUT/native-sha256.txt"
step linux-cuda-ctest ctest --test-dir "$B" --output-on-failure -j 1
step native-c-abi /usr/local/bin/python -B eng/tests/dsv4-execution-boundary.py --library "$LIB" --report "$OUT/native-c-abi.json"
step cuda-fixture-long /usr/local/bin/python -B eng/tests/dsv41-inference.py "$FX" --library "$LIB" --backend CUDA --gpus 1 --report "$OUT/cuda-fixture-long.json" --long-sparse-tokens 8500 --long-sparse-context 16384
step cuda-cpumoe-host-pread env TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=1 /usr/local/bin/python -B eng/tests/dsv41-inference.py "$FX" --library "$LIB" --backend CUDA --gpus 1 --cpu-moe 2 --report "$OUT/cuda-cpumoe-host-pread.json" --dump-logits "$OUT/cuda-cpumoe-host-pread.npz"
step cuda-cpumoe-host-pagewalk env TS_DSV41_ENGRAM_DEVICE=0 TS_DSV41_ENGRAM_WARM=1 TS_DSV4_WARM_PREAD=0 /usr/local/bin/python -B eng/tests/dsv41-inference.py "$FX" --library "$LIB" --backend CUDA --gpus 1 --cpu-moe 2 --report "$OUT/cuda-cpumoe-host-pagewalk.json" --dump-logits "$OUT/cuda-cpumoe-host-pagewalk.npz"
step auto-ubatch /usr/local/bin/python /workspace/ts-codex-20260917-dsv41-review/auto-ubatch.py "$LIB"
step sparse-prefill-bench env -u TS_DSV41_SPARSE_FA "$B/GgmlOpsCudaAttentionPrecisionTest" --benchmark-dsv41-prefill 512 33536 64 5
step tiled-prefill-bench env TS_DSV41_SPARSE_FA=0 "$B/GgmlOpsCudaAttentionPrecisionTest" --benchmark-dsv41-prefill 512 33536 64 3
echo "failed_steps=$failed" | tee -a "$OUT/summary.txt"
exit "$failed"
