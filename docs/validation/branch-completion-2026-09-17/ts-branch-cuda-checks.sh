#!/usr/bin/env bash
set -euo pipefail
export DOTNET_ROOT=/workspace/tensorsharp-no-patch-20260915/dotnet
export PATH="$DOTNET_ROOT:/usr/local/cuda/bin:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1
R=/workspace/ts-codex-branch-completion-20260917/repo
OUT=/workspace/ts-codex-branch-completion-20260917/results
mkdir -p "$OUT"
cd "$R"
export LD_LIBRARY_PATH="$R/TensorSharp.GGML.Native/build"
export TS_TEST_GGML_BACKEND=cuda
export MAX_CONTEXT=8192
export OMP_NUM_THREADS=8
case "$1" in
  native)
    exec 9>/workspace/locks/gpu5.lock
    exec 8>/workspace/locks/gpu6.lock
    flock 9; flock 8
    export CUDA_VISIBLE_DEVICES=5,6
    cp TensorSharp.GGML.Native/build/libGgmlOps.so InferenceWeb.Tests/bin/Release/net10.0/
    cp TensorSharp.GGML.Native/build/libGgmlOps.so benchmarks/AgentTurnBench/bin/Release/net10.0/
    ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure --output-junit "$OUT/native.xml" > "$OUT/native.log" 2>&1
    dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~FlashAttnUnsupportedShapeTests|FullyQualifiedName~HunyuanDenseServingTests' --logger "trx;LogFileName=$OUT/cuda-fallback.trx" > "$OUT/cuda-fallback.log" 2>&1
    ;;
  qwen)
    exec 9>/workspace/locks/gpu5.lock; flock 9
    export CUDA_VISIBLE_DEVICES=5 TS_TEST_MODEL_DIR=/workspace/models/qwen35-9b
    export TS_TEST_QWEN35_MMPROJ=/workspace/models/qwen35-9b/mmproj-F16.gguf
    dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~Qwen35ImageFollowUpExactnessTests|FullyQualifiedName~Qwen35InjectKVBlockRefusalTests|FullyQualifiedName~Qwen35ConvScratchTests' --logger "trx;LogFileName=$OUT/qwen35.trx" --logger 'console;verbosity=detailed' > "$OUT/qwen35.log" 2>&1
    ;;
  gemma)
    exec 9>/workspace/locks/gpu6.lock; flock 9
    export CUDA_VISIBLE_DEVICES=6 TS_TEST_MODEL_DIR=/workspace/models/gemma4-12b
    export TS_TEST_MEDIA_DIR=/workspace/ts-codex-branch-completion-20260917
    dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~Gemma4MediaAfterReusedPrefixExactnessTests' --logger "trx;LogFileName=$OUT/gemma4.trx" --logger 'console;verbosity=detailed' > "$OUT/gemma4.log" 2>&1
    ;;
  gptoss)
    exec 9>/workspace/locks/gpu5.lock; flock 9
    export CUDA_VISIBLE_DEVICES=5
    for repeat in 1 2; do
      dotnet benchmarks/AgentTurnBench/bin/Release/net10.0/AgentTurnBench.dll --model /workspace/models/gptoss20b/gpt-oss-20b-MXFP4.gguf --backend ggml_cuda --scenarios conc --conc 1,4,8 --conc-gate --new 32 --warmup 1 --out "$OUT/gptoss-$repeat.json" > "$OUT/gptoss-$repeat.log" 2>&1
    done
    python3 benchmarks/AgentTurnBench/compare.py "$OUT/gptoss-1.json" "$OUT/gptoss-2.json" --require-concurrent-identity --max-regression-percent 100 > "$OUT/gptoss-identity.log" 2>&1
    ;;
  *) exit 2 ;;
esac
