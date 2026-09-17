#!/usr/bin/env bash
set -euo pipefail
export DOTNET_ROOT=/workspace/tensorsharp-no-patch-20260915/dotnet
export PATH="$DOTNET_ROOT:/usr/local/cuda/bin:$PATH"
R=/workspace/ts-codex-branch-completion-20260917/repo
OUT=/workspace/ts-codex-branch-completion-20260917/results
cd "$R"
exec 9>/workspace/locks/gpu0.lock
exec 8>/workspace/locks/gpu1.lock
flock 9; flock 8
export CUDA_VISIBLE_DEVICES=0,1
export LD_LIBRARY_PATH="$R/TensorSharp.GGML.Native/build"
export TS_TEST_GGML_BACKEND=cuda TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCuda
export TS_TEST_QWEN4EXP_MTP_FIXTURE=/workspace/ts-q4x/repo/fixtures/qwen4exp-qsa
export TS_TEST_QWEN4EXP_NATIVE_SHA256=$(sha256sum InferenceWeb.Tests/bin/Release/net10.0/libGgmlOps.so | cut -d ' ' -f 1)
export TS_TEST_QWEN4EXP_QSA=1 TS_KV_INITIAL_TOKENS=8 MAX_CONTEXT=1024 TS_TEST_QWEN4EXP_LAYER_SPLIT=2
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'Category!=Bench&Requires=Models&FullyQualifiedName~Qwen4Exp' --logger "trx;LogFileName=$OUT/qwen4-cuda.trx" --logger 'console;verbosity=detailed' > "$OUT/qwen4-cuda.log" 2>&1
TS_Q4E_TEST_MMVQ_CHANNELS=1 dotnet test InferenceWeb.Tests -c Release --no-build --filter 'Category!=Bench&Requires=Models&FullyQualifiedName~Qwen4Exp' --logger "trx;LogFileName=$OUT/qwen4-channels.trx" --logger 'console;verbosity=detailed' > "$OUT/qwen4-channels.log" 2>&1
TensorSharp.GGML.Native/build/GgmlOpsQwen4ExpRowKernelProbe --repeats 10 > "$OUT/qwen4-row-probe.log" 2>&1
