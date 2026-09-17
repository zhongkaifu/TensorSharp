#!/usr/bin/env bash
set -euo pipefail
B=/workspace/ts-codex-branch-completion-20260917
R=$B/repo
OUT=$B/results
export DOTNET_ROOT=/workspace/tensorsharp-no-patch-20260915/dotnet
export PATH="$DOTNET_ROOT:/usr/local/cuda/bin:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 TENSORSHARP_GGML_NO_UPDATE=1
cd "$R"
tar --no-same-owner -xzf "$B/ts-final-integration.tar.gz"
python3 - <<'PY' > "$OUT/final-source-verification.json"
import json,hashlib,pathlib,subprocess
root=pathlib.Path.cwd(); manifest=json.loads((root.parent/'ts-final-source-manifest.json').read_text())
mismatch=[p for p,h in manifest['file_sha256'].items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
ggml=root/'ExternalProjects/ggml'
revision=subprocess.check_output(['git','-C',str(ggml),'rev-parse','HEAD'],text=True).strip()
status=subprocess.check_output(['git','-C',str(ggml),'status','--porcelain'],text=True)
result={'source_revision':manifest['source_revision'],'tracked_files_verified':len(manifest['file_sha256']),'mismatches':mismatch,'ggml_revision':revision,'ggml_status':status}
print(json.dumps(result,indent=2)); assert not mismatch and not status and revision=='456172ec733a135778adcd32d00e576a58232e45',result
PY
cmake -S TensorSharp.GGML.Native -B TensorSharp.GGML.Native/build -DCMAKE_BUILD_TYPE=Release -DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON -DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=86 > "$OUT/final-configure.log" 2>&1
cmake --build TensorSharp.GGML.Native/build -j6 > "$OUT/final-native-build.log" 2>&1
dotnet build InferenceWeb.Tests -c Release --no-restore -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true > "$OUT/final-managed-build.log" 2>&1
sha256sum TensorSharp.GGML.Native/build/libGgmlOps.so InferenceWeb.Tests/bin/Release/net10.0/libGgmlOps.so > "$OUT/final-native-sha256.txt"
export LD_LIBRARY_PATH="$R/TensorSharp.GGML.Native/build"
export TS_TEST_GGML_BACKEND=cuda TS_TEST_GLM_CUDA=1 OMP_NUM_THREADS=8 MAX_CONTEXT=8192
exec 9>/workspace/locks/gpu0.lock
exec 8>/workspace/locks/gpu1.lock
exec 7>/workspace/locks/gpu2.lock
flock 9; flock 8; flock 7
export CUDA_VISIBLE_DEVICES=0,1,2
ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure --output-junit "$OUT/final-native.xml" > "$OUT/final-native.log" 2>&1
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~GlmDsaNativeBatchedDecodeLayerSplitTests|FullyQualifiedName~HostLoadRefusalProcessTests|FullyQualifiedName~ModelLoadRefusalTests|FullyQualifiedName~TestGateModelPathTests|FullyQualifiedName~FlashAttnUnsupportedShapeTests' --logger "trx;LogFileName=$OUT/final-integrated.trx" --logger 'console;verbosity=detailed' > "$OUT/final-integrated.log" 2>&1
export TS_TEST_QWEN4EXP_MTP_BACKEND=GgmlCuda TS_TEST_QWEN4EXP_MTP_FIXTURE=/workspace/ts-q4x/repo/fixtures/qwen4exp-qsa
export TS_TEST_QWEN4EXP_NATIVE_SHA256=$(sha256sum InferenceWeb.Tests/bin/Release/net10.0/libGgmlOps.so | cut -d ' ' -f 1)
export TS_TEST_QWEN4EXP_QSA=1 TS_KV_INITIAL_TOKENS=8 MAX_CONTEXT=1024 TS_TEST_QWEN4EXP_LAYER_SPLIT=2
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'Category!=Bench&Requires=Models&FullyQualifiedName~Qwen4Exp' --logger "trx;LogFileName=$OUT/final-qwen4.trx" --logger 'console;verbosity=detailed' > "$OUT/final-qwen4.log" 2>&1
flock -u 7; flock -u 8
export CUDA_VISIBLE_DEVICES=0 MAX_CONTEXT=8192 TENSORSHARP_TP_DEGREE=1 TS_TEST_MODEL_DIR=/workspace/models/gptoss20b/gpt-oss-20b-MXFP4.gguf
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~PrefixCacheContractConformanceTests.GptOss20B' --logger "trx;LogFileName=$OUT/final-gptoss-conformance.trx" --logger 'console;verbosity=detailed' > "$OUT/final-gptoss-conformance.log" 2>&1
printf 'FINAL BUILD AND CHECKS PASSED\n'
