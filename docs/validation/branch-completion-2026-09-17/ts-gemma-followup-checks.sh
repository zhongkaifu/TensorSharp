#!/usr/bin/env bash
set -euo pipefail
B=/workspace/ts-codex-branch-completion-20260917
R=$B/repo
OUT=$B/results
export DOTNET_ROOT=/workspace/tensorsharp-no-patch-20260915/dotnet
export PATH="$DOTNET_ROOT:/usr/local/cuda/bin:$PATH"
export DOTNET_CLI_TELEMETRY_OPTOUT=1 TENSORSHARP_GGML_NO_UPDATE=1
cd "$R"
tar --no-same-owner -xzf "$B/ts-gemma-followup.tar.gz"
python3 - <<'PY' > "$OUT/gemma-followup-source.json"
import pathlib,hashlib,json
root=pathlib.Path.cwd(); data=json.loads((root.parent/'ts-gemma-followup-manifest.json').read_text())
mismatch=[p for p,h in data['changed_file_sha256'].items() if hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
data['mismatches']=mismatch; assert not mismatch and not data['native_source_changed']; print(json.dumps(data,indent=2))
PY
dotnet build InferenceWeb.Tests -c Release --no-restore -p:TensorSharpSkipGgmlNative=true -p:TensorSharpSkipMlxNative=true > "$OUT/gemma-followup-build.log" 2>&1
sha256sum TensorSharp.GGML.Native/build/libGgmlOps.so InferenceWeb.Tests/bin/Release/net10.0/libGgmlOps.so > "$OUT/gemma-followup-native-sha256.txt"
export LD_LIBRARY_PATH="$R/TensorSharp.GGML.Native/build"
export DOTNET_TieredCompilation=0 TS_TEST_GGML_BACKEND=cuda CUDA_VISIBLE_DEVICES=0 MAX_CONTEXT=8192 OMP_NUM_THREADS=8 TENSORSHARP_TP_DEGREE=1
export TS_TEST_MODEL_DIR=/workspace/models/gemma4-e4b
exec 9>/workspace/locks/gpu0.lock
flock 9
dotnet test InferenceWeb.Tests -c Release --no-build --filter 'FullyQualifiedName~Gemma4BatchedDecodeLifetimeTests|FullyQualifiedName~Gemma4BatchedFusedDecodeParityTests' --logger "trx;LogFileName=$OUT/gemma-followup-cuda.trx" --logger 'console;verbosity=detailed' > "$OUT/gemma-followup-cuda.log" 2>&1
printf 'GEMMA FOLLOW-UP CHECKS PASSED\n'
