import argparse
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['single', 'split', 'cpu'], required=True)
parser.add_argument('--run', required=True)
parser.add_argument('--base', default='/workspace/ts-qwen-retained-r4')
parser.add_argument('--filter')
parser.add_argument('--gpu', choices=['1', '6'], default='1')
parser.add_argument('--fixture')
parser.add_argument('--disable-fusion', action='store_true')
args = parser.parse_args()
base = pathlib.Path(args.base)
app = base / 'tests/bin/Release/net10.0'
fixture = pathlib.Path(args.fixture or '/workspace/ts-q4e-retained/repo/fixtures/qwen4exp-qsa')
manifest = json.loads((base / 'prepare.json').read_text())
assert manifest['build_exit'] == 0 and manifest['production_identity_verified']
out = base / 'results' / args.run
out.mkdir(parents=True, exist_ok=False)

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def audit():
    hashes = {name: digest(app / name) for name in manifest['production_identity']}
    assert hashes == manifest['production_identity']
    assert digest(app / 'InferenceWeb.Tests.dll') == manifest['actual_test_dll_sha256']
    return hashes

env = os.environ.copy()
env['DOTNET_ROOT'] = str(pathlib.Path(manifest['dotnet']).parent)
env['PATH'] = env['DOTNET_ROOT'] + ':' + env['PATH']
env.update(CUDA_VISIBLE_DEVICES=args.gpu if args.mode == 'single' else '1,5',
           TS_TEST_GGML_BACKEND='cuda', TS_TEST_QWEN4EXP_MTP_BACKEND='GgmlCuda',
           TS_TEST_QWEN4EXP_QSA='1', TS_TEST_QWEN4EXP_MTP_FIXTURE=str(fixture),
           TS_TEST_QWEN4EXP_NATIVE_SHA256=manifest['production_identity']['libGgmlOps.so'],
           TS_KV_INITIAL_TOKENS='8')
if args.mode == 'cpu':
    env.update(CUDA_VISIBLE_DEVICES='', TS_TEST_GGML_BACKEND='cpu', TS_TEST_QWEN4EXP_MTP_BACKEND='GgmlCpu')
if args.mode == 'split':
    env['TS_TEST_QWEN4EXP_LAYER_SPLIT'] = '2'
    selected = 'FullyQualifiedName~LayerSplitCandidate_CloneCorePreservesAllDeviceStateAndFullLogits|FullyQualifiedName~LayerSplitCheckpoint_PreservesAllDeviceStateAndFullLogits'
else:
    env.pop('TS_TEST_QWEN4EXP_LAYER_SPLIT', None)
    selected = '(FullyQualifiedName~Qwen4ExpMtpIntegrationTests|FullyQualifiedName~Qwen4ExpRetainedCache|FullyQualifiedName~Qwen4ExpVideoForwardTests|FullyQualifiedName~Qwen4ExpQsaHistoryTests|FullyQualifiedName~Qwen4ExpMtpStateTests)&FullyQualifiedName!~LayerSplit'
if args.disable_fusion:
    env['GGML_CUDA_DISABLE_FUSION'] = '1'
if args.filter:
    selected = args.filter
command = [manifest['dotnet'], 'test', str(base / 'tests/InferenceWeb.Tests.csproj'), '-c', 'Release',
           '--no-build', '--no-restore', '--filter', selected,
           '--logger', 'console;verbosity=normal', '--logger', 'trx;LogFileName=result.trx',
           '--results-directory', str(out), '-m:1', '/nodeReuse:false']
record = {'mode': args.mode, 'argv': command, 'runner_sha256': digest(pathlib.Path(__file__)),
          'env': {k: env[k] for k in env if k.startswith('TS_') or k.startswith('GGML_') or k == 'CUDA_VISIBLE_DEVICES'},
          'before': audit(), 'test_dll_sha256': manifest['actual_test_dll_sha256'],
          'fixture': {name: digest(fixture / name) for name in ['target.gguf', 'head.gguf', 'manifest.json']},
          'devices': subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name', '--format=csv,noheader'], text=True),
          'started_ns': time.time_ns(), 'timing_qualified': False}
(out / 'owner.json').write_text(json.dumps(record, indent=2) + '\n')
with (out / 'run.log').open('w') as log:
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record['pid'] = process.pid
    record['process_start_ticks'] = pathlib.Path(f'/proc/{process.pid}/stat').read_text().split()[21]
    try:
        record['exit'] = process.wait(timeout=300)
    except subprocess.TimeoutExpired:
        record['timed_out'] = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        record['exit'] = 124
record['finished_ns'] = time.time_ns()
record['after'] = audit()
record['process_gone'] = not pathlib.Path(f'/proc/{record["pid"]}').exists()
(out / 'owner.json').write_text(json.dumps(record, indent=2) + '\n')
print((out / 'run.log').read_text()[-12000:])
print('OWNER', out / 'owner.json')
raise SystemExit(record['exit'])
