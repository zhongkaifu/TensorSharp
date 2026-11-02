#!/usr/bin/env python3
"""Bounded CPU setup for optional Qwen3.8 engines; plan by default.

Never starts a model. Emits frozen-launcher commands for the VM owner to run in
an exclusive GPU window. Setup success is not readiness, quality, or performance.
"""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

EOLE_REV = '572c78712aa5b6a5403c3fea5181ca4e91beb85a'
FOCUS_REV = '23d297950b70d0008c675932e108f4bbd23e65f5'
MODEL_REV = 'd4a1a9ceefb98bf052a0055269655615b2426393'
DRAFT_REV = '015e795645c74b1a0eeef3b570031fb62e769bc5'
FA_URL = 'https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.10.0/flash_attn-2.8.3%2Bcu130torch2.14-cp312-cp312-linux_x86_64.whl'
FA_SHA = '623308fbb31944b6c312b0cc0c500d73d41e5c41e9d8933c27d290e0ca45caac'
GGUF_PINS = dict(target='d5659ecb29adaf75f313fdae03eb7ff17baf3f986f0c3b1348e8afda1670c9a8', draft='1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd', mmproj='de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8')
BENCH_SHA = '4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c'
HAYSTACK_SHA = 'f9560cacfc773eaa2e2e34de92f10f867d6c7c269c8d72e11dfc72b2e3fe8ca3'
SHORT = 'short_chat,long_prompt_4k,long_prompt_16k,long_prompt_32k,json_object,json_schema,tool_call_single,tool_call_parallel,agentic_loop,multi_turn,image,long_generation,parallel_2,parallel_4,quality_tasks,quality_gsm8k'
LONG = 'long_prompt_128k,long_prompt_250k'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for b in iter(lambda: source.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def norm(name): return name.lower().replace('_', '-').replace('.', '-')
def protected(name): return norm(name) in ('torch', 'triton') or norm(name).startswith('nvidia-')


def resolved_requirements(report):
    result = []
    for row in report['install']:
        name = row['metadata']['name']
        if norm(name) == 'eole': continue  # installed separately from pinned local source
        if protected(name): raise ValueError('Resolver attempted protected SG dependency: ' + name)
        info = row['download_info']; url = info['url']
        digest = info.get('archive_info', {}).get('hashes', {}).get('sha256')
        if not url.startswith('https://') or not digest or len(digest) != 64:
            raise ValueError('Unpinned dependency URL/hash: ' + name)
        result.append(dict(name=name, version=row['metadata']['version'], url=url, sha256=digest))
    return result


def output_guard(args):
    if args.out != Path(os.path.normpath(args.out)):
        raise ValueError('Output must be a normalized absolute path')
    for parent in (args.out, *args.out.parents):
        if parent.is_symlink(): raise ValueError('Output path has a symlink ancestor')
    for protected_root in (args.sg_python.parent.parent, args.harness_root, Path('/root/TensorSharp')):
        if args.out.is_relative_to(protected_root.absolute()):
            raise ValueError('Output overlaps protected source/runtime: ' + str(protected_root))


def inventory(root):
    """All regular files and link identities below explicit root; no link-directory traversal."""
    root = Path(root); rows = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in sorted(dirs + files):
            p = Path(parent) / name
            if p.is_symlink():
                row = dict(path=str(p.relative_to(root)), symlink=os.readlink(p))
                if p.is_file(): row.update(resolved=str(p.resolve()), sha256=sha(p), bytes=p.stat().st_size)
                rows.append(row)
            elif p.is_file(): rows.append(dict(path=str(p.relative_to(root)), sha256=sha(p), bytes=p.stat().st_size))
    return sorted(rows, key=lambda r: r['path'])


def proc_identity(pid):
    raw = Path(f'/proc/{pid}/stat').read_text(); tail = raw[raw.rindex(')') + 2:].split()
    return dict(pid=pid, starttime=tail[19], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


class Runner:
    def __init__(self, out, timeout, step_timeout, env):
        self.out = out; self.deadline = time.monotonic() + timeout; self.step_timeout = step_timeout; self.env = env; self.events = []

    def call(self, name, argv, cwd=None, env=None):
        remaining = min(self.step_timeout, self.deadline - time.monotonic())
        if remaining <= 0: raise TimeoutError('Global setup deadline exhausted')
        event = dict(name=name, argv=list(map(str, argv)), cwd=str(cwd) if cwd else None, timeout_s=remaining, status='STARTING')
        self.events.append(event); save(self.out / 'events.json', self.events)
        log = self.out / (f'{len(self.events):02d}-' + name + '.log'); event['log'] = str(log)
        with log.open('wb') as stream:
            child = subprocess.Popen(event['argv'], cwd=cwd, env=env or self.env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                # Registration and its disk write are part of the owned lifetime:
                # interruption here must not leak the newly created setup child.
                event['process'] = proc_identity(child.pid); event['status'] = 'RUNNING'; save(self.out / 'events.json', self.events)
                code = child.wait(timeout=remaining)
            except BaseException:
                # This synchronous setup child owns its own session. Never kill a
                # PID/group whose Linux boot/start identity changed.
                previous = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM)}
                signaled = False
                try:
                    registered = event.get('process')
                    # Without registration, a live, unreaped Popen child remains
                    # ours and cannot have a reused PID; it created this session.
                    owns = child.poll() is None and os.getpgid(child.pid) == child.pid
                    if owns and registered is not None: owns = proc_identity(child.pid) == registered
                    if owns:
                        os.killpg(child.pid, signal.SIGTERM)
                        signaled = True
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            if child.poll() is None and (registered is None or proc_identity(child.pid) == registered):
                                os.killpg(child.pid, signal.SIGKILL)
                            child.wait(timeout=10)
                finally:
                    for sig, handler in previous.items(): signal.signal(sig, handler)
                    event.update(status='FAILED_OR_INTERRUPTED', exit_code=child.poll(), cleanup_group_signaled=signaled,
                                 cleanup_scope='Verified own group signaled; direct child waited. Independent full descendant drain NOT_PROVEN.' if signaled else 'No signal sent: child exited or ownership not verified. Full descendant drain NOT_PROVEN.'); save(self.out / 'events.json', self.events)
                raise
        event.update(exit_code=code, status='COMPLETE' if code == 0 else 'ERROR', log_sha256=sha(log)); save(self.out / 'events.json', self.events)
        if code: raise RuntimeError(f'{name} exited {code}; {log}')
        return log.read_text(errors='replace')

    def json(self, name, argv, cwd=None, env=None):
        lines = self.call(name, argv, cwd, env).splitlines()
        return json.loads(next(line for line in reversed(lines) if line.startswith('{')))


def source(run, out, engine):
    repo, rev = ('https://github.com/eole-nlp/eole.git', EOLE_REV) if engine == 'eole' else ('https://github.com/edwardyoon/focus-llama.git', FOCUS_REV)
    src = out / 'source'
    run.call('clone', ['git', 'clone', '--filter=blob:none', '--no-checkout', repo, src])
    run.call('checkout', ['git', '-C', src, 'checkout', '--detach', rev])
    run.call('submodules', ['git', '-C', src, 'submodule', 'update', '--init', '--recursive'])
    actual = run.call('source-revision', ['git', '-C', src, 'rev-parse', 'HEAD']).strip()
    if actual != rev: raise ValueError('Source revision mismatch')
    return src, dict(repo=repo, revision=rev)


def source_clean(run, src, prefix):
    status = run.call(prefix, ['git', '-C', src, 'status', '--porcelain', '--untracked-files=no']).strip()
    if status: raise ValueError('Upstream tracked source was changed: ' + status)


def pip_resolve(run, py, out, name, requirements, env):
    report = out / (name + '-resolve.json')
    run.call(name + '-resolve', [py, '-B', '-m', 'pip', 'install', '--dry-run', '--report', report, '--no-build-isolation', '-c', out / 'protected-constraints.txt', *requirements], env=env)
    rows = resolved_requirements(json.loads(report.read_text())); save(out / (name + '-locked.json'), rows)
    if rows:
        req = out / (name + '-locked-requirements.txt')
        req.write_text(''.join(row['url'].split('#')[0] + '#sha256=' + row['sha256'] + '\n' for row in rows))
        run.call(name + '-install', [py, '-B', '-m', 'pip', 'install', '--ignore-installed', '--no-deps', '--no-build-isolation', '--require-hashes', '-r', req], env=env)


def hf_model(run, py, out, model, repo, revision, expected=None, include=None):
    # Compare every pinned repository file: LFS SHA256 or exact Git-blob SHA1.
    # Existing directories are read-only and must already be complete.
    code = '''import hashlib,json,pathlib,sys
from huggingface_hub import HfApi,snapshot_download
repo,rev,dest,meta,include=sys.argv[1:]; p=pathlib.Path(dest); names=json.loads(include)
i=HfApi().model_info(repo,revision=rev,files_metadata=True)
assert i.sha==rev, (i.sha,rev)
items=[x for x in i.siblings if names is None or x.rfilename in names]
if not p.exists(): snapshot_download(repo,revision=rev,local_dir=p,allow_patterns=[x.rfilename for x in items])
rows=[]
for x in items:
 f=p/x.rfilename
 assert f.is_file() and f.stat().st_size==x.size, str(f)
 h=hashlib.sha256(); g=hashlib.sha1();g.update(('blob '+str(x.size)+'\\0').encode())
 with f.open('rb') as s:
  for b in iter(lambda:s.read(8<<20),b''):h.update(b);g.update(b)
 if x.lfs: assert h.hexdigest()==x.lfs.sha256, str(f)
 else: assert g.hexdigest()==x.blob_id, str(f)
 rows.append(dict(path=str(f),bytes=x.size,sha256=h.hexdigest(),git_blob=x.blob_id))
pathlib.Path(meta).write_text(json.dumps(dict(repo=repo,revision=rev,files=rows),indent=2)+'\\n')
'''
    meta = out / ('model-' + repo.replace('/', '-') + '.json')
    run.call('model-' + repo.replace('/', '-'), [py, '-B', '-c', code, repo, revision, model, meta, json.dumps(include)])
    data = json.loads(meta.read_text())
    if expected and not any(row['sha256'] == expected for row in data['files']): raise ValueError('Expected target weight not in pinned checkpoint')
    return meta


def eole(run, args, src, env):
    info = run.json('sg-runtime', [args.sg_python, '-B', '-c', 'import sys,sysconfig,torch,triton,json;print(json.dumps(dict(base=sys._base_executable,site=sysconfig.get_paths()["purelib"],python=list(sys.version_info[:2]),torch=torch.__version__,cuda=torch.version.cuda,triton=triton.__version__,abi=torch._C._GLIBCXX_USE_CXX11_ABI,torch_file=torch.__file__)))'])
    if info['python'] != [3, 12] or info['torch'] != '2.14.1+cu130' or info['cuda'] != '13.0' or info['triton'] != '3.8.0': raise ValueError('Shared SG Torch/ABI selection differs from pinned CU130 CP312 plan')
    save(args.out / 'sg-runtime.json', info)
    sgroot = args.sg_python.parent.parent; before = inventory(sgroot); save(args.out / 'sg-before.json', before)
    primary_error = None
    try:
        venv = args.out / 'venv'; run.call('venv', [info['base'], '-m', 'venv', venv]); py = venv / 'bin/python'
        ownsite = Path(run.call('own-site', [py, '-B', '-c', 'import sysconfig;print(sysconfig.get_paths()["purelib"])']).strip())
        if not ownsite.resolve().is_relative_to(venv.resolve()): raise ValueError('New venv site escaped own directory')
        (ownsite / 'sg-frozen-reference.pth').write_text(info['site'] + '\n')
        runtime_env = env | dict(PATH=str(venv / 'bin') + ':' + env['PATH'], VIRTUAL_ENV=str(venv), EOLE_SKIP_CUDA_BUILD='1')
        constraints = run.call('protected-versions', [py, '-B', '-c', 'import importlib.metadata as m;print("\\n".join(sorted(d.metadata["Name"]+"=="+d.version for d in m.distributions() if d.metadata["Name"].lower().replace("_","-") in ("torch","triton") or d.metadata["Name"].lower().startswith("nvidia-"))))'], env=runtime_env)
        (args.out / 'protected-constraints.txt').write_text(constraints)
        pip_resolve(run, py, args.out, 'bootstrap', ['setuptools<85', 'wheel', 'packaging', 'psutil', 'ninja'], runtime_env)
        pip_resolve(run, py, args.out, 'eole-dependencies', ['-e', str(src)], runtime_env)
        run.call('eole-editable', [py, '-B', '-m', 'pip', 'install', '--no-deps', '--ignore-installed', '--no-build-isolation', '-e', src], env=runtime_env)
        wheel = args.out / 'flash_attn-2.8.3+cu130torch2.14-cp312-cp312-linux_x86_64.whl'
        run.call('flashattn-download', ['curl', '--fail', '--location', '--max-time', '300', '--output', wheel, FA_URL])
        if sha(wheel) != FA_SHA: raise ValueError('FlashAttention wheel SHA differs')
        run.call('flashattn-install', [py, '-B', '-m', 'pip', 'install', '--no-deps', '--ignore-installed', wheel], env=runtime_env)
        pip_resolve(run, py, args.out, 'recurrent-dependencies', ['fla-core==0.6.0', 'causal-conv1d'], runtime_env)
        buildenv = runtime_env.copy(); buildenv.pop('EOLE_SKIP_CUDA_BUILD', None)
        run.call('eole-native-build', [py, '-B', 'setup.py', 'build_ext', '--inplace', '--force', '--build-temp', args.out / 'tmp/eole-ops'], cwd=src, env=buildenv)
        run.call('import-native-modules', [py, '-B', '-c', 'import torch,eole._ops,flash_attn,fla,causal_conv1d,json;print(json.dumps({x.__name__:dict(file=x.__file__,version=getattr(x,"__version__",None)) for x in (torch,eole._ops,flash_attn,fla,causal_conv1d)}))'], cwd=src, env=buildenv)
        run.call('pip-check', [py, '-B', '-m', 'pip', 'check'], env=buildenv)
        run.call('eole-help', [venv / 'bin/eole', '--help'], env=buildenv)
        run.call('eole-serve-help', [venv / 'bin/eole', 'serve', '--help'], env=buildenv)
        target_meta = hf_model(run, py, args.out, args.eole_model, 'eole-nlp/Qwen3.8-27B-INT4-eole', MODEL_REV, '59da81f0c03e3f277dff3ef765d5245b90e6d03c2fd5159f1c9e31d6b41c6509')
        draft_meta = hf_model(run, py, args.out, args.hf_draft, 'incoai/Qwen3.8-27B-DFlash2', DRAFT_REV, include=['config.json', 'model.safetensors'])
        run.call('checkpoint-check', [py, '-B', src / 'recipes/qwen38/check_checkpoint.py', args.eole_model], cwd=src, env=buildenv)
        save(args.out / 'native-binaries.json', [row for folder in (src / 'eole', ownsite) for row in inventory(folder) if '.so' in row['path']])
        save(args.out / 'model-manifest.json', dict(engine='eole', target=json.loads(target_meta.read_text()), draft=json.loads(draft_meta.read_text()), native_inventory=dict(path=str(args.out / 'native-binaries.json'), sha256=sha(args.out / 'native-binaries.json'))))
        return dict(EOLE_REPO=str(src), EOLE_VENV=str(venv), EOLE_SERVE=str(venv / 'bin/eole'), EOLE_MODEL=str(args.eole_model), HF_DRAFT=str(args.hf_draft), EOLE_TORCH_COMPILE='1', EOLE_COMPILE_MODE='0', EOLE_PREFILL_CHUNK='2048', MODEL_MANIFEST_JSON=str(args.out / 'model-manifest.json'))
    except BaseException as error:
        primary_error = error
        raise
    finally:
        try:
            after = inventory(sgroot); save(args.out / 'sg-after.json', after)
            if after != before: raise ValueError('Protected SG environment file/link inventory changed')
        except BaseException as error:
            save(args.out / 'sg-postcheck-error.json', dict(error=repr(error), primary_error=repr(primary_error)))
            if primary_error is None: raise


def focus(run, args, src, env):
    build = args.out / 'build'
    run.call('cmake-configure', ['cmake', '-S', src, '-B', build, '-DGGML_CUDA=ON', '-DCMAKE_CUDA_ARCHITECTURES=120a', '-DCMAKE_CUDA_COMPILER=' + str(args.nvcc), '-DCMAKE_BUILD_TYPE=Release', '-DLLAMA_BUILD_TESTS=OFF', '-DLLAMA_BUILD_EXAMPLES=OFF', '-DLLAMA_BUILD_APP=OFF', '-DLLAMA_BUILD_UI=OFF'])
    run.call('cmake-build', ['cmake', '--build', build, '--parallel', '2', '--target', 'llama-server'])
    server = build / 'bin/llama-server'; helptext = run.call('focus-help', [server, '--help'])
    for flag in ('--no-da-prompt-scan', '--no-da-auto', '--no-fm-offload', '--kv-cache-size'):
        if flag not in helptext: raise ValueError('Required full-attention flag missing: ' + flag)
    models = []
    for key, path in dict(target=args.target, draft=args.draft, mmproj=args.mmproj).items():
        actual = sha(path)
        if actual != GGUF_PINS[key]: raise ValueError('GGUF model pin differs: ' + key)
        models.append(dict(role=key, path=str(path), sha256=actual, bytes=path.stat().st_size))
    save(args.out / 'models.json', models); save(args.out / 'native-binaries.json', inventory(build / 'bin'))
    save(args.out / 'model-manifest.json', dict(engine='focus-llama', files=models, native_inventory=dict(path=str(args.out / 'native-binaries.json'), sha256=sha(args.out / 'native-binaries.json'))))
    return dict(FOCUS_LLAMA_REPO=str(src), FOCUS_LLAMA_BIN=str(build / 'bin'), LLAMA_MODEL=str(args.target), LLAMA_DRAFT=str(args.draft), MMPROJ=str(args.mmproj), LLAMA_KV_K='q8_0', LLAMA_KV_V='q8_0', LLAMA_BATCH='2048', LLAMA_UBATCH='2048', LLAMA_CHECKPOINTS='0', LLAMA_CACHE_RAM='0', MODEL_MANIFEST_JSON=str(args.out / 'model-manifest.json'), ENGINE_EXTRA_ARGS_JSON=json.dumps(['--no-da-prompt-scan', '--no-da-auto', '--no-fm-offload', '--kv-cache-size', '0']))


def profiles(args, runtime):
    result = []
    for mode in ('plain', 'dflash2'):
        for size, ctx, lanes, scenarios in [('short', 40960, 1 if args.engine == 'eole' else 4, SHORT), ('long', 262144, 1, LONG)]:
            label = f'{args.label_prefix}{args.engine}-{mode}-{args.out.name}-{size}'
            env = runtime | dict(OUT=str(args.out / 'profiles' / label), CTX=str(ctx), LANES=str(lanes), PY=str(args.bench_python), TOKENIZER=str(args.tokenizer), HAYSTACK=str(args.haystack), NONCE_SEED='20261101', SPEC_DRAFT='7', WARMUP='3', STARTUP_TIMEOUT_SECONDS='3600', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='0', CUDA_HOME=str(args.nvcc.parent.parent), XDG_CACHE_HOME=str(args.out / 'runtime-cache'), TORCHINDUCTOR_CACHE_DIR=str(args.out / 'runtime-cache/inductor'), TRITON_CACHE_DIR=str(args.out / 'runtime-cache/triton'))
            result.append(dict(label=label, environment=env, unset_environment_prefixes=['LLAMA_ARG_'], unset_environment=['QUALITY_BASELINE', 'PYTHONPATH', 'PYTHONHOME', 'ENGINE_EXTRA_ARGS_JSON'], bench_argv=['bash', str(args.harness_root / 'eng/qwen38-engine-servers.sh'), 'bench', args.engine, mode, label, scenarios, '3', '--', '--answer-tokens', '512', '--decode-tokens', '1024', '--temperature', '0', '--top-p', '1', '--sampling-seed', '1234', '--quality-tasks', '8', '--gsm8k-limit', '8', '--long-order-quality', '--keep-text', '12000', '--allow-failures'], readiness_and_cleanup='Unset inherited overrides first, then apply environment. Frozen launcher owns the actual process; unique OUT prevents stopping another run. DO NOT run profiles during setup/formal of another engine.', expected_cases=48 if size == 'short' else 6, quality='NOT_RUN', limitation='Completion is not PASS. Eole stream lacks actual usage and ignores response_format; strict gates remain unchanged.'))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--engine', choices=['eole', 'focus-llama'], required=True); p.add_argument('--out', type=Path, required=True); p.add_argument('--execute', action='store_true'); p.add_argument('--timeout', type=int, default=2400); p.add_argument('--step-timeout', type=int, default=1200); p.add_argument('--lock', type=Path, default=Path('/root/ts-eval-gpu.lock')); p.add_argument('--sg-python', type=Path, default=Path('/root/venvs/sglang/bin/python')); p.add_argument('--nvcc', type=Path, default=Path('/usr/local/cuda/bin/nvcc')); p.add_argument('--eole-model', type=Path, default=Path('/workspace/hf/Qwen3.8-27B-INT4-eole')); p.add_argument('--hf-draft', type=Path, default=Path('/workspace/hf/Qwen3.8-27B-DFlash2')); p.add_argument('--target', type=Path, default=Path('/workspace/models/Qwen3.8-27B-NVFP4-MTP-N4_0.gguf')); p.add_argument('--draft', type=Path, default=Path('/workspace/models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf')); p.add_argument('--mmproj', type=Path, default=Path('/workspace/models/mmproj-BF16.gguf')); p.add_argument('--harness-root', type=Path, default=Path('/root/ts-harness')); p.add_argument('--bench-python', type=Path, default=Path('/root/venvs/bench/bin/python')); p.add_argument('--tokenizer', type=Path, default=Path('/workspace/hf/Qwen3.8-27B-NVFP4-nvidia/tokenizer.json')); p.add_argument('--haystack', type=Path, default=Path('/workspace/ts-eval/artifacts/haystack.txt')); p.add_argument('--label-prefix', default='vm2-'); return p


def main(argv=None):
    args = parser().parse_args(argv); args.out = args.out.absolute()
    if args.timeout <= 0 or args.step_timeout <= 0: raise ValueError('Timeouts must be positive')
    plan = dict(schema='qwen38-optional-engine-setup/1', engine=args.engine, out=str(args.out), source_revision=EOLE_REV if args.engine == 'eole' else FOCUS_REV, execution='NOT_RUN', setup_only=True, gpu='NOT_RUN', quality='NOT_RUN', torch='2.14.1+cu130 / CP312 / Triton3.8.0 read-only SG reference' if args.engine == 'eole' else None, precision='converted RedHat W4A16/native head/BF16 KV; not TS checkpoint' if args.engine == 'eole' else 'Same pinned NVFP4 GGUF + Q8 K/V; DA/FM disabled', future_profiles=profiles(args, {}))
    if not args.execute: print(json.dumps(plan, indent=2)); return 0
    if platform.system() != 'Linux': raise ValueError('Execution requires Linux /proc ownership checks')
    output_guard(args)
    if args.out.exists() or args.out.is_symlink(): raise ValueError('Fresh output directory required')
    # Acquire the same owner lock before any setup writes; CPU installation and
    # compilation must not compete with a formal GPU/CPU measurement window.
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        args.out.mkdir(parents=True, exist_ok=False); (args.out / 'tmp').mkdir()
        env = {k:v for k,v in os.environ.items() if not k.startswith(('PIP_', 'LLAMA_ARG_', 'TS_', 'EOLE_')) and k not in ('PYTHONPATH', 'PYTHONHOME', 'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE')}
        env.update(CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PIP_CONFIG_FILE='/dev/null', PIP_NO_CACHE_DIR='1', PIP_REQUIRE_VIRTUALENV='1', MAX_JOBS='2', CMAKE_BUILD_PARALLEL_LEVEL='2', TORCH_CUDA_ARCH_LIST='12.0', CUDA_HOME=str(args.nvcc.parent.parent), TMPDIR=str(args.out / 'tmp'), HF_HOME=str(args.out / 'hf-cache'), HF_XET_CACHE=str(args.out / 'hf-cache/xet'), XDG_CACHE_HOME=str(args.out / 'cache'), TORCHINDUCTOR_CACHE_DIR=str(args.out / 'cache/inductor'), TRITON_CACHE_DIR=str(args.out / 'cache/triton'))
        run = Runner(args.out, args.timeout, args.step_timeout, env); status = plan | dict(execution='STARTING', started_utc=datetime.now(timezone.utc).isoformat(), tool_sha256=sha(__file__))
        save(args.out / 'status.json', status)
        def interrupted(sig, frame): raise KeyboardInterrupt(f'signal{sig}')
        oldhandlers = {s:signal.signal(s, interrupted) for s in (signal.SIGINT, signal.SIGTERM)}
        src = None
        try:
            for name in ('qwen38-engine-servers.sh', 'qwen38-engine-process.py', 'qwen38-engine-bench.py'):
                path = args.harness_root / 'eng' / name
                status.setdefault('harness', []).append(dict(path=str(path), sha256=sha(path)))
            if sha(args.harness_root / 'eng/qwen38-engine-bench.py') != BENCH_SHA: raise ValueError('Frozen benchmark SHA differs')
            haystack = run.json('benchmark-inputs', [args.bench_python, '-B', '-c', 'import importlib.util,json,sys;spec=importlib.util.spec_from_file_location("frozen_bench",sys.argv[1]);m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);value=m.load_haystack(sys.argv[2]);value.pop("paragraphs");print(json.dumps(value))', args.harness_root / 'eng/qwen38-engine-bench.py', args.haystack])
            if haystack['sha256'] != HAYSTACK_SHA or haystack['chars'] <= 0: raise ValueError('Frozen normalized haystack differs or is empty')
            status['benchmark_inputs'] = dict(haystack=haystack, tokenizer=dict(path=str(args.tokenizer), sha256=sha(args.tokenizer)))
            nvcc_version = run.call('nvcc-version', [args.nvcc, '--version'])
            if 'V13.2.86' not in nvcc_version: raise ValueError('Pinned setup expects NVCC13.2.86; actual version retained in log')
            src, status['source'] = source(run, args.out, args.engine); source_clean(run, src, 'source-precheck')
            runtime = eole(run, args, src, env) if args.engine == 'eole' else focus(run, args, src, env)
            save(args.out / 'runtime-env.json', runtime); save(args.out / 'profile-commands.json', profiles(args, runtime))
            status.update(execution='SETUP_COMPLETE_MODEL_NOT_LOADED', setup_complete=True, readiness='NOT_RUN', quality='NOT_RUN')
        except BaseException as error:
            status.update(execution='ERROR', setup_complete=False, error=repr(error))
        finally:
            if src is not None:
                try: source_clean(run, src, 'source-postcheck')
                except BaseException as error: status.update(execution='ERROR', setup_complete=False, source_postcheck_error=repr(error))
            status['finished_utc'] = datetime.now(timezone.utc).isoformat(); save(args.out / 'status.json', status)
            for sig, handler in oldhandlers.items(): signal.signal(sig, handler)
        print(json.dumps(status, indent=2)); return 0 if status.get('setup_complete') else 1


if __name__ == '__main__': raise SystemExit(main())
