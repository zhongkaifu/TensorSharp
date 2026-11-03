"""Literal frozen helper subset for controlled CUA validation; no executable main."""
import datetime as dt, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path
ORIGIN_SHA256='b8872d490db0e448a2ac00f9ffbebd3281d2c2c61274f258120436e17ee75965'
FROZEN_FUNCTION_AST_SHA256={'require': 'f46168cac2c1405d810c7d6199474e7314b55d4b1d230fe16c95eab3f5925a8d', 'sha': 'af88b5889f179d898121d5e6fcc95aee41b915ce7ebf55dd827229b778ca233c', 'read': '165365da25c105273b069e139db74406526caba3b4ae1c2885a35919c4ebf7fe', 'save': '721cf70f59f291393b9b2d352930810c9fae43343151c7afb2c36ee477231316', 'normalized': 'fad6d6b0e35d30e6e4ee0b04196b2f705a73eaf848e6035d74ca412387bfddb3', 'local_file': '373c790de711f6c19dc55393b30c23eb70a04d88e523161044dd6e4206e5d8c2', 'work_scope': 'e70898f226f4ebe83848e589dd408761f3fa1b186e8d8daf04de17889071aaf6', 'inventory': '905fd9554fdeda9266f2bb29966c9eaf5a416a0302541f19a308c6ea0e0e59b4', 'served_binary_map': 'cd8b68db55c30c86882b59021c75b2ad9f6c2d3a3a5871a79ba7c916f809d928', 'source_snapshot': '878a36a53b429716ccdbf6101e4bdfbbc0ff7c6ebff86fe1c34587b76f361861', 'module': '038f6e8cd1c403e56d5f9207cd214dc7eee2f2b347fb79b2712a6fae95af1907', 'require_lease': 'fd5733048110bcd9377d2f3448cdb5ea38ce664a3964cac582a58640dfc8cc2a', 'idle': '20f97806582ec84770a28936bcb350b3b437d02ae76ae6c28244085d6fb30fca', 'dependency': 'c4f70d655dfc92c8360146a806db29cecb63f926ebc0a1422197192972ff69ca', 'environment': 'd6c5cdce07c750e13287167b72af18ecd83a2836847810245a57a08afeaade8a'}

PINS = {
    'eng/qwen38-engine-bench.py': '4e1fa314732250648f43a06d62c02d82b9c17e8842edfeda812f969aac8c363c',
    'eng/qwen38-engine-report.py': 'fbd215eb170d8de33190d0eb5e3ca8e32e9126b132daaa327947cdf977476d3c',
    'eng/qwen38-engine-servers.sh': '966034de15a0a85903445a7e4f0aa24ce376bb529b68bcaac87987c0f846745e',
    'eng/qwen38-engine-process.py': '7e7c22eef60606d3c9e06bc0677b0a801793e4c2b71e5c21279fe4310409c7af'}


UPSTREAM = 'facdc44a67da17df4433adcc468b424f48e9ded2'


UUID = 'GPU-d0e9b3a8-d23f-0967-761d-fdf2369da2e3'


MODEL_PINS = {'Qwen3.8-27B-NVFP4-MTP-N4_0.gguf': 'd5659ecb29adaf75f313fdae03eb7ff17baf3f986f0c3b1348e8afda1670c9a8',
              'Qwen3.8-27B-DFlash2-Q4_K_M.gguf': '1a25c56858e1ebe93f2718ac1d49d1151f9323325c1bbfd6209370f4db131ebd',
              'mmproj-BF16.gguf': 'de2a49866988ea272c43dc2a43ee2662c25725a717ca5b9059ea420a97f53fe8'}


ALT = ('TS_Q35_CUDA_GRAPHS', 'TS_Q35_VERIFY_GRAPHS', 'TS_Q35_TOPK_PARTITIONED',
       'TS_Q35_GDN_PREPARED_PREFILL', 'TS_Q35_GREEDY_VERIFY', 'TS_Q35_VERIFY_DECODE_Q81',
       'TS_Q35_PROJECTION_TRACE', 'TS_Q35_SHARED_SCRATCH', 'TS_Q35_RECLAIM_VISION_POOL')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def normalized(path):
    path = Path(path)
    require(path.is_absolute() and str(path) == os.path.normpath(str(path)), 'absolute normalized path required: ' + str(path))
    for ancestor in (path, *path.parents):
        require(not ancestor.is_symlink(), 'aliased directory/path: ' + str(ancestor))
    return path


def local_file(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe relative inventory path')
    return root / name


def work_scope(work, protected):
    work = normalized(work)
    resolved = work.resolve()
    for path in protected:
        path = Path(path).absolute()
        for destination, root in ((work, path), (resolved, path.resolve())):
            require(not destination.is_relative_to(root) and not root.is_relative_to(destination), 'work overlaps protected source/runtime/harness/models: ' + str(path))
    return work


def inventory(root):
    return [dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=sha(p),
                 symlink_target=str(p.resolve()) if p.is_symlink() else None)
            for p in sorted(root.rglob('*')) if p.is_file()]


def served_binary_map(runtime, rows):
    """Frozen process7e resolves/deduplicates provider symlinks in manifests."""
    result = {}
    for row in rows:
        paths = [str(local_file(runtime, row['path']))]
        if row.get('symlink_target') is not None:
            require(Path(row['symlink_target']).is_absolute(), 'runtime provider target must be absolute')
            paths.append(row['symlink_target'])
        value = dict(sha256=row['sha256'], size_bytes=row['bytes'])
        for path in paths:
            require(path not in result or result[path] == value, 'conflicting aliases for resolved runtime provider')
            result[path] = value
    return result


def source_snapshot(source, rows):
    require(isinstance(rows, list) and rows and len({r['path'] for r in rows}) == len(rows), 'nonempty unique source catalog required')
    result = []
    for row in rows:
        path = local_file(source, row['path'])
        require(path.resolve().is_relative_to(source), 'catalog source escaped supplied source root')
        require(path.is_file() and not path.is_symlink(), 'missing/aliased catalog source: ' + str(path))
        actual = dict(path=row['path'], bytes=path.stat().st_size, sha256=sha(path))
        require(all(actual[k] == row[k] for k in actual), 'catalog source bytes changed: ' + row['path'])
        result.append(actual)
    return result


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def require_lease(path, seconds):
    value = read(path)
    now = dt.datetime.now(dt.timezone.utc)
    start = dt.datetime.fromisoformat(value['not_before_utc'].replace('Z', '+00:00'))
    end = dt.datetime.fromisoformat(value['expires_utc'].replace('Z', '+00:00'))
    require(value.get('schema') == 'qwen38-root-gpu-lease/2' and value.get('owner') == 'root'
            and value.get('active') is True and value.get('exclusive') is True and value.get('gpu_uuid') == UUID
            and value.get('delegated_executor') in (None, '', 'root') and start.tzinfo is not None and end.tzinfo is not None
            and start <= now and (end - now).total_seconds() > seconds + 45, 'active ROOT-only lease cannot cover operation/cleanup')
    return dict(path=str(path), sha256=sha(path), observed=value)


def idle(wait=0):
    end = time.monotonic() + wait
    while True:
        budget = min(3, max(.1, end - time.monotonic())) if wait else 3
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, timeout=budget)
        require(result.returncode == 0, 'cannot verify GPU idle: ' + result.stderr)
        if not result.stdout.strip():
            return
        require(time.monotonic() < end, 'GPU busy; never kill unknown GPU PIDs: ' + result.stdout.strip())
        time.sleep(.25)


def dependency(source):
    path = source / 'ExternalProjects/ggml'
    run = lambda *args: subprocess.check_output(['git', '--no-optional-locks', '-C', str(path), *args], text=True)
    value = dict(path=str(path), resolved=str(path.resolve()), revision=run('rev-parse', 'HEAD').strip(), status=run('status', '--porcelain'))
    require(value['revision'] == UPSTREAM and not value['status'], 'unchanged clean upstream ggml required')
    return value


def environment(args, job, directory):
    env = {key: os.environ[key] for key in ('LANG', 'LC_ALL', 'TZ') if key in os.environ}
    env.update(PATH='/root/dotnet:/usr/local/cuda/bin:/usr/bin:/bin', DOTNET_ROOT='/root/dotnet',
               LD_LIBRARY_PATH='/usr/local/cuda/lib64:/root/cuda-13.4/lib', TMPDIR=str(args.work),
               CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', TS_BIN=str(args.runtime), TS_REPO=str(args.source),
               TENSORSHARP_REPO=str(args.source), OUT=str(directory / 'server'), MODELS=str(args.models),
               TOKENIZER=str(args.tokenizer), HAYSTACK=str(args.haystack), MODEL_MANIFEST_JSON=str(args.model_manifest),
               PY=str(args.bench_python), CONTROL_PY=sys.executable, TS_PORT=str(args.port), TS_BACKEND='cuda',
               CTX=str(job['context']), LANES=str(job['lanes']), SPEC_DRAFT='7', SPEC_PMIN='0',
               TS_PREFIX_CACHE_ISOLATION='lineage', TS_Q35_UBATCH='4096', GGML_CUDA_PDL='0',
               STARTUP_TIMEOUT_SECONDS=str(args.startup_timeout), STOP_TIMEOUT='30')
    env.update({key: '0' for key in ALT})
    env.update(job['env'])
    for key, name in [('TENSORSHARP_PREFIX_CACHE_DIR', 'cache'), ('TENSORSHARP_UPLOAD_DIR', 'uploads'),
                      ('TENSORSHARP_LOG_DIR', 'logs'), ('TS_STUDIO_DIRECTORY', 'studio'), ('ASPNETCORE_WEBROOT', 'wwwroot')]:
        env[key] = str(directory / name)
    env['MANIFEST_ENV_ALLOWLIST'] = ','.join(sorted(set(env) | {'TS_Q35_DRAFT_SPLITK', 'TS_Q35_GEMM_SPLITK'}))
    return env


