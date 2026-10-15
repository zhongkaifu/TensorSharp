#!/usr/bin/env python3
"""Run a separately activated, receipt-bound text-only Host prefix observation.

This tool never builds/stages binaries or stops another server. Startup wall,
HTTP latency and attributed model timings remain different observations.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value

http = module('prefix_host_http', HERE / 'prefix-warmup-http.py')
bench = module('prefix_host_sampler', HERE / 'multimodel-quality-bench.py')
owned = module('prefix_host_cleanup', HERE / 'llama-planned-chat.py')
SOURCE_FILES = (Path(__file__).resolve(), HERE / 'prefix-warmup-http.py',
                HERE / 'unified-qwen-concurrency-bench.py', HERE / 'verify-fused-batch-evidence.py',
                HERE / 'multimodel-quality-bench.py', HERE / 'linux-cgroup-telemetry.py',
                HERE / 'llama-planned-chat.py')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def pin(path):
    path = Path(path).resolve()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=bench.sha(path))

def save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')

def safe_relative(root, relative):
    require(isinstance(relative, str) and relative and not Path(relative).is_absolute(), 'Invalid receipt-relative path')
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Receipt path escapes its root')
    return path

def closure(root):
    # Logs are deployment data, never members of a staged immutable SDK closure.
    return {p.relative_to(root).as_posix(): dict(Bytes=p.stat().st_size, Sha256=bench.sha(p))
            for p in sorted(root.rglob('*')) if p.is_file() and p.relative_to(root).parts[0] != 'logs'}

def qualify(plan):
    """Require complete SDK delivery and native staging, before any launch."""
    require(plan.get('schema') == 'tensorsharp.prefix-warmup-host-plan.v1', 'Unknown launch-plan schema')
    require(type(plan.get('sdk_version')) is int and plan['sdk_version'] > 0, 'Invalid SDK version')
    require(isinstance(plan.get('activation'), str) and plan['activation'].startswith('ROOT-'), 'Missing explicit root activation')
    managed, native, stage = (Path(plan[k]).resolve() for k in ('managed_directory', 'native_directory', 'stage_directory'))
    paths = dict(delivery=managed / 'final-delivery.json', host=managed / 'host-publish/receipt.json',
                 probe=managed / 'glm-session-sdk-publish/receipt.json',
                 self_test=managed / 'published-probe-self-test-receipt.json',
                 managed_summary=managed / 'final-summary.json', native_summary=native / 'build-summary.json',
                 stage_summary=stage / 'stage-summary.json', deployment=stage / 'deployment-receipt.json')
    require(all(path.is_file() for path in paths.values()), 'Actual complete delivery and staged-native receipts are required; no staging is performed')
    receipts = {key: read(path) for key, path in paths.items()}
    delivery, staged, build = (receipts[k] for k in ('delivery', 'stage_summary', 'native_summary'))
    require(delivery.get('passed') is True and delivery.get('cpu_version') == plan['sdk_version'], 'SDK delivery failed/version differs')
    if plan.get('require_publication_complete'):
        require(delivery.get('publication_complete') is True, 'SDK publication is not complete')
    require(delivery.get('source_dll_control_userbin_bookends_equal') is True
            and delivery.get('published_probe_windows_wsl_selftest_passed') is True, 'SDK bookends/published self-tests unavailable')
    pins = delivery.get('receipt_pins', {})
    mandatory = {'final-summary.json', 'host-publish/receipt.json', 'glm-session-sdk-publish/receipt.json', 'published-probe-self-test-receipt.json'}
    require(mandatory <= pins.keys(), 'SDK delivery lacks mandatory receipt pins')
    for relative, digest in pins.items():
        path = safe_relative(managed, relative)
        require(re.fullmatch('[0-9a-f]{64}', digest or '') and bench.sha(path) == digest, 'SDK receipt pin differs: ' + relative)
        paths['delivery/' + relative] = path
    published = {}
    for name in ('host', 'probe'):
        receipt = receipts[name]
        location = managed / ('host-publish/publish' if name == 'host' else 'glm-session-sdk-publish/publish')
        require(receipt.get('publish_complete') is True and receipt.get('exit_code') == 0
                and receipt.get('compiled_unchanged') is True and receipt.get('user_bin_unchanged') is True
                and receipt.get('project_unchanged' if name == 'host' else 'source_unchanged') is True
                and receipt.get('native_injected') is False, 'Unqualified SDK publish: ' + name)
        require(Path(receipt['publish']).resolve() == location.resolve()
                and Path(delivery[name + '_publish']).resolve() == location.resolve(), 'SDK publish path differs')
        published[name] = {key: dict(Bytes=value['bytes'], Sha256=value['sha256']) for key, value in receipt['published_files'].items()}
        require(closure(location) == published[name], 'SDK published closure differs: ' + name)
    self_test = receipts['self_test']
    require(self_test.get('passed') is True and self_test.get('published_unchanged') is True
            and self_test.get('published_before') == self_test.get('published_after')
            and self_test.get('publish_receipt_sha256') == bench.sha(paths['probe']), 'Published Probe self-test is not identity-qualified')
    require(len(self_test.get('cases', [])) == 2 and {row.get('platform') for row in self_test['cases']} == {'windows', 'wsl'}
            and all(row.get('exit_code') == 0 and row.get('owned_process_reaped') is True for row in self_test['cases']), 'Published Probe Windows/WSL self-tests unavailable')
    require({key: dict(bytes=value['Bytes'], sha256=value['Sha256']) for key, value in published['probe'].items()}
            == self_test.get('published_before'), 'Published Probe differs from self-tested closure')
    require(build.get('passed') is True and build.get('before') == build.get('after')
            and build.get('no_upstream_compilation') is True, 'Native build is not identity-qualified against unchanged upstream')
    upstream = build.get('before', {}).get('upstream', {})
    require(upstream.get('revision') == plan.get('upstream_revision') and upstream.get('status') == '', 'Native dependency revision/clean upstream differs')
    for name, expected in plan.get('production_capabilities', {}).items():
        require(build.get('production_constant_capability_queries', {}).get(name) == expected, 'Native production capability differs: ' + name)
    server = stage / 'host'
    require(staged.get('Passed') is True and Path(staged['HostDirectory']).resolve() == server.resolve(), 'Unqualified Host staging path')
    for field, key in (('CompleteDeliverySha256', 'delivery'), ('PublishedSelfTestSha256', 'self_test'),
                       ('NativeBuildSummarySha256', 'native_summary'), ('ManagedSummarySha256', 'managed_summary')):
        require(staged.get(field) == bench.sha(paths[key]), 'Stage receipt pin differs: ' + field)
    require(staged.get('PublishReceiptSha256') == {key: bench.sha(paths[key]) for key in ('host', 'probe')}, 'Stage publish receipt pins differ')
    require(staged.get('Original', {}).get('host') == published['host'], 'Stage original Host differs from completed SDK publication')
    native_binary = native / 'build-production/GgmlOps.dll'
    native_sha = build.get('production_artifacts', {}).get('GgmlOps.dll')
    require(native_sha == bench.sha(native_binary), 'Native production binary differs from its completed build')
    paths['production_native'] = native_binary
    expected = dict(published['host'])
    require('GgmlOps.dll' not in expected, 'SDK closure already contains an injected native library')
    expected['GgmlOps.dll'] = dict(Bytes=native_binary.stat().st_size, Sha256=native_sha)
    require(staged.get('HostFiles') == expected and closure(server) == expected, 'Full staged Host closure differs')
    source_pins = plan.get('source_pins', {})
    require({str(path.resolve()) for path in SOURCE_FILES} <= source_pins.keys(), 'Plan does not bind the complete executing validation controller/helper closure')
    for path, expected_pin in source_pins.items():
        require(pin(path) == expected_pin, 'Executing validation source differs: ' + path)
        paths['tool/' + path] = Path(path)
    return dict(server=str(server), native_sha256=native_sha, expected_closure=expected,
                receipt_and_tool_pins={key: pin(path) for key, path in paths.items()},
                source_scope='Completed SDK source/control inventories and immutable staged closure; does not claim the current worktree equals an older SDK source snapshot.')

def model_stats(model):
    return [dict(path=str(path.resolve()), bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns)
            for path in bench.checkpoint_files(model)]

def child_environment(inherited, log_directory, prefix_directory, startup_warmup_thinking=None, validation_token_trace=False):
    overrides = {key: value for key, value in inherited.items() if value and re.match(r'^(TS_|TENSORSHARP_|GGML_|MAX_CONTEXT$|MAX_TOKENS$|KV_CACHE_|CUDA_VISIBLE|OMP_|BACKEND$)', key)}
    require(not overrides, 'Inherited inference override would change the default configuration: ' + ', '.join(sorted(overrides)))
    settings = dict(MAX_CONTEXT='32768', TENSORSHARP_LOG_DIR=str(log_directory), TENSORSHARP_PREFIX_CACHE_DIR=str(prefix_directory))
    require(startup_warmup_thinking is None or startup_warmup_thinking in ('0', '1'),
            'Startup warmup thinking override must be an explicit0/1 string')
    if startup_warmup_thinking is not None:
        settings['TS_STARTUP_WARMUP_THINKING'] = startup_warmup_thinking
    require(type(validation_token_trace) is bool, 'Validation token trace must be an explicit boolean')
    if validation_token_trace:
        settings['TS_VALIDATION_TOKEN_TRACE'] = '1'
    result = dict(inherited); result.update(settings)
    return result, settings

def thinking_startup(state, plan):
    """An explicit one-slot plan must observe its actual preferred stage."""
    if not plan.get('require_single_preferred_variant'):
        return
    expected = plan.get('startup_warmup_thinking') != '0'
    status = state.get('startupWarmup', {})
    stages = status.get('stages', [])
    require(status.get('preferredThinking') is expected
            and status.get('singleVariantReason') == 'single-public-checkpoint-budget'
            and len(stages) == 1 and stages[0].get('thinking') is expected
            and stages[0].get('completed') is True and status.get('completed') is True,
            'Actual startup did not complete its single preferred thinking variant')

def followup_token_limit(plan):
    value = plan.get('follow_up_max_tokens')
    require(value is None or type(value) is int and value in (8, 16, 64, 128),
            'Invalid bounded followup token cap')
    require(value is None or plan.get('client_visible_follow_up') is True,
            'Followup token cap requires this launcher\'s explicit client-visible followup')
    return value

def long_new_chat_limit(plan):
    value = plan.get('long_new_chat_max_tokens')
    expected = plan.get('long_new_chat_expected_cached_tokens')
    require(value is None or type(value) is int and value == 256, 'Long new-chat probe requires bounded256 cap')
    require((value is None and expected is None) or value == 256 and type(expected) is int and expected > 0,
            'Long new-chat probe requires an exact positive public cached-token expectation')
    require(value is None or plan.get('thinking') == 'on' and plan.get('output_gate') == 'stream'
            and not (plan.get('client_visible_follow_up') or plan.get('follow_up') or plan.get('cancellation')),
            'Long new-chat probe is a separate thinking-on stream observation without followup/cancellation')
    return value, expected

def http_timeout(plan):
    value = plan.get('http_timeout_seconds', 240)
    require(type(value) in (int, float) and math.isfinite(value) and 1 <= value <= 1200,
            'HTTP timeout must be finite and bounded1..1200 seconds')
    return value

def token_trace_control(enabled):
    require(type(enabled) is bool, 'Validation token trace must be an explicit boolean')
    return dict(enabled=enabled, trace_disqualifies_performance_comparison=enabled,
        scope='Explicit owned-child raw-token diagnostic. Trace-enabled timing is diagnostic only. Absence of this trace does not establish performance comparability, quiet hardware or token identity.')

def server_log_path(directory):
    logs = sorted(Path(directory).glob('tensorsharp-server-*.jsonl'))
    require(logs, 'Expected a fresh structured Host log family')
    family = http.logger_paths(logs[0])
    require(set(family) == {path.resolve() for path in logs},
            'Fresh Host structured log directory differs from its canonical rotation family')
    return family[0]

def command_output(command):
    value = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, creationflags=bench.PROCESS_FLAGS)
    require(value.returncode == 0, 'Preflight command failed: ' + str(command))
    return dict(command=command, exit_code=value.returncode, stdout=value.stdout, stderr=value.stderr)

def gpu_background(owned_pid=None):
    query = command_output(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])
    owners = []
    for cells in csv.reader(query['stdout'].splitlines()):
        require(len(cells) == 2 and cells[0].strip().isdigit(), 'Unparseable GPU process owner; absence of competing owners is unknown')
        pid = int(cells[0].strip())
        if pid == owned_pid: continue
        process = bench.psutil.Process(pid)
        owners.append(dict(pid=pid, birth=process.create_time(), name=process.name(), executable=str(Path(process.exe()).resolve())))
    return sorted(owners, key=lambda row: row['pid']), query

def gpu_allowance(expected, owned_pid=None):
    require(isinstance(expected, list), 'An explicit desktop GPU identity snapshot is required')
    for row in expected:
        require(isinstance(row, dict) and set(row) == {'pid', 'birth', 'name', 'executable'}
                and type(row['pid']) is int and row['pid'] > 0
                and type(row['birth']) in (float, int) and math.isfinite(row['birth']) and row['birth'] > 0
                and isinstance(row['name'], str) and isinstance(row['executable'], str)
                and Path(row['executable']).is_absolute(), 'Invalid background GPU process identity')
        require(not re.search(r'tensorsharp.*host|llama.*server|glm.*probe|strata|maya|^dotnet(?:\.exe)?$', row['name'], re.I), 'A model/probe cannot be allowed as desktop GPU background')
    require(len({row['pid'] for row in expected}) == len(expected), 'Duplicate background GPU PID')
    actual, query = gpu_background(owned_pid)
    require(actual == sorted(expected, key=lambda row: row['pid']), 'GPU owners differ from the exact desktop background allowance; no process is interrupted')
    return dict(owners=actual, query=query, scope='Identity-bound preexisting WDDM desktop clients remain active. Their GPU traffic is unquantified; this is not an idle or exclusive GPU claim. New/unlisted/reused-PID owners fail.')

def quiet_lane(port, background_gpu):
    occupied = [row for row in bench.psutil.net_connections(kind='tcp') if row.status == bench.psutil.CONN_LISTEN and row.laddr.port == port]
    require(not occupied, 'Requested port is occupied; no existing server is stopped')
    competitors = []
    names = {'tensorsharp.server.host', 'tensorsharp.server.host.exe', 'llama-server', 'llama-server.exe',
             'glmsessionbudgetprobe', 'glmsessionbudgetprobe.exe', 'glmgraphbudgetprobe', 'glmgraphbudgetprobe.exe', 'strata', 'strata.exe'}
    for process in bench.psutil.process_iter(['pid', 'name']):
        if (process.info.get('name') or '').lower() in names: competitors.append(process.info)
    require(not competitors, 'A model/server process already exists; no process is interrupted: ' + repr(competitors))
    gpu = gpu_allowance(background_gpu)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', port))
    return dict(port_available=True, model_processes=competitors, background_gpu=gpu,
                scope='No recognized existing model/server, exclusive free port, exact allowed desktop GPU owners; not a globally idle device.')

def process_identity(process, expected_executable, birth):
    require(process.is_running() and process.create_time() == birth and Path(process.exe()).resolve() == expected_executable.resolve(), 'Owned Host identity changed')
    return dict(pid=process.pid, birth=birth, executable=process.exe(), cmdline=process.cmdline(),
                memory=process.memory_info()._asdict(), cpu=process.cpu_times()._asdict())

def listener(process, port):
    rows = [row for row in bench.psutil.net_connections(kind='tcp') if row.status == bench.psutil.CONN_LISTEN and row.laddr.port == port]
    require(rows and all(row.pid == process.pid for row in rows), 'Port is not exclusively owned by this Host')
    return [dict(address=row.laddr.ip, port=row.laddr.port, pid=row.pid) for row in rows]

def startup_state(state, model, architecture):
    return (state.get('unifiedMemory') is True and state.get('architecture') == architecture
            and state.get('loaded') == model.name and state.get('visionReady') is False
            and state.get('startupWarmup', {}).get('completed') is True)

class OwnedDescendants:
    """Track only proven descendants; freeze the live root before final enumeration.

    This is ownership-checked cleanup, not an OS process-tree containment claim.
    A crashed/expired root makes exhaustive descendant qualification unavailable.
    """
    def __init__(self, root, executable, birth):
        self.root, self.executable, self.birth = root, executable, birth
        self.seen, self.errors = {}, []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    @staticmethod
    def matches(process, identity):
        return process.is_running() and process.create_time() == identity['birth'] \
            and str(Path(process.exe()).resolve()) == identity['executable']

    def capture(self):
        process_identity(self.root, self.executable, self.birth)
        for process in self.root.children(recursive=True):
            try:
                identity = dict(pid=process.pid, birth=process.create_time(), executable=str(Path(process.exe()).resolve()))
                self.seen[(identity['pid'], identity['birth'])] = (process, identity)
            except bench.psutil.NoSuchProcess:
                pass

    def run(self):
        while not self.stop.wait(.5):
            try:
                self.capture()
            except bench.psutil.NoSuchProcess:
                return
            except Exception as error:
                self.errors.append('descendant observation: ' + repr(error)); return

    def finish(self):
        self.stop.set()
        if self.thread.ident is not None: self.thread.join(timeout=2)
        errors = list(self.errors)
        if self.thread.is_alive(): errors.append('descendant observer did not stop')
        exhaustive = False
        try:
            # The owned Host cannot start another tool while its threads are
            # suspended. Freeze any existing descendants before a stable scan.
            process_identity(self.root, self.executable, self.birth)
            self.root.suspend()
            for attempt in range(4):
                self.capture(); count = len(self.seen)
                require(count <= 128, 'Descendant cleanup enumeration bound exceeded')
                for process, identity in list(self.seen.values()):
                    try:
                        if self.matches(process, identity): process.suspend()
                    except bench.psutil.NoSuchProcess: pass
                self.capture()
                if len(self.seen) == count:
                    exhaustive = True; break
            require(exhaustive, 'Descendant set did not become stable while the owned tree was frozen')
        except Exception as error:
            errors.append('exhaustive descendant scan unavailable: ' + repr(error))
        processes = []
        for process, identity in reversed(list(self.seen.values())):
            try:
                if self.matches(process, identity):
                    process.terminate(); processes.append(process)
                elif process.is_running():
                    errors.append('Observed descendant identity changed; refused to stop PID ' + str(identity['pid']))
            except bench.psutil.NoSuchProcess: pass
            except Exception as error: errors.append('owned descendant terminate: ' + repr(error))
        _, alive = bench.psutil.wait_procs(processes, timeout=5)
        for process in alive:
            try:
                identity = next(value for item, value in self.seen.values() if item is process)
                require(self.matches(process, identity), 'Descendant identity changed before final kill')
                process.kill()
            except bench.psutil.NoSuchProcess: pass
            except Exception as error: errors.append('owned descendant kill: ' + repr(error))
        _, alive = bench.psutil.wait_procs(alive, timeout=5)
        if alive: errors.append('Owned descendants did not exit: ' + repr([p.pid for p in alive]))
        return dict(passed=exhaustive and not errors, errors=errors,
                    observed=[identity for _, identity in self.seen.values()], live_after=[p.pid for p in alive],
                    root_frozen_for_cleanup=exhaustive,
                    scope='Only descendants observed under the identified owned Host are stopped after PID/birth/executable checks. Passing requires a stable live suspended-tree scan of current parentage; it does not prove discovery of children reparented before observation. No OS containment or proof about descendants after an unobserved root crash.')

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--activation', required=True)
    args = parser.parse_args(argv)
    plan = read(args.plan)
    require(args.activation == plan.get('activation'), 'This launch requires its exact root activation')
    require(type(plan.get('port')) is int and 1024 <= plan['port'] <= 65535, 'Invalid owned port')
    require(plan.get('cache_expectation') in ('observe', 'enabled', 'disabled'), 'Invalid cache control')
    thinking, output_gate = plan.get('thinking', 'off'), plan.get('output_gate', 'visible')
    require(thinking in ('on', 'off') and output_gate in ('visible', 'stream'), 'Invalid explicit thinking/output gate')
    require(output_gate != 'stream' or thinking == 'on', 'Stream gate requires thinking on')
    require(output_gate != 'stream' or not (plan.get('client_visible_follow_up') or plan.get('follow_up')),
            'Thinking stream scope cannot qualify production-history continuation')
    follow_up_max_tokens = followup_token_limit(plan)
    long_new_chat_max_tokens, long_new_chat_expected_cached_tokens = long_new_chat_limit(plan)
    request_timeout = http_timeout(plan)
    validation_token_trace = plan.get('validation_token_trace', False)
    require(type(validation_token_trace) is bool, 'Validation token trace must be an explicit boolean')
    timeout, ready_timeout = plan.get('lifetime_seconds', 1500), plan.get('ready_timeout_seconds', 600)
    require(type(timeout) in (float, int) and math.isfinite(timeout) and 60 <= timeout <= 3600
            and type(ready_timeout) in (float, int) and 1 <= ready_timeout < timeout, 'Invalid bounded lifetime')
    output = bench.evidence_directory(plan['output_directory'])
    require(not output.exists(), 'Preserve earlier evidence; output must be fresh')
    output.mkdir(parents=True)
    report = dict(schema='tensorsharp.prefix-warmup-host.v1', started_utc=datetime.now(timezone.utc).isoformat(),
                  passed=False, errors=[], plan_pin=pin(args.plan), scope=plan['scope'],
                  gpu_parallelism_claim=False, private_kv_state_identity_claim=False,
                  model_content_identity_scope='All split GGUF paths/sizes/mtime before/after; no fresh 93GB payload hash or cold-storage claim.')
    server = sampler = watchdog = process = descendants = None
    qualified = model_before = None
    stdout = stderr = None
    expired = threading.Event()
    try:
        qualified = qualify(plan)
        report['qualification_before'] = qualified
        server_dir = Path(qualified['server']); exe = server_dir / 'TensorSharp.Server.Host.exe'
        model = Path(plan['model']).resolve()
        model_before = model_stats(model); report['model_before'] = model_before
        environment, settings = child_environment(os.environ, output / 'server-logs', output / 'fresh-prefix-cache',
                                                   plan.get('startup_warmup_thinking'), validation_token_trace)
        report['child_environment'] = settings
        report['token_trace_control'] = token_trace_control(validation_token_trace)
        report['thinking_control'] = dict(request_thinking=thinking, output_gate=output_gate,
            startup_override=plan.get('startup_warmup_thinking'),
            scope='A null startup override preserves the actual Host default. Request mode and startup preference are recorded independently.')
        report['prefix_storage_scope'] = 'Fresh per-run disk directory for both arms; startup includes model load and default warmup. No preexisting disk checkpoint is assumed.'
        report['model_lane_preflight'] = quiet_lane(plan['port'], plan.get('background_gpu_allowance'))
        report['hardware'] = dict(platform=platform.platform(), python=sys.version, cpu=platform.processor(),
            logical_cpus=bench.psutil.cpu_count(), physical_cpus=bench.psutil.cpu_count(logical=False),
            virtual_memory=bench.psutil.virtual_memory()._asdict(), gpu=command_output(['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv']))
        command = [str(exe), '--model', str(model), '--backend', 'ggml_cuda', '--unified-memory',
                   '--max-tokens', '20000', '--host', '127.0.0.1', '--port', str(plan['port'])]
        report['command'] = command; report['mmproj'] = None; report['modality'] = 'text-only'
        # Immutable preflight can be pinned by the HTTP observer before final cleanup.
        save(output / 'preflight.json', report)
        stdout = (output / 'stdout.log').open('xb'); stderr = (output / 'stderr.log').open('xb')
        launch_at = time.perf_counter()
        server = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr, creationflags=bench.PROCESS_FLAGS)
        process = bench.psutil.Process(server.pid); birth = process.create_time()
        report['owned_process'] = process_identity(process, exe, birth); save(output / 'owned-process.json', report['owned_process'])
        descendants = OwnedDescendants(process, exe, birth); descendants.thread.start()
        def expire():
            expired.set()
            if server.poll() is None: server.kill()
        watchdog = threading.Timer(timeout, expire); watchdog.daemon = True; watchdog.start()
        sampler = bench.Sampler(output / 'resource-samples.jsonl'); sampler.process = process
        sampler.sample(); sampler.thread.start()
        client = http.common.Client('http://127.0.0.1:' + str(plan['port']), output, 2)
        while True:
            require(server.poll() is None and not expired.is_set(), 'Owned Host exited/expired during startup')
            require(time.perf_counter() - launch_at < ready_timeout, 'Owned Host readiness timeout')
            try:
                state = client.json('GET', '/api/models')
                if startup_state(state, model, plan['architecture']): break
            except (OSError, ValueError, json.JSONDecodeError): pass
            time.sleep(.25)
        report['startup_wall_seconds'] = time.perf_counter() - launch_at
        report['startup_wall_scope'] = 'Process launch to model/unified/text-only HTTP readiness, including startup warmup; not new-chat TTFT or cold SSD latency.'
        report['initial_model'] = state; save(output / 'initial-model.json', state)
        thinking_startup(state, plan)
        report['ready_process'] = process_identity(process, exe, birth)
        report['listener_before'] = listener(process, plan['port'])
        report['background_gpu_ready'] = gpu_allowance(plan['background_gpu_allowance'], server.pid)
        loaded = bench.native_module_paths(server.pid)
        require(len(loaded) == 1 and Path(loaded[0]).resolve() == (server_dir / 'GgmlOps.dll').resolve()
                and bench.sha(loaded[0]) == qualified['native_sha256'], 'Actual loaded native library differs from staged production')
        report['loaded_native'] = pin(loaded[0])
        save(output / 'ready-process.json', report['ready_process'])
        structured_log = server_log_path(output / 'server-logs')
        arguments = ['--base-url', 'http://127.0.0.1:' + str(plan['port']), '--output', str(output / 'http'),
                     '--server-log', str(structured_log), '--provenance', str(output / 'preflight.json'), '--think', thinking,
                     '--output-gate', output_gate,
                     '--max-tokens', '16', '--cache-expectation', plan['cache_expectation'], '--timeout', str(request_timeout), '--label', plan['label']]
        if plan.get('cancellation'): arguments.append('--cancellation')
        if plan.get('client_visible_follow_up'): arguments.append('--client-visible-follow-up')
        if follow_up_max_tokens is not None:
            arguments.extend(['--follow-up-max-tokens', str(follow_up_max_tokens)])
        if long_new_chat_max_tokens is not None:
            arguments.extend(['--long-new-chat-max-tokens', str(long_new_chat_max_tokens),
                              '--long-new-chat-expected-cached-tokens', str(long_new_chat_expected_cached_tokens)])
        report['http_arguments'] = arguments
        report['http_exit_code'] = http.main(arguments)
        measured = read(output / 'http/report.json')
        require(report['http_exit_code'] == 0 and measured.get('passed') is True, 'HTTP prefix/visible consistency gate failed')
        if plan.get('require_first_new_chat_cached'):
            require(measured['serial'][0]['usage']['prompt_tokens_details']['cached_tokens'] > 0, 'First new chat did not observe startup prefix reuse')
        if plan.get('require_all_new_chats_cached'):
            require(all(row['usage']['prompt_tokens_details']['cached_tokens'] > 0 for row in measured['serial'][:3]),
                    'A new chat did not observe public prefix reuse')
        report['http_report_pin'] = pin(output / 'http/report.json')
        report['first_new_chat_cached_tokens'] = measured['serial'][0]['usage']['prompt_tokens_details']['cached_tokens']
        report['listener_after'] = listener(process, plan['port'])
        report['final_process'] = process_identity(process, exe, birth)
        report['background_gpu_after_http'] = gpu_allowance(plan['background_gpu_allowance'], server.pid)
        sampler.sample()
        require(not expired.is_set(), 'Whole Host lifetime watchdog expired')
        report['execution_complete'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if descendants:
            try:
                report['owned_descendant_cleanup'] = descendants.finish()
                require(report['owned_descendant_cleanup']['passed'], 'Owned descendant cleanup/identity qualification failed')
            except Exception as error:
                report['errors'].append('Descendant cleanup: ' + type(error).__name__ + ': ' + str(error))
        report.update(owned.cleanup_owned(server, watchdog, sampler))
        if stdout: stdout.close()
        if stderr: stderr.close()
        try:
            if qualified:
                report['qualification_after'] = qualify(plan)
                require(qualified == report['qualification_after'], 'Receipts/source/staged binary bookends changed')
            if model_before:
                report['model_after'] = model_stats(Path(plan['model']))
                require(model_before == report['model_after'], 'Checkpoint file metadata changed')
            report['plan_pin_after'] = pin(args.plan)
            require(report['plan_pin'] == report['plan_pin_after'], 'Launch plan changed')
        except Exception as error:
            report['errors'].append('Bookends: ' + type(error).__name__ + ': ' + str(error))
        if sampler:
            report['resources'] = bench.sampled_memory_summary(sampler.rows, sampler.errors)
            report['resource_scope'] = 'Sampled process CPU times/RSS/private and board VRAM across load, warmup and HTTP; not complete physical ownership, instantaneous peaks, or a hard-limit certificate.'
            report['resource_observation_complete'] = bool(sampler.rows) and not sampler.errors and any('cpu' in row and 'memory' in row for row in sampler.rows)
        report['watchdog_expired'] = expired.is_set()
        report['passed'] = report.get('execution_complete') is True and report.get('owned_server_exited') is True \
            and report.get('sampler_stopped') is True and report.get('resource_observation_complete') is True \
            and not report['errors'] and not report.get('cleanup_errors') and not expired.is_set()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output / 'launch-audit.json', report)
    print(json.dumps({key: report.get(key) for key in ('passed', 'errors', 'cleanup_errors', 'startup_wall_seconds', 'owned_server_exited', 'resource_observation_complete')}, ensure_ascii=False))
    return int(not report['passed'])

if __name__ == '__main__':
    raise SystemExit(main())
