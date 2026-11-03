"""Fixed Linux SSH probe supervisor for explicitly authorized distributed profiles.

No discovery, deployment, key installation, arbitrary commands, or model download.
Remote Python 3, prebuilt runtimes, authenticated SSH-agent access and known hosts
must already exist. Model/runtime manifests are hashed before any model launch.
"""
from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

if __package__:
    from .processes import ManagedProcess, run_process, containment_failure_evidence
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from regression.processes import ManagedProcess, run_process, containment_failure_evidence

MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}")
_HOST = re.compile(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}|[0-9A-Fa-f:]+")
_USER = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,63}")
_REMOTE_PATH = re.compile(r"/[A-Za-z0-9._+/-]+")
_ARTIFACT = re.compile(r"probe\.json|step-[0-9]{3,5}-(?:prefill|decode|chunk|refill|reset-prefill)\.f32")


class RemoteUnavailable(ValueError):
    """Every node was safely unavailable before any model process was started."""

# The anchor remains a live process-group leader after the probe exits. This
# prevents the PGID from being recycled before the supervisor verifies teardown.
# Losing the supervisor pipe independently destroys the anchor's own group.
ANCHOR_SOURCE = r'''
import base64,json,os,signal,subprocess,sys,threading,time
from pathlib import Path
c=json.loads(base64.b64decode(sys.argv[1]))
for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP): signal.signal(sig,signal.SIG_IGN)
lost=threading.Event()
def watch():
    sys.stdin.buffer.read(1)
    lost.set()
threading.Thread(target=watch,daemon=True).start()
deadline=time.monotonic()+c['timeout_seconds']+8
log=Path(c['run'])/'process.log'
p=subprocess.Popen(c['command'],cwd=c['run'],env=c['environment'],stdin=subprocess.DEVNULL,
                   stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
truncated=[False]
def drain():
    with log.open('xb') as target:
        kept=0
        while True:
            block=p.stdout.read(65536)
            if not block: break
            take=min(len(block),1048576-kept)
            if take: target.write(block[:take]); kept+=take
            if take<len(block): truncated[0]=True
reader=threading.Thread(target=drain,daemon=True); reader.start()
published=False
while True:
    code=p.poll()
    if code is not None and not published:
        reader.join(timeout=1)
        complete=not reader.is_alive()
        target=Path(c['run'])/'child-exit.json'
        tmp=target.with_suffix('.pending')
        tmp.write_text(json.dumps({'exit_code':code,'log_truncated':truncated[0] or not complete,'log_complete':complete}))
        tmp.replace(target)
        published=True
    if lost.is_set() or time.monotonic()>=deadline:
        os.killpg(os.getpgrp(),signal.SIGKILL)
    time.sleep(.05)
'''

_REMOTE_BODY = r'''
import base64,hashlib,json,math,os,re,shutil,signal,subprocess,sys,threading,time,zipfile
from pathlib import Path
CAP=128*1024*1024
ARTIFACT=re.compile(r'probe\.json|step-[0-9]{3,5}-(?:prefill|decode|chunk|refill|reset-prefill)\.f32')
class PreflightUnavailable(RuntimeError): pass

def stat_identity(path):
    s=path.stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]

def process_identity(pid):
    text=Path('/proc/%d/stat'%pid).read_text()
    fields=text[text.rfind(')')+2:].split()
    return {'pid':pid,'state':fields[0],'pgrp':int(fields[2]),'session':int(fields[3]),'start_ticks':int(fields[19])}

def group_members(pgid):
    members=[]
    for item in Path('/proc').iterdir():
        if not item.name.isdecimal(): continue
        try:
            value=process_identity(int(item.name))
            if value['pgrp']==pgid and value['state'] not in ('Z','X'): members.append(value)
        except (FileNotFoundError,ProcessLookupError): pass
    return members

def hardware_budget(c):
    mem={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1)
        if key in ('MemTotal','MemAvailable'): mem[key]=int(value.split()[0])*1024
    if set(mem)!=set(('MemTotal','MemAvailable')): raise RuntimeError('RAM availability evidence missing')
    cgroup=None
    for limit_path,used_path in (('/sys/fs/cgroup/memory.max','/sys/fs/cgroup/memory.current'),
                                ('/sys/fs/cgroup/memory/memory.limit_in_bytes','/sys/fs/cgroup/memory/memory.usage_in_bytes')):
        if Path(limit_path).is_file() and Path(used_path).is_file():
            text=Path(limit_path).read_text().strip()
            if text.isdecimal() and 0<int(text)<mem['MemTotal']:
                limit,used=int(text),int(Path(used_path).read_text().strip())
                mem['MemTotal']=limit; mem['MemAvailable']=min(mem['MemAvailable'],max(0,limit-used))
                cgroup={'limit_bytes':limit,'used_bytes':used,'source':limit_path}
            break
    reserve=max(4*1024**3,math.ceil(mem['MemTotal']*.2))
    ram=math.ceil(c['resources']['ram_required_gib']*1024**3)
    result={'ram_total_bytes':mem['MemTotal'],'ram_available_bytes':mem['MemAvailable'],
            'ram_reserved_bytes':reserve,'ram_required_bytes':ram,'devices':[],'cgroup':cgroup}
    query=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.total','--format=csv,noheader,nounits'],
                         stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=min(10,c['timeout_seconds']))
    if query.returncode!=0 or len(query.stdout)>65536: raise RuntimeError('nvidia-smi inventory unavailable')
    devices={}
    for line in query.stdout.splitlines():
        parts=[x.strip() for x in line.split(',')]
        if len(parts)!=3 or not all(x.isdecimal() for x in parts): raise RuntimeError('Invalid GPU memory inventory')
        index,free,total=map(int,parts)
        if not 0<total or not 0<=free<=total: raise RuntimeError('Impossible GPU memory inventory')
        if index in devices: raise RuntimeError('Duplicate GPU inventory index')
        devices[index]=(free*1024**2,total*1024**2)
    for index in c['devices']:
        if index not in devices: raise RuntimeError('Selected physical GPU missing: '+str(index))
        free,total=devices[index]
        headroom=max(2*1024**3,math.ceil(total*.1))
        required=math.ceil(c['resources']['per_gpu_required_gib'][str(index)]*1024**3)
        result['devices'].append({'index':index,'free_bytes':free,'total_bytes':total,
                                  'reserved_bytes':headroom,'required_bytes':required,'eligible':free>=required+headroom})
    result['eligible']=mem['MemAvailable']>=ram+reserve and all(x['eligible'] for x in result['devices'])
    return result

def stop_owned(anchor,identity):
    evidence={'attempted':True,'verified':False,'mechanism':'live-session-anchor','identity':identity,'actions':[]}
    try:
        current=process_identity(anchor.pid)
        if current['start_ticks']!=identity['start_ticks'] or current['pgrp']!=anchor.pid or current['session']!=anchor.pid:
            raise RuntimeError('Anchor identity changed; refusing to signal any process')
        # A live anchor prevents reuse; all processes in this new session/group
        # were started by the fixed probe command. TERM then KILL includes it.
        for sig,seconds in ((signal.SIGTERM,2),(signal.SIGKILL,3)):
            try: os.killpg(anchor.pid,sig)
            except ProcessLookupError: pass
            evidence['actions'].append(signal.Signals(sig).name)
            limit=time.monotonic()+seconds
            while time.monotonic()<limit:
                if not group_members(anchor.pid):
                    anchor.wait(timeout=1)
                    evidence['verified']=True
                    evidence['remaining']=[]
                    return evidence
                time.sleep(.05)
        evidence['remaining']=group_members(anchor.pid)
    except (OSError,ValueError,RuntimeError,subprocess.TimeoutExpired) as error:
        # If the anchor's independent watchdog already killed its group, the
        # disappearance of every member is sufficient; never signal a new PID.
        try: evidence['remaining']=group_members(anchor.pid)
        except OSError: evidence['remaining']=None
        evidence['verified']=evidence['remaining']==[]
        evidence['error']=str(error)
    return evidence

def main():
    raw=sys.stdin.buffer.readline(262145)
    if len(raw)>262144 or not raw.endswith(b'\n'): raise RuntimeError('Bounded launch payload missing')
    c=json.loads(raw)
    run=None; owned_run=False; anchor=None; identity=None; stopped=threading.Event()
    receipt={'schema_version':1,'run_id':c['run_id'],'node_index':c['node_index'],
             'status':'failed','stage':'preflight','unavailable':False,'hashes':[],'teardown':{'attempted':False,'verified':True},'artifacts':[]}
    deadline=time.monotonic()+c['timeout_seconds']
    def signal_stop(*_): stopped.set()
    for sig in (signal.SIGHUP,signal.SIGTERM,signal.SIGINT): signal.signal(sig,signal_stop)
    def watch_connection():
        sys.stdin.buffer.read(1)
        stopped.set()
    threading.Thread(target=watch_connection,daemon=True).start()
    def check():
        if stopped.is_set(): raise RuntimeError('SSH input closed or termination signal received')
        if time.monotonic()>=deadline: raise TimeoutError('Remote monotonic deadline exceeded')
    try:
        if sys.platform!='linux' or not Path('/proc/self/stat').is_file():
            raise PreflightUnavailable('Linux /proc process ownership evidence is required')
        # Exact raw bytes, including a trailing newline, match sha256sum's
        # contract. Unique expected pins prevent alias nodes from both loading
        # models onto the same host before the controller can collect receipts.
        path=Path('/etc/machine-id')
        if not path.is_file(): raise RuntimeError('Pinned /etc/machine-id identity unavailable')
        value=path.read_bytes()
        if not 0<len(value)<=256 or not value.strip(): raise RuntimeError('Pinned /etc/machine-id identity invalid')
        machine=hashlib.sha256(value).hexdigest()
        receipt['node_identity']={'platform':sys.platform,'hostname':os.uname().nodename,'machine_id_sha256':machine}
        if machine!=c['node']['expected_machine_id_sha256']:
            raise RuntimeError('Remote machine identity hash mismatch; model launch refused')
        root=Path(c['node']['root'])
        if not root.is_dir(): raise PreflightUnavailable('Explicit remote artifact parent does not exist')
        if shutil.disk_usage(root).free < 512*1024*1024: raise PreflightUnavailable('Need512MiB free for bounded probe evidence')
        run=root/('.tensorsharp-regression-'+c['run_id']+'-node'+str(c['node_index']))
        run.mkdir(mode=0o700)
        owned_run=True
        receipt['remote_run_dir']=str(run)
        try: receipt['hardware_budget']=hardware_budget(c)
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as error: raise PreflightUnavailable(str(error)) from error
        if not receipt['hardware_budget']['eligible']: raise PreflightUnavailable('Declared peak resources exceed remote free memory with headroom')
        receipt['stage']='hash_validation'
        snapshots={}
        runtime_dir=Path(c['node']['probe_path']).parent
        declared={x['path'] for x in c['node']['runtime_files']}
        deployed={str(x) for x in runtime_dir.glob('TensorSharp*.dll')}
        deployed.update(str(x) for pattern in ('*.deps.json','*.runtimeconfig.json','libGgmlOps.so') for x in runtime_dir.glob(pattern))
        if not deployed or not deployed<=declared: raise RuntimeError('Every deployed TensorSharp DLL beside the probe must be hash pinned')
        for role in ('model_files','runtime_files','source_files'):
            for item in c['node'][role]:
                check(); path=Path(item['path'])
                if role in ('model_files','runtime_files') and not path.is_file(): raise PreflightUnavailable('Pinned model/runtime file missing: '+str(path))
                before=stat_identity(path); digest=hashlib.sha256()
                with path.open('rb') as source:
                    while True:
                        block=source.read(1024*1024)
                        if not block: break
                        check(); digest.update(block)
                if before!=stat_identity(path): raise RuntimeError('File changed while hashing: '+str(path))
                actual=digest.hexdigest()
                receipt['hashes'].append({'role':role,'path':str(path),'sha256':actual,'bytes':before[2]})
                if actual!=item['sha256']: raise RuntimeError('Hash mismatch: '+str(path))
                snapshots[str(path)]=before
        check()
        token_bytes=base64.b64decode(c['tokens_base64'],validate=True)
        if len(token_bytes)>65536 or hashlib.sha256(token_bytes).hexdigest()!=c['tokens_sha256'] or json.loads(token_bytes)!=c['tokens']:
            raise RuntimeError('Pinned token fixture bytes differ')
        tokens=run/'tokens.json'; tokens.write_bytes(token_bytes)
        output=run/'probe'
        # The model host receives no inherited loader hooks, tuning, credentials,
        # profiler, extra dependency path or arbitrary startup environment.
        env={'PATH':'/usr/local/bin:/usr/bin:/bin','LANG':'C.UTF-8','TZ':'UTC',
             'HOME':os.environ.get('HOME',str(run)), 'DOTNET_CLI_TELEMETRY_OPTOUT':'1',
             'DOTNET_SKIP_FIRST_TIME_EXPERIENCE':'1'}
        temporary=run/'native-tmp'; temporary.mkdir()
        env.update(TMPDIR=str(temporary),TMP=str(temporary),TEMP=str(temporary))
        env.update(MAX_CONTEXT=str(c['context_tokens']),KV_CACHE_DTYPE='f16',TS_N_CPU_MOE='0',TS_CPU_MOE='0',
                   CUDA_VISIBLE_DEVICES=','.join(map(str,c['devices'])),
                   TENSORSHARP_TP_CONNECT_TIMEOUT_SECONDS=str(max(1,math.ceil(deadline-time.monotonic()))),
                   TENSORSHARP_TP_RECV_TIMEOUT_SECONDS=str(max(1,math.ceil(deadline-time.monotonic()))))
        native=[str(Path(x['path']).parent) for x in c['node']['runtime_files'] if Path(x['path']).name=='libGgmlOps.so']
        if native: env['LD_LIBRARY_PATH']=':'.join(dict.fromkeys(native))
        else: env.pop('LD_LIBRARY_PATH',None)
        command=[c['node']['dotnet'],c['node']['probe_path'],'--model',c['node']['model_path'],
                 '--tokens',str(tokens),'--weights-id',c['weights_id'],'--output',str(output),
                 '--tp' if c['mode']=='tp' else '--layer-split',str(c['local_degree']),
                 '--tp-node-id',str(c['node_index']),'--tp-peers',','.join(c['peer_endpoints']),
                 '--chunk',str(c['chunk']),'--decode-count',str(c['decode_count']),'--backend',c['backend']]
        receipt['command']=command
        check()
        receipt['hardware_budget_before_hash']=receipt['hardware_budget']
        try: receipt['hardware_budget']=hardware_budget(c)
        except (OSError,RuntimeError,subprocess.TimeoutExpired) as error: raise PreflightUnavailable(str(error)) from error
        check()
        if not receipt['hardware_budget']['eligible']: raise PreflightUnavailable('Remote resources became insufficient after hashing; probe not launched')
        receipt['stage']='execution'
        anchor_config={'run':str(run),'command':command,'environment':env,
                       'timeout_seconds':max(1,deadline-time.monotonic())}
        anchor=subprocess.Popen([sys.executable,'-I','-c',ANCHOR_SOURCE,base64.b64encode(json.dumps(anchor_config).encode()).decode()],
                                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,start_new_session=True)
        identity=process_identity(anchor.pid)
        receipt['anchor_identity']=identity
        while True:
            check()
            marker=run/'child-exit.json'
            if marker.is_file():
                child=json.loads(marker.read_text()); receipt['child']=child
                if type(child.get('exit_code')) is not int or child['exit_code']!=0: raise RuntimeError('Probe exited unsuccessfully')
                break
            if anchor.poll() is not None: raise RuntimeError('Owned group anchor exited before probe receipt')
            if output.exists():
                files=list(output.iterdir())
                if len(files)>1024 or sum(x.stat().st_size for x in files if x.is_file())>CAP-2*1024*1024:
                    raise RuntimeError('Probe evidence exceeds128MiB budget')
            time.sleep(.05)
        receipt['status']='passed'
    except BaseException as error:
        receipt['error']=type(error).__name__+': '+str(error)
        receipt['unavailable']=isinstance(error,PreflightUnavailable)
        if receipt['unavailable']: receipt['stage']='preflight'
    finally:
        if anchor is not None:
            if identity is None:
                # Anchor still self-terminates on pipe EOF/deadline; do not claim
                # verified ownership when /proc identity could not be captured.
                receipt['teardown']={'attempted':True,'verified':False,'reason':'anchor identity unavailable'}
            else: receipt['teardown']=stop_owned(anchor,identity)
            if anchor.stdin: anchor.stdin.close()
        if not receipt['teardown']['verified']: receipt['status']='failed'
        if owned_run:
            try:
                for path,before in locals().get('snapshots',{}).items():
                    if stat_identity(Path(path))!=before: raise RuntimeError('Pinned file changed during run: '+path)
                files=[]
                probe=run/'probe'
                if probe.is_dir():
                    for path in sorted(probe.iterdir()):
                        if path.is_symlink() or not path.is_file() or not ARTIFACT.fullmatch(path.name):
                            raise RuntimeError('Unexpected probe artifact: '+path.name)
                        files.append(path)
                log=run/'process.log'
                if log.is_file() and not log.is_symlink(): files.append(log)
                if len(files)>1025 or sum(p.stat().st_size for p in files)>CAP-2*1024*1024:
                    raise RuntimeError('Artifact collection exceeds128MiB budget')
                bundle=run/'evidence.zip'
                with zipfile.ZipFile(bundle,'x',compression=zipfile.ZIP_STORED) as archive:
                    for path in files:
                        name=str(path.relative_to(run)); digest=hashlib.sha256(path.read_bytes()).hexdigest()
                        receipt['artifacts'].append({'path':name,'bytes':path.stat().st_size,'sha256':digest})
                        archive.write(path,name)
                if bundle.stat().st_size>CAP: raise RuntimeError('Bundle exceeds128MiB budget')
                receipt['bundle']={'path':str(bundle),'bytes':bundle.stat().st_size,'sha256':hashlib.sha256(bundle.read_bytes()).hexdigest()}
            except BaseException as error:
                receipt['status']='failed'; receipt['collection_error']=type(error).__name__+': '+str(error)
            target=run/'receipt.json'
            text=json.dumps(receipt,allow_nan=False)
            if len(text.encode())>1024*1024: text=json.dumps({'schema_version':1,'status':'failed','error':'receipt budget exceeded','teardown':receipt['teardown']})
            target.write_text(text)
        try: print(json.dumps(receipt,allow_nan=False),flush=True)
        except (BrokenPipeError,OSError): pass
    return 0 if receipt['status']=='passed' else 1

if __name__=='__main__': raise SystemExit(main())
'''
REMOTE_SOURCE = "ANCHOR_SOURCE=" + repr(ANCHOR_SOURCE) + "\n" + _REMOTE_BODY


def _integer(value, low, high, label):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{label} must be an integer in {low}..{high}")


def _remote_path(value, label):
    if not isinstance(value, str) or not _REMOTE_PATH.fullmatch(value) or ".." in PurePosixPath(value).parts:
        raise ValueError(label + " must be a plain absolute POSIX path without traversal")


def _host_identity(host):
    host = host.strip("[]")
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        return host.lower().rstrip(".")


def _reject_nonfinite(value):
    raise ValueError("nonfinite JSON value is forbidden: " + value)


def _source_suffix(value):
    parts = PurePosixPath(value).parts
    roots = {"TensorSharp.Core", "TensorSharp.Models", "TensorSharp.Distributed", "TensorSharp.Runtime",
             "TensorSharp.Tensors", "TensorSharp.Backends.GGML", "TensorSharp.Backends.Cuda", "eng"}
    for index, part in enumerate(parts):
        if part in roots:
            suffix = PurePosixPath(*parts[index:]).as_posix()
            if part == "eng" and not suffix.startswith("eng/validation/DeepSeek41DistributedProbe/"):
                raise ValueError("only the fixed distributed probe sources are accepted under eng")
            return suffix
    raise ValueError("source snapshot path lacks a recognized TensorSharp project root")


def validate_remote_profile(profile):
    p = profile["parameters"]
    keys = {"mode", "local_degree", "devices", "nodes", "peer_endpoints", "chunk", "decode_count", "timeout_seconds"}
    if set(p) != keys:
        raise ValueError("distributed parameters must contain exactly " + ", ".join(sorted(keys)))
    if p["mode"] not in {"tp", "layer"}:
        raise ValueError("distributed mode must be tp or layer")
    for key, low, high in (("local_degree", 1, 8), ("chunk", 1, 2048), ("decode_count", 1, 128), ("timeout_seconds", 1, 3600)):
        _integer(p[key], low, high, key)
    devices = p["devices"]
    if not isinstance(devices, list) or len(devices) != p["local_degree"] or len(set(devices)) != len(devices):
        raise ValueError("devices must uniquely specify every local rank")
    for device in devices:
        _integer(device, 0, 255, "device index")
    nodes, peers = p["nodes"], p["peer_endpoints"]
    if not isinstance(nodes, list) or not 2 <= len(nodes) <= 8 or not isinstance(peers, list) or len(peers) != len(nodes):
        raise ValueError("distributed requires2..8 explicitly authorized nodes and matching peer endpoints")
    destinations = set()
    machine_pins = set()
    for node in nodes:
        if set(node) != {"host", "port", "user", "authorized", "expected_machine_id_sha256", "root", "model_path", "probe_path", "dotnet", "model_files", "runtime_files", "source_files"}:
            raise ValueError("unexpected or missing distributed node fields")
        if node["authorized"] is not True or not isinstance(node["host"], str) or not _HOST.fullmatch(node["host"]):
            raise ValueError("node needs explicit authorization and a literal host name/address")
        if not isinstance(node["user"], str) or not _USER.fullmatch(node["user"]):
            raise ValueError("invalid explicit SSH user")
        _integer(node["port"], 1, 65535, "SSH port")
        destination = (_host_identity(node["host"]), node["port"], node["user"])
        if destination in destinations:
            raise ValueError("duplicate authorized node")
        destinations.add(destination)
        machine_pin = node["expected_machine_id_sha256"]
        if not isinstance(machine_pin, str) or not _SHA.fullmatch(machine_pin) or machine_pin in machine_pins:
            raise ValueError("each node requires a unique lowercase SHA256 pin of raw /etc/machine-id bytes")
        machine_pins.add(machine_pin)
        for key in ("root", "model_path", "probe_path", "dotnet"):
            _remote_path(node[key], key)
        if PurePosixPath(node["probe_path"]).name != "DeepSeek41DistributedProbe.dll" or not node["model_path"].endswith(".gguf") or PurePosixPath(node["dotnet"]).name != "dotnet":
            raise ValueError("fixed distributed probe and explicit GGUF model required")
        for role in ("model_files", "runtime_files", "source_files"):
            files = node[role]
            if not isinstance(files, list) or not 1 <= len(files) <= 128:
                raise ValueError(role + " must enumerate1..128 pinned files")
            paths = set()
            for item in files:
                if set(item) != {"path", "sha256"} or not isinstance(item["sha256"], str) or not _SHA.fullmatch(item["sha256"]):
                    raise ValueError("each pinned file needs only path and lowercase SHA256")
                _remote_path(item["path"], role)
                if item["path"] in paths:
                    raise ValueError("duplicate pinned path")
                paths.add(item["path"])
            if role == "model_files" and node["model_path"] not in paths:
                raise ValueError("model entry point must be hash pinned")
            if role == "runtime_files" and not {node["probe_path"], node["dotnet"]} <= paths:
                raise ValueError("probe and dotnet host binaries must be hash pinned")
            if role == "source_files":
                suffixes = {_source_suffix(path) for path in paths}
                required = {"eng/validation/DeepSeek41DistributedProbe/Program.cs",
                            "eng/validation/DeepSeek41DistributedProbe/DeepSeek41DistributedProbe.csproj"}
                if not required <= suffixes or any(not any(s.startswith(root + "/") for s in suffixes)
                                                   for root in ("TensorSharp.Core", "TensorSharp.Models", "TensorSharp.Distributed")):
                    raise ValueError("source snapshot needs probe Program/project and core/models/distributed anchors")
        shard = re.fullmatch(r"(.+)-[0-9]{5}-of-([0-9]{5})\.gguf", node["model_path"])
        if shard:
            expected = {f"{shard[1]}-{i:05d}-of-{int(shard[2]):05d}.gguf" for i in range(1, int(shard[2]) + 1)}
            if {x["path"] for x in node["model_files"]} != expected:
                raise ValueError("all and only GGUF sibling shards must be hash pinned")
    for index, endpoint in enumerate(peers):
        if not isinstance(endpoint, str) or not re.fullmatch(r"(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:]+\]):[0-9]{1,5}", endpoint):
            raise ValueError("peer endpoints must be explicit host:port values")
        _integer(int(endpoint.rsplit(":", 1)[1]), 1, 65535, "peer port")
        if _host_identity(endpoint.rsplit(":", 1)[0]) != _host_identity(nodes[index]["host"]):
            raise ValueError("each peer endpoint must use its explicitly authorized node host; separate tunnel routes are not inferred")
    if len(set(peers)) != len(peers):
        raise ValueError("peer endpoints must be distinct")
    weights = profile.get("expected", {}).get("weights_id")
    if not isinstance(weights, str) or not weights.strip() or len(weights) > 256 or any(ord(c) < 32 for c in weights):
        raise ValueError("expected.weights_id must identify the pinned checkpoint")
    if not {"tokens", "known_hosts", "probe"} <= profile.get("fixtures", {}).keys():
        raise ValueError("distributed requires hash-pinned tokens, known_hosts and local probe fixtures")
    sources = [{(_source_suffix(f["path"]), f["sha256"]) for f in node["source_files"]} for node in nodes]
    runtimes = [{(PurePosixPath(f["path"]).name, f["sha256"]) for f in node["runtime_files"]} for node in nodes]
    if any(value != sources[0] for value in sources) or any(value != runtimes[0] for value in runtimes):
        raise ValueError("every remote rank must pin the same source snapshot and runtime binary identities")
    resources = profile.get("resources", {})
    per_gpu = resources.get("per_gpu_required_gib")
    if not isinstance(per_gpu, dict) or set(per_gpu) != {str(device) for device in devices}:
        raise ValueError("resources.per_gpu_required_gib must map exactly the selected physical device indices")
    for label, value in [("ram_required_gib", resources.get("ram_required_gib")),
                         ("vram_required_gib", resources.get("vram_required_gib")), *per_gpu.items()]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(label + " must be a positive finite declared memory budget")


def _pinned_fixture(profile, name):
    item = profile["fixtures"][name]
    path = Path(item["path"]).resolve()
    maximum = 64 * 1024 if name == "tokens" else 1024 * 1024
    if not path.is_file() or path.stat().st_size > maximum:
        raise ValueError(name + " fixture missing or exceeds its byte budget")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != item["sha256"]:
        raise ValueError(name + " fixture hash mismatch")
    return path, content


def _validate_local_identities(profile, context, require_model=True):
    identities = context.get("profile_models", {}).get("primary", {}).get("identity", [])
    model_hashes = sorted(item.get("sha256", "") for item in identities
                          if str(item.get("path", "")).lower().endswith(".gguf"))
    if require_model and (not model_hashes or any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in model_hashes)):
        raise ValueError("full local primary GGUF hashes are required for distributed model identity")
    probe = profile["fixtures"]["probe"]
    probe_path = Path(probe["path"])
    if probe_path.name != "DeepSeek41DistributedProbe.dll" or not probe_path.is_file():
        raise ValueError("local control requires the pinned DeepSeek41DistributedProbe.dll")
    if hashlib.sha256(probe_path.read_bytes()).hexdigest() != probe["sha256"]:
        raise ValueError("local probe fixture hash mismatch")
    for node in profile["parameters"]["nodes"]:
        if require_model and sorted(f["sha256"] for f in node["model_files"]) != model_hashes:
            raise ValueError("remote model shards differ from the resolved local primary model")
        if next(f["sha256"] for f in node["runtime_files"] if f["path"] == node["probe_path"]) != probe["sha256"]:
            raise ValueError("remote probe binary differs from the pinned local control probe")
    for item in profile["parameters"]["nodes"][0]["source_files"]:
        local = Path(context["repo"]) / _source_suffix(item["path"])
        if not local.is_file() or hashlib.sha256(local.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("remote supplied source snapshot differs from local checkout: " + str(local))


def availability(profile, context):
    """Local-only preflight; never contacts any node or claims remote capacity."""
    try:
        validate_remote_profile(profile)
        if context.get("backend", profile.get("backend")) != "ggml_cuda":
            return False, "distributed numerical qualification currently requires ggml_cuda"
        if not shutil.which("ssh") or not shutil.which("scp"):
            return False, "preinstalled SSH and SCP required; no automatic installation"
        _pinned_fixture(profile, "tokens")
        _pinned_fixture(profile, "known_hosts")
        _validate_local_identities(profile, context, require_model=not context.get("preparing_model", False))
        return True, "explicit node list and pinned local inputs verified; remote memory/hash checks precede launch"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, str(error)


def qualify_node_probe(path, profile, index, tokens, token_hash):
    """Require actual mesh/placement logs and the fixed probe control stream."""
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("complete bounded probe.json missing")
    data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    p = profile["parameters"]
    nodes, degree = len(p["nodes"]), p["local_degree"]
    total, offset = nodes * degree, index * degree
    expected_topology = {"mode": "tensor_parallel" if p["mode"] == "tp" else "layer_split",
                         "local_degree": degree, "global_degree": total, "rank_offset": offset, "node_count": nodes}
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        raise ValueError("unsupported probe schema")
    if data.get("status") != ("completed" if index == 0 else "worker_loop_returned") or data.get("disposed") is not True:
        raise ValueError("driver/worker completion and model disposal must be explicit")
    if data.get("topology") != expected_topology or data.get("tokens") != tokens or data.get("tokens_sha256") != token_hash or data.get("weights_id") != profile["expected"]["weights_id"]:
        raise ValueError("probe topology or fixed input/model identity differs")
    _integer(data.get("vocab_size"), 1, 1_048_576, "probe vocabulary")
    if any(token >= data["vocab_size"] for token in tokens):
        raise ValueError("fixed token ID exceeds model vocabulary")
    log = (path.parent.parent / "process.log").read_text(encoding="utf-8", errors="replace")
    joined = f"[TcpCommunicator] Rank {index}/{nodes} connected to all peers"
    backends = set(re.findall(r"(?m)^\s*Backend:\s*(\w+)\s*$", log))
    if joined not in log or backends != {"GgmlCuda"}:
        raise ValueError("actual all-rank connection and GgmlCuda backend evidence missing")
    if __package__:
        from .scenarios import backend_evidence
    else:
        from regression.scenarios import backend_evidence
    if not backend_evidence(log, "ggml_cuda")["verified"]:
        raise ValueError("remote probe switched/fell back to another backend")
    if re.search(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|Cannot allocate memory|std::bad_alloc|OutOfMemoryException", log, re.I):
        raise ValueError("remote probe reported an allocation failure")
    mode = "tensor parallelism" if p["mode"] == "tp" else "layer splitting"
    banner = f"Distributed {mode}: node {index}/{nodes}, {degree} local GPU(s), {total} total"
    if banner not in log:
        raise ValueError("actual distributed group initialization evidence missing")
    placement = []
    if p["mode"] == "tp":
        patterns = [rf"TP weight sharding complete \((?:{total}|{degree}) GPU",
                    rf"routed-MoE tensor parallelism: {degree} local / {total} global ranks, offset {offset}; sharded gate/up/down weights",
                    rf"tensor parallel across {total} global rank\(s\), {degree} local at offset {offset}:"]
        placement = [line for line in log.splitlines() if any(re.search(pattern, line) for pattern in patterns)]
    else:
        devices = {}
        for line in log.splitlines():
            match = re.search(r"\[(?:dsv4|glm)\]\s+device (\d+): layers (\d+)\.\.(\d+) \((\d+)\)", line)
            if match and int(match[3]) - int(match[2]) + 1 == int(match[4]) and int(match[4]) > 0:
                devices[int(match[1])] = int(match[4]); placement.append(line)
        qwen = re.search(rf"Qwen distributed layer split: node {index}/{nodes}, stage \d+, layers \[(\d+),(\d+)\), {degree} local devices\.", log)
        if qwen and int(qwen[2]) > int(qwen[1]):
            local = {int(d): int(n) for d, n in re.findall(r"gpu(\d+)=(\d+) layers/", log)}
            if degree == 1 or all(local.get(d, 0) > 0 for d in range(degree)):
                devices = {d: 1 for d in range(degree)}; placement.append(qwen[0])
        if set(devices) != set(range(degree)):
            placement = []
    if not placement:
        raise ValueError("completed sharding or nonempty layer ownership on every assigned GPU was not reported")
    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("probe operation stream missing")
    if index != 0:
        if steps:
            raise ValueError("worker receipt unexpectedly claims driver numerical forwards")
    else:
        prefix = len(tokens) - p["decode_count"]
        wanted = [("reset", []), ("prefill", tokens[:prefix])]
        wanted += [("decode", [token]) for token in tokens[prefix:]] + [("reset", [])]
        wanted += [("chunk", tokens[first:min(prefix, first + p["chunk"])]) for first in range(0, prefix, p["chunk"])]
        wanted += [("refill", [token]) for token in tokens[prefix:]] + [("reset", []), ("reset-prefill", tokens[:prefix])]
        if len(steps) != len(wanted):
            raise ValueError("driver did not execute every fixed reset/prefill/decode/chunk/refill operation")
        for position, (step, (operation, inputs)) in enumerate(zip(steps, wanted)):
            if step.get("operation") != operation or step.get("tokens") != inputs:
                raise ValueError("driver operation inputs changed")
            if operation == "reset" and step.get("logits") is not None:
                raise ValueError("reset control unexpectedly contains logits")
            if operation != "reset" and (step.get("count") != data["vocab_size"] or step.get("logits") != f"step-{position:03d}-{operation}.f32"):
                raise ValueError("driver logit dimensions/path differ from the fixed operation")
    return {"topology": expected_topology, "joined": joined, "placement_lines": placement,
            "scope": "runtime mesh and completed placement diagnostics; numerical parity is gated separately"}


def ssh_prefix(node, known_hosts, executable="ssh", config=None):
    if config is None:
        raise ValueError("owned empty SSH configuration is required")
    host = node["host"]
    # ssh receives host and port separately; brackets belong only to scp's
    # host:path syntax and would otherwise become part of the SSH hostname.
    destination = node["user"] + "@" + host
    hosts_path = Path(known_hosts).as_posix()
    if any(ord(character) < 32 for character in hosts_path):
        raise ValueError("known-hosts path contains control characters")
    hosts_value = '"' + hosts_path.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return [executable, "-F", str(config), "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
            "-o", "UserKnownHostsFile=" + hosts_value, "-o", "GlobalKnownHostsFile=" + hosts_value,
            "-o", "ControlMaster=no", "-o", "ControlPath=none", "-o", "ProxyCommand=none", "-o", "ProxyJump=none", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2",
            "-p", str(node["port"]), "-T", destination]


def _ssh_worker(spec_path):
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    payload = (json.dumps(spec["payload"], separators=(",", ":")) + "\n").encode()
    if len(payload) > 256 * 1024:
        raise ValueError("remote launch metadata exceeds256KiB")
    command = "python3 -I -c " + shlex.quote(REMOTE_SOURCE)
    # Command construction is fixed; no profile string becomes executable code.
    child = subprocess.Popen(ssh_prefix(spec["payload"]["node"], spec["known_hosts"], spec["ssh"], spec["ssh_config"]) + [command],
                             stdin=subprocess.PIPE)
    try:
        child.stdin.write(payload)
        child.stdin.flush()
        return child.wait()
    finally:
        if child.stdin:
            child.stdin.close()


def _scp_command(node, known_hosts, remote_path, local_path, executable, config):
    prefix = ssh_prefix(node, known_hosts, executable, config)
    # scp uses -P and has no -T SSH terminal option. Remote paths are restricted
    # POSIX paths and generated UUID names; no wildcard/remote shell expansion.
    prefix.pop()
    host = node["host"]
    destination = node["user"] + "@" + ("[" + host + "]" if ":" in host else host)
    prefix.pop()
    prefix[prefix.index("-p")] = "-P"
    return prefix + [destination + ":" + remote_path, str(local_path)]


def _validate_receipt(receipt, run_id, index):
    if (not isinstance(receipt, dict) or type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1
            or receipt.get("run_id") != run_id or type(receipt.get("node_index")) is not int or receipt["node_index"] != index):
        raise ValueError("remote receipt identity does not match owned run")
    return receipt


def _read_receipt(path, run_id, index):
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("remote receipt missing or exceeds1MiB")
    return _validate_receipt(json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite), run_id, index)


def extract_bundle(bundle, receipt, output):
    """Verify all bytes and archive paths before creating any extracted file."""
    expected = receipt.get("bundle", {})
    if type(expected.get("bytes")) is not int or not 0 < expected["bytes"] <= MAX_ARTIFACT_BYTES:
        raise ValueError("invalid remote artifact byte budget")
    if bundle.stat().st_size != expected["bytes"] or hashlib.sha256(bundle.read_bytes()).hexdigest() != expected.get("sha256"):
        raise ValueError("transferred artifact bundle hash/size mismatch")
    declared = receipt.get("artifacts")
    if not isinstance(declared, list) or not 1 <= len(declared) <= 1025:
        raise ValueError("missing bounded artifact manifest")
    manifests = {}
    for item in declared:
        name = item["path"]
        if name != "process.log" and not (name.startswith("probe/") and _ARTIFACT.fullmatch(name[6:])):
            raise ValueError("unexpected artifact path")
        if name in manifests or type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= MAX_ARTIFACT_BYTES or not _SHA.fullmatch(item.get("sha256", "")):
            raise ValueError("invalid/duplicate artifact manifest")
        manifests[name] = item
    with zipfile.ZipFile(bundle) as archive:
        infos = archive.infolist()
        if len(infos) != len(manifests) or {i.filename for i in infos} != set(manifests) or sum(i.file_size for i in infos) > MAX_ARTIFACT_BYTES:
            raise ValueError("archive contents differ from bounded manifest")
        verified = {}
        for info in infos:
            if info.is_dir() or (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("archive links/directories are forbidden")
            content = archive.read(info)
            item = manifests[info.filename]
            if len(content) != item["bytes"] or hashlib.sha256(content).hexdigest() != item["sha256"]:
                raise ValueError("artifact hash/size mismatch")
            verified[info.filename] = content
    output.mkdir(exist_ok=False)
    for name, content in verified.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return output / "probe" / "probe.json"


def run_remote_nodes(profile, context, result):
    """Run and collect authorized nodes; numerical parity remains the caller's gate.

    Raises ValueError on failed execution or unverified cleanup, while retaining
    receipts and process evidence. No new nodes or arbitrary remote commands exist.
    """
    validate_remote_profile(profile)
    p = profile["parameters"]
    _, token_bytes = _pinned_fixture(profile, "tokens")
    _, known_host_bytes = _pinned_fixture(profile, "known_hosts")
    tokens = json.loads(token_bytes)
    context_tokens = context.get("context_tokens", 2048)
    if not isinstance(tokens, list) or not p["decode_count"] + 2 <= len(tokens) <= min(1024, context_tokens) or any(type(t) is not int or not 0 <= t < 2**31 for t in tokens):
        raise ValueError("token fixture must fit context and contain decode_count+2 valid integer IDs")
    backend = context.get("backend", profile.get("backend"))
    if backend != "ggml_cuda":
        raise ValueError("distributed numerical qualification currently requires ggml_cuda")
    _validate_local_identities(profile, context)
    if backend == "ggml_cuda" and any(not any(PurePosixPath(f["path"]).name == "libGgmlOps.so" for f in n["runtime_files"]) for n in p["nodes"]):
        raise ValueError("each GGML node needs a hash-pinned libGgmlOps.so")
    ssh, scp = shutil.which("ssh"), shutil.which("scp")
    if not ssh or not scp:
        raise ValueError("preinstalled SSH/SCP required")
    run_id = uuid.uuid4().hex
    folder = Path(context["artifact_dir"]) / ("distributed-" + run_id)
    folder.mkdir(parents=True, exist_ok=False)
    known_hosts, ssh_config = folder / "known_hosts", folder / "ssh_config"
    known_hosts.write_bytes(known_host_bytes)
    ssh_config.write_text("", encoding="utf-8")
    result.setdefault("processes", [])
    evidence = result.setdefault("evidence", {}).setdefault("remote_nodes", [])
    result["evidence"]["remote_identity_scope"] = "Pinned supplied source snapshot and deployed runtime binaries; not reproducible-build proof."
    result["evidence"]["remote_supervisor_sha256"] = hashlib.sha256(REMOTE_SOURCE.encode()).hexdigest()
    result["evidence"]["remote_anchor_sha256"] = hashlib.sha256(ANCHOR_SOURCE.encode()).hexdigest()
    active, receipts, problem = [], [], None
    deadline = time.monotonic() + p["timeout_seconds"] + 20
    try:
        for index in list(range(len(p["nodes"]) - 1, 0, -1)) + [0]:
            node = p["nodes"][index]
            payload = {**{k: p[k] for k in ("mode", "local_degree", "devices", "peer_endpoints", "chunk", "decode_count", "timeout_seconds")},
                       "node": node, "node_index": index, "run_id": run_id, "tokens": tokens,
                       "tokens_base64": base64.b64encode(token_bytes).decode(), "tokens_sha256": hashlib.sha256(token_bytes).hexdigest(),
                       "weights_id": profile["expected"]["weights_id"], "backend": backend, "context_tokens": context_tokens,
                       "resources": profile["resources"]}
            spec = folder / f"node{index}.launch.json"
            spec.write_text(json.dumps({"payload": payload, "known_hosts": str(known_hosts), "ssh": ssh,
                                        "ssh_config": str(ssh_config)}), encoding="utf-8")
            managed = ManagedProcess([sys.executable, "-I", str(Path(__file__).resolve()), "--ssh-worker", str(spec)],
                                     cwd=context["repo"], artifact_dir=folder, stage=f"node{index}")
            active.append((index, node, managed))
        while True:
            codes = [managed.process.poll() for _, _, managed in active]
            if any(code is not None and code != 0 for code in codes):
                problem = "a node SSH supervisor failed; remaining connections cancelled"
                break
            if all(code == 0 for code in codes):
                for _, _, managed in active:
                    item = managed.wait(p["timeout_seconds"] + 20)
                    result["processes"].append(item)
                    if containment_failure_evidence(item):
                        problem = "local SSH process containment was not verified"
                break
            if time.monotonic() >= deadline:
                for _, _, managed in active:
                    managed.timed_out = True
                problem = "distributed supervisory deadline exceeded"
                break
            time.sleep(.1)
    except BaseException as error:
        problem = error
    finally:
        # Closing every local SSH pipe activates the independent remote EOF
        # watchdog even if an earlier node failed or the user cancelled.
        for index, node, managed in active:
            managed.stop()
            snapshot = managed.snapshot()
            if not any(x.get("stage") == snapshot["stage"] for x in result["processes"]):
                result["processes"].append(snapshot)
        for index, node, managed in active:
            remote = node["root"].rstrip("/") + "/.tensorsharp-regression-" + run_id + "-node" + str(index)
            record = {"node_index": index, "remote_run_dir": remote, "teardown": {"attempted": True, "verified": False}}
            evidence.append(record)
            try:
                receipt = None
                stdout = managed.snapshot().get("stdout", "")
                if stdout and len(stdout.encode()) <= 1024 * 1024:
                    try:
                        candidate = _validate_receipt(json.loads(stdout, parse_constant=_reject_nonfinite), run_id, index)
                        if (candidate.get("stage") == "preflight" and candidate.get("status") == "failed"
                                and candidate.get("teardown") == {"attempted": False, "verified": True}
                                and "anchor_identity" not in candidate and "child" not in candidate):
                            receipt = candidate
                    except (ValueError, TypeError):
                        pass
                # Fresh connection collects the durable receipt if the original
                # SSH stdout was lost. No remote mutation occurs on this path.
                if receipt is None:
                    path = folder / f"node{index}.receipt.json"
                    for attempt in range(3):
                        fetched = run_process(_scp_command(node, known_hosts, remote + "/receipt.json", path, scp, ssh_config),
                                              cwd=context["repo"], artifact_dir=folder, stage=f"node{index}-receipt-{attempt}", timeout_seconds=10)
                        result["processes"].append(fetched)
                        if fetched["exit_code"] == 0:
                            receipt = _read_receipt(path, run_id, index)
                            break
                        if attempt < 2:
                            time.sleep(1)
                    if receipt is None:
                        raise ValueError("remote teardown receipt could not be collected")
                record.update(receipt)
                if receipt.get("teardown", {}).get("verified") is not True:
                    raise ValueError("remote process-group teardown not verified")
                if receipt.get("stage") == "preflight" and receipt.get("unavailable") is True and receipt.get("teardown", {}).get("attempted") is False:
                    continue
                pinned = {(role, f["path"], f["sha256"]) for role in ("model_files", "runtime_files", "source_files") for f in node[role]}
                actual = {(f["role"], f["path"], f["sha256"]) for f in receipt.get("hashes", [])}
                if (receipt.get("status") != "passed" or actual != pinned or receipt.get("child", {}).get("exit_code") != 0
                        or receipt.get("child", {}).get("log_truncated") is not False
                        or receipt.get("child", {}).get("log_complete") is not True
                        or receipt.get("teardown", {}).get("attempted") is not True
                        or not receipt.get("anchor_identity") or receipt.get("hardware_budget", {}).get("eligible") is not True):
                    raise ValueError("remote probe/hash validation failed: " + str(receipt.get("error", "invalid receipt")))
                if receipt.get("node_identity", {}).get("machine_id_sha256") != node["expected_machine_id_sha256"]:
                    raise ValueError("collected node identity differs from its authorized machine pin")
                bundle_info = receipt.get("bundle", {})
                if bundle_info.get("path") != remote + "/evidence.zip" or type(bundle_info.get("bytes")) is not int or not 0 < bundle_info["bytes"] <= MAX_ARTIFACT_BYTES:
                    raise ValueError("remote bundle path or budget differs from owned run")
                artifacts = receipt.get("artifacts", [])
                if (not isinstance(artifacts, list) or not 1 <= len(artifacts) <= 1025
                        or any(type(item.get("bytes")) is not int or not 0 <= item["bytes"] <= MAX_ARTIFACT_BYTES for item in artifacts)):
                    raise ValueError("remote extracted artifact sizes are invalid")
                extracted_bytes = sum(item["bytes"] for item in artifacts)
                if extracted_bytes > MAX_ARTIFACT_BYTES:
                    raise ValueError("remote extracted artifacts exceed128MiB")
                headroom = context.get("disk_headroom_gib", 5)
                if isinstance(headroom, bool) or not isinstance(headroom, (int, float)) or not math.isfinite(headroom) or headroom < 0:
                    raise ValueError("disk headroom must be finite and nonnegative")
                required = bundle_info["bytes"] + extracted_bytes + math.ceil(headroom * 1024**3)
                free = shutil.disk_usage(folder).free
                record["local_artifact_budget"] = {"archive_bytes": bundle_info["bytes"], "extracted_bytes": extracted_bytes,
                                                   "headroom_bytes": math.ceil(headroom * 1024**3), "free_bytes": free}
                if free < required:
                    raise ValueError("insufficient local disk for remote bundle plus extracted artifacts and headroom")
                bundle = folder / f"node{index}.zip"
                copied = run_process(_scp_command(node, known_hosts, bundle_info["path"], bundle, scp, ssh_config),
                                     cwd=context["repo"], artifact_dir=folder, stage=f"node{index}-artifacts", timeout_seconds=60)
                result["processes"].append(copied)
                if copied["exit_code"] != 0:
                    raise ValueError("remote artifact transfer failed")
                probe_path = extract_bundle(bundle, receipt, folder / f"node{index}")
                record["local_probe_path"] = str(probe_path)
                record["qualification"] = qualify_node_probe(probe_path, profile, index, tokens, hashlib.sha256(token_bytes).hexdigest())
                receipts.append(record)
            except BaseException as error:
                record["collection_error"] = type(error).__name__ + ": " + str(error)
                problem = problem or error
                if record.get("teardown", {}).get("verified") is not True:
                    result["containment_failure"] = True
    if isinstance(problem, KeyboardInterrupt):
        raise problem
    if (len(evidence) == len(p["nodes"]) and not result.get("containment_failure")
            and all(node.get("unavailable") is True and node.get("stage") == "preflight"
                    and node.get("teardown") == {"attempted": False, "verified": True} for node in evidence)):
        raise RemoteUnavailable("All nodes were unavailable before model launch: " + "; ".join(node.get("error", "unavailable") for node in evidence))
    if problem is not None or len(receipts) != len(p["nodes"]):
        raise ValueError("Distributed execution did not qualify: " + str(problem))
    machines = [node.get("node_identity", {}).get("machine_id_sha256") for node in receipts]
    if any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in machines) or len(set(machines)) != len(machines):
        raise ValueError("distinct effective machine identities are required; aliases/co-located ranks do not qualify multi-host coverage")
    result.setdefault("assertions", []).append({"name": "all authorized nodes hashed inputs and verified group teardown", "passed": True})
    return sorted(receipts, key=lambda value: value["node_index"])


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--ssh-worker":
        raise SystemExit("Only the internal --ssh-worker SPEC invocation is supported")
    raise SystemExit(_ssh_worker(sys.argv[2]))
