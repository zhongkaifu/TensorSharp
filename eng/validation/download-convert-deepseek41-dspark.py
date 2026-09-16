#!/usr/bin/env python3
"""Run a separately pinned, exclusive VM download/conversion plan. No inference."""
import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.request

GIB = 1024 ** 3
ROOT = Path('/workspace/models/deepseek41-dspark')
FORMATS = {0:(1,4), 8:(32,34), 30:(1,2), 39:(32,17)}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    temporary.replace(path)


def verify_package(package, expected):
    raw = (package/'manifest.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected: raise ValueError('Package manifest SHA256 mismatch')
    manifest = json.loads(raw)
    actual = {str(p.relative_to(package)).replace('\\','/') for p in package.rglob('*') if p.is_file()}
    if actual != set(manifest['files']) | {'manifest.json'}: raise ValueError('Package file set changed')
    for name, value in manifest['files'].items():
        if digest(package/name) != value['sha256']: raise ValueError('Package source drift: '+name)
    return manifest


def validate_range(status, headers, begin, end, total):
    if status != 206: raise ValueError('Range request was not HTTP206; body not read')
    if headers.get('Content-Range') != f'bytes {begin}-{end}/{total}':
        raise ValueError('Incorrect Content-Range; body not read')
    if headers.get('Content-Length') != str(end-begin+1):
        raise ValueError('Incorrect ranged Content-Length; body not read')
    if headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('Encoded payload cannot be resumed safely')


def verify_source(path, shard):
    if path.stat().st_size != shard['file_bytes'] or digest(path) != shard['publisher_sha256']:
        raise ValueError('Whole-file publisher size/SHA256 mismatch: '+path.name)
    with path.open('rb') as source: prefix = source.read(shard['header_prefix_bytes'])
    if hashlib.sha256(prefix).hexdigest() != shard['header_prefix_sha256']:
        raise ValueError('Original safetensors header prefix mismatch: '+path.name)


def download(shard, directory, events, chunk_size=64*1024*1024, opener=urllib.request.urlopen):
    final = directory/shard['file']
    if final.exists():
        verify_source(final, shard)
        return {'file':final.name, 'status':'existing-whole-file-verified'}
    part = final.with_suffix(final.suffix+'.part')
    identity = final.with_suffix(final.suffix+'.resume.json')
    binding = {k:shard[k] for k in ['file','file_bytes','publisher_sha256','origin_url']}
    if part.exists():
        if not identity.is_file() or json.loads(identity.read_text()) != binding:
            raise ValueError('Existing partial has no matching immutable resume identity')
    else:
        if identity.exists() and json.loads(identity.read_text()) != binding:
            raise ValueError('Conflicting resume identity')
        save(identity,binding)
        part.touch(exist_ok=False)
    if part.stat().st_size > shard['file_bytes']: raise ValueError('Partial exceeds source size')
    with part.open('ab') as target:
        while target.tell() < shard['file_bytes']:
            begin = target.tell()
            end = min(begin+chunk_size, shard['file_bytes'])-1
            request = urllib.request.Request(shard['origin_url'], headers={
                'Range':f'bytes={begin}-{end}', 'Accept-Encoding':'identity',
                'User-Agent':'TensorSharp-pinned-DSpark-validation'})
            with opener(request, timeout=120) as response:
                validate_range(response.status,response.headers,begin,end,shard['file_bytes'])
                remaining = end-begin+1
                while remaining:
                    block = response.read(min(1024*1024,remaining))
                    if not block: raise IOError('Short ranged response; verified partial retained')
                    target.write(block); remaining -= len(block)
                if response.read(1): raise ValueError('Ranged body exceeds declared length')
            target.flush(); os.fsync(target.fileno())
            events({'event':'download-range', 'file':final.name, 'begin':begin, 'end':end})
    verify_source(part,shard)
    # Publish without replacing an unrelated or concurrently created final file.
    os.link(part,final)
    part.unlink()
    return {'file':final.name, 'status':'downloaded-whole-file-verified', 'sha256':digest(final)}


def process_identity(folder):
    """Identity readable without ptrace; bind later maps reads to this process."""
    stat = (folder/'stat').read_text()
    tail = stat[stat.rfind(')')+2:].split()
    if len(tail) < 20: raise RuntimeError('Malformed process stat: '+str(folder))
    status = {line.split(':',1)[0]:line.split(':',1)[1].strip()
              for line in (folder/'status').read_text().splitlines() if ':' in line}
    uids = [int(v) for v in status['Uid'].split()]
    gids = [int(v) for v in status['Gid'].split()]
    if len(uids)!=4 or len(gids)!=4: raise RuntimeError('Malformed process credentials')
    command = (folder/'cmdline').read_bytes()
    return {'pid':int(folder.name),'start_ticks':int(tail[19]),'uids':uids,'gids':gids,
            'command_sha256':hashlib.sha256(command).hexdigest(),
            'command':command.replace(b'\0',b' ').decode(errors='replace')}


def maps_for_identity(folder, expected):
    if process_identity(folder)!=expected: raise RuntimeError('Process identity changed before maps read')
    maps = (folder/'maps').read_text(errors='replace')
    if process_identity(folder)!=expected: raise RuntimeError('Process identity changed after maps read')
    return maps


_MAP_READER = """import hashlib,json,pathlib,sys
path=pathlib.Path(sys.argv[1]); raw=path.read_bytes()
if hashlib.sha256(raw).hexdigest()!=sys.argv[2]: raise RuntimeError('Map-reader source changed')
scope={'__name__':'dspark_map_reader','__file__':str(path)}
exec(compile(raw,str(path),'exec'),scope)
print(json.dumps({'maps':scope['maps_for_identity'](pathlib.Path(sys.argv[3]),json.loads(sys.argv[4]))}))
"""


def read_process_maps(folder):
    identity = process_identity(folder)
    method = 'direct'
    try:
        maps = maps_for_identity(folder,identity)
    except PermissionError:
        # Container root may lack CAP_SYS_PTRACE. A reader with the process's
        # observed credentials can still satisfy procfs's ordinary same-user
        # check. Drop supplementary groups too; never change the parent's IDs.
        if os.geteuid()!=0: raise
        source = Path(__file__).resolve()
        child = subprocess.run([sys.executable,'-I','-B','-c',_MAP_READER,
            str(source),digest(source),str(folder),json.dumps(identity)],
            user=identity['uids'][1],group=identity['gids'][1],extra_groups=[],
            capture_output=True,text=True,timeout=15,check=False)
        if process_identity(folder)!=identity: raise RuntimeError('Process identity changed during credential reader')
        if child.returncode:
            raise PermissionError('Credential-scoped maps reader failed: '+child.stderr[-2000:])
        maps = json.loads(child.stdout)['maps']
        if not isinstance(maps,str): raise RuntimeError('Invalid maps-reader response')
        method = 'same-effective-uid-gid-subprocess'
    if process_identity(folder)!=identity: raise RuntimeError('Process identity changed after maps observation')
    return identity,maps,method


def cgroup_memory_available(host_available, proc=Path('/proc'), fs_root=Path('/')):
    """Use controller paths/mount roots, including v1 and ancestor limits."""
    memberships=[]
    for line in (proc/'self/cgroup').read_text().splitlines():
        hierarchy,controllers,path=line.split(':',2)
        if 'memory' in controllers.split(','): memberships.append(('v1',path))
        elif hierarchy=='0' and not controllers: memberships.append(('v2',path))
    # Hybrid systems also list a unified hierarchy for other controllers even
    # when memory remains explicitly assigned to v1. That 0:: entry is not an
    # additional memory controller and may have no visible cgroup2 mount.
    if any(version=='v1' for version,_ in memberships):
        memberships=[entry for entry in memberships if entry[0]=='v1']
    mounts=[]
    for line in (proc/'self/mountinfo').read_text().splitlines():
        left,right=line.split(' - ',1);fields=left.split();kind,_,options=right.split()[:3]
        def unescape(value): return re.sub(r'\\([0-7]{3})',lambda m:chr(int(m.group(1),8)),value)
        if kind=='cgroup2': mounts.append(('v2',unescape(fields[3]),unescape(fields[4])))
        elif kind=='cgroup' and 'memory' in options.split(','):
            mounts.append(('v1',unescape(fields[3]),unescape(fields[4])))
    observations=[];available=host_available
    for version,membership in memberships:
        candidates=[m for m in mounts if m[0]==version]
        if not candidates: raise RuntimeError('Memory controller mount is not visible: '+version)
        resolved=False
        for _,mount_root,mount_point in candidates:
            member=PurePosixPath(membership);anchor=PurePosixPath(mount_root);point=PurePosixPath(mount_point)
            if not member.is_absolute() or not anchor.is_absolute() or not point.is_absolute() or any('..' in p.parts for p in (member,anchor,point)):
                raise RuntimeError('Unsafe cgroup path')
            if member==PurePosixPath('/'): relative=PurePosixPath('.') # namespace root
            else:
                try: relative=member.relative_to(anchor)
                except ValueError: continue
            mapped_root=fs_root.joinpath(*point.parts[1:]);folder=mapped_root.joinpath(*relative.parts)
            resolved=True
            while True:
                limit_name='memory.limit_in_bytes' if version=='v1' else 'memory.max'
                current_name='memory.usage_in_bytes' if version=='v1' else 'memory.current'
                limit_path=folder/limit_name
                if limit_path.exists():
                    raw=limit_path.read_text().strip()
                    maximum=None if raw=='max' else int(raw)
                    current=int((folder/current_name).read_text())
                    stats={line.split()[0]:int(line.split()[1]) for line in (folder/'memory.stat').read_text().splitlines()}
                    inactive=stats.get('total_inactive_file',stats.get('inactive_file',0)) if version=='v1' else stats.get('inactive_file',0)
                    if current<0 or inactive<0 or (maximum is not None and maximum<0): raise RuntimeError('Negative memory controller value')
                    # Linux v1 uses a page-rounded huge integer for no limit.
                    finite=maximum is not None and maximum < (1<<60)
                    effective=max(0,maximum-current+min(inactive,current)) if finite else None
                    if effective is not None: available=min(available,effective)
                    observations.append({'version':version,'path':str(folder),'maximum':maximum,'current':current,
                        'inactive_file':inactive,'conservative_available':effective})
                elif folder!=mapped_root:
                    raise RuntimeError('Memory controller files missing at mapped cgroup: '+str(folder))
                if folder==mapped_root: break
                folder=folder.parent
        if not resolved: raise RuntimeError('Cannot bind memory membership to visible mount root')
    if memberships and not observations: raise RuntimeError('No memory limits could be inspected')
    return available,observations


def quiet_resources(required_disk, required_memory=24*GIB):
    if sys.platform != 'linux': raise RuntimeError('This execution plan is VM/Linux only')
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],
                         capture_output=True,text=True,check=True)
    if gpu.stdout.strip(): raise RuntimeError('An active GPU process prevents this exclusive preparation')
    offenders, unreadable, credential_readers = [], [], []
    for folder in Path('/proc').iterdir():
        if not folder.name.isdigit() or int(folder.name)==os.getpid(): continue
        try:
            identity,maps,method = read_process_maps(folder)
            command = identity['command']
            if method!='direct': credential_readers.append({'identity':identity,'method':method,
                'maps_sha256':hashlib.sha256(maps.encode()).hexdigest()})
        except FileNotFoundError: continue
        except PermissionError:
            unreadable.append(folder.name); continue
        if any(x in command for x in ['TensorSharp.Server.Host.dll','AgentTurnBench.dll',
                'capture-deepseek-teacher-logits.py','generate-deepseek-teacher-continuations.py']):
            offenders.append({'pid':int(folder.name),'reason':'model process command'})
        elif 'GgmlOps' in maps or any('/workspace/models/' in line and
                any(x in line for x in ['.gguf','.safetensors','.bin']) for line in maps.splitlines()):
            offenders.append({'pid':int(folder.name),'reason':'native/model mappings'})
    if offenders or unreadable: raise RuntimeError(f'Model/process quiescence not established: {offenders}; unreadable={unreadable}')
    mem = {line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()}
    available,cgroup_values = cgroup_memory_available(mem['MemAvailable'])
    parent = ROOT if ROOT.exists() else ROOT.parent
    free = shutil.disk_usage(parent).free
    observation = {'at_unix':time.time(),'disk_free':free,'disk_required':required_disk,
                   'memory_available':available,'memory_required':required_memory,'cgroup':cgroup_values,
                   'gpu_compute_pids':[], 'active_model_processes':[], 'credential_scoped_map_readers':credential_readers}
    if free < required_disk or available < required_memory: raise RuntimeError('Insufficient resources: '+json.dumps(observation))
    return observation


def gguf_directory(path):
    with path.open('rb') as source:
        def unpack(fmt):
            count = struct.calcsize('<'+fmt); data = source.read(count)
            if len(data)!=count: raise ValueError('Truncated GGUF directory')
            return struct.unpack('<'+fmt,data)
        def text():
            count, = unpack('Q')
            if count>1024*1024: raise ValueError('Oversized GGUF string')
            data=source.read(count)
            if len(data)!=count: raise ValueError('Truncated GGUF string')
            return data.decode('utf-8')
        magic,version,nt,nk=unpack('IIQQ')
        if magic!=0x46554747 or version!=3 or nt>512 or nk>128: raise ValueError('Unexpected GGUF header')
        metadata={}
        for _ in range(nk):
            key=text(); kind,=unpack('I')
            if key in metadata: raise ValueError('Duplicate GGUF metadata')
            if kind==8: value=text()
            elif kind==4: value,=unpack('I')
            elif kind==9:
                item,count=unpack('IQ')
                if item!=5 or count>64: raise ValueError('Unexpected metadata array')
                value=list(unpack('i'*count))
            else: raise ValueError('Unexpected GGUF metadata type')
            metadata[key]=value
        tensors={}
        for _ in range(nt):
            name=text(); ndim,=unpack('I')
            if name in tensors or not 1<=ndim<=4: raise ValueError('Duplicate tensor or invalid rank')
            shape=list(unpack('Q'*ndim)); kind,offset=unpack('IQ')
            if kind not in FORMATS: raise ValueError('Unexpected tensor encoding')
            block,size=FORMATS[kind]
            if not all(shape) or shape[0]%block: raise ValueError('Invalid tensor block geometry')
            tensors[name]={'shape':shape,'type':kind,'offset':offset,'bytes':math.prod(shape)//block*size}
        start=(source.tell()+31)//32*32
    return metadata,tensors,start


def audit_output(path, expected):
    metadata,tensors,start=gguf_directory(path)
    if metadata!=expected['metadata']: raise ValueError('GGUF metadata differs from official-header plan')
    if set(tensors)!=set(expected['tensors']): raise ValueError('GGUF tensor set differs from official-header plan')
    end=0
    for name,actual in sorted(tensors.items(),key=lambda item:item[1]['offset']):
        wanted=expected['tensors'][name]
        if any(actual[key]!=wanted[key] for key in ['shape','type','bytes']): raise ValueError('GGUF tensor mismatch: '+name)
        if actual['offset']!=end: raise ValueError('GGUF payload gap/overlap/alignment mismatch: '+name)
        end=(end+actual['bytes']+31)//32*32
    if path.stat().st_size!=start+end: raise ValueError('GGUF file size/payload coverage mismatch')
    return {'status':'passed-directory-audit','sha256':digest(path),'bytes':path.stat().st_size,
            'metadata':metadata,'tensors':tensors,'data_start':start,'release_qualified':False,
            'scope':'Header-derived names/shapes/types/offsets and complete-file identity; no inference/numerical oracle.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--mode',choices=['download','convert','all'],required=True)
    parser.add_argument('--run-id',required=True)
    args=parser.parse_args()
    if not re.fullmatch('[a-zA-Z0-9_-]{1,64}',args.run_id): parser.error('Invalid run-id')
    package=Path(__file__).resolve().parent
    manifest=verify_package(package,args.manifest_sha256)
    plan=json.loads((package/'plan.json').read_text())
    checkpoint=ROOT/'DeepSeek-V4.1-Flash'
    sources=sum(row['file_bytes'] for row in plan['shards'])
    # Full source+output need is deliberately conservative even for a resume.
    resource=quiet_resources(sources+plan['estimated_output_bytes']+4*GIB)
    ROOT.mkdir(parents=True,exist_ok=True)
    import fcntl
    with (ROOT/'.prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=ROOT/('preparation-'+args.run_id); out.mkdir(exist_ok=False)
        report={'status':'running','release_qualified':False,'manifest_sha256':args.manifest_sha256,
                'mode':args.mode,'resources_before':resource,'started_at_unix':time.time(),'source_files':[],
                'python_executable':sys.executable,'python_version':sys.version,
                'numpy_version':importlib.metadata.version('numpy')}
        def event(data):
            with (out/'events.jsonl').open('a',encoding='utf-8') as log:
                log.write(json.dumps({'at_unix':time.time(),**data})+'\n')
        def check_metadata():
            for name in ['config.json','model.safetensors.index.json']:
                if digest(checkpoint/name)!=digest(package/'metadata'/name):
                    raise ValueError('Original checkpoint metadata changed: '+name)
        try:
            checkpoint.mkdir(exist_ok=True)
            for name in ['config.json','model.safetensors.index.json']:
                source=package/'metadata'/name; target=checkpoint/name
                if target.exists() and digest(target)!=digest(source): raise ValueError('Existing original metadata differs')
                if not target.exists(): shutil.copy2(source,target)
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',
                     DS_CONVERTER=str(package/'dsv4-dspark-to-gguf.py'))
            for label,script,count in [('preparation-tests','test_deepseek41_dspark_preparation.py',24),
                                       ('converter-tests','test_dspark_converter.py',16)]:
                with (out/(label+'.log')).open('w') as log:
                    test=subprocess.run([sys.executable,'-B',str(package/script)],env=env,
                                        stdout=log,stderr=subprocess.STDOUT,timeout=300)
                text=(out/(label+'.log')).read_text()
                report[label]={'exit_code':test.returncode,'expected_count':count}
                if test.returncode or f'Ran {count} tests' not in text:
                    raise RuntimeError(f'Pinned {label} failed')
            for shard in plan['shards']:
                event({'event':'exclusive-resource-gate','observation':quiet_resources(plan['estimated_output_bytes']+4*GIB)})
                if args.mode in ['download','all']:
                    report['source_files'].append(download(shard,checkpoint,event))
                else:
                    verify_source(checkpoint/shard['file'],shard)
                    report['source_files'].append({'file':shard['file'],'status':'existing-whole-file-verified'})
                save(out/'run.json',report)
            if args.mode in ['convert','all']:
                report['resources_before_conversion']=quiet_resources(plan['estimated_output_bytes']+4*GIB)
                verify_package(package,args.manifest_sha256)
                check_metadata()
                final=ROOT/'DeepSeek-V4.1-Flash-DSpark-MXFP4.gguf'
                partial=ROOT/(args.run_id+'.partial.gguf')
                if final.exists() or partial.exists(): raise FileExistsError('Conversion output exists; it will not be overwritten')
                argv=[sys.executable,'-B',str(package/'dsv4-dspark-to-gguf.py'),'--checkpoint',str(checkpoint),
                      '--out',str(partial),'--expert-type','mxfp4']
                report['conversion_argv']=argv
                with (out/'conversion.log').open('w') as log:
                    result=subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=4*3600)
                report['conversion_exit_code']=result.returncode
                import resource
                report['children_max_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                if result.returncode: raise RuntimeError('Conversion failed; partial output retained')
                expected=json.loads((package/'expected-gguf.json').read_text())
                audit=audit_output(partial,expected);save(out/'output-audit.json',audit)
                # The original source remains pinned after conversion as well.
                for shard in plan['shards']: verify_source(checkpoint/shard['file'],shard)
                os.link(partial,final);partial.unlink()
                report['output']={'path':str(final),'sha256':audit['sha256'],'bytes':audit['bytes']}
            report['resources_after']=quiet_resources(0,required_memory=0)
            check_metadata()
            verify_package(package,args.manifest_sha256)
            report['status']='passed-preparation-only'
        except Exception as error:
            report['status']='failed';report['error']=repr(error)
        finally:
            report['finished_at_unix']=time.time();save(out/'run.json',report)
        return 0 if report['status']=='passed-preparation-only' else 1


if __name__=='__main__': raise SystemExit(main())
