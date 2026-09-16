#!/usr/bin/env python3
"""Build pinned owned native sources in a fresh VM tree; never publish a library.

The reference tree supplies read-only, hash-verified upstream/fixture inputs.
Every gate outcome is retained, including failures and hardware skips. Passing
native gates does not qualify a full model or replace earlier failed evidence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import signal
import shutil
import subprocess
import tarfile
import time
import xml.etree.ElementTree as ET


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            sha.update(chunk)
    return sha.hexdigest()


def safe_work_root(work, reference):
    work, reference = Path(work).resolve(), Path(reference).resolve()
    if work == reference or reference in work.parents or work in reference.parents:
        raise ValueError('Fresh work tree must not overlap the frozen reference tree')
    if work.parent != Path('/workspace').resolve() or not work.name.startswith('tensorsharp-v6-native-'):
        raise ValueError('Use a fresh /workspace/tensorsharp-v6-native-* directory')
    if work.exists():
        raise ValueError('Work tree already exists; prior evidence must remain intact')
    return work


def validate_members(members, expected):
    names = []
    for member in members:
        name = PurePosixPath(member.name)
        if (not member.isfile() or not name.parts or name.is_absolute() or '..' in name.parts or
                name.parts[0] not in ('TensorSharp.GGML.Native', 'eng', 'fixtures', 'inputs')):
            raise ValueError('Archive contains an unsafe or unowned member: ' + member.name)
        names.append(member.name)
    if len(set(names)) != len(names) or set(names) != set(expected):
        raise ValueError('Archive members do not exactly match the pinned manifest')


def result_status(steps, audits):
    if any(not row.get('passed', False) for row in audits):
        return 'failed'
    if any(row.get('status') == 'failed' for row in steps):
        return 'failed'
    if any(row.get('status') == 'unavailable' for row in steps):
        return 'incomplete_native_gates'
    return 'native_gates_passed_full_model_not_run'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path, default=Path('/workspace/tensorsharp-no-patch-20260915'))
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=2)
    parser.add_argument('--python', default='/usr/local/bin/python')
    args = parser.parse_args()
    work = safe_work_root(args.work_root, args.reference_root)
    assert digest(args.manifest) == args.manifest_sha256, 'Manifest hash mismatch'
    inputs = json.loads(args.manifest.read_text())
    assert digest(args.source) == inputs['archive_sha256'], 'Archive hash mismatch'
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'GPU lane occupied'
    reference = args.reference_root.resolve()
    frozen_native = reference/'repo/TensorSharp.GGML.Native/build/libGgmlOps.so'
    assert digest(frozen_native) == inputs['reference_native_sha256'], 'Reference native pin changed'
    repo, out = work/'repo', work/'results'
    out.mkdir(parents=True)
    report = {'status':'running', 'started_at_unix':time.time(), 'steps':[], 'audits':[],
              'source_archive_sha256':inputs['archive_sha256'], 'manifest_sha256':args.manifest_sha256,
              'full_model_qualification':'not_run', 'release_qualified':False,
              'reference_native_sha256':inputs['reference_native_sha256'],
              'prior_tp_gate':'v5 Q2_K/Q3_K and Q4_K/Q6_K checkpoint-shaped pairwise failures remain recorded; no bound changes'}
    save = lambda: (out/'build.json').write_text(json.dumps(report,indent=2)+'\n')
    save()

    def run(name, command, extra=None, timeout=3600):
        row={'name':name,'command':command,'environment_overrides':extra or {},'started_at_unix':time.time(),'status':'running'}
        report['steps'].append(row);save()
        env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',
                 TS_DSV4_THREADS='2',TS_DSV41_ENGRAM_THREADS='2',PYTHONPATH=str(reference/'fixture-python-deps'))
        env.pop('GGML_CUDA_CUBLAS_COMPUTE_TYPE',None)
        env.pop('NVIDIA_TF32_OVERRIDE',None)
        env.update(extra or {})
        with (out/(name+'.log')).open('x') as stream:
            process=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                code=124
        row.update(exit_code=code,wall_seconds=time.time()-row['started_at_unix'],
                   status='passed' if code==0 else 'unavailable' if code==77 else 'failed')
        save(); print(name, row['status'], code, flush=True)
        return code

    def audit(name, action):
        try:
            detail=action();report['audits'].append({'name':name,'passed':True,'detail':detail})
        except Exception as error:
            report['audits'].append({'name':name,'passed':False,'error':repr(error)})
        save()

    def verify_files(root, expected):
        for name, item in expected.items():
            assert digest(root/name)==item['sha256'], 'Input changed: '+str(root/name)
        return {'verified_files':len(expected)}

    def verify_upstream(root, expected):
        variants=[]
        for name,item in expected.items():
            path=root/name
            data=os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            actual=hashlib.sha256(data).hexdigest()
            if actual!=item['sha256']:
                assert hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()==item['lf_sha256'], 'Upstream changed: '+name
                variants.append(name)
        extra=[p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()
               and '.git' not in p.relative_to(root).parts and p.relative_to(root).as_posix() not in expected]
        assert not extra, 'Unexpected upstream files: '+str(extra)
        return {'verified_files':len(expected),'line_ending_variants':variants}

    native=repo/'TensorSharp.GGML.Native';build=native/'build'
    try:
        with tarfile.open(args.source) as archive:
            members=archive.getmembers();validate_members(members,inputs['files'])
            for member in members:
                target=repo/member.name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as dest:
                    shutil.copyfileobj(archive.extractfile(member),dest)
        verify_files(repo,inputs['files'])
        frozen_sources=json.loads((repo/'inputs/reference-v5-native-source.json').read_text())
        frozen_source_files={name:{'sha256':sha} for name,sha in frozen_sources['files'].items()}
        verify_files(reference/'repo',frozen_source_files)
        upstream=json.loads((repo/'inputs/upstream.json').read_text())
        assert upstream['revision']=='456172ec733a135778adcd32d00e576a58232e45'
        old_ggml=reference/'repo/ExternalProjects/ggml'
        report['upstream_before']=verify_upstream(old_ggml,upstream['files'])
        for name in upstream['files']:
            target=repo/'ExternalProjects/ggml'/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(old_ggml/name,target,follow_symlinks=False)
        report['upstream_revision']=upstream['revision']
        fixture_manifest=json.loads((repo/'inputs/allocation-recovery-fixtures-manifest.json').read_text())
        fixture_files={row['path']:{'sha256':row['sha256']} for row in fixture_manifest['files']}
        verify_files(reference,fixture_files)
        assert digest(reference/'fixture-python-deps-manifest.json')==digest(repo/'inputs/fixture-python-deps-manifest.json'), 'Reader manifest changed'
        reader_manifest=json.loads((repo/'inputs/fixture-python-deps-manifest.json').read_text())
        reader_files={name:{'sha256':sha} for name,sha in reader_manifest['files'].items()}
        verify_files(reference/'fixture-python-deps',reader_files)
        report['reader_manifest_sha256']=digest(reference/'fixture-python-deps-manifest.json')
        report['python_environment']=json.loads((reference/'fixture-python-environment.json').read_text())
        report['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,memory.total,driver_version','--format=csv'],text=True)
        old_cache=reference/'repo/TensorSharp.GGML.Native/build/CMakeCache.txt'
        cache={line.split(':',1)[0]:line.split('=',1)[1] for line in old_cache.read_text().splitlines()
               if ':' in line and '=' in line and not line.startswith(('#','//'))}
        configure=['cmake','-S',str(native),'-B',str(build),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release',
                   '-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON','-DTENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF',
                   '-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON','-DCMAKE_CUDA_ARCHITECTURES=86-real']
        for key in ('CMAKE_CUDA_COMPILER','TSG_CUDNN_INCLUDE_DIR','TENSORSHARP_GGML_NATIVE_ENABLE_NCCL'):
            if cache.get(key) and not cache[key].endswith('-NOTFOUND'):
                configure.append('-D'+key+'='+cache[key])
        if run('configure',configure) or run('build-all',['cmake','--build',str(build),'--parallel',str(args.jobs)]):
            raise RuntimeError('Native configure/build failed; executable gates were not run')
        report['native_sha256']=digest(build/'libGgmlOps.so')
        report['test_products']={p.name:digest(p) for p in build.glob('*Test') if p.is_file()}
        shutil.copy2(build/'CMakeCache.txt',out/'CMakeCache.txt');save()
        with (out/'ctest-inventory.json').open('x') as stream:
            subprocess.run(['ctest','--test-dir',str(build),'--show-only=json-v1'],stdout=stream,check=True)
        run('native-tests',['ctest','--test-dir',str(build),'-V','--output-on-failure','-j','1','--output-junit',str(out/'native-tests.xml')])
        junit=ET.parse(out/'native-tests.xml').getroot()
        report['ctest_cases']=[{'name':case.attrib.get('name'),'status':'failed' if case.find('failure') is not None
                               else 'unavailable' if case.find('skipped') is not None else 'passed'}
                              for case in junit.iter('testcase')]
        unavailable=[case['name'] for case in report['ctest_cases'] if case['status']=='unavailable']
        if unavailable:
            report['steps'].append({'name':'ctest-hardware-skips','status':'unavailable','cases':unavailable})
        run('precision-overrides',['ctest','--test-dir',str(build),'-V','--output-on-failure','-j','1',
                                   '-R','^cuda-(explicit-f32-(attention|matmul)|q8-f32-projection)$'],
            {'GGML_CUDA_CUBLAS_COMPUTE_TYPE':'f16','NVIDIA_TF32_OVERRIDE':'1'})
        # Keep the original full gate and bounds. Its failure must not prevent
        # recording independent recovery/retention outcomes on the same build.
        run('tp7-checkpoint-shape',[str(build/'GgmlOpsDsv41TpTest'),'--cuda','7','--checkpoint-shape'])
        run('c-abi-boundary',[args.python,str(repo/'eng/tests/dsv4-execution-boundary.py'),'--library',str(build/'libGgmlOps.so'),
                              '--report',str(out/'c-abi-boundary.json')])
        for label,backend,gpus,tp in [('cpu','CPU',1,0),('cuda1','CUDA',1,0),('cuda7-tp7','CUDA',7,7)]:
            run('fixture-recovery-'+label,[args.python,str(repo/'eng/tests/dsv41-failure-state.py'),
                str(reference/'allocation-recovery-fixtures/text-f32-cuda-index'),'--vision-fixture',
                str(reference/'allocation-recovery-fixtures/vision-f32'),'--library',str(build/'libGgmlOps.so'),
                '--report',str(out/('fixture-recovery-'+label+'.json')),'--backend',backend,'--gpus',str(gpus),'--tp',str(tp)])
        for label,backend,extra in [('cpu','CPU',[]),('cpu-reserve','CPU',['--cpu-admission-reserve-mb','1048576']),('cuda1','CUDA',[])]:
            run('slot-retention-'+label,[args.python,str(repo/'eng/tests/dsv41-slot-retention.py'),
                str(repo/'fixtures/text-f32-small'),'--library',str(build/'libGgmlOps.so'),
                '--report',str(out/('slot-retention-'+label+'.json')),'--backend',backend]+extra,
                {'TS_DSV4_GRAPH_CACHE_HEADROOM_MB':'1048576'} if label=='cpu-reserve' else {})
        run('q8-memcheck',['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full',
                           '--error-exitcode','99',str(build/'GgmlOpsCudaQ8PrecisionTest')])
    except Exception as error:
        report['error']=repr(error)
        report['steps'].append({'name':'orchestration','status':'failed','error':repr(error)})
    finally:
        audit('owned-source-and-fixture-pins',lambda:verify_files(repo,inputs['files']))
        audit('reference-native-unchanged',lambda: (_ for _ in ()).throw(AssertionError('Reference native changed'))
              if digest(frozen_native)!=inputs['reference_native_sha256'] else {'sha256':digest(frozen_native)})
        if 'upstream' in locals():
            audit('reference-upstream-unchanged',lambda:verify_upstream(old_ggml,upstream['files']))
            audit('fresh-upstream-unchanged',lambda:verify_upstream(repo/'ExternalProjects/ggml',upstream['files']))
        if 'fixture_files' in locals():
            audit('reference-fixtures-unchanged',lambda:verify_files(reference,fixture_files))
        if 'frozen_source_files' in locals():
            audit('reference-native-sources-unchanged',lambda:verify_files(reference/'repo',frozen_source_files))
        if 'reader_files' in locals():
            audit('reference-reader-unchanged',lambda:verify_files(reference/'fixture-python-deps',reader_files))
        if 'native_sha256' in report:
            audit('built-products-unchanged',lambda:verify_files(build,dict(
                {'libGgmlOps.so':{'sha256':report['native_sha256']}},
                **{name:{'sha256':sha} for name,sha in report['test_products'].items()})))
        report.update(status=result_status(report['steps'],report['audits']),finished_at_unix=time.time())
        save()
    print(json.dumps({key:report[key] for key in ('status','release_qualified','full_model_qualification')}))
    return 0 if report['status']=='native_gates_passed_full_model_not_run' else 1


if __name__=='__main__':
    raise SystemExit(main())
