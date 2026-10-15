#!/usr/bin/env python3
"""Capture full-vocabulary logits for fixed valid IDs through the native executor.

This checks numerical invariance between configurations of the same checkpoint,
not language quality or an independent implementation. Run each configuration
in a fresh process and compare .f32 files exactly. No throughput claim: file I/O
and full-logit transfers are deliberately included in this correctness run.
"""
import argparse
import ctypes as c
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time
import numpy as np


def checkpoint_mapping_ranges(text, paths, page):
    allowed = {str(Path(path).resolve()) for path in paths}
    result = []
    for line in text.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or fields[5] not in allowed:
            continue
        start, end = (int(value, 16) for value in fields[0].split('-'))
        if start <= 0 or end <= start or start % page or end % page or fields[1][0] != 'r' or 'w' in fields[1]:
            raise ValueError('Refuse advising an invalid or writable checkpoint mapping')
        result.append(dict(address=start, length=end-start, path=fields[5]))
    if not result:
        raise ValueError('No owned read-only checkpoint mappings found')
    return result


def evict_owned_checkpoint_pages(paths):
    """Only call between synchronous forwards in this standalone owner process."""
    import mmap
    if not Path('/proc/self/maps').exists():
        raise RuntimeError('Owned mapping eviction requires Linux')
    ranges = checkpoint_mapping_ranges(Path('/proc/self/maps').read_text(), paths, os.sysconf('SC_PAGE_SIZE'))
    libc = c.CDLL(None, use_errno=True)
    libc.madvise.argtypes = (c.c_void_p, c.c_size_t, c.c_int)
    libc.madvise.restype = c.c_int
    for row in ranges:
        if libc.madvise(row['address'], row['length'], mmap.MADV_DONTNEED):
            code = c.get_errno()
            raise OSError(code, os.strerror(code))
    spec = importlib.util.spec_from_file_location('checkpoint_cache', Path(__file__).with_name('checkpoint-page-cache.py'))
    cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache)
    return dict(owned_mappings=ranges, cache=cache.cache_state(paths, evict=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--checkpoint-manifest',type=Path,required=True)
    p.add_argument('--host-staging-budget-mib',type=int,help='Attach the real native host budget callbacks; verify credit returns after model destruction')
    p.add_argument('--evict-after-warm',action='store_true',help='Linux: evict only this paused owner\'s checkpoint mappings, then compare restored logits with a warm reference')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.host_staging_budget_mib is not None and args.host_staging_budget_mib <= 0:
        p.error('Host staging budget must be positive')
    root=Path(__file__).resolve().parents[2]
    out=args.output.resolve()
    if not any(out.is_relative_to(root/d) for d in ('artifacts','docs/validation')):
        p.error('Evidence belongs in ignored artifact directories')
    out.mkdir(parents=True,exist_ok=False)
    library=args.library.resolve(strict=True)
    spec=importlib.util.spec_from_file_location('checkpoint_identity',Path(__file__).with_name('multimodel-quality-bench.py'))
    identity=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(identity)
    shards=identity.verified_identities(identity.checkpoint_files(args.model),args.checkpoint_manifest)
    dll=c.CDLL(str(library))
    def bind(name,result,types):
        fn=getattr(dll,name);fn.restype=result;fn.argtypes=types;return fn
    load=bind('TSGgml_Dsv4LoadModel',c.c_void_p,[c.c_char_p,c.c_int,c.c_int,c.c_int,c.c_int,c.c_char_p,c.c_int,c.c_char_p,c.c_int])
    vocab=bind('TSGgml_Dsv4VocabSize',c.c_int,[c.c_void_p])
    forward=bind('TSGgml_Dsv4Forward',c.c_int,[c.c_void_p,c.POINTER(c.c_int32),c.c_int,c.POINTER(c.c_float)])
    reset=bind('TSGgml_Dsv4ResetChecked',c.c_int,[c.c_void_p])
    free=bind('TSGgml_Dsv4Free',None,[c.c_void_p])
    error=bind('TSGgml_GetLastError',c.c_char_p,[])
    report=dict(started_unix=time.time(),library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        model=str(args.model.resolve()),checkpoint_shards=shards,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        environment={k:v for k,v in os.environ.items() if k.startswith(('TS_', 'GGML_', 'OMP_', 'GOMP_')) or k=='CUDA_VISIBLE_DEVICES'},
        rows=[],complete=False,limitation='Fixed-ID numerical invariance only; not semantic quality or independent reference logits.')
    budget=None
    if args.host_staging_budget_mib is not None:
        budget=dict(limit=args.host_staging_budget_mib*2**20,used=0,peak=0,next=1,live={},errors=[])
        reserve_type=c.CFUNCTYPE(c.c_uint64,c.c_void_p,c.c_int,c.c_int,c.c_int64)
        commit_type=c.CFUNCTYPE(c.c_int,c.c_void_p,c.c_uint64)
        release_type=c.CFUNCTYPE(None,c.c_void_p,c.c_uint64)
        @reserve_type
        def reserve_callback(context,rank,kind,amount):
            try:
                if amount<=0: raise ValueError('Invalid native reservation')
                if kind==3 and amount>budget['limit']-budget['used']: return 0
                token=budget['next'];budget['next']+=1
                budget['live'][token]=(kind,amount)
                if kind==3:
                    budget['used']+=amount;budget['peak']=max(budget['peak'],budget['used'])
                return token
            except Exception as ex: budget['errors'].append(str(ex));return 0
        @commit_type
        def commit_callback(context,token):
            return int(token in budget['live'])
        @release_type
        def release_callback(context,token):
            try:
                kind,amount=budget['live'].pop(token)
                if kind==3: budget['used']-=amount
            except Exception as ex: budget['errors'].append(str(ex))
        attach=bind('TSGgml_AttachSharedCacheBudgetWithHost',c.c_int,[c.c_void_p,reserve_type,commit_type,release_type,c.c_int])
        detach=bind('TSGgml_DetachSharedCacheBudget',c.c_int,[c.c_void_p])
        if attach(c.c_void_p(1),reserve_callback,commit_callback,release_callback,0)!=1:
            raise RuntimeError('Cannot attach native host staging budget')
    handle=None
    try:
        handle=load(str(args.model.resolve()).encode(),2,4096,0,16,None,-1,b'CUDA',0)
        if not handle: raise RuntimeError((error() or b'Native load failed').decode())
        size=vocab(handle)
        if size < 8192: raise RuntimeError('Checkpoint vocabulary smaller than fixture')
        report['vocabulary']=size
        batches=[[1,42,103,512,1024,2048,3072,4096,511,17,8191,73,2026,256,333,789,555]]
        batches += [[token] for token in (88,1025,304,77,1234,7777,3000,5)]
        batches += [[(i*127+17)%8192 for i in range(31)]]
        def run(ids):
            tokens=(c.c_int32*len(ids))(*ids)
            output=np.empty(size,dtype='<f4')
            if forward(handle,tokens,len(ids),output.ctypes.data_as(c.POINTER(c.c_float)))!=0:
                raise RuntimeError((error() or b'Native forward failed').decode())
            if not np.isfinite(output).all(): raise RuntimeError('Nonfinite logits')
            return output
        def capture(ids, do_reset=False, label=None):
            if do_reset and reset(handle)!=1: raise RuntimeError('Native reset failed')
            if label is not None: print(json.dumps(dict(capture_phase=label,step=len(report['rows']))),flush=True)
            output=run(ids)
            data=output.tobytes()
            step=len(report['rows'])
            name=f'{step:02d}.f32'
            (out/name).write_bytes(data)
            report['rows'].append(dict(step=step,tokens=ids,reset=do_reset,
                file=name,sha256=hashlib.sha256(data).hexdigest(),argmax=int(output.argmax())))
            if label is not None: report['rows'][-1]['phase']=label
            return data
        for step,ids in enumerate(batches):
            capture(ids, step==len(batches)-1)
        if args.evict_after_warm:
            import resource
            # Use precisely the same KV/token history on either side of the
            # eviction. Validate replay before perturbing the OS cache.
            replay=[]
            for i,ids in enumerate(batches[:-1]):
                replay.append(capture(ids, i==0, 'pressure-warm'))
            expected=capture([999], label='pressure-reference')
            if reset(handle)!=1: raise RuntimeError('Pressure replay reset failed')
            for ids, expected_replay in zip(batches[:-1], replay):
                if run(ids).tobytes()!=expected_replay: raise RuntimeError('Pressure fixture replay changed logits')
            print(json.dumps(dict(capture_phase='pressure-eviction',owned_process=os.getpid())),flush=True)
            report['pressure']=evict_owned_checkpoint_pages([row['path'] for row in shards])
            if any(row['resident_after'] for row in report['pressure']['cache']['files']):
                raise RuntimeError('Kernel did not evict every checkpoint page; pressure coverage unavailable')
            report['pressure']['major_faults_before']=resource.getrusage(resource.RUSAGE_SELF).ru_majflt
            actual=capture([999], label='pressure-after-eviction')
            report['pressure']['major_faults_after']=resource.getrusage(resource.RUSAGE_SELF).ru_majflt
            report['pressure']['bit_identical']=actual==expected
            if actual!=expected: raise RuntimeError('Eviction changed full-vocabulary logits')
        report['complete']=True
    finally:
        if handle: free(handle)
        if budget is not None:
            detached=detach(c.c_void_p(1))==1
            report['staging_budget']=dict(limit_bytes=budget['limit'],peak_bytes=budget['peak'],
                remaining_bytes=budget['used'],remaining_allocations=len(budget['live']),detached=detached,errors=budget['errors'])
            if budget['used'] or budget['live'] or budget['errors'] or not detached: report['complete']=False
        report['finished_unix']=time.time()
        (out/'report.json').write_text(json.dumps(report,indent=2))
    if not report['complete']: raise RuntimeError('Incomplete capture or native staging budget cleanup')
    print(json.dumps(dict(complete=report['complete'],steps=len(report['rows']),vocabulary=report.get('vocabulary'))))


if __name__=='__main__': main()
