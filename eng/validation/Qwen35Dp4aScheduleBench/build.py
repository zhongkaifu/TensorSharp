#!/usr/bin/env python3
"""Private N8/R4 launch-bounds and weight-cache ablation; offline build only."""
import argparse,hashlib,importlib.util,json,os,re,shutil,signal,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
OLD=ROOT/'eng/validation/Qwen35Dp4aRowsBench'
OLD_PINS={'build.py':'124d2e5279f07f63d4b7f780e8a6ae2290470caa379ded2d65f37e9c73fd734d','run.py':'f89cd630ba8376957041bf56509d4aa9ca5ab86e115f456963314b4b274b2f18','bench.cu':'d0697ebdc24518f9fd0ab5e0125196de9410498334c94939a6b1e446d528ea88'}
SPEC=importlib.util.spec_from_file_location('frozen_dp4a_rows_helpers',OLD/'build.py');BASE=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(BASE)
SEALED=BASE.SEALED;SOURCE=BASE.SOURCE;FLAGS=list(BASE.FLAGS)
ORACLE='ts_q35_gemv_nvfp4_8'
CANDIDATES=[dict(symbol='ts_bench_dp4a_r4_bounds3',rows_per_warp=4,block_threads=128,role='bounds3_only',launch_bounds_min_blocks=3,weights_cache='original'),dict(symbol='ts_bench_dp4a_r4_stream',rows_per_warp=4,block_threads=128,role='weights_stream_only',launch_bounds_min_blocks=None,weights_cache='__ldcs'),dict(symbol='ts_bench_dp4a_r4_bounds3_stream',rows_per_warp=4,block_threads=128,role='bounds3_and_weights_stream',launch_bounds_min_blocks=3,weights_cache='__ldcs')]
SYMBOLS=(ORACLE,*(v['symbol']for v in CANDIDATES));RUNNER='dp4a-schedule-bench';PERMIT='dp4a-schedule-offline-built.json'
ARTIFACTS=('kernels.cu','kernels.cubin','kernels.ptx','plan.json','plan.h','variants.h',RUNNER)
TOOLS=('build.py','run.py','bench.cu');TEMPLATE_SHA='e1cdd124f506ec0d8d4d94677ffaaf3908029e473fe44cfd757fc336a3712c89'
OLD_VECTOR='            const uint4 c0 = c[0], c1 = c[1];'
NEW_VECTOR='            const uint4 c0 = __ldcs(c), c1 = __ldcs(c + 1);'
OLD_SCALE='            dw[j] = *(const uint32_t*)(scales + r * (k / 16) + (long long)b * 4);'
NEW_SCALE='            dw[j] = __ldcs((const uint32_t*)(scales + r * (k / 16) + (long long)b * 4));'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def evidence_dir(path):
    absolute=path.absolute()
    if str(absolute)!=os.path.normpath(str(absolute))or absolute.resolve()!=absolute or not any(absolute.is_relative_to(ROOT/r)for r in('artifacts','docs/validation')):raise ValueError('require normalized nonalias ignored output')
    return absolute
def fixture_pins():return {str(p.relative_to(ROOT)):digest(p)for p in [*(OLD/n for n in OLD_PINS),*(BASE.FROZEN/n for n in BASE.FROZEN_FILES)]}
def check_frozen():
    if any(digest(OLD/n)!=sha for n,sha in OLD_PINS.items()):raise ValueError('old frozen DP4a tools changed')
def template(source):
    start=source.index('template <int N, int R>');end=source.index('// Rows per warp by activation rows:',start);s=source[start:end]
    if hashlib.sha256(s.encode()).hexdigest()!=TEMPLATE_SHA or s.count(OLD_VECTOR)!=1 or s.count(OLD_SCALE)!=1:raise ValueError('original reviewed N8/R4 template changed')
    return s
def generate(source,candidates=None):
    prefix=SEALED.oracle_prefix(source);original=template(source)
    stream=original.replace('void ts_q35_gemv_nvfp4(', 'void ts_bench_private_gemv_nvfp4_stream(',1).replace(OLD_VECTOR,NEW_VECTOR,1).replace(OLD_SCALE,NEW_SCALE,1)
    if stream.replace('void ts_bench_private_gemv_nvfp4_stream(', 'void ts_q35_gemv_nvfp4(',1).replace(NEW_VECTOR,OLD_VECTOR,1).replace(NEW_SCALE,OLD_SCALE,1)!=original:raise ValueError('cache arm changed math or control')
    wrappers=[]
    for variant in (CANDIDATES if candidates is None else candidates):
        bound=' __launch_bounds__(128,3)'if variant['launch_bounds_min_blocks']else''
        function='ts_bench_private_gemv_nvfp4_stream'if variant['weights_cache']=='__ldcs'else'ts_q35_gemv_nvfp4'
        wrappers.append(f'''extern "C" __global__{bound} void {variant['symbol']}(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y)
{{ {function}<8,4>(codes,scales,outDim,k,xq,xd,alpha,y); }}
''')
    return '#include <cuda_runtime.h>\n'+prefix+'\n// Private cache/lifetime arms, production source remains unchanged.\n'+stream+'\n'.join(wrappers)
def parse_resources(log,symbols=SYMBOLS):
    out={}
    for name in symbols:
        records=re.findall(r"Compiling entry function ['\"]"+re.escape(name)+r"['\"][\s\S]*?(?=Compiling entry function|\Z)",log)
        if len(records)!=1:raise ValueError('missing/duplicate actual resource:'+name)
        text=records[0];r=re.search(r'Used (\d+) registers',text);s=re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',text);sm=re.search(r'(\d+) bytes smem',text)
        if not r or not s:raise ValueError('incomplete actual resources:'+name)
        x=dict(registers_per_thread=int(r[1]),stack_bytes=int(s[1]),spill_store_bytes=int(s[2]),spill_load_bytes=int(s[3]),static_shared_bytes=int(sm[1])if sm else 0,block_threads=128)
        x['offline_resource_qualified']=0<x['registers_per_thread']<=255 and all(x[k]==0 for k in('stack_bytes','spill_store_bytes','spill_load_bytes','static_shared_bytes'));out[name]=x
    return out
def command(argv,timeout,stdout=None,on_started=None):
    # Single-threaded CLI only. Block registration signals briefly, and undo the
    # inherited mask in the new child before exec (setsid otherwise escapes the
    # outer controller's process group). The Popen handle remains unreaped until
    # communicate/cleanup, so its PID cannot identify a recycled foreign group.
    process=None;oldmask=None
    handlers={s:signal.getsignal(s)for s in(signal.SIGINT,signal.SIGTERM)}
    def interrupt(signum,frame):
        for s in handlers:signal.signal(s,signal.SIG_IGN)
        raise KeyboardInterrupt('owned command signal '+str(signum))
    try:
        for s in handlers:signal.signal(s,interrupt)
        oldmask=signal.pthread_sigmask(signal.SIG_BLOCK,set(handlers))
        try:
            process=subprocess.Popen(argv,stdout=subprocess.PIPE if stdout is None else stdout,stderr=subprocess.STDOUT,text=True,start_new_session=True,preexec_fn=lambda:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask))
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        if on_started is not None:on_started(process)
        output,_=process.communicate(timeout=timeout);return process.returncode,output or ''
    except BaseException:
        for s in handlers:signal.signal(s,signal.SIG_IGN)
        if process is not None and process.returncode is None:
            try:
                if os.getpgid(process.pid)==process.pid:os.killpg(process.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.getpgid(process.pid)==process.pid:os.killpg(process.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                process.communicate()
        raise
    finally:
        if oldmask is not None:signal.pthread_sigmask(signal.SIG_SETMASK,oldmask)
        for s,handler in handlers.items():signal.signal(s,handler)
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',required=True,type=Path);ap.add_argument('--source',type=Path,default=SOURCE);ap.add_argument('--generate-only',action='store_true');ap.add_argument('--nvcc',default='nvcc');ap.add_argument('--cuobjdump',default='cuobjdump');ap.add_argument('--timeout',type=int,default=600);ap.add_argument('--arms',choices=('all','bounds','stream','combo'),default='all',help='Explicit fresh build selection; no automatic pruning/fallback.');a=ap.parse_args(argv)
    try:
        if any(os.environ.get(n,'').strip()for n in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS')):raise ValueError('implicit compiler flags forbidden')
        check_frozen();out=evidence_dir(a.out)
        if out.exists()or out.is_symlink()or a.timeout<=0:raise ValueError('fresh output and positive timeout required')
    except ValueError as e:ap.error(str(e))
    candidates=CANDIDATES if a.arms=='all'else[CANDIDATES[('bounds','stream','combo').index(a.arms)]]
    out.mkdir(parents=True);state=dict(schema='q35-dp4a-schedule-build/1',complete=False,generation_complete=False,source=str(a.source.resolve()),source_sha256=digest(a.source),tools={n:digest(HERE/n)for n in TOOLS},frozen_fixture_tools=fixture_pins(),flags=FLAGS,activation_rows=8,oracle=dict(symbol=ORACLE,rows_per_warp=4,block_threads=128),oracle_is_freshly_recompiled_NOT_production_BBD=True,arm_selection=a.arms,candidates=candidates,gpu_quality='NOT_RUN',gpu_timing='NOT_RUN',hardware_counters='NOT_RUN',production_integration=False,build_environment={n:os.environ.get(n)for n in('NVCC_PREPEND_FLAGS','NVCC_APPEND_FLAGS','NVCC_CCBIN','CUDA_HOME','CUDA_PATH')})
    save(out/'manifest.json',state)
    try:
        source=a.source.read_text();generated=generate(source,candidates);cases=SEALED.plan()
        if '__shared__'in generated:raise ValueError('unexpected shared object')
        (out/'kernels.cu').write_text(generated);save(out/'plan.json',cases);(out/'plan.h').write_text(SEALED.plan_header(cases))
        (out/'variants.h').write_text('std::array<Variant,'+str(len(candidates))+'> candidates{{'+','.join('{4,128,"'+v['symbol']+'"}'for v in candidates)+'}};\n')
        state.update(generation_complete=True,generated_sha256=digest(out/'kernels.cu'),strict_cases=len(cases),timed_shapes=sum(s['timed']for s in cases),planned_comparisons=len(cases)*len(candidates),planned_ABBA_events=sum(s['timed']for s in cases)*len(candidates)*5)
        if a.generate_only:state['status']='GENERATED_ONLY_GPU_NOT_RUN';save(out/'manifest.json',state);print(json.dumps(state,indent=2));return 0
        code,text=command([a.nvcc,'--version'],30);(out/'nvcc-version.log').write_text(text);state['nvcc_version']=text
        if code:raise RuntimeError('nvcc version failed')
        common=[a.nvcc,'-std=c++17','-arch=sm_120a',*FLAGS]
        for key,cmd,log in [('cubin',common+['-cubin','-Xptxas=-v,-warn-spills',str(out/'kernels.cu'),'-o',str(out/'kernels.cubin')],'kernels.build.log'),('ptx',common+['-ptx',str(out/'kernels.cu'),'-o',str(out/'kernels.ptx')],'ptx.build.log')]:
            code,text=command(cmd,a.timeout);(out/log).write_text(text);state[key+'_command']=cmd;state[key+'_exit_code']=code
            if code:raise RuntimeError(key+' compilation failed')
            if key=='cubin':state['resources']=parse_resources(text,(ORACLE,*(v['symbol']for v in candidates)))
        if not all(r['offline_resource_qualified']for r in state['resources'].values()):raise RuntimeError('actual stack/spill/local-resource policy rejected; no permit')
        dump=shutil.which(a.cuobjdump)
        if dump:
            code,text=command([dump,'--dump-sass',str(out/'kernels.cubin')],a.timeout);(out/'kernels.sass.txt').write_text(text);state['sass_exit_code']=code
            if code:raise RuntimeError('SASS extraction failed')
        cmd=[a.nvcc,'-std=c++17','-O3','-I',str(out),str(HERE/'bench.cu'),'-lcuda','-o',str(out/RUNNER)];code,text=command(cmd,a.timeout);(out/'runner.build.log').write_text(text);state['runner_command']=cmd;state['runner_exit_code']=code
        if code:raise RuntimeError('runner build failed')
        if digest(a.source)!=state['source_sha256']or state['tools']!={n:digest(HERE/n)for n in TOOLS}or state['frozen_fixture_tools']!=fixture_pins():raise RuntimeError('source/tools changed during build')
        state.update(complete=True,status='OFFLINE_BUILT_NO_GPU_VALIDATION',artifacts={n:digest(out/n)for n in ARTIFACTS});save(out/'manifest.json',state);save(out/PERMIT,dict(schema='q35-dp4a-schedule-offline/1',manifest_sha256=digest(out/'manifest.json'),artifacts=state['artifacts'],status='DRIVER_AND_FULLBITS_NOT_RUN'))
        print(json.dumps(state,indent=2));return 0
    except BaseException as error:
        state.update(complete=False,status='BUILD_REJECTED_GPU_NOT_RUN',error=repr(error));save(out/'manifest.json',state)
        if isinstance(error,(KeyboardInterrupt,SystemExit)):raise
        print(json.dumps(state,indent=2));return 1
if __name__=='__main__':raise SystemExit(main())
