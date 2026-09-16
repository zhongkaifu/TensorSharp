from pathlib import Path
import shlex, subprocess
root=Path('/workspace/ts-tp-active-prototype-20260916')
repo=Path('/workspace/ts-codex-20260916-r3/repo')
build=repo/'TensorSharp.GGML.Native/build'
includes=['-I'+str(repo/'TensorSharp.GGML.Native'),'-I'+str(repo/'ExternalProjects/ggml/src'),'-I'+str(repo/'ExternalProjects/ggml/include'),'-I/usr/local/cuda/include']
with (root/'strip-benchmark-build.log').open('w') as log:
    def run(args):
        log.write(shlex.join(args)+'\n');log.flush()
        subprocess.run(args,cwd=build,stdout=log,stderr=subprocess.STDOUT,check=True)
    replacements = {}
    for name in ['ggml_ops_matmul_precision.cpp','ggml_ops_matmul_precision.cu','ggml_ops_dsv4_fused.cu','ggml_ops_deepseek41_tp.cpp']:
        output=root/(name+'.cmakeflags.o')
        source=root/name
        command=(['/usr/local/cuda/bin/nvcc','-O3','-DNDEBUG','-std=c++17','-arch=sm_86','-extended-lambda',*(['--ftz=false','--prec-div=true','--prec-sqrt=true'] if name=='ggml_ops_matmul_precision.cu' else ['-use_fast_math'])] if name.endswith('.cu') else ['c++','-O3','-DNDEBUG','-std=c++17'])
        if not output.exists() or output.stat().st_mtime < max(source.stat().st_mtime, (root/'ggml_ops_matmul_quant_strip.cuh').stat().st_mtime):
            run(command+['-DTSG_GGML_TEST_HOOKS','-DTSG_GGML_USE_CUDA=1','-I'+str(root),*includes,'-c',str(source),'-o',str(output)])
        replacements['CMakeFiles/GgmlOpsDsv41TpTest.dir/'+name+'.o']=str(output)
    run(['c++','-O3','-DNDEBUG','-std=c++17','-DTSG_GGML_TEST_HOOKS','-DTSG_GGML_USE_CUDA=1','-I'+str(root),*includes,'-c',str(root/'ts-tp-strip-benchmark.cpp'),'-o',str(root/'strip-benchmark.o')])
    replacements['CMakeFiles/GgmlOpsDsv41TpTest.dir/tests/dsv41_tp_test.cpp.o']=str(root/'strip-benchmark.o')
    link=shlex.split((build/'CMakeFiles/GgmlOpsDsv41TpTest.dir/link.txt').read_text())
    link=[replacements.get(x,x) for x in link]
    link[link.index('-o')+1]=str(root/'strip-benchmark')
    run(link)
