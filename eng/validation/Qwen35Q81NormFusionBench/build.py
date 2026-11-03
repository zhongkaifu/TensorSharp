#!/usr/bin/env python3
"""Generate a private exact-order norm/Q81 fusion kernel; CUDA compile is explicit."""
import argparse, hashlib, json, os, re, subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / 'TensorSharp.Backends.Cuda/native/kernels/tensorsharp_q35_kernels.cu'
SYMBOL = 'ts_bench_add_rmsnorm_q81_f32'
NORM_SHA = 'bd9bfb966a5d8de2b72ea192b974c78bd9a5e270986d6e0ce27050dd7cb17100'
PREFIX_SHA = 'e7ac0b2f131f68e2d28350a27c141704b4b6d2939406f2d01cbce6b17044df66'
FLAGS = ['-std=c++17', '-O3', '--extra-device-vectorization', '--ftz=false',
         '--prec-div=true', '--prec-sqrt=true', '--fmad=true', '-lineinfo']
STORE = '        y[row * width + i] = xr[i] * scale * w[i];'
VOLATILE_STORE = '        ((volatile float*)y)[row * width + i] = xr[i] * scale * w[i];'
TAIL = '''
    // Norm's F32 stores are a real materialization boundary, just as between
    // the original two launches. All this row's writers finish before readers.
    __syncthreads();
    const int blocks = width >> 5;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    for (int b = warp; b < blocks; b += (blockDim.x >> 5))
    {
        const float v = ((const volatile float*)y)[row * width + b * 32 + lane];
        const float amax = ts_q35_warp_max(fabsf(v));
        const float d = amax / 127.0f;
        const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
        xq[row * width + b * 32 + lane] = (int8_t)q;
        if (lane == 0)
            xd[row * blocks + b] = d;
    }
'''

def sha(data): return hashlib.sha256(data).hexdigest()
def file_sha(path): return sha(Path(path).read_bytes())
def save(path, value): path.write_text(json.dumps(value, indent=2) + '\n')

def generate(source):
    prefix = source[:source.index('// ---------------------------------------------------------------------------\n// Activation quantization:')]
    begin = source.index('extern "C" __global__ void ts_q35_add_rmsnorm_f32(')
    end = source.index('\n// ---------------------------------------------------------------------------', begin)
    norm = source[begin:end]
    if sha(prefix.encode()) != PREFIX_SHA or sha(norm.encode()) != NORM_SHA:
        raise ValueError('reviewed original norm/helper bytes changed')
    signature = '    float* __restrict__ y, int width, float eps)'
    extended = '    float* __restrict__ y, int width, float eps, int8_t* __restrict__ xq, float* __restrict__ xd)'
    if norm.count(STORE) != 1 or norm.count(signature) != 1 or not norm.endswith('}\n'):
        raise ValueError('reviewed norm structure changed')
    candidate = norm.replace('ts_q35_add_rmsnorm_f32(', SYMBOL+'(', 1).replace(signature, extended, 1).replace(STORE, VOLATILE_STORE, 1)
    candidate = candidate[:-2] + TAIL + '}\n'
    restored = candidate.replace(TAIL, '', 1).replace(VOLATILE_STORE, STORE, 1).replace(extended, signature, 1).replace(SYMBOL+'(', 'ts_q35_add_rmsnorm_f32(', 1)
    if restored != norm: raise ValueError('norm arithmetic/reduction was modified')
    return prefix + '\n// Standalone private diagnostic; production kernel/ABI unchanged.\n' + candidate, norm

def geometry(rows, width):
    if not 1 <= rows <= 8 or width <= 0 or width % 32: raise ValueError('unsupported component geometry')
    return [(row, b, lane) for row in range(rows) for warp in range(16)
            for b in range(warp, width//32, 16) for lane in range(32)]

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--compile', action='store_true')
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--timeout', type=int, default=600)
    args = parser.parse_args(argv)
    if any(os.environ.get(k) for k in ('NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS')):
        parser.error('implicit NVCC flags are forbidden')
    absolute = args.out.absolute()
    if args.out != absolute or args.out.exists() or args.out.is_symlink() or args.timeout <= 0:
        parser.error('output must be fresh absolute normalized and timeout positive')
    if absolute != absolute.resolve() or not any(absolute.is_relative_to(ROOT/r) for r in ('artifacts','docs/validation')):
        parser.error('output must have no aliases and be inside ignored artifacts/ or docs/validation/')
    text = args.source.read_text(); generated, norm = generate(text)
    args.out.mkdir(parents=True)
    (args.out/'kernels.cu').write_text(generated)
    state = dict(schema='q35-q81-norm-fusion-offline/1', complete=False,
                 quality='GPU_NOT_RUN', compiled=False, source_path=str(args.source),
                 source_sha256=sha(text.encode()), original_norm_sha256=sha(norm.encode()),
                 original_helpers_sha256=PREFIX_SHA, generated_sha256=sha(generated.encode()),
                 tools={p.name:file_sha(p) for p in (HERE/'build.py', HERE/'Program.cs', HERE/'Native.cs', HERE/'Qwen35Q81NormFusionBench.csproj')},
                 flags=FLAGS, fixed_block_threads=512, math='Original norm order; volatile F32 store/load boundary; original warp32 Q81 math.',
                 limitations=['No GPU is initialized by build.py.', '8 fused CTAs may underutilize the GPU; launch elimination is not a measured speedup.'])
    save(args.out/'build.json', state)
    if args.compile:
        try:
            version = subprocess.run([args.nvcc,'--version'], capture_output=True,text=True,timeout=30)
            (args.out/'nvcc-version.log').write_text(version.stdout+version.stderr)
            if version.returncode: raise RuntimeError('nvcc version failed')
            state['nvcc_version'] = version.stdout+version.stderr
            command = [args.nvcc,*FLAGS,'-arch=sm_120a','-cubin','-Xptxas=-v,-warn-spills',str(args.out/'kernels.cu'),'-o',str(args.out/'kernels.cubin')]
            state['command'] = command
            result = subprocess.run(command,capture_output=True,text=True,timeout=args.timeout)
            log = result.stdout+result.stderr; (args.out/'nvcc.log').write_text(log)
            state['compile_exit_code'] = result.returncode
            if result.returncode: raise RuntimeError('nvcc compilation failed')
            records = re.findall(r"Compiling entry function ['\"]"+SYMBOL+r"['\"][\s\S]*?(?=Compiling entry function|\Z)",log)
            if len(records) != 1: raise RuntimeError('missing actual ptxas resource record')
            resource = records[0]
            regs = re.search(r'Used (\d+) registers',resource)
            stack = re.search(r'(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads',resource)
            shared = re.search(r'(\d+) bytes smem',resource)
            if not regs or not stack: raise RuntimeError('incomplete actual resources')
            state['ptxas'] = dict(registers=int(regs[1]),stack_bytes=int(stack[1]),spill_store_bytes=int(stack[2]),spill_load_bytes=int(stack[3]),static_shared_bytes=int(shared[1])if shared else 0)
            state['cubin_sha256'] = file_sha(args.out/'kernels.cubin')
            state['compiled'] = state['complete'] = True
        except BaseException as error:
            state['error'] = repr(error); save(args.out/'build.json',state); raise
    save(args.out/'build.json',state)
    print(json.dumps(dict(out=str(args.out),compiled=state['compiled'],quality='GPU_NOT_RUN')))

if __name__ == '__main__': main()
