// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Tensor-core flash prefill attention (direct-CUDA prefill plan, WP1).
//
// The tensor-core flash attention family for prefill-sized query blocks: GQA groups folded
// into the query tile, F16 K/V tiles in shared memory, online softmax with sinks, causal and
// sliding-window masking, split-KV for long contexts. Called from
// CudaKernelOps.TryGqaPrefillAttention(WithSinks); decode and verify windows keep their kernels.
//
// Scaffold (P0): the module holds only its capability probe so far. The committed
// compute_75 PTX in native/ptx is what every build without nvcc loads, so it is a
// tracked artifact that must be refreshed with the source (DEVELOPMENT.md). The probe
// is how the host tells a stale image from a current one: CudaFlashPrefillKernels
// compares TS_FLASH_PREFILL_VERSION, as the loaded image reports it, with the version its
// C# side was written against, and refuses a module that does not match - a
// stale committed PTX then keeps every route on its fallback instead of launching
// kernels that are not there (or that take other arguments).
//
// Bump TS_FLASH_PREFILL_VERSION, and CudaFlashPrefillKernels.CurrentVersion with it, whenever
// a kernel is added or its signature changes.
//
// sm_80-only code (cp.async, m16n8k16, bf16 mma) must sit under
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 and be gated on the host with
// CudaKernels.ArchGuardedKernelsUsable; prefer sm_75-clean code (m16n8k8 + ldmatrix)
// so the committed compute_75 PTX carries working kernels.
// ---------------------------------------------------------------------------

#define TS_FLASH_PREFILL_VERSION 1

/// Writes the module version to out[0] and the architecture this image was compiled for
/// (__CUDA_ARCH__ of the code the driver selected: the SASS of a fatbin, or the PTX target
/// a JIT compiled from) to out[1]. One thread; launch <<<1, 1>>>.
extern "C" __global__ void ts_flash_prefill_caps(int* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = TS_FLASH_PREFILL_VERSION;
#if defined(__CUDA_ARCH__)
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}
