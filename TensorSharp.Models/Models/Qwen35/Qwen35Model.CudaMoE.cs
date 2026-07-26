using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    // On-device direct-CUDA MoE decode for the Qwen3.5/3.6 MoE architectures. The
    // legacy MoEForward decode path routed the top-k on host (a DtoH sync of the
    // router logits) and ran each selected expert as a separate matmul whose output
    // was read back to host for the weighted accumulate — 82% of decode time on
    // Qwen3.6-35B-A3B-IQ2_XXS was these per-expert host round-trips, and they also
    // aborted the decode CUDA-graph capture.
    //
    // This mirrors Gemma4Model.TryCudaMoEForwardOnDevice, but for Qwen's separate
    // (unfused) gate/up expert weights, SwiGLU (not GEGLU) activation, and gated
    // shared expert. Router + gate/up + SwiGLU + down + weighted accumulate + the
    // shared expert all run as device kernels with no host readback, so the decode
    // layer loop stays CUDA-graph capturable. See CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU
    // and the ts_moe_* / ts_silu_mul / ts_moe_shared_gated_add kernels.
    public partial class Qwen35Model
    {
        private static readonly bool s_qwenCudaMoeOnDeviceEnabled =
            Environment.GetEnvironmentVariable("TS_CUDA_MOE_ONDEVICE") != "0";

        // On-device batched MoE for PREFILL is opt-in (TS_CUDA_MOE_PREFILL_ONDEVICE=1).
        // It is numerically byte-identical to the host path and removes all per-expert
        // host round-trips, but the current per-(token,expert) kernels re-read each
        // expert weight once per routed token with no cross-token batched-GEMM reuse,
        // so on this hardware it is SLOWER than the host expert-grouped batched-matmul
        // path (measured ~0.58x on Qwen3.6-35B-A3B-IQ2_XXS, 512-token prefill). Kept
        // off by default until a weight-reuse batched-GEMM prefill kernel lands; the
        // decode path (the reported gap) uses the always-on single-token kernels.
        private static readonly bool s_qwenCudaMoePrefillOnDevice =
            Environment.GetEnvironmentVariable("TS_CUDA_MOE_PREFILL_ONDEVICE") == "1";

        private IntPtr[] _qwenMoEGatePtrTable;   // per layer: device u64[numExperts]
        private IntPtr[] _qwenMoEUpPtrTable;     // per layer: device u64[numExperts]
        private IntPtr[] _qwenMoEDownPtrTable;   // per layer: device u64[numExperts]
        private int[] _qwenMoEGateUpType;        // per layer gate/up expert quant type
        private int[] _qwenMoEDownType;          // per layer down expert quant type
        private bool _qwenCudaMoETablesReady;    // tables built (or unavailable)
        private bool _qwenCudaMoEUsable;

        // Build the per-layer device pointer tables addressing the resident per-expert
        // gate / up / down quantized weights by expert id, so one captured graph can
        // replay for every token regardless of routing. Any gap (a non-resident
        // expert, a mismatched gate/up type) keeps the whole model on the host path.
        private void EnsureQwenCudaMoETablesBuilt()
        {
            if (_qwenCudaMoETablesReady)
                return;
            _qwenCudaMoETablesReady = true;
            _qwenCudaMoEUsable = false;

            if (_backend != BackendType.Cuda || !s_qwenCudaMoeOnDeviceEnabled
                || _numExperts <= 0 || _allocator is not CudaAllocator cudaAllocator)
                return;
            if (_expertGateQW == null || _expertUpQW == null || _expertDownQW == null)
                return;

            int numLayers = Config.NumLayers;
            _qwenMoEGatePtrTable = new IntPtr[numLayers];
            _qwenMoEUpPtrTable = new IntPtr[numLayers];
            _qwenMoEDownPtrTable = new IntPtr[numLayers];
            _qwenMoEGateUpType = new int[numLayers];
            _qwenMoEDownType = new int[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                if (_isMoeLayer == null || l >= _isMoeLayer.Length || !_isMoeLayer[l])
                    continue;
                if (_expertGateQW[l] == null || _expertUpQW[l] == null || _expertDownQW[l] == null)
                    return;

                var gPtrs = new IntPtr[_numExperts];
                var uPtrs = new IntPtr[_numExperts];
                var dPtrs = new IntPtr[_numExperts];
                int gateUpType = -1;
                int downType = -1;
                bool ok = true;

                for (int e = 0; e < _numExperts; e++)
                {
                    var gw = _expertGateQW[l][e];
                    var uw = _expertUpQW[l][e];
                    var dw = _expertDownQW[l][e];
                    if (gw == null || uw == null || dw == null) { ok = false; break; }

                    if (!CudaQuantizedOps.TryGetResidentDevicePtr(cudaAllocator, gw.EnsureDeviceCacheKey(), out IntPtr gDev, out int gType)
                        || !CudaQuantizedOps.TryGetResidentDevicePtr(cudaAllocator, uw.EnsureDeviceCacheKey(), out IntPtr uDev, out int uType)
                        || !CudaQuantizedOps.TryGetResidentDevicePtr(cudaAllocator, dw.EnsureDeviceCacheKey(), out IntPtr dDev, out int dType)
                        || gType != uType)
                    {
                        // gate and up must share a type (they run the same projection
                        // kernel against one q8_1 activation); down may differ.
                        ok = false;
                        break;
                    }

                    if (gateUpType < 0) { gateUpType = gType; downType = dType; }
                    else if (gateUpType != gType || downType != dType) { ok = false; break; }

                    gPtrs[e] = gDev;
                    uPtrs[e] = uDev;
                    dPtrs[e] = dDev;
                }

                if (!ok || gateUpType < 0)
                    return; // any gap -> keep the host path for the whole model

                _qwenMoEGatePtrTable[l] = CudaQuantizedOps.CreateDevicePointerTable(cudaAllocator, gPtrs);
                _qwenMoEUpPtrTable[l] = CudaQuantizedOps.CreateDevicePointerTable(cudaAllocator, uPtrs);
                _qwenMoEDownPtrTable[l] = CudaQuantizedOps.CreateDevicePointerTable(cudaAllocator, dPtrs);
                _qwenMoEGateUpType[l] = gateUpType;
                _qwenMoEDownType[l] = downType;
            }

            _qwenCudaMoEUsable = true;
        }

        private bool CanUseQwenCudaMoEOnDevice(int layer)
        {
            EnsureQwenCudaMoETablesBuilt();
            return _qwenCudaMoEUsable
                && _numExpertsUsed > 0
                && _numExpertsUsed <= _numExperts
                && _numExpertsUsed <= CudaFusedOps.MaxMoEExpertsUsed
                && _qwenMoEGatePtrTable != null
                && layer >= 0
                && layer < _qwenMoEGatePtrTable.Length
                && _qwenMoEGatePtrTable[layer] != IntPtr.Zero;
        }

        /// <summary>
        /// On-device MoE decode (seqLen == 1). <paramref name="input"/> is the already
        /// RMSNorm'd MoE input row (the router logits were computed from it upstream and
        /// are passed in). Returns the MoE output tensor on success (disposing
        /// <paramref name="routerLogits"/>), or null to fall back to the host
        /// <see cref="MoEForward"/> path (leaving <paramref name="routerLogits"/> alive).
        /// </summary>
        private Tensor TryCudaMoEForwardOnDevice(Tensor input, Tensor routerLogits, int layer)
        {
            if (!CanUseQwenCudaMoEOnDevice(layer))
                return null;

            int hiddenDim = Config.HiddenSize;
            int nFf = _expertFfnLength;
            int nUsed = _numExpertsUsed;
            int gateUpType = _qwenMoEGateUpType[layer];
            int downType = _qwenMoEDownType[layer];

            long t0 = Stopwatch.GetTimestamp();

            // Shared expert (dense FFN applied to every token): gate/up/SwiGLU/down as
            // device matmuls, then an on-device sigmoid-gated accumulate. Computed here
            // (needs the quantized-weight matmul infra) and handed to the fused op.
            Tensor sharedDown = null;
            IntPtr sharedGateVecPtr = IntPtr.Zero;
            if (_hasSharedExperts != null && layer < _hasSharedExperts.Length && _hasSharedExperts[layer])
            {
                Tensor sg = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor su = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sg != null && su != null)
                {
                    Ops.SiLUMul(sg, sg, su);
                    sharedDown = LinearForwardCached(sg, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                su?.Dispose();
                sg?.Dispose();

                bool sharedGateRequired = sharedDown != null
                    && _hasSharedExpertGate != null
                    && layer < _hasSharedExpertGate.Length
                    && _hasSharedExpertGate[layer];
                if (sharedGateRequired)
                {
                    Tensor sharedGateVec = _ffnGateInpShexpVec != null
                        && layer < _ffnGateInpShexpVec.Length
                        ? _ffnGateInpShexpVec[layer]
                        : null;
                    // The gated shared expert dots the FULL hidden-dim input against the
                    // gate vector; require the vector to be at least hidden-sized so the
                    // device kernel (gate_dim == hiddenDim) matches the host VecDot.
                    if (sharedGateVec != null && sharedGateVec.ElementCount() >= hiddenDim)
                    {
                        sharedGateVecPtr = CudaFusedOps.GetDeviceResidentPtr(sharedGateVec);
                        if (sharedGateVecPtr == IntPtr.Zero)
                        {
                            sharedDown.Dispose();
                            return null;
                        }
                    }
                    else
                    {
                        sharedDown.Dispose();
                        return null; // uncommon geometry -> host path
                    }
                }
            }

            var output = new Tensor(_allocator, DType.Float32, 1, hiddenDim);
            var selected = new Tensor(_allocator, DType.Int32, nUsed);
            var routeW = new Tensor(_allocator, DType.Float32, nUsed);
            var gateOut = new Tensor(_allocator, DType.Float32, nUsed, nFf);
            var upOut = new Tensor(_allocator, DType.Float32, nUsed, nFf);

            // q8_1 activation scratch for the dp4a expert paths (per projection type).
            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gateUpType)
                && IsMoeDp4aDimensionSupported(gateUpType, hiddenDim);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(downType)
                && IsMoeDp4aDimensionSupported(downType, nFf);
            Tensor moeInputQ8 = null, gateOutQ8 = null;
            if (gateUpDp4a)
                moeInputQ8 = new Tensor(_allocator, DType.UInt8, (long)(hiddenDim / 32) * CudaFusedOps.Q81BlockBytes);
            if (downDp4a)
                gateOutQ8 = new Tensor(_allocator, DType.UInt8, (long)nUsed * (nFf / 32) * CudaFusedOps.Q81BlockBytes);

            bool ok = CudaFusedOps.TryMoEExpertFFNDecodeSwiGLU(
                routerLogits, input, output, selected, routeW, gateOut, upOut,
                IntPtr.Zero, // Qwen renormalizes over the selected top-k (no per-expert scale)
                _qwenMoEGatePtrTable[layer], _qwenMoEUpPtrTable[layer], _qwenMoEDownPtrTable[layer],
                gateUpType, downType, _numExperts, nUsed, hiddenDim, nFf,
                sharedDown, sharedGateVecPtr,
                moeInputQ8, gateOutQ8, gateUpDp4a, downDp4a);

            gateOutQ8?.Dispose();
            moeInputQ8?.Dispose();
            upOut.Dispose();
            gateOut.Dispose();
            routeW.Dispose();
            selected.Dispose();
            sharedDown?.Dispose();

            if (!ok)
            {
                output.Dispose();
                return null; // routerLogits still alive for the host fallback
            }

            routerLogits.Dispose();
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return output;
        }

        // CUDA gridDim.y / gridDim.z max; above this the batched prefill kernels
        // cannot launch, so fall back to the host prefill path for that (rare) chunk.
        private const int CudaMaxGridDim = 65535;

        /// <summary>
        /// Batched on-device MoE for a PREFILL chunk (seqLen == numTokens &gt; 1).
        /// Mirrors <see cref="TryCudaMoEForwardOnDevice"/> but batched over tokens:
        /// routing, gate/up, SwiGLU, down and the shared expert all run on device with
        /// no per-expert host gather/scatter (the dominant cost of the old prefill
        /// path). Returns the [numTokens, hiddenDim] output, or null to fall back.
        /// </summary>
        private Tensor TryCudaMoEForwardPrefillOnDevice(Tensor input, Tensor routerLogits, int layer, int seqLen)
        {
            if (seqLen <= 1 || seqLen > CudaMaxGridDim || !CanUseQwenCudaMoEOnDevice(layer))
                return null;

            int hiddenDim = Config.HiddenSize;
            int nFf = _expertFfnLength;
            int nUsed = _numExpertsUsed;
            int gateUpType = _qwenMoEGateUpType[layer];
            int downType = _qwenMoEDownType[layer];

            long t0 = Stopwatch.GetTimestamp();

            // Shared expert over ALL tokens (batched dense matmuls) + on-device gate.
            Tensor sharedDown = null;
            IntPtr sharedGateVecPtr = IntPtr.Zero;
            if (_hasSharedExperts != null && layer < _hasSharedExperts.Length && _hasSharedExperts[layer])
            {
                Tensor sg = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor su = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sg != null && su != null)
                {
                    Ops.SiLUMul(sg, sg, su);
                    sharedDown = LinearForwardCached(sg, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                su?.Dispose();
                sg?.Dispose();

                bool sharedGateRequired = sharedDown != null
                    && _hasSharedExpertGate != null
                    && layer < _hasSharedExpertGate.Length
                    && _hasSharedExpertGate[layer];
                if (sharedGateRequired)
                {
                    Tensor sharedGateVec = _ffnGateInpShexpVec != null
                        && layer < _ffnGateInpShexpVec.Length
                        ? _ffnGateInpShexpVec[layer]
                        : null;
                    if (sharedGateVec != null && sharedGateVec.ElementCount() >= hiddenDim)
                    {
                        sharedGateVecPtr = CudaFusedOps.GetDeviceResidentPtr(sharedGateVec);
                        if (sharedGateVecPtr == IntPtr.Zero)
                        {
                            sharedDown.Dispose();
                            return null;
                        }
                    }
                    else
                    {
                        sharedDown.Dispose();
                        return null;
                    }
                }
            }

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            var selected = new Tensor(_allocator, DType.Int32, (long)seqLen * nUsed);
            var routeW = new Tensor(_allocator, DType.Float32, (long)seqLen * nUsed);
            var gateOut = new Tensor(_allocator, DType.Float32, (long)seqLen * nUsed, nFf);
            var upOut = new Tensor(_allocator, DType.Float32, (long)seqLen * nUsed, nFf);

            bool gateUpDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(gateUpType)
                && IsMoeDp4aDimensionSupported(gateUpType, hiddenDim);
            bool downDp4a = CudaQuantizedOps.IsMoeDp4aEnabledForType(downType)
                && IsMoeDp4aDimensionSupported(downType, nFf);
            Tensor moeInputQ8 = null, gateOutQ8 = null;
            if (gateUpDp4a)
                moeInputQ8 = new Tensor(_allocator, DType.UInt8, (long)seqLen * (hiddenDim / 32) * CudaFusedOps.Q81BlockBytes);
            if (downDp4a)
                gateOutQ8 = new Tensor(_allocator, DType.UInt8, (long)seqLen * nUsed * (nFf / 32) * CudaFusedOps.Q81BlockBytes);

            bool ok = CudaFusedOps.TryMoEExpertFFNPrefillSwiGLU(
                routerLogits, input, output, selected, routeW, gateOut, upOut,
                IntPtr.Zero,
                _qwenMoEGatePtrTable[layer], _qwenMoEUpPtrTable[layer], _qwenMoEDownPtrTable[layer],
                gateUpType, downType, _numExperts, nUsed, hiddenDim, nFf, seqLen,
                sharedDown, sharedGateVecPtr,
                moeInputQ8, gateOutQ8, gateUpDp4a, downDp4a);

            gateOutQ8?.Dispose();
            moeInputQ8?.Dispose();
            upOut.Dispose();
            gateOut.Dispose();
            routeW.Dispose();
            selected.Dispose();
            sharedDown?.Dispose();

            if (!ok)
            {
                output.Dispose();
                return null;
            }

            routerLogits.Dispose();
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            InvalidateTensorDeviceCache(output);
            return output;
        }

        private static bool IsMoeDp4aDimensionSupported(int type, int inDim)
            => inDim > 0 && (type == 2
                ? (inDim & 31) == 0
                : (type == 12 || type == 16 || type == 22) && (inDim & 255) == 0);

        private void FreeQwenCudaMoETables()
        {
            if (_allocator is not CudaAllocator cudaAllocator)
                return;
            FreeQwenMoePtrTable(_qwenMoEGatePtrTable, cudaAllocator);
            FreeQwenMoePtrTable(_qwenMoEUpPtrTable, cudaAllocator);
            FreeQwenMoePtrTable(_qwenMoEDownPtrTable, cudaAllocator);
            _qwenMoEGatePtrTable = null;
            _qwenMoEUpPtrTable = null;
            _qwenMoEDownPtrTable = null;
            _qwenCudaMoEUsable = false;
        }

        private static void FreeQwenMoePtrTable(IntPtr[] tables, CudaAllocator allocator)
        {
            if (tables == null)
                return;
            for (int i = 0; i < tables.Length; i++)
            {
                if (tables[i] != IntPtr.Zero)
                {
                    CudaQuantizedOps.FreeDeviceBuffer(allocator, tables[i]);
                    tables[i] = IntPtr.Zero;
                }
            }
        }
    }
}
