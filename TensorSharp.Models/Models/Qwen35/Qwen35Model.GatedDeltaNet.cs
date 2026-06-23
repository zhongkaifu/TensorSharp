// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// Qwen35Model.GatedDeltaNet.cs
//
// Partial of <see cref="Qwen35Model"/> that owns everything specific to the
// recurrent (GatedDeltaNet) layer:
//
//   * SSM dimension fields parsed from GGUF (`_ssmDInner`, `_headKDim`, etc.)
//   * Per-layer key strings, weight handles and pre-transposed conv weights
//     used by the recurrent block
//   * Recurrent state (`_convState`, `_convStateWriteIdx`, `_deltaStateTensor`)
//   * All scratch buffers used by the chunked prefill and per-token decode
//     paths
//   * The `RecurrentBlock` / `GatedDeltaNet` / `GatedDeltaNetChunkedPrefill`
//     / `GatedDeltaNetStep` implementation (with its small SIMD helpers)
//   * GDN-only timing counters and `PrintGdnTimingStats` / `DisposeGdnState`
//     helpers invoked from the main partial.
//
// The main `Qwen35Model.cs` keeps shared infrastructure (FullAttention, MoE,
// cache management, `Dispose`, `PrintTimingStats` orchestration). Hooks into
// this file are kept narrow: a small set of partial methods + a helper
// (`CacheRecurrentLayerWeights`) called from the existing weight-cache loop.
// ============================================================================
using System;
using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // ====================================================================
        // GDN configuration constants and environment overrides
        // ====================================================================

        // Minimum prefill length at which the fused chunked GatedDeltaNet GGML kernel
        // beats the per-token CPU loop. The chunked kernel pads to a multiple of
        // GdnChunkSize (64) and dispatches once per layer — one graph submit
        // instead of the per-token managed loop's per-layer host/device
        // round-trips. Backend-dependent default (resolved in InitGDNBuffers):
        //   - ggml_cuda: 2 — measured on Qwen3.6-27B IQ2_XXS (RTX 3080 Laptop)
        //     the chunked kernel wins for every multi-token batch (MTP
        //     speculative verify batches of 2–9 tokens dropped from 217 to
        //     174 ms/token end-to-end) because the per-token loop's per-layer
        //     sync cost dwarfs the 64-padding waste on CUDA;
        //   - other backends (Metal/CPU): 6 — compute-bound, so the padding
        //     waste matters and the measured crossover sits near 6.
        // Set GDN_CHUNK_PREFILL_MIN_SEQ_LEN=N to override (e.g. =64 for the
        // old long-prefill-only behavior, =1000000 to disable). -1 = unset.
        private static readonly int GdnChunkedPrefillMinSeqLenEnv = ResolveGdnChunkedPrefillMinSeqLen();

        private static int ResolveGdnChunkedPrefillMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("GDN_CHUNK_PREFILL_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return -1;
        }

        private static readonly bool GdnChunkedPrefillDisabledEnv =
            string.Equals(Environment.GetEnvironmentVariable("GDN_DISABLE_CHUNKED_PREFILL"), "1",
                StringComparison.Ordinal);

        // GDN_VERIFY_CHUNKED=1 enables an inline correctness check that runs
        // BOTH the chunked path and the per-token path on identical starting
        // state and compares the resulting gated outputs and recurrent state
        // tensor element-by-element. Tracking the maximum absolute / relative
        // diff per call lets us catch regressions in the fused GGML kernel
        // without a separate unit-test harness; the per-token path is treated
        // as the ground truth and used downstream so a divergent chunked path
        // never poisons subsequent layers. This roughly doubles the GDN time
        // for a forward pass and is intended for CI / debugging only.
        private static readonly bool GdnVerifyChunkedEnv =
            string.Equals(Environment.GetEnvironmentVariable("GDN_VERIFY_CHUNKED"), "1",
                StringComparison.Ordinal);

        // Direct CUDA runs the packed Qwen3.5/Qwen3.6 GDN recurrence on device
        // instead of downloading the packed projection, conv state, and SSM state
        // for the managed per-token loop. Set TS_CUDA_QWEN35_GDN_NATIVE=0 to
        // force the legacy path for A/B benchmarking.
        private static readonly bool CudaGdnNativeEnabledEnv =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QWEN35_GDN_NATIVE"), "0",
                StringComparison.Ordinal) &&
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_QWEN35_GDN_NATIVE"), "false",
                StringComparison.OrdinalIgnoreCase);

        // Tolerance above which the verification mode logs a warning. A few
        // ULPs of drift are normal because the chunked GGML path executes the
        // recurrence in F32 on the GPU vs. scalar F32 on the CPU and the order
        // of FP additions inside the per-chunk triangular solve is different.
        // Empirically we observe |Δ| ~1e-3 over T=256 tokens of accumulation
        // for Qwen3.6 on Metal; we set the warn threshold a few times above
        // that so a real divergence (e.g. an off-by-one chunk index) rings
        // loudly while normal FP noise is just summarised in the stats line.
        private const float GdnVerifyAbsDiffWarn = 5e-3f;
        private const float GdnVerifyRelDiffWarn = 5e-2f;

        // ====================================================================
        // Per-layer cached key strings for recurrent weights
        // (built in BuildLayerKeys via InitGdnLayerKeyArrays / SetGdnLayerKeys)
        // ====================================================================
        private string[] _ssmInProjKey;
        private string[] _attnQkvRecKey;
        private string[] _attnGateRecKey;
        private string[] _ssmBetaKey;
        private string[] _ssmAlphaKey;
        private string[] _ssmConv1dKey;
        private string[] _ssmDtBiasKey;
        private string[] _ssmAKey;
        private string[] _ssmNormKey;
        private string[] _ssmOutKey;

        // ====================================================================
        // Pre-resolved weight tensors (GDN only)
        // ====================================================================
        private Tensor[] _ssmConv1dW;
        private Tensor[] _ssmDtBiasW;
        private Tensor[] _ssmAW;
        private Tensor[] _ssmNormW;

        private QuantizedWeight[] _ssmInProjQW;
        private Tensor[] _ssmInProjF32;
        private QuantizedWeight[] _attnQkvRecQW;
        private Tensor[] _attnQkvRecF32;
        private QuantizedWeight[] _attnGateRecQW;
        private Tensor[] _attnGateRecF32;
        private QuantizedWeight[] _ssmBetaQW;
        private Tensor[] _ssmBetaF32;
        private QuantizedWeight[] _ssmAlphaQW;
        private Tensor[] _ssmAlphaF32;
        private QuantizedWeight[] _ssmOutQW;
        private Tensor[] _ssmOutF32;

        // ====================================================================
        // SSM dimensions (parsed once in constructor via ParseGdnConfig)
        // ====================================================================
        private int _ssmDInner;   // headVDim * numVHeads
        private int _ssmDState;   // headKDim
        private int _ssmNGroup;   // numKHeads
        private int _ssmDtRank;   // numVHeads
        private int _convKernel;
        private int _headVDim;
        private int _headKDim;
        private int _numVHeads;
        private int _numKHeads;

        // ====================================================================
        // Recurrent state (one entry per recurrent layer; null for attention layers)
        // ====================================================================
        // _convState uses a circular buffer indexed by _convStateWriteIdx[layer] to avoid
        // O(convDim*qkvDim) Array.Copy per token in the recurrent step.
        private float[][] _convState;  // [layer][convChannels * (convKernelSize-1)]
        private int[] _convStateWriteIdx;
        private Tensor[] _cudaGdnConvStateTensor; // [layer]: Tensor[convDim, qkvDim] for direct CUDA
        private Tensor[] _deltaStateTensor; // [layer]: Tensor[numVHeads, headVDim, headKDim]
        private MlxFusedOps.GatedDeltaNetCache[] _mlxGdnCache;

        // [1, packedDim] - reused fused norm + input proj output for GatedDeltaNet decode.
        private Tensor _gdnDecodePackedBuf;

        // ====================================================================
        // Decode-path scratch (per-token GDN step)
        // ====================================================================
        private float[] _gdnQ, _gdnK, _gdnV;
        private float[] _gdnQExp, _gdnKExp;
        private float[] _gdnDelta, _gdnCore;

        // Transposed convolution weights laid out [kernelSize, qkvDim] for cache-friendly SIMD
        // access along the channel dimension while iterating over kernel taps.
        private float[][] _gdnConvWT;
        // Whether to use parallel per-head update in GatedDeltaNetStep.
        private bool _gdnParallelHeads;

        // _gdnConvOutBuf is a managed scratch array (no GGML allocation needed).
        private float[] _gdnConvOutBuf; // [qkvDim]
        // _gdnSiluTempBuf holds sigmoid(x) so SiLU can be done as TensorPrimitives.Sigmoid
        // followed by an element-wise multiply, both of which dispatch SIMD intrinsics.
        // Grows on demand to cover batched SiLU over the whole prefill (seqLen * qkvDim).
        private float[] _gdnSiluTempBuf;
        private Tensor _gdnGatedOutT;   // [1, ssmDInner] (passed to LinearForward, must be a Tensor)

        // ====================================================================
        // Chunked-prefill scratch (Phase 1 conv input + Phase 2 staging)
        // ====================================================================
        //
        // The chunked prefill path processes the full [seqLen, qkvDim] input through the
        // per-channel 1D causal convolution in parallel rather than token-by-token.
        //
        // `_gdnConvExtendedBuf` is laid out as [(convDim + seqLen), qkvDim] row-major.
        // Rows 0..convDim-1 hold the linearised recurrent ring buffer (oldest-first)
        // and rows convDim..convDim+seqLen-1 hold the current step's packed QKV input.
        // With this layout the convolution for token `s` is a pure window read of
        // rows `s..s+convKernel`, so every output row is independent and the loop
        // can be dispatched as a `Parallel.ForEach`.
        private float[] _gdnConvExtendedBuf;
        private int _gdnConvExtendedCapacity;

        // Chunked GatedDeltaNet (prefill) acceleration. The chunked path runs the entire
        // recurrent block as a single fused GGML graph dispatch per layer (via
        // GgmlBasicOps.GatedDeltaNetChunked) which moves L2Norm / mul_mat / triangular solve
        // / RMSNorm onto the GPU backend. CPU-side conv1d runs upstream and writes Q/K/V into
        // these reusable [seqLen, H, D] staging buffers.
        //
        // The path is enabled only when:
        //   * the runtime backend is GGML (Metal/CUDA/CPU)
        //   * headKDim == headVDim (chunked kernel pre-condition)
        //   * seqLen >= _gdnChunkPrefillThreshold
        //   * the kernel has not previously failed for this run (kill switch)
        private const int GdnChunkSize = 64;
        // Conservative placeholder until InitGDNBuffers resolves the
        // backend-dependent default (env override > per-backend value).
        private int _gdnChunkPrefillThreshold = GdnChunkSize;
        private bool _gdnDisableChunkedPrefill = GdnChunkedPrefillDisabledEnv;
        private long _gdnChunkedTicks;       // Total time spent in the chunked path
        private long _gdnPerTokenTicks;      // Total time spent in the per-token path (prefill only)
        private long _gdnCudaNativeTicks;    // Total time spent in the direct CUDA native GDN path
        private int _gdnChunkedCalls;        // Number of times the chunked path has been used
        private int _gdnCudaNativeCalls;     // Number of times direct CUDA native GDN has been used
        private long _gdnChunkedCpuPrepTicks; // Time spent in CPU prep (Conv1D + memcpy + SiLU)
        private long _gdnChunkedKernelTicks; // Time spent in the GGML kernel call (incl. sync + download)

        // Verification telemetry (only populated when GDN_VERIFY_CHUNKED=1).
        // Captures the worst-case absolute and relative divergence we observed
        // between the chunked and per-token paths over all verified calls so
        // PrintGdnTimingStats can surface a single summary line.
        private int _gdnVerifyCalls;
        private float _gdnVerifyMaxOutAbs;
        private float _gdnVerifyMaxOutRel;
        private float _gdnVerifyMaxStateAbs;
        private float _gdnVerifyMaxStateRel;
        private int _gdnVerifyWarnings;

        // Reusable staging buffers for the chunked prefill path. These are sized for the
        // largest seqLen we have seen so far; subsequent calls reuse the same memory and
        // build a transient sub-view at the actual seqLen, avoiding (numLayers) allocator
        // round-trips per forward pass.
        private Tensor _gdnChunkedQBuf;
        private Tensor _gdnChunkedKBuf;
        private Tensor _gdnChunkedVBuf;
        private Tensor _gdnChunkedZBuf;
        private Tensor _gdnChunkedAlphaBuf;
        private Tensor _gdnChunkedBetaBuf;
        private int _gdnChunkedBufCapacity; // SeqLen capacity covered by the staging buffers
        private int _gdnPerTokenCalls;       // Number of times the per-token path has been used (excluding decode)

        // ====================================================================
        // Constructor / lifecycle hooks (invoked from main Qwen35Model.cs)
        // ====================================================================

        /// <summary>
        /// Parse the SSM-related dimensions from the GGUF metadata. Called from the
        /// main constructor immediately after the base config is parsed.
        /// </summary>
        private void ParseGdnConfig(string arch)
        {
            _ssmDInner = (int)_gguf.GetUint32($"{arch}.ssm.inner_size");
            _ssmDState = (int)_gguf.GetUint32($"{arch}.ssm.state_size");
            _ssmNGroup = (int)_gguf.GetUint32($"{arch}.ssm.group_count");
            _ssmDtRank = (int)_gguf.GetUint32($"{arch}.ssm.time_step_rank");
            _convKernel = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");

            _numVHeads = _ssmDtRank;
            _numKHeads = _ssmNGroup;
            _headVDim = _ssmDInner / _numVHeads;
            _headKDim = _ssmDState;
        }

        /// <summary>
        /// Allocate the per-layer key arrays for recurrent weights so the BuildLayerKeys
        /// loop in the main partial can fill them. Called from BuildLayerKeys.
        /// </summary>
        private void InitGdnLayerKeyArrays(int n)
        {
            _ssmInProjKey = new string[n];
            _attnQkvRecKey = new string[n];
            _attnGateRecKey = new string[n];
            _ssmBetaKey = new string[n];
            _ssmAlphaKey = new string[n];
            _ssmConv1dKey = new string[n];
            _ssmDtBiasKey = new string[n];
            _ssmAKey = new string[n];
            _ssmNormKey = new string[n];
            _ssmOutKey = new string[n];
        }

        /// <summary>
        /// Fill all the GDN-specific layer key strings for one layer. Called from
        /// the BuildLayerKeys loop in the main partial.
        /// </summary>
        private void SetGdnLayerKeys(int l, string p)
        {
            _ssmInProjKey[l] = p + "ssm_in_proj.weight";
            _attnQkvRecKey[l] = p + "attn_qkv.weight";
            _attnGateRecKey[l] = p + "attn_gate.weight";
            _ssmBetaKey[l] = p + "ssm_beta.weight";
            _ssmAlphaKey[l] = p + "ssm_alpha.weight";
            _ssmConv1dKey[l] = p + "ssm_conv1d.weight";
            _ssmDtBiasKey[l] = p + "ssm_dt.bias";
            _ssmAKey[l] = p + "ssm_a";
            _ssmNormKey[l] = p + "ssm_norm.weight";
            _ssmOutKey[l] = p + "ssm_out.weight";
        }

        /// <summary>
        /// Allocate the per-layer recurrent weight arrays (matching the array shapes
        /// used by InitGdnLayerKeyArrays). Called once from CacheRecurrentWeights.
        /// </summary>
        private void InitGdnWeightArrays(int n)
        {
            _ssmConv1dW = new Tensor[n];
            _ssmDtBiasW = new Tensor[n];
            _ssmAW = new Tensor[n];
            _ssmNormW = new Tensor[n];
            _gdnConvWT = new float[n][];

            _ssmInProjQW = new QuantizedWeight[n];
            _ssmInProjF32 = new Tensor[n];
            _attnQkvRecQW = new QuantizedWeight[n];
            _attnQkvRecF32 = new Tensor[n];
            _attnGateRecQW = new QuantizedWeight[n];
            _attnGateRecF32 = new Tensor[n];
            _ssmBetaQW = new QuantizedWeight[n];
            _ssmBetaF32 = new Tensor[n];
            _ssmAlphaQW = new QuantizedWeight[n];
            _ssmAlphaF32 = new Tensor[n];
            _ssmOutQW = new QuantizedWeight[n];
            _ssmOutF32 = new Tensor[n];
        }

        /// <summary>
        /// Resolve and cache all GDN-specific weight references for one recurrent
        /// layer, plus the SIMD-friendly transposed conv1d weight. Called from
        /// CacheRecurrentWeights for layers where _isRecurrent[l] is true.
        /// </summary>
        private unsafe void CacheRecurrentLayerWeights(int l, int qkvDim)
        {
            _weights.TryGetValue(_ssmConv1dKey[l], out _ssmConv1dW[l]);
            _weights.TryGetValue(_ssmDtBiasKey[l], out _ssmDtBiasW[l]);
            _weights.TryGetValue(_ssmAKey[l], out _ssmAW[l]);
            _weights.TryGetValue(_ssmNormKey[l], out _ssmNormW[l]);

            _quantWeights.TryGetValue(_ssmInProjKey[l], out _ssmInProjQW[l]);
            _weights.TryGetValue(_ssmInProjKey[l], out _ssmInProjF32[l]);
            _quantWeights.TryGetValue(_attnQkvRecKey[l], out _attnQkvRecQW[l]);
            _weights.TryGetValue(_attnQkvRecKey[l], out _attnQkvRecF32[l]);
            _quantWeights.TryGetValue(_attnGateRecKey[l], out _attnGateRecQW[l]);
            _weights.TryGetValue(_attnGateRecKey[l], out _attnGateRecF32[l]);
            _quantWeights.TryGetValue(_ssmBetaKey[l], out _ssmBetaQW[l]);
            _weights.TryGetValue(_ssmBetaKey[l], out _ssmBetaF32[l]);
            _quantWeights.TryGetValue(_ssmAlphaKey[l], out _ssmAlphaQW[l]);
            _weights.TryGetValue(_ssmAlphaKey[l], out _ssmAlphaF32[l]);
            _quantWeights.TryGetValue(_ssmOutKey[l], out _ssmOutQW[l]);
            _weights.TryGetValue(_ssmOutKey[l], out _ssmOutF32[l]);

            if (_ssmConv1dW[l] != null)
            {
                // Stored as [qkvDim, kernelSize] (each row = filter for one channel).
                // Transpose to [kernelSize, qkvDim] so that for a fixed kernel tap ki we
                // access a contiguous block of channel weights, enabling SIMD across ch.
                float* src = GetFloatPtr(_ssmConv1dW[l]);
                var wt = new float[_convKernel * qkvDim];
                for (int ch = 0; ch < qkvDim; ch++)
                    for (int ki = 0; ki < _convKernel; ki++)
                        wt[ki * qkvDim + ch] = src[ch * _convKernel + ki];
                _gdnConvWT[l] = wt;
            }
        }

        /// <summary>
        /// Allocate the per-layer recurrent state. Called from InitCaches for layers
        /// where _isRecurrent[l] is true.
        /// </summary>
        private void InitGdnLayerCache(int l, int qkvDim)
        {
            int convDim = _convKernel - 1;
            _convState[l] = new float[convDim * qkvDim];
            _convStateWriteIdx[l] = 0;
            if (_cudaGdnConvStateTensor != null && convDim > 0)
            {
                _cudaGdnConvStateTensor[l] = new Tensor(_allocator, DType.Float32, convDim, qkvDim);
                Ops.Fill(_cudaGdnConvStateTensor[l], 0);
            }
            _deltaStateTensor[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
            Ops.Fill(_deltaStateTensor[l], 0);
            if (_mlxGdnCache != null)
                _mlxGdnCache[l] = new MlxFusedOps.GatedDeltaNetCache();
        }

        /// <summary>
        /// Allocate the per-recurrent-layer cache arrays so InitCaches can fill them
        /// in its layer loop. Called from InitCaches.
        /// </summary>
        private void InitGdnCacheArrays(int numLayers)
        {
            _convState = new float[numLayers][];
            _convStateWriteIdx = new int[numLayers];
            _cudaGdnConvStateTensor = _backend == BackendType.Cuda
                ? new Tensor[numLayers]
                : null;
            _deltaStateTensor = new Tensor[numLayers];
            _mlxGdnCache = _backend == BackendType.Mlx
                ? new MlxFusedOps.GatedDeltaNetCache[numLayers]
                : null;
        }

        /// <summary>
        /// Reset the per-layer recurrent state. Called from ResetKVCache.
        /// </summary>
        private void ResetGdnLayerCache(int l)
        {
            Array.Clear(_convState[l]);
            _convStateWriteIdx[l] = 0;
            if (_cudaGdnConvStateTensor?[l] != null)
                Ops.Fill(_cudaGdnConvStateTensor[l], 0);
            Ops.Fill(_deltaStateTensor[l], 0);
            _mlxGdnCache?[l]?.Reset();
        }

        private void SyncCudaGdnConvStateToHost(int layer)
        {
            Tensor conv = _cudaGdnConvStateTensor?[layer];
            if (conv == null || _convState?[layer] == null)
                return;

            float[] values = conv.GetElementsAsFloat((int)conv.ElementCount());
            Array.Copy(values, _convState[layer], values.Length);
        }

        private void SyncCudaGdnConvStateFromHost(int layer)
        {
            Tensor conv = _cudaGdnConvStateTensor?[layer];
            if (conv == null || _convState?[layer] == null)
                return;

            conv.SetElementsAsFloat(_convState[layer]);
        }

        /// <summary>
        /// Pre-allocate the small pinned scratch buffers used by the per-token GDN
        /// decode step. Called from the constructor.
        /// </summary>
        private void InitGDNBuffers()
        {
            // Backend-dependent chunked-kernel threshold (see the comment on
            // GdnChunkedPrefillMinSeqLenEnv). The env override, when set,
            // always wins; _backend is only known here (instance init), not
            // at static-field time.
            if (GdnChunkedPrefillMinSeqLenEnv > 0)
                _gdnChunkPrefillThreshold = GdnChunkedPrefillMinSeqLenEnv;
            else
                _gdnChunkPrefillThreshold = _backend == BackendType.GgmlCuda ? 2 : 6;

            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            _gdnQ = new float[qkDim];
            _gdnK = new float[qkDim];
            _gdnV = new float[vDim];
            _gdnQExp = new float[_headKDim * _numVHeads];
            _gdnKExp = new float[_headKDim * _numVHeads];
            _gdnDelta = new float[vDim];
            _gdnCore = new float[vDim];

            _gdnConvOutBuf = new float[qkvDim];
            _gdnSiluTempBuf = new float[qkvDim];
            _gdnGatedOutT = new Tensor(_allocator, DType.Float32, 1, _ssmDInner);

            // Pre-allocated fused norm + input projection output for GatedDeltaNet decode.
            // Shape matches the packed projection used in the recurrent block hot path.
            int packedDim = qkvDim + (_headVDim * _numVHeads) + _numVHeads * 2;
            if (packedDim > 0)
                _gdnDecodePackedBuf = new Tensor(_allocator, DType.Float32, 1, packedDim);

            // Heuristic: only parallelize per-head GDN work for models with many V-heads
            // (where the per-head work amortizes the parallel dispatch overhead).
            _gdnParallelHeads = _numVHeads >= 16 && Environment.ProcessorCount > 1;
        }

        /// <summary>
        /// Dispose all GDN-owned tensors and tensor caches. Called from the main
        /// Dispose override.
        /// </summary>
        private void DisposeGdnState()
        {
            if (_cudaGdnConvStateTensor != null)
                foreach (var t in _cudaGdnConvStateTensor) t?.Dispose();
            if (_deltaStateTensor != null)
                foreach (var t in _deltaStateTensor) t?.Dispose();
            if (_mlxGdnCache != null)
                foreach (var cache in _mlxGdnCache) cache?.Dispose();
            if (_q35GdnSlotMlxCache != null)
            {
                foreach (var layerSlots in _q35GdnSlotMlxCache)
                {
                    if (layerSlots == null) continue;
                    foreach (var cache in layerSlots) cache?.Dispose();
                }
            }
            if (_q35GdnSlotSsmTensor != null)
            {
                foreach (var layerSlots in _q35GdnSlotSsmTensor)
                {
                    if (layerSlots == null) continue;
                    foreach (var t in layerSlots) t?.Dispose();
                }
            }

            if (_mtpGdnConvDevSnap != null)
                foreach (var t in _mtpGdnConvDevSnap) t?.Dispose();
            if (_mtpGdnDeltaDevSnap != null)
                foreach (var t in _mtpGdnDeltaDevSnap) t?.Dispose();

            _gdnGatedOutT?.Dispose();
            _gdnChunkedQBuf?.Dispose();
            _gdnChunkedKBuf?.Dispose();
            _gdnChunkedVBuf?.Dispose();
            _gdnChunkedZBuf?.Dispose();
            _gdnChunkedAlphaBuf?.Dispose();
            _gdnChunkedBetaBuf?.Dispose();
            _gdnDecodePackedBuf?.Dispose();
        }

        /// <summary>
        /// Print GDN-specific timing stats. Called from PrintTimingStats in the main
        /// partial after the base / shared stats have been printed.
        /// </summary>
        private void PrintGdnTimingStats()
        {
            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double chunkedMs = _gdnChunkedTicks * msPerTick;
            double perTokenMs = _gdnPerTokenTicks * msPerTick;
            double cudaNativeMs = _gdnCudaNativeTicks * msPerTick;

            if (_gdnChunkedCalls == 0 && _gdnPerTokenCalls == 0 && _gdnCudaNativeCalls == 0)
                return;

            double cpuPrepMs = _gdnChunkedCpuPrepTicks * msPerTick;
            double kernelMs = _gdnChunkedKernelTicks * msPerTick;

            Console.WriteLine($"  GatedDeltaNet:");
            Console.WriteLine($"    chunked path:   {_gdnChunkedCalls} calls, {chunkedMs:F0} ms total" +
                (_gdnChunkedCalls > 0 ? $", {chunkedMs / _gdnChunkedCalls:F2} ms/call" : ""));
            if (_gdnChunkedCalls > 0)
            {
                Console.WriteLine($"      cpu prep:     {cpuPrepMs:F0} ms total, {cpuPrepMs / _gdnChunkedCalls:F2} ms/call");
                Console.WriteLine($"      ggml kernel:  {kernelMs:F0} ms total, {kernelMs / _gdnChunkedCalls:F2} ms/call");
            }
            Console.WriteLine($"    cuda native:    {_gdnCudaNativeCalls} calls, {cudaNativeMs:F0} ms total" +
                (_gdnCudaNativeCalls > 0 ? $", {cudaNativeMs / _gdnCudaNativeCalls:F2} ms/call" : ""));
            Console.WriteLine($"    per-token path: {_gdnPerTokenCalls} prefill calls, {perTokenMs:F0} ms total" +
                (_gdnPerTokenCalls > 0 ? $", {perTokenMs / _gdnPerTokenCalls:F2} ms/call" : ""));
            if (_gdnDisableChunkedPrefill)
                Console.WriteLine($"    (chunked path disabled at runtime)");
            else
                Console.WriteLine($"    (chunked threshold: seqLen >= {_gdnChunkPrefillThreshold}, chunkSize {GdnChunkSize})");

            if (_gdnVerifyCalls > 0)
            {
                Console.WriteLine($"    verification: {_gdnVerifyCalls} calls, " +
                    $"max output |Δ|={_gdnVerifyMaxOutAbs:E2} (rel {_gdnVerifyMaxOutRel:E2}), " +
                    $"max state |Δ|={_gdnVerifyMaxStateAbs:E2} (rel {_gdnVerifyMaxStateRel:E2}), " +
                    $"warnings={_gdnVerifyWarnings}");
            }
        }

        /// <summary>
        /// Reset all GDN timing counters. Called from ResetKVCache after the shared
        /// counters are reset.
        /// </summary>
        private void ResetGdnTimingCounters()
        {
            _gdnChunkedTicks = 0;
            _gdnPerTokenTicks = 0;
            _gdnCudaNativeTicks = 0;
            _gdnChunkedCalls = 0;
            _gdnCudaNativeCalls = 0;
            _gdnChunkedCpuPrepTicks = 0;
            _gdnChunkedKernelTicks = 0;
            _gdnPerTokenCalls = 0;
            _gdnVerifyCalls = 0;
            _gdnVerifyMaxOutAbs = 0f;
            _gdnVerifyMaxOutRel = 0f;
            _gdnVerifyMaxStateAbs = 0f;
            _gdnVerifyMaxStateRel = 0f;
            _gdnVerifyWarnings = 0;
        }

        // ====================================================================
        // Recurrent block (called from the main Forward loop)
        // ====================================================================

        /// <summary>
        /// GatedDeltaNet recurrent block: SSM conv1d -> gated delta net -> norm + gate -> output.
        /// Both decode and prefill use the same recurrent core. Prefill batches the large
        /// input/output projections across the whole chunk, then walks the recurrent state
        /// token-by-token in CPU memory.
        /// </summary>
        private Tensor RecurrentBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            bool isMoeLayer = _isMoeLayer != null && _isMoeLayer[layer];
            bool profilePrefill = _profilePrefillStages && seqLen > 1;

            // ---- Path A: Fused dense FFN (non-MoE layers) ----
            bool canFuseDenseFFN = IsGgmlBackend && !isMoeLayer
                && _ssmOutQW[layer] != null && _postAttnNormW[layer] != null
                && _ffnGateUpQW[layer] != null && _ffnDownQW[layer] != null;

            if (canFuseDenseFFN)
            {
                Tensor gated = GatedDeltaNet(hidden, _attnNormW[layer], layer, seqLen,
                    residual: null, skipOutputProj: true);
                if (gated != null)
                {
                    long ffnStart = profilePrefill ? Stopwatch.GetTimestamp() : 0;
                    int intermSize = Config.IntermediateSize;
                    int halfDim = intermSize > 0 ? intermSize : (int)(_ffnGateUpQW[layer].Ne1 / 2);

                    if (halfDim > 0 && hidden.DimensionCount == 2 && gated.DimensionCount == 2
                        && hidden.Sizes[0] == gated.Sizes[0])
                    {
                        try
                        {
                            long t0 = Stopwatch.GetTimestamp();
                            GgmlBasicOps.FusedOutProjFFN(hidden, gated,
                                _ssmOutQW[layer].CacheKey, _ssmOutQW[layer].GgmlType,
                                _ssmOutQW[layer].Ne0, _ssmOutQW[layer].Ne1, _ssmOutQW[layer].RawBytes,
                                _postAttnNormW[layer], Config.Eps,
                                _ffnGateUpQW[layer].CacheKey, _ffnGateUpQW[layer].GgmlType,
                                _ffnGateUpQW[layer].Ne0, _ffnGateUpQW[layer].Ne1, _ffnGateUpQW[layer].RawBytes,
                                _ffnDownQW[layer].CacheKey, _ffnDownQW[layer].GgmlType,
                                _ffnDownQW[layer].Ne0, _ffnDownQW[layer].Ne1, _ffnDownQW[layer].RawBytes,
                                halfDim);
                            _linearTicks += Stopwatch.GetTimestamp() - t0;
                            gated.Dispose();
                            if (profilePrefill) _prefillRecFfnTicks += Stopwatch.GetTimestamp() - ffnStart;
                            return hidden;
                        }
                        catch { /* fall through */ }
                    }
                    // Fallback: do output proj + residual, then standard FFN.
                    if (TryLinearAddInto(hidden, gated, _ssmOutQW[layer]))
                        gated.Dispose();
                    else
                    {
                        var o = LinearForwardCached(gated, _ssmOutQW[layer], _ssmOutF32[layer]);
                        gated.Dispose();
                        Ops.Add(hidden, hidden, o);
                        o.Dispose();
                    }
                    // Fall through to standard dense FFN below.
                }
                else
                {
                    // GDN returned null (fused residual add already done).
                }
                long ffnStart2 = profilePrefill ? Stopwatch.GetTimestamp() : 0;
                Tensor ffnOut = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
                if (ffnOut != null) { Ops.Add(hidden, hidden, ffnOut); ffnOut.Dispose(); }
                if (profilePrefill) _prefillRecFfnTicks += Stopwatch.GetTimestamp() - ffnStart2;
                return hidden;
            }

            // ---- Path B: Fused MoE router (MoE decode) ----
            bool canFuseMoeRouter = IsGgmlBackend && isMoeLayer && seqLen == 1
                && _ssmOutQW[layer] != null && _postAttnNormW[layer] != null
                && (_ffnGateInpQW[layer] != null || _ffnGateInpF32[layer] != null)
                && _moeTokenInput != null && _moeTokenInput.Sizes[1] == Config.HiddenSize;

            if (canFuseMoeRouter)
            {
                Tensor gated = GatedDeltaNet(hidden, _attnNormW[layer], layer, seqLen,
                    residual: null, skipOutputProj: true);
                if (gated != null && hidden.DimensionCount == 2 && gated.DimensionCount == 2)
                {
                    using var routerBuf = new Tensor(_allocator, DType.Float32, 1, _numExperts);
                    try
                    {
                        // Router weight can be quantized or F32; resolve the appropriate pointer/type.
                        IntPtr rtPtr; int rtType; long rtNe0, rtNe1, rtBytes;
                        if (_ffnGateInpQW[layer] != null)
                        {
                            var qw = _ffnGateInpQW[layer];
                            rtPtr = qw.CacheKey; rtType = qw.GgmlType;
                            rtNe0 = qw.Ne0; rtNe1 = qw.Ne1; rtBytes = qw.RawBytes;
                        }
                        else
                        {
                            var fw = _ffnGateInpF32[layer];
                            unsafe { rtPtr = (IntPtr)GetFloatPtr(fw); }
                            rtType = 0; // GGML_TYPE_F32
                            rtNe0 = fw.Sizes[fw.DimensionCount - 1];
                            rtNe1 = fw.Sizes[0];
                            rtBytes = fw.ElementCount() * 4;
                        }

                        long t0r = Stopwatch.GetTimestamp();
                        GgmlBasicOps.FusedOutProjNormRouter(hidden, gated,
                            _ssmOutQW[layer].CacheKey, _ssmOutQW[layer].GgmlType,
                            _ssmOutQW[layer].Ne0, _ssmOutQW[layer].Ne1, _ssmOutQW[layer].RawBytes,
                            _postAttnNormW[layer], Config.Eps,
                            _moeTokenInput,
                            rtPtr, rtType, rtNe0, rtNe1, rtBytes,
                            routerBuf);
                        _linearTicks += Stopwatch.GetTimestamp() - t0r;
                        gated.Dispose();

                        InvalidateTensorDeviceCache(_moeTokenInput);
                        InvalidateTensorDeviceCache(hidden);
                        InvalidateTensorDeviceCache(routerBuf);

                        if (TryMoEResidualDecodeWithRouter(hidden, _moeTokenInput, routerBuf, layer))
                            return hidden;
                    }
                    catch
                    {
                        // Fallback: do output proj separately, fall through to standard MoE.
                        if (gated != null)
                        {
                            if (TryLinearAddInto(hidden, gated, _ssmOutQW[layer]))
                                gated.Dispose();
                            else
                            {
                                var o = LinearForwardCached(gated, _ssmOutQW[layer], _ssmOutF32[layer]);
                                gated.Dispose();
                                Ops.Add(hidden, hidden, o);
                                o.Dispose();
                            }
                        }
                    }
                }
                else
                {
                    // gated was null or shape mismatch - GDN already did output proj + residual
                    if (gated != null) { Ops.Add(hidden, hidden, gated); gated.Dispose(); }
                }
                // Fall through to standard MoE path.
            }

            // ---- Path C: Standard path (no fusion) ----
            if (!canFuseDenseFFN && !canFuseMoeRouter)
            {
                Tensor attnOut = GatedDeltaNet(hidden, _attnNormW[layer], layer, seqLen, residual: hidden);
                if (attnOut != null) { Ops.Add(hidden, hidden, attnOut); attnOut.Dispose(); }
            }

            long ffnStartC = profilePrefill ? Stopwatch.GetTimestamp() : 0;
            Tensor ffnOutC;
            if (isMoeLayer)
            {
                Tensor normed2; bool ownsNormed2 = true;
                if (seqLen == 1 && IsGgmlBackend && _moeTokenInput != null
                    && _moeTokenInput.Sizes[1] == Config.HiddenSize && _postAttnNormW[layer] != null)
                {
                    RMSNormToBufferCpu(_moeTokenInput, hidden, _postAttnNormW[layer], Config.HiddenSize, Config.Eps);
                    normed2 = _moeTokenInput; ownsNormed2 = false;
                }
                else normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);

                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                    ffnOutC = null;
                else
                    ffnOutC = MoEForward(normed2, layer, seqLen);
                if (ownsNormed2) normed2.Dispose();
            }
            else
            {
                ffnOutC = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
            }
            if (ffnOutC != null) { Ops.Add(hidden, hidden, ffnOutC); ffnOutC.Dispose(); }
            if (profilePrefill) _prefillRecFfnTicks += Stopwatch.GetTimestamp() - ffnStartC;
            return hidden;
        }

        // ====================================================================
        // Full-model fused decode (ggml_cuda). Runs the whole hybrid transformer
        // (full-attention + GatedDeltaNet recurrent layers + per-layer dense FFN)
        // as ONE GGML graph per token via TSGgml_Qwen35ModelDecode, collapsing the
        // ~120-400 per-op kernel dispatches/token (the WDDM per-submit tax that
        // dominates decode) into a single graph_compute. Falls back to the per-op
        // layer loop on any unsupported shape. Default ON; TS_QWEN35_FULL_DECODE=0
        // disables. Dense models only for now (MoE FFN routes to per-op).
        // ====================================================================
        private static readonly bool _fullDecodeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_FULL_DECODE"), "0", StringComparison.Ordinal);

        private Qwen35LayerDecodeArgs[] _fdLayers;
        private IntPtr _fdConvScratch;    // unmanaged [numGdnLayers * convDim * qkvDim], STABLE addr
                                          // (cache key for the device-resident conv state) ggml [time,channel]
        private int[] _fdGdnSlot;         // layer -> index into _fdConvScratch blocks (-1 for attn layers)
        private bool _fdUnsupported;      // latched: gating failed once, never retry
        private bool _fdDiagPrinted;
        private bool _fdStateResident;    // GDN conv/delta state currently device-resident
        private bool _fdSpecSessionActive; // MTP/spec in use this session -> disable fused decode

        // Called whenever host-side GDN state becomes the source of truth again
        // (reset / prefill / per-op recurrent execution), forcing the next fused
        // decode to re-seed the device-resident state from the host buffers.
        internal void InvalidateFullDecodeState()
        {
            _fdStateResident = false;
            // The batched-fused device KV pool is seeded from the host paged pool;
            // any state reset/rebuild must force a re-seed before the next fused decode.
            _bfdPoolSeeded = false;
            if (_backend == BackendType.GgmlCuda)
                GgmlBasicOps.Qwen35ResetBatchedDecodeCache();
            // The persistent (CUDA-graph-captured) decode graph pins the GDN
            // conv/delta device-buffer addresses; a re-seed can move them, so drop
            // the cached graph too. No-op when persist mode is off.
            //
            // NOTE: the fused VERIFY cache is NOT reset here. InvalidateFullDecodeState
            // runs on every EnterSpecSession (i.e. every spec step), but the verify
            // cache only goes stale when the KV/GDN device TENSORS are reallocated
            // (ResetKVCache / EnsureCacheCapacity grow) — it uploads fresh GDN state
            // each call, so a per-step reset would defeat the whole cache. Those two
            // call sites reset it explicitly via InvalidateVerifyCache().
            if (_backend == BackendType.GgmlCuda)
                GgmlBasicOps.Qwen35ResetDecodeCache();
        }

        /// <summary>Drop the persistent fused-verify graph cache (it pins the KV +
        /// GDN-state device buffers). Call only when those buffers may have moved
        /// (KV reset / capacity grow), NOT on the per-step spec latch.</summary>
        internal void InvalidateVerifyCache()
        {
            if (_backend == BackendType.GgmlCuda)
                GgmlBasicOps.Qwen35ResetVerifyCache();
        }

        // MTP/spec interleaves the per-op verify/draft (which read/write the HOST
        // GDN state) with plain decode steps; the fused decode keeps state
        // device-resident, so once any spec op runs this session the fused decode
        // must stay off (host state is the single source of truth). Latched until
        // the next ResetKVCache. Spec is net-negative for this model anyway.
        internal void EnterSpecSession() { _fdSpecSessionActive = true; InvalidateFullDecodeState(); }

        // Resolve a linear-projection weight to (ptr, ggml-type, ne0, ne1, bytes)
        // from EITHER its quantized form or its F32 form (small projections such as
        // ssm_beta / ssm_alpha are stored F32). F32 weights are [out, in] tensors,
        // mapped to ggml ne0 = in (last dim), ne1 = out (dim 0).
        private static unsafe (IntPtr ptr, int type, long ne0, long ne1, long bytes)
            ResolveW(QuantizedWeight qw, Tensor f32)
        {
            if (qw != null)
                return (qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            long ne0 = f32.Sizes[f32.DimensionCount - 1];
            long ne1 = f32.Sizes[0];
            return ((IntPtr)GetFloatPtr(f32), 0, ne0, ne1, f32.ElementCount() * 4L);
        }

        internal unsafe bool TryFullModelDecode(Tensor hidden, int position, float[] logitsOut)
        {
            if (logitsOut == null || logitsOut.Length < Config.VocabSize)
                return false;
            // Fold final-norm + lm_head into the graph requires both present.
            if ((_lmHeadQW == null && _lmHeadF32 == null) || _finalNormW == null)
                return false;
            if (!_fullDecodeEnabled || _fdUnsupported || _fdSpecSessionActive || _backend != BackendType.GgmlCuda)
                return false;
            if (hidden == null || hidden.DimensionCount != 2 || hidden.Sizes[0] != 1
                || hidden.ElementType != DType.Float32)
                return false;

            int n = Config.NumLayers;
            int headDim = Config.HeadDim;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            if (convDim <= 0)
                return false;

            // --- one-time gate: every layer must have its required weights/state ---
            if (_fdLayers == null)
            {
                static bool HasW(QuantizedWeight q, Tensor f) => q != null || f != null;
                for (int l = 0; l < n; l++)
                {
                    bool isMoeL = _isMoeLayer != null && _isMoeLayer[l];
                    bool ffnOk = isMoeL
                        ? ((_ffnGateInpQW[l] != null || _ffnGateInpF32[l] != null)
                            && _layerStackedGate[l] != null && _layerStackedUp[l] != null && _layerStackedDown[l] != null
                            && HasW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]) && HasW(_ffnUpShexpQW[l], _ffnUpShexpF32[l])
                            && HasW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]) && _ffnGateInpShexpVec[l] != null)
                        : (HasW(_ffnGateUpQW[l], _ffnGateUpF32[l]) && HasW(_ffnDownQW[l], _ffnDownF32[l]));
                    bool ok = _attnNormW[l] != null && _postAttnNormW[l] != null && ffnOk;
                    if (ok && !_isRecurrent[l])
                        ok = (HasW(_attnQkvQW[l], _attnQkvF32[l])
                              || (HasW(_attnQQW[l], _attnQF32[l]) && HasW(_attnKQW[l], _attnKF32[l]) && HasW(_attnVQW[l], _attnVF32[l])))
                            && HasW(_attnOutputQW[l], _attnOutputF32[l])
                            && _attnQNormW[l] != null && _attnKNormW[l] != null
                            && _kvCacheK[l] != null && _kvCacheV[l] != null
                            && (_kvCacheK[l].ElementType == DType.Float32 || _kvCacheK[l].ElementType == DType.Float16);
                    if (ok && _isRecurrent[l])
                        ok = HasW(_attnQkvRecQW[l], _attnQkvRecF32[l]) && HasW(_attnGateRecQW[l], _attnGateRecF32[l])
                            && HasW(_ssmBetaQW[l], _ssmBetaF32[l]) && HasW(_ssmAlphaQW[l], _ssmAlphaF32[l])
                            && _ssmConv1dW[l] != null && _ssmDtBiasW[l] != null && _ssmAW[l] != null
                            && _ssmNormW[l] != null && HasW(_ssmOutQW[l], _ssmOutF32[l])
                            && _deltaStateTensor[l] != null && _convState[l] != null;
                    if (!ok)
                    {
                        _fdUnsupported = true; return false;
                    }
                }
                int gdnCount = 0;
                _fdGdnSlot = new int[n];
                for (int l = 0; l < n; l++)
                    _fdGdnSlot[l] = _isRecurrent[l] ? gdnCount++ : -1;
                _fdConvScratch = Marshal.AllocHGlobal(Math.Max(1, gdnCount) * convDim * qkvDim * sizeof(float));
                _fdLayers = new Qwen35LayerDecodeArgs[n];
            }

            int cacheSize = 0;
            int kvCacheType = 0;
            for (int l = 0; l < n; l++)
            {
                if (!_isRecurrent[l])
                {
                    cacheSize = (int)_kvCacheK[l].Sizes[1];
                    kvCacheType = _kvCacheK[l].ElementType == DType.Float16 ? 1 : 0;
                    break;
                }
            }
            if (cacheSize <= 0 || position >= cacheSize)
                return false;

            int structBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>();
            int convBlock = convDim * qkvDim;

            {
                float* convBase = (float*)_fdConvScratch;
                for (int l = 0; l < n; l++)
                {
                    var a = default(Qwen35LayerDecodeArgs);
                    a.StructBytes = structBytes;
                    a.AttnNormW = (IntPtr)GetFloatPtr(_attnNormW[l]);
                    a.PostAttnNormW = (IntPtr)GetFloatPtr(_postAttnNormW[l]);
                    // FFN: dense SwiGLU or MoE
                    bool isMoe = _isMoeLayer != null && _isMoeLayer[l];
                    a.IsMoe = isMoe ? 1 : 0;
                    if (!isMoe)
                    {
                        var gu = ResolveW(_ffnGateUpQW[l], _ffnGateUpF32[l]);
                        var dn = ResolveW(_ffnDownQW[l], _ffnDownF32[l]);
                        a.GuW = gu.ptr; a.GuType = gu.type; a.GuNe0 = gu.ne0; a.GuNe1 = gu.ne1; a.GuBytes = gu.bytes;
                        a.DownW = dn.ptr; a.DownType = dn.type; a.DownNe0 = dn.ne0; a.DownNe1 = dn.ne1; a.DownBytes = dn.bytes;
                        a.FfDense = (int)(gu.ne1 / 2);
                    }
                    else
                    {
                        var gi = ResolveW(_ffnGateInpQW[l], _ffnGateInpF32[l]);
                        a.GateInpW = gi.ptr; a.GateInpType = gi.type; a.GateInpNe0 = gi.ne0; a.GateInpNe1 = gi.ne1; a.GateInpBytes = gi.bytes;
                        var sg = _layerStackedGate[l]; var su = _layerStackedUp[l]; var sd = _layerStackedDown[l];
                        a.GateExps = sg.Data; a.GateExpsType = sg.GgmlType; a.GateExpsBytes = sg.TotalRawBytes;
                        a.UpExps = su.Data; a.UpExpsType = su.GgmlType; a.UpExpsBytes = su.TotalRawBytes;
                        a.DownExps = sd.Data; a.DownExpsType = sd.GgmlType; a.DownExpsBytes = sd.TotalRawBytes;
                        var shg = ResolveW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]);
                        var shu = ResolveW(_ffnUpShexpQW[l], _ffnUpShexpF32[l]);
                        var shd = ResolveW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]);
                        a.ShexpGateW = shg.ptr; a.ShexpGateType = shg.type; a.ShexpGateNe0 = shg.ne0; a.ShexpGateNe1 = shg.ne1; a.ShexpGateBytes = shg.bytes;
                        a.ShexpUpW = shu.ptr; a.ShexpUpType = shu.type; a.ShexpUpNe0 = shu.ne0; a.ShexpUpNe1 = shu.ne1; a.ShexpUpBytes = shu.bytes;
                        a.ShexpDownW = shd.ptr; a.ShexpDownType = shd.type; a.ShexpDownNe0 = shd.ne0; a.ShexpDownNe1 = shd.ne1; a.ShexpDownBytes = shd.bytes;
                        a.ShexpGateInpW = (IntPtr)GetFloatPtr(_ffnGateInpShexpVec[l]);
                    }

                    if (!_isRecurrent[l])
                    {
                        a.IsRecurrent = 0;
                        var o = ResolveW(_attnOutputQW[l], _attnOutputF32[l]);
                        a.OW = o.ptr; a.OType = o.type; a.ONe0 = o.ne0; a.ONe1 = o.ne1; a.OBytes = o.bytes;
                        if (_attnQkvQW[l] != null || _attnQkvF32[l] != null)
                        {
                            var qkv = ResolveW(_attnQkvQW[l], _attnQkvF32[l]);
                            a.QkvW = qkv.ptr; a.QkvType = qkv.type; a.QkvNe0 = qkv.ne0; a.QkvNe1 = qkv.ne1; a.QkvBytes = qkv.bytes;
                            a.SeparateQkv = 0;
                        }
                        else
                        {
                            var q = ResolveW(_attnQQW[l], _attnQF32[l]);
                            var k = ResolveW(_attnKQW[l], _attnKF32[l]);
                            var v = ResolveW(_attnVQW[l], _attnVF32[l]);
                            a.QkvW = q.ptr; a.QkvType = q.type; a.QkvNe0 = q.ne0; a.QkvNe1 = q.ne1; a.QkvBytes = q.bytes;
                            a.KW = k.ptr; a.KType = k.type; a.KNe0 = k.ne0; a.KNe1 = k.ne1; a.KBytes = k.bytes;
                            a.VW = v.ptr; a.VType = v.type; a.VNe0 = v.ne0; a.VNe1 = v.ne1; a.VBytes = v.bytes;
                            a.SeparateQkv = 1;
                        }
                        a.QNormW = (IntPtr)GetFloatPtr(_attnQNormW[l]);
                        a.KNormW = (IntPtr)GetFloatPtr(_attnKNormW[l]);
                        a.KCache = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                        a.VCache = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
                    }
                    else
                    {
                        a.IsRecurrent = 1;
                        var gq = ResolveW(_attnQkvRecQW[l], _attnQkvRecF32[l]);
                        var gz = ResolveW(_attnGateRecQW[l], _attnGateRecF32[l]);
                        var sb = ResolveW(_ssmBetaQW[l], _ssmBetaF32[l]);
                        var sa = ResolveW(_ssmAlphaQW[l], _ssmAlphaF32[l]);
                        var so = ResolveW(_ssmOutQW[l], _ssmOutF32[l]);
                        a.GdnQkvW = gq.ptr; a.GdnQkvType = gq.type; a.GdnQkvNe0 = gq.ne0; a.GdnQkvNe1 = gq.ne1; a.GdnQkvBytes = gq.bytes;
                        a.GdnGateW = gz.ptr; a.GdnGateType = gz.type; a.GdnGateNe0 = gz.ne0; a.GdnGateNe1 = gz.ne1; a.GdnGateBytes = gz.bytes;
                        a.SsmBetaW = sb.ptr; a.SsmBetaType = sb.type; a.SsmBetaNe0 = sb.ne0; a.SsmBetaNe1 = sb.ne1; a.SsmBetaBytes = sb.bytes;
                        a.SsmAlphaW = sa.ptr; a.SsmAlphaType = sa.type; a.SsmAlphaNe0 = sa.ne0; a.SsmAlphaNe1 = sa.ne1; a.SsmAlphaBytes = sa.bytes;
                        a.SsmOutW = so.ptr; a.SsmOutType = so.type; a.SsmOutNe0 = so.ne0; a.SsmOutNe1 = so.ne1; a.SsmOutBytes = so.bytes;
                        a.Conv1dW = (IntPtr)GetFloatPtr(_ssmConv1dW[l]);
                        a.SsmDtW = (IntPtr)GetFloatPtr(_ssmDtBiasW[l]);
                        a.SsmAW = (IntPtr)GetFloatPtr(_ssmAW[l]);
                        a.SsmNormW = (IntPtr)GetFloatPtr(_ssmNormW[l]);

                        // GDN recurrent state is device-resident across decode tokens.
                        // On (re)seed, convert the host conv ring -> ggml [time, channel]
                        // layout and force the device cache to re-upload the host state;
                        // on resident tokens the device buffer is authoritative (no host
                        // touch, no per-token transfer).
                        float* conv = convBase + (long)_fdGdnSlot[l] * convBlock;
                        IntPtr deltaPtr = (IntPtr)GetFloatPtr(_deltaStateTensor[l]);
                        if (!_fdStateResident)
                        {
                            float[] ring = _convState[l];
                            int w = _convStateWriteIdx[l];
                            for (int t = 0; t < convDim; t++)
                            {
                                int slot = (w + t) % convDim;
                                int srcBase = slot * qkvDim;
                                for (int ch = 0; ch < qkvDim; ch++)
                                    conv[ch * convDim + t] = ring[srcBase + ch];
                            }
                            GgmlBasicOps.InvalidateHostBuffer((IntPtr)conv);
                            GgmlBasicOps.InvalidateHostBuffer(deltaPtr);
                        }
                        a.ConvStateIn = (IntPtr)conv;
                        a.ConvStateOut = (IntPtr)conv;
                        a.DeltaStateIn = deltaPtr;
                        a.DeltaStateOut = deltaPtr;
                    }
                    _fdLayers[l] = a;
                }

                // Fold final-norm + lm_head into the graph: output logits directly.
                var lmh = ResolveW(_lmHeadQW, _lmHeadF32);
                IntPtr finalNormPtr = (IntPtr)GetFloatPtr(_finalNormW);
                bool ok2;
                fixed (float* lp = logitsOut)
                {
                    ok2 = GgmlBasicOps.Qwen35ModelDecode(
                        _fdLayers, n,
                        (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, position,
                        Config.NumHeads, Config.NumKVHeads, headDim, cacheSize,
                        headDim, 2, kvCacheType,
                        _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                        Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                        _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                        _normTopKProb ? 1 : 0, 1.0f,
                        (IntPtr)lp, Config.VocabSize,
                        lmh.ptr, lmh.type, lmh.ne0, lmh.ne1, lmh.bytes,
                        finalNormPtr);
                }
                if (!ok2)
                {
                    if (!_fdDiagPrinted)
                    {
                        _fdDiagPrinted = true;
                        Console.Error.WriteLine($"[full-decode] disabled (native returned 0); falling back to per-op decode.");
                    }
                    _fdUnsupported = true;   // don't retry a failing kernel every token
                    return false;
                }

                // GDN state is now device-resident and was updated in-place; no
                // host write-back. Mark resident so subsequent tokens skip seeding.
                _fdStateResident = true;
            }

            // logitsOut now holds the final logits (final-norm + lm_head folded
            // into the graph); the caller skips the separate LM head.
            return true;
        }

        // Fused multi-token VERIFY (MTP speculative trunk): runs the whole hybrid
        // transformer over `seqLen` tokens of ONE sequence as a single GGML graph
        // (the N-token sibling of TryFullModelDecode), replacing the op-by-op
        // SpecForward layer loop. Writes the per-row logits [vocab, seqLen] and the
        // post-final-norm hidden [hidden, seqLen] (normedOut, the MTP draft head's
        // input), appends the N rows to the attention KV cache, and advances every
        // recurrent layer's GDN state (conv ring + delta) by N tokens. Returns
        // false (caller falls back to the op-by-op trunk) on any unsupported shape.
        // Gated TS_QWEN35_FUSED_VERIFY (default ON on ggml_cuda). The non-persist
        // fused verify is ~8.6x faster per verify step than the op-by-op trunk
        // (~121 ms vs ~1041 ms) and crash-free, so --mtp-spec routes through it; set
        // TS_QWEN35_FUSED_VERIFY=0 to fall back to the op-by-op batched trunk. The
        // persistent-cache + capture path (TS_Q35_VERIFY_PERSIST=1) would amortize the
        // remaining graph build to ~8 ms warm but currently access-violates on reuse
        // (interleaved op-by-op draft/catch-up disturbs shared cacheable buffers).
        // See native TSGgml_Qwen35ModelVerify.
        private static readonly bool _fusedVerifyEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_FUSED_VERIFY"), "0", StringComparison.Ordinal);
        private Qwen35LayerDecodeArgs[] _fvLayers;
        private int[] _fvGdnSlot;
        private IntPtr _fvConvIn;    // unmanaged [numGdnLayers * convDim * qkvDim] ggml layout (pre-window state)
        private IntPtr _fvConvOut;   // unmanaged, same size (post-window state)
        private bool _fvUnsupported;

        internal unsafe bool TryFullModelVerify(Tensor hidden, int startPos, int seqLen, float[] normedOut, float[] logitsOut)
        {
            if (!_fusedVerifyEnabled || _fvUnsupported || _backend != BackendType.GgmlCuda)
                return false;
            if (seqLen < 1 || hidden == null)
                return false;
            if (logitsOut == null || logitsOut.Length < (long)Config.VocabSize * seqLen)
                return false;
            if ((_lmHeadQW == null && _lmHeadF32 == null) || _finalNormW == null)
                return false;
            if (_headKDim != _headVDim)   // gated_delta_net needs S_k == S_v
                return false;

            int n = Config.NumLayers;
            int headDim = Config.HeadDim;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            if (convDim <= 0)
                return false;

            int cacheSize = 0;
            int kvCacheType = 0;
            for (int l = 0; l < n; l++)
            {
                if (!_isRecurrent[l])
                {
                    cacheSize = (int)_kvCacheK[l].Sizes[1];
                    kvCacheType = _kvCacheK[l].ElementType == DType.Float16 ? 1 : 0;
                    break;
                }
            }
            if (cacheSize <= 0 || startPos + seqLen > cacheSize)
                return false;

            int convBlock = convDim * qkvDim;
            int structBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>();

            // One-time capability gate + scratch allocation (mirrors TryFullModelDecode).
            if (_fvLayers == null)
            {
                static bool HasW(QuantizedWeight q, Tensor f) => q != null || f != null;
                int gdnCount = 0;
                _fvGdnSlot = new int[n];
                for (int l = 0; l < n; l++)
                {
                    bool isMoeL = _isMoeLayer != null && _isMoeLayer[l];
                    bool ffnOk = isMoeL
                        ? ((_ffnGateInpQW[l] != null || _ffnGateInpF32[l] != null)
                            && _layerStackedGate[l] != null && _layerStackedUp[l] != null && _layerStackedDown[l] != null
                            && HasW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]) && HasW(_ffnUpShexpQW[l], _ffnUpShexpF32[l])
                            && HasW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]) && _ffnGateInpShexpVec[l] != null)
                        : (HasW(_ffnGateUpQW[l], _ffnGateUpF32[l]) && HasW(_ffnDownQW[l], _ffnDownF32[l]));
                    bool ok = _attnNormW[l] != null && _postAttnNormW[l] != null && ffnOk;
                    if (ok && !_isRecurrent[l])
                        ok = (HasW(_attnQkvQW[l], _attnQkvF32[l])
                              || (HasW(_attnQQW[l], _attnQF32[l]) && HasW(_attnKQW[l], _attnKF32[l]) && HasW(_attnVQW[l], _attnVF32[l])))
                            && HasW(_attnOutputQW[l], _attnOutputF32[l])
                            && _attnQNormW[l] != null && _attnKNormW[l] != null
                            && _kvCacheK[l] != null && _kvCacheV[l] != null
                            && (_kvCacheK[l].ElementType == DType.Float32 || _kvCacheK[l].ElementType == DType.Float16);
                    if (ok && _isRecurrent[l])
                        ok = HasW(_attnQkvRecQW[l], _attnQkvRecF32[l]) && HasW(_attnGateRecQW[l], _attnGateRecF32[l])
                            && HasW(_ssmBetaQW[l], _ssmBetaF32[l]) && HasW(_ssmAlphaQW[l], _ssmAlphaF32[l])
                            && _ssmConv1dW[l] != null && _ssmDtBiasW[l] != null && _ssmAW[l] != null
                            && _ssmNormW[l] != null && HasW(_ssmOutQW[l], _ssmOutF32[l])
                            && _deltaStateTensor[l] != null && _convState[l] != null;
                    if (!ok) { _fvUnsupported = true; return false; }
                    _fvGdnSlot[l] = _isRecurrent[l] ? gdnCount++ : -1;
                }
                _fvConvIn = Marshal.AllocHGlobal(Math.Max(1, gdnCount) * convBlock * sizeof(float));
                _fvConvOut = Marshal.AllocHGlobal(Math.Max(1, gdnCount) * convBlock * sizeof(float));
                _fvLayers = new Qwen35LayerDecodeArgs[n];
            }

            float* convInBase = (float*)_fvConvIn;
            float* convOutBase = (float*)_fvConvOut;
            for (int l = 0; l < n; l++)
            {
                var a = default(Qwen35LayerDecodeArgs);
                a.StructBytes = structBytes;
                a.AttnNormW = (IntPtr)GetFloatPtr(_attnNormW[l]);
                a.PostAttnNormW = (IntPtr)GetFloatPtr(_postAttnNormW[l]);
                bool isMoe = _isMoeLayer != null && _isMoeLayer[l];
                a.IsMoe = isMoe ? 1 : 0;
                if (!isMoe)
                {
                    var gu = ResolveW(_ffnGateUpQW[l], _ffnGateUpF32[l]);
                    var dn = ResolveW(_ffnDownQW[l], _ffnDownF32[l]);
                    a.GuW = gu.ptr; a.GuType = gu.type; a.GuNe0 = gu.ne0; a.GuNe1 = gu.ne1; a.GuBytes = gu.bytes;
                    a.DownW = dn.ptr; a.DownType = dn.type; a.DownNe0 = dn.ne0; a.DownNe1 = dn.ne1; a.DownBytes = dn.bytes;
                    a.FfDense = (int)(gu.ne1 / 2);
                }
                else
                {
                    var gi = ResolveW(_ffnGateInpQW[l], _ffnGateInpF32[l]);
                    a.GateInpW = gi.ptr; a.GateInpType = gi.type; a.GateInpNe0 = gi.ne0; a.GateInpNe1 = gi.ne1; a.GateInpBytes = gi.bytes;
                    var sg = _layerStackedGate[l]; var su = _layerStackedUp[l]; var sd = _layerStackedDown[l];
                    a.GateExps = sg.Data; a.GateExpsType = sg.GgmlType; a.GateExpsBytes = sg.TotalRawBytes;
                    a.UpExps = su.Data; a.UpExpsType = su.GgmlType; a.UpExpsBytes = su.TotalRawBytes;
                    a.DownExps = sd.Data; a.DownExpsType = sd.GgmlType; a.DownExpsBytes = sd.TotalRawBytes;
                    var shg = ResolveW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]);
                    var shu = ResolveW(_ffnUpShexpQW[l], _ffnUpShexpF32[l]);
                    var shd = ResolveW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]);
                    a.ShexpGateW = shg.ptr; a.ShexpGateType = shg.type; a.ShexpGateNe0 = shg.ne0; a.ShexpGateNe1 = shg.ne1; a.ShexpGateBytes = shg.bytes;
                    a.ShexpUpW = shu.ptr; a.ShexpUpType = shu.type; a.ShexpUpNe0 = shu.ne0; a.ShexpUpNe1 = shu.ne1; a.ShexpUpBytes = shu.bytes;
                    a.ShexpDownW = shd.ptr; a.ShexpDownType = shd.type; a.ShexpDownNe0 = shd.ne0; a.ShexpDownNe1 = shd.ne1; a.ShexpDownBytes = shd.bytes;
                    a.ShexpGateInpW = (IntPtr)GetFloatPtr(_ffnGateInpShexpVec[l]);
                }

                if (!_isRecurrent[l])
                {
                    a.IsRecurrent = 0;
                    var o = ResolveW(_attnOutputQW[l], _attnOutputF32[l]);
                    a.OW = o.ptr; a.OType = o.type; a.ONe0 = o.ne0; a.ONe1 = o.ne1; a.OBytes = o.bytes;
                    if (_attnQkvQW[l] != null || _attnQkvF32[l] != null)
                    {
                        var qkv = ResolveW(_attnQkvQW[l], _attnQkvF32[l]);
                        a.QkvW = qkv.ptr; a.QkvType = qkv.type; a.QkvNe0 = qkv.ne0; a.QkvNe1 = qkv.ne1; a.QkvBytes = qkv.bytes;
                        a.SeparateQkv = 0;
                    }
                    else
                    {
                        var q = ResolveW(_attnQQW[l], _attnQF32[l]);
                        var k = ResolveW(_attnKQW[l], _attnKF32[l]);
                        var v = ResolveW(_attnVQW[l], _attnVF32[l]);
                        a.QkvW = q.ptr; a.QkvType = q.type; a.QkvNe0 = q.ne0; a.QkvNe1 = q.ne1; a.QkvBytes = q.bytes;
                        a.KW = k.ptr; a.KType = k.type; a.KNe0 = k.ne0; a.KNe1 = k.ne1; a.KBytes = k.bytes;
                        a.VW = v.ptr; a.VType = v.type; a.VNe0 = v.ne0; a.VNe1 = v.ne1; a.VBytes = v.bytes;
                        a.SeparateQkv = 1;
                    }
                    a.QNormW = (IntPtr)GetFloatPtr(_attnQNormW[l]);
                    a.KNormW = (IntPtr)GetFloatPtr(_attnKNormW[l]);
                    a.KCache = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                    a.VCache = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
                }
                else
                {
                    a.IsRecurrent = 1;
                    var gq = ResolveW(_attnQkvRecQW[l], _attnQkvRecF32[l]);
                    var gz = ResolveW(_attnGateRecQW[l], _attnGateRecF32[l]);
                    var sb = ResolveW(_ssmBetaQW[l], _ssmBetaF32[l]);
                    var sa = ResolveW(_ssmAlphaQW[l], _ssmAlphaF32[l]);
                    var so = ResolveW(_ssmOutQW[l], _ssmOutF32[l]);
                    a.GdnQkvW = gq.ptr; a.GdnQkvType = gq.type; a.GdnQkvNe0 = gq.ne0; a.GdnQkvNe1 = gq.ne1; a.GdnQkvBytes = gq.bytes;
                    a.GdnGateW = gz.ptr; a.GdnGateType = gz.type; a.GdnGateNe0 = gz.ne0; a.GdnGateNe1 = gz.ne1; a.GdnGateBytes = gz.bytes;
                    a.SsmBetaW = sb.ptr; a.SsmBetaType = sb.type; a.SsmBetaNe0 = sb.ne0; a.SsmBetaNe1 = sb.ne1; a.SsmBetaBytes = sb.bytes;
                    a.SsmAlphaW = sa.ptr; a.SsmAlphaType = sa.type; a.SsmAlphaNe0 = sa.ne0; a.SsmAlphaNe1 = sa.ne1; a.SsmAlphaBytes = sa.bytes;
                    a.SsmOutW = so.ptr; a.SsmOutType = so.type; a.SsmOutNe0 = so.ne0; a.SsmOutNe1 = so.ne1; a.SsmOutBytes = so.bytes;
                    a.Conv1dW = (IntPtr)GetFloatPtr(_ssmConv1dW[l]);
                    a.SsmDtW = (IntPtr)GetFloatPtr(_ssmDtBiasW[l]);
                    a.SsmAW = (IntPtr)GetFloatPtr(_ssmAW[l]);
                    a.SsmNormW = (IntPtr)GetFloatPtr(_ssmNormW[l]);

                    // Convert the C# conv ring -> ggml [time, channel] (pre-window state).
                    float* convIn = convInBase + (long)_fvGdnSlot[l] * convBlock;
                    float* convOut = convOutBase + (long)_fvGdnSlot[l] * convBlock;
                    float[] ring = _convState[l];
                    int w = _convStateWriteIdx[l];
                    for (int t = 0; t < convDim; t++)
                    {
                        int slot = (w + t) % convDim;
                        int srcBase = slot * qkvDim;
                        for (int ch = 0; ch < qkvDim; ch++)
                            convIn[ch * convDim + t] = ring[srcBase + ch];
                    }
                    IntPtr deltaPtr = (IntPtr)GetFloatPtr(_deltaStateTensor[l]);
                    GgmlBasicOps.InvalidateHostBuffer((IntPtr)convIn);
                    GgmlBasicOps.InvalidateHostBuffer(deltaPtr);
                    a.ConvStateIn = (IntPtr)convIn;
                    a.ConvStateOut = (IntPtr)convOut;
                    a.DeltaStateIn = deltaPtr;
                    a.DeltaStateOut = deltaPtr;   // in-place: overwritten with the post-window state
                }
                _fvLayers[l] = a;
            }

            var lmh = ResolveW(_lmHeadQW, _lmHeadF32);
            IntPtr finalNormPtr = (IntPtr)GetFloatPtr(_finalNormW);
            bool ok2;
            fixed (float* lp = logitsOut)
            fixed (float* np = normedOut)
            {
                ok2 = GgmlBasicOps.Qwen35ModelVerify(
                    _fvLayers, n,
                    (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, startPos, seqLen,
                    Config.NumHeads, Config.NumKVHeads, headDim, cacheSize,
                    headDim, 2, kvCacheType,
                    _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                    Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                    _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                    _normTopKProb ? 1 : 0, 1.0f,
                    (IntPtr)lp, Config.VocabSize,
                    lmh.ptr, lmh.type, lmh.ne0, lmh.ne1, lmh.bytes,
                    finalNormPtr, normedOut != null ? (IntPtr)np : IntPtr.Zero);
            }
            if (!ok2)
            {
                _fvUnsupported = true;
                return false;
            }

            // Write the post-window GDN state back to the C# representation: conv
            // ggml [time, channel] -> ring (normalised writeIdx=0); delta was written
            // in-place to _deltaStateTensor (invalidate so later op-by-op ops re-read).
            for (int l = 0; l < n; l++)
            {
                if (!_isRecurrent[l]) continue;
                float* convOut = convOutBase + (long)_fvGdnSlot[l] * convBlock;
                float[] ring = _convState[l];
                for (int t = 0; t < convDim; t++)
                {
                    int dstBase = t * qkvDim;
                    for (int ch = 0; ch < qkvDim; ch++)
                        ring[dstBase + ch] = convOut[ch * convDim + t];
                }
                _convStateWriteIdx[l] = 0;
                GgmlBasicOps.InvalidateHostBuffer((IntPtr)GetFloatPtr(_deltaStateTensor[l]));
            }
            return true;
        }

        /// <summary>
        /// GatedDeltaNet recurrent step with batched input/output projections.
        /// Prefill projects the whole chunk once, then walks the recurrent state token-by-token.
        /// Decode follows the same path with seqLen=1, avoiding several tiny GGML dispatches.
        /// </summary>
        private unsafe Tensor GatedDeltaNet(Tensor input, Tensor inputNormW, int layer, int seqLen,
            Tensor residual = null, bool skipOutputProj = false)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            long stageStart = profilePrefill ? t0 : 0;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int zDim = _headVDim * _numVHeads;
            int packedDim = qkvDim + zDim + _numVHeads * 2;

            Tensor packedInput = null;
            bool ownsPackedInput = true;
            if (inputNormW != null && _ssmInProjQW[layer] != null && IsGgmlBackend)
            {
                // Fused input norm + packed input projection (single GGML kernel).
                // Decode reuses a pre-allocated [1, packedDim] buffer to avoid one
                // tensor allocation per layer per token.
                if (seqLen == 1 && _gdnDecodePackedBuf != null
                    && _gdnDecodePackedBuf.Sizes[1] == _ssmInProjQW[layer].Ne1)
                {
                    packedInput = TryFusedNormLinearInto(_gdnDecodePackedBuf, input, inputNormW, _ssmInProjQW[layer]);
                    if (packedInput != null)
                        ownsPackedInput = false;
                }
                if (packedInput == null)
                    packedInput = FusedNormLinear(input, inputNormW, _ssmInProjQW[layer], _ssmInProjF32[layer]);
            }

            Tensor normedInput = null;
            if (packedInput == null)
            {
                normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();
                packedInput = LinearForwardCached(normedInput, _ssmInProjQW[layer], _ssmInProjF32[layer]);
            }

            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillRecInputProjTicks += now - stageStart; stageStart = now; }

            Tensor qkvRaw = null;
            Tensor zRaw = null;
            Tensor betaRaw = null;
            Tensor alphaRaw = null;

            float* packedPtr = null;
            float* qkvBase = null;
            float* zBase = null;
            float* betaBase = null;
            float* alphaBase = null;
            Tensor gated = null;
            float* gatedBase = null;
            bool ranMlxNativeGdn = false;
            bool ranCudaNativeGdn = false;

            if (packedInput != null)
            {
                if (_backend == BackendType.Mlx
                    && _mlxGdnCache?[layer] != null
                    && _ssmConv1dW[layer] != null
                    && _ssmDtBiasW[layer] != null
                    && _ssmAW[layer] != null
                    && _ssmNormW[layer] != null
                    && (seqLen == 1 || string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QWEN35_GDN_PACKED_KERNELS"), "1", StringComparison.Ordinal)))
                {
                    gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
                    ranMlxNativeGdn = _mlxGdnCache[layer].TryRunQwen35Packed(
                        gated,
                        packedInput,
                        _ssmConv1dW[layer],
                        _ssmDtBiasW[layer],
                        _ssmAW[layer],
                        _ssmNormW[layer],
                        seqLen,
                        packedDim,
                        qkvDim,
                        qkDim,
                        vDim,
                        _numKHeads,
                        _numVHeads,
                        _headKDim,
                        _headVDim,
                        _convKernel,
                        Config.Eps);
                    if (!ranMlxNativeGdn)
                    {
                        if (seqLen > 1)
                            gated.Dispose();
                        gated = null;
                    }
                }

                if (!ranMlxNativeGdn)
                    ranCudaNativeGdn = TryRunCudaNativeGdnPacked(
                        packedInput, layer, seqLen, packedDim, qkvDim, qkDim, vDim, out gated);

                if (!ranMlxNativeGdn && !ranCudaNativeGdn)
                    packedPtr = GetFloatPtr(packedInput);
            }
            else
            {
                if (normedInput == null)
                    normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();

                qkvRaw = LinearForwardCached(normedInput, _attnQkvRecQW[layer], _attnQkvRecF32[layer]);
                zRaw = LinearForwardCached(normedInput, _attnGateRecQW[layer], _attnGateRecF32[layer]);
                betaRaw = LinearForwardCached(normedInput, _ssmBetaQW[layer], _ssmBetaF32[layer]);
                alphaRaw = LinearForwardCached(normedInput, _ssmAlphaQW[layer], _ssmAlphaF32[layer]);

                if (_backend == BackendType.Mlx
                    && _mlxGdnCache?[layer] != null
                    && _ssmConv1dW[layer] != null
                    && _ssmDtBiasW[layer] != null
                    && _ssmAW[layer] != null
                    && _ssmNormW[layer] != null)
                {
                    gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
                    ranMlxNativeGdn = _mlxGdnCache[layer].TryRunQwen35(
                        gated,
                        qkvRaw,
                        zRaw,
                        betaRaw,
                        alphaRaw,
                        _ssmConv1dW[layer],
                        _ssmDtBiasW[layer],
                        _ssmAW[layer],
                        _ssmNormW[layer],
                        seqLen,
                        qkvDim,
                        qkDim,
                        vDim,
                        _numKHeads,
                        _numVHeads,
                        _headKDim,
                        _headVDim,
                        _convKernel,
                        Config.Eps);
                    if (!ranMlxNativeGdn)
                    {
                        if (seqLen > 1)
                            gated.Dispose();
                        gated = null;
                    }
                }

                if (!ranMlxNativeGdn && CudaGdnNativeEnabledEnv && _backend == BackendType.Cuda)
                {
                    bool reuseDecodePacked = seqLen == 1 && _gdnDecodePackedBuf != null;
                    Tensor cudaPacked = reuseDecodePacked
                        ? _gdnDecodePackedBuf
                        : new Tensor(_allocator, DType.Float32, seqLen, packedDim);
                    bool packedOk = CudaFusedOps.TryQwen35GatedDeltaNetPackInputs(
                        cudaPacked,
                        qkvRaw,
                        zRaw,
                        betaRaw,
                        alphaRaw,
                        seqLen,
                        qkvDim,
                        zDim,
                        _numVHeads,
                        packedDim);
                    if (packedOk)
                    {
                        ranCudaNativeGdn = TryRunCudaNativeGdnPacked(
                            cudaPacked, layer, seqLen, packedDim, qkvDim, qkDim, vDim, out gated);
                    }
                    if (!reuseDecodePacked)
                        cudaPacked.Dispose();
                }

                if (!ranMlxNativeGdn && !ranCudaNativeGdn)
                {
                    qkvBase = GetFloatPtr(qkvRaw);
                    zBase = GetFloatPtr(zRaw);
                    betaBase = GetFloatPtr(betaRaw);
                    alphaBase = GetFloatPtr(alphaRaw);
                }
            }

            if (!ranMlxNativeGdn && !ranCudaNativeGdn)
            {
                // Pre-resolved layer constants (cached at construction; no dictionary lookup here).
                float* dtBiasPtr = GetFloatPtr(_ssmDtBiasW[layer]);
                float* aPtr = GetFloatPtr(_ssmAW[layer]);
                float* ssmNormPtr = GetFloatPtr(_ssmNormW[layer]);

                gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
                gatedBase = GetFloatPtr(gated);

                float[] convWT = _gdnConvWT[layer];

                // Prefer the fused chunked GGML kernel when running on a GGML backend with
                // sufficient sequence length and matching K/V head dims. The chunked kernel
                // packs the entire delta-net block (L2Norm, mul_mat, triangular solve, RMSNorm,
                // gating) into a single GPU dispatch per layer, drastically reducing CPU work
                // during prefill.
                bool useChunked = !_gdnDisableChunkedPrefill
                    && IsGgmlBackend
                    && seqLen >= _gdnChunkPrefillThreshold
                    && _headKDim == _headVDim
                    && _ssmDtBiasW[layer] != null
                    && _ssmAW[layer] != null
                    && _ssmNormW[layer] != null;

                if (useChunked && GdnVerifyChunkedEnv && seqLen > 1)
                {
                    // Verification mode: run the chunked path on a snapshot of the
                    // recurrent state, then restore the state, run the per-token
                    // path on the SAME starting state, and compare. Use the
                    // per-token path's output as the canonical forward result so
                    // that any regression in the chunked kernel is bounded to the
                    // verify counters and never leaks into downstream layers.
                    VerifyAndRunPerTokenAfterChunked(
                        packedPtr, qkvBase, zBase, betaBase, alphaBase,
                        layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                        convWT, dtBiasPtr, aPtr, ssmNormPtr, gated, gatedBase);
                }
                else if (useChunked)
                {
                    long tChunk = Stopwatch.GetTimestamp();
                    bool chunkedOk = false;
                    try
                    {
                        GatedDeltaNetChunkedPrefill(
                            packedPtr, qkvBase, zBase, betaBase, alphaBase,
                            layer, seqLen, qkvDim, qkDim, zDim, packedDim, gated);
                        chunkedOk = true;
                    }
                    catch (Exception ex)
                    {
                        // First failure trips the kill switch so subsequent layers / forwards
                        // do not pay the same overhead twice. Fall back to the per-token loop.
                        _gdnDisableChunkedPrefill = true;
                        Console.WriteLine($"[Qwen35] GatedDeltaNetChunked disabled (layer {layer}, seqLen {seqLen}): {ex.Message}");
                    }

                    if (chunkedOk)
                    {
                        _gdnChunkedTicks += Stopwatch.GetTimestamp() - tChunk;
                        _gdnChunkedCalls++;
                    }
                    else
                    {
                        // Fall back: clean state was already mutated only inside the helper. Run
                        // the per-token path on the same gated buffer.
                        long tFallback = Stopwatch.GetTimestamp();
                        RunPerTokenLoop(packedPtr, qkvBase, zBase, betaBase, alphaBase,
                            layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                            convWT, dtBiasPtr, aPtr, ssmNormPtr, gatedBase);
                        _gdnPerTokenTicks += Stopwatch.GetTimestamp() - tFallback;
                        if (seqLen > 1) _gdnPerTokenCalls++;
                    }
                }
                else
                {
                    long tLoop = Stopwatch.GetTimestamp();
                    RunPerTokenLoop(packedPtr, qkvBase, zBase, betaBase, alphaBase,
                        layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                        convWT, dtBiasPtr, aPtr, ssmNormPtr, gatedBase);
                    _gdnPerTokenTicks += Stopwatch.GetTimestamp() - tLoop;
                    if (seqLen > 1) _gdnPerTokenCalls++;
                }

                InvalidateTensorDeviceCache(gated);
            }
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillRecCoreTicks += now - stageStart; stageStart = now; }

            // When skipOutputProj is set, return the raw gated output so the caller
            // can fuse output_proj + FFN into a single GPU dispatch.
            if (skipOutputProj)
            {
                Tensor gatedOut = (seqLen == 1) ? gated.CopyRef() : gated;
                normedInput?.Dispose();
                if (ownsPackedInput) packedInput?.Dispose();
                qkvRaw?.Dispose(); zRaw?.Dispose(); betaRaw?.Dispose(); alphaRaw?.Dispose();
                if (profilePrefill) _prefillRecOutputTicks += Stopwatch.GetTimestamp() - stageStart;
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return gatedOut;
            }

            // Fast path: fuse SSM output projection with the residual add.
            Tensor output;
            bool fusedAdd = false;
            if (residual != null
                && _ssmOutQW[layer] != null
                && residual.DimensionCount == 2
                && gated.DimensionCount == 2
                && residual.Sizes[0] == gated.Sizes[0]
                && TryLinearAddInto(residual, gated, _ssmOutQW[layer]))
            {
                output = null;
                fusedAdd = true;
            }
            else
            {
                output = LinearForwardCached(gated, _ssmOutQW[layer], _ssmOutF32[layer]);
            }

            if (seqLen > 1)
                gated.Dispose();

            normedInput?.Dispose();
            if (ownsPackedInput) packedInput?.Dispose();
            qkvRaw?.Dispose();
            zRaw?.Dispose();
            betaRaw?.Dispose();
            alphaRaw?.Dispose();

            if (profilePrefill) _prefillRecOutputTicks += Stopwatch.GetTimestamp() - stageStart;
            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return fusedAdd ? null : output;
        }

        private bool TryRunCudaNativeGdnPacked(
            Tensor packedInput,
            int layer,
            int seqLen,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            out Tensor gated)
        {
            gated = null;
            if (!CudaGdnNativeEnabledEnv
                || _backend != BackendType.Cuda
                || packedInput == null
                || _cudaGdnConvStateTensor?[layer] == null
                || _ssmConv1dW[layer] == null
                || _ssmDtBiasW[layer] == null
                || _ssmAW[layer] == null
                || _ssmNormW[layer] == null)
            {
                return false;
            }

            gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
            long tCuda = Stopwatch.GetTimestamp();
            bool ok = CudaFusedOps.TryQwen35GatedDeltaNetPacked(
                gated,
                packedInput,
                _cudaGdnConvStateTensor[layer],
                _deltaStateTensor[layer],
                _ssmConv1dW[layer],
                _ssmDtBiasW[layer],
                _ssmAW[layer],
                _ssmNormW[layer],
                seqLen,
                packedDim,
                qkvDim,
                qkDim,
                vDim,
                _numKHeads,
                _numVHeads,
                _headKDim,
                _headVDim,
                _convKernel,
                _convStateWriteIdx[layer],
                Config.Eps);

            if (!ok)
            {
                if (seqLen > 1)
                    gated.Dispose();
                gated = null;
                return false;
            }

            int convDim = _convKernel - 1;
            if (convDim > 0)
                _convStateWriteIdx[layer] = (_convStateWriteIdx[layer] + seqLen) % convDim;
            _gdnCudaNativeTicks += Stopwatch.GetTimestamp() - tCuda;
            _gdnCudaNativeCalls++;
            return true;
        }

        /// <summary>
        /// Verification harness for the chunked GatedDeltaNet prefill path. Snapshots
        /// the recurrent state, runs the chunked kernel, then restores the snapshot,
        /// runs the per-token loop on the same starting state, and compares the two
        /// gated outputs and resulting recurrent states. The per-token result is left
        /// in <paramref name="gated"/> so the rest of the forward pass uses the
        /// well-tested code path even if the chunked kernel diverges.
        ///
        /// Enabled via <c>GDN_VERIFY_CHUNKED=1</c>; intended for CI / debugging.
        /// </summary>
        private unsafe void VerifyAndRunPerTokenAfterChunked(
            float* packedPtr, float* qkvBase, float* zBase, float* betaBase, float* alphaBase,
            int layer, int seqLen, int qkvDim, int qkDim, int vDim, int zDim, int packedDim,
            float[] convWT, float* dtBiasPtr, float* aPtr, float* ssmNormPtr,
            Tensor gated, float* gatedBase)
        {
            int convStateLen = _convState[layer].Length;
            float[] convStateSnap = ArrayPool<float>.Shared.Rent(convStateLen);
            Array.Copy(_convState[layer], convStateSnap, convStateLen);
            int convWriteIdxSnap = _convStateWriteIdx[layer];

            Tensor deltaState = _deltaStateTensor[layer];
            int deltaStateLen = (int)deltaState.ElementCount();
            float[] deltaStateSnap = ArrayPool<float>.Shared.Rent(deltaStateLen);
            float* deltaStatePtr = GetFloatPtr(deltaState);
            fixed (float* dst = deltaStateSnap)
            {
                long bytes = (long)deltaStateLen * sizeof(float);
                Buffer.MemoryCopy(deltaStatePtr, dst, bytes, bytes);
            }

            // Run the chunked path. On failure we fall through to the per-token path
            // below which is exactly the production fallback behaviour.
            bool chunkedOk = false;
            long tChunk = Stopwatch.GetTimestamp();
            try
            {
                GatedDeltaNetChunkedPrefill(
                    packedPtr, qkvBase, zBase, betaBase, alphaBase,
                    layer, seqLen, qkvDim, qkDim, zDim, packedDim, gated);
                chunkedOk = true;
            }
            catch (Exception ex)
            {
                _gdnDisableChunkedPrefill = true;
                Console.WriteLine($"[Qwen35][verify] GatedDeltaNetChunked failed (layer {layer}, seqLen {seqLen}): {ex.Message}");
            }
            if (chunkedOk)
            {
                _gdnChunkedTicks += Stopwatch.GetTimestamp() - tChunk;
                _gdnChunkedCalls++;
            }

            int gatedLen = seqLen * _ssmDInner;
            float[] chunkedGated = ArrayPool<float>.Shared.Rent(gatedLen);
            float[] chunkedDelta = ArrayPool<float>.Shared.Rent(deltaStateLen);
            if (chunkedOk)
            {
                fixed (float* dst = chunkedGated)
                {
                    long bytes = (long)gatedLen * sizeof(float);
                    Buffer.MemoryCopy(gatedBase, dst, bytes, bytes);
                }
                fixed (float* dst = chunkedDelta)
                {
                    long bytes = (long)deltaStateLen * sizeof(float);
                    Buffer.MemoryCopy(deltaStatePtr, dst, bytes, bytes);
                }
            }

            // Restore conv state and delta state to the pre-chunked snapshot so the
            // per-token loop sees the same starting conditions.
            Array.Copy(convStateSnap, _convState[layer], convStateLen);
            _convStateWriteIdx[layer] = convWriteIdxSnap;
            fixed (float* src = deltaStateSnap)
            {
                long bytes = (long)deltaStateLen * sizeof(float);
                Buffer.MemoryCopy(src, deltaStatePtr, bytes, bytes);
            }
            InvalidateTensorDeviceCache(deltaState);

            // Per-token reference run.
            long tLoop = Stopwatch.GetTimestamp();
            RunPerTokenLoop(packedPtr, qkvBase, zBase, betaBase, alphaBase,
                layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                convWT, dtBiasPtr, aPtr, ssmNormPtr, gatedBase);
            _gdnPerTokenTicks += Stopwatch.GetTimestamp() - tLoop;
            _gdnPerTokenCalls++;

            if (chunkedOk)
            {
                ComputeAbsRelDiff(chunkedGated, gatedBase, gatedLen,
                    out float outAbs, out float outRel);
                ComputeAbsRelDiff(chunkedDelta, deltaStatePtr, deltaStateLen,
                    out float stateAbs, out float stateRel);

                _gdnVerifyCalls++;
                if (outAbs > _gdnVerifyMaxOutAbs) _gdnVerifyMaxOutAbs = outAbs;
                if (outRel > _gdnVerifyMaxOutRel) _gdnVerifyMaxOutRel = outRel;
                if (stateAbs > _gdnVerifyMaxStateAbs) _gdnVerifyMaxStateAbs = stateAbs;
                if (stateRel > _gdnVerifyMaxStateRel) _gdnVerifyMaxStateRel = stateRel;

                if (outAbs > GdnVerifyAbsDiffWarn && outRel > GdnVerifyRelDiffWarn)
                {
                    _gdnVerifyWarnings++;
                    if (_gdnVerifyWarnings <= 4)
                    {
                        Console.WriteLine($"[Qwen35][verify] layer {layer} seqLen={seqLen} " +
                            $"output |Δ|max={outAbs:E2} rel={outRel:E2} " +
                            $"state |Δ|max={stateAbs:E2} rel={stateRel:E2}");
                    }
                }
            }

            ArrayPool<float>.Shared.Return(convStateSnap);
            ArrayPool<float>.Shared.Return(deltaStateSnap);
            ArrayPool<float>.Shared.Return(chunkedGated);
            ArrayPool<float>.Shared.Return(chunkedDelta);
        }

        /// <summary>
        /// Compute the maximum absolute and relative element-wise difference between
        /// a managed reference array <paramref name="reference"/> (sized
        /// <paramref name="length"/>) and a native pointer <paramref name="actual"/>.
        /// Returns 0/0 when the reference is identically zero so the metric stays
        /// well-defined for a freshly reset state tensor.
        /// </summary>
        private static unsafe void ComputeAbsRelDiff(
            float[] reference, float* actual, int length,
            out float maxAbs, out float maxRel)
        {
            float absMax = 0f;
            float relMax = 0f;
            for (int i = 0; i < length; i++)
            {
                float r = reference[i];
                float a = actual[i];
                float d = MathF.Abs(r - a);
                if (d > absMax) absMax = d;
                float denom = MathF.Max(MathF.Abs(r), MathF.Abs(a));
                if (denom > 1e-6f)
                {
                    float rel = d / denom;
                    if (rel > relMax) relMax = rel;
                }
            }
            maxAbs = absMax;
            maxRel = relMax;
        }

        /// <summary>
        /// Per-token recurrent loop that walks the chunk one input at a time. Used both
        /// for decode (seqLen=1) and as the chunked-path fallback for prefill.
        /// </summary>
        private unsafe void RunPerTokenLoop(
            float* packedPtr, float* qkvBase, float* zBase, float* betaBase, float* alphaBase,
            int layer, int seqLen, int qkvDim, int qkDim, int vDim, int zDim, int packedDim,
            float[] convWT, float* dtBiasPtr, float* aPtr, float* ssmNormPtr, float* gatedBase)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* qkvPtr;
                float* zPtr;
                float* betaPtr;
                float* alphaPtr;

                if (packedPtr != null)
                {
                    float* row = packedPtr + (long)s * packedDim;
                    qkvPtr = row;
                    zPtr = row + qkvDim;
                    betaPtr = zPtr + zDim;
                    alphaPtr = betaPtr + _numVHeads;
                }
                else
                {
                    qkvPtr = qkvBase + (long)s * qkvDim;
                    zPtr = zBase + (long)s * zDim;
                    betaPtr = betaBase + (long)s * _numVHeads;
                    alphaPtr = alphaBase + (long)s * _numVHeads;
                }

                GatedDeltaNetStep(qkvPtr, zPtr, betaPtr, alphaPtr,
                    layer, qkvDim, qkDim, vDim,
                    convWT, dtBiasPtr, aPtr, ssmNormPtr,
                    gatedBase + (long)s * _ssmDInner);
            }
        }

        /// <summary>
        /// Chunked GatedDeltaNet prefill path.
        ///
        /// Runs the conv1d step token-by-token on CPU (so the recurrent ring state is
        /// updated correctly), packs the per-token Q/K/V/Z/alpha/beta into row-major
        /// [seqLen, H, D] (or [seqLen, H]) tensors, and dispatches a single fused GGML
        /// graph that does L2Norm, Q-scale, sigmoid(beta), softplus(alpha), per-chunk
        /// (k @ q) attention with triangular solve, RMSNorm, and silu(z) gating - all on
        /// the GGML backend (Metal / CUDA when available).
        ///
        /// The recurrent state tensor (_deltaStateTensor[layer], shape [H, D, D]) is
        /// passed in as input/output. The kernel updates it in place on the device and
        /// downloads the new value back to the host buffer.
        /// </summary>
        private unsafe void GatedDeltaNetChunkedPrefill(
            float* packedPtr, float* qkvBase, float* zBase, float* betaBase, float* alphaBase,
            int layer, int seqLen, int qkvDim, int qkDim, int zDim, int packedDim,
            Tensor gated)
        {
            int H = _numVHeads;
            int Dk = _headKDim;
            int Dv = _headVDim;
            int hKDim = H * Dk;
            int hVDim = H * Dv;
            int convDim = _convKernel - 1;
            int convKernel = _convKernel;

            EnsureChunkedStagingBuffers(seqLen, H, Dk, Dv);
            EnsureConv1DScratchBuffers(seqLen, qkvDim, convDim, H);

            // The staging tensors are sized for the largest seqLen seen so far. We work
            // on sub-views at the actual seqLen so the native kernel sees the correct shape.
            Tensor qBuf = _gdnChunkedQBuf.Narrow(0, 0, seqLen);
            Tensor kBuf = _gdnChunkedKBuf.Narrow(0, 0, seqLen);
            Tensor vBuf = _gdnChunkedVBuf.Narrow(0, 0, seqLen);
            Tensor zBuf = _gdnChunkedZBuf.Narrow(0, 0, seqLen);
            Tensor alphaBuf = _gdnChunkedAlphaBuf.Narrow(0, 0, seqLen);
            Tensor betaBuf = _gdnChunkedBetaBuf.Narrow(0, 0, seqLen);

            try
            {
                long tCpuPrepStart = Stopwatch.GetTimestamp();

                float* qPtr = GetFloatPtr(qBuf);
                float* kPtr = GetFloatPtr(kBuf);
                float* vPtr = GetFloatPtr(vBuf);
                float* zPtr = GetFloatPtr(zBuf);
                float* alphaPtr = GetFloatPtr(alphaBuf);
                float* betaPtr = GetFloatPtr(betaBuf);

                float[] convState = _convState[layer];
                float[] convWT = _gdnConvWT[layer];
                int writeIdx = _convStateWriteIdx[layer];

                // ------------------------------------------------------------------
                // Phase 1: build the extended Conv1D input buffer.
                // Rows 0..convDim-1 hold the linearised ring state (oldest tap first).
                // Rows convDim..convDim+seqLen-1 hold this prefill's packed QKV input.
                // ------------------------------------------------------------------
                fixed (float* extPtrPinned = _gdnConvExtendedBuf)
                fixed (float* statePtr = convState)
                {
                    if (convDim > 0)
                    {
                        for (int ki = 0; ki < convDim; ki++)
                        {
                            int slot = (writeIdx + ki) % convDim;
                            long bytes = (long)qkvDim * sizeof(float);
                            Buffer.MemoryCopy(
                                statePtr + (long)slot * qkvDim,
                                extPtrPinned + (long)ki * qkvDim,
                                bytes, bytes);
                        }
                    }

                    // The input bytes are scattered across either a packed row
                    // ([qkv|z|beta|alpha]) or separate bases. Either way we need a
                    // contiguous [seqLen, qkvDim] block starting at row `convDim`.
                    float* inputBase = extPtrPinned + (long)convDim * qkvDim;
                    if (packedPtr != null && packedDim == qkvDim)
                    {
                        // Rare: packed layout is exactly qkvDim so we can do one memcpy.
                        long bytes = (long)seqLen * qkvDim * sizeof(float);
                        Buffer.MemoryCopy(packedPtr, inputBase, bytes, bytes);
                    }
                    else if (packedPtr == null)
                    {
                        long bytes = (long)seqLen * qkvDim * sizeof(float);
                        Buffer.MemoryCopy(qkvBase, inputBase, bytes, bytes);
                    }
                    else
                    {
                        long rowBytes = (long)qkvDim * sizeof(float);
                        for (int s = 0; s < seqLen; s++)
                        {
                            float* row = packedPtr + (long)s * packedDim;
                            Buffer.MemoryCopy(row, inputBase + (long)s * qkvDim, rowBytes, rowBytes);
                        }
                    }

                    // Update the ring state in place from the last `convDim` input rows.
                    // Earlier writes would be overwritten anyway, so we only replay the
                    // tail of the input.
                    if (convDim > 0)
                    {
                        int startS = Math.Max(0, seqLen - convDim);
                        long rowBytes = (long)qkvDim * sizeof(float);
                        for (int s = startS; s < seqLen; s++)
                        {
                            int slot = (writeIdx + s) % convDim;
                            Buffer.MemoryCopy(
                                inputBase + (long)s * qkvDim,
                                statePtr + (long)slot * qkvDim,
                                rowBytes, rowBytes);
                        }
                        writeIdx = (writeIdx + seqLen) % convDim;
                        _convStateWriteIdx[layer] = writeIdx;
                    }
                }

                // ------------------------------------------------------------------
                // Phase 2: single parallel pass over tokens that fuses Conv1D + SiLU +
                // staging memcpys. Keeping everything in one Parallel.For means:
                //   * one thread-pool dispatch per call instead of two (saves the
                //     Parallel.For overhead x 2 that adds up across 24 layers)
                //   * each token's conv output stays hot in L1 for the subsequent
                //     per-token SiLU and memcpys - the batched 24 MB SiLU pass over
                //     seqLen x qkvDim would otherwise round-trip through DRAM.
                //
                // Pointers are captured via IntPtr because C# cannot close over
                // pointer locals, and we lift the pins outside the Parallel.For so
                // each worker walks already-pinned memory without its own `fixed`.
                // ------------------------------------------------------------------
                IntPtr extPtrIP;
                IntPtr convWTIP;
                fixed (float* extPin = _gdnConvExtendedBuf)
                fixed (float* wtPin = convWT)
                {
                    extPtrIP = (IntPtr)extPin;
                    convWTIP = (IntPtr)wtPin;

                    int capturedQkvDim = qkvDim;
                    int capturedKernel = convKernel;
                    int capturedQkDim = qkDim;
                    int capturedZDim = zDim;
                    int capturedHKDim = hKDim;
                    int capturedHVDim = hVDim;
                    int capturedDk = Dk;
                    int capturedH = H;
                    int capturedNumKHeads = _numKHeads;
                    int capturedNumVHeads = _numVHeads;
                    int capturedPackedDim = packedDim;

                    IntPtr qPtrIP = (IntPtr)qPtr;
                    IntPtr kPtrIP = (IntPtr)kPtr;
                    IntPtr vPtrIP = (IntPtr)vPtr;
                    IntPtr zPtrIP = (IntPtr)zPtr;
                    IntPtr alphaPtrIP = (IntPtr)alphaPtr;
                    IntPtr betaPtrIP = (IntPtr)betaPtr;
                    IntPtr packedPtrIP = (IntPtr)packedPtr;
                    IntPtr zBaseIP = (IntPtr)zBase;
                    IntPtr alphaBaseIP = (IntPtr)alphaBase;
                    IntPtr betaBaseIP = (IntPtr)betaBase;

                    // Partition by range so the framework packs ~seqLen/nCores
                    // contiguous iterations per worker, amortising Parallel.For
                    // dispatch overhead (~100 us) over a larger compute block.
                    var partitioner = Partitioner.Create(0, seqLen);

                    Parallel.ForEach(partitioner,
                        // 2 * qkvDim: first half is conv output, second half is the
                        // TensorPrimitives.Sigmoid scratch for thread-safe SiLU.
                        localInit: () => new float[2 * capturedQkvDim],
                        body: (range, loopState, scratch) =>
                        {
                            unsafe
                            {
                                float* ext = (float*)extPtrIP;
                                float* wt  = (float*)convWTIP;
                                float* pp  = (float*)packedPtrIP;

                                fixed (float* scratchPtr = scratch)
                                {
                                    Span<float> scratchSpan = scratch.AsSpan();
                                    Span<float> convSpan = scratchSpan.Slice(0, capturedQkvDim);
                                    Span<float> siluTmp  = scratchSpan.Slice(capturedQkvDim, capturedQkvDim);

                                    for (int s = range.Item1; s < range.Item2; s++)
                                    {
                                        ComputeConv1DRowScratch(s, capturedQkvDim, capturedKernel,
                                            ext, scratchPtr, wt);
                                        ApplySiLUInPlaceScratch(convSpan, siluTmp);

                                        float* qDst = (float*)qPtrIP + (long)s * capturedHKDim;
                                        float* kDst = (float*)kPtrIP + (long)s * capturedHKDim;
                                        float* vDst = (float*)vPtrIP + (long)s * capturedHVDim;
                                        float* zDst = (float*)zPtrIP + (long)s * capturedHVDim;
                                        float* alphaDst = (float*)alphaPtrIP + (long)s * capturedH;
                                        float* betaDst  = (float*)betaPtrIP  + (long)s * capturedH;

                                        float* zSrc;
                                        float* alphaSrc;
                                        float* betaSrc;
                                        if (pp != null)
                                        {
                                            float* row = pp + (long)s * capturedPackedDim;
                                            zSrc     = row + capturedQkvDim;
                                            betaSrc  = zSrc + capturedZDim;
                                            alphaSrc = betaSrc + capturedNumVHeads;
                                        }
                                        else
                                        {
                                            zSrc     = (float*)zBaseIP + (long)s * capturedZDim;
                                            betaSrc  = (float*)betaBaseIP + (long)s * capturedNumVHeads;
                                            alphaSrc = (float*)alphaBaseIP + (long)s * capturedNumVHeads;
                                        }

                                        long vBytes = (long)capturedHVDim * sizeof(float);
                                        Buffer.MemoryCopy(scratchPtr + 2 * capturedQkDim, vDst, vBytes, vBytes);
                                        Buffer.MemoryCopy(zSrc, zDst, vBytes, vBytes);

                                        long aBytes = (long)capturedH * sizeof(float);
                                        Buffer.MemoryCopy(alphaSrc, alphaDst, aBytes, aBytes);
                                        Buffer.MemoryCopy(betaSrc,  betaDst,  aBytes, aBytes);

                                        if (capturedNumKHeads == capturedNumVHeads)
                                        {
                                            long kBytes = (long)capturedHKDim * sizeof(float);
                                            Buffer.MemoryCopy(scratchPtr,                 qDst, kBytes, kBytes);
                                            Buffer.MemoryCopy(scratchPtr + capturedQkDim, kDst, kBytes, kBytes);
                                        }
                                        else
                                        {
                                            long perHeadBytes = (long)capturedDk * sizeof(float);
                                            for (int h = 0; h < capturedH; h++)
                                            {
                                                int srcHead = h % capturedNumKHeads;
                                                Buffer.MemoryCopy(scratchPtr + srcHead * capturedDk,
                                                    qDst + h * capturedDk, perHeadBytes, perHeadBytes);
                                                Buffer.MemoryCopy(scratchPtr + capturedQkDim + srcHead * capturedDk,
                                                    kDst + h * capturedDk, perHeadBytes, perHeadBytes);
                                            }
                                        }
                                    }
                                }
                            }
                            return scratch;
                        },
                        localFinally: _ => { });
                }

                // ------------------------------------------------------------------
                // Phase 3: CPU pre-compute of gate and beta_sig. This replaces four
                // GPU ops (add(dt_bias), softplus, mul(a_log), sigmoid) with one CPU
                // pass over [seqLen, H] - a trivially small tensor even at seqLen=4K.
                // ------------------------------------------------------------------
                float* dtBiasPtr = GetFloatPtr(_ssmDtBiasW[layer]);
                float* aLogPtr   = GetFloatPtr(_ssmAW[layer]);
                PrecomputeGateAndBetaSig(alphaPtr, betaPtr, dtBiasPtr, aLogPtr, seqLen, H);

                long tCpuPrepEnd = Stopwatch.GetTimestamp();
                _gdnChunkedCpuPrepTicks += tCpuPrepEnd - tCpuPrepStart;

                // Tell the GGML host-buffer cache that the staging buffers were written
                // on host. The chunked kernel uses backend_tensor_set internally so this
                // is a no-op safety belt for any other consumer that may read these.
                InvalidateTensorDeviceCache(qBuf);
                InvalidateTensorDeviceCache(kBuf);
                InvalidateTensorDeviceCache(vBuf);
                InvalidateTensorDeviceCache(zBuf);
                InvalidateTensorDeviceCache(alphaBuf);
                InvalidateTensorDeviceCache(betaBuf);

                Tensor state = _deltaStateTensor[layer];
                InvalidateTensorDeviceCache(state);

                long tKernelStart = Stopwatch.GetTimestamp();

                // The GGML kernel writes [T, H, D] - same memory as gated[T, H*D].
                Tensor gated3D = gated.View(seqLen, H, Dv);
                try
                {
                    // dt_bias and a_log are no longer read by the fused kernel (gate is
                    // pre-computed), but we still pass valid pointers to satisfy the
                    // non-null contract that was documented for the C ABI.
                    GgmlBasicOps.GatedDeltaNetChunked(
                        qBuf, kBuf, vBuf, zBuf,
                        alphaBuf, betaBuf,
                        state, gated3D,
                        new IntPtr(dtBiasPtr),
                        new IntPtr(aLogPtr),
                        new IntPtr(GetFloatPtr(_ssmNormW[layer])),
                        chunkSize: GdnChunkSize, eps: Config.Eps);
                }
                finally
                {
                    gated3D.Dispose();
                }

                _gdnChunkedKernelTicks += Stopwatch.GetTimestamp() - tKernelStart;

                // The kernel downloaded fresh state and gated bytes back to the host
                // buffer; downstream GGML kernels need to re-upload those bytes.
                InvalidateTensorDeviceCache(state);
                InvalidateTensorDeviceCache(gated);
            }
            finally
            {
                // Sub-views increment the underlying storage refcount; dispose them so
                // we don't leak. The persistent backing tensors (_gdnChunked*Buf) are
                // released in Dispose().
                qBuf.Dispose();
                kBuf.Dispose();
                vBuf.Dispose();
                zBuf.Dispose();
                alphaBuf.Dispose();
                betaBuf.Dispose();
            }
        }

        /// <summary>
        /// Vectorised per-row 1D convolution: computes
        /// <c>outRow = sum_ki in [0, convKernel) ext[s + ki] * wt[ki]</c> into the
        /// caller-supplied scratch buffer. Safe to invoke concurrently because each
        /// worker owns its own <paramref name="outRow"/> (thread-local in
        /// <c>Parallel.ForEach</c>'s <c>localInit</c>).
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ComputeConv1DRowScratch(int s, int qkvDim, int convKernel,
            float* ext, float* outRow, float* wt)
        {
            int vLen = Vector<float>.Count;

            // Initialise with the first tap so we don't need a separate zeroing pass.
            {
                float* inRow = ext + (long)s * qkvDim;
                float* wtRow = wt;
                int ch = 0;
                for (; ch <= qkvDim - vLen; ch += vLen)
                {
                    var iv = LdVecLocal(inRow + ch);
                    var wv = LdVecLocal(wtRow + ch);
                    StVecLocal(outRow + ch, iv * wv);
                }
                for (; ch < qkvDim; ch++)
                    outRow[ch] = inRow[ch] * wtRow[ch];
            }

            for (int ki = 1; ki < convKernel; ki++)
            {
                float* inRow = ext + (long)(s + ki) * qkvDim;
                float* wtRow = wt + (long)ki * qkvDim;

                int ch = 0;
                for (; ch <= qkvDim - vLen; ch += vLen)
                {
                    var acc = LdVecLocal(outRow + ch);
                    var iv = LdVecLocal(inRow + ch);
                    var wv = LdVecLocal(wtRow + ch);
                    StVecLocal(outRow + ch, acc + iv * wv);
                }
                for (; ch < qkvDim; ch++)
                    outRow[ch] += inRow[ch] * wtRow[ch];
            }
        }

        /// <summary>
        /// Pre-compute gate = <c>-A_log[h] * softplus(alpha_raw[s, h] + dt_bias[h])</c>
        /// and beta_sig = <c>sigmoid(beta_raw[s, h])</c> on the CPU, overwriting the
        /// already-packed alpha and beta staging buffers in place. The fused GGML
        /// kernel reads these as the final gate / sigmoid-beta inputs and skips the
        /// equivalent add / softplus / mul / sigmoid ops on the device - four kernel
        /// dispatches saved per layer without touching the GPU hot path.
        ///
        /// The work is O(seqLen * H) which is trivially small (~32 K floats even at
        /// seqLen=1024, H=32) so we keep this scalar for clarity. TensorPrimitives
        /// handles the sigmoid(beta) pass in one vectorised call on every modern ISA.
        /// </summary>
        private unsafe void PrecomputeGateAndBetaSig(
            float* alphaPtr, float* betaPtr, float* dtBiasPtr, float* aLogPtr,
            int seqLen, int H)
        {
            int total = seqLen * H;

            // Beta: in-place sigmoid.
            Span<float> betaSpan = new Span<float>(betaPtr, total);
            TensorPrimitives.Sigmoid(betaSpan, betaSpan);

            // Gate = a_log * softplus(alpha_raw + dt_bias). The stored a_log tensor
            // already holds -A_log so no explicit negation is needed.
            for (int s = 0; s < seqLen; s++)
            {
                float* alphaRow = alphaPtr + (long)s * H;
                for (int h = 0; h < H; h++)
                {
                    float x = alphaRow[h] + dtBiasPtr[h];
                    alphaRow[h] = SoftplusScalar(x) * aLogPtr[h];
                }
            }
        }

        /// <summary>
        /// Lazily (re)allocate the chunked-prefill staging tensors so they cover at least
        /// <paramref name="seqLen"/> rows. The buffers are sized to the largest seqLen we
        /// have seen and reused for every layer in the same forward pass and every
        /// subsequent forward pass with seqLen <= capacity.
        /// </summary>
        private void EnsureChunkedStagingBuffers(int seqLen, int H, int Dk, int Dv)
        {
            if (_gdnChunkedQBuf != null && _gdnChunkedBufCapacity >= seqLen)
                return;

            // Free old buffers (if any) before resizing upward.
            _gdnChunkedQBuf?.Dispose();
            _gdnChunkedKBuf?.Dispose();
            _gdnChunkedVBuf?.Dispose();
            _gdnChunkedZBuf?.Dispose();
            _gdnChunkedAlphaBuf?.Dispose();
            _gdnChunkedBetaBuf?.Dispose();

            _gdnChunkedQBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dk);
            _gdnChunkedKBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dk);
            _gdnChunkedVBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dv);
            _gdnChunkedZBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dv);
            _gdnChunkedAlphaBuf = new Tensor(_allocator, DType.Float32, seqLen, H);
            _gdnChunkedBetaBuf  = new Tensor(_allocator, DType.Float32, seqLen, H);
            _gdnChunkedBufCapacity = seqLen;
        }

        /// <summary>
        /// Make sure the parallel-Conv1D scratch buffers cover at least <paramref name="seqLen"/>
        /// tokens for a projection of width <paramref name="qkvDim"/>. The buffers grow in
        /// place and are reused across layers (32 in Qwen3.5) and benchmark runs; we never
        /// reallocate for the same shape.
        /// </summary>
        private void EnsureConv1DScratchBuffers(int seqLen, int qkvDim, int convDim, int H)
        {
            long extendedLen = (long)(convDim + seqLen) * qkvDim;
            if (_gdnConvExtendedBuf == null || _gdnConvExtendedBuf.Length < extendedLen)
            {
                _gdnConvExtendedBuf = new float[extendedLen];
                _gdnConvExtendedCapacity = seqLen;
            }

            // _gdnSiluTempBuf is only used by the per-token step path (sized `qkvDim`).
            // The chunked prefill path uses per-worker scratch from localInit, so we do
            // not need to grow this buffer for parallel SiLU any more.
            if (_gdnSiluTempBuf == null || _gdnSiluTempBuf.Length < qkvDim)
                _gdnSiluTempBuf = new float[qkvDim];
        }

        /// <summary>
        /// Single-token GatedDeltaNet step.
        ///
        /// Optimizations vs the original implementation:
        /// 1. Convolution uses a circular buffer keyed by `_convStateWriteIdx[layer]`. This
        ///    eliminates the O(convDim * qkvDim) `Array.Copy` shift that ran every token; the
        ///    new step writes the latest input row into one ring slot and reads previous
        ///    slots in modular order. For convKernel=4 and qkvDim ~ 16k this saves ~50k float
        ///    copies per token per recurrent layer.
        /// 2. The conv1d weight is pre-transposed to `[kernelSize, qkvDim]` so the inner
        ///    SIMD loop accumulates `state_tap * weight_tap` over a whole channel block at
        ///    once via System.Numerics vectors instead of scalar accumulation per channel.
        /// 3. SiLU on the conv output uses the SIMD vector helper from VecApplySiLU.
        /// 4. Per-head state updates can run on a thread pool when `_numVHeads >= 16`, which
        ///    matches MoE/qwen3-next configurations that use 32 V-heads.
        /// </summary>
        private unsafe void GatedDeltaNetStep(float* qkvPtr, float* zPtr, float* betaPtr, float* alphaPtr,
            int layer, int qkvDim, int qkDim, int vDim,
            float[] convWT, float* dtBiasPtr, float* aPtr, float* ssmNormPtr,
            float* gatedOutPtr)
        {
            int convDim = _convKernel - 1;
            float[] convState = _convState[layer];
            int writeIdx = _convStateWriteIdx[layer];

            fixed (float* convOutPtr = _gdnConvOutBuf)
            fixed (float* qBase = _gdnQ, kBase = _gdnK, vBase = _gdnV)
            {
                ComputeConv1DStep(qkvPtr, qkvDim, convDim, writeIdx, convState, convWT, _gdnConvOutBuf, convOutPtr);

                if (convDim > 0)
                {
                    fixed (float* statePtr = convState)
                    {
                        Buffer.MemoryCopy(qkvPtr, statePtr + writeIdx * qkvDim,
                            qkvDim * sizeof(float), qkvDim * sizeof(float));
                    }
                    _convStateWriteIdx[layer] = (writeIdx + 1) % convDim;
                }

                Buffer.MemoryCopy(convOutPtr, qBase, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(convOutPtr + qkDim, kBase, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(convOutPtr + 2 * qkDim, vBase, vDim * sizeof(float), vDim * sizeof(float));
            }

            float[] qActive = _gdnQ;
            float[] kActive = _gdnK;
            if (_numKHeads != _numVHeads)
            {
                qActive = _gdnQExp;
                kActive = _gdnKExp;
                for (int h = 0; h < _numVHeads; h++)
                {
                    int srcHead = h % _numKHeads;
                    Array.Copy(_gdnQ, srcHead * _headKDim, qActive, h * _headKDim, _headKDim);
                    Array.Copy(_gdnK, srcHead * _headKDim, kActive, h * _headKDim, _headKDim);
                }
            }

            L2NormalizePerHead(qActive, _numVHeads, _headKDim);
            L2NormalizePerHead(kActive, _numVHeads, _headKDim);

            float qScale = 1.0f / MathF.Sqrt(_headVDim);
            int totalQK = _numVHeads * _headKDim;
            fixed (float* qPtr = qActive)
                VecScale(qPtr, qScale, totalQK);

            // Capture pointers/values for parallel head update
            Tensor state = _deltaStateTensor[layer];
            float* statePtrBase = GetFloatPtr(state);
            int statePerHead = _headVDim * _headKDim;
            int headKDim = _headKDim;
            int headVDim = _headVDim;
            float eps = Config.Eps;

            fixed (float* qPin = qActive, kPin = kActive, vPin = _gdnV,
                          deltaPin = _gdnDelta, corePin = _gdnCore)
            {
                float* qPtr = qPin;
                float* kPtr = kPin;
                float* vPtr = vPin;
                float* deltaPtr = deltaPin;
                float* corePtr = corePin;

                if (_gdnParallelHeads)
                {
                    // Local copies because pointers cannot be captured by reference in lambdas.
                    float* qPtrLocal = qPtr;
                    float* kPtrLocal = kPtr;
                    float* vPtrLocal = vPtr;
                    float* deltaPtrLocal = deltaPtr;
                    float* corePtrLocal = corePtr;
                    float* statePtrLocal = statePtrBase;
                    float* zPtrLocal = zPtr;
                    float* dtBiasPtrLocal = dtBiasPtr;
                    float* aPtrLocal = aPtr;
                    float* alphaPtrLocal = alphaPtr;
                    float* betaPtrLocal = betaPtr;
                    float* ssmNormPtrLocal = ssmNormPtr;
                    float* gatedOutPtrLocal = gatedOutPtr;

                    Parallel.For(0, _numVHeads, h =>
                    {
                        ProcessHead(h, qPtrLocal, kPtrLocal, vPtrLocal,
                            deltaPtrLocal, corePtrLocal, statePtrLocal,
                            zPtrLocal, dtBiasPtrLocal, aPtrLocal,
                            alphaPtrLocal, betaPtrLocal, ssmNormPtrLocal,
                            gatedOutPtrLocal,
                            headKDim, headVDim, statePerHead, eps);
                    });
                }
                else
                {
                    for (int h = 0; h < _numVHeads; h++)
                    {
                        ProcessHead(h, qPtr, kPtr, vPtr, deltaPtr, corePtr, statePtrBase,
                            zPtr, dtBiasPtr, aPtr, alphaPtr, betaPtr, ssmNormPtr, gatedOutPtr,
                            headKDim, headVDim, statePerHead, eps);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ProcessHead(int h,
            float* qPtr, float* kPtr, float* vPtr,
            float* deltaPtr, float* corePtr, float* statePtrBase,
            float* zPtr, float* dtBiasPtr, float* aPtr,
            float* alphaPtr, float* betaPtr, float* ssmNormPtr,
            float* gatedOutPtr,
            int headKDim, int headVDim, int statePerHead, float eps)
        {
            float* stateHead = statePtrBase + h * statePerHead;
            float* qHead = qPtr + h * headKDim;
            float* kHead = kPtr + h * headKDim;
            float* vHead = vPtr + h * headVDim;
            float* deltaHead = deltaPtr + h * headVDim;
            float* coreHead = corePtr + h * headVDim;
            float* zHead = zPtr + h * headVDim;
            float* gatedHead = gatedOutPtr + h * headVDim;

            float alphaBiased = alphaPtr[h] + dtBiasPtr[h];
            float gateH = SoftplusScalar(alphaBiased) * aPtr[h];
            VecScale(stateHead, MathF.Exp(gateH), statePerHead);

            float betaH = SigmoidScalar(betaPtr[h]);

            // delta_row = (v_row - dot(state_row, k)) * beta
            for (int row = 0; row < headVDim; row++)
            {
                float kvMem = VecDot(stateHead + row * headKDim, kHead, headKDim);
                deltaHead[row] = (vHead[row] - kvMem) * betaH;
            }

            // state_row += k * delta_row;  core_row = dot(state_row, q)
            for (int row = 0; row < headVDim; row++)
            {
                float* stateRow = stateHead + row * headKDim;
                VecScaleAdd(stateRow, kHead, deltaHead[row], headKDim);
                coreHead[row] = VecDot(stateRow, qHead, headKDim);
            }

            float rmsInv = 1.0f / MathF.Sqrt((VecSumSq(coreHead, headVDim) / headVDim) + eps);
            for (int i = 0; i < headVDim; i++)
                gatedHead[i] = coreHead[i] * rmsInv * ssmNormPtr[i] * SiLUScalar(zHead[i]);
        }

        /// <summary>
        /// Vectorized 1D convolution step using a circular state buffer and a transposed
        /// weight layout. For each channel ch and kernel tap ki, we read the state slot
        /// from a logical index that wraps around the ring, and multiply by the contiguous
        /// weight tap row. SIMD vectorization runs along the channel dimension. After the
        /// reduction, SiLU is applied in-place over the channel vector.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void ComputeConv1DStep(float* qkvPtr, int qkvDim, int convDim,
            int writeIdx, float[] convState, float[] convWT, float[] convOutArray, float* convOutPtr)
        {
            int vLen = Vector<float>.Count;

            fixed (float* statePtr = convState, wtPtr = convWT)
            {
                int firstAccumTap = 1;

                if (convDim > 0)
                {
                    // Kernel taps 0..convDim-1 sample from the previous (ring buffered) state.
                    // Initialise from the first tap directly; this saves a full zero pass plus
                    // one read/modify/write pass over qkvDim on every decode step.
                    int slot = writeIdx;
                    float* statePos = statePtr + slot * qkvDim;
                    float* wtPos = wtPtr;

                    int ch = 0;
                    for (; ch <= qkvDim - vLen; ch += vLen)
                    {
                        var sv = LdVecLocal(statePos + ch);
                        var wv = LdVecLocal(wtPos + ch);
                        StVecLocal(convOutPtr + ch, sv * wv);
                    }
                    for (; ch < qkvDim; ch++)
                        convOutPtr[ch] = statePos[ch] * wtPos[ch];
                }
                else
                {
                    firstAccumTap = 0;
                }

                for (int ki = firstAccumTap; ki < convDim; ki++)
                {
                    int slot = (writeIdx + ki) % convDim;
                    float* statePos = statePtr + slot * qkvDim;
                    float* wtPos = wtPtr + (long)ki * qkvDim;

                    int ch = 0;
                    for (; ch <= qkvDim - vLen; ch += vLen)
                    {
                        var acc = LdVecLocal(convOutPtr + ch);
                        var sv = LdVecLocal(statePos + ch);
                        var wv = LdVecLocal(wtPos + ch);
                        StVecLocal(convOutPtr + ch, acc + sv * wv);
                    }
                    for (; ch < qkvDim; ch++)
                        convOutPtr[ch] += statePos[ch] * wtPos[ch];
                }

                // Final tap reads the new input qkvPtr. If there is no historical state
                // (convKernel=1), initialise from this tap directly.
                {
                    float* wtPos = wtPtr + (long)convDim * qkvDim;
                    int ch = 0;
                    if (convDim > 0)
                    {
                        for (; ch <= qkvDim - vLen; ch += vLen)
                        {
                            var acc = LdVecLocal(convOutPtr + ch);
                            var iv = LdVecLocal(qkvPtr + ch);
                            var wv = LdVecLocal(wtPos + ch);
                            StVecLocal(convOutPtr + ch, acc + iv * wv);
                        }
                        for (; ch < qkvDim; ch++)
                            convOutPtr[ch] += qkvPtr[ch] * wtPos[ch];
                    }
                    else
                    {
                        for (; ch <= qkvDim - vLen; ch += vLen)
                        {
                            var iv = LdVecLocal(qkvPtr + ch);
                            var wv = LdVecLocal(wtPos + ch);
                            StVecLocal(convOutPtr + ch, iv * wv);
                        }
                        for (; ch < qkvDim; ch++)
                            convOutPtr[ch] = qkvPtr[ch] * wtPos[ch];
                    }
                }
            }

            ApplySiLUInPlace(convOutArray, convOutPtr, qkvDim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVecLocal(float* p) =>
            TensorComputePrimitives.LoadVector(p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVecLocal(float* p, Vector<float> v) =>
            TensorComputePrimitives.StoreVector(p, v);

        /// <summary>
        /// Apply SiLU(x) = x * sigmoid(x) in place using the hardware-accelerated
        /// <see cref="TensorPrimitives.Sigmoid(System.ReadOnlySpan{float}, System.Span{float})"/>
        /// followed by an element-wise multiply.
        ///
        /// On Apple Silicon and modern x86 this reduces the per-call cost from a scalar
        /// MathF.Exp loop (~3 ns/element) to a vectorised polynomial-based sigmoid
        /// (typically &lt;0.5 ns/element). For Qwen3.5 prefill that turns ~16 MFLOPs of
        /// scalar work per layer into a SIMD pass that disappears into the noise.
        ///
        /// Uses the shared <c>_gdnSiluTempBuf</c> scratch and therefore MUST NOT be
        /// called concurrently from multiple threads - the chunked-prefill parallel
        /// loop uses <see cref="ApplySiLUInPlaceScratch"/> with a per-worker scratch.
        /// </summary>
        private void ApplySiLUInPlace(float[] dataArray, int n)
        {
            if (_gdnSiluTempBuf == null || _gdnSiluTempBuf.Length < n)
                _gdnSiluTempBuf = new float[n];

            ReadOnlySpan<float> input = dataArray.AsSpan(0, n);
            Span<float> tmp = _gdnSiluTempBuf.AsSpan(0, n);
            Span<float> output = dataArray.AsSpan(0, n);

            // sigmoid(x) -> tmp, then output = input * tmp.
            TensorPrimitives.Sigmoid(input, tmp);
            TensorPrimitives.Multiply(input, tmp, output);
        }

        /// <summary>
        /// Thread-safe variant of <see cref="ApplySiLUInPlace(float[], int)"/> that
        /// writes <c>sigmoid(x)</c> into the caller-owned <paramref name="tmp"/>
        /// scratch. Each Parallel.For worker keeps its own scratch so we can fuse
        /// Conv1D + SiLU + packing memcpys into one pass over the tokens.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ApplySiLUInPlaceScratch(Span<float> data, Span<float> tmp) =>
            TensorComputePrimitives.ApplySiLUInPlace(data, tmp);

        /// <summary>
        /// Pointer-based wrapper used inside the per-token loop where we already have a
        /// <c>fixed</c> on the conv output buffer. Marshals to the array overload to avoid
        /// duplicating the SIMD plumbing.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void ApplySiLUInPlace(float[] dataArray, float* dataPtr, int n)
        {
            // The pointer is from a fixed on dataArray, so the array overload is safe to call:
            // its TensorPrimitives spans walk the same backing storage.
            ApplySiLUInPlace(dataArray, n);
        }

        private unsafe void L2NormalizePerHead(float[] data, int numHeads, int headDim)
        {
            fixed (float* ptr = data)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = ptr + h * headDim;
                    float inv = 1.0f / MathF.Sqrt(VecSumSq(head, headDim) + Config.Eps);
                    VecScale(head, inv, headDim);
                }
            }
        }

        private static float SigmoidScalar(float x) => TensorComputePrimitives.Sigmoid(x);

        private static float SiLUScalar(float x) => TensorComputePrimitives.SiLU(x);

        private static float SoftplusScalar(float x) => TensorComputePrimitives.Softplus(x);
    }
}
