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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // ====================================================================
        // GDN configuration constants and environment overrides
        // ====================================================================

        // Minimum prefill length at which the fused chunked GatedDeltaNet GGML kernel
        // beats the per-token CPU loop. The chunked kernel pads to a multiple of
        // GdnChunkSize (64) and dispatches once per layer; for very short prefills the
        // padding overhead and the host->device copies dominate. Set
        // GDN_CHUNK_PREFILL_MIN_SEQ_LEN=N to override (e.g. =1 to always use it,
        // =1000000 to disable).
        private static readonly int GdnChunkedPrefillMinSeqLenEnv = ResolveGdnChunkedPrefillMinSeqLen();

        private static int ResolveGdnChunkedPrefillMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("GDN_CHUNK_PREFILL_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return GdnChunkSize;
        }

        private static readonly bool GdnChunkedPrefillDisabledEnv =
            string.Equals(Environment.GetEnvironmentVariable("GDN_DISABLE_CHUNKED_PREFILL"), "1",
                StringComparison.Ordinal);

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
        private Tensor[] _deltaStateTensor; // [layer]: Tensor[numVHeads, headVDim, headKDim]

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
        private int _gdnChunkPrefillThreshold = GdnChunkedPrefillMinSeqLenEnv;
        private bool _gdnDisableChunkedPrefill = GdnChunkedPrefillDisabledEnv;
        private long _gdnChunkedTicks;       // Total time spent in the chunked path
        private long _gdnPerTokenTicks;      // Total time spent in the per-token path (prefill only)
        private int _gdnChunkedCalls;        // Number of times the chunked path has been used
        private long _gdnChunkedCpuPrepTicks; // Time spent in CPU prep (Conv1D + memcpy + SiLU)
        private long _gdnChunkedKernelTicks; // Time spent in the GGML kernel call (incl. sync + download)

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
            _deltaStateTensor[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
            Ops.Fill(_deltaStateTensor[l], 0);
        }

        /// <summary>
        /// Allocate the per-recurrent-layer cache arrays so InitCaches can fill them
        /// in its layer loop. Called from InitCaches.
        /// </summary>
        private void InitGdnCacheArrays(int numLayers)
        {
            _convState = new float[numLayers][];
            _convStateWriteIdx = new int[numLayers];
            _deltaStateTensor = new Tensor[numLayers];
        }

        /// <summary>
        /// Reset the per-layer recurrent state. Called from ResetKVCache.
        /// </summary>
        private void ResetGdnLayerCache(int l)
        {
            Array.Clear(_convState[l]);
            _convStateWriteIdx[l] = 0;
            Ops.Fill(_deltaStateTensor[l], 0);
        }

        /// <summary>
        /// Pre-allocate the small pinned scratch buffers used by the per-token GDN
        /// decode step. Called from the constructor.
        /// </summary>
        private void InitGDNBuffers()
        {
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
            if (_deltaStateTensor != null)
                foreach (var t in _deltaStateTensor) t?.Dispose();

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

            if (_gdnChunkedCalls == 0 && _gdnPerTokenCalls == 0)
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
            Console.WriteLine($"    per-token path: {_gdnPerTokenCalls} prefill calls, {perTokenMs:F0} ms total" +
                (_gdnPerTokenCalls > 0 ? $", {perTokenMs / _gdnPerTokenCalls:F2} ms/call" : ""));
            if (_gdnDisableChunkedPrefill)
                Console.WriteLine($"    (chunked path disabled at runtime)");
            else
                Console.WriteLine($"    (chunked threshold: seqLen >= {_gdnChunkPrefillThreshold}, chunkSize {GdnChunkSize})");
        }

        /// <summary>
        /// Reset all GDN timing counters. Called from ResetKVCache after the shared
        /// counters are reset.
        /// </summary>
        private void ResetGdnTimingCounters()
        {
            _gdnChunkedTicks = 0;
            _gdnPerTokenTicks = 0;
            _gdnChunkedCalls = 0;
            _gdnChunkedCpuPrepTicks = 0;
            _gdnChunkedKernelTicks = 0;
            _gdnPerTokenCalls = 0;
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
            // Fused pre-norm + GatedDeltaNet input projection inside GatedDeltaNet, and fused
            // output projection + residual via TryLinearAddInto when possible.
            Tensor attnOut = GatedDeltaNet(hidden, _attnNormW[layer], layer, seqLen, residual: hidden);
            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            long ffnStart = profilePrefill ? Stopwatch.GetTimestamp() : 0;

            Tensor ffnOut;
            if (_isMoeLayer != null && _isMoeLayer[layer])
            {
                // Decode hot path: do RMSNorm on CPU into the pre-allocated input buffer.
                Tensor normed2;
                bool ownsNormed2 = true;
                if (seqLen == 1 && IsGgmlBackend && _moeTokenInput != null
                    && _moeTokenInput.Sizes[1] == Config.HiddenSize
                    && _postAttnNormW[layer] != null)
                {
                    RMSNormToBufferCpu(_moeTokenInput, hidden, _postAttnNormW[layer], Config.HiddenSize, Config.Eps);
                    normed2 = _moeTokenInput;
                    ownsNormed2 = false;
                }
                else
                {
                    normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);
                }

                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                {
                    ffnOut = null;
                }
                else
                {
                    ffnOut = MoEForward(normed2, layer, seqLen);
                }
                if (ownsNormed2)
                    normed2.Dispose();
            }
            else
            {
                ffnOut = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
            }

            if (ffnOut != null)
            {
                Ops.Add(hidden, hidden, ffnOut);
                ffnOut.Dispose();
            }

            if (profilePrefill)
                _prefillRecFfnTicks += Stopwatch.GetTimestamp() - ffnStart;

            return hidden;
        }

        /// <summary>
        /// GatedDeltaNet recurrent step with batched input/output projections.
        /// Prefill projects the whole chunk once, then walks the recurrent state token-by-token.
        /// Decode follows the same path with seqLen=1, avoiding several tiny GGML dispatches.
        /// </summary>
        private unsafe Tensor GatedDeltaNet(Tensor input, Tensor inputNormW, int layer, int seqLen,
            Tensor residual = null)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool profilePrefill = _profilePrefillStages && seqLen > 1;
            long stageStart = profilePrefill ? t0 : 0;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int zDim = _headVDim * _numVHeads;
            int packedDim = qkvDim + zDim + _numVHeads * 2;

            // Fused input norm + packed input projection (single GGML kernel). Decode reuses
            // a pre-allocated [1, packedDim] buffer to avoid one tensor allocation per layer
            // per token.
            Tensor packedInput = null;
            bool ownsPackedInput = true;
            if (inputNormW != null && _ssmInProjQW[layer] != null && IsGgmlBackend)
            {
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

            if (packedInput != null)
            {
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

                qkvBase = GetFloatPtr(qkvRaw);
                zBase = GetFloatPtr(zRaw);
                betaBase = GetFloatPtr(betaRaw);
                alphaBase = GetFloatPtr(alphaRaw);
            }

            // Pre-resolved layer constants (cached at construction; no dictionary lookup here).
            float* dtBiasPtr = GetFloatPtr(_ssmDtBiasW[layer]);
            float* aPtr = GetFloatPtr(_ssmAW[layer]);
            float* ssmNormPtr = GetFloatPtr(_ssmNormW[layer]);

            Tensor gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
            float* gatedBase = GetFloatPtr(gated);

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

            if (useChunked)
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
            if (profilePrefill) { long now = Stopwatch.GetTimestamp(); _prefillRecCoreTicks += now - stageStart; stageStart = now; }

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

            VecZero(convOutPtr, qkvDim);

            fixed (float* statePtr = convState, wtPtr = convWT)
            {
                // Kernel taps 0..convDim-1 sample from the previous (ring buffered) state.
                // The element written most recently is at slot ((writeIdx + convDim - 1) % convDim)
                // and corresponds to the newest historical input (kernel position convDim-1).
                for (int ki = 0; ki < convDim; ki++)
                {
                    int slot = (writeIdx + ki) % convDim;
                    float* statePos = statePtr + slot * qkvDim;
                    float* wtPos = wtPtr + ki * qkvDim;

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

                // Final tap reads the new input qkvPtr.
                {
                    float* wtPos = wtPtr + convDim * qkvDim;
                    int ch = 0;
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
            }

            ApplySiLUInPlace(convOutArray, convOutPtr, qkvDim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVecLocal(float* p) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVecLocal(float* p, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref *(byte*)p, v);

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
        private static void ApplySiLUInPlaceScratch(Span<float> data, Span<float> tmp)
        {
            TensorPrimitives.Sigmoid(data, tmp);
            TensorPrimitives.Multiply(data, tmp, data);
        }

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

        private static float SigmoidScalar(float x)
        {
            if (x >= 0)
            {
                float e = MathF.Exp(-x);
                return 1.0f / (1.0f + e);
            }

            float en = MathF.Exp(x);
            return en / (1.0f + en);
        }

        private static float SiLUScalar(float x) => x * SigmoidScalar(x);

        private static float SoftplusScalar(float x)
        {
            if (x > 20f)
                return x;
            if (x < -20f)
                return MathF.Exp(x);
            return MathF.Log(1.0f + MathF.Exp(x));
        }
    }
}
