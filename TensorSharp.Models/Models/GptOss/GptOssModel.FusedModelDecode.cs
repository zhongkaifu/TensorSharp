// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Whole-model fused decode for GPT-OSS: one GGML graph dispatch per token
    /// covering every layer's attention + MoE FFN plus the final norm and LM head
    /// (TSGgml_GptOssModelDecode).
    ///
    /// The per-layer fused kernels this replaces still cost ~48 graph submissions
    /// per token — each with its own ggml context, buffer allocation and full
    /// device synchronise — which left the GPU at 40-62% utilisation waiting on
    /// the host between layers. Folding the token into a single (CUDA-graph
    /// capturable) graph removes that entirely, and folding in the LM head keeps
    /// the 201K-vocab projection inside the same replay.
    ///
    /// The kernel writes the KV cache device-side and does NOT mirror it back to
    /// host memory per token, so <see cref="_kvCacheHostDirty"/> is set and any
    /// host reader (KV snapshot, cache growth, the legacy per-op attention path)
    /// goes through <see cref="EnsureKvCacheHostSynchronized"/> first.
    /// </summary>
    public partial class GptOssModel
    {
        // TS_GPTOSS_MODEL_DECODE=0 falls back to the per-layer fused path.
        private static readonly bool FusedModelDecodeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_GPTOSS_MODEL_DECODE"), "0", StringComparison.Ordinal);

        // TS_GPTOSS_MODEL_PREFILL=0 falls back to the per-layer prefill path.
        private static readonly bool FusedModelPrefillEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_GPTOSS_MODEL_PREFILL"), "0", StringComparison.Ordinal);

        private GptOssLayerDecodeArgs[] _modelDecodeArgs;
        private bool _modelDecodeUnavailable;
        private bool _modelPrefillUnavailable;
        private bool _kvCacheHostDirty;
        private float[] _foldLogitsBuffer;
        private GCHandle _foldLogitsHandle;
        private IntPtr _foldLogitsPtr;

        // Pinned host arrays the kernel binds by pointer. Pinning once per array
        // keeps the pointers stable so the native cacheable-buffer path recognises
        // them across calls (and so the captured graph keeps its bindings).
        private readonly List<GCHandle> _pinnedDecodeArrays = new List<GCHandle>();

        // Per-layer, per-expert gate/up biases split out of _layerGateUpBiasStacked
        // into the [n_ff, num_experts] layout ggml_add_id expects for the separate
        // gate/up expert weights.
        private float[][] _layerGateBiasStacked;
        private float[][] _layerUpBiasStacked;

        private IntPtr PinArray(float[] array)
        {
            if (array == null) return IntPtr.Zero;
            var handle = GCHandle.Alloc(array, GCHandleType.Pinned);
            _pinnedDecodeArrays.Add(handle);
            return handle.AddrOfPinnedObject();
        }

        private void FreePinnedDecodeArrays()
        {
            foreach (var h in _pinnedDecodeArrays)
            {
                if (h.IsAllocated) h.Free();
            }
            _pinnedDecodeArrays.Clear();
        }

        /// <summary>
        /// Splits the interleaved [2*n_ff, num_experts] gate||up bias table into the
        /// two [n_ff, num_experts] tables the separate-projection expert path needs.
        /// Built once (the per-call split the fused MoE prefill used to do allocated
        /// two arrays per layer per token).
        /// </summary>
        private void EnsureSplitExpertBiases()
        {
            if (_layerGateBiasStacked != null || _layerGateUpBiasStacked == null)
                return;

            int numLayers = Config.NumLayers;
            int nFf = _expertFfnLength;
            _layerGateBiasStacked = new float[numLayers][];
            _layerUpBiasStacked = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                float[] fused = _layerGateUpBiasStacked[l];
                if (fused == null) continue;
                var gate = new float[nFf * _numExperts];
                var up = new float[nFf * _numExperts];
                for (int e = 0; e < _numExperts; e++)
                {
                    Array.Copy(fused, e * 2 * nFf, gate, e * nFf, nFf);
                    Array.Copy(fused, e * 2 * nFf + nFf, up, e * nFf, nFf);
                }
                _layerGateBiasStacked[l] = gate;
                _layerUpBiasStacked[l] = up;
            }
        }

        /// <summary>
        /// True when the whole-model decode kernel can serve this step. Checked
        /// before the KV host sync is skipped, so the answer must not depend on
        /// anything the kernel discovers at runtime.
        /// </summary>
        private bool WillUseFusedModelDecode(int seqLen)
        {
            if (!FusedModelDecodeEnabled || _modelDecodeUnavailable || IsTensorParallel)
                return false;
            // MoE CPU offload stays on this path: the native graph is segmented at
            // each offloaded layer's router so the host multiplies those experts
            // while attention, norms and the LM head keep running as one fused
            // graph (see tsg::HostMoeSegment).
            if (seqLen != 1 || !IsGgmlBackend || _layerStackedReady == 0)
                return false;
            int kvType = _kvCacheDtype.GgmlType();
            return kvType == 0 /* F32 */ || kvType == 1 /* F16 */;
        }

        /// <summary>
        /// The logits buffer the folded LM head writes into, pinned once so the
        /// native side can write it directly (and so the pointer stays stable
        /// across the captured replays).
        /// </summary>
        private float[] EnsureFoldLogitsBuffer()
        {
            if (_foldLogitsBuffer != null && _foldLogitsBuffer.Length == Config.VocabSize)
                return _foldLogitsBuffer;
            if (_foldLogitsHandle.IsAllocated)
                _foldLogitsHandle.Free();
            _foldLogitsBuffer = new float[Config.VocabSize];
            _foldLogitsHandle = GCHandle.Alloc(_foldLogitsBuffer, GCHandleType.Pinned);
            _foldLogitsPtr = _foldLogitsHandle.AddrOfPinnedObject();
            return _foldLogitsBuffer;
        }

        /// <summary>
        /// Builds (once) the per-layer descriptor array. Returns false when any
        /// layer is missing a weight the kernel needs, in which case the fused
        /// model decode is permanently disabled for this model instance.
        /// </summary>
        // ---- F16 prefill-GEMM weight copies (TS_GPTOSS_PREFILL_F16=1) ----
        // Prefill is compute-bound: at 2048-token chunks quantized MMQ runs the
        // big GEMMs well below the F16 tensor-core rate, and forcing cuBLAS on
        // quantized weights loses to the per-chunk dequant. With VRAM headroom
        // we dequantize the seven heavy weights ONCE per layer (Q8_0 -> F16 is
        // exact: int8 x f16-scale fits an F16 significand) and the prefill
        // kernel binds them; decode stays on the bandwidth-friendly quantized
        // copies. Costs ~2 host+device bytes per parameter.
        private static readonly bool PrefillF16WeightsEnabled =
            Environment.GetEnvironmentVariable("TS_GPTOSS_PREFILL_F16") == "1";
        private List<IntPtr> _f16PrefillBuffers;

        private static unsafe void DequantRowToHalf(
            int ggmlType, IntPtr src, long rowBytes, int row, float[] scratch, IntPtr dst, long ne0)
        {
            fixed (float* fb = scratch)
            {
                ManagedQuantizedOps.DequantizeRowToFloat32(
                    ggmlType, (IntPtr)((byte*)src + (long)row * rowBytes), fb, ne0);
                System.Half* hp = (System.Half*)dst + (long)row * ne0;
                for (long i = 0; i < ne0; i++)
                    hp[i] = (System.Half)fb[i];
            }
        }

        private unsafe IntPtr BuildF16WeightCopy(IntPtr src, int ggmlType, long ne0, long rows)
        {
            if (src == IntPtr.Zero || rows <= 0 || ne0 <= 0)
                return IntPtr.Zero;
            if (ggmlType == (int)GgmlTensorType.F16)
                return IntPtr.Zero;   // already the fast-GEMM type; use the original
            if (!ManagedQuantizedOps.SupportsDequantization((GgmlTensorType)ggmlType))
                return IntPtr.Zero;
            long elems = ne0 * rows;
            IntPtr dst;
            try { dst = (IntPtr)NativeMemory.AlignedAlloc((nuint)(elems * 2), 64); }
            catch (OutOfMemoryException) { return IntPtr.Zero; }
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, ne0);
            try
            {
                Parallel.For(0, checked((int)rows),
                    () => new float[ne0],
                    (r, _, scratch) => { DequantRowToHalf(ggmlType, src, rowBytes, r, scratch, dst, ne0); return scratch; },
                    _ => { });
            }
            catch (Exception)
            {
                NativeMemory.AlignedFree((void*)dst);
                return IntPtr.Zero;
            }
            (_f16PrefillBuffers ??= new List<IntPtr>()).Add(dst);
            return dst;
        }

        private unsafe void FreeF16PrefillBuffers()
        {
            if (_f16PrefillBuffers == null) return;
            foreach (var p in _f16PrefillBuffers)
            {
                // Drop the resident device copy keyed by this host pointer
                // before the memory goes away.
                GgmlBasicOps.InvalidateHostBuffer(p);
                NativeMemory.AlignedFree((void*)p);
            }
            _f16PrefillBuffers = null;
        }

        private unsafe bool TryBuildModelDecodeArgs()
        {
            if (_modelDecodeArgs != null)
                return true;
            if (_modelDecodeUnavailable)
                return false;

            EnsureSplitExpertBiases();

            int numLayers = Config.NumLayers;
            var args = new GptOssLayerDecodeArgs[numLayers];
            int structBytes = Marshal.SizeOf<GptOssLayerDecodeArgs>();

            for (int l = 0; l < numLayers; l++)
            {
                string[] wn = _layerNames[l];
                if (!_quantWeights.TryGetValue(wn[1], out var qkvQw)) { _modelDecodeUnavailable = true; break; }
                if (!_quantWeights.TryGetValue(wn[3], out var oQw)) { _modelDecodeUnavailable = true; break; }
                if (!_weights.TryGetValue(wn[0], out var attnNormW)) { _modelDecodeUnavailable = true; break; }
                if (!_weights.TryGetValue(wn[5], out var postAttnNormW)) { _modelDecodeUnavailable = true; break; }
                if (!_weights.TryGetValue(wn[6], out var gateInpW)) { _modelDecodeUnavailable = true; break; }
                _weights.TryGetValue(wn[2], out var qkvBias);
                _weights.TryGetValue(wn[4], out var oBias);
                _weights.TryGetValue(wn[7], out var gateInpBias);

                QuantizedWeight kQw = null, vQw = null;
                Tensor kBias = null, vBias = null;
                if (!_isQkvFused)
                {
                    if (!_quantWeights.TryGetValue(wn[8], out kQw)) { _modelDecodeUnavailable = true; break; }
                    if (!_quantWeights.TryGetValue(wn[10], out vQw)) { _modelDecodeUnavailable = true; break; }
                    _weights.TryGetValue(wn[9], out kBias);
                    _weights.TryGetValue(wn[11], out vBias);
                }

                var gateW = _layerStackedGate[l];
                var upW = _layerStackedUp[l];
                var downW = _layerStackedDown[l];
                if (gateW == null || upW == null || downW == null) { _modelDecodeUnavailable = true; break; }

                args[l] = new GptOssLayerDecodeArgs
                {
                    AttnNormW = (IntPtr)GetFloatPtr(attnNormW),
                    QkvW = qkvQw.CacheKey,
                    QkvB = qkvBias != null ? (IntPtr)GetFloatPtr(qkvBias) : IntPtr.Zero,
                    KW = kQw != null ? kQw.CacheKey : IntPtr.Zero,
                    KB = kBias != null ? (IntPtr)GetFloatPtr(kBias) : IntPtr.Zero,
                    VW = vQw != null ? vQw.CacheKey : IntPtr.Zero,
                    VB = vBias != null ? (IntPtr)GetFloatPtr(vBias) : IntPtr.Zero,
                    OW = oQw.CacheKey,
                    OB = oBias != null ? (IntPtr)GetFloatPtr(oBias) : IntPtr.Zero,
                    KCache = IntPtr.Zero,   // refreshed per call (the cache is regrown)
                    VCache = IntPtr.Zero,
                    // Reuses the per-layer sinks pin the fused attention kernel
                    // already holds, so both paths hand the native cacheable-buffer
                    // registry the same address.
                    Sinks = _layerSinks?[l] != null ? (IntPtr)GetFloatArrayPtr(_layerSinks[l], l) : IntPtr.Zero,
                    PostAttnNormW = (IntPtr)GetFloatPtr(postAttnNormW),
                    GateInpW = (IntPtr)GetFloatPtr(gateInpW),
                    GateInpB = gateInpBias != null ? (IntPtr)GetFloatPtr(gateInpBias) : IntPtr.Zero,
                    GateExps = gateW.Data,
                    GateExpsB = PinArray(_layerGateBiasStacked?[l]),
                    UpExps = upW.Data,
                    UpExpsB = PinArray(_layerUpBiasStacked?[l]),
                    DownExps = downW.Data,
                    DownExpsB = PinArray(_layerDownBiasStacked?[l]),

                    QkvNe0 = qkvQw.Ne0, QkvNe1 = qkvQw.Ne1, QkvBytes = qkvQw.RawBytes,
                    KNe0 = kQw?.Ne0 ?? 0, KNe1 = kQw?.Ne1 ?? 0, KBytes = kQw?.RawBytes ?? 0,
                    VNe0 = vQw?.Ne0 ?? 0, VNe1 = vQw?.Ne1 ?? 0, VBytes = vQw?.RawBytes ?? 0,
                    ONe0 = oQw.Ne0, ONe1 = oQw.Ne1, OBytes = oQw.RawBytes,
                    GeNe0 = gateW.PerExpertNe0, GeNe1 = gateW.PerExpertNe1, GeBytes = gateW.TotalRawBytes,
                    UeNe0 = upW.PerExpertNe0, UeNe1 = upW.PerExpertNe1, UeBytes = upW.TotalRawBytes,
                    DeNe0 = downW.PerExpertNe0, DeNe1 = downW.PerExpertNe1, DeBytes = downW.TotalRawBytes,

                    StructBytes = structBytes,
                    HiddenSize = Config.HiddenSize,
                    NumHeads = Config.NumHeads,
                    NumKvHeads = Config.NumKVHeads,
                    HeadDim = Config.HeadDim,
                    CacheSize = 0,          // refreshed per call
                    IsSwa = (l % 2 == 0) ? 1 : 0,
                    SlidingWindow = _slidingWindow,
                    RopeNDims = Config.HeadDim,
                    OrigCtxLen = Config.OriginalContextLength,
                    KvCacheType = _kvCacheDtype.GgmlType(),
                    NumExperts = _numExperts,
                    NumExpertsUsed = _numExpertsUsed,
                    SeparateQkv = _isQkvFused ? 0 : 1,
                    QkvType = qkvQw.GgmlType,
                    KType = kQw?.GgmlType ?? 0,
                    VType = vQw?.GgmlType ?? 0,
                    OType = oQw.GgmlType,
                    GeType = gateW.GgmlType,
                    UeType = upW.GgmlType,
                    DeType = downW.GgmlType,
                    // MoE CPU offload: this layer's routed experts stay in system
                    // RAM and its expert matmuls run on the host between
                    // accelerator graph segments.
                    CpuMoe = MoeCpuOffloadConfig.IsLayerOnCpu(l) ? 1 : 0,

                    Eps = Config.Eps,
                    RopeBase = Config.RopeBase,
                    RopeFreqScale = 1.0f / Config.RopeScale,
                    OaiAlpha = SiluAlpha,
                    OaiLimit = SiluLimit,
                };

                if (PrefillF16WeightsEnabled)
                {
                    ref var a = ref args[l];
                    if (qkvQw.HasHostData)
                        a.QkvWF16 = BuildF16WeightCopy(a.QkvW, a.QkvType, a.QkvNe0, a.QkvNe1);
                    if (kQw != null && kQw.HasHostData)
                        a.KWF16 = BuildF16WeightCopy(a.KW, a.KType, a.KNe0, a.KNe1);
                    if (vQw != null && vQw.HasHostData)
                        a.VWF16 = BuildF16WeightCopy(a.VW, a.VType, a.VNe0, a.VNe1);
                    if (oQw.HasHostData)
                        a.OWF16 = BuildF16WeightCopy(a.OW, a.OType, a.ONe0, a.ONe1);
                    a.GateExpsF16 = BuildF16WeightCopy(a.GateExps, a.GeType, a.GeNe0, a.GeNe1 * _numExperts);
                    a.UpExpsF16 = BuildF16WeightCopy(a.UpExps, a.UeType, a.UeNe0, a.UeNe1 * _numExperts);
                    a.DownExpsF16 = BuildF16WeightCopy(a.DownExps, a.DeType, a.DeNe0, a.DeNe1 * _numExperts);
                }
            }

            if (_modelDecodeUnavailable)
            {
                FreePinnedDecodeArrays();
                return false;
            }

            _modelDecodeArgs = args;
            return true;
        }

        /// <summary>
        /// Runs the whole model for one token in a single graph dispatch. On success
        /// the pinned fold-logits buffer holds the LM-head output and the caller
        /// skips its own final-norm / LM-head tail.
        /// </summary>
        private unsafe bool TryFusedModelDecode(Tensor hidden, int startPos)
        {
            if (!TryBuildModelDecodeArgs())
                return false;

            // The LM head is folded in so the vocab projection stays inside the
            // captured replay; without a quantized output weight we cannot bind it.
            if (!_quantWeights.TryGetValue("output.weight", out var lmHead) &&
                !_quantWeights.TryGetValue("token_embd.weight", out lmHead))
            {
                _modelDecodeUnavailable = true;
                return false;
            }
            if (!_weights.TryGetValue("output_norm.weight", out var finalNorm))
            {
                _modelDecodeUnavailable = true;
                return false;
            }

            EnsureFoldLogitsBuffer();

            int numLayers = Config.NumLayers;
            int cacheSize = (int)_kvCacheK[0].Sizes[1];
            for (int l = 0; l < numLayers; l++)
            {
                _modelDecodeArgs[l].KCache = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                _modelDecodeArgs[l].VCache = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
                _modelDecodeArgs[l].CacheSize = cacheSize;
            }

            bool ok = GgmlBasicOps.TryGptOssModelDecode(
                _modelDecodeArgs, numLayers,
                (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, startPos,
                _foldLogitsPtr, Config.VocabSize,
                lmHead.CacheKey, lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                (IntPtr)GetFloatPtr(finalNorm));

            if (ok)
                _kvCacheHostDirty = true;
            return ok;
        }

        /// <summary>
        /// True when the whole-model prefill kernel can serve this chunk. Same
        /// gate as the decode one plus a seqLen &gt; 1, and it answers before the
        /// KV host sync is skipped, so it must not depend on anything the kernel
        /// only discovers at runtime.
        /// </summary>
        private bool WillUseFusedModelPrefill(int seqLen)
        {
            if (!FusedModelPrefillEnabled || _modelPrefillUnavailable || _modelDecodeUnavailable || IsTensorParallel)
                return false;
            if (seqLen <= 1 || !IsGgmlBackend || _layerStackedReady == 0)
                return false;
            int kvType = _kvCacheDtype.GgmlType();
            return kvType == 0 /* F32 */ || kvType == 1 /* F16 */;
        }

        /// <summary>
        /// Runs the whole model over a prompt chunk in a single graph dispatch.
        /// On success the pinned fold-logits buffer holds the last token's
        /// LM-head output and the caller skips its own final-norm / LM-head tail.
        ///
        /// This is what replaced GPT-OSS's per-layer prefill: that path spent two
        /// host round trips per layer (the router scores came back to do top-k on
        /// the CPU, the MoE result went back up) on top of ~6 graph submissions,
        /// which is why a 512-token prefill measured 987 tok/s against
        /// llama.cpp's 17k on the same card.
        /// </summary>
        private unsafe bool TryFusedModelPrefill(Tensor hidden, int startPos, int seqLen)
        {
            if (!TryBuildModelDecodeArgs())
                return false;

            if (!_quantWeights.TryGetValue("output.weight", out var lmHead) &&
                !_quantWeights.TryGetValue("token_embd.weight", out lmHead))
            {
                _modelPrefillUnavailable = true;
                return false;
            }
            if (!_weights.TryGetValue("output_norm.weight", out var finalNorm))
            {
                _modelPrefillUnavailable = true;
                return false;
            }

            EnsureFoldLogitsBuffer();

            int numLayers = Config.NumLayers;
            int cacheSize = (int)_kvCacheK[0].Sizes[1];
            for (int l = 0; l < numLayers; l++)
            {
                _modelDecodeArgs[l].KCache = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                _modelDecodeArgs[l].VCache = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
                _modelDecodeArgs[l].CacheSize = cacheSize;
            }

            bool ok = GgmlBasicOps.TryGptOssModelPrefill(
                _modelDecodeArgs, numLayers,
                (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, seqLen, startPos,
                _foldLogitsPtr, Config.VocabSize,
                lmHead.CacheKey, lmHead.GgmlType, lmHead.Ne0, lmHead.Ne1, lmHead.RawBytes,
                (IntPtr)GetFloatPtr(finalNorm));

            if (ok)
            {
                _kvCacheHostDirty = true;
            }
            else
            {
                // A refusal is structural (a shape or a weight the kernel cannot
                // bind), not transient: stop paying the setup cost every chunk.
                // Say so once — the per-layer fallback is an order of magnitude
                // slower, so a silent demotion looks exactly like a perf bug.
                _modelPrefillUnavailable = true;
                Console.WriteLine("  GPT-OSS whole-model prefill unavailable, falling back to the per-layer path: "
                    + GgmlBasicOps.LastNativeError());
            }
            return ok;
        }

        /// <summary>
        /// Bring the host KV mirror up to date after fused decodes wrote rows
        /// device-side only. No-op unless a fused decode ran since the last sync.
        /// </summary>
        private void EnsureKvCacheHostSynchronized()
        {
            if (!_kvCacheHostDirty || !IsGgmlBackend)
                return;

            if (_kvCacheK == null)
            {
                // Tensor parallelism keeps its KV in the per-rank arrays, and the
                // fused TP decode advances them device-side only. Without this the
                // sync silently no-opped and any host reader (snapshot, truncate,
                // the per-op chain) saw a stale prefix.
                if (_tpKvCacheK == null || _tpKvCacheK[0] == null || _tpKvCacheK[0][0] == null)
                    return;
                int tpCacheSize = (int)_tpKvCacheK[0][0].Sizes[1];
                int previousRank = GgmlBasicOps.GetActiveRank();
                try
                {
                    for (int r = 0; r < TpDegree; r++)
                    {
                        GgmlBasicOps.SetActiveRank(r);
                        for (int l = 0; l < Config.NumLayers; l++)
                        {
                            if (_tpKvCacheK[l] == null || _tpKvCacheV[l] == null) continue;
                            if (_tpKvCacheK[l][r] == null || _tpKvCacheV[l][r] == null) continue;
                            GgmlBasicOps.GptOssSyncKvCacheToHost(
                                TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[l][r]),
                                TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[l][r]),
                                tpCacheSize, _cacheSeqLen);
                        }
                    }
                }
                finally
                {
                    GgmlBasicOps.SetActiveRank(previousRank);
                }
                _kvCacheHostDirty = false;
                return;
            }

            int cacheSize = (int)_kvCacheK[0].Sizes[1];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvCacheK[l] == null || _kvCacheV[l] == null) continue;
                GgmlBasicOps.GptOssSyncKvCacheToHost(
                    TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]),
                    TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]),
                    cacheSize, _cacheSeqLen);
            }
            _kvCacheHostDirty = false;
        }

        /// <summary>
        /// Drop the persistent decode graph. The captured graph pins the ggml-cuda
        /// compute pool and the KV windows, so anything that can move them (a
        /// prefill, a KV reset, a cache grow) invalidates it.
        /// </summary>
        private void ResetFusedModelDecodeCache()
        {
            if (IsGgmlBackend)
            {
                GgmlBasicOps.GptOssResetDecodeCache();
                CountDecodeGraphReset();
            }
            // The per-rank TP graphs come from the same native pools.
            _tpFdBuiltCapacity = -1;
        }

        private void DisposeFusedModelDecodeState()
        {
            ResetFusedModelDecodeCache();
            FreeF16PrefillBuffers();
            // The token-batched arena pool deliberately survives prefills (and
            // the reset above); on model teardown it must go explicitly or its
            // per-entry arena buffers keep their VRAM until process exit.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.GptOssResetBatchedDecodeCache();
                CountDecodeGraphReset();
            }
            FreePinnedDecodeArrays();
            if (_foldLogitsHandle.IsAllocated)
                _foldLogitsHandle.Free();
            _foldLogitsPtr = IntPtr.Zero;
            _modelDecodeArgs = null;
        }
    }
}
