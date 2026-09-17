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
// ---------------------------------------------------------------------------
// Whole-model fused forward.
//
// The per-op path submits roughly 600 GGML dispatches per token across 52
// layers; measured on an RTX 3080 that is ~262 ms of fixed cost per forward
// regardless of how many rows it carries, which is what kept decode at 4.3 tok/s
// against llama.cpp's 22. TSGgml_MuseGlimmerModelForward
// (TensorSharp.GGML.Native/ggml_ops_muse_glimmer.cpp) builds the entire model -
// every layer, the final norm, the LM head, the logit scale and the tanh softcap
// - as ONE ggml graph:
//
//   * decode (1 row) reuses a persistent graph with stable tensor addresses, so
//     ggml-cuda captures it and a token becomes one graph replay;
//   * prefill (N rows) builds a transient graph, still one submission.
//
// The graph is numerically the same chain as TransformerBlock: identical op
// order, identical epsilons (pre-norms at Config.Eps, post-norms at 1e-8), the
// same NORM-flavour RoPE on sliding-window layers only, the same attention
// output gate. TS_MUSE_GLIMMER_FUSED=0 forces the per-op path for A/B.
// ---------------------------------------------------------------------------
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class MuseGlimmerModel : ModelBase
    {
        /// <summary>Per-layer pointer/shape tables for the fused kernel, built once.</summary>
        private sealed class FusedArrays
        {
            public IntPtr[] AttnNorm, Q, K, V, Gate, QNorm, KNorm, O, PostAttnNorm, FfnNorm, Gu, Down, PostFfnNorm;
            public IntPtr[] KCache, VCache;
            public int[] IsSwa;
            public int[] QType, KType, VType, GateType, OType, GuType, DownType;
            public long[] QNe0, QNe1, QBytes;
            public long[] KNe0, KNe1, KBytes;
            public long[] VNe0, VNe1, VBytes;
            public long[] GateNe0, GateNe1, GateBytes;
            public long[] ONe0, ONe1, OBytes;
            public long[] GuNe0, GuNe1, GuBytes;
            public long[] DownNe0, DownNe1, DownBytes;

            public IntPtr LmHead;
            public int LmHeadType;
            public long LmHeadNe0, LmHeadNe1, LmHeadBytes;
            public IntPtr FinalNorm;

            /// <summary>Quantized token-embedding table for the in-graph input stage.</summary>
            public IntPtr TokEmbd;
            public int TokEmbdType;
            public long TokEmbdNe0, TokEmbdNe1, TokEmbdBytes;

            /// <summary>KV-cache capacity the pointers were captured at; a grow rebuilds.</summary>
            public int CacheCapacity;
        }

        private FusedArrays _fusedArrays;
        private bool _fusedProbed;

        /// <summary>
        /// True when the kernel can do the embedding gather + input norm itself, so a
        /// caller may hand it token ids instead of a hidden tensor. Only armed when the
        /// table is already device-resident (tied LM head) - see BuildFusedArrays.
        /// </summary>
        /// <remarks>
        /// Written the long way on purpose. <c>GetFusedArrays()?.TokEmbd != IntPtr.Zero</c>
        /// is an <c>IntPtr?</c> comparison, and C#'s lifted <c>!=</c> yields TRUE when one
        /// side is null - so on a backend with no fused kernel at all (MLX, GgmlCpu) the
        /// property claimed the in-graph embedding path WAS available.
        /// </remarks>
        internal bool FusedInGraphEmbedAvailable
        {
            get
            {
                var a = GetFusedArrays();
                return a != null && a.TokEmbd != IntPtr.Zero;
            }
        }

        private static readonly bool FusedForwardEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_FUSED"), "0", StringComparison.Ordinal);

        // ====================================================================
        // Layer trace (TS_MUSE_GLIMMER_LAYER_TRACE=1) - diagnostics only.
        // ====================================================================
        //
        // Prints a checksum of the residual ENTERING every layer (and the final
        // residual) for the FIRST forward of the per-op loop. The fused kernel emits
        // the identical summary under the same variable, so diffing the two runs is
        // what localizes a fused-vs-per-op divergence to a layer; without it the only
        // observable is the sampled text, which says nothing about where they parted.
        // TS_MUSE_GLIMMER_LAYER_TRACE_POS selects the first forward at or past that
        // absolute position (which is what skips the startup warmup, whose own prefill
        // and decode run at unrelated positions), and ..._N how many consecutive
        // forwards to trace from there.
        private static readonly bool LayerTraceEnabled =
            !string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_LAYER_TRACE")) &&
            Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_LAYER_TRACE") != "0";
        private static readonly int LayerTracePos =
            int.TryParse(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_LAYER_TRACE_POS"), out int _ltp) ? _ltp : 0;
        private static readonly int LayerTraceCount =
            int.TryParse(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_LAYER_TRACE_N"), out int _ltn) && _ltn > 0 ? _ltn : 1;
        private int _layerTraceForwards;
        private int _layerTraceRows;
        private int _layerTracePos;

        private bool TraceActive =>
            LayerTraceEnabled && _layerTraceForwards < LayerTraceCount && _layerTracePos >= LayerTracePos;

        internal void TraceLayerResidual(Tensor hidden, int layer)
        {
            if (!TraceActive || hidden == null)
                return;
            float[] v = TensorToFloatArray(hidden);
            double sum = 0.0, abs = 0.0;
            float max = float.NegativeInfinity;
            int nonFinite = 0;
            for (int i = 0; i < v.Length; i++)
            {
                sum += v[i];
                abs += Math.Abs((double)v[i]);
                if (v[i] > max) max = v[i];
                if (!float.IsFinite(v[i])) nonFinite++;
            }
            Console.Error.WriteLine($"[MGTRACE-perop] pos={_layerTracePos} rows={_layerTraceRows} layer={layer} " +
                $"n={v.Length} sum={sum:F4} abs={abs:F4} max={max:F6} nonfinite={nonFinite}");
            Console.Error.Flush();
            string dir = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_LAYER_TRACE_DIR");
            if (!string.IsNullOrEmpty(dir))
            {
                var bytes = new byte[v.Length * 4];
                Buffer.BlockCopy(v, 0, bytes, 0, bytes.Length);
                System.IO.File.WriteAllBytes(
                    System.IO.Path.Combine(dir, $"perop_S{_layerTraceForwards:D2}_L{layer:D2}.bin"), bytes);
            }
        }

        /// <summary>Top-5 of the trace forward's logits, for a fused-vs-per-op A/B.</summary>
        internal void TraceLogits(float[] logits)
        {
            if (!TraceActive || logits == null)
                return;
            int n = Math.Min(logits.Length, Config.VocabSize);
            var best = new int[5];
            for (int k = 0; k < 5; k++)
            {
                int bi = -1;
                for (int i = 0; i < n; i++)
                {
                    bool taken = false;
                    for (int j = 0; j < k; j++) if (best[j] == i) { taken = true; break; }
                    if (taken) continue;
                    if (bi < 0 || logits[i] > logits[bi]) bi = i;
                }
                best[k] = bi;
            }
            var sb = new System.Text.StringBuilder($"[MGTRACE-logits] pos={_layerTracePos} rows={_layerTraceRows} top5:");
            for (int k = 0; k < 5; k++)
                sb.Append($" ({best[k]}, {logits[best[k]]:F6})");
            Console.Error.WriteLine(sb.ToString());
            Console.Error.Flush();
        }

        internal void TraceLayerSetRows(int rows, int startPos)
        {
            _layerTraceRows = rows;
            _layerTracePos = startPos;
        }

        internal void TraceLayerDone()
        {
            if (TraceActive)
                _layerTraceForwards++;
        }

        /// <summary>
        /// The fused kernel is enabled on every GGML backend.
        ///
        /// Metal was originally left out because the kernel writes the KV cache
        /// with ggml_set_rows, which ggml-metal did not implement at the time; it
        /// does now (GGML_OP_SET_ROWS, F32 -> F16/F32/quant, see
        /// ggml-metal-device.m). Without this the 52-layer model fell back to the
        /// per-op path and decoded at 246 ms/token on an M5 Pro (attention + norms
        /// alone were 60% of the token) versus 26 ms/token fused - a 9.4x
        /// regression relative to every other architecture on the same machine.
        /// Metal also runs the kernel's persist/replay branch (no CUDA-style
        /// capture, but the per-token graph rebuild, re-binds, gallocr re-plan and
        /// mask regeneration are gone).
        ///
        /// GgmlCpu is included the same way GPT-OSS and Gemma 4 include it for
        /// their whole-model kernels: the per-op path is ~940 synchronous ggml
        /// graph submissions per decoded token, each of which used to spawn a
        /// disposable 4-thread pool. TS_MUSE_GLIMMER_FUSED_CPU=0 restores the
        /// per-op CPU path (the historical default) if a regression appears.
        /// </summary>
        private static readonly bool FusedCpuEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_FUSED_CPU"), "0", StringComparison.Ordinal);

        private bool CanUseFusedForward =>
            FusedForwardEnabled && !IsTensorParallel &&
            (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan ||
             _backend == BackendType.GgmlMetal ||
             (_backend == BackendType.GgmlCpu && FusedCpuEnabled));

        /// <summary>
        /// Build (or rebuild) the fused kernel's pointer tables. Every 2D projection
        /// must be quantized-resident: the kernel binds them by (CacheKey, ggml type,
        /// ne0, ne1, rawBytes), which is how they stay device-resident across calls.
        /// </summary>
        private unsafe void BuildFusedArrays()
        {
            _fusedProbed = true;
            _fusedArrays = null;
            if (!CanUseFusedForward)
                return;

            int n = Config.NumLayers;
            var a = new FusedArrays
            {
                AttnNorm = new IntPtr[n], Q = new IntPtr[n], K = new IntPtr[n], V = new IntPtr[n],
                Gate = new IntPtr[n], QNorm = new IntPtr[n], KNorm = new IntPtr[n], O = new IntPtr[n],
                PostAttnNorm = new IntPtr[n], FfnNorm = new IntPtr[n], Gu = new IntPtr[n],
                Down = new IntPtr[n], PostFfnNorm = new IntPtr[n],
                KCache = new IntPtr[n], VCache = new IntPtr[n],
                IsSwa = new int[n],
                QType = new int[n], KType = new int[n], VType = new int[n], GateType = new int[n],
                OType = new int[n], GuType = new int[n], DownType = new int[n],
                QNe0 = new long[n], QNe1 = new long[n], QBytes = new long[n],
                KNe0 = new long[n], KNe1 = new long[n], KBytes = new long[n],
                VNe0 = new long[n], VNe1 = new long[n], VBytes = new long[n],
                GateNe0 = new long[n], GateNe1 = new long[n], GateBytes = new long[n],
                ONe0 = new long[n], ONe1 = new long[n], OBytes = new long[n],
                GuNe0 = new long[n], GuNe1 = new long[n], GuBytes = new long[n],
                DownNe0 = new long[n], DownNe1 = new long[n], DownBytes = new long[n],
                CacheCapacity = _kvCacheCapacity,
            };

            for (int l = 0; l < n; l++)
            {
                string[] wn = _layerWeightNames[l];
                if (!TryQuant(wn[1], out var q) || !TryQuant(wn[2], out var k) || !TryQuant(wn[3], out var v)
                    || !TryQuant(wn[4], out var gate) || !TryQuant(wn[7], out var o)
                    || !TryQuant(wn[10], out var gu) || !TryQuant(wn[11], out var down))
                {
                    Console.WriteLine($"  Muse-Glimmer fused forward disabled: layer {l} has a non-quantized projection.");
                    return;
                }
                if (!_weights.TryGetValue(wn[0], out var attnNorm) || !_weights.TryGetValue(wn[5], out var qNorm)
                    || !_weights.TryGetValue(wn[6], out var kNorm) || !_weights.TryGetValue(wn[8], out var postAttn)
                    || !_weights.TryGetValue(wn[9], out var ffnNorm) || !_weights.TryGetValue(wn[12], out var postFfn))
                {
                    Console.WriteLine($"  Muse-Glimmer fused forward disabled: layer {l} is missing a norm weight.");
                    return;
                }

                a.AttnNorm[l] = (IntPtr)GetFloatPtr(attnNorm);
                a.QNorm[l] = (IntPtr)GetFloatPtr(qNorm);
                a.KNorm[l] = (IntPtr)GetFloatPtr(kNorm);
                a.PostAttnNorm[l] = (IntPtr)GetFloatPtr(postAttn);
                a.FfnNorm[l] = (IntPtr)GetFloatPtr(ffnNorm);
                a.PostFfnNorm[l] = (IntPtr)GetFloatPtr(postFfn);

                a.Q[l] = q.CacheKey; a.QType[l] = q.GgmlType; a.QNe0[l] = q.Ne0; a.QNe1[l] = q.Ne1; a.QBytes[l] = q.RawBytes;
                a.K[l] = k.CacheKey; a.KType[l] = k.GgmlType; a.KNe0[l] = k.Ne0; a.KNe1[l] = k.Ne1; a.KBytes[l] = k.RawBytes;
                a.V[l] = v.CacheKey; a.VType[l] = v.GgmlType; a.VNe0[l] = v.Ne0; a.VNe1[l] = v.Ne1; a.VBytes[l] = v.RawBytes;
                a.Gate[l] = gate.CacheKey; a.GateType[l] = gate.GgmlType; a.GateNe0[l] = gate.Ne0; a.GateNe1[l] = gate.Ne1; a.GateBytes[l] = gate.RawBytes;
                a.O[l] = o.CacheKey; a.OType[l] = o.GgmlType; a.ONe0[l] = o.Ne0; a.ONe1[l] = o.Ne1; a.OBytes[l] = o.RawBytes;
                a.Gu[l] = gu.CacheKey; a.GuType[l] = gu.GgmlType; a.GuNe0[l] = gu.Ne0; a.GuNe1[l] = gu.Ne1; a.GuBytes[l] = gu.RawBytes;
                a.Down[l] = down.CacheKey; a.DownType[l] = down.GgmlType; a.DownNe0[l] = down.Ne0; a.DownNe1[l] = down.Ne1; a.DownBytes[l] = down.RawBytes;

                a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[l]);
                a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[l]);
                a.IsSwa[l] = _isSwaLayer[l] ? 1 : 0;
            }

            // Folded output head. Without it the kernel returns the hidden state and
            // the caller runs the final norm + 202K-vocab projection as separate
            // dispatches, which for decode is a large share of the token.
            string headName = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            if (_quantWeights.TryGetValue(headName, out var head) && _weights.TryGetValue("output_norm.weight", out var fnorm))
            {
                a.LmHead = head.CacheKey;
                a.LmHeadType = head.GgmlType;
                a.LmHeadNe0 = head.Ne0;
                a.LmHeadNe1 = head.Ne1;
                a.LmHeadBytes = head.RawBytes;
                a.FinalNorm = (IntPtr)GetFloatPtr(fnorm);
            }

            // In-graph embedding only pays when the table costs nothing extra to bind:
            // when the LM head is tied to it (already device-resident), or on Metal,
            // where quantized weights are zero-copy host-pointer wraps of the GGUF
            // mmap - binding token_embd pins no second copy on unified memory. On a
            // discrete card with a separate output.weight it would pin ~1.1 GB and
            // evict layer weights (measured 18.5 -> 16.1 tok/s on a 16 GB card).
            // TS_MUSE_GLIMMER_INGRAPH_EMBED=1 forces it on, =0 forces it off (A/B).
            string ingraphEnv = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_INGRAPH_EMBED");
            bool wantInGraphEmbed = string.Equals(ingraphEnv, "0", StringComparison.Ordinal)
                ? false
                : _hasTiedOutput || _backend == BackendType.GgmlMetal || _backend == BackendType.GgmlCpu ||
                  string.Equals(ingraphEnv, "1", StringComparison.Ordinal);
            if (wantInGraphEmbed && _quantWeights.TryGetValue("token_embd.weight", out var tokEmbd))
            {
                a.TokEmbd = tokEmbd.CacheKey;
                a.TokEmbdType = tokEmbd.GgmlType;
                a.TokEmbdNe0 = tokEmbd.Ne0;
                a.TokEmbdNe1 = tokEmbd.Ne1;
                a.TokEmbdBytes = tokEmbd.RawBytes;
            }

            _fusedArrays = a;
            Console.WriteLine($"  Muse-Glimmer fused forward armed ({n} layers, folded LM head: {a.LmHead != IntPtr.Zero}, " +
                $"in-graph embedding: {a.TokEmbd != IntPtr.Zero}).");
        }

        private bool TryQuant(string name, out QuantizedWeight qw)
            => _quantWeights.TryGetValue(name, out qw) && qw != null && qw.CacheKey != IntPtr.Zero;

        private FusedArrays GetFusedArrays()
        {
            if (!_fusedProbed)
                BuildFusedArrays();
            // A cache grow reallocates every KV tensor, so the captured pointers and
            // the kernel's baked cache_size are both stale.
            if (_fusedArrays != null && _fusedArrays.CacheCapacity != _kvCacheCapacity)
            {
                GgmlBasicOps.MuseGlimmerResetDecodeCache();
                CountDecodeGraphReset();
                BuildFusedArrays();
            }
            return _fusedArrays;
        }

        /// <summary>
        /// Run the whole model in one GGML graph. <paramref name="hidden"/> is
        /// [nTokens, hidden] and already embedded + input-normed. Returns the logits
        /// of the LAST row when the LM head is folded in, otherwise null (and
        /// <paramref name="hidden"/> holds the updated hidden states).
        /// </summary>
        private unsafe float[] TryFusedForward(Tensor hidden, int seqLen, int startPos,
            float[] captureBuffer, int[] captureLayers, int[] tokenIds = null,
            bool allLogitsRows = false, float[] logitsOut = null)
        {
            var a = GetFusedArrays();
            if (a == null)
                return null;

            bool fold = a.LmHead != IntPtr.Zero;
            if (!fold)
                return null;

            // Fold the embedding gather + weightless input norm into the graph when
            // the caller has plain token ids. That drops two host-visible dispatches
            // per step and leaves the token id as the only per-token input, which is
            // what makes the captured decode graph cheap to refresh. A multimodal
            // step passes tokenIds = null: its rows carry spliced image embeddings.
            IntPtr tokEmbd = IntPtr.Zero;
            int tokEmbdType = 0;
            long tokEmbdNe0 = 0, tokEmbdNe1 = 0, tokEmbdBytes = 0;
            if (tokenIds != null && a.TokEmbd != IntPtr.Zero)
            {
                tokEmbd = a.TokEmbd;
                tokEmbdType = a.TokEmbdType;
                tokEmbdNe0 = a.TokEmbdNe0;
                tokEmbdNe1 = a.TokEmbdNe1;
                tokEmbdBytes = a.TokEmbdBytes;
            }
            else
            {
                tokenIds = null;
            }

            // The verify batch wants logits for every row; ordinary decode only the
            // last. Route straight into the caller's buffer when it supplied one.
            long needed = allLogitsRows ? (long)seqLen * Config.VocabSize : Config.VocabSize;
            float[] target = logitsOut;
            if (target == null || target.LongLength < needed)
            {
                if (_fusedLogits == null || _fusedLogits.LongLength < needed)
                    _fusedLogits = new float[needed];
                target = _fusedLogits;
            }

            int captureCount = captureLayers != null ? captureLayers.Length : 0;

            long t0 = Stopwatch.GetTimestamp();
            bool ok;
            IntPtr hiddenPtr = hidden != null ? (IntPtr)GetFloatPtr(hidden) : IntPtr.Zero;
            fixed (float* logitsPtr = target)
            fixed (float* capturePtr = captureBuffer)
            {
                ok = GgmlBasicOps.MuseGlimmerModelForward(
                    hiddenPtr, Config.HiddenSize, seqLen, Config.NumLayers,
                    a.AttnNorm, a.Q, a.K, a.V, a.Gate, a.QNorm, a.KNorm, a.O,
                    a.PostAttnNorm, a.FfnNorm, a.Gu, a.Down, a.PostFfnNorm,
                    a.KCache, a.VCache, a.IsSwa,
                    a.QType, a.QNe0, a.QNe1, a.QBytes,
                    a.KType, a.KNe0, a.KNe1, a.KBytes,
                    a.VType, a.VNe0, a.VNe1, a.VBytes,
                    a.GateType, a.GateNe0, a.GateNe1, a.GateBytes,
                    a.OType, a.ONe0, a.ONe1, a.OBytes,
                    a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                    a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                    Config.NumHeads, Config.NumKVHeads, _headDim, _kvCacheCapacity, _kvSwaRows,
                    startPos, _slidingWindow,
                    Config.Eps, PostNormEps, Config.RopeBase, 1.0f / Config.RopeScale,
                    1f / MathF.Sqrt(_headDim), (int)_kvCacheDtype.GgmlType(),
                    (IntPtr)logitsPtr, Config.VocabSize,
                    a.LmHead, a.LmHeadType, a.LmHeadNe0, a.LmHeadNe1, a.LmHeadBytes,
                    a.FinalNorm, _finalLogitScale, _finalLogitSoftcap,
                    (IntPtr)capturePtr, captureLayers, captureCount,
                    tokEmbd, tokEmbdType, tokEmbdNe0, tokEmbdNe1, tokEmbdBytes, tokenIds,
                    allLogitsRows ? 1 : 0);
            }
            _linearTicks += Stopwatch.GetTimestamp() - t0;

            if (!ok)
            {
                // A declined call leaves the KV cache untouched for this step, so the
                // caller can safely re-run the per-op path for it. The decline is NOT
                // sticky: a shape-specific refusal (an unusually large prefill chunk,
                // say) must not cost every later decode token the fast path. Only a
                // run of consecutive failures disarms.
                if (++_fusedFailures <= 3)
                {
                    Console.WriteLine($"  Muse-Glimmer fused forward declined (seqLen={seqLen} startPos={startPos} cap={_kvCacheCapacity}): " +
                        $"{GgmlBasicOps.LastNativeError()}; falling back to the per-op path.");
                }
                if (_fusedFailures >= 16)
                {
                    Console.WriteLine("  Muse-Glimmer fused forward disabled after repeated failures.");
                    _fusedArrays = null;
                }
                return null;
            }
            _fusedFailures = 0;

            // The kernel wrote K/V on-device. The per-op attention kernels read the
            // caches through host pointers, so a later fallback (or a KV snapshot)
            // must see those rows.
            _fusedKvDirty = true;
            return target;
        }

        private float[] _fusedLogits;
        private bool _fusedKvDirty;
        private int _fusedFailures;

        /// <summary>
        /// Pull device-side KV writes made by the fused kernel back to the host copy.
        /// Called before anything that walks the caches with host pointers.
        /// </summary>
        private void EnsureFusedKvHostSynchronized()
        {
            if (!_fusedKvDirty || _kvCacheK == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                SyncTensorHostCache(_kvCacheK[l]);
                SyncTensorHostCache(_kvCacheV[l]);
            }
            _fusedKvDirty = false;
        }

        /// <summary>
        /// Drop the persistent decode graph. It bakes ggml-cuda's scratch-pool and
        /// KV-cache device addresses, both of which a prefill or a cache reset/grow
        /// can move; replaying a stale capture hangs the GPU.
        /// </summary>
        private void ResetFusedDecodeCache()
        {
            if (IsGgmlBackend)
            {
                GgmlBasicOps.MuseGlimmerResetDecodeCache();
                CountDecodeGraphReset();
            }
        }
    }
}
