// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Direct-backend MiniMax-H3 text encoder: the language-model trunk of
// TSGgml_MiniMaxH3TextEncode expressed against the shared direct primitives, so
// BackendType.Cpu (the 100% pure-C# backend) can run it with no ggml.
//
// Two details are specific to this trunk and easy to get wrong:
//   - Q and K are RMS-normalized per head before RoPE, not after.
//   - the checkpoint has no final norm; the DiT consumes the raw last-layer state.
using System;
using System.Globalization;
using System.IO;

using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Managed MiniMax-H3 text-encoder trunk (no ggml).</summary>
    internal sealed class MiniMaxH3DirectTextEncoder : IDisposable
    {
        private sealed class LayerW
        {
            public Tensor InputNorm, PostAttnNorm, QNorm, KNorm;
            public DirectLinear Q, K, V, O, Gate, Up, Down;
        }

        private readonly DirectContext _ctx;
        private readonly MiniMaxH3TensorOps _tensorOps;
        private readonly MiniMaxH3TextEncoderConfig _cfg;
        private readonly GgufFile _gguf; // borrowed; the text-encoder owner outlives this trunk
        private readonly LayerW[] _layers;
        private readonly System.Collections.Generic.List<DirectLinear> _owned = new();
        private bool _disposed;
        private bool _prefaultAttempted;

        public MiniMaxH3DirectTextEncoder(GgufFile gguf, MiniMaxH3TextEncoderConfig config,
                                          IAllocator allocator)
        {
            _cfg = config;
            _gguf = gguf;
            _ctx = new DirectContext(allocator);
            _tensorOps = new MiniMaxH3TensorOps(_ctx);

            DirectLinear Lin(string w, string b)
            {
                var l = DirectLinear.FromGguf(_ctx, gguf, w, gguf.Tensors.ContainsKey(b ?? "") ? b : null,
                    prescale: true, h3PreserveFloat32: true);
                _owned.Add(l);
                return l;
            }
            Tensor Vec(string name) =>
                _ctx.Own(_ctx.FromFloats(DirectOps.DequantTensor(gguf, name),
                                         (long)gguf.Tensors[name].NumElements));

            _layers = new LayerW[config.NumLayers];
            for (int i = 0; i < _layers.Length; i++)
            {
                string p = "model.layers." + i + ".";
                _layers[i] = new LayerW
                {
                    InputNorm = Vec(p + "input_layernorm.weight"),
                    PostAttnNorm = Vec(p + "post_attention_layernorm.weight"),
                    QNorm = Vec(p + "self_attn.q_norm.weight"),
                    KNorm = Vec(p + "self_attn.k_norm.weight"),
                    Q = Lin(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.bias"),
                    K = Lin(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.bias"),
                    V = Lin(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.bias"),
                    O = Lin(p + "self_attn.o_proj.weight", null),
                    Gate = Lin(p + "mlp.gate_proj.weight", null),
                    Up = Lin(p + "mlp.up_proj.weight", null),
                    Down = Lin(p + "mlp.down_proj.weight", null),
                };
            }
        }

        /// <summary>Per-head RMS norm over the head dim, in place on a fresh tensor.</summary>
        private Tensor NormHeads(Tensor x, Tensor gain, int seq, int nHeads, int hd)
        {
            using (x)
            using (Tensor flat = x.View((long)seq * nHeads, hd))
            using (Tensor n = DirectOps.RmsNorm(_ctx, flat, gain, _cfg.Eps))
                return n.View(seq, (long)nHeads * hd);
        }

        /// <summary>
        /// Rotate-half RoPE over a whole head (rot == headDim here), with per-token
        /// tables [seq, hd]. Shared shape with the DiT, but the DiT rotates only a
        /// prefix of each head, so the two are kept separate rather than merged
        /// behind a flag.
        /// </summary>
        private unsafe void RopeHalf(Tensor x, float[] cos, float[] sin,
                                            int seq, int nHeads, int hd)
        {
            if (_tensorOps.IsAccelerated)
            {
                _tensorOps.RopeHalf(x, cos, sin, seq, nHeads, hd, hd);
                return;
            }
            int half = hd / 2;
            float* px = (float*)CpuNativeHelpers.GetBufferStart(x);
            fixed (float* pcos = cos, psin = sin)
            {
                float* pc = pcos, ps = psin;
                CpuWorkerPool.Shared.For(seq, t =>
                {
                    float* ct = pc + (long)t * hd;
                    float* st = ps + (long)t * hd;
                    float* row = px + (long)t * nHeads * hd;
                    for (int h = 0; h < nHeads; h++)
                    {
                        float* v = row + (long)h * hd;
                        for (int i = 0; i < half; i++)
                        {
                            float a = v[i], b = v[i + half];
                            v[i] = a * ct[i] - b * st[i];
                            v[i + half] = b * ct[i + half] + a * st[i + half];
                        }
                    }
                });
            }
        }

        /// <summary>
        /// Causal GQA attention, per head, over token-major projections.
        /// A [heads, seq, seq] additive mask through the shared attention entry
        /// would be hundreds of MB at prompt lengths this trunk sees, so the mask
        /// is applied as a loop bound instead of as a tensor.
        /// </summary>
        private unsafe Tensor CausalGqaAttention(Tensor q, Tensor k, Tensor v,
                                                 int seq, int heads, int kvHeads, int hd)
        {
            if (_tensorOps.IsAccelerated)
                return _tensorOps.CausalGqaAttention(q, k, v, seq, heads, kvHeads, hd);
            float scale = 1.0f / MathF.Sqrt(hd);
            int group = heads / kvHeads;
            Tensor outT = _ctx.NewF32(seq, (long)heads * hd);

            float* pq = (float*)CpuNativeHelpers.GetBufferStart(q);
            float* pk = (float*)CpuNativeHelpers.GetBufferStart(k);
            float* pv = (float*)CpuNativeHelpers.GetBufferStart(v);
            float* po = (float*)CpuNativeHelpers.GetBufferStart(outT);

            long qRow = (long)heads * hd, kvRow = (long)kvHeads * hd;

            // One head per work item: the scores buffer is then [seq] per query row
            // rather than a full [seq, seq] matrix, which keeps the working set in
            // cache and needs no scratch allocation per worker.
            CpuWorkerPool.Shared.For(heads, h =>
            {
                int kv = h / group;
                var buf = new float[seq];
                fixed (float* sc = buf)
                {
                    for (int i = 0; i < seq; i++)
                    {
                        float* qi = pq + (long)i * qRow + (long)h * hd;
                        float max = float.NegativeInfinity;
                        for (int j = 0; j <= i; j++)
                        {
                            float* kj = pk + (long)j * kvRow + (long)kv * hd;
                            float dot = 0f;
                            for (int c = 0; c < hd; c++) dot += qi[c] * kj[c];
                            dot *= scale;
                            sc[j] = dot;
                            if (dot > max) max = dot;
                        }
                        float sum = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float e = MathF.Exp(sc[j] - max);
                            sc[j] = e;
                            sum += e;
                        }
                        float inv = 1f / sum;
                        float* oi = po + (long)i * qRow + (long)h * hd;
                        for (int c = 0; c < hd; c++) oi[c] = 0f;
                        for (int j = 0; j <= i; j++)
                        {
                            float w = sc[j] * inv;
                            float* vj = pv + (long)j * kvRow + (long)kv * hd;
                            for (int c = 0; c < hd; c++) oi[c] += w * vj[c];
                        }
                    }
                }
            });
            return outT;
        }

        /// <summary>
        /// Run the trunk over pre-looked-up embeddings. deepstack, when present, is
        /// [numDeepstack, seq, hidden] and its taps are added after the first
        /// numDeepstack layers.
        /// </summary>
        public float[] Encode(float[] embeddings, int seq, float[] cos, float[] sin,
                              float[] deepstack, int numDeepstack, int layerLimit)
        {
            int hidden = _cfg.Hidden, heads = _cfg.Heads, kvh = _cfg.KvHeads, hd = _cfg.HeadDim;
            float eps = _cfg.Eps;
            int nl = layerLimit > 0 ? Math.Min(layerLimit, _layers.Length) : _layers.Length;

            if (_ctx.IsCuda && !_prefaultAttempted)
            {
                _prefaultAttempted = true;
                PrefaultTextWeights(_gguf,nl);
            }

            Tensor h = _ctx.FromFloats(embeddings, seq, hidden);
            try
            {
                for (int l = 0; l < nl; l++)
                {
                    LayerW lw = _layers[l];

                    using (Tensor n1 = DirectOps.RmsNorm(_ctx, h, lw.InputNorm, eps))
                    {
                        Tensor q = lw.Q.Forward(n1);
                        Tensor k = lw.K.Forward(n1);
                        using Tensor v = lw.V.Forward(n1);

                        // This encoder normalizes Q and K per head before RoPE.
                        if (lw.QNorm != null) q = NormHeads(q, lw.QNorm, seq, heads, hd);
                        if (lw.KNorm != null) k = NormHeads(k, lw.KNorm, seq, kvh, hd);
                        RopeHalf(q, cos, sin, seq, heads, hd);
                        RopeHalf(k, cos, sin, seq, kvh, hd);

                        using (Tensor merged = CausalGqaAttention(q, k, v, seq, heads, kvh, hd))
                        using (Tensor proj = lw.O.Forward(merged))
                            Ops.Add(h, h, proj);
                        q.Dispose();
                        k.Dispose();
                    }

                    using (Tensor n2 = DirectOps.RmsNorm(_ctx, h, lw.PostAttnNorm, eps))
                    using (Tensor g = lw.Gate.Forward(n2))
                    using (Tensor u = lw.Up.Forward(n2))
                    using (Tensor act = DirectOps.Silu(_ctx, g))
                    {
                        Ops.Mul(act, act, u);
                        using Tensor down = lw.Down.Forward(act);
                        Ops.Add(h, h, down);
                    }

                    if (l < numDeepstack && deepstack != null)
                    {
                        var slice = new float[(long)seq * hidden];
                        Array.Copy(deepstack, (long)l * seq * hidden, slice, 0, slice.LongLength);
                        using Tensor tap = _ctx.FromFloats(slice, seq, hidden);
                        Ops.Add(h, h, tap);
                    }
                }
                return DirectOps.ToArray(h);
            }
            finally
            {
                h.Dispose();
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var l in _owned) l.Dispose();
            _owned.Clear();
            _tensorOps.Dispose();
            _ctx.Dispose();
        }

        // CUDA's first projection uploads directly from the mapped pointer.
        // Warming only the consumed matrix ranges keeps page faults out of
        // those synchronous driver copies, without changing any device bytes.
        // The GGUF helper bounds total warming to half the effective host RAM,
        // skips already-resident Linux pages, and joins all readers before use.
        internal static void PrefaultTextWeights(GgufFile file,int layers)
        {
            if (layers <= 0 || Environment.GetEnvironmentVariable("TS_H3_TEXT_PREFAULT") == "0") return;
            try
            {
                // Unknown capacity must not turn optional warming into an
                // unbounded read. GC reports the effective container limit.
                if (GC.GetGCMemoryInfo().TotalAvailableMemoryBytes <= 0) return;
                file.PrefaultFileCache(info => IsTextProjection(info.Name,layers),logProgress:false);
            }
            catch (Exception error) when (error is IOException or UnauthorizedAccessException or OutOfMemoryException)
            {
                // Optional read-ahead may fail under storage/memory pressure;
                // the original mapped uploads retain their strict validation.
            }
        }

        private static bool IsTextProjection(string name,int layers)
        {
            const string prefix="model.layers.";
            if (!name.StartsWith(prefix,StringComparison.Ordinal)) return false;
            ReadOnlySpan<char> remaining=name.AsSpan(prefix.Length);
            int separator=remaining.IndexOf('.');
            if (separator <= 0 || !int.TryParse(remaining[..separator],NumberStyles.None,CultureInfo.InvariantCulture,out int layer) ||
                layer >= layers) return false;
            ReadOnlySpan<char> projection=remaining[(separator+1)..];
            return projection.SequenceEqual("self_attn.q_proj.weight") || projection.SequenceEqual("self_attn.k_proj.weight") ||
                projection.SequenceEqual("self_attn.v_proj.weight") || projection.SequenceEqual("self_attn.o_proj.weight") ||
                projection.SequenceEqual("mlp.gate_proj.weight") || projection.SequenceEqual("mlp.up_proj.weight") ||
                projection.SequenceEqual("mlp.down_proj.weight");
        }
    }
}
