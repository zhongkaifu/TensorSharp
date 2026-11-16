// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Qwen3-style transformer pieces every part of ACE-Step 1.5 is made of, on the direct tensor
// primitives (DirectOps: --backend cuda with resident BF16/F16 weights, or --backend cpu):
// grouped-query attention with per-head RMS QK-norm and NeoX rotary embeddings, SwiGLU MLPs,
// pre-norm residual blocks, and a causal decoder with a KV cache for the planner LM.
using System;
using System.Collections.Generic;
using TensorSharp.Models.Direct;
using TensorSharp.Runtime;

namespace TensorSharp.Models.AceStep
{
    internal sealed class AceAttention
    {
        public DirectLinear Q, K, V, O;
        public Tensor QNorm, KNorm;
        public int Heads, KvHeads, HeadDim;

        public static AceAttention Load(DirectContext ctx, GgufFile gguf, string p, int heads, int kvHeads, int headDim) => new()
        {
            Q = DirectLinear.FromGguf(ctx, gguf, p + "q_proj.weight", p + "q_proj.bias"),
            K = DirectLinear.FromGguf(ctx, gguf, p + "k_proj.weight", p + "k_proj.bias"),
            V = DirectLinear.FromGguf(ctx, gguf, p + "v_proj.weight", p + "v_proj.bias"),
            O = DirectLinear.FromGguf(ctx, gguf, p + "o_proj.weight", p + "o_proj.bias"),
            QNorm = gguf.Tensors.ContainsKey(p + "q_norm.weight") ? AceOps.Vec(ctx, gguf, p + "q_norm.weight") : null,
            KNorm = gguf.Tensors.ContainsKey(p + "k_norm.weight") ? AceOps.Vec(ctx, gguf, p + "k_norm.weight") : null,
            Heads = heads, KvHeads = kvHeads, HeadDim = headDim,
        };
    }

    internal sealed class AceMlp
    {
        public DirectLinear Gate, Up, Down;

        public static AceMlp Load(DirectContext ctx, GgufFile gguf, string p) => new()
        {
            Gate = DirectLinear.FromGguf(ctx, gguf, p + "gate_proj.weight"),
            Up = DirectLinear.FromGguf(ctx, gguf, p + "up_proj.weight"),
            Down = DirectLinear.FromGguf(ctx, gguf, p + "down_proj.weight"),
        };

        public Tensor Forward(DirectContext ctx, Tensor x)
        {
            using var g = Gate.Forward(x);
            using var u = Up.Forward(x);
            Ops.SiLUMul(g, g, u);
            return Down.Forward(g);
        }
    }

    internal static class AceOps
    {
        public const float Eps = 1e-6f;

        public static Tensor Vec(DirectContext ctx, GgufFile gguf, string name)
        {
            float[] v = DirectOps.DequantTensor(gguf, name);
            return ctx.Own(ctx.FromFloats(v, v.LongLength));
        }

        /// <summary>RMSNorm over each head of x [rows, heads * hd] with a [hd] gain.</summary>
        public static Tensor HeadNorm(DirectContext ctx, Tensor x, Tensor gain, int heads, int hd)
        {
            long rows = x.Sizes[0];
            using var flat = x.View(rows * heads, hd);
            using var normed = DirectOps.RmsNorm(ctx, flat, gain, Eps);
            return normed.View(rows, (long)heads * hd);
        }

        /// <summary>NeoX (rotate-half) rotary embedding in place on x [rows, heads * hd] at the given positions.</summary>
        public static void Rope(DirectContext ctx, Tensor x, int heads, int hd, int[] positions, float theta)
        {
            int rows = positions.Length;
            var pos = new int[rows * heads];
            for (int r = 0; r < rows; r++)
                for (int h = 0; h < heads; h++) pos[r * heads + h] = positions[r];
            using var posT = new Tensor(ctx.Allocator, DType.Int32, pos.Length);
            posT.SetElementsAsInt(pos);
            using var view = x.View(1, rows, heads, hd);
            Ops.RoPEEx(view, view, posT, hd, 2, 0, theta, 1f, 0f, 1f, 0f, 0f);
        }

        /// <summary>Repeat each of the kvHeads heads of x [rows, kvHeads * hd] for its query group: [rows, heads * hd].</summary>
        public static Tensor ExpandKv(DirectContext ctx, Tensor x, int heads, int kvHeads, int hd)
        {
            if (heads == kvHeads) return x.CopyRef();
            long rows = x.Sizes[0];
            var y = ctx.NewF32(rows, (long)heads * hd);
            int group = heads / kvHeads;
            for (int h = 0; h < heads; h++)
            {
                using var dst = y.Narrow(1, (long)h * hd, hd);
                using var src = x.Narrow(1, (long)(h / group) * hd, hd);
                Ops.Copy(dst, src);
            }
            return y;
        }

        public static int[] Range(int start, int count)
        {
            var r = new int[count];
            for (int i = 0; i < count; i++) r[i] = start + i;
            return r;
        }

        /// <summary>
        /// Attention of x's queries over keys/values from <paramref name="kv"/> (x itself for self-attention).
        /// Rotary embeddings apply to self-attention only (positions null leaves them out).
        /// </summary>
        public static Tensor Attend(DirectContext ctx, AceAttention a, Tensor x, Tensor kv, int[] qPositions, int[] kPositions,
            float theta, Tensor bias)
        {
            using var q0 = a.Q.Forward(x);
            using var q = a.QNorm != null ? HeadNorm(ctx, q0, a.QNorm, a.Heads, a.HeadDim) : Keep(q0);
            using var k0 = a.K.Forward(kv);
            using var k = a.KNorm != null ? HeadNorm(ctx, k0, a.KNorm, a.KvHeads, a.HeadDim) : Keep(k0);
            using var v = a.V.Forward(kv);
            if (qPositions != null)
            {
                Rope(ctx, q, a.Heads, a.HeadDim, qPositions, theta);
                Rope(ctx, k, a.KvHeads, a.HeadDim, kPositions ?? qPositions, theta);
            }
            using var kf = ExpandKv(ctx, k, a.Heads, a.KvHeads, a.HeadDim);
            using var vf = ExpandKv(ctx, v, a.Heads, a.KvHeads, a.HeadDim);
            using var o = DirectOps.Attention(ctx, q, kf, vf, a.Heads, a.HeadDim, 1f / MathF.Sqrt(a.HeadDim), bias);
            return a.O.Forward(o);
        }

        private static Tensor Keep(Tensor t) => t.CopyRef();

        /// <summary>An additive [heads, n, n] bias that keeps |i - j| &lt;= window (ACE-Step's sliding layers).</summary>
        public static Tensor WindowBias(DirectContext ctx, int heads, int n, int window)
        {
            var b = new float[(long)heads * n * n];
            for (int i = 0; i < n; i++)
                for (int j = 0; j < n; j++)
                    if (Math.Abs(i - j) > window)
                        for (int h = 0; h < heads; h++) b[((long)h * n + i) * n + j] = float.MinValue / 4;
            return ctx.FromFloats(b, heads, n, n);
        }
    }

    /// <summary>A pre-norm transformer block: x += attn(norm(x)); x += mlp(norm(x)).</summary>
    internal sealed class AceEncoderLayer
    {
        private readonly Tensor _ln1, _ln2;
        private readonly AceAttention _attn;
        private readonly AceMlp _mlp;

        public AceEncoderLayer(DirectContext ctx, GgufFile gguf, string p, int heads, int kvHeads, int hd)
        {
            _ln1 = AceOps.Vec(ctx, gguf, p + "input_layernorm.weight");
            _ln2 = AceOps.Vec(ctx, gguf, p + "post_attention_layernorm.weight");
            _attn = AceAttention.Load(ctx, gguf, p + "self_attn.", heads, kvHeads, hd);
            _mlp = AceMlp.Load(ctx, gguf, p + "mlp.");
        }

        public void Forward(DirectContext ctx, Tensor x, int[] positions, float theta, Tensor bias)
        {
            using (var n1 = DirectOps.RmsNorm(ctx, x, _ln1, AceOps.Eps))
            using (var a = AceOps.Attend(ctx, _attn, n1, n1, positions, positions, theta, bias))
                Ops.Add(x, x, a);
            using (var n2 = DirectOps.RmsNorm(ctx, x, _ln2, AceOps.Eps))
            using (var m = _mlp.Forward(ctx, n2))
                Ops.Add(x, x, m);
        }
    }

    /// <summary>
    /// A Qwen3 decoder (ACE-Step's caption encoder and planner LM): token embeddings, causal blocks with
    /// a KV cache, and the final RMSNorm. Keys at padding positions are masked out; their positions still
    /// count, as ComfyUI's left-padded CFG batch has them.
    /// </summary>
    internal sealed class AceQwen3 : IDisposable
    {
        private readonly DirectContext _ctx;
        private readonly GgufFile _gguf;
        private readonly string _p;
        private readonly Tensor[] _ln1, _ln2;
        private readonly AceAttention[] _attn;
        private readonly AceMlp[] _mlp;
        private readonly Tensor _norm;

        public int Hidden { get; }
        public int Layers { get; }
        public int Heads { get; }
        public int KvHeads { get; }
        public int HeadDim { get; }
        public float Theta { get; } = 1_000_000f;
        public string EmbedName => _p + "embed_tokens.weight";

        public AceQwen3(DirectContext ctx, GgufFile gguf, string prefix, int heads = 16, int kvHeads = 8, int headDim = 128)
        {
            _ctx = ctx;
            _gguf = gguf;
            _p = prefix;
            Hidden = (int)gguf.Tensors[EmbedName].Shape[0];
            int layers = 0;
            while (gguf.Tensors.ContainsKey($"{prefix}layers.{layers}.input_layernorm.weight")) layers++;
            Layers = layers;
            Heads = heads; KvHeads = kvHeads; HeadDim = headDim;
            _ln1 = new Tensor[layers]; _ln2 = new Tensor[layers];
            _attn = new AceAttention[layers]; _mlp = new AceMlp[layers];
            for (int i = 0; i < layers; i++)
            {
                string p = $"{prefix}layers.{i}.";
                _ln1[i] = AceOps.Vec(ctx, gguf, p + "input_layernorm.weight");
                _ln2[i] = AceOps.Vec(ctx, gguf, p + "post_attention_layernorm.weight");
                _attn[i] = AceAttention.Load(ctx, gguf, p + "self_attn.", heads, kvHeads, headDim);
                _mlp[i] = AceMlp.Load(ctx, gguf, p + "mlp.");
            }
            _norm = AceOps.Vec(ctx, gguf, prefix + "norm.weight");
        }

        /// <summary>Token embeddings [n, hidden] on the host.</summary>
        public float[] Embed(IReadOnlyList<int> ids)
        {
            var rows = new int[ids.Count];
            for (int i = 0; i < rows.Length; i++) rows[i] = ids[i];
            var e = new float[(long)rows.Length * Hidden];
            DirectOps.DequantRows(_gguf, EmbedName, rows, e);
            return e;
        }

        /// <summary>A KV cache for one sequence of at most <paramref name="capacity"/> positions.</summary>
        public AceKvCache NewCache(int capacity) => new(_ctx, Layers, capacity, KvHeads * HeadDim);

        /// <summary>
        /// Run <paramref name="ids"/> at positions start .. start+n-1 and return the final-normed hidden
        /// states of every row (the caller disposes). <paramref name="keyValid"/> (length start+n) marks
        /// the cache positions queries may attend to; null attends to all of them.
        /// </summary>
        public Tensor Forward(IReadOnlyList<int> ids, int start, bool[] keyValid, AceKvCache cache)
        {
            int n = ids.Count, total = start + n;
            if (total > cache.Capacity) throw new InvalidOperationException($"the KV cache holds {cache.Capacity} positions; {total} asked.");
            var x = _ctx.FromFloats(Embed(ids), n, Hidden);
            // Causal mask plus masked keys: [heads, n, total].
            Tensor bias = null;
            bool anyMasked = keyValid != null && Array.IndexOf(keyValid, false) >= 0;
            if (n > 1 || anyMasked)
            {
                var b = new float[(long)Heads * n * total];
                for (int i = 0; i < n; i++)
                    for (int j = 0; j < total; j++)
                    {
                        bool masked = j > start + i || (keyValid != null && !keyValid[j]);
                        if (!masked) continue;
                        for (int h = 0; h < Heads; h++) b[((long)h * n + i) * total + j] = float.MinValue / 4;
                    }
                bias = _ctx.FromFloats(b, Heads, n, total);
            }
            int[] positions = AceOps.Range(start, n);
            try
            {
                for (int li = 0; li < Layers; li++)
                {
                    var a = _attn[li];
                    using (var xn = DirectOps.RmsNorm(_ctx, x, _ln1[li], AceOps.Eps))
                    {
                        using var q0 = a.Q.Forward(xn);
                        using var q = AceOps.HeadNorm(_ctx, q0, a.QNorm, a.Heads, a.HeadDim);
                        using var k0 = a.K.Forward(xn);
                        using var k = AceOps.HeadNorm(_ctx, k0, a.KNorm, a.KvHeads, a.HeadDim);
                        using var v = a.V.Forward(xn);
                        AceOps.Rope(_ctx, q, a.Heads, a.HeadDim, positions, Theta);
                        AceOps.Rope(_ctx, k, a.KvHeads, a.HeadDim, positions, Theta);
                        using (var kd = cache.K[li].Narrow(0, start, n)) Ops.Copy(kd, k);
                        using (var vd = cache.V[li].Narrow(0, start, n)) Ops.Copy(vd, v);
                        using var kAll = cache.K[li].Narrow(0, 0, total);
                        using var vAll = cache.V[li].Narrow(0, 0, total);
                        using var kf = AceOps.ExpandKv(_ctx, kAll, a.Heads, a.KvHeads, a.HeadDim);
                        using var vf = AceOps.ExpandKv(_ctx, vAll, a.Heads, a.KvHeads, a.HeadDim);
                        using var o = DirectOps.Attention(_ctx, q, kf, vf, a.Heads, a.HeadDim, 1f / MathF.Sqrt(a.HeadDim), bias);
                        using var ao = a.O.Forward(o);
                        Ops.Add(x, x, ao);
                    }
                    using (var xn = DirectOps.RmsNorm(_ctx, x, _ln2[li], AceOps.Eps))
                    using (var m = _mlp[li].Forward(_ctx, xn))
                        Ops.Add(x, x, m);
                }
                return DirectOps.RmsNorm(_ctx, x, _norm, AceOps.Eps);
            }
            finally
            {
                x.Dispose();
                bias?.Dispose();
            }
        }

        public void Dispose() { }
    }

    /// <summary>Per-layer key and value rows [capacity, kvHeads * headDim] of one sequence.</summary>
    internal sealed class AceKvCache : IDisposable
    {
        public Tensor[] K { get; }
        public Tensor[] V { get; }
        public int Capacity { get; }

        public AceKvCache(DirectContext ctx, int layers, int capacity, int width)
        {
            Capacity = capacity;
            K = new Tensor[layers];
            V = new Tensor[layers];
            for (int i = 0; i < layers; i++)
            {
                K[i] = ctx.NewF32(capacity, width);
                V[i] = ctx.NewF32(capacity, width);
            }
        }

        public void Dispose()
        {
            foreach (var t in K) t?.Dispose();
            foreach (var t in V) t?.Dispose();
        }
    }
}
