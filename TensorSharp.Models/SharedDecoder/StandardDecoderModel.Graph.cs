// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Whole-forward graphs on the GGML backends. A step - one request's prefill chunk or decode
    /// token, or several requests' rows over their slots - is emitted as ONE ggml graph: every
    /// layer's norms, projections, rotary, cache writes, attention (per sequence, on its slot),
    /// recurrent mixers and FFN or experts, then the head. The family contributes only what is
    /// its own (<see cref="EmitQueryKey"/>, <see cref="EmitRecurrentMixer"/>); everything else is
    /// the runtime's, written once for every family. A step the lowering cannot express runs the
    /// per-op path, unchanged.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Test seam: false keeps every step on the per-op path.</summary>
        internal bool GraphLoweringEnabled { get; set; } = true;

        /// <summary>Steps served by a whole-forward graph (tests check the lowering ran).</summary>
        internal int GraphRuns { get; private set; }

        private string _graphDeclined;

        /// <summary>The family's rotary as one ggml rope over [head_dim, heads, rows]; null when
        /// the family has none the graph can express (the step then runs per op).</summary>
        protected virtual RotarySpec GraphRotary => null;

        /// <summary>Per-head transforms of Q [head_dim, heads, rows] and K [head_dim, kv_heads, rows]
        /// before attention: the rotary, plus whatever the family adds (QK-norm, a position-dependent
        /// query scale). The default rotates both with <see cref="GraphRotary"/>.</summary>
        protected virtual void EmitQueryKey(GraphStep step, int layer, ref int q, ref int k)
        {
            q = step.Rope(q, GraphRotary);
            k = step.Rope(k, GraphRotary);
        }

        /// <summary>A recurrent layer's mixer over the normed input [hidden, rows], per sequence on
        /// that sequence's state (<see cref="GraphStep.SequenceState"/>); returns [hidden, rows].</summary>
        protected virtual int EmitRecurrentMixer(GraphStep step, int layer, int normed)
            => throw new NotSupportedException($"{Config.Architecture} has no graph recurrent mixer.");

        /// <summary>Whether the family's recurrent layers lower to the graph.</summary>
        protected virtual bool RecurrentMixerLowers => false;

        /// <summary>A reason the current configuration cannot run whole-forward graphs, or null.</summary>
        private string GraphUnsupportedReason()
        {
            if (!GraphLoweringEnabled) return "disabled";
            if (!IsGgmlBackend) return "not a GGML backend";
            if (!_mesh.IsSingleDevice) return "multi-device placement";
            if (LayerSplitGroup != null) return "multi-node pipeline";
            if (GraphRotary == null) return "the family has no graph rotary";
            if (_anyRecurrent && !RecurrentMixerLowers) return "recurrent layers without a graph mixer";
            if (StepAdapter != null) return "a request adapter is selected";
            for (int l = 0; l < Config.NumLayers; l++)
                if (_moeLayer[l] && MoeCpuOffloadConfig.IsLayerOnCpu(l)) return "host-offloaded experts";
            DType kv = _kvCacheDtype.ToDType();
            if (kv != DType.Float32 && kv != DType.Float16 && kv != DType.Q8_0 && kv != DType.Q4_0) return $"KV dtype {kv}";
            return null;
        }

        private static int KvGgmlType(DType dt) => dt switch
        {
            DType.Float32 => GgmlType.F32,
            DType.Float16 => GgmlType.F16,
            DType.Q4_0 => GgmlType.Q4_0,
            DType.Q8_0 => GgmlType.Q8_0,
            _ => throw new ArgumentException($"Unsupported KV cache dtype: {dt}"),
        };

        /// <summary>
        /// Run one step as a whole-forward graph. <paramref name="hidden"/> is the embedded rows
        /// [rows, hidden] (media already injected); sequence s owns rows [first[s], first[s] +
        /// counts[s]) and continues <paramref name="slots"/>[s] at starts[s]. Returns each
        /// sequence's last-row logits (or null with <paramref name="wantLogits"/> false); false
        /// when the step cannot be lowered (nothing was written).
        /// </summary>
        private bool TryRunGraph(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            bool wantLogits, out float[][] logits)
        {
            logits = null;
            string reason = GraphUnsupportedReason();
            if (reason != null)
            {
                if (_graphDeclined != reason && reason != "disabled")
                {
                    _graphDeclined = reason;
                    Console.WriteLine($"  {Config.Architecture}: per-op execution ({reason}).");
                }
                return false;
            }

            int n = slots.Length;
            int rows = (int)hidden.Sizes[0];
            using var step = new GraphStep(this, rows, slots, first, counts, starts);
            float[] all = null;
            try
            {
                int h = step.Input(hidden, Config.HiddenSize, rows);
                var positions = new int[rows];
                for (int s = 0; s < n; s++)
                    for (int i = 0; i < counts[s]; i++) positions[first[s] + i] = starts[s] + i;
                step.SetPositions(positions);

                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    int normed = step.RmsNorm(h, $"blk.{layer}.attn_norm.weight");
                    int mix = _recurrentLayer[layer] ? EmitRecurrentMixer(step, layer, normed) : EmitAttention(step, layer, normed);
                    h = step.Op(GgmlGraphOp.Add, h, mix);
                    int ffnIn = step.RmsNorm(h, FfnNormWeight(layer));
                    int ffn = _moeLayer[layer] ? EmitMoe(step, layer, ffnIn) : EmitDenseFfn(step, layer, ffnIn);
                    h = step.Op(GgmlGraphOp.Add, h, ffn);
                }

                if (wantLogits)
                {
                    int final = step.RmsNorm(h, "output_norm.weight");
                    int selected;
                    if (n == 1 && rows == 1) selected = final;
                    else if (n == 1) selected = step.Rows2(final, Config.HiddenSize, rows - 1, 1);
                    else
                    {
                        var last = new int[n];
                        for (int s = 0; s < n; s++) last[s] = first[s] + counts[s] - 1;
                        selected = step.Op(GgmlGraphOp.GetRows, final, step.InputI32(last, n));
                    }
                    int head = step.MatMul(_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")
                        ? "output.weight" : "token_embd.weight", selected);
                    all = new float[(long)Config.VocabSize * n];
                    step.Output(head, all);
                }
                if (!step.Run())
                    throw new InvalidOperationException($"{Config.Architecture} graph step failed: {GgmlBasicOps.LastNativeError()}");
            }
            catch (InvalidOperationException) when (step.Rejected)
            {
                // The lowering met a case it does not express before anything ran.
                return false;
            }
            GraphRuns++;
            foreach (Slot slot in slots) slot.DeviceDirty = true;
            if (all != null)
            {
                int vocab = Config.VocabSize;
                logits = new float[n][];
                for (int s = 0; s < n; s++)
                {
                    logits[s] = new float[vocab];
                    Array.Copy(all, (long)s * vocab, logits[s], 0, vocab);
                }
            }
            return true;
        }

        // ---------------------------------------------------------------- branches

        private int EmitAttention(GraphStep step, int layer, int x)
        {
            string p = $"blk.{layer}.";
            int heads = _localHeads, kvHeads = _localKvHeads, d = _keyDim, dv = _valueDim, rows = step.Rows;
            const long f = sizeof(float);
            long qDim = (long)heads * d, kDim = (long)kvHeads * d, vDim = (long)kvHeads * dv;
            long qWidth = GatedAttentionOutput ? 2 * qDim : qDim;
            long qStride = GatedAttentionOutput ? 2L * d : d;

            int q, k, v, gate = -1;
            if (_qkvFused[layer])
            {
                int qkv = step.MatMul(p + "attn_qkv.weight", x);
                long width = qWidth + kDim + vDim;
                q = step.View3(qkv, d, heads, rows, qStride * f, width * f, 0);
                if (GatedAttentionOutput) gate = step.View3(qkv, d, heads, rows, qStride * f, width * f, d * f);
                k = step.View3(qkv, d, kvHeads, rows, d * f, width * f, qWidth * f);
                v = step.View3(qkv, dv, kvHeads, rows, dv * f, width * f, (qWidth + kDim) * f);
            }
            else
            {
                int qa = step.MatMul(p + "attn_q.weight", x);
                q = step.View3(qa, d, heads, rows, qStride * f, qWidth * f, 0);
                if (GatedAttentionOutput) gate = step.View3(qa, d, heads, rows, qStride * f, qWidth * f, d * f);
                k = step.Reshape(step.MatMul(p + "attn_k.weight", x), d, kvHeads, rows);
                v = step.Reshape(step.MatMul(p + "attn_v.weight", x), dv, kvHeads, rows);
            }
            q = step.Op(GgmlGraphOp.Cont, q);
            k = step.Op(GgmlGraphOp.Cont, k);
            v = step.Op(GgmlGraphOp.Cont, v);
            EmitQueryKey(step, layer, ref q, ref k);

            int kvType = KvGgmlType(_kvCacheDtype.ToDType());
            float scale = 1.0f / MathF.Sqrt(d);
            int n = step.Sequences;
            int attn = -1;
            for (int s = 0; s < n; s++)
            {
                int count = step.Counts[s], start = step.Starts[s];
                Slot slot = step.Slots[s];
                int qs = n == 1 ? q : step.Rows3(q, step.First[s], count);
                int ks = n == 1 ? k : step.Rows3(k, step.First[s], count);
                int vs = n == 1 ? v : step.Rows3(v, step.First[s], count);
                int qPerm = step.Op(GgmlGraphOp.Permute, stackalloc[] { qs }, stackalloc long[] { 0, 2, 1, 3 });
                int kFresh = step.Op(GgmlGraphOp.Cont, step.Op(GgmlGraphOp.Permute, stackalloc[] { ks }, stackalloc long[] { 0, 2, 1, 3 }));
                int vFresh = step.Op(GgmlGraphOp.Cont, step.Op(GgmlGraphOp.Permute, stackalloc[] { vs }, stackalloc long[] { 0, 2, 1, 3 }));

                Tensor cacheK = slot.K[layer][0], cacheV = slot.V[layer][0];
                int cap = slot.Capacity;
                int ck = step.State(cacheK, kvType, d, cap, kvHeads);
                int cv = step.State(cacheV, kvType, dv, cap, kvHeads);
                var (_, nbK, _) = step.G.Shape(ck);
                var (_, nbV, _) = step.G.Shape(cv);
                step.G.Effect(step.Op(GgmlGraphOp.Cpy, kFresh, step.View3(ck, d, count, kvHeads, nbK[1], nbK[2], start * nbK[1])));
                step.G.Effect(step.Op(GgmlGraphOp.Cpy, vFresh, step.View3(cv, dv, count, kvHeads, nbV[1], nbV[2], start * nbV[1])));

                int fa;
                if (count == 1)
                {
                    // Decode reads the cache (the row just written included), as a single request does.
                    int gqa = heads / kvHeads;
                    int kWin = step.Op(GgmlGraphOp.KvWindow, stackalloc[] { ck }, stackalloc long[] { d, cap, kvHeads, 0, start + 1, kvType, 1, gqa });
                    int vWin = step.Op(GgmlGraphOp.KvWindow, stackalloc[] { cv }, stackalloc long[] { dv, cap, kvHeads, 0, start + 1, kvType, 1, gqa });
                    fa = step.FlashAttention(qPerm, kWin, vWin, -1, scale);
                }
                else
                {
                    // A prompt chunk attends to the cached prefix and its own fresh rows (F32).
                    int kAttn = kFresh, vAttn = vFresh;
                    if (start > 0)
                    {
                        int kPrev = step.View3(ck, d, start, kvHeads, nbK[1], nbK[2], 0);
                        int vPrev = step.View3(cv, dv, start, kvHeads, nbV[1], nbV[2], 0);
                        if (kvType != GgmlType.F32)
                        {
                            kPrev = step.Op(GgmlGraphOp.Cast, stackalloc[] { kPrev }, stackalloc long[] { GgmlType.F32 });
                            vPrev = step.Op(GgmlGraphOp.Cast, stackalloc[] { vPrev }, stackalloc long[] { GgmlType.F32 });
                        }
                        kAttn = step.Op(GgmlGraphOp.Concat, stackalloc[] { kPrev, kFresh }, stackalloc long[] { 1 });
                        vAttn = step.Op(GgmlGraphOp.Concat, stackalloc[] { vPrev, vFresh }, stackalloc long[] { 1 });
                    }
                    fa = step.FlashAttention(qPerm, kAttn, vAttn, step.CausalMask(start, count), scale);
                }
                int part = step.Reshape(fa, heads * dv, count);
                attn = attn < 0 ? part : step.Op(GgmlGraphOp.Concat, stackalloc[] { attn, part }, stackalloc long[] { 1 });
            }

            if (gate >= 0)
            {
                int g = step.Reshape(step.Op(GgmlGraphOp.Cont, gate), (int)qDim, rows);
                attn = step.Op(GgmlGraphOp.Mul, attn, step.Op(GgmlGraphOp.Sigmoid, g));
            }
            return step.MatMul(p + "attn_output.weight", attn);
        }

        private int EmitDenseFfn(GraphStep step, int layer, int x)
        {
            string p = $"blk.{layer}.";
            int act = _gateUpFused[layer]
                ? step.Op(GgmlGraphOp.Swiglu, step.MatMul(p + "ffn_gate_up.weight", x))
                : step.Op(GgmlGraphOp.SwigluSplit, step.MatMul(p + "ffn_gate.weight", x), step.MatMul(p + "ffn_up.weight", x));
            return step.MatMul(p + "ffn_down.weight", act);
        }

        private int EmitMoe(GraphStep step, int layer, int x)
        {
            string p = $"blk.{layer}.";
            int rows = step.Rows, hidden = Config.HiddenSize, k = _expertsUsed;
            int probs = step.Op(GgmlGraphOp.SoftMaxExt, step.MatMul(p + "ffn_gate_inp.weight", x));   // [E, rows]
            int selected = step.Op(GgmlGraphOp.TopK, stackalloc[] { probs }, stackalloc long[] { k });      // [K, rows]
            int weights = step.Op(GgmlGraphOp.GetRows, step.Reshape(probs, 1, _numExperts, rows), selected); // [1, K, rows]
            if (NormalizeExpertWeights)
            {
                int sum = step.Op(GgmlGraphOp.SumRows, step.Reshape(weights, k, rows));                     // [1, rows]
                weights = step.Op(GgmlGraphOp.Div, weights, step.Reshape(sum, 1, 1, rows));
            }

            int input = step.Reshape(x, hidden, 1, rows);
            int gate = step.Op(GgmlGraphOp.MulMatId, step.Experts(p + "ffn_gate_exps.weight"), input, selected);
            int up = step.Op(GgmlGraphOp.MulMatId, step.Experts(p + "ffn_up_exps.weight"), input, selected);
            int act = step.Op(GgmlGraphOp.SwigluSplit, gate, up);                                            // [ff, K, rows]
            int down = step.Op(GgmlGraphOp.MulMatId, step.Experts(p + "ffn_down_exps.weight"), act, selected); // [hidden, K, rows]
            down = step.Op(GgmlGraphOp.Mul, down, weights);
            var (_, nb, _) = step.G.Shape(down);
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, hidden, rows, nb[2], j * nb[1]);
                output = output < 0 ? slice : step.Op(GgmlGraphOp.Add, output, slice);
            }
            output = step.Op(GgmlGraphOp.Cont, output);

            if (HasWeight(p + "ffn_gate_shexp.weight"))
            {
                int sa = step.Op(GgmlGraphOp.SwigluSplit, step.MatMul(p + "ffn_gate_shexp.weight", x), step.MatMul(p + "ffn_up_shexp.weight", x));
                int shared = step.MatMul(p + "ffn_down_shexp.weight", sa);
                if (_weights.ContainsKey(p + "ffn_gate_inp_shexp.weight"))
                {
                    int g = step.Op(GgmlGraphOp.MulMat, step.WeightAs(p + "ffn_gate_inp_shexp.weight", hidden, 1), x);   // [1, rows]
                    shared = step.Op(GgmlGraphOp.Mul, shared, step.Op(GgmlGraphOp.Sigmoid, g));
                }
                output = step.Op(GgmlGraphOp.Add, output, shared);
            }
            return output;
        }

        // ---------------------------------------------------------------- the step

        /// <summary>One graph under construction: its sequences and the helpers branch code uses.</summary>
        protected sealed class GraphStep : IDisposable
        {
            private readonly StandardDecoderModel _m;
            private readonly Dictionary<string, int> _weights = new(StringComparer.Ordinal);
            private readonly Dictionary<Tensor, int> _states = new(ReferenceEqualityComparer.Instance);
            private readonly List<GCHandle> _pins = new();
            private readonly List<int> _inputs = new();
            private readonly List<IntPtr> _inputData = new();
            private readonly List<long> _inputBytes = new();
            private int _mask = -1;
            private int _maskStart = -1, _maskCount = -1;

            internal GraphStep(StandardDecoderModel model, int rows, Slot[] slots, int[] first, int[] counts, int[] starts)
            {
                _m = model;
                Rows = rows;
                Slots = slots;
                First = first;
                Counts = counts;
                Starts = starts;
                G = new GgmlGraph(16384);
            }

            internal GgmlGraph G { get; }
            internal Slot[] Slots { get; }
            public int Rows { get; }
            public int Sequences => Slots.Length;
            public int[] First { get; }
            public int[] Counts { get; }
            public int[] Starts { get; }
            /// <summary>Token positions [rows] (I32).</summary>
            public int Positions { get; private set; } = -1;
            /// <summary>The positions' values (row order).</summary>
            public int[] PositionValues { get; private set; }
            internal bool Rejected { get; private set; }

            public int Op(GgmlGraphOp op, int a) => G.Op(op, a);
            public int Op(GgmlGraphOp op, int a, int b) => G.Op(op, a, b);
            public int Op(GgmlGraphOp op, int a, int b, int c) => G.Op(op, a, b, c);
            public int Op(GgmlGraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
                => G.Op(op, srcs, ip, fp);

            /// <summary>A model weight by name: quantized (its device-cache key) or F32.</summary>
            public int Weight(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (_m._quantWeights.TryGetValue(name, out QuantizedWeight q))
                {
                    IntPtr key = q.CacheKey != IntPtr.Zero ? q.CacheKey : q.Data;
                    node = G.Leaf(GgmlLeafRole.Weight, q.GgmlType, stackalloc long[] { q.Ne0, q.Ne1 }, key, q.RawBytes);
                }
                else if (_m._weights.TryGetValue(name, out Tensor t))
                {
                    node = TensorLeaf(GgmlLeafRole.Weight, t, GgmlType.F32, Extents(t));
                }
                else
                {
                    throw Reject($"weight '{name}' is missing");
                }
                _weights[name] = node;
                return node;
            }

            /// <summary>An F32 weight viewed with the given extents (a vector as a [n, 1] matrix).</summary>
            public int WeightAs(string name, params long[] ne)
            {
                if (!_m._weights.TryGetValue(name, out Tensor t)) throw Reject($"weight '{name}' is missing");
                return TensorLeaf(GgmlLeafRole.Weight, t, GgmlType.F32, ne);
            }

            /// <summary>Stacked experts [ne0, ne1, experts] of a routed-expert layer.</summary>
            public int Experts(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (!_m._stackedExpertWeights.TryGetValue(name, out StackedExpertWeights w)) throw Reject($"experts '{name}' are not stacked");
                node = G.Leaf(GgmlLeafRole.Weight, w.GgmlType, stackalloc long[] { w.PerExpertNe0, w.PerExpertNe1, w.NumExperts }, w.Data, w.TotalRawBytes);
                _weights[name] = node;
                return node;
            }

            /// <summary>Read-write state the graph updates in place (a slot's cache or recurrent state).</summary>
            public int State(Tensor t, int type, params long[] ne)
            {
                if (_states.TryGetValue(t, out int node)) return node;
                node = TensorLeaf(GgmlLeafRole.State, t, type, ne);
                _states[t] = node;
                return node;
            }

            /// <summary>Sequence s's recurrent state of (layer, rank 0).</summary>
            internal RecurrentState SequenceState(int s, int layer) => Slots[s].Recurrent[layer][0];

            private int TensorLeaf(GgmlLeafRole role, Tensor t, int type, ReadOnlySpan<long> ne)
            {
                IntPtr data = TensorComputePrimitives.GetStoragePointer(t);
                return G.Leaf(role, type, ne, data, t.Storage.ByteLength);
            }

            private static long[] Extents(Tensor t)
            {
                var ne = new long[t.DimensionCount];
                for (int i = 0; i < ne.Length; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
                return ne;
            }

            /// <summary>An F32 activation [cols, rows] read from a host tensor every run.</summary>
            internal int Input(Tensor t, long cols, long rows)
            {
                int node = G.Leaf(GgmlLeafRole.Input, GgmlType.F32, stackalloc long[] { cols, rows }, IntPtr.Zero, 0);
                AddInput(node, TensorComputePrimitives.GetStoragePointer(t), cols * rows * sizeof(float));
                return node;
            }

            public int InputI32(int[] values, long count)
            {
                int node = G.Leaf(GgmlLeafRole.Input, GgmlType.I32, stackalloc long[] { count }, IntPtr.Zero, 0);
                AddInput(node, Pin(values), count * sizeof(int));
                return node;
            }

            public int InputF32(float[] values, params long[] ne)
            {
                int node = G.Leaf(GgmlLeafRole.Input, GgmlType.F32, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), (long)values.Length * sizeof(float));
                return node;
            }

            internal void SetPositions(int[] positions)
            {
                PositionValues = positions;
                Positions = InputI32(positions, positions.Length);
            }

            /// <summary>An F32 tensor the family prepared (device constants, norms) as a read-only leaf.</summary>
            public int TensorWeight(Tensor t, params long[] ne) => TensorLeaf(GgmlLeafRole.Weight, t, GgmlType.F32, ne.Length == 0 ? Extents(t) : ne);

            /// <summary>rms_norm(x, eps / n) / sqrt(n) over ne0: the L2 normalization ggml's GDN path uses.</summary>
            public int L2Norm(int x, long n, float eps)
                => Op(GgmlGraphOp.Scale, stackalloc[] { Op(GgmlGraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps / n }) },
                    default, stackalloc[] { 1.0f / MathF.Sqrt(n) });

            /// <summary>A causal F16 mask [start + count, count] for a chunk continuing at <paramref name="start"/>.</summary>
            public int CausalMask(int start, int count)
            {
                if (_mask >= 0 && _maskStart == start && _maskCount == count) return _mask;
                int kv = start + count;
                var mask = new ushort[(long)kv * count];
                const ushort negInf = 0xFC00;
                for (int qi = 0; qi < count; qi++)
                    for (int ki = start + qi + 1; ki < kv; ki++) mask[(long)qi * kv + ki] = negInf;
                _mask = G.Leaf(GgmlLeafRole.Input, GgmlType.F16, stackalloc long[] { kv, count }, IntPtr.Zero, 0);
                AddInput(_mask, Pin(mask), mask.LongLength * sizeof(ushort));
                _maskStart = start;
                _maskCount = count;
                return _mask;
            }

            private IntPtr Pin(Array values)
            {
                GCHandle handle = GCHandle.Alloc(values, GCHandleType.Pinned);
                _pins.Add(handle);
                return handle.AddrOfPinnedObject();
            }

            private void AddInput(int node, IntPtr data, long bytes)
            {
                _inputs.Add(node);
                _inputData.Add(data);
                _inputBytes.Add(bytes);
            }

            public void Output(int node, float[] destination) => G.Output(node, Pin(destination), destination.LongLength * sizeof(float));

            // ---- composites

            /// <summary>rms_norm(x) * weight.</summary>
            public int RmsNorm(int x, string weight) => RmsNorm(x, Weight(weight), _m.Config.Eps);

            public int RmsNorm(int x, int weight, float eps)
                => Op(GgmlGraphOp.Mul, Op(GgmlGraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), weight);

            /// <summary>weight · x, with a quantized weight's per-tensor scale applied.</summary>
            public int MatMul(string weight, int x)
            {
                int y = Op(GgmlGraphOp.MulMat, Weight(weight), x);
                if (_m._quantWeights.TryGetValue(weight, out QuantizedWeight q) && q.Scale != 1.0f)
                    y = Op(GgmlGraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { q.Scale });
                return y;
            }

            public int Reshape(int x, params long[] ne) => Op(GgmlGraphOp.Reshape, stackalloc[] { x }, ne);

            public int View2(int x, long ne0, long ne1, long nb1, long offset)
                => Op(GgmlGraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });

            public int View3(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
                => Op(GgmlGraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });

            /// <summary>Rows [first, first + count) of a 2-D [cols, rows] node.</summary>
            public int Rows2(int x, long cols, long first, long count)
            {
                var (_, nb, _) = G.Shape(x);
                return View2(x, cols, count, nb[1], first * nb[1]);
            }

            /// <summary>Rows [first, first + count) (ne2) of a 3-D [a, b, rows] node.</summary>
            public int Rows3(int x, long first, long count)
            {
                var (ne, nb, _) = G.Shape(x);
                return View3(x, ne[0], ne[1], count, nb[1], nb[2], first * nb[2]);
            }

            public int Rope(int x, RotarySpec spec)
                => Op(GgmlGraphOp.RopeExt, stackalloc[] { x, Positions },
                    stackalloc long[] { spec.Dims, spec.NeoX ? 2 : 0, spec.OriginalContext },
                    stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });

            /// <summary>Flash attention of q [d, rows, heads] over k/v [d, kv, kv_heads]: [dv, heads, rows].</summary>
            public int FlashAttention(int q, int k, int v, int mask, float scale)
            {
                Span<int> srcs = stackalloc[] { q, k, v, mask };
                return Op(GgmlGraphOp.FlashAttnExt, mask >= 0 ? srcs : srcs[..3], stackalloc long[] { 10 /* F32 accumulation */ },
                    stackalloc[] { scale, 0f, 0f });
            }

            private InvalidOperationException Reject(string why)
            {
                Rejected = true;
                return new InvalidOperationException(why);
            }

            internal bool Run() => G.Run(
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_inputs),
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_inputData),
                System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_inputBytes));

            public void Dispose()
            {
                G.Dispose();
                foreach (GCHandle pin in _pins) pin.Free();
                _pins.Clear();
            }
        }
    }
}
