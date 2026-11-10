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

        /// <summary>Let ggml-metal reorder the graph for concurrency. Off by default: it widens the
        /// live set and measured slower on some families; a family opts in where it measures faster.</summary>
        protected virtual bool GraphReorderOnMetal => false;

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

        // Decode graphs (every sequence one row) are replayed while their trace matches.
        private const int GraphCacheEntries = 32;   // decode + verify widths x window buckets
        private readonly Dictionary<(ulong, ulong), CachedGraph> _graphCache = new();
        private readonly LinkedList<(ulong, ulong)> _graphLru = new();

        /// <summary>A built graph kept for replay, with the buffer its logits land in.</summary>
        internal sealed class CachedGraph : IDisposable
        {
            public required GgmlGraph Graph { get; init; }
            /// <summary>The step's results, in the order they were declared (GraphStep.Output).</summary>
            public required float[][] Outputs { get; init; }
            public required GCHandle[] Pins { get; init; }

            public void Dispose()
            {
                Graph.Dispose();
                foreach (GCHandle pin in Pins) if (pin.IsAllocated) pin.Free();
            }
        }

        /// <summary>What a step returns besides advancing the caches.</summary>
        private sealed class GraphRequest
        {
            /// <summary>Logits of each sequence's last row, or of every row.</summary>
            public bool LastRowLogits { get; init; } = true;
            public bool AllRowLogits { get; init; }
            /// <summary>Layers whose INPUT residual [hidden, rows] is returned (a drafter's features).</summary>
            public int[] CaptureLayers { get; init; } = Array.Empty<int>();
            /// <summary>Return the final-normed rows [hidden, rows] (a per-token draft head's input).</summary>
            public bool CaptureFinal { get; init; }
            /// <summary>Keep the recurrent state after every row (a verify the accept count rewinds).</summary>
            public bool RecurrentSnapshots { get; init; }
            /// <summary>Flat (T, H, W) rotary positions of a single sequence's rows (multi-axis rotary).</summary>
            public int[] PositionTable { get; init; }
        }

        private sealed class GraphResult
        {
            public float[] Logits;
            public float[][] Captures;
            public float[] Final;
        }

        /// <summary>Rows per sequence up to which a step writes K/V by index and attends to the padded
        /// cache window (as decode does), so the graph replays; longer chunks attend to their fresh rows.</summary>
        internal const int SmallChunkRows = 16;

        /// <summary>Drop replayable graphs: a bound cache or state was reallocated, freed, or
        /// rewritten on the host, so their device bindings may be stale.</summary>
        private void ClearGraphCache()
        {
            OnGraphCacheCleared();
            foreach (CachedGraph entry in _graphCache.Values) entry.Dispose();
            _graphCache.Clear();
            _graphLru.Clear();
        }

        /// <summary>Steps that replayed a cached graph (tests check decode replays).</summary>
        internal int GraphReplays { get; private set; }

        /// <summary>
        /// Run one step as a whole-forward graph. <paramref name="hidden"/> is the embedded rows
        /// [rows, hidden] (media already injected); sequence s owns rows [first[s], first[s] +
        /// counts[s]) and continues <paramref name="slots"/>[s] at starts[s]. Returns each
        /// sequence's last-row logits (or null with <paramref name="wantLogits"/> false); false
        /// when the step cannot be lowered (nothing was written).
        /// </summary>
        private bool TryRunGraph(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            bool wantLogits, out float[][] logits, int[] positionTable = null)
        {
            logits = null;
            var request = new GraphRequest { LastRowLogits = wantLogits, PositionTable = positionTable };
            if (!TryRunGraphStep(hidden, slots, first, counts, starts, request, out GraphResult result))
                return false;
            if (result.Logits != null)
            {
                int n = slots.Length, vocab = Config.VocabSize;
                logits = new float[n][];
                if (n == 1)
                {
                    logits[0] = result.Logits;
                }
                else
                {
                    for (int s = 0; s < n; s++)
                    {
                        logits[s] = new float[vocab];
                        Array.Copy(result.Logits, (long)s * vocab, logits[s], 0, vocab);
                    }
                }
            }
            return true;
        }

        private bool TryRunGraphStep(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            GraphRequest request, out GraphResult result)
        {
            result = null;
            string reason = GraphUnsupportedReason();
            if (reason != null)
            {
                if (_graphDeclined != reason && reason != "disabled")
                {
                    _graphDeclined = reason;
                    Console.WriteLine($"  {Config.Architecture}: per-op execution ({reason}).");
                }
                ClearGraphCache();
                return false;
            }

            int rows = (int)hidden.Sizes[0];
            bool replayable = true;
            foreach (int c in counts) replayable &= c <= SmallChunkRows;
            CachedGraph graph = RunTraced(rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, 0,
                step => EmitStep(step, hidden, request), out StepOutputs outputs, out bool owned);
            if (graph == null) return false;
            // A replayed graph's buffers are rewritten by its next run (the model-owned buffer
            // contract); a one-shot graph's arrays are the caller's.
            result = new GraphResult
            {
                Logits = outputs.Logits >= 0 ? graph.Outputs[outputs.Logits] : null,
                Final = outputs.Final >= 0 ? graph.Outputs[outputs.Final] : null,
                Captures = outputs.Captures == null ? null : Array.ConvertAll(outputs.Captures, i => graph.Outputs[i]),
            };
            if (owned) graph.Dispose();
            return true;
        }

        /// <summary>
        /// Trace <paramref name="emit"/> and run it: a replayable step first traces in hash-only
        /// mode and replays the cached graph with the same signature (only this step's inputs are
        /// recorded); otherwise the trace is recorded and built (and cached when replayable).
        /// Returns the graph whose outputs hold the results (the caller disposes it when
        /// <paramref name="owned"/>), or null when the lowering met a case it does not express
        /// (nothing was run).
        /// </summary>
        private CachedGraph RunTraced(int rows, Slot[] slots, int[] first, int[] counts, int[] starts, bool replayable,
            bool snapshots, int attentionFloor, Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned)
        {
            GraphStep step = null;
            CachedGraph graph = null;
            owned = false;
            outputs = default;
            try
            {
                if (replayable && _graphCache.Count > 0)
                {
                    step = new GraphStep(this, rows, slots, first, counts, starts, replayable, snapshots, record: false) { AttentionFloor = attentionFloor };
                    outputs = emit(step);
                    if (_graphCache.TryGetValue(step.Signature, out graph))
                    {
                        _graphLru.Remove(step.Signature);
                        _graphLru.AddFirst(step.Signature);
                        GraphReplays++;
                    }
                    else
                    {
                        step.Dispose();
                        step = null;
                    }
                }
                if (graph == null)
                {
                    step = new GraphStep(this, rows, slots, first, counts, starts, replayable, snapshots, record: true) { AttentionFloor = attentionFloor };
                    outputs = emit(step);
                    graph = step.Build(replayable, reorder: replayable && GraphReorderOnMetal);
                    if (replayable)
                    {
                        var key = step.Signature;
                        _graphCache[key] = graph;
                        _graphLru.AddFirst(key);
                        while (_graphLru.Count > GraphCacheEntries)
                        {
                            var evict = _graphLru.Last.Value;
                            _graphLru.RemoveLast();
                            _graphCache.Remove(evict, out CachedGraph old);
                            old?.Dispose();
                        }
                    }
                    else
                    {
                        owned = true;
                    }
                }

                if (!step.Run(graph.Graph))
                    throw new InvalidOperationException($"{Config.Architecture} graph step failed: {GgmlBasicOps.LastNativeError()}");
                GraphRuns++;
                foreach (Slot slot in slots) slot.DeviceDirty = true;
                return graph;
            }
            catch (InvalidOperationException) when (step != null && step.Rejected)
            {
                // The lowering met a case it does not express; nothing was built or run.
                if (owned) graph?.Dispose();
                owned = false;
                ClearGraphCache();
                return null;
            }
            catch
            {
                if (owned) graph?.Dispose();
                throw;
            }
            finally
            {
                step?.Dispose();
            }
        }

        /// <summary>
        /// One layer's recurrent mixer as its own graph, for the per-op path where the family's
        /// lowering is the implementation for these weights: <paramref name="raw"/> is the layer input
        /// [rows, hidden] (its norm runs in the graph), <paramref name="state"/> the sequence's state,
        /// host-current before and after (the per-op path's invariant). Returns [rows, hidden].
        /// </summary>
        protected Tensor RunRecurrentMixerGraph(RankScope scope, Tensor raw, int layer, RecurrentState state)
        {
            int rows = (int)raw.Sizes[0], hiddenSize = Config.HiddenSize;
            var slot = new Slot { Recurrent = new RecurrentState[TotalLayers][] };
            slot.Recurrent[layer] = new[] { state };
            CachedGraph graph = RunTraced(rows, new[] { slot }, new[] { 0 }, new[] { rows }, new[] { 0 },
                replayable: rows <= SmallChunkRows, snapshots: false, attentionFloor: 0,
                step =>
                {
                    int x = step.RmsNorm(step.Input(raw, hiddenSize, rows), LayerName(layer, "attn_norm.weight"));
                    return new StepOutputs { Logits = -1, Final = step.Output(EmitRecurrentMixer(step, layer, x), (long)hiddenSize * rows) };
                },
                out StepOutputs outputs, out bool owned);
            if (graph == null)
                throw new NotSupportedException($"{Config.Architecture}: layer {layer}'s recurrent mixer cannot run as a graph.");
            try
            {
                foreach (Tensor t in state.HostTensors) SyncStateTensor(t);
                var result = new Tensor(scope.Allocator, DType.Float32, rows, hiddenSize);
                result.SetElementsAsFloat(graph.Outputs[outputs.Final]);
                HostWritten(result);
                return result;
            }
            finally
            {
                if (owned) graph.Dispose();
            }
        }

        private struct StepOutputs
        {
            public int Logits, Final;
            public int[] Captures;
        }

        /// <summary>Trace one step: every layer, then what the request asks for.</summary>
        private StepOutputs EmitStep(GraphStep step, Tensor hidden, GraphRequest request)
        {
            int n = step.Sequences, rows = step.Rows, hiddenSize = Config.HiddenSize;
            var outputs = new StepOutputs { Logits = -1, Final = -1 };
            int h = step.Input(hidden, hiddenSize, rows);
            var positions = new int[rows];
            for (int s = 0; s < n; s++)
                for (int i = 0; i < step.Counts[s]; i++) positions[step.First[s] + i] = step.Starts[s] + i + step.Slots[s].RopeDelta;
            step.SetPositions(positions);
            if (request.PositionTable != null)
            {
                if (n != 1) throw new InvalidOperationException("A rotary position table drives one sequence's rows.");
                step.SetPositionTable(AxisMajorPositions(request.PositionTable, rows));
            }

            if (request.CaptureLayers.Length > 0) outputs.Captures = new int[request.CaptureLayers.Length];
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                for (int c = 0; c < request.CaptureLayers.Length; c++)
                    if (request.CaptureLayers[c] == layer)
                        outputs.Captures[c] = step.Output(h, (long)hiddenSize * rows);
                int normed = step.RmsNorm(h, LayerName(layer, "attn_norm.weight"));
                int mix = _recurrentLayer[layer] ? EmitRecurrentMixer(step, layer, normed) : EmitAttention(step, layer, normed);
                h = step.Op(GgmlGraphOp.Add, h, mix);
                int ffnIn = step.RmsNorm(h, FfnNormWeight(layer));
                int ffn = _moeLayer[layer] ? EmitMoe(step, layer, ffnIn) : EmitDenseFfn(step, layer, ffnIn);
                h = step.Op(GgmlGraphOp.Add, h, ffn);
            }
            for (int c = 0; c < request.CaptureLayers.Length; c++)
                if (request.CaptureLayers[c] >= Config.NumLayers)
                    outputs.Captures[c] = step.Output(h, (long)hiddenSize * rows);
            if (!request.LastRowLogits && !request.AllRowLogits && !request.CaptureFinal) return outputs;

            int final = step.RmsNorm(h, "output_norm.weight");
            if (request.CaptureFinal) outputs.Final = step.Output(final, (long)hiddenSize * rows);
            string headName = _quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")
                ? "output.weight" : "token_embd.weight";
            if (request.AllRowLogits)
            {
                outputs.Logits = step.Output(step.MatMul(headName, final), (long)Config.VocabSize * rows);
            }
            else if (request.LastRowLogits)
            {
                int selected;
                if (n == 1 && rows == 1) selected = final;
                else if (n == 1) selected = step.View2(final, hiddenSize, 1, hiddenSize * 4L, (rows - 1) * hiddenSize * 4L);
                else
                {
                    var last = new int[n];
                    for (int s = 0; s < n; s++) last[s] = step.First[s] + step.Counts[s] - 1;
                    selected = step.Op(GgmlGraphOp.GetRows, final, step.InputI32(last, n));
                }
                outputs.Logits = step.Output(step.MatMul(headName, selected), (long)Config.VocabSize * n);
            }
            return outputs;
        }

        private readonly Dictionary<(int Layer, string Suffix), string> _layerNames = new();

        /// <summary><c>blk.{layer}.{suffix}</c>, built once (traces run every token).</summary>
        protected string LayerName(int layer, string suffix)
        {
            if (!_layerNames.TryGetValue((layer, suffix), out string name))
                _layerNames[(layer, suffix)] = name = $"blk.{layer}.{suffix}";
            return name;
        }

        // ---------------------------------------------------------------- branches

        private int EmitAttention(GraphStep step, int layer, int x)
        {
            int heads = _localHeads, kvHeads = _localKvHeads, d = _keyDim, dv = _valueDim, rows = step.Rows;
            const long f = sizeof(float);
            long qDim = (long)heads * d, kDim = (long)kvHeads * d, vDim = (long)kvHeads * dv;
            long qWidth = GatedAttentionOutput ? 2 * qDim : qDim;
            long qStride = GatedAttentionOutput ? 2L * d : d;

            int q, k, v, gate = -1;
            if (_qkvFused[layer])
            {
                int qkv = step.MatMul(LayerName(layer, "attn_qkv.weight"), x);
                long width = qWidth + kDim + vDim;
                q = step.View3(qkv, d, heads, rows, qStride * f, width * f, 0);
                if (GatedAttentionOutput) gate = step.View3(qkv, d, heads, rows, qStride * f, width * f, d * f);
                k = step.View3(qkv, d, kvHeads, rows, d * f, width * f, qWidth * f);
                v = step.View3(qkv, dv, kvHeads, rows, dv * f, width * f, (qWidth + kDim) * f);
            }
            else
            {
                int qa = step.MatMul(LayerName(layer, "attn_q.weight"), x);
                q = step.View3(qa, d, heads, rows, qStride * f, qWidth * f, 0);
                if (GatedAttentionOutput) gate = step.View3(qa, d, heads, rows, qStride * f, qWidth * f, d * f);
                k = step.Reshape(step.MatMul(LayerName(layer, "attn_k.weight"), x), d, kvHeads, rows);
                v = step.Reshape(step.MatMul(LayerName(layer, "attn_v.weight"), x), dv, kvHeads, rows);
            }
            q = step.Cont(q);
            k = step.Cont(k);
            v = step.Cont(v);
            EmitQueryKey(step, layer, ref q, ref k);

            int kvType = KvGgmlType(_kvCacheDtype.ToDType());
            float scale = 1.0f / MathF.Sqrt(d);
            long kRow = RowBytes(kvType, d), vRow = RowBytes(kvType, dv);
            int n = step.Sequences;
            int attn = -1;
            for (int s = 0; s < n; s++)
            {
                int count = step.Counts[s], start = step.Starts[s];
                Slot slot = step.Slots[s];
                long firstRow = step.First[s];
                int qs = n == 1 ? q : step.View3(q, d, heads, count, d * f, d * heads * f, firstRow * d * heads * f);
                int ks = n == 1 ? k : step.View3(k, d, kvHeads, count, d * f, d * kvHeads * f, firstRow * d * kvHeads * f);
                int vs = n == 1 ? v : step.View3(v, dv, kvHeads, count, dv * f, dv * kvHeads * f, firstRow * dv * kvHeads * f);
                int qPerm = step.Permute(qs, 0, 2, 1, 3);
                // [d, heads, count] -> [d, count, heads]; one row is the same bytes either way.
                int kFresh = count == 1 ? step.Reshape(ks, d, 1, kvHeads) : step.Cont(step.Permute(ks, 0, 2, 1, 3));
                int vFresh = count == 1 ? step.Reshape(vs, dv, 1, kvHeads) : step.Cont(step.Permute(vs, 0, 2, 1, 3));

                int cap = slot.Capacity;
                int ck = step.State(slot.K[layer][0], kvType, d, cap, kvHeads);
                int cv = step.State(slot.V[layer][0], kvType, dv, cap, kvHeads);

                int fa;
                if (count <= SmallChunkRows)
                {
                    // Decode and short chunks (a verify window): rows are written by index and
                    // attention reads the padded cache window under a causal mask, so every row is
                    // computed as a decode row is and a window bucket replays the same graph.
                    int rowsAt = step.WriteRows(start, count);
                    step.Effect(step.Op(GgmlGraphOp.SetRows, ck, kFresh, rowsAt));
                    step.Effect(step.Op(GgmlGraphOp.SetRows, cv, vFresh, rowsAt));
                    int stepRows = DecodeWindowStep;
                    int window = Math.Min(cap, (start + count + stepRows - 1) / stepRows * stepRows);
                    int gqa = heads / kvHeads;
                    int kWin = step.Op(GgmlGraphOp.KvWindow, stackalloc[] { ck }, stackalloc long[] { d, cap, kvHeads, 0, window, kvType, count, gqa });
                    int vWin = step.Op(GgmlGraphOp.KvWindow, stackalloc[] { cv }, stackalloc long[] { dv, cap, kvHeads, 0, window, kvType, count, gqa });
                    fa = step.FlashAttention(qPerm, kWin, vWin, step.ChunkMask(window, start, count), scale);
                }
                else
                {
                    // A prompt chunk is written at its offset and attends to the cached prefix and
                    // its own fresh rows (F32), as the fused prefill graphs do.
                    step.Effect(step.Op(GgmlGraphOp.Cpy, kFresh, step.View3(ck, d, count, kvHeads, kRow, kRow * cap, start * kRow)));
                    step.Effect(step.Op(GgmlGraphOp.Cpy, vFresh, step.View3(cv, dv, count, kvHeads, vRow, vRow * cap, start * vRow)));
                    int kAttn = kFresh, vAttn = vFresh;
                    if (start > 0)
                    {
                        int kPrev = step.View3(ck, d, start, kvHeads, kRow, kRow * cap, 0);
                        int vPrev = step.View3(cv, dv, start, kvHeads, vRow, vRow * cap, 0);
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
                int g = step.Reshape(step.Cont(gate), (int)qDim, rows);
                attn = step.Op(GgmlGraphOp.Mul, attn, step.Op(GgmlGraphOp.Sigmoid, g));
            }
            return step.MatMul(LayerName(layer, "attn_output.weight"), attn);
        }

        private int EmitDenseFfn(GraphStep step, int layer, int x)
        {
            int act = _gateUpFused[layer]
                ? step.Op(GgmlGraphOp.Swiglu, step.MatMul(LayerName(layer, "ffn_gate_up.weight"), x))
                : step.Op(GgmlGraphOp.SwigluSplit, step.MatMul(LayerName(layer, "ffn_gate.weight"), x), step.MatMul(LayerName(layer, "ffn_up.weight"), x));
            return step.MatMul(LayerName(layer, "ffn_down.weight"), act);
        }

        private int EmitMoe(GraphStep step, int layer, int x)
        {
            string p = LayerName(layer, "");
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
            long rowBytes = hidden * 4L;
            int output = -1;
            for (int j = 0; j < k; j++)
            {
                int slice = step.View2(down, hidden, rows, rowBytes * k, j * rowBytes);
                output = output < 0 ? slice : step.Op(GgmlGraphOp.Add, output, slice);
            }
            output = step.Cont(output);

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

        /// <summary>Rows a decode attention window grows by: 64 on Metal, whose padded tail costs
        /// more than a replay rebuild every 64 tokens (measured with Qwen 3.5's fused decode); 256
        /// (the flash-attention KV stride) elsewhere.</summary>
        private int DecodeWindowStep => _backend == BackendType.GgmlMetal ? 64 : 256;

        /// <summary>Bytes of one row of <paramref name="ne0"/> elements of a ggml type.</summary>
        private static long RowBytes(int type, long ne0) => type switch
        {
            GgmlType.F32 => ne0 * 4,
            GgmlType.F16 => ne0 * 2,
            GgmlType.Q8_0 => ne0 / 32 * 34,
            GgmlType.Q4_0 => ne0 / 32 * 18,
            _ => throw new ArgumentException($"no row size for ggml type {type}"),
        };

        /// <summary>
        /// One step's graph as a trace: branch code emits nodes here (no native calls), the trace
        /// carries a signature over everything that shapes the graph - ops, parameters, extents and
        /// the bound host pointers - and is built natively only when no cached graph has that
        /// signature. Per-run inputs (activations, positions, masks, write indices) are not part
        /// of the signature; their values ride the run.
        /// </summary>
        protected sealed class GraphStep : IDisposable
        {
            private enum Kind : byte { Leaf, Op, Effect, Output, Alias }

            private readonly struct Rec
            {
                public Kind Kind { get; init; }
                public int Op { get; init; }
                public int[] Srcs { get; init; }
                public long[] Ip { get; init; }
                public float[] Fp { get; init; }
                public GgmlLeafRole Role { get; init; }
                public int Type { get; init; }
                public long[] Ne { get; init; }
                public IntPtr Data { get; init; }
                public long Bytes { get; init; }
                public int Node { get; init; }
            }

            private readonly StandardDecoderModel _m;
            private readonly List<Rec> _recs = new();
            private readonly Dictionary<string, int> _weights = new(StringComparer.Ordinal);
            private readonly Dictionary<Tensor, int> _states = new(ReferenceEqualityComparer.Instance);
            private readonly List<GCHandle> _pins = new();
            private readonly List<int> _inputs = new();
            private readonly List<IntPtr> _inputData = new();
            private readonly List<long> _inputBytes = new();
            private int _nodes;
            private ulong _h1 = 14695981039346656037UL, _h2 = 0x9E3779B97F4A7C15UL;
            private int _mask = -1, _maskStart = -1, _maskCount = -1;
            private int _decodeMask = -1, _decodeMaskWindow = -1, _decodeMaskValid = -1;   // valid: start * 64 + count
            private readonly Dictionary<int, int> _writeIndex = new();
            private readonly HashSet<int> _strided = new();   // nodes whose layout is not contiguous

            private readonly bool _record;

            internal GraphStep(StandardDecoderModel model, int rows, Slot[] slots, int[] first, int[] counts, int[] starts,
                bool decode, bool snapshots, bool record)
            {
                RecurrentSnapshots = snapshots;
                _m = model;
                _record = record;
                Rows = rows;
                Slots = slots;
                First = first;
                Counts = counts;
                Starts = starts;
                IsDecode = decode;
            }

            internal Slot[] Slots { get; }
            public int Rows { get; }
            public int Sequences => Slots.Length;
            public int[] First { get; }
            public int[] Counts { get; }
            public int[] Starts { get; }
            /// <summary>Every sequence forwards a short chunk (the graph may be replayed).</summary>
            public bool IsDecode { get; }
            /// <summary>Keep the recurrent state after every row (a verify window).</summary>
            public bool RecurrentSnapshots { get; }
            /// <summary>Cache rows below this are masked out (a draft layer that resumed after a gap).</summary>
            public int AttentionFloor { get; init; }
            /// <summary>Token positions [rows] (I32).</summary>
            public int Positions { get; private set; } = -1;
            /// <summary>The positions' values (row order).</summary>
            public int[] PositionValues { get; private set; }
            internal bool Rejected { get; private set; }
            internal (ulong, ulong) Signature => (_h1, _h2);

            // ---- tracing

            private void Mix(long value)
            {
                ulong v = unchecked((ulong)value);
                _h1 = (_h1 ^ v) * 1099511628211UL;
                _h2 = (_h2 ^ (v + 0x9E3779B97F4A7C15UL + (_h2 << 6) + (_h2 >> 2))) * 0xC2B2AE3D27D4EB4FUL;
            }

            private void Mix(ReadOnlySpan<long> values) { Mix(values.Length); foreach (long v in values) Mix(v); }

            private void Mix(ReadOnlySpan<int> values) { Mix(values.Length); foreach (int v in values) Mix(v); }

            private void Mix(ReadOnlySpan<float> values) { Mix(values.Length); foreach (float v in values) Mix(BitConverter.SingleToInt32Bits(v)); }

            private int Leaf(GgmlLeafRole role, int type, ReadOnlySpan<long> ne, IntPtr data, long bytes)
            {
                Mix(1); Mix((int)role); Mix(type); Mix(ne);
                if (role != GgmlLeafRole.Input) { Mix(data.ToInt64()); Mix(bytes); }
                if (_record) _recs.Add(new Rec { Kind = Kind.Leaf, Role = role, Type = type, Ne = ne.ToArray(), Data = data, Bytes = bytes });
                return _nodes++;
            }

            public int Op(GgmlGraphOp op, int a) => Op(op, stackalloc[] { a });
            public int Op(GgmlGraphOp op, int a, int b) => Op(op, stackalloc[] { a, b });
            public int Op(GgmlGraphOp op, int a, int b, int c) => Op(op, stackalloc[] { a, b, c });

            public int Op(GgmlGraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
            {
                foreach (int src in srcs)
                    if (src < 0 || src >= _nodes) throw new ArgumentOutOfRangeException(nameof(srcs), $"{op}: source node {src} does not exist");
                Mix(2); Mix((int)op); Mix(srcs); Mix(ip); Mix(fp);
                if (_record) _recs.Add(new Rec { Kind = Kind.Op, Op = (int)op, Srcs = srcs.ToArray(), Ip = ip.ToArray(), Fp = fp.ToArray() });
                return _nodes++;
            }

            /// <summary>Run <paramref name="node"/> for its side effect (a cache or state write).</summary>
            public void Effect(int node)
            {
                Mix(3); Mix(node);
                if (_record) _recs.Add(new Rec { Kind = Kind.Effect, Node = node });
            }

            /// <summary>Write <paramref name="node"/>'s result in place inside the state leaf
            /// <paramref name="baseLeaf"/> at <paramref name="offset"/> bytes; where the base is not
            /// resident, the <paramref name="fallback"/> copy runs instead (see GgmlGraph.Alias).</summary>
            public void Alias(int node, int baseLeaf, long offset, int fallback)
            {
                Mix(5); Mix(node); Mix(baseLeaf); Mix(offset); Mix(fallback);
                if (_record) _recs.Add(new Rec { Kind = Kind.Alias, Node = node, Srcs = new[] { baseLeaf, fallback }, Bytes = offset });
            }

            private int _outputs;

            /// <summary>A step result: <paramref name="floats"/> F32 values of <paramref name="node"/>;
            /// returns its index among the step's results.</summary>
            internal int Output(int node, long floats)
            {
                Mix(4); Mix(node); Mix(floats);
                if (_record) _recs.Add(new Rec { Kind = Kind.Output, Node = node, Bytes = floats });
                return _outputs++;
            }

            // ---- leaves

            /// <summary>A model weight by name: quantized (its device-cache key) or F32.</summary>
            public int Weight(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (_m._quantWeights.TryGetValue(name, out QuantizedWeight q))
                {
                    IntPtr key = q.CacheKey != IntPtr.Zero ? q.CacheKey : q.Data;
                    node = Leaf(GgmlLeafRole.Weight, q.GgmlType, stackalloc long[] { q.Ne0, q.Ne1 }, key, q.RawBytes);
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
                node = Leaf(GgmlLeafRole.Weight, w.GgmlType, stackalloc long[] { w.PerExpertNe0, w.PerExpertNe1, w.NumExperts }, w.Data, w.TotalRawBytes);
                _weights[name] = node;
                return node;
            }

            /// <summary>An F32 tensor the family prepared (device constants, norms) as a read-only leaf.</summary>
            public int TensorWeight(Tensor t, params long[] ne) => TensorLeaf(GgmlLeafRole.Weight, t, GgmlType.F32, ne.Length == 0 ? Extents(t) : ne);

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
                => Leaf(role, type, ne, TensorComputePrimitives.GetStoragePointer(t), t.Storage.ByteLength);

            private static long[] Extents(Tensor t)
            {
                var ne = new long[t.DimensionCount];
                for (int i = 0; i < ne.Length; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
                return ne;
            }

            // ---- per-run inputs

            /// <summary>An F32 activation [cols, rows] read from a host tensor every run.</summary>
            internal int Input(Tensor t, long cols, long rows)
            {
                int node = Leaf(GgmlLeafRole.Input, GgmlType.F32, stackalloc long[] { cols, rows }, IntPtr.Zero, 0);
                AddInput(node, TensorComputePrimitives.GetStoragePointer(t), cols * rows * sizeof(float));
                return node;
            }

            public int InputI32(int[] values, long count)
            {
                int node = Leaf(GgmlLeafRole.Input, GgmlType.I32, stackalloc long[] { count }, IntPtr.Zero, 0);
                AddInput(node, Pin(values), count * sizeof(int));
                return node;
            }

            public int InputF32(float[] values, params long[] ne)
            {
                int node = Leaf(GgmlLeafRole.Input, GgmlType.F32, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), (long)values.Length * sizeof(float));
                return node;
            }

            internal void SetPositions(int[] positions)
            {
                PositionValues = positions;
                Positions = InputI32(positions, positions.Length);
            }

            /// <summary>Axis-major [4 * rows] rotary positions (ggml_rope_multi); rotary ops then take them.</summary>
            internal void SetPositionTable(int[] axisMajor) => PositionTable = InputI32(axisMajor, axisMajor.Length);

            /// <summary>Multi-axis rotary positions [4 * rows] (I32), or -1.</summary>
            public int PositionTable { get; private set; } = -1;

            /// <summary>The rows [count, 1] (I64) a short chunk is written at: start .. start + count - 1.</summary>
            internal int WriteRows(int start, int count)
            {
                if (_writeIndex.TryGetValue(start * 64 + count, out int node)) return node;
                node = Leaf(GgmlLeafRole.Input, GgmlType.I64, stackalloc long[] { count, 1 }, IntPtr.Zero, 0);
                var rows = new long[count];
                for (int i = 0; i < count; i++) rows[i] = start + i;
                AddInput(node, Pin(rows), count * sizeof(long));
                _writeIndex[start * 64 + count] = node;
                return node;
            }

            /// <summary>An F16 causal mask [window, count]: row q sees cache rows 0 .. start + q.</summary>
            internal int ChunkMask(int window, int start, int count)
            {
                if (_decodeMask >= 0 && _decodeMaskWindow == window && _decodeMaskValid == start * 64 + count) return _decodeMask;
                var mask = new ushort[(long)window * count];
                for (int q = 0; q < count; q++)
                {
                    for (int k = 0; k < Math.Min(AttentionFloor, window); k++) mask[(long)q * window + k] = 0xFC00;   // -inf
                    for (int k = start + q + 1; k < window; k++) mask[(long)q * window + k] = 0xFC00;
                }
                _decodeMask = Leaf(GgmlLeafRole.Input, GgmlType.F16, stackalloc long[] { window, count }, IntPtr.Zero, 0);
                AddInput(_decodeMask, Pin(mask), mask.LongLength * sizeof(ushort));
                _decodeMaskWindow = window;
                _decodeMaskValid = start * 64 + count;
                return _decodeMask;
            }

            /// <summary>A causal F16 mask [start + count, count] for a chunk continuing at <paramref name="start"/>.</summary>
            public int CausalMask(int start, int count)
            {
                if (_mask >= 0 && _maskStart == start && _maskCount == count) return _mask;
                int kv = start + count;
                var mask = new ushort[(long)kv * count];
                for (int qi = 0; qi < count; qi++)
                {
                    for (int ki = 0; ki < Math.Min(AttentionFloor, kv); ki++) mask[(long)qi * kv + ki] = 0xFC00;
                    for (int ki = start + qi + 1; ki < kv; ki++) mask[(long)qi * kv + ki] = 0xFC00;
                }
                _mask = Leaf(GgmlLeafRole.Input, GgmlType.F16, stackalloc long[] { kv, count }, IntPtr.Zero, 0);
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

            // ---- composites

            /// <summary>rms_norm(x) * weight.</summary>
            public int RmsNorm(int x, string weight) => RmsNorm(x, Weight(weight), _m.Config.Eps);

            public int RmsNorm(int x, int weight, float eps)
                => Op(GgmlGraphOp.Mul, Op(GgmlGraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), weight);

            /// <summary>rms_norm(x, eps / n) / sqrt(n) over ne0: the L2 normalization ggml's GDN path uses.</summary>
            public int L2Norm(int x, long n, float eps)
                => Op(GgmlGraphOp.Scale, stackalloc[] { Op(GgmlGraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps / n }) },
                    default, stackalloc[] { 1.0f / MathF.Sqrt(n) });

            /// <summary>weight · x, with a quantized weight's per-tensor scale applied.</summary>
            public int MatMul(string weight, int x)
            {
                int y = Op(GgmlGraphOp.MulMat, Weight(weight), x);
                if (_m._quantWeights.TryGetValue(weight, out QuantizedWeight q) && q.Scale != 1.0f)
                    y = Op(GgmlGraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { q.Scale });
                return y;
            }

            /// <summary>weight[firstRow .. firstRow + rows) · x: a row block of a (packed) weight as its
            /// own product, so the result is contiguous.</summary>
            public int MatMulRows(string weight, long firstRow, long rows, int x)
            {
                int w = Weight(weight);
                long ne0, rowBytes;
                float scale = 1.0f;
                if (_m._quantWeights.TryGetValue(weight, out QuantizedWeight q))
                {
                    ne0 = q.Ne0;
                    rowBytes = q.RawBytes / q.Ne1;
                    scale = q.Scale;
                }
                else
                {
                    Tensor t = _m._weights[weight];
                    ne0 = t.Sizes[t.DimensionCount - 1];
                    rowBytes = ne0 * sizeof(float);
                }
                int block = Op(GgmlGraphOp.View, stackalloc[] { w }, stackalloc long[] { 2, ne0, rows, 1, 1, rowBytes, 0, 0, firstRow * rowBytes });
                int y = Op(GgmlGraphOp.MulMat, block, x);
                if (scale != 1.0f)
                    y = Op(GgmlGraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { scale });
                return y;
            }

            public int Reshape(int x, params long[] ne) => Op(GgmlGraphOp.Reshape, stackalloc[] { x }, ne);

            /// <summary>A view of an F32 node; recorded as strided unless its rows are packed.</summary>
            public int View2(int x, long ne0, long ne1, long nb1, long offset)
            {
                int node = Op(GgmlGraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });
                if (ne1 > 1 && nb1 != ne0 * 4) _strided.Add(node);
                return node;
            }

            public int View4(int x, long ne0, long ne1, long ne2, long ne3, long nb1, long nb2, long nb3, long offset)
            {
                int node = Op(GgmlGraphOp.View, stackalloc[] { x }, stackalloc long[] { 4, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset });
                if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4) || (ne3 > 1 && nb3 != ne0 * ne1 * ne2 * 4))
                    _strided.Add(node);
                return node;
            }

            /// <summary>The backend's norms, unary ops and recurrent kernels read row-contiguous strided
            /// views (ggml-metal, ggml-cuda); elsewhere a strided input is copied contiguous first.</summary>
            public bool StridedInputs => _m._backend is BackendType.GgmlMetal or BackendType.GgmlCuda;

            public int View3(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
            {
                int node = Op(GgmlGraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });
                if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4)) _strided.Add(node);
                return node;
            }

            /// <summary>Permute axes (strided).</summary>
            public int Permute(int x, long a0, long a1, long a2, long a3)
            {
                int node = Op(GgmlGraphOp.Permute, stackalloc[] { x }, stackalloc long[] { a0, a1, a2, a3 });
                _strided.Add(node);
                return node;
            }

            public int Transpose(int x)
            {
                int node = Op(GgmlGraphOp.Transpose, x);
                _strided.Add(node);
                return node;
            }

            /// <summary>A contiguous copy of <paramref name="x"/>, or <paramref name="x"/> itself when it
            /// already is contiguous (every op result is; views, permutes and transposes may not be).</summary>
            public int Cont(int x) => _strided.Contains(x) ? Op(GgmlGraphOp.Cont, x) : x;

            /// <summary>Rotary at the step's positions; with a position table the family's multi-axis
            /// rotary (ggml's interleaved M-RoPE over its sections), which equals the scalar one for text.</summary>
            public int Rope(int x, RotarySpec spec)
            {
                if (PositionTable >= 0 && _m.RotarySections is { } sections)
                    return Op(GgmlGraphOp.RopeMulti, stackalloc[] { x, PositionTable },
                        stackalloc long[] { spec.Dims, 40 /* GGML_ROPE_TYPE_IMROPE */, spec.OriginalContext,
                            sections[0], sections[1], sections[2], sections.Length > 3 ? sections[3] : 0 },
                        stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });
                return Op(GgmlGraphOp.RopeExt, stackalloc[] { x, Positions },
                    stackalloc long[] { spec.Dims, spec.NeoX ? 2 : 0, spec.OriginalContext },
                    stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });
            }

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

            // ---- native

            /// <summary>Build the trace as a native graph (node handles equal trace indices).</summary>
            internal CachedGraph Build(bool replayable, bool reorder)
            {
                if (!_record) throw new InvalidOperationException("a hash-only trace cannot be built");
                var g = new GgmlGraph(Math.Max(1024, _nodes + 64), replayable, reorder);
                var outputs = new List<float[]>();
                var pins = new List<GCHandle>();
                try
                {
                    foreach (Rec r in _recs)
                    {
                        switch (r.Kind)
                        {
                            case Kind.Leaf: g.Leaf(r.Role, r.Type, r.Ne, r.Data, r.Bytes); break;
                            case Kind.Op: g.Op((GgmlGraphOp)r.Op, r.Srcs, r.Ip, r.Fp); break;
                            case Kind.Effect: g.Effect(r.Node); break;
                            case Kind.Alias: g.Alias(r.Node, r.Srcs[0], r.Bytes, r.Srcs[1]); break;
                            case Kind.Output:
                                var values = new float[r.Bytes];
                                GCHandle pin = GCHandle.Alloc(values, GCHandleType.Pinned);
                                outputs.Add(values);
                                pins.Add(pin);
                                g.Output(r.Node, pin.AddrOfPinnedObject(), r.Bytes * sizeof(float));
                                break;
                        }
                    }
                    return new CachedGraph { Graph = g, Outputs = outputs.ToArray(), Pins = pins.ToArray() };
                }
                catch
                {
                    g.Dispose();
                    foreach (GCHandle pin in pins) pin.Free();
                    throw;
                }
            }

            internal bool Run(GgmlGraph graph) => graph.Run(
                CollectionsMarshal.AsSpan(_inputs), CollectionsMarshal.AsSpan(_inputData), CollectionsMarshal.AsSpan(_inputBytes));

            public void Dispose()
            {
                foreach (GCHandle pin in _pins) pin.Free();
                _pins.Clear();
            }
        }
    }
}
