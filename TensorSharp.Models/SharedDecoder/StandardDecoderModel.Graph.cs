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

using TensorSharp.Graph;

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
        protected virtual bool GraphConcurrencyReorder => false;

        /// <summary>A reason the current configuration cannot run whole-forward graphs, or null.</summary>
        private string GraphUnsupportedReason()
        {
            if (!GraphLoweringEnabled) return "disabled";
            if (GraphExecutor == null) return "this backend registers no graph executor";
            if (_mesh.TensorParallel > 1) return "tensor-parallel placement";
            if (LayerSplitGroup != null) return "multi-node pipeline";
            if (GraphRotary == null) return "the family has no graph rotary";
            if (_anyRecurrent && !RecurrentMixerLowers) return "recurrent layers without a graph mixer";
            if (StepAdapter != null) return "a request adapter is selected";
            for (int l = 0; l < Config.NumLayers; l++)
                if (_moeLayer[l] && IsExpertLayerOffloaded(l) && !GraphExecutor.Capabilities.HostExperts) return "host-offloaded experts";
            DType kv = _kvCacheDtype.ToDType();
            if (kv != DType.Float32 && kv != DType.Float16 && kv != DType.Q8_0 && kv != DType.Q4_0) return $"KV dtype {kv}";
            if ((kv == DType.Q8_0 || kv == DType.Q4_0) && !GraphExecutor.Capabilities.QuantizedKvCache) return $"KV dtype {kv} on this backend's executor";
            return null;
        }

        private readonly Dictionary<string, GraphExpertSet> _graphExpertSets = new();

        /// <summary>Step kinds (shape, emitter) that ran as recorded programs: a streaming executor runs
        /// later steps of the kind as they are emitted.</summary>
        private readonly HashSet<string> _streamedKinds = new();

        /// <summary>The recorded program of each streamed kind: its readers per node let a stream release
        /// tensors after their last use.</summary>
        private readonly Dictionary<string, GraphProgram> _streamTemplates = new();

        /// <summary>A stacked expert weight with its members (each a quantized weight with its own
        /// resident copy), for executors that run routed experts one at a time.</summary>
        private GraphExpertSet ExpertSet(string name, StackedExpertWeights stacked)
        {
            lock (_graphExpertSets)
            {
                if (_graphExpertSets.TryGetValue(name, out GraphExpertSet set)) return set;
                string baseName = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^7] : name;
                var members = new QuantizedWeight[stacked.NumExperts];
                for (int e = 0; e < members.Length; e++)
                    _quantWeights.TryGetValue($"{baseName}.{e}.weight", out members[e]);
                return _graphExpertSets[name] = new GraphExpertSet(stacked, members);
            }
        }

        private sealed class GraphExpertSet : IGraphExpertWeights
        {
            private readonly StackedExpertWeights _w;
            private readonly QuantizedWeight[] _members;

            public GraphExpertSet(StackedExpertWeights w, QuantizedWeight[] members) { _w = w; _members = members; }

            public int Format => _w.GgmlType;
            public long Ne0 => _w.PerExpertNe0;
            public long Ne1 => _w.PerExpertNe1;
            public long Ne2 => _w.NumExperts;
            public IntPtr Data => _w.Data;
            public IntPtr CacheKey => _w.Data;
            public long Bytes => _w.TotalRawBytes;

            public IGraphWeight Expert(int index) => _members[index]
                ?? throw new NotSupportedException($"expert {index} has no member weight");
        }

        private protected static int KvGgmlType(DType dt) => dt switch
        {
            DType.Float32 => GraphFormat.F32,
            DType.Float16 => GraphFormat.F16,
            DType.Q4_0 => GraphFormat.Q4_0,
            DType.Q8_0 => GraphFormat.Q8_0,
            _ => throw new ArgumentException($"Unsupported KV cache dtype: {dt}"),
        };

        // Decode graphs (every sequence one row) are replayed while their trace matches.
        private const int GraphCacheEntries = 32;   // decode + verify widths x window buckets
        private readonly Dictionary<(ulong, ulong), CachedGraph> _graphCache = new();
        private readonly LinkedList<(ulong, ulong)> _graphLru = new();

        /// <summary>A compiled program kept for replay, with the buffers its results land in.</summary>
        internal sealed class CachedGraph : IDisposable
        {
            public required IGraphPlan Plan { get; init; }
            /// <summary>The step's results, in the order they were declared (GraphStep.Output).</summary>
            public float[][] Outputs => Plan.Outputs;

            public void Dispose() => Plan.Dispose();
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

            /// <summary>What the request changes in the emitted program (a streamed step's template key).</summary>
            public string Shape =>
                $"{LastRowLogits}{AllRowLogits}{CaptureFinal}{PositionTable != null}:{string.Join(".", CaptureLayers)}";
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
        // ---------------------------------------------------------------- decode steps without a trace

        /// <summary>The family's decode steps depend on the rows' positions only through the step's standard inputs
        /// (positions, write rows, decode masks under one window bucket), and its emission has no side effects: a
        /// decode step may then run as the previous one advanced by a position (<see cref="TryAdvanceDecode"/>).</summary>
        protected virtual bool DecodeStepsAdvance => false;

        /// <summary>Trace every step (tests compare against it).</summary>
        internal bool DecodeAdvanceDisabled { get; set; }

        /// <summary>What else, beyond the rows, counts and outputs, decides what a step emits: a family whose graph
        /// depends on where its sequences are (QSA selects past its budget) names it, so a streamed step never frees
        /// nodes by the last readers of a template emitted the other way.</summary>
        protected virtual string EmissionVariant(int[] counts, int[] starts) => null;

        /// <summary>Decode steps run as the previous one advanced, without a trace.</summary>
        internal int DecodeAdvances { get; private set; }

        /// <summary>The last decode step, kept to run again one position on.</summary>
        private sealed class DecodeReplay
        {
            public GraphStep Step;
            public CachedGraph Graph;
            public (ulong, ulong) Signature;
            public StepOutputs Outputs;
            public Slot[] Slots;
            public Tensor[][][] K;
            public RecurrentState[][][] Recurrent;
            public int[] Capacity, RopeDelta, Starts;
            public int Floor;
            public string Shape;
            public long Epoch;
        }

        private DecodeReplay _decodeReplay;
        private long _graphEpoch;   // bumped whenever cached graphs are dropped

        private void KeepDecodeReplay(GraphStep step, CachedGraph graph, StepOutputs outputs, Slot[] slots, int[] starts,
            int floor, string shape)
        {
            DropDecodeReplay();
            int n = slots.Length;
            var r = new DecodeReplay
            {
                Step = step, Graph = graph, Signature = step.Signature, Outputs = outputs, Slots = (Slot[])slots.Clone(),
                K = new Tensor[n][][], Recurrent = new RecurrentState[n][][], Capacity = new int[n], RopeDelta = new int[n],
                Starts = (int[])starts.Clone(), Floor = floor, Shape = shape, Epoch = _graphEpoch,
            };
            for (int s = 0; s < n; s++)
            {
                r.K[s] = slots[s].K;
                r.Recurrent[s] = slots[s].Recurrent;
                r.Capacity[s] = slots[s].Capacity;
                r.RopeDelta[s] = slots[s].RopeDelta;
            }
            _decodeReplay = r;
        }

        private void DropDecodeReplay()
        {
            _decodeReplay?.Step.Dispose();
            _decodeReplay = null;
        }

        /// <summary>
        /// Run a decode step as the previous one advanced by a position, without tracing it: the same slots (their
        /// caches, states and capacity unchanged), every sequence one position on, the same request shape and
        /// attention floor, no graph dropped since, and every decode mask still inside its window. Its inputs are
        /// rewritten in place (positions, write rows, masks, the embedded rows); the cached graph's signature is the
        /// one the trace would compute. A trace was ~2% of a decode step (Qwen3.6-35B-A3B on an M5 Pro: 0.28 of 15 ms).
        /// </summary>
        private bool TryAdvanceDecode(Slot[] slots, int[] starts, int floor, string shape, Tensor hidden,
            out CachedGraph graph, out StepOutputs outputs)
        {
            graph = null;
            outputs = default;
            DecodeReplay r = _decodeReplay;
            if (r == null || r.Epoch != _graphEpoch || r.Floor != floor || r.Shape != shape || r.Slots.Length != slots.Length)
                return false;
            for (int s = 0; s < slots.Length; s++)
            {
                Slot slot = slots[s];
                if (!ReferenceEquals(slot, r.Slots[s]) || !ReferenceEquals(slot.K, r.K[s]) || !ReferenceEquals(slot.Recurrent, r.Recurrent[s])
                    || slot.Capacity != r.Capacity[s] || slot.RopeDelta != r.RopeDelta[s] || starts[s] != r.Starts[s] + 1)
                    return false;
            }
            if (!_graphCache.TryGetValue(r.Signature, out CachedGraph cached) || !ReferenceEquals(cached, r.Graph)
                || !r.Step.TryAdvance(hidden))
            {
                DropDecodeReplay();
                return false;
            }
            Array.Copy(starts, r.Starts, starts.Length);
            _graphLru.Remove(r.Signature);
            _graphLru.AddFirst(r.Signature);
            if (cached.Plan is IGraphTensorOutputs keep) keep.KeepTensors = false;
            if (!r.Step.Run(cached.Plan))
                throw new InvalidOperationException($"{Config.Architecture} graph step failed: {cached.Plan.LastError}");
            GraphRuns++;
            GraphReplays++;
            DecodeAdvances++;
            foreach (Slot slot in slots) slot.DeviceDirty = true;
            graph = cached;
            outputs = r.Outputs;
            return true;
        }

        private void ClearGraphCache()
        {
            _graphEpoch++;
            DropDecodeReplay();
            OnGraphCacheCleared();
            foreach (CachedGraph entry in _graphCache.Values) entry.Dispose();
            _graphCache.Clear();
            _graphLru.Clear();
            _streamedKinds.Clear();
            _streamTemplates.Clear();
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
            if (reason != null && RequiresGraphs)
                throw new NotSupportedException($"{Config.Architecture} runs only as graphs on the shared runtime, which this step cannot use ({reason}).");
            if (reason != null && _graphTpRanks != null && reason != "disabled")
                throw new NotSupportedException(
                    $"{Config.Architecture}: tensor parallelism runs as whole-forward graphs, which this step cannot use ({reason}).");
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
            if (_mesh.Stages > 1)
                return TryRunStagedGraphStep(hidden, slots, first, counts, starts, request, replayable, out result);
            CachedGraph graph = RunTraced(rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, 0,
                step => EmitStep(step, hidden, request), out StepOutputs outputs, out bool owned, shape: request.Shape,
                advanceHidden: request.PositionTable == null ? hidden : null);
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
        /// A step over whole-layer stages on this process's devices: each stage's layers are one graph
        /// on its device (bound to the stage's caches, states and weights), the rows pass from stage to
        /// stage through the host, and the last stage runs the head. A capture comes from whichever
        /// stage holds its layer.
        /// </summary>
        private bool TryRunStagedGraphStep(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            GraphRequest request, bool replayable, out GraphResult result)
        {
            result = null;
            int rows = (int)hidden.Sizes[0], stages = _mesh.Stages, hiddenSize = Config.HiddenSize;
            float[][] captures = request.CaptureLayers.Length > 0 ? new float[request.CaptureLayers.Length][] : null;
            Tensor input = hidden;
            try
            {
                for (int s = 0; s < stages; s++)
                {
                    EnterStage(s);
                    (int from, int to) = StageLayers(s);
                    bool last = s == stages - 1;
                    Tensor stageInput = input;
                    CachedGraph graph = RunTraced(rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, 0,
                        step => EmitStep(step, stageInput, request, from, to, last), out StepOutputs outputs, out bool owned,
                        shape: $"{request.Shape}|stage{s}");
                    if (graph == null)
                    {
                        if (s == 0) return false;
                        throw new InvalidOperationException($"{Config.Architecture}: stage {s}'s graph declined after stage 0 ran.");
                    }
                    try
                    {
                        if (outputs.Captures != null)
                            for (int c = 0; c < outputs.Captures.Length; c++)
                                if (outputs.Captures[c] >= 0) captures[c] = (float[])graph.Outputs[outputs.Captures[c]].Clone();
                        if (!last)
                        {
                            float[] rowsOut = graph.Outputs[outputs.Hidden];
                            if (!ReferenceEquals(input, hidden)) input.Dispose();
                            EnterStage(s + 1);
                            input = CreateFloatTensor((float[])rowsOut.Clone(), rows, StreamWidth);
                            continue;
                        }
                        result = new GraphResult
                        {
                            Logits = outputs.Logits >= 0 ? graph.Outputs[outputs.Logits] : null,
                            Final = outputs.Final >= 0 ? graph.Outputs[outputs.Final] : null,
                            Captures = captures,
                        };
                    }
                    finally
                    {
                        if (owned) graph.Dispose();
                    }
                }
                return true;
            }
            finally
            {
                if (!ReferenceEquals(input, hidden)) input.Dispose();
                EnterStage(0);
            }
        }

        /// <summary>The decoder layers of pipeline stage <paramref name="stage"/>: [from, to).</summary>
        private (int From, int To) StageLayers(int stage)
        {
            int from = -1, to = -1;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_mesh.StageOf(l) != stage) continue;
                if (from < 0) from = l;
                else if (to != l) throw new InvalidOperationException($"{Config.Architecture}: stage {stage}'s layers are not contiguous.");
                to = l + 1;
            }
            return from < 0 ? (0, 0) : (from, to);
        }

        /// <summary>
        /// Trace <paramref name="emit"/> and run it: a replayable step first traces in hash-only
        /// mode and replays the cached graph with the same signature (only this step's inputs are
        /// recorded); otherwise the trace is recorded and built (and cached when replayable).
        /// Returns the graph whose outputs hold the results (the caller disposes it when
        /// <paramref name="owned"/>), or null when the lowering met a case it does not express
        /// (nothing was run).
        /// </summary>
        /// <param name="advanceHidden">The step's embedded rows, when a decode step may run as the previous one
        /// advanced (<see cref="TryAdvanceDecode"/>) instead of being traced.</param>
        private CachedGraph RunTraced(int rows, Slot[] slots, int[] first, int[] counts, int[] starts, bool replayable,
            bool snapshots, int attentionFloor, Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned,
            bool deviceOutputs = false, string shape = null, Tensor advanceHidden = null)
        {
            GraphStep step = null;
            CachedGraph graph = null;
            owned = false;
            outputs = default;
            bool advance = replayable && advanceHidden != null && !deviceOutputs && !snapshots && DecodeStepsAdvance
                && !DecodeAdvanceDisabled && _mesh.Stages == 1 && GraphExecutor.Capabilities.Replays && Array.TrueForAll(counts, c => c == 1);
            if (advance && TryAdvanceDecode(slots, starts, attentionFloor, shape, advanceHidden, out graph, out outputs))
                return graph;
            try
            {
                // An executor whose plans do not replay records every step once (no signature pass); a
                // step of a kind that already ran as a recorded program streams (its ops run as they are
                // emitted, overlapping the device): it cannot be rejected half way through.
                replayable &= GraphExecutor.Capabilities.Replays;
                string kind = null;
                if (GraphExecutor is IStreamingGraphExecutor streaming)
                {
                    // Everything that changes what the step emits: a stream frees each node after the template's
                    // last reader, so a step that emits differently (other outputs, another stage) must not share it.
                    kind = $"{rows}|{slots.Length}|{snapshots}|{attentionFloor > 0}|{emit.Method.MetadataToken}|{string.Join(",", counts)}|{shape}|{EmissionVariant(counts, starts)}";
                    if (_streamedKinds.Contains(kind))
                    {
                        IGraphStream stream = streaming.OpenStream(_streamTemplates.GetValueOrDefault(kind));
                        if (deviceOutputs && stream is IGraphTensorOutputs keepStream) keepStream.KeepTensors = true;
                        graph = new CachedGraph { Plan = stream };
                        owned = true;
                        step = new GraphStep(this, rows, slots, first, counts, starts, false, snapshots, record: false)
                            { AttentionFloor = attentionFloor, Sink = stream };
                        StepOutputs emitted = default;
                        GraphStep streamed = step;
                        bool ok = stream.Execute(() =>
                        {
                            emitted = emit(streamed);
                            return streamed.Run(stream);
                        });
                        outputs = emitted;
                        if (!ok)
                            throw new InvalidOperationException($"{Config.Architecture} graph step failed: {stream.LastError}");
                        GraphRuns++;
                        foreach (Slot slot in slots) slot.DeviceDirty = true;
                        return graph;
                    }
                }
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
                    graph = step.Build(replayable, reorder: replayable && GraphConcurrencyReorder);
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

                if (graph.Plan is IGraphTensorOutputs keep) keep.KeepTensors = deviceOutputs;
                if (!step.Run(graph.Plan))
                    throw new InvalidOperationException($"{Config.Architecture} graph step failed: {graph.Plan.LastError}");
                GraphRuns++;
                if (kind != null && _streamedKinds.Add(kind)) _streamTemplates[kind] = step.Program;
                foreach (Slot slot in slots) slot.DeviceDirty = true;
                if (advance && !owned && step.Advanceable)
                {
                    KeepDecodeReplay(step, graph, outputs, slots, starts, attentionFloor, shape);
                    step = null;   // the replay owns it now
                }
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
                out StepOutputs outputs, out bool owned, deviceOutputs: true);
            if (graph == null)
                throw new NotSupportedException($"{Config.Architecture}: layer {layer}'s recurrent mixer cannot run as a graph.");
            try
            {
                // A tensor executor hands the result over on the device (no host copy, no wait).
                if (graph.Plan is IGraphTensorOutputs device && device.KeepTensors)
                {
                    using Tensor y = device.TakeOutput(outputs.Final);
                    return y.View(rows, hiddenSize);
                }
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
            public int Logits, Final, Hidden;
            public int[] Captures;   // -1: the captured layer is another stage's
        }

        /// <summary>Trace one step: every layer, then what the request asks for.</summary>
        private StepOutputs EmitStep(GraphStep step, Tensor hidden, GraphRequest request)
            => EmitStep(step, hidden, request, 0, Config.NumLayers, head: true);

        /// <summary>Trace layers [<paramref name="from"/>, <paramref name="to"/>) (one pipeline stage), then
        /// the head when <paramref name="head"/>, else the stage's output rows for the next stage.</summary>
        private StepOutputs EmitStep(GraphStep step, Tensor hidden, GraphRequest request, int from, int to, bool head)
        {
            int n = step.Sequences, rows = step.Rows, hiddenSize = Config.HiddenSize, width = StreamWidth;
            var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            // The first stage widens the embedded rows into the residual stream; later stages receive it.
            int h = from == 0 ? EmitStreamInput(step, step.HiddenInput(hidden, hiddenSize, rows)) : step.HiddenInput(hidden, width, rows);
            var positions = new int[rows];
            for (int s = 0; s < n; s++)
                for (int i = 0; i < step.Counts[s]; i++) positions[step.First[s] + i] = step.Starts[s] + i + step.Slots[s].RopeDelta;
            step.SetPositions(positions);
            if (request.PositionTable != null)
            {
                if (n != 1) throw new InvalidOperationException("A rotary position table drives one sequence's rows.");
                step.SetPositionTable(AxisMajorPositions(request.PositionTable, rows));
            }

            if (request.CaptureLayers.Length > 0)
            {
                outputs.Captures = new int[request.CaptureLayers.Length];
                Array.Fill(outputs.Captures, -1);
            }
            for (int layer = from; layer < to; layer++)
            {
                for (int c = 0; c < request.CaptureLayers.Length; c++)
                    if (request.CaptureLayers[c] == layer)
                        outputs.Captures[c] = step.Output(EmitCaptureRow(step, h), (long)hiddenSize * rows);
                h = EmitLayerPrologue(step, layer, h);
                int normed = EmitBlockInput(step, layer, h, BlockSite.Mixer, out int mixContext);
                int mix = _recurrentLayer[layer] ? EmitRecurrentMixer(step, layer, normed) : EmitAttention(step, layer, normed);
                h = EmitBlockOutput(step, layer, h, BlockSite.Mixer, mixContext, mix);
                int ffnIn = EmitBlockInput(step, layer, h, BlockSite.Ffn, out int ffnContext);
                int ffn = _moeLayer[layer] ? EmitMoe(step, layer, ffnIn) : EmitDenseFfn(step, layer, ffnIn);
                h = EmitBlockOutput(step, layer, h, BlockSite.Ffn, ffnContext, ffn);
                step.Boundary(h);
            }
            if (!head)
            {
                outputs.Hidden = step.Output(h, (long)width * rows);
                return outputs;
            }
            for (int c = 0; c < request.CaptureLayers.Length; c++)
                if (request.CaptureLayers[c] >= Config.NumLayers)
                    outputs.Captures[c] = step.Output(EmitCaptureRow(step, h), (long)hiddenSize * rows);
            if (!request.LastRowLogits && !request.AllRowLogits && !request.CaptureFinal) return outputs;

            int final = EmitStreamOutput(step, h);
            if (request.CaptureFinal)
                outputs.Final = DraftChainsStream ? step.Output(h, (long)width * rows) : step.Output(final, (long)hiddenSize * rows);
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
                    selected = step.Op(GraphOp.GetRows, final, step.ConstantInputI32(last, n));
                }
                outputs.Logits = step.Output(step.MatMul(headName, selected), (long)Config.VocabSize * n);
            }
            return outputs;
        }

        // ---------------------------------------------------------------- the residual stream

        /// <summary>Where a block reads the residual stream.</summary>
        protected enum BlockSite { Mixer, Ffn }

        /// <summary>A family whose blocks the per-op path does not implement (hyper-connections): a step that
        /// cannot run as a graph fails instead of falling back.</summary>
        protected virtual bool RequiresGraphs => false;

        /// <summary>What a layer adds to the residual stream before its blocks (Flash Next's PLE); default none.</summary>
        protected virtual int EmitLayerPrologue(GraphStep step, int layer, int stream) => stream;

        /// <summary>The width of the residual stream between layers and stages (several streams of the
        /// hidden size where a family keeps hyper-connections).</summary>
        protected virtual int StreamWidth => Config.HiddenSize;

        /// <summary>The residual stream entering the stack from the embedded rows [hidden, rows].</summary>
        protected virtual int EmitStreamInput(GraphStep step, int embedded) => embedded;

        /// <summary>A block's input [hidden, rows] from the stream; <paramref name="context"/> carries what the
        /// family's <see cref="EmitBlockOutput"/> needs (-1: nothing). Default: the pre-norm.</summary>
        protected virtual int EmitBlockInput(GraphStep step, int layer, int stream, BlockSite site, out int context)
        {
            context = -1;
            return step.RmsNorm(stream, site == BlockSite.Mixer ? LayerName(layer, "attn_norm.weight") : FfnNormWeight(layer));
        }

        /// <summary>The stream after a block's output [hidden, rows] joins it. Default: the residual add.</summary>
        protected virtual int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
            => step.Op(GraphOp.Add, stream, output);

        /// <summary>The head's input [hidden, rows] from the stream. Default: the output norm.</summary>
        protected virtual int EmitStreamOutput(GraphStep step, int stream) => step.RmsNorm(stream, "output_norm.weight");

        /// <summary>A DFlash drafter's tapped row [hidden, rows] of the stream entering a layer. Default: the stream.</summary>
        protected virtual int EmitCaptureRow(GraphStep step, int stream) => stream;

        private readonly Dictionary<(int Layer, string Suffix), string> _layerNames = new();

        /// <summary><c>blk.{layer}.{suffix}</c>, built once (traces run every token).</summary>
        protected string LayerName(int layer, string suffix)
        {
            if (!_layerNames.TryGetValue((layer, suffix), out string name))
                _layerNames[(layer, suffix)] = name = $"blk.{layer}.{suffix}";
            return name;
        }

        // ---------------------------------------------------------------- branches

        /// <summary>What a family adds where an attention layer reads its input <paramref name="x"/> [hidden, rows], before
        /// the attention (Qwen 3.8 writes its QSA indexer keys); default nothing.</summary>
        protected virtual void EmitAttentionInput(GraphStep step, int layer, int x) { }

        /// <summary>The graph type of the K/V caches (and extra key caches).</summary>
        private protected int KvCacheGraphType => KvGgmlType(_kvCacheDtype.ToDType());

        /// <summary>The mask sequence <paramref name="s"/> attends under in <paramref name="layer"/>: <paramref name="mask"/>
        /// ([<paramref name="keys"/>, its rows], F16, causal) unless the family narrows it (Qwen 3.8's QSA selection).</summary>
        protected virtual int EmitAttentionMask(GraphStep step, int layer, int x, int s, int mask, int keys) => mask;

        /// <summary>A bound on the cells any row of sequence <paramref name="s"/> attends in <paramref name="layer"/> under
        /// its mask (a selection's width), so the backend may attend only those; 0 for none.</summary>
        protected virtual int AttentionKvMax(GraphStep step, int layer, int s) => 0;

        private int EmitAttention(GraphStep step, int layer, int x)
        {
            EmitAttentionInput(step, layer, x);
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
                int ck = step.State(slot.K[layer][0], kvType, KvCacheSplit(), d, cap, kvHeads);
                int cv = step.State(slot.V[layer][0], kvType, KvCacheSplit(), dv, cap, kvHeads);

                int fa;
                // Tensor-parallel graphs write every chunk by index too: ggml's meta device splits
                // set_rows by KV head, while a copy into a strided cache view fails on the ranks.
                GraphCapabilities caps = GraphExecutor.Capabilities;
                if (count <= SmallChunkRows || _graphTpRanks != null || caps.IndexedCacheWrites
                    || (caps.PaddedCacheAttention && kvType != GraphFormat.F32))
                {
                    // Decode and short chunks (a verify window): rows are written by index and
                    // attention reads the padded cache window under a causal mask, so every row is
                    // computed as a decode row is and a window bucket replays the same graph.
                    int rowsAt = step.WriteRows(start, count);
                    step.Effect(step.Op(GraphOp.SetRows, ck, kFresh, rowsAt));
                    step.Effect(step.Op(GraphOp.SetRows, cv, vFresh, rowsAt));
                    int stepRows = DecodeWindowStep;
                    int window = Math.Min(cap, (start + count + stepRows - 1) / stepRows * stepRows);
                    int gqa = heads / kvHeads;
                    int kWin = step.Op(GraphOp.KvWindow, stackalloc[] { ck }, stackalloc long[] { d, cap, kvHeads, 0, window, kvType, count, gqa });
                    int vWin = step.Op(GraphOp.KvWindow, stackalloc[] { cv }, stackalloc long[] { dv, cap, kvHeads, 0, window, kvType, count, gqa });
                    fa = step.FlashAttention(qPerm, kWin, vWin, EmitAttentionMask(step, layer, x, s, step.ChunkMask(window, start, count), window), scale,
                        AttentionKvMax(step, layer, s));
                }
                else
                {
                    // A prompt chunk is written at its offset and attends to the cached prefix and
                    // its own fresh rows (F32), as the fused prefill graphs do.
                    step.Effect(step.Op(GraphOp.Cpy, kFresh, step.View3(ck, d, count, kvHeads, kRow, kRow * cap, start * kRow)));
                    step.Effect(step.Op(GraphOp.Cpy, vFresh, step.View3(cv, dv, count, kvHeads, vRow, vRow * cap, start * vRow)));
                    int kAttn = kFresh, vAttn = vFresh;
                    if (start > 0)
                    {
                        int kPrev = step.View3(ck, d, start, kvHeads, kRow, kRow * cap, 0);
                        int vPrev = step.View3(cv, dv, start, kvHeads, vRow, vRow * cap, 0);
                        if (kvType != GraphFormat.F32)
                        {
                            kPrev = step.Op(GraphOp.Cast, stackalloc[] { kPrev }, stackalloc long[] { GraphFormat.F32 });
                            vPrev = step.Op(GraphOp.Cast, stackalloc[] { vPrev }, stackalloc long[] { GraphFormat.F32 });
                        }
                        kAttn = step.Op(GraphOp.Concat, stackalloc[] { kPrev, kFresh }, stackalloc long[] { 1 });
                        vAttn = step.Op(GraphOp.Concat, stackalloc[] { vPrev, vFresh }, stackalloc long[] { 1 });
                    }
                    fa = step.FlashAttention(qPerm, kAttn, vAttn, EmitAttentionMask(step, layer, x, s, step.CausalMask(start, count), start + count), scale);
                }
                int part = step.Reshape(fa, heads * dv, count);
                attn = attn < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { attn, part }, stackalloc long[] { 1 });
            }

            if (gate >= 0)
            {
                int g = step.Reshape(step.Cont(gate), (int)qDim, rows);
                attn = step.Op(GraphOp.Mul, attn, step.Op(GraphOp.Sigmoid, g));
            }
            return step.MatMul(LayerName(layer, "attn_output.weight"), attn);
        }

        private int EmitDenseFfn(GraphStep step, int layer, int x)
        {
            bool fused = _gateUpFused[layer];
            string gate = LayerName(layer, fused ? "ffn_gate_up.weight" : "ffn_gate.weight"), up = LayerName(layer, "ffn_up.weight");
            string down = LayerName(layer, "ffn_down.weight");
            // Quantized weights without per-tensor scales form a region a backend may run as one feed-forward.
            bool region = UnscaledQuantized(gate) && UnscaledQuantized(down) && (fused || UnscaledQuantized(up));
            if (region)
            {
                if (fused) step.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, step.Weight(gate), step.Weight(down) });
                else step.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, step.Weight(gate), step.Weight(up), step.Weight(down) }, stackalloc long[] { 1 });
            }
            int act = fused
                ? step.Op(GraphOp.Swiglu, step.MatMul(gate, x))
                : step.Op(GraphOp.SwigluSplit, step.MatMul(gate, x), step.MatMul(up, x));
            int y = step.MatMul(down, act);
            if (region) step.EndFused(stackalloc[] { y });
            return y;
        }

        private bool UnscaledQuantized(string name) => _quantWeights.TryGetValue(name, out QuantizedWeight q) && q.Scale == 1.0f;

        private int EmitMoe(GraphStep step, int layer, int x)
        {
            string p = LayerName(layer, "");
            int rows = step.Rows, hidden = Config.HiddenSize, k = _expertsUsed;
            int logits = step.MatMul(p + "ffn_gate_inp.weight", x);                                    // [E, rows]
            // Experts on the device: routing and experts are one region an executor may run with its own MoE kernels
            // (MLX keeps the top-K on the device instead of reading the logits back every layer).
            bool region = !(IsExpertLayerOffloaded(layer) && GraphExecutor.Capabilities.HostExperts);
            if (region)
                step.BeginFused(GraphFusion.RoutedExperts, stackalloc[]
                {
                    x, logits, step.Experts(p + "ffn_gate_exps.weight"), step.Experts(p + "ffn_up_exps.weight"),
                    step.Experts(p + "ffn_down_exps.weight"),
                }, stackalloc long[] { k, NormalizeExpertWeights ? 1 : 0 });
            int probs = step.Op(GraphOp.SoftMaxExt, logits);                                            // [E, rows]
            int selected = step.Op(GraphOp.TopK, stackalloc[] { probs }, stackalloc long[] { k });      // [K, rows]
            int weights = step.Op(GraphOp.GetRows, step.Reshape(probs, 1, _numExperts, rows), selected); // [1, K, rows]
            if (NormalizeExpertWeights)
            {
                // llama.cpp's build_moe_ffn chain with its closing scale, which ggml-metal runs as one kernel with the
                // softmax and top-k. Neither the clamp nor the scale changes a value (K of the softmax weights sum far
                // above the clamp); the scale also keeps the chain's last node the one a host-expert pause pins.
                int picked = step.Reshape(weights, k, rows);
                int sum = step.Op(GraphOp.Clamp, stackalloc[] { step.Op(GraphOp.SumRows, picked) }, default,
                    stackalloc[] { 6.103515625e-5f, float.PositiveInfinity });                              // [1, rows]
                weights = step.Op(GraphOp.Scale, stackalloc[] { step.Reshape(step.Op(GraphOp.Div, picked, sum), 1, k, rows) }, default,
                    stackalloc[] { 1f });
            }

            int output;
            if (IsExpertLayerOffloaded(layer) && GraphExecutor.Capabilities.HostExperts)
            {
                // The experts stay in host memory: the program pauses for them (and a registered accelerator tier
                // runs the routes it holds).
                int gateStack = step.Experts(p + "ffn_gate_exps.weight"), upStack = step.Experts(p + "ffn_up_exps.weight"),
                    downStack = step.Experts(p + "ffn_down_exps.weight");
                // The next offloaded layer's router and gate stack: the seam may read that layer's predicted misses
                // into its tier while the accelerator works (Strata's read-ahead; the GGML plugin decides).
                int next = layer + 1;
                output = next < Config.NumLayers && _moeLayer[next] && IsExpertLayerOffloaded(next) && OwnsLayer(next)
                         && StageOfLayer(next) == StageOfLayer(layer)
                    ? step.Op(GraphOp.MoeHostExperts, stackalloc[]
                    {
                        x, selected, weights, gateStack, upStack, downStack,
                        step.Weight(LayerName(next, "ffn_gate_inp.weight")), step.Experts(LayerName(next, "ffn_gate_exps.weight")),
                    }, stackalloc long[] { layer })
                    : step.Op(GraphOp.MoeHostExperts, stackalloc[] { x, selected, weights, gateStack, upStack, downStack }, stackalloc long[] { layer });
            }
            else
            {
                int input = step.Reshape(x, hidden, 1, rows);
                int gate = step.Op(GraphOp.MulMatId, step.Experts(p + "ffn_gate_exps.weight"), input, selected);
                int up = step.Op(GraphOp.MulMatId, step.Experts(p + "ffn_up_exps.weight"), input, selected);
                int act = step.Op(GraphOp.SwigluSplit, gate, up);                                            // [ff, K, rows]
                int down = step.Op(GraphOp.MulMatId, step.Experts(p + "ffn_down_exps.weight"), act, selected); // [hidden, K, rows]
                down = step.Op(GraphOp.Mul, down, weights);
                long rowBytes = hidden * 4L;
                output = -1;
                for (int j = 0; j < k; j++)
                {
                    int slice = step.View2(down, hidden, rows, rowBytes * k, j * rowBytes);
                    output = output < 0 ? slice : step.Op(GraphOp.Add, output, slice);
                }
                output = step.Cont(output);
                step.EndFused(stackalloc[] { output });
            }

            if (HasWeight(p + "ffn_gate_shexp.weight"))
            {
                int sa = step.Op(GraphOp.SwigluSplit, step.MatMul(p + "ffn_gate_shexp.weight", x), step.MatMul(p + "ffn_up_shexp.weight", x));
                int shared = step.MatMul(p + "ffn_down_shexp.weight", sa);
                if (_weights.ContainsKey(p + "ffn_gate_inp_shexp.weight"))
                {
                    int g = step.Op(GraphOp.MulMat, step.WeightAs(p + "ffn_gate_inp_shexp.weight", hidden, 1), x);   // [1, rows]
                    shared = step.Op(GraphOp.Mul, shared, step.Op(GraphOp.Sigmoid, g));
                }
                output = step.Op(GraphOp.Add, output, shared);
            }
            return output;
        }

        // ---------------------------------------------------------------- the step

        /// <summary>Rows a decode attention window grows by: 64 on Metal, whose padded tail costs
        /// more than a replay rebuild every 64 tokens (measured with Qwen 3.5's fused decode); 256
        /// (the flash-attention KV stride) elsewhere.</summary>
        private int DecodeWindowStep => GraphExecutor?.Capabilities.DecodeWindowRows ?? 256;

        private IGraphExecutor _graphExecutor;
        private bool _graphExecutorResolved;

        /// <summary>The backend's graph executor plugin (by the model's allocator), or null.</summary>
        protected IGraphExecutor GraphExecutor
        {
            get
            {
                if (!_graphExecutorResolved)
                {
                    _graphExecutor = GraphExecutors.For(_allocator);
                    _graphExecutorResolved = true;
                }
                return _graphExecutor;
            }
        }

        /// <summary>Bytes of one row of <paramref name="ne0"/> elements of a ggml type.</summary>
        private protected static long RowBytes(int type, long ne0) => type switch
        {
            GraphFormat.F32 => ne0 * 4,
            GraphFormat.F16 => ne0 * 2,
            GraphFormat.Q8_0 => ne0 / 32 * 34,
            GraphFormat.Q4_0 => ne0 / 32 * 18,
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
            private readonly StandardDecoderModel _m;
            private readonly GraphProgramBuilder _b;
            private readonly Dictionary<string, int> _weights = new(StringComparer.Ordinal);
            private readonly Dictionary<Tensor, int> _states = new(ReferenceEqualityComparer.Instance);
            private readonly List<GCHandle> _pins = new();
            private readonly List<int> _inputs = new();
            private readonly List<IntPtr> _inputData = new();
            private readonly List<long> _inputBytes = new();
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
                _b = new GraphProgramBuilder(record);
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
            internal (ulong, ulong) Signature => _b.Signature;

            // ---- tracing

            private void Mix(long value) => _b.Mix(value);

            private int Leaf(GraphLeafRole role, int type, ReadOnlySpan<long> ne, IntPtr data, long bytes, LeafSplit split = null, object handle = null)
                => _b.Leaf(role, type, ne, data, bytes, split == null ? null : new GraphSplit(split.Axis, split.Ne, split.Nr), handle);

            public int Op(GraphOp op, int a) => Op(op, stackalloc[] { a });
            public int Op(GraphOp op, int a, int b) => Op(op, stackalloc[] { a, b });
            public int Op(GraphOp op, int a, int b, int c) => Op(op, stackalloc[] { a, b, c });

            public int Op(GraphOp op, ReadOnlySpan<int> srcs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
                => _b.Op(op, srcs, ip, fp);

            /// <summary>Run <paramref name="node"/> for its side effect (a cache or state write).</summary>
            public void Effect(int node) => _b.Effect(node);

            /// <summary>A layer's output: a point where a lazily building executor may start running.</summary>
            public void Boundary(int node) => _b.Boundary(node);

            /// <summary>The ops up to <see cref="EndFused"/> are a region of <paramref name="kind"/> (see
            /// <see cref="GraphFusion"/>): an executor with its kernel computes the results in one dispatch.</summary>
            public void BeginFused(GraphFusion kind, ReadOnlySpan<int> inputs, ReadOnlySpan<long> ip = default, ReadOnlySpan<float> fp = default)
                => _b.BeginFused(kind, inputs, ip, fp);

            public void EndFused(ReadOnlySpan<int> results) => _b.EndFused(results);

            /// <summary>Write <paramref name="node"/>'s result in place inside the state leaf
            /// <paramref name="baseLeaf"/> at <paramref name="offset"/> bytes; where the base is not
            /// resident, the <paramref name="fallback"/> copy runs instead (see GgmlGraph.Alias).</summary>
            public void Alias(int node, int baseLeaf, long offset, int fallback) => _b.Alias(node, baseLeaf, offset, fallback);

            /// <summary>A step result: <paramref name="floats"/> F32 values of <paramref name="node"/>;
            /// returns its index among the step's results.</summary>
            internal int Output(int node, long floats) => _b.Output(node, floats);

            // ---- leaves

            /// <summary>A model weight by name: quantized (its device-cache key) or F32.</summary>
            public int Weight(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (_m._quantWeights.TryGetValue(name, out QuantizedWeight q))
                {
                    IntPtr key = q.CacheKey != IntPtr.Zero ? q.CacheKey : q.Data;
                    long[] ne = { q.Ne0, q.Ne1 };
                    node = Leaf(GraphLeafRole.Weight, q.GgmlType, ne, key, q.RawBytes, _m.SplitOf(name, ne, q.GgmlType), q);
                }
                else if (_m._weights.TryGetValue(name, out Tensor t))
                {
                    long[] ne = Extents(t);
                    node = TensorLeaf(GraphLeafRole.Weight, t, GraphFormat.F32, ne, _m.SplitOf(name, ne, GraphFormat.F32));
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
                return TensorLeaf(GraphLeafRole.Weight, t, GraphFormat.F32, ne);
            }

            /// <summary>Stacked experts [ne0, ne1, experts] of a routed-expert layer.</summary>
            public int Experts(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (!_m._stackedExpertWeights.TryGetValue(name, out StackedExpertWeights w)) throw Reject($"experts '{name}' are not stacked");
                long[] ne = { w.PerExpertNe0, w.PerExpertNe1, w.NumExperts };
                node = Leaf(GraphLeafRole.Weight, w.GgmlType, ne, w.Data, w.TotalRawBytes, _m.SplitOf(name, ne, w.GgmlType), _m.ExpertSet(name, w));
                _weights[name] = node;
                return node;
            }

            /// <summary>An F32 tensor the family prepared (device constants, norms) as a read-only leaf.</summary>
            public int TensorWeight(Tensor t, params long[] ne) => TensorLeaf(GraphLeafRole.Weight, t, GraphFormat.F32, ne.Length == 0 ? Extents(t) : ne);

            /// <summary>Read-write state the graph updates in place (a slot's cache or recurrent state).</summary>
            public int State(Tensor t, int type, params long[] ne) => State(t, type, null, ne);

            /// <summary>A state spread over tensor-parallel ranks by <paramref name="split"/> (null: mirrored).</summary>
            internal int State(Tensor t, int type, LeafSplit split, params long[] ne)
            {
                if (_states.TryGetValue(t, out int node)) return node;
                node = TensorLeaf(GraphLeafRole.State, t, type, ne, _m._graphTpRanks != null ? split : null);
                _states[t] = node;
                return node;
            }

            /// <summary>Sequence s's recurrent state of (layer, rank 0).</summary>
            internal RecurrentState SequenceState(int s, int layer) => Slots[s].Recurrent[layer][0];

            private int TensorLeaf(GraphLeafRole role, Tensor t, int type, ReadOnlySpan<long> ne, LeafSplit split = null)
                => Leaf(role, type, ne, LeafIdentity(t), t.Storage.ByteLength, split, t);

            /// <summary>A tensor leaf's bytes for an executor that binds host memory, else a stable identity
            /// (a device storage's host pointer would sync it every step).</summary>
            private IntPtr LeafIdentity(Tensor t) => _m.GraphExecutor.Capabilities.BindsHostPointers
                ? TensorComputePrimitives.GetStoragePointer(t)
                : GraphIdentity.Of(t.Storage, t.StorageOffset);

            private static long[] Extents(Tensor t)
            {
                var ne = new long[t.DimensionCount];
                for (int i = 0; i < ne.Length; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
                return ne;
            }

            // ---- per-run inputs

            /// <summary>An F32 activation [cols, rows] read from a host tensor every run.</summary>
            internal int Input(Tensor t, long cols, long rows) => Input(t, cols, rows, InputKind.Opaque);

            /// <summary>The step's embedded (or a stage's incoming) rows: the one tensor input an advanced step swaps.</summary>
            internal int HiddenInput(Tensor t, long cols, long rows) => Input(t, cols, rows, InputKind.Hidden);

            private int Input(Tensor t, long cols, long rows, InputKind kind)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F32, stackalloc long[] { cols, rows }, IntPtr.Zero, 0);
                IntPtr data = _m.GraphExecutor.Capabilities.BindsHostPointers ? TensorComputePrimitives.GetStoragePointer(t) : IntPtr.Zero;
                AddInput(node, data, cols * rows * sizeof(float), t, kind);
                return node;
            }

            public int InputI32(int[] values, long count) => InputI32(values, count, InputKind.Opaque);

            /// <summary>I32 values that stay what they are while the rows' positions move (the rows each sequence's
            /// logits come from).</summary>
            internal int ConstantInputI32(int[] values, long count) => InputI32(values, count, InputKind.Constant);

            private int InputI32(int[] values, long count, InputKind kind)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.I32, stackalloc long[] { count }, IntPtr.Zero, 0);
                AddInput(node, Pin(values), count * sizeof(int), kind: kind, values: values);
                return node;
            }

            public int InputF32(float[] values, params long[] ne)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), (long)values.Length * sizeof(float));
                return node;
            }

            internal void SetPositions(int[] positions)
            {
                PositionValues = positions;
                Positions = InputI32(positions, positions.Length, InputKind.Positions);
            }

            /// <summary>Axis-major [4 * rows] rotary positions (ggml_rope_multi); rotary ops then take them.</summary>
            internal void SetPositionTable(int[] axisMajor) => PositionTable = InputI32(axisMajor, axisMajor.Length);

            /// <summary>Multi-axis rotary positions [4 * rows] (I32), or -1.</summary>
            public int PositionTable { get; private set; } = -1;

            /// <summary>The rows [count, 1] (I64) a short chunk is written at: start .. start + count - 1.</summary>
            internal int WriteRows(int start, int count)
            {
                if (_writeIndex.TryGetValue(start * 64 + count, out int node)) return node;
                node = Leaf(GraphLeafRole.Input, GraphFormat.I64, stackalloc long[] { count, 1 }, IntPtr.Zero, 0);
                var rows = new long[count];
                for (int i = 0; i < count; i++) rows[i] = start + i;
                AddInput(node, Pin(rows), count * sizeof(long), kind: InputKind.WriteRows, values: rows);
                _writeIndex[start * 64 + count] = node;
                return node;
            }

            /// <summary>An F16 causal mask [window, count]: row q sees cache rows 0 .. start + q.</summary>
            internal int ChunkMask(int window, int start, int count)
            {
                if (_decodeMask >= 0 && _decodeMaskWindow == window && _decodeMaskValid == start * 64 + count) return _decodeMask;
                var mask = new ushort[(long)window * count];
                FillChunkMask(mask, window, start, count);
                _decodeMask = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { window, count }, IntPtr.Zero, 0);
                AddInput(_decodeMask, Pin(mask), mask.LongLength * sizeof(ushort), kind: InputKind.DecodeMask, values: mask,
                    window: window, start: start, count: count);
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
                _mask = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { kv, count }, IntPtr.Zero, 0);
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

            /// <summary>The program <see cref="Build"/> recorded.</summary>
            internal GraphProgram Program { get; private set; }

            /// <summary>A streaming executor's sink: entries and input values go there as they are emitted.</summary>
            internal IGraphSink Sink { get => _b.Sink; init => _b.Sink = value; }

            private void FillChunkMask(ushort[] mask, int window, int start, int count)
            {
                Array.Clear(mask);
                for (int q = 0; q < count; q++)
                {
                    for (int k = 0; k < Math.Min(AttentionFloor, window); k++) mask[(long)q * window + k] = 0xFC00;   // -inf
                    for (int k = start + q + 1; k < window; k++) mask[(long)q * window + k] = 0xFC00;
                }
            }

            private void AddInput(int node, IntPtr data, long bytes, object handle = null, InputKind kind = InputKind.Opaque,
                Array values = null, int window = 0, int start = 0, int count = 0)
            {
                _b.Sink?.InputData(node, data, bytes, handle);
                _inputs.Add(node);
                _inputData.Add(data);
                _inputBytes.Add(bytes);
                _inputHandles.Add(handle);
                _inputMeta.Add(new InputMeta(kind, values, window, start, count));
            }

            private readonly List<object> _inputHandles = new();

            // ---- advancing a decode step

            /// <summary>How an input follows the rows' positions.</summary>
            private enum InputKind : byte { Opaque, Hidden, Positions, WriteRows, DecodeMask, Constant }

            private readonly record struct InputMeta(InputKind Kind, Array Values, int Window, int Start, int Count);

            private readonly List<InputMeta> _inputMeta = new();

            /// <summary>Every input follows positions in a known way, so the step can be run again one position on.</summary>
            internal bool Advanceable => !_inputMeta.Exists(m => m.Kind == InputKind.Opaque);

            private List<(int Start, int Max)> _advanceLimits;

            /// <summary>The graph a trace builds for a sequence continuing at <paramref name="start"/> stays the same only
            /// up to <paramref name="maxStart"/> (a family whose graph changes past a position: QSA attends densely only
            /// within its budget); an advance past it is refused.</summary>
            internal void LimitAdvance(int start, int maxStart) => (_advanceLimits ??= new()).Add((start, maxStart));

            /// <summary>Rewrite this step's inputs for every row one position on, with <paramref name="hidden"/> as its
            /// rows: positions and write rows plus one, decode masks one column wider (each must stay inside its
            /// window and every limit honored: false, nothing changed, when one would not).</summary>
            internal bool TryAdvance(Tensor hidden)
            {
                foreach (InputMeta m in _inputMeta)
                    if (m.Kind == InputKind.DecodeMask && m.Start + m.Count + 1 > m.Window) return false;
                if (_advanceLimits != null)
                {
                    foreach ((int start, int max) in _advanceLimits)
                        if (start + 1 > max) return false;
                    for (int i = 0; i < _advanceLimits.Count; i++)
                        _advanceLimits[i] = (_advanceLimits[i].Start + 1, _advanceLimits[i].Max);
                }
                for (int i = 0; i < _inputMeta.Count; i++)
                {
                    InputMeta m = _inputMeta[i];
                    switch (m.Kind)
                    {
                        case InputKind.Hidden:
                            _inputData[i] = _m.GraphExecutor.Capabilities.BindsHostPointers ? TensorComputePrimitives.GetStoragePointer(hidden) : IntPtr.Zero;
                            _inputHandles[i] = hidden;
                            break;
                        case InputKind.Positions:
                            foreach (ref int p in ((int[])m.Values).AsSpan()) p++;
                            break;
                        case InputKind.WriteRows:
                            foreach (ref long r in ((long[])m.Values).AsSpan()) r++;
                            break;
                        case InputKind.DecodeMask:
                            FillChunkMask((ushort[])m.Values, m.Window, m.Start + 1, m.Count);
                            _inputMeta[i] = m with { Start = m.Start + 1 };
                            break;
                    }
                }
                return true;
            }

            // ---- composites

            /// <summary>rms_norm(x) * weight.</summary>
            public int RmsNorm(int x, string weight) => RmsNorm(x, Weight(weight), _m.Config.Eps);

            public int RmsNorm(int x, int weight, float eps)
                => Op(GraphOp.Mul, Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), weight);

            /// <summary>rms_norm(x, eps / n) / sqrt(n) over ne0: the L2 normalization ggml's GDN path uses.</summary>
            public int L2Norm(int x, long n, float eps)
                => Op(GraphOp.Scale, stackalloc[] { Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps / n }) },
                    default, stackalloc[] { 1.0f / MathF.Sqrt(n) });

            /// <summary>weight · x, with a quantized weight's per-tensor scale applied.</summary>
            public int MatMul(string weight, int x)
            {
                int y = Op(GraphOp.MulMat, Weight(weight), x);
                if (_m._quantWeights.TryGetValue(weight, out QuantizedWeight q) && q.Scale != 1.0f)
                    y = Op(GraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { q.Scale });
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
                int block = Op(GraphOp.View, stackalloc[] { w }, stackalloc long[] { 2, ne0, rows, 1, 1, rowBytes, 0, 0, firstRow * rowBytes });
                int y = Op(GraphOp.MulMat, block, x);
                if (scale != 1.0f)
                    y = Op(GraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { scale });
                return y;
            }

            public int Reshape(int x, params long[] ne) => Op(GraphOp.Reshape, stackalloc[] { x }, ne);

            /// <summary>An F16 input [ne...] read from <paramref name="values"/> every run (a family's mask).</summary>
            internal int InputF16(ushort[] values, params long[] ne)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), values.LongLength * sizeof(ushort));
                return node;
            }

            /// <summary>Rotary over x [d, heads, n] at <paramref name="positions"/> (I32 [n]) rather than the step's rows'
            /// positions (keys pooled at their blocks' first cells).</summary>
            internal int RopeAt(int x, RotarySpec spec, int positions)
                => Op(GraphOp.RopeExt, stackalloc[] { x, positions },
                    stackalloc long[] { spec.Dims, spec.NeoX ? 2 : 0, spec.OriginalContext },
                    stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });

            /// <summary>Positions [count] of sequence <paramref name="s"/>'s rows (a view of the step's).</summary>
            internal int SequencePositions(int s)
                => Sequences == 1 ? Positions
                    : Op(GraphOp.View, stackalloc[] { Positions }, stackalloc long[] { 1, Counts[s], 1, 1, 1, Counts[s] * 4L, 0, 0, First[s] * 4L });

            /// <summary>A view of an F32 node; recorded as strided unless its rows are packed.</summary>
            public int View2(int x, long ne0, long ne1, long nb1, long offset)
            {
                int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 2, ne0, ne1, 1, 1, nb1, 0, 0, offset });
                if (ne1 > 1 && nb1 != ne0 * 4) _strided.Add(node);
                return node;
            }

            public int View4(int x, long ne0, long ne1, long ne2, long ne3, long nb1, long nb2, long nb3, long offset)
            {
                int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 4, ne0, ne1, ne2, ne3, nb1, nb2, nb3, offset });
                if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4) || (ne3 > 1 && nb3 != ne0 * ne1 * ne2 * 4))
                    _strided.Add(node);
                return node;
            }

            /// <summary>The backend's norms, unary ops and recurrent kernels read row-contiguous strided
            /// views (ggml-metal, ggml-cuda); elsewhere a strided input is copied contiguous first.</summary>
            public bool StridedInputs => _m.GraphExecutor.Capabilities.StridedViews;

            public int View3(int x, long ne0, long ne1, long ne2, long nb1, long nb2, long offset)
            {
                int node = Op(GraphOp.View, stackalloc[] { x }, stackalloc long[] { 3, ne0, ne1, ne2, 1, nb1, nb2, 0, offset });
                if ((ne1 > 1 && nb1 != ne0 * 4) || (ne2 > 1 && nb2 != ne0 * ne1 * 4)) _strided.Add(node);
                return node;
            }

            /// <summary>Permute axes (strided).</summary>
            public int Permute(int x, long a0, long a1, long a2, long a3)
            {
                int node = Op(GraphOp.Permute, stackalloc[] { x }, stackalloc long[] { a0, a1, a2, a3 });
                _strided.Add(node);
                return node;
            }

            public int Transpose(int x)
            {
                int node = Op(GraphOp.Transpose, x);
                _strided.Add(node);
                return node;
            }

            /// <summary>A contiguous copy of <paramref name="x"/>, or <paramref name="x"/> itself when it
            /// already is contiguous (every op result is; views, permutes and transposes may not be).</summary>
            public int Cont(int x) => _strided.Contains(x) ? Op(GraphOp.Cont, x) : x;

            /// <summary>Rotary at the step's positions; with a position table the family's multi-axis
            /// rotary (ggml's interleaved M-RoPE over its sections), which equals the scalar one for text.</summary>
            public int Rope(int x, RotarySpec spec)
            {
                if (PositionTable >= 0 && _m.RotarySections is { } sections)
                    return Op(GraphOp.RopeMulti, stackalloc[] { x, PositionTable },
                        stackalloc long[] { spec.Dims, 40 /* GGML_ROPE_TYPE_IMROPE */, spec.OriginalContext,
                            sections[0], sections[1], sections[2], sections.Length > 3 ? sections[3] : 0 },
                        stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });
                return Op(GraphOp.RopeExt, stackalloc[] { x, Positions },
                    stackalloc long[] { spec.Dims, spec.NeoX ? 2 : 0, spec.OriginalContext },
                    stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });
            }

            /// <summary>Flash attention of q [d, rows, heads] over k/v [d, kv, kv_heads]: [dv, heads, rows].</summary>
            /// <summary>Flash attention; <paramref name="kvMax"/> &gt; 0 bounds the finite entries of every mask row, which
            /// lets a backend attend only those (ggml_flash_attn_ext_set_n_kv_max).</summary>
            public int FlashAttention(int q, int k, int v, int mask, float scale, int kvMax = 0)
            {
                Span<int> srcs = stackalloc[] { q, k, v, mask };
                ReadOnlySpan<long> ip = kvMax > 0 ? stackalloc long[] { 10 /* F32 accumulation */, kvMax } : stackalloc long[] { 10 };
                return Op(GraphOp.FlashAttnExt, mask >= 0 ? srcs : srcs[..3], ip, stackalloc[] { scale, 0f, 0f });
            }

            private InvalidOperationException Reject(string why)
            {
                Rejected = true;
                return new InvalidOperationException(why);
            }

            // ---- native

            /// <summary>Build the trace as a native graph (node handles equal trace indices).</summary>
            /// <summary>Compile the recorded program with the backend's executor plugin.</summary>
            internal CachedGraph Build(bool replayable, bool reorder)
            {
                if (!_record) throw new InvalidOperationException("a hash-only trace cannot be built");
                GraphProgram program = _b.ToProgram(replayable, reorder, _m._graphTpRanks);
                Program = program;
                return new CachedGraph { Plan = _m.GraphExecutor.Compile(program) };
            }

            internal bool Run(IGraphPlan plan) => plan.Run(CollectionsMarshal.AsSpan(_inputs), CollectionsMarshal.AsSpan(_inputData),
                CollectionsMarshal.AsSpan(_inputBytes), CollectionsMarshal.AsSpan(_inputHandles));

            public void Dispose()
            {
                foreach (GCHandle pin in _pins) pin.Free();
                _pins.Clear();
            }
        }
    }
}
