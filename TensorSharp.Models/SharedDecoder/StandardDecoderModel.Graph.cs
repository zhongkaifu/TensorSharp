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

        /// <summary>Causal mask inputs the traced steps emitted (<see cref="GraphStep.ChunkMask"/>,
        /// <see cref="GraphStep.CausalMask"/>): one a sequence and step, whatever the attention layers.</summary>
        internal long CausalMaskInputs { get; private set; }

        /// <summary>Tests: a step caches only the last causal mask of each kind it emitted, as before masks were cached per
        /// sequence (a step of several sequences then emitted every sequence's mask again in every attention layer).</summary>
        internal bool SingleMaskCacheForTests { get; set; }

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

        /// <summary>
        /// Whether a step that reads only each sequence's last row (its logits or its final hidden row) runs the last layer's
        /// feed-forward on those rows alone, as llama.cpp's graphs do: the stream leaving the last layer's mixer is cut to the
        /// sequences' last rows, and the other rows' feed-forward (dead work: nothing reads it) is not computed. The rows kept
        /// take products of fewer rows (a matrix-vector kernel where the step had a matrix-matrix one), so their logits round
        /// differently from the unpruned step's: off by default, a family opts in where it measures faster and its parity
        /// holds. A MoE last layer, a step reading every row's logits or the final rows, captures past the stack, draft follows
        /// and the drafter's rings never prune (<see cref="PrunesLastLayer"/>).
        /// </summary>
        protected virtual bool PrunesLastLayerRows => false;

        /// <summary>
        /// Whether a prompt chunk on ggml-vulkan writes its K/V rows by index and attends the padded F16 cache window, as decode
        /// rows do (and every chunk does on ggml-metal and ggml-cuda, GraphCapabilities.PaddedCacheAttention), instead of F32
        /// copies of the whole cached prefix concatenated with its own rows (a cast and a concat a layer, then flash attention
        /// over F32 K/V). Off by default: a family opts in once its prompts are validated that way on ggml-vulkan.
        /// </summary>
        protected virtual bool PadsPromptWindowsOnVulkan => false;

        /// <summary>A reason the current configuration cannot run whole-forward graphs, or null.</summary>
        private string GraphUnsupportedReason()
        {
            if (!GraphLoweringEnabled) return "disabled";
            if (GraphExecutor == null) return "this backend registers no graph executor";
            if (_mesh.TensorParallel > 1 && !_rankGraphs) return "tensor-parallel placement";
            if (LayerSplitGroup != null) return "multi-node pipeline";
            if (GraphRotary == null) return "the family has no graph rotary";
            if (_anyRecurrent && !RecurrentMixerLowers) return "recurrent layers without a graph mixer";
            if (StepAdapter != null) return "a request adapter is selected";
            for (int l = 0; l < Config.NumLayers; l++)
                if (_moeLayer[l] && IsExpertLayerOffloaded(l) && !GraphExecutor.Capabilities.HostExperts) return "host-offloaded experts";
            DType kv = _kvCacheDtype.ToDType();
            if (kv != DType.Float32 && kv != DType.Float16 && kv != DType.Q8_0 && kv != DType.Q4_0) return $"KV dtype {kv}";
            if ((kv == DType.Q8_0 || kv == DType.Q4_0) && !GraphExecutor.TakesKvCacheFormat(KvGgmlType(kv), _keyDim, _valueDim))
                return $"KV dtype {kv} on this backend's executor";
            // An executor that takes the block format in its attention kernels and row writes only (not in every view of a cache:
            // direct CUDA's q8_0) serves the families whose attention is those kernels' alone.
            if ((kv == DType.Q8_0 || kv == DType.Q4_0) && !GraphExecutor.Capabilities.QuantizedKvCache && !AttentionRoutesTakeQuantizedKv)
                return $"KV dtype {kv} with {Config.Architecture}'s attention routes on this backend's executor";
            return null;
        }

        /// <summary>Tests: why this model runs no whole-forward graphs (null where it runs them).</summary>
        internal string GraphRefusalForTests => GraphUnsupportedReason();

        /// <summary>Whether every route this family's attention takes reads and writes its K/V caches through the plain
        /// attention and row writes (no extra key caches, no selective-attention regions or masks of its own), so an executor
        /// that takes a block-quantized cache in those alone (<see cref="IGraphExecutor.TakesKvCacheFormat"/> without
        /// <see cref="GraphCapabilities.QuantizedKvCache"/>) runs the family's graphs over one. Default true, except for a family
        /// with sliding-window layers (<see cref="SlidingWindow"/>): their bands are selections read on the device
        /// (<see cref="GraphCapabilities.BandMasksOnDevice"/>, F16 only on direct CUDA), and their rings' rows wrap past the last
        /// cell, which direct CUDA's q8_0 row writes do not take (its writes of a recorded step's rows are not taken modulo the
        /// cells), so such a family keeps a block-quantized cache off that executor's graphs (its per-op path runs it).</summary>
        protected virtual bool AttentionRoutesTakeQuantizedKv => !(_anySlidingWindow ??= AnySlidingWindow());

        private bool? _anySlidingWindow;

        private bool AnySlidingWindow()
        {
            for (int l = 0; l < Config.NumLayers; l++)
                if (SlidingWindow(l) > 0) return true;
            return false;
        }

        private readonly Dictionary<string, IGraphExpertWeights> _graphExpertSets = new();

        /// <summary>Step kinds (shape, emitter) that ran as recorded programs: a streaming executor runs
        /// later steps of the kind as they are emitted.</summary>
        private readonly HashSet<string> _streamedKinds = new();

        /// <summary>The recorded program of each streamed kind: its readers per node let a stream release
        /// tensors after their last use.</summary>
        private readonly Dictionary<string, GraphProgram> _streamTemplates = new();

        /// <summary>The streamed kinds of steps that keep their recurrent states' snapshots (verify windows): only their
        /// emission reads the states' verify buffers (<see cref="ForgetProgramsKeyedOtherwise"/>).</summary>
        private readonly HashSet<string> _snapshotStreamedKinds = new();

        /// <summary>A stacked expert weight with its members (each a quantized weight with its own
        /// resident copy), for executors that run routed experts one at a time.</summary>
        private IGraphExpertWeights ExpertSet(string name, StackedExpertWeights stacked)
        {
            lock (_graphExpertSets)
            {
                if (_graphExpertSets.TryGetValue(name, out IGraphExpertWeights set)) return set;
                string baseName = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^7] : name;
                var members = new QuantizedWeight[stacked.NumExperts];
                for (int e = 0; e < members.Length; e++)
                    _quantWeights.TryGetValue($"{baseName}.{e}.weight", out members[e]);
                // An offloaded layer on a tensor executor: its stacks carry the model's expert host (host compute, tier).
                ModelExpertHost host = ExpertLayerOf(name) is int layer and >= 0 ? HostFor(layer) : null;
                return _graphExpertSets[name] = host != null
                    ? new HostedExpertSet(stacked, members, host)
                    : new GraphExpertSet(stacked, members);
            }
        }

        /// <summary>The layer of a stacked expert weight's name (blk.N.ffn_*_exps.weight), or -1.</summary>
        private static int ExpertLayerOf(string name)
        {
            if (!name.StartsWith("blk.", StringComparison.Ordinal)) return -1;
            int dot = name.IndexOf('.', 4);
            return dot > 4 && int.TryParse(name.AsSpan(4, dot - 4), out int layer) ? layer : -1;
        }

        private readonly Dictionary<(string Name, int Rank), IGraphExpertWeights> _rankExpertSets = new();

        /// <summary>Rank <paramref name="rank"/>'s slices of a stacked expert weight (each member's shard, sharded like a
        /// dense feed-forward), or null when the members were not sharded.</summary>
        private IGraphExpertWeights RankExpertSet(string name, int rank)
        {
            lock (_rankExpertSets)
            {
                if (_rankExpertSets.TryGetValue((name, rank), out IGraphExpertWeights set)) return set;
                if (_rankStacks.TryGetValue((name, _mesh.GlobalDevice(0, rank)), out RankExpertStack own)) return _rankExpertSets[(name, rank)] = own;
                if (!_stackedExpertWeights.TryGetValue(name, out StackedExpertWeights stacked)) return null;
                string baseName = name.EndsWith(".weight", StringComparison.Ordinal) ? name[..^7] : name;
                var members = new QuantizedWeight[stacked.NumExperts];
                for (int e = 0; e < members.Length; e++)
                {
                    if (!_tpQuantWeights.TryGetValue($"{baseName}.{e}.weight", out QuantizedWeight[] shards)) return null;
                    members[e] = shards[rank];
                }
                return _rankExpertSets[(name, rank)] = new ExpertShards(members);
            }
        }

        /// <summary>One rank's slices of a stacked expert weight: its members' shards (no stacked copy; executors that run
        /// routed experts through the members read them one at a time).</summary>
        private sealed class ExpertShards : IGraphExpertWeights
        {
            private readonly QuantizedWeight[] _members;

            public ExpertShards(QuantizedWeight[] members)
            {
                _members = members;
                foreach (QuantizedWeight m in members) Bytes += m.RawBytes;
                CacheKey = GraphIdentity.Of(this);
            }

            public int Format => _members[0].GgmlType;
            public long Ne0 => _members[0].Ne0;
            public long Ne1 => _members[0].Ne1;
            public long Ne2 => _members.Length;
            public IntPtr Data => IntPtr.Zero;
            public IntPtr CacheKey { get; }
            public long Bytes { get; }
            public IGraphWeight Expert(int index) => _members[index];
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
            /// <summary>Every row's greedy token (the largest logit, a tie to the lower id) in place of its logits: a greedy
            /// verify reads back an id a row, where it had read the whole vocabulary's logits and scanned them on the host.</summary>
            public bool GreedyIds { get; init; }
            /// <summary>Layers whose INPUT residual [hidden, rows] is returned (a drafter's features).</summary>
            public int[] CaptureLayers { get; init; } = Array.Empty<int>();
            /// <summary>Capture only the step's last this-many rows (0: every row): a block drafter's context holds only
            /// the rows nearest the prompt's end.</summary>
            public int CaptureTail { get; init; }
            /// <summary>Return the final-normed rows [hidden, rows] (a per-token draft head's input).</summary>
            public bool CaptureFinal { get; init; }
            /// <summary>Return each sequence's last row after the final norm [hidden, sequences] (a step driver's input:
            /// speech output); with <see cref="LastRowLogits"/> false the step runs no head.</summary>
            public bool LastRowHidden { get; init; }
            /// <summary>Keep the recurrent state after every row (a verify the accept count rewinds).</summary>
            public bool RecurrentSnapshots { get; init; }
            /// <summary>Flat (T, H, W) rotary positions of a single sequence's rows (multi-axis rotary).</summary>
            public int[] PositionTable { get; init; }
            /// <summary>The draft layer takes the step's rows as the trunk finishes them (<see cref="EmitDraftFollow"/>).</summary>
            public DraftFollow FollowDraft { get; init; }
            /// <summary>The followed rows' draft embedding [rows, hidden] when the step's input rows are not it; null reads
            /// them.</summary>
            public float[] DraftRows { get; init; }
            /// <summary>The DFlash drafter's rings take the following sequences' rows as the trunk finishes them
            /// (<see cref="EmitDFlashFollow"/>).</summary>
            public DFlashFollowPlan FollowDFlash { get; init; }
            /// <summary>The step forwards prompt tokens: each sequence whose rows number more than <see cref="SmallChunkRows"/>
            /// forwards a prompt chunk, and its attention op says so (<see cref="GraphAttentionRows.PromptChunk"/>). Set only by
            /// the prompt entry points (<see cref="PromptStep"/>), never for a verify window, a re-forward of generated rows or
            /// a draft step.</summary>
            public bool PromptChunks { get; init; }

            /// <summary>What the request changes in the emitted program (a streamed step's template key).</summary>
            public string Shape =>
                $"{LastRowLogits}{AllRowLogits}{CaptureFinal}{PositionTable != null}:{string.Join(".", CaptureLayers)}:{CaptureTail}"
                + (GreedyIds ? ":g" : "") + (LastRowHidden ? ":h" : "")
                + (FollowDraft == DraftFollow.None ? "" : $":f{(int)FollowDraft}{(DraftRows != null ? "e" : "")}")
                + (FollowDFlash == null ? "" : $":df{FollowDFlash.Key}")
                + (PromptChunks ? ":p" : "");

            /// <summary>This request with the DFlash rings' follow <paramref name="plan"/> (null: none).</summary>
            public GraphRequest WithDFlashFollow(DFlashFollowPlan plan) => new()
            {
                LastRowLogits = LastRowLogits, AllRowLogits = AllRowLogits, GreedyIds = GreedyIds, CaptureLayers = CaptureLayers,
                CaptureTail = CaptureTail, CaptureFinal = CaptureFinal, LastRowHidden = LastRowHidden, RecurrentSnapshots = RecurrentSnapshots,
                PositionTable = PositionTable, FollowDraft = FollowDraft, DraftRows = DraftRows, FollowDFlash = plan,
                PromptChunks = PromptChunks,
            };
        }

        /// <summary>Whether a step of these per-sequence row <paramref name="counts"/> forwarding prompt tokens marks its
        /// attention as a prompt chunk's (<see cref="GraphRequest.PromptChunks"/>): some sequence forwards more than
        /// <see cref="SmallChunkRows"/> rows. A decode step's and a short prompt tail's programs then stay as they were (their
        /// shape, signature and advance).</summary>
        private static bool PromptStep(int[] counts) => Array.Exists(counts, c => c > SmallChunkRows);

        /// <summary>Tests: no step marks its attention as a prompt chunk's (the programs as they were before the mark).</summary>
        internal bool PromptChunkMarksDisabledForTests { get; set; }

        /// <summary>Tests: when set, every attention op a traced step emits adds its ip here (GraphStep.FlashAttention).</summary>
        internal List<long[]> AttentionIpForTests { get; set; }

        /// <summary>Set while a prompt entry point that forwards through <see cref="RunSequence"/> or
        /// <see cref="SpecForward"/> runs (<see cref="ForwardRefillCore"/>, <see cref="SpecForwardPrompt"/>): its chunks are
        /// prompt tokens.</summary>
        private bool _promptScope;

        /// <summary>Whether a step of <paramref name="counts"/> rows a sequence marks its prompt chunks: it forwards prompt tokens
        /// (<paramref name="prompt"/>) and some sequence's rows are a chunk (<see cref="PromptStep"/>).</summary>
        private bool MarksPromptChunks(bool prompt, int[] counts) => prompt && !PromptChunkMarksDisabledForTests && PromptStep(counts);

        private sealed class GraphResult
        {
            public float[] Logits;
            public float[][] Captures;
            public float[] Final;
            /// <summary>Each sequence's last row after the final norm [hidden, sequences] (<see cref="GraphRequest.LastRowHidden"/>).</summary>
            public float[] LastHidden;
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

        /// <summary>The widest verify window a sequence advances (and records) instead of tracing each time.</summary>
        private const int MaxAdvancedWindowRows = 8;

        /// <summary>What else, beyond the rows, counts and outputs, decides what a step emits: a family whose graph
        /// depends on where its sequences are (QSA selects past its budget) names it, so a streamed step never frees
        /// nodes by the last readers of a template emitted the other way.</summary>
        protected virtual string EmissionVariant(int[] counts, int[] starts) => null;

        /// <summary>Decode steps run as the previous one advanced, without a trace.</summary>
        internal int DecodeAdvances { get; private set; }

        /// <summary>Of <see cref="DecodeAdvances"/>, the verify windows (more than one row a sequence).</summary>
        internal int WindowAdvances { get; private set; }

        /// <summary>Of <see cref="DecodeAdvances"/>, the draft layer's steps.</summary>
        internal int DraftAdvances { get; private set; }

        /// <summary>Of <see cref="DraftAdvances"/>, steps over a block drafter's ring moved to a position behind their last one
        /// (<see cref="GraphStep.TryRebase"/>; tests check a new request's drafts reuse the kept steps).</summary>
        internal int DraftRebases { get; private set; }

        /// <summary>Tests: true for a kept step's shape fails its recording's next launch as a launch the device has no memory
        /// for does (nothing launched; the step runs as usual and its recording is not tried again).</summary>
        internal Func<string, bool> RecordedLaunchFailsForTests { get; set; }

        /// <summary>The last decode step, kept to run again one position on.</summary>
        private sealed class DecodeReplay
        {
            public GraphStep Step;
            public CachedGraph Graph;
            public (ulong, ulong) Signature;
            /// <summary>The signature without the state leaves' identities (GraphStep.StructuralSignature).</summary>
            public (ulong, ulong) Structure;
            public StepOutputs Outputs;
            public Slot[] Slots;
            public Tensor[][][] K, V;
            public RecurrentState[][][] Recurrent;
            public int[] Capacity, RopeDelta, Starts;
            public int Floor;
            /// <summary>Each sequence's floor where they differ (GraphStep.AttentionFloors); null: <see cref="Floor"/>.</summary>
            public int[] Floors;
            public string Shape;
            public long Epoch;
            /// <summary><see cref="RecurrentState.BindingChanges"/> when the step was kept.</summary>
            public long Bindings;
            /// <summary>The binding versions of the states the step binds when it was kept (<see cref="BindingStamp"/>).</summary>
            public long Stamp;
            /// <summary>The step recorded on the device (an executor that records decode steps; one sequence).</summary>
            public IDecodeCapture Capture;
            public Tensor[] CaptureOutputs;
            public int Advances;
            public bool CaptureFailed;
            /// <summary>A step over a block drafter's ring: the ring's <see cref="Slot.DraftRingOwners"/> when the step last ran.</summary>
            public int RingOwners;
            /// <summary>The storages the step reads and writes in place (each slot's caches and recurrent states), as kept.</summary>
            public object[] Bound;

            public void DisposeCapture()
            {
                if (CaptureOutputs != null) foreach (Tensor t in CaptureOutputs) t?.Dispose();
                CaptureOutputs = null;
                Capture?.Dispose();
                Capture = null;
            }
        }

        /// <summary>The kept decode step of each shape (a layer-split step keeps one a stage).</summary>
        private readonly Dictionary<string, DecodeReplay> _decodeReplays = new(StringComparer.Ordinal);
        private long _graphEpoch;   // bumped whenever cached graphs are dropped

        private void KeepDecodeReplay(GraphStep step, CachedGraph graph, StepOutputs outputs, Slot[] slots, int[] starts,
            int floor, string shape)
        {
            DropDecodeReplay(shape);
            step.OwnPositions();
            int n = slots.Length;
            var r = new DecodeReplay
            {
                Step = step, Graph = graph, Signature = step.Signature, Structure = step.StructuralSignature, Outputs = outputs, Slots = (Slot[])slots.Clone(),
                K = new Tensor[n][][], V = new Tensor[n][][], Recurrent = new RecurrentState[n][][], Capacity = new int[n], RopeDelta = new int[n],
                Starts = (int[])starts.Clone(), Floor = floor, Floors = (int[])step.AttentionFloors?.Clone(), Shape = shape, Epoch = _graphEpoch,
                Bindings = System.Threading.Interlocked.Read(ref RecurrentState.BindingChanges),
            };
            for (int s = 0; s < n; s++)
            {
                r.K[s] = slots[s].K;
                r.V[s] = slots[s].V;
                r.Recurrent[s] = slots[s].Recurrent;
                r.Capacity[s] = slots[s].Capacity;
                r.RopeDelta[s] = slots[s].RopeDelta;
            }
            r.Stamp = BindingStamp(r.Recurrent);
            r.Bound = BoundStorages(slots);
            if (n == 1 && slots[0].DraftRing) r.RingOwners = slots[0].DraftRingOwners;
            _decodeReplays[shape ?? ""] = r;
        }

        /// <summary>
        /// Whether the buffers a kept step binds changed since it was kept (<paramref name="bindings"/> and
        /// <paramref name="stamp"/> as it was kept, <paramref name="recurrent"/> its slots' states). Where recorded steps bind a
        /// slot's buffers (<see cref="SlotBoundPolicies"/>) only the states it binds count (<see cref="BindingStamp"/>): a step
        /// binds its own slots' states, compared by reference before it advances, and the freed verify buffer of CUDA error 700
        /// changed the version of the state that held it. Elsewhere any state's change counts
        /// (<see cref="RecurrentState.BindingChanges"/>), as before.
        /// </summary>
        private bool BindingsChanged(long bindings, long stamp, RecurrentState[][][] recurrent)
            => SlotBoundPolicies ? stamp != BindingStamp(recurrent) : bindings != System.Threading.Interlocked.Read(ref RecurrentState.BindingChanges);

        /// <summary>The binding versions of every state of <paramref name="recurrent"/> (a step's slots' states, [slot][layer][rank]),
        /// summed: versions only grow, so the sum changes exactly when one of them does.</summary>
        private static long BindingStamp(RecurrentState[][][] recurrent)
        {
            long stamp = 0;
            if (recurrent == null) return stamp;
            foreach (RecurrentState[][] states in recurrent)
                if (states != null)
                    foreach (RecurrentState[] layer in states)
                        if (layer != null)
                            foreach (RecurrentState state in layer)
                                if (state != null) stamp += state.BindingVersion;
            return stamp;
        }

        private void DropDecodeReplay()
        {
            foreach (DecodeReplay r in _decodeReplays.Values)
            {
                r.DisposeCapture();
                r.Step.Dispose();
            }
            _decodeReplays.Clear();
            DisposeParkedRecordings();
        }

        /// <summary>
        /// A slot ends a request (released, retained, or the primary cache reset): the steps kept to advance on its caches and
        /// states go, so the next request on the slot traces anew. A pooled slot keeps its caches and states, which every
        /// check of an advance compares, and a window recorded for one request replayed some 3,000 positions on for the
        /// next (CUDA error 700 on an A40). Their recordings park: a later trace of the same plan replays one only at its
        /// own window (<see cref="TryRunParked"/>).
        /// </summary>
        private void DropReplaysOf(Tensor[][] k, RecurrentState[][] recurrent, bool park = true)
        {
            bool Binds(Tensor[][][] ks, RecurrentState[][][] rs)
            {
                for (int s = 0; s < ks.Length; s++)
                    if ((k != null && ReferenceEquals(ks[s], k)) || (recurrent != null && ReferenceEquals(rs[s], recurrent))) return true;
                return false;
            }
            List<string> drop = null;
            foreach (var (shape, r) in _decodeReplays)
                if (Binds(r.K, r.Recurrent)) (drop ??= new List<string>()).Add(shape);
            if (drop != null)
                foreach (string shape in drop) DropDecodeReplay(shape, park);
            if (_rankedReplay != null && Binds(_rankedReplay.K, _rankedReplay.Recurrent)) DropRankedReplay();
        }

        private bool _capturesDroppedReported;

        /// <summary>Every kept step's recording goes (their pools back to the allocator); the steps stay kept, and record
        /// again on later advances.</summary>
        private void DropDecodeCaptures()
        {
            if (!_capturesDroppedReported)
            {
                _capturesDroppedReported = true;
                Console.Error.WriteLine($"[{Config.Architecture}] the device could not hold another recorded step: the recordings were dropped (their steps run unrecorded until they record again).");
            }
            foreach (DecodeReplay r in _decodeReplays.Values)
                if (r.Capture != null)
                {
                    r.DisposeCapture();
                    r.Advances = 0;
                }
            DisposeParkedRecordings();
        }

        /// <summary>Drop the kept step of <paramref name="shape"/>; its recording parks (<see cref="_parkedRecordings"/>) unless
        /// what it binds changed (<paramref name="park"/> false).</summary>
        private void DropDecodeReplay(string shape, bool park = true)
        {
            if (!_decodeReplays.Remove(shape ?? "", out DecodeReplay r)) return;
            if (park) ParkOrDisposeCapture(r);
            else DisposeRecording(r.DisposeCapture);
            r.Step.Dispose();
        }

        /// <summary>A recorded step's recording, kept after its step was dropped (its request ended, its window filled).</summary>
        private sealed class ParkedRecording
        {
            public IDecodeCapture Capture;
            public Tensor[] Outputs;
            public object[] Bound;
            public (ulong, ulong) Structure;

            public void Dispose()
            {
                if (Outputs != null) foreach (Tensor t in Outputs) t?.Dispose();
                Outputs = null;
                Capture?.Dispose();
                Capture = null;
            }
        }

        /// <summary>
        /// Recordings of dropped steps, by their plan's signature. A plan's signature hashes every state and weight leaf by
        /// identity (GraphIdentity), so a later step that traces to the same signature runs the same kernels over the same
        /// slot's caches and states: it replays the parked recording instead of running once and recording again. A pooled
        /// slot's next request, and a window a request had filled before, then cost a trace instead of a run and a recording
        /// (the direct engines' graphs outlive their requests; recording again cost a Qwen3.8 Flash Next MTP request on an
        /// A40 about 0.1 s of 2.3: four target widths recorded at its start and again past 256 positions).
        /// </summary>
        private readonly Dictionary<(ulong, ulong), ParkedRecording> _parkedRecordings = new();
        private readonly LinkedList<(ulong, ulong)> _parkedLru = new();

        /// <summary>Parked recordings at most (each keeps the pool blocks its step used). Memory pressure drops them all: a
        /// prompt step that does not fit clears the graph cache first, and a step that cannot record clears them.</summary>
        private const int ParkedRecordingsMax = 24;

        /// <summary>Steps that took over a parked recording (tests check that a request reuses them).</summary>
        internal int RecordingsAdopted { get; private set; }

        internal int ParkedRecordingCount => _parkedRecordings.Count;

        /// <summary>Park <paramref name="r"/>'s recording (one that recorded, since the last time every graph went); dispose it
        /// otherwise. A plan owns no device memory, so the recording does not need its plan cached: the graph cache's 32 plans
        /// turned over with every new slot's, and had taken their recordings with them.</summary>
        private void ParkOrDisposeCapture(DecodeReplay r)
        {
            if (r.Capture is { Captured: true } && !r.CaptureFailed && r.Epoch == _graphEpoch)
            {
                DisposeParked(r.Signature);
                _parkedRecordings[r.Signature] = new ParkedRecording
                {
                    Capture = r.Capture, Outputs = r.CaptureOutputs, Bound = r.Bound, Structure = r.Structure,
                };
                _parkedLru.AddFirst(r.Signature);
                r.Capture = null;
                r.CaptureOutputs = null;
                while (_parkedLru.Count > ParkedRecordingsMax) DisposeParked(_parkedLru.Last.Value);
            }
            r.DisposeCapture();
        }

        private void DisposeParked((ulong, ulong) signature)
        {
            if (!_parkedRecordings.Remove(signature, out ParkedRecording p)) return;
            _parkedLru.Remove(signature);
            DisposeRecording(p.Dispose);
        }

        /// <summary>Set while a slot's caches are replaced before its request's step (<see cref="ForgetGraphsOfGrowing"/>):
        /// the recordings that go are disposed later.</summary>
        private List<Action> _recordingDisposals;

        /// <summary>Dispose a dropped recording now, or later while a slot grows (<see cref="ForgetGraphsOfGrowing"/>).</summary>
        private void DisposeRecording(Action dispose)
        {
            if (_recordingDisposals != null) _recordingDisposals.Add(dispose);
            else dispose();
        }

        /// <summary>
        /// <see cref="ForgetGraphsOf"/> for a slot whose caches are replaced by larger ones before its request's step: the
        /// steps kept over the slot and the recordings bound to it go as there, but the recordings are disposed while the
        /// device runs the next recorded step (<see cref="IGraphExecutor.RunWhileNextRecordedStepRuns"/>; now where the
        /// executor runs none), not before the prompt that waits for the new caches. Nothing runs a dropped recording again;
        /// only its device memory comes back later. Disposing them first took 1.7-7.1 ms of a 4K prompt's first token
        /// (Qwen3.8-27B on an RTX PRO 6000 slice, whose pooled 1,024-cell slot held a request's decode and verify recordings).
        /// </summary>
        private void ForgetGraphsOfGrowing(Slot slot)
        {
            _recordingDisposals = new List<Action>();
            List<Action> due;
            try
            {
                ForgetGraphsOf(slot);
            }
            finally
            {
                due = _recordingDisposals;
                _recordingDisposals = null;
            }
            if (due.Count == 0) return;
            void DisposeDue()
            {
                foreach (Action dispose in due) dispose();
            }
            if (GraphExecutor?.RunWhileNextRecordedStepRuns(DisposeDue) != true) DisposeDue();
        }

        /// <summary>Parked recordings that bind any of <paramref name="storages"/> (<see cref="BoundStorages"/> pairs) go.</summary>
        private void DisposeParkedBinding(object[] storages)
        {
            if (_parkedRecordings.Count == 0 || storages.Length == 0) return;
            var gone = new HashSet<object>(ReferenceEqualityComparer.Instance);
            for (int i = 0; i < storages.Length; i += 2) gone.Add(storages[i]);
            List<(ulong, ulong)> drop = null;
            foreach (var (signature, p) in _parkedRecordings)
                for (int i = 0; i < p.Bound.Length; i += 2)
                    if (gone.Contains(p.Bound[i]))
                    {
                        (drop ??= new()).Add(signature);
                        break;
                    }
            if (drop != null) foreach (var signature in drop) DisposeParked(signature);
        }

        /// <summary>Whether cached plans bind their state leaves by object identity (GraphIdentity), which a freed tensor never
        /// has again; an executor that binds host addresses can see a freed buffer's address come back.</summary>
        private bool GraphsKeyedByIdentity => GraphExecutor != null && !GraphExecutor.Capabilities.BindsHostPointers;

        /// <summary>
        /// <paramref name="slot"/>'s caches and states are about to be freed. Plans keyed by identity never bind them again, so
        /// only the steps kept over them and the recordings that hold them go (a finished conversation's slot reclaimed from the
        /// prefix cache had dropped every plan and recording of the conversations still running); elsewhere every graph goes.
        /// </summary>
        private void ForgetGraphsOf(Slot slot)
        {
            if (!GraphsKeyedByIdentity)
            {
                ClearGraphCache();
                return;
            }
            DropReplaysOf(slot.K, slot.Recurrent, park: false);
            DisposeParkedBinding(BoundStorages(new[] { slot }));
            ForgetProgramsKeyedOtherwise();
        }

        /// <summary>
        /// Programs cached by something other than their leaves' identities: a family's (Qwen3.5's commit copies, keyed by one
        /// state's identity) and the streamed kinds' templates of steps that keep snapshots. A stream template is a step's
        /// structure only - when each value is read last (the executor's liveness), never what it binds - and a slot's caches
        /// grown, moved or freed change no step's structure; of a recurrent state's buffers only the verify buffers do (a
        /// verify window's in-place recurrence, Qwen3.8's PLE history), and only a step that keeps snapshots reads them. Every
        /// template had gone: a DFlash2 request's first verify window grows its slot's verify buffers, and every next request's
        /// prompt step then recorded and built before the device started on it (1.9 ms of a Qwen3.8-27B 62-token prompt's
        /// first token on an RTX PRO 6000 slice), and a 4K prompt's slot growth made both its chunks do so (3.2 and 3.5 ms).
        /// </summary>
        private void ForgetProgramsKeyedOtherwise()
        {
            OnGraphCacheCleared();
            ClearShortConvCommits();
            foreach (string kind in _snapshotStreamedKinds)
            {
                _streamedKinds.Remove(kind);
                _streamTemplates.Remove(kind);
            }
            _snapshotStreamedKinds.Clear();
        }

        private void DisposeParkedRecordings()
        {
            foreach (ParkedRecording p in _parkedRecordings.Values) p.Dispose();
            _parkedRecordings.Clear();
            _parkedLru.Clear();
        }

        /// <summary>
        /// A traced decode step whose plan has a parked recording over the same slot: kept as its shape's step with that
        /// recording, and run as it (no eager run, no recording). False leaves the step to the caller, nothing run.
        /// </summary>
        private bool TryRunParked(GraphStep step, CachedGraph graph, StepOutputs outputs, Slot[] slots, int[] starts, int floor, string shape)
        {
            // The shape's kept step of this plan (a draft step back a position after a rejection) parks its recording first.
            if (_decodeReplays.TryGetValue(shape ?? "", out DecodeReplay current) && current.Signature == step.Signature)
                DropDecodeReplay(shape);
            ParkedRecording p;
            if (_parkedRecordings.Remove(step.Signature, out p))
            {
                _parkedLru.Remove(step.Signature);
                if (!SameBound(p.Bound, slots))
                {
                    p.Dispose();
                    return false;
                }
            }
            else if ((p = TakeRelocated(step, slots)) == null)
                return false;
            KeepDecodeReplay(step, graph, outputs, slots, starts, floor, shape);
            DecodeReplay r = _decodeReplays[shape ?? ""];
            r.Capture = p.Capture;
            r.CaptureOutputs = p.Outputs;
            r.Advances = 1;
            RecordingsAdopted++;
            if (!TryRunRecorded(r, graph, slots, starts)) RunUnrecorded(r, graph);
            return true;
        }

        /// <summary>
        /// A parked recording of the same plan structure over another slot, moved to this step's state leaves
        /// (IDecodeCapture.TryRelocate): a request on a new slot - each one with the prefix cache keeping finished slots -
        /// had recorded every width again (~30 ms each for Qwen3.8 Flash Next on an A40), where the direct engines' graphs
        /// serve every slot. Null when none can move.
        /// </summary>
        private ParkedRecording TakeRelocated(GraphStep step, Slot[] slots)
        {
            if (slots.Length != 1) return null;
            (ulong, ulong) found = default;
            bool any = false;
            // The least recently parked first: the most recent is likelier to serve its own slot again.
            for (LinkedListNode<(ulong, ulong)> n = _parkedLru.Last; n != null; n = n.Previous)
                if (_parkedRecordings[n.Value].Structure == step.StructuralSignature)
                {
                    found = n.Value;
                    any = true;
                    break;
                }
            if (!any) return null;
            ParkedRecording p = _parkedRecordings[found];
            if (p.Capture.TryRelocate(step.StateLeaves, WrittenTensors(slots[0])))
            {
                _parkedRecordings.Remove(found);
                _parkedLru.Remove(found);
                RecordingsRelocated++;
                return p;
            }
            if (!p.Capture.Captured) DisposeParked(found);   // broken part way: it must never run
            return null;
        }

        /// <summary>Steps that took over a parked recording moved from another slot (tests check that a new slot reuses them).</summary>
        internal int RecordingsRelocated { get; private set; }

        /// <summary>Run a kept step's plan as usual (its recording declined or failed; nothing ran).</summary>
        private void RunUnrecorded(DecodeReplay r, CachedGraph graph)
        {
            if (graph.Plan is IGraphTensorOutputs keep) keep.KeepTensors = false;
            if (!r.Step.Run(graph.Plan))
                throw new InvalidOperationException($"{Config.Architecture} graph step failed: {graph.Plan.LastError}");
        }


        /// <summary>Decode steps record on executors that can (<see cref="IDecodeCaptureExecutor"/>).</summary>
        internal bool DecodeCaptureDisabled { get; set; }

        /// <summary>Decode steps recorded on the device (tests check how often a request records).</summary>
        internal int DecodeRecordings { get; private set; }

        /// <summary>Kept decode steps that ran as their device recording's replay (tests check that recordings serve steps).</summary>
        internal int RecordedDecodeReplays { get; private set; }

        private bool _slidingWindowUnrecordedReported;

        /// <summary>Kept decode steps a recorder ran unrecorded because a sliding-window layer's floor moved inside its window.</summary>
        internal int SlidingWindowUnrecordedSteps { get; private set; }

        /// <summary>Count (and report once) a decode step that runs unrecorded because a sliding-window mask's floor moves.</summary>
        private void NoteSlidingWindowUnrecorded()
        {
            SlidingWindowUnrecordedSteps++;
            if (_slidingWindowUnrecordedReported) return;
            _slidingWindowUnrecordedReported = true;
            Console.Error.WriteLine($"[{Config.Architecture}] sliding-window layers past their window: decode steps run as their " +
                "plans instead of device recordings (a recording would replay its first window's attention floor). Output is exact; " +
                "each such step pays its launches on the host.");
        }

        /// <summary>Programs traced and compiled (a graph cache miss; tests check how often a request builds one).</summary>
        internal int GraphBuilds { get; private set; }

        /// <summary>
        /// Run a kept decode step as the device's recording of it: its first run records it and every later one (each
        /// advance) replays the recording with this position's inputs and each sequence's attention length and cache write
        /// row - a batched step's sequences as well, up to the recorder's <see cref="IDecodeCaptureExecutor.MaxRecordedSequences"/>
        /// (four Qwen3.8-27B conversations decoding together had run every step unrecorded, 14% slower each than the direct
        /// engine's replayed batched graphs). False runs the step as usual; a recording that fails is tried once more on the
        /// next advance, then not again.
        /// </summary>
        private bool TryRunRecorded(DecodeReplay r, CachedGraph cached, Slot[] slots, int[] starts)
        {
            // A floored step (the draft layer after a restored checkpoint) runs as usual: some decode kernels read the
            // recorded length as the count of keys past the floor.
            if (r.CaptureFailed || DecodeCaptureDisabled || r.Floor > 0 || AnyFloor(r.Floors) || GraphExecutor is not IDecodeCaptureExecutor recorder
                || slots.Length > recorder.MaxRecordedSequences
                || cached.Plan is not IGraphTensorOutputs keep || (_expertHost != null && !GraphExecutor.Capabilities.RecordsHostExperts))
                return false;   // (a step whose host experts read the routes back mid-step cannot be recorded)
            if (r.Step.SlidingFloorInWindow)
            {
                // A recording replays its launches' scalars, and the attention kernels take the floor a mask's values give
                // them as one: past a sliding window the floor moves every step, so such a step runs as its plan does.
                NoteSlidingWindowUnrecorded();
                return false;
            }
            if (r.Capture == null) r.Advances++;
            // Too little device memory left for another recording: this step runs as usual and tries again later, and the next
            // step first gives finished conversations' retained slots back (RoomForRecordings).
            // Parked recordings give their memory back first.
            if (r.Capture == null && _parkedRecordings.Count > 0 && !recorder.HasRoomToRecord()) DisposeParkedRecordings();
            if (r.Capture == null && !recorder.HasRoomToRecord())
            {
                r.Advances = 0;
                _recordingWantsRoom = true;
                return false;
            }
            r.Capture ??= recorder.OpenDecodeCapture();
            if (r.Capture == null) { r.CaptureFailed = true; return false; }
            int n = slots.Length;
            Span<int> writeRows = stackalloc int[n], attends = stackalloc int[n];
            for (int s = 0; s < n; s++)
            {
                writeRows[s] = starts[s];
                attends[s] = starts[s] + r.Step.Counts[s];
            }
            bool bound = false;
            try
            {
                bound = r.Step.BindStable(r.Capture);
                if (!bound) { r.CaptureFailed = true; return false; }
                if (!r.Capture.Captured)
                {
                    keep.KeepTensors = true;
                    r.Capture.NoteStates(r.Step.StateLeaves);
                    // Each sequence's launches name it by its caches; the step writes every sequence's in place.
                    var sequences = new IReadOnlyList<Tensor>[n];
                    var written = new List<Tensor>();
                    for (int s = 0; s < n; s++)
                    {
                        List<Tensor> own = WrittenTensors(slots[s]);
                        sequences[s] = own;
                        written.AddRange(own);
                    }
                    bool recorded = r.Capture.CaptureAndLaunch(() => r.Step.Run(cached.Plan), attends, writeRows, sequences, written);
                    if (!recorded)
                    {
                        keep.KeepTensors = false;
                        r.DisposeCapture();
                        r.CaptureFailed = r.Advances > 1;
                        return false;
                    }
                    DecodeRecordings++;
                    r.CaptureOutputs = new Tensor[cached.Outputs.Length];
                    for (int o = 0; o < r.CaptureOutputs.Length; o++) r.CaptureOutputs[o] = keep.TakeOutput(o);
                    keep.KeepTensors = false;
                }
                else
                {
                    if (RecordedLaunchFailsForTests?.Invoke(r.Shape) == true || !r.Capture.TryLaunch(attends, writeRows))
                    {
                        // Out of device memory (a 4K request beside Qwen3.8 Flash Next's ISTA pack and MTP head on an A40): every
                        // recording gives its memory back and this step runs as usual; this one is not recorded again.
                        DropDecodeCaptures();
                        r.CaptureFailed = true;
                        return false;
                    }
                    r.Capture.MarkWritten(r.CaptureOutputs);
                    RecordedDecodeReplays++;
                }
                for (int o = 0; o < r.CaptureOutputs.Length; o++)
                {
                    Tensor t = r.CaptureOutputs[o];
                    float[] into = cached.Outputs[o];
                    if (t.ElementType == DType.Int32)
                    {
                        int count = (int)Math.Min(into.LongLength, t.ElementCount());
                        int[] ints = t.GetElementsAsInt(count);
                        for (int i = 0; i < count; i++) into[i] = ints[i];
                    }
                    else if (t.ElementCount() == into.LongLength) t.CopyToArray(into);
                    else Array.Copy(t.GetElementsAsFloat((int)Math.Min(into.LongLength, t.ElementCount())), into, (int)Math.Min(into.LongLength, t.ElementCount()));
                }
                return true;
            }
            finally
            {
                if (bound) r.Step.UnbindStable();
            }
        }

        /// <summary>The storages behind <see cref="WrittenTensors"/> of every slot, in order: a kept step binds them, so one
        /// replaced inside the same arrays (a cache or state tensor swapped for another) must retrace it.</summary>
        private static object[] BoundStorages(Slot[] slots)
        {
            var bound = new List<object>();
            foreach (Slot slot in slots)
                foreach (Tensor t in WrittenTensors(slot))
                {
                    bound.Add(t.Storage);
                    bound.Add(t.StorageOffset);
                }
            return bound.ToArray();
        }

        private static bool SameBound(object[] kept, Slot[] slots)
        {
            object[] now = BoundStorages(slots);
            if (kept == null || now.Length != kept.Length) return false;
            for (int i = 0; i < now.Length; i += 2)
                if (!ReferenceEquals(now[i], kept[i]) || !Equals(now[i + 1], kept[i + 1])) return false;
            return true;
        }

        /// <summary>The tensors a decode step of <paramref name="slot"/> writes in place: its K/V caches and recurrent states.</summary>
        private static List<Tensor> WrittenTensors(Slot slot)
        {
            var written = new List<Tensor>();
            foreach (Tensor[][] caches in new[] { slot.K, slot.V })
                if (caches != null)
                    foreach (Tensor[] layer in caches)
                        if (layer != null)
                            foreach (Tensor t in layer)
                                if (t != null) written.Add(t);
            if (slot.Recurrent != null)
                foreach (RecurrentState[] layer in slot.Recurrent)
                    if (layer != null)
                        foreach (RecurrentState state in layer)
                            if (state?.GraphTensors is { } tensors)
                                foreach (Tensor t in tensors)
                                    if (t != null) written.Add(t);
            return written;
        }

        /// <summary>
        /// Run a decode step as the previous one advanced by a position, without tracing it: the same slots (their
        /// caches, states and capacity unchanged), every sequence one position on, the same request shape and
        /// attention floor, no graph dropped since, and every decode mask still inside its window. Its inputs are
        /// rewritten in place (positions, write rows, masks, the embedded rows); the cached graph's signature is the
        /// one the trace would compute. A trace was ~2% of a decode step (Qwen3.6-35B-A3B on an M5 Pro: 0.28 of 15 ms).
        /// </summary>
        private bool TryAdvanceDecode(Slot[] slots, int[] starts, int floor, int[] floors, string shape, Tensor[] hidden,
            out CachedGraph graph, out StepOutputs outputs)
        {
            graph = null;
            outputs = default;
            DecodeReplay r = _decodeReplays.GetValueOrDefault(shape ?? "");
            if (r == null || r.Epoch != _graphEpoch || r.Floor != floor || !SameFloors(r.Floors, floors) || r.Shape != shape
                || r.Slots.Length != slots.Length)
                return false;
            // A block drafter's ring holds no recurrent state: another request's state buffers changing (a new request's first
            // verify window grows its own) leaves the steps kept over the ring, and their recordings, as they are (where states
            // carry their own binding versions the ring's stamp is 0 anyway; elsewhere the global counter would drop them).
            bool ring = Array.TrueForAll(slots, slot => slot.DraftRing);
            if (!ring && BindingsChanged(r.Bindings, r.Stamp, r.Recurrent))
            {
                DropDecodeReplay(shape, park: false);
                return false;
            }
            // Every sequence moves on by its own positions: one for decode steps, a verify window's accepted rows and one (a
            // batched verify's sequences keep different counts; held to one count, four Qwen3.8 Flash Next MTP conversations on
            // an A40 replayed their 8-row verify at 21 of 106 steps and traced the rest, 47.5 ms a step against the engine's 42).
            var deltas = new int[slots.Length];
            for (int s = 0; s < slots.Length; s++)
            {
                Slot slot = slots[s];
                deltas[s] = starts[s] - r.Starts[s];
                // What the kept graph binds: the same caches and states (a batched step may hand them in another slot
                // object), the same capacity and rotary offset.
                if ((deltas[s] < 1 && !ring) || !ReferenceEquals(slot.K, r.K[s]) || !ReferenceEquals(slot.V, r.V[s])
                    || !ReferenceEquals(slot.Recurrent, r.Recurrent[s]) || slot.Capacity != r.Capacity[s] || slot.RopeDelta != r.RopeDelta[s])
                    return false;
            }
            if (!SameBound(r.Bound, slots))
            {
                DropDecodeReplay(shape, park: false);
                return false;
            }
            // A step over the ring runs at a position behind its last one too (a new request's on the model's ring: the step,
            // its plan and its recording are the ones a trace there would make, so it moves there instead of being traced).
            bool rebase = ring && Array.Exists(deltas, d => d < 1);
            if (!_graphCache.TryGetValue(r.Signature, out CachedGraph cached) || !ReferenceEquals(cached, r.Graph)
                || !(rebase ? r.Step.TryRebase(hidden, starts) : r.Step.TryAdvance(hidden, deltas)))
            {
                DropDecodeReplay(shape);
                return false;
            }
            Array.Copy(starts, r.Starts, starts.Length);
            _graphLru.Remove(r.Signature);
            _graphLru.AddFirst(r.Signature);
            if (ring && slots.Length == 1 && r.RingOwners != slots[0].DraftRingOwners)
            {
                // The ring changed hands since the step last ran (a new request's drafts): a recording that failed for the
                // last owner (a launch out of memory, a capture failing twice) is tried again. Kept across requests, a failed
                // drafter step had run unrecorded for the rest of the process, where each request used to trace it afresh.
                r.RingOwners = slots[0].DraftRingOwners;
                if (r.CaptureFailed)
                {
                    r.CaptureFailed = false;
                    r.Advances = 0;
                }
            }
            if (!TryRunRecorded(r, cached, slots, starts))
            {
                if (cached.Plan is IGraphTensorOutputs keep) keep.KeepTensors = false;
                if (!r.Step.Run(cached.Plan))
                    throw new InvalidOperationException($"{Config.Architecture} graph step failed: {cached.Plan.LastError}");
            }
            GraphRuns++;
            GraphReplays++;
            DecodeAdvances++;
            if (r.Step.Counts.Length > 0 && r.Step.Counts[0] > 1) WindowAdvances++;
            if (shape != null && shape.StartsWith(DraftShapePrefix, StringComparison.Ordinal)) DraftAdvances++;
            if (rebase) DraftRebases++;
            foreach (Slot slot in slots) slot.DeviceDirty = true;
            graph = cached;
            outputs = r.Outputs;
            return true;
        }

        /// <summary>Whether any sequence has an attention floor (<paramref name="floors"/> null: none of its own).</summary>
        private static bool AnyFloor(int[] floors) => floors != null && Array.Exists(floors, f => f > 0);

        private static bool SameFloors(int[] a, int[] b)
            => ReferenceEquals(a, b) || (a != null && b != null && a.AsSpan().SequenceEqual(b));

        private void ClearGraphCache()
        {
            _graphEpoch++;
            DropDecodeReplay();
            OnGraphCacheCleared();
            ClearShortConvCommits();
            foreach (CachedGraph entry in _graphCache.Values) entry.Dispose();
            _graphCache.Clear();
            _graphLru.Clear();
            ClearRankGraphCache();
            _streamedKinds.Clear();
            _streamTemplates.Clear();
            _snapshotStreamedKinds.Clear();
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
        private bool _headLayoutPrepared;

        /// <summary>A step could not record for lack of device memory (<see cref="TryRunRecorded"/>).</summary>
        private bool _recordingWantsRoom;

        /// <summary>Device memory a lone CUDA step's recordings want free (with the pool's cached blocks): the executor records
        /// only above 1 GiB, and a speculating request records a step for each window and draft width.</summary>
        private const long RecordingRoomBytes = 3L << 29;

        /// <summary>
        /// After a step could not record: finished conversations' retained slots (prefix-cache payloads), the oldest first,
        /// go back until the recordings fit, as a prompt step's do (ReleaseRetainedSlotsFor), before this step traces or
        /// replays anything (releasing a slot drops the cached graphs). An unrecorded verify window of Qwen3.8-27B took 101 ms
        /// against 41 recorded on a 24 GB RTX PRO 4000, for the whole request after two 4K prompts' slots were kept.
        /// </summary>
        /// <remarks>On a device this model manages, the room is made in its fixed order (<see cref="EnsureDeviceRoom"/>) short of
        /// the parked recordings, which are what wants it: pooled slots, then finished conversations, the oldest first.</remarks>
        private void RoomForRecordings()
        {
            if (!_recordingWantsRoom) return;
            _recordingWantsRoom = false;
            if (DeviceRoomManaged)
            {
                if (_retainedSlots is { Count: > 0 } || _slotPool is { Count: > 0 })
                    EnsureDeviceRoom(RecordingRoomBytes, RunningSlots(), RoomUse.Recording, mayEvict: true, "recorded steps");
                return;
            }
            if (_allocator is TensorSharp.Cuda.CudaAllocator cuda && _retainedSlots is { Count: > 0 } && PromptFreeBytes(cuda) < RecordingRoomBytes)
                ReleaseRetainedSlotsFor(RecordingRoomBytes, cuda);
        }

        /// <summary>
        /// The untied LM head is read only by its products (never gathered by rows): the executor may keep it in a layout only
        /// they read - direct CUDA regroups a Q6_K head for its int8 tensor-core kernels (a 248,320 x 5,120 head's 8-row product
        /// 3.7 ms on the stored rows on an RTX PRO 4000). Once, on the first step (the weights are resident by then; one device).
        /// </summary>
        private void PrepareHeadLayout()
        {
            if (_headLayoutPrepared) return;
            _headLayoutPrepared = true;
            if (_mesh.IsSingleDevice && !_rankGraphs && _quantWeights.TryGetValue("output.weight", out QuantizedWeight head)
                && head != null && head.CacheKey != IntPtr.Zero)
                GraphExecutor?.KeepForProductsOnly(head.CacheKey);
        }

        private bool _drafterWeightsMarked;

        /// <summary>
        /// A block drafter's weights ("dflash.") are read only for proposals a verify checks, in the trunk's steps too: a
        /// speculating request's prompt step writes the drafter's ring (<see cref="EmitDFlashFollow"/>). The executor may then
        /// run their products as it runs a draft step's (<see cref="IGraphExecutor.MarkDrafterWeight"/>), as the inject program
        /// that wrote the ring before ran them, where the trunk's step had taken its own exact kernels for them. Once the
        /// drafter is there, on one device (where its programs run).
        /// </summary>
        private void MarkDrafterWeights()
        {
            if (_drafterWeightsMarked || !HasDFlash || !_mesh.IsSingleDevice || _rankGraphs || GraphExecutor is not { } executor) return;
            _drafterWeightsMarked = true;
            foreach (var (name, weight) in _quantWeights)
                if (name.StartsWith(DFlashConfig.WeightPrefix, StringComparison.Ordinal) && weight != null)
                    executor.MarkDrafterWeight(weight);
        }

        private bool TryRunGraph(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            bool wantLogits, out float[][] logits, int[] positionTable = null, DraftFollow follow = DraftFollow.None,
            float[] draftRows = null, DFlashFollowPlan dflash = null, bool prompt = false)
            => TryRunGraph(hidden, slots, first, counts, starts, wantLogits, out logits, null, positionTable, follow, draftRows, dflash, prompt);

        /// <summary><see cref="TryRunGraph(Tensor, Slot[], int[], int[], int[], bool, out float[][], int[], DraftFollow, float[], DFlashFollowPlan, bool)"/>
        /// that also writes each sequence's last final-normed row into <paramref name="lastHidden"/>[s] where it is set
        /// (a step driver's input); with <paramref name="wantLogits"/> false the step runs no head. <paramref name="prompt"/>: the
        /// step forwards prompt tokens (<see cref="GraphRequest.PromptChunks"/>, <see cref="MarksPromptChunks"/>).</summary>
        private bool TryRunGraph(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            bool wantLogits, out float[][] logits, float[][] lastHidden, int[] positionTable = null, DraftFollow follow = DraftFollow.None,
            float[] draftRows = null, DFlashFollowPlan dflash = null, bool prompt = false)
        {
            logits = null;
            bool greedy = wantLogits && _greedyLastRows && GraphExecutor.Capabilities.GreedyIdOutputs;
            bool wantHidden = lastHidden != null && Array.Exists(lastHidden, h => h != null);
            var request = new GraphRequest
            {
                LastRowLogits = wantLogits, GreedyIds = greedy, PositionTable = positionTable, FollowDraft = follow, DraftRows = draftRows,
                FollowDFlash = dflash, LastRowHidden = wantHidden, PromptChunks = MarksPromptChunks(prompt, counts),
            };
            if (!TryRunGraphStep(hidden, slots, first, counts, starts, request, out GraphResult result))
                return false;
            if (wantHidden)
            {
                int width = Config.HiddenSize;
                if (result.LastHidden == null || result.LastHidden.LongLength < (long)width * slots.Length)
                    throw new InvalidOperationException($"{Config.Architecture}: the step returned no final hidden rows.");
                for (int s = 0; s < slots.Length; s++)
                    if (lastHidden[s] != null) Array.Copy(result.LastHidden, (long)s * width, lastHidden[s], 0, width);
            }
            if (result.Logits != null && greedy)
            {
                // Each sequence's id as its F32 value (TryForwardUnifiedGreedy reads it).
                logits = new float[slots.Length][];
                for (int s = 0; s < slots.Length; s++) logits[s] = new[] { result.Logits[s] };
            }
            else if (result.Logits != null)
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
            PrepareHeadLayout();
            MarkDrafterWeights();
            RoomForRecordings();
            string reason = GraphUnsupportedReason();
            if (reason != null && RequiresGraphs)
                throw new NotSupportedException($"{Config.Architecture} runs only as graphs on the shared runtime, which this step cannot use ({reason}).");
            if (reason != null && _graphTpRanks != null && reason != "disabled")
                throw new NotSupportedException(
                    $"{Config.Architecture}: tensor parallelism runs as whole-forward graphs, which this step cannot use ({reason}).");
            if (reason != null && _rankGraphs)
                throw new NotSupportedException(
                    $"{Config.Architecture}: tensor parallelism runs as per-rank graphs, which this step cannot use ({reason}).");
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
            if (_rankGraphs)
                return TryRunRankedGraphStep(hidden, slots, first, counts, starts, request, out result);
            CachedGraph graph = RunTraced(rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, 0,
                step => EmitStep(step, hidden, request), out StepOutputs outputs, out bool owned, shape: request.Shape,
                advanceHidden: request.PositionTable == null ? hidden : null);
            if (graph == null) return false;
            CountDeviceRoutes(graph, outputs);
            // A replayed graph's buffers are rewritten by its next run (the model-owned buffer
            // contract); a one-shot graph's arrays are the caller's.
            result = new GraphResult
            {
                Logits = outputs.Logits >= 0 ? graph.Outputs[outputs.Logits] : null,
                Final = outputs.Final >= 0 ? graph.Outputs[outputs.Final] : null,
                Captures = outputs.Captures == null ? null : Array.ConvertAll(outputs.Captures, i => graph.Outputs[i]),
                LastHidden = outputs.LastHiddenAt > 0 ? graph.Outputs[outputs.LastHiddenAt - 1] : null,
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
                    // Each stage keeps its decode step to advance (one replay a stage shape).
                    CachedGraph graph = RunTraced(rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, 0,
                        step => EmitStep(step, stageInput, request, from, to, last), out StepOutputs outputs, out bool owned,
                        shape: $"{request.Shape}|stage{s}", advanceHidden: request.PositionTable == null ? stageInput : null);
                    if (graph == null)
                    {
                        if (s == 0) return false;
                        throw new InvalidOperationException($"{Config.Architecture}: stage {s}'s graph declined after stage 0 ran.");
                    }
                    try
                    {
                        CountDeviceRoutes(graph, outputs);
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
                            LastHidden = outputs.LastHiddenAt > 0 ? (float[])graph.Outputs[outputs.LastHiddenAt - 1].Clone() : null,
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
        /// <param name="attentionFloors">Each sequence's attention floor in place of <paramref name="attentionFloor"/>
        /// (GraphStep.AttentionFloors), or null.</param>
        private CachedGraph RunTraced(int rows, Slot[] slots, int[] first, int[] counts, int[] starts, bool replayable,
            bool snapshots, int attentionFloor, Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned,
            bool deviceOutputs = false, string shape = null, Tensor advanceHidden = null, Tensor[] advanceInputs = null,
            int[] attentionFloors = null)
        {
            // The rows from each start on are written: the checkpoints sharing them past it go first (StandardDecoderModel.RowShares.cs).
            PrepareWrites(slots, starts);
            // A drafter's steps (IGraphExecutor.Drafting): its proposals are verified, so its products may take kernels whose
            // last bits vary (the direct engines ran their drafters' Q4_K products on the int8 tensor cores).
            IGraphExecutor executor = GraphExecutor;
            bool drafting = executor.Drafting;
            int sequences = executor.StepSequences;
            executor.Drafting = shape != null && shape.StartsWith(DraftShapePrefix, StringComparison.Ordinal);
            executor.StepSequences = counts.Length;
            try
            {
                return RunTracedAs(rows, slots, first, counts, starts, replayable, snapshots, attentionFloor, emit, out outputs, out owned,
                    deviceOutputs, shape, advanceHidden, advanceInputs, attentionFloors);
            }
            finally
            {
                executor.Drafting = drafting;
                executor.StepSequences = sequences;
            }
        }

        private CachedGraph RunTracedAs(int rows, Slot[] slots, int[] first, int[] counts, int[] starts, bool replayable,
            bool snapshots, int attentionFloor, Func<GraphStep, StepOutputs> emit, out StepOutputs outputs, out bool owned,
            bool deviceOutputs, string shape, Tensor advanceHidden, Tensor[] advanceInputs, int[] attentionFloors)
        {
            GraphStep step = null;
            CachedGraph graph = null;
            owned = false;
            outputs = default;
            // A decode step, or a verify window of up to 8 rows a sequence (its recurrent snapshots indexed by row, not by
            // position), runs as the kept step of its width advanced to its positions; each width keeps its own.
            Tensor[] advancing = advanceInputs ?? (advanceHidden != null ? new[] { advanceHidden } : null);
            bool advance = replayable && advancing != null && !deviceOutputs && DecodeStepsAdvance
                && !DecodeAdvanceDisabled && GraphExecutor.Capabilities.Replays && counts.Length > 0 && counts[0] >= 1
                && counts[0] <= MaxAdvancedWindowRows && Array.TrueForAll(counts, c => c == counts[0]) && (counts[0] > 1 || !snapshots);
            string replayShape = advance && (counts[0] > 1 || snapshots) ? $"{shape}|w{counts[0]}|{(snapshots ? "s" : "")}" : shape;
            if (advance && TryAdvanceDecode(slots, starts, attentionFloor, attentionFloors, replayShape, advancing, out graph, out outputs))
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
                    string floored = attentionFloors == null ? $"{attentionFloor > 0}" : string.Concat(Array.ConvertAll(attentionFloors, f => f > 0 ? '1' : '0'));
                    kind = $"{rows}|{slots.Length}|{snapshots}|{floored}|{emit.Method.MetadataToken}|{string.Join(",", counts)}|{shape}|{EmissionVariant(counts, starts)}" +
                        RingWrapVariant(counts, starts);
                    // A decode step that could advance runs as the recorded program instead, so it is kept for the next
                    // step to advance (a streamed step keeps nothing: decode would re-emit every node of every step).
                    if (_streamedKinds.Contains(kind) && !advance)
                    {
                        IGraphStream stream = streaming.OpenStream(_streamTemplates.GetValueOrDefault(kind));
                        if (deviceOutputs && stream is IGraphTensorOutputs keepStream) keepStream.KeepTensors = true;
                        graph = new CachedGraph { Plan = stream };
                        owned = true;
                        step = new GraphStep(this, rows, slots, first, counts, starts, false, snapshots, record: false)
                            { AttentionFloor = attentionFloor, AttentionFloors = attentionFloors, Sink = stream };
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
                    step = new GraphStep(this, rows, slots, first, counts, starts, replayable, snapshots, record: false)
                        { AttentionFloor = attentionFloor, AttentionFloors = attentionFloors };
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
                    step = new GraphStep(this, rows, slots, first, counts, starts, replayable, snapshots, record: true)
                        { AttentionFloor = attentionFloor, AttentionFloors = attentionFloors };
                    outputs = emit(step);
                    graph = step.Build(replayable, reorder: replayable && GraphConcurrencyReorder);
                    GraphBuilds++;
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

                // A decode step whose plan an earlier request recorded over this slot replays that recording.
                if (advance && !owned && step.Advanceable && (_parkedRecordings.Count > 0 || _decodeReplays.ContainsKey(replayShape ?? ""))
                    && TryRunParked(step, graph, outputs, slots, starts, attentionFloor, replayShape))
                {
                    step = null;   // the kept step owns it now
                    GraphRuns++;
                    foreach (Slot slot in slots) slot.DeviceDirty = true;
                    return graph;
                }

                if (graph.Plan is IGraphTensorOutputs keep) keep.KeepTensors = deviceOutputs;
                // A decode step that will advance records on its first run: running it unrecorded first cost every width a
                // request starts and every 256 positions one more step (~1.5% of a Qwen3.8 Flash Next MTP reply on an A40).
                if (advance && !owned && step.Advanceable && kind == null && GraphExecutor is IDecodeCaptureExecutor)
                {
                    KeepDecodeReplay(step, graph, outputs, slots, starts, attentionFloor, replayShape);
                    DecodeReplay kept = _decodeReplays[replayShape ?? ""];
                    step = null;   // the kept step owns it now
                    if (!TryRunRecorded(kept, graph, slots, starts)) RunUnrecorded(kept, graph);
                    GraphRuns++;
                    foreach (Slot slot in slots) slot.DeviceDirty = true;
                    return graph;
                }
                if (!step.Run(graph.Plan))
                    throw new InvalidOperationException($"{Config.Architecture} graph step failed: {graph.Plan.LastError}");
                GraphRuns++;
                if (kind != null && _streamedKinds.Add(kind))
                {
                    _streamTemplates[kind] = step.Program;
                    if (snapshots) _snapshotStreamedKinds.Add(kind);
                }
                foreach (Slot slot in slots) slot.DeviceDirty = true;
                if (advance && !owned && step.Advanceable)
                {
                    KeepDecodeReplay(step, graph, outputs, slots, starts, attentionFloor, replayShape);
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
            /// <summary>The output of the sequences' last final-normed rows, plus one (0: none), so every construction that
            /// does not set it means none.</summary>
            public int LastHiddenAt;
            public int[] RankLogits;   // each rank's slice of the vocabulary (a vocabulary-split head), in rank order
            public int[] Captures;   // -1: the captured layer is another stage's
            public (int Output, int Layer)[] Routes;   // resident MoE layers' top-K ids while device routes are recorded
            public int[] Draft;   // a block draft's results: its ids' outputs, then its confidences' (see EmitDFlashBlock)
        }

        /// <summary>Trace one step: every layer, then what the request asks for.</summary>
        private StepOutputs EmitStep(GraphStep step, Tensor hidden, GraphRequest request)
            => EmitStep(step, hidden, request, 0, Config.NumLayers, head: true);

        /// <summary>Trace layers [<paramref name="from"/>, <paramref name="to"/>) (one pipeline stage), then
        /// the head when <paramref name="head"/>, else the stage's output rows for the next stage.</summary>
        private StepOutputs EmitStep(GraphStep step, Tensor hidden, GraphRequest request, int from, int to, bool head)
        {
            step.PromptChunks = request.PromptChunks;
            int n = step.Sequences, rows = step.Rows, hiddenSize = Config.HiddenSize, width = StreamWidth;
            var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            // The first stage widens the embedded rows into the residual stream; later stages receive it.
            int embedded = from == 0 ? step.HiddenInput(hidden, hiddenSize, rows) : -1;
            int h = from == 0 ? EmitStreamInput(step, embedded) : step.HiddenInput(hidden, width, rows);
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
            // The DFlash rings' follow taps the drafter's target layers' rows and writes the rings once it has them all.
            DFlashFollowPlan dflash = request.FollowDFlash;
            int[] targets = null, taps = null;
            if (dflash != null)
            {
                if (from != 0 || to != Config.NumLayers || !head)
                    throw new InvalidOperationException("The drafter's rings follow a step over the whole stack.");
                targets = DFlashSettings.TargetLayerIds;
                taps = new int[targets.Length];
                Array.Fill(taps, -1);
            }
            bool prune = head && PrunesLastLayer(step, request, from, to);
            if (prune) LastLayerPrunedSteps++;
            for (int layer = from; layer < to; layer++)
            {
                for (int c = 0; c < request.CaptureLayers.Length; c++)
                    if (request.CaptureLayers[c] == layer)
                        outputs.Captures[c] = EmitCaptureOutput(step, h, request);
                if (TapDFlashFollow(step, h, layer, targets, taps))
                    EmitDFlashFollow(step, taps, dflash);
                h = EmitLayerPrologue(step, layer, h);
                int normed = EmitBlockInput(step, layer, h, BlockSite.Mixer, out int mixContext);
                int mix = _recurrentLayer[layer] ? EmitRecurrentMixer(step, layer, normed) : EmitAttention(step, layer, normed);
                h = EmitBlockOutput(step, layer, h, BlockSite.Mixer, mixContext, mix);
                // The last layer's feed-forward over the rows the head reads only (PrunesLastLayerRows).
                if (prune && layer == Config.NumLayers - 1) h = EmitLastRows(step, h, width);
                int ffnIn = EmitBlockInput(step, layer, h, BlockSite.Ffn, out int ffnContext);
                int ffn = _moeLayer[layer] ? EmitMoe(step, layer, ffnIn) : EmitDenseFfn(step, layer, ffnIn);
                h = EmitBlockOutput(step, layer, h, BlockSite.Ffn, ffnContext, ffn);
                step.Boundary(h);
            }
            outputs.Routes = step.Routes;
            if (!head)
            {
                outputs.Hidden = step.Output(h, (long)width * rows);
                return outputs;
            }
            // A target past the stack taps the stream leaving it.
            if (TapDFlashFollow(step, h, Config.NumLayers, targets, taps))
                EmitDFlashFollow(step, taps, dflash);
            EmitHead(step, h, request, ref outputs, embedded, lastRowsOnly: prune);
            return outputs;
        }

        /// <summary>Whether this step runs its last layer's feed-forward on each sequence's last row only
        /// (<see cref="PrunesLastLayerRows"/>): the family opts in, the step holds the last layer and more rows than sequences, the
        /// layer's feed-forward is dense, and the head reads nothing but those rows.</summary>
        private bool PrunesLastLayer(GraphStep step, GraphRequest request, int from, int to)
        {
            int last = Config.NumLayers - 1;
            return PrunesLastLayerRows && !LastLayerPruningDisabledForTests && from <= last && to == Config.NumLayers && step.Rows > step.Sequences && !_moeLayer[last]
                && (request.LastRowLogits || request.LastRowHidden) && !request.AllRowLogits && !request.CaptureFinal
                && request.FollowDraft == DraftFollow.None && request.FollowDFlash == null
                && !Array.Exists(request.CaptureLayers, c => c >= Config.NumLayers);
        }

        /// <summary>Tests: every step runs its last layer over every row (<see cref="PrunesLastLayerRows"/> off).</summary>
        internal bool LastLayerPruningDisabledForTests { get; set; }

        /// <summary>Steps traced with their last layer's feed-forward over the sequences' last rows only (tests).</summary>
        internal int LastLayerPrunedSteps { get; private set; }

        /// <summary>Each sequence's last row of the stream <paramref name="x"/> [<paramref name="width"/>, rows], in sequence
        /// order: [width, sequences].</summary>
        private static int EmitLastRows(GraphStep step, int x, int width)
        {
            int n = step.Sequences, rows = step.Rows;
            if (n == 1) return step.View2(x, width, 1, width * 4L, (rows - 1) * (long)width * 4);
            var last = new int[n];
            for (int s = 0; s < n; s++) last[s] = step.First[s] + step.Counts[s] - 1;
            return step.Op(GraphOp.GetRows, x, step.ConstantInputI32(last, n));
        }

        /// <summary>Tap the stream <paramref name="h"/> entering <paramref name="layer"/> (the stack's end at NumLayers) for
        /// the DFlash rings' follow where the drafter reads it; true when this tap completes the drafter's rows.</summary>
        private bool TapDFlashFollow(GraphStep step, int h, int layer, int[] targets, int[] taps)
        {
            if (taps == null) return false;
            bool tapped = false;
            int row = -1;
            for (int t = 0; t < targets.Length; t++)
            {
                if (taps[t] >= 0 || (layer < Config.NumLayers ? targets[t] != layer : targets[t] < Config.NumLayers)) continue;
                taps[t] = row = row >= 0 ? row : EmitCaptureRow(step, h);
                tapped = true;
            }
            return tapped && Array.TrueForAll(taps, x => x >= 0);
        }

        /// <summary>A profiling load's resident MoE layers: their recorded top-K ids go to the routing counters.</summary>
        private void CountDeviceRoutes(CachedGraph graph, in StepOutputs outputs)
        {
            if (outputs.Routes == null) return;
            foreach ((int output, int layer) in outputs.Routes)
                if (TryGetLayerExperts(layer, out var gate, out _, out _))
                    GgmlBasicOps.MoeRoutingCount(gate.Data, _numExperts, System.Runtime.InteropServices.MemoryMarshal.Cast<float, int>(graph.Outputs[output]));
        }

        /// <summary>The head over the stream <paramref name="h"/> leaving the last layer: the captures past it, the final
        /// rows and the logits the request asks for. <paramref name="lastRowsOnly"/>: <paramref name="h"/> holds each sequence's
        /// last row only, in sequence order (<see cref="PrunesLastLayer"/>).</summary>
        private void EmitHead(GraphStep step, int h, GraphRequest request, ref StepOutputs outputs, int embedded = -1, bool lastRowsOnly = false)
        {
            int n = step.Sequences, rows = step.Rows, hiddenSize = Config.HiddenSize, width = StreamWidth;
            for (int c = 0; c < request.CaptureLayers.Length; c++)
                if (request.CaptureLayers[c] >= Config.NumLayers)
                    outputs.Captures[c] = EmitCaptureOutput(step, h, request);
            int final = -1;
            if (request.FollowDraft != DraftFollow.None)
            {
                // The draft layer reads the rows the speculator's features are: the stream, or the final-normed rows.
                if (!DraftChainsStream) final = EmitStreamOutput(step, h);
                EmitDraftFollow(step, DraftChainsStream ? h : final, embedded, request);
            }
            if (!request.LastRowLogits && !request.AllRowLogits && !request.CaptureFinal && !request.LastRowHidden) return;

            if (final < 0) final = EmitStreamOutput(step, h);
            if (request.CaptureFinal)
                outputs.Final = DraftChainsStream ? step.Output(h, (long)width * rows) : step.Output(final, (long)hiddenSize * rows);
            string headName = _quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")
                ? "output.weight" : "token_embd.weight";
            if (request.AllRowLogits)
            {
                int logits = step.MatMul(headName, final);
                outputs.Logits = request.GreedyIds
                    ? step.Output(step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { 1 })), rows)
                    : step.Output(logits, (long)Config.VocabSize * rows);
            }
            else if (request.LastRowLogits || request.LastRowHidden)
            {
                int selected;
                if (lastRowsOnly || (n == 1 && rows == 1)) selected = final;
                else if (n == 1) selected = step.View2(final, hiddenSize, 1, hiddenSize * 4L, (rows - 1) * hiddenSize * 4L);
                else
                {
                    var last = new int[n];
                    for (int s = 0; s < n; s++) last[s] = step.First[s] + step.Counts[s] - 1;
                    selected = step.Op(GraphOp.GetRows, final, step.ConstantInputI32(last, n));
                }
                if (request.LastRowHidden)
                    outputs.LastHiddenAt = 1 + step.Output(n == 1 && rows > 1 && !lastRowsOnly ? step.Op(GraphOp.Cont, selected) : selected,
                        (long)hiddenSize * n);
                // A step that wants only the rows (every sequence's next slot is made from its hidden row) runs no head.
                if (!request.LastRowLogits) return;
                int logits = step.MatMul(headName, selected);
                outputs.Logits = request.GreedyIds
                    ? step.Output(step.Op(GraphOp.Cont, step.Op(GraphOp.TopK, stackalloc[] { logits }, stackalloc long[] { 1 })), n)
                    : step.Output(logits, (long)Config.VocabSize * n);
            }
        }

        /// <summary>
        /// One step's layers over a tensor-parallel group, as each rank's program (see <see cref="TryRunRankedGraphStep"/>):
        /// a block's partial joins the rank's copy of the residual stream once <see cref="GraphOp.AllReduce"/> summed it,
        /// and rank 0 runs the head. The ranks' nodes are emitted block by block, so every rank's reduce follows every
        /// rank's partial.
        /// </summary>
        private StepOutputs EmitRankedStep(GraphStep[] steps, Tensor[] hidden, GraphRequest request)
        {
            GraphStep lead = steps[0];
            int n = lead.Sequences, rows = lead.Rows, hiddenSize = Config.HiddenSize, ranks = steps.Length;
            var outputs = new StepOutputs { Logits = -1, Final = -1, Hidden = -1 };
            var positions = new int[rows];
            for (int s = 0; s < n; s++)
                for (int i = 0; i < lead.Counts[s]; i++) positions[lead.First[s] + i] = lead.Starts[s] + i + lead.Slots[s].RopeDelta;
            var h = new int[ranks];
            for (int r = 0; r < ranks; r++)
            {
                steps[r].PromptChunks = request.PromptChunks;
                steps[r].SetPositions(r == 0 ? positions : (int[])positions.Clone());   // each rank's own: an advance bumps it
                if (request.PositionTable != null)
                {
                    if (n != 1) throw new InvalidOperationException("A rotary position table drives one sequence's rows.");
                    steps[r].SetPositionTable(AxisMajorPositions(request.PositionTable, rows));
                }
                h[r] = EmitStreamInput(steps[r], steps[r].HiddenInput(hidden[r], hiddenSize, rows));
            }
            if (request.CaptureLayers.Length > 0)
            {
                outputs.Captures = new int[request.CaptureLayers.Length];
                Array.Fill(outputs.Captures, -1);
            }
            var block = new int[ranks];
            var context = new int[ranks];
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                for (int c = 0; c < request.CaptureLayers.Length; c++)
                    if (request.CaptureLayers[c] == layer)
                        outputs.Captures[c] = EmitCaptureOutput(lead, h[0], request);
                for (int r = 0; r < ranks; r++)
                {
                    h[r] = EmitLayerPrologue(steps[r], layer, h[r]);
                    int normed = EmitBlockInput(steps[r], layer, h[r], BlockSite.Mixer, out context[r]);
                    block[r] = _recurrentLayer[layer] ? EmitRecurrentMixer(steps[r], layer, normed) : EmitAttention(steps[r], layer, normed);
                }
                for (int r = 0; r < ranks; r++) block[r] = steps[r].Op(GraphOp.AllReduce, block[r]);
                for (int r = 0; r < ranks; r++)
                {
                    h[r] = EmitBlockOutput(steps[r], layer, h[r], BlockSite.Mixer, context[r], block[r]);
                    int ffnIn = EmitBlockInput(steps[r], layer, h[r], BlockSite.Ffn, out context[r]);
                    block[r] = _moeLayer[layer] ? EmitMoe(steps[r], layer, ffnIn) : EmitDenseFfn(steps[r], layer, ffnIn);
                }
                for (int r = 0; r < ranks; r++) block[r] = steps[r].Op(GraphOp.AllReduce, block[r]);
                for (int r = 0; r < ranks; r++)
                {
                    h[r] = EmitBlockOutput(steps[r], layer, h[r], BlockSite.Ffn, context[r], block[r]);
                    steps[r].Boundary(h[r]);
                }
            }
            if (_vocabHeadRows != null && request.LastRowLogits && !request.AllRowLogits)
                EmitVocabSplitHead(steps, h, request, ref outputs);
            else
                EmitHead(lead, h[0], request, ref outputs);
            return outputs;
        }

        /// <summary><see cref="EmitHead"/> over the ranks: each rank normalizes its copy of the stream and scores its
        /// slice of the vocabulary for the sequences' last rows (<see cref="ShardHeadByVocab"/>).</summary>
        private void EmitVocabSplitHead(GraphStep[] steps, int[] h, GraphRequest request, ref StepOutputs outputs)
        {
            GraphStep lead = steps[0];
            int n = lead.Sequences, rows = lead.Rows, hiddenSize = Config.HiddenSize, width = StreamWidth;
            for (int c = 0; c < request.CaptureLayers.Length; c++)
                if (request.CaptureLayers[c] >= Config.NumLayers)
                    outputs.Captures[c] = EmitCaptureOutput(lead, h[0], request);
            var last = new int[n];
            for (int s = 0; s < n; s++) last[s] = lead.First[s] + lead.Counts[s] - 1;
            outputs.RankLogits = new int[steps.Length];
            for (int r = 0; r < steps.Length; r++)
            {
                GraphStep step = steps[r];
                int final = EmitStreamOutput(step, h[r]);
                if (r == 0 && request.CaptureFinal)
                    outputs.Final = DraftChainsStream ? step.Output(h[0], (long)width * rows) : step.Output(final, (long)hiddenSize * rows);
                int selected = n == 1 && rows == 1 ? final
                    : n == 1 ? step.View2(final, hiddenSize, 1, hiddenSize * 4L, (rows - 1) * hiddenSize * 4L)
                    : step.Op(GraphOp.GetRows, final, step.ConstantInputI32((int[])last.Clone(), n));
                if (r == 0 && request.LastRowHidden)
                    outputs.LastHiddenAt = 1 + step.Output(n == 1 && rows > 1 ? step.Op(GraphOp.Cont, selected) : selected, (long)hiddenSize * n);
                outputs.RankLogits[r] = step.Output(step.MatMul(VocabHeadName, selected), (long)_vocabHeadRows[r] * n);
            }
        }

        /// <summary>The ranks' vocabulary slices joined into [vocab, n] logits (a buffer of the model's, as a plan's
        /// outputs are: the caller copies what it keeps).</summary>
        private float[] JoinVocabSlices(int[] rankLogits, int n, Func<int, int, float[]> output)
        {
            int vocab = Config.VocabSize;
            long size = (long)vocab * n;
            if (_vocabLogits == null || _vocabLogits.LongLength != size) _vocabLogits = new float[size];
            int offset = 0;
            for (int r = 0; r < rankLogits.Length; r++)
            {
                float[] part = output(r, rankLogits[r]);
                int rows = _vocabHeadRows[r];
                for (int s = 0; s < n; s++) Array.Copy(part, (long)s * rows, _vocabLogits, (long)s * vocab + offset, rows);
                offset += rows;
            }
            return _vocabLogits;
        }

        private float[] _vocabLogits;

        /// <summary>Each kind's programs of the ranks (recorded once, before the kind's first step streams), whose readers
        /// let each rank's stream free its tensors.</summary>
        private readonly Dictionary<string, GraphProgram[]> _rankTemplates = new();

        /// <summary>
        /// A step over a tensor-parallel group as per-rank graphs (<see cref="_rankGraphs"/>): every rank's program is
        /// traced on one thread, block by block across the ranks, into its executor's stream, so the ops run as they are
        /// emitted and the devices work at once; a block's all-reduce runs when the last rank hands its partial in. A kind
        /// that has not run records the ranks' programs first (nothing runs), for the streams to free tensors by.
        /// </summary>
        private bool TryRunRankedGraphStep(Tensor hidden, Slot[] slots, int[] first, int[] counts, int[] starts,
            GraphRequest request, out GraphResult result)
        {
            if (ExecutorFor(_mesh.Allocator(0, 0)) is IRankedGraphExecutor compiled)
                return TryRunRankedCompiledStep(compiled, hidden, slots, first, counts, starts, request, out result);
            result = null;
            int ranks = _mesh.TensorParallel, rows = (int)hidden.Sizes[0];
            string kind = $"{rows}|{slots.Length}|{request.RecurrentSnapshots}|{string.Join(",", counts)}|{request.Shape}|" +
                          $"{request.LastRowLogits}|{request.AllRowLogits}|{request.CaptureFinal}|{string.Join(",", request.CaptureLayers)}|" +
                          $"{request.PositionTable != null}|{EmissionVariant(counts, starts)}" + RingWrapVariant(counts, starts);
            Tensor[] inputs = RankInputs(hidden);
            var steps = new GraphStep[ranks];
            var streams = new IGraphStream[ranks];
            try
            {
                if (!_rankTemplates.TryGetValue(kind, out GraphProgram[] templates))
                {
                    var recorded = new GraphStep[ranks];
                    try
                    {
                        for (int r = 0; r < ranks; r++)
                            recorded[r] = new GraphStep(this, rows, slots, first, counts, starts, false, request.RecurrentSnapshots, record: true) { Rank = r };
                        EmitRankedStep(recorded, inputs, request);
                        templates = Array.ConvertAll(recorded, step => step.Record());
                    }
                    finally
                    {
                        foreach (GraphStep step in recorded) step?.Dispose();
                    }
                    _rankTemplates[kind] = templates;
                }
                for (int r = 0; r < ranks; r++)
                {
                    streams[r] = ((IStreamingGraphExecutor)ExecutorFor(_mesh.Allocator(0, r))).OpenStream(templates[r]);
                    steps[r] = new GraphStep(this, rows, slots, first, counts, starts, false, request.RecurrentSnapshots, record: false)
                        { Rank = r, Sink = streams[r] };
                }
                StepOutputs outputs = default;
                bool ok = streams[0].Execute(() =>
                {
                    outputs = EmitRankedStep(steps, inputs, request);
                    bool all = true;
                    for (int r = 0; r < ranks; r++) all &= steps[r].Run(streams[r]);
                    return all;
                });
                if (!ok)
                {
                    string error = Array.Find(streams, st => st.LastError != null)?.LastError;
                    throw new InvalidOperationException($"{Config.Architecture} tensor-parallel graph step failed: {error}");
                }
                GraphRuns++;
                foreach (Slot slot in slots) slot.DeviceDirty = true;
                IGraphPlan lead = streams[0];
                result = new GraphResult
                {
                    Logits = outputs.Logits >= 0 ? lead.Outputs[outputs.Logits]
                        : outputs.RankLogits != null ? JoinVocabSlices(outputs.RankLogits, slots.Length, (r, o) => ((IGraphPlan)streams[r]).Outputs[o]) : null,
                    Final = outputs.Final >= 0 ? lead.Outputs[outputs.Final] : null,
                    Captures = outputs.Captures == null ? null : Array.ConvertAll(outputs.Captures, i => i >= 0 ? lead.Outputs[i] : null),
                    LastHidden = outputs.LastHiddenAt > 0 ? lead.Outputs[outputs.LastHiddenAt - 1] : null,
                };
                return true;
            }
            finally
            {
                foreach (GraphStep step in steps) step?.Dispose();
                foreach (IGraphStream stream in streams) stream?.Dispose();
                for (int r = 0; r < inputs.Length; r++)
                    if (!ReferenceEquals(inputs[r], hidden)) inputs[r].Dispose();
            }
        }

        /// <summary>Whether every tensor-parallel partial must be summed exactly in F32, prompt chunks' too (a fast transport may
        /// carry wide sums in BF16). Default: decode and verify steps only.</summary>
        protected virtual bool ExactTensorParallelSums => false;

        /// <summary>Compiled per-rank plans by the ranks' combined signature (decode kinds replay), most recent first.</summary>
        private readonly Dictionary<(ulong, ulong), CachedGraph[]> _rankGraphCache = new();
        private readonly LinkedList<(ulong, ulong)> _rankGraphLru = new();

        /// <summary>The last per-rank decode step, kept to run again one position on (<see cref="DecodeReplay"/>'s ranked form).</summary>
        private sealed class RankedReplay
        {
            public GraphStep[] Steps;
            public CachedGraph[] Plans;
            public (ulong, ulong) Signature;
            public StepOutputs Outputs;
            public Slot[] Slots;
            public Tensor[][][] K;
            public RecurrentState[][][] Recurrent;
            public int[] Capacity, RopeDelta, Starts;
            public string Shape;
            public long Epoch;
            public long Bindings, Stamp;
        }

        private RankedReplay _rankedReplay;

        private void KeepRankedReplay(GraphStep[] steps, CachedGraph[] plans, StepOutputs outputs, Slot[] slots, int[] starts, string shape)
        {
            DropRankedReplay();
            foreach (GraphStep step in steps) step.OwnPositions();   // each rank's step moves its own positions on
            int n = slots.Length;
            var r = new RankedReplay
            {
                Steps = (GraphStep[])steps.Clone(), Plans = plans, Signature = RankedSignature(steps), Outputs = outputs,
                Slots = (Slot[])slots.Clone(), K = new Tensor[n][][], Recurrent = new RecurrentState[n][][], Capacity = new int[n],
                RopeDelta = new int[n], Starts = (int[])starts.Clone(), Shape = shape, Epoch = _graphEpoch,
                Bindings = System.Threading.Interlocked.Read(ref RecurrentState.BindingChanges),
            };
            for (int s = 0; s < n; s++)
            {
                r.K[s] = slots[s].K;
                r.Recurrent[s] = slots[s].Recurrent;
                r.Capacity[s] = slots[s].Capacity;
                r.RopeDelta[s] = slots[s].RopeDelta;
            }
            r.Stamp = BindingStamp(r.Recurrent);
            _rankedReplay = r;
        }

        private void DropRankedReplay()
        {
            if (_rankedReplay == null) return;
            foreach (GraphStep step in _rankedReplay.Steps) step?.Dispose();
            _rankedReplay = null;
        }

        /// <summary>The kept per-rank decode step advanced one position with <paramref name="inputs"/> as the ranks' rows: the
        /// same slots (caches, states, capacity), every sequence one position on, the same shape, no graph dropped since,
        /// and every rank's masks still inside their window (the ranks' steps agree on that, so all advance or none).</summary>
        private bool TryAdvanceRanked(Slot[] slots, int[] starts, string shape, Tensor[] inputs, out CachedGraph[] plans, out StepOutputs outputs)
        {
            plans = null;
            outputs = default;
            RankedReplay r = _rankedReplay;
            if (r == null || r.Epoch != _graphEpoch || r.Shape != shape || r.Slots.Length != slots.Length
                || BindingsChanged(r.Bindings, r.Stamp, r.Recurrent)) return false;
            for (int s = 0; s < slots.Length; s++)
            {
                Slot slot = slots[s];
                if (!ReferenceEquals(slot, r.Slots[s]) || !ReferenceEquals(slot.K, r.K[s]) || !ReferenceEquals(slot.Recurrent, r.Recurrent[s])
                    || slot.Capacity != r.Capacity[s] || slot.RopeDelta != r.RopeDelta[s] || starts[s] != r.Starts[s] + 1)
                    return false;
            }
            if (!_rankGraphCache.TryGetValue(r.Signature, out CachedGraph[] cached) || !ReferenceEquals(cached, r.Plans)
                || !r.Steps[0].TryAdvance(new[] { inputs[0] }))
            {
                DropRankedReplay();
                return false;
            }
            for (int rank = 1; rank < r.Steps.Length; rank++)
                if (!r.Steps[rank].TryAdvance(new[] { inputs[rank] }))
                    throw new InvalidOperationException($"{Config.Architecture}: tensor-parallel rank {rank} could not advance where rank 0 did.");
            Array.Copy(starts, r.Starts, starts.Length);
            _rankGraphLru.Remove(r.Signature);
            _rankGraphLru.AddFirst(r.Signature);
            plans = cached;
            outputs = r.Outputs;
            return true;
        }

        private void ClearRankGraphCache()
        {
            DropRankedReplay();
            foreach (CachedGraph[] plans in _rankGraphCache.Values)
                foreach (CachedGraph plan in plans) plan.Dispose();
            _rankGraphCache.Clear();
            _rankGraphLru.Clear();
        }

        private static (ulong, ulong) RankedSignature(GraphStep[] steps)
        {
            ulong a = 0xcbf29ce484222325UL, b = (ulong)steps.Length;
            foreach (GraphStep step in steps)
            {
                (ulong x, ulong y) = step.Signature;
                a = (a ^ x) * 0x100000001b3UL;
                b = (b ^ y) * 0x9e3779b97f4a7c15UL + 1;
            }
            return (a, b);
        }

        /// <summary>
        /// A step over a tensor-parallel group on an executor that compiles whole programs (GGML): every rank's program is
        /// traced block by block across the ranks, compiled for the rank's device, and the ranks' plans run together,
        /// pausing at each <see cref="GraphOp.AllReduce"/> while the devices sum the partial (IRankedGraphExecutor). Decode
        /// kinds replay: a later step only re-traces to bind its inputs and finds the plans by the ranks' signatures.
        /// </summary>
        private bool TryRunRankedCompiledStep(IRankedGraphExecutor executor, Tensor hidden, Slot[] slots, int[] first, int[] counts,
            int[] starts, GraphRequest request, out GraphResult result)
        {
            result = null;
            int ranks = _mesh.TensorParallel, rows = (int)hidden.Sizes[0];
            bool replayable = executor.Capabilities.Replays;
            foreach (int c in counts) replayable &= c <= SmallChunkRows;
            // A decode step one position on from the last runs as it advanced, without a trace (TryAdvanceDecode's rule).
            bool advance = replayable && DecodeStepsAdvance && !DecodeAdvanceDisabled && request.PositionTable == null
                && !request.RecurrentSnapshots && Array.TrueForAll(counts, c => c == 1);
            Tensor[] inputs = RankInputs(hidden);
            var steps = new GraphStep[ranks];
            CachedGraph[] plans = null;
            bool owned = false, replayed = false;
            StepOutputs outputs = default;
            try
            {
                if (advance && TryAdvanceRanked(slots, starts, request.Shape, inputs, out plans, out outputs))
                {
                    Array.Copy(_rankedReplay.Steps, steps, ranks);
                    replayed = true;
                    DecodeAdvances++;
                    GraphReplays++;
                }
                else if (replayable && _rankGraphCache.Count > 0)
                {
                    for (int r = 0; r < ranks; r++)
                        steps[r] = new GraphStep(this, rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, record: false) { Rank = r };
                    outputs = EmitRankedStep(steps, inputs, request);
                    var key = RankedSignature(steps);
                    if (_rankGraphCache.TryGetValue(key, out plans))
                    {
                        _rankGraphLru.Remove(key);
                        _rankGraphLru.AddFirst(key);
                        GraphReplays++;
                    }
                    else
                    {
                        foreach (GraphStep step in steps) step.Dispose();
                        Array.Clear(steps);
                    }
                }
                if (plans == null)
                {
                    for (int r = 0; r < ranks; r++)
                        steps[r] = new GraphStep(this, rows, slots, first, counts, starts, replayable, request.RecurrentSnapshots, record: true) { Rank = r };
                    outputs = EmitRankedStep(steps, inputs, request);
                    plans = new CachedGraph[ranks];
                    try
                    {
                        for (int r = 0; r < ranks; r++)
                            plans[r] = steps[r].BuildForRank(executor, _mesh.GlobalDevice(0, r), replayable);
                    }
                    catch
                    {
                        foreach (CachedGraph plan in plans) plan?.Dispose();
                        throw;
                    }
                    if (replayable)
                    {
                        var key = RankedSignature(steps);
                        _rankGraphCache[key] = plans;
                        _rankGraphLru.AddFirst(key);
                        while (_rankGraphLru.Count > GraphCacheEntries)
                        {
                            var evict = _rankGraphLru.Last.Value;
                            _rankGraphLru.RemoveLast();
                            if (_rankGraphCache.Remove(evict, out CachedGraph[] old))
                                foreach (CachedGraph plan in old) plan.Dispose();
                        }
                    }
                    else
                    {
                        owned = true;
                    }
                }
                // Decode and verify partials are summed exactly; a prompt chunk's take the fast transport (as Qwen35Model's
                // tensor-parallel graphs do), unless the family needs exact sums throughout.
                bool exact = ExactTensorParallelSums || replayable;
                if (!executor.RunRanked(Array.ConvertAll(plans, p => p.Plan), Array.ConvertAll(steps, step => step.RunInputs()), exact, out string error))
                {
                    if (replayed) DropRankedReplay();
                    throw new InvalidOperationException($"{Config.Architecture} tensor-parallel graph step failed: {error}");
                }
                GraphRuns++;
                foreach (Slot slot in slots) slot.DeviceDirty = true;
                if (replayed)
                {
                    Array.Clear(steps);   // the replay owns them
                }
                else if (advance && !owned && Array.TrueForAll(steps, step => step.Advanceable))
                {
                    KeepRankedReplay(steps, plans, outputs, slots, starts, request.Shape);
                    Array.Clear(steps);
                }
                CachedGraph lead = plans[0];
                result = new GraphResult
                {
                    // A one-shot plan is disposed below: its outputs are the caller's arrays, which disposal does not free.
                    Logits = outputs.Logits >= 0 ? lead.Outputs[outputs.Logits]
                        : outputs.RankLogits != null ? JoinVocabSlices(outputs.RankLogits, slots.Length, (r, o) => plans[r].Outputs[o]) : null,
                    Final = outputs.Final >= 0 ? lead.Outputs[outputs.Final] : null,
                    Captures = outputs.Captures == null ? null : Array.ConvertAll(outputs.Captures, i => i >= 0 ? lead.Outputs[i] : null),
                    LastHidden = outputs.LastHiddenAt > 0 ? lead.Outputs[outputs.LastHiddenAt - 1] : null,
                };
                return true;
            }
            finally
            {
                if (!replayed)
                    foreach (GraphStep step in steps) step?.Dispose();
                if (owned)
                    foreach (CachedGraph plan in plans) plan?.Dispose();
                for (int r = 0; r < inputs.Length; r++)
                    if (!ReferenceEquals(inputs[r], hidden)) inputs[r].Dispose();
            }
        }

        /// <summary>The step's rows on every rank's allocator. Rank 0 copies them too unless they are its allocator's: another
        /// allocator of the same GPU (the model's) works on another stream, which the rank's kernels do not wait for.</summary>
        private Tensor[] RankInputs(Tensor hidden)
        {
            var inputs = new Tensor[_mesh.TensorParallel];
            float[] values = null;
            for (int r = 0; r < inputs.Length; r++)
            {
                IAllocator alloc = _mesh.Allocator(0, r);
                if (ReferenceEquals(hidden.Storage.Allocator, alloc))
                {
                    inputs[r] = hidden;
                    continue;
                }
                values ??= TensorToFloatArray(hidden);
                inputs[r] = new Tensor(alloc, DType.Float32, hidden.Sizes);
                inputs[r].SetElementsAsFloat(values);
                HostWritten(inputs[r]);
            }
            return inputs;
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

        /// <summary>The stream after a block's output [hidden, rows] joins it. Default: the residual add, after the block's
        /// post-norm where the family has one (<see cref="MixerPostNormWeight"/>, <see cref="FfnPostNormWeight"/>). The normed
        /// output is the add's first operand: ggml-metal fuses RMS_NORM + MUL + ADD only when each fused node's first source is
        /// the previous node (llama.cpp's LFM3 graph swapped its adds for that, e574a17b0).</summary>
        protected virtual int EmitBlockOutput(GraphStep step, int layer, int stream, BlockSite site, int context, int output)
        {
            string post = site == BlockSite.Mixer ? MixerPostNormWeight(layer) : FfnPostNormWeight(layer);
            return post == null ? step.Op(GraphOp.Add, stream, output) : step.Op(GraphOp.Add, step.RmsNorm(output, post), stream);
        }

        /// <summary>The head's input [hidden, rows] from the stream. Default: the output norm.</summary>
        protected virtual int EmitStreamOutput(GraphStep step, int stream) => step.RmsNorm(stream, "output_norm.weight");

        /// <summary>A DFlash drafter's tapped row [hidden, rows] of the stream entering a layer. Default: the stream.</summary>
        protected virtual int EmitCaptureRow(GraphStep step, int stream) => stream;

        /// <summary>A captured layer input as an output: every row's, or the step's last
        /// <see cref="GraphRequest.CaptureTail"/> rows' (copied out, so the rest is freed with the step).</summary>
        private int EmitCaptureOutput(GraphStep step, int stream, GraphRequest request)
        {
            int rows = step.Rows, hiddenSize = Config.HiddenSize;
            int x = EmitCaptureRow(step, stream);
            int tail = request.CaptureTail > 0 && request.CaptureTail < rows ? request.CaptureTail : rows;
            if (tail < rows)
                x = step.Cont(step.View2(x, hiddenSize, tail, hiddenSize * 4L, (rows - tail) * (long)hiddenSize * 4));
            return step.Output(x, (long)hiddenSize * tail);
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

        /// <summary>One sequence's attention operands over its cache window: its rows' queries [d, heads, rows] (rotated, before
        /// the permute the attention takes), the K and V caches [d, capacity, kvHeads], the rows the step wrote them at (I64
        /// [rows]: their KV indices), the window's keys, the local heads and KV heads, the head width and the scale.</summary>
        protected readonly record struct WindowAttentionOperands(int Queries, int KCache, int VCache, int Rows, int Window, int Heads, int KvHeads,
            int HeadDim, float Scale);

        /// <summary>Open a fused region around sequence <paramref name="s"/>'s mask and attention over its cache window in
        /// <paramref name="layer"/>: <see cref="EmitAttentionMask"/>'s ops and the attention follow, and the attention's
        /// result closes it. A family whose mask an executor may run with the attention as its own kernels opens one (Qwen
        /// 3.8's QSA selection), what the region reads from outside it emitted before it opens; default none.</summary>
        protected virtual bool BeginWindowAttention(GraphStep step, int layer, int x, int s, in WindowAttentionOperands attention) => false;

        /// <summary>A one-sequence step's attention prep operands in a layer: its rows' q [d, heads, rows] and k [d, kvHeads, rows]
        /// as projected (views of the products, before the copies and <see cref="EmitQueryKey"/>'s ops), v [dv, kvHeads, rows], the
        /// K and V caches [d, capacity, kvHeads], the rows the step writes them at (I64 [rows]: their KV indices), the local heads
        /// and KV heads, and the key and value head widths.</summary>
        protected readonly record struct AttentionPrepOperands(int Queries, int Keys, int Values, int KCache, int VCache, int Rows, int Heads,
            int KvHeads, int HeadDim, int ValueDim);

        /// <summary>Whether a one-sequence step whose rows are written into the caches by index takes <paramref name="layer"/>'s
        /// query/key prep and cache writes as a region (<see cref="BeginAttentionPrep"/>); default none.</summary>
        protected virtual bool TakesAttentionPrep(GraphStep step, int layer) => false;

        /// <summary>Open a fused region over <paramref name="layer"/>'s query/key prep (the copies of q, k and v and
        /// <see cref="EmitQueryKey"/>'s ops) and its K/V cache writes, which the rotated queries close: a family whose prep an
        /// executor may run as one kernel opens one where <see cref="TakesAttentionPrep"/> said so.</summary>
        protected virtual void BeginAttentionPrep(GraphStep step, int layer, in AttentionPrepOperands prep)
            => throw new NotSupportedException($"{Config.Architecture} takes no attention prep region.");

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
            int kvType = KvGgmlType(_kvCacheDtype.ToDType());
            GraphCapabilities caps = GraphExecutor.Capabilities;
            // Decode rows and short chunks (a verify window) are written by index. Tensor-parallel graphs write every chunk by
            // index too: ggml's meta device splits set_rows by KV head, while a copy into a strided cache view fails on the ranks.
            bool Indexed(int count) => count <= SmallChunkRows || _graphTpRanks != null || caps.IndexedCacheWrites
                || ((caps.PaddedCacheAttention || (PadsPromptWindowsOnVulkan && _backend == BackendType.GgmlVulkan)) && kvType != GraphFormat.F32);
            // One sequence's rows written by index: a family may take their query/key prep and cache writes as one region, which
            // the rotated queries close once the rows are written (the caches' leaves and the write rows exist before it opens).
            // (A sliding-window ring past its first lap writes its rows at their cells, which the prep region does not take.)
            bool prep = step.Sequences == 1 && Indexed(step.Counts[0]) && !RingWrapped(layer, step.Starts[0], step.Counts[0])
                && TakesAttentionPrep(step, layer);
            if (prep)
            {
                Slot only = step.Slots[0];
                int cells = CacheCells(layer, only.Capacity);
                BeginAttentionPrep(step, layer, new AttentionPrepOperands(q, k, v,
                    step.State(only.K[layer][step.RankIndex], kvType, KvCacheSplit(), d, cells, kvHeads),
                    step.State(only.V[layer][step.RankIndex], kvType, KvCacheSplit(), dv, cells, kvHeads),
                    step.WriteRows(step.Starts[0], step.Counts[0], 0), heads, kvHeads, d, dv));
            }
            // Where the executor's ops read a packed product's row blocks where they lie (GraphCapabilities.PackedProductBlocks),
            // a fused QKV product's queries and keys go to the family's prep as its views (the norms and rotary write fresh
            // rows) and one sequence's values to the cache writes (set_rows reads any strides): no copies of every prompt
            // chunk's q, k and v. Elsewhere, and for what the prep leaves a view, contiguous copies.
            bool blocks = GraphExecutor.Capabilities.PackedProductBlocks && !prep;
            // A prompt chunk's fresh rows as their permuted views (GraphCapabilities.FreshRowViews): written by index, attended
            // where they lie.
            bool freshViews = caps.FreshRowViews && _graphTpRanks == null;
            if (!blocks)
            {
                q = step.Cont(q);
                k = step.Cont(k);
            }
            if (!blocks || step.Sequences > 1 || !(Indexed(step.Counts[0]) || freshViews) || _graphTpRanks != null) v = step.Cont(v);
            EmitQueryKey(step, layer, ref q, ref k);
            q = step.Cont(q);
            k = step.Cont(k);

            float scale = 1.0f / MathF.Sqrt(d);
            long kRow = RowBytes(kvType, d), vRow = RowBytes(kvType, dv);
            int n = step.Sequences;
            int band = SlidingWindow(layer);
            int attn = -1;
            for (int s = 0; s < n; s++)
            {
                int count = step.Counts[s], start = step.Starts[s];
                Slot slot = step.Slots[s];
                long firstRow = step.First[s];
                // This sequence's rows are a prompt chunk (a mixed step marks only its prompt sequences' attention).
                bool promptChunk = step.PromptChunks && count > SmallChunkRows;
                int qs = n == 1 ? q : step.View3(q, d, heads, count, d * f, d * heads * f, firstRow * d * heads * f);
                int ks = n == 1 ? k : step.View3(k, d, kvHeads, count, d * f, d * kvHeads * f, firstRow * d * kvHeads * f);
                int vs = n == 1 ? v : step.View3(v, dv, kvHeads, count, dv * f, dv * kvHeads * f, firstRow * dv * kvHeads * f);
                // Under a prep region the queries are read past its end only (its result): their view follows it.
                int qPerm = prep ? -1 : step.Permute(qs, 0, 2, 1, 3);
                // [d, heads, count] -> [d, count, heads]; one row is the same bytes either way. Rows written by index are read
                // from the permuted view (set_rows reads rows at any strides: no copy of every prompt chunk's K and V, 22 of a
                // LFM3 prompt step's 471 Metal dispatches); a chunk that attends its own rows, and the tensor-parallel ranks'
                // writes (ggml's meta device splits set_rows by KV head), take them contiguous.
                // A chunk that attends its own rows reads them from the views too where the executor takes them so
                // (FreshRowViews; not past a ring's first lap, whose runs address the rows' contiguous copies).
                bool wrapped = RingWrapped(layer, start, count);
                bool indexedRows = (Indexed(count) || (freshViews && !wrapped)) && _graphTpRanks == null;
                int kFresh = count == 1 ? step.Reshape(ks, d, 1, kvHeads) : indexedRows ? step.Permute(ks, 0, 2, 1, 3) : step.Cont(step.Permute(ks, 0, 2, 1, 3));
                int vFresh = count == 1 ? step.Reshape(vs, dv, 1, kvHeads) : indexedRows ? step.Permute(vs, 0, 2, 1, 3) : step.Cont(step.Permute(vs, 0, 2, 1, 3));

                // The layer's cells: a sliding-window layer's ring (StandardDecoderModel.SlidingRings.cs) holds at most its own.
                int cap = CacheCells(layer, slot.Capacity);
                int ring = RingOf(layer);
                int ck = step.State(slot.K[layer][step.RankIndex], kvType, KvCacheSplit(), d, cap, kvHeads);
                int cv = step.State(slot.V[layer][step.RankIndex], kvType, KvCacheSplit(), dv, cap, kvHeads);

                int fa;
                if (Indexed(count) && wrapped)
                {
                    // Rows reaching a ring past its first lap: written at their cells, and attending the whole ring under the mask
                    // of the positions its cells hold (one graph for every position on: decode steps advance in place).
                    int rowsAt = step.RingWriteRows(start, count, s, ring);
                    step.Effect(step.Op(GraphOp.SetRows, ck, kFresh, rowsAt));
                    step.Effect(step.Op(GraphOp.SetRows, cv, vFresh, rowsAt));
                    int gqa = heads / kvHeads;
                    int kWin = step.Op(GraphOp.KvWindow, stackalloc[] { ck }, stackalloc long[] { d, cap, kvHeads, 0, ring, kvType, count, gqa });
                    int vWin = step.Op(GraphOp.KvWindow, stackalloc[] { cv }, stackalloc long[] { dv, cap, kvHeads, 0, ring, kvType, count, gqa });
                    int mask = step.RingBandMask(ring, start, count, s, band);
                    fa = step.FlashAttention(qPerm, kWin, vWin, EmitAttentionMask(step, layer, x, s, mask, ring), scale,
                        AttentionKvMax(step, layer, s), promptChunk, DeviceBand(step, band, d, kvType));
                }
                else if (wrapped)
                {
                    // A prompt chunk over a ring past its first lap: its rows go to their cells (at most two runs), and it attends
                    // the cached rows its first row's window reaches (at most two runs, in position order) and its own. The ring's
                    // W - 1 + rows cells keep the two sets apart (MaxSequenceStepRows).
                    (int c1, int n1, int c2, int n2) = RingRuns(ring, start, start + count);
                    WriteRun(kFresh, ck, d, kRow, c1, 0, n1);
                    WriteRun(vFresh, cv, dv, vRow, c1, 0, n1);
                    if (n2 > 0)
                    {
                        WriteRun(kFresh, ck, d, kRow, c2, n1, n2);
                        WriteRun(vFresh, cv, dv, vRow, c2, n1, n2);
                    }
                    int first = Math.Max(0, start - band + 1);
                    (int p1, int m1, int p2, int m2) = RingRuns(ring, first, start);
                    int kAttn = kFresh, vAttn = vFresh;
                    if (m1 > 0)
                    {
                        kAttn = step.Op(GraphOp.Concat, stackalloc[] { ReadRuns(ck, d, kRow, p1, m1, p2, m2), kFresh }, stackalloc long[] { 1 });
                        vAttn = step.Op(GraphOp.Concat, stackalloc[] { ReadRuns(cv, dv, vRow, p1, m1, p2, m2), vFresh }, stackalloc long[] { 1 });
                    }
                    int mask = step.BandCausalMask(first, start, count, s, band);
                    fa = step.FlashAttention(qPerm, kAttn, vAttn, EmitAttentionMask(step, layer, x, s, mask, start + count - first), scale,
                        promptChunk: promptChunk);

                    void WriteRun(int fresh, int cache, int width, long rowBytes, int cell, int from, int n)
                        => step.Effect(step.Op(GraphOp.Cpy,
                            step.View3(fresh, width, n, kvHeads, width * f, width * f * count, from * width * f),
                            step.View3(cache, width, n, kvHeads, rowBytes, rowBytes * cap, cell * rowBytes)));

                    int ReadRuns(int cache, int width, long rowBytes, int cell, int n, int cell2, int n2)
                    {
                        int run = Read(cell, n);
                        return n2 > 0 ? step.Op(GraphOp.Concat, stackalloc[] { run, Read(cell2, n2) }, stackalloc long[] { 1 }) : run;

                        int Read(int at, int rows)
                        {
                            int view = step.View3(cache, width, rows, kvHeads, rowBytes, rowBytes * cap, at * rowBytes);
                            return kvType != GraphFormat.F32 ? step.Op(GraphOp.Cast, stackalloc[] { view }, stackalloc long[] { GraphFormat.F32 }) : view;
                        }
                    }
                }
                else if (Indexed(count))
                {
                    // Decode and short chunks (a verify window): rows are written by index and
                    // attention reads the padded cache window under a causal mask, so every row is
                    // computed as a decode row is and a window bucket replays the same graph.
                    int rowsAt = step.WriteRows(start, count, s);
                    step.Effect(step.Op(GraphOp.SetRows, ck, kFresh, rowsAt));
                    step.Effect(step.Op(GraphOp.SetRows, cv, vFresh, rowsAt));
                    if (prep)
                    {
                        step.EndFused(stackalloc[] { q });
                        qPerm = step.Permute(qs, 0, 2, 1, 3);
                    }
                    int stepRows = DecodeWindowStep;
                    bool capacity = CapacityWindow(step, d, dv, kvType);
                    int window = capacity ? cap : Math.Min(cap, (start + count + stepRows - 1) / stepRows * stepRows);
                    // A sliding-window layer reads only the cells its rows attend: from the bucket below the first row's
                    // window (so the graph is the same for a bucket's positions, as the window's end is) to the window's end.
                    // A capacity window keeps the slot's whole extent (one graph at every position) and masks the band.
                    int first = band > 0 && !capacity ? SlidingWindowFirstCell(start, band, stepRows) : 0;
                    if (band > 0 && !capacity) step.LimitAdvance(start, first + stepRows + band - 1, s);
                    int keys = window - first;
                    int gqa = heads / kvHeads;
                    int kWin = step.Op(GraphOp.KvWindow, stackalloc[] { ck }, stackalloc long[] { d, cap, kvHeads, first, keys, kvType, count, gqa });
                    int vWin = step.Op(GraphOp.KvWindow, stackalloc[] { cv }, stackalloc long[] { dv, cap, kvHeads, first, keys, kvType, count, gqa });
                    bool region = BeginWindowAttention(step, layer, x, s, new WindowAttentionOperands(qs, ck, cv, rowsAt, window, heads, kvHeads, d, scale));
                    int mask = band > 0 ? step.BandMask(keys, first, start, count, s, band) : step.ChunkMask(window, start, count, s);
                    fa = step.FlashAttention(qPerm, kWin, vWin, EmitAttentionMask(step, layer, x, s, mask, keys), scale,
                        AttentionKvMax(step, layer, s), promptChunk, capacity ? DeviceBand(step, band, d, kvType) : 0);
                    if (region) step.EndFused(stackalloc[] { fa });
                }
                else
                {
                    // A prompt chunk is written at its offset and attends to the cached prefix and
                    // its own fresh rows (F32), as the fused prefill graphs do.
                    if (freshViews)
                    {
                        // From the permuted views, by index (set_rows converts a row at a time).
                        int rowsAt = step.WriteRows(start, count, s);
                        step.Effect(step.Op(GraphOp.SetRows, ck, kFresh, rowsAt));
                        step.Effect(step.Op(GraphOp.SetRows, cv, vFresh, rowsAt));
                    }
                    else
                    {
                        step.Effect(step.Op(GraphOp.Cpy, kFresh, step.View3(ck, d, count, kvHeads, kRow, kRow * cap, start * kRow)));
                        step.Effect(step.Op(GraphOp.Cpy, vFresh, step.View3(cv, dv, count, kvHeads, vRow, vRow * cap, start * vRow)));
                    }
                    int kAttn = kFresh, vAttn = vFresh;
                    // A sliding-window layer's chunk attends the cached rows its first row's window reaches, not the whole prefix.
                    int first = band > 0 ? Math.Max(0, start - band + 1) : 0;
                    if (start > first)
                    {
                        int kPrev = step.View3(ck, d, start - first, kvHeads, kRow, kRow * cap, first * kRow);
                        int vPrev = step.View3(cv, dv, start - first, kvHeads, vRow, vRow * cap, first * vRow);
                        if (kvType != GraphFormat.F32)
                        {
                            kPrev = step.Op(GraphOp.Cast, stackalloc[] { kPrev }, stackalloc long[] { GraphFormat.F32 });
                            vPrev = step.Op(GraphOp.Cast, stackalloc[] { vPrev }, stackalloc long[] { GraphFormat.F32 });
                        }
                        kAttn = step.Op(GraphOp.Concat, stackalloc[] { kPrev, kFresh }, stackalloc long[] { 1 });
                        vAttn = step.Op(GraphOp.Concat, stackalloc[] { vPrev, vFresh }, stackalloc long[] { 1 });
                    }
                    int mask = band > 0 ? step.BandCausalMask(first, start, count, s, band) : step.CausalMask(start, count, s);
                    fa = step.FlashAttention(qPerm, kAttn, vAttn, EmitAttentionMask(step, layer, x, s, mask, start + count - first), scale,
                        promptChunk: promptChunk);
                }
                int part = step.Reshape(fa, heads * dv, count);
                attn = attn < 0 ? part : step.Op(GraphOp.Concat, stackalloc[] { attn, part }, stackalloc long[] { 1 });
            }

            if (gate >= 0)
            {
                // One region a backend may run as one kernel reading the gate where the projection put it.
                step.BeginFused(GraphFusion.SigmoidGate, stackalloc[] { attn, gate });
                int g = step.Reshape(step.Cont(gate), (int)qDim, rows);
                attn = step.Op(GraphOp.Mul, attn, step.Op(GraphOp.Sigmoid, g));
                step.EndFused(stackalloc[] { attn });
            }
            attn = EmitAttentionOutput(step, layer, x, attn);
            return step.MatMul(LayerName(layer, "attn_output.weight"), attn);
        }

        /// <summary>What a family applies to the attention result [heads * dv, rows] (the rank's heads) before the output
        /// projection, given the layer's input <paramref name="x"/> [hidden, rows]: LFM2/LFM3's head-wise sigmoid gate
        /// (a separate projection to one logit per head). Default: nothing. The per-op path's counterpart is
        /// <see cref="GateAttentionOutput"/>; a per-head weight it reads shards by heads under tensor parallelism
        /// (<see cref="HeadParallelLayerWeights"/>).</summary>
        protected virtual int EmitAttentionOutput(GraphStep step, int layer, int x, int attn) => attn;

        /// <summary>The band a sliding-window attention over a ring or a capacity window passes its executor, where that executor
        /// reads band masks on the device (<see cref="GraphCapabilities.BandMasksOnDevice"/>) and attends this cache that way
        /// (decode rows and verify windows, <see cref="MaxAdvancedWindowRows"/> rows a sequence at most); 0 otherwise (the
        /// mask is classified from its host values, and a recorded step then cannot replay the band's moving floor).</summary>
        private int DeviceBand(GraphStep step, int band, int headDim, int kvType)
            => band > 0 && GraphExecutor.Capabilities.BandMasksOnDevice && Array.TrueForAll(step.Counts, c => c <= MaxAdvancedWindowRows)
               && GraphExecutor.AttendsCapacityWindows(headDim, kvType) ? band : 0;

        /// <summary>The first cache cell a sliding-window layer's rows continuing at <paramref name="start"/> read: the bucket of
        /// <paramref name="stepRows"/> cells below the first row's window (positions (start - window, start]), so every position
        /// of a bucket reads the same cells (a decode step advances through them) and a window moves a bucket at a time.</summary>
        private static int SlidingWindowFirstCell(int start, int window, int stepRows)
            => start <= window ? 0 : (start - window) / stepRows * stepRows;

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

        /// <summary>Tests: a shared expert runs in place, not as a concurrent region.</summary>
        internal bool ConcurrentRegionsDisabledForTests { get; set; }

        /// <summary>Tests: a host-served layer's selection runs as its ops, not as a <see cref="GraphFusion.MoeSelect"/> region.</summary>
        internal bool MoeSelectRegionDisabledForTests { get; set; }

        private int EmitMoe(GraphStep step, int layer, int x)
        {
            string p = LayerName(layer, "");
            int rows = step.Rows, hidden = Config.HiddenSize, k = _expertsUsed;
            // The shared expert reads only x and is read only at the end: a region an executor may run beside the router and
            // the routed experts (the Qwen3.8 engines' forked shared-expert stream), emitted first so it can start first.
            int shared = -1;
            if (HasWeight(p + "ffn_gate_shexp.weight"))
            {
                bool beside = !ConcurrentRegionsDisabledForTests;
                if (beside) step.BeginFused(GraphFusion.ConcurrentBranch, stackalloc[] { x });
                int sa = step.Op(GraphOp.SwigluSplit, step.MatMul(p + "ffn_gate_shexp.weight", x), step.MatMul(p + "ffn_up_shexp.weight", x));
                shared = step.MatMul(p + "ffn_down_shexp.weight", sa);
                if (beside) step.EndFused(stackalloc[] { shared });
            }
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
            // A host-served layer's selection is one region an executor may run as one kernel (its routes then stay on the
            // device as they do in a routed-experts region).
            bool selectRegion = !region && !MoeSelectRegionDisabledForTests;
            if (selectRegion)
                step.BeginFused(GraphFusion.MoeSelect, stackalloc[] { logits }, stackalloc long[] { k, NormalizeExpertWeights ? 1 : 0 });
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
            if (selectRegion) step.EndFused(stackalloc[] { selected, weights });

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
                // A profiling load counts a resident layer's routes too (host seams count the offloaded ones): the ids are
                // read back after the run (an output costs ggml-metal its softmax / top-k fusion, on this load only).
                if (_recordingDeviceRoutes && GraphExecutor.Capabilities.HostExperts && !_rankGraphs)
                    step.RecordRoutes(step.Output(step.Op(GraphOp.Cont, selected), (long)k * rows), layer);   // the top-K is a view of the sort
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

            if (shared >= 0)
            {
                if (_weights.ContainsKey(p + "ffn_gate_inp_shexp.weight"))
                {
                    // One region a backend may run as one kernel (the direct engines' shared_gate): four launches a layer
                    // otherwise (the gate's product, its sigmoid, the scale and the add).
                    int w = step.WeightAs(p + "ffn_gate_inp_shexp.weight", hidden, 1);
                    step.BeginFused(GraphFusion.SharedExpertGate, stackalloc[] { output, shared, x, w });
                    int g = step.Op(GraphOp.MulMat, w, x);   // [1, rows]
                    shared = step.Op(GraphOp.Mul, shared, step.Op(GraphOp.Sigmoid, g));
                    output = step.Op(GraphOp.Add, output, shared);
                    step.EndFused(stackalloc[] { output });
                }
                else
                    output = step.Op(GraphOp.Add, output, shared);
            }
            return output;
        }

        // ---------------------------------------------------------------- the step

        /// <summary>Rows a decode attention window grows by: 64 on Metal, whose padded tail costs
        /// more than a replay rebuild every 64 tokens (measured with Qwen 3.5's fused decode); 256
        /// (the flash-attention KV stride) elsewhere.</summary>
        private int DecodeWindowStep => GraphExecutor?.Capabilities.DecodeWindowRows ?? 256;

        /// <summary>
        /// Whether a decode row's or verify window's padded attention window is its slot's whole capacity: where the executor's
        /// kernels compute from the written length what the bucket window around it computes
        /// (<see cref="IGraphExecutor.AttendsCapacityWindows"/>), so the step's graph is the same at every position of the slot and
        /// one recorded step of a width replays at all of them. Bucket windows had made a new graph - a trace, a compile and a
        /// recording, 25-100 ms - every <see cref="DecodeWindowStep"/> positions and for every verify width (and a batched step
        /// whenever any of its sequences crossed one). Only the steps that are kept and recorded (every sequence the same rows, up
        /// to <see cref="MaxAdvancedWindowRows"/>, no more sequences than the recorder batches): the others run as they are
        /// traced, and a capacity's mask is filled, widened and classified on the host at every run of them.
        /// </summary>
        private bool CapacityWindow(GraphStep step, int keyDim, int valueDim, int kvType)
        {
            int[] counts = step.Counts;
            if (counts[0] > MaxAdvancedWindowRows || !Array.TrueForAll(counts, c => c == counts[0]) || keyDim != valueDim
                || CapacityWindowsDisabledForTests || GraphExecutor?.AttendsCapacityWindows(keyDim, kvType) != true)
                return false;
            return step.Sequences == 1 || (GraphExecutor is IDecodeCaptureExecutor recorder && step.Sequences <= recorder.MaxRecordedSequences);
        }

        /// <summary>Tests: short chunks attend the bucket window around their length, as before capacity windows.</summary>
        internal bool CapacityWindowsDisabledForTests { get; set; }

        private readonly Dictionary<IAllocator, IGraphExecutor> _graphExecutors = new(ReferenceEqualityComparer.Instance);

        /// <summary>The backend's graph executor plugin for the current stage's allocator, or null. An executor that binds
        /// leaves by object (direct CUDA) allocates its per-run tensors on its allocator's device, so each stage of a
        /// layer split runs on its own (one for stage 0's GPU read stage 1's tensors from the wrong device); GGML's
        /// executor is one per backend and switches ranks itself, so every stage gets the same one.</summary>
        protected IGraphExecutor GraphExecutor => ExecutorFor(_allocator);

        /// <summary>The executor of <paramref name="allocator"/>'s device: what a stage's state binds its device copies
        /// through (a layer split's states live on their stage's device whichever stage is current).</summary>
        protected IGraphExecutor ExecutorFor(IAllocator allocator)
        {
            if (allocator == null) return null;
            if (!_graphExecutors.TryGetValue(allocator, out IGraphExecutor executor))
                _graphExecutors[allocator] = executor = GraphExecutors.For(allocator);
            return executor;
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
            // Causal masks by what fills them: every attention layer reads its sequence's. A step of several sequences had kept
            // one mask, which each sequence's evicted, so every layer emitted every sequence's again (a 2048-row chunk's 67M
            // values at 32K keys built 16 times in a step that carried other requests' decode rows).
            private readonly Dictionary<(int Start, int Count, int Floor), int> _causalMasks = new();
            private readonly Dictionary<(int Window, int Start, int Count, int Sequence), int> _chunkMasks = new();
            // A chunk's write rows by (start, count): a step writes every sequence's chunk by index on executors that take
            // indexed cache writes at any width, and start * 64 + count had named (0, 2048) and (31, 64) alike.
            private readonly Dictionary<(int Start, int Count, int Sequence, int Ring), int> _writeIndex = new();
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

            /// <summary>The tensor-parallel rank whose shard this step traces (its weights' shards, caches, states and
            /// device), or -1: the model's own weights (one device, a stage, or ggml's meta device).</summary>
            internal int Rank { get; init; } = -1;

            /// <summary>The rank's index into per-rank arrays (0 without a rank).</summary>
            internal int RankIndex => Rank < 0 ? 0 : Rank;

            /// <summary>Ranks the step's heads and widths are split over (1 without a rank).</summary>
            public int Degree => Rank < 0 ? 1 : _m._mesh.TensorParallel;

            public int Rows { get; }
            public int Sequences => Slots.Length;
            public int[] First { get; }
            public int[] Counts { get; }
            public int[] Starts { get; private set; }

            /// <summary>A kept decode step gets its own positions (the caller's array is the step's it ran for; stages
            /// share it), which <see cref="TryAdvance"/> moves on.</summary>
            internal void OwnPositions() => Starts = (int[])Starts.Clone();
            /// <summary>Every sequence forwards a short chunk (the graph may be replayed).</summary>
            public bool IsDecode { get; }
            /// <summary>The step forwards prompt tokens (<see cref="GraphRequest.PromptChunks"/>): a sequence's attention over more
            /// than <see cref="SmallChunkRows"/> rows is marked as its prompt chunk's. Draft steps never set it.</summary>
            internal bool PromptChunks { get; set; }
            /// <summary>Keep the recurrent state after every row (a verify window).</summary>
            public bool RecurrentSnapshots { get; }
            /// <summary>Cache rows below this are masked out (a draft layer that resumed after a gap).</summary>
            public int AttentionFloor { get; init; }
            /// <summary>Each sequence's floor in place of <see cref="AttentionFloor"/> (a batched draft step's lanes, whose
            /// layers resumed at different rows); null: <see cref="AttentionFloor"/> for every sequence.</summary>
            public int[] AttentionFloors { get; init; }
            /// <summary>The attention floor of sequence <paramref name="sequence"/> (-1: <see cref="AttentionFloor"/>).</summary>
            public int FloorOf(int sequence)
                => AttentionFloors != null && (uint)sequence < (uint)AttentionFloors.Length ? AttentionFloors[sequence] : AttentionFloor;
            /// <summary>Token positions [rows] (I32).</summary>
            public int Positions { get; private set; } = -1;
            /// <summary>The positions' values (row order).</summary>
            public int[] PositionValues { get; private set; }
            internal bool Rejected { get; private set; }
            internal (ulong, ulong) Signature => _b.Signature;

            /// <summary>The signature without the state leaves' identities (<see cref="StateLeaves"/>).</summary>
            internal (ulong, ulong) StructuralSignature => _b.StructuralSignature;

            /// <summary>The state leaves the step binds, in the order its plan binds them.</summary>
            internal IReadOnlyList<Tensor> StateLeaves => _stateOrder;
            private readonly List<Tensor> _stateOrder = new();

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

            /// <summary>Output <paramref name="output"/> holds layer <paramref name="layer"/>'s top-K expert ids (I32).</summary>
            internal void RecordRoutes(int output, int layer) => (_routes ??= new()).Add((output, layer));

            private List<(int Output, int Layer)> _routes;

            /// <summary>The route outputs this emission recorded (null: none).</summary>
            internal (int Output, int Layer)[] Routes => _routes?.ToArray();

            // ---- leaves

            /// <summary>A model weight by name: quantized (its device-cache key) or F32.</summary>
            public int Weight(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (Rank >= 0 && _m._tpQuantWeights.TryGetValue(name, out QuantizedWeight[] shards))
                {
                    QuantizedWeight shard = shards[Rank];
                    long[] ne = { shard.Ne0, shard.Ne1 };
                    node = Leaf(GraphLeafRole.Weight, shard.GgmlType, ne, shard.CacheKey != IntPtr.Zero ? shard.CacheKey : shard.Data,
                        shard.RawBytes, null, shard);
                }
                else if (Rank >= 0 && _m._tpWeights.TryGetValue(name, out Tensor[] tensors))
                {
                    node = TensorLeaf(GraphLeafRole.Weight, tensors[Rank], GraphFormat.F32, Extents(tensors[Rank]));
                }
                else if (_m._quantWeights.TryGetValue(name, out QuantizedWeight q))
                {
                    IntPtr key = q.CacheKey != IntPtr.Zero ? q.CacheKey : q.Data;
                    long[] ne = { q.Ne0, q.Ne1 };
                    node = Leaf(GraphLeafRole.Weight, q.GgmlType, ne, key, q.RawBytes, _m.SplitOf(name, ne, q.GgmlType), q);
                }
                else if (_m._weights.TryGetValue(name, out Tensor t))
                {
                    long[] ne = Extents(t);
                    node = TensorLeaf(GraphLeafRole.Weight, Replicated(t), GraphFormat.F32, ne, _m.SplitOf(name, ne, GraphFormat.F32));
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
                return TensorLeaf(GraphLeafRole.Weight, Replicated(t), GraphFormat.F32, ne);
            }

            /// <summary>Stacked experts [ne0, ne1, experts] of a routed-expert layer.</summary>
            public int Experts(string name)
            {
                if (_weights.TryGetValue(name, out int node)) return node;
                if (Rank >= 0 && _m._mesh.TensorParallel > 1)
                {
                    IGraphExpertWeights shard = _m.RankExpertSet(name, Rank) ?? throw Reject($"experts '{name}' have no shards for rank {Rank}");
                    node = Leaf(GraphLeafRole.Weight, shard.Format, stackalloc long[] { shard.Ne0, shard.Ne1, shard.Ne2 }, shard.CacheKey,
                        shard.Bytes, null, shard);
                    _weights[name] = node;
                    return node;
                }
                if (!_m._stackedExpertWeights.TryGetValue(name, out StackedExpertWeights w)) throw Reject($"experts '{name}' are not stacked");
                long[] ne = { w.PerExpertNe0, w.PerExpertNe1, w.NumExperts };
                node = Leaf(GraphLeafRole.Weight, w.GgmlType, ne, w.Data, w.TotalRawBytes, _m.SplitOf(name, ne, w.GgmlType), _m.ExpertSet(name, w));
                _weights[name] = node;
                return node;
            }

            /// <summary>An F32 tensor the family prepared (device constants, norms) as a read-only leaf.</summary>
            public int TensorWeight(Tensor t, params long[] ne)
            {
                // An executor that binds tensors by object reads them on the rank's device; GGML binds host bytes and keeps
                // each rank's device copy itself, so any rank may read any tensor.
                if (Rank >= 0 && !_m.IsGgmlBackend && !ReferenceEquals(t.Storage.Allocator, _m._mesh.Allocator(0, Rank)))
                    throw new InvalidOperationException($"A leaf of tensor-parallel rank {Rank} lives on another allocator: pass the rank's " +
                        "own tensor, or Replicated() for one every rank reads.");
                return TensorLeaf(GraphLeafRole.Weight, t, GraphFormat.F32, ne.Length == 0 ? Extents(t) : ne);
            }

            /// <summary><paramref name="t"/> as the step's rank reads it: a copy on the rank's allocator (made once) for a tensor
            /// every rank reads whole. Rank 0 too: the model's own allocator of that GPU is another one, with its own stream,
            /// which the rank's kernels and its group's all-reduce do not order against (Qwen3.6-35B-A3B --tp 2 on direct
            /// CUDA: decode 19.9 -> 29.9 tok/s with rank 0 on its own copies).</summary>
            public Tensor Replicated(Tensor t) => Rank >= 0 ? _m.Replica(t, 0, Rank) : t;

            /// <summary>Read-write state the graph updates in place (a slot's cache or recurrent state).</summary>
            public int State(Tensor t, int type, params long[] ne) => State(t, type, null, ne);

            /// <summary>A state spread over tensor-parallel ranks by <paramref name="split"/> (null: mirrored).</summary>
            internal int State(Tensor t, int type, LeafSplit split, params long[] ne)
            {
                if (_states.TryGetValue(t, out int node)) return node;
                node = TensorLeaf(GraphLeafRole.State, t, type, ne, _m._graphTpRanks != null ? split : null);
                _states[t] = node;
                _stateOrder.Add(t);
                return node;
            }

            /// <summary>Sequence s's recurrent state of (layer, rank 0).</summary>
            internal RecurrentState SequenceState(int s, int layer) => Slots[s].Recurrent[layer][RankIndex];

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

            public int InputF32(float[] values, params long[] ne) => InputF32(values, InputKind.Opaque, ne);

            /// <summary>F32 values that stay what they are while the rows' positions move.</summary>
            internal int ConstantInputF32(float[] values, params long[] ne) => InputF32(values, InputKind.Constant, ne);

            private int InputF32(float[] values, InputKind kind, long[] ne)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F32, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), (long)values.Length * sizeof(float), kind: kind, values: values);
                return node;
            }

            internal void SetPositions(int[] positions)
            {
                PositionValues = positions;
                Positions = InputI32(positions, positions.Length, InputKind.Positions);
            }

            /// <summary>Axis-major [4 * rows] rotary positions (ggml_rope_multi); rotary ops then take them.</summary>
            internal void SetPositionTable(int[] axisMajor)
            {
                PositionTable = InputI32(axisMajor, axisMajor.Length);
                PositionTableValues = axisMajor;
            }

            /// <summary>The host values of <see cref="PositionTable"/> (axis-major T, H, W, extra), or null.</summary>
            internal int[] PositionTableValues { get; private set; }

            /// <summary>The rotary position (T, H, W) of row <paramref name="i"/> of sequence <paramref name="s"/>: the
            /// position table's where one drives the step, else its KV index plus the slot's rotary delta on every axis.</summary>
            internal (int T, int H, int W) RowPosition(int s, int i)
            {
                if (PositionTableValues is { } table)
                {
                    int rows = table.Length / 4, r = First[s] + i;
                    return (table[r], table[rows + r], table[2 * rows + r]);
                }
                int p = Starts[s] + i + Slots[s].RopeDelta;
                return (p, p, p);
            }

            /// <summary>Multi-axis rotary over x [d, heads, n] at the axis-major <paramref name="table"/> (I32 [4 n]) rather
            /// than the step's (keys pooled at their blocks' first cells, some inside media).</summary>
            internal int RopeMultiAt(int x, RotarySpec spec, int table)
            {
                int[] sections = _m.RotarySections ?? throw new NotSupportedException($"{_m.Config.Architecture} has no multi-axis rotary.");
                return Op(GraphOp.RopeMulti, stackalloc[] { x, table },
                    stackalloc long[] { spec.Dims, 40 /* GGML_ROPE_TYPE_IMROPE */, spec.OriginalContext,
                        sections[0], sections[1], sections[2], sections.Length > 3 ? sections[3] : 0 },
                    stackalloc[] { spec.FreqBase, spec.FreqScale, spec.ExtFactor, spec.AttnFactor, spec.BetaFast, spec.BetaSlow });
            }

            /// <summary>Multi-axis rotary positions [4 * rows] (I32), or -1.</summary>
            public int PositionTable { get; private set; } = -1;

            /// <summary>Rows [n, 1] (I64) a scattered write goes to (they do not follow the rows' positions).</summary>
            internal int InputRows(int[] rows)
            {
                var values = new long[rows.Length];
                for (int i = 0; i < rows.Length; i++) values[i] = rows[i];
                int node = Leaf(GraphLeafRole.Input, GraphFormat.I64, stackalloc long[] { rows.Length, 1 }, IntPtr.Zero, 0);
                AddInput(node, Pin(values), values.LongLength * sizeof(long), values: values);
                return node;
            }

            /// <summary>The rows [count, 1] (I64) sequence <paramref name="sequence"/>'s short chunk is written at: start ..
            /// start + count - 1 (one input a sequence: an advance moves each sequence's on by its own rows).</summary>
            internal int WriteRows(int start, int count, int sequence) => RingWriteRows(start, count, sequence, 0);

            /// <summary>The cells [count, 1] (I64) sequence <paramref name="sequence"/>'s rows are written at in a ring of
            /// <paramref name="ring"/> cells (0: a linear cache, cell = position): (start + i) mod ring. An advance moves them
            /// round the ring.</summary>
            internal int RingWriteRows(int start, int count, int sequence, int ring)
            {
                if (_writeIndex.TryGetValue((start, count, sequence, ring), out int node)) return node;
                node = Leaf(GraphLeafRole.Input, GraphFormat.I64, stackalloc long[] { count, 1 }, IntPtr.Zero, 0);
                var rows = new long[count];
                for (int i = 0; i < count; i++) rows[i] = ring > 0 ? (start + i) % ring : start + i;
                AddInput(node, Pin(rows), count * sizeof(long), kind: InputKind.WriteRows, values: rows, sequence: sequence, ring: ring);
                _writeIndex[(start, count, sequence, ring)] = node;
                return node;
            }

            /// <summary>An F16 causal mask [window, count] of sequence <paramref name="sequence"/>: row q sees cache rows 0 ..
            /// start + q. A prompt chunk's (more rows than a verify window: never replayed, advanced or recorded) goes to an
            /// executor that takes <see cref="GraphCapabilities.StructuralCausalMasks"/> as its pattern, no host values: their
            /// fill, widening and scan had cost about 2 ns a value at every step (137M values for a 2048-row chunk at 67K keys)
            /// for a kernel that reads only the pattern. Decode rows and verify windows keep their values, which an advance
            /// opens a column on.</summary>
            internal int ChunkMask(int window, int start, int count, int sequence)
            {
                var key = (window, start, count, sequence);
                if (_chunkMasks.TryGetValue(key, out int cached)) return cached;
                var pattern = new CausalMaskInput(window, start, count, FloorOf(sequence));
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { window, count }, IntPtr.Zero, 0);
                if (count > SmallChunkRows && _m.GraphExecutor?.Capabilities.StructuralCausalMasks == true)
                {
                    AddInput(node, IntPtr.Zero, pattern.Values * sizeof(ushort), handle: pattern);
                    return CacheMask(_chunkMasks, key, node);
                }
                var mask = GC.AllocateUninitializedArray<ushort>(checked((int)pattern.Values));   // the fill writes every cell
                pattern.FillF16(mask);
                AddInput(node, Pin(mask), mask.LongLength * sizeof(ushort), kind: InputKind.DecodeMask, values: mask,
                    window: window, start: start, count: count, sequence: sequence);
                return CacheMask(_chunkMasks, key, node);
            }

            /// <summary>Sliding-window masks of this step by (window, offset, start, count, sequence, band): every layer of one
            /// window reads the same input.</summary>
            private Dictionary<(int Window, int Offset, int Start, int Count, int Sequence, int Band), int> _bandMasks;

            /// <summary>
            /// A sliding-window F16 mask [<paramref name="window"/>, count] of sequence <paramref name="sequence"/> over cache rows
            /// [<paramref name="offset"/>, offset + window): row q (position start + q) sees the keys at positions
            /// (start + q - <paramref name="band"/>, start + q] (llama.cpp's LLAMA_SWA_TYPE_STANDARD: a key is masked when
            /// p_query - p_key &gt;= band) and none before the sequence's attention floor. An advance closes the cells the band leaves
            /// as it opens the new ones (<see cref="TryAdvance(Tensor[], int[])"/>).
            /// </summary>
            internal int BandMask(int window, int offset, int start, int count, int sequence, int band)
            {
                if (band <= 0) throw new ArgumentOutOfRangeException(nameof(band), band, "a sliding window needs a positive width");
                var key = (window, offset, start, count, sequence, band);
                if (_bandMasks != null && _bandMasks.TryGetValue(key, out int cached)) return cached;
                var mask = GC.AllocateUninitializedArray<ushort>(checked(window * count));   // the fill writes every cell
                FillBandRows(mask, window, offset, start, count, FloorOf(sequence), band);
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { window, count }, IntPtr.Zero, 0);
                AddInput(node, Pin(mask), mask.LongLength * sizeof(ushort), kind: InputKind.DecodeMask, values: mask,
                    window: window, start: start, count: count, sequence: sequence, offset: offset, band: band);
                (_bandMasks ??= new())[key] = node;
                return node;
            }

            /// <summary>A sliding-window F16 mask [start + count - offset, count] for a prompt chunk of sequence
            /// <paramref name="sequence"/> attending to cache rows from <paramref name="offset"/> on (see <see cref="BandMask"/>);
            /// its values do not follow an advance (a chunk runs once).</summary>
            internal int BandCausalMask(int offset, int start, int count, int sequence, int band)
            {
                if (band <= 0) throw new ArgumentOutOfRangeException(nameof(band), band, "a sliding window needs a positive width");
                int kv = start + count - offset;
                var key = (kv, offset, start, count, sequence, -band);   // negative band: a chunk's mask, not a decode mask
                if (_bandMasks != null && _bandMasks.TryGetValue(key, out int cached)) return cached;
                var mask = GC.AllocateUninitializedArray<ushort>(checked(kv * count));
                FillBandRows(mask, kv, offset, start, count, FloorOf(sequence), band);
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { kv, count }, IntPtr.Zero, 0);
                AddInput(node, Pin(mask), mask.LongLength * sizeof(ushort));
                (_bandMasks ??= new())[key] = node;
                return node;
            }

            /// <summary>Ring band masks of this step by (ring, start, count, sequence, band): every ring layer of one geometry reads
            /// the same input.</summary>
            private Dictionary<(int Ring, int Start, int Count, int Sequence, int Band), int> _ringMasks;

            /// <summary>
            /// A sliding-window F16 mask [<paramref name="ring"/>, count] of sequence <paramref name="sequence"/> over a ring's
            /// cells, position p in cell p mod ring: row q (position start + q) sees the cells of positions
            /// (start + q - <paramref name="band"/>, start + q] (none before the sequence's attention floor). The ring holds
            /// those positions (ring &gt;= band - 1 + count), so the open cells are one run round the ring. An advance moves it
            /// (<see cref="TryAdvance(Tensor[], int[])"/>); a ring never runs out of cells.
            /// </summary>
            internal int RingBandMask(int ring, int start, int count, int sequence, int band)
            {
                if (band <= 0 || ring < band - 1 + count)
                    throw new ArgumentOutOfRangeException(nameof(ring), ring, $"a ring of {ring} cells does not hold a {band}-key window and {count} rows");
                var key = (ring, start, count, sequence, band);
                if (_ringMasks != null && _ringMasks.TryGetValue(key, out int cached)) return cached;
                var mask = GC.AllocateUninitializedArray<ushort>(checked(ring * count));   // the fill writes every cell
                FillRingBandRows(mask, ring, start, count, FloorOf(sequence), band);
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { ring, count }, IntPtr.Zero, 0);
                AddInput(node, Pin(mask), mask.LongLength * sizeof(ushort), kind: InputKind.DecodeMask, values: mask,
                    window: ring, start: start, count: count, sequence: sequence, band: band, ring: ring);
                (_ringMasks ??= new())[key] = node;
                return node;
            }

            /// <summary>A ring band mask (<see cref="RingBandMask"/>): each row zero over the cells of its window's positions,
            /// -inf elsewhere.</summary>
            private static void FillRingBandRows(ushort[] mask, int ring, int start, int count, int attentionFloor, int band)
            {
                const ushort negInf = 0xFC00;
                mask.AsSpan(0, ring * count).Fill(negInf);
                for (int q = 0; q < count; q++)
                {
                    Span<ushort> row = mask.AsSpan(q * ring, ring);
                    int p = start + q, lo = Math.Max(attentionFloor, p - band + 1);
                    if (lo > p) continue;
                    int cell = lo % ring, n = p - lo + 1;
                    int first = Math.Min(n, ring - cell);
                    row.Slice(cell, first).Clear();
                    if (n > first) row[..(n - first)].Clear();
                }
            }

            /// <summary>Whether a sliding-window mask of this step closes cells inside its window: rows that attend from past the
            /// window's first cell, whose lower bound a recorded step would replay stale (<see cref="BandMask"/>). The floors
            /// rise by one a row, so the step's last row decides: a verify window recorded below the band whose last rows
            /// cross it would otherwise replay their floor as the first row's.</summary>
            internal bool SlidingFloorInWindow
            {
                get
                {
                    if (DeviceBandMasks) return false;   // the executor reads the bands where they are, at every run
                    foreach (InputMeta m in _inputMeta)
                        if (m.Kind == InputKind.DecodeMask && m.Band > 0 && m.Start + m.Count - m.Band > m.Offset) return true;
                    return false;
                }
            }

            /// <summary>A causal F16 mask [start + count, count] for a chunk of sequence <paramref name="sequence"/> (-1: any)
            /// continuing at <paramref name="start"/>.</summary>
            public int CausalMask(int start, int count, int sequence = -1)
            {
                int floor = FloorOf(sequence);
                var key = (start, count, floor);
                if (_causalMasks.TryGetValue(key, out int cached)) return cached;
                int kv = start + count;
                var mask = GC.AllocateUninitializedArray<ushort>(checked(kv * count));
                new CausalMaskInput(kv, start, count, floor).FillF16(mask);
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, stackalloc long[] { kv, count }, IntPtr.Zero, 0);
                AddInput(node, Pin(mask), mask.LongLength * sizeof(ushort));
                return CacheMask(_causalMasks, key, node);
            }

            /// <summary>Keep a new mask input for every later reader of its key (with
            /// <see cref="SingleMaskCacheForTests"/>, only until the next mask of its kind).</summary>
            private int CacheMask<TKey>(Dictionary<TKey, int> masks, TKey key, int node)
            {
                if (_m.SingleMaskCacheForTests) masks.Clear();
                masks[key] = node;
                _m.CausalMaskInputs++;
                return node;
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

            /// <summary>A causal mask (<see cref="CausalMaskInput"/>'s values) for rows continuing at <paramref name="from"/>, moved on to
            /// <paramref name="to"/>: the cells each row opens since cleared, the rest as they are (the same values a fill at
            /// <paramref name="to"/> writes). A decode window as wide as its slot had been filled whole at every step.</summary>
            private static void OpenCausalRows(ushort[] mask, int width, int from, int to, int count, int attentionFloor)
            {
                int floor = Math.Min(attentionFloor, width);
                for (int q = 0; q < count; q++)
                {
                    int lo = Math.Max(floor, Math.Min(width, from + q + 1)), hi = Math.Min(width, to + q + 1);
                    if (hi > lo) mask.AsSpan(q * width + lo, hi - lo).Clear();
                }
            }

            /// <summary>The open cells [Lo, Hi) of row q of a sliding-window mask over cache rows [offset, offset + width), for rows
            /// continuing at <paramref name="start"/>: positions (start + q - band, start + q], none before the attention floor.</summary>
            private static (int Lo, int Hi) BandRow(int width, int offset, int start, int q, int attentionFloor, int band)
            {
                int p = start + q;
                int lo = Math.Clamp(Math.Max(attentionFloor, p - band + 1) - offset, 0, width);
                int hi = Math.Clamp(p + 1 - offset, 0, width);
                return (lo, Math.Max(lo, hi));
            }

            /// <summary>A sliding-window mask (<see cref="BandMask"/>): each row zero over its open cells, -inf elsewhere.</summary>
            private static void FillBandRows(ushort[] mask, int width, int offset, int start, int count, int attentionFloor, int band)
            {
                const ushort negInf = 0xFC00;
                // Large masks row-parallel, as CausalMaskInput's fill (a chunk of linear sliding-window caches attends its whole length).
                if ((long)width * count >= 1 << 20) System.Threading.Tasks.Parallel.For(0, count, Row);
                else for (int q = 0; q < count; q++) Row(q);

                void Row(int q)
                {
                    Span<ushort> row = mask.AsSpan(q * width, width);
                    (int lo, int hi) = BandRow(width, offset, start, q, attentionFloor, band);
                    row[..lo].Fill(negInf);
                    row[lo..hi].Clear();
                    row[hi..].Fill(negInf);
                }
            }

            /// <summary>A sliding-window mask moved from rows at <paramref name="from"/> to rows at <paramref name="to"/>: per row the
            /// cells the band left closed and the cells it reached opened (the same values a fill at <paramref name="to"/> writes).</summary>
            private static void MoveBandRows(ushort[] mask, int width, int offset, int from, int to, int count, int attentionFloor, int band)
            {
                const ushort negInf = 0xFC00;
                for (int q = 0; q < count; q++)
                {
                    Span<ushort> row = mask.AsSpan(q * width, width);
                    (int oldLo, int oldHi) = BandRow(width, offset, from, q, attentionFloor, band);
                    (int lo, int hi) = BandRow(width, offset, to, q, attentionFloor, band);
                    if (lo >= oldHi || hi <= oldLo)
                    {
                        row[oldLo..oldHi].Fill(negInf);
                        row[lo..hi].Clear();
                        continue;
                    }
                    if (lo > oldLo) row[oldLo..lo].Fill(negInf);
                    if (hi > oldHi) row[oldHi..hi].Clear();
                }
            }

            private void AddInput(int node, IntPtr data, long bytes, object handle = null, InputKind kind = InputKind.Opaque,
                Array values = null, int window = 0, int start = 0, int count = 0, int sequence = -1, int offset = 0, int band = 0, int ring = 0)
            {
                _b.Sink?.InputData(node, data, bytes, handle);
                _inputs.Add(node);
                _inputData.Add(data);
                _inputBytes.Add(bytes);
                _inputHandles.Add(handle);
                _inputMeta.Add(new InputMeta(kind, values, window, start, count, Sequence: sequence, Offset: offset, Band: band, Ring: ring));
            }

            private readonly List<object> _inputHandles = new();

            // ---- advancing a decode step

            /// <summary>How an input follows the rows' positions.</summary>
            private enum InputKind : byte { Opaque, Hidden, Positions, WriteRows, DecodeMask, Constant, Refill }

            /// <summary>A decode mask's <see cref="Window"/> cells cover cache rows [<see cref="Offset"/>, Offset + Window); a
            /// sliding-window mask has its <see cref="Band"/> (0: causal). A mask or write rows over a sliding-window ring of
            /// <see cref="Ring"/> cells (0: a linear cache) follow positions round it.</summary>
            private readonly record struct InputMeta(InputKind Kind, Array Values, int Window, int Start, int Count,
                Func<bool> Refill = null, int Sequence = -1, int Offset = 0, int Band = 0, int Ring = 0);

            /// <summary>An input the family rewrites when the step runs again one position on: <paramref name="refill"/>
            /// runs after the positions moved (through <see cref="RewriteInput"/>, or into the input's own array) and
            /// returns false where the step would emit another graph there, which refuses the advance.</summary>
            internal void RefillOnAdvance(int node, Func<bool> refill)
            {
                int i = _inputs.LastIndexOf(node);
                if (i < 0) throw new InvalidOperationException($"node {node} is not an input of this step.");
                _inputMeta[i] = _inputMeta[i] with { Kind = InputKind.Refill, Refill = refill };
            }

            /// <summary>Overwrite an I32 or I64 input's values in place (its bound array).</summary>
            internal void RewriteInput(int node, ReadOnlySpan<int> values)
            {
                int i = _inputs.LastIndexOf(node);
                switch (i < 0 ? null : _inputMeta[i].Values)
                {
                    case int[] ints when ints.Length >= values.Length: values.CopyTo(ints); break;
                    case long[] longs when longs.Length >= values.Length:
                        for (int k = 0; k < values.Length; k++) longs[k] = values[k];
                        break;
                    default: throw new InvalidOperationException($"node {node} is not an I32/I64 input of this step that holds {values.Length} values.");
                }
            }

            private readonly List<InputMeta> _inputMeta = new();

            /// <summary>Every input follows positions in a known way, so the step can be run again one position on.</summary>
            internal bool Advanceable => !_inputMeta.Exists(m => m.Kind == InputKind.Opaque);

            /// <summary>The executor fuses regions at any row count (<see cref="GraphCapabilities.RegionsAtAnyWidth"/>): a family
            /// marks its prompt-width regions for decode rows too.</summary>
            internal bool RegionsAtAnyWidth => _m.GraphExecutor?.Capabilities.RegionsAtAnyWidth == true;

            private List<(int Start, int Max, int Sequence)> _advanceLimits;

            /// <summary>The graph a trace builds for sequence <paramref name="sequence"/> continuing at <paramref name="start"/>
            /// stays the same only up to <paramref name="maxStart"/> (a family whose graph changes past a position: QSA attends
            /// densely only within its budget); an advance past it is refused.</summary>
            internal void LimitAdvance(int start, int maxStart, int sequence) => (_advanceLimits ??= new()).Add((start, maxStart, sequence));

            /// <summary>Rewrite this step's inputs for every row one position on, with <paramref name="hidden"/> as its
            /// rows (its row inputs in order; one tensor binds them all): positions and write rows plus one, decode masks
            /// one column wider (each must stay inside its window and every limit honored: false, nothing changed, when
            /// one would not).</summary>
            internal bool TryAdvance(Tensor[] hidden, int delta = 1)
            {
                var deltas = new int[Starts.Length];
                Array.Fill(deltas, delta);
                return TryAdvance(hidden, deltas);
            }

            /// <summary><see cref="TryAdvance(Tensor[], int)"/> with each sequence moving on by its own
            /// <paramref name="deltas"/>[s] (a batched verify's sequences keep different rows): each sequence's rows' positions,
            /// write rows, masks and limits by its own; an input no one sequence owns refuses deltas that differ.</summary>
            internal bool TryAdvance(Tensor[] hidden, int[] deltas)
            {
                int hiddenAt = 0;
                if (deltas.Length != Starts.Length || deltas.Length == 0) return false;
                bool uniform = true;
                foreach (int d in deltas)
                {
                    if (d < 1) return false;
                    uniform &= d == deltas[0];
                }
                // A sequence's input moves by its delta; one no sequence owns, by the deltas' common value (-1: they differ).
                int Delta(int sequence) => sequence >= 0 && sequence < deltas.Length ? deltas[sequence] : uniform ? deltas[0] : -1;
                foreach (InputMeta m in _inputMeta)
                {
                    if (m.Kind is InputKind.DecodeMask or InputKind.WriteRows && Delta(m.Sequence) < 1) return false;
                    if (m.Kind == InputKind.DecodeMask && m.Ring == 0 && m.Start + m.Count + Delta(m.Sequence) > m.Offset + m.Window) return false;
                    if (m.Kind == InputKind.Positions && !uniform && m.Values.Length != Rows) return false;
                }
                if (_advanceLimits != null)
                {
                    foreach ((int start, int max, int sequence) in _advanceLimits)
                        if (Delta(sequence) < 1 || start + Delta(sequence) > max) return false;
                    for (int i = 0; i < _advanceLimits.Count; i++)
                        _advanceLimits[i] = _advanceLimits[i] with { Start = _advanceLimits[i].Start + Delta(_advanceLimits[i].Sequence) };
                }
                for (int i = 0; i < _inputMeta.Count; i++)
                {
                    InputMeta m = _inputMeta[i];
                    switch (m.Kind)
                    {
                        case InputKind.Hidden:
                        {
                            Tensor rows = hidden[Math.Min(hiddenAt++, hidden.Length - 1)];
                            _inputData[i] = _m.GraphExecutor.Capabilities.BindsHostPointers ? TensorComputePrimitives.GetStoragePointer(rows) : IntPtr.Zero;
                            _inputHandles[i] = rows;
                            break;
                        }
                        case InputKind.Positions:
                        {
                            Span<int> positions = ((int[])m.Values).AsSpan();
                            if (uniform)
                            {
                                foreach (ref int p in positions) p += deltas[0];
                                break;
                            }
                            // Every row's position, in sequence order: each sequence's rows by its own delta.
                            for (int s = 0; s < deltas.Length; s++)
                                foreach (ref int p in positions.Slice(First[s], Counts[s])) p += deltas[s];
                            break;
                        }
                        case InputKind.WriteRows:
                        {
                            int d = Delta(m.Sequence);
                            if (m.Ring > 0)
                                foreach (ref long r in ((long[])m.Values).AsSpan()) r = (r + d) % m.Ring;
                            else
                                foreach (ref long r in ((long[])m.Values).AsSpan()) r += d;
                            break;
                        }
                        case InputKind.DecodeMask:
                        {
                            int d = Delta(m.Sequence);
                            if (m.Ring > 0)
                                FillRingBandRows((ushort[])m.Values, m.Ring, m.Start + d, m.Count, FloorOf(m.Sequence), m.Band);
                            else if (m.Band > 0)
                                MoveBandRows((ushort[])m.Values, m.Window, m.Offset, m.Start, m.Start + d, m.Count, FloorOf(m.Sequence), m.Band);
                            else
                                OpenCausalRows((ushort[])m.Values, m.Window, m.Start, m.Start + d, m.Count, FloorOf(m.Sequence));
                            _inputMeta[i] = m with { Start = m.Start + d };
                            break;
                        }
                    }
                }
                for (int s = 0; s < Starts.Length; s++) Starts[s] += deltas[s];
                foreach (InputMeta m in _inputMeta)
                    if (m.Kind == InputKind.Refill && !m.Refill()) return false;
                return true;
            }

            /// <summary>
            /// <see cref="TryAdvance(Tensor[], int[])"/> for a step of one sequence moved to <paramref name="starts"/>[0] wherever
            /// that is, behind its last position too: the positions and write rows by the difference, every refill run, the rows
            /// rebound. Only a step whose inputs follow from its position alone takes it (a block drafter's ring steps): one with
            /// a decode mask (opened in place, never closed), an opaque input or an advance limit refuses before anything changes
            /// (false); a refill that refuses returns false too, and the caller drops the step.
            /// </summary>
            internal bool TryRebase(Tensor[] hidden, int[] starts)
            {
                if (starts.Length != 1 || Starts.Length != 1 || _advanceLimits != null
                    || _inputMeta.Exists(m => m.Kind is InputKind.DecodeMask or InputKind.Opaque))
                    return false;
                int delta = starts[0] - Starts[0], hiddenAt = 0;
                for (int i = 0; i < _inputMeta.Count; i++)
                {
                    InputMeta m = _inputMeta[i];
                    switch (m.Kind)
                    {
                        case InputKind.Hidden:
                        {
                            Tensor rows = hidden[Math.Min(hiddenAt++, hidden.Length - 1)];
                            _inputData[i] = _m.GraphExecutor.Capabilities.BindsHostPointers ? TensorComputePrimitives.GetStoragePointer(rows) : IntPtr.Zero;
                            _inputHandles[i] = rows;
                            break;
                        }
                        case InputKind.Positions:
                            foreach (ref int p in ((int[])m.Values).AsSpan()) p += delta;
                            break;
                        case InputKind.WriteRows:
                            foreach (ref long r in ((long[])m.Values).AsSpan()) r += delta;
                            break;
                    }
                }
                Starts[0] = starts[0];
                foreach (InputMeta m in _inputMeta)
                    if (m.Kind == InputKind.Refill && !m.Refill()) return false;
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
            /// own product, so the result is contiguous. <paramref name="alone"/>: no other block of the weight is read with
            /// these rows (a draft head scoring part of the vocabulary), so an executor may compute the block alone instead
            /// of the whole product its blocks share.</summary>
            public int MatMulRows(string weight, long firstRow, long rows, int x, bool alone = false)
            {
                int w = Weight(weight);
                long ne0, rowBytes;
                float scale = 1.0f;
                QuantizedWeight q = Rank >= 0 && _m._tpQuantWeights.TryGetValue(weight, out QuantizedWeight[] shards) ? shards[Rank]
                    : _m._quantWeights.GetValueOrDefault(weight);
                if (q != null)
                {
                    ne0 = q.Ne0;
                    rowBytes = q.RawBytes / q.Ne1;
                    scale = q.Scale;
                }
                else
                {
                    Tensor t = Rank >= 0 && _m._tpWeights.TryGetValue(weight, out Tensor[] tensors) ? tensors[Rank] : _m._weights[weight];
                    ne0 = t.Sizes[t.DimensionCount - 1];
                    rowBytes = ne0 * sizeof(float);
                }
                int block = Op(GraphOp.View, stackalloc[] { w }, stackalloc long[] { 2, ne0, rows, 1, 1, rowBytes, 0, 0, firstRow * rowBytes });
                int y = alone ? Op(GraphOp.MulMat, stackalloc[] { block, x }, stackalloc long[] { 0, 1 }) : Op(GraphOp.MulMat, block, x);
                if (scale != 1.0f)
                    y = Op(GraphOp.Scale, stackalloc[] { y }, default, stackalloc[] { scale });
                return y;
            }

            public int Reshape(int x, params long[] ne) => Op(GraphOp.Reshape, stackalloc[] { x }, ne);

            /// <summary>An F16 input [ne...] read from <paramref name="values"/> every run (a family's mask).</summary>
            internal int InputF16(ushort[] values, params long[] ne)
            {
                int node = Leaf(GraphLeafRole.Input, GraphFormat.F16, ne, IntPtr.Zero, 0);
                AddInput(node, Pin(values), values.LongLength * sizeof(ushort), values: values);
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
            /// lets a backend attend only those (ggml_flash_attn_ext_set_n_kv_max). <paramref name="promptChunk"/>: the rows are one
            /// sequence's prompt chunk (<see cref="GraphAttentionRows.PromptChunk"/>), set only by <see cref="EmitAttention"/> under
            /// <see cref="PromptChunks"/>.</summary>
            /// <param name="band">A sliding-window layer's window (ip[3]; ip[2] is then the rows' kind, see
            /// <see cref="GraphAttentionRows"/>) where the executor reads such masks on the device
            /// (<see cref="GraphCapabilities.BandMasksOnDevice"/>; <see cref="DeviceBandMasks"/>), 0 elsewhere.</param>
            public int FlashAttention(int q, int k, int v, int mask, float scale, int kvMax = 0, bool promptChunk = false, int band = 0)
            {
                Span<int> srcs = stackalloc[] { q, k, v, mask };
                // ip[2] only on a prompt chunk's op or a device band's (GraphAttentionRows), ip[3] only on a device band's: every
                // other op keeps the ip, bytes and signature it had.
                long rows = promptChunk ? GraphAttentionRows.PromptChunk : GraphAttentionRows.Unsaid;
                ReadOnlySpan<long> ip = band > 0 ? stackalloc long[] { 10 /* F32 accumulation */, Math.Max(kvMax, 0), rows, band }
                    : promptChunk ? stackalloc long[] { 10, Math.Max(kvMax, 0), rows }
                    : kvMax > 0 ? stackalloc long[] { 10, kvMax } : stackalloc long[] { 10 };
                if (band > 0) DeviceBandMasks = true;
                _m.AttentionIpForTests?.Add(ip.ToArray());
                return Op(GraphOp.FlashAttnExt, mask >= 0 ? srcs : srcs[..3], ip, stackalloc[] { scale, 0f, 0f });
            }

            /// <summary>The step's sliding-window attentions read their masks on the device (their band passed to the executor,
            /// <see cref="GraphCapabilities.BandMasksOnDevice"/>): a recording replays each position's band.</summary>
            internal bool DeviceBandMasks { get; private set; }

            private InvalidOperationException Reject(string why)
            {
                Rejected = true;
                return new InvalidOperationException(why);
            }

            // ---- native

            /// <summary>Build the trace as a native graph (node handles equal trace indices).</summary>
            /// <summary>Compile the recorded program with the backend's executor plugin.</summary>
            /// <summary>The recorded program, not compiled (a stream's template).</summary>
            internal GraphProgram Record()
            {
                if (!_record) throw new InvalidOperationException("a hash-only trace records no program");
                return Program = _b.ToProgram(false, false, _m._graphTpRanks);
            }

            internal CachedGraph Build(bool replayable, bool reorder)
            {
                if (!_record) throw new InvalidOperationException("a hash-only trace cannot be built");
                GraphProgram program = _b.ToProgram(replayable, reorder, _m._graphTpRanks);
                Program = program;
                return new CachedGraph { Plan = _m.GraphExecutor.Compile(program) };
            }

            /// <summary>Compile this rank's recorded program for device <paramref name="device"/> of a compiling executor.</summary>
            internal CachedGraph BuildForRank(IRankedGraphExecutor executor, int device, bool replayable)
            {
                if (!_record) throw new InvalidOperationException("a hash-only trace cannot be built");
                GraphProgram program = _b.ToProgram(replayable, false, _m._graphTpRanks);
                Program = program;
                return new CachedGraph { Plan = executor.CompileForRank(program, device) };
            }

            /// <summary>This step's per-run inputs, for a run of several ranks' plans together.</summary>
            internal GraphRunInputs RunInputs() => new(_inputs.ToArray(), _inputData.ToArray(), _inputBytes.ToArray());

            internal bool Run(IGraphPlan plan) => plan.Run(CollectionsMarshal.AsSpan(_inputs), CollectionsMarshal.AsSpan(_inputData),
                CollectionsMarshal.AsSpan(_inputBytes), CollectionsMarshal.AsSpan(_inputHandles));

            private object[] _unstable;

            /// <summary>Bind every input to its <see cref="IDecodeCapture.Stable"/> tensor, uploaded with this step's
            /// values (the embedded rows copied on the device): a recorded step reads its inputs there. False when an
            /// input is not one a recording can rebind (it then runs as usual).</summary>
            internal bool BindStable(IDecodeCapture capture)
            {
                if (_unstable != null) throw new InvalidOperationException("inputs already bound to a recording");
                var saved = _inputHandles.ToArray();
                for (int i = 0; i < _inputMeta.Count; i++)
                {
                    InputMeta m = _inputMeta[i];
                    switch (m.Kind)
                    {
                        case InputKind.Hidden when _inputHandles[i] is Tensor hidden:
                        {
                            Tensor stable = capture.Stable(i, DType.Float32, hidden.Sizes.ToArray());
                            Ops.Copy(stable, hidden);
                            _inputHandles[i] = stable;
                            break;
                        }
                        case InputKind.Positions or InputKind.Constant or InputKind.Refill when m.Values is int[] ints:
                        {
                            Tensor stable = capture.Stable(i, DType.Int32, new long[] { ints.Length }, m.Kind == InputKind.Refill);
                            stable.SetElementsAsInt(ints);
                            _inputHandles[i] = new StableInput(stable) { Varies = m.Kind == InputKind.Refill };
                            break;
                        }
                        case InputKind.WriteRows or InputKind.Refill when m.Values is long[] longs:
                        {
                            Tensor stable = capture.Stable(i, DType.Int32, new long[] { longs.Length }, m.Kind == InputKind.Refill);
                            var ints = new int[longs.Length];
                            for (int j = 0; j < ints.Length; j++) ints[j] = checked((int)longs[j]);
                            stable.SetElementsAsInt(ints);
                            _inputHandles[i] = new StableInput(stable) { Varies = m.Kind == InputKind.Refill };
                            break;
                        }
                        case InputKind.Constant or InputKind.Refill when m.Values is float[] values:
                        {
                            Tensor stable = capture.Stable(i, DType.Float32, new long[] { values.Length }, m.Kind == InputKind.Refill);
                            stable.SetElementsAsFloat(values);
                            _inputHandles[i] = new StableInput(stable) { Varies = m.Kind == InputKind.Refill };
                            break;
                        }
                        case InputKind.DecodeMask when m.Values is ushort[] half:
                        {
                            Tensor stable = capture.Stable(i, DType.Float32, new long[] { half.Length });
                            BindDecodeMask(i, m, half, stable);
                            _inputHandles[i] = new StableInput(stable);
                            break;
                        }
                        case InputKind.Refill when m.Values is ushort[] half:
                        {
                            Tensor stable = capture.Stable(i, DType.Float32, new long[] { half.Length }, true);
                            var floats = new float[half.Length];
                            for (int j = 0; j < floats.Length; j++) floats[j] = (float)BitConverter.UInt16BitsToHalf(half[j]);
                            stable.SetElementsAsFloat(floats);
                            _inputHandles[i] = new StableInput(stable) { Varies = true };
                            break;
                        }
                        default:
                            for (int j = 0; j < saved.Length; j++) _inputHandles[j] = saved[j];
                            return false;
                    }
                }
                _unstable = saved;
                return true;
            }

            /// <summary>The stable tensor each decode mask input was last bound to, and the rows' start its values hold.</summary>
            private Dictionary<int, (Tensor Stable, int Start)> _maskBindings;

            private readonly Dictionary<int, float[]> _zeroRuns = new();

            /// <summary>
            /// A decode mask's values as F32 in its recording's stable tensor: whole when the tensor is new to this step, then only
            /// the cells its rows opened since its last binding (the tensor's host copy holds the rest: nothing writes it on the
            /// device), patched on both copies where the storage can (<see cref="Storage.PatchElementsAsFloat"/>), so the launch
            /// uploads only them. A window as wide as its slot had been converted whole at every step: a 4-row window of a
            /// 16K-cell slot 64K values, where a decode step opens one cell a row.
            /// </summary>
            private void BindDecodeMask(int input, in InputMeta m, ushort[] half, Tensor stable)
            {
                // A sliding-window mask also closes cells as it moves: it is converted whole.
                if (m.Band == 0 && _maskBindings != null && _maskBindings.TryGetValue(input, out var bound) && ReferenceEquals(bound.Stable, stable)
                    && bound.Start <= m.Start)
                {
                    int floor = Math.Min(FloorOf(m.Sequence), m.Window);
                    for (int q = 0; q < m.Count; q++)
                    {
                        int lo = Math.Max(floor, Math.Min(m.Window, bound.Start + q + 1)), hi = Math.Min(m.Window, m.Start + q + 1);
                        if (hi <= lo) continue;
                        if (!_zeroRuns.TryGetValue(hi - lo, out float[] zeros)) _zeroRuns[hi - lo] = zeros = new float[hi - lo];
                        stable.Storage.PatchElementsAsFloat(stable.StorageOffset + (long)q * m.Window + lo, zeros);
                    }
                }
                else
                {
                    var floats = new float[half.Length];
                    for (int j = 0; j < floats.Length; j++) floats[j] = (float)BitConverter.UInt16BitsToHalf(half[j]);
                    stable.SetElementsAsFloat(floats);
                }
                (_maskBindings ??= new())[input] = (stable, m.Start);
            }

            /// <summary>Restore the inputs <see cref="BindStable"/> rebound.</summary>
            internal void UnbindStable()
            {
                if (_unstable == null) return;
                for (int j = 0; j < _unstable.Length; j++) _inputHandles[j] = _unstable[j];
                _unstable = null;
            }

            public void Dispose()
            {
                foreach (GCHandle pin in _pins) pin.Free();
                _pins.Clear();
            }
        }
    }
}
