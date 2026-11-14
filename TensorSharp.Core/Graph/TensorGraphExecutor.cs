// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
// The interpreter passes null result tensors to Ops (the op registry allocates the result), which
// Ops' signatures do not annotate; nullable analysis is off for this file.
#nullable disable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;

namespace TensorSharp.Graph
{
    /// <summary>
    /// A graph executor for backends whose kernels are tensor operations (the managed CPU, direct
    /// CUDA, MLX): it runs a program node by node on <see cref="Tensor"/>s. ggml's extents
    /// (innermost first) and byte strides become tensor views over the same storage (sizes
    /// outermost first, strides in elements), so cache and state writes land in the leaves' own
    /// tensors. Generic ops go through <see cref="Ops"/> (each backend's op registry); a backend
    /// overrides the kernels it has its own implementation of (quantized products, attention,
    /// the recurrences).
    /// </summary>
    public abstract class TensorGraphExecutor : IStreamingGraphExecutor
    {
        protected TensorGraphExecutor(IAllocator allocator)
            => Allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));

        public IAllocator Allocator { get; }

        /// <summary>The tensor-parallel group this executor's device belongs to, for <see cref="GraphOp.AllReduce"/>
        /// (null: a program with a reduce is rejected).</summary>
        public IGraphCollective Collective { get; private set; }

        /// <summary>This executor's rank in <see cref="Collective"/>.</summary>
        public int CollectiveRank { get; private set; }

        /// <summary>Make this executor rank <paramref name="rank"/> of <paramref name="collective"/>.</summary>
        public void JoinCollective(IGraphCollective collective, int rank)
        {
            if (collective != null && (rank < 0 || rank >= collective.Degree)) throw new ArgumentOutOfRangeException(nameof(rank));
            Collective = collective;
            CollectiveRank = rank;
        }

        /// <summary>Views are tensor views here (any strides), and a program is cheap to compile, so
        /// decode windows are exact (no padded rows to attend over).</summary>
        /// <summary>Exact decode windows; prompt chunks are written into the cache by row index and attend
        /// over its window, as the per-op paths do (the backends' cache kernels write and read it).</summary>
        public virtual GraphCapabilities Capabilities { get; } = new() { StridedViews = true, DecodeWindowRows = 1, Replays = false, IndexedCacheWrites = true, HostExperts = true };

        /// <summary>The backend's binary kernels take a broadcast (stride-0) operand; otherwise it is
        /// gathered into a contiguous tensor first.</summary>
        protected virtual bool BroadcastsInKernels => true;

        /// <summary>A window of a K/V cache an attention operand is: the cache tensor (sizes [heads, capacity,
        /// head_dim]) and its first row; null where the operand is not a cache window.</summary>
        protected sealed record CacheWindow(Tensor Cache, long Start);

        /// <summary>Write rows <paramref name="first"/> .. + count of <paramref name="dst"/> (along ne1) from
        /// <paramref name="src"/> ([.., count, ne0], F32) with a backend kernel; false to use copies.</summary>
        protected virtual bool TryWriteRows(Tensor dst, Tensor src, int first, int count) => false;

        /// <summary>The backend's element-wise kernels read strided views (no contiguous copy first).</summary>
        protected virtual bool KernelsReadStridedViews => false;

        /// <summary>Whether a fused region of <paramref name="kind"/> takes its tensor inputs as they are (views of a packed
        /// projection, say) rather than contiguous copies: <see cref="TryFused"/> then handles (or declines) any layout.</summary>
        protected virtual bool RegionReadsStridedInputs(GraphFusion kind) => KernelsReadStridedViews;

        private Tensor In(Tensor t, RunState run) => KernelsReadStridedViews ? t : Contig(t, run);

        private static readonly HashSet<GraphOp> s_ops = new()
        {
            GraphOp.GetRows, GraphOp.Add, GraphOp.Sub, GraphOp.Mul, GraphOp.Div, GraphOp.Scale, GraphOp.RmsNorm,
            GraphOp.L2Norm, GraphOp.MulMat, GraphOp.MulMatId, GraphOp.Silu, GraphOp.Sigmoid, GraphOp.Softplus,
            GraphOp.Exp, GraphOp.Neg, GraphOp.Tanh, GraphOp.Relu, GraphOp.SwigluSplit, GraphOp.Swiglu, GraphOp.Reshape,
            GraphOp.View, GraphOp.Permute, GraphOp.Transpose, GraphOp.Cont, GraphOp.Cast, GraphOp.Cpy, GraphOp.SetRows,
            GraphOp.RopeExt, GraphOp.RopeMulti, GraphOp.FlashAttnExt, GraphOp.SoftMaxExt, GraphOp.TopK, GraphOp.SumRows,
            GraphOp.SsmConv, GraphOp.GatedDeltaNet, GraphOp.Concat, GraphOp.KvWindow, GraphOp.Sqr, GraphOp.Sqrt, GraphOp.Clamp,
            GraphOp.LightningIndexer, GraphOp.TopKMask, GraphOp.AllReduce, GraphOp.MoeHostExperts,
        };

        public IGraphPlan Compile(GraphProgram program)
        {
            ArgumentNullException.ThrowIfNull(program);
            if (program.TensorParallelRanks != null)
                throw new NotSupportedException($"{GetType().Name} runs a program on one device.");
            return new Plan(this, program);
        }

        /// <summary>Boundary <paramref name="index"/> (1-based; a layer's output) of <paramref name="count"/>
        /// (-1: not known while streaming) was reached with <paramref name="value"/>: a lazy backend may
        /// start running the work so far.</summary>
        protected virtual void AtBoundary(Tensor value, int index, int count) { }

        public virtual void SyncToHost(IntPtr data, long bytes) { }

        public virtual void Invalidate(IntPtr data) { }

        // ================================================================ kernels a backend provides

        /// <summary>x [n, Ne0] (contiguous F32) times the quantized weight's rows: [n, Ne1].</summary>
        protected abstract Tensor QuantizedMatMul(IGraphWeight weight, Tensor x);

        /// <summary>The managed product of a weight's host blocks with host rows (x [rows, Ne0] -> [rows, Ne1]),
        /// registered by the assembly that holds the managed kernels. A backend without a kernel for a weight's
        /// format runs it through here, as its per-op path does.</summary>
        public static Action<IGraphWeight, float[], int, float[]> HostQuantizedProduct { get; set; }

        /// <summary><see cref="HostQuantizedProduct"/> on <paramref name="x"/>'s rows (copied to the host and back).</summary>
        protected Tensor HostProduct(IGraphWeight weight, Tensor x)
        {
            if (HostQuantizedProduct == null || weight.Data == IntPtr.Zero)
                throw new NotSupportedException($"{GetType().Name} has no product for weight type {weight.Format}, and no host copy to run it on");
            int rows = (int)x.Sizes[0];
            float[] input = x.GetElementsAsFloat((int)x.ElementCount());
            var output = new float[(long)rows * weight.Ne1];
            HostQuantizedProduct(weight, input, rows, output);
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            result.SetElementsAsFloat(output);
            return result;
        }

        /// <summary>x [n, Ne0] times expert <paramref name="expert"/> of stacked weights: [n, Ne1].</summary>
        protected virtual Tensor ExpertMatMul(IGraphExpertWeights weights, int expert, Tensor x)
            => QuantizedMatMul(weights.Expert(expert), x);

        /// <summary>Drop the device copy an expert streamed in for one step left behind (its experts are not the
        /// layer's tier). Executors that keep device copies of weights release it; the default keeps nothing.</summary>
        protected virtual void ReleaseStreamedExpert(IGraphWeight expert) { }

        /// <summary>Attention of q [heads, nq, d] over k [kvHeads, nkv, d] / v [kvHeads, nkv, dv] with an
        /// additive mask [nq, nkv] (or null); query head h reads KV head h / (heads / kvHeads). The
        /// operands may be strided views (the K/V windows are views of the caches). Returns
        /// [heads, nq, dv].</summary>
        protected virtual Tensor Attention(Tensor q, Tensor k, Tensor v, Tensor mask, float scale)
        {
            var owned = new List<Tensor>();
            try
            {
                q = Contig(F32(q, owned), owned);
                k = Contig(F32(k, owned), owned);
                v = Contig(F32(v, owned), owned);
                int heads = (int)q.Sizes[0], kvHeads = (int)k.Sizes[0], group = heads / kvHeads;
                long nq = q.Sizes[1], nkv = k.Sizes[1], dv = v.Sizes[2];
                Tensor kh = group == 1 ? k : RepeatHeads(k, group, owned);
                Tensor vh = group == 1 ? v : RepeatHeads(v, group, owned);
                using Tensor kt = kh.Transpose(1, 2);
                var scores = new Tensor(Allocator, DType.Float32, heads, nq, nkv);
                owned.Add(scores);
                Ops.AddmmBatch(scores, 0, scores, scale, q, kt);
                if (mask != null)
                {
                    Tensor m = Contig(F32(mask, owned), owned);
                    using Tensor m3 = m.View(1, nq, nkv);
                    using Tensor me = m3.Expand(heads, nq, nkv);
                    Ops.Add(scores, scores, me);
                }
                Ops.Softmax(scores, scores);
                var result = new Tensor(Allocator, DType.Float32, heads, nq, dv);
                Ops.AddmmBatch(result, 0, result, 1.0f, scores, vh);
                return result;
            }
            finally
            {
                foreach (Tensor t in owned) t.Dispose();
            }
        }

        /// <summary>ggml's flash attention: q [heads, nq, d], k/v [kvHeads, nkv, d] (views), the mask [nq, nkv]
        /// with its pattern; returns ggml's layout [nq, heads, dv]. Default: <see cref="Attention"/>, permuted.</summary>
        protected virtual Tensor FlashAttention(Tensor q, Tensor k, Tensor v, AttentionMask mask, float scale,
            CacheWindow kWindow, CacheWindow vWindow)
        {
            using Tensor a = Attention(q, k, v, mask.Pattern == MaskPattern.None && mask.Floor == 0 ? null : mask.Tensor, scale);
            using Tensor p = a.Permute(1, 0, 2);
            return Ops.NewContiguous(p);
        }

        private Tensor RepeatHeads(Tensor t, int group, List<Tensor> owned)
        {
            // [h, n, d] -> [h * group, n, d], each head repeated group times (block mapping).
            using Tensor v4 = t.View(t.Sizes[0], 1, t.Sizes[1], t.Sizes[2]);
            using Tensor e = v4.Expand(t.Sizes[0], group, t.Sizes[1], t.Sizes[2]);
            Tensor c = Ops.NewContiguous(e);
            owned.Add(c);
            Tensor r = c.View(t.Sizes[0] * group, t.Sizes[1], t.Sizes[2]);
            owned.Add(r);
            return r;
        }

        /// <summary>
        /// ggml's gated delta rule: q, k [n_s, n_t, Hk, S] and v [n_s, n_t, H, S] (contiguous), g
        /// [n_s, n_t, H, 1 or S], beta [n_s, n_t, H, 1], state [n_s, H, S, S] (row j of a head's state
        /// holds column j of S). Returns ggml's result as one F32 vector: the outputs [n_s, n_t, H, S],
        /// then <paramref name="snapshots"/> state slots [n_s, H, S, S] (slot 0 the newest; a slot older
        /// than the step's tokens is left zero). Default: on the host, as ggml-cpu computes it.
        /// </summary>
        protected virtual Tensor GatedDeltaNet(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
            => GatedDeltaNetHost(q, k, v, g, beta, state, snapshots);

        /// <summary>ggml_ssm_conv: x [n_s, channels, width - 1 + n_t] (time innermost), kernel
        /// [channels, width]: [n_s, n_t, channels].</summary>
        protected virtual Tensor SsmConv(Tensor x, Tensor kernel)
        {
            long ns = x.Sizes[0], channels = x.Sizes[1], width = kernel.Sizes[1];
            long nt = x.Sizes[2] - width + 1;
            var acc = new Tensor(Allocator, DType.Float32, ns, channels, nt);
            try
            {
                for (long kk = 0; kk < width; kk++)
                {
                    using Tensor xs = x.Narrow(2, kk, nt);
                    using Tensor w = kernel.Narrow(1, kk, 1);
                    using Tensor w3 = Ops.NewContiguous(w);
                    using Tensor w3v = w3.View(1, channels, 1);
                    using Tensor we = w3v.Expand(ns, channels, nt);
                    if (kk == 0) Ops.Mul(acc, xs, we);
                    else
                    {
                        using Tensor prod = Ops.Mul(null, xs, we);
                        Ops.Add(acc, acc, prod);
                    }
                }
                using Tensor p = acc.Permute(0, 2, 1);
                return Ops.NewContiguous(p);
            }
            finally
            {
                acc.Dispose();
            }
        }

        // ================================================================ the plan

        /// <summary>A node's value: a tensor (sizes outermost first), a quantized weight (or a block of
        /// its rows), and for index tensors their host values.</summary>
        protected readonly struct Value
        {
            public Tensor Tensor { get; init; }
            public IGraphWeight Weight { get; init; }
            public long FirstRow { get; init; }
            public long Rows { get; init; }
            public int[] Ints { get; init; }
            /// <summary>Host values of an F32/F16 input (a mask's pattern is read from them).</summary>
            public float[] Floats { get; init; }
            /// <summary>Sizes of a host-only input (no tensor until an op reads it).</summary>
            public long[] HostSizes { get; init; }
            /// <summary>For a cache window (KvWindow): the cache it views and its first row.</summary>
            public CacheWindow Window { get; init; }
            /// <summary>The tensor's memory is laid out unlike ggml's node (a strided view where ggml has a
            /// contiguous result): a view, whose offsets and strides assume ggml's layout, copies it first.</summary>
            public bool Relaid { get; init; }
        }

        /// <summary>What an attention mask is, read from its host values: none (every key visible),
        /// causal aligned to the last key, or anything else (applied as given). Keys before
        /// <see cref="Floor"/> are masked for every query in the first two cases.</summary>
        protected enum MaskPattern { None, Causal, Explicit }

        protected sealed class AttentionMask
        {
            private readonly Func<Tensor> _make;
            private Tensor _tensor;

            public AttentionMask(MaskPattern pattern, int floor, Func<Tensor> make, int keys = -1,
                MaskPattern trimmed = MaskPattern.Explicit, int trimmedFloor = 0)
            {
                Pattern = pattern;
                Floor = floor;
                Keys = keys;
                Trimmed = trimmed;
                TrimmedFloor = trimmedFloor;
                _make = make;
            }

            /// <summary>The most keys a query attends (a sparse attention's budget: the op's hint), 0 when unknown.</summary>
            public int KvMax { get; set; }

            /// <summary>The mask [nq, nkv] as a device tensor (uploaded on first use: a kernel that runs the
            /// pattern never needs it).</summary>
            public Tensor Tensor => _tensor ??= _make?.Invoke();
            public bool HasTensor => _make != null;
            public MaskPattern Pattern { get; }
            public int Floor { get; }

            /// <summary>Keys before the trailing ones every query masks (a padded decode window's unwritten cells), -1 when
            /// unknown.</summary>
            public int Keys { get; }

            /// <summary>The pattern over keys [<see cref="TrimmedFloor"/>, <see cref="Keys"/>) alone (nothing past Keys is
            /// attended), for a kernel that stops at Keys: a padded window's causal mask is Explicit over the whole window
            /// but causal over its written cells.</summary>
            public MaskPattern Trimmed { get; }
            public int TrimmedFloor { get; }

            public static readonly AttentionMask None = new(MaskPattern.None, 0, null);
        }

        private static AttentionMask Classify(Func<Tensor> mask, float[] host, long rows, long keys)
        {
            if (mask == null) return AttentionMask.None;
            if (host == null || host.LongLength < rows * keys) return new AttentionMask(MaskPattern.Explicit, 0, mask);
            // The leading keys every query masks.
            int floor = 0;
            while (floor < keys)
            {
                bool all = true;
                for (long r = 0; r < rows && all; r++) all = float.IsNegativeInfinity(host[r * keys + floor]);
                if (!all) break;
                floor++;
            }
            // The trailing keys every query masks (a padded window's unwritten cells): the pattern is judged before them.
            long valid = keys;
            while (valid > floor)
            {
                bool all = true;
                for (long r = 0; r < rows && all; r++) all = float.IsNegativeInfinity(host[r * keys + valid - 1]);
                if (!all) break;
                valid--;
            }
            bool none = true, causal = true;
            for (long r = 0; r < rows && (none || causal); r++)
            {
                long last = valid - rows + r;   // the last key row r sees when causal
                for (long c = floor; c < valid; c++)
                {
                    float m = host[r * keys + c];
                    none &= m == 0f;
                    causal &= c <= last ? m == 0f : float.IsNegativeInfinity(m);
                }
            }
            MaskPattern pattern = none ? MaskPattern.None : causal ? MaskPattern.Causal : MaskPattern.Explicit;
            int trimmedFloor = pattern == MaskPattern.Explicit ? 0 : floor;
            // Over fewer keys than the window the pattern holds only for kernels that stop at Keys (Trimmed); for every
            // other reader the mask stays Explicit, as it was.
            if (pattern != MaskPattern.Explicit && valid < keys)
                return new AttentionMask(MaskPattern.Explicit, 0, mask, (int)valid, pattern, trimmedFloor);
            return new AttentionMask(pattern, trimmedFloor, mask, (int)valid, pattern, trimmedFloor);
        }

        protected sealed class RunState
        {
            public readonly List<Tensor> Owned = new();
            /// <summary>Full products of a weight with a node's rows, shared by the row blocks a
            /// packed weight is split into.</summary>
            public readonly Dictionary<(IGraphWeight, int), Tensor> Products = new();
            /// <summary>Per-row rotary positions by (positions node, heads): every layer rotates with the same.</summary>
            public readonly Dictionary<(int, int), Tensor> RowPositions = new();
            /// <summary>A mask node's pattern, read once per step (every layer attends under it).</summary>
            public readonly Dictionary<(int, long), AttentionMask> Masks = new();

            /// <summary>Tensors that live until the step ends (caches several nodes read: uploaded inputs,
            /// rotary positions, masks); <see cref="Owned"/> is the current node's scratch.</summary>
            public readonly List<Tensor> Lasting = new();

            public Tensor Keep(Tensor t) { Owned.Add(t); return t; }

            public Tensor KeepLasting(Tensor t) { Lasting.Add(t); return t; }
        }

        /// <summary>How many times each node of a program is read (ops, outputs, boundaries), from a recorded
        /// program; a session releases a node's tensor when its last reader has run.</summary>
        private sealed class Liveness
        {
            private const int Forever = 1 << 28;
            public readonly int Count;
            public readonly int[] Uses;
            private readonly GraphNode[] _nodes;

            public Liveness(GraphProgram program)
            {
                Count = program.NodeCount;
                Uses = new int[Count];
                _nodes = new GraphNode[Count];
                int n = 0;
                foreach (GraphNode e in program.Entries)
                {
                    switch (e.Kind)
                    {
                        case GraphNodeKind.Leaf:
                            _nodes[n++] = e;
                            break;
                        case GraphNodeKind.Op:
                            _nodes[n++] = e;
                            foreach (int src in e.Srcs) Uses[src]++;
                            break;
                        case GraphNodeKind.Output:
                        case GraphNodeKind.Effect:
                            Uses[e.Node] += Forever;
                            break;
                        case GraphNodeKind.Alias:
                            Uses[e.Node] += Forever;
                            Uses[e.Srcs[1]] += Forever;
                            break;
                        case GraphNodeKind.Boundary:
                            Uses[e.Node]++;
                            break;
                    }
                }
            }

            /// <summary>The template keeps <paramref name="node"/> to the end (an output, effect or alias reads it).</summary>
            public bool Pins(int node) => node < Count && Uses[node] >= Forever;

            /// <summary>The streamed entry is the template's node (so its counts apply).</summary>
            public bool Matches(int node, in GraphNode e)
            {
                if (node >= Count) return false;
                GraphNode t = _nodes[node];
                if (t.Kind != e.Kind || t.Op != e.Op || t.Role != e.Role) return false;
                if ((t.Srcs?.Length ?? 0) != (e.Srcs?.Length ?? 0)) return false;
                for (int i = 0; i < (e.Srcs?.Length ?? 0); i++) if (t.Srcs[i] != e.Srcs[i]) return false;
                return true;
            }
        }

        private readonly System.Runtime.CompilerServices.ConditionalWeakTable<GraphProgram, Liveness> _liveness = new();

        private Liveness LivenessOf(GraphProgram program) => program == null ? null : _liveness.GetValue(program, p => new Liveness(p));

        /// <summary>
        /// The evaluation of one program as its entries arrive (a recorded plan replays its entries; a
        /// stream receives them while the runtime emits). Nodes run in order; a residual Add and an
        /// RMSNorm wait for their consumer, so a norm scaled by a weight runs as one weighted norm and,
        /// where the backend has the kernel, the Add before it joins as well. A waiting node that nothing
        /// consumes never runs.
        /// </summary>
        private sealed class Session
        {
            private const byte None = 0, Pending = 1, Deferred = 2, Done = 3, Lazy = 4;   // Lazy: inside a fused region
            private const byte Skipped = 5;   // a fused region's op its results did not need: it never runs
            private const int Released = int.MinValue / 2;   // a released node's count (later reads never reach 0)

            private readonly TensorGraphExecutor _x;
            private readonly bool[] _dead;
            private readonly Liveness _live;
            private bool _freeing;
            private GraphNode[] _nodes;
            private Value[] _values;
            private byte[] _state;
            private bool[] _fresh;
            private Tensor[] _own;     // the tensor a node's value owns (released after its last reader)
            private int[] _uses;       // readers still to run
            private readonly List<int> _setDuring = new();
            private int _count;
            private readonly Dictionary<int, int> _boundaryPending = new();
            private int _boundaries;
            private readonly List<int> _outputs = new();
            private readonly List<long> _outputFloats = new();
            public readonly RunState Run = new();

            public Session(TensorGraphExecutor executor, bool[] dead = null, Liveness live = null)
            {
                executor.BeginRun();
                _x = executor;
                _dead = dead;
                _live = live;
                _freeing = live != null;
                // The tables of the executor's previous session (a step allocates none).
                Buffers b = Interlocked.Exchange(ref executor._buffers, null) ?? new Buffers();
                (_nodes, _values, _state, _fresh, _own, _uses) = (b.Nodes, b.Values, b.State, b.Fresh, b.Own, b.Uses);
            }

            private GraphNode _region;   // the open fused region this executor has a kernel for (Kind Fused), else default
            private readonly List<int> _regionOps = new();   // the ops the open region deferred, in order
            private int _nested;         // regions opened inside it (part of it: their ends do not close it)
            private bool InRegion => _region.Kind == GraphNodeKind.Fused;

            public void Entry(int node, in GraphNode e)
            {
                switch (e.Kind)
                {
                    case GraphNodeKind.Leaf:
                    case GraphNodeKind.Op:
                    {
                        Add(node, e);
                        if (_freeing)
                        {
                            if (_live.Matches(node, e)) _uses[node] = _live.Uses[node];
                            else _freeing = false;   // the step is not the template's: keep everything to the end
                        }
                        if (_dead != null && _dead[node]) return;
                        GraphNode entry = e;
                        if (e.Kind == GraphNodeKind.Op && InRegion)
                        {
                            _state[node] = Lazy;
                            _regionOps.Add(node);
                            break;
                        }
                        if (e.Kind == GraphNodeKind.Leaf)
                        {
                            if (e.Role == GraphLeafRole.Input) _state[node] = Pending;
                            else Attribute(() => Set(node, _x.BindLeaf(node, entry, null, Run), fresh: false));
                        }
                        else
                        {
                            Attribute(() => Op(node, entry));
                        }
                        break;
                    }
                    case GraphNodeKind.Effect: { int n = e.Node; Pinned(n); Attribute(() => Force(n)); break; }
                    // The executor places the result in the leaf if it can, else the alias's copy runs.
                    case GraphNodeKind.Alias:
                    {
                        GraphNode alias = e;
                        Pinned(e.Node);
                        Pinned(alias.Srcs[1]);
                        Attribute(() =>
                        {
                            if (!_x.TryAlias(Get(alias.Node), Get(alias.Srcs[0]), alias.Bytes)) Force(alias.Srcs[1]);
                        });
                        break;
                    }
                    case GraphNodeKind.Output: Pinned(e.Node); _outputs.Add(e.Node); _outputFloats.Add(e.Bytes); break;
                    case GraphNodeKind.Boundary:
                        if (_state[e.Node] == Done) Boundary(e.Node);
                        else _boundaryPending[e.Node] = 0;
                        break;
                    case GraphNodeKind.Fused:
                        if (InRegion) _nested++;
                        else if (_x.Fuses((GraphFusion)e.Node)) _region = e;
                        break;
                    case GraphNodeKind.FusedEnd:
                        if (InRegion && _nested > 0) _nested--;
                        else if (InRegion)
                        {
                            GraphNode region = _region, end = e;
                            _region = default;
                            Attribute(() => RunFused(region, end.Srcs));
                            SkipUnrun();
                        }
                        break;
                }
            }

            /// <summary>
            /// The closed region's ops that did not run (its kernel computed the results, or the results did not need
            /// them) never will: their reads are spent, so their sources are released after their other readers (else a
            /// region's inputs - a prompt chunk's in-projections - stayed alive to the end of the step).
            /// </summary>
            private void SkipUnrun()
            {
                for (int i = _regionOps.Count - 1; i >= 0; i--)
                {
                    int op = _regionOps[i];
                    if (_state[op] != Lazy) continue;
                    _state[op] = Skipped;
                    Consumed(op);
                }
                _regionOps.Clear();
            }

            /// <summary>The region's results from its kernel, else from its ops.</summary>
            private void RunFused(in GraphNode region, int[] results)
            {
                var inputs = new Tensor[region.Srcs.Length];
                var weights = new IGraphWeight[region.Srcs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    Force(region.Srcs[i]);
                    if (_values[region.Srcs[i]].Weight is { } weight)
                    {
                        weights[i] = weight;
                        continue;
                    }
                    inputs[i] = Get(region.Srcs[i]);
                    if (!_x.RegionReadsStridedInputs((GraphFusion)region.Node) && !inputs[i].IsContiguous()) inputs[i] = _x.Contig(inputs[i], Run);
                }
                var outs = new Tensor[results.Length];
                if (_x.TryFused((GraphFusion)region.Node, inputs, weights, region.Ip, region.Fp, outs))
                {
                    for (int i = 0; i < results.Length; i++)
                    {
                        _state[results[i]] = None;
                        Set(results[i], new Value { Tensor = Run.Keep(outs[i]) }, fresh: true);
                    }
                    return;
                }
                foreach (Tensor t in outs) t?.Dispose();
                foreach (int r in results) Force(r);
            }

            public void InputData(int node, IntPtr data, long bytes, object handle)
            {
                if (_state[node] != Pending) return;
                var input = new Dictionary<int, (IntPtr, long, object)> { [node] = (data, bytes, handle) };
                Attribute(() => Set(node, _x.BindLeaf(node, _nodes[node], input, Run), fresh: handle == null));
            }

            public long[] OutputFloats => _outputFloats.ToArray();

            /// <summary>Read the outputs into <paramref name="outputs"/>.</summary>
            public void Finish(float[][] outputs, Tensor[] keep = null)
            {
                _x.FinishWrites();
                for (int o = 0; o < _outputs.Count; o++)
                {
                    Tensor c = _x.Contig(_x.F32(Get(_outputs[o]), Run.Owned), Run.Owned);
                    if (keep != null)
                    {
                        keep[o] = c.CopyRef();
                        continue;
                    }
                    if (c.ElementCount() == outputs[o].LongLength)
                    {
                        c.CopyToArray(outputs[o]);
                        continue;
                    }
                    int count = (int)Math.Min(outputs[o].LongLength, c.ElementCount());
                    Array.Copy(c.GetElementsAsFloat(count), outputs[o], count);
                }
            }

            public void Dispose()
            {
                // A run that ended early (an exception) still finishes the writes it left for later, while their tensors
                // live; a failure there is the run's own error again, already on its way out.
                try { _x.FinishWrites(); }
                catch (Exception) { }
                for (int i = 0; i < _count; i++) _own[i]?.Dispose();
                foreach (Tensor t in Run.Owned) t.Dispose();
                foreach (Tensor t in Run.Lasting) t.Dispose();
                foreach (Tensor t in Run.Products.Values) t.Dispose();
                Array.Clear(_nodes, 0, _count);
                Array.Clear(_values, 0, _count);
                Array.Clear(_state, 0, _count);
                Array.Clear(_fresh, 0, _count);
                Array.Clear(_own, 0, _count);
                Array.Clear(_uses, 0, _count);
                _regionOps.Clear();
                _x._buffers = new Buffers { Nodes = _nodes, Values = _values, State = _state, Fresh = _fresh, Own = _own, Uses = _uses };
                _count = 0;
            }

            /// <summary>Run <paramref name="work"/>: the tensors it made that became a node's value stay owned by
            /// that node; the rest (views, copies, products' blocks) were its scratch and are released now.</summary>
            private void Attribute(Action work)
            {
                int mark = Run.Owned.Count, setMark = _setDuring.Count;
                work();
                for (int i = mark; i < Run.Owned.Count; i++)
                {
                    Tensor t = Run.Owned[i];
                    int owner = -1;
                    for (int j = setMark; j < _setDuring.Count && owner < 0; j++)
                        if (ReferenceEquals(_values[_setDuring[j]].Tensor, t) && _own[_setDuring[j]] == null) owner = _setDuring[j];
                    // A node made and read for the last time within this work (a fused region forcing its sources, the
                    // add-and-norm peephole) was released before it had an owner: its tensor goes now, not at the step's end.
                    if (owner >= 0 && _uses[owner] != Released) _own[owner] = t;
                    else t.Dispose();
                }
                Run.Owned.RemoveRange(mark, Run.Owned.Count - mark);
                _setDuring.RemoveRange(setMark, _setDuring.Count - setMark);
            }

            /// <summary>An output, effect or alias reads <paramref name="node"/> at the end: a template that did not
            /// keep it is another step's, so nothing more is released (and a node it already released fails here,
            /// not as a read of a destroyed tensor).</summary>
            private void Pinned(int node)
            {
                if (!_freeing || _live.Pins(node)) return;
                _freeing = false;
                if (_uses[node] < 0)
                    throw new InvalidOperationException(
                        $"streamed step diverged from its template: node {node} was released before an output read it");
            }

            /// <summary>Node <paramref name="node"/> has been read once more: release it after its last reader.</summary>

            private void Use(int node)
            {
                if (!_freeing || node < 0) return;
                if (--_uses[node] != 0) return;
                _uses[node] = Released;
                Tensor t = _own[node];
                if (t != null)
                {
                    t.Dispose();
                    _own[node] = null;
                }
                foreach (var key in Run.Products.Keys.Where(k => k.Item2 == node).ToList())
                {
                    Run.Products[key].Dispose();
                    Run.Products.Remove(key);
                }
            }

            /// <summary>Every source of <paramref name="node"/> was read.</summary>
            private void Consumed(int node)
            {
                if (_nodes[node].Kind != GraphNodeKind.Op) return;
                foreach (int src in _nodes[node].Srcs) Use(src);
            }

            private void Add(int node, in GraphNode e)
            {
                if (node != _count) throw new InvalidOperationException($"entry {node} arrived out of order (expected {_count})");
                if (_count == _nodes.Length)
                {
                    int size = _nodes.Length * 2;
                    Array.Resize(ref _nodes, size);
                    Array.Resize(ref _values, size);
                    Array.Resize(ref _state, size);
                    Array.Resize(ref _fresh, size);
                    Array.Resize(ref _own, size);
                    Array.Resize(ref _uses, size);
                }
                _nodes[_count++] = e;
            }

            private void Set(int node, Value v, bool fresh)
            {
                _values[node] = v;
                _state[node] = Done;
                _fresh[node] = fresh;
                _setDuring.Add(node);
                Consumed(node);
                if (_boundaryPending.Count > 0 && _boundaryPending.Remove(node)) Boundary(node);
            }

            private void Boundary(int node)
            {
                if (_values[node].Tensor != null) _x.AtBoundary(_values[node].Tensor, ++_boundaries, -1);
                Use(node);
            }

            private Tensor Get(int node)
            {
                Force(node);
                return _x.Device(_values, node, Run);
            }

            private void Force(int node)
            {
                if (_state[node] == Lazy)
                {
                    // A fused region's op or a copy something reads or forces: run it now (its sources first).
                    _state[node] = None;
                    Op(node, _nodes[node], forced: true);
                }
                switch (_state[node])
                {
                    case Done: return;
                    case Pending: throw new InvalidOperationException($"input node {node} was not given a value");
                    case None: throw new InvalidOperationException($"node {node} was not evaluated");
                    case Skipped: throw new InvalidOperationException($"node {node} is inside a fused region its kernel served: only the region's results are read after it");
                }
                GraphNode n = _nodes[node];
                Tensor result = n.Op == GraphOp.Add
                    ? _x.Binary(Get(n.Srcs[0]), Get(n.Srcs[1]), Run, (r, a, b) => Ops.Add(r, a, b), GraphOp.Add)
                    : _x.RowNorm(Get(n.Srcs[0]), Eps(n), Run);
                Set(node, new Value { Tensor = Run.Keep(result) }, fresh: true);
            }

            private static float Eps(in GraphNode norm) => norm.Fp != null && norm.Fp.Length > 0 ? norm.Fp[0] : 1e-6f;

            private bool Waiting(int node) => _state[node] == Deferred || (_state[node] == Done && _fresh[node]);

            private void Op(int node, in GraphNode n, bool forced = false)
            {
                switch (n.Op)
                {
                    // A copy runs when an effect, an alias (whose executor may place the result instead) or a
                    // reader asks for it, as ggml schedules it: an alias's fallback never runs ahead of the alias.
                    case GraphOp.Cpy when !forced:
                        _state[node] = Lazy;
                        return;
                    case GraphOp.Add when Waiting(n.Srcs[0]) && Waiting(n.Srcs[1]):
                    case GraphOp.RmsNorm when Waiting(n.Srcs[0]):
                        _state[node] = Deferred;
                        return;
                    // A scaled norm (ggml's l2_norm as the runtime writes it) is a norm weighted by a constant.
                    case GraphOp.Scale when _state[n.Srcs[0]] == Deferred && _nodes[n.Srcs[0]].Op == GraphOp.RmsNorm
                        && (n.Fp == null || n.Fp.Length < 2 || n.Fp[1] == 0f):
                    {
                        GraphNode norm = _nodes[n.Srcs[0]];
                        Tensor x = Get(norm.Srcs[0]);
                        Value scaled = _x.WeightedNorm(x, Eps(norm), _x.Constant(x.Sizes[^1], n.Fp?[0] ?? 1f), Run);
                        Consumed(n.Srcs[0]);
                        Set(node, scaled, fresh: true);
                        return;
                    }
                    case GraphOp.Mul when _state[n.Srcs[0]] == Deferred && _nodes[n.Srcs[0]].Op == GraphOp.RmsNorm && IsWeightVector(n.Srcs[1]):
                    {
                        GraphNode norm = _nodes[n.Srcs[0]];
                        Tensor weight = _values[n.Srcs[1]].Tensor;
                        int sum = norm.Srcs[0];
                        if (_state[sum] == Deferred && _nodes[sum].Op == GraphOp.Add)
                        {
                            GraphNode add = _nodes[sum];
                            Tensor a = Get(add.Srcs[0]), b = Get(add.Srcs[1]);
                            if (a.Sizes.SequenceEqual(b.Sizes) && _x.TryAddWeightedNorm(a, b, weight, Eps(norm), out Tensor total, out Tensor normed))
                            {
                                Set(sum, new Value { Tensor = Run.Keep(total) }, fresh: true);
                                Consumed(n.Srcs[0]);
                                Set(node, new Value { Tensor = Run.Keep(normed) }, fresh: true);
                                return;
                            }
                        }
                        Value weighted = _x.WeightedNorm(Get(sum), Eps(norm), weight, Run);
                        Consumed(n.Srcs[0]);
                        Set(node, weighted, fresh: true);
                        return;
                    }
                }
                foreach (int src in n.Srcs) Force(src);
                Value v = _x.Eval(n, _values, Run);
                bool fresh = n.Op switch
                {
                    GraphOp.View or GraphOp.Reshape or GraphOp.Permute or GraphOp.Transpose or GraphOp.KvWindow => _fresh[n.Srcs[0]],
                    GraphOp.Cpy or GraphOp.SetRows => false,
                    _ => true,
                };
                Set(node, v, fresh);
            }

            private bool IsWeightVector(int node)
            {
                GraphNode w = _nodes[node];
                return w.Kind == GraphNodeKind.Leaf && w.Role == GraphLeafRole.Weight && _values[node].Tensor != null
                    && w.Ne.Skip(1).All(e => e == 1);
            }
        }

        /// <summary>A session's node tables, kept between steps.</summary>
        private sealed class Buffers
        {
            public GraphNode[] Nodes = new GraphNode[4096];
            public Value[] Values = new Value[4096];
            public byte[] State = new byte[4096];
            public bool[] Fresh = new bool[4096];
            public Tensor[] Own = new Tensor[4096];
            public int[] Uses = new int[4096];
        }

        private Buffers _buffers;

        /// <summary>A compiled program: validated once, and run by replaying its entries into a session
        /// (nodes nothing reaches are skipped).</summary>
        private sealed class Plan : IGraphPlan, IGraphTensorOutputs
        {
            public bool KeepTensors { get; set; }
            private Tensor[] _kept;

            public Tensor TakeOutput(int index)
            {
                Tensor t = _kept?[index] ?? throw new InvalidOperationException($"output {index} was not kept");
                _kept[index] = null;
                return t;
            }

            private readonly TensorGraphExecutor _x;
            private readonly GraphProgram _program;
            private readonly bool[] _dead;

            public Plan(TensorGraphExecutor executor, GraphProgram program)
            {
                _x = executor;
                _program = program;
                var nodes = new GraphNode[program.NodeCount];
                var roots = new List<int>();
                int n = 0;
                foreach (GraphNode e in program.Entries)
                {
                    switch (e.Kind)
                    {
                        case GraphNodeKind.Leaf:
                        case GraphNodeKind.Op: nodes[n++] = e; break;
                        case GraphNodeKind.Output:
                        case GraphNodeKind.Effect: roots.Add(e.Node); break;
                        case GraphNodeKind.Alias: roots.Add(e.Srcs[1]); break;
                    }
                }
                var live = new bool[n];
                var stack = new Stack<int>(roots);
                while (stack.Count > 0)
                {
                    int i = stack.Pop();
                    if (live[i]) continue;
                    live[i] = true;
                    if (nodes[i].Kind == GraphNodeKind.Op)
                        foreach (int s in nodes[i].Srcs) stack.Push(s);
                }
                _dead = new bool[n];
                for (int i = 0; i < n; i++)
                {
                    _dead[i] = !live[i];
                    if (!live[i]) continue;
                    GraphNode node = nodes[i];
                    if (node.Kind == GraphNodeKind.Op && !s_ops.Contains(node.Op))
                        throw new NotSupportedException($"{executor.GetType().Name} has no kernel for {node.Op}");
                    if (node.Kind == GraphNodeKind.Leaf && node.Role != GraphLeafRole.Input && node.Handle is not (Tensor or IGraphWeight))
                        throw new NotSupportedException($"{executor.GetType().Name}: leaf {i} ({node.Role}) carries no tensor or weight");
                }
                Outputs = program.OutputFloats.Select(f => new float[f]).ToArray();
            }

            public float[][] Outputs { get; }

            public string LastError { get; private set; }

            public bool IsLeafResident(int node) => true;

            public bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes)
                => Run(inputs, data, bytes, default);

            public bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes, ReadOnlySpan<object> handles)
            {
                var bound = new Dictionary<int, (IntPtr Data, long Bytes, object Handle)>(inputs.Length);
                for (int i = 0; i < inputs.Length; i++)
                    bound[inputs[i]] = (data[i], bytes[i], handles.Length > i ? handles[i] : null);
                return _x.OnBackendThread(() =>
                {
                    var session = new Session(_x, _dead, _x.LivenessOf(_program));
                    try
                    {
                        int node = 0;
                        foreach (GraphNode e in _program.Entries)
                        {
                            if (e.Kind is GraphNodeKind.Leaf or GraphNodeKind.Op)
                            {
                                int index = node++;
                                session.Entry(index, e);
                                if (e.Kind == GraphNodeKind.Leaf && e.Role == GraphLeafRole.Input && bound.TryGetValue(index, out var input))
                                    session.InputData(index, input.Data, input.Bytes, input.Handle);
                            }
                            else
                            {
                                session.Entry(-1, e);
                            }
                        }
                        _kept = KeepTensors ? new Tensor[Outputs.Length] : null;
                        session.Finish(Outputs, _kept);
                        return true;
                    }
                    catch (Exception ex) when (ex is NotSupportedException or ArgumentException or InvalidOperationException)
                    {
                        LastError = ex.ToString();
                        return false;
                    }
                    finally
                    {
                        session.Dispose();
                    }
                });
            }

            public void Dispose() { }
        }

        /// <summary>A program run as the runtime emits it (<see cref="IGraphStream"/>).</summary>
        private sealed class Stream : IGraphStream, IGraphTensorOutputs
        {
            public bool KeepTensors { get; set; }
            private Tensor[] _kept;

            public Tensor TakeOutput(int index)
            {
                Tensor t = _kept?[index] ?? throw new InvalidOperationException($"output {index} was not kept");
                _kept[index] = null;
                return t;
            }

            private readonly TensorGraphExecutor _x;
            private readonly Session _session;

            public Stream(TensorGraphExecutor executor, GraphProgram template)
            {
                _x = executor;
                _session = new Session(executor, null, executor.LivenessOf(template));
            }

            public void Entry(int node, in GraphNode entry) => _session.Entry(node, entry);

            public void InputData(int node, IntPtr data, long bytes, object handle) => _session.InputData(node, data, bytes, handle);

            public bool Execute(Func<bool> body) => _x.OnBackendThread(body);

            public float[][] Outputs { get; private set; }

            public string LastError { get; private set; }

            public bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes)
            {
                // Every input arrived with its leaf; the outputs are read where the ops ran.
                return _x.OnBackendThread(() =>
                {
                    try
                    {
                        long[] floats = _session.OutputFloats;
                        if (KeepTensors)
                        {
                            Outputs = floats.Select(_ => Array.Empty<float>()).ToArray();
                            _kept = new Tensor[floats.Length];
                            _session.Finish(null, _kept);
                            return true;
                        }
                        Outputs = floats.Select(f => new float[f]).ToArray();
                        _session.Finish(Outputs);
                        return true;
                    }
                    catch (Exception ex) when (ex is NotSupportedException or ArgumentException or InvalidOperationException)
                    {
                        LastError = ex.ToString();
                        return false;
                    }
                });
            }

            public bool IsLeafResident(int node) => true;

            public void Dispose() => _x.OnBackendThread(() => { _session.Dispose(); return true; });
        }

        public IGraphStream OpenStream(GraphProgram template = null) => new Stream(this, template);

        /// <summary>Run a whole plan where the backend's ops run (one hop, not one per op).</summary>
        protected virtual bool OnBackendThread(Func<bool> run) => run();

        /// <summary>sum = a + b and rms_norm(sum) * weight in one kernel (same shapes; the weight a vector
        /// over the rows), both fresh tensors; false where the backend has no such kernel.</summary>
        /// <summary>Make <paramref name="leaf"/>'s bytes at <paramref name="offsetBytes"/> hold <paramref name="result"/>
        /// without the alias's copy (the executor shares the result's memory); false runs the copy.</summary>
        protected virtual bool TryAlias(Tensor result, Tensor leaf, long offsetBytes) => false;

        /// <summary>Rotary positions per row are a broadcast of the positions tensor on the device, not an upload of
        /// host values per step (a backend whose recorded steps must read their positions from the device).</summary>
        protected virtual bool BroadcastsPositionsOnDevice => false;

        /// <summary>Whether this executor has a kernel for regions of <paramref name="kind"/> (its sessions then
        /// defer the region's ops).</summary>
        protected virtual bool Fuses(GraphFusion kind) => false;

        /// <summary>A fused region's results (<see cref="GraphFusion"/> documents each kind's inputs and results,
        /// in ggml's extents) from its inputs: a tensor in <paramref name="inputs"/>, or a quantized weight in
        /// <paramref name="weights"/> at the same index; false (nothing set) where the kernel does not apply,
        /// and the region's ops run instead.</summary>
        protected virtual bool TryFused(GraphFusion kind, Tensor[] inputs, IGraphWeight[] weights, long[] ip, float[] fp, Tensor[] results) => false;

        protected virtual bool TryAddWeightedNorm(Tensor a, Tensor b, Tensor weight, float eps, out Tensor sum, out Tensor normed)
        {
            sum = normed = null;
            return false;
        }

        private Value BindLeaf(int index, in GraphNode leaf, Dictionary<int, (IntPtr, long, object)> inputs, RunState run)
        {
            if (leaf.Role != GraphLeafRole.Input)
            {
                return leaf.Handle switch
                {
                    IGraphWeight w => new Value { Weight = w, FirstRow = 0, Rows = w.Ne1 },
                    Tensor t => new Value { Tensor = Shaped(t, leaf.Ne, run) },
                    _ => throw new NotSupportedException($"leaf {index} carries no tensor or weight"),
                };
            }
            if (inputs == null || !inputs.TryGetValue(index, out var input))
                throw new InvalidOperationException($"input node {index} was not given a value");
            (IntPtr data, long bytes, object handle) = input;
            if (handle is Tensor given)
                return new Value { Tensor = Shaped(given, leaf.Ne, run) };
            if (handle is StableInput stable)
            {
                // A varying input binds its device copy alone: an op reading it on the host synchronizes, which aborts
                // the recording, instead of recording this launch's values.
                if (stable.Varies) return new Value { Tensor = Shaped(stable.Device, leaf.Ne, run) };
                // The device copy a captured step binds; the host values still classify a mask and feed host reads.
                Value host = HostValues(leaf.Format, leaf.Ne, data, bytes);
                return host with { Tensor = Shaped(stable.Device, leaf.Ne, run) };
            }
            return FromHost(leaf.Format, leaf.Ne, data, bytes, run);
        }

        /// <summary>A host input's values alone (no tensor): F32/F16 as floats, I32/I64 as ints.</summary>
        private static unsafe Value HostValues(int format, long[] ne, IntPtr data, long bytes)
        {
            long[] sizes = Sizes(ne);
            long count = 1;
            foreach (long s in sizes) count *= s;
            switch (format)
            {
                case GraphFormat.F32:
                {
                    var values = new float[count];
                    fixed (float* dst = values) Buffer.MemoryCopy((void*)data, dst, count * 4L, Math.Min(bytes, count * 4L));
                    return new Value { Floats = values, HostSizes = sizes };
                }
                case GraphFormat.F16:
                {
                    var values = new float[count];
                    var src = (System.Half*)data;
                    for (long i = 0; i < count; i++) values[i] = (float)src[i];
                    return new Value { Floats = values, HostSizes = sizes };
                }
                case GraphFormat.I32:
                case GraphFormat.I64:
                {
                    var values = new int[count];
                    if (format == GraphFormat.I32)
                        fixed (int* dst = values) Buffer.MemoryCopy((void*)data, dst, count * 4L, Math.Min(bytes, count * 4L));
                    else
                        for (long i = 0; i < count; i++) values[i] = checked((int)((long*)data)[i]);
                    return new Value { Ints = values };
                }
                default:
                    throw new NotSupportedException($"input format {format}");
            }
        }

        // ================================================================ shapes

        /// <summary>ggml extents of a tensor (innermost first, padded with ones to four).</summary>
        protected static long[] Ne(Tensor t)
        {
            var ne = new long[] { 1, 1, 1, 1 };
            for (int i = 0; i < t.DimensionCount && i < 4; i++) ne[i] = t.Sizes[t.DimensionCount - 1 - i];
            return ne;
        }

        /// <summary>Tensor sizes of ggml extents (outermost first, trailing ones dropped).</summary>
        protected static long[] Sizes(ReadOnlySpan<long> ne)
        {
            int nd = ne.Length;
            while (nd > 1 && ne[nd - 1] == 1) nd--;
            var sizes = new long[nd];
            for (int i = 0; i < nd; i++) sizes[i] = ne[nd - 1 - i];
            return sizes;
        }

        private static long[] Sizes4(long[] ne) => new[] { ne[3], ne[2], ne[1], ne[0] };

        /// <summary>A four-dimensional view of any tensor (size-one dims prepended; no copy).</summary>
        private static Tensor As4(Tensor t, RunState run)
        {
            if (t.DimensionCount == 4) return t;
            var sizes = new long[4];
            var strides = new long[4];
            int pad = 4 - t.DimensionCount;
            // A size-one dim may take any stride: the row-major one keeps the view a box of its
            // storage (backends write strided boxes in place).
            long outer = t.DimensionCount == 0 ? 1 : t.Strides[0] * t.Sizes[0];
            for (int i = 0; i < 4; i++)
            {
                sizes[i] = i < pad ? 1 : t.Sizes[i - pad];
                strides[i] = i < pad ? Math.Max(1, outer) : t.Strides[i - pad];
            }
            return run.Keep(new Tensor(sizes, strides, t.Storage, t.StorageOffset));
        }

        private Tensor Contig(Tensor t, List<Tensor> owned)
        {
            if (t.IsContiguous()) return t;
            Tensor c = Contiguous(t);
            owned.Add(c);
            return c;
        }

        /// <summary>A contiguous copy of <paramref name="t"/> (any element type).</summary>
        private Tensor Contiguous(Tensor t)
        {
            if (t.ElementType == DType.Float32) return Ops.NewContiguous(t);
            var c = new Tensor(t.Allocator, t.ElementType, t.Sizes);
            CopyInto(c, t);
            return c;
        }

        /// <summary>Copy <paramref name="src"/> into <paramref name="dst"/> (the same sizes; either may be a
        /// strided view), converting the element type. Every copy of the interpreter goes through here:
        /// a backend whose generic copy does not convert overrides it. Default: <see cref="Ops.Copy"/>.</summary>
        protected virtual void CopyInto(Tensor dst, Tensor src) => Ops.Copy(dst, src);

        private Tensor Contig(Tensor t, RunState run) => Contig(t, run.Owned);

        private Tensor F32(Tensor t, List<Tensor> owned)
        {
            if (t.ElementType == DType.Float32) return t;
            var r = new Tensor(t.Allocator, DType.Float32, t.Sizes);
            CopyInto(r, t);
            owned.Add(r);
            return r;
        }

        /// <summary><paramref name="t"/> with the leaf's extents (same elements).</summary>
        private Tensor Shaped(Tensor t, long[] ne, RunState run)
        {
            long[] sizes = Sizes(ne);
            if (t.Sizes.SequenceEqual(sizes)) return t;
            return run.Keep(Contig(t, run).View(sizes));
        }

        private unsafe Value FromHost(int format, long[] ne, IntPtr data, long bytes, RunState run)
        {
            long[] sizes = Sizes(ne);
            long count = 1;
            foreach (long s in sizes) count *= s;
            switch (format)
            {
                case GraphFormat.F32:
                {
                    var values = new float[count];
                    fixed (float* dst = values) Buffer.MemoryCopy((void*)data, dst, count * 4L, Math.Min(bytes, count * 4L));
                    return HostFloats(values, sizes, run);
                }
                case GraphFormat.F16:
                {
                    var values = new float[count];
                    var src = (System.Half*)data;
                    for (long i = 0; i < count; i++) values[i] = (float)src[i];
                    return HostFloats(values, sizes, run);
                }
                case GraphFormat.I32:
                case GraphFormat.I64:
                {
                    var values = new int[count];
                    if (format == GraphFormat.I32)
                        fixed (int* dst = values) Buffer.MemoryCopy((void*)data, dst, count * 4L, Math.Min(bytes, count * 4L));
                    else
                        for (long i = 0; i < count; i++) values[i] = checked((int)((long*)data)[i]);
                    return new Value { Tensor = IntTensor(values, sizes, run), Ints = values };
                }
                default:
                    throw new NotSupportedException($"input format {format}");
            }
        }

        /// <summary>Host floats of an input: small ones go to the device now, large ones (a prompt's mask)
        /// when an op first reads the tensor (<see cref="Device"/>).</summary>
        private Value HostFloats(float[] values, long[] sizes, RunState run)
        {
            if (values.LongLength > 65536) return new Value { Floats = values, HostSizes = sizes };
            var t = run.Keep(new Tensor(Allocator, DType.Float32, sizes));
            t.SetElementsAsFloat(values);
            return new Value { Tensor = t, Floats = values };
        }

        /// <summary>The value's tensor, uploading a host-only input on first use (stored back).</summary>
        private Tensor Device(Value[] values, int node, RunState run)
        {
            Value v = values[node];
            if (PendingWrites && v.Tensor != null) BeforeRead(v.Tensor);
            if (v.Tensor != null || v.HostSizes == null) return v.Tensor;
            var t = run.KeepLasting(new Tensor(Allocator, DType.Float32, v.HostSizes));
            t.SetElementsAsFloat(v.Floats);
            values[node] = new Value { Tensor = t, Floats = v.Floats };
            return t;
        }

        private Tensor NewIntTensor(int[] values, long[] sizes)
        {
            var t = new Tensor(Allocator, DType.Int32, sizes);
            t.SetElementsAsInt(values);
            return t;
        }

        private Tensor IntTensor(int[] values, long[] sizes, RunState run)
        {
            var t = run.Keep(new Tensor(Allocator, DType.Int32, sizes));
            t.SetElementsAsInt(values);
            return t;
        }

        protected static int[] Ints(Value v) => v.Ints ?? v.Tensor.GetElementsAsInt((int)v.Tensor.ElementCount());

        // ================================================================ ops

        private Value Eval(GraphNode n, Value[] values, RunState run)
        {
            if (PendingWrites && n.Srcs != null)
                foreach (int src in n.Srcs)
                    if (src >= 0 && values[src].Tensor is { } read) BeforeRead(read);
            Tensor T(int s) => Device(values, n.Srcs[s], run) ?? throw new NotSupportedException($"{n.Op}: source {s} is a quantized weight");
            Value V(Tensor t) => new() { Tensor = run.Keep(t) };
            long Ip(int i, long fallback) => n.Ip != null && n.Ip.Length > i ? n.Ip[i] : fallback;
            float Fp(int i, float fallback) => n.Fp != null && n.Fp.Length > i ? n.Fp[i] : fallback;

            switch (n.Op)
            {
                case GraphOp.Add: return V(Binary(T(0), T(1), run, (r, a, b) => Ops.Add(r, a, b), GraphOp.Add));
                case GraphOp.Sub: return V(Binary(T(0), T(1), run, (r, a, b) => Ops.Sub(r, a, b), GraphOp.Sub));
                case GraphOp.Mul: return V(Binary(T(0), T(1), run, (r, a, b) => Ops.Mul(r, a, b), GraphOp.Mul));
                case GraphOp.Div: return V(Binary(T(0), T(1), run, (r, a, b) => Ops.Div(r, a, b), GraphOp.Div));
                case GraphOp.Sqr: { Tensor x = In(T(0), run); return V(Ops.Mul(null, x, x)); }
                case GraphOp.Sqrt: return V(Sqrt(In(T(0), run)));
                case GraphOp.Clamp: return V(Ops.Clamp(null, In(T(0), run), Fp(0, float.MinValue), Fp(1, float.MaxValue)));
                case GraphOp.Scale: return V(Ops.Mul(null, In(T(0), run), Fp(0, 1f)));
                case GraphOp.Silu: return V(Ops.SiLU(null, In(T(0), run)));
                case GraphOp.Sigmoid: return V(Ops.Sigmoid(null, In(T(0), run)));
                case GraphOp.Exp: return V(Ops.Exp(null, In(T(0), run)));
                case GraphOp.Neg: return V(Ops.Neg(null, In(T(0), run)));
                case GraphOp.Tanh: return V(Ops.Tanh(null, In(T(0), run)));
                case GraphOp.Relu: return V(Ops.Relu(null, In(T(0), run)));
                case GraphOp.Softplus: return V(Softplus(In(T(0), run)));
                case GraphOp.SwigluSplit: return V(Ops.SiLUMul(null, In(T(0), run), In(T(1), run)));
                case GraphOp.Swiglu:
                {
                    Tensor x = T(0);
                    int last = x.DimensionCount - 1;
                    long half = x.Sizes[last] / 2;
                    using Tensor gate = x.Narrow(last, 0, half);
                    using Tensor up = x.Narrow(last, half, half);
                    return V(Ops.SiLUMul(null, In(gate, run), In(up, run)));
                }
                case GraphOp.RmsNorm: return V(RowNorm(T(0), Fp(0, 1e-6f), run));
                case GraphOp.L2Norm:
                {
                    // x / max(|x|, eps) as an unweighted RMS norm: 1 / sqrt(sum / n + eps^2 / n) / sqrt(n).
                    Tensor x = T(0);
                    long cols = x.Sizes[^1];
                    Tensor r = run.Keep(RowNorm(x, Fp(0, 1e-6f) * Fp(0, 1e-6f) / cols, run));
                    return V(Ops.Mul(null, r, 1f / MathF.Sqrt(cols)));
                }
                case GraphOp.MulMat:
                {
                    Tensor y = MatMul(values[n.Srcs[0]], n.Srcs[1], T(1), run);
                    return new Value { Tensor = y, Relaid = !y.IsContiguous() };
                }
                case GraphOp.MulMatId: return V(MatMulId(values[n.Srcs[0]], T(1), values[n.Srcs[2]], run));
                case GraphOp.MoeHostExperts: return V(HostExperts(T(0), values[n.Srcs[1]], values[n.Srcs[2]], values[n.Srcs[3]],
                    values[n.Srcs[4]], values[n.Srcs[5]], (int)Ip(0, 0), run));
                case GraphOp.Reshape: return V(Contig(T(0), run).View(Sizes(n.Ip)));
                case GraphOp.View:
                {
                    Value src = values[n.Srcs[0]];
                    if (src.Weight != null) return WeightRows(src, n.Ip);
                    return V(View(src.Relaid ? Contig(src.Tensor, run) : src.Tensor, n.Ip));
                }
                case GraphOp.Permute:
                    return new Value { Tensor = run.Keep(Permute(As4(T(0), run), (int)Ip(0, 0), (int)Ip(1, 1), (int)Ip(2, 2), (int)Ip(3, 3))),
                        Relaid = values[n.Srcs[0]].Relaid };
                case GraphOp.Transpose:
                    return new Value { Tensor = run.Keep(As4(T(0), run).Transpose(2, 3)), Relaid = values[n.Srcs[0]].Relaid };
                case GraphOp.Cont: return V(Contiguous(T(0)));
                case GraphOp.Cast:
                {
                    if (Ip(0, GraphFormat.F32) != GraphFormat.F32) throw new NotSupportedException($"cast to format {Ip(0, 0)}");
                    Tensor x = T(0);
                    return x.ElementType == DType.Float32 ? V(Contiguous(x)) : new Value { Tensor = F32(x, run.Owned) };
                }
                case GraphOp.Cpy:
                {
                    Tensor dst = T(1);
                    CopyInto(dst, Same(T(0), dst, run));
                    return new Value { Tensor = dst };
                }
                case GraphOp.SetRows: return new Value { Tensor = SetRows(T(0), T(1), values[n.Srcs[2]], run) };
                case GraphOp.GetRows: return V(GetRows(T(0), values[n.Srcs[1]], run));
                case GraphOp.Concat: return V(Concat(T(0), T(1), (int)Ip(0, 0), run));
                case GraphOp.KvWindow:
                    return new Value { Tensor = run.Keep(KvWindow(T(0), n.Ip)), Window = new CacheWindow(T(0), n.Ip[3] % n.Ip[1]) };
                case GraphOp.RopeExt:
                case GraphOp.RopeMulti:
                {
                    if (n.Srcs.Length > 2) throw new NotSupportedException("rotary frequency factors");
                    return V(Rope(T(0), values[n.Srcs[1]], n, n.Op == GraphOp.RopeMulti, run, n.Srcs[1]));
                }
                case GraphOp.LightningIndexer: return V(LightningIndexer(T(0), T(1), T(2), T(3), run));
                case GraphOp.AllReduce:
                {
                    if (Collective == null) throw new NotSupportedException("an all-reduce outside a tensor-parallel group");
                    // The reduce sums in place: the result is the partial's storage (its own reference, so the
                    // session releases the two independently); the collective sums once every rank handed its in.
                    Tensor partial = Contig(T(0), run);
                    // Collectives move whole storages: a partial viewing part of one gets its own.
                    if (partial.StorageOffset != 0 || partial.Storage.ElementCount != partial.ElementCount())
                        partial = run.Keep(Ops.NewContiguous(partial));
                    Collective.Contribute(CollectiveRank, partial.CopyRef());
                    return V(partial.CopyRef());
                }
                case GraphOp.TopKMask:
                {
                    int at = 2;
                    Tensor score = Ip(0, 0) != 0 ? T(at++) : null;
                    Value blocks = Ip(1, 0) != 0 ? values[n.Srcs[at++]] : default;
                    Value extra = Ip(2, 0) != 0 ? values[n.Srcs[at++]] : default;
                    return V(TopKMask(T(0), values[n.Srcs[1]], score, blocks, extra, run));
                }
                case GraphOp.FlashAttnExt:
                {
                    if (n.Srcs.Length > 4) throw new NotSupportedException("attention sinks");
                    if (Fp(1, 0f) != 0f || Fp(2, 0f) != 0f) throw new NotSupportedException("ALiBi or a logit softcap in flash attention");
                    // ggml: q [d, nq, H], k [d, nkv, Hkv], v [dv, nkv, Hkv], mask [nkv, nq_pad]; the
                    // result [dv, H, nq] is sizes [nq, H, dv].
                    Tensor q = Heads(T(0), run), k = Heads(T(1), run), v = Heads(T(2), run);
                    long nq = q.Sizes[1], nkv = k.Sizes[1];
                    AttentionMask mask = AttentionMask.None;
                    if (n.Srcs.Length > 3)
                    {
                        Value m = values[n.Srcs[3]];
                        long keys = m.HostSizes != null ? m.HostSizes[^1] : Ne(m.Tensor)[0];
                        if (keys != nkv) throw new NotSupportedException("a mask wider than the keys");
                        int maskNode = n.Srcs[3];
                        if (!run.Masks.TryGetValue((maskNode, nq), out mask))
                            run.Masks[(maskNode, nq)] = mask = Classify(() => Mask(Device(values, maskNode, run), nq, run), m.Floats, nq, keys);
                        if (Ip(1, 0) > 0) mask.KvMax = (int)Ip(1, 0);
                    }
                    return V(FlashAttention(q, k, v, mask, Fp(0, 1f), values[n.Srcs[1]].Window, values[n.Srcs[2]].Window));
                }
                case GraphOp.SoftMaxExt:
                {
                    if (Fp(1, 0f) != 0f) throw new NotSupportedException("ALiBi in soft_max_ext");
                    Tensor x = Contig(T(0), run);
                    float scale = Fp(0, 1f);
                    Tensor s = scale == 1f ? Ops.NewContiguous(x) : Ops.Mul(null, x, scale);
                    if (n.Srcs.Length > 1)
                    {
                        Tensor m = Mask(T(1), Ne(x)[1], run);
                        Tensor sum = Binary(s, m, run, (r, a, b) => Ops.Add(r, a, b), GraphOp.Add);
                        s.Dispose();
                        s = sum;
                    }
                    Ops.Softmax(s, s);
                    return V(s);
                }
                case GraphOp.TopK: return TopK(Contig(T(0), run), (int)Ip(0, 1), run);
                case GraphOp.SumRows: return V(SumRows(Contig(T(0), run)));
                case GraphOp.SsmConv:
                {
                    Tensor x = Contig(T(0), run), kernel = Contig(T(1), run);
                    if (x.DimensionCount == 2) x = run.Keep(x.View(1, x.Sizes[0], x.Sizes[1]));
                    return V(SsmConv(x, kernel));
                }
                case GraphOp.GatedDeltaNet:
                {
                    Tensor q = Contig(T(0), run), k = Contig(T(1), run), v = Contig(T(2), run);
                    Tensor g = Contig(T(3), run), b = Contig(T(4), run), s = Contig(T(5), run);
                    return V(GatedDeltaNet(As4(q, run), As4(k, run), As4(v, run), As4(g, run), As4(b, run), As4(s, run), (int)Ip(0, 1)));
                }
                default:
                    throw new NotSupportedException($"{GetType().Name} has no kernel for {n.Op}");
            }
        }

        /// <summary>ggml's binary ops: b repeats over a (b's extents are ones or a's).</summary>
        private Tensor Binary(Tensor a, Tensor b, RunState run, Action<Tensor, Tensor, Tensor> op, GraphOp kind)
        {
            long[] na = Ne(a), nb = Ne(b);
            Tensor ac = In(a, run);
            var result = new Tensor(Allocator, DType.Float32, ac.Sizes);
            using Tensor r4 = result.View(Sizes4(na));
            Tensor a4 = As4(ac, run);
            Tensor b4 = As4(In(b, run), run);
            if (na.SequenceEqual(nb))
            {
                op(r4, a4, b4);
                return result;
            }
            for (int i = 0; i < 4; i++)
                if (nb[i] != 1 && nb[i] != na[i])
                    throw new NotSupportedException($"binary op repeating [{string.Join(",", nb)}] over [{string.Join(",", na)}]");
            if (TryBinaryBroadcast(kind, r4, a4, b4, na, nb))
                return result;
            Tensor be = BroadcastsInKernels ? run.Keep(b4.Expand(Sizes4(na))) : Gathered(b4, Sizes4(na), run);
            op(r4, a4, be);
            return result;
        }

        /// <summary>A binary op (<paramref name="kind"/>: Add, Sub, Mul or Div) whose b repeats over a, in one kernel reading
        /// b in place; false (the repeated b is materialized first) where the backend has none for these operands.</summary>
        protected virtual bool TryBinaryBroadcast(GraphOp kind, Tensor result, Tensor a, Tensor b, long[] na, long[] nb) => false;

        /// <summary>ggml_sum_rows over a contiguous tensor: the sums of its innermost rows (executors with a kernel
        /// override it).</summary>
        protected virtual Tensor SumRows(Tensor x) => Ops.Sum(null, x, x.DimensionCount - 1);

        /// <summary><paramref name="result"/> [n, m] = <paramref name="rows"/> [n, k] times the F32 <paramref name="weight"/>
        /// [m, k] transposed (contiguous operands): an unquantized weight's product. A backend with a faster kernel for some
        /// shapes overrides it.</summary>
        protected virtual void DenseProduct(Tensor result, Tensor rows, Tensor weight)
        {
            using Tensor wt = weight.Transpose();
            Ops.Addmm(result, 0, result, 1f, rows, wt);
        }

        /// <summary>Whether the executor queued a write to a node's tensor that it finishes later (<see cref="BeforeRead"/>):
        /// then every op's sources pass through <see cref="BeforeRead"/> first.</summary>
        protected bool PendingWrites { get; set; }

        /// <summary>An op is about to read <paramref name="t"/> (called while <see cref="PendingWrites"/> is set): an executor
        /// that left a write to it for later (a doorbell's host sum, joined after the work that does not need it) finishes
        /// it now.</summary>
        protected virtual void BeforeRead(Tensor t) { }

        /// <summary>The run's outputs are about to be read: every write left for later is finished.</summary>
        protected virtual void FinishWrites() { }

        /// <summary>A run (one step's program) begins: an executor drops what it cached for the previous one.</summary>
        protected virtual void BeginRun() { }



        /// <summary>The gather indices of <see cref="Gathered"/> by shape, on the device once (the first run, never inside a
        /// recording: a recorded step reads the copy its recording bound).</summary>
        private readonly Dictionary<string, Tensor> _gatherIndices = new();

        /// <summary><paramref name="b4"/> repeated to <paramref name="sizes"/> as a contiguous tensor: the rows
        /// of its longest unrepeated suffix gathered by index (one kernel on any backend).</summary>
        private Tensor Gathered(Tensor b4, long[] sizes, RunState run)
        {
            int k = 4;
            long width = 1;
            while (k > 0 && b4.Sizes[k - 1] == sizes[k - 1]) { width *= sizes[k - 1]; k--; }
            long outer = 1, bOuter = 1;
            for (int i = 0; i < k; i++) { outer *= sizes[i]; bOuter *= b4.Sizes[i]; }
            Tensor rows = run.Keep(Contig(b4, run).View(bOuter, width));
            string key = $"{string.Join(",", b4.Sizes.ToArray())}>{string.Join(",", sizes)}";
            Tensor cached;
            lock (_gatherIndices)
                if (!_gatherIndices.TryGetValue(key, out cached))
                    _gatherIndices[key] = cached = GatherIndex(b4, sizes, k, outer);
            Tensor gathered = run.Keep(Ops.IndexSelect(null, rows, cached));
            return run.Keep(gathered.View(sizes));
        }

        private Tensor GatherIndex(Tensor b4, long[] sizes, int k, long outer)
        {
            var index = new int[outer];
            var at = new long[Math.Max(1, k)];
            for (long o = 0; o < outer; o++)
            {
                long bo = 0;
                for (int i = 0; i < k; i++) bo = bo * b4.Sizes[i] + (b4.Sizes[i] == 1 ? 0 : at[i]);
                index[o] = checked((int)bo);
                for (int i = k - 1; i >= 0; i--)
                {
                    if (++at[i] < sizes[i]) break;
                    at[i] = 0;
                }
            }
            return NewIntTensor(index, new long[] { outer });
        }

        /// <summary>log(1 + exp(x)) as max(x, 0) + log(1 + exp(-|x|)) (no overflow).</summary>
        /// <summary>ggml_sqrt, elementwise (executors with a kernel of their own override it).</summary>
        protected virtual Tensor Sqrt(Tensor x) => Ops.Pow(null, x, 0.5f);

        protected virtual Tensor Softplus(Tensor x)
        {
            using Tensor abs = Ops.Abs(null, x);
            using Tensor neg = Ops.Neg(null, abs);
            using Tensor e = Ops.Exp(null, neg);
            using Tensor e1 = Ops.Add(null, e, 1f);
            using Tensor log = Ops.Log(null, e1);
            using Tensor pos = Ops.Relu(null, x);
            return Ops.Add(null, pos, log);
        }

        /// <summary>rms_norm(x) * weight as one kernel (the weight a vector over x's rows).</summary>
        private readonly Dictionary<(long, float), Tensor> _constants = new();

        /// <summary>A device vector of <paramref name="length"/> copies of <paramref name="value"/> (kept for the executor's life).</summary>
        private Tensor Constant(long length, float value)
        {
            if (_constants.TryGetValue((length, value), out Tensor t)) return t;
            t = new Tensor(Allocator, DType.Float32, length);
            Ops.Fill(t, value);
            _constants[(length, value)] = t;
            return t;
        }

        private Value WeightedNorm(Tensor x0, float eps, Tensor weight, RunState run)
        {
            Tensor x = Contig(x0, run);
            long cols = x.Sizes[^1];
            if (weight.ElementCount() != cols)
            {
                Tensor plain = run.Keep(RowNorm(x, eps, run));
                return new Value { Tensor = run.Keep(Binary(plain, weight, run, (r, a, b) => Ops.Mul(r, a, b), GraphOp.Mul)) };
            }
            using Tensor rows = x.View(x.ElementCount() / cols, cols);
            using Tensor gamma = Contig(weight, run).View(cols);
            using Tensor normed = Ops.RMSNorm(null, rows, gamma, null, eps);
            return new Value { Tensor = run.Keep(normed.View(x.Sizes)) };
        }

        private Tensor RowNorm(Tensor x, float eps, RunState run)
        {
            Tensor c = Contig(x, run);
            long cols = c.Sizes[^1];
            using Tensor rows = c.View(c.ElementCount() / cols, cols);
            using Tensor normed = Ops.RMSNorm(null, rows, null, null, eps);
            return normed.View(c.Sizes);
        }

        /// <summary>ggml_mul_mat(a, b) = b · aᵀ: [a.ne1, b.ne1, b.ne2, b.ne3], a repeating over b's batches.</summary>
        private Tensor MatMul(Value a, int bNode, Tensor b, RunState run)
        {
            long[] nb = Ne(b);
            if (a.Weight != null)
            {
                IGraphWeight w = a.Weight;
                if (nb[0] != w.Ne0) throw new NotSupportedException($"product of a {w.Ne0}-wide weight with {nb[0]}-wide rows");
                if (!run.Products.TryGetValue((w, bNode), out Tensor full))
                {
                    Tensor rows = Contig(b, run);
                    using Tensor x2 = rows.View(rows.ElementCount() / w.Ne0, w.Ne0);
                    full = QuantizedMatMul(w, x2);
                    run.Products[(w, bNode)] = full;
                }
                Tensor y = full;
                if (a.FirstRow != 0 || a.Rows != w.Ne1)
                {
                    Tensor block = run.Keep(full.Narrow(1, a.FirstRow, a.Rows));
                    // A row block of a packed weight's product: a strided view where kernels read them.
                    if (KernelsReadStridedViews && nb[2] * nb[3] == 1) return block;
                    y = run.Keep(Ops.NewContiguous(block));
                }
                return run.Keep(y.View(Sizes(stackalloc long[] { a.Rows, nb[1], nb[2], nb[3] })));
            }

            Tensor at = a.Tensor;
            long[] na = Ne(at);
            if (na[0] != nb[0]) throw new NotSupportedException($"mul_mat of [{string.Join(",", na)}] with [{string.Join(",", nb)}]");
            long m = na[1], n = nb[1], kd = na[0], batches = nb[2] * nb[3];
            Tensor ac = Contig(F32(at, run.Owned), run), bc = Contig(F32(b, run.Owned), run);
            if (na[2] * na[3] == 1 && batches == 1)
            {
                using Tensor a2 = ac.View(m, kd);
                using Tensor b2 = bc.View(n, kd);
                var r2 = run.Keep(new Tensor(Allocator, DType.Float32, n, m));
                DenseProduct(r2, b2, a2);
                return run.Keep(r2.View(Sizes(stackalloc long[] { m, n, 1, 1 })));
            }
            Tensor a3 = run.Keep(ac.View(na[2] * na[3], m, kd));
            if (na[2] * na[3] != batches)
            {
                if (nb[2] % na[2] != 0 || nb[3] % na[3] != 0) throw new NotSupportedException("mul_mat batches that do not repeat");
                long r2 = nb[2] / na[2], r3 = nb[3] / na[3];
                using Tensor a6 = ac.View(na[3], 1, na[2], 1, m, kd);
                using Tensor e = a6.Expand(na[3], r3, na[2], r2, m, kd);
                Tensor c = run.Keep(Ops.NewContiguous(e));
                a3 = run.Keep(c.View(batches, m, kd));
            }
            using Tensor b3 = bc.View(batches, n, kd);
            using Tensor a3t = a3.Transpose(1, 2);
            var result = run.Keep(new Tensor(Allocator, DType.Float32, batches, n, m));
            Ops.AddmmBatch(result, 0, result, 1f, b3, a3t);
            return run.Keep(result.View(Sizes(stackalloc long[] { m, n, nb[2], nb[3] })));
        }

        /// <summary>A block of a quantized weight's rows (a packed projection split by a view).</summary>
        private static Value WeightRows(Value w, long[] ip)
        {
            IGraphWeight weight = w.Weight;
            long rowBytes = weight.Bytes / (weight.Ne1 * weight.Ne2);
            long ne0 = ip[1], rows = ip[0] > 1 ? ip[2] : 1, offset = ip[8];
            if (ne0 != weight.Ne0 || (ip[0] > 1 && ip[5] != rowBytes) || offset % rowBytes != 0 || ip[0] > 2)
                throw new NotSupportedException("a view of a quantized weight that is not a block of whole rows");
            return new Value { Weight = weight, FirstRow = w.FirstRow + offset / rowBytes, Rows = rows };
        }

        /// <summary>ggml_mul_mat_id(as, b, ids): [as.ne1, K, rows], row (j, t) = expert ids[t, j] · b[t, j or 0].</summary>
        private Tensor MatMulId(Value experts, Tensor b, Value ids, RunState run)
        {
            if (experts.Weight is not IGraphExpertWeights set) throw new NotSupportedException("mul_mat_id over weights that are not stacked experts");
            long[] nb = Ne(b);
            int[] selected = Ints(ids);
            long[] nid = Ids(ids);
            int k = (int)nid[0], rows = (int)nid[1];
            if (nb[2] != rows || (nb[1] != 1 && nb[1] != k)) throw new NotSupportedException("mul_mat_id input rows");
            Tensor bc = Contig(b, run);
            using Tensor b2 = bc.View(nb[1] * rows, nb[0]);

            var byExpert = new List<int>[set.Ne2];
            for (int t = 0; t < rows; t++)
                for (int j = 0; j < k; j++)
                    (byExpert[selected[t * k + j]] ??= new List<int>()).Add(t * k + j);
            var pieces = new List<Tensor>();
            var inverse = new int[rows * k];
            int at = 0;
            for (int e = 0; e < byExpert.Length; e++)
            {
                if (byExpert[e] == null) continue;
                int[] slots = byExpert[e].ToArray();
                int[] sourceRows = slots.Select(s => nb[1] == 1 ? s / k : s).ToArray();
                Tensor idx = IntTensor(sourceRows, new long[] { sourceRows.Length }, run);
                Tensor xs = run.Keep(Ops.IndexSelect(null, b2, idx));
                pieces.Add(run.Keep(ExpertMatMul(set, e, xs)));
                foreach (int s in slots) inverse[s] = at++;
            }
            Tensor sorted = pieces.Count == 1 ? pieces[0] : run.Keep(Ops.Concat(null, 0, pieces.ToArray()));
            Tensor order = IntTensor(inverse, new long[] { inverse.Length }, run);
            Tensor flat = run.Keep(Ops.IndexSelect(null, sorted, order));
            return flat.View(rows, k, set.Ne1);
        }

        private static long[] Ids(Value v) => Ne(v.Tensor);

        /// <summary>
        /// <see cref="GraphOp.MoeHostExperts"/> on any tensor executor: x [rows, hidden], the router's top-K ids and
        /// weights ([rows, K]). The routes whose expert the layer's tier keeps on this device run here, each selected
        /// expert once over its rows; the model's <see cref="IGraphExpertHost"/> computes the rest on the host. A step
        /// the host streams (a prompt chunk) runs every route here, the experts outside the tier read in for it and
        /// dropped after. Returns the weighted sum [rows, hidden].
        /// </summary>
        protected virtual Tensor HostExperts(Tensor x, Value selected, Value weightValue, Value gateValue, Value upValue, Value downValue,
            int layer, RunState run)
        {
            if (gateValue.Weight is not IGraphHostedExperts gate || upValue.Weight is not IGraphExpertWeights up
                || downValue.Weight is not IGraphExpertWeights down)
                throw new NotSupportedException("host experts need the model's expert host on their stacks");
            IGraphExpertHost host = gate.Host;
            // ggml's order: x [hidden, rows] (a decode step's single row arrives as [hidden]), ids [K, rows].
            int hidden = (int)Ne(x)[0];
            int[] ids = Ints(selected);
            long[] nid = Ids(selected);
            int k = (int)nid[0], rows = (int)nid[1], routes = k * rows;
            Tensor xc = run.Keep(Contig(x, run).View(rows, hidden));
            float[] weights = weightValue.Floats ?? Contig(weightValue.Tensor, run).GetElementsAsFloat(routes);
            host.Record(layer, ids, GraphExpertAdapt.Weight(rows));

            bool stream = host.StreamsStep(layer, rows);
            var here = new byte[routes];   // 1: the route runs on this device
            bool anyDevice = false;
            for (int i = 0; i < routes; i++)
            {
                here[i] = stream || host.IsResident(layer, ids[i]) ? (byte)1 : (byte)0;
                anyDevice |= here[i] != 0;
            }
            // The host's routes, less those the route-tail skip leaves out (an expert every row routes at the rank or lower).
            var skip = (byte[])here.Clone();
            int tail = stream ? 0 : host.TailSkipRank(layer);
            if (tail > 0)
                for (int i = 0; i < routes; i++)
                {
                    if (skip[i] != 0 || ids[i] < 0) continue;
                    bool high = false;
                    for (int j = 0; j < routes && !high; j++) high = ids[j] == ids[i] && j % k < tail;
                    if (!high) skip[i] = 1;
                }
            bool anyHost = false;
            for (int i = 0; i < routes; i++) anyHost |= skip[i] == 0 && ids[i] >= 0;

            // The returned tensor is the caller's to keep (V keeps it); the pieces summed into it are this run's.
            Tensor device = anyDevice ? DeviceExperts(xc, ids, weights, here, k, gate, up, down, host, layer, stream, run) : null;
            if (!anyHost)
            {
                if (device != null) return device;
                var zeros = new Tensor(Allocator, DType.Float32, rows, hidden);   // every route left out
                Ops.Fill(zeros, 0f);
                return zeros;
            }
            float[] xs = xc.GetElementsAsFloat(rows * hidden);
            var output = new float[(long)rows * hidden];
            if (!host.Compute(layer, xs, ids, weights, skip, rows, output))
                throw new NotSupportedException($"layer {layer}: the host's expert kernels do not cover these experts");
            var fromHost = new Tensor(Allocator, DType.Float32, rows, hidden);
            fromHost.SetElementsAsFloat(output);
            if (device == null)
                return fromHost;
            return Ops.Add(null, run.Keep(device), run.Keep(fromHost));
        }

        /// <summary>The device's routes of <see cref="HostExperts"/>: each expert once over the rows routed to it,
        /// weighted, then every row's routes summed (a host route reads a zero row).</summary>
        private Tensor DeviceExperts(Tensor x, int[] ids, float[] weights, byte[] onDevice, int k, IGraphExpertWeights gate,
            IGraphExpertWeights up, IGraphExpertWeights down, IGraphExpertHost host, int layer, bool stream, RunState run)
        {
            int rows = (int)x.Sizes[0], hidden = (int)x.Sizes[1], routes = ids.Length;
            var byExpert = new List<int>[gate.Ne2];
            for (int i = 0; i < routes; i++)
                if (onDevice[i] != 0) (byExpert[ids[i]] ??= new List<int>()).Add(i);
            var pieces = new List<Tensor>();
            var position = new int[routes];
            int at = 0;
            for (int e = 0; e < byExpert.Length; e++)
            {
                List<int> list = byExpert[e];
                if (list == null) continue;
                var sourceRows = new int[list.Count];
                var routeWeights = new float[list.Count];
                for (int j = 0; j < list.Count; j++)
                {
                    sourceRows[j] = list[j] / k;
                    routeWeights[j] = weights[list[j]];
                    position[list[j]] = at + j;
                }
                at += list.Count;
                Tensor xs = run.Keep(Ops.IndexSelect(null, x, IntTensor(sourceRows, new long[] { sourceRows.Length }, run)));
                Tensor g = run.Keep(ExpertMatMul(gate, e, xs));
                Tensor u = run.Keep(ExpertMatMul(up, e, xs));
                Tensor act = run.Keep(Ops.SiLUMul(null, g, u));
                Tensor d = run.Keep(ExpertMatMul(down, e, act));
                var w = run.Keep(new Tensor(Allocator, DType.Float32, list.Count, 1));
                w.SetElementsAsFloat(routeWeights);
                pieces.Add(run.Keep(Binary(d, w, run, (r, a, b) => Ops.Mul(r, a, b), GraphOp.Mul)));
                if (stream && !host.IsResident(layer, e))
                {
                    ReleaseStreamedExpert(gate.Expert(e));
                    ReleaseStreamedExpert(up.Expert(e));
                    ReleaseStreamedExpert(down.Expert(e));
                }
            }
            // A host route reads the zero row past the device's pieces.
            Tensor zero = run.Keep(new Tensor(Allocator, DType.Float32, 1, hidden));
            Ops.Fill(zero, 0f);
            pieces.Add(zero);
            for (int i = 0; i < routes; i++)
                if (onDevice[i] == 0) position[i] = at;
            Tensor all = pieces.Count == 1 ? pieces[0] : run.Keep(Ops.Concat(null, 0, pieces.ToArray()));
            Tensor byRoute = run.Keep(Ops.IndexSelect(null, all, IntTensor(position, new long[] { routes }, run)));   // [rows * K, hidden]
            using Tensor perRow = byRoute.View(rows, k, hidden);
            Tensor summed = run.Keep(Ops.Sum(null, perRow, 1));                                                   // [rows, 1, hidden]
            return summed.View(rows, hidden);
        }

        /// <summary>ggml_view_Nd: ip = nd, ne0..ne3, nb1..nb3, offset (bytes).</summary>
        private static Tensor View(Tensor src, long[] ip)
        {
            int nd = (int)ip[0];
            long elem = src.ElementType.Size();
            var ne = new long[] { ip[1], nd > 1 ? ip[2] : 1, nd > 2 ? ip[3] : 1, nd > 3 ? ip[4] : 1 };
            long nb1 = nd > 1 ? ip[5] : elem * ne[0];
            long nb2 = nd > 2 ? ip[6] : nb1 * ne[1];
            long nb3 = nd > 3 ? ip[7] : nb2 * ne[2];
            long[] nb = { elem, nb1, nb2, nb3 };
            if (ip[8] % elem != 0 || nb.Any(b => b % elem != 0))
                throw new NotSupportedException("a view that does not land on whole elements");
            var sizes = new long[4];
            var strides = new long[4];
            for (int i = 0; i < 4; i++)
            {
                sizes[i] = ne[3 - i];
                strides[i] = nb[3 - i] / elem;
            }
            return Checked(new Tensor(sizes, strides, src.Storage, src.StorageOffset + ip[8] / elem), "view");
        }

        /// <summary>A view must stay inside its storage: kernels write through views (cache rows, state
        /// copies), and a view past the end would corrupt memory instead of failing.</summary>
        private static Tensor Checked(Tensor view, string what)
        {
            long last = view.StorageOffset;
            for (int i = 0; i < view.DimensionCount; i++)
            {
                if (view.Sizes[i] == 0) return view;
                last += (view.Sizes[i] - 1) * view.Strides[i];
            }
            if (view.StorageOffset < 0 || last >= view.Storage.ElementCount)
            {
                long count = view.Storage.ElementCount;
                view.Dispose();
                throw new InvalidOperationException(
                    $"a {what} reaches element {last} of a {count}-element storage (offset {view.StorageOffset})");
            }
            return view;
        }

        /// <summary>ggml_permute(a0..a3): source axis i becomes axis a_i.</summary>
        private static Tensor Permute(Tensor x4, int a0, int a1, int a2, int a3)
        {
            int[] axes = { a0, a1, a2, a3 };
            var perm = new int[4];
            for (int i = 0; i < 4; i++) perm[3 - axes[i]] = 3 - i;
            return x4.Permute(perm);
        }

        private Tensor Same(Tensor src, Tensor like, RunState run)
        {
            if (src.Sizes.SequenceEqual(like.Sizes)) return src;
            if (src.ElementCount() != like.ElementCount()) throw new NotSupportedException("a copy between different element counts");
            // Leading unit extents only (a 4-D view of a 2-D box): the same strided elements, viewed without a copy. Made
            // contiguous first, a conv window's update ran a pitched copy of 6144 rows of 3 floats per layer on CUDA.
            int lead = 0;
            while (src.DimensionCount - lead > like.DimensionCount && src.Sizes[lead] == 1) lead++;
            if (lead > 0 && src.Sizes.Slice(lead).SequenceEqual(like.Sizes))
                return run.Keep(new Tensor(src.Sizes.Slice(lead).ToArray(), src.Strides.Slice(lead).ToArray(), src.Storage, src.StorageOffset));
            return run.Keep(Contig(src, run).View(like.Sizes));
        }

        /// <summary>ggml_set_rows(dst, src, ids): dst row ids[i] (along ne1) takes src row i, for every
        /// higher index.</summary>
        private Tensor SetRows(Tensor dst, Tensor src, Value ids, RunState run)
        {
            // Rows only on the device (a recorded step's varying input): written there.
            if (ids.Ints == null && ids.Tensor != null && TrySetRowsOnDevice(dst, src, ids.Tensor)) return dst;
            int[] rows = Ints(ids);
            Tensor d4 = As4(dst, run);
            Tensor s4 = As4(F32(src, run.Owned), run);
            int count = (int)s4.Sizes[2];
            if (rows.Length < count) throw new InvalidOperationException($"set_rows of {count} rows got {rows.Length} indices");
            for (int i = 0; i < count; i++)
                if (rows[i] < 0 || rows[i] >= d4.Sizes[2])
                    throw new InvalidOperationException($"set_rows index {rows[i]} outside {d4.Sizes[2]} rows");
            bool run1 = true;
            for (int i = 1; i < count; i++) run1 &= rows[i] == rows[0] + i;
            if (run1 && TryWriteRows(dst, s4, rows[0], count)) return dst;
            if (run1)
            {
                using Tensor target = d4.Narrow(2, rows[0], count);
                CopyInto(target, s4);
                return dst;
            }
            for (int i = 0; i < count; i++)
            {
                using Tensor target = d4.Narrow(2, rows[i], 1);
                using Tensor row = s4.Narrow(2, i, 1);
                CopyInto(target, row);
            }
            return dst;
        }

        /// <summary>ggml_set_rows with the row ids on the device ([n] I32 or I64, every higher index of dst one block): false
        /// where this executor has no kernel for it (the ids are then read on the host).</summary>
        protected virtual bool TrySetRowsOnDevice(Tensor dst, Tensor src, Tensor ids) => false;

        /// <summary>ggml_get_rows(a, ids): a [ne0, ne1, ne2, ne3], ids [n, ne2, ne3]: [ne0, n, ne2, ne3]; the ids are read
        /// on the host (executors with a gather kernel override it).</summary>
        protected virtual Tensor GetRows(Tensor src, Value ids, RunState run)
        {
            long[] na = Ne(src), ni = Ne(ids.Tensor);
            int[] index = Ints(ids);
            long n = ni[0], batches = ni[1] * ni[2];
            if (batches != na[2] * na[3]) throw new NotSupportedException("get_rows batches");
            var flat = new int[index.Length];
            for (long b = 0; b < batches; b++)
                for (long i = 0; i < n; i++)
                    flat[b * n + i] = checked((int)(b * na[1] + index[b * n + i]));
            // ggml's get_rows reads F16 rows as F32 (a family pooling an F16 key cache sums them as F32); the
            // host gather copies 4-byte elements, so an F16 source is widened first.
            Tensor s = Contig(src, run);
            if (s.ElementType == DType.Float16) s = F32(s, run.Owned);
            using Tensor s2 = s.View(na[1] * batches, na[0]);
            Tensor idx = IntTensor(flat, new long[] { flat.Length }, run);
            using Tensor r = Ops.IndexSelect(null, s2, idx);
            return r.View(Sizes(stackalloc long[] { na[0], n, ni[1], ni[2] }));
        }

        private Tensor Concat(Tensor a, Tensor b, int ggmlDim, RunState run)
        {
            Tensor a4 = run.Keep(Contig(a, run).View(Sizes4(Ne(a))));
            Tensor b4 = run.Keep(Contig(b, run).View(Sizes4(Ne(b))));
            using Tensor r = Ops.Concat(null, 3 - ggmlDim, a4, b4);
            return r.View(Sizes(Ne(r)));
        }

        /// <summary>view_kv_cache_window: ip = head_dim, cache_size, kv_heads, start, length, ...: the cache
        /// [kv_heads, cache_size, head_dim]'s rows start .. start + length (no wrap).</summary>
        private static Tensor KvWindow(Tensor cache, long[] ip)
        {
            long d = ip[0], cap = ip[1], heads = ip[2], start = ip[3] % ip[1], length = ip[4];
            if (start + length > cap) throw new NotSupportedException("a wrapped cache window");
            return Checked(new Tensor(new[] { heads, length, d }, new[] { cap * d, d, 1L }, cache.Storage, cache.StorageOffset + start * d), "cache window");
        }

        /// <summary>q/k/v of ggml's flash attention ([d, n, heads, 1]) as [heads, n, d] (a view).</summary>
        private static Tensor Heads(Tensor t, RunState run)
        {
            Tensor t4 = As4(t, run);
            if (t4.Sizes[0] != 1) throw new NotSupportedException("batched flash attention operands");
            return run.Keep(t4.Select(0, 0));
        }

        /// <summary>A mask [nkv, nq_pad(, 1, 1)] narrowed to <paramref name="rows"/> query rows: [rows, nkv].</summary>
        /// <summary>A mask's [rows, keys] view; it is cached for every layer, so it lives to the step's end.</summary>
        private static Tensor Mask(Tensor mask, long rows, RunState run)
        {
            long[] ne = Ne(mask);
            using Tensor m4 = mask.View(Sizes4(ne));
            using Tensor m3 = m4.Select(0, 0);
            Tensor m2 = m3.Select(0, 0);
            if (ne[1] == rows) return run.KeepLasting(m2);
            using (m2) return run.KeepLasting(m2.Narrow(0, 0, rows));
        }

        /// <summary>ggml_lightning_indexer on the host: q [d, heads, rows], keys [d, 1, n], weights [heads, rows],
        /// mask [n, rows]; [n, rows] = sum over heads of weight * relu(q . key) + mask.</summary>
        protected virtual Tensor LightningIndexer(Tensor q, Tensor keys, Tensor weights, Tensor mask, RunState run)
        {
            long[] nq = Ne(q), nk = Ne(keys);
            int d = (int)nq[0], heads = (int)nq[1], rows = (int)nq[2], n = (int)nk[2];
            float[] qv = HostFloats(q, run), kv = HostFloats(keys, run), wv = HostFloats(weights, run), mv = HostFloats(mask, run);
            var result = new float[(long)rows * n];
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int h = 0; h < heads; h++)
                    {
                        long qo = ((long)r * heads + h) * d, ko = (long)j * d;
                        double dot = 0;
                        for (int i = 0; i < d; i++) dot += qv[qo + i] * kv[ko + i];
                        if (dot > 0) sum += wv[(long)r * heads + h] * dot;
                    }
                    result[(long)r * n + j] = (float)sum + mv[(long)r * n + j];
                }
            var t = new Tensor(Allocator, DType.Float32, rows, n);
            t.SetElementsAsFloat(result);
            return t;
        }

        /// <summary>ggml_top_k_mask (see <see cref="GraphOp.TopKMask"/>): an F32 [n_kv, rows] mask; the picks, blocks and
        /// cells read on the host (executors with a kernel override it).</summary>
        protected virtual Tensor TopKMask(Tensor mask, Value picks, Tensor score, Value blocks, Value extra, RunState run)
            => TopKMask(mask, Ints(picks), Ne(picks.Tensor), score, blocks.Tensor == null && blocks.Ints == null ? null : Ints(blocks),
                blocks.Tensor == null && blocks.Ints == null ? null : Ne(blocks.Tensor),
                extra.Tensor == null && extra.Ints == null ? null : Ints(extra), extra.Tensor == null && extra.Ints == null ? null : Ne(extra.Tensor), run);

        private Tensor TopKMask(Tensor mask, int[] picks, long[] pickNe, Tensor score, int[] blocks, long[] blockNe,
            int[] extra, long[] extraNe, RunState run)
        {
            long[] nm = Ne(mask);
            int nkv = (int)nm[0], maskRows = (int)(mask.ElementCount() / nkv), top = (int)pickNe[0];
            int rows = (int)(picks.Length / top);
            float[] mv = HostFloats(mask, run), sv = score != null ? HostFloats(score, run) : null;
            int nPick = score != null ? (int)Ne(score)[0] : 0;
            int cells = blocks != null ? (int)blockNe[0] : 1;
            var result = new float[(long)rows * nkv];
            Array.Fill(result, float.NegativeInfinity);
            for (int r = 0; r < rows; r++)
            {
                long src = (long)(r % maskRows) * nkv, dst = (long)r * nkv;
                void Keep(int c)
                {
                    if (c >= 0 && c < nkv) result[dst + c] = mv[src + c];
                }
                for (int t = 0; t < top; t++)
                {
                    int pick = picks[(long)r * top + t];
                    if (sv != null && (pick < 0 || pick >= nPick || float.IsNegativeInfinity(sv[(long)r * nPick + pick]))) continue;
                    if (blocks == null) { Keep(pick); continue; }
                    if (pick < 0 || pick >= blockNe[1]) continue;
                    for (int c = 0; c < cells; c++) Keep(blocks[(long)pick * cells + c]);
                }
                if (extra != null)
                    for (int e = 0; e < extraNe[0]; e++) Keep(extra[(long)r * extraNe[0] + e]);
            }
            var t2 = new Tensor(Allocator, DType.Float32, rows, nkv);
            t2.SetElementsAsFloat(result);
            return t2;
        }

        /// <summary>A tensor's values on the host as F32 (an F16 mask converted).</summary>
        private float[] HostFloats(Tensor t, RunState run)
        {
            Tensor x = t.ElementType == DType.Float32 ? Contig(t, run) : F32(t, run.Owned);
            return x.GetElementsAsFloat((int)x.ElementCount());
        }

        /// <summary>ggml_top_k on the host (executors with a kernel override it; their result need not carry host ints).</summary>
        protected virtual Value TopK(Tensor x, int k, RunState run)
        {
            long cols = x.Sizes[^1], rows = x.ElementCount() / cols;
            float[] v = x.GetElementsAsFloat((int)x.ElementCount());
            var picked = new int[rows * k];
            var order = new int[cols];
            for (long r = 0; r < rows; r++)
            {
                for (int c = 0; c < cols; c++) order[c] = c;
                long b = r * cols;
                Array.Sort(order, (i, j) => v[b + j].CompareTo(v[b + i]) is var cmp && cmp != 0 ? cmp : i.CompareTo(j));
                Array.Copy(order, 0, picked, r * k, k);
            }
            return new Value { Tensor = IntTensor(picked, Sizes(stackalloc long[] { k, rows }), run), Ints = picked };
        }

        /// <summary>ggml's rotary over x [d, heads, tokens] at positions [tokens] (multi-axis: [4 * tokens],
        /// axis-major); ip = dims, mode, n_ctx_orig[, sections]; fp = base, freq_scale, ext_factor,
        /// attn_factor, beta_fast, beta_slow.</summary>
        private Tensor Rope(Tensor x, Value positions, GraphNode n, bool multi, RunState run, int positionsNode = -1)
        {
            long[] ne = Ne(x);
            int d = (int)ne[0], heads = (int)ne[1], tokens = (int)ne[2];
            int dims = (int)n.Ip[0], mode = (int)n.Ip[1], ctx = (int)n.Ip[2];
            float freqBase = n.Fp[0], freqScale = n.Fp[1], extFactor = n.Fp[2], attnFactor = n.Fp[3];
            // Read on the host only where the host needs them (multi-axis rows, or no device positions): a recorded step's
            // varying positions are never read back.
            int[] pos = multi || !BroadcastsPositionsOnDevice || positions.Tensor == null ? Ints(positions) : null;
            // Multi-axis rotary whose axes all carry the same position (every text token) is NeoX rotary at
            // that position: the device kernel, not the host loop below (which reads and writes the rows).
            if (multi && extFactor == 0f && pos.Length >= 3 * tokens && SameAxes(pos, tokens, n.Ip.Length > 6 && n.Ip[6] > 0 ? 4 : 3))
            {
                multi = false;
                mode = 2;   // GGML_ROPE_TYPE_NEOX
            }
            if (!multi)
            {
                if (!run.RowPositions.TryGetValue((positionsNode, heads), out Tensor p) || positionsNode < 0)
                {
                    Tensor device = positions.Tensor;
                    if (BroadcastsPositionsOnDevice && device != null && device.ElementType == DType.Int32 && device.IsContiguous()
                        && ReferenceEquals(device.Allocator, Allocator) && device.ElementCount() >= tokens)
                    {
                        // Every row of a token takes its position: a broadcast copy of the positions already on the
                        // device (no upload per step, so a recorded decode step reads the positions it is given).
                        p = run.KeepLasting(new Tensor(Allocator, DType.Int32, (long)tokens * heads));
                        using Tensor first = device.View(device.ElementCount()).Narrow(0, 0, tokens);
                        using Tensor column = first.View(tokens, 1);
                        using Tensor rows = column.Expand(tokens, heads);
                        using Tensor target = p.View(tokens, heads);
                        Ops.Copy(target, rows);
                    }
                    else
                    {
                        pos ??= Ints(positions);
                        var perRow = new int[tokens * heads];
                        for (int t = 0; t < tokens; t++)
                            for (int h = 0; h < heads; h++) perRow[t * heads + h] = pos[t];
                        p = run.KeepLasting(NewIntTensor(perRow, new long[] { perRow.Length }));
                    }
                    if (positionsNode >= 0) run.RowPositions[(positionsNode, heads)] = p;
                }
                if (KernelsReadStridedViews)
                {
                    using Tensor rotated = Ops.RoPEEx(null, As4(x, run), p, dims, mode, ctx, freqBase, freqScale, extFactor, attnFactor, n.Fp[4], n.Fp[5]);
                    return rotated.View(Sizes(ne));
                }
                Tensor copy = Ops.NewContiguous(x);
                using Tensor r4 = copy.View(1, tokens, heads, d);
                Ops.RoPEEx(r4, r4, p, dims, mode, ctx, freqBase, freqScale, extFactor, attnFactor, n.Fp[4], n.Fp[5]);
                return copy;
            }
            Tensor result = Ops.NewContiguous(x);
            if (extFactor != 0f) throw new NotSupportedException("YaRN multi-axis rotary");
            // Interleaved multi-axis rotary with NeoX pairs: pair i takes the H position when i mod 3 == 1
            // (i < 3 * sections[1]), W when 2 (< 3 * sections[2]), T when 0 (< 3 * sections[0]), else axis 3.
            int[] sections = { (int)n.Ip[3], (int)n.Ip[4], (int)n.Ip[5], n.Ip.Length > 6 ? (int)n.Ip[6] : 0 };
            int sectDims = sections.Sum(), half = dims / 2;
            float[] v = result.GetElementsAsFloat((int)result.ElementCount());
            float thetaScale = MathF.Pow(freqBase, -2.0f / dims);
            for (int t = 0; t < tokens; t++)
            {
                for (int i = 0; i < half; i++)
                {
                    int sector = i % sectDims;
                    int axis = sector % 3 == 1 && sector < 3 * sections[1] ? 1
                        : sector % 3 == 2 && sector < 3 * sections[2] ? 2
                        : sector % 3 == 0 && sector < 3 * sections[0] ? 0 : 3;
                    float theta = freqScale * pos[axis * tokens + t] * MathF.Pow(thetaScale, i);
                    float c = MathF.Cos(theta) * attnFactor, s = MathF.Sin(theta) * attnFactor;
                    for (int h = 0; h < heads; h++)
                    {
                        long b = ((long)t * heads + h) * d;
                        float x0 = v[b + i], x1 = v[b + i + half];
                        v[b + i] = x0 * c - x1 * s;
                        v[b + i + half] = x0 * s + x1 * c;
                    }
                }
            }
            result.SetElementsAsFloat(v);
            return result;
        }

        private static bool SameAxes(int[] pos, int tokens, int axes)
        {
            for (int a = 1; a < axes; a++)
                for (int t = 0; t < tokens; t++)
                    if (pos[a * tokens + t] != pos[t]) return false;
            return true;
        }

        /// <summary>The host gated delta rule (ggml-cpu's loop).</summary>
        protected Tensor GatedDeltaNetHost(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, Tensor state, int snapshots)
        {
            long[] nq = Ne(q), nv = Ne(v), ng = Ne(g);
            int sv = (int)nv[0], h = (int)nv[1], nt = (int)nv[2], ns = (int)nv[3], hq = (int)nq[1];
            bool kda = ng[0] == sv && sv > 1;
            float[] qv = q.GetElementsAsFloat((int)q.ElementCount()), kv = k.GetElementsAsFloat((int)k.ElementCount());
            float[] vv = v.GetElementsAsFloat((int)v.ElementCount()), gv = g.GetElementsAsFloat((int)g.ElementCount());
            float[] bv = beta.GetElementsAsFloat((int)beta.ElementCount()), s0 = state.GetElementsAsFloat((int)state.ElementCount());
            int slots = Math.Max(1, snapshots);
            long attn = (long)sv * h * nt * ns, perSnap = (long)sv * sv * h * ns;
            var outv = new float[attn + perSnap * slots];
            float scale = 1f / MathF.Sqrt(sv);
            var s = new float[(long)sv * sv];
            var delta = new float[sv];
            for (int seq = 0; seq < ns; seq++)
            {
                for (int head = 0; head < h; head++)
                {
                    int qh = head % hq;
                    Array.Copy(s0, ((long)seq * h + head) * sv * sv, s, 0, (long)sv * sv);
                    for (int t = 0; t < nt; t++)
                    {
                        long qo = (((long)seq * nt + t) * hq + qh) * sv;
                        long vo = (((long)seq * nt + t) * h + head) * sv;
                        long go = (((long)seq * nt + t) * h + head) * (kda ? sv : 1);
                        float b = bv[((long)seq * nt + t) * h + head];
                        for (int j = 0; j < sv; j++)
                            for (int i = 0; i < sv; i++)
                                s[(long)j * sv + i] *= MathF.Exp(kda ? gv[go + i] : gv[go]);
                        for (int j = 0; j < sv; j++)
                        {
                            float sum = 0;
                            for (int i = 0; i < sv; i++) sum += s[(long)j * sv + i] * kv[qo + i];
                            delta[j] = (vv[vo + j] - sum) * b;
                        }
                        for (int j = 0; j < sv; j++)
                            for (int i = 0; i < sv; i++) s[(long)j * sv + i] += kv[qo + i] * delta[j];
                        for (int j = 0; j < sv; j++)
                        {
                            float sum = 0;
                            for (int i = 0; i < sv; i++) sum += s[(long)j * sv + i] * qv[qo + i];
                            outv[vo + j] = sum * scale;
                        }
                        int slot = nt - 1 - t;
                        if (slot < slots)
                            Array.Copy(s, 0, outv, attn + slot * perSnap + ((long)seq * h + head) * sv * sv, (long)sv * sv);
                    }
                }
            }
            var result = new Tensor(Allocator, DType.Float32, outv.LongLength);
            result.SetElementsAsFloat(outv);
            return result;
        }
    }
}
