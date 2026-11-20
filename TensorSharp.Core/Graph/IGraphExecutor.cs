// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>What a backend's executor does beyond running a program.</summary>
    public sealed record GraphCapabilities
    {
        /// <summary>Its kernels read row-contiguous strided views directly (no contiguous copies needed).</summary>
        public bool StridedViews { get; init; }
        /// <summary>A prompt chunk's packed projection is one product whose row blocks its element-wise ops read where they lie
        /// (strided views, no copies), as decode rows' is; otherwise a chunk takes each block as its own product, so each is
        /// contiguous.</summary>
        public bool PackedProductBlocks { get; init; }
        /// <summary>An op may write its result in place inside a state leaf (<see cref="GraphNodeKind.Alias"/>).</summary>
        public bool InPlaceStateAlias { get; init; }
        /// <summary>A sliding-window attention over a capacity window or a ring (<see cref="GraphOp.FlashAttnExt"/> ip[3]: its band)
        /// reads its mask on the device as the selection it is, every run: the band's floor moves with each position, so a
        /// recorded decode step replays the band of the position it runs at (a mask classified from its host values would
        /// replay the recorded position's floor). Only for windows this executor attends that way
        /// (<see cref="IGraphExecutor.AttendsCapacityWindows"/>).</summary>
        public bool BandMasksOnDevice { get; init; }

        /// <summary>A verify window's recurrence writes only its rows' outputs and saves its scan inputs (a gated delta rule
        /// with destination leaf 6), the live state read and left as it was: the commit steps the kept rows on it again, in
        /// place (the direct engines' verify scan and replay). Elsewhere the window keeps a state after every row.</summary>
        public bool WindowStateInPlace { get; init; }
        /// <summary>A verify window's delta-net regions (prep, gated norm) read their inputs where a packed in-projection put
        /// them, a packed row apart: the window takes the projection as one product, as decode rows do, instead of a product
        /// per row block.</summary>
        public bool WindowRegionsReadPackedRows { get; init; }
        /// <summary>Its recorded steps bind a slot's buffers: a request that runs in a new slot relocates or records them again
        /// (tens of milliseconds a step shape), so a finished request is retained as a copy and its slot serves the next.</summary>
        public bool SlotBoundRecordings { get; init; }
        /// <summary>Reordering independent work pays off (and is safe) on this device.</summary>
        public bool ConcurrencyReorder { get; init; }
        /// <summary>The bucket a decode step's padded attention window grows by (a window's programs replay).</summary>
        public int DecodeWindowRows { get; init; } = 256;
        /// <summary>Programs may spread over several ranks (<see cref="GraphProgram.TensorParallelRanks"/>).</summary>
        public bool TensorParallel { get; init; }
        /// <summary>Prompt chunks write K/V by row index (no copy into a strided cache view).</summary>
        public bool IndexedCacheWrites { get; init; }
        /// <summary>A prompt chunk that attends its own fresh K/V rows takes them as the permuted views of its projections: it
        /// writes them into the caches by index from those views, and its attention (and the concatenation with the cached
        /// prefix) reads them where they lie, instead of contiguous copies of every chunk's K and V first. For executors whose
        /// set_rows, concat and flash attention read rows at any strides (ggml-cpu).</summary>
        public bool FreshRowViews { get; init; }
        /// <summary>A prompt chunk over an F16 or block-quantized K/V cache is written by row index and attends the
        /// cache's window padded to <see cref="DecodeWindowRows"/>, as decode does, instead of its own fresh F32 rows:
        /// the device's fast attention needs reduced-precision K/V in whole windows. An F32 cache keeps the fresh rows
        /// (exact, and no faster that way).</summary>
        public bool PaddedCacheAttention { get; init; }
        /// <summary>K/V caches may be block-quantized (Q8_0, Q4_0): views and writes address their blocks.</summary>
        public bool QuantizedKvCache { get; init; }
        /// <summary>It runs <see cref="GraphOp.MoeHostExperts"/>: routed experts computed from host memory at a
        /// pause in the program.</summary>
        public bool HostExperts { get; init; }

        /// <summary>A decode step's <see cref="GraphOp.MoeHostExperts"/> stays on the device's stream (the host answers a
        /// doorbell while the device waits), so a step with host experts can be recorded and replayed.</summary>
        public bool RecordsHostExperts { get; init; }

        /// <summary>The executor's kernels take a family's fusion regions at any row count, decode rows too (a family marks
        /// some regions for prompt chunks only, where the backends that fuse them had prompt-width kernels).</summary>
        public bool RegionsAtAnyWidth { get; init; }
        /// <summary>A compiled plan is worth keeping to run again (a replayable program is cached and
        /// looked up by its signature); otherwise every step is recorded and compiled once.</summary>
        public bool Replays { get; init; } = true;
        /// <summary>Leaves are bound by their host bytes (GGML's device caches are keyed by host
        /// address); otherwise by the leaf's object (<see cref="GraphNode.Handle"/>), and a leaf's
        /// <see cref="GraphNode.Data"/> is only an identity.</summary>
        public bool BindsHostPointers { get; init; }
        /// <summary>A leaf bound by its host bytes is computed on through a separate device copy, moved to and from
        /// the host by syncs; false where the device reads the host bytes themselves (Metal's shared buffers, the
        /// CPU), so a host copy of them moves nothing across.</summary>
        public bool SeparateDeviceCopies { get; init; }
        /// <summary>A program may end in each row's greedy token (a top-1 over the logits, a tie to the lower id), read
        /// back as its F32 value: a greedy verify then reads an id a row instead of the vocabulary's logits. The head is
        /// <see cref="GraphOp.TopK"/> with Ip { 1, 1 } (k 1, greedy) read through a <see cref="GraphOp.Cont"/> as one float a
        /// row; an executor lowers it to whatever gives exactly those ids (GGML: each row's top candidates on the device,
        /// the lowest id of the largest taken on the host, a row whose candidates all tie read in full).</summary>
        public bool GreedyIdOutputs { get; init; }
        /// <summary>A greedy head (<see cref="GreedyIdOutputs"/>) may take a second source, an F32 [vocab] mask of -0 (kept) and
        /// -inf (suppressed) added to every row before the pick: the ids are then the host argmax over the logits with the
        /// suppressed tokens' set to -inf, as TokenSampler sets them.</summary>
        public bool GreedyIdSuppression { get; init; }
        /// <summary>An F16 causal mask input may arrive as its pattern (a <see cref="CausalMaskInput"/> as the input's handle,
        /// no host bytes): the executor classifies it from the pattern and makes its values only for an op that reads them.
        /// The shared runtime gives prompt chunks' masks so (more rows than a verify window: never replayed, advanced or
        /// recorded); decode rows and verify windows keep their host values, which their advances move on.</summary>
        public bool StructuralCausalMasks { get; init; }
    }

    /// <summary>A stable identity for a storage object, for program signatures of executors that bind
    /// leaves by object (taking a device storage's host pointer would sync it).</summary>
    public static class GraphIdentity
    {
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, object> s_ids = new();
        private static long s_next = 1L << 40;

        public static IntPtr Of(object storage, long offset = 0)
        {
            object id = s_ids.GetValue(storage, _ => System.Threading.Interlocked.Increment(ref s_next));
            return new IntPtr(((long)id << 8) + offset);
        }
    }

    /// <summary>
    /// A backend's graph executor: the plugin through which every model runs. It compiles a
    /// backend-neutral <see cref="GraphProgram"/> into a plan for its devices and keeps the
    /// leaves' bytes coherent between the host and the device.
    /// </summary>
    public interface IGraphExecutor
    {
        GraphCapabilities Capabilities { get; }

        /// <summary>Compile <paramref name="program"/>; throws <see cref="NotSupportedException"/> when it
        /// uses something this backend cannot run.</summary>
        IGraphPlan Compile(GraphProgram program);

        /// <summary>The newest bytes of a state leaf at <paramref name="data"/> are copied back to the host.</summary>
        void SyncToHost(IntPtr data, long bytes);

        /// <summary>The host bytes at <paramref name="data"/> changed (or are being freed): drop device copies.</summary>
        void Invalidate(IntPtr data);

        /// <summary>A weight the model reads only through its products (an untied LM head, never gathered by rows), by its
        /// cache key: the executor may keep it in a layout only its own products read. False where it keeps the weight as it
        /// is.</summary>
        bool KeepForProductsOnly(IntPtr weightKey) => false;

        /// <summary>A weight only a drafter reads: its products make proposals a verify checks, in whichever step they run, so
        /// they may take the kernels a <see cref="Drafting"/> step's products take there too (the trunk's prompt step that writes
        /// a block drafter's ring). False where the executor runs them as any other weight's (it reads nothing of the weight:
        /// a weight's device cache key is made on first use by the executors that key on it).</summary>
        bool MarkDrafterWeight(IGraphWeight weight) => false;

        /// <summary>Make <paramref name="weights"/> resident on the device now, through the backend's bulk uploader, so the first
        /// program that binds them (as weight leaves of the same host bytes) uploads nothing: a component's load (a diffusion
        /// denoiser's weights before its first step). Returns how many were copied now; 0 where the executor binds weights as
        /// its programs reach them (it keeps no device copies, or reads host memory in place).</summary>
        int Preload(IReadOnlyList<IGraphWeight> weights, string tag) => 0;

        /// <summary>Run <paramref name="weight"/>'s product over zero rows once for each count in <paramref name="rows"/>, the
        /// results discarded, so whatever a product of that many rows loads at its first launch (a library's kernels the driver
        /// loads lazily) is loaded before a request's step runs it. False where the executor does nothing.</summary>
        bool WarmProducts(IGraphWeight weight, IReadOnlyList<int> rows) => false;

        /// <summary>Whether decode rows and short chunks over a K/V cache of <paramref name="kvFormat"/> with
        /// <paramref name="headDim"/>-wide heads may attend their slot's whole capacity as their padded window: the executor's
        /// kernels then compute, from the rows' written length (read on the device when the step is recorded), what the window
        /// of <see cref="GraphCapabilities.DecodeWindowRows"/>-row buckets around that length computes, bit for bit - so one
        /// recorded step of a width serves every position of its slot. False where a window's width shapes the arithmetic.</summary>
        bool AttendsCapacityWindows(int headDim, int kvFormat) => false;

        /// <summary>A scope in which a one-shot step's run hands the host nothing back returns with its device work queued, so
        /// the host traces and builds the next step while the device runs this one (a prompt's steps of one sequence, as
        /// llama.cpp's scheduler overlaps a prompt's ubatches): the work is waited for before anything reads, releases or
        /// rewrites what it touches, and when the scope ends. Null where every run completes before it returns.</summary>
        IDisposable OverlapSteps() => null;

        /// <summary>Whether a model's whole-forward graphs may hold their K/V caches in <paramref name="kvFormat"/> (a
        /// <see cref="GraphFormat"/>) with <paramref name="keyDim"/>-wide keys and <paramref name="valueDim"/>-wide values: F32 and
        /// F16 always, the block-quantized Q8_0 and Q4_0 where every view and write addresses their blocks
        /// (<see cref="GraphCapabilities.QuantizedKvCache"/>). An executor whose attention kernels and row writes read and write a
        /// block format without the generic views of it (direct CUDA's q8_0) says so here.</summary>
        bool TakesKvCacheFormat(int kvFormat, int keyDim, int valueDim)
            => kvFormat is GraphFormat.F32 or GraphFormat.F16
               || (kvFormat is GraphFormat.Q8_0 or GraphFormat.Q4_0 && Capabilities.QuantizedKvCache);

        /// <summary>Host work to run while the device runs the next recorded step: after its launch, before anything waits for
        /// its outputs (a recorded step leaves the host idle for its whole run). False where the executor runs no recorded
        /// steps to hide it behind: the caller then does the work itself, or not at all.</summary>
        bool RunWhileNextRecordedStepRuns(Action work) => false;

        /// <summary>The steps run now are a drafter's: its proposals are verified, not trusted, so its products need not be
        /// row-invariant (a verify window's rows a plain decode step's bit for bit) and may take faster kernels whose last
        /// bits vary. Set around a draft step's runs, its recordings' included.</summary>
        bool Drafting { get => false; set { } }

        /// <summary>The sequences of the steps run now (0: not said). Set around a step's runs, its recordings' included: an
        /// executor whose row-invariant products cost more past a few rows keeps them for the rows of one sequence's window
        /// (several sequences' rows past them take its multi-row products, as they did before).</summary>
        int StepSequences { get => 0; set { } }
    }

    /// <summary>A compiled program. Outputs are the plan's arrays, rewritten by each run.</summary>
    public interface IGraphPlan : IDisposable
    {
        float[][] Outputs { get; }

        /// <summary>Run with <paramref name="inputs"/>' values (node, host bytes, byte count); false with
        /// <see cref="LastError"/> set on failure.</summary>
        bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes);

        /// <summary>As <see cref="Run(ReadOnlySpan{int}, ReadOnlySpan{IntPtr}, ReadOnlySpan{long})"/>, with an
        /// optional object per input (a <see cref="Tensor"/> an executor binding objects reads directly).</summary>
        bool Run(ReadOnlySpan<int> inputs, ReadOnlySpan<IntPtr> data, ReadOnlySpan<long> bytes, ReadOnlySpan<object> handles)
            => Run(inputs, data, bytes);

        /// <summary>Leaf <paramref name="node"/> is bound to device memory that outlives a run.</summary>
        bool IsLeafResident(int node);

        string LastError { get; }
    }

    /// <summary>Receives a program's entries as they are recorded (node index -1 for entries that are
    /// not nodes), and each input leaf's per-run value right after the leaf.</summary>
    public interface IGraphSink
    {
        void Entry(int node, in GraphNode entry);
        void InputData(int node, IntPtr data, long bytes, object handle);
    }

    /// <summary>
    /// A program that runs as it is recorded: its nodes are evaluated as they arrive (the emitting code
    /// runs inside <see cref="Execute"/>, on the backend's thread), and <see cref="IGraphPlan.Run"/>
    /// then only reads the outputs. For executors whose plans would not replay anyway, so recording
    /// overlaps the device's work instead of preceding it.
    /// </summary>
    public interface IGraphStream : IGraphSink, IGraphPlan
    {
        bool Execute(Func<bool> body);
    }

    /// <summary>An executor that can also run programs as they are recorded.</summary>
    public interface IStreamingGraphExecutor : IGraphExecutor
    {
        /// <summary>A stream for one step. <paramref name="template"/>, when given, is a recorded program of
        /// the same kind: the stream releases each node's tensors after its last reader (it checks that the
        /// step matches the template node by node, and keeps everything to the end where it does not).</summary>
        IGraphStream OpenStream(GraphProgram? template = null);
    }

    /// <summary>One rank's inputs for <see cref="IRankedGraphExecutor.RunRanked"/>: the input nodes with their per-run
    /// host bytes and byte counts.</summary>
    public readonly record struct GraphRunInputs(int[] Nodes, IntPtr[] Data, long[] Bytes);

    /// <summary>
    /// An executor whose compiled plans of one tensor-parallel step run together: plan r, compiled on rank r, runs
    /// with its inputs while every rank pauses at each <see cref="GraphOp.AllReduce"/> to sum the partial in place
    /// (on the devices where the backend has a collective), then continues. For executors that compile whole
    /// programs; a streaming executor reduces as the ranks' nodes arrive instead.
    /// </summary>
    public interface IRankedGraphExecutor : IGraphExecutor
    {
        /// <summary>Whether steps over <paramref name="ranks"/> devices can run here (the devices' collective is up).</summary>
        bool RunsRanks(int ranks);

        /// <summary>Compile <paramref name="program"/> for rank <paramref name="rank"/>'s device.</summary>
        IGraphPlan CompileForRank(GraphProgram program, int rank);

        /// <summary>Run every rank's plan together; false with <paramref name="error"/> on failure. <paramref name="exactSums"/>:
        /// sum the partials in F32 whatever the transport would round (a fast transport may carry large sums in BF16).</summary>
        bool RunRanked(IGraphPlan[] plans, GraphRunInputs[] inputs, bool exactSums, out string error);
    }

    /// <summary>A plan whose outputs can stay on the device: set <see cref="KeepTensors"/> before a run and
    /// take each output as a tensor after it (no host copy, no wait for the device).</summary>
    public interface IGraphTensorOutputs
    {
        bool KeepTensors { get; set; }

        /// <summary>Output <paramref name="index"/> of the last run (the caller owns it).</summary>
        Tensor TakeOutput(int index);
    }

    /// <summary>
    /// What one run of a plan evaluates and moves, for censuses of steps (eng/lfm3/Lfm3Probe batch): the executor's nodes, those
    /// that run a kernel (not a view, reshape, permute or transpose), and the leaves besides the inputs that every run copies to
    /// the device and back (ggml: states and weights placed in the graph's compute memory) with their bytes.
    /// </summary>
    public readonly record struct GraphPlanCensus(long Nodes, long Kernels, long LeafUploads, long LeafUploadBytes, long LeafDownloads,
        long LeafDownloadBytes);

    /// <summary>A plan that reports its <see cref="GraphPlanCensus"/>. Read only by censuses: nothing the runtime decides.</summary>
    public interface IGraphPlanCensus
    {
        /// <summary>One run's census, or null where the plan does not know it yet (a graph built on its first run).</summary>
        GraphPlanCensus? Census { get; }
    }

    /// <summary>Executor plugins, by the allocator type of the backend they serve.</summary>
    public static class GraphExecutors
    {
        private static readonly ConcurrentDictionary<Type, Func<IAllocator, IGraphExecutor>> s_factories = new();

        /// <summary>Register the executor of backends whose allocators are <paramref name="allocatorType"/>
        /// (a backend assembly calls this from its module initializer).</summary>
        public static void Register(Type allocatorType, Func<IAllocator, IGraphExecutor> factory)
            => s_factories[allocatorType ?? throw new ArgumentNullException(nameof(allocatorType))] =
                factory ?? throw new ArgumentNullException(nameof(factory));

        /// <summary>The executor for <paramref name="allocator"/>'s backend, or null when none is registered.</summary>
        public static IGraphExecutor? For(IAllocator? allocator)
        {
            if (allocator == null) return null;
            for (Type? t = allocator.GetType(); t != null; t = t.BaseType)
                if (s_factories.TryGetValue(t, out Func<IAllocator, IGraphExecutor>? factory))
                    return factory(allocator);
            return null;
        }
    }
}
