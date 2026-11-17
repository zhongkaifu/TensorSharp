// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// The gated short convolution of LiquidAI's LFM2 / LFM2.5 / LFM3 conv layers, once for every family that has it:
    /// <c>[b | c | x] = in_proj · h</c>, <c>y = c * conv(b * x)</c> with a causal depthwise convolution of width L (no bias,
    /// no activation) over the products, carried across steps as the last L - 1 of them, then <c>out_proj · y</c>. The state
    /// (<see cref="ShortConvState"/>) holds the window per (sequence, layer, rank) like any recurrent state, so slots,
    /// checkpoints, prefix reuse, tensor parallelism (channel-sharded: each rank convolves its channels) and speculation
    /// (a verify window keeps its inputs; the commit copies the accepted rows' window) work as for every hybrid stack.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>
        /// One rank's short-conv window for one sequence: <see cref="Conv"/> [channels, L - 1] (time innermost, oldest first;
        /// ggml_ssm_conv's window layout), the last L - 1 products b * x of the sequence (zeros before its first). A verify window
        /// leaves it as it was and writes its inputs into <see cref="ConvHistory"/>; the commit takes the accepted rows' window
        /// from there. Where the executor binds host bytes (GGML) the tensors are host-backed and the device copy is authoritative
        /// between steps (as <c>Qwen35RuntimeModel.DeviceGdnState</c>'s); elsewhere they are the backend's own tensors.
        /// </summary>
        protected internal sealed class ShortConvState : RecurrentState
        {
            private readonly IAllocator _allocator;
            private readonly IGraphExecutor _executor;
            private readonly bool _host;

            public ShortConvState(IAllocator allocator, IGraphExecutor executor, int channels, int convRows)
            {
                if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
                if (convRows <= 0) throw new ArgumentOutOfRangeException(nameof(convRows), convRows, "a short conv carries at least one column");
                _allocator = allocator;
                _executor = executor;
                _host = executor?.Capabilities.BindsHostPointers ?? true;
                Channels = channels;
                ConvRows = convRows;
                Conv = new Tensor(allocator, DType.Float32, channels, convRows);
                // A fresh mapping is zero already (clearing it would only fault its pages in).
                if (_host && Conv.Storage.TakeAllocationZeroGuarantee()) Invalidate(Conv);
                else Clear(Conv);
            }

            public Tensor Conv { get; }
            public int Channels { get; }
            public int ConvRows { get; }

            /// <summary>A verify window's conv inputs [channels, ConvRows + n] at the window's own pitch (room for ConvRows +
            /// <see cref="SnapshotCapacity"/> a channel): columns [k, k + ConvRows) are the window after k of its rows.</summary>
            public Tensor ConvHistory { get; private set; }
            public int SnapshotCapacity { get; private set; }

            /// <summary>Room for a <paramref name="rows"/>-row verify window; true when the buffer moved.</summary>
            public bool EnsureSpecBuffers(int rows)
            {
                if (rows <= SnapshotCapacity) return false;
                int capacity = 4;
                while (capacity < rows) capacity *= 2;
                Free(ConvHistory);
                ConvHistory = new Tensor(_allocator, DType.Float32, Channels, ConvRows + capacity);
                SnapshotCapacity = capacity;
                NoteBindingsChanged();
                return true;
            }

            protected internal override void ReleaseSpecBuffers()
            {
                if (ConvHistory == null) return;
                NoteBindingsChanged();
                Free(ConvHistory);
                ConvHistory = null;
                SnapshotCapacity = 0;
            }

            protected internal override long StateBytes => Conv.Storage.ByteLength + (ConvHistory?.Storage.ByteLength ?? 0);

            public override void Reset() => Clear(Conv);

            public override bool ResetInPlace()
            {
                if (!_host)
                {
                    Reset();   // the backend's own tensor, filled where it lives
                    return true;
                }
                unsafe
                {
                    IntPtr p = HostPointer(Conv);
                    new Span<float>((float*)p, checked((int)Conv.ElementCount())).Clear();
                    GgmlBasicOps.UploadHostBufferRanges(p, stackalloc long[] { 0 }, stackalloc long[] { Conv.ElementCount() * sizeof(float) });
                }
                return true;
            }

            protected internal override IEnumerable<Tensor> HostTensors => new[] { Conv };

            protected internal override bool CanCopy => true;

            protected internal override IEnumerable<Tensor> CheckpointTensors => new[] { Conv };

            /// <summary>Graphs bind the window; a copy on the device copies it.</summary>
            protected internal override Tensor[] GraphTensors => _host ? new[] { Conv } : null;

            protected internal override bool GraphTensorsResetToZero => true;

            protected internal override unsafe void CopyFrom(RecurrentState source)
            {
                var other = (ShortConvState)source;
                if (!_host)
                {
                    Ops.Copy(Conv, other.Conv);
                    return;
                }
                long bytes = Conv.ElementCount() * sizeof(float);
                Buffer.MemoryCopy((void*)HostPointer(other.Conv), (void*)HostPointer(Conv), bytes, bytes);
                OnHostWritten();
            }

            protected internal override unsafe bool CopyFromInPlace(RecurrentState source)
            {
                if (!_host)
                {
                    CopyFrom(source);
                    return true;
                }
                var other = (ShortConvState)source;
                long bytes = Conv.ElementCount() * sizeof(float);
                Buffer.MemoryCopy((void*)HostPointer(other.Conv), (void*)HostPointer(Conv), bytes, bytes);
                GgmlBasicOps.UploadHostBufferRanges(HostPointer(Conv), stackalloc long[] { 0 }, stackalloc long[] { bytes });
                return true;
            }

            protected internal override void OnHostWritten()
            {
                if (_host) Invalidate(Conv);
            }

            /// <summary>The window's values [channels, ConvRows] (host-current: the runtime syncs a state before host code).</summary>
            public float[] ReadWindow() => Conv.GetElementsAsFloat(checked(Channels * ConvRows));

            /// <summary>Write the window from the host (its device copies drop).</summary>
            public void WriteWindow(float[] values)
            {
                Conv.SetElementsAsFloat(values);
                OnHostWritten();
            }

            // ---- a per-op verify window (BeginVerify .. CommitVerify): the window before it and the window's products.
            private float[] _verifyWindow, _verifyProducts;
            private int _verifyRows;

            /// <summary>A per-op verify window is open on this state.</summary>
            public bool Verifying { get; private set; }

            /// <summary>The window after k rows is the last ConvRows of [window before | the rows' products] - no replay.</summary>
            protected internal override bool SupportsVerify => true;

            protected internal override void BeginVerify()
            {
                if (_host) _executor?.SyncToHost(HostPointer(Conv), Conv.ElementCount() * sizeof(float));
                _verifyWindow = ReadWindow();
                _verifyProducts = null;
                _verifyRows = 0;
                Verifying = true;
            }

            /// <summary>The per-op mixer ran a verify window's <paramref name="rows"/> products b * x ([rows, channels], row-major).</summary>
            public void RecordVerifyProducts(float[] products, int rows)
            {
                _verifyProducts = products;
                _verifyRows = rows;
            }

            protected internal override void CommitVerify(int kept)
            {
                if (!Verifying) return;
                Verifying = false;
                try
                {
                    if (_verifyProducts == null || kept >= _verifyRows) return;   // the live window is the window's end
                    WriteWindow(AdvanceWindow(_verifyWindow, _verifyProducts, kept, Channels, ConvRows));
                }
                finally
                {
                    _verifyWindow = _verifyProducts = null;
                    _verifyRows = 0;
                }
            }

            /// <summary>The window [channels, convRows] after <paramref name="rows"/> more products ([rows, channels] row-major):
            /// per channel the last convRows of [window | products].</summary>
            internal static float[] AdvanceWindow(float[] window, float[] products, int rows, int channels, int convRows)
            {
                var next = new float[channels * convRows];
                for (int c = 0; c < channels; c++)
                    for (int r = 0; r < convRows; r++)
                    {
                        int t = rows + r;   // the column of [window | products] that lands at r
                        next[c * convRows + r] = t < convRows ? window[c * convRows + t] : products[(t - convRows) * channels + c];
                    }
                return next;
            }

            private static IntPtr HostPointer(Tensor t) => TensorComputePrimitives.GetStoragePointer(t);

            private void Invalidate(Tensor t)
            {
                _executor?.Invalidate(HostPointer(t));
                GgmlBasicOps.InvalidateTensorHostBuffer(t);
            }

            private unsafe void Clear(Tensor t)
            {
                if (!_host)
                {
                    Ops.Fill(t, 0);
                    return;
                }
                new Span<float>((float*)HostPointer(t), checked((int)t.ElementCount())).Clear();
                Invalidate(t);
            }

            /// <summary>Dispose <paramref name="t"/> with the device copies graphs made of its host bytes (keyed by host pointer,
            /// they would outlive it, and an allocation reusing the address could bind a stale one).</summary>
            private void Free(Tensor t)
            {
                if (t == null) return;
                if (_host) Invalidate(t);
                t.Dispose();
            }

            public override void Dispose()
            {
                Free(Conv);
                Free(ConvHistory);
                ConvHistory = null;
            }
        }

        // ---------------------------------------------------------------- the graph mixer

        /// <summary>
        /// A short-conv layer's mixer over the normed rows <paramref name="normed"/> [hidden, rows] of the step, each sequence on its
        /// slot's <see cref="ShortConvState"/>: <paramref name="inProj"/> gives [b | c | x] ([3 * channels, rows] in the rank's
        /// channels), <paramref name="kernel"/> is the depthwise filter leaf [L, channels], <paramref name="outProj"/> the output
        /// projection (row-parallel under tensor parallelism: the runtime reduces its partials). A decode row's window updates in
        /// place; a verify window (<see cref="GraphStep.RecurrentSnapshots"/>) leaves it and writes its inputs into the state's
        /// history for the commit. Returns [hidden, rows].
        /// </summary>
        protected int EmitShortConvMixer(GraphStep step, int layer, int normed, string inProj, int kernel, string outProj, int channels, int convRows)
        {
            const long f = sizeof(float);
            long width = channels;
            // Decode rows (every sequence one row) take the packed product once, and so do prompt chunks where the executor's
            // element-wise ops read its row blocks where they lie (GraphCapabilities.PackedProductBlocks: one product of 3C rows,
            // as llama.cpp's graph takes it, not three); elsewhere a chunk takes the three row blocks as their own products, so
            // each is contiguous for the executors that copy strided operands.
            bool packedBlocks = GraphExecutor?.Capabilities.PackedProductBlocks == true;
            bool split = step.Rows > step.Sequences && !packedBlocks;
            int proj = -1, bAll = -1, cAll = -1, xAll = -1;
            if (split)
            {
                bAll = step.MatMulRows(inProj, 0, width, normed);
                cAll = step.MatMulRows(inProj, width, width, normed);
                xAll = step.MatMulRows(inProj, 2 * width, width, normed);
            }
            else
            {
                proj = step.MatMul(inProj, normed);   // [3 * channels, rows]
            }
            long packed = 3 * width * f;
            int output = -1;
            for (int s = 0; s < step.Sequences; s++)
            {
                int count = step.Counts[s];
                long first = step.First[s];
                int b, c, x;
                if (split)
                {
                    bool one = step.Sequences == 1;
                    b = one ? bAll : step.View2(bAll, width, count, width * f, first * width * f);
                    c = one ? cAll : step.View2(cAll, width, count, width * f, first * width * f);
                    x = one ? xAll : step.View2(xAll, width, count, width * f, first * width * f);
                }
                else
                {
                    b = step.View2(proj, width, count, packed, first * packed);
                    c = step.View2(proj, width, count, packed, first * packed + width * f);
                    x = step.View2(proj, width, count, packed, first * packed + 2 * width * f);
                    if (!packedBlocks)
                    {
                        b = step.Cont(b);
                        c = step.Cont(c);
                        x = step.Cont(x);
                    }
                }
                var state = (ShortConvState)step.SequenceState(s, layer);
                int window = step.State(state.Conv, GraphFormat.F32, convRows, channels);
                // ggml_concat writes a contiguous result, so it reads the transposed products directly.
                int bx = step.Op(GraphOp.Mul, b, x);                                                                   // [C, count]
                int convIn = step.Op(GraphOp.Concat, stackalloc[] { window, step.Transpose(bx) }, stackalloc long[] { 0 });   // [R + count, C]
                int conv = step.Op(GraphOp.SsmConv, convIn, kernel);                                                   // [C, count]
                int y = step.Op(GraphOp.Mul, c, conv);
                long pitch = (convRows + count) * f;
                if (step.RecurrentSnapshots && count > 1)
                {
                    // A verify window leaves the live window as it was: its inputs at the window's own pitch, from which the commit
                    // takes the accepted rows' window (CommitShortConvWindows).
                    if (state.EnsureSpecBuffers(count)) NoteGraphBindingsChanged();
                    int history = step.State(state.ConvHistory, GraphFormat.F32, convRows + state.SnapshotCapacity, channels);
                    step.Effect(step.Op(GraphOp.Cpy, convIn, step.View2(history, convRows + count, channels, pitch, 0)));
                }
                else
                {
                    // convIn is a fresh contiguous [R + count, C]: its last R columns are the new window.
                    step.Effect(step.Op(GraphOp.Cpy, step.View2(convIn, convRows, channels, pitch, count * f), window));
                }
                output = output < 0 ? y : step.Op(GraphOp.Concat, stackalloc[] { output, y }, stackalloc long[] { 1 });
            }
            return step.MatMul(outProj, output);
        }

        // ---------------------------------------------------------------- the per-op mixer

        /// <summary>
        /// The short-conv mixer on the per-op path for one rank and one sequence (<see cref="RecurrentMixer"/>): the in-projection
        /// through <see cref="Linear"/> (LoRA and tensor-parallel shards apply), the products, the convolution with the window and
        /// the gate on the host in F32 - each output summing its L taps in ggml_ssm_conv's order - then the output projection.
        /// <paramref name="kernel"/> is the rank's filter [channels, L] (taps innermost). Advances <paramref name="state"/>, or,
        /// in a per-op verify window, records its products for the commit.
        /// </summary>
        protected Tensor ShortConvMixer(RankScope scope, Tensor normed, ShortConvState state, string inProj, float[] kernel, string outProj)
        {
            int tokens = (int)normed.Sizes[0], channels = state.Channels, convRows = state.ConvRows, taps = convRows + 1;
            if (kernel.Length != channels * taps)
                throw new ArgumentException($"{Config.Architecture}: a short-conv filter of {kernel.Length} values for {channels} channels x {taps} taps.");
            float[] bcx;
            using (Tensor projected = Linear(scope, normed, inProj))
                bcx = projected.GetElementsAsFloat(checked(tokens * 3 * channels));
            float[] window = state.ReadWindow();
            var products = new float[tokens * channels];
            for (int t = 0; t < tokens; t++)
            {
                int row = t * 3 * channels;
                for (int ch = 0; ch < channels; ch++)
                    products[t * channels + ch] = bcx[row + ch] * bcx[row + 2 * channels + ch];
            }
            var y = new float[tokens * channels];
            Parallel.For(0, channels, ch =>
            {
                for (int t = 0; t < tokens; t++)
                {
                    float sum = 0;
                    for (int k = 0; k < taps; k++)
                    {
                        int at = t + k;   // column of [window | products]
                        float v = at < convRows ? window[ch * convRows + at] : products[(at - convRows) * channels + ch];
                        sum += v * kernel[ch * taps + k];
                    }
                    y[t * channels + ch] = bcx[t * 3 * channels + channels + ch] * sum;
                }
            });
            if (state.Verifying) state.RecordVerifyProducts(products, tokens);
            state.WriteWindow(ShortConvState.AdvanceWindow(window, products, tokens, channels, convRows));

            using var gated = new Tensor(scope.Allocator, DType.Float32, tokens, channels);
            gated.SetElementsAsFloat(y);
            HostWritten(gated);
            return Linear(scope, gated, outProj);
        }

        // ---------------------------------------------------------------- verify commits

        /// <summary>Commit programs of graph verify windows by (first history's identity, kept, rows): copies only, no inputs.</summary>
        private readonly Dictionary<(IntPtr, int, int), IGraphPlan> _shortConvCommits = new();

        private void ClearShortConvCommits()
        {
            foreach (IGraphPlan plan in _shortConvCommits.Values) plan.Dispose();
            _shortConvCommits.Clear();
        }

        /// <summary>A per-op verify window is open on one of the active request's short-conv states (they commit themselves).</summary>
        private bool ShortConvPerOpVerifyOpen()
        {
            foreach (RecurrentState state in ActiveRecurrentStates())
                if (state is ShortConvState { Verifying: true }) return true;
            return false;
        }

        /// <summary>
        /// The active request's short-conv windows after <paramref name="kept"/> rows of the last <paramref name="rows"/>-row verify
        /// that ran as a graph: columns [kept, kept + L - 1) of each layer's history (the window's own pitch), one copy program per
        /// pipeline stage and tensor-parallel rank, on that device. Nothing for 0 kept rows: the live windows are the ones before
        /// the window.
        /// </summary>
        private void CommitShortConvWindows(int kept, int rows)
        {
            if (kept <= 0 || _recurrentPrimary == null || ShortConvPerOpVerifyOpen()) return;
            for (int stage = 0; stage < _mesh.Stages; stage++)
            for (int rank = 0; rank < _mesh.TensorParallel; rank++)
            {
                var states = new List<ShortConvState>();
                for (int layer = 0; layer < Config.NumLayers; layer++)
                    if (_recurrentLayer[layer] && OwnsLayer(layer) && _mesh.StageOf(layer) == stage
                        && ActiveRecurrentState(layer, rank) is ShortConvState { ConvHistory: not null } state && rows <= state.SnapshotCapacity)
                        states.Add(state);
                if (states.Count == 0) continue;
                EnterStage(stage);
                try
                {
                    var key = (StateLeafIdentity(states[0].ConvHistory), kept, rows);
                    if (!_shortConvCommits.TryGetValue(key, out IGraphPlan plan))
                    {
                        var g = new GraphProgramBuilder(record: true);
                        const long f = sizeof(float);
                        foreach (ShortConvState state in states)
                        {
                            int r = state.ConvRows, c = state.Channels;
                            int history = g.Leaf(GraphLeafRole.State, GraphFormat.F32, stackalloc long[] { r + state.SnapshotCapacity, c },
                                StateLeafIdentity(state.ConvHistory), state.ConvHistory.Storage.ByteLength, handle: state.ConvHistory);
                            int window = g.Leaf(GraphLeafRole.State, GraphFormat.F32, stackalloc long[] { r, c },
                                StateLeafIdentity(state.Conv), state.Conv.Storage.ByteLength, handle: state.Conv);
                            int accepted = g.Op(GraphOp.View, stackalloc[] { history },
                                stackalloc long[] { 2, r, c, 1, 1, (r + rows) * f, 0, 0, kept * f });
                            g.Effect(g.Op(GraphOp.Cpy, accepted, window));
                        }
                        plan = CompileForDevice(g.ToProgram(replayable: true, reorder: false), stage, rank);
                        _shortConvCommits[key] = plan;
                    }
                    if (!plan.Run(ReadOnlySpan<int>.Empty, ReadOnlySpan<IntPtr>.Empty, ReadOnlySpan<long>.Empty))
                        throw new InvalidOperationException($"{Config.Architecture} short-conv window commit failed: {plan.LastError}");
                }
                finally
                {
                    EnterStage(0);
                }
            }
        }

        /// <summary>A state tensor's bytes where the executor binds host memory, else a stable identity.</summary>
        private IntPtr StateLeafIdentity(Tensor t) => GraphExecutor.Capabilities.BindsHostPointers
            ? TensorComputePrimitives.GetStoragePointer(t)
            : GraphIdentity.Of(t.Storage, t.StorageOffset);
    }
}
