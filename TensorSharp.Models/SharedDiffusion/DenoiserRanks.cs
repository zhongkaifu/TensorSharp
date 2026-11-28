// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// A diffusion denoiser's blocks sharded over the devices of a tensor-parallel group (<c>--tp N</c>), the same mechanism
    /// for every family: Megatron sharding, each rank holding its heads of every attention (rows of q, k and v; columns of
    /// the output projection) and its slice of every feed-forward's width (rows of gate and up; columns of down), the
    /// modulation, norms, refiners and heads whole on every rank. A family emits each rank's program
    /// (<see cref="Run"/>'s callback gets the rank), with a <see cref="GraphOp.AllReduce"/> after every row-parallel
    /// projection, before the gated residual adds it to the rank's copy of the stream, so every rank holds the whole stream
    /// between blocks and the result is rank 0's.
    /// <para>
    /// On the tensor executors (direct CUDA, the managed one of a simulated group) each rank is its own executor on its
    /// device, joined to the group's collective (<see cref="TensorGraphExecutor.JoinCollective"/>), and the ranks' compiled
    /// plans run in lockstep from one all-reduce to the next (<see cref="TensorGraphExecutor.RunRanked"/>; on direct CUDA the
    /// device-ordered P2P sum). On GGML one ranked executor compiles each rank's program for its device and runs them
    /// together (<see cref="IRankedGraphExecutor.RunRanked"/>), the sums exact in F32 (a few-step distilled sampler amplifies
    /// rounding).
    /// </para>
    /// </summary>
    public sealed class DenoiserRanks : IDisposable
    {
        private readonly IGraphExecutor[] _executors;
        private readonly IAllocator[] _tensorAllocators;
        private readonly IRankedGraphExecutor _ranked;
        private readonly int[] _devices;
        private IGraphPlan[] _plans;
        private (ulong, ulong) _signature;
        private bool _disposed;

        /// <param name="executorFor">A rank device's tensor executor where the family configures its own on one device (direct
        /// products, served wide DiT regions); null takes the backend's registered executor. Ignored on GGML.</param>
        public DenoiserRanks(ITensorParallelGroup group, Func<IAllocator, IGraphExecutor> executorFor = null)
        {
            ArgumentNullException.ThrowIfNull(group);
            int n = group.Degree;
            if (n < 2) throw new ArgumentException("tensor parallelism needs two devices or more", nameof(group));
            if (group.NodeCount > 1)
                throw new NotSupportedException("A diffusion denoiser shards over this process's devices only, not over nodes.");
            _executors = new IGraphExecutor[n];
            _tensorAllocators = new IAllocator[n];
            _devices = new int[n];
            IAllocator first = group.GetAllocator(0);
            IGraphExecutor lead = executorFor?.Invoke(first) ?? GraphExecutors.For(first)
                ?? throw new NotSupportedException($"{first.GetType().Name} has no graph executor.");
            if (lead is TensorGraphExecutor)
            {
                var collective = new GroupCollective(group);
                for (int r = 0; r < n; r++)
                {
                    IAllocator a = group.GetAllocator(r);
                    var x = (r == 0 ? lead : executorFor?.Invoke(a) ?? GraphExecutors.For(a)) as TensorGraphExecutor
                        ?? throw new NotSupportedException($"rank {r}'s device has no tensor executor");
                    x.JoinCollective(collective, r);
                    _executors[r] = x;
                    _tensorAllocators[r] = a;
                    _devices[r] = r;
                }
            }
            else if (lead is IRankedGraphExecutor ranked && ranked.RunsRanks(n))
            {
                _ranked = ranked;
                for (int r = 0; r < n; r++)
                {
                    _executors[r] = lead;
                    _devices[r] = group.GetAllocator(r) is GgmlAllocator g ? g.DeviceId : r;
                }
            }
            else
                throw new NotSupportedException($"{lead.GetType().Name} runs no per-rank programs over {n} devices.");
        }

        /// <summary>The ranks (devices).</summary>
        public int Count => _executors.Length;

        /// <summary>The executor rank <paramref name="rank"/>'s program compiles for.</summary>
        public IGraphExecutor Executor(int rank) => _executors[rank];

        /// <summary>Where rank <paramref name="rank"/>'s tensor executor keeps the F32 vectors it binds as objects (null on GGML,
        /// which binds host pointers).</summary>
        public IAllocator TensorAllocator(int rank) => _tensorAllocators[rank];

        /// <summary>Rank programs compiled (a test seam: the rest replayed).</summary>
        public int Compiles { get; private set; }

        /// <summary>The plans the last step ran, one a rank (a test seam), or null.</summary>
        public IReadOnlyList<IGraphPlan> Plans => _plans;

        /// <summary>Run <paramref name="body"/> with rank <paramref name="rank"/>'s device active (GGML: the context's active
        /// rank; a tensor executor needs nothing).</summary>
        public T OnDevice<T>(int rank, Func<T> body)
        {
            if (_ranked == null) return body();
            int previous = GgmlBasicOps.GetActiveRank();
            GgmlBasicOps.SetActiveRank(_devices[rank]);
            try { return body(); }
            finally { GgmlBasicOps.SetActiveRank(previous); }
        }

        /// <summary><see cref="OnDevice{T}"/> for an action.</summary>
        public void OnDevice(int rank, Action body) => OnDevice(rank, () => { body(); return 0; });

        /// <summary>
        /// One step over every rank: <paramref name="emit"/> traces rank r's program (a hash-only trace with this step's
        /// inputs); a combined signature the cached plans were not compiled for records every rank's program and compiles it
        /// for the rank's device. The plans then run together, each paused at its all-reduces while the group sums the
        /// partials. Returns rank 0's outputs, valid until the next step.
        /// </summary>
        /// <param name="floatProducts">The emitters' F32 products (GraphEmitter's floatProducts), as the family's one-device
        /// step sets it.</param>
        public float[][] Run(Action<GraphEmitter, int> emit, string what, bool floatProducts = false)
        {
            ArgumentNullException.ThrowIfNull(emit);
            ObjectDisposedException.ThrowIf(_disposed, this);
            int n = Count;
            var traces = new GraphEmitter[n];
            try
            {
                ulong a = 0xcbf29ce484222325UL, b = (ulong)n;
                for (int r = 0; r < n; r++)
                {
                    traces[r] = new GraphEmitter(_executors[r], record: false, floatProducts);
                    emit(traces[r], r);
                    (ulong x, ulong y) = traces[r].Signature;
                    a = (a ^ x) * 0x100000001b3UL;
                    b = (b ^ y) * 0x9e3779b97f4a7c15UL + 1;
                }
                if (_plans == null || _signature != (a, b))
                {
                    DropPlans();
                    var plans = new IGraphPlan[n];
                    try
                    {
                        for (int r = 0; r < n; r++)
                        {
                            using var recorded = new GraphEmitter(_executors[r], record: true, floatProducts);
                            emit(recorded, r);
                            GraphProgram program = recorded.ToProgram(replayable: true);
                            plans[r] = _ranked != null ? _ranked.CompileForRank(program, _devices[r]) : _executors[r].Compile(program);
                            Compiles++;
                        }
                    }
                    catch
                    {
                        foreach (IGraphPlan p in plans) p?.Dispose();
                        throw;
                    }
                    _plans = plans;
                    _signature = (a, b);
                }
                GraphRunInputs[] inputs = Array.ConvertAll(traces, t => t.RunInputs());
                string error;
                bool ok = _ranked != null
                    ? _ranked.RunRanked(_plans, inputs, exactSums: true, out error)
                    : TensorGraphExecutor.RunRanked(_plans, inputs, out error);
                if (!ok) throw new InvalidOperationException($"{what} tensor-parallel step over {n} devices failed: {error}");
                return _plans[0].Outputs;
            }
            finally
            {
                foreach (GraphEmitter t in traces) t?.Dispose();
            }
        }

        /// <summary>Drop the ranks' plans (a LoRA set or the weights they bind changed).</summary>
        public void DropPlans()
        {
            if (_plans == null) return;
            foreach (IGraphPlan p in _plans) p?.Dispose();
            _plans = null;
        }

        /// <summary>Drop every rank device's copy of host buffer <paramref name="data"/>.</summary>
        public void InvalidateEverywhere(IntPtr data)
        {
            for (int r = 0; r < Count; r++)
            {
                IGraphExecutor x = _executors[r];
                OnDevice(r, () => x.Invalidate(data));
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DropPlans();
        }

        /// <summary>The group's sum for the tensor executors' ranks: the last rank's partial runs the group's all-reduce
        /// (direct CUDA's device-ordered P2P sum; a simulated group's exact host sum).</summary>
        private sealed class GroupCollective : IGraphCollective
        {
            private readonly ITensorParallelGroup _group;
            private readonly Tensor[] _parts;
            private int _arrived;

            public GroupCollective(ITensorParallelGroup group)
            {
                _group = group;
                _parts = new Tensor[group.Degree];
            }

            public int Degree => _parts.Length;

            public void Contribute(int rank, Tensor partial)
            {
                if (_parts[rank] != null)
                {
                    partial.Dispose();
                    throw new InvalidOperationException($"Tensor-parallel rank {rank} reduced twice before every rank reduced once.");
                }
                _parts[rank] = partial;
                if (++_arrived < _parts.Length) return;
                try
                {
                    _group.AllReduce(_parts);
                }
                finally
                {
                    for (int r = 0; r < _parts.Length; r++)
                    {
                        _parts[r].Dispose();
                        _parts[r] = null;
                    }
                    _arrived = 0;
                }
            }
        }
    }

    /// <summary>
    /// Megatron shards of a diffusion block's projections, in the file's own block format (a projection [Ne0 = in, Ne1 =
    /// out], rows contiguous): a column-parallel projection keeps a rank's output rows, a view where they are one run, a copy
    /// where a stacked weight ([q | k | v], [gate | value]) gives each rank a run of every part; a row-parallel one keeps a
    /// rank's input columns, copied block-aligned row by row. The copies go to <c>owned</c>, freed with the denoiser.
    /// </summary>
    public static class DitShards
    {
        /// <summary>A projection's bytes: its format, extents and host data.</summary>
        public readonly record struct Slice(int Format, long Ne0, long Ne1, IntPtr Data, long Bytes);

        /// <summary>Rank <paramref name="rank"/> of <paramref name="ranks"/>'s output rows of each of <paramref name="parts"/>
        /// equal stacked parts of <paramref name="w"/> (1: a plain projection), in part order.</summary>
        public static unsafe Slice Rows(Slice w, int parts, int rank, int ranks, List<IntPtr> owned)
        {
            if (parts < 1 || w.Ne1 % parts != 0 || (w.Ne1 / parts) % ranks != 0)
                throw new NotSupportedException($"{w.Ne1} output rows in {parts} part(s) do not split over {ranks} devices.");
            long rowBytes = w.Bytes / w.Ne1, part = w.Ne1 / parts, local = part / ranks;
            if (parts == 1)
                return w with { Ne1 = local, Data = w.Data + (nint)(rank * local * rowBytes), Bytes = local * rowBytes };
            IntPtr data = QuantizedWeight.AllocateBuffer(checked(parts * local * rowBytes));
            owned.Add(data);
            for (int p = 0; p < parts; p++)
                Buffer.MemoryCopy((byte*)w.Data + (p * part + rank * local) * rowBytes, (byte*)data + p * local * rowBytes,
                    local * rowBytes, local * rowBytes);
            return w with { Ne1 = parts * local, Data = data, Bytes = parts * local * rowBytes };
        }

        /// <summary>Rank <paramref name="rank"/> of <paramref name="ranks"/>'s input columns of <paramref name="w"/>, copied
        /// block-aligned per row; refused where a rank's share would cut a quant block.</summary>
        public static unsafe Slice Columns(Slice w, int rank, int ranks, List<IntPtr> owned)
        {
            var type = (GgmlTensorType)w.Format;
            long block = GgufFile.GetBlockSize(type), typeSize = GgufFile.GetTypeSize(type);
            if (w.Ne0 % (block * ranks) != 0)
                throw new NotSupportedException($"a {w.Ne0}-wide {type} row does not split over {ranks} devices on {block}-element blocks.");
            long local = w.Ne0 / ranks, sourceRow = w.Bytes / w.Ne1, row = local / block * typeSize, offset = rank * row;
            IntPtr data = QuantizedWeight.AllocateBuffer(checked(row * w.Ne1));
            owned.Add(data);
            byte* source = (byte*)w.Data, destination = (byte*)data;
            for (long o = 0; o < w.Ne1; o++)
                Buffer.MemoryCopy(source + o * sourceRow + offset, destination + o * row, row, row);
            return w with { Ne0 = local, Data = data, Bytes = row * w.Ne1 };
        }

        /// <summary>Rank <paramref name="rank"/>'s slice [start, start + count) of an F32 vector (a column-parallel bias, a
        /// per-head norm's heads), as a new array.</summary>
        public static float[] Vector(ReadOnlySpan<float> values, int parts, int rank, int ranks)
        {
            if (parts < 1 || values.Length % parts != 0 || (values.Length / parts) % ranks != 0)
                throw new NotSupportedException($"{values.Length} values in {parts} part(s) do not split over {ranks} devices.");
            int part = values.Length / parts, local = part / ranks;
            var result = new float[parts * local];
            for (int p = 0; p < parts; p++) values.Slice(p * part + rank * local, local).CopyTo(result.AsSpan(p * local));
            return result;
        }
    }
}
