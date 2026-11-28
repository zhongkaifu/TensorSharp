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

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// A diffusion denoiser's blocks over several devices as whole-block stages (<c>--layer-split N</c>), the same placement
    /// for every family: stage s runs blocks [<see cref="First"/>, <see cref="End"/>) on device s, the first stage also what
    /// comes before the blocks (inputs, text refiners, patch embeddings) and the last what comes after (the final layer and the
    /// head). A family emits each stage as its own program (<see cref="DenoiserSpan"/>), with the residual stream it hands on
    /// as the stage's output and the next stage's input; the per-step values every block reads (the time embedding, rotary
    /// tables, segment tables) are inputs of every stage, so nothing else crosses. Each stage keeps its compiled plan, its
    /// weights stay resident on its own device, and a LoRA set's factors go where their blocks are.
    /// <para>
    /// On GGML one executor serves every device of the context and a stage compiles and runs with its device active
    /// (<see cref="GgmlBasicOps.SetActiveRank"/>); a tensor executor (direct CUDA, the managed one) is one a device, built for
    /// the stage's allocator. The split balances the blocks' weight bytes: the stages run one after another, so a stage only
    /// holds its share of the weights; it does not make one step faster than one device that holds them all.
    /// </para>
    /// </summary>
    public sealed class DenoiserStages : IDisposable
    {
        private readonly IGraphExecutor[] _executors;
        private readonly IAllocator[] _tensorAllocators;
        private readonly int[] _bounds;
        private readonly bool _ggml;
        private readonly IGraphPlan[] _plans;
        private readonly (ulong, ulong)[] _signatures;
        private bool _disposed;

        /// <param name="blockBytes">Each block's weight bytes, in block order (the split balances them).</param>
        /// <param name="devices">Each stage's device allocator: GGML allocators of one context, or one allocator a device of a
        /// tensor executor's backend.</param>
        /// <param name="executorFor">A stage device's executor where the family configures its own on one device (MiniMax-H3's
        /// managed executor serves the wide DiT regions), so a stage runs its blocks as one device runs them; null, or null for a
        /// device, takes the backend's registered executor (<see cref="GraphExecutors.For"/>).</param>
        public DenoiserStages(IReadOnlyList<long> blockBytes, IReadOnlyList<IAllocator> devices,
            Func<IAllocator, IGraphExecutor> executorFor = null)
        {
            ArgumentNullException.ThrowIfNull(blockBytes);
            ArgumentNullException.ThrowIfNull(devices);
            int n = devices.Count;
            if (n < 2) throw new ArgumentException("whole-block stages need two devices or more", nameof(devices));
            if (blockBytes.Count < n)
                throw new NotSupportedException($"--layer-split {n} needs a block a device: the denoiser has {blockBytes.Count}.");
            _ggml = devices[0] is GgmlAllocator;
            _executors = new IGraphExecutor[n];
            _tensorAllocators = new IAllocator[n];
            for (int s = 0; s < n; s++)
            {
                IAllocator a = devices[s] ?? throw new ArgumentException($"stage {s} has no device", nameof(devices));
                if ((a is GgmlAllocator) != _ggml) throw new ArgumentException("the stages' devices are of different backends", nameof(devices));
                _executors[s] = executorFor?.Invoke(a) ?? GraphExecutors.For(a)
                    ?? throw new NotSupportedException($"{a.GetType().Name} has no graph executor.");
                _tensorAllocators[s] = _executors[s].Capabilities.BindsHostPointers ? null : a;
            }
            _bounds = Balance(blockBytes, n);
            _plans = new IGraphPlan[n];
            _signatures = new (ulong, ulong)[n];
            Devices = new int[n];
            for (int s = 0; s < n; s++) Devices[s] = devices[s] is GgmlAllocator g ? g.DeviceId : s;
        }

        /// <summary>The stages (devices).</summary>
        public int Count => _executors.Length;

        /// <summary>Each stage's device (a GGML rank, or the stage's index for a tensor executor).</summary>
        public int[] Devices { get; }

        /// <summary>Stage <paramref name="stage"/>'s first block.</summary>
        public int First(int stage) => _bounds[stage];

        /// <summary>One past stage <paramref name="stage"/>'s last block.</summary>
        public int End(int stage) => _bounds[stage + 1];

        /// <summary>Stage <paramref name="stage"/>'s span of the blocks.</summary>
        public DenoiserSpan Span(int stage) => new(_bounds[stage], _bounds[stage + 1], stage == 0, stage == Count - 1);

        /// <summary>The executor stage <paramref name="stage"/> runs on.</summary>
        public IGraphExecutor Executor(int stage) => _executors[stage];

        /// <summary>Where stage <paramref name="stage"/>'s tensor executor keeps the F32 vectors it binds as objects (null on
        /// GGML, which binds host pointers).</summary>
        public IAllocator TensorAllocator(int stage) => _tensorAllocators[stage];

        /// <summary>Stage programs compiled (a test seam: the rest replayed their stage's plan).</summary>
        public int Compiles { get; private set; }

        /// <summary>The stage that holds block <paramref name="block"/>.</summary>
        public int StageOf(int block)
        {
            for (int s = 0; s < Count; s++)
                if (block < _bounds[s + 1]) return s;
            throw new ArgumentOutOfRangeException(nameof(block));
        }

        /// <summary>Run <paramref name="body"/> with stage <paramref name="stage"/>'s device active (GGML: the context's active
        /// rank; a tensor executor needs nothing).</summary>
        public T OnDevice<T>(int stage, Func<T> body)
        {
            if (!_ggml) return body();
            int previous = GgmlBasicOps.GetActiveRank();
            GgmlBasicOps.SetActiveRank(Devices[stage]);
            try { return body(); }
            finally { GgmlBasicOps.SetActiveRank(previous); }
        }

        /// <summary><see cref="OnDevice{T}"/> for an action.</summary>
        public void OnDevice(int stage, Action body) => OnDevice(stage, () => { body(); return 0; });

        /// <summary>
        /// One step of stage <paramref name="stage"/>: <paramref name="emit"/> traces its program (a hash-only trace with this
        /// step's inputs); a signature the stage's cached plan was not compiled for records the program and compiles it on the
        /// stage's device, after the old plan's memory is released. Returns the plan's first output (the residual stream the
        /// next stage reads, or the last stage's result), valid until the stage runs again.
        /// </summary>
        public float[] Run(int stage, Action<GraphEmitter> emit, string what, bool floatProducts = false)
            => RunAll(stage, emit, what, floatProducts)[0];

        /// <summary><see cref="Run"/> returning every output of the stage's plan, in emission order: a family whose blocks carry
        /// several residual streams (a dual-stream DiT's image and text, or video and audio) hands each on as its own output.
        /// <paramref name="floatProducts"/>: the emitters' F32 products (GraphEmitter's floatProducts; a denoiser that sets it
        /// on one device sets it here).</summary>
        public float[][] RunAll(int stage, Action<GraphEmitter> emit, string what, bool floatProducts = false)
        {
            ArgumentNullException.ThrowIfNull(emit);
            ObjectDisposedException.ThrowIf(_disposed, this);
            return OnDevice(stage, () =>
            {
                IGraphExecutor x = _executors[stage];
                using var trace = new GraphEmitter(x, record: false, floatProducts);
                emit(trace);
                if (_plans[stage] == null || _signatures[stage] != trace.Signature)
                {
                    _plans[stage]?.Dispose();
                    _plans[stage] = null;
                    using var recorded = new GraphEmitter(x, record: true, floatProducts);
                    emit(recorded);
                    _plans[stage] = recorded.Compile(replayable: true);
                    _signatures[stage] = recorded.Signature;
                    Compiles++;
                }
                if (!trace.Run(_plans[stage]))
                    throw new InvalidOperationException($"{what} stage {stage} (blocks {First(stage)}-{End(stage) - 1}) failed: " +
                        _plans[stage].LastError);
                return _plans[stage].Outputs;
            });
        }

        /// <summary>The plan stage <paramref name="stage"/> ran last (a test seam), or null.</summary>
        public IGraphPlan Plan(int stage) => _plans[stage];

        /// <summary>Drop every stage's plan (a LoRA set or the weights they bind changed).</summary>
        public void DropPlans()
        {
            for (int s = 0; s < _plans.Length; s++)
            {
                _plans[s]?.Dispose();
                _plans[s] = null;
            }
        }

        /// <summary>Drop every stage device's copy of host buffer <paramref name="data"/> (a LoRA factor about to be freed).</summary>
        public void InvalidateEverywhere(IntPtr data)
        {
            for (int s = 0; s < Count; s++)
            {
                IGraphExecutor x = _executors[s];
                OnDevice(s, () => x.Invalidate(data));
            }
        }

        /// <summary>
        /// Contiguous spans of the blocks, one a stage, whose largest weight total is as small as a split at the running total's
        /// k/n points makes it: stage k ends at the block whose running total is nearest k/n of the whole, every stage keeping a
        /// block at least.
        /// </summary>
        internal static int[] Balance(IReadOnlyList<long> blockBytes, int stages)
        {
            int blocks = blockBytes.Count;
            var prefix = new double[blocks + 1];
            for (int i = 0; i < blocks; i++) prefix[i + 1] = prefix[i] + Math.Max(0, blockBytes[i]);
            var bounds = new int[stages + 1];
            bounds[stages] = blocks;
            for (int k = 1; k < stages; k++)
            {
                double target = prefix[blocks] * k / stages;
                int lo = bounds[k - 1] + 1, hi = blocks - (stages - k);
                int best = lo;
                for (int b = lo; b <= hi; b++)
                    if (Math.Abs(prefix[b] - target) < Math.Abs(prefix[best] - target)) best = b;
                bounds[k] = best;
            }
            return bounds;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            DropPlans();
        }
    }

    /// <summary>The blocks [<paramref name="First"/>, <paramref name="End"/>) a stage's program runs, and whether it opens the
    /// step (<paramref name="IsFirst"/>: the inputs and what comes before the blocks) and closes it (<paramref name="IsLast"/>:
    /// the final layer and the head). One device's step is <see cref="Whole"/>.</summary>
    public readonly record struct DenoiserSpan(int First, int End, bool IsFirst, bool IsLast)
    {
        /// <summary>Every block of <paramref name="blocks"/>, opening and closing the step.</summary>
        public static DenoiserSpan Whole(int blocks) => new(0, blocks, true, true);
    }
}
