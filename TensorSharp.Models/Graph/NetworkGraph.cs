// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;

namespace TensorSharp.Models.Graph
{
    /// <summary>
    /// A non-decoder network (a vision tower, an audio encoder, a depthformer, a codec decoder) run as one graph program per
    /// call on the executor plugin of its weights' allocator (GGML metal/cuda/cpu/vulkan, the managed CPU, MLX, direct CUDA).
    /// <para>
    /// <see cref="Run(Action{NetworkBuilder})"/> takes an emission callback that writes the call with a
    /// <see cref="NetworkBuilder"/> (weights by name, per-run inputs, layer helpers, outputs). The callback runs every call
    /// (it is cheap: no native work) and must emit the same program for the same input shape: a program is compiled once per
    /// distinct signature (ops, parameters, extents, bound weights and states - in practice the input shape) and replayed with
    /// the new inputs while it stays in the cache (<see cref="MaxCachedPlans"/>, least recently used out). An executor
    /// whose plans do not replay (MLX) compiles every call, which there costs a walk over the program.
    /// </para>
    /// <para>
    /// Nothing falls back silently: a program the executor cannot run throws <see cref="NotSupportedException"/> naming the op
    /// at compile, a failed run throws <see cref="InvalidOperationException"/> with the executor's error. Runs are serialized
    /// per instance.
    /// </para>
    /// </summary>
    public sealed class NetworkGraph : IDisposable
    {
        private readonly object _sync = new();
        private readonly Dictionary<(ulong, ulong), IGraphPlan> _plans = new();
        private readonly LinkedList<(ulong, ulong)> _lru = new();
        private bool _disposed;
        private static bool s_hostOutputsNoted;

        /// <param name="weights">The network's weights; their allocator picks the executor.</param>
        /// <param name="name">A name for messages (the network's role: "lfm3 vision tower").</param>
        public NetworkGraph(NetworkWeights weights, string name = null) : this(weights, null, name) { }

        /// <param name="weights">The network's weights.</param>
        /// <param name="executor">The executor to run on in place of the one registered for the weights' allocator (null: that
        /// one): a stage's own device executor, or a reference executor in a test.</param>
        /// <param name="name">A name for messages.</param>
        public NetworkGraph(NetworkWeights weights, IGraphExecutor executor, string name)
        {
            Weights = weights ?? throw new ArgumentNullException(nameof(weights));
            Name = name ?? "network";
            Executor = executor ?? GraphExecutors.For(weights.Allocator)
                ?? throw new NotSupportedException($"{Name}: no graph executor is registered for {weights.Allocator.GetType().Name} " +
                    "(its backend assembly registers one when it loads)");
            AttentionHalfKv = Executor is TensorSharp.GGML.GgmlGraphExecutor;
        }

        public NetworkWeights Weights { get; }

        public string Name { get; }

        /// <summary>The executor plugin of the weights' allocator.</summary>
        public IGraphExecutor Executor { get; }

        /// <summary>Compiled programs kept for replay (distinct input shapes; a replayable GGML program keeps its own compute
        /// memory). 0 compiles every call.</summary>
        public int MaxCachedPlans
        {
            get => _maxCachedPlans;
            set
            {
                if (value < 0) throw new ArgumentOutOfRangeException(nameof(value));
                lock (_sync)
                {
                    _maxCachedPlans = value;
                    Trim();
                }
            }
        }

        private int _maxCachedPlans = 4;

        /// <summary>Let the backend reorder independent work for concurrency (ggml-metal's graph optimizer) in cached
        /// programs.</summary>
        public bool Reorder { get; set; }

        /// <summary>
        /// <see cref="NetworkBuilder.Attention"/> rounds keys and values to F16 before the attention kernel (llama.cpp's mtmd
        /// encoders do). Default: on GGML's executors, whose flash attention over F16 K/V took 46 ms where F32 took 98 (six
        /// 16/8-head layers over 4096 rows, ggml-metal, M5 Pro); off elsewhere (the tensor executors attend F32 rows and cast
        /// to F32 only).
        /// </summary>
        public bool AttentionHalfKv { get; set; }

        /// <summary>
        /// <see cref="NetworkBuilder.Attention"/> as two products around a masked softmax (q k^T, softmax, then the values), all
        /// in F32, where it is otherwise one flash-attention op: llama.cpp's mtmd graphs with flash attention off (its LFM3
        /// speech projector), for a network whose attention is a few rows over a few dozen keys. Default off.
        /// </summary>
        public bool AttentionByProducts { get; set; }

        /// <summary>Programs compiled (cache misses).</summary>
        public int Builds { get; private set; }

        /// <summary>Runs that replayed a cached program.</summary>
        public int Replays { get; private set; }

        /// <summary>Runs.</summary>
        public int Runs { get; private set; }

        /// <summary>Nodes of the last run's program.</summary>
        public int LastNodeCount { get; private set; }

        public int CachedPlans
        {
            get { lock (_sync) return _plans.Count; }
        }

        /// <summary>
        /// Emit the call with <paramref name="emit"/> and run it; returns the outputs' F32 values in the order
        /// <see cref="NetworkBuilder.Output"/> declared them. The arrays of a cached program are rewritten by its next run: copy
        /// what must outlive the next call of this network.
        /// </summary>
        public float[][] Run(Action<NetworkBuilder> emit) => RunCore(emit, keepTensors: false, out _);

        /// <summary>
        /// As <see cref="Run(Action{NetworkBuilder})"/>, the outputs as tensors on the weights' allocator (the caller owns them):
        /// where the executor keeps outputs on its device (the tensor executors), as they lie; elsewhere (GGML, whose outputs are
        /// read back into host arrays) a copy of the host values, which another program then reads as an input.
        /// </summary>
        public Tensor[] RunToTensors(Action<NetworkBuilder> emit)
        {
            float[][] host = RunCore(emit, keepTensors: true, out Tensor[] kept);
            if (kept != null) return kept;
            if (!s_hostOutputsNoted)
            {
                s_hostOutputsNoted = true;
                Console.WriteLine($"  {Name}: {Executor.GetType().Name} returns graph outputs through host arrays; device tensors of them are copies.");
            }
            var tensors = new Tensor[host.Length];
            for (int i = 0; i < host.Length; i++)
            {
                tensors[i] = new Tensor(Weights.Allocator, DType.Float32, host[i].LongLength);
                tensors[i].SetElementsAsFloat(host[i]);
            }
            return tensors;
        }

        private float[][] RunCore(Action<NetworkBuilder> emit, bool keepTensors, out Tensor[] kept)
        {
            ArgumentNullException.ThrowIfNull(emit);
            kept = null;
            lock (_sync)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                bool cache = Executor.Capabilities.Replays && MaxCachedPlans > 0;
                NetworkBuilder step = null;
                IGraphPlan plan = null;
                bool owned = false;
                try
                {
                    if (cache && _plans.Count > 0)
                    {
                        // A signature pass: nothing recorded but the hash and this call's inputs.
                        step = new NetworkBuilder(this, record: false);
                        emit(step);
                        if (_plans.TryGetValue(step.Signature, out plan))
                        {
                            _lru.Remove(step.Signature);
                            _lru.AddFirst(step.Signature);
                            Replays++;
                        }
                        else
                        {
                            step.Dispose();
                            step = null;
                        }
                    }
                    if (plan == null)
                    {
                        step = new NetworkBuilder(this, record: true);
                        emit(step);
                        if (step.OutputCount == 0) throw new InvalidOperationException($"{Name}: the emitted program declares no output");
                        GraphProgram program = step.ToProgram(replayable: cache, reorder: cache && Reorder);
                        plan = Executor.Compile(program);
                        Builds++;
                        if (cache)
                        {
                            _plans[step.Signature] = plan;
                            _lru.AddFirst(step.Signature);
                            Trim();
                        }
                        else
                        {
                            owned = true;
                        }
                    }
                    LastNodeCount = step.NodeCount;
                    var keep = keepTensors ? plan as IGraphTensorOutputs : null;
                    if (plan is IGraphTensorOutputs outputs) outputs.KeepTensors = keep != null;
                    if (!step.Run(plan))
                        throw new InvalidOperationException($"{Name}: the graph run failed on {Executor.GetType().Name}: {plan.LastError}");
                    Runs++;
                    if (keep != null)
                    {
                        kept = new Tensor[plan.Outputs.Length];
                        for (int i = 0; i < kept.Length; i++) kept[i] = keep.TakeOutput(i);
                    }
                    return plan.Outputs;
                }
                finally
                {
                    step?.Dispose();
                    if (owned) plan?.Dispose();
                }
            }
        }

        /// <summary>The least recently run programs past <see cref="MaxCachedPlans"/> out.</summary>
        private void Trim()
        {
            while (_lru.Count > _maxCachedPlans)
            {
                var evict = _lru.Last.Value;
                _lru.RemoveLast();
                if (_plans.Remove(evict, out IGraphPlan old)) old.Dispose();
            }
        }

        /// <summary>Drop every cached program (and the compute memory GGML programs keep).</summary>
        public void ClearCache()
        {
            lock (_sync)
            {
                foreach (IGraphPlan plan in _plans.Values) plan.Dispose();
                _plans.Clear();
                _lru.Clear();
            }
        }

        /// <summary>Disposes the cached programs; the weights stay the caller's.</summary>
        public void Dispose()
        {
            lock (_sync)
            {
                if (_disposed) return;
                _disposed = true;
                foreach (IGraphPlan plan in _plans.Values) plan.Dispose();
                _plans.Clear();
                _lru.Clear();
            }
        }
    }
}
