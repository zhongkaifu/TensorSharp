// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen-Image-2.1's denoiser on the diffusion runtime: one velocity prediction is one GraphProgram (the time MLP and shared
// AdaLN, the text conditioner, the packed single stream of 32 blocks, the output head), emitted through GraphEmitter and run
// by the backend's graph executor: the GGML executor on ggml_cpu, ggml_metal, ggml_cuda and ggml_vulkan, the managed tensor
// executor on cpu. The program is the graph the retired TSGgml_QwenImage21Forward built (ggml_ops_qwen_image21.cpp), op for
// op and with each backend's attention form, so a GGML backend computes the velocity it computed.
//
// The prefix K/V cache: text and reference rows are modulated at t = 0 and never attend the target, so each layer's keys
// and values of those rows are the same at every step. A request's first step ("extract") runs the whole sequence and copies
// them into state leaves (the attention's own K/V type: F16 where the attention reads F16, F32 elsewhere); its later steps
// ("cached") compute only the target rows, attending over [stored prefix; target]. A cache that would take more than half of
// the device's free memory is declined, and that request recomputes the prefix every step ("full").
//
// LoRA plug-ins run unmerged as the retired graph ran them: per projection a shrink and an expand at F32 accumulation added to
// the base product (q, k and v, and gate and up, share one stacked shrink where their factors lie back to back), a DoRA row
// scale on the base, per-head and text norm replacements, and a parallel-decoding bundle's per-step output head in place of
// proj_out, uploaded as an input every step so the steps share one plan.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage21
{
    /// <summary>The Qwen-Image-2.1 transformer as an <see cref="IDenoiser"/>.</summary>
    public sealed partial class QwenImage21Denoiser : IDenoiser
    {
        /// <summary>Which graph a prediction ran.</summary>
        internal enum Mode
        {
            /// <summary>The whole sequence; no prefix cache (none asked for, none possible, or declined).</summary>
            Full,
            /// <summary>The whole sequence, storing the prefix's keys and values.</summary>
            Extract,
            /// <summary>The target rows alone, attending over the stored prefix.</summary>
            Cached,
        }

        /// <summary>How a backend's attention is emitted (the retired graph's per-backend branches).</summary>
        internal enum Style { Cpu, Metal, Cuda, Vulkan, Managed }

        /// <summary>The managed executor's intermediates kept for the next step (they repeat step after step); returned when a
        /// request's denoising ends.</summary>
        internal const long ManagedRetainedBytes = 1L << 30;

        /// <summary>Rows of the MLP the managed executor computes at once: its [rows, 2 ff] product stays bounded (400 MB at 4096
        /// rows whole), as the retired pure-C# transformer chunked it.</summary>
        internal const int ManagedMlpRows = 1024;

        // The retired graph kept two plans a device: the two CFG branches' (cached) graphs.
        private const int RetainedPlans = 2;
        // A request holds a cache per CFG branch; the bound protects a caller that never finishes its requests.
        private const int MaxPrefixCaches = 2;

        private readonly QwenImage21Weights _w;
        private readonly IGraphExecutor _executor;
        private readonly IAllocator _tensorAllocator;   // a tensor executor binds leaves as objects: its F32 vectors live here
        private readonly Style _style;
        private readonly object _gate = new();
        private readonly Dictionary<IntPtr, object> _handles = new();
        private readonly List<RetainedPlan> _plans = new();
        private readonly List<PrefixCache> _caches = new();
        private long _uses;
        private bool _preloaded;
        private bool _disposed;
        // The LoRA set the cached plans bind (null: none).
        private QwenImage21LoraSet _lora;

        /// <summary>The denoiser in <paramref name="path"/> on <paramref name="backend"/>.</summary>
        public QwenImage21Denoiser(string path, BackendType backend)
            : this(new QwenImage21Weights(path ?? throw new ArgumentNullException(nameof(path))), backend, null, null)
        {
        }

        /// <param name="weights">The transformer's weights (the denoiser owns and disposes them).</param>
        /// <param name="executor">The executor the steps run on; null resolves it from <paramref name="backend"/>.</param>
        /// <param name="tensorAllocator">Where a tensor executor's F32 vectors live (null: the managed CPU's).</param>
        /// <param name="tensorParallel">A GPU group the blocks are sharded over (whole heads and MLP columns a GPU, two exact
        /// all-reduces a block), or null for one device.</param>
        internal QwenImage21Denoiser(QwenImage21Weights weights, BackendType backend, IGraphExecutor executor, IAllocator tensorAllocator,
            ITensorParallelGroup tensorParallel = null)
        {
            _w = weights ?? throw new ArgumentNullException(nameof(weights));
            Backend = backend;
            try
            {
                (_executor, _tensorAllocator) = executor != null
                    ? (executor, executor.Capabilities.BindsHostPointers ? null : tensorAllocator ?? new CpuAllocator(BlasEnum.DotNet))
                    : Resolve(backend);
                _style = StyleOf(_executor);
                if (tensorParallel is { Degree: > 1 })
                {
                    _ranks = tensorParallel.Degree;
                    if (_executor is not IRankedGraphExecutor ranked || !ranked.RunsRanks(_ranks))
                        throw new NotSupportedException(
                            $"Qwen-Image-2.1 tensor parallelism over {_ranks} GPUs needs a GGML backend whose GPUs run ranked graphs (ggml_cuda, ggml_vulkan).");
                    _rankBlocks = ShardBlocks(_w, _ranks, _shardOwned);
                    Console.WriteLine($"Qwen-Image-2.1 DiT: {_w.Layers} layers, {_w.Dim} hidden, {_w.Heads} heads sharded over {_ranks} GPUs " +
                        $"({_w.Heads / _ranks} heads and {_w.Ff / _ranks} MLP columns each), graph programs.");
                }
            }
            catch
            {
                foreach (IntPtr p in _shardOwned) QuantizedWeight.FreeBuffer(p);
                _w.Dispose();
                throw;
            }
        }

        // Tensor parallelism: the ranks, each rank's block weights, and the column copies the shards own.
        private readonly int _ranks = 1;
        private readonly QwenImage21Block[][] _rankBlocks;
        private readonly List<IntPtr> _shardOwned = new();

        /// <summary>GPUs the blocks are sharded over (1 when unsharded).</summary>
        internal int TensorParallelRanks => _ranks;

        /// <summary>Megatron sharding of the blocks. Q/K/V, gate and up keep rows (outputs) per rank, contiguous in every GGUF
        /// format, so they are views; to_out and img_mlp.out keep columns (inputs), copied block-aligned per row. A fused [gate;
        /// up] projection becomes separate gate and up slices so each rank's pair stays matched.</summary>
        internal static QwenImage21Block[][] ShardBlocks(QwenImage21Weights w, int ranks, List<IntPtr> owned)
        {
            if (ranks < 2 || w.Heads % ranks != 0 || w.Ff % ranks != 0)
                throw new NotSupportedException($"Qwen-Image-2.1 tensor parallelism needs a GPU count that divides its {w.Heads} heads, not {ranks}.");
            var result = new QwenImage21Block[ranks][];
            for (int r = 0; r < ranks; r++)
            {
                result[r] = new QwenImage21Block[w.Layers];
                for (int i = 0; i < w.Layers; i++)
                {
                    QwenImage21Block b = w.Blocks[i];
                    long ffLocal = w.Ff / ranks, local = w.Dim / ranks;
                    result[r][i] = new QwenImage21Block
                    {
                        Q = b.Q.Rows(r * local, local),
                        K = b.K.Rows(r * local, local),
                        V = b.V.Rows(r * local, local),
                        Out = Columns(b.Out, r * local, local, ranks, owned),
                        Gate = b.Gate.Rows(r * ffLocal, ffLocal),
                        Up = b.Up != null ? b.Up.Rows(r * ffLocal, ffLocal) : b.Gate.Rows(w.Ff + r * ffLocal, ffLocal),
                        Down = Columns(b.Down, r * ffLocal, ffLocal, ranks, owned),
                        NormQ = b.NormQ,
                        NormK = b.NormK,
                    };
                }
            }
            return result;
        }

        private static unsafe QwenImage21Weight Columns(QwenImage21Weight w, long start, long count, int ranks, List<IntPtr> owned)
        {
            var type = (GgmlTensorType)w.Format;
            long block = GgufFile.GetBlockSize(type), typeSize = GgufFile.GetTypeSize(type);
            if (w.Ne0 % (block * ranks) != 0)
                throw new NotSupportedException(
                    $"Qwen-Image-2.1 tensor parallelism cannot split a {w.Ne0}-wide {type} row over {ranks} GPUs on {block}-element blocks.");
            long sourceRow = w.Bytes / w.Ne1, row = count / block * typeSize, offset = start / block * typeSize;
            IntPtr data = QuantizedWeight.AllocateBuffer(checked(row * w.Ne1));
            owned.Add(data);
            byte* source = (byte*)w.Data, destination = (byte*)data;
            for (long o = 0; o < w.Ne1; o++)
                Buffer.MemoryCopy(source + o * sourceRow + offset, destination + o * row, row, row);
            return w with { Data = data, Ne0 = count, Bytes = row * w.Ne1 };
        }

        private static (IGraphExecutor, IAllocator) Resolve(BackendType backend)
        {
            switch (backend)
            {
                case BackendType.GgmlCuda: return (Ggml(GgmlBackendType.Cuda), null);
                case BackendType.GgmlMetal: return (Ggml(GgmlBackendType.Metal), null);
                case BackendType.GgmlVulkan: return (Ggml(GgmlBackendType.Vulkan), null);
                case BackendType.GgmlCpu: return (Ggml(GgmlBackendType.Cpu), null);
                case BackendType.Cpu:
                    var allocator = new RecyclingCpuAllocator(BlasEnum.DotNet, ManagedRetainedBytes);
                    return (new ManagedGraphExecutor(allocator), allocator);
                default:
                    throw new NotSupportedException(
                        $"Qwen-Image-2.1's denoiser runs on ggml_cuda, ggml_metal, ggml_vulkan, ggml_cpu or cpu, not {Architecture.BackendNames.OptionName(backend)}.");
            }
        }

        /// <summary>The GGML executor of <paramref name="type"/>, the process's GGML backend selected for it (a denoiser made
        /// before any model of the backend loads).</summary>
        private static IGraphExecutor Ggml(GgmlBackendType type)
        {
            GgmlBasicOps.EnsureBackendAvailable(type);
            return GgmlGraphExecutor.For(type);
        }

        private static Style StyleOf(IGraphExecutor executor) => executor switch
        {
            GgmlGraphExecutor { Backend: GgmlBackendType.Metal } => Style.Metal,
            GgmlGraphExecutor { Backend: GgmlBackendType.Cuda } => Style.Cuda,
            GgmlGraphExecutor { Backend: GgmlBackendType.Vulkan } => Style.Vulkan,
            GgmlGraphExecutor => Style.Cpu,
            _ => Style.Managed,
        };

        public BackendType Backend { get; }

        internal QwenImage21Weights Weights => _w;

        /// <summary>The executor the steps run on.</summary>
        internal IGraphExecutor Executor => _executor;

        /// <summary>Programs compiled so far (a test seam: the rest replayed a retained plan).</summary>
        internal int Compiles { get; private set; }

        /// <summary>The graph the last prediction ran (a test seam).</summary>
        internal Mode LastMode { get; private set; }

        public DenoiserCaps Caps { get; } = new() { PrefixCacheable = true };

        /// <summary>The transformer's device copies go after the denoise (the decode's full-resolution maps need the room), as
        /// before the runtime; a test that drives the steps itself keeps them.</summary>
        public bool ReleaseAfterPhase { get; init; }

        /// <summary>Drops the GGML scratch with the device copies (the model's release of its compute buffers), or null.</summary>
        internal Action ReleaseCompute { get; init; }

        /// <summary>Test hook: the free memory a prefix cache is measured against, in place of the device's or the machine's
        /// (-1: not known, no cache is declined).</summary>
        internal Func<long> FreeMemory { get; init; }

        public long WeightBytes => _w.Bytes + (AdapterBytes?.Invoke() ?? 0);

        /// <summary>The device bytes of the LoRA factors the next steps bind (the family's current set), for residency.</summary>
        internal Func<long> AdapterBytes { get; set; }

        /// <summary>
        /// A step's compute memory for <paramref name="workload"/>'s packed sequence, from the retired graph's allocations: per row
        /// the residual stream and its normed and modulated copies, q, k and v with their rotated and attention copies, the
        /// attention output, and the MLP's [gate | up] product with its activation (F32), plus the prefix cache's keys and
        /// values at F16 or F32.
        /// </summary>
        public long WorkingBytes(DiffusionWorkload workload)
        {
            long rows = RowsOf(workload, out long prefix);
            long dim = _w.Dim, ff = _w.Ff;
            int kv = _style is Style.Metal or Style.Cuda ? 2 : 4;
            long perRow = 4L * (10 * dim + 3 * ff);
            long cache = (long)_w.Layers * 2 * dim * prefix * kv;
            return 256L * 1024 * 1024 + rows * perRow + cache;
        }

        /// <summary>The packed rows a step over <paramref name="workload"/> holds: the target's and the references' latent cells
        /// and about a token every four characters of the prompt (the references' vision slots are their latent rows).</summary>
        internal static long RowsOf(DiffusionWorkload workload, out long prefix)
        {
            long target = (long)workload.Target.Height * workload.Target.Width;
            prefix = Math.Max(1, (workload.Prompt?.Length ?? 0) / 4) + 32;
            if (workload.References != null)
                foreach (LatentShape r in workload.References) prefix += (long)r.Height * r.Width;
            return target + prefix;
        }

        /// <summary>Make the weights resident now (the GGML executor's bulk uploader); the first step does it otherwise.</summary>
        public int Preload()
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return PreloadLocked();
            }
        }

        private int PreloadLocked()
        {
            if (_preloaded) return 0;
            int copied = 0;
            // A tensor-parallel group's ranks bind their own shards as their graphs reach them.
            if (_executor.Capabilities.BindsHostPointers && _ranks == 1)
            {
                var projections = new List<IGraphWeight>();
                foreach (QwenImage21Weight w in _w.All) if (!w.Vector) projections.Add(w);
                copied = _executor.Preload(projections, "qwen-image-2.1/denoiser");
            }
            _preloaded = true;
            return copied;
        }

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            cancellationToken.ThrowIfCancellationRequested();
            var layout = inputs.Layout.FamilyLayout as QwenImage21Layout
                ?? throw new ArgumentException("the layout was not packed by Qwen-Image-2.1's family", nameof(inputs));
            ConditioningSet text = inputs.Conditioning;
            var references = inputs.References ?? Array.Empty<LatentRef>();
            float[] velocity = Predict(inputs.Latent, text, references, layout, inputs.Step.Sigma, inputs.Step.Index,
                inputs.Recipe.TimestepBf16, LoraOf(adapters));
            return new StepOutput(velocity);
        }

        /// <summary>The packed LoRA set a request's adapters carry (each adapter's payload is the one set the model loaded for
        /// them), or null for none.</summary>
        private QwenImage21LoraSet LoraOf(AdapterSet adapters)
        {
            if (adapters is null or { IsEmpty: true }) return null;
            var set = adapters.Adapters[0].Payload as QwenImage21LoraSet
                ?? throw new ArgumentException($"LoRA {adapters.Adapters[0].Name} was not loaded by Qwen-Image-2.1's LoRA loader.", nameof(adapters));
            foreach (DiffusionAdapter a in adapters.Adapters)
                if (!ReferenceEquals(a.Payload, set))
                    throw new ArgumentException("A request's LoRA adapters are one loaded set: the adapters carry different sets.", nameof(adapters));
            return set;
        }

        /// <summary>
        /// One velocity prediction for the target's token-major latent (<paramref name="target"/>, cells x channels) at
        /// <paramref name="sigma"/>, conditioned on <paramref name="conditioning"/> and <paramref name="references"/> (the
        /// arrays a request keeps for its whole loop: its prefix cache is bound to them). The result is the plan's output
        /// buffer, valid until the next prediction.
        /// </summary>
        internal float[] Predict(float[] target, ConditioningSet conditioning, IReadOnlyList<LatentRef> references,
            QwenImage21Layout layout, float sigma, int step, bool bf16Timestep, QwenImage21LoraSet lora)
        {
            ArgumentNullException.ThrowIfNull(target);
            ArgumentNullException.ThrowIfNull(conditioning);
            ArgumentNullException.ThrowIfNull(layout);
            references ??= Array.Empty<LatentRef>();
            if (!float.IsFinite(sigma) || sigma < 0 || sigma > 1) throw new ArgumentOutOfRangeException(nameof(sigma));
            int channels = _w.Channels;
            QwenImage21Segment targetSegment = layout.Target;
            if (target.LongLength != (long)targetSegment.Length * channels)
                throw new ArgumentException($"The target must hold {targetSegment.Length} x {channels} token-major values.");
            if (conditioning.Width != _w.TextDim || conditioning.Rows <= 0)
                throw new ArgumentException($"Conditioning must hold rows of {_w.TextDim} values.");
            if (layout.HeadDim != _w.HeadDim) throw new ArgumentException("The layout was built for another head width.");
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                BindLora(lora);
                if (lora != null && lora.OutputHeads.Length > 0 && (step < 0 || step >= lora.OutputHeads.Length))
                    throw new ArgumentOutOfRangeException(nameof(step),
                        $"The LoRA bundle has {lora.OutputHeads.Length} step heads; step {step + 1} has none.");
                PreloadLocked();

                PrefixCache cache = layout.Prefix > 0 ? CacheFor(conditioning, references, layout) : null;
                Mode mode = cache == null || cache.Declined ? Mode.Full : cache.Filled ? Mode.Cached : Mode.Extract;
                var s = new StepValues
                {
                    Mode = mode,
                    Layout = layout,
                    Cache = cache,
                    Lora = lora,
                    Step = step,
                    Time = TimeEmbedding(sigma, bf16Timestep),
                    TextRows = conditioning.Rows,
                };
                if (mode == Mode.Cached)
                {
                    s.Images = target;
                    s.ImageRows = targetSegment.Length;
                    (s.Cos, s.Sin) = layout.TargetTables();
                }
                else
                {
                    s.Images = Concat(references, target, out int imageRows);
                    s.ImageRows = imageRows;
                    s.Text = conditioning.Hidden;
                    s.Cos = layout.Cos;
                    s.Sin = layout.Sin;
                }
                if (s.ImageRows != targetSegment.SourceStart + targetSegment.Length && mode != Mode.Cached)
                    throw new ArgumentException("The references' latent rows do not match the layout.");

                float[] output = Run(s);
                if (mode == Mode.Extract)
                {
                    cache.Filled = true;
                    Console.WriteLine($"  [qwen21] prefix KV cache: {layout.Prefix} tokens, {cache.Bytes / (1024.0 * 1024.0):F1} MiB " +
                                      $"{FormatName(cache.Format)}/{FormatName(cache.Format)}; later steps compute only the target image");
                }
                LastMode = mode;
                foreach (float v in output)
                    if (!float.IsFinite(v))
                        throw new InvalidOperationException("Qwen-Image-2.1 DiT produced non-finite latent velocities.");
                return output;
            }
        }

        private static string FormatName(int format) => format == GraphFormat.F16 ? "F16" : "F32";

        /// <summary>The 256-channel sinusoidal embeddings of the target's timestep (row 0) and of t = 0 (row 1), cos first.</summary>
        internal static float[] TimeEmbedding(float sigma, bool bf16)
        {
            if (bf16) sigma = RoundBf16(RoundBf16(sigma * 1000f) / 1000f);
            var time = new float[2 * QwenImage21Weights.TimestepChannels];
            int half = QwenImage21Weights.TimestepChannels / 2;
            for (int i = 0; i < half; ++i)
            {
                float angle = sigma * 1000f * MathF.Exp(-MathF.Log(10000f) * i / half);
                time[i] = MathF.Cos(angle);
                time[i + half] = MathF.Sin(angle);
                time[2 * half + i] = 1f;
            }
            return time;
        }

        /// <summary>Round to the nearest bfloat16 (ties to even), as torch's .to(torch.bfloat16).</summary>
        internal static float RoundBf16(float value)
        {
            int bits = BitConverter.SingleToInt32Bits(value);
            bits += 0x7FFF + ((bits >> 16) & 1);
            return BitConverter.Int32BitsToSingle(bits & unchecked((int)0xFFFF0000));
        }

        private float[] _joined;
        private IReadOnlyList<LatentRef> _joinedReferences;
        private int _joinedPrefixValues;

        /// <summary>The references' rows then the target's (the image tokens of the whole sequence); text to image passes the
        /// target itself. The references' part is copied once per request.</summary>
        private float[] Concat(IReadOnlyList<LatentRef> references, float[] target, out int rows)
        {
            if (references.Count == 0)
            {
                rows = target.Length / _w.Channels;
                return target;
            }
            if (!ReferenceEquals(references, _joinedReferences) || _joined == null || _joined.Length != _joinedPrefixValues + target.Length)
            {
                long values = 0;
                foreach (LatentRef r in references) values += r.Rows.LongLength;
                _joined = new float[checked(values + target.LongLength)];
                long at = 0;
                foreach (LatentRef r in references)
                {
                    Array.Copy(r.Rows, 0, _joined, at, r.Rows.LongLength);
                    at += r.Rows.LongLength;
                }
                _joinedReferences = references;
                _joinedPrefixValues = checked((int)values);
            }
            Array.Copy(target, 0, _joined, _joinedPrefixValues, target.Length);
            rows = _joined.Length / _w.Channels;
            return _joined;
        }

        // ---- plans

        private sealed class RetainedPlan
        {
            public (ulong, ulong) Signature;
            public IGraphPlan Plan;
            public IGraphPlan[] Plans;   // a tensor-parallel step's, a plan a rank
            public long Used;
            public PrefixCache Cache;
        }

        /// <summary>Run the step's program: a retained plan of the same signature replays, else the program is recorded and
        /// compiled (an extract step's once and dropped: it runs once a request).</summary>
        private float[] Run(StepValues s)
        {
            if (_ranks > 1) return RunRanked(s);
            using var trace = new QwenImage21Emitter(_executor, record: false);
            Emit(trace, s, -1);
            if (s.Mode == Mode.Extract)
            {
                using var recorded = new QwenImage21Emitter(_executor, record: true);
                Emit(recorded, s, -1);
                using IGraphPlan once = recorded.Compile(replayable: false);
                Compiles++;
                if (!trace.Run(once))
                    throw new InvalidOperationException("Qwen-Image-2.1 denoiser step failed: " + once.LastError);
                return (float[])once.Outputs[0].Clone();
            }
            RetainedPlan entry = _plans.Find(p => p.Signature == trace.Signature);
            if (entry == null)
            {
                // Make room first, so the old plan's compute memory is free before the new one allocates.
                while (_plans.Count >= RetainedPlans)
                {
                    RetainedPlan oldest = _plans[0];
                    foreach (RetainedPlan p in _plans) if (p.Used < oldest.Used) oldest = p;
                    DropPlan(oldest);
                }
                using var recorded = new QwenImage21Emitter(_executor, record: true);
                Emit(recorded, s, -1);
                entry = new RetainedPlan { Signature = recorded.Signature, Plan = recorded.Compile(replayable: true), Cache = s.Cache };
                Compiles++;
                _plans.Add(entry);
            }
            entry.Used = ++_uses;
            if (!trace.Run(entry.Plan))
                throw new InvalidOperationException("Qwen-Image-2.1 denoiser step failed: " + entry.Plan.LastError);
            return entry.Plan.Outputs[0];
        }

        /// <summary>A tensor-parallel step: each rank's program (its heads and MLP columns, two all-reduces a block), compiled for
        /// its GPU and run together, the partials summed exactly in F32 (a distilled sampler amplifies rounding). Rank 0's output
        /// is the prediction.</summary>
        private float[] RunRanked(StepValues s)
        {
            var executor = (IRankedGraphExecutor)_executor;
            var traces = new QwenImage21Emitter[_ranks];
            QwenImage21Emitter[] recorded = null;
            try
            {
                ulong h1 = 17, h2 = 31;
                for (int r = 0; r < _ranks; r++)
                {
                    traces[r] = new QwenImage21Emitter(_executor, record: false);
                    Emit(traces[r], s, r);
                    (h1, h2) = (h1 * 1_000_003 ^ traces[r].Signature.Item1, h2 * 1_000_033 ^ traces[r].Signature.Item2);
                }
                bool retain = s.Mode != Mode.Extract;
                RetainedPlan entry = retain ? _plans.Find(p => p.Signature == (h1, h2)) : null;
                IGraphPlan[] plans = entry?.Plans;
                if (plans == null)
                {
                    if (retain)
                        while (_plans.Count >= RetainedPlans)
                        {
                            RetainedPlan oldest = _plans[0];
                            foreach (RetainedPlan p in _plans) if (p.Used < oldest.Used) oldest = p;
                            DropPlan(oldest);
                        }
                    recorded = new QwenImage21Emitter[_ranks];
                    plans = new IGraphPlan[_ranks];
                    try
                    {
                        for (int r = 0; r < _ranks; r++)
                        {
                            recorded[r] = new QwenImage21Emitter(_executor, record: true);
                            Emit(recorded[r], s, r);
                            plans[r] = executor.CompileForRank(recorded[r].ToProgram(retain), r);
                        }
                    }
                    catch
                    {
                        foreach (IGraphPlan p in plans) p?.Dispose();
                        throw;
                    }
                    Compiles++;
                    if (retain)
                    {
                        entry = new RetainedPlan { Signature = (h1, h2), Plans = plans, Plan = plans[0], Cache = s.Cache };
                        _plans.Add(entry);
                    }
                }
                if (entry != null) entry.Used = ++_uses;
                try
                {
                    if (!executor.RunRanked(plans, Array.ConvertAll(traces, t => t.RunInputs()), exactSums: true, out string error))
                        throw new InvalidOperationException("Qwen-Image-2.1 tensor-parallel denoiser step failed: " + error);
                    return retain ? plans[0].Outputs[0] : (float[])plans[0].Outputs[0].Clone();
                }
                finally
                {
                    if (!retain) foreach (IGraphPlan p in plans) p.Dispose();
                }
            }
            finally
            {
                foreach (QwenImage21Emitter t in traces) t?.Dispose();
                if (recorded != null) foreach (QwenImage21Emitter t in recorded) t?.Dispose();
            }
        }

        private void DropPlan(RetainedPlan plan)
        {
            _plans.Remove(plan);
            if (plan.Plans != null) foreach (IGraphPlan p in plan.Plans) p.Dispose();
            else plan.Plan.Dispose();
        }

        private void DropPlans(Predicate<RetainedPlan> which)
        {
            foreach (RetainedPlan p in _plans.FindAll(which)) DropPlan(p);
        }

        // ---- the prefix cache

        /// <summary>One request's stored prefix keys and values for one conditioning (a CFG branch), per layer [D, prefix, heads].</summary>
        private sealed class PrefixCache
        {
            public ConditioningSet Conditioning;
            public IReadOnlyList<LatentRef> References;
            public int Prefix, Format;
            public bool Declined, Filled;
            public long Bytes, Used;
            public IntPtr[][] Keys, Values;     // host blocks [rank][layer] (an executor binding host bytes)
            public Tensor[] KeyTensors, ValueTensors;   // a tensor executor's
        }

        private PrefixCache CacheFor(ConditioningSet conditioning, IReadOnlyList<LatentRef> references, QwenImage21Layout layout)
        {
            PrefixCache cache = _caches.Find(c => ReferenceEquals(c.Conditioning, conditioning) && SameReferences(c.References, references));
            if (cache != null && cache.Prefix != layout.Prefix)
            {
                ReleaseCache(cache);
                cache = null;
            }
            if (cache == null)
            {
                while (_caches.Count >= MaxPrefixCaches)
                {
                    PrefixCache oldest = _caches[0];
                    foreach (PrefixCache c in _caches) if (c.Used < oldest.Used) oldest = c;
                    ReleaseCache(oldest);
                }
                cache = CreateCache(conditioning, references, layout.Prefix);
                _caches.Add(cache);
            }
            cache.Used = ++_uses;
            return cache;
        }

        private static bool SameReferences(IReadOnlyList<LatentRef> a, IReadOnlyList<LatentRef> b)
        {
            if (a.Count != b.Count) return false;
            for (int i = 0; i < a.Count; i++) if (!ReferenceEquals(a[i], b[i])) return false;
            return true;
        }

        /// <summary>The attention's K/V type: F16 where Metal's and CUDA's flash attention read F16, F32 elsewhere.</summary>
        private int AttentionFormat => _style is Style.Metal or Style.Cuda ? GraphFormat.F16 : GraphFormat.F32;

        private PrefixCache CreateCache(ConditioningSet conditioning, IReadOnlyList<LatentRef> references, int prefix)
        {
            int format = AttentionFormat;
            long element = format == GraphFormat.F16 ? 2 : 4;
            // Each GPU of a tensor-parallel group stores its own heads.
            long layerBytes = (long)_w.HeadDim * prefix * (_w.Heads / _ranks) * element;
            var cache = new PrefixCache
            {
                Conditioning = conditioning, References = references, Prefix = prefix, Format = format,
                Bytes = 2 * layerBytes * _w.Layers,
            };
            long free = FreeBytes();
            // The prefix is at most half of what is free (when that is known), so the step's compute memory still fits beside it.
            if (free >= 0 && cache.Bytes > free / 2)
            {
                cache.Declined = true;
                Console.Error.WriteLine(
                    $"[qwen21] prefix KV cache declined: {prefix} prefix tokens need {cache.Bytes / (1024.0 * 1024.0):F1} MiB ({FormatName(format)}), " +
                    $"the device reports {free / (1024.0 * 1024.0):F1} MiB free (the cache may use half); this request recomputes the prefix every step.");
                return cache;
            }
            if (_tensorAllocator == null)
            {
                cache.Keys = new IntPtr[_ranks][];
                cache.Values = new IntPtr[_ranks][];
                for (int r = 0; r < _ranks; r++)
                {
                    cache.Keys[r] = new IntPtr[_w.Layers];
                    cache.Values[r] = new IntPtr[_w.Layers];
                    for (int i = 0; i < _w.Layers; i++)
                    {
                        cache.Keys[r][i] = QuantizedWeight.AllocateBuffer(layerBytes);
                        cache.Values[r][i] = QuantizedWeight.AllocateBuffer(layerBytes);
                    }
                }
            }
            else
            {
                cache.KeyTensors = new Tensor[_w.Layers];
                cache.ValueTensors = new Tensor[_w.Layers];
                for (int i = 0; i < _w.Layers; i++)
                {
                    cache.KeyTensors[i] = new Tensor(_tensorAllocator, DType.Float32, _w.Heads, prefix, _w.HeadDim);
                    cache.ValueTensors[i] = new Tensor(_tensorAllocator, DType.Float32, _w.Heads, prefix, _w.HeadDim);
                }
            }
            return cache;
        }

        /// <summary>The memory a prefix cache may take half of: the device's free memory on a GGML backend, the machine's on cpu;
        /// -1 when it is not known (no cache is declined then).</summary>
        private long FreeBytes()
        {
            if (FreeMemory != null) return FreeMemory();
            if (_tensorAllocator != null) return HostFreeBytes();
            if (_ranks == 1) return GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) && total > 0 ? free : -1;
            // A tensor-parallel group's least free GPU: one decision for every rank.
            long least = long.MaxValue;
            int previous = GgmlBasicOps.GetActiveRank();
            try
            {
                for (int r = 0; r < _ranks; r++)
                {
                    GgmlBasicOps.SetActiveRank(r);
                    if (GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) && total > 0) least = Math.Min(least, free);
                }
            }
            finally { GgmlBasicOps.SetActiveRank(previous); }
            return least == long.MaxValue ? -1 : least;
        }

        /// <summary>Physical memory not in use, from the GC's memory-load sample (-1 when there is none). The sample is taken at
        /// a collection and this path allocates natively, so a gen-0 collection refreshes it first; once per request and branch.</summary>
        private static long HostFreeBytes()
        {
            GC.Collect(0, GCCollectionMode.Forced, blocking: true);
            var memory = GC.GetGCMemoryInfo();
            return memory.TotalAvailableMemoryBytes > 0 ? Math.Max(0, memory.TotalAvailableMemoryBytes - memory.MemoryLoadBytes) : -1;
        }

        private void ReleaseCache(PrefixCache cache)
        {
            _caches.Remove(cache);
            // A plan that binds the cache's leaves must not outlive them (its signature names their addresses, which the next
            // cache may reuse).
            DropPlans(p => ReferenceEquals(p.Cache, cache));
            if (cache.Keys != null)
                for (int r = 0; r < cache.Keys.Length; r++)
                    for (int i = 0; i < cache.Keys[r].Length; i++)
                    {
                        InvalidateOnRank(r, cache.Keys[r][i]);
                        InvalidateOnRank(r, cache.Values[r][i]);
                        QuantizedWeight.FreeBuffer(cache.Keys[r][i]);
                        QuantizedWeight.FreeBuffer(cache.Values[r][i]);
                    }
            if (cache.KeyTensors != null)
                for (int i = 0; i < cache.KeyTensors.Length; i++)
                {
                    cache.KeyTensors[i].Dispose();
                    cache.ValueTensors[i].Dispose();
                }
            cache.Keys = cache.Values = null;
            cache.KeyTensors = cache.ValueTensors = null;
        }

        /// <summary>Drop rank <paramref name="rank"/>'s device copy of <paramref name="data"/> (the active rank's, one device).</summary>
        private void InvalidateOnRank(int rank, IntPtr data)
        {
            if (_ranks == 1)
            {
                _executor.Invalidate(data);
                return;
            }
            int previous = GgmlBasicOps.GetActiveRank();
            try
            {
                GgmlBasicOps.SetActiveRank(rank);
                _executor.Invalidate(data);
            }
            finally { GgmlBasicOps.SetActiveRank(previous); }
        }

        /// <summary>Every rank's device copies of <paramref name="data"/>.</summary>
        private void InvalidateEverywhere(IntPtr data)
        {
            for (int r = 0; r < _ranks; r++) InvalidateOnRank(r, data);
        }

        private void ReleaseCaches()
        {
            foreach (PrefixCache c in _caches.ToArray()) ReleaseCache(c);
        }

        /// <summary>The request's prefix caches go (and the plans that bind them); the cached plans of a declined or
        /// prefix-free request stay for the next request of its geometry.</summary>
        public void FinishRequest()
        {
            lock (_gate)
            {
                if (_disposed) return;
                ReleaseCaches();
                _joined = null;
                _joinedReferences = null;
                // The managed executor's kept intermediates go back before the decode needs the memory.
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
            }
        }

        // ---- LoRA

        /// <summary>Bind <paramref name="lora"/> for the steps that follow: a different set than the cached plans' drops them,
        /// the request's caches (computed with the other factors) and the previous set's device copies first.</summary>
        private void BindLora(QwenImage21LoraSet lora)
        {
            if (ReferenceEquals(lora, _lora)) return;
            DropPlans(_ => true);
            ReleaseCaches();
            ReleaseLoraDevice();
            _lora = lora;
        }

        private void ReleaseLoraDevice()
        {
            if (_lora == null) return;
            if (_tensorAllocator == null)
                foreach (IntPtr p in _lora.FactorPointers()) InvalidateEverywhere(p);
            foreach (IntPtr p in _lora.FactorPointers())
                if (_handles.Remove(p, out object h) && h is IDisposable d) d.Dispose();
        }

        /// <summary>Forget the bound LoRA set (its device copies and the plans that bind it): the model calls it before it
        /// disposes a set it replaces.</summary>
        internal void ReleaseLora()
        {
            lock (_gate)
            {
                if (_disposed) return;
                BindLora(null);
            }
        }

        /// <summary>The host addresses every weight leaf binds: the file's, and each rank's shard views and column copies.</summary>
        private IEnumerable<IntPtr> WeightPointers()
        {
            foreach (QwenImage21Weight w in _w.All) yield return w.Data;
            if (_rankBlocks == null) yield break;
            foreach (QwenImage21Block[] rank in _rankBlocks)
                foreach (QwenImage21Block b in rank)
                    foreach (QwenImage21Weight w in new[] { b.Q, b.K, b.V, b.Out, b.Gate, b.Up, b.Down })
                        yield return w.Data;
        }

        // ---- residency

        public void ReleaseWorkspace()
        {
            lock (_gate)
            {
                DropPlans(_ => true);
                ReleaseCaches();
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
            }
        }

        /// <summary>Drops the weights' device copies (and the plans, which bind them); the mapping stays, so the next step
        /// uploads them from memory again.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                DropPlans(_ => true);
                ReleaseCaches();
                foreach (IntPtr p in WeightPointers()) InvalidateEverywhere(p);
                if (_lora != null && _tensorAllocator == null)
                    foreach (IntPtr p in _lora.FactorPointers()) InvalidateEverywhere(p);
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
                _preloaded = false;
            }
            ReleaseCompute?.Invoke();
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                DropPlans(_ => true);
                ReleaseCaches();
                ReleaseLoraDevice();
                _lora = null;
                foreach (IntPtr p in WeightPointers()) InvalidateEverywhere(p);
                foreach (IntPtr p in _shardOwned) QuantizedWeight.FreeBuffer(p);
                _shardOwned.Clear();
                foreach (object h in _handles.Values) (h as IDisposable)?.Dispose();
                _handles.Clear();
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
                _disposed = true;
                _w.Dispose();
            }
        }
    }
}
