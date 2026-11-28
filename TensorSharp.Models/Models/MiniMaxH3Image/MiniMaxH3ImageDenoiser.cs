// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's denoiser on the diffusion runtime: one step is one GraphProgram (the token refiner, the 50 blocks
// and the final layer with the step's PDD head), emitted through GraphEmitter with the diffusion-transformer regions and
// run by the backend's graph executor. The program is the graph the retired TSGgml_MiniMaxH3DitForward built, node for node,
// so a backend that lowers it as that native graph did computes the same velocity bit for bit (see the emission below).
//
// On direct CUDA (--backend cuda) the same program runs on the CUDA tensor executor as MiniMax-H3 video's denoiser runs its
// (the same 50-block trunk): F32 activations in every product, the diffusion-transformer regions served by the executor's
// own kernels. Over several GPUs it runs as whole-block stages (--layer-split N, DenoiserStages) or Megatron-sharded
// (--tp N, DenoiserRanks): each rank holds its heads of every attention and its slice of every MLP, two all-reduces a block.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>
    /// The MiniMax-H3-Image DiT as an <see cref="IDenoiser"/>. Making one maps the file and reads its small tensors on the
    /// host; its first step makes the weights resident (<see cref="Preload"/>, MiniMax-H3's uploader on the GGML backends)
    /// and compiles the step's program, which every later step of the same geometry replays with new inputs.
    /// <para>
    /// Over a tensor-parallel group (<c>--tp N</c> on <see cref="BackendType.Cuda"/>; a simulated group's host devices in the
    /// tests) the blocks are sharded Megatron-style (<see cref="DenoiserRanks"/>, <see cref="DitShards"/>): each rank holds
    /// rows of every block's stacked [q | k | v] (its heads) and [gate | value] projections and the matching input columns of
    /// the attention output and fc2, so a rank's attention and SwiGLU are its own heads and MLP columns, and each of the two
    /// row-parallel products is summed over the ranks (<see cref="GraphOp.AllReduce"/>) before the gated residual adds it to the
    /// rank's copy of the stream. The token refiner, the modulations, norms, patch projections, final layer and PDD heads are
    /// whole on every rank, so the stream is the same on every rank between blocks and the velocity is rank 0's.
    /// </para>
    /// </summary>
    public sealed class MiniMaxH3ImageDenoiser : IDenoiser
    {
        /// <summary>MiniMax-H3's q8_1 overflow guard around a product of a block-quantized weight (h3_mm's kH3MmScale).</summary>
        internal const float GuardScale = 1024f;

        /// <summary>Why ggml_cuda cannot shard this denoiser: its ranked runner refuses graphs that carry diffusion-transformer
        /// offload nodes (TSGgml_GraphRunRanked, <c>dit_regions &gt; 0</c>), and ggml_cuda serves this step's regions as exactly
        /// those fused nodes. The serving contract's gap and the denoiser's refusal both say it.</summary>
        internal const string GgmlTensorParallelGap =
            "minimax-h3-image runs its tensor-parallel denoiser on --backend cuda only: ggml_cuda's ranked graph runner refuses " +
            "programs with diffusion-transformer offload nodes, which its step is served as. Use --backend cuda --tp N, or " +
            "--layer-split N on ggml_cuda.";

        private readonly MiniMaxH3ImageWeights _w;
        private readonly IGraphExecutor _executor;
        private readonly IAllocator _tensorAllocator;   // tensor executors bind weights as objects: their F32 vectors live here
        // A tensor executor's F32 vectors, one tensor per weight and device (the stage or rank whose program binds it).
        private readonly Dictionary<(MiniMaxH3ImageWeight, int), Tensor> _tensors = new();
        private readonly object _gate = new();
        private readonly Dictionary<(int, int), (float[] Weight, float[] Bias)> _heads = new();
        private float[] _headGrid;
        private (float[] Positions, float[] Cos, float[] Sin, float[] SinSigned) _rope;
        private IGraphPlan _plan;
        private (ulong, ulong) _planSignature;
        private bool _preloaded;
        private bool _disposed;
        // The LoRA set the cached plan binds (null: none), and a tensor executor's handles for its factors (keyed by the bytes
        // and extents they bind: a rank's slot of a factor starts where the whole factor does).
        private MiniMaxH3ImageLoraSet _lora;
        private readonly Dictionary<(IntPtr, long, long), MiniMaxH3ImageWeight> _loraHandles = new();
        // Whole-block stages over several devices (--layer-split N), or null for one device; while a stage's program is
        // emitted, the stage whose tensor executor holds the F32 vectors its weights bind.
        private readonly DenoiserStages _stages;
        private int _emitStage;
        // Tensor parallelism (--tp N), or null: the ranks, each rank's blocks (their sharded projections, the rest shared), the
        // projection copies the shards own, the bound LoRA set's per-rank factors (trunk blocks; the refiner's are whole) and
        // the copies those own; while a rank's program is emitted, its rank.
        private readonly DenoiserRanks _ranks;
        private readonly MiniMaxH3ImageBlock[][] _rankBlocks;
        private readonly List<IntPtr> _shardOwned = new();
        private MiniMaxH3ImageLoraSet.Factors[][] _rankLora;
        private readonly List<IntPtr> _rankLoraOwned = new();
        private int _emitRank;

        /// <summary>The denoiser in <paramref name="path"/> (a GGUF, or the safetensors release in its original names) on
        /// <paramref name="backend"/>: a GGML backend (its graph executor), <see cref="BackendType.Cpu"/> (the managed executor)
        /// or <see cref="BackendType.Cuda"/> (the CUDA tensor executor on <paramref name="allocator"/>'s device).</summary>
        /// <param name="devices">The devices of a whole-block split (<c>--layer-split N</c>, <see cref="DenoiserStages"/>), one a
        /// stage; null or one for one device.</param>
        /// <param name="allocator">The device of a one-device step on <see cref="BackendType.Cuda"/> (a
        /// <see cref="TensorSharp.Cuda.CudaAllocator"/>); unused elsewhere.</param>
        /// <param name="tensorParallel">The group the blocks shard over (<c>--tp N</c>, <see cref="DenoiserRanks"/>): direct CUDA's
        /// <see cref="TensorSharp.Cuda.TensorParallelGroup"/>, or a simulated group's host devices on <see cref="BackendType.Cpu"/>;
        /// null or a group of one for one device.</param>
        public MiniMaxH3ImageDenoiser(string path, BackendType backend, IReadOnlyList<IAllocator> devices = null, IAllocator allocator = null,
            ITensorParallelGroup tensorParallel = null)
            : this(new MiniMaxH3WeightFile(path ?? throw new ArgumentNullException(nameof(path))), ownsFile: true, backend, null, allocator,
                devices, tensorParallel)
        {
        }

        /// <param name="executor">The executor the steps run on; null resolves it from <paramref name="backend"/>.</param>
        /// <param name="tensorAllocator">Where a tensor executor's F32 vectors live (null: the managed CPU's); with no
        /// <paramref name="executor"/>, the direct-CUDA device.</param>
        internal MiniMaxH3ImageDenoiser(MiniMaxH3WeightFile file, bool ownsFile, BackendType backend, IGraphExecutor executor,
            IAllocator tensorAllocator, IReadOnlyList<IAllocator> devices = null, ITensorParallelGroup tensorParallel = null)
        {
            Backend = backend;
            _w = new MiniMaxH3ImageWeights(file, ownsFile, backend);   // disposes an owned file when it throws
            try
            {
                if (tensorParallel is { Degree: > 1 })
                {
                    if (devices is { Count: > 1 })
                        throw new NotSupportedException(
                            "MiniMax-H3-Image shards its denoiser over a tensor-parallel group or splits its blocks over devices, not both.");
                    if (IsGgml(backend)) throw new NotSupportedException(GgmlTensorParallelGap);
                    if (backend is not (BackendType.Cuda or BackendType.Cpu))
                        throw new NotSupportedException(
                            $"MiniMax-H3-Image shards its denoiser on --backend cuda, not {Architecture.BackendNames.OptionName(backend)}.");
                    if (executor != null) throw new ArgumentException("a tensor-parallel denoiser makes one executor a rank", nameof(executor));
                    if (TensorParallelRefusal(file, tensorParallel.Degree) is string refusal) throw new NotSupportedException(refusal);
                    _ranks = new DenoiserRanks(tensorParallel, DeviceExecutor);
                    (_executor, _tensorAllocator) = (_ranks.Executor(0), _ranks.TensorAllocator(0));
                    _rankBlocks = ShardBlocks(_w, _ranks.Count, _shardOwned);
                }
                else if (devices is { Count: > 1 })
                {
                    // Each stage's executor is the one a single device of its backend runs (DeviceExecutor), so a stage runs
                    // its blocks with the products and served regions one device runs them with.
                    _stages = new DenoiserStages(_w.BlockBytes(), devices, DeviceExecutor);
                    (_executor, _tensorAllocator) = (_stages.Executor(0), _stages.TensorAllocator(0));
                }
                else if (executor != null)
                    (_executor, _tensorAllocator) = (executor,
                        executor.Capabilities.BindsHostPointers ? null : tensorAllocator ?? throw new ArgumentNullException(nameof(tensorAllocator)));
                else
                    (_executor, _tensorAllocator) = Resolve(backend, tensorAllocator);
            }
            catch
            {
                FreeShards();
                _w.Dispose();
                throw;
            }
        }

        private static bool IsGgml(BackendType backend) =>
            backend is BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlCpu or BackendType.GgmlVulkan;

        private static (IGraphExecutor, IAllocator) Resolve(BackendType backend, IAllocator allocator) => backend switch
        {
            BackendType.GgmlCuda => (GgmlGraphExecutor.For(GgmlBackendType.Cuda), null),
            BackendType.GgmlMetal => (GgmlGraphExecutor.For(GgmlBackendType.Metal), null),
            BackendType.GgmlCpu => (GgmlGraphExecutor.For(GgmlBackendType.Cpu), null),
            BackendType.Cpu => ManagedExecutor(),
            // The registered CUDA graph executor, as MiniMax-H3 video's denoiser takes it on cuda (GraphExecutors.For).
            BackendType.Cuda => allocator is TensorSharp.Cuda.CudaAllocator cuda
                ? (DeviceExecutor(cuda), cuda)
                : throw new ArgumentException("MiniMax-H3-Image's denoiser on --backend cuda runs on a CUDA device's allocator.", nameof(allocator)),
            _ => throw new NotSupportedException(
                $"MiniMax-H3-Image's denoiser runs on ggml_cuda, ggml_metal, ggml_cpu, cuda or cpu, not {Architecture.BackendNames.OptionName(backend)}."),
        };

        private static (IGraphExecutor, IAllocator) ManagedExecutor()
        {
            var allocator = new CpuAllocator(BlasEnum.DotNet);
            return (new ManagedGraphExecutor(allocator), allocator);
        }

        /// <summary>The executor one device of a tensor executor's backend runs the steps on, the same for one device, each
        /// whole-block stage and each tensor-parallel rank: on the host the managed executor as the one-device step makes it
        /// (its regions as their ops, its own products), on a CUDA device the registered CUDA graph executor (F32 products,
        /// served regions: <see cref="FloatProducts"/>). Null on GGML (the stages take the backend's one executor).</summary>
        private static IGraphExecutor DeviceExecutor(IAllocator device) => device switch
        {
            CpuAllocator cpu => new ManagedGraphExecutor(cpu),
            TensorSharp.Cuda.CudaAllocator cuda => GraphExecutors.For(cuda),
            _ => null,
        };

        /// <summary>
        /// Whether the steps take F32 products (<see cref="GraphEmitter"/>'s floatProducts): on direct CUDA, as MiniMax-H3 video's
        /// denoiser takes them there (the retired direct DiT's bounded F32 products, no q8_1 rounding of the activations, so no
        /// overflow guard), with the executor's own kernels for the wide program's regions; the managed executor keeps its own
        /// products and runs the regions' ops, and the GGML executor takes them as the retired native graph did.
        /// </summary>
        private bool FloatProducts => _tensorAllocator != null && _executor is not ManagedGraphExecutor;

        public BackendType Backend { get; }

        public MiniMaxH3Config Config => _w.Config;

        /// <summary>The executor the steps run on (rank 0's under tensor parallelism, the first stage's under a split).</summary>
        internal IGraphExecutor Executor => _executor;

        /// <summary>Whether the steps run on a tensor executor (weights bound as objects, F32 LoRA factors).</summary>
        internal bool OnTensorExecutor => _tensorAllocator != null;

        /// <summary>The plan the last step ran (a test seam: its served and lowered regions); the last stage's under a split,
        /// rank 0's under tensor parallelism.</summary>
        internal IGraphPlan LastPlan => _ranks != null ? _ranks.Plans?[0] : _stages?.Plan(_stages.Count - 1) ?? _plan;

        /// <summary>Steps that compiled their program (the rest replayed the cached plan); under a split, stage programs; under
        /// tensor parallelism, rank programs.</summary>
        internal int Compiles => _compiles + (_stages?.Compiles ?? 0) + (_ranks?.Compiles ?? 0);

        private int _compiles;

        /// <summary>The whole-block stages, or null for one device (a test seam).</summary>
        internal DenoiserStages Stages => _stages;

        /// <summary>The tensor-parallel ranks, or null (a test seam).</summary>
        internal DenoiserRanks Ranks => _ranks;

        public DenoiserCaps Caps { get; } = new();

        /// <summary>The weights' device bytes: under a split, the largest stage's share; under tensor parallelism, a rank's
        /// (the replicated weights and its shards; each device holds its own).</summary>
        public long WeightBytes => StageWeightBytes() + (AdapterBytes?.Invoke() ?? 0);

        private long StageWeightBytes()
        {
            if (_ranks != null)
            {
                long rank = 0;
                foreach (MiniMaxH3ImageWeight w in RankWeights(0)) rank += w.Bytes;
                return rank;
            }
            if (_stages == null) return _w.Bytes;
            long most = 0;
            for (int s = 0; s < _stages.Count; s++)
            {
                long bytes = 0;
                foreach (MiniMaxH3ImageWeight w in _w.WeightsOf(_stages.Span(s))) bytes += w.Bytes;
                most = Math.Max(most, bytes);
            }
            return most;
        }

        /// <summary>The weight leaves rank <paramref name="rank"/>'s program binds: the replicated ones (the token refiner, the
        /// patch and condition projections, the final layer) and its blocks' (their shards, norms and modulations).</summary>
        private IEnumerable<MiniMaxH3ImageWeight> RankWeights(int rank)
        {
            foreach (MiniMaxH3ImageWeight w in _w.WeightsOf(new DenoiserSpan(0, 0, true, true))) yield return w;
            foreach (MiniMaxH3ImageBlock block in _rankBlocks[rank])
                foreach (MiniMaxH3ImageWeight w in MiniMaxH3ImageWeights.Of(block)) yield return w;
        }

        // ---- tensor parallelism

        /// <summary>
        /// Why the denoiser in <paramref name="file"/> cannot shard over <paramref name="ranks"/> devices, or null: every rank
        /// holds whole heads (the heads divide), an equal slice of the MLP (the FFN width divides), and the row-parallel products'
        /// input columns (attention output: inner / ranks; fc2: FFN / ranks) cut no quant block of the stored format (Q4_K's
        /// 256-element blocks: 7168 / ranks and 14336 / ranks for the release, so --tp 2 and 4, not 8), and its block
        /// projections carry no bias (a row-parallel bias would join every rank's partial).
        /// </summary>
        internal static string TensorParallelRefusal(MiniMaxH3WeightFile file, int ranks)
        {
            ArgumentNullException.ThrowIfNull(file);
            if (ranks < 2) return null;
            MiniMaxH3Config c = MiniMaxH3Config.Detect(file.Tensors);
            if (c.NumAttentionHeads % ranks != 0)
                return $"MiniMax-H3-Image tensor parallelism needs a GPU count that divides its {c.NumAttentionHeads} attention heads " +
                       $"(every GPU holds whole heads); --tp {ranks} does not.";
            if (c.FfnHiddenSize % ranks != 0)
                return $"MiniMax-H3-Image tensor parallelism needs a GPU count that divides its {c.FfnHiddenSize}-wide MLP; --tp {ranks} does not.";
            for (int i = 0; i < c.NumLayers; i++)
            {
                string p = $"{c.Prefix}blocks.{i}.";
                foreach (string bias in new[] { "attn.qkv_proj.bias", "attn.out_proj.bias", "mlp.fc1.bias", "mlp.fc2.bias" })
                    if (file.Tensors.ContainsKey(p + bias))
                        return $"MiniMax-H3-Image tensor parallelism shards bias-free projections; {p + bias} is a bias.";
                foreach (var (name, width) in new[] { ("attn.out_proj.weight", (long)c.AttentionInnerDim), ("mlp.fc2.weight", (long)c.FfnHiddenSize) })
                {
                    if (!file.Tensors.TryGetValue(p + name, out GgufTensorInfo info)) continue;
                    long block = GgufFile.GetBlockSize(info.Type);
                    if (width % (block * ranks) != 0)
                        return $"MiniMax-H3-Image tensor parallelism over {ranks} GPUs gives each {width / ranks} input columns of " +
                               $"{p + name}, which cut its {info.Type} format's {block}-element blocks; use a --tp that divides " +
                               $"{width / block} blocks.";
                }
            }
            return null;
        }

        /// <summary>Each rank's blocks: rows of the stacked [q | k | v] (its heads of each third) and [gate | value] (its MLP
        /// columns of each half) as copies of those runs, the attention output's and fc2's matching input columns as
        /// block-aligned copies (<see cref="DitShards"/>, owned by <paramref name="owned"/>); norms and modulations shared.</summary>
        private static MiniMaxH3ImageBlock[][] ShardBlocks(MiniMaxH3ImageWeights w, int ranks, List<IntPtr> owned)
        {
            static MiniMaxH3ImageLinear Shard(MiniMaxH3ImageLinear l, Func<DitShards.Slice, DitShards.Slice> cut)
            {
                MiniMaxH3ImageWeight x = l.Weight;
                DitShards.Slice s = cut(new DitShards.Slice(x.Format, x.Ne0, x.Ne1, x.Data, x.Bytes));
                return new MiniMaxH3ImageLinear(new MiniMaxH3ImageWeight(s.Format, s.Ne0, s.Ne1, s.Data, s.Bytes), null);
            }
            var result = new MiniMaxH3ImageBlock[ranks][];
            for (int r = 0; r < ranks; r++)
            {
                int rank = r;
                result[r] = new MiniMaxH3ImageBlock[w.Blocks.Length];
                for (int i = 0; i < w.Blocks.Length; i++)
                {
                    MiniMaxH3ImageBlock b = w.Blocks[i];
                    result[r][i] = new MiniMaxH3ImageBlock
                    {
                        Norm1 = b.Norm1, Norm2 = b.Norm2, QNorm = b.QNorm, KNorm = b.KNorm, AdaLn = b.AdaLn,
                        Qkv = Shard(b.Qkv, s => DitShards.Rows(s, 3, rank, ranks, owned)),
                        Out = Shard(b.Out, s => DitShards.Columns(s, rank, ranks, owned)),
                        Fc1 = Shard(b.Fc1, s => DitShards.Rows(s, 2, rank, ranks, owned)),
                        Fc2 = Shard(b.Fc2, s => DitShards.Columns(s, rank, ranks, owned)),
                    };
                }
            }
            return result;
        }

        /// <summary>
        /// The bound LoRA set's trunk factors for each rank, as its projections are sharded: a column-parallel projection's
        /// (qkv, fc1) down whole and the rank's output rows of up (a view where a part is one slot, a copy of the rank's run of
        /// each stacked part where one part spans all of them), its slot rows divided; a row-parallel projection's (attention
        /// output, fc2) the rank's input columns of down (a copy) and up whole, so the update joins the rank's partial before
        /// the all-reduce sums them. The copies go to <see cref="_rankLoraOwned"/>, freed when the set is unbound.
        /// </summary>
        private MiniMaxH3ImageLoraSet.Factors[][] ShardLora(MiniMaxH3ImageLoraSet lora)
        {
            int ranks = _ranks.Count, blocks = _w.Blocks.Length;
            var result = new MiniMaxH3ImageLoraSet.Factors[ranks][];
            for (int r = 0; r < ranks; r++)
            {
                result[r] = new MiniMaxH3ImageLoraSet.Factors[blocks * MiniMaxH3ImageLoraSet.ProjectionCount];
                for (int bi = 0; bi < blocks; bi++)
                    for (int pi = 0; pi < MiniMaxH3ImageLoraSet.ProjectionCount; pi++)
                    {
                        var projection = (MiniMaxH3ImageLoraSet.Projection)pi;
                        MiniMaxH3ImageLoraSet.Factors f = lora.Get(false, bi, projection);
                        if (f == null) continue;
                        bool columnParallel = projection is MiniMaxH3ImageLoraSet.Projection.Qkv or MiniMaxH3ImageLoraSet.Projection.Fc1;
                        int stacked = projection == MiniMaxH3ImageLoraSet.Projection.Qkv ? 3 : 2;
                        var parts = new MiniMaxH3ImageLoraSet.Part[f.Parts.Length];
                        for (int i = 0; i < parts.Length; i++)
                        {
                            MiniMaxH3ImageLoraSet.Part p = f.Parts[i];
                            var down = new DitShards.Slice(p.Format, p.In, p.Rank, p.Down, p.DownBytes);
                            var up = new DitShards.Slice(p.Format, p.Rank, p.Out, p.Up, p.UpBytes);
                            if (columnParallel)
                                up = DitShards.Rows(up, f.SlotRows > 0 ? 1 : stacked, r, ranks, _rankLoraOwned);
                            else
                                down = DitShards.Columns(down, r, ranks, _rankLoraOwned);
                            parts[i] = new MiniMaxH3ImageLoraSet.Part
                            {
                                Slot = p.Slot, Format = p.Format, Rank = p.Rank, In = down.Ne0, Out = up.Ne1,
                                Down = down.Data, DownBytes = down.Bytes, Up = up.Data, UpBytes = up.Bytes,
                            };
                        }
                        result[r][bi * MiniMaxH3ImageLoraSet.ProjectionCount + pi] = new MiniMaxH3ImageLoraSet.Factors
                        {
                            SlotRows = columnParallel && f.SlotRows > 0 ? f.SlotRows / ranks : f.SlotRows,
                            Parts = parts,
                        };
                    }
            }
            return result;
        }

        private void FreeShards()
        {
            foreach (IntPtr p in _shardOwned) QuantizedWeight.FreeBuffer(p);
            _shardOwned.Clear();
        }

        /// <summary>The device bytes of the LoRA factors the next steps bind (the family's current set), for residency.</summary>
        internal Func<long> AdapterBytes { get; set; }

        /// <summary>The LoRA set the cached plan was compiled with (a test seam).</summary>
        internal MiniMaxH3ImageLoraSet BoundLora => _lora;

        /// <summary>The step's activations at <paramref name="workload"/>: the compute buffer's high water
        /// (<see cref="WorkingBytesFor"/>), from the packed sequence's rows.</summary>
        public long WorkingBytes(DiffusionWorkload workload)
        {
            bool quantized = false, bf16Route = false;
            foreach (MiniMaxH3ImageWeight w in _w.All)
            {
                quantized |= w.Quantized;
                bf16Route |= w.Format == GgmlQ4K;
            }
            return WorkingBytesFor(_w.Config, RowsOf(workload), fused: Backend == BackendType.GgmlCuda, quantized, bf16Route,
                edit: workload.References is { Count: > 0 });
        }

        /// <summary>GGML_TYPE_Q4_K: the block type the CUDA step's wide products take to cuBLAS BF16 (ggml_ops_dit_offload.cpp,
        /// h3_q4k_matmul_wants); every other block type runs on ggml's MMQ.</summary>
        private const int GgmlQ4K = 12;

        /// <summary>The packed sequence's rows for <paramref name="workload"/>: the target's and the references' 2x2 patches,
        /// about a token every three characters of the prompt, and an edit's vision blocks in the text (a merged patch of
        /// each picture's vision copy, its markers and its label).</summary>
        internal static long RowsOf(DiffusionWorkload workload)
        {
            long rows = (long)(workload.Target.Height / 2) * (workload.Target.Width / 2);
            if (workload.References != null)
                foreach (LatentShape r in workload.References) rows += (long)(r.Height / 2) * (r.Width / 2);
            if (workload.Pictures != null)
                foreach (PresentedImage p in workload.Pictures)
                    rows += MiniMaxH3ImageConditioner.VisionTokens(p.Image.Width, p.Image.Height) + 2 + 6;
            return rows + Math.Max(1, (workload.Prompt?.Length ?? 0) / 3);
        }

        /// <summary>
        /// A step's compute buffer for <paramref name="rows"/> packed rows, calibrated to the measured device peaks of the
        /// release DiT (hidden 5376, 56 x 128 heads, FFN 14336; stage S4's A/B runs, device memory during the denoise less
        /// the resident weights):
        /// <list type="bullet">
        /// <item>ggml_cuda's fused step (<paramref name="fused"/>): fc1's [gate | value] product beside the residual stream,
        /// the qkv product, the hidden copies and K and V at F16; the SwiGLU is in fc2's narrowing. 315 KB a row: bf16 at
        /// 2048x2048 measured 1.13 GB (272 KB a row), and at 4096x4096 4.9 GB (300 KB a row). Block-quantized projections add
        /// 384 MiB (the largest weight's BF16 copy with its workspace, or MMQ's quantized activations), and where a Q4_K
        /// projection takes the cuBLAS BF16 route (<paramref name="bf16Route"/>) the BF16 copies of the activations too (57 KB
        /// a row). Q4_K_M measured 1.75 GB at 2048x2048 and 5.92 GB at 4096x4096 (2.09 and 6.67 GB estimated); on the RTX
        /// A4500 the files with Q4_K projections measured 1.06-1.08x under the estimate up to 4480x4480, while Q6_K, all of it
        /// on MMQ, measured 2.98, 3.33 and 3.82 GB at 2880, 3072 and 3328 (3.12, 3.47 and 3.97 GB estimated without the copies;
        /// 3.59, 4.01 and 4.60 with them).</item>
        /// <item>The unfused step (ggml_metal, ggml_cpu, the managed executor): the SwiGLU's own output, the q/k norms, RoPE
        /// and padded attention copies and the modulated rows as separate F32 tensors. 745 KB a row: Q8_0 on ggml_metal
        /// measured 2.95 GB at 2048x2048 (710 KB a row) and Q4_K_M 0.23 GB at 512x512 (770 KB a row, with the fixed part).</item>
        /// <item>An edit (<paramref name="edit"/>: references in the sequence, modulated at t = 0 beside the target's sigma)
        /// adds two F32 rows of the hidden width a row (43 KB at the release width), calibrated to stage S5's A40 peaks: bf16
        /// with 4 references (12,412 rows, the DiT alone resident) measured 4,160 MiB, 1.08x the text-to-image formula's
        /// 3,861 MiB; Q4_K_M with 16 references (37,129 rows) peaked 1.1 GB over its plan in the decode. The estimate is
        /// 1.05x the first; the second's plan rises by 1.6 GB. Text to image is unchanged.</item>
        /// </list>
        /// </summary>
        internal static long WorkingBytesFor(MiniMaxH3Config c, long rows, bool fused, bool quantized, bool bf16Route = true,
            bool edit = false)
        {
            ArgumentNullException.ThrowIfNull(c);
            long ffn = c.FfnHiddenSize, inner = c.AttentionInnerDim, hidden = c.HiddenSize;
            long segments = edit ? rows * 2L * 4 * hidden : 0;
            if (fused)
                return (quantized ? 512L : 128L) * 1024 * 1024 + segments
                    + rows * (4L * (2 * ffn + 3 * inner + 4 * hidden) + 4L * inner + (quantized && bf16Route ? 4L * ffn : 0));
            return 64L * 1024 * 1024 + segments + rows * 4L * (4 * ffn + 12 * inner + 8 * hidden);
        }

        /// <summary>Make the weights resident now (MiniMax-H3's uploader on the GGML backends: direct reads from the file
        /// where the device allows, else parallel copies out of the mapping); the first step does it otherwise. Returns how
        /// many weights were copied.</summary>
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
            if (_executor.Capabilities.BindsHostPointers)
            {
                if (_stages == null) copied = _executor.Preload(new List<IGraphWeight>(_w.All), "minimax-h3-image/denoiser");
                else
                    for (int st = 0; st < _stages.Count; st++)
                    {
                        // Each stage's weights onto its own device.
                        IGraphExecutor x = _stages.Executor(st);
                        var weights = new List<IGraphWeight>(_w.WeightsOf(_stages.Span(st)));
                        copied += _stages.OnDevice(st, () => x.Preload(weights, $"minimax-h3-image/denoiser stage {st}"));
                    }
            }
            _preloaded = true;
            return copied;
        }

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            MiniMaxH3ImageLoraSet lora = LoraOf(adapters);
            cancellationToken.ThrowIfCancellationRequested();
            var layout = inputs.Layout.FamilyLayout as MiniMaxH3PackedLayout
                ?? throw new ArgumentException("the layout was not packed by MiniMax-H3-Image's family", nameof(inputs));
            ConditioningSet text = inputs.Conditioning;
            float[] references = null;
            int referenceRows = 0;
            if (inputs.References is { Count: > 0 })
            {
                long values = 0;
                foreach (LatentRef r in inputs.References) values += r.Rows.LongLength;
                references = new float[values];
                long at = 0;
                foreach (LatentRef r in inputs.References)
                {
                    Array.Copy(r.Rows, 0, references, at, r.Rows.LongLength);
                    at += r.Rows.LongLength;
                }
                referenceRows = checked((int)(values / _w.Config.VideoPatchDim));
            }
            int imageRows = checked((int)(inputs.Latent.LongLength / _w.Config.VideoPatchDim));
            float[] velocity = Forward(inputs.Latent, imageRows, text.Hidden, text.Rows, layout, inputs.Recipe.Grid,
                inputs.Step.HeadStart, inputs.Step.HeadStop, references, referenceRows, lora);
            return new StepOutput(velocity);
        }

        /// <summary>The packed LoRA set a request's adapters carry (each adapter's payload is the one set the model loaded for
        /// them), or null for none.</summary>
        private MiniMaxH3ImageLoraSet LoraOf(AdapterSet adapters)
        {
            if (adapters is null or { IsEmpty: true }) return null;
            var set = adapters.Adapters[0].Payload as MiniMaxH3ImageLoraSet
                ?? throw new ArgumentException($"LoRA {adapters.Adapters[0].Name} was not loaded by MiniMax-H3-Image's LoRA loader.", nameof(adapters));
            foreach (DiffusionAdapter a in adapters.Adapters)
                if (!ReferenceEquals(a.Payload, set))
                    throw new ArgumentException("A request's LoRA adapters are one loaded set: the adapters carry different sets.", nameof(adapters));
            if (!set.Fits(_w.Config))
                throw new ArgumentException("The LoRA set was loaded for a MiniMax-H3-Image checkpoint of another shape.", nameof(adapters));
            if (set.ForTensorExecutor != OnTensorExecutor)
                throw new ArgumentException($"The LoRA set is packed for {set.Backend}; this denoiser runs on {Backend}.", nameof(adapters));
            return set;
        }

        /// <summary>Bind <paramref name="lora"/> for the steps that follow: a different set than the cached plan's drops the plan
        /// and the previous set's device copies first, so no step ever runs with factors of a set it was not given (a set
        /// disposed in between may hand its addresses to the next one). Under tensor parallelism the new set's trunk factors are
        /// cut for each rank here, once a set.</summary>
        private void BindLora(MiniMaxH3ImageLoraSet lora)
        {
            if (ReferenceEquals(lora, _lora)) return;
            DropPlans();
            ReleaseLoraDevice();
            _lora = lora;
            if (lora != null && _ranks != null) _rankLora = ShardLora(lora);
        }

        /// <summary>Drop the bound set's device copies (the GGML executor's, keyed by the set's factors; a tensor executor's, keyed
        /// by the factors and rank slices it bound, on every device that bound them), the tensor executor's handles for them,
        /// and the rank slices the set's sharding copied.</summary>
        private void ReleaseLoraDevice()
        {
            if (_lora == null) return;
            if (_tensorAllocator == null)
                foreach (IntPtr p in _lora.FactorPointers()) Invalidate(p);
            foreach ((IntPtr data, _, _) in _loraHandles.Keys) ReleaseEverywhere(data);
            _loraHandles.Clear();
            _rankLora = null;
            foreach (IntPtr p in _rankLoraOwned) QuantizedWeight.FreeBuffer(p);
            _rankLoraOwned.Clear();
        }

        /// <summary>Forget the bound LoRA set (and its device copies and the plan that binds them): the model calls it under its
        /// generation lock before it disposes a set it replaces.</summary>
        internal void ReleaseLora()
        {
            lock (_gate)
            {
                if (_disposed) return;
                BindLora(null);
            }
        }

        /// <summary>
        /// One step, as the legacy MiniMaxH3DiT.ForwardImage took it (deleted in stage S10): [text | references | target] through the trunk,
        /// the target's velocity out of the PDD head for grid intervals [<paramref name="headStart"/>,
        /// <paramref name="headStop"/>), negated so that <c>x += v * (next - sigma)</c> is the Euler step. The result is the
        /// plan's output buffer, valid until the next step.
        /// </summary>
        internal float[] Forward(float[] imageTokens, int imageCount, float[] textHidden, int textCount, MiniMaxH3PackedLayout layout,
            float[] grid, int headStart, int headStop, float[] referenceTokens = null, int referenceCount = 0,
            MiniMaxH3ImageLoraSet lora = null)
        {
            ArgumentNullException.ThrowIfNull(imageTokens);
            ArgumentNullException.ThrowIfNull(layout);
            ArgumentNullException.ThrowIfNull(grid);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                MiniMaxH3Config c = _w.Config;
                if ((long)imageCount * c.VideoPatchDim != imageTokens.LongLength)
                    throw new ArgumentException($"{imageCount} image tokens hold {(long)imageCount * c.VideoPatchDim} values, not {imageTokens.LongLength}.");
                if (referenceCount > 0 && (referenceTokens is null || referenceTokens.LongLength != (long)referenceCount * c.VideoPatchDim))
                    throw new ArgumentException("the reference tokens do not hold referenceCount patches.");
                if (textCount > 0 && (textHidden is null || textHidden.LongLength != (long)textCount * c.TextDim))
                    throw new ArgumentException($"{textCount} text rows of {c.TextDim} need {(long)textCount * c.TextDim} values.");
                if (layout.VideoSpan.End - layout.VideoSpan.Start != imageCount || layout.TokenCount != textCount + referenceCount + imageCount)
                    throw new ArgumentException("the layout does not pack these text, reference and image rows.");
                if (lora != null && !lora.Fits(c))
                    throw new ArgumentException("The LoRA set was loaded for a MiniMax-H3-Image checkpoint of another shape.", nameof(lora));
                BindLora(lora);

                var step = new StepValues
                {
                    Video = referenceCount > 0 ? Concat(referenceTokens, imageTokens) : imageTokens,
                    ImageCount = imageCount,
                    ReferenceCount = referenceCount,
                    Text = textCount > 0 ? textHidden : null,
                    TextCount = textCount,
                    Layout = layout,
                    Segments = Segments(layout),
                    TimeEmbed = TimeEmbedding(layout.Timesteps),
                    Head = Head(grid, headStart, headStop),
                    Lora = lora,
                };
                (step.Cos, step.Sin) = Rope(layout.Positions);
                step.SinSigned = _rope.SinSigned;
                PreloadLocked();
                bool floatProducts = FloatProducts;

                // Tensor parallelism: every rank's program (its heads and MLP columns, two all-reduces a block) traced, compiled
                // once a geometry and LoRA set, and run in lockstep; the velocity is rank 0's.
                if (_ranks != null)
                {
                    try
                    {
                        DenoiserSpan whole = DenoiserSpan.Whole(_w.Blocks.Length);
                        float[][] outputs = _ranks.Run((e, rank) =>
                        {
                            _emitRank = rank;
                            Emit(e, step, whole, null, rank);
                        }, "MiniMax-H3-Image denoiser", floatProducts);
                        return outputs[0];
                    }
                    finally { _emitRank = 0; }
                }

                // Whole-block stages: each stage's program on its device, the residual stream handed from one to the next.
                if (_stages != null)
                {
                    float[] carry = null;
                    try
                    {
                        for (int st = 0; st < _stages.Count; st++)
                        {
                            DenoiserSpan span = _stages.Span(st);
                            float[] input = carry;
                            _emitStage = st;
                            carry = _stages.Run(st, e => Emit(e, step, span, input), "MiniMax-H3-Image denoiser", floatProducts);
                        }
                    }
                    finally { _emitStage = 0; }
                    return carry;
                }

                // The program's signature first (a hash-only trace, whose inputs this step runs with); a geometry the cached
                // plan was not compiled for records the program and compiles it, after the old plan's memory is released.
                using (var trace = new GraphEmitter(_executor, record: false, floatProducts))
                {
                    Emit(trace, step);
                    if (_plan == null || _planSignature != trace.Signature)
                    {
                        _plan?.Dispose();
                        _plan = null;
                        using var recorded = new GraphEmitter(_executor, record: true, floatProducts);
                        Emit(recorded, step);
                        _plan = recorded.Compile(replayable: true);
                        _planSignature = recorded.Signature;
                        _compiles++;
                    }
                    if (!trace.Run(_plan))
                        throw new InvalidOperationException("MiniMax-H3-Image denoiser step failed: " + _plan.LastError);
                }
                return _plan.Outputs[0];
            }
        }

        private static float[] Concat(float[] a, float[] b)
        {
            var r = new float[a.LongLength + b.LongLength];
            Array.Copy(a, r, a.LongLength);
            Array.Copy(b, 0, r, a.LongLength, b.LongLength);
            return r;
        }

        // ---- the step's host values

        /// <summary>The layout's modulation spans as the regions' segments: a row timestep * 3 + modality maps to column
        /// base timestep * 18 + modality * 6 of the block projection viewed as [hidden, 18 * timesteps] (the legacy ForwardImage's).</summary>
        private static GraphSegment[] Segments(MiniMaxH3PackedLayout layout)
        {
            var segments = new GraphSegment[layout.ModulationSpans.Count];
            for (int i = 0; i < segments.Length; i++)
            {
                MiniMaxH3ModulationSpan span = layout.ModulationSpans[i];
                int ts = span.Row / MiniMaxH3Modality.Count, modality = span.Row % MiniMaxH3Modality.Count;
                segments[i] = new GraphSegment(span.Start, span.End,
                    ts * (MiniMaxH3Modality.VectorsPerRow * MiniMaxH3Modality.Count) + modality * MiniMaxH3Modality.VectorsPerRow);
            }
            return segments;
        }

        /// <summary>The AdaLN curve table interpolated at each timestep (MiniMaxH3DiT.TimeEmbedding).</summary>
        internal float[] TimeEmbedding(IReadOnlyList<float> timesteps)
        {
            int dim = _w.Config.TimeEmbedDim, grid = _w.Config.AdaLnCurveGrid;
            float[] table = _w.CurveTable;
            var outp = new float[(long)dim * timesteps.Count];
            for (int t = 0; t < timesteps.Count; t++)
            {
                var (lo, hi, frac) = MiniMaxH3Layout.CurveIndex(timesteps[t], grid);
                long dst = (long)t * dim;
                for (int c = 0; c < dim; c++)
                {
                    float a = table[(long)lo * dim + c];
                    float b = table[(long)hi * dim + c];
                    outp[dst + c] = a + (b - a) * frac;
                }
            }
            return outp;
        }

        /// <summary>The 3-axis partial-RoPE tables of the layout's float positions (MiniMaxH3DiT.BuildRope), kept for the
        /// request: the positions are the same every step.</summary>
        internal (float[] Cos, float[] Sin) Rope(float[] positions)
        {
            ArgumentNullException.ThrowIfNull(positions);
            if (_rope.Positions != null && MemoryMarshal.AsBytes(positions.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(_rope.Positions.AsSpan())))
                return (_rope.Cos, _rope.Sin);
            int nf = _w.Config.RopeInvFreqLength;
            int rot = _w.Config.RopeRotaryDim;
            int half = rot / 2;
            int tokens = positions.Length / 3;
            float[] invFreq = _w.InvFreq;
            var cos = new float[(long)rot * tokens];
            var sin = new float[(long)rot * tokens];
            for (int t = 0; t < tokens; t++)
            {
                long b = (long)t * rot;
                for (int axis = 0; axis < 3; axis++)
                {
                    float pos = positions[t * 3 + axis];
                    for (int k = 0; k < nf; k++)
                    {
                        double a = (double)pos * invFreq[k];
                        int j = axis * nf + k;
                        float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                        cos[b + j] = c; cos[b + j + half] = c;
                        sin[b + j] = s; sin[b + j + half] = s;
                    }
                }
            }
            // [-sin(first half), sin(second half)] a row: the table GraphDitRegions.SignedSine makes from sin (negation is exact),
            // which the F32-products program binds as an input so an executor serving the rotary reads the table's host values.
            var signed = (float[])sin.Clone();
            for (int t = 0; t < tokens; t++)
                for (int j = 0; j < half; j++) signed[(long)t * rot + j] = -signed[(long)t * rot + j];
            _rope = ((float[])positions.Clone(), cos, sin, signed);
            return (cos, sin);
        }

        /// <summary>The PDD head of grid intervals [start, stop) (<see cref="MiniMaxH3ImageSampling.CombineHead"/>), combined
        /// once per interval: [hidden, patch] and [patch] F32, bound as the step's inputs.</summary>
        private (float[] Weight, float[] Bias) Head(float[] grid, int start, int stop)
        {
            if (_headGrid == null || !grid.AsSpan().SequenceEqual(_headGrid))
            {
                _heads.Clear();
                _headGrid = (float[])grid.Clone();
            }
            if (_heads.TryGetValue((start, stop), out var head)) return head;
            head = MiniMaxH3ImageSampling.CombineHead(_w.PddWeight, _w.PddBias, _w.Config.VideoPatchDim, _w.Config.HiddenSize, _headGrid,
                start, stop);
            _heads[(start, stop)] = head;
            return head;
        }

        // ---- emission

        private sealed class StepValues
        {
            public float[] Video;          // [patch, references + image] (references first: they share the patch projection)
            public int ImageCount, ReferenceCount;
            public float[] Text;           // [textDim, textCount] or null
            public int TextCount;
            public MiniMaxH3PackedLayout Layout;
            public GraphSegment[] Segments;
            public float[] TimeEmbed;      // [timeEmbedDim, timesteps]
            public float[] Cos, Sin;       // [rot, tokens]
            public float[] SinSigned;      // [rot, tokens]: [-sin(first half), sin(second half)], the F32-products program's input
            public (float[] Weight, float[] Bias) Head;
            public MiniMaxH3ImageLoraSet Lora;   // null: no adapters
        }

        /// <summary>
        /// The step's program, in the order and with the operands of the retired TSGgml_MiniMaxH3DitForward's graph (ggml_ops_minimax_h3.cpp,
        /// audio_count 0): the q8_1 guard around every block-quantized product but the AdaLN projections' (h3_mm), the token
        /// refiner unmodulated and without RoPE, each block's modulations, attention, gated residuals and SwiGLU as the
        /// diffusion-transformer regions (whose ops are that graph's unfused nodes, and which ggml_cuda serves as its fused
        /// nodes in a WideTokenProducts program), the final layer's per-timestep shift and scale, the head, and the negation.
        /// The per-step values (latent, time embedding, head) are inputs; the segment table, positions and text are fixed for
        /// a request, so a geometry's steps share one plan.
        /// </summary>
        private void Emit(GraphEmitter e, StepValues s) => Emit(e, s, DenoiserSpan.Whole(_w.Blocks.Length), null);

        /// <summary>
        /// The program of <paramref name="span"/>'s blocks: where it opens the step, the inputs, the text refiner and the patch
        /// projections; else the residual stream the previous stage handed on (<paramref name="carry"/>, [hidden, tokens]).
        /// Where it closes the step, the final layer and the velocity; else the residual stream as its output. Every stage takes
        /// the step's time embedding, rotary and segment tables as its own inputs.
        /// <para>Under tensor parallelism, rank <paramref name="rank"/>'s program of the whole step: its blocks' shards (heads / N
        /// of q, k and v, FFN / N of gate and value, the matching input columns of the attention output and fc2) and the LoRA
        /// set's slices of them, an all-reduce after each row-parallel product, the rest whole. With F32 products
        /// (<see cref="FloatProducts"/>, direct CUDA) a block-quantized product takes no q8_1 guard and the signed rotary sine is a
        /// host input; the GGML and managed programs are the ones they always were.</para>
        /// </summary>
        private void Emit(GraphEmitter e, StepValues s, DenoiserSpan span, float[] carry, int rank = 0)
        {
            MiniMaxH3Config c = _w.Config;
            GraphProgramBuilder b = e.Builder;
            int hidden = c.HiddenSize, heads = c.NumAttentionHeads, hd = c.AttentionHeadDim, inner = c.AttentionInnerDim;
            int rot = c.RopeRotaryDim, ntok = s.Layout.TokenCount, nts = s.Layout.Timesteps.Length;
            float eps = c.NormEps;
            float scale = 1f / MathF.Sqrt(hd);
            int ranks = _ranks?.Count ?? 1;
            // A rank's heads and attention width in the blocks (the token refiner is whole on every rank).
            int rankHeads = heads / ranks, rankInner = inner / ranks;
            bool floatProducts = FloatProducts;
            var quantized = new HashSet<int>();
            var leaves = new Dictionary<MiniMaxH3ImageWeight, int>();

            int Leaf(MiniMaxH3ImageWeight w)
            {
                if (w == null) return -1;
                if (leaves.TryGetValue(w, out int node)) return node;
                ReadOnlySpan<long> ne = w.Vector ? stackalloc long[] { w.Ne0 } : stackalloc long[] { w.Ne0, w.Ne1 };
                node = _tensorAllocator == null
                    ? e.HostWeight(w.Format, ne, w.Data, w.Bytes)
                    : e.HostWeight(w.Format, ne, GraphIdentity.Of(w), w.Bytes, Handle(w));
                leaves[w] = node;
                if (w.Quantized) quantized.Add(node);
                return node;
            }
            int Scale(int x, float f) => b.Op(GraphOp.Scale, stackalloc[] { x }, default, stackalloc[] { f });
            // h3_mm: a block-quantized weight's product takes the q8_1 guard (on ggml_cuda's wide routes, which need none, the
            // executor drops the pair where cuBLAS takes the product: GraphProgram.WideTokenProducts). F32 products never round
            // their activations to q8_1, and MiniMax-H3 video's denoiser takes them without it on direct CUDA.
            int Product(GraphProgramBuilder g, int w, int x) => quantized.Contains(w) && !floatProducts
                ? Scale(g.Op(GraphOp.MulMat, w, Scale(x, 1f / GuardScale)), GuardScale)
                : g.Op(GraphOp.MulMat, w, x);
            int Linear(MiniMaxH3ImageLinear l, int x, bool guard = true)
            {
                int w = Leaf(l.Weight);
                int o = guard ? Product(b, w, x) : b.Op(GraphOp.MulMat, w, x);
                return l.Bias != null ? b.Op(GraphOp.Add, o, Leaf(l.Bias)) : o;
            }
            // A block projection with the LoRA set's unmerged updates (a LoraLinear region: w x plus up (down x) on each part's
            // slot, the guard around w x where Linear takes it), then its bias; exactly Linear where the set has none for it.
            MiniMaxH3ImageLoraSet lora = s.Lora;
            int Factor(MiniMaxH3ImageLoraSet.Part p, bool up)
            {
                IntPtr data = up ? p.Up : p.Down;
                long bytes = up ? p.UpBytes : p.DownBytes;
                ReadOnlySpan<long> ne = up ? stackalloc long[] { p.Rank, p.Out } : stackalloc long[] { p.In, p.Rank };
                if (_tensorAllocator == null) return e.HostWeight(p.Format, ne, data, bytes);
                if (!_loraHandles.TryGetValue((data, ne[0], ne[1]), out MiniMaxH3ImageWeight handle))
                    _loraHandles[(data, ne[0], ne[1])] = handle = new MiniMaxH3ImageWeight(p.Format, ne[0], ne[1], data, bytes);
                return e.HostWeight(p.Format, ne, GraphIdentity.Of(handle), bytes, handle);
            }
            // The set's factors of a projection: a trunk block's are the rank's slices under tensor parallelism.
            MiniMaxH3ImageLoraSet.Factors FactorsOf(bool refiner, int block, MiniMaxH3ImageLoraSet.Projection projection) =>
                refiner || _rankLora == null
                    ? lora?.Get(refiner, block, projection)
                    : _rankLora[rank][block * MiniMaxH3ImageLoraSet.ProjectionCount + (int)projection];
            int Adapted(MiniMaxH3ImageLinear l, int x, long rows, bool refiner, int block, MiniMaxH3ImageLoraSet.Projection projection)
            {
                MiniMaxH3ImageLoraSet.Factors f = FactorsOf(refiner, block, projection);
                if (f == null) return Linear(l, x);
                var parts = new GraphDitRegions.LoraPart[f.Parts.Length];
                for (int i = 0; i < parts.Length; i++)
                    parts[i] = new GraphDitRegions.LoraPart(f.Parts[i].Slot, Factor(f.Parts[i], up: false), Factor(f.Parts[i], up: true));
                int y = GraphDitRegions.LoraLinear(b, x, Leaf(l.Weight), parts, l.Weight.Ne1, f.SlotRows, rows, Product);
                return l.Bias != null ? b.Op(GraphOp.Add, y, Leaf(l.Bias)) : y;
            }
            bool HasLora(bool refiner, int block, MiniMaxH3ImageLoraSet.Projection projection) =>
                FactorsOf(refiner, block, projection) != null;
            int Norm(int x, MiniMaxH3ImageWeight w) => b.Op(GraphOp.Mul, b.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), Leaf(w));
            // q, k and v of a stacked [q | k | v] product of n heads (width n * hd each): [hd, n, rows] views, rows a product row
            // apart (a rank's product stacks its own heads of each).
            int Third(int qkv, int i, int rows, int n, int width) => b.Op(GraphOp.View, stackalloc[] { qkv },
                stackalloc long[] { 3, hd, n, rows, 1, hd * 4L, 3L * width * 4, 0, (long)i * width * 4 });
            int Rows(int x, long width, int start, int count) => b.Op(GraphOp.View, stackalloc[] { x },
                stackalloc long[] { 2, width, count, 1, 1, width * 4, 0, 0, (long)start * width * 4 });

            e.MarkWideTokenProducts();
            // A plan binds one LoRA set's factors: another set compiles its own even where it reuses the addresses.
            if (lora != null) e.Mix(lora.Generation);

            // ---- inputs
            int vin = span.IsFirst ? e.InputF32(s.Video, c.VideoPatchDim, s.ReferenceCount + s.ImageCount) : -1;
            int temb = e.InputF32(s.TimeEmbed, c.TimeEmbedDim, nts);
            // A projection of the time embedding viewed as [hidden, columns]: the modulation matrix.
            int Modulations(MiniMaxH3ImageLinear l, long columns) => b.Op(GraphOp.Reshape, stackalloc[] { Linear(l, temb, guard: false) },
                stackalloc long[] { hidden, columns });
            int cosT = e.InputF32(s.Cos, rot, ntok);
            int sinS;
            // The signed sine made on the host (the same values) where an executor serves the rotary from its tables' host
            // values (direct CUDA's TableRopeQkNorm: the device would otherwise read the computed table back every block).
            if (floatProducts) sinS = e.InputF32(s.SinSigned, rot, ntok);
            else
            {
                int sinT = e.InputF32(s.Sin, rot, ntok);
                sinS = GraphDitRegions.SignedSine(b, sinT, rot, ntok);
            }
            int tin = span.IsFirst && s.TextCount > 0 ? e.InputF32(s.Text, c.TextDim, s.TextCount) : -1;
            GraphDitRegions.CheckSegments(s.Segments, ntok, 18L * nts, 5);
            int table = e.InputI32(GraphDitRegions.SegmentTable(s.Segments), 3, s.Segments.Length);

            // ---- text: project, then refine (no modulation, no RoPE; its attention is the native graph's h3_attend)
            int x = -1;
            if (tin >= 0)
            {
                int ctx = Linear(_w.ConditionProj, tin);
                for (int ri = 0; ri < _w.Refiner.Length; ri++)
                {
                    MiniMaxH3ImageBlock r = _w.Refiner[ri];
                    int n1 = Norm(ctx, r.Norm1);
                    int qkv = Adapted(r.Qkv, n1, s.TextCount, true, ri, MiniMaxH3ImageLoraSet.Projection.Qkv);
                    int q = Third(qkv, 0, s.TextCount, heads, inner), k = Third(qkv, 1, s.TextCount, heads, inner);
                    int v = Third(qkv, 2, s.TextCount, heads, inner);
                    q = r.QNorm != null ? Norm(q, r.QNorm) : b.Op(GraphOp.Cont, q);
                    k = r.KNorm != null ? Norm(k, r.KNorm) : b.Op(GraphOp.Cont, k);
                    int merged = GraphDitRegions.Attention(b, q, k, v, hd, heads, s.TextCount, scale, GraphDitRegions.FlashVScale(s.TextCount));
                    ctx = b.Op(GraphOp.Add, ctx, Adapted(r.Out, merged, s.TextCount, true, ri, MiniMaxH3ImageLoraSet.Projection.Out));
                    int n2 = Norm(ctx, r.Norm2);
                    int gv = Adapted(r.Fc1, n2, s.TextCount, true, ri, MiniMaxH3ImageLoraSet.Projection.Fc1);
                    ctx = b.Op(GraphOp.Add, ctx, Adapted(r.Fc2, b.Op(GraphOp.Swiglu, gv), s.TextCount, true, ri,
                        MiniMaxH3ImageLoraSet.Projection.Fc2));
                }
                if (_w.RefinerFinalNorm != null) ctx = Norm(ctx, _w.RefinerFinalNorm);
                x = ctx;
            }

            // ---- patchify the references and the target with one projection, then pack [text | references | target]; a later
            // stage starts from the residual stream the previous one handed on
            if (!span.IsFirst)
                x = e.InputF32(carry ?? throw new ArgumentNullException(nameof(carry)), hidden, ntok);
            else if (s.ReferenceCount > 0)
            {
                int allVideo = Linear(_w.VideoPatch, vin);
                int condTok = Rows(allVideo, hidden, 0, s.ReferenceCount);
                x = x < 0 ? condTok : b.Op(GraphOp.Concat, stackalloc[] { x, condTok }, stackalloc long[] { 1 });
                int videoTok = Rows(allVideo, hidden, s.ReferenceCount, s.ImageCount);
                x = b.Op(GraphOp.Concat, stackalloc[] { x, videoTok }, stackalloc long[] { 1 });
            }
            else
            {
                int allVideo = Linear(_w.VideoPatch, vin);
                x = x < 0 ? allVideo : b.Op(GraphOp.Concat, stackalloc[] { x, allVideo }, stackalloc long[] { 1 });
            }

            // ---- the blocks
            float vScale = GraphDitRegions.FlashVScale(ntok);
            for (int bi = span.First; bi < span.End; bi++)
            {
                MiniMaxH3ImageBlock blk = _rankBlocks?[rank][bi] ?? _w.Blocks[bi];
                // The curve-table variant feeds the interpolated embedding straight in (no SiLU), and takes no guard.
                int mods = Modulations(blk.AdaLn, 18L * nts);
                int h = GraphDitRegions.AdaLnModulate(b, x, Leaf(blk.Norm1), mods, table, s.Segments, hidden, 0, 1, eps);
                int qkv = Adapted(blk.Qkv, h, ntok, false, bi, MiniMaxH3ImageLoraSet.Projection.Qkv);
                int merged = GraphDitRegions.TableRopeQkNormAttention(b, Third(qkv, 0, ntok, rankHeads, rankInner),
                    Third(qkv, 1, ntok, rankHeads, rankInner), Third(qkv, 2, ntok, rankHeads, rankInner),
                    Leaf(blk.QNorm), Leaf(blk.KNorm), cosT, sinS, hd, rankHeads, ntok, rot, eps, scale, vScale);
                int attn = Adapted(blk.Out, merged, ntok, false, bi, MiniMaxH3ImageLoraSet.Projection.Out);
                // A rank's output product is its heads' partial: the ranks' sum is the whole product.
                if (ranks > 1) attn = b.Op(GraphOp.AllReduce, attn);
                x = GraphDitRegions.GatedResidual(b, x, attn, -1, -1, mods, table, s.Segments, hidden, 2);
                h = GraphDitRegions.AdaLnModulate(b, x, Leaf(blk.Norm2), mods, table, s.Segments, hidden, 3, 4, eps);
                // With an update on fc1 or fc2 the MLP is its two projections around the SwiGLU (a SwigluFfn region holds
                // only its weights' products); without, the region the native graph fuses.
                int m = HasLora(false, bi, MiniMaxH3ImageLoraSet.Projection.Fc1) || HasLora(false, bi, MiniMaxH3ImageLoraSet.Projection.Fc2)
                    ? Adapted(blk.Fc2, b.Op(GraphOp.Swiglu, Adapted(blk.Fc1, h, ntok, false, bi, MiniMaxH3ImageLoraSet.Projection.Fc1)),
                        ntok, false, bi, MiniMaxH3ImageLoraSet.Projection.Fc2)
                    : GraphDitRegions.SwigluFfn(b, h, Leaf(blk.Fc1.Weight), Leaf(blk.Fc2.Weight), -1, -1, Product);
                // fc2's product over a rank's MLP columns is its partial, as the attention output's.
                if (ranks > 1) m = b.Op(GraphOp.AllReduce, m);
                x = GraphDitRegions.GatedResidual(b, x, m, -1, -1, mods, table, s.Segments, hidden, 5);
            }

            // A stage that does not close the step hands its residual stream on.
            if (!span.IsLast)
            {
                e.Output(x, (long)hidden * ntok);
                return;
            }

            // ---- final layer: shift and scale by TIMESTEP row (timestep * 2), the head, and the velocity's sign
            int fmods = Modulations(_w.FinalAdaLn, 2L * nts);
            int col = s.Layout.VideoSpan.Row * 2;
            int part = Norm(Rows(x, hidden, s.Layout.VideoSpan.Start, s.ImageCount), _w.FinalNorm);
            int sh = Rows(fmods, hidden, col, 1);
            int sc = Rows(fmods, hidden, col + 1, 1);
            part = b.Op(GraphOp.Add, b.Op(GraphOp.Add, part, b.Op(GraphOp.Mul, part, sc)), sh);
            int headW = e.InputF32(s.Head.Weight, hidden, c.VideoPatchDim);
            int headB = e.InputF32(s.Head.Bias, c.VideoPatchDim);
            int velocity = Scale(b.Op(GraphOp.Add, b.Op(GraphOp.MulMat, headW, part), headB), -1f);
            e.Output(velocity, (long)c.VideoPatchDim * s.ImageCount);
        }

        /// <summary>A tensor executor's object for a weight leaf: an F32 vector as a tensor on the device whose program binds it
        /// (the one device, the stage being emitted, or the rank being emitted: every rank binds the replicated vectors), a
        /// projection as itself.</summary>
        private object Handle(MiniMaxH3ImageWeight w)
        {
            if (!w.Vector) return w;
            int device = _ranks != null ? _emitRank : _emitStage;
            if (!_tensors.TryGetValue((w, device), out Tensor t))
            {
                var values = new float[w.Ne0];
                Marshal.Copy(w.Data, values, 0, values.Length);
                IAllocator allocator = _ranks?.TensorAllocator(_emitRank) ?? _stages?.TensorAllocator(_emitStage) ?? _tensorAllocator;
                t = new Tensor(allocator, DType.Float32, w.Ne0);
                t.SetElementsAsFloat(values);
                _tensors[(w, device)] = t;
            }
            return t;
        }

        // ---- residency

        public void ReleaseWorkspace()
        {
            lock (_gate) DropPlans();
        }

        /// <summary>A request's denoise ended. On direct CUDA the plans go and the step's intermediates the allocators pooled
        /// return to the driver (<see cref="TensorSharp.Cuda.CudaAllocator.TrimPool"/>), as MiniMax-H3 video's denoiser hands
        /// its device over to the VAE: their shapes are of no use to the decode. The weights stay. Elsewhere nothing changes.</summary>
        public void FinishRequest()
        {
            if (_tensorAllocator is not TensorSharp.Cuda.CudaAllocator) return;
            lock (_gate)
            {
                if (_disposed) return;
                DropPlans();
                foreach (IAllocator a in TensorAllocators()) (a as TensorSharp.Cuda.CudaAllocator)?.TrimPool();
            }
        }

        /// <summary>Where the tensor executors keep what the steps allocate: the one device's allocator, each stage's or each
        /// rank's (null on GGML).</summary>
        private IEnumerable<IAllocator> TensorAllocators()
        {
            if (_ranks != null)
                for (int r = 0; r < _ranks.Count; r++) yield return _ranks.TensorAllocator(r);
            else if (_stages != null)
                for (int s = 0; s < _stages.Count; s++) yield return _stages.TensorAllocator(s);
            else yield return _tensorAllocator;
        }

        /// <summary>The cached plan, or every stage's, or every rank's.</summary>
        private void DropPlans()
        {
            _plan?.Dispose();
            _plan = null;
            _stages?.DropPlans();
            _ranks?.DropPlans();
        }

        /// <summary>Drop the GGML executor's device copies of host buffer <paramref name="data"/>: every stage device's under a
        /// split.</summary>
        private void Invalidate(IntPtr data)
        {
            if (_stages != null) _stages.InvalidateEverywhere(data);
            else _executor.Invalidate(data);
        }

        /// <summary>Drop every device copy of the host bytes at <paramref name="data"/>. The GGML graph executor keys it by those
        /// bytes (<see cref="Invalidate"/>). The CUDA tensor executor keeps a projection's copy in the backend's resident-weight
        /// cache under the same address, which <see cref="IGraphExecutor.Invalidate"/> does not reach (MiniMax-H3 video's
        /// denoiser releases its copies the same way): a copy left there outlives the denoiser, its device memory stays taken,
        /// and a file or LoRA set mapped later at that address would compute with the old values. Every device that may hold one
        /// is asked (each stage's, each rank's: releasing a key a device does not hold is a no-op).</summary>
        private void ReleaseEverywhere(IntPtr data)
        {
            if (data == IntPtr.Zero) return;
            if (_tensorAllocator == null)
            {
                Invalidate(data);
                return;
            }
            if (_ranks != null)
                for (int r = 0; r < _ranks.Count; r++) ReleaseCopy(data, _ranks.TensorAllocator(r), _ranks.Executor(r));
            else if (_stages != null)
                for (int s = 0; s < _stages.Count; s++) ReleaseCopy(data, _stages.TensorAllocator(s), _stages.Executor(s));
            else ReleaseCopy(data, _tensorAllocator, _executor);
        }

        private static void ReleaseCopy(IntPtr data, IAllocator allocator, IGraphExecutor executor)
            // The backend's resident weight by the address where it keeps one, else the executor's copy (DeviceResidency).
            => DeviceResidency.ReleaseWeightCopies(allocator, data, data, executor);

        /// <summary>Every weight leaf a device may hold a copy of: the file's and, under tensor parallelism, the ranks' shards.</summary>
        private IEnumerable<MiniMaxH3ImageWeight> BoundWeights()
        {
            foreach (MiniMaxH3ImageWeight w in _w.All) yield return w;
            foreach (MiniMaxH3ImageWeight w in ShardWeights()) yield return w;
        }

        /// <summary>The ranks' sharded projections (none without tensor parallelism).</summary>
        private IEnumerable<MiniMaxH3ImageWeight> ShardWeights()
        {
            if (_rankBlocks == null) yield break;
            foreach (MiniMaxH3ImageBlock[] blocks in _rankBlocks)
                foreach (MiniMaxH3ImageBlock b in blocks)
                    foreach (MiniMaxH3ImageLinear l in new[] { b.Qkv, b.Out, b.Fc1, b.Fc2 })
                        yield return l.Weight;
        }

        /// <summary>The host addresses the device copies of the projections, the shards and the bound LoRA set's factors are
        /// keyed by (tests).</summary>
        internal IReadOnlyList<IntPtr> DeviceCopyKeysForTest()
        {
            lock (_gate)
            {
                var keys = new List<IntPtr>();
                foreach (MiniMaxH3ImageWeight w in BoundWeights())
                    if (!w.Vector) keys.Add(w.Data);
                foreach ((IntPtr data, _, _) in _loraHandles.Keys) keys.Add(data);
                return keys;
            }
        }

        /// <summary>Drops the weights' device copies (and the plan, which binds them); the mapping stays, so the next step
        /// uploads them from memory again.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                DropPlans();
                foreach (MiniMaxH3ImageWeight w in BoundWeights()) ReleaseEverywhere(w.Data);
                // The LoRA factors' copies too (a tensor executor's of the factors and rank slices it bound); the set stays
                // bound and the next step uploads them again.
                if (_lora != null && _tensorAllocator == null)
                    foreach (IntPtr p in _lora.FactorPointers()) Invalidate(p);
                foreach ((IntPtr data, _, _) in _loraHandles.Keys) ReleaseEverywhere(data);
                _preloaded = false;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                DropPlans();
                ReleaseLoraDevice();
                _lora = null;
                // The pinned vectors are this denoiser's own; a borrowed mapping's projections are its owner's to release. The
                // ranks' shards are this denoiser's: their copies go before their bytes are freed.
                foreach (MiniMaxH3ImageWeight w in _w.All)
                    if (w.Vector || _w.OwnsFile) ReleaseEverywhere(w.Data);
                foreach (MiniMaxH3ImageWeight w in ShardWeights()) ReleaseEverywhere(w.Data);
                _stages?.Dispose();
                _ranks?.Dispose();
                foreach (Tensor t in _tensors.Values) t.Dispose();
                _tensors.Clear();
                FreeShards();
                _w.Dispose();
            }
        }
    }
}
