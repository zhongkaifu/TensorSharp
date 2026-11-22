// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;

using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// The vision tower of the <c>muse-glimmer</c> projector type, which Muse-Glimmer and Muse Spark both load: preprocessed
    /// pixels in, one row of the language model's embedding width per merged 2 x 2 patch block out. The multimodal injector
    /// takes it (<see cref="ModelMultimodalInjector.ProcessMuseGlimmerHistory"/>); <see cref="MuseGlimmerVisionTower"/> is the
    /// graph program every executor runs (the legacy per-op tower, its oracle until stage S9, was deleted then).
    /// </summary>
    public interface IMuseGlimmerVisionTower : IDisposable
    {
        /// <summary>Width of the rows (<c>clip.vision.projection_dim</c>): the language model's embedding width.</summary>
        int ProjectionDim { get; }

        /// <summary>The preprocessing: the grid choice, the Lanczos stretch and the normalization.</summary>
        MuseGlimmerImageProcessor ImageProcessor { get; }

        /// <summary>
        /// Encode channel-first [3, <paramref name="height"/>, <paramref name="width"/>] normalized pixels (sides multiples of
        /// patch * merge) into [tokens, <see cref="ProjectionDim"/>] F32 rows in raster order over the merged blocks, on the
        /// tower's allocator (the caller owns them). <paramref name="betweenParts"/> runs where the tower can pause between
        /// pieces of the work (a caller holding its model's GPU lock yields it there).
        /// </summary>
        Tensor Encode(float[] pixels, int width, int height, Action betweenParts = null);
    }

    /// <summary>
    /// The Muse-Glimmer / Muse Spark vision tower (mmproj projector type <c>muse-glimmer</c>) as graph programs on the executor
    /// of the model's allocator (GGML metal/cuda/cpu/vulkan, the managed CPU, MLX, direct CUDA; <see cref="NetworkGraph"/>).
    /// The network is llama.cpp's <c>tools/mtmd/models/muse-glimmer.cpp</c> and HF transformers' <c>MuseGlimmerVisionModel</c>:
    /// <list type="number">
    /// <item>patchify: the 14 x 14 x 3 patches (host im2col, rows in window order) times <c>v.patch_embd.weight</c>, plus the
    /// learned 32 x 32 position grid resampled to the image's grid (read transposed, as trained:
    /// <see cref="MuseGlimmerVisionGrid.ResamplePositionTable"/>), then <c>v.pre_ln</c>;</item>
    /// <item>the patches grouped into windows of side sqrt(position rows) = 32 (row-major windows, row-major patches in each,
    /// edge windows clipped), so a window's patches are consecutive rows;</item>
    /// <item>50 pre-LN layers: LayerNorm (with bias), biased q/k/v, a 2-D NORM rotary of base 10000 (the first half of each
    /// head turned by the 1-indexed column, the second by the row, adjacent pairs in each half), attention in each window
    /// (one attention a window over its rows), or over every patch where (l + 1) % 4 == 0 and in the last layer, the biased
    /// output product, residual; LayerNorm, an erf-GELU MLP, residual; then <c>v.post_ln</c>;</item>
    /// <item>the patches back in raster order and each 2 x 2 block's four concatenated channel-outer (out[c * 4 + s]), then
    /// the adapter mm.0, erf-GELU, mm.1, erf-GELU, mm.2 (no biases).</item>
    /// </list>
    /// A large image runs as several programs over consecutive layer ranges when the caller can yield between them
    /// (<see cref="YieldPatches"/>). Calls are serialized per instance.
    /// </summary>
    public sealed class MuseGlimmerVisionTower : IMuseGlimmerVisionTower
    {
        /// <summary>hparams.muse_glimmer_sparse_factor in llama.cpp: three window layers, then one global layer.</summary>
        private const int SparseFactor = 4;

        private const string PatchKernel = "v.patch_embd.weight";
        private const string PositionTable = "v.position_embd.weight";

        /// <summary>Bytes the executors whose attention holds every score at once (the managed CPU, direct CUDA) may hold for one
        /// piece of query rows' scores: a global layer over a 4096-token image's 16384 patches would take 16 GiB at once.</summary>
        private const long AttentionScoreBudget = 256L << 20;

        /// <summary>Most geometries whose host tables are kept (the resampled position grid of a 4096-token image is 100 MB).</summary>
        private const int MaxCachedGeometries = 4;

        private readonly bool _materializesAttention;
        private readonly bool _packsRotaryViews;
        private readonly float[] _positionTable;
        private readonly Dictionary<long, Geometry> _geometries = new();
        private readonly LinkedList<long> _geometryOrder = new();
        private readonly object _sync = new();
        private bool _disposed;

        /// <summary>
        /// The tower's claim on its device (<see cref="GpuMemoryBudget.Reserve"/>) on the backends whose caches are sized from
        /// the device's free memory (ggml_cuda, ggml_vulkan and ggml_metal: the device the GGML budget reads; direct CUDA: the
        /// allocator's): its weights until the first image uploads them, and the compute of the largest image it accepts
        /// (<see cref="ReservedComputeBytes"/>) while it is loaded. A KV cache that grows before any image has come then leaves
        /// the first image its room, as llama.cpp's mtmd does by allocating both when it loads. On ggml_metal a command buffer
        /// past the working set fails outright and latches the backend's error, where CUDA returns out of memory. Null
        /// elsewhere: MLX's caches are not sized from a device reading, and the CPU backends have none.
        /// </summary>
        private readonly DeviceReservation _reservation;
        private bool _weightsUploaded;

        /// <summary>The programs run on ggml_metal (<see cref="ComputeBytes"/>'s factor).</summary>
        private readonly bool _metal;

        /// <summary>Load the vision tower of the mmproj at <paramref name="mmprojPath"/> onto <paramref name="allocator"/> (the
        /// model's; <paramref name="backend"/> is its backend, which decides the weights' formats).</summary>
        public MuseGlimmerVisionTower(string mmprojPath, IAllocator allocator, BackendType backend)
            : this(OpenGguf(mmprojPath), allocator, backend, ownsFile: true)
        {
        }

        /// <summary>Load the vision tower of an opened mmproj (the caller keeps <paramref name="mmproj"/>).</summary>
        public MuseGlimmerVisionTower(GgufFile mmproj, IAllocator allocator, BackendType backend)
            : this(mmproj, allocator, backend, ownsFile: false)
        {
        }

        /// <summary>As the public constructor, the programs run on <paramref name="executor"/> in place of the one registered
        /// for the allocator (a test's reference executor).</summary>
        internal MuseGlimmerVisionTower(GgufFile mmproj, IAllocator allocator, BackendType backend, IGraphExecutor executor)
            : this(mmproj, allocator, backend, ownsFile: false, executor)
        {
        }

        private MuseGlimmerVisionTower(GgufFile gguf, IAllocator allocator, BackendType backend, bool ownsFile, IGraphExecutor executor = null)
        {
            ArgumentNullException.ThrowIfNull(gguf);
            ArgumentNullException.ThrowIfNull(allocator);
            try
            {
                ReadHyperparameters(gguf);
                _positionTable = ReadPositionTable(gguf);
                Weights = new NetworkWeights(allocator, backend);
                Weights.LoadGguf(gguf, n => (n.StartsWith("v.", StringComparison.Ordinal) && n != PositionTable)
                    || n == "mm.0.weight" || n == "mm.1.weight" || n == "mm.2.weight");
                ValidateWeights();
                // One program a geometry (and a part of a split one): a cached GGML program keeps its own compute memory,
                // which grows with the image, so few are kept. The products keep their activations in F32 on direct CUDA, whose
                // float-activation lane is held to llama.cpp's Metal goldens and to the per-op tower's F32 products: its int8
                // and F16 products left the rows at cosine 0.9975 (worst row 0.94) on a 448 x 168 image, the F32 copies give
                // 1.0000000. The managed executor keeps its integer dot: F32 copies there took 2.2x its time (448 x 224 on 28
                // cores, 9.0 s against 4.1 s) for 1 - cosine 9e-4 to the F32 rows; the tests' F32 reference tower sets
                // Graph.FloatProducts itself. GGML and MLX do not read the flag.
                Graph = new NetworkGraph(Weights, executor, "muse-glimmer vision tower") { MaxCachedPlans = MaxParts };
                Graph.FloatProducts = Graph.Executor is not ManagedGraphExecutor;
                _materializesAttention = Graph.Executor is TensorGraphExecutor && Graph.Executor is not MlxGraphExecutor;
                _metal = backend == BackendType.GgmlMetal;
                ReservedComputeBytes = ComputeBytes((long)MaxTokens * MergeSize * MergeSize);
                _reservation = ReserveDevice(allocator, backend, Weights.QuantizedBytes + Weights.DenseBytes + ReservedComputeBytes);
                // GGML's rotary kernels read a strided half-head view where it lies; the tensor executors' do not (MLX's
                // fast rope takes contiguous rows and turns a strided view element by element on the host: 8 s for a
                // 448 x 448 image where 0.3 s were the device's), so there the views are packed first.
                _packsRotaryViews = Graph.Executor is TensorGraphExecutor;
            }
            catch
            {
                _reservation?.Dispose();
                Graph?.Dispose();
                Weights?.Dispose();
                throw;
            }
            finally
            {
                if (ownsFile) gguf.Dispose();
            }
            ImageProcessor = new MuseGlimmerImageProcessor(PatchSize, MergeSize, MaxTokens, _imageMean, _imageStd);
        }

        private static GgufFile OpenGguf(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            return new GgufFile(path);
        }

        /// <summary>The tower's weights (v.* without the position grid, and the adapter mm.0-2).</summary>
        public NetworkWeights Weights { get; }

        /// <summary>The graph runner.</summary>
        public NetworkGraph Graph { get; }

        public MuseGlimmerImageProcessor ImageProcessor { get; }

        /// <summary>Tower width (<c>clip.vision.embedding_length</c>).</summary>
        public int EmbeddingLength { get; private set; }

        public int BlockCount { get; private set; }
        public int HeadCount { get; private set; }
        public int HeadDim { get; private set; }
        public int FeedForwardLength { get; private set; }
        public int PatchSize { get; private set; }
        public int MergeSize { get; private set; }

        /// <summary>LayerNorm epsilon (<c>clip.vision.attention.layer_norm_epsilon</c>).</summary>
        public float NormEps { get; private set; }

        /// <summary>The 2-D rotary's base: llama.cpp hardcodes 10000 for this projector (the mmproj carries none).</summary>
        public float RopeTheta { get; private set; }

        public int ProjectionDim { get; private set; }

        /// <summary>The position grid's side, sqrt(position rows): 32, also the window side in patches.</summary>
        public int GridSide { get; private set; }

        /// <summary>Most merged tokens an image takes (mtmd's set_limit_image_tokens(1, 4096)).</summary>
        public const int MaxTokens = 4096;

        /// <summary>
        /// Query rows per attention piece on the executors whose attention holds every score at once (the managed CPU and
        /// direct CUDA): 0 sizes the pieces to keep each piece's scores under 256 MiB; a positive value forces that size
        /// (tests). GGML (flash attention) and MLX (fused attention) run one piece.
        /// </summary>
        public int AttentionChunkRows { get; set; }

        /// <summary>
        /// Patches of work per program when the caller can yield between programs (an <c>Encode</c> callback: the model's
        /// GPU-lock yield): a larger image runs the tower as ceil(patches / this) programs over consecutive layer ranges (at
        /// most <see cref="MaxParts"/>), the activations passing between them in F32, so concurrent requests step between the
        /// parts of a large image. 0 never splits; without a callback an image is one program. Default 1024: a 448 x 448
        /// image (1024 patches, about 0.3 s on ggml_metal) runs as one program, a 1036 x 1036 one (5476) as five; measured on
        /// ggml_metal (M5 Pro), the five took 2076 ms where one took 2090, the rows bit-identical.
        /// </summary>
        public int YieldPatches { get; set; } = 1024;

        /// <summary>Most programs a split image runs as (<see cref="YieldPatches"/>).</summary>
        public const int MaxParts = 5;

        /// <summary>The last encode's wall time, its programs and its rows (for the probes).</summary>
        internal double LastEncodeMs { get; private set; }

        internal int LastParts { get; private set; }

        private float[] _imageMean, _imageStd;

        // ================================================================ loading

        private void ReadHyperparameters(GgufFile gguf)
        {
            string projector = gguf.GetString("clip.vision.projector_type") ?? gguf.GetString("clip.projector_type");
            if (!string.Equals(projector, "muse-glimmer", StringComparison.Ordinal))
                throw new NotSupportedException($"muse-glimmer vision: the mmproj's vision projector is '{projector ?? "(none)"}', not 'muse-glimmer'");
            PatchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            MergeSize = (int)gguf.GetUint32("clip.vision.spatial_merge_size", 2);
            EmbeddingLength = (int)gguf.GetUint32("clip.vision.embedding_length", 1536);
            FeedForwardLength = (int)gguf.GetUint32("clip.vision.feed_forward_length", 8960);
            HeadCount = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            BlockCount = (int)gguf.GetUint32("clip.vision.block_count", 50);
            NormEps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
            ProjectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 6656);
            // The mmproj carries no rope base for this projector; llama.cpp hardcodes hparams.rope_theta = 10000.
            RopeTheta = gguf.GetFloat32("clip.vision.rope.freq_base", 10000f);
            if (PatchSize < 1 || MergeSize != 2 || EmbeddingLength < 1 || HeadCount < 1 || EmbeddingLength % HeadCount != 0
                || BlockCount < 1 || ProjectionDim < 1)
                throw new NotSupportedException($"muse-glimmer vision: patch {PatchSize}, merge {MergeSize}, width {EmbeddingLength}, " +
                    $"{HeadCount} heads, {BlockCount} layers, projection {ProjectionDim} (this port runs merge 2, as llama.cpp)");
            HeadDim = EmbeddingLength / HeadCount;
            if (HeadDim % 4 != 0)
                throw new NotSupportedException($"muse-glimmer vision: a {HeadDim}-wide head (the 2-D rotary turns each half in pairs)");
            _imageMean = gguf.GetFloatArray("clip.vision.image_mean");
            _imageStd = gguf.GetFloatArray("clip.vision.image_std");
        }

        private float[] ReadPositionTable(GgufFile gguf)
        {
            if (!gguf.Tensors.TryGetValue(PositionTable, out GgufTensorInfo info) || info.Shape.Length != 2 || (int)info.Shape[0] != EmbeddingLength)
                throw new NotSupportedException($"muse-glimmer vision: the mmproj has no [{EmbeddingLength}, side * side] '{PositionTable}'");
            long rows = (long)info.Shape[1];
            GridSide = (int)Math.Round(Math.Sqrt(rows));
            if ((long)GridSide * GridSide != rows)
                throw new NotSupportedException($"muse-glimmer vision: {PositionTable} has {rows} rows, which is not a square grid");
            var values = new float[info.NumElements];
            if (info.Type == GgmlTensorType.F32)
                gguf.ReadTensorDataToFloat32(info, values, values.LongLength);
            else
                NativeDequant.DequantizeToFloat32((int)info.Type, gguf.ReadTensorData(info), 0, values, 0, values.LongLength);
            return values;
        }

        private void ValidateWeights()
        {
            Require(PatchKernel, -1);
            foreach (string n in new[] { "v.pre_ln.weight", "v.pre_ln.bias", "v.post_ln.weight", "v.post_ln.bias", "mm.0.weight", "mm.1.weight", "mm.2.weight" })
                Require(n, -1);
            string[] block = { "ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias", "attn_q.weight", "attn_q.bias", "attn_k.weight", "attn_k.bias",
                "attn_v.weight", "attn_v.bias", "attn_out.weight", "attn_out.bias", "ffn_up.weight", "ffn_up.bias", "ffn_down.weight", "ffn_down.bias" };
            for (int l = 0; l < BlockCount; l++)
                foreach (string n in block) Require($"v.blk.{l}.{n}", l);
            long patchValues = 3L * PatchSize * PatchSize;
            if (Weights.Get(PatchKernel).Elements != patchValues * EmbeddingLength)
                throw new NotSupportedException($"muse-glimmer vision: {PatchKernel} has {Weights.Get(PatchKernel).Elements} values, not {patchValues} x {EmbeddingLength}");
            if (Weights.Get("mm.2.weight").Rows != ProjectionDim)
                throw new NotSupportedException($"muse-glimmer vision: mm.2 has {Weights.Get("mm.2.weight").Rows} rows, projection_dim is {ProjectionDim}");

            void Require(string name, int layer)
            {
                if (!Weights.Contains(name))
                    throw new NotSupportedException(layer < 0
                        ? $"muse-glimmer vision: the mmproj lacks '{name}'"
                        : $"muse-glimmer vision: layer {layer} lacks '{name}'");
            }
        }

        // ================================================================ encoding

        public Tensor Encode(float[] pixels, int width, int height, Action betweenParts = null)
        {
            float[] rows = EncodeToHost(pixels, width, height, out int tokens, betweenParts);
            var result = new Tensor(Weights.Allocator, DType.Float32, tokens, ProjectionDim);
            result.SetElementsAsFloat(rows);
            return result;
        }

        /// <summary>As <see cref="Encode"/>, the rows as host values [tokens * <see cref="ProjectionDim"/>].</summary>
        internal float[] EncodeToHost(float[] pixels, int width, int height, out int tokens, Action betweenParts = null)
        {
            ArgumentNullException.ThrowIfNull(pixels);
            ObjectDisposedException.ThrowIf(_disposed, this);
            int unit = PatchSize * MergeSize;
            if (width <= 0 || height <= 0 || width % unit != 0 || height % unit != 0)
                throw new ArgumentException($"muse-glimmer vision expects a {unit}-aligned image, got {width}x{height}.");
            if (pixels.LongLength < 3L * width * height)
                throw new ArgumentException($"pixels holds {pixels.Length} floats, expected {3L * width * height} for [3, {height}, {width}].");

            long start = Stopwatch.GetTimestamp();
            int gw = width / PatchSize, gh = height / PatchSize;
            Geometry geo = GeometryOf(gw, gh);
            int n = geo.Patches;
            int values = 3 * PatchSize * PatchSize;
            float[] patches = MuseGlimmerVisionGrid.Patchify(pixels, width, height, PatchSize, geo.Order);
            int parts = PartsFor(n, betweenParts != null);
            float[] rows;
            lock (_sync)
            {
                if (parts == 1)
                {
                    rows = Graph.Run(b => b.Output(EmitTower(b, b.Input(patches, values, n), geo, 0, BlockCount, stem: true, head: true)))[0];
                }
                else
                {
                    float[] x = null;
                    for (int part = 0; part < parts; part++)
                    {
                        int first = part * BlockCount / parts, last = (part + 1) * BlockCount / parts;
                        if (part > 0) betweenParts();
                        float[] input = x;
                        x = Graph.Run(b => b.Output(part == 0
                            ? EmitTower(b, b.Input(patches, values, n), geo, first, last, stem: true, head: false)
                            : EmitTower(b, b.Input(input, EmbeddingLength, n), geo, first, last, stem: false, head: last == BlockCount)))[0];
                    }
                    rows = x;
                }
                rows = (float[])rows.Clone();
            }
            tokens = geo.Tokens;
            if (_reservation != null && !_weightsUploaded)
            {
                // The weights are on the device now, and the device's readings count them.
                _weightsUploaded = true;
                _reservation.Bytes = ReservedComputeBytes;
            }
            LastParts = parts;
            LastEncodeMs = (Stopwatch.GetTimestamp() - start) * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"  Vision encode: {LastEncodeMs:F0} ms total ({parts} program{(parts == 1 ? "" : "s")} on {Graph.Executor.GetType().Name}), " +
                $"{width}x{height} -> {gw}x{gh} patches ({n}) -> {tokens} tokens");
            return rows;
        }

        /// <summary>Programs an image of <paramref name="patches"/> patches runs as (<see cref="YieldPatches"/>).</summary>
        internal int PartsFor(int patches, bool canYield)
            => !canYield || YieldPatches <= 0 || patches <= YieldPatches
                ? 1
                : Math.Min(Math.Min(MaxParts, BlockCount), (patches + YieldPatches - 1) / YieldPatches);

        // ================================================================ geometry

        /// <summary>Everything that depends only on the patch grid, host-side: the window order, the windows' row ranges, the
        /// rotary positions and the position rows in that order, and the gather back to raster order and into 2 x 2 blocks.</summary>
        internal sealed class Geometry
        {
            public int GridW, GridH, Patches, Tokens;

            /// <summary>Window-order row i holds raster patch Order[i].</summary>
            public int[] Order;

            /// <summary>Window w covers window-order rows [Windows[w], Windows[w + 1]).</summary>
            public int[] Windows;

            /// <summary>1-indexed column and row of each window-order row (the rotary's positions).</summary>
            public int[] Columns, Rows;

            /// <summary>The window-order row each pixel-shuffle slot reads (the un-permute and the 2 x 2 gather in one).</summary>
            public int[] MergeGather;

            /// <summary>[Patches * EmbeddingLength] position rows in window order.</summary>
            public float[] PositionRows;

            public int WindowCount => Windows.Length - 1;
        }

        internal Geometry GeometryOf(int gw, int gh)
        {
            long key = ((long)gh << 32) | (uint)gw;
            lock (_geometries)
            {
                if (_geometries.TryGetValue(key, out Geometry cached))
                {
                    _geometryOrder.Remove(key);
                    _geometryOrder.AddFirst(key);
                    return cached;
                }
            }
            Geometry g = MakeGeometry(gw, gh);
            lock (_geometries)
            {
                if (_geometries.TryAdd(key, g))
                {
                    _geometryOrder.AddFirst(key);
                    while (_geometryOrder.Count > MaxCachedGeometries)
                    {
                        _geometries.Remove(_geometryOrder.Last.Value);
                        _geometryOrder.RemoveLast();
                    }
                }
                return _geometries[key];
            }
        }

        private unsafe Geometry MakeGeometry(int gw, int gh)
        {
            if (gw % MergeSize != 0 || gh % MergeSize != 0)
                throw new ArgumentException($"a {gw}x{gh} patch grid does not split into {MergeSize}x{MergeSize} blocks");
            int[] order = MuseGlimmerVisionGrid.WindowOrder(gw, gh, GridSide, out int[] windows);
            int n = order.Length;
            var columns = new int[n];
            var rows = new int[n];
            for (int i = 0; i < n; i++)
            {
                columns[i] = order[i] % gw + 1;   // 1-indexed, as clip.cpp's set_input and the reference
                rows[i] = order[i] / gw + 1;
            }
            var positions = new float[(long)n * EmbeddingLength];
            fixed (float* table = _positionTable)
            fixed (float* dst = positions)
                MuseGlimmerVisionGrid.ResamplePositionTable(table, GridSide, EmbeddingLength, gw, gh, order, dst);
            return new Geometry
            {
                GridW = gw,
                GridH = gh,
                Patches = n,
                Tokens = n / (MergeSize * MergeSize),
                Order = order,
                Windows = windows,
                Columns = columns,
                Rows = rows,
                MergeGather = MuseGlimmerVisionGrid.MergeGather(gw, gh, MergeSize, order),
                PositionRows = positions,
            };
        }

        // ================================================================ the network

        /// <summary>Layers [<paramref name="firstLayer"/>, <paramref name="lastLayer"/>) over [width, patches] activations in
        /// window order: from the patch rows [3 * patch * patch, patches] when <paramref name="stem"/> (the patch product, the
        /// position rows and pre_ln first), with post_ln, the merge and the adapter after them when <paramref name="head"/>
        /// ([<see cref="ProjectionDim"/>, tokens]).</summary>
        internal NetValue EmitTower(NetworkBuilder b, NetValue input, Geometry geo, int firstLayer, int lastLayer, bool stem, bool head)
        {
            int c = EmbeddingLength, n = geo.Patches;
            float eps = NormEps;
            NetValue x = input;
            if (stem)
            {
                x = b.MatMul(b.Weight(PatchKernel, 3L * PatchSize * PatchSize, c), input);
                x = b.Add(x, b.Input(geo.PositionRows, c, n));
                x = b.LayerNorm(x, "v.pre_ln.weight", "v.pre_ln.bias", eps);
            }
            NetValue columns = default, rows = default;
            for (int l = firstLayer; l < lastLayer; l++)
            {
                string p = $"v.blk.{l}.";
                bool global = l == BlockCount - 1 || (l + 1) % SparseFactor == 0;
                NetValue h = b.LayerNorm(x, p + "ln1.weight", p + "ln1.bias", eps);
                NetValue q = b.Reshape(b.Linear(h, p + "attn_q.weight", p + "attn_q.bias"), HeadDim, HeadCount, n);
                NetValue k = b.Reshape(b.Linear(h, p + "attn_k.weight", p + "attn_k.bias"), HeadDim, HeadCount, n);
                NetValue v = b.Reshape(b.Linear(h, p + "attn_v.weight", p + "attn_v.bias"), HeadDim, HeadCount, n);
                if (!columns.Valid)
                {
                    columns = b.InputI32(geo.Columns);
                    rows = b.InputI32(geo.Rows);
                }
                q = Rotate(b, q, columns, rows);
                k = Rotate(b, k, columns, rows);
                NetValue attended = global || geo.WindowCount == 1 ? Attend(b, q, k, v, 0, n) : AttendWindows(b, q, k, v, geo.Windows);
                x = b.Add(x, b.Linear(attended, p + "attn_out.weight", p + "attn_out.bias"));
                h = b.LayerNorm(x, p + "ln2.weight", p + "ln2.bias", eps);
                x = b.Add(x, b.GeluMlp(h, p + "ffn_up.weight", p + "ffn_up.bias", p + "ffn_down.weight", p + "ffn_down.bias"));
            }
            if (!head) return x;
            x = b.LayerNorm(x, "v.post_ln.weight", "v.post_ln.bias", eps);
            // Raster order and 2 x 2 blocks in one gather, then each block's four patches channel-outer: out[c * 4 + s].
            int group = MergeSize * MergeSize;
            NetValue grouped = b.GetRows(x, b.InputI32(geo.MergeGather));
            NetValue merged = b.Reshape(b.Permute(b.Reshape(grouped, c, group, geo.Tokens), 1, 0, 2, 3), (long)c * group, geo.Tokens);
            NetValue a = b.GeluErf(b.Linear(merged, "mm.0.weight"));
            a = b.GeluErf(b.Linear(a, "mm.1.weight"));
            return b.Linear(a, "mm.2.weight");
        }

        /// <summary>
        /// The 2-D rotary over <paramref name="x"/> [d, heads, n] (llama.cpp's build_rope_2d without interleave_freq): ggml NORM
        /// (adjacent pairs) over d / 2 dims, the first half of each head at <paramref name="columns"/>, the second at
        /// <paramref name="rows"/>, both halves with the frequencies of a d / 2-dim rotary. Each half is a strided view of the
        /// heads, rotated by its positions, and the two are concatenated: llama.cpp's second rotary is offset to the second
        /// half (ggml_rope_set_offset), which the graph IR's rotary does not carry. Both lowerings were measured on every Mac
        /// executor (S4): the same rows on managed, ggml_cpu and ggml_metal, and the views' rotary 1.6% slower on ggml_metal
        /// (307 vs 302 ms a 448 x 448 image); the offset needed a new IR parameter, and its tensor-executor emulation was 6.7e-3
        /// off on MLX.
        /// </summary>
        private NetValue Rotate(NetworkBuilder b, NetValue x, NetValue columns, NetValue rows)
        {
            int half = HeadDim / 2;
            long d = x.Ne0, heads = x.Ne1, n = x.Ne2;
            NetValue first = b.View(x, new[] { (long)half, heads, n }, new[] { d * 4, d * heads * 4 }, 0);
            NetValue second = b.View(x, new[] { (long)half, heads, n }, new[] { d * 4, d * heads * 4 }, half * 4L);
            if (_packsRotaryViews)
            {
                first = b.Cont(first);
                second = b.Cont(second);
            }
            return b.Concat(RopeNorm(b, first, columns, half), RopeNorm(b, second, rows, half), 0);
        }

        /// <summary>GraphOp.RopeExt in ggml's NORM mode (adjacent pairs) over the <paramref name="dims"/> values of each head of
        /// <paramref name="x"/> at <paramref name="positions"/>.</summary>
        private NetValue RopeNorm(NetworkBuilder b, NetValue x, NetValue positions, int dims)
            => b.Op(GraphOp.RopeExt, new[] { x, positions }, new[] { x.Ne0, x.Ne1, x.Ne2 }, new long[] { dims, 0, 0 },
                new[] { RopeTheta, 1f, 0f, 1f, 32f, 1f });

        /// <summary>Attention of rows [<paramref name="start"/>, + <paramref name="length"/>) over themselves: [d * heads, length].</summary>
        private NetValue Attend(NetworkBuilder b, NetValue q, NetValue k, NetValue v, int start, int length)
        {
            NetValue qs = b.Slice(q, 2, start, length), ks = b.Slice(k, 2, start, length), vs = b.Slice(v, 2, start, length);
            float scale = 1f / MathF.Sqrt(HeadDim);
            int chunk = QueryChunk(length);
            if (chunk >= length) return b.Attention(qs, ks, vs, scale);
            var pieces = new List<NetValue>();
            for (int at = 0; at < length; at += chunk)
                pieces.Add(b.Attention(b.Slice(qs, 2, at, Math.Min(chunk, length - at)), ks, vs, scale));
            return ConcatAll(b, pieces, 1);
        }

        /// <summary>Each window's rows attended over themselves, the windows' results side by side.</summary>
        private NetValue AttendWindows(NetworkBuilder b, NetValue q, NetValue k, NetValue v, int[] windows)
        {
            var pieces = new List<NetValue>(windows.Length - 1);
            for (int w = 0; w + 1 < windows.Length; w++)
                pieces.Add(Attend(b, q, k, v, windows[w], windows[w + 1] - windows[w]));
            return ConcatAll(b, pieces, 1);
        }

        /// <summary>The values joined along <paramref name="axis"/> pairwise (each element is copied log2(count) times, not up to
        /// count times as a running concat would).</summary>
        private static NetValue ConcatAll(NetworkBuilder b, List<NetValue> pieces, int axis)
        {
            while (pieces.Count > 1)
            {
                var next = new List<NetValue>((pieces.Count + 1) / 2);
                for (int i = 0; i < pieces.Count; i += 2)
                    next.Add(i + 1 < pieces.Count ? b.Concat(pieces[i], pieces[i + 1], axis) : pieces[i]);
                pieces = next;
            }
            return pieces[0];
        }

        /// <summary>The device memory the compute of the largest image takes beyond the weights (an image of
        /// <see cref="MaxTokens"/> merged tokens), which <see cref="_reservation"/> keeps claimed.</summary>
        internal long ReservedComputeBytes { get; }

        /// <summary>The bytes the tower claims on its device now (0 where it claims none).</summary>
        internal long ReservedBytes => _disposed ? 0 : _reservation?.Bytes ?? 0;

        /// <summary>
        /// The device memory an image of <paramref name="patches"/> patches takes to encode beyond the weights: per patch, a
        /// layer's activations at its widest (the MLP's <see cref="FeedForwardLength"/> beside the
        /// <see cref="EmbeddingLength"/> rows) times what the executor keeps live, and on the executors whose attention holds
        /// its scores the score budget of one piece. Measured on an RTX PRO 4500 Blackwell with a 1792 x 1792 image (16384
        /// patches) as the first image of a process, beyond the 1951 MiB of weights: ggml_cuda 1496 MiB and ggml_vulkan 1216
        /// MiB (flash attention, one arena; nvidia-smi's peak, or ggml's reading where higher), direct CUDA 3463 MiB
        /// (materialized scores in 256 MiB pieces, the pool's blocks); these factors give 1640 and 3864 MiB. On an M5 Pro
        /// (48 GB) ggml_metal took 795 MiB by ggml's reading and 1010 MiB by the wired memory's growth past the weights; its
        /// factor gives 1148 MiB.
        /// </summary>
        internal long ComputeBytes(long patches)
        {
            long perPatch = ((long)FeedForwardLength + EmbeddingLength) * sizeof(float);
            if (_materializesAttention) return patches * perPatch * 11 / 2 + AttentionScoreBudget;
            return _metal ? patches * perPatch * 7 / 4 : patches * perPatch * 5 / 2;
        }

        private DeviceReservation ReserveDevice(IAllocator allocator, BackendType backend, long bytes)
        {
            DeviceReservation reservation = backend switch
            {
                BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlMetal
                    => GpuMemoryBudget.Reserve(backend, GpuMemoryBudget.GgmlDevice, "muse-glimmer vision tower", bytes),
                BackendType.Cuda when allocator is TensorSharp.Cuda.CudaAllocator cuda => GpuMemoryBudget.Reserve(backend, cuda.DeviceId,
                    "muse-glimmer vision tower", bytes),
                _ => null,
            };
            if (reservation != null)
                Console.WriteLine($"muse-glimmer vision tower: {bytes / 1048576.0:F0} MiB reserved on {backend} device {reservation.Device} " +
                    $"({(Weights.QuantizedBytes + Weights.DenseBytes) / 1048576.0:F0} MiB of weights until the first image, " +
                    $"{ReservedComputeBytes / 1048576.0:F0} MiB for a {MaxTokens}-token image's compute)");
            return reservation;
        }

        private int QueryChunk(int n)
        {
            if (AttentionChunkRows > 0) return AttentionChunkRows;
            if (!_materializesAttention) return n;
            long perRow = (long)HeadCount * n * sizeof(float);
            long rows = AttentionScoreBudget / perRow / 64 * 64;
            return rows >= n ? n : (int)Math.Max(64, rows);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _reservation?.Dispose();
            Graph?.Dispose();
            Weights?.Dispose();
            lock (_geometries)
            {
                _geometries.Clear();
                _geometryOrder.Clear();
            }
        }
    }
}
