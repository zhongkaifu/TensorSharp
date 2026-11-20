// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Muse-Glimmer vision tower (clip.projector_type "muse-glimmer"), ported from
    /// llama.cpp tools/mtmd/models/muse-glimmer.cpp + clip.cpp build_vit.
    ///
    /// Shape of the graph (hidden 1536, 16 heads, head_dim 96, 50 blocks):
    ///   a) conv2d patch embed (stride 14, no bias), run as im2col + GEMM;
    ///   b) + bilinearly resized learned position embedding (a 32x32 grid of
    ///      1536-dim vectors, ggml GGML_SCALE_MODE_BILINEAR: half-pixel centres,
    ///      align_corners = false);
    ///   c) sparse window permutation: patches are regrouped into 32x32 windows;
    ///   d) pre_ln, then 50 pre-LN blocks (LayerNorm with bias, biased QKV/out,
    ///      2-linear erf-GELU MLP, no gate, no q/k norm), then post_ln. Layer il is
    ///      GLOBAL when (il + 1) % 4 == 0 or il == 49; every other layer attends
    ///      only inside its own window. Because the permutation makes the window
    ///      mask block-diagonal over CONTIGUOUS row ranges, the window layers run
    ///      as one attention call per window instead of materializing an
    ///      n_tok x n_tok mask (n_tok reaches 16384);
    ///   e) 2D RoPE per block: head channels [0,48) rotate with the 1-indexed
    ///      column position, [48,96) with the 1-indexed row position. ggml mode 0
    ///      (NORM) => interleaved adjacent pairs inside each 48-channel slice;
    ///   f) un-permute + 2x2 pixel shuffle with CHANNEL-OUTER packing;
    ///   g) adapter mm.0 -> erf-GELU -> mm.1 -> erf-GELU -> mm.2 (no biases).
    ///
    /// Weight residency: 2D matmul weights stay in their GGUF quantization on any
    /// backend whose quantized mul_mat can consume them (GGML, and the MLX subset
    /// MlxQuantizedOps.CanPreloadQuantizedType admits); everything else is
    /// dequantized to F32 at load and stored ALREADY TRANSPOSED ([inDim, outDim])
    /// so only one copy of each is resident. This tower is ~1.9B parameters, so
    /// the all-F32 working set is ~7.4 GB; see the note on LoadWeights.
    /// </summary>
    public sealed class MuseGlimmerVisionEncoder : IDisposable
    {
        /// <summary>
        /// Transposed [3*14*14, hidden] view of v.patch_embd.weight used by the
        /// im2col GEMM. Stored under its own key because it is not a plain linear.
        /// </summary>
        private const string PatchEmbedKey = "v.patch_embd.weight.T";

        private const string PositionEmbedName = "v.position_embd.weight";

        /// <summary>
        /// hparams.muse_glimmer_sparse_factor in llama.cpp: 3 window layers then 1
        /// global layer, repeating; the last layer is always global.
        /// </summary>
        private const int SparseFactor = 4;

        private readonly Dictionary<string, Tensor> _weights = new();

        /// <summary>
        /// The tower's 2D matmul weights kept in their GGUF quantization on any
        /// backend that can multiply against them: GgmlBasicOps.AddmmQuant (GGML,
        /// which keeps them device-resident behind the shared cacheable-buffer
        /// cache) or MlxQuantizedOps.TryAddmmQuantizedToFloat32 (MLX, behind its
        /// own device-weight cache keyed by QuantizedWeight.EnsureDeviceCacheKey).
        /// </summary>
        private readonly Dictionary<string, QuantizedWeight> _quantWeights = new();
        private readonly Dictionary<long, Geometry> _geometryCache = new();
        private readonly Dictionary<long, Tensor> _positionEmbeddingCache = new();

        private readonly IAllocator _allocator;
        // GGML backend: native flash attention is available through
        // Ops.ScaledDotProductAttention, GgmlBasicOps.AddmmQuant can multiply
        // against ANY GGUF quantization (so no per-type capability predicate is
        // needed on this backend), and host writes need an explicit device-cache
        // invalidation. Note this flag is a genuine backend-identity test - it
        // selects ggml entry points - and must not be used as a proxy for
        // "the device can do X"; see LoadWeights and AttentionSegment.
        private readonly bool _useNativeAttention;
        // Direct-CUDA backend: tensor data lives in device memory, so every raw
        // host pointer checkout forces a synchronous DtoH copy plus a re-upload.
        // Write-only fills stage through pinned host memory instead (HostStaging).
        private readonly bool _cudaDirect;
        // MLX backend: has both a native flash-attention op and quantized matmuls
        // (for the subset of GGUF types MlxQuantizedOps.CanPreloadQuantizedType
        // admits). Host writes need no explicit invalidation - MlxStorage marks
        // itself host-dirty when a raw pointer is checked out.
        private readonly bool _mlxDirect;

        private ModelBase _hostModel;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _mergeSize;
        private readonly float _ropeTheta;
        /// <summary>sqrt(position_embd rows) = 32; also the sparse-window side.</summary>
        private readonly int _posGridPerSide;
        private readonly string[] _blockPrefixes;

        public int ProjectionDim => _projectionDim;
        public int PatchSize => _patchSize;
        public int MergeSize => _mergeSize;
        public MuseGlimmerImageProcessor ImageProcessor { get; }

        /// <summary>
        /// Optional ModelBase reference so the encoder can yield GpuComputeLock
        /// between blocks (see Qwen35VisionEncoder / Gemma4VisionEncoder).
        /// </summary>
        public void SetHostModel(ModelBase model) => _hostModel = model;

        private bool _fusedVisionBlockUnavailable;

        // TS_MUSE_GLIMMER_VENC_TRACE=1 prints a checksum of the residual stream at
        // every encoder stage, for localizing a numeric divergence against
        // llama.cpp's cb() dumps. Forces a host read, so it is debug-only.
        private static readonly bool s_traceEnabled =
            Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_VENC_TRACE") == "1";

        public MuseGlimmerVisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            _cudaDirect = allocator is TensorSharp.Cuda.CudaAllocator;
            _mlxDirect = allocator is MlxAllocator;

            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("clip.vision.image_size", 896);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1536);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 8960);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 50);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 6656);
            _mergeSize = (int)gguf.GetUint32("clip.vision.spatial_merge_size", 2);
            // The mmproj does not carry a rope base for this projector; llama.cpp
            // hardcodes hparams.rope_theta = 10000 in the MUSE_GLIMMER branch.
            _ropeTheta = gguf.GetFloat32("clip.vision.rope.freq_base", 10000f);
            _headDim = _hiddenSize / _numHeads;

            float[] mean = gguf.GetFloatArray("clip.vision.image_mean");
            float[] std = gguf.GetFloatArray("clip.vision.image_std");

            _blockPrefixes = new string[_blockCount];
            for (int i = 0; i < _blockCount; i++)
                _blockPrefixes[i] = $"v.blk.{i}";

            LoadWeights(gguf);
            gguf.Dispose();

            long posRows = _weights[PositionEmbedName].Sizes[0];
            _posGridPerSide = (int)Math.Round(Math.Sqrt(posRows));
            if ((long)_posGridPerSide * _posGridPerSide != posRows)
                throw new InvalidOperationException($"{PositionEmbedName} has {posRows} rows, which is not a square grid.");

            // set_limit_image_tokens(1, 4096): image_max_pixels = 4096 * patch_area,
            // so max_tokens = image_max_pixels / patch_area = 4096.
            ImageProcessor = new MuseGlimmerImageProcessor(_patchSize, _mergeSize, 4096, mean, std);

            Console.WriteLine($"Muse-Glimmer vision encoder: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, intermediate={_intermediateSize}, heads={_numHeads}, headDim={_headDim}, " +
                $"blocks={_blockCount}, projDim={_projectionDim}, mergeSize={_mergeSize}, " +
                $"posGrid={_posGridPerSide}x{_posGridPerSide}, ropeTheta={_ropeTheta}, eps={_eps}");
        }

        /// <summary>
        /// Dequantize to F32 every mmproj tensor the active backend cannot multiply
        /// against in quantized form. 2D matmul weights land already transposed to
        /// [inDim, outDim] so Ops.Addmm can consume them directly and only one copy
        /// stays resident (the reference encoders keep both the original and a
        /// lazily built transpose; at 1.9B parameters that would double an already
        /// large F32 footprint).
        ///
        /// v.position_embd.weight is excluded: it is a lookup table, not a linear.
        /// </summary>
        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading Muse-Glimmer vision encoder weights...");
            int count = 0;
            int quantized = 0;
            long quantBytes = 0;
            long f32Bytes = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                // Spark packages vision and audio in one file. The vision tower
                // must not allocate or upload the unrelated audio pyramid.
                if (info.Name.StartsWith("a.", StringComparison.Ordinal))
                    continue;
                byte[] raw = gguf.ReadTensorData(info);

                // Keep the big 2D matmul weights in their GGUF quantization and let
                // the backend's quantized mul_mat consume them directly. This tower
                // is ~1.9B parameters: dequantized to F32 it is ~7.4 GB, which on a
                // 16 GB card evicts the language model's device-resident weights and
                // collapses decode throughput. The GGUF layout is already
                // [ne0 = inDim, ne1 = outDim], exactly what AddmmQuant wants, so the
                // host transpose disappears too.
                //
                // Residency is a CAPABILITY question, not an allocator-type one.
                // Gating this on "allocator is GgmlAllocator" made MLX dequantize
                // the whole mmproj: on an M5 Pro with the same file, ggml_metal
                // loaded "303 kept quantized: 1942 MB quantized + 13 MB F32" while
                // MLX loaded "0 kept quantized: 0 MB quantized + 7327 MB F32" - 7.3
                // GB taken straight out of the language model's unified-memory
                // budget. MLX can only multiply against the subset that
                // CanPreloadQuantizedType admits (the predicate is the authority;
                // it grows as MLX gains kernels), so ask it per tensor and let the
                // rest fall through to the dequantized path below.
                bool keepQuantized = _useNativeAttention
                    || (_mlxDirect && MlxQuantizedOps.CanPreloadQuantizedType((int)info.Type));
                if (keepQuantized
                    && info.Type != GgmlTensorType.F32
                    && info.Shape.Length == 2
                    && info.Name != PositionEmbedName
                    && info.Name.EndsWith(".weight", StringComparison.Ordinal))
                {
                    _quantWeights[info.Name] = new QuantizedWeight(
                        raw, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
                    quantBytes += raw.Length;
                    quantized++;
                    count++;
                    continue;
                }

                long numElements = info.NumElements;
                float[] f32 = new float[numElements];
                if (info.Type == GgmlTensorType.F32)
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);
                f32Bytes += numElements * sizeof(float);

                // GGUF stores ne-order (fastest dim first); TensorSharp shapes are
                // the reverse.
                long[] tsShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    tsShape[i] = (long)info.Shape[info.Shape.Length - 1 - i];

                if (info.Name == "v.patch_embd.weight")
                {
                    // [hidden, 3, 14, 14] laid out row-major per output channel in
                    // (ic, ky, kx) order, exactly the order the im2col rows use.
                    int outDim = (int)tsShape[0];
                    int inDim = (int)(numElements / outDim);
                    _weights[PatchEmbedKey] = UploadTransposed(f32, outDim, inDim);
                    count++;
                    continue;
                }

                if (tsShape.Length == 2 && info.Name != PositionEmbedName &&
                    info.Name.EndsWith(".weight", StringComparison.Ordinal))
                {
                    _weights[info.Name] = UploadTransposed(f32, (int)tsShape[0], (int)tsShape[1]);
                    count++;
                    continue;
                }

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors, {quantized} kept quantized: " +
                $"{quantBytes / (1024 * 1024)} MB quantized + {f32Bytes / (1024 * 1024)} MB F32)");
        }

        /// <summary>Host transpose of a row-major [rows, cols] buffer into a [cols, rows] tensor.</summary>
        private unsafe Tensor UploadTransposed(float[] src, int rows, int cols)
        {
            var dst = new float[(long)rows * cols];
            fixed (float* srcPtr = src)
            fixed (float* dstPtr = dst)
            {
                long srcL = (long)srcPtr;
                long dstL = (long)dstPtr;
                Parallel.For(0, cols, c =>
                {
                    float* s = (float*)srcL;
                    float* d = (float*)dstL + (long)c * rows;
                    for (int r = 0; r < rows; r++)
                        d[r] = s[(long)r * cols + c];
                });
            }

            var tensor = new Tensor(_allocator, DType.Float32, cols, rows);
            tensor.SetElementsAsFloat(dst);
            return tensor;
        }

        // ------------------------------------------------------------------
        // Public API
        // ------------------------------------------------------------------

        /// <summary>
        /// Encode preprocessed pixels into vision embeddings ready for injection
        /// into the text model.
        /// </summary>
        /// <param name="pixels">Channel-first [3, height, width] normalized pixels.</param>
        /// <param name="width">Image width in pixels (a multiple of patch * merge).</param>
        /// <param name="height">Image height in pixels (a multiple of patch * merge).</param>
        /// <returns>[numTokens, ProjectionDim] embeddings in original grid order.</returns>
        public Tensor Encode(float[] pixels, int width, int height)
        {
            if (pixels == null)
                throw new ArgumentNullException(nameof(pixels));

            int patchHw = _patchSize * _mergeSize;
            if (width <= 0 || height <= 0 || width % patchHw != 0 || height % patchHw != 0)
                throw new ArgumentException($"Muse-Glimmer expects a {patchHw}-aligned image, got {width}x{height}.");
            if ((long)pixels.Length < 3L * width * height)
                throw new ArgumentException($"pixels holds {pixels.Length} floats, expected {3L * width * height} for [3, {height}, {width}].");

            long encodeStart = Stopwatch.GetTimestamp();

            int gridW = width / _patchSize;
            int gridH = height / _patchSize;
            Geometry geo = GetOrCreateGeometry(gridW, gridH);
            int numPatches = geo.NumTokens;

            // (a) patchify. im2col rows are emitted directly in sparse-window order,
            // and (b) the position embedding table is built in the same order, so
            // the sum reproduces the reference "conv -> add posemb -> get_rows"
            // chain without a separate gather pass.
            long t0 = Stopwatch.GetTimestamp();
            Tensor x = PatchEmbed(pixels, width, height, geo);
            long patchEmbedTicks = Stopwatch.GetTimestamp() - t0;
            Trace("patchEmbed", x);

            Ops.Add(x, x, GetOrCreatePositionEmbedding(geo));
            Trace("after_posemb+sp_perm", x);

            // (d) pre_ln
            Tensor normed = LayerNormOp(x, "v.pre_ln.weight", "v.pre_ln.bias");
            x.Dispose();
            x = normed;
            Trace("pre_ln", x);

            long blocksStart = Stopwatch.GetTimestamp();
            for (int il = 0; il < _blockCount; il++)
            {
                bool isGlobal = (il == _blockCount - 1) || ((il + 1) % SparseFactor == 0);
                if (!s_traceEnabled)
                    Console.Write($"\r  Vision encoder block {il + 1}/{_blockCount}...");
                EncoderBlock(x, il, geo, isGlobal);
                Trace($"block{il}", x);
                // Yield GpuComputeLock between blocks so concurrent decode
                // requests on the engine worker stay responsive.
                _hostModel?.YieldGpuComputeLock();
            }
            if (_cudaDirect && _allocator is TensorSharp.Cuda.CudaAllocator blocksAllocator)
                blocksAllocator.Synchronize();
            long blocksTicks = Stopwatch.GetTimestamp() - blocksStart;
            if (!s_traceEnabled)
                Console.WriteLine(" done");

            Tensor postNormed = LayerNormOp(x, "v.post_ln.weight", "v.post_ln.bias");
            x.Dispose();
            Trace("post_ln", postNormed);

            // (f) + (g) un-permute and 2x2 pixel shuffle fused into one gather.
            long projStart = Stopwatch.GetTimestamp();
            Tensor merged = PixelShuffle(postNormed, geo);
            postNormed.Dispose();
            Trace("encoder_out", merged);

            // (h) adapter: mm.0 -> erf-GELU -> mm.1 -> erf-GELU -> mm.2, no biases.
            Tensor h0 = LinearForwardWithBias(merged, "mm.0.weight", null);
            merged.Dispose();
            GeluErf(h0);
            Tensor h1 = LinearForwardWithBias(h0, "mm.1.weight", null);
            h0.Dispose();
            GeluErf(h1);
            Tensor projected = LinearForwardWithBias(h1, "mm.2.weight", null);
            h1.Dispose();
            Trace("projected", projected);

            // Direct-CUDA only queues its kernels; the caller consumes the
            // embeddings immediately, so drain before reporting timings.
            if (_cudaDirect && _allocator is TensorSharp.Cuda.CudaAllocator cudaAllocator)
                cudaAllocator.Synchronize();
            long projTicks = Stopwatch.GetTimestamp() - projStart;

            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double totalMs = (Stopwatch.GetTimestamp() - encodeStart) * msPerTick;
            Console.WriteLine($"  Vision encode: {totalMs:F0} ms total " +
                $"(patchEmbed {patchEmbedTicks * msPerTick:F0} ms, " +
                $"blocks {blocksTicks * msPerTick:F0} ms, " +
                $"proj {projTicks * msPerTick:F0} ms), " +
                $"{width}x{height} -> {gridW}x{gridH} patches ({numPatches}) -> {projected.Sizes[0]} tokens");

            return projected;
        }

        /// <summary>
        /// Self-check helper: run <see cref="Encode"/> and return the result as a
        /// flat host array of numTokens * ProjectionDim floats (row-major).
        /// </summary>
        public float[] EncodeToHostArray(float[] pixels, int width, int height, out int numTokens)
        {
            using Tensor result = Encode(pixels, width, height);
            numTokens = (int)result.Sizes[0];
            if (result.IsContiguous())
                return result.GetElementsAsFloat((int)result.ElementCount());

            using Tensor contiguous = Ops.NewContiguous(result);
            return contiguous.GetElementsAsFloat((int)contiguous.ElementCount());
        }

        // ------------------------------------------------------------------
        // Per-geometry host tables
        // ------------------------------------------------------------------

        /// <summary>
        /// Everything that depends only on (gridW, gridH): the sparse-window
        /// permutation, the window row ranges, the fused un-permute + pixel-shuffle
        /// gather, and the 2D RoPE tables (already in permuted order).
        /// </summary>
        private sealed class Geometry
        {
            public int GridW;
            public int GridH;
            public int NumTokens;

            /// <summary>Permuted row i reads raster patch SpPerm[i] (gy * gridW + gx).</summary>
            public int[] SpPerm;

            /// <summary>
            /// Prefix sums of the per-window token counts: window w covers permuted
            /// rows [WindowOffsets[w], WindowOffsets[w + 1]). The reference builds an
            /// n_tok x n_tok mask that is exactly zero on these diagonal blocks and
            /// -inf elsewhere.
            /// </summary>
            public int[] WindowOffsets;

            /// <summary>
            /// MergeGather[j] = inv_perm[ds_perm[j]]: the permuted-order row that
            /// feeds pixel-shuffle slot j. Fuses the un-permute gather and the
            /// pixel-shuffle gather into one pass.
            /// </summary>
            public int[] MergeGather;

            /// <summary>
            /// [NumTokens, headDim/2] rotation tables. Entry (i, p) drives the
            /// interleaved channel pair (2p, 2p + 1): p in [0, headDim/4) uses the
            /// 1-indexed column position, p in [headDim/4, headDim/2) the 1-indexed
            /// row position.
            /// </summary>
            public float[] RopeCos;
            public float[] RopeSin;

            /// <summary>
            /// 1-indexed width/height positions in sparse-permuted token order.
            /// The fused GGML block feeds these directly to ggml_rope_ext.
            /// </summary>
            public int[] RopePosW;
            public int[] RopePosH;
        }

        private Geometry GetOrCreateGeometry(int gridW, int gridH)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (_geometryCache.TryGetValue(key, out var cached))
                return cached;

            int numTokens = gridW * gridH;
            int win = _posGridPerSide;
            int nwinH = (gridH + win - 1) / win;
            int nwinW = (gridW + win - 1) / win;

            var spPerm = new int[numTokens];
            var offsets = new List<int> { 0 };
            int idx = 0;
            for (int wy = 0; wy < nwinH; wy++)
            {
                for (int wx = 0; wx < nwinW; wx++)
                {
                    int cnt = 0;
                    for (int hh = 0; hh < win; hh++)
                    {
                        for (int ww = 0; ww < win; ww++)
                        {
                            int gy = wy * win + hh;
                            int gx = wx * win + ww;
                            if (gy < gridH && gx < gridW)
                            {
                                spPerm[idx++] = gy * gridW + gx;
                                cnt++;
                            }
                        }
                    }
                    if (cnt > 0)
                        offsets.Add(idx);
                }
            }

            var invPerm = new int[numTokens];
            for (int i = 0; i < numTokens; i++)
                invPerm[spPerm[i]] = i;

            // 2D RoPE tables. Each half of the head dim is roped independently with
            // n_dims = headDim / 2 and ggml mode 0 (adjacent pairs), so the
            // frequency exponent denominator is headDim / 2, not headDim.
            int pairs = _headDim / 2;              // 48: pairs per head
            int ropeDim = _headDim / 2;            // 48: n_dims of each half
            int pairsPerHalf = ropeDim / 2;        // 24: pairs inside one half
            var invFreq = new float[pairsPerHalf];
            for (int j = 0; j < pairsPerHalf; j++)
                invFreq[j] = MathF.Pow(_ropeTheta, -2f * j / ropeDim);

            var ropeCos = new float[(long)numTokens * pairs];
            var ropeSin = new float[(long)numTokens * pairs];
            var ropePosW = new int[numTokens];
            var ropePosH = new int[numTokens];
            for (int i = 0; i < numTokens; i++)
            {
                int orig = spPerm[i];
                int posW = (orig % gridW) + 1;     // 1-indexed, as in clip.cpp set_input
                int posH = (orig / gridW) + 1;
                ropePosW[i] = posW;
                ropePosH[i] = posH;
                long b = (long)i * pairs;
                for (int j = 0; j < pairsPerHalf; j++)
                {
                    float aw = posW * invFreq[j];
                    ropeCos[b + j] = MathF.Cos(aw);
                    ropeSin[b + j] = MathF.Sin(aw);

                    float ah = posH * invFreq[j];
                    ropeCos[b + pairsPerHalf + j] = MathF.Cos(ah);
                    ropeSin[b + pairsPerHalf + j] = MathF.Sin(ah);
                }
            }

            // Pixel-shuffle gather over the ORIGINAL grid order, composed with the
            // inverse of the window permutation so it can be applied directly to the
            // permuted block output.
            int f = _mergeSize;
            var mergeGather = new int[(gridH / f) * (gridW / f) * f * f];
            int m = 0;
            for (int oy = 0; oy < gridH / f; oy++)
            {
                for (int ox = 0; ox < gridW / f; ox++)
                {
                    for (int ry = 0; ry < f; ry++)
                    {
                        for (int rx = 0; rx < f; rx++)
                            mergeGather[m++] = invPerm[(oy * f + ry) * gridW + (ox * f + rx)];
                    }
                }
            }

            cached = new Geometry
            {
                GridW = gridW,
                GridH = gridH,
                NumTokens = numTokens,
                SpPerm = spPerm,
                WindowOffsets = offsets.ToArray(),
                MergeGather = mergeGather,
                RopeCos = ropeCos,
                RopeSin = ropeSin,
                RopePosW = ropePosW,
                RopePosH = ropePosH,
            };
            _geometryCache[key] = cached;
            return cached;
        }

        // ------------------------------------------------------------------
        // Graph stages
        // ------------------------------------------------------------------

        /// <summary>
        /// Conv2d patch embedding (stride = patch size, no padding, no bias)
        /// reformulated as im2col + GEMM. Rows are emitted in sparse-window order.
        /// The im2col row layout is (channel, ky, kx) with kx fastest, matching the
        /// per-output-channel layout of v.patch_embd.weight.
        /// </summary>
        private unsafe Tensor PatchEmbed(float[] pixels, int width, int height, Geometry geo)
        {
            int numTokens = geo.NumTokens;
            const int channels = 3;
            int p = _patchSize;
            int patchStride = channels * p * p;
            int gridW = geo.GridW;

            Tensor weightT = _weights[PatchEmbedKey];

            var im2col = new Tensor(_allocator, DType.Float32, numTokens, patchStride);
            using (var staging = new HostStaging(this, im2col, _cudaDirect))
            {
                float* dstPtr = staging.Ptr;
                fixed (float* pixSrc = pixels)
                fixed (int* orderSrc = geo.SpPerm)
                {
                    long pixL = (long)pixSrc;
                    long orderL = (long)orderSrc;
                    long dstL = (long)dstPtr;
                    Parallel.For(0, numTokens, t =>
                    {
                        float* pix = (float*)pixL;
                        int* order = (int*)orderL;
                        float* outRow = (float*)dstL + (long)t * patchStride;

                        int raster = order[t];
                        int py = raster / gridW;
                        int px = raster - py * gridW;
                        int yBase = py * p;
                        int xBase = px * p;

                        for (int c = 0; c < channels; c++)
                        {
                            long imgChannelOffset = (long)c * height * width;
                            long outChannelOffset = (long)c * p * p;
                            for (int ky = 0; ky < p; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(yBase + ky) * width + xBase;
                                long dstOffset = outChannelOffset + (long)ky * p;
                                Buffer.MemoryCopy(pix + srcOffset, outRow + dstOffset,
                                    p * sizeof(float), p * sizeof(float));
                            }
                        }
                    });
                }
            }

            var result = new Tensor(_allocator, DType.Float32, numTokens, _hiddenSize);
            Ops.Addmm(result, 0, result, 1.0f, im2col, weightT);
            im2col.Dispose();
            return result;
        }

        /// <summary>
        /// Bilinearly resize the learned [posGrid * posGrid, hidden] position table
        /// to the image's patch grid, emitting the rows in sparse-window order: the
        /// table read transposed to the grid, as the model was trained
        /// (<see cref="MuseGlimmerVisionGrid.ResamplePositionTable"/>).
        /// </summary>
        private unsafe Tensor GetOrCreatePositionEmbedding(Geometry geo)
        {
            long key = ((long)geo.GridH << 32) | (uint)geo.GridW;
            if (_positionEmbeddingCache.TryGetValue(key, out var cached))
                return cached;

            cached = new Tensor(_allocator, DType.Float32, geo.NumTokens, _hiddenSize);
            float* tablePtr = GetFloatPtr(_weights[PositionEmbedName]);
            using (var staging = new HostStaging(this, cached, _cudaDirect))
                MuseGlimmerVisionGrid.ResamplePositionTable(tablePtr, _posGridPerSide, _hiddenSize, geo.GridW, geo.GridH, geo.SpPerm, staging.Ptr);

            _positionEmbeddingCache[key] = cached;
            return cached;
        }

        /// <summary>
        /// One pre-LN block, updating the residual stream in place.
        /// FFN is a plain 2-linear erf-GELU MLP (no gate).
        /// </summary>
        private void EncoderBlock(Tensor x, int blockIdx, Geometry geo, bool isGlobal)
        {
            string prefix = _blockPrefixes[blockIdx];
            int numTokens = geo.NumTokens;

            if (TryFusedEncoderBlock(x, prefix, geo, isGlobal))
                return;

            Tensor q, k, v;
            using (Tensor ln1 = LayerNormOp(x, $"{prefix}.ln1.weight", $"{prefix}.ln1.bias"))
            {
                q = LinearForwardWithBias(ln1, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
                k = LinearForwardWithBias(ln1, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
                v = LinearForwardWithBias(ln1, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");
            }

            var attn = new Tensor(_allocator, DType.Float32, numTokens, _hiddenSize);
            try
            {
                ApplyRope2D(q, geo);
                ApplyRope2D(k, geo);

                if (isGlobal)
                {
                    AttentionSegment(attn, q, k, v, 0, numTokens);
                }
                else
                {
                    int[] offsets = geo.WindowOffsets;
                    for (int w = 0; w + 1 < offsets.Length; w++)
                        AttentionSegment(attn, q, k, v, offsets[w], offsets[w + 1] - offsets[w]);
                }
            }
            finally
            {
                q.Dispose();
                k.Dispose();
                v.Dispose();
            }

            Tensor proj = LinearForwardWithBias(attn, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
            attn.Dispose();
            Ops.Add(x, x, proj);
            proj.Dispose();

            using (Tensor ln2 = LayerNormOp(x, $"{prefix}.ln2.weight", $"{prefix}.ln2.bias"))
            {
                Tensor up = LinearForwardWithBias(ln2, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
                GeluErf(up);
                Tensor down = LinearForwardWithBias(up, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
                up.Dispose();
                Ops.Add(x, x, down);
                down.Dispose();
            }
        }

        /// <summary>
        /// One exact, bounded GGML graph for LN/QKV/RoPE/windowed-or-global
        /// flash-attention/output projection plus the erf-GELU MLP. The portable
        /// implementation below remains authoritative on other allocators or if
        /// the active GGML backend cannot execute head-dim-96 flash attention.
        /// </summary>
        private bool TryFusedEncoderBlock(Tensor x, string prefix, Geometry geo, bool isGlobal)
        {
            // CUDA is the validated high-performance target. Metal/Vulkan keep
            // the existing portable path until this graph has backend-specific
            // numerical and allocator coverage there.
            if (!_useNativeAttention || _allocator is not GgmlAllocator ggmlAllocator ||
                ggmlAllocator.Context.BackendType != GgmlBackendType.Cuda ||
                _fusedVisionBlockUnavailable)
                return false;

            if (!_quantWeights.TryGetValue($"{prefix}.attn_q.weight", out QuantizedWeight q) ||
                !_quantWeights.TryGetValue($"{prefix}.attn_k.weight", out QuantizedWeight k) ||
                !_quantWeights.TryGetValue($"{prefix}.attn_v.weight", out QuantizedWeight v) ||
                !_quantWeights.TryGetValue($"{prefix}.attn_out.weight", out QuantizedWeight o) ||
                !_quantWeights.TryGetValue($"{prefix}.ffn_up.weight", out QuantizedWeight up) ||
                !_quantWeights.TryGetValue($"{prefix}.ffn_down.weight", out QuantizedWeight down))
            {
                return false;
            }

            bool ok = GgmlBasicOps.TryMuseGlimmerVisionBlock(
                x,
                _weights[$"{prefix}.ln1.weight"], _weights[$"{prefix}.ln1.bias"],
                q.CacheKey, q.GgmlType, q.Ne0, q.Ne1, q.RawBytes, _weights[$"{prefix}.attn_q.bias"],
                k.CacheKey, k.GgmlType, k.Ne0, k.Ne1, k.RawBytes, _weights[$"{prefix}.attn_k.bias"],
                v.CacheKey, v.GgmlType, v.Ne0, v.Ne1, v.RawBytes, _weights[$"{prefix}.attn_v.bias"],
                o.CacheKey, o.GgmlType, o.Ne0, o.Ne1, o.RawBytes, _weights[$"{prefix}.attn_out.bias"],
                _weights[$"{prefix}.ln2.weight"], _weights[$"{prefix}.ln2.bias"],
                up.CacheKey, up.GgmlType, up.Ne0, up.Ne1, up.RawBytes, _weights[$"{prefix}.ffn_up.bias"],
                down.CacheKey, down.GgmlType, down.Ne0, down.Ne1, down.RawBytes, _weights[$"{prefix}.ffn_down.bias"],
                geo.RopePosW, geo.RopePosH, geo.WindowOffsets, isGlobal,
                _numHeads, _headDim, _eps, _ropeTheta);
            if (ok)
                return true;

            // A geometry/backend rejection is stable for the rest of this
            // encoder instance. Avoid paying 49 more failed graph builds.
            _fusedVisionBlockUnavailable = true;
            return false;
        }

        /// <summary>
        /// Attention over one contiguous range of permuted rows. Window layers call
        /// this once per window (each range is one diagonal block of the reference's
        /// sp_mask, so no mask tensor is needed); global layers call it once over
        /// the whole sequence.
        /// </summary>
        private void AttentionSegment(Tensor attnOut, Tensor q, Tensor k, Tensor v, int offset, int length)
        {
            if (length <= 0)
                return;

            float scale = 1f / MathF.Sqrt(_headDim);

            using Tensor qSeg = q.Narrow(0, offset, length);
            using Tensor kSeg = k.Narrow(0, offset, length);
            using Tensor vSeg = v.Narrow(0, offset, length);
            using Tensor oSeg = attnOut.Narrow(0, offset, length);

            // MLX registers "scaled_dot_product_attention" too, so it belongs here:
            // excluding it sent all 50 ViT blocks (one call per window on the 37
            // sparse layers) down AttentionSegmentFallback's batched-matmul +
            // materialized-scores path for no reason.
            if (_useNativeAttention || _cudaDirect || _mlxDirect)
            {
                try
                {
                    using Tensor q4 = qSeg.View(1, length, _numHeads, _headDim);
                    using Tensor k4 = kSeg.View(1, length, _numHeads, _headDim);
                    using Tensor v4 = vSeg.View(1, length, _numHeads, _headDim);
                    using Tensor o4 = oSeg.View(1, length, _numHeads, _headDim);
                    Ops.ScaledDotProductAttention(o4, q4, k4, v4, null, scale);
                    return;
                }
                catch (Exception)
                {
                    // No native kernel for this head dim / layout: fall through to
                    // the portable batched-matmul path.
                }
            }

            AttentionSegmentFallback(oSeg, qSeg, kSeg, vSeg, length, scale);
        }

        /// <summary>
        /// Portable attention: head-first batched matmul + softmax, streaming the
        /// query rows in chunks so the [heads, chunk, keys] scores intermediate
        /// stays bounded (a global layer at 16384 tokens would otherwise need 16 GB).
        /// </summary>
        private void AttentionSegmentFallback(Tensor oSeg, Tensor qSeg, Tensor kSeg, Tensor vSeg,
            int length, float scale)
        {
            using Tensor kHeads = HeadFirst(kSeg, length);
            using Tensor vHeads = HeadFirst(vSeg, length);
            using Tensor kT = kHeads.Transpose(1, 2);

            int chunkSize = ComputeAttentionChunkSize(length);
            for (int qOff = 0; qOff < length; qOff += chunkSize)
            {
                int qLen = Math.Min(chunkSize, length - qOff);

                using Tensor qSub = qSeg.Narrow(0, qOff, qLen);
                using Tensor qHeads = HeadFirst(qSub, qLen);

                using var scores = new Tensor(_allocator, DType.Float32, _numHeads, qLen, length);
                Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                Ops.Softmax(scores, scores);

                using var outHeadFirst = new Tensor(_allocator, DType.Float32, _numHeads, qLen, _headDim);
                Ops.AddmmBatch(outHeadFirst, 0, outHeadFirst, 1.0f, scores, vHeads);

                using Tensor seqMajor = outHeadFirst.Transpose(0, 1);
                using Tensor contiguous = Ops.NewContiguous(seqMajor);
                using Tensor flat = contiguous.View(qLen, _hiddenSize);
                using Tensor dst = oSeg.Narrow(0, qOff, qLen);
                Ops.Copy(dst, flat);
            }
        }

        /// <summary>[rows, heads * headDim] -> contiguous [heads, rows, headDim].</summary>
        private Tensor HeadFirst(Tensor rowMajor, int rows)
        {
            using Tensor reshaped = rowMajor.View(rows, _numHeads, _headDim);
            using Tensor transposed = reshaped.Transpose(0, 1);
            return Ops.NewContiguous(transposed);
        }

        private const long AttentionChunkBudgetBytes = 256L * 1024 * 1024;

        private int ComputeAttentionChunkSize(int length)
        {
            long perRowBytes = (long)_numHeads * length * sizeof(float);
            if (perRowBytes <= 0)
                return length;
            long maxRows = AttentionChunkBudgetBytes / perRowBytes;
            if (maxRows >= length)
                return length;
            return (int)Math.Max(64, Math.Min(length, maxRows));
        }

        /// <summary>
        /// 2D RoPE over [numTokens, heads * headDim]. ggml's build_rope_2d ropes
        /// channels [0, headDim/2) with the width position and [headDim/2, headDim)
        /// with the height position, both at mode 0 (GGML NORM), i.e. INTERLEAVED
        /// ADJACENT PAIRS inside each slice - see TensorApplyCPU.RoPEEx's non-NeoX
        /// branch. interleave_freq is false, so the second half's freq_scale is 1.
        /// </summary>
        private unsafe void ApplyRope2D(Tensor data, Geometry geo)
        {
            int numTokens = geo.NumTokens;
            int numHeads = _numHeads;
            int headDim = _headDim;
            int pairs = headDim / 2;

            float* ptr = GetFloatPtr(data);
            fixed (float* cosSrc = geo.RopeCos)
            fixed (float* sinSrc = geo.RopeSin)
            {
                long ptrL = (long)ptr;
                long cosL = (long)cosSrc;
                long sinL = (long)sinSrc;

                Parallel.For(0, numTokens, i =>
                {
                    float* tokenBase = (float*)ptrL + (long)i * numHeads * headDim;
                    float* cosRow = (float*)cosL + (long)i * pairs;
                    float* sinRow = (float*)sinL + (long)i * pairs;

                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = tokenBase + (long)h * headDim;
                        for (int j = 0; j < pairs; j++)
                        {
                            float c = cosRow[j];
                            float s = sinRow[j];
                            float x0 = head[2 * j];
                            float x1 = head[2 * j + 1];
                            head[2 * j] = x0 * c - x1 * s;
                            head[2 * j + 1] = x1 * c + x0 * s;
                        }
                    }
                });
            }

            InvalidateTensorDeviceCache(data);
        }

        /// <summary>
        /// Un-permute (inv_perm) + 2x2 pixel shuffle in one gather.
        ///
        /// ggml does reshape_3d(x, n_embd, ds*ds, n_out) -> permute(1, 0, 2) ->
        /// cont -> reshape_2d(n_embd * ds * ds, n_out). In row-major C# terms the
        /// output row o holds
        ///     out[o][c * (ds*ds) + s] = gathered[o * (ds*ds) + s][c]
        /// i.e. CHANNEL-OUTER packing (c is the slow index). The s-outer spelling
        /// is the classic trap here and is silently wrong.
        /// </summary>
        private unsafe Tensor PixelShuffle(Tensor x, Geometry geo)
        {
            int group = _mergeSize * _mergeSize;
            int numOut = (geo.GridW / _mergeSize) * (geo.GridH / _mergeSize);
            int hiddenSize = _hiddenSize;
            int mergedDim = hiddenSize * group;

            var result = new Tensor(_allocator, DType.Float32, numOut, mergedDim);
            float* srcPtr = GetFloatPtr(x);

            using (var staging = new HostStaging(this, result, _cudaDirect))
            {
                float* dstPtr = staging.Ptr;
                fixed (int* gatherSrc = geo.MergeGather)
                {
                    long srcL = (long)srcPtr;
                    long dstL = (long)dstPtr;
                    long gatherL = (long)gatherSrc;

                    Parallel.For(0, numOut, o =>
                    {
                        float* src = (float*)srcL;
                        float* dstRow = (float*)dstL + (long)o * mergedDim;
                        int* gather = (int*)gatherL;

                        for (int s = 0; s < group; s++)
                        {
                            float* row = src + (long)gather[o * group + s] * hiddenSize;
                            float* dst = dstRow + s;
                            for (int c = 0; c < hiddenSize; c++)
                                dst[c * group] = row[c];
                        }
                    });
                }
            }

            return result;
        }

        // ------------------------------------------------------------------
        // Primitives
        // ------------------------------------------------------------------

        /// <summary>
        /// result = input @ weightT (+ bias). Matmul weights are stored transposed
        /// ([inDim, outDim]) by LoadWeights. Pass biasName = null for the adapter
        /// matrices, which have no bias.
        /// </summary>
        private Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            int seqLen = (int)input.Sizes[0];

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            Tensor result;
            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                // GGUF layout is [ne0 = inDim, ne1 = outDim] - the orientation
                // the quantized mul_mat kernels expect, so no transpose is involved.
                result = new Tensor(_allocator, DType.Float32, seqLen, (int)qw.Ne1);
                AddmmQuant(result, src, qw);
            }
            else
            {
                Tensor weightT = _weights[weightName];
                result = new Tensor(_allocator, DType.Float32, seqLen, (int)weightT.Sizes[1]);
                Ops.Addmm(result, 0, result, 1.0f, src, weightT);
            }

            contiguousInput?.Dispose();

            if (biasName != null && _weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        /// <summary>
        /// result = src @ weight for a weight still in its GGUF quantization,
        /// dispatched on the allocator exactly like ModelBase.AddmmQuantManaged.
        /// Hard-wiring GgmlBasicOps.AddmmQuant here would have been a silent
        /// correctness bug the moment LoadWeights started retaining quantized
        /// weights on MLX, since the ggml entry point knows nothing about MLX
        /// storages.
        /// </summary>
        private unsafe void AddmmQuant(Tensor result, Tensor src, QuantizedWeight qw)
        {
            // EnsureDeviceCacheKey, not Data: MLX caches the repacked device-side
            // weight under that key, so passing the raw host pointer would repack
            // the weight on every call.
            if (_mlxDirect && MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                    result, src, qw.EnsureDeviceCacheKey(), qw.Data,
                    qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
            {
                return;
            }

            if (_useNativeAttention)
            {
                // CacheKey, not Data: the GGML device cache is keyed by it, and
                // after a preload it is an opaque handle rather than the host
                // pointer. Passing Data would miss the resident copy and re-upload
                // the weight on every call.
                GgmlBasicOps.AddmmQuant(result, src, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                return;
            }

            // MLX declined (an input shape/layout its validator rejects). There is
            // no F32 copy of this weight - LoadWeights kept only the quantized
            // bytes - so dequantize through the managed kernel, the same last
            // resort ModelBase.AddmmQuantManaged falls back to.
            ManagedQuantizedOps.AddmmQuantizedToFloat32(
                qw.GgmlType,
                qw.Data,
                qw.Ne0,
                qw.Ne1,
                GetFloatPtr(src),
                (int)qw.Ne0,
                (int)src.Sizes[0],
                GetFloatPtr(result),
                (int)qw.Ne1);
            InvalidateTensorDeviceCache(result);
        }

        /// <summary>LayerNorm with bias (clip.cpp NORM_TYPE_NORMAL).</summary>
        private Tensor LayerNormOp(Tensor input, string weightName, string biasName)
        {
            _weights.TryGetValue(biasName, out var bias);
            return Ops.LayerNorm(null, input, _weights[weightName], bias, _eps);
        }

        /// <summary>1 / sqrt(2), the constant ggml's gelu_erf feeds to erff.</summary>
        private const float InvSqrt2 = 0.70710678118654752440084436210484f;

        /// <summary>
        /// ggml_gelu_erf: 0.5 * x * (1 + erf(x / sqrt(2))). TensorSharp's Ops.GELU
        /// is the tanh approximation, which is NOT what the reference graph uses,
        /// so this runs as a host loop.
        /// </summary>
        private unsafe void GeluErf(Tensor t)
        {
            long total = t.ElementCount();
            if (total == 0)
                return;

            int rows = (int)t.Sizes[0];
            long cols = total / rows;
            float* ptr = GetFloatPtr(t);
            long ptrL = (long)ptr;

            Parallel.For(0, rows, r =>
            {
                float* row = (float*)ptrL + r * cols;
                for (long i = 0; i < cols; i++)
                    row[i] = 0.5f * row[i] * (1.0f + Erf(row[i] * InvSqrt2));
            });

            InvalidateTensorDeviceCache(t);
        }

        /// <summary>
        /// Abramowitz and Stegun 7.1.26, evaluated in double precision: maximum
        /// absolute error 1.5e-7, i.e. at the resolution of the float32 result this
        /// feeds. Used because .NET has no Math.Erf and the exact series/continued
        /// fraction forms are far too slow for the ~1e9 evaluations a full-size
        /// image costs.
        /// </summary>
        private static float Erf(float value)
        {
            double x = value;
            double sign = x < 0.0 ? -1.0 : 1.0;
            x = Math.Abs(x);
            if (x > 6.0)
                return (float)sign;   // |erf| is 1 to well within float precision

            double t = 1.0 / (1.0 + 0.3275911 * x);
            double poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
                            - 0.284496736) * t + 0.254829592) * t;
            double y = 1.0 - poly * Math.Exp(-x * x);
            return (float)(sign * y);
        }

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        /// <summary>
        /// Drop the GGML backend's device-side copy of a tensor after writing
        /// through its raw host pointer, so the next kernel re-uploads.
        /// </summary>
        private void InvalidateTensorDeviceCache(Tensor tensor)
        {
            if (!_useNativeAttention || tensor == null)
                return;

            GgmlBasicOps.InvalidateHostBuffer(TensorComputePrimitives.GetStoragePointer(tensor));
        }

        /// <summary>
        /// Host write buffer for a tensor that may live in device memory. On the
        /// direct-CUDA backend, checking out a raw host pointer first copies the
        /// tensor's uninitialized device bytes back to the host; for a write-only
        /// fill that copy is pure waste, so the write is staged in pinned memory
        /// and pushed with a single bulk CopyToStorage. On host-resident
        /// allocators the tensor's own buffer is used directly and the GGML device
        /// cache is invalidated on release.
        /// </summary>
        private sealed unsafe class HostStaging : IDisposable
        {
            private readonly MuseGlimmerVisionEncoder _owner;
            private readonly Tensor _tensor;
            private readonly float[] _buffer;
            private GCHandle _handle;

            public float* Ptr { get; }

            public HostStaging(MuseGlimmerVisionEncoder owner, Tensor tensor, bool stageOnHost)
            {
                _owner = owner;
                _tensor = tensor;
                if (!stageOnHost)
                {
                    Ptr = TensorComputePrimitives.GetFloatPointer(tensor);
                    return;
                }

                _buffer = new float[tensor.ElementCount()];
                _handle = GCHandle.Alloc(_buffer, GCHandleType.Pinned);
                Ptr = (float*)_handle.AddrOfPinnedObject();
            }

            public void Dispose()
            {
                if (_buffer == null)
                {
                    _owner.InvalidateTensorDeviceCache(_tensor);
                    return;
                }

                _tensor.Storage.CopyToStorage(_tensor.StorageOffset, _handle.AddrOfPinnedObject(),
                    _tensor.ElementCount() * sizeof(float));
                _handle.Free();
            }
        }

        private static void Trace(string label, Tensor t)
        {
            if (!s_traceEnabled || t == null)
                return;

            using var contiguous = t.IsContiguous() ? null : Ops.NewContiguous(t);
            Tensor src = contiguous ?? t;
            int n = (int)src.ElementCount();
            float[] data = src.GetElementsAsFloat(n);
            double sum = 0, sumsq = 0;
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float v = data[i];
                sum += v;
                sumsq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            Console.WriteLine($"[mg-venc-trace] {label,-24} shape={string.Join("x", src.Sizes.ToArray())} " +
                $"sum={sum:F4} sumsq={sumsq:F4} min={min:F5} max={max:F5} " +
                $"first={data[0]:F6},{data[1]:F6},{data[2]:F6}");
        }

        public void Dispose()
        {
            foreach (var t in _positionEmbeddingCache.Values)
                t.Dispose();
            _positionEmbeddingCache.Clear();

            foreach (var t in _weights.Values)
                t.Dispose();
            _weights.Clear();

            // Drop the MLX device-side copies BEFORE the host buffers they may wrap
            // zero-copy go away (same ordering as ModelBase.Dispose).
            if (_mlxDirect && _allocator is MlxAllocator mlxAllocator)
            {
                foreach (var qw in _quantWeights.Values)
                {
                    if (qw != null)
                        MlxQuantizedOps.ReleaseQuantizedWeight(mlxAllocator, qw.CacheKey);
                }
            }

            foreach (var qw in _quantWeights.Values)
                qw?.Dispose();
            _quantWeights.Clear();

            _geometryCache.Clear();
        }
    }
}
