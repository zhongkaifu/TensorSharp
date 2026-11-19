// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>The projector rows of one still image or one video frame pair.</summary>
    public sealed class Lfm3VisionEmbedding
    {
        internal Lfm3VisionEmbedding(float[] rows, int width, int height, int rowWidth)
        {
            Rows = rows;
            Width = width;
            Height = height;
            RowWidth = rowWidth;
            RowCount = Lfm3ImageProcessor.TokenCount(width, height);
            if (rows.LongLength != (long)RowCount * rowWidth) throw new InvalidOperationException($"{rows.LongLength} values for {RowCount} rows of {rowWidth}");
        }

        /// <summary>[<see cref="RowCount"/>][<see cref="RowWidth"/>] (the language model's embedding width), raster order over
        /// the merged 32 x 32 blocks (x fastest): the order the rows take in the prompt.</summary>
        public float[] Rows { get; }

        /// <summary>The preprocessed frame width the rows come from (a multiple of 32).</summary>
        public int Width { get; }

        /// <summary>The preprocessed frame height (a multiple of 32).</summary>
        public int Height { get; }

        public int RowWidth { get; }

        public int RowCount { get; }

        /// <summary>Merged blocks across (<see cref="Width"/> / 32).</summary>
        public int GridWidth => Width / Lfm3ImageProcessor.Factor;

        /// <summary>Merged blocks down (<see cref="Height"/> / 32).</summary>
        public int GridHeight => Height / Lfm3ImageProcessor.Factor;

        /// <summary>The rows as a [rows, width] F32 tensor on <paramref name="allocator"/> (what <c>QueueMedia</c> takes).</summary>
        public Tensor ToTensor(IAllocator allocator)
        {
            var t = new Tensor(allocator, DType.Float32, RowCount, RowWidth);
            t.SetElementsAsFloat(Rows);
            return t;
        }
    }

    /// <summary>A clip's encoding: its plan, one embedding per frame pair (block g of <see cref="Layout"/>), and its span.</summary>
    public sealed class Lfm3VideoEncoding
    {
        internal Lfm3VideoEncoding(Lfm3VideoPlan plan, IReadOnlyList<Lfm3VisionEmbedding> groups)
        {
            Plan = plan;
            Groups = groups;
            Layout = plan.Layout();
        }

        public Lfm3VideoPlan Plan { get; }

        public IReadOnlyList<Lfm3VisionEmbedding> Groups { get; }

        public Lfm3VisionLayout Layout { get; }

        /// <summary>Every group's rows one after the other ([<see cref="Lfm3VisionLayout.RowCount"/>][width]).</summary>
        public float[] AllRows()
        {
            long total = 0;
            foreach (Lfm3VisionEmbedding e in Groups) total += e.Rows.LongLength;
            var all = new float[total];
            long at = 0;
            foreach (Lfm3VisionEmbedding e in Groups)
            {
                Array.Copy(e.Rows, 0, all, at, e.Rows.LongLength);
                at += e.Rows.LongLength;
            }
            return all;
        }
    }

    /// <summary>
    /// The LFM3-VL / LFM3-Omni vision tower (mmproj projector type <c>lfm3vl</c>) as one graph program per image or frame
    /// pair, on the executor of the model's allocator (GGML metal/cuda/cpu/vulkan, the managed CPU, MLX, direct CUDA;
    /// <see cref="NetworkGraph"/>). The network is llama.cpp's <c>tools/mtmd/models/lfm3vl.cpp</c>:
    /// <list type="number">
    /// <item>patchifier: two temporal 16 x 16 kernels (<c>v.patch_embd.weight</c> for the first frame of a pair,
    /// <c>.weight.1</c> for the second); a still image (and the odd last frame of a clip) fills both slices, so it is
    /// multiplied by their sum, a pair by the two side by side (one product each); then RMSNorm (<c>v.embedding_norm</c>);</item>
    /// <item>14 layers: RMSNorm, a mixer (a gated depthwise 3 x 3 conv over the patch grid where <c>conv2d_in_proj</c>
    /// exists, else 16/8-head GQA attention with per-head q/k RMSNorm and a 2-D rotary of base 1e6, bidirectional over the
    /// frame's patches), residual, RMSNorm, SwiGLU FFN, residual;</item>
    /// <item>projector: LayerNorm (eps 1e-6) per patch, 2 x 2 neighbours concatenated in (0,0), (0,1), (1,0), (1,1) order,
    /// Linear, exact (erf) GELU, Linear: one row of the language model's width per 32 x 32 block.</item>
    /// </list>
    /// Preprocessing is <see cref="Lfm3ImageProcessor"/>; prompt layout <see cref="Lfm3VisionLayout"/>; video sampling
    /// <see cref="Lfm3Video"/>. A call runs serialized on the instance (the caller holds the model's compute lock).
    /// </summary>
    public sealed class Lfm3VisionEncoder : IDisposable
    {
        /// <summary><c>&lt;|vision_start|&gt;</c> in the LFM3-Omni vocabulary.</summary>
        public const int VisionStartToken = 125009;

        /// <summary><c>&lt;|vision_end|&gt;</c>.</summary>
        public const int VisionEndToken = 125010;

        /// <summary><c>&lt;|image|&gt;</c>: the placeholder the chat template renders for an image (HF fills each row's slot with it).</summary>
        public const int ImageToken = 125030;

        /// <summary><c>&lt;|video|&gt;</c>: the template's placeholder for a whole video.</summary>
        public const int VideoToken = 125029;

        private const string StillKernel = "v.patch_embd.still";
        private const string PairKernel = "v.patch_embd.pair";

        // Bytes the managed CPU's and direct CUDA's attention may hold for one chunk of query rows' scores (their attention
        // materializes heads x rows x keys F32 scores; a 2048-token image's 8192 patches would take 4 GiB in one piece).
        private const long AttentionScoreBudget = 256L << 20;

        private readonly bool _ownsWeights;
        private readonly bool _materializesAttention;
        private bool _disposed;

        /// <summary>Load the vision tower of the mmproj at <paramref name="mmprojPath"/> onto <paramref name="allocator"/>
        /// (the model's; <paramref name="backend"/> is its backend, which decides the weights' formats).</summary>
        public Lfm3VisionEncoder(string mmprojPath, IAllocator allocator, BackendType backend)
            : this(OpenGguf(mmprojPath), allocator, backend, ownsFile: true)
        {
        }

        /// <summary>Load the vision tower of an opened mmproj (the caller keeps <paramref name="mmproj"/>).</summary>
        public Lfm3VisionEncoder(GgufFile mmproj, IAllocator allocator, BackendType backend)
            : this(mmproj, allocator, backend, ownsFile: false)
        {
        }

        /// <summary>As the public constructor, the programs run on <paramref name="executor"/> in place of the one registered
        /// for the allocator (a test's reference executor).</summary>
        internal Lfm3VisionEncoder(GgufFile mmproj, IAllocator allocator, BackendType backend, IGraphExecutor executor)
            : this(mmproj, allocator, backend, ownsFile: false, executor)
        {
        }

        private Lfm3VisionEncoder(GgufFile gguf, IAllocator allocator, BackendType backend, bool ownsFile, IGraphExecutor executor = null)
        {
            ArgumentNullException.ThrowIfNull(gguf);
            ArgumentNullException.ThrowIfNull(allocator);
            try
            {
                ReadHyperparameters(gguf);
                Weights = new NetworkWeights(allocator, backend);
                _ownsWeights = true;
                Weights.LoadGguf(gguf, n => n.StartsWith("v.", StringComparison.Ordinal)
                    || (n.StartsWith("mm.", StringComparison.Ordinal) && !n.StartsWith("mm.a.", StringComparison.Ordinal)));
                ValidateWeights();
                AddPatchKernels();
                // One program a frame size: a clip needs two (its pairs, an odd last frame), an image one, or up to MaxParts
                // when it is split for yields (YieldPatches). A cached GGML program keeps its own compute memory, which
                // grows with the frame, so few are kept (a miss recompiles: tens of ms against an encode of tens to
                // hundreds; about half a second for a 2048-row image).
                Graph = new NetworkGraph(Weights, executor, "lfm3 vision tower") { MaxCachedPlans = MaxParts };
                _materializesAttention = Graph.Executor is TensorGraphExecutor && Graph.Executor is not MlxGraphExecutor;
            }
            catch
            {
                Graph?.Dispose();
                Weights?.Dispose();
                throw;
            }
            finally
            {
                if (ownsFile) gguf.Dispose();
            }
        }

        private static GgufFile OpenGguf(string path)
        {
            ArgumentException.ThrowIfNullOrEmpty(path);
            return new GgufFile(path);
        }

        /// <summary>The tower's weights (v.*, mm.* without mm.a.*, plus the two derived patch kernels).</summary>
        public NetworkWeights Weights { get; }

        /// <summary>The graph runner (its <see cref="NetworkGraph.MaxCachedPlans"/> bounds compiled programs: one a frame
        /// size).</summary>
        public NetworkGraph Graph { get; }

        /// <summary>Tower width (<c>clip.vision.embedding_length</c>).</summary>
        public int EmbeddingLength { get; private set; }

        public int BlockCount { get; private set; }
        public int HeadCount { get; private set; }
        public int KvHeadCount { get; private set; }
        public int HeadDim { get; private set; }
        public int FeedForwardLength { get; private set; }

        /// <summary>RMSNorm epsilon of the tower (<c>clip.vision.attention.layer_norm_epsilon</c>).</summary>
        public float NormEps { get; private set; }

        /// <summary>The 2-D rotary's base: mtmd hardcodes 1e6 (the GGUF carries none, and the conversion refuses others).</summary>
        public float RopeTheta { get; } = 1e6f;

        /// <summary>The projector's LayerNorm epsilon: mtmd hardcodes 1e-6 (the conversion refuses others).</summary>
        public float ProjectorNormEps { get; } = 1e-6f;

        /// <summary>Projector row width (<c>clip.vision.projection_dim</c>): the language model's embedding width.</summary>
        public int ProjectionDim { get; private set; }

        /// <summary>Image pixel bounds from the mmproj (<c>clip.vision.image_min_pixels</c> / <c>image_max_pixels</c>; the
        /// maximum is capped at 4096 tokens as mtmd caps it).</summary>
        public long ImageMinPixels { get; set; }

        /// <summary>See <see cref="ImageMinPixels"/>.</summary>
        public long ImageMaxPixels { get; set; }

        /// <summary>Most frames a video samples: 384 when the mmproj also carries the audio tower (LFM3-Omni hears the
        /// soundtrack), else 768 (LFM3-VL), as mtmd picks its video template.</summary>
        public int MaxVideoFrames { get; private set; }

        /// <summary>
        /// Query rows per attention piece on the executors whose attention holds every score at once (the managed CPU and
        /// direct CUDA): 0 sizes the pieces to keep each piece's scores under 256 MiB; a positive value forces that size
        /// (tests). GGML (flash attention) and MLX (fused attention) run one piece.
        /// </summary>
        public int AttentionChunkRows { get; set; }

        /// <summary>
        /// Patches of work per program when the caller can yield between programs (a <c>betweenParts</c> callback: the
        /// model's GPU-lock yield): a larger frame runs the tower as several programs split at layer boundaries,
        /// ceil(patches / this) of them (at most <see cref="MaxParts"/>), the activations passing between them in F32, so
        /// concurrent requests step between the parts of a large image instead of waiting for all of it. 0 never splits;
        /// without a callback a frame is one program. Default 4096: a 2048-row image (about 8000 patches) runs as two
        /// programs. Measured on ggml-metal (1920 x 1056, interleaved, medians of 7 under load): one program 891 ms, two
        /// 893 ms, four 945 ms, the rows bit-identical; on MLX, which compiles every call, 679 / 720 / 752 ms and the rows
        /// within 1.4e-3 / 2.5e-3 relative (its fusion differs across the boundary).
        /// </summary>
        public int YieldPatches { get; set; } = 4096;

        /// <summary>Most programs a split frame runs as (<see cref="YieldPatches"/>).</summary>
        public const int MaxParts = 4;

        private void ReadHyperparameters(GgufFile gguf)
        {
            string projector = gguf.GetString("clip.vision.projector_type");
            if (!string.Equals(projector, "lfm3vl", StringComparison.Ordinal))
                throw new NotSupportedException($"lfm3 vision: the mmproj's vision projector is '{projector ?? "(none)"}', not 'lfm3vl'");
            if (!gguf.GetBool("clip.has_vision_encoder"))
                throw new NotSupportedException("lfm3 vision: the mmproj has no vision encoder (clip.has_vision_encoder)");
            int patch = (int)gguf.GetUint32("clip.vision.patch_size");
            int merge = (int)gguf.GetUint32("clip.vision.spatial_merge_size");
            if (patch != Lfm3ImageProcessor.PatchSize || merge != Lfm3ImageProcessor.MergeSize)
                throw new NotSupportedException($"lfm3 vision: patch {patch} / merge {merge}; this port runs patch {Lfm3ImageProcessor.PatchSize}, merge {Lfm3ImageProcessor.MergeSize} (as llama.cpp, which supports merge 2 only)");
            EmbeddingLength = (int)gguf.GetUint32("clip.vision.embedding_length");
            BlockCount = (int)gguf.GetUint32("clip.vision.block_count");
            HeadCount = (int)gguf.GetUint32("clip.vision.attention.head_count");
            KvHeadCount = (int)gguf.GetUint32("clip.vision.attention.head_count_kv", (uint)HeadCount);
            FeedForwardLength = (int)gguf.GetUint32("clip.vision.feed_forward_length");
            NormEps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
            ProjectionDim = (int)gguf.GetUint32("clip.vision.projection_dim");
            if (EmbeddingLength < 1 || BlockCount < 1 || HeadCount < 1 || KvHeadCount < 1 || HeadCount % KvHeadCount != 0
                || EmbeddingLength % HeadCount != 0 || ProjectionDim < 1)
                throw new NotSupportedException($"lfm3 vision: width {EmbeddingLength}, {BlockCount} layers, {HeadCount}/{KvHeadCount} heads, projection {ProjectionDim}");
            HeadDim = EmbeddingLength / HeadCount;
            if (HeadDim % 4 != 0) throw new NotSupportedException($"lfm3 vision: a {HeadDim}-wide head (the 2-D rotary splits a head in quarters)");
            if (!gguf.GetBool("clip.use_silu", true))
                throw new NotSupportedException("lfm3 vision: the tower's FFN is not SwiGLU (clip.use_silu = false)");
            float[] mean = gguf.GetFloatArray("clip.vision.image_mean"), std = gguf.GetFloatArray("clip.vision.image_std");
            if (mean != null && Array.Exists(mean, v => v != 0.5f) || std != null && Array.Exists(std, v => v != 0.5f))
                throw new NotSupportedException($"lfm3 vision: image mean [{string.Join(", ", mean ?? Array.Empty<float>())}] / std [{string.Join(", ", std ?? Array.Empty<float>())}]; this port normalizes with 0.5 / 0.5");
            string resize = gguf.GetString("clip.vision.image_resize_algo", "bicubic");
            if (!string.Equals(resize, "bicubic", StringComparison.Ordinal))
                throw new NotSupportedException($"lfm3 vision: the mmproj resizes with '{resize}'; this port resizes bicubic (torchvision)");
            const long patchArea = Lfm3ImageProcessor.Factor * Lfm3ImageProcessor.Factor;
            ImageMinPixels = gguf.GetUint32("clip.vision.image_min_pixels", (uint)Lfm3ImageProcessor.ImageMinPixels);
            ImageMaxPixels = Math.Min(4096 * patchArea, (long)gguf.GetUint32("clip.vision.image_max_pixels", (uint)Lfm3ImageProcessor.ImageMaxPixels));
            MaxVideoFrames = gguf.GetBool("clip.has_audio_encoder") ? Lfm3Video.MaxFramesOmni : Lfm3Video.MaxFramesVl;
        }

        private void ValidateWeights()
        {
            string[] global = { "v.patch_embd.weight", "v.patch_embd.weight.1", "v.embedding_norm.weight", "mm.input_norm.weight",
                "mm.input_norm.bias", "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias" };
            foreach (string name in global)
                if (!Weights.Contains(name))
                    throw new NotSupportedException($"lfm3 vision: the mmproj lacks '{name}' (a single temporal kernel or another projector is not this port's)");
            string[] common = { "ln1.weight", "ln2.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight" };
            string[] conv = { "conv2d_in_proj.weight", "conv2d_spatial.weight", "conv2d_out_proj.weight" };
            string[] attn = { "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_out.weight", "attn_q_norm.weight", "attn_k_norm.weight" };
            for (int l = 0; l < BlockCount; l++)
            {
                string p = $"v.blk.{l}.";
                bool isConv = Weights.Contains(p + "conv2d_in_proj.weight");
                foreach (string n in common) Require(p + n, l);
                foreach (string n in isConv ? conv : attn) Require(p + n, l);
            }
            if (Weights.Get("mm.2.bias").Elements != ProjectionDim)
                throw new NotSupportedException($"lfm3 vision: mm.2.bias has {Weights.Get("mm.2.bias").Elements} values, projection_dim is {ProjectionDim}");

            void Require(string name, int layer)
            {
                if (!Weights.Contains(name))
                    throw new NotSupportedException($"lfm3 vision: layer {layer} is neither a complete conv block nor a complete attention block (no '{name}')");
            }
        }

        /// <summary>The still kernel W0 + W1 [768, C] and the pair kernel [W0 | W1] [1536, C] (F32; 9 MiB at C = 1024).</summary>
        private void AddPatchKernels()
        {
            float[] w0 = Weights.ReadF32("v.patch_embd.weight"), w1 = Weights.ReadF32("v.patch_embd.weight.1");
            const int k = Lfm3ImageProcessor.PatchValues;
            if (w0.Length != k * EmbeddingLength || w1.Length != w0.Length)
                throw new NotSupportedException($"lfm3 vision: patch kernels of {w0.Length} / {w1.Length} values, not {k} x {EmbeddingLength}");
            var still = new float[w0.Length];
            var pair = new float[2 * w0.Length];
            for (int o = 0; o < EmbeddingLength; o++)
            {
                int src = o * k;
                for (int i = 0; i < k; i++) still[src + i] = w0[src + i] + w1[src + i];
                Array.Copy(w0, src, pair, 2 * src, k);
                Array.Copy(w1, src, pair, 2 * src + k, k);
            }
            Weights.AddF32(StillKernel, still, k, EmbeddingLength);
            Weights.AddF32(PairKernel, pair, 2 * k, EmbeddingLength);
        }

        // ================================================================ entry points

        /// <summary>The preprocessed size of a still image of <paramref name="width"/> x <paramref name="height"/>.</summary>
        public (int Width, int Height) ImageSize(int width, int height)
            => Lfm3ImageProcessor.SmartResize(width, height, ImageMinPixels, ImageMaxPixels);

        /// <summary>Embedding rows a still image of <paramref name="width"/> x <paramref name="height"/> takes.</summary>
        public int ImageTokenCount(int width, int height)
        {
            (int w, int h) = ImageSize(width, height);
            return Lfm3ImageProcessor.TokenCount(w, h);
        }

        /// <summary>Decode, preprocess and encode the image file at <paramref name="path"/>.</summary>
        public Lfm3VisionEmbedding EncodeImage(string path) => EncodeImage(Lfm3ImageProcessor.LoadImage(path));

        /// <summary>Preprocess (<see cref="ImageSize"/>, torchvision bicubic) and encode a still image.</summary>
        public Lfm3VisionEmbedding EncodeImage(Lfm3RgbImage image, Action betweenParts = null)
        {
            ArgumentNullException.ThrowIfNull(image);
            (int w, int h) = ImageSize(image.Width, image.Height);
            return EncodeFrame(Lfm3ImageProcessor.Resize(image, w, h), betweenParts);
        }

        /// <summary>Encode one preprocessed frame (sides multiples of 32) through both temporal kernels: a still image, or the
        /// odd last frame of a clip (HF repeats it to fill the pair). <paramref name="betweenParts"/> runs between the
        /// programs of a frame split for yields (<see cref="YieldPatches"/>).</summary>
        public Lfm3VisionEmbedding EncodeFrame(Lfm3RgbImage frame, Action betweenParts = null)
        {
            ArgumentNullException.ThrowIfNull(frame);
            CheckFrame(frame);
            return Run(Lfm3ImageProcessor.Patchify(frame), Lfm3ImageProcessor.PatchValues, StillKernel, frame.Width, frame.Height, betweenParts);
        }

        /// <summary>Encode two preprocessed frames of one size as one temporal patch (the first through
        /// <c>v.patch_embd.weight</c>, the second through <c>.weight.1</c>). <paramref name="second"/> null, or the same
        /// frame, runs <see cref="EncodeFrame"/> (identical rows, one product fewer).</summary>
        public Lfm3VisionEmbedding EncodePair(Lfm3RgbImage first, Lfm3RgbImage second, Action betweenParts = null)
        {
            ArgumentNullException.ThrowIfNull(first);
            if (second == null || ReferenceEquals(first, second)) return EncodeFrame(first, betweenParts);
            CheckFrame(first);
            if (second.Width != first.Width || second.Height != first.Height)
                throw new ArgumentException($"a frame pair of {first.Width} x {first.Height} and {second.Width} x {second.Height}");
            const int k = Lfm3ImageProcessor.PatchValues;
            int n = first.Width / Lfm3ImageProcessor.PatchSize * (first.Height / Lfm3ImageProcessor.PatchSize);
            var patches = new float[(long)n * 2 * k];
            Lfm3ImageProcessor.Patchify(first, patches, 2 * k, 0);
            Lfm3ImageProcessor.Patchify(second, patches, 2 * k, k);
            return Run(patches, 2 * k, PairKernel, first.Width, first.Height, betweenParts);
        }

        /// <summary>Sample (<see cref="Lfm3Video"/>), decode with <paramref name="decoder"/> (null:
        /// <see cref="TensorSharp.Models.Media.MediaCodecs.Video"/>), resize and encode the clip at <paramref name="path"/>,
        /// one program run a frame pair; <paramref name="betweenGroups"/> runs between pairs (a caller holding the model's
        /// compute lock yields it there).</summary>
        public Lfm3VideoEncoding EncodeVideo(string path, TensorSharp.Models.Media.IVideoDecoder decoder = null, Action betweenGroups = null)
        {
            (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(path, MaxVideoFrames, decoder);
            return EncodeVideo(plan, frames, betweenGroups);
        }

        /// <summary>Encode the groups of <paramref name="plan"/> over <paramref name="frames"/> (one per
        /// <see cref="Lfm3VideoPlan.FrameIndices"/> entry; frames of another size are resized to the plan's first).</summary>
        public Lfm3VideoEncoding EncodeVideo(Lfm3VideoPlan plan, IReadOnlyList<Lfm3RgbImage> frames, Action betweenGroups = null)
        {
            ArgumentNullException.ThrowIfNull(plan);
            ArgumentNullException.ThrowIfNull(frames);
            if (frames.Count != plan.FrameIndices.Length)
                throw new ArgumentException($"{frames.Count} frames for a plan of {plan.FrameIndices.Length}");
            var groups = new Lfm3VisionEmbedding[plan.Groups.Count];
            for (int g = 0; g < groups.Length; g++)
            {
                if (g > 0) betweenGroups?.Invoke();
                Lfm3VideoGroup group = plan.Groups[g];
                Lfm3RgbImage first = Lfm3ImageProcessor.Resize(frames[group.First], plan.Width, plan.Height);
                groups[g] = group.Second == group.First
                    ? EncodeFrame(first, betweenGroups)
                    : EncodePair(first, Lfm3ImageProcessor.Resize(frames[group.Second], plan.Width, plan.Height), betweenGroups);
            }
            return new Lfm3VideoEncoding(plan, groups);
        }

        private static void CheckFrame(Lfm3RgbImage frame)
        {
            if (frame.Width % Lfm3ImageProcessor.Factor != 0 || frame.Height % Lfm3ImageProcessor.Factor != 0)
                throw new ArgumentException($"a preprocessed frame's sides are multiples of {Lfm3ImageProcessor.Factor}, not {frame.Width} x {frame.Height}");
        }

        /// <summary>Encode patch rows [<paramref name="patchWidth"/>, patches] (raster order) of a <paramref name="width"/> x
        /// <paramref name="height"/> frame with patch kernel <paramref name="kernel"/>: one program, or with
        /// <paramref name="betweenParts"/> and more than <see cref="YieldPatches"/> patches, consecutive layer ranges as
        /// programs of their own with <paramref name="betweenParts"/> between them.</summary>
        private Lfm3VisionEmbedding Run(float[] patches, int patchWidth, string kernel, int width, int height, Action betweenParts)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int gw = width / Lfm3ImageProcessor.PatchSize, gh = height / Lfm3ImageProcessor.PatchSize;
            int n = gw * gh;
            int parts = PartsFor(n, betweenParts != null);
            float[] rows;
            if (parts == 1)
            {
                rows = Graph.Run(b => b.Output(EmitTower(b, b.Input(patches, patchWidth, (long)n), kernel, gw, gh)))[0];
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
                        ? EmitTower(b, b.Input(patches, patchWidth, (long)n), kernel, gw, gh, first, last, projector: false)
                        : EmitTower(b, b.Input(input, EmbeddingLength, (long)n), null, gw, gh, first, last, projector: last == BlockCount)))[0];
                }
                rows = x;
            }
            return new Lfm3VisionEmbedding((float[])rows.Clone(), width, height, ProjectionDim);
        }

        /// <summary>Programs a frame of <paramref name="patches"/> patches runs as (<see cref="YieldPatches"/>).</summary>
        internal int PartsFor(int patches, bool canYield)
            => !canYield || YieldPatches <= 0 || patches <= YieldPatches
                ? 1
                : Math.Min(Math.Min(MaxParts, BlockCount), (patches + YieldPatches - 1) / YieldPatches);

        // ================================================================ the network

        /// <summary>The tower and projector over patch rows <paramref name="patches"/> of a <paramref name="gw"/> x
        /// <paramref name="gh"/> patch grid: [<see cref="ProjectionDim"/>, gw * gh / 4].</summary>
        internal NetValue EmitTower(NetworkBuilder b, NetValue patches, string kernel, int gw, int gh)
            => EmitTower(b, patches, kernel, gw, gh, 0, BlockCount, projector: true);

        /// <summary>Layers [<paramref name="firstLayer"/>, <paramref name="lastLayer"/>) of the tower: from the patch rows
        /// (<paramref name="kernel"/> set, the patchifier and its norm first) or from the activations [width, patches] the
        /// previous part left (<paramref name="kernel"/> null), with the projector after the last layer when
        /// <paramref name="projector"/>.</summary>
        internal NetValue EmitTower(NetworkBuilder b, NetValue input, string kernel, int gw, int gh, int firstLayer, int lastLayer,
            bool projector)
        {
            int c = EmbeddingLength, n = gw * gh;
            float eps = NormEps;
            NetValue x = kernel != null ? b.RmsNorm(b.Linear(input, kernel), "v.embedding_norm.weight", eps) : input;
            NetValue positions = default;
            for (int l = firstLayer; l < lastLayer; l++)
            {
                string p = $"v.blk.{l}.";
                NetValue h = b.RmsNorm(x, p + "ln1.weight", eps);
                NetValue mixed;
                if (Weights.Contains(p + "conv2d_in_proj.weight"))
                {
                    // in_proj thirds B | C | x: conv(B * x) over the patch grid (zero borders), times C, out_proj.
                    NetValue proj = b.Linear(h, p + "conv2d_in_proj.weight");
                    NetValue gated = b.Reshape(b.Mul(b.Columns(proj, 0, c), b.Columns(proj, 2 * c, c)), c, gw, gh);
                    NetValue conv = b.DepthwiseConv2d(gated, b.Weight(p + "conv2d_spatial.weight"), 1, 1, 1, 1);
                    mixed = b.Linear(b.Mul(b.Columns(proj, c, c), b.Reshape(conv, c, n)), p + "conv2d_out_proj.weight");
                }
                else
                {
                    if (!positions.Valid) positions = b.RasterPositions(gw, gh);
                    mixed = Attention(b, h, p, positions, n);
                }
                x = b.Add(x, mixed);
                x = b.Add(x, b.FeedForward(b.RmsNorm(x, p + "ln2.weight", eps), p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"));
            }
            if (!projector) return x;
            // Projector: LayerNorm per patch, then each 2 x 2 block's patches side by side in (0,0), (0,1), (1,0), (1,1) order.
            NetValue e = b.LayerNorm(x, "mm.input_norm.weight", "mm.input_norm.bias", ProjectorNormEps);
            NetValue merged = b.Reshape(b.Permute(b.Reshape(e, 2 * c, gw / 2, 2, gh / 2), 0, 2, 1, 3), 4 * c, n / 4);
            return b.GeluMlp(merged, "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
        }

        /// <summary>Bidirectional GQA over the frame's patches: q/k RMSNorm per head, then the 2-D rotary (y on the first
        /// quarter of pairs, x on the second), scale 1/sqrt(d).</summary>
        private NetValue Attention(NetworkBuilder b, NetValue h, string p, NetValue positions, int n)
        {
            int d = HeadDim;
            NetValue q = b.Reshape(b.Linear(h, p + "attn_q.weight"), d, HeadCount, n);
            NetValue k = b.Reshape(b.Linear(h, p + "attn_k.weight"), d, KvHeadCount, n);
            NetValue v = b.Reshape(b.Linear(h, p + "attn_v.weight"), d, KvHeadCount, n);
            q = b.Rope2D(b.RmsNorm(q, p + "attn_q_norm.weight", NormEps), positions, RopeTheta);
            k = b.Rope2D(b.RmsNorm(k, p + "attn_k_norm.weight", NormEps), positions, RopeTheta);
            float scale = 1f / MathF.Sqrt(d);
            int chunk = QueryChunk(n);
            NetValue attended;
            if (chunk >= n)
            {
                attended = b.Attention(q, k, v, scale);
            }
            else
            {
                attended = default;
                for (int start = 0; start < n; start += chunk)
                {
                    NetValue piece = b.Attention(b.Slice(q, 2, start, Math.Min(chunk, n - start)), k, v, scale);
                    attended = attended.Valid ? b.Concat(attended, piece, 1) : piece;
                }
            }
            return b.Linear(attended, p + "attn_out.weight");
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
            Graph?.Dispose();
            if (_ownsWeights) Weights?.Dispose();
        }
    }
}
