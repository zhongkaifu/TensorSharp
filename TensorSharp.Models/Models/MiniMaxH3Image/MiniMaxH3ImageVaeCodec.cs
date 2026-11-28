// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's VAE on the diffusion runtime (minimax_h3_image_vae_fp16.safetensors). Its encoder (editing's references)
// is MiniMaxH3ImageVaeCodec.Encoder.cs. The decoder: the latent is cut into
// ComfyUI's 256-pixel tiles, each row of tiles goes through the 2-D ViT decoder as one GraphProgram (the tiles packed along
// the rows, each with its own register and zero tokens), and the decoded tiles are blended into the canvas row by row. The
// program is the graph h3_vit_decode builds, node for node: ggml_cuda serves its regions as that graph's fused nodes (F16
// products on cuBLAS, one cuDNN attention over the row of tiles, the scaled residuals), and every other backend runs their
// ops, which are its unfused nodes. Its native graph is sized from the program, so any batch of tiles decodes on every
// backend (the native batched decode sizes a graph for the fused nodes only and aborts elsewhere past four tiles). On direct
// CUDA the same program runs on the CUDA tensor executor (its projections bound as weights, its F32 leaves as tensors on the
// device), which runs the regions' ops; the managed executor binds it the same way in the tests.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    /// <summary>
    /// The MiniMax-H3-Image VAE as an <see cref="IVaeCodec"/>: the tiled encoder of edits' references and the tiled decoder.
    /// Making one maps the file and reads the decoder's small tensors on the host; its first decode makes the decoder's weights
    /// resident. The encoder's weights are made on the host when an edit first needs them (<see cref="EnsureEncoder"/>).
    /// </summary>
    public sealed partial class MiniMaxH3ImageVaeCodec : IVaeCodec
    {
        public const int LatentChannels = 24;
        public const int Downsample = 16;
        public const int NumRegisterTokens = 4;
        public const float RopeTheta = 100f;
        public const float Eps = 1e-5f;
        public const int OutChannels = 3;
        public const int TileSize = 256;
        public const int TileOverlapMin = 64;
        public const int HeadDim = 64;

        /// <summary>Tiles one decoder program takes at most: a row of a 16 MP canvas is 21 to 33 tiles, about 15 MB of
        /// activations each (the legacy decoder's default, which TS_H3_IMAGE_VAE_BATCH overrode before stage S10 retired it).</summary>
        public const int MaxBatch = 64;

        /// <summary>Plans kept per codec: a canvas decodes with one tile shape and at most two batch sizes.</summary>
        private const int MaxPlans = 2;

        /// <summary>The H3 latent statistics (comfy/ldm/minimax/vae.py LATENTS_MEAN/STD), which the image VAE file does not
        /// carry.</summary>
        internal static readonly float[] ReleaseLatentsMean =
        {
            0.858090341091156f, -0.9606591463088989f, 1.0661640167236328f, -0.5090325474739075f,
            -0.2727581858634949f, -1.3675414323806763f, -0.2553254961967468f, -0.26907554268836975f,
            -0.5376840829849243f, -0.0464097298681736f, 0.6657370328903198f, 0.19690127670764923f,
            -0.5460608005523682f, -0.4035342037677765f, -0.23683024942874908f, 0.25928452610969543f,
            -0.30133944749832153f, 0.211341992020607f, -1.1206848621368408f, 0.3581933379173279f,
            -0.04225143790245056f, 0.2604829967021942f, 0.22864092886447906f, 0.7056031823158264f,
        };
        internal static readonly float[] ReleaseLatentsStd =
        {
            1.2223774194717407f, 1.2767263650894165f, 1.68317747116088865f, 1.7549455165863037f,
            1.5636216402053833f, 2.194143533706665f, 0.96531379222869875f, 1.05698859691619875f,
            0.841948926448822f, 0.7729952931404114f, 1.8955937623977661f, 0.946841835975647f,
            0.7996809482574463f, 0.44988900423049925f, 0.7197399735450745f, 0.69362932443618775f,
            2.961095094680786f, 2.7694199085235595f, 3.0496184825897215f, 2.1088054180145265f,
            3.276226282119751f, 3.1627357006073f, 2.28168129920959475f, 2.6127843856811525f,
        };

        private sealed record Linear(MiniMaxH3ImageWeight Weight, MiniMaxH3ImageWeight Bias);

        private sealed class Block
        {
            public MiniMaxH3ImageWeight Norm1, Norm2, Scale1, Scale2;
            public Linear Qkv, Out, W1, W2;
        }

        private readonly SafetensorsModel _model;
        private readonly IGraphExecutor _executor;
        // A tensor executor's device (direct CUDA; the managed one in tests), whose F32 leaves are tensors there; null on GGML,
        // whose executors bind every leaf by host pointer.
        private readonly IAllocator _tensorAllocator;
        private readonly Dictionary<MiniMaxH3ImageWeight, Tensor> _tensors = new();
        private readonly HashSet<MiniMaxH3ImageWeight> _hostValues = new();   // the pinned F32 leaves (Vec): tensors, never products
        private readonly List<GCHandle> _pins = new();
        private readonly List<MiniMaxH3ImageWeight> _all = new();
        private readonly Block[] _blocks;
        private readonly Linear _postQuant, _xEmbedder, _projOut;
        private readonly MiniMaxH3ImageWeight _registerTokens, _normOutW, _normOutB;
        private readonly object _gate = new();
        private readonly List<(int Batch, int H, int W, (ulong, ulong) Signature, IGraphPlan Plan)> _plans = new();
        private (int H, int W, float[] Cos, float[] Sin) _rope;
        private bool _preloaded;
        private bool _disposed;

        /// <param name="path">The VAE file (Comfy-Org/MiniMax-H3-Image vae/minimax_h3_image_vae_fp16.safetensors).</param>
        /// <param name="backend">A GGML backend (its graph executor runs the decoder) or <see cref="BackendType.Cuda"/> (the CUDA
        /// tensor executor on <paramref name="allocator"/>'s device).</param>
        /// <param name="allocator">The device on <see cref="BackendType.Cuda"/> (a <see cref="TensorSharp.Cuda.CudaAllocator"/>).</param>
        public MiniMaxH3ImageVaeCodec(string path, BackendType backend, IAllocator allocator = null)
            : this(path, backend switch
            {
                BackendType.GgmlCuda => GgmlGraphExecutor.For(GgmlBackendType.Cuda),
                BackendType.GgmlMetal => GgmlGraphExecutor.For(GgmlBackendType.Metal),
                BackendType.GgmlCpu => GgmlGraphExecutor.For(GgmlBackendType.Cpu),
                BackendType.Cuda => allocator is TensorSharp.Cuda.CudaAllocator cuda
                    ? GraphExecutors.For(cuda)
                    : throw new ArgumentException("MiniMax-H3-Image's VAE on --backend cuda runs on a CUDA device's allocator.", nameof(allocator)),
                _ => throw new NotSupportedException(
                    $"MiniMax-H3-Image's VAE runs on ggml_cuda, ggml_metal, ggml_cpu or cuda, not {Architecture.BackendNames.OptionName(backend)}."),
            }, backend == BackendType.Cuda ? allocator : null)
        {
        }

        /// <param name="executor">The decoder's graph executor: a GGML one (binds every leaf by host pointer) or a tensor
        /// executor on <paramref name="tensorAllocator"/>'s device (its projections bound as weights, its F32 leaves as tensors
        /// there).</param>
        internal MiniMaxH3ImageVaeCodec(string path, IGraphExecutor executor, IAllocator tensorAllocator = null)
        {
            ArgumentNullException.ThrowIfNull(path);
            _executor = executor ?? throw new ArgumentNullException(nameof(executor));
            if (!executor.Capabilities.BindsHostPointers && tensorAllocator == null)
                throw new NotSupportedException("MiniMax-H3-Image's VAE binds its weights by host pointer; this executor does not.");
            _tensorAllocator = executor.Capabilities.BindsHostPointers ? null : tensorAllocator;
            _model = SafetensorsModel.Open(path);
            try
            {
                int n = 0;
                while (_model.HasTensor($"decoder.transformer_blocks.{n}.norm1.weight")) n++;
                if (n == 0)
                    throw new InvalidOperationException($"'{path}' has no MiniMax-H3-Image VAE decoder blocks.");
                long[] xe = _model.TensorShape("decoder.x_embedder.weight");
                Dim = checked((int)xe[0]);
                if (xe[1] != LatentChannels || Dim % HeadDim != 0)
                    throw new InvalidOperationException($"'{path}' is not a MiniMax-H3-Image VAE (x_embedder {xe[0]}x{xe[1]}).");
                Heads = Dim / HeadDim;
                FfnInner = checked((int)_model.TensorShape("decoder.transformer_blocks.0.ff.w2.weight")[1]);
                RotDim = HeadDim / 2;

                _postQuant = Lin("post_quant_conv.weight", "post_quant_conv.bias");
                _xEmbedder = Lin("decoder.x_embedder.weight", "decoder.x_embedder.bias");
                _projOut = Lin("decoder.proj_out.weight", "decoder.proj_out.bias");
                if (_projOut.Weight.Ne1 != PatchDim)
                    throw new InvalidOperationException($"'{path}' decodes {_projOut.Weight.Ne1} values a token, not {PatchDim} (16x16 RGB).");
                _registerTokens = Vec("decoder.register_tokens", Dim);
                _normOutW = Vec("decoder.norm_out.weight")
                    ?? throw new InvalidOperationException($"'{path}' has no decoder.norm_out.weight.");
                _normOutB = Vec("decoder.norm_out.bias");
                _blocks = new Block[n];
                for (int i = 0; i < n; i++)
                {
                    string p = $"decoder.transformer_blocks.{i}.";
                    _blocks[i] = new Block
                    {
                        Norm1 = Vec(p + "norm1.weight"),
                        Norm2 = Vec(p + "norm2.weight"),
                        Scale1 = Vec(p + "scale1"),
                        Scale2 = Vec(p + "scale2"),
                        Qkv = Lin(p + "attn.to_qkv.weight", p + "attn.to_qkv.bias"),
                        Out = Lin(p + "attn.to_out.weight", p + "attn.to_out.bias"),
                        W1 = Lin(p + "ff.w1.weight", p + "ff.w1.bias"),
                        W2 = Lin(p + "ff.w2.weight", p + "ff.w2.bias"),
                    };
                    if (_blocks[i].Norm1 == null || _blocks[i].Norm2 == null)
                        throw new InvalidOperationException($"'{path}' block {i} has no norm weights.");
                    if (_blocks[i].W1.Weight.Ne1 != 2L * FfnInner)
                        throw new InvalidOperationException($"'{path}' block {i}'s ff.w1 emits {_blocks[i].W1.Weight.Ne1} values, not 2 x {FfnInner}.");
                }
                LatentsMean = _model.HasTensor("latents_mean") ? _model.ReadFloat32("latents_mean") : ReleaseLatentsMean;
                LatentsStd = _model.HasTensor("latents_std") ? _model.ReadFloat32("latents_std") : ReleaseLatentsStd;
            }
            catch
            {
                DisposeHost();
                throw;
            }
        }

        public int Dim { get; }
        public int Heads { get; }
        public int FfnInner { get; }
        /// <summary>Rotated dims per head: 2 axes x 8 frequencies, duplicated (rope_dim_ratio 0.5).</summary>
        public int RotDim { get; }
        public int PatchDim => OutChannels * Downsample * Downsample;
        public int NumBlocks => _blocks.Length;
        public float[] LatentsMean { get; }
        public float[] LatentsStd { get; }

        /// <summary>Seconds of the last <see cref="Decode"/> spent running the decoder programs.</summary>
        public double LastNativeSeconds { get; private set; }

        /// <summary>Tiles the last <see cref="Decode"/> decoded.</summary>
        public int LastTileCount { get; private set; }

        /// <summary>Programs compiled (the rest replayed a cached plan; a test seam).</summary>
        internal int Compiles { get; private set; }

        /// <summary>The plan the last decode call ran (a test seam: its served regions).</summary>
        internal IGraphPlan LastPlan { get; private set; }

        /// <summary>The nodes of the last program compiled (a test seam: its native graph is sized from them).</summary>
        internal int LastProgramNodes { get; private set; }

        /// <summary>Tiles a program takes at most (a test seam; <see cref="MaxBatch"/> otherwise).</summary>
        internal int BatchLimit { get; set; } = MaxBatch;

        // ---- weights

        private Linear Lin(string weight, string bias)
        {
            if (!_model.TensorOwners.TryGetValue(weight, out SafetensorsFile owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            SafetensorTensorInfo info = owner.GetInfo(weight);
            for (int i = 2; i < info.Shape.Length; i++)
                if (info.Shape[i] != 1)
                    throw new InvalidOperationException($"'{weight}' is not a linear or 1x1 convolution.");
            if (!owner.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            int format = info.Dtype switch
            {
                SafetensorDtype.F32 => GraphFormat.F32,
                SafetensorDtype.F16 => GraphFormat.F16,
                SafetensorDtype.BF16 => GraphFormat.BF16,
                _ => throw new NotSupportedException($"'{weight}' is {info.Dtype}; the VAE reads F32, F16 or BF16."),
            };
            // [out, in] in the file is ggml's [in, out].
            var w = new MiniMaxH3ImageWeight(format, info.Shape[1], info.Shape[0], ptr, info.ByteCount);
            _all.Add(w);
            return new Linear(w, _model.HasTensor(bias) ? Vec(bias) : null);
        }

        /// <summary>A small tensor as a pinned F32 leaf (the legacy decoder's PinF32 values), null when the file has none:
        /// a vector, or with <paramref name="rows"/> a [rows, count / rows] matrix.</summary>
        private MiniMaxH3ImageWeight Vec(string name, long rows = 0)
        {
            if (!_model.HasTensor(name)) return null;
            float[] values = _model.ReadFloat32(name);
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            var w = rows > 0
                ? new MiniMaxH3ImageWeight(GraphFormat.F32, rows, values.LongLength / rows, h.AddrOfPinnedObject(), values.LongLength * sizeof(float))
                : new MiniMaxH3ImageWeight(GraphFormat.F32, values.LongLength, 1, h.AddrOfPinnedObject(), values.LongLength * sizeof(float), Vector: true);
            _all.Add(w);
            _hostValues.Add(w);
            return w;
        }

        /// <summary>
        /// A decoder leaf in the program: on GGML its host bytes; on a tensor executor a projection as itself (the executor
        /// uploads and caches it by its address) and a pinned F32 leaf (a bias, a norm, a scale, the register tokens: values the
        /// program adds, multiplies or concatenates, never a product's weight) as a tensor on the device, made once.
        /// </summary>
        private int LeafOf(GraphEmitter e, MiniMaxH3ImageWeight w)
        {
            ReadOnlySpan<long> ne = w.Vector ? stackalloc long[] { w.Ne0 } : stackalloc long[] { w.Ne0, w.Ne1 };
            if (_tensorAllocator == null) return e.HostWeight(w.Format, ne, w.Data, w.Bytes);
            object handle = w;
            if (_hostValues.Contains(w))
            {
                if (!_tensors.TryGetValue(w, out Tensor t))
                {
                    var values = new float[w.Bytes / sizeof(float)];
                    Marshal.Copy(w.Data, values, 0, values.Length);
                    t = w.Vector ? new Tensor(_tensorAllocator, DType.Float32, w.Ne0) : new Tensor(_tensorAllocator, DType.Float32, w.Ne1, w.Ne0);
                    t.SetElementsAsFloat(values);
                    _tensors[w] = t;
                }
                handle = t;
            }
            return e.HostWeight(w.Format, ne, GraphIdentity.Of(w), w.Bytes, handle);
        }

        /// <summary>Drop every device copy of the leaf's bytes: the GGML executor's by address (<see cref="IGraphExecutor.Invalidate"/>);
        /// on direct CUDA a projection's copy in the backend's resident-weight cache, which Invalidate does not reach (the
        /// denoiser releases its copies the same way). A pinned F32 leaf's tensor stays (the codec's own, freed with it).</summary>
        private void Release(MiniMaxH3ImageWeight w)
        {
            if (_tensorAllocator is TensorSharp.Cuda.CudaAllocator cuda) TensorSharp.Cuda.CudaQuantizedOps.ReleaseQuantizedWeight(cuda, w.Data);
            else _executor.Invalidate(w.Data);
        }

        // ---- IResidentComponent

        /// <summary>The decoder's weights, and the encoder's once an edit has made them.</summary>
        public long WeightBytes
        {
            get
            {
                long total = EncoderWeightBytes;
                foreach (MiniMaxH3ImageWeight w in _all) total += w.Bytes;
                return total;
            }
        }

        /// <summary>
        /// The decoder's compute memory for <paramref name="workload"/>'s canvas: one program's batch of tiles (a row, at
        /// most <see cref="MaxBatch"/>) times a tile's activations, calibrated on ggml_cuda and ggml_metal (the fc1 output
        /// and its SwiGLU beside the residual stream and the qkv product, all F32).
        /// </summary>
        public long WorkingBytes(DiffusionWorkload workload) => WorkingBytesFor(Dim, FfnInner, workload.Target.Width * Downsample);

        /// <summary>The decode's working memory (<see cref="WorkingBytes(DiffusionWorkload)"/>), or for
        /// <see cref="DiffusionPhase.EncodeReferences"/> one encoder tile's (<see cref="EncodeWorkingBytesFor"/>); sizing an
        /// encode makes the encoder's weights on the host, so <see cref="WeightBytes"/> counts them from then on.</summary>
        public long WorkingBytes(DiffusionWorkload workload, DiffusionPhase phase)
        {
            if (phase != DiffusionPhase.EncodeReferences) return WorkingBytes(workload);
            EnsureEncoder();
            return EncodeWorkingBytesFor(EncoderChannels);
        }

        /// <summary>The decoder's compute memory for a canvas <paramref name="width"/> pixels wide (one program's row of tiles,
        /// at most <see cref="MaxBatch"/>): 263 MiB for the release decoder's row of 11 tiles, where ggml_cuda measured
        /// 123 MiB and ggml_metal 167 MiB.</summary>
        internal static long WorkingBytesFor(int dim, int ffnInner, int width)
        {
            int batch = Math.Min(MaxBatch, SplitTiles(width).Start.Length);
            long tokens = (long)(TileSize / Downsample) * (TileSize / Downsample) + NumRegisterTokens + 1;
            long perToken = 4L * (2L * ffnInner + ffnInner + 3L * dim + 3L * dim);
            return 64L * 1024 * 1024 + batch * tokens * perToken;
        }

        public void ReleaseWorkspace()
        {
            lock (_gate)
            {
                DisposePlans();
                DisposeEncoderPlans();
            }
        }

        /// <summary>Drops the weights' device copies (and the plans, which bind them); the mapping stays.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                DisposePlans();
                DisposeEncoderPlans();
                foreach (MiniMaxH3ImageWeight w in _all) Release(w);
                foreach (MiniMaxH3ImageWeight w in _encAll) Release(w);
                _preloaded = false;
            }
        }

        /// <summary>Make the weights resident now (MiniMax-H3's uploader on the GGML backends: parallel copies out of the
        /// mapping, as the native decoder uploads them); the first decode does it otherwise. Returns how many were copied.</summary>
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
            // A tensor executor uploads a projection at its first product (its F32 leaves are tensors already).
            int copied = _tensorAllocator == null ? _executor.Preload(new List<IGraphWeight>(_all), "minimax-h3-image/vae") : 0;
            _preloaded = true;
            return copied;
        }

        private void DisposePlans()
        {
            foreach (var p in _plans) p.Plan.Dispose();
            _plans.Clear();
            LastPlan = null;
        }

        // ---- IVaeCodec

        public RgbImage Decode(LatentTensor latent, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(latent);
            LatentShape s = latent.Shape;
            if (s.Channels != LatentChannels || s.Frames != 1)
                throw new ArgumentException($"MiniMax-H3-Image latents are [{LatentChannels}, 1, H, W], not {s}.", nameof(latent));
            // [C, H, W] to the decoder's channel-last tokens.
            int plane = s.Height * s.Width;
            var hwc = new float[latent.Data.LongLength];
            for (int c = 0; c < LatentChannels; c++)
                for (int i = 0; i < plane; i++)
                    hwc[(long)i * LatentChannels + c] = latent.Data[(long)c * plane + i];
            return Decode(hwc, s.Height, s.Width, cancellationToken);
        }

        /// <summary>A decode at the full size, resized so the longer side is about <paramref name="maxSide"/>.</summary>
        public RgbImage Preview(LatentTensor latent, int maxSide)
        {
            RgbImage full = Decode(latent, CancellationToken.None);
            int longer = Math.Max(full.Width, full.Height);
            if (maxSide <= 0 || longer <= maxSide) return full;
            double f = (double)maxSide / longer;
            return QwenImage.ImageIO.Resize(full, Math.Max(1, (int)Math.Round(full.Width * f)), Math.Max(1, (int)Math.Round(full.Height * f)));
        }

        /// <summary>
        /// Decode a normalized image latent, [latentH][latentW][24] channel-last (the DiT's working space), to an RGB image:
        /// denormalize with the latent statistics, decode and blend the 256-pixel tiles (ComfyUI's MiniMaxH3ImageVAE), and
        /// map the ImageNet-normalized output to [0, 1] (clamped). <paramref name="cancellationToken"/> is checked between
        /// tile rows.
        /// </summary>
        public RgbImage Decode(float[] latent, int latentH, int latentW, CancellationToken cancellationToken = default)
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (latent == null || latent.Length != (long)latentH * latentW * LatentChannels)
                    throw new ArgumentException("the image latent must be [latentH][latentW][24].", nameof(latent));
                var raw = new float[latent.Length];
                for (long i = 0; i < raw.Length; i++)
                {
                    int c = (int)(i % LatentChannels);
                    raw[i] = latent[i] * LatentsStd[c] + LatentsMean[c];
                }
                PreloadLocked();

                int height = latentH * Downsample, width = latentW * Downsample;
                var (yIdx, yLen, yOverlap) = SplitTiles(height);
                var (xIdx, xLen, xOverlap) = SplitTiles(width);
                var canvas = new float[(long)OutChannels * height * width];
                float[] strip = null;        // [3][yOverlap of the row above][width], already blended
                int stripRows = 0;
                int outY = 0;
                var clock = Stopwatch.StartNew();
                double native = 0;
                int tiles = 0;
                int maxBatch = Math.Max(1, BatchLimit);

                for (int i = 0; i < yIdx.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    int zi = yIdx[i] / Downsample, zh = yLen[i] / Downsample;
                    float[] newStrip = null;
                    int newStripRows = i < yIdx.Length - 1 ? yOverlap[i] : 0;
                    if (newStripRows > 0) newStrip = new float[(long)OutChannels * newStripRows * width];
                    float[] leftTail = null;     // [3][th][xOverlap[j-1]]
                    int leftTailW = 0, outX = 0, tileHeight = 0;

                    // The row's tiles share a shape (256 px, or the whole side of a narrow canvas), so they go to the
                    // decoder together, maxBatch at a time.
                    int zwRow = xLen[0] / Downsample;
                    var rowTiles = new float[xIdx.Length][];
                    for (int first = 0; first < xIdx.Length; first += maxBatch)
                    {
                        int count = Math.Min(maxBatch, xIdx.Length - first);
                        long per = (long)zh * zwRow * LatentChannels;
                        var batchLatent = new float[per * count];
                        for (int t = 0; t < count; t++)
                        {
                            int zj = xIdx[first + t] / Downsample;
                            for (int y = 0; y < zh; y++)
                                Array.Copy(raw, ((long)(zi + y) * latentW + zj) * LatentChannels,
                                           batchLatent, t * per + (long)y * zwRow * LatentChannels, (long)zwRow * LatentChannels);
                        }
                        long before = clock.ElapsedTicks;
                        float[][] decoded = DecodeTiles(batchLatent, zh, zwRow, count);
                        native += (clock.ElapsedTicks - before) / (double)Stopwatch.Frequency;
                        Array.Copy(decoded, 0, rowTiles, first, count);
                        tiles += count;
                    }

                    for (int j = 0; j < xIdx.Length; j++)
                    {
                        int zw = xLen[j] / Downsample;
                        float[] tile = rowTiles[j];
                        rowTiles[j] = null;
                        int th = zh * Downsample, tw = zw * Downsample;

                        if (i > 0)
                            BlendRows(strip, stripRows, width, xIdx[j], tile, th, tw, yOverlap[i - 1]);
                        if (j > 0)
                            BlendColumns(leftTail, leftTailW, tile, th, tw, xOverlap[j - 1]);

                        int keepW = tw;
                        if (j < xIdx.Length - 1)
                        {
                            leftTailW = xOverlap[j];
                            leftTail = new float[(long)OutChannels * th * leftTailW];
                            for (int c = 0; c < OutChannels; c++)
                                for (int y = 0; y < th; y++)
                                    Array.Copy(tile, ((long)c * th + y) * tw + tw - leftTailW,
                                               leftTail, ((long)c * th + y) * leftTailW, leftTailW);
                            keepW = tw - leftTailW;
                        }
                        int keepH = th;
                        if (newStrip != null)
                        {
                            for (int c = 0; c < OutChannels; c++)
                                for (int y = 0; y < newStripRows; y++)
                                    Array.Copy(tile, ((long)c * th + th - newStripRows + y) * tw,
                                               newStrip, ((long)c * newStripRows + y) * width + outX, keepW);
                            keepH = th - newStripRows;
                        }
                        for (int c = 0; c < OutChannels; c++)
                            for (int y = 0; y < keepH; y++)
                                Array.Copy(tile, ((long)c * th + y) * tw,
                                           canvas, ((long)c * height + outY + y) * width + outX, keepW);
                        tileHeight = keepH;
                        outX += keepW;
                    }
                    strip = newStrip;
                    stripRows = newStripRows;
                    outY += tileHeight;
                }
                LastNativeSeconds = native;
                LastTileCount = tiles;

                // _finalize_pixels: x * std + mean, clamped to [0, 1], interleaved HWC.
                var rgb = new float[(long)height * width * 3];
                float[] mean = MiniMaxH3VideoVae.PixelMean, std = MiniMaxH3VideoVae.PixelStd;
                long planeSize = (long)height * width;
                System.Threading.Tasks.Parallel.For(0, height, y =>
                {
                    long row = (long)y * width;
                    for (long p = row; p < row + width; p++)
                        for (int c = 0; c < 3; c++)
                            rgb[p * 3 + c] = Math.Clamp(canvas[c * planeSize + p] * std[c] + mean[c], 0f, 1f);
                });
                return new RgbImage(width, height, rgb);
            }
        }

        // ---- tiles (ComfyUI's MiniMaxH3ImageVAE, as the legacy decoder did them)

        /// <summary>ComfyUI's split_tiles: the fewest 256-pixel tiles that cover <paramref name="length"/> pixels with
        /// overlaps of at least 64, the spare overlap spread in 16-pixel units from the first gap on. (start, length) per tile
        /// and the overlap after each but the last.</summary>
        internal static (int[] Start, int[] Length, int[] Overlap) SplitTiles(int length)
        {
            if (TileSize >= length) return (new[] { 0 }, new[] { length }, Array.Empty<int>());
            int n = (length + TileSize - 1) / TileSize;
            int[] overlaps;
            int remaining;
            while (true)
            {
                overlaps = new int[n - 1];
                for (int i = 0; i < overlaps.Length; i++) overlaps[i] = TileOverlapMin;
                remaining = TileSize * n - TileOverlapMin * (n - 1) - length;
                if (remaining < 0) n++;
                else break;
            }
            int units = remaining / Downsample;
            for (int i = 0; i < units; i++) overlaps[i % (n - 1)] += Downsample;
            var start = new int[n];
            var len = new int[n];
            for (int i = 0; i < n; i++)
            {
                len[i] = TileSize;
                if (i > 0) start[i] = start[i - 1] + TileSize - overlaps[i - 1];
            }
            return (start, len, overlaps);
        }

        /// <summary>Rotate-half tables for a tile of <paramref name="h"/> x <paramref name="w"/> tokens plus the suffix tokens
        /// (identity): the 2-D create_token_ids grid, (i + 0.5) / n * 2 - 1, at angles 2*pi*pos*theta^(-k/8).</summary>
        private (float[] Cos, float[] Sin) Rope(int h, int w)
        {
            if (_rope.Cos != null && _rope.H == h && _rope.W == w) return (_rope.Cos, _rope.Sin);
            int perAxis = RotDim / 4;   // 2 axes, each frequency rotating a pair
            int half = RotDim / 2;
            var invFreq = new double[perAxis];
            for (int k = 0; k < perAxis; k++) invFreq[k] = Math.Pow(RopeTheta, -(double)k / perAxis);
            int tokens = h * w, seq = tokens + NumRegisterTokens + 1;
            var cos = new float[(long)RotDim * seq];
            var sin = new float[(long)RotDim * seq];
            for (int y = 0; y < h; y++)
            {
                float py = (y + 0.5f) / h * 2f - 1f;
                for (int x = 0; x < w; x++)
                {
                    float px = (x + 0.5f) / w * 2f - 1f;
                    long b = (long)(y * w + x) * RotDim;
                    for (int axis = 0; axis < 2; axis++)
                    {
                        float pos = axis == 0 ? py : px;
                        for (int k = 0; k < perAxis; k++)
                        {
                            // float32 as torch: 2*pi * id * inv_freq
                            float a = (float)(2.0 * Math.PI) * pos * (float)invFreq[k];
                            int j = axis * perAxis + k;
                            float c = MathF.Cos(a), s = MathF.Sin(a);
                            cos[b + j] = c; cos[b + j + half] = c;
                            sin[b + j] = s; sin[b + j + half] = s;
                        }
                    }
                }
            }
            for (int i = tokens; i < seq; i++)
            {
                long b = (long)i * RotDim;
                for (int j = 0; j < RotDim; j++) { cos[b + j] = 1f; sin[b + j] = 0f; }
            }
            _rope = (h, w, cos, sin);
            return (cos, sin);
        }

        /// <summary><paramref name="count"/> tiles through the decoder in one program: each [h*w][24] raw (denormalized)
        /// latent tokens, one after another, to ImageNet-normalized pixels [3][h*16][w*16] per tile.</summary>
        private float[][] DecodeTiles(float[] tokensIn, int h, int w, int count)
        {
            int tokens = h * w;
            var (cos, sin) = Rope(h, w);
            float[] raw = Run(tokensIn, cos, sin, h, w, count);

            // unpatchify: a token's 768 values are [c][py][px].
            int ph = h * Downsample, pw = w * Downsample;
            var result = new float[count][];
            System.Threading.Tasks.Parallel.For(0, count, t =>
            {
                var pixels = new float[(long)OutChannels * ph * pw];
                long tileBase = (long)t * tokens * PatchDim;
                for (int ty = 0; ty < h; ty++)
                    for (int tx = 0; tx < w; tx++)
                    {
                        long src = tileBase + (long)(ty * w + tx) * PatchDim;
                        for (int c = 0; c < OutChannels; c++)
                            for (int py = 0; py < Downsample; py++)
                            {
                                long dst = ((long)c * ph + ty * Downsample + py) * pw + tx * Downsample;
                                long s = src + (c * Downsample + py) * Downsample;
                                Array.Copy(raw, s, pixels, dst, Downsample);
                            }
                    }
                result[t] = pixels;
            });
            return result;
        }

        /// <summary>One decoder program over <paramref name="count"/> tiles: its signature first (a hash-only trace, whose
        /// inputs the run binds), compiled once per tile shape and batch and replayed after. Returns the plan's output
        /// [patch, count * tokens], valid until the next run.</summary>
        private float[] Run(float[] tokensIn, float[] cos, float[] sin, int h, int w, int count)
        {
            using var trace = new GraphEmitter(_executor, record: false);
            Emit(trace, tokensIn, cos, sin, h * w, count);
            int at = _plans.FindIndex(p => p.Batch == count && p.H == h && p.W == w);
            if (at >= 0 && _plans[at].Signature != trace.Signature)
            {
                _plans[at].Plan.Dispose();
                _plans.RemoveAt(at);
                at = -1;
            }
            if (at < 0)
            {
                if (_plans.Count >= MaxPlans)
                {
                    _plans[0].Plan.Dispose();
                    _plans.RemoveAt(0);
                }
                using var recorded = new GraphEmitter(_executor, record: true);
                Emit(recorded, tokensIn, cos, sin, h * w, count);
                IGraphPlan compiled = recorded.Compile(replayable: true);
                LastProgramNodes = recorded.Program.NodeCount;
                _plans.Add((count, h, w, recorded.Signature, compiled));
                Compiles++;
                at = _plans.Count - 1;
            }
            IGraphPlan plan = _plans[at].Plan;
            LastPlan = plan;
            if (!trace.Run(plan))
                throw new InvalidOperationException("MiniMax-H3-Image VAE decode failed: " + plan.LastError);
            return plan.Outputs[0];
        }

        /// <summary>
        /// The decoder over <paramref name="count"/> tiles of <paramref name="tokens"/> latent tokens, in the order and with
        /// the operands of h3_vit_decode's graph (ggml_ops_minimax_h3.cpp): post_quant_conv and x_embedder as per-token
        /// linears, each tile's register tokens and zero token appended, the blocks (RMSNorm, the attention as <see
        /// cref="GraphFusion.VitAttention"/>, the residuals as <see cref="GraphFusion.ScaledResidual"/>), the LayerNorm,
        /// proj_out, and each tile's real patches.
        /// </summary>
        private void Emit(GraphEmitter e, float[] tokensIn, float[] cos, float[] sin, int tokens, int count)
        {
            GraphProgramBuilder b = e.Builder;
            int seq = tokens + NumRegisterTokens + 1;
            float scale = 1f / MathF.Sqrt(HeadDim);
            var leaves = new Dictionary<MiniMaxH3ImageWeight, int>();
            int Leaf(MiniMaxH3ImageWeight w)
            {
                if (w == null) return -1;
                if (leaves.TryGetValue(w, out int node)) return node;
                node = LeafOf(e, w);
                leaves[w] = node;
                return node;
            }
            int Lin(Linear l, int x)
            {
                int o = b.Op(GraphOp.MulMat, Leaf(l.Weight), x);
                return l.Bias != null ? b.Op(GraphOp.Add, o, Leaf(l.Bias)) : o;
            }
            int Rows(int x, long width, long start, long n) => b.Op(GraphOp.View, stackalloc[] { x },
                stackalloc long[] { 2, width, n, 1, 1, width * 4, 0, 0, start * width * 4 });
            int Concat1(int x, int y) => b.Op(GraphOp.Concat, stackalloc[] { x, y }, stackalloc long[] { 1 });
            int Norm(int x, MiniMaxH3ImageWeight w) =>
                b.Op(GraphOp.Mul, b.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { Eps }), Leaf(w));

            int latent = e.InputF32(tokensIn, LatentChannels, (long)tokens * count);
            int cosT = e.InputF32(cos, RotDim, seq);
            int sinT = e.InputF32(sin, RotDim, seq);
            int sinS = GraphDitRegions.SignedSine(b, sinT, RotDim, seq);

            // post_quant_conv is a 1x1 convolution, i.e. a per-token linear; then the embedder.
            int h = Lin(_xEmbedder, Lin(_postQuant, latent));        // [dim, tokens * count]

            // Each tile gets the learned register tokens and a zero token (torch.zeros_like, not the unused mask_token).
            int reg = Leaf(_registerTokens);
            int WithSuffix(int t)
            {
                if (reg >= 0) t = Concat1(t, reg);
                int zero = b.Op(GraphOp.Scale, stackalloc[] { b.Op(GraphOp.Cont, Rows(t, Dim, 0, 1)) }, default, stackalloc[] { 0f });
                return Concat1(t, zero);
            }
            if (count == 1) h = WithSuffix(h);
            else
            {
                int all = -1;
                for (int t = 0; t < count; t++)
                {
                    int one = WithSuffix(Rows(h, Dim, (long)t * tokens, tokens));
                    all = all < 0 ? one : Concat1(all, one);
                }
                h = all;
            }

            foreach (Block blk in _blocks)
            {
                int merged = GraphDitRegions.VitAttention(b, Norm(h, blk.Norm1), Leaf(blk.Qkv.Weight), Leaf(blk.Qkv.Bias), cosT, sinS,
                    HeadDim, Heads, seq, count, RotDim, Eps, scale);
                h = GraphDitRegions.ScaledResidual(b, h, merged, Leaf(blk.Out.Weight), Leaf(blk.Out.Bias), Leaf(blk.Scale1));
                int n2 = Norm(h, blk.Norm2);
                if ((blk.W1.Bias == null) != (blk.W2.Bias == null))
                    throw new InvalidOperationException("the VAE's ff.w1 and ff.w2 carry a bias together or not at all.");
                h = GraphDitRegions.ScaledSwigluResidual(b, h, n2, Leaf(blk.W1.Weight), Leaf(blk.W1.Bias), Leaf(blk.W2.Weight),
                    Leaf(blk.W2.Bias), Leaf(blk.Scale2));
            }

            // norm_out is a LayerNorm (mean-centred, with bias), unlike the blocks' RMSNorm.
            h = b.Op(GraphOp.Mul, b.Op(GraphOp.Norm, stackalloc[] { h }, default, stackalloc[] { Eps }), Leaf(_normOutW));
            if (_normOutB != null) h = b.Op(GraphOp.Add, h, Leaf(_normOutB));
            h = Lin(_projOut, h);                                     // [patch, rows]

            // Each tile's real patches; the suffix tokens carry no pixels.
            int patches = -1;
            for (int t = 0; t < count; t++)
            {
                int one = b.Op(GraphOp.Cont, Rows(h, PatchDim, (long)t * seq, tokens));
                patches = patches < 0 ? one : Concat1(patches, one);
            }
            e.Output(patches, (long)PatchDim * tokens * count);
        }

        // blend(strip slice, tile, extent, dim=-2): the tile's top `extent` rows fade in over the strip's last rows (the
        // strip holds exactly the overlap of the row above).
        private static void BlendRows(float[] strip, int stripRows, int width, int x0, float[] tile, int th, int tw, int overlap)
        {
            int extent = Math.Min(Math.Min(stripRows, th), overlap);
            for (int c = 0; c < OutChannels; c++)
                for (int y = 0; y < extent; y++)
                {
                    float wb = (float)y / extent, wa = 1f - wb;
                    long a = ((long)c * stripRows + (stripRows - extent + y)) * width + x0;
                    long b = ((long)c * th + y) * tw;
                    for (int x = 0; x < tw; x++) tile[b + x] = strip[a + x] * wa + tile[b + x] * wb;
                }
        }

        // blend(left_tail, tile, extent, dim=-1).
        private static void BlendColumns(float[] tail, int tailW, float[] tile, int th, int tw, int overlap)
        {
            int extent = Math.Min(Math.Min(tailW, tw), overlap);
            for (int c = 0; c < OutChannels; c++)
                for (int y = 0; y < th; y++)
                {
                    long a = ((long)c * th + y) * tailW + (tailW - extent);
                    long b = ((long)c * th + y) * tw;
                    for (int x = 0; x < extent; x++)
                    {
                        float wb = (float)x / extent;
                        tile[b + x] = tail[a + x] * (1f - wb) + tile[b + x] * wb;
                    }
                }
        }

        private void DisposeHost()
        {
            foreach (GCHandle h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _model?.Dispose();
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                DisposePlans();
                DisposeEncoderPlans();
                foreach (MiniMaxH3ImageWeight w in _all) Release(w);
                foreach (MiniMaxH3ImageWeight w in _encAll) Release(w);
                foreach (Tensor t in _tensors.Values) t.Dispose();
                _tensors.Clear();
                DisposeHost();
            }
        }
    }
}
