// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's VAE encoder on the diffusion runtime (editing, docs/design/diffusion-runtime.md section 4.2): the
// MiniMax-H3 causal 3-D CNN run over one frame. A single frame is padded in time with zero frames only, so every
// convolution reduces exactly to a 2-D one with its kernel's last temporal tap (diffusers' MiniMaxH3ImageCausalConv3d,
// MiniMaxH3VideoVae.LastSlice); spatial padding is reflect. Each 256-pixel tile is one GraphProgram (GraphOp.Conv2d,
// GraphOp.PadReflect1d, GroupNorm, Silu), replayed for every tile of the shape; the tiles' moments are stitched as
// diffusers' AutoencoderKLMiniMaxH3Image._stitch_tiles does (a linear ramp against each RAW neighbour, the one above and
// then the one to the left, then the overlap cropped) and split into the posterior's mean and log-variance. The weights
// are the file's F16 values widened to F32 on the host, so the convolutions run at F32 (diffusers keeps the encoder in
// float32).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.MiniMaxH3Image
{
    public sealed partial class MiniMaxH3ImageVaeCodec
    {
        /// <summary>GroupNorm groups and epsilon of the encoder (norm_num_groups, norm_eps).</summary>
        public const int EncoderGroups = 32;
        public const float EncoderEps = 1e-6f;

        /// <summary>The moments a tile encodes to: mean and log-variance of every latent channel.</summary>
        public const int MomentChannels = 2 * LatentChannels;

        private sealed class EncConv
        {
            public MiniMaxH3ImageWeight Kernel, Bias;
            public int K, Ic, Oc;
        }

        private sealed class EncBlock
        {
            public MiniMaxH3ImageWeight Norm1W, Norm1B, Norm2W, Norm2B;
            public EncConv Conv1, Conv2, Shortcut;
        }

        private sealed class EncLevel
        {
            public EncBlock[] Blocks;
            public EncConv Down;
        }

        private readonly List<MiniMaxH3ImageWeight> _encAll = new();
        private readonly List<(int H, int W, (ulong, ulong) Signature, IGraphPlan Plan)> _encPlans = new();
        private EncConv _encIn, _encOut, _encQuant;
        private EncLevel[] _encLevels;
        private MiniMaxH3ImageWeight _encNormW, _encNormB;

        /// <summary>Whether the file carries the encoder (Comfy-Org's minimax_h3_image_vae_fp16 does).</summary>
        public bool HasEncoder => _model.HasTensor("encoder.conv_in.weight");

        /// <summary>Why the encoder does not run on a tensor executor (direct CUDA): its program's GroupNorms are
        /// <see cref="GraphOp.GroupNorm"/> nodes, which the tensor executors (TensorGraphExecutor, so CudaGraphExecutor) do not
        /// implement; its Conv2d and PadReflect1d they do. Edits, which encode their references, run on ggml_cuda until the op
        /// is there.</summary>
        internal const string TensorExecutorEncoderGap =
            "MiniMax-H3-Image's VAE encoder (an edit's references) does not run on --backend cuda: its GroupNorm nodes " +
            "(GraphOp.GroupNorm) are not implemented by the CUDA graph executor. Edit on ggml_cuda; text to image runs on cuda.";

        /// <summary>Tiles the last <see cref="Encode"/> encoded, and the seconds its programs ran.</summary>
        public int LastEncodeTiles { get; private set; }

        /// <summary>Tiles every <see cref="Encode"/> so far encoded.</summary>
        public long EncodedTiles { get; private set; }
        public double LastEncodeNativeSeconds { get; private set; }

        /// <summary>Encoder programs compiled (the rest replayed a cached plan; a test seam).</summary>
        internal int EncoderCompiles { get; private set; }

        /// <summary>The encoder's first-level channels (128 in the release), 0 before the encoder is made.</summary>
        internal int EncoderChannels => _encIn?.Oc ?? 0;

        /// <summary>
        /// Make the encoder's weights on the host (widened to F32, the last temporal tap of every kernel): before the first
        /// encode, or when a request with references sizes its memory (their bytes then count in <see cref="WeightBytes"/>).
        /// </summary>
        internal void EnsureEncoder()
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                if (_encLevels != null) return;
                if (!HasEncoder)
                    throw new InvalidOperationException("This MiniMax-H3-Image VAE file has no encoder, so it cannot encode reference images.");
                _encIn = EncConvOf("encoder.conv_in");
                _encOut = EncConvOf("encoder.conv_out");
                _encQuant = EncConvOf("quant_conv");
                _encNormW = EncVec("encoder.norm_out.weight");
                _encNormB = EncVec("encoder.norm_out.bias");
                int levels = 0;
                while (_model.HasTensor($"encoder.down.{levels}.block.0.conv1.weight")) levels++;
                if (levels == 0) throw new InvalidOperationException("The VAE file's encoder has no levels.");
                var all = new EncLevel[levels];
                for (int l = 0; l < levels; l++)
                {
                    string p = $"encoder.down.{l}.";
                    int n = 0;
                    while (_model.HasTensor($"{p}block.{n}.conv1.weight")) n++;
                    var blocks = new EncBlock[n];
                    for (int b = 0; b < n; b++)
                    {
                        string q = $"{p}block.{b}.";
                        blocks[b] = new EncBlock
                        {
                            Norm1W = EncVec(q + "norm1.weight"), Norm1B = EncVec(q + "norm1.bias"),
                            Norm2W = EncVec(q + "norm2.weight"), Norm2B = EncVec(q + "norm2.bias"),
                            Conv1 = EncConvOf(q + "conv1"), Conv2 = EncConvOf(q + "conv2"),
                            Shortcut = _model.HasTensor(q + "nin_shortcut.weight") ? EncConvOf(q + "nin_shortcut") : null,
                        };
                    }
                    all[l] = new EncLevel
                    {
                        Blocks = blocks,
                        // A downsample strides 2 in space wherever it is (the release's 2, 2, 2, 2 of six levels).
                        Down = _model.HasTensor(p + "downsample.conv.weight") ? EncConvOf(p + "downsample.conv") : null,
                    };
                }
                int downs = 0;
                foreach (EncLevel l in all) if (l.Down != null) downs++;
                if (1 << downs != Downsample)
                    throw new InvalidOperationException($"The VAE encoder downsamples {downs} times; MiniMax-H3-Image's latent is 1/{Downsample} of the picture.");
                if (_encQuant.Oc != MomentChannels || _encOut.Oc != MomentChannels)
                    throw new InvalidOperationException($"The VAE encoder emits {_encQuant.Oc} moments, not {MomentChannels}.");
                _encLevels = all;
            }
        }

        /// <summary>A kernel <c>[OC, IC, KD, KH, KW]</c> (or <c>[OC, IC, KH, KW]</c>) as its last temporal tap, widened to F32:
        /// ggml's [KW, KH, IC, OC].</summary>
        private EncConv EncConvOf(string prefix)
        {
            string name = prefix + ".weight";
            long[] shape = _model.TensorShape(name);
            if (shape.Length is not (4 or 5) || shape[^1] != shape[^2] || shape[^1] is not (1 or 3))
                throw new InvalidOperationException($"'{name}' is not a 1x1 or 3x3 convolution kernel.");
            int oc = checked((int)shape[0]), ic = checked((int)shape[1]), k = checked((int)shape[^1]);
            int kd = shape.Length == 5 ? checked((int)shape[2]) : 1;
            float[] src = _model.ReadFloat32(name);
            long plane = (long)k * k;
            var dst = new float[(long)oc * ic * plane];
            for (long o = 0; o < oc; o++)
                for (long i = 0; i < ic; i++)
                    Array.Copy(src, ((o * ic + i) * kd + (kd - 1)) * plane, dst, (o * ic + i) * plane, plane);
            GCHandle h = GCHandle.Alloc(dst, GCHandleType.Pinned);
            _pins.Add(h);
            var kernel = new MiniMaxH3ImageWeight(GraphFormat.F32, (long)k * k * ic, oc, h.AddrOfPinnedObject(), dst.LongLength * sizeof(float));
            _encAll.Add(kernel);
            return new EncConv { Kernel = kernel, Bias = EncVec(prefix + ".bias"), K = k, Ic = ic, Oc = oc };
        }

        private MiniMaxH3ImageWeight EncVec(string name)
        {
            if (!_model.HasTensor(name)) return null;
            float[] values = _model.ReadFloat32(name);
            GCHandle h = GCHandle.Alloc(values, GCHandleType.Pinned);
            _pins.Add(h);
            var w = new MiniMaxH3ImageWeight(GraphFormat.F32, values.LongLength, 1, h.AddrOfPinnedObject(), values.LongLength * sizeof(float), Vector: true);
            _encAll.Add(w);
            return w;
        }

        private long EncoderWeightBytes
        {
            get
            {
                long total = 0;
                foreach (MiniMaxH3ImageWeight w in _encAll) total += w.Bytes;
                return total;
            }
        }

        /// <summary>
        /// The encoder's compute memory for one tile (a program a tile): the first level's im2col columns (the tile's pixels
        /// times 9 taps of every channel, F32) beside about six of its activations. 0.57 GB for the release encoder's
        /// 256-pixel tile (128 channels).
        /// </summary>
        internal static long EncodeWorkingBytesFor(int channels, int tileSide = TileSize)
        {
            long pixels = (long)tileSide * tileSide;
            return 64L * 1024 * 1024 + pixels * channels * 4L * (9 + 6);
        }

        // ---- encode

        /// <summary>
        /// Encode a picture (its sides whole 16-pixel cells, pixels in [0, 1]) to its posterior: ImageNet normalization, the
        /// tiles of <see cref="SplitTiles"/> through the encoder, their moments stitched (diffusers' _stitch_tiles) and split
        /// into mean and log-variance, <c>[24, 1, H/16, W/16]</c>. <paramref name="cancellationToken"/> is checked between tiles.
        /// </summary>
        public LatentPosterior Encode(RgbImage image, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(image);
            if (image.Width % Downsample != 0 || image.Height % Downsample != 0)
                throw new ArgumentException($"MiniMax-H3-Image encodes whole {Downsample}-pixel cells; the picture is {image.Width}x{image.Height}.", nameof(image));
            if (_tensorAllocator != null) throw new NotSupportedException(TensorExecutorEncoderGap);
            EnsureEncoder();
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                int width = image.Width, height = image.Height;
                // (x / 255 - mean) / std, channel-planar (the normalized pixels diffusers' encode step feeds the VAE).
                float[] mean = MiniMaxH3VideoVae.PixelMean, std = MiniMaxH3VideoVae.PixelStd;
                long plane = (long)width * height;
                var chw = new float[3 * plane];
                for (long p = 0; p < plane; p++)
                    for (int c = 0; c < 3; c++)
                        chw[c * plane + p] = (image.Pixels[p * 3 + c] - mean[c]) / std[c];

                var (yIdx, yLen, yOverlap) = SplitTiles(height);
                var (xIdx, xLen, xOverlap) = SplitTiles(width);
                var tiles = new float[yIdx.Length][][];
                var clock = Stopwatch.StartNew();
                double native = 0;
                int count = 0;
                for (int i = 0; i < yIdx.Length; i++)
                {
                    tiles[i] = new float[xIdx.Length][];
                    for (int j = 0; j < xIdx.Length; j++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        int th = yLen[i], tw = xLen[j];
                        var tile = new float[3L * th * tw];
                        for (int c = 0; c < 3; c++)
                            for (int y = 0; y < th; y++)
                                Array.Copy(chw, c * plane + (long)(yIdx[i] + y) * width + xIdx[j], tile, ((long)c * th + y) * tw, tw);
                        long before = clock.ElapsedTicks;
                        float[] moments = RunEncoder(tile, th, tw);
                        native += (clock.ElapsedTicks - before) / (double)Stopwatch.Frequency;
                        tiles[i][j] = (float[])moments.Clone();
                        count++;
                    }
                }
                LastEncodeTiles = count;
                EncodedTiles += count;
                LastEncodeNativeSeconds = native;

                var stitched = StitchMoments(tiles, yLen, xLen, yOverlap, xOverlap, out int lh, out int lw);
                var shape = new LatentShape(LatentChannels, 1, lh, lw);
                long half = shape.Count;
                var meanOut = new float[half];
                var logvar = new float[half];
                Array.Copy(stitched, 0, meanOut, 0, half);
                Array.Copy(stitched, half, logvar, 0, half);
                return new LatentPosterior(shape, meanOut, logvar);
            }
        }

        /// <summary>
        /// diffusers' _stitch_tiles over latent tiles <c>[48][th/16][tw/16]</c>: each tile blends its top rows over the RAW
        /// tile above (dim -2) and then its left columns over the RAW tile to its left (dim -1), with weights
        /// <c>1 - k/extent</c> and <c>k/extent</c>, then drops its overlap with the next tile; rows of tiles are concatenated.
        /// </summary>
        internal static float[] StitchMoments(float[][][] tiles, int[] yLen, int[] xLen, int[] yOverlap, int[] xOverlap,
            out int height, out int width)
        {
            int rows = tiles.Length, cols = tiles[0].Length;
            var keepH = new int[rows];
            var keepW = new int[cols];
            height = 0; width = 0;
            for (int i = 0; i < rows; i++) { keepH[i] = yLen[i] / Downsample - (i < rows - 1 ? yOverlap[i] / Downsample : 0); height += keepH[i]; }
            for (int j = 0; j < cols; j++) { keepW[j] = xLen[j] / Downsample - (j < cols - 1 ? xOverlap[j] / Downsample : 0); width += keepW[j]; }
            var result = new float[(long)MomentChannels * height * width];
            int outY = 0;
            for (int i = 0; i < rows; i++)
            {
                int outX = 0;
                int th = yLen[i] / Downsample;
                for (int j = 0; j < cols; j++)
                {
                    int tw = xLen[j] / Downsample;
                    float[] tile = (float[])tiles[i][j].Clone();
                    if (i > 0)
                    {
                        float[] above = tiles[i - 1][j];
                        int ah = yLen[i - 1] / Downsample;
                        int extent = Math.Min(Math.Min(ah, th), yOverlap[i - 1] / Downsample);
                        for (int c = 0; c < MomentChannels; c++)
                            for (int k = 0; k < extent; k++)
                            {
                                float wb = (float)k / extent, wa = 1f - wb;
                                long a = ((long)c * ah + (ah - extent + k)) * tw;
                                long b = ((long)c * th + k) * tw;
                                for (int x = 0; x < tw; x++)
                                {
                                    float left = above[a + x] * wa;
                                    float right = tile[b + x] * wb;
                                    tile[b + x] = left + right;
                                }
                            }
                    }
                    if (j > 0)
                    {
                        float[] leftTile = tiles[i][j - 1];
                        int lw = xLen[j - 1] / Downsample;
                        int extent = Math.Min(Math.Min(lw, tw), xOverlap[j - 1] / Downsample);
                        for (int c = 0; c < MomentChannels; c++)
                            for (int y = 0; y < th; y++)
                            {
                                long a = ((long)c * th + y) * lw + (lw - extent);
                                long b = ((long)c * th + y) * tw;
                                for (int k = 0; k < extent; k++)
                                {
                                    float wb = (float)k / extent, wa = 1f - wb;
                                    float left = leftTile[a + k] * wa;
                                    float right = tile[b + k] * wb;
                                    tile[b + k] = left + right;
                                }
                            }
                    }
                    for (int c = 0; c < MomentChannels; c++)
                        for (int y = 0; y < keepH[i]; y++)
                            Array.Copy(tile, ((long)c * th + y) * tw, result, ((long)c * height + outY + y) * width + outX, keepW[j]);
                    outX += keepW[j];
                }
                outY += keepH[i];
            }
            return result;
        }

        /// <summary>One tile through the encoder program: compiled once per tile shape, replayed after. Returns the plan's
        /// output, the moments <c>[48][h/16][w/16]</c>, valid until the next run.</summary>
        private float[] RunEncoder(float[] tile, int h, int w)
        {
            using var trace = new GraphEmitter(_executor, record: false);
            EmitEncoder(trace, tile, h, w);
            int at = _encPlans.FindIndex(p => p.H == h && p.W == w);
            if (at >= 0 && _encPlans[at].Signature != trace.Signature)
            {
                _encPlans[at].Plan.Dispose();
                _encPlans.RemoveAt(at);
                at = -1;
            }
            if (at < 0)
            {
                // A picture's tiles are one shape (256 pixels, or the whole of a short side), at most a few distinct ones.
                if (_encPlans.Count >= 2 * MaxPlans)
                {
                    _encPlans[0].Plan.Dispose();
                    _encPlans.RemoveAt(0);
                }
                using var recorded = new GraphEmitter(_executor, record: true);
                EmitEncoder(recorded, tile, h, w);
                IGraphPlan compiled = recorded.Compile(replayable: true);
                _encPlans.Add((h, w, recorded.Signature, compiled));
                EncoderCompiles++;
                at = _encPlans.Count - 1;
            }
            IGraphPlan plan = _encPlans[at].Plan;
            if (!trace.Run(plan))
                throw new InvalidOperationException("MiniMax-H3-Image VAE encode failed: " + plan.LastError);
            return plan.Outputs[0];
        }

        /// <summary>
        /// The encoder over one tile <c>[3][h][w]</c> (ggml <c>[w, h, 3, 1]</c>), diffusers' MiniMaxH3ImageEncoder3d at one
        /// frame: conv_in, per level its resnet blocks (GroupNorm, SiLU, conv, GroupNorm, SiLU, conv, the shortcut added) and
        /// the downsample (a bottom/right reflect pad of 1, a 3x3 convolution of stride 2), then norm_out, SiLU, conv_out and
        /// quant_conv. A 3x3 convolution reflect-pads each side by 1 first; a 1x1 one pads nothing.
        /// </summary>
        private void EmitEncoder(GraphEmitter e, float[] tile, int h, int w)
        {
            GraphProgramBuilder b = e.Builder;
            var leaves = new Dictionary<MiniMaxH3ImageWeight, int>();
            int Vec(MiniMaxH3ImageWeight v, int channels)
            {
                if (v == null) return -1;
                if (!leaves.TryGetValue(v, out int node))
                {
                    node = e.HostWeight(v.Format, stackalloc long[] { v.Ne0 }, v.Data, v.Bytes);
                    leaves[v] = node;
                }
                return b.Op(GraphOp.Reshape, stackalloc[] { node }, stackalloc long[] { 1, 1, channels, 1 });
            }
            int Kernel(EncConv c)
            {
                if (!leaves.TryGetValue(c.Kernel, out int node))
                {
                    node = e.HostWeight(c.Kernel.Format, stackalloc long[] { c.K, c.K, c.Ic, c.Oc }, c.Kernel.Data, c.Kernel.Bytes);
                    leaves[c.Kernel] = node;
                }
                return node;
            }
            // x [W, H, C, 1]: rows of one height (a view of row y), for the height axis's reflect pad.
            int Row(int x, long width, long rows, long channels, long y) => b.Op(GraphOp.View, stackalloc[] { x },
                stackalloc long[] { 4, width, 1, channels, 1, width * 4, width * rows * 4, width * rows * channels * 4, y * width * 4 });
            int Concat(int x, int y, long dim) => b.Op(GraphOp.Concat, stackalloc[] { x, y }, stackalloc long[] { dim });
            int PadW(int x, int left, int right) => b.Op(GraphOp.PadReflect1d, stackalloc[] { x }, stackalloc long[] { left, right });
            int Conv(EncConv c, int x, int stride)
            {
                int y = b.Op(GraphOp.Conv2d, stackalloc[] { Kernel(c), x }, stackalloc long[] { stride, stride, 0, 0, 1, 1 });
                return c.Bias != null ? b.Op(GraphOp.Add, y, Vec(c.Bias, c.Oc)) : y;
            }
            // A 3x3 convolution of stride 1 over x [W, H, C, 1], each side reflect-padded by 1.
            int Conv3(EncConv c, int x, long width, long height)
            {
                int p = PadW(x, 1, 1);
                long pw = width + 2;
                p = Concat(Concat(Row(p, pw, height, c.Ic, 1), p, 1), Row(p, pw, height, c.Ic, height - 2), 1);
                return Conv(c, p, 1);
            }
            int NormSilu(int x, MiniMaxH3ImageWeight weight, MiniMaxH3ImageWeight bias, int channels)
            {
                int n = b.Op(GraphOp.GroupNorm, stackalloc[] { x }, stackalloc long[] { EncoderGroups }, stackalloc[] { EncoderEps });
                if (weight != null) n = b.Op(GraphOp.Mul, n, Vec(weight, channels));
                if (bias != null) n = b.Op(GraphOp.Add, n, Vec(bias, channels));
                return b.Op(GraphOp.Silu, n);
            }

            long cw = w, ch = h;
            int hNode = e.InputF32(tile, w, h, 3, 1);
            hNode = Conv3(_encIn, hNode, cw, ch);
            foreach (EncLevel level in _encLevels)
            {
                foreach (EncBlock blk in level.Blocks)
                {
                    int t = NormSilu(hNode, blk.Norm1W, blk.Norm1B, blk.Conv1.Ic);
                    t = Conv3(blk.Conv1, t, cw, ch);
                    t = NormSilu(t, blk.Norm2W, blk.Norm2B, blk.Conv2.Ic);
                    t = Conv3(blk.Conv2, t, cw, ch);
                    int skip = blk.Shortcut != null ? Conv(blk.Shortcut, hNode, 1) : hNode;
                    hNode = b.Op(GraphOp.Add, skip, t);
                }
                if (level.Down != null)
                {
                    // Downsample3d: reflect-pad the right and the bottom by 1, then the unpadded stride-2 3x3 convolution.
                    int p = PadW(hNode, 0, 1);
                    long pw = cw + 1;
                    p = Concat(p, Row(p, pw, ch, level.Down.Ic, ch - 2), 1);
                    hNode = Conv(level.Down, p, 2);
                    cw /= 2;
                    ch /= 2;
                }
            }
            hNode = NormSilu(hNode, _encNormW, _encNormB, _encOut.Ic);
            hNode = Conv3(_encOut, hNode, cw, ch);
            hNode = Conv(_encQuant, hNode, 1);
            e.Output(hNode, MomentChannels * cw * ch);
        }

        private void DisposeEncoderPlans()
        {
            foreach (var p in _encPlans) p.Plan.Dispose();
            _encPlans.Clear();
        }
    }
}
