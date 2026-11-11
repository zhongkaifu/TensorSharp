// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image VAE decoder (minimax_h3_image_vae_fp16.safetensors).
//
// The image VAE keeps the H3 video VAE's latent space (24 channels, 16x spatial,
// the same latent statistics) and replaces the video ViT decoder with a 2-D one:
// 24 blocks of width 1024 (16 heads of 64), RoPE over the (h, w) token grid on half
// of each head, four register tokens and a zero token, 16x16 RGB patches. Its
// post-quant convolution is a 2-D 1x1, i.e. a per-token linear.
//
// Decoding follows ComfyUI's MiniMaxH3ImageVAE exactly: the latent is cut into
// 256-pixel tiles (the decoder's training size) overlapping by at least 64 pixels,
// each tile is one decoder call, and the tiles are blended into the canvas row by
// row, each against its already-blended neighbours. The transformer is the video
// VAE's native decode (TSGgml_MiniMaxH3VideoVaeDecode) with the image decoder's
// geometry.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    internal sealed class MiniMaxH3ImageVae : IDisposable
    {
        public const int LatentChannels = 24;
        public const int Downsample = 16;
        public const int NumRegisterTokens = 4;
        public const float RopeTheta = 100f;
        public const float Eps = 1e-5f;
        public const int OutChannels = 3;
        public const int TileSize = 256;
        public const int TileOverlapMin = 64;

        /// <summary>The H3 latent statistics (comfy/ldm/minimax/vae.py LATENTS_MEAN/STD), which
        /// the image VAE file does not carry.</summary>
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

        private readonly SafetensorsModel _model;
        private readonly List<GCHandle> _pins = new();
        private readonly List<IntPtr> _bound = new();
        private readonly H3VitBlockW[] _blocks;
        private GCHandle _blocksPin;
        private readonly H3Lin _postQuant, _xEmbedder, _projOut;
        private readonly IntPtr _registerTokens, _normOutW, _normOutB;
        private bool _disposed;

        public int Dim { get; }
        public int Heads { get; }
        public int HeadDim { get; } = 64;
        public int FfnInner { get; }
        /// <summary>Rotated dims per head: 2 axes x 8 frequencies, duplicated (rope_dim_ratio 0.5).</summary>
        public int RotDim { get; }
        public int PatchDim => OutChannels * Downsample * Downsample;
        public int NumBlocks => _blocks.Length;
        public float[] LatentsMean { get; }
        public float[] LatentsStd { get; }

        /// <summary>Seconds of the last <see cref="Decode"/> spent in the native tile decodes.</summary>
        public double LastNativeSeconds { get; private set; }
        public int LastTileCount { get; private set; }

        public MiniMaxH3ImageVae(string path)
        {
            _model = SafetensorsModel.Open(path);
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

            _blocks = new H3VitBlockW[n];
            _postQuant = Lin("post_quant_conv.weight", "post_quant_conv.bias");
            _xEmbedder = Lin("decoder.x_embedder.weight", "decoder.x_embedder.bias");
            _projOut = Lin("decoder.proj_out.weight", "decoder.proj_out.bias");
            if (_projOut.Ne1 != PatchDim)
                throw new InvalidOperationException($"'{path}' decodes {_projOut.Ne1} values a token, not {PatchDim} (16x16 RGB).");
            _registerTokens = PinF32("decoder.register_tokens");
            _normOutW = PinF32("decoder.norm_out.weight");
            _normOutB = PinF32("decoder.norm_out.bias");
            for (int i = 0; i < n; i++)
            {
                string p = $"decoder.transformer_blocks.{i}.";
                _blocks[i] = new H3VitBlockW
                {
                    Norm1 = PinF32(p + "norm1.weight"),
                    Norm2 = PinF32(p + "norm2.weight"),
                    Scale1 = PinF32(p + "scale1"),
                    Scale2 = PinF32(p + "scale2"),
                    Qkv = Lin(p + "attn.to_qkv.weight", p + "attn.to_qkv.bias"),
                    Out = Lin(p + "attn.to_out.weight", p + "attn.to_out.bias"),
                    W1 = Lin(p + "ff.w1.weight", p + "ff.w1.bias"),
                    W2 = Lin(p + "ff.w2.weight", p + "ff.w2.bias"),
                };
            }
            _blocksPin = GCHandle.Alloc(_blocks, GCHandleType.Pinned);
            LatentsMean = _model.HasTensor("latents_mean") ? _model.ReadFloat32("latents_mean") : ReleaseLatentsMean;
            LatentsStd = _model.HasTensor("latents_std") ? _model.ReadFloat32("latents_std") : ReleaseLatentsStd;
        }

        private H3Lin Lin(string weight, string bias)
        {
            if (!_model.TensorOwners.TryGetValue(weight, out var owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            var info = owner.GetInfo(weight);
            for (int i = 2; i < info.Shape.Length; i++)
                if (info.Shape[i] != 1)
                    throw new InvalidOperationException($"'{weight}' is not a linear or 1x1 convolution.");
            if (!owner.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            _bound.Add(ptr);
            return new H3Lin
            {
                W = ptr,
                B = _model.HasTensor(bias) ? PinF32(bias) : IntPtr.Zero,
                Ne0 = info.Shape[1],
                Ne1 = info.Shape[0],
                Bytes = info.ByteCount,
                Type = info.Dtype switch
                {
                    SafetensorDtype.F32 => (int)GgmlTensorType.F32,
                    SafetensorDtype.F16 => (int)GgmlTensorType.F16,
                    SafetensorDtype.BF16 => (int)GgmlTensorType.BF16,
                    _ => throw new NotSupportedException($"'{weight}' is {info.Dtype}; the VAE reads F32, F16 or BF16."),
                },
            };
        }

        private IntPtr PinF32(string name)
        {
            if (!_model.HasTensor(name)) return IntPtr.Zero;
            var h = GCHandle.Alloc(_model.ReadFloat32(name), GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        // ---- tiles ------------------------------------------------------------

        /// <summary>ComfyUI's split_tiles: the fewest 256-pixel tiles that cover
        /// <paramref name="length"/> pixels with overlaps of at least 64, the spare overlap
        /// spread in 16-pixel units from the first gap on. (start, length) per tile and the
        /// overlap after each but the last.</summary>
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

        /// <summary>Rotate-half tables for a tile of <paramref name="h"/> x <paramref name="w"/>
        /// tokens plus the suffix tokens (identity): the 2-D create_token_ids grid,
        /// (i + 0.5) / n * 2 - 1, at angles 2*pi*pos*theta^(-k/8).</summary>
        private (float[] Cos, float[] Sin) Rope(int h, int w)
        {
            // Every tile but a small canvas's single one is 16 x 16 latents.
            if (_rope.Cos != null && _ropeH == h && _ropeW == w) return _rope;
            _rope = BuildRope(h, w);
            _ropeH = h; _ropeW = w;
            return _rope;
        }

        private (float[] Cos, float[] Sin) _rope;
        private int _ropeH, _ropeW;

        private (float[] Cos, float[] Sin) BuildRope(int h, int w)
        {
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
            return (cos, sin);
        }

        /// <summary>Tiles a decoder call takes at most (TS_H3_IMAGE_VAE_BATCH, default 64): a row
        /// of a 16 MP canvas is 21 to 33 tiles, each about 9 MB of activations.</summary>
        private static readonly int MaxBatch = int.TryParse(
            Environment.GetEnvironmentVariable("TS_H3_IMAGE_VAE_BATCH"), out int b) && b > 0 ? b : 64;

        /// <summary><paramref name="count"/> tiles through the decoder in one call: each
        /// [h*w][24] raw (denormalized) latent tokens, one after another, to ImageNet-normalized
        /// pixels [3][h*16][w*16] per tile.</summary>
        private float[][] DecodeTiles(float[] tokensIn, int h, int w, int count)
        {
            int tokens = h * w;
            var (cos, sin) = Rope(h, w);
            var raw = new float[(long)count * tokens * PatchDim];
            var pins = new[]
            {
                GCHandle.Alloc(tokensIn, GCHandleType.Pinned), GCHandle.Alloc(raw, GCHandleType.Pinned),
                GCHandle.Alloc(cos, GCHandleType.Pinned), GCHandle.Alloc(sin, GCHandleType.Pinned),
            };
            try
            {
                var args = new H3VideoVaeDecodeArgs
                {
                    StructBytes = Marshal.SizeOf<H3VideoVaeDecodeArgs>(),
                    NumBlocks = _blocks.Length,
                    Latent = pins[0].AddrOfPinnedObject(),
                    Out = pins[1].AddrOfPinnedObject(),
                    Cos = pins[2].AddrOfPinnedObject(),
                    Sin = pins[3].AddrOfPinnedObject(),
                    RegisterTokens = _registerTokens,
                    NormOutW = _normOutW,
                    NormOutB = _normOutB,
                    PostQuant = _postQuant,
                    XEmbedder = _xEmbedder,
                    ProjOut = _projOut,
                    Blocks = _blocksPin.AddrOfPinnedObject(),
                    Tokens = tokens,
                    LatentC = LatentChannels,
                    Dim = Dim,
                    Heads = Heads,
                    HeadDim = HeadDim,
                    Inner = FfnInner,
                    RotDim = RotDim,
                    NumRegister = NumRegisterTokens,
                    PatchDim = PatchDim,
                    Eps = Eps,
                };
                if (!(count == 1 ? GgmlBasicOps.TryMiniMaxH3VideoVaeDecode(in args)
                                 : GgmlBasicOps.TryMiniMaxH3VitDecodeBatch(in args, count)))
                    throw new InvalidOperationException("MiniMax-H3-Image VAE decode failed: " + GgmlBasicOps.LastNativeError());
            }
            finally
            {
                foreach (var p in pins) p.Free();
            }

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

        /// <summary>
        /// Decode a normalized image latent, [latentH][latentW][24] channel-last (the DiT's
        /// working space), to an RGB image: denormalize with the latent statistics, decode
        /// and blend the 256-pixel tiles, and map the ImageNet-normalized output to [0, 1]
        /// (clamped).
        /// </summary>
        public QwenImage.RgbImage Decode(float[] latent, int latentH, int latentW)
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

            for (int i = 0; i < yIdx.Length; i++)
            {
                int zi = yIdx[i] / Downsample, zh = yLen[i] / Downsample;
                float[] newStrip = null;
                int newStripRows = i < yIdx.Length - 1 ? yOverlap[i] : 0;
                if (newStripRows > 0) newStrip = new float[(long)OutChannels * newStripRows * width];
                float[] leftTail = null;     // [3][th][xOverlap[j-1]]
                int leftTailW = 0, outX = 0, tileHeight = 0;

                // The row's tiles share a shape (256 px, or the whole side of a narrow canvas),
                // so they go to the decoder together, MaxBatch at a time.
                int zwRow = xLen[0] / Downsample;
                var rowTiles = new float[xIdx.Length][];
                for (int first = 0; first < xIdx.Length; first += MaxBatch)
                {
                    int count = Math.Min(MaxBatch, xIdx.Length - first);
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
            long plane = (long)height * width;
            // By rows across the cores (50 M values at 16 MP).
            System.Threading.Tasks.Parallel.For(0, height, y =>
            {
                long row = (long)y * width;
                for (long p = row; p < row + width; p++)
                    for (int c = 0; c < 3; c++)
                        rgb[p * 3 + c] = Math.Clamp(canvas[c * plane + p] * std[c] + mean[c], 0f, 1f);
            });
            return new QwenImage.RgbImage(width, height, rgb);
        }

        // blend(strip slice, tile, extent, dim=-2): the tile's top `extent` rows fade in over
        // the strip's last rows (the strip holds exactly the overlap of the row above).
        private static void BlendRows(float[] strip, int stripRows, int width, int x0,
                                      float[] tile, int th, int tw, int overlap)
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

        /// <summary>Hand the decoder's device copies back (the DiT needs the room between images).</summary>
        public void ReleaseDeviceResidency()
        {
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            ReleaseDeviceResidency();
            if (_blocksPin.IsAllocated) _blocksPin.Free();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _model.Dispose();
        }
    }
}
