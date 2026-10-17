// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 video VAE decode.
//
// The decoder is a pure 36-block transformer -- there is no convolutional decoder
// at all. Every latent voxel becomes one token, 4 learned register tokens and one
// zero token are appended, and a single final Linear(2048 -> 3072) plus a
// depth-to-space reshape performs the whole 4x temporal + 16x spatial upsample.
//
// Weights come from a fp16 safetensors file and are bound to the graph straight
// from the mmap, so the 2.4B-parameter decoder costs ~4.6 GB rather than the
// 9.6 GB an F32 copy would.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 video VAE's transformer decoder.</summary>
    public sealed class MiniMaxH3VideoVae : IDisposable
    {
        // Fixed by the checkpoint (source_config.json vit_decoder_kwargs).
        public const int Dim = 2048;
        public const int Heads = 32;
        public const int HeadDim = 64;
        public const int FfnInner = 8192;
        public const int NumRegisterTokens = 4;
        public const float RopeTheta = 100.0f;
        /// <summary>Fraction of each head that rotates (rope_dim_ratio 0.75).</summary>
        public const int RotDim = 48;
        public const float Eps = 1e-5f;
        public const int PatchSpatial = 16;
        public const int PatchTemporal = 4;
        public const int OutChannels = 3;
        public const int LatentChannels = 24;

        /// <summary>Pixel normalization the decoder's output is expressed in
        /// ("pixel_norm_type": "imagenet").</summary>
        public static readonly float[] PixelMean = { 0.485f, 0.456f, 0.406f };
        public static readonly float[] PixelStd = { 0.229f, 0.224f, 0.225f };

        private readonly SafetensorsModel _model;
        private readonly List<GCHandle> _pins = new();
        // Weights bound resident on the device are cached by host pointer; the cache
        // has to be told when those pointers stop being valid, or the backend still
        // holds their buffers at teardown.
        private readonly List<IntPtr> _bound = new();
        private readonly H3VitBlockW[] _blocks;
        private GCHandle _blocksPin;
        private H3Lin _postQuant, _xEmbedder, _projOut;
        private IntPtr _registerTokens, _normOutW, _normOutB;
        // Non-null on the backends with no ggml graph to call (BackendType.Cpu).
        private MiniMaxH3DirectVideoVae _direct;
        private MiniMaxH3DirectVideoVaeEncoder3D _directEnc3D;
        private readonly IAllocator _allocator;
        private readonly bool _useDirect;
        private bool _disposed;

        /// <summary>Per-channel latent statistics shipped inside the safetensors file.
        /// The transformer works in whitened latent space; decode needs the raw one.</summary>
        public float[] LatentsMean { get; }
        public float[] LatentsStd { get; }

        public int NumBlocks => _blocks.Length;
        public int PatchDim => OutChannels * PatchTemporal * PatchSpatial * PatchSpatial;

        public MiniMaxH3VideoVae(string path) : this(path, BackendType.GgmlCuda, null) { }

        public MiniMaxH3VideoVae(string path, BackendType backend, IAllocator allocator)
        {
            _model = SafetensorsModel.Open(path);

            int n = 0;
            while (_model.HasTensor($"decoder.transformer_blocks.{n}.norm1.weight")) n++;
            if (n == 0)
                throw new InvalidOperationException(
                    $"'{path}' has no MiniMax-H3 video VAE decoder blocks.");
            _blocks = new H3VitBlockW[n];

            _postQuant = Lin("post_quant_conv.weight", "post_quant_conv.bias");
            _xEmbedder = Lin("decoder.x_embedder.weight", "decoder.x_embedder.bias");
            _projOut = Lin("decoder.proj_out.weight", "decoder.proj_out.bias");
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

            // Backends with no ggml graph to call decode through the shared direct
            // primitives instead.
            _allocator = allocator;
            _useDirect = allocator != null && backend is not (BackendType.GgmlCuda
                or BackendType.GgmlCpu or BackendType.GgmlMetal or BackendType.GgmlVulkan);
            LatentsMean = ReadF32("latents_mean");
            LatentsStd = ReadF32("latents_std");
        }

        // Encode-only requests need the causal encoder, so defer the large direct
        // decoder projections until the first decode. The mapped native descriptors
        // above remain cheap and preserve immediate shape/dtype validation.
        private MiniMaxH3DirectVideoVae EnsureDirectDecoder()
        {
            return _direct ??= new MiniMaxH3DirectVideoVae(
                _allocator,
                ReadF32,
                name =>
                {
                    var info=_model.TensorOwners[name].GetInfo(name);
                    return (info.Shape[0],info.Shape[1]);
                },
                _blocks.Length,Dim,Heads,HeadDim,FfnInner,RotDim,NumRegisterTokens,Eps,
                mappedWeight: name =>
                {
                    var owner=_model.TensorOwners[name];
                    var info=owner.GetInfo(name);
                    if(!owner.TryGetTensorDataPointer(info,out IntPtr pointer))
                        throw new InvalidOperationException($"could not map '{name}'.");
                    return (pointer,GgmlTypeOf(info.Dtype),info.ByteCount);
                });
        }

        // ---- weight binding -------------------------------------------------
        // Big matrices are bound straight from the mmap in their on-disk dtype.
        // Vectors (norms, LayerScale, biases) are tiny, so they are converted to
        // F32 once and pinned — the native side wants F32 for those.

        private H3Lin Lin(string weight, string bias)
        {
            if (!_model.TensorOwners.TryGetValue(weight, out var owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            var info = owner.GetInfo(weight);
            if (info.Shape.Length != 2 && !(info.Shape.Length == 5 && info.Shape[2] == 1))
                throw new InvalidOperationException(
                    $"'{weight}' has unexpected rank {info.Shape.Length}.");
            long outFeatures = info.Shape[0];
            long inFeatures = info.Shape[1];
            if (!owner.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            _bound.Add(ptr);
            return new H3Lin
            {
                W = ptr,
                B = _model.HasTensor(bias) ? PinF32(bias) : IntPtr.Zero,
                Ne0 = inFeatures,      // ggml order: ne0 is the contraction dim
                Ne1 = outFeatures,
                Bytes = info.ByteCount,
                Type = GgmlTypeOf(info.Dtype),
            };
        }

        private static int GgmlTypeOf(SafetensorDtype dtype) => dtype switch
        {
            SafetensorDtype.F32 => (int)GgmlTensorType.F32,
            SafetensorDtype.F16 => (int)GgmlTensorType.F16,
            SafetensorDtype.BF16 => (int)GgmlTensorType.BF16,
            _ => throw new NotSupportedException($"unsupported VAE weight dtype {dtype}"),
        };

        private float[] ReadF32(string name) =>
            _model.HasTensor(name) ? _model.ReadFloat32(name) : null;

        private IntPtr PinF32(string name)
        {
            float[] data = ReadF32(name);
            if (data == null) return IntPtr.Zero;
            var h = GCHandle.Alloc(data, GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        // ---- rope -----------------------------------------------------------

        /// <summary>Length-normalized cell-centre coordinate: maps index i of an
        /// axis of length n onto (-1, 1).</summary>
        private static float NormalizedCoord(int i, int n) => (i + 0.5f) / n * 2f - 1f;

        /// <summary>Build the rotate-half cos/sin tables. The 48 rotated dims are
        /// 3 axes x 8 frequencies, duplicated once so pairs (j, j+24) share an angle.
        /// The trailing suffix tokens sit at position 0, i.e. identity rotation.</summary>
        internal static (float[] Cos, float[] Sin) BuildRope(
            int latentT, int latentH, int latentW, int suffixTokens)
        {
            int perAxis = RotDim / (2 * 3);            // 8
            var invFreq = new float[perAxis];
            for (int k = 0; k < perAxis; k++)
                invFreq[k] = (float)Math.Pow(RopeTheta, -(double)k / perAxis);

            int tokens = latentT * latentH * latentW;
            int seq = tokens + suffixTokens;
            var cos = new float[(long)RotDim * seq];
            var sin = new float[(long)RotDim * seq];

            int half = RotDim / 2;                     // 24
            for (int t = 0; t < latentT; t++)
            {
                float pt = NormalizedCoord(t, latentT);
                for (int h = 0; h < latentH; h++)
                {
                    float ph = NormalizedCoord(h, latentH);
                    for (int w = 0; w < latentW; w++)
                    {
                        float pw = NormalizedCoord(w, latentW);
                        int token = (t * latentH + h) * latentW + w;
                        long b = (long)token * RotDim;
                        for (int axis = 0; axis < 3; axis++)
                        {
                            float pos = axis == 0 ? pt : axis == 1 ? ph : pw;
                            for (int k = 0; k < perAxis; k++)
                            {
                                double a = 2.0 * Math.PI * pos * invFreq[k];
                                int j = axis * perAxis + k;
                                float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                                cos[b + j] = c; cos[b + j + half] = c;
                                sin[b + j] = s; sin[b + j + half] = s;
                            }
                        }
                    }
                }
            }
            // Suffix tokens: angle 0 -> cos 1, sin 0.
            for (int i = tokens; i < seq; i++)
            {
                long b = (long)i * RotDim;
                for (int j = 0; j < RotDim; j++) { cos[b + j] = 1f; sin[b + j] = 0f; }
            }
            return (cos, sin);
        }

        // ---- decode ---------------------------------------------------------

        /// <summary>Decode one latent chunk. <paramref name="latent"/> holds
        /// [latentT][latentH][latentW][24] with the channel axis contiguous, i.e.
        /// exactly the token order the transformer wants. Returns the raw patch
        /// tokens, [tokens][patchDim].</summary>
        public float[] DecodeTokens(float[] latent, int latentT, int latentH, int latentW)
        {
            if (latent is null) throw new ArgumentNullException(nameof(latent));
            int tokens = latentT * latentH * latentW;
            long expected = (long)tokens * LatentChannels;
            if (latent.Length != expected)
                throw new ArgumentException(
                    $"latent must hold {expected} floats for {latentT}x{latentH}x{latentW}, got {latent.Length}.",
                    nameof(latent));

            int suffix = NumRegisterTokens + 1;
            var (cos, sin) = BuildRope(latentT, latentH, latentW, suffix);

            if (_useDirect)
            {
                // Pure-managed path: the identical ViT, with no native call.
                return EnsureDirectDecoder().DecodeTokens(latent, tokens, LatentChannels, cos, sin, PatchDim);
            }

            var output = new float[(long)tokens * PatchDim];

            var latentPin = GCHandle.Alloc(latent, GCHandleType.Pinned);
            var cosPin = GCHandle.Alloc(cos, GCHandleType.Pinned);
            var sinPin = GCHandle.Alloc(sin, GCHandleType.Pinned);
            var outPin = GCHandle.Alloc(output, GCHandleType.Pinned);
            try
            {
                var args = new H3VideoVaeDecodeArgs
                {
                    StructBytes = Marshal.SizeOf<H3VideoVaeDecodeArgs>(),
                    NumBlocks = _blocks.Length,
                    Latent = latentPin.AddrOfPinnedObject(),
                    Out = outPin.AddrOfPinnedObject(),
                    Cos = cosPin.AddrOfPinnedObject(),
                    Sin = sinPin.AddrOfPinnedObject(),
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
                if (!GgmlBasicOps.TryMiniMaxH3VideoVaeDecode(in args))
                    throw new InvalidOperationException("MiniMax-H3 video VAE decode failed.");
            }
            finally
            {
                latentPin.Free(); cosPin.Free(); sinPin.Free(); outPin.Free();
            }
            return output;
        }

        /// <summary>Rearrange patch tokens into pixels: [3][T*4][H*16][W*16], the
        /// depth-to-space half of the decoder's single-step upsample. Output is in
        /// ImageNet-normalized space; see <see cref="ToRgb"/>.</summary>
        public static float[] UnpackPatches(
            float[] patches, int latentT, int latentH, int latentW)
        {
            const int pt = PatchTemporal, ps = PatchSpatial, oc = OutChannels;
            int patchDim = oc * pt * ps * ps;
            int frames = latentT * pt, height = latentH * ps, width = latentW * ps;
            var outp = new float[(long)oc * frames * height * width];

            for (int t = 0; t < latentT; t++)
                for (int h = 0; h < latentH; h++)
                    for (int w = 0; w < latentW; w++)
                    {
                        long token = ((long)t * latentH + h) * latentW + w;
                        long src = token * patchDim;
                        for (int c = 0; c < oc; c++)
                            for (int kt = 0; kt < pt; kt++)
                            {
                                int frame = t * pt + kt;
                                for (int kh = 0; kh < ps; kh++)
                                {
                                    long dst = (((long)c * frames + frame) * height + (h * ps + kh))
                                               * width + (long)w * ps;
                                    long s = src + (((long)c * pt + kt) * ps + kh) * ps;
                                    for (int kw = 0; kw < ps; kw++)
                                        outp[dst + kw] = patches[s + kw];
                                }
                            }
                    }
            return outp;
        }

        /// <summary>Latent voxels per decode tile along each spatial axis
        /// (256 px / 16). The decoder's RoPE coordinates are length-normalized over
        /// whatever extent it is handed, and it was trained on this one: feeding it a
        /// whole 40x24 frame in one call produces visibly sheared output even though
        /// every individual tile is exact. Tiling is therefore a correctness
        /// requirement here, not a memory optimization.</summary>
        public const int TileLatent = 16;

        /// <summary>Minimum tile overlap in latent voxels (64 px / 16).</summary>
        public const int TileOverlapMin = 4;

        /// <summary>One tile's placement along an axis, in latent voxels.</summary>
        internal readonly record struct Tile(int Start, int Length, int LeadOverlap, int TrailOverlap);

        /// <summary>Lay out tiles along one axis. Tiles are always exactly
        /// <see cref="TileLatent"/> long and the leftover is absorbed into the seams,
        /// so coverage sums to <paramref name="length"/> exactly. For 40 voxels this
        /// gives 3 tiles overlapping by 4, and for 24 it gives 2 overlapping by 8 —
        /// matching the reference's 0.25 / 0.5 overlap fractions.</summary>
        internal static List<Tile> PlanTiles(int length)
        {
            var tiles = new List<Tile>();
            if (length <= TileLatent)
            {
                tiles.Add(new Tile(0, length, 0, 0));
                return tiles;
            }
            // Grow the tile count until the leftover is enough to give every seam at
            // least the minimum overlap. An exact multiple of the tile size (48 = 3x16)
            // would otherwise leave the tiles abutting with no cross-fade at all, and
            // the seam would show.
            int n = (length + TileLatent - 1) / TileLatent;
            while (n * TileLatent - length < TileOverlapMin * (n - 1)) n++;
            int excess = n * TileLatent - length;
            int seams = n - 1;
            var overlaps = new int[seams];
            for (int i = 0; i < seams; i++)
                overlaps[i] = excess / seams + (i < excess % seams ? 1 : 0);

            int start = 0;
            for (int i = 0; i < n; i++)
            {
                int lead = i == 0 ? 0 : overlaps[i - 1];
                int trail = i == seams ? 0 : overlaps[i];
                tiles.Add(new Tile(start, TileLatent, lead, trail));
                start += TileLatent - trail;
            }
            return tiles;
        }

        /// <summary>Latent frames the decoder consumes per temporal chunk, and the
        /// pixel frames one chunk contributes. The ViT decoder is trained on a short
        /// window; handing it a whole long clip at once still produces a plausible
        /// video, but detail washes out progressively with length — text goes to mush
        /// and a conditioning keyframe's content visibly drifts away. Measured against
        /// the source photo, frame 0 fell from 0.97 correlation at 22 frames to 0.86
        /// at 90 and worse beyond.</summary>
        private const int DecodeTokensPerChunk = 5;
        private const int DecodeTokenOverlap = 2;
        private const int DecodeTokenDrop = 3;
        private const int DecodeFramesPerChunk = 20;
        private const int DecodeFramePrePadding = 3;
        private const int DecodeFrameOverlap = 5;

        /// <summary>Pixel frames a latent of this length decodes to: the 5k+2 -> 17k+5
        /// inverse of the encoder's grid.</summary>
        public static int DecodedFrameCount(int latentT) =>
            latentT <= 1 ? 1 : Math.Max(1, (latentT - 2) / 5 * 17 + 5);

        /// <summary>Decode a whole clip, chunk by chunk along time.
        ///
        /// <para>Each chunk covers 5 latent frames plus 2 of look-ahead, contributes 20
        /// pixel frames after dropping the decoder's 3-frame warm-up, and cross-fades
        /// into its predecessor over 5 frames. Returns channel-planar
        /// [3][frames][H*16][W*16] in ImageNet-normalized space with exactly
        /// <see cref="DecodedFrameCount"/> frames; call <see cref="ToRgb"/> for RGB.</para></summary>
        public float[] Decode(float[] latent, int latentT, int latentH, int latentW,
                              Action<int, int> onTile = null)
        {
            int expected = DecodedFrameCount(latentT);
            long voxel = (long)latentH * latentW * LatentChannels;

            // Pad by repeating the last latent frame so the chunk loop divides evenly.
            int pseudo = latentT + DecodeTokenDrop;
            int pad = (DecodeTokensPerChunk - pseudo % DecodeTokensPerChunk) % DecodeTokensPerChunk;
            pseudo += pad;
            int chunks = pseudo / DecodeTokensPerChunk - 1;
            if (chunks < 1) { pad += DecodeTokensPerChunk; chunks += 1; }

            int paddedT = latentT + pad;
            float[] padded = latent;
            if (pad > 0)
            {
                padded = new float[(long)paddedT * voxel];
                Array.Copy(latent, padded, (long)latentT * voxel);
                for (int i = 0; i < pad; i++)
                    Array.Copy(latent, (long)(latentT - 1) * voxel,
                               padded, (long)(latentT + i) * voxel, voxel);
            }

            int height = latentH * PatchSpatial, width = latentW * PatchSpatial;
            long hw = (long)height * width;
            var result = new List<float[]>();       // each is [3][n][hw], channel-planar
            var counts = new List<int>();
            float[] overlap = null; int overlapCount = 0;
            int totalChunks = chunks;

            for (int i = 0; i < chunks; i++)
            {
                int start = i * DecodeTokensPerChunk;
                int end = Math.Min(start + DecodeTokensPerChunk + DecodeTokenOverlap, paddedT);
                int take = end - start;
                var chunk = new float[(long)take * voxel];
                Array.Copy(padded, (long)start * voxel, chunk, 0, (long)take * voxel);

                // Every window tiles the same plane, so the count runs over the whole clip.
                int window = i;
                float[] pixels = DecodeWindow(chunk, take, latentH, latentW,
                    onTile == null ? null : (done, tiles) => onTile(window * tiles + done, chunks * tiles));
                int decoded = take * PatchTemporal;

                int firstEnd = Math.Min(DecodeFramesPerChunk, decoded);
                int firstStart = Math.Min(DecodeFramePrePadding, firstEnd);
                int firstCount = firstEnd - firstStart;
                float[] first = SliceFrames(pixels, decoded, hw, firstStart, firstCount);

                if (overlap != null)
                {
                    BlendTemporal(overlap, overlapCount, first, firstCount, hw, DecodeFrameOverlap);
                    overlap = null; overlapCount = 0;
                }
                result.Add(first); counts.Add(firstCount);

                if (decoded > DecodeFramesPerChunk + DecodeFramePrePadding)
                {
                    overlapCount = decoded - (DecodeFramesPerChunk + DecodeFramePrePadding);
                    overlap = SliceFrames(pixels, decoded, hw,
                                          DecodeFramesPerChunk + DecodeFramePrePadding, overlapCount);
                }
                if (i == totalChunks - 1 && overlap != null)
                {
                    result.Add(overlap); counts.Add(overlapCount);
                    overlap = null; overlapCount = 0;
                }
            }

            int total = 0;
            foreach (int c in counts) total += c;
            int keep = Math.Min(total, expected);
            var outp = new float[3 * (long)keep * hw];
            int written = 0;
            for (int p = 0; p < result.Count && written < keep; p++)
            {
                int n = Math.Min(counts[p], keep - written);
                for (int ch = 0; ch < 3; ch++)
                    Array.Copy(result[p], (long)ch * counts[p] * hw,
                               outp, ((long)ch * keep + written) * hw, (long)n * hw);
                written += n;
            }
            return outp;
        }

        // One frame range out of a channel-planar [3][frames][hw] buffer.
        private static float[] SliceFrames(float[] src, int frames, long hw, int start, int count)
        {
            var dst = new float[3 * (long)count * hw];
            for (int ch = 0; ch < 3; ch++)
                Array.Copy(src, ((long)ch * frames + start) * hw,
                           dst, (long)ch * count * hw, (long)count * hw);
            return dst;
        }

        // Cross-fade the tail of `previous` into the head of `current`, in place on
        // `current`. Linear over `extent` frames, so the two chunks' weights sum to one.
        private static void BlendTemporal(float[] previous, int previousCount,
                                          float[] current, int currentCount, long hw, int extent)
        {
            extent = Math.Min(extent, Math.Min(previousCount, currentCount));
            int from = previousCount - extent;
            for (int ch = 0; ch < 3; ch++)
                for (int t = 0; t < extent; t++)
                {
                    float wb = t / (float)extent, wa = 1f - wb;
                    long dst = ((long)ch * currentCount + t) * hw;
                    long src = ((long)ch * previousCount + (from + t)) * hw;
                    for (long i = 0; i < hw; i++)
                        current[dst + i] = previous[src + i] * wa + current[dst + i] * wb;
                }
        }

        /// <summary>Decode ONE temporal window, tiling spatially and cross-fading the
        /// seams. Returns channel-planar [3][T*4][H*16][W*16].</summary>
        private float[] DecodeWindow(float[] latent, int latentT, int latentH, int latentW,
                                     Action<int, int> onTile = null)
        {
            var rows = PlanTiles(latentH);
            var cols = PlanTiles(latentW);
            int frames = latentT * PatchTemporal;
            int height = latentH * PatchSpatial, width = latentW * PatchSpatial;
            long hw = (long)height * width;
            long plane = (long)frames * hw;

            var acc = new float[3 * plane];
            var weight = new float[plane];   // shared across channels
            int done = 0, totalTiles = rows.Count * cols.Count;

            foreach (var r in rows)
                foreach (var c in cols)
                {
                    var tile = new float[(long)latentT * r.Length * c.Length * LatentChannels];
                    for (int t = 0; t < latentT; t++)
                        for (int y = 0; y < r.Length; y++)
                            for (int x = 0; x < c.Length; x++)
                            {
                                long src = (((long)t * latentH + (r.Start + y)) * latentW
                                            + (c.Start + x)) * LatentChannels;
                                long dst = (((long)t * r.Length + y) * c.Length + x) * LatentChannels;
                                Array.Copy(latent, src, tile, dst, LatentChannels);
                            }

                    float[] patches = DecodeTokens(tile, latentT, r.Length, c.Length);
                    float[] px = UnpackPatches(patches, latentT, r.Length, c.Length);

                    int th = r.Length * PatchSpatial, tw = c.Length * PatchSpatial;
                    long thw = (long)th * tw;
                    long tplane = (long)frames * thw;
                    int y0 = r.Start * PatchSpatial, x0 = c.Start * PatchSpatial;

                    // Linear ramps on the overlapping edges. Two neighbours' ramps sum
                    // to 1 across a seam, so the normalized accumulation reproduces the
                    // reference's pairwise linear cross-fade.
                    var wy = Ramp(th, r.LeadOverlap * PatchSpatial, r.TrailOverlap * PatchSpatial);
                    var wx = Ramp(tw, c.LeadOverlap * PatchSpatial, c.TrailOverlap * PatchSpatial);

                    for (int f = 0; f < frames; f++)
                        for (int y = 0; y < th; y++)
                        {
                            long dstRow = ((long)f * height + (y0 + y)) * width + x0;
                            long srcRow = ((long)f * th + y) * tw;
                            for (int x = 0; x < tw; x++)
                            {
                                float w = wy[y] * wx[x];
                                weight[dstRow + x] += w;
                                for (int ch = 0; ch < 3; ch++)
                                    acc[ch * plane + dstRow + x] += px[ch * tplane + srcRow + x] * w;
                            }
                        }
                    onTile?.Invoke(++done, totalTiles);
                }

            for (long i = 0; i < plane; i++)
            {
                float w = weight[i];
                if (w <= 0f) continue;
                float inv = 1f / w;
                for (int ch = 0; ch < 3; ch++) acc[ch * plane + i] *= inv;
            }
            return acc;
        }

        private static float[] Ramp(int length, int lead, int trail)
        {
            var w = new float[length];
            for (int i = 0; i < length; i++)
            {
                float v = 1f;
                if (lead > 0 && i < lead) v = (i + 0.5f) / lead;
                if (trail > 0 && i >= length - trail) v = Math.Min(v, (length - i - 0.5f) / trail);
                w[i] = Math.Max(v, 1e-6f);
            }
            return w;
        }

        /// <summary>Undo the ImageNet pixel normalization and clamp to [0,1], in place.</summary>
        public static void ToRgb(float[] pixels, int frames, int height, int width)
        {
            long plane = (long)frames * height * width;
            for (int c = 0; c < OutChannels; c++)
            {
                float m = PixelMean[c], s = PixelStd[c];
                long b = c * plane;
                for (long i = 0; i < plane; i++)
                    pixels[b + i] = Math.Clamp(pixels[b + i] * s + m, 0f, 1f);
            }
        }

        // ---- encode (single frame, for image conditioning) -------------------

        private H3EncLevel[] _encLevels;
        private GCHandle _encLevelsPin;
        private H3Conv _encConvIn, _encConvOut, _encQuantConv;
        private IntPtr _encNormOutW, _encNormOutB;

        /// <summary>True when the checkpoint carries encoder weights.</summary>
        public bool HasEncoder => _model.HasTensor("encoder.conv_in.weight");

        private void EnsureEncoder()
        {
            if (_encLevels != null) return;
            if (!HasEncoder)
                throw new InvalidOperationException(
                    "this video VAE checkpoint has no encoder weights, so image " +
                    "conditioning is unavailable.");

            _encConvIn = Conv("encoder.conv_in");
            _encConvOut = Conv("encoder.conv_out");
            _encQuantConv = Conv("quant_conv");
            _encNormOutW = PinF32("encoder.norm_out.weight");
            _encNormOutB = PinF32("encoder.norm_out.bias");

            int levels = 0;
            while (_model.HasTensor($"encoder.down.{levels}.block.0.conv1.weight")) levels++;
            _encLevels = new H3EncLevel[levels];
            for (int l = 0; l < levels; l++)
            {
                string p = $"encoder.down.{l}.";
                var block0 = EncBlock(p + "block.0.");
                var block1 = EncBlock(p + "block.1.");
                var down = _model.HasTensor(p + "downsample.conv.weight")
                    ? Conv(p + "downsample.conv") : default;
                // Downsample3D always strides 2 spatially when it is present at all;
                // levels whose only change is temporal keep the spatial extent.
                _encLevels[l] = new H3EncLevel
                {
                    Block0 = block0,
                    Block1 = block1,
                    Downsample = down,
                    SpaceStride = down.W != IntPtr.Zero ? 2 : 1,
                };
            }
            _encLevelsPin = GCHandle.Alloc(_encLevels, GCHandleType.Pinned);
        }

        private H3EncResBlock EncBlock(string prefix) => new()
        {
            Norm1W = PinF32(prefix + "norm1.weight"),
            Norm1B = PinF32(prefix + "norm1.bias"),
            Norm2W = PinF32(prefix + "norm2.weight"),
            Norm2B = PinF32(prefix + "norm2.bias"),
            Conv1 = Conv(prefix + "conv1"),
            Conv2 = Conv(prefix + "conv2"),
            Shortcut = _model.HasTensor(prefix + "nin_shortcut.weight")
                ? Conv(prefix + "nin_shortcut") : default,
        };

        /// <summary>Bind a 3-D convolution as its LAST temporal kernel slice.
        ///
        /// The encoder is causal, so a single frame is padded with (k-1) leading ZERO
        /// frames. Every earlier tap therefore multiplies zero and only the last slice
        /// contributes: conv3d([0,0,x]) == conv2d(W[:,:,k-1], x). This is exact, not an
        /// approximation, and it turns a 3-D network into a 2-D one for image
        /// conditioning.</summary>
        private H3Conv Conv(string prefix)
        {
            string weight = prefix + ".weight";
            if (!_model.TensorOwners.TryGetValue(weight, out var owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            var info = owner.GetInfo(weight);
            // safetensors [OC, IC, KD, KH, KW]
            if (info.Shape.Length != 5)
                throw new InvalidOperationException(
                    $"'{weight}' has rank {info.Shape.Length}; expected a 3-D conv kernel.");
            int oc = (int)info.Shape[0], ic = (int)info.Shape[1];
            int kd = (int)info.Shape[2], kh = (int)info.Shape[3], kw = (int)info.Shape[4];

            float[] src = owner.ReadFloat32(weight);
            var dst = new float[(long)oc * ic * kh * kw];
            int last = kd - 1;
            for (int o = 0; o < oc; o++)
                for (int i = 0; i < ic; i++)
                    for (int y = 0; y < kh; y++)
                        for (int x = 0; x < kw; x++)
                            dst[(((long)o * ic + i) * kh + y) * kw + x] =
                                src[((((long)o * ic + i) * kd + last) * kh + y) * kw + x];

            var h = GCHandle.Alloc(dst, GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return new H3Conv
            {
                W = addr,
                B = _model.HasTensor(prefix + ".bias") ? PinF32(prefix + ".bias") : IntPtr.Zero,
                Kw = kw, Kh = kh, Ic = ic, Oc = oc,
                Type = (int)GgmlTensorType.F32,
            };
        }

        // ---- multi-frame (causal 3-D) encode --------------------------------

        /// <summary>Temporal strides per encoder level, from the checkpoint's own
        /// <c>source_config.json</c> (<c>time_down</c>). All six downsample kernels are
        /// 3x3x3, so the file cannot say which of them stride in time — this does.</summary>
        private static readonly int[] TimeDown = { 1, 2, 2, 1, 1, 1 };

        /// <summary>Spatial strides per encoder level (<c>space_down</c>). Their product
        /// is the 16x spatial ratio; <see cref="TimeDown"/>'s is the 4x temporal one.</summary>
        private static readonly int[] SpaceDown = { 2, 2, 2, 2, 1, 1 };

        /// <summary>Pixel frames one encode call consumes. A clip is encoded as
        /// independent chunks of this length plus a final short one; the causal
        /// padding means a chunk carries no history from its predecessor anyway.</summary>
        public const int ChunkFrames = 17;

        /// <summary>Latent frames a chunk of <paramref name="frames"/> pixel frames
        /// produces: 17 -> 5 and 5 -> 2, which is what makes 17k+5 map to 5k+2.</summary>
        public static int ChunkLatentFrames(int frames)
        {
            // Two levels stride 2 in time, each with a k=3 causal front pad of 2.
            int t = frames;
            foreach (int stride in TimeDown)
                if (stride > 1) t = (t - 1) / stride + 1;
            return t;
        }

        private H3EncLevel3D[] _encLevels3D;
        private H3EncLevel3D[] _encFlat;
        private GCHandle _encLevels3DPin, _encFlatPin;
        private H3Conv3d _encConvIn3D, _encConvOut3D, _encQuantConv3D;

        private void EnsureEncoder3D()
        {
            EnsureEncoder();
            if (_encLevels3D != null) return;

            _encConvIn3D = Conv3d("encoder.conv_in");
            _encConvOut3D = Conv3d("encoder.conv_out");
            _encQuantConv3D = Conv3d("quant_conv");

            int levels = _encLevels.Length;
            if (levels != TimeDown.Length)
                throw new InvalidOperationException(
                    $"the checkpoint has {levels} encoder levels but the stride table has " +
                    $"{TimeDown.Length}; they must agree.");
            _encLevels3D = new H3EncLevel3D[levels];
            for (int l = 0; l < levels; l++)
            {
                string p = $"encoder.down.{l}.";
                bool hasDown = _model.HasTensor(p + "downsample.conv.weight");
                _encLevels3D[l] = new H3EncLevel3D
                {
                    Block0 = EncBlock3D(p + "block.0."),
                    Block1 = EncBlock3D(p + "block.1."),
                    Downsample = hasDown ? Conv3d(p + "downsample.conv") : default,
                    SpaceStride = SpaceDown[l],
                    TimeStride = TimeDown[l],
                };
            }
            _encLevels3DPin = GCHandle.Alloc(_encLevels3D, GCHandleType.Pinned);

            _encFlat = new H3EncLevel3D[levels];
            for (int l = 0; l < levels; l++)
                _encFlat[l] = new H3EncLevel3D
                {
                    Block0 = LastSlice(_encLevels3D[l].Block0),
                    Block1 = LastSlice(_encLevels3D[l].Block1),
                    Downsample = LastSlice(_encLevels3D[l].Downsample),
                    SpaceStride = _encLevels3D[l].SpaceStride,
                    TimeStride = _encLevels3D[l].TimeStride,
                };
            _encFlatPin = GCHandle.Alloc(_encFlat, GCHandleType.Pinned);
        }

        private H3EncResBlock3D EncBlock3D(string prefix) => new()
        {
            Norm1W = PinF32(prefix + "norm1.weight"),
            Norm1B = PinF32(prefix + "norm1.bias"),
            Norm2W = PinF32(prefix + "norm2.weight"),
            Norm2B = PinF32(prefix + "norm2.bias"),
            Conv1 = Conv3d(prefix + "conv1"),
            Conv2 = Conv3d(prefix + "conv2"),
            Shortcut = _model.HasTensor(prefix + "nin_shortcut.weight")
                ? Conv3d(prefix + "nin_shortcut") : default,
        };

        /// <summary>Bind a 3-D convolution as KD separate 2-D kernels.
        ///
        /// <para>The native side runs a temporal convolution as KD batched 2-D
        /// convolutions of shifted frames, because ggml's own 3-D convolutions are
        /// either unimplemented on Metal (im2col-3d) or several times slower (the
        /// direct kernel). That needs each temporal slice as its own contiguous
        /// [KW, KH, IC, OC] block, so the depth axis is hoisted to the outside here —
        /// the one place in this file where a kernel is genuinely repacked.</para></summary>
        private H3Conv3d Conv3d(string prefix)
        {
            string weight = prefix + ".weight";
            if (!_model.TensorOwners.TryGetValue(weight, out var owner))
                throw new InvalidOperationException($"missing tensor '{weight}'");
            var info = owner.GetInfo(weight);
            if (info.Shape.Length != 5)
                throw new InvalidOperationException(
                    $"'{weight}' has rank {info.Shape.Length}; expected a 3-D conv kernel.");

            int oc = (int)info.Shape[0], ic = (int)info.Shape[1];
            int kd = (int)info.Shape[2], kh = (int)info.Shape[3], kw = (int)info.Shape[4];
            float[] src = owner.ReadFloat32(weight);
            var dst = new float[src.LongLength];
            long spatial = (long)kh * kw;
            for (int j = 0; j < kd; j++)
                for (int o = 0; o < oc; o++)
                    for (int i = 0; i < ic; i++)
                    {
                        long from = (((long)o * ic + i) * kd + j) * spatial;
                        long to = ((long)j * oc * ic + (long)o * ic + i) * spatial;
                        Array.Copy(src, from, dst, to, spatial);
                    }

            var h = GCHandle.Alloc(dst, GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            // EVERY slice address, not just the base: the native side binds slice j at
            // base + j*sliceBytes, so those are the keys the device-buffer cache holds.
            // Registering only the base leaves the other slices cached against a
            // dangling host pointer, and the next model to land on the same address
            // silently reuses their contents.
            long sliceBytes = (long)kw * kh * ic * oc * sizeof(float);
            for (int j = 0; j < kd; j++) _bound.Add(IntPtr.Add(addr, (int)(j * sliceBytes)));
            return new H3Conv3d
            {
                W = addr,
                B = _model.HasTensor(prefix + ".bias") ? PinF32(prefix + ".bias") : IntPtr.Zero,
                Kw = kw, Kh = kh, Kd = kd, Ic = ic, Oc = oc,
                Type = (int)GgmlTensorType.F32,
            };
        }

        /// <summary>The same kernel restricted to its LAST temporal slice, i.e. the
        /// only one a single causally-padded frame can reach.
        ///
        /// <para>A one-frame clip is padded with kd-1 leading ZERO frames, so every
        /// earlier tap multiplies zero: conv3d([0,0,x]) == conv2d(W[:,:,kd-1], x). The
        /// slices are already stored contiguously, so restricting to the last one is
        /// pointer arithmetic — one convolution per layer instead of three, from the
        /// same weights.</para></summary>
        private static H3Conv3d LastSlice(H3Conv3d c)
        {
            if (c.W == IntPtr.Zero || c.Kd <= 1) return c;
            long stride = c.Kw * c.Kh * c.Ic * c.Oc * sizeof(float);
            c.W = IntPtr.Add(c.W, (int)((c.Kd - 1) * stride));
            c.Kd = 1;
            return c;
        }

        private static H3EncResBlock3D LastSlice(H3EncResBlock3D b) => new()
        {
            Norm1W = b.Norm1W, Norm1B = b.Norm1B, Norm2W = b.Norm2W, Norm2B = b.Norm2B,
            Conv1 = LastSlice(b.Conv1), Conv2 = LastSlice(b.Conv2), Shortcut = LastSlice(b.Shortcut),
        };

        /// <summary>Encode one chunk of frames through the real causal 3-D encoder.
        ///
        /// <para><paramref name="frames"/> holds pixels in [0,1], frame-major and
        /// planar per frame. The result is [Tl][H/16][W/16][24] with the channel axis
        /// contiguous, in the VAE's own scale — <see cref="NormalizeLatent"/> puts it
        /// into the space the denoiser conditions on.</para></summary>
        public float[] EncodeChunk(IReadOnlyList<RgbImage> frames)
        {
            if (frames is not { Count: > 0 }) throw new ArgumentException("no frames", nameof(frames));
            int fullW = frames[0].Width, fullH = frames[0].Height;
            var rows = PlanTiles(fullH / PatchSpatial);
            var cols = PlanTiles(fullW / PatchSpatial);
            if (rows.Count == 1 && cols.Count == 1) return EncodeChunkWhole(frames);

            // Spatially tiled, like the decoder and like the reference: one graph over
            // a 1088x768 clip has activations in the hundreds of megabytes per tensor
            // and runs several times slower per voxel than the same work in 256-pixel
            // tiles. Seams cross-fade in LATENT space on the same linear ramps the
            // decoder uses, so two neighbours' weights sum to one across a seam.
            int latentT = ChunkLatentFrames(frames.Count);
            int latentH = fullH / PatchSpatial, latentW = fullW / PatchSpatial;
            long voxels = (long)latentT * latentH * latentW;
            var acc = new float[voxels * LatentChannels];
            var weight = new float[voxels];

            foreach (var r in rows)
                foreach (var c in cols)
                {
                    int y0 = r.Start * PatchSpatial, x0 = c.Start * PatchSpatial;
                    int th = r.Length * PatchSpatial, tw = c.Length * PatchSpatial;
                    var crop = new List<RgbImage>(frames.Count);
                    foreach (var f in frames) crop.Add(Crop(f, x0, y0, tw, th));

                    float[] tile = EncodeChunkWhole(crop);
                    var wy = Ramp(r.Length, r.LeadOverlap, r.TrailOverlap);
                    var wx = Ramp(c.Length, c.LeadOverlap, c.TrailOverlap);

                    for (int t = 0; t < latentT; t++)
                        for (int y = 0; y < r.Length; y++)
                            for (int x = 0; x < c.Length; x++)
                            {
                                float w = wy[y] * wx[x];
                                long dst = ((long)t * latentH + (r.Start + y)) * latentW + (c.Start + x);
                                long src = ((long)t * r.Length + y) * c.Length + x;
                                weight[dst] += w;
                                for (int ch = 0; ch < LatentChannels; ch++)
                                    acc[dst * LatentChannels + ch] +=
                                        tile[src * LatentChannels + ch] * w;
                            }
                }

            for (long i = 0; i < voxels; i++)
            {
                float w = weight[i];
                if (w <= 0f) continue;
                float inv = 1f / w;
                for (int ch = 0; ch < LatentChannels; ch++) acc[i * LatentChannels + ch] *= inv;
            }
            return acc;
        }

        private static RgbImage Crop(RgbImage source, int x0, int y0, int width, int height)
        {
            if (x0 == 0 && y0 == 0 && width == source.Width && height == source.Height) return source;
            var px = new float[(long)width * height * 3];
            for (int y = 0; y < height; y++)
            {
                long src = ((long)(y0 + y) * source.Width + x0) * 3;
                long dst = (long)y * width * 3;
                Array.Copy(source.Pixels, src, px, dst, (long)width * 3);
            }
            return new RgbImage(width, height, px);
        }

        /// <summary>Build the managed causal 3-D encoder on first use, mirroring the
        /// stride tables and group/eps constants the GGML descriptor is given.</summary>
        private void EnsureDirectEncoder3D()
        {
            if (_directEnc3D != null) return;
            EnsureEncoder3D();
            _directEnc3D = new MiniMaxH3DirectVideoVaeEncoder3D(
                _allocator,
                ReadF32,
                // MUST return null rather than throw for an absent tensor: the loader
                // uses it to detect optional weights (a residual block only carries
                // nin_shortcut when its channel count changes), and indexing the
                // dictionary directly turns "not present" into a KeyNotFoundException.
                name => _model.HasTensor(name)
                    ? _model.TensorOwners[name].GetInfo(name).Shape
                    : null,
                (int[])TimeDown.Clone(), (int[])SpaceDown.Clone(), TimeDown.Length,
                LatentChannels, 32, 1e-6f);
        }

        private float[] EncodeChunkWhole(IReadOnlyList<RgbImage> frames)
        {
            if (frames is not { Count: > 0 }) throw new ArgumentException("no frames", nameof(frames));
            int width = frames[0].Width, height = frames[0].Height;
            if (width % PatchSpatial != 0 || height % PatchSpatial != 0)
                throw new ArgumentException(
                    $"{width}x{height} is not a multiple of {PatchSpatial}.", nameof(frames));
            foreach (var f in frames)
                if (f.Width != width || f.Height != height)
                    throw new ArgumentException("every frame must have the same size.", nameof(frames));
            EnsureEncoder3D();

            int t = frames.Count;
            int latentT = ChunkLatentFrames(t);
            int lw = width / PatchSpatial, lh = height / PatchSpatial;

            // ggml order [W, H, C, T]: time is the BATCH axis, which is what lets the
            // temporal convolutions run as batched 2-D ones and the group norm run per
            // frame without a transpose. Pixels are [0,1] then ImageNet-normalized —
            // NOT the [-1,1] rescale most VAEs use.
            var video = new float[(long)width * height * t * 3];
            long plane = (long)width * height;
            for (int fi = 0; fi < t; fi++)
            {
                var px = frames[fi].Pixels;
                for (int c = 0; c < 3; c++)
                {
                    float mean = PixelMean[c], std = PixelStd[c];
                    long dst = ((long)fi * 3 + c) * plane;
                    for (long i = 0; i < plane; i++)
                        video[dst + i] = (px[i * 3 + c] - mean) / std;
                }
            }

            var outp = new float[(long)lw * lh * latentT * LatentChannels];

            if (_useDirect)
            {
                // Pure-managed path: the identical causal 3-D encoder, no native call.
                EnsureDirectEncoder3D();
                _directEnc3D.Encode(video, width, height, t, latentT, outp);
            }
            else
            {
            var vPin = GCHandle.Alloc(video, GCHandleType.Pinned);
            var oPin = GCHandle.Alloc(outp, GCHandleType.Pinned);
            try
            {
                bool single = t == 1;
                var args = new H3VideoVaeEncode3DArgs
                {
                    StructBytes = Marshal.SizeOf<H3VideoVaeEncode3DArgs>(),
                    NumLevels = _encLevels3D.Length,
                    Video = vPin.AddrOfPinnedObject(),
                    Out = oPin.AddrOfPinnedObject(),
                    ConvIn = single ? LastSlice(_encConvIn3D) : _encConvIn3D,
                    Levels = single ? _encFlatPin.AddrOfPinnedObject()
                                    : _encLevels3DPin.AddrOfPinnedObject(),
                    NormOutW = _encNormOutW,
                    NormOutB = _encNormOutB,
                    ConvOut = single ? LastSlice(_encConvOut3D) : _encConvOut3D,
                    QuantConv = _encQuantConv3D,
                    Width = width, Height = height, Frames = t,
                    LatentFrames = latentT,
                    LatentChannels = LatentChannels,
                    Groups = 32,
                    Eps = 1e-6f,
                };
                if (!GgmlBasicOps.TryMiniMaxH3VideoVaeEncode3D(in args))
                    throw new InvalidOperationException("MiniMax-H3 3-D video encode failed.");
            }
            finally { vPin.Free(); oPin.Free(); }
            }

            // Native emits [W, H, C, T]; the token layout wants the channel contiguous.
            var tokens = new float[outp.LongLength];
            for (int ti = 0; ti < latentT; ti++)
                for (int c = 0; c < LatentChannels; c++)
                    for (int y = 0; y < lh; y++)
                        for (int x = 0; x < lw; x++)
                        {
                            long src = ((long)ti * LatentChannels + c) * lh * lw + (long)y * lw + x;
                            long dst = (((long)ti * lh + y) * lw + x) * LatentChannels + c;
                            tokens[dst] = outp[src];
                        }
            return tokens;
        }

        /// <summary>Encode a whole clip, chunk by chunk.
        ///
        /// <para>The reference splits a 17k+5 clip into k chunks of 17 frames plus a
        /// final 5, encoding each independently — the causal padding means a chunk
        /// starts from zeros regardless, so nothing is lost by not carrying state.</para></summary>
        public float[] EncodeClip(IReadOnlyList<RgbImage> frames, out int latentFrames)
        {
            if (frames is not { Count: > 0 }) throw new ArgumentException("no frames", nameof(frames));
            var parts = new List<float[]>();
            latentFrames = 0;
            for (int start = 0; start < frames.Count;)
            {
                int take = Math.Min(ChunkFrames, frames.Count - start);
                // A trailing chunk shorter than the grid's 5-frame tail cannot produce a
                // whole latent frame, so fold it into the previous chunk's tail instead.
                if (take < 5 && parts.Count > 0) break;
                var chunk = new List<RgbImage>(take);
                for (int i = 0; i < take; i++) chunk.Add(frames[start + i]);
                parts.Add(EncodeChunk(chunk));
                latentFrames += ChunkLatentFrames(take);
                start += take;
            }
            if (parts.Count == 1) return parts[0];
            long total = 0;
            foreach (var p in parts) total += p.LongLength;
            var all = new float[total];
            long off = 0;
            foreach (var p in parts) { Array.Copy(p, 0, all, off, p.Length); off += p.Length; }
            return all;
        }

        /// <summary>Encode one RGB frame to a single latent frame, for image
        /// conditioning. <paramref name="image"/> holds values in [0,1]; the result is
        /// [H/16][W/16][24] with the channel axis contiguous — the same token layout
        /// <see cref="DecodeTokens"/> consumes.
        ///
        /// <para>This is the causal encoder with a one-frame clip, which reduces
        /// exactly to a 2-D network: see <see cref="LastSlice(H3Conv3d)"/>.</para></summary>
        public float[] EncodeFrame(RgbImage image)
        {
            if (image is null) throw new ArgumentNullException(nameof(image));
            return EncodeChunk(new[] { image });
        }

        /// <summary>Normalize a raw VAE latent into the transformer's whitened space —
        /// the inverse of <see cref="DenormalizeLatent"/>.</summary>
        public void NormalizeLatent(float[] latent)
        {
            for (long i = 0; i < latent.Length; i += LatentChannels)
                for (int c = 0; c < LatentChannels; c++)
                    latent[i + c] = (latent[i + c] - LatentsMean[c]) / LatentsStd[c];
        }

        /// <summary>Denormalize a whitened transformer latent into VAE latent space,
        /// in place. <paramref name="latent"/> is channel-contiguous per token.</summary>
        public void DenormalizeLatent(float[] latent)
        {
            for (long i = 0; i < latent.Length; i += LatentChannels)
                for (int c = 0; c < LatentChannels; c++)
                    latent[i + c] = latent[i + c] * LatentsStd[c] + LatentsMean[c];
        }

        /// <summary>Hand the device copies of the VAE's weights back without tearing it down,
        /// exactly as <see cref="MiniMaxH3DiT.ReleaseDeviceResidency"/> does for the denoiser:
        /// the host pointers stay valid and the next encode or decode binds them again.</summary>
        public void ReleaseDeviceResidency()
        {
            if (_disposed) return;
            // Direct linear cache entries use shared-owner leases. Context-owned
            // tensors and the 3D encoder's raw weight tensors remain until disposal.
            _direct?.ReleaseDeviceResidency();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _direct?.Dispose();
            if (_blocksPin.IsAllocated) _blocksPin.Free();
            if (_encLevelsPin.IsAllocated) _encLevelsPin.Free();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
            _bound.Clear();
            if (_encLevels3DPin.IsAllocated) _encLevels3DPin.Free();
            if (_encFlatPin.IsAllocated) _encFlatPin.Free();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
            _model?.Dispose();
        }
    }
}
