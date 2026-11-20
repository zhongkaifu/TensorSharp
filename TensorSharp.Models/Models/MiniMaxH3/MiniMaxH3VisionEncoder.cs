// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3-VL vision tower, used by MiniMax-H3's reference conditioning.
//
// It produces FOUR outputs of width 5120. Output[0] replaces the <|image_pad|>
// embeddings in the language model's input; outputs[1..3] — the "DeepStack" taps
// after vision blocks 8/16/24 — are added residually to the language model's
// hidden states after decoder layers 0, 1 and 2.
//
// Everything that is cheaper or clearer on the host lives here: the resize rules,
// the patch flattening, the bilinear resample of the learned 48x48 position grid,
// and the 2-D RoPE tables.
using System;
using System.IO;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Vision-tower outputs for one image: the merged embedding plus the
    /// DeepStack taps, each [tokens][5120].</summary>
    public sealed class MiniMaxH3VisionOutput
    {
        /// <summary>Merged patch embeddings; these replace the image placeholder tokens.</summary>
        public float[] Merged { get; init; }
        /// <summary>DeepStack taps, injected after language layers 0, 1 and 2.</summary>
        public float[][] DeepStack { get; init; }
        /// <summary>Number of merged tokens (one per 2x2 patch block).</summary>
        public int TokenCount { get; init; }
        /// <summary>Pre-merge patch grid, needed for the language model's M-RoPE.</summary>
        public int GridHeight { get; init; }
        public int GridWidth { get; init; }
    }

    /// <summary>One reference image placed in a prompt: where its placeholder tokens
    /// start, and what the vision tower produced for it.</summary>
    public sealed class MiniMaxH3PromptImage
    {
        /// <summary>Index of the first placeholder token this image replaces.</summary>
        public int TokenIndex { get; init; }
        public MiniMaxH3VisionOutput Vision { get; init; }
    }

    /// <summary>The Qwen3-VL vision tower.</summary>
    public sealed class MiniMaxH3VisionEncoder : IDisposable
    {
        public const int PatchSize = 16;
        public const int TemporalPatch = 2;
        public const int SpatialMerge = 2;
        /// <summary>Patch size x spatial merge: every image dimension is a multiple of this.</summary>
        public const int Factor = PatchSize * SpatialMerge;
        public const int NumGridPerSide = 48;      // sqrt(2304) learned position slots
        public const float RopeTheta = 10000f;
        public const float Eps = 1e-6f;
        /// <summary>Smallest and largest pixel count the reference allows.</summary>
        public const long MinPixels = 56L * 56;
        public const long MaxPixels = 3584L * 3584;

        private readonly GgufFile _gguf;
        private readonly List<GCHandle> _pins = new();
        private readonly List<IntPtr> _bound = new();
        private readonly H3VisBlockW[] _blocks;
        private readonly H3VisMerger[] _mergers;
        private readonly int[] _deepstackLayers;
        private readonly float[] _posEmbedTable;   // [2304][dim]
        private GCHandle _blocksPin, _mergersPin, _layersPin;
        private H3Lin _patchEmbed;
        // Non-null on the backends with no ggml graph to call (BackendType.Cpu).
        private readonly MiniMaxH3DirectVisionEncoder _direct;
        private bool _disposed;

        public int Dim { get; }
        public int Heads { get; } = 16;            // never stored in the checkpoint
        public int HeadDim => Dim / Heads;
        public int OutDim { get; }
        public int NumBlocks => _blocks.Length;
        public int PatchDim => 3 * TemporalPatch * PatchSize * PatchSize;

        public MiniMaxH3VisionEncoder(GgufFile gguf) : this(gguf, BackendType.GgmlCuda, null) { }

        public MiniMaxH3VisionEncoder(GgufFile gguf, BackendType backend, IAllocator allocator)
        {
            _gguf = gguf ?? throw new ArgumentNullException(nameof(gguf));
            if (!gguf.Tensors.ContainsKey("visual.patch_embed.proj.weight"))
                throw new InvalidOperationException(
                    "this text-encoder checkpoint has no vision tower, so reference " +
                    "conditioning is unavailable.");

            Dim = (int)gguf.Tensors["visual.blocks.0.norm1.weight"].Shape[0];
            OutDim = (int)gguf.Tensors["visual.merger.linear_fc2.weight"].Shape[1];

            // The Conv3d kernel is [1152*3, 2, 16, 16] in ne order; as a per-patch
            // linear it is [patchDim, dim], which is the same memory.
            _patchEmbed = Lin("visual.patch_embed.proj.weight", "visual.patch_embed.proj.bias",
                              overrideNe0: PatchDim, overrideNe1: Dim);

            int n = 0;
            while (gguf.Tensors.ContainsKey($"visual.blocks.{n}.norm1.weight")) n++;
            _blocks = new H3VisBlockW[n];
            for (int i = 0; i < n; i++)
            {
                string p = $"visual.blocks.{i}.";
                _blocks[i] = new H3VisBlockW
                {
                    Norm1W = PinF32(p + "norm1.weight"),
                    Norm1B = PinF32(p + "norm1.bias"),
                    Norm2W = PinF32(p + "norm2.weight"),
                    Norm2B = PinF32(p + "norm2.bias"),
                    Qkv = Lin(p + "attn.qkv.weight", p + "attn.qkv.bias"),
                    Proj = Lin(p + "attn.proj.weight", p + "attn.proj.bias"),
                    Fc1 = Lin(p + "mlp.linear_fc1.weight", p + "mlp.linear_fc1.bias"),
                    Fc2 = Lin(p + "mlp.linear_fc2.weight", p + "mlp.linear_fc2.bias"),
                };
            }
            _blocksPin = GCHandle.Alloc(_blocks, GCHandleType.Pinned);

            // Tap layers are not stored either; they follow the depth, per Qwen3-VL.
            _deepstackLayers = n == 27 ? new[] { 8, 16, 24 }
                             : n == 24 ? new[] { 5, 11, 17 }
                             : throw new NotSupportedException(
                                 $"unexpected vision depth {n}; DeepStack taps are only defined for 24 or 27.");
            _layersPin = GCHandle.Alloc(_deepstackLayers, GCHandleType.Pinned);

            // Backends with no ggml graph to call run the same tower through the
            // shared direct primitives instead.
            if (allocator != null && backend is not (BackendType.GgmlCuda or BackendType.GgmlCpu
                    or BackendType.GgmlMetal or BackendType.GgmlVulkan))
                _direct = new MiniMaxH3DirectVisionEncoder(
                    gguf, Dim, Heads, OutDim, PatchDim, SpatialMerge, _deepstackLayers, Eps, allocator);

            // [0] is the final merger, which normalizes BEFORE the merge at width dim;
            // the DeepStack mergers normalize AFTER it at width dim*4.
            _mergers = new H3VisMerger[1 + _deepstackLayers.Length];
            _mergers[0] = Merger("visual.merger.", normBeforeMerge: true);
            for (int i = 0; i < _deepstackLayers.Length; i++)
                _mergers[1 + i] = Merger($"visual.deepstack_merger_list.{i}.", normBeforeMerge: false);
            _mergersPin = GCHandle.Alloc(_mergers, GCHandleType.Pinned);

            _posEmbedTable = ReadF32("visual.pos_embed.weight");
        }

        // ---- weights --------------------------------------------------------

        private H3VisMerger Merger(string prefix, bool normBeforeMerge) => new()
        {
            NormW = PinF32(prefix + "norm.weight"),
            NormB = PinF32(prefix + "norm.bias"),
            Fc1 = Lin(prefix + "linear_fc1.weight", prefix + "linear_fc1.bias"),
            Fc2 = Lin(prefix + "linear_fc2.weight", prefix + "linear_fc2.bias"),
            NormBeforeMerge = normBeforeMerge ? 1 : 0,
        };

        private H3Lin Lin(string weight, string bias, int overrideNe0 = 0, int overrideNe1 = 0)
        {
            var info = _gguf.Tensors[weight];
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{weight}'.");
            _bound.Add(ptr);
            long ne0 = overrideNe0 > 0 ? overrideNe0 : (long)info.Shape[0];
            long ne1 = overrideNe1 > 0 ? overrideNe1 : (long)info.Shape[1];
            return new H3Lin
            {
                W = ptr,
                B = _gguf.Tensors.ContainsKey(bias) ? PinF32(bias) : IntPtr.Zero,
                Ne0 = ne0,
                Ne1 = ne1,
                Type = (int)info.Type,
            };
        }

        private unsafe float[] ReadF32(string name)
        {
            var info = _gguf.Tensors[name];
            var data = new float[info.NumElements];
            if (!_gguf.TryGetTensorDataPointer(info, out IntPtr ptr))
                throw new InvalidOperationException($"could not map '{name}'.");
            switch (info.Type)
            {
                case GgmlTensorType.F32:
                    for (long i = 0; i < data.LongLength; i++) data[i] = ((float*)ptr)[i];
                    break;
                case GgmlTensorType.F16:
                    for (long i = 0; i < data.LongLength; i++)
                        data[i] = (float)BitConverter.UInt16BitsToHalf(((ushort*)ptr)[i]);
                    break;
                case GgmlTensorType.BF16:
                    for (long i = 0; i < data.LongLength; i++)
                        data[i] = BitConverter.UInt32BitsToSingle((uint)((ushort*)ptr)[i] << 16);
                    break;
                default:
                    throw new NotSupportedException($"'{name}' has unexpected type {info.Type}.");
            }
            return data;
        }

        private IntPtr PinF32(string name)
        {
            if (!_gguf.Tensors.ContainsKey(name)) return IntPtr.Zero;
            var h = GCHandle.Alloc(ReadF32(name), GCHandleType.Pinned);
            _pins.Add(h);
            IntPtr addr = h.AddrOfPinnedObject();
            _bound.Add(addr);
            return addr;
        }

        // ---- preprocessing --------------------------------------------------

        /// <summary>Resize an image to the grid the tower expects: every side a
        /// multiple of 32 (patch 16 x merge 2), with the pixel count clamped into the
        /// reference's allowed band while keeping the aspect ratio.</summary>
        public static RgbImage FitForVision(RgbImage image)
        {
            int w = Snap(image.Width), h = Snap(image.Height);
            long pixels = (long)w * h;
            if (pixels > MaxPixels || pixels < MinPixels)
            {
                double scale = Math.Sqrt((pixels > MaxPixels ? MaxPixels : MinPixels) / (double)pixels);
                w = Snap((int)Math.Round(image.Width * scale));
                h = Snap((int)Math.Round(image.Height * scale));
            }
            return w == image.Width && h == image.Height ? image : QwenImage.ImageIO.Resize(image, w, h);
        }

        private static int Snap(int v)
        {
            int r = (int)Math.Round(v / (double)Factor) * Factor;
            return Math.Max(Factor, r);
        }

        /// <summary>Flatten one or two frames into per-patch rows in MERGE-BLOCK order
        /// (block-row, block-col, inner-row, inner-col) — the order the mergers, the
        /// position embeddings and the RoPE tables all assume.
        ///
        /// <para>Each row is (channel, temporal, row, col) = 3*2*16*16 values. A STILL
        /// image fills both temporal slots with itself; a VIDEO BLOCK fills them with
        /// two consecutive frames, and that is the <i>only</i> thing that differs
        /// between the two — the token count, the position embeddings, the RoPE and
        /// the DeepStack taps are identical either way.</para>
        ///
        /// <para>Pixels are mapped to [-1,1]; note this is NOT CLIP mean/std
        /// normalization, unlike every other Qwen-VL path.</para></summary>
        public static float[] Patchify(RgbImage image, out int gridH, out int gridW)
            => Patchify(image, null, out gridH, out gridW);

        public static float[] Patchify(RgbImage first, RgbImage second, out int gridH, out int gridW)
        {
            var image = first;
            if (second != null && (second.Width != first.Width || second.Height != first.Height))
                throw new ArgumentException(
                    "both frames of a video block must have the same size.", nameof(second));
            var frames = new[] { first.Pixels, (second ?? first).Pixels };
            gridH = image.Height / PatchSize;
            gridW = image.Width / PatchSize;
            int patchDim = 3 * TemporalPatch * PatchSize * PatchSize;
            var rows = new float[(long)gridH * gridW * patchDim];
            int w = image.Width;

            int token = 0;
            for (int bh = 0; bh < gridH / SpatialMerge; bh++)
                for (int bw = 0; bw < gridW / SpatialMerge; bw++)
                    for (int ih = 0; ih < SpatialMerge; ih++)
                        for (int iw = 0; iw < SpatialMerge; iw++)
                        {
                            int py = (bh * SpatialMerge + ih) * PatchSize;
                            int pxx = (bw * SpatialMerge + iw) * PatchSize;
                            long dst = (long)token * patchDim;
                            for (int c = 0; c < 3; c++)
                                for (int kt = 0; kt < TemporalPatch; kt++)
                                    for (int ky = 0; ky < PatchSize; ky++)
                                        for (int kx = 0; kx < PatchSize; kx++)
                                        {
                                            long src = ((long)(py + ky) * w + (pxx + kx)) * 3 + c;
                                            float v = Math.Clamp(frames[kt][src], 0f, 1f) * 2f - 1f;
                                            rows[dst + (((long)c * TemporalPatch + kt) * PatchSize + ky)
                                                     * PatchSize + kx] = v;
                                        }
                            token++;
                        }
            return rows;
        }

        /// <summary>Bilinearly resample the learned 48x48 position grid onto the
        /// actual patch grid, emitted in the same merge-block order as the patches.</summary>
        public float[] BuildPositionEmbeddings(int gridH, int gridW)
        {
            int tokens = gridH * gridW;
            var outp = new float[(long)tokens * Dim];
            double maxIndex = NumGridPerSide - 1;

            int token = 0;
            for (int bh = 0; bh < gridH / SpatialMerge; bh++)
                for (int bw = 0; bw < gridW / SpatialMerge; bw++)
                    for (int ih = 0; ih < SpatialMerge; ih++)
                    {
                        int y = bh * SpatialMerge + ih;
                        double hp = gridH == 1 ? 0 : maxIndex * y / (gridH - 1);
                        int hf = (int)Math.Floor(hp);
                        int hc = Math.Min(hf + 1, NumGridPerSide - 1);
                        double dh = hp - hf;
                        for (int iw = 0; iw < SpatialMerge; iw++)
                        {
                            int x = bw * SpatialMerge + iw;
                            double wp = gridW == 1 ? 0 : maxIndex * x / (gridW - 1);
                            int wf = (int)Math.Floor(wp);
                            int wc = Math.Min(wf + 1, NumGridPerSide - 1);
                            double dw = wp - wf;

                            long dst = (long)token * Dim;
                            Blend(outp, dst, (hf * NumGridPerSide + wf), (1 - dh) * (1 - dw));
                            Blend(outp, dst, (hf * NumGridPerSide + wc), (1 - dh) * dw);
                            Blend(outp, dst, (hc * NumGridPerSide + wf), dh * (1 - dw));
                            Blend(outp, dst, (hc * NumGridPerSide + wc), dh * dw);
                            token++;
                        }
                    }
            return outp;
        }

        private void Blend(float[] dst, long dstOffset, int row, double weight)
        {
            if (weight == 0) return;
            long src = (long)row * Dim;
            float w = (float)weight;
            for (int i = 0; i < Dim; i++) dst[dstOffset + i] += _posEmbedTable[src + i] * w;
        }

        /// <summary>2-D rotate-half RoPE over the patch grid: half the frequency pairs
        /// encode the row, half the column.</summary>
        public (float[] Cos, float[] Sin) BuildRope(int gridH, int gridW)
        {
            int hd = HeadDim;
            int half = hd / 2;                 // frequency pairs
            int perAxis = half / 2;            // pairs per axis
            int tokens = gridH * gridW;
            var cos = new float[(long)hd * tokens];
            var sin = new float[(long)hd * tokens];

            int token = 0;
            for (int bh = 0; bh < gridH / SpatialMerge; bh++)
                for (int bw = 0; bw < gridW / SpatialMerge; bw++)
                    for (int ih = 0; ih < SpatialMerge; ih++)
                        for (int iw = 0; iw < SpatialMerge; iw++)
                        {
                            int y = bh * SpatialMerge + ih;
                            int x = bw * SpatialMerge + iw;
                            long b = (long)token * hd;
                            for (int j = 0; j < half; j++)
                            {
                                int axis = j < perAxis ? 0 : 1;
                                int k = axis == 0 ? j : j - perAxis;
                                double inv = Math.Pow(RopeTheta, -(2.0 * k) / (perAxis * 2));
                                double a = (axis == 0 ? y : x) * inv;
                                float c = (float)Math.Cos(a), s = (float)Math.Sin(a);
                                cos[b + j] = c; cos[b + j + half] = c;
                                sin[b + j] = s; sin[b + j + half] = s;
                            }
                            token++;
                        }
            return (cos, sin);
        }

        // ---- encode ---------------------------------------------------------

        /// <summary>Run the tower on one still image.</summary>
        public MiniMaxH3VisionOutput Encode(RgbImage image) => Encode(image, null);

        /// <summary>Run the tower on a video block: two consecutive frames sharing one
        /// set of patches. <paramref name="second"/> null falls back to a still image.</summary>
        public MiniMaxH3VisionOutput Encode(RgbImage first, RgbImage second)
        {
            RgbImage fitted = FitForVision(first);
            RgbImage fittedSecond = second == null
                ? null
                : second.Width == fitted.Width && second.Height == fitted.Height
                    ? second
                    : QwenImage.ImageIO.Resize(second, fitted.Width, fitted.Height);
            float[] patches = Patchify(fitted, fittedSecond, out int gridH, out int gridW);
            int tokens = gridH * gridW;
            int merged = tokens / (SpatialMerge * SpatialMerge);

            float[] pos = BuildPositionEmbeddings(gridH, gridW);
            var (cos, sin) = BuildRope(gridH, gridW);
            int outputs = 1 + _deepstackLayers.Length;
            float[] flat;

            if (_direct != null)
            {
                // Pure-managed path: the identical tower, with no native call.
                flat = _direct.Encode(patches, pos, cos, sin, tokens);
            }
            else
            {
            flat = new float[(long)merged * outputs * OutDim];

            var pPin = GCHandle.Alloc(patches, GCHandleType.Pinned);
            var ePin = GCHandle.Alloc(pos, GCHandleType.Pinned);
            var cPin = GCHandle.Alloc(cos, GCHandleType.Pinned);
            var sPin = GCHandle.Alloc(sin, GCHandleType.Pinned);
            var oPin = GCHandle.Alloc(flat, GCHandleType.Pinned);
            try
            {
                var args = new H3VisionEncodeArgs
                {
                    StructBytes = Marshal.SizeOf<H3VisionEncodeArgs>(),
                    NumBlocks = _blocks.Length,
                    NumDeepstack = _deepstackLayers.Length,
                    Patches = pPin.AddrOfPinnedObject(),
                    PosEmbed = ePin.AddrOfPinnedObject(),
                    Cos = cPin.AddrOfPinnedObject(),
                    Sin = sPin.AddrOfPinnedObject(),
                    Out = oPin.AddrOfPinnedObject(),
                    PatchEmbed = _patchEmbed,
                    Blocks = _blocksPin.AddrOfPinnedObject(),
                    DeepstackLayers = _layersPin.AddrOfPinnedObject(),
                    Mergers = _mergersPin.AddrOfPinnedObject(),
                    Tokens = tokens,
                    Dim = Dim,
                    Heads = Heads,
                    HeadDim = HeadDim,
                    PatchDim = PatchDim,
                    MergeSize = SpatialMerge,
                    OutDim = OutDim,
                    Eps = Eps,
                };
                if (!GgmlBasicOps.TryMiniMaxH3VisionEncode(in args))
                    throw new InvalidOperationException("MiniMax-H3 vision encode failed.");
            }
            finally { pPin.Free(); ePin.Free(); cPin.Free(); sPin.Free(); oPin.Free(); }
            }

            // TS_H3_DUMP_VIS writes the tower output for BOTH paths, so the managed
            // tower can be compared against GGML on ONE forward instead of through a
            // denoise that amplifies whatever it differed by.
            string dumpPath = Environment.GetEnvironmentVariable("TS_H3_DUMP_VIS");
            if (!string.IsNullOrEmpty(dumpPath))
            {
                using var fs = new FileStream(dumpPath, FileMode.Create, FileAccess.Write);
                using var bw = new BinaryWriter(fs);
                foreach (float v in flat) bw.Write(v);
            }

            long span = (long)merged * OutDim;
            var mergedOut = new float[span];
            Array.Copy(flat, 0, mergedOut, 0, span);
            var deep = new float[_deepstackLayers.Length][];
            for (int i = 0; i < deep.Length; i++)
            {
                deep[i] = new float[span];
                Array.Copy(flat, (i + 1) * span, deep[i], 0, span);
            }
            return new MiniMaxH3VisionOutput
            {
                Merged = mergedOut,
                DeepStack = deep,
                TokenCount = merged,
                GridHeight = gridH,
                GridWidth = gridW,
            };
        }

        /// <summary>Bytes the tower's weights take on the device once an encode has bound them (every pinned or mapped
        /// tensor it binds).</summary>
        public long DeviceWeightBytes
        {
            get
            {
                long total = (long)(_patchEmbed.Ne0 * _patchEmbed.Ne1 * ElementBytes(_patchEmbed.Type));
                foreach (H3VisBlockW b in _blocks)
                    total += LinBytes(b.Qkv) + LinBytes(b.Proj) + LinBytes(b.Fc1) + LinBytes(b.Fc2) + 4L * Dim * 4;
                foreach (H3VisMerger m in _mergers)
                    total += LinBytes(m.Fc1) + LinBytes(m.Fc2);
                return total;
            }
        }

        private static long LinBytes(H3Lin l) => (long)(l.Ne0 * l.Ne1 * ElementBytes(l.Type)) + (l.B != IntPtr.Zero ? 4L * l.Ne1 : 0);

        // Bytes an element of a ggml type takes, averaged over its blocks (the tower's weights are F32, F16 or BF16; a
        // quantized one counts at about a byte).
        private static double ElementBytes(int type) => (GgmlTensorType)type switch
        {
            GgmlTensorType.F32 => 4,
            GgmlTensorType.F16 or GgmlTensorType.BF16 => 2,
            _ => 1,
        };

        /// <summary>Hand the tower's device copies back (a diffusion pipeline's phase residency); the weights stay pinned and
        /// mapped, so the next encode uploads them from memory.</summary>
        public void ReleaseDeviceResidency()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _direct?.Dispose();
            foreach (IntPtr ptr in _bound)
                if (ptr != IntPtr.Zero) GgmlBasicOps.InvalidateHostBuffer(ptr);
            _bound.Clear();
            if (_blocksPin.IsAllocated) _blocksPin.Free();
            if (_mergersPin.IsAllocated) _mergersPin.Free();
            if (_layersPin.IsAllocated) _layersPin.Free();
            foreach (var h in _pins) if (h.IsAllocated) h.Free();
            _pins.Clear();
        }
    }
}
