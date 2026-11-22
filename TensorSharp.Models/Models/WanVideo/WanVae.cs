// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Wan causal 3D video VAE on the GGML backends (decoder + encoder), covering
// both generations:
//   - Wan 2.1 (wan_2.1_vae.safetensors): z=16, 8x spatial / 4x temporal.
//   - Wan 2.2 TI2V (Wan2.2_VAE.safetensors): z=48, 16x spatial (8x conv + 2x2
//     pixel patchify) / 4x temporal, residual AvgDown/DupUp scale shortcuts.
// Weights load through the shared WanVaeWeights parser; conv taps are
// registered as stable unmanaged F16 buffers (the graphs always convolved in
// F16 — converting at load removes the per-graph F32->F16 casts) and norms /
// biases / attention weights as F32. Each causal 3D conv is pre-sliced into
// its temporal taps so the native graphs (TSGgml_WanVaeDecode /
// TSGgml_WanVaeEncode) run them as batched 2D convolutions. Decode/encode
// iterate their temporal chunks with the causal feature caches inside the one
// native graph (port of stable-diffusion.cpp wan_vae.hpp / diffusers
// AutoencoderKLWan). The pixel-side pipeline (normalization, tiling, frame
// assembly) lives in WanVaeBase.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorSharp.Models.WanVideo
{
    internal sealed class WanVae : WanVaeBase, IGgmlHostMemoryOwner
    {
        private readonly List<IntPtr> _allocs = new();
        private readonly WanVaeDecodeArgsTemplate _t;
        private readonly WanVaeEncodeArgsTemplate _e;

        // The descriptors minus the per-call fields, built once at load.
        private struct WanVaeDecodeArgsTemplate
        {
            public WanVaeConv Conv2, Conv1;
            public WanVaeResBlockW Mid0, Mid2;
            public WanVaeAttnW Mid1;
            public WanVaeResBlockW[] Res;   // 12
            public WanVaeUpsampleW[] Up;    // 3
            public WanVaeNorm HeadNorm;
            public WanVaeConv HeadConv;
        }

        private struct WanVaeEncodeArgsTemplate
        {
            public WanVaeConv Stem;
            public WanVaeResBlockW[] Res;   // 8
            public WanVaeDownW[] Down;      // 3
            public WanVaeResBlockW Mid0, Mid2;
            public WanVaeAttnW Mid1;
            public WanVaeNorm HeadNorm;
            public WanVaeConv HeadConv;
            public WanVaeConv Quant;
        }

        public WanVae(string vaePath)
        {
            // Parse the checkpoint once into managed tables (shared with the
            // direct-backend WanDirectVae), then register stable unmanaged F32
            // copies for the native whole-graph kernels.
            var wts = WanVaeWeights.Load(vaePath);
            InitFromWeights(wts);

            WanVaeConv Conv(WanVaeConvWeights c)
            {
                if (c == null) return default;
                // Taps are registered as F16 (TapType = 1). The native graph
                // always ran these convs with F16 kernels anyway (it cast
                // F32 -> F16 in-graph); converting at load is numerically
                // identical, halves the resident weight bytes, and drops one
                // cast node per conv tap per chunk from every VAE graph.
                var r = new WanVaeConv
                {
                    Kd = c.Kd, K = c.K, Ic = c.Ic, Oc = c.Oc,
                    Bias = c.Bias != null ? Reg(c.Bias) : IntPtr.Zero,
                    TapType = 1,
                };
                for (int tap = 0; tap < c.Kd; tap++)
                {
                    IntPtr p = Reg16(c.Taps[tap]);
                    if (tap == 0) r.Tap0 = p; else if (tap == 1) r.Tap1 = p; else r.Tap2 = p;
                }
                return r;
            }
            WanVaeNorm Norm(WanVaeNormWeights n) => new WanVaeNorm { Gamma = Reg(n.Gamma), C = n.C };
            WanVaeResBlockW Res(WanVaeResBlockWeights r) => new WanVaeResBlockW
            {
                N0 = Norm(r.N0),
                C2 = Conv(r.C2),
                N3 = Norm(r.N3),
                C6 = Conv(r.C6),
                Shortcut = Conv(r.Shortcut),
            };
            WanVaeAttnW Attn(WanVaeAttnWeights a) => new WanVaeAttnW
            {
                Norm = Norm(a.Norm),
                QkvW = Reg(a.QkvW),
                QkvB = Reg(a.QkvB),
                ProjW = Reg(a.ProjW),
                ProjB = Reg(a.ProjB),
                C = a.C,
            };
            WanVaeUpsampleW Up(WanVaeUpsampleWeights u) => new WanVaeUpsampleW
            {
                SConv = Conv(u.SConv),
                TimeConv = Conv(u.TimeConv),
            };
            WanVaeDownW Down(WanVaeDownWeights d) => new WanVaeDownW
            {
                SConv = Conv(d.SConv),
                TConv = Conv(d.TConv),
            };

            _t.Conv2 = Conv(wts.Conv2);
            _t.Conv1 = Conv(wts.Conv1);
            _t.Mid0 = Res(wts.Mid0);
            _t.Mid2 = Res(wts.Mid2);
            _t.Mid1 = Attn(wts.Mid1);
            _t.Res = new WanVaeResBlockW[12];
            for (int i = 0; i < 12; i++) _t.Res[i] = Res(wts.Res[i]);
            _t.Up = new WanVaeUpsampleW[3];
            for (int i = 0; i < 3; i++) _t.Up[i] = Up(wts.Up[i]);
            _t.HeadNorm = Norm(wts.HeadNorm);
            _t.HeadConv = Conv(wts.HeadConv);

            if (HasEncoder)
            {
                _e.Stem = Conv(wts.EncStem);
                _e.Res = new WanVaeResBlockW[8];
                for (int i = 0; i < 8; i++) _e.Res[i] = Res(wts.EncRes[i]);
                _e.Down = new WanVaeDownW[3];
                for (int i = 0; i < 3; i++) _e.Down[i] = Down(wts.EncDown[i]);
                _e.Mid0 = Res(wts.EncMid0);
                _e.Mid2 = Res(wts.EncMid2);
                _e.Mid1 = Attn(wts.EncMid1);
                _e.HeadNorm = Norm(wts.EncHeadNorm);
                _e.HeadConv = Conv(wts.EncHeadConv);
                _e.Quant = Conv(wts.EncQuant);
            }
        }

        /// <summary>
        /// Band tiling exists to keep a full-resolution decode inside a small card's
        /// VRAM, and it is not free: the bands overlap, so the decoder runs more
        /// latent rows than the plane has, plus a band buffer alongside the canvas.
        /// Measured at 1088x832x121f on an M5 Pro, decoding the plane whole is both
        /// FASTER and LIGHTER than two bands — 565 s vs 655 s, peak RSS 4.85 vs
        /// 5.37 GB. So scale the threshold with the memory actually available
        /// instead of pinning it at a 16 GB card's budget; the constant is anchored
        /// so that ~16 GB free reproduces the historical 640 kpx.
        /// </summary>
        protected override long TilePixelThreshold => _tileThreshold ??= ResolveTileThreshold();

        private long? _tileThreshold;

        private static long ResolveTileThreshold()
        {
            const long Anchor = 640_000;                 // px, the previous fixed value
            const long AnchorFreeBytes = 16L << 30;      // the card size it was chosen for
            try
            {
                if (GgmlBasicOps.TryGetDeviceMemoryInfo(out long freeBytes, out _) && freeBytes > 0)
                    return Math.Max(Anchor / 4, (long)((double)freeBytes / AnchorFreeBytes * Anchor));
            }
            catch (DllNotFoundException) { /* managed-only host */ }
            return Anchor;
        }

        private IntPtr Reg(float[] data)
        {
            long bytes = data.LongLength * sizeof(float);
            IntPtr p = Marshal.AllocHGlobal((IntPtr)bytes);
            _allocs.Add(p);
            Marshal.Copy(data, 0, p, data.Length);
            return p;
        }

        /// <summary>Registers a stable unmanaged F16 copy of <paramref name="data"/>.
        /// (Half) rounds to nearest even — bit-identical to the ggml F32→F16 cast
        /// these weights previously went through in-graph.</summary>
        private unsafe IntPtr Reg16(float[] data)
        {
            long bytes = data.LongLength * sizeof(ushort);
            IntPtr p = Marshal.AllocHGlobal((IntPtr)bytes);
            _allocs.Add(p);
            System.Threading.Tasks.Parallel.For(0, (int)((data.LongLength + 1048575) / 1048576), block =>
            {
                ushort* dst = (ushort*)p;
                long start = (long)block * 1048576;
                long end = Math.Min(start + 1048576, data.LongLength);
                for (long i = start; i < end; i++)
                    dst[i] = BitConverter.HalfToUInt16Bits((System.Half)data[i]);
            });
            return p;
        }

        protected override unsafe void DecodeNative(float[] z, int t, int latRows, int lw, float[] pixels)
        {
            long outLen = (long)(lw * SpatialScale) * (latRows * SpatialScale) * 3 * (1 + (t - 1) * TemporalScale);
            fixed (float* zp = z, op = pixels)
            {
                var d = new WanVaeDecodeArgs
                {
                    Z = (IntPtr)zp,
                    Out = (IntPtr)op,
                    OutLen = outLen,
                    Conv2 = _t.Conv2,
                    Conv1 = _t.Conv1,
                    Mid0 = _t.Mid0,
                    Mid2 = _t.Mid2,
                    Mid1 = _t.Mid1,
                    Res0 = _t.Res[0], Res1 = _t.Res[1], Res2 = _t.Res[2], Res3 = _t.Res[3],
                    Res4 = _t.Res[4], Res5 = _t.Res[5], Res6 = _t.Res[6], Res7 = _t.Res[7],
                    Res8 = _t.Res[8], Res9 = _t.Res[9], Res10 = _t.Res[10], Res11 = _t.Res[11],
                    Up0 = _t.Up[0], Up1 = _t.Up[1], Up2 = _t.Up[2],
                    HeadNorm = _t.HeadNorm,
                    HeadConv = _t.HeadConv,
                    Zw = lw, Zh = latRows, Zt = t, Zc = ZDim,
                    Version = Version,
                    PatchSize = PatchSize,
                    StructBytes = Marshal.SizeOf<WanVaeDecodeArgs>(),
                };
                if (!GgmlBasicOps.TryWanVaeDecode(in d))
                    throw new InvalidOperationException("Wan VAE decode failed (see native error above).");
            }
        }

        protected override unsafe void EncodeNative(float[] x, int t, int ph, int pw, int pc, float[] mu)
        {
            fixed (float* xp = x, op = mu)
            {
                var d = new WanVaeEncodeArgs
                {
                    X = (IntPtr)xp,
                    Out = (IntPtr)op,
                    OutLen = mu.LongLength,
                    Stem = _e.Stem,
                    Res0 = _e.Res[0], Res1 = _e.Res[1], Res2 = _e.Res[2], Res3 = _e.Res[3],
                    Res4 = _e.Res[4], Res5 = _e.Res[5], Res6 = _e.Res[6], Res7 = _e.Res[7],
                    Down0 = _e.Down[0], Down1 = _e.Down[1], Down2 = _e.Down[2],
                    Mid0 = _e.Mid0, Mid2 = _e.Mid2, Mid1 = _e.Mid1,
                    HeadNorm = _e.HeadNorm,
                    HeadConv = _e.HeadConv,
                    Quant = _e.Quant,
                    PxW = pw, PxH = ph, PxC = pc, PxT = t,
                    ZDim = ZDim,
                    Version = Version,
                    StructBytes = Marshal.SizeOf<WanVaeEncodeArgs>(),
                };
                if (!GgmlBasicOps.TryWanVaeEncode(in d))
                    throw new InvalidOperationException("Wan VAE encode failed (see native error above).");
            }
        }

        /// <summary>The weight copies this decoder allocated.</summary>
        void IGgmlHostMemoryOwner.CollectHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys) => keys.AddRange(_allocs);

        public override void Dispose()
        {
            foreach (var p in _allocs)
            {
                // The native side resident-caches device copies by host pointer; a later
                // allocation reusing this address must not hit a stale device buffer.
                GgmlBasicOps.InvalidateHostBuffer(p);
                Marshal.FreeHGlobal(p);
            }
            _allocs.Clear();
        }
    }
}
