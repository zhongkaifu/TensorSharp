// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// Fused whole-VAE graph for Qwen-Image-2.1 (TSGgml_QwenVaeRun).
//
// The per-conv path (QwenImage21Vae / VaeReferenceMath) runs the VAE as a C# op
// chain where only the convs go to the device — each conv re-uploads its weights
// and the full feature map and downloads the result, and SiLU / channel-RMSNorm /
// nearest-upsample run as C# loops over the host arrays in between.
//
// This class emits the SAME topology as QwenImage21Vae's Encode21/Decode21 as a
// flat op list that the native kernel executes as ONE device-resident ggml graph:
// features never leave the GPU, weights are bound resident from stable unmanaged
// buffers (uploaded once, reused across encode/decode/edits), and every conv
// keeps F32 intermediates (Aux=1), because the 2.1 decoder exceeds the F16 range.
// Per call: one input upload, one compute, one sync, one output download.
//
// The op lists are resolution-independent (shapes flow from the input tensor),
// so one build serves every image size. A non-finite result falls back to the
// per-conv path.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models.QwenImage
{
    internal sealed class QwenImageVaeGraph : IDisposable, IGgmlHostMemoryOwner
    {
        private const int KConv = 0, KNorm = 1, KSilu = 2, KUp = 3, KSave = 4, KAdd = 5, KAttn = 6,
            KAverageDown21 = 7, KDuplicateUp21 = 8;
        private const int LatentChannels = 64;
        private const int PixelChannels = 4;   // RGBA
        private const int SpatialFactor = 16;

        private readonly List<IntPtr> _allocs = new();
        private QwenVaeWeightRef[] _weights;
        private QwenVaeOp[] _encodeOps, _decodeOps;
        private bool _disposed;

        public static QwenImageVaeGraph TryBuild(VaeWeights w)
        {
            var g = new QwenImageVaeGraph();
            try
            {
                g.Build(w);
                return g;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  [vae-fused] build failed, using per-conv path: {ex.Message}");
                g.Dispose();
                return null;
            }
        }

        public unsafe bool TryEncode(float[] chw, int H, int W, out float[] z32, out int lh, out int lw)
        {
            lh = H / SpatialFactor; lw = W / SpatialFactor;
            z32 = null;
            if (H % SpatialFactor != 0 || W % SpatialFactor != 0) return false;
            var outp = new float[2L * LatentChannels * lh * lw];
            if (!Run(_encodeOps, chw, W, H, PixelChannels, outp) || !Finite21(outp)) return false;
            z32 = outp;
            return true;
        }

        public unsafe bool TryDecode(float[] latent, int lh, int lw, out float[] rgb, out int H, out int W)
        {
            H = lh * SpatialFactor; W = lw * SpatialFactor;
            rgb = null;
            var outp = new float[(long)PixelChannels * H * W];
            if (!Run(_decodeOps, latent, lw, lh, LatentChannels, outp) || !Finite21(outp)) return false;
            rgb = outp;
            return true;
        }

        private unsafe bool Run(QwenVaeOp[] ops, float[] input, int inW, int inH, int inC, float[] output)
        {
            fixed (float* ip = input, op = output)
            fixed (QwenVaeOp* opsPtr = ops)
            fixed (QwenVaeWeightRef* wPtr = _weights)
            {
                var a = new QwenVaeArgs
                {
                    Input = (IntPtr)ip, InW = inW, InH = inH, InC = inC,
                    Output = (IntPtr)op, OutLen = output.LongLength,
                    Ops = (IntPtr)opsPtr, NumOps = ops.Length,
                    Weights = (IntPtr)wPtr, NumWeights = _weights.Length,
                    StructBytes = Marshal.SizeOf<QwenVaeArgs>(),
                };
                bool ok = GgmlBasicOps.TryQwenVaeRun(in a);
                if (Environment.GetEnvironmentVariable("TS_QWEN21_VAE_TRACE") == "1")
                    Console.WriteLine($"  [vae21] fused {inC}x{inH}x{inW}: ops={ops.Length} success={ok}");
                return ok;
            }
        }

        private bool Finite21(float[] values)
        {
            if (!Array.Exists(values, v => !float.IsFinite(v))) return true;
            Console.WriteLine("  [vae-fused] non-finite output, retrying the per-conv path.");
            return false;
        }

        // ---- build ------------------------------------------------------------

        private void Build(VaeWeights w)
        {
            var weights = new List<QwenVaeWeightRef>();

            int Reg(float[] data)
            {
                long bytes = data.LongLength * sizeof(float);
                IntPtr p = Marshal.AllocHGlobal((IntPtr)bytes);
                _allocs.Add(p);
                Marshal.Copy(data, 0, p, data.Length);
                weights.Add(new QwenVaeWeightRef { Data = p, Bytes = bytes });
                return weights.Count - 1;
            }
            // Causal Conv3d on T=1 uses only the last temporal slice (kd = KD-1) of the
            // 5D (oc,ic,kd,kh,kw) weight — same slicing as VaeReferenceMath.LastTemporalSlice
            // (which the managed decoder reads once per layer through VaeWeights.KernelSlice / PackedKernel).
            int RegSlice(string name, int OC, int IC, int KD, int KH, int KW)
            {
                float[] w5d = w.Get(name);
                if (KD == 1) return Reg(w5d);
                var slice = new float[(long)OC * IC * KH * KW];
                int khw = KH * KW, kdhw = KD * KH * KW;
                Parallel.For(0, OC, oc =>
                {
                    for (int ic = 0; ic < IC; ic++)
                    {
                        long src = ((long)oc * IC + ic) * kdhw + (long)(KD - 1) * khw;
                        long dst = ((long)oc * IC + ic) * khw;
                        Array.Copy(w5d, src, slice, dst, khw);
                    }
                });
                return Reg(slice);
            }

            List<QwenVaeOp> ops = null;
            // Every conv keeps F32 intermediates (Aux=1): the 2.1 decoder's activations
            // exceed the F16 range before its final norm.
            void Emit(int kind, int wi = -1, int bi = -1, int oc = 0, int ic = 0, int kh = 0, int kw = 0,
                      int sh = 1, int sw = 1, int pt = 0, int pb = 0, int pl = 0, int pr = 0,
                      int src = 0, int dst = 0, int aux = 0) =>
                ops.Add(new QwenVaeOp { Kind = kind, W = wi, B = bi, Oc = oc, Ic = ic, Kh = kh, Kw = kw,
                                        Sh = sh, Sw = sw, Pt = pt, Pb = pb, Pl = pl, Pr = pr,
                                        Src = src, Dst = dst, Aux = kind == KConv ? 1 : aux });

            void Norm(string gammaName) => Emit(KNorm, Reg(w.Get(gammaName)));
            void Silu() => Emit(KSilu);

            void AttentionBlock(string prefix, int c)
            {
                Emit(KSave, src: 0, dst: 1);
                Norm(prefix + ".norm.gamma");
                Emit(KConv, Reg(w.Get(prefix + ".to_qkv.weight")), Reg(w.Get(prefix + ".to_qkv.bias")), 3 * c, c, 1, 1);
                Emit(KAttn, oc: c);
                Emit(KConv, Reg(w.Get(prefix + ".proj.weight")), Reg(w.Get(prefix + ".proj.bias")), c, c, 1, 1);
                Emit(KAdd, src: 0, dst: 0, aux: 1);
            }
            void Conv21(string prefix, int pad, int src = 0, int dst = 0, int stride = 1, bool endPad = false)
            {
                var shape = w.Shape(prefix + ".weight");
                int oc = (int)shape[0], ic = (int)shape[1], kh = (int)shape[^2], kw = (int)shape[^1];
                int kd = shape.Length == 5 ? (int)shape[2] : 1;
                Emit(KConv, RegSlice(prefix + ".weight", oc, ic, kd, kh, kw), Reg(w.Get(prefix + ".bias")),
                    oc, ic, kh, kw, sh: stride, sw: stride, pt: pad, pb: endPad ? 1 : pad,
                    pl: pad, pr: endPad ? 1 : pad, src: src, dst: dst);
            }
            void Residual21(string prefix)
            {
                if (w.Has(prefix + ".shortcut.weight")) Conv21(prefix + ".shortcut", 0, dst: 1);
                else Emit(KSave, dst: 1);
                Norm(prefix + ".residual.0.gamma"); Silu();
                Conv21(prefix + ".residual.2", 1);
                Norm(prefix + ".residual.3.gamma"); Silu();
                Conv21(prefix + ".residual.6", 1);
                Emit(KAdd, aux: 1);
            }
            void Mid21(string prefix)
            {
                Residual21(prefix + ".0");
                AttentionBlock(prefix + ".1", w.Get(prefix + ".1.norm.gamma").Length);
                Residual21(prefix + ".2");
            }

            // ---- encoder (mirrors VaeReferenceMath.Encode21 after pixel normalize) ----
            ops = new List<QwenVaeOp>();
            Conv21("encoder.conv1", 1);
            int[] encChannels = { 96, 192, 384, 768, 768 };
            for (int stage = 0; stage < 5; ++stage)
            {
                string prefix = $"encoder.downsamples.{stage}.downsamples";
                Emit(KAverageDown21, oc: encChannels[stage], kh: stage is >= 1 and <= 3 ? 2 : 1,
                    kw: stage < 4 ? 2 : 1, dst: 2);
                Residual21(prefix + ".0"); Residual21(prefix + ".1");
                if (stage < 4) Conv21(prefix + ".2.resample.1", 0, stride: 2, endPad: true);
                Emit(KAdd, aux: 2);
            }
            Mid21("encoder.middle");
            Norm("encoder.head.0.gamma"); Silu();
            Conv21("encoder.head.2", 1); Conv21("conv1", 0);
            _encodeOps = ops.ToArray();

            // ---- decoder (mirrors VaeReferenceMath.Decode21 before the [-1,1]->[0,1] map) ----
            ops = new List<QwenVaeOp>();
            Conv21("conv2", 0); Conv21("decoder.conv1", 1);
            Mid21("decoder.middle");
            int[] decChannels = { 1152, 1152, 576, 288, 144 };
            for (int stage = 0; stage < 5; ++stage)
            {
                string prefix = $"decoder.upsamples.{stage}.upsamples";
                if (stage < 4) Emit(KDuplicateUp21, oc: decChannels[stage], kh: stage < 3 ? 2 : 1, kw: 2, dst: 2);
                for (int j = 0; j < 3; ++j) Residual21(prefix + $".{j}");
                if (stage < 4)
                {
                    Emit(KUp); Conv21(prefix + ".3.resample.1", 1);
                    Emit(KAdd, aux: 2);
                }
            }
            Norm("decoder.head.0.gamma"); Silu(); Conv21("decoder.head.2", 1);
            _decodeOps = ops.ToArray();
            _weights = weights.ToArray();
        }

        /// <summary>The weight copies this graph allocated (their device copies are keyed by them).</summary>
        void IGgmlHostMemoryOwner.CollectHostMemory(System.Collections.Generic.List<(IntPtr Base, long Length)> ranges,
            System.Collections.Generic.List<IntPtr> keys) => keys.AddRange(_allocs);

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            foreach (var p in _allocs)
            {
                GgmlBasicOps.InvalidateHostBuffer(p);
                Marshal.FreeHGlobal(p);
            }
            _allocs.Clear();
            _weights = null;
        }
    }
}
