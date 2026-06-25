// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Stage 7: native fused-block path for the DiT. The managed Block (QwenImageDiT.cs)
// is the verified correctness reference; this routes the attention + MLP sub-layers
// through the single-graph native kernels (TSGgml_QwenImageJointAttn /
// TSGgml_QwenImageModMlp) to eliminate the per-op host round-trips that leave the
// GPU idle in the managed path.
using System;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageDiT
    {
        // The native fused-block path runs the DiT on the device in one graph per block
        // (vs the managed per-op path that is launch-bound and ~4x slower). It only
        // applies to GGML backends (uses ggml ops); default-on there, opt out with
        // TS_QWEN_DIT_NATIVE=0. Non-GGML backends (pure CUDA/CPU/MLX) use the managed path.
        internal bool NativeBlockOn =>
            IsGgmlBackend && Environment.GetEnvironmentVariable("TS_QWEN_DIT_NATIVE") != "0";

        // Whole-block fusion (attn + both MLPs in one native graph). Default-on under
        // the native path; opt out with TS_QWEN_DIT_FUSED_BLOCK=0 to use the 3-call path.
        internal static readonly bool FusedBlockOn =
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_FUSED_BLOCK") != "0";

        // Per-step DiT block-loop wall-time print (isolates the DiT forward from the
        // fixed VAE/text-encoder cost). TS_QWEN_DIT_TIMING=1.
        internal static readonly bool TimingOn =
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_TIMING") == "1";

        // Raw GGUF weight (+optional bias) descriptor for the native kernels.
        private QImgAttnW GgufW(string weightName, string biasName)
        {
            var info = DitGgufLocal.Tensors[weightName];
            DitGgufLocal.TryGetTensorDataPointer(info, out IntPtr wp);
            IntPtr bp = IntPtr.Zero;
            if (biasName != null && DitGgufLocal.Tensors.TryGetValue(biasName, out var binfo))
                DitGgufLocal.TryGetTensorDataPointer(binfo, out bp);
            return new QImgAttnW
            {
                W = wp,
                Type = (int)info.Type,
                Ne0 = (long)info.Shape[0],
                Ne1 = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
                Bytes = DitGgufLocal.GetTensorByteCount(info),
                B = bp,
            };
        }

        private IntPtr GgufF32Ptr(string name)
        {
            DitGgufLocal.TryGetTensorDataPointer(DitGgufLocal.Tensors[name], out IntPtr p);
            return p;
        }

        private GgufFile DitGgufLocal => _gguf;

        // Per-token (1+scale), shift, gate from the [2,18432] mod params (folds modulate_index).
        private static void PrecomputeMod(float[] modParams, int half, int[] modIndex, int seq,
            out float[] scale1, out float[] shift, out float[] gate)
        {
            int dim = Dim, baseOff = half * 3 * dim;
            scale1 = new float[(long)seq * dim];
            shift = new float[(long)seq * dim];
            gate = new float[(long)seq * dim];
            for (int s = 0; s < seq; s++)
            {
                int idx = modIndex != null ? modIndex[s] : 0;
                long mb = (long)idx * 18432 + baseOff;
                long o = (long)s * dim;
                for (int c = 0; c < dim; c++)
                {
                    scale1[o + c] = 1f + modParams[mb + dim + c];
                    shift[o + c] = modParams[mb + c];
                    gate[o + c] = modParams[mb + 2 * dim + c];
                }
            }
        }

        // Managed reference for the attention sub-layer (uses the verified Block primitives).
        internal void ManagedAttnSubLayer(float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            float[] imgMod, float[] txtMod, int[] modIndex, DitRope rope, int layer)
        {
            string b = $"transformer_blocks.{layer}";
            Tensor img = HostToTensor(imgHost, imgSeq, Dim);
            Tensor txt = HostToTensor(txtHost, txtSeq, Dim);
            using (Tensor imgN1 = LayerNormNoAffine(img, imgSeq))
            using (Tensor txtN1 = LayerNormNoAffine(txt, txtSeq))
            using (Tensor imgMod1 = Modulate(imgN1, imgSeq, imgMod, 0, modIndex, out float[] imgGate1))
            using (Tensor txtMod1 = Modulate(txtN1, txtSeq, txtMod, 0, null, out float[] txtGate1))
            {
                (Tensor imgAttn, Tensor txtAttn) = JointAttention(imgMod1, txtMod1, imgSeq, txtSeq, b, rope);
                GatedAddInPlace(img, imgAttn, imgSeq, imgGate1, modIndex); imgAttn.Dispose();
                GatedAddInPlace(txt, txtAttn, txtSeq, txtGate1, null); txtAttn.Dispose();
            }
            Array.Copy(TensorToHost(img, (long)imgSeq * Dim), imgHost, (long)imgSeq * Dim); img.Dispose();
            Array.Copy(TensorToHost(txt, (long)txtSeq * Dim), txtHost, (long)txtSeq * Dim); txt.Dispose();
        }

        // Managed reference for one MLP sub-layer stream.
        internal void ManagedMlpSubLayer(float[] xHost, int seq, float[] mod, int[] modIndex, string mlpPrefix)
        {
            Tensor x = HostToTensor(xHost, seq, Dim);
            using (Tensor n2 = LayerNormNoAffine(x, seq))
            using (Tensor mod2 = Modulate(n2, seq, mod, 1, modIndex, out float[] gate2))
            using (Tensor mlp = GeGluMlp(mod2, mlpPrefix))
                GatedAddInPlace(x, mlp, seq, gate2, modIndex);
            Array.Copy(TensorToHost(x, (long)seq * Dim), xHost, (long)seq * Dim); x.Dispose();
        }

        // [seq*64] (DitRope) -> [seq*128] interleaved-duplicated (cos[2i]=cos[2i+1]=cos_i).
        private static float[] CosFull(float[] cos64, int seq)
        {
            int half = HeadDim / 2;     // 64
            var full = new float[(long)seq * HeadDim];
            for (int s = 0; s < seq; s++)
                for (int i = 0; i < half; i++)
                {
                    float v = cos64[(long)s * half + i];
                    full[(long)s * HeadDim + 2 * i] = v;
                    full[(long)s * HeadDim + 2 * i + 1] = v;
                }
            return full;
        }

        // Run one block's attention sub-layer via the native joint-attention kernel.
        // imgHost/txtHost ([seq*Dim]) are updated in place.
        internal unsafe bool NativeAttnSubLayer(float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            float[] imgMod, float[] txtMod, int[] modIndex, DitRope rope, int layer)
        {
            string b = $"transformer_blocks.{layer}";
            PrecomputeMod(imgMod, 0, modIndex, imgSeq, out var iS1, out var iSh, out var iG);
            PrecomputeMod(txtMod, 0, null, txtSeq, out var tS1, out var tSh, out var tG);
            float[] iCos = CosFull(rope.ImgCos, imgSeq), iSin = CosFull(rope.ImgSin, imgSeq);
            float[] tCos = CosFull(rope.TxtCos, txtSeq), tSin = CosFull(rope.TxtSin, txtSeq);

            fixed (float* img = imgHost, txt = txtHost, is1 = iS1, ish = iSh, ig = iG, ts1 = tS1, tsh = tSh, tg = tG,
                          ic = iCos, isn = iSin, tc = tCos, tsn = tSin)
            {
                var d = new QwenImageJointAttnArgs
                {
                    Img = (IntPtr)img, Txt = (IntPtr)txt,
                    ImgScale1 = (IntPtr)is1, ImgShift = (IntPtr)ish, ImgGate = (IntPtr)ig,
                    TxtScale1 = (IntPtr)ts1, TxtShift = (IntPtr)tsh, TxtGate = (IntPtr)tg,
                    ImgCos = (IntPtr)ic, ImgSin = (IntPtr)isn, TxtCos = (IntPtr)tc, TxtSin = (IntPtr)tsn,
                    ToQ = GgufW($"{b}.attn.to_q.weight", $"{b}.attn.to_q.bias"),
                    ToK = GgufW($"{b}.attn.to_k.weight", $"{b}.attn.to_k.bias"),
                    ToV = GgufW($"{b}.attn.to_v.weight", $"{b}.attn.to_v.bias"),
                    ToOut = GgufW($"{b}.attn.to_out.0.weight", $"{b}.attn.to_out.0.bias"),
                    AddQ = GgufW($"{b}.attn.add_q_proj.weight", $"{b}.attn.add_q_proj.bias"),
                    AddK = GgufW($"{b}.attn.add_k_proj.weight", $"{b}.attn.add_k_proj.bias"),
                    AddV = GgufW($"{b}.attn.add_v_proj.weight", $"{b}.attn.add_v_proj.bias"),
                    ToAddOut = GgufW($"{b}.attn.to_add_out.weight", $"{b}.attn.to_add_out.bias"),
                    NormQ = GgufF32Ptr($"{b}.attn.norm_q.weight"),
                    NormK = GgufF32Ptr($"{b}.attn.norm_k.weight"),
                    NormAq = GgufF32Ptr($"{b}.attn.norm_added_q.weight"),
                    NormAk = GgufF32Ptr($"{b}.attn.norm_added_k.weight"),
                    StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<QwenImageJointAttnArgs>(),
                    Dim = Dim, Heads = NumHeads, HeadDim = HeadDim, ImgSeq = imgSeq, TxtSeq = txtSeq, Eps = Eps,
                };
                return GgmlBasicOps.TryQwenImageJointAttn(in d);
            }
        }

        // Run a WHOLE block (attention + both MLP streams) via the single fused native
        // graph (TSGgml_QwenImageBlock). imgHost/txtHost ([seq*Dim]) updated in place.
        internal unsafe bool NativeBlock(float[] imgHost, int imgSeq, float[] txtHost, int txtSeq,
            float[] imgMod, float[] txtMod, int[] modIndex, DitRope rope, int layer)
        {
            string b = $"transformer_blocks.{layer}";
            // attn-half modulation (index 0) and mlp-half modulation (index 1)
            PrecomputeMod(imgMod, 0, modIndex, imgSeq, out var iS1a, out var iSha, out var iGa);
            PrecomputeMod(txtMod, 0, null, txtSeq, out var tS1a, out var tSha, out var tGa);
            PrecomputeMod(imgMod, 1, modIndex, imgSeq, out var iS1m, out var iShm, out var iGm);
            PrecomputeMod(txtMod, 1, null, txtSeq, out var tS1m, out var tShm, out var tGm);
            float[] iCos = CosFull(rope.ImgCos, imgSeq), iSin = CosFull(rope.ImgSin, imgSeq);
            float[] tCos = CosFull(rope.TxtCos, txtSeq), tSin = CosFull(rope.TxtSin, txtSeq);

            fixed (float* img = imgHost, txt = txtHost,
                          is1a = iS1a, isha = iSha, iga = iGa, ts1a = tS1a, tsha = tSha, tga = tGa,
                          is1m = iS1m, ishm = iShm, igm = iGm, ts1m = tS1m, tshm = tShm, tgm = tGm,
                          ic = iCos, isn = iSin, tc = tCos, tsn = tSin)
            {
                var d = new QwenImageBlockArgs
                {
                    Img = (IntPtr)img, Txt = (IntPtr)txt,
                    IS1a = (IntPtr)is1a, ISha = (IntPtr)isha, IGa = (IntPtr)iga,
                    TS1a = (IntPtr)ts1a, TSha = (IntPtr)tsha, TGa = (IntPtr)tga,
                    IS1m = (IntPtr)is1m, IShm = (IntPtr)ishm, IGm = (IntPtr)igm,
                    TS1m = (IntPtr)ts1m, TShm = (IntPtr)tshm, TGm = (IntPtr)tgm,
                    ICos = (IntPtr)ic, ISin = (IntPtr)isn, TCos = (IntPtr)tc, TSin = (IntPtr)tsn,
                    ToQ = GgufW($"{b}.attn.to_q.weight", $"{b}.attn.to_q.bias"),
                    ToK = GgufW($"{b}.attn.to_k.weight", $"{b}.attn.to_k.bias"),
                    ToV = GgufW($"{b}.attn.to_v.weight", $"{b}.attn.to_v.bias"),
                    ToOut = GgufW($"{b}.attn.to_out.0.weight", $"{b}.attn.to_out.0.bias"),
                    AddQ = GgufW($"{b}.attn.add_q_proj.weight", $"{b}.attn.add_q_proj.bias"),
                    AddK = GgufW($"{b}.attn.add_k_proj.weight", $"{b}.attn.add_k_proj.bias"),
                    AddV = GgufW($"{b}.attn.add_v_proj.weight", $"{b}.attn.add_v_proj.bias"),
                    ToAddOut = GgufW($"{b}.attn.to_add_out.weight", $"{b}.attn.to_add_out.bias"),
                    NormQ = GgufF32Ptr($"{b}.attn.norm_q.weight"),
                    NormK = GgufF32Ptr($"{b}.attn.norm_k.weight"),
                    NormAq = GgufF32Ptr($"{b}.attn.norm_added_q.weight"),
                    NormAk = GgufF32Ptr($"{b}.attn.norm_added_k.weight"),
                    INet0 = GgufWb($"{b}.img_mlp.net.0.proj.weight", $"{b}.img_mlp.net.0.proj.bias"),
                    INet2 = GgufWb($"{b}.img_mlp.net.2.weight", $"{b}.img_mlp.net.2.bias"),
                    TNet0 = GgufWb($"{b}.txt_mlp.net.0.proj.weight", $"{b}.txt_mlp.net.0.proj.bias"),
                    TNet2 = GgufWb($"{b}.txt_mlp.net.2.weight", $"{b}.txt_mlp.net.2.bias"),
                    StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<QwenImageBlockArgs>(),
                    Dim = Dim, Heads = NumHeads, HeadDim = HeadDim, Ff = 12288,
                    ImgSeq = imgSeq, TxtSeq = txtSeq, Eps = Eps,
                };
                return GgmlBasicOps.TryQwenImageBlock(in d);
            }
        }

        // Like GgufW but the bias may be [ff] (net0) or [dim] (net2) — length inferred natively.
        private QImgAttnW GgufWb(string weightName, string biasName) => GgufW(weightName, biasName);

        // Run one block's MLP sub-layer (one stream) via the native modulated-MLP kernel.
        internal unsafe bool NativeMlpSubLayer(float[] xHost, int seq, float[] mod, int[] modIndex, string mlpPrefix)
        {
            PrecomputeMod(mod, 1, modIndex, seq, out var s1, out var sh, out var g);
            var n0 = GgufW($"{mlpPrefix}.net.0.proj.weight", null);
            var n2 = GgufW($"{mlpPrefix}.net.2.weight", null);
            IntPtr n0b = GgufF32Ptr($"{mlpPrefix}.net.0.proj.bias");
            IntPtr n2b = GgufF32Ptr($"{mlpPrefix}.net.2.bias");
            fixed (float* x = xHost, sc = s1, shp = sh, gp = g)
            {
                var d = new QwenImageModMlpArgs
                {
                    X = (IntPtr)x, ScalePlus1 = (IntPtr)sc, Shift = (IntPtr)shp, Gate = (IntPtr)gp,
                    Net0W = n0.W, Net0Type = n0.Type, Net0Ne0 = n0.Ne0, Net0Ne1 = n0.Ne1, Net0Bytes = n0.Bytes, Net0B = n0b,
                    Net2W = n2.W, Net2Type = n2.Type, Net2Ne0 = n2.Ne0, Net2Ne1 = n2.Ne1, Net2Bytes = n2.Bytes, Net2B = n2b,
                    StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<QwenImageModMlpArgs>(),
                    Dim = Dim, Ff = 12288, Seq = seq, Eps = Eps,
                };
                return GgmlBasicOps.TryQwenImageModMlp(in d);
            }
        }
    }
}
