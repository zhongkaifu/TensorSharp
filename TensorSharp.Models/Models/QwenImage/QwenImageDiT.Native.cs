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

        // CFG-batched block path: run BOTH true-CFG branches (conditional + unconditional)
        // through one native dispatch that shares the per-block weights (TSGgml_QwenImageBlockCfg).
        // The denoise is launch-bound (GPU idle ~40% between the per-block sync + host round-trip),
        // so batching the branches doubles the work per dispatch and halves the per-block weight
        // upload + sync. Validated bit-exact vs the single-branch path and faster (captured-
        // combined ~1.9x denoise; non-persist combined ~1.1x), with a safe VRAM fallback, so
        // default-on. TS_QWEN_DIT_CFG_BATCH=0 forces the two-separate-forwards path.
        // CFG-batching's real win was the CUDA persist/CUDA-graph-capture path (replays the
        // combined block, ~1.9x vs the per-block weight-streaming forwards it was measured
        // against). On Metal the non-persist combined block runs both branches serially in one
        // (gallocr-packed) graph and measured ~2x SLOWER than two separate single-block
        // forwards, so it was always CUDA-only.
        //
        // SUPERSEDED by the whole-model resident-weight path: two whole-model captured
        // forwards measured 2.2x FASTER than the CFG-batched per-block path (1.77s vs 3.85s
        // per pair at imgSeq=468 — the per-block path re-uploads ~4.4 GB of weights per pair,
        // the whole-model path uploads none), and the per-block CFG captured graph also shows
        // a replay-drift defect (identical inputs, 5e-2 output drift on the first captured
        // replays) that the whole-model path does not (bitwise replay-stable). So prefer the
        // whole-model route whenever it is available; TS_QWEN_DIT_WHOLE=0 (which disables the
        // whole-model path) restores the old CFG-batched behavior.
        internal bool UseCfgBatch =>
            NativeBlockOn && _backend == BackendType.GgmlCuda && !WholeModelOn &&
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_CFG_BATCH") != "0";

        // CFG-batching only helps up to the token count where the captured-combined block fits
        // device VRAM. Above it, the combined kernel falls to the non-persist path, which holds
        // BOTH branches' activations at once and (at large token counts) oversubscribes VRAM ->
        // slower than two separate forwards. So gate the batched path to this budget; larger
        // images use the (unchanged) two-forward path. ~1536 (img+txt) tokens fits a 16 GB GPU;
        // raise on bigger GPUs via TS_QWEN_DIT_CFG_BATCH_MAXTOK.
        internal int CfgBatchMaxTokens
        {
            get
            {
                var v = Environment.GetEnvironmentVariable("TS_QWEN_DIT_CFG_BATCH_MAXTOK");
                return v != null && int.TryParse(v, out var n) && n > 0 ? n : 1536;
            }
        }

        // Per-step DiT block-loop wall-time print (isolates the DiT forward from the
        // fixed VAE/text-encoder cost). TS_QWEN_DIT_TIMING=1.
        internal static readonly bool TimingOn =
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_TIMING") == "1";

        // Raw GGUF weight (+optional bias) descriptor for the native kernels. When a
        // runtime LoRA is loaded and covers this weight, the descriptor also carries the
        // F32 factor pointers (honored by the whole-model forward path — see QImgAttnW).
        private QImgAttnW GgufW(string weightName, string biasName)
        {
            var info = DitGgufLocal.Tensors[weightName];
            DitGgufLocal.TryGetTensorDataPointer(info, out IntPtr wp);
            IntPtr bp = IntPtr.Zero;
            if (biasName != null && DitGgufLocal.Tensors.TryGetValue(biasName, out var binfo))
                DitGgufLocal.TryGetTensorDataPointer(binfo, out bp);
            long ne0 = (long)info.Shape[0];
            long ne1 = info.Shape.Length > 1 ? (long)info.Shape[1] : 1;
            var w = new QImgAttnW
            {
                W = wp,
                Type = (int)info.Type,
                Ne0 = ne0,
                Ne1 = ne1,
                Bytes = DitGgufLocal.GetTensorByteCount(info),
                B = bp,
            };
            if (_loraTable != null && _loraTable.TryGet(weightName, out var le))
            {
                if (le.In == ne0 && le.Out == ne1)
                {
                    w.LoraA = le.A; w.LoraB = le.B; w.LoraRank = le.Rank; w.LoraScale = le.Scale;
                }
                else
                {
                    Console.WriteLine($"  [lora] SKIP {weightName}: shape mismatch (lora {le.Out}x{le.Rank}x{le.In} vs weight {ne1}x{ne0})");
                }
            }
            return w;
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

        // Run one layer for BOTH true-CFG branches in a single native dispatch that shares the
        // per-block weights (TSGgml_QwenImageBlockCfg). imgHostC/txtHostC and imgHostN/txtHostN are
        // updated in place. The img/txt modulation is identical for both branches (same timestep +
        // layer weights), so it is computed once; only the txt content/length and txt rope differ.
        private void RunNativeLayerCfg(int layer,
            float[] imgHostC, int imgSeq, float[] txtHostC, int txtSeqC, DitRope ropeC,
            float[] imgHostN, float[] txtHostN, int txtSeqN, DitRope ropeN,
            Tensor temb, int[] modulateIndex)
        {
            string bn = $"transformer_blocks.{layer}";
            float[] imgMod = ModParams(temb, $"{bn}.img_mod.1.weight", $"{bn}.img_mod.1.bias");
            float[] txtMod = ModParams(temb, $"{bn}.txt_mod.1.weight", $"{bn}.txt_mod.1.bias");
            NativeBlockCfg(imgHostC, imgSeq, txtHostC, txtSeqC, ropeC,
                           imgHostN, txtHostN, txtSeqN, ropeN, imgMod, txtMod, modulateIndex, layer);
        }

        // CFG-batched whole block: builds the cond + neg block descriptors (sharing the same per-block
        // weights, img modulation, and img rope) and runs them in one native graph. Falls back to two
        // single NativeBlock calls if the combined kernel returns false. Updates all four host arrays.
        internal unsafe bool NativeBlockCfg(
            float[] imgHostC, int imgSeq, float[] txtHostC, int txtSeqC, DitRope ropeC,
            float[] imgHostN, float[] txtHostN, int txtSeqN, DitRope ropeN,
            float[] imgMod, float[] txtMod, int[] modIndex, int layer)
        {
            string b = $"transformer_blocks.{layer}";

            // Shared img modulation (attn idx 0 + mlp idx 1) and shared img rope (same for both branches).
            PrecomputeMod(imgMod, 0, modIndex, imgSeq, out var iS1a, out var iSha, out var iGa);
            PrecomputeMod(imgMod, 1, modIndex, imgSeq, out var iS1m, out var iShm, out var iGm);
            float[] iCos = CosFull(ropeC.ImgCos, imgSeq), iSin = CosFull(ropeC.ImgSin, imgSeq);

            // Per-branch txt modulation + txt rope (txt mod values are identical per row but the
            // sequence length differs, and the txt rope tables differ in length).
            PrecomputeMod(txtMod, 0, null, txtSeqC, out var tS1aC, out var tShaC, out var tGaC);
            PrecomputeMod(txtMod, 1, null, txtSeqC, out var tS1mC, out var tShmC, out var tGmC);
            PrecomputeMod(txtMod, 0, null, txtSeqN, out var tS1aN, out var tShaN, out var tGaN);
            PrecomputeMod(txtMod, 1, null, txtSeqN, out var tS1mN, out var tShmN, out var tGmN);
            float[] tCosC = CosFull(ropeC.TxtCos, txtSeqC), tSinC = CosFull(ropeC.TxtSin, txtSeqC);
            float[] tCosN = CosFull(ropeN.TxtCos, txtSeqN), tSinN = CosFull(ropeN.TxtSin, txtSeqN);

            var handles = new System.Collections.Generic.List<System.Runtime.InteropServices.GCHandle>(40);
            IntPtr Pin(float[] a)
            {
                var h = System.Runtime.InteropServices.GCHandle.Alloc(a, System.Runtime.InteropServices.GCHandleType.Pinned);
                handles.Add(h);
                return h.AddrOfPinnedObject();
            }
            try
            {
                var dc = new QwenImageBlockArgs
                {
                    Img = Pin(imgHostC), Txt = Pin(txtHostC),
                    IS1a = Pin(iS1a), ISha = Pin(iSha), IGa = Pin(iGa),
                    TS1a = Pin(tS1aC), TSha = Pin(tShaC), TGa = Pin(tGaC),
                    IS1m = Pin(iS1m), IShm = Pin(iShm), IGm = Pin(iGm),
                    TS1m = Pin(tS1mC), TShm = Pin(tShmC), TGm = Pin(tGmC),
                    ICos = Pin(iCos), ISin = Pin(iSin), TCos = Pin(tCosC), TSin = Pin(tSinC),
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
                    INet0 = GgufW($"{b}.img_mlp.net.0.proj.weight", $"{b}.img_mlp.net.0.proj.bias"),
                    INet2 = GgufW($"{b}.img_mlp.net.2.weight", $"{b}.img_mlp.net.2.bias"),
                    TNet0 = GgufW($"{b}.txt_mlp.net.0.proj.weight", $"{b}.txt_mlp.net.0.proj.bias"),
                    TNet2 = GgufW($"{b}.txt_mlp.net.2.weight", $"{b}.txt_mlp.net.2.bias"),
                    StructBytes = System.Runtime.InteropServices.Marshal.SizeOf<QwenImageBlockArgs>(),
                    Dim = Dim, Heads = NumHeads, HeadDim = HeadDim, Ff = 12288,
                    ImgSeq = imgSeq, TxtSeq = txtSeqC, Eps = Eps,
                };
                // The unconditional branch shares everything except its own img/txt streams,
                // txt modulation, txt rope, and txt length.
                var dn = dc;
                dn.Img = Pin(imgHostN); dn.Txt = Pin(txtHostN); dn.TxtSeq = txtSeqN;
                dn.TS1a = Pin(tS1aN); dn.TSha = Pin(tShaN); dn.TGa = Pin(tGaN);
                dn.TS1m = Pin(tS1mN); dn.TShm = Pin(tShmN); dn.TGm = Pin(tGmN);
                dn.TCos = Pin(tCosN); dn.TSin = Pin(tSinN);

                return GgmlBasicOps.TryQwenImageBlockCfg(in dc, in dn);
            }
            finally
            {
                foreach (var h in handles) h.Free();
            }
        }
    }
}
