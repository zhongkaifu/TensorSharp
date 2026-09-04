// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML
{

public enum GgmlBackendType
{
    Metal = 1,
    Cpu = 2,
    Cuda = 3,
    Vulkan = 4,
}

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView2D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly long RawBytes;

        public GgmlTensorView2D(IntPtr data, int dim0, int dim1, int stride0, int stride1, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Stride0 = stride0;
            Stride1 = stride1;
            RawBytes = rawBytes;
        }
    }

    [StructLayout(LayoutKind.Sequential)]
    internal readonly struct GgmlTensorView3D
    {
        public readonly IntPtr Data;
        public readonly int Dim0;
        public readonly int Dim1;
        public readonly int Dim2;
        public readonly int Stride0;
        public readonly int Stride1;
        public readonly int Stride2;
        public readonly long RawBytes;

        public GgmlTensorView3D(IntPtr data, int dim0, int dim1, int dim2, int stride0, int stride1, int stride2, long rawBytes)
        {
            Data = data;
            Dim0 = dim0;
            Dim1 = dim1;
            Dim2 = dim2;
            Stride0 = stride0;
            Stride1 = stride1;
            Stride2 = stride2;
            RawBytes = rawBytes;
        }
    }

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlTensorView4D
{
    public readonly IntPtr Data;
    public readonly int Ne0;
    public readonly int Ne1;
    public readonly int Ne2;
    public readonly int Ne3;
    public readonly long Nb1;
    public readonly long Nb2;
    public readonly long Nb3;
    public readonly long RawBytes;

    public GgmlTensorView4D(IntPtr data, int ne0, int ne1, int ne2, int ne3, long nb1, long nb2, long nb3, long rawBytes)
    {
        Data = data;
        Ne0 = ne0;
        Ne1 = ne1;
        Ne2 = ne2;
        Ne3 = ne3;
        Nb1 = nb1;
        Nb2 = nb2;
        Nb3 = nb3;
        RawBytes = rawBytes;
    }
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlContiguousTensor
{
    public readonly IntPtr Data;
    public readonly long ElementCount;
    public readonly int ElementType;

    public GgmlContiguousTensor(IntPtr data, long elementCount, DType elementType)
    {
        Data = data;
        ElementCount = elementCount;
        ElementType = (int)elementType;
    }
}

// Field order and packing must match MuseGlimmerVisionBlockDesc in
// ggml_ops_muse_glimmer_vision.cpp. Keep all pointer-sized fields first, then
// int64 shapes, int32 scalars, and floats so the native sizeof guard catches an
// accidental ABI drift.
[StructLayout(LayoutKind.Sequential)]
internal struct GgmlMuseGlimmerVisionBlockArgs
{
    public GgmlTensorView2D Hidden;

    public IntPtr Ln1W;
    public IntPtr Ln1B;
    public IntPtr QW;
    public IntPtr QB;
    public IntPtr KW;
    public IntPtr KB;
    public IntPtr VW;
    public IntPtr VB;
    public IntPtr OutW;
    public IntPtr OutB;

    public IntPtr Ln2W;
    public IntPtr Ln2B;
    public IntPtr UpW;
    public IntPtr UpB;
    public IntPtr DownW;
    public IntPtr DownB;

    public IntPtr PosW;
    public IntPtr PosH;
    public IntPtr WindowOffsets;

    public long QNe0;
    public long QNe1;
    public long QBytes;
    public long KNe0;
    public long KNe1;
    public long KBytes;
    public long VNe0;
    public long VNe1;
    public long VBytes;
    public long OutNe0;
    public long OutNe1;
    public long OutBytes;
    public long UpNe0;
    public long UpNe1;
    public long UpBytes;
    public long DownNe0;
    public long DownNe1;
    public long DownBytes;

    public int StructBytes;
    public int HiddenSize;
    public int IntermediateSize;
    public int NumTokens;
    public int NumHeads;
    public int HeadDim;
    public int WindowCount;
    public int IsGlobal;
    public int QType;
    public int KType;
    public int VType;
    public int OutType;
    public int UpType;
    public int DownType;

    public float Eps;
    public float RopeTheta;
}

[StructLayout(LayoutKind.Sequential)]
internal readonly struct GgmlQuantizedWeight
{
    public readonly IntPtr Data;
    public readonly int GgmlType;
    public readonly long Ne0;
    public readonly long Ne1;
    public readonly long RawBytes;

    public GgmlQuantizedWeight(IntPtr data, int ggmlType, long ne0, long ne1, long rawBytes)
    {
        Data = data;
        GgmlType = ggmlType;
        Ne0 = ne0;
        Ne1 = ne1;
        RawBytes = rawBytes;
    }
}

// Descriptor for the GPT-OSS whole-model decode kernel
// (TSGgml_GptOssModelDecode). Field order/types MUST match the native
// TSGgmlGptOssLayerDesc struct EXACTLY: 21 pointers, then 21 int64, then
// 21 int32, then 5 float. StructBytes is a sizeof() sanity check the native
// side validates before use.
[StructLayout(LayoutKind.Sequential)]
public struct GptOssLayerDecodeArgs
{
    // pointers (21)
    public IntPtr AttnNormW;
    public IntPtr QkvW;
    public IntPtr QkvB;
    public IntPtr KW;
    public IntPtr KB;
    public IntPtr VW;
    public IntPtr VB;
    public IntPtr OW;
    public IntPtr OB;
    public IntPtr KCache;
    public IntPtr VCache;
    public IntPtr Sinks;
    public IntPtr PostAttnNormW;
    public IntPtr GateInpW;
    public IntPtr GateInpB;
    public IntPtr GateExps;
    public IntPtr GateExpsB;
    public IntPtr UpExps;
    public IntPtr UpExpsB;
    public IntPtr DownExps;
    public IntPtr DownExpsB;

    // int64 weight shapes (21)
    public long QkvNe0, QkvNe1, QkvBytes;
    public long KNe0, KNe1, KBytes;
    public long VNe0, VNe1, VBytes;
    public long ONe0, ONe1, OBytes;
    public long GeNe0, GeNe1, GeBytes;
    public long UeNe0, UeNe1, UeBytes;
    public long DeNe0, DeNe1, DeBytes;

    // int32 scalars (21)
    public int StructBytes;
    public int HiddenSize;
    public int NumHeads;
    public int NumKvHeads;
    public int HeadDim;
    public int CacheSize;
    public int IsSwa;
    public int SlidingWindow;
    public int RopeNDims;
    public int OrigCtxLen;
    public int KvCacheType;
    public int NumExperts;
    public int NumExpertsUsed;
    public int SeparateQkv;
    public int QkvType, KType, VType, OType;
    public int GeType, UeType, DeType;
    /// <summary>Non-zero keeps this layer's routed experts in system RAM and runs its
    /// MoE FFN on the host (MoeCpuOffloadConfig / --n-cpu-moe).</summary>
    public int CpuMoe;

    // float scalars (5)
    public float Eps;
    public float RopeBase;
    public float RopeFreqScale;
    public float OaiAlpha;
    public float OaiLimit;

    // Optional F16 prefill-GEMM weight copies (may be zero). Only the PREFILL
    // kernel reads these: prefill is compute-bound and F16 tensor-core GEMMs
    // beat the quantized MMQ path at large token counts, while decode stays on
    // the small quantized reads. Populated when TS_GPTOSS_PREFILL_F16=1.
    public IntPtr QkvWF16;
    public IntPtr KWF16;
    public IntPtr VWF16;
    public IntPtr OWF16;
    public IntPtr GateExpsF16;
    public IntPtr UpExpsF16;
    public IntPtr DownExpsF16;
}

// Descriptor for the fused single-layer Gemma 4 MoE decode kernel
// (TSGgml_Gemma4MoELayerDecode). Field order/types MUST match the native
// TSGgmlGemma4MoELayerDesc struct EXACTLY: all 8-byte fields (pointers then
// int64) first, then 4-byte fields (int32 then float). StructBytes is a
// sizeof() sanity check the native side validates before use.
[StructLayout(LayoutKind.Sequential)]
public struct Gemma4MoELayerDecodeArgs
{
    // pointers (24)
    public IntPtr Hidden;
    public IntPtr AttnNormW;
    public IntPtr QkvW;
    public IntPtr KW;
    public IntPtr VW;
    public IntPtr QNormW;
    public IntPtr KNormW;
    public IntPtr OW;
    public IntPtr PostAttnNormW;
    public IntPtr KCache;
    public IntPtr VCache;
    public IntPtr FreqFactors;
    public IntPtr FfnNormW;
    public IntPtr GuW;
    public IntPtr DownW;
    public IntPtr PostFfwNorm1W;
    public IntPtr GateInpW;
    public IntPtr GateInpScale;
    public IntPtr PreFfwNorm2W;
    public IntPtr GateUpExps;
    public IntPtr DownExps;
    public IntPtr DownExpsScale;
    public IntPtr PostFfwNorm2W;
    public IntPtr PostFfwNormW;

    // int64 weight shapes (24)
    public long QkvNe0, QkvNe1, QkvBytes;
    public long KNe0, KNe1, KBytes;
    public long VNe0, VNe1, VBytes;
    public long ONe0, ONe1, OBytes;
    public long GuNe0, GuNe1, GuBytes;
    public long DownNe0, DownNe1, DownBytes;
    public long GueNe0, GueNe1, GueBytes;
    public long DeNe0, DeNe1, DeBytes;

    // int32 scalars / shapes (24)
    public int StructBytes;
    public int HiddenSize;
    public int NumHeads;
    public int NumKvHeads;
    public int HeadDim;
    public int CacheSize;
    public int IsLocal;
    public int IsShared;
    public int SlidingWindow;
    public int Position;
    public int RopeNDims;
    public int KvCacheType;
    public int NumExperts;
    public int NumExpertsUsed;
    public int FreqFactorsLen;
    public int QkvType;
    public int KType;
    public int VType;
    public int OType;
    public int GuType;
    public int DownType;
    public int GueType;
    public int DeType;
    public int SeparateQkv;
    /// <summary>Non-zero keeps this layer's routed experts in system RAM and runs its
    /// MoE FFN on the host (MoeCpuOffloadConfig / --n-cpu-moe).</summary>
    public int CpuMoe;

    // float scalars (4)
    public float Eps;
    public float RopeBase;
    public float InvSqrtHidden;
    public float LayerOutputScale;
}

    /// <summary>
    /// Mirrors TSGgmlQwen4ExpPleArgs in ggml_ops_qwen4exp.cpp - the PLE block run
    /// inside the span; only the n-gram hash and the table gather stay host-side.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen4ExpPleArgs
    {
        public IntPtr KeyW, ValueW, NormKey, NormQuery, NormConv, Conv1dT, ConvState;
        public long KeyBytes, ValueBytes;
        public int KeyType, ValueType, Kern, Dil;
    }

    /// <summary>
    /// Mirrors TSGgmlQwen4ExpHeadArgs in ggml_ops_qwen4exp.cpp - the final
    /// hyper-connection mixer (which IS the output norm) plus the LM head.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen4ExpHeadArgs
    {
        public IntPtr HcNorm, HcDown, HcUp, Head;
        public long HcDownBytes, HcUpBytes, HeadBytes;
        public int HcDownType, HcUpType, HeadType;
        public int Vocab;
    }

    /// <summary>
    /// Mirrors TSGgmlQwen4ExpAttnArgs in ggml_ops_qwen4exp.cpp - the full-attention
    /// half of a qwen4exp layer. Pointers first, then int64, then int32.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen4ExpAttnArgs
    {
        public IntPtr HcNorm, HcDown, HcUp, HcInject;
        public IntPtr Wq, Wk, Wv, Wo;
        public IntPtr QNorm, KNorm;
        public IntPtr KCache, VCache;

        public long HcDownBytes, HcUpBytes, HcInjectBytes;
        public long WqBytes, WkBytes, WvBytes, WoBytes;
        public long KvBytes;

        public int HcDownType, HcUpType, HcInjectType;
        public int WqType, WkType, WvType, WoType;
        public int KvType;
    }

    /// <summary>
    /// Mirrors TSGgmlQwen4ExpGdnArgs in ggml_ops_qwen4exp.cpp - the recurrent half
    /// of a qwen4exp layer. Pointers first, then int64, then int32.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen4ExpGdnArgs
    {
        public IntPtr HcNorm, HcDown, HcUp, HcInject;
        public IntPtr Qkv, Gate, Beta, Alpha;
        public IntPtr Conv1d, SsmDt, SsmA, SsmNorm, OutProj;
        public IntPtr ConvState, SsmState;

        public long HcDownBytes, HcUpBytes, HcInjectBytes;
        public long QkvBytes, GateBytes, BetaBytes, AlphaBytes, OutProjBytes;

        public int HcDownType, HcUpType, HcInjectType;
        public int QkvType, GateType, BetaType, AlphaType, OutProjType;
    }

    /// <summary>
    /// Mirrors TSGgmlQwen4ExpFfnArgs in ggml_ops_qwen4exp.cpp. Pointers first,
    /// then int64, then int32 - append within a run rather than reordering.
    /// </summary>
    [StructLayout(LayoutKind.Sequential)]
    public struct Qwen4ExpFfnArgs
    {
        public IntPtr HcNorm, HcDown, HcUp, HcInject;
        public IntPtr Router, GateExps, UpExps, DownExps;
        public IntPtr ShGateInp, ShGate, ShUp, ShDown;

        public long HcDownBytes, HcUpBytes, HcInjectBytes;
        public long RouterBytes, GateExpsBytes, UpExpsBytes, DownExpsBytes;
        public long ShGateBytes, ShUpBytes, ShDownBytes;

        public int HcDownType, HcUpType, HcInjectType;
        public int RouterType, GateExpsType, UpExpsType, DownExpsType;
        public int ShGateType, ShUpType, ShDownType;
    }

// Descriptor for the Qwen3.5/3.6 full-model decode kernel
// (TSGgml_Qwen35ModelDecode). Field order/types MUST match the native
// TSGgmlQwen35LayerDesc struct EXACTLY: 23 pointers, then 27 int64, then 13 int32.
[StructLayout(LayoutKind.Sequential)]
public struct Qwen35LayerDecodeArgs
{
    // pointers (23)
    public IntPtr AttnNormW;
    public IntPtr PostAttnNormW;
    public IntPtr QkvW;
    public IntPtr QNormW;
    public IntPtr KNormW;
    public IntPtr OW;
    public IntPtr KCache;
    public IntPtr VCache;
    public IntPtr GdnQkvW;
    public IntPtr GdnGateW;
    public IntPtr SsmBetaW;
    public IntPtr SsmAlphaW;
    public IntPtr Conv1dW;
    public IntPtr SsmDtW;
    public IntPtr SsmAW;
    public IntPtr SsmNormW;
    public IntPtr SsmOutW;
    public IntPtr ConvStateIn;
    public IntPtr DeltaStateIn;
    public IntPtr ConvStateOut;
    public IntPtr DeltaStateOut;
    public IntPtr GuW;
    public IntPtr DownW;
    public IntPtr KW;
    public IntPtr VW;
    public IntPtr GateInpW;
    public IntPtr GateExps;
    public IntPtr UpExps;
    public IntPtr DownExps;
    public IntPtr ShexpGateW;
    public IntPtr ShexpUpW;
    public IntPtr ShexpDownW;
    public IntPtr ShexpGateInpW;
    /// <summary>Dense FFN with gate and up UNFUSED, for the mixed-quant "UD"
    /// layers where the two tensors have different GGML types and no
    /// imatrix-free requantization can bring them together. Non-zero exactly
    /// when <see cref="GuW"/> is zero; the graph then runs two matmuls.</summary>
    public IntPtr FfnGateW;
    public IntPtr FfnUpW;
    /// <summary>Optional host pointer to 16 F32 per-projection matmul-output
    /// scales (NVFP4 per-tensor scale2 sidecars; native TSQ35_SC_* slot order).
    /// Zero when no projection of this layer carries a scale.</summary>
    public IntPtr ProjScales;

    // int64 weight shapes
    public long QkvNe0, QkvNe1, QkvBytes;
    public long ONe0, ONe1, OBytes;
    public long KNe0, KNe1, KBytes;
    public long VNe0, VNe1, VBytes;
    public long GdnQkvNe0, GdnQkvNe1, GdnQkvBytes;
    public long GdnGateNe0, GdnGateNe1, GdnGateBytes;
    public long SsmBetaNe0, SsmBetaNe1, SsmBetaBytes;
    public long SsmAlphaNe0, SsmAlphaNe1, SsmAlphaBytes;
    public long SsmOutNe0, SsmOutNe1, SsmOutBytes;
    public long GuNe0, GuNe1, GuBytes;
    public long DownNe0, DownNe1, DownBytes;
    public long GateInpNe0, GateInpNe1, GateInpBytes;
    public long GateExpsBytes, UpExpsBytes, DownExpsBytes;
    public long ShexpGateNe0, ShexpGateNe1, ShexpGateBytes;
    public long ShexpUpNe0, ShexpUpNe1, ShexpUpBytes;
    public long ShexpDownNe0, ShexpDownNe1, ShexpDownBytes;
    public long FfnGateNe0, FfnGateNe1, FfnGateBytes;
    public long FfnUpNe0, FfnUpNe1, FfnUpBytes;

    // int32 scalars
    public int StructBytes;
    public int IsRecurrent;
    public int IsMoe;
    public int QkvType, OType;
    public int GdnQkvType, GdnGateType, SsmBetaType, SsmAlphaType, SsmOutType;
    public int GuType, DownType;
    public int FfDense;
    public int SeparateQkv, KType, VType;
    public int GateInpType, GateExpsType, UpExpsType, DownExpsType;
    public int ShexpGateType, ShexpUpType, ShexpDownType;
    /// <summary>Non-zero keeps this layer's routed experts in system RAM and runs its
    /// MoE FFN on the host (MoeCpuOffloadConfig / --n-cpu-moe).</summary>
    public int CpuMoe;
    public int FfnGateType;
    public int FfnUpType;
}

// Descriptor for the fused DiffusionGemma decode-layer kernel
// (TSGgml_DiffusionDecodeLayer). Field order/types MUST match the native
// TSGgmlDiffusionDecodeLayerDesc struct EXACTLY.
[StructLayout(LayoutKind.Sequential)]
public struct DiffusionDecodeLayerArgs
{
    // pointers (25)
    public IntPtr Hidden;
    public IntPtr AttnNormW;
    public IntPtr QW;
    public IntPtr KW;
    public IntPtr VW;
    public IntPtr QNormW;
    public IntPtr KNormW;
    public IntPtr OW;
    public IntPtr PostAttnNormW;
    public IntPtr PromptK;
    public IntPtr PromptV;
    public IntPtr FreqFactors;
    public IntPtr FfnNormW;
    public IntPtr GateW;
    public IntPtr UpW;
    public IntPtr DownW;
    public IntPtr PostFfwNorm1W;
    public IntPtr GateInpW;
    public IntPtr GateInpScale;
    public IntPtr PreFfwNorm2W;
    public IntPtr GateUpExps;
    public IntPtr DownExps;
    public IntPtr DownExpsScale;
    public IntPtr PostFfwNorm2W;
    public IntPtr PostFfwNormW;

    // int64 weight shapes (27)
    public long QNe0, QNe1, QBytes;
    public long KNe0, KNe1, KBytes;
    public long VNe0, VNe1, VBytes;
    public long ONe0, ONe1, OBytes;
    public long GateNe0, GateNe1, GateBytes;
    public long UpNe0, UpNe1, UpBytes;
    public long DownNe0, DownNe1, DownBytes;
    public long GueNe0, GueNe1, GueBytes;
    public long DeNe0, DeNe1, DeBytes;

    // int32 scalars / shapes (23)
    public int StructBytes;
    public int HiddenSize;
    public int CanvasLen;
    public int PromptLen;
    public int NumHeads;
    public int NumKvHeads;
    public int HeadDim;
    public int IsLocal;
    public int HasVProj;
    public int SlidingWindow;
    public int RopeNDims;
    public int NumExperts;
    public int NumExpertsUsed;
    public int FreqFactorsLen;
    public int QType, KType, VType, OType;
    public int GateType, UpType, DownType;
    public int GueType, DeType;
    /// <summary>Non-zero keeps this layer's routed experts in system RAM and runs its
    /// MoE FFN on the host (MoeCpuOffloadConfig / --n-cpu-moe).</summary>
    public int CpuMoe;

    // float scalars (4)
    public float Eps;
    public float RopeBase;
    public float InvSqrtHidden;
    public float DecScale;
}

// Descriptor for the fused Qwen-Image DiT modulated-MLP kernel
// (TSGgml_QwenImageModMlp). Field order/types MUST match the native
// TSGgmlQwenImageModMlpDesc struct EXACTLY.
[StructLayout(LayoutKind.Sequential)]
public struct QwenImageModMlpArgs
{
    public IntPtr X;
    public IntPtr ScalePlus1;
    public IntPtr Shift;
    public IntPtr Gate;
    public IntPtr Net0W; public int Net0Type; public long Net0Ne0, Net0Ne1, Net0Bytes;
    public IntPtr Net0B;
    public IntPtr Net2W; public int Net2Type; public long Net2Ne0, Net2Ne1, Net2Bytes;
    public IntPtr Net2B;
    public int StructBytes;
    public int Dim;
    public int Ff;
    public int Seq;
    public float Eps;
}

// One projection weight (+ optional bias) for the joint-attention kernel.
// MUST match native TSGImgAttnW exactly.
[StructLayout(LayoutKind.Sequential)]
public struct QImgAttnW
{
    public IntPtr W;
    public int Type;
    public long Ne0, Ne1, Bytes;
    public IntPtr B;
    // Optional runtime LoRA side-path: y = W·x + b + LoraScale * B·(A·x), computed in
    // F32 next to the quantized base matmul (a LoRA merged into 2-bit weights is
    // swallowed by requantization noise — the deltas are far below the quant step).
    // LoraA = [rank, ne0] row-major F32 (lora_down), LoraB = [ne1, rank] row-major F32
    // (lora_up); both must be STABLE allocations (resident-cached by pointer).
    // Zero/default = no LoRA. Currently honored by the whole-model forward path.
    public IntPtr LoraA;
    public IntPtr LoraB;
    public long LoraRank;
    public float LoraScale;
}

// Descriptor for the fused Qwen-Image DiT joint-attention sub-layer
// (TSGgml_QwenImageJointAttn). MUST match native TSGgmlQwenImageJointAttnDesc.
[StructLayout(LayoutKind.Sequential)]
public struct QwenImageJointAttnArgs
{
    public IntPtr Img, Txt;
    public IntPtr ImgScale1, ImgShift, ImgGate;
    public IntPtr TxtScale1, TxtShift, TxtGate;
    public IntPtr ImgCos, ImgSin, TxtCos, TxtSin;
    public QImgAttnW ToQ, ToK, ToV, ToOut;
    public QImgAttnW AddQ, AddK, AddV, ToAddOut;
    public IntPtr NormQ, NormK, NormAq, NormAk;
    public int StructBytes, Dim, Heads, HeadDim, ImgSeq, TxtSeq;
    public float Eps;
}

// Descriptor for a single device 2D convolution (TSGgml_Conv2d), used to move the
// Qwen-Image VAE conv stack off the CPU. MUST match native TSGgmlConv2dDesc exactly.
[StructLayout(LayoutKind.Sequential)]
public struct Conv2dArgs
{
    public IntPtr Input; public int W, H, C;
    public IntPtr Weight; public int WType, KW, KH, IC, OC;
    public long WeightBytes;
    public IntPtr Bias;
    public IntPtr Output;
    public int StrideW, StrideH, PadL, PadR, PadT, PadB;
    public int StructBytes;
}

// One op of the fused whole-VAE graph (TSGgml_QwenVaeRun). MUST match native TSGVaeOp.
// Kinds: 0 conv, 1 channel-RMS-norm (*gamma at W), 2 silu, 3 nearest-upsample x2,
// 4 save (slots[Dst] = slots[Src], alias), 5 add (slots[Dst] = slots[Src] + slots[Aux]),
// 6 spatial single-head attention (slots[Src] = qkv [W,H,3C], Oc = C).
[StructLayout(LayoutKind.Sequential)]
public struct QwenVaeOp
{
    public int Kind;
    public int W, B;                       // weight / bias table indices (-1 = none)
    public int Oc, Ic, Kh, Kw;
    public int Sh, Sw, Pt, Pb, Pl, Pr;
    public int Src, Dst, Aux;
}

// Stable F32 host pointer for a fused-VAE weight. MUST match native TSGVaeWeightRef.
[StructLayout(LayoutKind.Sequential)]
public struct QwenVaeWeightRef
{
    public IntPtr Data;
    public long Bytes;
}

// Descriptor for the whole fused VAE encode/decode graph (TSGgml_QwenVaeRun).
// MUST match native TSGgmlQwenVaeDesc exactly.
[StructLayout(LayoutKind.Sequential)]
public struct QwenVaeArgs
{
    public IntPtr Input; public int InW, InH, InC;
    public IntPtr Output; public long OutLen;
    public IntPtr Ops; public int NumOps;
    public IntPtr Weights; public int NumWeights;
    public int StructBytes;
}

// One layer of the fused conditioning-encoder trunk (TSGgml_QwenTeTrunk).
// MUST match native TSGTeLayerW. MaskKind: 0 full, 1 causal, 2 uploaded window mask.
[StructLayout(LayoutKind.Sequential)]
public struct QwenTeLayerW
{
    public IntPtr Ln1, Ln2;                              // [hidden] F32 (stable ptrs)
    public QImgAttnW Q, K, V, O, Gate, Up, Down;         // .B = optional F32 bias
    public int MaskKind;
    public int Pad;
}

// Descriptor for the fused transformer trunk (TSGgml_QwenTeTrunk): the Qwen2.5-VL
// text-encoder LLM (GQA, causal) and vision tower (MHA, window masks) run their
// whole layer stack as ONE graph. MUST match native TSGgmlQwenTeTrunkDesc.
[StructLayout(LayoutKind.Sequential)]
public struct QwenTeTrunkArgs
{
    public IntPtr X;                 // [hidden, seq] F32 input states
    public IntPtr Out;               // [hidden, seq] F32 output (post final norm)
    public IntPtr CosF, SinF;        // [head_dim, seq] F32 rotate-half tables
    public IntPtr WinMask;           // [seq, seq] F32 additive window mask (or zero)
    public IntPtr FinalNorm;         // [hidden] F32 (or zero = skip)
    public IntPtr Layers; public int NumLayers;
    public int StructBytes, Hidden, Heads, KvHeads, HeadDim, Seq;
    public float Eps;
}

// Descriptor for the whole fused DiT block (attn + both MLP streams in one graph)
// (TSGgml_QwenImageBlock). MUST match native TSGgmlQwenImageBlockDesc exactly.
[StructLayout(LayoutKind.Sequential)]
public struct QwenImageBlockArgs
{
    public IntPtr Img, Txt;
    public IntPtr IS1a, ISha, IGa, TS1a, TSha, TGa;   // attn modulation (mod index 0)
    public IntPtr IS1m, IShm, IGm, TS1m, TShm, TGm;   // mlp modulation (mod index 1)
    public IntPtr ICos, ISin, TCos, TSin;
    public QImgAttnW ToQ, ToK, ToV, ToOut;
    public QImgAttnW AddQ, AddK, AddV, ToAddOut;
    public IntPtr NormQ, NormK, NormAq, NormAk;
    public QImgAttnW INet0, INet2, TNet0, TNet2;       // mlp weights (+bias in .B)
    public int StructBytes, Dim, Heads, HeadDim, Ff, ImgSeq, TxtSeq;
    public float Eps;
}

// Per-block weight set for the whole-DiT forward (TSGgml_QwenImageForward).
// MUST match native TSGImgBlockW exactly.
[StructLayout(LayoutKind.Sequential)]
public struct QImgBlockW
{
    public QImgAttnW ImgMod, TxtMod;                   // [dim, 6*dim] (+bias)
    public QImgAttnW ToQ, ToK, ToV, ToOut;
    public QImgAttnW AddQ, AddK, AddV, ToAddOut;
    public IntPtr NormQ, NormK, NormAq, NormAk;        // [head_dim] f32
    public QImgAttnW INet0, INet2, TNet0, TNet2;       // mlp (+bias in .B)
}

// Descriptor for the 60-block DiT body in one resident-weight graph
// (TSGgml_QwenImageForward). Img/Txt are the post-prelude residual streams (in/out),
// so the C# img_in/txt_in/norm_out/proj_out stay shared with the per-block path.
// MUST match native TSGgmlQwenImageForwardDesc exactly.
[StructLayout(LayoutKind.Sequential)]
public struct QwenImageForwardArgs
{
    public IntPtr Img, Txt, Temb, ImgCos, ImgSin, TxtCos, TxtSin, ModulateIndex;
    public IntPtr Blocks;                              // -> QImgBlockW[NumLayers]
    public int StructBytes, Dim, Heads, HeadDim, Ff, ImgSeq, TxtSeq, NumLayers;
    public float Eps;
}

// One (possibly quantized) weight matrix + optional F32 bias for the Wan video
// kernels. MUST match native TSGWanW exactly.
[StructLayout(LayoutKind.Sequential)]
public struct WanW
{
    public IntPtr W;
    public int Type;
    public int Reserved;
    public long Ne0, Ne1;      // ggml dims: ne0 = input dim, ne1 = output dim
    public long Bytes;
    public IntPtr B;           // [Ne1] F32 bias or zero
}

// One UMT5 encoder layer. MUST match native TSGWanT5LayerW.
[StructLayout(LayoutKind.Sequential)]
public struct WanT5LayerW
{
    public WanW Q, K, V, O;
    public WanW Gate, Up, Down;      // wi_0 / wi_1 / wo
    public IntPtr AttnNorm;          // [dim] F32 RMS gain
    public IntPtr FfnNorm;           // [dim] F32 RMS gain
    public IntPtr RelB;              // [heads, 32] F32 relative-attention bias
}

// Whole UMT5 encoder forward (TSGgml_WanT5Encode). MUST match native TSGgmlWanT5Desc.
[StructLayout(LayoutKind.Sequential)]
public struct WanT5EncodeArgs
{
    public IntPtr Tokens;            // I32 [NTokens]
    public IntPtr RelBucket;         // I32 [NTokens * NTokens], bucket[q*n + k]
    public IntPtr AttnMask;          // F32 [NTokens] additive (0 / -inf) or zero
    public WanW TokEmbd;
    public IntPtr Layers;            // -> WanT5LayerW[NumLayers]
    public IntPtr FinalNorm;         // [dim] F32
    public IntPtr Out;               // F32 [dim * NTokens] written
    public int NTokens, NumLayers, Dim, Ff, Heads, HeadDim;
    public float Eps;
    public int StructBytes;
}

// One Wan DiT block. MUST match native TSGWanDitBlockW.
[StructLayout(LayoutKind.Sequential)]
public struct WanDitBlockW
{
    public IntPtr Modulation;        // [6*dim] F32
    public WanW SQ, SK, SV, SO;      // self-attention (+bias)
    public IntPtr SNormQ, SNormK;    // [dim] F32 full-dim RMS gains
    public IntPtr Norm3W, Norm3B;    // [dim] F32 cross-attn LayerNorm affine
    public WanW XQ, XK, XV, XO;      // cross-attention (+bias)
    public IntPtr XNormQ, XNormK;    // [dim] F32
    public WanW Ffn0, Ffn2;          // FFN (+bias)
}

// One Wan DiT denoising-step forward (TSGgml_WanDitForward). MUST match native
// TSGgmlWanDitDesc.
[StructLayout(LayoutKind.Sequential)]
public struct WanDitForwardArgs
{
    public IntPtr X;                 // F32 [in_tok, Seq] patchified tokens (in)
    public IntPtr Out;               // F32 [out_tok, Seq] velocity tokens (out)
    public IntPtr Context;           // F32 [TextDim, CtxLen]
    public IntPtr TSin;              // F32 [FreqDim] sinusoidal timestep embedding
    public IntPtr TSin0;             // F32 [FreqDim] sinusoid for tokens [0, Seq0) or zero
    public IntPtr CosF, SinF;        // F32 [HeadDim, Seq] pair-duplicated RoPE tables
    public WanW Patch;               // [in_tok, dim] (+bias)
    public WanW Text0, Text2;
    public WanW Time0, Time2;
    public WanW TProj;
    public WanW Head;
    public IntPtr HeadMod;           // [2*dim] F32
    public IntPtr Blocks;            // -> WanDitBlockW[NumLayers]
    // Seq0 > 0: tokens [0, Seq0) are AdaLN-modulated with TSin0's timestep (Wan 2.2
    // TI2V i2v conditions the first latent frame at timestep 0). 0 = uniform.
    public int NumLayers, Dim, Ff, Heads, HeadDim, Seq, CtxLen, FreqDim, TextDim, Seq0;
    public float Eps;
    public int StructBytes;
}

// One causal conv3d of the Wan VAE, pre-sliced by temporal tap on the managed side.
// MUST match native TSGWanVaeConv.
[StructLayout(LayoutKind.Sequential)]
public struct WanVaeConv
{
    public IntPtr Tap0, Tap1, Tap2;  // [k, k, ic, oc] kernels (Tap1/2 zero when Kd == 1)
    public IntPtr Bias;              // [oc] F32 or zero
    public int Kd, K, Ic, Oc;
    // 1 = taps are F16 (pre-converted at load with the same round-to-nearest
    // the graph's F32->F16 cast applied), 0 = legacy F32 taps cast in-graph.
    // F16 halves the resident weight bytes and removes one cast node per conv
    // tap per chunk from every VAE graph.
    public int TapType;
    public int Reserved2;
}

[StructLayout(LayoutKind.Sequential)]
public struct WanVaeNorm
{
    public IntPtr Gamma;             // [C] F32
    public int C, Pad;
}

[StructLayout(LayoutKind.Sequential)]
public struct WanVaeResBlockW
{
    public WanVaeNorm N0, N3;
    public WanVaeConv C2, C6;
    public WanVaeConv Shortcut;      // Tap0 == zero => identity
}

[StructLayout(LayoutKind.Sequential)]
public struct WanVaeAttnW
{
    public WanVaeNorm Norm;
    public IntPtr QkvW;              // [c, 3c] F32
    public IntPtr QkvB;              // [3c] F32
    public IntPtr ProjW;             // [c, c] F32
    public IntPtr ProjB;             // [c] F32
    public int C, Pad;
}

[StructLayout(LayoutKind.Sequential)]
public struct WanVaeUpsampleW
{
    public WanVaeConv TimeConv;      // (3,1,1) dim -> 2*dim; Tap0 == zero => spatial-only
    public WanVaeConv SConv;         // 3x3 2D conv after nearest x2
}

// Whole Wan VAE decode (TSGgml_WanVaeDecode). MUST match native
// TSGgmlWanVaeDecodeDesc exactly (Res0..Res11 = 4 scales x 3 residual blocks).
[StructLayout(LayoutKind.Sequential)]
public struct WanVaeDecodeArgs
{
    public IntPtr Z;                 // F32 [zw, zh, zc, zt] ([W,H,C,T] layout)
    public IntPtr Out;               // F32 [zw*8*patch, zh*8*patch, 3, 1+(zt-1)*4] written
    public long OutLen;
    public WanVaeConv Conv2;         // post-quant 1x1x1
    public WanVaeConv Conv1;         // 3x3x3 (zc -> 384 / 1024)
    public WanVaeResBlockW Mid0, Mid2;
    public WanVaeAttnW Mid1;
    public WanVaeResBlockW Res0, Res1, Res2, Res3, Res4, Res5, Res6, Res7, Res8, Res9, Res10, Res11;
    public WanVaeUpsampleW Up0, Up1, Up2;
    public WanVaeNorm HeadNorm;
    public WanVaeConv HeadConv;      // 3x3x3 (96 -> 3, or 256 -> 12 for wan2.2)
    public int Zw, Zh, Zt, Zc;
    public int Version;              // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V
    public int PatchSize;            // pixel unpatchify factor (2 for wan2.2, else 1)
    public int StructBytes;
}

// ---------------------------------------------------------------------------
// MiniMax-H3. MUST match the structs in ggml_ops_minimax_h3.cpp exactly.
// ---------------------------------------------------------------------------

/// <summary>One linear layer. Ne0/Ne1 are ggml order: Ne0 = in_features,
/// Ne1 = out_features. The weight keeps its on-disk dtype (F16 for the VAEs);
/// the bias, when present, is always F32.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Lin
{
    public IntPtr W;
    public IntPtr B;              // IntPtr.Zero = no bias
    public long Ne0, Ne1;
    public long Bytes;            // weight byte count
    public int Type;              // ggml_type of the weight
    public int Pad;
}

/// <summary>One MiniMax-H3 video-VAE ViT decoder block.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VitBlockW
{
    public IntPtr Norm1;          // [dim] F32, RMSNorm affine
    public IntPtr Norm2;
    public IntPtr Scale1;         // [dim] F32, LayerScale on the attention branch
    public IntPtr Scale2;
    public H3Lin Qkv;             // [dim, 3*dim], per-head interleaved
    public H3Lin Out;             // [dim, dim]
    public H3Lin W1;              // [dim, 2*inner] (gate | value)
    public H3Lin W2;              // [inner, dim]
}

/// <summary>Whole MiniMax-H3 video VAE ViT decode (TSGgml_MiniMaxH3VideoVaeDecode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VideoVaeDecodeArgs
{
    public int StructBytes;
    public int NumBlocks;

    public IntPtr Latent;         // F32 [latentC, tokens], tokens in (t,h,w) order
    public IntPtr Out;            // F32 [patchDim, tokens] written
    public IntPtr Cos;            // F32 [rotDim, tokens + numRegister + 1]
    public IntPtr Sin;
    public IntPtr RegisterTokens; // F32 [dim, numRegister]
    public IntPtr NormOutW;       // F32 [dim]
    public IntPtr NormOutB;       // F32 [dim]

    public H3Lin PostQuant;       // the 1x1x1 conv, as a per-token linear
    public H3Lin XEmbedder;
    public H3Lin ProjOut;

    public IntPtr Blocks;         // H3VitBlockW[NumBlocks]

    public int Tokens;
    public int LatentC;
    public int Dim;
    public int Heads;
    public int HeadDim;
    public int Inner;
    public int RotDim;
    public int NumRegister;
    public int PatchDim;
    public float Eps;
}

/// <summary>One Qwen3-VL text-encoder layer for MiniMax-H3.
/// MUST match native TSGH3TeLayerW.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3TeLayerW
{
    public IntPtr InputNorm;      // [hidden] F32
    public IntPtr PostAttnNorm;   // [hidden] F32
    public IntPtr QNorm;          // [headDim] F32, optional per-head QK norm
    public IntPtr KNorm;          // [headDim] F32
    public H3Lin Q, K, V, O;
    public H3Lin Gate, Up, Down;
}

/// <summary>Qwen3-VL text-encoder prefill (TSGgml_MiniMaxH3TextEncode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3TextEncodeArgs
{
    public int StructBytes;
    public int NumLayers;

    public IntPtr Embeddings;     // F32 [hidden, seq]
    public IntPtr Out;            // F32 [hidden, seq] written
    public IntPtr Cos;            // F32 [headDim, seq]
    public IntPtr Sin;
    public IntPtr FinalNorm;      // F32 [hidden]; Zero = skip (H3 has no final norm)
    /// <summary>Qwen3-VL DeepStack residuals, dense F32 [hidden, seq, NumDeepstack]
    /// and zero outside the image spans. Zero when there are no reference images.</summary>
    public IntPtr Deepstack;

    public IntPtr Layers;         // H3TeLayerW[NumLayers]

    public int Hidden;
    public int Heads;
    public int KvHeads;
    public int HeadDim;
    public int Seq;
    public int Causal;
    public float Eps;
    public int NumDeepstack;
}

/// <summary>One run of conditioning tokens and the projection it goes through:
/// 0 = video patches, 1 = audio latents. MUST match native TSGH3CondChunk.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3CondChunk
{
    public int Kind;
    public int Count;
}

/// <summary>A run of tokens sharing one AdaLN row. <see cref="Col"/> is the base
/// column into the block's modulation matrix viewed as [hidden, 18*nTimesteps];
/// parameter p sits at Col + p. MUST match native TSGH3DitSegment.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3DitSegment
{
    public int Start;
    public int End;
    public int Col;
    public int Pad;
}

/// <summary>A token-refiner block: the DiT block layout minus AdaLN.
/// MUST match native TSGH3RefinerBlockW.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3RefinerBlockW
{
    public IntPtr Norm1, Norm2, QNorm, KNorm;
    public H3Lin Qkv, Out, Fc1, Fc2;
}

/// <summary>One MiniMax-H3 DiT block. MUST match native TSGH3DitBlockW.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3DitBlockW
{
    public IntPtr Norm1, Norm2, QNorm, KNorm;
    public H3Lin AdaLn;      // [timeEmbedDim, 18*hidden] + bias
    public H3Lin Qkv;        // [hidden, 3*inner]
    public H3Lin Out;        // [inner, hidden]
    public H3Lin Fc1;        // [hidden, 2*ffn] (gate | value)
    public H3Lin Fc2;        // [ffn, hidden]
}

/// <summary>One MiniMax-H3 diffusion step (TSGgml_MiniMaxH3DitForward).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3DitForwardArgs
{
    public int StructBytes;
    public int NumBlocks;
    public int NumRefinerBlocks;
    public int NumSegments;

    public IntPtr VideoTokens;    // F32 [videoPatchDim, videoCount], pre-patchified
    public IntPtr AudioTokens;    // F32 [audioChannels, audioCount]
    public IntPtr TextHidden;     // F32 [textDim, textCount]
    public IntPtr TimeEmbed;      // F32 [timeEmbedDim, nTimesteps]
    public IntPtr Cos;            // F32 [rotDim, nTok]
    public IntPtr Sin;
    public IntPtr VideoOut;       // F32 [videoPatchDim, videoCount] written
    public IntPtr AudioOut;       // F32 [audioChannels, audioCount] written

    public H3Lin VideoPatchProj, AudioPatchProj, ConditionProj;

    public IntPtr Refiner;        // H3RefinerBlockW[NumRefinerBlocks]
    public IntPtr RefinerFinalNorm;
    public IntPtr Blocks;         // H3DitBlockW[NumBlocks]
    public IntPtr Segments;       // H3DitSegment[NumSegments]

    public IntPtr FinalNorm;
    public H3Lin FinalAdaLn, FinalVideoOut, FinalAudioOut;

    public int NTok;
    public int TextCount;
    /// <summary>Conditioning video tokens, prepended to VideoTokens and sharing its
    /// projection; they sit between the text and the target audio in the sequence.</summary>
    public int ConditionCount;
    /// <summary>Ref2VA condition audio latents, F32 [audioChannels, ConditionAudioCount].
    /// Zero for FL2VA, whose conditioning is all pictures.</summary>
    public IntPtr ConditionAudio;
    /// <summary>H3CondChunk[NumCondChunks] describing the conditioning run in
    /// sequence order. Zero keeps the plain "all video patches" layout.</summary>
    public IntPtr CondChunks;
    public int NumCondChunks;
    public int ConditionAudioCount;
    public int AudioStart, AudioCount;
    public int VideoStart, VideoCount;
    public int AudioCol, VideoCol;
    public int Hidden, Heads, HeadDim, Inner, Ffn;
    public int RotDim, TimeEmbedDim, NTimesteps;
    public int VideoPatchDim, AudioChannels, TextDim;
    public float Eps;
    public float VideoScale;
    public float AudioScale;
}

/// <summary>A 2-D convolution kernel in ggml order [KW, KH, IC, OC].
/// MUST match native TSGH3Conv.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Conv
{
    public IntPtr W;
    public IntPtr B;          // nullable, F32 [OC]
    public long Kw, Kh, Ic, Oc;
    public int Type;
    public int Pad;
}

/// <summary>One encoder residual block. MUST match native TSGH3EncResBlock.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3EncResBlock
{
    public IntPtr Norm1W, Norm1B, Norm2W, Norm2B;
    public H3Conv Conv1, Conv2;
    public H3Conv Shortcut;   // W = Zero when the channel count is unchanged
}

/// <summary>One encoder level. MUST match native TSGH3EncLevel.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3EncLevel
{
    public H3EncResBlock Block0, Block1;
    public H3Conv Downsample; // W = Zero when the level does not downsample
    public int SpaceStride;
    public int Pad;
}

/// <summary>Single-frame MiniMax-H3 video VAE encode (TSGgml_MiniMaxH3VideoVaeEncode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VideoVaeEncodeArgs
{
    public int StructBytes;
    public int NumLevels;

    public IntPtr Image;      // F32 [W, H, 3], ImageNet-normalized
    public IntPtr Out;        // F32 [W/16, H/16, latentChannels] written

    public H3Conv ConvIn;
    public IntPtr Levels;     // H3EncLevel[NumLevels]
    public IntPtr NormOutW, NormOutB;
    public H3Conv ConvOut;
    public H3Conv QuantConv;

    public int Width, Height;
    public int LatentChannels;
    public int Groups;
    public float Eps;
}

/// <summary>A 1-D convolution. MUST match native TSGH3Conv1d.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Conv1d
{
    public IntPtr W;          // [K, IC, OC] ggml order
    public IntPtr B;          // nullable
    public long K, Ic, Oc;
    /// <summary>Bias length. A transposed conv's weight is [Cin, Cout, K] in torch,
    /// so reversing the dims puts Cout in <see cref="Ic"/> while the bias stays Cout
    /// long — hence carrying this explicitly.</summary>
    public long BiasLen;
    public int Type;
    public int Stride, Padding, Dilation;
}

/// <summary>An alias-free SnakeBeta activation. MUST match native TSGH3Act1d.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Act1d
{
    public IntPtr Alpha;      // [C] F32, log-scale
    public IntPtr Beta;       // [C] F32, log-scale
    public IntPtr UpFilter;   // [K] F32, kaiser
    public IntPtr DownFilter;
    public int Channels;
    public int Kernel;
}

/// <summary>Mono MiniMax-H3 audio VAE decode (TSGgml_MiniMaxH3AudioVaeDecode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3AudioVaeDecodeArgs
{
    public int StructBytes;
    public int NumStages;
    public int NumConvs;
    public int NumActs;

    public IntPtr Latent;     // F32 [T, latentChannels]
    public IntPtr Out;        // F32 [samples] written

    public H3Conv1d DecInProj, ConvPre, ConvPost;

    public IntPtr Convs;      // H3Conv1d[]: ups first, then per (stage, amp) convs1[3] + convs2[3]
    public IntPtr Acts;       // H3Act1d[]: per (stage, amp) 6, then activation_post
    public IntPtr Rates;      // int[NumStages]

    public int LatentLen;
    public int LatentChannels;
    public int AmpsPerStage;
    public int Samples;
    public float SnakeEps;
}

/// <summary>One Qwen3-VL vision block. MUST match native TSGH3VisBlockW.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VisBlockW
{
    public IntPtr Norm1W, Norm1B, Norm2W, Norm2B;
    public H3Lin Qkv, Proj, Fc1, Fc2;
}

/// <summary>A vision merger. MUST match native TSGH3VisMerger.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VisMerger
{
    public IntPtr NormW, NormB;
    public H3Lin Fc1, Fc2;
    /// <summary>1 = normalize at <c>dim</c> BEFORE the 2x2 merge (the final merger);
    /// 0 = normalize at <c>dim*4</c> AFTER it (the DeepStack mergers).</summary>
    public int NormBeforeMerge;
    public int Pad;
}

/// <summary>A 3-D convolution kernel, ggml order [KW, KH, KD, IC*OC] — which is
/// torch's [OC, IC, KD, KH, KW] read backwards. MUST match native TSGH3Conv3d.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Conv3d
{
    public IntPtr W;
    public IntPtr B;
    public long Kw, Kh, Kd, Ic, Oc;
    public int Type;
    public int Pad;
}

/// <summary>MUST match native TSGH3EncResBlock3D.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3EncResBlock3D
{
    public IntPtr Norm1W, Norm1B, Norm2W, Norm2B;
    public H3Conv3d Conv1, Conv2, Shortcut;
}

/// <summary>MUST match native TSGH3EncLevel3D.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3EncLevel3D
{
    public H3EncResBlock3D Block0, Block1;
    public H3Conv3d Downsample;
    public int SpaceStride;
    public int TimeStride;
}

/// <summary>Multi-frame causal 3-D video VAE encode
/// (TSGgml_MiniMaxH3VideoVaeEncode3D).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VideoVaeEncode3DArgs
{
    public int StructBytes;
    public int NumLevels;
    public IntPtr Video;      // F32 [W, H, T, 3], ImageNet-normalized
    public IntPtr Out;        // F32 [W/16, H/16, Tl, latentChannels] written
    public H3Conv3d ConvIn;
    public IntPtr Levels;     // H3EncLevel3D[NumLevels]
    public IntPtr NormOutW, NormOutB;
    public H3Conv3d ConvOut, QuantConv;
    public int Width, Height, Frames, LatentFrames, LatentChannels, Groups;
    public float Eps;
}

/// <summary>DAC Snake1d. alpha is LINEAR here, unlike the decoder's log-scale
/// alpha/beta; <see cref="AlphaEps"/> carries alpha + 1e-9 so the divide guard
/// needs no scalar in the graph. MUST match native TSGH3Snake1d.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3Snake1d
{
    public IntPtr Alpha;
    public IntPtr AlphaEps;
    public int Channels;
    public int Pad;
}

/// <summary>One DAC residual unit. MUST match native TSGH3AudioEncResUnit.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3AudioEncResUnit
{
    public H3Snake1d Act1;
    public H3Conv1d Conv1;
    public H3Snake1d Act2;
    public H3Conv1d Conv2;
}

/// <summary>One DAC encoder stage. MUST match native TSGH3AudioEncBlock.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3AudioEncBlock
{
    public H3AudioEncResUnit Unit0, Unit1, Unit2;   // dilations 1, 3, 9
    public H3Snake1d Act;
    public H3Conv1d Down;
}

/// <summary>The 2048 -> 32 causal-attention projection. MUST match native
/// TSGH3AudioAttnProj.</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3AudioAttnProj
{
    public IntPtr Norm1W, Norm1B, Norm3W, Norm3B, Norm2W, Norm2B, MlpNormW, MlpNormB;
    public H3Lin Qkv;
    /// <summary>Concatenated q_bias, ZEROS, v_bias — the checkpoint has no key bias.</summary>
    public IntPtr QkvBias;
    public H3Lin AttnProj, Proj, W0, W1, W2;
    public int Heads;
    public int Pad;
}

/// <summary>MiniMax-H3 audio VAE encoder (TSGgml_MiniMaxH3AudioVaeEncode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3AudioVaeEncodeArgs
{
    public int StructBytes;
    public int NumBlocks;
    public IntPtr Wave;          // F32 [samples], one mono plane
    public IntPtr Out;           // F32 [frames, latentChannels] written
    public H3Conv1d ConvIn;
    public IntPtr Blocks;        // H3AudioEncBlock[NumBlocks]
    public H3Snake1d FinalAct;
    public H3Conv1d FinalConv;
    public H3AudioAttnProj Pre;
    public H3Conv1d MeanProj;
    public int Samples, Frames, LatentChannels, TrunkChannels;
    public float Eps;
    public int Pad;
}

/// <summary>Qwen3-VL vision tower (TSGgml_MiniMaxH3VisionEncode).</summary>
[StructLayout(LayoutKind.Sequential)]
public struct H3VisionEncodeArgs
{
    public int StructBytes;
    public int NumBlocks;
    public int NumDeepstack;
    public int Pad;

    public IntPtr Patches;    // F32 [patchDim, tokens]
    public IntPtr PosEmbed;   // F32 [dim, tokens]
    public IntPtr Cos;        // F32 [headDim, tokens]
    public IntPtr Sin;
    public IntPtr Out;        // F32 [outDim, merged * (1 + numDeepstack)] written

    public H3Lin PatchEmbed;
    public IntPtr Blocks;             // H3VisBlockW[NumBlocks]
    public IntPtr DeepstackLayers;    // int[NumDeepstack]
    public IntPtr Mergers;            // H3VisMerger[1 + NumDeepstack]; [0] is the final one

    public int Tokens, Dim, Heads, HeadDim, PatchDim, MergeSize, OutDim;
    public float Eps;
}

// One encoder Resample stage. MUST match native TSGWanVaeDownW.
[StructLayout(LayoutKind.Sequential)]
public struct WanVaeDownW
{
    public WanVaeConv SConv;         // 3x3 stride-2 2D conv (Kd == 1)
    public WanVaeConv TConv;         // (3,1,1) stride-2 temporal conv; Tap0 == zero => downsample2d
}

// Whole Wan VAE encode (TSGgml_WanVaeEncode). MUST match native
// TSGgmlWanVaeEncodeDesc exactly (Res0..Res7 = 4 scales x 2 residual blocks).
[StructLayout(LayoutKind.Sequential)]
public struct WanVaeEncodeArgs
{
    public IntPtr X;                 // F32 [PxW, PxH, PxC, PxT] pixels in [-1,1] ([W,H,C,T] layout)
    public IntPtr Out;               // F32 [lw, lh, ZDim, lt] posterior mean, written
    public long OutLen;
    public WanVaeConv Stem;          // encoder conv1 (PxC -> dim, 3x3x3 causal)
    public WanVaeResBlockW Res0, Res1, Res2, Res3, Res4, Res5, Res6, Res7;
    public WanVaeDownW Down0, Down1, Down2;
    public WanVaeResBlockW Mid0, Mid2;
    public WanVaeAttnW Mid1;
    public WanVaeNorm HeadNorm;
    public WanVaeConv HeadConv;      // 3x3x3 -> 2*ZDim
    public WanVaeConv Quant;         // 1x1x1 (2z -> 2z)
    public int PxW, PxH, PxC, PxT;
    public int ZDim;
    public int Version;              // 1 (or 0) = wan2.1; 2 = wan2.2 TI2V
    public int StructBytes;
    public int Reserved;
}

internal enum GgmlUnaryOp
{
    Neg = 1,
    Exp = 2,
    Log = 3,
    Sqrt = 4,
    Relu = 5,
    Sigmoid = 6,
    Tanh = 7,
    SiLU = 8,
    Step = 9,
    Abs = 10,
    Sign = 11,
    GELU = 12,
}

internal enum GgmlFusedActMulOp
{
    SiLUMul = 1,
    GELUMul = 2,
    SigmoidMul = 3,
}

internal enum GgmlBinaryTensorOp
{
    Add = 1,
    Sub = 2,
    Mul = 3,
    Div = 4,
}

internal enum GgmlBinaryScalarOp
{
    Add = 1,
    Sub = 2,
    ReverseSub = 3,
    Mul = 4,
    Div = 5,
    ReverseDiv = 6,
}

internal enum GgmlNormOp
{
    LayerNorm = 1,
    RmsNorm = 2,
}

internal enum GgmlReductionOp
{
    Sum = 1,
    Mean = 2,
}

internal enum GgmlIndexReductionOp
{
    Argmin = 1,
    Argmax = 2,
}

    internal static partial class GgmlNative
    {
        private const string DllName = "GgmlOps";
        private static int s_windowsDependencySearchPathsInitialized;

        static GgmlNative()
        {
            NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);
            ApplyEarlyNativeTunables();
        }

        // Forces this type's static constructor so the assembly-wide DllImport
        // resolver is registered before other classes (e.g. Interop.GgmlApi)
        // issue their first P/Invoke into the GgmlOps module.
        internal static void EnsureImportResolverRegistered()
        {
        }

        /// <summary>
        /// Push the tunables that must be decided before the backend's first
        /// compute into the *native* environment.
        ///
        /// ggml-cuda reads GGML_CUDA_DISABLE_GRAPHS once and caches it in a
        /// function-local static the first time it considers capturing a graph —
        /// which happens while the model is still loading, long before the
        /// tensor-parallel context exists. Deciding this from the TP setup code
        /// (where the degree is obviously known) is therefore too late to have
        /// any effect, and .NET's Environment.SetEnvironmentVariable would not
        /// reach getenv anyway. This runs before the first P/Invoke into
        /// GgmlOps, which is early enough.
        ///
        /// Multi-GPU runs keep capture ON — it is worth 45% of decode throughput
        /// on a tensor-parallel token, which is dozens of small per-rank
        /// submissions that replay far more cheaply than they re-issue (4xA40:
        /// Qwen3.5-9B tp4 88 → 128.5 tok/s, Qwen3.5-35B-A3B tp2 71.3 → 104.1).
        /// This method exists for the opt-out: TS_GGML_TP_CUDA_GRAPHS=0 has to
        /// be turned into a native GGML_CUDA_DISABLE_GRAPHS *here*, because by
        /// the time the TP context is built the loader has already latched the
        /// old value and the setting would silently do nothing.
        /// </summary>
        private static void ApplyEarlyNativeTunables()
        {
            try
            {
                ApplySmallBarVulkanWorkaround();
                ApplyTensorParallelCudaGraphTunable();
            }
            catch (DllNotFoundException)
            {
                // No native library on this host (e.g. a managed-only unit test):
                // nothing to configure.
            }
            catch (EntryPointNotFoundException)
            {
                // Older GgmlOps without the setter; the native-side backstop in
                // TSGgml_TensorParallelInit still applies where it can.
            }
        }

        private static void ApplyTensorParallelCudaGraphTunable()
        {
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_GGML_TP_CUDA_GRAPHS"), "0",
                               StringComparison.Ordinal))
                return;

            string degreeText = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE");
            if (!int.TryParse(degreeText, System.Globalization.NumberStyles.Integer,
                              System.Globalization.CultureInfo.InvariantCulture, out int degree)
                || degree <= 1)
                return;

            SetNativeEnvironmentVariable("GGML_CUDA_DISABLE_GRAPHS", "1", overwrite: false);
        }

        /// <summary>
        /// ggml-vulkan's default device-buffer preference is
        /// DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT — the Resizable-BAR memory
        /// type. On a discrete AMD GPU without ReBAR only the 256 MB BAR heap
        /// carries those flags, and RADV over-commits that heap instead of
        /// failing the allocation, so the fallback to plain DEVICE_LOCAL never
        /// fires: every weight buffer is silently backed by GTT and the GPU
        /// re-reads the whole model across PCIe on every token (measured
        /// 1.3 → 35.9 tok/s on Qwen3.8-27B Q4_K_XL, RX 7900 XTX, RADV).
        /// Detect the small-BAR case via amdgpu sysfs and pre-set
        /// GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM before the Vulkan device is
        /// created — ggml-vulkan latches the variable at device init, so this
        /// must run before the first backend call. A user-set value of the
        /// variable (any value, including empty) is always respected.
        /// </summary>
        private static void ApplySmallBarVulkanWorkaround()
        {
            if (!OperatingSystem.IsLinux())
                return;
            if (Environment.GetEnvironmentVariable("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM") != null)
                return;

            try
            {
                foreach (string dev in System.IO.Directory.GetDirectories("/sys/class/drm"))
                {
                    string vramPath = System.IO.Path.Combine(dev, "device", "mem_info_vram_total");
                    string visPath = System.IO.Path.Combine(dev, "device", "mem_info_vis_vram_total");
                    if (!System.IO.File.Exists(vramPath) || !System.IO.File.Exists(visPath))
                        continue;
                    if (!long.TryParse(System.IO.File.ReadAllText(vramPath).Trim(), out long vramTotal) ||
                        !long.TryParse(System.IO.File.ReadAllText(visPath).Trim(), out long visTotal))
                        continue;

                    const long OneGiB = 1L << 30;
                    if (vramTotal >= 4 * OneGiB && visTotal <= OneGiB && visTotal < vramTotal / 4)
                    {
                        if (SetNativeEnvironmentVariable("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1", overwrite: false))
                        {
                            Console.Error.WriteLine(
                                $"ggml-vulkan small-BAR workaround: {System.IO.Path.GetFileName(dev)} exposes " +
                                $"{visTotal >> 20} MB CPU-visible VRAM of {vramTotal >> 30} GB total (Resizable BAR off); " +
                                "set GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1 so model weights stay in VRAM. " +
                                "Enabling Resizable BAR in the BIOS removes the need for this.");
                        }
                        return;
                    }
                }
            }
            catch (System.IO.IOException)
            {
                // sysfs unavailable/odd permissions: leave ggml-vulkan defaults alone.
            }
            catch (UnauthorizedAccessException)
            {
            }
        }

        /// <summary>
        /// Set an environment variable as native code sees it (see
        /// <see cref="ApplyEarlyNativeTunables"/> for why managed
        /// <c>Environment.SetEnvironmentVariable</c> is not enough).
        /// </summary>
        internal static bool SetNativeEnvironmentVariable(string name, string value, bool overwrite)
        {
            return TSGgml_SetNativeEnvironmentVariable(name, value, overwrite ? 1 : 0) != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_GetLastError();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_HasBackendFailure();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_GetBackendFailureText();

        [LibraryImport(DllName, StringMarshalling = StringMarshalling.Utf8)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SetNativeEnvironmentVariable(
            string name,
            string value, int overwrite);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_CanInitializeBackend(int backendType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_IsBackendAvailable(int backendType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SetVulkanDeviceIndex(int deviceIndex);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetVulkanDeviceCount();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetVulkanDeviceDescription(int deviceIndex, byte[] description, int descriptionSize);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AddmmF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlTensorView2D m1,
            GgmlTensorView2D m2,
            float beta,
            float alpha);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AddmmQuantF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedRmsNormMatMulQuantF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        // ------------------------------------------------------------------
        // The tensor-parallel PLAN SLOT contract (applies to every entry point
        // below that takes a `tpPlanOut`).
        //
        // These kernels either RUN their graph or, under tensor parallelism,
        // build it and hand back a plan the caller executes once per rank. The
        // native side picks between the two by testing whether `tp_plan_out` is
        // a null pointer.
        //
        // So the parameter MUST be `IntPtr[]`, which marshals a null array to a
        // real null pointer. Declaring it `out IntPtr` passes the address of a
        // stack local, i.e. a NON-null slot on every call — including from
        // callers that are not tensor-parallel. Those callers then silently land
        // in plan mode: the graph is built, parked, and never executed, while
        // the entry point still returns success. The op computes nothing and
        // reports that it worked.
        //
        // Note the native gate widens a caller's tpDegree to the process-wide TP
        // degree, so passing tpDegree=1 is NOT enough to stay out of plan mode
        // inside a tensor-parallel process — only a null slot is. This is what
        // silently turned Nemotron's Mamba2 residual add (a replicated, rank-0
        // computation) into a no-op under --tp 2.
        //
        // GgmlTensorParallelPlanSlotContractTests guards this.
        // ------------------------------------------------------------------
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedMatMulQuantAddF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_ReleaseFusedMatmulAddTpGraphs();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedFFNSwiGLUQuantF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData,
            int gateUpGgmlType,
            long gateUpNe0,
            long gateUpNe1,
            long gateUpRawBytes,
            IntPtr downData,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytes,
            int halfDim,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_ReleaseFusedFfnTpGraphs();

        public static void ReleaseFusedFfnTpGraphs()
        {
            try { TSGgml_ReleaseFusedFfnTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedFFNActProjectQuantF32(
            GgmlTensorView2D output,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData,
            int gateUpGgmlType,
            long gateUpNe0,
            long gateUpNe1,
            long gateUpRawBytes,
            IntPtr downData,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytes,
            int halfDim,
            int actType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedRmsNormResidualAddF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedPleBlockQuantF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D perLayerInput,
            IntPtr inpGateData, int inpGateGgmlType, long inpGateNe0, long inpGateNe1, long inpGateRawBytes,
            IntPtr projData, int projGgmlType, long projNe0, long projNe1, long projRawBytes,
            IntPtr postNormData, int postNormCount, float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedOutProjNormRouterQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outBytes,
            IntPtr normData, int normCount, float eps,
            GgmlTensorView2D normedOut,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerBytes,
            GgmlTensorView2D routerOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedVisionMLPF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr upB, int upBDim,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr downB, int downBDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MuseGlimmerVisionBlockQuantF32(
            in GgmlMuseGlimmerVisionBlockArgs args);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedOutProjFFNQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            IntPtr ffnNormData, int ffnNormCount, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedVisionAttentionF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr qkvW, int qkvNe0, int qkvNe1, long qkvBytes,
            IntPtr qkvB, int qkvBDim,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr outB, int outBDim,
            IntPtr cosTable, IntPtr sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35VisionEncoderF32(
            GgmlTensorView2D hidden,
            int blockCount, float eps, float attnScale,
            int numPatches, int numHeads, int headDim, int halfDim,
            IntPtr cosTable, IntPtr sinTable,
            IntPtr[] ln1W, IntPtr[] ln1B,
            IntPtr[] qkvW, IntPtr[] qkvB,
            IntPtr[] outW, IntPtr[] outB,
            IntPtr[] ln2W, IntPtr[] ln2B,
            IntPtr[] upW, IntPtr[] upB,
            IntPtr[] downW, IntPtr[] downB,
            int lnDim,
            int qkvNe0, int qkvNe1, long qkvBytes, int qkvBDim,
            int outNe0, int outNe1, long outBytes, int outBDim,
            int upNe0, int upNe1, long upBytes, int upBDim,
            int downNe0, int downNe1, long downBytes, int downBDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GlmVisionEncoderF32(
            GgmlTensorView2D hidden,
            int blockCount, float eps, float attnScale, float swigluLimit,
            int numPatches, int numHeads, int headDim, int halfDim,
            IntPtr cosTable, IntPtr sinTable,
            IntPtr[] ln1W,
            IntPtr[] qkvW, IntPtr[] qkvB,
            IntPtr[] qnW, IntPtr[] knW,
            IntPtr[] outW, IntPtr[] outB,
            IntPtr[] ln2W,
            IntPtr[] gateW, IntPtr[] gateB,
            IntPtr[] upW, IntPtr[] upB,
            IntPtr[] downW, IntPtr[] downB,
            int lnDim,
            int qkvNe0, int qkvNe1, long qkvBytes,
            int outNe0, int outNe1, long outBytes,
            int ffnNe0, int ffnNe1, long ffnUpBytes, long ffnDownBytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedGemma4VisionBlockF32(
            GgmlTensorView2D hidden, float eps,
            IntPtr ln1W,
            IntPtr qW, int qNe0, int qNe1, long qBytes,
            IntPtr kW, int kNe0, int kNe1, long kBytes,
            IntPtr vW, int vNe0, int vNe1, long vBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr attnPostNormW,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr cosx, IntPtr sinx, IntPtr cosy, IntPtr siny,
            IntPtr ln2W,
            IntPtr gateW, int gateNe0, int gateNe1, long gateBytes,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr ffnPostNormW,
            IntPtr clamps,
            int numPatches, int numHeads, int headDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetRowsQuantF32(
            GgmlTensorView2D result,
            IntPtr srcData,
            int srcGgmlType,
            long srcNe0,
            long srcNe1,
            long srcRawBytes,
            GgmlContiguousTensor indices);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MoEExpertsForwardF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MoEExpertsSwiGLUForwardF32(
            GgmlTensorView2D result,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int gateGgmlType,
            long gateNe0,
            long gateNe1,
            long gateRawBytesEach,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MoEExpertsSwiGLUResidualF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs,
            IntPtr[] upDataPtrs,
            IntPtr[] downDataPtrs,
            int gateGgmlType,
            long gateNe0,
            long gateNe1,
            long gateRawBytesEach,
            int upGgmlType,
            long upNe0,
            long upNe1,
            long upRawBytesEach,
            int downGgmlType,
            long downNe0,
            long downNe1,
            long downRawBytesEach,
            float[] routeWeights,
            int useShared,
            IntPtr sharedGateData,
            IntPtr sharedUpData,
            IntPtr sharedDownData,
            int sharedGateGgmlType,
            long sharedGateNe0,
            long sharedGateNe1,
            long sharedGateRawBytes,
            int sharedUpGgmlType,
            long sharedUpNe0,
            long sharedUpNe1,
            long sharedUpRawBytes,
            int sharedDownGgmlType,
            long sharedDownNe0,
            long sharedDownNe1,
            long sharedDownRawBytes,
            float sharedScalar);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AddmmQuantBatchF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2RawBytes,
            int batchCount,
            long[] weightOffsets,
            long[] weightNe1Arr);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AddmmBatchF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView3D m1,
            GgmlTensorView3D m2,
            float beta,
            float alpha);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_ReduceLastDimF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_IndexReductionF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SoftmaxF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // In-place softmax with causal+SWA mask and optional attention sinks.
        // Replaces the GptOss CPU softmax-with-sinks loop. See native side:
        // attention_softmax_with_sinks_f32_impl in ggml_ops_norm_attn.cpp.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AttentionSoftmaxWithSinksF32(
            GgmlTensorView3D scores,
            IntPtr sinksData,         // float* [num_heads], or IntPtr.Zero for no sinks
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale);

        // Fused MoE FFN prefill (mul_mat_id-based).
        // Collapses an entire layer's MoE forward (gate + up + SwiGLU + down +
        // expert weighting + aggregation) into one GGML graph dispatch.
        // See native side: TSGgml_MoEFFNPrefillSwiGLUQuantF32 in ggml_ops_moe.cpp.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
            IntPtr hiddenIn,
            IntPtr hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,    // int32* [seqLen, nUsed]
            IntPtr routingWeights,     // float* [seqLen, nUsed]
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,           // optional float* [biasDim, numExperts] (biasDim = nFf or 2*nFf for fused gate_up); IntPtr.Zero to skip
            IntPtr upBias,             // optional, only valid when up_data != null
            IntPtr downBias,           // optional float* [hiddenDim, numExperts]
            int activationType,        // 0 = SwiGLU split, 1 = SwiGLU OAI, 2 = GEGLU split, 3 = ReLU-squared
            float oaiAlpha,
            float oaiLimit,
            int runOnCpu);            // non-zero: run this layer on the host ggml CPU backend (MoE CPU offload)

        // Gemma 4 MoE GEGLU + post_norm + residual add fused kernel.
        // Computes residual_in_out += rms_norm(moe_ffn(hidden_in), eps) * post_norm_w
        // in a single GGML graph dispatch. Mirrors the existing
        // TSGgml_MoEFFNPrefillSwiGLUQuantF32 ABI but adds the residual buffer,
        // the post_ffw_norm_2 weight, and an RMSNorm epsilon.
        // See native side: TSGgml_Gemma4MoEGEGLUResidualF32 in ggml_ops_moe.cpp.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4MoEGEGLUResidualF32(
            IntPtr hiddenIn,
            IntPtr residualInOut,      // float* [seqLen, hiddenDim] - dense FFN result; kernel adds normed MoE output to it in place
            IntPtr postNormW,          // float* [hiddenDim] - post_ffw_norm_2.weight
            float postNormEps,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit,
            int runOnCpu);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_ScaledDotProductAttentionF32(
            GgmlTensorView4D result,
            GgmlTensorView4D query,
            GgmlTensorView4D key,
            GgmlTensorView4D value,
            GgmlTensorView4D mask,
            int hasMask,
            float scale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SoftmaxGradF32(
            GgmlTensorView4D result,
            GgmlTensorView4D adj,
            GgmlTensorView4D val,
            int addGrad);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_AdamF32(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4LayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr postAttnNormW,
            IntPtr ffnNormW,
            IntPtr guW, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downW, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr postFfnNormW,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isLocal, int slidingWindow,
            float ropeBase, int ropeDims,
            IntPtr ropeFreqFactors, int freqFactorsLen,
            float layerScalar, float eps,
            IntPtr swaPrevK, IntPtr swaPrevV, int prevWindowLen,
            IntPtr pleInputData, int pleDim,
            IntPtr pleGateW, int pleGateType, long pleGateNe0, long pleGateNe1, long pleGateBytes,
            IntPtr pleProjW, int pleProjType, long pleProjNe0, long pleProjNe1, long pleProjBytes,
            IntPtr plePostNormW,
            IntPtr freshKOut, IntPtr freshVOut,
            int isShared,
            IntPtr donorK, IntPtr donorV, int donorKvLen,
            int kvCacheType);

        public static void Gemma4LayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr postAttnNormW,
            IntPtr ffnNormW,
            IntPtr guW, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downW, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr postFfnNormW,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isLocal, int slidingWindow,
            float ropeBase, int ropeDims,
            IntPtr ropeFreqFactors, int freqFactorsLen,
            float layerScalar, float eps,
            IntPtr swaPrevK, IntPtr swaPrevV, int prevWindowLen,
            IntPtr pleInputData, int pleDim,
            IntPtr pleGateW, int pleGateType, long pleGateNe0, long pleGateNe1, long pleGateBytes,
            IntPtr pleProjW, int pleProjType, long pleProjNe0, long pleProjNe1, long pleProjBytes,
            IntPtr plePostNormW,
            IntPtr freshKOut, IntPtr freshVOut,
            int isShared,
            IntPtr donorK, IntPtr donorV, int donorKvLen,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_Gemma4LayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                postAttnNormW,
                ffnNormW,
                guW, guType, guNe0, guNe1, guBytes,
                downW, downType, downNe0, downNe1, downBytes,
                postFfnNormW,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isLocal, slidingWindow,
                ropeBase, ropeDims,
                ropeFreqFactors, freqFactorsLen,
                layerScalar, eps,
                swaPrevK, swaPrevV, prevWindowLen,
                pleInputData, pleDim,
                pleGateW, pleGateType, pleGateNe0, pleGateNe1, pleGateBytes,
                pleProjW, pleProjType, pleProjNe0, pleProjNe1, pleProjBytes,
                plePostNormW,
                freshKOut, freshVOut,
                isShared,
                donorK, donorV, donorKvLen,
                kvCacheType), "gemma4_layer_prefill");
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedPrefillAttentionF32(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow,
            float scale, int inputFormat);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedPrefillAttentionF16KV(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvCacheLen,
            int maskStartPos, int slidingWindow,
            float scale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FlashAttnDecodeF32(
            IntPtr qData, IntPtr kData, IntPtr vData,
            IntPtr kCacheData, IntPtr vCacheData,
            IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position,
            float scale, int kvCacheType);

        // Device-resident paged K/V pool. The pool tensors live on the backend
        // for the model's lifetime; only this step's new rows (scatter) and the
        // per-sequence row-index vectors (attention) cross the bus.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_PagedKvPoolCreate(
            int numLayers, int numBlocks, int blockSize, int numKvHeads, int headDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_PagedKvPoolFree(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial long TSGgml_PagedKvPoolBytes(IntPtr handle);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedKvPoolGrow(IntPtr handle, int newNumBlocks);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedKvPoolScatter(
            IntPtr handle, int layer, IntPtr kData, IntPtr vData,
            IntPtr slotMapping, int numTokens);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedKvPoolAttention(
            IntPtr handle, int layer, IntPtr qData, IntPtr outData,
            IntPtr queryStartLoc, IntPtr seqLens, IntPtr positions,
            IntPtr blockTableFlat, IntPtr blockTableOffsets,
            int numSeqs, int numTokens, int numHeads, int slidingWindow, float scale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedAttentionForward(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedAttentionForwardWithSinks(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale,
            IntPtr sinksData);          // [numHeads] F32 or IntPtr.Zero

        // GPU-resident variant: qData and outData point to existing backend
        // (Tensor storage) buffers, so the kernel can zero-copy bind them
        // instead of round-tripping through host arrays + ggml_backend_synchronize.
        // Eliminates the per-layer queue drain that GetElementsAsFloat would
        // otherwise force.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedAttentionForwardDevice(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PagedAttentionForwardDeviceWithSinks(
            IntPtr qData,
            IntPtr pagedKData,
            IntPtr pagedVData,
            IntPtr outData,
            IntPtr queryStartLoc,
            IntPtr seqLens,
            IntPtr positions,
            IntPtr blockTableFlat,
            IntPtr blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int slidingWindow,
            float scale,
            IntPtr sinksData);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35AttentionLayerDecode(
            IntPtr residualData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeNDims, int ropeMode, int kvCacheType);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssAttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qkvB,
            int isQkvFused,
            IntPtr kW, int kType, long kNe0, long kNe1, long kBytes,
            IntPtr kB,
            IntPtr vW, int vType, long vNe0, long vNe1, long vBytes,
            IntPtr vB,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr oB,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isSwa, int slidingWindow,
            IntPtr sinksData,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int originalContextLength,
            int kvCacheType,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35AttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int ropeMode,
            int kvCacheType,
            float eps,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ReleaseAttentionTpGraphs();

        public static void Qwen35ReleaseAttentionTpGraphs()
        {
            try { TSGgml_Qwen35ReleaseAttentionTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        public static void Qwen35AttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int ropeMode,
            int kvCacheType,
            float eps,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            CheckResult(TSGgml_Qwen35AttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormW, kNormW,
                oW, oType, oNe0, oNe1, oBytes,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                ropeBase, ropeFreqScale, ropeDims,
                ropeMode, kvCacheType, eps,
                tpDegree, tpPlanOut), "qwen35_attention_layer_prefill");
        }

        public static void GptOssAttentionLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int seqLen,
            IntPtr attnNormW,
            IntPtr qkvW, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qkvB,
            int isQkvFused,
            IntPtr kW, int kType, long kNe0, long kNe1, long kBytes,
            IntPtr kB,
            IntPtr vW, int vType, long vNe0, long vNe1, long vBytes,
            IntPtr vB,
            IntPtr oW, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr oB,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int kvHeads, int headDim,
            int cacheSize, int startPos,
            int isSwa, int slidingWindow,
            IntPtr sinksData,
            float ropeBase, float ropeFreqScale, int ropeDims,
            int originalContextLength,
            int kvCacheType,
            float eps)
        {
            CheckResult(TSGgml_GptOssAttentionLayerPrefill(
                hiddenData, hiddenSize, seqLen,
                attnNormW,
                qkvW, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qkvB,
                isQkvFused,
                kW, kType, kNe0, kNe1, kBytes,
                kB,
                vW, vType, vNe0, vNe1, vBytes,
                vB,
                oW, oType, oNe0, oNe1, oBytes,
                oB,
                kCacheData, vCacheData,
                numHeads, kvHeads, headDim,
                cacheSize, startPos,
                isSwa, slidingWindow,
                sinksData,
                ropeBase, ropeFreqScale, ropeDims,
                originalContextLength,
                kvCacheType,
                eps), "gpt_oss_attention_layer_prefill");
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4ModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            int[] kvSourceArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int position,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr logitsData, int vocabSize,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNormData, float logitSoftcap,
            IntPtr pleTokenEmbdData, int pleTokenEmbdType,
            long pleTokenEmbdNe0, long pleTokenEmbdNe1, long pleTokenEmbdBytes,
            int pleTokenId,
            IntPtr pleModelProjData, int pleModelProjType,
            long pleModelProjNe0, long pleModelProjNe1, long pleModelProjBytes,
            IntPtr pleModelProjNormData,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);


        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DFlashInject(
            float[] featRows, int featureSize, int nRows,
            long[] ringRowsIdx, int[] positions,
            int numLayers, int hiddenSize, int headDim, int numKvHeads, int ringRows,
            float eps, float ropeBase, float ropeFreqScale,
            IntPtr fcData, int fcType, long fcNe0, long fcNe1, long fcBytes,
            IntPtr encNormData,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr[] kNormArr,
            IntPtr[] ringKArr, IntPtr[] ringVArr,
            int ringDtype);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DFlashDraftBlock(
            int[] blockIds, int blockLen, int[] positions,
            int numLayers, int hiddenSize, int headDim, int numHeads, int numKvHeads, int ringRows,
            float eps, float ropeBase, float ropeFreqScale, float kqScale,
            int[] ringSlotPos, int slidingWindow,
            IntPtr[] attnNormArr,
            IntPtr[] qArr, int[] qTypeArr, long[] qNe0Arr, long[] qNe1Arr, long[] qBytesArr,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            IntPtr[] ffnNormArr,
            IntPtr[] gateArr, int[] gateTypeArr, long[] gateNe0Arr, long[] gateNe1Arr, long[] gateBytesArr,
            IntPtr[] upArr, int[] upTypeArr, long[] upNe0Arr, long[] upNe1Arr, long[] upBytesArr,
            IntPtr[] downArr, int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            IntPtr[] ringKArr, IntPtr[] ringVArr, int ringDtype,
            IntPtr outNormData,
            IntPtr tokEmbdData, int tokEmbdType, long tokEmbdNe0, long tokEmbdNe1, long tokEmbdBytes,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            int vocabSize, int[] idsOut, float[] confOut,
            // DFlash2 grouped dynamic convolution. convTaps == 0 disables it and
            // every array below may be null (a first-generation drafter).
            int convTaps, int convGroupSize, int convNumGroups,
            IntPtr[] attnConvBaseArr,
            IntPtr[] attnConvProjArr, int[] attnConvProjTypeArr,
            long[] attnConvProjNe0Arr, long[] attnConvProjNe1Arr, long[] attnConvProjBytesArr,
            IntPtr[] ffnConvBaseArr,
            IntPtr[] ffnConvProjArr, int[] ffnConvProjTypeArr,
            long[] ffnConvProjNe0Arr, long[] ffnConvProjNe1Arr, long[] ffnConvProjBytesArr,
            // DFlash2 candidate selector. selRank == 0 disables it; when it is on,
            // idsOut/confOut are left untouched and the lattice comes back instead.
            int selRank, int selTopK, float selLogitScale, float selLogitSoftcap,
            IntPtr selHiddenData, int selHiddenType, long selHiddenNe0, long selHiddenNe1, long selHiddenBytes,
            IntPtr selPredData, int selPredType, long selPredNe0, long selPredNe1, long selPredBytes,
            IntPtr selSuccData, int selSuccType, long selSuccNe0, long selSuccNe1, long selSuccBytes,
            float[] selScoresOut, int[] selCandOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_DFlashResetCaches();

        /// <summary>DFlash PASS A+B in one graph. False = declined, caller falls back.</summary>
        public static bool DFlashInject(
            float[] featRows, int featureSize, int nRows,
            long[] ringRowsIdx, int[] positions,
            int numLayers, int hiddenSize, int headDim, int numKvHeads, int ringRows,
            float eps, float ropeBase, float ropeFreqScale,
            IntPtr fcData, int fcType, long fcNe0, long fcNe1, long fcBytes,
            IntPtr encNormData,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr[] kNormArr, IntPtr[] ringKArr, IntPtr[] ringVArr, int ringDtype)
            => TSGgml_DFlashInject(featRows, featureSize, nRows, ringRowsIdx, positions,
                numLayers, hiddenSize, headDim, numKvHeads, ringRows,
                eps, ropeBase, ropeFreqScale,
                fcData, fcType, fcNe0, fcNe1, fcBytes, encNormData,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                kNormArr, ringKArr, ringVArr, ringDtype) != 0;

        /// <summary>
        /// DFlash PASS C in one graph.
        ///
        /// Plain DFlash returns the on-device argmax id and its softmax probability
        /// per block row. A DFlash2 drafter (selRank &gt; 0) instead returns the
        /// candidate ids and the transition lattice the caller walks:
        /// selCandOut is [selTopK, gamma] and selScoresOut holds the anchor row
        /// (selTopK floats, block position 0 scored against the verified anchor)
        /// followed by one [selTopK(pred), selTopK(cand)] matrix per following
        /// position, candidate-fastest. That is ~7 KB per step against the 12.9 MB
        /// a [vocab, block] readback would cost.
        /// </summary>
        public static bool DFlashDraftBlock(
            int[] blockIds, int blockLen, int[] positions,
            int numLayers, int hiddenSize, int headDim, int numHeads, int numKvHeads, int ringRows,
            float eps, float ropeBase, float ropeFreqScale, float kqScale,
            int[] ringSlotPos, int slidingWindow,
            IntPtr[] attnNormArr,
            IntPtr[] qArr, int[] qTypeArr, long[] qNe0Arr, long[] qNe1Arr, long[] qBytesArr,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            IntPtr[] ffnNormArr,
            IntPtr[] gateArr, int[] gateTypeArr, long[] gateNe0Arr, long[] gateNe1Arr, long[] gateBytesArr,
            IntPtr[] upArr, int[] upTypeArr, long[] upNe0Arr, long[] upNe1Arr, long[] upBytesArr,
            IntPtr[] downArr, int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            IntPtr[] ringKArr, IntPtr[] ringVArr, int ringDtype,
            IntPtr outNormData,
            IntPtr tokEmbdData, int tokEmbdType, long tokEmbdNe0, long tokEmbdNe1, long tokEmbdBytes,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            int vocabSize, int[] idsOut, float[] confOut,
            // DFlash2 grouped dynamic convolution. convTaps == 0 disables it and
            // every array below may be null (a first-generation drafter).
            int convTaps, int convGroupSize, int convNumGroups,
            IntPtr[] attnConvBaseArr,
            IntPtr[] attnConvProjArr, int[] attnConvProjTypeArr,
            long[] attnConvProjNe0Arr, long[] attnConvProjNe1Arr, long[] attnConvProjBytesArr,
            IntPtr[] ffnConvBaseArr,
            IntPtr[] ffnConvProjArr, int[] ffnConvProjTypeArr,
            long[] ffnConvProjNe0Arr, long[] ffnConvProjNe1Arr, long[] ffnConvProjBytesArr,
            // DFlash2 candidate selector. selRank == 0 disables it; when it is on,
            // idsOut/confOut are left untouched and the lattice comes back instead.
            int selRank, int selTopK, float selLogitScale, float selLogitSoftcap,
            IntPtr selHiddenData, int selHiddenType, long selHiddenNe0, long selHiddenNe1, long selHiddenBytes,
            IntPtr selPredData, int selPredType, long selPredNe0, long selPredNe1, long selPredBytes,
            IntPtr selSuccData, int selSuccType, long selSuccNe0, long selSuccNe1, long selSuccBytes,
            float[] selScoresOut, int[] selCandOut)
            => TSGgml_DFlashDraftBlock(blockIds, blockLen, positions,
                numLayers, hiddenSize, headDim, numHeads, numKvHeads, ringRows,
                eps, ropeBase, ropeFreqScale, kqScale, ringSlotPos, slidingWindow,
                attnNormArr,
                qArr, qTypeArr, qNe0Arr, qNe1Arr, qBytesArr,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                qNormArr, kNormArr,
                oArr, oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                ffnNormArr,
                gateArr, gateTypeArr, gateNe0Arr, gateNe1Arr, gateBytesArr,
                upArr, upTypeArr, upNe0Arr, upNe1Arr, upBytesArr,
                downArr, downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                ringKArr, ringVArr, ringDtype, outNormData,
                tokEmbdData, tokEmbdType, tokEmbdNe0, tokEmbdNe1, tokEmbdBytes,
                lmHeadData, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                vocabSize, idsOut, confOut,
                convTaps, convGroupSize, convNumGroups,
                attnConvBaseArr, attnConvProjArr, attnConvProjTypeArr,
                attnConvProjNe0Arr, attnConvProjNe1Arr, attnConvProjBytesArr,
                ffnConvBaseArr, ffnConvProjArr, ffnConvProjTypeArr,
                ffnConvProjNe0Arr, ffnConvProjNe1Arr, ffnConvProjBytesArr,
                selRank, selTopK, selLogitScale, selLogitSoftcap,
                selHiddenData, selHiddenType, selHiddenNe0, selHiddenNe1, selHiddenBytes,
                selPredData, selPredType, selPredNe0, selPredNe1, selPredBytes,
                selSuccData, selSuccType, selSuccNe0, selSuccNe1, selSuccBytes,
                selScoresOut, selCandOut) != 0;

        /// <summary>Drop the persistent DFlash graphs (ring reallocation / KV reset).</summary>
        public static void DFlashResetCaches() => TSGgml_DFlashResetCaches();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MuseGlimmerModelForward(
            IntPtr hiddenData, int hiddenSize, int nTokens, int numLayers,
            IntPtr[] attnNormArr,
            IntPtr[] qArr, IntPtr[] kArr, IntPtr[] vArr, IntPtr[] gateArr,
            IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr,
            IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr,
            IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] isSwaArr,
            int[] qTypeArr, long[] qNe0Arr, long[] qNe1Arr, long[] qBytesArr,
            int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            int[] gateTypeArr, long[] gateNe0Arr, long[] gateNe1Arr, long[] gateBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int numKvHeads, int headDim, int cacheSize, int swaCacheSize,
            int startPos, int slidingWindow,
            float eps, float postNormEps, float ropeBase, float ropeFreqScale,
            float kqScale, int kvCacheType,
            IntPtr logitsData, int vocabSize,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNormData, float logitScale, float logitSoftcap,
            IntPtr captureData, int[] captureLayers, int captureCount,
            IntPtr tokEmbdData, int tokEmbdType, long tokEmbdNe0, long tokEmbdNe1, long tokEmbdBytes,
            int[] tokenIds, int allLogitsRows,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_MuseGlimmerResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_MuseGlimmerReleaseTpGraphs();

        /// <summary>
        /// Whole-model Muse-Glimmer forward in a single GGML graph. nTokens == 1 uses
        /// a persistent, CUDA-graph-capturable graph; nTokens &gt; 1 builds a transient
        /// prefill graph. Returns false (without throwing) when the kernel declines,
        /// so the caller can fall back to the per-op path.
        /// <para>
        /// With <paramref name="tpDegree"/> &gt; 1 and a non-null
        /// <paramref name="tpPlanOut"/> the kernel BUILDS the graph for the currently
        /// active rank (see <c>SetActiveRank</c>) and returns its execution plan in
        /// <c>tpPlanOut[0]</c> instead of running it; the caller collects one plan per
        /// rank and executes them together through <c>TensorParallelExecutePlans</c>,
        /// which AllReduces the per-layer partial sums at the segment boundaries.
        /// </para>
        /// </summary>
        public static bool MuseGlimmerModelForward(
            IntPtr hiddenData, int hiddenSize, int nTokens, int numLayers,
            IntPtr[] attnNormArr,
            IntPtr[] qArr, IntPtr[] kArr, IntPtr[] vArr, IntPtr[] gateArr,
            IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr,
            IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr,
            IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] isSwaArr,
            int[] qTypeArr, long[] qNe0Arr, long[] qNe1Arr, long[] qBytesArr,
            int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            int[] gateTypeArr, long[] gateNe0Arr, long[] gateNe1Arr, long[] gateBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int numKvHeads, int headDim, int cacheSize, int swaCacheSize,
            int startPos, int slidingWindow,
            float eps, float postNormEps, float ropeBase, float ropeFreqScale,
            float kqScale, int kvCacheType,
            IntPtr logitsData, int vocabSize,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNormData, float logitScale, float logitSoftcap,
            IntPtr captureData, int[] captureLayers, int captureCount,
            IntPtr tokEmbdData, int tokEmbdType, long tokEmbdNe0, long tokEmbdNe1, long tokEmbdBytes,
            int[] tokenIds, int allLogitsRows,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            return TSGgml_MuseGlimmerModelForward(
                hiddenData, hiddenSize, nTokens, numLayers,
                attnNormArr, qArr, kArr, vArr, gateArr, qNormArr, kNormArr, oArr,
                postAttnNormArr, ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr, isSwaArr,
                qTypeArr, qNe0Arr, qNe1Arr, qBytesArr,
                kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                gateTypeArr, gateNe0Arr, gateNe1Arr, gateBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, numKvHeads, headDim, cacheSize, swaCacheSize, startPos, slidingWindow,
                eps, postNormEps, ropeBase, ropeFreqScale, kqScale, kvCacheType,
                logitsData, vocabSize,
                lmHeadData, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNormData, logitScale, logitSoftcap,
                captureData, captureLayers, captureCount,
                tokEmbdData, tokEmbdType, tokEmbdNe0, tokEmbdNe1, tokEmbdBytes, tokenIds, allLogitsRows,
                tpDegree, tpPlanOut) != 0;
        }

        /// <summary>
        /// Drop the persistent (CUDA-graph-captured) Muse-Glimmer decode graphs. The
        /// captured graph pins ggml-cuda's scratch-pool and KV-cache device addresses,
        /// which a prefill or a KV reset/grow can move; a stale replay then hangs.
        /// </summary>
        public static void MuseGlimmerResetDecodeCache() => TSGgml_MuseGlimmerResetDecodeCache();

        /// <summary>
        /// Release every rank's parked tensor-parallel Muse-Glimmer graph (the
        /// transient prefill path's ggml context and per-call buffers). Call on
        /// dispose and on KV reset, while the backends are still alive.
        /// </summary>
        public static void MuseGlimmerReleaseTpGraphs()
        {
            try { TSGgml_MuseGlimmerReleaseTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4ModelDecodeBatched(
            IntPtr hiddenData, int hiddenSize, int numLayers, int nSeqs,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int[] positions,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr logitsData, int vocabSize,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNormData, float logitSoftcap);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4ModelVerify(
            IntPtr hiddenData, int hiddenSize, int numLayers, int numTokens,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int startPos,
            float eps,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            int[] kvSourceArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            byte[] isExceptArr,
            IntPtr pleTokenEmbdData, int pleTokenEmbdType,
            long pleTokenEmbdNe0, long pleTokenEmbdNe1, long pleTokenEmbdBytes,
            int[] pleTokenIds,
            IntPtr pleProjWData, int pleProjWType,
            long pleProjWNe0, long pleProjWNe1, long pleProjWBytes,
            IntPtr pleProjNormData,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4DraftStep(
            int token, IntPtr hPrev, int fixedPos,
            int backbone, int draftHidden, int numDLayers, int numHeads, int vocab,
            float eps, int kvCacheType,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            IntPtr tgtTokEmbd, int tteType, long tteNe0, long tteNe1, long tteBytes,
            IntPtr nextnPre, int npreType, long npreNe0, long npreNe1, long npreBytes,
            IntPtr nextnPost, int npostType, long npostNe0, long npostNe1, long npostBytes,
            IntPtr draftTokEmbd, int dteType, long dteNe0, long dteNe1, long dteBytes,
            IntPtr outputNormW,
            IntPtr[] attnNormArr, IntPtr[] wqArr, int[] wqType, long[] wqNe0, long[] wqNe1, long[] wqBytes,
            IntPtr[] qNormArr, IntPtr[] woArr, int[] woType, long[] woNe0, long[] woNe1, long[] woBytes,
            IntPtr[] postAttnNormArr, IntPtr[] ffnNormArr,
            IntPtr[] gateArr, int[] gateType, long[] gateNe0, long[] gateNe1, long[] gateBytes,
            IntPtr[] upArr, int[] upType, long[] upNe0, long[] upNe1, long[] upBytes,
            IntPtr[] downArr, int[] downType, long[] downNe0, long[] downNe1, long[] downBytes,
            IntPtr[] postFfwNormArr, float[] outScaleArr,
            int[] hdArr, int[] kvHeadsArr, int[] isLocalArr, float[] ropeBaseArr, int[] ropeDimsArr,
            IntPtr[] donorKArr, IntPtr[] donorVArr, int[] donorCacheSizeArr,
            IntPtr logitsOut, IntPtr hOut);

        /// <summary>Fused Gemma 4 MTP draft step. Returns false (no throw) when the
        /// native kernel declines (e.g. fixed_pos past the donor SWA window) so the
        /// caller falls back to the per-op draft.</summary>
        public static unsafe bool Gemma4DraftStep(
            int token, IntPtr hPrev, int fixedPos,
            int backbone, int draftHidden, int numDLayers, int numHeads, int vocab,
            float eps, int kvCacheType,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            IntPtr tgtTokEmbd, int tteType, long tteNe0, long tteNe1, long tteBytes,
            IntPtr nextnPre, int npreType, long npreNe0, long npreNe1, long npreBytes,
            IntPtr nextnPost, int npostType, long npostNe0, long npostNe1, long npostBytes,
            IntPtr draftTokEmbd, int dteType, long dteNe0, long dteNe1, long dteBytes,
            IntPtr outputNormW,
            IntPtr[] attnNormArr, IntPtr[] wqArr, int[] wqType, long[] wqNe0, long[] wqNe1, long[] wqBytes,
            IntPtr[] qNormArr, IntPtr[] woArr, int[] woType, long[] woNe0, long[] woNe1, long[] woBytes,
            IntPtr[] postAttnNormArr, IntPtr[] ffnNormArr,
            IntPtr[] gateArr, int[] gateType, long[] gateNe0, long[] gateNe1, long[] gateBytes,
            IntPtr[] upArr, int[] upType, long[] upNe0, long[] upNe1, long[] upBytes,
            IntPtr[] downArr, int[] downType, long[] downNe0, long[] downNe1, long[] downBytes,
            IntPtr[] postFfwNormArr, float[] outScaleArr,
            int[] hdArr, int[] kvHeadsArr, int[] isLocalArr, float[] ropeBaseArr, int[] ropeDimsArr,
            IntPtr[] donorKArr, IntPtr[] donorVArr, int[] donorCacheSizeArr,
            IntPtr logitsOut, IntPtr hOut)
        {
            int r = TSGgml_Gemma4DraftStep(
                token, hPrev, fixedPos,
                backbone, draftHidden, numDLayers, numHeads, vocab,
                eps, kvCacheType,
                ropeFreqFactors, ropeFreqFactorsLen,
                tgtTokEmbd, tteType, tteNe0, tteNe1, tteBytes,
                nextnPre, npreType, npreNe0, npreNe1, npreBytes,
                nextnPost, npostType, npostNe0, npostNe1, npostBytes,
                draftTokEmbd, dteType, dteNe0, dteNe1, dteBytes,
                outputNormW,
                attnNormArr, wqArr, wqType, wqNe0, wqNe1, wqBytes,
                qNormArr, woArr, woType, woNe0, woNe1, woBytes,
                postAttnNormArr, ffnNormArr,
                gateArr, gateType, gateNe0, gateNe1, gateBytes,
                upArr, upType, upNe0, upNe1, upBytes,
                downArr, downType, downNe0, downNe1, downBytes,
                postFfwNormArr, outScaleArr,
                hdArr, kvHeadsArr, isLocalArr, ropeBaseArr, ropeDimsArr,
                donorKArr, donorVArr, donorCacheSizeArr,
                logitsOut, hOut);
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc);

        public static void Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc)
        {
            CheckResult(TSGgml_Gemma4MoELayerDecode(in desc), nameof(TSGgml_Gemma4MoELayerDecode));
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DiffusionDecodeLayer(in DiffusionDecodeLayerArgs desc);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenImageModMlp(in QwenImageModMlpArgs desc);

        public static bool TryQwenImageModMlp(in QwenImageModMlpArgs desc) => TSGgml_QwenImageModMlp(in desc) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenImageJointAttn(in QwenImageJointAttnArgs desc);

        public static bool TryQwenImageJointAttn(in QwenImageJointAttnArgs desc)
        {
            int r = TSGgml_QwenImageJointAttn(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-attn FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenImageBlock(in QwenImageBlockArgs desc);

        public static bool TryQwenImageBlock(in QwenImageBlockArgs desc)
        {
            int r = TSGgml_QwenImageBlock(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-block FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenImageBlockCfg(in QwenImageBlockArgs condDesc, in QwenImageBlockArgs negDesc);

        // CFG-batched block: both true-CFG branches in one dispatch sharing the weights.
        public static bool TryQwenImageBlockCfg(in QwenImageBlockArgs condDesc, in QwenImageBlockArgs negDesc)
        {
            int r = TSGgml_QwenImageBlockCfg(in condDesc, in negDesc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-block-cfg FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenImageForward(in QwenImageForwardArgs desc);

        // Whole 60-block DiT forward in one resident-weight graph (in-graph modulation).
        public static bool TryQwenImageForward(in QwenImageForwardArgs desc)
        {
            int r = TSGgml_QwenImageForward(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-forward FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_WanT5Encode(in WanT5EncodeArgs desc);

        // Whole UMT5-XXL encoder forward in one resident-weight graph.
        public static bool TryWanT5Encode(in WanT5EncodeArgs desc)
        {
            int r = TSGgml_WanT5Encode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[wan-t5 FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_WanDitForward(in WanDitForwardArgs desc);

        // Whole Wan DiT forward (one denoising-step velocity prediction) in one
        // resident-weight graph; persistent + CUDA-graph-captured per shape on CUDA.
        public static bool TryWanDitForward(in WanDitForwardArgs desc)
        {
            int r = TSGgml_WanDitForward(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[wan-dit FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_WanVaeDecode(in WanVaeDecodeArgs desc);

        // Whole Wan 2.1 video VAE decode (chunked causal 3D decoder) in one graph.
        public static bool TryWanVaeDecode(in WanVaeDecodeArgs desc)
        {
            int r = TSGgml_WanVaeDecode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[wan-vae FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3VideoVaeDecode(in H3VideoVaeDecodeArgs desc);

        /// <summary>MiniMax-H3 video VAE ViT decode: one graph for the whole
        /// 36-block transformer, latent tokens in and pixel patches out.</summary>
        public static bool TryMiniMaxH3VideoVaeDecode(in H3VideoVaeDecodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3VideoVaeDecode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-video-vae FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3TextEncode(in H3TextEncodeArgs desc);

        /// <summary>Qwen3-VL text-encoder prefill for MiniMax-H3: the whole
        /// 50-layer trunk in one graph, returning raw hidden states.</summary>
        public static bool TryMiniMaxH3TextEncode(in H3TextEncodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3TextEncode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-te FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3DitForward(in H3DitForwardArgs desc);

        /// <summary>One MiniMax-H3 diffusion step: the whole 50-block packed
        /// audio-video transformer in one graph.</summary>
        public static bool TryMiniMaxH3DitForward(in H3DitForwardArgs desc)
        {
            int r = TSGgml_MiniMaxH3DitForward(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-dit FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3VideoVaeEncode(in H3VideoVaeEncodeArgs desc);

        /// <summary>Single-frame MiniMax-H3 video VAE encode, for image conditioning.</summary>
        public static bool TryMiniMaxH3VideoVaeEncode(in H3VideoVaeEncodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3VideoVaeEncode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-vae-encode FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3AudioVaeDecode(in H3AudioVaeDecodeArgs desc);

        /// <summary>Mono MiniMax-H3 audio VAE (BigVGAN) decode in one graph.</summary>
        public static bool TryMiniMaxH3AudioVaeDecode(in H3AudioVaeDecodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3AudioVaeDecode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-audio-vae FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3VideoVaeEncode3D(in H3VideoVaeEncode3DArgs desc);

        /// <summary>Causal 3-D video encode: a clip of frames to its latent.</summary>
        public static bool TryMiniMaxH3VideoVaeEncode3D(in H3VideoVaeEncode3DArgs desc)
        {
            int r = TSGgml_MiniMaxH3VideoVaeEncode3D(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-video-encode3d FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3AudioVaeEncode(in H3AudioVaeEncodeArgs desc);

        /// <summary>DAC audio encoder: one mono plane of PCM to its latent.</summary>
        public static bool TryMiniMaxH3AudioVaeEncode(in H3AudioVaeEncodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3AudioVaeEncode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-audio-encode FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_MiniMaxH3VisionEncode(in H3VisionEncodeArgs desc);

        /// <summary>Qwen3-VL vision tower: one graph producing the final merger output
        /// plus the three DeepStack outputs.</summary>
        public static bool TryMiniMaxH3VisionEncode(in H3VisionEncodeArgs desc)
        {
            int r = TSGgml_MiniMaxH3VisionEncode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[h3-vision FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_WanVaeEncode(in WanVaeEncodeArgs desc);

        // Whole Wan video VAE encode (chunked causal 3D encoder) in one graph.
        public static bool TryWanVaeEncode(in WanVaeEncodeArgs desc)
        {
            int r = TSGgml_WanVaeEncode(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[wan-vae-encode FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_QwenImageSetOffload(int on);

        // CPU-offload mode for the Qwen-Image DiT kernels: disables the persistent /
        // CUDA-graph-captured entries (whose one-time resident weight upload is their
        // whole point) so the non-persist reuse-gallocr path streams the weights per
        // call. Set per request by the pipeline with the device-copy residency budget.
        public static void QwenImageSetOffload(bool on) => TSGgml_QwenImageSetOffload(on ? 1 : 0);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Conv2d(in Conv2dArgs desc);

        public static bool TryConv2d(in Conv2dArgs desc)
        {
            int r = TSGgml_Conv2d(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[conv2d FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        /// <summary>Fused DiffusionGemma decode layer. Returns true on success; false (without throwing)
        /// when the backend can't run it (e.g. flash-attn unsupported) so the caller falls back.</summary>
        public static bool TryDiffusionDecodeLayer(in DiffusionDecodeLayerArgs desc)
        {
            int r = TSGgml_DiffusionDecodeLayer(in desc);
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DiffusionLmHead(
            IntPtr hidden, int hiddenSize, int canvasLen,
            IntPtr outputNormW,
            IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float eps, float finalLogitSoftcap);

        /// <summary>Fused DiffusionGemma lm_head tail (output_norm + lm_head + softcap) in one GGML graph.
        /// Reads canvas hidden [H*C], writes canvas logits [C*vocab]. Returns false on failure.</summary>
        public static bool TryDiffusionLmHead(
            IntPtr hidden, int hiddenSize, int canvasLen,
            IntPtr outputNormW, IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float eps, float finalLogitSoftcap)
        {
            int r = TSGgml_DiffusionLmHead(hidden, hiddenSize, canvasLen, outputNormW,
                lmHeadW, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, logitsOut, vocab, eps, finalLogitSoftcap);
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DiffusionLmHeadSample(
            IntPtr hidden, int hiddenSize, int canvasLen,
            IntPtr outputNormW,
            IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            int vocab, float eps, float finalLogitSoftcap,
            float invTemp, IntPtr uHost, int topK,
            IntPtr argmaxOut, IntPtr entropyOut, IntPtr sampledOut,
            IntPtr topTokensOut, IntPtr topProbsOut);

        /// <summary>Fused DiffusionGemma lm_head + on-device sample (CUDA only): runs output_norm + lm_head
        /// as one graph producing device logits, then a CUDA kernel computes per canvas position the argmax,
        /// entropy, multinomial sample (with the pre-drawn <paramref name="uHost"/>), and top-K tokens/weights
        /// — downloading only the small per-position outputs instead of the full [vocab,C] logits. Returns
        /// false (caller falls back to <see cref="TryDiffusionLmHead"/> + host sampling) on non-CUDA / failure.</summary>
        public static bool TryDiffusionLmHeadSample(
            IntPtr hidden, int hiddenSize, int canvasLen,
            IntPtr outputNormW, IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            int vocab, float eps, float finalLogitSoftcap,
            float invTemp, IntPtr uHost, int topK,
            IntPtr argmaxOut, IntPtr entropyOut, IntPtr sampledOut, IntPtr topTokensOut, IntPtr topProbsOut)
        {
            int r = TSGgml_DiffusionLmHeadSample(hidden, hiddenSize, canvasLen, outputNormW,
                lmHeadW, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, vocab, eps, finalLogitSoftcap,
                invTemp, uHost, topK, argmaxOut, entropyOut, sampledOut, topTokensOut, topProbsOut);
            return r != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DiffusionModelDecode(
            [In] DiffusionDecodeLayerArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int canvasLen, int promptLen,
            IntPtr outputNormW,
            IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float finalLogitSoftcap);

        /// <summary>Fused DiffusionGemma whole-model decode: all layers + output_norm + lm_head + softcap
        /// in one GGML graph (canvas hidden stays on-device). Writes canvas logits [C*vocab] to logitsOut.
        /// Returns false (caller falls back) when the backend can't run it.</summary>
        public static bool TryDiffusionModelDecode(
            DiffusionDecodeLayerArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int canvasLen, int promptLen,
            IntPtr outputNormW, IntPtr lmHeadW, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr logitsOut, int vocab, float finalLogitSoftcap)
        {
            int r = TSGgml_DiffusionModelDecode(layers, numLayers, hidden, hiddenSize, canvasLen, promptLen,
                outputNormW, lmHeadW, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, logitsOut, vocab, finalLogitSoftcap);
            return r != 0;
        }

        // Model-wide MoE decode: the whole transformer as one graph/token.
        // GPT-OSS whole-model decode: all layers + MoE + folded final norm/LM head
        // in ONE graph dispatch per token (see ggml_ops_gptoss_decode.cpp).
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssModelDecode(
            [In] GptOssLayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm);

        /// <summary>
        /// Runs the whole GPT-OSS transformer for one token as a single graph.
        /// Returns false (with the native error recorded) when the kernel cannot
        /// handle the shape, so the caller can fall back to the per-layer path.
        /// </summary>
        public static bool TryGptOssModelDecode(
            GptOssLayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm)
            => TSGgml_GptOssModelDecode(layers, numLayers, hidden, hiddenSize, position,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm) != 0;

        // Same graph, tensor-parallel plan mode: builds this rank's graph and
        // hands it back UNEXECUTED for tp_execute_plans to drive segment by
        // segment (see ggml_ops_gptoss_decode.cpp).
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssModelDecodeTP(
            [In] GptOssLayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        /// <summary>
        /// Builds one rank's whole-model decode graph and returns a plan pointer
        /// in <paramref name="tpPlanOut"/> instead of running it. Returns false
        /// when the kernel declines, leaving the caller on the per-op TP chain.
        /// </summary>
        public static bool TryGptOssModelDecodeTP(
            GptOssLayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm,
            int tpDegree, IntPtr[] tpPlanOut)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            int rc = TSGgml_GptOssModelDecodeTP(layers, numLayers, hidden, hiddenSize, position,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm,
                tpDegree, tpPlanOut);
            return rc != 0;
        }

        // Same graph, tensor-parallel plan mode.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssModelPrefillTP(
            [In] GptOssLayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int numTokens, int startPos,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        /// <summary>
        /// Builds one rank's whole-model prefill graph and returns a plan pointer
        /// instead of running it.
        /// </summary>
        public static bool TryGptOssModelPrefillTP(
            GptOssLayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize,
            int numTokens, int startPos, IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm,
            int tpDegree, IntPtr[] tpPlanOut)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            int rc = TSGgml_GptOssModelPrefillTP(layers, numLayers, hidden, hiddenSize, numTokens,
                startPos, logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tpDegree, tpPlanOut);
            return rc != 0;
        }

        // GPT-OSS TRUE token-batched decode: N concurrent sequences, one token
        // each, in ONE graph (see ggml_ops_gptoss_batched.cpp). kCaches/vCaches
        // are [layer * nSeqs + seq] HOST cache pointers (the device-window keys).
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssModelDecodeBatched(
            [In] GptOssLayerDecodeArgs[] layers, int numLayers, int nSeqs,
            IntPtr hidden,
            [In] IntPtr[] kCaches, [In] IntPtr[] vCaches,
            [In] int[] cacheSizes, [In] int[] positions,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, IntPtr sampled, int wantLogits);

        /// <summary>
        /// Decode one token for each of N concurrent GPT-OSS sequences in a single
        /// fused graph, each against its own KV cache. Logits land in
        /// <paramref name="logits"/> as [vocab, nSeqs]. Returns false (native
        /// error recorded) when the kernel declines, so the caller falls back to
        /// the round-robin per-sequence path.
        /// </summary>
        public static bool TryGptOssModelDecodeBatched(
            GptOssLayerDecodeArgs[] layers, int numLayers, int nSeqs, IntPtr hidden,
            IntPtr[] kCaches, IntPtr[] vCaches, int[] cacheSizes, int[] positions,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm,
            IntPtr sampled, bool wantLogits)
            => TSGgml_GptOssModelDecodeBatched(layers, numLayers, nSeqs, hidden,
                kCaches, vCaches, cacheSizes, positions,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm,
                sampled, wantLogits ? 1 : 0) != 0;

        // GPT-OSS whole-model prefill: N tokens through every layer + MoE +
        // folded final norm/LM head in ONE graph (see ggml_ops_gptoss_prefill.cpp).
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssModelPrefill(
            [In] GptOssLayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int numTokens, int startPos,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm);

        /// <summary>
        /// Runs the whole GPT-OSS transformer over a prompt chunk as a single
        /// graph, leaving the last token's logits in <paramref name="logits"/>.
        /// Returns false (with the native error recorded) when the kernel cannot
        /// handle the shape, so the caller can fall back to the per-layer path.
        /// </summary>
        public static bool TryGptOssModelPrefill(
            GptOssLayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize,
            int numTokens, int startPos,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType,
            long lmHeadNe0, long lmHeadNe1, long lmHeadBytes, IntPtr finalNorm)
            => TSGgml_GptOssModelPrefill(layers, numLayers, hidden, hiddenSize, numTokens, startPos,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_GptOssResetDecodeCache();

        /// <summary>
        /// Drops every cached GPT-OSS whole-model decode graph. Call before a
        /// prefill and on any KV reset/grow: the captured graph pins the compute
        /// pool and the KV windows.
        /// </summary>
        public static void GptOssResetDecodeCache() => TSGgml_GptOssResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_GptOssResetBatchedDecodeCache();

        /// <summary>
        /// Drops the token-batched GPT-OSS decode state (slot-stable arena
        /// graphs). Unlike the solo pool this survives prefills; call it on
        /// model teardown so a disposed model's arenas release their VRAM.
        /// Dirty arena rows are flushed to the host mirrors first.
        /// </summary>
        public static void GptOssResetBatchedDecodeCache() => TSGgml_GptOssResetBatchedDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GptOssSyncKvCacheToHost(
            IntPtr kCache, IntPtr vCache, int cacheSize, int rows);

        /// <summary>
        /// Copies the device-resident GPT-OSS KV rows back into their host mirror.
        /// The fused decode graph never writes them back per token, so anything
        /// reading the host cache must call this first.
        /// </summary>
        public static void GptOssSyncKvCacheToHost(IntPtr kCache, IntPtr vCache, int cacheSize, int rows)
            => TSGgml_GptOssSyncKvCacheToHost(kCache, vCache, cacheSize, rows);

        // `layers` is one Gemma4MoELayerDecodeArgs per layer (blittable, marshalled
        // as a contiguous TSGgmlGemma4MoELayerDesc array). hidden/position come from
        // the explicit params; the per-element Hidden/Position fields are ignored.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4MoEModelDecode(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position)
        {
            CheckResult(TSGgml_Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position,
                IntPtr.Zero, 0, IntPtr.Zero, 0, 0, 0, 0, IntPtr.Zero, 0.0f,
                1, null), nameof(TSGgml_Gemma4MoEModelDecode));
        }

        // Folded variant: appends final-norm + lm_head + softcap so logits[vocab] are
        // written to <paramref name="logits"/> as part of the captured replay.
        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            CheckResult(TSGgml_Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm, logitSoftcap,
                tpDegree, tpPlanOut),
                nameof(TSGgml_Gemma4MoEModelDecode));
        }

        // TRUE token-batched MoE decode: N concurrent sequences, one token each, in
        // one captured graph. Reuses the per-layer descriptor array for weights;
        // KV caches are per-(layer,seq) [layer*nSeqs+seq]; positions per seq.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4MoEModelDecodeBatched(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers, int nSeqs,
            IntPtr hidden,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] positions,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap);

        public static bool Gemma4MoEModelDecodeBatched(Gemma4MoELayerDecodeArgs[] layers, int numLayers, int nSeqs,
            IntPtr hidden, IntPtr[] kCacheArr, IntPtr[] vCacheArr, int[] positions,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap)
        {
            int rc = TSGgml_Gemma4MoEModelDecodeBatched(layers, numLayers, nSeqs, hidden,
                kCacheArr, vCacheArr, positions, logits, vocabSize,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm, logitSoftcap);
            return rc != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4ResetMoEBatchedDecodeCache();
        public static void Gemma4ResetMoEBatchedDecodeCache() => TSGgml_Gemma4ResetMoEBatchedDecodeCache();

        // Model-wide MoE multi-token verify: the whole MoE transformer over N tokens
        // as one graph. Reuses the same descriptor array as the decode; start_pos +
        // num_tokens are explicit. Returns 0 (false) when the kernel cannot handle
        // the shape so the caller falls back to the per-op verify.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Gemma4MoEModelVerify(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int startPos, int numTokens,
            byte[] mmIsExcept,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4MoEReleaseVerifyTpGraphs();

        public static void Gemma4MoEReleaseVerifyTpGraphs()
        {
            try { TSGgml_Gemma4MoEReleaseVerifyTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        public static bool Gemma4MoEModelVerify(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int startPos, int numTokens,
            byte[] mmIsExcept = null, int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            int rc = TSGgml_Gemma4MoEModelVerify(layers, numLayers, hidden, hiddenSize, startPos, numTokens,
                mmIsExcept, tpDegree, tpPlanOut);
            return rc != 0;
        }

        // Qwen3.5/3.6 full-model decode: the whole hybrid transformer (full-attention
        // + GatedDeltaNet recurrent layers + per-layer FFN) as one graph/token.
        // Returns 0 when it cannot handle the shape so the caller falls back to per-op.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ModelDecode(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers,
            [MarshalAs(UnmanagedType.Bool)] bool reseedState,
            IntPtr hidden, int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            int tpDegree, [In, Out] IntPtr[] tpPlanOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ModelDecodeToken(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers,
            [MarshalAs(UnmanagedType.Bool)] bool reseedState,
            int tokenId,
            IntPtr tokenEmbedding, int tokenEmbeddingType,
            long tokenEmbeddingNe0, long tokenEmbeddingNe1, long tokenEmbeddingBytes,
            int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ResetDecodeCache();

        // Qwen3.5/3.8 SLOT-STABLE ARENA token-batched decode (the GPT-OSS arena
        // design ported to the hybrid GDN + attention family; see
        // ggml_ops_qwen35_batched_arena.cpp). kCaches/vCaches are
        // [attn_layer * n + s]; convStates/deltaStates are [gdn_layer * n + s].
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ArenaDecodeBatched(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers, int nSeqs,
            [In] int[] tokenIds, [In] int[] positions,
            [In] IntPtr[] kCaches, [In] IntPtr[] vCaches,
            [In] IntPtr[] convStates, [In] IntPtr[] deltaStates,
            [In] int[] gdnHostAuth, [In] int[] cacheSizes,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr sampled, int wantLogits);

        public static bool TryQwen35ArenaDecodeBatched(
            Qwen35LayerDecodeArgs[] layers, int numLayers, int nSeqs,
            int[] tokenIds, int[] positions,
            IntPtr[] kCaches, IntPtr[] vCaches,
            IntPtr[] convStates, IntPtr[] deltaStates,
            int[] gdnHostAuth, int[] cacheSizes,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            IntPtr tokenEmbd, int tokenEmbdType,
            long tokenEmbdNe0, long tokenEmbdNe1, long tokenEmbdBytes,
            IntPtr sampled, bool wantLogits)
            => TSGgml_Qwen35ArenaDecodeBatched(layers, numLayers, nSeqs, tokenIds, positions,
                kCaches, vCaches, convStates, deltaStates, gdnHostAuth, cacheSizes,
                numHeads, numKvHeads, headDim, ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf, normTopk, expertWeightsScale,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tokenEmbd, tokenEmbdType, tokenEmbdNe0, tokenEmbdNe1, tokenEmbdBytes,
                sampled, wantLogits ? 1 : 0) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ArenaResetBatchedDecodeCache();

        /// <summary>Drops the qwen35 slot-stable arena batched-decode state
        /// (flushing dirty slots to their host bytes first). Survives prefills
        /// and holder churn by design; call on model teardown.</summary>
        public static void Qwen35ArenaResetBatchedDecodeCache() => TSGgml_Qwen35ArenaResetBatchedDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ArenaFlushHostPointer(IntPtr hostPtr);

        /// <summary>Flush-and-retire the qwen35 arena slot registered for this
        /// host cache/state pointer (no-op when none) — call before any managed
        /// path reads or replaces a holder's caches outside the hooked
        /// kernels (growth, host sync, snapshot extraction).</summary>
        public static void Qwen35ArenaFlushHostPointer(IntPtr hostPtr) => TSGgml_Qwen35ArenaFlushHostPointer(hostPtr);

        public static void Qwen35ResetDecodeCache() => TSGgml_Qwen35ResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4ResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4ReleaseVerifyTpGraphs();

        public static void Gemma4ReleaseVerifyTpGraphs()
        {
            try { TSGgml_Gemma4ReleaseVerifyTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        public static void Gemma4ResetDecodeCache() => TSGgml_Gemma4ResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4ResetBatchedDecodeCache();

        public static void Gemma4ResetBatchedDecodeCache() => TSGgml_Gemma4ResetBatchedDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Gemma4MoEResetDecodeCache();

        public static void Gemma4MoEResetDecodeCache() => TSGgml_Gemma4MoEResetDecodeCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ResetVerifyCache();

        public static void Qwen35ResetVerifyCache() => TSGgml_Qwen35ResetVerifyCache();

        // Qwen3.5/3.6 TRUE token-batched fused decode: N sequences' decode tokens
        // (one per sequence) through the whole hybrid transformer in ONE graph,
        // with PAGED KV (slot_mapping write + per-seq get_rows gather). Returns 0
        // when it cannot handle the shape so the caller falls back to op-by-op.
        // padKv = fixed per-seq gather length (round_up(maxSeqLen, stride)); gatherIdx is
        // [nSeqs*padKv] (real slots then pad), seqLens [nSeqs] drives the per-seq attn mask.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ModelDecodeBatched(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int nTokens, int nSeqs,
            IntPtr positions, IntPtr slotMapping,
            IntPtr gatherIdx, IntPtr seqLens, int padKv, int totalSlots,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ResetBatchedDecodeCache();

        public static void Qwen35ResetBatchedDecodeCache() => TSGgml_Qwen35ResetBatchedDecodeCache();

        public static bool Qwen35ModelDecodeBatched(
            Qwen35LayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int nTokens, int nSeqs,
            IntPtr positions, IntPtr slotMapping,
            IntPtr gatherIdx, IntPtr seqLens, int padKv, int totalSlots,
            int numHeads, int numKvHeads, int headDim,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale)
        {
            return TSGgml_Qwen35ModelDecodeBatched(
                layers, numLayers, hidden, hiddenSize, nTokens, nSeqs,
                positions, slotMapping, gatherIdx, seqLens, padKv, totalSlots,
                numHeads, numKvHeads, headDim, ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf,
                normTopk, expertWeightsScale) != 0;
        }

        public static bool Qwen35ModelDecode(
            Qwen35LayerDecodeArgs[] layers, int numLayers,
            bool reseedState,
            IntPtr hidden, int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            return TSGgml_Qwen35ModelDecode(
                layers, numLayers, reseedState,
                hidden, hiddenSize, position,
                numHeads, numKvHeads, headDim, cacheSize,
                ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf,
                normTopk, expertWeightsScale,
                logits, vocabSize,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, tpDegree, tpPlanOut) != 0;
        }

        public static bool Qwen35ModelDecodeToken(
            Qwen35LayerDecodeArgs[] layers, int numLayers,
            bool reseedState,
            int tokenId,
            IntPtr tokenEmbedding, int tokenEmbeddingType,
            long tokenEmbeddingNe0, long tokenEmbeddingNe1, long tokenEmbeddingBytes,
            int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm)
        {
            return TSGgml_Qwen35ModelDecodeToken(
                layers, numLayers, reseedState,
                tokenId,
                tokenEmbedding, tokenEmbeddingType,
                tokenEmbeddingNe0, tokenEmbeddingNe1, tokenEmbeddingBytes,
                hiddenSize, position,
                numHeads, numKvHeads, headDim, cacheSize,
                ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf,
                normTopk, expertWeightsScale,
                logits, vocabSize,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm) != 0;
        }

        // Qwen3.5/3.6 fused multi-token VERIFY: the whole hybrid transformer over
        // N tokens of ONE sequence as a single graph (prefill-style causal attention,
        // GDN recurrence via gated_delta_net(K=N), batched MoE/dense FFN, folded
        // final-norm). Outputs per-row logits [vocab, N] AND post-norm hidden
        // [hidden, N] (normedOut, for the MTP draft head). GDN state advances from
        // each layer's ConvStateIn/DeltaStateIn to ConvStateOut/DeltaStateOut.
        // Returns 0 when it cannot handle the shape so the caller falls back to per-op.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35ModelVerify(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int startPos, int numTokens,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, IntPtr normedOut, int nLogitRows,
            int[] mropePos, int[] mropeSections,
            int tpDegree, IntPtr[] tpPlanOut,
            IntPtr captureData, int[] captureLayers, int captureCount,
            int stateSnapshots, IntPtr stateSnapshotsUsed, int deviceStateCurrent,
            int deferStateDownload);

        public static bool Qwen35ModelVerify(
            Qwen35LayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int startPos, int numTokens,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, IntPtr normedOut, int nLogitRows,
            int[] mropePos = null, int[] mropeSections = null,
            int tpDegree = 1, IntPtr[] tpPlanOut = null,
            IntPtr captureData = default, int[] captureLayers = null, int captureCount = 0,
            int stateSnapshots = 1, IntPtr stateSnapshotsUsed = default,
            bool deviceStateCurrent = false, bool deferStateDownload = false)
        {
            return TSGgml_Qwen35ModelVerify(
                layers, numLayers, hidden, hiddenSize, startPos, numTokens,
                numHeads, numKvHeads, headDim, cacheSize,
                ropeNDims, ropeMode, kvCacheType,
                convKernel, headKDim, headVDim, numKHeads, numVHeads,
                eps, ropeBase, ropeFreqScale,
                numExperts, numExpertsUsed, expertFf, sharedFf,
                normTopk, expertWeightsScale,
                logits, vocabSize,
                lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNorm, normedOut, nLogitRows, mropePos, mropeSections,
                tpDegree, tpPlanOut, captureData, captureLayers, captureCount,
                stateSnapshots, stateSnapshotsUsed, deviceStateCurrent ? 1 : 0,
                deferStateDownload ? 1 : 0) != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35CommitStateSnapshot(int slot, int numRecurrentLayers);

        /// <summary>
        /// Commit one recurrent-state snapshot into the live device state, without a
        /// host round trip. The next verify can then skip its state upload, which is
        /// the point: that upload plus the matching download was the largest per-step
        /// cost of speculative decoding on a Qwen 3.5/3.8 hybrid trunk.
        ///
        /// <paramref name="slot"/> counts back from the end of the verified batch;
        /// -1 means the post-window state, which is what a single-row step commits.
        /// </summary>
        public static bool Qwen35CommitStateSnapshot(int slot, int numRecurrentLayers)
            => TSGgml_Qwen35CommitStateSnapshot(slot, numRecurrentLayers) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35DrainDeviceState(
            IntPtr[] convOut, IntPtr[] deltaOut, int numRecurrentLayers);

        /// <summary>Read the live device recurrent state back into the host mirrors,
        /// for anything that has to run the op-by-op recurrent path.</summary>
        public static bool Qwen35DrainDeviceState(IntPtr[] convOut, IntPtr[] deltaOut, int numRecurrentLayers)
            => TSGgml_Qwen35DrainDeviceState(convOut, deltaOut, numRecurrentLayers) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35FetchStateSnapshot(
            int slot, IntPtr[] convOut, IntPtr[] deltaOut, int numRecurrentLayers);

        /// <summary>
        /// Pull ONE per-token recurrent-state snapshot out of the verify that just
        /// ran, counting <paramref name="slot"/> tokens back from the end of that
        /// batch. False when there is nothing to pull (no snapshotting verify has
        /// run, or the slot is out of range), and the caller keeps its old
        /// restore-and-re-forward path.
        /// </summary>
        public static bool Qwen35FetchStateSnapshot(int slot, IntPtr[] convOut, IntPtr[] deltaOut,
            int numRecurrentLayers)
            => TSGgml_Qwen35FetchStateSnapshot(slot, convOut, deltaOut, numRecurrentLayers) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35ReleaseVerifyTpGraphs();

        public static void Qwen35ReleaseVerifyTpGraphs()
        {
            try { TSGgml_Qwen35ReleaseVerifyTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35RecurrentLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int n,
            IntPtr attnNormW,
            IntPtr gdnQkvW, int gdnQkvType, long gdnQkvNe0, long gdnQkvNe1, long gdnQkvBytes,
            IntPtr gdnGateW, int gdnGateType, long gdnGateNe0, long gdnGateNe1, long gdnGateBytes,
            IntPtr ssmBetaW, int ssmBetaType, long ssmBetaNe0, long ssmBetaNe1, long ssmBetaBytes,
            IntPtr ssmAlphaW, int ssmAlphaType, long ssmAlphaNe0, long ssmAlphaNe1, long ssmAlphaBytes,
            IntPtr ssmOutW, int ssmOutType, long ssmOutNe0, long ssmOutNe1, long ssmOutBytes,
            IntPtr conv1dW, IntPtr ssmDtW, IntPtr ssmAW, IntPtr ssmNormW,
            IntPtr convStateIn, IntPtr deltaStateIn,
            IntPtr convStateOut, IntPtr deltaStateOut,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps);

        public static bool Qwen35RecurrentLayerPrefill(
            IntPtr hiddenData, int hiddenSize, int n,
            IntPtr attnNormW,
            IntPtr gdnQkvW, int gdnQkvType, long gdnQkvNe0, long gdnQkvNe1, long gdnQkvBytes,
            IntPtr gdnGateW, int gdnGateType, long gdnGateNe0, long gdnGateNe1, long gdnGateBytes,
            IntPtr ssmBetaW, int ssmBetaType, long ssmBetaNe0, long ssmBetaNe1, long ssmBetaBytes,
            IntPtr ssmAlphaW, int ssmAlphaType, long ssmAlphaNe0, long ssmAlphaNe1, long ssmAlphaBytes,
            IntPtr ssmOutW, int ssmOutType, long ssmOutNe0, long ssmOutNe1, long ssmOutBytes,
            IntPtr conv1dW, IntPtr ssmDtW, IntPtr ssmAW, IntPtr ssmNormW,
            IntPtr convStateIn, IntPtr deltaStateIn,
            IntPtr convStateOut, IntPtr deltaStateOut,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps)
        {
            return TSGgml_Qwen35RecurrentLayerPrefill(
                hiddenData, hiddenSize, n, attnNormW,
                gdnQkvW, gdnQkvType, gdnQkvNe0, gdnQkvNe1, gdnQkvBytes,
                gdnGateW, gdnGateType, gdnGateNe0, gdnGateNe1, gdnGateBytes,
                ssmBetaW, ssmBetaType, ssmBetaNe0, ssmBetaNe1, ssmBetaBytes,
                ssmAlphaW, ssmAlphaType, ssmAlphaNe0, ssmAlphaNe1, ssmAlphaBytes,
                ssmOutW, ssmOutType, ssmOutNe0, ssmOutNe1, ssmOutBytes,
                conv1dW, ssmDtW, ssmAW, ssmNormW,
                convStateIn, deltaStateIn, convStateOut, deltaStateOut,
                convKernel, headKDim, headVDim, numKHeads, numVHeads, eps) != 0;
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_Qwen35GdnLayerTP(
            IntPtr hiddenData, int hiddenSize, int n,
            IntPtr attnNormW,
            IntPtr inprojW, int inprojType, long inprojNe0, long inprojNe1, long inprojBytes,
            IntPtr conv1dW, IntPtr dtBias, IntPtr aLog, IntPtr ssmNormW,
            IntPtr convState, IntPtr deltaState,
            IntPtr gatedOut,
            int packedDim, int qkvDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, float eps);

        /// <summary>
        /// One rank's GatedDeltaNet block (norm + packed in-projection + conv +
        /// delta-rule scan + gated norm) as a single ggml graph. The recurrent
        /// state stays device-resident between calls, keyed on
        /// <paramref name="convState"/> / <paramref name="deltaState"/>.
        /// </summary>
        public static void Qwen35GdnLayerTP(
            IntPtr hiddenData, int hiddenSize, int n,
            IntPtr attnNormW,
            IntPtr inprojW, int inprojType, long inprojNe0, long inprojNe1, long inprojBytes,
            IntPtr conv1dW, IntPtr dtBias, IntPtr aLog, IntPtr ssmNormW,
            IntPtr convState, IntPtr deltaState,
            IntPtr gatedOut,
            int packedDim, int qkvDim, int qkDim, int vDim,
            int numKHeads, int numVHeads, int headKDim, int headVDim,
            int convKernel, float eps)
        {
            CheckResult(TSGgml_Qwen35GdnLayerTP(
                hiddenData, hiddenSize, n, attnNormW,
                inprojW, inprojType, inprojNe0, inprojNe1, inprojBytes,
                conv1dW, dtBias, aLog, ssmNormW,
                convState, deltaState, gatedOut,
                packedDim, qkvDim, qkDim, vDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, eps), "qwen35_gdn_layer_tp");
        }

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Qwen35GdnDropTpGraphs();

        /// <summary>Free every cached per-rank TP GatedDeltaNet graph.</summary>
        public static void Qwen35GdnDropTpGraphs() => TSGgml_Qwen35GdnDropTpGraphs();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GatedDeltaNetChunkedF32(
            GgmlTensorView3D q,
            GgmlTensorView3D k,
            GgmlTensorView3D v,
            GgmlTensorView3D z,
            GgmlTensorView2D alpha,
            GgmlTensorView2D beta,
            GgmlTensorView3D state,
            GgmlTensorView3D gatedOut,
            IntPtr dtBiasData,
            IntPtr aLogData,
            IntPtr ssmNormWData,
            int chunkSize,
            float eps,
            int gateMode);

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpFfnBlock(
            ref Qwen4ExpFfnArgs args,
            IntPtr resData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int nExpert, int nExpertUsed, int nFf, int nFfShared,
            float eps, int cacheSlot, int resResident);

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpGdnBlock(
            ref Qwen4ExpGdnArgs args,
            IntPtr resData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            float eps, int cacheSlot, int resResident);

        [LibraryImport(DllName)]
        internal static partial void TSGgml_Qwen4ExpResetFfnCache();

        [LibraryImport(DllName)]
        internal static partial void TSGgml_Qwen4ExpInvalidateSeqState(IntPtr key);

        [LibraryImport(DllName)]
        internal static partial void TSGgml_Qwen4ExpReleaseAllSeqState();

        [LibraryImport(DllName)]
        internal static unsafe partial void TSGgml_Qwen4ExpReleaseSeqState(IntPtr* keys, int n);

        /// <summary>Re-arm the one-time seed upload for one recurrent-state entry
        /// (keyed by its host seed pointer); the next graph build re-uploads from
        /// the host copy. Used after the managed reset zeroes that copy.</summary>
        public static void Qwen4ExpInvalidateSeqState(IntPtr key) => TSGgml_Qwen4ExpInvalidateSeqState(key);

        /// <summary>Free every native sequence-state entry and cached graph
        /// (model dispose).</summary>
        public static void Qwen4ExpReleaseAllSeqState() => TSGgml_Qwen4ExpReleaseAllSeqState();

        /// <summary>Free the device recurrent-state entries of a released sequence
        /// holder and drop every cached graph (surviving holders rebuild and
        /// re-bind their own still-alive entries).</summary>
        public static unsafe void Qwen4ExpReleaseSeqState(IntPtr[] keys)
        {
            if (keys == null || keys.Length == 0) return;
            fixed (IntPtr* k = keys)
            {
                TSGgml_Qwen4ExpReleaseSeqState(k, keys.Length);
            }
        }

        /// <summary>
        /// One graph for the hyper-connection mixer, the 512-expert MoE and the
        /// scatter back into the wide residual. Returns false when the backend
        /// declines the shape, so the caller falls back to the op-by-op path.
        /// </summary>
        public static bool Qwen4ExpFfnBlock(ref Qwen4ExpFfnArgs args, IntPtr resData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int nExpert, int nExpertUsed, int nFf, int nFfShared, float eps, int cacheSlot,
            bool resResident)
        {
            return TSGgml_Qwen4ExpFfnBlock(ref args, resData,
                nEmbd, hc, hcLowRank, nTokens,
                nExpert, nExpertUsed, nFf, nFfShared, eps, cacheSlot,
                resResident ? 1 : 0) != 0;
        }

        public static bool Qwen4ExpGdnBlock(ref Qwen4ExpGdnArgs args, IntPtr resData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            float eps, int cacheSlot, bool resResident)
        {
            return TSGgml_Qwen4ExpGdnBlock(ref args, resData, nEmbd, hc, hcLowRank, nTokens,
                headKDim, headVDim, nKHeads, nVHeads, dConv, eps, cacheSlot,
                resResident ? 1 : 0) != 0;
        }

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpAttnBlock(
            ref Qwen4ExpAttnArgs args,
            IntPtr resData,
            IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            float eps, int cacheSlot, int resResident);

        public static bool Qwen4ExpAttnBlock(ref Qwen4ExpAttnArgs args, IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            float eps, int cacheSlot, bool resResident)
        {
            return TSGgml_Qwen4ExpAttnBlock(ref args, resData, maskData, nEmbd, hc, hcLowRank,
                nTokens, headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                nRot, ropeBase, ropeFreqScale, attnScale, eps, cacheSlot,
                resResident ? 1 : 0) != 0;
        }

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpTokenSpan(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd,
            IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot, int firstFfnOnly,
            IntPtr head, IntPtr logitsOut,
            IntPtr ple, int pleLayer, IntPtr pleEmb,
            IntPtr mropePos, IntPtr mropeSections, int ropePosition,
            int device);

        public static bool Qwen4ExpTokenSpan(
            IntPtr ffn, IntPtr gdn, IntPtr attn, IntPtr kinds,
            int layerBegin, int layerEnd,
            IntPtr resData, IntPtr maskData,
            int nEmbd, int hc, int hcLowRank, int nTokens,
            int headKDim, int headVDim, int nKHeads, int nVHeads, int dConv,
            int headDim, int nHead, int nHeadKv, int kvCapacity, int nKv, int position,
            int nRot, float ropeBase, float ropeFreqScale, float attnScale,
            int nExpert, int nExpertUsed, int nFf, int nFfSh,
            float eps, int cacheSlot, bool firstFfnOnly,
            IntPtr head, IntPtr logitsOut,
            IntPtr ple, int pleLayer, IntPtr pleEmb,
            IntPtr mropePos, IntPtr mropeSections, int ropePosition, int device)
        {
            return TSGgml_Qwen4ExpTokenSpan(ffn, gdn, attn, kinds, layerBegin, layerEnd,
                resData, maskData, nEmbd, hc, hcLowRank, nTokens,
                headKDim, headVDim, nKHeads, nVHeads, dConv,
                headDim, nHead, nHeadKv, kvCapacity, nKv, position,
                nRot, ropeBase, ropeFreqScale, attnScale,
                nExpert, nExpertUsed, nFf, nFfSh, eps, cacheSlot,
                firstFfnOnly ? 1 : 0, head, logitsOut, ple, pleLayer, pleEmb,
                mropePos, mropeSections, ropePosition, device) != 0;
        }

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpResUpload(IntPtr data, long bytes);

        [LibraryImport(DllName)]
        private static partial int TSGgml_Qwen4ExpResDownload(IntPtr data, long bytes);

        public static bool Qwen4ExpResUpload(IntPtr data, long bytes)
            => TSGgml_Qwen4ExpResUpload(data, bytes) != 0;

        public static bool Qwen4ExpResDownload(IntPtr data, long bytes)
            => TSGgml_Qwen4ExpResDownload(data, bytes) != 0;

        public static void Qwen4ExpResetFfnCache() => TSGgml_Qwen4ExpResetFfnCache();

        // Mirrors NemoMamba2BatchedSeqDesc in ggml_ops_mamba2.cpp; same 32-byte
        // POD layout on 64-bit (two ints, two padding ints, two pointers).
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_NemotronMamba2BatchedStepF32(
            int numSeqs,
            [In, Out] NemoMamba2BatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int dInProjTotal,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            IntPtr convWt,
            IntPtr convBias,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr dData,
            IntPtr ssmNormW,
            float eps,
            IntPtr outBatched);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GatedDeltaNetBatchedStepF32(
            int numSeqs,
            [In, Out] GdnBatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int zDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int ssmDInner,
            IntPtr convWt,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr ssmNormW,
            float eps,
            IntPtr gatedOut);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_NemotronMamba2PrefillF32(
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_NemotronMamba2DecodeF32(
            ulong stateKey,
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            int initializeState,
            int downloadState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_NemotronMamba2DecodeClear(ulong modelKey);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial IntPtr TSGgml_AlignedAlloc(UIntPtr size);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_AlignedFree(IntPtr ptr);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_ClearHostBufferCache();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_Shutdown();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_ReleaseReuseComputeBuffers();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_InvalidateHostBuffer(IntPtr ptr);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_SyncHostBuffer(IntPtr ptr, long byteCount);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial long TSGgml_DeviceCopyCacheResidentBytes();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetBackendMemory(out long freeBytes, out long totalBytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_IsActiveDeviceIntegrated();

        // Async dispatch (deferred ggml_backend_synchronize). When enabled, per-op
        // kernels return without waiting on the Metal command buffer; subsequent ops
        // chain through the Metal command queue, and host-side reads must call
        // TSGgml_HostReadBarrier first to drain pending GPU work. See
        // GgmlStorage.EnsureHostReadable for the C# entry point that triggers this.
        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_SetAsyncCompute(int enabled);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_SetHostMoeThreads(int threads);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_GetAsyncCompute();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_HostReadBarrier();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_RegisterOffloadable(IntPtr key);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_SetOffloadableBudget(long bytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_ClearOffloadableState();

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_SetDeviceCopyBudget(long bytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DeviceMemoryInfo(out long freeBytes, out long totalBytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_RegisterPinnedHostBuffer(IntPtr ptr, long bytes);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void TSGgml_UnregisterPinnedHostBuffer(IntPtr ptr);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial UIntPtr TSGgml_RowSize(int ggmlType, long ne);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_DequantizeToF32(int ggmlType, IntPtr src, long numElements, IntPtr dst);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_ApplyLoraDelta(IntPtr w, int ggmlType, long ne0, long ne1,
            IntPtr up, IntPtr down, int rank, float scale, int nThreads);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial void ggml_quantize_init(int type);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        [return: MarshalAs(UnmanagedType.I1)]
        private static partial bool ggml_quantize_requires_imatrix(int type);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial UIntPtr ggml_quantize_chunk(int type, IntPtr src, IntPtr dst,
            long start, long nrows, long nPerRow, IntPtr imatrix);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenVaeRun(in QwenVaeArgs args);

        /// <summary>Run a whole VAE encode/decode op-list as ONE device graph (see QwenVaeArgs).
        /// Returns false when the backend can't run it (caller falls back to the per-conv path).</summary>
        internal static bool TryQwenVaeRun(in QwenVaeArgs args) => TSGgml_QwenVaeRun(in args) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_QwenTeTrunk(in QwenTeTrunkArgs args);

        /// <summary>Run a whole conditioning-encoder transformer trunk as ONE device graph
        /// (see QwenTeTrunkArgs). Returns false when the backend can't run it (caller falls
        /// back to the per-op path).</summary>
        internal static bool TryQwenTeTrunk(in QwenTeTrunkArgs args) => TSGgml_QwenTeTrunk(in args) != 0;

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_CopyF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_UnaryF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_BinaryTensorF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D lhs,
            GgmlTensorView4D rhs);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedActMulF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D a,
            GgmlTensorView4D b);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_FusedActMulSplitF32(
            int op,
            GgmlTensorView2D result,
            GgmlTensorView2D gateUp,
            int halfDim);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_BinaryScalarF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            float scalar);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_NormF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D gamma,
            GgmlTensorView4D beta,
            int hasBeta,
            float eps);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_IndexSelectF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlContiguousTensor indices,
            int addToResult);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_IndexSelectGradF32(
            GgmlTensorView2D grad,
            GgmlTensorView2D adj,
            GgmlContiguousTensor indices);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_RoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            int seqLen,
            int rowOffset,
            int addToResult,
            int invertPositions);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_RoPEExF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            int invertPositions);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_RoPEMRoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int sect0, int sect1, int sect2, int sect3,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow);

        [LibraryImport(DllName)]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int TSGgml_RoPEExFreqFactorsF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            int addToResult,
            int invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen);

        public static void EnsureAvailable(GgmlBackendType backendType)
        {
            if (backendType == GgmlBackendType.Metal && !OperatingSystem.IsMacOS())
            {
                throw new PlatformNotSupportedException("The GGML Metal backend is available on macOS only.");
            }

            if (backendType == GgmlBackendType.Cuda && !IsCudaPlatformSupported())
            {
                throw new PlatformNotSupportedException("The GGML CUDA backend is supported on Windows and Linux only.");
            }

            if (backendType == GgmlBackendType.Vulkan && !IsVulkanPlatformSupported())
            {
                throw new PlatformNotSupportedException("The GGML Vulkan backend is supported on Windows and Linux only.");
            }

            if (backendType == GgmlBackendType.Vulkan)
            {
                ApplyVulkanDeviceFromEnvironment();
            }

            try
            {
                if (TSGgml_IsBackendAvailable((int)backendType) == 0)
                {
                    string backendName = backendType switch
                    {
                        GgmlBackendType.Metal => "ggml-metal",
                        GgmlBackendType.Cuda => "ggml-cuda",
                        GgmlBackendType.Vulkan => "ggml-vulkan",
                        _ => "ggml-cpu",
                    };
                    throw new InvalidOperationException($"Failed to initialize {backendName}. {GetBackendAvailabilityHint(backendType)}");
                }
            }
            catch (DllNotFoundException ex)
            {
                throw new InvalidOperationException("Failed to load the native GGML bridge. Build `TensorSharp.GGML.Native` first.", ex);
            }
            catch (EntryPointNotFoundException ex)
            {
                throw new InvalidOperationException("The native GGML bridge is out of date. Rebuild `TensorSharp.GGML.Native`.", ex);
            }
        }

        public static bool CanInitialize(GgmlBackendType backendType)
        {
            if (backendType == GgmlBackendType.Metal && !OperatingSystem.IsMacOS())
            {
                return false;
            }

            if (backendType == GgmlBackendType.Cuda && !IsCudaPlatformSupported())
            {
                return false;
            }

            if (backendType == GgmlBackendType.Vulkan && !IsVulkanPlatformSupported())
            {
                return false;
            }

            try
            {
                return TSGgml_CanInitializeBackend((int)backendType) != 0;
            }
            catch (DllNotFoundException)
            {
                return false;
            }
            catch (EntryPointNotFoundException)
            {
                return false;
            }
        }

        /// <summary>
        /// Environment variable holding the Vulkan device index for the GGML Vulkan
        /// backend (multi-GPU hosts, e.g. an integrated GPU next to a discrete one).
        /// Read from managed code when the Vulkan backend initializes — the native
        /// bridge cannot see env vars set after process start on Windows (the CRT
        /// snapshots the environment), so the value is pushed down via
        /// <see cref="SetVulkanDeviceIndex"/> instead.
        /// </summary>
        public const string VulkanDeviceEnvVar = "TS_GGML_VULKAN_DEVICE";

        /// <summary>
        /// Selects which Vulkan device the GGML Vulkan backend initializes on.
        /// Must be called before the backend first initializes; the device cannot
        /// change afterwards.
        /// </summary>
        public static void SetVulkanDeviceIndex(int deviceIndex)
        {
            try
            {
                if (TSGgml_SetVulkanDeviceIndex(deviceIndex) == 0)
                {
                    throw new InvalidOperationException(
                        GetLastErrorMessage($"Failed to select Vulkan device {deviceIndex}."));
                }
            }
            catch (EntryPointNotFoundException ex)
            {
                // Older native bridges hardcode device 0, so requesting it is a no-op.
                if (deviceIndex == 0)
                    return;
                throw new InvalidOperationException(
                    "The native GGML bridge is out of date and does not support Vulkan device selection. Rebuild `TensorSharp.GGML.Native`.", ex);
            }
        }

        /// <summary>Number of Vulkan devices visible to ggml-vulkan, or 0 when the build has no Vulkan support.</summary>
        public static int GetVulkanDeviceCount()
        {
            try
            {
                return Math.Max(0, TSGgml_GetVulkanDeviceCount());
            }
            catch (DllNotFoundException)
            {
                return 0;
            }
            catch (EntryPointNotFoundException)
            {
                return 0;
            }
        }

        /// <summary>Human-readable description of a Vulkan device (adapter name), or null when unavailable.</summary>
        public static string GetVulkanDeviceDescription(int deviceIndex)
        {
            byte[] buffer = new byte[256];
            try
            {
                if (TSGgml_GetVulkanDeviceDescription(deviceIndex, buffer, buffer.Length) == 0)
                    return null;
            }
            catch (DllNotFoundException)
            {
                return null;
            }
            catch (EntryPointNotFoundException)
            {
                return null;
            }

            int length = Array.IndexOf(buffer, (byte)0);
            if (length < 0)
                length = buffer.Length;
            return length == 0 ? null : System.Text.Encoding.UTF8.GetString(buffer, 0, length);
        }

        private static void ApplyVulkanDeviceFromEnvironment()
        {
            string raw = Environment.GetEnvironmentVariable(VulkanDeviceEnvVar);
            if (string.IsNullOrWhiteSpace(raw))
                return;

            if (!int.TryParse(raw, out int deviceIndex) || deviceIndex < 0)
            {
                throw new InvalidOperationException(
                    $"Invalid {VulkanDeviceEnvVar} value '{raw}'. Expected a non-negative Vulkan device index.");
            }

            SetVulkanDeviceIndex(deviceIndex);
        }

        public static void Addmm(GgmlTensorView2D result, GgmlTensorView2D src, GgmlTensorView2D m1, GgmlTensorView2D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmF32(result, src, m1, m2, beta, alpha), "addmm");
        }

        public static void AddmmQuant(GgmlTensorView2D result, GgmlTensorView2D m1, IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_AddmmQuantF32(result, m1, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "addmm_quant");
        }

        public static void FusedRmsNormMatMulQuant(
            GgmlTensorView2D result, GgmlTensorView2D input,
            IntPtr normWeightData, int normWeightCount, float eps,
            IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_FusedRmsNormMatMulQuantF32(
                result, input, normWeightData, normWeightCount, eps,
                m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "fused_rms_norm_matmul_quant");
        }

        public static void FusedMatMulQuantAdd(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            CheckResult(TSGgml_FusedMatMulQuantAddF32(
                residual, input, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes,
                tpDegree, tpPlanOut), "fused_matmul_quant_add");
        }

        public static void ReleaseFusedMatmulAddTpGraphs()
        {
            try { TSGgml_ReleaseFusedMatmulAddTpGraphs(); }
            catch (EntryPointNotFoundException) { }
        }

        public static void FusedFFNSwiGLUQuant(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            CheckResult(TSGgml_FusedFFNSwiGLUQuantF32(
                residual, input, normWeightData, normWeightCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim, tpDegree, tpPlanOut), "fused_ffn_swiglu_quant");
        }

        public static void FusedFFNActProjectQuant(
            GgmlTensorView2D output,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim,
            int actType)
        {
            CheckResult(TSGgml_FusedFFNActProjectQuantF32(
                output, input, normWeightData, normWeightCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim, actType), "fused_ffn_act_project_quant");
        }

        public static void FusedRmsNormResidualAdd(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr normWeightData, int normWeightCount, float eps)
        {
            CheckResult(TSGgml_FusedRmsNormResidualAddF32(
                residual, input, normWeightData, normWeightCount, eps), "fused_rms_norm_residual_add");
        }

        public static void FusedPleBlockQuant(
            GgmlTensorView2D residual, GgmlTensorView2D perLayerInput,
            IntPtr inpGateData, int inpGateGgmlType, long inpGateNe0, long inpGateNe1, long inpGateRawBytes,
            IntPtr projData, int projGgmlType, long projNe0, long projNe1, long projRawBytes,
            IntPtr postNormData, int postNormCount, float eps)
        {
            CheckResult(TSGgml_FusedPleBlockQuantF32(
                residual, perLayerInput,
                inpGateData, inpGateGgmlType, inpGateNe0, inpGateNe1, inpGateRawBytes,
                projData, projGgmlType, projNe0, projNe1, projRawBytes,
                postNormData, postNormCount, eps), "fused_ple_block_quant");
        }

        public static void FusedOutProjNormRouter(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outBytes,
            IntPtr normData, int normCount, float eps,
            GgmlTensorView2D normedOut,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerBytes,
            GgmlTensorView2D routerOut)
        {
            CheckResult(TSGgml_FusedOutProjNormRouterQuantF32(residual, input,
                outProjData, outProjType, outNe0, outNe1, outBytes,
                normData, normCount, eps, normedOut,
                routerData, routerType, routerNe0, routerNe1, routerBytes,
                routerOut), "fused_outproj_norm_router");
        }

        public static void FusedOutProjFFN(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            IntPtr ffnNormData, int ffnNormCount, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim)
        {
            CheckResult(TSGgml_FusedOutProjFFNQuantF32(residual, input,
                outProjData, outProjType, outNe0, outNe1, outRawBytes,
                ffnNormData, ffnNormCount, eps,
                guData, guType, guNe0, guNe1, guRawBytes,
                dnData, dnType, dnNe0, dnNe1, dnRawBytes,
                halfDim), "fused_outproj_ffn");
        }

        public static void FusedVisionMLP(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr upB, int upBDim,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr downB, int downBDim)
        {
            CheckResult(TSGgml_FusedVisionMLPF32(hidden,
                lnW, lnB, lnDim, eps,
                upW, upNe0, upNe1, upBytes, upB, upBDim,
                downW, downNe0, downNe1, downBytes, downB, downBDim), "fused_vision_mlp");
        }

        /// <summary>
        /// Runs one exact Muse-Glimmer vision block as a bounded, on-device graph.
        /// False means the backend/geometry is unsupported or workspace allocation
        /// failed; the caller retains its portable block implementation as fallback.
        /// </summary>
        internal static bool MuseGlimmerVisionBlock(in GgmlMuseGlimmerVisionBlockArgs args)
            => TSGgml_MuseGlimmerVisionBlockQuantF32(in args) != 0;

        public static void FusedVisionAttention(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr qkvW, int qkvNe0, int qkvNe1, long qkvBytes,
            IntPtr qkvB, int qkvBDim,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr outB, int outBDim,
            IntPtr cosTable, IntPtr sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale)
        {
            CheckResult(TSGgml_FusedVisionAttentionF32(hidden,
                lnW, lnB, lnDim, eps,
                qkvW, qkvNe0, qkvNe1, qkvBytes, qkvB, qkvBDim,
                outW, outNe0, outNe1, outBytes, outB, outBDim,
                cosTable, sinTable, numPatches, numHeads, headDim, halfDim,
                attnScale), "fused_vision_attention");
        }

        /// <summary>Whole Qwen3.5/3.6-VL vision encoder (all blocks, one graph).
        /// Returns false on native failure so the caller can fall back to the
        /// per-block path.</summary>
        public static bool Qwen35VisionEncoder(
            GgmlTensorView2D hidden,
            int blockCount, float eps, float attnScale,
            int numPatches, int numHeads, int headDim, int halfDim,
            IntPtr cosTable, IntPtr sinTable,
            IntPtr[] ln1W, IntPtr[] ln1B,
            IntPtr[] qkvW, IntPtr[] qkvB,
            IntPtr[] outW, IntPtr[] outB,
            IntPtr[] ln2W, IntPtr[] ln2B,
            IntPtr[] upW, IntPtr[] upB,
            IntPtr[] downW, IntPtr[] downB,
            int lnDim,
            int qkvNe0, int qkvNe1, long qkvBytes, int qkvBDim,
            int outNe0, int outNe1, long outBytes, int outBDim,
            int upNe0, int upNe1, long upBytes, int upBDim,
            int downNe0, int downNe1, long downBytes, int downBDim)
        {
            int rc = TSGgml_Qwen35VisionEncoderF32(hidden,
                blockCount, eps, attnScale,
                numPatches, numHeads, headDim, halfDim,
                cosTable, sinTable,
                ln1W, ln1B, qkvW, qkvB, outW, outB, ln2W, ln2B, upW, upB, downW, downB,
                lnDim,
                qkvNe0, qkvNe1, qkvBytes, qkvBDim,
                outNe0, outNe1, outBytes, outBDim,
                upNe0, upNe1, upBytes, upBDim,
                downNe0, downNe1, downBytes, downBDim);
            return rc != 0;
        }

        public static bool GlmVisionEncoder(
            GgmlTensorView2D hidden,
            int blockCount, float eps, float attnScale, float swigluLimit,
            int numPatches, int numHeads, int headDim, int halfDim,
            IntPtr cosTable, IntPtr sinTable,
            IntPtr[] ln1W,
            IntPtr[] qkvW, IntPtr[] qkvB,
            IntPtr[] qnW, IntPtr[] knW,
            IntPtr[] outW, IntPtr[] outB,
            IntPtr[] ln2W,
            IntPtr[] gateW, IntPtr[] gateB,
            IntPtr[] upW, IntPtr[] upB,
            IntPtr[] downW, IntPtr[] downB,
            int lnDim,
            int qkvNe0, int qkvNe1, long qkvBytes,
            int outNe0, int outNe1, long outBytes,
            int ffnNe0, int ffnNe1, long ffnUpBytes, long ffnDownBytes)
        {
            int rc = TSGgml_GlmVisionEncoderF32(hidden,
                blockCount, eps, attnScale, swigluLimit,
                numPatches, numHeads, headDim, halfDim,
                cosTable, sinTable,
                ln1W, qkvW, qkvB, qnW, knW, outW, outB, ln2W,
                gateW, gateB, upW, upB, downW, downB,
                lnDim,
                qkvNe0, qkvNe1, qkvBytes,
                outNe0, outNe1, outBytes,
                ffnNe0, ffnNe1, ffnUpBytes, ffnDownBytes);
            return rc != 0;
        }

        public static void FusedGemma4VisionBlock(
            GgmlTensorView2D hidden, float eps,
            IntPtr ln1W,
            IntPtr qW, int qNe0, int qNe1, long qBytes,
            IntPtr kW, int kNe0, int kNe1, long kBytes,
            IntPtr vW, int vNe0, int vNe1, long vBytes,
            IntPtr qNormW, IntPtr kNormW,
            IntPtr attnPostNormW,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr cosx, IntPtr sinx, IntPtr cosy, IntPtr siny,
            IntPtr ln2W,
            IntPtr gateW, int gateNe0, int gateNe1, long gateBytes,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr ffnPostNormW,
            IntPtr clamps,
            int numPatches, int numHeads, int headDim)
        {
            CheckResult(TSGgml_FusedGemma4VisionBlockF32(hidden, eps, ln1W,
                qW, qNe0, qNe1, qBytes,
                kW, kNe0, kNe1, kBytes,
                vW, vNe0, vNe1, vBytes,
                qNormW, kNormW, attnPostNormW,
                outW, outNe0, outNe1, outBytes,
                cosx, sinx, cosy, siny, ln2W,
                gateW, gateNe0, gateNe1, gateBytes,
                upW, upNe0, upNe1, upBytes,
                downW, downNe0, downNe1, downBytes,
                ffnPostNormW, clamps, numPatches, numHeads, headDim),
                "fused_gemma4_vision_block");
        }

        public static void GetRowsQuant(GgmlTensorView2D result, IntPtr srcData, int srcGgmlType, long srcNe0, long srcNe1, long srcRawBytes, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_GetRowsQuantF32(result, srcData, srcGgmlType, srcNe0, srcNe1, srcRawBytes, indices), "get_rows_quant");
        }

        public static void MoEExpertsForward(GgmlTensorView2D result, GgmlTensorView2D input,
            int numExperts, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            CheckResult(TSGgml_MoEExpertsForwardF32(result, input, numExperts,
                upDataPtrs, downDataPtrs,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights), "moe_experts_forward");
        }

        public static void MoEExpertsSwiGLUForward(GgmlTensorView2D result, GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights)
        {
            CheckResult(TSGgml_MoEExpertsSwiGLUForwardF32(result, input, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights), "moe_experts_swiglu_forward");
        }

        public static void MoEExpertsSwiGLUResidual(GgmlTensorView2D residual, GgmlTensorView2D input,
            int numExperts,
            IntPtr[] gateDataPtrs, IntPtr[] upDataPtrs, IntPtr[] downDataPtrs,
            int gateGgmlType, long gateNe0, long gateNe1, long gateRawBytesEach,
            int upGgmlType, long upNe0, long upNe1, long upRawBytesEach,
            int downGgmlType, long downNe0, long downNe1, long downRawBytesEach,
            float[] routeWeights,
            bool useShared,
            IntPtr sharedGateData, IntPtr sharedUpData, IntPtr sharedDownData,
            int sharedGateGgmlType, long sharedGateNe0, long sharedGateNe1, long sharedGateRawBytes,
            int sharedUpGgmlType, long sharedUpNe0, long sharedUpNe1, long sharedUpRawBytes,
            int sharedDownGgmlType, long sharedDownNe0, long sharedDownNe1, long sharedDownRawBytes,
            float sharedScalar)
        {
            CheckResult(TSGgml_MoEExpertsSwiGLUResidualF32(residual, input, numExperts,
                gateDataPtrs, upDataPtrs, downDataPtrs,
                gateGgmlType, gateNe0, gateNe1, gateRawBytesEach,
                upGgmlType, upNe0, upNe1, upRawBytesEach,
                downGgmlType, downNe0, downNe1, downRawBytesEach,
                routeWeights,
                useShared ? 1 : 0,
                sharedGateData, sharedUpData, sharedDownData,
                sharedGateGgmlType, sharedGateNe0, sharedGateNe1, sharedGateRawBytes,
                sharedUpGgmlType, sharedUpNe0, sharedUpNe1, sharedUpRawBytes,
                sharedDownGgmlType, sharedDownNe0, sharedDownNe1, sharedDownRawBytes,
                sharedScalar), "moe_experts_swiglu_residual");
        }

        public static void AddmmQuantBatch(GgmlTensorView2D result, GgmlTensorView2D m1, IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2RawBytes,
            int batchCount, long[] weightOffsets, long[] weightNe1Arr)
        {
            CheckResult(TSGgml_AddmmQuantBatchF32(result, m1, m2Data, m2GgmlType, m2Ne0, m2RawBytes, batchCount, weightOffsets, weightNe1Arr), "addmm_quant_batch");
        }

        public static void AddmmBatch(GgmlTensorView3D result, GgmlTensorView3D src, GgmlTensorView3D m1, GgmlTensorView3D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmBatchF32(result, src, m1, m2, beta, alpha), "addmmbatch");
        }

        public static void ReduceLastDim(GgmlReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_ReduceLastDimF32((int)op, result, src), op.ToString());
        }

        public static void IndexReduction(GgmlIndexReductionOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_IndexReductionF32((int)op, result, src), op.ToString());
        }

        public static void Softmax(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_SoftmaxF32(result, src), "softmax");
        }

        /// <summary>
        /// In-place softmax with causal+SWA mask and optional attention sinks.
        /// scores layout is [numHeads, seqLen, kvLen]. sinksData may be IntPtr.Zero
        /// when no sinks are needed; slidingWindow &lt;= 0 disables the SWA mask.
        ///
        /// Replaces three separate ops in the GptOss attention path: AddCausalMask
        /// (GPU) + ApplySWAMask (CPU) + ApplySoftmaxWithSinks (CPU). The CPU
        /// softmax-with-sinks loop dominated GptOss prefill (~76% of total time
        /// on pp2048) because it walked ~6 billion elements through MathF.Exp on
        /// a single thread; folding it into one Metal kernel collapses that.
        /// </summary>
        public static void AttentionSoftmaxWithSinks(
            GgmlTensorView3D scores,
            IntPtr sinksData,
            int numHeads,
            int seqLen,
            int kvLen,
            int maskStartPos,
            int slidingWindow,
            float scale)
        {
            CheckResult(TSGgml_AttentionSoftmaxWithSinksF32(
                scores, sinksData, numHeads, seqLen, kvLen,
                maskStartPos, slidingWindow, scale),
                "attention_softmax_with_sinks");
        }

        public static void MoEFFNPrefillSwiGLUQuant(
            IntPtr hiddenIn,
            IntPtr hiddenOut,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit,
            bool runOnCpu = false)
        {
            CheckResult(TSGgml_MoEFFNPrefillSwiGLUQuantF32(
                hiddenIn, hiddenOut, seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit, runOnCpu ? 1 : 0),
                "moe_ffn_prefill_swiglu_quant");
        }

        public static void Gemma4MoEGEGLUResidual(
            IntPtr hiddenIn,
            IntPtr residualInOut,
            IntPtr postNormW,
            float postNormEps,
            int seqLen,
            int hiddenDim,
            int nFf,
            int numExperts,
            int nUsed,
            IntPtr selectedExperts,
            IntPtr routingWeights,
            IntPtr gateData, int gateType, long gateNe0, long gateNe1, long gateTotalBytes,
            IntPtr upData,   int upType,   long upNe0,   long upNe1,   long upTotalBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downTotalBytes,
            IntPtr gateBias,
            IntPtr upBias,
            IntPtr downBias,
            int activationType,
            float oaiAlpha,
            float oaiLimit,
            bool runOnCpu = false)
        {
            CheckResult(TSGgml_Gemma4MoEGEGLUResidualF32(
                hiddenIn, residualInOut, postNormW, postNormEps,
                seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit, runOnCpu ? 1 : 0),
                "gemma4_moe_geglu_residual");
        }

        public static void ScaledDotProductAttention(GgmlTensorView4D result, GgmlTensorView4D query, GgmlTensorView4D key, GgmlTensorView4D value, GgmlTensorView4D mask, bool hasMask, float scale)
        {
            CheckResult(TSGgml_ScaledDotProductAttentionF32(result, query, key, value, mask, hasMask ? 1 : 0, scale), "scaled_dot_product_attention");
        }

        public static void SoftmaxGrad(GgmlTensorView4D result, GgmlTensorView4D adj, GgmlTensorView4D val, bool addGrad)
        {
            CheckResult(TSGgml_SoftmaxGradF32(result, adj, val, addGrad ? 1 : 0), "softmaxgrad");
        }

        public static void Adam(
            GgmlContiguousTensor weight,
            GgmlContiguousTensor gradient,
            GgmlContiguousTensor v,
            GgmlContiguousTensor m,
            float gradNormFactor,
            float stepSize,
            float clipValue,
            float regc,
            float decayRateV,
            float decayRateM,
            int iter,
            float eps)
        {
            CheckResult(TSGgml_AdamF32(weight, gradient, v, m, gradNormFactor, stepSize, clipValue, regc, decayRateV, decayRateM, iter, eps), "adam");
        }

        public static void Copy(GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_CopyF32(result, src), "copy");
        }

        public static void Unary(GgmlUnaryOp op, GgmlTensorView4D result, GgmlTensorView4D src)
        {
            CheckResult(TSGgml_UnaryF32((int)op, result, src), op.ToString());
        }

        public static void BinaryTensor(GgmlBinaryTensorOp op, GgmlTensorView4D result, GgmlTensorView4D lhs, GgmlTensorView4D rhs)
        {
            CheckResult(TSGgml_BinaryTensorF32((int)op, result, lhs, rhs), op.ToString());
        }

        public static void FusedActMul(GgmlFusedActMulOp op, GgmlTensorView4D result, GgmlTensorView4D a, GgmlTensorView4D b)
        {
            CheckResult(TSGgml_FusedActMulF32((int)op, result, a, b), op.ToString());
        }

        public static void FusedActMulSplit(GgmlFusedActMulOp op, GgmlTensorView2D result, GgmlTensorView2D gateUp, int halfDim)
        {
            CheckResult(TSGgml_FusedActMulSplitF32((int)op, result, gateUp, halfDim), op.ToString() + "Split");
        }

        public static void BinaryScalar(GgmlBinaryScalarOp op, GgmlTensorView4D result, GgmlTensorView4D src, float scalar)
        {
            CheckResult(TSGgml_BinaryScalarF32((int)op, result, src, scalar), op.ToString());
        }

        public static void Norm(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D gamma, GgmlTensorView4D beta, bool hasBeta, float eps)
        {
            CheckResult(TSGgml_NormF32((int)op, result, src, gamma, beta, hasBeta ? 1 : 0, eps), op.ToString());
        }

        public static void IndexSelect(GgmlTensorView2D result, GgmlTensorView2D src, GgmlContiguousTensor indices, bool addToResult)
        {
            CheckResult(TSGgml_IndexSelectF32(result, src, indices, addToResult ? 1 : 0), "indexselect");
        }

        public static void IndexSelectGrad(GgmlTensorView2D grad, GgmlTensorView2D adj, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_IndexSelectGradF32(grad, adj, indices), "indexselectgrad");
        }

        public static void RoPE(GgmlTensorView4D result, GgmlTensorView4D src, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, src, seqLen, rowOffset, 0, 0), "rope");
        }

        public static void RoPEGrad(GgmlTensorView4D result, GgmlTensorView4D adj, int seqLen, int rowOffset)
        {
            CheckResult(TSGgml_RoPEF32(result, adj, seqLen, rowOffset, 1, 1), "ropegrad");
        }

        public static void RoPEEx(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions)
        {
            CheckResult(
                TSGgml_RoPEExF32(
                    result,
                    src,
                    positions,
                    ropeDim,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow,
                    addToResult ? 1 : 0,
                    invertPositions ? 1 : 0),
                "rope_ex");
        }

        public static void RoPEMRoPE(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int sect0, int sect1, int sect2, int sect3,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow)
        {
            CheckResult(
                TSGgml_RoPEMRoPEF32(
                    result, src, positions,
                    ropeDim, mode,
                    sect0, sect1, sect2, sect3,
                    originalContextLength,
                    freqBase, freqScale,
                    extFactor, attnFactor,
                    betaFast, betaSlow),
                "rope_mrope");
        }

        public static void RoPEExWithFreqFactors(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlContiguousTensor positions,
            int ropeDim,
            int mode,
            int originalContextLength,
            float freqBase,
            float freqScale,
            float extFactor,
            float attnFactor,
            float betaFast,
            float betaSlow,
            bool addToResult,
            bool invertPositions,
            IntPtr freqFactors,
            int freqFactorsLen)
        {
            CheckResult(
                TSGgml_RoPEExFreqFactorsF32(
                    result,
                    src,
                    positions,
                    ropeDim,
                    mode,
                    originalContextLength,
                    freqBase,
                    freqScale,
                    extFactor,
                    attnFactor,
                    betaFast,
                    betaSlow,
                    addToResult ? 1 : 0,
                    invertPositions ? 1 : 0,
                    freqFactors,
                    freqFactorsLen),
                "rope_ex_ff");
        }

        /// <summary>
        /// Single-token flash attention decode kernel. Appends the new K/V to the persistent
        /// KV cache at <paramref name="position"/>, then runs <c>ggml_flash_attn_ext</c> on the
        /// device against the populated portion of the cache. Q, K, V, and the output buffer
        /// must point to F32 contiguous memory in (heads, head_dim) row-major layout.
        /// </summary>
        public static void FusedPrefillAttention(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow,
            float scale, int inputFormat = 0)
        {
            CheckResult(TSGgml_FusedPrefillAttentionF32(
                qData, kData, vData, outData,
                numHeads, numKvHeads, headDim,
                seqLen, kvLen,
                maskStartPos, slidingWindow, scale, inputFormat), "fused_prefill_attention");
        }

        public static void FusedPrefillAttentionF16KV(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvCacheLen,
            int maskStartPos, int slidingWindow, float scale)
        {
            CheckResult(TSGgml_FusedPrefillAttentionF16KV(
                qData, kData, vData, outData,
                numHeads, numKvHeads, headDim,
                seqLen, kvLen, kvCacheLen,
                maskStartPos, slidingWindow, scale), "fused_prefill_attention_f16kv");
        }

        public static void FlashAttnDecode(
            IntPtr qData, IntPtr kData, IntPtr vData,
            IntPtr kCacheData, IntPtr vCacheData,
            IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position,
            float scale, int kvCacheType = 0)
        {
            CheckResult(TSGgml_FlashAttnDecodeF32(
                qData, kData, vData,
                kCacheData, vCacheData,
                outData,
                numHeads, numKvHeads, headDim,
                maxSeqLen, position, scale, kvCacheType), "flash_attn_decode");
        }

        /// <summary>
        /// Native batched paged attention via <c>ggml_flash_attn_ext</c>. For
        /// each sequence in the batch, the C++ side gathers K and V from the
        /// paged buffer (walking the per-sequence block table), then runs the
        /// backend's fused flash-attention kernel. One Metal/CUDA kernel per
        /// sequence per layer, with the gather inside the native side so we
        /// don't pay the managed↔native border crossing N×L times.
        /// </summary>
        /// <param name="qData">[numTokens, numHeads * headDim] row-major float[].</param>
        /// <param name="pagedKData">[numBlocks * blockSize, numKvHeads, headDim] row-major.</param>
        /// <param name="pagedVData">Same layout as pagedKData.</param>
        /// <param name="outData">[numTokens, numHeads * headDim] (writes back).</param>
        /// <param name="queryStartLoc">[numSeqs + 1] cumulative query offsets.</param>
        /// <param name="seqLens">[numSeqs] total context length per sequence.</param>
        /// <param name="positions">[numTokens] absolute position per query token (drives the causal mask).</param>
        /// <param name="blockTableFlat">Concatenated per-sequence block tables.</param>
        /// <param name="blockTableOffsets">[numSeqs] offset of each seq's table inside blockTableFlat.</param>
        /// <summary>Allocate the device-resident paged K/V pool. Returns
        /// IntPtr.Zero when the backend could not allocate it (out of VRAM),
        /// which callers treat as "stay on the host pool".</summary>
        public static IntPtr PagedKvPoolCreate(
            int numLayers, int numBlocks, int blockSize, int numKvHeads, int headDim)
            => TSGgml_PagedKvPoolCreate(numLayers, numBlocks, blockSize, numKvHeads, headDim);

        public static void PagedKvPoolFree(IntPtr handle)
        {
            if (handle != IntPtr.Zero) TSGgml_PagedKvPoolFree(handle);
        }

        public static long PagedKvPoolBytes(IntPtr handle)
            => handle == IntPtr.Zero ? 0 : TSGgml_PagedKvPoolBytes(handle);

        /// <summary>Grow the pool to hold at least newNumBlocks blocks, copying
        /// what is already written on device. False means the pool kept its old
        /// size (typically out of VRAM) and the caller must not write past it.</summary>
        public static bool PagedKvPoolGrow(IntPtr handle, int newNumBlocks)
            => handle != IntPtr.Zero && TSGgml_PagedKvPoolGrow(handle, newNumBlocks) != 0;

        /// <summary>Write this step's K/V into the pool at the mapped slots.</summary>
        public static unsafe void PagedKvPoolScatter(
            IntPtr handle, int layer, float[] kData, float[] vData, int[] slotMapping, int numTokens)
        {
            fixed (float* k = kData)
            fixed (float* v = vData)
            fixed (int* sm = slotMapping)
            {
                CheckResult(TSGgml_PagedKvPoolScatter(
                    handle, layer, (IntPtr)k, (IntPtr)v, (IntPtr)sm, numTokens),
                    "paged_kv_pool_scatter");
            }
        }

        /// <summary>Batch flash attention against the device-resident pool. The
        /// sequence history is gathered on device via ggml_get_rows, so no K/V
        /// is uploaded per layer.</summary>
        public static unsafe void PagedKvPoolAttention(
            IntPtr handle, int layer, float[] qData, float[] outData,
            int[] queryStartLoc, int[] seqLens, int[] positions,
            int[] blockTableFlat, int[] blockTableOffsets,
            int numSeqs, int numTokens, int numHeads, float scale, int slidingWindow = 0)
        {
            fixed (float* q = qData)
            fixed (float* o = outData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            {
                CheckResult(TSGgml_PagedKvPoolAttention(
                    handle, layer, (IntPtr)q, (IntPtr)o,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos, (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, slidingWindow, scale),
                    "paged_kv_pool_attention");
            }
        }

        public static unsafe void PagedAttentionForward(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            fixed (float* q = qData)
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (float* o = outData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            {
                CheckResult(TSGgml_PagedAttentionForward(
                    (IntPtr)q, (IntPtr)kp, (IntPtr)vp, (IntPtr)o,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale), "paged_attention_forward");
            }
        }

        /// <summary>Native paged-attention forward with per-head attention
        /// sinks (gpt-oss style). Sinks is a [numHeads] F32 array; null
        /// degenerates to the regular paged attention. Goes through
        /// ggml_flash_attn_ext_add_sinks under the hood so the Metal/CUDA
        /// flash-attn kernel includes the sink as a virtual softmax position.</summary>
        public static unsafe void PagedAttentionForwardWithSinks(
            float[] qData,
            float[] pagedKData,
            float[] pagedVData,
            float[] outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            fixed (float* q = qData)
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (float* o = outData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            fixed (float* sink = sinksData)
            {
                CheckResult(TSGgml_PagedAttentionForwardWithSinks(
                    (IntPtr)q, (IntPtr)kp, (IntPtr)vp, (IntPtr)o,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale,
                    sinksData != null ? (IntPtr)sink : IntPtr.Zero),
                    "paged_attention_forward_with_sinks");
            }
        }

        /// <summary>
        /// GPU-resident paged-attention forward. <paramref name="qData"/> and
        /// <paramref name="outData"/> point to backend-allocated buffers
        /// (typically <c>tensor.Storage.PtrAtElement(...)</c> on the Metal /
        /// CUDA backend). The kernel zero-copy binds Q's tensor and writes
        /// the attention output directly into the caller's output tensor —
        /// no host-side memcpy round-trip, no per-layer
        /// <c>ggml_backend_synchronize</c>. K/V paged storage is still passed
        /// as host arrays.
        /// </summary>
        public static unsafe void PagedAttentionForwardDevice(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow = 0)
        {
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            {
                CheckResult(TSGgml_PagedAttentionForwardDevice(
                    qData, (IntPtr)kp, (IntPtr)vp, outData,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale),
                    "paged_attention_forward_device");
            }
        }

        /// <summary>GPU-resident paged-attention forward with per-head
        /// attention sinks. Pass <c>null</c> for <paramref name="sinksData"/>
        /// to match <see cref="PagedAttentionForwardDevice"/>.</summary>
        public static unsafe void PagedAttentionForwardDeviceWithSinks(
            IntPtr qData,
            float[] pagedKData,
            float[] pagedVData,
            IntPtr outData,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[] blockTableFlat,
            int[] blockTableOffsets,
            int numSeqs,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            float scale,
            int slidingWindow,
            float[] sinksData)
        {
            fixed (float* kp = pagedKData)
            fixed (float* vp = pagedVData)
            fixed (int* qsl = queryStartLoc)
            fixed (int* sl = seqLens)
            fixed (int* pos = positions)
            fixed (int* btf = blockTableFlat)
            fixed (int* bto = blockTableOffsets)
            fixed (float* sink = sinksData)
            {
                CheckResult(TSGgml_PagedAttentionForwardDeviceWithSinks(
                    qData, (IntPtr)kp, (IntPtr)vp, outData,
                    (IntPtr)qsl, (IntPtr)sl, (IntPtr)pos,
                    (IntPtr)btf, (IntPtr)bto,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    blockSize, slidingWindow, scale,
                    sinksData != null ? (IntPtr)sink : IntPtr.Zero),
                    "paged_attention_forward_device_with_sinks");
            }
        }

        public static void Qwen35AttentionLayerDecode(
            IntPtr residualData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeNDims, int ropeMode, int kvCacheType = 0)
        {
            CheckResult(TSGgml_Qwen35AttentionLayerDecode(
                residualData, hiddenSize,
                attnNormData,
                qkvData, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormData, kNormData, headDim,
                oData, oType, oNe0, oNe1, oBytes,
                kCacheData, vCacheData,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale, ropeNDims, ropeMode, kvCacheType), "qwen35_attention_layer_decode");
        }

        public static void Gemma4ModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            int[] kvSourceArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int position,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            int kvCacheType = 0,
            IntPtr[] kArr = null, int[] kTypeArr = null, long[] kNe0Arr = null, long[] kNe1Arr = null, long[] kBytesArr = null,
            IntPtr[] vArr = null, int[] vTypeArr = null, long[] vNe0Arr = null, long[] vNe1Arr = null, long[] vBytesArr = null,
            IntPtr logitsData = default, int vocabSize = 0,
            IntPtr lmHeadData = default, int lmHeadType = 0, long lmHeadNe0 = 0, long lmHeadNe1 = 0, long lmHeadBytes = 0,
            IntPtr finalNormData = default, float logitSoftcap = 0f,
            IntPtr pleTokenEmbdData = default, int pleTokenEmbdType = 0,
            long pleTokenEmbdNe0 = 0, long pleTokenEmbdNe1 = 0, long pleTokenEmbdBytes = 0,
            int pleTokenId = -1,
            IntPtr pleModelProjData = default, int pleModelProjType = 0,
            long pleModelProjNe0 = 0, long pleModelProjNe1 = 0, long pleModelProjBytes = 0,
            IntPtr pleModelProjNormData = default,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            CheckResult(TSGgml_Gemma4ModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                kvSourceArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, position,
                eps, slidingWindow,
                ropeFreqFactors, ropeFreqFactorsLen,
                ropeNDimsArr,
                pleData, pleDim,
                pleGateArr, pleGateTypeArr, pleGateNe0Arr, pleGateNe1Arr, pleGateBytesArr,
                pleProjArr, pleProjTypeArr, pleProjNe0Arr, pleProjNe1Arr, pleProjBytesArr,
                plePostNormArr, kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                logitsData, vocabSize,
                lmHeadData, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNormData, logitSoftcap,
                pleTokenEmbdData, pleTokenEmbdType,
                pleTokenEmbdNe0, pleTokenEmbdNe1, pleTokenEmbdBytes,
                pleTokenId,
                pleModelProjData, pleModelProjType,
                pleModelProjNe0, pleModelProjNe1, pleModelProjBytes,
                pleModelProjNormData,
                tpDegree, tpPlanOut), "gemma4_model_decode");
        }

        /// <summary>True token-batched dense decode: N concurrent sequences, one
        /// token each, in a single fused graph. Returns false (without throwing)
        /// when the native kernel declines (e.g. a sequence exceeds the cache
        /// window) so the caller can fall back to round-robin.</summary>
        public static bool Gemma4ModelDecodeBatched(
            IntPtr hiddenData, int hiddenSize, int numLayers, int nSeqs,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int[] positions,
            float eps, int slidingWindow,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen,
            int[] ropeNDimsArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            IntPtr logitsData, int vocabSize,
            IntPtr lmHeadData, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNormData, float logitSoftcap)
        {
            int rc = TSGgml_Gemma4ModelDecodeBatched(
                hiddenData, hiddenSize, numLayers, nSeqs,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, positions,
                eps, slidingWindow,
                ropeFreqFactors, ropeFreqFactorsLen,
                ropeNDimsArr,
                kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                logitsData, vocabSize,
                lmHeadData, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes,
                finalNormData, logitSoftcap);
            return rc != 0;
        }

        /// <summary>Fused multi-token verify (the speculative trunk's verify batch).
        /// Returns false (without throwing) when the native kernel declines (e.g.
        /// total length exceeds the SWA window) so the caller can fall back to the
        /// per-op path.</summary>
        public static bool Gemma4ModelVerify(
            IntPtr hiddenData, int hiddenSize, int numLayers, int numTokens,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] postAttnNormArr,
            IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr, IntPtr[] postFfnNormArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int[] headDimArr, int[] kvHeadsArr, int[] cacheSizeArr, int[] isLocalArr,
            float[] ropeBaseArr, float[] layerScalarArr,
            int[] qkvTypeArr, long[] qkvNe0Arr, long[] qkvNe1Arr, long[] qkvBytesArr,
            int[] oTypeArr, long[] oNe0Arr, long[] oNe1Arr, long[] oBytesArr,
            int[] guTypeArr, long[] guNe0Arr, long[] guNe1Arr, long[] guBytesArr,
            int[] downTypeArr, long[] downNe0Arr, long[] downNe1Arr, long[] downBytesArr,
            int numHeads, int startPos, float eps,
            IntPtr ropeFreqFactors, int ropeFreqFactorsLen, int[] ropeNDimsArr,
            int kvCacheType,
            IntPtr[] kArr, int[] kTypeArr, long[] kNe0Arr, long[] kNe1Arr, long[] kBytesArr,
            IntPtr[] vArr, int[] vTypeArr, long[] vNe0Arr, long[] vNe1Arr, long[] vBytesArr,
            int[] kvSourceArr,
            IntPtr pleData, int pleDim,
            IntPtr[] pleGateArr, int[] pleGateTypeArr, long[] pleGateNe0Arr, long[] pleGateNe1Arr, long[] pleGateBytesArr,
            IntPtr[] pleProjArr, int[] pleProjTypeArr, long[] pleProjNe0Arr, long[] pleProjNe1Arr, long[] pleProjBytesArr,
            IntPtr[] plePostNormArr,
            byte[] isExceptArr = null,
            IntPtr pleTokenEmbdData = default, int pleTokenEmbdType = 0,
            long pleTokenEmbdNe0 = 0, long pleTokenEmbdNe1 = 0, long pleTokenEmbdBytes = 0,
            int[] pleTokenIds = null,
            IntPtr pleProjWData = default, int pleProjWType = 0,
            long pleProjWNe0 = 0, long pleProjWNe1 = 0, long pleProjWBytes = 0,
            IntPtr pleProjNormData = default,
            int tpDegree = 1, IntPtr[] tpPlanOut = null)
        {
            if (tpPlanOut != null) tpPlanOut[0] = IntPtr.Zero;
            int r = TSGgml_Gemma4ModelVerify(
                hiddenData, hiddenSize, numLayers, numTokens,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, postAttnNormArr,
                ffnNormArr, guArr, downArr, postFfnNormArr,
                kCacheArr, vCacheArr,
                headDimArr, kvHeadsArr, cacheSizeArr, isLocalArr,
                ropeBaseArr, layerScalarArr,
                qkvTypeArr, qkvNe0Arr, qkvNe1Arr, qkvBytesArr,
                oTypeArr, oNe0Arr, oNe1Arr, oBytesArr,
                guTypeArr, guNe0Arr, guNe1Arr, guBytesArr,
                downTypeArr, downNe0Arr, downNe1Arr, downBytesArr,
                numHeads, startPos, eps,
                ropeFreqFactors, ropeFreqFactorsLen, ropeNDimsArr,
                kvCacheType,
                kArr, kTypeArr, kNe0Arr, kNe1Arr, kBytesArr,
                vArr, vTypeArr, vNe0Arr, vNe1Arr, vBytesArr,
                kvSourceArr,
                pleData, pleDim,
                pleGateArr, pleGateTypeArr, pleGateNe0Arr, pleGateNe1Arr, pleGateBytesArr,
                pleProjArr, pleProjTypeArr, pleProjNe0Arr, pleProjNe1Arr, pleProjBytesArr,
                plePostNormArr,
                isExceptArr,
                pleTokenEmbdData, pleTokenEmbdType,
                pleTokenEmbdNe0, pleTokenEmbdNe1, pleTokenEmbdBytes,
                pleTokenIds,
                pleProjWData, pleProjWType,
                pleProjWNe0, pleProjWNe1, pleProjWBytes,
                pleProjNormData,
                tpDegree, tpPlanOut);
            return r != 0;
        }

        public static void GatedDeltaNetChunked(
            GgmlTensorView3D q,
            GgmlTensorView3D k,
            GgmlTensorView3D v,
            GgmlTensorView3D z,
            GgmlTensorView2D alpha,
            GgmlTensorView2D beta,
            GgmlTensorView3D state,
            GgmlTensorView3D gatedOut,
            IntPtr dtBiasData,
            IntPtr aLogData,
            IntPtr ssmNormWData,
            int chunkSize,
            float eps,
            int gateMode = 0)
        {
            CheckResult(TSGgml_GatedDeltaNetChunkedF32(
                q, k, v, z, alpha, beta, state, gatedOut,
                dtBiasData, aLogData, ssmNormWData,
                chunkSize, eps, gateMode), "gated_delta_net_chunked");
        }

        // Batched per-token Nemotron Mamba2 step. Runs all (seq, token) pairs
        // for an active decode/prefill batch in one native call, indexing each
        // seq's persistent conv FIFO + SSM state via the seqs[] descriptors.
        public static void NemotronMamba2BatchedStep(
            NemoMamba2BatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int dInProjTotal,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            IntPtr convWt,
            IntPtr convBias,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr dData,
            IntPtr ssmNormW,
            float eps,
            IntPtr outBatched)
        {
            CheckResult(TSGgml_NemotronMamba2BatchedStepF32(
                seqs?.Length ?? 0, seqs, numTokens,
                packedBatched, dInProjTotal,
                dInner, dState, nHead, headDim, nGroup, dConv,
                convWt, convBias, dtBias, aLog, dData, ssmNormW,
                eps, outBatched),
                "nemotron_mamba2_batched_step");
        }

        // Batched per-token Qwen3.5 GDN step. Runs all (seq, token) pairs for
        // an active decode/prefill batch in one native call, swapping in the
        // matching per-slot conv ring + ssm state via the seqs[] descriptors.
        // The descriptors' ConvWriteIdx field is updated in place — caller
        // copies it back to its per-slot bookkeeping after the call returns.
        public static void GatedDeltaNetBatchedStep(
            GdnBatchedSeqDesc[] seqs,
            int numTokens,
            IntPtr packedBatched,
            int packedDim,
            int qkvDim,
            int qkDim,
            int vDim,
            int zDim,
            int numKHeads,
            int numVHeads,
            int headKDim,
            int headVDim,
            int convKernel,
            int ssmDInner,
            IntPtr convWt,
            IntPtr dtBias,
            IntPtr aLog,
            IntPtr ssmNormW,
            float eps,
            IntPtr gatedOut)
        {
            CheckResult(TSGgml_GatedDeltaNetBatchedStepF32(
                seqs?.Length ?? 0, seqs, numTokens,
                packedBatched, packedDim, qkvDim, qkDim, vDim, zDim,
                numKHeads, numVHeads, headKDim, headVDim,
                convKernel, ssmDInner,
                convWt, dtBias, aLog, ssmNormW, eps, gatedOut),
                "gated_delta_net_batched_step");
        }

        public static void NemotronMamba2Prefill(
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            CheckResult(TSGgml_NemotronMamba2PrefillF32(
                projected, hiddenOut,
                convStateData, convStateElements,
                ssmStateData, ssmStateElements,
                convWeightData, convBiasData, dtBiasData, aData, dData, ssmNormData,
                dInner, dState, nHead, headDim, nGroup, dConv, eps), "nemotron_mamba2_prefill");
        }

        public static void NemotronMamba2Decode(
            ulong stateKey,
            GgmlTensorView2D projected,
            GgmlTensorView2D hiddenOut,
            IntPtr convStateData,
            int convStateElements,
            IntPtr ssmStateData,
            int ssmStateElements,
            bool initializeState,
            bool downloadState,
            IntPtr convWeightData,
            IntPtr convBiasData,
            IntPtr dtBiasData,
            IntPtr aData,
            IntPtr dData,
            IntPtr ssmNormData,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            float eps)
        {
            CheckResult(TSGgml_NemotronMamba2DecodeF32(
                stateKey, projected, hiddenOut,
                convStateData, convStateElements,
                ssmStateData, ssmStateElements,
                initializeState ? 1 : 0,
                downloadState ? 1 : 0,
                convWeightData, convBiasData, dtBiasData, aData, dData, ssmNormData,
                dInner, dState, nHead, headDim, nGroup, dConv, eps), "nemotron_mamba2_decode");
        }

        public static void NemotronMamba2DecodeClear(ulong modelKey)
        {
            TSGgml_NemotronMamba2DecodeClear(modelKey);
        }

        /// <summary>Allocate memory with 16 KB alignment (page-aligned for Metal host_ptr).</summary>
        public static IntPtr AlignedAlloc(long size)
        {
            IntPtr ptr = TSGgml_AlignedAlloc(new UIntPtr((ulong)size));
            if (ptr == IntPtr.Zero && size > 0)
                throw new OutOfMemoryException($"Failed to allocate {size} bytes of aligned memory.");
            return ptr;
        }

        /// <summary>Free memory allocated by AlignedAlloc.</summary>
        public static void AlignedFree(IntPtr ptr)
        {
            TSGgml_AlignedFree(ptr);
        }

        /// <summary>Free all cached Metal host_ptr buffer objects.</summary>
        public static void ClearHostBufferCache()
        {
            TSGgml_ClearHostBufferCache();
        }

        /// <summary>
        /// Tear down the process-global GGML backend before the C runtime
        /// finalisers run. On macOS the ggml-metal device singleton asserts
        /// that its resource set is empty when its static destructor fires;
        /// if the backend, host-buffer cache, and preloaded-buffer cache
        /// outlive the .NET host the assertion aborts the process on exit.
        /// Hook this onto AppDomain.ProcessExit / ApplicationStopped.
        /// </summary>
        public static void Shutdown()
        {
            TSGgml_Shutdown();
        }

        /// <summary>
        /// Free the reusable per-graph compute buffer + gallocr without tearing down the
        /// backend. Used to hand the DiT denoise scratch back before a memory-heavy VAE decode.
        /// </summary>
        public static void ReleaseReuseComputeBuffers()
        {
            TSGgml_ReleaseReuseComputeBuffers();
        }

        public static void InvalidateHostBuffer(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
            {
                TSGgml_InvalidateHostBuffer(ptr);
            }
        }

        /// <summary>Diagnostic: total bytes of device-local COPY buffers resident in the GGML
        /// host-buffer cache (excludes zero-copy weight wrappers). Used by tests to assert that
        /// per-block activation/KV device copies are reclaimed rather than leaked.</summary>
        public static long DeviceCopyCacheResidentBytes() => TSGgml_DeviceCopyCacheResidentBytes();

        /// <summary>Diagnostic: active backend device memory. On Metal <paramref name="totalBytes"/>
        /// is recommendedMaxWorkingSetSize and <paramref name="freeBytes"/> = total - currentAllocatedSize,
        /// so (total - free) is the bytes currently resident. Returns false if unavailable.</summary>
        public static bool TryGetBackendMemory(out long freeBytes, out long totalBytes)
            => TSGgml_GetBackendMemory(out freeBytes, out totalBytes) != 0;

        /// <summary>True if the active GGML backend device is an integrated GPU
        /// (unified-memory iGPU, e.g. Intel UHD / AMD APU via ggml-vulkan). Such
        /// devices are memory-bandwidth bound; callers use this to skip heavy
        /// startup warmup that would otherwise take minutes. Returns false when the
        /// backend is unavailable or the query is not supported.</summary>
        public static bool IsActiveDeviceIntegrated()
        {
            try { return TSGgml_IsActiveDeviceIntegrated() != 0; }
            catch (EntryPointNotFoundException) { return false; }
            catch (DllNotFoundException) { return false; }
        }

        public static void SyncHostBuffer(IntPtr ptr, long byteCount)
        {
            if (ptr == IntPtr.Zero || byteCount <= 0)
                return;

            CheckResult(TSGgml_SyncHostBuffer(ptr, byteCount), "sync_host_buffer");
        }

        /// <summary>
        /// Enable lazy synchronization on the Metal backend. When on, per-op kernels
        /// return immediately after committing their command buffer instead of
        /// blocking on `[cmd_buf waitUntilCompleted]`. Subsequent ops chain through
        /// the Metal command queue, and host-side reads (via
        /// TensorComputePrimitives.GetFloatPointer / GetHalfPointer, which call
        /// Storage.EnsureHostReadable) drain pending work on demand.
        ///
        /// This mirrors llama.cpp's Metal backend: ggml_metal_graph_compute commits
        /// its command buffer and returns; only an explicit ggml_backend_synchronize
        /// blocks. For TensorSharp's per-op driving model, lazy sync collapses the
        /// per-op `[cmd_buf waitUntilCompleted]` round-trip overhead (~30-100 µs each
        /// on M-series Macs) that dominates prefill on long prompts.
        /// </summary>
        public static void SetAsyncCompute(bool enabled)
        {
            TSGgml_SetAsyncCompute(enabled ? 1 : 0);
        }

        /// <summary>
        /// Worker threads for the host-side MoE matmul (<c>--cpu-moe-threads</c>);
        /// 0 restores the default. Passed explicitly rather than through
        /// <c>TS_CPU_MOE_THREADS</c> because .NET's
        /// <see cref="Environment.SetEnvironmentVariable(string,string)"/> writes
        /// only the managed environment on Linux, so the native
        /// <c>std::getenv</c> never observed the flag.
        /// </summary>
        public static void SetHostMoeThreads(int threads)
        {
            TSGgml_SetHostMoeThreads(threads);
        }

        /// <summary>True if async compute is currently enabled on the GGML backend.</summary>
        public static bool GetAsyncCompute()
        {
            return TSGgml_GetAsyncCompute() != 0;
        }

        /// <summary>
        /// Drain any GPU work that was deferred under async compute. Cheap when no
        /// work is pending (single atomic exchange on the C++ side); when work is
        /// pending it does one ggml_backend_synchronize on the Metal command queue.
        /// </summary>
        public static void HostReadBarrier()
        {
            TSGgml_HostReadBarrier();
        }

        /// <summary>
        /// Preload a quantized weight into a device-resident buffer keyed by
        /// <paramref name="cacheKey"/>. Returns true when the weight is (now)
        /// device-resident; false when the device cannot hold it in a single
        /// backend buffer (e.g. ggml-vulkan's per-buffer maxBufferSize cap) —
        /// the caller must keep the host copy and use its host fallback path.
        /// Throws on any other native failure.
        /// </summary>
        public static bool PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (cacheKey == IntPtr.Zero || hostData == IntPtr.Zero || rawBytes <= 0)
                throw new ArgumentException("PreloadQuantizedWeight requires valid cache key, host data, and size.");

            int result = TSGgml_PreloadQuantizedWeight(cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            CheckResult(result, "preload_quantized_weight");
            return result != 2;
        }

        /// <summary>
        /// Mark a host data pointer as eligible for the MoE expert offload LRU.
        /// After registration, the GGML native cache touches an LRU on lookup
        /// hits for this pointer and evicts from the LRU tail when residency
        /// exceeds the budget configured by <see cref="SetOffloadableBudget"/>.
        /// Registration is sticky; call <see cref="ClearOffloadableState"/> on
        /// model unload to reset.
        /// </summary>
        public static void RegisterOffloadable(IntPtr key)
        {
            if (key == IntPtr.Zero)
                return;
            TSGgml_RegisterOffloadable(key);
        }

        /// <summary>
        /// Configure the byte ceiling for the offloadable cache LRU. Zero
        /// disables eviction (registered entries still participate in the LRU
        /// but nothing is freed).
        /// </summary>
        public static void SetOffloadableBudget(long bytes)
        {
            TSGgml_SetOffloadableBudget(bytes > 0 ? bytes : 0);
        }

        /// <summary>
        /// Reset offloadable registrations, LRU state, and byte accounting.
        /// Does not touch the underlying CachedHostBuffer entries.
        /// </summary>
        public static void ClearOffloadableState()
        {
            TSGgml_ClearOffloadableState();
        }

        /// <summary>
        /// Byte ceiling for device-local copy residency (discrete-GPU weight
        /// caching). Once resident copies reach the budget, further cacheable
        /// binds stream per-graph instead of becoming resident — this is what
        /// keeps VRAM from oversubscribing (and WDDM from paging) when the
        /// model is larger than the GPU. Zero disables the cap.
        /// </summary>
        public static void SetDeviceCopyBudget(long bytes)
        {
            TSGgml_SetDeviceCopyBudget(bytes > 0 ? bytes : 0);
        }

        /// <summary>
        /// Free/total memory of the active backend device in bytes (VRAM on
        /// CUDA). Returns false when the backend has no meaningful device
        /// memory (e.g. CPU).
        /// </summary>
        public static bool TryGetDeviceMemoryInfo(out long freeBytes, out long totalBytes)
        {
            freeBytes = 0;
            totalBytes = 0;
            return TSGgml_DeviceMemoryInfo(out freeBytes, out totalBytes) != 0;
        }

        /// <summary>
        /// Page-lock a host region (cudaHostRegister) so per-step device uploads
        /// from it take the fast DMA path (~2x pageable throughput). CUDA only;
        /// returns false (no-op) elsewhere. Callers MUST unregister before the
        /// memory is unmapped or freed.
        /// </summary>
        public static bool TryRegisterPinnedHostBuffer(IntPtr ptr, long bytes)
        {
            return TSGgml_RegisterPinnedHostBuffer(ptr, bytes) != 0;
        }

        public static void UnregisterPinnedHostBuffer(IntPtr ptr)
        {
            TSGgml_UnregisterPinnedHostBuffer(ptr);
        }

        /// <summary>Bytes for one row along ne[0]; 0 if type/shape invalid.</summary>
        internal static long RowSizeBytesOrZero(int ggmlType, long ne0)
        {
            return (long)TSGgml_RowSize(ggmlType, ne0).ToUInt64();
        }

        internal static void DequantizeGgufTensorToFloat32(int ggmlType, byte[] src, int srcOffset, float[] dst, int dstOffset, long numElements)
        {
            if (numElements < 0 || numElements > int.MaxValue)
            {
                throw new ArgumentOutOfRangeException(nameof(numElements));
            }

            int n = (int)numElements;
            if (srcOffset < 0 || dstOffset < 0 || checked(dstOffset + n) > dst.Length || srcOffset > src.Length)
            {
                throw new ArgumentException("Invalid src/dst range for dequantization.");
            }

            GCHandle hSrc = GCHandle.Alloc(src, GCHandleType.Pinned);
            GCHandle hDst = GCHandle.Alloc(dst, GCHandleType.Pinned);
            try
            {
                IntPtr pSrc = IntPtr.Add(hSrc.AddrOfPinnedObject(), srcOffset);
                IntPtr pDst = IntPtr.Add(hDst.AddrOfPinnedObject(), dstOffset * sizeof(float));
                int r = TSGgml_DequantizeToF32(ggmlType, pSrc, numElements, pDst);
                if (r == -1)
                {
                    throw new ArgumentException("Dequantization failed (invalid arguments).");
                }

                if (r == -2)
                {
                    throw new NotSupportedException(
                        $"GGML tensor type {ggmlType} cannot be dequantized to float32.");
                }
            }
            finally
            {
                if (hSrc.IsAllocated)
                {
                    hSrc.Free();
                }

                if (hDst.IsAllocated)
                {
                    hDst.Free();
                }
            }
        }

        /// <summary>
        /// Merge a LoRA delta into a (possibly quantized) weight IN PLACE:
        /// W[r,:] += scale * up[r,:] · down (dequantize row -> add -> requantize to the
        /// same type, the stable-diffusion.cpp apply path). <paramref name="w"/> points at
        /// the ggml row-major weight [ne1 x ne0]; up is [ne1, rank], down is [rank, ne0].
        /// Returns 0 on success; negative = validation/type error (weight untouched).
        /// </summary>
        public static unsafe int ApplyLoraDelta(IntPtr w, int ggmlType, long ne0, long ne1,
            float[] up, float[] down, int rank, float scale, int nThreads = 0)
        {
            if (w == IntPtr.Zero || up == null || down == null) return -1;
            if (up.LongLength < ne1 * rank || down.LongLength < (long)rank * ne0) return -1;
            fixed (float* pu = up, pd = down)
            {
                return TSGgml_ApplyLoraDelta(w, ggmlType, ne0, ne1, (IntPtr)pu, (IntPtr)pd, rank, scale, nThreads);
            }
        }

        /// <summary>
        /// Quantize FP32 rows into a GGML quantized row layout via ggml_quantize_chunk.
        /// Returns bytes written, or 0 when the target type cannot be produced without
        /// an importance matrix (IQ1/IQ2 families).
        /// </summary>
        internal static long QuantizeFloat32RowsOrZero(int ggmlType, IntPtr src, IntPtr dst, long nrows, long nPerRow)
        {
            if (src == IntPtr.Zero || dst == IntPtr.Zero || nrows <= 0 || nPerRow <= 0)
            {
                throw new ArgumentException("Invalid src/dst pointers or shape for quantization.");
            }

            if (ggml_quantize_requires_imatrix(ggmlType))
            {
                return 0;
            }

            ggml_quantize_init(ggmlType);
            return (long)ggml_quantize_chunk(ggmlType, src, dst, 0, nrows, nPerRow, IntPtr.Zero).ToUInt64();
        }

        internal static void DequantizeGgufTensorToFloat32Native(int ggmlType, IntPtr src, IntPtr dst, long numElements)
        {
            if (src == IntPtr.Zero || dst == IntPtr.Zero || numElements < 0)
            {
                throw new ArgumentException("Invalid src/dst pointers or element count for dequantization.");
            }

            int r = TSGgml_DequantizeToF32(ggmlType, src, numElements, dst);
            if (r == -1)
            {
                throw new ArgumentException("Dequantization failed (invalid arguments).");
            }

            if (r == -2)
            {
                throw new NotSupportedException(
                    $"GGML tensor type {ggmlType} cannot be dequantized to float32.");
            }
        }

        private static void CheckResult(int result, string opName)
        {
            if (result != 0)
            {
                return;
            }

            throw new InvalidOperationException($"Native GGML {opName} failed. {GetLastErrorMessage("Unknown native GGML error.")}");
        }

        private static IntPtr ImportResolver(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
        {
            if (!string.Equals(libraryName, DllName, StringComparison.Ordinal))
            {
                return IntPtr.Zero;
            }

            EnsureWindowsNativeDependencySearchPaths();

            foreach (string candidate in GetCandidatePaths(assembly))
            {
                if (File.Exists(candidate) && NativeLibrary.TryLoad(candidate, out IntPtr handle))
                {
                    return handle;
                }
            }

            return IntPtr.Zero;
        }

        private static IEnumerable<string> GetCandidatePaths(Assembly assembly)
        {
            string baseDirectory = AppContext.BaseDirectory;
            string assemblyDirectory = Path.GetDirectoryName(assembly.Location) ?? baseDirectory;

            foreach (string fileName in GetCandidateFileNames())
            {
                yield return Path.Combine(baseDirectory, fileName);
                yield return Path.Combine(assemblyDirectory, fileName);
            }

            foreach (string root in EnumerateRepoRoots(baseDirectory))
            {
                foreach (string fileName in GetCandidateFileNames())
                {
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build", "Release", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build-windows", fileName);
                    yield return Path.Combine(root, "TensorSharp.GGML.Native", "build-windows", "Release", fileName);
                }
            }
        }

        private static IEnumerable<string> EnumerateRepoRoots(string startDirectory)
        {
            DirectoryInfo current = new DirectoryInfo(startDirectory);
            while (current != null)
            {
                if (IsRepoRoot(current.FullName))
                {
                    yield return current.FullName;
                }

                current = current.Parent;
            }
        }

        private static IEnumerable<string> GetCandidateFileNames()
        {
            yield return OperatingSystem.IsWindows() ? "GgmlOps.dll" :
                OperatingSystem.IsMacOS() ? "libGgmlOps.dylib" :
                "libGgmlOps.so";
        }

        private static bool IsCudaPlatformSupported()
        {
            return OperatingSystem.IsWindows() || OperatingSystem.IsLinux();
        }

        private static bool IsVulkanPlatformSupported()
        {
            // Metal is the GPU backend on macOS; ggml-vulkan is built for
            // Windows and Linux only (see TensorSharp.GGML.Native/CMakeLists.txt).
            return OperatingSystem.IsWindows() || OperatingSystem.IsLinux();
        }

        private static void EnsureWindowsNativeDependencySearchPaths()
        {
            if (!OperatingSystem.IsWindows())
                return;

            if (Interlocked.Exchange(ref s_windowsDependencySearchPathsInitialized, 1) != 0)
                return;

            string currentPath = Environment.GetEnvironmentVariable("PATH") ?? string.Empty;
            var existingEntries = new HashSet<string>(
                currentPath.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries),
                StringComparer.OrdinalIgnoreCase);

            var additions = EnumerateWindowsNativeDependencyDirectories()
                .Where(path => Directory.Exists(path) && !existingEntries.Contains(path))
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .ToArray();

            if (additions.Length == 0)
                return;

            Environment.SetEnvironmentVariable(
                "PATH",
                string.Join(Path.PathSeparator, additions.Concat(new[] { currentPath })));
        }

        private static IEnumerable<string> EnumerateWindowsNativeDependencyDirectories()
        {
            foreach (string variableName in new[] { "CUDA_PATH", "CUDA_HOME" })
            {
                string root = Environment.GetEnvironmentVariable(variableName);
                if (!string.IsNullOrWhiteSpace(root))
                    yield return Path.Combine(root, "bin");
            }

            string programFiles = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            string cudaRoot = Path.Combine(programFiles, "NVIDIA GPU Computing Toolkit", "CUDA");
            if (!Directory.Exists(cudaRoot))
                yield break;

            foreach (string versionDir in Directory.EnumerateDirectories(cudaRoot, "v*").OrderByDescending(path => path))
                yield return Path.Combine(versionDir, "bin");
        }

        private static bool IsRepoRoot(string path)
        {
            string[] markers =
            {
                "TensorSharp.slnx",
                "TensorSharp.sln",
                "Seq2SeqSharp.sln",
            };

            return markers.Any(marker => File.Exists(Path.Combine(path, marker)))
                || Directory.Exists(Path.Combine(path, ".git"));
        }

        /// <summary>
        /// The native bridge's last error string. Public so the model layer can
        /// report WHY a fused kernel declined: a silent fall-through to a slower
        /// (or, under TP, unsupported) path is how that class of bug hides.
        /// </summary>
        public static string LastNativeError(string fallback = "(no native error)") => GetLastErrorMessage(fallback);

        private static string GetLastErrorMessage(string fallback)
        {
            IntPtr errPtr = TSGgml_GetLastError();
            string message = errPtr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(errPtr);
            return string.IsNullOrWhiteSpace(message) ? fallback : message;
        }

        /// <summary>
        /// True once a GPU command buffer has failed in this process. The op that
        /// drained the dead buffer still returned success — ggml_backend_synchronize
        /// cannot report otherwise — so without this check the first visible symptom
        /// is an unrelated op failing one or more forwards later, over results that
        /// were already undefined.
        ///
        /// Sticky and unrecoverable in-process: see TSGgml_HasBackendFailure.
        /// </summary>
        public static bool HasBackendFailure() => TSGgml_HasBackendFailure() != 0;

        /// <summary>What ggml logged about the failure, or an empty string.</summary>
        public static string BackendFailureText()
        {
            IntPtr ptr = TSGgml_GetBackendFailureText();
            return ptr == IntPtr.Zero ? string.Empty : (Marshal.PtrToStringAnsi(ptr) ?? string.Empty);
        }

        private static string GetBackendAvailabilityHint(GgmlBackendType backendType)
        {
            string defaultMessage = "Build the native GGML bridge and ensure the requested GGML backend is available.";
            string backendMessage = GetLastErrorMessage(defaultMessage);

            if (backendType == GgmlBackendType.Cuda && IsCudaPlatformSupported())
            {
                string rebuildHint = OperatingSystem.IsWindows()
                    ? "Rebuild the native GGML bridge with CUDA enabled, for example: `powershell -ExecutionPolicy Bypass -File TensorSharp.GGML.Native/build-windows.ps1 --cuda` or `set TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON` before `dotnet build`."
                    : "Rebuild the native GGML bridge with CUDA enabled, for example: `bash TensorSharp.GGML.Native/build-linux.sh --cuda` or `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build`.";

                if (string.IsNullOrWhiteSpace(backendMessage))
                    return rebuildHint;

                if (backendMessage.Contains("not available in this build", StringComparison.OrdinalIgnoreCase))
                    return $"{backendMessage} {rebuildHint}";
            }

            if (backendType == GgmlBackendType.Vulkan && IsVulkanPlatformSupported())
            {
                string rebuildHint = OperatingSystem.IsWindows()
                    ? "Rebuild the native GGML bridge with Vulkan enabled, for example: `powershell -ExecutionPolicy Bypass -File TensorSharp.GGML.Native/build-windows.ps1 --vulkan` or `set TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON` before `dotnet build`."
                    : "Rebuild the native GGML bridge with Vulkan enabled, for example: `bash TensorSharp.GGML.Native/build-linux.sh --vulkan` or `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=ON dotnet build`.";

                if (string.IsNullOrWhiteSpace(backendMessage))
                    return rebuildHint;

                if (backendMessage.Contains("not available in this build", StringComparison.OrdinalIgnoreCase))
                    return $"{backendMessage} {rebuildHint}";

                // The backend is compiled in but device enumeration came back empty.
                // ggml-vulkan only auto-selects GPU devices; software rasterizers
                // (llvmpipe/lavapipe) are skipped unless forced. The common trap is
                // WSL: NVIDIA ships no Vulkan ICD for WSL guests and Ubuntu's mesa
                // has no dzn driver, so no GPU Vulkan device exists inside WSL even
                // though the host GPU supports Vulkan.
                if (backendMessage.Contains("No Vulkan device", StringComparison.OrdinalIgnoreCase))
                    return backendMessage +
                        " Install a GPU driver with Vulkan support. ggml-vulkan only auto-selects GPU devices;" +
                        " a software rasterizer (e.g. llvmpipe) is used only when forced with GGML_VK_VISIBLE_DEVICES=0." +
                        " Note: inside WSL no NVIDIA Vulkan device is exposed — run the server on the Windows host" +
                        " (or a Linux host with native GPU drivers) to use ggml_vulkan on the GPU.";
            }

            return backendMessage;
        }
    }
}
