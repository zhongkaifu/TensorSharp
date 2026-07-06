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

    // float scalars (4)
    public float Eps;
    public float RopeBase;
    public float InvSqrtHidden;
    public float LayerOutputScale;
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

internal enum GgmlActivationGradOp
{
    Relu = 1,
    Sigmoid = 2,
    Tanh = 3,
    SiLU = 4,
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

    internal static class GgmlNative
    {
        private const string DllName = "GgmlOps";
        private const CallingConvention CallingConventionType = CallingConvention.Cdecl;
        private static int s_windowsDependencySearchPathsInitialized;

        static GgmlNative()
        {
            NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_GetLastError();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsMetalAvailable();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CanInitializeBackend(int backendType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsBackendAvailable(int backendType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SetVulkanDeviceIndex(int deviceIndex);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetVulkanDeviceCount();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetVulkanDeviceDescription(int deviceIndex, byte[] description, int descriptionSize);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlTensorView2D m1,
            GgmlTensorView2D m2,
            float beta,
            float alpha);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmQuantF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedRmsNormMatMulQuantF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedMatMulQuantAddF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2Ne1,
            long m2RawBytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedFFNSwiGLUQuantF32(
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
            int halfDim);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedFFNActProjectQuantF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedRmsNormResidualAddF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedPleBlockQuantF32(
            GgmlTensorView2D residual,
            GgmlTensorView2D perLayerInput,
            IntPtr inpGateData, int inpGateGgmlType, long inpGateNe0, long inpGateNe1, long inpGateRawBytes,
            IntPtr projData, int projGgmlType, long projNe0, long projNe1, long projRawBytes,
            IntPtr postNormData, int postNormCount, float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedOutProjNormRouterQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outBytes,
            IntPtr normData, int normCount, float eps,
            GgmlTensorView2D normedOut,
            IntPtr routerData, int routerType, long routerNe0, long routerNe1, long routerBytes,
            GgmlTensorView2D routerOut);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedVisionMLPF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr upW, int upNe0, int upNe1, long upBytes,
            IntPtr upB, int upBDim,
            IntPtr downW, int downNe0, int downNe1, long downBytes,
            IntPtr downB, int downBDim);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedOutProjFFNQuantF32(
            GgmlTensorView2D residual, GgmlTensorView2D input,
            IntPtr outProjData, int outProjType, long outNe0, long outNe1, long outRawBytes,
            IntPtr ffnNormData, int ffnNormCount, float eps,
            IntPtr guData, int guType, long guNe0, long guNe1, long guRawBytes,
            IntPtr dnData, int dnType, long dnNe0, long dnNe1, long dnRawBytes,
            int halfDim);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedVisionAttentionF32(
            GgmlTensorView2D hidden,
            IntPtr lnW, IntPtr lnB, int lnDim, float eps,
            IntPtr qkvW, int qkvNe0, int qkvNe1, long qkvBytes,
            IntPtr qkvB, int qkvBDim,
            IntPtr outW, int outNe0, int outNe1, long outBytes,
            IntPtr outB, int outBDim,
            IntPtr cosTable, IntPtr sinTable,
            int numPatches, int numHeads, int headDim, int halfDim,
            float attnScale);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35VisionEncoderF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedGemma4VisionBlockF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetRowsQuantF32(
            GgmlTensorView2D result,
            IntPtr srcData,
            int srcGgmlType,
            long srcNe0,
            long srcNe1,
            long srcRawBytes,
            GgmlContiguousTensor indices);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsForwardF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsSwiGLUForwardF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEExpertsSwiGLUResidualF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmQuantBatchF32(
            GgmlTensorView2D result,
            GgmlTensorView2D m1,
            IntPtr m2Data,
            int m2GgmlType,
            long m2Ne0,
            long m2RawBytes,
            int batchCount,
            long[] weightOffsets,
            long[] weightNe1Arr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddmmBatchF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView3D m1,
            GgmlTensorView3D m2,
            float beta,
            float alpha);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MulMatIdF32(
            GgmlTensorView3D result,
            GgmlTensorView3D expertWeights,
            GgmlTensorView3D input,
            GgmlContiguousTensor ids,
            int idsRows,
            int idsCols);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AddIdF32(
            GgmlTensorView3D result,
            GgmlTensorView3D src,
            GgmlTensorView2D bias,
            GgmlContiguousTensor ids,
            int idsRows,
            int idsCols);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ReduceLastDimF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexReductionF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        // In-place softmax with causal+SWA mask and optional attention sinks.
        // Replaces the GptOss CPU softmax-with-sinks loop. See native side:
        // attention_softmax_with_sinks_f32_impl in ggml_ops_norm_attn.cpp.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AttentionSoftmaxWithSinksF32(
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
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_MoEFFNPrefillSwiGLUQuantF32(
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
            float oaiLimit);

        // Gemma 4 MoE GEGLU + post_norm + residual add fused kernel.
        // Computes residual_in_out += rms_norm(moe_ffn(hidden_in), eps) * post_norm_w
        // in a single GGML graph dispatch. Mirrors the existing
        // TSGgml_MoEFFNPrefillSwiGLUQuantF32 ABI but adds the residual buffer,
        // the post_ffw_norm_2 weight, and an RMSNorm epsilon.
        // See native side: TSGgml_Gemma4MoEGEGLUResidualF32 in ggml_ops_moe.cpp.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEGEGLUResidualF32(
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
            float oaiLimit);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ScaledDotProductAttentionF32(
            GgmlTensorView4D result,
            GgmlTensorView4D query,
            GgmlTensorView4D key,
            GgmlTensorView4D value,
            GgmlTensorView4D mask,
            int hasMask,
            float scale);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SoftmaxGradF32(
            GgmlTensorView4D result,
            GgmlTensorView4D adj,
            GgmlTensorView4D val,
            int addGrad);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossF32(
            out float lossValue,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float smooth,
            float labelSmooth);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CrossEntropyLossBackwardF32(
            GgmlTensorView4D grad,
            GgmlTensorView4D probs,
            GgmlContiguousTensor targetIndices,
            float lossGradient,
            float smooth,
            float labelSmooth,
            int addGrad);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_AdamF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TransformerLayerDecode(
            IntPtr hiddenData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr ffnNormData,
            IntPtr guData, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4LayerPrefill(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedPrefillAttentionF32(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen,
            int maskStartPos, int slidingWindow,
            float scale, int inputFormat);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedPrefillAttentionF16KV(
            IntPtr qData, IntPtr kData, IntPtr vData, IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int seqLen, int kvLen, int kvCacheLen,
            int maskStartPos, int slidingWindow,
            float scale);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FlashAttnDecodeF32(
            IntPtr qData, IntPtr kData, IntPtr vData,
            IntPtr kCacheData, IntPtr vCacheData,
            IntPtr outData,
            int numHeads, int numKvHeads, int headDim,
            int maxSeqLen, int position,
            float scale, int kvCacheType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForward(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardWithSinks(
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
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardDevice(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PagedAttentionForwardDeviceWithSinks(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35AttentionLayerDecode(
            IntPtr residualData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int ropeMode, int kvCacheType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GptOssAttentionLayerPrefill(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35AttentionLayerPrefill(
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
            float eps);

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
            float eps)
        {
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
                ropeMode, kvCacheType, eps), "qwen35_attention_layer_prefill");
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_TransformerModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            int oType, long oNe0, long oNe1, long oBytes,
            int guType, long guNe0, long guNe1, long guBytes,
            int downType, long downNe0, long downNe1, long downBytes,
            int headDim, int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4ModelDecode(
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
            IntPtr pleModelProjNormData);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4ModelDecodeBatched(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4ModelVerify(
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
            IntPtr pleProjNormData);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4DraftStep(
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
            if (r == 0 && Environment.GetEnvironmentVariable("TS_GGML_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[gemma4-draft FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc);

        public static void Gemma4MoELayerDecode(in Gemma4MoELayerDecodeArgs desc)
        {
            CheckResult(TSGgml_Gemma4MoELayerDecode(in desc), nameof(TSGgml_Gemma4MoELayerDecode));
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DiffusionDecodeLayer(in DiffusionDecodeLayerArgs desc);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenImageModMlp(in QwenImageModMlpArgs desc);

        public static bool TryQwenImageModMlp(in QwenImageModMlpArgs desc) => TSGgml_QwenImageModMlp(in desc) != 0;

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenImageJointAttn(in QwenImageJointAttnArgs desc);

        public static bool TryQwenImageJointAttn(in QwenImageJointAttnArgs desc)
        {
            int r = TSGgml_QwenImageJointAttn(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-attn FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenImageBlock(in QwenImageBlockArgs desc);

        public static bool TryQwenImageBlock(in QwenImageBlockArgs desc)
        {
            int r = TSGgml_QwenImageBlock(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-block FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenImageBlockCfg(in QwenImageBlockArgs condDesc, in QwenImageBlockArgs negDesc);

        // CFG-batched block: both true-CFG branches in one dispatch sharing the weights.
        public static bool TryQwenImageBlockCfg(in QwenImageBlockArgs condDesc, in QwenImageBlockArgs negDesc)
        {
            int r = TSGgml_QwenImageBlockCfg(in condDesc, in negDesc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-block-cfg FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenImageForward(in QwenImageForwardArgs desc);

        // Whole 60-block DiT forward in one resident-weight graph (in-graph modulation).
        public static bool TryQwenImageForward(in QwenImageForwardArgs desc)
        {
            int r = TSGgml_QwenImageForward(in desc);
            if (r == 0)
                Console.Error.WriteLine($"[qwen-image-forward FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Conv2d(in Conv2dArgs desc);

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
            if (r == 0 && Environment.GetEnvironmentVariable("DIFFUSION_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[fused-decode FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DiffusionLmHead(
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
            if (r == 0 && Environment.GetEnvironmentVariable("DIFFUSION_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[fused-lmhead FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DiffusionLmHeadSample(
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
            if (r == 0 && Environment.GetEnvironmentVariable("DIFFUSION_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[fused-lmhead-sample FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DiffusionModelDecode(
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
            if (r == 0 && Environment.GetEnvironmentVariable("DIFFUSION_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[fused-model FAIL] {GetLastErrorMessage("(no native error)")}");
            return r != 0;
        }

        // Model-wide MoE decode: the whole transformer as one graph/token.
        // `layers` is one Gemma4MoELayerDecodeArgs per layer (blittable, marshalled
        // as a contiguous TSGgmlGemma4MoELayerDesc array). hidden/position come from
        // the explicit params; the per-element Hidden/Position fields are ignored.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEModelDecode(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap);

        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position)
        {
            CheckResult(TSGgml_Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position,
                IntPtr.Zero, 0, IntPtr.Zero, 0, 0, 0, 0, IntPtr.Zero, 0.0f), nameof(TSGgml_Gemma4MoEModelDecode));
        }

        // Folded variant: appends final-norm + lm_head + softcap so logits[vocab] are
        // written to <paramref name="logits"/> as part of the captured replay.
        public static void Gemma4MoEModelDecode(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int position,
            IntPtr logits, int vocabSize, IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm, float logitSoftcap)
        {
            CheckResult(TSGgml_Gemma4MoEModelDecode(layers, numLayers, hidden, hiddenSize, position,
                logits, vocabSize, lmHead, lmHeadType, lmHeadNe0, lmHeadNe1, lmHeadBytes, finalNorm, logitSoftcap),
                nameof(TSGgml_Gemma4MoEModelDecode));
        }

        // TRUE token-batched MoE decode: N concurrent sequences, one token each, in
        // one captured graph. Reuses the per-layer descriptor array for weights;
        // KV caches are per-(layer,seq) [layer*nSeqs+seq]; positions per seq.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEModelDecodeBatched(
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
            if (rc == 0 && Environment.GetEnvironmentVariable("TS_BATCHED_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[g4moe-batched-native FAIL] {GetLastErrorMessage("(no native error)")}");
            return rc != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Gemma4ResetMoEBatchedDecodeCache();
        public static void Gemma4ResetMoEBatchedDecodeCache() => TSGgml_Gemma4ResetMoEBatchedDecodeCache();

        // Model-wide MoE multi-token verify: the whole MoE transformer over N tokens
        // as one graph. Reuses the same descriptor array as the decode; start_pos +
        // num_tokens are explicit. Returns 0 (false) when the kernel cannot handle
        // the shape so the caller falls back to the per-op verify.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Gemma4MoEModelVerify(
            [In] Gemma4MoELayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int startPos, int numTokens);

        public static bool Gemma4MoEModelVerify(Gemma4MoELayerDecodeArgs[] layers, int numLayers, IntPtr hidden, int hiddenSize, int startPos, int numTokens)
        {
            return TSGgml_Gemma4MoEModelVerify(layers, numLayers, hidden, hiddenSize, startPos, numTokens) != 0;
        }

        // Qwen3.5/3.6 full-model decode: the whole hybrid transformer (full-attention
        // + GatedDeltaNet recurrent layers + per-layer FFN) as one graph/token.
        // Returns 0 when it cannot handle the shape so the caller falls back to per-op.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35ModelDecode(
            [In] Qwen35LayerDecodeArgs[] layers, int numLayers,
            IntPtr hidden, int hiddenSize, int position,
            int numHeads, int numKvHeads, int headDim, int cacheSize,
            int ropeNDims, int ropeMode, int kvCacheType,
            int convKernel, int headKDim, int headVDim, int numKHeads, int numVHeads,
            float eps, float ropeBase, float ropeFreqScale,
            int numExperts, int numExpertsUsed, int expertFf, int sharedFf,
            int normTopk, float expertWeightsScale,
            IntPtr logits, int vocabSize,
            IntPtr lmHead, int lmHeadType, long lmHeadNe0, long lmHeadNe1, long lmHeadBytes,
            IntPtr finalNorm);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Qwen35ResetDecodeCache();

        public static void Qwen35ResetDecodeCache() => TSGgml_Qwen35ResetDecodeCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Gemma4ResetDecodeCache();

        public static void Gemma4ResetDecodeCache() => TSGgml_Gemma4ResetDecodeCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Gemma4ResetBatchedDecodeCache();

        public static void Gemma4ResetBatchedDecodeCache() => TSGgml_Gemma4ResetBatchedDecodeCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Gemma4MoEResetDecodeCache();

        public static void Gemma4MoEResetDecodeCache() => TSGgml_Gemma4MoEResetDecodeCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Qwen35ResetVerifyCache();

        public static void Qwen35ResetVerifyCache() => TSGgml_Qwen35ResetVerifyCache();

        // Qwen3.5/3.6 TRUE token-batched fused decode: N sequences' decode tokens
        // (one per sequence) through the whole hybrid transformer in ONE graph,
        // with PAGED KV (slot_mapping write + per-seq get_rows gather). Returns 0
        // when it cannot handle the shape so the caller falls back to op-by-op.
        // padKv = fixed per-seq gather length (round_up(maxSeqLen, stride)); gatherIdx is
        // [nSeqs*padKv] (real slots then pad), seqLens [nSeqs] drives the per-seq attn mask.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35ModelDecodeBatched(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Qwen35ResetBatchedDecodeCache();

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
            IntPtr hidden, int hiddenSize, int position,
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
            return TSGgml_Qwen35ModelDecode(
                layers, numLayers, hidden, hiddenSize, position,
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
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35ModelVerify(
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
            int[] mropePos, int[] mropeSections);

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
            int[] mropePos = null, int[] mropeSections = null)
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
                finalNorm, normedOut, nLogitRows, mropePos, mropeSections) != 0;
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_Qwen35RecurrentLayerPrefill(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GatedDeltaNetChunkedF32(
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
            float eps);

        // Mirrors NemoMamba2BatchedSeqDesc in ggml_ops_mamba2.cpp; same 32-byte
        // POD layout on 64-bit (two ints, two padding ints, two pointers).
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2BatchedStepF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GatedDeltaNetBatchedStepF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2PrefillF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NemotronMamba2DecodeF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_NemotronMamba2DecodeClear(ulong modelKey);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_AlignedAlloc(UIntPtr size);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_AlignedFree(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ClearHostBufferCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_Shutdown();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ReleaseReuseComputeBuffers();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_InvalidateHostBuffer(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_SyncHostBuffer(IntPtr ptr, long byteCount);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern long TSGgml_DeviceCopyCacheResidentBytes();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetBackendMemory(out long freeBytes, out long totalBytes);

        // Async dispatch (deferred ggml_backend_synchronize). When enabled, per-op
        // kernels return without waiting on the Metal command buffer; subsequent ops
        // chain through the Metal command queue, and host-side reads must call
        // TSGgml_HostReadBarrier first to drain pending GPU work. See
        // GgmlStorage.EnsureHostReadable for the C# entry point that triggers this.
        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_SetAsyncCompute(int enabled);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_GetAsyncCompute();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_HostReadBarrier();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_RegisterOffloadable(IntPtr key);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_SetOffloadableBudget(long bytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ClearOffloadableState();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_SetDeviceCopyBudget(long bytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DeviceMemoryInfo(out long freeBytes, out long totalBytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RegisterPinnedHostBuffer(IntPtr ptr, long bytes);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_UnregisterPinnedHostBuffer(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern UIntPtr TSGgml_RowSize(int ggmlType, long ne);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DequantizeToF32(int ggmlType, IntPtr src, long numElements, IntPtr dst);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ApplyLoraDelta(IntPtr w, int ggmlType, long ne0, long ne1,
            IntPtr up, IntPtr down, int rank, float scale, int nThreads);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenVaeRun(in QwenVaeArgs args);

        /// <summary>Run a whole VAE encode/decode op-list as ONE device graph (see QwenVaeArgs).
        /// Returns false when the backend can't run it (caller falls back to the per-conv path).</summary>
        internal static bool TryQwenVaeRun(in QwenVaeArgs args) => TSGgml_QwenVaeRun(in args) != 0;

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_QwenTeTrunk(in QwenTeTrunkArgs args);

        /// <summary>Run a whole conditioning-encoder transformer trunk as ONE device graph
        /// (see QwenTeTrunkArgs). Returns false when the backend can't run it (caller falls
        /// back to the per-op path).</summary>
        internal static bool TryQwenTeTrunk(in QwenTeTrunkArgs args) => TSGgml_QwenTeTrunk(in args) != 0;

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_CopyF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_UnaryF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryTensorF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D lhs,
            GgmlTensorView4D rhs);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedActMulF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D a,
            GgmlTensorView4D b);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_FusedActMulSplitF32(
            int op,
            GgmlTensorView2D result,
            GgmlTensorView2D gateUp,
            int halfDim);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_BinaryScalarF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            float scalar);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_ActivationGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D grad,
            GgmlTensorView4D accumulation,
            int hasAccumulation);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            GgmlTensorView4D gamma,
            GgmlTensorView4D beta,
            int hasBeta,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_NormGradF32(
            int op,
            GgmlTensorView4D result,
            GgmlTensorView4D gradGamma,
            GgmlTensorView4D gradBeta,
            GgmlTensorView4D adj,
            GgmlTensorView4D x,
            GgmlTensorView4D gamma,
            int hasGradBeta,
            float eps);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectF32(
            GgmlTensorView2D result,
            GgmlTensorView2D src,
            GgmlContiguousTensor indices,
            int addToResult);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IndexSelectGradF32(
            GgmlTensorView2D grad,
            GgmlTensorView2D adj,
            GgmlContiguousTensor indices);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEF32(
            GgmlTensorView4D result,
            GgmlTensorView4D src,
            int seqLen,
            int rowOffset,
            int addToResult,
            int invertPositions);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEExF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEMRoPEF32(
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

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_RoPEExFreqFactorsF32(
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
            IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_FusedMatMulQuantAddF32(
                residual, input, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "fused_matmul_quant_add");
        }

        public static void FusedFFNSwiGLUQuant(
            GgmlTensorView2D residual,
            GgmlTensorView2D input,
            IntPtr normWeightData,
            int normWeightCount,
            float eps,
            IntPtr gateUpData, int gateUpGgmlType, long gateUpNe0, long gateUpNe1, long gateUpRawBytes,
            IntPtr downData, int downGgmlType, long downNe0, long downNe1, long downRawBytes,
            int halfDim)
        {
            CheckResult(TSGgml_FusedFFNSwiGLUQuantF32(
                residual, input, normWeightData, normWeightCount, eps,
                gateUpData, gateUpGgmlType, gateUpNe0, gateUpNe1, gateUpRawBytes,
                downData, downGgmlType, downNe0, downNe1, downRawBytes,
                halfDim), "fused_ffn_swiglu_quant");
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
            if (rc == 0 && Environment.GetEnvironmentVariable("TS_VBENCH_DIAG") == "1")
                Console.Error.WriteLine($"\n[diag] Qwen35VisionEncoder native FAIL: {GetLastErrorMessage("(no native error)")}");
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

        public static void MulMatId(GgmlTensorView3D result, GgmlTensorView3D expertWeights, GgmlTensorView3D input, GgmlContiguousTensor ids, int idsRows, int idsCols)
        {
            CheckResult(TSGgml_MulMatIdF32(result, expertWeights, input, ids, idsRows, idsCols), "mulmatid");
        }

        public static void AddId(GgmlTensorView3D result, GgmlTensorView3D src, GgmlTensorView2D bias, GgmlContiguousTensor ids, int idsRows, int idsCols)
        {
            CheckResult(TSGgml_AddIdF32(result, src, bias, ids, idsRows, idsCols), "addid");
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
            float oaiLimit)
        {
            CheckResult(TSGgml_MoEFFNPrefillSwiGLUQuantF32(
                hiddenIn, hiddenOut, seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit),
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
            float oaiLimit)
        {
            CheckResult(TSGgml_Gemma4MoEGEGLUResidualF32(
                hiddenIn, residualInOut, postNormW, postNormEps,
                seqLen, hiddenDim, nFf,
                numExperts, nUsed, selectedExperts, routingWeights,
                gateData, gateType, gateNe0, gateNe1, gateTotalBytes,
                upData,   upType,   upNe0,   upNe1,   upTotalBytes,
                downData, downType, downNe0, downNe1, downTotalBytes,
                gateBias, upBias, downBias,
                activationType, oaiAlpha, oaiLimit),
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

        public static float CrossEntropyLoss(GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float smooth, float labelSmooth)
        {
            CheckResult(TSGgml_CrossEntropyLossF32(out float lossValue, probs, targetIndices, smooth, labelSmooth), "crossentropyloss");
            return lossValue;
        }

        public static void CrossEntropyLossBackward(GgmlTensorView4D grad, GgmlTensorView4D probs, GgmlContiguousTensor targetIndices, float lossGradient, float smooth, float labelSmooth, bool addGrad)
        {
            CheckResult(TSGgml_CrossEntropyLossBackwardF32(grad, probs, targetIndices, lossGradient, smooth, labelSmooth, addGrad ? 1 : 0), "crossentropyloss_backward");
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

        public static void ActivationGrad(GgmlActivationGradOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D grad, GgmlTensorView4D accumulation, bool hasAccumulation)
        {
            CheckResult(TSGgml_ActivationGradF32((int)op, result, src, grad, accumulation, hasAccumulation ? 1 : 0), $"{op}Grad");
        }

        public static void Norm(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D src, GgmlTensorView4D gamma, GgmlTensorView4D beta, bool hasBeta, float eps)
        {
            CheckResult(TSGgml_NormF32((int)op, result, src, gamma, beta, hasBeta ? 1 : 0, eps), op.ToString());
        }

        public static void NormGrad(GgmlNormOp op, GgmlTensorView4D result, GgmlTensorView4D gradGamma, GgmlTensorView4D gradBeta, GgmlTensorView4D adj, GgmlTensorView4D x, GgmlTensorView4D gamma, bool hasGradBeta, float eps)
        {
            CheckResult(TSGgml_NormGradF32((int)op, result, gradGamma, gradBeta, adj, x, gamma, hasGradBeta ? 1 : 0, eps), $"{op}Grad");
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

        public static void TransformerLayerDecode(
            IntPtr hiddenData, int hiddenSize,
            IntPtr attnNormData,
            IntPtr qkvData, int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            IntPtr qNormData, IntPtr kNormData, int headDim,
            IntPtr oData, int oType, long oNe0, long oNe1, long oBytes,
            IntPtr ffnNormData,
            IntPtr guData, int guType, long guNe0, long guNe1, long guBytes,
            IntPtr downData, int downType, long downNe0, long downNe1, long downBytes,
            IntPtr kCacheData, IntPtr vCacheData,
            int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_TransformerLayerDecode(
                hiddenData, hiddenSize,
                attnNormData,
                qkvData, qkvType, qkvNe0, qkvNe1, qkvBytes,
                qNormData, kNormData, headDim,
                oData, oType, oNe0, oNe1, oBytes,
                ffnNormData,
                guData, guType, guNe0, guNe1, guBytes,
                downData, downType, downNe0, downNe1, downBytes,
                kCacheData, vCacheData,
                numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType), "transformer_layer_decode");
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
            int ropeMode, int kvCacheType = 0)
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
                eps, ropeBase, ropeFreqScale, ropeMode, kvCacheType), "qwen35_attention_layer_decode");
        }

        public static void TransformerModelDecode(
            IntPtr hiddenData, int hiddenSize, int numLayers,
            IntPtr[] attnNormArr, IntPtr[] qkvArr, IntPtr[] qNormArr, IntPtr[] kNormArr,
            IntPtr[] oArr, IntPtr[] ffnNormArr, IntPtr[] guArr, IntPtr[] downArr,
            IntPtr[] kCacheArr, IntPtr[] vCacheArr,
            int qkvType, long qkvNe0, long qkvNe1, long qkvBytes,
            int oType, long oNe0, long oNe1, long oBytes,
            int guType, long guNe0, long guNe1, long guBytes,
            int downType, long downNe0, long downNe1, long downBytes,
            int headDim, int numHeads, int numKvHeads,
            int maxSeqLen, int position,
            float eps, float ropeBase, float ropeFreqScale,
            int intermediateSize, int ropeMode,
            int kvCacheType = 0)
        {
            CheckResult(TSGgml_TransformerModelDecode(
                hiddenData, hiddenSize, numLayers,
                attnNormArr, qkvArr, qNormArr, kNormArr,
                oArr, ffnNormArr, guArr, downArr,
                kCacheArr, vCacheArr,
                qkvType, qkvNe0, qkvNe1, qkvBytes,
                oType, oNe0, oNe1, oBytes,
                guType, guNe0, guNe1, guBytes,
                downType, downNe0, downNe1, downBytes,
                headDim, numHeads, numKvHeads,
                maxSeqLen, position,
                eps, ropeBase, ropeFreqScale,
                intermediateSize, ropeMode, kvCacheType), "transformer_model_decode");
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
            IntPtr pleModelProjNormData = default)
        {
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
                pleModelProjNormData), "gemma4_model_decode");
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
            if (rc == 0 && Environment.GetEnvironmentVariable("TS_BATCHED_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[g4-batched-native FAIL] {GetLastErrorMessage("(no native error)")}");
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
            IntPtr pleProjNormData = default)
        {
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
                pleProjNormData);
            if (r == 0 && Environment.GetEnvironmentVariable("TS_GGML_FUSED_DEBUG") == "1")
                Console.Error.WriteLine($"[gemma4-verify FAIL] {GetLastErrorMessage("(no native error)")}");
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
            float eps)
        {
            CheckResult(TSGgml_GatedDeltaNetChunkedF32(
                q, k, v, z, alpha, beta, state, gatedOut,
                dtBiasData, aLogData, ssmNormWData,
                chunkSize, eps), "gated_delta_net_chunked");
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
                TSGgml_InvalidateHostBuffer(ptr);
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

        public static void PreloadQuantizedWeight(IntPtr cacheKey, IntPtr hostData, int ggmlType, long ne0, long ne1, long rawBytes)
        {
            if (cacheKey == IntPtr.Zero || hostData == IntPtr.Zero || rawBytes <= 0)
                throw new ArgumentException("PreloadQuantizedWeight requires valid cache key, host data, and size.");

            CheckResult(TSGgml_PreloadQuantizedWeight(cacheKey, hostData, ggmlType, ne0, ne1, rawBytes), "preload_quantized_weight");
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

        private static string GetLastErrorMessage(string fallback)
        {
            IntPtr errPtr = TSGgml_GetLastError();
            string message = errPtr == IntPtr.Zero ? null : Marshal.PtrToStringAnsi(errPtr);
            return string.IsNullOrWhiteSpace(message) ? fallback : message;
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
