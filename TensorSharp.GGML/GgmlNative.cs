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

namespace TensorSharp.GGML
{

public enum GgmlBackendType
{
    Metal = 1,
    Cpu = 2,
    Cuda = 3,
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

        static GgmlNative()
        {
            NativeLibrary.SetDllImportResolver(typeof(GgmlNative).Assembly, ImportResolver);
        }

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_GetLastError();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsMetalAvailable();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_IsBackendAvailable(int backendType);

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
        private static extern int TSGgml_GetRowsQuantF32(
            GgmlTensorView2D result,
            IntPtr srcData,
            int srcGgmlType,
            long srcNe0,
            long srcNe1,
            long srcRawBytes,
            GgmlContiguousTensor indices);

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
            int intermediateSize, int ropeMode);

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
            int intermediateSize, int ropeMode);

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
            IntPtr[] plePostNormArr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern IntPtr TSGgml_AlignedAlloc(UIntPtr size);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_AlignedFree(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_ClearHostBufferCache();

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern void TSGgml_InvalidateHostBuffer(IntPtr ptr);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern UIntPtr TSGgml_RowSize(int ggmlType, long ne);

        [DllImport(DllName, CallingConvention = CallingConventionType)]
        private static extern int TSGgml_DequantizeToF32(int ggmlType, IntPtr src, long numElements, IntPtr dst);

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

        public static void EnsureAvailable(GgmlBackendType backendType)
        {
            if (backendType == GgmlBackendType.Metal && !OperatingSystem.IsMacOS())
            {
                throw new PlatformNotSupportedException("The GGML Metal backend is available on macOS only.");
            }

            if (backendType == GgmlBackendType.Cuda && !OperatingSystem.IsLinux())
            {
                throw new PlatformNotSupportedException("The GGML CUDA backend is currently supported on Linux only.");
            }

            try
            {
                if (TSGgml_IsBackendAvailable((int)backendType) == 0)
                {
                    string backendName = backendType switch
                    {
                        GgmlBackendType.Metal => "ggml-metal",
                        GgmlBackendType.Cuda => "ggml-cuda",
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

        public static void Addmm(GgmlTensorView2D result, GgmlTensorView2D src, GgmlTensorView2D m1, GgmlTensorView2D m2, float beta, float alpha)
        {
            CheckResult(TSGgml_AddmmF32(result, src, m1, m2, beta, alpha), "addmm");
        }

        public static void AddmmQuant(GgmlTensorView2D result, GgmlTensorView2D m1, IntPtr m2Data, int m2GgmlType, long m2Ne0, long m2Ne1, long m2RawBytes)
        {
            CheckResult(TSGgml_AddmmQuantF32(result, m1, m2Data, m2GgmlType, m2Ne0, m2Ne1, m2RawBytes), "addmm_quant");
        }

        public static void GetRowsQuant(GgmlTensorView2D result, IntPtr srcData, int srcGgmlType, long srcNe0, long srcNe1, long srcRawBytes, GgmlContiguousTensor indices)
        {
            CheckResult(TSGgml_GetRowsQuantF32(result, srcData, srcGgmlType, srcNe0, srcNe1, srcRawBytes, indices), "get_rows_quant");
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
            int intermediateSize, int ropeMode)
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
                intermediateSize, ropeMode), "transformer_layer_decode");
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
            int intermediateSize, int ropeMode)
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
                intermediateSize, ropeMode), "transformer_model_decode");
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
            IntPtr[] plePostNormArr)
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
                plePostNormArr), "gemma4_model_decode");
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

        public static void InvalidateHostBuffer(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
                TSGgml_InvalidateHostBuffer(ptr);
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

            if (backendType == GgmlBackendType.Cuda && OperatingSystem.IsLinux())
            {
                const string rebuildHint = "Rebuild the native GGML bridge with CUDA enabled, for example: `bash TensorSharp.GGML.Native/build-linux.sh --cuda` or `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON dotnet build`.";

                if (string.IsNullOrWhiteSpace(backendMessage))
                    return rebuildHint;

                if (backendMessage.Contains("not available in this build", StringComparison.OrdinalIgnoreCase))
                    return $"{backendMessage} {rebuildHint}";
            }

            return backendMessage;
        }
    }
}
