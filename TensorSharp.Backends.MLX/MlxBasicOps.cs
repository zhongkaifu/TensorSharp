using System;
using TensorSharp.Core;

namespace TensorSharp.MLX
{
    [OpsClass]
    public sealed class MlxBasicOps
    {
        [RegisterOpStorageType("fill", typeof(MlxStorage))]
        public static void Fill(Tensor result, float value)
        {
            if (!CanUseNativeWriteTarget(result))
            {
                FallbackVoid("fill", result, value);
                return;
            }

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray output = default;
                try
                {
                    output = MlxNative.Full(ToIntArray(result.Sizes), value, result.ElementType);
                    SetDeviceResult(result, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(output);
                }
            });
        }

        [RegisterOpStorageType("copy", typeof(MlxStorage))]
        public static void Copy(Tensor result, Tensor src)
        {
            if (!CanUseNativeCopy(result, src))
            {
                FallbackVoid("copy", result, src);
                return;
            }

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray casted = default;
                MlxNative.MlxArray contiguous = default;
                try
                {
                    srcView = GetView(src);
                    MlxNative.MlxArray copySource = srcView;
                    if (src.ElementType != result.ElementType)
                    {
                        casted = MlxNative.Astype(srcView, result.ElementType);
                        copySource = casted;
                    }

                    contiguous = MlxNative.Contiguous(copySource);
                    SetDeviceResult(result, contiguous);
                    contiguous = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(casted);
                    MlxNative.FreeArray(contiguous);
                }
            });
        }

        [RegisterOpStorageType("addmm", typeof(MlxStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src, m1, m2))
                return FallbackTensor("addmm", writeTarget, beta, src, alpha, m1, m2);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray m1View = default;
                MlxNative.MlxArray m2View = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    m1View = GetView(m1);
                    m2View = GetView(m2);
                    output = MlxNative.Addmm(srcView, m1View, m2View, alpha, beta);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(m1View);
                    MlxNative.FreeArray(m2View);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("addmmbatch", typeof(MlxStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src, m1, m2))
                return FallbackTensor("addmmbatch", writeTarget, beta, src, alpha, m1, m2);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray m1View = default;
                MlxNative.MlxArray m2View = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    m1View = GetView(m1);
                    m2View = GetView(m2);
                    output = MlxNative.Addmm(srcView, m1View, m2View, alpha, beta);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(m1View);
                    MlxNative.FreeArray(m2View);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("abs", typeof(MlxStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => Unary("abs", result, src, MlxNative.MlxUnaryOp.Abs);

        [RegisterOpStorageType("neg", typeof(MlxStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => Unary("neg", result, src, MlxNative.MlxUnaryOp.Neg);

        [RegisterOpStorageType("sqrt", typeof(MlxStorage))]
        public static Tensor Sqrt(Tensor result, Tensor src) => Unary("sqrt", result, src, MlxNative.MlxUnaryOp.Sqrt);

        [RegisterOpStorageType("rsqrt", typeof(MlxStorage))]
        public static Tensor Rsqrt(Tensor result, Tensor src) => Unary("rsqrt", result, src, MlxNative.MlxUnaryOp.Rsqrt);

        [RegisterOpStorageType("exp", typeof(MlxStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => Unary("exp", result, src, MlxNative.MlxUnaryOp.Exp);

        [RegisterOpStorageType("log", typeof(MlxStorage))]
        public static Tensor Log(Tensor result, Tensor src) => Unary("log", result, src, MlxNative.MlxUnaryOp.Log);

        [RegisterOpStorageType("log1p", typeof(MlxStorage))]
        public static Tensor Log1p(Tensor result, Tensor src) => Unary("log1p", result, src, MlxNative.MlxUnaryOp.Log1p);

        [RegisterOpStorageType("floor", typeof(MlxStorage))]
        public static Tensor Floor(Tensor result, Tensor src) => Unary("floor", result, src, MlxNative.MlxUnaryOp.Floor);

        [RegisterOpStorageType("ceil", typeof(MlxStorage))]
        public static Tensor Ceil(Tensor result, Tensor src) => Unary("ceil", result, src, MlxNative.MlxUnaryOp.Ceil);

        [RegisterOpStorageType("sin", typeof(MlxStorage))]
        public static Tensor Sin(Tensor result, Tensor src) => Unary("sin", result, src, MlxNative.MlxUnaryOp.Sin);

        [RegisterOpStorageType("cos", typeof(MlxStorage))]
        public static Tensor Cos(Tensor result, Tensor src) => Unary("cos", result, src, MlxNative.MlxUnaryOp.Cos);

        [RegisterOpStorageType("tanh", typeof(MlxStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => Unary("tanh", result, src, MlxNative.MlxUnaryOp.Tanh);

        [RegisterOpStorageType("sigmoid", typeof(MlxStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => Unary("sigmoid", result, src, MlxNative.MlxUnaryOp.Sigmoid);

        [RegisterOpStorageType("relu", typeof(MlxStorage))]
        public static Tensor Relu(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("relu", writeTarget, src);

            MlxNative.MlxArray srcView = default;
            MlxNative.MlxArray zero = default;
            MlxNative.MlxArray output = default;
            try
            {
                srcView = GetView(src);
                zero = MlxNative.NewScalar(0.0f);
                output = MlxNative.Binary(MlxNative.MlxBinaryOp.Maximum, srcView, zero);
                SetDeviceResult(writeTarget, output);
                output = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(srcView);
                MlxNative.FreeArray(zero);
                MlxNative.FreeArray(output);
            }
        }

        [RegisterOpStorageType("SiLU", typeof(MlxStorage))]
        public static Tensor SiLU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("SiLU", writeTarget, src);

            MlxNative.MlxArray srcView = default;
            MlxNative.MlxArray sigmoid = default;
            MlxNative.MlxArray output = default;
            try
            {
                srcView = GetView(src);
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.SiLU(srcView);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, srcView);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, srcView, sigmoid);
                }
                SetDeviceResult(writeTarget, output);
                output = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(srcView);
                MlxNative.FreeArray(sigmoid);
                MlxNative.FreeArray(output);
            }
        }

        [RegisterOpStorageType("GELU", typeof(MlxStorage))]
        public static Tensor GELU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("GELU", writeTarget, src);

            MlxNative.MlxArray srcView = default;
            MlxNative.MlxArray output = default;
            try
            {
                srcView = GetView(src);
                if (!MlxCompiledOps.Disabled)
                    output = MlxCompiledOps.GeluTanh(srcView);
                else
                    output = Gelu(srcView);
                SetDeviceResult(writeTarget, output);
                output = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(srcView);
                MlxNative.FreeArray(output);
            }
        }

        [RegisterOpStorageType("addt", typeof(MlxStorage))]
        public static Tensor AddTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("addt", result, lhs, rhs, MlxNative.MlxBinaryOp.Add);

        [RegisterOpStorageType("subt", typeof(MlxStorage))]
        public static Tensor SubTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("subt", result, lhs, rhs, MlxNative.MlxBinaryOp.Sub);

        [RegisterOpStorageType("mult", typeof(MlxStorage))]
        public static Tensor MulTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("mult", result, lhs, rhs, MlxNative.MlxBinaryOp.Mul);

        [RegisterOpStorageType("divt", typeof(MlxStorage))]
        public static Tensor DivTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("divt", result, lhs, rhs, MlxNative.MlxBinaryOp.Div);

        [RegisterOpStorageType("addv", typeof(MlxStorage))]
        public static Tensor AddValue(Tensor result, Tensor lhs, float rhs) => Scalar("addv", result, lhs, rhs, MlxNative.MlxBinaryOp.Add, false);

        [RegisterOpStorageType("subv", typeof(MlxStorage))]
        public static Tensor SubValue(Tensor result, Tensor lhs, float rhs) => Scalar("subv", result, lhs, rhs, MlxNative.MlxBinaryOp.Sub, false);

        [RegisterOpStorageType("rsubv", typeof(MlxStorage))]
        public static Tensor RSubValue(Tensor result, float lhs, Tensor rhs) => Scalar("rsubv", result, rhs, lhs, MlxNative.MlxBinaryOp.Sub, true);

        [RegisterOpStorageType("mulv", typeof(MlxStorage))]
        public static Tensor MulValue(Tensor result, Tensor lhs, float rhs) => Scalar("mulv", result, lhs, rhs, MlxNative.MlxBinaryOp.Mul, false);

        [RegisterOpStorageType("divv", typeof(MlxStorage))]
        public static Tensor DivValue(Tensor result, Tensor lhs, float rhs) => Scalar("divv", result, lhs, rhs, MlxNative.MlxBinaryOp.Div, false);

        [RegisterOpStorageType("rdivv", typeof(MlxStorage))]
        public static Tensor RDivValue(Tensor result, float lhs, Tensor rhs) => Scalar("rdivv", result, rhs, lhs, MlxNative.MlxBinaryOp.Div, true);

        [RegisterOpStorageType("SiLUMul", typeof(MlxStorage))]
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(gate, up))
                return FallbackTensor("SiLUMul", writeTarget, gate, up);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray gateView = default;
                MlxNative.MlxArray upView = default;
                MlxNative.MlxArray sigmoid = default;
                MlxNative.MlxArray silu = default;
                MlxNative.MlxArray output = default;
                try
                {
                    gateView = GetView(gate);
                    upView = GetView(up);
                    if (!MlxCompiledOps.Disabled)
                    {
                        // Single fused kernel: silu(gate) * up.
                        output = MlxCompiledOps.SwiGLU(gateView, upView);
                    }
                    else
                    {
                        sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gateView);
                        silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gateView, sigmoid);
                        output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, upView);
                    }
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(gateView);
                    MlxNative.FreeArray(upView);
                    MlxNative.FreeArray(sigmoid);
                    MlxNative.FreeArray(silu);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("SiLUMulSplit", typeof(MlxStorage))]
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            if (gateUp == null)
                throw new ArgumentNullException(nameof(gateUp));

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(gateUp)
                || gateUp.DimensionCount != 2
                || halfDim <= 0
                || gateUp.Sizes[0] > int.MaxValue
                || gateUp.Sizes[1] < halfDim * 2L)
            {
                return FallbackTensor("SiLUMulSplit", writeTarget, gateUp, halfDim);
            }

            int rows = checked((int)gateUp.Sizes[0]);
            MlxNative.MlxArray gateUpView = default;
            MlxNative.MlxArray gate = default;
            MlxNative.MlxArray up = default;
            MlxNative.MlxArray sigmoid = default;
            MlxNative.MlxArray silu = default;
            MlxNative.MlxArray output = default;
            try
            {
                gateUpView = GetView(gateUp);
                gate = MlxNative.Slice(gateUpView, new[] { 0, 0 }, new[] { rows, halfDim }, new[] { 1, 1 });
                up = MlxNative.Slice(gateUpView, new[] { 0, halfDim }, new[] { rows, halfDim * 2 }, new[] { 1, 1 });
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.SwiGLU(gate, up);
                }
                else
                {
                    sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gate);
                    silu = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gate, sigmoid);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, silu, up);
                }
                SetDeviceResult(writeTarget, output);
                output = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(gateUpView);
                MlxNative.FreeArray(gate);
                MlxNative.FreeArray(up);
                MlxNative.FreeArray(sigmoid);
                MlxNative.FreeArray(silu);
                MlxNative.FreeArray(output);
            }
        }

        [RegisterOpStorageType("GELUMul", typeof(MlxStorage))]
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(gate, up))
                return FallbackTensor("GELUMul", writeTarget, gate, up);

            MlxNative.MlxArray gateView = default;
            MlxNative.MlxArray upView = default;
            MlxNative.MlxArray gelu = default;
            MlxNative.MlxArray output = default;
            try
            {
                gateView = GetView(gate);
                upView = GetView(up);
                if (!MlxCompiledOps.Disabled)
                {
                    output = MlxCompiledOps.GeGLU(gateView, upView);
                }
                else
                {
                    gelu = Gelu(gateView);
                    output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, gelu, upView);
                }
                SetDeviceResult(writeTarget, output);
                output = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(gateView);
                MlxNative.FreeArray(upView);
                MlxNative.FreeArray(gelu);
                MlxNative.FreeArray(output);
            }
        }

        [RegisterOpStorageType("SigmoidMul", typeof(MlxStorage))]
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(x, gate))
                return FallbackTensor("SigmoidMul", writeTarget, x, gate);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray xView = default;
                MlxNative.MlxArray gateView = default;
                MlxNative.MlxArray sigmoid = default;
                MlxNative.MlxArray output = default;
                try
                {
                    xView = GetView(x);
                    gateView = GetView(gate);
                    if (!MlxCompiledOps.Disabled)
                    {
                        output = MlxCompiledOps.SigmoidMul(xView, gateView);
                    }
                    else
                    {
                        sigmoid = MlxNative.Unary(MlxNative.MlxUnaryOp.Sigmoid, gateView);
                        output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, xView, sigmoid);
                    }
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(xView);
                    MlxNative.FreeArray(gateView);
                    MlxNative.FreeArray(sigmoid);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("softmax", typeof(MlxStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("softmax", writeTarget, src);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    output = MlxNative.SoftmaxLastAxis(srcView);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("rope", typeof(MlxStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (seqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            long cols = LastDimension(src);
            if (cols <= 0 || src.ElementCount() % cols != 0 || src.ElementCount() / cols > int.MaxValue)
                return FallbackTensor("rope", result, src, seqLen, rowOffset);

            int rows = (int)(src.ElementCount() / cols);
            int[] positions = new int[rows];
            for (int row = 0; row < positions.Length; row++)
                positions[row] = row % seqLen + rowOffset;

            using Tensor positionTensor = Tensor.FromArray(src.Allocator, positions);
            return RoPEEx(result, src, positionTensor, (int)cols, 0, 0, 500000.0f, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f, false, false);
        }

        [RegisterOpStorageType("rope_ex", typeof(MlxStorage))]
        public static Tensor RoPEEx(
            Tensor result,
            Tensor src,
            Tensor positions,
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
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(src)
                || !IsMlxInt32(positions)
                || !src.IsContiguous()
                || !positions.IsContiguous()
                || extFactor != 0.0f
                || attnFactor != 1.0f
                || betaFast != 0.0f
                || betaSlow != 0.0f
                || ropeDim <= 0
                || (ropeDim & 1) != 0)
            {
                return FallbackTensor(
                    "rope_ex",
                    writeTarget,
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
                    addToResult,
                    invertPositions);
            }

            long cols = LastDimension(src);
            long rowsLong = src.ElementCount() / cols;
            if (cols <= 0
                || src.ElementCount() % cols != 0
                || cols > int.MaxValue
                || rowsLong > int.MaxValue
                || positions.ElementCount() != rowsLong
                || ropeDim > cols)
            {
                return FallbackTensor(
                    "rope_ex",
                    writeTarget,
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
                    addToResult,
                    invertPositions);
            }

            bool traditional = (mode & 2) == 0;
            int rows = (int)rowsLong;
            int features = (int)cols;
            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray positionsView = default;
                MlxNative.MlxArray flattened = default;
                MlxNative.MlxArray roped = default;
                MlxNative.MlxArray reshaped = default;
                MlxNative.MlxArray combined = default;
                try
                {
                    srcView = GetView(src);
                    positionsView = GetView(positions);
                    flattened = MlxNative.Reshape(srcView, new[] { rows, 1, 1, features });
                    roped = MlxNative.FastRopeDynamic(flattened, ropeDim, traditional, freqBase, freqScale, positionsView);
                    reshaped = MlxNative.Reshape(roped, ToIntArray(src.Sizes));

                    if (addToResult && result != null)
                    {
                        MlxNative.MlxArray resultView = default;
                        try
                        {
                            resultView = GetView(result);
                            combined = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, reshaped, resultView);
                            SetDeviceResult(writeTarget, combined);
                            combined = default;
                        }
                        finally
                        {
                            MlxNative.FreeArray(resultView);
                        }
                    }
                    else
                    {
                        SetDeviceResult(writeTarget, reshaped);
                        reshaped = default;
                    }
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(positionsView);
                    MlxNative.FreeArray(flattened);
                    MlxNative.FreeArray(roped);
                    MlxNative.FreeArray(reshaped);
                    MlxNative.FreeArray(combined);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("scaled_dot_product_attention", typeof(MlxStorage))]
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            long[] outputSizes = new[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, query.ElementType, false, outputSizes);
            if (!CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(query, key, value)
                || !AreOptionalFloat32(mask)
                || query.DimensionCount != 4
                || key.DimensionCount != 4
                || value.DimensionCount != 4
                || (mask != null && mask.DimensionCount != 4))
            {
                return FallbackTensor("scaled_dot_product_attention", writeTarget, query, key, value, mask, scale);
            }

            MlxNative.MlxArray queryView = default;
            MlxNative.MlxArray keyView = default;
            MlxNative.MlxArray valueView = default;
            MlxNative.MlxArray maskView = default;
            MlxNative.MlxArray queryHeadMajor = default;
            MlxNative.MlxArray keyHeadMajor = default;
            MlxNative.MlxArray valueHeadMajor = default;
            MlxNative.MlxArray attentionHeadMajor = default;
            MlxNative.MlxArray attentionSeqMajor = default;
            MlxNative.MlxArray contiguous = default;
            try
            {
                int[] headMajorAxes = { 0, 2, 1, 3 };
                queryView = GetView(query);
                keyView = GetView(key);
                valueView = GetView(value);
                maskView = GetOptionalView(mask);

                queryHeadMajor = MlxNative.Transpose(queryView, headMajorAxes);
                keyHeadMajor = MlxNative.Transpose(keyView, headMajorAxes);
                valueHeadMajor = MlxNative.Transpose(valueView, headMajorAxes);
                attentionHeadMajor = MlxNative.FastScaledDotProductAttention(
                    queryHeadMajor,
                    keyHeadMajor,
                    valueHeadMajor,
                    scale,
                    mask == null ? string.Empty : "array",
                    maskView);
                attentionSeqMajor = MlxNative.Transpose(attentionHeadMajor, headMajorAxes);
                contiguous = MlxNative.Contiguous(attentionSeqMajor);
                SetDeviceResult(writeTarget, contiguous);
                contiguous = default;
                return writeTarget;
            }
            finally
            {
                MlxNative.FreeArray(queryView);
                MlxNative.FreeArray(keyView);
                MlxNative.FreeArray(valueView);
                MlxNative.FreeArray(maskView);
                MlxNative.FreeArray(queryHeadMajor);
                MlxNative.FreeArray(keyHeadMajor);
                MlxNative.FreeArray(valueHeadMajor);
                MlxNative.FreeArray(attentionHeadMajor);
                MlxNative.FreeArray(attentionSeqMajor);
                MlxNative.FreeArray(contiguous);
            }
        }

        [RegisterOpStorageType("indexselect", typeof(MlxStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indices, bool isAdd)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, indices.Sizes[0], src.Sizes[1]);
            if (isAdd
                || !CanUseNativeWriteTarget(writeTarget)
                || !AreFloat32(src)
                || !IsMlxInt32(indices)
                || src.DimensionCount != 2
                || indices.DimensionCount != 1)
            {
                return FallbackTensor("indexselect", writeTarget, src, indices, isAdd);
            }

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray indicesView = default;
                MlxNative.MlxArray output = default;
                MlxNative.MlxArray contiguous = default;
                try
                {
                    srcView = GetView(src);
                    indicesView = GetView(indices);
                    output = MlxNative.TakeAxis(srcView, indicesView, 0);
                    contiguous = MlxNative.Contiguous(output);
                    SetDeviceResult(writeTarget, contiguous);
                    contiguous = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(indicesView);
                    MlxNative.FreeArray(output);
                    MlxNative.FreeArray(contiguous);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("repeat_interleave", typeof(MlxStorage))]
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (dim < 0 || dim >= src.DimensionCount)
                throw new ArgumentOutOfRangeException(nameof(dim));
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            long[] resultSizes = src.Sizes.ToArray();
            resultSizes[dim] *= repeats;
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src.Allocator, src.ElementType, false, resultSizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor("repeat_interleave", writeTarget, src, repeats, dim);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray output = default;
                MlxNative.MlxArray contiguous = default;
                try
                {
                    srcView = GetView(src);
                    output = MlxNative.RepeatAxis(srcView, repeats, dim);
                    contiguous = MlxNative.Contiguous(output);
                    SetDeviceResult(writeTarget, contiguous);
                    contiguous = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(output);
                    MlxNative.FreeArray(contiguous);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("layernorm", typeof(MlxStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src) || !AreOptionalFloat32(alpha, beta))
                return FallbackTensor("layernorm", writeTarget, src, alpha, beta, eps);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray alphaView = default;
                MlxNative.MlxArray betaView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    alphaView = GetOptionalView(alpha);
                    betaView = GetOptionalView(beta);
                    output = MlxNative.FastLayerNorm(srcView, alphaView, betaView, eps);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(alphaView);
                    MlxNative.FreeArray(betaView);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("rmsnorm", typeof(MlxStorage))]
        public static Tensor RmsNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (beta != null || !CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src) || !AreOptionalFloat32(alpha))
                return FallbackTensor("rmsnorm", writeTarget, src, alpha, beta, eps);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray alphaView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    alphaView = GetOptionalView(alpha);
                    output = MlxNative.FastRmsNorm(srcView, alphaView, eps);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(alphaView);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        [RegisterOpStorageType("add_causal_mask", typeof(MlxStorage))]
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            if (seqLen <= 0)
                throw new ArgumentOutOfRangeException(nameof(seqLen));

            long cols = LastDimension(tensor);
            long rowsLong = cols == 0 ? 0 : tensor.ElementCount() / cols;
            if (!CanUseNativeWriteTarget(tensor)
                || !tensor.IsContiguous()
                || cols <= 0
                || tensor.ElementCount() % cols != 0
                || cols > int.MaxValue
                || rowsLong > int.MaxValue)
            {
                FallbackVoid("add_causal_mask", tensor, seqLen, startPos, maskedValue);
                return;
            }

            int rows = (int)rowsLong;
            int keyLength = (int)cols;
            MlxNative.MlxArray tensorView = default;
            MlxNative.MlxArray scores2d = default;
            MlxNative.MlxArray rowRange = default;
            MlxNative.MlxArray row2d = default;
            MlxNative.MlxArray seqScalar = default;
            MlxNative.MlxArray rowInSequence = default;
            MlxNative.MlxArray startScalar = default;
            MlxNative.MlxArray threshold = default;
            MlxNative.MlxArray colRange = default;
            MlxNative.MlxArray col2d = default;
            MlxNative.MlxArray futureMask = default;
            MlxNative.MlxArray replacement = default;
            MlxNative.MlxArray addScalar = default;
            MlxNative.MlxArray maskedScores = default;
            MlxNative.MlxArray reshaped = default;
            try
            {
                tensorView = GetView(tensor);
                scores2d = MlxNative.Reshape(tensorView, new[] { rows, keyLength });
                rowRange = MlxNative.Arange(0, rows, 1, DType.Int32);
                row2d = MlxNative.Reshape(rowRange, new[] { rows, 1 });
                seqScalar = MlxNative.NewScalar(seqLen);
                rowInSequence = MlxNative.Remainder(row2d, seqScalar);
                startScalar = MlxNative.NewScalar(startPos);
                threshold = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, rowInSequence, startScalar);
                colRange = MlxNative.Arange(0, keyLength, 1, DType.Int32);
                col2d = MlxNative.Reshape(colRange, new[] { 1, keyLength });
                futureMask = MlxNative.Greater(col2d, threshold);

                if (float.IsNegativeInfinity(maskedValue))
                {
                    replacement = MlxNative.Full(new[] { rows, keyLength }, float.NegativeInfinity, DType.Float32);
                }
                else
                {
                    addScalar = MlxNative.NewScalar(maskedValue);
                    replacement = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, scores2d, addScalar);
                }

                maskedScores = MlxNative.Where(futureMask, replacement, scores2d);
                reshaped = MlxNative.Reshape(maskedScores, ToIntArray(tensor.Sizes));
                SetDeviceResult(tensor, reshaped);
                reshaped = default;
            }
            finally
            {
                MlxNative.FreeArray(tensorView);
                MlxNative.FreeArray(scores2d);
                MlxNative.FreeArray(rowRange);
                MlxNative.FreeArray(row2d);
                MlxNative.FreeArray(seqScalar);
                MlxNative.FreeArray(rowInSequence);
                MlxNative.FreeArray(startScalar);
                MlxNative.FreeArray(threshold);
                MlxNative.FreeArray(colRange);
                MlxNative.FreeArray(col2d);
                MlxNative.FreeArray(futureMask);
                MlxNative.FreeArray(replacement);
                MlxNative.FreeArray(addScalar);
                MlxNative.FreeArray(maskedScores);
                MlxNative.FreeArray(reshaped);
            }
        }

        private static Tensor Unary(string opName, Tensor result, Tensor src, MlxNative.MlxUnaryOp op)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(src))
                return FallbackTensor(opName, writeTarget, src);

            // Batch the entire sub-graph into one worker round-trip:
            // GetView + Unary + SetDeviceResult + 2 FreeArrays are
            // five separate queue hand-offs in the naive path. With a
            // single outer Invoke they all run inline on the worker
            // thread (IsOnWorkerThread short-circuits the queue).
            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray srcView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    srcView = GetView(src);
                    output = MlxNative.Unary(op, srcView);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(srcView);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        private static Tensor Binary(string opName, Tensor result, Tensor lhs, Tensor rhs, MlxNative.MlxBinaryOp op)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(lhs, rhs))
                return FallbackTensor(opName, writeTarget, lhs, rhs);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray lhsView = default;
                MlxNative.MlxArray rhsView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    lhsView = GetView(lhs);
                    rhsView = GetView(rhs);
                    output = MlxNative.Binary(op, lhsView, rhsView);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(lhsView);
                    MlxNative.FreeArray(rhsView);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        private static Tensor Scalar(string opName, Tensor result, Tensor tensor, float scalar, MlxNative.MlxBinaryOp op, bool scalarIsLhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, tensor, false, tensor.Sizes);
            if (!CanUseNativeWriteTarget(writeTarget) || !AreFloat32(tensor))
                return scalarIsLhs
                    ? FallbackTensor(opName, writeTarget, scalar, tensor)
                    : FallbackTensor(opName, writeTarget, tensor, scalar);

            MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray tensorView = default;
                MlxNative.MlxArray scalarArray = default;
                MlxNative.MlxArray output = default;
                try
                {
                    tensorView = GetView(tensor);
                    scalarArray = MlxNative.NewScalar(scalar);
                    output = scalarIsLhs
                        ? MlxNative.Binary(op, scalarArray, tensorView)
                        : MlxNative.Binary(op, tensorView, scalarArray);
                    SetDeviceResult(writeTarget, output);
                    output = default;
                }
                finally
                {
                    MlxNative.FreeArray(tensorView);
                    MlxNative.FreeArray(scalarArray);
                    MlxNative.FreeArray(output);
                }
            });
            return writeTarget;
        }

        private static MlxNative.MlxArray Gelu(MlxNative.MlxArray input)
        {
            MlxNative.MlxArray coeffCubic = default;
            MlxNative.MlxArray coeffInner = default;
            MlxNative.MlxArray one = default;
            MlxNative.MlxArray half = default;
            MlxNative.MlxArray squared = default;
            MlxNative.MlxArray cubed = default;
            MlxNative.MlxArray scaledCubic = default;
            MlxNative.MlxArray inner = default;
            MlxNative.MlxArray scaledInner = default;
            MlxNative.MlxArray tanh = default;
            MlxNative.MlxArray onePlusTanh = default;
            MlxNative.MlxArray halfInput = default;
            MlxNative.MlxArray output = default;
            try
            {
                coeffCubic = MlxNative.NewScalar(0.044715f);
                coeffInner = MlxNative.NewScalar(0.7978845608f);
                one = MlxNative.NewScalar(1.0f);
                half = MlxNative.NewScalar(0.5f);

                squared = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, input);
                cubed = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, squared, input);
                scaledCubic = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, cubed, coeffCubic);
                inner = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, input, scaledCubic);
                scaledInner = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, inner, coeffInner);
                tanh = MlxNative.Unary(MlxNative.MlxUnaryOp.Tanh, scaledInner);
                onePlusTanh = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, one, tanh);
                halfInput = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, input, half);
                output = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, halfInput, onePlusTanh);
                MlxNative.MlxArray result = output;
                output = default;
                return result;
            }
            finally
            {
                MlxNative.FreeArray(coeffCubic);
                MlxNative.FreeArray(coeffInner);
                MlxNative.FreeArray(one);
                MlxNative.FreeArray(half);
                MlxNative.FreeArray(squared);
                MlxNative.FreeArray(cubed);
                MlxNative.FreeArray(scaledCubic);
                MlxNative.FreeArray(inner);
                MlxNative.FreeArray(scaledInner);
                MlxNative.FreeArray(tanh);
                MlxNative.FreeArray(onePlusTanh);
                MlxNative.FreeArray(halfInput);
                MlxNative.FreeArray(output);
            }
        }

        private static bool CanUseNativeUnary(Tensor result, Tensor src)
        {
            return result != null &&
                src != null &&
                CanUseNativeWriteTarget(result) &&
                AreFloat32(src);
        }

        private static bool CanUseNativeCopy(Tensor result, Tensor src)
        {
            return result != null &&
                src != null &&
                result.Storage is MlxStorage &&
                src.Storage is MlxStorage &&
                result.IsContiguous();
        }

        private static bool CanUseNativeWriteTarget(Tensor tensor)
        {
            return tensor != null &&
                tensor.Storage is MlxStorage &&
                tensor.ElementType == DType.Float32 &&
                tensor.IsContiguous();
        }

        private static bool AreFloat32(params Tensor[] tensors)
        {
            foreach (Tensor tensor in tensors)
            {
                if (tensor == null || tensor.Storage is not MlxStorage || tensor.ElementType != DType.Float32)
                    return false;
            }

            return true;
        }

        private static bool AreOptionalFloat32(params Tensor[] tensors)
        {
            foreach (Tensor tensor in tensors)
            {
                if (tensor != null && (tensor.Storage is not MlxStorage || tensor.ElementType != DType.Float32))
                    return false;
            }

            return true;
        }

        private static bool IsMlxInt32(Tensor tensor)
        {
            return tensor != null && tensor.Storage is MlxStorage && tensor.ElementType == DType.Int32;
        }

        private static MlxNative.MlxArray GetView(Tensor tensor)
        {
            return ((MlxStorage)tensor.Storage).CreateArrayView(tensor);
        }

        private static MlxNative.MlxArray GetOptionalView(Tensor tensor)
        {
            return tensor == null ? default : GetView(tensor);
        }

        private static int[] ToIntArray(ReadOnlySpan<long> values)
        {
            int[] result = new int[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > int.MaxValue)
                    throw new NotSupportedException("MLX tensor dimensions larger than Int32.MaxValue are not supported yet.");
                result[i] = (int)values[i];
            }

            return result;
        }

        private static long LastDimension(Tensor tensor)
        {
            return tensor.DimensionCount == 0 ? 1 : tensor.Sizes[tensor.DimensionCount - 1];
        }

        private static void SetDeviceResult(Tensor tensor, MlxNative.MlxArray output)
        {
            MlxStorage storage = (MlxStorage)tensor.Storage;
            if (tensor.StorageOffset == 0 && tensor.Storage.ElementCount == tensor.ElementCount())
            {
                storage.ReplaceDeviceArray(output);
            }
            else
            {
                storage.UpdateDeviceSlice(tensor, output);
                MlxNative.FreeArray(output);
            }
        }

        private static Tensor FallbackTensor(string opName, params object[] args)
        {
            return (Tensor)MlxCpuFallback.Invoke(opName, MlxFallbackReturnKind.Tensor, new[] { 0 }, args);
        }

        private static void FallbackVoid(string opName, params object[] args)
        {
            MlxCpuFallback.Invoke(opName, MlxFallbackReturnKind.Void, new[] { 0 }, args);
        }
    }
}
