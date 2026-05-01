using System;
using TensorSharp.Core;

namespace TensorSharp.Cuda
{
    [OpsClass]
    public sealed class CudaBasicOps
    {
        [RegisterOpStorageType("fill", typeof(CudaStorage))]
        public static void Fill(Tensor result, float value)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));

            if (CudaKernelOps.TryFill(result, value))
                return;

            CudaCpuFallback.InvokeVoid("fill", result, result, value);
        }

        [RegisterOpStorageType("copy", typeof(CudaStorage))]
        public static void Copy(Tensor result, Tensor src)
        {
            if (CudaKernelOps.TryCopy(result, src))
                return;

            CudaCpuFallback.CopyLogical(result, src);
        }

        [RegisterOpStorageType("dot", typeof(CudaStorage))]
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor("dot", result, result, lhs, rhs);
        }

        [RegisterOpStorageType("addmm", typeof(CudaStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (CudaBlas.TryAddmm(writeTarget, beta, src, alpha, m1, m2))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmm", result, result, beta, src, alpha, m1, m2);
        }

        [RegisterOpStorageType("addmmbatch", typeof(CudaStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaBlas.TryAddmmBatch(writeTarget, beta, src, alpha, m1, m2))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmmbatch", result, result, beta, src, alpha, m1, m2);
        }

        [RegisterOpStorageType("mulmatid", typeof(CudaStorage))]
        public static Tensor MulmatID(Tensor result, Tensor expertWeights, Tensor input, Tensor ids)
        {
            return CudaCpuFallback.InvokeTensor("mulmatid", result, result, expertWeights, input, ids);
        }

        [RegisterOpStorageType("addid", typeof(CudaStorage))]
        public static Tensor AddID(Tensor result, Tensor src, Tensor bias, Tensor ids)
        {
            return CudaCpuFallback.InvokeTensor("addid", result, result, src, bias, ids);
        }

        [RegisterOpStorageType("abs", typeof(CudaStorage))]
        public static Tensor Abs(Tensor result, Tensor src) => Unary("abs", result, src);

        [RegisterOpStorageType("neg", typeof(CudaStorage))]
        public static Tensor Neg(Tensor result, Tensor src) => Unary("neg", result, src);

        [RegisterOpStorageType("sign", typeof(CudaStorage))]
        public static Tensor Sign(Tensor result, Tensor src) => Unary("sign", result, src);

        [RegisterOpStorageType("sqrt", typeof(CudaStorage))]
        public static Tensor Sqrt(Tensor result, Tensor src) => Unary("sqrt", result, src);

        [RegisterOpStorageType("rsqrt", typeof(CudaStorage))]
        public static Tensor Rsqrt(Tensor result, Tensor src) => Unary("rsqrt", result, src);

        [RegisterOpStorageType("exp", typeof(CudaStorage))]
        public static Tensor Exp(Tensor result, Tensor src) => Unary("exp", result, src);

        [RegisterOpStorageType("log", typeof(CudaStorage))]
        public static Tensor Log(Tensor result, Tensor src) => Unary("log", result, src);

        [RegisterOpStorageType("log1p", typeof(CudaStorage))]
        public static Tensor Log1p(Tensor result, Tensor src) => Unary("log1p", result, src);

        [RegisterOpStorageType("floor", typeof(CudaStorage))]
        public static Tensor Floor(Tensor result, Tensor src) => Unary("floor", result, src);

        [RegisterOpStorageType("ceil", typeof(CudaStorage))]
        public static Tensor Ceil(Tensor result, Tensor src) => Unary("ceil", result, src);

        [RegisterOpStorageType("round", typeof(CudaStorage))]
        public static Tensor Round(Tensor result, Tensor src) => Unary("round", result, src);

        [RegisterOpStorageType("trunc", typeof(CudaStorage))]
        public static Tensor Trunc(Tensor result, Tensor src) => Unary("trunc", result, src);

        [RegisterOpStorageType("frac", typeof(CudaStorage))]
        public static Tensor Frac(Tensor result, Tensor src) => Unary("frac", result, src);

        [RegisterOpStorageType("relu", typeof(CudaStorage))]
        public static Tensor Relu(Tensor result, Tensor src) => Unary("relu", result, src, CudaUnaryOp.Relu);

        [RegisterOpStorageType("sin", typeof(CudaStorage))]
        public static Tensor Sin(Tensor result, Tensor src) => Unary("sin", result, src);

        [RegisterOpStorageType("cos", typeof(CudaStorage))]
        public static Tensor Cos(Tensor result, Tensor src) => Unary("cos", result, src);

        [RegisterOpStorageType("tan", typeof(CudaStorage))]
        public static Tensor Tan(Tensor result, Tensor src) => Unary("tan", result, src);

        [RegisterOpStorageType("asin", typeof(CudaStorage))]
        public static Tensor Asin(Tensor result, Tensor src) => Unary("asin", result, src);

        [RegisterOpStorageType("acos", typeof(CudaStorage))]
        public static Tensor Acos(Tensor result, Tensor src) => Unary("acos", result, src);

        [RegisterOpStorageType("atan", typeof(CudaStorage))]
        public static Tensor Atan(Tensor result, Tensor src) => Unary("atan", result, src);

        [RegisterOpStorageType("sinh", typeof(CudaStorage))]
        public static Tensor Sinh(Tensor result, Tensor src) => Unary("sinh", result, src);

        [RegisterOpStorageType("cosh", typeof(CudaStorage))]
        public static Tensor Cosh(Tensor result, Tensor src) => Unary("cosh", result, src);

        [RegisterOpStorageType("tanh", typeof(CudaStorage))]
        public static Tensor Tanh(Tensor result, Tensor src) => Unary("tanh", result, src, CudaUnaryOp.Tanh);

        [RegisterOpStorageType("sigmoid", typeof(CudaStorage))]
        public static Tensor Sigmoid(Tensor result, Tensor src) => Unary("sigmoid", result, src, CudaUnaryOp.Sigmoid);

        [RegisterOpStorageType("SiLU", typeof(CudaStorage))]
        public static Tensor SiLU(Tensor result, Tensor src) => Unary("SiLU", result, src, CudaUnaryOp.SiLU);

        [RegisterOpStorageType("GELU", typeof(CudaStorage))]
        public static Tensor GELU(Tensor result, Tensor src) => Unary("GELU", result, src, CudaUnaryOp.GELU);

        [RegisterOpStorageType("addv", typeof(CudaStorage))]
        public static Tensor AddValue(Tensor result, Tensor lhs, float rhs) => Scalar("addv", result, lhs, rhs, CudaScalarOp.Add);

        [RegisterOpStorageType("subv", typeof(CudaStorage))]
        public static Tensor SubValue(Tensor result, Tensor lhs, float rhs) => Scalar("subv", result, lhs, rhs, CudaScalarOp.Sub);

        [RegisterOpStorageType("rsubv", typeof(CudaStorage))]
        public static Tensor RSubValue(Tensor result, float lhs, Tensor rhs)
        {
            return Scalar("rsubv", result, rhs, lhs, CudaScalarOp.ReverseSub);
        }

        [RegisterOpStorageType("mulv", typeof(CudaStorage))]
        public static Tensor MulValue(Tensor result, Tensor lhs, float rhs) => Scalar("mulv", result, lhs, rhs, CudaScalarOp.Mul);

        [RegisterOpStorageType("divv", typeof(CudaStorage))]
        public static Tensor DivValue(Tensor result, Tensor lhs, float rhs) => Scalar("divv", result, lhs, rhs, CudaScalarOp.Div);

        [RegisterOpStorageType("rdivv", typeof(CudaStorage))]
        public static Tensor RDivValue(Tensor result, float lhs, Tensor rhs)
        {
            return Scalar("rdivv", result, rhs, lhs, CudaScalarOp.ReverseDiv);
        }

        [RegisterOpStorageType("modv", typeof(CudaStorage))]
        public static Tensor ModValue(Tensor result, Tensor lhs, float rhs) => ScalarFallback("modv", result, lhs, rhs);

        [RegisterOpStorageType("addt", typeof(CudaStorage))]
        public static Tensor AddTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("addt", result, lhs, rhs, CudaBinaryOp.Add);

        [RegisterOpStorageType("subt", typeof(CudaStorage))]
        public static Tensor SubTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("subt", result, lhs, rhs, CudaBinaryOp.Sub);

        [RegisterOpStorageType("mult", typeof(CudaStorage))]
        public static Tensor MulTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("mult", result, lhs, rhs, CudaBinaryOp.Mul);

        [RegisterOpStorageType("divt", typeof(CudaStorage))]
        public static Tensor DivTensor(Tensor result, Tensor lhs, Tensor rhs) => Binary("divt", result, lhs, rhs, CudaBinaryOp.Div);

        [RegisterOpStorageType("modt", typeof(CudaStorage))]
        public static Tensor ModTensor(Tensor result, Tensor lhs, Tensor rhs) => BinaryFallback("modt", result, lhs, rhs);

        [RegisterOpStorageType("addmul", typeof(CudaStorage))]
        public static Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryTernary(writeTarget, x, y, z, CudaTernaryOp.AddMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmul", writeTarget, writeTarget, x, y, z);
        }

        [RegisterOpStorageType("adddiv", typeof(CudaStorage))]
        public static Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryTernary(writeTarget, x, y, z, CudaTernaryOp.AddDiv))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("adddiv", writeTarget, writeTarget, x, y, z);
        }

        [RegisterOpStorageType("addmulv", typeof(CudaStorage))]
        public static Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryAddMulScalar(writeTarget, x, y, z))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("addmulv", writeTarget, writeTarget, x, y, z);
        }

        [RegisterOpStorageType("mulmuladd", typeof(CudaStorage))]
        public static Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryMulMulAdd(writeTarget, x, y, z, w))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("mulmuladd", writeTarget, writeTarget, x, y, z, w);
        }

        [RegisterOpStorageType("SiLUMul", typeof(CudaStorage))]
        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, gate, up, CudaBinaryActivationOp.SiLUMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("SiLUMul", writeTarget, writeTarget, gate, up);
        }

        [RegisterOpStorageType("SiLUMulSplit", typeof(CudaStorage))]
        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gateUp.Allocator, DType.Float32, false, gateUp.Sizes[0], halfDim);
            if (CudaKernelOps.TrySiLUMulSplit(writeTarget, gateUp, halfDim))
                return writeTarget;

            using (Tensor gate = gateUp.Narrow(1, 0, halfDim))
            using (Tensor up = gateUp.Narrow(1, halfDim, halfDim))
            {
                Ops.Copy(writeTarget, gate);
                Ops.SiLUMul(writeTarget, writeTarget, up);
            }

            return writeTarget;
        }

        [RegisterOpStorageType("GELUMul", typeof(CudaStorage))]
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, gate, up, CudaBinaryActivationOp.GELUMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("GELUMul", writeTarget, writeTarget, gate, up);
        }

        [RegisterOpStorageType("SigmoidMul", typeof(CudaStorage))]
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            if (CudaKernelOps.TryBinaryActivation(writeTarget, x, gate, CudaBinaryActivationOp.SigmoidMul))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("SigmoidMul", writeTarget, writeTarget, x, gate);
        }

        [RegisterOpStorageType("sum", typeof(CudaStorage))]
        public static Tensor Sum(Tensor result, Tensor src, int dimension) => Reduce("sum", result, src, dimension);

        [RegisterOpStorageType("mean", typeof(CudaStorage))]
        public static Tensor Mean(Tensor result, Tensor src, int dimension) => Reduce("mean", result, src, dimension);

        [RegisterOpStorageType("prod", typeof(CudaStorage))]
        public static Tensor Prod(Tensor result, Tensor src, int dimension) => Reduce("prod", result, src, dimension);

        [RegisterOpStorageType("min", typeof(CudaStorage))]
        public static Tensor Min(Tensor result, Tensor src, int dimension) => Reduce("min", result, src, dimension);

        [RegisterOpStorageType("max", typeof(CudaStorage))]
        public static Tensor Max(Tensor result, Tensor src, int dimension) => Reduce("max", result, src, dimension);

        [RegisterOpStorageType("argmin", typeof(CudaStorage))]
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) => Reduce("argmin", result, src, dimension);

        [RegisterOpStorageType("argmax", typeof(CudaStorage))]
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) => Reduce("argmax", result, src, dimension);

        [RegisterOpStorageType("softmax", typeof(CudaStorage))]
        public static Tensor Softmax(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TrySoftmax(writeTarget, src))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("softmax", writeTarget, writeTarget, src);
        }

        [RegisterOpStorageType("softmaxgrad", typeof(CudaStorage))]
        public static Tensor SoftmaxGrad(Tensor grad, Tensor adj, Tensor val, bool addGrad)
        {
            return CudaCpuFallback.InvokeTensor("softmaxgrad", grad, grad, adj, val, addGrad);
        }

        [RegisterOpStorageType("scaled_dot_product_attention", typeof(CudaStorage))]
        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            long[] outputSizes = new long[] { query.Sizes[0], query.Sizes[1], query.Sizes[2], value.Sizes[3] };
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, query.Allocator, query.ElementType, false, outputSizes);
            if (CudaKernelOps.TryScaledDotProductAttention(writeTarget, query, key, value, mask, scale))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("scaled_dot_product_attention", writeTarget, writeTarget, query, key, value, mask, scale);
        }

        [RegisterOpStorageType("indexselect", typeof(CudaStorage))]
        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, indice.Sizes[0], src.Sizes[1]);
            if (CudaKernelOps.TryIndexSelect(writeTarget, src, indice, isAdd))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("indexselect", writeTarget, writeTarget, src, indice, isAdd);
        }

        [RegisterOpStorageType("indexselectgrad", typeof(CudaStorage))]
        public static Tensor IndexSelectGrad(Tensor grad, Tensor adj, Tensor indice)
        {
            return CudaCpuFallback.InvokeTensor("indexselectgrad", grad, grad, adj, indice);
        }

        [RegisterOpStorageType("repeat_interleave", typeof(CudaStorage))]
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            return CudaCpuFallback.InvokeTensor("repeat_interleave", result, result, src, repeats, dim);
        }

        [RegisterOpStorageType("add_causal_mask", typeof(CudaStorage))]
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (CudaKernelOps.TryAddCausalMask(tensor, seqLen, startPos, maskedValue))
                return;

            CudaCpuFallback.InvokeVoid("add_causal_mask", tensor, tensor, seqLen, startPos, maskedValue);
        }

        [RegisterOpStorageType("topK", typeof(CudaStorage))]
        public static Tensor TopK(Tensor outVal, Tensor outIdx, Tensor inVal, int k)
        {
            return CudaCpuFallback.InvokeTensor("topK", outVal, outVal, outIdx, inVal, k);
        }

        [RegisterOpStorageType("rope", typeof(CudaStorage))]
        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TryRoPE(writeTarget, src, seqLen, rowOffset))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("rope", writeTarget, writeTarget, src, seqLen, rowOffset);
        }

        [RegisterOpStorageType("ropegrad", typeof(CudaStorage))]
        public static Tensor RoPEGrad(Tensor grad, Tensor adj, int seqLen, int rowOffset)
        {
            return CudaCpuFallback.InvokeTensor("ropegrad", grad, grad, adj, seqLen, rowOffset);
        }

        [RegisterOpStorageType("rope_ex", typeof(CudaStorage))]
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
            if (CudaKernelOps.TryRoPEEx(
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
                addToResult))
            {
                return writeTarget;
            }

            return CudaCpuFallback.InvokeTensor(
                "rope_ex",
                writeTarget,
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

        [RegisterOpStorageType("layernorm", typeof(CudaStorage))]
        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            return CudaCpuFallback.InvokeTensor("layernorm", result, result, src, alpha, beta, eps);
        }

        [RegisterOpStorageType("rmsnorm", typeof(CudaStorage))]
        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            if (CudaKernelOps.TryRMSNorm(writeTarget, src, alpha, beta, eps))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor("rmsnorm", writeTarget, writeTarget, src, alpha, beta, eps);
        }

        [RegisterOpStorageType("addlayernorm", typeof(CudaStorage))]
        public static Tensor AddLayerNorm(Tensor result, Tensor src1, Tensor src2, Tensor alpha, Tensor beta, float eps)
        {
            return CudaCpuFallback.InvokeTensor("addlayernorm", result, result, src1, src2, alpha, beta, eps);
        }

        [RegisterOpStorageType("gather", typeof(CudaStorage))]
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("gather", result, result, src, dim, indices);
        }

        [RegisterOpStorageType("scatter", typeof(CudaStorage))]
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter", result, result, src, dim, indices);
        }

        [RegisterOpStorageType("scatter_add", typeof(CudaStorage))]
        public static Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter_add", result, result, src, dim, indices);
        }

        [RegisterOpStorageType("scatter_fill", typeof(CudaStorage))]
        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices)
        {
            return CudaCpuFallback.InvokeTensor("scatter_fill", result, result, value, dim, indices);
        }

        [RegisterOpStorageType("float2half", typeof(CudaStorage))]
        public static Tensor Float2Half(Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor("float2half", result, result, src);
        }

        [RegisterOpStorageType("half2float", typeof(CudaStorage))]
        public static Tensor Half2Float(Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor("half2float", result, result, src);
        }

        [RegisterOpStorageType("atomicadd", typeof(CudaStorage))]
        public static Tensor AtomicAdd(Tensor result, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor("atomicadd", result, result, rhs);
        }

        private static Tensor Unary(string opName, Tensor result, Tensor src)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, src);
        }

        private static Tensor Unary(string opName, Tensor result, Tensor src, CudaUnaryOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            if (CudaKernelOps.TryUnary(writeTarget, src, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, src);
        }

        private static Tensor Binary(string opName, Tensor result, Tensor lhs, Tensor rhs, CudaBinaryOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (CudaKernelOps.TryBinary(writeTarget, lhs, rhs, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, lhs, rhs);
        }

        private static Tensor Scalar(string opName, Tensor result, Tensor lhs, float rhs, CudaScalarOp cudaOp)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            if (CudaKernelOps.TryScalar(writeTarget, lhs, rhs, cudaOp))
                return writeTarget;

            return CudaCpuFallback.InvokeTensor(opName, writeTarget, writeTarget, lhs, rhs);
        }

        private static Tensor BinaryFallback(string opName, Tensor result, Tensor lhs, Tensor rhs)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, lhs, rhs);
        }

        private static Tensor ScalarFallback(string opName, Tensor result, Tensor lhs, float rhs)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, lhs, rhs);
        }

        private static Tensor Reduce(string opName, Tensor result, Tensor src, int dimension)
        {
            return CudaCpuFallback.InvokeTensor(opName, result, result, src, dimension);
        }
    }
}
