// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/Seq2SeqSharp
//
// This file is part of Seq2SeqSharp.
//
// Seq2SeqSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Seq2SeqSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using AdvUtils;
using System;
using System.Data;
using System.Reflection;
using TensorSharp.Core;

namespace TensorSharp.Cpu
{
    [OpsClass]
    public class CpuBasicOps
    {
        public CpuBasicOps()
        {
        }


        [RegisterOpStorageType("dot", typeof(CpuStorage))]
        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs)
        {
            if (lhs.DimensionCount == 1 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Dot(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 1)
            {
                return MatrixMultiplication.Mul_M_V(result, lhs, rhs);
            }
            else if (lhs.DimensionCount == 2 && rhs.DimensionCount == 2)
            {
                return MatrixMultiplication.Mul_M_M(result, lhs, rhs);
            }
            else
            {
                throw new NotSupportedException(message: string.Format("Multiplication of {0}D with {1}D tensor is not supported"));
            }
        }

        [RegisterOpStorageType("addmm", typeof(CpuStorage))]
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            try
            {
                if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
                {
                    throw new InvalidOperationException($"All tensors must have the same element type  src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result.ElementType}'");
                }

                if (result != null && !(result.Storage is CpuStorage))
                {
                    throw new ArgumentException("result must be a CPU tensor", nameof(result));
                }

                if (!(m1.Storage is CpuStorage))
                {
                    throw new ArgumentException("m1 must be a CPU tensor", nameof(m1));
                }

                if (!(m2.Storage is CpuStorage))
                {
                    throw new ArgumentException("m2 must be a CPU tensor", nameof(m2));
                }

                if (src.DimensionCount != 2)
                {
                    throw new ArgumentException("src must be a matrix", nameof(src));
                }

                if (m1.DimensionCount != 2)
                {
                    throw new ArgumentException("m1 must be a matrix", nameof(m1));
                }

                if (m2.DimensionCount != 2)
                {
                    throw new ArgumentException("m2 must be a matrix", nameof(m2));
                }

                if (src.Sizes[0] != m1.Sizes[0] || src.Sizes[1] != m2.Sizes[1] || m1.Sizes[1] != m2.Sizes[0])
                {
                    throw new InvalidOperationException("Size mismatch");
                }

                Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);

                if (writeTarget != src)
                {
                    Ops.Copy(writeTarget, src);
                }


                MatrixMultiplication.Gemm(alpha, m1, m2, beta, writeTarget);


                return writeTarget;
            }
            catch (Exception err)
            {
                Logger.WriteLine(Logger.Level.err, $"Exception = '{err.Message}'.");
                Logger.WriteLine(Logger.Level.debug, $"Call stack = '{err.StackTrace}'");

                throw;
            }
        }

        [RegisterOpStorageType("addmmbatch", typeof(CpuStorage))]
        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2)
        {
            if (src.ElementType != m1.ElementType || src.ElementType != m2.ElementType || (result != null && result.ElementType != src.ElementType))
            {
                throw new InvalidOperationException($"All tensors must have the same element type  src = '{src.ElementType}', m1 = '{m1.ElementType}', m2 = '{m2.ElementType}' result = '{result.ElementType}'");
            }

            if (result != null && !(result.Storage is CpuStorage))
            {
                throw new ArgumentException("result must be a CPU tensor", nameof(result));
            }

            if (!(m1.Storage is CpuStorage))
            {
                throw new ArgumentException("m1 must be a CPU tensor", nameof(m1));
            }

            if (!(m2.Storage is CpuStorage))
            {
                throw new ArgumentException("m2 must be a CPU tensor", nameof(m2));
            }

            if (src.DimensionCount != 3)
            {
                throw new ArgumentException("src must be a 3D batch tensor [batch, M, N]", nameof(src));
            }

            if (m1.DimensionCount != 3)
            {
                throw new ArgumentException("m1 must be a 3D batch tensor [batch, M, K]", nameof(m1));
            }

            if (m2.DimensionCount != 3)
            {
                throw new ArgumentException("m2 must be a 3D batch tensor [batch, K, N]", nameof(m2));
            }

            if (src.Sizes[1] != m1.Sizes[1] || src.Sizes[2] != m2.Sizes[2] || m1.Sizes[2] != m2.Sizes[1])
            {
                throw new InvalidOperationException($"Size mismatch, srcSize0 = {src.Sizes[0]}, m1Size0 = {m1.Sizes[0]}, srcSize1 = {src.Sizes[1]}, m2Size1 = {m2.Sizes[1]}, m1Size1 = '{m1.Sizes[1]}', m2Size0 = '{m2.Sizes[0]}'");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);

            if (writeTarget != src)
            {
                Ops.Copy(writeTarget, src);
            }

            MatrixMultiplication.GemmBatch(alpha, m1, m2, beta, writeTarget);

            return writeTarget;
        }




        [RegisterOpStorageType("abs", typeof(CpuStorage))]
        public Tensor Abs(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Abs(writeTarget, src);
            return writeTarget;
        }

        [RegisterOpStorageType("neg", typeof(CpuStorage))]
        public Tensor Neg(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Neg(writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo sign_func = NativeWrapper.GetMethod("TS_Sign");
        [RegisterOpStorageType("sign", typeof(CpuStorage))]
        public Tensor Sign(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sign_func, result, src); }


        private readonly MethodInfo sqrt_func = NativeWrapper.GetMethod("TS_Sqrt");

        [RegisterOpStorageType("exp", typeof(CpuStorage))]
        public Tensor Exp(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Exp(writeTarget, src);

            return writeTarget;
        }

        [RegisterOpStorageType("log", typeof(CpuStorage))]
        public Tensor Log(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Log(writeTarget, src);

            return writeTarget;
        }

        private readonly MethodInfo log1p_func = NativeWrapper.GetMethod("TS_Log1p");
        [RegisterOpStorageType("log1p", typeof(CpuStorage))]
        public Tensor Log1p(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(log1p_func, result, src); }

        private readonly MethodInfo floor_func = NativeWrapper.GetMethod("TS_Floor");
        [RegisterOpStorageType("floor", typeof(CpuStorage))]
        public Tensor Floor(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(floor_func, result, src); }

        private readonly MethodInfo ceil_func = NativeWrapper.GetMethod("TS_Ceil");
        [RegisterOpStorageType("ceil", typeof(CpuStorage))]
        public Tensor Ceil(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(ceil_func, result, src); }

        private readonly MethodInfo round_func = NativeWrapper.GetMethod("TS_Round");
        [RegisterOpStorageType("round", typeof(CpuStorage))]
        public Tensor Round(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(round_func, result, src); }

        private readonly MethodInfo trunc_func = NativeWrapper.GetMethod("TS_Trunc");
        [RegisterOpStorageType("trunc", typeof(CpuStorage))]
        public Tensor Trunc(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(trunc_func, result, src); }

        private readonly MethodInfo frac_func = NativeWrapper.GetMethod("TS_Frac");
        [RegisterOpStorageType("frac", typeof(CpuStorage))]
        public Tensor Frac(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(frac_func, result, src); }


        [RegisterOpStorageType("relu", typeof(CpuStorage))]
        public Tensor Relu(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Relu(writeTarget, src);

            return writeTarget;
        }


        private readonly MethodInfo sin_func = NativeWrapper.GetMethod("TS_Sin");
        private readonly MethodInfo cos_func = NativeWrapper.GetMethod("TS_Cos");
        [RegisterOpStorageType("cos", typeof(CpuStorage))]
        public Tensor Cos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cos_func, result, src); }

        private readonly MethodInfo tan_func = NativeWrapper.GetMethod("TS_Tan");
        [RegisterOpStorageType("tan", typeof(CpuStorage))]
        public Tensor Tan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tan_func, result, src); }


        private readonly MethodInfo asin_func = NativeWrapper.GetMethod("TS_Asin");
        [RegisterOpStorageType("asin", typeof(CpuStorage))]
        public Tensor Asin(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(asin_func, result, src); }

        private readonly MethodInfo acos_func = NativeWrapper.GetMethod("TS_Acos");
        [RegisterOpStorageType("acos", typeof(CpuStorage))]
        public Tensor Acos(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(acos_func, result, src); }

        private readonly MethodInfo atan_func = NativeWrapper.GetMethod("TS_Atan");
        [RegisterOpStorageType("atan", typeof(CpuStorage))]
        public Tensor Atan(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(atan_func, result, src); }


        private readonly MethodInfo sinh_func = NativeWrapper.GetMethod("TS_Sinh");
        [RegisterOpStorageType("sinh", typeof(CpuStorage))]
        public Tensor Sinh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(sinh_func, result, src); }

        private readonly MethodInfo cosh_func = NativeWrapper.GetMethod("TS_Cosh");
        [RegisterOpStorageType("cosh", typeof(CpuStorage))]
        public Tensor Cosh(Tensor result, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(cosh_func, result, src); }

        [RegisterOpStorageType("tanh", typeof(CpuStorage))]
        public Tensor Tanh(Tensor result, Tensor src) 
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Tanh(writeTarget, src);

            return writeTarget;
        }


        [RegisterOpStorageType("sigmoid", typeof(CpuStorage))]
        public Tensor Sigmoid(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Sigmoid(writeTarget, src);

            return writeTarget;
        }





        private readonly MethodInfo add3_func = NativeWrapper.GetMethod("TS_Add3");
        private readonly MethodInfo add4_func = NativeWrapper.GetMethod("TS_Add4");

        [RegisterOpStorageType("addmul", typeof(CpuStorage))]
        public Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddMul(writeTarget, x, y, z);

            return writeTarget;
        }

        [RegisterOpStorageType("adddiv", typeof(CpuStorage))]
        public Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddDiv(writeTarget, x, y, z);

            return writeTarget;
        }






        [RegisterOpStorageType("addmulv", typeof(CpuStorage))]
        public Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.AddMulV(writeTarget, x, y, z);
            return writeTarget;

        }

        private readonly MethodInfo maskfill_func = NativeWrapper.GetMethod("TS_MaskFill");

        private readonly MethodInfo atan2_func = NativeWrapper.GetMethod("TS_Atan2");
        [RegisterOpStorageType("atan2", typeof(CpuStorage))]
        public Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return NativeWrapper.InvokeNullableResultElementwise(atan2_func, result, srcY, srcX); }


        [RegisterOpStorageType("pow", typeof(CpuStorage))]
        public Tensor Pow(Tensor result, Tensor src, float value)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Pow(writeTarget, src, value);

            return writeTarget;

        }

        private readonly MethodInfo tpow_func = NativeWrapper.GetMethod("TS_Tpow");
        [RegisterOpStorageType("tpow", typeof(CpuStorage))]
        public Tensor Tpow(Tensor result, float value, Tensor src) { return NativeWrapper.InvokeNullableResultElementwise(tpow_func, result, value, src); }

        private readonly MethodInfo lerp_func = NativeWrapper.GetMethod("TS_Lerp");
        [RegisterOpStorageType("lerp", typeof(CpuStorage))]
        public Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return NativeWrapper.InvokeNullableResultElementwise(lerp_func, result, srcA, srcB, weight); }

        [RegisterOpStorageType("clamp", typeof(CpuStorage))]
        public Tensor Clamp(Tensor result, Tensor src, float min, float max)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.Clamp(writeTarget, src, min, max);

            return writeTarget;
        }


        [RegisterOpStorageType("mulmuladd", typeof(CpuStorage))]
        public Tensor MulMulAdd(Tensor result, Tensor srcX, Tensor srcY, Tensor srcZ, Tensor srcW)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcX, false, srcX.Sizes);
            TensorApplyCPU.MulMulAdd(writeTarget, srcX, srcY, srcZ, srcW);

            return writeTarget;
        }



        private readonly MethodInfo addtanh3_func = NativeWrapper.GetMethod("TS_AddTanh3");

        [RegisterOpStorageType("SiLU", typeof(CpuStorage))]
        public Tensor SiLU(Tensor result, Tensor srcW)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, srcW, false, srcW.Sizes);
            TensorApplyCPU.SiLU(writeTarget, srcW);
            return writeTarget;
        }

        [RegisterOpStorageType("GELU", typeof(CpuStorage))]
        public Tensor GELU(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, src.Sizes);
            TensorApplyCPU.GELU(writeTarget, src);
            return writeTarget;
        }

        [RegisterOpStorageType("GELUMul", typeof(CpuStorage))]
        public Tensor GELUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            TensorApplyCPU.GELUMul(writeTarget, gate, up);
            return writeTarget;
        }

        [RegisterOpStorageType("SiLUMul", typeof(CpuStorage))]
        public Tensor SiLUMul(Tensor result, Tensor gate, Tensor up)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            TensorApplyCPU.SiLUMul(writeTarget, gate, up);
            return writeTarget;
        }

        [RegisterOpStorageType("SiLUMulClamp", typeof(CpuStorage))]
        public Tensor SiLUMulClamp(Tensor result, Tensor gate, Tensor up, float limit)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, gate, false, gate.Sizes);
            TensorApplyCPU.SiLUMulClamp(writeTarget, gate, up, limit);
            return writeTarget;
        }

        [RegisterOpStorageType("SigmoidMul", typeof(CpuStorage))]
        public Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, x, false, x.Sizes);
            TensorApplyCPU.SigmoidMul(writeTarget, x, gate);
            return writeTarget;
        }




        [RegisterOpStorageType("addv", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, float rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Add(writeTarget, lhs, rhs);

            return writeTarget;
        }

        [RegisterOpStorageType("subv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, float rhs)
        {
            // x - r and x + (-r) are the same IEEE operation.
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Add(writeTarget, lhs, -rhs);
            return writeTarget;
        }


        [RegisterOpStorageType("rsubv", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, float lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, rhs, false, rhs.Sizes);
            TensorApplyCPU.RSub(writeTarget, lhs, rhs);

            return writeTarget;
        }


        [RegisterOpStorageType("mulv", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, float rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Mul(writeTarget, lhs, rhs);

            return writeTarget;
        }

        [RegisterOpStorageType("divv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, float rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Div(writeTarget, lhs, rhs);

            return writeTarget;
        }

        [RegisterOpStorageType("rdivv", typeof(CpuStorage))]
        public Tensor Div(Tensor result, float lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, rhs, false, rhs.Sizes);
            TensorApplyCPU.RDiv(writeTarget, lhs, rhs);
            return writeTarget;
        }

        private readonly MethodInfo mod_func = NativeWrapper.GetMethod("TS_Mod");

        private readonly MethodInfo gtValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("gtValue", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtValue_func, result, lhs, rhs); }

        private readonly MethodInfo ltValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("ltValue", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltValue_func, result, lhs, rhs); }

        private readonly MethodInfo geValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("geValue", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(geValue_func, result, lhs, rhs); }

        private readonly MethodInfo leValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("leValue", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(leValue_func, result, lhs, rhs); }

        private readonly MethodInfo eqValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("eqValue", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqValue_func, result, lhs, rhs); }

        private readonly MethodInfo neValue_func = NativeWrapper.GetMethod("TS_gtValue");
        [RegisterOpStorageType("neValue", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return NativeWrapper.InvokeNullableResultElementwise(neValue_func, result, lhs, rhs); }


        [RegisterOpStorageType("addt", typeof(CpuStorage))]
        public Tensor Add(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Add(writeTarget, lhs, rhs);

            return writeTarget;
        }

        [RegisterOpStorageType("subt", typeof(CpuStorage))]
        public Tensor Sub(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Sub(writeTarget, lhs, rhs);
            return writeTarget;
        }

        [RegisterOpStorageType("mult", typeof(CpuStorage))]
        public Tensor Mul(Tensor result, Tensor lhs, Tensor rhs)        
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Mul(writeTarget, lhs, rhs);
            return writeTarget;
        }

        [RegisterOpStorageType("divt", typeof(CpuStorage))]
        public Tensor Div(Tensor result, Tensor lhs, Tensor rhs)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, lhs, false, lhs.Sizes);
            TensorApplyCPU.Div(writeTarget, lhs, rhs);

            return writeTarget;
        }

        private readonly MethodInfo cmod_func = NativeWrapper.GetMethod("TS_CMod");

        private readonly MethodInfo gtTensor_func = NativeWrapper.GetMethod("TS_gtTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(gtTensor_func, result, lhs, rhs); }

        private readonly MethodInfo ltTensor_func = NativeWrapper.GetMethod("TS_ltTensor");
        [RegisterOpStorageType("gtTensor", typeof(CpuStorage))]
        public Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(ltTensor_func, result, lhs, rhs); }

        private readonly MethodInfo geTensor_func = NativeWrapper.GetMethod("TS_geTensor");
        [RegisterOpStorageType("geTensor", typeof(CpuStorage))]
        public Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(geTensor_func, result, lhs, rhs); }

        private readonly MethodInfo leTensor_func = NativeWrapper.GetMethod("TS_leTensor");
        [RegisterOpStorageType("leTensor", typeof(CpuStorage))]
        public Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(leTensor_func, result, lhs, rhs); }

        private readonly MethodInfo eqTensor_func = NativeWrapper.GetMethod("TS_eqTensor");
        [RegisterOpStorageType("eqTensor", typeof(CpuStorage))]
        public Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(eqTensor_func, result, lhs, rhs); }

        private readonly MethodInfo neTensor_func = NativeWrapper.GetMethod("TS_neTensor");
        [RegisterOpStorageType("neTensor", typeof(CpuStorage))]
        public Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return NativeWrapper.InvokeNullableResultElementwise(neTensor_func, result, lhs, rhs); }


        [RegisterOpStorageType("sum", typeof(CpuStorage))]
        public Tensor Sum(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Sum(writeTarget, src, dimension);

            return writeTarget;
        }

        [RegisterOpStorageType("mean", typeof(CpuStorage))]
        public Tensor Mean(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Mean(writeTarget, src, dimension);

            return writeTarget;
        }


        private readonly MethodInfo prod_func = NativeWrapper.GetMethod("TS_Prod");
        [RegisterOpStorageType("prod", typeof(CpuStorage))]
        public Tensor Prod(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(prod_func, result, src, dimension); }

        private readonly MethodInfo min_func = NativeWrapper.GetMethod("TS_Min");
        [RegisterOpStorageType("min", typeof(CpuStorage))]
        public Tensor Min(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(min_func, result, src, dimension); }


        [RegisterOpStorageType("max", typeof(CpuStorage))]
        public Tensor Max(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Max(writeTarget, src, dimension);

            return writeTarget;
        }




        private readonly MethodInfo argmin_func = NativeWrapper.GetMethod("TS_Argmin");
        [RegisterOpStorageType("argmin", typeof(CpuStorage))]
        public Tensor Argmin(Tensor result, Tensor src, int dimension) { return NativeWrapper.InvokeNullableResultDimensionwise(argmin_func, result, src, dimension); }


        [RegisterOpStorageType("argmax", typeof(CpuStorage))]
        public Tensor Argmax(Tensor result, Tensor src, int dimension)
        {
            Tensor writeTarget = NativeWrapper.CreateResultDimensionwise(result, src, dimension);
            TensorApplyCPU.Argmax(writeTarget, src, dimension);

            return writeTarget;

        }

        private readonly MethodInfo norm_func = NativeWrapper.GetMethod("TS_Norm");
        [RegisterOpStorageType("norm", typeof(CpuStorage))]
        public Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return NativeWrapper.InvokeNullableResultDimensionwise(norm_func, result, src, dimension, value); }

        private readonly MethodInfo std_func = NativeWrapper.GetMethod("TS_Std");
        [RegisterOpStorageType("std", typeof(CpuStorage))]
        public Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(std_func, result, src, dimension, normByN); }

        private readonly MethodInfo var_func = NativeWrapper.GetMethod("TS_Var");
        [RegisterOpStorageType("var", typeof(CpuStorage))]
        public Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return NativeWrapper.InvokeNullableResultDimensionwise(var_func, result, src, dimension, normByN); }



        private readonly MethodInfo sumall_func = NativeWrapper.GetMethod("TS_SumAll");
        [RegisterOpStorageType("sumall", typeof(CpuStorage))]
        public Tensor SumAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(sumall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo prodall_func = NativeWrapper.GetMethod("TS_ProdAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor ProdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(prodall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo minall_func = NativeWrapper.GetMethod("TS_MinAll");
        [RegisterOpStorageType("prodall", typeof(CpuStorage))]
        public Tensor MinAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(minall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo maxall_func = NativeWrapper.GetMethod("TS_MaxAll");
        [RegisterOpStorageType("maxall", typeof(CpuStorage))]
        public Tensor MaxAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(maxall_func, writeTarget, src);
            return writeTarget;
        }


        private readonly MethodInfo meanall_func = NativeWrapper.GetMethod("TS_MeanAll");
        [RegisterOpStorageType("meanall", typeof(CpuStorage))]
        public Tensor MeanAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(meanall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo varall_func = NativeWrapper.GetMethod("TS_VarAll");
        [RegisterOpStorageType("varall", typeof(CpuStorage))]
        public Tensor VarAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(varall_func, writeTarget, src);
            return writeTarget;
        }

        private readonly MethodInfo stdall_func = NativeWrapper.GetMethod("TS_StdAll");
        [RegisterOpStorageType("stdall", typeof(CpuStorage))]
        public Tensor StdAll(Tensor result, Tensor src)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(stdall_func, writeTarget, src);
            return writeTarget;
        }



        [RegisterOpStorageType("layernorm", typeof(CpuStorage))]
        public Tensor LayerNorm(Tensor result, Tensor src, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.LayerNorm(writeTarget, src, gamma_, beta_, eps, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        [RegisterOpStorageType("rmsnorm", typeof(CpuStorage))]
        public Tensor RMSNorm(Tensor result, Tensor src, Tensor gamma_, Tensor beta_, float eps)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RMSNorm(writeTarget, src, gamma_, beta_, eps, (int)src.Sizes[0], (int)src.Sizes[1]);
            return writeTarget;
        }

        private readonly MethodInfo addlayerNorm_func = NativeWrapper.GetMethod("TS_AddLayerNorm");
        private readonly MethodInfo addlayerNormGrad_func = NativeWrapper.GetMethod("TS_AddLayerNormGrad");
        [RegisterOpStorageType("indexselect", typeof(CpuStorage))]
        public Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd)
        {
            // The kernel copies (and adds as) 4-byte floats: a narrower element would be read and written at the
            // wrong width (an F16 source wrote twice its result's bytes).
            if (src.ElementType.Size() != sizeof(float) || (isAdd && src.ElementType != DType.Float32)
                || (result != null && result.ElementType != src.ElementType))
                throw new NotSupportedException($"index_select over {src.ElementType}{(isAdd ? " (add)" : "")}: 4-byte elements only.");
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, new long[] { indice.Sizes[0], src.Sizes[1] });

            int ndim = writeTarget.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(writeTarget.Sizes, writeTarget.Strides);
            long cols = writeTarget.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            TensorApplyCPU.IndexSelect(writeTarget, src, indice, (int)rows, (int)cols, isAdd);
            return writeTarget;
        }


        [RegisterOpStorageType("repeat_interleave", typeof(CpuStorage))]
        public Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim)
        {
            if (dim < 0 || dim >= src.DimensionCount)
                throw new ArgumentOutOfRangeException(nameof(dim));
            if (repeats < 1)
                throw new ArgumentOutOfRangeException(nameof(repeats));

            long[] resultSizes = src.Sizes.ToArray();
            resultSizes[dim] *= repeats;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, resultSizes);

            using var contiguousSrc = Ops.AsContiguous(src);

            int sliceCount = 1;
            for (int d = 0; d < dim; d++)
                sliceCount *= (int)contiguousSrc.Sizes[d];

            int innerSize = 1;
            for (int d = dim + 1; d < contiguousSrc.DimensionCount; d++)
                innerSize *= (int)contiguousSrc.Sizes[d];

            int dimSize = (int)contiguousSrc.Sizes[dim];
            int sliceSize = innerSize;

            unsafe
            {
                float* srcPtr = (float*)CpuNativeHelpers.GetBufferStart(contiguousSrc);
                float* dstPtr = (float*)CpuNativeHelpers.GetBufferStart(writeTarget);

                for (int outer = 0; outer < sliceCount; outer++)
                {
                    float* srcBatch = srcPtr + outer * dimSize * sliceSize;
                    float* dstBatch = dstPtr + outer * dimSize * repeats * sliceSize;
                    TensorApplyCPU.RepeatInterleave(dstBatch, srcBatch, dimSize, repeats, sliceSize);
                }
            }

            return writeTarget;
        }

        [RegisterOpStorageType("add_causal_mask", typeof(CpuStorage))]
        public void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue)
        {
            if (!tensor.IsContiguous())
                throw new InvalidOperationException("AddCausalMask requires a contiguous tensor.");

            int ndim = tensor.DimensionCount;
            long cols = tensor.Sizes[ndim - 1];
            long totalElements = tensor.ElementCount();
            long totalRows = totalElements / cols;

            unsafe
            {
                float* ptr = (float*)CpuNativeHelpers.GetBufferStart(tensor);
                TensorApplyCPU.AddCausalMask(ptr, (int)totalRows, (int)cols, seqLen, startPos, maskedValue);
            }
        }

        [RegisterOpStorageType("rope", typeof(CpuStorage))]
        public Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RoPE(writeTarget, src, (int)rows, (int)cols, seqLen, rowOffset);
            return writeTarget;
        }

        [RegisterOpStorageType("rope_ex", typeof(CpuStorage))]
        public Tensor RoPEEx(Tensor result, Tensor src, Tensor positions, int ropeDim, int mode, int originalContextLength, float freqBase, float freqScale, float extFactor, float attnFactor, float betaFast, float betaSlow, bool addToResult, bool invertPositions)
        {
            if (positions == null)
            {
                throw new ArgumentNullException(nameof(positions));
            }

            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;
            if (positions.ElementCount() != rows)
            {
                throw new InvalidOperationException("rope_ex expects one explicit position value per logical row.");
            }

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.RoPEEx(writeTarget, src, positions, (int)rows, (int)cols, ropeDim, mode, freqBase, freqScale, originalContextLength, extFactor, attnFactor, betaFast, betaSlow);

            if (addToResult && result != null)
            {
                Ops.Add(writeTarget, writeTarget, result);
            }

            return writeTarget;
        }


        [RegisterOpStorageType("softmax", typeof(CpuStorage))]
        public Tensor Softmax(Tensor result, Tensor src)
        {
            int ndim = src.DimensionCount;
            long storageSize = TensorDimensionHelpers.GetStorageSize(src.Sizes, src.Strides);
            long cols = src.Sizes[ndim - 1];

            if (storageSize % cols != 0)
            {
                throw new Exception($"Invalid tensor storage size = '{storageSize}', and cols = '{cols}'");
            }

            long rows = storageSize / cols;

            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, true, src.Sizes);
            TensorApplyCPU.Softmax(writeTarget, src, (int)rows, (int)cols);
            return writeTarget;
        }


        private readonly MethodInfo rmsProp_func = NativeWrapper.GetMethod("TS_RMSProp");
        private readonly MethodInfo normall_func = NativeWrapper.GetMethod("TS_NormAll");
        [RegisterOpStorageType("normall", typeof(CpuStorage))]
        public Tensor NormAll(Tensor result, Tensor src, float value)
        {
            Tensor writeTarget = TensorResultBuilder.GetWriteTarget(result, src, false, 1);
            NativeWrapper.InvokeTypeMatch(normall_func, writeTarget, src, value);
            return writeTarget;
        }
    }
}
