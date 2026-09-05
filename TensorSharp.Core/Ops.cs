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

using TensorSharp.Core;

namespace TensorSharp
{
    public static class Ops
    {
        /// <summary>
        /// Logit offset for invalid attention positions before softmax. Matches GGML
        /// <c>ggml_compute_forward_diag_mask_inf</c> (-INFINITY) and Ollama/llama.cpp causal masking.
        /// </summary>
        public const float AttentionMaskMaskedLogit = float.NegativeInfinity;

        public static Tensor NewContiguous(Tensor src)
        {
            Tensor result = new Tensor(src.Allocator, src.ElementType, src.Sizes);
            Copy(result, src);
            return result;
        }

        public static Tensor AsContiguous(Tensor src)
        {
            if (src.IsContiguous())
            {
                return src.CopyRef();
            }
            else
            {
                return NewContiguous(src);
            }
        }

        public static Tensor Concat(Tensor result, int dimension, params Tensor[] inputs)
        {
            return TensorConcatenation.Concat(result, dimension, inputs);
        }

        // Note: the null-forgiving operator is used in those cases where we can safely assume that the result of an Invoke() operation will never be null (verifying this assumption at runtime is computationally too expensive).

        public static void Copy(Tensor result, Tensor src) { OpRegistry.Invoke("copy", result, src); }
        public static void Fill(Tensor result, float value) { OpRegistry.Invoke("fill", result, value); }

        public static Tensor Dot(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("dot", result, lhs, rhs)!; }
        public static Tensor Addmm(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmm", result, beta, src, alpha, m1, m2)!; }

        public static Tensor AddmmBatch(Tensor result, float beta, Tensor src, float alpha, Tensor m1, Tensor m2) { return (Tensor)OpRegistry.Invoke("addmmbatch", result, beta, src, alpha, m1, m2)!; }

        public static Tensor Abs(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("abs", result, src)!; }
        public static Tensor Neg(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("neg", result, src)!; }
        public static Tensor Sign(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sign", result, src)!; }



        public static Tensor SiLU(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("SiLU", result, src)!; }
        public static Tensor GELU(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("GELU", result, src)!; }

        public static Tensor SiLUMul(Tensor result, Tensor gate, Tensor up) { return (Tensor)OpRegistry.Invoke("SiLUMul", result, gate, up)!; }

        /// <summary>
        /// Clamped SwiGLU: result = clamp(up, -limit, +limit) * SiLU(min(gate, limit)).
        /// A non-positive <paramref name="limit"/> disables the clamp, making this
        /// exactly <see cref="SiLUMul"/>.
        ///
        /// The asymmetry is deliberate and matches the DeepSeek V4 reference (and
        /// llama.cpp's ggml_swiglu_clamped): the gate is only clamped from above,
        /// because SiLU already saturates on the negative side, while the up
        /// projection is clamped on both.
        /// </summary>
        public static Tensor SiLUMulClamp(Tensor result, Tensor gate, Tensor up, float limit) { return (Tensor)OpRegistry.Invoke("SiLUMulClamp", result, gate, up, limit)!; }

        public static Tensor SiLUMulSplit(Tensor result, Tensor gateUp, int halfDim) { return (Tensor)OpRegistry.Invoke("SiLUMulSplit", result, gateUp, halfDim)!; }
        public static Tensor GELUMul(Tensor result, Tensor gate, Tensor up) { return (Tensor)OpRegistry.Invoke("GELUMul", result, gate, up)!; }
        public static Tensor SigmoidMul(Tensor result, Tensor x, Tensor gate) { return (Tensor)OpRegistry.Invoke("SigmoidMul", result, x, gate)!; }







        public static Tensor Relu(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("relu", result, src)!; }











        public static Tensor Exp(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("exp", result, src)!; }
        public static Tensor Log(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log", result, src)!; }
        public static Tensor Log1p(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("log1p", result, src)!; }
        public static Tensor Floor(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("floor", result, src)!; }
        public static Tensor Ceil(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("ceil", result, src)!; }
        public static Tensor Round(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("round", result, src)!; }
        public static Tensor Trunc(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("trunc", result, src)!; }
        public static Tensor Frac(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("frac", result, src)!; }

        public static Tensor Cos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cos", result, src)!; }
        public static Tensor Tan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tan", result, src)!; }

        public static Tensor Asin(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("asin", result, src)!; }
        public static Tensor Acos(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("acos", result, src)!; }
        public static Tensor Atan(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("atan", result, src)!; }

        public static Tensor Sinh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sinh", result, src)!; }
        public static Tensor Cosh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("cosh", result, src)!; }
        public static Tensor Tanh(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("tanh", result, src)!; }

        public static Tensor Sigmoid(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("sigmoid", result, src)!; }








        public static Tensor MulMulAdd(Tensor result, Tensor x, Tensor y, Tensor z, Tensor w) { return (Tensor)OpRegistry.Invoke("mulmuladd", result, x, y, z, w)!; }

        public static Tensor AddMul(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("addmul", result, x, y, z)!; }
        public static Tensor AddMulV(Tensor result, Tensor x, Tensor y, float z) { return (Tensor)OpRegistry.Invoke("addmulv", result, x, y, z)!; }

        public static Tensor AddDiv(Tensor result, Tensor x, Tensor y, Tensor z) { return (Tensor)OpRegistry.Invoke("adddiv", result, x, y, z)!; }




        public static Tensor Atan2(Tensor result, Tensor srcY, Tensor srcX) { return (Tensor)OpRegistry.Invoke("atan2", result, srcY, srcX)!; }
        public static Tensor Pow(Tensor result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("pow", result, src, value)!; }
        public static Tensor Tpow(Tensor result, float value, Tensor src) { return (Tensor)OpRegistry.Invoke("tpow", result, value, src)!; }
        public static Tensor Lerp(Tensor result, Tensor srcA, Tensor srcB, float weight) { return (Tensor)OpRegistry.Invoke("lerp", result, srcA, srcB, weight)!; }
        public static Tensor Clamp(Tensor result, Tensor src, float min, float max) { return (Tensor)OpRegistry.Invoke("clamp", result, src, min, max)!; }

        public static Tensor Add(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("addv", result, lhs, rhs)!; }
        public static Tensor Sub(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("subv", result, lhs, rhs)!; }
        public static Tensor Sub(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rsubv", result, lhs, rhs)!; }
        public static Tensor Mul(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("mulv", result, lhs, rhs)!; }
        public static Tensor Div(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("divv", result, lhs, rhs)!; }
        public static Tensor Div(Tensor result, float lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("rdivv", result, lhs, rhs)!; }

        public static Tensor GreaterThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("gtValue", result, lhs, rhs)!; }
        public static Tensor LessThan(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("ltValue", result, lhs, rhs)!; }
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("geValue", result, lhs, rhs)!; }
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("leValue", result, lhs, rhs)!; }
        public static Tensor EqualTo(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("eqValue", result, lhs, rhs)!; }
        public static Tensor NotEqual(Tensor result, Tensor lhs, float rhs) { return (Tensor)OpRegistry.Invoke("neValue", result, lhs, rhs)!; }

        public static Tensor Add(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("addt", result, lhs, rhs)!; }
        public static Tensor Sub(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("subt", result, lhs, rhs)!; }
        public static Tensor Mul(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("mult", result, lhs, rhs)!; }
        public static Tensor Div(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("divt", result, lhs, rhs)!; }



        public static Tensor GreaterThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("gtTensor", result, lhs, rhs)!; }
        public static Tensor LessThan(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("ltTensor", result, lhs, rhs)!; }
        public static Tensor GreaterOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("geTensor", result, lhs, rhs)!; }
        public static Tensor LessOrEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("leTensor", result, lhs, rhs)!; }
        public static Tensor EqualTo(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("eqTensor", result, lhs, rhs)!; }
        public static Tensor NotEqual(Tensor result, Tensor lhs, Tensor rhs) { return (Tensor)OpRegistry.Invoke("neTensor", result, lhs, rhs)!; }


        public static Tensor Sum(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("sum", result, src, dimension)!; }
        public static Tensor Prod(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("prod", result, src, dimension)!; }
        public static Tensor Min(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("min", result, src, dimension)!; }
        public static Tensor Max(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("max", result, src, dimension)!; }
        public static Tensor Argmin(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmin", result, src, dimension)!; }
        public static Tensor Argmax(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("argmax", result, src, dimension)!; }

        public static Tensor Mean(Tensor result, Tensor src, int dimension) { return (Tensor)OpRegistry.Invoke("mean", result, src, dimension)!; }
        public static Tensor Norm(Tensor result, Tensor src, int dimension, float value) { return (Tensor)OpRegistry.Invoke("norm", result, src, dimension, value)!; }
        public static Tensor Std(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("std", result, src, dimension, normByN)!; }
        public static Tensor Var(Tensor result, Tensor src, int dimension, bool normByN) { return (Tensor)OpRegistry.Invoke("var", result, src, dimension, normByN)!; }

        public static Tensor Softmax(Tensor result, Tensor src) { return (Tensor)OpRegistry.Invoke("softmax", result, src)!; }


        public static Tensor IndexSelect(Tensor result, Tensor src, Tensor indice, bool isAdd = false) { return (Tensor)OpRegistry.Invoke("indexselect", result, src, indice, isAdd)!; }

        /// <summary>
        /// Repeat each slice along <paramref name="dim"/> <paramref name="repeats"/> times consecutively.
        /// Input must be contiguous. Result shape is the same as src except dimension <paramref name="dim"/>
        /// is multiplied by <paramref name="repeats"/>.
        /// </summary>
        public static Tensor RepeatInterleave(Tensor result, Tensor src, int repeats, int dim) { return (Tensor)OpRegistry.Invoke("repeat_interleave", result, src, repeats, dim)!; }

        /// <summary>
        /// In-place causal mask: for each logical row t (with period <paramref name="seqLen"/> across outer dims),
        /// add <paramref name="maskedValue"/> to all positions s where s &gt; startPos + t.
        /// The tensor's last dimension is the key-sequence dimension.
        /// </summary>
        public static void AddCausalMask(Tensor tensor, int seqLen, int startPos, float maskedValue) { OpRegistry.Invoke("add_causal_mask", tensor, seqLen, startPos, maskedValue); }



        public static Tensor RoPE(Tensor result, Tensor src, int seqLen, int rowOffset) { return (Tensor)OpRegistry.Invoke("rope", result, src, seqLen, rowOffset)!; }
        public static Tensor RoPEEx(Tensor result, Tensor src, Tensor positions, int ropeDim, int mode, int originalContextLength, float freqBase, float freqScale, float extFactor = 0.0f, float attnFactor = 1.0f, float betaFast = 0.0f, float betaSlow = 0.0f, bool addToResult = false, bool invertPositions = false)
        {
            return (Tensor)OpRegistry.Invoke("rope_ex", result, src, positions, ropeDim, mode, originalContextLength, freqBase, freqScale, extFactor, attnFactor, betaFast, betaSlow, addToResult, invertPositions)!;
        }








        public static Tensor LayerNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("layernorm", result, src, alpha, beta, eps)!; }


        public static Tensor ScaledDotProductAttention(Tensor result, Tensor query, Tensor key, Tensor value, Tensor mask, float scale)
        {
            return (Tensor)OpRegistry.Invoke("scaled_dot_product_attention", result, query, key, value, mask, scale)!;
        }


        public static Tensor RMSNorm(Tensor result, Tensor src, Tensor alpha, Tensor beta, float eps = 1e-09f) { return (Tensor)OpRegistry.Invoke("rmsnorm", result, src, alpha, beta, eps)!; }




        public static Tensor SumAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("sumall", result, src)!; }
        public static Tensor ProdAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("prodall", result, src)!; }
        public static Tensor MinAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("minall", result, src)!; }
        public static Tensor MaxAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("maxall", result, src)!; }

        public static Tensor MeanAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("meanall", result, src)!; }
        public static Tensor NormAll(Tensor? result, Tensor src, float value) { return (Tensor)OpRegistry.Invoke("normall", result, src, value)!; }
        public static Tensor StdAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("stdall", result, src)!; }
        public static Tensor VarAll(Tensor? result, Tensor src) { return (Tensor)OpRegistry.Invoke("varall", result, src)!; }


        public static float SumAll(Tensor src) { using (Tensor resultTensor = SumAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float ProdAll(Tensor src) { using (Tensor resultTensor = ProdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float MinAll(Tensor src) { using (Tensor resultTensor = MinAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float MaxAll(Tensor src) { using (Tensor resultTensor = MaxAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }

        public static float MeanAll(Tensor src) { using (Tensor resultTensor = MeanAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float VarAll(Tensor src) { using (Tensor resultTensor = VarAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float StdAll(Tensor src) { using (Tensor resultTensor = StdAll(null, src)) { return resultTensor.GetElementAsFloat(0); } }
        public static float NormAll(Tensor src, float value) { using (Tensor resultTensor = NormAll(null, src, value)) { return resultTensor.GetElementAsFloat(0); } }


     //   public static Tensor IndexSelect(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("index_select", result, src, dim, indices); }
        public static Tensor Gather(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("gather", result, src, dim, indices)!; } 
        public static Tensor Scatter(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter", result, src, dim, indices)!; }


        public static Tensor ScatterAdd(Tensor result, Tensor src, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_add", result, src, dim, indices)!; } 

        public static Tensor ScatterFill(Tensor result, float value, int dim, Tensor indices) { return (Tensor)OpRegistry.Invoke("scatter_fill", result, value, dim, indices)!; }



        private static int? GetSeed(RandomGenerator src)
        {
            return src == null ? (int?)null : src.NextSeed();
        }

        public static void RandomUniform(Tensor result, RandomGenerator seedSource, float min, float max) { OpRegistry.Invoke("random_uniform", result, GetSeed(seedSource), min, max); }
        public static void RandomNormal(Tensor result, RandomGenerator seedSource, float mean, float stdv) { OpRegistry.Invoke("random_normal", result, GetSeed(seedSource), mean, stdv); }
        public static void RandomExponential(Tensor result, RandomGenerator seedSource, float lambda) { OpRegistry.Invoke("random_exponential", result, GetSeed(seedSource), lambda); }
        public static void RandomCauchy(Tensor result, RandomGenerator seedSource, float median, float sigma) { OpRegistry.Invoke("random_cauchy", result, GetSeed(seedSource), median, sigma); }
        public static void RandomLogNormal(Tensor result, RandomGenerator seedSource, float mean, float stdv) { OpRegistry.Invoke("random_lognormal", result, GetSeed(seedSource), mean, stdv); }
        public static void RandomGeometric(Tensor result, RandomGenerator seedSource, float p) { OpRegistry.Invoke("random_geometric", result, GetSeed(seedSource), p); }
        public static void RandomBernoulli(Tensor result, RandomGenerator seedSource, float p) { OpRegistry.Invoke("random_bernoulli", result, GetSeed(seedSource), p); }
    }
}
