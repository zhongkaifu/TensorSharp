using System;
using System.Collections.Generic;
using System.Threading;

namespace TensorSharp.MLX
{
    internal static class MlxFallbackOps
    {
        private static int registered;

        public static void Register()
        {
            if (Interlocked.Exchange(ref registered, 1) != 0)
                return;

            foreach (OpSpec spec in Specs)
            {
                OpRegistry.Register(
                    spec.Name,
                    args => MlxCpuFallback.Invoke(spec.Name, spec.ReturnKind, spec.ModifiedTensorIndexes, args),
                    new OpConstraint[]
                    {
                        new ArgCountConstraint(spec.ArgCount),
                        new MlxTensorParticipationConstraint(),
                    });
            }
        }

        private static OpSpec Tensor(string name, int argCount, params int[] modified) =>
            new(name, argCount, MlxFallbackReturnKind.Tensor, modified.Length == 0 ? new[] { 0 } : modified);

        private static OpSpec Void(string name, int argCount, params int[] modified) =>
            new(name, argCount, MlxFallbackReturnKind.Void, modified.Length == 0 ? new[] { 0 } : modified);

        private static OpSpec Raw(string name, int argCount) =>
            new(name, argCount, MlxFallbackReturnKind.Raw, Array.Empty<int>());

        private static readonly OpSpec[] Specs =
        {
            Void("fill", 2),
            Void("copy", 2),
            Tensor("dot", 3),
            Tensor("addmm", 6),
            Tensor("addmmbatch", 6),
            Tensor("mulmatid", 4),
            Tensor("addid", 4),
            Tensor("abs", 2),
            Tensor("neg", 2),
            Tensor("sign", 2),
            Tensor("sqrt", 2),
            Tensor("rsqrt", 2),
            Tensor("exp", 2),
            Tensor("log", 2),
            Tensor("log1p", 2),
            Tensor("floor", 2),
            Tensor("ceil", 2),
            Tensor("round", 2),
            Tensor("trunc", 2),
            Tensor("frac", 2),
            Tensor("relu", 2),
            Tensor("sin", 2),
            Tensor("cos", 2),
            Tensor("tan", 2),
            Tensor("asin", 2),
            Tensor("acos", 2),
            Tensor("atan", 2),
            Tensor("sinh", 2),
            Tensor("cosh", 2),
            Tensor("tanh", 2),
            Tensor("sigmoid", 2),
            Tensor("SiLU", 2),
            Tensor("GELU", 2),
            Tensor("SiLUMul", 3),
            Tensor("SiLUMulSplit", 3),
            Tensor("GELUMul", 3),
            Tensor("SigmoidMul", 3),
            Tensor("SiLUD", 3),
            Tensor("AddSiLUD", 4),
            Tensor("relud", 3),
            Tensor("addrelud", 4),
            Tensor("sigmoidD", 3),
            Tensor("addsigmoidD", 4),
            Tensor("tanhD", 3),
            Tensor("addtanhD", 4),
            Tensor("LeakyReLU", 2),
            Tensor("LeakyReLUD", 3),
            Tensor("AddLeakyReLUD", 4),
            Tensor("addtanh", 3),
            Tensor("addtanh3", 4),
            Tensor("addmul", 4),
            Tensor("adddiv", 4),
            Tensor("addmulv", 4),
            Tensor("mulmuladd", 5),
            Tensor("maskfill", 4),
            Tensor("atan2", 3),
            Tensor("pow", 3),
            Tensor("tpow", 3),
            Tensor("lerp", 4),
            Tensor("clamp", 4),
            Tensor("addv", 3),
            Tensor("subv", 3),
            Tensor("rsubv", 3),
            Tensor("mulv", 3),
            Tensor("divv", 3),
            Tensor("rdivv", 3),
            Tensor("modv", 3),
            Tensor("gtValue", 3),
            Tensor("ltValue", 3),
            Tensor("geValue", 3),
            Tensor("leValue", 3),
            Tensor("eqValue", 3),
            Tensor("neValue", 3),
            Tensor("addt", 3),
            Tensor("subt", 3),
            Tensor("mult", 3),
            Tensor("divt", 3),
            Tensor("modt", 3),
            Tensor("gtTensor", 3),
            Tensor("ltTensor", 3),
            Tensor("geTensor", 3),
            Tensor("leTensor", 3),
            Tensor("eqTensor", 3),
            Tensor("neTensor", 3),
            Tensor("atomicadd", 2),
            Tensor("sum", 3),
            Tensor("mean", 3),
            Tensor("prod", 3),
            Tensor("min", 3),
            Tensor("max", 3),
            Tensor("argmin", 3),
            Tensor("argmax", 3),
            Tensor("norm", 4),
            Tensor("std", 4),
            Tensor("var", 4),
            Tensor("sumall", 2),
            Tensor("prodall", 2),
            Tensor("minall", 2),
            Tensor("maxall", 2),
            Tensor("meanall", 2),
            Tensor("normall", 3),
            Tensor("stdall", 2),
            Tensor("varall", 2),
            Raw("iscorrupted", 1),
            Tensor("softmax", 2),
            Tensor("softmaxgrad", 4),
            Tensor("scaled_dot_product_attention", 6),
            Tensor("indexselect", 4),
            Tensor("indexselectgrad", 3),
            Tensor("repeat_interleave", 4),
            Void("add_causal_mask", 4),
            Tensor("rope", 4),
            Tensor("ropegrad", 4),
            Tensor("rope_ex", 14),
            Tensor("buildsrctgtmask", 7),
            Tensor("buildselfmask", 5),
            Tensor("buildselftrimask", 5),
            Tensor("buildtrimask", 3),
            Tensor("topK", 4, 0, 1),
            Tensor("layernorm", 5),
            Tensor("layernormgrad", 9),
            Tensor("rmsnorm", 5),
            Tensor("rmsnormgrad", 9),
            Tensor("addlayernorm", 6),
            Tensor("addlayernormgrad", 10),
            Tensor("gather", 4),
            Tensor("scatter", 4),
            Tensor("scatter_add", 4),
            Tensor("scatter_fill", 4),
            Tensor("float2half", 2),
            Tensor("half2float", 2),
            Void("random_uniform", 4),
            Void("random_normal", 4),
            Void("random_exponential", 3),
            Void("random_cauchy", 4),
            Void("random_lognormal", 4),
            Void("random_geometric", 3),
            Void("random_bernoulli", 3),
        };

        private readonly record struct OpSpec(string Name, int ArgCount, MlxFallbackReturnKind ReturnKind, int[] ModifiedTensorIndexes);

        private sealed class MlxTensorParticipationConstraint : OpConstraint
        {
            public override bool SatisfiedFor(object[] args)
            {
                bool hasMlxTensor = false;
                foreach (object arg in args)
                {
                    if (arg is Tensor tensor && tensor.Storage is MlxStorage)
                        hasMlxTensor = true;
                }

                return hasMlxTensor;
            }
        }
    }
}
