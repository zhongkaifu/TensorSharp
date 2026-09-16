using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Independent double arithmetic over a small subspace sampled from the pinned
/// Q8_0 MTP head. Activations are synthetic. This tests production CPU mixer math,
/// not complete draft-head inference, CUDA kernels or speculative quality.
/// </summary>
public class Qwen4ExpMtpMathTests
{
    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(65)]
    public void HeadMixer_MatchesGroupedNormLowRankGateAndMeanOracle(int tokens)
    {
        using var fixture = new MixerFixture();
        float[] input = Inputs(tokens);
        using var residual = fixture.Tensor(input, tokens, 32);
        float[] before = residual.GetElementsAsFloat(tokens * 32);
        object?[] args = { residual, tokens, "norm", "down", "up", null, null };
        using var output = (Tensor)Invoke(fixture.Model, "HcMix", args)!;
        Assert.Null(args[6]);
        float[] actual = output.GetElementsAsFloat(tokens * 8);
        double[] expected = HeadOracle(input, tokens);
        AssertClose(expected, actual);
        Assert.Equal(before, residual.GetElementsAsFloat(tokens * 32));

        // The fixture must distinguish the mistakes this regression is intended
        // to catch. A single RMS across all four streams is a different model.
        double[] globalRms = HeadOracle(input, tokens, globalNorm: true);
        Assert.True(expected.Zip(globalRms, (a, b) => Math.Abs(a - b)).Max() > 1e-3);
        double[] gammaAddedTwice = HeadOracle(input, tokens, addOneToGamma: true);
        Assert.True(expected.Zip(gammaAddedTwice, (a, b) => Math.Abs(a - b)).Max() > 1e-3);
    }

    internal static float[] Inputs(int tokens)
    {
        var result = new float[tokens * 32];
        for (int t = 0; t < tokens; t++)
            for (int c = 0; c < 4; c++)
                for (int i = 0; i < 8; i++)
                    result[t * 32 + c * 8 + i] = (float)(((t + 3) * (i + 2) + 5 * c) % 17 - 8)
                        * (1 << c) + (i + 1) * .125f;
        return result;
    }

    private static double[] HeadOracle(float[] input, int tokens, bool globalNorm = false, bool addOneToGamma = false)
    {
        var output = new double[tokens * 8];
        for (int t = 0; t < tokens; t++)
        {
            var normed = new double[32];
            for (int c = 0; c < 4; c++)
            {
                int first = globalNorm ? 0 : c * 8;
                int count = globalNorm ? 32 : 8;
                double square = 0;
                for (int i = first; i < first + count; i++) square += (double)input[t * 32 + i] * input[t * 32 + i];
                double rms = Math.Sqrt(square / count + Qwen4ExpMtpSample.Epsilon);
                for (int i = 0; i < 8; i++)
                    normed[c * 8 + i] = input[t * 32 + c * 8 + i] / rms
                        * (Qwen4ExpMtpSample.HeadNorm[c * 8 + i] + (addOneToGamma ? 1 : 0));
            }
            var low = new double[3];
            for (int j = 0; j < 3; j++)
            {
                double dot = 0;
                for (int i = 0; i < 32; i++) dot += Qwen4ExpMtpSample.HeadDown[j * 32 + i] * normed[i];
                dot /= 4; // The fork applies this BEFORE SiLU.
                low[j] = dot / (1 + Math.Exp(-dot));
            }
            for (int i = 0; i < 8; i++)
                for (int c = 0; c < 4; c++)
                {
                    int channel = c * 8 + i;
                    double dot = 0;
                    for (int j = 0; j < 3; j++) dot += Qwen4ExpMtpSample.HeadUp[channel * 3 + j] * low[j];
                    output[t * 8 + i] += normed[channel] / (1 + Math.Exp(-dot)) / 4;
                }
        }
        return output;
    }

    internal static void AssertClose(double[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"Non-finite output at {i}");
            double bound = 2e-6 + 2e-5 * Math.Abs(expected[i]);
            Assert.True(Math.Abs(expected[i] - actual[i]) <= bound,
                $"index {i}: expected {expected[i]:R}, actual {actual[i]:R}, bound {bound:R}");
        }
    }

    private sealed class MixerFixture : IDisposable
    {
        internal readonly Qwen4ExpModel Model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        private readonly CpuAllocator allocator = new(BlasEnum.DotNet);
        private readonly List<Tensor> weights = new();
        internal MixerFixture()
        {
            Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig
            { HiddenSize = 8, NumLayers = 1, Eps = Qwen4ExpMtpSample.Epsilon });
            Set(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.Cpu));
            Set(typeof(ModelBase), Model, "_backend", BackendType.Cpu);
            Set(typeof(ModelBase), Model, "_allocator", allocator);
            Set(typeof(Qwen4ExpModel), Model, "_hc", 4);
            Set(typeof(Qwen4ExpModel), Model, "_hcDim", 32);
            Set(typeof(Qwen4ExpModel), Model, "_hcLowRank", 3);
            var dictionary = new Dictionary<string, Tensor>();
            Set(typeof(ModelBase), Model, "_weights", dictionary);
            var quant = typeof(ModelBase).GetField("_quantWeights", BindingFlags.Instance | BindingFlags.NonPublic)!;
            quant.SetValue(Model, Activator.CreateInstance(quant.FieldType));
            dictionary["norm"] = Weight(Qwen4ExpMtpSample.HeadNorm, 32);
            dictionary["down"] = Weight(Qwen4ExpMtpSample.HeadDown, 3, 32);
            dictionary["up"] = Weight(Qwen4ExpMtpSample.HeadUp, 32, 3);
        }
        private Tensor Weight(float[] values, params long[] shape)
        {
            var tensor = Tensor(values, shape); weights.Add(tensor); return tensor;
        }
        internal Tensor Tensor(float[] values, params long[] shape)
        {
            var tensor = new Tensor(allocator, DType.Float32, shape);
            tensor.SetElementsAsFloat(values);
            return tensor;
        }
        public void Dispose()
        {
            foreach (Tensor weight in weights) weight.Dispose();
            GC.SuppressFinalize(Model);
        }
    }

    internal static void Set(Type type, object target, string field, object value) => type.GetField(field,
        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.SetValue(target, value);
    internal static object? Invoke(object target, string method, object?[] args)
    {
        try { return target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args); }
        catch (TargetInvocationException e) when (e.InnerException != null)
        { ExceptionDispatchInfo.Capture(e.InnerException).Throw(); throw; }
    }
}
