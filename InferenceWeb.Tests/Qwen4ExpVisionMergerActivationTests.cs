// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen4Exp's patch merger uses GELU(erf), while its transformer-block MLPs
/// use gelu_pytorch_tanh. The primary reference is Qwen4ExpVisionPatchMerger in
/// https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L1705.
/// These tests use a synthetic projector and independent scalar goldens;
/// they qualify activation wiring, not trained-model vision quality.
/// </summary>
public sealed class Qwen4ExpVisionMergerActivationTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "qwen4-merger-" + Guid.NewGuid().ToString("N"));
    private readonly string _mmproj;
    private readonly IAllocator _allocator = new CpuAllocator(BlasEnum.DotNet);
    private static readonly float[] Inputs = [-3f, -2f, -1f, -.5f, .5f, 1f, 2f, 3f];
    // 0.5*x*(1+erf(x/sqrt(2))), evaluated independently with Python math.erf.
    private static readonly float[] Erf = [-.0040496941f, -.0455002639f, -.1586552539f, -.1542687694f,
        .3457312306f, .8413447461f, 1.9544997361f, 2.9959503059f];
    // 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3))).
    private static readonly float[] Tanh = [-.0036373921f, -.0454023059f, -.1588080094f, -.1542859902f,
        .3457140098f, .8411919906f, 1.9545976941f, 2.9963626079f];

    public Qwen4ExpVisionMergerActivationTests()
    {
        Directory.CreateDirectory(_directory);
        _mmproj = QwenVLSyntheticMmprojBuilder.Write(Path.Combine(_directory, "mmproj.gguf"));
    }

    [Fact]
    public void MergerErfOptInChangesFinalProjectionAndKeepsBlockMlpTanh()
    {
        using var qwen4 = new Qwen35VisionEncoder(_mmproj, _allocator, mergerGeluErf: true);
        using var legacy = new Qwen35VisionEncoder(_mmproj, _allocator);

        // Exercise the actual block MLP under the same opt-in. A mistaken shared
        // activation switch would turn this output into the distinct erf goldens.
        var weights = Weights(qwen4);
        int intermediate = QwenVLSyntheticMmprojBuilder.Intermediate;
        weights["v.blk.0.ffn_up.weight"].SetElementsAsFloat(new float[intermediate * Inputs.Length]);
        var bias = new float[intermediate];
        Inputs.CopyTo(bias, 0);
        weights["v.blk.0.ffn_up.bias"].SetElementsAsFloat(bias);
        weights["v.blk.0.ffn_down.weight"].SetElementsAsFloat(SelectionMatrix(Inputs.Length, intermediate));
        weights["v.blk.0.ffn_down.bias"].SetElementsAsFloat(new float[Inputs.Length]);
        using var hidden = new Tensor(_allocator, DType.Float32, 1, Inputs.Length);
        hidden.SetElementsAsFloat(new float[Inputs.Length]);
        using var block = (Tensor)typeof(Qwen35VisionEncoder).GetMethod("VisionMLP", BindingFlags.Instance | BindingFlags.NonPublic)!
            .Invoke(qwen4, [hidden, "v.blk.0"])!;
        AssertClose(Tanh, block.GetElementsAsFloat(Inputs.Length));
        AssertClose(Erf, EncodeControlledMerger(qwen4));
        AssertClose(Tanh, EncodeControlledMerger(legacy));
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("0", false)]
    [InlineData("1", true)]
    public void Qwen4ExpProjectorLoaderRequiresExplicitErfOptIn(string? optIn, bool erf)
    {
        // LoadVisionEncoder needs only the allocator. Keep language-weight loading
        // out of this activation regression, then detach the incomplete host model.
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        typeof(ModelBase).GetField("_allocator", BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, _allocator);
        string? previous = Environment.GetEnvironmentVariable("TS_Q4E_VISION_MERGER_ERF");
        try
        {
            Environment.SetEnvironmentVariable("TS_Q4E_VISION_MERGER_ERF", optIn);
            model.LoadVisionEncoder(_mmproj);
            model.VisionEncoder.SetHostModel(null);
            AssertClose(erf ? Erf : Tanh, EncodeControlledMerger(model.VisionEncoder));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_Q4E_VISION_MERGER_ERF", previous);
            model.VisionEncoder?.Dispose();
            GC.SuppressFinalize(model);
        }
    }

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(1, 8, false)]
    [InlineData(1, 8, true)]
    [InlineData(300, 4608, false)]
    [InlineData(300, 4608, true)]
    [InlineData(3, 4609, true)]
    public void MetalErfUnaryMatchesIndependentGoldensAndPreservesSource(int rows, int width, bool inPlace)
    {
        var context = new GgmlContext([0], GgmlBackendType.Metal);
        var allocator = new GgmlAllocator(context, 0);
        var values = new float[rows * width];
        var expected = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = Inputs[i % Inputs.Length];
            expected[i] = Erf[i % Erf.Length];
        }
        using var source = new Tensor(allocator, DType.Float32, rows, width);
        source.SetElementsAsFloat(values);
        using var separate = inPlace ? null : new Tensor(allocator, DType.Float32, rows, width);
        Tensor destination = separate ?? source;
        GgmlBasicOps.GELUErf(destination, source);
        // Reading drains async Metal compute. Cover the actual 300x4608 card
        // merger shape, aliasing, and a width with a vector-kernel tail.
        AssertClose(expected, destination.GetElementsAsFloat(expected.Length));
        if (!inPlace) Assert.Equal(values, source.GetElementsAsFloat(values.Length));
    }

    [GgmlFact(BackendType.GgmlMetal)]
    public void MetalEncodeCorePatchMergerMatchesIndependentErfGoldens()
    {
        var context = new GgmlContext([0], GgmlBackendType.Metal);
        var allocator = new GgmlAllocator(context, 0);
        using var encoder = new Qwen35VisionEncoder(_mmproj, allocator, mergerGeluErf: true);
        AssertClose(Erf, EncodeControlledMerger(encoder));
    }

    private float[] EncodeControlledMerger(Qwen35VisionEncoder encoder)
    {
        var weights = Weights(encoder);
        int merged = QwenVLSyntheticMmprojBuilder.Hidden * QwenVLSyntheticMmprojBuilder.MergeSize * QwenVLSyntheticMmprojBuilder.MergeSize;
        // A zero first projection plus known bias isolates activation in the real
        // EncodeCore merger; the output projection selects all eight test values.
        weights["mm.0.weight"].SetElementsAsFloat(new float[merged * merged]);
        var bias = new float[merged];
        Inputs.CopyTo(bias, 0);
        weights["mm.0.bias"].SetElementsAsFloat(bias);
        weights["mm.2.weight"].SetElementsAsFloat(SelectionMatrix(Inputs.Length, merged));
        weights["mm.2.bias"].SetElementsAsFloat(new float[Inputs.Length]);
        int side = QwenVLSyntheticMmprojBuilder.ImageSize;
        using var encoded = encoder.Encode(new float[3 * side * side], side, side);
        Assert.Equal(new long[] { 1, Inputs.Length }, encoded.Sizes.ToArray());
        return encoded.GetElementsAsFloat(Inputs.Length);
    }

    private static float[] SelectionMatrix(int outputs, int inputs)
    {
        var values = new float[outputs * inputs];
        for (int i = 0; i < outputs; i++) values[i * inputs + i] = 1f;
        return values;
    }

    private static Dictionary<string, Tensor> Weights(Qwen35VisionEncoder encoder) =>
        (Dictionary<string, Tensor>)typeof(Qwen35VisionEncoder).GetField("_weights", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(encoder)!;

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(float.IsFinite(actual[i]) && Math.Abs(expected[i] - actual[i]) <= 2e-6f,
                $"activation({Inputs[i % Inputs.Length]}): expected {expected[i]}, actual {actual[i]}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, true);
    }
}
