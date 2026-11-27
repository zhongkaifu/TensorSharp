using System.Text;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed partial class Ltx25DenoiserTests
{
    /// <summary>y = original + B (A x) on <paramref name="executor"/>, the adapter's term of the denoiser's projections.</summary>
    private static float[] WithAdapter(TensorGraphExecutor executor, (Ltx25Weight Down, Ltx25Weight Up) factors, float[] x, int rows,
        float[] original)
    {
        using var e = new GraphEmitter(executor);
        int Leaf(Ltx25Weight w) => e.HostWeight(w.Format, new[] { w.Ne0, w.Ne1 }, GraphIdentity.Of(w), w.Bytes, w);
        int input = e.InputF32(x, factors.Down.Ne0, rows);
        int y = e.Op(GraphOp.Add, e.InputF32(original, factors.Up.Ne1, rows),
            e.Op(GraphOp.MulMat, Leaf(factors.Up), e.Op(GraphOp.MulMat, Leaf(factors.Down), input)));
        int output = e.Output(y, (long)rows * factors.Up.Ne1);
        using IGraphPlan plan = e.Compile(replayable: true);
        Assert.True(e.Run(plan), plan.LastError);
        return (float[])plan.Outputs[output].Clone();
    }

    [IngredientsWeightsFact]
    public void ActualIngredientsRank128AttentionAndFeedForwardFactorsMatchIndependentDoubleOracle()
    {
        using var backend = new Backend();
        using var checkpoint = new GgufFile(Environment.GetEnvironmentVariable("TS_LTX25_TEST_DENOISER")!);
        string loraPath = Environment.GetEnvironmentVariable("TS_LTX25_TEST_LORA")!;
        using var factors = new SafetensorsFile(loraPath);
        using var adapter = Ltx25LoraSet.Open(loraPath, checkpoint, Ltx25Config.FromGguf(checkpoint));
        var executor = Ltx25Denoiser.CreateExecutor(backend.Allocator);
        using var release = executor as IDisposable;
        Assert.Equal(480, adapter.ProjectionCount);
        foreach (string target in new[] { "transformer_blocks.0.attn1.to_q", "transformer_blocks.23.ff.net.0.proj", "transformer_blocks.47.ff.net.2" })
        {
            string aName = "diffusion_model." + target + ".lora_A.weight";
            string bName = "diffusion_model." + target + ".lora_B.weight";
            int inner = checked((int)factors.GetInfo(aName).Shape[1]);
            int rank = checked((int)factors.GetInfo(aName).Shape[0]);
            int outer = checked((int)factors.GetInfo(bName).Shape[0]);
            Assert.Equal(128, rank);
            float[] a = factors.ReadFloat32(aName), b = factors.ReadFloat32(bName);
            float[] values = Enumerable.Range(0, inner * 3).Select(i => MathF.Sin(i * .12379f) * .25f + i % 7 * .000137f).ToArray();
            float[] original = Enumerable.Range(0, outer * 3).Select(i => (i % 17 - 8) * .001f).ToArray();
            float[] actual = WithAdapter(executor, adapter.Factors(target)!.Value, values, 3, original);
            var lowRank = new double[rank * 3];
            for (int row = 0; row < 3; ++row)
                for (int r = 0; r < rank; ++r)
                    for (int i = 0; i < inner; ++i) lowRank[row * rank + r] += (double)values[row * inner + i] * a[r * inner + i];
            double errorSquared = 0, expectedSquared = 0, maximumError = 0, maximumExpected = 0;
            for (int row = 0; row < 3; ++row)
                for (int channel = 0; channel < outer; ++channel)
                {
                    double expected = original[row * outer + channel];
                    for (int r = 0; r < rank; ++r) expected += lowRank[row * rank + r] * b[channel * rank + r];
                    double error = actual[row * outer + channel] - expected;
                    errorSquared += error * error; expectedSquared += expected * expected;
                    maximumError = Math.Max(maximumError, Math.Abs(error)); maximumExpected = Math.Max(maximumExpected, Math.Abs(expected));
                }
            double relativeL2 = Math.Sqrt(errorSquared / expectedSquared);
            Assert.True(relativeL2 < 2e-5 && maximumError < Math.Max(2e-5, maximumExpected * 2e-5),
                $"Actual Ingredients {target}: relativeL2={relativeL2:G9}, maxError={maximumError:G9}, maxExpected={maximumExpected:G9}");
            Console.WriteLine($"[ltx25-ingredients-oracle] {target} shape={outer}x{inner} rank={rank} relativeL2={relativeL2:G9} maxError={maximumError:G9}");
        }
        Assert.Equal(3, adapter.LoadedProjectionCount);
    }

    [Theory]
    [InlineData("F32")]
    [InlineData("BF16")]
    [InlineData("F16")]
    public void IngredientsProjectionMatchesIndependentUnscaledBAMatmul(string dtype)
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var checkpoint = new GgufFile(fixture.Path);
        var config = Ltx25Config.FromGguf(checkpoint);
        using var factors = new LoraFixture(dtype);
        using var adapter = Ltx25LoraSet.Open(factors.Path, checkpoint, config);
        var executor = Ltx25Denoiser.CreateExecutor(backend.Allocator);
        using var release = executor as IDisposable;
        Assert.Equal(1, adapter.ProjectionCount);
        Assert.Equal(0, adapter.LoadedProjectionCount);
        var pair = adapter.Factors(LoraFixture.Target)!.Value;
        // Down = A [in 16, rank 2] and up = B [rank 2, out 16], widened exactly to F32 and taken as exact products.
        Assert.Equal((16L, 2L, 2L, 16L), (pair.Down.Ne0, pair.Down.Ne1, pair.Up.Ne0, pair.Up.Ne1));
        Assert.True(pair.Down.ExactProducts && pair.Up.ExactProducts);
        Assert.Equal(1, adapter.LoadedProjectionCount);
        Assert.Equal(pair, adapter.Factors(LoraFixture.Target)!.Value);
        float[] values = Enumerable.Range(0, 48).Select(i => (i % 9 - 4) * 1.17391f + i * .000137f).ToArray();
        float[] original = Enumerable.Range(0, 48).Select(i => i * .013f).ToArray();
        var expected = (float[])original.Clone();
        for (int row = 0; row < 3; ++row)
            for (int channel = 0; channel < 16; ++channel)
            {
                double update = 0;
                for (int rank = 0; rank < 2; ++rank)
                {
                    double lowRank = 0;
                    for (int i = 0; i < 16; ++i) lowRank += (double)values[row * 16 + i] * factors.A[rank * 16 + i];
                    update += lowRank * factors.B[channel * 2 + rank];
                }
                expected[row * 16 + channel] += (float)update;
            }
        Equal(expected, WithAdapter(executor, pair, values, 3, original), 2e-5, "Ingredients unscaled B@A");
        // No adapter update exists for gates or audio projections.
        Assert.Null(adapter.Factors("transformer_blocks.0.audio_attn1.to_q"));
        Assert.Null(adapter.Factors("transformer_blocks.0.attn1.to_gate_logits"));
        Assert.Equal(1, adapter.LoadedProjectionCount);
        Assert.Equal(2 * (16 + 16) * sizeof(float), adapter.FactorBytes);
    }

    [Fact]
    public void IngredientsDenoiserIsOptInAndCannotLeakIntoBaseGeneration()
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var factors = new LoraFixture();
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator, factors.Path);
        var reference = Oracle.GetProperty("cases")[0];
        Assert.True(model.ReferenceLoraLoaded);
        Assert.Equal(1, model.ReferenceLoraProjections);
        var before = model.Forward(Input(reference));
        Assert.Equal(0, model.CachedReferenceLoraProjections);
        var adapted = model.Forward(Input(reference, useReferenceLora: true));
        Assert.Equal(1, model.CachedReferenceLoraProjections);
        Assert.True(before.VideoVelocity.Zip(adapted.VideoVelocity).Any(pair => Math.Abs(pair.First - pair.Second) > 1e-6));
        var after = model.Forward(Input(reference));
        Assert.Equal(before.VideoVelocity, after.VideoVelocity);
        Assert.Equal(before.AudioVelocity, after.AudioVelocity);
        model.ReleaseDevice();
        var rebuilt = model.Forward(Input(reference));
        Equal(before.VideoVelocity, rebuilt.VideoVelocity, 1e-6, "base after adapter eviction");
        var readapted = model.Forward(Input(reference, useReferenceLora: true));
        Assert.Equal(adapted.VideoVelocity, readapted.VideoVelocity);
        Assert.Equal(adapted.AudioVelocity, readapted.AudioVelocity);
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        Assert.Throws<OperationCanceledException>(() => model.Forward(Input(reference, true, cancellation.Token)));
        var recovered = model.Forward(Input(reference));
        Assert.Equal(before.VideoVelocity, recovered.VideoVelocity);
    }

    [Fact]
    public void IngredientsRequestsFailBeforeWorkWithoutLoadedAdapter()
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator);
        Assert.False(model.ReferenceLoraLoaded);
        var reference = Oracle.GetProperty("cases")[0];
        Assert.Throws<InvalidOperationException>(() => model.Forward(Input(reference, true)));
        Equal(Floats(reference, "videoVelocity"), model.Forward(Input(reference)).VideoVelocity, 2e-5, "base after rejected adapter");
    }

    [Theory]
    [InlineData("version")]
    [InlineData("downscale")]
    [InlineData("audio")]
    [InlineData("gate")]
    [InlineData("block")]
    [InlineData("rank")]
    [InlineData("input")]
    [InlineData("output")]
    [InlineData("missing_a")]
    [InlineData("missing_b")]
    [InlineData("extra_tensor")]
    [InlineData("dtype")]
    public void IngredientsRejectsIncompatibleMetadataTargetsAndFactorPairs(string corruption)
    {
        using var fixture = new Checkpoint();
        using var checkpoint = new GgufFile(fixture.Path);
        using var factors = new LoraFixture(corruption: corruption);
        Assert.Throws<InvalidDataException>(() => Ltx25LoraSet.Open(factors.Path, checkpoint, Ltx25Config.FromGguf(checkpoint)));
    }

    [Fact]
    public void IngredientsNonfiniteFactorsAndDisposalLeaveNoLoadedProjection()
    {
        using var fixture = new Checkpoint();
        using var backend = new Backend();
        using var checkpoint = new GgufFile(fixture.Path);
        using var factors = new LoraFixture(corruption: "nonfinite");
        var adapter = Ltx25LoraSet.Open(factors.Path, checkpoint, Ltx25Config.FromGguf(checkpoint));
        Assert.Throws<InvalidDataException>(() => adapter.Factors(LoraFixture.Target));
        Assert.Equal(0, adapter.LoadedProjectionCount);
        adapter.Dispose(); adapter.Dispose();
        Assert.Throws<ObjectDisposedException>(() => adapter.Factors(LoraFixture.Target));
        // A denoiser whose adapter cannot be read fails the adapted step and still runs the base model.
        using var model = new Ltx25Denoiser(fixture.Path, backend.Allocator, factors.Path);
        var reference = Oracle.GetProperty("cases")[0];
        Assert.Throws<InvalidDataException>(() => model.Forward(Input(reference, true)));
        Assert.Equal(0, model.CachedReferenceLoraProjections);
        Equal(Floats(reference, "videoVelocity"), model.Forward(Input(reference)).VideoVelocity, 2e-5, "base after unreadable adapter");
    }

    private sealed class LoraFixture : IDisposable
    {
        internal const string Target = "transformer_blocks.0.attn1.to_q";
        public string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "tensorsharp-ltx25-lora-" + Guid.NewGuid().ToString("N") + ".safetensors");
        public float[] A { get; } = Enumerable.Range(0, 32).Select(i => (i % 13 - 6) * .03125f).ToArray();
        public float[] B { get; } = Enumerable.Range(0, 32).Select(i => (i % 11 - 5) * .0625f).ToArray();
        public LoraFixture(string dtype = "F32", string corruption = null)
        {
            var metadata = new Dictionary<string, string> { ["model_version"] = "2.5", ["reference_downscale_factor"] = "1" };
            if (corruption == "version") metadata["model_version"] = "2.4";
            if (corruption == "downscale") metadata["reference_downscale_factor"] = "2";
            string target = corruption switch
            {
                "audio" => "transformer_blocks.0.audio_attn1.to_q",
                "gate" => "transformer_blocks.0.attn1.to_gate_logits",
                "block" => "transformer_blocks.2.attn1.to_q",
                _ => Target,
            };
            if (corruption == "nonfinite") B[0] = float.NaN;
            var header = new Dictionary<string, object> { ["__metadata__"] = metadata };
            using var data = new MemoryStream();
            void Add(string name, float[] values, int[] shape)
            {
                long begin = data.Position;
                using var writer = new BinaryWriter(data, Encoding.UTF8, true);
                foreach (float value in values)
                    if (dtype == "F32") writer.Write(value);
                    else if (dtype == "BF16") writer.Write((ushort)(BitConverter.SingleToUInt32Bits(value) >> 16));
                    else writer.Write(BitConverter.HalfToUInt16Bits((System.Half)value));
                header.Add(name, new { dtype = corruption == "dtype" ? "I32" : dtype, shape, data_offsets = new[] { begin, data.Position } });
            }
            if (corruption != "missing_a") Add("diffusion_model." + target + ".lora_A.weight", A,
                corruption == "input" ? [1, 32] : [2, 16]);
            if (corruption != "missing_b") Add("diffusion_model." + target + ".lora_B.weight", B,
                corruption == "rank" ? [8, 4] : corruption == "output" ? [8, 4] : [16, 2]);
            if (corruption == "extra_tensor") Add("diffusion_model.unexpected.weight", [1], [1, 1]);
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(header);
            using var file = File.Create(Path);
            using var output = new BinaryWriter(file);
            int paddedLength = checked((json.Length + 7) / 8 * 8);
            output.Write((ulong)paddedLength); output.Write(json);
            for (int i = json.Length; i < paddedLength; ++i) output.Write((byte)' ');
            data.Position = 0; data.CopyTo(file);
        }
        public void Dispose() => File.Delete(Path);
    }
}

public sealed class IngredientsWeightsFactAttribute : FactAttribute
{
    public IngredientsWeightsFactAttribute()
    {
        if (!File.Exists(Environment.GetEnvironmentVariable("TS_LTX25_TEST_DENOISER")) ||
            !File.Exists(Environment.GetEnvironmentVariable("TS_LTX25_TEST_LORA")))
            Skip = "Set TS_LTX25_TEST_DENOISER and TS_LTX25_TEST_LORA to the authorized real LTX 2.5 checkpoint and Ingredients adapter.";
    }
}
