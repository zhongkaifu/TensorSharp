using TensorSharp;
using TensorSharp.Cuda;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Separate approximation diagnostics. Original pedantic operator
/// tests and their bounds remain unchanged. These TF32 cases keep the same
/// unrounded Double bounds wherever TF32 can meet them, and a failure there
/// remains a failure, never a looser gate. Where a Double bound lies below
/// TF32's own operand-rounding floor on its input, so that no TF32
/// implementation can meet it, the case gates on a bound derived from TF32
/// precision that still rejects coarser formats, and reports how far it
/// exceeds the Double bound. Only the large Q/K exponent case is such a case
/// (<see cref="LargeExponentTf32Bound"/>).</summary>
public sealed class CudaEmbeddingTf32AttentionTests
{
    /// <summary>
    /// The large-exponent case's TF32 bound. Q and K of magnitude 8e4 make each score a 256-term
    /// sum of products up to 3.4e8 that cancels 54-fold (largest sum of |q k| x scale 32.2, largest
    /// |score| 0.60), so rounding Q, K, P and V to TF32 (10 mantissa bits, unit roundoff 2^-11)
    /// alone moves the outputs by up to 8.1e-5 from the Double reference: an exact host emulation
    /// over all 29,696 outputs, 8.1e-5 rounding to nearest and 7.4e-5 truncating (6.7e-5 and
    /// 6.0e-5 at the 60 outputs the pedantic gate checks; a first-order rounding sigma of
    /// 3.0e-5..3.1e-5 per output). The RTX 5090 measures 6.3e-5 with 24% of its outputs above the
    /// 3e-5 Double bound (pedantic F32: 1.6e-8), within 3.0e-5 of the round-to-nearest emulation
    /// (1.2e-4 of the truncating one), and repeats bit for bit. This bound sits
    /// 1.5x above the emulated floor and still fails BF16 operands (7 mantissa bits: 4,764 outputs
    /// above it, max 3.3e-4; 1.7e-4 and 10 of the 60 checked outputs) and any lost exponent range,
    /// which leaves non-finite scores.
    /// </summary>
    private const double LargeExponentTf32Bound = 1.2e-4;

    private readonly ITestOutputHelper _output;

    public CudaEmbeddingTf32AttentionTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void ConstructorPolicyDefaultsToPedanticAndExposesOnlyTwoExplicitModes()
    {
        Assert.Equal(69, CudaEmbeddingAttentionPlan.ComputeType);
        Assert.Equal(77, CudaEmbeddingAttentionPlan.Tf32ComputeType);
        var parameter = typeof(CudaEmbeddingAttentionPlan).GetConstructors().Single().GetParameters().Single(p => p.Name == "computePolicy");
        Assert.True(parameter.IsOptional);
        Assert.Equal(0, Convert.ToInt32(parameter.DefaultValue));
        Assert.Equal(new[] { CudaEmbeddingAttentionComputePolicy.PedanticF32, CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage },
            Enum.GetValues<CudaEmbeddingAttentionComputePolicy>());
        Assert.False(CudaEmbeddingAttentionPlan.SupportsFastTf32(7));
        Assert.False(CudaEmbeddingAttentionPlan.SupportsFastTf32(-1));
        Assert.True(CudaEmbeddingAttentionPlan.SupportsFastTf32(8));
        Assert.True(CudaEmbeddingAttentionPlan.SupportsFastTf32(9));
    }

    [CudaTheory]
    [InlineData(1, 13, 256, 4, 2, 2, 1f, -1)]
    [InlineData(2, 29, 512, 4, 1, 1, 1f, -1)]
    [InlineData(1, 129, 512, 4, 1, 2, 1f, -1)]
    [InlineData(1, 514, 256, 4, 2, 2, 1f, 512)]
    [InlineData(1, 1025, 256, 4, 2, 2, 1f, 512)]
    [InlineData(1, 29, 512, 4, 1, 2, 80000f, -1)]
    public void ExplicitTf32NonzeroScoresTailsOffsetsAndWindowEdgesKeepOriginalDoubleGate(
        int batches, int tokens, int dimensions, int heads, int kvHeads, int tile, float magnitude, int radius)
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        using var allocator = new CudaAllocator();
        var qData = Values(batches + 2, tokens, heads, dimensions, .19317f, 1);
        var kData = Values(batches + 2, tokens, kvHeads, dimensions, .23791f, 1);
        var vData = Values(batches + 2, tokens, kvHeads, dimensions, .29371f, magnitude);
        using var qb = Tensor.FromArray(allocator, qData); using var kb = Tensor.FromArray(allocator, kData);
        using var vb = Tensor.FromArray(allocator, vData); using var yb = Tensor.FromArray(allocator, Fill(batches + 2, tokens, heads, dimensions, 37));
        using var q = qb.Narrow(0, 1, batches); using var k = kb.Narrow(0, 1, batches);
        using var v = vb.Narrow(0, 1, batches); using var y = yb.Narrow(0, 1, batches);
        using var plan = new CudaEmbeddingAttentionPlan(allocator, tile, computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        int[] lengths = Enumerable.Repeat(tokens, batches).ToArray(); const float scale = .7f;
        bool Run() => radius < 0 ? CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, lengths, scale) :
            CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, radius, scale);
        Assert.True(Run()); float[] actual = yb.GetElementsAsFloat((batches + 2) * tokens * heads * dimensions); plan.NotifyReadbackCompleted();
        Assert.Equal(77, plan.SelectedComputeType); Assert.Equal(CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage, plan.ComputePolicy);
        Assert.Equal(1, plan.SuccessfulCalls); Assert.Equal((long)batches * tokens * heads, plan.SuccessfulQueryRows);
        Assert.Contains(actual.Skip(tokens * heads * dimensions).Take(batches * tokens * heads * dimensions), value => value != 37);
        for (int batch = 1; batch <= batches; batch++)
        foreach (int token in new[] { 0, 1, tokens / 2, tokens - 2, tokens - 1 })
            Check(actual, qData, kData, vData, batch, token, 0, tokens, heads, kvHeads, dimensions, scale, magnitude, radius);
        int batchElements = tokens * heads * dimensions;
        for (int i = 0; i < batchElements; i++) { Assert.Equal(37, actual[i]); Assert.Equal(37, actual[(batches + 1) * batchElements + i]); }
        long capacity = plan.ScoreCapacityBytes;
        Assert.True(Run()); Assert.Equal(actual, yb.GetElementsAsFloat(actual.Length)); plan.NotifyReadbackCompleted();
        Assert.Equal(capacity, plan.ScoreCapacityBytes); Assert.Equal(2, plan.SuccessfulCalls);
    }

    [CudaFact]
    public void ExplicitTf32PreservesLargeQueryKeyExponentRangeWithinItsTf32RoundingBound()
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        const int tokens = 29, dimensions = 256, heads = 4, kvHeads = 2;
        using var allocator = new CudaAllocator();
        var qData = Values(1, tokens, heads, dimensions, .19317f, 80000);
        var kData = Values(1, tokens, kvHeads, dimensions, .23791f, 80000);
        var vData = Values(1, tokens, kvHeads, dimensions, .29371f, 1);
        using var q = Tensor.FromArray(allocator, qData); using var k = Tensor.FromArray(allocator, kData); using var v = Tensor.FromArray(allocator, vData);
        using var y = new Tensor(allocator, DType.Float32, 1, tokens, heads, dimensions);
        using var plan = new CudaEmbeddingAttentionPlan(allocator, computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        const float scale = 1e-9f;
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, [tokens], scale));
        float[] actual = y.GetElementsAsFloat(tokens * heads * dimensions); plan.NotifyReadbackCompleted();
        Assert.All(actual, value => Assert.True(float.IsFinite(value)));
        double worst = 0; int abovePedantic = 0;
        for (int token = 0; token < tokens; token++)
        {
            var (error, above) = Check(actual, qData, kData, vData, 0, token, 0, tokens, heads, kvHeads, dimensions, scale, 1, -1,
                LargeExponentTf32Bound, everyDimension: true);
            worst = Math.Max(worst, error); abovePedantic += above;
        }
        _output.WriteLine($"TF32 large-exponent attention: largest error {worst:E3} of {actual.Length} outputs (bound {LargeExponentTf32Bound:E1}); " +
            $"{abovePedantic} above the 3e-5 Double bound");
        Assert.Equal(77, plan.SelectedComputeType); Assert.Equal(1, plan.SuccessfulCalls);
    }

    [CudaFact]
    public void PackedUnequalSegmentsRemainIsolatedWithTf32AndInterleavedOutputGuards()
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        const int dimensions = 256, heads = 4, kvHeads = 2;
        int[] lengths = [13, 31, 514]; int tokens = lengths.Sum();
        using var allocator = new CudaAllocator();
        var qData = Values(1, tokens + 2, heads, dimensions, .19317f, 1);
        var kData = Values(1, tokens + 2, kvHeads, dimensions, .23791f, 1);
        var vData = Values(1, tokens + 2, kvHeads, dimensions, .29371f, 1);
        using var qb = Tensor.FromArray(allocator, qData); using var kb = Tensor.FromArray(allocator, kData); using var vb = Tensor.FromArray(allocator, vData);
        using var yb = Tensor.FromArray(allocator, Fill(1, tokens + 2, heads, dimensions, 37));
        using var q = qb.Narrow(1, 1, tokens); using var k = kb.Narrow(1, 1, tokens); using var v = vb.Narrow(1, 1, tokens); using var y = yb.Narrow(1, 1, tokens);
        using var plan = new CudaEmbeddingAttentionPlan(allocator, computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        Assert.True(CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, lengths, 512));
        float[] actual = yb.GetElementsAsFloat((tokens + 2) * heads * dimensions); plan.NotifyReadbackCompleted();
        int start = 1;
        foreach (int length in lengths)
        {
            foreach (int token in new[] { start, start + length / 2, start + length - 1 })
                Check(actual, qData, kData, vData, 0, token, start, length, heads, kvHeads, dimensions, 1, 1, 512);
            start += length;
        }
        for (int i = 0; i < heads * dimensions; i++) { Assert.Equal(37, actual[i]); Assert.Equal(37, actual[(tokens + 1) * heads * dimensions + i]); }
        Assert.Equal(3, plan.SuccessfulSegments); Assert.Equal(1, plan.MaskEntries); Assert.Equal(77, plan.SelectedComputeType);
    }

    [CudaFact]
    public void PreCanceledTf32AndInvalidShapeAliasBudgetRequestsDoNotAllocateOrWrite()
    {
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, 514, 4, 256, .19317f, 1));
        using var k = Tensor.FromArray(allocator, Values(1, 514, 2, 256, .23791f, 1));
        using var v = Tensor.FromArray(allocator, Values(1, 514, 2, 256, .29371f, 1));
        using var y = Tensor.FromArray(allocator, Fill(1, 514, 4, 256, 37));
        using var plan = new CudaEmbeddingAttentionPlan(allocator, computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        using var canceled = new CancellationTokenSource(); canceled.Cancel();
        Assert.Throws<OperationCanceledException>(() => CudaEmbeddingAttentionOps.TryWindowAttention(y, q, k, v, plan, [514], 512, cancellationToken: canceled.Token));
        Assert.Throws<OperationCanceledException>(() => CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, [514], cancellationToken: canceled.Token));
        Assert.Equal(0, plan.ScoreCapacityBytes); Assert.Equal(0, plan.MaskEntries); Assert.Equal(0, plan.SuccessfulCalls);
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, plan, [513]));
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(q, q, k, v, plan, [514]));
        using var tiny = new CudaEmbeddingAttentionPlan(allocator, scoreBudgetBytes: 64, computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        Assert.False(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, tiny, [514]));
        Assert.All(y.GetElementsAsFloat(514 * 4 * 256), value => Assert.Equal(37, value));
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaEmbeddingAttentionPlan(allocator, computePolicy: (CudaEmbeddingAttentionComputePolicy)2));
    }

    [CudaFact]
    public void Tf32PlanDisposalDoesNotChangePedanticPeerOrInvalidateCallerStorage()
    {
        Assert.Null(Environment.GetEnvironmentVariable("NVIDIA_TF32_OVERRIDE"));
        using var allocator = new CudaAllocator();
        using var q = Tensor.FromArray(allocator, Values(1, 29, 4, 256, .19317f, 1));
        using var k = Tensor.FromArray(allocator, Values(1, 29, 2, 256, .23791f, 1));
        using var v = Tensor.FromArray(allocator, Values(1, 29, 2, 256, .29371f, 1));
        using var y = new Tensor(allocator, DType.Float32, 1, 29, 4, 256);
        using var exact = new CudaEmbeddingAttentionPlan(allocator); using var approximate = new CudaEmbeddingAttentionPlan(allocator,
            computePolicy: CudaEmbeddingAttentionComputePolicy.FastTf32WithF32Storage);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, exact, [29])); var expected = y.GetElementsAsFloat(29 * 4 * 256);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, approximate, [29])); _ = y.GetElementsAsFloat(expected.Length);
        approximate.NotifyReadbackCompleted(); approximate.Dispose(); approximate.Dispose();
        Assert.Equal(0, approximate.ScoreCapacityBytes); Assert.Equal(0, approximate.MaskEntries);
        Assert.Equal(69, exact.SelectedComputeType); Assert.Equal(CudaEmbeddingAttentionComputePolicy.PedanticF32, exact.ComputePolicy);
        Assert.True(CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, exact, [29])); Assert.Equal(expected, y.GetElementsAsFloat(expected.Length));
        exact.NotifyReadbackCompleted(); Assert.Equal(2, exact.SuccessfulCalls);
        Assert.Throws<ObjectDisposedException>(() => CudaEmbeddingAttentionOps.TryGlobalAttention(y, q, k, v, approximate, [29]));
    }

    private static float[,,,] Values(int batches, int tokens, int heads, int dimensions, float frequency, float magnitude)
    {
        var values = new float[batches, tokens, heads, dimensions];
        for (int b = 0; b < batches; b++) for (int t = 0; t < tokens; t++) for (int h = 0; h < heads; h++) for (int d = 0; d < dimensions; d++)
            values[b, t, h, d] = magnitude * (.01317f + .21739f * MathF.Sin((b * 11 + t * 7 + h * 13 + d * 17) * frequency));
        return values;
    }
    private static float[,,,] Fill(int batches, int tokens, int heads, int dimensions, float value)
    { var data = new float[batches, tokens, heads, dimensions]; for (int b = 0; b < batches; b++) for (int t = 0; t < tokens; t++) for (int h = 0; h < heads; h++) for (int d = 0; d < dimensions; d++) data[b, t, h, d] = value; return data; }
    /// <summary>Checks a query row against the Double reference within bound x magnitude; returns its
    /// largest error and how many checked outputs exceed the 3e-5 Double bound.</summary>
    private static (double Worst, int AbovePedantic) Check(float[] actual, float[,,,] q, float[,,,] k, float[,,,] v, int batch, int token,
        int begin, int length, int heads, int kvHeads, int dimensions, float scale, float magnitude, int radius,
        double bound = 3e-5, bool everyDimension = false)
    {
        int total = q.GetLength(1), end = begin + length;
        int[] checkedDimensions = everyDimension ? Enumerable.Range(0, dimensions).ToArray() : [0, 1, dimensions / 2, dimensions - 2, dimensions - 1];
        double worst = 0; int abovePedantic = 0;
        int first = radius < 0 ? begin : Math.Max(begin, token - radius), last = radius < 0 ? end : Math.Min(end, token + radius + 1);
        for (int h = 0; h < heads; h++)
        {
            int kv = h / (heads / kvHeads); var scores = new double[last - first]; double max = double.NegativeInfinity;
            for (int key = first; key < last; key++)
            {
                double dot = 0; for (int d = 0; d < dimensions; d++) dot += (double)q[batch, token, h, d] * k[batch, key, kv, d];
                scores[key - first] = dot * scale; max = Math.Max(max, scores[key - first]);
            }
            double sum = 0; for (int key = 0; key < scores.Length; key++) sum += scores[key] = Math.Exp(scores[key] - max);
            foreach (int d in checkedDimensions)
            {
                double expected = 0; for (int key = first; key < last; key++) expected += scores[key - first] / sum * v[batch, key, kv, d];
                double error = Math.Abs(actual[((batch * total + token) * heads + h) * dimensions + d] - expected);
                Assert.InRange(error, 0, bound * magnitude);
                worst = Math.Max(worst, error); if (error > 3e-5 * magnitude) abovePedantic++;
            }
        }
        return (worst, abovePedantic);
    }
}
