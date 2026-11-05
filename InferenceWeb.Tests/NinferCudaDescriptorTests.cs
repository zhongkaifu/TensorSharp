// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

/// <summary>CPU pre-device contracts only; these descriptors do not contain a model fixture.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public sealed class NinferCudaDescriptorTests
{
    private sealed class UnreadSource : ICudaWeightSource
    {
        public int Calls;
        public void Read(long offset, IntPtr destination, long bytes)
        {
            Calls++;
            throw new InvalidOperationException("Metadata validation must not read fixture weights.");
        }
    }
    private static readonly string[] IncompatibleOptions = {
        "TS_Q35_VERIFY_DECODE_Q81", "TS_Q35_VERIFY_GRAPHS", "TS_Q35_CUDA_GRAPHS",
        "TS_Q35_GDN_PREPARED_PREFILL", "TS_Q35_SHARED_SCRATCH"
    };
    private sealed class Options : IDisposable
    {
        private readonly Dictionary<string, string?> _before = IncompatibleOptions.ToDictionary(x => x, Environment.GetEnvironmentVariable);
        public Options() { foreach (string key in IncompatibleOptions) Environment.SetEnvironmentVariable(key, null); }
        public void Dispose() { foreach (var entry in _before) Environment.SetEnvironmentVariable(entry.Key, entry.Value); }
    }
    private static CudaNinferProjectionDesc Matrix(UnreadSource source, int rows, int columns,
        CudaNinferProjectionFormat format = CudaNinferProjectionFormat.NvFp4Planes)
    {
        long n = (long)rows * columns;
        long bytes = format switch
        {
            CudaNinferProjectionFormat.NvFp4Planes => n / 2 + n / 16,
            CudaNinferProjectionFormat.BFloat16 => n * 2,
            CudaNinferProjectionFormat.W8G32 => n + n / 16,
            CudaNinferProjectionFormat.Fp8Row => n + (long)rows * 2,
            _ => throw new ArgumentOutOfRangeException(nameof(format))
        };
        return new(source, 0, bytes, rows, columns, format);
    }
    private static Q35CudaEngine.ModelDesc Target(UnreadSource source)
    {
        float[] norm = Enumerable.Repeat(1f, 5120).ToArray(), qkNorm = Enumerable.Repeat(1f, 256).ToArray();
        var recipe = new CudaNinferModelDesc
        {
            Embedding = Matrix(source, 248320, 5120, CudaNinferProjectionFormat.Fp8Row),
            Head = Matrix(source, 248320, 5120, CudaNinferProjectionFormat.Fp8Row),
            Layers = new CudaNinferLayerDesc[64], ModulePath = Path.GetFullPath("cpu-only-not-created-body.ptx"),
            Provenance = "cpu-only-metadata-contract"
        };
        var model = new Q35CudaEngine.ModelDesc
        {
            NLayer = 64, NEmbd = 5120, NFf = 17408, NVocab = 248320,
            NHead = 24, NKvHead = 4, HeadDim = 256, GdnKHeads = 16, GdnVHeads = 48, DConv = 4,
            NCtx = 8192, NUbatch = 512, SectT = 11, SectH = 11, SectW = 10,
            Eps = 1e-6f, RopeBase = 10000000, Ninfer = recipe, OutputNorm = norm, Layers = new Q35CudaEngine.LayerDesc[64]
        };
        for (int i = 0; i < 64; i++)
        {
            bool recurrent = (i + 1) % 4 != 0;
            bool fp8Mlp = i >= 56;
            model.Layers[i] = new Q35CudaEngine.LayerDesc
            {
                Recurrent = recurrent, AttnNorm = norm, PostNorm = norm,
                QNorm = recurrent ? null : qkNorm, KNorm = recurrent ? null : qkNorm,
                ConvW = recurrent ? new float[40960] : null,
                DtBias = recurrent ? new float[48] : null, SsmA = recurrent ? new float[48] : null,
                SsmNorm = recurrent ? Enumerable.Repeat(1f, 128).ToArray() : null
            };
            recipe.Layers[i] = new CudaNinferLayerDesc
            {
                Input = Matrix(source, recurrent ? 16384 : 14336, 5120, CudaNinferProjectionFormat.Fp8Row),
                Output = Matrix(source, 5120, 6144, CudaNinferProjectionFormat.Fp8Row),
                GateUp = Matrix(source, 34816, 5120, fp8Mlp ? CudaNinferProjectionFormat.Fp8Row : CudaNinferProjectionFormat.NvFp4Planes),
                Down = Matrix(source, 5120, 17408, fp8Mlp ? CudaNinferProjectionFormat.Fp8Row : CudaNinferProjectionFormat.NvFp4Planes),
                AlphaBeta = recurrent ? Matrix(source, 96, 5120, CudaNinferProjectionFormat.BFloat16) : null
            };
        }
        return model;
    }

    [Theory]
    [InlineData(CudaNinferProjectionFormat.NvFp4Planes, 128, 64, 4608)]
    [InlineData(CudaNinferProjectionFormat.BFloat16, 96, 5120, 983040)]
    [InlineData(CudaNinferProjectionFormat.W8G32, 3, 128, 408)]
    public void ExactStorageContractsBorrowSourcesWithoutReading(CudaNinferProjectionFormat format, int rows, int columns, long bytes)
    {
        var source = new UnreadSource();
        var d = new CudaNinferProjectionDesc(source, 17, bytes, rows, columns, format, 2.5f, 3.5f, false);
        Assert.Same(source, d.Source); Assert.Equal(17, d.SourceOffset); Assert.Equal(bytes, d.Bytes);
        Assert.Equal(2.5f, d.InputScaleDivisor); Assert.Equal(3.5f, d.WeightScaleDivisor); Assert.False(d.AllowA4);
        Assert.Equal(0, source.Calls);
        Assert.Throws<ArgumentException>(() => new CudaNinferProjectionDesc(source, 0, bytes - 1, rows, columns, format));
        Assert.Throws<ArgumentException>(() => new CudaNinferProjectionDesc(source, 0, bytes + 1, rows, columns, format));
    }

    [Fact]
    public void ProjectionRefusesSourceRangeOverflowAndUnsupportedQuantizedGeometry()
    {
        var source = new UnreadSource();
        Assert.Throws<ArgumentOutOfRangeException>(() => new CudaNinferProjectionDesc(source, long.MaxValue - 1, 4608, 128, 64, CudaNinferProjectionFormat.NvFp4Planes));
        Assert.Throws<NotSupportedException>(() => new CudaNinferProjectionDesc(source, 0, 456, 127, 64, CudaNinferProjectionFormat.NvFp4Planes));
        Assert.Throws<NotSupportedException>(() => new CudaNinferProjectionDesc(source, 0, 456, 128, 63, CudaNinferProjectionFormat.NvFp4Planes));
        Assert.Throws<NotSupportedException>(() => new CudaNinferProjectionDesc(source, 0, 456, 3, 127, CudaNinferProjectionFormat.W8G32));
        Assert.Equal(0, source.Calls);
    }

    [Theory]
    [InlineData(0f, 1f)] [InlineData(-1f, 1f)] [InlineData(float.NaN, 1f)]
    [InlineData(float.PositiveInfinity, 1f)] [InlineData(1f, 0f)] [InlineData(1f, float.NegativeInfinity)]
    public void InvalidDivisorsFailBeforeWeightsOrDeviceAreTouched(float input, float weight)
        => Assert.Throws<ArgumentException>(() => new CudaNinferProjectionDesc(new UnreadSource(), 0, 4608, 128, 64,
            CudaNinferProjectionFormat.NvFp4Planes, input, weight));

    [Theory]
    [InlineData(float.MaxValue, 2f)] [InlineData(float.Epsilon, 1f)]
    [InlineData(1f, float.Epsilon)] [InlineData(float.Epsilon, float.Epsilon)]
    public void FiniteDivisorsCannotYieldZeroOrNonfiniteDispatchScales(float input, float weight)
        => Assert.Throws<ArgumentException>(() => new CudaNinferProjectionDesc(new UnreadSource(), 0, 4608, 128, 64,
            CudaNinferProjectionFormat.NvFp4Planes, input, weight));

    [Fact]
    public void KnownTargetMetadataPassesWithoutOpeningAnyModuleOrReadingWeights()
    {
        using var options = new Options(); var source = new UnreadSource(); var model = Target(source);
        Q35CudaEngine.ValidateNinferModel(model); Assert.Equal(0, source.Calls);
        Assert.False(File.Exists(model.Ninfer.ModulePath));
        Assert.Equal(8, model.Ninfer.Layers.Count(l => l.GateUp.Format == CudaNinferProjectionFormat.Fp8Row));
    }

    [Fact]
    public void TargetRecipeRejectsMissingOrCrossFamilyGeometryWithoutCudaInitialization()
    {
        using var options = new Options(); var source = new UnreadSource();
        var m = Target(source); m.NVocab = 248077; Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.NHead = 32; Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Ninfer.Layers[0] = null; Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Layers[3].Recurrent = true; Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Ninfer.Layers[0].Input = Matrix(source, 14336, 5120); Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Ninfer.Layers[0].AlphaBeta = null; Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Ninfer.Head = Matrix(source, 248320, 5120, CudaNinferProjectionFormat.W8G32); Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateNinferModel(m));
        m = Target(source); m.Ninfer.Layers[3].AlphaBeta = Matrix(source, 96, 5120, CudaNinferProjectionFormat.BFloat16); Assert.Throws<ArgumentException>(() => Q35CudaEngine.ValidateNinferModel(m));
        Assert.Equal(0, source.Calls);
    }

    [Theory]
    [InlineData("TS_Q35_VERIFY_DECODE_Q81")] [InlineData("TS_Q35_VERIFY_GRAPHS")]
    [InlineData("TS_Q35_CUDA_GRAPHS")] [InlineData("TS_Q35_GDN_PREPARED_PREFILL")]
    [InlineData("TS_Q35_SHARED_SCRATCH")]
    public void GgufExperimentalPoliciesCannotSilentlyAlterTheNinferRecipe(string key)
    {
        using var options = new Options(); Environment.SetEnvironmentVariable(key, "1");
        Assert.Throws<NotSupportedException>(() => Q35CudaEngine.ValidateNinferModel(Target(new UnreadSource())));
    }
}
