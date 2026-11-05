// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Cuda;

namespace InferenceWeb.Tests;

public sealed class NinferFlashInferBodyIntegrationPolicyTests
{
    private sealed class Source(int rows, int columns, CudaNinferOriginalRowMapping mapping) : ICudaNinferOriginalProjectionSource
    {
        public long OriginalBytes { get; set; } = (long)rows * columns / 2 + (long)rows * columns / 16 + 4;
        public int Rows { get; set; } = rows;
        public int Columns { get; set; } = columns;
        public CudaNinferOriginalRowMapping RowMapping { get; set; } = mapping;
        public void Read(long offset, IntPtr dst, long bytes) => throw new InvalidOperationException("Planning must not read payloads.");
        public void ReadOriginal(long offset, IntPtr dst, long bytes) => throw new InvalidOperationException("Planning must not read payloads.");
    }
    private sealed class PlanesOnly : ICudaWeightSource
    {
        public void Read(long offset, IntPtr dst, long bytes) => throw new InvalidOperationException();
    }
    private static CudaNinferProjectionDesc Descriptor(ICudaWeightSource source, int n, int k,
        CudaNinferProjectionFormat format = CudaNinferProjectionFormat.NvFp4Planes, long offset = 0)
        => new(source, offset, format == CudaNinferProjectionFormat.NvFp4Planes ? (long)n * k / 2 + (long)n * k / 16 : (long)n * k * 2,
            n, k, format);

    [Fact]
    public void ActualV3DirectoryAdmitsExactlyItsNvfp4MlpMatricesWithoutReadingPayloads()
    {
        using var fixture = typeof(NinferFlashInferBodyIntegrationPolicyTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.TestData.Ninfer.qwen38-nvfp4-v3-directory.json")!;
        using var memory = new MemoryStream(); fixture.CopyTo(memory); byte[] raw = memory.ToArray();
        using var document = JsonDocument.Parse(raw);
        // Object id -> its first text binding (fused objects bind several row ranges).
        var names = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var binding in document.RootElement.GetProperty("bindings").EnumerateObject())
        {
            if (!binding.Name.StartsWith("text/", StringComparison.Ordinal)) continue;
            string id = binding.Value.TryGetProperty("object", out var whole) ? whole.GetString()!
                : binding.Value.GetProperty("parts")[0].GetProperty("object").GetString()!;
            names.TryAdd(id, binding.Name);
        }
        var counts = new Dictionary<(int, int), int>(); long bytes = 0;
        foreach (var tensor in document.RootElement.GetProperty("objects").EnumerateArray())
        {
            if (tensor.GetProperty("kind").GetString() != "tensor" || tensor.GetProperty("format").GetString() != "nvfp4") continue;
            string name = names[tensor.GetProperty("id").GetString()!];
            int n = tensor.GetProperty("shape")[0].GetInt32(), k = tensor.GetProperty("shape")[1].GetInt32();
            var mapping = name.EndsWith("/mlp/gate", StringComparison.Ordinal) ? CudaNinferOriginalRowMapping.GateUpRows8
                : CudaNinferOriginalRowMapping.Identity;
            var source = new Source(n, k, mapping);
            Assert.True(Q35CudaEngine.TryPlanNinferFlashInferProjection(Descriptor(source, n, k), out var plan));
            Assert.Equal(tensor.GetProperty("bytes").GetInt64(), plan.OriginalBytes);
            Assert.Equal(mapping, plan.Mapping); Assert.Equal(source.OriginalBytes, plan.CodeBytes + plan.ScaleBytes + 4);
            counts[(n, k)] = counts.GetValueOrDefault((n, k)) + 1; bytes += plan.OriginalBytes;
        }
        Assert.Equal(112, counts.Values.Sum());
        Assert.Equal(56, counts[(34816, 5120)]); Assert.Equal(56, counts[(5120, 17408)]);
        Assert.Equal(56L * (100270084 + 50135044), bytes);
    }

    [Theory]
    [InlineData(16384, 5120)] // GDN input remains the ordinary A4 route.
    [InlineData(96, 5120)] // Alpha/beta is never an FI role.
    [InlineData(5120, 5120)] // Draft projection geometry must not enter the target route.
    public void UnqualifiedGeometryIsExcluded(int n, int k)
        => Assert.False(Q35CudaEngine.TryPlanNinferFlashInferProjection(Descriptor(new Source(n, k, CudaNinferOriginalRowMapping.Identity), n, k,
            n == 96 ? CudaNinferProjectionFormat.BFloat16 : CudaNinferProjectionFormat.NvFp4Planes), out _));

    [Fact]
    public void Bfloat16ExceptionsAreExcludedEvenWhenShapeMatches()
        => Assert.False(Q35CudaEngine.TryPlanNinferFlashInferProjection(Descriptor(new PlanesOnly(), 14336, 5120, CudaNinferProjectionFormat.BFloat16), out _));

    [Fact]
    public void PlanesOnlySourceCannotSilentlyBecomeRaw()
        => Assert.Throws<NotSupportedException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(Descriptor(new PlanesOnly(), 5120, 6144), out _));

    [Fact]
    public void ForeignGeometryMappingOffsetAndLengthAreRejectedBeforeAnyRead()
    {
        var source = new Source(34816, 5120, CudaNinferOriginalRowMapping.GateUpRows8);
        var descriptor = Descriptor(source, 34816, 5120);
        source.RowMapping = CudaNinferOriginalRowMapping.Identity;
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(descriptor, out _));
        source.RowMapping = CudaNinferOriginalRowMapping.GateUpRows8; source.Rows = 34944;
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(descriptor, out _));
        source.Rows = 34816; source.Columns = 6144;
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(descriptor, out _));
        source.Columns = 5120; source.OriginalBytes--;
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(descriptor, out _));
        source.OriginalBytes++;
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.TryPlanNinferFlashInferProjection(Descriptor(source, 34816, 5120, offset: 4), out _));
        Assert.True(Q35CudaEngine.TryPlanNinferFlashInferProjection(descriptor, out var plan));
        Assert.Equal(Q35CudaEngine.NinferFlashInferPlaneScratchBytes, plan.CodeBytes + plan.ScaleBytes);
    }

    [Theory]
    [InlineData(false, false, false, false, true)]
    [InlineData(false, true, false, true, true)] // Ordinary exact verify remains independent.
    [InlineData(false, true, true, false, false)] // Orphan bridge is a configuration error.
    [InlineData(true, false, true, false, false)]
    [InlineData(true, true, false, false, false)]
    [InlineData(true, true, true, true, false)]
    [InlineData(true, true, true, false, true)]
    public void ExplicitPolicyAdmissionPrecedesCudaAllocation(bool enabled, bool ninfer, bool bridge, bool exact, bool accepted)
    {
        Exception? error = Record.Exception(() => Q35CudaEngine.ValidateNinferFlashInferFlags(enabled, ninfer, bridge, exact));
        Assert.Equal(accepted, error == null);
    }

    [Fact]
    public void EveryRuntimeImageAndRawPayloadParticipatesInStateIdentity()
    {
        string[] hashes = Enumerable.Range(1, 5).Select(x => x.ToString("x").PadLeft(64, '0')).ToArray();
        string original = Q35CudaEngine.BuildNinferFlashInferStateSuffix("test-policy", hashes);
        for (int i = 0; i < hashes.Length; i++)
        {
            string[] changed = hashes.ToArray(); changed[i] = new string('a', 64);
            Assert.NotEqual(original, Q35CudaEngine.BuildNinferFlashInferStateSuffix("test-policy", changed));
        }
        Assert.NotEqual(original, Q35CudaEngine.BuildNinferFlashInferStateSuffix("another-policy", hashes));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferFlashInferStateSuffix("bad:policy", hashes));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferFlashInferStateSuffix("test-policy", hashes[..4]));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.BuildNinferFlashInferStateSuffix("test-policy", new[] { new string('g', 64), hashes[1], hashes[2], hashes[3], hashes[4] }));
    }

    [Fact]
    public void ExplicitPathsNeverFallBackFromMissingRelativeOrUnnormalizedNames()
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid() + ".json"); File.WriteAllText(path, "{}");
        try
        {
            Assert.Null(Q35CudaEngine.ResolveNinferFlashInferPath(null, "option", false));
            Assert.Equal(path, Q35CudaEngine.ResolveNinferFlashInferPath(path, "option", true));
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferFlashInferPath(null, "option", true));
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferFlashInferPath("module.json", "option", true));
            Assert.Throws<ArgumentException>(() => Q35CudaEngine.ResolveNinferFlashInferPath(Path.GetDirectoryName(path) + "/./" + Path.GetFileName(path), "option", true));
            Assert.Throws<FileNotFoundException>(() => Q35CudaEngine.ResolveNinferFlashInferPath(path + ".missing", "option", true));
        }
        finally { File.Delete(path); }
    }
}
