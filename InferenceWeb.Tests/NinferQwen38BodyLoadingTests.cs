// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Format/bootstrap/layout gates. These do not claim a GPU or full-model validation.</summary>
public sealed class NinferQwen38BodyLoadingTests
{
    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.Mlx)]
    public void FactoryRejectsUnsupportedBackendBeforeOpeningOrInitializingADevice(BackendType backend)
        => Assert.Throws<NotSupportedException>(() => ModelBase.Create("/not-created/body.ninfer", backend));

    [Fact]
    public void FactoryRejectsUnsupportedParallelAndDraftModesBeforeOpening()
    {
        Assert.Throws<NotSupportedException>(() => ModelBase.Create("/not-created/body.ninfer", BackendType.Cuda, tpDegree: 2));
        Assert.Throws<NotSupportedException>(() => ModelBase.Create("/not-created/body.ninfer", BackendType.Cuda, layerSplitDegree: 2));
        Assert.Throws<NotSupportedException>(() => ModelBase.Create("/not-created/body.ninfer", BackendType.Cuda, draftModelPath: "/draft.gguf"));
    }

    [Theory]
    [InlineData(1)] [InlineData(2)] [InlineData(3)]
    public void FormatDetectionDoesNotDependOnTheFilenameAndPreservesVersionRefusal(int version)
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, new byte[] { (byte)'N', (byte)'I', (byte)'N', (byte)'F', (byte)'E', (byte)'R', 0, (byte)version });
            Assert.True(NinferQwen38ModelSource.IsNinferArtifact(path));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public void OtherFormatsAreLeftToTheirOwnParser()
    {
        string path = Path.GetTempFileName();
        try { File.WriteAllText(path, "GGUF-not-a-ninfer"); Assert.False(NinferQwen38ModelSource.IsNinferArtifact(path)); }
        finally { File.Delete(path); }
        Assert.True(NinferQwen38ModelSource.IsNinferArtifact("/not-created/BODY.NINFER"));
    }

    [Fact]
    public void KnownGgufMagicOverridesANinferFilenameHint()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-format-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        string path = Path.Combine(directory, "renamed.ninfer");
        try
        {
            File.WriteAllBytes(path, new byte[] { (byte)'G', (byte)'G', (byte)'U', (byte)'F', 3, 0, 0, 0 });
            Assert.False(NinferArtifactReader.IsNinferArtifact(path));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Theory]
    [InlineData(null, 65536)] [InlineData("", 65536)] [InlineData("131072", 131072)]
    [InlineData("256000", 256000)] [InlineData("262144", 262144)] [InlineData("1", 1)]
    public void ContextKeepsTheArtifactBoundAndAConservativeUnspecifiedDefault(string? value, int expected)
        => Assert.Equal(expected, Qwen35CudaModel.ResolveNinferContext(value, 262144));

    [Theory]
    [InlineData("0")] [InlineData("-1")] [InlineData("262145")] [InlineData("garbage")] [InlineData("999999999999")]
    public void InvalidContextDoesNotSilentlyFallback(string value)
        => Assert.Throws<ArgumentException>(() => Qwen35CudaModel.ResolveNinferContext(value, 262144));

    [Fact]
    public void RawBfloatParametersPreserveBitsAndNormOffsetsDoNotRoundBackToBfloat()
    {
        byte[] raw = { 0x80, 0x3b, 0x00, 0x80, 0x80, 0xbf };
        float[] values = NinferQwen38ModelSource.DecodePlainFloats(raw, NinferTensorFormat.BFloat16);
        Assert.Equal(new[] { 0x3b800000, unchecked((int)0x80000000), unchecked((int)0xbf800000) }, values.Select(BitConverter.SingleToInt32Bits));
        float[] offset = NinferQwen38ModelSource.EffectiveNorm(values, NinferQwen38NormConvention.StoredUnitOffset);
        Assert.Equal(1.00390625f, offset[0]); // Rounding the effective gamma to BF16 would lose this half-ULP.
        Assert.Equal(0f, offset[2]);
        Assert.Equal(0x3b800000, BitConverter.SingleToInt32Bits(values[0]));
        Assert.Equal(values.Select(BitConverter.SingleToInt32Bits),
            NinferQwen38ModelSource.EffectiveNorm(values, NinferQwen38NormConvention.StoredPlain).Select(BitConverter.SingleToInt32Bits));
        Assert.Throws<InvalidDataException>(() => NinferQwen38ModelSource.EffectiveNorm(values, NinferQwen38NormConvention.NotNorm));
    }

    [Fact]
    public void PlainParameterFailuresDoNotSubstituteOrClipValues()
    {
        Assert.Throws<InvalidDataException>(() => NinferQwen38ModelSource.DecodePlainFloats(new byte[] { 0 }, NinferTensorFormat.BFloat16));
        Assert.Throws<InvalidDataException>(() => NinferQwen38ModelSource.DecodePlainFloats(new byte[] { 0x80, 0x7f }, NinferTensorFormat.BFloat16));
        Assert.Throws<NotSupportedException>(() => NinferQwen38ModelSource.DecodePlainFloats(new byte[4], NinferTensorFormat.NvFp4));
    }

    [Fact]
    public void ConvolutionTransposesTapsWithoutChangingValues()
    {
        float[] source = { 11, 12, 13, 21, 22, 23, 31, 32, 33, 41, 42, 43 };
        Assert.Equal(new float[] { 11, 21, 31, 41, 12, 22, 32, 42, 13, 23, 33, 43 },
            NinferQwen38ModelSource.ChannelMajorConvolution(source, 3, 4));
        Assert.Equal(12, source[1]);
        Assert.Throws<ArgumentException>(() => NinferQwen38ModelSource.ChannelMajorConvolution(source, 2, 4));
    }

    [Theory]
    [InlineData(0u)] [InlineData(0x80000000u)] [InlineData(0xbf800000u)]
    [InlineData(0x7f800000u)] [InlineData(0xff800000u)] [InlineData(0x7fc01234u)]
    public void InvalidActivationDivisorsCannotReachACudaDescriptor(uint bits)
        => Assert.Throws<InvalidDataException>(() => Qwen35CudaExecutor.PositiveDivisor(bits));

    [Theory]
    [InlineData(1u)] [InlineData(0x3f800000u)] [InlineData(0x40200000u)] [InlineData(0x7f7fffffu)]
    public void PositiveCalibrationRetainsExactSourceBits(uint bits)
        => Assert.Equal(bits, unchecked((uint)BitConverter.SingleToInt32Bits(Qwen35CudaExecutor.PositiveDivisor(bits))));

    [Fact]
    public void ModuleResolutionFindsPackagedCudaAssetsAndPreservesAppBasePriority()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-ninfer-module-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(directory, "cuda_kernels"));
        const string filename = "fixture.ptx";
        string root = Path.Combine(directory, filename), packaged = Path.Combine(directory, "cuda_kernels", filename);
        try
        {
            File.WriteAllText(packaged, "CPU-resolution-fixture-only");
            Assert.Equal(packaged, Qwen35CudaExecutor.ResolveNinferModulePath("FIXTURE_MODULE", filename, null, directory));
            File.WriteAllText(root, "CPU-resolution-priority-fixture-only");
            Assert.Equal(root, Qwen35CudaExecutor.ResolveNinferModulePath("FIXTURE_MODULE", filename, null, directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    [Fact]
    public void ExplicitModuleSettingDoesNotFallbackToPackagedAssets()
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-ninfer-explicit-module-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(directory, "cuda_kernels"));
        const string filename = "fixture.ptx";
        try
        {
            File.WriteAllText(Path.Combine(directory, "cuda_kernels", filename), "CPU-resolution-fixture-only");
            string configured = Path.Combine(directory, "explicitly-missing.ptx");
            var missing = Assert.Throws<FileNotFoundException>(() =>
                Qwen35CudaExecutor.ResolveNinferModulePath("FIXTURE_MODULE", filename, configured, directory));
            Assert.Equal(configured, missing.FileName);
            Assert.Throws<ArgumentException>(() =>
                Qwen35CudaExecutor.ResolveNinferModulePath("FIXTURE_MODULE", filename, " ", directory));
            string explicitFile = Path.Combine(directory, "chosen.fatbin");
            File.WriteAllText(explicitFile, "CPU-explicit-resolution-fixture-only");
            Assert.Equal(explicitFile, Qwen35CudaExecutor.ResolveNinferModulePath("FIXTURE_MODULE", filename, explicitFile, directory));
        }
        finally { Directory.Delete(directory, true); }
    }

    private sealed class TensorFixture : IDisposable
    {
        private readonly NinferV3File _file;
        internal NinferArtifactReader Reader { get; }
        internal NinferTensorDescriptor Tensor => Reader.GetBinding("weights").Object;
        internal TensorFixture(string format, int rows, int columns, byte[] payload)
        {
            var b = new NinferV3Builder();
            b.Bind("weights", b.TensorBytes(format, new[] { (ulong)rows, (ulong)columns }, payload));
            b.Resource("text", "unrelated", new byte[] { 0x55, 0x66 }); // Not part of the tensor.
            _file = b.Write();
            Reader = new NinferArtifactReader(_file.Path);
        }
        public void Dispose() { Reader.Dispose(); _file.Dispose(); }
    }

    [Fact]
    public void VirtualNvfp4PlanesCrossBoundariesWithoutLeakingDivisorBytes()
    {
        const int rows = 128, columns = 64, codeBytes = rows * columns / 2, scaleBytes = rows * columns / 16;
        var encoded = new byte[codeBytes + scaleBytes + 4];
        for (int i = 0; i < codeBytes; i++) encoded[i] = (byte)i;
        for (int i = 0; i < scaleBytes; i++) encoded[codeBytes + i] = (byte)(i % 127);
        BinaryPrimitives.WriteUInt32LittleEndian(encoded.AsSpan(codeBytes + scaleBytes), 0x40200000);
        using var fixture = new TensorFixture("nvfp4", rows, columns, encoded);
        using var source = new NinferNvfp4WeightSource(new NinferNvfp4PlaneReader(fixture.Reader, fixture.Tensor));
        using var independentTileReader = new NinferNvfp4PlaneReader(fixture.Reader, fixture.Tensor);
        var tile = independentTileReader.ReadTile(0);
        byte[] expected = tile.Codes.ToArray().Concat(tile.Scales.ToArray()).ToArray();
        Assert.Equal(expected, Read(source, 0, expected.Length));
        Assert.Equal(expected.AsSpan(codeBytes - 3, 8).ToArray(), Read(source, codeBytes - 3, 8));
        Assert.Throws<ArgumentOutOfRangeException>(() => Read(source, expected.Length, 1));
        source.Dispose();
        Assert.Throws<ObjectDisposedException>(() => source.Read(0, IntPtr.Zero, 0));
        fixture.Reader.ReadObjectRange(fixture.Tensor, 0, Span<byte>.Empty); // Does not own the original artifact.
    }

    private static byte[] Read(TensorSharp.Cuda.ICudaWeightSource source, long offset, int bytes)
    {
        IntPtr buffer = Marshal.AllocHGlobal(bytes + 2);
        try
        {
            Marshal.WriteByte(buffer, 0x91); Marshal.WriteByte(buffer, bytes + 1, 0x92);
            source.Read(offset, buffer + 1, bytes);
            var result = new byte[bytes]; Marshal.Copy(buffer + 1, result, 0, bytes);
            Assert.Equal(0x91, Marshal.ReadByte(buffer)); Assert.Equal(0x92, Marshal.ReadByte(buffer, bytes + 1));
            return result;
        }
        finally { Marshal.FreeHGlobal(buffer); }
    }
}
