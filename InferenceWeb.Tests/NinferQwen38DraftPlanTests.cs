// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>Original directory/schema/loading checks only; no native draft inference or quality claim.</summary>
public sealed class NinferQwen38DraftPlanTests
{
    private const long Payload = 212992, FileBytes = 19406942468;
    private static readonly NinferArtifactIdentity Identity = new("qwen3.8-27b", "nvfp4full");
    private static byte[] DirectoryBytes()
    {
        using var stream = typeof(NinferQwen38DraftPlanTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.TestData.Ninfer.qwen38-nvfp4full-dflash2-directory.json")!;
        using var bytes = new MemoryStream(); stream.CopyTo(bytes); byte[] result = bytes.ToArray();
        Assert.Equal("0be5ebb894fb71e5845e5c4e7704295f10fc36c12cf8499be1c585eed815b5cb", Convert.ToHexString(SHA256.HashData(result)).ToLowerInvariant());
        return result;
    }

    // Shapes/offsets are public actual directory facts; the divisor2.5 is synthetic payload metadata.
    private static List<NinferQwen38TensorMetadata> Metadata(bool shortlist = true)
    {
        using var document = JsonDocument.Parse(DirectoryBytes()); var root = document.RootElement;
        Assert.Equal("222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658", root.GetProperty("provenance").GetProperty("directory_sha256").GetString());
        Assert.Equal(68, root.GetProperty("objects").GetArrayLength());
        var result = new List<NinferQwen38TensorMetadata>();
        foreach (var row in root.GetProperty("objects").EnumerateArray())
        {
            string name = row.GetProperty("name").GetString()!;
            if (!shortlist && name.StartsWith("text/", StringComparison.Ordinal)) continue;
            var format = row.GetProperty("format").GetString() switch
            {
                "NVFP4" => NinferTensorFormat.NvFp4, "BF16" => NinferTensorFormat.BFloat16,
                "I32" => NinferTensorFormat.Int32, "Q4G64_F16S" => NinferTensorFormat.Q4G64Float16Scale,
                _ => throw new InvalidDataException("Unexpected fixture format")
            };
            var layout = row.GetProperty("layout").GetString() switch
            {
                "blockscale-k16-m128x4-v1" => NinferTensorLayout.BlockScaleK16M128x4V1,
                "row-split-k128-v1" => NinferTensorLayout.RowSplitK128V1,
                "contiguous-le-v1" => NinferTensorLayout.ContiguousLittleEndianV1,
                _ => throw new InvalidDataException("Unexpected fixture layout")
            };
            ulong[] shape = row.GetProperty("shape").EnumerateArray().Select(x => x.GetUInt64()).ToArray();
            ulong count = shape.Aggregate(1UL, (a, b) => a * b);
            // Independent directory-format arithmetic, without using draft-plan geometry helpers.
            var geometry = format switch
            {
                NinferTensorFormat.NvFp4 => new NinferTensorGeometry(count / 2 + count / 16 + 4, 256, LowPlaneBytes: count / 2,
                    ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 16, DivisorOffset: count / 2 + count / 16),
                NinferTensorFormat.Q4G64Float16Scale => new(count / 2 + count / 32, 256, PaddedK: 5120, GroupSize: 64,
                    LowPlaneBytes: count / 2, HighPlaneOffset: count / 2, HighPlaneBytes: 0, ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 32),
                _ => new(count * (format == NinferTensorFormat.BFloat16 ? 2UL : 4UL), 256)
            };
            ulong offset = row.GetProperty("offset").GetUInt64();
            result.Add(new(name, format, layout, Array.AsReadOnly(shape), offset, Payload + (long)offset,
                row.GetProperty("bytes").GetUInt64(), geometry, format == NinferTensorFormat.NvFp4 ? 2.5f : null,
                format == NinferTensorFormat.NvFp4 ? 0x40200000u : null));
        }
        return result;
    }
    private static IReadOnlyList<NinferQwen38DraftSpec> Validate(List<NinferQwen38TensorMetadata> tensors)
        => NinferQwen38DraftPlan.ValidateMetadata(Identity, tensors, Payload, FileBytes);

    [Theory]
    [InlineData(false, 66)]
    [InlineData(true, 68)]
    public void ActualDirectoryBindsOriginalModuleAndExplicitOptionalHead(bool shortlist, int count)
    {
        var metadata = Metadata(shortlist); var specs = Validate(metadata);
        Assert.Equal(count, specs.Count); Assert.Equal(34, specs.Count(x => x.Format == NinferTensorFormat.NvFp4));
        Assert.Equal(32, specs.Count(x => x.Format == NinferTensorFormat.BFloat16));
        Assert.Equal(1082862216UL, metadata.Where(x => x.Name.StartsWith("dflash2/", StringComparison.Ordinal)).Aggregate(0UL, (a, b) => a + b.ByteLength));
        Assert.Equal(22, specs.Count(x => x.PlainNorm));
        Assert.Equal(5, specs.Count(x => x.Mapping == NinferNvfp4RowMapping.GateUpRows8));
        Assert.DoesNotContain(specs, x => x.Mapping == NinferNvfp4RowMapping.AttentionQueryGateKeyValue);
        Assert.Throws<NotSupportedException>(() => ((IList<NinferQwen38DraftSpec>)specs).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<ulong>)specs[0].Shape)[0] = 1);
    }

    [Fact]
    public void DraftContractKeepsOutputTapsFullRopeBf16RingAndRawConvOrder()
    {
        Assert.Equal(new[] { 5, 19, 33, 47, 61 }, NinferQwen38DraftPlan.TargetOutputLayers);
        Assert.Equal(NinferQwen38DraftPlan.TargetOutputLayers, CudaNinferDraftDesc.TargetOutputLayers);
        Assert.Equal(25600, NinferQwen38DraftPlan.FeatureWidth); Assert.Equal(2048, NinferQwen38DraftPlan.RingPositions);
        Assert.Equal(41943040, NinferQwen38DraftPlan.KeyValueBytesPerSequence);
        Assert.Equal(32, NinferQwen38DraftPlan.QueryHeads); Assert.Equal(8, NinferQwen38DraftPlan.KeyValueHeads);
        Assert.Equal(128, NinferQwen38DraftPlan.HeadDimension); Assert.Equal(248070, NinferQwen38DraftPlan.MaskToken);
        Assert.Equal(7, NinferQwen38DraftPlan.MaximumDraftTokens); Assert.Equal(16, NinferQwen38DraftPlan.TopK);
        Assert.Equal(1e7f, NinferQwen38DraftPlan.RopeTheta); Assert.Equal(1e-6f, NinferQwen38DraftPlan.RmsEpsilon);
        var specs = Validate(Metadata());
        Assert.All(specs.Where(x => x.Role is NinferQwen38DraftRole.AttentionConvBase or NinferQwen38DraftRole.MlpConvBase),
            x => Assert.Equal(new ulong[] { 2, 2, 5120 }, x.Shape));
        Assert.Equal(new ulong[] { 6144, 5120 }, specs.Single(x => x.Name == "dflash2/layers/0/attention/query_key_value").Shape);
    }

    [Theory]
    [InlineData("dflash2/feature_projection")]
    [InlineData("dflash2/context_norm")]
    [InlineData("dflash2/layers/4/mlp/conv_proj")]
    [InlineData("dflash2/selector/predecessor")]
    [InlineData("text/draft_head")]
    [InlineData("text/draft_head_token_ids")]
    public void MissingOriginalTensorIsRefused(string name)
    {
        var rows = Metadata(); rows.RemoveAll(x => x.Name == name);
        Assert.Throws<InvalidDataException>(() => Validate(rows));
    }

    [Theory]
    [InlineData("dflash2/layers/0/attention/query_key_value", 1)]
    [InlineData("dflash2/layers/0/attention/query_norm", 2)]
    [InlineData("dflash2/layers/0/mlp/conv_base", 3)]
    [InlineData("dflash2/feature_projection", 4)]
    [InlineData("text/draft_head", 5)]
    [InlineData("text/draft_head_token_ids", 6)]
    [InlineData("dflash2/final_norm", 7)]
    [InlineData("dflash2/selector/hidden", 8)]
    [InlineData("dflash2/selector/successor", 9)]
    public void AlteredShapeFormatGeometryRangeOrDivisorIsRefused(string name, int mutation)
    {
        var rows = Metadata(); int i = rows.FindIndex(x => x.Name == name); var x = rows[i];
        rows[i] = mutation switch
        {
            1 => x with { Shape = new ulong[] { 14336, 5120 } }, // target attention layout cannot masquerade as draft QKV
            2 => x with { Shape = new ulong[] { 256 } },
            3 => x with { Shape = new ulong[] { 5120, 2, 2 } },
            4 => x with { WeightDivisor = float.NaN, WeightDivisorBits = 0x7fc00000 },
            5 => x with { Format = NinferTensorFormat.NvFp4 },
            6 => x with { ByteLength = x.ByteLength - 4 },
            7 => x with { WeightDivisor = 1f, WeightDivisorBits = 0x3f800000 },
            8 => x with { AbsoluteOffset = FileBytes - 1 },
            _ => x with { Geometry = x.Geometry with { ScalePlaneOffset = x.Geometry.ScalePlaneOffset + 256 } }
        };
        Assert.Throws<InvalidDataException>(() => Validate(rows));
    }

    [Theory]
    [InlineData("dflash2/layers/0/attention/input_scale_divisor")]
    [InlineData("dflash2/layers/5/input_norm")]
    [InlineData("text/draft_other_head")]
    public void ExtraCalibrationOrAlternateDraftSchemaIsRefused(string name)
    {
        var rows = Metadata(); rows.Add(rows[0] with { Name = name });
        Assert.Throws<InvalidDataException>(() => Validate(rows));
    }

    [Fact]
    public void DuplicateOverlapIdentityAndDivisorBitMismatchAreRefused()
    {
        var rows = Metadata(); rows.Add(rows[0]); Assert.Throws<InvalidDataException>(() => Validate(rows));
        rows = Metadata(); rows[1] = rows[1] with { PayloadOffset = rows[0].PayloadOffset, AbsoluteOffset = rows[0].AbsoluteOffset };
        Assert.Throws<InvalidDataException>(() => Validate(rows));
        rows = Metadata(); int i = rows.FindIndex(x => x.Format == NinferTensorFormat.NvFp4);
        rows[i] = rows[i] with { WeightDivisorBits = 0x3f800000 }; Assert.Throws<InvalidDataException>(() => Validate(rows));
        Assert.Throws<NotSupportedException>(() => NinferQwen38DraftPlan.ValidateMetadata(new("other", "nvfp4full"), Metadata(), Payload, FileBytes));
    }

    // Sparse original-format objects: only synthetic BF16 vectors/divisor trailers/map are written.
    // This exercises real ownership/bounds/loading without storing model payloads in the repository.
    private sealed class Fixture : IDisposable
    {
        private readonly string _path = Path.GetTempFileName();
        internal NinferArtifactReader Reader { get; }
        internal NinferQwen38DraftPlan Plan { get; }
        internal Fixture(bool shortlist = true)
        {
            using var directory = JsonDocument.Parse(DirectoryBytes());
            var rows = directory.RootElement.GetProperty("objects").EnumerateArray().Where(x => shortlist || x.GetProperty("name").GetString()!.StartsWith("dflash2/", StringComparison.Ordinal)).Select(x => x.Clone()).ToArray();
            byte[] json = JsonSerializer.SerializeToUtf8Bytes(new { identity = new { model_id = "qwen3.8-27b", weights_id = "nvfp4full" }, objects = rows });
            using (var file = File.Create(_path))
            {
                file.Write("NINFER\0\x02"u8); Span<byte> scalar = stackalloc byte[8];
                BinaryPrimitives.WriteUInt64LittleEndian(scalar, (ulong)json.Length); file.Write(scalar); file.Write(json);
                long payload = (16L + json.Length + 4095) / 4096 * 4096;
                file.SetLength(payload + (long)rows.Max(x => x.GetProperty("offset").GetUInt64() + x.GetProperty("bytes").GetUInt64()));
                foreach (var row in rows.Where(x => x.GetProperty("format").GetString() == "NVFP4"))
                {
                    file.Position = payload + (long)(row.GetProperty("offset").GetUInt64() + row.GetProperty("bytes").GetUInt64() - 4);
                    BinaryPrimitives.WriteUInt32LittleEndian(scalar, 0x40200000); file.Write(scalar[..4]);
                }
            }
            Reader = new(_path); Plan = NinferQwen38DraftPlan.Create(Reader);
        }
        internal void Write(string name, long offset, ReadOnlySpan<byte> bytes)
        {
            using var file = new FileStream(_path, FileMode.Open, FileAccess.Write, FileShare.ReadWrite);
            file.Position = Reader.GetObject(name).AbsoluteOffset + offset; file.Write(bytes);
        }
        public void Dispose() { Reader.Dispose(); File.Delete(_path); }
    }

    [Fact]
    public void RealReaderBindingsRefuseForeignPlanDisposedReaderAndObjectBoundaryEscape()
    {
        using var a = new Fixture(false); using var b = new Fixture(false);
        Assert.False(a.Plan.InferenceSupported); Assert.False(a.Plan.HasShortlist);
        var norm = a.Plan.GetTensor("dflash2/context_norm"); Assert.True(norm.IsPlainNorm);
        Assert.False(norm.AllowActivationQuantization);
        Assert.Throws<ArgumentException>(() => b.Plan.ReadTensorRange(norm, 0, new byte[2]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.Plan.ReadTensorRange(norm, norm.Descriptor.ByteLength - 1, new byte[2]));
        a.Reader.Dispose(); Assert.Throws<ObjectDisposedException>(() => a.Plan.ReadTensorRange(norm, 0, Span<byte>.Empty));
        Assert.Throws<ObjectDisposedException>(() => NinferQwen38DraftPlan.Create(a.Reader));
    }

    [Fact]
    public void PlainNormsAndConvBasesLoadWithoutOffsetsTranspositionOrBf16Rerounding()
    {
        using var fixture = new Fixture(false);
        byte[] raw = { 0x80, 0x3f, 0x80, 0xbf, 0, 0x80, 1, 0 }; // +1,-1,-0,minimum BF16 subnormal
        foreach (string name in new[] { "dflash2/context_norm", "dflash2/layers/0/attention/conv_base" })
        {
            fixture.Write(name, 0, raw);
            var loaded = NinferQwen38DraftSource.ReadPlainParameter(fixture.Plan, fixture.Plan.GetTensor(name));
            Assert.Equal(new[] { 0x3f800000, unchecked((int)0xbf800000), int.MinValue, 0x00010000 }, loaded.Take(4).Select(BitConverter.SingleToInt32Bits));
        }
        // Mark one channel in all four [side][tap] planes, rather than only one adjacent run.
        for (int side = 0; side < 2; side++) for (int tap = 0; tap < 2; tap++)
            fixture.Write("dflash2/layers/0/attention/conv_base", ((side * 2 + tap) * 5120 + 19) * 2,
                new byte[] { 0, (byte)(0x40 + side * 2 + tap) });
        var conv = NinferQwen38DraftSource.ReadPlainParameter(fixture.Plan, fixture.Plan.GetTensor("dflash2/layers/0/attention/conv_base"));
        Assert.Equal(new[] { 0x40000000, 0x41000000, 0x42000000, 0x43000000 },
            Enumerable.Range(0, 4).Select(i => BitConverter.SingleToInt32Bits(conv[i * 5120 + 19])));
        fixture.Write("dflash2/context_norm", 2, new byte[] { 0x80, 0x7f });
        Assert.Throws<InvalidDataException>(() => NinferQwen38DraftSource.ReadPlainParameter(fixture.Plan, fixture.Plan.GetTensor("dflash2/context_norm")));
    }

    [Theory]
    [InlineData(-1)]
    [InlineData(248320)]
    public void OriginalShortlistMapRejectsOutOfVocabularyIds(int invalid)
    {
        using var fixture = new Fixture(); Span<byte> raw = stackalloc byte[4]; BinaryPrimitives.WriteInt32LittleEndian(raw, invalid);
        fixture.Write("text/draft_head_token_ids", 4 * 131071, raw);
        Assert.Throws<InvalidDataException>(() => NinferQwen38DraftSource.ReadShortlistTokenIds(fixture.Plan, fixture.Plan.GetTensor("text/draft_head_token_ids")));
    }

    [Fact]
    public void OriginalShortlistMapPreservesRowsAndDuplicateTokensWithoutInventedReordering()
    {
        using var fixture = new Fixture(); Span<byte> raw = stackalloc byte[12];
        BinaryPrimitives.WriteInt32LittleEndian(raw, 248319); BinaryPrimitives.WriteInt32LittleEndian(raw[4..], 7); BinaryPrimitives.WriteInt32LittleEndian(raw[8..], 7);
        fixture.Write("text/draft_head_token_ids", 0, raw);
        int[] map = NinferQwen38DraftSource.ReadShortlistTokenIds(fixture.Plan, fixture.Plan.GetTensor("text/draft_head_token_ids"));
        Assert.Equal(new[] { 248319, 7, 7 }, map.Take(3));
        var head = new CudaNinferDraftShortlistDesc(new NoReadSource(), 356515840, map); map[0] = 3;
        Assert.Equal(248319, head.TokenIds[0]); Assert.Equal(20971520, CudaNinferDraftShortlistDesc.ScaleBytes);
    }

    [Theory]
    [InlineData(0x7c00)]
    [InlineData(0xfc00)]
    [InlineData(0x7e01)]
    public void OriginalShortlistScaleValidationRefusesNonfiniteF16(int bits)
    {
        using var fixture = new Fixture(); var binding = fixture.Plan.GetTensor("text/draft_head");
        Span<byte> raw = stackalloc byte[2]; BinaryPrimitives.WriteUInt16LittleEndian(raw, (ushort)bits);
        fixture.Write("text/draft_head", (long)binding.Descriptor.Geometry.ScalePlaneOffset!.Value, raw);
        Assert.Throws<InvalidDataException>(() => NinferQwen38DraftSource.ValidateShortlistScales(fixture.Plan, binding));
    }

    [Fact]
    public void BoundedSourceBuildsAll34OriginalA16ProjectionsAndOwnsTheirLifetime()
    {
        using var fixture = new Fixture(false);
        long initialBytes = fixture.Reader.TotalBytesRead;
        using var source = new NinferQwen38DraftSource(fixture.Plan, 7, CudaNinferDraftHeadPolicy.FullTargetVocabulary);
        var descriptor = source.Descriptor;
        Assert.Equal(535040, fixture.Reader.TotalBytesRead - initialBytes); // Plain parameters only; quantized matrices stay streamed.
        Assert.Contains(":draft7:headFullTargetVocabulary", descriptor.Provenance);
        Assert.Equal(8, descriptor.BlockSize); Assert.Null(descriptor.Shortlist); Assert.Equal(5, descriptor.Layers.Count);
        var projections = new List<CudaNinferProjectionDesc> { descriptor.FeatureProjection, descriptor.SelectorHidden, descriptor.SelectorPredecessor, descriptor.SelectorSuccessor };
        foreach (var layer in descriptor.Layers) projections.AddRange(new[] { layer.QueryKeyValue, layer.AttentionOutput, layer.AttentionConvProjection, layer.GateUp, layer.Down, layer.MlpConvProjection });
        Assert.Equal(34, projections.Count);
        Assert.All(projections, x => { Assert.False(x.AllowA4); Assert.Equal(1f, x.InputScaleDivisor); Assert.Equal(2.5f, x.WeightScaleDivisor); });
        var raw = Assert.IsAssignableFrom<ICudaNinferOriginalProjectionSource>(descriptor.FeatureProjection.Source);
        Assert.Equal(25600, raw.Columns); Assert.Equal(73728004, raw.OriginalBytes);
        Assert.Equal(CudaNinferOriginalRowMapping.Identity, raw.RowMapping);
        Assert.Equal(CudaNinferOriginalRowMapping.GateUpRows8, ((ICudaNinferOriginalProjectionSource)descriptor.Layers[0].GateUp.Source).RowMapping);
        source.Dispose(); source.Dispose(); Assert.Throws<ObjectDisposedException>(() => raw.ReadOriginal(0, IntPtr.Zero, 0));
        fixture.Reader.ReadRange(0, Span<byte>.Empty); // Borrowed original remains open.
    }

    [Theory]
    [InlineData(0)]
    [InlineData(8)]
    public void UnsupportedDraftWindowIsRefusedBeforeReadingPayload(int tokens)
    {
        using var fixture = new Fixture(false); long reads = fixture.Reader.ReadOperations;
        Assert.Throws<ArgumentOutOfRangeException>(() => new NinferQwen38DraftSource(fixture.Plan, tokens, CudaNinferDraftHeadPolicy.FullTargetVocabulary));
        Assert.Equal(reads, fixture.Reader.ReadOperations);
    }

    [Fact]
    public void AbsentShortlistAndAccidentalA4CannotBeSilentlySelected()
    {
        using var fixture = new Fixture(false);
        Assert.Throws<NotSupportedException>(() => new NinferQwen38DraftSource(fixture.Plan, 3, CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist));
        const int n = 256, k = 5120; var source = new NoReadSource();
        var a4 = new CudaNinferProjectionDesc(source, 0, (long)n * k * 9 / 16, n, k, CudaNinferProjectionFormat.NvFp4Planes);
        Assert.Throws<ArgumentException>(() => CudaNinferDraftDesc.Projection(a4, n, k));
        var altered = new CudaNinferProjectionDesc(source, 0, (long)n * k * 9 / 16, n, k, CudaNinferProjectionFormat.NvFp4Planes, 2, 1, false);
        Assert.Throws<ArgumentException>(() => CudaNinferDraftDesc.Projection(altered, n, k));
        Assert.Throws<ArgumentException>(() => CudaNinferDraftDesc.Plain(new[] { 1.1f }, 1));
    }

    [Fact]
    public void SelectedOriginalShortlistBuildsItsMappedHeadWithoutRequantization()
    {
        using var fixture = new Fixture();
        using var source = new NinferQwen38DraftSource(fixture.Plan, 3, CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist);
        var head = Assert.IsType<CudaNinferDraftShortlistDesc>(source.Descriptor.Shortlist);
        Assert.Equal(CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist, source.Descriptor.HeadPolicy);
        Assert.Equal(356515840, head.Bytes); Assert.Equal(131072, head.TokenIds.Count);
        Assert.All(head.TokenIds, x => Assert.Equal(0, x));
        Assert.Equal(4, source.Descriptor.BlockSize);
    }

    [Fact]
    public void FullHeadChoiceDoesNotScanOrUseAnUnselectedShortlist()
    {
        using var fixture = new Fixture(); var head = fixture.Plan.GetTensor("text/draft_head");
        fixture.Write("text/draft_head", (long)head.Descriptor.Geometry.ScalePlaneOffset!.Value, new byte[] { 0, 0x7c });
        fixture.Write("text/draft_head_token_ids", 0, new byte[] { 0xff, 0xff, 0xff, 0xff });
        using var source = new NinferQwen38DraftSource(fixture.Plan, 1, CudaNinferDraftHeadPolicy.FullTargetVocabulary);
        Assert.Null(source.Descriptor.Shortlist);
        Assert.Throws<InvalidDataException>(() => new NinferQwen38DraftSource(fixture.Plan, 1, CudaNinferDraftHeadPolicy.OriginalQ4G64Shortlist));
        fixture.Reader.ReadRange(0, Span<byte>.Empty); // Failed selected-head load did not dispose its owner's reader.
    }
    private sealed class NoReadSource : ICudaWeightSource
    {
        public void Read(long offset, IntPtr dst, long bytes) => throw new InvalidOperationException("Metadata construction must not read payload.");
    }
}
