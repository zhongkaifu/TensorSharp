using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class NinferQwen38TargetPlanTests
{
    private const long Payload = 212992, FileBytes = 19406942468;
    private static readonly NinferArtifactIdentity Identity = new("qwen3.8-27b", "nvfp4full");
    private const string FixtureSha = "97dbfacf273f8db61e905f9dbb600db9199e3565d6385c5f3888234feb234238";

    // Public directory facts only. Synthetic 2.5 weight-divisor metadata below is
    // deliberately NOT actual payload evidence and cannot create reader-owned descriptors.
    private static List<NinferQwen38TensorMetadata> Metadata()
    {
        using var source = typeof(NinferQwen38TargetPlanTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.TestData.Ninfer.qwen38-nvfp4full-target-directory.json")!;
        using var bytes = new MemoryStream(); source.CopyTo(bytes);
        Assert.Equal(FixtureSha, Convert.ToHexString(SHA256.HashData(bytes.ToArray())).ToLowerInvariant());
        using var document = JsonDocument.Parse(bytes.ToArray()); var root = document.RootElement;
        Assert.Equal("222b7540e961a6455fff9f03cce451860e4d248785a4ef0a805f538c8eb82658", root.GetProperty("provenance").GetProperty("directory_sha256").GetString());
        Assert.Equal(1325, root.GetProperty("provenance").GetProperty("original_objects").GetInt32());
        Assert.Equal(906, root.GetProperty("objects").GetArrayLength());
        var result = new List<NinferQwen38TensorMetadata>();
        foreach (var row in root.GetProperty("objects").EnumerateArray())
        {
            var format = row.GetProperty("format").GetString() switch
            {
                "NVFP4" => NinferTensorFormat.NvFp4, "BF16" => NinferTensorFormat.BFloat16,
                "FP32" => NinferTensorFormat.Float32, "W8G32_F16S" => NinferTensorFormat.W8G32Float16Scale,
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
            ulong count = shape.Aggregate(1UL, (a, b) => checked(a * b));
            NinferTensorGeometry geometry = format switch
            {
                NinferTensorFormat.NvFp4 => new(count / 2 + count / 16 + 4, 256, LowPlaneBytes: count / 2,
                    ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 16, DivisorOffset: count / 2 + count / 16),
                NinferTensorFormat.W8G32Float16Scale => new(count + count / 16, 256, PaddedK: 5120, GroupSize: 32,
                    LowPlaneBytes: count, HighPlaneOffset: count, HighPlaneBytes: 0, ScalePlaneOffset: count, ScalePlaneBytes: count / 16),
                _ => new(count * (format == NinferTensorFormat.BFloat16 ? 2UL : 4UL), 256)
            };
            ulong offset = row.GetProperty("offset").GetUInt64();
            result.Add(new(row.GetProperty("name").GetString()!, format, layout, Array.AsReadOnly(shape), offset,
                checked(Payload + (long)offset), row.GetProperty("bytes").GetUInt64(), geometry,
                format == NinferTensorFormat.NvFp4 ? 2.5f : null, format == NinferTensorFormat.NvFp4 ? 0x40200000u : null));
        }
        return result;
    }
    private static IReadOnlyList<NinferQwen38TensorSpec> Validate(List<NinferQwen38TensorMetadata> rows) =>
        NinferQwen38TargetPlan.ValidateMetadata(Identity, rows, Payload, FileBytes);
    private static void Mutate(List<NinferQwen38TensorMetadata> rows, string name,
        Func<NinferQwen38TensorMetadata, NinferQwen38TensorMetadata> mutation)
    {
        int i = rows.FindIndex(x => x.Name == name); Assert.True(i >= 0); rows[i] = mutation(rows[i]);
    }

    [Fact]
    public void Public906DirectoryValidatesComplete64LayerProfileWithoutReadingWeights()
    {
        var specs = Validate(Metadata()); Assert.Equal(906, specs.Count);
        Assert.Equal(247, specs.Count(x => x.Format == NinferTensorFormat.NvFp4));
        Assert.Equal(314, specs.Count(x => x.Format == NinferTensorFormat.BFloat16));
        Assert.Equal(343, specs.Count(x => x.Format == NinferTensorFormat.Float32));
        Assert.Equal(2, specs.Count(x => x.Format == NinferTensorFormat.W8G32Float16Scale));
        Assert.Equal(48, specs.Count(x => x.Role == NinferQwen38TensorRole.GdnInput));
        Assert.Equal(16, specs.Count(x => x.Role == NinferQwen38TensorRole.AttentionInput));
        Assert.Equal(64, specs.Count(x => x.Role == NinferQwen38TensorRole.GateUp));
        Assert.Throws<NotSupportedException>(() => ((IList<NinferQwen38TensorSpec>)specs).Clear());
        Assert.Throws<NotSupportedException>(() => ((IList<ulong>)specs[0].Shape)[0] = 1);
    }

    [Fact]
    public void GroupedHeadMappingCannotSilentlyUseLegacyModuloOrder()
    {
        for (int h = 0; h < 48; h++) Assert.Equal(h / 3, NinferQwen38TargetPlan.GdnQueryKeyHead(h));
        Assert.Equal(0, NinferQwen38TargetPlan.GdnQueryKeyHead(1)); // legacy h%16 would be1
        Assert.Equal(5, NinferQwen38TargetPlan.GdnQueryKeyHead(16)); // legacy would be0
        Assert.Throws<ArgumentOutOfRangeException>(() => NinferQwen38TargetPlan.GdnQueryKeyHead(-1));
        Assert.Throws<ArgumentOutOfRangeException>(() => NinferQwen38TargetPlan.GdnQueryKeyHead(48));
    }

    [Fact]
    public void GdnRawSemanticsKeepSeparateABRawLogTapOrderAndPlainGatedNorm()
    {
        var specs = Validate(Metadata()).ToDictionary(x => x.Name);
        Assert.Equal(NinferQwen38RowOrder.GdnQueryKeyValueZ, specs["text/layers/0/gdn/query_key_value_z"].Rows);
        Assert.Equal(new ulong[] { 16384, 5120 }, specs["text/layers/0/gdn/query_key_value_z"].Shape);
        Assert.Equal(NinferQwen38RowOrder.GdnAlphaThenBeta, specs["text/layers/0/gdn/a_b_projection"].Rows);
        Assert.Equal(NinferTensorFormat.BFloat16, specs["text/layers/0/gdn/a_b_projection"].Format);
        Assert.Equal(NinferQwen38ParameterSemantics.RawALogRequiresNegativeExpf, specs["text/layers/0/gdn/a_log"].Parameter);
        Assert.Equal(NinferQwen38RowOrder.ConvTapThenChannel, specs["text/layers/0/gdn/convolution"].Rows);
        Assert.Equal(NinferQwen38NormConvention.StoredPlain, specs["text/layers/0/gdn/norm"].Norm);
        Assert.Equal(NinferQwen38RowOrder.Unchanged, specs["text/layers/0/gdn/output"].Rows);
    }

    [Fact]
    public void AttentionAndMlpRetainArtifactRowOrderAndUnitOffsetNorms()
    {
        var specs = Validate(Metadata()).ToDictionary(x => x.Name);
        Assert.Equal(NinferQwen38RowOrder.AttentionQueryKeyGateValue, specs["text/layers/3/attention/query_key_gate_value"].Rows);
        Assert.Equal(NinferQwen38RowOrder.MlpGateThenUp, specs["text/layers/3/mlp/gate_up"].Rows);
        foreach (string name in new[] { "text/final_norm", "text/layers/0/input_norm", "text/layers/0/post_attention_norm",
            "text/layers/3/attention/query_norm", "text/layers/3/attention/key_norm" })
            Assert.Equal(NinferQwen38NormConvention.StoredUnitOffset, specs[name].Norm);
    }

    [Fact]
    public void EveryNvfp4HasIndependentUnreadActivationDivisorDescriptor()
    {
        var specs = Validate(Metadata()).ToDictionary(x => x.Name);
        foreach (var p in specs.Values.Where(x => x.Format == NinferTensorFormat.NvFp4))
        {
            Assert.NotNull(p.InputScaleDivisorName); var d = specs[p.InputScaleDivisorName!];
            Assert.Equal(NinferTensorFormat.Float32, d.Format); Assert.Empty(d.Shape);
            Assert.Equal(NinferQwen38ParameterSemantics.CalibratedActivationDivisor, d.Parameter);
            Assert.Equal(NinferQwen38TensorRole.InputScaleDivisor, d.Role);
        }
        Assert.Equal(247, specs.Values.Count(x => x.Role == NinferQwen38TensorRole.InputScaleDivisor));
        Assert.Null(specs["text/layers/3/attention/query_key_gate_value"].InputScaleDivisorName);
    }

    [Fact]
    public void ExactlyNineLargeBf16ExceptionsAndBothW8EndpointsAreRequired()
    {
        var rows = Metadata(); var actual = rows.Where(x => x.Format == NinferTensorFormat.BFloat16 && x.Shape.Count == 2
            && x.Shape[0] != 4 && x.Shape[0] != 96).Select(x => x.Name).OrderBy(x => x).ToArray();
        string[] expected = { "text/layers/3/attention/query_key_gate_value", "text/layers/7/attention/query_key_gate_value",
            "text/layers/11/attention/query_key_gate_value", "text/layers/15/attention/query_key_gate_value",
            "text/layers/19/attention/query_key_gate_value", "text/layers/23/attention/query_key_gate_value",
            "text/layers/3/attention/output", "text/layers/7/attention/output", "text/layers/4/gdn/output" };
        Assert.Equal(expected.OrderBy(x => x), actual);
        foreach (string n in new[] { "text/token_embedding", "text/output_head" })
        {
            var w = Assert.Single(rows, x => x.Name == n);
            Assert.Equal(new ulong[] { 248320, 5120 }, w.Shape); Assert.Equal(1350860800UL, w.ByteLength);
        }
        foreach (string name in expected.Append("text/token_embedding").Append("text/output_head").Append("text/layers/27/attention/query_key_gate_value"))
        {
            var bad = Metadata(); Mutate(bad, name, x => x with { Format = x.Format == NinferTensorFormat.NvFp4 ? NinferTensorFormat.BFloat16 : NinferTensorFormat.NvFp4 });
            Assert.Throws<InvalidDataException>(() => Validate(bad));
        }
    }

    [Theory]
    [InlineData("text/layers/0/gdn/query_key_value_z", 16480, 5120)]
    [InlineData("text/layers/0/gdn/a_b_projection", 48, 5120)]
    [InlineData("text/layers/0/gdn/convolution", 10240, 4)]
    [InlineData("text/layers/3/attention/query_key_gate_value", 16384, 5120)]
    [InlineData("text/layers/3/mlp/gate_up", 17408, 5120)]
    public void WrongLogicalRowOrConvolutionOrderIsRejected(string name, int n, int k)
    {
        var rows = Metadata(); Mutate(rows, name, x => x with { Shape = new ulong[] { (ulong)n, (ulong)k } });
        Assert.Throws<InvalidDataException>(() => Validate(rows));
    }

    [Fact]
    public void WrongNormTypeAndHeadNormWidthAreRejected()
    {
        foreach (string name in new[] { "text/final_norm", "text/layers/0/gdn/norm", "text/layers/3/attention/query_norm" })
        {
            var rows = Metadata(); Mutate(rows, name, x => x with { Format = NinferTensorFormat.Float32 });
            Assert.Throws<InvalidDataException>(() => Validate(rows));
        }
        var bad = Metadata(); Mutate(bad, "text/layers/3/attention/key_norm", x => x with { Shape = new ulong[] { 128 } });
        Assert.Throws<InvalidDataException>(() => Validate(bad));
    }

    [Fact]
    public void DivisorPresenceBitsAndPositiveFiniteStateCannotBeConfused()
    {
        foreach (float? value in new float?[] { null, 0, -1, float.NaN, float.PositiveInfinity })
        {
            var rows = Metadata(); Mutate(rows, "text/layers/0/mlp/gate_up", x => x with { WeightDivisor = value });
            Assert.Throws<InvalidDataException>(() => Validate(rows));
        }
        var mismatch = Metadata(); Mutate(mismatch, "text/layers/0/mlp/gate_up", x => x with { WeightDivisorBits = 0x3f800000 });
        Assert.Throws<InvalidDataException>(() => Validate(mismatch));
        var wrong = Metadata(); Mutate(wrong, "text/layers/0/mlp/gate_up_projection/input_scale_divisor", x => x with { WeightDivisor = 2.5f, WeightDivisorBits = 0x40200000 });
        Assert.Throws<InvalidDataException>(() => Validate(wrong));
        var inputRank = Metadata(); Mutate(inputRank, "text/layers/0/mlp/gate_up_projection/input_scale_divisor", x => x with { Shape = new ulong[] { 1 } });
        Assert.Throws<InvalidDataException>(() => Validate(inputRank));
    }

    [Fact]
    public void MissingDuplicateUnexpectedTargetAndInvalidRangeOrGeometryFailClosed()
    {
        var missing = Metadata(); missing.RemoveAt(0); Assert.Throws<InvalidDataException>(() => Validate(missing));
        var duplicate = Metadata(); duplicate.Add(duplicate[0]); Assert.Throws<InvalidDataException>(() => Validate(duplicate));
        var extra = Metadata(); extra.Add(extra[0] with { Name = "text/layers/64/input_norm" }); Assert.Throws<InvalidDataException>(() => Validate(extra));
        foreach (string mutation in new[] { "absolute", "overlap", "bytes", "geometry", "layout" })
        {
            var bad = Metadata(); Mutate(bad, "text/layers/0/mlp/gate_up", x => mutation switch
            {
                "absolute" => x with { AbsoluteOffset = x.AbsoluteOffset + 256 },
                "overlap" => x with { PayloadOffset = 0, AbsoluteOffset = Payload },
                "bytes" => x with { ByteLength = x.ByteLength - 1 },
                "geometry" => x with { Geometry = x.Geometry with { DivisorOffset = 1 } },
                _ => x with { Layout = NinferTensorLayout.RowSplitK128V1 }
            });
            Assert.Throws<InvalidDataException>(() => Validate(bad));
        }
    }

    [Theory]
    [InlineData("unknown", "nvfp4full")]
    [InlineData("qwen3.8-27b", "hybrid")]
    public void UnknownFamilyOrWeightProfileIsNotAnInferenceFallback(string model, string weights)
        => Assert.Throws<NotSupportedException>(() => NinferQwen38TargetPlan.ValidateMetadata(new(model, weights), Metadata(), Payload, FileBytes));

    [Fact]
    public void ActualCreateUsesRealReaderAndDoesNotOwnItsLifetimeOrReadRejectedPayload()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-ninfer-plan-" + Guid.NewGuid().ToString("N")); Directory.CreateDirectory(dir);
        try
        {
            string path = Path.Combine(dir, "tiny.ninfer");
            byte[] json = Encoding.UTF8.GetBytes("{\"identity\":{\"model_id\":\"qwen3.8-27b\",\"weights_id\":\"nvfp4full\"},\"objects\":[{\"name\":\"text/final_norm\",\"kind\":\"tensor\",\"shape\":[1],\"format\":\"BF16\",\"layout\":\"contiguous-le-v1\",\"offset\":0,\"bytes\":2}]}");
            using (var f = File.Create(path))
            {
                f.Write("NINFER\0\x02"u8); Span<byte> length = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)json.Length); f.Write(length); f.Write(json);
                f.SetLength(4098); f.Position = 4096; f.Write(new byte[] { 0x80, 0x3f });
            }
            using var reader = new NinferArtifactReader(path); long before = reader.TotalBytesRead;
            Assert.Throws<InvalidDataException>(() => NinferQwen38TargetPlan.Create(reader));
            Assert.Equal(before, reader.TotalBytesRead); byte[] value = new byte[2]; reader.ReadObjectRange(reader.Tensors[0], 0, value);
            Assert.Equal(new byte[] { 0x80, 0x3f }, value); reader.Dispose();
            Assert.Throws<ObjectDisposedException>(() => NinferQwen38TargetPlan.Create(reader));
        }
        finally { Directory.Delete(dir, true); }
    }
}
