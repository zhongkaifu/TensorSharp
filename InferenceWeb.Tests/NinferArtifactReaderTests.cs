using System.Buffers.Binary;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

// Generated tiny/sparse files only. No model, ignored fixture, CUDA or native loader dependency.
public sealed class NinferArtifactReaderTests
{
    private static JsonObject Tensor(string format = "NVFP4", string layout = "blockscale-k16-m128x4-v1",
        ulong[]? shape = null, ulong bytes = 4612, ulong offset = 0, string name = "w") => new()
    {
        ["name"] = name, ["kind"] = "tensor", ["shape"] = new JsonArray((shape ?? new ulong[] { 128, 64 }).Select(x => JsonValue.Create(x)).ToArray()),
        ["format"] = format, ["layout"] = layout, ["offset"] = offset, ["bytes"] = bytes
    };
    private static JsonObject Resource(string name = "frontend/tokenizer.json", ulong bytes = 3, ulong offset = 0) => new()
    { ["name"] = name, ["kind"] = "resource", ["encoding"] = "raw-bytes-v1", ["offset"] = offset, ["bytes"] = bytes };
    private static JsonObject DirectoryOf(params JsonObject[] rows) => new()
    { ["identity"] = new JsonObject { ["model_id"] = "tiny", ["weights_id"] = "fixture" }, ["objects"] = new JsonArray(rows.Select(x => x.DeepClone()).ToArray()) };
    private sealed class Fixture : IDisposable
    {
        private readonly string _dir = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ts-ninfer-reader-" + Guid.NewGuid().ToString("N"));
        internal string Path { get; }
        internal int JsonBytes { get; }
        internal long Payload { get; }
        internal Fixture(JsonObject directory, float divisor = 1, byte[]? raw = null, long payloadBytes = 16384)
        {
            System.IO.Directory.CreateDirectory(_dir); Path = System.IO.Path.Combine(_dir, "tiny.ninfer");
            byte[] json = raw ?? Encoding.UTF8.GetBytes(directory.ToJsonString()); JsonBytes = json.Length;
            Payload = (16L + json.Length + 4095) / 4096 * 4096;
            using var stream = File.Create(Path); stream.Write("NINFER\0\x02"u8);
            Span<byte> length = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)json.Length); stream.Write(length); stream.Write(json);
            stream.SetLength(Payload + payloadBytes);
            foreach (var row in directory["objects"]!.AsArray())
            {
                if (row?["format"]?.GetValue<string>() != "NVFP4") continue;
                if (!row["offset"]!.AsValue().TryGetValue<ulong>(out var offset) || !row["bytes"]!.AsValue().TryGetValue<ulong>(out var bytes)) continue;
                if (bytes < 4 || offset > (ulong)payloadBytes || bytes > (ulong)payloadBytes - offset) continue;
                stream.Position = checked(Payload + (long)offset + (long)bytes - 4);
                BinaryPrimitives.WriteInt32LittleEndian(length[..4], BitConverter.SingleToInt32Bits(divisor)); stream.Write(length[..4]);
            }
        }
        public void Dispose() => System.IO.Directory.Delete(_dir, true);
    }

    [Theory]
    [InlineData("BF16", 2, NinferTensorFormat.BFloat16)]
    [InlineData("FP32", 4, NinferTensorFormat.Float32)]
    [InlineData("I32", 4, NinferTensorFormat.Int32)]
    [InlineData("I64", 8, NinferTensorFormat.Int64)]
    public void DirectTypesAndRankZeroHaveExactOriginalBytes(string format, int width, NinferTensorFormat expected)
    {
        using var f = new Fixture(DirectoryOf(Tensor(format, "contiguous-le-v1", Array.Empty<ulong>(), (ulong)width)));
        using var reader = new NinferArtifactReader(f.Path);
        var tensor = Assert.Single(reader.Tensors);
        Assert.Equal(expected, tensor.Format); Assert.Empty(tensor.Shape);
        Assert.Equal((ulong)width, tensor.ByteLength); Assert.Equal((ulong)width, tensor.Geometry.EncodedBytes);
        Assert.Equal(16 + f.JsonBytes, reader.InitializationBytesRead);
        Assert.Null(tensor.WeightDivisor); Assert.Null(tensor.Geometry.ScalePlaneOffset);
    }

    [Theory]
    [InlineData("Q4G64_F16S", 536, 64, 512, 24)]
    [InlineData("Q5G64_F16S", 792, 64, 768, 24)]
    [InlineData("Q6G64_F16S", 792, 64, 768, 24)]
    [InlineData("W8G32_F16S", 816, 32, 768, 48)]
    public void RowSplitPaddedKAndIndependentPlaneSizes(string format, int bytes, int group, int scale, int scaleBytes)
    {
        using var f = new Fixture(DirectoryOf(Tensor(format, "row-split-k128-v1", new ulong[] { 3, 129 }, (ulong)bytes)));
        using var reader = new NinferArtifactReader(f.Path);
        var tensor = Assert.Single(reader.Tensors); var g = tensor.Geometry;
        Assert.Equal(new ulong[] { 3, 129 }, tensor.Shape);
        Assert.Equal(256UL, g.PaddedK); Assert.Equal(group, g.GroupSize);
        Assert.Equal((ulong)bytes, g.EncodedBytes); Assert.Equal((ulong)scale, g.ScalePlaneOffset); Assert.Equal((ulong)scaleBytes, g.ScalePlaneBytes);
        Assert.Equal(NinferTensorLayout.RowSplitK128V1, tensor.Layout);
    }

    [Fact]
    public void Nvfp4ReadsOnlyHeaderDirectoryAndTrailingFinitePositiveDivisor()
    {
        using var f = new Fixture(DirectoryOf(Tensor()), divisor: 2.5f);
        using var reader = new NinferArtifactReader(f.Path);
        var tensor = Assert.Single(reader.Tensors); var g = tensor.Geometry;
        Assert.Equal(4096UL, g.LowPlaneBytes); Assert.Equal(4096UL, g.ScalePlaneOffset);
        Assert.Equal(512UL, g.ScalePlaneBytes); Assert.Equal(4608UL, g.DivisorOffset);
        Assert.Equal(4612UL, tensor.ByteLength); Assert.Equal(2.5f, tensor.WeightDivisor);
        Assert.Equal(unchecked((uint)BitConverter.SingleToInt32Bits(2.5f)), tensor.WeightDivisorBits);
        Assert.Equal(16 + f.JsonBytes + 4, reader.InitializationBytesRead);
        Assert.Equal(4, reader.InitializationReadOperations); Assert.False(reader.FullArtifactSha256Recomputed);
        Assert.Equal("tiny", reader.Identity.ModelId); Assert.Equal("fixture", reader.Identity.WeightsId);
        Assert.Equal(64, reader.DirectorySha256.Length); Assert.Equal(64, reader.PrefixSha256.Length);
        Assert.Throws<NotSupportedException>(() => ((IList<ulong>)tensor.Shape)[0] = 0);
        Assert.Throws<NotSupportedException>(() => ((IList<NinferTensorDescriptor>)reader.Tensors).Clear());
    }

    [Fact]
    public void ActualW8VocabularyGeometryNeedsNoModelOrSparseArtifactFixture()
    {
        var method = typeof(NinferArtifactReader).GetMethod("Geometry", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)!;
        var geometry = (NinferTensorGeometry)method.Invoke(null, new object[]
            { NinferTensorFormat.W8G32Float16Scale, NinferTensorLayout.RowSplitK128V1, new ulong[] { 248320, 5120 } })!;
        Assert.Equal(1350860800UL, geometry.EncodedBytes); // Independent actual shape size, not an inspected file.
        Assert.Equal(1271398400UL, geometry.ScalePlaneOffset);
        Assert.Equal(79462400UL, geometry.ScalePlaneBytes);
    }

    [Fact]
    public void PositiveFiniteDivisorsPreserveExactSubnormalAndMaximumBits()
    {
        foreach (float value in new[] { BitConverter.Int32BitsToSingle(1), float.MaxValue })
        {
            using var f = new Fixture(DirectoryOf(Tensor()), divisor: value);
            using var reader = new NinferArtifactReader(f.Path);
            Assert.Equal(unchecked((uint)BitConverter.SingleToInt32Bits(value)), reader.Tensors[0].WeightDivisorBits);
        }
    }

    [Fact]
    public void Fp8RowScaleGeometryAndMultidimensionalDirectTensorAreDistinct()
    {
        using var f = new Fixture(DirectoryOf(Tensor("FP8_E4M3FN_ROW_BF16S", "row-scale-v1", new ulong[] { 3, 67 }, 262),
            Tensor("FP32", "contiguous-le-v1", new ulong[] { 2, 3, 5 }, 120, 512, "norm")));
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Equal(256UL, reader.Tensors[0].Geometry.ScalePlaneOffset);
        Assert.Equal(6UL, reader.Tensors[0].Geometry.ScalePlaneBytes);
        Assert.Equal(120UL, reader.Tensors[1].ByteLength);
    }

    [Fact]
    public void BoundedResourceReadDoesNotInterpretOrAutomaticallyBindTokenizer()
    {
        using var f = new Fixture(DirectoryOf(Resource(), Tensor("I32", "contiguous-le-v1", Array.Empty<ulong>(), 4, 256)));
        using (var writer = new FileStream(f.Path, FileMode.Open, FileAccess.Write))
        { writer.Position = f.Payload; writer.Write("{}\n"u8); }
        using var reader = new NinferArtifactReader(f.Path);
        var resource = Assert.Single(reader.Resources);
        Assert.Equal("raw-bytes-v1", resource.Encoding);
        Assert.Equal("{}\n"u8.ToArray(), reader.ReadResource(resource, 3));
        Assert.Same(resource, reader.GetObject("frontend/tokenizer.json"));
        Assert.Throws<InvalidOperationException>(() => reader.ReadResource(resource, 2));
        Assert.Throws<ArgumentOutOfRangeException>(() => reader.ReadResource(resource, 0));
    }

    [Fact]
    public async Task ParallelPositionalReadsDoNotShareASeekCursorOrMutatePayload()
    {
        byte[] expected = Enumerable.Range(0, 4096).Select(i => (byte)(i % 251)).ToArray();
        using var f = new Fixture(DirectoryOf(Resource(bytes: 4096)));
        using (var writer = new FileStream(f.Path, FileMode.Open, FileAccess.Write)) { writer.Position = f.Payload; writer.Write(expected); }
        using var reader = new NinferArtifactReader(f.Path);
        long before = reader.TotalBytesRead;
        await Task.WhenAll(Enumerable.Range(0, 16).Select(lane => Task.Run(() =>
        {
            byte[] actual = new byte[31];
            for (int i = 0; i < 100; i++)
            {
                int offset = (lane * 199 + i * 17) % (4096 - actual.Length);
                reader.ReadObjectRange(reader.Resources[0], (ulong)offset, actual);
                Assert.Equal(expected.AsSpan(offset, actual.Length).ToArray(), actual);
            }
        })));
        Assert.Equal(16 * 100 * 31, reader.TotalBytesRead - before);
        Assert.Equal(expected, reader.ReadResource(reader.Resources[0], 4096));
    }

    [Fact]
    public void ReadRangesAreBoundedAndForeignDescriptorsOrDisposedReadersAreRejected()
    {
        using var f = new Fixture(DirectoryOf(Resource()));
        using var a = new NinferArtifactReader(f.Path); var b = new NinferArtifactReader(f.Path);
        Assert.Throws<ArgumentException>(() => a.ReadResource(b.Resources[0]));
        Assert.Throws<ArgumentException>(() => a.ReadObjectRange(b.Resources[0], 0, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.ReadRange(-1, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.ReadRange(long.MaxValue, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.ReadRange(a.FileBytes, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.ReadObjectRange(a.Resources[0], ulong.MaxValue, new byte[1]));
        Assert.Throws<ArgumentOutOfRangeException>(() => a.ReadObjectRange(a.Resources[0], 2, new byte[2]));
        a.ReadRange(a.FileBytes, Span<byte>.Empty); a.ReadObjectRange(a.Resources[0], 3, Span<byte>.Empty);
        b.Dispose(); b.Dispose();
        Assert.Throws<ObjectDisposedException>(() => b.ReadRange(0, new byte[1]));
    }

    [Fact]
    public void V3ViAndKnownOpaqueFormatsAreExplicitlyUnsupported()
    {
        foreach (byte[] magic in new[] { "NINFER\0\x03"u8.ToArray(), "VELOXVI\x01"u8.ToArray(), "NINFER\0\x01"u8.ToArray() })
        {
            using var f = new Fixture(DirectoryOf(Resource())); File.WriteAllBytes(f.Path, magic);
            Assert.Contains("UNSUPPORTED_SCHEMA", Assert.Throws<NotSupportedException>(() => new NinferArtifactReader(f.Path)).Message);
        }
        foreach (var row in new[] { Tensor("TRELLIS_MUL1_K3", "trellis-tile16-v1", new ulong[] { 128, 128 }, 4096), Tensor("Q4G32_F16S", "row-interleaved-v1", new ulong[] { 1, 32 }, 18) })
        {
            using var f = new Fixture(DirectoryOf(row));
            Assert.Contains("UNSUPPORTED_TENSOR_GEOMETRY", Assert.Throws<NotSupportedException>(() => new NinferArtifactReader(f.Path)).Message);
        }
    }

    [Fact]
    public void ClosedSchemaDuplicatesAndMalformedUtf8AreRejected()
    {
        string valid = DirectoryOf(Tensor()).ToJsonString();
        foreach (string raw in new[]
        {
            valid[..^1] + ",\"extra\":1}", valid.Replace("\"model_id\":\"tiny\"", "\"model_id\":\"tiny\",\"model_id\":\"other\""),
            valid.Replace("\"weights_id\":\"fixture\"", "\"weights_id\":null"), valid.Replace("\"kind\":\"tensor\"", "\"kind\":\"tensor\",\"kind\":\"tensor\""),
            valid.Replace("\"kind\":\"tensor\"", "\"kind\":\"unknown\""), valid.Replace("\"name\":\"w\"", "\"name\":\"\""),
            valid.Replace("\"format\":\"NVFP4\"", "\"format\":\"NVFP4\",\"extra\":0"), "{\"identity\":{}}", "{\"objects\":NaN}"
        })
        {
            using var f = new Fixture(DirectoryOf(Tensor()), raw: Encoding.UTF8.GetBytes(raw));
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        }
        using var bad = new Fixture(DirectoryOf(Tensor()), raw: new byte[] { 0xff });
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(bad.Path));
        using var duplicate = new Fixture(DirectoryOf(Tensor(), Tensor(offset: 4864)));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(duplicate.Path));
    }

    [Fact]
    public void OffsetsAlignmentOrderSizeAndUnsignedNumberTypesFailClosed()
    {
        foreach (Action<JsonObject> mutate in new Action<JsonObject>[]
        {
            row => row["offset"] = true, row => row["offset"] = -1, row => row["bytes"] = JsonNode.Parse("4612.0"),
            row => row["bytes"] = 0, row => row["bytes"] = 4611, row => row["offset"] = 1,
            row => row["offset"] = 16384, row => row["offset"] = ulong.MaxValue - 255,
            row => row["shape"] = new JsonArray(128, 63), row => row["shape"] = new JsonArray(0, 64),
            row => row["shape"] = new JsonArray(true, 64), row => row["shape"] = new JsonArray(128),
            row => row["layout"] = "contiguous-le-v1", row => row["format"] = "FP16",
        })
        {
            var row = Tensor(); mutate(row); using var f = new Fixture(DirectoryOf(row));
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        }
        using var overlap = new Fixture(DirectoryOf(Tensor(name: "a"), Tensor(offset: 256, name: "b")));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(overlap.Path));
        using var order = new Fixture(DirectoryOf(Tensor(offset: 4864, name: "a"), Tensor(name: "b")));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(order.Path));
        using var badResource = new Fixture(DirectoryOf(new JsonObject { ["name"] = "r", ["kind"] = "resource", ["encoding"] = "json", ["offset"] = 0, ["bytes"] = 1 }));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(badResource.Path));
    }

    [Fact]
    public void OverflowAndInvalidRanksAreRefusedBeforePayloadReads()
    {
        foreach (var row in new[]
        {
            Tensor("FP32", "contiguous-le-v1", new[] { ulong.MaxValue, 2UL }, 4),
            Tensor("W8G32_F16S", "row-split-k128-v1", new[] { 1UL, ulong.MaxValue }, 4),
            Tensor("FP32", "contiguous-le-v1", Enumerable.Repeat(1UL, 17).ToArray(), 4)
        })
        {
            using var f = new Fixture(DirectoryOf(row)); Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        }
    }

    [Fact]
    public void AllNonpositiveOrNonfiniteNvfp4DivisorsAreRefused()
    {
        foreach (float value in new[] { 0f, -0f, -1f, float.NaN, float.PositiveInfinity, float.NegativeInfinity })
        {
            using var f = new Fixture(DirectoryOf(Tensor()), divisor: value);
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        }
    }

    [Fact]
    public void HeaderTruncationLengthOverflowAndExplicitLimitsAreEnforced()
    {
        foreach (byte[] raw in new[] { Array.Empty<byte>(), "GGUFxxxx"u8.ToArray(), "NINFER\0\x02"u8.ToArray(), "NINFER\0\x02"u8.ToArray().Concat(new byte[8]).ToArray(), "NINFER\0\x02"u8.ToArray().Concat(Enumerable.Repeat((byte)255, 8)).ToArray() })
        {
            using var f = new Fixture(DirectoryOf(Resource())); File.WriteAllBytes(f.Path, raw);
            Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(f.Path));
        }
        using var valid = new Fixture(DirectoryOf(Resource(), Tensor("I32", "contiguous-le-v1", Array.Empty<ulong>(), 4, 256)));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(valid.Path, maxMetadataBytes: 1));
        Assert.Throws<InvalidDataException>(() => new NinferArtifactReader(valid.Path, maxObjects: 1));
        Assert.Throws<ArgumentOutOfRangeException>(() => new NinferArtifactReader(valid.Path, maxObjects: 0));
    }

    [Fact]
    public void SparseUnclaimedTailIsReportedButNeverReadOrHashed()
    {
        using var f = new Fixture(DirectoryOf(Tensor()), payloadBytes: (1L << 30) + 4612);
        using var reader = new NinferArtifactReader(f.Path);
        Assert.Equal(1UL << 30, reader.TrailingUnclaimedPayloadBytes);
        Assert.Equal(16 + f.JsonBytes + 4, reader.InitializationBytesRead);
        Assert.False(reader.FullArtifactSha256Recomputed);
    }
}
