using System.Buffers.Binary;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class NinferNvfp4PlaneReaderTests
{
    private static int Code(int row, int column) => (row * 17 + column * 5 + (column / 16) * 3) & 15;
    private static byte Scale(int row, int group) => (byte)((row * 13 + group * 7) % 127);
    private sealed class Fixture : IDisposable
    {
        private readonly string _directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "ts-ninfer-planes-" + Guid.NewGuid().ToString("N"));
        internal string Path { get; }
        internal int Rows { get; }
        internal int Columns { get; }
        internal byte[] Encoded { get; }
        internal NinferArtifactReader Reader { get; }
        internal NinferTensorDescriptor Weights => (NinferTensorDescriptor)Reader.GetObject("weights");
        internal NinferTensorDescriptor Input => (NinferTensorDescriptor)Reader.GetObject("input_divisor");
        internal Fixture(int rows = 128, int columns = 64, uint divisorBits = 0x40200000,
            int invalidScaleIndex = -1, byte invalidScale = 0, uint inputBits = 0x7fc00001)
        {
            Rows = rows; Columns = columns; int codeBytes = checked(rows * columns / 2), scaleBytes = checked(rows * columns / 16);
            Encoded = new byte[checked(codeBytes + scaleBytes + 4)];
            // Independently expressed source writer: adjacent logical values per byte.
            for (int r = 0; r < rows; r++)
                for (int j = 0; j < columns / 2; j++) Encoded[r * (columns / 2) + j] = (byte)(Code(r, 2 * j) | (Code(r, 2 * j + 1) << 4));
            // Independent five-axis producer order; no use of the converter's address formula.
            int at = codeBytes;
            for (int mb = 0; mb < rows / 128; mb++)
                for (int kb = 0; kb < columns / 64; kb++)
                    for (int rm = 0; rm < 32; rm++)
                        for (int quarter = 0; quarter < 4; quarter++)
                            for (int lane = 0; lane < 4; lane++) Encoded[at++] = Scale(mb * 128 + quarter * 32 + rm, kb * 4 + lane);
            Assert.Equal(codeBytes + scaleBytes, at);
            if (invalidScaleIndex >= 0) Encoded[codeBytes + invalidScaleIndex] = invalidScale;
            BinaryPrimitives.WriteUInt32LittleEndian(Encoded.AsSpan(at), divisorBits);
            int inputOffset = (Encoded.Length + 255) / 256 * 256;
            byte[] directory = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(new
            {
                identity = new { model_id = "tiny-real-reader", weights_id = "fixture" },
                objects = new object[]
                {
                    new { name = "weights", kind = "tensor", shape = new[] { rows, columns }, format = "NVFP4", layout = "blockscale-k16-m128x4-v1", offset = 0, bytes = Encoded.Length },
                    new { name = "input_divisor", kind = "tensor", shape = Array.Empty<int>(), format = "FP32", layout = "contiguous-le-v1", offset = inputOffset, bytes = 4 }
                }
            }));
            System.IO.Directory.CreateDirectory(_directory); Path = System.IO.Path.Combine(_directory, "fixture.ninfer");
            using (var stream = File.Create(Path))
            {
                stream.Write("NINFER\0\x02"u8); Span<byte> n = stackalloc byte[8]; BinaryPrimitives.WriteUInt64LittleEndian(n, (ulong)directory.Length); stream.Write(n); stream.Write(directory);
                long payload = (16L + directory.Length + 4095) / 4096 * 4096;
                stream.SetLength(payload + inputOffset + 4); stream.Position = payload; stream.Write(Encoded);
                stream.Position = payload + inputOffset; BinaryPrimitives.WriteUInt32LittleEndian(n[..4], inputBits); stream.Write(n[..4]);
            }
            Reader = new NinferArtifactReader(Path);
        }
        public void Dispose() { Reader.Dispose(); System.IO.Directory.Delete(_directory, true); }
    }

    // Source-to-destination mapping is deliberately the inverse direction of the converter.
    private static int[] ExpectedDestinationSources(int n, NinferNvfp4RowMapping mapping)
    {
        var result = Enumerable.Repeat(-1, n).ToArray();
        for (int s = 0; s < n; s++)
        {
            int d = s;
            if (mapping == NinferNvfp4RowMapping.GateUpRows8)
            {
                int part = s / (n / 2), r = s % (n / 2); d = r / 8 * 16 + part * 8 + r % 8;
            }
            else if (mapping == NinferNvfp4RowMapping.AttentionQueryGateKeyValue)
            {
                d = s < 6144 ? s / 256 * 512 + s % 256 : s < 7168 ? 12288 + s - 6144
                    : s < 13312 ? (s - 7168) / 256 * 512 + 256 + (s - 7168) % 256 : s;
            }
            Assert.Equal(-1, result[d]); result[d] = s;
        }
        Assert.DoesNotContain(-1, result); return result;
    }
    private static byte NativeCode(ReadOnlySpan<byte> codes, int row, int c, int k)
    {
        byte b = codes[row * (k / 2) + (c / 16) * 8 + c % 8];
        return (byte)((c % 16 < 8 ? b : b >> 4) & 15);
    }
    private static void CheckTile(NinferNvfp4PlaneTile tile, int[] originalRows)
    {
        var codes = tile.Codes.Span; var scales = tile.Scales.Span; int k = tile.Columns;
        for (int r = 0; r < 128; r++)
        {
            int s = originalRows[tile.FirstRow + r];
            for (int c = 0; c < k; c++) Assert.Equal(Code(s, c), NativeCode(codes, r, c, k));
            for (int g = 0; g < k / 16; g++) Assert.Equal(Scale(s, g), scales[r * (k / 16) + g]);
        }
    }
    private static byte[] InvertAll(NinferNvfp4PlaneReader reader)
    {
        int n = reader.Rows, k = reader.Columns; var original = new byte[n * k / 2 + n * k / 16];
        var logicalScales = new byte[n * k / 16]; int[] map = ExpectedDestinationSources(n, reader.RowMapping);
        for (int t = 0; t < reader.TileCount; t++)
        {
            var tile = reader.ReadTile(t); CheckTile(tile, map);
            for (int r = 0; r < 128; r++)
            {
                int s = map[tile.FirstRow + r];
                for (int c = 0; c < k; c += 2) original[s * (k / 2) + c / 2] = (byte)(NativeCode(tile.Codes.Span, r, c, k) | NativeCode(tile.Codes.Span, r, c + 1, k) << 4);
                tile.Scales.Span.Slice(r * (k / 16), k / 16).CopyTo(logicalScales.AsSpan(s * (k / 16)));
            }
        }
        int at = n * k / 2;
        for (int mb = 0; mb < n / 128; mb++) for (int kb = 0; kb < k / 64; kb++)
            for (int rm = 0; rm < 32; rm++) for (int q = 0; q < 4; q++) for (int lane = 0; lane < 4; lane++)
                original[at++] = logicalScales[(mb * 128 + q * 32 + rm) * (k / 16) + kb * 4 + lane];
        return original;
    }

    [Fact]
    public void AllSignedE2m1CodesAndAllUnsignedFiniteScaleBytesArePreserved()
    {
        using var f = new Fixture(128, 256); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights);
        var tile = convert.ReadTile(0); CheckTile(tile, ExpectedDestinationSources(128, NinferNvfp4RowMapping.Identity));
        Assert.Equal(Enumerable.Range(0, 16), Enumerable.Range(0, 256).Select(c => (int)NativeCode(tile.Codes.Span, 0, c, 256)).Distinct().Order());
        Assert.Equal(Enumerable.Range(0, 127).Select(x => (byte)x), tile.Scales.ToArray().Distinct().Order());
    }

    [Fact]
    public void IndependentInverseRestoresThreeTilesAndNonunitDivisorWithoutPayloadMutation()
    {
        using var f = new Fixture(384, 192, divisorBits: 0x3f800001); byte[] before = SHA256.HashData(File.ReadAllBytes(f.Path));
        using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights, f.Input);
        Assert.Equal(f.Encoded[..^4], InvertAll(convert));
        Assert.Equal(0x3f800001u, convert.WeightDivisorBits);
        Assert.Equal(before, SHA256.HashData(File.ReadAllBytes(f.Path)));
        for (int t = 0; t < 3; t++) Assert.Equal(new[] { t }, convert.ReadTile(t).OriginalSourceRowTiles);
    }

    [Fact]
    public void GateUpRowsReadBothNonadjacentSourceTilesAndInvertEveryRow()
    {
        using var f = new Fixture(512, 64); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: NinferNvfp4RowMapping.GateUpRows8);
        Assert.Equal(new[] { 0, 2 }, convert.ReadTile(0).OriginalSourceRowTiles);
        Assert.Equal(new[] { 1, 3 }, convert.ReadTile(3).OriginalSourceRowTiles);
        Assert.Equal(f.Encoded[..^4], InvertAll(convert));
    }

    [Fact]
    public void GqaKnownRowOrderReadsOriginalQGateAndKSourceBlocksRatherThanDestinationBlocks()
    {
        using var f = new Fixture(14336, 64); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: NinferNvfp4RowMapping.AttentionQueryGateKeyValue);
        Assert.Equal(new[] { 56 }, convert.ReadTile(2).OriginalSourceRowTiles); // first gate128
        Assert.Equal(new[] { 48 }, convert.ReadTile(96).OriginalSourceRowTiles); // first K128
        Assert.Equal(f.Encoded[..^4], InvertAll(convert));
    }

    [Fact]
    public void NativePlanesMatchIndependentLegacy36ByteGgufBlockRepack()
    {
        using var f = new Fixture(128, 256); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights);
        var tile = convert.ReadTile(0); var legacy = new byte[128 * (256 / 64) * 36];
        // Build logical GGUF block_nvfp4 bytes, independently of adjacent source bytes.
        for (int r = 0; r < 128; r++) for (int b = 0; b < 4; b++)
        {
            int at = (r * 4 + b) * 36;
            for (int s = 0; s < 4; s++)
            {
                legacy[at + s] = Scale(r, b * 4 + s);
                for (int j = 0; j < 8; j++) legacy[at + 4 + s * 8 + j] = (byte)(Code(r, b * 64 + s * 16 + j) | Code(r, b * 64 + s * 16 + j + 8) << 4);
            }
        }
        var expectedCodes = new byte[tile.Codes.Length]; var expectedScales = new byte[tile.Scales.Length];
        for (int block = 0; block < 128 * 4; block++)
        {
            legacy.AsSpan(block * 36, 4).CopyTo(expectedScales.AsSpan(block * 4));
            legacy.AsSpan(block * 36 + 4, 32).CopyTo(expectedCodes.AsSpan(block * 32));
        }
        Assert.Equal(expectedCodes, tile.Codes.ToArray()); Assert.Equal(expectedScales, tile.Scales.ToArray());
    }

    [Theory]
    [InlineData(0x7f)] [InlineData(0x80)] [InlineData(0x81)] [InlineData(0xfe)] [InlineData(0xff)]
    public void NativeIncompatibleScaleSignNegativeZeroOrNanNeverGetsClipped(int value)
    {
        using var f = new Fixture(invalidScaleIndex: 511, invalidScale: (byte)value); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights);
        var e = Assert.Throws<NotSupportedException>(() => convert.ReadTile(0)); Assert.Contains($"0x{value:x2}", e.Message);
    }

    [Fact]
    public void FailureInSecondGateUpSourceNeverReturnsAPartialDestinationTile()
    {
        using var f = new Fixture(256, 64, invalidScaleIndex: 512, invalidScale: 0x80); using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: NinferNvfp4RowMapping.GateUpRows8);
        NinferNvfp4PlaneTile? result = null;
        Assert.Throws<NotSupportedException>(() => result = convert.ReadTile(0)); Assert.Null(result);
    }

    [Fact]
    public void SeparateActivationDivisorEvenNanRemainsUnreadAndWeightBitsAreUncombined()
    {
        using var f = new Fixture(divisorBits: 0x40500000, inputBits: 0x7fc00001); long bytes = f.Reader.TotalBytesRead, calls = f.Reader.ReadOperations;
        using var convert = new NinferNvfp4PlaneReader(f.Reader, f.Weights, f.Input);
        Assert.Equal(bytes, f.Reader.TotalBytesRead); Assert.Equal(calls, f.Reader.ReadOperations);
        var tile = convert.ReadTile(0); Assert.Same(f.Input, tile.InputScaleDivisor);
        Assert.Equal(NinferQwen38ContentValidation.NotRead, tile.InputScaleDivisorContentValidation);
        Assert.Equal(0x40500000u, tile.WeightDivisorBits); Assert.Equal(3.25f, tile.WeightDivisor); Assert.True(tile.RequiresWeightDivisorAwareDispatch);
        Assert.Equal(bytes + convert.CodeBytesPerTile + convert.ScaleBytesPerTile, f.Reader.TotalBytesRead); Assert.Equal(calls + 2, f.Reader.ReadOperations);
    }

    [Fact]
    public void BoundsForeignOwnersAndDisposalHaveNoReadBypassOrReaderLifetimeTransfer()
    {
        using var a = new Fixture(); using var b = new Fixture();
        Assert.Throws<ArgumentException>(() => new NinferNvfp4PlaneReader(a.Reader, b.Weights));
        Assert.Throws<ArgumentException>(() => new NinferNvfp4PlaneReader(a.Reader, a.Weights, b.Input));
        Assert.Throws<InvalidDataException>(() => new NinferNvfp4PlaneReader(a.Reader, a.Input));
        var convert = new NinferNvfp4PlaneReader(a.Reader, a.Weights); long before = a.Reader.TotalBytesRead;
        foreach (int i in new[] { -1, 1, int.MaxValue }) Assert.Throws<ArgumentOutOfRangeException>(() => convert.ReadTile(i));
        Assert.Equal(before, a.Reader.TotalBytesRead); var tile = convert.ReadTile(0); byte[] retained = tile.Codes.ToArray(); convert.Dispose(); convert.Dispose();
        Assert.Throws<ObjectDisposedException>(() => convert.ReadTile(0)); Assert.Equal(retained, tile.Codes.ToArray());
        a.Reader.ReadObjectRange(a.Weights, 0, new byte[1]); // conversion did not dispose its caller's reader
        using var alive = new NinferNvfp4PlaneReader(b.Reader, b.Weights); b.Reader.Dispose();
        Assert.Throws<ObjectDisposedException>(() => alive.ReadTile(0));
    }

    [Fact]
    public void UnsupportedMappingsAndWidthBudgetAreRejectedBeforeTileAllocation()
    {
        using var f = new Fixture();
        Assert.Throws<InvalidDataException>(() => new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: NinferNvfp4RowMapping.GateUpRows8));
        Assert.Throws<InvalidDataException>(() => new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: NinferNvfp4RowMapping.AttentionQueryGateKeyValue));
        Assert.Throws<InvalidDataException>(() => new NinferNvfp4PlaneReader(f.Reader, f.Weights, mapping: (NinferNvfp4RowMapping)99));
        using var large = new Fixture(128, 65600);
        Assert.Throws<InvalidDataException>(() => new NinferNvfp4PlaneReader(large.Reader, large.Weights));
    }
}
