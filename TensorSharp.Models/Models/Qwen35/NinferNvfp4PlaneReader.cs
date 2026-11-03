// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public enum NinferNvfp4RowMapping { Identity, GateUpRows8, AttentionQueryGateKeyValue }

/// <summary>One complete successful 128-row conversion. Code/scale bytes are never requantized.
/// The original weight divisor and the separate unread activation-divisor descriptor remain explicit.</summary>
public sealed class NinferNvfp4PlaneTile
{
    public int FirstRow { get; }
    public int Rows => NinferNvfp4PlaneReader.TileRows;
    public int Columns { get; }
    public ReadOnlyMemory<byte> Codes { get; }
    public ReadOnlyMemory<byte> Scales { get; }
    public IReadOnlyList<int> OriginalSourceRowTiles { get; }
    public uint WeightDivisorBits { get; }
    public float WeightDivisor => BitConverter.Int32BitsToSingle(unchecked((int)WeightDivisorBits));
    public NinferTensorDescriptor? InputScaleDivisor { get; }
    public NinferQwen38ContentValidation InputScaleDivisorContentValidation => NinferQwen38ContentValidation.NotRead;
    public string Layout => "q35-native-nvfp4-code16-low8-high8-scale-rowmajor/1";
    /// <summary>Existing Q35 Planes/alpha=1 dispatch cannot silently drop the retained divisor.</summary>
    public bool RequiresWeightDivisorAwareDispatch => true;
    internal NinferNvfp4PlaneTile(int firstRow, int columns, byte[] codes, byte[] scales,
        int[] sourceTiles, uint divisor, NinferTensorDescriptor? inputDivisor)
        => (FirstRow, Columns, Codes, Scales, OriginalSourceRowTiles, WeightDivisorBits, InputScaleDivisor)
            = (firstRow, columns, codes, scales, Array.AsReadOnly(sourceTiles), divisor, inputDivisor);
}

/// <summary>
/// Bounded, read-only NInfer-v2 NVFP4 layout conversion, with no CUDA or model integration.
/// Pinned Ignis e600 nvfp4_codec.cuh uses adjacent-K nibbles and m128x4 scale swizzling;
/// TS tensorsharp_q35_kernels.cu stores each16 codes as low[0..7]/high[8..15] and row-major scales.
/// This only permutes encoded bytes. It does not combine a divisor, evaluate weights, or establish
/// activation precision/F32 arithmetic equivalence. Caller owns the reader and file-identity guard.
/// Kernel dispatch geometry is a separate gate. At the maximum K, internal source buffers plus
/// one returned tile total9MiB; callers must consume/release tiles rather than collect a full tensor.
/// </summary>
public sealed class NinferNvfp4PlaneReader : IDisposable
{
    public const int TileRows = 128, MaximumColumns = 65536;
    private readonly NinferArtifactReader _reader;
    private readonly object _gate = new();
    private byte[] _sourceCodes, _sourceScales;
    private bool _disposed;
    public NinferTensorDescriptor OriginalWeights { get; }
    public NinferTensorDescriptor? InputScaleDivisor { get; }
    public NinferNvfp4RowMapping RowMapping { get; }
    public int Rows { get; }
    public int Columns { get; }
    public int TileCount => Rows / TileRows;
    public int CodeBytesPerTile => TileRows * (Columns / 2);
    public int ScaleBytesPerTile => TileRows * (Columns / 16);
    public uint WeightDivisorBits { get; }

    public static NinferNvfp4PlaneReader FromTarget(NinferArtifactReader reader, NinferQwen38TensorBinding tensor)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        var mapping = tensor.RowOrder switch
        {
            NinferQwen38RowOrder.Unchanged or NinferQwen38RowOrder.GdnQueryKeyValueZ => NinferNvfp4RowMapping.Identity,
            NinferQwen38RowOrder.MlpGateThenUp => NinferNvfp4RowMapping.GateUpRows8,
            NinferQwen38RowOrder.AttentionQueryKeyGateValue => NinferNvfp4RowMapping.AttentionQueryGateKeyValue,
            _ => throw new NotSupportedException("This semantic row order is not an NVFP4 plane conversion.")
        };
        return new(reader, tensor.Descriptor, tensor.InputScaleDivisor, mapping);
    }

    // The internal entry permits small, real-reader CPU fixtures without a fake 906-tensor plan.
    // It never creates foreign/unowned descriptors or opens another artifact path.
    internal NinferNvfp4PlaneReader(NinferArtifactReader reader, NinferTensorDescriptor weights,
        NinferTensorDescriptor? inputDivisor = null, NinferNvfp4RowMapping mapping = NinferNvfp4RowMapping.Identity)
    {
        ArgumentNullException.ThrowIfNull(reader); ArgumentNullException.ThrowIfNull(weights);
        reader.ReadObjectRange(weights, 0, Span<byte>.Empty); // Own-reader + disposed check, no payload read.
        if (inputDivisor != null)
        {
            reader.ReadObjectRange(inputDivisor, 0, Span<byte>.Empty);
            Require(inputDivisor.Format == NinferTensorFormat.Float32 && inputDivisor.Layout == NinferTensorLayout.ContiguousLittleEndianV1
                && inputDivisor.Shape.Count == 0 && inputDivisor.ByteLength == 4 && inputDivisor.WeightDivisor == null,
                "Activation divisor must remain a separate raw FP32 scalar descriptor.");
        }
        Require(weights.Format == NinferTensorFormat.NvFp4 && weights.Layout == NinferTensorLayout.BlockScaleK16M128x4V1
            && weights.Shape.Count == 2 && weights.Shape[0] <= int.MaxValue && weights.Shape[1] <= MaximumColumns,
            "Expected bounded NInfer v2 NVFP4 m128x4 tensor.");
        Rows = checked((int)weights.Shape[0]); Columns = checked((int)weights.Shape[1]);
        Require(Rows > 0 && Rows % TileRows == 0 && Columns > 0 && Columns % 64 == 0, "Invalid NVFP4 row/column geometry.");
        ulong count = checked((ulong)Rows * (ulong)Columns);
        var expected = new NinferTensorGeometry(count / 2 + count / 16 + 4, 256, LowPlaneBytes: count / 2,
            ScalePlaneOffset: count / 2, ScalePlaneBytes: count / 16, DivisorOffset: count / 2 + count / 16);
        Require(weights.Geometry == expected && weights.ByteLength == expected.EncodedBytes, "Original encoded planes differ.");
        Require(weights.WeightDivisor is float divisor && float.IsFinite(divisor) && divisor > 0
            && weights.WeightDivisorBits is uint bits && bits == unchecked((uint)BitConverter.SingleToInt32Bits(divisor)),
            "Original positive-finite weight divisor bits are required.");
        if (mapping == NinferNvfp4RowMapping.GateUpRows8) Require(Rows % 256 == 0, "Gate/up halves must contain whole128-row source tiles.");
        else if (mapping == NinferNvfp4RowMapping.AttentionQueryGateKeyValue) Require(Rows == 14336, "Only the known Q6144|K1024|gate6144|V1024 order is supported.");
        else Require(mapping == NinferNvfp4RowMapping.Identity, "Unknown row mapping.");
        _reader = reader; OriginalWeights = weights; InputScaleDivisor = inputDivisor; RowMapping = mapping;
        WeightDivisorBits = weights.WeightDivisorBits!.Value;
        _sourceCodes = new byte[CodeBytesPerTile]; _sourceScales = new byte[ScaleBytesPerTile];
    }

    public int SourceRowForDestination(int row)
    {
        if ((uint)row >= (uint)Rows) throw new ArgumentOutOfRangeException(nameof(row));
        return RowMapping switch
        {
            NinferNvfp4RowMapping.Identity => row,
            NinferNvfp4RowMapping.GateUpRows8 => row / 16 * 8 + row % 8 + (row % 16 >= 8 ? Rows / 2 : 0),
            NinferNvfp4RowMapping.AttentionQueryGateKeyValue when row < 12288 => row / 512 * 256 + row % 256 + (row % 512 >= 256 ? 7168 : 0),
            NinferNvfp4RowMapping.AttentionQueryGateKeyValue when row < 13312 => 6144 + row - 12288,
            _ => row
        };
    }

    /// <summary>Reads full source128-row blocks, then returns a complete converted destination tile.
    /// Gate/up can use two nonadjacent source blocks; GQA never assumes source equals destination.
    /// Every original E2M1 nibble, including negative codes/negativezero, is preserved.
    /// E4M3FN sign/negativezero/NaN bytes are rejected because existing TS UE decoding is incompatible.</summary>
    public NinferNvfp4PlaneTile ReadTile(int tileIndex)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if ((uint)tileIndex >= (uint)TileCount) throw new ArgumentOutOfRangeException(nameof(tileIndex));
            int first = checked(tileIndex * TileRows), groups = Columns / 16;
            Span<int> blocks = stackalloc int[2]; int blockCount = 0;
            for (int r = 0; r < TileRows; r++)
            {
                int sourceBlock = SourceRowForDestination(first + r) / TileRows;
                bool found = false; for (int i = 0; i < blockCount; i++) found |= blocks[i] == sourceBlock;
                if (!found)
                {
                    if (blockCount == blocks.Length) throw new InvalidOperationException("Row mapping exceeded the proved two-source-tile bound.");
                    blocks[blockCount++] = sourceBlock;
                }
            }
            // These arrays remain private until ALL selected source blocks pass. No partial tile
            // is returned on read/scale failure, even if an earlier source block already converted.
            byte[] codes = new byte[CodeBytesPerTile], scales = new byte[ScaleBytesPerTile];
            for (int bi = 0; bi < blockCount; bi++)
            {
                int block = blocks[bi];
                _reader.ReadObjectRange(OriginalWeights, checked((ulong)block * (ulong)CodeBytesPerTile), _sourceCodes);
                _reader.ReadObjectRange(OriginalWeights, checked(OriginalWeights.Geometry.ScalePlaneOffset!.Value + (ulong)block * (ulong)ScaleBytesPerTile), _sourceScales);
                for (int i = 0; i < _sourceScales.Length; i++)
                    if (_sourceScales[i] >= 0x7f)
                        throw new NotSupportedException($"Original E4M3FN scale byte 0x{_sourceScales[i]:x2} in source tile {block} is incompatible with TS unsigned finite scale semantics; no clipping/sign removal/NaN substitution is permitted.");
                for (int r = 0; r < TileRows; r++)
                {
                    int original = SourceRowForDestination(first + r);
                    if (original / TileRows != block) continue;
                    int sr = original % TileRows, srcRow = sr * (Columns / 2), dstRow = r * (Columns / 2);
                    for (int g = 0; g < groups; g++)
                    {
                        int src = srcRow + g * 8, dst = dstRow + g * 8;
                        for (int j = 0; j < 8; j++)
                        {
                            int shift = (j & 1) * 4;
                            codes[dst + j] = (byte)(((_sourceCodes[src + (j >> 1)] >> shift) & 15)
                                | (((_sourceCodes[src + 4 + (j >> 1)] >> shift) & 15) << 4));
                        }
                        scales[r * groups + g] = _sourceScales[(g / 4) * 512 + (sr % 32) * 16 + (sr / 32) * 4 + g % 4];
                    }
                }
            }
            return new(first, Columns, codes, scales, blocks[..blockCount].ToArray(), WeightDivisorBits, InputScaleDivisor);
        }
    }

    public void Dispose()
    {
        lock (_gate) { _disposed = true; _sourceCodes = Array.Empty<byte>(); _sourceScales = Array.Empty<byte>(); }
        // Reader and any successfully returned tile remain owned by their callers.
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
}
