// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace TensorSharp.Cuda;

/// <summary>CPU-owned original W8G32/F16-scale bytes. A successful load no longer borrows
/// its source. No dequantized full embedding or TensorSharp CPU tensor backend is used.</summary>
internal sealed unsafe class CudaNinferHostEmbedding : IDisposable
{
    internal const long MaximumPayloadBytes = 1_350_860_800;
    internal const int MaximumReadBytes = 8 << 20;
    private readonly object _gate = new();
    private IntPtr _payload;
    private bool _disposed;
    internal int Vocabulary { get; }
    internal int Width { get; }
    internal long PayloadBytes { get; }
    internal long CodeBytes { get; }
    internal string PayloadSha256 { get; }

    internal CudaNinferHostEmbedding(CudaNinferProjectionDesc descriptor, long payloadBudgetBytes = MaximumPayloadBytes)
    {
        ArgumentNullException.ThrowIfNull(descriptor);
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("NInfer host embedding requires little-endian FP16 storage.");
        if (descriptor.Format != CudaNinferProjectionFormat.W8G32 || descriptor.Columns % 128 != 0
            || payloadBudgetBytes <= 0 || payloadBudgetBytes > MaximumPayloadBytes || descriptor.Bytes > payloadBudgetBytes)
            throw new ArgumentException("Original W8G32 geometry and the bounded host-payload budget are required.", nameof(descriptor));
        Vocabulary = descriptor.Rows; Width = descriptor.Columns; PayloadBytes = descriptor.Bytes;
        CodeBytes = checked((long)Vocabulary * Width);
        if (PayloadBytes != checked(CodeBytes + CodeBytes / 32 * sizeof(ushort)))
            throw new ArgumentException("Embedding payload geometry differs.", nameof(descriptor));
        IntPtr allocation = (IntPtr)NativeMemory.Alloc(checked((nuint)PayloadBytes));
        if (allocation == IntPtr.Zero) throw new OutOfMemoryException("NInfer host embedding payload allocation failed.");
        try
        {
            using var digest = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            for (long offset = 0; offset < PayloadBytes;)
            {
                int count = checked((int)Math.Min(MaximumReadBytes, PayloadBytes - offset));
                IntPtr destination = allocation + (nint)offset;
                descriptor.Source.Read(checked(descriptor.SourceOffset + offset), destination, count);
                var bytes = new ReadOnlySpan<byte>((void*)destination, count);
                ValidateScaleChunk(bytes, offset, CodeBytes, PayloadBytes);
                digest.AppendData(bytes);
                offset += count;
            }
            PayloadSha256 = Convert.ToHexString(digest.GetHashAndReset()).ToLowerInvariant();
            _payload = allocation;
            allocation = IntPtr.Zero;
        }
        finally
        {
            if (allocation != IntPtr.Zero) NativeMemory.Free((void*)allocation);
        }
    }

    internal static void ValidateScaleChunk(ReadOnlySpan<byte> bytes, long offset, long codeBytes, long totalBytes)
    {
        if (offset < 0 || (offset & 1) != 0 || (bytes.Length & 1) != 0 || codeBytes < 0 || (codeBytes & 1) != 0
            || totalBytes < codeBytes || offset > totalBytes || bytes.Length > totalBytes - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
        long begin = Math.Max(offset, codeBytes);
        for (long at = begin; at < offset + bytes.Length; at += sizeof(ushort))
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(bytes.Slice(checked((int)(at - offset)), sizeof(ushort)));
            if ((bits & 0x7c00) == 0x7c00)
                throw new InvalidOperationException("The original W8 embedding contains a nonfinite FP16 group scale.");
        }
    }

    /// <summary>Writes BF16-rounded F32 embedding rows into a caller-owned buffer. Every token
    /// is validated before any destination write; valid calls allocate no managed memory.</summary>
    internal void GatherRows(ReadOnlySpan<int> tokenIds, Span<float> destination)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            int values = checked(tokenIds.Length * Width);
            if (destination.Length < values) throw new ArgumentException("Embedding destination is too small.", nameof(destination));
            foreach (int token in tokenIds)
                if ((uint)token >= (uint)Vocabulary) throw new ArgumentOutOfRangeException(nameof(tokenIds), "Embedding token is outside the physical vocabulary.");
            var codes = (sbyte*)_payload;
            var scales = (ushort*)((byte*)_payload + CodeBytes);
            int groups = Width / 32;
            for (int row = 0; row < tokenIds.Length; row++)
            {
                int token = tokenIds[row];
                long codeRow = (long)token * Width, scaleRow = (long)token * groups;
                int outputRow = row * Width;
                for (int group = 0; group < groups; group++)
                {
                    float scale = (float)BitConverter.UInt16BitsToHalf(scales[scaleRow + group]);
                    for (int lane = 0; lane < 32; lane++)
                    {
                        int column = group * 32 + lane;
                        // int8 * finite FP16 is exactly representable in F32: at most
                        // 18 significant bits and magnitude below 8.4 million.
                        float value = (float)codes[codeRow + column] * scale;
                        destination[outputRow + column] = RoundFiniteBfloat(value);
                    }
                }
            }
        }
    }

    internal static float RoundFiniteBfloat(float value)
    {
        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
        return BitConverter.UInt32BitsToSingle(rounded & 0xffff0000u);
    }

    private void Release()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _disposed = true;
            var payload = _payload; _payload = IntPtr.Zero;
            if (payload != IntPtr.Zero) NativeMemory.Free((void*)payload);
        }
    }

    public void Dispose() { Release(); GC.SuppressFinalize(this); }
    ~CudaNinferHostEmbedding() => Release();
}
