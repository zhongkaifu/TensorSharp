// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Lfm3
{
    /// <summary>
    /// A row table read on the host, in its GGUF format: rows are dequantized on demand (a dozen a frame: the depthformer's
    /// code embeddings, the LM feedback row's codebook embeddings), so a gather costs microseconds and no device round trip.
    /// The bytes are either its own copy (read from the GGUF) or borrowed from a <see cref="QuantizedWeight"/> that outlives
    /// it. Reads are serialized (one row scratch).
    /// </summary>
    internal sealed class Lfm3SpeechHostRows : IDisposable
    {
        private IntPtr _data;
        private readonly bool _owns;
        private readonly float[] _row;

        private Lfm3SpeechHostRows(IntPtr data, bool owns, int ggmlType, long rowWidth, long rows)
        {
            _data = data;
            _owns = owns;
            GgmlType = ggmlType;
            RowWidth = rowWidth;
            Rows = rows;
            RowBytes = Graph.NetworkWeights.RowBytes(ggmlType, rowWidth);
            _row = new float[rowWidth];
        }

        public int GgmlType { get; }
        public long RowWidth { get; }
        public long Rows { get; }
        public long RowBytes { get; }

        /// <summary>Host bytes held (0 when borrowed).</summary>
        public long OwnedBytes => _owns ? RowBytes * Rows : 0;

        /// <summary>Tensor <paramref name="name"/> of <paramref name="file"/> as rows of its ne0, copied to native memory.</summary>
        public static Lfm3SpeechHostRows Load(GgufFile file, string name)
        {
            if (!file.Tensors.TryGetValue(name, out GgufTensorInfo info)) throw new System.IO.InvalidDataException($"the GGUF has no tensor '{name}'");
            long width = (long)info.Shape[0], rows = info.NumElements / width;
            long bytes = file.GetTensorByteCount(info);
            IntPtr ptr = QuantizedWeight.AllocateBuffer(bytes);
            try
            {
                file.ReadTensorDataToNative(info, ptr, bytes);
            }
            catch
            {
                QuantizedWeight.FreeBuffer(ptr);
                throw;
            }
            return new Lfm3SpeechHostRows(ptr, true, (int)info.Type, width, rows);
        }

        /// <summary>Rows of <paramref name="rowWidth"/> values from the raw bytes of GGUF type <paramref name="ggmlType"/>,
        /// copied.</summary>
        public static Lfm3SpeechHostRows FromRaw(int ggmlType, ReadOnlySpan<byte> raw, long rowWidth)
        {
            long rowBytes = Graph.NetworkWeights.RowBytes(ggmlType, rowWidth);
            if (raw.Length == 0 || raw.Length % rowBytes != 0) throw new ArgumentException($"{raw.Length} bytes are not rows of {rowBytes}");
            IntPtr ptr = QuantizedWeight.AllocateBuffer(raw.Length);
            unsafe
            {
                fixed (byte* src = raw) Buffer.MemoryCopy(src, (void*)ptr, raw.Length, raw.Length);
            }
            return new Lfm3SpeechHostRows(ptr, true, ggmlType, rowWidth, raw.Length / rowBytes);
        }

        /// <summary>F32 rows of <paramref name="rowWidth"/> values, copied.</summary>
        public static Lfm3SpeechHostRows FromFloats(float[] values, long rowWidth)
            => FromRaw(0, System.Runtime.InteropServices.MemoryMarshal.AsBytes(values.AsSpan()), rowWidth);

        /// <summary>The host bytes of <paramref name="weight"/> (borrowed: the weight must outlive this table).</summary>
        public static Lfm3SpeechHostRows Borrow(QuantizedWeight weight)
        {
            if (weight == null || !weight.HasHostData) throw new ArgumentException("a weight with host bytes", nameof(weight));
            return new Lfm3SpeechHostRows(weight.Data, false, weight.GgmlType, weight.Ne0, weight.Ne1);
        }

        /// <summary>Row <paramref name="row"/> as F32 into <paramref name="destination"/> (RowWidth values).</summary>
        public void Read(long row, Span<float> destination)
        {
            lock (_row)
            {
                Dequantize(row);
                _row.AsSpan().CopyTo(destination);
            }
        }

        /// <summary>Add row <paramref name="row"/> into <paramref name="accumulator"/> (RowWidth values).</summary>
        public void AddTo(long row, Span<float> accumulator)
        {
            lock (_row)
            {
                Dequantize(row);
                for (int i = 0; i < _row.Length; i++) accumulator[i] += _row[i];
            }
        }

        private void Dequantize(long row)
        {
            ObjectDisposedException.ThrowIf(_data == IntPtr.Zero, this);
            if ((ulong)row >= (ulong)Rows) throw new ArgumentOutOfRangeException(nameof(row), $"row {row} of {Rows}");
            NativeDequant.DequantizeToFloat32(GgmlType, _data + (nint)(row * RowBytes), _row, 0, RowWidth);
        }

        public void Dispose()
        {
            if (_owns && _data != IntPtr.Zero) QuantizedWeight.FreeBuffer(_data);
            _data = IntPtr.Zero;
        }
    }
}
