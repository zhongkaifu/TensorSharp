// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// A common read surface for "fetch a named tensor as a flat F32 array" so model code can be
    /// fed weights from either a GGUF (<see cref="GgufFloatTensorStore"/>) or a safetensors file
    /// (<see cref="SafetensorsFile"/> / <see cref="SafetensorsModel"/>) without caring which.
    /// </summary>
    public interface IFloatTensorStore
    {
        bool HasTensor(string name);

        /// <summary>Fetch a tensor by name as a freshly-allocated flat F32 array in the file's
        /// native element order (row-major for safetensors / PyTorch). Throws if absent.</summary>
        float[] ReadFloat32(string name);

        /// <summary>Logical shape of the named tensor (row-major, outermost dim first). Empty if absent.</summary>
        long[] TensorShape(string name);
    }

    /// <summary>safetensors element type. See https://github.com/huggingface/safetensors. </summary>
    public enum SafetensorDtype
    {
        Unknown = 0,
        F64, F32, F16, BF16,
        I64, I32, I16, I8, U8, Bool,
        F8_E4M3, F8_E5M2,
    }

    public sealed class SafetensorTensorInfo
    {
        public string Name { get; init; } = string.Empty;
        /// <summary>PyTorch/row-major shape (outermost dim first). A 0-d scalar has an empty shape.</summary>
        public long[] Shape { get; init; } = Array.Empty<long>();
        public SafetensorDtype Dtype { get; init; }
        /// <summary>Byte offset of this tensor's data, relative to the start of the data section.</summary>
        public long Begin { get; init; }
        public long End { get; init; }

        public long ByteCount => End - Begin;

        public long NumElements
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= d;
                return n;
            }
        }
    }

    /// <summary>
    /// Reads a single <c>.safetensors</c> file. The format is an 8-byte little-endian u64 giving the
    /// JSON header length, the UTF-8 JSON header (<c>{ name: {dtype, shape, data_offsets:[begin,end]}, ... }</c>),
    /// then the raw tensor blob — every tensor's <c>data_offsets</c> are relative to the start of that blob.
    /// The file is memory-mapped so tensor reads are zero-copy page-ins; <see cref="ReadFloat32"/> converts
    /// BF16/F16/F64/integer tensors to F32 on the fly (BF16 via a vectorised top-16-bit widen, which is
    /// bit-identical to the canonical numpy <c>(u16&lt;&lt;16).view(f32)</c> upcast).
    /// </summary>
    public sealed class SafetensorsFile : IFloatTensorStore, IDisposable
    {
        public string Path { get; }
        public Dictionary<string, SafetensorTensorInfo> Tensors { get; } = new(StringComparer.Ordinal);
        /// <summary>The optional <c>__metadata__</c> string map from the header (may be empty).</summary>
        public Dictionary<string, string> Metadata { get; } = new(StringComparer.Ordinal);

        /// <summary>Absolute byte offset where the tensor data blob begins (8 + header length).</summary>
        public long DataOffset { get; private set; }

        private readonly long _fileLength;
        private MemoryMappedFile? _mappedFile;
        private MemoryMappedViewAccessor? _mappedView;
        private unsafe byte* _mappedBase;
        private bool _mappedPointerAcquired;

        public SafetensorsFile(string path)
        {
            Path = path;
            _fileLength = new FileInfo(path).Length;
            ParseHeader();
        }

        private void ParseHeader()
        {
            using var fs = File.OpenRead(Path);
            Span<byte> lenBytes = stackalloc byte[8];
            fs.ReadExactly(lenBytes);
            ulong headerLen = BitConverter.ToUInt64(lenBytes);
            if (headerLen == 0 || (long)headerLen > _fileLength - 8)
                throw new InvalidDataException($"safetensors header length {headerLen} is out of range for '{Path}' ({_fileLength} bytes).");

            byte[] headerBytes = new byte[headerLen];
            fs.ReadExactly(headerBytes, 0, (int)headerLen);
            DataOffset = 8 + (long)headerLen;

            long dataBytes = _fileLength - DataOffset;
            using var doc = JsonDocument.Parse(headerBytes);
            foreach (var prop in doc.RootElement.EnumerateObject())
            {
                if (prop.NameEquals("__metadata__"))
                {
                    if (prop.Value.ValueKind == JsonValueKind.Object)
                        foreach (var m in prop.Value.EnumerateObject())
                            Metadata[m.Name] = m.Value.ToString();
                    continue;
                }

                var el = prop.Value;
                var dtype = ParseDtype(el.GetProperty("dtype").GetString());

                var shapeEl = el.GetProperty("shape");
                var shape = new long[shapeEl.GetArrayLength()];
                int si = 0;
                foreach (var d in shapeEl.EnumerateArray()) shape[si++] = d.GetInt64();

                var off = el.GetProperty("data_offsets");
                long begin = off[0].GetInt64();
                long end = off[1].GetInt64();
                if (begin < 0 || end < begin || end > dataBytes)
                    throw new InvalidDataException($"safetensors tensor '{prop.Name}' has data_offsets [{begin},{end}] outside the {dataBytes}-byte data section of '{Path}'.");

                var info = new SafetensorTensorInfo { Name = prop.Name, Shape = shape, Dtype = dtype, Begin = begin, End = end };
                long expected = info.NumElements * DtypeSize(dtype);
                if (info.ByteCount != expected)
                    throw new InvalidDataException($"safetensors tensor '{prop.Name}' byte length {info.ByteCount} != {expected} expected for {dtype}{Stringify(shape)}.");
                Tensors[prop.Name] = info;
            }
        }

        public bool HasTensor(string name) => Tensors.ContainsKey(name);

        public long[] TensorShape(string name) =>
            Tensors.TryGetValue(name, out var info) ? (long[])info.Shape.Clone() : Array.Empty<long>();

        public SafetensorTensorInfo GetInfo(string name) =>
            Tensors.TryGetValue(name, out var info) ? info
                : throw new KeyNotFoundException($"safetensors tensor not found: {name}");

        /// <summary>Raw on-disk bytes of a tensor (no dtype conversion), copied out of the mmap.</summary>
        public byte[] ReadRawBytes(string name)
        {
            var info = GetInfo(name);
            var dst = new byte[info.ByteCount];
            CopyRaw(info, dst);
            return dst;
        }

        private unsafe void CopyRaw(SafetensorTensorInfo info, Span<byte> dst)
        {
            EnsureMappedView();
            var src = new ReadOnlySpan<byte>(_mappedBase + DataOffset + info.Begin, (int)info.ByteCount);
            src.CopyTo(dst);
        }

        public float[] ReadFloat32(string name)
        {
            var info = GetInfo(name);
            long n = info.NumElements;
            var dst = new float[n];
            ReadTensorDataToFloat32(info, dst, n);
            return dst;
        }

        /// <summary>
        /// Convert the named tensor into <paramref name="dest"/> as F32 (length <paramref name="numElements"/>),
        /// reading straight from the memory-mapped blob.
        /// </summary>
        public unsafe void ReadTensorDataToFloat32(SafetensorTensorInfo info, float[] dest, long numElements)
        {
            if (numElements > dest.LongLength)
                throw new ArgumentException($"destination too small: {dest.LongLength} < {numElements}");
            EnsureMappedView();
            byte* src = _mappedBase + DataOffset + info.Begin;
            fixed (float* d = dest)
            {
                ConvertToFloat32(info.Dtype, src, d, numElements);
            }
        }

        /// <summary>Pointer to the tensor's raw mmap bytes (valid only while this file is not disposed).
        /// Convenient for zero-copy upload of already-F32/-F16/-BF16 weights.</summary>
        public unsafe bool TryGetTensorDataPointer(SafetensorTensorInfo info, out IntPtr dataPtr)
        {
            EnsureMappedView();
            dataPtr = (IntPtr)(_mappedBase + DataOffset + info.Begin);
            return _mappedBase != null;
        }

        // ---- dtype conversion -------------------------------------------------

        private static unsafe void ConvertToFloat32(SafetensorDtype dtype, byte* src, float* dst, long n)
        {
            switch (dtype)
            {
                case SafetensorDtype.F32:
                    Buffer.MemoryCopy(src, dst, n * 4, n * 4);
                    break;
                case SafetensorDtype.BF16:
                    SafetensorsKernels.Bf16ToF32((ushort*)src, dst, n);
                    break;
                case SafetensorDtype.F16:
                    for (long i = 0; i < n; i++) dst[i] = (float)BitConverter.UInt16BitsToHalf(((ushort*)src)[i]);
                    break;
                case SafetensorDtype.F64:
                    for (long i = 0; i < n; i++) dst[i] = (float)((double*)src)[i];
                    break;
                case SafetensorDtype.I64:
                    for (long i = 0; i < n; i++) dst[i] = ((long*)src)[i];
                    break;
                case SafetensorDtype.I32:
                    for (long i = 0; i < n; i++) dst[i] = ((int*)src)[i];
                    break;
                case SafetensorDtype.I16:
                    for (long i = 0; i < n; i++) dst[i] = ((short*)src)[i];
                    break;
                case SafetensorDtype.I8:
                    for (long i = 0; i < n; i++) dst[i] = ((sbyte*)src)[i];
                    break;
                case SafetensorDtype.U8:
                case SafetensorDtype.Bool:
                    for (long i = 0; i < n; i++) dst[i] = src[i];
                    break;
                default:
                    throw new NotSupportedException($"safetensors dtype {dtype} cannot be converted to F32.");
            }
        }

        private static SafetensorDtype ParseDtype(string? s) => s?.ToUpperInvariant() switch
        {
            "F64" or "FLOAT64" => SafetensorDtype.F64,
            "F32" or "FLOAT32" => SafetensorDtype.F32,
            "F16" or "FLOAT16" => SafetensorDtype.F16,
            "BF16" or "BFLOAT16" => SafetensorDtype.BF16,
            "I64" or "INT64" => SafetensorDtype.I64,
            "I32" or "INT32" => SafetensorDtype.I32,
            "I16" or "INT16" => SafetensorDtype.I16,
            "I8" or "INT8" => SafetensorDtype.I8,
            "U8" or "UINT8" => SafetensorDtype.U8,
            "BOOL" => SafetensorDtype.Bool,
            "F8_E4M3" or "F8_E4M3FN" => SafetensorDtype.F8_E4M3,
            "F8_E5M2" or "F8_E5M2FNUZ" => SafetensorDtype.F8_E5M2,
            _ => throw new NotSupportedException($"unknown safetensors dtype '{s}'"),
        };

        internal static int DtypeSize(SafetensorDtype d) => d switch
        {
            SafetensorDtype.F64 or SafetensorDtype.I64 => 8,
            SafetensorDtype.F32 or SafetensorDtype.I32 => 4,
            SafetensorDtype.F16 or SafetensorDtype.BF16 or SafetensorDtype.I16 => 2,
            SafetensorDtype.I8 or SafetensorDtype.U8 or SafetensorDtype.Bool
                or SafetensorDtype.F8_E4M3 or SafetensorDtype.F8_E5M2 => 1,
            _ => throw new NotSupportedException($"unknown safetensors dtype {d}"),
        };

        private static string Stringify(long[] shape) => "[" + string.Join(",", shape) + "]";

        private unsafe void EnsureMappedView()
        {
            if (_mappedBase != null) return;
            _mappedFile ??= MemoryMappedFile.CreateFromFile(Path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            _mappedView ??= _mappedFile.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* viewPtr = null;
            _mappedView.SafeMemoryMappedViewHandle.AcquirePointer(ref viewPtr);
            viewPtr += _mappedView.PointerOffset;
            _mappedBase = viewPtr;
            _mappedPointerAcquired = true;
        }

        public unsafe void Dispose()
        {
            if (_mappedPointerAcquired && _mappedView != null)
            {
                _mappedView.SafeMemoryMappedViewHandle.ReleasePointer();
                _mappedPointerAcquired = false;
                _mappedBase = null;
            }
            _mappedView?.Dispose();
            _mappedView = null;
            _mappedFile?.Dispose();
            _mappedFile = null;
        }
    }

    /// <summary>
    /// A multi-shard safetensors model: opens a single <c>.safetensors</c>, a
    /// <c>model.safetensors.index.json</c> (HuggingFace <c>weight_map</c> sharding), or a directory of
    /// <c>*.safetensors</c>, and routes by tensor name to the owning shard. Implements
    /// <see cref="IFloatTensorStore"/> so it is a drop-in weight source.
    /// </summary>
    public sealed class SafetensorsModel : IFloatTensorStore, IDisposable
    {
        private readonly List<SafetensorsFile> _files = new();
        private readonly Dictionary<string, SafetensorsFile> _byName = new(StringComparer.Ordinal);

        private SafetensorsModel() { }

        public IReadOnlyDictionary<string, SafetensorsFile> TensorOwners => _byName;
        public IReadOnlyList<SafetensorsFile> Files => _files;
        public int Count => _byName.Count;

        /// <summary>Open a single file, a <c>*.index.json</c>, or a directory containing either.</summary>
        public static SafetensorsModel Open(string pathOrDir)
        {
            var m = new SafetensorsModel();
            try
            {
                if (Directory.Exists(pathOrDir))
                {
                    string index = System.IO.Path.Combine(pathOrDir, "model.safetensors.index.json");
                    if (File.Exists(index)) m.LoadIndex(index);
                    else
                    {
                        var shards = Directory.GetFiles(pathOrDir, "*.safetensors");
                        if (shards.Length == 0)
                            throw new FileNotFoundException($"no .safetensors files found in '{pathOrDir}'.");
                        Array.Sort(shards, StringComparer.Ordinal);
                        foreach (var s in shards) m.AddFile(s);
                    }
                }
                else if (pathOrDir.EndsWith(".index.json", StringComparison.OrdinalIgnoreCase))
                {
                    m.LoadIndex(pathOrDir);
                }
                else
                {
                    m.AddFile(pathOrDir);
                }
            }
            catch
            {
                m.Dispose();
                throw;
            }
            return m;
        }

        private void LoadIndex(string indexPath)
        {
            string dir = System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(indexPath)) ?? ".";
            using var doc = JsonDocument.Parse(File.ReadAllBytes(indexPath));
            var weightMap = doc.RootElement.GetProperty("weight_map");
            var shardNames = new SortedSet<string>(StringComparer.Ordinal);
            foreach (var w in weightMap.EnumerateObject())
                shardNames.Add(w.Value.GetString()!);
            foreach (var shard in shardNames)
                AddFile(System.IO.Path.Combine(dir, shard));
        }

        private void AddFile(string path)
        {
            var f = new SafetensorsFile(path);
            _files.Add(f);
            foreach (var name in f.Tensors.Keys)
                _byName[name] = f; // last writer wins (shards never overlap in practice)
        }

        public bool HasTensor(string name) => _byName.ContainsKey(name);

        public float[] ReadFloat32(string name) =>
            (_byName.TryGetValue(name, out var f) ? f : throw new KeyNotFoundException($"safetensors tensor not found: {name}"))
                .ReadFloat32(name);

        public long[] TensorShape(string name) =>
            _byName.TryGetValue(name, out var f) ? f.TensorShape(name) : Array.Empty<long>();

        public IEnumerable<string> TensorNames => _byName.Keys;

        public void Dispose()
        {
            foreach (var f in _files) f.Dispose();
            _files.Clear();
            _byName.Clear();
        }
    }

    /// <summary>Adapts a <see cref="GgufFile"/> to <see cref="IFloatTensorStore"/> so the same model
    /// weight-loading code can be fed from a GGUF without changes.</summary>
    public sealed class GgufFloatTensorStore : IFloatTensorStore
    {
        private readonly GgufFile _gguf;
        public GgufFloatTensorStore(GgufFile gguf) { _gguf = gguf; }

        public bool HasTensor(string name) => _gguf.Tensors.ContainsKey(name);

        public float[] ReadFloat32(string name)
        {
            if (!_gguf.Tensors.TryGetValue(name, out var info))
                throw new KeyNotFoundException($"GGUF tensor not found: {name}");
            long n = info.NumElements;
            var dst = new float[n];
            _gguf.ReadTensorDataToFloat32(info, dst, n);
            return dst;
        }

        public long[] TensorShape(string name)
        {
            if (!_gguf.Tensors.TryGetValue(name, out var info)) return Array.Empty<long>();
            // GGUF stores ne0-fastest; reverse to a row-major/outermost-first logical shape.
            var ne = info.Shape;
            var shape = new long[ne.Length];
            for (int i = 0; i < ne.Length; i++) shape[i] = (long)ne[ne.Length - 1 - i];
            return shape;
        }
    }

    /// <summary>Low-level dtype conversion kernels used by the safetensors reader (exposed so they can be
    /// reused / benchmarked independently of file I/O).</summary>
    public static class SafetensorsKernels
    {
        /// <summary>
        /// BF16 -&gt; F32 by widening the bf16 bits into the top 16 bits of the f32 (the low 16 mantissa
        /// bits are zero). This is exact (bf16 is a truncated f32) and matches numpy's
        /// <c>(u16.astype(u32) &lt;&lt; 16).view(f32)</c>. Vectorised 16 lanes at a time when the hardware
        /// supports it; scalar tail otherwise.
        /// </summary>
        public static unsafe void Bf16ToF32(ushort* src, float* dst, long n)
        {
            long i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                for (; i + 16 <= n; i += 16)
                {
                    Vector256<ushort> v = Vector256.Load(src + i);
                    (Vector256<uint> lo, Vector256<uint> hi) = Vector256.Widen(v);
                    Vector256.Store(Vector256.ShiftLeft(lo, 16).AsSingle(), dst + i);
                    Vector256.Store(Vector256.ShiftLeft(hi, 16).AsSingle(), dst + i + 8);
                }
            }
            for (; i < n; i++)
                dst[i] = BitConverter.UInt32BitsToSingle((uint)src[i] << 16);
        }
    }
}
