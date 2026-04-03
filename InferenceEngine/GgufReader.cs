using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace InferenceEngine
{
    public enum GgufValueType : uint
    {
        Uint8 = 0, Int8 = 1, Uint16 = 2, Int16 = 3,
        Uint32 = 4, Int32 = 5, Float32 = 6, Bool = 7,
        String = 8, Array = 9, Uint64 = 10, Int64 = 11, Float64 = 12
    }

    public enum GgmlTensorType : uint
    {
        F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
        Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
        Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
        I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28,
        BF16 = 30,
    }

    public class GgufTensorInfo
    {
        public string Name { get; set; }
        public ulong[] Shape { get; set; }
        public GgmlTensorType Type { get; set; }
        public ulong Offset { get; set; }

        public long NumElements
        {
            get
            {
                long n = 1;
                foreach (var d in Shape) n *= (long)d;
                return n;
            }
        }
    }

    public class GgufFile : IDisposable
    {
        public uint Version { get; private set; }
        public Dictionary<string, object> Metadata { get; } = new();
        public Dictionary<string, GgufTensorInfo> Tensors { get; } = new();
        public long DataOffset { get; private set; }

        private FileStream _stream;
        private string _path;

        public GgufFile(string path)
        {
            _path = path;
            _stream = File.OpenRead(path);
            Parse();
        }

        private void Parse()
        {
            using var reader = new BinaryReader(_stream, Encoding.UTF8, leaveOpen: true);

            uint magic = reader.ReadUInt32();
            if (magic != 0x46554747) // "GGUF" in little-endian
                throw new InvalidDataException($"Not a GGUF file (magic: 0x{magic:X8})");

            Version = reader.ReadUInt32();
            if (Version < 2)
                throw new NotSupportedException($"GGUF version {Version} not supported");

            ulong tensorCount = reader.ReadUInt64();
            ulong kvCount = reader.ReadUInt64();

            for (ulong i = 0; i < kvCount; i++)
            {
                string key = ReadString(reader);
                var valType = (GgufValueType)reader.ReadUInt32();
                object value = ReadValue(reader, valType);
                Metadata[key] = value;
            }

            for (ulong i = 0; i < tensorCount; i++)
            {
                var info = new GgufTensorInfo();
                info.Name = ReadString(reader);
                uint dims = reader.ReadUInt32();
                info.Shape = new ulong[dims];
                for (uint d = 0; d < dims; d++)
                    info.Shape[d] = reader.ReadUInt64();
                info.Type = (GgmlTensorType)reader.ReadUInt32();
                info.Offset = reader.ReadUInt64();
                Tensors[info.Name] = info;
            }

            long pos = _stream.Position;
            int alignment = 32;
            if (Metadata.TryGetValue("general.alignment", out var a))
                alignment = Convert.ToInt32(a);
            DataOffset = pos + (alignment - pos % alignment) % alignment;
        }

        public string GetString(string key, string defaultValue = null)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return v as string ?? defaultValue;
        }

        public uint GetUint32(string key, uint defaultValue = 0)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is int[] ia && ia.Length > 0) return (uint)ia[0];
            if (v is uint[] ua && ua.Length > 0) return ua[0];
            return Convert.ToUInt32(v);
        }

        public float GetFloat32(string key, float defaultValue = 0f)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            if (v is float[] fa && fa.Length > 0) return fa[0];
            return Convert.ToSingle(v);
        }

        public bool GetBool(string key, bool defaultValue = false)
        {
            if (!Metadata.TryGetValue(key, out var v)) return defaultValue;
            return Convert.ToBoolean(v);
        }

        public string[] GetStringArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is string[] sa) return sa;
            return null;
        }

        public float[] GetFloatArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is float[] fa) return fa;
            return null;
        }

        public int[] GetInt32Array(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is int[] ia) return ia;
            if (v is uint[] ua)
            {
                var result = new int[ua.Length];
                for (int i = 0; i < ua.Length; i++) result[i] = (int)ua[i];
                return result;
            }
            return null;
        }

        public bool[] GetBoolArray(string key)
        {
            if (!Metadata.TryGetValue(key, out var v)) return null;
            if (v is bool[] ba) return ba;
            return null;
        }

        public byte[] ReadTensorData(GgufTensorInfo tensorInfo)
        {
            long byteCount = GetTensorByteCount(tensorInfo);
            byte[] data = new byte[byteCount];
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            _stream.ReadExactly(data, 0, data.Length);
            return data;
        }

        /// <summary>
        /// Read F32 tensor data directly into a float array in chunks (for tensors > 2GB raw bytes).
        /// </summary>
        public unsafe void ReadTensorDataToFloat32(GgufTensorInfo tensorInfo, float[] dest, long numElements)
        {
            long totalBytes = numElements * 4;
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            const int chunkBytes = 16 * 1024 * 1024;
            byte[] buffer = new byte[chunkBytes];
            long bytesRead = 0;

            fixed (float* destBase = dest)
            {
                while (bytesRead < totalBytes)
                {
                    int toRead = (int)Math.Min(totalBytes - bytesRead, chunkBytes);
                    _stream.ReadExactly(buffer, 0, toRead);
                    fixed (byte* srcPtr = buffer)
                    {
                        Buffer.MemoryCopy(srcPtr, (byte*)destBase + bytesRead,
                            totalBytes - bytesRead, toRead);
                    }
                    bytesRead += toRead;
                }
            }
        }

        /// <summary>
        /// Read F32 tensor data directly into native memory pointed to by dest (for tensors > 2G elements).
        /// </summary>
        public unsafe void ReadTensorDataToFloat32Native(GgufTensorInfo tensorInfo, IntPtr dest, long numElements)
        {
            long totalBytes = numElements * 4;
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            const int chunkBytes = 16 * 1024 * 1024;
            byte[] buffer = new byte[chunkBytes];
            long bytesRead = 0;
            byte* destPtr = (byte*)dest;

            while (bytesRead < totalBytes)
            {
                int toRead = (int)Math.Min(totalBytes - bytesRead, chunkBytes);
                _stream.ReadExactly(buffer, 0, toRead);
                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, (IntPtr)(destPtr + bytesRead), toRead);
                bytesRead += toRead;
            }
        }

        /// <summary>
        /// Read tensor data directly into pre-allocated native memory (for tensors > 2GB).
        /// </summary>
        public unsafe void ReadTensorDataToNative(GgufTensorInfo tensorInfo, IntPtr dest, long byteCount)
        {
            _stream.Seek(DataOffset + (long)tensorInfo.Offset, SeekOrigin.Begin);
            byte[] buffer = new byte[Math.Min(byteCount, 8 * 1024 * 1024)];
            long remaining = byteCount;
            byte* destPtr = (byte*)dest.ToPointer();
            while (remaining > 0)
            {
                int toRead = (int)Math.Min(remaining, buffer.Length);
                _stream.ReadExactly(buffer, 0, toRead);
                System.Runtime.InteropServices.Marshal.Copy(buffer, 0, (IntPtr)destPtr, toRead);
                destPtr += toRead;
                remaining -= toRead;
            }
        }

        public long GetTensorByteCount(GgufTensorInfo tensorInfo)
        {
            long ne0 = (long)tensorInfo.Shape[0];
            long rows = 1;
            for (int i = 1; i < tensorInfo.Shape.Length; i++)
                rows *= (long)tensorInfo.Shape[i];

            long rowBytes = GetRowBytes(tensorInfo.Type, ne0);
            return rowBytes * rows;
        }

        private static long GetRowBytes(GgmlTensorType type, long ne0)
        {
            long blockSize = GetBlockSize(type);
            long typeSize = GetTypeSize(type);
            return (ne0 / blockSize) * typeSize;
        }

        public static long GetBlockSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32:
                case GgmlTensorType.F16:
                case GgmlTensorType.BF16:
                case GgmlTensorType.I8:
                case GgmlTensorType.I16:
                case GgmlTensorType.I32:
                case GgmlTensorType.I64:
                case GgmlTensorType.F64:
                    return 1;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q4_1:
                case GgmlTensorType.Q5_0:
                case GgmlTensorType.Q5_1:
                case GgmlTensorType.Q8_0:
                case GgmlTensorType.Q8_1:
                    return 32;
                default:
                    return 256;
            }
        }

        public static long GetTypeSize(GgmlTensorType type)
        {
            switch (type)
            {
                case GgmlTensorType.F32: return 4;
                case GgmlTensorType.F16: return 2;
                case GgmlTensorType.BF16: return 2;
                case GgmlTensorType.Q4_0: return 2 + 32 / 2;
                case GgmlTensorType.Q4_1: return 2 + 2 + 32 / 2;
                case GgmlTensorType.Q5_0: return 2 + 4 + 32 / 2;
                case GgmlTensorType.Q5_1: return 2 + 2 + 4 + 32 / 2;
                case GgmlTensorType.Q8_0: return 2 + 32;
                case GgmlTensorType.Q8_1: return 2 + 2 + 32;
                case GgmlTensorType.Q2_K: return 256 / 16 + 256 / 4 + 2 + 2;
                case GgmlTensorType.Q3_K: return 256 / 8 + 256 / 4 + 12 + 2;
                case GgmlTensorType.Q4_K: return 2 + 2 + 12 + 256 / 2;
                case GgmlTensorType.Q5_K: return 2 + 2 + 12 + 256 / 8 + 256 / 2;
                case GgmlTensorType.Q6_K: return 256 / 2 + 256 / 4 + 256 / 16 + 2;
                case GgmlTensorType.Q8_K: return 4 + 256 + 2 * 256 / 16;
                case GgmlTensorType.I8: return 1;
                case GgmlTensorType.I16: return 2;
                case GgmlTensorType.I32: return 4;
                case GgmlTensorType.I64: return 8;
                case GgmlTensorType.F64: return 8;
                default:
                    throw new NotSupportedException($"Unknown GGML tensor type: {type}");
            }
        }

        private string ReadString(BinaryReader reader)
        {
            ulong len = reader.ReadUInt64();
            byte[] bytes = reader.ReadBytes((int)len);
            return Encoding.UTF8.GetString(bytes);
        }

        private object ReadValue(BinaryReader reader, GgufValueType type)
        {
            switch (type)
            {
                case GgufValueType.Uint8: return reader.ReadByte();
                case GgufValueType.Int8: return reader.ReadSByte();
                case GgufValueType.Uint16: return reader.ReadUInt16();
                case GgufValueType.Int16: return reader.ReadInt16();
                case GgufValueType.Uint32: return reader.ReadUInt32();
                case GgufValueType.Int32: return reader.ReadInt32();
                case GgufValueType.Float32: return reader.ReadSingle();
                case GgufValueType.Bool: return reader.ReadByte() != 0;
                case GgufValueType.String: return ReadString(reader);
                case GgufValueType.Uint64: return reader.ReadUInt64();
                case GgufValueType.Int64: return reader.ReadInt64();
                case GgufValueType.Float64: return reader.ReadDouble();
                case GgufValueType.Array: return ReadArray(reader);
                default:
                    throw new NotSupportedException($"Unknown GGUF value type: {type}");
            }
        }

        private object ReadArray(BinaryReader reader)
        {
            var elemType = (GgufValueType)reader.ReadUInt32();
            ulong count = reader.ReadUInt64();

            switch (elemType)
            {
                case GgufValueType.Uint32:
                {
                    var arr = new uint[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt32();
                    return arr;
                }
                case GgufValueType.Int32:
                {
                    var arr = new int[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt32();
                    return arr;
                }
                case GgufValueType.Float32:
                {
                    var arr = new float[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSingle();
                    return arr;
                }
                case GgufValueType.String:
                {
                    var arr = new string[count];
                    for (ulong i = 0; i < count; i++) arr[i] = ReadString(reader);
                    return arr;
                }
                case GgufValueType.Uint8:
                {
                    var arr = new byte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte();
                    return arr;
                }
                case GgufValueType.Int8:
                {
                    var arr = new sbyte[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadSByte();
                    return arr;
                }
                case GgufValueType.Uint16:
                {
                    var arr = new ushort[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt16();
                    return arr;
                }
                case GgufValueType.Int16:
                {
                    var arr = new short[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt16();
                    return arr;
                }
                case GgufValueType.Uint64:
                {
                    var arr = new ulong[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadUInt64();
                    return arr;
                }
                case GgufValueType.Int64:
                {
                    var arr = new long[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadInt64();
                    return arr;
                }
                case GgufValueType.Float64:
                {
                    var arr = new double[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadDouble();
                    return arr;
                }
                case GgufValueType.Bool:
                {
                    var arr = new bool[count];
                    for (ulong i = 0; i < count; i++) arr[i] = reader.ReadByte() != 0;
                    return arr;
                }
                default:
                    throw new NotSupportedException($"Unknown array element type: {elemType}");
            }
        }

        public void Dispose()
        {
            _stream?.Dispose();
        }
    }
}
