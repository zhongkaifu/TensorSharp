// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;

internal static partial class Program
{
    private const int Guard = 256;
    private static readonly byte[] Poisons = { 0xc1, 0xd2, 0xc1 };
    private static void Need(bool value, string message) { if (!value) throw new InvalidDataException(message); }
    private static byte[] Bytes<T>(T[] data) where T : unmanaged => MemoryMarshal.AsBytes(data.AsSpan()).ToArray();
    private static string Sha(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileSha(string file) { using var s = File.OpenRead(file); return Convert.ToHexString(SHA256.HashData(s)).ToLowerInvariant(); }
    private static byte[] Guarded(byte[] body)
    {
        var result = new byte[checked(body.Length + 2 * Guard)]; Array.Fill(result, (byte)0xa5);
        body.CopyTo(result, Guard); return result;
    }
    private static bool Guards(byte[] data) => data.AsSpan(0, Guard).IndexOfAnyExcept((byte)0xa5) < 0 && data.AsSpan(data.Length - Guard).IndexOfAnyExcept((byte)0xa5) < 0;
    private static void NewFile(string file, byte[] bytes)
    { using var s = new FileStream(file, FileMode.CreateNew, FileAccess.Write, FileShare.None); s.Write(bytes); s.Flush(true); }

    private sealed unsafe class Memory : IDisposable
    {
        internal sealed record Buffer(IntPtr Allocation, IntPtr Pointer, int BodyBytes, string Name, string? ImmutableSha);
        private readonly List<Buffer> _buffers = new();
        internal IReadOnlyList<Buffer> Buffers => _buffers;
        internal Buffer Put(string name, byte[] body, bool immutable = true)
        {
            byte[] all = Guarded(body); CudaDriverApi.cuMemAlloc(out var allocation, (UIntPtr)all.Length).ThrowOnError();
            var result = new Buffer(allocation, allocation + Guard, body.Length, name, immutable ? Sha(all) : null);
            _buffers.Add(result); Write(allocation, all); return result;
        }
        internal Buffer Empty(string name, long bytes)
        { Need(bytes > 0 && bytes <= int.MaxValue - 2 * Guard, "Probe allocation exceeds its bounded span."); return Put(name, new byte[(int)bytes], false); }
        internal static void Write(IntPtr pointer, byte[] data)
        { fixed (byte* src = data) CudaDriverApi.cuMemcpyHtoD(pointer, (IntPtr)src, (UIntPtr)data.Length).ThrowOnError(); }
        internal static byte[] Read(IntPtr pointer, int bytes)
        { byte[] data = new byte[bytes]; fixed (byte* dst = data) CudaDriverApi.cuMemcpyDtoH((IntPtr)dst, pointer, (UIntPtr)bytes).ThrowOnError(); return data; }
        internal static byte[] All(Buffer buffer) => Read(buffer.Allocation, buffer.BodyBytes + 2 * Guard);
        internal static byte[] Body(Buffer buffer) => Read(buffer.Pointer, buffer.BodyBytes);
        internal static void Poison(Buffer buffer, byte poison) { var body = new byte[buffer.BodyBytes]; Array.Fill(body, poison); Write(buffer.Allocation, Guarded(body)); }
        internal bool CheckInputs() => _buffers.Where(x => x.ImmutableSha != null).All(x => Sha(All(x)) == x.ImmutableSha);
        internal bool CheckGuards() => _buffers.All(x => Guards(All(x)));
        public void Dispose() { foreach (var b in _buffers.AsEnumerable().Reverse()) CudaDriverApi.cuMemFree(b.Allocation).ThrowOnError(); }
    }

    // Explicit pinned image, not the old runtime's implicit environment search.
    // Same original twelve-argument kernel ABI and ordinary-prefill launch.
    private sealed unsafe class LegacyAttention : IDisposable
    {
        private readonly CudaModule _module;
        private readonly IntPtr _kernel;
        internal string ImageSha256 => _module.ImageSha256;
        internal LegacyAttention(string path)
        {
            _module = CudaModule.LoadFromFile(path, identify: true);
            try
            {
                _kernel = _module.GetFunction("ts_ninfer_attn_bf16_f32"); CudaDriverApi.cuFuncLoad(_kernel).ThrowOnError();
                ulong[] offsets = { 0, 8, 16, 24, 32, 40, 48, 56, 60, 64, 68, 72 }, sizes = { 8, 8, 8, 8, 4, 8, 8, 4, 4, 4, 4, 8 };
                for (uint i = 0; i < offsets.Length; i++)
                {
                    CudaDriverApi.cuFuncGetParamInfo(_kernel, (UIntPtr)i, out var offset, out var size).ThrowOnError();
                    Need(offset.ToUInt64() == offsets[i] && size.ToUInt64() == sizes[i], "Frozen legacy attention device ABI differs.");
                }
                Need(CudaDriverApi.cuFuncGetParamInfo(_kernel, (UIntPtr)12, out _, out _) == 1, "Unexpected legacy attention parameter.");
            }
            catch { _module.Dispose(); throw; }
        }
        internal void Run(IntPtr q, IntPtr gate, IntPtr ends, IntPtr blocks, int maxBlocks, long kOff, long vOff, int rows, IntPtr output)
        {
            int heads = 24, kvHeads = 4; float scale = .0625f;
            void** args = stackalloc void*[] { &q, &gate, &ends, &blocks, &maxBlocks, &kOff, &vOff, &rows, &heads, &kvHeads, &scale, &output };
            CudaDriverApi.cuLaunchKernel(_kernel, checked((uint)((rows * 6 + 127) / 128)), 4, 1, 256, 1, 1, 0, IntPtr.Zero, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }
        public void Dispose() => _module.Dispose();
    }
}
