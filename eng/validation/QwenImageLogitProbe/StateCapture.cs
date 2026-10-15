using System.Collections;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;

// Bounded, explicitly charged diagnostic D2H scratch. Exported recurrent/QSA
// state is authoritative native data, never a hash of its stale managed seed.
sealed unsafe class StateCapture : IDisposable
{
    const int Capacity = 8 << 20;
    readonly Qwen4ExpModel _model;
    BudgetReservation? _root, _payload;
    IntPtr _buffer;
    public StateCapture(Qwen4ExpModel model, MemoryBudget budget)
    {
        _model = model;
        try
        {
            _root = budget.Reserve([new(UnifiedMemoryModelSession.HostPool, Capacity)]);
            _payload = _root.TryTake([new(UnifiedMemoryModelSession.HostPool, Capacity)])
                ?? throw new InvalidOperationException("No diagnostic state-buffer credit.");
            _buffer = Marshal.AllocHGlobal(Capacity);
            _payload.Commit();
        }
        catch { Dispose(); throw; }
    }
    static object? Field(object owner, string name) => owner.GetType().GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!
        .GetValue(owner);
    static string BytesHash(void* pointer, int bytes) => Convert.ToHexString(SHA256.HashData(new ReadOnlySpan<byte>(pointer, bytes))).ToLowerInvariant();
    static object Floats(void* pointer, long bytes)
    {
        if (bytes % 4 != 0 || bytes > int.MaxValue) throw new InvalidDataException("Invalid F32 state layout.");
        var values = new ReadOnlySpan<float>(pointer, (int)(bytes / 4));
        double squares = 0; int finite = 0;
        foreach (float v in values) { if (float.IsFinite(v)) finite++; squares += (double)v * v; }
        return new { Bytes = bytes, Sha256 = BytesHash(pointer, (int)bytes), Finite = finite == values.Length,
            L2 = finite == values.Length ? (double?)Math.Sqrt(squares) : null };
    }
    void Export(IntPtr key, long bytes)
    {
        if (bytes <= 0 || bytes > Capacity) throw new InvalidOperationException("State exceeds the explicit8MiB diagnostic buffer.");
        if (!GgmlBasicOps.Qwen4ExpCopyQsaCache(key, _buffer, bytes, 0))
            throw new InvalidOperationException("Authoritative native state export failed: " + GgmlBasicOps.LastNativeError());
    }
    static object LiveKv(Tensor tensor, int rows)
    {
        if (tensor.Sizes.Length != 3 || rows < 0 || rows > tensor.Sizes[1]) throw new InvalidDataException("KV shape mismatch.");
        IntPtr pointer = TensorComputePrimitives.GetStorageBasePointer(tensor);
        tensor.Storage.EnsureHostReadable();
        GgmlBasicOps.SyncHostBuffer(pointer, tensor.Storage.ByteLength);
        int heads = checked((int)tensor.Sizes[0]);
        int elementBytes = tensor.ElementType == DType.Float16 ? 2 : tensor.ElementType == DType.Float32 ? 4
            : throw new InvalidDataException("Unexpected KV element type.");
        long stride = checked(tensor.Sizes[1] * tensor.Sizes[2] * elementBytes);
        int live = checked((int)((long)rows * tensor.Sizes[2] * elementBytes));
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        for (int h = 0; h < heads; h++) hash.AppendData(new ReadOnlySpan<byte>((byte*)pointer + h * stride, live));
        return new { Rows = rows, Capacity = tensor.Sizes[1], Bytes = (long)heads * live,
            Sha256 = Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant() };
    }
    public object PreparedMedia(object injector, string id)
    {
        var buckets = (IDictionary)Field(injector, "_visionByRequest")!;
        var spans = (IEnumerable)buckets[id]!;
        var embeddings = new List<object>();
        foreach (object span in spans)
        {
            object entry = span.GetType().GetProperty("CacheEntry")!.GetValue(span)!;
            var tensor = (Tensor)entry.GetType().GetProperty("Embeddings")!.GetValue(entry)!;
            tensor.Storage.EnsureHostReadable();
            if (tensor.ElementType != DType.Float32) throw new InvalidDataException("Unexpected projected embedding dtype.");
            long bytes = checked(tensor.ElementCount() * sizeof(float));
            embeddings.Add(new { Shape = tensor.Sizes.ToArray(), Bytes = bytes,
                Sha256 = BytesHash(TensorComputePrimitives.GetStoragePointer(tensor).ToPointer(), checked((int)bytes)) });
        }
        var positions = (IDictionary)Field(injector, "_mropePositionsByRequest")!;
        return new { Embeddings = embeddings, MropePositions = (int[])positions[id]!,
            Scope = "Exact prepared projected embeddings and absolute T/H/W positions, not only image-file identity." };
    }
    public object Capture(string requestId, int rows)
    {
        if (!Equals(Field(_model, "_activeFusedKey"), requestId) || (int)Field(_model, "_cacheSeqLen")! != rows)
            throw new InvalidOperationException("Observer is not bound to the just-forwarded holder.");
        var conv = (Tensor[])Field(_model, "_gdnConvStateT")!;
        var ssm = (Tensor[])Field(_model, "_gdnStateT")!;
        var keys = (Tensor[])Field(_model, "_idxKCache")!;
        var k = (Tensor[])Field(_model, "_kCache")!;
        var v = (Tensor[])Field(_model, "_vCache")!;
        var layers = new List<object>();
        for (int l = 0; l < conv.Length; l++)
        {
            if (conv[l] != null)
            {
                long offset = (conv[l].Storage.ByteLength + 255) & ~255L;
                Export(TensorComputePrimitives.GetStoragePointer(conv[l]), offset + ssm[l].Storage.ByteLength);
                layers.Add(new { Layer = l, Kind = "GDN", Conv = Floats(_buffer.ToPointer(), conv[l].Storage.ByteLength),
                    Ssm = Floats((byte*)_buffer + offset, ssm[l].Storage.ByteLength) });
            }
            else
            {
                object? qsa = null;
                if (keys[l] != null)
                {
                    Export(TensorComputePrimitives.GetStoragePointer(keys[l]), keys[l].Storage.ByteLength);
                    int live = checked((int)((long)rows * keys[l].Sizes[2] * (keys[l].ElementType == DType.Float16 ? 2 : 4)));
                    qsa = new { Bytes = live, Sha256 = BytesHash(_buffer.ToPointer(), live) };
                }
                layers.Add(new { Layer = l, Kind = "attention", K = LiveKv(k[l], rows), V = LiveKv(v[l], rows), Qsa = qsa });
            }
        }
        var ple = (float[]?)Field(_model, "_pleConvState"); object? pleState = null;
        if (ple is { Length: > 0 })
        {
            Export(Marshal.UnsafeAddrOfPinnedArrayElement(ple, 0), checked((long)ple.Length * sizeof(float)));
            pleState = Floats(_buffer.ToPointer(), checked((long)ple.Length * sizeof(float)));
        }
        int count = (int)Field(_model, "_qsaPositionCount")!;
        var positions = (int[]?)Field(_model, "_qsaPositions");
        return new { Rows = rows, Layers = layers, Ple = pleState, PleHistory = ((List<int>?)Field(_model, "_pleHistory"))?.ToArray(),
            PleNextPosition = Field(_model, "_pleNextPos"), MropeGap = Field(_model, "_mropeCacheGap"),
            QsaPositionCount = count, QsaPositions = positions?.Take(3 * count).ToArray(),
            DeviceAuthoritative = Field(_model, "_deviceStateAuthoritative") };
    }
    public void Dispose()
    {
        if (_buffer != IntPtr.Zero) { Marshal.FreeHGlobal(_buffer); _buffer = IntPtr.Zero; }
        _payload?.Dispose(); _payload = null; _root?.Dispose(); _root = null;
    }
}
