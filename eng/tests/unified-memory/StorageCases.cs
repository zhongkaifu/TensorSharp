// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using static Fixture;

static partial class Cases
{
    public static async Task GgufCatalog()
    {
        await using var f = new Fixture();
        string first = Path.Combine(f.Root, "model-00001-of-00002.gguf");
        string second = Path.Combine(f.Root, "model-00002-of-00002.gguf");
        byte[] rawA = Enumerable.Range(0, 32).Select(i => (byte)i).ToArray();
        byte[] rawB = Enumerable.Range(0, 68).Select(i => (byte)(i + 70)).ToArray();
        WriteShard(first, 0, "a", GgmlTensorType.F32, new ulong[] { 4, 2 }, rawA);
        WriteShard(second, 1, "b", GgmlTensorType.Q8_0, new ulong[] { 32, 2 }, rawB);
        using var gguf = new GgufFile(first);
        using var catalog = gguf.CreateMemoryCatalog("synthetic-revision-v1", 9);
        Check.Equal(Path.GetFullPath(second), gguf.GetTensorFileRegion("b").Path);
        Check.Equal(68L, gguf.GetTensorFileRegion("b").ByteLength);
        Check.Equal(2, catalog.Resources.Count());
        foreach (string name in new[] { "a", "b" })
        {
            var (resource, source) = catalog.Get(name);
            f.Scheduler.Register(resource, source);
            var actual = await f.Read(resource.Key);
            Check.Bytes(name == "a" ? rawA : rawB, actual);
            f.Scheduler.Unregister(resource.Key);
        }
        var slice = catalog.GetSlice("b", "expert/b/row1", 34, 34, ResourceKind.Expert);
        f.Scheduler.Register(slice.Resource, slice.Source);
        var actualSlice = await f.Read(slice.Resource.Key);
        Check.Bytes(rawB.AsSpan(34), actualSlice);
        f.Scheduler.Unregister(slice.Resource.Key);
        Check.Throws<ArgumentOutOfRangeException>(() => catalog.GetSlice("b", "bad", 67, 2));
    }

    private static void WriteShard(string path, int shard, string tensor, GgmlTensorType type, ulong[] shape, byte[] payload)
    {
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        void Str(string value) { var bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
        writer.Write(0x46554747u); writer.Write(3u); writer.Write(1UL); writer.Write(2UL);
        Str("split.count"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)2);
        Str("split.no"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)shard);
        Str(tensor); writer.Write((uint)shape.Length); foreach (var d in shape) writer.Write(d);
        writer.Write((uint)type); writer.Write(0UL);
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        writer.Write(payload);
    }

    public static async Task StreamingMatvec()
    {
        const int rows = 512, cols = 256, tileRows = 8, batch = 8;
        const int tileBytes = tileRows * cols * sizeof(float);
        await using var f = new Fixture(ram: tileBytes * 2, chunk: Page);
        string path = Path.Combine(f.Root, "matrix.f32");
        using (var writer = new BinaryWriter(File.Create(path)))
            for (int row = 0; row < rows; row++)
                for (int col = 0; col < cols; col++) writer.Write(Weight(row, col));
        using var file = new FileDataSource(path);
        var input = Enumerable.Range(0, batch).Select(b => Enumerable.Range(0, cols).Select(c => (c % 19 + b) / 16f).ToArray()).ToArray();
        var output = Enumerable.Range(0, batch).Select(_ => new float[rows]).ToArray();
        var keys = new List<ResourceKey>();
        for (int row = 0; row < rows; row += tileRows)
        {
            var key = f.Add($"matrix/{row}", bytes: tileBytes, source: new FileRegionSource(file, (long)row * cols * sizeof(float), tileBytes));
            keys.Add(key);
            using var lease = await f.Scheduler.AcquireAsync(key, f.Host.Location);
            ComputeTile(lease.Pointer, row, tileRows, cols, input, output);
        }
        // Compute the mathematical reference without allocating the whole matrix.
        // The accumulation order and precision are identical to the streamed path.
        for (int b = 0; b < batch; b++)
            for (int row = 0; row < rows; row++)
            {
                float expected = 0;
                for (int col = 0; col < cols; col++) expected += Weight(row, col) * input[b][col];
                Check.Equal(BitConverter.SingleToInt32Bits(expected), BitConverter.SingleToInt32Bits(output[b][row]));
            }
        Check.Equal((long)rows * cols * sizeof(float), f.Transfers.BytesCopied); // one load shared across all eight requests.
        Check.True(file.ByteLength > f.Budget.Snapshot().Single(x => x.Pool == "ram").Capacity * 20);
        Console.WriteLine($"  F32 matrix: {file.ByteLength} B, managed payload+staging budget: {tileBytes * 2 + Page} B, batch: {batch}, exact outputs: {rows * batch}.");
        foreach (var key in keys) f.Scheduler.Unregister(key);
    }
    private static float Weight(int row, int col) => ((row * 31 + col * 17) % 101 - 50) / 64f;
    private static unsafe void ComputeTile(nint pointer, int firstRow, int tileRows, int cols, float[][] input, float[][] output)
    {
        var weights = (float*)pointer;
        for (int b = 0; b < input.Length; b++)
            for (int row = 0; row < tileRows; row++)
            {
                float sum = 0;
                for (int col = 0; col < cols; col++) sum += weights[row * cols + col] * input[b][col];
                output[b][firstRow + row] = sum;
            }
    }
}
