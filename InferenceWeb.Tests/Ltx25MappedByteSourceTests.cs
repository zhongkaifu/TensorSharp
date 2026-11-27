using System.Text;
using System.Text.Json;
using TensorSharp.Models.Ltx25;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class Ltx25MappedByteSourceTests
{
    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(true, true)]
    public void PositionalReadsUseTheMappedInodeAndRejectDisposedOwner(bool safetensors, bool replaceBeforeMapping)
    {
        string path = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + (safetensors ? ".safetensors" : ".gguf"));
        string previous = path + ".previous";
        byte[] values = Enumerable.Range(0, 4096).Select(i => (byte)(i * 43 + i / 127)).ToArray();
        IDisposable reader = null;
        try
        {
            long dataOffset = Write(path, values, safetensors);
            ReadAt read;
            Action checkDisposedMapping;
            if (safetensors)
            {
                var file = new SafetensorsFile(path); reader = file;
                if (!replaceBeforeMapping) Assert.True(file.TryGetTensorDataPointer(file.GetInfo("weight"), out _));
                read = file.ReadMappedBytes;
                checkDisposedMapping = () =>
                {
                    Assert.Throws<ObjectDisposedException>(() => file.TryGetTensorDataPointer(file.GetInfo("weight"), out _));
                    Assert.Throws<ObjectDisposedException>(() => file.ReadFloat32("weight"));
                };
            }
            else
            {
                var file = new GgufFile(path); reader = file;
                if (!replaceBeforeMapping) Assert.True(file.TryGetTensorDataPointer(file.Tensors["weight"], out _));
                read = file.ReadMappedBytes;
                checkDisposedMapping = () => Assert.Throws<ObjectDisposedException>(() => file.ReadMappedBytes(dataOffset, new byte[1]));
            }
            {
                void ReplacePath()
                {
                    if (OperatingSystem.IsWindows())
                        Assert.Throws<IOException>(() => File.Move(path, previous));
                    else
                    {
                        File.Move(path, previous);
                        Write(path, Enumerable.Repeat((byte)211, values.Length).ToArray(), safetensors);
                    }
                }
                if (replaceBeforeMapping) ReplacePath();
                byte[] firstByte = new byte[1];
                Assert.Equal(1, read(dataOffset, firstByte));
                Assert.Equal(values[0], firstByte[0]);
                Parallel.For(0, 16, i =>
                {
                    byte[] slice = new byte[137];
                    Assert.Equal(slice.Length, read(dataOffset + i, slice));
                    Assert.Equal(values.AsSpan(i, slice.Length).ToArray(), slice);
                });
                byte[] tail = new byte[128];
                Assert.Equal(7, read(dataOffset + values.Length - 7, tail));
                Assert.Equal(values.AsSpan(values.Length - 7).ToArray(), tail.AsSpan(0, 7).ToArray());
                Assert.Equal(0, read(dataOffset + values.Length, tail));
                Assert.Throws<ArgumentOutOfRangeException>(() => read(-1, tail));
                Assert.Throws<ArgumentOutOfRangeException>(() => read(long.MaxValue - 1, tail));

                // A path replacement after mapping must not change the bytes read.
                // Windows read-share locking rejects it while the reader lives.
                if (!replaceBeforeMapping) ReplacePath();
                byte[] actual = new byte[values.Length];
                Assert.Equal(values.Length, read(dataOffset, actual));
                Assert.Equal(values, actual);
                reader.Dispose();
                checkDisposedMapping();
                Assert.Throws<ObjectDisposedException>(() => read(dataOffset, tail));
            }
        }
        finally
        {
            reader?.Dispose();
            File.Delete(path); File.Delete(previous);
        }
    }

    private delegate int ReadAt(long offset, Span<byte> destination);

    [Fact]
    public void InvalidSafetensorsHeaderClosesItsOwnedStream()
    {
        string path = Path.GetTempFileName();
        try
        {
            File.WriteAllBytes(path, new byte[8]);
            Assert.Throws<InvalidDataException>(() => new SafetensorsFile(path));
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            Assert.Equal(8, exclusive.Length);
        }
        finally { File.Delete(path); }
    }

    private static long Write(string path, byte[] values, bool safetensors)
    {
        using var file = File.Create(path);
        using var writer = new BinaryWriter(file, Encoding.UTF8, leaveOpen: true);
        if (safetensors)
        {
            byte[] header = JsonSerializer.SerializeToUtf8Bytes(new
            {
                weight = new { dtype = "F32", shape = new[] { values.Length / sizeof(float) }, data_offsets = new[] { 0, values.Length } },
            });
            writer.Write((ulong)header.Length); writer.Write(header);
        }
        else
        {
            writer.Write(0x46554747u); writer.Write(3u); writer.Write(1ul); writer.Write(0ul);
            writer.Write(6ul); writer.Write(Encoding.UTF8.GetBytes("weight"));
            writer.Write(1u); writer.Write((ulong)(values.Length / sizeof(float))); writer.Write(0u); writer.Write(0ul);
            while (file.Position % 32 != 0) writer.Write((byte)0);
        }
        long offset = file.Position; writer.Write(values); return offset;
    }
}
