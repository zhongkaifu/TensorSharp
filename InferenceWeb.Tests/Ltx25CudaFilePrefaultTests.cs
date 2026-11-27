using System.IO.MemoryMappedFiles;
using TensorSharp.Models.Ltx25;

namespace InferenceWeb.Tests;

public sealed class Ltx25CudaFilePrefaultTests
{
    [Theory]
    [InlineData(1)] [InlineData(8)]
    public unsafe void SelectedUnalignedRangeCrossesBufferBoundaryWithoutReadingTheWholeFile(int workers)
    {
        string path = Path.GetTempFileName();
        try
        {
            using (var stream = File.OpenWrite(path)) stream.SetLength(12 << 20);
            using var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            try
            {
                using var prefetch = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
                const int offset = 31, length = (8 << 20) + 123;
                prefetch.Warm((IntPtr)(pointer + offset), offset, length);
                Assert.Null(prefetch.Failure);
                Assert.Equal(length, prefetch.ReadBytes);
                Assert.Equal(0, prefetch.ResidentBytes);
                Assert.Equal(1, prefetch.Ranges);
                Assert.Equal(workers, prefetch.Workers);
                Assert.True(prefetch.Milliseconds >= 0);
            }
            finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally { File.Delete(path); }
    }

    [Theory]
    [InlineData(1)] [InlineData(8)]
    public unsafe void TruncatedRangeRecordsPartialReadAndDisablesOptionalWarming(int workers)
    {
        string path = Path.GetTempFileName();
        try
        {
            int length = (workers == 8 ? 2 << 20 : 8 << 20) + 4;
            using (var stream = File.OpenWrite(path)) stream.SetLength(length);
            using var file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
            using var view = file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
            byte* pointer = null; view.SafeMemoryMappedViewHandle.AcquirePointer(ref pointer);
            try
            {
                using var prefetch = new Ltx25CudaFilePrefault(path, checkResident: false, workers: workers);
                prefetch.Warm((IntPtr)pointer, 0, length + 12);
                Assert.StartsWith(workers == 8 ? nameof(AggregateException) : nameof(EndOfStreamException), prefetch.Failure);
                if (workers == 8) Assert.InRange(prefetch.ReadBytes, 4, length);
                else Assert.Equal(length, prefetch.ReadBytes);
                Assert.Equal(0, prefetch.Ranges);
                long read = prefetch.ReadBytes;
                prefetch.Warm((IntPtr)pointer, 0, 16);
                Assert.Equal(read, prefetch.ReadBytes);
            }
            finally { view.SafeMemoryMappedViewHandle.ReleasePointer(); }
        }
        finally { File.Delete(path); }
    }

}
