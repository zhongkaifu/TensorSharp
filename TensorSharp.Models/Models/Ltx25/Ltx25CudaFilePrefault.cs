using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Bounded file-cache reads before CUDA uploads of mapped weights.
/// One handle stays open so warming never reopens the model between matrices.</summary>
internal sealed partial class Ltx25CudaFilePrefault : IDisposable
{
    private const int BufferBytes = 8 << 20;
    private readonly SafeFileHandle _handle;
    private delegate int ReadAtOffset(long offset, Span<byte> destination);
    private ReadAtOffset _read;
    private bool _disposed;
    private readonly byte[] _buffer;
    private readonly byte[] _residency;
    private readonly bool _checkResident;
    private readonly ParallelOptions _parallel;
    private long _readBytes;
    internal long ReadBytes => Interlocked.Read(ref _readBytes);
    internal int Workers { get; }
    internal long ResidentBytes { get; private set; }
    internal long Ranges { get; private set; }
    internal double Milliseconds { get; private set; }
    internal string Failure { get; private set; }

    internal Ltx25CudaFilePrefault(GgufFile file) : this((string)null) => _read = file.ReadMappedBytes;
    internal Ltx25CudaFilePrefault(SafetensorsFile file) : this((string)null) => _read = file.ReadMappedBytes;

    internal Ltx25CudaFilePrefault(string path, bool checkResident = true, int? workers = null)
    {
        int requested = workers ?? (OperatingSystem.IsLinux() ? 8 : 1);
        Workers = Math.Clamp(requested, 1, 8);
        _parallel = new ParallelOptions { MaxDegreeOfParallelism = Workers };
        _checkResident = checkResident && OperatingSystem.IsLinux();
        try
        {
            if (path != null)
            {
                _handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read, FileOptions.SequentialScan);
                _read = (offset, destination) => RandomAccess.Read(_handle, destination, offset);
            }
            _buffer = new byte[BufferBytes];
            _residency = new byte[BufferBytes / Environment.SystemPageSize + 2];
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException)
        { Failure = error.GetType().Name + ": " + error.Message; }
    }

    /// <summary>Read <paramref name="length"/> bytes of the mapped weight at <paramref name="mappedPointer"/> into the file
    /// cache before its upload (the pages already resident are skipped).</summary>
    internal void Prepare(IntPtr mappedPointer, long fileOffset, long length) => Warm(mappedPointer, fileOffset, length);

    internal void Warm(IntPtr mappedPointer, long fileOffset, long length)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (_read == null || Failure != null || length == 0) return;
        if (mappedPointer == IntPtr.Zero || fileOffset < 0 || length < 0)
            throw new ArgumentOutOfRangeException(nameof(length));
        long started = Stopwatch.GetTimestamp();
        try
        {
            for (long offset = 0; offset < length;)
            {
                int count = (int)Math.Min(BufferBytes, length - offset);
                var pointer = new IntPtr(checked(mappedPointer.ToInt64() + offset));
                if (_checkResident && ResidentState(pointer, count) == true)
                    ResidentBytes += count;
                else
                {
                    if (Workers == 1 || count <= (1 << 20))
                        ReadSlice(0, count, checked(fileOffset + offset));
                    else
                    {
                        // Every worker owns a disjoint 1 MiB slice of the same
                        // 8 MiB buffer. Join before that buffer is reused.
                        long position = checked(fileOffset + offset);
                        Parallel.For(0, (count + (1 << 20) - 1) >> 20, _parallel, slice =>
                        {
                            int begin = slice << 20, bytes = Math.Min(1 << 20, count - begin);
                            ReadSlice(begin, bytes, checked(position + begin));
                        });
                    }
                }
                offset += count;
            }
            ++Ranges;
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException ||
            error is AggregateException aggregate && aggregate.Flatten().InnerExceptions.All(inner => inner is IOException or UnauthorizedAccessException))
        {
            // Warming is optional. Actual tensor reads and uploads retain their
            // strict validation; no inference operation falls back to the CPU.
            Failure = error.GetType().Name + ": " + error.Message;
        }
        finally { Milliseconds += Stopwatch.GetElapsedTime(started).TotalMilliseconds; }
    }

    private void ReadSlice(int bufferOffset, int length, long fileOffset)
    {
        int done = 0;
        while (done < length)
        {
            int read = _read(checked(fileOffset + done), _buffer.AsSpan(bufferOffset + done, length - done));
            if (read == 0) throw new EndOfStreamException("LTX checkpoint changed during mapped-weight warmup.");
            done += read;
            // Preserve successful partial reads even when a different worker
            // reaches EOF or fails. Parallel.For joins all started readers.
            Interlocked.Add(ref _readBytes, read);
        }
    }

    private unsafe bool? ResidentState(IntPtr pointer, int length)
    {
        long address = pointer.ToInt64(), page = Environment.SystemPageSize;
        long aligned = address - address % page, bytes = address - aligned + length;
        int pages = checked((int)((bytes + page - 1) / page));
        try
        {
            fixed (byte* vector = _residency)
            {
                if (mincore((void*)aligned, (nuint)bytes, vector) != 0) return null;
                for (int i = 0; i < pages; ++i) if ((vector[i] & 1) == 0) return false;
            }
            return true;
        }
        catch (Exception error) when (error is DllNotFoundException or EntryPointNotFoundException) { return null; }
    }

    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static unsafe partial int mincore(void* address, nuint length, byte* vector);

    public void Dispose()
    {
        if (_disposed) return;
        _handle?.Dispose();
        _read = null;
        _disposed = true;
    }
}
