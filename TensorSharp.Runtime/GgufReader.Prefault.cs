// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;
using TensorSharp.Memory;

namespace TensorSharp.Runtime;

public partial class GgufFile
{
    private bool _prefaulted;
    internal long PrefaultReadBytes { get; private set; }
    internal long PrefaultResidentBytes { get; private set; }

    /// <summary>Warm tensor data with bounded parallel reads. On Linux, resident
    /// pages are checked without copying them; repeated loads avoid rereading the
    /// whole checkpoint. Sparse host lookup tables may be excluded by the model.</summary>
    public void PrefaultFileCache() => PrefaultFileCache(null);

    public void PrefaultFileCache(Func<GgufTensorInfo, bool>? includeTensor)
    {
        // Phones only: Mac Catalyst reports itself as iOS but is a desktop process
        // with a desktop page cache, and warming it is what a desktop load wants.
        if (_prefaulted || Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT") == "0" ||
            (OperatingSystem.IsIOS() && !OperatingSystem.IsMacCatalyst()) || OperatingSystem.IsTvOS())
            return;
        _prefaulted = true;

        // Apply the budget to the entire split checkpoint, not independently to
        // each shard: warming seven individually small shards can evict the data
        // that the loader is about to use.
        var plans = new List<(GgufFile File, List<(long Offset, long Length)> Ranges)>();
        foreach (GgufFile file in new[] { this }.Concat(_shards))
        {
            var ranges = GetPrefaultRanges(file, includeTensor);
            if (ranges.Count > 0) plans.Add((file, ranges));
        }
        long total = plans.Sum(p => p.Ranges.Sum(r => r.Length));
        int threads = Math.Min(16, Environment.ProcessorCount);
        if (int.TryParse(Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT_THREADS"), out int requested) && requested > 0)
            threads = Math.Min(requested, Math.Max(1, Environment.ProcessorCount));
        long available;
        try { available = HostMemoryInfo.Capture().Available; }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        { return; } // Optional warming cannot treat unknown physical availability as unlimited.
        if (!CanPrefaultFileCache(total, available, threads)) return;
        long readBytes = 0, residentBytes = 0;
        var watch = Stopwatch.StartNew();
        try
        {
            foreach (var plan in plans)
            {
                bool checkResident = OperatingSystem.IsLinux() &&
                    Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT_RESIDENT") != "0";
                if (checkResident)
                {
                    try { plan.File.EnsureMappedView(); }
                    catch { checkResident = false; }
                }
                long length = plan.Ranges.Sum(r => r.Length);
                int workers = (int)Math.Min(threads, Math.Max(1, (length + (8 << 20) - 1) / (8 << 20)));
                long regionBytes = (length + workers - 1) / workers;
                var handles = new SafeFileHandle[workers];
                try
                {
                    // Some FUSE mounts invalidate cached pages on open. Acquire
                    // every stream before warming so a late worker cannot evict
                    // the pages another worker has just read.
                    for (int worker = 0; worker < workers; worker++)
                        handles[worker] = File.OpenHandle(plan.File._path, FileMode.Open, FileAccess.Read, FileShare.Read);
                    Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, region =>
                    {
                        long start = region * regionBytes, end = Math.Min(start + regionBytes, length);
                        long position = 0, localRead = 0, localResident = 0;
                        PrefaultReadBuffer? buffer = null;
                        // One handle per worker preserves network filesystem readahead.
                        SafeFileHandle handle = handles[region];
                        byte[] residency = checkResident ? new byte[((8 << 20) / Environment.SystemPageSize) + 2] : Array.Empty<byte>();
                        try
                        {
                            foreach (var range in plan.Ranges)
                            {
                                long begin = Math.Max(start - position, 0);
                                long limit = Math.Min(end - position, range.Length);
                                for (long offset = begin; offset < limit;)
                                {
                                    int count = (int)Math.Min(8 << 20, limit - offset);
                                    long fileOffset = range.Offset + offset;
                                    if (checkResident && plan.File.IsRangeResident(fileOffset, count, residency))
                                        localResident += count;
                                    else
                                    {
                                        buffer ??= new PrefaultReadBuffer((int)Math.Min(8 << 20, end - start));
                                        int done = 0;
                                        while (done < count)
                                        {
                                            int read = RandomAccess.Read(handle, buffer.Span.Slice(done, count - done), fileOffset + done);
                                            if (read == 0) throw new EndOfStreamException("GGUF changed during page-cache warmup.");
                                            done += read;
                                        }
                                        localRead += count;
                                    }
                                    offset += count;
                                }
                                position += range.Length;
                                if (position >= end) break;
                            }
                        }
                        finally
                        {
                            buffer?.Dispose();
                            Interlocked.Add(ref readBytes, localRead);
                            Interlocked.Add(ref residentBytes, localResident);
                        }
                    });
                }
                finally
                {
                    foreach (var handle in handles) handle?.Dispose();
                }
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or AggregateException)
        {
            // Warming is optional; actual tensor reads retain their strict checks.
        }
        PrefaultReadBytes = readBytes;
        PrefaultResidentBytes = residentBytes;
        if (watch.Elapsed.TotalSeconds >= 1)
            Console.WriteLine($"  Prefaulted GGUF page cache in {watch.Elapsed.TotalSeconds:F1}s with {threads} read streams: " +
                $"{readBytes / 1073741824.0:F2} GiB read, {residentBytes / 1073741824.0:F2} GiB already resident");
    }

    internal static bool CanPrefaultFileCache(long tensorBytes, long availableBytes, int threads)
    {
        if (tensorBytes <= 0 || availableBytes <= 0 || threads <= 0) return false;
        // Leave half of CURRENT physical availability for execution/other users.
        // Read buffers coexist with the warmed file pages. A GC heap ceiling is
        // neither this availability nor an OS page-cache allowance.
        long buffers = Math.Min(tensorBytes, (long)threads * (8 << 20));
        return (decimal)tensorBytes + buffers <= availableBytes / 2;
    }

    private sealed unsafe class PrefaultReadBuffer : IDisposable
    {
        private byte* _data;
        private readonly int _length;
        private HostAllocationBudgetScope.Allocation? _charge;

        internal PrefaultReadBuffer(int length)
        {
            _length = length;
            _charge = HostAllocationBudgetScope.Reserve(length);
            try
            {
                _data = (byte*)NativeMemory.Alloc((nuint)length);
                if (_data == null) throw new OutOfMemoryException();
                _charge?.Commit();
            }
            catch { Dispose(); throw; }
        }

        internal Span<byte> Span => new(_data, _length);
        public void Dispose()
        {
            NativeMemory.Free(_data);
            _data = null;
            _charge?.Dispose();
            _charge = null;
        }
    }

    // Resolve ownership through the root table; sibling tensors are merged into
    // it at parse time. Coalesce only adjacent selected ranges, keeping sparse
    // tables excluded even when they sit between two eagerly loaded weights.
    private List<(long Offset, long Length)> GetPrefaultRanges(GgufFile file, Func<GgufTensorInfo, bool>? includeTensor)
    {
        long fileLength = file._stream.Length;
        var ranges = new List<(long Offset, long Length)>();
        foreach (var info in Tensors.Values)
        {
            if (!ReferenceEquals(OwnerOf(info), file) || (includeTensor != null && !includeTensor(info))) continue;
            long offset, length;
            try
            {
                offset = checked(file.DataOffset + (long)info.Offset);
                length = GetTensorByteCount(info);
            }
            catch (Exception error) when (error is NotSupportedException or IndexOutOfRangeException or OverflowException)
            {
                // An optional warmup must not reject an unconsumed auxiliary
                // tensor. The actual loader validates tensors it needs.
                continue;
            }
            if (offset < 0 || length <= 0 || offset > fileLength || length > fileLength - offset) continue;
            ranges.Add((offset, length));
        }
        ranges.Sort((a, b) => a.Offset.CompareTo(b.Offset));
        var merged = new List<(long Offset, long Length)>();
        foreach (var range in ranges)
        {
            if (merged.Count > 0 && range.Offset <= merged[^1].Offset + merged[^1].Length)
            {
                var previous = merged[^1];
                merged[^1] = (previous.Offset, Math.Max(previous.Offset + previous.Length, range.Offset + range.Length) - previous.Offset);
            }
            else merged.Add(range);
        }
        return merged;
    }

    private unsafe bool IsRangeResident(long offset, int length, byte[] residency)
    {
        long pageSize = Environment.SystemPageSize;
        long aligned = offset - offset % pageSize;
        long bytes = offset - aligned + length;
        int pages = checked((int)((bytes + pageSize - 1) / pageSize));
        fixed (byte* vector = residency)
        {
            if (mincore(_mappedBase + aligned, (nuint)bytes, vector) != 0) return false;
            for (int i = 0; i < pages; i++)
                if ((vector[i] & 1) == 0) return false;
        }
        return true;
    }

    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static unsafe partial int mincore(void* address, nuint length, byte* vector);
}
