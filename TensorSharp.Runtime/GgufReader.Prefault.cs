// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

public partial class GgufFile
{
    private bool _prefaulted;

    /// <summary>Tests: the memory the prefault budget is half of, instead of the process's.</summary>
    internal static long? AvailableMemoryOverrideForTests;
    internal long PrefaultReadBytes { get; private set; }
    internal long PrefaultResidentBytes { get; private set; }

    /// <summary>Warm tensor data with bounded parallel reads. On Linux, resident
    /// pages are checked without copying them; repeated loads avoid rereading the
    /// whole checkpoint. Sparse host lookup tables may be excluded by the model.</summary>
    public void PrefaultFileCache() => PrefaultFileCache(null);

    public void PrefaultFileCache(Func<GgufTensorInfo, bool>? includeTensor)
        => PrefaultFileCache(includeTensor, logProgress: true);

    public void PrefaultFileCache(Func<GgufTensorInfo, bool>? includeTensor, bool logProgress)
        => PrefaultFileCache(includeTensor, null, logProgress);

    /// <summary>
    /// As <see cref="PrefaultFileCache(Func{GgufTensorInfo, bool}?, bool)"/>. When the selected
    /// tensors are more than half the memory available to the process, the page cache could not
    /// keep them all and warming them would only evict what the loader is about to read, so only
    /// <paramref name="whenConstrained"/>'s subset is warmed: the tensors the load reads at once
    /// (dense weights bound for the accelerator, F32 tensors), not the routed experts that are
    /// read later and only in part. Without it nothing is warmed, and on network storage every
    /// eagerly loaded tensor then arrives through serial page faults (Qwen3.8-Flash-Next on a
    /// geesefs mount: ~50 MB/s for the 4 GB of dense weights, against hundreds of MB/s read in
    /// parallel).
    /// </summary>
    public void PrefaultFileCache(Func<GgufTensorInfo, bool>? includeTensor,
        Func<GgufTensorInfo, bool>? whenConstrained, bool logProgress)
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
        long available = AvailableMemoryOverrideForTests ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        var plans = PrefaultPlans(includeTensor, out long total);
        string scope = "";
        if (total == 0 || (available > 0 && total > available / 2))
        {
            if (whenConstrained == null) return;
            plans = PrefaultPlans(info => (includeTensor?.Invoke(info) ?? true) && whenConstrained(info), out total);
            if (total == 0 || (available > 0 && total > available / 2)) return;
            scope = " (the eagerly loaded tensors only: the checkpoint is larger than the page cache can hold)";
        }

        // Every read goes through the files' handle pools, opened here once, before any tensor
        // data is read: some FUSE mounts drop a file's whole page cache at every open().
        try { OpenReadHandles(); }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException) { return; }
        int threads = DefaultWarmThreads();
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
                SafeFileHandle[] handles = plan.File.ReadHandles();
                {
                    Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, region =>
                    {
                        long start = region * regionBytes, end = Math.Min(start + regionBytes, length);
                        long position = 0, localRead = 0, localResident = 0;
                        byte[]? buffer = null;
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
                                        buffer ??= ArrayPool<byte>.Shared.Rent(8 << 20);
                                        int done = 0;
                                        while (done < count)
                                        {
                                            int read = RandomAccess.Read(handle, buffer.AsSpan(done, count - done), fileOffset + done);
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
                            if (buffer != null) ArrayPool<byte>.Shared.Return(buffer);
                            Interlocked.Add(ref readBytes, localRead);
                            Interlocked.Add(ref residentBytes, localResident);
                        }
                    });
                }
            }
        }
        catch (Exception error) when (error is IOException or UnauthorizedAccessException or AggregateException)
        {
            // Warming is optional; actual tensor reads retain their strict checks.
        }
        PrefaultReadBytes = readBytes;
        PrefaultResidentBytes = residentBytes;
        if (logProgress && watch.Elapsed.TotalSeconds >= 1)
            Console.WriteLine($"  Prefaulted GGUF page cache in {watch.Elapsed.TotalSeconds:F1}s with {threads} read streams: " +
                $"{readBytes / 1073741824.0:F2} GiB read, {residentBytes / 1073741824.0:F2} GiB already resident{scope}");
    }

    private List<(GgufFile File, List<(long Offset, long Length)> Ranges)> PrefaultPlans(
        Func<GgufTensorInfo, bool>? includeTensor, out long total)
    {
        var plans = new List<(GgufFile File, List<(long Offset, long Length)> Ranges)>();
        total = 0;
        foreach (GgufFile file in new[] { this }.Concat(_shards))
        {
            var ranges = GetPrefaultRanges(file, includeTensor);
            if (ranges.Count == 0) continue;
            plans.Add((file, ranges));
            total += ranges.Sum(r => r.Length);
        }
        return plans;
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
