// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

/// <summary>
/// Explicit, parallel reads of a mapped checkpoint into the page cache, through handles that are
/// opened ONCE per file and kept for the file's lifetime.
/// </summary>
/// <remarks>
/// A FUSE mount that does not keep its cache across opens (geesefs, measured on a 5090 VM: a
/// 64 MB range read through one descriptor went from 64 MB resident to 0 the moment a second
/// descriptor was opened, with the mapping still live) drops a file's ENTIRE page cache at every
/// open(). A warm-up that opened a fresh handle per call (or per worker, lazily) therefore threw
/// away what the previous one had read, and everything the loader had read before it. So every
/// explicit read of a model file goes through this pool, which is opened before the first byte is
/// read and never re-opened.
/// </remarks>
public partial class GgufFile
{
    /// <summary>Handles per file. One descriptor per reader keeps per-descriptor readahead on
    /// network filesystems (MooseFS lost ~3x with a shared cursor); geesefs scales with requests in
    /// flight (85 MB/s on one stream, 466 MB/s at 32, while another job was writing to the mount).</summary>
    internal const int ReadHandlePoolSize = 32;

    private SafeFileHandle[]? _readHandles;
    private readonly object _readHandleLock = new();

    /// <summary>The pool of read handles of this one file (not its shards), opened on first use.</summary>
    internal SafeFileHandle[] ReadHandles()
    {
        lock (_readHandleLock)
        {
            if (_readHandles != null)
                return _readHandles;
            var handles = new SafeFileHandle[ReadHandlePoolSize];
            try
            {
                for (int i = 0; i < handles.Length; i++)
                    handles[i] = File.OpenHandle(_path, FileMode.Open, FileAccess.Read, FileShare.Read);
            }
            catch
            {
                foreach (var handle in handles) handle?.Dispose();
                throw;
            }
            return _readHandles = handles;
        }
    }

    /// <summary>Open every file's read pool now, before the loader reads tensor data: an open after
    /// that point could drop pages already read (see the type remarks).</summary>
    public void OpenReadHandles()
    {
        ReadHandles();
        foreach (var shard in _shards)
            shard.ReadHandles();
    }

    private void DisposeReadHandles()
    {
        lock (_readHandleLock)
        {
            if (_readHandles == null) return;
            foreach (var handle in _readHandles) handle?.Dispose();
            _readHandles = null;
        }
    }

    /// <summary>The file (this one or a shard) and file offset of a pointer into a mapped view, or
    /// false when the pointer is not inside one.</summary>
    public unsafe bool TryResolveMapped(IntPtr pointer, long bytes, out GgufFile file, out long offset)
    {
        file = null!;
        offset = 0;
        long p = pointer.ToInt64();
        foreach (var candidate in Files())
        {
            if (candidate._mappedBase == null) continue;
            long start = (long)candidate._mappedBase, length = candidate._stream.Length;
            if (p >= start && p + bytes <= start + length)
            {
                file = candidate;
                offset = p - start;
                return true;
            }
        }
        return false;
    }

    private IEnumerable<GgufFile> Files()
    {
        yield return this;
        foreach (var shard in _shards)
            yield return shard;
    }

    /// <summary>
    /// Read the mapped byte ranges <paramref name="ranges"/> (pointers into this checkpoint's views)
    /// into the page cache with up to <paramref name="threads"/> parallel large reads. Ranges are
    /// resolved to files, sorted, coalesced across gaps up to 1 MiB, and each worker reads one
    /// contiguous share. Pointers outside every view are skipped. Returns the bytes read.
    /// </summary>
    public long WarmMappedRanges(IReadOnlyList<(IntPtr Pointer, long Bytes)> ranges, int threads,
        CancellationToken cancel = default)
    {
        var fileRanges = new List<(GgufFile File, long Offset, long Length)>(ranges.Count);
        foreach (var (pointer, bytes) in ranges)
            if (bytes > 0 && TryResolveMapped(pointer, bytes, out var file, out long offset))
                fileRanges.Add((file, offset, bytes));
        return WarmFileRanges(fileRanges, threads, cancel);
    }

    /// <summary>As <see cref="WarmMappedRanges"/>, for (file, offset, length) ranges of this checkpoint.</summary>
    internal static long WarmFileRanges(List<(GgufFile File, long Offset, long Length)> ranges, int threads,
        CancellationToken cancel = default)
    {
        const long Gap = 1L << 20;
        const int Chunk = 8 << 20;
        if (ranges.Count == 0) return 0;
        ranges.Sort((a, b) => a.File == b.File ? a.Offset.CompareTo(b.Offset)
            : string.CompareOrdinal(a.File._path, b.File._path));
        var merged = new List<(GgufFile File, long Offset, long Length)>();
        foreach (var r in ranges)
        {
            if (merged.Count > 0 && merged[^1].File == r.File && r.Offset <= merged[^1].Offset + merged[^1].Length + Gap)
            {
                var last = merged[^1];
                merged[^1] = (last.File, last.Offset, Math.Max(last.Offset + last.Length, r.Offset + r.Length) - last.Offset);
            }
            else merged.Add(r);
        }
        long total = 0;
        foreach (var r in merged) total += r.Length;
        int workers = (int)Math.Clamp((total + Chunk - 1) / Chunk, 1, Math.Max(1, threads));
        long share = (total + workers - 1) / workers;
        long read = 0;
        Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers, CancellationToken = cancel }, worker =>
        {
            long begin = worker * share, end = Math.Min(total, begin + share), position = 0, local = 0;
            byte[] buffer = ArrayPool<byte>.Shared.Rent(Chunk);
            try
            {
                foreach (var r in merged)
                {
                    if (position + r.Length <= begin) { position += r.Length; continue; }
                    if (position >= end || cancel.IsCancellationRequested) break;
                    SafeFileHandle handle = r.File.ReadHandles()[worker % ReadHandlePoolSize];
                    long from = Math.Max(begin - position, 0), to = Math.Min(end - position, r.Length);
                    for (long at = from; at < to && !cancel.IsCancellationRequested;)
                    {
                        int count = (int)Math.Min(Chunk, to - at);
                        int got = RandomAccess.Read(handle, buffer.AsSpan(0, count), r.Offset + at);
                        if (got <= 0) break;
                        at += got;
                        local += got;
                    }
                    position += r.Length;
                }
            }
            catch (IOException) { }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
                Interlocked.Add(ref read, local);
            }
        });
        return read;
    }

    /// <summary>Default reader count for warming: one per pooled handle. The readers mostly wait on
    /// storage, so the count does not follow the CPU quota (a 32-core VM with a 13.6-CPU cgroup
    /// reports 14 processors); more requests in flight pay off on network storage, and an NVMe
    /// queue is deeper than this.</summary>
    public static int DefaultWarmThreads()
    {
        if (int.TryParse(Environment.GetEnvironmentVariable("TS_GGUF_PREFAULT_THREADS"), out int requested) && requested > 0)
            return Math.Min(requested, ReadHandlePoolSize);
        return ReadHandlePoolSize;
    }
}
