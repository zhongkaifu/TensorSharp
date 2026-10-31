// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Warms host-mapped lookup tables into the page cache. A table read a few scattered rows per
    /// token (DeepSeek V4.1's Engram tables, Qwen3.8-Flash-Next's PLE n-gram table) pays a storage
    /// round trip for every row whose page is not cached: about a millisecond on a network
    /// filesystem, so a cold table can cost several times the decode step. Reading the tables once,
    /// after the model is ready, takes that off the requests.
    /// </summary>
    /// <remarks>
    /// <para>Budget. A warm table is page cache, and page cache is charged to the process's memory cgroup:
    /// the kernel evicts it as soon as anything else needs the room, so warming more than the cgroup can
    /// hold only pushes the first half out with the second. The budget is the cgroup's limit minus its
    /// unreclaimable memory (anonymous and shmem) and a headroom, and also the host's MemAvailable minus the
    /// headroom; tables that do not fit are not warmed, with the numbers in the message.</para>
    /// <para>Re-warm. Where the table is mapped, each block is first checked with mincore and only the pages
    /// that are not resident are read, so a process started after another one, or after a job evicted part
    /// of the table, reads only what is missing.</para>
    /// <para>Reads. A fixed number of threads, each with its own file handle (FUSE serializes positional reads
    /// on a shared descriptor), take 64 MiB blocks in order.</para>
    /// </remarks>
    internal static class MappedTableWarm
    {
        internal const long Block = 64L << 20;
        internal const long DefaultHeadroom = 8L << 30;
        internal const int DefaultThreads = 16;
        // Two missing runs this close are read as one: re-reading a cached page costs a copy, a separate read a
        // round trip.
        private const long MergeGap = 1L << 20;

        /// <summary>One table: its shard, its byte range there, and (optionally) where the process maps those
        /// bytes, which lets the warm skip what is already resident.</summary>
        internal readonly record struct Range(string Path, long Offset, long Bytes, IntPtr Mapped);

        internal sealed class Options
        {
            public int Threads = DefaultThreads;
            public long Headroom = DefaultHeadroom;
            /// <summary>Check residency with mincore and read only the missing pages (needs <see cref="Range.Mapped"/>).</summary>
            public bool SkipResident = true;
            /// <summary>The cgroup to budget against (tests); null reads this process's.</summary>
            public Func<CgroupMemory> Cgroup;
            /// <summary>Host MemAvailable in bytes (tests); null reads /proc/meminfo.</summary>
            public Func<long> HostAvailable;
        }

        /// <summary>What one warm did.</summary>
        internal readonly record struct Result(long Bytes, long Read, long AlreadyResident, int Blocks, int BlocksSkipped,
            bool Cancelled, string Error, TimeSpan Elapsed);

        /// <summary>The verdict of the budget check, with the numbers it was made from.</summary>
        internal readonly record struct Budget(bool Fits, long Bytes, long Available, string Description);

        /// <summary>
        /// Start reading <paramref name="ranges"/> (shard path, byte offset, byte count), or return null
        /// with a note when the host has no room to keep them cached. The task completes when every
        /// block has been read, <paramref name="cancel"/> fires, or a read fails (reported, not thrown).
        /// </summary>
        public static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges, string tag, string what,
            CancellationToken cancel)
        {
            if (ranges == null)
                return null;
            var mapped = new List<Range>(ranges.Count);
            foreach (var (path, offset, bytes) in ranges)
                mapped.Add(new Range(path, offset, bytes, IntPtr.Zero));
            return Start(mapped, tag, what, cancel, null);
        }

        public static Task Start(IReadOnlyList<Range> ranges, string tag, string what, CancellationToken cancel, Options options)
        {
            if (ranges == null || ranges.Count == 0)
                return null;
            options ??= new Options();
            long bytes = 0;
            foreach (var range in ranges)
                bytes += range.Bytes;
            Budget budget = CheckBudget(bytes, options);
            if (!budget.Fits)
            {
                Console.Error.WriteLine($"[{tag}] {what} warming refused: {Gib(bytes)} of tables would not stay cached. " +
                    budget.Description + ". Free memory in this cgroup, raise its limit, or move tables to another node " +
                    "(TS_DSV41_ENGRAM_OWNER=balance on a cross-node layer split); TS_DSV41_ENGRAM_WARM=0 silences this.");
                return null;
            }
            Console.Error.WriteLine($"[{tag}] warming {Gib(bytes)} of {what} with {options.Threads} threads; {budget.Description}");
            return Task.Factory.StartNew(() =>
            {
                Result r = Run(ranges, options.Threads, options.SkipResident, cancel);
                if (r.Error != null)
                    Console.Error.WriteLine($"[{tag}] {what} warming stopped: {r.Error}");
                else if (!r.Cancelled)
                    Console.Error.WriteLine($"[{tag}] warmed {Gib(r.Bytes)} of {what} in {r.Elapsed.TotalSeconds:F1}s: " +
                        $"{Gib(r.AlreadyResident)} already cached ({r.BlocksSkipped} of {r.Blocks} blocks whole), " +
                        $"read {Gib(r.Read)} ({r.Read / 1e9 / Math.Max(1e-9, r.Elapsed.TotalSeconds):F2} GB/s)");
            }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        }

        /// <summary>
        /// Whether <paramref name="bytes"/> of tables fit: under the cgroup's limit less its unreclaimable memory and
        /// the headroom, and under the host's MemAvailable less the headroom (whichever exist).
        /// </summary>
        internal static Budget CheckBudget(long bytes, Options options)
        {
            options ??= new Options();
            CgroupMemory cgroup = options.Cgroup != null ? options.Cgroup() : CgroupMemory.Read();
            long host = options.HostAvailable != null ? options.HostAvailable() : HostMemoryAvailable();
            long available = long.MaxValue;
            var parts = new List<string>();
            if (cgroup != null)
            {
                long cg = cgroup.CacheBudget(options.Headroom);
                if (cgroup.Limited)
                    available = Math.Min(available, cg);
                parts.Add(cgroup.Describe() + (cgroup.Limited
                    ? $"; room for tables {Gib(Math.Max(0, cg))} after {Gib(options.Headroom)} headroom"
                    : "; no cgroup limit"));
            }
            else
            {
                parts.Add("no memory cgroup to read");
            }
            if (host > 0)
            {
                available = Math.Min(available, host - options.Headroom);
                parts.Add($"host MemAvailable {Gib(host)}");
            }
            return new Budget(bytes <= available, bytes, available, string.Join("; ", parts));
        }

        /// <summary>
        /// Reads <paramref name="ranges"/> into the page cache with <paramref name="threads"/> threads, each on its own
        /// handles, skipping pages a mapping reports resident when <paramref name="skipResident"/>. Synchronous; the
        /// first read error stops it and is returned, not thrown.
        /// </summary>
        internal static Result Run(IReadOnlyList<Range> ranges, int threads, bool skipResident, CancellationToken cancel)
        {
            var sw = Stopwatch.StartNew();
            var blocks = new List<(Range Range, long At, long Bytes)>();
            long total = 0;
            foreach (Range range in ranges)
            {
                total += range.Bytes;
                for (long at = 0; at < range.Bytes; at += Block)
                    blocks.Add((range, at, Math.Min(Block, range.Bytes - at)));
            }
            // mincore lies (every page resident) to a process that may not write the file: read those ranges whole.
            var visible = new Dictionary<string, bool>(StringComparer.Ordinal);
            if (skipResident)
            {
                foreach (Range range in ranges)
                {
                    if (range.Mapped != IntPtr.Zero && !visible.ContainsKey(range.Path))
                        visible[range.Path] = PageCacheHints.ResidencyVisible(range.Path);
                }
            }
            long read = 0, resident = 0;
            int next = -1, skipped = 0;
            string error = null;
            threads = Math.Clamp(threads, 1, 64);
            var workers = new Thread[Math.Min(threads, Math.Max(1, blocks.Count))];
            for (int w = 0; w < workers.Length; w++)
            {
                workers[w] = new Thread(() =>
                {
                    var handles = new Dictionary<string, Microsoft.Win32.SafeHandles.SafeFileHandle>(StringComparer.Ordinal);
                    var runs = new List<(long Offset, long Bytes)>();
                    byte[] buffer = null;
                    try
                    {
                        while (!cancel.IsCancellationRequested && Volatile.Read(ref error) == null)
                        {
                            int i = Interlocked.Increment(ref next);
                            if (i >= blocks.Count)
                                break;
                            var (range, at, length) = blocks[i];
                            bool known = skipResident && range.Mapped != IntPtr.Zero && visible[range.Path]
                                && PageCacheHints.TryMissingRuns(range.Mapped + (nint)at, length, MergeGap, runs);
                            if (!known)
                            {
                                runs.Clear();
                                runs.Add((0, length));
                            }
                            long missing = 0;
                            foreach (var (_, b) in runs)
                                missing += b;
                            Interlocked.Add(ref resident, length - missing);
                            if (runs.Count == 0)
                            {
                                Interlocked.Increment(ref skipped);
                                continue;
                            }
                            if (!handles.TryGetValue(range.Path, out var handle))
                                handles[range.Path] = handle = File.OpenHandle(range.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
                            buffer ??= new byte[Block];
                            foreach (var (offset, b) in runs)
                            {
                                long done = 0;
                                while (done < b)
                                {
                                    int got = RandomAccess.Read(handle, new Span<byte>(buffer, 0, (int)(b - done)), range.Offset + at + offset + done);
                                    if (got <= 0)
                                        throw new IOException($"short read at offset {range.Offset + at + offset + done} in {range.Path}");
                                    done += got;
                                }
                            }
                            Interlocked.Add(ref read, missing);
                        }
                    }
                    catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
                    {
                        Interlocked.CompareExchange(ref error, ex.Message, null);
                    }
                    finally
                    {
                        foreach (var handle in handles.Values)
                            handle.Dispose();
                    }
                })
                { IsBackground = true, Name = $"table-warm-{w}" };
                workers[w].Start();
            }
            foreach (Thread worker in workers)
                worker.Join();
            return new Result(total, read, resident, blocks.Count, skipped, cancel.IsCancellationRequested, error, sw.Elapsed);
        }

        /// <summary>MemAvailable from /proc/meminfo in bytes, or 0 where there is none.</summary>
        public static long HostMemoryAvailable()
        {
            try
            {
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                        continue;
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1], CultureInfo.InvariantCulture) * 1024;
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return 0;
        }

        internal static string Gib(long bytes)
            => (bytes / (double)(1L << 30)).ToString("F1", CultureInfo.InvariantCulture) + " GiB";
    }
}
