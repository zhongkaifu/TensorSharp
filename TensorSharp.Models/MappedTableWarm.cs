// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// Warms host-mapped lookup tables into the page cache. A table read a few scattered rows per
    /// token (DeepSeek V4.1's Engram tables, Qwen3.8-Flash-Next's PLE n-gram table) pays a storage
    /// round trip for every row whose page is not cached: about a millisecond on a network
    /// filesystem, so a cold table can cost several times the decode step. Reading the tables once,
    /// on a background thread after the model is ready, takes that off the requests.
    /// </summary>
    internal static class MappedTableWarm
    {
        private const int Block = 8 << 20;
        private const long Headroom = 8L << 30;
        private const int MaxWorkers = 4;
        // Two models must not each admit a whole-table warm against the same available memory.
        // Recheck after acquiring the gate; completed reads remain reusable in the OS page cache.
        private static readonly SemaphoreSlim WarmGate = new(1, 1);

        /// <summary>
        /// Start reading <paramref name="ranges"/> (shard path, byte offset, byte count), or return null
        /// with a note when the host has no room to keep them cached. The task completes when every
        /// block has been read, <paramref name="cancel"/> fires, or a read fails (reported, not thrown).
        /// </summary>
        public static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges, string tag, string what,
            CancellationToken cancel)
            => Start(ranges, tag, what, cancel, HostMemoryAvailable, Headroom, Block, MaxWorkers,
                Console.Error.WriteLine);

        internal static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges, string tag, string what,
            CancellationToken cancel, Func<long> memoryAvailable, long headroom, int blockBytes, int maxWorkers,
            Action<string> log)
        {
            if (ranges == null || ranges.Count == 0)
                return null;
            ArgumentNullException.ThrowIfNull(memoryAvailable);
            ArgumentNullException.ThrowIfNull(log);
            if (headroom < 0 || blockBytes <= 0 || maxWorkers <= 0)
                throw new ArgumentOutOfRangeException(nameof(blockBytes));
            var snapshot = ranges.ToArray();
            long bytes = 0;
            foreach (var range in snapshot)
            {
                if (string.IsNullOrEmpty(range.Path) || range.Offset < 0 || range.Bytes < 0 ||
                    range.Offset > long.MaxValue - range.Bytes)
                    throw new ArgumentException("Invalid mapped-table file range.", nameof(ranges));
                bytes = checked(bytes + range.Bytes);
            }
            if (bytes == 0 || cancel.IsCancellationRequested)
                return null;

            int Workers()
            {
                long available = memoryAvailable();
                int count = WorkerCount(bytes, available, headroom, blockBytes, maxWorkers);
                if (count == 0)
                    log(available <= 0
                        ? $"[{tag}] {what} warming skipped: available host memory is unknown"
                        : $"[{tag}] {what} warming skipped: {available / (double)(1L << 30):F1} GiB available cannot hold " +
                          $"{bytes / (double)(1L << 30):F1} GiB of tables plus headroom and read buffers");
                return count;
            }

            if (Workers() == 0)
                return null;
            return Task.Run(async () =>
            {
                bool entered = false;
                try
                {
                    await WarmGate.WaitAsync(cancel).ConfigureAwait(false);
                    entered = true;
                    int workers = Workers();
                    if (workers == 0)
                        return;
                    var sw = Stopwatch.StartNew();
                    var cursorLock = new object();
                    int rangeIndex = 0;
                    long rangeOffset = 0;
                    long readBytes = 0;
                    bool pressure = false;
                    using var stopped = CancellationTokenSource.CreateLinkedTokenSource(cancel);

                    // Fixed worker bodies bound the number of retained byte arrays, unlike
                    // thread-local arrays which can accumulate when Parallel changes workers.
                    Parallel.For(0, workers, new ParallelOptions { MaxDegreeOfParallelism = workers }, _ =>
                    {
                        var open = new Dictionary<string, Microsoft.Win32.SafeHandles.SafeFileHandle>();
                        try
                        {
                            var buffer = new byte[blockBytes];
                            while (!stopped.IsCancellationRequested)
                            {
                                (string Path, long Offset, int Bytes) block;
                                lock (cursorLock)
                                {
                                    while (rangeIndex < snapshot.Length && rangeOffset == snapshot[rangeIndex].Bytes)
                                    {
                                        rangeIndex++;
                                        rangeOffset = 0;
                                    }
                                    if (rangeIndex == snapshot.Length)
                                        return;
                                    long available = memoryAvailable();
                                    // Unknown memory is not permission to continue a large read.
                                    if (available <= 0 || available - Math.Min(available, headroom) < (long)workers * blockBytes)
                                    {
                                        pressure = true;
                                        stopped.Cancel();
                                        return;
                                    }
                                    var range = snapshot[rangeIndex];
                                    int count = (int)Math.Min(blockBytes, range.Bytes - rangeOffset);
                                    block = (range.Path, range.Offset + rangeOffset, count);
                                    rangeOffset += count;
                                }
                                if (!open.TryGetValue(block.Path, out var handle))
                                    open[block.Path] = handle = File.OpenHandle(block.Path, FileMode.Open, FileAccess.Read, FileShare.Read);
                                int done = 0;
                                while (done < block.Bytes)
                                {
                                    stopped.Token.ThrowIfCancellationRequested();
                                    int count = RandomAccess.Read(handle, buffer.AsSpan(done, block.Bytes - done), block.Offset + done);
                                    if (count == 0)
                                        throw new EndOfStreamException($"Mapped table ended before the requested range in '{block.Path}'.");
                                    done += count;
                                }
                                Interlocked.Add(ref readBytes, done);
                            }
                        }
                        catch
                        {
                            stopped.Cancel();
                            throw;
                        }
                        finally
                        {
                            foreach (var handle in open.Values)
                                handle.Dispose();
                        }
                    });
                    if (pressure)
                        log($"[{tag}] {what} warming stopped: available host memory fell below the headroom and read-buffer allowance");
                    else if (!cancel.IsCancellationRequested && readBytes == bytes)
                        log($"[{tag}] warmed {bytes / (double)(1L << 30):F1} GiB of {what} in {sw.Elapsed.TotalSeconds:F1}s");
                }
                catch (OperationCanceledException) { }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException ||
                    ex is AggregateException aggregate && aggregate.Flatten().InnerExceptions.All(
                        inner => inner is IOException or UnauthorizedAccessException or OperationCanceledException))
                {
                    log($"[{tag}] {what} warming stopped: {ex.GetBaseException().Message}");
                }
                finally
                {
                    if (entered)
                        WarmGate.Release();
                }
            });
        }

        internal static int WorkerCount(long bytes, long available, long headroom, int blockBytes, int maxWorkers)
        {
            if (bytes <= 0 || available <= 0 || headroom < 0 || blockBytes <= 0 || maxWorkers <= 0 ||
                headroom >= available || bytes >= available - headroom)
                return 0;
            long room = available - headroom - bytes;
            long blocks = bytes / blockBytes + (bytes % blockBytes == 0 ? 0 : 1);
            return (int)Math.Min(Math.Min(maxWorkers, blocks), room / blockBytes);
        }

        /// <summary>Currently available physical RAM, including reclaimable cache, or 0 if unknown.</summary>
        public static long HostMemoryAvailable()
        {
            if (OperatingSystem.IsWindows())
            {
                var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
                return GlobalMemoryStatusEx(ref status) && status.AvailablePhysical <= long.MaxValue
                    ? (long)status.AvailablePhysical : 0;
            }
            if (!OperatingSystem.IsLinux())
                return 0;
            try
            {
                return HostMemoryAvailability.CaptureLinux().Available;
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return 0;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct MemoryStatus
        {
            public uint Length;
            public uint MemoryLoad;
            public ulong TotalPhysical;
            public ulong AvailablePhysical;
            public ulong TotalPageFile;
            public ulong AvailablePageFile;
            public ulong TotalVirtual;
            public ulong AvailableVirtual;
            public ulong AvailableExtendedVirtual;
        }

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
    }
}
