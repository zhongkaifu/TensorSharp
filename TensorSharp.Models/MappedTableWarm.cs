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
        private const long Block = 64L << 20;
        internal const long Headroom = 8L << 30;

        /// <summary>
        /// Start reading <paramref name="ranges"/> (shard path, byte offset, byte count), or return null
        /// with a note when the host has no room to keep them cached. The task completes when every
        /// block has been read, <paramref name="cancel"/> fires, or a read fails (reported, not thrown).
        /// </summary>
        public static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges, string tag, string what,
            CancellationToken cancel)
        {
            if (ranges == null || ranges.Count == 0)
                return null;
            long bytes = 0;
            foreach (var range in ranges)
                bytes += range.Bytes;
            long available = HostMemoryAvailable();
            if (available > 0 && available < bytes + Headroom)
            {
                Console.Error.WriteLine($"[{tag}] {what} warming skipped: {available / (double)(1L << 30):F1} GiB " +
                    $"of host memory available would not keep {bytes / (double)(1L << 30):F1} GiB of tables cached");
                return null;
            }
            return Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                var blocks = new List<(string Path, long Offset, long Bytes)>();
                foreach (var (path, offset, length) in ranges)
                    for (long at = 0; at < length; at += Block)
                        blocks.Add((path, offset + at, Math.Min(Block, length - at)));
                // Every handle is opened before the first read: some FUSE mounts (geesefs) drop a
                // file's whole page cache at each open(), so a worker opening its handle lazily
                // threw away the blocks the others had just read.
                const int Readers = 16;
                var handles = new Dictionary<string, Microsoft.Win32.SafeHandles.SafeFileHandle[]>();
                using var buffers = new ThreadLocal<byte[]>(() => new byte[Block]);
                int next = 0;
                try
                {
                    foreach (var (path, _, _) in ranges)
                        if (!handles.ContainsKey(path))
                        {
                            var pool = new Microsoft.Win32.SafeHandles.SafeFileHandle[Readers];
                            handles[path] = pool;
                            for (int i = 0; i < Readers; i++)
                                pool[i] = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
                        }
                    Parallel.ForEach(blocks, new ParallelOptions { MaxDegreeOfParallelism = Readers, CancellationToken = cancel }, block =>
                    {
                        var handle = handles[block.Path][(Interlocked.Increment(ref next) & int.MaxValue) % Readers];
                        RandomAccess.Read(handle, new Span<byte>(buffers.Value, 0, (int)block.Bytes), block.Offset);
                    });
                    Console.Error.WriteLine($"[{tag}] warmed {bytes / (double)(1L << 30):F1} GiB of {what} in {sw.Elapsed.TotalSeconds:F1}s");
                }
                catch (OperationCanceledException) { }
                catch (IOException ex)
                {
                    Console.Error.WriteLine($"[{tag}] {what} warming stopped: {ex.Message}");
                }
                finally
                {
                    foreach (var pool in handles.Values)
                        foreach (var handle in pool)
                            handle?.Dispose();
                }
            });
        }

        /// <summary>Host memory a table could still occupy as page cache, in bytes, or 0 where
        /// the platform reports none: MemAvailable from /proc/meminfo, capped by the process's
        /// cgroup. A container sees the host's MemAvailable, which can be far more than its own
        /// memory limit lets it cache.</summary>
        public static long HostMemoryAvailable()
        {
            long available = MemInfoAvailable();
            long cgroup = CgroupMemoryAvailable();
            if (cgroup > 0 && (available <= 0 || cgroup < available))
                available = cgroup;
            return available;
        }

        private static long MemInfoAvailable()
        {
            try
            {
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemAvailable:", StringComparison.Ordinal))
                        continue;
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1]) * 1024;
                }
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            return 0;
        }

        /// <summary>The cgroup's limit minus what it holds that the kernel cannot reclaim
        /// (its usage less its file cache), or 0 without a limited memory cgroup. cgroup v2
        /// (<c>memory.max</c>) and v1 (<c>memory.limit_in_bytes</c>).</summary>
        internal static long CgroupMemoryAvailable()
        {
            try
            {
                string v2 = null, v1 = null;
                if (File.Exists("/proc/self/cgroup"))
                    foreach (string line in File.ReadLines("/proc/self/cgroup"))
                    {
                        if (line.StartsWith("0::", StringComparison.Ordinal)) v2 = line.Substring(3);
                        else
                        {
                            string[] f = line.Split(':', 3);
                            if (f.Length == 3 && Array.IndexOf(f[1].Split(','), "memory") >= 0) v1 = f[2];
                        }
                    }
                foreach (string dir in Candidates("/sys/fs/cgroup", v2))
                    if (TryCgroup(dir, "memory.max", "memory.current", "active_file", "inactive_file", out long a))
                        return a;
                foreach (string dir in Candidates("/sys/fs/cgroup/memory", v1))
                    if (TryCgroup(dir, "memory.limit_in_bytes", "memory.usage_in_bytes", "total_active_file", "total_inactive_file", out long a))
                        return a;
            }
            catch (IOException) { }
            catch (UnauthorizedAccessException) { }
            catch (FormatException) { }
            return 0;
        }

        private static IEnumerable<string> Candidates(string root, string relative)
        {
            if (!string.IsNullOrEmpty(relative) && relative != "/")
                yield return root + relative;
            yield return root;
        }

        internal static bool TryCgroup(string dir, string limitFile, string usageFile, string activeKey, string inactiveKey,
            out long available)
        {
            available = 0;
            string limitPath = Path.Combine(dir, limitFile), usagePath = Path.Combine(dir, usageFile);
            string statPath = Path.Combine(dir, "memory.stat");
            if (!File.Exists(limitPath) || !File.Exists(usagePath) || !File.Exists(statPath))
                return false;
            string limitText = File.ReadAllText(limitPath).Trim();
            // "max" (v2) or a page-rounded huge number (v1): no limit, the host figure stands.
            if (limitText == "max" || !long.TryParse(limitText, out long limit) || limit <= 0 || limit >= (1L << 62))
                return false;
            long usage = long.Parse(File.ReadAllText(usagePath).Trim());
            long cache = 0;
            foreach (string line in File.ReadLines(statPath))
            {
                string[] kv = line.Split(' ', 2);
                if (kv.Length == 2 && (kv[0] == activeKey || kv[0] == inactiveKey))
                    cache += long.Parse(kv[1]);
            }
            available = Math.Max(1, limit - usage + cache);
            return true;
        }
    }
}
