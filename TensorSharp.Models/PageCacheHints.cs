// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Models
{
    /// <summary>
    /// Linux page-cache hints for the big files a direct-CUDA load reads: drop a byte range of a file from the page
    /// cache once it has been copied out (posix_fadvise DONTNEED), tell the kernel a mapping is read at random
    /// (posix_madvise RANDOM, so a fault reads one page instead of a read-around window), and ask which pages of a
    /// mapping are resident (mincore). Every call is a hint: it never reads or changes data, and on other platforms
    /// it reports <see cref="AdviceStatus.Unsupported"/> and does nothing.
    /// </summary>
    internal static unsafe class PageCacheHints
    {
        private const int PosixFadvDontNeed = 4;
        private const int PosixMadvRandom = 1;
        private const long TmpfsMagic = 0x01021994;
        private const long RamfsMagic = unchecked((long)0x858458F6);

        public enum AdviceStatus { Applied, Empty, Unsupported, Failed }

        /// <summary>The page size the kernel tracks residency and advice in.</summary>
        public static int PageSize => Environment.SystemPageSize;

        /// <summary>
        /// Drops the whole pages of <paramref name="bytes"/> bytes of the file at <paramref name="offset"/> from the page
        /// cache (clean pages no process maps; a pread reader maps none). Returns 0 (also when no whole page lies inside),
        /// the error number, or -1 where the platform has no such call.
        /// </summary>
        public static int DropFileRange(SafeFileHandle handle, long offset, long bytes) => DropFileRange(handle, offset, bytes, out _);

        /// <summary><see cref="DropFileRange(SafeFileHandle, long, long)"/>, also returning the bytes of the whole pages it
        /// asked the kernel to drop.</summary>
        public static int DropFileRange(SafeFileHandle handle, long offset, long bytes, out long wholePageBytes)
        {
            wholePageBytes = 0;
            if (!OperatingSystem.IsLinux())
                return -1;
            // Whole pages inside the range only (the kernel rounds the same way): a neighbouring tensor's shared page,
            // or a mapped table's, stays cached.
            long page = PageSize;
            long start = (offset + page - 1) / page * page, end = (offset + bytes) / page * page;
            if (bytes <= 0 || offset < 0 || end <= start)
                return 0;
            bool added = false;
            try
            {
                handle.DangerousAddRef(ref added);
                int rc = posix_fadvise((int)handle.DangerousGetHandle(), start, end - start, PosixFadvDontNeed);
                if (rc == 0)
                    wholePageBytes = end - start;
                return rc;
            }
            finally
            {
                if (added)
                    handle.DangerousRelease();
            }
        }

        /// <summary>
        /// Advises the kernel that <paramref name="bytes"/> bytes of a file mapping starting at
        /// <paramref name="data"/> are read at random. The start is rounded down to its page, which lies inside the
        /// mapping because mappings start on a page; the end stays at the range's end (the kernel covers the last
        /// page). The same range rule as the native loader's dsv41_engram_advice.h.
        /// </summary>
        public static AdviceStatus AdviseRandom(IntPtr data, long bytes, out int error)
        {
            error = 0;
            if (!OperatingSystem.IsLinux())
                return AdviceStatus.Unsupported;
            if (data == IntPtr.Zero || bytes <= 0)
                return AdviceStatus.Empty;
            long page = PageSize;
            long address = data.ToInt64();
            long aligned = address / page * page;
            error = posix_madvise((void*)aligned, (nuint)(address - aligned + bytes), PosixMadvRandom);
            return error == 0 ? AdviceStatus.Applied : AdviceStatus.Failed;
        }

        /// <summary>
        /// The pages of a mapped range that are not resident, as byte runs relative to <paramref name="data"/>
        /// (clamped to [0, <paramref name="bytes"/>), adjacent pages merged, runs closer than
        /// <paramref name="mergeGap"/> bytes joined). False when residency cannot be asked here (not Linux, or the
        /// call failed): the caller then treats the whole range as missing.
        /// </summary>
        public static bool TryMissingRuns(IntPtr data, long bytes, long mergeGap, List<(long Offset, long Bytes)> runs)
        {
            runs.Clear();
            if (!OperatingSystem.IsLinux() || data == IntPtr.Zero)
                return false;
            if (bytes <= 0)
                return true;
            long page = PageSize;
            long address = data.ToInt64();
            long aligned = address / page * page;
            long lead = address - aligned;
            long span = lead + bytes;
            long pages = (span + page - 1) / page;
            byte[] vector = new byte[pages];
            int rc;
            fixed (byte* v = vector)
                rc = mincore((void*)aligned, (nuint)span, v);
            if (rc != 0)
                return false;
            long runStart = -1, runEnd = -1;
            for (long i = 0; i < pages; i++)
            {
                if ((vector[i] & 1) != 0)
                    continue;
                long from = Math.Max(0, i * page - lead);
                long to = Math.Min(bytes, (i + 1) * page - lead);
                if (runStart >= 0 && from - runEnd <= mergeGap)
                {
                    runEnd = to;
                    continue;
                }
                if (runStart >= 0)
                    runs.Add((runStart, runEnd - runStart));
                runStart = from;
                runEnd = to;
            }
            if (runStart >= 0)
                runs.Add((runStart, runEnd - runStart));
            return true;
        }

        /// <summary>
        /// Whether mincore tells this process the truth about a file's page cache. Since Linux 5.2 it reports the page
        /// cache of a file mapping only to a process that owns the file, may write it, or holds CAP_FOWNER; to any other
        /// it reports every page resident, which would make a re-warm skip a cold table. Owner is not checked here (no
        /// portable stat), so a non-root reader of a read-only file it owns is distrusted too: it reads everything.
        /// </summary>
        public static bool ResidencyVisible(string path)
        {
            if (!OperatingSystem.IsLinux())
                return false;
            try
            {
                return geteuid() == 0 || access(path, AccessWrite) == 0;
            }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException)
            {
                return false;
            }
        }

        private const int AccessWrite = 2;

        /// <summary>Resident bytes of a mapped range (whole pages, clamped to the range), or -1 when residency cannot be
        /// asked here.</summary>
        public static long ResidentBytes(IntPtr data, long bytes)
        {
            var runs = new List<(long, long)>();
            if (!TryMissingRuns(data, bytes, 0, runs))
                return -1;
            long missing = 0;
            foreach (var (_, length) in runs)
                missing += length;
            return bytes - missing;
        }

        /// <summary>Whether <paramref name="path"/> lives on tmpfs or ramfs, where the page cache is the file's only
        /// copy: dropping it would not free anything (the kernel keeps such pages), and nothing should try.</summary>
        public static bool IsMemoryBacked(string path)
        {
            if (!OperatingSystem.IsLinux())
                return false;
            // struct statfs is 120 bytes on 64-bit glibc and starts with f_type; the buffer leaves room.
            byte* buffer = stackalloc byte[512];
            if (statfs(path, buffer) != 0)
                return false;
            long type = *(long*)buffer;
            return type == TmpfsMagic || type == RamfsMagic;
        }

        [DllImport("libc", EntryPoint = "posix_fadvise", SetLastError = false)]
        private static extern int posix_fadvise(int fd, long offset, long length, int advice);

        [DllImport("libc", EntryPoint = "posix_madvise", SetLastError = false)]
        private static extern int posix_madvise(void* address, nuint length, int advice);

        [DllImport("libc", EntryPoint = "mincore", SetLastError = true)]
        private static extern int mincore(void* address, nuint length, byte* vector);

        [DllImport("libc", EntryPoint = "statfs", SetLastError = true, CharSet = CharSet.Ansi, BestFitMapping = false)]
        private static extern int statfs(string path, byte* buffer);

        [DllImport("libc", EntryPoint = "geteuid")]
        private static extern uint geteuid();

        [DllImport("libc", EntryPoint = "access", SetLastError = true, CharSet = CharSet.Ansi, BestFitMapping = false)]
        private static extern int access(string path, int mode);
    }

    /// <summary>
    /// <c>TS_DSV4_LOAD_DROP_CACHE</c> for the direct-CUDA DeepSeek loader, with the native loader's meaning
    /// (dsv4_file_warm.h decide_drop_cache): whether each weight chunk's page cache is dropped once the loader has read
    /// it. Unset is automatic: drop exactly when the bytes this node uploads, plus the host-mapped tables it keeps (and
    /// warms), plus 8 GiB, exceed the host allowance (the cgroup limit, else MemTotal), and keep the cache whenever the
    /// allowance is unknown, so a reload of a checkpoint that fits stays warm. A set value drops when it parses to a
    /// non-zero number (as atoi reads it), so <c>0</c> never drops and <c>1</c> always drops.
    /// </summary>
    internal static class LoadPageCachePolicy
    {
        public const string Variable = "TS_DSV4_LOAD_DROP_CACHE";
        public const long DefaultHeadroom = 8L << 30;

        public readonly record struct Decision(bool Drop, bool Automatic, bool AllowanceKnown);

        public static Decision Decide(string value, long uploadBytes, long mappedBytes, long allowance, long headroom = DefaultHeadroom)
        {
            bool known = allowance > 0;
            if (!string.IsNullOrEmpty(value))
                return new Decision(Atoi(value) != 0, false, known);
            return new Decision(known && uploadBytes + mappedBytes + headroom > allowance, true, known);
        }

        public static string Describe(Decision d, long uploadBytes, long mappedBytes, long allowance, long headroom = DefaultHeadroom)
        {
            static string G(long b) => (b / (double)(1L << 30)).ToString("F1", CultureInfo.InvariantCulture);
            if (!d.Automatic)
                return $"{(d.Drop ? "dropping" : "keeping")} each uploaded chunk's page cache ({Variable} set; {G(uploadBytes)} GiB upload + " +
                       $"{G(mappedBytes)} GiB host-mapped + {G(headroom)} GiB headroom against a {G(allowance)} GiB allowance)";
            if (!d.AllowanceKnown)
                return $"keeping each uploaded chunk's page cache (automatic: host memory allowance unknown; {G(uploadBytes)} GiB upload + " +
                       $"{G(mappedBytes)} GiB host-mapped; {Variable}=1 drops)";
            return $"{(d.Drop ? "dropping" : "keeping")} each uploaded chunk's page cache (automatic: {G(uploadBytes)} GiB upload + " +
                   $"{G(mappedBytes)} GiB host-mapped + {G(headroom)} GiB headroom {(d.Drop ? "exceeds" : "fits")} the {G(allowance)} GiB " +
                   $"allowance; {Variable}={(d.Drop ? "0" : "1")} overrides)";
        }

        /// <summary>The host allowance as the native loader reads it: this process's cgroup limit, else MemTotal, else 0
        /// (unknown).</summary>
        public static long Allowance(CgroupMemory cgroup = null)
        {
            cgroup ??= CgroupMemory.Read();
            if (cgroup is { Limited: true })
                return cgroup.Limit;
            try
            {
                foreach (string line in File.ReadLines("/proc/meminfo"))
                {
                    if (!line.StartsWith("MemTotal:", StringComparison.Ordinal))
                        continue;
                    string[] parts = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
                    return long.Parse(parts[1], CultureInfo.InvariantCulture) * 1024;
                }
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or FormatException) { }
            return 0;
        }

        /// <summary>As C atoi reads it: optional whitespace and sign, then digits; anything else stops it (0 without a
        /// digit).</summary>
        private static long Atoi(string value)
        {
            int i = 0;
            while (i < value.Length && char.IsWhiteSpace(value[i]))
                i++;
            bool negative = i < value.Length && value[i] == '-';
            if (i < value.Length && (value[i] == '-' || value[i] == '+'))
                i++;
            long v = 0;
            while (i < value.Length && value[i] >= '0' && value[i] <= '9' && v < int.MaxValue)
                v = v * 10 + (value[i++] - '0');
            return negative ? -v : v;
        }
    }

    /// <summary>
    /// This process's memory cgroup as the kernel charges it: the limit and what in it cannot be reclaimed (anonymous
    /// memory and shmem/tmpfs pages). Page cache is charged too, but the kernel evicts it before it fails anything,
    /// so a table kept warm in the page cache has room for exactly the limit minus the unreclaimable part.
    /// </summary>
    internal sealed record CgroupMemory(int Version, string Directory, long Limit, long Usage, long Anon, long Shmem, long FileCache)
    {
        /// <summary>A limit at or above this is "no limit" (v1 reports 0x7FFFFFFFFFFFF000 for one).</summary>
        public const long Unlimited = 1L << 60;

        public bool Limited => Limit > 0 && Limit < Unlimited;

        /// <summary>
        /// Reads the cgroup of this process: /proc/self/cgroup names it, under the v2 unified mount or the v1 memory
        /// controller's mount. Inside a container the named path may not exist (the namespace roots it), and then the
        /// mount's own root is the cgroup. Null when there is no memory cgroup to read (not Linux, no files).
        /// </summary>
        public static CgroupMemory Read(string mountRoot = "/sys/fs/cgroup", string procSelfCgroup = "/proc/self/cgroup")
        {
            string[] lines;
            try
            {
                lines = File.ReadAllLines(procSelfCgroup);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return null;
            }
            string v1 = null, v2 = null;
            foreach (string line in lines)
            {
                string[] parts = line.Split(':', 3);
                if (parts.Length != 3)
                    continue;
                if (parts[0] == "0" && parts[1].Length == 0)
                    v2 = parts[2];
                else if (Array.IndexOf(parts[1].Split(','), "memory") >= 0)
                    v1 = parts[2];
            }
            // A v1 memory controller wins: on a hybrid host the unified tree carries no memory files for it.
            if (v1 != null)
            {
                string mount = Path.Combine(mountRoot, "memory");
                foreach (string dir in Candidates(mount, v1))
                {
                    long limit = ReadLong(Path.Combine(dir, "memory.limit_in_bytes"));
                    if (limit < 0)
                        continue;
                    var stat = ReadStat(Path.Combine(dir, "memory.stat"));
                    long anon = Stat(stat, "total_rss", "rss");
                    long shmem = Stat(stat, "total_shmem", "shmem");
                    long file = Stat(stat, "total_cache", "cache");
                    return new CgroupMemory(1, dir, limit, ReadLong(Path.Combine(dir, "memory.usage_in_bytes")),
                        anon, Math.Max(0, shmem), file);
                }
            }
            if (v2 != null)
            {
                foreach (string dir in Candidates(mountRoot, v2))
                {
                    string max = ReadText(Path.Combine(dir, "memory.max"));
                    if (max == null)
                        continue;
                    long limit = max == "max" ? long.MaxValue : ParseLong(max);
                    if (limit < 0)
                        continue;
                    var stat = ReadStat(Path.Combine(dir, "memory.stat"));
                    return new CgroupMemory(2, dir, limit, ReadLong(Path.Combine(dir, "memory.current")),
                        Stat(stat, "anon"), Math.Max(0, Stat(stat, "shmem")), Stat(stat, "file"));
                }
            }
            return null;
        }

        /// <summary>Bytes the page cache can keep for tables here: the limit minus the unreclaimable memory and
        /// <paramref name="headroom"/>; <see cref="long.MaxValue"/> without a limit.</summary>
        public long CacheBudget(long headroom)
            => Limited ? Limit - Math.Max(0, Anon) - Math.Max(0, Shmem) - headroom : long.MaxValue;

        public string Describe()
        {
            static string G(long b) => b < 0 ? "?" : (b / (double)(1L << 30)).ToString("F1", CultureInfo.InvariantCulture) + " GiB";
            return $"cgroup v{Version} {Directory}: limit {(Limited ? G(Limit) : "none")}, usage {G(Usage)} " +
                   $"(anon {G(Anon)}, shmem {G(Shmem)}, file {G(FileCache)})";
        }

        private static IEnumerable<string> Candidates(string mount, string relative)
        {
            string trimmed = relative.Trim().TrimStart('/');
            if (trimmed.Length > 0)
                yield return Path.Combine(mount, trimmed);
            yield return mount;
        }

        private static string ReadText(string path)
        {
            try
            {
                return File.Exists(path) ? File.ReadAllText(path).Trim() : null;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return null;
            }
        }

        private static long ReadLong(string path) => ReadText(path) is { } text ? ParseLong(text) : -1;

        private static long ParseLong(string text)
            => long.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out long v) ? v
             : ulong.TryParse(text, NumberStyles.None, CultureInfo.InvariantCulture, out _) ? long.MaxValue : -1;

        private static Dictionary<string, long> ReadStat(string path)
        {
            var stat = new Dictionary<string, long>(StringComparer.Ordinal);
            string text = ReadText(path);
            if (text == null)
                return stat;
            foreach (string line in text.Split('\n'))
            {
                string[] kv = line.Split(' ', 2, StringSplitOptions.RemoveEmptyEntries);
                if (kv.Length == 2)
                    stat[kv[0]] = ParseLong(kv[1].Trim());
            }
            return stat;
        }

        private static long Stat(Dictionary<string, long> stat, params string[] names)
        {
            foreach (string name in names)
            {
                if (stat.TryGetValue(name, out long v))
                    return v;
            }
            return -1;
        }
    }
}
