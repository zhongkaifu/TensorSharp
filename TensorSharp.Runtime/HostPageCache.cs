// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;

namespace TensorSharp.Runtime;

/// <summary>A byte range of a file that a "cold" measurement drops from the page cache: a host-mapped
/// lookup table such as DeepSeek V4.1's Engram tables.</summary>
public readonly record struct HostFileRange(string Name, string Path, long Offset, long Bytes);

/// <summary>How <see cref="HostPageCache.Evict"/> measures page-cache residency around an eviction.</summary>
public enum HostPageCacheMeasure
{
    /// <summary>No residency measurement.</summary>
    None = 0,
    /// <summary>mincore over at most <see cref="HostPageCache.SampleWindows"/> evenly spaced 64 MiB windows per
    /// range, scaled to the whole range. A few hundred milliseconds for a 126 GiB table.</summary>
    Sampled = 1,
    /// <summary>mincore over every page of every range.</summary>
    Full = 2,
}

/// <summary>What happened to one range.</summary>
public sealed class HostPageCacheRangeReport
{
    public string Name { get; init; } = string.Empty;
    public string Path { get; init; } = string.Empty;
    public long Offset { get; init; }
    public long Bytes { get; init; }
    /// <summary>Mappings of this file in this process that overlap the range.</summary>
    public int Mappings { get; set; }
    /// <summary>Overlapping mappings whose page-table entries were dropped (madvise MADV_DONTNEED).</summary>
    public int MappingsAdvised { get; set; }
    /// <summary>Overlapping writable mappings left alone: dropping a private writable page would lose data.</summary>
    public int MappingsSkipped { get; set; }
    /// <summary>Bytes of address space the madvise calls covered.</summary>
    public long AdvisedBytes { get; set; }
    /// <summary>Page-cache bytes of the range before / after (an estimate when sampled); -1 when not measured.</summary>
    public long ResidentBefore { get; set; } = -1;
    public long ResidentAfter { get; set; } = -1;
    /// <summary>Share of the range's pages that mincore actually inspected (1 for a full measurement).</summary>
    public double SampledFraction { get; set; }
    /// <summary>The first failure (errno text), or null.</summary>
    public string? Error { get; set; }
}

/// <summary>The outcome of one <see cref="HostPageCache.Evict"/> call.</summary>
public sealed class HostPageCacheReport
{
    public bool Supported { get; init; }
    /// <summary>Why nothing was done, when <see cref="Supported"/> is false.</summary>
    public string? Reason { get; init; }
    public HostPageCacheMeasure Measure { get; init; }
    public IReadOnlyList<HostPageCacheRangeReport> Ranges { get; init; } = Array.Empty<HostPageCacheRangeReport>();
    public double MeasureBeforeMs { get; set; }
    public double EvictMs { get; set; }
    public double MeasureAfterMs { get; set; }
    public double ElapsedMs => MeasureBeforeMs + EvictMs + MeasureAfterMs;

    public long Bytes { get { long b = 0; foreach (var r in Ranges) b += r.Bytes; return b; } }
    public int Mappings { get { int n = 0; foreach (var r in Ranges) n += r.Mappings; return n; } }
    public int MappingsAdvised { get { int n = 0; foreach (var r in Ranges) n += r.MappingsAdvised; return n; } }
    public int MappingsSkipped { get { int n = 0; foreach (var r in Ranges) n += r.MappingsSkipped; return n; } }
    public int Errors { get { int n = 0; foreach (var r in Ranges) n += r.Error != null ? 1 : 0; return n; } }
    public long ResidentBefore => Sum(r => r.ResidentBefore);
    public long ResidentAfter => Sum(r => r.ResidentAfter);

    /// <summary>True when the measurement shows the ranges left the page cache (at most 0.1% of their bytes still
    /// resident: the widened range should leave nothing), false when it shows they did not, null when nothing was
    /// measured. Errors are
    /// counted separately (<see cref="Errors"/>). A process that
    /// may not query a file's residency is told every page is resident (mincore's rule), so it reads false here:
    /// an unverifiable eviction never passes as a cold one.</summary>
    public bool? Verified => !Supported || Measure == HostPageCacheMeasure.None || Ranges.Count == 0 || ResidentAfter < 0
        ? null
        : ResidentAfter <= Bytes / 1000;

    private long Sum(Func<HostPageCacheRangeReport, long> field)
    {
        long total = 0;
        foreach (var r in Ranges)
        {
            long v = field(r);
            if (v < 0) return -1;
            total += v;
        }
        return total;
    }

    /// <summary>One log line: what was evicted and the residency before and after.</summary>
    public string Describe()
    {
        var c = CultureInfo.InvariantCulture;
        if (!Supported)
            return $"supported=false reason=\"{Reason}\"";
        static string Mib(long v) => v < 0 ? "na" : (v / 1048576.0).ToString("F1", CultureInfo.InvariantCulture);
        string verified = Verified is bool v ? (v ? "true" : "false") : "na";
        return string.Format(c,
            "supported=true ranges={0} bytesMiB={1} mappings={2} advised={3} skipped={4} errors={5} measure={6} " +
            "residentBeforeMiB={7} residentAfterMiB={8} verified={9} measureMs={10:F0} evictMs={11:F0}",
            Ranges.Count, Mib(Bytes), Mappings, MappingsAdvised, MappingsSkipped, Errors, Measure.ToString().ToLowerInvariant(),
            Mib(ResidentBefore), Mib(ResidentAfter), verified, MeasureBeforeMs + MeasureAfterMs, EvictMs);
    }
}

/// <summary>
/// Drops a file range from the page cache while this process keeps it memory-mapped, so a benchmark can measure a
/// cold host-mapped table without unloading the model.
///
/// <para>Linux only. <c>posix_fadvise(POSIX_FADV_DONTNEED)</c> alone does not work here: the kernel skips every page
/// that any process still maps. <c>madvise(MADV_DONTNEED)</c> alone does not either: on a file mapping it only drops
/// this process's page-table entries and the pages stay cached. So for every mapping of the file in this process
/// (found in <c>/proc/self/maps</c>, whoever created it: a .NET <c>MemoryMappedFile</c>, or the native ggml loader's
/// <c>mmap</c>) the part that overlaps the range gets <c>MADV_DONTNEED</c>, and then the whole range gets
/// <c>POSIX_FADV_DONTNEED</c>. The mappings stay valid; the next read faults the page back in from the file, so the
/// bytes a reader sees never change. Writable mappings are left alone (dropping a private written page would lose its
/// data), and their pages then stay cached, which the residency measurement reports.</para>
///
/// <para>The range is widened to 2 MiB boundaries (the largest page-cache folio: the kernel drops a folio only
/// when the range covers all of it), so up to 2 MiB on each side that hold a neighbouring tensor are dropped too.
/// Pages other processes map stay cached (the kernel never drops a mapped page), and so does anything a FUSE daemon
/// caches in its own memory; both hold for a whole-file <c>posix_fadvise</c> between processes as well, which is what
/// an out-of-process eviction such as <c>evict.py</c> does. An mlock()ed mapping cannot be dropped at all (madvise
/// fails with EINVAL); the report carries the error and the measurement shows the pages still cached.</para>
/// </summary>
public static unsafe partial class HostPageCache
{
    /// <summary>Most mincore windows a sampled measurement inspects per range.</summary>
    public const int SampleWindows = 64;

    private const long Window = 64L << 20;
    /// <summary>The eviction's alignment: the largest page-cache folio (PMD order, 2 MiB on x86-64 with 4 KiB pages).</summary>
    internal const long EvictAlignment = 2L << 20;
    private const int MADV_DONTNEED = 4;
    private const int POSIX_FADV_DONTNEED = 4;
    private const int PROT_READ = 1;
    private const int MAP_SHARED = 1;
    private const int AT_EMPTY_PATH = 0x1000;
    private const uint STATX_INO = 0x100;

    public static bool IsSupported => OperatingSystem.IsLinux();

    /// <summary>The reason a non-Linux host reports.</summary>
    public const string UnsupportedReason =
        "dropping a mapped file range from the page cache needs Linux (madvise + posix_fadvise); this OS has no equivalent";

    /// <summary>Drop <paramref name="ranges"/> from the page cache, measuring residency before and after as
    /// <paramref name="measure"/> says. Never throws for a range it cannot evict: the range's report carries the
    /// error and the eviction counts as unverified.</summary>
    public static HostPageCacheReport Evict(IReadOnlyList<HostFileRange> ranges, HostPageCacheMeasure measure)
    {
        ArgumentNullException.ThrowIfNull(ranges);
        if (!IsSupported)
            return new HostPageCacheReport { Supported = false, Reason = UnsupportedReason, Measure = measure };

        var reports = new List<HostPageCacheRangeReport>(ranges.Count);
        foreach (var r in ranges)
        {
            if (r.Bytes < 0 || r.Offset < 0 || string.IsNullOrEmpty(r.Path))
                throw new ArgumentException($"Invalid page-cache range {r.Name}: {r.Path} @{r.Offset} +{r.Bytes}.", nameof(ranges));
            reports.Add(new HostPageCacheRangeReport { Name = r.Name, Path = r.Path, Offset = r.Offset, Bytes = r.Bytes });
        }
        var report = new HostPageCacheReport { Supported = true, Measure = measure, Ranges = reports };

        lock (Serial)
        {
            var sw = Stopwatch.StartNew();
            if (measure != HostPageCacheMeasure.None)
                foreach (var r in reports)
                    MeasureInto(r, measure, before: true);
            report.MeasureBeforeMs = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            string maps;
            try { maps = File.ReadAllText("/proc/self/maps"); }
            catch (IOException ex) { maps = string.Empty; foreach (var r in reports) r.Error ??= "/proc/self/maps: " + ex.Message; }
            List<MapsEntry> entries = ParseMaps(maps);
            foreach (var r in reports)
                EvictOne(r, entries);
            report.EvictMs = sw.Elapsed.TotalMilliseconds;

            sw.Restart();
            if (measure != HostPageCacheMeasure.None)
                foreach (var r in reports)
                    MeasureInto(r, measure, before: false);
            report.MeasureAfterMs = sw.Elapsed.TotalMilliseconds;
        }
        return report;
    }

    /// <summary>
    /// One eviction or residency measurement at a time in this process. Two at once interfere, as when two pipeline
    /// nodes hosted in one process (the in-process multi-node tests) evict the same table together: each finds the
    /// other's short-lived measuring mapping in /proc/self/maps and madvises it after it is gone (ENOMEM, or, had the
    /// address range been reused meanwhile, someone else's pages), and one's mincore holds folios the other's
    /// posix_fadvise is dropping, so they stay cached and the eviction reads as unverified. A node that is alone in
    /// its process, as on separate hosts, never waits here.
    /// </summary>
    private static readonly object Serial = new();

    /// <summary>Page-cache bytes of a file range (an estimate when sampled), and the share of pages inspected.
    /// -1 on a host without mincore or when the file cannot be mapped.</summary>
    public static long ResidentBytes(string path, long offset, long bytes, HostPageCacheMeasure measure, out double sampledFraction)
    {
        sampledFraction = 0;
        if (!IsSupported || measure == HostPageCacheMeasure.None)
            return -1;
        return ResidentBytes(path, offset, bytes, measure, Window, SampleWindows, out sampledFraction);
    }

    /// <summary><see cref="ResidentBytes(string, long, long, HostPageCacheMeasure, out double)"/> with the window size
    /// and the sampled window count spelled out (tests use small ones).</summary>
    internal static long ResidentBytes(string path, long offset, long bytes, HostPageCacheMeasure measure, long windowBytes,
        int sampleWindows, out double sampledFraction)
    {
        sampledFraction = 0;
        if (!IsSupported || measure == HostPageCacheMeasure.None)
            return -1;
        using SafeFileHandle handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
        lock (Serial)   // its measuring mapping must not show up in a concurrent Evict's /proc/self/maps
            return Resident(handle, offset, bytes, measure, windowBytes, sampleWindows, out sampledFraction, out _);
    }

    private static void MeasureInto(HostPageCacheRangeReport r, HostPageCacheMeasure measure, bool before)
    {
        try
        {
            using SafeFileHandle handle = File.OpenHandle(r.Path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete);
            long resident = Resident(handle, r.Offset, r.Bytes, measure, Window, SampleWindows, out double fraction, out string? error);
            if (before) r.ResidentBefore = resident; else r.ResidentAfter = resident;
            r.SampledFraction = fraction;
            if (error != null) r.Error ??= error;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
            r.Error ??= "residency: " + ex.Message;
        }
    }

    private static void EvictOne(HostPageCacheRangeReport r, List<MapsEntry> maps)
    {
        if (r.Bytes == 0)
            return;
        // Widened to whole folios: the kernel drops a large folio only when the range covers all of it, and the page
        // cache keeps folios of up to 2 MiB (PMD order on x86-64), so an edge folio that also holds a neighbouring
        // tensor's bytes would stay cached (and mapped) otherwise. At most 2 MiB of neighbours go on each side.
        (long start, long length) = OutwardPages(r.Offset, r.Bytes, (int)Math.Max(EvictAlignment, Environment.SystemPageSize));
        SafeFileHandle handle;
        try { handle = File.OpenHandle(r.Path, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete); }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { r.Error ??= "open: " + ex.Message; return; }
        using (handle)
        {
            bool added = false;
            handle.DangerousAddRef(ref added);
            try
            {
                int fd = (int)handle.DangerousGetHandle();
                // A mapping names its file by device and inode, and by the path the kernel resolves; either
                // identifies it (a stacked filesystem such as overlayfs can show the inode of the file underneath).
                FileIdentity? id = Identify(fd);
                string? canonical = null;
                try { canonical = new FileInfo($"/proc/self/fd/{fd}").LinkTarget; }
                catch (IOException) { }
                foreach (MapsEntry e in maps)
                {
                    if (!IsSameFile(e, id, canonical) || !TryOverlap(e, start, length, out ulong address, out long bytes))
                        continue;
                    r.Mappings++;
                    if (e.Writable)
                    {
                        r.MappingsSkipped++;
                        continue;
                    }
                    if (Madvise((void*)address, (nuint)bytes, MADV_DONTNEED) != 0)
                    {
                        int errno = Marshal.GetLastPInvokeError();
                        // EINVAL on an aligned range: the mapping is mlock()ed (the pure C# DeepSeek executor pins its
                        // shards unless TS_DSV4_MLOCK=0), and a locked page cannot leave the page cache.
                        r.Error ??= $"madvise(MADV_DONTNEED) {address:x}+{bytes}: errno {errno}" +
                            (errno == 22 ? " (EINVAL: the mapping is probably mlock()ed; locked pages stay cached)" : "");
                        continue;
                    }
                    r.MappingsAdvised++;
                    r.AdvisedBytes += bytes;
                }
                int rc = Fadvise(fd, start, length, POSIX_FADV_DONTNEED);
                if (rc != 0)
                    r.Error ??= $"posix_fadvise(POSIX_FADV_DONTNEED): errno {rc}";
            }
            finally
            {
                if (added) handle.DangerousRelease();
            }
        }
    }

    private static long Resident(SafeFileHandle handle, long offset, long bytes, HostPageCacheMeasure measure,
        long windowBytes, int sampleWindows, out double sampledFraction, out string? error)
    {
        sampledFraction = 0;
        error = null;
        if (bytes <= 0)
        {
            sampledFraction = 1;
            return 0;
        }
        int page = Environment.SystemPageSize;
        (long start, long length) = OutwardPages(offset, bytes, page);
        long window = Math.Max(page, windowBytes - windowBytes % page);
        long windows = (length + window - 1) / window;
        long stride = measure == HostPageCacheMeasure.Full ? 1 : Math.Max(1, (windows + sampleWindows - 1) / sampleWindows);
        long totalPages = length / page, inspected = 0, resident = 0;
        var vector = new byte[window / page];
        bool added = false;
        handle.DangerousAddRef(ref added);
        try
        {
            int fd = (int)handle.DangerousGetHandle();
            for (long w = 0; w < windows; w += stride)
            {
                long at = start + w * window;
                long len = Math.Min(window, start + length - at);
                void* map = Mmap(null, (nuint)len, PROT_READ, MAP_SHARED, fd, at);
                if (map == (void*)-1)
                {
                    error = $"mmap for mincore: errno {Marshal.GetLastPInvokeError()}";
                    return -1;
                }
                try
                {
                    fixed (byte* v = vector)
                    {
                        if (Mincore(map, (nuint)len, v) != 0)
                        {
                            error = $"mincore: errno {Marshal.GetLastPInvokeError()}";
                            return -1;
                        }
                        long pages = len / page;
                        for (long i = 0; i < pages; i++)
                            resident += v[i] & 1;
                        inspected += pages;
                    }
                }
                finally
                {
                    Munmap(map, (nuint)len);
                }
            }
        }
        finally
        {
            if (added) handle.DangerousRelease();
        }
        sampledFraction = totalPages == 0 ? 1 : inspected / (double)totalPages;
        if (inspected == 0)
            return 0;
        // Pages, scaled to the whole range when sampled, then clipped to the range's own bytes.
        double estimate = resident * (double)totalPages / inspected * page;
        return Math.Min(bytes, (long)Math.Round(estimate));
    }

    // ------------------------------------------------------------------ pure helpers (tested on every OS)

    /// <summary>One line of <c>/proc/self/maps</c>.</summary>
    internal readonly record struct MapsEntry(ulong Start, ulong End, string Perms, long FileOffset, uint DevMajor,
        uint DevMinor, ulong Inode, string Path)
    {
        public bool Writable => Perms.Length > 1 && Perms[1] == 'w';
    }

    internal readonly record struct FileIdentity(uint DevMajor, uint DevMinor, ulong Inode);

    /// <summary>Parses <c>/proc/self/maps</c>; lines it cannot read are skipped (they are never file mappings this
    /// code would touch).</summary>
    internal static List<MapsEntry> ParseMaps(string text)
    {
        var entries = new List<MapsEntry>();
        foreach (string raw in text.Split('\n'))
        {
            string line = raw.TrimEnd('\r');
            // address perms offset dev inode [path]; the path may contain spaces.
            int a = line.IndexOf(' ');
            if (a <= 0) continue;
            int b = line.IndexOf(' ', a + 1);
            if (b <= a) continue;
            int c = line.IndexOf(' ', b + 1);
            if (c <= b) continue;
            int d = line.IndexOf(' ', c + 1);
            if (d <= c) continue;
            int e = line.IndexOf(' ', d + 1);
            string inodeText = e < 0 ? line[(d + 1)..] : line[(d + 1)..e];
            string path = e < 0 ? string.Empty : line[(e + 1)..].TrimStart(' ');
            string range = line[..a];
            int dash = range.IndexOf('-');
            string dev = line[(c + 1)..d];
            int colon = dev.IndexOf(':');
            if (dash <= 0 || colon <= 0
                || !ulong.TryParse(range[..dash], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong start)
                || !ulong.TryParse(range[(dash + 1)..], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out ulong end)
                || !long.TryParse(line[(b + 1)..c], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out long offset)
                || !uint.TryParse(dev[..colon], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out uint major)
                || !uint.TryParse(dev[(colon + 1)..], NumberStyles.HexNumber, CultureInfo.InvariantCulture, out uint minor)
                || !ulong.TryParse(inodeText, NumberStyles.None, CultureInfo.InvariantCulture, out ulong inode)
                || end <= start)
                continue;
            entries.Add(new MapsEntry(start, end, line[(a + 1)..b], offset, major, minor, inode, path));
        }
        return entries;
    }

    internal static bool IsSameFile(MapsEntry e, FileIdentity? id, string? canonicalPath)
        => e.Inode != 0 && ((id is FileIdentity f && e.DevMajor == f.DevMajor && e.DevMinor == f.DevMinor && e.Inode == f.Inode)
            || (canonicalPath != null && string.Equals(e.Path, canonicalPath, StringComparison.Ordinal)));

    /// <summary>The part of mapping <paramref name="e"/> that shows file bytes [<paramref name="start"/>,
    /// <paramref name="start"/> + <paramref name="length"/>): its address and length (page aligned whenever the
    /// range is, since a mapping's file offset is).</summary>
    internal static bool TryOverlap(MapsEntry e, long start, long length, out ulong address, out long bytes)
    {
        address = 0;
        bytes = 0;
        if (e.Inode == 0 || length <= 0)
            return false;
        long mapEnd = e.FileOffset + (long)(e.End - e.Start);
        long from = Math.Max(start, e.FileOffset), to = Math.Min(start + length, mapEnd);
        if (to <= from)
            return false;
        address = e.Start + (ulong)(from - e.FileOffset);
        bytes = to - from;
        return true;
    }

    /// <summary>[<paramref name="offset"/>, <paramref name="offset"/> + <paramref name="bytes"/>) widened to whole
    /// pages: fadvise rounds a partial page inward and would keep it.</summary>
    internal static (long Start, long Length) OutwardPages(long offset, long bytes, int page)
    {
        long start = offset - offset % page;
        long end = offset + bytes;
        long rem = end % page;
        if (rem != 0) end += page - rem;
        return (start, end - start);
    }

    /// <summary>statx's device and inode for an open file, or null where statx is unavailable.</summary>
    private static FileIdentity? Identify(int fd)
    {
        byte* buffer = stackalloc byte[256];
        byte* empty = stackalloc byte[1];
        empty[0] = 0;
        new Span<byte>(buffer, 256).Clear();
        try
        {
            if (Statx(fd, empty, AT_EMPTY_PATH, STATX_INO, buffer) != 0)
                return null;
        }
        catch (EntryPointNotFoundException) { return null; }
        // struct statx: stx_ino at 32, stx_dev_major at 136, stx_dev_minor at 140 (include/uapi/linux/stat.h).
        return new FileIdentity(*(uint*)(buffer + 136), *(uint*)(buffer + 140), *(ulong*)(buffer + 32));
    }

    [LibraryImport("libc", EntryPoint = "madvise", SetLastError = true)]
    private static partial int Madvise(void* address, nuint length, int advice);

    [LibraryImport("libc", EntryPoint = "posix_fadvise")]
    private static partial int Fadvise(int fd, long offset, long length, int advice);

    [LibraryImport("libc", EntryPoint = "mmap", SetLastError = true)]
    private static partial void* Mmap(void* address, nuint length, int prot, int flags, int fd, long offset);

    [LibraryImport("libc", EntryPoint = "munmap", SetLastError = true)]
    private static partial int Munmap(void* address, nuint length);

    [LibraryImport("libc", EntryPoint = "mincore", SetLastError = true)]
    private static partial int Mincore(void* address, nuint length, byte* vector);

    [LibraryImport("libc", EntryPoint = "statx", SetLastError = true)]
    private static partial int Statx(int dirfd, byte* path, int flags, uint mask, byte* buffer);
}
