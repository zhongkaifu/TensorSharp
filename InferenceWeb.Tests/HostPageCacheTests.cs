// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// TensorSharp.Runtime/HostPageCache.cs: the in-process page-cache eviction behind a cold --bench-workload row.
// The /proc/self/maps parsing and the range arithmetic are tested on every OS; the eviction itself runs on Linux
// (the CI runners) and must report "not supported" everywhere else. eng/validation/tests/host-page-cache checks the
// Linux behaviour in more depth (controls, two mappings, a writable mapping, sampling) on WSL or a Linux host.
using System.IO.MemoryMappedFiles;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class HostPageCacheTests : IDisposable
{
    // On the test output directory, not /tmp: a tmpfs /tmp keeps every page in memory.
    private readonly string _dir = Path.Combine(AppContext.BaseDirectory, "host-page-cache-" + Guid.NewGuid().ToString("N"));

    public HostPageCacheTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private const string Maps =
        "55d0c0a00000-55d0c0a21000 r--p 00000000 08:01 1835014                    /usr/bin/dotnet\n" +
        "7f3c00000000-7f3c40000000 r--s 00000000 00:31 9001                       /workspace/models/My Model-00006-of-00006.gguf\n" +
        "7f3c50000000-7f3c50200000 r--s 1f400000 00:31 9001                       /workspace/models/My Model-00006-of-00006.gguf\n" +
        "7f3c60000000-7f3c60001000 rw-p 00000000 00:00 0                          [heap]\n" +
        "7f3c70000000-7f3c70100000 rw-p 00002000 08:01 77                         /tmp/scratch (deleted)\n" +
        "7f3c80000000-7f3c80001000 ---p 00000000 00:00 0 \n" +
        "garbage line\n" +
        "7ffd00000000-7ffd00021000 rw-p 00000000 00:00 0                          [stack]";

    [Fact]
    public void ParseMaps_ReadsFileMappings_WithSpacesInPaths_AndSkipsGarbage()
    {
        var entries = HostPageCache.ParseMaps(Maps);
        Assert.Equal(7, entries.Count);
        var model = entries[1];
        Assert.Equal(0x7f3c00000000UL, model.Start);
        Assert.Equal(0x7f3c40000000UL, model.End);
        Assert.Equal("r--s", model.Perms);
        Assert.Equal(0, model.FileOffset);
        Assert.Equal((0u, 0x31u, 9001UL), (model.DevMajor, model.DevMinor, model.Inode));
        Assert.Equal("/workspace/models/My Model-00006-of-00006.gguf", model.Path);
        Assert.False(model.Writable);
        Assert.Equal(0x1f400000L, entries[2].FileOffset);
        Assert.True(entries[3].Writable);
        Assert.Equal("[heap]", entries[3].Path);
        Assert.Equal("/tmp/scratch (deleted)", entries[4].Path);
        Assert.Equal(string.Empty, entries[5].Path);
        Assert.Equal(0UL, entries[5].Inode);
    }

    [Fact]
    public void TryOverlap_MapsAFileRangeOntoEachMapping()
    {
        var entries = HostPageCache.ParseMaps(Maps);
        var whole = entries[1];      // file bytes [0, 1 GiB) at 0x7f3c00000000
        var window = entries[2];     // file bytes [0x1f400000, +2 MiB) at 0x7f3c50000000

        Assert.True(HostPageCache.TryOverlap(whole, 0x100000, 0x300000, out ulong a, out long n));
        Assert.Equal((0x7f3c00100000UL, 0x300000L), (a, n));
        // Clipped to the mapping's end.
        Assert.True(HostPageCache.TryOverlap(whole, 0x3ff00000, 0x400000, out a, out n));
        Assert.Equal((0x7f3c3ff00000UL, 0x100000L), (a, n));
        // A mapping that starts at a file offset: the range's address is relative to that offset.
        Assert.True(HostPageCache.TryOverlap(window, 0x1f300000, 0x200000, out a, out n));
        Assert.Equal((0x7f3c50000000UL, 0x100000L), (a, n));
        Assert.False(HostPageCache.TryOverlap(window, 0, 0x1f400000, out _, out _));
        Assert.False(HostPageCache.TryOverlap(window, 0x1f600000, 0x1000, out _, out _));
        // Anonymous mappings never match.
        Assert.False(HostPageCache.TryOverlap(entries[3], 0, 0x1000, out _, out _));
    }

    [Fact]
    public void IsSameFile_MatchesByDeviceAndInode_OrByPath_NeverAnAnonymousMapping()
    {
        var entries = HostPageCache.ParseMaps(Maps);
        var id = new HostPageCache.FileIdentity(0, 0x31, 9001);
        Assert.True(HostPageCache.IsSameFile(entries[1], id, null));
        Assert.False(HostPageCache.IsSameFile(entries[1], new HostPageCache.FileIdentity(0, 0x32, 9001), null));
        Assert.True(HostPageCache.IsSameFile(entries[1], null, "/workspace/models/My Model-00006-of-00006.gguf"));
        Assert.False(HostPageCache.IsSameFile(entries[1], null, "/workspace/models/other.gguf"));
        Assert.False(HostPageCache.IsSameFile(entries[5], new HostPageCache.FileIdentity(0, 0, 0), string.Empty));
    }

    [Theory]
    [InlineData(100, 50, 4096, 0, 4096)]
    [InlineData(4096, 4096, 4096, 4096, 4096)]
    [InlineData(4095, 2, 4096, 0, 8192)]
    [InlineData((5L << 20) + 100, (32L << 20) + 777, 2L << 20, 4L << 20, 34L << 20)]
    public void OutwardPages_WidensToWholeUnits(long offset, long bytes, int unit, long start, long length)
        => Assert.Equal((start, length), HostPageCache.OutwardPages(offset, bytes, unit));

    [Fact]
    public void Report_VerifiedNeedsAMeasurement_AndAtMostAThousandthLeft()
    {
        HostPageCacheReport Report(long bytes, long after, HostPageCacheMeasure measure = HostPageCacheMeasure.Full) => new()
        {
            Supported = true,
            Measure = measure,
            Ranges = new[] { new HostPageCacheRangeReport { Name = "t", Path = "p", Bytes = bytes, ResidentBefore = bytes, ResidentAfter = after } },
        };
        long gib = 1L << 30;
        Assert.True(Report(gib, 0).Verified);
        Assert.True(Report(gib, gib / 1000).Verified);
        Assert.False(Report(gib, gib / 1000 + 1).Verified);
        // No floor: a small table must leave the cache too (a locked mapping keeps it, and must not pass).
        Assert.False(Report(156288, 156288).Verified);
        Assert.True(Report(156288, 0).Verified);
        Assert.Null(Report(gib, -1).Verified);
        Assert.Null(Report(gib, 0, HostPageCacheMeasure.None).Verified);
        Assert.Null(new HostPageCacheReport { Supported = false, Measure = HostPageCacheMeasure.Full }.Verified);
        Assert.Contains("verified=true", Report(gib, 0).Describe());
    }

    /// <summary>Linux: a range this process maps leaves the page cache, the rest of the file stays, and the mapping
    /// still reads the same bytes. Anywhere else: unsupported, never "verified".</summary>
    [Fact]
    public unsafe void Evict_DropsAMappedRange_OnLinux_AndIsUnsupportedElsewhere()
    {
        string path = Path.Combine(_dir, "table.bin");
        const long FileBytes = 12L << 20;
        var data = new byte[FileBytes];
        new Random(7).NextBytes(data);
        File.WriteAllBytes(path, data);
        long tableOffset = (4L << 20) + 123, tableBytes = (3L << 20) + 45;
        var range = new HostFileRange("table", path, tableOffset, tableBytes);

        if (!OperatingSystem.IsLinux())
        {
            HostPageCacheReport unsupported = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
            Assert.False(unsupported.Supported);
            Assert.Equal(HostPageCache.UnsupportedReason, unsupported.Reason);
            Assert.Null(unsupported.Verified);
            Assert.False(HostPageCache.IsSupported);
            Assert.Equal(-1, HostPageCache.ResidentBytes(path, 0, FileBytes, HostPageCacheMeasure.Full, out _));
            return;
        }

        File.ReadAllBytes(path);
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* p = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref p);
        try
        {
            p += view.PointerOffset;
            long sum = 0;
            for (long i = tableOffset; i < tableOffset + tableBytes; i += 256)
                sum += p[i];

            HostPageCacheReport r = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
            Assert.True(r.Supported, r.Reason);
            Assert.Equal(0, r.Errors);
            Assert.True(r.MappingsAdvised >= 1, r.Describe());
            Assert.True(r.ResidentBefore >= tableBytes - Environment.SystemPageSize, r.Describe());
            Assert.Equal(0, r.ResidentAfter);
            Assert.True(r.Verified);
            // Outside the range widened to 2 MiB boundaries: still cached.
            Assert.Equal(4L << 20, HostPageCache.ResidentBytes(path, 0, 4L << 20, HostPageCacheMeasure.Full, out _));
            Assert.Equal(4L << 20, HostPageCache.ResidentBytes(path, 8L << 20, 4L << 20, HostPageCacheMeasure.Full, out _));

            long again = 0;
            for (long i = tableOffset; i < tableOffset + tableBytes; i += 256)
                again += p[i];
            Assert.Equal(sum, again);
        }
        finally
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>
    /// Linux: two evictions of one table at once in one process, as two pipeline nodes hosted in one process start
    /// them (the second up to 0.4 ms after the first, as a control op reaches a worker), take turns and both come out
    /// clean and verified. Without the turns each found the other's short-lived measuring mapping in /proc/self/maps
    /// and madvised it after it was gone: about 1% of such reports carried an ENOMEM error, and a few left pages
    /// cached (s10/integrate, 2,000 rounds on WSL).
    /// </summary>
    [LinuxFact]
    public void TwoEvictionsAtOnce_InOneProcess_TakeTurns_AndBothComeOutClean()
    {
        string path = Path.Combine(_dir, "shared.bin");
        var data = new byte[11L << 20];
        new Random(5).NextBytes(data);
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write))
        {
            stream.Write(data);
            stream.Flush(flushToDisk: true);   // posix_fadvise skips dirty pages
        }
        const long Offset = 5L << 20, Bytes = 156_288;   // the synthetic DeepSeek V4.1 fixture's Engram table
        var range = new[] { new HostFileRange("table", path, Offset, Bytes) };
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        // Four views, as the in-process two-node test maps the shard (each node's shard set and table view).
        var views = Enumerable.Range(0, 4).Select(_ => mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read)).ToArray();
        try
        {
            long sink = 0;
            var problems = new List<string>();
            for (int round = 0; round < 400; round++)
            {
                for (long o = Offset; o < Offset + Bytes; o += 4096)
                    sink += views[round % 2].ReadByte(o);
                using var barrier = new Barrier(2);
                long delay = Random.Shared.NextInt64(0, System.Diagnostics.Stopwatch.Frequency * 400 / 1_000_000);
                HostPageCacheReport[] reports = Task.WhenAll(Enumerable.Range(0, 2).Select(k => Task.Run(() =>
                {
                    barrier.SignalAndWait();
                    if (k == 1)
                    {
                        long until = System.Diagnostics.Stopwatch.GetTimestamp() + delay;
                        while (System.Diagnostics.Stopwatch.GetTimestamp() < until) { }
                    }
                    return HostPageCache.Evict(range, HostPageCacheMeasure.Sampled);
                }))).GetAwaiter().GetResult();
                foreach (HostPageCacheReport r in reports)
                    if (r.Errors != 0 || r.Verified != true)
                        problems.Add($"round {round}: {r.Describe()} {string.Join("; ", r.Ranges.Select(x => x.Error))}");
            }
            Assert.True(problems.Count == 0, $"{problems.Count} of 800 reports not clean (sink {sink % 7}):\n" +
                string.Join("\n", problems.Take(5)));
        }
        finally
        {
            foreach (var v in views)
                v.Dispose();
        }
    }
}
