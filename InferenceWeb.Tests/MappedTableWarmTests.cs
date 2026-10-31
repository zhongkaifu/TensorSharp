// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The host side of keeping DeepSeek V4.1's Engram tables warm (TensorSharp.Models/MappedTableWarm.cs and
// PageCacheHints.cs): the memory-cgroup budget a warm is held to, the re-warm that reads only the pages that are
// not resident, and the page-cache hints the loader and the Engram reader give the kernel.
//
// The cgroup parsing and the budget are plain file reads and arithmetic, tested everywhere on fixture
// directories. The page-cache tests need Linux (mincore, posix_fadvise, posix_madvise) and are marked
// LinuxFact, so on Windows they are reported as skipped, not passed; eng/validation/wsl-host-page-cache-tests.sh
// runs this file under WSL. This file uses only xunit and these types so that script can build it alone.
using System.IO.MemoryMappedFiles;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class MappedTableWarmTests : IDisposable
{
    private readonly List<string> _dirs = new();

    public void Dispose()
    {
        foreach (string dir in _dirs)
        {
            try { Directory.Delete(dir, recursive: true); } catch { /* best effort */ }
        }
    }

    /// <summary>A fresh directory on a disk-backed filesystem: where /tmp is tmpfs, its pages cannot be dropped, so
    /// the page-cache tests use the test's own directory instead.</summary>
    private string TempDir()
    {
        string parent = Path.GetTempPath();
        if (PageCacheHints.IsMemoryBacked(parent))
            parent = Path.Combine(AppContext.BaseDirectory, "ts-warm-tmp");
        string dir = Path.Combine(parent, $"ts-warm-{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        _dirs.Add(dir);
        return dir;
    }

    private static void Write(string root, string relative, string text)
    {
        string path = Path.Combine(root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
    }

    private const long GiB = 1L << 30;

    // ================================================================ cgroup parsing (fixture files)

    /// <summary>The VMs' layout: cgroup v1, the memory controller named in /proc/self/cgroup, rss and shmem from the
    /// hierarchical totals. 99,999,997,952 is the VMs' limit.</summary>
    [Fact]
    public void V1_ReadsTheLimitAndTheUnreclaimableMemory()
    {
        string root = TempDir();
        Write(root, "proc-self-cgroup", "12:pids:/docker/abc\n4:memory:/docker/abc\n0::/\n");
        Write(root, "memory/docker/abc/memory.limit_in_bytes", "99999997952\n");
        Write(root, "memory/docker/abc/memory.usage_in_bytes", "50000000000\n");
        Write(root, "memory/docker/abc/memory.stat", "cache 9\nrss 1\nshmem 2\ntotal_cache 40000000000\ntotal_rss 4000000000\n" +
            "total_shmem 1000000000\n");
        CgroupMemory cg = CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup"));
        Assert.NotNull(cg);
        Assert.Equal(1, cg!.Version);
        Assert.True(cg.Limited);
        Assert.Equal(99_999_997_952, cg.Limit);
        Assert.Equal(50_000_000_000, cg.Usage);
        Assert.Equal(4_000_000_000, cg.Anon);
        Assert.Equal(1_000_000_000, cg.Shmem);
        Assert.Equal(40_000_000_000, cg.FileCache);
        // Page cache is not subtracted: the kernel reclaims it before it fails anything.
        Assert.Equal(99_999_997_952 - 4_000_000_000 - 1_000_000_000 - 8 * GiB, cg.CacheBudget(8 * GiB));
        Assert.Contains("limit 93.1 GiB", cg.Describe());
    }

    /// <summary>Inside a container the path /proc/self/cgroup names may not exist under the mount (the namespace roots
    /// it): the mount's own files are the cgroup. Without the hierarchical totals, the local ones are read.</summary>
    [Fact]
    public void V1_InsideAContainer_ReadsTheMountRoot()
    {
        string root = TempDir();
        Write(root, "proc-self-cgroup", "4:memory:/kubepods/pod1/c1\n");
        Write(root, "memory/memory.limit_in_bytes", "1073741824\n");
        Write(root, "memory/memory.usage_in_bytes", "10\n");
        Write(root, "memory/memory.stat", "cache 5\nrss 7\n");
        CgroupMemory cg = CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup"))!;
        Assert.Equal(Path.Combine(root, "memory"), cg.Directory);
        Assert.Equal(GiB, cg.Limit);
        Assert.Equal(7, cg.Anon);
        Assert.Equal(0, cg.Shmem);
        Assert.Equal(5, cg.FileCache);
    }

    /// <summary>v1 reports "no limit" as a page-rounded 2^63: no limit, and no budget from the cgroup.</summary>
    [Fact]
    public void V1_WithoutALimit_IsUnlimited()
    {
        string root = TempDir();
        Write(root, "proc-self-cgroup", "4:memory:/\n");
        Write(root, "memory/memory.limit_in_bytes", "9223372036854771712\n");
        Write(root, "memory/memory.stat", "total_rss 1\n");
        CgroupMemory cg = CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup"))!;
        Assert.False(cg.Limited);
        Assert.Equal(long.MaxValue, cg.CacheBudget(8 * GiB));
        Assert.Contains("limit none", cg.Describe());
    }

    /// <summary>cgroup v2: memory.max as a number or "max", anon and shmem from memory.stat.</summary>
    [Theory]
    [InlineData("max", false)]
    [InlineData("68719476736", true)]
    public void V2_ReadsMemoryMax(string max, bool limited)
    {
        string root = TempDir();
        Write(root, "proc-self-cgroup", "0::/user.slice/job.scope\n");
        Write(root, "user.slice/job.scope/memory.max", max + "\n");
        Write(root, "user.slice/job.scope/memory.current", "123\n");
        Write(root, "user.slice/job.scope/memory.stat", "anon 2147483648\nfile 99\nkernel 5\nshmem 1073741824\n");
        CgroupMemory cg = CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup"))!;
        Assert.Equal(2, cg.Version);
        Assert.Equal(limited, cg.Limited);
        Assert.Equal(123, cg.Usage);
        Assert.Equal(2 * GiB, cg.Anon);
        Assert.Equal(GiB, cg.Shmem);
        Assert.Equal(99, cg.FileCache);
        Assert.Equal(limited ? 64 * GiB - 3 * GiB - 8 * GiB : long.MaxValue, cg.CacheBudget(8 * GiB));
    }

    /// <summary>A hybrid host lists both; the v1 memory controller is the one that charges.</summary>
    [Fact]
    public void Hybrid_PrefersTheV1MemoryController()
    {
        string root = TempDir();
        Write(root, "proc-self-cgroup", "0::/a\n9:cpu,memory:/b\n");
        Write(root, "a/memory.max", "1000\n");
        Write(root, "memory/b/memory.limit_in_bytes", "2000\n");
        CgroupMemory cg = CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup"))!;
        Assert.Equal(1, cg.Version);
        Assert.Equal(2000, cg.Limit);
        Assert.Equal(-1, cg.Anon);   // no memory.stat: unknown, and the budget treats it as 0
        Assert.Equal(2000 - 10, cg.CacheBudget(10));
    }

    /// <summary>No /proc/self/cgroup (not Linux), or no memory files for the named cgroup: nothing to read.</summary>
    [Fact]
    public void MissingFiles_ReadAsNoCgroup()
    {
        string root = TempDir();
        Assert.Null(CgroupMemory.Read(root, Path.Combine(root, "absent")));
        Write(root, "proc-self-cgroup", "4:memory:/x\n0::/y\n");
        Assert.Null(CgroupMemory.Read(root, Path.Combine(root, "proc-self-cgroup")));
    }

    // ================================================================ the warm budget

    private static MappedTableWarm.Options Budgeted(long limit, long anon, long hostAvailable) => new()
    {
        Cgroup = () => new CgroupMemory(1, "/fixture", limit, -1, anon, 0, 0),
        HostAvailable = () => hostAvailable,
    };

    /// <summary>
    /// The VMs' numbers: a 93.1 GiB cgroup and a host that reports far more free memory than the cgroup has. Both
    /// 1017 slices (125.9 GiB) do not fit - the cgroup says so although MemAvailable would have let them in, which is
    /// how S9's warm filled the cgroup - and one slice (62.95 GiB) does, with the numbers in the verdict.
    /// </summary>
    [Fact]
    public void TheBudget_IsTheCgroup_NotTheHost()
    {
        var options = Budgeted(99_999_997_952, 4 * GiB, hostAvailable: 400 * GiB);
        long slice = 67_588_000_000;   // 62.95 GiB
        MappedTableWarm.Budget both = MappedTableWarm.CheckBudget(2 * slice, options);
        Assert.False(both.Fits);
        Assert.Equal(99_999_997_952 - 4 * GiB - 8 * GiB, both.Available);
        Assert.Contains("cgroup v1 /fixture: limit 93.1 GiB", both.Description);
        Assert.Contains("room for tables 81.1 GiB after 8.0 GiB headroom", both.Description);
        Assert.Contains("host MemAvailable 400.0 GiB", both.Description);
        Assert.True(MappedTableWarm.CheckBudget(slice, options).Fits);
        // More anonymous memory in the cgroup leaves less room for the slice.
        Assert.False(MappedTableWarm.CheckBudget(slice, Budgeted(99_999_997_952, 24 * GiB, 400 * GiB)).Fits);
    }

    /// <summary>The host's MemAvailable still bounds a warm when the cgroup has no limit (or there is none).</summary>
    [Fact]
    public void TheHost_BoundsAWarm_WithoutACgroupLimit()
    {
        var unlimited = Budgeted(long.MaxValue, 0, hostAvailable: 70 * GiB);
        Assert.True(MappedTableWarm.CheckBudget(62 * GiB, unlimited).Fits);
        Assert.False(MappedTableWarm.CheckBudget(63 * GiB, unlimited).Fits);
        var none = new MappedTableWarm.Options { Cgroup = () => null, HostAvailable = () => 0 };
        MappedTableWarm.Budget open = MappedTableWarm.CheckBudget(1L << 50, none);
        Assert.True(open.Fits);
        Assert.Contains("no memory cgroup to read", open.Description);
    }

    /// <summary>A warm the budget refuses reads nothing and says why (Start returns no task).</summary>
    [Fact]
    public void ARefusedWarm_ReadsNothing()
    {
        string dir = TempDir();
        string path = Path.Combine(dir, "table.bin");
        File.WriteAllBytes(path, new byte[4096]);
        var options = Budgeted(16 * GiB, 4 * GiB, 400 * GiB);
        var ranges = new[] { new MappedTableWarm.Range(path, 0, 8 * GiB, IntPtr.Zero) };
        Assert.Null(MappedTableWarm.Start(ranges, "test", "tables", CancellationToken.None, options));
    }

    // ================================================================ reading (every platform)

    /// <summary>Without a mapping the warm reads every byte of every range once, in 64 MiB blocks, on its threads.</summary>
    [Fact]
    public void Run_WithoutAMapping_ReadsEverything()
    {
        string dir = TempDir();
        string path = Path.Combine(dir, "table.bin");
        File.WriteAllBytes(path, new byte[4 << 20]);
        var ranges = new[]
        {
            new MappedTableWarm.Range(path, 4096, 1 << 20, IntPtr.Zero),
            new MappedTableWarm.Range(path, 2 << 20, (1 << 20) + 123, IntPtr.Zero),
        };
        MappedTableWarm.Result r = MappedTableWarm.Run(ranges, threads: 4, skipResident: true, CancellationToken.None);
        Assert.Null(r.Error);
        Assert.False(r.Cancelled);
        Assert.Equal((2 << 20) + 123, r.Bytes);
        Assert.Equal(r.Bytes, r.Read);
        Assert.Equal(0, r.AlreadyResident);
        Assert.Equal(2, r.Blocks);
    }

    /// <summary>A range past the end of its file is a short read: reported, not thrown.</summary>
    [Fact]
    public void Run_AShortRead_IsReported()
    {
        string dir = TempDir();
        string path = Path.Combine(dir, "short.bin");
        File.WriteAllBytes(path, new byte[1000]);
        MappedTableWarm.Result r = MappedTableWarm.Run(new[] { new MappedTableWarm.Range(path, 0, 5000, IntPtr.Zero) },
            threads: 2, skipResident: false, CancellationToken.None);
        Assert.NotNull(r.Error);
        Assert.Contains("short read", r.Error);
    }

    // ================================================================ page cache (Linux)

    /// <summary>A file of <paramref name="pages"/> random pages, flushed to disk (dirty pages cannot be dropped).</summary>
    private string PageFile(int pages)
    {
        string path = Path.Combine(TempDir(), "pages.bin");
        var bytes = new byte[(long)pages * PageCacheHints.PageSize];
        new Random(pages).NextBytes(bytes);
        using (var stream = new FileStream(path, FileMode.CreateNew, FileAccess.Write))
        {
            stream.Write(bytes);
            stream.Flush(flushToDisk: true);
        }
        return path;
    }

    /// <summary>
    /// Leaves exactly <paramref name="pages"/> of the file in the page cache. The whole file is dropped first (a
    /// whole-file drop takes every folio), then each page is read alone with readahead off (POSIX_FADV_RANDOM), so each
    /// is a folio of its own: a kernel with large folios (ext4 on 6.16+) would otherwise cache a written or read-ahead
    /// file in folios of many pages, which a partial drop leaves whole.
    /// </summary>
    private static void Cache(string path, IEnumerable<int> pages)
    {
        long page = PageCacheHints.PageSize;
        long length = new FileInfo(path).Length;
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Assert.Equal(0, PageCacheHints.DropFileRange(handle, 0, length));
        Assert.Equal(0, posix_fadvise((int)handle.DangerousGetHandle(), 0, 0, PosixFadvRandom));
        var buffer = new byte[page];
        foreach (int p in pages)
            Assert.Equal((int)page, RandomAccess.Read(handle, buffer, p * page));
    }

    private const int PosixFadvRandom = 1;

    [System.Runtime.InteropServices.DllImport("libc", EntryPoint = "posix_fadvise")]
    private static extern int posix_fadvise(int fd, long offset, long length, int advice);

    private static void Drop(string path, long offset, long bytes)
    {
        using var handle = File.OpenHandle(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Assert.Equal(0, PageCacheHints.DropFileRange(handle, offset, bytes));
    }

    private static unsafe (MemoryMappedFile Map, MemoryMappedViewAccessor View, IntPtr Base) Map(string path)
    {
        var map = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        var view = map.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* b = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref b);
        return (map, view, (IntPtr)(b + view.PointerOffset));
    }

    /// <summary>
    /// The loader's drop (posix_fadvise DONTNEED after a read): whole pages inside the range leave the page cache, and a
    /// page the range only partly covers stays, so a neighbouring tensor (or a mapped table beside a weight) keeps its
    /// shared page. mincore on a mapping of the file sees it. (The kernel drops whole folios only; the pages here are
    /// cached one per folio, see <see cref="Cache"/>.)
    /// </summary>
    [LinuxFact]
    public void TheDrop_EvictsWholePagesOnly()
    {
        long page = PageCacheHints.PageSize;
        string path = PageFile(10);
        Cache(path, Enumerable.Range(0, 10));
        var (map, view, b) = Map(path);
        using (map)
        using (view)
        {
            Assert.Equal(10 * page, PageCacheHints.ResidentBytes(b, 10 * page));
            Drop(path, page / 2, 3 * page);   // [0.5, 3.5) pages: only pages 1 and 2 lie wholly inside
            var runs = new List<(long, long)>();
            Assert.True(PageCacheHints.TryMissingRuns(b, 10 * page, 0, runs));
            Assert.Equal(new[] { (page, 2 * page) }, runs);
            Drop(path, 0, 10 * page);
            Assert.Equal(0, PageCacheHints.ResidentBytes(b, 10 * page));
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>
    /// The re-warm reads only what is missing: with the first half of a table cached, a warm reads the second half and
    /// counts the first as resident; a second warm reads nothing and skips the block whole. The runs are page-exact
    /// and include a range that starts inside a page.
    /// </summary>
    [LinuxFact]
    public void TheRewarm_ReadsOnlyTheMissingPages()
    {
        long page = PageCacheHints.PageSize;
        string path = PageFile(64);
        var (map, view, b) = Map(path);
        using (map)
        using (view)
        {
            Cache(path, Enumerable.Range(0, 32));
            long offset = 100;   // the table starts inside page 0, as a tensor in a shard does
            long bytes = 64 * page - offset;
            var ranges = new[] { new MappedTableWarm.Range(path, offset, bytes, b + (nint)offset) };
            MappedTableWarm.Result first = MappedTableWarm.Run(ranges, threads: 3, skipResident: true, CancellationToken.None);
            Assert.Null(first.Error);
            Assert.Equal(32 * page, first.Read);
            Assert.Equal(bytes - 32 * page, first.AlreadyResident);
            Assert.Equal(64 * page, PageCacheHints.ResidentBytes(b, 64 * page));
            MappedTableWarm.Result second = MappedTableWarm.Run(ranges, threads: 3, skipResident: true, CancellationToken.None);
            Assert.Equal(0, second.Read);
            Assert.Equal(bytes, second.AlreadyResident);
            Assert.Equal(1, second.BlocksSkipped);
            // Without the residency check the same warm reads everything again.
            Assert.Equal(bytes, MappedTableWarm.Run(ranges, 3, skipResident: false, CancellationToken.None).Read);
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>Nearby missing pages are read as one run (a cached page costs a copy, a separate read a round trip).</summary>
    [LinuxFact]
    public void MissingRuns_MergeAcrossSmallGaps()
    {
        long page = PageCacheHints.PageSize;
        string path = PageFile(16);
        var (map, view, b) = Map(path);
        using (map)
        using (view)
        {
            Cache(path, Enumerable.Range(0, 16).Except(new[] { 2, 4, 12, 13 }));
            var runs = new List<(long, long)>();
            Assert.True(PageCacheHints.TryMissingRuns(b, 16 * page, 0, runs));
            Assert.Equal(new[] { (2 * page, page), (4 * page, page), (12 * page, 2 * page) }, runs);
            Assert.True(PageCacheHints.TryMissingRuns(b, 16 * page, page, runs));
            Assert.Equal(new[] { (2 * page, 3 * page), (12 * page, 2 * page) }, runs);
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>MADV_RANDOM on a file mapping is applied on Linux (from a pointer inside a page, as a table inside a shard
    /// starts) and reported unsupported elsewhere; it changes no byte the mapping reads.</summary>
    [Fact]
    public unsafe void RandomAdvice_IsAppliedOnLinuxOnly()
    {
        string path = PageFile(4);
        byte[] expected = File.ReadAllBytes(path);
        var (map, view, b) = Map(path);
        using (map)
        using (view)
        {
            var status = PageCacheHints.AdviseRandom(b + 100, 3 * PageCacheHints.PageSize, out int error);
            Assert.Equal(OperatingSystem.IsLinux() ? PageCacheHints.AdviceStatus.Applied : PageCacheHints.AdviceStatus.Unsupported, status);
            Assert.Equal(0, error);
            Assert.Equal(expected[100], *((byte*)b + 100));
            Assert.Equal(expected[^1], *((byte*)b + expected.Length - 1));
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }
    }

    /// <summary>tmpfs is memory-backed (dropping its pages frees nothing); an ordinary directory is not.</summary>
    [LinuxFact]
    public void Tmpfs_IsRecognised()
    {
        Assert.False(PageCacheHints.IsMemoryBacked(TempDir()));
        if (Directory.Exists("/dev/shm"))
            Assert.True(PageCacheHints.IsMemoryBacked("/dev/shm"));
    }
}

/// <summary>[Fact] that needs Linux: skips visibly elsewhere.</summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class LinuxFactAttribute : FactAttribute
{
    public LinuxFactAttribute()
    {
        if (!OperatingSystem.IsLinux())
            Skip = "needs Linux (page cache residency and advice); eng/validation/wsl-host-page-cache-tests.sh runs it under WSL";
    }
}
