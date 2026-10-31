// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Checks TensorSharp.Runtime/HostPageCache.cs on a real Linux kernel (WSL or a Linux host; no GPU, no model):
// a file range that this process keeps memory-mapped leaves the page cache, and only that range. Also shows the
// two controls the design rests on: posix_fadvise(DONTNEED) alone keeps pages the process maps, and
// madvise(MADV_DONTNEED) alone keeps them cached. Exit 0 = every check passed, 1 = a check failed, 2 = cannot run.
//
//   dotnet run -c Release --project eng/validation/tests/host-page-cache -- [scratch directory]
//
// The scratch directory must be on a disk filesystem (ext4, xfs, a FUSE mount, ...): tmpfs pages are memory and
// can never be dropped. The default is ~/.cache/ts-host-page-cache.
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;

unsafe
{
    if (!OperatingSystem.IsLinux())
    {
        Console.Error.WriteLine("This check needs Linux; on this OS HostPageCache reports: " + HostPageCache.UnsupportedReason);
        var unsupported = HostPageCache.Evict(new[] { new HostFileRange("x", "x", 0, 1) }, HostPageCacheMeasure.Full);
        Console.Error.WriteLine($"Evict -> Supported={unsupported.Supported} Verified={unsupported.Verified?.ToString() ?? "null"}");
        return 2;
    }

    string dir = args.Length > 0 ? args[0]
        : Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "ts-host-page-cache");
    Directory.CreateDirectory(dir);
    string fsType = FileSystemOf(dir);
    Console.WriteLine($"kernel {File.ReadAllText("/proc/sys/kernel/osrelease").Trim()}, scratch {dir} ({fsType}), " +
                      $"page {Environment.SystemPageSize} B, euid-owned files");
    if (fsType is "tmpfs" or "ramfs")
    {
        Console.Error.WriteLine("The scratch directory is on " + fsType + ": its pages cannot leave memory. Pass a disk directory.");
        return 2;
    }

    int page = Environment.SystemPageSize;
    const long FileBytes = (48L << 20) + 12345;
    long tableOffset = (5L << 20) + 100;           // not page aligned, like a GGUF tensor
    long tableBytes = (32L << 20) + 777;
    long tableEnd = tableOffset + tableBytes;
    long alStart = tableOffset - tableOffset % page;
    long alEnd = (tableEnd + page - 1) / page * page;
    // What Evict drops: the table widened to 2 MiB folio boundaries.
    long evStart = tableOffset - tableOffset % HostPageCache.EvictAlignment;
    long evEnd = (tableEnd + HostPageCache.EvictAlignment - 1) / HostPageCache.EvictAlignment * HostPageCache.EvictAlignment;
    string path = Path.Combine(dir, $"table-{Environment.ProcessId}.bin");
    int checks = 0, failures = 0;
    var notRun = new List<string>();
    void Check(bool ok, string what)
    {
        checks++;
        if (!ok) failures++;
        Console.WriteLine((ok ? "PASS " : "FAIL ") + what);
    }
    static string Mib(long v) => v < 0 ? "n/a" : $"{v / 1048576.0:F2} MiB";
    long Res(long off, long len) => HostPageCache.ResidentBytes(path, off, len, HostPageCacheMeasure.Full, out _);

    try
    {
        // A deterministic file, written and flushed.
        using (var f = new FileStream(path, FileMode.CreateNew, FileAccess.Write))
        {
            var block = new byte[1 << 20];
            ulong x = 0x9E3779B97F4A7C15;
            for (long done = 0; done < FileBytes; done += block.Length)
            {
                for (int i = 0; i < block.Length; i++)
                {
                    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
                    block[i] = (byte)x;
                }
                f.Write(block, 0, (int)Math.Min(block.Length, FileBytes - done));
            }
            f.Flush(flushToDisk: true);
        }
        var range = new HostFileRange("table", path, tableOffset, tableBytes);

        // 0. Unmapped: fadvise alone drops it. 1. A read brings it all back.
        FadviseFile(path, 0, 0);
        Check(Res(0, FileBytes) <= 2L * page, $"an unmapped file leaves the cache with fadvise alone ({Mib(Res(0, FileBytes))} left)");
        File.ReadAllBytes(path);
        long all = Res(0, FileBytes);
        Check(all >= FileBytes - page, $"reading the file caches all of it ({Mib(all)} of {Mib(FileBytes)})");

        // 2. A .NET read-only mapping (what GgufFile / GgufShardSet create) with every table page touched.
        using var mmf = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        using var view = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        byte* basePtr = null;
        view.SafeMemoryMappedViewHandle.AcquirePointer(ref basePtr);
        try
        {
            byte* mapped = basePtr + view.PointerOffset;
            ulong sum = Touch(mapped + tableOffset, tableBytes);

            // 3. Control: fadvise alone keeps the pages this process maps.
            FadviseFile(path, alStart, alEnd - alStart);
            long afterFadvise = Res(tableOffset, tableBytes);
            Check(afterFadvise >= tableBytes - page,
                $"control: posix_fadvise(DONTNEED) alone keeps mapped pages ({Mib(afterFadvise)} of {Mib(tableBytes)} still cached)");

            // 4. Control: madvise alone drops the mapping's page-table entries but keeps the pages cached.
            int rc = Madvise(mapped + alStart, (nuint)(alEnd - alStart), 4);
            long afterMadvise = Res(tableOffset, tableBytes);
            Check(rc == 0 && afterMadvise >= tableBytes - page,
                $"control: madvise(MADV_DONTNEED) alone keeps the pages cached ({Mib(afterMadvise)} still cached)");
            Check(Touch(mapped + tableOffset, tableBytes) == sum, "the mapping reads the same bytes after madvise");

            // 5. The eviction: both steps on every mapping of the file this process has.
            var entries = HostPageCache.ParseMaps(File.ReadAllText("/proc/self/maps"));
            Check(entries.Any(e => e.Path == Path.GetFullPath(path)), "/proc/self/maps lists the test mapping");
            HostPageCacheReport rep = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
            Console.WriteLine("  report: " + rep.Describe());
            Check(rep.Supported && rep.Mappings >= 1 && rep.MappingsAdvised == rep.Mappings && rep.Errors == 0,
                $"Evict found and advised the .NET mapping (mappings={rep.Mappings}, advised={rep.MappingsAdvised}, errors={rep.Errors})");
            Check(rep.ResidentBefore >= tableBytes - page, $"Evict measured the table cached before ({Mib(rep.ResidentBefore)})");
            Check(rep.ResidentAfter == 0, $"Evict left none of the table cached ({Mib(rep.ResidentAfter)})");
            Check(rep.Verified == true, "Evict reports verified=true");

            // 6. Only the range (widened to 2 MiB folio boundaries) left: the rest of the file stays cached.
            long head = Res(0, evStart), tail = Res(evEnd, FileBytes - evEnd);
            Check(head == evStart && tail >= FileBytes - evEnd - page,
                $"pages outside the widened table stay cached (head {Mib(head)} of {Mib(evStart)}, tail {Mib(tail)} of {Mib(FileBytes - evEnd)})");

            // 7. The mapping still works and reads the original bytes; reading it caches the table again.
            Check(Touch(mapped + tableOffset, tableBytes) == sum, "the mapping reads the original bytes after the eviction");
            Check(Res(tableOffset, tableBytes) >= tableBytes - page, "reading through the mapping caches the table again");

            // 8. A second mapping created with mmap directly (the native ggml loader's way) is found and advised too.
            using (var raw = File.OpenHandle(path, FileMode.Open, FileAccess.Read))
            {
                void* m2 = Mmap(null, (nuint)FileBytes, 1, 1, (int)raw.DangerousGetHandle(), 0);
                Check(m2 != (void*)-1, "raw mmap (PROT_READ, MAP_SHARED) succeeded");
                try
                {
                    Check(Touch((byte*)m2 + tableOffset, tableBytes) == sum, "the raw mapping reads the same bytes");
                    HostPageCacheReport two = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
                    Console.WriteLine("  report: " + two.Describe());
                    Check(two.Mappings == 2 && two.MappingsAdvised == 2 && two.ResidentAfter == 0 && two.Verified == true,
                        $"two mappings (.NET + raw mmap): both advised and the table left the cache (mappings={two.Mappings}, after {Mib(two.ResidentAfter)})");
                }
                finally
                {
                    Munmap(m2, (nuint)FileBytes);
                }
            }

            // 9. A writable private mapping is left alone (dropping a written private page would lose it): the
            //    pages it maps stay cached and the eviction says it is not verified, never a silent "cold".
            Touch(mapped + tableOffset, tableBytes);
            using (var raw = File.OpenHandle(path, FileMode.Open, FileAccess.Read))
            {
                void* priv = Mmap(null, (nuint)FileBytes, 3, 2, (int)raw.DangerousGetHandle(), 0);   // PROT_READ|WRITE, MAP_PRIVATE
                Check(priv != (void*)-1, "raw mmap (PROT_READ|PROT_WRITE, MAP_PRIVATE) succeeded");
                try
                {
                    Touch((byte*)priv + tableOffset, tableBytes);
                    HostPageCacheReport w = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
                    Console.WriteLine("  report: " + w.Describe());
                    Check(w.MappingsSkipped == 1 && w.MappingsAdvised == 1 && w.ResidentAfter > 0 && w.Verified == false,
                        $"a writable private mapping is skipped and the eviction is NOT verified (skipped={w.MappingsSkipped}, after {Mib(w.ResidentAfter)})");
                }
                finally
                {
                    Munmap(priv, (nuint)FileBytes);
                }
            }
            HostPageCacheReport again = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
            Check(again.ResidentAfter == 0 && again.Verified == true, "without the writable mapping the next eviction is verified");

            // 9b. An mlock()ed part of a mapping (the pure C# DeepSeek executor pins its shards) cannot leave the
            //     cache: the eviction reports the EINVAL and is NOT verified, never a silent "cold".
            Touch(mapped + tableOffset, tableBytes);
            byte* lockAt = mapped + alStart + (1L << 20);
            const long LockBytes = 64L << 10;
            if (Native.Mlock(lockAt, (nuint)LockBytes) != 0)
            {
                notRun.Add($"mlock of {LockBytes} B failed (errno {Marshal.GetLastPInvokeError()}, RLIMIT_MEMLOCK?): the locked-mapping check did not run");
            }
            else
            {
                try
                {
                    HostPageCacheReport locked = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
                    Console.WriteLine("  report: " + locked.Describe());
                    Check(locked.Errors == 1 && (locked.Ranges[0].Error ?? "").Contains("mlock") && locked.ResidentAfter >= LockBytes
                          && locked.Verified == false,
                        $"an mlock()ed mapping is reported and the eviction is NOT verified (after {Mib(locked.ResidentAfter)}: {locked.Ranges[0].Error})");
                }
                finally
                {
                    Native.Munlock(lockAt, (nuint)LockBytes);
                }
                HostPageCacheReport unlocked = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.Full);
                Check(unlocked.Errors == 0 && unlocked.ResidentAfter == 0 && unlocked.Verified == true, "after munlock the next eviction is verified");
            }

            // 10. A sampled measurement estimates a partly cached range: every other 1 MiB cached, 8 sampled windows.
            using (var rd = File.OpenHandle(path, FileMode.Open, FileAccess.Read))
            {
                Native.Fadvise((int)rd.DangerousGetHandle(), 0, 0, 1);   // POSIX_FADV_RANDOM: no read-ahead into the gaps
                var buf = new byte[1 << 20];
                for (long at = alStart; at < alEnd; at += 2L << 20)
                    RandomAccess.Read(rd, buf.AsSpan(0, (int)Math.Min(buf.Length, alEnd - at)), at);
            }
            long full = HostPageCache.ResidentBytes(path, tableOffset, tableBytes, HostPageCacheMeasure.Full, 1L << 20, 8, out double fullShare);
            long sampled = HostPageCache.ResidentBytes(path, tableOffset, tableBytes, HostPageCacheMeasure.Sampled, 1L << 20, 8, out double share);
            Check(fullShare == 1 && Math.Abs(full - tableBytes / 2.0) <= 2 << 20,
                $"full measurement of a half-cached table ({Mib(full)} of {Mib(tableBytes)})");
            Check(share > 0.1 && share < 0.4 && Math.Abs(sampled - full) <= tableBytes * 0.15,
                $"sampled estimate within 15% of the table ({Mib(sampled)} vs full {Mib(full)}, inspected {share:P0} of pages)");

            // 11. No measurement: nothing to verify.
            HostPageCacheReport none = HostPageCache.Evict(new[] { range }, HostPageCacheMeasure.None);
            Check(none.Verified == null && none.ResidentBefore < 0 && none.ResidentAfter < 0 && none.MappingsAdvised >= 1,
                "measure=none still evicts and reports verified=null");
            Check(Res(tableOffset, tableBytes) == 0, "and the table did leave the cache");
        }
        finally
        {
            view.SafeMemoryMappedViewHandle.ReleasePointer();
        }

        // 12. A missing file is reported, not thrown.
        HostPageCacheReport missing = HostPageCache.Evict(
            new[] { new HostFileRange("gone", Path.Combine(dir, "no-such-file.bin"), 0, 4096) }, HostPageCacheMeasure.Full);
        Check(missing.Supported && missing.Errors == 1 && missing.Verified != true, "a missing file is an error, never verified");
    }
    finally
    {
        try { File.Delete(path); } catch (IOException) { }
    }

    Console.WriteLine($"{checks - failures}/{checks} checks passed");
    foreach (string why in notRun)
        Console.WriteLine("NOT RUN " + why);
    return failures == 0 ? 0 : 1;

    static ulong Touch(byte* p, long bytes)
    {
        ulong sum = 0;
        for (long i = 0; i < bytes; i += 512)
            sum = sum * 31 + p[i];
        return sum + p[bytes - 1];
    }

    static void FadviseFile(string file, long offset, long length)
    {
        using var h = File.OpenHandle(file, FileMode.Open, FileAccess.Read);
        int rc = Native.Fadvise((int)h.DangerousGetHandle(), offset, length, 4);
        if (rc != 0) throw new IOException("posix_fadvise: " + rc);
    }

    static int Madvise(byte* p, nuint len, int advice) => Native.Madvise(p, len, advice);
    static void* Mmap(void* a, nuint len, int prot, int flags, int fd, long off) => Native.Mmap(a, len, prot, flags, fd, off);
    static int Munmap(void* a, nuint len) => Native.Munmap(a, len);

    static string FileSystemOf(string directory)
    {
        string full = Path.GetFullPath(directory);
        string best = "", type = "unknown";
        foreach (string line in File.ReadLines("/proc/self/mounts"))
        {
            string[] f = line.Split(' ');
            if (f.Length > 2 && (full == f[1] || full.StartsWith(f[1].TrimEnd('/') + "/", StringComparison.Ordinal)) && f[1].Length >= best.Length)
                (best, type) = (f[1], f[2]);
        }
        return type;
    }
}

static unsafe class Native
{
    [DllImport("libc", EntryPoint = "posix_fadvise")] public static extern int Fadvise(int fd, long offset, long length, int advice);
    [DllImport("libc", EntryPoint = "madvise", SetLastError = true)] public static extern int Madvise(void* address, nuint length, int advice);
    [DllImport("libc", EntryPoint = "mmap", SetLastError = true)] public static extern void* Mmap(void* address, nuint length, int prot, int flags, int fd, long offset);
    [DllImport("libc", EntryPoint = "munmap", SetLastError = true)] public static extern int Munmap(void* address, nuint length);
    [DllImport("libc", EntryPoint = "mlock", SetLastError = true)] public static extern int Mlock(void* address, nuint length);
    [DllImport("libc", EntryPoint = "munlock", SetLastError = true)] public static extern int Munlock(void* address, nuint length);
}
