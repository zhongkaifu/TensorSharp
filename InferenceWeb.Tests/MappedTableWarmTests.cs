using System.Collections.Concurrent;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class MappedTableWarmTests
{
    [Fact]
    public void AdmissionIncludesReadBuffersAndHeadroomWithoutOverflow()
    {
        Assert.Equal(0, MappedTableWarm.WorkerCount(100, 0, 20, 16, 4));
        Assert.Equal(0, MappedTableWarm.WorkerCount(100, 135, 20, 16, 4));
        Assert.Equal(1, MappedTableWarm.WorkerCount(100, 136, 20, 16, 4));
        Assert.Equal(3, MappedTableWarm.WorkerCount(100, 168, 20, 16, 4));
        Assert.Equal(4, MappedTableWarm.WorkerCount(100, 200, 20, 16, 4));
        Assert.Equal(1, MappedTableWarm.WorkerCount(1, 100, 20, 16, 4));
        Assert.Equal(0, MappedTableWarm.WorkerCount(long.MaxValue, long.MaxValue, 20, 16, 4));
        Assert.Equal(0, MappedTableWarm.WorkerCount(28_800_138_240, 32L << 30, 8L << 30, 8 << 20, 4));
    }

    [Fact]
    public void PhysicalMemoryProbeReturnsAvailableBytesOnSupportedHosts()
    {
        long available = MappedTableWarm.HostMemoryAvailable();
        if (OperatingSystem.IsWindows() || OperatingSystem.IsLinux())
            Assert.True(available > 0, "The OS physical-memory query failed; warm admission must not treat this as unlimited RAM.");
        else
            Assert.Equal(0, available);
    }

    [Fact]
    public async Task UnknownOrLowMemoryDoesNotOpenFilesAndDoesNotPermanentlyDisableWarming()
    {
        var messages = new ConcurrentQueue<string>();
        var missing = new[] { (Path: "missing-warm-fixture", Offset: 0L, Bytes: 24L) };
        Assert.Null(Start(missing, () => 0, messages));
        Assert.Null(Start(missing, () => 24 + 20 + 7, messages));
        Assert.Contains(messages, message => message.Contains("unknown"));
        Assert.DoesNotContain(messages, message => message.Contains("stopped"));

        string path = NewFile(32);
        try
        {
            // Repeated loads can reuse normal page-cache data; a previous pressure skip is not sticky.
            for (int repeat = 0; repeat < 2; repeat++)
                await Start(new[] { (path, 4L, 24L) }, () => 100, messages);
            Assert.Equal(2, messages.Count(message => message.Contains("] warmed ")));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task PressureDuringWarmStopsBeforeReportingCompletion()
    {
        string path = NewFile(32);
        var messages = new ConcurrentQueue<string>();
        int checks = 0;
        try
        {
            // Initial admission, admission under the gate, one block; then real memory pressure.
            await Start(new[] { (path, 0L, 32L) }, () => Interlocked.Increment(ref checks) <= 3 ? 100 : 19, messages);
            Assert.Contains(messages, message => message.Contains("warming stopped"));
            Assert.DoesNotContain(messages, message => message.Contains("] warmed "));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task TruncatedFileIsNotReportedAsWarmedAndReleasesHandle()
    {
        string path = NewFile(9);
        var messages = new ConcurrentQueue<string>();
        try
        {
            await Start(new[] { (path, 0L, 24L) }, () => 100, messages);
            Assert.Contains(messages, message => message.Contains("ended before"));
            Assert.DoesNotContain(messages, message => message.Contains("] warmed "));
            using var exclusive = File.Open(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
            Assert.Equal(9, exclusive.Length);
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task QueuedWarmRechecksCapacityInsteadOfSharingAnOldAdmission()
    {
        string path = NewFile(16);
        var messages = new ConcurrentQueue<string>();
        using var reading = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        Task first = null;
        Task second = null;
        int checks = 0;
        long secondAvailable = 100;
        try
        {
            first = Start(new[] { (path, 0L, 16L) }, () =>
            {
                if (Interlocked.Increment(ref checks) == 3)
                {
                    reading.Set();
                    if (!release.Wait(TimeSpan.FromSeconds(10)))
                        throw new TimeoutException("Warm fixture did not release the read.");
                }
                return 100;
            }, messages);
            Assert.True(reading.Wait(TimeSpan.FromSeconds(10)));
            second = Start(new[] { (path, 0L, 16L) }, () => Interlocked.Read(ref secondAvailable), messages);
            Assert.NotNull(second);
            Interlocked.Exchange(ref secondAvailable, 0);
            release.Set();
            await Task.WhenAll(first, second);
            Assert.Single(messages.Where(message => message.Contains("] warmed ")));
            Assert.Contains(messages, message => message.Contains("unknown"));
        }
        finally
        {
            release.Set();
            if (first != null) await first;
            if (second != null) await second;
            File.Delete(path);
        }
    }

    [Fact]
    public void CancelledAndInvalidRangesDoNotStartIo()
    {
        var messages = new ConcurrentQueue<string>();
        using var cancelled = new CancellationTokenSource();
        cancelled.Cancel();
        Assert.Null(MappedTableWarm.Start(new[] { ("missing", 0L, 24L) }, "test", "fixture", cancelled.Token,
            () => 100, 20, 8, 1, messages.Enqueue));
        Assert.Throws<ArgumentException>(() => { _ = Start(new[] { ("missing", long.MaxValue, 1L) }, () => 100, messages); });
        Assert.Throws<OverflowException>(() => { _ = Start(new[] { ("missing", 0L, long.MaxValue), ("missing", 0L, 1L) },
            () => long.MaxValue, messages); });
        Assert.Empty(messages);
    }

    private static Task Start(IReadOnlyList<(string Path, long Offset, long Bytes)> ranges,
        Func<long> memory, ConcurrentQueue<string> messages) =>
        MappedTableWarm.Start(ranges, "test", "fixture", CancellationToken.None, memory, 20, 8, 1, messages.Enqueue);

    private static string NewFile(int bytes)
    {
        string path = Path.Combine(Path.GetTempPath(), $"tensorsharp-warm-{Guid.NewGuid():N}.bin");
        File.WriteAllBytes(path, Enumerable.Range(0, bytes).Select(value => (byte)value).ToArray());
        return path;
    }
}
