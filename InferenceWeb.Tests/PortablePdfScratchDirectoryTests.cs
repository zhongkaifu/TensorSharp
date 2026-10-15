using System.Diagnostics;
using TensorSharp.Server.Knowledge;

namespace InferenceWeb.Tests;

public sealed class PortablePdfScratchDirectoryTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "pdf-cleanup-" + Guid.NewGuid().ToString("N"));

    public PortablePdfScratchDirectoryTests() => Directory.CreateDirectory(_directory);

    [Fact]
    public async Task AnOwnedDirectoryAndItsContentsAreActuallyDeleted()
    {
        string nested = Directory.CreateDirectory(Path.Combine(_directory, "child")).FullName;
        await File.WriteAllTextAsync(Path.Combine(nested, "中文.txt"), "fixture contents");
        await PortablePdfScratchDirectory.DeleteAsync(_directory);
        Assert.False(Directory.Exists(_directory));
    }

    [Fact]
    public async Task EmptyRelativeAndFilesystemRootPathsAreRejectedBeforeDeletion()
    {
        foreach (string path in new[] { "", " ", ".", Path.GetPathRoot(_directory)! })
            await Assert.ThrowsAnyAsync<ArgumentException>(() => PortablePdfScratchDirectory.DeleteAsync(path));
        Assert.True(Directory.Exists(_directory));
    }

    [Fact]
    public async Task AFileInPlaceOfADirectoryIsNotIgnored()
    {
        string path = Path.Combine(_directory, "not-a-directory");
        await File.WriteAllTextAsync(path, "fixture contents");
        await Assert.ThrowsAnyAsync<IOException>(() => PortablePdfScratchDirectory.DeleteAsync(path));
        Assert.True(File.Exists(path));
    }

    [Fact]
    public async Task AFailedCleanupOnTheSuccessPathIsReported()
    {
        string absent = Path.Combine(_directory, "absent");
        await Assert.ThrowsAsync<DirectoryNotFoundException>(() =>
            PortablePdfScratchDirectory.DeletePreservingFailureAsync(absent, null));
    }

    [Fact]
    public async Task AnExistingCancellationKeepsItsTokenAndStackAlongsideTheCleanupDiagnostic()
    {
        using var cancellation = new CancellationTokenSource();
        cancellation.Cancel();
        var primary = Assert.Throws<OperationCanceledException>(() => cancellation.Token.ThrowIfCancellationRequested());
        string stack = primary.StackTrace!;
        string absent = Path.Combine(_directory, "absent");
        await PortablePdfScratchDirectory.DeletePreservingFailureAsync(absent, primary);
        Assert.IsType<DirectoryNotFoundException>(primary.Data[PortablePdfScratchDirectory.CleanupErrorKey]);
        Assert.Equal(absent, primary.Data[PortablePdfScratchDirectory.CleanupDirectoryKey]);
        Assert.Equal(cancellation.Token, primary.CancellationToken);
        Assert.Equal(stack, primary.StackTrace);
    }

    [PdfWindowsCleanupFact]
    public async Task ReleasingARealWindowsFileLockAllowsTheRetryToDeleteTheDirectory()
    {
        using var file = File.Open(Path.Combine(_directory, "locked"), FileMode.Create,
            FileAccess.ReadWrite, FileShare.None);
        var error = Assert.Throws<IOException>(() => Directory.Delete(_directory, recursive: true));
        Assert.Equal(unchecked((int)0x80070020), error.HResult);
        Task cleanup = PortablePdfScratchDirectory.DeleteAsync(_directory);
        Assert.False(cleanup.IsCompleted);
        file.Dispose();
        await cleanup;
        Assert.False(Directory.Exists(_directory));
    }

    [PdfWindowsCleanupFact]
    public async Task ARealPersistentWindowsFileLockIsReportedWithinTheRetryBudget()
    {
        using var file = File.Open(Path.Combine(_directory, "locked"), FileMode.Create,
            FileAccess.ReadWrite, FileShare.None);
        var elapsed = Stopwatch.StartNew();
        var error = await Assert.ThrowsAsync<IOException>(() => PortablePdfScratchDirectory.DeleteAsync(_directory));
        elapsed.Stop();
        Assert.Equal(unchecked((int)0x80070020), error.HResult);
        Assert.True(Directory.Exists(_directory));
        // The monotonic retry deadline is one second. Allow scheduling/OS-call
        // latency here; the helper cannot cancel Directory.Delete inside the OS.
        Assert.True(elapsed.Elapsed < TimeSpan.FromSeconds(2), $"Cleanup took {elapsed.Elapsed}.");
    }

    [PdfWindowsCleanupFact]
    public async Task AReadOnlyWindowsFileFailsWithoutBeingTreatedAsASharingViolation()
    {
        string path = Path.Combine(_directory, "read-only");
        await File.WriteAllTextAsync(path, "fixture contents");
        File.SetAttributes(path, FileAttributes.ReadOnly);
        try
        {
            await Assert.ThrowsAsync<UnauthorizedAccessException>(() => PortablePdfScratchDirectory.DeleteAsync(_directory));
            Assert.True(File.Exists(path));
        }
        finally { File.SetAttributes(path, FileAttributes.Normal); }
    }

    [PdfWindowsCleanupFact]
    public async Task AnUnreleasedWindowsLockIsAttachedToTheActualProcessingFailure()
    {
        using var file = File.Open(Path.Combine(_directory, "locked"), FileMode.Create,
            FileAccess.ReadWrite, FileShare.None);
        var primary = Assert.Throws<InvalidDataException>((Action)(() => throw new InvalidDataException("fixture processing failure")));
        string stack = primary.StackTrace!;
        await PortablePdfScratchDirectory.DeletePreservingFailureAsync(_directory, primary);
        var cleanup = Assert.IsType<IOException>(primary.Data[PortablePdfScratchDirectory.CleanupErrorKey]);
        Assert.Equal(unchecked((int)0x80070020), cleanup.HResult);
        Assert.Equal(_directory, primary.Data[PortablePdfScratchDirectory.CleanupDirectoryKey]);
        Assert.True(Directory.Exists(_directory));
        Assert.Equal(stack, primary.StackTrace);
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) PortablePdfScratchDirectory.Delete(_directory);
    }
}

internal sealed class PdfWindowsCleanupFactAttribute : FactAttribute
{
    public PdfWindowsCleanupFactAttribute()
    {
        if (!OperatingSystem.IsWindows())
            Skip = "Requires actual Windows delete-sharing semantics; no Windows file-lock coverage is claimed on other platforms.";
    }
}
