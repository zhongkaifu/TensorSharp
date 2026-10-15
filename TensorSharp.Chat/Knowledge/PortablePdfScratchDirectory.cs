using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace TensorSharp.Server.Knowledge;

/// <summary>Cleanup for a private scratch directory already owned by the caller.
/// It does not authorize deleting an arbitrary directory or provide a filesystem sandbox.</summary>
internal static class PortablePdfScratchDirectory
{
    internal const string CleanupErrorKey = "TensorSharp.PdfScratchCleanupError";
    internal const string CleanupDirectoryKey = "TensorSharp.PdfScratchCleanupDirectory";
    private static readonly TimeSpan RetryBudget = TimeSpan.FromSeconds(1);
    private static readonly TimeSpan RetryDelay = TimeSpan.FromMilliseconds(20);

    internal static void Delete(string ownedDirectory)
        => DeleteAsync(ownedDirectory).GetAwaiter().GetResult();

    internal static async Task DeleteAsync(string ownedDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(ownedDirectory);
        if (!Path.IsPathFullyQualified(ownedDirectory))
            throw new ArgumentException("A private scratch directory must have an absolute path.", nameof(ownedDirectory));
        string directory = Path.GetFullPath(ownedDirectory);
        if (string.Equals(Path.TrimEndingDirectorySeparator(directory),
                Path.TrimEndingDirectorySeparator(Path.GetPathRoot(directory)!),
                OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal))
            throw new ArgumentException("A filesystem root cannot be a scratch directory.", nameof(ownedDirectory));

        long started = Stopwatch.GetTimestamp();
        while (true)
        {
            try
            {
                Directory.Delete(directory, recursive: true);
                return;
            }
            catch (IOException error) when (IsTransientSharingViolation(error))
            {
                // Process termination can precede Windows releasing a file handle.
                // Retry only its exact sharing/lock errors, independent of the now
                // canceled request token. Other filesystem errors escape immediately.
                TimeSpan remaining = RetryBudget - Stopwatch.GetElapsedTime(started);
                if (remaining <= TimeSpan.Zero) throw;
                await Task.Delay(remaining < RetryDelay ? remaining : RetryDelay).ConfigureAwait(false);
                if (Stopwatch.GetElapsedTime(started) >= RetryBudget) throw;
            }
        }
    }

    internal static async Task DeletePreservingFailureAsync(string ownedDirectory, Exception? primaryFailure)
    {
        try
        {
            await DeleteAsync(ownedDirectory).ConfigureAwait(false);
        }
        catch (Exception cleanup) when (primaryFailure is not null &&
            cleanup is IOException or UnauthorizedAccessException)
        {
            // Keep cancellation's token/stack and the actual processing exception.
            // The retained cleanup error and path explicitly report unreleased files;
            // callers must not interpret this as a successful deletion.
            primaryFailure.Data[CleanupErrorKey] = cleanup;
            primaryFailure.Data[CleanupDirectoryKey] = Path.GetFullPath(ownedDirectory);
        }
    }

    private static bool IsTransientSharingViolation(IOException error)
        => OperatingSystem.IsWindows() &&
            (error.HResult == unchecked((int)0x80070020) || error.HResult == unchecked((int)0x80070021));
}
