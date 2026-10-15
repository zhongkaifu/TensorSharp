// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace InferenceWeb.Tests;

/// <summary>Discovery-time gate, so hosts without symlink permission report skipped coverage.</summary>
public sealed class SymbolicLinkFactAttribute : FactAttribute
{
    private static readonly Lazy<string?> SkipReason = new(() =>
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-symlink-probe-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(directory);
            string target = Path.Combine(directory, "target");
            File.WriteAllText(target, "probe");
            File.CreateSymbolicLink(Path.Combine(directory, "link"), target);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        { return "Requires permission and filesystem support for creating symbolic links."; }
        finally { try { Directory.Delete(directory, recursive: true); } catch (IOException) { } }
    });
    public SymbolicLinkFactAttribute() => Skip = SkipReason.Value;
}
