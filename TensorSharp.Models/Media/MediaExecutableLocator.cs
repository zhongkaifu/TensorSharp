// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Linq;

namespace TensorSharp.Models.Media;

/// <summary>Shared executable discovery for installed local media tools. Never downloads or invokes a shell.</summary>
public static class MediaExecutableLocator
{
    public static string Find(string executable, string environmentVariable, string siblingOf = null)
    {
        string configured = Environment.GetEnvironmentVariable(environmentVariable);
        if (!string.IsNullOrWhiteSpace(configured)) return configured; // An explicit invalid setting must remain an actionable failure.
        string name = OperatingSystem.IsWindows() ? executable + ".exe" : executable;
        string baseDir = AppContext.BaseDirectory;
        var directories = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator)
            .Concat(new[] { baseDir, Path.Combine(baseDir, "ffmpeg"), Path.Combine(baseDir, "ffmpeg", "bin") });
        if (!string.IsNullOrEmpty(siblingOf) && Path.IsPathRooted(siblingOf))
            directories = new[] { Path.GetDirectoryName(siblingOf) }.Concat(directories);
        // Finder-launched applications often receive only /usr/bin:/bin. These are
        // installed-tool locations, not automatic package-manager installation.
        if (OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst())
            directories = directories.Concat(new[] { "/opt/homebrew/bin", "/usr/local/bin" });
        foreach (string directory in directories)
        {
            try
            {
                if (!string.IsNullOrWhiteSpace(directory))
                {
                    string candidate = Path.Combine(directory, name);
                    if (File.Exists(candidate)) return candidate;
                }
            }
            catch (ArgumentException) { }
        }
        return null;
    }
}
