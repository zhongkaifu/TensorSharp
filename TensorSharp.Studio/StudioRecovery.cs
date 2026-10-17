// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.RegularExpressions;

namespace TensorSharp.Studio;

public sealed partial class StudioService
{
    // Recovery holds the exclusive host lease, before any workers start. Limit deletion/traversal
    // work per startup; never enumerate arbitrary filesystem trees or read partial media into RAM.
    const int StartupCleanupBudget = 256;
    static bool IsManagedId(string name) => name.Length == 32 && name.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f');
    static bool IsAtomicTemporary(string name, string documentName) => name.StartsWith(documentName + ".", StringComparison.Ordinal)
        && name.EndsWith(".tmp", StringComparison.Ordinal) && name.Length == documentName.Length + 37
        && IsManagedId(name.Substring(documentName.Length + 1, 32));

    void CleanupPortableScratch(ref int budget)
    {
        var packages = StudioPaths.SafeChild(RootPath, "packages");
        if (Directory.Exists(packages))
            foreach (var path in Directory.EnumerateFiles(packages).Take(10000))
            {
                if (budget <= 0) break;
                // Completed packages, custom files, and arbitrary *.tmp names are preserved.
                var name = Path.GetFileName(path);
                if (name.Length <= 128 && Regex.IsMatch(name, @"\A[0-9a-f]{32}-[0-9]{1,19}\.tsstudio\.[0-9a-f]{32}\.tmp\z", RegexOptions.CultureInvariant))
                    DeleteScratchFile(path, ref budget);
            }
        var imports = StudioPaths.SafeChild(RootPath, "imports");
        if (Directory.Exists(imports))
            foreach (var path in Directory.EnumerateFileSystemEntries(imports).Take(10000))
            {
                if (budget <= 0) break;
                var name = Path.GetFileName(path);
                if (name.Length == 36 && name.EndsWith(".zip", StringComparison.Ordinal) && IsManagedId(name[..32])) DeleteScratchFile(path, ref budget);
                else if (IsManagedId(name) && Directory.Exists(path)) CleanupScratchTree(path, ref budget);
            }
        foreach (var directory in Directory.EnumerateDirectories(StudioPaths.SafeChild(RootPath, "projects")).Take(10000))
        {
            if (budget <= 0) break;
            if (!IsManagedId(Path.GetFileName(directory)) || File.GetAttributes(directory).HasFlag(FileAttributes.ReparsePoint)) continue;
            CleanupAtomicTemporaries(directory, "project.json", ref budget);
        }
    }

    static void CleanupAtomicTemporaries(string directory, string documentName, ref int budget)
    {
        if (budget <= 0) return;
        foreach (var path in Directory.EnumerateFiles(directory, documentName + ".*.tmp").Take(256))
        {
            if (budget <= 0) break;
            if (IsAtomicTemporary(Path.GetFileName(path), documentName)) DeleteScratchFile(path, ref budget);
        }
    }
    static void DeleteScratchFile(string path, ref int budget)
    {
        if (budget <= 0) return;
        budget--;
        try { File.Delete(path); }
        catch (IOException) { /* Keep a locked partial for a later startup, without blocking projects. */ }
        catch (UnauthorizedAccessException) { }
    }
    static void CleanupScratchTree(string directory, ref int budget, int depth = 0)
    {
        if (budget <= 0 || depth > 8 || !Directory.Exists(directory)) return;
        try
        {
            if (File.GetAttributes(directory).HasFlag(FileAttributes.ReparsePoint))
            {
                // Delete the directory link only; never follow it into another tree.
                budget--; Directory.Delete(directory); return;
            }
            foreach (var path in Directory.EnumerateFileSystemEntries(directory))
            {
                if (budget <= 0) return;
                if (Directory.Exists(path))
                {
                    budget--; CleanupScratchTree(path, ref budget, depth + 1);
                }
                else DeleteScratchFile(path, ref budget);
            }
            if (budget > 0) { budget--; Directory.Delete(directory, recursive: false); }
        }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}
