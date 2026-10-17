// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json;

namespace TensorSharp.Studio;

internal static class StudioPaths
{
    public static string Id(string? value)
    {
        if (value is null || value.Length != 32 || !value.All(c => c is >= '0' and <= '9' or >= 'a' and <= 'f'))
            throw new StudioException("invalid_id", "Identifiers must be 32 lowercase hexadecimal characters.");
        return value;
    }
    public static string FileName(string value)
    {
        if (string.IsNullOrWhiteSpace(value) || value.Length > 160 || value != Path.GetFileName(value) || value.IndexOfAny(['\\', ':', '<', '>', '"', '|', '?', '*', '\0']) >= 0 || value.EndsWith('.') || value.EndsWith(' ') || value is "." or ".." ||
            new[] { "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9" }.Contains(value.Split('.')[0], StringComparer.OrdinalIgnoreCase))
            throw new StudioException("invalid_path", "Media references must be safe file names.");
        return value;
    }
    public static string SafeChild(string root, params string[] parts)
    {
        var path = Path.GetFullPath(root);
        foreach (var part in parts)
        {
            FileName(part); path = Path.Combine(path, part);
            if ((File.Exists(path) || Directory.Exists(path)) && File.GetAttributes(path).HasFlag(FileAttributes.ReparsePoint))
                throw new StudioException("unsafe_path", "Symbolic links are not permitted in managed project storage.");
        }
        return path;
    }
}

internal sealed class StudioDocument
{
    public StudioProject Project { get; set; } = new();
    public List<StudioProject> Undo { get; set; } = [];
    public List<StudioProject> Redo { get; set; } = [];
    // Optional targeted asset changes keyed by the associated history snapshot's revision.
    // Older documents have empty Assets in history and no map: their edits preserve the library.
    public Dictionary<long, StudioAssetHistory> AssetHistory { get; set; } = [];
    public Dictionary<string, string> AppliedCommands { get; set; } = [];
    public List<StudioProposal> Proposals { get; set; } = [];
}

internal sealed class StudioAssetHistory
{
    public List<StudioAssetRestore> Restore { get; set; } = [];
    public List<string> Remove { get; set; } = [];
}

internal sealed class StudioAssetRestore
{
    public StudioAsset Asset { get; set; } = new();
    public int Index { get; set; }
}

internal static class StudioStorage
{
    public static T Read<T>(string path)
    {
        if (!File.Exists(path)) throw new StudioException("not_found", "The requested project or job does not exist.");
        if (new FileInfo(path).Length > 64L * 1024 * 1024) throw new StudioException("document_too_large", "Project metadata exceeds the supported document size.");
        try { return JsonSerializer.Deserialize<T>(File.ReadAllText(path), StudioJson.Options) ?? throw new JsonException("Empty document"); }
        catch (JsonException e) { throw new StudioException("invalid_document", $"Stored project metadata could not be read: {e.Message}"); }
    }
    public static void Save<T>(string path, T value)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var temporary = path + "." + StudioJson.NewId() + ".tmp";
        try
        {
            using (var stream = new FileStream(temporary, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            {
                JsonSerializer.Serialize(stream, value, StudioJson.Options);
                if (stream.Length > 64L * 1024 * 1024) throw new StudioException("document_too_large", "Project metadata and history exceed the supported 64 MiB document size. Export/import as a new project to reset its edit history.");
                stream.Flush(true);
            }
            File.Move(temporary, path, true);
        }
        finally { if (File.Exists(temporary)) File.Delete(temporary); }
    }
}
