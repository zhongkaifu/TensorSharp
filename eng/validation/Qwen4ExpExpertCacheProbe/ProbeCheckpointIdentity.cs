using System.Security.Cryptography;
using System.Text.Json;
using System.Text.RegularExpressions;

internal static class ProbeCheckpointIdentity
{
    // Reuse a separate, completed publisher-hash verification instead of reading
    // a 74 GB model again during every route/timing arm. Size/time checks detect
    // ordinary edits; they are not a substitute for a new hash after mutation.
    internal static object Read(string modelPath, string manifestPath)
    {
        byte[] manifestBytes = File.ReadAllBytes(manifestPath);
        using var manifest = JsonDocument.Parse(manifestBytes);
        var records = manifest.RootElement.GetProperty("files").EnumerateArray()
            .ToDictionary(entry => Path.GetFullPath(entry.GetProperty("path").GetString()!),
                entry => entry, OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal);
        string full = Path.GetFullPath(modelPath);
        Match split = Regex.Match(full, @"^(.*)-(\d{5})-of-(\d{5})\.gguf$", RegexOptions.IgnoreCase);
        int count = split.Success ? int.Parse(split.Groups[3].Value) : 1;
        if (count <= 0 || (split.Success && int.Parse(split.Groups[2].Value) != 1))
            throw new InvalidDataException("Checkpoint identity must start at the first GGUF shard.");
        var files = new List<object>();
        for (int index = 0; index < count; index++)
        {
            string path = split.Success ? $"{split.Groups[1].Value}-{index + 1:D5}-of-{count:D5}{Path.GetExtension(full)}" : full;
            if (!records.TryGetValue(path, out var record) || record.GetProperty("status").GetString() != "verified")
                throw new InvalidDataException($"No completed hash verification for {path}.");
            string hash = record.GetProperty("sha256").GetString() ?? "";
            var expected = record.GetProperty("expected");
            if (!Regex.IsMatch(hash, "^[0-9a-fA-F]{64}$") ||
                !string.Equals(hash, expected.GetProperty("sha256").GetString(), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException($"Publisher hash verification is absent or inconsistent for {path}.");
            var file = new FileInfo(path);
            long timestamp = checked((file.LastWriteTimeUtc.Ticks - DateTime.UnixEpoch.Ticks) * 100);
            if (!file.Exists || file.Length != record.GetProperty("bytes").GetInt64() ||
                file.Length != expected.GetProperty("bytes").GetInt64() ||
                timestamp != record.GetProperty("mtime_ns").GetInt64())
                throw new InvalidDataException($"Checkpoint changed since its full hash verification: {path}.");
            if (count > 1)
            {
                var structure = record.GetProperty("structure");
                if (structure.GetProperty("split_no").GetInt32() != index ||
                    structure.GetProperty("split_count").GetInt32() != count)
                    throw new InvalidDataException($"Incomplete or inconsistent GGUF shard identity: {path}.");
            }
            files.Add(new { path, bytes = file.Length, mtime_ns = timestamp, sha256 = hash.ToLowerInvariant(),
                repository = expected.GetProperty("repository").GetString(),
                source_revision = expected.GetProperty("source_revision").GetString() });
        }
        return new { model_path = full, manifest_path = Path.GetFullPath(manifestPath),
            manifest_sha256 = Convert.ToHexString(SHA256.HashData(manifestBytes)).ToLowerInvariant(), files,
            current_metadata_checked = true, content_rehashed = false,
            limitation = "Previously verified complete publisher hashes, reused only while file size/mtime agree; not a new per-run content hash." };
    }
}
