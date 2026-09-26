// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>
/// The shared-prefix checkpoints of one model, kept on the device between launches.
///
/// <para>
/// What it is for: the first message of every launch. The engine starts each new chat
/// from a copy of the model's state at the end of the prompt every conversation shares
/// (<see cref="IPrefixCheckpointStore"/>), but that state has to exist first, and in
/// a fresh process it does not: the warm-up after a load prefills it, which on the
/// phone is forty seconds for Qwen 9B and two and a half minutes for Bonsai 27B, and
/// a message sent before the warm-up is done pays the same. Measured on the iPhone 17
/// Pro Max: first-message time to first token 36-48 s (Qwen3.5 9B), 152 s (Bonsai
/// 27B). Written once, read back at the next load, the checkpoint costs the size of
/// a read from flash instead -- a hundred to a few hundred megabytes.
/// </para>
///
/// <para>
/// One directory per catalog model. A file is named by a hash of the model's K/V
/// identity and the exact prefix tokens, and carries both in its header so a hit is
/// checked byte for byte before the engine sees any payload; a file that fails that
/// check, or that cannot be read at all, is deleted and counts as absent. A save
/// goes to a temporary file and is renamed into place, so a process killed mid-write
/// (which on a phone is not rare) never leaves a half-checkpoint to be found. At
/// most <see cref="MaxFilesPerModel"/> files are kept, the least recently used going
/// first; a prefix changes whenever the skills, the tools or the thinking default
/// do, and each variant is a file.
/// </para>
/// </summary>
public sealed class PrefixCheckpointFileStore : IPrefixCheckpointStore
{
    private const uint Magic = 0x54535043;   // "TSPC"
    private const int Version = 1;
    private const string Extension = ".ckpt";

    /// <summary>Files kept per model: a prompt saves one per public checkpoint boundary
    /// (two, the system instructions and the whole shared prefix) and a host warms both
    /// thinking modes, so four keep a restart from prefilling either mode again (at two,
    /// the second mode's pair evicted the first's). Older ones go when a new one is saved.</summary>
    public const int MaxFilesPerModel = 4;

    private readonly ILogger _log;

    /// <param name="directory">Where this model's checkpoints live.</param>
    /// <param name="weightsIdentity">
    /// What the checkpoint's bytes were computed FROM. The engine's own fingerprint
    /// names the K/V geometry and precision, which two different files of the same
    /// architecture share: a re-downloaded model, a catalog entry moved to another
    /// quantisation with the same shape. A checkpoint restored across that would be a
    /// state no prefill of these weights produces, silently. So the store folds the
    /// weights' identity -- <see cref="WeightsIdentityOf"/>: name, length and write
    /// time of every file the model was loaded from -- into everything it names and
    /// checks.
    /// </param>
    public PrefixCheckpointFileStore(string directory, string? weightsIdentity = null, ILogger? log = null)
    {
        Directory = directory ?? throw new ArgumentNullException(nameof(directory));
        WeightsIdentity = weightsIdentity ?? string.Empty;
        _log = log ?? NullLogger.Instance;
        SweepStrayTemporaries();
    }

    /// <summary>Where this model's checkpoints live.</summary>
    public string Directory { get; }

    /// <summary>The weights these checkpoints belong to; part of every file's identity.</summary>
    public string WeightsIdentity { get; }

    /// <summary>Name, length and last write time of each file, in order -- enough to tell
    /// a re-download or a swapped file from the one the checkpoint was made with, at no
    /// cost (a hash of five gigabytes is not worth a launch).</summary>
    public static string WeightsIdentityOf(params string?[] files)
    {
        var sb = new StringBuilder();
        foreach (string? file in files)
        {
            if (string.IsNullOrEmpty(file)) continue;
            try
            {
                var info = new FileInfo(file);
                sb.Append(info.Name).Append('|').Append(info.Exists ? info.Length : -1).Append('|')
                  .Append(info.Exists ? info.LastWriteTimeUtc.Ticks : 0).Append(';');
            }
            catch (Exception)
            {
                sb.Append(Path.GetFileName(file)).Append("|?;");
            }
        }
        return sb.ToString();
    }

    private string Identity(string modelFingerprint) => (modelFingerprint ?? string.Empty) + "|" + WeightsIdentity;

    /// <summary>The file a checkpoint for these tokens would be kept in.</summary>
    public string PathFor(string modelFingerprint, ReadOnlySpan<int> prefixTokens)
        => Path.Combine(Directory, FileNameFor(Identity(modelFingerprint), prefixTokens));

    /// <summary>
    /// The file name for an identity that ALREADY carries the weights identity.
    ///
    /// <para>
    /// Internal, not public. It takes the pre-combined string <see cref="Identity"/>
    /// builds, not a bare model fingerprint, and a caller outside this class that passed
    /// a fingerprint would compute a name this store never writes — silently, and only
    /// whenever <see cref="WeightsIdentity"/> is non-empty, which is always in a real
    /// host. <see cref="PathFor"/> is the supported way to ask where a checkpoint lives.
    /// </para>
    /// </summary>
    internal static string FileNameFor(string modelFingerprint, ReadOnlySpan<int> prefixTokens)
    {
        using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        sha.AppendData(Encoding.UTF8.GetBytes(modelFingerprint ?? string.Empty));
        sha.AppendData(new byte[] { 0 });
        sha.AppendData(MemoryMarshal.AsBytes(prefixTokens));
        return Convert.ToHexString(sha.GetHashAndReset())[..24].ToLowerInvariant() + Extension;
    }

    /// <summary>Bytes on disk for this model, for a Models list that shows what a model costs.</summary>
    public long TotalBytes()
    {
        try
        {
            if (!System.IO.Directory.Exists(Directory)) return 0;
            return new DirectoryInfo(Directory).EnumerateFiles("*" + Extension).Sum(f => f.Length);
        }
        catch (Exception) { return 0; }
    }

    public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out Stream payload)
    {
        payload = null!;
        string path = PathFor(modelFingerprint, prefixTokens);
        FileStream? stream = null;
        try
        {
            if (!File.Exists(path))
                return false;
            stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 20, FileOptions.SequentialScan);
            if (!HeaderMatches(stream, Identity(modelFingerprint), prefixTokens))
            {
                stream.Dispose();
                stream = null;
                _log.LogWarning("prefix checkpoint {File} does not describe this model and prefix; deleting it", path);
                TryDelete(path);
                return false;
            }
            try { File.SetLastAccessTimeUtc(path, DateTime.UtcNow); } catch (Exception) { /* LRU only */ }
            payload = stream;
            stream = null;
            return true;
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "prefix checkpoint {File} could not be read; deleting it", path);
            stream?.Dispose();
            TryDelete(path);
            return false;
        }
    }

    public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<Stream> writePayload)
    {
        ArgumentNullException.ThrowIfNull(writePayload);
        string path = PathFor(modelFingerprint, prefixTokens);
        string temp = path + ".tmp-" + Guid.NewGuid().ToString("N");
        try
        {
            System.IO.Directory.CreateDirectory(Directory);
            using (var stream = new FileStream(temp, FileMode.CreateNew, FileAccess.Write, FileShare.None, 1 << 20))
            {
                WriteHeader(stream, Identity(modelFingerprint), prefixTokens);
                writePayload(stream);
                stream.Flush(flushToDisk: true);
            }
            File.Move(temp, path, overwrite: true);
            EvictBeyond(MaxFilesPerModel, keep: path);
            return true;
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "prefix checkpoint {File} could not be saved", path);
            TryDelete(temp);
            return false;
        }
    }

    /// <summary>Remove everything kept for this model (the model is being deleted).</summary>
    public void Clear()
    {
        try
        {
            if (System.IO.Directory.Exists(Directory))
                System.IO.Directory.Delete(Directory, recursive: true);
        }
        catch (Exception ex)
        {
            _log.LogWarning(ex, "prefix checkpoints under {Directory} could not be removed", Directory);
        }
    }

    private static void WriteHeader(Stream stream, string modelFingerprint, ReadOnlySpan<int> prefixTokens)
    {
        var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);
        w.Write(Magic);
        w.Write(Version);
        w.Write(modelFingerprint ?? string.Empty);
        w.Write(prefixTokens.Length);
        w.Flush();
        stream.Write(MemoryMarshal.AsBytes(prefixTokens));
    }

    private static bool HeaderMatches(Stream stream, string modelFingerprint, ReadOnlySpan<int> prefixTokens)
    {
        var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        if (r.ReadUInt32() != Magic || r.ReadInt32() != Version)
            return false;
        if (!string.Equals(r.ReadString(), modelFingerprint ?? string.Empty, StringComparison.Ordinal))
            return false;
        int count = r.ReadInt32();
        if (count != prefixTokens.Length || count < 0)
            return false;
        var stored = new int[count];
        stream.ReadExactly(MemoryMarshal.AsBytes(stored.AsSpan()));
        return prefixTokens.SequenceEqual(stored);
    }

    private void EvictBeyond(int keepCount, string keep)
    {
        try
        {
            var files = new DirectoryInfo(Directory).EnumerateFiles("*" + Extension)
                .OrderByDescending(f => string.Equals(f.FullName, keep, StringComparison.Ordinal))
                .ThenByDescending(f => f.LastAccessTimeUtc > f.LastWriteTimeUtc ? f.LastAccessTimeUtc : f.LastWriteTimeUtc)
                .ToList();
            foreach (FileInfo old in files.Skip(keepCount))
            {
                _log.LogInformation("removing the least recently used prefix checkpoint {File} ({MB:F0} MB)", old.Name, old.Length / 1048576.0);
                TryDelete(old.FullName);
            }
            // Temporary files a killed process left behind are never valid.
            foreach (FileInfo stray in new DirectoryInfo(Directory).EnumerateFiles("*.tmp-*"))
                if (stray.LastWriteTimeUtc < DateTime.UtcNow.AddMinutes(-10))
                    TryDelete(stray.FullName);
        }
        catch (Exception) { /* housekeeping only */ }
    }

    /// <summary>Temporary files a killed process left behind are never valid; remove the
    /// ones old enough that no writer can still own them.</summary>
    private void SweepStrayTemporaries()
    {
        try
        {
            if (!System.IO.Directory.Exists(Directory)) return;
            foreach (FileInfo stray in new DirectoryInfo(Directory).EnumerateFiles("*.tmp-*"))
                if (stray.LastWriteTimeUtc < DateTime.UtcNow.AddMinutes(-10))
                    TryDelete(stray.FullName);
        }
        catch (Exception) { /* housekeeping only */ }
    }

    /// <summary>
    /// Remove the checkpoint directories of models the catalog no longer has, and say
    /// how many bytes that freed. The same rule as the model store's orphan sweep: a
    /// directory nobody can reach through the Models list has no delete button.
    /// </summary>
    public static long SweepOrphans(string prefixCacheRoot, Func<string, bool> catalogHas, ILogger? log = null)
    {
        long freed = 0;
        try
        {
            if (!System.IO.Directory.Exists(prefixCacheRoot)) return 0;
            foreach (string dir in System.IO.Directory.EnumerateDirectories(prefixCacheRoot))
            {
                string id = Path.GetFileName(dir);
                if (catalogHas(id)) continue;
                long bytes = new DirectoryInfo(dir).EnumerateFiles("*", SearchOption.AllDirectories).Sum(f => f.Length);
                System.IO.Directory.Delete(dir, recursive: true);
                freed += bytes;
                (log ?? NullLogger.Instance).LogInformation("removed the prefix checkpoints of {Id}, which is no longer in the catalog ({MB:F0} MB)", id, bytes / 1048576.0);
            }
        }
        catch (Exception ex)
        {
            (log ?? NullLogger.Instance).LogWarning(ex, "sweeping orphaned prefix checkpoints under {Root} failed", prefixCacheRoot);
        }
        return freed;
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); } catch (Exception) { }
    }
}
