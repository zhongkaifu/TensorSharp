// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Security.Cryptography;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;

// Model-free schedule contract, also linked into ContractTests. Output names are
// unique portable filenames; reference aliases carry the canonical token history.
internal static class ProbeSchedule
{
    public static Dictionary<string, ProbeCase> Validate(ProbeCase[] cases)
    {
        if (cases == null || cases.Length == 0) throw new ArgumentException("Case schedule is empty");
        var byName = new Dictionary<string, ProbeCase>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in cases)
        {
            if (item == null) throw new ArgumentException("Case schedule contains null");
            ValidateName(item.Name);
            if (!byName.TryAdd(item.Name, item)) throw new ArgumentException("Duplicate case name: " + item.Name);
            if (item.ReferenceCase != null) ValidateName(item.ReferenceCase);
            if (item.TargetTokens is <= 0) throw new ArgumentException("target_tokens must be positive");
            if (item.PromptTokens is { Length: 0 }) throw new ArgumentException("Empty prompt token history");
            if (item.PromptTokens?.Any(token => token < 0) == true || item.GeneratedTokens?.Any(token => token < 0) == true)
                throw new ArgumentException("Token IDs must be nonnegative");
        }
        return byName;
    }

    public static void ValidateReferences(ProbeCase[] cases, IReadOnlyDictionary<string, ProbeCase>? external)
    {
        var earlier = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var item in cases)
        {
            string? referenceName = external != null ? item.ReferenceCase ?? item.Name : item.ReferenceCase;
            if (referenceName != null)
            {
                if (external != null)
                {
                    if (!external.TryGetValue(referenceName, out var reference))
                        throw new ArgumentException("Unknown external reference case: " + referenceName);
                    RequireRecordedHistory(reference);
                }
                else if (!earlier.Contains(referenceName))
                    throw new ArgumentException("In-process reference must name an earlier completed case: " + referenceName);
            }
            else if (item.PromptTokens == null && string.IsNullOrWhiteSpace(item.Prompt))
                throw new ArgumentException("Missing prompt for case " + item.Name);
            earlier.Add(item.Name);
        }
    }

    public static void UseReferenceHistory(ProbeCase item, ProbeCase reference)
    {
        RequireRecordedHistory(reference);
        if (item.PromptTokens != null && !item.PromptTokens.SequenceEqual(reference.PromptTokens!))
            throw new ArgumentException("Reference prompt tokens differ for " + item.Name);
        if (item.TargetTokens is int target && target != reference.PromptTokens!.Length)
            throw new ArgumentException("Reference prompt length differs for " + item.Name);
        if (item.GeneratedTokens != null && !item.GeneratedTokens.SequenceEqual(reference.GeneratedTokens!))
            throw new ArgumentException("Reference generated tokens differ for " + item.Name);
        if (item.Prompt != null && item.Prompt != reference.Prompt)
            throw new ArgumentException("Reference prompt text differs for " + item.Name);
        item.Prompt = reference.Prompt;
        item.PromptTokens = (int[])reference.PromptTokens!.Clone();
        item.GeneratedTokens = (int[])reference.GeneratedTokens!.Clone();
        item.TargetTokens = reference.PromptTokens.Length;
    }

    public static string TokenHash(IEnumerable<int> tokens)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        Span<byte> bytes = stackalloc byte[sizeof(int)];
        foreach (int token in tokens)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes, token);
            hash.AppendData(bytes);
        }
        return Convert.ToHexString(hash.GetHashAndReset()).ToLowerInvariant();
    }

    public static LogitArtifactBudget PlanLogitArtifacts(int rows, int vocabularySize, long availableBytes, double headroomGiB)
    {
        if (rows <= 0 || vocabularySize <= 0 || availableBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(rows), "Rows/vocabulary must be positive and available bytes nonnegative");
        if (!double.IsFinite(headroomGiB) || headroomGiB < 0)
            throw new ArgumentOutOfRangeException(nameof(headroomGiB), "Disk headroom must be finite and nonnegative");
        try
        {
            long logitBytes = checked((long)rows * vocabularySize * sizeof(float));
            long headroomBytes = checked((long)Math.Ceiling(headroomGiB * 1024d * 1024d * 1024d));
            long requiredBytes = checked(logitBytes + headroomBytes);
            return new(logitBytes, headroomBytes, availableBytes, requiredBytes);
        }
        catch (OverflowException)
        {
            throw new ArgumentOutOfRangeException(nameof(headroomGiB), "Logit artifact byte budget exceeds Int64 capacity");
        }
    }

    public static long AvailableDiskBytes(string outputDirectory)
    {
        string path = Path.GetFullPath(outputDirectory);
        if (!Directory.Exists(path)) throw new DirectoryNotFoundException(path);
        if (OperatingSystem.IsWindows())
        {
            // Query the actual path, not its drive letter: a junction can lead to
            // a different volume. Available bytes also respect per-user quotas.
            if (!GetDiskFreeSpaceExW(path, out ulong available, out _, out _))
                throw new IOException("Cannot inspect logit output disk: " + path, new Win32Exception(Marshal.GetLastWin32Error()));
            return checked((long)available);
        }
        // On Unix DriveInfo queries statfs at the existing path. Passing '/' would
        // incorrectly budget the root volume when artifacts live on another mount.
        return new DriveInfo(path).AvailableFreeSpace;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true, ExactSpelling = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GetDiskFreeSpaceExW(string directory, out ulong available, out ulong total, out ulong free);

    private static void RequireRecordedHistory(ProbeCase reference)
    {
        if (reference.PromptTokens is not { Length: > 0 } || reference.GeneratedTokens is not { Length: > 0 })
            throw new ArgumentException("Reference has no complete recorded token history: " + reference.Name);
    }

    private static void ValidateName(string name)
    {
        if (string.IsNullOrWhiteSpace(name) || name is "." or ".." ||
            !Regex.IsMatch(name, @"\A[A-Za-z0-9_.-]+\z") || name.EndsWith('.') ||
            Regex.IsMatch(name.Split('.')[0], @"\A(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])\z", RegexOptions.IgnoreCase))
            throw new ArgumentException("Invalid case name: " + name);
    }
}

internal sealed record LogitArtifactBudget(long LogitBytes, long HeadroomBytes, long AvailableBytes, long RequiredBytes)
{
    public bool Fits => AvailableBytes >= RequiredBytes;
}

internal sealed class ProbeCase
{
    public string Name { get; set; } = "";
    public string? ReferenceCase { get; set; }
    public string? Prompt { get; set; }
    public int[]? PromptTokens { get; set; }
    public int? TargetTokens { get; set; }
    public int[]? GeneratedTokens { get; set; }
}
