// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;

internal static class ProbeEvidenceValidation
{
    internal static ValidatedReference LoadReferenceRequests(string metricsPath, JsonElement root, int steps, int width)
    {
        if (steps < 1 || width < 2) throw new InvalidDataException("Invalid reference dimensions");
        var cases = new Dictionary<string, float[][]>(StringComparer.Ordinal);
        var files = new List<ReferenceBinaryEvidence>();
        foreach (var request in root.GetProperty("requests").EnumerateArray())
        {
            string name = request.GetProperty("caseName").GetString() ?? throw new InvalidDataException("Missing reference case name");
            string filename = request.GetProperty("binaryFile").GetString() ?? throw new InvalidDataException("Missing reference binary name");
            if (string.IsNullOrWhiteSpace(name) || string.IsNullOrWhiteSpace(filename) || filename is "." or ".." ||
                filename.IndexOfAny(['/', '\\']) >= 0 || Path.GetFileName(filename) != filename)
                throw new InvalidDataException("Reference binary must be a sibling file with a valid case name");
            string path = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(metricsPath))!, filename);
            long expectedBytes = checked((long)steps * width * sizeof(float));
            var rows = new float[steps][];
            using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            using (var stream = File.OpenRead(path))
            {
                if (stream.Length != expectedBytes) throw new InvalidDataException("Reference binary dimensions differ");
                for (int step = 0; step < steps; step++)
                {
                    rows[step] = new float[width];
                    Span<byte> bytes = MemoryMarshal.AsBytes(rows[step].AsSpan());
                    stream.ReadExactly(bytes);
                    hash.AppendData(bytes);
                    if (rows[step].Any(x => !float.IsFinite(x))) throw new InvalidDataException("Reference contains nonfinite values");
                }
                if (stream.Length != expectedBytes) throw new InvalidDataException("Reference changed while reading");
            }
            string digest = Convert.ToHexString(hash.GetHashAndReset());
            if (!digest.Equals(request.GetProperty("binarySha256").GetString(), StringComparison.OrdinalIgnoreCase))
                throw new InvalidDataException("Reference binary hash mismatch");
            if (cases.TryGetValue(name, out var prior))
            {
                // Check every duplicate independently of the summary's allExact flag.
                // Float equality would hide signed-zero changes; compare the exact bits.
                for (int step = 0; step < steps; step++)
                    if (!MemoryMarshal.AsBytes(rows[step].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(prior[step].AsSpan())))
                        throw new InvalidDataException("Reference has unequal repeated histories for case " + name);
            }
            else cases.Add(name, rows);
            files.Add(new(name, path, digest, expectedBytes));
        }
        return new(cases, files);
    }

    internal static string[] UnverifiedCases(IEnumerable<string> sequence, ISet<string> comparedCases)
        => sequence.Distinct(StringComparer.Ordinal).Where(name => !comparedCases.Contains(name)).ToArray();

    internal static bool? StrictPassed(bool complete, bool recordOnly, bool allFinite, bool allExact,
        int comparisonCount, IReadOnlyCollection<string> unverifiedCases)
        => complete && !recordOnly ? allFinite && allExact && comparisonCount > 0 && unverifiedCases.Count == 0 : null;
}

internal sealed record ReferenceBinaryEvidence(string CaseName, string Path, string Sha256, long Bytes);
internal sealed record ValidatedReference(Dictionary<string, float[][]> Cases, List<ReferenceBinaryEvidence> Files);
