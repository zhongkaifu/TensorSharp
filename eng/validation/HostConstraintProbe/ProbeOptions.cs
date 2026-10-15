// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;

internal sealed record ProbeOptions(string Output, long AllocationBytes, long? ExpectedCommitCeiling)
{
    internal const long MaximumAllocationBytes = 64L << 20;

    public static ProbeOptions Parse(string[] args)
    {
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        var allowed = new HashSet<string>(["--output", "--allocation-bytes", "--expect-commit-ceiling-bytes"], StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 == args.Length || !allowed.Contains(args[i]) || !options.TryAdd(args[i], args[i + 1]))
                throw new ArgumentException("Expected unique --output PATH, optional --allocation-bytes N and --expect-commit-ceiling-bytes N only; no model or device options.");
        }
        if (!options.TryGetValue("--output", out string? output) || string.IsNullOrWhiteSpace(output))
            throw new ArgumentException("A fresh --output JSON path is required.");
        long ParseNumber(string value)
        {
            if (value.Length == 0 || value.Any(c => c is < '0' or > '9')
                || !long.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out long bytes))
                throw new ArgumentException("Byte counts must be nonnegative ASCII decimal Int64 values.");
            return bytes;
        }
        long allocation = options.TryGetValue("--allocation-bytes", out string? bytes) ? ParseNumber(bytes) : 0;
        if (allocation > MaximumAllocationBytes) throw new ArgumentOutOfRangeException(nameof(args), "Probe allocation is capped at 64 MiB.");
        long? ceiling = options.TryGetValue("--expect-commit-ceiling-bytes", out string? limit) ? ParseNumber(limit) : null;
        if (ceiling is <= 0) throw new ArgumentOutOfRangeException(nameof(args), "Expected commit ceiling must be positive.");
        return new(Path.GetFullPath(output), allocation, ceiling);
    }
}
