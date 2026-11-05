// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;

namespace TensorSharp.Validation;

/// <summary>Canonical SHA-256 text identity for validation metadata. This changes
/// hex letter case only; malformed text and different byte digests remain distinct.</summary>
internal static class ValidationSha256
{
    internal static string Normalize(string value)
    {
        ArgumentNullException.ThrowIfNull(value);
        if (value.Length != 64) throw new ArgumentException("SHA-256 requires exactly64 hexadecimal digits.", nameof(value));
        foreach (char c in value) if (!Uri.IsHexDigit(c)) throw new ArgumentException("Invalid SHA-256 hexadecimal digit.", nameof(value));
        return value.ToLowerInvariant();
    }

    internal static bool Matches(string left, string right)
        => string.Equals(Normalize(left), Normalize(right), StringComparison.Ordinal);
}
