// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using System.Security.Cryptography;

namespace GemmaScalarRingProbe;

internal sealed record ExactComparison(int Count, int ReferenceNonfinite, int ActualNonfinite,
    int DifferentBits, int? FirstDifferentIndex, double? MaximumAbsoluteDifference,
    string ReferenceSha256, string ActualSha256, bool Passed)
{
    internal static ExactComparison Compare(ReadOnlySpan<float> reference, ReadOnlySpan<float> actual)
    {
        if (reference.Length == 0 || reference.Length != actual.Length)
            throw new ArgumentException("Comparisons require equally sized, nonempty rows.");
        int rn = 0, an = 0, different = 0;
        int? first = null;
        double maximum = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            if (!float.IsFinite(reference[i])) rn++;
            if (!float.IsFinite(actual[i])) an++;
            if (BitConverter.SingleToInt32Bits(reference[i]) != BitConverter.SingleToInt32Bits(actual[i]))
            { different++; first ??= i; }
            if (float.IsFinite(reference[i]) && float.IsFinite(actual[i]))
                maximum = Math.Max(maximum, Math.Abs((double)reference[i] - actual[i]));
        }
        static string Hash(ReadOnlySpan<float> row) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(row))).ToLowerInvariant();
        return new(reference.Length, rn, an, different, first, rn + an == 0 ? maximum : null,
            Hash(reference), Hash(actual), rn == 0 && an == 0 && different == 0);
    }

    internal static int SelfTest()
    {
        int checks = 0;
        void Require(bool value) { checks++; if (!value) throw new InvalidOperationException($"Self-test {checks} failed."); }
        Require(Compare([1, 2], [1, 2]).Passed);
        var mismatch = Compare([1, 2], [1, 3]);
        Require(!mismatch.Passed && mismatch.DifferentBits == 1 && mismatch.FirstDifferentIndex == 1 && mismatch.MaximumAbsoluteDifference == 1);
        Require(!Compare([0f], [-0f]).Passed);
        Require(!Compare([float.NaN], [float.NaN]).Passed);
        Require(!Compare([float.PositiveInfinity], [float.PositiveInfinity]).Passed);
        Require(!Compare([1], [float.NegativeInfinity]).Passed);
        Require(Compare([1], [float.NaN]).MaximumAbsoluteDifference == null);
        Require(Compare([float.MaxValue], [-float.MaxValue]).MaximumAbsoluteDifference > float.MaxValue);
        foreach (var pair in new[] { (Array.Empty<float>(), Array.Empty<float>()), (new[] { 1f }, new[] { 1f, 2f }) })
        {
            bool rejected = false;
            try { Compare(pair.Item1, pair.Item2); } catch (ArgumentException) { rejected = true; }
            Require(rejected);
        }
        Console.WriteLine($"{{\"passed\":{checks},\"failed\":0,\"skipped\":0,\"scope\":\"CPU comparison helper only; no model or GPU execution\"}}");
        return 0;
    }
}
