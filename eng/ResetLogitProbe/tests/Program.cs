// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Synthetic evidence only: no model assemblies, model files, or native backend.
using System.Security.Cryptography;
using System.Text.Json;

var tests = new (string Name, Action Test)[]
{
    ("Valid duplicates retain every binary as evidence", ValidDuplicates),
    ("Corrupt duplicate is rejected despite allExact=true", CorruptDuplicate),
    ("Missing duplicate is rejected despite allExact=true", MissingDuplicate),
    ("Different duplicate with its own valid hash is rejected", DifferentDuplicate),
    ("Signed-zero duplicate differs at the bit level", SignedZeroDuplicate),
    ("Wrong binary dimensions are rejected", WrongDimensions),
    ("Nonfinite binary values are rejected", NonfiniteValues),
    ("Uncompared requested case prevents strict pass", UncomparedCase),
    ("Record-only result stays null", RecordOnly),
    ("Every requested case compared permits strict pass", FullyCompared),
    ("Zero comparisons cannot pass", ZeroComparisons)
};
int failures = 0;
foreach (var (name, test) in tests)
{
    try { test(); Console.WriteLine("PASS " + name); }
    catch (Exception exception) { failures++; Console.Error.WriteLine("FAIL " + name + ": " + exception); }
}
Console.WriteLine($"{tests.Length - failures}/{tests.Length} synthetic evidence tests passed");
return failures == 0 ? 0 : 1;

static void ValidDuplicates()
{
    using var fixture = new EvidenceFixture();
    var first = fixture.Write("p32", "first.f32", [1, 2, 3, 4, 5, 6]);
    var second = fixture.Write("p32", "second.f32", [1, 2, 3, 4, 5, 6]);
    using var metrics = fixture.Metrics(first, second);
    var result = ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3);
    Check(result.Cases.Count == 1 && result.Cases.ContainsKey("p32"), "Expected one reference case");
    Check(result.Cases["p32"].Length == 2 && result.Cases["p32"][1].SequenceEqual(new float[] { 4, 5, 6 }), "Rows changed");
    Check(result.Files.Count == 2, "Duplicate request evidence was discarded");
    Check(result.Files.All(file => file.CaseName == "p32" && file.Bytes == 24), "Wrong evidence identity/length");
    Check(result.Files.Select(file => Path.GetFileName(file.Path)).Order().SequenceEqual(new[] { "first.f32", "second.f32" }), "Missing binary evidence");
    Check(result.Files.All(file => file.Sha256 == first.binarySha256), "Wrong evidence hash");
}

static void CorruptDuplicate()
{
    using var fixture = new EvidenceFixture();
    var first = fixture.Write("p32", "first.f32", [1, 2, 3, 4, 5, 6]);
    var second = fixture.Write("p32", "second.f32", [1, 2, 3, 4, 5, 6]);
    // Preserve the recorded hash, then replace only the later binary's bytes.
    fixture.Write("p32", "second.f32", [9, 2, 3, 4, 5, 6]);
    using var metrics = fixture.Metrics(first, second);
    Throws<InvalidDataException>(() => ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3));
}

static void MissingDuplicate()
{
    using var fixture = new EvidenceFixture();
    var first = fixture.Write("p32", "first.f32", [1, 2, 3, 4, 5, 6]);
    var second = fixture.Write("p32", "second.f32", [1, 2, 3, 4, 5, 6]);
    File.Delete(Path.Combine(fixture.Root, second.binaryFile));
    using var metrics = fixture.Metrics(first, second);
    try { ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3); }
    catch (FileNotFoundException) { return; }
    catch (InvalidDataException) { return; }
    throw new Exception("Missing duplicate reference binary was accepted");
}

static void DifferentDuplicate()
{
    using var fixture = new EvidenceFixture();
    var first = fixture.Write("p32", "first.f32", [1, 2, 3, 4, 5, 6]);
    var second = fixture.Write("p32", "second.f32", [1, 2, 3, 4, 5, 7]);
    // Each binary matches its own hash. The aggregate allExact flag is stale.
    using var metrics = fixture.Metrics(first, second);
    Throws<InvalidDataException>(() => ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3));
}

static void SignedZeroDuplicate()
{
    using var fixture = new EvidenceFixture();
    var first = fixture.Write("p32", "first.f32", [0, 2, 3, 4, 5, 6]);
    var second = fixture.Write("p32", "second.f32", [BitConverter.Int32BitsToSingle(unchecked((int)0x80000000)), 2, 3, 4, 5, 6]);
    using var metrics = fixture.Metrics(first, second);
    Throws<InvalidDataException>(() => ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3));
}

static void WrongDimensions()
{
    using var fixture = new EvidenceFixture();
    var shortFile = fixture.Write("p32", "short.f32", [1, 2, 3, 4, 5]);
    using var metrics = fixture.Metrics(shortFile);
    Throws<InvalidDataException>(() => ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3));
}

static void NonfiniteValues()
{
    foreach (float invalid in new[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity })
    {
        using var fixture = new EvidenceFixture();
        var file = fixture.Write("p32", "nonfinite.f32", [1, 2, 3, 4, 5, invalid]);
        using var metrics = fixture.Metrics(file);
        Throws<InvalidDataException>(() => ProbeEvidenceValidation.LoadReferenceRequests(fixture.MetricsPath, metrics.RootElement, 2, 3));
    }
}

static void UncomparedCase()
{
    var missing = ProbeEvidenceValidation.UnverifiedCases(["p32", "p32", "p64"], new HashSet<string>(StringComparer.Ordinal) { "p32" });
    Check(missing.SequenceEqual(new[] { "p64" }), "Expected the unmatched p64 case");
    Check(ProbeEvidenceValidation.StrictPassed(true, false, true, true, 2, missing) == false, "A comparison for p32 cannot establish p64 consistency");
}

static void RecordOnly()
{
    var missing = ProbeEvidenceValidation.UnverifiedCases(["p32", "p32", "p64"], new HashSet<string>(StringComparer.Ordinal) { "p32" });
    Check(ProbeEvidenceValidation.StrictPassed(true, true, true, true, 2, missing) is null, "Record-only capture must not assert a pass/fail");
    Check(ProbeEvidenceValidation.StrictPassed(true, true, true, true, 2, Array.Empty<string>()) is null, "Record-only must stay null even with complete coverage");
}

static void FullyCompared()
{
    var missing = ProbeEvidenceValidation.UnverifiedCases(["p32", "p32", "p64"], new HashSet<string>(StringComparer.Ordinal) { "p32", "p64" });
    Check(missing.Length == 0, "All requested cases have comparisons");
    Check(ProbeEvidenceValidation.StrictPassed(true, false, true, true, 4, missing) == true, "Fully compared exact finite run should pass");
    Check(ProbeEvidenceValidation.StrictPassed(false, false, true, true, 4, missing) is null, "Incomplete evidence must not assert a pass/fail");
    Check(ProbeEvidenceValidation.StrictPassed(true, false, false, true, 4, missing) == false, "Nonfinite result must fail");
    Check(ProbeEvidenceValidation.StrictPassed(true, false, true, false, 4, missing) == false, "Inexact result must fail");
}

static void ZeroComparisons() => Check(ProbeEvidenceValidation.StrictPassed(true, false, true, true, 0, Array.Empty<string>()) == false, "Zero comparisons cannot pass");
static void Check(bool condition, string message) { if (!condition) throw new Exception(message); }
static void Throws<T>(Action action) where T : Exception
{
    try { action(); }
    catch (T) { return; }
    throw new Exception("Expected " + typeof(T).Name);
}

sealed record RequestEvidence(string caseName, string binaryFile, string binarySha256);
sealed class EvidenceFixture : IDisposable
{
    private readonly string parent = Path.TrimEndingDirectorySeparator(Path.GetFullPath(Path.GetTempPath()));
    public string Root { get; }
    public string MetricsPath => Path.Combine(Root, "metrics.json");
    public EvidenceFixture()
    {
        Root = Path.Combine(parent, "tensorsharp-reset-evidence-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Root);
    }
    public RequestEvidence Write(string name, string filename, float[] values)
    {
        string path = Path.Combine(Root, filename);
        using (var file = File.Create(path))
        using (var writer = new BinaryWriter(file))
            foreach (float value in values) writer.Write(value); // BinaryWriter uses little endian.
        using var input = File.OpenRead(path);
        return new(name, filename, Convert.ToHexString(SHA256.HashData(input)));
    }
    public JsonDocument Metrics(params RequestEvidence[] requests) => JsonDocument.Parse(JsonSerializer.Serialize(new
    {
        complete = true, allFinite = true, allExact = true, requests
    }));
    public void Dispose()
    {
        // Delete only the uniquely named temporary directory this fixture created.
        string absolute = Path.GetFullPath(Root);
        var comparer = OperatingSystem.IsWindows() ? StringComparer.OrdinalIgnoreCase : StringComparer.Ordinal;
        if (!comparer.Equals(Path.GetDirectoryName(absolute), parent) ||
            !Path.GetFileName(absolute).StartsWith("tensorsharp-reset-evidence-tests-", StringComparison.Ordinal))
            throw new InvalidOperationException("Refusing cleanup outside the owned temporary directory");
        Directory.Delete(absolute, recursive: true);
    }
}
