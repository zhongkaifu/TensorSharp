// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using R = TensorSharp.Validation.NinferPrefillAttentionReference;
using S = TensorSharp.Validation.ValidationSha256;

internal static partial class Program
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
    private static void Save(string path, object data) { using var f = File.Create(path); JsonSerializer.Serialize(f, data, Json); f.WriteByte(10); }
    private static void Complete(string path, object data)
    {
        byte[] bytes = JsonSerializer.SerializeToUtf8Bytes(data, Json).Concat(new byte[] { 10 }).ToArray();
        NewFile(Path.Combine(path, "completed-result.json"), bytes);
        NewFile(Path.Combine(path, "completed-result.json.sha256"), Encoding.ASCII.GetBytes(Sha(bytes) + "  completed-result.json\n"));
    }
    private static void Fresh(string path)
    {
        Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path, "Canonical absolute output required.");
        var parts = path.Split(Path.DirectorySeparatorChar, StringSplitOptions.RemoveEmptyEntries);
        Need(parts.Contains("artifacts") || Enumerable.Range(0, Math.Max(0, parts.Length - 1)).Any(i => parts[i] == "docs" && parts[i + 1] == "validation"), "Ignored output required.");
        bool Link(string x) => new FileInfo(x).LinkTarget != null || new DirectoryInfo(x).LinkTarget != null;
        Need(!Path.Exists(path) && !Link(path), "Fresh output required.");
        for (string? p = Path.GetDirectoryName(path); p != null; p = Path.GetDirectoryName(p)) Need(!Link(p) && !File.Exists(p), "Aliased output parent refused.");
    }
    private static string Cas(byte[] data, string output, Dictionary<string, string> files)
    {
        string sha = Sha(data), key = "cas/" + sha + ".bin", path = Path.Combine(output, key);
        if (files.ContainsKey(key)) Need(File.ReadAllBytes(path).AsSpan().SequenceEqual(data), "CAS digest collision or changed bytes.");
        else { Need(!File.Exists(path), "Unregistered preexisting CAS object."); NewFile(path, data); files.Add(key, sha); }
        return key;
    }
    private static Dictionary<string, string> FixtureDigests(R.Fixture f)
    {
        var result = new Dictionary<string, string>();
        void Add(string name, byte[] bytes) => result.Add(f.Case.Name + "/" + name, Sha(bytes));
        Add("q.bf16.bin", Bytes(f.Query)); Add("q.f32.bin", Bytes(f.QueryF32)); Add("gate.f32.bin", Bytes(f.Gate));
        Add("logical-k.bf16.bin", Bytes(f.LogicalK)); Add("logical-v.bf16.bin", Bytes(f.LogicalV));
        Add("ideal-core.f64.bin", Bytes(f.IdealCore)); Add("ideal-gated.f32.bin", Bytes(f.IdealGated));
        return result;
    }
    private static void MatchFixture(Dictionary<string, string> expected, Dictionary<string, string> actual)
    {
        Need(actual.Count == 7, "All seven unchanged fixture blobs are mandatory.");
        foreach (var entry in actual) Need(expected.TryGetValue(entry.Key, out var sha) && S.Matches(sha, entry.Value), "Regenerated fixture bytes differ from pinned R20: " + entry.Key);
    }
    private static Dictionary<string, string> ReadCertificate(byte[] bytes, string moduleSha)
    {
        using var document = JsonDocument.Parse(bytes); var root = document.RootElement;
        Need(root.GetProperty("schema").GetString() == "ninfer-original-bf16-prefill-paged-component/1" && root.GetProperty("complete").GetBoolean() && root.GetProperty("passed").GetBoolean(), "Only complete passing R20 fixture certificates are admitted.");
        Need(S.Matches(root.GetProperty("legacyModuleSha256").GetString()!, moduleSha), "R20 source/control module identity differs.");
        var records = root.GetProperty("records").EnumerateArray().ToArray(); var ids = new HashSet<string>();
        Need(records.Length == 16, "R20 must contain every fixture/lifetime.");
        foreach (var record in records)
        {
            Need(record.GetProperty("passed").GetBoolean() && ids.Add(record.GetProperty("id").GetString()!), "Incomplete or duplicated R20 record.");
            var passes = record.GetProperty("passes").EnumerateArray().ToArray(); Need(passes.Length == 3, "R20 poison coverage differs.");
            foreach (var pass in passes) Need(pass.GetProperty("passed").GetBoolean() && pass.GetProperty("originalCoreBitExact").GetBoolean() && pass.GetProperty("originalPartialBitsExact").GetBoolean() && pass.GetProperty("packedQueryBitsExact").GetBoolean() && pass.GetProperty("bridgeCoreBitsExact").GetBoolean(), "R20 certificate contains a failed component gate.");
        }
        foreach (var c in R.Cases()) foreach (int lifetime in new[] { 0, 1 }) Need(ids.Contains($"{c.Name}/life{lifetime}"), "R20 fixture identity missing.");
        var files = root.GetProperty("files").EnumerateObject().ToDictionary(x => x.Name, x => S.Normalize(x.Value.GetString()!));
        Need(files.Count == 296, "R20 complete raw-evidence coverage differs."); return files;
    }
    private static void SelfTest()
    {
        const string lower = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        Need(S.Matches(lower, lower.ToUpperInvariant()) && !S.Matches(lower, "1" + lower[1..]), "SHA case/different-byte proof differs.");
        foreach (string malformed in new[] { lower[..63], lower + "0", "g" + lower[1..] })
        { bool rejected = false; try { _ = S.Matches(lower, malformed); } catch (ArgumentException) { rejected = true; } Need(rejected, "Malformed SHA admitted."); }
        long values = 0; var names = new HashSet<string>();
        foreach (var c in R.Cases())
        {
            Need(names.Add(c.Name), "Duplicate fixture."); var f = R.Make(c); values += f.Gate.Length;
            var hashes = FixtureDigests(f); MatchFixture(hashes, hashes);
            var wrong = new Dictionary<string, string>(hashes); wrong[hashes.First().Key] = new string('0', 64);
            bool rejected = false; try { MatchFixture(hashes, wrong); } catch (InvalidDataException) { rejected = true; } Need(rejected, "Changed fixture digest was admitted.");
            Need(Pass(CheckGated(Bytes(f.IdealGated), f)), "Frozen independent gate changed.");
            int count = (c.Keys + 255) / 256 + 1; var p = Permutation(count, 0x65e19bd3U); Need(p.Distinct().Count() == count, "Physical permutation is not bijective.");
            if (c.Name == "constant-t64-p0")
            {
                var bad = Bytes(f.IdealGated); BitConverter.GetBytes(float.NaN).CopyTo(bad, 0); var n = CheckGated(bad, f);
                Need(!Pass(n) && n.Nonfinite == 1, "Nonfinite output was admitted."); _ = JsonSerializer.Serialize(n, Json);
            }
        }
        Need(names.Count == 8 && values == 11618304, "Fixed output coverage changed.");
        Need((512 * 6 + 127) / 128 == 24 && (512 * 6 + 63) / 64 == 48 && (416 * 6 + 127) / 128 == 20 && (416 * 6 + 63) / 64 == 39, "Measured full/tail scheduling geometry changed.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", cases = 8, regeneratedFixtureBlobsPlanned = 56, lifecycleRecordsPlanned = 16, poisonPassesPlanned = 48, kernelSubmissionsPlanned = 96, fullF32ValuesPerPass = values, GPU = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" }));
    }
    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false;
        var records = new List<Dictionary<string, object?>>(); var files = new Dictionary<string, string>(); var fixtureShas = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-bf16-attention-sameimage-schedule/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records, ["files"] = files, ["reconstructedFixtureSha256"] = fixtureShas };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian fixture bytes required.");
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length > 0 && args[0] == "--timing") return TimingMain(args);
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module OLD_IMAGE --expect-sha256 SHA --fixture-certificate R20_JSON --expect-certificate-sha256 SHA --out FRESH_IGNORED_DIR; --self-test is CPU-only."); return 0; }
            Need(args.Length == 11 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--expect-sha256" && args[5] == "--fixture-certificate" && args[7] == "--expect-certificate-sha256" && args[9] == "--out", "Exact explicit execution arguments required.");
            string Pinned(string path, string sha)
            { Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path && File.Exists(path), "Canonical existing pinned path required."); Need(S.Matches(FileSha(path), sha), "Pinned bytes differ."); return path; }
            string image = Pinned(args[2], args[4]), certificate = Pinned(args[6], args[8]), imageSha = S.Normalize(args[4]), certificateSha = S.Normalize(args[8]);
            byte[] certificateBytes = File.ReadAllBytes(certificate); Need(S.Matches(Sha(certificateBytes), certificateSha), "Certificate changed while opening.");
            var expectedFixtures = ReadCertificate(certificateBytes, imageSha);
            output = args[10]; Fresh(output); Directory.CreateDirectory(output); Directory.CreateDirectory(Path.Combine(output, "cas")); owned = true;
            NewFile(Path.Combine(output, "fixture-certificate.json"), certificateBytes);
            state["modulePath"] = image; state["moduleSha256"] = imageSha; state["fixtureCertificatePath"] = certificate; state["fixtureCertificateSha256"] = certificateSha;
            state["toolAssemblySha256"] = FileSha(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileSha(typeof(CudaBackend).Assembly.Location);
            state["policy"] = "same-ts-ninfer-attn-bf16-f32-image-block128-vs256-v1";
            state["scope"] = "Only256→128 hostthread/grid scheduling differs; same function/image/arguments/arithmetic. Eight frozenR20fixtures reconstructed with56pinnedblobSHAs. FullF32control bits, fixedFP64/analytic, AA/crosslife, guards and inputimmutability are mandatory; defaultstream only. No modelquality/performance proof.";
            state["fixedFp64Envelope"] = new { absolute = R.Absolute, relative = R.Relative };
            state["expected"] = new { cases = 8, records = 16, poisonPasses = 48, kernelSubmissions = 96, fullF32ValuesPerPass = 11618304 };
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            var first = new Dictionary<string, string>(); CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); CudaDriverApi.cuDeviceGetAttribute(out int sms, 16, device).ThrowOnError(); Need(sms == 170, "This campaign requires the pinned170SM GPU.");
                byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError(); state["device"] = new { ordinal = device, name = Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver, sms }; state["GPU"] = "ATTEMPTED";
                foreach (var c in R.Cases())
                {
                    var f = R.Make(c); var hashes = FixtureDigests(f); MatchFixture(expectedFixtures, hashes);
                    foreach (var entry in hashes) fixtureShas.Add(entry.Key, entry.Value); Checkpoint();
                    for (int lifetime = 0; lifetime < 2; lifetime++)
                    {
                        using var module = new LegacyAttention(image); Need(S.Matches(module.ImageSha256, imageSha), "Actually loaded image differs; use directfatbin.");
                        RunCase(module, f, lifetime, output, records, first, files, Checkpoint);
                    }
                    Need(files.Sum(x => new FileInfo(Path.Combine(output, x.Key)).Length) < 256L * 1024 * 1024, "Fixed CAS evidence budget exceeded.");
                }
            }
            Need(records.Count == 16 && records.Select(x => (string)x["id"]!).Distinct().Count() == 16 && records.All(x => (bool)x["passed"]! && ((List<object>)x["passes"]!).Count == 3 && (long)x["kernelSubmissions"]! == 6), "All eight cases/two lifetimes/three poisons are mandatory.");
            foreach (var c in R.Cases()) foreach (int lifetime in new[] { 0, 1 }) Need(records.Any(x => (string)x["id"]! == $"{c.Name}/life{lifetime}"), "Required actual case missing.");
            Need(first.Count == 16 && fixtureShas.Count == 56 && files.Count == first.Values.Distinct().Count() && files.Count <= 8, "Complete fixture/repeat/CAS coverage differs.");
            Need(files.All(x => S.Matches(FileSha(Path.Combine(output, x.Key)), x.Value)), "Retained complete raw CAS bytes changed.");
            Need(S.Matches(FileSha(image), imageSha) && S.Matches(FileSha(certificate), certificateSha) && S.Matches(FileSha(Path.Combine(output, "fixture-certificate.json")), certificateSha), "Pinned image/certificate changed during execution.");
            state["GPU"] = "EXECUTED"; state["complete"] = true; state["passed"] = true; state["actualKernelSubmissions"] = 96; state["allRawEvidenceRehashed"] = true; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, records = 16, poisonPasses = 48, kernelSubmissions = 96, modelQuality = "NOT_RUN", performance = "NOT_RUN" })); return 0;
        }
        catch (Exception e) { state["passed"] = false; state["error"] = e.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
