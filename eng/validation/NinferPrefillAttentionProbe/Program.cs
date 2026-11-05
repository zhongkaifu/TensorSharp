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
    private static void SelfTest()
    {
        const string lower = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
        string upper = lower.ToUpperInvariant();
        Need(S.Matches(lower, upper) && S.Matches(upper, lower) && S.Matches(lower, lower), "SHA hexadecimal case changed byte identity.");
        Need(!S.Matches(lower, "1" + lower[1..]), "Different SHA bytes were admitted.");
        foreach (string malformed in new[] { lower[..63], lower + "0", "g" + lower[1..], " " + lower[1..] })
        {
            bool rejected = false; try { _ = S.Matches(lower, malformed); } catch (ArgumentException) { rejected = true; }
            Need(rejected, "Malformed SHA identity was admitted.");
        }
        var cases = R.Cases(); Need(cases.Length == 8 && cases.Select(x => x.Name).Distinct().Count() == 8, "Eight unique fixtures required.");
        long outputValues = 0;
        foreach (var c in cases)
        {
            var f = R.Make(c); outputValues += f.Query.Length;
            Need(f.Query.Length == c.Rows * 6144 && f.LogicalK.Length == c.Keys * 1024 && f.Gate.Length == f.Query.Length, "Fixture span differs.");
            Need(f.Query.Select((x, i) => BitConverter.SingleToInt32Bits(R.Float(x)) == BitConverter.SingleToInt32Bits(f.QueryF32[i])).All(x => x), "Represented BF16 query differs.");
            Need(CudaNinferPrefillAttentionModule.SplitCount(c.Rows, 170) == c.ExpectedSplits, "Pinned source split policy differs.");
            var p = CudaNinferPrefillAttentionModule.Plan(c.Rows, c.Keys, (c.Keys + 255) / 256 + 1, 170);
            Need(p.Splits == c.ExpectedSplits && p.Bytes % 256 == 0 && p.Pages == (c.Keys + 63) / 64, "Workspace source contract differs.");
            foreach (int page in new[] { 64, 256 })
            {
                int n = (c.Keys + page - 1) / page + 1; var perm = Permutation(n, 0x65e19bd3U);
                Need(perm.Distinct().Count() == n && perm.All(x => x >= 0 && x < n), "Cache permutation is not a bijection.");
            }
            Need(Pass(CheckCore(Bytes(f.IdealCore.Select(x => R.Bf((float)x)).ToArray()), f)), "Independent ideal BF16 narrowing violates its fixed gate.");
            Need(Pass(CheckGated(Bytes(f.IdealGated), f)), "Independent gated ideal violates its fixed gate.");
            if (c.Name == "constant-t64-p0")
            {
                var bad = Bytes(f.IdealGated); BitConverter.GetBytes(float.NaN).CopyTo(bad, 0);
                var rejected = CheckGated(bad, f); Need(!Pass(rejected) && rejected.Nonfinite == 1, "Nonfinite GPU output was not rejected.");
                _ = JsonSerializer.Serialize(rejected, Json); // Failure evidence must remain serializable.
            }
        }
        Need(outputValues == 11618304, "Fixed full-output coverage differs.");
        CudaNinferPrefillAttentionModule.ValidatePlaneOffsets(KOffset, VOffset);
        Need(!CudaNinferPrefillAttentionModule.Supports(64, 2, false) && !CudaNinferPrefillAttentionModule.Supports(64, 1, true) && !CudaNinferPrefillAttentionModule.Supports(64, 1, false, false), "Unsupported batching/Verify/padded rows accepted.");
        Console.WriteLine(JsonSerializer.Serialize(new { CPU = "PASS", shaIdentity = "PASS", malformedDigestCases = 4, cases = 8, lifecycleRecordsPlanned = 16, poisonPassesPlanned = 48, fullOutputValuesPerPass = outputValues, coreHostSubmissionsPlanned = 144, actualKernelsPlanned = 462, GPU = "NOT_RUN", modelQuality = "NOT_RUN", performance = "NOT_RUN" }));
    }
    internal static int Main(string[] args)
    {
        string? output = null; bool owned = false; var records = new List<Dictionary<string, object?>>(); var files = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-original-bf16-prefill-paged-component/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["performance"] = "NOT_RUN", ["records"] = records, ["files"] = files };
        try
        {
            Need(BitConverter.IsLittleEndian, "Little-endian fixture bytes required.");
            if (args.SequenceEqual(new[] { "--self-test" })) { SelfTest(); return 0; }
            if (args.Length == 0) { Console.WriteLine("PLAN: --execute --module NEW_IMAGE --expect-sha256 SHA --legacy-module OLD_IMAGE --expect-legacy-sha256 SHA --out FRESH_IGNORED_DIR; --self-test is CPU-only."); return 0; }
            Need(args.Length == 11 && args[0] == "--execute" && args[1] == "--module" && args[3] == "--expect-sha256" && args[5] == "--legacy-module" && args[7] == "--expect-legacy-sha256" && args[9] == "--out", "Exact explicit execution arguments required.");
            string Pinned(string path, string sha)
            {
                Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path && File.Exists(path), "Canonical existing module path required.");
                Need(S.Matches(FileSha(path), sha), "Pinned image bytes differ."); return path;
            }
            string image = Pinned(args[2], args[4]), legacyImage = Pinned(args[6], args[8]), imageSha = args[4].ToLowerInvariant(), legacySha = args[8].ToLowerInvariant();
            output = args[10]; Fresh(output); Directory.CreateDirectory(output); owned = true;
            state["modulePath"] = image; state["moduleSha256"] = imageSha; state["legacyModulePath"] = legacyImage; state["legacyModuleSha256"] = legacySha;
            state["toolAssemblySha256"] = FileSha(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileSha(typeof(CudaBackend).Assembly.Location);
            state["policy"] = CudaNinferPrefillAttentionModule.Policy;
            state["scope"] = "Synthetic represented-BF16 causal Q24/KV4/HD256, original64-page versus ownedTS256-page core and partial bits, complete independentFP64 envelope, exactconstant/sentinel controls, F32pack/gate and oldGPU diagnostics; defaultstream only. No model/quality/performance qualification.";
            state["fixedFp64Envelope"] = new { absolute = R.Absolute, relative = R.Relative, description = "Predeclared prior attention envelope for these bounded BF16 fixtures; not universal mathematical proof and not old/new arithmetic equivalence." };
            state["expected"] = new { cases = 8, lifecycleRecords = 16, passes = 48, coreHostSubmissions = 144, nativeKernelSubmissions = 462 };
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            var repeat = new Dictionary<string, string>(); CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError();
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError();
                CudaDriverApi.cuDeviceGetAttribute(out int sms, 16, device).ThrowOnError(); Need(sms == 170, "This fixed split-policy campaign requires the pinned170SM GPU.");
                byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError(); state["device"] = new { ordinal = device, name = Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver, sms }; state["GPU"] = "EXECUTED";
                foreach (var c in R.Cases())
                {
                    // One case, including long-context KV and oracle, is live at a time.
                    var f = R.Make(c); Directory.CreateDirectory(Path.Combine(output, c.Name));
                    Capture(c.Name + "/q.bf16.bin", Bytes(f.Query), output, files); Capture(c.Name + "/q.f32.bin", Bytes(f.QueryF32), output, files); Capture(c.Name + "/gate.f32.bin", Bytes(f.Gate), output, files);
                    Capture(c.Name + "/logical-k.bf16.bin", Bytes(f.LogicalK), output, files); Capture(c.Name + "/logical-v.bf16.bin", Bytes(f.LogicalV), output, files);
                    Capture(c.Name + "/ideal-core.f64.bin", Bytes(f.IdealCore), output, files); Capture(c.Name + "/ideal-gated.f32.bin", Bytes(f.IdealGated), output, files);
                    for (int lifetime = 0; lifetime < 2; lifetime++)
                    {
                        using var module = new CudaNinferPrefillAttentionModule(image); using var legacy = new LegacyAttention(legacyImage);
                        Need(S.Matches(module.ImageSha256, imageSha) && S.Matches(legacy.ImageSha256, legacySha), "Actually loaded image differs from its pinned bytes; use directfatbin paths.");
                        state["kernelSymbols"] = module.KernelSymbols;
                        RunCase(module, legacy, f, lifetime, output, records, repeat, files, Checkpoint);
                    }
                    Need(files.Sum(x => new FileInfo(Path.Combine(output, x.Key)).Length) < 1900L * 1024 * 1024, "Fixed evidence budget exceeded.");
                }
            }
            Need(records.Count == 16 && records.Select(x => (string)x["id"]!).Distinct().Count() == 16 && records.All(x => (bool)x["passed"]! && ((List<object>)x["passes"]!).Count == 3), "All eight cases/two lifetimes/three poisons are mandatory.");
            Need(repeat.Count == 40 && files.Count == 296, "Complete repeat/output/input evidence coverage differs.");
            Need(files.All(x => FileSha(Path.Combine(output, x.Key)) == x.Value), "Retained evidence bytes changed before completion.");
            Need(S.Matches(FileSha(image), imageSha) && S.Matches(FileSha(legacyImage), legacySha), "Pinned image changed during execution.");
            state["complete"] = true; state["passed"] = true; state["actualKernelSubmissions"] = 462; state["allRawEvidenceRehashed"] = true; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, records = 16, poisonPasses = 48, modelQuality = "NOT_RUN", performance = "NOT_RUN" })); return 0;
        }
        catch (Exception e) { state["passed"] = false; state["error"] = e.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
