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
    private sealed record Qualification(Dictionary<string, string> FixtureShas, Dictionary<string, string> ControlFiles, Dictionary<string, string> RawFiles);
    private static Qualification ReadQualification(byte[] bytes, string folder, string moduleSha)
    {
        using var document = JsonDocument.Parse(bytes); var root = document.RootElement;
        Need(root.GetProperty("schema").GetString() == "ninfer-bf16-attention-sameimage-schedule/1" && root.GetProperty("complete").GetBoolean() && root.GetProperty("passed").GetBoolean()
            && root.GetProperty("actualKernelSubmissions").GetInt32() == 96, "Only the complete96-launch component certificate admits timing.");
        Need(S.Matches(root.GetProperty("moduleSha256").GetString()!, moduleSha), "Timing and qualified component images differ.");
        var files = root.GetProperty("files").EnumerateObject().ToDictionary(x => x.Name, x => S.Normalize(x.Value.GetString()!));
        Need(files.Count is > 0 and <= 8, "Component full-body CAS coverage differs.");
        foreach (var entry in files)
        {
            Need(entry.Key == "cas/" + entry.Value + ".bin", "Noncanonical CAS key refused.");
            Need(S.Matches(FileSha(Path.Combine(folder, entry.Key)), entry.Value), "Qualified complete raw bytes changed.");
        }
        var records = root.GetProperty("records").EnumerateArray().ToArray(); Need(records.Length == 16, "Qualified lifecycle coverage differs.");
        var ids = new HashSet<string>(); var controls = new Dictionary<string, string>();
        foreach (var record in records)
        {
            string id = record.GetProperty("id").GetString()!, name = record.GetProperty("case").GetString()!; int life = record.GetProperty("lifetime").GetInt32();
            var c = R.Cases().Single(x => x.Name == name);
            Need(life is 0 or 1 && id == $"{name}/life{life}" && ids.Add(id) && record.GetProperty("passed").GetBoolean() && record.GetProperty("kernelSubmissions").GetInt32() == 6
                && record.GetProperty("rows").GetInt32() == c.Rows && record.GetProperty("past").GetInt32() == c.Past, "Qualified actual case identity differs.");
            var passes = record.GetProperty("passes").EnumerateArray().ToArray(); Need(passes.Length == 3, "Qualified poison coverage differs.");
            for (int p = 0; p < 3; p++)
            {
                var pass = passes[p]; Need(pass.GetProperty("pass").GetInt32() == p && pass.GetProperty("poisonByte").GetInt32() == Poisons[p] && pass.GetProperty("passed").GetBoolean()
                    && pass.GetProperty("fullF32BitsEqual").GetBoolean() && pass.GetProperty("aaAndCrossLifetimeBits").GetBoolean()
                    && pass.GetProperty("immutableInputs").GetBoolean() && pass.GetProperty("allAllocationGuards").GetBoolean(), "Qualified component contains a failed gate.");
                string control = pass.GetProperty("controlFile").GetString()!, candidate = pass.GetProperty("candidateFile").GetString()!;
                Need(control == candidate && files.ContainsKey(control), "Qualified output bodies differ or are missing.");
                if (controls.TryGetValue(name, out var previous)) Need(previous == control, "Qualified case changed acrosslifetimes/poisons."); else controls.Add(name, control);
            }
        }
        foreach (var c in R.Cases()) foreach (int life in new[] { 0, 1 }) Need(ids.Contains($"{c.Name}/life{life}"), "Qualified case is missing.");
        Need(files.Count == controls.Values.Distinct().Count(), "Unreferenced or omitted qualified CAS data.");
        var fixtures = root.GetProperty("reconstructedFixtureSha256").EnumerateObject().ToDictionary(x => x.Name, x => S.Normalize(x.Value.GetString()!));
        Need(fixtures.Count == 56, "Qualified fixture reconstruction coverage differs."); return new(fixtures, controls, files);
    }
    private static double Median(IEnumerable<double> values)
    { var sorted = values.OrderBy(x => x).ToArray(); Need(sorted.Length == 10, "Ten paired samples perarm required."); return (sorted[4] + sorted[5]) / 2; }

    private static bool RunTimingCase(LegacyAttention module, R.Fixture f, byte[] qualified, string output,
        Dictionary<string, string> files, List<Dictionary<string, object?>> timings, Action checkpoint)
    {
        using var memory = new Memory(); var c = f.Case; int elements = c.Rows * 6144;
        Need(qualified.Length == elements * 4 && Pass(CheckGated(qualified, f)), "Qualified full-body reference does not fit the fixed independent oracle.");
        var q = memory.Put("Q-F32", Bytes(f.QueryF32)); var gate = memory.Put("gate-F32", Bytes(f.Gate));
        var ends = memory.Put("key-ends", Bytes(Enumerable.Range(0, c.Rows).Select(i => c.Past + i + 1).ToArray()));
        var cache = MakeCache(memory, f); var result = memory.Empty("timed-result", elements * 4L);
        var intervals = new List<object>(); var control = new List<double>(); var candidate = new List<double>(); var roundWins = new List<bool>();
        var record = new Dictionary<string, object?> { ["case"] = c.Name, ["rows"] = c.Rows, ["past"] = c.Past, ["intervals"] = intervals,
            ["blockPermutation"] = cache.BlockPermutation, ["inputAllocationSha256"] = memory.Buffers.Where(x => x.ImmutableSha != null).ToDictionary(x => x.Name, x => x.ImmutableSha),
            ["qualifiedControlBodySha256"] = Sha(qualified), ["passed"] = false }; timings.Add(record); checkpoint();
        for (int round = 0; round < 5; round++)
        {
            var samples = new double[4]; int[] order = { 256, 128, 128, 256 };
            for (int position = 0; position < order.Length; position++)
            {
                int threads = order[position]; Memory.Poison(result, Poisons[(round + position) % 3]);
                // Exactly one same-address warmup outside the measured interval.
                module.Run(q.Pointer, gate.Pointer, ends.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, threads, result.Pointer);
                CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                CudaDriverApi.cuEventCreate(out var begin, 0).ThrowOnError(); IntPtr end = IntPtr.Zero; float milliseconds;
                try
                {
                    CudaDriverApi.cuEventCreate(out end, 0).ThrowOnError(); CudaDriverApi.cuEventRecord(begin, IntPtr.Zero).ThrowOnError();
                    for (int replay = 0; replay < 6; replay++) module.Run(q.Pointer, gate.Pointer, ends.Pointer, cache.Table.Pointer, cache.Blocks, KOffset, VOffset, c.Rows, threads, result.Pointer);
                    CudaDriverApi.cuEventRecord(end, IntPtr.Zero).ThrowOnError(); CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                    CudaDriverApi.cuEventElapsedTime(out milliseconds, begin, end).ThrowOnError();
                }
                finally { if (end != IntPtr.Zero) CudaDriverApi.cuEventDestroy(end).ThrowOnError(); CudaDriverApi.cuEventDestroy(begin).ThrowOnError(); }
                Need(float.IsFinite(milliseconds) && milliseconds > 0, "Invalid CUDA-event interval.");
                byte[] actual = Memory.Body(result); bool exact = actual.AsSpan().SequenceEqual(qualified), guards = memory.CheckGuards(), immutable = memory.CheckInputs(); var numeric = CheckGated(actual, f);
                string file = Cas(actual, output, files); double perKernel = milliseconds / 6.0; samples[position] = perKernel;
                (threads == 256 ? control : candidate).Add(perKernel);
                bool ok = exact && guards && immutable && Pass(numeric);
                intervals.Add(new { round, position, threads, timedReplays = 6, outsideWarmups = 1, intervalMilliseconds = milliseconds, millisecondsPerKernel = perKernel,
                    file, fullQualifiedControlBitsEqual = exact, allAllocationGuards = guards, immutableInputs = immutable, fp64 = numeric, passed = ok }); checkpoint();
                Need(ok, "Timed interval changed qualified output bits/guards/inputs; raw retained.");
            }
            roundWins.Add((samples[1] + samples[2]) / 2 < (samples[0] + samples[3]) / 2);
        }
        Need(module.KernelSubmissions == 140 && intervals.Count == 20, "Exactly120timed+20warmup submissions percase required.");
        double a = Median(control), b = Median(candidate), speedup = a / b; int wins = roundWins.Count(x => x);
        bool scheduleWin = speedup > 1.05 && wins >= 4;
        record["controlMedianMilliseconds"] = a; record["candidateMedianMilliseconds"] = b; record["medianSpeedup"] = speedup;
        record["pairedRoundWins"] = wins; record["roundWinFlags"] = roundWins; record["predeclaredConservativeScheduleWin"] = scheduleWin;
        record["kernelSubmissions"] = module.KernelSubmissions; record["passed"] = true; checkpoint(); return scheduleWin;
    }

    private static int TimingMain(string[] args)
    {
        string? output = null; bool owned = false; var timings = new List<Dictionary<string, object?>>(); var files = new Dictionary<string, string>();
        var state = new Dictionary<string, object?> { ["schema"] = "ninfer-bf16-attention-sameimage-schedule-timing/1", ["complete"] = false, ["GPU"] = "NOT_RUN", ["modelQuality"] = "NOT_RUN", ["endToEndPerformance"] = "NOT_RUN", ["timings"] = timings, ["files"] = files };
        try
        {
            Need(args.Length == 11 && args[0] == "--timing" && args[1] == "--module" && args[3] == "--expect-sha256" && args[5] == "--component-certificate" && args[7] == "--expect-certificate-sha256" && args[9] == "--out", "Exact explicit timing arguments required.");
            string Pinned(string path, string sha)
            { Need(Path.IsPathFullyQualified(path) && Path.GetFullPath(path) == path && File.Exists(path), "Canonical existing pinned path required."); Need(S.Matches(FileSha(path), sha), "Pinned bytes differ."); return path; }
            string image = Pinned(args[2], args[4]), certificate = Pinned(args[6], args[8]), imageSha = S.Normalize(args[4]), certificateSha = S.Normalize(args[8]);
            byte[] certificateBytes = File.ReadAllBytes(certificate); Need(S.Matches(Sha(certificateBytes), certificateSha), "Component certificate changed while opening.");
            string qualifiedFolder = Path.GetDirectoryName(certificate)!; var qualification = ReadQualification(certificateBytes, qualifiedFolder, imageSha);
            output = args[10]; Fresh(output); Directory.CreateDirectory(output); Directory.CreateDirectory(Path.Combine(output, "cas")); owned = true;
            NewFile(Path.Combine(output, "component-certificate.json"), certificateBytes);
            state["modulePath"] = image; state["moduleSha256"] = imageSha; state["componentCertificatePath"] = certificate; state["componentCertificateSha256"] = certificateSha;
            state["toolAssemblySha256"] = FileSha(typeof(Program).Assembly.Location); state["backendAssemblySha256"] = FileSha(typeof(CudaBackend).Assembly.Location);
            state["scope"] = "Kernel-only pairedCUDAevents: same qualified fatbin/arguments, threads256vs128, exactlytwo observedlate32Kgeometries. Same-addresses/defaultstream; cachewarmups outsideinterval, no claimofphysicalcoldcache or model/E2Espeed.";
            state["protocol"] = new { rounds = 5, order = new[] { 256, 128, 128, 256 }, timedReplaysPerInterval = 6, outsideWarmupsPerInterval = 1, minimumMedianSpeedupExclusive = 1.05, minimumRoundWins = 4, casesMustBothWin = true };
            void Checkpoint() => Save(Path.Combine(output!, "result.json"), state); Checkpoint();
            CudaBackend.Register(); CudaDriverApi.cuInit(0).ThrowOnError(); bool allWin = true;
            using (var context = CudaContext.Create(0))
            {
                CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError(); CudaDriverApi.cuDriverGetVersion(out int driver).ThrowOnError(); CudaDriverApi.cuDeviceGetAttribute(out int sms, 16, device).ThrowOnError(); Need(sms == 170, "Pinned170SM campaign required.");
                byte[] name = new byte[256]; CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError(); state["device"] = new { ordinal = device, name = Encoding.UTF8.GetString(name).TrimEnd('\0'), driverVersion = driver, sms }; state["GPU"] = "ATTEMPTED";
                foreach (string nameCase in new[] { "dense-t512-p31744", "constant-t416-p32256" })
                {
                    var f = R.Make(R.Cases().Single(x => x.Name == nameCase)); MatchFixture(qualification.FixtureShas, FixtureDigests(f));
                    string controlFile = qualification.ControlFiles[nameCase]; byte[] qualified = File.ReadAllBytes(Path.Combine(qualifiedFolder, controlFile));
                    Need(S.Matches(Sha(qualified), qualification.RawFiles[controlFile]), "Qualified body changed while opening.");
                    using var module = new LegacyAttention(image); Need(S.Matches(module.ImageSha256, imageSha), "Actually loaded image differs.");
                    allWin &= RunTimingCase(module, f, qualified, output, files, timings, Checkpoint);
                }
            }
            Need(timings.Count == 2 && timings.All(x => (bool)x["passed"]! && ((List<object>)x["intervals"]!).Count == 20 && (long)x["kernelSubmissions"]! == 140), "Both complete20-interval timing series are mandatory.");
            Need(files.Count <= 2 && files.All(x => S.Matches(FileSha(Path.Combine(output, x.Key)), x.Value)), "Full timed output CAS evidence changed.");
            Need(S.Matches(FileSha(image), imageSha) && S.Matches(FileSha(certificate), certificateSha) && S.Matches(FileSha(Path.Combine(output, "component-certificate.json")), certificateSha), "Pinned image/qualification changed during timing.");
            Need(qualification.RawFiles.All(x => S.Matches(FileSha(Path.Combine(qualifiedFolder, x.Key)), x.Value)), "Qualified original raw changed during timing.");
            state["GPU"] = "EXECUTED"; state["complete"] = true; state["passed"] = true; state["kernelPerformance"] = "MEASURED";
            state["actualTimedKernelSubmissions"] = 240; state["actualOutsideWarmupSubmissions"] = 40; state["bothCasesMeetPredeclaredScheduleWin"] = allWin; Checkpoint(); Complete(output, state);
            Console.WriteLine(JsonSerializer.Serialize(new { complete = true, passed = true, cases = 2, timedKernels = 240, outsideWarmups = 40, bothCasesMeetPredeclaredScheduleWin = allWin, modelQuality = "NOT_RUN", endToEndPerformance = "NOT_RUN" })); return 0;
        }
        catch (Exception e) { state["passed"] = false; state["error"] = e.ToString(); if (owned && output != null) Save(Path.Combine(output, "result.json"), state); Console.Error.WriteLine(e); return 1; }
    }
}
