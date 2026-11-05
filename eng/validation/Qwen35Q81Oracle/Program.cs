using System.Buffers.Binary;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;

return Oracle.Run(args);

internal static unsafe class Oracle
{
    internal const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    internal static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private const string Bbd = "bbdaca53a0fb56ec78644356edfa981f6552e57c011f6b8cbfe015ac7b608515";
    private const string Target = "d5659ecb29adaf75f313fdae03eb7ff17baf3f986f0c3b1348e8afda1670c9a8";
    internal static void Require(bool value, string reason) { if (!value) throw new InvalidOperationException(reason); }
    internal static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    internal static string FileHash(string p) { using var f = File.OpenRead(p); return Convert.ToHexString(SHA256.HashData(f)).ToLowerInvariant(); }
    internal static void Write(string p, object value) => File.WriteAllText(p, JsonSerializer.Serialize(value, Json) + "\n");
    internal static FieldInfo Field(Type type, string name, Type? expected = null, bool nonpublic = true)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.DeclaredOnly | (nonpublic ? BindingFlags.NonPublic : BindingFlags.Public));
        Require(field != null && !field.IsStatic && (expected == null || field.FieldType == expected), "Unsupported field: " + type + "." + name);
        return field!;
    }
    internal static MethodInfo Method(Type type, string name, bool stat, Type result, params Type[] parameters)
    {
        var method = type.GetMethod(name, BindingFlags.DeclaredOnly | BindingFlags.Public | BindingFlags.NonPublic | (stat ? BindingFlags.Static : BindingFlags.Instance), null, parameters, null);
        Require(method != null && method.IsStatic == stat && method.ReturnType == result, "Unsupported method: " + type + "." + name);
        return method!;
    }
    private static string Root(string? requested)
    {
        if (requested != null) return Path.GetFullPath(requested);
        for (var d = new DirectoryInfo(Environment.CurrentDirectory); d != null; d = d.Parent)
            if (File.Exists(Path.Combine(d.FullName, "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.VerifyDecodeQ81.cs"))) return d.FullName;
        throw new IOException("Run within the isolated worktree or pass --source-root.");
    }
    internal static string Fresh(string path)
    {
        string full = Path.GetFullPath(path);
        Require(Path.TrimEndingDirectorySeparator(path) == Path.TrimEndingDirectorySeparator(full), "Output must be absolute and normalized.");
        Require(!File.Exists(full) && !Directory.Exists(full), "Output must be fresh.");
        for (var d = new DirectoryInfo(Path.GetDirectoryName(full)!); d != null; d = d.Parent)
            Require(!d.Exists || d.LinkTarget == null, "Output ancestor may not be a symlink.");
        Directory.CreateDirectory(full); return full;
    }
    internal static Dictionary<string, object> Sources(string root)
    {
        string[] names = { "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.cs", "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.VerifyDecodeQ81.cs", "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.Scratch.cs", "TensorSharp.Backends.Cuda/Qwen35/Q35Kernels.cs", "TensorSharp.Backends.Cuda/CudaModule.cs", "TensorSharp.Backends.Cuda/CudaKernels.cs", "eng/validation/Qwen35CudaEngineProbe/MixedCapacityTrace.cs", "eng/validation/Qwen35CudaEngineProbe/SharedScratchTrace.cs", "eng/validation/Qwen35Q81Oracle/Program.cs", "eng/validation/Qwen35Q81Oracle/Native.cs", "eng/validation/Qwen35Q81Oracle/Qwen35Q81Oracle.csproj" };
        return names.ToDictionary(n => n, n => (object)new { bytes = new FileInfo(Path.Combine(root, n)).Length, sha256 = FileHash(Path.Combine(root, n)) });
    }
    private static Dictionary<string, object> Assemblies() => AppDomain.CurrentDomain.GetAssemblies().Where(a => !a.IsDynamic && !string.IsNullOrEmpty(a.Location))
        .OrderBy(a => a.Location).ToDictionary(a => a.Location, a => (object)new { bytes = new FileInfo(a.Location).Length, sha256 = FileHash(a.Location) });
    private static string Gpu()
    {
        using var p = Process.Start(new ProcessStartInfo("nvidia-smi", "--query-gpu=uuid,name,driver_version,memory.total --format=csv,noheader") { RedirectStandardOutput = true, RedirectStandardError = true })!;
        var text = p.StandardOutput.ReadToEndAsync(); var error = p.StandardError.ReadToEndAsync();
        if (!p.WaitForExit(10000)) { p.Kill(entireProcessTree: true); p.WaitForExit(); throw new TimeoutException("Owned nvidia-smi timed out."); }
        Require(p.ExitCode == 0, "nvidia-smi failed: " + error.GetAwaiter().GetResult()); return text.GetAwaiter().GetResult().Trim();
    }
    internal static object Compare(float[] a, float[] b, int rows, int width)
    {
        Require(a.Length == rows * width && b.Length == a.Length, "F32 comparison dimensions differ.");
        int differing = 0, first = -1, top1 = 0; long maxUlp = 0;
        for (int i = 0; i < a.Length; i++)
        {
            uint x = BitConverter.SingleToUInt32Bits(a[i]), y = BitConverter.SingleToUInt32Bits(b[i]);
            if (x != y) { differing++; if (first < 0) first = i; if (float.IsFinite(a[i]) && float.IsFinite(b[i])) maxUlp = Math.Max(maxUlp, Math.Abs((long)Ordered(x) - Ordered(y))); }
        }
        for (int r = 0; r < rows; r++) if (Argmax(a, r * width, width) != Argmax(b, r * width, width)) top1++;
        return new { strict_pass = differing == 0, differing_words = differing, first_index = first, first_reference_bits = first < 0 ? (uint?)null : BitConverter.SingleToUInt32Bits(a[first]), first_actual_bits = first < 0 ? (uint?)null : BitConverter.SingleToUInt32Bits(b[first]), max_finite_ulp = maxUlp, top1_disagreements = top1, nonfinite_reference = a.Count(x => !float.IsFinite(x)), nonfinite_actual = b.Count(x => !float.IsFinite(x)) };
    }
    private static uint Ordered(uint x) => (x & 0x80000000) != 0 ? ~x : x | 0x80000000;
    internal static int Argmax(float[] values, int start, int count) { int best = 0; for (int i = 1; i < count; i++) if (values[start + i] > values[start + best]) best = i; return best; }
    internal static bool Strict(object comparison) => JsonSerializer.SerializeToElement(comparison).GetProperty("strict_pass").GetBoolean();
    private static void Flags()
    {
        Require(Environment.GetEnvironmentVariable("TS_Q35_VERIFY_DECODE_Q81") is "0" or "1", "Set TS_Q35_VERIFY_DECODE_Q81=0 or1.");
        Require(Environment.GetEnvironmentVariable("TS_Q35_PROJECTION_TRACE") == "1", "Set TS_Q35_PROJECTION_TRACE=1.");
        Require(Environment.GetEnvironmentVariable("TS_Q35_GDN_PREPARED_PREFILL") == "0", "Set prepared GDN explicitly OFF.");
        foreach (string name in new[] { "TS_Q35_CUDA_GRAPHS", "TS_Q35_VERIFY_GRAPHS", "TS_Q35_GREEDY_VERIFY", "TS_Q35_TOPK_PARTITIONED" }) Require(Environment.GetEnvironmentVariable(name) is null or "0", "Keep " + name + " OFF.");
        foreach (string name in new[] { "TS_Q35_GEMM_SPLITK", "TS_Q35_DRAFT_SPLITK", "TS_Q35_ATTN_PREFILL" }) Require(Environment.GetEnvironmentVariable(name) is null or "1", "Keep original production default for " + name);
    }
    private static float[] Floats(byte[] bytes) { Require(bytes.Length % 4 == 0, "Unaligned F32 file."); var a = new float[bytes.Length / 4]; Buffer.BlockCopy(bytes, 0, a, 0, bytes.Length); return a; }
    internal static void SaveFloats(string file, float[] values) => File.WriteAllBytes(file, MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
    public static int Run(string[] args)
    {
        if (args.Length == 1 && args[0] == "--host-guard-self-test") return PartitionHostGuards.SelfTest();
        string? output = null, model = null, rootArg = null, reference = null; string phase = "all"; bool execute = false, self = false;
        for (int i = 0; i < args.Length; i++) switch (args[i])
        {
            case "--out": output = args[++i]; break; case "--model": model = args[++i]; break; case "--source-root": rootArg = args[++i]; break;
            case "--phase": phase = args[++i]; break; case "--mixed-reference": reference = args[++i]; break; case "--execute": execute = true; break; case "--self-test": self = true; break;
            default: throw new ArgumentException("Unknown option: " + args[i]);
        }
        if (self) return SelfTest.Run();
        Require(output != null && phase is "all" or "component" or "partition" or "mixed", "Use --out ABS_FRESH [--phase all|component|partition|mixed] [--execute --model FILE] [--mixed-reference BASE/mixed].");
        string root = Root(rootArg);
        string normalizedOut = Path.GetFullPath(output!);
        Require(!normalizedOut.StartsWith(root + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || normalizedOut.StartsWith(Path.Combine(root, "artifacts") + Path.DirectorySeparatorChar, StringComparison.Ordinal)
            || normalizedOut.StartsWith(Path.Combine(root, "docs", "validation") + Path.DirectorySeparatorChar, StringComparison.Ordinal), "In-tree outputs must be under ignored artifacts or docs/validation.");
        string dir = Fresh(output!); var before = Sources(root); var results = new Dictionary<string, object>();
        var state = new Dictionary<string, object?> { ["schema"] = "q35-q81-component-partition-oracle-v1", ["complete"] = false, ["quality_passed"] = false, ["status"] = "PLAN_ONLY_GPU_NOT_RUN", ["phase_selection"] = phase, ["source_root"] = root, ["source_before"] = before, ["gpu_not_run"] = true, ["results"] = results, ["host_guard_diagnostic"] = new { version = 2, guard_words = PartitionHostGuards.Words, poison_u32 = PartitionHostGuards.Poison, initialization = "uint32-bitpaint-BlockCopy", before_all_three_asserted = true, after_all_three_tail_bits_saved = true },
            ["limitations"] = new[] { "No timing/ranking claim. Invocation must be inside operator-owned exclusive GPU lease; this tool starts no server or background process.", "Candidate module ImageSha256 is normally null. Candidate identity is actual located files + loader source + module handle/function arch, not a direct driver image SHA.", "N1/N8 numerical differences remain strict FAIL; no tolerance or quality downgrade.", "Old A4 versus Q81 verify values intentionally differ; only preserved prefill is cross-flag equal.", "Component directly invokes private helper once per fixture: not serving eligibility and not component A/A. Q81 flag1 still records Dictionary counters even TRACE0, plus per-projection Q81Chunks allocations; no zero-observer performance claim." } };
        void Persist() => Write(Path.Combine(dir, "run.json"), state);
        Persist();
        try
        {
            Native.CheckContract(); state["reflection_contract_cpu"] = "PASS_NO_NATIVE_CALLS";
            if (!execute) { state["plan"] = new { component = "Real six weight planes;8..15/17 rows atoffset3; all tail sizes finite plus zero-subnormal/nonfinite17-row patterns; independent guarded buffers; Q81 bytes/scales and explicit N8/tail vs private candidate; N1 strict comparator.", partition = "Same physical2 slots, prefix7/equalmaxKeys, VerifyN2x8/N1x8 AA+ABBA, full F32/drawn/features; matching reVerifyN2 beforeCommit and8teachersteps for joint/separate commit arms.", mixed = "Frozen4096+64,38-record AA trace; role/dtype/range host API counts. Crossflag reference requires same model/source/assembly/inputs and compares both full prefill rows." }; Persist(); return 0; }
            Flags(); Require(model != null && BitConverter.IsLittleEndian, "GPU run needs --model and little endian.");
            model = Path.GetFullPath(model!); state["status"] = "RUNNING_NOT_PASS"; state["flags"] = Environment.GetEnvironmentVariables().Cast<System.Collections.DictionaryEntry>().Where(e => ((string)e.Key).StartsWith("TS_Q35_", StringComparison.Ordinal)).ToDictionary(e => (string)e.Key, e => e.Value?.ToString());
            string modelHash = FileHash(model); state["model"] = new { path = model, bytes = new FileInfo(model).Length, sha256 = modelHash }; Require(modelHash == Target, "Target GGUF does not match the pinned model.");
            state["gpu"] = Gpu(); state["host"] = new { machine = Environment.MachineName, boot_id = File.Exists("/proc/sys/kernel/random/boot_id") ? File.ReadAllText("/proc/sys/kernel/random/boot_id").Trim() : "unavailable", cuda_visible_devices = Environment.GetEnvironmentVariable("CUDA_VISIBLE_DEVICES") }; state["assemblies_before"] = Assemblies(); state["native_before"] = Native.Provider(); state["gpu_not_run"] = false; Persist();
            if (phase is "all" or "component" or "partition")
            {
                using var exec = new Qwen35CudaExecutor(model, 8192, 4096, draftGgufPath: null!); Require(!exec.Engine.HasDrafter, "Drafter must be absent.");
                using var native = new Native(exec.Engine); state["actual_candidate_module"] = native.CandidateModule; state["actual_identified_reference_module"] = native.ReferenceModule; Persist();
                if (phase is "all" or "component") { results["component"] = Component.Run(exec.Engine, native, Path.Combine(dir, "component")); Persist(); }
                if (phase is "all" or "partition") { results["partition"] = Partition.Run(exec.Engine, native, model, Path.Combine(dir, "partition")); Persist(); }
            }
            if (phase is "all" or "mixed")
            {
                string? previousSource = Environment.GetEnvironmentVariable("TS_PROBE_SOURCE_ROOT");
                Require(previousSource == null || Path.GetFullPath(previousSource) == root, "Existing TS_PROBE_SOURCE_ROOT differs.");
                string mixed = Path.Combine(dir, "mixed"); int code;
                try { Environment.SetEnvironmentVariable("TS_PROBE_SOURCE_ROOT", root); code = MixedCapacityTrace.Run(model, mixed, projectionDiagnostic: true); }
                finally { Environment.SetEnvironmentVariable("TS_PROBE_SOURCE_ROOT", previousSource); }
                Require(code == 0, "Mixed trace failed."); results["mixed_reset_AA"] = new { strict_pass = true, path = mixed, run_sha256 = FileHash(Path.Combine(mixed, "run.json")) }; Persist();
                if (reference != null) { results["mixed_prefill_cross_flag"] = MixedCompare.Run(reference, mixed); Persist(); }
                else results["mixed_prefill_cross_flag"] = new { strict_pass = false, status = "NOT_RUN_REFERENCE_REQUIRED", note = "Baseline capture is not a full candidate validation." };
            }
            state["complete"] = true; state["quality_passed"] = results.Values.All(Strict); state["status"] = (bool)state["quality_passed"]! ? "STRICT_PASS_SELECTED_PHASES" : "STRICT_FAIL_OR_NOT_RUN";
        }
        catch (Exception error) { state["status"] = "ERROR_NOT_PASS"; state["error"] = error.ToString(); }
        finally
        {
            try { var after = Sources(root); state["source_after"] = after; state["source_pre_post_equal"] = JsonSerializer.Serialize(after) == JsonSerializer.Serialize(before); Require((bool)state["source_pre_post_equal"]!, "Sources changed."); if (execute) { state["assemblies_after"] = Assemblies();
                    var assemblyBefore = (Dictionary<string, object>)state["assemblies_before"]!; var assemblyAfter = (Dictionary<string, object>)state["assemblies_after"]!;
                    state["existing_loaded_assemblies_pre_post_equal"] = assemblyBefore.All(p => assemblyAfter.TryGetValue(p.Key, out var value) && JsonSerializer.Serialize(p.Value) == JsonSerializer.Serialize(value));
                    Require((bool)state["existing_loaded_assemblies_pre_post_equal"]!, "Loaded assembly bytes changed."); state["native_after"] = Native.Provider(); state["native_pre_post_equal"] = JsonSerializer.Serialize(state["native_before"]) == JsonSerializer.Serialize(state["native_after"]); state["model_post_sha256"] = FileHash(model!); Require((bool)state["native_pre_post_equal"]! && (string)state["model_post_sha256"]! == Target, "Provider or model changed."); } }
            catch (Exception error) { state["complete"] = false; state["quality_passed"] = false; state["postcheck_error"] = error.ToString(); state["status"] = "POSTCHECK_ERROR_NOT_PASS"; }
            Persist();
        }
        return state.TryGetValue("quality_passed", out var passed) && passed is true ? 0 : 1;
    }

    internal static class MixedCompare
    {
        internal static object Run(string original, string actual)
        {
            original = Path.GetFullPath(original); actual = Path.GetFullPath(actual);
            using var outerA = JsonDocument.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(original)!, "run.json")));
            using var outerB = JsonDocument.Parse(File.ReadAllText(Path.Combine(Path.GetDirectoryName(actual)!, "run.json")));
            var oa = outerA.RootElement; var ob = outerB.RootElement;
            Require(oa.GetProperty("schema").GetString() == "q35-q81-component-partition-oracle-v1" && ob.GetProperty("schema").GetString() == "q35-q81-component-partition-oracle-v1"
                && oa.GetProperty("complete").GetBoolean() && oa.GetProperty("source_pre_post_equal").GetBoolean() && oa.GetProperty("native_pre_post_equal").GetBoolean(), "Baseline capture/postchecks are incomplete.");
            foreach (string key in new[] { "model", "gpu", "host", "source_before", "native_before" }) Require(oa.GetProperty(key).GetRawText() == ob.GetProperty(key).GetRawText(), "Outer source/model/GPU/provider differs: " + key);
            Require(oa.GetProperty("results").GetProperty("mixed_reset_AA").GetProperty("run_sha256").GetString() == FileHash(Path.Combine(original, "run.json"))
                && ob.GetProperty("results").GetProperty("mixed_reset_AA").GetProperty("run_sha256").GetString() == FileHash(Path.Combine(actual, "run.json")), "Mixed manifest is not bound to the current outer campaign.");
            using var a = JsonDocument.Parse(File.ReadAllText(Path.Combine(original, "run.json"))); using var b = JsonDocument.Parse(File.ReadAllText(Path.Combine(actual, "run.json")));
            var x = a.RootElement; var y = b.RootElement; Require(x.GetProperty("complete").GetBoolean() && y.GetProperty("complete").GetBoolean(), "Both mixed traces must complete.");
            foreach (string key in new[] { "model", "inputs_sha256", "geometry", "provenance" }) Require(x.GetProperty(key).GetRawText() == y.GetProperty(key).GetRawText(), "Mixed reference changed: " + key);
            var fa = x.GetProperty("flags").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetRawText()); var fb = y.GetProperty("flags").EnumerateObject().ToDictionary(p => p.Name, p => p.Value.GetRawText());
            Require(fa["TS_Q35_VERIFY_DECODE_Q81"] == "\"0\"" && fb["TS_Q35_VERIFY_DECODE_Q81"] == "\"1\"", "Crossflag needs original0/candidate1."); fa.Remove("TS_Q35_VERIFY_DECODE_Q81"); fb.Remove("TS_Q35_VERIFY_DECODE_Q81"); Require(fa.Count == fb.Count && fa.All(p => fb.TryGetValue(p.Key, out var value) && p.Value == value), "Other mixed flags differ.");
            Require(FileHash(Path.Combine(original, "inputs.json")) == x.GetProperty("inputs_sha256").GetString()
                && FileHash(Path.Combine(actual, "inputs.json")) == y.GetProperty("inputs_sha256").GetString(), "Mixed input files differ from recorded hashes.");
            using var ta = JsonDocument.Parse(File.ReadAllText(Path.Combine(original, "trace.json"))); using var tb = JsonDocument.Parse(File.ReadAllText(Path.Combine(actual, "trace.json")));
            var rows = new List<object>(); bool pass = true;
            for (int repeat = 0; repeat < 2; repeat++) { string file = $"repeat-{repeat}.mixed-prefill.f32";
                var ra = ta.RootElement.EnumerateArray().Single(r => r.GetProperty("key").GetString() == "mixed-prefill" && r.GetProperty("repeat").GetInt32() == repeat);
                var rb = tb.RootElement.EnumerateArray().Single(r => r.GetProperty("key").GetString() == "mixed-prefill" && r.GetProperty("repeat").GetInt32() == repeat);
                Require(ra.GetProperty("sha256").GetString() == FileHash(Path.Combine(original, file)) && rb.GetProperty("sha256").GetString() == FileHash(Path.Combine(actual, file)), "Prefill row does not match full trace hash."); var aa = Floats(File.ReadAllBytes(Path.Combine(original, file))); var bb = Floats(File.ReadAllBytes(Path.Combine(actual, file))); Require(aa.Length == 248320 && bb.Length == aa.Length, "Incomplete prefill row."); var comparison = Compare(aa, bb, 1, aa.Length); pass &= Strict(comparison); rows.Add(new { repeat, reference_sha256 = FileHash(Path.Combine(original, file)), actual_sha256 = FileHash(Path.Combine(actual, file)), comparison }); }
            var result = new { strict_pass = pass, comparisons = rows, verify_A4_vs_Q81_equivalence_claim = false, scope = "Two repeats, one full248320-F32 last-prefill vocab row each; NOT4096 per-position logits or full recurrent-state equivalence." }; Write(Path.Combine(actual, "prefill-crossflag.json"), result); return result;
        }
    }
}
