// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using GemmaAttentionShapeProbe;
using TensorSharp;
using TensorSharp.GGML;

return Probe.Run(args);

internal static class Probe
{
    private const int Prefix = 127, MaxRows = 8, PaddedKeys = 256;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };

    internal static int Run(string[] args)
    {
        if (args.SequenceEqual(new[] { "--self-test" })) return OracleSelfTests.Run();
        var options = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 0; i < args.Length; i += 2)
        {
            if (i + 1 >= args.Length || args[i] is not ("--output" or "--expected-native-sha256" or "--repo-root"))
                throw new ArgumentException("--output FRESH_DIRECTORY --expected-native-sha256 SHA256 [--repo-root REPOSITORY]; or --self-test");
            options.Add(args[i], args[i + 1]);
        }
        string directory = Path.GetFullPath(options["--output"]);
        string repository = Path.GetFullPath(options.GetValueOrDefault("--repo-root", Environment.CurrentDirectory));
        string expectedNative = options["--expected-native-sha256"].ToLowerInvariant();
        if (expectedNative.Length != 64 || !expectedNative.All(Uri.IsHexDigit)) throw new ArgumentException("Expected a SHA256 native identity.");
        if (Directory.Exists(directory) || File.Exists(directory)) throw new IOException("Use a fresh output directory.");
        Directory.CreateDirectory(directory);
        var sourceBefore = SourceIdentity(repository);
        var managedBefore = ManagedIdentity();
        var cases = new List<object>();
        var geometries = new[] { new Geometry("local", 16, 8, 256, 1024), new Geometry("global", 16, 2, 512, 2048) };
        using var process = Process.GetCurrentProcess();
        DateTime processStartUtc = process.StartTime.ToUniversalTime();
        var stopwatch = Stopwatch.StartNew();
        GgmlContext? context = null;
        FileIdentity[] native = [];
        string? error = null;
        bool cleanup = false, allFinite = true, allThresholds = true;
        try
        {
            context = new GgmlContext([0], GgmlBackendType.Cuda);
            native = NativeIdentity();
            if (native.Length != 1 || native[0].Sha256 != expectedNative)
                throw new InvalidDataException("Actual loaded GgmlOps identity does not match the required SHA256.");
            var allocator = new GgmlAllocator(context, 0);
            foreach (var geometry in geometries)
            foreach (string profile in new[] { "gentle", "unit-rms" })
            {
                string stem = $"{geometry.Name}-{profile}";
                var input = AttentionOracle.Inputs(geometry, profile, MaxRows, Prefix);
                object[] inputs = [
                    Save(directory, stem + "-q.f32", MemoryMarshal.AsBytes(input.Q.AsSpan()).ToArray()),
                    Save(directory, stem + "-k.f16", HalfBytes(input.K)),
                    Save(directory, stem + "-v.f16", HalfBytes(input.V))];
                using var k = new Tensor(allocator, DType.Float16, geometry.KvHeads, geometry.CacheStride, geometry.Dim);
                using var v = new Tensor(allocator, DType.Float16, geometry.KvHeads, geometry.CacheStride, geometry.Dim);
                Copy(k, HalfBytes(input.K)); Copy(v, HalfBytes(input.V));
                foreach (int rows in new[] { 1, 2, 4, 8 })
                {
                    float[] q = AttentionOracle.QuerySlice(input.Q, geometry.Heads, MaxRows, geometry.Dim, 0, rows);
                    var oracleTimer = Stopwatch.StartNew();
                    double[] reference = AttentionOracle.Evaluate(geometry, q, input.K, input.V, rows, Prefix);
                    oracleTimer.Stop();
                    object referenceFile = Save(directory, $"{stem}-n{rows}-oracle.f64", MemoryMarshal.AsBytes(reference.AsSpan()).ToArray());
                    var arms = new List<object>();
                    var outputs = new Dictionary<string, float[]>();
                    foreach (var phase in new[] { "batch-native-span", "batch-padded", "sequential-native-span", "sequential-padded" })
                    {
                        bool sequential = phase.StartsWith("sequential", StringComparison.Ordinal);
                        bool padded = phase.EndsWith("padded", StringComparison.Ordinal);
                        var execution = Execute(allocator, geometry, q, k, v, rows, sequential, padded);
                        outputs.Add(phase, execution.Values);
                        ErrorMetrics metrics = AttentionOracle.Compare(reference, execution.Values);
                        allFinite &= metrics.ActualNonfinite == 0 && metrics.ReferenceNonfinite == 0;
                        allThresholds &= metrics.WithinDiagnosticThreshold;
                        object[] byRow = Enumerable.Range(0, rows).Select(row => (object)new {
                            row, metrics = AttentionOracle.Compare(reference.AsSpan(row * geometry.Heads * geometry.Dim, geometry.Heads * geometry.Dim).ToArray(),
                                execution.Values.AsSpan(row * geometry.Heads * geometry.Dim, geometry.Heads * geometry.Dim).ToArray()) }).ToArray();
                        arms.Add(new { phase, execution.Calls, native_call_seconds = execution.Calls.Sum(call => call.NativeCallSeconds),
                            metrics, rows = byRow,
                            output = Save(directory, $"{stem}-n{rows}-{phase}.f32", MemoryMarshal.AsBytes(execution.Values.AsSpan()).ToArray()) });
                    }
                    object[] comparisons = new[] {
                        ("batch-native-span", "sequential-padded"), ("batch-padded", "sequential-padded"),
                        ("sequential-native-span", "sequential-padded"), ("batch-native-span", "batch-padded")
                    }.Select(pair => (object)new { actual = pair.Item1, reference = pair.Item2,
                        metrics = AttentionOracle.Compare(outputs[pair.Item2].Select(x => (double)x).ToArray(), outputs[pair.Item1]),
                        bit_exact = MemoryMarshal.AsBytes(outputs[pair.Item1].AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(outputs[pair.Item2].AsSpan())) }).ToArray();
                    var row = new { geometry, profile, query_rows = rows, prefix = Prefix, inputs, reference = referenceFile,
                        oracle_seconds = oracleTimer.Elapsed.TotalSeconds, arms, comparisons };
                    cases.Add(row);
                    Console.WriteLine(JsonSerializer.Serialize(row));
                }
            }
        }
        catch (Exception failure) { error = failure.ToString(); Console.Error.WriteLine(failure); }
        finally
        {
            try
            {
                if (context != null)
                {
                    GgmlBasicOps.HostReadBarrier(); GgmlBasicOps.ClearHostBufferCache();
                    context.ReleasePoolForDisposal(); GgmlBasicOps.Shutdown();
                }
                cleanup = true;
            }
            catch (Exception failure) { error = (error ?? "") + "\nCleanup: " + failure; }
        }
        FileIdentity[] managedAfter = ManagedIdentity(), sourceAfter = SourceIdentity(repository);
        FileIdentity[] nativeAfter = native.Select(x => Identity(x.Path)).ToArray();
        bool identitiesUnchanged = managedBefore.SequenceEqual(managedAfter) && sourceBefore.SequenceEqual(sourceAfter) && native.SequenceEqual(nativeAfter);
        bool completed = error == null && cleanup && identitiesUnchanged && cases.Count == 16;
        File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new {
            completed, error, cleanup, all_finite = completed && allFinite,
            diagnostic_thresholds_satisfied = completed && allThresholds,
            model_quality_checked = false, identities_unchanged = identitiesUnchanged,
            process_id = Environment.ProcessId, process_start_utc = processStartUtc, elapsed_seconds = stopwatch.Elapsed.TotalSeconds,
            backend = "ggml_cuda", expected_native_sha256 = expectedNative, native_loaded = native, native_files_after = nativeAfter,
            managed_before = managedBefore, managed_after = managedAfter, source_before = sourceBefore, source_after = sourceAfter,
            geometry_source = "Gemma26 production log: heads16, localKV8/dim256/window1024, globalKV2/dim512/context2048; synthetic post-RoPE inputs, no model loaded.",
            arithmetic = new { q = "F32", kv = "stored F16, same rounded values used by FP64 oracle", output = "F32", oracle = "scalar FP64 dot/exp/weighted sum", attention_scale = 1,
                native_accumulator_request = "GGML_PREC_F32 (does not promise F32 for every kernel intermediate)", output_layout = "query, head, dimension", raw_byte_order = "little-endian" },
            thresholds = new { max_abs = 2e-3, relative_l2 = 1e-3, scope = "Diagnostic alert limits inherited from existing standalone flash tests; not model quality acceptance or a proof of row equivalence." },
            limitations = new[] {
                "StreamingFlashAttention copies each host KV head into contiguous device storage; it does not preserve resident cache head strides or CUDA graph capture.",
                "Q is contiguous head-first here; fused model Q comes from a token-first projection then permute. This probe does not replicate that view stride.",
                "No RMSNorm/RoPE, quantized projection, router, expert, residual, or LM-head nodes are executed. Shared attention inputs isolate attention only.",
                "Global dim512 native API always pads to256 even for native-span calls; local dim256 native-span preserves128..135 logical KV lengths.",
                "Finite nonzero future/padding keys are present and must be excluded by the causal mask; no SWA wrap occurs at prefix127.",
                "Per-call times include scoped graph construction, upload, execution and download, with no balanced warmup; they are diagnostic timings, not production throughput.",
                "Exit0 means completed finite execution and stable identities; consult numeric diagnostics separately. No whole-model PASS is produced.",
                "Source before/after hashes identify files present during execution, not an embedded build-source attestation." },
            cases
        }, Json));
        return completed && allFinite ? 0 : 1;
    }

    private static Execution Execute(GgmlAllocator allocator, Geometry g, float[] query, Tensor k, Tensor v, int rows, bool sequential, bool padded)
    {
        var result = new float[rows * g.Heads * g.Dim];
        var calls = new List<Call>();
        int count = sequential ? rows : 1;
        for (int row = 0; row < count; row++)
        {
            int queryRows = sequential ? 1 : rows, maskStart = Prefix + (sequential ? row : 0);
            int logicalSpan = maskStart + queryRows, physicalKeys = padded ? PaddedKeys : logicalSpan;
            float[] data = sequential ? AttentionOracle.QuerySlice(query, g.Heads, rows, g.Dim, row, 1) : query;
            using var q = new Tensor(allocator, DType.Float32, g.Heads, queryRows, g.Dim);
            using var output = new Tensor(allocator, DType.Float32, queryRows, g.Heads * g.Dim);
            Copy(q, MemoryMarshal.AsBytes(data.AsSpan()).ToArray());
            var timer = Stopwatch.StartNew();
            GgmlBasicOps.StreamingFlashAttention(q, k, v, output, g.Heads, g.KvHeads, g.Dim,
                queryRows, physicalKeys, g.CacheStride, maskStart, 0, 1f);
            timer.Stop();
            float[] values = output.GetElementsAsFloat(queryRows * g.Heads * g.Dim);
            Array.Copy(values, 0, result, sequential ? row * g.Heads * g.Dim : 0, values.Length);
            calls.Add(new(row, queryRows, maskStart, logicalSpan, physicalKeys,
                g.Dim == 512 ? ((physicalKeys + 255) / 256) * 256 : physicalKeys, timer.Elapsed.TotalSeconds));
        }
        return new(result, calls.ToArray());
    }

    private static byte[] HalfBytes(float[] values) => MemoryMarshal.AsBytes(values.Select(x => (System.Half)x).ToArray().AsSpan()).ToArray();
    private static void Copy(Tensor tensor, byte[] bytes)
    {
        Marshal.Copy(bytes, 0, tensor.Storage.PtrAtElement(0), bytes.Length);
        GgmlBasicOps.InvalidateHostBuffer(tensor.Storage.PtrAtElement(0));
    }
    private static FileIdentity Save(string directory, string name, byte[] bytes)
    {
        if (!BitConverter.IsLittleEndian) throw new PlatformNotSupportedException("Raw evidence requires little-endian output.");
        string path = Path.Combine(directory, name); File.WriteAllBytes(path, bytes); return Identity(path);
    }
    private static FileIdentity Identity(string path)
    {
        using var stream = File.OpenRead(path);
        return new(Path.GetFullPath(path), stream.Length, Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant());
    }
    private static FileIdentity[] ManagedIdentity() => Directory.GetFiles(AppContext.BaseDirectory, "*.dll")
        .Where(path => Path.GetFileName(path).StartsWith("TensorSharp", StringComparison.Ordinal) || Path.GetFileName(path) == "GemmaAttentionShapeProbe.dll")
        .Order(StringComparer.Ordinal).Select(Identity).ToArray();
    private static FileIdentity[] SourceIdentity(string repository) => new[] {
        "eng/validation/GemmaAttentionShapeProbe/Program.cs", "eng/validation/GemmaAttentionShapeProbe/AttentionOracle.cs", "eng/validation/GemmaAttentionShapeProbe/OracleSelfTests.cs",
        "TensorSharp.Backends.GGML/GgmlStreamingAttention.cs", "TensorSharp.GGML.Native/ggml_ops_norm_attn.cpp", "TensorSharp.GGML.Native/ggml_ops_gemma4_moe.cpp",
        "TensorSharp.GGML.Native/ggml_ops_transformer_common.h", "ExternalProjects/ggml/src/ggml-cuda/fattn.cu", "ExternalProjects/ggml/src/ggml-cuda/mmvq.cu"
    }.Select(path => Identity(Path.Combine(repository, path))).ToArray();
    private static FileIdentity[] NativeIdentity()
    {
        using var process = Process.GetCurrentProcess();
        return process.Modules.Cast<ProcessModule>().Where(module => module.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
            .Select(module => module.FileName).Distinct(StringComparer.OrdinalIgnoreCase).Order(StringComparer.Ordinal).Select(Identity).ToArray();
    }
    private sealed record FileIdentity(string Path, long Bytes, string Sha256);
    private sealed record Call(int QueryOffset, int QueryRows, int MaskStart, int LogicalKeySpan, int ApiKvLength, int DeviceKvLength, double NativeCallSeconds);
    private sealed record Execution(float[] Values, Call[] Calls);
}
