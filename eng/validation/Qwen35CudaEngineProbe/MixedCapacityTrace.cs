// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime;

// An exact capacity/continuation diagnostic, not a timing benchmark. Each process
// loads one model with no drafter and repeats precisely the same mixed schedule.
// Compare separate prepared=0/1 processes; separate-vs-mixed changes FP4 shapes
// and is deliberately NOT the oracle.
internal static class MixedCapacityTrace
{
    private const int UBatch = 4096, VerifyLanes = 8, Window = 8, Slots = 9;
    private const int PrefixRows = 7, TeacherSteps = 8, Repeats = 2, Context = 8192;
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    private static readonly string[] Flags = {
        "TS_Q35_GDN_PREPARED_PREFILL", "TS_Q35_GEMM_SPLITK", "TS_Q35_ATTN_PREFILL",
        "TS_Q35_CUDA_GRAPHS", "TS_Q35_VERIFY_GRAPHS", "TS_Q35_GREEDY_VERIFY",
        "TS_Q35_TOPK_PARTITIONED", "TS_Q35_DRAFT_SPLITK"
    };

    private static FieldInfo Field(string name, Type type)
    {
        var field = typeof(Q35CudaEngine).GetField(name, PrivateInstance);
        if (field == null || field.FieldType != type || field.IsStatic || !field.IsPrivate)
            throw new InvalidOperationException($"Unsupported engine diagnostic field {name}; expected private instance {type.FullName}.");
        return field;
    }

    // Keep the native API private to its production assembly. This probe checks
    // its exact signature and invokes only the existing synchronous D2H wrapper.
    private delegate int DeviceToHost(IntPtr destination, IntPtr source, UIntPtr bytes);
    private static MethodInfo CopyMethod()
    {
        var driver = typeof(Q35CudaEngine).Assembly.GetType("TensorSharp.Cuda.Interop.CudaDriverApi", throwOnError: true)!;
        var method = driver.GetMethod("cuMemcpyDtoH", BindingFlags.Public | BindingFlags.Static | BindingFlags.DeclaredOnly,
            null, new[] { typeof(IntPtr), typeof(IntPtr), typeof(UIntPtr) }, null);
        if (method == null || method.ReturnType != typeof(int) || !method.IsStatic)
            throw new InvalidOperationException("Unsupported cuMemcpyDtoH diagnostic signature.");
        return method;
    }

    private static string FreshOutput(string output)
    {
        output = Path.GetFullPath(output);
        if (Directory.Exists(output) || File.Exists(output))
            throw new IOException("Output must be fresh; existing evidence is never overwritten.");
        Directory.CreateDirectory(output);
        return output;
    }

    private static string Hash(ReadOnlySpan<byte> bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    private static string FileHash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
    private static void Write(string path, object value) => File.WriteAllText(path, JsonSerializer.Serialize(value, Json) + "\n");
    private static void Require(bool ok, string message) { if (!ok) throw new InvalidOperationException(message); }

    private static string SourceRoot()
    {
        if (Environment.GetEnvironmentVariable("TS_PROBE_SOURCE_ROOT") is { Length: > 0 } root)
            return Path.GetFullPath(root);
        for (DirectoryInfo? at = new(Environment.CurrentDirectory); at != null; at = at.Parent)
            if (File.Exists(Path.Combine(at.FullName, "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.cs")))
                return at.FullName;
        throw new IOException("Run in the source checkout or set TS_PROBE_SOURCE_ROOT.");
    }

    private static object Provenance()
    {
        string root = SourceRoot();
        string[] sourcePaths = {
            "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.cs",
            "TensorSharp.Backends.Cuda/Qwen35/Q35Kernels.cs",
            "eng/validation/Qwen35CudaEngineProbe/MixedCapacityTrace.cs",
            "eng/validation/Qwen35CudaEngineProbe/Program.cs"
        };
        // These assertions describe the private-buffer contract being inspected.
        // Source hashes are evidence, not a claim that a source file proves which
        // assembly was built. The caller must also retain build/assembly provenance.
        string engine = File.ReadAllText(Path.Combine(root, sourcePaths[0]));
        Require(engine.Contains("Head(_xn, nv, null, drawnOut);", StringComparison.Ordinal)
            && engine.Contains("_logits = Alloc((long)MaxVerifyRows * _m.NVocab * 4);", StringComparison.Ordinal)
            && engine.Contains("private void Head(IntPtr normed, int rows, float[] logitsOut, int[] argmaxOut)", StringComparison.Ordinal),
            "Reviewed MixedVerify/Head private-buffer source contract has changed.");
        return new {
            source_root = root,
            source_files = sourcePaths.Select(p => new { path = p, sha256 = FileHash(Path.Combine(root, p)) }).ToArray(),
            engine_assembly = new { path = typeof(Q35CudaEngine).Assembly.Location, sha256 = FileHash(typeof(Q35CudaEngine).Assembly.Location) },
            probe_assembly = new { path = typeof(MixedCapacityTrace).Assembly.Location, sha256 = FileHash(typeof(MixedCapacityTrace).Assembly.Location) },
            reflection_contract = "DeclaredOnly private instance _logits:IntPtr, _verifyRow0:Int32, _m:ModelDesc, _proj:IntPtr, _gdnPreparedScratch:IntPtr, _gdnPreparedScratchBytes:Int64; exact static cuMemcpyDtoH(IntPtr,IntPtr,UIntPtr):Int32; read only, no Head recomputation",
            source_to_assembly_match_requires_external_build_manifest = true
        };
    }

    private static void CheckReflection()
    {
        _ = Field("_logits", typeof(IntPtr));
        _ = Field("_verifyRow0", typeof(int));
        _ = Field("_m", typeof(Q35CudaEngine.ModelDesc));
        _ = Field("_proj", typeof(IntPtr));
        _ = Field("_gdnPreparedScratch", typeof(IntPtr));
        _ = Field("_gdnPreparedScratchBytes", typeof(long));
        _ = (DeviceToHost)CopyMethod().CreateDelegate(typeof(DeviceToHost));
    }

    private static object Definition() => new {
        schema = "q35-mixed-capacity-trace-v1", ubatch = UBatch, context = Context,
        actual_models_to_load = 1, drafter = false, slots = Slots,
        prefix_rows_per_slot = PrefixRows, mixed_prefill_rows = UBatch,
        verify_lanes = VerifyLanes, verify_window = Window, verify_rows = VerifyLanes * Window,
        total_mixed_rows = UBatch + VerifyLanes * Window,
        commit_kept = Enumerable.Range(1, VerifyLanes).ToArray(),
        teacher_steps = TeacherSteps, teacher_batch_lanes = Slots,
        reset_repeats = Repeats, float_records_per_repeat = Slots + 2 + TeacherSteps,
        full_vocab_rows_per_repeat = Slots + 1 + VerifyLanes * Window + TeacherSteps * Slots,
        integer_records_per_repeat = 1 + TeacherSteps,
        oracle = "Fresh-process prepared0/1 with identical mixed schedule; in-process reset A/A is exact. No separate-vs-mixed oracle.",
        timing_is_benchmark = false
    };

    internal static int Plan(string output)
    {
        output = FreshOutput(output);
        CheckReflection();
        Require(Slots == 1 + VerifyLanes && Slots <= Q35CudaEngine.MaxSeqs, "Combined slot capacity.");
        Require(UBatch + VerifyLanes * Window == 4160 && VerifyLanes * Window == Q35CudaEngine.MaxVerifyRows, "Exact row capacity.");
        Require(PrefixRows < 8 && PrefixRows + UBatch + TeacherSteps < Context, "Prefix or context geometry.");
        Require(Slots + 2 + TeacherSteps == 19 && Slots + 1 + VerifyLanes * Window + TeacherSteps * Slots == 146, "Output inventory.");
        Write(Path.Combine(output, "plan.json"), new { complete = false, status = "PLAN_ONLY_GPU_NOT_RUN", definition = Definition(), provenance = Provenance(), reflection_and_geometry_checks = "PASS_CPU_ONLY" });
        Console.WriteLine("[q35-mixed-capacity] CPU plan/geometry/reflection checks passed; model, CUDA and GPU NOT_RUN.");
        return 0;
    }

    private static Dictionary<string, string?> Configuration()
    {
        var result = Flags.ToDictionary(key => key, Environment.GetEnvironmentVariable);
        foreach (System.Collections.DictionaryEntry item in Environment.GetEnvironmentVariables())
            if (item.Key is string key && key.StartsWith("TS_Q35_", StringComparison.Ordinal))
                result[key] = item.Value as string;
        Require(result["TS_Q35_GDN_PREPARED_PREFILL"] is "0" or "1", "Set prepared GDN explicitly to 0 or 1 in separate processes.");
        foreach (string key in new[] { "TS_Q35_GEMM_SPLITK", "TS_Q35_ATTN_PREFILL", "TS_Q35_DRAFT_SPLITK" })
            Require(result[key] is null or "1", $"{key} must retain production default (unset or 1).");
        foreach (string key in new[] { "TS_Q35_CUDA_GRAPHS", "TS_Q35_VERIFY_GRAPHS", "TS_Q35_GREEDY_VERIFY", "TS_Q35_TOPK_PARTITIONED" })
            Require(result[key] is null or "0", $"Keep {key} off for this single-feature diagnostic.");
        return result;
    }

    // MixedVerify's last Head has already materialized all verify rows and read
    // the argmax IDs. No kernel, projection or Head is submitted here. Read before
    // Commit/any next Forward can reuse _logits, and fail closed on private layout.
    private static unsafe float[] ReadLastMixedLogits(Q35CudaEngine engine, int vocab)
    {
        CheckReflection();
        Require(engine.UBatch == UBatch && engine.MixedRowCapacity == 4160 && engine.VocabSize == vocab, "Engine row/vocabulary geometry changed.");
        Require((int)Field("_verifyRow0", typeof(int)).GetValue(engine)! == UBatch, "The last operation was not this 4096-row mixed prefill.");
        IntPtr source = (IntPtr)Field("_logits", typeof(IntPtr)).GetValue(engine)!;
        Require(source != IntPtr.Zero, "Last-Head logits buffer is null.");
        var values = new float[checked(VerifyLanes * Window * vocab)];
        engine.Synchronize();
        var copy = (DeviceToHost)CopyMethod().CreateDelegate(typeof(DeviceToHost));
        fixed (float* destination = values)
        {
            int result = copy((IntPtr)destination, source, new UIntPtr(checked((ulong)values.LongLength * sizeof(float))));
            Require(result == 0, $"Diagnostic cuMemcpyDtoH failed with CUDA result {result}.");
        }
        return values;
    }

    internal static int Run(string modelPath, string output)
    {
        output = FreshOutput(output);
        object? provenance = null;
        Dictionary<string, string?>? flags = null;
        var trace = new List<object>();
        var reference = new Dictionary<string, string>();
        var dispatch = new List<object>();
        string phase = "preflight";
        long preparedCount = 0;
        object? geometry = null;
        object? modelIdentity = null;
        string? inputHash = null;
        void Manifest(bool complete, string? error = null) => Write(Path.Combine(output, "run.json"), new {
            complete, status = complete ? "COMPLETE_STRICT_RESET_AA" : "INCOMPLETE_NOT_PASS", error, phase,
            definition = Definition(), provenance, flags, model = modelIdentity, geometry,
            inputs_sha256 = inputHash, records = trace.Count, prepared_gdn_launches = preparedCount,
            actual_dispatch_observations = dispatch,
            transfer_boundary = "Private last-Head _logits read immediately after MixedVerify and before Commit. This is a diagnostic host copy only, not a production API.",
            no_performance_claim = true
        });
        Manifest(false);
        try
        {
            Require(BitConverter.IsLittleEndian, "Snapshots require little-endian F32/I32.");
            CheckReflection();
            flags = Configuration();
            provenance = Provenance();
            var modelFile = new FileInfo(modelPath);
            modelIdentity = new { path = modelFile.FullName, bytes = modelFile.Length, sha256 = FileHash(modelFile.FullName) };
            Manifest(false);
            using var metadata = new GgufFile(modelPath);
            var tokenizer = ModelBase.CreateTokenizerFromGguf(metadata);
            int[] text = tokenizer.Encode("A library keeps a catalogue of books. Each book has an author and a title. " +
                "A careful program sums the integers two, three and five to obtain ten. Tokyo is the capital of Japan. " +
                "Every row is checked in a fixed order and a second pass repeats exactly the same inputs.\n", addSpecial: false).ToArray();
            Require(text.Length > PrefixRows, "Tokenizer did not provide the prescribed teacher tokens.");
            int[] Tokens(int offset, int count) => Enumerable.Range(0, count).Select(i => text[(offset + i) % text.Length]).ToArray();
            var prefixes = Enumerable.Range(0, Slots).Select(lane => Tokens(lane * 13, PrefixRows)).ToArray();
            int[] prefill = Tokens(37, UBatch), verify = Tokens(71, VerifyLanes * Window);
            var teacher = Enumerable.Range(0, TeacherSteps).Select(step => Enumerable.Range(0, Slots)
                .Select(lane => text[(113 + lane * 11 + step) % text.Length]).ToArray()).ToArray();
            Write(Path.Combine(output, "inputs.json"), new { prefixes, prefill, verify, teacher, kept = Enumerable.Range(1, VerifyLanes).ToArray() });
            inputHash = FileHash(Path.Combine(output, "inputs.json"));
            using var exec = new Qwen35CudaExecutor(modelPath, Context, UBatch, draftGgufPath: null!); // Never consult draft env.
            var engine = exec.Engine;
            var model = (Q35CudaEngine.ModelDesc)Field("_m", typeof(Q35CudaEngine.ModelDesc)).GetValue(engine)!;
            int recurrentLayers = model.Layers.Count(layer => layer.Recurrent), vocab = exec.VocabSize;
            Require(!engine.HasDrafter && engine.UBatch == UBatch && engine.ContextSize == Context && engine.MixedRowCapacity == 4160, "Actual engine does not have the exact no-drafter capacity.");
            Require(model.NEmbd == 5120 && model.NFf == 17408 && model.NLayer == 64 && model.GdnKHeads == 16 && model.GdnVHeads == 48,
                "This boundary diagnostic requires the actual Qwen3.8-27B geometry.");
            long packedWidth = checked((2L * model.GdnKHeads + 2L * model.GdnVHeads) * Q35CudaEngine.GdnHeadDim + 2L * model.GdnVHeads);
            long widest = Math.Max(Math.Max((2L * model.NHead + 2L * model.NKvHead) * model.HeadDim, packedWidth), 2L * model.NFf);
            long projectionBytes = checked(engine.MixedRowCapacity * widest * sizeof(float));
            long aliasOffset = checked(engine.MixedRowCapacity * packedWidth * sizeof(float) + 255) & ~255L;
            var projection = (IntPtr)Field("_proj", typeof(IntPtr)).GetValue(engine)!;
            var scratch = (IntPtr)Field("_gdnPreparedScratch", typeof(IntPtr)).GetValue(engine)!;
            long scratchBytes = (long)Field("_gdnPreparedScratchBytes", typeof(long)).GetValue(engine)!;
            bool enabled = flags["TS_Q35_GDN_PREPARED_PREFILL"] == "1";
            Require(!enabled || (engine.PreparedGdnScratchReusesProjection && scratch == projection + checked((nint)aliasOffset)
                && scratchBytes > 0 && aliasOffset + scratchBytes <= projectionBytes), "Enabled run did not exercise the bounded projection-tail alias.");
            Require(enabled || (scratch == IntPtr.Zero && scratchBytes == 0 && !engine.PreparedGdnScratchReusesProjection), "Off run unexpectedly allocated prepared scratch.");
            geometry = new { vocab, actual_ubatch = engine.UBatch, actual_context = engine.ContextSize, actual_mixed_row_capacity = engine.MixedRowCapacity,
                actual_slots = Slots, no_drafter = !engine.HasDrafter, model.NLayer, recurrent_layers = recurrentLayers, model.NEmbd, model.NFf, model.NHead, model.NKvHead, model.HeadDim,
                model.GdnKHeads, model.GdnVHeads, packed_width = packedWidth, projection_bytes = projectionBytes,
                alias_offset = aliasOffset, prepared_scratch_bytes = scratchBytes, projection_tail_reused = engine.PreparedGdnScratchReusesProjection,
                float_bytes_per_repeat = checked(146L * vocab * sizeof(float)), input_token_count = PrefixRows * Slots + UBatch + VerifyLanes * Window + TeacherSteps * Slots,
                input_token_bytes = checked((PrefixRows * Slots + UBatch + VerifyLanes * Window + TeacherSteps * Slots) * sizeof(int)),
                input_token_binary_sha256 = Hash(MemoryMarshal.AsBytes(prefixes.SelectMany(x => x).Concat(prefill).Concat(verify).Concat(teacher.SelectMany(x => x)).ToArray().AsSpan())) };
            var slots = new Q35CudaEngine.Slot[Slots];
            slots[0] = engine.Active;
            try
            {
                for (int lane = 1; lane < Slots; lane++) slots[lane] = engine.CreateSlot(0);
                var verifySlots = slots[1..];
                for (int repeat = 0; repeat < Repeats; repeat++)
                {
                    foreach (var slot in slots) engine.ResetSlot(slot);
                    void Observe(string key, float[] values, int rows, int[]? drawn, int[] before, int[] after)
                    {
                        Require(values.LongLength == checked((long)rows * vocab), "Incomplete full-vocabulary output.");
                        string file = $"repeat-{repeat}.{key}.f32";
                        File.WriteAllBytes(Path.Combine(output, file), MemoryMarshal.AsBytes(values.AsSpan()));
                        string? drawnFile = null;
                        if (drawn != null)
                        {
                            Require(drawn.Length == rows, "Incomplete drawn tokens.");
                            drawnFile = $"repeat-{repeat}.{key}.i32";
                            File.WriteAllBytes(Path.Combine(output, drawnFile), MemoryMarshal.AsBytes(drawn.AsSpan()));
                        }
                        var observation = new { repeat, key, rows, vocab, file, bytes = checked(values.LongLength * sizeof(float)),
                            sha256 = Hash(MemoryMarshal.AsBytes(values.AsSpan())), drawn_file = drawnFile, drawn,
                            drawn_sha256 = drawn == null ? null : Hash(MemoryMarshal.AsBytes(drawn.AsSpan())), positions_before = before, positions_after = after };
                        trace.Add(observation);
                        Write(Path.Combine(output, "trace.json"), trace);
                        // Preserve every raw byte before any semantic/finiteness/A-A gate.
                        Require(values.All(float.IsFinite), $"Non-finite logits at {key}; raw retained.");
                        if (drawn != null)
                            for (int row = 0; row < rows; row++)
                            {
                                int best = 0;
                                for (int i = 1; i < vocab; i++) if (values[row * vocab + i] > values[row * vocab + best]) best = i;
                                Require(drawn[row] == best, $"Drawn argmax differs from complete CPU first-max at {key}, row {row}.");
                            }
                        if (repeat == 0) reference.Add(key, file);
                        else Require(File.ReadAllBytes(Path.Combine(output, reference[key])).AsSpan().SequenceEqual(MemoryMarshal.AsBytes(values.AsSpan())),
                            $"Reset A/A changed full F32 bits at {key}; both raw files retained.");
                        if (repeat > 0 && drawnFile != null)
                            Require(File.ReadAllBytes(Path.Combine(output, $"repeat-0.{key}.i32")).AsSpan().SequenceEqual(MemoryMarshal.AsBytes(drawn!.AsSpan())),
                                $"Reset A/A changed drawn IDs at {key}.");
                    }
                    for (int lane = 0; lane < Slots; lane++)
                    {
                        phase = $"repeat-{repeat}-prefix-{lane}";
                        engine.Active = slots[lane];
                        var logits = new float[vocab];
                        engine.Forward(prefixes[lane], logits);
                        Require(slots[lane].NPast == PrefixRows, "Prefix position drift.");
                        Observe($"prefix-{lane}", logits, 1, null, new[] { 0 }, new[] { slots[lane].NPast });
                    }
                    engine.Active = slots[0];
                    phase = $"repeat-{repeat}-mixed";
                    int[] beforeMixed = slots.Select(slot => slot.NPast).ToArray();
                    long countBefore = engine.PreparedGdnPrefillLaunchCount;
                    var prefillLogits = new float[vocab];
                    var drawnVerify = new int[VerifyLanes * Window];
                    engine.MixedVerify(slots[0], prefill, prefillLogits, verifySlots, Window, verify, drawnVerify);
                    // Preserve the public outputs even if the private-layout gate
                    // refuses readback; never substitute recomputed verify logits.
                    File.WriteAllBytes(Path.Combine(output, $"repeat-{repeat}.mixed-prefill.f32"), MemoryMarshal.AsBytes(prefillLogits.AsSpan()));
                    File.WriteAllBytes(Path.Combine(output, $"repeat-{repeat}.mixed-verify.i32"), MemoryMarshal.AsBytes(drawnVerify.AsSpan()));
                    // Read the last Head BEFORE Observe/Commit/continuation can reuse it.
                    float[] verifyLogits = ReadLastMixedLogits(engine, vocab);
                    File.WriteAllBytes(Path.Combine(output, $"repeat-{repeat}.mixed-verify.f32"), MemoryMarshal.AsBytes(verifyLogits.AsSpan()));
                    preparedCount = engine.PreparedGdnPrefillLaunchCount;
                    long delta = preparedCount - countBefore;
                    dispatch.Add(new { repeat, operation = "MixedVerify", actual_rows = UBatch + drawnVerify.Length, actual_slots = slots.Length,
                        prepared_gdn_launch_delta = delta, expected_prepared_gdn_launch_delta = enabled ? recurrentLayers : 0,
                        source_selected_projection_route = "total n=4160 >8 => NVFP4 GEMM; source-derived, no kernel launch profiler",
                        note = "Only the public prepared-launch counter is an actual dispatch count; other counts are submitted API schedule/shape." });
                    Require(delta == (enabled ? recurrentLayers : 0), "Prepared mixed-prefill dispatch coverage is incomplete.");
                    Require(slots[0].NPast == PrefixRows + UBatch && verifySlots.All(slot => slot.NPast == PrefixRows), "Mixed prefill/verify position effects differ.");
                    int[] afterMixed = slots.Select(slot => slot.NPast).ToArray();
                    Observe("mixed-prefill", prefillLogits, 1, null, beforeMixed, afterMixed);
                    Observe("mixed-verify", verifyLogits, VerifyLanes * Window, drawnVerify, beforeMixed, afterMixed);
                    phase = $"repeat-{repeat}-commit";
                    int[] kept = Enumerable.Range(1, VerifyLanes).ToArray();
                    engine.Commit(verifySlots, Window, kept);
                    Require(verifySlots.Select(slot => slot.NPast).SequenceEqual(kept.Select(k => PrefixRows + k)), "Partial/full commit position drift.");
                    for (int step = 0; step < TeacherSteps; step++)
                    {
                        phase = $"repeat-{repeat}-teacher-{step}";
                        int[] before = slots.Select(slot => slot.NPast).ToArray();
                        var logits = new float[checked(Slots * vocab)];
                        var drawn = new int[Slots];
                        engine.DecodeBatch(slots, teacher[step], logits, drawn);
                        Require(slots.Select(slot => slot.NPast).SequenceEqual(before.Select(p => p + 1)), "Teacher continuation position drift.");
                        Observe($"teacher-{step}", logits, Slots, drawn, before, slots.Select(slot => slot.NPast).ToArray());
                    }
                    Require(engine.PreparedGdnPrefillLaunchCount == preparedCount, "Unexpected prepared GDN on prefix/teacher/verify path.");
                    Require(trace.Count == (repeat + 1) * 19, "Missing output records.");
                    Manifest(false);
                }
                Require(trace.Count == 38 && reference.Count == 19, "Incomplete reset A/A inventory.");
            }
            finally
            {
                engine.Active = slots[0];
                foreach (var slot in slots.Skip(1)) if (slot != null) engine.FreeSlot(slot);
            }
            phase = "finished";
            Manifest(true);
            Console.WriteLine($"[q35-mixed-capacity] complete: 38 full-logit records, exact reset A/A, prepared launches={preparedCount}; no timing claim.");
            return 0;
        }
        catch (Exception error)
        {
            Manifest(false, error.ToString());
            throw;
        }
    }
}
