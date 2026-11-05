// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Cuda;

// Reuses the sealed mixed schedule; records allocation layout separately so the
// existing CLI, inputs, trace and full-vocabulary snapshots retain their meaning.
internal static class SharedScratchTrace
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.DeclaredOnly;
    [DllImport("cuda", EntryPoint = "cuMemGetAddressRange_v2")]
    private static extern int AddressRange(out IntPtr allocationBase, out UIntPtr allocationBytes, IntPtr pointer);
    private sealed record Range(bool Available, int? Result, long Base, ulong Bytes, string? Error);
    private static Range DriverRange(IntPtr pointer)
    {
        try { int result = AddressRange(out var start, out var size, pointer); return new(true, result, start.ToInt64(), size.ToUInt64(), null); }
        catch (Exception e) when (e is DllNotFoundException or EntryPointNotFoundException) { return new(false, null, 0, 0, e.GetType().Name); }
    }
    private static void Require(bool condition, string message) { if (!condition) throw new InvalidOperationException(message); }
    private static FieldInfo Field(string name) => typeof(Q35CudaEngine).GetField(name, Private)
        ?? throw new InvalidOperationException($"Missing private diagnostic field {name}.");
    private static IntPtr Pointer(Q35CudaEngine engine, string name)
    {
        var field = Field(name); Require(field.FieldType == typeof(IntPtr) && field.IsPrivate && !field.IsStatic, $"Unsupported pointer field {name}.");
        return (IntPtr)field.GetValue(engine)!;
    }
    private static string Hex(long value) => $"0x{value:x}";
    private static long Aligned(long bytes) => checked((Math.Max(256, bytes) + 255) & ~255L);

    // CPU geometry only, independently checks actual published spans at runtime.
    private static void ValidateRanges(bool enabled, long backing, long bytes, IReadOnlyList<Q35CudaEngine.ScratchBufferSpan> spans)
    {
        Require(spans.Count > 0 && spans.Select(s => s.Name).Distinct().Count() == spans.Count, "Missing/duplicate scratch views.");
        if (enabled) Require(backing > 0 && backing % 256 == 0 && bytes > 0 && bytes % 256 == 0, "Invalid shared owner range.");
        foreach (var span in spans)
        {
            long start = span.Pointer.ToInt64(), end = checked(start + span.CapacityBytes);
            Require(start > 0 && start % 256 == 0 && span.Bytes >= 0 && span.CapacityBytes >= Math.Max(256, span.Bytes)
                && span.CapacityBytes % 256 == 0, $"Invalid scratch extent/alignment: {span.Name}.");
            if (enabled) Require(start >= backing && end <= checked(backing + bytes), $"View outside requested owner: {span.Name}.");
        }
        foreach (var group in enabled ? spans.GroupBy(s => s.Phase) : spans.GroupBy(_ => "all independent allocations"))
        {
            long end = 0;
            foreach (var span in group.OrderBy(s => s.Pointer.ToInt64()))
            {
                Require(span.Pointer.ToInt64() >= end, $"Overlapping simultaneously live scratch views: {span.Name}.");
                end = checked(span.Pointer.ToInt64() + span.CapacityBytes);
            }
        }
    }
    private static void CpuGeometryChecks()
    {
        var a = new Q35CudaEngine.ScratchBufferSpan("a", "attention", new IntPtr(0x10000), 256, 256);
        var b = new Q35CudaEngine.ScratchBufferSpan("b", "attention", new IntPtr(0x10100), 256, 256);
        var c = new Q35CudaEngine.ScratchBufferSpan("c", "gdn", new IntPtr(0x10000), 512, 512);
        ValidateRanges(true, 0x10000, 512, new[] { a, b, c });
        void Reject(Action action) { bool rejected = false; try { action(); } catch (InvalidOperationException) { rejected = true; } Require(rejected, "Unsafe synthetic layout was accepted."); }
        Reject(() => ValidateRanges(true, 0x10000, 512, new[] { a, b with { Pointer = a.Pointer } }));
        Reject(() => ValidateRanges(false, 0, 0, new[] { a, c }));
        Reject(() => ValidateRanges(true, 0x10000, 256, new[] { b }));
        Reject(() => ValidateRanges(true, 0x10000, 512, new[] { a with { Pointer = new IntPtr(0x10001) } }));
    }
    internal static int Plan(string output)
    {
        CpuGeometryChecks();
        int result = MixedCapacityTrace.Plan(output);
        File.WriteAllText(Path.Combine(output, "shared-scratch-plan.json"), JsonSerializer.Serialize(new {
            status = "CPU_LAYOUT_PLAN_GPU_NOT_RUN", complete = false, actual_device_pointers_observed = false,
            geometry_checks = "safe cross-phase alias; same-phase/off-mode overlap, misalignment and owner overflow rejected",
            arms = new[] { "shared0-prepared0", "shared0-prepared1", "shared1-prepared0", "shared1-prepared1" },
            workflow = "unchanged mixed-capacity:4160 rows,9 slots,38 full-logit records,292 rows,18 drawn records,2 reset repetitions",
            oracle = "all four processes complete F32/I32/input/trace bits plus in-process reset AA; no timing claim",
            driver_bounds = "cuMemGetAddressRange_v2 when available; unavailable is explicitly recorded, never called passing driver coverage"
        }, Json) + "\n");
        return result;
    }
    internal static int Run(string modelPath, string output) => MixedCapacityTrace.Run(modelPath, output, diagnoseSharedScratch: true);

    internal static void Capture(Q35CudaEngine engine, string output, string checkpoint)
    {
        var spans = engine.ScratchBufferSpans.ToArray();
        var ranges = spans.Select(s => DriverRange(s.Pointer)).ToArray();
        var owned = (IEnumerable<IntPtr>)Field("_owned").GetValue(engine)!;
        var owners = owned.ToArray();
        var model = (Q35CudaEngine.ModelDesc)Field("_m").GetValue(engine)!;
        var expected = new Dictionary<string, (string Phase, int Width)> {
            ["_xn"] = ("normalize", model.NEmbd), ["_q"] = ("attention", model.NHead * model.HeadDim),
            ["_gate"] = ("attention", model.NHead * model.HeadDim), ["_attn"] = ("attention", model.NHead * model.HeadDim),
            ["_act"] = ("attention", model.NFf), ["_gdnQkv"] = ("gdn", (2 * model.GdnKHeads + model.GdnVHeads) * Q35CudaEngine.GdnHeadDim),
            ["_gdnG"] = ("gdn", model.GdnVHeads), ["_gdnBeta"] = ("gdn", model.GdnVHeads),
            ["_gdnO"] = ("gdn", model.GdnVHeads * Q35CudaEngine.GdnHeadDim), ["_gdnY"] = ("gdn", model.GdnVHeads * Q35CudaEngine.GdnHeadDim)
        };
        string[] excludedNames = { "_x", "_block", "_proj", "_logits", "_q8", "_q8d", "_fp4c", "_fp4s", "_fp4r", "_vRaw", "_vQkv", "_vG", "_vBeta" };
        var excluded = excludedNames.Select(name => new { name, pointer = Pointer(engine, name).ToInt64() }).ToArray();
        var sources = new[] { "TensorSharp.Backends.Cuda/Qwen35/Q35CudaEngine.Scratch.cs", "eng/validation/Qwen35CudaEngineProbe/SharedScratchTrace.cs" };
        string root = Environment.GetEnvironmentVariable("TS_PROBE_SOURCE_ROOT") ?? Environment.CurrentDirectory;
        var state = new Dictionary<string, object?> {
            ["schema"] = "q35-shared-scratch-trace/1", ["checkpoint"] = checkpoint, ["complete"] = false,
            ["shared_setting"] = Environment.GetEnvironmentVariable("TS_Q35_SHARED_SCRATCH"), ["actual_enabled"] = engine.SharedScratchEnabled,
            ["pool_base"] = Hex(engine.SharedScratchBase.ToInt64()), ["pool_base_address"] = engine.SharedScratchBase.ToInt64(),
            ["requested_pool_bytes"] = engine.SharedScratchPoolBytes, ["actual_saved_bytes"] = engine.SharedScratchBytesSaved,
            ["actual_mixed_rows"] = engine.MixedRowCapacity, ["actual_prepared_gdn_launches"] = engine.PreparedGdnPrefillLaunchCount,
            ["views"] = spans.Select((s, i) => new { s.Name, s.Phase, pointer = Hex(s.Pointer.ToInt64()), address = s.Pointer.ToInt64(),
                bytes = s.Bytes, capacity_bytes = s.CapacityBytes, shared_offset = engine.SharedScratchEnabled ? s.Pointer.ToInt64() - engine.SharedScratchBase.ToInt64() : (long?)null,
                private_field_matches = Pointer(engine, s.Name) == s.Pointer, driver_owner = ranges[i] }).ToArray(),
            ["phase_ranges"] = spans.GroupBy(s => s.Phase).Select(g => new { phase = g.Key, begin = g.Min(s => s.Pointer.ToInt64()), end = g.Max(s => checked(s.Pointer.ToInt64() + s.CapacityBytes)),
                views = g.Select(s => s.Name).ToArray(), live_range_meaning = "declared source liveness phase, not GPU-profiler kernel timestamps" }).ToArray(),
            ["excluded_buffers"] = excluded, ["shared_base_owned_count"] = owners.Count(p => p == engine.SharedScratchBase),
            ["driver_bounds_available_for_all_views"] = ranges.All(r => r.Available),
            ["source_files"] = sources.Select(p => new { path = p, sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(Path.Combine(root, p)))).ToLowerInvariant() }).ToArray(),
            ["no_performance_claim"] = true
        };
        void Save() => File.WriteAllText(Path.Combine(output, $"shared-scratch-{checkpoint}.json"), JsonSerializer.Serialize(state, Json) + "\n");
        Save(); // Raw pointer/driver observations survive any following failed gate.
        try
        {
            string? setting = Environment.GetEnvironmentVariable("TS_Q35_SHARED_SCRATCH");
            Require(setting is "0" or "1" && engine.SharedScratchEnabled == (setting == "1"), "Shared scratch opt-in differs from actual engine.");
            Require(engine.MixedRowCapacity == 4160 && !engine.HasDrafter, "Shared diagnostic needs exact no-drafter mixed geometry.");
            Require(spans.Length == 10 && spans.Select(s => s.Name).ToHashSet().SetEquals(expected.Keys), "Unexpected scratch view inventory.");
            ValidateRanges(engine.SharedScratchEnabled, engine.SharedScratchBase.ToInt64(), engine.SharedScratchPoolBytes, spans);
            foreach (var span in spans)
            {
                Require(Pointer(engine, span.Name) == span.Pointer && span.Phase == expected[span.Name].Phase
                    && span.Bytes == checked(4160L * expected[span.Name].Width * sizeof(float)) && span.CapacityBytes == Aligned(span.Bytes), "Published/private view or model geometry differs: " + span.Name);
            }
            if (engine.SharedScratchEnabled)
            {
                long expectedPool = spans.GroupBy(s => s.Phase).Max(g => g.Sum(s => s.CapacityBytes));
                Require(engine.SharedScratchPoolBytes == expectedPool && engine.SharedScratchBytesSaved == spans.Sum(s => Math.Max(256, s.Bytes)) - expectedPool
                    && owners.Count(p => p == engine.SharedScratchBase) == 1, "Shared requested pool/saving/unique owner differs.");
                foreach (var group in spans.GroupBy(s => s.Phase))
                {
                    long offset = 0;
                    foreach (var span in group) { Require(span.Pointer.ToInt64() == checked(engine.SharedScratchBase.ToInt64() + offset), "Shared phase did not alias the expected base/offset."); offset = checked(offset + span.CapacityBytes); }
                }
                Require(spans.Where(s => s.Pointer != engine.SharedScratchBase).All(s => !owners.Contains(s.Pointer)), "A shared interior view was separately owned.");
                Require(excluded.All(e => e.pointer < engine.SharedScratchBase.ToInt64() || e.pointer >= checked(engine.SharedScratchBase.ToInt64() + expectedPool)), "Independent persistent/quantized buffer aliases shared scratch.");
            }
            else Require(engine.SharedScratchBase == IntPtr.Zero && engine.SharedScratchPoolBytes == 0 && engine.SharedScratchBytesSaved == 0
                && spans.All(s => owners.Count(p => p == s.Pointer) == 1), "Off arm changed independent allocation ownership.");
            foreach (var (span, range) in spans.Zip(ranges))
            {
                if (!range.Available) continue;
                Require(range.Result == 0 && range.Base > 0 && range.Bytes <= long.MaxValue && span.Pointer.ToInt64() >= range.Base
                    && checked(span.Pointer.ToInt64() + span.CapacityBytes) <= checked(range.Base + (long)range.Bytes), "Driver allocation bounds refused view: " + span.Name);
                if (engine.SharedScratchEnabled) Require(range.Base == engine.SharedScratchBase.ToInt64() && range.Bytes >= (ulong)engine.SharedScratchPoolBytes, "Driver allocation differs from shared base/request.");
            }
            state["complete"] = true; state["status"] = "ACTUAL_LAYOUT_AND_REQUESTED_BOUNDS_PASS"; Save();
        }
        catch (Exception error) { state["status"] = "INCOMPLETE_LAYOUT_NOT_PASS"; state["error"] = error.ToString(); Save(); throw; }
    }
}
