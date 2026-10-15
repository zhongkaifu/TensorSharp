// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text.Json;
using System.Runtime.InteropServices;
using InferenceWeb.Tests;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Runtime;

if (args.Length is < 2 or > 10 || args[0] is not ("CPU" or "CUDA")
    || args.Skip(2).Any(flag => flag is not ("--shared-graphs" or "--dispatch-faults" or "--selected-experts" or "--swiglu-clamp" or "--bf16-sensitive" or "--slot-copy")
        && !flag.StartsWith("--quantized-source=", StringComparison.Ordinal) && !flag.StartsWith("--prefill-rows=", StringComparison.Ordinal)))
    throw new ArgumentException("Usage: GlmGraphBudgetProbe CPU|CUDA <new-output-directory> [--shared-graphs] [--dispatch-faults] [--selected-experts] [--swiglu-clamp] [--bf16-sensitive] [--slot-copy] [--quantized-source=<GLM-GGUF>] [--prefill-rows=1..1024]");
bool sharedGraphs = args.Skip(2).Contains("--shared-graphs"), testDispatchFaults = args.Skip(2).Contains("--dispatch-faults");
bool selectedExperts = args.Skip(2).Contains("--selected-experts");
bool bf16Sensitive = args.Skip(2).Contains("--bf16-sensitive");
bool testSlotCopy = args.Skip(2).Contains("--slot-copy");
string? quantizedSource = args.Skip(2).SingleOrDefault(flag => flag.StartsWith("--quantized-source=", StringComparison.Ordinal))?["--quantized-source=".Length..];
float swigluClamp = args.Skip(2).Contains("--swiglu-clamp") ? 0.0001f : quantizedSource != null ? 10 : 0;
int vocab = quantizedSource != null || bf16Sensitive ? 256 : 128, fixtureLayers = quantizedSource != null || bf16Sensitive ? 4 : 2;
int expertGroups = quantizedSource != null ? 3 : 1, seamsPerGraph = quantizedSource != null || bf16Sensitive ? 3 : 2;
string? rowsOption = args.Skip(2).SingleOrDefault(flag => flag.StartsWith("--prefill-rows=", StringComparison.Ordinal));
int? explicitPrefillRows = rowsOption == null ? null : int.Parse(rowsOption["--prefill-rows=".Length..], System.Globalization.CultureInfo.InvariantCulture);
if (explicitPrefillRows is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(explicitPrefillRows));
// Preserve historical fixture geometry through 128. Wider cases need room for
// the complete prefill and the subsequent independent/joint decode checks.
int context = explicitPrefillRows != null ? Math.Max(256, explicitPrefillRows.Value * 2) : 128;
int prefillChunk = explicitPrefillRows ?? 16;
// Larger synthetic cases explicitly own a larger selected graph floor. Keep
// the historical <=512 quota and all pressure/refusal checks unchanged.
long expertBudget = prefillChunk > 512 ? 64L << 20 : 32L << 20;
string backend = args[0], directory = Path.GetFullPath(args[1]);
if (selectedExperts && backend != "CUDA") throw new ArgumentException("Selected-expert qualification requires actual CUDA.");
if (testSlotCopy && selectedExperts) throw new ArgumentException("The dedicated exact slot-copy fixture uses resident experts; qualify selected dispatch separately.");
if (Directory.Exists(directory)) throw new IOException("Use a fresh evidence directory.");
Directory.CreateDirectory(directory);
GgmlGlmNative.RequireUnifiedBudgetSupport();
string model = quantizedSource != null || bf16Sensitive
    ? GlmDsaSyntheticModelBuilder.WriteGlm5NextCudaFixture(Path.Combine(directory, "tiny-glm5-next.gguf"),
        context, 8, quantizedSource, swigluClamp, Path.Combine(directory, "source-expert-blocks.json"), bf16Sensitive)
    : GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(Path.Combine(directory, "tiny-glm5-next.gguf"),
    4, true, numLayers: 2, mixedAttention: true, routedExperts: true,
    quantizeExperts: true, contextLength: context, indexerTopK: 8, architecture: "glm5-next", swigluClamp: swigluClamp);
int checks = 0;
void Require(bool value, string message)
{
    ++checks;
    if (!value)
    {
        string? nativeError = null;
        try { nativeError = typeof(GgmlBasicOps).Assembly.GetType("TensorSharp.GGML.GgmlNative")?
            .GetMethod("LastNativeError", System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Static)?
            .Invoke(null, ["(no native error)"]) as string; } catch { }
        throw new InvalidOperationException(message + " Native: " + nativeError);
    }
}
int bf16SensitiveTensorCount = 0;
if (bf16Sensitive)
{
    using var fixture = GgufFile.OpenForMemoryPlanning(model);
    string[] names = Enumerable.Range(0, 4).SelectMany(layer => new[] { "hc_attn_fn.weight", "hc_ffn_fn.weight" }
        .Concat(layer % 2 == 0 ? new[] { "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight", "ssm_beta.weight" } : [])
        .Select(suffix => $"blk.{layer}.{suffix}")).ToArray();
    Require(names.Length == 18 && names.All(name => fixture.Tensors.TryGetValue(name, out var tensor)
        && tensor.Type == GgmlTensorType.BF16), "Sensitive fixture does not contain all18 actual BF16 recurrent/mixing tensors.");
    bf16SensitiveTensorCount = names.Length;
}
var nativeModule = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Single(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase));
IntPtr hookLibrary = testDispatchFaults ? NativeLibrary.Load(nativeModule.FileName) : IntPtr.Zero;
var injectDispatchFault = testDispatchFaults ? Marshal.GetDelegateForFunctionPointer<BatchedDispatchFault>(
    NativeLibrary.GetExport(hookLibrary, "TSGgml_GlmTestBatchedDispatchFault")) : null;
IntPtr Load(bool selected = false) => GgmlGlmNative.LoadModel(model, backend == "CUDA" ? 1 : 0, context, prefillChunk, 2,
    nCpuMoe: backend == "CUDA" && !selected ? 0 : fixtureLayers, backendName: backend, tp: 1, ctxIsHardLimit: true, loadMtp: false);
var samples = new List<GgmlAllocationUsage[]>();
var executions = new List<object>();
GgmlHostMoeExpertCacheScope? selectedScope = null;
float[] Evaluate(IntPtr handle, GgmlCacheBudgetScope? scope = null, MemoryBudget? budget = null,
    bool selected = false, BudgetReservation? expertEnvelope = null)
{
    Require(handle != IntPtr.Zero, "Successful load returned null.");
    Require(GgmlGlmNative.ConfigureGraphCache(handle, 1), "Graph-cache capacity configuration failed.");
    if (sharedGraphs) Require(GgmlGlmNative.ConfigureSharedGraphOwner(handle, true), "Quiescent shared graph ownership configuration failed.");
    if (selected)
    {
        Require(GgmlGlmNative.SelectedExpertGroupCount(handle) == expertGroups, "Exact mapped quantized expert group geometry was not retained.");
        int allocations = scope?.ActiveAllocations ?? 0;
        var cacheBefore = GgmlBasicOps.GetHostMoeExpertCacheUsage();
        long minimum = GgmlGlmNative.SelectedExpertMinimumBudget(handle, 3);
        Require(minimum > 0 && minimum <= selectedScope!.BudgetBytes, "Measured full-union compact minimum exceeds the test's dedicated expert ceiling.");
        Require((scope?.ActiveAllocations ?? 0) == allocations && GgmlBasicOps.GetHostMoeExpertCacheUsage() == cacheBefore,
            "Size-only compact planning allocated a payload, changed a resident table or published cache usage.");
        long fullBudget = selectedScope!.BudgetBytes;
        selectedScope.UpdateBudget(minimum - 1);
        Require(!GgmlGlmNative.ConfigureSelectedExpertSeams(handle, true) && GgmlGlmNative.NPast(handle) == 0
            && GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Insufficient expert scope was not refused safely before graph dispatch.");
        selectedScope.UpdateBudget(fullBudget);
        Require(GgmlGlmNative.ConfigureSelectedExpertSeams(handle, true), "Quiescent selected-expert scheduler adapter configuration failed.");
        executions.Add(new { expert_groups = expertGroups, planned_decode_rows = 3, minimum_cache_bytes = minimum });
    }
    int[] slots = [0, -1, -1];
    BudgetReservation? serial = budget?.Reserve([new("ram", 256L << 20), new("cuda", 256L << 20)]);
    BudgetReservation? joint = budget?.Reserve([new("ram", 256L << 20), new("cuda", 256L << 20)]);
    scope?.SetSharedJointEnvelope(joint);
    var output = new List<float>();
    try
    {
        using (scope?.EnterExecution(serial!))
        {
            slots[1] = GgmlGlmNative.SlotAlloc(handle); slots[2] = GgmlGlmNative.SlotAlloc(handle);
            Require(slots.All(s => s >= 0), "Independent slots could not be allocated.");
        }
        // A one-byte request cannot fund any scheduler allocation. Shared mode
        // must draw those actual payloads from the separate global graph owner.
        using (var prefillOwner = sharedGraphs && budget != null ? budget.Reserve([new("ram", selected ? 64L << 20 : 1), new("cuda", 1)]) : null)
        {
            using (scope?.EnterExecution(prefillOwner ?? serial!))
            for (int lane = 0; lane < 3; lane++)
            {
                Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]), "Slot selection failed.");
                int[] prompt = Enumerable.Range(0, explicitPrefillRows ?? (selectedExperts ? 1 : 17 + 5 * lane)).Select(i => 3 + (i * 11 + lane) % 90).ToArray();
                float[] logits = new float[GgmlGlmNative.VocabSize(handle)];
                Require(GgmlGlmNative.Forward(handle, prompt, logits), "Prefill failed.");
                Require(logits.All(float.IsFinite), "Prefill produced nonfinite logits.");
                output.AddRange(logits);
            }
            if (sharedGraphs && scope != null)
                Require(scope.AllocationUsage.Any(a => a.Kind == 5 && a.CommittedBytes > 0), "Solo prefill graphs were not charged to the global owner.");
        }
        // Real late arrival: cached graphs already own facade backend IDs. Slot
        // admission measures them before materializing a further native slot.
        using (scope?.EnterExecution(serial!))
        {
            int late = GgmlGlmNative.SlotAlloc(handle);
            Require(late >= 0, "Late request slot allocation after cached prefill failed.");
            Require(GgmlGlmNative.SlotFree(handle, late), "Late request slot physical release failed.");
        }
        if (sharedGraphs)
        {
            Require(!GgmlGlmNative.ConfigureSharedGraphOwner(handle, false), "A live cached graph changed its physical owner.");
            if (scope != null) Require(scope.AllocationUsage.Any(a => a.Kind == 5 && a.CommittedBytes > 0), "Closing the request envelope refunded a live shared graph.");
        }
        if (scope != null) samples.Add(scope.AllocationUsage.ToArray());
        int[] positions = explicitPrefillRows is int promptRows ? [promptRows, promptRows, promptRows]
            : selectedExperts ? [1, 1, 1] : [17, 22, 27];
        float[] batch = new float[3 * GgmlGlmNative.VocabSize(handle)];
        Require(GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Serial prefill published batch evidence.");
        ulong accepted = 0;
        void CheckExecution(int[] before)
        {
            var actual = GgmlGlmNative.TryGetLastBatchedExecution(handle);
            Require(actual != null, "Successful N3 graph did not publish native evidence.");
            Require(actual!.ExecutionId == accepted + 1 && actual.Rows == 3 && actual.Nodes > 0 && actual.BackendSplits > 0,
                "Native successful graph id/rows/nodes/backend spans are invalid.");
            Require(actual.Slots.SequenceEqual(slots) && actual.PositionsBefore.SequenceEqual(before)
                && actual.PositionsAfter.SequenceEqual(before.Select(p => p + 1)), "Native evidence does not describe the exact slot heads.");
            accepted = actual.ExecutionId;
            executions.Add(new { budgeted = scope != null, actual });
        }
        for (int step = 0; step < 4; step++)
        {
            using (scope?.EnterExecution(joint!))
                Require(GgmlGlmNative.ForwardBatchedDecode(handle, slots, [17 + step, 31 + step, 43 + step], positions, batch),
                    "Native N3 single-graph decode was declined.");
            Require(batch.All(float.IsFinite), "N3 decode produced nonfinite logits.");
            CheckExecution(positions);
            output.AddRange(batch);
            for (int lane = 0; lane < 3; lane++) positions[lane]++;
            if (step == 0)
            {
                int livePayloads = scope?.ActiveAllocations ?? 0;
                Require(GgmlGlmNative.TrimMappedResidency(handle), "Quiescent mapped source trim was refused.");
                Require((scope?.ActiveAllocations ?? 0) == livePayloads,
                    "A mapped residency hint refunded unrelated live physical payload credit.");
                foreach (var (ids, tokens, heads) in new[] {
                    (new[] { slots[0], slots[0], slots[2] }, new[] { 17, 31, 43 }, (int[])positions.Clone()),
                    ((int[])slots.Clone(), new[] { 17, 31, 43 }, new[] { positions[0] + 1, positions[1], positions[2] }),
                    ((int[])slots.Clone(), new[] { -1, 31, 43 }, (int[])positions.Clone()) })
                {
                    Require(!GgmlGlmNative.ForwardBatchedDecode(handle, ids, tokens, heads, batch), "Invalid native batch was accepted.");
                    Require(GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Failed batch replayed stale success evidence.");
                    for (int lane = 0; lane < 3; ++lane)
                    {
                        Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]), "Failed-batch head check cannot select slot.");
                        Require(GgmlGlmNative.NPast(handle) == positions[lane], "Rejected batch advanced slot state.");
                    }
                }
            }
        }
        if (scope != null)
        {
            samples.Add(scope.AllocationUsage.ToArray());
            Require(scope.AllocationUsage.Any(a => a.Kind == 2 && a.CommittedBytes > 0), "Slot/weight/context payloads were not charged.");
            Require(scope.AllocationUsage.Any(a => a.Kind == 5 && a.CommittedBytes > 0), "N3 graph payloads were not charged to the joint owner.");
        }
        Require(GgmlGlmNative.ConfigureGraphCache(handle, 1, 1), "Graph reclamation failed.");
        if (scope != null)
        {
            Require(!scope.AllocationUsage.Any(a => a.Kind == 5), "Joint graph credit survived physical graph reclamation.");
            // Refuse a fresh graph without consuming a token or corrupting state.
            using var tiny = budget!.Reserve([new("ram", 1), new("cuda", 1)]);
            scope.SetSharedJointEnvelope(tiny);
            using (scope.EnterExecution(tiny))
                Require(!GgmlGlmNative.ForwardBatchedDecode(handle, slots, [71, 72, 73], positions, batch), "One-byte joint graph envelope did not refuse.");
            Require(GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Refused graph published success evidence.");
            for (int lane = 0; lane < 3; ++lane)
            {
                Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]), "Graph-refusal head check cannot select slot.");
                Require(GgmlGlmNative.NPast(handle) == positions[lane], "Refused joint graph advanced slot state.");
            }
            scope.SetSharedJointEnvelope(joint);
            using (scope.EnterExecution(joint!))
                Require(GgmlGlmNative.ForwardBatchedDecode(handle, slots, [71, 72, 73], positions, batch), "Joint graph retry failed.");
            CheckExecution(positions);
            for (int lane = 0; lane < 3; ++lane) positions[lane]++;
            Require(GgmlGlmNative.ConfigureGraphCache(handle, 1, 1), "Retry graph reclamation failed.");
            Require(GgmlGlmNative.SetActiveSlot(handle, slots[1]), "Refusal slot selection failed.");
            int before = GgmlGlmNative.NPast(handle);
            if (sharedGraphs) scope.SetSharedJointEnvelope(tiny);
            using (scope.EnterExecution(tiny))
                Require(!GgmlGlmNative.Forward(handle, [71], new float[GgmlGlmNative.VocabSize(handle)]), "One-byte graph envelope did not refuse.");
            Require(GgmlGlmNative.NPast(handle) == before, "Refused graph consumed a token.");
            Require(GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Refused serial graph left batch success evidence visible.");
            if (sharedGraphs) scope.SetSharedJointEnvelope(joint);
            using (scope.EnterExecution(serial!))
                Require(GgmlGlmNative.Forward(handle, [71], new float[GgmlGlmNative.VocabSize(handle)]), "Graph retry failed.");
            Require(GgmlGlmNative.NPast(handle) == before + 1, "Successful retry did not advance exactly one token.");
            positions[1]++;
        }
        // A graph has really executed and mutated state before each fault. None
        // of those failures may silently fall back, publish stale success, or
        // leave any sibling row eligible for a serial retry.
        var faults = testDispatchFaults ? new List<(int, int)> { (1, 1), (1, 2), (2, 1), (2, 2) } : [];
        if (selected) faults.Insert(0, (0, 0)); // Actual active-scope quota refusal after routing.
        if (selected && testDispatchFaults) faults.AddRange([(3, 1), (3, 2), (4, 1), (5, 1)]);
        long finalDrainHelperRows = 0;
        foreach (var (stage, kind) in faults)
        {
            var failureCacheBefore = stage is 4 or 5 ? GgmlBasicOps.GetHostMoeExpertCacheUsage() : default;
            var finalDrainExecutionBefore = stage == 5 ? GgmlGlmNative.SelectedExpertExecutionStats(handle) : default;
            const float unpublishedLogit = -12345.25f;
            if (stage is 4 or 5) Array.Fill(batch, unpublishedLogit);
            using var refusedExpertOwner = stage == 0 ? budget!.Reserve([new("cuda", 1)]) : null;
            if (stage == 0)
            {
                // An installed phase plan protects its immutable minimum.
                // Refuse a real payload allocation instead of invalidating that
                // plan; retire the old physical owner before changing envelopes.
                long originalBudget = selectedScope!.BudgetBytes;
                var before = selectedScope.ReadUsage();
                bool rejected = false;
                try { selectedScope.UpdateBudget(0); }
                catch (InvalidOperationException) { rejected = true; }
                Require(rejected && selectedScope.BudgetBytes == originalBudget
                    && selectedScope.ReadUsage() == before, "Sub-floor update mutated the installed phase plan or cache.");
                selectedScope.Trim(0);
                scope!.SetSharedExpertEnvelope(refusedExpertOwner);
            }
            else injectDispatchFault!(stage, kind);
            bool threw = false;
            try
            {
                using (scope?.EnterExecution(joint!))
                    GgmlGlmNative.ForwardBatchedDecode(handle, slots, [71, 72, 73], positions, batch);
            }
            catch (InvalidOperationException error) when (error.Message.Contains("after dispatch", StringComparison.Ordinal)) { threw = true; }
            Require(threw, $"Post-dispatch stage {stage}/kind {kind} did not throw before serial fallback.");
            Require(GgmlGlmNative.TryGetLastBatchedExecution(handle) == null, "Post-dispatch failure published success evidence.");
            if (stage is 4 or 5)
            {
                if (stage == 4)
                    Require(GgmlBasicOps.GetHostMoeExpertCacheUsage() == failureCacheBefore,
                        "Failed selected-ID download reached or mutated the expert cache.");
                else
                {
                    finalDrainHelperRows = GgmlBasicOps.GetHostMoeExpertCacheUsage().Calls - failureCacheBefore.Calls;
                    Require(finalDrainHelperRows == 3L * seamsPerGraph,
                        "Reported final drain error was not injected after every expert helper completed all N3 rows.");
                    Require(GgmlGlmNative.SelectedExpertExecutionStats(handle) == finalDrainExecutionBefore,
                        "Reported final drain error published successful selected graph/seam counters.");
                }
                Require(batch.All(value => BitConverter.SingleToInt32Bits(value) == BitConverter.SingleToInt32Bits(unpublishedLogit)),
                    "Reported selected-expert download/drain error published logits.");
            }
            using (scope?.EnterExecution(serial!))
                for (int lane = 0; lane < 3; ++lane)
                {
                    Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]), "Cannot select a failed participating slot.");
                    int head = GgmlGlmNative.NPast(handle);
                    if (stage is 4 or 5) Require(head == positions[lane], "Reported selected-expert download/drain error published a participating slot head.");
                    Require(!GgmlGlmNative.Forward(handle, [91], new float[GgmlGlmNative.VocabSize(handle)]),
                        "A partially mutated slot accepted an unsafe serial retry.");
                    Require(GgmlGlmNative.NPast(handle) == head, "Fenced serial retry mutated its slot head.");
                }
            if (stage == 4)
                executions.Add(new { selected_ids_download_failure = true, stage, kind,
                    cache_unchanged = true, logits_unpublished = true, heads_unpublished = true,
                    scope = "Injected reported download error, not an actual CUDA driver fault; participating slots remain fenced until explicit reset." });
            if (stage == 0)
            {
                selectedScope!.Trim(0);
                scope!.SetSharedExpertEnvelope(expertEnvelope
                    ?? throw new InvalidOperationException("Actual expert ownership is required for retry."));
            }
            using (scope?.EnterExecution(serial!))
                for (int lane = 0; lane < 3; ++lane)
                {
                    Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]) && GgmlGlmNative.ResetChecked(handle),
                        "Explicit successful reset did not recover a fenced slot.");
                    Require(GgmlGlmNative.Forward(handle, [3 + lane], new float[GgmlGlmNative.VocabSize(handle)]),
                        "Reset slot could not prefill again.");
                    positions[lane] = 1;
                }
        }
        using (scope?.EnterExecution(joint!))
            Require(GgmlGlmNative.ForwardBatchedDecode(handle, slots, [71, 72, 73], positions, batch), "Explicit reset did not recover N3.");
        CheckExecution(positions);
        Require(GgmlGlmNative.ConfigureGraphCache(handle, 1, 1), "Final graph reclamation failed.");
        if (selectedExperts)
        {
            // Compare row departure on fresh identical sibling state, so the
            // intentionally different number of refused/reset attempts above
            // cannot affect the reference. The departed primary is physically
            // freed before the exact N2 graph is built or submitted.
            using (scope?.EnterExecution(serial!))
                for (int lane = 1; lane < 3; ++lane)
                {
                    Require(GgmlGlmNative.SetActiveSlot(handle, slots[lane]) && GgmlGlmNative.ResetChecked(handle),
                        "Cannot establish the sibling continuation reference.");
                    var sibling = new float[GgmlGlmNative.VocabSize(handle)];
                    Require(GgmlGlmNative.Forward(handle, [23 + lane], sibling), "Sibling prefill after reset failed.");
                    output.AddRange(sibling);
                }
            Require(GgmlGlmNative.SlotFree(handle, slots[0]), "Departing primary native slot was not physically freed.");
            slots[0] = -1;
            var pair = new float[2 * GgmlGlmNative.VocabSize(handle)];
            using (scope?.EnterExecution(joint!))
                Require(GgmlGlmNative.ForwardBatchedDecode(handle, [slots[1], slots[2]], [57, 73], [1, 1], pair),
                    "N2 sibling continuation after row departure was refused.");
            var departure = GgmlGlmNative.TryGetLastBatchedExecution(handle);
            Require(departure != null && departure.ExecutionId == accepted + 1 && departure.Rows == 2
                && departure.Slots.SequenceEqual(new[] { slots[1], slots[2] })
                && departure.PositionsBefore.SequenceEqual(new[] { 1, 1 })
                && departure.PositionsAfter.SequenceEqual(new[] { 2, 2 }), "N2 completion published incorrect sibling heads.");
            Require(pair.All(float.IsFinite), "N2 sibling continuation produced nonfinite logits.");
            output.AddRange(pair); executions.Add(new { row_departure = departure });
            Require(GgmlGlmNative.ConfigureGraphCache(handle, 1, 1), "N2 graph physical reclamation failed.");
            if (selected && testDispatchFaults)
                executions.Add(new { selected_final_drain_failure = true, stage = 5, kind = 1,
                    participating_rows = 3, expert_seams = seamsPerGraph, completed_helper_rows = finalDrainHelperRows,
                    successful_graphs_unchanged = true, logits_unpublished = true, heads_unpublished = true,
                    slots_fenced = true, reset_recovered_n3 = true, primary_departure_recovered_n2 = true,
                    scope = "Injected reported error after the last expert seam drain, not an actual CUDA driver fault; cache helpers completed but logits, heads and success counters remained unpublished." });
        }
        if (selected)
        {
            var seamEvidence = GgmlGlmNative.SelectedExpertExecutionStats(handle);
            Require(seamEvidence.Graphs > 0 && seamEvidence.Seams == (ulong)seamsPerGraph * seamEvidence.Graphs,
                "Actual successful routed graph did not execute exactly every routed expert seam.");
            var cacheEvidence = GgmlBasicOps.GetHostMoeExpertCacheUsage();
            Require(cacheEvidence.Calls > 0 && cacheEvidence.Hits > 0 && cacheEvidence.Misses > 0,
                "Selected-expert reuse was not observed across real layer sources/rows.");
            executions.Add(new { selected_expert_graphs = seamEvidence.Graphs, selected_expert_seams = seamEvidence.Seams, cache = cacheEvidence });
        }
        if (sharedGraphs) Require(GgmlGlmNative.ConfigureSharedGraphOwner(handle, false), "Released graph cache did not permit a new owner policy.");
        Require(GgmlGlmNative.SlotFree(handle, slots[1]) && GgmlGlmNative.SlotFree(handle, slots[2]), "Slot release failed.");
        slots[1] = slots[2] = -1;
        scope?.SetSharedJointEnvelope(null);
        return output.ToArray();
    }
    finally
    {
        // The caller physically destroys the entire model before disposing any
        // surviving envelope children after a failed assertion.
        joint?.Dispose(); serial?.Dispose();
    }
}
if (testSlotCopy)
{
    Require(GgmlGlmNative.SupportsSlotCopy(), "Native exact slot-copy capability is unavailable.");
    SlotCopyFault? copyFault = testDispatchFaults ? Marshal.GetDelegateForFunctionPointer<SlotCopyFault>(
        NativeLibrary.GetExport(hookLibrary, "TSGgml_GlmTestSlotCopyFault")) : null;
    SlotAllocFault? allocFault = testDispatchFaults ? Marshal.GetDelegateForFunctionPointer<SlotAllocFault>(
        NativeLibrary.GetExport(hookLibrary, "TSGgml_GlmTestSlotAllocFault")) : null;
    SlotCompare? compareSlots = testDispatchFaults ? Marshal.GetDelegateForFunctionPointer<SlotCompare>(
        NativeLibrary.GetExport(hookLibrary, "TSGgml_GlmTestSlotCompare")) : null;
    var copyBudget = new MemoryBudget([new("ram", 1L << 30), new("cuda", 1L << 30)]);
    var copyEvidence = new List<object>();
    int continuationVectors = 0, completeStateComparisons = 0;
    long completeStateBytes = 0, allocatedSlotBytes = 0;
    using (var copyScope = new GgmlCacheBudgetScope(copyBudget,
        backend == "CUDA" ? new[] { new[] { "cuda" }, new[] { "ram" } } : new[] { new[] { "ram" } },
        includeGraphBuffers: true, hostPools: ["ram"]))
    {
        using var serialOwner = copyBudget.Reserve([new("ram", 256L << 20), new("cuda", 256L << 20)]);
        using var graphOwner = copyBudget.Reserve([new("ram", 256L << 20), new("cuda", 256L << 20)]);
        using var zeroOwner = copyBudget.Reserve([new("ram", 0), new("cuda", 0)]);
        copyScope.SetSharedJointEnvelope(graphOwner);
        IntPtr handle = Load();
        try
        {
            Require(handle != IntPtr.Zero, "Slot-copy model did not load.");
            Require(GgmlGlmNative.ConfigureGraphCache(handle, 1), "Slot-copy graph limit refused.");
            if (sharedGraphs) Require(GgmlGlmNative.ConfigureSharedGraphOwner(handle, true), "Slot-copy shared graph owner refused.");
            int[] slots = [0, -1, -1, -1];
            using (copyScope.EnterExecution(serialOwner))
                for (int i = 1; i < slots.Length; ++i) slots[i] = GgmlGlmNative.SlotAlloc(handle);
            Require(slots.All(slot => slot >= 0) && slots.Distinct().Count() == 4, "Slot-copy requires four independent preallocated slots.");
            allocatedSlotBytes = GgmlGlmNative.SlotBytes(handle, slots[0]);
            Require(allocatedSlotBytes > 0 && slots.All(slot => GgmlGlmNative.SlotBytes(handle, slot) == allocatedSlotBytes)
                && GgmlGlmNative.SlotBytes(handle, -1) == 0, "Actual complete slot-byte query is inconsistent.");
            var ownedSlotCharges = copyScope.AllocationUsage.Where(row => row.Kind == 2).Sum(row => row.CommittedBytes);
            Require(ownedSlotCharges >= allocatedSlotBytes * 4, "Four live slot payloads are absent from physical accounting.");
            (string, long, long)[] Charges() => copyBudget.Snapshot().Select(row => (row.Pool, row.Committed, row.Reserved)).ToArray();
            int Head(int slot)
            {
                Require(GgmlGlmNative.SetActiveSlot(handle, slot), "Copy fixture slot selection failed.");
                return GgmlGlmNative.NPast(handle);
            }
            float[] Forward(int slot, int[] tokens)
            {
                Head(slot);
                var logits = new float[vocab];
                using (copyScope.EnterExecution(serialOwner))
                    Require(GgmlGlmNative.Forward(handle, tokens, logits), "Slot-copy continuation failed.");
                Require(logits.All(float.IsFinite), "Slot-copy continuation is nonfinite.");
                return logits;
            }
            void Reset(int slot)
            {
                Head(slot);
                using (copyScope.EnterExecution(serialOwner))
                    Require(GgmlGlmNative.ResetChecked(handle), "Copy destination checked reset failed.");
                Require(Head(slot) == 0, "Checked reset did not clear destination head.");
            }
            void Equal(float[] left, float[] right)
            {
                Require(left.Select(BitConverter.SingleToInt32Bits).SequenceEqual(right.Select(BitConverter.SingleToInt32Bits)),
                    "Full continuation logits differ from independently prefilling the same history.");
                ++continuationVectors;
            }
            void CompleteStateEqual(int left, int right)
            {
                if (compareSlots == null) return;
                Require(compareSlots(handle, left, right, out long bytes, out int tensors) == 1 && bytes > 0 && tensors >= 4,
                    "Complete MLA/indexer/KDA cache bytes differ after copy or independent continuation.");
                ++completeStateComparisons; completeStateBytes += bytes;
            }
            bool Copy(int source, int target, int head)
            {
                var before = Charges(); int allocations = copyScope.ActiveAllocations;
                bool copied;
                using (copyScope.EnterExecution(zeroOwner)) copied = GgmlGlmNative.SlotCopyInto(handle, source, target, head);
                Require(before.SequenceEqual(Charges()) && allocations == copyScope.ActiveAllocations,
                    "Preallocated slot copy allocated or refunded a native payload/graph.");
                return copied;
            }
            if (allocFault != null)
                for (int stage = 1; stage <= 3; ++stage)
                {
                    var before = Charges(); int allocations = copyScope.ActiveAllocations;
                    allocFault(stage);
                    using (copyScope.EnterExecution(serialOwner))
                        Require(GgmlGlmNative.SlotAlloc(handle) == -1, "Injected slot construction OOM crossed ABI or returned a live slot.");
                    Require(before.SequenceEqual(Charges()) && allocations == copyScope.ActiveAllocations && copyScope.CallbackError == null,
                        "Slot construction failure leaked an unpublished payload or refunded a live owner.");
                    copyEvidence.Add(new { allocation_fault_stage = stage, rejected_without_payload_leak = true });
                }
            int[] prefix = Enumerable.Range(0, 64).Select(i => 3 + i * 11 % 90).ToArray();
            var sourcePrefix = Forward(slots[0], prefix);
            Forward(slots[3], prefix.Take(7).ToArray());
            Head(slots[3]);
            Require(Copy(slots[0], slots[1], 64) && GgmlGlmNative.NPast(handle) == 7,
                "Exact copy failed or changed the unrelated active selection.");
            Require(Head(slots[0]) == 64 && Head(slots[1]) == 64, "Exact copy changed source or failed to publish destination head.");
            CompleteStateEqual(slots[0], slots[1]);
            Equal(sourcePrefix, Forward(slots[2], prefix));
            CompleteStateEqual(slots[0], slots[2]);
            Require(!Copy(slots[0], slots[0], 64) && !Copy(slots[0], slots[1], 63) && !Copy(-1, slots[1], 64),
                "Invalid identity or nonexact source head was accepted.");
            Require(Head(slots[1]) == 64, "Preflight refusal changed destination head.");
            int next = 17;
            void ContinueAll()
            {
                var original = Forward(slots[0], [next]);
                Equal(original, Forward(slots[1], [next]));
                Equal(original, Forward(slots[2], [next]));
                CompleteStateEqual(slots[0], slots[1]); CompleteStateEqual(slots[0], slots[2]);
                ++next;
            }
            for (int i = 0; i < 4; ++i) ContinueAll();
            if (copyFault != null)
                foreach (int stage in new[] { 1, 2, 3 })
                    foreach (int kind in new[] { 1, 2 })
                    {
                        Reset(slots[1]);
                        int head = Head(slots[0]); Head(slots[3]);
                        copyFault(stage, kind);
                        Require(!Copy(slots[0], slots[1], head) && GgmlGlmNative.NPast(handle) == 7,
                            "Injected copy failure escaped ABI, published success or changed active selection.");
                        Require(Head(slots[1]) == 0 && Head(slots[0]) == head, "Failed copy published a destination head or changed source head.");
                        CompleteStateEqual(slots[0], slots[2]);
                        if (stage >= 2)
                        {
                            Head(slots[1]);
                            var sentinel = Enumerable.Repeat(123456f, vocab).ToArray();
                            using (copyScope.EnterExecution(serialOwner))
                                Require(!GgmlGlmNative.Forward(handle, [21], sentinel), "Partially copied destination executed before reset.");
                            Require(sentinel.All(value => value == 123456f) && Head(slots[1]) == 0,
                                "Poisoned destination published logits/head after failed copy.");
                            Require(!Copy(slots[0], slots[1], head), "Poisoned destination accepted another copy before reset.");
                        }
                        Reset(slots[1]);
                        Require(Copy(slots[0], slots[1], head), "Copy did not recover after checked destination reset.");
                        CompleteStateEqual(slots[0], slots[1]);
                        ContinueAll();
                        copyEvidence.Add(new { copy_fault_stage = stage, kind, source_unchanged = true,
                            destination_head_unpublished = true, partial_copy_reset_required = stage >= 2,
                            checked_reset_recovery = true, reported_or_recoverable_only = true });
                    }
        }
        finally { if (handle != IntPtr.Zero) GgmlGlmNative.Free(handle); }
        copyScope.SetSharedJointEnvelope(null);
        serialOwner.Dispose(); graphOwner.Dispose(); zeroOwner.Dispose();
        Require(copyScope.ActiveAllocations == 0 && copyScope.CallbackError == null
            && copyBudget.Snapshot().All(row => row.Reserved + row.Committed == 0), "Slot-copy teardown leaked physical payload ownership.");
    }
    var copyNative = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .Where(module => module.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
        .Select(module => new { path = module.FileName, sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(module.FileName))).ToLowerInvariant() }).ToArray();
    Require(copyNative.Length == 1, "Slot-copy fixture loaded an ambiguous native module.");
    File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new {
        passed = true, backend, checks, slot_copy_tested = true, fault_hooks_tested = testDispatchFaults,
        native = copyNative, fixture_sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes(model))).ToLowerInvariant(),
        actual_allocated_slot_bytes = allocatedSlotBytes, complete_state_comparisons = completeStateComparisons,
        complete_state_bytes_compared = completeStateBytes, full_continuation_vectors = continuationVectors,
        complete_cache_bytes_checked = compareSlots != null, fault_cases = copyEvidence, final_budget = copyBudget.Snapshot(),
        scope = "Four preallocated mixed KDA/MLA slots; allocation-free exact-head state copies, independent complete-logit continuation, checked reset and physical teardown. Hook builds additionally compare every cache byte and inject three construction / six copy faults. No real-model TTFT, MTP/TP, other-device or OS-hard-cap acceptance."
    }, new JsonSerializerOptions { WriteIndented = true }));
    if (hookLibrary != IntPtr.Zero) NativeLibrary.Free(hookLibrary);
    Console.WriteLine($"PASS {checks} exact slot-copy checks; complete-state comparisons={completeStateComparisons}, continuation vectors={continuationVectors}");
    return;
}
float[] reference;
IntPtr first = Load();
try { reference = Evaluate(first); }
finally { if (first != IntPtr.Zero) GgmlGlmNative.Free(first); }
var budget = new MemoryBudget([new("ram", 1), new("cuda", 1)]);
float[] actual;
long refusals;
using (var scope = new GgmlCacheBudgetScope(budget,
    backend == "CUDA" ? new[] { new[] { "cuda" }, new[] { "ram" } } : new[] { new[] { "ram" } },
    includeGraphBuffers: true, hostPools: ["ram"]))
{
    using var experts = selectedExperts ? new GgmlHostMoeExpertCacheScope(expertBudget, prefillChunk, expertGroups, 3) : null;
    selectedScope = experts;
    IntPtr refused = Load();
    try { Require(refused == IntPtr.Zero, "One-byte physical payload quota did not refuse model loading."); }
    finally { if (refused != IntPtr.Zero) GgmlGlmNative.Free(refused); }
    Require(scope.ActiveAllocations == 0 && budget.Snapshot().All(p => p.Reserved + p.Committed == 0), "Refused load leaked payload ownership.");
    Require(budget.TrySetCapacity("ram", 1L << 30) && budget.TrySetCapacity("cuda", 1L << 30), "Failed to raise quota after refusal.");
    using var expertOwner = selectedExperts ? budget.Reserve([new("cuda", expertBudget)]) : null;
    if (expertOwner != null) scope.SetSharedExpertEnvelope(expertOwner);
    IntPtr handle = Load(selectedExperts);
    try { actual = Evaluate(handle, scope, budget, selectedExperts, expertOwner); }
    finally { if (handle != IntPtr.Zero) GgmlGlmNative.Free(handle); }
    experts?.Dispose(); // Retire shared host file workspace while accounting is attached.
    if (expertOwner != null) { scope.SetSharedExpertEnvelope(null); expertOwner.Dispose(); }
    refusals = scope.AllocationRefusals.Count;
    Require(refusals >= 2 && scope.CallbackError == null, "Expected ordinary load and graph pressure without callback exceptions.");
    Require(scope.ActiveAllocations == 0 && budget.Snapshot().All(p => p.Reserved + p.Committed == 0), "Model teardown leaked physical payload ownership.");
}
string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
byte[] encoded = new byte[actual.Length * sizeof(float)]; Buffer.BlockCopy(actual, 0, encoded, 0, encoded.Length);
File.WriteAllBytes(Path.Combine(directory, "logits.f32"), encoded);
byte[] referenceBytes = new byte[reference.Length * sizeof(float)]; Buffer.BlockCopy(reference, 0, referenceBytes, 0, referenceBytes.Length);
File.WriteAllBytes(Path.Combine(directory, "reference-logits.f32"), referenceBytes);
bool bitExact = reference.Select(BitConverter.SingleToInt32Bits).SequenceEqual(actual.Select(BitConverter.SingleToInt32Bits));
var delta = reference.Zip(actual, (a, b) => Math.Abs((double)a - b)).ToArray();
var comparedRows = Enumerable.Range(0, reference.Length / vocab).Select(row => {
    int Argmax(float[] values) => Enumerable.Range(0, vocab).MaxBy(i => values[row * vocab + i]);
    return new { row, max_abs = delta.Skip(row * vocab).Take(vocab).Max(), reference_argmax = Argmax(reference), actual_argmax = Argmax(actual) };
}).ToArray();
File.WriteAllText(Path.Combine(directory, "logit-comparison.json"), JsonSerializer.Serialize(new {
    passed = bitExact, selectedExperts, swigluClamp, quantizedSource, expertGroups, prefill_rows = explicitPrefillRows,
    bf16_sensitive_projections = bf16Sensitive, bf16_sensitive_tensor_count = bf16SensitiveTensorCount, checks, logit_elements = actual.Length,
    differing_elements = reference.Zip(actual).Count(pair => BitConverter.SingleToInt32Bits(pair.First) != BitConverter.SingleToInt32Bits(pair.Second)),
    max_abs = delta.Max(), rmse = Math.Sqrt(delta.Average(d => d * d)), rows = comparedRows,
    logits_sha256 = Hash(encoded), reference_sha256 = Hash(referenceBytes), executions
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"Full-logit comparison: exact={bitExact}, max_abs={delta.Max():G9}");
Require(bitExact, "Native payload accounting or selected-expert adaptation changed prefill or N3 full logits.");
var native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
    .Where(m => m.ModuleName.Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
    .Select(m => new { path = m.FileName, sha256 = Hash(File.ReadAllBytes(m.FileName)) }).ToArray();
Require(native.Length == 1, "Expected exactly one native module.");
File.WriteAllText(Path.Combine(directory, "report.json"), JsonSerializer.Serialize(new
{
    passed = true, backend, checks, refusals, shared_graph_owner = sharedGraphs, dispatch_faults_tested = testDispatchFaults,
    selected_experts_tested = selectedExperts,
    bf16_sensitive_projections = bf16Sensitive, bf16_sensitive_tensor_count = bf16SensitiveTensorCount,
    swiglu_clamp = swigluClamp,
    quantized_source = quantizedSource, expert_groups = expertGroups,
    prefill_rows = explicitPrefillRows,
    native, fixture_sha256 = Hash(File.ReadAllBytes(model)),
    full_logits_bit_exact = true, logit_elements = actual.Length, logits_sha256 = Hash(encoded), samples, executions,
    peaks = budget.HighWatermarks(), final_budget = budget.Snapshot(),
    scope = $"Synthetic mixed KDA/MLA GLM5-next with {(quantizedSource == null ? "Q8" : "three original IQ block layouts and synthetic dense/recurrent weights")}: " +
        "real native loading, three independent prefills, four N3 single-graph decode steps and primary-departure N2; " +
        "actual completed graph/slot-head evidence, precompute load/graph refusal and safe retry, physical reclamation and complete payload ledger teardown. " +
        "Optional shared-owner and dispatch-fault coverage are explicitly indicated; disabled scenarios are not qualified. No real-model semantics, OS hard cap, MTP or TP qualification."
}, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"PASS {checks} checks; {actual.Length} full-vocabulary logits bit-exact; native {native[0].sha256}");
if (hookLibrary != IntPtr.Zero) NativeLibrary.Free(hookLibrary);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate void BatchedDispatchFault(int stage, int kind);

[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate void SlotCopyFault(int stage, int kind);
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate void SlotAllocFault(int stage);
[UnmanagedFunctionPointer(CallingConvention.Cdecl)]
delegate int SlotCompare(IntPtr handle, int sourceSlot, int destinationSlot, out long bytes, out int tensors);
