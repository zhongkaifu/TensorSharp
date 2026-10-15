// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Opt-in trained-model qualification of the actual session maintenance
/// owner and complete retained-holder native export/release/reseed path. Live
/// block paging, multimodal input, MTP/TP and HTTP quality are separate checks.</summary>
[Collection("Qwen4Exp MTP integration")]
public sealed class Qwen4ExpHostKvDemotionIntegrationTests(ITestOutputHelper output)
{
    [ModelFact("TS_TEST_QWEN4EXP_MODEL", GgmlBackend = BackendType.GgmlCuda)]
    public void PressureMaintenance_ClosedRequestHolderClonesAndRebindsWithCompleteLogitParity()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_MODEL")!;
        using var env = new EnvScope(); env.ClearSpeculationVars();
        env.Set("MAX_CONTEXT", "1024"); env.Set("TS_KV_INITIAL_TOKENS", "512");
        env.Set("TS_Q4E_DISABLE_ARENA_DECODE", null);
        env.Set("TS_Q4E_RETAINED_CACHE_MB", "4096");
        MoeCpuOffloadConfig.Reset();
        var reports = new List<object>(); var phases = new List<object>();
        var requests = new List<SequenceState>();
        bool completed = false; string? failure = null;
        UnifiedMemoryModelSession? session = null;
        MemoryBudget? budget = null;
        PrefixCacheCoordinator? cache = null;
        BlockPool? pool = null;
        try
        {
            session = Assert.IsType<UnifiedMemoryModelSession>(UnifiedMemoryModelSession.Create(path, null,
                BackendType.GgmlCuda, new()
                {
                    ContextTokens = 1024, PrefillChunkTokens = 128, MaximumRunningRequests = 3,
                    PrefixCheckpointCount = 0, RetainedConversationCount = 1,
                    MaximumExpertCacheBytes = 4096L << 20,
                }));
            budget = session.Budget;
            var model = Assert.IsType<Qwen4ExpModel>(session.Model);
            Assert.True(model.SupportsUnifiedRetainedHostKvDemotion);
            Assert.True(Get<bool[]>(model, "_isRecurrent").Any(x => x));
            Assert.True(Get<int>(model, "_pleHeads") > 0);
            Assert.True(Get<int>(model, "_indexerHeads") > 0);
            Assert.False(model.HasDraftHead);
            var native = Get<GgmlCacheBudgetScope>(session, "_native");
            var host = Get<HostAllocationBudgetScope>(session, "_host");
            var config = session.ConfigureScheduler(new SchedulerConfig { EnablePrefixCaching = true });
            var admission = config.MemoryAdmission!;
            pool = new BlockPool(config.NumBlocks, config.BlockSize, 0);
            var scheduler = new ContinuousBatchScheduler(config, pool);
            cache = new(model, pool, scheduler, model.GetPrefixCacheCapabilities(), NullLogger.Instance);
            string[] texts = ["请详细介绍最终幻想7", "请详细介绍时间简史", "请详细介绍最终幻想8"];
            int[] pad = model.Tokenizer.Encode("Read this context carefully. ", addSpecial: false).ToArray();
            Assert.NotEmpty(pad);
            for (int trial = 0; trial < texts.Length; ++trial)
            {
                int[] suffix = model.Tokenizer.Encode("<|im_start|>user\n" + texts[trial]
                    + "<|im_end|>\n<|im_start|>assistant\n", addSpecial: false).ToArray();
                int[] prompt = Enumerable.Range(0, Math.Max(0, 129 - suffix.Length))
                    .Select(i => pad[i % pad.Length]).Concat(suffix).ToArray();
                var source = NewRequest("source-" + trial, prompt);
                float[] initial = Prefill(source);
                int[] prefixTeacher = new int[4];
                for (int i = 0; i < prefixTeacher.Length; ++i)
                {
                    prefixTeacher[i] = ArgMax(initial);
                    initial = Forward(source, prefixTeacher[i]);
                }
                var retention = new
                {
                    request = source.RequestId, computed = source.NumComputedTokens, total = source.NumTotalTokens,
                    model_length = model.CacheSeqLen, registered = model.HasFusedSequenceCache(source.RequestId),
                    supports_retention = model.SupportsRetainedFusedCache,
                    supports_sequence = model.SupportsPerSequenceFusedForward,
                    tree_capabilities = cache.Tree.Caps, execution_options = config.ResolveExecutionOptions(),
                    unsupported_token_graph = Get<bool>(model, "_tokenGraphUnsupported"),
                    unsupported_ffn = Get<bool>(model, "_fusedFfnUnsupported"),
                    unsupported_gdn = Get<bool>(model, "_fusedGdnUnsupported"),
                    unsupported_attention = Get<bool>(model, "_fusedAttnUnsupported"),
                    callback_error = native.CallbackError?.ToString(), owned = source.MemoryEnvelope!.OwnedCharges,
                };
                output.WriteLine("Before real source retention: " + JsonSerializer.Serialize(retention));
                Assert.True(cache.RetainFinished(source, primary: false), JsonSerializer.Serialize(retention));
                string key = Assert.Single(model.RetainedPayloadKeys);
                source.Status = SequenceStatus.FinishedStopped;
                source.MemoryEnvelope!.Dispose(); // Real child allocations remain charged.
                source.BlockTable.ReleaseAll(pool);
                cache.ReleaseRequest(source);
                var retained = RetainedHolder(key);
                var metadata = Metadata(retained);
                Assert.True(((Array)Field(retained, "IdxK")).Cast<object?>().Any(x => x != null));
                Assert.True(((Array)Field(retained, "GdnConvStateT")).Cast<object?>().Any(x => x != null));
                Assert.True(((float[])Field(retained, "PleConvState")).Length > 0);
                Assert.Equal(prompt.Length + prefixTeacher.Length, (int)Field(retained, "QsaPositionCount"));
                Assert.True((bool)Field(retained, "DeviceStateSeeded"));
                Assert.True((bool)Field(retained, "DeviceMirrored"));
                Assert.True((bool)Field(retained, "KvHostStale"));
                Assert.True(source.MemoryEnvelope.LiveChildCharges.Any(c => c.Bytes > 0));

                // An independently prefetched resident control computes the
                // same boundary, without reading or synchronizing the source.
                var control = NewRequest("control-" + trial, prompt);
                float[] reference = Prefill(control);
                foreach (int token in prefixTeacher) reference = Forward(control, token);
                Assert.Equal(initial, reference);
                int[] continuation = new int[8]; var expected = new float[continuation.Length][];
                for (int i = 0; i < continuation.Length; ++i)
                {
                    continuation[i] = ArgMax(reference);
                    expected[i] = reference = Forward(control, continuation[i]);
                }
                Release(control);
                Assert.Equal(1, model.UnifiedRetainedDeviceHolderCount);
                Assert.True((bool)Field(retained, "KvHostStale"));
                var before = Snapshot("before-demotion");
                long countBefore = model.UnifiedRetainedHostKvDemotions;
                long scratch = model.GetUnifiedRetainedHostKvDemotionScratchBytes();
                var constraints = (IReadOnlyList<MemoryCharge>)Invoke(session, "PhysicalConstraints")!;
                var watch = Stopwatch.StartNew();
                // Invoke the production pressure-maintenance boundary. It
                // admits a separate host scratch owner, enters both real
                // allocation scopes, transfers/fences all native state, then
                // closes only its unused credit.
                Assert.True((bool)Invoke(session, "TryDemoteRetainedHostKv", constraints)!);
                double demotionSeconds = watch.Elapsed.TotalSeconds;
                var after = Snapshot("after-demotion");
                cache.Drain();
                Assert.Equal(0, cache.Tree.Cached.DeviceKv);
                Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(cache.Tree, new(Quiescent: true)));
                Assert.Equal(countBefore + 1, model.UnifiedRetainedHostKvDemotions);
                Assert.Equal(0, model.UnifiedRetainedDeviceHolderCount);
                Assert.Equal(metadata, Metadata(retained));
                Assert.False((bool)Field(retained, "DeviceStateSeeded"));
                Assert.False((bool)Field(retained, "DeviceMirrored"));
                Assert.False((bool)Field(retained, "KvHostStale"));
                Assert.True(model.CanReuseRetainedPrefix(key, prompt.Length + prefixTeacher.Length,
                    prompt.Length + prefixTeacher.Length));
                long freedDevice = before.DeviceCommitted - after.DeviceCommitted;
                Assert.True(freedDevice > 0, "Native device children must physically release their ledger ownership.");
                Assert.True(source.MemoryEnvelope.OwnedCharges.Single(c => c.Pool == UnifiedMemoryModelSession.HostPool).Bytes > 0);

                int[] fullPrefix = prompt.Concat(prefixTeacher).ToArray();
                foreach (bool clone in new[] { true, false })
                {
                    var target = NewRequest((clone ? "clone-" : "donation-") + trial, fullPrefix);
                    Admit(target);
                    RadixNode? donatedNode = null;
                    LockReceipt donation = default;
                    if (!clone)
                    {
                        Assert.True(cache.Tree.TryGetNodeByKey(key, out donatedNode));
                        donation = cache.Tree.AcquirePath(donatedNode);
                    }
                    try
                    {
                        using (admission.EnterSerialExecution!(target))
                            Assert.True(clone ? model.TryCloneRetainedCache(key, target.RequestId)
                                : model.TryRebindRetainedCache(key, target.RequestId));
                        // This direct transfer qualifies model clone/donation;
                        // it does not use coordinator plan matching. Follow its
                        // real tree lock/detach/unlock contract so the final
                        // unlock collects the consumed payload-less leaf.
                        if (donatedNode != null)
                            cache.Tree.DetachEndState(donatedNode, ReleaseReason.Rollback, enqueue: false);
                    }
                    finally { cache.Tree.Release(ref donation); cache.Drain(); }
                    using (admission.EnterSerialExecution!(target)) Assert.False(model.BindSequenceCache(target.RequestId));
                    target.AdvanceComputedTokens(fullPrefix.Length); // Publish only after actual adoption.
                    admission.AfterPrefixMaterialization!(target);
                    for (int step = 0; step < continuation.Length; ++step)
                    {
                        float[] actual = Forward(target, continuation[step]);
                        Assert.Equal(model.Config.VocabSize, actual.Length);
                        double maximum = 0;
                        for (int token = 0; token < actual.Length; ++token)
                        {
                            Assert.True(float.IsFinite(actual[token]) && float.IsFinite(expected[step][token]));
                            maximum = Math.Max(maximum, Math.Abs((double)actual[token] - expected[step][token]));
                        }
                        reports.Add(new { prompt = texts[trial], clone, step, vocabulary = actual.Length,
                            maximum_absolute_logit_difference = maximum });
                        Assert.Equal(0, maximum);
                    }
                    Release(target);
                }
                phases.Add(new { prompt = texts[trial], scratch_bytes = scratch, demotion_seconds = demotionSeconds,
                    freed_instrumented_device_bytes = freedDevice,
                    observed_device_available_delta = after.Hardware.DeviceAvailable - before.Hardware.DeviceAvailable,
                    closed_source_owner_after_reseed = source.MemoryEnvelope.OwnedCharges,
                    before, after });
                Assert.Equal(0, model.RetainedCacheCount);
                Assert.True(cache.Tree.Cached.IsZero);
                Assert.Empty(PrefixTreeInvariantChecker.CheckOnce(cache.Tree, new(Quiescent: true)));
                Assert.Null(native.CallbackError);
                output.WriteLine($"Host retention demotion: {texts[trial]}, {freedDevice / 1048576.0:F1} MiB actual device children released, "
                    + $"clone and donation each passed 8 complete-vocabulary steps.");
            }
            cache.Reset(); cache.Detach(); cache = null;
            pool.Storage.Dispose(); pool = null;
            session.Dispose(); session = null;
            Assert.All(budget.Snapshot(), pool => { Assert.Equal(0, pool.Committed); Assert.Equal(0, pool.Reserved); });
            completed = true;

            SequenceState NewRequest(string id, int[] tokens)
            {
                var request = new SequenceState(id, tokens, 20000, 256, SamplingConfig.Greedy, cacheScope: "demotion-" + id);
                requests.Add(request); return request;
            }
            void Admit(SequenceState request)
            {
                var limits = admission.BeforeAdmissionWithRunning!(request, []);
                request.MemoryEnvelope = budget.TryReserve(admission.EstimateMaterializationPeak!(request), limits)
                    ?? throw new MemoryPressureException("The real materialization phase could not reserve its owner.");
                request.BlockTable.EnsureBlockCapacity(4);
                var blocks = pool.AllocateNew(4);
                Assert.NotNull(blocks);
                foreach (var block in blocks!) request.BlockTable.AppendBlock(block);
                request.Status = SequenceStatus.Running;
            }
            float[] Prefill(SequenceState request)
            {
                Admit(request);
                using (admission.EnterSerialExecution!(request)) Assert.True(model.BindSequenceCache(request.RequestId));
                admission.AfterPrefixMaterialization!(request);
                float[] logits = [];
                for (int offset = 0; offset < request.PromptTokens.Count; offset += 128)
                {
                    int count = Math.Min(128, request.PromptTokens.Count - offset);
                    using (admission.EnterSerialExecution!(request))
                    {
                        model.BindSequenceCache(request.RequestId);
                        Assert.Equal(request.NumComputedTokens, model.CacheSeqLen);
                        logits = (float[])model.Forward(request.PromptTokens.GetRange(offset, count).ToArray()).Clone();
                        Assert.Equal(request.NumComputedTokens + count, model.CacheSeqLen);
                    }
                    request.AdvanceComputedTokens(count);
                }
                admission.AfterPrefill!(request);
                return logits;
            }
            float[] Forward(SequenceState request, int token)
            {
                float[] logits;
                using (admission.EnterSerialExecution!(request))
                {
                    // Session prefill maintenance restores the primary cache.
                    // BatchExecutor binds the owner at every step; the direct
                    // qualification path must follow that same contract.
                    Assert.False(model.BindSequenceCache(request.RequestId));
                    Assert.Equal(request.NumComputedTokens, model.CacheSeqLen);
                    logits = (float[])model.Forward([token]).Clone();
                    Assert.Equal(request.NumComputedTokens + 1, model.CacheSeqLen);
                }
                request.AppendOutputToken(token);
                request.AdvanceComputedTokens(1);
                return logits;
            }
            void Release(SequenceState request)
            {
                model.OnSequenceReleased(request.RequestId);
                request.BlockTable.ReleaseAll(pool);
                request.Status = SequenceStatus.FinishedStopped;
                request.MemoryEnvelope!.Dispose();
            }
            object RetainedHolder(string key) => ((IDictionary)Get<object>(model, "_retainedFusedHolders"))[key]!;
            string Metadata(object holder) => JsonSerializer.Serialize(new
            {
                capacity = Field(holder, "KvCapacity"), length = Field(holder, "CacheSeqLen"),
                qsa_count = Field(holder, "QsaPositionCount"), qsa_positions = Field(holder, "QsaPositions"),
                gdn_rings = Field(holder, "GdnConvWriteIdx"), ple_history = Field(holder, "PleHistory"),
                ple_next = Field(holder, "PleNextPos"), rotary_gap = Field(holder, "MropeCacheGap"),
            });
            (long DeviceCommitted, InferenceHardwareMemory Hardware, object Usage) Snapshot(string phase)
            {
                var physical = InferenceHardwareMemory.CaptureCuda();
                var ledger = budget.Snapshot(); var hostUsage = host.Usage;
                return (ledger.Single(c => c.Pool == UnifiedMemoryModelSession.DevicePool).Committed, physical,
                    new { phase, ledger, native = native.AllocationUsage,
                        prefix_resources = cache.Tree.Cached,
                        host = new { hostUsage.Bytes, hostUsage.PeakBytes, hostUsage.Allocations },
                        process_working_set = Process.GetCurrentProcess().WorkingSet64 });
            }
        }
        catch (Exception error) { failure = error.ToString(); throw; }
        finally
        {
            try
            {
                // A failure can stop before the normal release fence. Closing
                // unused roots still preserves every allocated child for the
                // model's authoritative physical cleanup below.
                foreach (var request in requests) request.MemoryEnvelope?.Dispose();
                cache?.Reset(); cache?.Detach();
                if (pool != null) foreach (var request in requests) request.BlockTable.ReleaseAll(pool);
                pool?.Storage.Dispose();
                session?.Dispose();
            }
            finally
            {
                MoeCpuOffloadConfig.Reset();
                string? directory = Environment.GetEnvironmentVariable("TS_TEST_QWEN4EXP_REPORT_DIR");
                if (!string.IsNullOrWhiteSpace(directory))
                {
                    Directory.CreateDirectory(directory);
                    File.WriteAllText(Path.Combine(directory, "host-kv-demotion-parity.json"), JsonSerializer.Serialize(
                        new { completed, failure, model = path, reports, phases, final_budget = budget?.Snapshot(),
                            scope = "Actual session maintenance under native and anonymous-host allocation ownership; complete inactive holder demotion, clone and donation continuation. No live block paging, N3 execution, media, MTP/TP or HTTP quality qualification." },
                        new JsonSerializerOptions { WriteIndented = true, IncludeFields = true }));
                }
            }
        }
    }

    private static int ArgMax(float[] values)
    {
        Assert.NotEmpty(values); int best = 0;
        for (int i = 0; i < values.Length; ++i)
        { Assert.True(float.IsFinite(values[i])); if (values[i] > values[best]) best = i; }
        return best;
    }
    private static T Get<T>(object value, string name) => (T)value.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(value)!;
    private static object Field(object value, string name) => value.GetType()
        .GetField(name, BindingFlags.Instance | BindingFlags.Public)!.GetValue(value)!;
    private static object? Invoke(object value, string name, params object[] parameters)
    {
        try { return value.GetType().GetMethod(name, BindingFlags.Instance | BindingFlags.NonPublic)!.Invoke(value, parameters); }
        catch (TargetInvocationException error) when (error.InnerException is { } inner)
        { System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(inner).Throw(); throw; }
    }
}
