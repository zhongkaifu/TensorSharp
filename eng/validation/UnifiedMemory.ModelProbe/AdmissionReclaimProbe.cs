// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

internal static class AdmissionReclaimProbe
{
    // A synthetic admission peak isolates the wakeup/reclaim
    // contract from the (still incomplete) model-specific request estimator.
    // Retained payloads and graph allocations themselves are real native buffers.
    internal static async Task<object> Run(ModelBase model, MemoryBudget budget,
        int[] prompt, int steps, int[] expected)
    {
        // The snapshot comparisons disable holder execution. This separate test
        // needs the real native holder/prefix route, otherwise only the live
        // primary remains and there is deliberately no idle cache to reclaim.
        const string setting = "TS_SCHED_DISABLE_BATCHED";
        string? previous = Environment.GetEnvironmentVariable(setting);
        Environment.SetEnvironmentVariable(setting, "0");
        try { return await RunCore(model, budget, prompt, steps, expected); }
        finally { Environment.SetEnvironmentVariable(setting, previous); }
    }

    private static async Task<object> RunCore(ModelBase model, MemoryBudget budget,
        int[] prompt, int steps, int[] expected)
    {
        long peak = 0;
        model.ResetKVCache();
        var initial = budget.Snapshot();
        long originalCapacity = initial.Single(p => p.Pool == AdaptiveModelSession.DevicePool).Capacity;
        using var engine = new InferenceEngine(model, new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 512, MaxNumRunningSequences = 1,
            MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 16,
            DecodeQuantumTokens = 256, EnablePrefixCaching = true, StopRepetition = false,
            MemoryAdmission = new(budget, s =>
                [new(AdaptiveModelSession.DevicePool, s.RequestId.StartsWith("reclaim-seed", StringComparison.Ordinal) ? 0 : peak)])
        });
        SequenceState Request(string id) => new(id, prompt, steps, 16, SamplingConfig.Greedy,
            cacheScope: id, sharedPrefixTokens: prompt.Length / 16 * 16);
        try
        {
            var seed = Request("reclaim-seed");
            await engine.SubmitRequest(seed).Completion.WaitAsync(TimeSpan.FromSeconds(60));
            if (!seed.OutputTokens.SequenceEqual(expected)) throw new InvalidOperationException("Reclaim seed differs from isolated output.");
            // Displace the primary once: a single warm request may leave only a
            // live primary plus host-only checkpoints, with no idle GPU payload.
            var displaced = Request("reclaim-seed-displaced");
            await engine.SubmitRequest(displaced).Completion.WaitAsync(TimeSpan.FromSeconds(60));
            if (!displaced.OutputTokens.SequenceEqual(expected)) throw new InvalidOperationException("Displaced seed differs from isolated output.");
            if (displaced.PrefixCacheReusedTokens == 0) throw new InvalidOperationException("No native prefix was reused before the pressure test.");
            var before = budget.Snapshot();
            var device = before.Single(p => p.Pool == AdaptiveModelSession.DevicePool);
            // Require more than idle caches can release while model weights stay
            // live. This exercises partial reclaim followed by a capacity refresh,
            // without pretending 64 KiB covers rebuilding the native graph arena.
            long occupied = checked(device.Committed + device.Reserved);
            peak = Math.Min(occupied, (originalCapacity - occupied) / 2);
            if (peak <= 0) throw new InvalidOperationException("The pressure probe needs headroom to resume after its capacity refresh.");
            if (!budget.TrySetCapacity(device.Pool, checked(device.Committed + device.Reserved)))
                throw new InvalidOperationException("Could not constrain idle device headroom.");
            if (budget.CanReserve([new(device.Pool, peak)])) throw new InvalidOperationException("Admission was not actually blocked.");
            Console.WriteLine($"admission-reclaim: constrained device ledger to {device.Committed + device.Reserved} bytes; seed reuse={displaced.PrefixCacheReusedTokens}");
            Task released = budget.ChangeSignal;
            var request = Request("reclaim-followup");
            var watch = Stopwatch.StartNew();
            var handle = engine.SubmitRequest(request);
            await released.WaitAsync(TimeSpan.FromSeconds(30));
            // Join the step boundary: the budget's first release can occur in
            // the middle of a holder disposal. Inspect only after it completes.
            IReadOnlyList<MemoryPoolSnapshot> reclaimed;
            lock (model.GpuComputeLock)
            {
                reclaimed = budget.Snapshot();
                if (reclaimed.Single(p => p.Pool == device.Pool).Committed >= device.Committed)
                    throw new InvalidOperationException("No native budget ownership was reclaimed.");
                if (handle.Completion.IsCompleted || budget.CanReserve([new(device.Pool, peak)]))
                    throw new InvalidOperationException("The partial-reclaim request did not remain blocked.");
                if (!budget.TrySetCapacity(device.Pool, originalCapacity))
                    throw new InvalidOperationException("Could not restore execution headroom after cache reclamation.");
            }
            double reclaimMilliseconds = watch.Elapsed.TotalMilliseconds;
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(60));
            watch.Stop();
            if (!request.OutputTokens.SequenceEqual(expected)) throw new InvalidOperationException("Reclaimed prefix recomputation changed output.");
            var after = budget.Snapshot();
            if (after.Any(p => p.Reserved + p.Committed > p.Capacity)) throw new InvalidOperationException("Reclaim exceeded its ledger ceiling.");
            return new { Passed = true, SyntheticDeviceAdmissionBytes = peak, Initial = initial, Before = before,
                ConstrainedDeviceCapacity = device.Committed + device.Reserved, Reclaimed = reclaimed,
                ReclaimMilliseconds = reclaimMilliseconds, After = after,
                Milliseconds = watch.Elapsed.TotalMilliseconds, completion.PrefixCacheReusedTokens,
                Tokens = request.OutputTokens.ToArray(), Completion = completion.Status.ToString(),
                Scope = "Real native prefix/parked-buffer partial reclamation under a lowered shared device ledger, then an explicit capacity refresh to resume a synthetic large request envelope. This validates reclamation, waiting, wakeup and exact output after graph rebuilding; not a complete request estimator, physical VRAM cap or tight-budget forward performance." };
        }
        finally
        {
            // Release the constraint before joining on a failed/timeout probe;
            // engine disposal still owns all request cleanup.
            budget.TrySetCapacity(AdaptiveModelSession.DevicePool, originalCapacity);
        }
    }
}
