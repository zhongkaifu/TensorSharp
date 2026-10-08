// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory.Planning;

/// <summary>Pure admission/placement advice. Prefer a legal resident graph, including
/// a smaller prefill chunk, before host caching or SSD streaming. This is not a global
/// throughput optimizer. No probing, allocation, capacity mutation, eviction or lease
/// revocation occurs here. Refresh observations at admission, apply capacities safely,
/// and atomically reserve AdditionalCharges; a plan is never permission to allocate.
/// Byte estimates cover the supplied components, not unhooked process RSS or total VRAM.</summary>
public static class InferenceMemoryPlanner
{
    public static InferenceMemoryPlan Plan(InferenceMemoryPlanningInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        Validate(input);
        var capacities = input.Pools.Select(Capacity).ToArray();
        var rejections = new List<InferenceCandidateRejection>();
        foreach (var candidate in input.Candidates.OrderBy(c => c.Placement)
                     .ThenByDescending(c => c.PreservesExecutionGraph)
                     .ThenByDescending(c => c.PrefillChunkTokens))
        {
            if (candidate.CapabilityRefusal != null)
            {
                rejections.Add(new(candidate.Name, InferenceMemoryRejectionKind.Unsupported,
                    candidate.CapabilityRefusal));
                continue;
            }
            try
            {
                var components = Components(input, candidate);
                var peaks = Peaks(input, candidate, components);
                int previous = rejections.Count;
                foreach (var peak in peaks)
                {
                    if (peak.Peak == 0) continue;
                    var pool = capacities.Single(p => p.Pool == peak.Pool);
                    if (pool.OverTargetBytes > 0)
                        rejections.Add(new(candidate.Name, InferenceMemoryRejectionKind.ProtectedOwners,
                            $"Pool '{pool.Pool}' is {pool.OverTargetBytes} bytes below existing owners; " +
                            "stop new admission until owners release or capacity recovers.",
                            pool.Pool, peak.Additional, pool.AdditionalAvailable));
                    else if (peak.Additional > pool.AdditionalAvailable)
                        rejections.Add(new(candidate.Name, InferenceMemoryRejectionKind.Capacity,
                            $"Pool '{pool.Pool}' requires {peak.Additional} additional bytes for the " +
                            $"selected weights, KV/state, workspace and transfers; only {pool.AdditionalAvailable} remain.",
                            pool.Pool, peak.Additional, pool.AdditionalAvailable));
                }
                if (rejections.Count == previous)
                    return new InferenceMemoryPlan
                    {
                        SelectedCandidate = candidate, Capacities = Array.AsReadOnly(capacities),
                        Components = components.AsReadOnly(), PoolPeaks = peaks.AsReadOnly(),
                        AdditionalCharges = Array.AsReadOnly(peaks.Where(p => p.Additional > 0)
                            .Select(p => new MemoryCharge(p.Pool, p.Additional)).ToArray()),
                        Rejections = rejections.AsReadOnly()
                    };
            }
            catch (OverflowException)
            {
                rejections.Add(new(candidate.Name, InferenceMemoryRejectionKind.SizeOverflow,
                    "The declared allocation peak exceeds Int64 capacity; no truncated estimate is admitted."));
            }
        }
        return new InferenceMemoryPlan
        { Capacities = Array.AsReadOnly(capacities), Rejections = rejections.AsReadOnly() };
    }

    private static InferencePoolCapacity Capacity(InferenceMemoryPool pool)
    {
        var a = pool.Accounting;
        // decimal avoids overflow when a total-size counter is near Int64.MaxValue.
        // Hardware free excludes committed; add it back once to derive an absolute
        // budget. Reserved is not physically allocated and must not be added here.
        decimal physical = Math.Min((decimal)pool.TotalBytes,
            (decimal)a.Committed + pool.AvailableBytes);
        long desired = (long)Math.Max(0, Math.Min(pool.MaximumBudgetBytes, physical - pool.HeadroomBytes));
        long owners = checked(a.Reserved + a.Committed);
        return new(pool.Pool, desired, Math.Max(desired, owners), a.Reserved, a.Committed,
            Math.Max(0, desired - owners), Math.Max(0, owners - desired));
    }

    private static List<InferenceMemoryComponent> Components(
        InferenceMemoryPlanningInput input, InferenceExecutionCandidate candidate)
    {
        var model = input.Model;
        long hostWeights = checked(model.DenseWeights.Host + model.ExpertWeights.Host);
        long deviceWeights = checked(model.DenseWeights.Device + model.ExpertWeights.Device);
        InferenceMemoryBytes weights = candidate.Placement switch
        {
            InferenceWeightPlacement.Resident => new(candidate.KeepResidentHostWeights ? hostWeights : 0, deviceWeights),
            InferenceWeightPlacement.HostCached => new(hostWeights, candidate.DeviceWeightCacheBytes),
            _ => new(candidate.HostWeightCacheBytes, candidate.DeviceWeightCacheBytes)
        };
        var components = new List<InferenceMemoryComponent>
        {
            new("weight-placement", InferenceMemoryPhase.Persistent, weights),
            new("model-persistent", InferenceMemoryPhase.Persistent, model.Persistent),
            new("execution-persistent", InferenceMemoryPhase.Persistent, candidate.Persistent),
            new("recurrent-state", InferenceMemoryPhase.Persistent,
                Multiply(model.RecurrentStatePerSequence, input.Workload.ConcurrentSequences))
        };
        foreach (var (kv, index) in model.KvCaches.Select((kv, index) => (kv, index)))
        {
            long tokens = kv.WindowTokens == 0 ? input.Workload.ContextTokens
                : Math.Min(input.Workload.ContextTokens, kv.WindowTokens);
            tokens = checked((tokens + kv.AllocationBlockTokens - 1) / kv.AllocationBlockTokens * kv.AllocationBlockTokens);
            long bytes = checked(checked(checked(tokens * kv.BytesPerToken) * kv.LayerCount) * input.Workload.ConcurrentSequences);
            components.Add(new($"kv-{index}", InferenceMemoryPhase.Persistent,
                kv.Tier == MemoryTier.Host ? new(Host: bytes) : new(Device: bytes)));
        }
        components.Add(new("transfer-buffers", InferenceMemoryPhase.Persistent,
            Multiply(candidate.TransferBuffer, candidate.TransferBufferCount)));
        components.Add(new("loading-workspace", InferenceMemoryPhase.Loading, candidate.LoadingWorkspace));
        components.Add(new("prefill-workspace", InferenceMemoryPhase.Prefill, candidate.PrefillWorkspace));
        components.Add(new("decode-workspace", InferenceMemoryPhase.Decode, candidate.DecodeWorkspace));
        return components;
    }

    private static InferenceMemoryBytes Multiply(InferenceMemoryBytes bytes, int count)
        => new(checked(bytes.Host * count), checked(bytes.Device * count), checked(bytes.Ssd * count));

    private static List<InferencePoolPeak> Peaks(InferenceMemoryPlanningInput input,
        InferenceExecutionCandidate candidate, List<InferenceMemoryComponent> components)
    {
        var result = new List<InferencePoolPeak>();
        foreach (var pool in input.Pools)
        {
            long Amount(InferenceMemoryBytes bytes)
            {
                long n = input.HostPools.Contains(pool.Pool) ? bytes.Host : 0;
                if (input.DevicePools.Contains(pool.Pool)) n = checked(n + bytes.Device);
                if (input.SsdPools.Contains(pool.Pool)) n = checked(n + bytes.Ssd);
                return n;
            }
            long persistent = 0, loading = 0, prefill = 0, decode = 0;
            foreach (var component in components)
            {
                long bytes = Amount(component.Bytes);
                switch (component.Phase)
                {
                    case InferenceMemoryPhase.Persistent: persistent = checked(persistent + bytes); break;
                    case InferenceMemoryPhase.Loading: loading = checked(loading + bytes); break;
                    case InferenceMemoryPhase.Prefill: prefill = checked(prefill + bytes); break;
                    case InferenceMemoryPhase.Decode: decode = checked(decode + bytes); break;
                }
            }
            loading = checked(loading + persistent);
            prefill = checked(prefill + persistent);
            decode = checked(decode + persistent);
            long peak = Math.Max(loading, Math.Max(prefill, decode));
            long reusable = candidate.ReusableCommittedCharges.SingleOrDefault(c => c.Pool == pool.Pool).Bytes;
            reusable = Math.Min(reusable, peak);
            result.Add(new(pool.Pool, loading, prefill, decode, peak, reusable, peak - reusable));
        }
        // A missing SSD mapping cannot silently erase a requested spill allocation.
        if (input.SsdPools.Count == 0 && components.Any(c => c.Bytes.Ssd != 0))
            throw new ArgumentException("An SSD allocation requires at least one SSD pool.", nameof(input));
        return result;
    }

    private static void Validate(InferenceMemoryPlanningInput input)
    {
        ArgumentNullException.ThrowIfNull(input.Pools);
        ArgumentNullException.ThrowIfNull(input.Model);
        ArgumentNullException.ThrowIfNull(input.Workload);
        ArgumentNullException.ThrowIfNull(input.Candidates);
        var pools = new Dictionary<string, InferenceMemoryPool>(StringComparer.Ordinal);
        foreach (var pool in input.Pools)
        {
            ArgumentNullException.ThrowIfNull(pool);
            ArgumentException.ThrowIfNullOrWhiteSpace(pool.Pool);
            var a = pool.Accounting;
            if (!pools.TryAdd(pool.Pool, pool) || a.Pool != pool.Pool)
                throw new ArgumentException("Pool names must be unique and match their accounting snapshots.", nameof(input));
            if (pool.TotalBytes < 0 || pool.AvailableBytes < 0 || pool.AvailableBytes > pool.TotalBytes
                || pool.HeadroomBytes < 0 || pool.MaximumBudgetBytes < 0 || a.Capacity < 0
                || a.Reserved < 0 || a.Committed < 0 || a.Committed > a.Capacity
                || a.Reserved > a.Capacity - a.Committed)
                throw new ArgumentOutOfRangeException(nameof(input), "Invalid capacity or accounting observation.");
        }
        if (pools.Count == 0) throw new ArgumentException("At least one pool is required.", nameof(input));
        void Mapping(IReadOnlyList<string> names, bool required)
        {
            ArgumentNullException.ThrowIfNull(names);
            if ((required && names.Count == 0) || names.Distinct(StringComparer.Ordinal).Count() != names.Count
                || names.Any(name => string.IsNullOrWhiteSpace(name) || !pools.ContainsKey(name)))
                throw new ArgumentException("Pool mappings must be distinct names of observed pools.", nameof(input));
        }
        Mapping(input.HostPools, true); Mapping(input.DevicePools, true); Mapping(input.SsdPools, false);
        static void Bytes(InferenceMemoryBytes bytes)
        {
            if (bytes.Host < 0 || bytes.Device < 0 || bytes.Ssd < 0)
                throw new ArgumentOutOfRangeException(nameof(input), "Allocation sizes cannot be negative.");
        }
        var model = input.Model;
        if (model.DenseWeights.Host < 0 || model.DenseWeights.Device < 0
            || model.ExpertWeights.Host < 0 || model.ExpertWeights.Device < 0
            || model.ExpertCount < 0 || model.ActiveExpertsPerToken < 0
            || model.ActiveExpertsPerToken > model.ExpertCount
            || (model.ExpertCount == 0 && (model.ExpertWeights.Host != 0 || model.ExpertWeights.Device != 0))
            || (model.ExpertCount > 0 && model.ActiveExpertsPerToken == 0))
            throw new ArgumentOutOfRangeException(nameof(input), "Invalid model weight or expert geometry.");
        Bytes(model.Persistent); Bytes(model.RecurrentStatePerSequence);
        ArgumentNullException.ThrowIfNull(model.KvCaches);
        foreach (var kv in model.KvCaches)
            if (kv == null || kv.BytesPerToken < 0 || kv.LayerCount <= 0 || kv.WindowTokens < 0
                || kv.AllocationBlockTokens <= 0 || !Enum.IsDefined(kv.Tier))
                throw new ArgumentOutOfRangeException(nameof(input), "Invalid KV geometry.");
        var workload = input.Workload;
        if (workload.ContextTokens <= 0 || workload.PrefillTokens <= 0 || workload.PrefillTokens > workload.ContextTokens
            || workload.BatchSize <= 0 || workload.ConcurrentSequences < workload.BatchSize)
            throw new ArgumentOutOfRangeException(nameof(input), "Invalid context, prefill or concurrency limit.");
        var names = new HashSet<string>(StringComparer.Ordinal);
        foreach (var candidate in input.Candidates)
        {
            ArgumentNullException.ThrowIfNull(candidate);
            ArgumentException.ThrowIfNullOrWhiteSpace(candidate.Name);
            if (!names.Add(candidate.Name)) throw new ArgumentException("Candidate names must be unique.", nameof(input));
            if (!Enum.IsDefined(candidate.Placement) || candidate.PrefillChunkTokens <= 0
                || candidate.PrefillChunkTokens > workload.PrefillTokens || candidate.TransferBufferCount < 0
                || candidate.HostWeightCacheBytes < 0 || candidate.DeviceWeightCacheBytes < 0)
                throw new ArgumentOutOfRangeException(nameof(input), "Invalid execution candidate.");
            if ((candidate.Placement == InferenceWeightPlacement.Resident
                    && (candidate.HostWeightCacheBytes != 0 || candidate.DeviceWeightCacheBytes != 0))
                || (candidate.Placement == InferenceWeightPlacement.HostCached && candidate.HostWeightCacheBytes != 0)
                || (candidate.Placement != InferenceWeightPlacement.Resident && candidate.KeepResidentHostWeights))
                throw new ArgumentException("Redundant cache/copy flags would misdescribe this placement.", nameof(input));
            Bytes(candidate.Persistent); Bytes(candidate.LoadingWorkspace); Bytes(candidate.PrefillWorkspace);
            Bytes(candidate.DecodeWorkspace); Bytes(candidate.TransferBuffer);
            if (candidate.TransferBufferCount == 0 && candidate.TransferBuffer != default)
                throw new ArgumentException("A nonempty transfer buffer needs a positive count.", nameof(input));
            ArgumentNullException.ThrowIfNull(candidate.ReusableCommittedCharges);
            var reused = new HashSet<string>(StringComparer.Ordinal);
            foreach (var charge in candidate.ReusableCommittedCharges)
                if (string.IsNullOrWhiteSpace(charge.Pool) || !pools.TryGetValue(charge.Pool, out var pool)
                    || !reused.Add(charge.Pool) || charge.Bytes < 0 || charge.Bytes > pool.Accounting.Committed)
                    throw new ArgumentException("Reuse must identify distinct existing committed model allocations.", nameof(input));
        }
    }
}
