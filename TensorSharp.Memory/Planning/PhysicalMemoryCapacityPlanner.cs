// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory.Hardware;

namespace TensorSharp.Memory.Planning;

/// <summary>Additional free space for uninstrumented future growth. Current
/// external/process/driver use is already absent from measured free space and is
/// not deducted again. Recent growth and observed peak rebound are automatically
/// added to FutureBurstBytes; MinimumBytes is a floor, not a percentage of weights.
/// These margins are forecasts, not a hard limit on uninstrumented allocations.</summary>
public sealed record AutomaticMemoryHeadroom
{
    public long MinimumBytes { get; init; } = 256L * 1024 * 1024;
    public long FutureBurstBytes { get; init; } = 128L * 1024 * 1024;
}

public sealed record PhysicalMemoryAttribution(
    string Constraint, string Pool, string Domain, PhysicalMemoryMetric Metric,
    long MeasuredUnavailableBytes, long ProvenIncludedCommittedBytes,
    long UnattributedUnavailableUpperBound, long UnprovenCommittedBytes,
    long RecentGrowthBytes, long PeakReboundBytes, long HeadroomBytes);

public sealed record PhysicalMemoryCapacityPlan
{
    /// <summary>The observations are usable; this does not mean a workload fits.
    /// Desired capacities can be zero or below existing owners.</summary>
    public bool Accepted => Refusals.Count == 0;
    public IReadOnlyList<string> Refusals { get; init; } = [];
    /// <summary>At most one record per ledger pool: the most restrictive current
    /// constraint wins. Distinct UMA constraints remain distinct and are never summed.</summary>
    public IReadOnlyList<InferenceMemoryPool> Pools { get; init; } = [];
    public IReadOnlyList<MemoryCharge> DesiredCapacities { get; init; } = [];
    public IReadOnlyList<PhysicalMemoryAttribution> Attribution { get; init; } = [];
}

/// <summary>Admission-boundary observation history and conservative capacity advice.
/// No allocation, eviction, OS enforcement or budget mutation. A rejected plan has
/// no applicable pools/targets. Keep one instance per observation scope and call
/// after obtaining fresh OS/allocator samples; never sample on every token. History
/// is bounded and expires. Root namespaces may hide ancestor limits: only visible
/// verified observations are represented, not a guarantee about hidden limits.</summary>
public sealed class PhysicalMemoryCapacityPlanner
{
    private readonly object _gate = new();
    private readonly Dictionary<string, List<Sample>> _history = new(StringComparer.Ordinal);
    private readonly AutomaticMemoryHeadroom _defaults;
    private readonly TimeSpan _maximumAge, _historyWindow;
    private const int MaximumSamples = 64;
    private sealed record Sample(DateTimeOffset At, long Total, long Unattributed,
        string Pool, string Domain, PhysicalMemoryMetric Metric);

    public PhysicalMemoryCapacityPlanner(AutomaticMemoryHeadroom? defaults = null,
        TimeSpan? maximumObservationAge = null, TimeSpan? historyWindow = null)
    {
        _defaults = defaults ?? new();
        ValidateHeadroom(_defaults);
        _maximumAge = maximumObservationAge ?? TimeSpan.FromSeconds(10);
        _historyWindow = historyWindow ?? TimeSpan.FromMinutes(5);
        if (_maximumAge <= TimeSpan.Zero || _historyWindow <= TimeSpan.Zero)
            throw new ArgumentOutOfRangeException(nameof(maximumObservationAge));
    }

    public PhysicalMemoryCapacityPlan Plan(IReadOnlyList<PhysicalMemoryObservation> observations,
        IReadOnlyList<MemoryPoolSnapshot> accounting, DateTimeOffset? now = null)
    {
        ArgumentNullException.ThrowIfNull(observations);
        ArgumentNullException.ThrowIfNull(accounting);
        var at = now ?? DateTimeOffset.UtcNow;
        var ledger = accounting.ToDictionary(p => p.Pool, StringComparer.Ordinal);
        var refusals = new List<string>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        if (observations.Count == 0) refusals.Add("No physical memory observations are available.");
        foreach (var observation in observations)
        {
            ArgumentNullException.ThrowIfNull(observation);
            ArgumentException.ThrowIfNullOrWhiteSpace(observation.Constraint);
            ArgumentException.ThrowIfNullOrWhiteSpace(observation.Domain);
            if (!names.Add(observation.Constraint)) throw new ArgumentException("Duplicate physical constraint.", nameof(observations));
            if (!ledger.TryGetValue(observation.Pool, out var snapshot))
                throw new ArgumentException($"Unknown ledger pool '{observation.Pool}'.", nameof(observations));
            var pool = ToPool(observation, snapshot, 0);
            // Treat a physically inconsistent otherwise well-formed sample as a
            // refresh refusal; malformed ledger/negative overlap remains a caller error.
            InferenceMemoryPlanner.GetCapacity(pool with { IncludedCommittedBytes = 0 });
            if (observation.IncludedCommittedBytes is long included)
            {
                if (included < 0) throw new ArgumentOutOfRangeException(nameof(observations), "Negative certified ledger overlap.");
                if (included > snapshot.Committed)
                    refusals.Add($"Constraint '{observation.Constraint}' has overlap newer than its ledger; recapture both.");
            }
            if (!Enum.IsDefined(observation.Metric)) throw new ArgumentOutOfRangeException(nameof(observations));
            const MemoryHardLimitCoverage knownCoverage = MemoryHardLimitCoverage.AllPhysical | MemoryHardLimitCoverage.PrivateCommit;
            if ((observation.RequiredHardLimitCoverage & ~knownCoverage) != 0
                || (observation.Enforcement != null && (observation.Enforcement.Coverage & ~knownCoverage) != 0))
                throw new ArgumentOutOfRangeException(nameof(observations), "Unknown hard-limit coverage.");
            ValidateHeadroom(observation.Headroom ?? _defaults);
            if (observation.TotalBytes == 0)
                refusals.Add($"Constraint '{observation.Constraint}' has no known positive total capacity.");
            if (observation.ObservedAt > at || at - observation.ObservedAt > _maximumAge)
                refusals.Add($"Constraint '{observation.Constraint}' has a future or stale observation.");
            if (observation.RequiredHardLimitCoverage != MemoryHardLimitCoverage.None)
            {
                var evidence = observation.Enforcement;
                if (evidence is not { Active: true } || evidence.LimitBytes <= 0
                    || observation.TotalBytes > evidence.LimitBytes
                    || string.IsNullOrWhiteSpace(evidence.Mechanism)
                    || (evidence.Coverage & observation.RequiredHardLimitCoverage) != observation.RequiredHardLimitCoverage)
                    refusals.Add($"Constraint '{observation.Constraint}' lacks active verified hard-limit coverage " +
                        $"for {observation.RequiredHardLimitCoverage}; an availability forecast or ledger is insufficient.");
            }
        }
        if (refusals.Count != 0) return new() { Refusals = refusals.AsReadOnly() };

        lock (_gate)
        {
            var candidates = new List<InferenceMemoryPool>();
            var attribution = new List<PhysicalMemoryAttribution>();
            var pending = new List<(string Constraint, Sample Sample)>();
            foreach (var observation in observations)
            {
                if (_history.TryGetValue(observation.Constraint, out var known)
                    && known.Count != 0 && observation.ObservedAt < known[^1].At)
                {
                    refusals.Add($"Constraint '{observation.Constraint}' is older than its last accepted observation.");
                    continue;
                }
                long unavailable = observation.TotalBytes - observation.AvailableBytes;
                long included = observation.IncludedCommittedBytes ?? 0;
                // A lower bound cannot exceed all unavailable bytes in its metric.
                // Samples can race: refuse an inconsistent pair rather than invent credit.
                if (included > unavailable)
                {
                    refusals.Add($"Constraint '{observation.Constraint}' has inconsistent certified overlap; recapture it with the ledger.");
                    continue;
                }
                long unattributed = unavailable - included;
                var previous = _history.GetValueOrDefault(observation.Constraint)?.Where(s =>
                    s.At >= at - _historyWindow && s.At <= observation.ObservedAt
                    && s.Total == observation.TotalBytes && s.Pool == observation.Pool
                    && s.Domain == observation.Domain && s.Metric == observation.Metric).ToArray() ?? [];
                long growth = previous.Length == 0 ? 0 : Math.Max(0, unattributed - previous[^1].Unattributed);
                long rebound = previous.Length == 0 ? 0 : Math.Max(0, previous.Max(s => s.Unattributed) - unattributed);
                var policy = observation.Headroom ?? _defaults;
                long headroom = (long)Math.Min(long.MaxValue,
                    Math.Max((decimal)policy.MinimumBytes, (decimal)policy.FutureBurstBytes + growth + rebound));
                var snapshot = ledger[observation.Pool];
                candidates.Add(ToPool(observation, snapshot, headroom));
                attribution.Add(new(observation.Constraint, observation.Pool, observation.Domain, observation.Metric,
                    unavailable, included, unattributed, snapshot.Committed - included, growth, rebound, headroom));
                pending.Add((observation.Constraint, new(observation.ObservedAt, observation.TotalBytes,
                    unattributed, observation.Pool, observation.Domain, observation.Metric)));
            }
            if (refusals.Count != 0) return new() { Refusals = refusals.AsReadOnly(), Attribution = attribution.AsReadOnly() };
            // Update history only when the entire observation set is admissible.
            foreach (var (constraint, sample) in pending)
            {
                if (!_history.TryGetValue(constraint, out var samples)) _history.Add(constraint, samples = []);
                samples.RemoveAll(s => s.At < at - _historyWindow || s.At == sample.At);
                samples.Add(sample);
                samples.Sort((a, b) => a.At.CompareTo(b.At));
                if (samples.Count > MaximumSamples) samples.RemoveRange(0, samples.Count - MaximumSamples);
            }
            foreach (var key in _history.Keys.Where(key => _history[key].All(s => s.At < at - _historyWindow)).ToArray())
                _history.Remove(key);
            var pools = candidates.GroupBy(p => p.Pool, StringComparer.Ordinal)
                .Select(group => group.MinBy(p => InferenceMemoryPlanner.GetCapacity(p).DesiredCapacity)!).ToArray();
            return new()
            {
                Pools = Array.AsReadOnly(pools), Attribution = attribution.AsReadOnly(),
                DesiredCapacities = Array.AsReadOnly(pools.Select(p => new MemoryCharge(p.Pool,
                    InferenceMemoryPlanner.GetCapacity(p).DesiredCapacity)).ToArray())
            };
        }
    }

    private static InferenceMemoryPool ToPool(PhysicalMemoryObservation observation,
        MemoryPoolSnapshot accounting, long headroom) => new(observation.Pool, observation.TotalBytes,
            observation.AvailableBytes, headroom, accounting, observation.MaximumBudgetBytes)
        { IncludedCommittedBytes = observation.IncludedCommittedBytes };

    private static void ValidateHeadroom(AutomaticMemoryHeadroom policy)
    {
        if (policy.MinimumBytes < 0 || policy.FutureBurstBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(policy));
    }
}
