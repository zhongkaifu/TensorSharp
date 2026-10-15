using System.Reflection;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace QwenUnifiedPrefillProbe;

internal sealed record ScopeObservation(string RequestId, string Kind, DateTime FinishedUtc,
    int BeforeComputed, int AfterComputed, int PromptTokens, int BeforeLimit, int AfterLimit,
    int OutputTokens)
{
    public int SuccessfulPrefillRows => BeforeComputed < PromptTokens
        ? Math.Max(0, Math.Min(AfterComputed, PromptTokens) - BeforeComputed) : 0;
}

/// <summary>Validation-only transparent admission proxy. It neither supplies a
/// budget nor substitutes any production selection/estimate/cleanup operation.</summary>
internal sealed class AdmissionObservation
{
    private readonly object _gate = new();
    private readonly List<ScopeObservation> _observations = [];
    public Action<IReadOnlyList<ScopeObservation>>? AfterScope { get; set; }
    public ScopeObservation[] Snapshot() { lock (_gate) return _observations.ToArray(); }

    internal RequestMemoryAdmission Wrap(RequestMemoryAdmission original)
        => new(original.Budget, original.EstimatePeak, original.MaxQueuedRequests)
        {
            EstimateMaterializationPeak = original.EstimateMaterializationPeak,
            BeforeAdmission = original.BeforeAdmission,
            BeforeAdmissionWithRunning = original.BeforeAdmissionWithRunning,
            EnterSerialExecution = original.EnterSerialExecution == null ? null : seq =>
                Observe(original, [seq], "serial", () => original.EnterSerialExecution(seq)),
            EnterBatchedExecution = original.EnterBatchedExecution == null ? null : seqs =>
                Observe(original, seqs, "joint", () => original.EnterBatchedExecution(seqs)),
            AfterPrefill = original.AfterPrefill,
            AfterPrefixMaterialization = original.AfterPrefixMaterialization,
            PrefillTokenLimit = original.PrefillTokenLimit,
            RevalidateExecution = original.RevalidateExecution,
            AfterSequenceMemoryRelease = original.AfterSequenceMemoryRelease,
            UsePerSequenceHolders = original.UsePerSequenceHolders,
            ReestimateAfterPhysicalCheck = original.ReestimateAfterPhysicalCheck,
            ExecutionShape = original.ExecutionShape
        };

    private IDisposable Observe(RequestMemoryAdmission original, IReadOnlyList<SequenceState> sequences,
        string kind, Func<IDisposable> enter)
    {
        // All callbacks execute on the engine worker under the real compute lock.
        var before = sequences.Select(s => (Sequence: s, Computed: s.NumComputedTokens,
            Limit: original.PrefillTokenLimit?.Invoke(s) ?? int.MaxValue)).ToArray();
        IDisposable actual = enter();
        return new ObservedScope(() =>
        {
            // Leave the real owner first; no recording exception may skip its disposal.
            actual.Dispose();
            var finished = DateTime.UtcNow;
            var rows = before.Select(b => new ScopeObservation(b.Sequence.RequestId, kind, finished,
                b.Computed, b.Sequence.NumComputedTokens, b.Sequence.PromptTokens.Count, b.Limit,
                original.PrefillTokenLimit?.Invoke(b.Sequence) ?? int.MaxValue,
                b.Sequence.OutputTokens.Count)).ToArray();
            lock (_gate) _observations.AddRange(rows);
            AfterScope?.Invoke(rows);
        });
    }

    internal static SchedulerConfig WithAdmission(SchedulerConfig c, RequestMemoryAdmission admission) => new()
    {
        MaxNumBatchedTokens = c.MaxNumBatchedTokens, MaxNumRunningSequences = c.MaxNumRunningSequences,
        PrefillChunkTokenLimit = c.PrefillChunkTokenLimit, MaxPrefillChunkSize = c.MaxPrefillChunkSize,
        SoloPrefillChunkSize = c.SoloPrefillChunkSize, NumBlocks = c.NumBlocks, BlockSize = c.BlockSize,
        EnablePrefixCaching = c.EnablePrefixCaching, StopRepetition = c.StopRepetition,
        DecodeQuantumTokens = c.DecodeQuantumTokens, Speculation = c.Speculation, KvSnapshots = c.KvSnapshots,
        ExecutionOptionsOverride = c.ExecutionOptionsOverride, MemoryAdmission = admission
    };

    private sealed class ObservedScope(Action dispose) : IDisposable
    {
        private Action? _dispose = dispose;
        public void Dispose() => Interlocked.Exchange(ref _dispose, null)?.Invoke();
    }

    internal static void SelfTest()
    {
        int calls = 0, disposals = 0, checks = 0;
        void Check(bool ok, string name) { checks++; if (!ok) throw new InvalidOperationException(name); }
        var budget = new MemoryBudget([new("host", 1 << 20)]);
        var shape = new RequestMemoryExecutionShape(new KvSnapshotOptions(1 << 20, 0, "unused-self-test"), 16, 2, 128);
        var original = new RequestMemoryAdmission(budget, _ => [], 123)
        {
            EstimateMaterializationPeak = _ => [], BeforeAdmission = _ => [],
            BeforeAdmissionWithRunning = (_, _) => [],
            EnterSerialExecution = _ => { calls++; return new ObservedScope(() => disposals++); },
            EnterBatchedExecution = _ => { calls++; return new ObservedScope(() => disposals++); },
            AfterPrefill = _ => calls++, AfterPrefixMaterialization = _ => calls++,
            PrefillTokenLimit = _ => 4, RevalidateExecution = _ => calls++, AfterSequenceMemoryRelease = _ => calls++,
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true, ExecutionShape = shape
        };
        var observer = new AdmissionObservation();
        var proxy = observer.Wrap(original);
        foreach (PropertyInfo property in typeof(RequestMemoryAdmission).GetProperties(BindingFlags.Public | BindingFlags.Instance))
        {
            if (property.Name is "EnterSerialExecution" or "EnterBatchedExecution")
                Check(property.GetValue(proxy) != null, "Lost wrapped scope: " + property.Name);
            else Check(Equals(property.GetValue(original), property.GetValue(proxy)), "Lost admission property: " + property.Name);
        }
        var configuration = new SchedulerConfig { MemoryAdmission = original, MaxNumBatchedTokens = 77,
            MaxNumRunningSequences = 2, MaxPrefillChunkSize = 6, SoloPrefillChunkSize = 88, PrefillChunkTokenLimit = 99,
            NumBlocks = 100, BlockSize = 16, EnablePrefixCaching = false, StopRepetition = false, DecodeQuantumTokens = 1 };
        var wrapped = WithAdmission(configuration, proxy);
        foreach (PropertyInfo property in typeof(SchedulerConfig).GetProperties(BindingFlags.Public | BindingFlags.Instance))
            if (property.Name != "MemoryAdmission")
                Check(Equals(property.GetValue(configuration), property.GetValue(wrapped)), "Lost scheduler property: " + property.Name);
        Check(ReferenceEquals(proxy, wrapped.MemoryAdmission), "Proxy missing from scheduler");
        var seq = new SequenceState("self-test", [1, 2, 3, 4, 5, 6], 3, 16, SamplingConfig.Greedy);
        var pool = new BlockPool(1, 16, 0);
        seq.BlockTable.AppendBlock(pool.AllocateNew(1)![0]);
        using (proxy.EnterSerialExecution!(seq)) seq.AdvanceComputedTokens(4);
        using (proxy.EnterSerialExecution!(seq)) seq.AdvanceComputedTokens(2);
        using (proxy.EnterBatchedExecution!([seq])) seq.AdvanceComputedTokens(1);
        Check(calls == 3 && disposals == 3, "Real scopes not entered/disposed exactly once");
        Check(observer.Snapshot().Select(r => r.SuccessfulPrefillRows).SequenceEqual([4, 2, 0]), "Scope row attribution");
        Check(observer.Snapshot().All(r => r.BeforeLimit == 4 && r.AfterLimit == 4), "Production getter lost");
        proxy.RevalidateExecution!(seq); proxy.AfterPrefill!(seq); proxy.AfterPrefixMaterialization!(seq); proxy.AfterSequenceMemoryRelease!([]);
        Check(calls == 7, "Lifecycle callbacks not forwarded");
        var nullOriginal = new RequestMemoryAdmission(budget, _ => []);
        var nullProxy = observer.Wrap(nullOriginal);
        Check(nullProxy.EnterSerialExecution == null && nullProxy.EnterBatchedExecution == null, "Null scope semantics changed");
        observer.AfterScope = _ => throw new IOException("observer-fault");
        try { using var scope = proxy.EnterSerialExecution(seq); }
        catch (IOException) { }
        Check(disposals == 4, "Observation fault skipped real disposal");
        pool.Storage.Dispose();
        InputSpec.SelfTest();
        Console.WriteLine($"PASS admission/configuration transparency and attribution: {checks} checks; no model/native execution.");
    }
}
