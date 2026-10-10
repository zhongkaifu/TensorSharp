// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Memory.Planning;

public enum InferenceWeightPlacement { Resident, HostCached, SsdStreaming }
public enum InferenceMemoryPhase { Persistent, Loading, Prefill, Decode }
public enum InferenceMemoryRejectionKind { Unsupported, Capacity, ProtectedOwners, SizeOverflow }

/// <summary>Distinct physical allocations. A device allocation may constrain both
/// RAM and a device working set through the pool mapping; that does not create two copies.</summary>
public readonly record struct InferenceMemoryBytes(long Host = 0, long Device = 0, long Ssd = 0);

/// <summary>Byte sizes of the actual host and device weight representations,
/// including quantization blocks, alignment, padding and any required fusion copies.</summary>
public readonly record struct InferenceWeightBytes(long Host, long Device);

/// <summary>A contemporaneous hardware observation plus accounting snapshot.
/// AvailableBytes is hardware free/available space, NOT MemoryPoolSnapshot.Available.
/// It already excludes committed physical allocations, but not unallocated reservations.
/// HeadroomBytes is additional space to leave free, supplied by the caller.
/// MaximumBudgetBytes is an explicit operator limit, independent of the previous capacity.</summary>
public sealed record InferenceMemoryPool(
    string Pool, long TotalBytes, long AvailableBytes, long HeadroomBytes,
    MemoryPoolSnapshot Accounting, long MaximumBudgetBytes = long.MaxValue);

/// <summary>KV includes both K and V, metadata and dtype-specific storage per token.
/// WindowTokens=0 means full context. AllocationBlockTokens models page rounding.
/// Count only physically distinct layers (shared-KV donors once).</summary>
public sealed record InferenceKvCache(
    long BytesPerToken, int LayerCount = 1, int WindowTokens = 0,
    int AllocationBlockTokens = 1, MemoryTier Tier = MemoryTier.Accelerator);

public sealed record InferenceModelMemory
{
    public InferenceWeightBytes DenseWeights { get; init; }
    public InferenceWeightBytes ExpertWeights { get; init; }
    public int ExpertCount { get; init; }
    public int ActiveExpertsPerToken { get; init; }
    /// <summary>Small parameters, graph holders and other non-weight permanent allocations.</summary>
    public InferenceMemoryBytes Persistent { get; init; }
    public InferenceMemoryBytes RecurrentStatePerSequence { get; init; }
    public IReadOnlyList<InferenceKvCache> KvCaches { get; init; } = [];
}

/// <summary>ContextTokens is the admitted maximum including generation. BatchSize
/// is the number of sequences in a prefill batch; ConcurrentSequences bounds live KV/state.</summary>
public sealed record InferenceMemoryWorkload(
    int ContextTokens, int PrefillTokens, int BatchSize, int ConcurrentSequences);

/// <summary>An adapter-qualified execution path for ONE prefill chunk size. All
/// workspace values are aggregate upper bounds for the supplied workload, not per-token
/// coefficients. Native size queries can populate them before planning. Include graph
/// arenas, outputs, full-matrix arithmetic scratch, temporary copies and request state
/// not described by Model. Missing backend/quantization/modality support must set
/// CapabilityRefusal; the planner never invents an implementation or fallback.
/// Loading, prefill and decode are exclusive phases of this execution lane; put any
/// simultaneously active work from other lanes into Persistent or each phase estimate.</summary>
public sealed record InferenceExecutionCandidate
{
    public required string Name { get; init; }
    public required InferenceWeightPlacement Placement { get; init; }
    public required int PrefillChunkTokens { get; init; }
    public string? CapabilityRefusal { get; init; }
    public bool PreservesExecutionGraph { get; init; }
    /// <summary>Resident execution can retain an additional complete host weight copy.</summary>
    public bool KeepResidentHostWeights { get; init; }
    /// <summary>Additional partial caches for offloaded paths. HostCached already
    /// includes all host weights; Resident already includes all device weights.</summary>
    public long HostWeightCacheBytes { get; init; }
    public long DeviceWeightCacheBytes { get; init; }
    public InferenceMemoryBytes Persistent { get; init; }
    public InferenceMemoryBytes LoadingWorkspace { get; init; }
    public InferenceMemoryBytes PrefillWorkspace { get; init; }
    public InferenceMemoryBytes DecodeWorkspace { get; init; }
    public InferenceMemoryBytes TransferBuffer { get; init; }
    /// <summary>Use two only if the adapter actually supports two overlapping buffers.</summary>
    public int TransferBufferCount { get; init; }
    /// <summary>Only already committed allocations belonging to this same model and
    /// reusable by THIS path. Never include another request's owner, pending frees,
    /// or allocations whose layouts change during the transition. They remain charged.</summary>
    public IReadOnlyList<MemoryCharge> ReusableCommittedCharges { get; init; } = [];
}

public sealed record InferenceMemoryPlanningInput
{
    public required IReadOnlyList<InferenceMemoryPool> Pools { get; init; }
    public required IReadOnlyList<string> HostPools { get; init; }
    public required IReadOnlyList<string> DevicePools { get; init; }
    /// <summary>Only newly created spill/cache files consume SSD quota. The existing
    /// model file is already reflected in hardware free space and is not charged again.</summary>
    public IReadOnlyList<string> SsdPools { get; init; } = [];
    public required InferenceModelMemory Model { get; init; }
    public required InferenceMemoryWorkload Workload { get; init; }
    public required IReadOnlyList<InferenceExecutionCandidate> Candidates { get; init; }
}

/// <summary>DesiredCapacity follows the observation. ProtectedCapacity never drops
/// below existing reserved+committed owners. When OverTargetBytes is nonzero, pause
/// admission and retry after physical release; keeping this floor is not new credit.</summary>
public sealed record InferencePoolCapacity(
    string Pool, long DesiredCapacity, long ProtectedCapacity,
    long ExistingReserved, long ExistingCommitted, long AdditionalAvailable, long OverTargetBytes);

public sealed record InferenceMemoryComponent(
    string Name, InferenceMemoryPhase Phase, InferenceMemoryBytes Bytes);

public sealed record InferencePoolPeak(
    string Pool, long Loading, long Prefill, long Decode, long Peak,
    long ReusedCommitted, long Additional);

public sealed record InferenceCandidateRejection(
    string Candidate, InferenceMemoryRejectionKind Kind, string Reason,
    string? Pool = null, long RequiredBytes = 0, long AvailableBytes = 0);

public sealed record InferenceMemoryPlan
{
    public InferenceExecutionCandidate? SelectedCandidate { get; init; }
    public bool Accepted => SelectedCandidate != null;
    public int SelectedChunkTokens => SelectedCandidate?.PrefillChunkTokens ?? 0;
    public IReadOnlyList<InferencePoolCapacity> Capacities { get; init; } = [];
    public IReadOnlyList<InferenceMemoryComponent> Components { get; init; } = [];
    public IReadOnlyList<InferencePoolPeak> PoolPeaks { get; init; } = [];
    public IReadOnlyList<MemoryCharge> AdditionalCharges { get; init; } = [];
    public IReadOnlyList<InferenceCandidateRejection> Rejections { get; init; } = [];
}
