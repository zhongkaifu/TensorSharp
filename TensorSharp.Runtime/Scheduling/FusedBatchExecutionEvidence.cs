// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Runtime.Scheduling;

/// <summary>Successful native execution of one logical multi-request graph.
/// Host MoE seams may split it into several backend dispatches. This is not
/// a GPU launch count or a scheduler plan.</summary>
public sealed record FusedBatchExecutionEvidence(ulong NativeAcceptedExecutionId,
    int ActiveSequenceCount, int NativeNodeCount, int NativeHostSegmentCount, int NativeExecutionKind);
