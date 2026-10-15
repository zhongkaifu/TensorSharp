// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class GgmlExecutionOwnerTests
{
    [GgmlFact(BackendType.GgmlCpu)]
    public void ExecutionOwnerFailureDoesNotPublishAnUnownedNativeEnvelope()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var scope = new GgmlCacheBudgetScope(budget, [["device"]]);
        using var envelope = budget.Reserve([new("device", 100)]);
        scope.BeforeExecutionOwnerAllocation = () => throw new OutOfMemoryException("controlled execution owner failure");
        Assert.Throws<OutOfMemoryException>(() => scope.EnterExecution(envelope));
        Assert.Equal(0, scope.ActiveAllocations);
        Assert.Null(scope.CallbackError);
        Assert.False(envelope.HasChildren);
        scope.BeforeExecutionOwnerAllocation = null;
        using (scope.EnterExecution(envelope)) { }
        Assert.Equal(100, budget.Snapshot().Single().Reserved);
        scope.Dispose();
        using var reattached = new GgmlCacheBudgetScope(budget, [["device"]]);
        Assert.Equal(0, reattached.ActiveAllocations);
    }
}
