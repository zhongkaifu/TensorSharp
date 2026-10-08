// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class WeightStreamingFailureBoundaryTests : IDisposable
{
    private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-stream-failure-{Guid.NewGuid():N}.gguf");
    public WeightStreamingFailureBoundaryTests() => BackendFailureWarmupTests.WriteProbeGguf(_path);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PartialForwardMustResetBeforeRetry(bool refill)
    {
        using var model = new PartialForwardModel(_path, streaming: true);
        Assert.Throws<MemoryPressureException>(() => Run());
        Assert.Equal(1, model.State);
        model.FailForward = false;
        Assert.Contains("ResetKVCache", Assert.Throws<InvalidOperationException>(() => Run()).Message);
        Assert.Equal(1, model.State); // Retrying never enters another model operation.
        model.FailReset = true;
        Assert.Throws<InvalidOperationException>(model.ResetKVCache);
        Assert.Throws<InvalidOperationException>(() => Run());
        model.FailReset = false;
        model.ResetKVCache();
        Assert.Equal(0, model.State);
        Assert.Equal(1f, Run()[0]);
        float[] Run() => refill ? model.ForwardRefill(new[] { 1, 2 }) : model.Forward(new[] { 1 });
    }

    [Fact]
    public void ExistingNonStreamingErrorContractIsUnchanged()
    {
        using var model = new PartialForwardModel(_path, streaming: false);
        Assert.Throws<MemoryPressureException>(() => model.Forward(new[] { 1 }));
        model.FailForward = false;
        Assert.Equal(2f, model.Forward(new[] { 1 })[0]);
    }

    [Fact]
    public void FailureWhileResettingHealthyStateAlsoRequiresACompleteReset()
    {
        using var model = new PartialForwardModel(_path, streaming: true) { FailForward = false };
        model.Forward(new[] { 1 });
        model.FailReset = true;
        Assert.Throws<InvalidOperationException>(model.ResetKVCache);
        Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1 }));
        Assert.Equal(1, model.State);
        model.FailReset = false;
        model.ResetKVCache();
        Assert.Equal(1f, model.Forward(new[] { 1 })[0]);
    }

    [Fact]
    public void DiagnosticFailureAfterForwardDoesNotPermitDuplicateTokenRetry()
    {
        using var model = new PartialForwardModel(_path, streaming: true) { FailForward = false };
        string previous = Environment.GetEnvironmentVariable("TS_DUMP_LOGITS");
        try
        {
            Environment.SetEnvironmentVariable("TS_DUMP_LOGITS",
                Path.Combine(Path.GetTempPath(), $"ts-missing-{Guid.NewGuid():N}", "logits.bin"));
            Assert.Throws<DirectoryNotFoundException>(() => model.Forward(new[] { 1 }));
            Assert.Throws<InvalidOperationException>(() => model.Forward(new[] { 1 }));
            Assert.Equal(1, model.State);
            model.ResetKVCache();
            Assert.Equal(1f, model.Forward(new[] { 1 })[0]);
        }
        finally { Environment.SetEnvironmentVariable("TS_DUMP_LOGITS", previous); }
    }

    public void Dispose() => File.Delete(_path);

    private sealed class PartialForwardModel : ModelBase
    {
        public int State;
        public bool FailForward = true, FailReset;
        public PartialForwardModel(string path, bool streaming) : base(path, BackendType.Cpu,
            weightStreaming: streaming ? new WeightStreamingOptions(new MemoryBudget(new[] {
                new MemoryCharge("ram", 1024), new MemoryCharge("gpu", 1024) }), "ram", new[] { "gpu" }) : null) { }
        protected override float[] ForwardCore(int[] tokens)
        {
            State++; // Simulates a layer advancing recurrent state before pressure.
            if (FailForward) throw new MemoryPressureException("Another owner consumed the later layer's workspace.");
            return new[] { (float)State };
        }
        protected override void ResetKVCacheCore()
        {
            if (FailReset) throw new InvalidOperationException("Reset did not complete.");
            State = 0;
        }
    }
}
