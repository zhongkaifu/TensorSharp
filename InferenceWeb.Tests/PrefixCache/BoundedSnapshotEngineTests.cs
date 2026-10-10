// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

[Collection(EngineEnvironmentCollection.Name)]
public class BoundedSnapshotEngineTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task BoundedSnapshotsReuseSpilledPagesWithoutUnbudgetedEndStates(bool pagedEndStates)
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-prefix-tiered-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var model = new OracleModel(new OracleTraits
            {
                Name = "bounded-snapshots", Class = FamilyClass.P,
                Pages = PageSupport.Both, EndState = EndStateSupport.CopyAndDonate,
                CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true,
                PrimaryResident = false, PagedEndStates = pagedEndStates,
                Truncation = TruncationKind.Any,
            }, blockSize: 8);
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 32, MaxNumBatchedTokens = 16,
                SoloPrefillChunkSize = 16, MaxPrefillChunkSize = 8,
                DecodeQuantumTokens = 1, EnablePrefixCaching = true, StopRepetition = false,
                KvSnapshots = new(192, 512 * 4096, root, 64),
            });
            var prompt = Enumerable.Range(1, 35).ToArray();
            var cold = new SequenceState("cold", prompt, 3, 8, SamplingConfig.Greedy,
                cacheScope: "conversation", sharedPrefixTokens: 8);
            await engine.SubmitRequest(cold).Completion.WaitAsync(TimeSpan.FromSeconds(10));
            var warm = new SequenceState("warm", prompt, 3, 8, SamplingConfig.Greedy,
                cacheScope: "conversation", sharedPrefixTokens: 8);
            await engine.SubmitRequest(warm).Completion.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(32, warm.PrefixCacheReusedTokens);
            Assert.Equal(cold.OutputTokens, warm.OutputTokens);
            Assert.True(engine.SnapshotResidencyStats!.Value.Spills > 0);
            Assert.Empty(model.RetainedPayloadKeys);
            Assert.Equal(0, model.PrivateHolderCount);
            Assert.Equal(0, model.PrimaryConversionCalls);
            engine.Dispose();
            Assert.All(engine.SnapshotMemoryUsage!, p => Assert.Equal(0, p.Reserved + p.Committed));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
