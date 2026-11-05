// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>Requests that clone the same cached prefix share its blocks in the scheduler's pool when
/// the model shares the clone's K/V (<see cref="PrefixCacheCapabilities.ClonesShareKv"/>): a pool sized
/// for one context then runs several chats on one long cached document at once.</summary>
[Collection(EngineEnvironmentCollection.Name)]
public class SharedCloneBlockTests
{
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ClonesOfOneCheckpointShareItsFullBlocks_OnlyWhenTheModelSharesTheirKv(bool sharesKv)
    {
        using var model = new OracleModel(new OracleTraits
        {
            Name = "shared-clone", Class = FamilyClass.N, EndState = EndStateSupport.CopyAndDonate,
            CanCaptureCopy = true, Truncation = TruncationKind.None, ClonesShareKv = sharesKv,
        }, 8);
        var pool = new BlockPool(64, 8, 0);
        var scheduler = new ContinuousBatchScheduler(new SchedulerConfig { BlockSize = 8, NumBlocks = 64 }, pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler, model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        // A 20-token shared prefix: two full blocks and a partial third.
        var seed = new SequenceState("seed", Enumerable.Range(1, 20).ToList(), 3, 8, SamplingConfig.Greedy, sharedPrefixTokens: 20);
        foreach (var block in pool.AllocateNew(3)!) seed.BlockTable.AppendBlock(block);
        var requests = new List<SequenceState>();
        try
        {
            model.Forward(seed.PromptTokens.ToArray());
            seed.SetComputedTokensForPrefixAdoption(20);
            cache.CaptureCheckpoint(seed);
            pool.Free(seed.BlockTable.Clear());
            int free = pool.NumFreeBlocks;

            SequenceState Chat(string id)
            {
                var r = new SequenceState(id, Enumerable.Range(1, 20).Append(500 + requests.Count).ToList(), 100, 8,
                    SamplingConfig.Greedy, cacheScope: id, sharedPrefixTokens: 20);
                requests.Add(r);
                return r;
            }

            var a = Chat("a");
            Assert.Equal(0, cache.ShareablePrefixBlocks(a));
            Assert.Equal(20, cache.ComputeReusablePrefix(a));
            Assert.True(cache.TryAdopt(a, 20, (_, _) => throw new InvalidOperationException("no truncation")));
            Assert.Equal(free - 3, pool.NumFreeBlocks);

            var b = Chat("b");
            Assert.Equal(sharesKv ? 2 : 0, cache.ShareablePrefixBlocks(b));
            Assert.Equal(20, cache.ComputeReusablePrefix(b));
            Assert.True(cache.TryAdopt(b, 20, (_, _) => throw new InvalidOperationException("no truncation")));
            Assert.Equal(3, b.BlockTable.NumBlocks);
            Assert.Equal(sharesKv, ReferenceEquals(a.BlockTable.Blocks[0], b.BlockTable.Blocks[0]));
            Assert.Equal(sharesKv, ReferenceEquals(a.BlockTable.Blocks[1], b.BlockTable.Blocks[1]));
            Assert.False(ReferenceEquals(a.BlockTable.Blocks[2], b.BlockTable.Blocks[2]));   // the partial block is private
            Assert.Equal(free - (sharesKv ? 4 : 6), pool.NumFreeBlocks);

            // a finishes: b keeps the shared blocks, and a later clone shares them through b.
            pool.Free(a.BlockTable.Clear());
            model.OnSequenceReleased(a.RequestId);
            Assert.Equal(sharesKv ? 1 : 1, b.BlockTable.Blocks[0].RefCount);
            var c = Chat("c");
            Assert.Equal(sharesKv ? 2 : 0, cache.ShareablePrefixBlocks(c));
            Assert.Equal(20, cache.ComputeReusablePrefix(c));
            Assert.True(cache.TryAdopt(c, 20, (_, _) => throw new InvalidOperationException("no truncation")));
            Assert.Equal(sharesKv, ReferenceEquals(b.BlockTable.Blocks[0], c.BlockTable.Blocks[0]));
            Assert.Equal(sharesKv ? 2 : 1, c.BlockTable.Blocks[0].RefCount);
        }
        finally
        {
            foreach (var r in requests)
            {
                pool.Free(r.BlockTable.Clear());
                model.OnSequenceReleased(r.RequestId);
            }
            cache.Reset();
        }
    }
}
