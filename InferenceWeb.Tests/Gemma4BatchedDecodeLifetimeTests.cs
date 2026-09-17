using System;
using System.Linq;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

[Collection("PrefixCacheModelConformance")]
public sealed class Gemma4BatchedDecodeLifetimeTests
{
    // Run with DOTNET_TieredCompilation=0 to exercise optimized local liveness
    // immediately. The old code can remain accidentally rooted in cold Tier 0.
    [ModelFact("TS_TEST_MODEL_DIR", "gemma-4-e4b")]
    public void BatchedDecode_KeepsInputStorageAliveAcrossForcedCollection()
    {
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), "gemma-4-e4b");
        using var model = ModelBase.Create(path, TestGates.PinnedGgmlBackend);
        var gemma = Assert.IsType<Gemma4Model>(model);
        var sequences = (IBatchedPagedModel)gemma;
        Assert.True(sequences.SupportsPerSequenceFusedForward);

        const int count = 8;
        string[] ids = Enumerable.Range(0, count).Select(i => "lifetime-" + i).ToArray();
        int[] prompt = model.Tokenizer.Encode("The capital of France is", addSpecial: true).ToArray();
        var tokens = new int[count];
        var positions = Enumerable.Repeat(prompt.Length, count).ToArray();
        int boundaries = 0;
        try
        {
            for (int i = 0; i < count; i++)
            {
                Assert.True(sequences.BindSequenceCache(ids[i]));
                float[] logits = model.Forward(prompt);
                tokens[i] = ArgMax(logits);
            }

            gemma.BeforeBatchedDecodeNativeForTest = storage =>
            {
                AssertLiveAfterCollection(storage);
                boundaries++;
            };

            // Rebuild and replay graphs while the active request set shrinks,
            // including seven rows: the width of the recorded Metal failure.
            foreach (int width in new[] { 8, 8, 7, 7, 4, 4, 2, 2 })
            {
                var output = new float[width][];
                Assert.True(sequences.TryForwardBatchedFusedDecode(
                    ids.Take(width).ToArray(), tokens.Take(width).ToArray(),
                    positions.Take(width).ToArray(), output), "The lifetime regression must execute fused decode.");
                for (int i = 0; i < width; i++)
                {
                    Assert.Equal(model.Config.VocabSize, output[i].Length);
                    Assert.All(output[i], value => Assert.True(float.IsFinite(value)));
                    tokens[i] = ArgMax(output[i]);
                    positions[i]++;
                }
            }
            Assert.Equal(8, boundaries);
        }
        finally
        {
            gemma.BeforeBatchedDecodeNativeForTest = null;
            foreach (string id in ids) sequences.OnSequenceReleased(id);
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static void AssertLiveAfterCollection(WeakReference<Storage> reference)
    {
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
        Assert.True(reference.TryGetTarget(out Storage storage), "Batched hidden storage was collected before native upload.");
        Assert.NotEqual(IntPtr.Zero, storage.PtrAtElement(0));
        Assert.True(storage.IsOwnerExclusive(), "Batched input storage must still own its allocation.");
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }
}
