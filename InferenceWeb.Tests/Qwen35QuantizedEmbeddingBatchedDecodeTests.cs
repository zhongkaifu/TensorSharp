// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Validation;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]

public sealed class Qwen35QuantizedEmbeddingBatchedDecodeTests
{
    private const string ModelPattern = "qwen3.8-27b-ud-iq3_xxs";
    private const int DecodeSteps = 32;
    private const int HeldOutDecodeSteps = 8;
    private const int LongDecodeSteps = 128;
    private readonly ITestOutputHelper _output;
    public Qwen35QuantizedEmbeddingBatchedDecodeTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_TEST_MODEL_DIR", ModelPattern, GgmlBackend = BackendType.GgmlCuda)]
    public void Q2KEmbedding_BatchesTwoThreeAndFourRequests_WithSerialDistributionParity()
    {
        string path = TestGates.FindSmallestGguf(Environment.GetEnvironmentVariable("TS_TEST_MODEL_DIR"), ModelPattern);
        Assert.False(string.IsNullOrEmpty(path), "The model gate admitted the test but the exact IQ3_XXS model is missing.");
        string? oldContext = Environment.GetEnvironmentVariable("MAX_CONTEXT");
        string? oldInitial = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS");
        string? oldLengths = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS");
        string? oldSnapshots = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_SNAPSHOT_DIR");
        KvCacheDtype oldDtype = KvCacheDtypeConfig.Current;
        bool oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", "4096");
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", "128");
            Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS", null);
            Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_SNAPSHOT_DIR", null);
            KvCacheDtypeConfig.Set(KvCacheDtype.F16);
            // Qwen35Model's batched decode (the factory routes one dense ggml_cuda device to the runtime).
            using var model = new Qwen35Model(path, BackendType.GgmlCuda);
            foreach (int width in new[] { 3, 2, 4 })
            {
                // The discovered width-four drift first violated the KL budget at
                // step 28; eight decode steps were insufficient to regress it.
                var result = Qwen35DecodeProbe.Compare(model, width, steps: DecodeSteps, batchedFirst: true,
                    verifySampled: true, replaceLastRequest: width == 3);
                Assert.True(result.NumericalPassed);
                Assert.True(result.DistributionPassed);
                Assert.Equal(DecodeSteps, result.Batched.FusedSteps);
                Assert.Equal("shared-checkpoint-clones", result.InitialStateMode);
                Assert.Equal(0, result.Batched.FallbackSteps);
                Assert.Equal(DecodeSteps, result.Batched.NativeFusedSteps);
                Assert.NotNull(result.Sampled);
                Assert.Equal(DecodeSteps, result.Sampled.FusedSteps);
                Assert.Equal(DecodeSteps, result.Sampled.NativeFusedSteps);
                Assert.Equal(width == 3 ? 1 : 0, result.Batched.RequestReplacements);
                _output.WriteLine($"width={width}, fused={result.Batched.FusedSteps}, cosine={result.MinCosine:F8}, normalized RMSE={result.MaxNormalizedRmse:F6}, softmax KL={result.MaxSoftmaxKl:F8}, argmax differences={result.TopTokenDifferences}, capacities={string.Join(',', result.Batched.CacheCapacities)}");
                foreach (var flip in result.ArgmaxFlips)
                    _output.WriteLine(System.Text.Json.JsonSerializer.Serialize(flip));
            }
            // Fixed held-out text, defined before evaluating the distribution
            // budgets on it. Keeps initialization and ownership paths identical.
            string[] heldOutSources =
            {
                "A violin has four strings tuned in fifths. An orchestra follows the conductor through changes in rhythm. ",
                "Bread dough rises as yeast releases carbon dioxide. The baker folds the dough before its final rest. ",
                "A lunar eclipse occurs when Earth casts its shadow on the Moon. Observers record its duration and colour. "
            };
            var heldOut = Qwen35DecodeProbe.Compare(model, width: 3, steps: HeldOutDecodeSteps, batchedFirst: false,
                promptSources: heldOutSources);
            Assert.True(heldOut.NumericalPassed);
            Assert.True(heldOut.DistributionPassed);
            Assert.Equal(HeldOutDecodeSteps, heldOut.Batched.FusedSteps);
            Assert.Equal("shared-checkpoint-clones", heldOut.InitialStateMode);
            Assert.Equal(0, heldOut.ConfidentTopTokenDifferences);
            _output.WriteLine($"held-out width=3, cosine={heldOut.MinCosine:F8}, normalized RMSE={heldOut.MaxNormalizedRmse:F6}, softmax KL={heldOut.MaxSoftmaxKl:F8}, argmax differences={heldOut.TopTokenDifferences}");
            foreach (var flip in heldOut.ArgmaxFlips)
                _output.WriteLine(System.Text.Json.JsonSerializer.Serialize(flip));

            // The short fixtures missed attention-window geometry drift. Cross
            // several window boundaries with unequal histories in the same batch.
            string? beforeLongLengths = Environment.GetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS");
            try
            {
                Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS", "500,1010,2030");
                var longHeldOut = Qwen35DecodeProbe.Compare(model, width: 3, steps: LongDecodeSteps,
                    batchedFirst: false, promptSources: heldOutSources, greedyReference: true);
                Assert.True(longHeldOut.NumericalPassed);
                Assert.True(longHeldOut.DistributionPassed);
                Assert.True(longHeldOut.GreedyContinuationsMatch);
                Assert.Equal("shared-checkpoint-clones", longHeldOut.InitialStateMode);
                Assert.Equal(new[] { 500, 1010, 2030 }, longHeldOut.Batched.PromptLengths);
                Assert.Equal(LongDecodeSteps, longHeldOut.Batched.FusedSteps);
                Assert.Equal(LongDecodeSteps, longHeldOut.Batched.NativeFusedSteps);
                Assert.Equal(0, longHeldOut.Batched.FallbackSteps);
                _output.WriteLine($"long held-out width=3, steps={LongDecodeSteps}, cosine={longHeldOut.MinCosine:F8}, normalized RMSE={longHeldOut.MaxNormalizedRmse:F6}, softmax KL={longHeldOut.MaxSoftmaxKl:F8}, argmax differences={longHeldOut.TopTokenDifferences}, greedy match={longHeldOut.GreedyContinuationsMatch}, capacities={string.Join(',', longHeldOut.Batched.CacheCapacities)}");
            }
            finally
            {
                Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS", beforeLongLengths);
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable("MAX_CONTEXT", oldContext);
            Environment.SetEnvironmentVariable("TS_KV_INITIAL_TOKENS", oldInitial);
            Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_LENGTHS", oldLengths);
            Environment.SetEnvironmentVariable("TS_QWEN35_PROBE_SNAPSHOT_DIR", oldSnapshots);
            KvCacheDtypeConfig.RestoreForTests(oldDtype, oldExplicit);
        }
    }
}
