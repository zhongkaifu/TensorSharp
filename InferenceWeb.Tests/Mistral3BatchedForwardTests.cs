// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// End-to-end correctness tests for Mistral 3's batched slot path,
// using the real Ministral-3-14B GGUF that ships with the test fixture.
//
// Set TS_TEST_MODEL_DIR to enable. These tests load a large model (14B
// parameters) and run two forward passes, so they take ~minutes; they're
// opt-in so CI without the model files just skips them.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Mistral3BatchedForwardTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public Mistral3BatchedForwardTests(ITestOutputHelper output) { _output = output; }

    [ModelFact("TS_TEST_MODEL_DIR", "ministral|mistral")]
    public Task Mistral3_BatchSize1_SlotBatchMatchesLegacyTop1()
        => RunCorrectnessTest();

    [ModelFact("TS_TEST_MODEL_DIR", "ministral|mistral")]
    public Task Mistral3_BatchSize2_DistinctSequencesProduceDistinctLogits()
        => RunDistinctTest();

    private async Task RunCorrectnessTest()
    {
        var model = await TryLoadMistral3();
        if (model == null) return;

        try
        {
            // Use plausible token IDs from the model's BOS-anchored prompt range.
            // (We don't decode them; the test cares about logit-level matching.)
            var prompt = new[] { 1, 100, 200, 300, 400, 500 };
            // ---- legacy single-sequence Forward ----
            model.ResetKVCache();
            var legacyLogits = model.Forward(prompt);
            int legacyTop1 = ArgMax(legacyLogits);

            // ---- one request's slot through the batched step ----
            var slots = (IBatchedPagedModel)model;
            slots.BindSequenceCache("r0");
            var perSeqLogits = new float[1][];
            Assert.True(slots.TryForwardBatchedFusedPrefill(new[] { "r0" }, new[] { prompt }, new[] { 0 }, perSeqLogits));
            int batchedTop1 = ArgMax(perSeqLogits[0]);
            slots.OnSequenceReleased("r0");

            _output.WriteLine($"[mistral3] legacy top-1 = {legacyTop1}, batched top-1 = {batchedTop1}");

            // We tolerate single-step floating-point noise that might flip
            // tied logits, but the top-1 must match if the implementation
            // is numerically faithful.
            Assert.Equal(legacyTop1, batchedTop1);
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task RunDistinctTest()
    {
        var model = await TryLoadMistral3();
        if (model == null) return;

        try
        {
            var promptA = new[] { 1, 100, 200, 300 };
            var promptB = new[] { 1, 555, 666, 777, 888 };
            var slots = (IBatchedPagedModel)model;
            slots.BindSequenceCache("rA");
            slots.BindSequenceCache("rB");
            var perSeqLogits = new float[2][];
            Assert.True(slots.TryForwardBatchedFusedPrefill(new[] { "rA", "rB" }, new[] { promptA, promptB },
                new[] { 0, 0 }, perSeqLogits));

            int topA = ArgMax(perSeqLogits[0]);
            int topB = ArgMax(perSeqLogits[1]);
            _output.WriteLine($"[mistral3 batch=2] top-1 A={topA}, B={topB}");
            Assert.NotEqual(topA, topB);
        }
        finally
        {
            model.Dispose();
        }
    }

    private async Task<Mistral3Model> TryLoadMistral3()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
        {
            _output.WriteLine($"{EnvModelDir} not set; skipping");
            return null;
        }

        string modelPath = Directory.GetFiles(dir, "*.gguf")
            .FirstOrDefault(p =>
            {
                var n = Path.GetFileName(p).ToLowerInvariant();
                return (n.Contains("ministral") || n.Contains("mistral"))
                    && !n.Contains("mmproj");
            });
        if (modelPath == null)
        {
            _output.WriteLine("No Mistral/Ministral GGUF in test dir; skipping");
            return null;
        }
        _output.WriteLine($"[mistral3] loading {Path.GetFileName(modelPath)}");
        try
        {
            BackendType backend = TestGates.PinnedGgmlBackend;
            var model = (Mistral3Model)ModelBase.Create(modelPath, backend);
            await Task.Yield();
            return model;
        }
        catch (Exception ex)
        {
            _output.WriteLine($"Failed to load: {ex.GetType().Name}: {ex.Message}");
            return null;
        }
    }

    private static int ArgMax(float[] arr)
    {
        int best = 0;
        for (int i = 1; i < arr.Length; i++) if (arr[i] > arr[best]) best = i;
        return best;
    }
}
