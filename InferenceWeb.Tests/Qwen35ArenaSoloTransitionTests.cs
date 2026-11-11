// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A request that was decoding ALONE when a second one arrived moves from the solo
// fused decode to the arena batched decode. The solo decode keeps its recurrent
// (GatedDeltaNet) state in its own device mirrors - on Metal the delta state under
// its backing allocation's base pointer - which the arena's slot seeding does not
// look up: it seeded that request from the stale host bytes, and the request
// decoded on from a state missing its last solo step, 1.4-2.8 max |dlogit| on
// Qwen3.5-9B for the rest of the request. Fluent text, wrong numbers; it surfaced
// as a greedy flip at a 0.17 margin in the concurrent image-conversation test. The
// arena now brings such a holder's recurrent state back to the host before it
// seeds the slot.
//
// Opt in with TS_TEST_MODEL_DIR pointing at a directory holding a Qwen3.5-9B GGUF,
// on the pinned metal or cuda backend (the arena exists only there).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen35ArenaSoloTransitionTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string ModelPattern = "Qwen3.5-9B";

    /// <summary>Arena against solo decode of the same sequence: the measured difference is
    /// 2e-4 to 1.6e-3 on Metal (different kernels for one column and two); the defect
    /// this guards moved the logits by 1.4 to 2.8.</summary>
    private const double Bound = 0.1;

    private readonly ITestOutputHelper _output;
    public Qwen35ArenaSoloTransitionTests(ITestOutputHelper output) => _output = output;

    [ModelFact(EnvModelDir, ModelPattern)]
    public void ArenaStepsAfterASoloStep_MatchSoloDecode()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        if (backend is not (BackendType.GgmlMetal or BackendType.GgmlCuda))
        {
            _output.WriteLine($"{backend} has no arena batched decode; test not applicable");
            return;
        }
        string path = TestGates.FindGguf(Environment.GetEnvironmentVariable(EnvModelDir), ModelPattern)!;
        // Qwen35Model's arena, built directly (dense one-device checkpoints route to the shared runtime).
        using ModelBase model = new Qwen35Model(path, backend);
        var paged = (IBatchedPagedModel)model;
        var q35 = Assert.IsType<Qwen35Model>(model);

        int[] prompt = model.Tokenizer.Encode(
            "The lighthouse keeper counted the ships each evening, writing their names in a ledger " +
            "bound in green cloth. On the night of the storm he", addSpecial: true).ToArray();
        int[] other = model.Tokenizer.Encode(
            "List three rivers in Europe and say which of them is the longest.", addSpecial: true).ToArray();

        float[] Prefill(string id, int[] tokens)
        {
            Assert.True(paged.BindSequenceCache(id));
            float[] logits = (float[])model.Forward(tokens).Clone();
            paged.RestorePrimaryCache();
            return logits;
        }
        float[] Solo(string id, int token)
        {
            paged.BindSequenceCache(id);
            float[] logits = (float[])model.Forward(new[] { token }).Clone();
            paged.RestorePrimaryCache();
            return logits;
        }

        int first = ArgMax(Prefill("reference", prompt));
        Assert.Equal(first, ArgMax(Prefill("joined", prompt)));
        Prefill("other", other);

        // The reference decodes alone throughout; the joined copy decodes its first step alone,
        // then with the other request in the arena.
        const int Steps = 10;
        var tokens = new List<int> { first };
        var reference = new List<float[]>();
        for (int i = 0; i < Steps; i++)
        {
            reference.Add(Solo("reference", tokens[i]));
            tokens.Add(ArgMax(reference[i]));
        }

        long arenaBefore = q35.ArenaBatchedDecodeSteps;
        double worst = 0;
        float[] soloStep = Solo("joined", tokens[0]);
        Assert.Equal(0, MaxDiff(reference[0], soloStep));
        for (int i = 1; i < Steps; i++)
        {
            var outLogits = new float[2][];
            Assert.True(paged.TryForwardBatchedFusedDecode(new[] { "joined", "other" },
                new[] { tokens[i], 11 }, new[] { prompt.Length + i, other.Length + i - 1 }, outLogits),
                $"the arena declined step {i}: {paged.BatchedFusedDecodeDeclineReason}");
            double diff = MaxDiff(reference[i], outLogits[0]);
            worst = Math.Max(worst, diff);
            _output.WriteLine($"step {i}: arena after a solo step vs solo decode, max |dlogit| {diff:G4}");
        }
        Assert.Equal(Steps - 1, q35.ArenaBatchedDecodeSteps - arenaBefore);
        Assert.True(worst < Bound,
            $"the arena decoded a request that had just decoded alone {worst:G4} away from decoding it alone (bound {Bound})");

        foreach (string id in new[] { "reference", "joined", "other" })
            paged.OnSequenceReleased(id);
    }

    private static double MaxDiff(float[] a, float[] b)
    {
        double worst = 0;
        for (int v = 0; v < a.Length; v++)
            worst = Math.Max(worst, Math.Abs(a[v] - b[v]));
        return worst;
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }
}
