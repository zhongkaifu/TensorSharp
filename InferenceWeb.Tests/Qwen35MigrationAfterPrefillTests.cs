// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The linear->paged migration a second request triggers (TryMigrateLinearKVToPaged) when the
// first request has just finished its prompt prefill and not yet decoded. On ggml_cuda the
// prefill runs through the fused verify graph with keepDeviceState, so the live recurrent
// state stays on the device (_fvDeviceStateCurrent) and the host mirrors trail it. The
// migration read those mirrors without draining them, so the migrated request decoded from
// its pre-prompt GDN state: in parallel_4 on the BatchedPaged plan (TS_PER_SEQ_FUSED=0) the
// request that arrived a step ahead of its peers answered "PRODUCTS\n# 2025年..." for
// "PRODUCT: <n>" from its second token on.
//
// Opt-in: TS_TEST_MODEL_DIR=<dir with a Qwen3.5-family GGUF> TS_TEST_GGML_BACKEND=cuda.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Qwen35MigrationAfterPrefillTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string Qwen35Family = "qwen3.8-27b-ud-iq3_xxs|qwen3.5-9b-q8_0";
    private const int BlockSize = 16;
    private const int Steps = 8;
    private readonly ITestOutputHelper _output;

    public Qwen35MigrationAfterPrefillTests(ITestOutputHelper output) { _output = output; }

    /// <summary>Prefill a prompt on the linear cache, migrate it to paged storage before its first decode (what
    /// the executor does when a second request joins then), and decode on the batched path: each step tracks the
    /// linear cache's own decode of the same tokens.</summary>
    [ModelFact(EnvModelDir, Qwen35Family, GgmlBackend = BackendType.GgmlCuda)]
    public void MigratedRightAfterThePromptPrefill_DecodesLikeTheLinearCache()
    {
        string path = TestGates.FindSmallestGguf(Environment.GetEnvironmentVariable(EnvModelDir), Qwen35Family);
        Assert.True(path != null, "the gate admitted the model but the loader found none");
        // Qwen35Model's own paged migration (one ggml_cuda device routes dense checkpoints to the
        // shared runtime, so the family model is built directly).
        using var model = new Qwen35Model(path, BackendType.GgmlCuda);
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.True(batched.SupportsLinearKVMigration, "no linear migration on this lane");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config?.ChatTemplate,
            new List<ChatMessage> { new() { Role = "user", Content =
                "Multiply 37 by 18, write 'PRODUCT: <n>', then explain in detail how lighthouses work." } },
            model.Config?.Architecture ?? string.Empty, addGenerationPrompt: true, tools: null,
            enableThinking: false).ToArray();

        // The linear cache: prefill, then Steps greedy decodes. Their inputs drive the batched run too.
        model.ResetKVCache();
        var inputs = new List<int> { ArgMax(model.Forward(prompt)) };
        var linear = new List<float[]>();
        for (int s = 0; s < Steps; s++)
        {
            linear.Add((float[])model.Forward(new[] { inputs[s] }).Clone());
            inputs.Add(ArgMax(linear[s]));
        }

        // The same prefill, then the migration right away, then the same inputs through ForwardBatch.
        model.ResetKVCache();
        model.Forward(prompt);
        var seq = NewSequence(prompt);
        var paged = new List<float[]>();
        try
        {
            Assert.True(batched.TryMigrateLinearKVToPaged(seq, BlockSize), "migration refused");
            for (int s = 0; s < Steps; s++)
            {
                paged.Add((float[])batched.ForwardBatch(BuildContext(seq, inputs[s]))[0].Clone());
                seq.AdvanceComputedTokens(1);
            }
        }
        finally
        {
            batched.OnSequenceReleased(seq.RequestId);
            model.ResetKVCache();
        }

        int firstDivergent = -1;
        for (int s = 0; s < Steps; s++)
        {
            double diff = MaxAbsDiff(linear[s], paged[s]);
            int l = ArgMax(linear[s]), p = ArgMax(paged[s]);
            if (l != p && firstDivergent < 0) firstDivergent = s;
            _output.WriteLine($"decode {s + 1} after migration: max|dlogit|={diff:F3} argmax linear={l} " +
                $"({Show(model, l)}) paged={p} ({Show(model, p)}) linear margin={Margin(linear[s]):F2}");
        }
        _output.WriteLine($"prompt {prompt.Length} tokens; first divergent decode: " +
            (firstDivergent < 0 ? "none" : (firstDivergent + 1).ToString()));

        // The first decode reads exactly what the migration moved, so a missed state shows there in full.
        double first = MaxAbsDiff(linear[0], paged[0]);
        Assert.True(first < 3.0, $"the migrated history decodes differently (max|dlogit| {first:F2})");
        Assert.Equal(ArgMax(linear[0]), ArgMax(paged[0]));
        // Later steps: no confident choice may flip (near-ties may, between two kernels).
        for (int s = 1; s < Steps; s++)
            if (Margin(linear[s]) > 1.0)
                Assert.Equal(ArgMax(linear[s]), ArgMax(paged[s]));

        // Timing only (no assertion): what the drain adds to a migration. A migration right after the prefill drains
        // the device-held recurrent state inside it; against that, the state drained beforehand, then the migration.
        for (int round = 1; round <= 4; round++)
        {
            double inside = TimeMigration(model, batched, prompt, drainFirst: false, out _);
            double after = TimeMigration(model, batched, prompt, drainFirst: true, out double drain);
            _output.WriteLine($"timing round {round}: migration with the state on the device {inside:F2} ms; " +
                $"drain {drain:F2} ms, then migration {after:F2} ms");
        }
    }

    private static SequenceState NewSequence(int[] prompt)
    {
        var seq = new SequenceState($"mig-{Guid.NewGuid():N}", prompt, Steps + 1, BlockSize, SamplingConfig.Greedy);
        for (int b = 0; b < (prompt.Length + Steps + BlockSize - 1) / BlockSize; b++)
            seq.BlockTable.AppendBlock(new TensorSharp.Runtime.Paged.KvBlock(b));
        seq.AdvanceComputedTokens(prompt.Length);
        return seq;
    }

    /// <summary>Milliseconds of a migration right after a prompt prefill; with <paramref name="drainFirst"/> the
    /// device-held recurrent state is drained first (timed into <paramref name="drainMs"/>).</summary>
    private static double TimeMigration(Qwen35Model model, IBatchedPagedModel batched, int[] prompt, bool drainFirst,
        out double drainMs)
    {
        model.ResetKVCache();
        model.Forward(prompt);
        drainMs = 0;
        if (drainFirst)
        {
            var d = System.Diagnostics.Stopwatch.StartNew();
            model.DrainDeviceRecurrentState();
            drainMs = d.Elapsed.TotalMilliseconds;
        }
        var seq = NewSequence(prompt);
        try
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            Assert.True(batched.TryMigrateLinearKVToPaged(seq, BlockSize), "migration refused");
            return sw.Elapsed.TotalMilliseconds;
        }
        finally
        {
            batched.OnSequenceReleased(seq.RequestId);
            model.ResetKVCache();
        }
    }

    private static BatchedForwardContext BuildContext(SequenceState seq, int token)
    {
        int pos = seq.NumComputedTokens;
        var ids = new int[seq.BlockTable.NumBlocks];
        for (int b = 0; b < ids.Length; b++) ids[b] = seq.BlockTable.Blocks[b].Id;
        return new BatchedForwardContext
        {
            Sequences = new List<SequenceState> { seq },
            NumScheduledTokens = new List<int> { 1 },
            QueryStartLoc = new List<int> { 0, 1 },
            Positions = new List<int> { pos },
            SlotMapping = new List<int> { ids[pos / BlockSize] * BlockSize + pos % BlockSize },
            BlockTables = new[] { ids },
            OverrideFlatTokens = new[] { token },
            MaxQueryLen = 1,
            MaxSeqLen = pos + 1,
        };
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static float Margin(float[] v)
    {
        float top = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float x in v)
        {
            if (x > top) { second = top; top = x; }
            else if (x > second) second = x;
        }
        return top - second;
    }

    private static double MaxAbsDiff(float[] a, float[] b)
    {
        double m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static string Show(ModelBase model, int token) =>
        model.Tokenizer.Decode(new List<int> { token }).Replace("\n", "\\n");
}
