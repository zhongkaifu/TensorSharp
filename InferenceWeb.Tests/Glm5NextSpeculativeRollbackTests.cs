// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The KDA recurrent-state rollback that speculative decoding needs on
// GLM-5.3-Flash (glm5next), on a synthetic two-layer KDA fixture.
//
// What is proven, for the managed per-op path (`cpu`), the native executor on
// the CPU backend (`ggml_cpu`) and - when TS_TEST_GLM_CUDA=1 - the native
// executor on CUDA (`ggml_cuda`):
//
//   0. (the `ggml_cuda` rows are opt-in: TS_TEST_GLM_CUDA=1 with the default
//      cpu backend pin, because the GLM executor picks its CUDA devices itself)
//   1. the speculative loop with the weight-free n-gram drafter emits exactly
//      the plain greedy stream, with drafts proposed and windows partially
//      rejected, so the rollback actually ran (the prompt is built from the
//      model's own periodic continuation with one planted contradiction, so
//      the drafter's most recent match is wrong at least once);
//   2. a drafter that is wrong by construction at the end of every window
//      (n-gram proposals with the last token corrupted) still yields the plain
//      stream: every window is a PARTIAL acceptance, the harshest rollback case;
//   3. after such a run the trunk's recurrent state is the one a plain decode
//      would have left: the next token's logits agree over the whole vocabulary;
//   4. the raw protocol - snapshot, multi-row verify, restore, rewind - leaves
//      the state equal to a plain decode of the accepted prefix, and the verify
//      rows themselves equal the sequential decode's logits (the multi-token
//      KDA scan is the single-token update, K+1 times);
//   5. the managed and the native SpecForward agree row for row, hidden states
//      included, so the two implementations of the same verify can stand in for
//      each other as a reference.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class Glm5NextSpeculativeRollbackTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-glm5next-rollback-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public Glm5NextSpeculativeRollbackTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
    }

    public void Dispose() { try { Directory.Delete(_directory, true); } catch (IOException) { } }

    private string Fixture() => GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
        Path.Combine(_directory, "kda2.gguf"), numHeads: 4, quantizeAttentionOutput: false, numLayers: 2);

    private static EnvScope EnvironmentForFixture()
    {
        var env = new EnvScope();
        env.ClearSpeculationVars();
        env.Set("MAX_CONTEXT", "256");
        env.Set("TS_GLM_NATIVE", null);
        return env;
    }

    private static int[] SeedPrompt()
    {
        var p = new List<int>();
        for (int i = 0; i < 6; i++) { p.Add(65); p.Add(66); p.Add(67); }
        p.Add(65); p.Add(66);
        return p.ToArray();
    }

    /// <summary>
    /// A prompt the n-gram drafter is guaranteed to get wrong at least once.
    /// A random-weight trunk settles into a short periodic continuation (its
    /// attractor); the prompt quotes that period several times and then plants
    /// one contradiction: <c>… a b X a b</c>. The drafter's index keeps the MOST
    /// RECENT occurrence of a suffix, so when the trunk continues <c>a b</c>
    /// with <c>c</c> the drafter proposes <c>X</c> - a partial rejection - while
    /// the earlier tokens of that same window are accepted.
    /// </summary>
    private static int[] ContradictedPrompt(ModelBase model, int vocab, out int[] period)
    {
        int[] probe = PlainGreedy(model, SeedPrompt(), 18);
        period = null;
        for (int p = 1; p <= 6 && period == null; p++)
        {
            bool periodic = true;
            for (int i = probe.Length - 12; i < probe.Length - p && periodic; i++)
                periodic = probe[i] == probe[i + p];
            if (periodic)
            {
                period = new int[p];
                Array.Copy(probe, probe.Length - p, period, 0, p);
            }
        }
        Assert.True(period != null, "the trunk's greedy continuation is not periodic: " + string.Join(",", probe));

        int contradiction = 100;
        while (Array.IndexOf(period, contradiction) >= 0) contradiction = (contradiction + 1) % vocab;

        var prompt = new List<int>();
        for (int i = 0; i < 5; i++) prompt.AddRange(period);
        prompt.AddRange(period[..^1]);
        prompt.Add(contradiction);
        prompt.AddRange(period[..^1]);
        return prompt.ToArray();
    }

    private static int[] PlainGreedy(ModelBase model, int[] prompt, int count)
    {
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        var produced = new int[count];
        for (int i = 0; i < count; i++)
        {
            int t = Argmax(logits);
            produced[i] = t;
            if (i + 1 < count)
                logits = model.Forward(new[] { t });
        }
        return produced;
    }

    private static int Argmax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++)
            if (logits[i] > logits[best]) best = i;
        return best;
    }

    private static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double allowed = tol * (1.0 + Math.Abs(expected[i]));
            Assert.True(Math.Abs(expected[i] - actual[i]) <= allowed,
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]} (tolerance {allowed:G3})");
        }
    }

    private static ISpeculator NGram(ISpeculativeTarget target, int maxDraft)
    {
        var speculator = SpeculatorRegistry.Create(target, new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = SpeculatorRegistry.NGram,
            MaxDraftTokens = maxDraft,
            // Accept the shortest match: the point is to draft (and be
            // rejected) as often as possible, not to draft well.
            MinDraftProb = 0f,
        }, out string decline);
        Assert.True(speculator != null, decline);
        return speculator;
    }

    /// <summary>An n-gram drafter whose LAST proposed token is always wrong, so
    /// every drafted window is a partial acceptance: the accepted prefix must
    /// be kept and the rejected tail undone, on every single step.</summary>
    private sealed class PartiallyWrongDrafter : ISpeculator
    {
        private readonly ISpeculator _inner;
        private readonly int _vocab;
        public PartiallyWrongDrafter(ISpeculator inner, int vocab) { _inner = inner; _vocab = vocab; }
        public string Name => "wrong-tail(" + _inner.Name + ")";
        public int MaxDraftTokens => _inner.MaxDraftTokens;
        public float MinDraftProb { get => _inner.MinDraftProb; set => _inner.MinDraftProb = value; }
        public float DefaultMinDraftProb => _inner.DefaultMinDraftProb;
        public bool NeedsHiddenState => _inner.NeedsHiddenState;
        public bool HandlesOwnPrefill => _inner.HandlesOwnPrefill;
        public bool CanArmAfterPrefixReuse => _inner.CanArmAfterPrefixReuse;
        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int n = _inner.Propose(ctx, draftOut);
            if (n > 0)
            {
                // Something the trunk cannot argmax to from this context: a token
                // that never occurs in the corpus is as good as any, but simply
                // shifting the proposal is enough - equality is what is verified.
                draftOut[^1] = (draftOut[^1] + 1) % _vocab;
            }
            return n;
        }
        public void Commit(int[] tokens, float[] hRows, int startPos) => _inner.Commit(tokens, hRows, startPos);
        public void Reset() => _inner.Reset();
        public void Dispose() => _inner.Dispose();
    }

    private void RunAndCheck(BackendType backend, Func<ISpeculativeTarget, int, ISpeculator> drafter,
                             bool requireRollback, string label)
    {
        using var env = EnvironmentForFixture();
        string path = Fixture();
        const int maxNew = 24;

        const int probeToken = 77;
        int[] prompt;
        int[] plain;
        float[] plainNext, plainAfterProbe;
        using (var model = ModelBase.Create(path, backend))
        {
            prompt = ContradictedPrompt(model, model.Config.VocabSize, out int[] period);
            _output.WriteLine($"{backend} {label}: period=[{string.Join(",", period)}] prompt=[{string.Join(",", prompt)}]");
            plain = PlainGreedy(model, prompt, maxNew);
            plainNext = (float[])model.Forward(new[] { plain[^1] }).Clone();
            plainAfterProbe = (float[])model.Forward(new[] { probeToken }).Clone();
        }

        using var specModel = ModelBase.Create(path, backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(specModel);
        Assert.True(target.SpeculationProfitable);
        var decoder = new SpeculativeDecoder(target, drafter(target, specModel.Config.VocabSize))
        {
            // Measure the contract, not the cost governor: on a 1 MB model every
            // step costs the same and its verdict would be noise.
            AdaptiveSpeculation = false,
        };
        List<int> produced = decoder.GenerateGreedy(prompt, maxNew);

        _output.WriteLine($"{backend} {label}: drafted={decoder.TokensDrafted} accepted={decoder.TokensAccepted} " +
                          $"verify={decoder.VerifySteps} plain={decoder.PlainSteps} rollbacks={decoder.RollbackSteps}");
        _output.WriteLine("plain: " + string.Join(",", plain));
        _output.WriteLine("spec : " + string.Join(",", produced));

        Assert.Equal(maxNew, produced.Count);
        Assert.Equal(plain, produced.ToArray());
        Assert.True(decoder.TokensDrafted > 0, "the drafter never proposed a token");
        if (requireRollback)
            Assert.True(decoder.RollbackSteps > 0, "no window was partially rejected, so the rollback never ran");

        // Greedy decoding forwards every emitted token except the last, so the
        // trunk holds prompt + N - 1 - unless the final window was fully accepted
        // at the maxNew boundary, in which case the decoder legitimately leaves
        // the last emitted token forwarded too (and drops the bonus token). Either
        // way, after the rollbacks the recurrent state must be the one a plain
        // decode of exactly the held tokens leaves: the continuation must agree
        // with the plain run over the whole vocabulary, not only at the argmax.
        int held = target.CacheSeqLen - prompt.Length;
        Assert.True(held == produced.Count - 1 || held == produced.Count,
            $"the trunk holds {held} of the {produced.Count} emitted tokens");
        if (held == produced.Count - 1)
        {
            float[] specNext = specModel.Forward(new[] { produced[^1] });
            AssertClose(plainNext, specNext, 1e-3, "continuation logits");
        }
        float[] specAfterProbe = specModel.Forward(new[] { probeToken });
        AssertClose(plainAfterProbe, specAfterProbe, 1e-3, "continuation logits after a probe token");
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void NGramSpeculativeGreedy_MatchesPlainGreedy_AndRollsBack(BackendType backend)
    {
        RunAndCheck(backend, (t, _) => NGram(t, maxDraft: 4), requireRollback: true, "ngram");
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void EveryWindowPartiallyRejected_StillMatchesPlainGreedy(BackendType backend)
    {
        RunAndCheck(backend, (t, vocab) => new PartiallyWrongDrafter(NGram(t, maxDraft: 4), vocab),
                    requireRollback: true, "wrong-tail");
    }

    [Theory]
    [InlineData(BackendType.Cpu)]
    [InlineData(BackendType.GgmlCpu)]
    public void SnapshotVerifyRestoreRewind_EqualsAPlainDecodeOfTheAcceptedPrefix(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        using var model = ModelBase.Create(Fixture(), backend);
        var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        int vocab = model.Config.VocabSize;
        int[] prompt = SeedPrompt();
        int[] window = { 70, 71, 72 };

        // The reference: a plain, sequential decode of the window.
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        var sequential = new float[window.Length][];
        for (int i = 0; i < window.Length; i++)
            sequential[i] = (float[])model.Forward(new[] { window[i] }).Clone();

        // The speculative protocol: snapshot at P, verify the window as ONE
        // batch, then undo it.
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        int position = target.CacheSeqLen;
        Assert.Equal(prompt.Length, position);
        target.SpecEnsureCapacity(position + window.Length);
        target.SpecSnapshotRecurrentState();
        var verify = new float[window.Length * vocab];
        target.SpecForward(window, null, verify, allLogitsRows: true);
        Assert.Equal(position + window.Length, target.CacheSeqLen);

        // The multi-row verify IS the sequential decode, row for row.
        for (int i = 0; i < window.Length; i++)
        {
            var row = new float[vocab];
            Array.Copy(verify, i * vocab, row, 0, vocab);
            AssertClose(sequential[i], row, 1e-3, $"verify row {i}");
        }

        // Reject everything after the first token: restore, rewind, re-forward
        // the accepted prefix - and the state must be a plain decode's.
        target.SpecRestoreRecurrentState();
        target.SpecRewindCache(position);
        Assert.Equal(position, target.CacheSeqLen);
        var replay0 = new float[vocab];
        target.SpecForward(new[] { window[0] }, null, replay0, allLogitsRows: false);
        AssertClose(sequential[0], replay0, 1e-3, "re-forward of the accepted prefix");
        float[] next1 = model.Forward(new[] { window[1] });
        AssertClose(sequential[1], next1, 1e-3, "continuation after the rollback");

        // A second window from here: the snapshot is re-taken, so the second
        // rollback lands on the NEW position, not the first snapshot's.
        int position2 = target.CacheSeqLen;
        target.SpecSnapshotRecurrentState();
        target.SpecForward(new[] { window[2], 73 }, null, new float[2 * vocab], allLogitsRows: true);
        target.SpecRestoreRecurrentState();
        target.SpecRewindCache(position2);
        Assert.Equal(position2, target.CacheSeqLen);
        float[] next2 = model.Forward(new[] { window[2] });
        AssertClose(sequential[2], next2, 1e-3, "continuation after the second rollback");
    }

    [GlmNativeCudaFact]
    public void CudaNgramRollback()
        => NGramSpeculativeGreedy_MatchesPlainGreedy_AndRollsBack(BackendType.GgmlCuda);

    [GlmNativeCudaFact]
    public void CudaEveryWindowPartiallyRejected()
        => EveryWindowPartiallyRejected_StillMatchesPlainGreedy(BackendType.GgmlCuda);

    [GlmNativeCudaFact]
    public void CudaSnapshotVerifyRestoreRewind()
        => SnapshotVerifyRestoreRewind_EqualsAPlainDecodeOfTheAcceptedPrefix(BackendType.GgmlCuda);

    [Fact]
    public void BoundSlots_AbaRollbackPreservesEachContinuation()
        => CheckBoundSlotsAbaRollback(BackendType.GgmlCpu);

    [GlmNativeCudaFact]
    public void CudaBoundSlots_AbaRollbackPreservesEachContinuation()
        => CheckBoundSlotsAbaRollback(BackendType.GgmlCuda);

    private void CheckBoundSlotsAbaRollback(BackendType backend)
    {
        using var env = EnvironmentForFixture();
        string path = Fixture();
        using var model = (GlmDsaModel)ModelBase.Create(path, backend);
        using var cold = ModelBase.Create(path, backend);
        var target = (ISpeculativeTarget)model;
        int[] a = { 65, 66, 67, 68, 69 };
        int[] b = { 71, 72 };
        int vocab = model.Config.VocabSize;
        model.Forward(a);
        model.BindSequenceCache("A");
        Assert.Equal(0, target.CacheSeqLen);
        model.Forward(a);
        model.BindSequenceCache("B");
        Assert.Equal(0, target.CacheSeqLen);
        model.Forward(b);
        model.BindSequenceCache("A");
        Assert.Equal(a.Length, target.CacheSeqLen);
        target.SpecSnapshotRecurrentState();
        target.SpecForward(new[] { 74, 75, 76 }, null, new float[3 * vocab], true);
        target.SpecRestoreRecurrentState();
        target.SpecRewindCache(a.Length);
        float[] actualA = (float[])model.Forward(new[] { 74 }).Clone();
        cold.Forward(a);
        AssertClose(cold.Forward(new[] { 74 }), actualA, 1e-6, "A rollback continuation");
        model.BindSequenceCache("B");
        Assert.Equal(b.Length, target.CacheSeqLen);
        float[] actualB = (float[])model.Forward(new[] { 77 }).Clone();
        cold.ResetKVCache();
        cold.Forward(b);
        AssertClose(cold.Forward(new[] { 77 }), actualB, 1e-6, "B isolated continuation");
        model.BindSequenceCache("A");
        Assert.Equal(a.Length + 1, target.CacheSeqLen);
        model.RestorePrimaryCache();
        Assert.Equal(a.Length, target.CacheSeqLen);
        model.OnSequenceReleased("A");
        model.OnSequenceReleased("B");
    }

    [Fact]
    public void ManagedAndNativeSpecForward_AgreeRowForRow()
    {
        using var env = EnvironmentForFixture();
        string path = Fixture();
        int[] prompt = SeedPrompt();
        int[] window = { 70, 71, 72, 73 };

        (float[] h, float[] logits) Run(BackendType backend)
        {
            using var model = ModelBase.Create(path, backend);
            var target = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            var h = new float[window.Length * target.SpecFeatureSize];
            var logits = new float[window.Length * model.Config.VocabSize];
            Array.Fill(h, float.NaN);
            Array.Fill(logits, float.NaN);
            target.SpecSnapshotRecurrentState();
            target.SpecForward(window, h, logits, allLogitsRows: true);
            return (h, logits);
        }

        var managed = Run(BackendType.Cpu);
        var native = Run(BackendType.GgmlCpu);
        Assert.All(native.h, v => Assert.False(float.IsNaN(v), "the native verify left a hidden-state row unwritten"));
        AssertClose(managed.h, native.h, 2e-3, "post-norm hidden states");
        AssertClose(managed.logits, native.logits, 2e-3, "all-rows logits");
    }
}
