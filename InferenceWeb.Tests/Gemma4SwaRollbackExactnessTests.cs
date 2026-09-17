// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Does a rejected speculative window leave Gemma 4's sliding-window cache exact?
//
// A verify forwards [t0, d1..dK] as one batch and writes every row's K/V at its
// true position. The SWA layers keep a circular cache of exactly one window (512
// slots, slot = position % 512), so once the context has wrapped, row p+i lands
// in the slot that held position p+i-512 - a position the NEXT decode step still
// attends to whenever two or more drafts are rejected. Rewinding the position
// counter cannot bring those slots back. Below the window nothing is clobbered,
// so the same protocol there is the control.
//
// The test drives the trunk directly (ISpeculativeTarget, no draft head needed):
// plain greedy decode after a prefill versus the same decode after a verify of
// deliberately wrong drafts and a rollback that keeps only t0 - the rollback
// LinearSpecTrunk performs. Under the window the streams must agree token for
// token and the first post-rollback logits must agree to floating-point noise;
// past the window they must agree the same way, or the cache is not exact.
//
//   TS_TEST_MODEL_DIR=~/work/models/gemma-4-E2B TS_TEST_GGML_BACKEND=metal
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4SwaRollbackExactnessTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private readonly ITestOutputHelper _output;

    public Gemma4SwaRollbackExactnessTests(ITestOutputHelper output) { _output = output; }

    [ModelTheory(EnvModelDir, "gemma-4-e2b")]
    [InlineData(300, 8)]    // under the 512 window: nothing to clobber (control)
    [InlineData(900, 8)]    // wrapped: rejected rows overwrote the window's oldest slots
    [InlineData(900, 2)]    // wrapped, one rejected draft: no needed slot is touched
    public void RejectedDraftWindow_LeavesTheNextDecodeExact(int promptTokens, int draftCount)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = dir == null ? null : TestGates.FindGguf(dir, "gemma-4-e2b");
        if (modelPath == null) { _output.WriteLine("no gemma-4-e2b model; skipping"); return; }
        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant() switch
        {
            "metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.GgmlCuda,
            _ => BackendType.GgmlCpu,
        };

        using var model = ModelBase.Create(modelPath, backend);
        var spec = (ISpeculativeTarget)model;
        int[] prompt = BuildPrompt(model, promptTokens);
        const int follow = 8;

        // ---- plain: prefill, then greedy decode `follow` tokens one at a time.
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        int t0 = Argmax(logits);
        var plainTokens = new List<int>();
        float[] plainFirst = null;
        int t = t0;
        for (int i = 0; i < follow; i++)
        {
            logits = model.Forward(new[] { t });
            if (i == 0) plainFirst = (float[])logits.Clone();
            t = Argmax(logits);
            plainTokens.Add(t);
        }

        // ---- speculative: the same prefill, then a verify of [t0, wrong x K].
        model.ResetKVCache();
        logits = model.ForwardRefill(prompt);
        Assert.Equal(t0, Argmax(logits));
        int position = spec.CacheSeqLen;
        Assert.Equal(prompt.Length, position);

        int vocab = model.Config.VocabSize;
        var batch = new int[draftCount + 1];
        batch[0] = t0;
        for (int i = 1; i <= draftCount; i++)
            batch[i] = WrongToken(t0, i, vocab);
        var verifyLogits = new float[(draftCount + 1) * vocab];
        spec.SpecEnsureCapacity(position + draftCount + 1);
        spec.SpecSnapshotRecurrentState();
        spec.SpecForward(batch, null, verifyLogits, allLogitsRows: true);
        int drawn = Argmax(verifyLogits.AsSpan(0, vocab));
        // Row 0 is t0's own next token; the wrong drafts guarantee it is rejected
        // at row 1 (unless the trunk happens to predict the junk, which it will not).
        Assert.NotEqual(batch[1], drawn);
        spec.SpecOnVerifyAccepted(0, draftCount);

        // The rollback LinearSpecTrunk performs: keep t0 only.
        if (spec.SpecVerifyPersistsAcceptedKv)
        {
            spec.SpecRewindCache(position + 1);
        }
        else
        {
            spec.SpecRestoreRecurrentState();
            spec.SpecRewindCache(position);
            var scratch = new float[vocab];
            spec.SpecForward(new[] { t0 }, null, scratch, allLogitsRows: false);
        }
        Assert.Equal(position + 1, spec.CacheSeqLen);

        // The drawn token is what plain decoding produced from the same row.
        Assert.Equal(plainTokens[0], drawn);
        float rowDiff = MaxAbsDiff(plainFirst, verifyLogits.AsSpan(0, vocab).ToArray());

        // ---- continue plainly from the rollback and compare.
        var specTokens = new List<int> { drawn };
        float[] specFirst = null;
        t = drawn;
        for (int i = 1; i < follow; i++)
        {
            logits = model.Forward(new[] { t });
            if (i == 1) specFirst = (float[])logits.Clone();
            t = Argmax(logits);
            specTokens.Add(t);
        }

        // Plain decode step 2's logits (after t1) versus the same step after the rollback.
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        model.Forward(new[] { t0 });
        float[] plainSecond = (float[])model.Forward(new[] { plainTokens[0] }).Clone();
        float secondDiff = MaxAbsDiff(plainSecond, specFirst);
        float scale = Math.Max(1e-6f, plainSecond.Max(Math.Abs));

        _output.WriteLine($"prompt={prompt.Length} drafts={draftCount} persistsKv={spec.SpecVerifyPersistsAcceptedKv}");
        _output.WriteLine($"plain: {string.Join(",", plainTokens)}");
        _output.WriteLine($"spec:  {string.Join(",", specTokens)}");
        _output.WriteLine($"verify row-0 vs plain step-1 logits: max|diff| {rowDiff:E2}");
        _output.WriteLine($"first decode after rollback vs plain step-2 logits: max|diff| {secondDiff:E2} (logit scale {scale:F1})");

        // Different kernels (verify batch vs single-token decode) may differ by
        // floating-point noise; a clobbered window differs by whole logits.
        Assert.True(secondDiff <= 0.02f * scale,
            $"logits after the rollback differ from plain decoding by {secondDiff:E2} " +
            $"(scale {scale:F1}); the rejected rows are still visible to attention");
        Assert.Equal(plainTokens, specTokens);
    }

    /// <summary>
    /// The other half of the protocol: a FULLY accepted verify past the window. The
    /// executor keeps every row and neither rolls back nor re-forwards, so the ring must
    /// hold exactly the verify's rows. Gated on a dense Gemma 4 without per-layer
    /// embeddings (12B), because that is the trunk whose verify KV is not kept on a
    /// partial acceptance - the case where the ring used to be "restored" on a full
    /// acceptance too, putting the evicted positions back over the committed rows.
    ///
    ///   TS_TEST_MODEL_DIR=~/work/models/gemma-4-12b TS_TEST_GGML_BACKEND=metal
    /// </summary>
    [ModelTheory(EnvModelDir, "gemma-4-12b")]
    [InlineData(1300, 7)]   // past the 1,024-token window
    [InlineData(600, 7)]    // under the window (control)
    public void AcceptedDraftWindow_LeavesTheNextDecodeExact(int promptTokens, int draftCount)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = dir == null ? null : TestGates.FindGguf(dir, "gemma-4-12b");
        if (modelPath == null) { _output.WriteLine("no gemma-4-12b model; skipping"); return; }
        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant() switch
        {
            "metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.GgmlCuda,
            _ => BackendType.GgmlCpu,
        };

        using var model = ModelBase.Create(modelPath, backend);
        var spec = (ISpeculativeTarget)model;
        int[] prompt = BuildPrompt(model, promptTokens);
        int vocab = model.Config.VocabSize;
        int follow = draftCount + 6;

        // ---- plain: prefill, then greedy decode one token at a time.
        // plainTokens[i] is fed at step i; plainLogits[i] is what feeding it returned.
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        var plainTokens = new List<int> { Argmax(logits) };
        var plainLogits = new List<float[]>();
        for (int i = 0; i < follow; i++)
        {
            logits = model.Forward(new[] { plainTokens[i] });
            plainLogits.Add((float[])logits.Clone());
            plainTokens.Add(Argmax(logits));
        }

        // ---- speculative: the same prefill, then ONE verify of [t0, t1..tK] with the
        // plain continuation as the drafts, so every draft is accepted.
        model.ResetKVCache();
        model.ForwardRefill(prompt);
        int position = spec.CacheSeqLen;
        var batch = plainTokens.Take(draftCount + 1).ToArray();
        var verifyLogits = new float[(draftCount + 1) * vocab];
        spec.SpecEnsureCapacity(position + draftCount + 1);
        spec.SpecSnapshotRecurrentState();
        spec.SpecForward(batch, null, verifyLogits, allLogitsRows: true);
        for (int row = 0; row < draftCount; row++)
            Assert.Equal(plainTokens[row + 1], Argmax(verifyLogits.AsSpan(row * vocab, vocab)));
        // Full acceptance: SpeculativeExecution reports it and keeps every row.
        spec.SpecOnVerifyAccepted(draftCount, draftCount);
        Assert.Equal(position + draftCount + 1, spec.CacheSeqLen);

        // ---- continue plainly from the committed window and compare.
        var specTokens = new List<int>();
        float maxDiff = 0, scale = 1e-6f;
        for (int i = draftCount + 1; i < follow; i++)
        {
            logits = model.Forward(new[] { plainTokens[i] });
            maxDiff = Math.Max(maxDiff, MaxAbsDiff(plainLogits[i], logits));
            scale = Math.Max(scale, plainLogits[i].Max(Math.Abs));
            specTokens.Add(Argmax(logits));
        }
        var expected = plainTokens.Skip(draftCount + 2).Take(specTokens.Count).ToList();

        _output.WriteLine($"prompt={prompt.Length} drafts={draftCount} persistsKv={spec.SpecVerifyPersistsAcceptedKv}");
        _output.WriteLine($"plain: {string.Join(",", expected)}");
        _output.WriteLine($"spec:  {string.Join(",", specTokens)}");
        _output.WriteLine($"decode after the accepted window vs plain: max|diff| {maxDiff:E2} (logit scale {scale:F1})");

        Assert.True(maxDiff <= 0.02f * scale,
            $"logits after a fully accepted verify differ from plain decoding by {maxDiff:E2} " +
            $"(scale {scale:F1}); the committed rows are not what the ring holds");
        Assert.Equal(expected, specTokens);
    }

    private static int[] BuildPrompt(ModelBase model, int tokens)
    {
        var sb = new System.Text.StringBuilder();
        int line = 1;
        while (true)
        {
            for (int i = 0; i < 40; i++, line++)
                sb.Append($"{line,4}  var item{line} = new Widget{line % 7}(); item{line}.Add({line * 3}); // total {line * line}\n");
            var enc = model.Tokenizer.Encode(sb.ToString(), addSpecial: true);
            if (enc.Count >= tokens)
                return enc.GetRange(0, tokens).ToArray();
        }
    }

    /// <summary>A draft that is wrong on purpose: a small fixed id far from anything
    /// the trunk would emit next in a code listing (and never t0 itself).</summary>
    private static int WrongToken(int t0, int i, int vocab)
    {
        int w = 1000 + i * 37;
        if (w == t0) w++;
        return w % vocab;
    }

    private static int Argmax(ReadOnlySpan<float> v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static int Argmax(float[] v) => Argmax(v.AsSpan());

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        int n = Math.Min(a.Length, b.Length);
        for (int i = 0; i < n; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }
}
