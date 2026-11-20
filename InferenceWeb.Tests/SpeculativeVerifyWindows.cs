// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// The verify windows a <see cref="SpeculativeDecoder"/> ran (<see cref="SpeculativeDecoder.VerifyObserverForTests"/>), and the
/// target's logits recomputed over one of them.
/// <para>A speculative stream's token comes from a verify row. Where a backend's verify rows round apart from its one-row decode
/// (ggml_cuda), the stream can part from plain greedy at a position that is not a near-tie of the one-row logits. Such a
/// parting is still exact speculation when the target's own logits over the window the stream was verified in prefer the
/// stream's token.</para>
/// <para>The recomputation resumes at the window's first token: the tokens before it are prefilled in one forward, and then the
/// window, the same tokens in the same number of rows, is verified whole. Where earlier verify windows wrote that context
/// in the run, the prefilled context rounds apart from it, so the recomputed row is the window's shape over a context of
/// other bits.</para>
/// </summary>
internal sealed class VerifyWindowLog
{
    /// <summary>One verify step: the position of the token the window starts from, the window (that token, then every draft),
    /// the drafts accepted, and the token drawn after them.</summary>
    internal readonly record struct Window(int Start, int[] Tokens, int Accepted, int Next);

    private readonly List<Window> _windows = new();

    public VerifyWindowLog(SpeculativeDecoder decoder)
        => decoder.VerifyObserverForTests = (start, tokens, accepted, next) =>
            _windows.Add(new Window(start, (int[])tokens.Clone(), accepted, next));

    public void Clear() => _windows.Clear();

    /// <summary>The window whose rows drew the token at <paramref name="position"/> (an absolute position), or null where a
    /// prefill or a plain step drew it. Row r of a window starting at s draws the token at s + 1 + r, for r up to the accepted
    /// drafts (the last of those rows draws the step's next token).</summary>
    public Window? Drawing(int position)
    {
        for (int i = _windows.Count - 1; i >= 0; i--)
        {
            Window w = _windows[i];
            if (position > w.Start && position <= w.Start + 1 + w.Accepted)
                return w;
        }
        return null;
    }

    /// <summary>The target's log-probabilities for the token at <paramref name="position"/>, recomputed over
    /// <paramref name="window"/>. <paramref name="context"/> is every token before the window's first, and is prefilled in
    /// one forward. Then <see cref="ISpeculativeTarget.SpecForward"/> runs the window whole, with every row's logits.</summary>
    public static double[] Recompute(ModelBase model, int[] context, Window window, int position)
    {
        Assert.Equal(window.Start, context.Length);
        var spec = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        model.ResetKVCache();
        float[] first = model.Forward(context);
        int stride = first.Length, vocab = Math.Min(model.Config.VocabSize, stride);
        var all = new float[(long)window.Tokens.Length * stride];
        spec.SpecForward(window.Tokens, new float[(long)window.Tokens.Length * spec.SpecFeatureSize], all, allLogitsRows: true);
        int row = position - window.Start - 1;
        Assert.InRange(row, 0, window.Tokens.Length - 1);
        return MuseGlimmerParity.LogSoftmax(all.AsSpan((int)((long)row * stride), vocab).ToArray(), vocab);
    }
}
