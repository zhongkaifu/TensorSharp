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

/// <summary>Speculative decoding checks shared by the runtime families.</summary>
internal static class SpeculativeRuntimeChecks
{
    /// <summary>Greedy speculation with n-gram drafts over a repetitive prompt (drafts both accepted and
    /// rejected) emits exactly the plain greedy tokens, and the verify path actually ran.</summary>
    public static void AssertMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24)
    {
        int[] prompt = Enumerable.Range(0, 12).Select(i => 40 + i % 6).Concat(Enumerable.Range(0, 9).Select(i => 40 + i % 6)).ToArray();

        var plain = new List<int>();
        using (ModelBase model = create())
        {
            float[] logits = model.ForwardRefill(prompt);
            for (int i = 0; i < maxNew; i++)
            {
                int token = ArgMax(logits);
                plain.Add(token);
                logits = model.Forward(new[] { token });
            }
        }

        using ModelBase target = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeTarget>(target);
        Assert.Null(spec.SpeculationRefusal);
        // Drafts from the known continuation, wrong at their third token every other time: verifies
        // are both fully and partly accepted, whatever the model's own repetition.
        var decoder = new SpeculativeDecoder(spec, new ScriptedSpeculator(prompt.Concat(plain).ToArray(), target.Config.VocabSize));
        List<int> drafted = decoder.GenerateGreedy(prompt, maxNew);
        Assert.Equal(plain.Take(drafted.Count), drafted);
        Assert.Equal(maxNew, drafted.Count);
        // The verify path ran, and some drafts were rejected (the prefix commit, not just a full accept).
        Assert.True(decoder.VerifySteps > 0, $"{label}: no verify ran");
        Assert.True(decoder.TokensAccepted > 0, $"{label}: no draft was accepted");
        Assert.True(decoder.TokensAccepted < decoder.TokensDrafted, $"{label}: every draft was accepted ({decoder.TokensAccepted})");
    }

    /// <summary>Proposes the next tokens of a known sequence, corrupting the third draft of every other
    /// proposal.</summary>
    private sealed class ScriptedSpeculator : ISpeculator
    {
        private readonly int[] _sequence;
        private readonly int _vocab;
        private int _calls;

        public ScriptedSpeculator(int[] sequence, int vocab)
        {
            _sequence = sequence;
            _vocab = vocab;
        }

        public string Name => "scripted";
        public int MaxDraftTokens => 4;
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0f;
        public bool NeedsHiddenState => false;
        public bool HandlesOwnPrefill => false;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int next = ctx.Position < _sequence.Length && _sequence[ctx.Position] == ctx.LastToken ? ctx.Position + 1 : ctx.Position;
            int count = Math.Min(Math.Min(ctx.MaxTokens, MaxDraftTokens), _sequence.Length - next);
            for (int i = 0; i < count; i++) draftOut.Add(_sequence[next + i]);
            if (_calls++ % 2 == 1 && draftOut.Count >= 3)
                draftOut[2] = (draftOut[2] + 1) % _vocab;
            return draftOut.Count;
        }

        public void Commit(int[] tokens, float[]? hRows, int startPos) { }
        public void Reset() { }
        public void Dispose() { }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
