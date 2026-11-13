// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    /// <summary>A prompt chunk committed with only the rows its drafter's context needs
    /// (<see cref="SpeculativeExecution.PrefillFeatureRows"/>: the context at the prompt's end and the row before it) hands
    /// the drafter every token in order, the same (token, preceding row) pair for each position that context holds as
    /// the whole chunk does, the older tokens without features, and carries the same last row into the first draft.</summary>
    [Theory]
    [InlineData(0, 12, 12)]    // the context starts inside the chunk
    [InlineData(6, 6, 12)]     // the chunk ends the prompt, the context starts before it: every row
    [InlineData(0, 6, 40)]     // the chunk is older than the context: its last row only
    public void CommitPrefill_TailRows_FeedTheDrafterContextAsAllRowsDo(int start, int count, int promptEnd)
    {
        const int context = 5;
        var fullModel = new FakeSpeculativeModel();
        int width = ((ISpeculativeTarget)fullModel).SpecFeatureSize;
        int[] chunk = Enumerable.Range(start + 1, count).ToArray();
        var rows = new float[count * width];
        for (int i = 0; i < rows.Length; i++) rows[i] = i + 1;

        var full = new ContextDrafter(context, width);
        var tail = new ContextDrafter(context, width);
        var fullExec = new SpeculativeExecution(fullModel, full);
        var tailExec = new SpeculativeExecution(new FakeSpeculativeModel(), tail);
        int c = tailExec.PrefillFeatureRows(start, count, promptEnd);
        int expected = Math.Max(1, Math.Min(count, start + count - (promptEnd - context - 1)));
        Assert.Equal(expected, c);

        fullExec.CommitPrefill(chunk, start, rows, promptEnd);
        tailExec.CommitPrefill(chunk, start, rows[^(c * width)..], promptEnd);

        Assert.Equal(full.Tokens, tail.Tokens);
        foreach (var (position, _) in full.Tokens)
        {
            if (position >= promptEnd - context)
                Assert.Equal(full.Rows[position], tail.Rows[position]);
            else
                Assert.True(tail.Rows[position] == null || tail.Rows[position]!.SequenceEqual(full.Rows[position] ?? Array.Empty<float>()),
                    $"position {position}, older than the context, was fed another row");
        }
    }

    /// <summary>Records what a block drafter with a <paramref name="context"/>-row window is handed.</summary>
    private sealed class ContextDrafter : ISpeculator
    {
        private readonly int _context, _width;

        public ContextDrafter(int context, int width)
        {
            _context = context;
            _width = width;
        }

        public List<(int Position, int Token)> Tokens { get; } = new();
        public Dictionary<int, float[]?> Rows { get; } = new();

        public string Name => "context";
        public int MaxDraftTokens => 4;
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0f;
        public bool NeedsHiddenState => true;
        public bool HandlesOwnPrefill => false;
        public int PrefillContextRows => _context;
        public bool ConditionsOnPrompt => true;

        public int Propose(in DraftContext ctx, List<int> draftOut) => 0;

        public void Commit(int[] tokens, float[]? hRows, int startPos)
        {
            for (int k = 0; k < tokens.Length; k++)
            {
                Tokens.Add((startPos + k, tokens[k]));
                Rows[startPos + k] = hRows?.AsSpan(k * _width, _width).ToArray();
            }
        }

        public void Reset() { }
        public void Dispose() { }
    }
}
