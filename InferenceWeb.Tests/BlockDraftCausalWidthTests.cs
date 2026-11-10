// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="BlockDraftSpeculator"/> with a head whose block attention is causal
/// (<see cref="IDraftHead.DraftBlockIsCausal"/>, DSpine): the first w rows of such a block ARE a w-row block, so a
/// step drafts only the rows it verifies. A non-causal head (DFlash / DFlash2) keeps drafting its trained width.
/// </summary>
public sealed class BlockDraftCausalWidthTests
{
    /// <summary>Records the block width each draft was asked for (the buffer length, as ModelBase.DFlashPropose
    /// reads it) and fills every row.</summary>
    private sealed class WidthRecordingHead : IDraftHead
    {
        public bool Causal;
        public readonly List<int> Widths = new();
        public DraftHeadKind DraftHeadKind => DraftHeadKind.Block;
        public int DraftBlockSize => 15;
        public bool DraftBlockIsCausal => Causal;
        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            Widths.Add(draftOut.Length);
            for (int i = 0; i < draftOut.Length; i++) { draftOut[i] = 100 + i; confOut[i] = 1f; }
            return draftOut.Length;
        }
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos) { }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ACausalBlockDraftsOnlyTheRowsAStepVerifies(bool causal)
    {
        var head = new WidthRecordingHead { Causal = causal };
        var spec = new BlockDraftSpeculator(head, 15, 15) { MinDraftProb = 0f };
        foreach (int max in new[] { 7, 15, 3, 7 })
        {
            var drafts = new List<int>();
            spec.Propose(new DraftContext { LastToken = 1, Position = 10, MaxTokens = max }, drafts);
            // Either way the step gets the block's first rows, as many as it verifies.
            Assert.Equal(Enumerable.Range(100, max), drafts);
        }
        // A causal (DSpine) block is drafted as wide as the step verifies; a non-causal (DFlash) block - whose rows
        // depend on the whole block - always at its trained width.
        Assert.Equal(causal ? new[] { 7, 15, 3, 7 } : new[] { 15, 15, 15, 15 }, head.Widths);
    }
}
