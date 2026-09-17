// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Pooled prefix blocks: shared across conversations only up to the public prefix, and
/// keyed on media by content per block rather than by one whole-prompt media string.
/// </summary>
public class PooledPrefixScopeAndMediaTests
{
    private const int BlockSize = 8;
    private const string Fingerprint = "fp-pooled-scope";

    private static ContinuousBatchScheduler NewScheduler(BlockPool pool) => new(
        new SchedulerConfig
        {
            MaxNumBatchedTokens = 1024,
            MaxNumRunningSequences = 16,
            MaxPrefillChunkSize = 64,
            NumBlocks = pool.NumBlocks,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = BlockSize,
        },
        pool, Fingerprint, NullLogger.Instance);

    private static SequenceState Seq(
        string id, int[] prompt, string scope = null, int sharedPrefix = 0, IReadOnlyList<PromptMediaSpan> media = null)
        => new(id, prompt.ToList(), 1, BlockSize, SamplingConfig.Default,
            mediaSpans: media, sharedPrefixTokens: sharedPrefix, cacheScope: scope);

    /// <summary>Register every full block of <paramref name="source"/>'s prompt as if its
    /// forward had written and captured them.</summary>
    private static void Publish(ContinuousBatchScheduler sched, BlockPool pool, SequenceState source, int tokens)
    {
        foreach (var block in pool.AllocateNew(tokens / BlockSize))
            source.BlockTable.AppendBlock(block);
        source.AdvanceComputedTokens(tokens);
        sched.OnBlocksCommitted(source, previousTokens: 0);
    }

    private static int AdmitAndReuse(ContinuousBatchScheduler sched, SequenceState seq)
    {
        sched.Submit(seq);
        sched.Schedule();
        return seq.PrefixCacheReusedTokens;
    }

    [Fact]
    public void AnotherScopesBlocks_AreSharedOnlyUpToThePublicPrefix()
    {
        var pool = new BlockPool(32, BlockSize, 0);
        var sched = NewScheduler(pool);
        int[] prompt = Enumerable.Range(1, 5 * BlockSize + 1).ToArray();

        Publish(sched, pool, Seq("A1", prompt, scope: "A", sharedPrefix: 2 * BlockSize), 5 * BlockSize);

        Assert.Equal(2 * BlockSize, AdmitAndReuse(sched, Seq("B1", prompt, scope: "B", sharedPrefix: 2 * BlockSize)));
        Assert.Equal(5 * BlockSize, AdmitAndReuse(sched, Seq("A2", prompt, scope: "A", sharedPrefix: 2 * BlockSize)));
    }

    [Fact]
    public void BlocksBeforeAMediaSpan_AreSharedWithATextOnlyPrompt()
    {
        var pool = new BlockPool(32, BlockSize, 0);
        var sched = NewScheduler(pool);
        int[] prompt = Enumerable.Range(1, 5 * BlockSize + 1).ToArray();

        Publish(sched, pool, Seq("text", prompt), 5 * BlockSize);

        // The same tokens, but block 3 holds an image: blocks 0-2 are text-only and
        // still match; the whole-prompt salt used to make all of them miss.
        var image = new[] { new PromptMediaSpan(3 * BlockSize, 4 * BlockSize, "img:x") };
        Assert.Equal(3 * BlockSize, AdmitAndReuse(sched, Seq("image", prompt, media: image)));
    }

    [Fact]
    public void AMediaSpan_MatchesByContent_AndADifferentImageStopsAtItsBlock()
    {
        var pool = new BlockPool(32, BlockSize, 0);
        var sched = NewScheduler(pool);
        int[] prompt = Enumerable.Range(1, 5 * BlockSize + 1).ToArray();
        var photo = new[] { new PromptMediaSpan(3 * BlockSize, 4 * BlockSize, "img:photo") };

        Publish(sched, pool, Seq("first", prompt, media: photo), 5 * BlockSize);

        Assert.Equal(5 * BlockSize, AdmitAndReuse(sched, Seq("same", prompt, media: photo)));
        var other = new[] { new PromptMediaSpan(3 * BlockSize, 4 * BlockSize, "img:other") };
        Assert.Equal(3 * BlockSize, AdmitAndReuse(sched, Seq("other", prompt, media: other)));
    }

    [Fact]
    public void AdoptionNeverEndsInsideAMediaSpan()
    {
        var pool = new BlockPool(32, BlockSize, 0);
        var sched = NewScheduler(pool);
        int[] prompt = Enumerable.Range(1, 4 * BlockSize + 1).ToArray();
        // The span runs from the middle of block 2 past the end of block 3.
        var photo = new[] { new PromptMediaSpan(2 * BlockSize + 4, 4 * BlockSize + 1, "img:photo") };

        Publish(sched, pool, Seq("first", prompt, media: photo), 4 * BlockSize);

        Assert.Equal(2 * BlockSize, AdmitAndReuse(sched, Seq("again", prompt, media: photo)));
    }

    [Fact]
    public void BlockHashes_WithoutMediaOrScope_AreUnchanged()
    {
        int[] prompt = Enumerable.Range(1, 3 * BlockSize).ToArray();
        var plain = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, Fingerprint);
        var noSalt = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, Fingerprint, _ => null);
        Assert.Equal(plain, noSalt);

        var salted = KvBlockHasher.ComputeBlockHashes(prompt, BlockSize, Fingerprint, b => b == 1 ? "mm:x" : null);
        Assert.Equal(plain[0], salted[0]);
        Assert.NotEqual(plain[1], salted[1]);
        Assert.NotEqual(plain[2], salted[2]);   // carried by the parent chain
    }

    [Theory]
    [InlineData(20, 20)]    // nothing to clamp: the spans are identical and whole
    [InlineData(10, 4)]     // a cut span stops the prefix at its start
    public void ClampReusablePrefix_StopsAtACutSpan(int reusable, int expected)
    {
        var spans = new[] { new PromptMediaSpan(4, 12, "img:a") };
        Assert.Equal(expected, PromptMediaSpans.ClampReusablePrefix(reusable, spans, spans));
    }

    [Fact]
    public void ClampReusablePrefix_ComparesByContentAndPosition()
    {
        var cached = new[] { new PromptMediaSpan(4, 12, "img:a"), new PromptMediaSpan(20, 28, "img:b") };
        Assert.Equal(40, PromptMediaSpans.ClampReusablePrefix(40, cached, cached));
        Assert.Equal(20, PromptMediaSpans.ClampReusablePrefix(40,
            new[] { new PromptMediaSpan(4, 12, "img:a"), new PromptMediaSpan(20, 28, "img:c") }, cached));
        Assert.Equal(4, PromptMediaSpans.ClampReusablePrefix(40,
            new[] { new PromptMediaSpan(4, 12, "img:a") }, cached, allowReuseAcrossSpans: false));
        // Media only on the cached side (the prompt's tokens match but it names no media
        // there) is not the same content either.
        Assert.Equal(4, PromptMediaSpans.ClampReusablePrefix(40, Array.Empty<PromptMediaSpan>(), cached));
        // Media that starts at or after the prefix is irrelevant.
        Assert.Equal(4, PromptMediaSpans.ClampReusablePrefix(4, Array.Empty<PromptMediaSpan>(), cached));
    }
}
