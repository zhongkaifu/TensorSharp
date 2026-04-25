// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for the ModelService-level conversation tracking that keeps raw output tokens
/// associated with assistant messages across HTTP requests, enabling the next turn's
/// prompt render to splice raw tokens in for cached KV state.
/// </summary>
public class ModelServiceKVCacheTests
{
    [Fact]
    public void ResolvePrefillChunkSize_CudaLongPrompt_UsesSafeChunkSize()
    {
        int chunkSize = ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, 11573);
        Assert.Equal(5120, chunkSize);
    }

    [Fact]
    public void ResolvePrefillChunkSize_NonCudaLongPrompt_ChunksAt2048()
    {
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCpu, 11573));
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.GgmlMetal, 11573));
        Assert.Equal(2048, ModelService.ResolvePrefillChunkSize(BackendType.Cpu, 11573));
    }

    [Fact]
    public void ResolvePrefillChunkSize_ZeroOrNegative_ReturnsZero()
    {
        Assert.Equal(0, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, 0));
        Assert.Equal(0, ModelService.ResolvePrefillChunkSize(BackendType.GgmlCuda, -5));
    }

    [Fact]
    public void AugmentWithCachedRawTokens_FreshService_ReturnsIncomingUnchanged()
    {
        // Without prior turns we can't augment anything.
        var tracked = new List<ChatMessage>();
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "hi" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Single(result);
        Assert.Equal("hi", result[0].Content);
        Assert.Null(result[0].RawOutputTokens);
    }

    [Fact]
    public void AugmentWithCachedRawTokens_NullInput_ReturnsNull()
    {
        Assert.Null(ModelService.AugmentWithCachedRawTokens(null, new List<ChatMessage>()));
    }

    [Fact]
    public void AugmentWithCachedRawTokens_PreservesIncomingRawTokensIfAlreadySet()
    {
        // If the caller already attached raw tokens (e.g. test harness), the service must
        // not overwrite them with stale tracked values.
        var tracked = new List<ChatMessage>();
        var explicitTokens = new List<int> { 9001, 9002 };
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u1" },
            new() { Role = "assistant", Content = "a1", RawOutputTokens = explicitTokens },
            new() { Role = "user", Content = "u2" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Same(explicitTokens, result[1].RawOutputTokens);
    }

    /// <summary>
    /// Reproduces the Qwen 3.5 / 3.6 WebUI cache-reset bug:
    /// the streaming output parser strips &lt;think&gt;...&lt;/think&gt; framing from the
    /// assistant text before the WebUI accumulates it, so on the next chat request the
    /// WebUI sends back an assistant message whose Content is a STRIPPED subset of what
    /// our tracked history stored. The augmenter MUST still recognise this as the same
    /// turn and splice the cached raw output tokens, otherwise every multi-turn chat with
    /// a thinking model degenerates to a full prompt re-prefill.
    /// </summary>
    [Fact]
    public void AugmentWithCachedRawTokens_WebUIParsedContentMismatch_StillSplicesRawTokens()
    {
        // Simulate the result of UpdateTrackedHistory() after a previous turn:
        //   - User asked "What is 1+1?"
        //   - Model raw-emitted "<think>let me think</think>1+1=2" (raw bytes from token decoding)
        //   - Tracked content holds the FULL raw text; raw output tokens are the model's bytes.
        var rawTokens = new List<int> { 11, 22, 33, 44, 55 };
        var tracked = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is 1+1?" },
            new()
            {
                Role = "assistant",
                Content = "<think>let me think</think>1+1=2",  // RAW text the model emitted
                RawOutputTokens = rawTokens,
            },
        };

        // Simulate what the WebUI sends back on the SECOND turn: the OutputParser stripped
        // the thinking block, so Content is just "1+1=2" (plus an optional Thinking field).
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "What is 1+1?" },          // unchanged
            new() { Role = "assistant", Content = "1+1=2", Thinking = "let me think" }, // PARSED!
            new() { Role = "user", Content = "What is 2+2?" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Equal(3, result.Count);
        Assert.Same(rawTokens, result[1].RawOutputTokens);
    }

    [Fact]
    public void AugmentWithCachedRawTokens_UserMessageEdited_StopsSplicingAtEdit()
    {
        // When the user edits an earlier message, the conversation diverges at that
        // position and no later assistant message corresponds to anything we have cached.
        var tracked = new List<ChatMessage>
        {
            new() { Role = "user", Content = "ORIGINAL" },
            new()
            {
                Role = "assistant",
                Content = "<think>x</think>response_to_original",
                RawOutputTokens = new List<int> { 1, 2, 3 },
            },
        };

        // User edited the first message to "EDITED" - the assistant message is no longer
        // a continuation of anything we have raw tokens for.
        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "EDITED" },
            new() { Role = "assistant", Content = "response_to_original" },
            new() { Role = "user", Content = "follow-up" },
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        // No augmentation should have happened: the assistant message keeps its caller-set
        // RawOutputTokens (null) since the user message before it diverged.
        Assert.Null(result[1].RawOutputTokens);
    }

    /// <summary>
    /// Reproduces the SECOND symptom from the Qwen 3.6 WebUI bug: even after the
    /// content-mismatch fix, multi-turn conversations beyond two turns still degraded to
    /// full reset because the previous turn's `UpdateTrackedHistory` re-cloned from the
    /// (raw-token-less) WebUI request body instead of the augmented history. So the
    /// raw tokens of every-but-the-most-recent assistant fell off the tracked record,
    /// and on turn 3 the augmenter could only restore raw tokens for the IMMEDIATELY
    /// previous assistant, not earlier ones.
    ///
    /// This test walks the augmenter through three simulated WebUI turns and asserts
    /// that all prior assistant turns are still augmentable.
    /// </summary>
    [Fact]
    public void AugmentWithCachedRawTokens_ThreeTurnsViaWebUIFlow_AllPriorAssistantsCarryRawTokens()
    {
        var tracked = new List<ChatMessage>();

        var raw1 = new List<int> { 11, 12, 13 };
        var raw2 = new List<int> { 21, 22, 23, 24 };

        // Simulate state after turn 1 generation (UpdateTrackedHistory was called).
        tracked.Add(new ChatMessage { Role = "user", Content = "Q1" });
        tracked.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "RAW1",
            RawOutputTokens = raw1,
        });

        // Simulate the WebUI sending TURN 2 - assistant1 content is parsed (no raw tokens).
        var turn2Incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
        };
        var turn2Augmented = ModelService.AugmentWithCachedRawTokens(turn2Incoming, tracked);
        Assert.Same(raw1, turn2Augmented[1].RawOutputTokens);

        // Simulate state AFTER turn 2 generation: tracked is rebuilt FROM THE AUGMENTED
        // history (the fix), with the new assistant turn appended. This is what the bug
        // breaks: if the rebuild uses turn2Incoming instead, raw1 would be lost forever.
        tracked.Clear();
        for (int i = 0; i < turn2Augmented.Count; i++)
            tracked.Add(turn2Augmented[i]);
        tracked.Add(new ChatMessage { Role = "assistant", Content = "RAW2", RawOutputTokens = raw2 });

        // Simulate the WebUI sending TURN 3 - both prior assistants have parsed content.
        var turn3Incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
            new() { Role = "assistant", Content = "PARSED2" },
            new() { Role = "user", Content = "Q3" },
        };
        var turn3Augmented = ModelService.AugmentWithCachedRawTokens(turn3Incoming, tracked);

        Assert.Same(raw1, turn3Augmented[1].RawOutputTokens);
        Assert.Same(raw2, turn3Augmented[3].RawOutputTokens);
    }

    [Fact]
    public void AugmentWithCachedRawTokens_ThreeTurnConversation_SplicesAllPriorAssistantsByPosition()
    {
        var tracked = new List<ChatMessage>();

        var raw1 = new List<int> { 100, 101 };
        var raw2 = new List<int> { 200, 201, 202 };
        tracked.Add(new ChatMessage { Role = "user", Content = "u1" });
        tracked.Add(new ChatMessage { Role = "assistant", Content = "<think>...</think>a1raw", RawOutputTokens = raw1 });
        tracked.Add(new ChatMessage { Role = "user", Content = "u2" });
        tracked.Add(new ChatMessage { Role = "assistant", Content = "<think>...</think>a2raw", RawOutputTokens = raw2 });

        var incoming = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u1" },
            new() { Role = "assistant", Content = "a1raw" },     // parsed
            new() { Role = "user", Content = "u2" },
            new() { Role = "assistant", Content = "a2raw" },     // parsed
            new() { Role = "user", Content = "u3" },             // new
        };

        var result = ModelService.AugmentWithCachedRawTokens(incoming, tracked);

        Assert.Same(raw1, result[1].RawOutputTokens);
        Assert.Same(raw2, result[3].RawOutputTokens);
        Assert.Null(result[4].RawOutputTokens);
    }
}
