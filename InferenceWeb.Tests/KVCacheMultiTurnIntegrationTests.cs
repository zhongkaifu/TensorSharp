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
/// End-to-end integration tests for the multi-turn KV cache flow. These tests use a
/// FAKE model that records what it was asked to forward / truncate / reset, which lets us
/// verify the orchestration logic without loading a real GGUF.
///
/// The conversation cache is exercised through three scenarios:
///   - "plain" model: every turn extends the prompt; cache should incrementally reuse.
///   - "thinking" model: assistant raw tokens differ from the parsed Content text; cache
///     should still find a perfect prefix match because raw tokens are spliced in.
///   - "recurrent" model: SupportsKVCacheTruncation is false; partial-match must fall
///     back to a full reset, but cache-extension reuse must still work.
/// </summary>
public class KVCacheMultiTurnIntegrationTests
{
    /// <summary>
    /// Records the sequence of operations the orchestrator drives on a model. This is the
    /// "model" in our integration tests - it lets us assert that the orchestrator only
    /// forwards the new (suffix) tokens on cache hits.
    /// </summary>
    private sealed class RecordingFakeModel
    {
        private int _cacheLen;

        public List<int> AllTokensForwarded { get; } = new();
        public List<(string op, int arg)> Operations { get; } = new();
        public bool SupportsTruncation { get; init; } = true;
        public int TotalForwardedThisRun { get; private set; }
        public int CacheLen => _cacheLen;

        public void Reset()
        {
            Operations.Add(("reset", 0));
            _cacheLen = 0;
        }

        public void Truncate(int n)
        {
            if (!SupportsTruncation && n != _cacheLen)
                throw new InvalidOperationException("Recurrent model can only truncate to the current cache length.");
            Operations.Add(("truncate", n));
            _cacheLen = n;
        }

        public float[] Forward(int[] tokens)
        {
            Operations.Add(("forward", tokens.Length));
            TotalForwardedThisRun = tokens.Length;
            for (int i = 0; i < tokens.Length; i++)
                AllTokensForwarded.Add(tokens[i]);
            _cacheLen += tokens.Length;
            // Return arbitrary deterministic logits so the cache.NextLogits is non-null.
            return new float[] { tokens.Length };
        }
    }

    /// <summary>
    /// Tokenizer that emits one token per character, plus a per-special-marker mapping.
    /// Stable enough for these tests to assert exact prefix matches.
    /// </summary>
    private sealed class SimpleTokenizer : ITokenizer
    {
        private readonly Dictionary<string, int> _specialTokens = new()
        {
            ["<bos>"] = 0,
            ["<eos>"] = 1,
            ["<u>"] = 2,
            ["</u>"] = 3,
            ["<a>"] = 4,
            ["</a>"] = 5,
        };

        private readonly Dictionary<char, int> _charIds = new();
        private readonly List<string> _vocab = new();

        public SimpleTokenizer()
        {
            // Pre-populate vocab with special tokens so their ids are stable.
            for (int i = 0; i < 6; i++)
                _vocab.Add($"<sp{i}>");
        }

        public string[] Vocab => _vocab.ToArray();
        public int BosTokenId => 0;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => _vocab.Count;

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var result = new List<int>();
            if (addSpecial)
                result.Add(BosTokenId);
            if (string.IsNullOrEmpty(text)) return result;

            int i = 0;
            while (i < text.Length)
            {
                bool matched = false;
                foreach (var kv in _specialTokens)
                {
                    if (i + kv.Key.Length <= text.Length && text.AsSpan(i, kv.Key.Length).SequenceEqual(kv.Key))
                    {
                        result.Add(kv.Value);
                        i += kv.Key.Length;
                        matched = true;
                        break;
                    }
                }
                if (matched) continue;

                char ch = text[i++];
                if (!_charIds.TryGetValue(ch, out int id))
                {
                    id = _vocab.Count;
                    _charIds[ch] = id;
                    _vocab.Add(ch.ToString());
                }
                result.Add(id);
            }
            return result;
        }

        public string Decode(List<int> ids)
        {
            var sb = new System.Text.StringBuilder();
            if (ids != null)
                foreach (var id in ids)
                    if (id < _vocab.Count)
                        sb.Append(_vocab[id]);
            return sb.ToString();
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            if (tokenId < _vocab.Count)
                buffer.AddRange(System.Text.Encoding.UTF8.GetBytes(_vocab[tokenId]));
        }

        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => _specialTokens.TryGetValue(tokenStr, out var id) ? id : -1;
    }

    /// <summary>
    /// Renderer with a simple Qwen-like format: <c>&lt;u&gt;...&lt;/u&gt;&lt;a&gt;...&lt;/a&gt;</c>
    /// for each turn, ending in &lt;a&gt; for the generation prompt.
    /// </summary>
    private sealed class SimpleRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder();
            foreach (var m in messages)
            {
                if (m.Role == "user")
                    sb.Append("<u>").Append(m.Content ?? "").Append("</u>");
                else if (m.Role == "assistant")
                    sb.Append("<a>").Append(m.Content ?? "").Append("</a>");
            }
            if (addGenerationPrompt)
                sb.Append("<a>");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Drive a multi-turn conversation against the recording fake model. Returns the per-turn
    /// list of (prompt-token-count, tokens-forwarded) pairs so the test can assert reuse.
    /// </summary>
    private static List<(int promptTokens, int forwardedTokens)> RunConversation(
        RecordingFakeModel model,
        SimpleTokenizer tokenizer,
        SimpleRenderer renderer,
        KVCachePromptRenderer kvRenderer,
        KVCache kvCache,
        IList<string> userTurns,
        IList<List<int>> rawOutputsToInject)
    {
        var history = new List<ChatMessage>();
        var result = new List<(int promptTokens, int forwardedTokens)>();

        for (int t = 0; t < userTurns.Count; t++)
        {
            history.Add(new ChatMessage { Role = "user", Content = userTurns[t] });

            var inputTokens = kvRenderer.RenderToTokens(
                tokenizer, chatTemplate: null, history,
                architecture: "fake", addGenerationPrompt: true);

            ReusePlan plan = kvCache.PlanReuse(inputTokens, model.SupportsTruncation);
            int forwarded;
            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                    forwarded = 0;
                    break;
                case ReusePlanKind.PartialReuse:
                {
                    model.Truncate(plan.ReusedPrefixLength);
                    kvCache.TruncateTo(plan.ReusedPrefixLength);
                    var suffix = new int[plan.TokensToForward];
                    for (int i = 0; i < plan.TokensToForward; i++)
                        suffix[i] = inputTokens[plan.ReusedPrefixLength + i];
                    var logits = model.Forward(suffix);
                    kvCache.RecordAppend(suffix, logits);
                    forwarded = suffix.Length;
                    break;
                }
                case ReusePlanKind.Reset:
                default:
                {
                    model.Reset();
                    kvCache.Reset();
                    var allTokens = inputTokens.ToArray();
                    var logits = model.Forward(allTokens);
                    kvCache.RecordAppend(allTokens, logits);
                    forwarded = allTokens.Length;
                    break;
                }
            }

            result.Add((inputTokens.Count, forwarded));

            // Inject the (test-controlled) raw output tokens of this turn into the cache so
            // subsequent turns can splice them in.
            if (t < rawOutputsToInject.Count)
            {
                var raw = rawOutputsToInject[t];
                if (raw != null && raw.Count > 0)
                {
                    model.Forward(raw.ToArray());
                    var logitsTrailing = new float[] { raw.Count };
                    kvCache.RecordAppend(raw, logitsTrailing);

                    history.Add(new ChatMessage
                    {
                        Role = "assistant",
                        Content = $"reply{t + 1}",
                        RawOutputTokens = raw,
                    });
                }
            }
        }

        return result;
    }

    [Fact]
    public void MultiTurn_PlainModel_FirstTurnFullPrefill_SubsequentTurnsForwardOnlyDelta()
    {
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        // Same generated tokens for each turn (clearly distinct from text content).
        var raw1 = new List<int> { 1001, 1002, 1003 };
        var raw2 = new List<int> { 2001, 2002 };

        var stats = RunConversation(model, tokenizer, renderer, kvRenderer, kvCache,
            new[] { "Hi", "More" },
            new[] { raw1, raw2 });

        // Turn 1: full forward + raw1
        // Turn 2: prefix already in cache; forward ONLY the new user delta + new <a>
        Assert.True(stats[1].forwardedTokens < stats[1].promptTokens,
            $"Turn 2 should reuse cache: forwarded={stats[1].forwardedTokens}, prompt={stats[1].promptTokens}");
        Assert.True(stats[1].forwardedTokens > 0, "Turn 2 must forward at least the user delta");

        // The reset operation should appear EXACTLY ONCE (turn 1's initial state wasn't reset
        // because cache was empty - so we expect the path "forward (turn1), forward (raw1),
        // truncate, forward (suffix), forward (raw2)").
        int resetCount = 0;
        foreach (var op in model.Operations)
            if (op.op == "reset") resetCount++;
        Assert.True(resetCount <= 1, $"Expected at most one reset, saw {resetCount}");
    }

    [Fact]
    public void MultiTurn_ThinkingModel_RawTokensDifferFromContent_StillReusesPrefixPerfectly()
    {
        // Simulate a thinking model: raw output tokens include "thinking" framing that the
        // parser strips. The Content field reflects only the parsed answer. Without
        // raw-token splicing, re-rendering would tokenize Content -> different tokens ->
        // cache miss. With splicing, the cache is preserved.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        // Raw tokens are arbitrary - they don't correspond to any text the renderer would
        // produce. This is the key scenario.
        var rawWithThinking = new List<int> { 7001, 7002, 7003, 7004, 7005 };

        var stats = RunConversation(model, tokenizer, renderer, kvRenderer, kvCache,
            new[] { "Q1", "Q2" },
            new[] { rawWithThinking, new List<int> { 8001 } });

        Assert.True(stats[1].forwardedTokens < stats[1].promptTokens,
            $"Thinking-model turn 2 should reuse cache: forwarded={stats[1].forwardedTokens}, prompt={stats[1].promptTokens}");

        // Cache hit means the only operation between turn1's raw output and turn2 is a
        // truncate (back off by one for fresh logits) followed by a forward on the suffix.
        // No reset.
        int turn2ResetCount = 0;
        // Count operations after the second-to-last forward operation (which would have been
        // the raw1 injection). All subsequent ops belong to turn 2.
        int lastSeparator = -1;
        for (int i = 0; i < model.Operations.Count; i++)
        {
            if (model.Operations[i].op == "forward" && model.Operations[i].arg == rawWithThinking.Count)
                lastSeparator = i;
        }
        for (int i = lastSeparator + 1; i < model.Operations.Count; i++)
            if (model.Operations[i].op == "reset") turn2ResetCount++;

        Assert.Equal(0, turn2ResetCount);
    }

    [Fact]
    public void MultiTurn_RecurrentModel_PrefixExtensionReuse_NoResetWhenCacheIsPrefix()
    {
        // For recurrent models (SupportsTruncation = false), the only legal reuse pattern
        // is "cache is already a prefix of the new input". When the new turn appends to
        // the conversation without modifying earlier turns, this property holds.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel { SupportsTruncation = false };

        var raw = new List<int> { 1001, 1002 };

        var stats = RunConversation(model, tokenizer, renderer, kvRenderer, kvCache,
            new[] { "Q1", "Q2" },
            new[] { raw, new List<int> { 2001 } });

        Assert.True(stats[1].forwardedTokens < stats[1].promptTokens,
            "Recurrent model turn 2 should still reuse cache when prompt extends prior conversation");

        // Should be no second reset.
        int resetCount = 0;
        foreach (var op in model.Operations)
            if (op.op == "reset") resetCount++;
        Assert.True(resetCount <= 1, $"Recurrent model: at most one reset; saw {resetCount}");
    }

    [Fact]
    public void MultiTurn_RecurrentModel_NewSystemPromptResetsState()
    {
        // If the user starts a fresh conversation that doesn't share a prefix with the
        // cache, a recurrent model must reset.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel { SupportsTruncation = false };

        // Turn 1: user "Q1"
        RunConversation(model, tokenizer, renderer, kvRenderer, kvCache,
            new[] { "Q1" },
            new[] { new List<int> { 1001 } });

        int opsAfterTurn1 = model.Operations.Count;

        // Turn 2: completely different opening user message -> no usable common prefix.
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "DIFFERENT" },
        };
        var inputTokens = kvRenderer.RenderToTokens(tokenizer, null, history,
            architecture: "fake", addGenerationPrompt: true);
        ReusePlan plan = kvCache.PlanReuse(inputTokens, model.SupportsTruncation);

        Assert.Equal(ReusePlanKind.Reset, plan.Kind);
    }

    [Fact]
    public void MultiTurn_RawTokenSplicingIsCorrect_FullPrefixReuseAchieved()
    {
        // Detailed assertion: after turn N+1 is rendered, the leading tokens (up to the
        // length of the cache after turn N) MUST match the cached tokens exactly.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        var raw = new List<int> { 1001, 1002, 1003 };

        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
        };

        // Turn 1: render + forward
        var t1 = kvRenderer.RenderToTokens(tokenizer, null, history,
            architecture: "fake", addGenerationPrompt: true);
        ApplyResetThenForward(model, kvCache, t1);

        // Inject assistant raw output
        model.Forward(raw.ToArray());
        kvCache.RecordAppend(raw, new float[] { raw.Count });
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "ANSWER",
            RawOutputTokens = raw,
        });

        // Turn 2: render with new user message
        history.Add(new ChatMessage { Role = "user", Content = "Q2" });
        var t2 = kvRenderer.RenderToTokens(tokenizer, null, history,
            architecture: "fake", addGenerationPrompt: true);

        // Verify: the first kvCache.Count tokens of t2 must match the cache contents EXACTLY.
        Assert.True(t2.Count >= kvCache.Count, $"t2 ({t2.Count}) should be at least as long as cache ({kvCache.Count})");
        for (int i = 0; i < kvCache.Count; i++)
        {
            Assert.True(t2[i] == kvCache.Tokens[i],
                $"Position {i}: cache={kvCache.Tokens[i]}, t2={t2[i]}");
        }
    }

    [Fact]
    public void MultiTurn_ThreeTurns_AllRawTokensSplicedInOrder()
    {
        // Verify that with multiple cached assistant turns, raw tokens are spliced for ALL
        // of them, in the right order.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        var raw1 = new List<int> { 1001, 1002 };
        var raw2 = new List<int> { 2001, 2002, 2003 };

        var history = new List<ChatMessage>();

        // Turn 1
        history.Add(new ChatMessage { Role = "user", Content = "u1" });
        var t1 = kvRenderer.RenderToTokens(tokenizer, null, history, "fake", true);
        ApplyResetThenForward(model, kvCache, t1);
        model.Forward(raw1.ToArray());
        kvCache.RecordAppend(raw1, new float[] { raw1.Count });
        history.Add(new ChatMessage { Role = "assistant", Content = "ANS1", RawOutputTokens = raw1 });

        // Turn 2
        history.Add(new ChatMessage { Role = "user", Content = "u2" });
        var t2 = kvRenderer.RenderToTokens(tokenizer, null, history, "fake", true);
        ApplyPlanAgainst(model, kvCache, t2);
        model.Forward(raw2.ToArray());
        kvCache.RecordAppend(raw2, new float[] { raw2.Count });
        history.Add(new ChatMessage { Role = "assistant", Content = "ANS2", RawOutputTokens = raw2 });

        // Turn 3 render
        history.Add(new ChatMessage { Role = "user", Content = "u3" });
        var t3 = kvRenderer.RenderToTokens(tokenizer, null, history, "fake", true);

        // Cache must be exactly a prefix of t3.
        Assert.True(t3.Count >= kvCache.Count);
        for (int i = 0; i < kvCache.Count; i++)
            Assert.True(t3[i] == kvCache.Tokens[i],
                $"Turn 3 prefix mismatch at {i}: cache={kvCache.Tokens[i]}, t3={t3[i]}");

        // Both raw1 and raw2 should appear, in order
        int idx1 = FindSubsequence(t3, raw1);
        int idx2 = FindSubsequence(t3, raw2);
        Assert.True(idx1 >= 0, "raw1 should appear");
        Assert.True(idx2 >= 0, "raw2 should appear");
        Assert.True(idx1 < idx2, "raw1 must come before raw2");
    }

    [Fact]
    public void MultiTurn_UserEditsPreviousMessage_CacheTruncatesToCommonPrefix()
    {
        // Real-world scenario: the user clicks "edit" on a previous message and resubmits.
        // The cache should reuse what's still common and reset the rest.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        var raw1 = new List<int> { 1001 };

        // Initial conversation
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "ORIGINAL" },
        };
        var t1 = kvRenderer.RenderToTokens(tokenizer, null, history, "fake", true);
        ApplyResetThenForward(model, kvCache, t1);
        model.Forward(raw1.ToArray());
        kvCache.RecordAppend(raw1, new float[] { raw1.Count });
        history.Add(new ChatMessage { Role = "assistant", Content = "A1", RawOutputTokens = raw1 });

        // User goes back, edits to "EDITED" instead of "ORIGINAL", and resubmits without
        // any prior assistant turn (because it's a re-edit).
        var editedHistory = new List<ChatMessage>
        {
            new() { Role = "user", Content = "EDITED" },
        };
        var t2 = kvRenderer.RenderToTokens(tokenizer, null, editedHistory, "fake", true);

        // The plan should be Reset (different first user message -> common prefix is just <bos>).
        ReusePlan plan = kvCache.PlanReuse(t2, supportsTruncation: true);
        // Common prefix is at least the BOS (token 0). May also include "<u>" tokens. Since
        // both prompts start with the same template prefix, plan should be PartialReuse with
        // a small reused prefix.
        Assert.NotEqual(ReusePlanKind.ExactMatch, plan.Kind);
        Assert.True(plan.ReusedPrefixLength < kvCache.Count,
            $"Cache should be truncated below its current size: cache={kvCache.Count}, reused={plan.ReusedPrefixLength}");
    }

    [Fact]
    public void MultiTurn_FirstTokenLatency_DropsAfterFirstTurn()
    {
        // Smoke test for the cache-effectiveness invariant: even with very short user
        // messages and large simulated raw outputs, turn 2's forward count must be much
        // smaller than turn 1's.
        var tokenizer = new SimpleTokenizer();
        var renderer = new SimpleRenderer();
        var kvRenderer = new KVCachePromptRenderer(renderer);
        var kvCache = new KVCache();
        var model = new RecordingFakeModel();

        // Big simulated raw output to amplify the effect.
        var bigRaw = new List<int>();
        for (int i = 0; i < 200; i++)
            bigRaw.Add(10000 + i);

        var stats = RunConversation(model, tokenizer, renderer, kvRenderer, kvCache,
            new[] { "First user turn that establishes context", "Q" },
            new[] { bigRaw, new List<int> { 9001 } });

        // Turn 1's forward was the full prompt + 200 raw tokens.
        // Turn 2's forward must be tiny (just the new user delta).
        Assert.True(stats[1].forwardedTokens < 30,
            $"Turn 2 should forward very few tokens with big cache; got {stats[1].forwardedTokens}");
        Assert.True(stats[1].forwardedTokens < stats[0].forwardedTokens / 5,
            $"Turn 2 forward ({stats[1].forwardedTokens}) should be <<< turn 1 ({stats[0].forwardedTokens})");
    }

    private static void ApplyResetThenForward(RecordingFakeModel model, KVCache kvCache, List<int> tokens)
    {
        model.Reset();
        kvCache.Reset();
        var arr = tokens.ToArray();
        var logits = model.Forward(arr);
        kvCache.RecordAppend(arr, logits);
    }

    private static void ApplyPlanAgainst(RecordingFakeModel model, KVCache kvCache, List<int> tokens)
    {
        ReusePlan plan = kvCache.PlanReuse(tokens, model.SupportsTruncation);
        switch (plan.Kind)
        {
            case ReusePlanKind.ExactMatch:
                break;
            case ReusePlanKind.PartialReuse:
            {
                model.Truncate(plan.ReusedPrefixLength);
                kvCache.TruncateTo(plan.ReusedPrefixLength);
                var suffix = new int[plan.TokensToForward];
                for (int i = 0; i < plan.TokensToForward; i++)
                    suffix[i] = tokens[plan.ReusedPrefixLength + i];
                var logits = model.Forward(suffix);
                kvCache.RecordAppend(suffix, logits);
                break;
            }
            case ReusePlanKind.Reset:
            default:
            {
                model.Reset();
                kvCache.Reset();
                var arr = tokens.ToArray();
                var logits = model.Forward(arr);
                kvCache.RecordAppend(arr, logits);
                break;
            }
        }
    }

    private static int FindSubsequence(IReadOnlyList<int> haystack, IReadOnlyList<int> needle)
    {
        if (needle.Count == 0 || haystack.Count < needle.Count) return -1;
        for (int i = 0; i + needle.Count <= haystack.Count; i++)
        {
            bool match = true;
            for (int j = 0; j < needle.Count; j++)
                if (haystack[i + j] != needle[j]) { match = false; break; }
            if (match) return i;
        }
        return -1;
    }
}
