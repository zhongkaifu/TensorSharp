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
/// Tests for the <see cref="KVCachePromptRenderer"/> that splices raw output tokens
/// directly into a rendered chat prompt instead of re-tokenizing assistant content.
///
/// The tests use a controllable fake renderer + tokenizer so we can validate the
/// placeholder / splice machinery deterministically without loading any model.
/// </summary>
public class KVCachePromptRendererTests
{
    /// <summary>
    /// A trivial renderer that emits one segment per message in a tag-delimited form,
    /// adding optional generation-prompt suffix. The format is intentionally simple but
    /// parallel in spirit to real chat templates: each message gets a structural prefix
    /// and suffix, with the message content (or placeholder, if KVCachePromptRenderer
    /// wrote one) sitting in the middle.
    /// </summary>
    private sealed class FakeRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("<|bos|>");
            foreach (var m in messages)
            {
                sb.Append('<').Append(m.Role).Append('>');
                sb.Append(m.Content ?? "");
                sb.Append("</").Append(m.Role).Append('>');
                sb.Append('\n');
            }
            if (addGenerationPrompt)
                sb.Append("<assistant>");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Tokenizer that splits text into single characters and assigns each unique
    /// character to a token id. <see cref="KVCachePromptRenderer.PlaceholderSentinel"/>
    /// is encoded normally (just like any other character).
    /// </summary>
    private sealed class CharTokenizer : ITokenizer
    {
        private readonly Dictionary<char, int> _ids = new();
        private readonly List<string> _vocab = new();
        private const int Bos = 0;
        private const int Eos = 1;

        public CharTokenizer()
        {
            _vocab.Add("<bos>");
            _vocab.Add("<eos>");
        }

        public string[] Vocab => _vocab.ToArray();
        public int BosTokenId => Bos;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => _vocab.Count;

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var result = new List<int>();
            if (addSpecial)
                result.Add(Bos);
            if (text != null)
            {
                foreach (var ch in text)
                {
                    if (!_ids.TryGetValue(ch, out int id))
                    {
                        id = _vocab.Count;
                        _ids[ch] = id;
                        _vocab.Add(ch.ToString());
                    }
                    result.Add(id);
                }
            }
            return result;
        }

        public string Decode(List<int> ids)
        {
            var sb = new System.Text.StringBuilder();
            if (ids != null)
                foreach (var id in ids)
                    if (id != Bos && id != Eos && id < _vocab.Count)
                        sb.Append(_vocab[id]);
            return sb.ToString();
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            if (tokenId == Bos || tokenId == Eos) return;
            if (tokenId < _vocab.Count)
                buffer.AddRange(System.Text.Encoding.UTF8.GetBytes(_vocab[tokenId]));
        }

        public bool IsEos(int tokenId) => tokenId == Eos;
        public int LookupToken(string tokenStr) => _ids.TryGetValue(tokenStr.Length == 1 ? tokenStr[0] : '\0', out var id) ? id : -1;
    }

    [Fact]
    public void RenderToTokens_NoRawTokens_FallsThroughToInnerRender()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true);

        // Should match the result of rendering the text and tokenizing the whole thing.
        var expectedText = new FakeRenderer().Render(null, messages, addGenerationPrompt: true);
        var expected = tokenizer.Encode(expectedText, addSpecial: true);

        Assert.Equal(expected, tokens);
    }

    [Fact]
    public void RenderToTokens_AssistantWithRawTokens_SplicesRawTokensAndOmitsContent()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        // Pre-allocate token ids that are clearly outside the alphabet of normal text so we
        // can spot them in the output sequence.
        var rawTokens = new List<int> { 1001, 1002, 1003 };

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "DOES_NOT_MATTER", RawOutputTokens = rawTokens },
            new() { Role = "user", Content = "again" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true);

        // None of the assistant content's distinctive characters should appear in the output:
        // the renderer should have used a placeholder, and the splice should have replaced
        // that placeholder with rawTokens.
        Assert.DoesNotContain("DOES_NOT_MATTER", tokenizer.Decode(tokens));

        // The raw tokens must appear contiguously, in order.
        bool foundRawSequence = false;
        for (int i = 0; i + rawTokens.Count <= tokens.Count; i++)
        {
            bool match = true;
            for (int j = 0; j < rawTokens.Count; j++)
            {
                if (tokens[i + j] != rawTokens[j]) { match = false; break; }
            }
            if (match) { foundRawSequence = true; break; }
        }
        Assert.True(foundRawSequence, "Expected raw tokens to appear contiguously in the output.");
    }

    [Fact]
    public void RenderToTokens_MultipleAssistantTurnsWithRawTokens_SplicesAllInOrder()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        var raw1 = new List<int> { 1001, 1002 };
        var raw2 = new List<int> { 2001, 2002, 2003 };

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "DUMMY1", RawOutputTokens = raw1 },
            new() { Role = "user", Content = "Q2" },
            new() { Role = "assistant", Content = "DUMMY2", RawOutputTokens = raw2 },
            new() { Role = "user", Content = "Q3" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true);

        // Raw tokens must appear in order: raw1 before raw2.
        int idx1 = FindSubsequence(tokens, raw1);
        int idx2 = FindSubsequence(tokens, raw2);
        Assert.True(idx1 >= 0, "raw1 should appear in output");
        Assert.True(idx2 >= 0, "raw2 should appear in output");
        Assert.True(idx1 < idx2, "raw tokens must appear in turn order");
    }

    [Fact]
    public void RenderToTokens_AssistantWithEmptyRawTokens_RendersContentNormally()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "u1" },
            new() { Role = "assistant", Content = "VISIBLE", RawOutputTokens = new List<int>() },
            new() { Role = "user", Content = "u2" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true);

        // Empty raw tokens should be treated as "no raw tokens" - the content text appears.
        Assert.Contains("VISIBLE", tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_AssistantWithRawTokens_PreservesSurroundingStructure()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "U" },
            new() { Role = "assistant", Content = "X", RawOutputTokens = new List<int> { 1001, 1002 } },
            new() { Role = "user", Content = "V" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true);

        // The structural framing characters from FakeRenderer should still be present.
        var decoded = tokenizer.Decode(tokens);
        Assert.Contains("<user>", decoded);
        Assert.Contains("</user>", decoded);
        Assert.Contains("<assistant>", decoded);
        Assert.Contains("</assistant>", decoded);
        Assert.Contains("U", decoded);
        Assert.Contains("V", decoded);
    }

    [Fact]
    public void RenderToTokens_PrefixMatchesAcrossTurns_EnablesKVCacheReuse()
    {
        // The CRITICAL invariant: turn N+1's rendered token sequence must start with
        // turn N's rendered token sequence + the raw output tokens of turn N + the
        // delta produced by adding a new user message.
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        var raw1 = new List<int> { 1001, 1002, 1003 };

        // Turn 1 prompt
        var turn1Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "First question" },
        };
        var turn1Tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null,
            turn1Messages, architecture: "fake", addGenerationPrompt: true);

        // After turn 1 the cache contains turn1Tokens + raw1 (the model generated raw1).
        var cachedAfterTurn1 = new List<int>(turn1Tokens);
        cachedAfterTurn1.AddRange(raw1);

        // Turn 2 prompt
        var turn2Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "First question" },
            new() { Role = "assistant", Content = "DOES_NOT_MATTER", RawOutputTokens = raw1 },
            new() { Role = "user", Content = "Follow-up" },
        };
        var turn2Tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null,
            turn2Messages, architecture: "fake", addGenerationPrompt: true);

        // turn2Tokens MUST start with cachedAfterTurn1 - that's what makes KV cache reuse possible.
        Assert.True(turn2Tokens.Count > cachedAfterTurn1.Count,
            $"turn2 ({turn2Tokens.Count} tokens) must be longer than cached ({cachedAfterTurn1.Count})");
        for (int i = 0; i < cachedAfterTurn1.Count; i++)
        {
            Assert.True(turn2Tokens[i] == cachedAfterTurn1[i],
                $"Token {i} mismatch: cache has {cachedAfterTurn1[i]} but turn2 rendered {turn2Tokens[i]}");
        }
    }

    [Fact]
    public void RenderToTokens_ThinkingTokensInRawOutput_AreCachedAndReused()
    {
        // Simulates a thinking model: the assistant's "raw" generation includes special
        // <think>...</think> framing tokens that the output parser would normally STRIP
        // out of ChatMessage.Content.
        //
        // Without the raw-token splicing, re-rendering the conversation for turn N+1 would
        // produce tokens for the (stripped) content - which would NOT match what's in the
        // cache. This test verifies that we splice the raw tokens (with thinking framing)
        // back into the rendered prompt.
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        // Raw tokens contain "thinking" segments that the output parser stripped.
        var rawWithThinking = new List<int> { 5001, 5002, 5003, 5004, 5005 };
        // Content has only the FINAL stripped answer (after parsing) - very different
        // from the raw tokens.
        var strippedContent = "Answer";

        var turn2Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q" },
            new() { Role = "assistant", Content = strippedContent, RawOutputTokens = rawWithThinking },
            new() { Role = "user", Content = "Q2" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, turn2Messages,
            architecture: "fake", addGenerationPrompt: true);

        // Raw tokens (with thinking) must appear in the output.
        Assert.True(FindSubsequence(tokens, rawWithThinking) >= 0,
            "Raw thinking tokens should be spliced into the prompt");
        // The stripped content text should NOT appear (because we used the raw tokens instead).
        Assert.DoesNotContain("Answer", tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_NullMessages_Throws()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        Assert.Throws<ArgumentNullException>(() =>
            renderer.RenderToTokens(tokenizer, chatTemplate: null, messages: null,
                architecture: "fake", addGenerationPrompt: true));
    }

    [Fact]
    public void RenderToTokens_NullTokenizer_Throws()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        Assert.Throws<ArgumentNullException>(() =>
            renderer.RenderToTokens(tokenizer: null, chatTemplate: null,
                messages: new List<ChatMessage>(), architecture: "fake",
                addGenerationPrompt: true));
    }

    [Fact]
    public void Constructor_NullInnerRenderer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => new KVCachePromptRenderer(null));
    }

    [Fact]
    public void RenderToTokens_PlaceholderSentinelIsUnique_TokensAreStable()
    {
        // Verify that two placeholder strings (with different indices) are distinct.
        // Otherwise the splicer could not tell them apart in case the chat template
        // duplicated content.
        var p1 = MakePlaceholder(0);
        var p2 = MakePlaceholder(1);
        Assert.NotEqual(p1, p2);
        Assert.StartsWith(KVCachePromptRenderer.PlaceholderSentinel.ToString(), p1);
        Assert.EndsWith(KVCachePromptRenderer.PlaceholderSentinel.ToString(), p1);
    }

    /// <summary>
    /// Renderer that mimics the production "Jinja + TrimEnd" code path: produces
    /// the chat-template text and then strips trailing whitespace at the very end.
    /// </summary>
    private sealed class TrimEndRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder();
            sb.Append("<|bos|>");
            foreach (var m in messages)
            {
                sb.Append('<').Append(m.Role).Append(">\n");
                sb.Append(m.Content ?? "");
                sb.Append("\n</").Append(m.Role).Append(">\n");
            }
            if (addGenerationPrompt)
                sb.Append("<assistant>\n");
            return sb.ToString().TrimEnd();
        }
    }

    [Fact]
    public void RenderToTokens_TrimEndRenderer_TurnNRendersConsistentlyWithTurn1()
    {
        // CRITICAL invariant: even with a renderer that applies TrimEnd to its output,
        // the turn-N rendering with raw token splicing must produce a token sequence
        // whose prefix matches what turn-1 produced for the same prompt prefix. Without
        // the renderer-agnostic trim handling in KVCachePromptRenderer this would fail
        // because the interior generation-prompt suffix's trailing whitespace is preserved
        // by the renderer (only the FINAL trailing whitespace is stripped) - and this
        // extra whitespace token at the boundary diverges from what was in the cache.
        var renderer = new KVCachePromptRenderer(new TrimEndRenderer());
        var tokenizer = new CharTokenizer();

        var raw = new List<int> { 1001, 1002, 1003 };

        // Turn 1: just user message + generation prompt.
        var turn1Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
        };
        var turn1Tokens = renderer.RenderToTokens(tokenizer, null, turn1Messages, "fake", true);

        // Cache after turn 1 = turn1Tokens + raw output tokens.
        var cachedTokens = new List<int>(turn1Tokens);
        cachedTokens.AddRange(raw);

        // Turn 2 includes the previous assistant turn (with raw output tokens) + new user.
        var turn2Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "DONT_RENDER_ME", RawOutputTokens = raw },
            new() { Role = "user", Content = "Q2" },
        };
        var turn2Tokens = renderer.RenderToTokens(tokenizer, null, turn2Messages, "fake", true);

        // The cached tokens MUST be a strict prefix of turn 2's tokens.
        Assert.True(turn2Tokens.Count >= cachedTokens.Count,
            $"turn2 ({turn2Tokens.Count}) must be longer than cached ({cachedTokens.Count})");
        for (int i = 0; i < cachedTokens.Count; i++)
        {
            Assert.True(turn2Tokens[i] == cachedTokens[i],
                $"Position {i}: cache={cachedTokens[i]}, turn2={turn2Tokens[i]} (TrimEnd renderer must mirror trim at interior placeholder boundary)");
        }
    }

    [Fact]
    public void RenderToTokens_NonTrimEndRenderer_TurnNRendersConsistentlyWithTurn1()
    {
        // Same invariant for renderers that DON'T apply TrimEnd. Here turn 1 keeps its
        // trailing whitespace; turn 2's interior boundary also keeps its whitespace; both
        // tokenize the same way, so the prefix matches without any trim-mirroring.
        var renderer = new KVCachePromptRenderer(new FakeRenderer()); // FakeRenderer doesn't TrimEnd
        var tokenizer = new CharTokenizer();

        var raw = new List<int> { 1001, 1002, 1003 };

        var turn1Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
        };
        var turn1Tokens = renderer.RenderToTokens(tokenizer, null, turn1Messages, "fake", true);

        var cachedTokens = new List<int>(turn1Tokens);
        cachedTokens.AddRange(raw);

        var turn2Messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "DONT_RENDER_ME", RawOutputTokens = raw },
            new() { Role = "user", Content = "Q2" },
        };
        var turn2Tokens = renderer.RenderToTokens(tokenizer, null, turn2Messages, "fake", true);

        Assert.True(turn2Tokens.Count >= cachedTokens.Count);
        for (int i = 0; i < cachedTokens.Count; i++)
            Assert.True(turn2Tokens[i] == cachedTokens[i],
                $"Position {i}: cache={cachedTokens[i]}, turn2={turn2Tokens[i]} (no-TrimEnd renderer must NOT alter interior boundary)");
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Gemma4ThinkingDisabled_ReturnsChannelBlock()
    {
        Assert.Equal("<|channel>thought\n<channel|>",
            KVCachePromptRenderer.GetAssistantGenerationSuffix("gemma4", enableThinking: false));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Gemma4ThinkingEnabled_ReturnsEmpty()
    {
        Assert.Equal(string.Empty,
            KVCachePromptRenderer.GetAssistantGenerationSuffix("gemma4", enableThinking: true));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Qwen35FamilyThinkingEnabled_ReturnsThinkOpen()
    {
        // Qwen 3.5 family with thinking ENABLED uses the Jinja template path which
        // emits `<think>\n` after the assistant role marker. Past-assistant rendering
        // in the same template does NOT re-emit `<think>...</think>` (only the latest
        // query's assistant turn gets the full framing), so we must inject it ourselves
        // for cached-assistant placeholders to match the KV cache.
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35", true));
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35moe", true));
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3next", true));
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3vl", true));
        Assert.Equal("<think>\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3vlmoe", true));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Qwen35FamilyThinkingDisabled_ReturnsEmpty()
    {
        // Thinking-disabled goes through the hardcoded renderer which already emits
        // `<think>\n\n</think>\n\n` for past assistant messages. No injection needed.
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35moe", false));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_OtherArchitectures_ReturnEmpty()
    {
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen3", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("gemma3", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("mistral3", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("gptoss", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("nemotron_h", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix(null, false));
    }

    private static string MakePlaceholder(int index)
    {
        // Mirrors KVCachePromptRenderer.MakePlaceholder so we don't have to expose it
        // publicly. We assert structural invariants instead of behavior.
        return $"{KVCachePromptRenderer.PlaceholderSentinel}R{index:D4}{KVCachePromptRenderer.PlaceholderSentinel}";
    }

    /// <summary>Find the first index of a contiguous subsequence in <paramref name="haystack"/>.</summary>
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
