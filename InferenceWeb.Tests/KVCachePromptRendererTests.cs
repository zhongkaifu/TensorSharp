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

    /// <summary>
    /// Models the relevant Gemma 4 template asymmetry. Ordinary generations end in
    /// an explicitly restored newline, but after a tool result the template continues
    /// the existing model turn and therefore ends in a non-whitespace control marker.
    /// The framing of an older assistant message itself does not change.
    /// </summary>
    private sealed class ToolContinuationTailRenderer : IPromptRenderer
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
                sb.Append("</").Append(m.Role).Append(">\n");
            }

            if (addGenerationPrompt)
            {
                if (messages.Count > 0 && messages[^1].Role == "tool")
                    sb.Append("<tool-continuation>");
                else
                    sb.Append("<assistant>\n");
            }
            return sb.ToString();
        }
    }

    /// <summary>
    /// Reconstructs an assistant tool-call round byte-for-byte, but puts a space at
    /// the generation boundary. The recorded live boundary is a newline, so merely
    /// finding the raw token run in this render is not sufficient for KV reuse.
    /// </summary>
    private sealed class BoundaryMismatchToolRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder("<|bos|>");
            foreach (ChatMessage message in messages)
            {
                sb.Append('<').Append(message.Role).Append("> ");
                sb.Append(message.Content ?? string.Empty);
                sb.Append("</").Append(message.Role).Append('>');
            }
            if (addGenerationPrompt)
                sb.Append("<assistant> ");
            return sb.ToString();
        }
    }

    /// <summary>
    /// Models a template that folds a tool result into the preceding assistant turn
    /// only while structured ToolCalls is present. Raw-token splicing clears that field,
    /// so the candidate optimized render drops the result. Its user header deliberately
    /// contains the word "user" to exercise the collision that substring guards miss.
    /// </summary>
    private sealed class ToolResultDroppingOnSpliceRenderer : IPromptRenderer
    {
        public string Render(string? template, List<ChatMessage> messages,
            bool addGenerationPrompt = true, string? architecture = null,
            List<ToolFunction>? tools = null, bool enableThinking = false)
        {
            var sb = new System.Text.StringBuilder();
            ChatMessage? previous = null;
            foreach (ChatMessage message in messages)
            {
                if (message.Role == "user")
                {
                    sb.Append("<|turn>user\n").Append(message.Content).Append("<turn|>");
                }
                else if (message.Role == "assistant")
                {
                    sb.Append("<|turn>model\n").Append(message.Content).Append("<turn|>");
                }
                else if (message.Role == "tool"
                    && previous?.ToolCalls is { Count: > 0 })
                {
                    sb.Append("<tool>").Append(message.Content).Append("</tool>");
                }
                previous = message;
            }
            if (addGenerationPrompt)
                sb.Append("<|turn>model\n");
            return sb.ToString();
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
    public void RenderToTokens_ToolContinuationTail_DoesNotRenormalizeOlderAssistantBoundary()
    {
        // Regression for the 2-D-game/code-exec incident. The ordinary follow-up
        // correctly cached `<assistant>\n` + the prior raw output. Adding a tool result
        // made the final rendered character non-whitespace, and the old global heuristic
        // then removed the newline before EVERY placeholder. The live-cache LCP stopped
        // at that first assistant boundary thousands of tokens before the new suffix.
        var renderer = new KVCachePromptRenderer(new ToolContinuationTailRenderer());
        var tokenizer = new CharTokenizer();
        var firstRaw = tokenizer.Encode("RAW_FIRST_ANSWER", addSpecial: false);

        var firstPrompt = renderer.RenderToTokens(
            tokenizer, null,
            new List<ChatMessage> { new() { Role = "user", Content = "Q1" } },
            "gemma4", addGenerationPrompt: true, out _, out string firstBoundary,
            enableThinking: true);
        var liveBeforeTool = new List<int>(firstPrompt);
        liveBeforeTool.AddRange(firstRaw);

        var toolRoundPrompt = renderer.RenderToTokens(
            tokenizer, null,
            new List<ChatMessage>
            {
                new() { Role = "user", Content = "Q1" },
                new()
                {
                    Role = "assistant",
                    Content = "parsed",
                    RawOutputTokens = firstRaw,
                    RawPromptTrailingWhitespace = firstBoundary,
                },
                new() { Role = "user", Content = "Q2" },
            },
            "gemma4", addGenerationPrompt: true, out _, out string toolBoundary,
            enableThinking: true);

        Assert.True(toolRoundPrompt.Take(liveBeforeTool.Count).SequenceEqual(liveBeforeTool),
            "The ordinary follow-up must reproduce the first turn's live prefix.");

        var toolRaw = tokenizer.Encode("RAW_TOOL_CALL", addSpecial: false);
        var afterToolResult = renderer.RenderToTokens(
            tokenizer, null,
            new List<ChatMessage>
            {
                new() { Role = "user", Content = "Q1" },
                new()
                {
                    Role = "assistant",
                    Content = "parsed",
                    RawOutputTokens = firstRaw,
                    RawPromptTrailingWhitespace = firstBoundary,
                },
                new() { Role = "user", Content = "Q2" },
                new()
                {
                    Role = "assistant",
                    Content = "tool call",
                    Thinking = "use tool",
                    ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
                    RawOutputTokens = toolRaw,
                    RawPromptTrailingWhitespace = toolBoundary,
                },
                new() { Role = "tool", Content = "ok" },
            },
            "gemma4", addGenerationPrompt: true, enableThinking: true);

        Assert.True(afterToolResult.Take(liveBeforeTool.Count).SequenceEqual(liveBeforeTool),
            "A tool-continuation tail must not change an older assistant boundary.");
    }

    [Fact]
    public void RenderToTokens_AdaptiveToolReconstruction_RejectsWrongExactBoundaryWhitespace()
    {
        var tokenizer = new CharTokenizer();
        const string rawText = "RAW_GENERATED_TOOL_CALL";
        List<int> rawTokens = tokenizer.Encode(rawText, addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "find it" },
            new()
            {
                Role = "assistant",
                // The unspliced render reproduces these exact token IDs, but the
                // renderer places a space before them. The live cache recorded '\n'.
                Content = rawText,
                RawOutputTokens = rawTokens,
                RawPromptTrailingWhitespace = "\n",
                ToolCalls = new List<ToolCall> { new() { Name = "search" } },
            },
            new() { Role = "tool", Content = "UNIQUE_RESULT" },
        };

        List<int> tokens = new KVCachePromptRenderer(new BoundaryMismatchToolRenderer())
            .RenderToTokens(
                tokenizer, null, messages, "gemma4",
                addGenerationPrompt: true, enableThinking: true);

        int rawStart = FindSubsequence(tokens, rawTokens);
        Assert.True(rawStart > 0, "The selected prompt must contain the generated tool-call run.");
        int newline = Assert.Single(tokenizer.Encode("\n", addSpecial: false));
        Assert.Equal(newline, tokens[rawStart - 1]);
        Assert.Contains("UNIQUE_RESULT", tokenizer.Decode(tokens));
    }

    [Fact]
    public void ConditionalToolSplice_DoesNotMistakeNextTurnHeaderForDroppedResult()
    {
        var tokenizer = new CharTokenizer();
        List<int> rawTokens = tokenizer.Encode("RAW_GENERATED_TOOL_CALL", addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "run it" },
            new()
            {
                Role = "assistant",
                Content = "structured call that differs from the generated tokens",
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
                RawOutputTokens = rawTokens,
            },
            // This result is absent from the candidate spliced render, but the same
            // word occurs in Gemma's following role header: <|turn>user.
            new() { Role = "tool", Content = "user" },
            new() { Role = "user", Content = "continue" },
        };

        List<int> tokens = new KVCachePromptRenderer(new ToolResultDroppingOnSpliceRenderer())
            .RenderToTokens(
                tokenizer, null, messages, "gemma4",
                addGenerationPrompt: true, enableThinking: true);

        string rendered = tokenizer.Decode(tokens);
        Assert.Contains("<tool>user</tool>", rendered);
        Assert.DoesNotContain("RAW_GENERATED_TOOL_CALL", rendered);
        Assert.DoesNotContain(KVCachePromptRenderer.ToolResultProofSentinel, rendered);
    }

    [Fact]
    public void ConditionalToolSplice_StripsProofWithoutChangingUnicodeResultText()
    {
        var tokenizer = new CharTokenizer();
        List<int> rawTokens = tokenizer.Encode("RAW_TOOL_CALL", addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "run" },
            new()
            {
                Role = "assistant",
                Content = "different structured rendering",
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
                RawOutputTokens = rawTokens,
            },
            new() { Role = "tool", Content = "  😀 ok  " },
        };

        List<int> tokens = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null, messages, "gemma4",
            addGenerationPrompt: true, enableThinking: true);

        string rendered = tokenizer.Decode(tokens);
        Assert.Contains("<tool>  😀 ok  </tool>", rendered);
        Assert.Contains("RAW_TOOL_CALL", rendered);
        Assert.DoesNotContain(KVCachePromptRenderer.ToolResultProofSentinel, rendered);
    }

    [Theory]
    [InlineData("")]
    [InlineData("   \n")]
    public void ConditionalToolSplice_EmptyResultFailsClosed(string toolResult)
    {
        var tokenizer = new CharTokenizer();
        List<int> rawTokens = tokenizer.Encode("RAW_TOOL_CALL", addSpecial: false);
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "run" },
            new()
            {
                Role = "assistant",
                Content = "STRUCTURED_TOOL_CALL",
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
                RawOutputTokens = rawTokens,
            },
            new() { Role = "tool", Content = toolResult },
        };

        List<int> tokens = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null, messages, "gemma4",
            addGenerationPrompt: true, enableThinking: true);

        string rendered = tokenizer.Decode(tokens);
        Assert.Contains("STRUCTURED_TOOL_CALL", rendered);
        Assert.DoesNotContain("RAW_TOOL_CALL", rendered);
        Assert.DoesNotContain(KVCachePromptRenderer.ToolResultProofSentinel, rendered);
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Gemma4ThinkingDisabled_ReturnsEmpty()
    {
        Assert.Equal(string.Empty,
            KVCachePromptRenderer.GetAssistantGenerationSuffix("gemma4", enableThinking: false));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_Gemma4ThinkingEnabled_ReturnsEmpty()
    {
        Assert.Equal(string.Empty,
            KVCachePromptRenderer.GetAssistantGenerationSuffix("gemma4", enableThinking: true));
    }

    [Theory]
    [InlineData("<|turn>model\n", "")]
    [InlineData("<|turn>model\n<|channel>thought\n<channel|>", "<|channel>thought\n<channel|>")]
    [InlineData("<tool_response|><|channel>thought\n", "<|channel>thought\n")]
    [InlineData("<tool_response|>", "")]
    public void Gemma4_RecordsActualPublisherGenerationSuffix(string tail, string suffix)
    {
        var tokenizer = new CharTokenizer();
        Assert.Equal(suffix, TensorSharp.Server.ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, tokenizer.Encode("Earlier conversation\n" + tail), "gemma4", false));
    }

    [Theory]
    [InlineData("")]
    [InlineData("<|channel>thought\n<channel|>")]
    public void Gemma4_ReplaysRecordedSuffixWithoutAddingADefault(string suffix)
    {
        var tokenizer = new CharTokenizer();
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var raw = tokenizer.Encode("Answer.", addSpecial: false);
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
            new() { Role = "assistant", Content = "Answer.", RawOutputTokens = raw,
                RawGenerationSuffix = suffix, RawPromptTrailingWhitespace = "" },
            new() { Role = "user", Content = "Again" },
        };
        string text = tokenizer.Decode(renderer.RenderToTokens(
            tokenizer, null, history, "gemma4", true));
        Assert.Contains("<assistant>" + suffix + "Answer.</assistant>", text);
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
    public void GetAssistantGenerationSuffix_Qwen35FamilyThinkingDisabled_ReturnsClosedEmptyBlock()
    {
        // Thinking-disabled renders through the same GGUF template as thinking-enabled
        // now, and that template frames the generation prompt with the CLOSED empty
        // block — which the cache therefore holds in front of every past answer and
        // which the template does not re-emit for turns before the last user message.
        Assert.Equal("<think>\n\n</think>\n\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35", false));
        Assert.Equal("<think>\n\n</think>\n\n", KVCachePromptRenderer.GetAssistantGenerationSuffix("qwen35moe", false));
    }

    [Fact]
    public void RenderToTokens_UsesEachTurnsRecordedGenerationSuffix_NotTheCurrentRequests()
    {
        // Turn 1 was answered with thinking ON (its prompt ended `<think>\n`), turn 2
        // is asked with thinking OFF. The cache holds `<think>\n` before turn 1's raw
        // tokens, so that is what must be put back — the current request's closed
        // block would diverge from the cache at the first answer.
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var raw = new List<int> { 1001, 1002 };
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new()
            {
                Role = "assistant", Content = "A1", RawOutputTokens = raw,
                RawGenerationSuffix = "<think>\n", RawPromptTrailingWhitespace = "\n",
            },
            new() { Role = "user", Content = "Q2" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "qwen35", addGenerationPrompt: true, enableThinking: false);

        int at = FindSubsequence(tokens, raw);
        Assert.True(at > 0, "raw tokens should appear in the output");
        string before = tokenizer.Decode(tokens.GetRange(0, at));
        Assert.EndsWith("<assistant><think>\n", before);
        Assert.DoesNotContain("</think>\n\n<think>", before);

        // Without a recorded suffix the current request's framing is the fallback.
        messages[1].RawGenerationSuffix = null;
        messages[1].RawPromptTrailingWhitespace = "\n\n";
        tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "qwen35", addGenerationPrompt: true, enableThinking: false);
        at = FindSubsequence(tokens, raw);
        Assert.EndsWith("<assistant><think>\n\n</think>\n\n", tokenizer.Decode(tokens.GetRange(0, at)));
    }

    [Fact]
    public void GetAssistantGenerationSuffix_OtherArchitectures_ReturnEmpty()
    {
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("mistral3", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("gptoss", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("nemotron_h", false));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix(null, false));
    }

    // ---- Empty <think> stripping (multi-turn prefix reuse) -------------------
    //
    // Repro: TensorSharp.Server + Qwen3.8-27B on ggml_metal, two turns, thinking on.
    // Turn 2 reported kvReusePercent=0. The live cache held
    //   ... `assistant` `\n` `<think>` `\n` <generated>
    // but the re-render produced
    //   ... `assistant` `\n` `<think>` `\n\n` `</think>` `\n\n` `<think>` `\n` <generated>
    // because the chat template emits an EMPTY think block for a PAST assistant turn
    // on top of the `<think>\n` this renderer injects. Four extra tokens at the first
    // assistant boundary zeroed an all-or-nothing prefix match.

    private static string P(int i) => MakePlaceholder(i);

    [Fact]
    public void StripEmptyThinkBlock_RemovesTemplateEmptyBlockAndItsTrailingWhitespace()
    {
        string rendered = "<|im_start|>assistant\n<think>\n\n</think>\n\n" + P(0) + "<|im_end|>\n";
        string stripped = KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(rendered);

        // What remains must be exactly the framing the cache saw, so that the
        // subsequent "<think>\n" injection reproduces it byte for byte.
        Assert.Equal("<|im_start|>assistant\n" + P(0) + "<|im_end|>\n", stripped);
    }

    [Fact]
    public void StripEmptyThinkBlock_PreservesAdjacentPrefixBreakpoint()
    {
        string breakpoint = KVCachePromptRenderer.MakeBreakpoint(0);
        string rendered =
            "<|im_start|>assistant\n<think>\n\n</think>\n\n"
            + breakpoint + P(0) + "<|im_end|>\n";

        string stripped = KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(rendered);

        Assert.Equal(
            "<|im_start|>assistant\n" + breakpoint + P(0) + "<|im_end|>\n",
            stripped);
    }

    [Fact]
    public void StripTemplateAssistantHeaders_PreservesAdjacentPrefixBreakpoint()
    {
        const string anchor = "<|start|>assistant";
        string breakpoint = KVCachePromptRenderer.MakeBreakpoint(3);
        string rendered =
            "prefix" + anchor + " to=user<|message|>" + breakpoint + P(0) + "suffix";

        string stripped = InvokePrivateStringMethod(
            "StripTemplateAssistantHeaders", rendered, anchor);

        Assert.Equal("prefix" + anchor + breakpoint + P(0) + "suffix", stripped);
    }

    [Fact]
    public void NormalizeWhitespace_CrossesFiveDigitBreakpointMarker()
    {
        string breakpoint = KVCachePromptRenderer.MakeBreakpoint(10_000);
        string rendered = "prefix \t" + breakpoint + P(0) + "suffix";

        string normalized = InvokePrivateStringMethod(
            "NormalizeWhitespaceBeforeEachPlaceholder",
            rendered,
            new string?[] { "\n" },
            false);

        Assert.Equal("prefix\n" + breakpoint + P(0) + "suffix", normalized);
    }

    [Fact]
    public void StripEmptyThinkBlock_LeavesNonEmptyThinkBlockAlone()
    {
        // A client that replays real prior reasoning must keep it.
        string rendered = "<|im_start|>assistant\n<think>\nprior reasoning\n</think>\n\n" + P(0) + "<|im_end|>\n";
        Assert.Equal(rendered, KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(rendered));
    }

    [Fact]
    public void StripEmptyThinkBlock_OnlyTouchesBlocksAdjacentToAPlaceholder()
    {
        // An empty block that is NOT immediately before a spliced turn (e.g. the
        // generation prompt for the CURRENT turn) is part of what the model will be
        // fed and must survive.
        string rendered = "<|im_start|>assistant\n" + P(0) + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        Assert.Equal(rendered, KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(rendered));
    }

    [Fact]
    public void StripEmptyThinkBlock_HandlesEveryPlaceholderInAMultiTurnRender()
    {
        string rendered =
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + P(0) + "<|im_end|>\n" +
            "<|im_start|>user\nagain<|im_end|>\n" +
            "<|im_start|>assistant\n<think>\n\n</think>\n\n" + P(1) + "<|im_end|>\n";
        string stripped = KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(rendered);

        Assert.Equal(
            "<|im_start|>assistant\n" + P(0) + "<|im_end|>\n" +
            "<|im_start|>user\nagain<|im_end|>\n" +
            "<|im_start|>assistant\n" + P(1) + "<|im_end|>\n",
            stripped);
    }

    [Fact]
    public void StripEmptyThinkBlock_NoPlaceholdersOrNoThinkTags_IsIdentity()
    {
        Assert.Equal("plain text", KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders("plain text"));
        Assert.Equal("<think>\n\n</think>\n\nno placeholder",
            KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders("<think>\n\n</think>\n\nno placeholder"));
        string noThink = "<|im_start|>assistant\n" + P(0) + "<|im_end|>";
        Assert.Equal(noThink, KVCachePromptRenderer.StripEmptyThinkBlockBeforePlaceholders(noThink));
    }

    // ---------------------------------------------------------------------
    // Explicit prompt-cache breakpoints (cache_control / prompt_cache_breakpoint).
    //
    // The contract these tests pin down: a marker changes NOTHING about the
    // token sequence handed to the model, and reports the token offset where
    // the marked prefix ends. A breakpoint index is only useful if
    // tokens[0..index] is exactly the prefix the client marked, so most of
    // these assert that identity directly.
    // ---------------------------------------------------------------------

    [Fact]
    public void RenderToTokens_NoCacheControl_ReportsNoBreakpoints()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys" },
            new() { Role = "user", Content = "Hi" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, chatTemplate: null, messages,
            architecture: "fake", addGenerationPrompt: true, out var breakpoints);

        Assert.Null(breakpoints);

        var expectedText = new FakeRenderer().Render(null, messages, addGenerationPrompt: true);
        Assert.Equal(tokenizer.Encode(expectedText, addSpecial: true), tokens);
    }

    [Fact]
    public void RenderToTokens_MessageCacheControl_LeavesTokenStreamIdentical()
    {
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys", CacheControl = new CacheControlMarker() },
            new() { Role = "user", Content = "Hi" },
        };
        var unmarked = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys" },
            new() { Role = "user", Content = "Hi" },
        };

        var marked = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null, messages, "fake", addGenerationPrompt: true, out var breakpoints);
        var plain = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null, unmarked, "fake", addGenerationPrompt: true, out _);

        // The marker must be invisible to the model: same tokens either way.
        Assert.Equal(plain, marked);

        // And no sentinel may survive into the prompt.
        Assert.DoesNotContain(KVCachePromptRenderer.BreakpointSentinel, tokenizer.Decode(marked));

        var bp = Assert.Single(breakpoints!);
        Assert.InRange(bp, 1, marked.Count);
    }

    [Fact]
    public void RenderToTokens_MessageCacheControl_BreakpointEndsTheMarkedPrefix()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys", CacheControl = new CacheControlMarker() },
            new() { Role = "user", Content = "Hi" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints);

        int bp = Assert.Single(breakpoints!);

        // The marker sits at the end of the system message's content, so the
        // prefix it closes is everything the fake template emits up to and
        // including "sys" - and not the "</system>" that follows it.
        Assert.Equal("<|bos|><system>sys", tokenizer.Decode(tokens.GetRange(0, bp)));
    }

    [Fact]
    public void RenderToTokens_MultipleCacheControls_AreReportedInAscendingOrder()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys", CacheControl = new CacheControlMarker() },
            new() { Role = "user", Content = "docs", CacheControl = new CacheControlMarker() },
            new() { Role = "user", Content = "question" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints);

        Assert.Equal(2, breakpoints!.Count);
        Assert.True(breakpoints[0] < breakpoints[1],
            $"Breakpoints must be in render order, got [{breakpoints[0]}, {breakpoints[1]}].");

        Assert.Equal("<|bos|><system>sys", tokenizer.Decode(tokens.GetRange(0, breakpoints[0])));
        Assert.Equal("<|bos|><system>sys</system>\n<user>docs",
            tokenizer.Decode(tokens.GetRange(0, breakpoints[1])));
    }

    [Fact]
    public void RenderToTokens_CacheControlOnMessageWithRawTokens_BreakpointFollowsSplicedTokens()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var rawTokens = new List<int> { 1001, 1002, 1003, 1004 };

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi" },
            new()
            {
                Role = "assistant",
                Content = "IGNORED",
                RawOutputTokens = rawTokens,
                CacheControl = new CacheControlMarker(),
            },
            new() { Role = "user", Content = "again" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints);

        int bp = Assert.Single(breakpoints!);

        // The breakpoint must land AFTER the spliced raw tokens, not at the
        // (much shorter) placeholder's position: the index is in the final
        // array's coordinate space.
        int rawStart = FindSubsequence(tokens, rawTokens);
        Assert.True(rawStart >= 0, "Expected the raw tokens to be spliced into the output.");
        Assert.Equal(rawStart + rawTokens.Count, bp);
    }

    [Fact]
    public void RenderToTokens_ToolCacheControl_MarksPrefixAheadOfTheFirstMessageContent()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var tools = new List<ToolFunction>
        {
            new() { Name = "search", Description = "d", CacheControl = new CacheControlMarker() },
        };
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys" },
            new() { Role = "user", Content = "Hi" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints, tools: tools);

        int bp = Assert.Single(breakpoints!);

        // The tool block has nowhere else to anchor, so the breakpoint goes
        // immediately before the first message's content. See the comment on
        // needsToolsMarker in KVCachePromptRenderer for why this only lines up
        // with the end of the tool block on templates that emit tools first.
        Assert.Equal("<|bos|><system>", tokenizer.Decode(tokens.GetRange(0, bp)));
        Assert.DoesNotContain(KVCachePromptRenderer.BreakpointSentinel, tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_ToolAndMessageCacheControl_ReportsBothInOrder()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var tools = new List<ToolFunction>
        {
            new() { Name = "search", Description = "d", CacheControl = new CacheControlMarker() },
        };
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "sys", CacheControl = new CacheControlMarker() },
            new() { Role = "user", Content = "Hi" },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints, tools: tools);

        Assert.Equal(2, breakpoints!.Count);
        Assert.Equal("<|bos|><system>", tokenizer.Decode(tokens.GetRange(0, breakpoints[0])));
        Assert.Equal("<|bos|><system>sys", tokenizer.Decode(tokens.GetRange(0, breakpoints[1])));
    }

    [Fact]
    public void RenderToTokens_CacheControlOnLastMessage_DoesNotSwallowGenerationPrompt()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi", CacheControl = new CacheControlMarker() },
        };

        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "fake",
            addGenerationPrompt: true, out var breakpoints);

        int bp = Assert.Single(breakpoints!);
        Assert.Equal("<|bos|><user>Hi", tokenizer.Decode(tokens.GetRange(0, bp)));
        Assert.True(bp < tokens.Count, "The generation prompt must fall outside the marked prefix.");
    }

    [Fact]
    public void RenderToTokens_PartScopedMarker_BreaksInsideTheMessage_NotAtItsEnd()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        // The shape a part-scoped marker exists for: a big cacheable document
        // part, then a volatile question part in the SAME message. The
        // breakpoint has to land between them.
        var msg = new ChatMessage { Role = "user", Content = "DOC\nQ" };
        msg.AddContentCacheBreakpoint(3); // end of "DOC"

        var tokens = renderer.RenderToTokens(tokenizer, null, new List<ChatMessage> { msg },
            "fake", addGenerationPrompt: true, out var breakpoints);

        int bp = Assert.Single(breakpoints!);
        Assert.Equal("<|bos|><user>DOC", tokenizer.Decode(tokens.GetRange(0, bp)));

        // The question must fall OUTSIDE the marked prefix - collapsing the
        // marker onto the message would have swallowed it.
        Assert.DoesNotContain("Q", tokenizer.Decode(tokens.GetRange(0, bp)));
    }

    [Fact]
    public void RenderToTokens_PartScopedMarkers_AreAllKept_AndDoNotAlterTheTokens()
    {
        var tokenizer = new CharTokenizer();

        var marked = new ChatMessage { Role = "user", Content = "AAA\nBBB\nCCC" };
        marked.AddContentCacheBreakpoint(3);  // end of "AAA"
        marked.AddContentCacheBreakpoint(7);  // end of "BBB"

        var tokens = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null, new List<ChatMessage> { marked }, "fake",
            addGenerationPrompt: true, out var breakpoints);

        // An earlier marker must not be lost to a later one.
        Assert.Equal(2, breakpoints!.Count);
        Assert.Equal("<|bos|><user>AAA", tokenizer.Decode(tokens.GetRange(0, breakpoints[0])));
        Assert.Equal("<|bos|><user>AAA\nBBB", tokenizer.Decode(tokens.GetRange(0, breakpoints[1])));

        // And the prompt itself is untouched by the markers.
        var plain = new KVCachePromptRenderer(new FakeRenderer()).RenderToTokens(
            tokenizer, null,
            new List<ChatMessage> { new() { Role = "user", Content = "AAA\nBBB\nCCC" } },
            "fake", addGenerationPrompt: true, out _);
        Assert.Equal(plain, tokens);
    }

    [Fact]
    public void RenderToTokens_PartScopedAndMessageScopedMarkers_AreNumberedInTextOrder()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var tools = new List<ToolFunction>
        {
            new() { Name = "search", Description = "d", CacheControl = new CacheControlMarker() },
        };

        // All three anchors at once on the first message: the tool breakpoint in
        // front of the content, a part breakpoint inside it, and a
        // message-scoped one closing it.
        var first = new ChatMessage
        {
            Role = "system",
            Content = "DOC\nQ",
            CacheControl = new CacheControlMarker(),
        };
        first.AddContentCacheBreakpoint(3);

        var tokens = renderer.RenderToTokens(tokenizer, null, new List<ChatMessage> { first },
            "fake", addGenerationPrompt: true, out var breakpoints, tools: tools);

        Assert.Equal(3, breakpoints!.Count);
        Assert.Equal("<|bos|><system>", tokenizer.Decode(tokens.GetRange(0, breakpoints[0])));
        Assert.Equal("<|bos|><system>DOC", tokenizer.Decode(tokens.GetRange(0, breakpoints[1])));
        Assert.Equal("<|bos|><system>DOC\nQ", tokenizer.Decode(tokens.GetRange(0, breakpoints[2])));

        // Numbering must follow the text, so the reported list is ascending.
        Assert.True(breakpoints[0] < breakpoints[1] && breakpoints[1] < breakpoints[2],
            $"Expected ascending breakpoints, got [{string.Join(", ", breakpoints)}].");
        Assert.DoesNotContain(KVCachePromptRenderer.BreakpointSentinel, tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_PartScopedMarkerOnSplicedAssistantTurn_UsesSafePreRawBoundary()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var rawTokens = new List<int> { 1001, 1002, 1003 };

        // The content these offsets addressed is replaced wholesale by the raw
        // tokens, so the exact offset is unknowable. It must collapse to a safe
        // boundary before the raw run rather than disappear into cache-all mode.
        var assistant = new ChatMessage
        {
            Role = "assistant",
            Content = "IGNORED",
            RawOutputTokens = rawTokens,
        };
        assistant.AddContentCacheBreakpoint(3);

        var tokens = renderer.RenderToTokens(tokenizer, null,
            new List<ChatMessage> { new() { Role = "user", Content = "Hi" }, assistant },
            "fake", addGenerationPrompt: true, out var breakpoints);

        int rawStart = FindSubsequence(tokens, rawTokens);
        Assert.True(rawStart >= 0);
        Assert.Equal(rawStart, Assert.Single(breakpoints!));
        Assert.DoesNotContain(KVCachePromptRenderer.BreakpointSentinel, tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_ConditionalToolSplice_PreservesPartMarkerInToolResult()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();
        var rawTokens = tokenizer.Encode("RAW_GENERATED_TOOL_CALL", addSpecial: false);

        var toolResult = new ChatMessage { Role = "tool", Content = "RESULT_BODY" };
        toolResult.AddContentCacheBreakpoint(6); // end of "RESULT"

        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "find it" },
            new()
            {
                Role = "assistant",
                Content = "template cannot reproduce the generated round",
                RawOutputTokens = rawTokens,
                ToolCalls = new List<ToolCall> { new() { Name = "search" } },
            },
            toolResult,
            new() { Role = "user", Content = "continue" },
        };

        // Gemma 4 conditionally retries with raw-token splicing when the active
        // template cannot reconstruct a tool-calling round. The breakpoint sentinel
        // temporarily splits RESULT_BODY in that retry, but it must not make the
        // tool-result survival check reject an otherwise safe splice.
        var tokens = renderer.RenderToTokens(tokenizer, null, messages, "gemma4",
            addGenerationPrompt: true, out var breakpoints, enableThinking: true);

        Assert.True(FindSubsequence(tokens, rawTokens) >= 0,
            "Expected the conditional retry to return the raw-token-spliced pass.");

        int bp = Assert.Single(breakpoints!);
        Assert.EndsWith("<tool>RESULT", tokenizer.Decode(tokens.GetRange(0, bp)));
        Assert.DoesNotContain(KVCachePromptRenderer.BreakpointSentinel, tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_RecognizedGemmaReplayTemplateThatFallsBack_DoesNotThrowOrLoseToolResult()
    {
        // This contains every signature used to recognize the canonical Gemma 4
        // raw-tool replay shape, but its loop is deliberately empty. Jinja therefore
        // drops the user's question and ChatTemplate abandons it for the hardcoded
        // renderer. That fallback does not understand the render-only replay field, so
        // the optimized pass loses its placeholder. The KV renderer must detect that
        // and retry through its established adaptive splice path instead of throwing.
        const string recognizedButDroppingTemplate =
            "{%- macro format_tool_response_block(tool_name, response) -%}" +
            "{{- response -}}" +
            "{%- endmacro -%}" +
            "{%- set ns = namespace(prev_message_type=None) -%}" +
            "{%- set ns_turn = namespace(last_user_idx=-1) -%}" +
            "{%- set loop_messages = [] -%}" +
            "{%- for message in loop_messages -%}" +
            "{%- if message['role'] != 'tool' -%}" +
            "{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}" +
            "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}" +
            "{%- endif -%}" +
            "{%- if message['tool_calls'] -%}" +
            "{%- set ns.prev_message_type = 'tool_call' -%}" +
            "{%- endif -%}" +
            "{%- if message.get('tool_responses') -%}" +
            "{%- elif message.get('tool_calls') -%}" +
            "{%- for k in range(loop.index0 + 1, loop_messages | length) -%}" +
            "{%- endfor -%}" +
            "{%- endif -%}" +
            "{%- endif -%}" +
            "{%- endfor -%}" +
            "DROPPED";

        Assert.True(ChatTemplate.SupportsGemma4RawToolCallReplay(
            recognizedButDroppingTemplate, "gemma4"));

        var tokenizer = new CharTokenizer();
        var rawTokens = new List<int> { 1001, 1002, 1003 };
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "question" },
            new()
            {
                Role = "assistant",
                Content = "",
                Thinking = "use the shell",
                ToolCalls = new List<ToolCall>
                {
                    new()
                    {
                        Name = "shell",
                        Arguments = new Dictionary<string, object> { ["command"] = "pwd" },
                    },
                },
                RawOutputTokens = rawTokens,
                RawPromptTrailingWhitespace = "\n",
            },
            new() { Role = "tool", Content = "UNIQUE_TOOL_RESULT" },
        };

        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> tokens = renderer.RenderToTokens(
            tokenizer,
            recognizedButDroppingTemplate,
            messages,
            architecture: "gemma4",
            addGenerationPrompt: true,
            enableThinking: true);

        Assert.True(FindSubsequence(tokens, rawTokens) >= 0,
            "The safe retry must splice the exact cached tool-call tokens.");
        Assert.Contains("UNIQUE_TOOL_RESULT", tokenizer.Decode(tokens));
        Assert.DoesNotContain(KVCachePromptRenderer.PlaceholderSentinel, tokenizer.Decode(tokens));
    }

    [Fact]
    public void RenderToTokens_PartScopedMarkerPastEndOfContent_ClampsInsteadOfThrowing()
    {
        var renderer = new KVCachePromptRenderer(new FakeRenderer());
        var tokenizer = new CharTokenizer();

        var msg = new ChatMessage { Role = "user", Content = "Hi" };
        msg.AddContentCacheBreakpoint(999);

        var tokens = renderer.RenderToTokens(tokenizer, null, new List<ChatMessage> { msg },
            "fake", addGenerationPrompt: true, out var breakpoints);

        int bp = Assert.Single(breakpoints!);
        Assert.Equal("<|bos|><user>Hi", tokenizer.Decode(tokens.GetRange(0, bp)));
    }

    private static string MakePlaceholder(int index)
    {
        // Mirrors KVCachePromptRenderer.MakePlaceholder so we don't have to expose it
        // publicly. We assert structural invariants instead of behavior.
        return $"{KVCachePromptRenderer.PlaceholderSentinel}R{index:D4}{KVCachePromptRenderer.PlaceholderSentinel}";
    }

    private static string InvokePrivateStringMethod(string name, params object[] arguments)
    {
        var method = typeof(KVCachePromptRenderer).GetMethod(
            name,
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(method);
        return Assert.IsType<string>(method!.Invoke(null, arguments));
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
