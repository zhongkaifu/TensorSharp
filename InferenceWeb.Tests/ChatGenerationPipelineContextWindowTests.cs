// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Diagnostics;
using System.Text;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class ChatGenerationPipelineContextWindowTests
{
    private const int ContextLimit = 8_192;
    private const int ResponseReserve = 2_048;
    private readonly ITestOutputHelper _output;

    public ChatGenerationPipelineContextWindowTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void EightKMultiSkillRepair_PreservesInstructionsTaskAndNewestFailure()
    {
        const string task = "搜索apple M6的信息，并对比M5芯片，然后生成pptx报告";
        const string systemSentinel = "SYSTEM_POLICY_SENTINEL";
        const string developerSentinel = "DEVELOPER_SKILL_SENTINEL";
        const string codePromptSentinel = "CODE_EDIT_CONTRACT_SENTINEL";
        const string oldSkillSentinel = "OLD_25KB_SKILL_BODY_SENTINEL";
        const string oldResearchSentinel = "OLD_20KB_RESEARCH_OUTPUT_SENTINEL";
        const string newestFailureSentinel = "NEWEST_PPTX_REPAIR_FAILURE_SENTINEL";

        var history = new List<ChatMessage>
        {
            new()
            {
                Role = "system",
                Content = systemSentinel + "\nKeep the selected skill instructions.\n" + new string('S', 1_800),
            },
            new()
            {
                Role = "developer",
                Content = developerSentinel + "\n" + codePromptSentinel
                    + "\nPatch buggy files; do not rewrite them.\n" + new string('D', 1_800),
            },
            new() { Role = "user", Content = "OLD_USER_TASK_SENTINEL" },
            new() { Role = "assistant", Content = "OLD_USER_ANSWER_SENTINEL" },
            new() { Role = "user", Content = task },
        };

        // Nineteen tool-loop generations mirrors the long observed Qwen repair shape.
        // The first two results model deferred skill bodies and search output; the later
        // small rounds ensure compaction is chronological rather than a largest-item hack.
        for (int round = 0; round < 18; round++)
        {
            history.Add(new ChatMessage
            {
                Role = "assistant",
                Content = $"OLD_ASSISTANT_TOOL_ROUND_{round:D2}",
                ToolCalls = new List<ToolCall>
                {
                    new() { Name = "skills_read" },
                },
            });
            string content = round switch
            {
                0 => oldSkillSentinel + "\n" + new string('K', 25 * 1_024),
                1 => oldResearchSentinel + "\n" + new string('R', 20 * 1_024),
                _ => $"OLD_TOOL_RESULT_{round:D2}\n" + new string('o', 800),
            };
            // Mistral-family skill results use role=user. Include one in the corpus so
            // it cannot be mistaken for the actual Chinese request anchor.
            history.Add(new ChatMessage
            {
                Role = round == 2 ? "user" : "tool",
                Content = round == 2 ? "Result of your skills_read call:\n" + content : content,
            });
        }

        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "Repair the existing deck spec with apply_patch, then rerun make_pptx.py.",
        });
        history.Add(new ChatMessage
        {
            Role = "tool",
            Content = newestFailureSentinel
                + "\nmake_pptx: invalid JSON at line 27; deck-spec.json still exists and contains M5/M6.",
        });

        var renderer = new KVCachePromptRenderer(new TaggedRenderer());
        var tokenizer = new CharacterTokenizer();
        List<int> Render(List<ChatMessage> messages) => renderer.RenderToTokens(
            tokenizer, chatTemplate: null, messages, architecture: "qwen3_5",
            addGenerationPrompt: true);

        List<int> original = Render(history);
        var stopwatch = Stopwatch.StartNew();
        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, original.Count, ContextLimit, ResponseReserve,
                preserveAllInput: false,
                candidate => Render(candidate).Count);
        List<int> compacted = Render(window.History);
        stopwatch.Stop();
        string rendered = tokenizer.Decode(compacted);

        _output.WriteLine(
            $"8K context window: {original.Count:N0} -> {compacted.Count:N0} tokens; "
            + $"removed {window.RemovedMessages}/{history.Count} messages in {stopwatch.Elapsed.TotalMilliseconds:N1} ms");

        Assert.True(original.Count > 60_000, $"synthetic prompt was only {original.Count} tokens");
        Assert.True(compacted.Count <= ContextLimit - ResponseReserve,
            $"compacted prompt still used {compacted.Count}/{ContextLimit - ResponseReserve} prompt tokens");
        Assert.True(compacted.Count + ResponseReserve <= ContextLimit);
        Assert.Equal(compacted.Count, window.FinalPromptTokens);
        Assert.True(window.RemovedMessages >= 6);
        Assert.Equal(43, history.Count); // input history was not mutated

        Assert.Contains(systemSentinel, rendered, StringComparison.Ordinal);
        Assert.Contains(developerSentinel, rendered, StringComparison.Ordinal);
        Assert.Contains(codePromptSentinel, rendered, StringComparison.Ordinal);
        Assert.Contains(task, rendered, StringComparison.Ordinal);
        Assert.Contains(newestFailureSentinel, rendered, StringComparison.Ordinal);

        Assert.DoesNotContain("OLD_USER_TASK_SENTINEL", rendered, StringComparison.Ordinal);
        Assert.DoesNotContain(oldSkillSentinel, rendered, StringComparison.Ordinal);
        Assert.DoesNotContain(oldResearchSentinel, rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void ManyCompletedTurns_UseLogarithmicPromptMeasurements()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_SENTINEL" },
        };
        for (int turn = 0; turn < 128; turn++)
        {
            history.Add(new ChatMessage
            {
                Role = "user",
                Content = $"OLD_TASK_{turn:D3}_" + new string('u', 96),
            });
            history.Add(new ChatMessage
            {
                Role = "assistant",
                Content = $"OLD_ANSWER_{turn:D3}_" + new string('a', 96),
            });
        }
        history.Add(new ChatMessage { Role = "user", Content = "LATEST_TASK_SENTINEL" });

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        int measurements = 0;
        int original = Cost(history);
        const int limit = 5_000;
        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContext(
                history, original, limit,
                candidate =>
                {
                    measurements++;
                    return Cost(candidate);
                });

        Assert.True(window.FinalPromptTokens <= limit);
        Assert.Contains(window.History, message => message.Content == "SYSTEM_SENTINEL");
        Assert.Contains(window.History, message => message.Content == "LATEST_TASK_SENTINEL");
        Assert.True(measurements <= 8,
            $"128 removable turns required {measurements} full prompt measurements; expected logarithmic search");
    }

    [Fact]
    public void HostArtifactCorrectionKeepsTheGenuineTaskAndNewestEvidence()
    {
        const string task = "GENUINE_APPLE_M5_M6_TASK_SENTINEL";
        const string rejectedAnswer = "NEWEST_UNVERIFIED_ANSWER_SENTINEL";
        const string correction = "HOST_COMPLETION_REASON_SENTINEL";
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_SENTINEL" },
            new() { Role = "user", Content = "OLD_TASK_" + new string('u', 4_000) },
            new() { Role = "assistant", Content = "OLD_ANSWER_" + new string('a', 4_000) },
            new() { Role = "user", Content = task },
            new() { Role = "assistant", Content = "OLD_REPAIR_" + new string('r', 4_000) },
            new() { Role = "tool", Content = "OLD_FAILURE_" + new string('f', 4_000) },
            new() { Role = "assistant", Content = rejectedAnswer },
            new TensorSharp.Server.Skills.SkillChatLoop.HostCompletionCorrectionMessage
            {
                Role = "user",
                Content = correction,
            },
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContext(
                history,
                Cost(history),
                promptTokenLimit: 1_000,
                Cost);

        Assert.Contains(window.History, message => message.Content == task);
        Assert.Contains(window.History, message => message.Content == rejectedAnswer);
        Assert.Contains(window.History, message => message.Content == correction);
        Assert.DoesNotContain(window.History, message =>
            message.Content.StartsWith("OLD_TASK_", StringComparison.Ordinal));
        Assert.DoesNotContain(window.History, message =>
            message.Content.StartsWith("OLD_REPAIR_", StringComparison.Ordinal));
        Assert.True(window.FinalPromptTokens <= 1_000);
    }

    [Fact]
    public void InterleavedDeveloperPolicyIsProtectedWhereverItAppears()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "LEADING_SYSTEM_SENTINEL" },
            new() { Role = "user", Content = "OLD_TASK_ONE_" + new string('u', 4_000) },
            new() { Role = "assistant", Content = "OLD_ANSWER_ONE_" + new string('a', 4_000) },
            new() { Role = "developer", Content = "INTERLEAVED_DEVELOPER_POLICY_SENTINEL" },
            new() { Role = "user", Content = "OLD_TASK_TWO_" + new string('v', 4_000) },
            new() { Role = "assistant", Content = "OLD_ANSWER_TWO_" + new string('b', 4_000) },
            new() { Role = "user", Content = "LATEST_TASK_SENTINEL" },
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, Cost(history), contextLimit: 2_048, requestedGenerationTokens: 512,
                preserveAllInput: false, Cost);

        Assert.Contains(window.History, message =>
            message.Content == "LEADING_SYSTEM_SENTINEL");
        Assert.Contains(window.History, message =>
            message.Content == "INTERLEAVED_DEVELOPER_POLICY_SENTINEL");
        Assert.Contains(window.History, message =>
            message.Content == "LATEST_TASK_SENTINEL");
        Assert.DoesNotContain(window.History, message =>
            message.Content.StartsWith("OLD_TASK_", StringComparison.Ordinal));
        Assert.True(window.FinalPromptTokens <= 1_536);
    }

    [Fact]
    public void ProtectedMinimumOverHardLimitFailsClosedInsteadOfSuffixSlicingPolicy()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_POLICY_SENTINEL_" + new string('s', 1_100) },
            new() { Role = "user", Content = "OLD_TASK_" + new string('o', 2_000) },
            new() { Role = "assistant", Content = "OLD_ANSWER_" + new string('a', 2_000) },
            new() { Role = "developer", Content = "DEVELOPER_POLICY_SENTINEL_" + new string('d', 1_100) },
            new() { Role = "user", Content = "LATEST_TASK_SENTINEL_" + new string('u', 600) },
            new() { Role = "assistant", Content = "LATEST_REPAIR_SENTINEL_" + new string('r', 600) },
            new() { Role = "tool", Content = "LATEST_FAILURE_SENTINEL_" + new string('f', 600) },
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        PromptContextOverflowException error = Assert.Throws<PromptContextOverflowException>(() =>
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, Cost(history), contextLimit: 3_000, requestedGenerationTokens: 512,
                preserveAllInput: false, Cost));

        Assert.Contains("cannot be safely truncated", error.Message, StringComparison.Ordinal);
        Assert.Contains("3000-token context window", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void MultimodalHistoryUsesTheProductionBudgetAndKeepsInstructionsAndLatestImage()
    {
        var latest = new ChatMessage
        {
            Role = "user",
            Content = "LATEST_IMAGE_TASK_" + new string('m', 3_300),
            ImagePaths = new List<string> { "latest.png" },
        };
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_SENTINEL_" + new string('s', 2_900) },
            new() { Role = "user", Content = "OLD_TASK" },
            new() { Role = "assistant", Content = "OLD_LARGE_ANSWER_" + new string('o', 8_000) },
            latest,
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history,
                Cost(history),
                ContextLimit,
                ResponseReserve,
                preserveAllInput: false,
                Cost);

        // The protected minimum is intentionally a little larger than the preferred
        // 6,144-token prompt budget. Production adopts it because it still fits the
        // hard 8,191-token ceiling and will shrink the reply reserve accordingly.
        Assert.True(window.FinalPromptTokens > ContextLimit - ResponseReserve);
        Assert.True(window.FinalPromptTokens < ContextLimit);
        Assert.True(window.RemovedMessages >= 2);
        Assert.Contains(window.History, message =>
            message.Content.StartsWith("SYSTEM_SENTINEL_", StringComparison.Ordinal));
        ChatMessage retained = Assert.Single(window.History, message =>
            message.Content.StartsWith("LATEST_IMAGE_TASK_", StringComparison.Ordinal));
        Assert.Same(latest, retained);
        Assert.Equal(new[] { "latest.png" }, retained.ImagePaths);
        Assert.DoesNotContain(window.History, message =>
            message.Content.StartsWith("OLD_LARGE_ANSWER_", StringComparison.Ordinal));
    }

    [Fact]
    public void RemovedOldMediaDoesNotCauseLaterTextTurnsToBeOverCompacted()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_" + new string('s', 100) },
            new()
            {
                Role = "user",
                Content = "OLD_IMAGE_TASK_" + new string('i', 180),
                ImagePaths = new List<string> { "old-image.png" },
            },
            new() { Role = "assistant", Content = "OLD_IMAGE_ANSWER_" + new string('a', 180) },
            new() { Role = "user", Content = "KEEPABLE_TEXT_TASK_" + new string('t', 180) },
            new() { Role = "assistant", Content = "KEEPABLE_TEXT_ANSWER_" + new string('x', 780) },
            new() { Role = "user", Content = "LATEST_TASK_" + new string('l', 180) },
        };

        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        int original = Cost(history);
        ChatGenerationPipeline.ContextHistoryWindow conservative =
            ChatGenerationPipeline.CompactHistoryForContext(
                history, original, promptTokenLimit: 500, Cost);
        Assert.DoesNotContain(conservative.History, message =>
            message.Content.StartsWith("KEEPABLE_TEXT_TASK_", StringComparison.Ordinal));

        ChatGenerationPipeline.ContextHistoryWindow recovered =
            ChatGenerationPipeline.RecoverHistoryAfterRemovedMedia(
                history,
                original,
                targetPromptLimit: 1_500,
                hardPromptLimit: 2_000,
                remainingMediaOverhead: 0,
                conservative,
                remainingMediaFingerprint: null,
                Cost);

        Assert.True(recovered.RemovedMessages < conservative.RemovedMessages);
        Assert.DoesNotContain(recovered.History, message =>
            message.ImagePaths is { Count: > 0 });
        Assert.Contains(recovered.History, message =>
            message.Content.StartsWith("KEEPABLE_TEXT_TASK_", StringComparison.Ordinal));
        Assert.Contains(recovered.History, message =>
            message.Content.StartsWith("LATEST_TASK_", StringComparison.Ordinal));
        Assert.True(recovered.FinalPromptTokens <= 1_500);
    }

    [Fact]
    public void UserTextThatLooksLikeAToolResult_IsStillTheLatestTask()
    {
        const string task = "Result of your skills_read call: explain why this literal text appears in logs.";
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "SYSTEM_SENTINEL" },
            new() { Role = "user", Content = "OLD_TASK" },
            new() { Role = "assistant", Content = "OLD_ANSWER\n" + new string('x', 10_000) },
            new() { Role = "user", Content = task },
            new() { Role = "assistant", Content = "LATEST_ANSWER_SENTINEL" },
        };

        var renderer = new KVCachePromptRenderer(new TaggedRenderer());
        var tokenizer = new CharacterTokenizer();
        List<int> Render(List<ChatMessage> messages) => renderer.RenderToTokens(
            tokenizer, chatTemplate: null, messages, architecture: "qwen3_5",
            addGenerationPrompt: true);

        List<int> original = Render(history);
        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContext(
                history, original.Count, promptTokenLimit: 2_000,
                candidate => Render(candidate).Count);
        string compacted = tokenizer.Decode(Render(window.History));

        Assert.Contains("SYSTEM_SENTINEL", compacted, StringComparison.Ordinal);
        Assert.Contains(task, compacted, StringComparison.Ordinal);
        Assert.Contains("LATEST_ANSWER_SENTINEL", compacted, StringComparison.Ordinal);
        Assert.DoesNotContain("OLD_TASK", compacted, StringComparison.Ordinal);
        Assert.DoesNotContain("OLD_ANSWER", compacted, StringComparison.Ordinal);
    }

    // Regression, from a phone's log: the reply length set to 262,144 tokens inside a
    // 32,768-token window made the compactor's reserve 32,767 and its prompt budget ONE
    // token, so every tool round removed the whole conversation but the instructions,
    // the latest request and the newest round ("prompt.history_compacted from 13544 to
    // 7789 tokens by removing 16 old messages") -- an agent that fetched the same page
    // eight times because each round had forgotten the last. The reserve the compactor
    // protects is now at most a quarter of the window; the reply still gets whatever the
    // kept prompt leaves (ClampGenerationReserve), which in a short chat is the window.
    [Fact]
    public void HistoryCompactionReserve_IsAtMostAQuarterOfTheWindow()
    {
        Assert.Equal(8192, ChatGenerationPipeline.HistoryCompactionReserve(262144, 32768));
        Assert.Equal(8192, ChatGenerationPipeline.HistoryCompactionReserve(32767, 32768));
        Assert.Equal(2048, ChatGenerationPipeline.HistoryCompactionReserve(2048, 32768));
        Assert.Equal(4096, ChatGenerationPipeline.HistoryCompactionReserve(262144, 16384));
        // A small window keeps a 1,024-token floor rather than a quarter.
        Assert.Equal(1024, ChatGenerationPipeline.HistoryCompactionReserve(262144, 2048));
        // Never the whole window, whatever was asked.
        Assert.Equal(511, ChatGenerationPipeline.HistoryCompactionReserve(262144, 512));
        // And never below one.
        Assert.Equal(1, ChatGenerationPipeline.HistoryCompactionReserve(0, 4096));
    }

    [Fact]
    public void AReplyLimitAboveTheWindow_NoLongerEmptiesTheConversation()
    {
        // Twelve completed tool rounds of 1,000 tokens after 6,000 of instructions:
        // 18,000 tokens of prompt in a 32,768 window fits with 8,192 held for the reply.
        var history = new List<ChatMessage> { new() { Role = "system", Content = "instructions" } };
        for (int i = 0; i < 12; i++)
        {
            history.Add(new() { Role = "user", Content = $"task {i}" });
            history.Add(new() { Role = "assistant", Content = $"tool call {i}" });
        }
        history.Add(new() { Role = "user", Content = "latest task" });

        static int Count(List<ChatMessage> messages) => 6000 + (messages.Count - 1) * 500;

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, originalPromptTokens: Count(history), contextLimit: 32768,
                requestedGenerationTokens: 262144, preserveAllInput: false, Count);

        Assert.Equal(0, window.RemovedMessages);
        Assert.Same(history, window.History);

        // The same request with the OLD reserve (the whole window) would have kept only
        // the protected minimum; prove the budget the compactor now works to.
        ChatGenerationPipeline.ContextHistoryWindow tight =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, originalPromptTokens: Count(history), contextLimit: 20000,
                requestedGenerationTokens: 262144, preserveAllInput: false, Count);
        Assert.True(tight.RemovedMessages > 0, "a genuinely tight window still compacts");
        Assert.True(tight.FinalPromptTokens <= 20000 - 5000, "to the window minus a quarter for the reply");
    }

    [Fact]
    public void HistoryThatFits_IsReturnedUnchangedWithoutAnotherRender()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "instructions" },
            new() { Role = "user", Content = "task" },
        };
        int measurements = 0;

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContext(
                history, originalPromptTokens: 100, promptTokenLimit: 8_191,
                _ => { measurements++; return 100; });

        Assert.Same(history, window.History);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Equal(100, window.FinalPromptTokens);
        Assert.Equal(0, measurements);
    }

    private sealed class TaggedRenderer : IPromptRenderer
    {
        public string Render(
            string? template,
            List<ChatMessage> messages,
            bool addGenerationPrompt = true,
            string? architecture = null,
            List<ToolFunction>? tools = null,
            bool enableThinking = false)
        {
            var text = new StringBuilder("<bos>");
            foreach (ChatMessage message in messages)
            {
                text.Append('<').Append(message.Role).Append('>')
                    .Append(message.Content)
                    .Append("</").Append(message.Role).Append('>');
            }
            if (addGenerationPrompt)
                text.Append("<assistant>");
            return text.ToString();
        }
    }

    private sealed class CharacterTokenizer : ITokenizer
    {
        private const int Bos = 0;
        private const int Eos = 1;
        private readonly Dictionary<char, int> _ids = new();
        private readonly List<string> _vocab = new() { "<bos>", "<eos>" };

        public string[] Vocab => _vocab.ToArray();
        public int BosTokenId => Bos;
        public int[] EosTokenIds => new[] { Eos };
        public int VocabSize => _vocab.Count;

        public List<int> Encode(string text, bool addSpecial = true)
        {
            var tokens = new List<int>(text?.Length + 1 ?? 1);
            if (addSpecial)
                tokens.Add(Bos);
            if (text == null)
                return tokens;

            foreach (char value in text)
            {
                if (!_ids.TryGetValue(value, out int id))
                {
                    id = _vocab.Count;
                    _ids[value] = id;
                    _vocab.Add(value.ToString());
                }
                tokens.Add(id);
            }
            return tokens;
        }

        public string Decode(List<int> ids)
        {
            var text = new StringBuilder();
            foreach (int id in ids)
            {
                if (id != Bos && id != Eos && id >= 0 && id < _vocab.Count)
                    text.Append(_vocab[id]);
            }
            return text.ToString();
        }

        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            if (tokenId != Bos && tokenId != Eos && tokenId >= 0 && tokenId < _vocab.Count)
                buffer.AddRange(Encoding.UTF8.GetBytes(_vocab[tokenId]));
        }

        public bool IsEos(int tokenId) => tokenId == Eos;

        public int LookupToken(string tokenStr) =>
            tokenStr?.Length == 1 && _ids.TryGetValue(tokenStr[0], out int id) ? id : -1;
    }
}
