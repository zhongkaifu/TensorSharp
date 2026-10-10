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

        // The repair's own old rounds are held to the preferred budget, oldest first; the
        // short completed turn before the task is decided on what they leave, and fits.
        Assert.Contains("OLD_USER_TASK_SENTINEL", rendered, StringComparison.Ordinal);
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

    // Live, Gemma 4 E2B at 8,192 with a ~6.2k shared prompt: every follow-up logged
    // "prompt.history_compacted ... removing 2 old messages", the photo and the answer about
    // it among them, and "what colour is the text in it?" was answered about nothing. The
    // reply now shares what the protected prompt leaves with completed turns, floored at
    // min(requested, 1,024), and only when that is less than twice the preferred reserve.
    [Fact]
    public void HistoryCompactionReserve_SharesATightWindowWithCompletedTurns()
    {
        // (8191 - 6200) / 2 = 995, floored at min(8192, 1024).
        Assert.Equal(1024, ChatGenerationPipeline.HistoryCompactionReserve(8192, 8192, protectedTokens: 6200));
        Assert.Equal(7168, ChatGenerationPipeline.AdaptivePromptLimit(8192, 8192, protectedTokens: 6200));
        // A middle case: (8191 - 5191) / 2 = 1500.
        Assert.Equal(1500, ChatGenerationPipeline.HistoryCompactionReserve(8192, 8192, protectedTokens: 5191));
        // A 32k window with the same shared prompt leaves 25,567 free: unchanged.
        Assert.Equal(8192, ChatGenerationPipeline.HistoryCompactionReserve(8192, 32768, protectedTokens: 7200));
        // Never more than the preferred reserve, never below what was asked when that is smaller.
        Assert.Equal(512, ChatGenerationPipeline.HistoryCompactionReserve(512, 8192, protectedTokens: 7900));
        Assert.Equal(1024, ChatGenerationPipeline.HistoryCompactionReserve(8192, 8192, protectedTokens: 9000));
        // Unmeasured protected prompt: the preferred reserve.
        Assert.Equal(2048, ChatGenerationPipeline.HistoryCompactionReserve(8192, 8192, protectedTokens: 0));
    }

    [Fact]
    public void ATightWindowKeepsTheEarlierImageTurnBesideTheSharedPrompt()
    {
        var earlierImage = new ChatMessage
        {
            Role = "user",
            Content = "U1_" + new string('i', 285),
            ImagePaths = new List<string> { "photo.png" },
        };
        List<ChatMessage> Conversation(int answerLength) => new()
        {
            new() { Role = "system", Content = new string('s', 6_150) },
            earlierImage,
            new() { Role = "assistant", Content = "A1_" + new string('a', answerLength - 3) },
            new() { Role = "user", Content = "U2_" + new string('u', 23) },
        };
        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        // Protected (instructions + latest request) is 6,200; the image turn is 600.
        List<ChatMessage> kept = Conversation(answerLength: 288);
        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                kept, Cost(kept), ContextLimit, requestedGenerationTokens: 8_192,
                preserveAllInput: false, Cost);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Same(kept, window.History);
        Assert.Equal(6_200, window.ProtectedTokens);
        Assert.Equal(7_168, window.PromptLimit);
        Assert.True(window.FinalPromptTokens + (ContextLimit - window.PromptLimit) <= ContextLimit);

        // A 3,000-token answer does not fit beside the shared prompt and the reply's 1,024.
        List<ChatMessage> tooLong = Conversation(answerLength: 2_988);
        ChatGenerationPipeline.ContextHistoryWindow removed =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                tooLong, Cost(tooLong), ContextLimit, requestedGenerationTokens: 8_192,
                preserveAllInput: false, Cost);
        Assert.Equal(2, removed.RemovedMessages);
        Assert.Equal(6_200, removed.FinalPromptTokens);
        Assert.DoesNotContain(removed.History, message => ReferenceEquals(message, earlierImage));
        Assert.True(removed.FinalPromptTokens + (ContextLimit - removed.PromptLimit) <= ContextLimit);
    }

    // The room the adaptive split buys is for COMPLETED turns. Older rounds of the request
    // in progress are fitted to the preferred reserve, so a tool loop never keeps its own
    // old results at the expense of the call it is about to write.
    [Fact]
    public void OlderRoundsOfTheActiveRequestKeepThePreferredReserve()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 4_000) },
            new() { Role = "user", Content = "OLD_TASK_" + new string('o', 991) },
            new() { Role = "assistant", Content = "OLD_ANSWER_" + new string('a', 989) },
            new() { Role = "user", Content = "LATEST_TASK_" + new string('t', 76) },
            new()
            {
                Role = "assistant", Content = "OLD_ROUND_" + new string('r', 478),
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
            },
            new() { Role = "tool", Content = "OLD_RESULT_" + new string('x', 1_227) },
            new()
            {
                Role = "assistant", Content = "NEWEST_ROUND_" + new string('n', 175),
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
            },
            new() { Role = "tool", Content = "NEWEST_RESULT_" + new string('y', 274) },
        };
        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);

        ChatGenerationPipeline.ContextHistoryWindow window =
            ChatGenerationPipeline.CompactHistoryForContextBudget(
                history, Cost(history), ContextLimit, requestedGenerationTokens: 8_192,
                preserveAllInput: false, Cost);

        // The request with all of its rounds is 6,362 tokens: inside what an adaptive split
        // on it would allow (6,403) but past the preferred 6,144, so the older round goes.
        Assert.DoesNotContain(window.History, message => message.Content.StartsWith("OLD_ROUND_", StringComparison.Ordinal));
        Assert.DoesNotContain(window.History, message => message.Content.StartsWith("OLD_RESULT_", StringComparison.Ordinal));
        Assert.Contains(window.History, message => message.Content.StartsWith("LATEST_TASK_", StringComparison.Ordinal));
        Assert.Contains(window.History, message => message.Content.StartsWith("NEWEST_RESULT_", StringComparison.Ordinal));
        // The completed turn is then decided on what is left, 4,612 tokens: (8,191 - 4,612) / 2
        // = 1,789 for the reply, a 6,403 budget, and the 2,024-token turn does not fit in it.
        Assert.Equal(4_612, window.ProtectedTokens);
        Assert.Equal(6_403, window.PromptLimit);
        Assert.Equal(4_612, window.FinalPromptTokens);
        Assert.DoesNotContain(window.History, message => message.Content.StartsWith("OLD_TASK_", StringComparison.Ordinal));
        Assert.Equal(2, window.RemovedTurnMessages);
        Assert.Equal(4, window.RemovedMessages);
    }

    // A tool loop on a phone, whose shared prompt alone is past the preferred 6,144: the
    // reply shares the window with the completed image turn on rounds one and two, and on
    // round three the request has an older round to give up. That round goes -- the active
    // request is held to the preferred budget -- and the completed turn is decided on what
    // is left, as before. Deciding it on the rounds as they were, and removing in order when
    // they did not fit, took the image turn with the old round mid-turn: the conversation
    // changed at u1 between rounds two and three, and "a PDF of that photo" lost the photo.
    [Fact]
    public void ACompletedTurnKeptOnTheFirstRounds_StaysWhenTheRequestHasAnOlderRound()
    {
        var system = new ChatMessage { Role = "system", Content = new string('s', 6_150) };
        var u1 = new ChatMessage { Role = "user", Content = "U1_" + new string('p', 185) };
        var a1 = new ChatMessage { Role = "assistant", Content = "A1_" + new string('a', 185) };
        var u2 = new ChatMessage { Role = "user", Content = "U2_MAKE_A_PDF_OF_THAT_PHOTO" };
        ChatMessage Call(string tag) => new()
        {
            Role = "assistant", Content = tag + "_CALL_" + new string('c', 182 - tag.Length),
            ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
        };
        ChatMessage Result(string tag) => new()
        {
            Role = "tool", Content = tag + "_RESULT_" + new string('r', 180 - tag.Length),
        };
        static int Cost(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12);
        ChatMessage call1 = Call("R1"), result1 = Result("R1"), call2 = Call("R2"), result2 = Result("R2");

        var rounds = new List<List<ChatMessage>>
        {
            new() { system, u1, a1, u2 },
            new() { system, u1, a1, u2, call1, result1 },
            new() { system, u1, a1, u2, call1, result1, call2, result2 },
        };
        var windows = rounds.Select(history => ChatGenerationPipeline.CompactHistoryForContextBudget(
            history, Cost(history), ContextLimit, requestedGenerationTokens: 8_192,
            preserveAllInput: false, Cost)).ToList();

        foreach (ChatGenerationPipeline.ContextHistoryWindow window in windows)
        {
            Assert.Same(u1, window.History[1]);
            Assert.Same(a1, window.History[2]);
            Assert.Equal(0, window.RemovedTurnMessages);
            Assert.True(window.FinalPromptTokens <= window.PromptLimit);
        }
        Assert.Equal(0, windows[0].RemovedMessages);
        Assert.Equal(0, windows[1].RemovedMessages);
        // Round three: the request without its older round is 6,601 tokens, past the
        // preferred budget even so; the reply keeps max(795, 1,024), and the 400-token turn
        // fits the 7,168 that leaves (7,001).
        Assert.Equal(new[] { system, u1, a1, u2, call2, result2 }, windows[2].History);
        Assert.Equal(2, windows[2].RemovedMessages);
        Assert.Equal(6_601, windows[2].ProtectedTokens);
        Assert.Equal(7_168, windows[2].PromptLimit);
        Assert.Equal(7_001, windows[2].FinalPromptTokens);
    }

    /// <summary>A prompt's cost with every image a placeholder token, and with every image
    /// expanded by the encoder's tokens for it.</summary>
    private sealed class MediaCost(Dictionary<string, int> expansion)
    {
        internal int Prepared { get; private set; }

        internal int Rendered { get; private set; }

        internal int Unexpanded(List<ChatMessage> messages)
        {
            Rendered++;
            return Size(messages);
        }

        internal int Expanded(List<ChatMessage> messages)
        {
            Prepared++;
            return Size(messages) + messages.Sum(message =>
                message.ImagePaths?.Sum(path => expansion[path]) ?? 0);
        }

        private static int Size(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12 + (message.ImagePaths?.Count ?? 0));
    }

    private static ChatMessage ImageMessage(string content, string path, string name) => new()
    {
        Role = "user",
        Content = content,
        ImagePaths = new List<string> { path },
        AttachmentPaths = new List<string> { path },
        AttachmentNames = new List<string> { name },
    };

    // An image is in the first user message, so removing whole turns oldest-first lost the
    // question about it AND the model's answer the moment the expanded prompt overflowed.
    // The pixels go first; the words stay, with a note saying the picture is gone.
    [Fact]
    public void AnEarlierImageGivesWayBeforeItsTurn_SoTheTurnsTextStays()
    {
        ChatMessage u1 = ImageMessage("U1_DESCRIBE_THIS_PHOTO", "/uploads/g1.png", "cat.png");
        var a1 = new ChatMessage { Role = "assistant", Content = "A1_IT_IS_A_RED_SIGN_" + new string('a', 99) };
        var u2 = new ChatMessage { Role = "user", Content = "What colour is the text in it?" };
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 5_000) },
            u1, a1, u2,
        };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 1_500 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Equal(4, window.History.Count);
        ChatMessage elided = window.History[1];
        Assert.Null(elided.ImagePaths);
        Assert.StartsWith("[earlier image 'cat.png' is no longer shown]", elided.Content, StringComparison.Ordinal);
        Assert.EndsWith("U1_DESCRIBE_THIS_PHOTO", elided.Content, StringComparison.Ordinal);
        Assert.Same(a1, window.History[2]);
        Assert.Same(u2, window.History[3]);
        Assert.Equal(cost.Unexpanded(window.History), window.FinalPromptTokens);
        Assert.True(window.FinalPromptTokens <= window.PromptLimit);
        // The caller's conversation is not touched.
        Assert.Equal(new[] { "/uploads/g1.png" }, u1.ImagePaths);
        Assert.Equal("U1_DESCRIBE_THIS_PHOTO", u1.Content);
    }

    [Fact]
    public void OnlyAsManyEarlierImagesGoAsTheWindowNeeds_OldestFirst()
    {
        ChatMessage u1 = ImageMessage("U1_FIRST_PHOTO", "/uploads/one.png", "one.png");
        ChatMessage u2 = ImageMessage("U2_SECOND_PHOTO", "/uploads/two.png", "two.png");
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 3_100) },
            u1,
            new() { Role = "assistant", Content = "A1_ANSWER_ONE" },
            u2,
            new() { Role = "assistant", Content = "A2_ANSWER_TWO" },
            new() { Role = "user", Content = "Compare them." },
        };
        var cost = new MediaCost(new() { ["/uploads/one.png"] = 1_500, ["/uploads/two.png"] = 1_500 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Null(window.History[1].ImagePaths);
        Assert.StartsWith("[earlier image 'one.png'", window.History[1].Content, StringComparison.Ordinal);
        Assert.Same(u2, window.History[3]);
        Assert.Equal(new[] { "/uploads/two.png" }, window.History[3].ImagePaths);
    }

    [Fact]
    public void TheLatestRequestsOwnImageIsNeverElided_AndIsChargedToTheProtectedPrompt()
    {
        ChatMessage u1 = ImageMessage("U1_OLD_PHOTO", "/uploads/old.png", "old.png");
        ChatMessage latest = ImageMessage("And this one?", "/uploads/new.png", "new.png");
        var system = new ChatMessage { Role = "system", Content = new string('s', 5_000) };
        var history = new List<ChatMessage>
        {
            system,
            u1,
            new() { Role = "assistant", Content = "A1_" + new string('a', 200) },
            latest,
        };
        var cost = new MediaCost(new() { ["/uploads/old.png"] = 1_500, ["/uploads/new.png"] = 1_500 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Null(window.History[1].ImagePaths);
        Assert.Same(latest, window.History[3]);
        // Protected: the instructions and the latest request, its own image expanded.
        Assert.Equal(cost.Unexpanded(new List<ChatMessage> { system, latest }) + 1_500, window.ProtectedTokens);
    }

    [Fact]
    public void WhenEveryEarlierImageIsGone_CompleteOldTurnsGoToo()
    {
        ChatMessage u1 = ImageMessage("U1_PHOTO", "/uploads/g1.png", "g1.png");
        var system = new ChatMessage { Role = "system", Content = new string('s', 5_000) };
        var latest = new ChatMessage { Role = "user", Content = "LATEST_TASK" };
        var history = new List<ChatMessage>
        {
            system,
            u1,
            new() { Role = "assistant", Content = "A1_" + new string('a', 3_000) },
            latest,
        };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 1_500 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Equal(2, window.RemovedMessages);
        Assert.Equal(new[] { system, latest }, window.History);
        Assert.Equal(cost.Unexpanded(window.History), window.FinalPromptTokens);
        Assert.True(window.FinalPromptTokens <= ContextLimit - 1);
    }

    [Fact]
    public void AnExpandedPromptThatFitsIsLeftAloneWithoutPreparingAnother()
    {
        ChatMessage u1 = ImageMessage("U1_PHOTO", "/uploads/g1.png", "g1.png");
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 1_000) },
            u1,
            new() { Role = "assistant", Content = "A1" },
            new() { Role = "user", Content = "LATEST_TASK" },
        };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 1_500 });
        int expanded = cost.Expanded(history);

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), expanded,
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Same(history, window.History);
        Assert.Equal(0, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Equal(1, cost.Prepared);
    }

    // The protected prompt is charged the expansion of the media it keeps, which only a
    // preparation of the prompt with every earlier picture elided measures -- and then this
    // prompt is prepared again, because that one was the last prepared. On a phone, whose
    // shared prompt alone is past the preferred budget, that was two renders and two media
    // preparations on every image turn after the first, for a prompt that fit as it was.
    // The expansion lies between none and all of this prompt's, and a prompt that fits at
    // both ends fits between them: nothing more is prepared, and the one render of the
    // protected prompt serves both ends.
    [Fact]
    public void AnImageTurnThatFitsAsItIs_PreparesNothingMore()
    {
        ChatMessage u1 = ImageMessage("U1_PHOTO", "/uploads/g1.png", "photo.png");
        ChatMessage latest = ImageMessage("U2_AND_THIS_ONE", "/uploads/g2.png", "second.png");
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 5_600) },
            u1,
            new() { Role = "assistant", Content = "A1_ANSWER" },
            latest,
        };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 300, ["/uploads/g2.png"] = 300 });
        int unexpanded = cost.Unexpanded(history);
        int expanded = cost.Expanded(history);
        Assert.True(expanded > ContextLimit - ResponseReserve);

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, unexpanded, expanded,
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Same(history, window.History);
        Assert.Equal(0, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        Assert.Equal(1, cost.Prepared);
        Assert.Equal(2, cost.Rendered);
    }

    // The same image turn, with the window too tight for it: the measurement is made, and
    // the earlier picture goes while its words stay.
    [Fact]
    public void AnImageTurnThatDoesNotFit_IsMeasuredAndLosesItsEarlierPicture()
    {
        ChatMessage u1 = ImageMessage("U1_PHOTO", "/uploads/g1.png", "photo.png");
        ChatMessage latest = ImageMessage("U2_AND_THIS_ONE", "/uploads/g2.png", "second.png");
        var system = new ChatMessage { Role = "system", Content = new string('s', 5_600) };
        var history = new List<ChatMessage>
        {
            system, u1, new() { Role = "assistant", Content = "A1_ANSWER" }, latest,
        };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 1_200, ["/uploads/g2.png"] = 300 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        Assert.StartsWith("[earlier image 'photo.png' is no longer shown]", window.History[1].Content, StringComparison.Ordinal);
        Assert.Same(latest, window.History[3]);
        Assert.Equal(2, cost.Prepared);   // this prompt, and the measurement, which is the result
    }

    // The protected prompt holds none of the completed turns' pictures, so the check that
    // the active request's rounds fit the preferred budget is not charged them; and older
    // rounds the request must give up are not a reason to drop an earlier picture that fits.
    // Charging the request the photo of the first turn elided it on the third round of a
    // tool loop in a 16k window although it fit (13,543 of a 14,058 budget), and rewrote that
    // message mid-turn -- on Gemma 4 a re-prefill of the request from the shared prompt.
    [Fact]
    public void AnEarlierImageIsNotChargedToTheActiveRequest_NorDroppedWithItsOlderRounds()
    {
        ChatMessage u1 = ImageMessage("U1_PHOTO", "/uploads/g1.png", "photo.png");
        var a1 = new ChatMessage { Role = "assistant", Content = "A1_" + new string('a', 276) };
        var system = new ChatMessage { Role = "system", Content = new string('s', 7_188) };
        var u2 = new ChatMessage { Role = "user", Content = "U2_MAKE_A_PDF_OF_IT" };
        var cost = new MediaCost(new() { ["/uploads/g1.png"] = 1_500 });
        const int window16k = 16_384;

        var history = new List<ChatMessage> { system, u1, a1, u2 };
        var calls = new List<ChatMessage>();
        for (int round = 1; round <= 5; round++)
        {
            var call = new ChatMessage
            {
                Role = "assistant", Content = $"R{round}_CALL_" + new string('c', 730),
                ToolCalls = new List<ToolCall> { new() { Name = "shell" } },
            };
            history.Add(call);
            history.Add(new ChatMessage { Role = "tool", Content = $"R{round}_RESULT_" + new string('r', 728) });
            calls.Add(call);
            if (round < 3)
                continue;

            ChatGenerationPipeline.MediaHistoryWindow window =
                ChatGenerationPipeline.CompactMediaHistoryForContext(
                    history, cost.Unexpanded(history), cost.Expanded(history),
                    window16k, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

            Assert.Equal(0, window.ElidedMedia);
            Assert.Same(u1, window.History[1]);
            Assert.Same(a1, window.History[2]);
            Assert.Same(calls[^1], window.History[^2]);
            Assert.True(window.FinalPromptTokens <= window.PromptLimit);
            // Round three keeps every round; then the oldest go, two messages each.
            Assert.Equal(2 * Math.Max(0, round - 3), window.RemovedMessages);
        }
    }

    // Elision is per picture, oldest first: a message with two photos loses its first and
    // keeps the second when that is enough, and the note names only the one that went.
    [Fact]
    public void AMessageWithTwoPhotosLosesOnlyAsManyAsTheWindowNeeds()
    {
        var u1 = new ChatMessage
        {
            Role = "user",
            Content = "U1_TWO_PHOTOS",
            ImagePaths = new List<string> { "/uploads/a.png", "/uploads/b.png" },
            ImageTimestamps = new List<double?> { null, null },
            AttachmentPaths = new List<string> { "/uploads/a.png", "/uploads/b.png" },
            AttachmentNames = new List<string> { "front.png", "back.png" },
        };
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 3_100) },
            u1,
            new() { Role = "assistant", Content = "A1_ANSWER" },
            new() { Role = "user", Content = "Which is brighter?" },
        };
        var cost = new MediaCost(new() { ["/uploads/a.png"] = 1_500, ["/uploads/b.png"] = 1_500 });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        ChatMessage kept = window.History[1];
        Assert.Equal(new[] { "/uploads/b.png" }, kept.ImagePaths);
        Assert.Equal(new double?[] { null }, kept.ImageTimestamps);
        Assert.Equal("[earlier image 'front.png' is no longer shown]\n\nU1_TWO_PHOTOS", kept.Content);
        Assert.True(window.FinalPromptTokens <= window.PromptLimit);
        Assert.Equal(new[] { "/uploads/a.png", "/uploads/b.png" }, u1.ImagePaths);
    }

    // A video is one item: half a video is not a smaller video. Its frames go together,
    // under one note, and the message is no longer a video.
    [Fact]
    public void AnEarlierVideoGivesWayWhole()
    {
        var clip = new ChatMessage
        {
            Role = "user",
            Content = "U1_CLIP",
            ImagePaths = new List<string> { "/uploads/f1.png", "/uploads/f2.png", "/uploads/f3.png" },
            ImageTimestamps = new List<double?> { 0.0, 1.0, 2.0 },
            IsVideo = true,
        };
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 3_100) },
            clip,
            new() { Role = "assistant", Content = "A1_ANSWER" },
            new() { Role = "user", Content = "What happened next?" },
        };
        var cost = new MediaCost(new()
        {
            ["/uploads/f1.png"] = 1_000, ["/uploads/f2.png"] = 1_000, ["/uploads/f3.png"] = 1_000,
        });

        ChatGenerationPipeline.MediaHistoryWindow window =
            ChatGenerationPipeline.CompactMediaHistoryForContext(
                history, cost.Unexpanded(history), cost.Expanded(history),
                ContextLimit, requestedGenerationTokens: 8_192, cost.Unexpanded, cost.Expanded);

        Assert.Equal(1, window.ElidedMedia);
        ChatMessage elided = window.History[1];
        Assert.Null(elided.ImagePaths);
        Assert.Null(elided.ImageTimestamps);
        Assert.False(elided.IsVideo);
        Assert.Equal("[an earlier video is no longer shown]\n\nU1_CLIP", elided.Content);
    }

    // The order is fixed -- message by message, a message's images then its recordings --
    // so which items go depends on the conversation and the window alone.
    [Fact]
    public void EarlierMediaItems_AreOldestFirstAndSkipTheLatestRequestAndInstructions()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "S", ImagePaths = new List<string> { "/uploads/sys.png" } },
            new()
            {
                Role = "user", Content = "U1",
                ImagePaths = new List<string> { "/uploads/a.png", "/uploads/b.png" },
                AudioPaths = new List<string> { "/uploads/a.wav" },
            },
            new() { Role = "assistant", Content = "A1" },
            new()
            {
                Role = "user", Content = "U2", IsVideo = true,
                ImagePaths = new List<string> { "/uploads/f1.png", "/uploads/f2.png" },
            },
            new() { Role = "assistant", Content = "A2" },
            new() { Role = "user", Content = "U3", ImagePaths = new List<string> { "/uploads/latest.png" } },
        };

        Assert.Equal(
            new[]
            {
                new ChatGenerationPipeline.EarlierMedia(1, 1, 0),
                new ChatGenerationPipeline.EarlierMedia(1, 1, 0),
                new ChatGenerationPipeline.EarlierMedia(1, 0, 1),
                new ChatGenerationPipeline.EarlierMedia(3, 2, 0),
            },
            ChatGenerationPipeline.EarlierMediaItems(history));

        // Two items: both of U1's images, its recording kept.
        List<ChatMessage> two = ChatGenerationPipeline.ElideEarlierMedia(
            history, ChatGenerationPipeline.EarlierMediaItems(history), 2);
        Assert.Null(two[1].ImagePaths);
        Assert.Equal(new[] { "/uploads/a.wav" }, two[1].AudioPaths);
        Assert.Same(history[3], two[3]);
        Assert.Same(history[5], two[5]);
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
