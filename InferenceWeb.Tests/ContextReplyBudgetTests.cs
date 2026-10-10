// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace InferenceWeb.Tests;

public sealed class ContextReplyBudgetTests
{
    private static ChatMessage Message(string role, int size) => new() { Role = role, Content = new string('x', size) };
    private static int Count(List<ChatMessage> history) => history.Sum(message => message.Content.Length);
    private static List<ChatMessage> History(int firstRound, int secondRound = 0)
    {
        var history = new List<ChatMessage> { Message("system", 6100), Message("user", 200) };
        history.Add(Message("assistant", firstRound / 2));
        history.Add(Message("tool", firstRound - firstRound / 2));
        if (secondRound > 0)
        {
            history.Add(Message("assistant", secondRound / 2));
            history.Add(Message("tool", secondRound - secondRound / 2));
        }
        history.Add(Message("assistant", 100));
        history.Add(Message("tool", 100));
        return history;
    }

    [Fact]
    public void ToolPolicyLargerThanPreferredTargetDoesNotEraseSuccessfulEarlierWork()
    {
        var history = History(400);
        var result = ChatGenerationPipeline.CompactHistoryForContextBudget(history, Count(history),
            8192, 4096, false, Count);
        Assert.Equal(0, result.RemovedMessages);
        Assert.Same(history, result.History);
        Assert.Equal(846, result.ReplyReserve);
        Assert.True(result.FinalPromptTokens + result.ReplyReserve <= 8192);
    }

    [Fact]
    public void PressureRemovesOnlyEnoughCompleteRoundsAndPreservesTaskAndNewestRepair()
    {
        var history = History(700, 700);
        var result = ChatGenerationPipeline.CompactHistoryForContextBudget(history, Count(history),
            8192, 4096, false, Count);
        Assert.Equal(2, result.RemovedMessages);
        Assert.DoesNotContain(history[2], result.History);
        Assert.Contains(history[4], result.History);
        Assert.Same(history[0], result.History[0]);
        Assert.Same(history[1], result.History[1]);
        Assert.Same(history[^1], result.History[^1]);
        Assert.True(result.FinalPromptTokens + result.ReplyReserve <= 8192);
    }

    [Fact]
    public void OrdinaryReachableTargetKeepsTheExistingReserve()
    {
        var history = History(400);
        var result = ChatGenerationPipeline.CompactHistoryForContextBudget(history, Count(history),
            16384, 4096, false, Count);
        Assert.Equal(4096, result.ReplyReserve);
        Assert.Equal(0, result.RemovedMessages);
    }

    [Fact]
    public void ProtectedOverflowStillFailsInsteadOfTruncatingPolicyOrTask()
    {
        var history = History(400);
        Assert.Throws<PromptContextOverflowException>(() => ChatGenerationPipeline.CompactHistoryForContextBudget(
            history, Count(history), 6400, 4096, false, Count));
    }

    [Fact]
    public void MediaExpansionUsesRemainingTextCapacityForTheSamePolicy()
    {
        var history = History(400);
        // An encoder's measured 4096 extra media tokens leave 8192 text slots.
        var result = ChatGenerationPipeline.CompactHistoryForReplyReserve(history, Count(history),
            12288 - 4096, 3072, Count);
        Assert.Equal(0, result.RemovedMessages);
        Assert.Equal(846, result.ReplyReserve);
        Assert.True(result.FinalPromptTokens + 4096 + result.ReplyReserve <= 12288);
    }

    [Fact]
    public void CurrentImageAndLargeToolPolicyStillShareReplySpaceWithoutErasingEarlierWork()
    {
        var history = History(400);
        history[1].ImagePaths = new List<string> { "/current.png" };
        int Expanded(List<ChatMessage> messages) => Count(messages)
            + messages.Sum(message => (message.ImagePaths?.Count ?? 0) * 400);

        var result = ChatGenerationPipeline.CompactMediaHistoryForContext(
            history, Count(history), Expanded(history), 8192, 4096, Count, Expanded);

        Assert.Same(history, result.History);
        Assert.Equal(0, result.ElidedMedia);
        Assert.Equal(0, result.RemovedMessages);
        Assert.Same(history[0], result.History[0]);
        Assert.Same(history[1], result.History[1]);
        Assert.Same(history[^1], result.History[^1]);
        Assert.Equal(7300, result.FinalPromptTokens);
        Assert.Equal(7546, result.PromptLimit);
        Assert.True(result.FinalPromptTokens <= result.PromptLimit);
        Assert.True(result.PromptLimit < 8192);
    }

    [Fact]
    public void AfterEarlierImageElisionReplyFallbackStillReportsRemovedCompletedTurns()
    {
        var history = History(400);
        history[0] = Message("system", 6600);
        history[1].ImagePaths = new List<string> { "/current.png" };
        var earlier = Message("user", 20);
        earlier.ImagePaths = new List<string> { "/earlier.png" };
        earlier.AttachmentPaths = new List<string> { "/earlier.png" };
        earlier.AttachmentNames = new List<string> { "earlier.png" };
        history.InsertRange(1, new[] { earlier, Message("assistant", 2000) });
        int Expanded(List<ChatMessage> messages) => Count(messages)
            + messages.Sum(message => (message.ImagePaths?.Count ?? 0) * 400);

        var result = ChatGenerationPipeline.CompactMediaHistoryForContext(
            history, Count(history), Expanded(history), 8192, 4096, Count, Expanded);

        Assert.Equal(1, result.ElidedMedia);
        Assert.Equal(4, result.RemovedMessages);
        Assert.Equal(new[] { history[0], history[3], history[^2], history[^1] }, result.History);
        Assert.Equal(7400, result.ProtectedTokens);
        Assert.Equal(7400, result.FinalPromptTokens);
        Assert.Equal(7796, result.PromptLimit);
        Assert.True(result.FinalPromptTokens <= result.PromptLimit);
        Assert.True(result.PromptLimit < 8192);
    }
}
