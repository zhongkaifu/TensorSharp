using System.Text;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public class WebUiRetryParserTests
{
    [Fact]
    public void ThinkingOffRetry_PrimesBeforeStreamingAnyThoughtText()
    {
        IOutputParser parser = new Gemma4OutputParser();
        parser.Init(false, null);
        var metadata = ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<|channel>thought\n" };
        var initial = WebUiChatService.ParseRetryUpdate(metadata, parser);
        Assert.Empty(initial.Piece);

        foreach (char ch in "Inspecting the tool result before answering.")
        {
            var delta = WebUiChatService.ParseRetryUpdate(ChatStreamUpdate.Text(ch.ToString()), parser);
            Assert.Empty(delta.Piece);
            Assert.Empty(delta.ThinkingPiece);
        }

        var answer = new StringBuilder();
        foreach (char ch in "<channel|>42")
            answer.Append(WebUiChatService.ParseRetryUpdate(ChatStreamUpdate.Text(ch.ToString()), parser).Piece);
        answer.Append(parser.Add("", true).Content);
        Assert.Equal("42", answer.ToString());
    }

    [Fact]
    public void ClosedPromptChannel_LeavesTheRetryAnswerVisible()
    {
        IOutputParser parser = new Gemma4OutputParser();
        parser.Init(false, null);
        var update = ChatStreamUpdate.Text("The answer") with
        {
            RawGenerationSuffix = "<|channel>thought\n<channel|>",
        };

        Assert.Equal("The answer", WebUiChatService.ParseRetryUpdate(update, parser).Piece);
    }

    [Fact]
    public void AlreadyParsedSkillUpdates_AreNotParsedAgain()
    {
        IOutputParser parser = new Gemma4OutputParser();
        parser.Init(false, null);
        parser.SetGenerationPromptSuffix("<|channel>thought\n");
        var calls = new[] { new ToolCall { Name = "client_tool" } };
        var answer = ChatStreamUpdate.Parsed("Literal <channel|> answer", "Separated thought", calls);
        var progress = ChatStreamUpdate.ToolProgress("running", "shell", "output", 2.5, "python");

        Assert.Equal(answer, WebUiChatService.ParseRetryUpdate(answer, parser));
        Assert.Equal(progress, WebUiChatService.ParseRetryUpdate(progress, parser));
    }

    /// <summary>
    /// The retry runs with thinking off, which is exactly when Nemotron-H Reasoning-128K
    /// reasons past its closed block. The retry's parser is primed by the pipeline's first
    /// update and takes the reasoning back once its close arrives.
    /// </summary>
    [Fact]
    public void ThinkingOffRetry_TakesStrayReasoningBack()
    {
        IOutputParser parser = ChatProtocolRegistry.For("nemotron_h")!.CreateOutputParser!();
        parser.Init(false, null);
        parser.AcceptRetractions();
        Assert.Empty(WebUiChatService.ParseRetryUpdate(
            ChatStreamUpdate.Text("") with { RawGenerationSuffix = "<think></think>" }, parser).Piece);

        var visible = new StringBuilder();
        var frames = new List<string>();
        string reply = NemotronHLoggedReplies.ToolRoundWithAnswer;
        bool retracted = false;
        foreach (char ch in reply)
        {
            ChatStreamUpdate update = WebUiChatService.ParseRetryUpdate(ChatStreamUpdate.Text(ch.ToString()), parser);
            retracted |= !string.IsNullOrEmpty(update.RetractedPiece);
            foreach (object frame in WebUiChatService.AnswerFrames(visible, update.Piece, update.ThinkingPiece, update.RetractedPiece))
                frames.Add(System.Text.Json.JsonSerializer.Serialize(frame));
        }

        Assert.True(retracted);
        Assert.StartsWith("The file 'notes.txt' has **7 lines**", visible.ToString().Trim());
        Assert.Contains(frames, f => f.StartsWith("{\"replace\":", StringComparison.Ordinal));
    }

    [Fact]
    public void ARetraction_IsSentAsItsReasoningThenTheWholeAnswer()
    {
        var visible = new StringBuilder("Round one.\n\nOkay, plan");
        var frames = WebUiChatService.AnswerFrames(visible, "Seven.", "Okay, plan", "Okay, plan")
            .Select(f => System.Text.Json.JsonSerializer.Serialize(f)).ToList();

        Assert.Equal(new[]
        {
            "{\"thinking\":\"Okay, plan\"}",
            "{\"replace\":\"Round one.\\n\\n\"}",
            "{\"token\":\"Seven.\"}",
        }, frames);
        Assert.Equal("Round one.\n\nSeven.", visible.ToString());

        // Text that is not what the page shows is never cut from it.
        visible = new StringBuilder("Something else");
        frames = WebUiChatService.AnswerFrames(visible, null, "Okay, plan", "Okay, plan")
            .Select(f => System.Text.Json.JsonSerializer.Serialize(f)).ToList();
        Assert.Equal(new[] { "{\"thinking\":\"Okay, plan\"}" }, frames);
        Assert.Equal("Something else", visible.ToString());
    }

    [Fact]
    public void AParserFreeRetry_PreservesTheTextUpdate()
    {
        var update = ChatStreamUpdate.Text("The answer");
        Assert.Equal(update, WebUiChatService.ParseRetryUpdate(update, null));
    }
}
