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

    [Fact]
    public void AParserFreeRetry_PreservesTheTextUpdate()
    {
        var update = ChatStreamUpdate.Text("The answer");
        Assert.Equal(update, WebUiChatService.ParseRetryUpdate(update, null));
    }
}
