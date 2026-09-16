using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public class Qwen4ExpOutputParserTests
{
    // Actual Qwen3.8 Flash Next response from the three-A40 release campaign.
    private const string WeatherCall = "\n\n<tool_call>\n<function=get_weather>\n" +
        "<parameter=city>\nParis\n</parameter>\n<parameter=units>\ncelsius\n</parameter>\n" +
        "</function>\n</tool_call>";

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(false, 7)]
    [InlineData(false, 1024)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    [InlineData(true, 7)]
    [InlineData(true, 1024)]
    public void ActualModelCall_BecomesStructuredAcrossChunkBoundaries(bool thinking, int chunk)
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(thinking, new List<ToolFunction>());
        string raw = (thinking ? "Choose the weather function.</think>" : "") + WeatherCall;
        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new List<ToolCall>();
        void Collect(ParsedOutput delta)
        {
            content.Append(delta.Content);
            reasoning.Append(delta.Thinking);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
        }
        for (int i = 0; i < raw.Length; i += chunk)
            Collect(parser.Add(raw.Substring(i, Math.Min(chunk, raw.Length - i)), false));
        Collect(parser.Add("", true));

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal("celsius", call.Arguments["units"]);
        Assert.Equal(0, call.Index);
        Assert.True(string.IsNullOrWhiteSpace(content.ToString()));
        Assert.Equal(thinking ? "Choose the weather function." : "", reasoning.ToString());
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void NonStreamingCollector_ExtractsTheSameCall(bool thinking)
    {
        var collector = new ChatStreamCollector();
        collector.Add(new ChatStreamUpdate { Piece = (thinking ? "Use weather.</think>" : "") + WeatherCall });
        var output = collector.Resolve("qwen4exp", thinking, new List<ToolFunction>());
        Assert.Equal("get_weather", Assert.Single(output.ToolCalls!).Name);
        Assert.True(string.IsNullOrWhiteSpace(output.Content));
    }

    [Fact]
    public void ReusedParser_ResetDoesNotCarryToolStateIntoTheNextAnswer()
    {
        var parser = OutputParserFactory.Create("qwen4exp");
        parser.Init(false, null);
        Assert.Equal("get_weather", Assert.Single(parser.Add(WeatherCall, true).ToolCalls!).Name);
        parser.Init(false, null);
        var answer = parser.Add("Paris is 18 degrees Celsius.", true);
        Assert.Equal("Paris is 18 degrees Celsius.", answer.Content);
        Assert.Null(answer.ToolCalls);
    }
}
