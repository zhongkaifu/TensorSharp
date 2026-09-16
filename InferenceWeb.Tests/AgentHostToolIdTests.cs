using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class AgentHostToolIdTests
{
    private const string Answer = "{\"choices\":[{\"finish_reason\":\"stop\",\"message\":{\"role\":\"assistant\",\"content\":\"done\"}}]}";

    private sealed class Transport(params string[] replies) : HttpMessageHandler
    {
        public List<string> Payloads { get; } = new();
        protected override async Task<HttpResponseMessage> SendAsync(HttpRequestMessage request, CancellationToken cancellationToken)
        {
            if (request.Method == HttpMethod.Get) return new HttpResponseMessage(HttpStatusCode.NotFound);
            Payloads.Add(await request.Content!.ReadAsStringAsync(cancellationToken));
            return new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StringContent(replies[Math.Min(Payloads.Count - 1, replies.Length - 1)], Encoding.UTF8, "application/json"),
            };
        }
    }

    private static ToolCall Call(string? id, string name = "read_file") => new() { Id = id, Name = name };

    [Fact]
    public async Task ClientPreservesExplicitParallelIdsAndOutOfOrderResults()
    {
        var calls = new List<ToolCall> { Call("opaque:second/7"), Call("call_0") };
        var history = new List<ChatMessage>
        {
            new() { Role = "assistant", ToolCalls = calls },
            new() { Role = "tool", ToolCallId = "call_0", Content = "second result" },
            new() { Role = "tool", ToolCallId = "opaque:second/7", Content = "first result" },
        };
        using var transport = new Transport(Answer);
        using var http = new HttpClient(transport);
        using var client = new SkillsChatClient(new SkillsChatClientOptions { Endpoint = "http://fixture/v1", DefaultModel = "fixture", Delivery = SkillDelivery.Server }, http);
        await client.CompleteAsync(new SkillsChatRequest { Messages = history });
        using var sent = JsonDocument.Parse(transport.Payloads.Single());
        var messages = sent.RootElement.GetProperty("messages");
        Assert.Equal("opaque:second/7", messages[0].GetProperty("tool_calls")[0].GetProperty("id").GetString());
        Assert.Equal("call_0", messages[0].GetProperty("tool_calls")[1].GetProperty("id").GetString());
        Assert.Equal("call_0", messages[1].GetProperty("tool_call_id").GetString());
        Assert.Equal("opaque:second/7", messages[2].GetProperty("tool_call_id").GetString());
        Assert.Equal(3, history.Count);
        Assert.Same(calls, history[0].ToolCalls);
    }

    [Fact]
    public async Task ClientReservesExplicitIdsBeforeMappingLegacyResultsWithoutMutatingHistory()
    {
        var legacyCall = Call(null);
        var legacyResult = new ChatMessage { Role = "tool", Content = "legacy second call" };
        var orphan = new ChatMessage { Role = "tool", Content = "unassociated legacy result" };
        var history = new List<ChatMessage>
        {
            new() { Role = "assistant", ToolCalls = [Call("call_0"), legacyCall] },
            legacyResult,
            new() { Role = "tool", ToolCallId = "call_0", Content = "explicit first call" },
            new() { Role = "user", Content = "new turn" }, orphan,
        };
        using var transport = new Transport(Answer);
        using var http = new HttpClient(transport);
        using var client = new SkillsChatClient(new SkillsChatClientOptions { Endpoint = "http://fixture/v1", DefaultModel = "fixture", Delivery = SkillDelivery.Server }, http);
        await client.CompleteAsync(new SkillsChatRequest { Messages = history });
        using var sent = JsonDocument.Parse(transport.Payloads.Single());
        var messages = sent.RootElement.GetProperty("messages");
        Assert.Equal("call_1", messages[0].GetProperty("tool_calls")[1].GetProperty("id").GetString());
        Assert.Equal("call_1", messages[1].GetProperty("tool_call_id").GetString());
        Assert.Equal("call_0", messages[2].GetProperty("tool_call_id").GetString());
        Assert.False(messages[4].TryGetProperty("tool_call_id", out _));
        Assert.Null(legacyCall.Id);
        Assert.Null(legacyResult.ToolCallId);
        Assert.Null(orphan.ToolCallId);
    }

    [Theory]
    [InlineData("server:opaque/3")]
    [InlineData(null)]
    public async Task ClientResponseRetainsExplicitIdOrLegacyAbsence(string? id)
    {
        string reply = JsonSerializer.Serialize(new { choices = new[] { new { finish_reason = "tool_calls", message = new
        {
            role = "assistant", content = "", tool_calls = new[] { new { id, type = "function", function = new { name = "lookup", arguments = "{}" } } },
        } } } });
        using var transport = new Transport(reply);
        using var http = new HttpClient(transport);
        using var client = new SkillsChatClient(new SkillsChatClientOptions { Endpoint = "http://fixture/v1", DefaultModel = "fixture", Delivery = SkillDelivery.Server }, http);
        var result = await client.CompleteAsync(SkillsChatRequest.User("lookup"));
        Assert.Equal(id, Assert.Single(result.ToolCalls).Id);
        Assert.Equal(id, Assert.Single(result.Messages[^1].ToolCalls!).Id);
    }

    [Fact]
    public async Task LocalClientKeepsResultIdsWhenUnknownToolIsAnsweredBeforeDeclaredSkill()
    {
        string directory = Path.Combine(Path.GetTempPath(), "agent-id-test-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            var registry = new SkillRegistry(new SkillRegistryOptions { Roots = [directory] });
            string calls = "{\"choices\":[{\"finish_reason\":\"tool_calls\",\"message\":{\"content\":\"\",\"tool_calls\":[{\"id\":\"list-A\",\"function\":{\"name\":\"skills_list\",\"arguments\":\"{}\"}},{\"id\":\"unknown-B\",\"function\":{\"name\":\"unknown_lookup\",\"arguments\":\"{}\"}}]}}]}";
            using var transport = new Transport(calls, Answer);
            using var http = new HttpClient(transport);
            using var client = new SkillsChatClient(new SkillsChatClientOptions
            {
                Endpoint = "http://fixture/v1", DefaultModel = "fixture", Delivery = SkillDelivery.Local, Registry = registry,
            }, http);
            var result = await client.CompleteAsync(new SkillsChatRequest { Messages = [new() { Role = "user", Content = "list" }], Tools = [] });
            Assert.Equal("done", result.Content);
            Assert.Equal(2, transport.Payloads.Count);
            using var sent = JsonDocument.Parse(transport.Payloads[1]);
            var toolResults = sent.RootElement.GetProperty("messages").EnumerateArray().Where(m => m.GetProperty("role").GetString() == "tool").ToArray();
            Assert.Equal(["unknown-B", "list-A"], toolResults.Select(m => m.GetProperty("tool_call_id").GetString()).ToArray());
            Assert.Contains("no tool called", toolResults[0].GetProperty("content").GetString());
        }
        finally { Directory.Delete(directory); }
    }

    [Fact]
    public async Task LoopRefusalAnswersEachExplicitCallAndPreservesLegacyMissingId()
    {
        var originalCalls = new List<ToolCall> { Call("first", "skills_list"), Call("second", "skills_list"), Call(null, "skills_list") };
        int rounds = 0;
        var input = new List<ChatMessage> { new() { Role = "user", Content = "list" } };
        var result = await SkillAgentLoop.RunAsync(input, null, new SkillToolContext(Array.Empty<Skill>()),
            (_, _, _) => Task.FromResult(new SkillTurnOutput(++rounds == 1 ? new ParsedOutput { ToolCalls = originalCalls } : new ParsedOutput { Content = "done" })),
            new SkillAgentLoopOptions { MaxCallsPerRound = 1 });
        var responses = result.Messages.Where(message => message.Role == "tool").ToArray();
        Assert.Equal(3, responses.Length);
        Assert.Equal("first", responses[0].ToolCallId);
        Assert.Equal("second", responses[1].ToolCallId);
        Assert.Null(responses[2].ToolCallId);
        Assert.Contains("too many tool calls", responses[1].Content);
        Assert.Single(result.Invocations);
        Assert.Single(input);
        Assert.Null(originalCalls[2].Id);
    }

    [Fact]
    public async Task RoundLimitGuidanceIsNotAnUnassociatedToolResult()
    {
        int rounds = 0;
        var result = await SkillAgentLoop.RunAsync([new() { Role = "user", Content = "list" }], null,
            new SkillToolContext(Array.Empty<Skill>()),
            (_, _, _) => Task.FromResult(new SkillTurnOutput(++rounds == 1
                ? new ParsedOutput { ToolCalls = [Call("finished-call", "skills_list")] }
                : new ParsedOutput { Content = "done" })), new SkillAgentLoopOptions { MaxRounds = 1 });
        var toolResult = Assert.Single(result.Messages.Where(message => message.Role == "tool"));
        Assert.Equal("finished-call", toolResult.ToolCallId);
        Assert.Equal("user", result.Messages[^1].Role);
        Assert.Contains("limit on tool calls", result.Messages[^1].Content);
    }
}
