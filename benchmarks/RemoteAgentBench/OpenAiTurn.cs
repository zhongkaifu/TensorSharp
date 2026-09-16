using System.Net.Http.Json;
using System.Text.Json;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;

namespace RemoteAgentBench;

/// <summary>Wire transport only. Skill/file/shell execution stays in AgentHost.</summary>
internal sealed class OpenAiTurn(HttpClient client, Options options, CaseResult result)
{
    private int _round;

    public async Task<SkillTurnOutput> Generate(List<ChatMessage> messages, List<ToolFunction>? tools, CancellationToken cancellation)
    {
        int round = ++_round;
        var payload = new
        {
            model = options.Model, stream = false, temperature = 0, max_tokens = options.MaxTokens,
            think = options.Thinking,
            // Explicit caller ownership: never ask the remote server to select or
            // execute local skills. All tool declarations are client-owned there.
            skills = Array.Empty<string>(), skills_discovery = false,
            messages = messages.Select(Message).ToArray(),
            tools = (tools ?? []).Select(Tool).ToArray(),
        };
        string directory = Path.Combine(options.Output, result.Variant, result.Scenario, result.Trial);
        Directory.CreateDirectory(directory);
        string requestPath = Path.Combine(directory, $"round-{round}.request.json");
        string responsePath = Path.Combine(directory, $"round-{round}.response.json");
        await File.WriteAllTextAsync(requestPath, JsonSerializer.Serialize(payload, Program.Json), cancellation);
        var entry = new Dictionary<string, object?>
        {
            ["round"] = round, ["started_unix_ms"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            ["request"] = requestPath, ["response"] = responsePath,
        };
        result.Requests.Add(entry);
        using var response = await client.PostAsJsonAsync(options.Endpoint, payload, Program.Json, cancellation);
        string body = await response.Content.ReadAsStringAsync(cancellation);
        await File.WriteAllTextAsync(responsePath, body, cancellation);
        entry["http_status"] = (int)response.StatusCode;
        entry["finished_unix_ms"] = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
        response.EnsureSuccessStatusCode();
        using var document = JsonDocument.Parse(body);
        var choices = document.RootElement.GetProperty("choices");
        if (choices.GetArrayLength() != 1) throw new InvalidDataException("Expected exactly one completion choice");
        var choice = choices[0];
        string? finish = choice.GetProperty("finish_reason").GetString();
        result.Finish = finish;
        if (finish is not ("stop" or "tool_calls"))
            throw new InvalidDataException("Unqualified completion finish reason: " + finish);
        if (document.RootElement.TryGetProperty("usage", out var usage)) entry["usage"] = usage.Clone();
        var message = choice.GetProperty("message");
        var parsed = new ParsedOutput { Content = String(message, "content") ?? "" };
        parsed.Thinking = String(message, "reasoning_content") ?? String(message, "reasoning") ?? String(message, "thinking") ?? "";
        if (message.TryGetProperty("tool_calls", out var calls))
        {
            parsed.ToolCalls = [];
            var ids = new HashSet<string>(StringComparer.Ordinal);
            foreach (var call in calls.EnumerateArray())
            {
                string id = call.GetProperty("id").GetString() ?? "";
                if (id.Length == 0 || !ids.Add(id)) throw new InvalidDataException("Missing or duplicate response tool ID");
                var function = call.GetProperty("function");
                var arguments = function.GetProperty("arguments");
                string raw = arguments.ValueKind == JsonValueKind.String ? arguments.GetString()! : arguments.GetRawText();
                parsed.ToolCalls.Add(new ToolCall
                {
                    Id = id, Index = parsed.ToolCalls.Count, Name = function.GetProperty("name").GetString()!,
                    Arguments = JsonSerializer.Deserialize<Dictionary<string, object>>(raw) ?? throw new InvalidDataException("Tool arguments must be an object"),
                });
            }
        }
        // OpenAI does not expose exact prompt/output token arrays. This topology
        // therefore cannot claim the local raw-token KV splice performance path.
        return new SkillTurnOutput(parsed);
    }

    private static string? String(JsonElement value, string name) =>
        value.TryGetProperty(name, out var property) && property.ValueKind == JsonValueKind.String ? property.GetString() : null;

    private static Dictionary<string, object?> Message(ChatMessage message)
    {
        var result = new Dictionary<string, object?> { ["role"] = message.Role, ["content"] = message.Content ?? "" };
        if (!string.IsNullOrEmpty(message.Thinking)) result["reasoning_content"] = message.Thinking;
        if (message.Role == "tool")
        {
            // Do not conceal a loop ID-loss defect by reassigning results here.
            if (string.IsNullOrEmpty(message.ToolCallId)) throw new InvalidDataException("AgentHost lost the tool result ID");
            result["tool_call_id"] = message.ToolCallId;
        }
        if (message.ToolCalls is { Count: > 0 })
            result["tool_calls"] = message.ToolCalls.Select(call => new
            {
                id = call.Id ?? throw new InvalidDataException("AgentHost lost the assistant tool ID"), type = "function",
                function = new { name = call.Name, arguments = JsonSerializer.Serialize(call.Arguments, Program.Json) },
            }).ToArray();
        return result;
    }

    private static object Tool(ToolFunction tool) => new
    {
        type = "function",
        function = new
        {
            name = tool.Name, description = tool.Description,
            parameters = new
            {
                type = "object", required = tool.Required ?? [],
                properties = (tool.Parameters ?? []).ToDictionary(pair => pair.Key, pair =>
                {
                    var schema = new Dictionary<string, object?> { ["type"] = pair.Value.Type, ["description"] = pair.Value.Description };
                    if (pair.Value.Enum is { Count: > 0 }) schema["enum"] = pair.Value.Enum;
                    return schema;
                }),
            },
        },
    };
}
