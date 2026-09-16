using System.Collections.Generic;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Gemma4ToolResultAssociationTests
{
    // The result lookup is the canonical Google Gemma 4 template's forward
    // scan and ID comparison. Keep it independent of TensorSharp's context
    // builder: replacing it with a positional test renderer would miss the bug.
    // Tool declarations and thinking/channel rendering are outside this test.
    private const string CanonicalResultLookup = """
        {%- set loop_messages = messages -%}
        {%- for message in loop_messages -%}
          {%- if message['role'] != 'tool' -%}
            {%- if message['role'] == 'user' -%}{{- message['content'] -}}{%- endif -%}
            {%- if message.get('tool_calls') -%}
              {%- set ns_tool_scan = namespace(stopped=false) -%}
              {%- for k in range(loop.index0 + 1, loop_messages | length) -%}
                {%- if ns_tool_scan.stopped -%}
                {%- elif loop_messages[k]['role'] != 'tool' -%}
                  {%- set ns_tool_scan.stopped = true -%}
                {%- else -%}
                  {%- set follow = loop_messages[k] -%}
                  {%- set ns_tname = namespace(name=follow.get('name') or 'unknown') -%}
                  {%- for tc in message.get('tool_calls') -%}
                    {%- if tc.get('id') == follow.get('tool_call_id') -%}
                      {%- set ns_tname.name = tc['function']['name'] -%}
                    {%- endif -%}
                  {%- endfor -%}
                  {{- '<|tool_response>response:' + ns_tname.name + '{value:<|"|>' + follow['content'] + '<|"|>}<tool_response|>' -}}
                {%- endif -%}
              {%- endfor -%}
            {%- endif -%}
          {%- endif -%}
        {%- endfor -%}
        """;

    [Fact]
    public void ExplicitIds_AssociateOutOfOrderResultsWithTheirFunctions()
    {
        var messages = Conversation(
            new[] { Call("read_file", "read-17"), Call("shell", "shell-23") },
            Result("process stdout", "shell-23"), Result("file contents", "read-17"));

        Assert.Equal("question" + Response("shell", "process stdout") + Response("read_file", "file contents"), Render(messages));
        Assert.Equal("shell-23", messages[2].ToolCallId);
        Assert.Equal("read-17", messages[1].ToolCalls![0].Id);
    }

    [Fact]
    public void ExplicitIds_ArePreservedVerbatimInTheJinjaContext()
    {
        const string template = """
            {{- messages[0]['content'] -}}
            {%- for tc in messages[1]['tool_calls'] -%}[{{- tc['id'] -}}]{%- endfor -%}
            [{{- messages[2]['tool_call_id'] -}}][{{- messages[3]['tool_call_id'] -}}]
            """;
        var messages = Conversation(new[] { Call("shell", "call-a"), Call("shell", "call-b") },
            Result("B", "call-b"), Result("A", "call-a"));

        Assert.Equal("question[call-a][call-b][call-b][call-a]",
            ChatTemplate.RenderFromGgufTemplate(template, messages, architecture: "gemma4"));
    }

    [Fact]
    public void LegacyIdlessCalls_UseResultOrderWithoutMutatingHistory()
    {
        var messages = Conversation(new[] { Call("read_file"), Call("shell") },
            Result("file contents"), Result("process stdout"));

        string expected = "question" + Response("read_file", "file contents") + Response("shell", "process stdout");
        Assert.Equal(expected, Render(messages));
        Assert.Equal(expected, Render(messages));
        Assert.All(messages[1].ToolCalls!, call => Assert.Null(call.Id));
        Assert.Null(messages[2].ToolCallId);
        Assert.Null(messages[3].ToolCallId);
    }

    [Fact]
    public void LegacySingleShellResult_KeepsItsFunctionAndStdout()
    {
        const string stdout = "exit 0 (0.01s, sandbox: bubblewrap)\n\nrelease-shell-4821\n";
        var messages = Conversation(new[] { Call("shell") }, Result(stdout));

        // This preserves the one-tool input shape. It does not assert that a
        // real model emits a final answer after receiving it.
        Assert.Equal("question" + Response("shell", stdout), Render(messages));
    }

    [Fact]
    public void MixedIds_ReserveExplicitResultsBeforeAssigningLegacyResults()
    {
        var messages = Conversation(new[] { Call("read_file", "known-read"), Call("shell") },
            Result("process stdout"), Result("file contents", "known-read"));

        Assert.Equal("question" + Response("shell", "process stdout") + Response("read_file", "file contents"), Render(messages));
    }

    [Fact]
    public void UnmatchedResults_DoNotAliasTheLastFunction()
    {
        var messages = Conversation(new[] { Call("shell", "known-shell") },
            Result("unknown explicit result", "other-id"), Result("process stdout"), Result("extra result"));

        Assert.Equal("question" + Response("unknown", "unknown explicit result")
            + Response("shell", "process stdout") + Response("unknown", "extra result"), Render(messages));
    }

    [Fact]
    public void SyntheticIds_DoNotCollideWithExplicitIds()
    {
        var messages = Conversation(new[] { Call("read_file", "__tensorsharp_render_call_0"), Call("shell") },
            Result("process stdout"), Result("file contents", "__tensorsharp_render_call_0"));

        Assert.Equal("question" + Response("shell", "process stdout") + Response("read_file", "file contents"), Render(messages));
    }

    private static string Render(List<ChatMessage> messages)
        => ChatTemplate.RenderFromGgufTemplate(CanonicalResultLookup, messages, architecture: "gemma4");

    private static ToolCall Call(string name, string? id = null)
        => new() { Name = name, Id = id };

    private static ChatMessage Result(string content, string? id = null)
        => new() { Role = "tool", Content = content, ToolCallId = id };

    private static List<ChatMessage> Conversation(ToolCall[] calls, params ChatMessage[] results)
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "question" },
            new() { Role = "assistant", Content = "", ToolCalls = new List<ToolCall>(calls) },
        };
        messages.AddRange(results);
        return messages;
    }

    private static string Response(string name, string content)
        => "<|tool_response>response:" + name + "{value:<|\"|>" + content + "<|\"|>}<tool_response|>";
}
