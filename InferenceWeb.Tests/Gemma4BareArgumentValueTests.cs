// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Gemma 4 writes tool-call arguments in its own syntax and, for a value that looks
/// like an identifier, leaves the string BARE: <c>call:read_invoice{invoice_id:INV-472}</c>.
/// The release campaign saw that call dropped with "'I' is an invalid start of a
/// value" and the client got an empty assistant message, finish_reason=stop and no
/// tool call. These pin the conversion to JSON and the two fallbacks around it.
/// </summary>
public class Gemma4BareArgumentValueTests
{
    private static Gemma4OutputParser NewParser()
    {
        var parser = new Gemma4OutputParser();
        parser.Init(enableThinking: false, null);
        return parser;
    }

    private static ParsedOutput ParseWhole(string text)
    {
        var parser = NewParser();
        return parser.Add(text, true);
    }

    [Fact]
    public void BareIdentifierValue_IsQuoted()
    {
        Assert.Equal("{\"invoice_id\":\"INV-472\"}", Gemma4OutputParser.Gemma4ArgsToJson("{invoice_id:INV-472}"));

        var parsed = ParseWhole("<|tool_call>call:read_invoice{invoice_id:INV-472}<tool_call|>");
        ToolCall call = Assert.Single(parsed.ToolCalls);
        Assert.Equal("read_invoice", call.Name);
        Assert.Equal("INV-472", call.Arguments["invoice_id"]);
        Assert.Equal(string.Empty, parsed.Content);
    }

    [Theory]
    [InlineData("{path:src/main.py}", "src/main.py")]
    [InlineData("{name:my-file.v2.txt}", "my-file.v2.txt")]
    [InlineData("{ when : 12:30 }", "12:30")]
    [InlineData("{city:New York}", "New York")]
    public void BareValuesWithDashesDotsSlashesAndSpaces_StayWhole(string args, string expected)
    {
        using var doc = JsonDocument.Parse(Gemma4OutputParser.Gemma4ArgsToJson(args));
        var prop = doc.RootElement.EnumerateObject().Single();
        Assert.Equal(JsonValueKind.String, prop.Value.ValueKind);
        Assert.Equal(expected, prop.Value.GetString());
    }

    [Fact]
    public void MixedQuotedAndBareValues_BothParse()
    {
        string json = Gemma4OutputParser.Gemma4ArgsToJson(
            "{city:<|\"|>New York, NY<|\"|>, units:celsius, note:<|\"|>a:b, {c}<|\"|>}");
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("New York, NY", doc.RootElement.GetProperty("city").GetString());
        Assert.Equal("celsius", doc.RootElement.GetProperty("units").GetString());
        Assert.Equal("a:b, {c}", doc.RootElement.GetProperty("note").GetString());
    }

    [Fact]
    public void NumbersBooleansAndNull_StayTyped()
    {
        var parsed = ParseWhole(
            "<|tool_call>call:calculate_total{unit_price:13.75, quantity:5, negative:-2, big:1e3, paid:true, off:false, none:null, zip:00501}<tool_call|>");
        ToolCall call = Assert.Single(parsed.ToolCalls);
        Assert.Equal(13.75, call.Arguments["unit_price"]);
        Assert.Equal(5L, call.Arguments["quantity"]);
        Assert.Equal(-2L, call.Arguments["negative"]);
        Assert.Equal(1000.0, call.Arguments["big"]);
        Assert.Equal(true, call.Arguments["paid"]);
        Assert.Equal(false, call.Arguments["off"]);
        Assert.Null(call.Arguments["none"]);
        // A leading zero is not a JSON number, so it is a string rather than a parse failure.
        Assert.Equal("00501", call.Arguments["zip"]);
    }

    [Fact]
    public void BareValuesInsideArraysAndNestedObjects_AreQuoted()
    {
        string json = Gemma4OutputParser.Gemma4ArgsToJson(
            "{ids:[INV-1, INV-2, 3, <|\"|>q<|\"|>], empty:[], nested:{who:alice, n:[true, x-1]}}");
        using var doc = JsonDocument.Parse(json);
        var ids = doc.RootElement.GetProperty("ids").EnumerateArray().ToList();
        Assert.Equal("INV-1", ids[0].GetString());
        Assert.Equal("INV-2", ids[1].GetString());
        Assert.Equal(3, ids[2].GetInt32());
        Assert.Equal("q", ids[3].GetString());
        Assert.Empty(doc.RootElement.GetProperty("empty").EnumerateArray());
        var nested = doc.RootElement.GetProperty("nested");
        Assert.Equal("alice", nested.GetProperty("who").GetString());
        var n = nested.GetProperty("n").EnumerateArray().ToList();
        Assert.True(n[0].GetBoolean());
        Assert.Equal("x-1", n[1].GetString());
    }

    [Fact]
    public void PlainJsonStrings_PassThroughUnchanged()
    {
        string json = Gemma4OutputParser.Gemma4ArgsToJson("{\"command\": \"ls -la, {x}\", flag: \"y\"}");
        using var doc = JsonDocument.Parse(json);
        Assert.Equal("ls -la, {x}", doc.RootElement.GetProperty("command").GetString());
        Assert.Equal("y", doc.RootElement.GetProperty("flag").GetString());
    }

    [Fact]
    public void UnparseableCall_SurfacesItsRawTextAsContent()
    {
        // No closing brace: the arguments cannot be read even after quoting.
        const string body = "call:read_invoice{invoice_id:INV-472";
        var parsed = ParseWhole("<|tool_call>" + body + "<tool_call|>");

        Assert.Null(parsed.ToolCalls);
        Assert.Equal(body, parsed.Content);
        Assert.Contains(body, parsed.ToolCallText, StringComparison.Ordinal);
    }

    [Fact]
    public void TwoStreamedCalls_CarryTheirOwnIndex()
    {
        const string text =
            "<|tool_call>call:read_invoice{invoice_id:INV-472}<tool_call|>" +
            "<|tool_call>call:calculate_total{unit_price:13.75, quantity:5}<tool_call|>";
        var parser = NewParser();
        var calls = new List<ToolCall>();
        // Three-character pieces so both calls close mid-piece at least once.
        for (int i = 0; i < text.Length; i += 3)
        {
            var delta = parser.Add(text.Substring(i, Math.Min(3, text.Length - i)), false);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
            Assert.Empty(delta.Content);
        }
        var last = parser.Add(string.Empty, true);
        if (last.ToolCalls != null) calls.AddRange(last.ToolCalls);

        Assert.Equal(2, calls.Count);
        Assert.Equal("read_invoice", calls[0].Name);
        Assert.Equal(0, calls[0].Index);
        Assert.Equal("calculate_total", calls[1].Name);
        Assert.Equal(1, calls[1].Index);
        Assert.Equal(5L, calls[1].Arguments["quantity"]);

        // A fresh turn starts numbering again.
        parser.Init(enableThinking: false, null);
        var again = parser.Add("<|tool_call>call:read_invoice{invoice_id:INV-1}<tool_call|>", true);
        Assert.Equal(0, Assert.Single(again.ToolCalls).Index);
    }
}
