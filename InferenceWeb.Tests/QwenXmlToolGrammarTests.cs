// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Runtime.Grammar;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="QwenXmlToolGrammar"/> (LFM3's tool_choice required / named) over a vocabulary of
/// single characters plus the call tags as USER_DEFINED tokens, so every byte of a reply is a
/// sampling decision: what the grammar admits, what it masks, and that the replies it admits
/// parse back with the family's parser. Found by eng/lfm3/e2e: a named tool_choice was answered
/// with a call to another function (llama.cpp does the same for this template).
/// </summary>
public class QwenXmlToolGrammarTests
{
    private static string Call(string name, string parameters) =>
        $"<tool_call>\n<function={name}>\n{parameters}</function>\n</tool_call>";
    private static string Param(string name, string value) => $"<parameter={name}>\n{value}\n</parameter>\n";

    private static readonly List<ToolFunction> Tools = ToolFunction.ParseList("""
        [
          {"name": "get_weather", "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}},
          {"name": "get_time", "parameters": {"type": "object",
            "properties": {"timezone": {"type": "string"}}, "required": ["timezone"]}},
          {"name": "set_timer", "parameters": {"type": "object", "properties": {
            "minutes": {"type": "integer"}, "label": {"type": "string"}, "loud": {"type": "boolean"}}, "required": ["minutes"]}},
          {"name": "create_event", "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "attendees": {"type": "array", "items": {"type": "string"}}}, "required": ["title"]}},
          {"name": "list_files", "parameters": {"type": "object", "properties": {}}}
        ]
        """);

    private static readonly string[] Tags =
        { "<tool_call>", "</tool_call>", "<function=", "</function>", "<parameter=", "</parameter>", "<think>", "</think>" };

    /// <summary>EOS, the tags as special tokens, then one token per character.</summary>
    private sealed class CharTokenizer : ITokenizer, ISpecialTokenVocabulary
    {
        public CharTokenizer()
        {
            var vocab = new List<string> { "<|im_end|>" };
            vocab.AddRange(Tags);
            vocab.Add("\n");
            for (char c = ' '; c <= '~'; c++) vocab.Add(c.ToString());
            vocab.Add("ü");
            Vocab = vocab.ToArray();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => [0];
        public int VocabSize => Vocab.Length;
        public IReadOnlyCollection<int> SpecialTokenIds => Enumerable.Range(0, Tags.Length + 1).ToArray();
        public List<int> Encode(string text, bool addSpecial = true)
        {
            var ids = new List<int>();
            for (int at = 0; at < text.Length;)
            {
                int tag = Array.FindIndex(Tags, m => string.CompareOrdinal(text, at, m, 0, m.Length) == 0);
                string piece = tag >= 0 ? Tags[tag] : text[at].ToString();
                ids.Add(LookupToken(piece));
                at += piece.Length;
            }
            return ids;
        }
        public string Decode(List<int> ids) => string.Concat(ids.Select(id => Vocab[id]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    private static readonly CharTokenizer Tokens = new();

    private static bool Allowed(GrammarConstraint constraint, int id)
        => (constraint.CurrentMask()[id >> 6] & (1UL << (id & 63))) != 0;

    private static int FirstRejected(GrammarConstraint constraint, ITokenizer tokenizer, string reply)
    {
        List<int> ids = tokenizer.Encode(reply, addSpecial: false);
        for (int i = 0; i < ids.Count; i++)
        {
            if (!Allowed(constraint, ids[i]))
                return i;
            constraint.Accept(ids[i]);
        }
        return -1;
    }

    private static void AssertAccepted(QwenXmlToolGrammar plan, string reply, bool thinking = false)
    {
        GrammarConstraint constraint = plan.NewConstraint(Tokens, thinking);
        int rejected = FirstRejected(constraint, Tokens, reply);
        Assert.True(rejected < 0, $"masked at token {rejected}: {JsonSerializer.Serialize(Tokens.Decode(Tokens.Encode(reply).Take(rejected + 1).ToList()))}");
        // EOS is never in a mask; the sampler lets it through when the grammar is complete.
        Assert.True(constraint.IsComplete && !constraint.IsDead, "the reply does not complete the grammar");
    }

    private static void AssertRejected(QwenXmlToolGrammar plan, string reply, string acceptedPrefix)
    {
        GrammarConstraint constraint = plan.NewConstraint(Tokens, false);
        int rejected = FirstRejected(constraint, Tokens, reply);
        Assert.True(rejected >= 0, $"{JsonSerializer.Serialize(reply)} was accepted");
        Assert.Equal(acceptedPrefix, Tokens.Decode(Tokens.Encode(reply).Take(rejected).ToList()));
    }

    public static IEnumerable<object[]> Replies() =>
    [
        [Call("get_weather", Param("city", "Paris"))],
        [Call("get_weather", Param("city", "a <b> & \"c\"\nZürich") + Param("unit", "celsius"))],
        [Call("set_timer", Param("minutes", "15") + Param("label", "pasta") + Param("loud", "true"))],
        [Call("create_event", Param("title", "Design review") + Param("attendees", "[\"ann@example.com\", \"raj@example.com\"]"))],
        [Call("list_files", "")],
        [Call("get_weather", Param("city", "London")) + "\n" + Call("get_time", Param("timezone", "Europe/Berlin"))],
    ];

    [Theory]
    [MemberData(nameof(Replies))]
    public void Required_AdmitsWellFormedCallsAndTheyParseBack(string calls)
    {
        var plan = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Required);
        _ = Grammar.Parse(plan.Source);
        AssertAccepted(plan, calls);
        var parser = ChatProtocolRegistry.For("lfm3")!.CreateOutputParser!();
        parser.Init(false, Tools);
        ParsedOutput parsed = parser.Add(calls, true);
        Assert.True(string.IsNullOrWhiteSpace(parsed.Content), parsed.Content);
        Assert.Equal(calls.Split("</tool_call>").Length - 1, parsed.ToolCalls!.Count);
        // Thinking on: the prompt opened the block, the reply closes it, then calls.
        AssertAccepted(plan, "The user wants it. Use the tool.</think>" + calls, thinking: true);
    }

    [Fact]
    public void Required_MasksWhatIsNotAWellFormedCall()
    {
        var plan = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Required);
        // Prose instead of a call.
        AssertRejected(plan, "Sure, I can check.", "");
        // An undeclared function.
        AssertRejected(plan, Call("patch_file", ""), "<tool_call>\n<function=");
        // An undeclared parameter (the 1.7B passed a path to list_files).
        AssertRejected(plan, Call("list_files", Param("path", "src")), "<tool_call>\n<function=list_files>\n");
        // Text where an integer goes.
        AssertRejected(plan, Call("set_timer", Param("minutes", "fifteen")), "<tool_call>\n<function=set_timer>\n<parameter=minutes>\n");
        // A required parameter left out.
        AssertRejected(plan, Call("get_weather", Param("unit", "celsius")), "<tool_call>\n<function=get_weather>\n<parameter=");
        // A string enum outside its values.
        AssertRejected(plan, Call("get_weather", Param("city", "Oslo") + Param("unit", "kelvin")),
            "<tool_call>\n<function=get_weather>\n<parameter=city>\nOslo\n</parameter>\n<parameter=unit>\n");
        // Nothing after the last call but the end of the turn.
        GrammarConstraint constraint = plan.NewConstraint(Tokens, false);
        Assert.Equal(-1, FirstRejected(constraint, Tokens, Call("list_files", "")));
        Assert.False(Allowed(constraint, Tokens.LookupToken("x")));
        Assert.True(constraint.IsComplete);
    }

    [Fact]
    public void NamedAndSingleCall_RestrictTheReply()
    {
        var named = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Named, "get_time");
        AssertAccepted(named, Call("get_time", Param("timezone", "Europe/Lisbon")));
        AssertRejected(named, Call("get_weather", Param("city", "Lisbon")), "<tool_call>\n<function=get_");

        var single = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Required, parallelToolCalls: false);
        string first = Call("get_time", Param("timezone", "UTC"));
        AssertAccepted(single, first);
        AssertRejected(single, first + "\n" + Call("get_time", Param("timezone", "UTC")), first + "\n");

        Assert.Throws<NotSupportedException>(() => QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Named, "missing"));
        Assert.Throws<NotSupportedException>(() => QwenXmlToolGrammar.Compile([], QwenXmlToolChoice.Required));
    }

    [Fact]
    public void ThinkingOn_LeavesTheReasoningFree()
    {
        var plan = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Named, "get_time");
        GrammarConstraint constraint = plan.NewConstraint(Tokens, thinking: true);
        Assert.Equal(-1, FirstRejected(constraint, Tokens, "Any reasoning at all, even <function=get_weather>, is free."));
        Assert.False(constraint.IsActive);
        Assert.Equal(-1, FirstRejected(constraint, Tokens, "</think>"));
        Assert.True(constraint.IsActive);
        Assert.False(Allowed(constraint, Tokens.LookupToken("S")));
    }

    [Fact]
    public void Adapter_CompilesOnlyRequiredOrNamedChoices()
    {
        using var auto = JsonDocument.Parse("""{"tool_choice": "auto"}""");
        Assert.Null(TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareQwenXmlToolGrammar(auto.RootElement, Tools, null));
        using var none = JsonDocument.Parse("""{"tool_choice": "none"}""");
        Assert.Null(TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareQwenXmlToolGrammar(none.RootElement, Tools, null));
        using var required = JsonDocument.Parse("""{"tool_choice": "required", "parallel_tool_calls": false}""");
        Assert.Equal(QwenXmlToolChoice.Required,
            TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareQwenXmlToolGrammar(required.RootElement, Tools, null)!.Choice);
        using var named = JsonDocument.Parse("""{"tool_choice": {"type": "function", "function": {"name": "get_time"}}}""");
        Assert.Equal(QwenXmlToolChoice.Named,
            TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareQwenXmlToolGrammar(named.RootElement, Tools, null)!.Choice);
        Assert.Throws<NotSupportedException>(() => TensorSharp.Server.ProtocolAdapters.OpenAIChatAdapter.PrepareQwenXmlToolGrammar(
            required.RootElement, Tools, StructuredOutputFormat.JsonObject()));
    }

    /// <summary>The same through the real LFM3 vocabulary, where the call tags are USER_DEFINED
    /// tokens and the rest is byte-level BPE (TS_TEST_LFM3_GGUF; only the tokenizer is read).</summary>
    [ModelFact("TS_TEST_LFM3_GGUF")]
    public void Required_AdmitsTheRealTokenizationOfWellFormedCalls()
    {
        ITokenizer tokenizer;
        using (var gguf = GgufFile.OpenWithoutSiblingShards(Environment.GetEnvironmentVariable("TS_TEST_LFM3_GGUF")!))
            tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var plan = QwenXmlToolGrammar.Compile(Tools, QwenXmlToolChoice.Required);
        foreach (object[] calls in Replies())
        {
            string reply = (string)calls[0];
            GrammarConstraint constraint = plan.NewConstraint(tokenizer, false);
            Assert.Equal(-1, FirstRejected(constraint, tokenizer, reply));
            Assert.True(constraint.IsComplete && !constraint.IsDead);
        }
        GrammarConstraint prose = plan.NewConstraint(tokenizer, false);
        Assert.False(Allowed(prose, tokenizer.Encode("Sure", false)[0]));
        Assert.True(Allowed(prose, tokenizer.LookupToken("<tool_call>")));
    }
}
