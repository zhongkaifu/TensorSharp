// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class DeepSeekTeacherTokenExporterTests
{
    private static JsonElement Body(string json) => JsonDocument.Parse(json).RootElement;
    private static DeepSeekTeacherTokenExporter.ExportedRequest Export(string json, int context = 65536,
        int maxTokens = 20000, bool pinned = false) => DeepSeekTeacherTokenExporter.Export(Body(json), new Utf8ByteTokenizer(),
            "deliberately invalid inherited Jinja {{", context, maxTokens, pinned);

    [Fact]
    public void PropertyOrderGuardRejectsSchemaReorderingButAcceptsEquivalentStringEscapes()
    {
        Assert.True(DeepSeekTeacherTokenExporter.OrderedJsonEqual(Body("{\"z\":\"caf\\u00e9\",\"a\":[1,2]}"), Body("{\"z\":\"caf\u00e9\",\"a\":[1,2]}")));
        Assert.False(DeepSeekTeacherTokenExporter.OrderedJsonEqual(Body("{\"z\":{},\"a\":{}}"), Body("{\"a\":{},\"z\":{}}")));
        Assert.False(DeepSeekTeacherTokenExporter.OrderedJsonEqual(Body("{\"a\":[1,2]}"), Body("{\"a\":[2,1]}")));
    }

    [Fact]
    public void ToolSchemaPropertyOrderReachesTheExactRenderedPrompt()
    {
        var result = Export("""
        {"model":"fixture","messages":[{"role":"user","content":"Use tool."}],
        "tools":[{"type":"function","function":{"name":"pick","parameters":{"type":"object","properties":{"zebra":{"type":"integer"},"apple":{"type":"string"}},"required":["zebra","apple"]}}}]}
        """);
        Assert.Contains("\"properties\":{\"zebra\":{\"type\":\"integer\"},\"apple\":{\"type\":\"string\"}}", result.Final.Text.Replace(" ", ""));
    }

    [Fact]
    public void ExactHeaderArtifactRetainsTensorTableAndBuildsOnlyTokenizerWithoutPayload()
    {
        using var fixture = new HeaderFixture();
        using var gguf = DeepSeekTeacherTokenExporter.OpenVerifiedMetadata(fixture.Source, fixture.FirstShard);
        Assert.Single(gguf.Tensors);
        Assert.Contains("unused.weight", gguf.Tensors.Keys);
        Assert.True(gguf.GetRequiredLength(out _) > new FileInfo(fixture.Path).Length);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        Assert.Equal(3, tokenizer.VocabSize);
        Assert.Equal(new[] { 2 }, tokenizer.EosTokenIds);
        Assert.Equal(new[] { 0 }, tokenizer.Encode("a"));
        DeepSeekTeacherTokenExporter.VerifyMetadataBytes(fixture.Source, fixture.FirstShard);
    }

    [Fact]
    public void HeaderArtifactRejectsChangedBytesEvenWhenNewArtifactHashIsSupplied()
    {
        using var fixture = new HeaderFixture();
        byte[] bytes = File.ReadAllBytes(fixture.Path);
        bytes[8] = 0; // A tensor_count=0 rewrite is not the original header.
        File.WriteAllBytes(fixture.Path, bytes);
        var source = System.Text.Json.Nodes.JsonNode.Parse(fixture.Source.GetRawText())!;
        source["artifact_sha256"] = DeepSeekTeacherTokenExporter.Sha(bytes);
        source["header_prefix_sha256"] = DeepSeekTeacherTokenExporter.Sha(bytes);
        Assert.Throws<InvalidDataException>(() => DeepSeekTeacherTokenExporter.OpenVerifiedMetadata(Body(source.ToJsonString()), fixture.FirstShard));
    }

    [Fact]
    public void HeaderArtifactRejectsAppendedPayloadAndWrongCheckpointIdentity()
    {
        using var fixture = new HeaderFixture();
        var wrong = System.Text.Json.Nodes.JsonNode.Parse(fixture.FirstShard.GetRawText())!;
        wrong["observed_sha256"] = new string('b', 64);
        Assert.Throws<InvalidDataException>(() => DeepSeekTeacherTokenExporter.OpenVerifiedMetadata(fixture.Source, Body(wrong.ToJsonString())));
        using (var append = File.Open(fixture.Path, FileMode.Append)) append.WriteByte(0);
        Assert.Throws<InvalidDataException>(() => DeepSeekTeacherTokenExporter.OpenVerifiedMetadata(fixture.Source, fixture.FirstShard));
    }

    [Fact]
    public void ExactRenderedUtf8AndTokenBytesUseOwnRendererAndKeepUnicodeAndWhitespace()
    {
        const string content = " \r\nCaf\u00e9 \u6771\u4eac \ud83e\udd8a\n ";
        string body = JsonSerializer.Serialize(new { model = "fixture", messages = new[] { new { role = "user", content } }, max_tokens = 256, think = false });
        var result = Export(body);
        string expected = "<｜begin▁of▁sentence｜><｜User｜>" + content + "<｜Assistant｜></think>";
        Assert.Equal(expected, result.Final.Text);
        byte[] utf8 = Encoding.UTF8.GetBytes(expected);
        Assert.Equal(utf8, result.Final.Utf8);
        Assert.Equal(utf8.Select(b => (int)b), result.Final.Tokens);
        var tokenBytes = utf8.SelectMany(b => new byte[] { b, 0, 0, 0 }).ToArray();
        Assert.Equal(tokenBytes, DeepSeekTeacherTokenExporter.TokenBytes(result.Final.Tokens));
        Assert.Equal(DeepSeekTeacherTokenExporter.Sha(tokenBytes), result.Final.TokensI32Sha256);
        Assert.Equal(0, result.RemovedMessages);
        Assert.Equal(256, result.EffectiveMaxTokens);
    }

    [Fact]
    public void ExplicitThinkingChangesGenerationBoundaryThroughProductionRenderer()
    {
        var result = Export("""{"model":"fixture","messages":[{"role":"user","content":"Compute."}],"think":true}""");
        Assert.True(result.Thinking);
        Assert.EndsWith("<｜Assistant｜><think>", result.Final.Text);
        Assert.Contains("Reasoning Effort: 50", result.Final.Text);
    }

    [Fact]
    public void NoneRemovesCatalogButPreservesExplicitOutOfOrderToolHistoryAndJsonInstruction()
    {
        var result = Export("""
        {"model":"fixture","tool_choice":"none","response_format":{"type":"json_object"},
         "tools":[{"type":"function","function":{"name":"lookup","parameters":{"type":"object","properties":{}}}}],
         "messages":[{"role":"user","content":"Read both."},
          {"role":"assistant","content":null,"tool_calls":[
           {"id":"a","type":"function","function":{"name":"lookup","arguments":"{}"}},
           {"id":"b","type":"function","function":{"name":"lookup","arguments":"{}"}}]},
          {"role":"tool","tool_call_id":"b","content":"SECOND"},
          {"role":"tool","tool_call_id":"a","content":"FIRST"}]}
        """);
        Assert.Single(result.DeclaredTools!);
        Assert.Null(result.EffectiveTools);
        Assert.Equal(5, result.PreparedHistory.Count);
        Assert.Equal("b", result.PreparedHistory[3].ToolCallId);
        Assert.Equal("a", result.PreparedHistory[4].ToolCallId);
        Assert.Contains("Return exactly one JSON object and nothing else.", result.Final.Text);
        Assert.Contains("<tool_result>FIRST</tool_result>\n\n<tool_result>SECOND</tool_result>", result.Final.Text);
        Assert.Contains("<｜DSML｜ invoke name=\"lookup\">", result.Final.Text);
        Assert.DoesNotContain("You MUST strictly follow the above defined tool name", result.Final.Text);
    }

    [Fact]
    public void JsonSchemaIsInjectedAndOriginalSystemTextAndRequestArePreserved()
    {
        const string input = """
        {"model":"fixture","messages":[{"role":"system","content":"Original policy.  "},{"role":"user","content":"Answer."}],
        "response_format":{"type":"json_schema","json_schema":{"name":"answer","strict":true,"schema":{"type":"object","properties":{"x":{"type":"integer"}},"required":["x"],"additionalProperties":false}}}}
        """;
        var result = Export(input);
        Assert.Equal(StructuredOutputKind.JsonSchema, result.ResponseFormat!.Kind);
        Assert.Equal("Original policy.  ", result.ParsedHistory[0].GetProperty("Content").GetString());
        Assert.StartsWith("Original policy.", result.PreparedHistory[0].Content);
        Assert.Contains("\"additionalProperties\":false", result.Final.Text.Replace(" ", ""));
    }

    [Theory]
    [InlineData("required")]
    [InlineData("named")]
    public void RequiredAndNamedToolChoicesUseProductionGrammarValidation(string choice)
    {
        string toolChoice = choice == "named" ? "{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}" : "\"required\"";
        var result = Export("{\"model\":\"fixture\",\"messages\":[{\"role\":\"user\",\"content\":\"Use tool.\"}],\"parallel_tool_calls\":false,\"tool_choice\":" + toolChoice +
            ",\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"type\":\"object\",\"properties\":{}}}}]}");
        Assert.Single(result.EffectiveTools!);
        Assert.Contains("You MUST strictly follow the above defined tool name", result.Final.Text);
    }

    [Theory]
    [InlineData("\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"x\"}}],\"response_format\":{\"type\":\"json_object\"}")]
    [InlineData("\"tool_choice\":\"required\"")]
    [InlineData("\"parallel_tool_calls\":\"false\"")]
    [InlineData("\"skills\":[\"x\"]")]
    public void InvalidOrOutOfScopeRequestIsRejected(string extra)
        => Assert.ThrowsAny<Exception>(() => Export("{\"model\":\"fixture\",\"messages\":[{\"role\":\"user\",\"content\":\"x\"}]," + extra + "}"));

    [Fact]
    public void MediaIsRejectedBeforeUploadResolution()
        => Assert.Throws<InvalidDataException>(() => Export("""{"model":"fixture","messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"file:///must-not-open"}}]}]}"""));

    [Fact]
    public void ContextCompactionRecordsBothHistoriesAndTokensWithoutSilentSubstitution()
    {
        string input = JsonSerializer.Serialize(new { model = "fixture", max_tokens = 100,
            messages = new[] { new { role = "system", content = "policy" }, new { role = "user", content = new string('x', 1000) },
                new { role = "assistant", content = "old answer" }, new { role = "user", content = "latest" } } });
        var result = Export(input, context: 512);
        Assert.Equal(2, result.RemovedMessages);
        Assert.Equal(4, result.PreparedHistory.Count);
        Assert.Equal(2, result.FinalHistory.Count);
        Assert.Contains(new string('x', 1000), result.Original.Text);
        Assert.DoesNotContain(new string('x', 1000), result.Final.Text);
        Assert.Contains("policy", result.Final.Text);
        Assert.Contains("latest", result.Final.Text);
        Assert.NotEqual(result.Original.TokensI32Sha256, result.Final.TokensI32Sha256);
    }

    [Fact]
    public void ProtectedPromptOverflowRejectsAndPinnedGenerationBudgetIsHonored()
    {
        string body = JsonSerializer.Serialize(new { model = "fixture", max_completion_tokens = 99,
            messages = new[] { new { role = "user", content = "latest" } } });
        Assert.Equal(23, Export(body, maxTokens: 23, pinned: true).RequestedMaxTokens);
        Assert.Equal(99, Export(body, maxTokens: 23, pinned: false).RequestedMaxTokens);
        Assert.Throws<PromptContextOverflowException>(() => Export(body, context: 32));
    }

    [TeacherBodyCorpusFact]
    public void All113PinnedBodiesTraverseProductionPreprocessingWithSyntheticTokenizer()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEACHER_BODY_CORPUS")!;
        string expected = Environment.GetEnvironmentVariable("TS_TEACHER_BODY_CORPUS_SHA256")!;
        Assert.Equal(expected, DeepSeekTeacherTokenExporter.FileSha(path));
        var corpus = JsonDocument.Parse(File.ReadAllBytes(path)).RootElement;
        string root = Path.GetDirectoryName(path)!;
        var plan = JsonDocument.Parse(File.ReadAllBytes(Path.Combine(root, "teacher-plan.json"))).RootElement;
        Assert.Equal(corpus.GetProperty("plan_sha256").GetString(), DeepSeekTeacherTokenExporter.FileSha(Path.Combine(root, "teacher-plan.json")));
        var ids = new HashSet<string>();
        int jsonObject = 0, jsonSchema = 0;
        foreach (var row in corpus.GetProperty("requests").EnumerateArray())
        {
            Assert.True(ids.Add(row.GetProperty("id").GetString()!));
            var original = plan.GetProperty("requests")[ids.Count - 1];
            Assert.Equal(original.GetProperty("id").GetString(), row.GetProperty("id").GetString());
            string enginePath = Path.Combine(root, row.GetProperty("engine_arguments_file").GetString()!);
            Assert.Equal(row.GetProperty("engine_arguments_sha256").GetString(), DeepSeekTeacherTokenExporter.FileSha(enginePath));
            Assert.Equal(original.GetProperty("request_canonical_sha256").GetString(), row.GetProperty("engine_arguments_canonical_sha256").GetString());
            Assert.True(DeepSeekTeacherTokenExporter.OrderedJsonEqual(original.GetProperty("request"), JsonDocument.Parse(File.ReadAllBytes(enginePath)).RootElement));
            string bodyPath = Path.Combine(root, row.GetProperty("body_file").GetString()!);
            Assert.Equal(row.GetProperty("body_sha256").GetString(), DeepSeekTeacherTokenExporter.FileSha(bodyPath));
            string raw = File.ReadAllText(bodyPath, Encoding.UTF8);
            // This is parser/render coverage, not actual model token counts:
            // the synthetic byte vocabulary needs a larger artificial window.
            var result = Export(raw, context: 2_000_000);
            Assert.False(result.Thinking);
            Assert.Equal(0, result.RemovedMessages);
            Assert.NotEmpty(result.Final.Tokens);
            if (result.ResponseFormat?.Kind == StructuredOutputKind.JsonObject) jsonObject++;
            if (result.ResponseFormat?.Kind == StructuredOutputKind.JsonSchema) jsonSchema++;
            Assert.Equal(raw, File.ReadAllText(bodyPath, Encoding.UTF8));
        }
        Assert.Equal(113, ids.Count);
        Assert.Equal(20, jsonObject);
        Assert.Equal(5, jsonSchema);
    }

    internal sealed class Utf8ByteTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 256;
        public int BosTokenId => 0;
        public int[] EosTokenIds => [];
        public List<int> Encode(string text, bool addSpecial = true) => Encoding.UTF8.GetBytes(text).Select(b => (int)b).ToList();
        public string Decode(List<int> ids) => Encoding.UTF8.GetString(ids.Select(i => (byte)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.Add((byte)tokenId);
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string text) => -1;
    }

    private sealed class HeaderFixture : IDisposable
    {
        private readonly string directory = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "teacher-header-" + Guid.NewGuid().ToString("N"));
        public string Path { get; }
        public JsonElement Source { get; }
        public JsonElement FirstShard { get; }
        public HeaderFixture()
        {
            Directory.CreateDirectory(directory);
            Path = System.IO.Path.Combine(directory, "synthetic.header.gguf");
            using (var stream = File.Create(Path))
            using (var writer = new BinaryWriter(stream, Encoding.UTF8))
            {
                void String(string value) { byte[] bytes = Encoding.UTF8.GetBytes(value); writer.Write((ulong)bytes.Length); writer.Write(bytes); }
                void Key(string name, uint type) { String(name); writer.Write(type); }
                writer.Write(Encoding.ASCII.GetBytes("GGUF")); writer.Write(3u); writer.Write(1ul); writer.Write(7ul);
                Key("general.architecture", 8); String("deepseek41");
                Key("tokenizer.ggml.model", 8); String("gpt2");
                Key("tokenizer.ggml.tokens", 9); writer.Write(8u); writer.Write(3ul); String("a"); String("b"); String("</s>");
                Key("tokenizer.ggml.token_type", 9); writer.Write(5u); writer.Write(3ul); writer.Write(1); writer.Write(1); writer.Write(3);
                Key("tokenizer.ggml.merges", 9); writer.Write(8u); writer.Write(0ul);
                Key("tokenizer.ggml.bos_token_id", 4); writer.Write(0u);
                Key("tokenizer.ggml.eos_token_id", 4); writer.Write(2u);
                String("unused.weight"); writer.Write(1u); writer.Write(4ul); writer.Write(0u); writer.Write(0ul);
            }
            long prefix = new FileInfo(Path).Length;
            long originalBytes = prefix + 4096;
            string prefixSha = DeepSeekTeacherTokenExporter.FileSha(Path);
            string originalPath = "/synthetic-only/checkpoint.gguf";
            string inventoryPath = System.IO.Path.Combine(directory, "inventory.json");
            File.WriteAllText(inventoryPath, JsonSerializer.Serialize(new { files = new[] { new {
                path = originalPath, file_bytes = originalBytes, header_bytes_read = prefix, header_sha256 = prefixSha, tensor_count = 1 } } }));
            FirstShard = JsonSerializer.SerializeToElement(new { path = "checkpoint.gguf", observed_size = originalBytes,
                observed_sha256 = new string('a', 64), expected_sha256 = new string('a', 64) });
            Source = JsonSerializer.SerializeToElement(new { kind = "original-header-prefix", path = Path, file_bytes = originalBytes,
                original_first_shard_sha256 = new string('a', 64), original_checkpoint_path = originalPath,
                header_prefix_bytes = prefix, header_prefix_sha256 = prefixSha, artifact_sha256 = prefixSha,
                header_inventory_path = inventoryPath, header_inventory_sha256 = DeepSeekTeacherTokenExporter.FileSha(inventoryPath) });
        }
        public void Dispose() => Directory.Delete(directory, recursive: true);
    }
}

public sealed class TeacherBodyCorpusFactAttribute : FactAttribute
{
    public TeacherBodyCorpusFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable("TS_TEACHER_BODY_CORPUS")))
            Skip = "Explicit pinned, model-free body corpus path required.";
    }
}
