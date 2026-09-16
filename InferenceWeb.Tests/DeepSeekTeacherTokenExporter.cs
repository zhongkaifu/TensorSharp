// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Buffers.Binary;
using System.Diagnostics;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Test-assembly-only entry to the production text preprocessing/tokenizer path.
/// This never creates a model, execution plan, backend or native executor.
/// It is scoped to the pinned text-only, skills-disabled DeepSeek teacher corpus.
/// </summary>
internal static class DeepSeekTeacherTokenExporter
{
    internal static readonly string[] RequiredSources = {
        "InferenceWeb.Tests/DeepSeekTeacherTokenExporter.cs", "InferenceWeb.Tests/DeepSeekTeacherTokenExporterTests.cs",
        "TensorSharp.Chat/RequestParsers/ChatMessageParser.cs", "TensorSharp.Chat/RequestParsers/ToolFunctionParser.cs",
        "TensorSharp.Chat/RequestParsers/SamplingConfigParser.cs", "TensorSharp.Chat/Hosting/ServerHostingOptions.cs",
        "TensorSharp.Chat/Hosting/SamplingDefaults.cs", "TensorSharp.Chat/ChatHistoryPreparer.cs",
        "TensorSharp.Chat/ChatGenerationPipeline.cs", "TensorSharp.Server/OpenAIResponseFormatParser.cs",
        "TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.cs", "TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.DeepSeek41Tools.cs",
        "TensorSharp.Runtime/StructuredOutputs.cs", "TensorSharp.Runtime/PromptRenderer.cs", "TensorSharp.Runtime/KVCachePromptRenderer.cs",
        "TensorSharp.Runtime/ChatTemplate.cs", "TensorSharp.Runtime/ChatTemplate.DeepSeek41.cs", "TensorSharp.Runtime/ChatProtocolRegistry.cs",
        "TensorSharp.Runtime/BpeTokenizer.cs", "TensorSharp.Runtime/GgufReader.cs", "TensorSharp.Models/ModelBase.cs"
    };
    internal static readonly JsonSerializerOptions JsonOptions = new() { WriteIndented = true };
    internal sealed record EncodedPrompt(string Text, byte[] Utf8, List<int> Tokens,
        List<int>? CacheBreakpoints, string GenerationTrailingWhitespace)
    {
        public string Utf8Sha256 => Sha(Utf8);
        public string TokensI32Sha256 => Sha(TokenBytes(Tokens));
    }
    internal sealed record ExportedRequest(JsonElement ParsedHistory,
        List<ChatMessage> PreparedHistory, List<ChatMessage> FinalHistory,
        List<ToolFunction>? DeclaredTools, List<ToolFunction>? EffectiveTools,
        bool Thinking, StructuredOutputFormat? ResponseFormat, SamplingConfig Sampling,
        int RequestedMaxTokens, int EffectiveMaxTokens, int ContextLimit,
        int RemovedMessages, EncodedPrompt Original, EncodedPrompt Final);

    internal static ExportedRequest Export(JsonElement body, ITokenizer tokenizer, string? template,
        int contextLimit = 65536, int defaultMaxTokens = 20000, bool maxTokensPinned = false)
    {
        // Fail closed before ParseOpenAI can resolve a media upload or a file.
        if (!body.TryGetProperty("model", out var model) || string.IsNullOrWhiteSpace(model.GetString()))
            throw new InvalidDataException("A concrete model identity is required.");
        var wireMessages = body.GetProperty("messages");
        if (wireMessages.ValueKind != JsonValueKind.Array || wireMessages.GetArrayLength() == 0)
            throw new InvalidDataException("Nonempty messages are required.");
        foreach (var message in wireMessages.EnumerateArray())
        {
            if (message.TryGetProperty("content", out var content) && content.ValueKind is not (JsonValueKind.String or JsonValueKind.Null))
                throw new InvalidDataException("Teacher token export accepts text history only.");
            // These corpus-external features require their own production path audit.
            foreach (string name in new[] { "images", "audio", "attachments", "cache_control", "raw_output_tokens" })
                if (message.TryGetProperty(name, out _)) throw new InvalidDataException("Unsupported teacher field: " + name);
        }
        foreach (string name in new[] { "skills", "skill", "session_id", "workspace_id", "raw" })
            if (body.TryGetProperty(name, out _)) throw new InvalidDataException("Unsupported teacher request field: " + name);
        if (contextLimit <= 1) throw new ArgumentOutOfRangeException(nameof(contextLimit));
        var settings = new ServerHostingOptions("unused", null!, "unused", [], defaultMaxTokens,
            maxTokensPinned, 0, 0, 0, 0, 0, null!, null!, null!, false, null!, skillsEnabled: false);
        int maxTokens = settings.ResolveMaxTokens(SamplingConfigParser.ReadRequestedMaxTokens(body, "max_tokens", "max_completion_tokens"));
        var sampling = SamplingConfigParser.ParseOpenAI(body, settings.SamplingDefaults);
        var messages = ChatMessageParser.ParseOpenAI(wireMessages, null!, architecture: "deepseek41");
        var parsed = JsonSerializer.SerializeToElement(messages);
        var tools = ToolFunctionParser.ParseOpenAI(body);
        bool disabled = body.TryGetProperty("tool_choice", out var choice) && choice.ValueKind == JsonValueKind.String && choice.GetString() == "none";
        var effectiveTools = disabled ? null : tools;
        bool thinking = body.TryGetProperty("think", out var think) && think.GetBoolean();
        if (!OpenAIResponseFormatParser.TryParse(body, out var format, out string? error))
            throw new InvalidDataException(error);
        // The adapter's private HTTP response writer is not called. Its pure
        // compatibility guards are explicit here; prompt/parser work below is
        // performed by the exact production helpers, not a transcript renderer.
        if (format != null)
        {
            if (effectiveTools is { Count: > 0 }) throw new InvalidDataException("response_format cannot be combined with tools");
            if (thinking && (string.IsNullOrEmpty(ChatProtocolRegistry.For("deepseek41")?.ThinkingGrammarActivationTrigger)
                || Environment.GetEnvironmentVariable("TS_JSON_GRAMMAR") == "0"))
                throw new InvalidDataException("Thinking structured output requires the delayed JSON grammar.");
            var valid = StructuredOutputValidator.ValidateSchema(format);
            if (!valid.IsValid) throw new InvalidDataException(valid.ErrorMessage);
        }
        _ = OpenAIChatAdapter.PrepareDeepSeek41ToolGrammar(body, effectiveTools, effectiveTools, format);
        messages = StructuredOutputPrompt.Apply(messages, format);
        messages = ChatHistoryPreparer.PrepareHistoryForInference(messages, "deepseek41");
        messages = ChatHistoryPreparer.AugmentWithCachedRawTokens(messages, []);
        if (messages.Any(m => m.RawOutputTokens is { Count: > 0 } || m.CacheControl != null || m.ContentCacheBreakpoints is { Count: > 0 }))
            throw new InvalidDataException("Unexpected cached-token or marker path in teacher history.");
        EncodedPrompt Render(List<ChatMessage> history) => Encode(tokenizer, template, history, effectiveTools, thinking);
        var original = Render(messages);
        var window = ChatGenerationPipeline.CompactHistoryForContextBudget(messages, original.Tokens.Count,
            contextLimit, maxTokens, preserveAllInput: false, candidate => Render(candidate).Tokens.Count);
        var final = window.RemovedMessages > 0 ? Render(window.History) : original;
        return new(parsed, messages, window.History, tools, effectiveTools, thinking, format, sampling,
            maxTokens, ChatGenerationPipeline.ClampGenerationReserve(maxTokens, final.Tokens.Count, contextLimit),
            contextLimit, window.RemovedMessages, original, final);
    }

    internal static EncodedPrompt Encode(ITokenizer tokenizer, string? template, List<ChatMessage> history,
        List<ToolFunction>? tools, bool thinking)
    {
        // Record the actual string passed to the production tokenizer. Wrapping
        // the tokenizer keeps the real GgufPromptRenderer type/dispatch intact.
        var recorder = new RecordingTokenizer(tokenizer);
        var tokens = new KVCachePromptRenderer(new GgufPromptRenderer()).RenderToTokens(recorder, template!, history,
            "deepseek41", true, out var breakpoints, out var whitespace, tools, thinking);
        if (recorder.Calls.Count != 1 || !recorder.Calls[0].AddSpecial || !tokens.SequenceEqual(recorder.Calls[0].Tokens))
            throw new InvalidDataException("Prompt used an unreviewed multi-encode/splicing path.");
        string text = recorder.Calls[0].Text;
        return new(text, new UTF8Encoding(false, true).GetBytes(text), tokens, breakpoints, whitespace);
    }

    internal static byte[] TokenBytes(IReadOnlyList<int> tokens)
    {
        var bytes = new byte[checked(tokens.Count * 4)];
        for (int i = 0; i < tokens.Count; i++) BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * 4, 4), tokens[i]);
        return bytes;
    }
    internal static string Sha(byte[] bytes) => Convert.ToHexStringLower(SHA256.HashData(bytes));
    internal static bool OrderedJsonEqual(JsonElement left, JsonElement right)
    {
        if (left.ValueKind != right.ValueKind) return false;
        if (left.ValueKind == JsonValueKind.Object)
        {
            var a = left.EnumerateObject().ToArray();
            var b = right.EnumerateObject().ToArray();
            return a.Length == b.Length && a.Zip(b).All(pair => pair.First.Name == pair.Second.Name && OrderedJsonEqual(pair.First.Value, pair.Second.Value));
        }
        if (left.ValueKind == JsonValueKind.Array)
        {
            var a = left.EnumerateArray().ToArray();
            var b = right.EnumerateArray().ToArray();
            return a.Length == b.Length && a.Zip(b).All(pair => OrderedJsonEqual(pair.First, pair.Second));
        }
        return left.ValueKind == JsonValueKind.String ? left.GetString() == right.GetString() : left.GetRawText() == right.GetRawText();
    }
    internal static string FileSha(string path) { using var file = File.OpenRead(path); return Convert.ToHexStringLower(SHA256.HashData(file)); }
    internal static void Require(bool condition, string message) { if (!condition) throw new InvalidDataException(message); }
    private static byte[] Pinned(string path, string hash)
    {
        byte[] bytes = File.ReadAllBytes(path);
        Require(hash.Length == 64 && Sha(bytes) == hash, "SHA256 mismatch: " + path);
        return bytes;
    }
    private static string SafeChild(string root, string relative)
    {
        string full = Path.GetFullPath(Path.Combine(root, relative));
        Require(!Path.IsPathRooted(relative) && full.StartsWith(Path.GetFullPath(root) + Path.DirectorySeparatorChar, StringComparison.Ordinal), "Path escapes artifact root.");
        return full;
    }

    internal static GgufFile OpenVerifiedMetadata(JsonElement source, JsonElement firstShard)
    {
        VerifyMetadataBytes(source, firstShard);
        var gguf = GgufFile.OpenWithoutSiblingShards(source.GetProperty("path").GetString()!);
        try
        {
            long prefix = source.GetProperty("header_prefix_bytes").GetInt64();
            int alignment = gguf.Metadata.TryGetValue("general.alignment", out var value) ? Convert.ToInt32(value) : 32;
            Require(alignment > 0 && alignment <= 4096, "Unreviewed GGUF alignment.");
            long expectedOffset = source.GetProperty("kind").GetString() == "original-header-prefix"
                ? checked(prefix + (alignment - prefix % alignment) % alignment) : prefix;
            Require(gguf.FilePaths.Count == 1 && gguf.DataOffset == expectedOffset && gguf.GetString("general.architecture") == "deepseek41", "Unexpected shard/architecture/header boundary.");
            return gguf;
        }
        catch { gguf.Dispose(); throw; }
    }

    internal static void VerifyMetadataBytes(JsonElement source, JsonElement firstShard)
    {
        string kind = source.GetProperty("kind").GetString()!;
        Require(kind is "original-first-shard" or "original-header-prefix", "Unsupported tokenizer artifact kind; never use a rewritten GGUF header.");
        bool prefixOnly = kind == "original-header-prefix";
        string path = source.GetProperty("path").GetString()!;
        long originalBytes = source.GetProperty("file_bytes").GetInt64();
        long prefixBytes = source.GetProperty("header_prefix_bytes").GetInt64();
        string prefixSha = source.GetProperty("header_prefix_sha256").GetString()!;
        string originalSha = source.GetProperty("original_first_shard_sha256").GetString()!;
        string originalPath = prefixOnly ? source.GetProperty("original_checkpoint_path").GetString()! : path;
        Require(firstShard.GetProperty("observed_sha256").GetString() == originalSha
            && firstShard.GetProperty("expected_sha256").GetString() == originalSha
            && firstShard.GetProperty("observed_size").GetInt64() == originalBytes
            && Path.GetFileName(originalPath) == firstShard.GetProperty("path").GetString(), "First-shard identity differs from pinned plan.");
        Require(prefixBytes > 24 && prefixBytes < originalBytes && prefixBytes <= 512L * 1024 * 1024, "Invalid metadata prefix bounds.");
        using var stream = File.Open(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Require(stream.Length == (prefixOnly ? prefixBytes : originalBytes), "Tokenizer artifact size mismatch.");
        var prefix = new byte[checked((int)prefixBytes)];
        stream.ReadExactly(prefix);
        Require(Sha(prefix) == prefixSha, "Checkpoint metadata/header bytes changed.");
        if (prefixOnly)
        {
            Require(source.GetProperty("artifact_sha256").GetString() == prefixSha, "Header-only artifact must be the unchanged exact original prefix.");
            var inventory = JsonDocument.Parse(Pinned(source.GetProperty("header_inventory_path").GetString()!,
                source.GetProperty("header_inventory_sha256").GetString()!)).RootElement;
            var records = inventory.GetProperty("files").EnumerateArray().Where(r => r.GetProperty("path").GetString() == originalPath).ToArray();
            Require(records.Length == 1, "Missing or ambiguous original header inventory entry.");
            var record = records[0];
            Require(record.GetProperty("file_bytes").GetInt64() == originalBytes
                && record.GetProperty("header_bytes_read").GetInt64() == prefixBytes
                && record.GetProperty("header_sha256").GetString() == prefixSha
                && BinaryPrimitives.ReadUInt64LittleEndian(prefix.AsSpan(8, 8)) == record.GetProperty("tensor_count").GetUInt64(), "Original header inventory binding changed.");
        }
    }

    /// <summary>Explicit config-driven export. Header verification reads no tensor payload.</summary>
    internal static void Run(string configPath, string configSha)
    {
        var config = JsonDocument.Parse(Pinned(configPath, configSha)).RootElement;
        string bundlePath = config.GetProperty("bodies_path").GetString()!;
        string bundleSha = config.GetProperty("bodies_sha256").GetString()!;
        string output = config.GetProperty("output").GetString()!;
        Require(!Directory.Exists(output) && !File.Exists(output), "Output must be new.");
        var bundle = JsonDocument.Parse(Pinned(bundlePath, bundleSha)).RootElement;
        string bundleRoot = Path.GetDirectoryName(Path.GetFullPath(bundlePath))!;
        const string engineSha = "d0c7b8dbacf70d6a663d39a51a6fa5e77cd36714d70591f8dc0e8ae9fe304ea6";
        const string profileSha = "0a29cf8c4b1bd6c4567848314ec246e813f6f145ae8d7c40b14253d7f831639d";
        Require(bundle.GetProperty("engine_sha256").GetString() == engineSha
            && bundle.GetProperty("source_profile_sha256").GetString() == profileSha, "Original engine/profile identity changed.");
        _ = Pinned(Path.Combine(bundleRoot, "original-engines.py"), engineSha);
        _ = Pinned(Path.Combine(bundleRoot, "source-profile.json"), profileSha);
        var plan = JsonDocument.Parse(Pinned(Path.Combine(bundleRoot, "teacher-plan.json"), bundle.GetProperty("plan_sha256").GetString()!)).RootElement;
        var rows = bundle.GetProperty("requests").EnumerateArray().ToArray();
        var ids = rows.Select(r => r.GetProperty("id").GetString()).ToArray();
        Require(rows.Length == 113 && ids.Distinct().Count() == 113 && ids.SequenceEqual(plan.GetProperty("requests").EnumerateArray().Select(r => r.GetProperty("id").GetString())), "Original 113 ordered IDs changed.");
        var settings = bundle.GetProperty("server_settings");
        Require(settings.GetProperty("architecture").GetString() == "deepseek41" && settings.GetProperty("context_limit").GetInt32() == 65536
            && settings.GetProperty("default_max_tokens").GetInt32() == 20000 && !settings.GetProperty("max_tokens_pinned").GetBoolean()
            && !settings.GetProperty("skills_enabled").GetBoolean() && !settings.GetProperty("sampling_defaults_pinned").GetBoolean()
            && settings.GetProperty("fresh_history_tracking").GetBoolean() && !settings.GetProperty("json_grammar_disabled").GetBoolean(), "Reviewed server preprocessing settings changed.");
        Require(Environment.GetEnvironmentVariable("TS_JSON_GRAMMAR") != "0", "JSON grammar environment differs from the pinned profile.");
        var before = AuditIdentity(config);
        var source = config.GetProperty("tokenizer_source");
        var firstShard = plan.GetProperty("model").GetProperty("files")[0];
        using var gguf = OpenVerifiedMetadata(source, firstShard);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        Directory.CreateDirectory(output);
        File.WriteAllBytes(Path.Combine(output, "config.json"), Pinned(configPath, configSha));
        var results = new List<object>();
        bool compacted = false;
        for (int i = 0; i < rows.Length; i++)
        {
            var row = rows[i];
            byte[] bodyBytes = Pinned(SafeChild(bundleRoot, row.GetProperty("body_file").GetString()!), row.GetProperty("body_sha256").GetString()!);
            byte[] engineBytes = Pinned(SafeChild(bundleRoot, row.GetProperty("engine_arguments_file").GetString()!), row.GetProperty("engine_arguments_sha256").GetString()!);
            var originalRow = plan.GetProperty("requests")[i];
            Require(row.GetProperty("engine_arguments_canonical_sha256").GetString() == originalRow.GetProperty("request_canonical_sha256").GetString()
                && OrderedJsonEqual(JsonDocument.Parse(engineBytes).RootElement, originalRow.GetProperty("request")), "Engine arguments/property order differ from original plan.");
            var body = JsonDocument.Parse(bodyBytes).RootElement;
            Require(body.GetProperty("model").GetString() == bundle.GetProperty("model_name").GetString() && !body.GetProperty("think").GetBoolean(), "Original model/thinking changed.");
            var result = Export(body, tokenizer, gguf.GetString("tokenizer.chat_template"));
            string prefix = i.ToString("D3");
            File.WriteAllBytes(Path.Combine(output, prefix + ".body.json"), bodyBytes);
            File.WriteAllBytes(Path.Combine(output, prefix + ".engine.json"), engineBytes);
            File.WriteAllBytes(Path.Combine(output, prefix + ".original.utf8"), result.Original.Utf8);
            File.WriteAllBytes(Path.Combine(output, prefix + ".original.i32"), TokenBytes(result.Original.Tokens));
            File.WriteAllBytes(Path.Combine(output, prefix + ".final.utf8"), result.Final.Utf8);
            File.WriteAllBytes(Path.Combine(output, prefix + ".final.i32"), TokenBytes(result.Final.Tokens));
            var record = new { id = ids[i], row, result, prompt_i32_sha256 = result.Final.TokensI32Sha256,
                prompt_utf8_sha256 = result.Final.Utf8Sha256, position_before = 0, position_after = result.Final.Tokens.Count,
                baseline_continuation = "not captured; do not retokenize assistant API text", vocab_size = tokenizer.VocabSize };
            File.WriteAllText(Path.Combine(output, prefix + ".json"), JsonSerializer.Serialize(record, JsonOptions), new UTF8Encoding(false));
            results.Add(new { id = ids[i], prefix, prompt_tokens = result.Final.Tokens.Count, result.RemovedMessages,
                result.Final.TokensI32Sha256, result.Final.Utf8Sha256, record_sha256 = FileSha(Path.Combine(output, prefix + ".json")) });
            compacted |= result.RemovedMessages != 0;
        }
        VerifyMetadataBytes(source, firstShard);
        var after = AuditIdentity(config);
        File.WriteAllText(Path.Combine(output, "token-export.json"), JsonSerializer.Serialize(new {
            schema_version = 1, status = compacted ? "compaction-detected-original-history-not-qualified" : "tokens-exported",
            release_qualified = false, config_sha256 = configSha, bodies_sha256 = bundleSha,
            plan_sha256 = bundle.GetProperty("plan_sha256").GetString(), tokenizer_source = source,
            tokenizer = new { tokenizer.VocabSize, tokenizer.BosTokenId, tokenizer.EosTokenIds }, before, after, requests = results,
            limits = new[] { "Only metadata/header bytes were read; full checkpoint SHA is an external pinned identity, not recomputed here.",
                "No weights, backend, native Forward, continuation or HTTP pre-inference parity was executed.",
                "Compacted and original histories/tokens are both preserved; compaction fails this original-history export gate." }
        }, JsonOptions), new UTF8Encoding(false));
        Require(!compacted, "History compaction occurred; original-history export is not qualified. Raw records were retained.");
    }

    private static object AuditIdentity(JsonElement config)
    {
        string root = config.GetProperty("source_root").GetString()!;
        var sources = config.GetProperty("source_sha256");
        foreach (string path in RequiredSources) Require(sources.TryGetProperty(path, out _), "Missing reviewed source pin: " + path);
        foreach (var file in sources.EnumerateObject()) Require(FileSha(SafeChild(root, file.Name)) == file.Value.GetString(), "Source changed: " + file.Name);
        var assemblies = config.GetProperty("assembly_sha256");
        string[] required = { "InferenceWeb.Tests.dll", "TensorSharp.Chat.dll", "TensorSharp.Server.dll", "TensorSharp.Models.dll", "TensorSharp.Runtime.dll" };
        foreach (string name in required) Require(assemblies.TryGetProperty(name, out _), "Missing assembly pin: " + name);
        foreach (var file in assemblies.EnumerateObject()) Require(FileSha(SafeChild(AppContext.BaseDirectory.TrimEnd(Path.DirectorySeparatorChar), file.Name)) == file.Value.GetString(), "Assembly changed: " + file.Name);
        // Record the assemblies actually loaded into this process, not only a
        // directory containing matching unused DLLs. Other TensorSharp/AdvUtils
        // dependencies are required in the caller's assembly pin set too.
        var loaded = AppDomain.CurrentDomain.GetAssemblies().Where(a => !a.IsDynamic &&
            (a.GetName().Name!.StartsWith("TensorSharp", StringComparison.Ordinal) || a.GetName().Name is "InferenceWeb.Tests" or "AdvUtils"))
            .ToDictionary(a => Path.GetFileName(a.Location), a => a.Location);
        foreach (var pair in loaded)
            Require(assemblies.TryGetProperty(pair.Key, out var expected) && FileSha(pair.Value) == expected.GetString()
                && Path.GetFullPath(pair.Value) == Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, pair.Key)), "Loaded assembly binding changed: " + pair.Key);
        string[] native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>().Select(m => m.FileName).Where(p => Path.GetFileName(p).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase)).ToArray();
        Require(native.Length == 0, "A GgmlOps native library is mapped; exporter must remain metadata-only.");
        return new { utc = DateTimeOffset.UtcNow, pid = Environment.ProcessId, source_sha256 = sources,
            assembly_sha256 = assemblies, loaded_assemblies = loaded, native_mappings = native };
    }

    private sealed class RecordingTokenizer(ITokenizer inner) : ITokenizer
    {
        public readonly List<(string Text, bool AddSpecial, List<int> Tokens)> Calls = [];
        public List<int> Encode(string text, bool addSpecial = true)
        { var tokens = inner.Encode(text, addSpecial); Calls.Add((text, addSpecial, tokens.ToList())); return tokens; }
        public string[] Vocab => inner.Vocab;
        public int BosTokenId => inner.BosTokenId;
        public int[] EosTokenIds => inner.EosTokenIds;
        public int VocabSize => inner.VocabSize;
        public string Decode(List<int> ids) => inner.Decode(ids);
        public void AppendTokenBytes(int id, List<byte> buffer) => inner.AppendTokenBytes(id, buffer);
        public bool IsEos(int id) => inner.IsEos(id);
        public int LookupToken(string text) => inner.LookupToken(text);
    }
}

public sealed class TeacherTokenExportFactAttribute : FactAttribute
{
    public TeacherTokenExportFactAttribute()
    {
        if (Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT") != "1")
            Skip = "Tokenizer export is explicit opt-in; normal discovery opens no GGUF.";
    }
}

public sealed class DeepSeekTeacherTokenExportFixture
{
    [TeacherTokenExportFact]
    [Trait("Category", "TeacherTokenExport")]
    public void ExportPinned113RequestsMetadataOnly()
    {
        string path = Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT_CONFIG") ?? throw new InvalidOperationException("Missing export config path.");
        string sha = Environment.GetEnvironmentVariable("TS_TEACHER_TOKEN_EXPORT_CONFIG_SHA256") ?? throw new InvalidOperationException("Missing export config SHA256.");
        DeepSeekTeacherTokenExporter.Run(path, sha);
    }
}
