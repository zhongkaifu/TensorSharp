// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// What an agent client that runs its own tool loop (pi, issue #285) needs from the
/// server: the developer role rendered as instructions, none of the host's tools unless
/// asked for, the thinking switches such clients send, an operator chat template file,
/// and warm-up log lines that do not read as a client request cut to one token.
/// </summary>
public sealed class ClientToolLoopCompatibilityTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-client-loop-" + Guid.NewGuid().ToString("N"));

    public ClientToolLoopCompatibilityTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    // The template the Qwen 3.5 GGUFs ship (here: sky7350/Mica-v0.1-4B's chat_template.jinja,
    // byte-identical to its Q4_K_M GGUF's tokenizer.chat_template).
    private static string Qwen35Template()
    {
        using Stream stream = typeof(ClientToolLoopCompatibilityTests).Assembly
            .GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Qwen35.chat_template.jinja")!;
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    private static readonly List<ToolFunction> ReadTool = new()
    {
        new ToolFunction
        {
            Name = "read",
            Description = "Read a file",
            Parameters = new Dictionary<string, ToolParameter> { ["path"] = new ToolParameter { Type = "string" } },
            Required = new List<string> { "path" },
        },
    };

    private static List<ChatMessage> Conversation(string instructionRole) => new()
    {
        new ChatMessage { Role = instructionRole, Content = "You are a coding assistant running inside pi." },
        new ChatMessage { Role = "user", Content = "hello" },
    };

    // ---- developer role ---------------------------------------------------

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Qwen35Template_RendersADeveloperMessageExactlyAsASystemMessage(bool withTools)
    {
        string template = Qwen35Template();
        List<ToolFunction> tools = withTools ? ReadTool : null;

        string developer = ChatTemplate.RenderFromGgufTemplate(template, Conversation("developer"), true, "qwen35", tools);
        string system = ChatTemplate.RenderFromGgufTemplate(template, Conversation("system"), true, "qwen35", tools);

        // pi's instructions used to vanish: the template raised "Unexpected message role"
        // for a developer message, and this engine renders a raise as nothing.
        Assert.Contains("You are a coding assistant running inside pi.", developer, StringComparison.Ordinal);
        Assert.Equal(system, developer);
    }

    [Fact]
    public void ATemplateThatNamesTheDeveloperRole_ReceivesIt()
    {
        const string template =
            "{%- for m in messages %}{%- if m.role == 'developer' %}[DEV]{{ m.content }}{%- else %}[{{ m.role }}]{{ m.content }}{%- endif %}{%- endfor %}";

        string rendered = ChatTemplate.RenderFromGgufTemplate(template, Conversation("developer"), true, "not-a-family");

        Assert.Contains("[DEV]You are a coding assistant", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void TheGenericChatMlRenderer_RendersDeveloperAsSystem()
    {
        string rendered = ChatTemplate.RenderFromGgufTemplate(null!, Conversation("developer"), true, "not-a-family");

        Assert.Contains("<|im_start|>system\nYou are a coding assistant", rendered, StringComparison.Ordinal);
        Assert.DoesNotContain("<|im_start|>developer", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void AFamilyThatGivesDeveloperItsOwnMeaning_KeepsIt()
    {
        // GLM renders a developer message as a user turn, as its template does.
        Assert.True(ChatProtocolRegistry.For("glm-dsa")!.RendersDeveloperRole);
        string rendered = ChatTemplate.RenderFromGgufTemplate(null!, Conversation("developer"), true, "glm-dsa");

        Assert.Contains("<|user|>You are a coding assistant", rendered, StringComparison.Ordinal);
        Assert.DoesNotContain("<|system|>You are a coding assistant", rendered, StringComparison.Ordinal);
    }

    [Fact]
    public void MapDeveloperRoleToSystem_CopiesWithoutTouchingTheCaller()
    {
        List<ChatMessage> plain = Conversation("system");
        Assert.Same(plain, ChatTemplate.MapDeveloperRoleToSystem(plain));

        var original = new MarkedMessage { Role = "developer", Content = "x", Thinking = "t" };
        var messages = new List<ChatMessage> { original, new() { Role = "user", Content = "y" } };
        List<ChatMessage> mapped = ChatTemplate.MapDeveloperRoleToSystem(messages);

        Assert.NotSame(messages, mapped);
        Assert.Equal("developer", original.Role);
        Assert.Equal("system", mapped[0].Role);
        Assert.IsType<MarkedMessage>(mapped[0]);
        Assert.Equal("t", mapped[0].Thinking);
        Assert.Same(messages[1], mapped[1]);
    }

    private sealed class MarkedMessage : ChatMessage { }

    [Fact]
    public void Qwen35Template_RendersAToolLoopExactlyAsTransformersDoes()
    {
        // Reference: eng/validation/qwen35_template_reference.py (transformers' Jinja environment).
        using Stream stream = typeof(ClientToolLoopCompatibilityTests).Assembly
            .GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Qwen35.pi_tool_loop.transformers.txt")!;
        string expected = new StreamReader(stream).ReadToEnd();
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are pi." },
            new() { Role = "user", Content = "Read notes.txt" },
            new() { Role = "assistant", Content = "", ToolCalls = new() { new ToolCall { Id = "c1", Name = "read", Arguments = new() { ["path"] = "notes.txt" } } } },
            new() { Role = "tool", Content = "hello world", ToolCallId = "c1" },
        };

        string rendered = ChatTemplate.RenderFromGgufTemplate(Qwen35Template(), messages, true, "qwen35", ReadTool);

        // The current turn's call carries the empty reasoning block it was generated after;
        // the template only emits it once it has found the last real user turn.
        Assert.Contains("<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>", rendered, StringComparison.Ordinal);
        Assert.Equal(expected, rendered);
    }

    [Theory]
    [InlineData("{{ not(a and b) }}", "True")]
    [InlineData("{{ not(a) }}", "False")]
    [InlineData("{{ not (a and b) }}", "True")]
    [InlineData("{%- if not(c.startswith('<x>') and c.endswith('</x>')) %}user{%- endif %}", "user")]
    public void Not_BindsToAParenthesizedOperand(string template, string expected)
    {
        var context = new Dictionary<string, object> { ["a"] = true, ["b"] = false, ["c"] = "hello" };
        Assert.Equal(expected, new Jinja2Template(template).Render(context));
    }

    [Fact]
    public void RaiseException_IsReportedAndTheRenderContinues()
    {
        var jinja = new Jinja2Template("a{{ raise_exception('boom') }}b{%- if never %}{{ raise_exception('unseen') }}{%- endif %}");

        string rendered = jinja.Render(new Dictionary<string, object> { ["never"] = false }, out IReadOnlyList<string> raised);

        Assert.Equal("ab", rendered);
        Assert.Equal(new[] { "boom" }, raised);
        Assert.Equal("ab", jinja.Render(new Dictionary<string, object>()));
    }

    // ---- --chat-template-file ---------------------------------------------

    private string WriteTemplate(string name, string text)
    {
        string path = Path.Combine(_root, name);
        File.WriteAllText(path, text);
        return path;
    }

    [Fact]
    public void ChatTemplateFile_LoadsAndRegistersAParsableTemplate()
    {
        string text = "{%- for m in messages %}<{{ m.role }}>{{ m.content }}{%- endfor %}" + Guid.NewGuid().ToString("N");
        ChatTemplateFile file = ChatTemplateFile.Load(WriteTemplate("ok.jinja", "﻿" + text));

        Assert.Equal(text, file.Template);
        Assert.True(Path.IsPathRooted(file.Path));
        Assert.True(ChatTemplate.IsOperatorTemplate(text));
        Assert.True(ChatTemplate.IsOperatorTemplate(new string(text.ToCharArray())));
        Assert.False(ChatTemplate.IsOperatorTemplate(text + " "));
    }

    [Fact]
    public void ChatTemplateFile_RefusesWhatItCannotUse()
    {
        Assert.Contains("is not an existing file",
            Assert.Throws<ArgumentException>(() => ChatTemplateFile.Load(Path.Combine(_root, "missing.jinja"))).Message);
        Assert.Contains("is empty",
            Assert.Throws<ArgumentException>(() => ChatTemplateFile.Load(WriteTemplate("empty.jinja", "  \n"))).Message);
        Assert.Contains("is not a chat template TensorSharp can parse",
            Assert.Throws<ArgumentException>(() => ChatTemplateFile.Load(WriteTemplate("bad.jinja", "{{ messages[0].content }"))).Message);
        Assert.Contains("a chat template is at most",
            Assert.Throws<ArgumentException>(() => ChatTemplateFile.Load(
                WriteTemplate("big.jinja", new string('x', ChatTemplateFile.MaxBytes + 1)))).Message);
        Assert.Contains("requires a path",
            Assert.Throws<ArgumentException>(() => ChatTemplateFile.Load(" ")).Message);
    }

    [Fact]
    public void ChatTemplateFile_TakePath_ReadsBothSpellingsAndKeepsTheRest()
    {
        var remaining = new List<string>();
        string path = ChatTemplateFile.TakePath(
            new[] { "--model", "m.gguf", "--chat-template-file", "a.jinja", "--CHAT-TEMPLATE-FILE=b.jinja", "--chat" }, remaining);

        Assert.Equal("b.jinja", path);
        Assert.Equal(new[] { "--model", "m.gguf", "--chat" }, remaining);
        Assert.Null(ChatTemplateFile.TakePath(new[] { "--model", "m.gguf" }, new List<string>()));
        Assert.Throws<ArgumentException>(() => ChatTemplateFile.TakePath(new[] { "--chat-template-file" }, new List<string>()));
    }

    [Fact]
    public void AnOperatorTemplate_IsRenderedEvenWhereTheFamilyPrefersItsOwnRenderer()
    {
        Assert.NotNull(ChatProtocolRegistry.For("mistral3")!.PreferOwnRenderer);
        string marker = "OPERATOR-" + Guid.NewGuid().ToString("N");
        string template = marker + "{%- for m in messages %}[{{ m.role }}]{{ m.content }}{%- endfor %}";

        string before = ChatTemplate.RenderFromGgufTemplate(template, Conversation("system"), true, "mistral3");
        Assert.DoesNotContain(marker, before, StringComparison.Ordinal);

        ChatTemplate.RegisterOperatorTemplate(template);
        string after = ChatTemplate.RenderFromGgufTemplate(template, Conversation("developer"), true, "mistral3");
        Assert.StartsWith(marker + "[system]You are a coding assistant", after, StringComparison.Ordinal);
    }

    [Fact]
    public void Server_ReadsTheChatTemplateFile_ForTheStartupModel()
    {
        string text = "{%- for m in messages %}{{ m.content }}{%- endfor %}";
        string file = WriteTemplate("server.jinja", text);
        string model = Path.Combine(_root, "hosted.gguf");

        ServerHostingOptions separate = ServerOptionsBuilder.Build(
            new[] { "--model", model, "--chat-template-file", file, "--no-skills" }, _root);
        ServerHostingOptions joined = ServerOptionsBuilder.Build(
            new[] { "--model", model, "--chat-template-file=" + file, "--no-skills" }, _root);
        ServerHostingOptions none = ServerOptionsBuilder.Build(new[] { "--model", model, "--no-skills" }, _root);

        Assert.Equal(text, separate.ChatTemplate!.Template);
        Assert.Equal(text, joined.ChatTemplate!.Template);
        Assert.Null(none.ChatTemplate);
    }

    [Fact]
    public void Server_RefusesAChatTemplateFileItCannotApply()
    {
        string file = WriteTemplate("refused.jinja", "{{ messages[0].content }}");

        Assert.Contains("requires --model", Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.Build(new[] { "--chat-template-file", file }, _root)).Message);
        Assert.Contains("does not apply to --embeddings", Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "e.gguf"), "--embeddings",
                "--embedding-threads", "1", "--chat-template-file", file, "--backend", "cpu" }, _root)).Message);
        Assert.Contains("Did you mean '--chat-template-file'", Assert.Throws<ArgumentException>(
            () => ServerOptionsBuilder.Build(new[] { "--chat-template-fil", file }, _root)).Message);
    }

    [Fact]
    public void Lifecycle_AppliesTheTemplateToTheConfiguredModelOnly()
    {
        string pathA = WriteMinimalGguf("model-a.gguf");
        string pathB = WriteMinimalGguf("model-b.gguf");
        ChatTemplateFile file = ChatTemplateFile.Load(WriteTemplate("life.jinja", "{{ messages[0].content }}"));
        var lifecycle = new ModelLifecycleService(NullLogger.Instance, (path, _, _, _) => new FakeModel(path))
        {
            ChatTemplateOverride = (file, pathA),
        };

        lifecycle.LoadModel(pathA, null, "cpu");
        Assert.Equal(file.Template, lifecycle.Model.Config.ChatTemplate);

        lifecycle.LoadModel(pathB, null, "cpu");
        Assert.Equal(FakeModel.ShippedTemplate, lifecycle.Model.Config.ChatTemplate);

        lifecycle.LoadModel(pathA, null, "cpu");
        Assert.Equal(file.Template, lifecycle.Model.Config.ChatTemplate);
        lifecycle.Unload();
    }

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_root, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);
        bw.Write(0UL);
        bw.Write(0UL);
        bw.Write(new byte[8]);
        return path;
    }

    private sealed class FakeModel : ModelBase
    {
        public const string ShippedTemplate = "the GGUF's own template";

        public FakeModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "qwen35", ChatTemplate = ShippedTemplate };
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();
        protected override void ResetKVCacheCore() { }
    }

    // ---- thinking switches and reasoning levels ---------------------------

    private static JsonElement Body(string json) => JsonDocument.Parse(json).RootElement;

    [Theory]
    [InlineData("{}", false)]
    [InlineData("{\"think\":true}", true)]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":true}}", true)]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":false},\"think\":false}", false)]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":true},\"think\":true}", true)]
    [InlineData("{\"chat_template_kwargs\":null,\"think\":null}", false)]
    public void ThinkingSwitch_ReadsEitherSpelling(string json, bool expected)
    {
        Assert.True(ThinkingSwitchParser.TryParse(Body(json), out bool think, out _, out string error), error);
        Assert.Equal(expected, think);
    }

    [Theory]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":true},\"think\":false}", "disagree")]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":\"yes\"}}", "must be a boolean")]
    [InlineData("{\"think\":\"yes\"}", "must be a boolean")]
    [InlineData("{\"chat_template_kwargs\":[1]}", "must be an object")]
    public void ThinkingSwitch_RefusesWhatItCannotResolve(string json, string reason)
    {
        Assert.False(ThinkingSwitchParser.TryParse(Body(json), out _, out _, out string error));
        Assert.Contains(reason, error, StringComparison.Ordinal);
    }

    [Fact]
    public void ThinkingSwitch_NamesTheKwargsItDoesNotRead()
    {
        Assert.True(ThinkingSwitchParser.TryParse(
            Body("{\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":true,\"x\":1}}"),
            out _, out IReadOnlyList<string> ignored, out _));
        Assert.Equal(new[] { "preserve_thinking", "x" }, ignored);
    }

    [Theory]
    [InlineData("{\"reasoning_effort\":\"minimal\"}", "low")]
    [InlineData("{\"reasoning_effort\":\"none\"}", "low")]
    [InlineData("{\"reasoning_effort\":\"xhigh\"}", "high")]
    [InlineData("{\"reasoning_effort\":\"MEDIUM\"}", "medium")]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":false}}", "low")]
    [InlineData("{}", null)]
    public void ReasoningEffort_ReadsOpenAIsOtherLevelsAsTheNearestOne(string json, string expected)
    {
        Assert.True(ReasoningEffortParser.TryParse(Body(json), out string effort, out string error), error);
        Assert.Equal(expected, effort);
    }

    [Theory]
    [InlineData("\"chat_template_kwargs\":{\"enable_thinking\":true},\"think\":false", 400)]
    [InlineData("\"chat_template_kwargs\":{\"enable_thinking\":1}", 400)]
    [InlineData("\"chat_template_kwargs\":{\"enable_thinking\":true,\"preserve_thinking\":true}", 404)]
    [InlineData("\"reasoning_effort\":\"minimal\"", 404)]
    [InlineData("\"reasoning_effort\":\"max\"", 400)]
    public async Task ChatCompletions_SettlesTheThinkingFieldsBeforeAdmission(string extra, int status)
    {
        // 404 is the hosted-model guard: the request was admitted.
        var context = new DefaultHttpContext();
        string request = "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false," + extra + "}";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() });
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        using (var service = new UnloadedService("qwen35"))
        {
            await new OpenAIChatAdapter(service, options, new UploadStoragePolicy(Path.Combine(_root, "uploads")), registry,
                null, new SessionWorkspaceManager(Path.Combine(_root, "workspaces")), NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        }
        Assert.Equal(status, context.Response.StatusCode);
    }

    private sealed class UnloadedService(string architecture) : ModelService { public override string Architecture => architecture; }

    // ---- the host's tools and a client's tool loop ------------------------

    [Fact]
    public void ClientRunsToolLoop_IsDeclaredToolsOrToolHistory()
    {
        var plain = new List<ChatMessage> { new() { Role = "user", Content = "hi" } };
        var history = new List<ChatMessage>
        {
            new() { Role = "user", Content = "read it" },
            new() { Role = "assistant", ToolCalls = new List<ToolCall> { new() { Name = "read" } } },
            new() { Role = "tool", Content = "contents" },
        };

        Assert.True(SkillRequestPlan.ClientRunsToolLoop(ReadTool, plain));
        Assert.True(SkillRequestPlan.ClientRunsToolLoop(null, history));
        Assert.True(SkillRequestPlan.ClientRunsToolLoop(new List<ToolFunction>(), history));
        Assert.False(SkillRequestPlan.ClientRunsToolLoop(null, plain));
        Assert.False(SkillRequestPlan.ClientRunsToolLoop(new List<ToolFunction>(), plain));
    }

    private (SkillRegistry Registry, ServerHostingOptions Options) SkillsHost(params string[] extraArgs)
    {
        string skills = Path.Combine(_root, "skills");
        string skill = Path.Combine(skills, "acme");
        Directory.CreateDirectory(skill);
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: acme\ndescription: Formats ACME invoice numbers.\n---\n\nThe format is ACME-<region>.\n");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skills } });
        var options = ServerOptionsBuilder.Build(
            new[] { "--model", Path.Combine(_root, "x.gguf"), "--skills-dir", skills }.Concat(extraArgs).ToArray(), _root);
        Assert.True(options.SkillsDiscovery);
        Assert.True(options.MultiAgent.Enabled);
        return (registry, options);
    }

    private static SkillRequestPlan Plan(SkillRegistry registry, ServerHostingOptions options, bool clientLoop,
        IReadOnlyList<string> skills = null, bool? discovery = null, bool? multiAgent = null)
    {
        SkillRequestPlan plan = SkillRequestPlan.Create(registry, skills, discovery, ReadTool, "qwen35", 32768, options,
            out IReadOnlyList<string> unknown, multiAgent: multiAgent, clientRunsToolLoop: clientLoop);
        Assert.Empty(unknown);
        return plan;
    }

    [Fact]
    public void AClientToolLoop_GetsNoneOfTheHostsTools()
    {
        var (registry, options) = SkillsHost();

        Assert.Null(Plan(registry, options, clientLoop: true));
    }

    [Fact]
    public void AClientToolLoop_DoesNotInheritTheOperatorsSkillSelection()
    {
        var (registry, options) = SkillsHost("--skill", "acme");

        Assert.Null(Plan(registry, options, clientLoop: true));
        Assert.Contains(Plan(registry, options, clientLoop: false).Selected, s => s.Id == "acme");
    }

    [Fact]
    public void AClientToolLoop_GetsWhatItAsksFor()
    {
        var (registry, options) = SkillsHost();

        SkillRequestPlan discovered = Plan(registry, options, clientLoop: true, discovery: true);
        Assert.Contains(discovered.Tools, t => SkillTools.IsSkillTool(t.Name));
        Assert.Contains(discovered.Tools, t => t.Name == "read");
        Assert.DoesNotContain(discovered.Tools, t => MultiAgentTools.IsTool(t.Name));

        SkillRequestPlan named = Plan(registry, options, clientLoop: true, skills: new[] { "acme" });
        Assert.Contains(named.Selected, s => s.Id == "acme");
        Assert.Empty(named.Prompt.Catalog);

        SkillRequestPlan delegated = Plan(registry, options, clientLoop: true, multiAgent: true);
        Assert.Contains(delegated.Tools, t => MultiAgentTools.IsTool(t.Name));
        Assert.DoesNotContain(delegated.Tools, t => SkillTools.IsSkillTool(t.Name));
    }

    [Fact]
    public void WithoutAClientToolLoop_TheHostsToolsAreMergedAsBefore()
    {
        var (registry, options) = SkillsHost();

        SkillRequestPlan plan = Plan(registry, options, clientLoop: false);
        Assert.Contains(plan.Tools, t => SkillTools.IsSkillTool(t.Name));
        Assert.Contains(plan.Tools, t => MultiAgentTools.IsTool(t.Name));
        Assert.Contains(plan.Tools, t => t.Name == "read");
    }

    // ---- warm-up log lines -------------------------------------------------

    [Fact]
    public async Task WarmupLogLines_SayTheyAreTheWarmup()
    {
        var logger = new RecordingLogger();
        var telemetry = new InferenceTelemetry(logger);
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hi" } };
        bool insideWarmup = false;

        async IAsyncEnumerable<object> Frames([EnumeratorCancellation] CancellationToken ct = default)
        {
            await Task.Yield();
            insideWarmup = PrefixCacheWarmup.IsRunningHere;
            telemetry.LogChatStarted("qwen35", 1, false, null, history, null);
            telemetry.LogChatFinished(false, 1, 10, 0, 0, 1, 1, 1, "max_tokens", "Hello");
            yield return new { done = true };
        }

        PrefixCacheWarmup.Result result = await PrefixCacheWarmup.RunAsync((_, ct) => Frames(ct), "warm", false, NullLogger.Instance);
        Assert.True(result.Warmed, result.Detail);
        Assert.True(insideWarmup);
        Assert.False(PrefixCacheWarmup.IsRunningHere);

        telemetry.LogChatStarted("qwen35", 8, false, null, history, null);
        telemetry.LogChatFinished(false, 2, 10, 0, 0, 1, 1, 1, "eos", "Hi");

        Assert.StartsWith("chat.start arch=qwen35 origin=prefix-cache-warmup maxTokens=1 ", logger.Messages[0], StringComparison.Ordinal);
        Assert.StartsWith("chat.complete tokens=1 origin=prefix-cache-warmup promptTokens=10 ", logger.Messages[1], StringComparison.Ordinal);
        Assert.StartsWith("chat.start arch=qwen35 maxTokens=8 ", logger.Messages[2], StringComparison.Ordinal);
        Assert.StartsWith("chat.complete tokens=2 promptTokens=10 ", logger.Messages[3], StringComparison.Ordinal);
    }

    private sealed class RecordingLogger : ILogger
    {
        public List<string> Messages { get; } = new();
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception,
            Func<TState, Exception, string> formatter)
        {
            lock (Messages) Messages.Add(formatter(state, exception));
        }
    }
}
