// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// An attachment must change nothing upstream of the message that attached it.
///
/// <para>
/// Its name used to be appended to the shell tool's declaration, and the declarations are
/// inside the prefix every conversation shares. Live on Qwen3.8 27B: the startup warm-up
/// published the 7,219-token shared prompt, a new chat with an image logged "reuse=0" and
/// prefilled 8,345 tokens, and a second image later in that chat re-prefilled everything
/// from inside the tools block. The names now ride on the message
/// (<see cref="ChatHistoryPreparer.AnnotateAttachmentNames"/>), so the declarations and the
/// shared prefix are pinned identical with and without attachments, and an annotated
/// message is pinned to render the same bytes on every later turn.
/// </para>
/// </summary>
public sealed class AttachmentPromptStabilityTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-attach-prefix-" + Guid.NewGuid().ToString("N"));
    private readonly ShellRunner _runner;
    private readonly CodeRunnerAdapter _adapter;

    public AttachmentPromptStabilityTests()
    {
        Directory.CreateDirectory(_root);
        var options = new CodeExecOptions
        {
            Enabled = true,
            Sandbox = SkillSandboxMode.Off,
            ScratchDirectory = _root,
            AllowInstall = true,
        };
        _runner = new ShellRunner(options);
        _adapter = new CodeRunnerAdapter(_runner, options);
    }

    public void Dispose()
    {
        _runner.Dispose();
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    private static readonly CodeInputFile[] Attachments =
    {
        new("IMG_1554.jpeg", "/uploads/a.jpeg"),
        new("image.png", "/uploads/b.png"),
        new("image-2.png", "/uploads/c.png"),
        new("survey results.csv", "/uploads/d.csv"),
    };

    private SkillRequestPlan Plan(bool skills, IReadOnlyList<CodeInputFile> files, SessionWorkspace workspace)
    {
        string skillsDir = Path.Combine(_root, "skills");
        string skill = Path.Combine(skillsDir, "documents");
        Directory.CreateDirectory(skill);
        File.WriteAllText(Path.Combine(skill, "SKILL.md"),
            "---\nname: documents\ndescription: Make PDF and Word documents from files.\n---\n\nUse the scripts.\n");
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { skillsDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(skills
            ? new[] { "--model", "x.gguf", "--skills-dir", skillsDir }
            : new[] { "--model", "x.gguf", "--no-skills" }, _root);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, null, discovery: skills, clientTools: null,
            architecture: "qwen35", contextTokens: 32_768, options,
            out IReadOnlyList<string> unknown,
            codeRunner: _adapter, codeInputFiles: files, workspace: workspace);
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        return plan;
    }

    private static string Json(SkillRequestPlan plan) => JsonSerializer.Serialize(plan.Tools);

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void ToolDeclarations_AreIdenticalWithAndWithoutAttachments(bool skills, bool persistent)
    {
        SessionWorkspace workspace = persistent
            ? new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("chat")
            : null;

        string without = Json(Plan(skills, null, workspace));
        string with = Json(Plan(skills, Attachments, workspace));

        Assert.Equal(without, with);
        foreach (CodeInputFile file in Attachments)
            Assert.DoesNotContain(file.Name, with, StringComparison.Ordinal);

        // One static sentence on the shell and on the argument it is written in keeps the
        // anti-retyping cue (gemma-4-E4B re-typed a CSV when only the tail said so).
        ToolFunction shell = Assert.Single(Plan(skills, Attachments, workspace).Tools,
            tool => tool.Name == SkillToolNames.Shell);
        Assert.Contains("names given in the conversation", shell.Description, StringComparison.Ordinal);
        Assert.Contains("names given in the conversation", shell.Parameters["command"].Description,
            StringComparison.Ordinal);
    }

    [Fact]
    public void SharedPrefix_IsIdenticalWithAndWithoutAnImageAttachment()
    {
        ModelBase model = Model();
        var renderer = new KVCachePromptRenderer(new ToolRenderer());
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance);
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance);
        using var pipeline = new ChatGenerationPipeline(lifecycle, host, renderer,
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);

        (int Length, string Hash) Shared(IReadOnlyList<CodeInputFile> files, ChatMessage user)
        {
            List<ToolFunction> tools = Plan(skills: true, files, workspace: null).Tools;
            var history = new List<ChatMessage>
            {
                new() { Role = "system", Content = new string('s', 200) },
                user,
            };
            List<int> prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history,
                "qwen35", addGenerationPrompt: true, tools: tools);
            int length = pipeline.ComputeSharedPrefixTokens(model, history, prompt, "qwen35", tools, false);
            return (length, ChatGenerationPipeline.SharedPrefixHash(prompt, length));
        }

        var warmup = Shared(null, new ChatMessage { Role = "user", Content = "hi" });
        var image = Shared(new[] { new CodeInputFile("IMG_1554.jpeg", "/uploads/a.jpeg") },
            new ChatMessage
            {
                Role = "user",
                Content = "What is in this photo?\n\n(Attached as file 'IMG_1554.jpeg' in the working directory.)",
                ImagePaths = new List<string> { "/uploads/a.jpeg" },
            });

        Assert.True(warmup.Length >= ChatGenerationPipeline.MinSharedPrefixTokens);
        Assert.Equal(warmup, image);
    }

    private static readonly (string Path, string Name)[] TurnOneFiles = { ("/uploads/g1.png", "image.png") };

    private static ChatMessage Attached(string content, params (string Path, string Name)[] files) => new()
    {
        Role = "user",
        Content = content,
        AttachmentPaths = files.Select(f => f.Path).ToList(),
        AttachmentNames = files.Select(f => f.Name).ToList(),
    };

    private static List<ChatMessage> Annotate(List<ChatMessage> messages,
        IReadOnlyDictionary<string, string> stagedCsv = null)
    {
        IReadOnlyList<CodeInputFile> collected = WebUiChatService.CollectCodeInputFiles(messages);
        if (stagedCsv != null)
            messages = ChatHistoryPreparer.UseFileBackedCsvAttachments(messages, stagedCsv);
        return ChatHistoryPreparer.AnnotateAttachmentNames(
            messages, WebUiChatService.StagedNamesBySource(collected));
    }

    // Turn 1 is annotated from turn 1's history; turn 2 sends turn 1 again with more after
    // it. The names of a message depend on it and the messages before it only, so turn 1's
    // message renders the same bytes both times -- including when turn 2 attaches a second
    // file the user also calls "image.png", and a CSV the file-backed reference names.
    [Fact]
    public void AnAnnotatedMessage_RendersTheSameOnEveryLaterTurn()
    {
        List<ChatMessage> TurnOne() => new() { Attached("Turn this into a PDF.", TurnOneFiles) };
        List<ChatMessage> TurnTwo()
        {
            List<ChatMessage> messages = TurnOne();
            messages.Add(new ChatMessage { Role = "assistant", Content = "Done: photo.pdf" });
            ChatMessage second = Attached("[File: results.csv]\nid,value\n1,2\n[End of file]\n\nAnd this one with the table.",
                ("/uploads/g2.png", "image.png"), ("/uploads/t.csv", "results.csv"));
            second.TextFilePaths = new List<string> { "/uploads/t.csv" };
            second.TextFileNames = new List<string> { "results.csv" };
            second.HasFileBackedTextAttachments = true;
            messages.Add(second);
            return messages;
        }

        List<ChatMessage> first = Annotate(TurnOne());
        List<ChatMessage> secondTurn = Annotate(TurnTwo(),
            new Dictionary<string, string>(StringComparer.Ordinal) { ["/uploads/t.csv"] = "results.csv" });

        Assert.Equal("Turn this into a PDF.\n\n(Attached as file 'image.png' in the working directory.)",
            first[0].Content);
        Assert.Equal(first[0].Content, secondTurn[0].Content);

        // The second "image.png" is a different upload: its own name, no spaces, and the
        // CSV the file-backed reference already names is not named twice.
        string latest = secondTurn[2].Content;
        Assert.StartsWith("[Attached CSV available to tools: 'results.csv']", latest, StringComparison.Ordinal);
        Assert.EndsWith("And this one with the table.\n\n(Attached as file 'image-2.png' in the working directory.)",
            latest, StringComparison.Ordinal);
        Assert.Equal(1, latest.Split("'results.csv'").Length - 1);

        // Rendered, turn two's prompt starts with turn one's up to the end of that message.
        var renderer = new KVCachePromptRenderer(new ToolRenderer());
        ITokenizer tokenizer = new CharTokenizer();
        List<int> one = renderer.RenderToTokens(tokenizer, null, first, "qwen35", addGenerationPrompt: false);
        List<int> two = renderer.RenderToTokens(tokenizer, null, secondTurn, "qwen35", addGenerationPrompt: false);
        Assert.Equal(one, two.Take(one.Count));

        // The persisted conversation is the request, never this copy.
        Assert.Equal("Turn this into a PDF.", TurnOne()[0].Content);
    }

    private static ChatMessage Photo(string content, string path, string name)
    {
        ChatMessage message = Attached(content, (path, name));
        message.ImagePaths = new List<string> { path };
        return message;
    }

    /// <summary>Every image the window has no room for is elided: each costs more than the
    /// window leaves once the instructions are in.</summary>
    private static ChatGenerationPipeline.MediaHistoryWindow ElideAll(
        List<ChatMessage> history, IReadOnlyDictionary<string, string> stagedNames)
    {
        static int Unexpanded(List<ChatMessage> messages) => messages.Sum(message =>
            (message.Content?.Length ?? 0) + 12 + (message.ImagePaths?.Count ?? 0));
        static int Expanded(List<ChatMessage> messages) => Unexpanded(messages)
            + messages.Sum(message => (message.ImagePaths?.Count ?? 0) * 3_000);
        return ChatGenerationPipeline.CompactMediaHistoryForContext(
            history, Unexpanded(history), Expanded(history), contextLimit: 8_192,
            requestedGenerationTokens: 8_192, Unexpanded, Expanded, stagedNames);
    }

    // Two pasted photos are both "image.png"; the code tools stage the second as
    // "image-2.png", and "image.png" in the workspace is the FIRST. A HEIC photo is staged
    // as its PNG. When an earlier picture no longer fits, its note must name the file the
    // way the message's own attachment note does -- the display name would send a model
    // asked about "the second photo" to the wrong file.
    [Fact]
    public void AnElidedPicture_IsNamedAsTheCodeToolsStageIt()
    {
        List<ChatMessage> Conversation() => new()
        {
            new() { Role = "system", Content = new string('s', 3_000) },
            Photo("first photo", "/uploads/g1.png", "image.png"),
            new() { Role = "assistant", Content = "A red sign." },
            Photo("second photo", "/uploads/g2.png", "image.png"),
            new() { Role = "assistant", Content = "A blue door." },
            Photo("an iPhone photo", "/uploads/h1.heic", "IMG_1.heic"),
            new() { Role = "assistant", Content = "A cat." },
            new() { Role = "user", Content = "Which one is brightest?" },
        };

        List<ChatMessage> messages = Conversation();
        IReadOnlyDictionary<string, string> staged =
            WebUiChatService.StagedNamesBySource(WebUiChatService.CollectCodeInputFiles(messages));
        messages = ChatHistoryPreparer.AnnotateAttachmentNames(messages, staged);

        ChatGenerationPipeline.MediaHistoryWindow window = ElideAll(messages, staged);
        Assert.Equal(3, window.ElidedMedia);
        Assert.Equal(0, window.RemovedMessages);
        foreach ((int index, string name) in new[] { (1, "image.png"), (3, "image-2.png"), (5, "IMG_1.png") })
        {
            ChatMessage message = window.History[index];
            Assert.Null(message.ImagePaths);
            Assert.StartsWith($"[earlier image '{name}' is no longer shown]\n\n", message.Content, StringComparison.Ordinal);
            Assert.EndsWith($"(Attached as file '{name}' in the working directory.)", message.Content, StringComparison.Ordinal);
        }

        // Without the code tools nothing is staged or annotated: the user's names are all
        // the conversation has.
        ChatGenerationPipeline.MediaHistoryWindow plain = ElideAll(Conversation(), stagedNames: null);
        Assert.Equal("[earlier image 'image.png' is no longer shown]\n\nsecond photo", plain.History[3].Content);
        Assert.Equal("[earlier image 'IMG_1.heic' is no longer shown]\n\nan iPhone photo", plain.History[5].Content);
    }

    // A frame the server extracted (a PDF page) was never named to anyone; its internal
    // upload name tells the model nothing, so the note says what went without a name.
    [Fact]
    public void AnElidedPictureNobodyNamed_IsNotedWithoutAName()
    {
        ChatMessage message = Photo("the report", "/uploads/r.pdf", "report.pdf");
        message.ImagePaths = new List<string> { "/uploads/3f2c9a-page-1.png" };
        var staged = new Dictionary<string, string>(StringComparer.Ordinal) { ["/uploads/r.pdf"] = "report.pdf" };

        Assert.Equal("[an earlier image is no longer shown]\n\nthe report",
            ChatHistoryPreparer.WithMediaReplacedByNote(message, images: 1, audio: 0, staged).Content);
        Assert.Equal("[an earlier image is no longer shown]\n\nthe report",
            ChatHistoryPreparer.WithMediaReplacedByNote(message, images: 1, audio: 0, stagedNames: null).Content);
    }

    // A HEIC photo is named by its PNG from the file list alone: the name cannot follow
    // whether this turn's conversion worked, or an earlier message's note would change.
    [Fact]
    public void StagedNames_AreDerivedFromTheCollectedFilesAlone()
    {
        var collected = new[]
        {
            new CodeInputFile("IMG_1.heic", "/uploads/h1.heic"),
            new CodeInputFile("IMG_1-2.png", "/uploads/p1.png"),
            new CodeInputFile("scan.HEIF", "/uploads/h2.HEIF"),
        };

        IReadOnlyDictionary<string, string> names = WebUiChatService.StagedNamesBySource(collected);

        Assert.Equal("IMG_1.png", names["/uploads/h1.heic"]);
        Assert.Equal("IMG_1-2.png", names["/uploads/p1.png"]);
        Assert.Equal("scan.png", names["/uploads/h2.HEIF"]);
    }

    // The note quotes a name with its apostrophe as ’, so the name cannot end the quotes
    // around it, and the file is staged under that same name. Staged as "Bob's report.pdf"
    // and announced as 'Bob’s report.pdf', it sent the model to a file that does not exist.
    [Fact]
    public void ANameWithAnApostrophe_IsStagedUnderTheNameItsNoteGives()
    {
        string source = Path.Combine(_root, "r1.pdf");
        File.WriteAllText(source, "report");
        var messages = new List<ChatMessage> { Attached("Summarize it.", (source, "Bob's report.pdf")) };

        IReadOnlyList<CodeInputFile> collected = WebUiChatService.CollectCodeInputFiles(messages);
        List<ChatMessage> annotated = ChatHistoryPreparer.AnnotateAttachmentNames(
            messages, WebUiChatService.StagedNamesBySource(collected));
        SessionWorkspace workspace = new SessionWorkspaceManager(Path.Combine(_root, "workspaces")).GetOrCreate("chat");
        IReadOnlySet<string> staged = CodeInputFileStager.Stage(collected, workspace);

        Assert.Equal("Summarize it.\n\n(Attached as file 'Bob’s report.pdf' in the working directory.)",
            annotated[0].Content);
        Assert.Equal(new[] { "Bob’s report.pdf" }, staged);
        Assert.Equal("report", File.ReadAllText(Path.Combine(workspace.WorkDirectory, "Bob’s report.pdf")));
        Assert.Equal("Bob's report.pdf", messages[0].AttachmentNames[0]);
    }

    [Fact]
    public void AnnotationIsLeftOutOfMessagesWithoutAttachmentsAndOfAssistantTurns()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "hello" },
            new()
            {
                Role = "assistant", Content = "made it",
                AttachmentPaths = new List<string> { "/uploads/x.png" },
                AttachmentNames = new List<string> { "x.png" },
            },
        };
        var names = new Dictionary<string, string>(StringComparer.Ordinal) { ["/uploads/x.png"] = "x.png" };

        Assert.Same(messages, ChatHistoryPreparer.AnnotateAttachmentNames(messages, names));
    }

    private static ModelBase Model()
    {
        var model = (Qwen35Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35Model));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(model, new CharTokenizer());
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { Architecture = "qwen35", ChatTemplate = "template" });
        return model;
    }

    /// <summary>Declarations first, as Qwen's template renders them, then the messages.</summary>
    private sealed class ToolRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
        {
            var text = new StringBuilder(template).Append('|');
            if (tools != null)
                foreach (ToolFunction tool in tools)
                    text.Append(JsonSerializer.Serialize(tool)).Append(tool.ParametersSchemaJson);
            foreach (ChatMessage message in messages)
                text.Append('<').Append(message.Role).Append('>').Append(message.Content).Append("</>");
            if (addGenerationPrompt) text.Append("<assistant>");
            return text.ToString();
        }
    }

    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(new[] { (char)tokenId }));
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }
}
