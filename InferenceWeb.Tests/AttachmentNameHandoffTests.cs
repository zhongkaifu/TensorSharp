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
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// The name an attachment is staged under reaches the model twice: in the note on the
/// message that attached it, and -- once that message's picture no longer fits the window
/// -- in the note that replaces the picture. Both are decided in <see cref="WebUiChatService"/>,
/// which stages the files and names them on their messages; the second is written by
/// <see cref="ChatGenerationPipeline"/> several hops later, from the map the request's
/// skill plan carries through <see cref="ModelService"/> to the turn
/// (<see cref="ChatTurnContext.StagedAttachmentNames"/>). Each hop is one assignment, and
/// without one of them the picture's note falls back to the name the user attached it
/// under: two pasted photos are both "image.png", the second is staged as "image-2.png",
/// and "image.png" in the workspace is the FIRST photo.
/// </summary>
/// <remarks>
/// Driven through the real request path -- the Web UI service, the model service, the
/// skill loop and the generation pipeline -- with a model whose "encoder" expands every
/// picture by a fixed number of tokens and whose answer is "OK". Nothing here needs
/// weights: what is checked is the prompt the pipeline prepared for the engine, and the
/// files in the session's workspace.
/// </remarks>
public sealed class AttachmentNameHandoffTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-name-handoff-" + Guid.NewGuid().ToString("N"));

    public AttachmentNameHandoffTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    // Turn one attaches an iPhone photo, whose bytes no decoder reads, and a photo pasted as
    // "image.png"; turn two another "image.png"; the question attaches a third picture. Each
    // costs more of the 32k window than it has, so every earlier picture becomes a note.
    [Fact]
    public async Task AnEarlierPictureThatNoLongerFits_IsNamedAsItsMessageNamesIt_ThroughTheRequestPath()
    {
        byte[] heic = { 0x00, 0x00, 0x00, 0x18, 0x66, 0x74, 0x79, 0x70, 0x68, 0x65, 0x69, 0x63, 0xde, 0xad };
        File.WriteAllBytes(Path.Combine(_root, "u-heic.heic"), heic);
        File.WriteAllText(Path.Combine(_root, "u-a.png"), "first image.png");
        File.WriteAllText(Path.Combine(_root, "u-b.png"), "second image.png");
        File.WriteAllText(Path.Combine(_root, "u-c.png"), "photo.png");
        string modelPath = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(modelPath);

        ExpandingModel model = null;
        using var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => model = new ExpandingModel(path));
        service.LoadModel(modelPath, null, "cpu");
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 4_096, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 4_096, MaxPrefillChunkSize = 4_096, SoloPrefillChunkSize = 4_096,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        var sessions = new SessionManager();
        ChatSession session = sessions.CreateSession();
        var workspaces = new SessionWorkspaceManager(Path.Combine(_root, "workspaces"));
        var chat = new WebUiChatService(service, sessions, Options(), new UploadStoragePolicy(_root),
            new SkillRegistry(new SkillRegistryOptions { Roots = [] }), new FileToolsRunner(), workspaces,
            codeArtifacts: null, NullLoggerFactory.Instance);

        static object Photos(string content, params (string File, string Name)[] files) => new
        {
            role = "user",
            content,
            imagePaths = files.Select(f => f.File).ToArray(),
            attachments = files.Select(f => new { file = f.File, fileName = f.Name, mediaType = "image" }).ToArray(),
        };
        JsonElement body = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(new
        {
            sessionId = session.Id,
            messages = new object[]
            {
                Photos("What are these?", ("u-heic.heic", "IMG_1.heic"), ("u-a.png", "image.png")),
                new { role = "assistant", content = "A cat, and a red sign." },
                Photos("And this one?", ("u-b.png", "image.png")),
                new { role = "assistant", content = "A blue door." },
                Photos("Which is brightest?", ("u-c.png", "photo.png")),
            },
        }));

        await foreach (object _ in chat.ChatStreamAsync(body, CancellationToken.None)) { }

        List<ChatMessage> prompt = model.PreparedBeforeFirstForward;
        Assert.NotNull(prompt);
        ChatMessage first = Assert.Single(prompt, m => m.Content.Contains("What are these?", StringComparison.Ordinal));
        ChatMessage second = Assert.Single(prompt, m => m.Content.Contains("And this one?", StringComparison.Ordinal));
        ChatMessage latest = Assert.Single(prompt, m => m.Content.Contains("Which is brightest?", StringComparison.Ordinal));

        // The HEIC photo is named by the PNG it is staged as even though nothing could
        // decode it; the second "image.png" by its own name, in both notes.
        Assert.Null(first.ImagePaths);
        Assert.Equal(
            "[earlier image 'IMG_1.png' is no longer shown]\n[earlier image 'image.png' is no longer shown]\n\n"
            + "What are these?\n\n(Attached as files 'IMG_1.png', 'image.png' in the working directory.)",
            first.Content);
        Assert.Null(second.ImagePaths);
        Assert.Equal(
            "[earlier image 'image-2.png' is no longer shown]\n\n"
            + "And this one?\n\n(Attached as file 'image-2.png' in the working directory.)",
            second.Content);
        Assert.Equal(new[] { Path.Combine(_root, "u-c.png") }, latest.ImagePaths);

        // And the names are what the workspace has: the undecodable photo under its PNG
        // name, its own bytes (the interpreter then reports Pillow's error, not a file that
        // is not there), and each "image.png" under the name its note gives.
        string work = workspaces.GetOrCreate(session.Id).WorkDirectory;
        Assert.Equal(heic, File.ReadAllBytes(Path.Combine(work, "IMG_1.png")));
        Assert.False(File.Exists(Path.Combine(work, "IMG_1.heic")));
        Assert.Equal("first image.png", File.ReadAllText(Path.Combine(work, "image.png")));
        Assert.Equal("second image.png", File.ReadAllText(Path.Combine(work, "image-2.png")));
        Assert.Equal("photo.png", File.ReadAllText(Path.Combine(work, "photo.png")));
        workspaces.Release(session.Id);
    }

    private ServerHostingOptions Options() => new(
        startupModelPath: Path.Combine(_root, "probe.gguf"),
        startupMmProjPath: null,
        defaultBackend: "ggml_cpu",
        supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
        defaultMaxTokens: 100,
        maxTokensPinned: false,
        defaultVideoFrames: 0,
        defaultVideoFps: 0,
        defaultVideoWidth: 0,
        defaultVideoHeight: 0,
        defaultVideoSteps: 0,
        defaultVideoMode: null,
        uploadDirectory: _root,
        logDirectory: Path.Combine(_root, "logs"),
        fileLoggingEnabled: false,
        samplingDefaults: null,
        skillsEnabled: false,
        skillsAllowScripts: false,
        skillsSandbox: SkillSandboxMode.Required,
        skillsAllowNetwork: false);

    /// <summary>The file tools a persistent workspace gets; nothing is ever run.</summary>
    private sealed class FileToolsRunner : ICodeRunner
    {
        public bool CanRun => true;
        public string UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = SkillToolNames.Shell };
        public IReadOnlyList<ToolFunction> DeclareTools(bool persists) =>
            (persists
                ? new[] { SkillToolNames.ReadFile, SkillToolNames.ApplyPatch, SkillToolNames.WriteFile, SkillToolNames.Shell }
                : new[] { SkillToolNames.Shell })
            .Select(name => new ToolFunction { Name = name }).ToArray();
        public SkillToolResult Execute(ToolCall call, IReadOnlyList<CodeInputFile> inputFiles = null,
            Action<string> onOutput = null, SessionWorkspace workspace = null, IReadOnlyList<string> skillDirectories = null) =>
            throw new InvalidOperationException("The model answers without calling a tool.");
    }

    /// <summary>
    /// A Qwen-family model with a vision tower that expands every picture of a prepared
    /// prompt by <see cref="TokensPerImage"/> tokens, and answers "OK". It keeps the
    /// conversation of each preparation, and the last one before the engine first ran it.
    /// </summary>
    private sealed class ExpandingModel : ModelBase, IMultimodalPromptExpander, IVisionCapableModel
    {
        private const int TokensPerImage = 15_000;
        private readonly object _gate = new();
        private List<ChatMessage> _lastPrepared;

        public ExpandingModel(string path) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "qwen35", VocabSize = 128, NumLayers = 1 };
            Tokenizer = new AsciiTokenizer();
            _maxContextLength = 32_768;
        }

        public List<ChatMessage> PreparedBeforeFirstForward { get; private set; }

        public bool IsVisionEncoderLoaded => true;
        public void LoadVisionEncoder(string mmProjPath) => throw new NotSupportedException();
        public void SetVisionEmbeddings(Tensor embeddings, int position) => embeddings?.Dispose();

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> tokens)
        {
            lock (_gate)
                _lastPrepared = new List<ChatMessage>(history);
            int images = history.Sum(m => m.ImagePaths?.Count ?? 0);
            // Ahead of the text, so the prompt still ends where the answer begins.
            var expanded = new List<int>(tokens.Count + images * TokensPerImage);
            expanded.AddRange(Enumerable.Repeat((int)'.', images * TokensPerImage));
            expanded.AddRange(tokens);
            return expanded;
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "name-handoff";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_gate)
                PreparedBeforeFirstForward ??= _lastPrepared;
            int last = tokens[^1];
            var logits = new float[128];
            logits[last == 'O' ? 'K' : last == 'K' ? 1 : 'O'] = 10;
            return logits;
        }

        protected override void ResetKVCacheCore() { }
    }

    /// <summary>One token per character; anything outside ASCII is '?'.</summary>
    private sealed class AsciiTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < 128 ? (int)c : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
