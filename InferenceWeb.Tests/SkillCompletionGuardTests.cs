// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.IO.Compression;
using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the narrow host completion contract used by TensorAgent's routed
/// research-to-PowerPoint request. Ordinary skill turns remain streaming and unchanged;
/// only an explicitly guarded plan may suppress an unverified final answer and spend one
/// extra generation repairing it.
/// </summary>
public sealed class SkillCompletionGuardTests : IDisposable
{
    private const string Architecture = "nemotron_h_moe";
    private const string ArtifactPrefix = "/artifacts";
    private readonly string _baseDir;
    private readonly SessionWorkspaceManager _workspaces;
    private readonly CodeArtifactStore _artifacts;

    public SkillCompletionGuardTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-completion-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
        _workspaces = new SessionWorkspaceManager(Path.Combine(_baseDir, "sessions"));
        _artifacts = new CodeArtifactStore(Path.Combine(_baseDir, "artifacts"));
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public async Task FalseSuccess_IsSuppressed_OneCorrectionCreatesAndLinksAValidDeck()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("corrected");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.Valid);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        var replay = new ReplayGeneration(
            "<think>done</think>I made it: [report](fake.pptx).",
            "<think>repairing the existing spec</think><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make deck\"}}\n"
                + "</tool_call>");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        Assert.Equal(1, runner.Executions);
        Assert.DoesNotContain("fake.pptx", Content(updates), StringComparison.Ordinal);
        Assert.DoesNotContain("done", Thinking(updates), StringComparison.Ordinal);
        Assert.DoesNotContain("repairing", Thinking(updates), StringComparison.Ordinal);
        Assert.Contains("[Download the .pptx report](/artifacts/", Content(updates), StringComparison.Ordinal);
        Assert.Contains("one corrective continuation", replay.Histories[1], StringComparison.Ordinal);
        Assert.Contains("smallest necessary edit", replay.Histories[1], StringComparison.Ordinal);
        Assert.IsType<SkillChatLoop.HostCompletionCorrectionMessage>(
            replay.MessageHistories[1][^1]);

        SkillToolInvocation invocation = Assert.Single(plan.Invocations);
        Assert.Equal(2, invocation.Round);
        Assert.Equal("report.pptx", Assert.Single(invocation.Files).Name);

        ChatStreamUpdate terminal = Assert.Single(updates, update => update.Done);
        Assert.Equal(20, terminal.PromptTokens);
        Assert.Equal(40, terminal.EvalTokens);
        Assert.Equal(10, terminal.KvCacheReusedTokens);
    }

    [Fact]
    public async Task InvalidCorrection_StopsAfterExactlyTwoGenerations_AndCannotLeakClaims()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("invalid-correction");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.TruncatedZip);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        var replay = new ReplayGeneration(
            "<think>done</think>Initial false success.",
            "<think>retry</think>Second false success.<tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make bad deck\"}}\n"
                + "</tool_call>");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);
        string content = Content(updates);

        Assert.Equal(2, replay.Calls);
        Assert.Equal(1, runner.Executions);
        Assert.DoesNotContain("Initial false success", content, StringComparison.Ordinal);
        Assert.DoesNotContain("Second false success", content, StringComparison.Ordinal);
        Assert.Contains("no valid downloadable .pptx", content, StringComparison.Ordinal);
        Assert.Contains("one corrective attempt", content, StringComparison.Ordinal);
        Assert.Single(updates, update => update.Done);
    }

    /// <summary>
    /// A guarded round and its correction hold their answer and reasoning until the artifact
    /// check says what they are worth, and the client sees nothing of them meanwhile. For
    /// every piece held the loop hands the client's stream loop an update with nothing in
    /// it: an append-only adapter's keep-alive runs on those (StreamKeepAlive), or a long
    /// held correction left the stream silent past a proxy's idle timeout.
    /// </summary>
    [Fact]
    public async Task HeldRoundsAndTheCorrection_HandOverAnUpdateForEveryPieceTheyHold()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("held-correction");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.TruncatedZip);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        const string first = "<think>done</think>Initial false success.";
        const string correction = "<think>retry</think>Second false success.";
        var replay = new ReplayGeneration(first, correction + "<tool_call>\n"
            + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make bad deck\"}}\n"
            + "</tool_call>");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        int empty = updates.Count(u => u.IsParsed && !u.Done && string.IsNullOrEmpty(u.Piece)
            && string.IsNullOrEmpty(u.ThinkingPiece) && u.ParsedToolCalls == null && u.ToolProgressPhase == null);
        Assert.True(empty >= first.Length + correction.Length, $"{empty} empty updates");
        Assert.Equal(2, replay.Calls);
    }

    [Fact]
    public async Task ValidArtifactBeforeFinal_ReleasesTheModelsAnswerInMultiplePieces()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("already-valid");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.Valid);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        var replay = new ReplayGeneration(
            "<think>write it</think><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make deck\"}}\n"
                + "</tool_call>",
            "<think>verified</think>The researched deck is ready.");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        Assert.StartsWith("The researched deck is ready.", Content(updates), StringComparison.Ordinal);
        Assert.Contains("verified", Thinking(updates), StringComparison.Ordinal);
        Assert.Contains("Download the .pptx report", Content(updates), StringComparison.Ordinal);
        Assert.True(updates.Count(update => !update.Done && !string.IsNullOrEmpty(update.Piece)) > 1,
            "a verified answer should retain its parsed pieces instead of collapsing to one buffered frame");
        Assert.DoesNotContain(replay.Histories, history =>
            history.Contains("Host completion check", StringComparison.Ordinal));
    }

    [Fact]
    public async Task CompletedRoutedWriter_StopsBeforeAnotherGenerationOrTrailingToolCall()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-early-stop");
        const string source = "https://research.example.test/apple-m6";
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "notes.md"), source);
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "pptx_spec.json"), "{}");
        var backend = new FakeShellBackend
        {
            Answer = launch =>
            {
                WriteValidPptx(
                    Path.Combine(launch.WorkingDirectory!, "report.pptx"),
                    slideCount: 4,
                    visibleText: "Apple M5 versus M6 — " + source);
                return FakeShellBackend.Ok("writer finished");
            },
        };
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None) { Backend = backend };
        WebUiArtifactRequirement policy = RoutedPolicy() with
        {
            CitationEvidencePath = "notes.md",
        };
        SkillRequestPlan plan = ScriptPlan(runner, workspace, policy);
        plan.Invocations.Add(RunInvocation("research", "scripts/research.py", ok: true));
        var replay = new ReplayGeneration(
            "<think>I may keep tinkering</think><tool_call>\n"
                + "{\"name\": \"skills_run\", \"arguments\": {\"skill\": \"documents\", "
                + "\"path\": \"scripts/make_pptx.py\"}}\n"
                + "</tool_call><tool_call>\n"
                + "{\"name\": \"skills_run\", \"arguments\": {\"skill\": \"documents\", "
                + "\"path\": \"scripts/validate_pptx.py\"}}\n"
                + "</tool_call><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"rewrite it again\"}}\n"
                + "</tool_call>");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(1, replay.Calls);
        Assert.Single(backend.Launches);
        Assert.Equal(0, runner.Executions);
        Assert.Equal(3, updates.Count(update =>
            string.Equals(update.ToolProgressPhase, "finished", StringComparison.Ordinal)));
        Assert.Equal(2, plan.Invocations.Count);
        Assert.DoesNotContain(plan.Invocations, invocation =>
            string.Equals(invocation.Tool, "shell", StringComparison.Ordinal));
        Assert.DoesNotContain(plan.Invocations, invocation =>
            string.Equals(invocation.ResourcePath, "scripts/validate_pptx.py", StringComparison.Ordinal));
        Assert.DoesNotContain("tinkering", Thinking(updates), StringComparison.Ordinal);
        Assert.Equal(
            "The requested PowerPoint report is ready: "
                + $"[Download the .pptx report]({plan.VerifiedArtifact!.Value.Url}).",
            Content(updates));
        Assert.Equal("stop", Assert.Single(updates, update => update.Done).FinishReason);
    }

    [Fact]
    public async Task InvalidRoutedArtifact_DoesNotEarlyStopToolsOrModelRounds()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-no-early-stop");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(
            runner, workspace, guarded: true, policy: RoutedPolicy());
        runner.OnExecute = (execution, currentWorkspace) =>
        {
            if (execution != 1)
                return;

            string path = Path.Combine(currentWorkspace.WorkDirectory, "report.pptx");
            File.WriteAllBytes(path, new byte[] { 0x50, 0x4b, 0x03, 0x04, 1, 2, 3, 4 });
            AddRoutedRuns(plan, Capture(currentWorkspace, "report.pptx"));
        };
        var replay = new ReplayGeneration(
            "<tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make invalid deck\"}}\n"
                + "</tool_call><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"inspect or repair\"}}\n"
                + "</tool_call>",
            "<think>false confidence</think>The deck is complete.",
            "<think>still false</think>Nothing else to do.");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(3, replay.Calls);
        Assert.Equal(2, runner.Executions);
        Assert.Null(plan.VerifiedArtifact);
        Assert.DoesNotContain("deck is complete", Content(updates), StringComparison.OrdinalIgnoreCase);
        Assert.Contains("no valid downloadable .pptx", Content(updates), StringComparison.Ordinal);
    }

    [Fact]
    public async Task PlainOrBacktickedUrl_DoesNotSuppressTheCanonicalDownloadLink()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("plain-url");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.Valid);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        int generation = 0;

        IAsyncEnumerable<ChatStreamUpdate> Generate(
            List<ChatMessage> _,
            List<ToolFunction> __,
            CancellationToken cancellationToken)
        {
            if (generation++ == 0)
            {
                return ReplayGeneration.Emit(
                    "<tool_call>\n{\"name\": \"shell\", \"arguments\": {\"command\": \"make deck\"}}\n</tool_call>",
                    "stop",
                    cancellationToken);
            }

            Assert.True(runner.LastArtifact.HasValue);
            return ReplayGeneration.Emit(
                $"<think>verified</think>Deck URL: `{runner.LastArtifact.Value.Url}`.",
                "stop",
                cancellationToken);
        }

        List<ChatStreamUpdate> updates = await Run(plan, Generate);
        string content = Content(updates);

        Assert.Contains("Deck URL: `/artifacts/", content, StringComparison.Ordinal);
        Assert.Contains("[Download the .pptx report](/artifacts/", content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task GuardStillGetsOneCorrectionAfterConfiguredRoundCap_AndNoMore()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("capped");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None, ArtifactMode.Valid);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, maxRounds: 1);
        var replay = new ReplayGeneration(
            "<think>first attempt</think><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"fail first\"}}\n"
                + "</tool_call>",
            "<think>at the cap and falsely confident</think>I can fix it now.<tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"dropped capped call\"}}\n"
                + "</tool_call>",
            "<think>repair</think><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make deck\"}}\n"
                + "</tool_call>")
        {
            FinishReason = "max_tokens",
        };

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(3, replay.Calls); // ordinary round + capped answer + exactly one correction
        Assert.Equal(2, runner.Executions);
        Assert.DoesNotContain("I can fix it now", Content(updates), StringComparison.Ordinal);
        Assert.DoesNotContain("falsely confident", Thinking(updates), StringComparison.Ordinal);
        Assert.Contains("Download the .pptx report", Content(updates), StringComparison.Ordinal);
        Assert.Equal(new[] { 1, 3 }, plan.Invocations.Select(invocation => invocation.Round).ToArray());
        Assert.Equal("stop", Assert.Single(updates, update => update.Done).FinishReason);
    }

    [Fact]
    public async Task PlanWithoutRequirement_RetainsOrdinaryOneRoundStreaming()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("ordinary");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: false);
        var replay = new ReplayGeneration("<think>plain</think>Ordinary answer.");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(1, replay.Calls);
        Assert.Equal("Ordinary answer.", Content(updates));
        Assert.True(updates.Count(update => !update.Done && !string.IsNullOrEmpty(update.Piece)) > 1);
    }

    [Fact]
    public async Task PlanWithoutRequirement_ExecutesEveryToolInAMultiCallBatch()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("ordinary-multi-call");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: false);
        var replay = new ReplayGeneration(
            "<tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"first\"}}\n"
                + "</tool_call><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"second\"}}\n"
                + "</tool_call>",
            "<think>both done</think>Ordinary batch complete.");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        Assert.Equal(2, runner.Executions);
        Assert.Equal(2, plan.Invocations.Count);
        Assert.Equal("Ordinary batch complete.", Content(updates));
    }

    [Fact]
    public void VerifierRejectsStaleEscapedAndMalformedDecks()
    {
        // A valid file already in a persistent session is stale unless a tool in THIS
        // request captured it.
        SessionWorkspace staleWorkspace = _workspaces.GetOrCreate("stale");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan stalePlan = Plan(runner, staleWorkspace, guarded: true);
        string stalePath = Path.Combine(staleWorkspace.WorkDirectory, "report.pptx");
        WriteValidPptx(stalePath);
        WorkspaceArtifactCompletionResult stale = WorkspaceArtifactCompletion.Verify(stalePlan);
        Assert.False(stale.Complete);
        Assert.Contains("this turn", stale.Reason, StringComparison.Ordinal);

        // Structured metadata cannot rename the file behind a real artifact-store URL.
        SessionWorkspace escapedWorkspace = _workspaces.GetOrCreate("escaped");
        SkillRequestPlan escapedPlan = Plan(runner, escapedWorkspace, guarded: true);
        string safePath = Path.Combine(escapedWorkspace.WorkDirectory, "safe.pptx");
        WriteValidPptx(safePath);
        SkillProducedFile safeCapture = Capture(escapedWorkspace, "safe.pptx");
        AddCapture(escapedPlan, safeCapture with { Name = "../escaped.pptx" });
        WorkspaceArtifactCompletionResult escaped = WorkspaceArtifactCompletion.Verify(escapedPlan);
        Assert.False(escaped.Complete);
        Assert.Contains("download URL", escaped.Reason, StringComparison.OrdinalIgnoreCase);

        // ZIP magic and an extension are not a PowerPoint package.
        SessionWorkspace malformedWorkspace = _workspaces.GetOrCreate("malformed");
        SkillRequestPlan malformedPlan = Plan(runner, malformedWorkspace, guarded: true);
        string malformedPath = Path.Combine(malformedWorkspace.WorkDirectory, "report.pptx");
        File.WriteAllBytes(malformedPath, new byte[] { 0x50, 0x4b, 0x03, 0x04, 1, 2, 3, 4 });
        AddCapture(malformedPlan, Capture(malformedWorkspace, "report.pptx"));
        Assert.False(WorkspaceArtifactCompletion.Verify(malformedPlan).Complete);
    }

    [Fact]
    public void VerifierRejectsTheFormerFourPartFakePackage()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("former-fake");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteFormerFakePptx(path);
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("required package part", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierRejectsACitedSlideWithoutItsLayoutRelationship()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("missing-layout");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, includeSlideLayoutRelationship: false);
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("slide layout", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierRejectsAMasterWithoutItsCitedLayoutRelationship()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("missing-master-layout");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, includeMasterLayoutRelationship: false);
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("slide-layout relationship", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierRejectsALayoutWithoutItsReciprocalMasterRelationship()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("missing-layout-master");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, includeLayoutMasterRelationship: false);
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("reciprocal slide-master", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierRejectsAValidDeckWithExternalLinkedMedia()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("external-linked-media");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(
            path,
            extraSlideRelationship:
                "<Relationship Id=\"rId9\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/image\" "
                + "Target=\"https://tracker.example.test/pixel.png\" TargetMode=\"External\"/>");
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("external or unsupported", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierRejectsMacroEnabledContentEvenWhenTheOrdinaryDeckIsOtherwiseValid()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("macro-content");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(
            path,
            extraContentType:
                "<Override PartName=\"/ppt/vbaProject.bin\" ContentType=\"application/vnd.ms-office.vbaProject\"/>",
            extraPart: ("ppt/vbaProject.bin", "not executable, but must still be rejected"));
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("passive PowerPoint content types", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierValidatesTheCapturedDownload_NotASameLengthWorkspaceReplacement()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("same-length-rewrite");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");

        byte[] invalid = MakePptxBytes(WriteFormerFakePptx);
        byte[] valid = MakePptxBytes(path => WriteValidPptx(path));
        int commonLength = Math.Max(invalid.Length, valid.Length);
        Array.Resize(ref invalid, commonLength);
        Array.Resize(ref valid, commonLength);

        File.WriteAllBytes(path, invalid);
        SkillProducedFile capturedInvalid = Capture(workspace, "report.pptx");
        AddCapture(plan, capturedInvalid);

        // This is the old false positive: the workspace now has a valid ZIP of exactly
        // the reported size, while the URL still downloads the invalid captured copy.
        File.WriteAllBytes(path, valid);
        Assert.Equal(capturedInvalid.Bytes, new FileInfo(path).Length);

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("required package part", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void VerifierAcceptsAnImmutableValidDownloadAfterWorkspaceChanges()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("workspace-changed");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path);
        AddCapture(plan, Capture(workspace, "report.pptx"));
        File.WriteAllBytes(path, new byte[] { 1, 2, 3, 4 });

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.True(result.Complete, result.Reason);
        Assert.Contains("/artifacts/", result.Artifact?.Url, StringComparison.Ordinal);
    }

    [Fact]
    public void TouchingAnUnchangedPreRequestDeckDoesNotCompleteThisTurn()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("touched-stale");
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path);

        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        File.SetLastWriteTimeUtc(path, DateTime.UtcNow.AddSeconds(2));
        AddCapture(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("unchanged deck", result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RoutedEvidence_AcceptsOnlyTheResearchBackedDocumentsWriterDeck()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-valid");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, slideCount: 4, visibleText: "Apple M5 versus M6 — https://example.test/source");
        AddRoutedRuns(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.True(result.Complete, result.Reason);
    }

    [Fact]
    public void RoutedEvidence_RequiresAVisibleCitationFromTheResearchOutput()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-citation");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        WebUiArtifactRequirement policy = RoutedPolicy() with
        {
            CitationEvidencePath = "notes.md",
        };
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: policy);
        File.WriteAllText(
            Path.Combine(workspace.WorkDirectory, "notes.md"),
            "Research source: https://research.example.test/apple-m6");
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(
            path,
            slideCount: 4,
            visibleText: "Apple M5 versus M6 — https://unrelated.example.test/source");
        AddRoutedRuns(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult rejected = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(rejected.Complete);
        Assert.Contains("does not visibly cite any URL from notes.md", rejected.Reason, StringComparison.Ordinal);

        SkillRequestPlan matchingPlan = Plan(
            runner, workspace, guarded: true, policy: policy);
        WriteValidPptx(
            path,
            slideCount: 4,
            visibleText: "Apple M5 versus M6 — https://research.example.test/apple-m6");
        AddRoutedRuns(matchingPlan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult accepted = WorkspaceArtifactCompletion.Verify(matchingPlan);
        Assert.True(accepted.Complete, accepted.Reason);
    }

    [Theory]
    [InlineData("https://research.example.test/apple-m6-fabricated")]
    [InlineData("https://unrelated.example.test/?next=https://research.example.test/apple-m6")]
    public void RoutedEvidence_DoesNotAcceptAResearchUrlAsADeckUrlSubstring(string visibleUrl)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-citation-substring");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        WebUiArtifactRequirement policy = RoutedPolicy() with
        {
            CitationEvidencePath = "notes.md",
        };
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: policy);
        File.WriteAllText(
            Path.Combine(workspace.WorkDirectory, "notes.md"),
            "Research source: https://research.example.test/apple-m6");
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(
            path,
            slideCount: 4,
            visibleText: $"Apple M5 versus M6 — {visibleUrl}");
        AddRoutedRuns(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("does not visibly cite any URL from notes.md", result.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void CitationEvidencePathMustRemainInsideTheWorkspace()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-citation-path");

        bool created = WorkspaceArtifactCompletionRequirement.TryCreate(
            RoutedPolicy() with { CitationEvidencePath = "../notes.md" },
            workspace,
            _artifacts,
            ArtifactPrefix,
            out _);

        Assert.False(created);
    }

    [Fact]
    public void RoutedEvidence_RejectsADeckWhenResearchDidNotSucceed()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-no-research");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, slideCount: 4, visibleText: "M5 M6 https://example.test/source");
        SkillProducedFile artifact = Capture(workspace, "report.pptx");
        plan.Invocations.Add(RunInvocation("research", "scripts/research.py", ok: false));
        plan.Invocations.Add(RunInvocation(
            "documents", "scripts/make_pptx.py", ok: true, artifact));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("research/scripts/research.py", result.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void RoutedEvidence_RejectsAnUnrelatedShellArtifact()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-shell-artifact");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, slideCount: 4, visibleText: "M5 M6 https://example.test/source");
        SkillProducedFile artifact = Capture(workspace, "report.pptx");
        plan.Invocations.Add(RunInvocation("research", "scripts/research.py", ok: true));
        plan.Invocations.Add(RunInvocation("documents", "scripts/make_pptx.py", ok: true));
        AddCapture(plan, artifact);

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("No .pptx file", result.Reason, StringComparison.Ordinal);
    }

    [Fact]
    public void RoutedEvidence_RejectsADeckProducedBeforeTheRequiredResearch()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-out-of-order-artifact");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, slideCount: 4, visibleText: "M5 M6 https://example.test/source");
        SkillProducedFile earlyArtifact = Capture(workspace, "report.pptx");

        plan.Invocations.Add(RunInvocation(
            "documents", "scripts/make_pptx.py", ok: true, earlyArtifact));
        plan.Invocations.Add(RunInvocation("research", "scripts/research.py", ok: true));
        plan.Invocations.Add(RunInvocation("documents", "scripts/make_pptx.py", ok: true));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains("No .pptx file", result.Reason, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(1, "M5 M6 https://example.test/source", "at least 4")]
    [InlineData(4, "M5 https://example.test/source", "term 'M6'")]
    [InlineData(4, "M5 M6 source listed in package namespaces only", "visibly contain an http")]
    public void RoutedEvidence_RejectsIncompleteVisibleReportContent(
        int slideCount,
        string visibleText,
        string expectedReason)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-content-" + Guid.NewGuid().ToString("N"));
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        string path = Path.Combine(workspace.WorkDirectory, "report.pptx");
        WriteValidPptx(path, slideCount: slideCount, visibleText: visibleText);
        AddRoutedRuns(plan, Capture(workspace, "report.pptx"));

        WorkspaceArtifactCompletionResult result = WorkspaceArtifactCompletion.Verify(plan);

        Assert.False(result.Complete);
        Assert.Contains(expectedReason, result.Reason, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void RoutedDefaults_CompleteOnlyOmittedFieldsForTheExactRequiredScript()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("routed-defaults");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        var policy = new WebUiArtifactRequirement(
            ".pptx",
            new[]
            {
                new WebUiSkillRunRequirement(
                    "research",
                    "scripts/research.py",
                    DefaultArguments: new[] { "query words", "--out", "notes.md" }),
            });
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: policy);
        var omitted = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["path"] = "./research/scripts/research.py",
                ["packages"] = "lxml",
            },
        };

        Assert.True(SkillChatLoop.ApplyRoutedDefaults(omitted, plan.CompletionRequirement));
        Assert.Equal("research", omitted.Arguments["skill"]);
        Assert.Equal("scripts/research.py", omitted.Arguments["path"]);
        Assert.Equal(
            new[] { "query words", "--out", "notes.md" },
            Assert.IsType<string[]>(omitted.Arguments["args"]));
        Assert.Equal("lxml", omitted.Arguments["packages"]);

        var supplied = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "research",
                ["path"] = "scripts/research.py",
                ["args"] = new[] { "custom query", "--out", "custom.md" },
            },
        };

        Assert.False(SkillChatLoop.ApplyRoutedDefaults(supplied, plan.CompletionRequirement));
        Assert.Equal(
            new[] { "custom query", "--out", "custom.md" },
            Assert.IsType<string[]>(supplied.Arguments["args"]));

        var suppliedAlias = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "research",
                ["script"] = "scripts/research.py",
                ["arguments"] = "alias query --out alias.md",
            },
        };

        Assert.False(SkillChatLoop.ApplyRoutedDefaults(suppliedAlias, plan.CompletionRequirement));
        Assert.False(suppliedAlias.Arguments.ContainsKey("args"));
        Assert.Equal("alias query --out alias.md", suppliedAlias.Arguments["arguments"]);

        var enforcedPolicy = new WebUiArtifactRequirement(
            ".pptx",
            new[]
            {
                new WebUiSkillRunRequirement(
                    "research",
                    "scripts/research.py",
                    DefaultArguments: new[] { "bounded query", "--pages", "3", "--out", "notes.md" },
                    EnforceArguments: true),
            });
        SkillRequestPlan enforcedPlan = Plan(
            runner, workspace, guarded: true, policy: enforcedPolicy);
        var overBroad = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "research",
                ["path"] = "scripts/research.py",
                ["args"] = new[] { "different query", "--pages", "50", "--out", "elsewhere.md" },
            },
        };

        Assert.True(SkillChatLoop.ApplyRoutedDefaults(
            overBroad, enforcedPlan.CompletionRequirement));
        Assert.Equal(
            new[] { "bounded query", "--pages", "3", "--out", "notes.md" },
            Assert.IsType<string[]>(overBroad.Arguments["args"]));

        var wrongSkill = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "documents",
                ["path"] = "scripts/research.py",
            },
        };
        Assert.False(SkillChatLoop.ApplyRoutedDefaults(
            wrongSkill, plan.CompletionRequirement));
        Assert.False(wrongSkill.Arguments.ContainsKey("args"));

        var unrelatedScript = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["skill"] = "research",
                ["path"] = "scripts/analyze.py",
            },
        };
        Assert.False(SkillChatLoop.ApplyRoutedDefaults(
            unrelatedScript, plan.CompletionRequirement));
        Assert.False(unrelatedScript.Arguments.ContainsKey("args"));

        var caseVariant = new ToolCall
        {
            Name = SkillTools.RunToolName,
            Arguments = new Dictionary<string, object>
            {
                ["path"] = "SCRIPTS/RESEARCH.PY",
            },
        };
        bool caseVariantChanged = SkillChatLoop.ApplyRoutedDefaults(
            caseVariant, plan.CompletionRequirement);
        Assert.Equal(
            SkillPathGuard.PathComparison == StringComparison.OrdinalIgnoreCase,
            caseVariantChanged);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RoutedWriter_MissingOrEmptyRequiredSpecNeverLaunchesItsBackend(bool createEmptySpec)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate(
            createEmptySpec ? "writer-empty-spec" : "writer-missing-spec");
        if (createEmptySpec)
            File.WriteAllBytes(Path.Combine(workspace.WorkDirectory, "pptx_spec.json"), Array.Empty<byte>());

        var backend = new FakeShellBackend();
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None) { Backend = backend };
        SkillRequestPlan plan = ScriptPlan(
            runner,
            workspace,
            RequiredInputPolicy("pptx_spec.json"));
        var replay = new ReplayGeneration(
            "<tool_call>\n"
                + "{\"name\": \"skills_run\", \"arguments\": {\"skill\": \"documents\", "
                + "\"path\": \"scripts/make_pptx.py\"}}\n"
                + "</tool_call>",
            "I think it is done.",
            "No repair call.");

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Empty(backend.Launches);
        SkillToolInvocation blocked = Assert.Single(plan.Invocations);
        Assert.False(blocked.Ok);
        Assert.Equal(SkillTools.RunToolName, blocked.Tool);
        Assert.Equal("documents", blocked.SkillId);
        Assert.Equal("scripts/make_pptx.py", blocked.ResourcePath);
        Assert.Contains(
            "required input 'pptx_spec.json' is missing or empty",
            replay.Histories[1],
            StringComparison.Ordinal);
        Assert.Null(plan.VerifiedArtifact);
        Assert.Contains("no valid downloadable .pptx", Content(updates), StringComparison.Ordinal);
    }

    [Fact]
    public async Task RoutedWriter_NonEmptyRequiredSpecIsAllowedToLaunch()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("writer-present-spec");
        File.WriteAllText(Path.Combine(workspace.WorkDirectory, "pptx_spec.json"), "{}");
        var backend = new FakeShellBackend();
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None) { Backend = backend };
        SkillRequestPlan plan = ScriptPlan(
            runner,
            workspace,
            RequiredInputPolicy("pptx_spec.json"));
        var replay = new ReplayGeneration(
            "<tool_call>\n"
                + "{\"name\": \"skills_run\", \"arguments\": {\"skill\": \"documents\", "
                + "\"path\": \"scripts/make_pptx.py\"}}\n"
                + "</tool_call>",
            "The writer ran but made no deck.",
            "No repair call.");

        await Run(plan, replay.Invoke);

        Assert.Single(backend.Launches);
        SkillToolInvocation invocation = Assert.Single(plan.Invocations);
        Assert.True(invocation.Ok);
        Assert.Equal("documents", invocation.SkillId);
        Assert.Equal("scripts/make_pptx.py", invocation.ResourcePath);
    }

    [Fact]
    public async Task RoutedWriter_WithNoInputPrerequisiteRetainsOrdinaryScriptBehavior()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("writer-no-prerequisite");
        var backend = new FakeShellBackend();
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None) { Backend = backend };
        SkillRequestPlan plan = ScriptPlan(runner, workspace, RequiredInputPolicy(requiredInputPath: null));
        var replay = new ReplayGeneration(
            "<tool_call>\n"
                + "{\"name\": \"skills_run\", \"arguments\": {\"skill\": \"documents\", "
                + "\"path\": \"scripts/make_pptx.py\"}}\n"
                + "</tool_call>",
            "The ordinary routed script ran.",
            "No repair call.");

        await Run(plan, replay.Invoke);

        Assert.Single(backend.Launches);
        Assert.True(Assert.Single(plan.Invocations).Ok);
    }

    [Theory]
    [InlineData("../pptx_spec.json")]
    [InlineData("inputs/../pptx_spec.json")]
    public void RequiredInputPathMustBeCanonicalAndStayInsideTheWorkspace(string requiredInputPath)
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("writer-invalid-input-path");

        bool created = WorkspaceArtifactCompletionRequirement.TryCreate(
            RequiredInputPolicy(requiredInputPath),
            workspace,
            _artifacts,
            ArtifactPrefix,
            out _);

        Assert.False(created);
    }

    [Fact]
    public void RequiredInputPathHasABoundedRouteContract()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("writer-long-input-path");

        bool created = WorkspaceArtifactCompletionRequirement.TryCreate(
            RequiredInputPolicy(new string('x', 513)),
            workspace,
            _artifacts,
            ArtifactPrefix,
            out _);

        Assert.False(created);
    }

    [Fact]
    public void GuardedWebUiTraceHoldsFilesUntilProof_AndReleasesOnlyVerifiedArtifact()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("guarded-trace");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true, policy: RoutedPolicy());
        var invalid = new SkillProducedFile(
            "broken.pptx", 4, "/artifacts/invalid/broken.pptx");
        var verified = new SkillProducedFile(
            "report.pptx", 100, "/artifacts/valid/report.pptx");
        plan.Invocations.Add(new SkillToolInvocation(
            1, SkillTools.RunToolName, "documents", "scripts/make_pptx.py", false, 10)
        {
            Files = new[] { invalid },
        });
        plan.Invocations.Add(new SkillToolInvocation(
            2, SkillTools.RunToolName, "documents", "scripts/make_pptx.py", true, 10)
        {
            Files = new[] { invalid, verified },
        });

        (SkillToolInvocation[] failed, int failedWatermark) =
            WebUiChatService.DrainSkillTrace(plan, reported: 0);
        Assert.Equal(2, failedWatermark);
        Assert.All(failed, invocation => Assert.Empty(invocation.Files));
        foreach (SkillToolInvocation invocation in failed)
        {
            using JsonDocument frame = JsonDocument.Parse(
                JsonSerializer.Serialize(WebUiSseEvents.SkillStep(invocation)));
            Assert.Equal(JsonValueKind.Null, frame.RootElement.GetProperty("files").ValueKind);
        }

        bool artifactReported = false;
        Assert.Null(WebUiChatService.TakeVerifiedArtifact(plan, ref artifactReported));
        plan.VerifiedArtifact = verified;
        (SkillToolInvocation[] successful, int successfulWatermark) =
            WebUiChatService.DrainSkillTrace(plan, reported: 0);
        Assert.Equal(2, successfulWatermark);
        Assert.Empty(successful[0].Files);
        Assert.Empty(successful[1].Files);

        SkillProducedFile released = Assert.IsType<SkillProducedFile>(
            WebUiChatService.TakeVerifiedArtifact(plan, ref artifactReported));
        Assert.Equal(verified, released);
        Assert.Null(WebUiChatService.TakeVerifiedArtifact(plan, ref artifactReported));

        using JsonDocument verifiedFrame = JsonDocument.Parse(
            JsonSerializer.Serialize(WebUiSseEvents.VerifiedArtifact(released)));
        Assert.True(verifiedFrame.RootElement.GetProperty("artifact_verified").GetBoolean());
        JsonElement file = Assert.Single(verifiedFrame.RootElement.GetProperty("files").EnumerateArray());
        Assert.Equal(verified.Url, file.GetProperty("url").GetString());
        Assert.NotEqual(invalid.Url, file.GetProperty("url").GetString());
    }

    /// <summary>
    /// The failure this answers, from a phone: a tool call whose script degenerated into
    /// <c>","+","+"</c> for five minutes. The engine now ends such a round with
    /// <c>repetition</c>; the loop must not dispatch what that round wrote, must tell
    /// the model what repeated, and must let it try once more, differently.
    /// </summary>
    [Fact]
    public async Task ARoundTheEngineStoppedForLooping_IsNotDispatched_AndTheModelIsToldWhatRepeated()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("looping");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: false);
        // A CLOSED call first, so the round really does carry a parsed tool call for the
        // loop to refuse; the runaway one after it is the truncated JSON a real
        // repetition stop leaves behind, and parses to nothing.
        string looping = "<think>write the deck</think><tool_call>\n"
            + "{\"name\": \"shell\", \"arguments\": {\"command\": \"mkdir deck\"}}\n"
            + "</tool_call><tool_call>\n"
            + "{\"name\": \"shell\", \"arguments\": {\"command\": \"python3 -c 'theme = \\\"<a:latin bon=\\\"Aa+"
            + string.Concat(Enumerable.Repeat("\",\"+", 80));
        var replay = new ReplayGeneration(looping, "<think>use a loop instead</think>Written differently.")
        {
            FinishReasons = new[] { "repetition", "stop" },
        };

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        // The round DID carry a dispatchable call, and it was not dispatched.
        Assert.Contains(replay.MessageHistories[1], m =>
            m.Role == "assistant" && m.ToolCalls is { Count: > 0 });
        Assert.Equal(0, runner.Executions);
        string content = Content(updates);
        Assert.Contains("Written differently.", content, StringComparison.Ordinal);
        Assert.DoesNotContain("\",\"+\",\"+", content, StringComparison.Ordinal);
        Assert.Contains("started repeating itself", Thinking(updates), StringComparison.Ordinal);
        string secondHistory = replay.Histories[1];
        Assert.Contains("Host: your previous output was stopped because it began repeating itself", secondHistory, StringComparison.Ordinal);
        Assert.Contains("repeated `\",\"+` 80 times in a row", secondHistory, StringComparison.Ordinal);
        Assert.Contains("nothing from it was run", secondHistory, StringComparison.Ordinal);
        Assert.Single(updates, update => update.Done);
    }

    /// <summary>
    /// The one-shot artifact correction EXECUTES what it parses, and it is the last
    /// round there is — so a correction the engine stopped for looping must not have its
    /// half-written tool call run. Reported instead, with the deck honestly declared
    /// missing.
    /// </summary>
    [Fact]
    public async Task ACorrectionRoundStoppedForLooping_RunsNothingAndSaysTheDeckIsMissing()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("correction-loops");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.Valid);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: true);
        var replay = new ReplayGeneration(
            "<think>done</think>I made it: [report](fake.pptx).",
            "<think>repairing</think><tool_call>\n"
                + "{\"name\": \"shell\", \"arguments\": {\"command\": \"make deck\"}}\n"
                + "</tool_call>then it drifts: " + string.Concat(Enumerable.Repeat("na ", 60)))
        {
            FinishReasons = new[] { "stop", "repetition" },
        };

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        // The correction carried a dispatchable call and it was NOT run.
        Assert.Equal(0, runner.Executions);
        string content = Content(updates);
        Assert.Contains("couldn\'t complete the PowerPoint report", content, StringComparison.Ordinal);
        Assert.Contains("started repeating itself and was stopped", content, StringComparison.Ordinal);
        Assert.DoesNotContain("fake.pptx", content, StringComparison.Ordinal);
        Assert.Single(updates, update => update.Done);
    }

    /// <summary>One retry, not a loop of retries: a second looping round ends the turn with the note.</summary>
    [Fact]
    public async Task ASecondLoopingRound_EndsTheTurnWithTheExplanation()
    {
        SessionWorkspace workspace = _workspaces.GetOrCreate("looping-twice");
        var runner = new ArtifactRunner(_artifacts, ArtifactMode.None);
        SkillRequestPlan plan = Plan(runner, workspace, guarded: false);
        string looping = "Here it is: " + string.Concat(Enumerable.Repeat("again and ", 40));
        var replay = new ReplayGeneration(looping, looping, "never reached")
        {
            FinishReasons = new[] { "repetition", "repetition", "stop" },
        };

        List<ChatStreamUpdate> updates = await Run(plan, replay.Invoke);

        Assert.Equal(2, replay.Calls);
        string content = Content(updates);
        Assert.Contains("The model's output started repeating itself and was stopped", content, StringComparison.Ordinal);
        Assert.Contains("repeated `again and ` 40 times in a row", content, StringComparison.Ordinal);
        ChatStreamUpdate done = Assert.Single(updates, update => update.Done);
        Assert.Equal("repetition", done.FinishReason);
        // The quoted note above IS the explanation, so a UI must not add a plain one.
        Assert.True(done.RepetitionExplained);
    }

    private SkillRequestPlan Plan(
        ArtifactRunner runner,
        SessionWorkspace workspace,
        bool guarded,
        int? maxRounds = null,
        WebUiArtifactRequirement? policy = null)
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        var args = new List<string> { "--model", "x.gguf", "--skills-dir", _baseDir };
        if (maxRounds.HasValue)
        {
            args.Add("--skills-max-rounds");
            args.Add(maxRounds.Value.ToString());
        }
        ServerHostingOptions options = ServerOptionsBuilder.Build(args.ToArray(), _baseDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, Array.Empty<string>(), false, null, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown,
            codeRunner: runner, workspace: workspace);
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        if (guarded)
        {
            WorkspaceArtifactCompletionRequirement requirement;
            bool created = policy == null
                ? WorkspaceArtifactCompletionRequirement.TryCreate(
                    ".pptx", workspace, _artifacts, ArtifactPrefix,
                    out requirement)
                : WorkspaceArtifactCompletionRequirement.TryCreate(
                    policy, workspace, _artifacts, ArtifactPrefix,
                    out requirement);
            Assert.True(created);
            plan.CompletionRequirement = requirement;
        }
        return plan;
    }

    private SkillRequestPlan ScriptPlan(
        ArtifactRunner runner,
        SessionWorkspace workspace,
        WebUiArtifactRequirement policy)
    {
        string root = Path.Combine(_baseDir, "documents");
        string scripts = Path.Combine(root, "scripts");
        Directory.CreateDirectory(scripts);
        File.WriteAllText(
            Path.Combine(root, "SKILL.md"),
            "---\nname: documents\ndescription: Create documents.\n---\n\nRun the bundled writer.\n");
        File.WriteAllText(
            Path.Combine(scripts, "make_pptx.py"),
            "from pathlib import Path\nPath('writer-entered').write_text('yes')\n");

        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[]
            {
                "--model", "x.gguf",
                SkillHostOptions.RootsFlag, _baseDir,
                SkillHostOptions.AllowScriptsFlag,
                SkillHostOptions.SandboxFlag, "off",
            },
            _baseDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, new[] { "documents" }, false, null, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown,
            codeRunner: runner,
            workspace: workspace,
            captureProducedFiles: (workDirectory, exclude) =>
            {
                string runId = Guid.NewGuid().ToString("N");
                return _artifacts.Capture(
                        runId,
                        workDirectory,
                        (id, relative, _) => CodeArtifactStore.UrlFor(ArtifactPrefix, id, relative),
                        out _,
                        exclude)
                    .Select(artifact => new SkillProducedFile(
                        artifact.Path, artifact.Bytes, artifact.Pointer))
                    .ToArray();
            });
        Assert.Empty(unknown);
        Assert.NotNull(plan);
        Assert.True(WorkspaceArtifactCompletionRequirement.TryCreate(
            policy, workspace, _artifacts, ArtifactPrefix,
            out WorkspaceArtifactCompletionRequirement requirement));
        plan.CompletionRequirement = requirement;
        return plan;
    }

    private static WebUiArtifactRequirement RequiredInputPolicy(string? requiredInputPath) => new(
        ".pptx",
        new[]
        {
            new WebUiSkillRunRequirement(
                "documents",
                "scripts/make_pptx.py",
                ProducesArtifact: true,
                RequiredInputPath: requiredInputPath),
        });

    private static WebUiArtifactRequirement RoutedPolicy() => new(
        ".pptx",
        new[]
        {
            new WebUiSkillRunRequirement("research", "scripts/research.py"),
            new WebUiSkillRunRequirement(
                "documents", "scripts/make_pptx.py", ProducesArtifact: true),
        },
        MinimumSlides: 4,
        RequiredVisibleTerms: new[] { "M5", "M6" },
        RequireVisibleHttpUrl: true);

    private static void AddRoutedRuns(SkillRequestPlan plan, SkillProducedFile artifact)
    {
        plan.Invocations.Add(RunInvocation("research", "scripts/research.py", ok: true));
        plan.Invocations.Add(RunInvocation(
            "documents", "scripts/make_pptx.py", ok: true, artifact));
    }

    private static SkillToolInvocation RunInvocation(
        string skill,
        string path,
        bool ok,
        SkillProducedFile? artifact = null) =>
        new(1, SkillTools.RunToolName, skill, path, ok, 0)
        {
            Files = artifact.HasValue
                ? new[] { artifact.Value }
                : Array.Empty<SkillProducedFile>(),
        };

    private static async Task<List<ChatStreamUpdate>> Run(
        SkillRequestPlan plan,
        SkillChatGeneration generate)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "make the deck" } };
        var updates = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync(
            Architecture, messages, plan, enableThinking: true,
            generate, logger: null, CancellationToken.None))
        {
            updates.Add(update);
        }
        return updates;
    }

    private static string Content(IEnumerable<ChatStreamUpdate> updates) =>
        string.Concat(updates.Where(update => !update.Done).Select(update => update.Piece));

    private static string Thinking(IEnumerable<ChatStreamUpdate> updates) =>
        string.Concat(updates.Where(update => !update.Done).Select(update => update.ThinkingPiece));

    private SkillProducedFile Capture(SessionWorkspace workspace, string relativePath)
    {
        string normalized = relativePath.Replace('\\', '/');
        string runId = Guid.NewGuid().ToString("N");
        CodeArtifact artifact = Assert.Single(_artifacts.Capture(
            runId,
            workspace.WorkDirectory,
            (id, relative, _) => CodeArtifactStore.UrlFor(ArtifactPrefix, id, relative),
            out _,
            relative => !string.Equals(relative.Replace('\\', '/'), normalized, StringComparison.Ordinal)));
        return new SkillProducedFile(artifact.Path, artifact.Bytes, artifact.Pointer);
    }

    private static void AddCapture(SkillRequestPlan plan, SkillProducedFile artifact)
    {
        plan.Invocations.Add(new SkillToolInvocation(1, "shell", null, null, true, 0)
        {
            Files = new[] { artifact },
        });
    }

    private static byte[] MakePptxBytes(Action<string> write)
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-pptx-fixture-" + Guid.NewGuid().ToString("N") + ".pptx");
        try
        {
            write(path);
            return File.ReadAllBytes(path);
        }
        finally
        {
            try { File.Delete(path); } catch { /* best effort */ }
        }
    }

    private static void WriteValidPptx(
        string path,
        bool includeSlideLayoutRelationship = true,
        int slideCount = 1,
        string visibleText = "TensorAgent report",
        bool includeMasterLayoutRelationship = true,
        bool includeLayoutMasterRelationship = true,
        string extraSlideRelationship = "",
        string extraContentType = "",
        (string Name, string Body)? extraPart = null)
    {
        if (slideCount < 1)
            throw new ArgumentOutOfRangeException(nameof(slideCount));
        string slideOverrides = string.Concat(Enumerable.Range(1, slideCount).Select(number =>
            $"<Override PartName=\"/ppt/slides/slide{number}.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slide+xml\"/>"));
        string slideIds = string.Concat(Enumerable.Range(1, slideCount).Select(number =>
            $"<p:sldId id=\"{255 + number}\" r:id=\"rId{1 + number}\"/>"));
        string slideRelationships = string.Concat(Enumerable.Range(1, slideCount).Select(number =>
            $"<Relationship Id=\"rId{1 + number}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide\" Target=\"slides/slide{number}.xml\"/>"));
        string escapedVisibleText = System.Security.SecurityElement.Escape(visibleText) ?? string.Empty;

        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);
        WriteEntry(archive, "[Content_Types].xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\">"
            + "<Default Extension=\"rels\" ContentType=\"application/vnd.openxmlformats-package.relationships+xml\"/>"
            + "<Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            + "<Override PartName=\"/ppt/presentation.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml\"/>"
            + "<Override PartName=\"/ppt/slideMasters/slideMaster1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml\"/>"
            + "<Override PartName=\"/ppt/slideLayouts/slideLayout1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml\"/>"
            + slideOverrides
            + "<Override PartName=\"/ppt/theme/theme1.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.theme+xml\"/>"
            + "<Override PartName=\"/ppt/presProps.xml\" ContentType=\"application/vnd.openxmlformats-officedocument.presentationml.presProps+xml\"/>"
            + extraContentType
            + "</Types>");
        WriteEntry(archive, "_rels/.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            + "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"ppt/presentation.xml\"/>"
            + "</Relationships>");
        WriteEntry(archive, "ppt/presentation.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<p:presentation xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" "
            + "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">"
            + "<p:sldMasterIdLst><p:sldMasterId id=\"2147483648\" r:id=\"rId1\"/></p:sldMasterIdLst>"
            + "<p:sldIdLst>" + slideIds + "</p:sldIdLst>"
            + "<p:sldSz cx=\"12192000\" cy=\"6858000\"/>"
            + "<p:notesSz cx=\"6858000\" cy=\"9144000\"/>"
            + "</p:presentation>");
        WriteEntry(archive, "ppt/_rels/presentation.xml.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            + "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"slideMasters/slideMaster1.xml\"/>"
            + slideRelationships
            + $"<Relationship Id=\"rId{slideCount + 2}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/presProps\" Target=\"presProps.xml\"/>"
            + $"<Relationship Id=\"rId{slideCount + 3}\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme\" Target=\"theme/theme1.xml\"/>"
            + "</Relationships>");
        WriteEntry(archive, "ppt/slideMasters/slideMaster1.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<p:sldMaster xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
            + "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
            + "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
            + "<p:cSld><p:spTree>" + EmptyShapeTree
            + "</p:spTree></p:cSld>" + ColorMap
            + "<p:sldLayoutIdLst><p:sldLayoutId id=\"2147483649\" r:id=\"rId1\"/></p:sldLayoutIdLst>"
            + "<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles>"
            + "</p:sldMaster>");
        WriteEntry(archive, "ppt/slideMasters/_rels/slideMaster1.xml.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            + (includeMasterLayoutRelationship
                ? "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>"
                : string.Empty)
            + "<Relationship Id=\"rId2\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme\" Target=\"../theme/theme1.xml\"/>"
            + "</Relationships>");
        WriteEntry(archive, "ppt/slideLayouts/slideLayout1.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<p:sldLayout xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
            + "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
            + "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\" type=\"blank\" preserve=\"1\">"
            + "<p:cSld name=\"Blank\"><p:spTree>" + EmptyShapeTree
            + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
            + "</p:sldLayout>");
        WriteEntry(archive, "ppt/slideLayouts/_rels/slideLayout1.xml.rels",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
            + (includeLayoutMasterRelationship
                ? "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster\" Target=\"../slideMasters/slideMaster1.xml\"/>"
                : string.Empty)
            + "</Relationships>");
        for (int number = 1; number <= slideCount; number++)
        {
            WriteEntry(archive, $"ppt/slides/slide{number}.xml",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                + "<p:sld xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" "
                + "xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\" "
                + "xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\">"
                + "<p:cSld><p:spTree>" + EmptyShapeTree
                + $"<p:sp><p:nvSpPr><p:cNvPr id=\"2\" name=\"Title {number}\"/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>"
                + "<p:spPr/><p:txBody><a:bodyPr/><a:lstStyle/><a:p><a:r><a:rPr lang=\"en-US\"/>"
                + $"<a:t>{escapedVisibleText}</a:t></a:r></a:p></p:txBody></p:sp>"
                + "</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>"
                + "</p:sld>");
            WriteEntry(archive, $"ppt/slides/_rels/slide{number}.xml.rels",
                "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                + "<Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">"
                + (includeSlideLayoutRelationship
                    ? "<Relationship Id=\"rId1\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout\" Target=\"../slideLayouts/slideLayout1.xml\"/>"
                    : string.Empty)
                + extraSlideRelationship
                + "</Relationships>");
        }
        WriteEntry(archive, "ppt/theme/theme1.xml", ThemeXml);
        WriteEntry(archive, "ppt/presProps.xml",
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            + "<p:presentationPr xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"/>");
        if (extraPart is { } part)
            WriteEntry(archive, part.Name, part.Body);
    }

    private static void WriteFormerFakePptx(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Create);
        WriteEntry(archive, "[Content_Types].xml",
            "<?xml version=\"1.0\"?><Types xmlns=\"http://schemas.openxmlformats.org/package/2006/content-types\"/>");
        WriteEntry(archive, "ppt/presentation.xml",
            "<?xml version=\"1.0\"?><p:presentation xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"/>");
        WriteEntry(archive, "ppt/_rels/presentation.xml.rels",
            "<?xml version=\"1.0\"?><Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\"/>");
        WriteEntry(archive, "ppt/slides/slide1.xml",
            "<?xml version=\"1.0\"?><p:sld xmlns:p=\"http://schemas.openxmlformats.org/presentationml/2006/main\"/>");
    }

    private const string EmptyShapeTree =
        "<p:nvGrpSpPr><p:cNvPr id=\"1\" name=\"\"/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>"
        + "<p:grpSpPr><a:xfrm><a:off x=\"0\" y=\"0\"/><a:ext cx=\"0\" cy=\"0\"/>"
        + "<a:chOff x=\"0\" y=\"0\"/><a:chExt cx=\"0\" cy=\"0\"/></a:xfrm></p:grpSpPr>";

    private const string ColorMap =
        "<p:clrMap bg1=\"lt1\" tx1=\"dk1\" bg2=\"lt2\" tx2=\"dk2\" accent1=\"accent1\" "
        + "accent2=\"accent2\" accent3=\"accent3\" accent4=\"accent4\" accent5=\"accent5\" "
        + "accent6=\"accent6\" hlink=\"hlink\" folHlink=\"folHlink\"/>";

    private const string ThemeXml =
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        + "<a:theme xmlns:a=\"http://schemas.openxmlformats.org/drawingml/2006/main\" name=\"TensorAgent\">"
        + "<a:themeElements><a:clrScheme name=\"TensorAgent\">"
        + "<a:dk1><a:sysClr val=\"windowText\" lastClr=\"000000\"/></a:dk1>"
        + "<a:lt1><a:sysClr val=\"window\" lastClr=\"FFFFFF\"/></a:lt1>"
        + "<a:dk2><a:srgbClr val=\"1F1F1F\"/></a:dk2><a:lt2><a:srgbClr val=\"F2F2F2\"/></a:lt2>"
        + "<a:accent1><a:srgbClr val=\"4472C4\"/></a:accent1><a:accent2><a:srgbClr val=\"ED7D31\"/></a:accent2>"
        + "<a:accent3><a:srgbClr val=\"A5A5A5\"/></a:accent3><a:accent4><a:srgbClr val=\"FFC000\"/></a:accent4>"
        + "<a:accent5><a:srgbClr val=\"5B9BD5\"/></a:accent5><a:accent6><a:srgbClr val=\"70AD47\"/></a:accent6>"
        + "<a:hlink><a:srgbClr val=\"0563C1\"/></a:hlink><a:folHlink><a:srgbClr val=\"954F72\"/></a:folHlink>"
        + "</a:clrScheme><a:fontScheme name=\"TensorAgent\">"
        + "<a:majorFont><a:latin typeface=\"Calibri Light\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:majorFont>"
        + "<a:minorFont><a:latin typeface=\"Calibri\"/><a:ea typeface=\"\"/><a:cs typeface=\"\"/></a:minorFont>"
        + "</a:fontScheme><a:fmtScheme name=\"TensorAgent\">"
        + "<a:fillStyleLst><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill>"
        + "<a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:fillStyleLst>"
        + "<a:lnStyleLst><a:ln w=\"6350\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln>"
        + "<a:ln w=\"12700\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln>"
        + "<a:ln w=\"19050\"><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:ln></a:lnStyleLst>"
        + "<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle><a:effectStyle><a:effectLst/></a:effectStyle>"
        + "<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>"
        + "<a:bgFillStyleLst><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill>"
        + "<a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill><a:solidFill><a:schemeClr val=\"phClr\"/></a:solidFill></a:bgFillStyleLst>"
        + "</a:fmtScheme></a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>";

    private static void WriteEntry(ZipArchive archive, string name, string body)
    {
        ZipArchiveEntry entry = archive.CreateEntry(name);
        using var writer = new StreamWriter(entry.Open());
        writer.Write(body);
    }

    private enum ArtifactMode
    {
        None,
        Valid,
        TruncatedZip,
    }

    private sealed class ArtifactRunner : ICodeRunner
    {
        private readonly CodeArtifactStore _artifacts;
        private readonly ArtifactMode[] _modes;

        public ArtifactRunner(CodeArtifactStore artifacts, params ArtifactMode[] modes)
        {
            _artifacts = artifacts;
            _modes = modes;
        }

        public int Executions { get; private set; }
        public SkillProducedFile? LastArtifact { get; private set; }
        public Action<int, SessionWorkspace>? OnExecute { get; set; }
        public IShellBackend? Backend { get; set; }
        public bool CanRun => true;
        public string UnavailableReason => null;
        public ToolFunction Declare() => new() { Name = "shell", Description = "run a test command" };

        public SkillToolResult Execute(
            ToolCall call,
            IReadOnlyList<CodeInputFile> inputFiles = null,
            Action<string> onOutput = null,
            SessionWorkspace workspace = null,
            IReadOnlyList<string> skillDirectories = null)
        {
            ArtifactMode mode = _modes[Math.Min(Executions, _modes.Length - 1)];
            Executions++;
            OnExecute?.Invoke(Executions, workspace!);
            if (mode == ArtifactMode.None)
                return new SkillToolResult(false, "writer failed", null, null);

            string path = Path.Combine(workspace!.WorkDirectory, "report.pptx");
            if (mode == ArtifactMode.Valid)
                WriteValidPptx(path);
            else
                File.WriteAllBytes(path, new byte[] { 0x50, 0x4b, 0x03, 0x04, 1, 2, 3, 4 });

            string runId = Guid.NewGuid().ToString("N");
            CodeArtifact artifact = Assert.Single(_artifacts.Capture(
                runId,
                workspace.WorkDirectory,
                (id, relative, _) => CodeArtifactStore.UrlFor(ArtifactPrefix, id, relative),
                out _,
                relative => !string.Equals(relative.Replace('\\', '/'), "report.pptx", StringComparison.Ordinal)));
            LastArtifact = new SkillProducedFile(artifact.Path, artifact.Bytes, artifact.Pointer);
            return new SkillToolResult(true, "writer finished", null, null)
            {
                Files = new[] { LastArtifact.Value },
            };
        }
    }

    private sealed class ReplayGeneration
    {
        private readonly string[] _rounds;

        public ReplayGeneration(params string[] rounds) => _rounds = rounds;

        /// <summary>A finish reason per round, when one round must end differently from the rest.</summary>
        public string[]? FinishReasons { get; init; }

        public int Calls { get; private set; }
        public string FinishReason { get; init; } = "stop";
        public List<string> Histories { get; } = new();
        public List<IReadOnlyList<ChatMessage>> MessageHistories { get; } = new();

        public IAsyncEnumerable<ChatStreamUpdate> Invoke(
            List<ChatMessage> messages,
            List<ToolFunction> tools,
            CancellationToken cancellationToken)
        {
            if (Calls >= _rounds.Length)
                throw new InvalidOperationException("The loop generated more turns than this bounded test allows.");
            Histories.Add(string.Join("\n", messages.Select(message => message.Content)));
            MessageHistories.Add(new List<ChatMessage>(messages));
            int index = Calls++;
            string finish = FinishReasons != null && index < FinishReasons.Length && FinishReasons[index] != null
                ? FinishReasons[index]
                : FinishReason;
            return Emit(_rounds[index], finish, cancellationToken);
        }

        internal static async IAsyncEnumerable<ChatStreamUpdate> Emit(
            string text,
            string finishReason,
            [EnumeratorCancellation] CancellationToken cancellationToken)
        {
            foreach (char character in text)
            {
                cancellationToken.ThrowIfCancellationRequested();
                yield return ChatStreamUpdate.Text(character.ToString());
                await Task.Yield();
            }
            yield return new ChatStreamUpdate(
                string.Empty, true, 10, 20, 5, 30, 10, 20, finishReason);
        }
    }
}
