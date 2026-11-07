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
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// A source-scanning regression guard for the content contract: user content (prompts,
/// model output, upload names, paths and URLs) reaches a log call only wrapped in
/// <c>LogContent.Of(...)</c>, which local sinks print unchanged and the OpenTelemetry exporter
/// redacts. The redaction tests prove the wrapper works; this proves the call sites still use
/// it, because an unwrapped argument is exported verbatim and nothing else would notice.
///
/// <para>
/// It reads every <c>ILogger</c> call (<c>LogTrace</c> .. <c>LogCritical</c>, and
/// <c>Log(LogLevel, ...)</c>) in the production projects with a small C# lexer that blanks
/// comments and string-literal contents (keeping interpolation holes as code), so
/// parentheses and commas inside strings, chars and comments, calls spanning lines, verbatim,
/// raw and concatenated templates are all read the way the compiler reads them. The one
/// thing it does not model is a comma inside generic type arguments
/// (<c>new Dictionary&lt;string, int&gt;()</c>) at the top level of an argument;
/// <see cref="EveryConstantTemplateHasOneArgumentPerPlaceholder"/> would flag such a mis-split.
/// </para>
/// </summary>
public class ContentContractTests
{
    /// <summary>The production projects whose logging calls are scanned.</summary>
    private static readonly string[] ScannedProjects =
    {
        "TensorSharp.Chat",
        "TensorSharp.Server",
        "TensorSharp.Server.Host",
        "TensorSharp.Runtime",
        "TensorSharp.Models",
        "TensorSharp.AgentHost",
        "TensorSharp.Knowledge",
        "TensorSharp.Studio",
        "TensorSharp.Cli",
    };

    /// <summary>
    /// An identifier in a log argument, or the placeholder it fills, whose name contains one of
    /// these (ignoring case) is taken to carry content. Deliberately broad: a count named
    /// <c>promptTokens</c> trips it too, and is then listed in <see cref="AllowList"/>.
    /// </summary>
    private static readonly string[] ContentNameFragments =
    {
        "prompt", "userInput", "lastUser", "assistantText", "preview",
        "fileName", "savedPath", "uploadUrl", "resultUrl",
        // A stored Responses API response's id: GET /v1/responses/{id} returns its output.
        "responseId",
    };

    /// <summary>
    /// An argument the scan flags but that is verified not to be user content. Matched on the
    /// argument's expression with all whitespace removed, so reformatting does not break an
    /// entry, and optionally narrowed to one file and to calls whose template contains
    /// <see cref="TemplateContains"/>. Every entry must still match a call site
    /// (<see cref="EveryAllowListEntryStillMatchesACallSite"/>), so the list cannot rot.
    /// </summary>
    private sealed record AllowEntry(string? File, string Argument, string? TemplateContains, string Reason);

    private static readonly AllowEntry[] AllowList =
    {
        // ---- Counts, lengths and token limits: numbers, never text. Any file. ----
        new(null, "window.OriginalPromptTokens", null, "token count"),
        new(null, "mediaWindow.OriginalPromptTokens", null, "token count"),
        new(null, "contextLimit - window.PromptLimit", null, "token arithmetic"),
        new(null, "contextLimit - mediaWindow.PromptLimit", null, "token arithmetic"),
        new(null, "promptTokens.Count", null, "token count"),
        new(null, "promptTokenCount", null, "token count"),
        new(null, "choice.PromptTokens", null, "token count (ImageIntentChoice.PromptTokens is an int)"),
        new(null, "skillPlan.Prompt.Deferred.Count", null, "count of skills announced"),
        new(null, "skillPlan.Prompt.Inlined.Count", null, "count of skills inlined"),
        new(null, "skillPlan.Prompt.Catalog.Count", null, "count of catalog entries"),
        new(null, "skillPlan.Prompt.ApproximateTokens", null, "token estimate"),
        new(null, "skillPlan.ApproximateTokens", null, "token estimate"),
        new(null, "prompt?.Length ?? 0", null, "character count"),
        new(null, "prompt.Count", null, "token count (CliInferenceSession's prompt is a token list)"),
        new(null, "seq.PromptTokens.Count", null, "token count"),
        new(null, "seq.PromptTokens.Count - seq.PrefixCacheReusedTokens", null, "token arithmetic"),
        new(null, "promptLimit", null, "token limit"),
        new(null, "inputTokens.Count", null, "token count"),
        new(null, "promptTokensCached[i]", null, "token count per benchmark turn (int[])"),

        // ---- Model, draft-model, projector and LoRA file names: operator configuration
        //      (the command line, TS_LORAS, the model directory), not a client's content. ----
        new(null, "Path.GetFileName(modelPath)", null, "model file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs", "Path.GetFileName(chatOverride.ModelPath)", null, "configured model file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs", "Path.GetFileName(mmProjPath ?? string.Empty)", null, "projector file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs", "Path.GetFileName(previousModelPath)", null, "model file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs", "Path.GetFileName(attachedDraft)", null, "draft model file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs", "Path.GetFileName(spec.Path)", null, "LoRA adapter file name"),
        new("TensorSharp.Chat/ModelLifecycleService.cs",
            "TensorSharp.Models.LoraHostPreflight.EnvNoPluginsWarning(LoraFileNames(loras), LoadedModelName, architecture)",
            null, "LoRA adapter file names from TS_LORAS"),
        new("TensorSharp.Chat/ModelLifecycleService.cs",
            "TensorSharp.Models.LoraHostPreflight.EnvForeignTypeWarning(LoraFileNames(group.Select(s => s.Spec)), group.Key, LoadedModelName, architecture, host.LoraPluginType)",
            null, "LoRA adapter file names from TS_LORAS"),
        new("TensorSharp.Chat/WebUiChatService.cs",
            "loras.Count == 0 ? \"(none)\" : string.Join(\", \", loras.Select(l => Path.GetFileName(l.Path) + \"@\" + l.Scale))",
            null, "LoRA adapter file names from the server's LoRA directory"),
        new("TensorSharp.Server.Host/Program.cs",
            "string.IsNullOrEmpty(blockDraft) ? \"(none)\" : Path.GetFileName(blockDraft)",
            null, "draft model file name"),
        new("TensorSharp.Cli/Program.cs", "fileName", "chat template scan",
            "GGUF model file names the --test template scan iterates"),

        // ---- The Responses API's response id: a bearer key to the stored response. ----
        new("TensorSharp.Server/ProtocolAdapters/OpenAIResponsesAdapter.cs", "store ? LogContent.Of(requestId) : requestId", null,
            "marked as content whenever the response is stored under the id (store defaults to true)"),
        new("TensorSharp.Server/ProtocolAdapters/OpenAIResponsesAdapter.cs", "requestId", "/v1/responses rejected",
            "a rejected request: nothing is stored under the id"),
        new("TensorSharp.Server/ProtocolAdapters/OpenAIResponsesAdapter.cs", "requestId", "/v1/responses text.format invalid",
            "a rejected request: nothing is stored under the id"),
        // ---- Fixed text the CLI itself supplies: no user input is involved. ----
        new("TensorSharp.Cli/Program.cs", "rawText", "using default prompt",
            "the fixed default prompt (\"What is 1+1?\") used when no --input is given"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(input)", "tokenizer test input=",
            "the tokenizer self-test's fixed inputs"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(decoded)", "tokenizer test input=",
            "the round trip of the tokenizer self-test's fixed inputs"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(rendered)", "chat template test rendered=",
            "the chat-template self-test's fixed conversation"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(expected)", "chat template test rendered=",
            "the chat-template self-test's expected literal"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(engineText)", "comparison test engine output",
            "model output for the comparison self-test's fixed prompt (\"What is 1+1?\")"),
        new("TensorSharp.Cli/Program.cs", "LoggingExtensions.SanitizeForLog(ollamaResponse)", "comparison test ollama output",
            "Ollama's output for the comparison self-test's fixed prompt"),
    };

    /// <summary>
    /// The content-bearing log lines audited when the OpenTelemetry pipeline was added: each
    /// row is a file, a fragment of the line's (constant) template, and the placeholder that
    /// carries the content. Every call in that file whose template contains the fragment must
    /// still fill the placeholder with <c>LogContent.Of(...)</c> (or a local initialised from it).
    /// </summary>
    public static TheoryData<string, string, string> AuditedTemplates => new()
    {
        // InferenceTelemetry: the per-request start and finish lines.
        { "TensorSharp.Chat/InferenceTelemetry.cs", "chat.start arch=", "Uploads" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "chat.start arch=", "LastUserContent" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "chat.start arch=", "FullInput" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "chat.cancelled tokens=", "AssistantContent" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "chat.complete tokens=", "AssistantContent" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "generate.start arch=", "Uploads" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "generate.start arch=", "Prompt" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "generate.cancelled tokens=", "Completion" },
        { "TensorSharp.Chat/InferenceTelemetry.cs", "generate.complete tokens=", "Completion" },

        // The protocol adapters' request lines.
        { "TensorSharp.Server/ProtocolAdapters/OpenAIChatAdapter.cs", "/v1/chat/completions request:", "LastUser" },
        { "TensorSharp.Server/ProtocolAdapters/OpenAIResponsesAdapter.cs", "/v1/responses request:", "LastUser" },
        { "TensorSharp.Server/ProtocolAdapters/OllamaAdapter.cs", "/api/chat/ollama request:", "LastUser" },
        { "TensorSharp.Server/ProtocolAdapters/OllamaAdapter.cs", "/api/generate request:", "Prompt" },
        { "TensorSharp.Server/ProtocolAdapters/WebUiAdapter.cs", "OpenAI video generation done:", "Url" },

        // WebUiChatService: uploads, image and video prompts, and result URLs.
        { "TensorSharp.Chat/WebUiChatService.cs", "Upload received: name=", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Upload received: name=", "SavedFile" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Upload received: name=", "SavedPath" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Upload rejected: unsupported extension", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Upload rejected: {Reason}", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video frame extraction failed:", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video frame extraction failed:", "SavedPath" },
        { "TensorSharp.Chat/WebUiChatService.cs", "PDF text extraction failed:", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "PDF text extraction failed:", "SavedPath" },
        { "TensorSharp.Chat/WebUiChatService.cs", "PDF page-image extraction failed:", "FileName" },
        { "TensorSharp.Chat/WebUiChatService.cs", "PDF page-image extraction failed:", "SavedPath" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Image request: prompt=", "Prompt" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Image request done:", "Url" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Image edit (stream): prompt=", "Prompt" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Image edit (stream) done:", "Url" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video generate: prompt=", "Prompt" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video generate done:", "Url" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video generate (stream): prompt=", "Prompt" },
        { "TensorSharp.Chat/WebUiChatService.cs", "Video generate (stream) done:", "Url" },

        // Lines that quote the model's own (repeated) output or a path it chose.
        { "TensorSharp.Runtime/Scheduling/InferenceEngine.cs", "stopped after {Emitted} tokens: {What}", "What" },
        { "TensorSharp.Chat/Skills/SkillChatLoop.cs", "skills.loop.repetition round={Round} retried={Retried}:", "What" },
        { "TensorSharp.Chat/Skills/SkillChatLoop.cs", "skills.loop.repetition round={Round} retried=False (artifact correction)", "What" },
        { "TensorSharp.AgentHost/CodeExec/ShellRunner.cs", "codeexec.read path=", "Path" },
        { "TensorSharp.AgentHost/CodeExec/ShellRunner.cs", "codeexec.edited path=", "Path" },
        { "TensorSharp.AgentHost/CodeExec/ShellRunner.cs", "codeexec.rewrote path=", "Path" },

        // The CLI's input and output preview lines.
        { "TensorSharp.Cli/Program.cs", "Loaded input from {InputFile}", "InputFile" },
        { "TensorSharp.Cli/Program.cs", "Tokenized prompt:", "First20" },
        { "TensorSharp.Cli/Program.cs", "Tokenized prompt:", "Last10" },
        { "TensorSharp.Cli/Program.cs", "cli.inference.start ", "Input" },
        { "TensorSharp.Cli/Program.cs", "cli.inference.start ", "Uploads" },
        { "TensorSharp.Cli/Program.cs", "cli.inference.complete ", "Preview" },
        { "TensorSharp.Cli/Program.cs", "multi-turn turn=", "User" },
        { "TensorSharp.Cli/Program.cs", "multi-turn thinking", "ThinkingPreview" },
        { "TensorSharp.Cli/Program.cs", "multi-turn content chars=", "ContentPreview" },
        { "TensorSharp.Cli/Program.cs", "multi-turn output contains <unused> tokens", "RawPreview" },
        { "TensorSharp.Cli/Program.cs", "rendered prompt thinking=", "Preview" },
        { "TensorSharp.Cli/Program.cs", "jsonl batch [{RequestId}] tokens=", "OutputPreview" },
    };

    // ------------------------------------------------------------------------------------
    // The guard over the production sources.
    // ------------------------------------------------------------------------------------

    private sealed record RepositoryScan(string Root, IReadOnlyList<LogCall> Calls, IReadOnlyList<string> Problems, int Files);

    private static readonly Lazy<RepositoryScan> s_scan = new(ScanRepository, LazyThreadSafetyMode.ExecutionAndPublication);

    private static RepositoryScan Scan => s_scan.Value;

    [Fact]
    public void TheScannerReadsEveryScannedProjectAndItsLoggingCalls()
    {
        var scan = Scan;
        Assert.True(scan.Problems.Count == 0, "The scan could not read part of the sources:\n  " + string.Join("\n  ", scan.Problems));
        // Guard against a vacuous pass: a scanner that found nothing would make every other
        // test here green while checking nothing (there are 500+ calls today).
        Assert.True(scan.Files > 300, $"Only {scan.Files} source files were scanned under {scan.Root}.");
        Assert.True(scan.Calls.Count > 400, $"Only {scan.Calls.Count} logging calls were found under {scan.Root}.");
        // A call known to be there, read completely: its constant template and every argument.
        var chatStart = Assert.Single(scan.Calls, c =>
            c.File == "TensorSharp.Chat/InferenceTelemetry.cs" && c.Template?.StartsWith("chat.start arch=", StringComparison.Ordinal) == true);
        Assert.Equal("LogInformation", chatStart.Method);
        Assert.Contains("LastUserContent", chatStart.Placeholders!);
        Assert.Equal(chatStart.Placeholders!.Count, chatStart.Arguments.Count - chatStart.TemplateIndex - 1);
    }

    [Fact]
    public void EveryConstantTemplateHasOneArgumentPerPlaceholder()
    {
        // Positional binding is what maps a placeholder to its argument below, so a
        // mismatch means either a broken call (CA2017) or a call the scanner mis-split.
        var mismatches = Scan.Calls
            .Where(c => c.Placeholders != null && c.Placeholders.Count != c.Arguments.Count - c.TemplateIndex - 1)
            .Select(c => $"{c.File}:{c.Line}: {c.Method} has {c.Placeholders!.Count} placeholder(s) but {c.Arguments.Count - c.TemplateIndex - 1} argument(s) after the template \"{Shorten(c.Template!)}\"")
            .ToList();
        Assert.True(mismatches.Count == 0,
            "These logging calls bind placeholders to arguments by position and the counts differ. Either the call is wrong, "
            + "or ContentContractTests mis-read it (a comma inside generic type arguments, say):\n  "
            + string.Join("\n  ", mismatches));
    }

    [Fact]
    public void EveryContentBearingArgumentIsMarkedAsContent()
    {
        var violations = FindViolations(Scan.Calls, AllowList, used: null);
        Assert.True(violations.Count == 0,
            $"{violations.Count} logging argument(s) look like user content but are not wrapped in LogContent.Of(...), so the "
            + "OpenTelemetry exporter would send them verbatim. Wrap each one at the call site (local logs print exactly the same "
            + "text), or, if it is verified not to be content, add it to ContentContractTests.AllowList with the reason:\n  "
            + string.Join("\n  ", violations));
    }

    [Fact]
    public void EveryAllowListEntryStillMatchesACallSite()
    {
        var used = new HashSet<AllowEntry>();
        FindViolations(Scan.Calls, AllowList, used);
        var stale = AllowList.Where(e => !used.Contains(e))
            .Select(e => $"{e.File ?? "(any file)"}: {e.Argument}" + (e.TemplateContains != null ? $" in a template containing \"{e.TemplateContains}\"" : string.Empty))
            .ToList();
        Assert.True(stale.Count == 0,
            "These ContentContractTests.AllowList entries no longer match any flagged logging argument; remove them, or "
            + "update them if the call was only reworded:\n  " + string.Join("\n  ", stale));
    }

    [Theory]
    [MemberData(nameof(AuditedTemplates))]
    public void AnAuditedContentTemplateStillMarksItsContent(string file, string templateFragment, string placeholder)
    {
        var calls = Scan.Calls
            .Where(c => c.File == file && c.Template != null && c.Template.Contains(templateFragment, StringComparison.Ordinal))
            .ToList();
        Assert.True(calls.Count > 0,
            $"{file}: no logging call has a constant template containing \"{templateFragment}\" any more. If the line was "
            + "reworded, update its row in ContentContractTests.AuditedTemplates; if it was removed, delete the row.");

        foreach (var call in calls)
        {
            int position = call.Placeholders!.ToList().IndexOf(placeholder);
            Assert.True(position >= 0,
                $"{call.File}:{call.Line}: the template \"{Shorten(call.Template!)}\" no longer has {{{placeholder}}}; "
                + "update its row in ContentContractTests.AuditedTemplates.");
            int index = call.TemplateIndex + 1 + position;
            Assert.True(index < call.Arguments.Count,
                $"{call.File}:{call.Line}: no argument fills {{{placeholder}}} in \"{Shorten(call.Template!)}\".");
            var argument = call.Arguments[index];
            Assert.True(IsMarkedAsContent(call.Source, argument),
                $"{call.File}:{argument.Line}: {{{placeholder}}} in \"{Shorten(call.Template!)}\" is filled by `{argument.Display}`, "
                + "which is not LogContent.Of(...): the OpenTelemetry exporter would send it verbatim.");
        }
    }

    // ------------------------------------------------------------------------------------
    // The scanner's own behaviour, on synthetic sources, so a parser regression shows up as
    // a precise failure here rather than as a confusing one above.
    // ------------------------------------------------------------------------------------

    private static string Lines(params string[] lines) => string.Join("\n", lines);

    [Fact]
    public void ParenthesesCommasAndQuotesInsideStringsCharsAndCommentsDoNotSplitArguments()
    {
        string source = Lines(
            "class C",
            "{",
            "    void M()",
            "    {",
            "        _logger.LogInformation(LogEventIds.ChatStarted,",
            "            \"a (tricky, template) {Count} with \\\"quotes\\\" and ')' {Prompt}\", // trailing, comment (",
            "            items.Count(x => x == ',' || x == '\\'' || x == '\"'),",
            "            /* inline, comment ) */ LogContent.Of(Sanitize(prompt, \")\")));",
            "        var c = '(';",
            "        logger?.LogWarning(\"next {Value}\", value);",
            "    }",
            "}");

        var calls = ScanSource("Synthetic/A.cs", source);

        Assert.Equal(2, calls.Count);
        var first = calls[0];
        Assert.Equal(5, first.Line);
        Assert.Equal("LogInformation", first.Method);
        Assert.Equal(4, first.Arguments.Count);
        Assert.Equal(1, first.TemplateIndex);
        Assert.Equal("a (tricky, template) {Count} with \"quotes\" and ')' {Prompt}", first.Template);
        Assert.Equal(new[] { "Count", "Prompt" }, first.Placeholders);
        Assert.Equal("items.Count(x => x == ',' || x == '\\'' || x == '\"')", first.Arguments[2].Display);
        Assert.Equal(7, first.Arguments[2].Line);
        Assert.Equal("LogContent.Of(Sanitize(prompt, \")\"))", first.Arguments[3].Display);
        Assert.True(IsMarkedAsContent(first.Source, first.Arguments[3]));
        Assert.Empty(FindViolations(calls, Array.Empty<AllowEntry>(), used: null));

        var second = calls[1];
        Assert.Equal(10, second.Line);
        Assert.Equal("LogWarning", second.Method);
        Assert.Equal(new[] { "Value" }, second.Placeholders);
    }

    [Fact]
    public void VerbatimRawAndConcatenatedTemplatesAreReadAsTheCompilerReadsThem()
    {
        string source = Lines(
            "_log.LogInformation(@\"verbatim \"\"{Quoted}\"\" (with, parens\" + \" more {A}\", quoted, a);",
            "_log.LogInformation(\"first {A} \" +",
            "    \"second {B} {{escaped}} \" +",
            "    \"third {C,8:F1}\", a, b, c);",
            "_log.LogDebug(\"\"\"raw \"{Raw}\" (x, y\"\"\", raw);",
            "_log.LogDebug(\"\", empty);");

        var calls = ScanSource("Synthetic/B.cs", source);

        Assert.Equal(4, calls.Count);
        Assert.Equal("verbatim \"{Quoted}\" (with, parens more {A}", calls[0].Template);
        Assert.Equal(new[] { "Quoted", "A" }, calls[0].Placeholders);
        Assert.Equal(3, calls[0].Arguments.Count);

        Assert.Equal(2, calls[1].Line);
        Assert.Equal("first {A} second {B} {{escaped}} third {C,8:F1}", calls[1].Template);
        Assert.Equal(new[] { "A", "B", "C" }, calls[1].Placeholders);
        Assert.Equal(4, calls[1].Arguments.Count);

        Assert.Equal("raw \"{Raw}\" (x, y", calls[2].Template);
        Assert.Equal(new[] { "Raw" }, calls[2].Placeholders);
        Assert.Equal(2, calls[2].Arguments.Count);

        Assert.Equal(string.Empty, calls[3].Template);
        Assert.Empty(calls[3].Placeholders!);
    }

    [Fact]
    public void InterpolationHolesAreCodeAndContentInterpolatedIntoATemplateIsAViolation()
    {
        string source = Lines(
            "_log.LogInformation($\"built {Describe(prompt, \"x,y\")} at run time {{literal}}\", count);",
            "_log.LogInformation($@\"{userInput.Length} chars\");");

        var calls = ScanSource("Synthetic/C.cs", source);

        Assert.Equal(2, calls.Count);
        Assert.Null(calls[0].Template);
        Assert.Null(calls[0].Placeholders);
        Assert.Equal(2, calls[0].Arguments.Count);
        var violations = FindViolations(calls, Array.Empty<AllowEntry>(), used: null);
        Assert.Equal(2, violations.Count);
        Assert.Contains("Synthetic/C.cs:1:", violations[0]);
        Assert.Contains("'prompt'", violations[0]);
        Assert.Contains("message template", violations[0]);
        Assert.Contains("'userInput'", violations[1]);
    }

    [Fact]
    public void UnwrappedContentIsReportedWithFileLineArgumentAndReason()
    {
        string source = Lines(
            "void M()",
            "{",
            "    _log.LogInformation(LogEventIds.ChatStarted, \"chat userInput={LastUser} n={N}\",",
            "        LoggingExtensions.SanitizeForLog(text), 3);",
            "    _log.LogWarning(ex, \"saved {Where}\", savedPath);",
            "    _log.LogInformation(\"A {Prompt}-token prompt\", rawText);",
            "    _log.LogInformation(\"decoded {Text}\", model.Tokenizer.Decode(ids));",
            "    _log.LogInformation(\"wrapped {Text}\", TensorSharp.Runtime.Logging.LogContent.Of(model.Tokenizer.Decode(ids)));",
            "    _log.LogInformation(\"half {Text}\", LogContent.Of(previewText) + suffix);",
            "}");

        var calls = ScanSource("Synthetic/D.cs", source);
        var violations = FindViolations(calls, Array.Empty<AllowEntry>(), used: null);

        Assert.Equal(5, violations.Count);
        Assert.StartsWith("Synthetic/D.cs:4: LogInformation argument 3 for {LastUser}: LoggingExtensions.SanitizeForLog(text)", violations[0]);
        Assert.Contains("calls SanitizeForLog", violations[0]);
        Assert.StartsWith("Synthetic/D.cs:5: LogWarning argument 3 for {Where}: savedPath", violations[1]);
        Assert.Contains("fills {Prompt}", violations[2]);
        Assert.Contains("decodes tokens", violations[3]);
        // Wrapping only part of the argument does not count: the concatenation is a string.
        Assert.StartsWith("Synthetic/D.cs:9:", violations[4]);

        // An allow-list entry silences exactly its argument, and is reported as used.
        var allow = new[] { new AllowEntry("Synthetic/D.cs", "rawText", "-token prompt", "test") };
        var used = new HashSet<AllowEntry>();
        Assert.Equal(4, FindViolations(calls, allow, used).Count);
        Assert.True(used.Contains(allow[0]));
        Assert.Equal(5, FindViolations(calls, new[] { allow[0] with { File = "Synthetic/Other.cs" } }, used: null).Count);
    }

    [Fact]
    public void ALocalInitialisedFromLogContentOfCountsAsMarked()
    {
        string source = Lines(
            "LogContent assistantContent = LogContent.Of(LoggingExtensions.SanitizeForLogFull(assistantText));",
            "var previewContent = LogContent.Of(preview);",
            "string promptPreview = text;",
            "_log.LogInformation(\"done {AssistantContent}\", assistantContent);",
            "_log.LogInformation(\"preview {Preview}\", previewContent);",
            "_log.LogInformation(\"preview {Preview}\", promptPreview);");

        var calls = ScanSource("Synthetic/E.cs", source);
        var violations = FindViolations(calls, Array.Empty<AllowEntry>(), used: null);

        Assert.True(IsMarkedAsContent(calls[0].Source, calls[0].Arguments[1]));
        Assert.True(IsMarkedAsContent(calls[1].Source, calls[1].Arguments[1]));
        var only = Assert.Single(violations);
        Assert.StartsWith("Synthetic/E.cs:6:", only);
    }

    [Fact]
    public void OnlyLoggerCallsAreRead()
    {
        string source = Lines(
            "float y = MathF.Log(x) + (float)Math.Log(2.0);",
            "Ops.Log(mel, mel);",
            "_logger.Log(isLowNoise ? LogLevel.Debug : LogLevel.Information, LogEventIds.HttpRequestStarted, \"HTTP {Path}\", logPath);",
            "_logger.Log(completionLevel, completionEvent, \"done {Path}\", logPath);",
            "factory.CreateLogger(\"x\")",
            "    ?.LogWarning(\"multi-line receiver {Value}\", value);",
            "// _logger.LogWarning(\"commented out {Prompt}\", prompt);",
            "string s = \"_logger.LogWarning(\\\"in a string {Prompt}\\\", prompt)\";",
            "public void LogInformation(string message) { }");

        var calls = ScanSource("Synthetic/F.cs", source);

        Assert.Equal(new[] { 3, 4, 6 }, calls.Select(c => c.Line));
        Assert.Equal(2, calls[0].TemplateIndex);
        Assert.Equal(2, calls[1].TemplateIndex);
        Assert.Equal(0, calls[2].TemplateIndex);
    }

    [Fact]
    public void ATemplateBuiltAtRunTimeIsFoundBehindTheLevelEventAndException()
    {
        string source = Lines(
            "logger.LogWarning(LogEventIds.UploadRejected, rejectionTemplate, quotaError);",
            "_logger.LogWarning(ex, retainFusedCache ? \"retention failed {Id}\" : \"bookkeeping failed {Id}\", requestId);");

        var calls = ScanSource("Synthetic/G.cs", source);

        Assert.Equal(1, calls[0].TemplateIndex);
        Assert.Null(calls[0].Template);
        Assert.Equal(1, calls[1].TemplateIndex);
        Assert.Null(calls[1].Template);
    }

    // ------------------------------------------------------------------------------------
    // Contract checks shared by the guard and the scanner tests.
    // ------------------------------------------------------------------------------------

    private static readonly Regex Identifier = new(@"[A-Za-z_][A-Za-z0-9_]*", RegexOptions.CultureInvariant);

    /// <summary>A tokenizer decode (<c>model.Tokenizer.Decode(ids)</c>, <c>_tokenizer?.DecodeTokens(...)</c>): token ids back to text.</summary>
    private static readonly Regex TokenizerDecode = new(@"\.\s*Decode\w*\s*\(", RegexOptions.CultureInvariant);

    /// <summary><c>LogContent.Of(</c>, optionally namespace-qualified, at the start of the argument.</summary>
    private static readonly Regex LogContentOf = new(@"^(?:[A-Za-z_][A-Za-z0-9_]*\s*\.\s*)*LogContent\s*\.\s*Of\s*\(", RegexOptions.CultureInvariant);

    private static readonly Regex BareIdentifier = new(@"^[A-Za-z_][A-Za-z0-9_]*$", RegexOptions.CultureInvariant);

    private static readonly Regex Whitespace = new(@"\s+", RegexOptions.CultureInvariant);

    /// <summary>
    /// True when the whole argument is <c>LogContent.Of(...)</c>, or is a local the same file
    /// declares as <c>LogContent name = LogContent.Of(...)</c> (or <c>var</c>).
    /// </summary>
    private static bool IsMarkedAsContent(SourceFile file, LogArgument argument)
    {
        string code = argument.Code;
        Match wrap = LogContentOf.Match(code);
        if (wrap.Success && MatchingClose(code, wrap.Index + wrap.Length - 1) == code.Length - 1)
            return true;
        return BareIdentifier.IsMatch(code)
            && Regex.IsMatch(file.Code, @"(?:\bLogContent\??|\bvar)\s+" + Regex.Escape(code) + @"\s*=\s*LogContent\s*\.\s*Of\s*\(", RegexOptions.CultureInvariant);
    }

    private static bool HasContentName(string name) =>
        ContentNameFragments.Any(f => name.Contains(f, StringComparison.OrdinalIgnoreCase));

    /// <summary>Why an argument looks like content; empty when it does not.</summary>
    private static List<string> ContentReasons(string code, string? placeholder)
    {
        var reasons = new List<string>();
        foreach (Match id in Identifier.Matches(code))
        {
            if (id.Value is "SanitizeForLog" or "SanitizeForLogFull")
                reasons.Add("calls " + id.Value);
            else if (HasContentName(id.Value))
                reasons.Add($"names '{id.Value}'");
        }
        if (TokenizerDecode.IsMatch(code))
            reasons.Add("decodes tokens");
        if (placeholder != null && HasContentName(placeholder))
            reasons.Add($"fills {{{placeholder}}}");
        return reasons.Distinct(StringComparer.Ordinal).ToList();
    }

    private static string Key(string expression) => Whitespace.Replace(expression, string.Empty);

    private static AllowEntry? FindAllowed(IReadOnlyList<AllowEntry> allowList, LogCall call, LogArgument argument)
    {
        string key = Key(argument.Display);
        string template = call.Template ?? (call.TemplateIndex < call.Arguments.Count ? call.Arguments[call.TemplateIndex].Display : string.Empty);
        foreach (var entry in allowList)
        {
            if (entry.File != null && !string.Equals(entry.File, call.File, StringComparison.Ordinal))
                continue;
            if (entry.TemplateContains != null && !template.Contains(entry.TemplateContains, StringComparison.Ordinal))
                continue;
            if (string.Equals(Key(entry.Argument), key, StringComparison.Ordinal))
                return entry;
        }
        return null;
    }

    /// <summary>
    /// Every argument after the template that looks like content and is neither marked nor
    /// allow-listed, as "file:line: Method argument N for {Placeholder}: expression  [reasons]".
    /// A template built at run time that interpolates content is reported too: it cannot be
    /// wrapped, and the exporter only ever sees the finished message.
    /// </summary>
    private static List<string> FindViolations(IEnumerable<LogCall> calls, IReadOnlyList<AllowEntry> allowList, HashSet<AllowEntry>? used)
    {
        var violations = new List<string>();
        foreach (var call in calls)
        {
            if (call.Template == null && call.TemplateIndex < call.Arguments.Count)
            {
                var template = call.Arguments[call.TemplateIndex];
                var reasons = ContentReasons(template.Code, placeholder: null);
                if (reasons.Count > 0)
                {
                    var allowed = FindAllowed(allowList, call, template);
                    if (allowed != null)
                        used?.Add(allowed);
                    else
                        violations.Add($"{call.File}:{template.Line}: {call.Method} message template built from content: {template.Display}  "
                            + $"[{string.Join("; ", reasons)}; pass it as a LogContent.Of(...) argument instead]");
                }
            }

            for (int i = call.TemplateIndex + 1; i < call.Arguments.Count; i++)
            {
                var argument = call.Arguments[i];
                int position = i - call.TemplateIndex - 1;
                string? placeholder = call.Placeholders != null && position < call.Placeholders.Count ? call.Placeholders[position] : null;
                var reasons = ContentReasons(argument.Code, placeholder);
                if (reasons.Count == 0 || IsMarkedAsContent(call.Source, argument))
                    continue;
                var allowed = FindAllowed(allowList, call, argument);
                if (allowed != null)
                {
                    used?.Add(allowed);
                    continue;
                }
                string slot = placeholder != null ? $" for {{{placeholder}}}" : string.Empty;
                violations.Add($"{call.File}:{argument.Line}: {call.Method} argument {i + 1}{slot}: {argument.Display}  [{string.Join("; ", reasons)}]");
            }
        }
        return violations;
    }

    private static string Shorten(string text) => text.Length <= 120 ? text : text[..117] + "...";

    // ------------------------------------------------------------------------------------
    // Reading the sources.
    // ------------------------------------------------------------------------------------

    private static string FindRepositoryRoot()
    {
        string? start = Path.GetDirectoryName(typeof(ContentContractTests).Assembly.Location);
        if (string.IsNullOrEmpty(start))
            start = AppContext.BaseDirectory;
        for (var dir = new DirectoryInfo(start); dir != null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
                return dir.FullName;
        }
        throw new DirectoryNotFoundException(
            $"No directory containing TensorSharp.slnx above {start}: the content-contract guard reads the production sources.");
    }

    private static RepositoryScan ScanRepository()
    {
        string root = FindRepositoryRoot();
        var calls = new List<LogCall>();
        var problems = new List<string>();
        int files = 0;
        foreach (string project in ScannedProjects)
        {
            string projectDir = Path.Combine(root, project);
            if (!Directory.Exists(projectDir))
            {
                problems.Add($"{project}: the project directory is missing under {root}; update ContentContractTests.ScannedProjects.");
                continue;
            }
            var sources = Directory.EnumerateFiles(projectDir, "*.cs", SearchOption.AllDirectories)
                .Where(p => !IsBuildOutput(Path.GetRelativePath(projectDir, p)))
                .OrderBy(p => p, StringComparer.Ordinal);
            foreach (string path in sources)
            {
                files++;
                string relative = Path.GetRelativePath(root, path).Replace('\\', '/');
                calls.AddRange(ScanSource(relative, File.ReadAllText(path), problems));
            }
        }
        return new RepositoryScan(root, calls, problems, files);
    }

    private static bool IsBuildOutput(string relativePath)
    {
        foreach (string segment in relativePath.Split('/', '\\'))
        {
            if (segment.Equals("bin", StringComparison.OrdinalIgnoreCase) || segment.Equals("obj", StringComparison.OrdinalIgnoreCase))
                return true;
        }
        return false;
    }

    private static List<LogCall> ScanSource(string relativePath, string text)
    {
        var problems = new List<string>();
        var calls = ScanSource(relativePath, text, problems);
        Assert.True(problems.Count == 0, string.Join("\n", problems));
        return calls;
    }

    private static readonly Regex LoggerCall = new(
        @"\.\s*(Log|LogTrace|LogDebug|LogInformation|LogWarning|LogError|LogCritical)\s*\(",
        RegexOptions.CultureInvariant);

    /// <summary>Leading arguments that come before a template: a level, an event id, an exception.</summary>
    private static readonly Regex BeforeTemplate = new(
        @"^(?:LogLevel\s*\.|LogEventIds\s*\.|new\s+EventId\b)|Level$|^(?:ex|e|exception|\w+Ex|\w+Exception)$",
        RegexOptions.CultureInvariant);

    private static readonly Regex StringLiteralStart = new(@"^(?:\$+@?|@\$*)?""", RegexOptions.CultureInvariant);

    private static readonly Regex LevelArgument = new(@"LogLevel|Level$", RegexOptions.CultureInvariant);

    private static List<LogCall> ScanSource(string relativePath, string text, List<string> problems)
    {
        var file = Lexer.Read(relativePath, text);
        var calls = new List<LogCall>();
        foreach (Match match in LoggerCall.Matches(file.Code))
        {
            string method = match.Groups[1].Value;
            int open = match.Index + match.Length - 1;
            int close = MatchingClose(file.Code, open);
            if (close < 0)
            {
                problems.Add($"{relativePath}:{file.LineOf(match.Index)}: the {method} call has no closing parenthesis the scanner can find.");
                continue;
            }

            var arguments = SplitArguments(file, open, close);
            // .Log( is also Math.Log, MathF.Log and tensor ops: only a logger's Log counts,
            // which takes a LogLevel first or is called on something named like a logger.
            if (method == "Log" && !ReceiverName(file.Code, match.Index).Contains("log", StringComparison.OrdinalIgnoreCase)
                && !(arguments.Count > 0 && LevelArgument.IsMatch(arguments[0].Code)))
            {
                continue;
            }

            int templateIndex = TemplateIndex(arguments);
            string? template = templateIndex < arguments.Count ? ConstantValue(file, arguments[templateIndex]) : null;
            calls.Add(new LogCall(file, relativePath, file.LineOf(match.Index), method, arguments, templateIndex,
                template, template != null ? Placeholders(template) : null));
        }
        return calls;
    }

    /// <summary>
    /// The template is the first argument that is a string literal; failing that, the first
    /// with a literal in it (a conditional between two templates); failing that, the first
    /// argument after any level, event id and exception (a template held in a variable).
    /// </summary>
    private static int TemplateIndex(IReadOnlyList<LogArgument> arguments)
    {
        for (int i = 0; i < arguments.Count; i++)
        {
            if (StringLiteralStart.IsMatch(arguments[i].Code))
                return i;
        }
        for (int i = 0; i < arguments.Count; i++)
        {
            if (arguments[i].Code.Contains('"'))
                return i;
        }
        int k = 0;
        while (k < arguments.Count && BeforeTemplate.IsMatch(arguments[k].Code))
            k++;
        return k;
    }

    /// <summary>The identifier the member access at <paramref name="dot"/> is made on (<c>CreateLogger</c> for <c>CreateLogger("x").Log(</c>).</summary>
    private static string ReceiverName(string code, int dot)
    {
        int k = dot - 1;
        while (k >= 0 && (char.IsWhiteSpace(code[k]) || code[k] == '?'))
            k--;
        if (k >= 0 && (code[k] == ')' || code[k] == ']'))
        {
            int depth = 0;
            for (; k >= 0; k--)
            {
                if (code[k] == ')' || code[k] == ']')
                    depth++;
                else if ((code[k] == '(' || code[k] == '[') && --depth == 0)
                {
                    k--;
                    break;
                }
            }
            while (k >= 0 && char.IsWhiteSpace(code[k]))
                k--;
        }
        int end = k + 1;
        while (k >= 0 && IsIdentifierChar(code[k]))
            k--;
        return code.Substring(k + 1, end - k - 1);
    }

    /// <summary>The index of the bracket closing the one at <paramref name="open"/>, or -1.</summary>
    private static int MatchingClose(string code, int open)
    {
        int depth = 0;
        for (int i = open; i < code.Length; i++)
        {
            char c = code[i];
            if (c is '(' or '[' or '{')
            {
                depth++;
            }
            else if (c is ')' or ']' or '}')
            {
                depth--;
                if (depth == 0)
                    return i;
            }
        }
        return -1;
    }

    private static List<LogArgument> SplitArguments(SourceFile file, int open, int close)
    {
        string code = file.Code;
        var arguments = new List<LogArgument>();
        int depth = 0, start = open + 1;
        for (int i = open + 1; i < close; i++)
        {
            char c = code[i];
            if (c is '(' or '[' or '{')
                depth++;
            else if (c is ')' or ']' or '}')
                depth--;
            else if (c == ',' && depth == 0)
            {
                arguments.Add(file.Argument(start, i));
                start = i + 1;
            }
        }
        if (arguments.Count > 0 || code.AsSpan(start, close - start).Trim().Length > 0)
            arguments.Add(file.Argument(start, close));
        return arguments;
    }

    /// <summary>
    /// The value of an argument made only of plain (not interpolated) string literals joined
    /// with <c>+</c>; null for anything computed at run time.
    /// </summary>
    private static string? ConstantValue(SourceFile file, LogArgument argument)
    {
        var literals = file.Literals
            .Where(l => l.Start >= argument.Start && l.End <= argument.End)
            .OrderBy(l => l.Start)
            .ToList();
        if (literals.Count == 0)
            return null;
        var value = new StringBuilder();
        int position = argument.Start;
        foreach (var literal in literals)
        {
            if (literal.Start < position)
                continue; // nested in an interpolation hole of the previous literal
            if (!OnlyConcatenation(file.Code, position, literal.Start))
                return null;
            string? part = LiteralValue(file.Text, literal);
            if (part == null)
                return null;
            value.Append(part);
            position = literal.End;
        }
        return OnlyConcatenation(file.Code, position, argument.End) ? value.ToString() : null;
    }

    private static bool OnlyConcatenation(string code, int from, int to)
    {
        for (int i = from; i < to; i++)
        {
            if (!char.IsWhiteSpace(code[i]) && code[i] != '+')
                return false;
        }
        return true;
    }

    private static string? LiteralValue(string text, StringLiteral literal)
    {
        if (literal.Dollars > 0)
            return null;
        if (literal.Quotes >= 3 && !literal.Verbatim)
        {
            int from = literal.QuoteIndex + literal.Quotes, to = literal.End - literal.Quotes;
            return to > from ? text[from..to] : string.Empty;
        }
        if (literal.Quotes == 2 && !literal.Verbatim)
            return string.Empty;
        int bodyStart = literal.QuoteIndex + 1, bodyEnd = literal.End - 1;
        if (bodyEnd <= bodyStart)
            return string.Empty;
        string body = text[bodyStart..bodyEnd];
        if (literal.Verbatim)
            return body.Replace("\"\"", "\"", StringComparison.Ordinal);
        var sb = new StringBuilder(body.Length);
        for (int i = 0; i < body.Length; i++)
        {
            if (body[i] == '\\' && i + 1 < body.Length)
            {
                char next = body[++i];
                sb.Append(next switch { 'n' => '\n', 't' => '\t', 'r' => '\r', '0' => '\0', _ => next });
            }
            else
            {
                sb.Append(body[i]);
            }
        }
        return sb.ToString();
    }

    /// <summary>
    /// The placeholder names of a message template, in order, the way
    /// Microsoft.Extensions.Logging reads them: <c>{{</c> and <c>}}</c> are literal braces, and
    /// a name ends at <c>,</c> (alignment) or <c>:</c> (format).
    /// </summary>
    private static IReadOnlyList<string> Placeholders(string template)
    {
        var names = new List<string>();
        int i = 0;
        while (i < template.Length)
        {
            char c = template[i];
            if (c == '{')
            {
                if (i + 1 < template.Length && template[i + 1] == '{')
                {
                    i += 2;
                    continue;
                }
                int close = template.IndexOf('}', i + 1);
                if (close < 0)
                    break;
                string inner = template.Substring(i + 1, close - i - 1);
                int cut = inner.IndexOfAny(new[] { ',', ':' });
                names.Add((cut >= 0 ? inner[..cut] : inner).Trim());
                i = close + 1;
                continue;
            }
            if (c == '}' && i + 1 < template.Length && template[i + 1] == '}')
            {
                i += 2;
                continue;
            }
            i++;
        }
        return names;
    }

    private static bool IsIdentifierChar(char c) => char.IsLetterOrDigit(c) || c == '_';

    /// <summary>One logging call: where it is, its arguments, and its template if constant.</summary>
    private sealed record LogCall(
        SourceFile Source,
        string File,
        int Line,
        string Method,
        IReadOnlyList<LogArgument> Arguments,
        int TemplateIndex,
        string? Template,
        IReadOnlyList<string>? Placeholders);

    /// <summary>
    /// One argument, trimmed. <see cref="Code"/> has comments and string contents blanked
    /// (what the checks read); <see cref="Display"/> is the expression with comments removed
    /// and whitespace collapsed (what failure messages and the allow-list show).
    /// </summary>
    private sealed record LogArgument(int Start, int End, int Line, string Code, string Display);

    /// <summary>A string literal: [Start, End), its $ count, @, and where its opening quotes are.</summary>
    private sealed record StringLiteral(int Start, int End, int Dollars, bool Verbatim, int QuoteIndex, int Quotes);

    private sealed class SourceFile
    {
        private readonly int[] _lineStarts;

        public SourceFile(string path, string text, string code, string noComments, IReadOnlyList<StringLiteral> literals)
        {
            Path = path;
            Text = text;
            Code = code;
            NoComments = noComments;
            Literals = literals;
            var starts = new List<int> { 0 };
            for (int i = 0; i < text.Length; i++)
            {
                if (text[i] == '\n')
                    starts.Add(i + 1);
            }
            _lineStarts = starts.ToArray();
        }

        public string Path { get; }

        /// <summary>The file as written.</summary>
        public string Text { get; }

        /// <summary>Same length as <see cref="Text"/>; comments and string-literal contents are spaces (line breaks kept).</summary>
        public string Code { get; }

        /// <summary>Same length as <see cref="Text"/>; only comments are spaces.</summary>
        public string NoComments { get; }

        public IReadOnlyList<StringLiteral> Literals { get; }

        public int LineOf(int offset)
        {
            int index = Array.BinarySearch(_lineStarts, offset);
            return (index >= 0 ? index : ~index - 1) + 1;
        }

        public LogArgument Argument(int start, int end)
        {
            while (start < end && char.IsWhiteSpace(Code[start]))
                start++;
            while (end > start && char.IsWhiteSpace(Code[end - 1]))
                end--;
            string display = Whitespace.Replace(NoComments[start..end], " ").Trim();
            return new LogArgument(start, end, LineOf(start), Code[start..end], display);
        }
    }

    /// <summary>
    /// A C# lexer just deep enough to tell code from comments and string literals: line and
    /// block comments; char literals; regular, verbatim, raw (<c>"""</c>) and interpolated
    /// (<c>$</c>, <c>$@</c>, <c>@$</c>, <c>$$"""</c>) strings, with interpolation holes read
    /// as code (so nested strings and braces inside them are handled).
    /// </summary>
    private sealed class Lexer
    {
        private readonly string _s;
        private readonly char[] _code;
        private readonly char[] _noComments;
        private readonly List<StringLiteral> _literals = new();

        private Lexer(string s)
        {
            _s = s;
            _code = s.ToCharArray();
            _noComments = s.ToCharArray();
        }

        public static SourceFile Read(string path, string text)
        {
            var lexer = new Lexer(text);
            lexer.ReadCode(0, inHole: false);
            return new SourceFile(path, text, new string(lexer._code), new string(lexer._noComments), lexer._literals);
        }

        private void BlankComment(int i)
        {
            if (_s[i] != '\r' && _s[i] != '\n')
            {
                _code[i] = ' ';
                _noComments[i] = ' ';
            }
        }

        private void BlankLiteral(int i)
        {
            if (_s[i] != '\r' && _s[i] != '\n')
                _code[i] = ' ';
        }

        /// <summary>Reads code from <paramref name="i"/>; inside a hole, stops at the unmatched '}' and returns its index.</summary>
        private int ReadCode(int i, bool inHole)
        {
            int n = _s.Length, depth = 0;
            while (i < n)
            {
                char c = _s[i];
                if (c == '/' && i + 1 < n && _s[i + 1] == '/')
                {
                    while (i < n && _s[i] != '\r' && _s[i] != '\n')
                        BlankComment(i++);
                    continue;
                }
                if (c == '/' && i + 1 < n && _s[i + 1] == '*')
                {
                    int end = _s.IndexOf("*/", i + 2, StringComparison.Ordinal);
                    end = end < 0 ? n : end + 2;
                    for (int k = i; k < end; k++)
                        BlankComment(k);
                    i = end;
                    continue;
                }
                if (c == '\'')
                {
                    int j = i + 1;
                    if (j < n && _s[j] == '\\')
                        j += 2;
                    while (j < n && _s[j] != '\'' && _s[j] != '\r' && _s[j] != '\n')
                        j++;
                    for (int k = i + 1; k < j && k < n; k++)
                        BlankLiteral(k);
                    i = j + 1;
                    continue;
                }
                if ((c == '$' || c == '@' || c == '"') && TryStringStart(i, out int dollars, out bool verbatim, out int quoteIndex, out int quotes))
                {
                    int end = ReadString(dollars, verbatim, quoteIndex, quotes);
                    _literals.Add(new StringLiteral(i, end, dollars, verbatim, quoteIndex, quotes));
                    i = end;
                    continue;
                }
                if (inHole)
                {
                    if (c == '{')
                        depth++;
                    else if (c == '}')
                    {
                        if (depth == 0)
                            return i;
                        depth--;
                    }
                }
                i++;
            }
            return i;
        }

        private bool TryStringStart(int i, out int dollars, out bool verbatim, out int quoteIndex, out int quotes)
        {
            int n = _s.Length, j = i;
            dollars = 0;
            verbatim = false;
            quoteIndex = -1;
            quotes = 0;
            while (j < n && _s[j] == '$') { dollars++; j++; }
            if (j < n && _s[j] == '@') { verbatim = true; j++; }
            while (j < n && _s[j] == '$') { dollars++; j++; }
            if (j >= n || _s[j] != '"')
                return false;
            if ((dollars > 0 || verbatim) && i > 0 && IsIdentifierChar(_s[i - 1]))
                return false;
            quoteIndex = j;
            for (int k = j; k < n && _s[k] == '"'; k++)
                quotes++;
            return true;
        }

        /// <summary>Reads a literal whose opening quotes start at <paramref name="quoteIndex"/>; returns the index just past it.</summary>
        private int ReadString(int dollars, bool verbatim, int quoteIndex, int quotes)
        {
            int n = _s.Length;
            if (quotes >= 3 && !verbatim)
            {
                // Raw literal: ends at the same number of quotes; with $s, a run of at least
                // that many braces opens a hole (the extra ones are text).
                int i = quoteIndex + quotes;
                while (i < n)
                {
                    if (_s[i] == '"')
                    {
                        int k = i;
                        while (k < n && _s[k] == '"')
                            k++;
                        if (k - i >= quotes)
                            return i + quotes;
                        for (int x = i; x < k; x++)
                            BlankLiteral(x);
                        i = k;
                        continue;
                    }
                    if (dollars > 0 && _s[i] == '{')
                    {
                        int k = i;
                        while (k < n && _s[k] == '{')
                            k++;
                        if (k - i >= dollars)
                        {
                            for (int x = i; x < k - dollars; x++)
                                BlankLiteral(x);
                            i = ReadCode(k, inHole: true);
                            for (int r = 0; r < dollars && i < n && _s[i] == '}'; r++)
                                i++;
                            continue;
                        }
                        for (int x = i; x < k; x++)
                            BlankLiteral(x);
                        i = k;
                        continue;
                    }
                    BlankLiteral(i);
                    i++;
                }
                return i;
            }

            if (quotes == 2 && !verbatim)
                return quoteIndex + 2; // ""

            int p = quoteIndex + 1;
            while (p < n)
            {
                char c = _s[p];
                if (verbatim)
                {
                    if (c == '"')
                    {
                        if (p + 1 < n && _s[p + 1] == '"')
                        {
                            BlankLiteral(p);
                            BlankLiteral(p + 1);
                            p += 2;
                            continue;
                        }
                        return p + 1;
                    }
                }
                else
                {
                    if (c == '\\' && p + 1 < n)
                    {
                        BlankLiteral(p);
                        BlankLiteral(p + 1);
                        p += 2;
                        continue;
                    }
                    if (c == '"')
                        return p + 1;
                    if (c == '\r' || c == '\n')
                        return p; // unterminated: the compiler would refuse it anyway
                }
                if (dollars > 0)
                {
                    if (c == '{')
                    {
                        if (p + 1 < n && _s[p + 1] == '{')
                        {
                            BlankLiteral(p);
                            BlankLiteral(p + 1);
                            p += 2;
                            continue;
                        }
                        p = ReadCode(p + 1, inHole: true) + 1;
                        continue;
                    }
                    if (c == '}' && p + 1 < n && _s[p + 1] == '}')
                    {
                        BlankLiteral(p);
                        BlankLiteral(p + 1);
                        p += 2;
                        continue;
                    }
                }
                BlankLiteral(p);
                p++;
            }
            return p;
        }
    }
}
