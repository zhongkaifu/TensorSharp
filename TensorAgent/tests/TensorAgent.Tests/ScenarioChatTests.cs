// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Settings;
using TensorSharp.AgentHost.Skills;

namespace TensorAgent.Tests;

/// <summary>
/// The work a person actually does with this app, done end to end.
///
/// <para>
/// <see cref="EndToEndChatTests"/> pins the CONTRACT — a turn streams, a session
/// remembers, the cache is reused and invalidated, an abort is honoured. None of
/// that says whether the app is any use: an app can answer every question, stream
/// every frame and reuse every token while being unable to read a long document,
/// run a program it just wrote, or produce a file. These are the scenarios that
/// answer that, and each of them fails for a different reason.
/// </para>
/// <para>
/// A four-bit model is not reliable, so nothing here asserts on prose where a fact
/// will do. What is asserted is what is actually deterministic: a frame the page
/// depends on appeared, a file exists where the shell would have put it, a number
/// the model reported matches one computed here in C#, a file begins with the bytes
/// its format requires. Where wording is unavoidable the model's own answer goes
/// into the failure message, because a live-model failure that does not quote the
/// model tells the reader nothing.
/// </para>
/// </summary>
[Collection(LiveModelHarness.Collection)]
public sealed class ScenarioChatTests : LiveModelHarness
{
    // =====================================================================================
    // 1. a short question
    // =====================================================================================

    /// <summary>
    /// Catches the app answering the wrong question, and the app answering the right
    /// one far too slowly to use.
    ///
    /// <para>
    /// The floor on wall-clock is deliberately loose — three minutes for a one-word
    /// answer — because it is not measuring the machine. It is there to catch the
    /// failures that make a phone unusable without making anything fail outright: a
    /// backend that quietly fell back to a scalar path, a prompt rebuilt from scratch
    /// per token, a preamble that grew until every trivial turn re-prefills thousands
    /// of tokens. It also pins <c>truncated</c>: half a kilotoken for one word and a
    /// truncated answer means the budget is not reaching the engine.
    /// </para>
    /// </summary>
    [LiveModelFact]
    public async Task AShortFactualQuestionComesBackRightAndDoesNotTakeMinutes()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();

        var clock = Stopwatch.StartNew();
        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(),
            messages = new[] { new { role = "user", content = "Which planet in our solar system is closest to the Sun? Answer with the planet's name." } },
            maxTokens = 512,
            think = false,
        });
        clock.Stop();

        string answer = TextOf(frames);
        TurnStats stats = StatsOf(frames);
        Console.WriteLine($"scenario short: {stats} on {LoadedBackend}");

        Assert.True(answer.Contains("Mercury", StringComparison.OrdinalIgnoreCase),
            "the model did not name the planet; it answered: " + answer);

        JsonElement done = frames.Last(f => f.TryGetProperty("done", out _));
        Assert.False(done.GetProperty("truncated").GetBoolean(),
            $"a one-word answer used the whole 512-token budget, so the budget is not reaching the engine; it answered: {answer}");

        Assert.True(clock.Elapsed < TimeSpan.FromMinutes(3),
            $"a one-line question took {clock.Elapsed.TotalSeconds:0.0}s on {LoadedBackend} "
            + $"({stats}), which is past the point where anyone would wait for it");
    }

    // =====================================================================================
    // 2. a long pasted document
    // =====================================================================================

    /// <summary>
    /// Catches a long paste being silently cut short.
    ///
    /// <para>
    /// Three distinct ways that happens, and none of them fails loudly: the transport
    /// truncates the request body, the prompt renderer drops the middle of an
    /// over-long message, or the context window is smaller than the app admits and the
    /// front of the conversation is evicted. All three leave a fluent answer behind.
    /// So the question is planted ONCE, nine tenths of the way down, where only a
    /// prompt that survived intact can answer it — and <c>promptTokens</c> is checked
    /// against the size of the text that was actually sent, since a prompt reported as
    /// a third of its real length is the same bug seen from the other end.
    /// </para>
    /// </summary>
    [LiveModelFact]
    public async Task ALongPastedDocumentIsReadToItsEndAndCountedHonestly()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights);
        await LoadAsync(model);

        const string partNumber = "QX-4417";
        string log = ShiftLog(
            lines: 140,
            factLine: 128,
            fact: $"Maintenance note: the replacement seal for the Ravensworth pump is part number {partNumber}, "
                + "ordered from the Tyneside depot and fitted the same afternoon.");

        JsonElement session = await OpenSessionAsync();
        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(),
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = "Here is a shift log.\n\n" + log
                        + "\n\nRead the whole log. What is the part number of the replacement seal for the "
                        + "Ravensworth pump? Answer with just the part number.",
                },
            },
            maxTokens = 512,
            think = false,
            // Greedy, for the same reason the picture scenario is: an assertion about
            // one exact string must not be a coin toss on top-p 0.95.
            temperature = 0.0,
        });

        string answer = TextOf(frames);
        TurnStats stats = StatsOf(frames);
        Console.WriteLine($"scenario long prompt: {log.Length} characters pasted, {stats}");
        Console.WriteLine($"scenario long prompt answer: {answer}");

        // Six characters per token is far below what English tokenizes at, so this
        // cannot fail on a tokenizer being efficient — only on text that never arrived.
        int floor = log.Length / 6;
        Assert.True(stats.PromptTokens >= floor,
            $"{log.Length} characters were pasted but the turn reports only {stats.PromptTokens} prompt tokens "
            + $"(at least {floor} were expected), so most of the document never reached the model");

        Assert.True(
            answer.Contains(partNumber, StringComparison.OrdinalIgnoreCase)
            || answer.Replace("-", string.Empty).Contains(partNumber.Replace("-", string.Empty), StringComparison.OrdinalIgnoreCase),
            $"the part number appears once, on line 128 of 140, and the model did not find it; it answered: {answer}");
    }

    // =====================================================================================
    // 3. code written, run, and then changed
    // =====================================================================================

    /// <summary>
    /// The reported Qwen failure, end to end: a time-sensitive request must use the
    /// network/code tool and return ten current rows. The assertion deliberately does
    /// not pin symbols, because a correct list changes throughout the trading day.
    /// </summary>
    [LiveNetworkCodeFact]
    public async Task QwenRetrievesTenCurrentStockGainersEndToEnd()
    {
        Assert.Null(Unavailable(out _, out _));
        CatalogModel model = ModelCatalog.BuiltIn.Single(candidate =>
            candidate.Family == CatalogFamily.Qwen35
            && string.Equals(candidate.Parameters, "9B", StringComparison.Ordinal));
        string modelDirectory = Environment.GetEnvironmentVariable(ModelDirVariable)!;
        string modelFile = Environment.GetEnvironmentVariable(ModelFileVariable)
            ?? model.Weights.FileName;
        string weights = Path.Combine(modelDirectory, modelFile);
        Assert.True(File.Exists(weights),
            $"the Qwen scenario needs {weights}; set {ModelFileVariable} to the local Qwen3.5 9B GGUF name");
        Start(model, weights, interpreter: true, maxTokens: 2048);

        AppSettings settings = Host.Settings.Load();
        bool defaultThink = settings.ThinkByDefault;
        Assert.False(defaultThink,
            "the stock-gainers scenario is intended to exercise a new chat with TensorAgent's default reasoning setting");
        settings.AllowNetwork = true;
        Host.Settings.Save(settings);
        Host.ApplySettings(settings);
        await LoadAsync(model);

        JsonElement loaded = await Client.GetFromJsonAsync<JsonElement>("/api/models");
        Assert.Equal(modelFile, loaded.GetProperty("loaded").GetString());
        Assert.Equal("qwen35", loaded.GetProperty("architecture").GetString());

        JsonElement session = await OpenSessionAsync();
        TimeSpan turnCeiling = TimeSpan.FromMinutes(3);
        using var turnDeadline = new CancellationTokenSource(turnCeiling);
        var clock = Stopwatch.StartNew();
        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(),
            messages = new[]
            {
                new { role = "user", content = "Retrieve 10 stocks with most gains today" },
            },
            maxTokens = 1536,
            // The page seeds a fresh conversation from ThinkByDefault and sends that
            // value explicitly. Keep the acceptance scenario on the path an unchanged
            // installation presents to its user rather than enabling a test-only mode.
            think = defaultThink,
        }, turnDeadline.Token);
        clock.Stop();

        string answer = TextOf(frames);
        List<Progress> progress = ProgressOf(frames);
        TurnStats stats = StatsOf(frames);
        Console.WriteLine($"scenario stock gainers: {stats} on {LoadedBackend}\n{answer}");
        foreach (Progress item in progress)
        {
            if (item.Detail.Length > 0)
                Console.WriteLine($"  {item.Phase}:{item.Tool} {item.Detail}");
        }

        // This was an eight-call exploration in the regression transcript. One finished
        // shell frame pins one execution; the streamed draft pins that execution to one
        // typed aggregate screener rather than ten serial lookups or a package install.
        string shellDraft = string.Concat(frames
            .Where(frame => Text(frame, "tool_progress") == "writing"
                && Text(frame, "tool") == SkillToolNames.Shell)
            .Select(frame => Text(frame, "text")));

        string shellOutput = string.Concat(frames
            .Where(frame => Text(frame, "tool_progress") == "running"
                && Text(frame, "tool") == SkillToolNames.Shell)
            .Select(frame => Text(frame, "text")));
        string shellDiagnostic = $"Frames: {Describe(progress)}\n"
            + $"Generated shell command(s):\n{shellDraft}\n"
            + $"Shell output:\n{shellOutput}\n"
            + $"Final answer:\n{answer}";

        Assert.True(progress.Any(item => item.Phase == "running" && item.Tool == SkillToolNames.Shell),
            "the model never ran its generated retrieval command. " + shellDiagnostic);
        Progress[] completedShells = progress.Where(item =>
                item.Phase == "finished" && item.Tool == SkillToolNames.Shell)
            .ToArray();
        Assert.True(completedShells.Length == 1,
            $"expected exactly one completed shell call, found {completedShells.Length}. {shellDiagnostic}");
        Progress completedShell = completedShells[0];

        // What the command must DO, not what it must say. These four assertions used to
        // pin a Yahoo Finance screener verbatim — `scrIds=day_gainers`, `quoteType ==
        // "EQUITY"`, `sorted(eligible` — and they passed for a reason that turned out to
        // be the bug: the host was pasting that exact program into the shell tool's
        // description on every turn, so the model was copying it back. With the recipe
        // gone the model picks its own source, and a test that demands one source is
        // testing the prompt rather than the model. What still has to be true is that it
        // FETCHED (rather than answering from memory), in one request, without installing
        // anything. Where the rows came from is checked below, against the output.
        Assert.DoesNotContain("pip install", shellDraft, StringComparison.OrdinalIgnoreCase);
        Assert.True(
            shellDraft.Contains("urlopen", StringComparison.Ordinal)
            || shellDraft.Contains("urlretrieve", StringComparison.Ordinal)
            || shellDraft.Contains("curl", StringComparison.Ordinal),
            "the command fetched nothing, so any rows in the answer were invented. " + shellDiagnostic);
        int requests = Regex.Matches(
            shellDraft,
            @"(?<![A-Za-z0-9_])(?:urllib\.request\.)?urlopen\s*\(",
            RegexOptions.CultureInvariant).Count;
        Assert.True(requests == 1,
            $"expected one aggregate network request, found {requests} urlopen calls in: {shellDraft}");

        const string PercentagePattern = @"(?<![A-Za-z0-9])\+?\d+(?:\.\d+)?%";
        string[] sourcedRows = shellOutput.Split('\n')
            .Where(line => Regex.IsMatch(line, PercentagePattern))
            .ToArray();
        Assert.True(sourcedRows.Length == 10
                && sourcedRows.All(line => TableCells(line).Length == 7),
            "the single aggregate command did not print exactly ten final-ready Markdown rows: " + shellOutput);
        string[][] sourcedCells = sourcedRows.Select(TableCells).ToArray();
        Assert.Equal(
            Enumerable.Range(1, 10),
            sourcedCells.Select(cells => ParseInteger(cells[0], "rank")));
        decimal[] dollarChanges = sourcedCells
            .Select(cells => ParseDecimal(cells[4], "dollar change"))
            .ToArray();
        decimal[] percentageChanges = sourcedCells
            .Select(cells => ParseDecimal(cells[5].TrimEnd('%'), "percentage change"))
            .ToArray();
        Assert.All(dollarChanges, change =>
            Assert.True(change > 0, $"a day-gainer row had a non-positive dollar change: {change}"));
        Assert.All(percentageChanges, change =>
            Assert.True(change > 0, $"a day-gainer row had a non-positive percentage change: {change}"));
        Assert.True(
            percentageChanges.Zip(percentageChanges.Skip(1), (left, right) => left >= right).All(value => value),
            "the aggregate command did not rank gainers by descending percentage: " + shellOutput);

        JsonElement done = frames.Last(frame => frame.TryGetProperty("done", out _));
        Assert.False(done.GetProperty("truncated").GetBoolean(),
            $"the model used the output budget before completing its final answer: {answer}");
        Assert.False(string.IsNullOrWhiteSpace(answer),
            $"the retrieval completed ({completedShell.Detail}) but the model wrote no final answer");
        Assert.DoesNotContain("model ended this turn without writing an answer", answer,
            StringComparison.OrdinalIgnoreCase);

        string[] percentageRows = answer.Split('\n')
            .Where(line => Regex.IsMatch(line, PercentagePattern))
            .ToArray();
        int percentages = percentageRows.Sum(line =>
            Regex.Matches(line, PercentagePattern).Count);
        Assert.True(percentageRows.Length == 10 && percentages == 10,
            $"expected exactly ten rows with one gain percentage each; found {percentageRows.Length} rows "
            + $"and {percentages} percentages: {answer}");
        Assert.True(percentageRows.All(line => TableCells(line).Length == 7),
            "the final answer did not present all sourced columns in one compact Markdown table: " + answer);

        // Ground every cell, not just the ticker. A plausible-looking row with a rounded
        // or invented price used to pass as long as its symbol appeared somewhere in the
        // tool output.
        Assert.Equal(sourcedRows.Select(NormalizeTableRow), percentageRows.Select(NormalizeTableRow));

        string[] tickers = percentageRows
            .Select(line => TableCells(line)[1])
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        Assert.True(tickers.Length == 10,
            $"expected ten distinct ticker rows, found {tickers.Length} ({string.Join(", ", tickers)}): {answer}");
        Assert.All(tickers, ticker => Assert.Contains(ticker, shellOutput, StringComparison.Ordinal));
        // No vendor is required: the model chooses the source now.
        // Everything the answer states about a row must be IN the command's output.
        // This replaces a byte-for-byte comparison against a thirteen-line table and a
        // heading regex that required the words "Yahoo Finance": both described the shape
        // of the one screener program the host used to paste into every prompt, so they
        // tested the prompt, not the model. What matters is unchanged and is what the
        // report was really about — the model must not state a figure it did not fetch.
        string[] answerRows = ContentLines(answer)
            .Where(line => line.TrimStart().StartsWith("|", StringComparison.Ordinal))
            .ToArray();
        Assert.True(answerRows.Length >= 10,
            $"expected at least ten table rows in the answer, found {answerRows.Length}. {shellDiagnostic}");

        foreach (string row in answerRows)
        {
            foreach (string cell in TableCells(row))
            {
                string value = cell.Trim();
                // Only the data cells are checked. Punctuation, alignment markers and
                // column headings are the model's own formatting and are its to choose.
                if (value.Length < 2 || !value.Any(char.IsAsciiLetterOrDigit))
                    continue;
                string bare = value.Trim('*', '`', '+', '$', '%', ' ');
                if (bare.Length < 2)
                    continue;
                Assert.True(
                    shellOutput.Contains(bare, StringComparison.OrdinalIgnoreCase)
                    || !bare.Any(char.IsAsciiDigit),
                    $"the answer states '{bare}', which is in no line the command printed. {shellDiagnostic}");
            }
        }

        // The rows are sourced, whatever source the model chose; a story about WHY a
        // price moved is not, and no response supports one.
        Assert.DoesNotMatch(
            new Regex(@"\b(?:catalyst|likely\s+(?:because|due)|probably\s+(?:because|due))\b",
                RegexOptions.IgnoreCase | RegexOptions.CultureInvariant),
            answer);
        Assert.DoesNotContain("**Note", answer, StringComparison.OrdinalIgnoreCase);

        // This is deliberately a loose acceptance ceiling, not a benchmark. The measured
        // default-mode Q8 Metal path completes in well under a minute; three minutes leaves
        // room for device and network variance while still rejecting the old
        // install/per-ticker loop, which spent many minutes retrying dependencies and
        // serial requests.
        Assert.True(clock.Elapsed < turnCeiling,
            $"the current-data turn took {clock.Elapsed.TotalSeconds:0.0}s on {LoadedBackend} ({stats}); "
            + "a single aggregate lookup should not take several minutes");

        static string[] TableCells(string row) => row
            .Split('|', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries);

        static string NormalizeTableRow(string row) =>
            string.Join("|", TableCells(row));

        static string[] ContentLines(string text) => text
            .Replace("\r\n", "\n", StringComparison.Ordinal)
            .Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

        static int ParseInteger(string text, string field)
        {
            Assert.True(int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value),
                $"the sourced {field} was not an integer: '{text}'");
            return value;
        }

        static decimal ParseDecimal(string text, string field)
        {
            Assert.True(decimal.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out decimal value),
                $"the sourced {field} was not numeric: '{text}'");
            return value;
        }
    }

    /// <summary>
    /// The reported whole workflow, using the user's prompt verbatim: discover the
    /// research and document skills, collect current information, compare the two
    /// chips, and leave behind a real downloadable PowerPoint deck.  Assertions stay
    /// on the observable work rather than the model's prose so a promise of a deck can
    /// never pass for one.
    /// </summary>
    [LiveNetworkCodeFact]
    public async Task AppleM6ResearchComparisonProducesARealPowerPointEndToEnd()
    {
        Assert.Null(Unavailable(out _, out _));
        CatalogModel model = ModelCatalog.BuiltIn.Single(candidate =>
            candidate.Family == CatalogFamily.Qwen35
            && string.Equals(candidate.Parameters, "9B", StringComparison.Ordinal));
        string modelDirectory = Environment.GetEnvironmentVariable(ModelDirVariable)!;
        string modelFile = Environment.GetEnvironmentVariable(ModelFileVariable)
            ?? model.Weights.FileName;
        string weights = Path.Combine(modelDirectory, modelFile);
        Assert.True(File.Exists(weights),
            $"the Apple M6 scenario needs {weights}; set {ModelFileVariable} to the local Qwen3.5 9B GGUF name");

        Start(model, weights, skills: true, interpreter: true, maxTokens: 2048);
        AppSettings settings = Host.Settings.Load();
        settings.AllowNetwork = true;
        Host.Settings.Save(settings);
        Host.ApplySettings(settings);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);

        var clock = Stopwatch.StartNew();
        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            messages = new[]
            {
                new { role = "user", content = "搜索apple M6的信息，并对比M5芯片，然后生成pptx报告" },
            },
            maxTokens = 2048,
            think = false,
            temperature = 0.0,
        });
        clock.Stop();

        string answer = TextOf(frames);
        List<SkillStep> steps = StepsOf(frames);
        List<Progress> progress = ProgressOf(frames);
        string toolDrafts = string.Concat(frames
            .Where(frame => Text(frame, "tool_progress") == "writing")
            .Select(frame => Text(frame, "text")));
        string toolOutput = string.Concat(frames
            .Where(frame => Text(frame, "tool_progress") == "running")
            .Select(frame => Text(frame, "text")));
        Console.WriteLine(
            $"scenario Apple M6 deck ({clock.Elapsed.TotalSeconds:0.0}s, {steps.Count} tool calls): "
            + $"{Describe(steps)} | {Describe(progress)}\n"
            + $"Generated calls:\n{toolDrafts}\nTool output:\n{toolOutput}\n{answer}");

        string diagnostic = $"Steps: {Describe(steps)}\nProgress: {Describe(progress)}\n"
            + $"Generated calls:\n{toolDrafts}\nTool output:\n{toolOutput}\n"
            + $"Workspace:\n  {Listing(workspace)}\nAnswer:\n{answer}";
        Assert.Contains(steps, step => step.Skill == "research" && step.Ok);
        Assert.Contains(steps, step => step.Skill == DocumentsSkillId && step.Ok);
        List<SkillStep> writerSteps = steps.Where(step =>
            step.Skill == DocumentsSkillId
            && string.Equals(step.Detail, "scripts/make_pptx.py", StringComparison.Ordinal)).ToList();
        Assert.InRange(writerSteps.Count, 1, 3);
        Assert.True(writerSteps[^1].Ok, "the final document-writer attempt must succeed");
        Assert.Equal(writerSteps[^1], steps[^1]);
        SkillStep specWrite = Assert.Single(steps, step => step.Tool == "write_file");
        if (writerSteps.Any(step => !step.Ok && step.Round > specWrite.Round))
        {
            Assert.Contains("<function=apply_patch>", toolDrafts, StringComparison.Ordinal);
        }
        Assert.Single(Regex.Matches(
            toolDrafts,
            "<function=write_file>",
            RegexOptions.CultureInvariant));
        Assert.DoesNotContain("<parameter=overwrite>", toolDrafts, StringComparison.Ordinal);

        string notes = Assert.Single(
            Directory.GetFiles(workspace, "notes.md", SearchOption.AllDirectories));
        string evidence = File.ReadAllText(notes);
        Assert.Contains("M5", evidence, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("M6", evidence, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("http", evidence, StringComparison.OrdinalIgnoreCase);
        Assert.Empty(Directory.GetFiles(workspace, "skill_*", SearchOption.AllDirectories));

        string deck = Assert.Single(
            Directory.GetFiles(workspace, "*.pptx", SearchOption.AllDirectories));
        Assert.True(StartsWith(deck, ZipMagic),
            $"the generated report is not a PowerPoint ZIP package. {diagnostic}");

        using (var archive = System.IO.Compression.ZipFile.OpenRead(deck))
        {
            List<System.IO.Compression.ZipArchiveEntry> slides = archive.Entries
                .Where(entry => entry.FullName.StartsWith("ppt/slides/slide", StringComparison.Ordinal)
                    && entry.FullName.EndsWith(".xml", StringComparison.Ordinal))
                .ToList();
            Assert.True(slides.Count >= 4,
                $"the report has only {slides.Count} slides; the routed workflow requires a concise four-slide deck");
            string slideText = string.Join("\n", slides.Select(entry =>
                {
                    using var reader = new StreamReader(entry.Open());
                    return reader.ReadToEnd();
                }));
            Assert.Contains("M5", slideText, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("M6", slideText, StringComparison.OrdinalIgnoreCase);
            Assert.Contains("http", slideText, StringComparison.OrdinalIgnoreCase);
        }

        string artifactUrl = Assert.Single(
            ArtifactUrlsOf(frames), url => url.EndsWith(".pptx", StringComparison.OrdinalIgnoreCase));
        using HttpResponseMessage download = await Client.GetAsync(artifactUrl);
        Assert.True(download.IsSuccessStatusCode,
            $"the generated deck exists but its download link returned {(int)download.StatusCode}. {diagnostic}");
    }

    /// <summary>
    /// Catches the whole code path being broken in any of the places it can break,
    /// and does it in one conversation rather than five.
    ///
    /// <para>
    /// The frames are asserted as well as the answer because they are what the user
    /// watches: without <c>writing</c> / <c>running</c> / <c>finished</c> the page is
    /// frozen for the length of a program being written and run, and nothing else in
    /// the suite would notice they had stopped arriving. The answer is asserted
    /// against a product no model computes in its head, so an answer that is right is
    /// evidence the program was executed rather than imagined. And the second turn
    /// re-reads the file from the workspace, because "it changed the file" is the one
    /// claim a model can make convincingly while having done nothing at all.
    /// </para>
    /// </summary>
    [LiveCodeFact]
    public async Task AProgramIsWrittenRunAndThenChangedInTheSameWorkspace()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, interpreter: true);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);

        long expectedSum = Readings.Sum(value => (long)value);
        long expectedProduct = Readings.Aggregate(1L, (running, value) => running * value);
        string list = string.Join(", ", Readings);

        var history = new List<object>
        {
            new
            {
                role = "user",
                content = $"Save a Python program to a file called readings.py in your working directory that "
                    + $"computes the SUM of these numbers: {list}. Do not use `python3 -c`; write the file, then "
                    + "run that file, and tell me the number it printed.",
            },
        };
        List<JsonElement> first = await StreamAsync(new { sessionId, messages = history, maxTokens = 900, think = false });
        string wrote = TextOf(first);
        List<Progress> progress = ProgressOf(first);
        Console.WriteLine($"scenario code, turn 1: {Describe(progress)}");

        Assert.True(progress.Any(p => p.Phase == "writing"),
            $"no `writing` tool-progress frame arrived, so the page would have shown nothing while the "
            + $"program was being typed. Frames: {Describe(progress)}; answer: {wrote}");
        Assert.True(progress.Any(p => p.Phase == "running"),
            $"no `running` tool-progress frame arrived, so the page would have shown nothing while the "
            + $"program ran. Frames: {Describe(progress)}; answer: {wrote}");
        Assert.True(progress.Any(p => p.Phase == "finished"),
            $"the `running` line was never taken down. Frames: {Describe(progress)}");

        Dictionary<string, string> written = PythonFilesIn(workspace);
        Assert.True(written.Count > 0,
            $"no Python file was written to the session workspace.\n  {Listing(workspace)}\n\nanswer: {wrote}");

        string source = string.Join("\n", written.Values);
        int quoted = Readings.Count(value => Mentions(source, value));
        Assert.True(quoted >= Readings.Length - 1,
            $"the program in the workspace names only {quoted} of the {Readings.Length} numbers it was given, "
            + $"so it is not the program that was asked for:\n{source}");

        Assert.True(States(wrote, expectedSum),
            $"the run should have printed {expectedSum}; the model answered: {wrote}");

        // The change. Same file, same session, so the workspace it edits is the one it
        // just wrote into.
        history.Add(new { role = "assistant", content = wrote });
        history.Add(new
        {
            role = "user",
            content = "Now change readings.py so that it computes the PRODUCT of the same numbers instead of the "
                + "sum. Run it again and tell me the new number.",
        });
        List<JsonElement> second = await StreamAsync(new { sessionId, messages = history, maxTokens = 900, think = false });
        string changed = TextOf(second);
        Console.WriteLine($"scenario code, turn 2: {Describe(ProgressOf(second))}");

        // THE file, not any file. "Some .py changed" passes when the model abandons
        // readings.py and writes product.py beside it, which is not an edit and not
        // what was asked; it also passes if a stray scratch file appears. The named
        // file has to still be there and its contents have to have moved.
        Dictionary<string, string> after = PythonFilesIn(workspace);
        KeyValuePair<string, string> target = written.First(f =>
            Path.GetFileName(f.Key).Equals("readings.py", StringComparison.OrdinalIgnoreCase));
        Assert.True(after.TryGetValue(target.Key, out string? nowSource),
            $"readings.py is gone from the workspace, so the second turn replaced it rather than editing it:\n  "
            + $"{Listing(workspace)}\n\nanswer: {changed}");
        Assert.True(nowSource != target.Value,
            $"readings.py is byte-for-byte what the first turn wrote, so the second turn edited nothing:\n"
            + $"{nowSource}\n\nanswer: {changed}");

        Assert.True(States(changed, expectedProduct),
            $"the edited program should have printed {expectedProduct}; the model answered: {changed}");
    }

    // =====================================================================================
    // 4. agent work over data
    // =====================================================================================

    /// <summary>
    /// Catches an agent that talks about a file instead of reading it.
    ///
    /// <para>
    /// Six revenue figures invented for this test, grouped by three region names
    /// invented with them: an answer matching the arithmetic done here in C# could
    /// only have come from the file. That is the assertion that matters. The frame
    /// check beside it tells the two failures apart — a model that answered without
    /// running anything, versus one that ran something which never found the CSV the
    /// test planted in its workspace — because those are fixed in different places.
    /// </para>
    /// </summary>
    [LiveCodeFact]
    public async Task AnAnswerAboutACsvInTheWorkspaceMatchesArithmeticDoneInCSharp()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, interpreter: true);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);
        File.WriteAllText(Path.Combine(workspace, SalesFileName), SalesCsv());

        long expectedTotal = Sales.Sum(row => (long)row.Revenue);
        IGrouping<string, (string Region, int Units, int Revenue)> best = Sales
            .GroupBy(row => row.Region, StringComparer.Ordinal)
            .OrderByDescending(group => group.Sum(row => (long)row.Revenue))
            .First();

        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = $"The file {SalesFileName} in your working directory has the columns region, units "
                        + "and revenue. Write and run a program that reads it and works out two things: the total "
                        + "of the revenue column across every row, and which region has the highest total revenue. "
                        + "Then tell me both answers, and make the last line of your reply exactly "
                        + "ANSWER: total=<number>, region=<name>",
                },
            },
            maxTokens = 900,
            think = false,
        });

        string answer = TextOf(frames);
        List<Progress> progress = ProgressOf(frames);
        Console.WriteLine($"scenario csv: {Describe(progress)}");

        Assert.True(
            progress.Any(p => p.Phase == "finished" && SkillToolNames.CodeTools.Contains(p.Tool, StringComparer.Ordinal)),
            $"no code tool finished, so nothing ever opened {SalesFileName}. Frames: {Describe(progress)}; "
            + $"answer: {answer}");

        Assert.True(States(answer, expectedTotal),
            $"the revenue column totals {expectedTotal}; the model answered: {answer}");

        // Parsed out of a line the request asked for, rather than searched for anywhere
        // in the prose. "Does the answer contain 'Bellhaven'" is satisfied by a reply
        // that tabulates all three regions, or by one that names it as the LOWEST --
        // both of which are wrong answers that the substring check called right. A
        // single verdict line is what makes the assertion about the model's conclusion
        // instead of about its vocabulary.
        Match verdict = Regex.Match(
            answer,
            @"ANSWER:\s*total\s*=\s*(?<total>[0-9][0-9,_ ]*(?:\.[0-9]+)?)\s*,\s*region\s*=\s*(?<region>[A-Za-z][A-Za-z '\-]*)",
            RegexOptions.IgnoreCase);
        Assert.True(verdict.Success,
            $"the reply has no 'ANSWER: total=..., region=...' line, so which region it settled on cannot be "
            + $"told apart from which regions it mentioned. The model answered: {answer}");

        decimal statedTotal = decimal.Parse(
            Regex.Replace(verdict.Groups["total"].Value, @"[,_ ]", string.Empty), CultureInfo.InvariantCulture);
        Assert.Equal((decimal)expectedTotal, statedTotal);

        Assert.Equal(best.Key, verdict.Groups["region"].Value.Trim(), StringComparer.OrdinalIgnoreCase);
    }

    // =====================================================================================
    // 5. finding a skill nobody named
    // =====================================================================================

    /// <summary>
    /// Catches skill discovery being off in practice while being on in configuration.
    ///
    /// <para>
    /// The request names no skill, so the only way <c>theme-factory</c> can be read
    /// is if the catalog was advertised, the model called <c>skills_read</c>, and the
    /// host answered it in process. Any break in that chain — a catalog that never
    /// reaches the prompt, a family whose tool declarations are dropped, a registry
    /// pointed at the wrong directory — leaves the model answering from memory, which
    /// reads perfectly well and is exactly the failure this exists to see. The
    /// <c>skill_step</c> frames are what the page draws its progress trace from, so
    /// asserting on them checks the user-visible half at the same time.
    /// </para>
    /// </summary>
    [LiveModelFact]
    public async Task ABundledSkillIsFoundAndReadWithoutTheRequestNamingIt()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true);
        await LoadAsync(model);

        Assert.True(Host.Skills.Skills.Count > 0,
            $"no skills were discovered under {RepoSkillsDirectory}, so this proves nothing");

        JsonElement session = await OpenSessionAsync();
        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId = session.GetProperty("sessionId").GetString(),
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = "I am designing a slide for an internal deck and it needs a consistent "
                        + "colour and font theme. Check the guidance available to you, then tell me which "
                        + "colours and fonts to use.",
                },
            },
            maxTokens = 700,
            think = false,
        });

        string answer = TextOf(frames);
        List<SkillStep> steps = StepsOf(frames);
        Console.WriteLine($"scenario skills: {Describe(steps)}");

        Assert.True(steps.Count > 0,
            $"no skill_step frame arrived, so no skill was consulted at all; the model answered: {answer}");

        Assert.True(steps.Any(step => step.Skill == "theme-factory" && step.Ok),
            "the request asked for a colour and font theme, which is what the bundled 'theme-factory' "
            + $"skill describes, and it was never read successfully. Steps: {Describe(steps)}; "
            + $"answer: {answer}");
    }

    [LiveModelFact]
    public async Task UnrelatedPromptsDoNotUseMarketData()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true);
        await LoadAsync(model);
        Assert.Contains(Host.Skills.Skills, skill => skill.Id == "market-data");

        foreach (string prompt in new[]
        {
            "明天天气怎么样？",
            "北京明天天气怎么样？",
            "What will the weather be like tomorrow?",
            "你好！",
        })
        {
            // Fresh conversations, automatic discovery, and the app's default of
            // thinking off. The reported Qwen failure called a nonexistent
            // market-data/scripts/weather.py before answering the Chinese prompt.
            JsonElement session = await OpenSessionAsync();
            List<JsonElement> frames = await StreamAsync(new
            {
                sessionId = session.GetProperty("sessionId").GetString(),
                messages = new[] { new { role = "user", content = prompt } },
                maxTokens = 512,
                think = false,
                temperature = 0,
                seed = 42,
            });

            string answer = TextOf(frames);
            List<SkillStep> steps = StepsOf(frames);
            // Failed reads/runs have no skill id in skill_step. Inspect the streamed
            // arguments too, including a hallucinated direct market-data tool name.
            string attemptedCalls = string.Concat(frames
                .Where(frame => Text(frame, "tool_progress") == "writing")
                .Select(frame => Text(frame, "text")));
            Assert.True(steps.All(step => step.Skill != "market-data" && step.Tool != "market-data")
                && !attemptedCalls.Contains("market-data", StringComparison.OrdinalIgnoreCase)
                && ProgressOf(frames).All(step => step.Tool != "market-data"),
                $"Unrelated prompt '{prompt}' attempted market-data. Calls: {attemptedCalls}; "
                + $"steps: {Describe(steps)}; answer: {answer}");
            Assert.False(string.IsNullOrWhiteSpace(answer),
                $"Prompt '{prompt}' produced no answer. Steps: {Describe(steps)}");
        }

        // Positive control: narrowing applicability must still allow a real quote
        // request to discover and read the skill. Network is off in this fixture, so
        // this checks activation without relying on a live market endpoint.
        JsonElement stockSession = await OpenSessionAsync();
        List<JsonElement> stockFrames = await StreamAsync(new
        {
            sessionId = stockSession.GetProperty("sessionId").GetString(),
            messages = new[] { new { role = "user", content = "What is the current AAPL stock price?" } },
            maxTokens = 512,
            think = false,
            temperature = 0,
            seed = 42,
        });
        Assert.Contains(StepsOf(stockFrames), step => step.Tool == SkillTools.ReadToolName
            && step.Skill == "market-data" && step.Ok);
    }

    // =====================================================================================
    // 6. running a skill's own script
    // =====================================================================================

    /// <summary>
    /// Catches a skill that can be read but not run.
    ///
    /// <para>
    /// Reading a skill is a file copy; running one crosses every boundary this app
    /// has — the script runner resolves a path inside the skill, the interpreter
    /// starts in process because iOS forbids a child, the sandbox lets it write to the
    /// session workspace and nowhere else, and openpyxl has to be importable from the
    /// staged runtime. A spreadsheet that is a real ZIP container is the shortest
    /// proof that all of it worked, and asserting the zip header rather than the
    /// extension is what makes it a proof: a traceback saved as <c>.xlsx</c> also has
    /// the right name.
    /// </para>
    /// </summary>
    /// <summary>
    /// The deck the model promises has to be REACHABLE, not merely written.
    ///
    /// <para>
    /// "Search the news and make me a pptx" was reported as never working. The deck was
    /// in fact being produced; what failed was the last step -- the link. The runner
    /// hands the page /api/code/artifacts/{runId}/{name} and the app mapped no such
    /// route, so every generated document answered "error: not found" when tapped.
    /// This walks the whole path: ask for a deck, take the URL out of the frames the
    /// page would render, and fetch it.
    /// </para>
    /// </summary>
    [LiveDocumentsFact]
    public async Task TheDeckTheModelAnnouncesCanActuallyBeDownloaded()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true, interpreter: true);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;

        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            skills = new[] { DocumentsSkillId },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = $"Use the {DocumentsSkillId} skill to make a PowerPoint deck saved as news.pptx in my "
                        + "working directory. Give it a title slide reading 'Top 10 Hot News' and one bullets slide "
                        + "listing three headlines you invent. Tell me when the file is there.",
                },
            },
            maxTokens = 1024,
            think = false,
        });

        List<string> urls = ArtifactUrlsOf(frames);
        Console.WriteLine($"scenario pptx link: {(urls.Count == 0 ? "(none)" : string.Join(", ", urls))}");
        Assert.True(urls.Count > 0, "the run produced no downloadable file at all: " + Describe(StepsOf(frames)));

        string deck = Assert.Single(urls, u => u.EndsWith(".pptx", StringComparison.OrdinalIgnoreCase));
        HttpResponseMessage response = await Client.GetAsync(deck);
        Assert.True(response.IsSuccessStatusCode,
            $"the link the model showed ({deck}) answered {(int)response.StatusCode} {response.StatusCode}");

        byte[] bytes = await response.Content.ReadAsByteArrayAsync();
        // A .pptx is a zip; "PK\x03\x04" is the only cheap proof it is not an error page.
        Assert.True(bytes.Length > 4 && bytes[0] == 0x50 && bytes[1] == 0x4B,
            $"the download was {bytes.Length} bytes and did not start with a zip header");
    }

    /// <summary>Every downloadable file URL the streamed frames advertise.</summary>
    private static List<string> ArtifactUrlsOf(List<JsonElement> frames)
    {
        var urls = new List<string>();
        foreach (JsonElement frame in frames)
        {
            if (!frame.TryGetProperty("files", out JsonElement files) || files.ValueKind != JsonValueKind.Array)
                continue;
            foreach (JsonElement file in files.EnumerateArray())
            {
                if (file.TryGetProperty("url", out JsonElement url) && url.GetString() is { Length: > 0 } text
                    && !urls.Contains(text))
                {
                    urls.Add(text);
                }
            }
        }
        return urls;
    }

    [LiveDocumentsFact]
    public async Task ASkillsOwnScriptRunsAndTheSpreadsheetItWritesIsARealWorkbook()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true, interpreter: true);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);
        File.WriteAllText(Path.Combine(workspace, SalesFileName), SalesCsv());

        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            skills = new[] { DocumentsSkillId },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = $"The file {SalesFileName} in my working directory has the columns region, units and "
                        + $"revenue. Use the {DocumentsSkillId} skill to turn it into an Excel workbook saved as "
                        + "sales.xlsx in my working directory. Tell me when the file is there.",
                },
            },
            maxTokens = 1024,
            think = false,
        });

        string answer = TextOf(frames);
        List<SkillStep> steps = StepsOf(frames);
        List<Progress> progress = ProgressOf(frames);
        Console.WriteLine($"scenario xlsx: {Describe(steps)} | {Describe(progress)}");

        Assert.True(steps.Any(step => step.Skill == DocumentsSkillId),
            $"the '{DocumentsSkillId}' skill was selected for this request and never touched. "
            + $"Steps: {Describe(steps)}; answer: {answer}");

        Assert.True(
            progress.Any(p => p.Phase == "finished"
                && (p.Tool == SkillTools.RunToolName || SkillToolNames.CodeTools.Contains(p.Tool, StringComparer.Ordinal))),
            $"nothing was executed: no script run and no shell call finished. Frames: {Describe(progress)}; "
            + $"answer: {answer}");

        string[] workbooks = Directory.GetFiles(workspace, "*.xlsx", SearchOption.AllDirectories);
        Assert.True(workbooks.Length > 0,
            $"no .xlsx reached the session workspace.\n  {Listing(workspace)}\n\nanswer: {answer}");

        string workbook = workbooks[0];
        long size = new FileInfo(workbook).Length;
        Assert.True(size > 1000, $"{workbook} is {size} bytes, which is too small to be a workbook of six rows");
        Assert.True(StartsWith(workbook, ZipMagic),
            $"{workbook} is named like a workbook but does not begin with a zip header, so it is not one. "
            + $"It starts: {Preview(workbook)}");

        // A zip header proves a container, not a conversion. An empty workbook, or one
        // built from numbers the model invented, has the same header as the right
        // answer -- so the invented region names are what tie these bytes back to the
        // CSV this test planted. They exist nowhere else.
        string sheets = SheetXmlOf(workbook);
        string[] regions = Sales.Select(row => row.Region).Distinct(StringComparer.Ordinal).ToArray();
        string[] absent = regions
            .Where(region => !sheets.Contains(region, StringComparison.OrdinalIgnoreCase))
            .ToArray();
        Assert.True(absent.Length == 0,
            $"the workbook does not mention {string.Join(" or ", absent)}, so whatever it contains did not come "
            + $"from {SalesFileName}. Answer: {answer}");

        // The skill's documented contract, and the whole reason make_xlsx.py exists:
        // nothing on this device recalculates a sheet, so a formula cell that carries
        // no cached value reads as EMPTY to every reader -- including this skill's own
        // analyze_table.py. A workbook full of formulas and no values looks perfect in
        // a file listing and is blank when opened.
        MatchCollection formulas = Regex.Matches(sheets, @"<c\b[^>]*>(?:(?!</c>).)*?<f[ >](?:(?!</c>).)*?</c>",
            RegexOptions.Singleline);
        // "<v[ >]" is not enough: openpyxl's own output for an uncalculated formula is
        // <v />, an EMPTY value element, which such a pattern accepts as a cached value
        // while every reader still sees a blank cell. The value has to have CONTENT,
        // hence a '>' followed by something that is not the start of the next tag.
        Match hollow = formulas.FirstOrDefault(cell => !Regex.IsMatch(cell.Value, @"<v[^>]*>[^<]"))!;
        Assert.True(hollow is null,
            $"a formula cell carries no cached value, so it reads as blank to anything that does not "
            + $"recalculate: {hollow?.Value}");
        Console.WriteLine($"scenario xlsx: {formulas.Count} formula cell(s), all cached; regions all present");
    }

    /// <summary>
    /// Every worksheet part of an .xlsx, plus the shared string table the cells point
    /// into -- which is where openpyxl puts text, so a search of the sheets alone finds
    /// no region names at all.
    /// </summary>
    private static string SheetXmlOf(string workbook)
    {
        using var archive = System.IO.Compression.ZipFile.OpenRead(workbook);
        var text = new StringBuilder();
        foreach (System.IO.Compression.ZipArchiveEntry entry in archive.Entries)
        {
            if (!entry.FullName.StartsWith("xl/worksheets/", StringComparison.Ordinal)
                && entry.FullName != "xl/sharedStrings.xml")
                continue;
            using var reader = new StreamReader(entry.Open());
            text.AppendLine(reader.ReadToEnd());
        }
        return text.ToString();
    }

    // =====================================================================================
    // 7. producing a document
    // =====================================================================================

    /// <summary>
    /// Catches "here is your report" with no report behind it.
    ///
    /// <para>
    /// The most convincing failure this app has: a model that describes a PDF it never
    /// wrote, in a turn where every frame looks healthy. Only the bytes settle it, so
    /// the assertion is on the five bytes a PDF has to begin with and on a size no
    /// error message reaches. It is separate from the workbook scenario above because
    /// the two fail for different reasons — that one for openpyxl and the script
    /// runner, this one for reportlab and the page-drawing path — and a single test
    /// covering both would report either failure as the same red line.
    /// </para>
    /// </summary>
    [LiveDocumentsFact]
    public async Task ARequestedPdfReportIsRealPdfBytesAndNotAPromiseOfOne()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true, interpreter: true);
        await LoadAsync(model);

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);
        File.WriteAllText(Path.Combine(workspace, SalesFileName), SalesCsv());

        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            skills = new[] { DocumentsSkillId },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = $"The file {SalesFileName} in my working directory has the columns region, units and "
                        + $"revenue. Use the {DocumentsSkillId} skill to produce a one-page PDF report of that data, "
                        + "saved as sales.pdf in my working directory. Tell me when the file is there.",
                },
            },
            maxTokens = 1024,
            think = false,
        });

        string answer = TextOf(frames);
        Console.WriteLine($"scenario pdf: {Describe(StepsOf(frames))} | {Describe(ProgressOf(frames))}");

        string[] reports = Directory.GetFiles(workspace, "*.pdf", SearchOption.AllDirectories);
        Assert.True(reports.Length > 0,
            $"the turn ended with no PDF in the session workspace.\n  {Listing(workspace)}\n\nanswer: {answer}");

        string report = reports[0];
        long size = new FileInfo(report).Length;
        Assert.True(size > 500, $"{report} is {size} bytes, which is smaller than an empty PDF");
        Assert.True(StartsWith(report, PdfMagic),
            $"{report} does not begin with %PDF-, so whatever was written is not a PDF. "
            + $"It starts: {Preview(report)}");
    }

    // =====================================================================================
    // 8. a photo the user attached, turned into a document
    // =====================================================================================

    /// <summary>
    /// "Here is a photo, make it a PDF." Reported as failing, and it did.
    ///
    /// <para>
    /// Two things were wrong and only the second is about documents. The app staged
    /// only TEXT uploads into the working directory a program runs in, so a photo was
    /// something the model could see and not something it could open: it was told,
    /// truthfully, that it could run programs, and then spent the turn guessing at a
    /// filename that never existed. And the PDF writer had no one-step way to put a
    /// picture on a page, so even with the file present the model had to compose a
    /// JSON document by hand for a task with no choices in it.
    /// </para>
    /// <para>
    /// The assertion is on bytes, as with the report scenario: a PDF that pypdf would
    /// refuse and a sentence claiming success look identical from the outside. What is
    /// additionally checked is that the picture is IN it — a one-page PDF of nothing
    /// would satisfy the magic-number test.
    /// </para>
    /// </summary>
    [LiveDocumentsFact]
    public async Task APhotoTheUserAttachedIsTurnedIntoARealPdf()
    {
        Assert.Null(Unavailable(out CatalogModel model, out string weights));
        Start(model, weights, skills: true, interpreter: true);
        await LoadAsync(model);

        // Uploaded through the same route the paperclip uses, so what is under test is
        // the path a photo really takes: /api/upload, then the stored name in the body.
        JsonElement upload = await UploadAsync(MediaFixtures.RedCircleOnWhitePng(512), "IMG_0004.png");
        Assert.Equal("image", upload.GetProperty("mediaType").GetString());
        string stored = upload.GetProperty("file").GetString()!;

        JsonElement session = await OpenSessionAsync();
        string sessionId = session.GetProperty("sessionId").GetString()!;
        string workspace = WorkspaceOf(sessionId);

        List<JsonElement> frames = await StreamAsync(new
        {
            sessionId,
            skills = new[] { DocumentsSkillId },
            messages = new[]
            {
                new
                {
                    role = "user",
                    content = "Convert the attached picture into a PDF file called photo.pdf. "
                        + "Use the documents skill.",
                    imagePaths = new[] { stored },
                    stillImagePaths = new[] { stored },
                    // The array the page now sends, and the one that makes the file
                    // exist on disk where a program can open it.
                    attachments = new[]
                    {
                        new { file = stored, fileName = "IMG_0004.png", mediaType = "image" },
                    },
                },
            },
            maxTokens = 1024,
            think = false,
        });

        string answer = TextOf(frames);
        Console.WriteLine($"scenario photo->pdf: {Describe(StepsOf(frames))} | {Describe(ProgressOf(frames))}");

        // The staging half, checked separately from the document half: "the model did
        // not make a PDF" and "the model never had the photo" are different failures
        // with different owners, and one red line for both is how an afternoon goes.
        string staged = Path.Combine(workspace, "IMG_0004.png");
        Assert.True(File.Exists(staged),
            "the attached photo was never staged into the working directory, so nothing the model "
            + $"ran could open it.\n  {Listing(workspace)}\n\nanswer: {answer}");

        string[] made = Directory.GetFiles(workspace, "*.pdf", SearchOption.AllDirectories);
        Assert.True(made.Length > 0,
            $"the turn ended with no PDF in the session workspace.\n  {Listing(workspace)}\n\nanswer: {answer}");

        string pdf = made[0];
        Assert.True(StartsWith(pdf, PdfMagic),
            $"{pdf} does not begin with %PDF-, so whatever was written is not a PDF. "
            + $"It starts: {Preview(pdf)}");

        // And the picture is IN it. Not asserted by size: a flat disc on white
        // compresses to a few kilobytes, so a threshold either passes on an empty page
        // or fails on a real one. What settles it is the image XObject reportlab writes
        // — an uncompressed dictionary in the file — carrying the source's own
        // dimensions. A one-page PDF of nothing has no such object at all.
        string raw = File.ReadAllText(pdf, System.Text.Encoding.Latin1);
        Assert.True(raw.Contains("/Subtype /Image", StringComparison.Ordinal),
            $"{pdf} carries no image object, so it is a page with nothing on it — which is "
            + $"not what converting a photo means.\n\nanswer: {answer}");
        Assert.True(raw.Contains("/Width 512", StringComparison.Ordinal)
                    && raw.Contains("/Height 512", StringComparison.Ordinal),
            $"the image in {pdf} is not the 512x512 picture that was attached.\n\nanswer: {answer}");
    }

    /// <summary>Upload through the app's own route, as the page's paperclip does.</summary>
    private async Task<JsonElement> UploadAsync(byte[] content, string fileName)
    {
        using var form = new MultipartFormDataContent();
        var part = new ByteArrayContent(content);
        part.Headers.ContentType = new MediaTypeHeaderValue("application/octet-stream");
        form.Add(part, "file", fileName);

        using HttpResponseMessage response = await Client.PostAsync("/api/upload", form);
        string payload = await response.Content.ReadAsStringAsync();
        Assert.True(response.IsSuccessStatusCode, $"uploading {fileName} failed: {(int)response.StatusCode} {payload}");
        return JsonSerializer.Deserialize<JsonElement>(payload);
    }

    // =====================================================================================
    // the data these scenarios are built on
    // =====================================================================================

    /// <summary>
    /// The six numbers the model is asked to add and then multiply.
    ///
    /// <para>
    /// Chosen so the product is 131,859,000 — a number no model produces from memory
    /// and none reaches by guessing. An answer that matches it is evidence the program
    /// ran, which is the only thing the code scenario is really asking.
    /// </para>
    /// </summary>
    private static readonly int[] Readings = [3, 14, 15, 92, 65, 35];

    private const string SalesFileName = "sales.csv";

    /// <summary>
    /// Six rows with invented region names, so an answer about them cannot come from
    /// anywhere but the file.
    /// </summary>
    private static readonly (string Region, int Units, int Revenue)[] Sales =
    [
        ("Northgate", 12, 4180),
        ("Ravensworth", 7, 2650),
        ("Bellhaven", 19, 7310),
        ("Northgate", 5, 1720),
        ("Bellhaven", 3, 940),
        ("Ravensworth", 11, 3980),
    ];

    private static string SalesCsv()
    {
        var csv = new StringBuilder("region,units,revenue\n");
        foreach ((string region, int units, int revenue) in Sales)
        {
            csv.Append(region).Append(',')
               .Append(units.ToString(CultureInfo.InvariantCulture)).Append(',')
               .Append(revenue.ToString(CultureInfo.InvariantCulture)).Append('\n');
        }
        return csv.ToString();
    }

    /// <summary>
    /// Ordinary English of a known length, with one fact planted in it.
    ///
    /// <para>
    /// Generated rather than pasted so the test can say exactly how long it is and
    /// where the answer sits, and written as sentences rather than as filler because
    /// the characters-per-token ratio the floor is derived from only holds for
    /// prose — a page of hex or of one repeated word tokenizes at a rate that would
    /// make the floor mean nothing.
    /// </para>
    /// </summary>
    private static string ShiftLog(int lines, int factLine, string fact)
    {
        var text = new StringBuilder();
        for (int line = 1; line <= lines; line++)
        {
            text.Append("Line ").Append(line.ToString("000", CultureInfo.InvariantCulture)).Append(". ");
            text.AppendLine(line == factLine ? fact : Filler(line));
        }
        return text.ToString();
    }

    private static string Filler(int line) => (line % 4) switch
    {
        0 => $"Shift {line} at the Ravensworth pumping station ran without incident; flow held at {40 + line % 9} "
            + "litres per second and the duty operator logged nothing unusual overnight.",
        1 => $"The inlet screen was cleared twice, the second time at {line % 12 + 1} minutes past the hour, and "
            + "the reservoir level recovered on its own before the morning handover.",
        2 => "Routine inspection of the Ravensworth valve house found the gaskets dry, the housing clean and the "
            + $"telemetry link steady for the whole of shift {line}.",
        _ => $"The duty engineer walked the {line % 7 + 2} kilometre culvert, noted the usual silt at the third "
            + "chamber, and reported no change from the previous week.",
    };

    // =====================================================================================
    // reading the frames the page reads
    // =====================================================================================

    /// <summary>One <c>tool_progress</c> frame, as the page's activity line reads it.</summary>
    private readonly record struct Progress(string Phase, string Tool, double Seconds, string Detail);

    /// <summary>One <c>skill_step</c> frame, as the page's trace reads it.</summary>
    private readonly record struct SkillStep(string Tool, string Skill, string Detail, bool Ok, int Round, int Files);

    private static List<Progress> ProgressOf(IEnumerable<JsonElement> frames) => frames
        .Where(frame => frame.TryGetProperty("tool_progress", out _))
        .Select(frame => new Progress(
            Text(frame, "tool_progress"),
            Text(frame, "tool"),
            frame.TryGetProperty("seconds", out JsonElement seconds) && seconds.ValueKind == JsonValueKind.Number
                ? seconds.GetDouble()
                : 0,
            Text(frame, "detail")))
        .ToList();

    private static List<SkillStep> StepsOf(IEnumerable<JsonElement> frames) => frames
        .Where(frame => frame.TryGetProperty("skill_step", out _))
        .Select(frame => new SkillStep(
            Text(frame, "skill_step"),
            Text(frame, "skill"),
            Text(frame, "detail"),
            frame.TryGetProperty("ok", out JsonElement ok) && ok.ValueKind == JsonValueKind.True,
            frame.TryGetProperty("round", out JsonElement round) && round.ValueKind == JsonValueKind.Number
                ? round.GetInt32()
                : 0,
            frame.TryGetProperty("files", out JsonElement files) && files.ValueKind == JsonValueKind.Array
                ? files.GetArrayLength()
                : 0))
        .ToList();

    /// <summary>A frame member as a string; absent and JSON null both read as empty.</summary>
    private static string Text(JsonElement frame, string name) =>
        frame.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
            ? value.GetString() ?? string.Empty
            : string.Empty;

    private static string Describe(List<Progress> progress) => progress.Count == 0
        ? "(no tool_progress frames)"
        : string.Join(", ", progress
            .Select(p => $"{p.Phase}:{(p.Tool.Length == 0 ? "?" : p.Tool)}")
            .Distinct(StringComparer.Ordinal));

    private static string Describe(List<SkillStep> steps) => steps.Count == 0
        ? "(no skill_step frames)"
        : string.Join(", ", steps.Select(s =>
            $"round {s.Round} {s.Tool} skill={(s.Skill.Length == 0 ? "-" : s.Skill)} "
            + $"path={(s.Detail.Length == 0 ? "-" : s.Detail)} ok={s.Ok} files={s.Files}"));

    // =====================================================================================
    // reading what the run left behind
    // =====================================================================================

    private static Dictionary<string, string> PythonFilesIn(string workspace) =>
        Directory.EnumerateFiles(workspace, "*.py", SearchOption.AllDirectories)
            .ToDictionary(path => path, File.ReadAllText, StringComparer.Ordinal);

    /// <summary>Everything in the workspace, for a failure message that says what IS there.</summary>
    private static string Listing(string workspace)
    {
        if (!Directory.Exists(workspace))
            return $"({workspace} does not exist)";
        string[] files = Directory.GetFiles(workspace, "*", SearchOption.AllDirectories);
        return files.Length == 0
            ? $"({workspace} is empty)"
            : string.Join("\n  ", files.Select(file =>
                $"{Path.GetRelativePath(workspace, file)} ({new FileInfo(file).Length} bytes)"));
    }

    private static readonly byte[] PdfMagic = "%PDF-"u8.ToArray();
    private static readonly byte[] ZipMagic = [0x50, 0x4B, 0x03, 0x04];

    /// <summary>
    /// Whether a file really is what its name claims. Asserted on the bytes rather
    /// than the extension because a traceback saved as <c>report.pdf</c> passes every
    /// check that only looks at the name.
    /// </summary>
    private static bool StartsWith(string path, byte[] magic)
    {
        using FileStream stream = File.OpenRead(path);
        byte[] head = new byte[magic.Length];
        return stream.ReadAtLeast(head, magic.Length, throwOnEndOfStream: false) == magic.Length
            && head.AsSpan().SequenceEqual(magic);
    }

    /// <summary>The first line of a file that failed its format check, so the failure says why.</summary>
    private static string Preview(string path)
    {
        byte[] head = new byte[96];
        using FileStream stream = File.OpenRead(path);
        int read = stream.ReadAtLeast(head, head.Length, throwOnEndOfStream: false);
        return string.Concat(Encoding.UTF8.GetString(head, 0, read)
            .Select(c => char.IsControl(c) ? '.' : c));
    }

    /// <summary>
    /// Whether <paramref name="text"/> names <paramref name="value"/> as a number in
    /// its own right. The digit boundaries are the point: without them 15 matches
    /// inside 3150 and every check here would be satisfied by any long enough number.
    /// </summary>
    private static bool Mentions(string text, long value) =>
        Regex.IsMatch(text, $@"(?<![\d.]){Regex.Escape(value.ToString(CultureInfo.InvariantCulture))}(?!\d)");

    /// <summary>
    /// Whether an answer states <paramref name="value"/>, however the model chose to
    /// punctuate it.
    ///
    /// <para>
    /// Thousands separators are removed first — a model writes 131,859,000 as often as
    /// 131859000 — and only the separators BETWEEN digit groups come out, so a list
    /// like "3, 14, 15" is not silently welded into one number by the normalisation
    /// that was meant to help.
    /// </para>
    /// </summary>
    private static bool States(string answer, long value) =>
        Mentions(answer, value)
        || Mentions(Regex.Replace(answer, @"(?<=\d)[,_ ](?=\d{3}(?!\d))", string.Empty), value);
}
