// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Replays logged Nemotron-H Reasoning-128K replies through the REAL output parser, the one
// ChatProtocolRegistry creates, the way each kind of consumer drives it: held (an API stream,
// the CLI, a collector) and retracting (the Web UI / TensorAgent), fed in pieces of 1, 3 and 7
// characters and as the whole text. It checks the stray-reasoning split against today's parse
// and reports how long a held stream waits and how many logged closes the window covers.

using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.Runtime;

internal static class Program
{
    private const string OpenBlock = "<think>\n";
    private const string ClosedBlock = "<think></think>";
    // The whole text first: every split after it must decide the reply as it did.
    private static readonly int[] Chunks = { 0, 1, 3, 7 };

    private sealed record Reply(
        string Source, int Line, bool RequestThinking, string PromptTail, string Text,
        int Tokens, double TokensPerSecond, string Finish);

    private sealed record Run(
        string Content, string Thinking, string Calls, int FirstContentAt, int FirstCallAt,
        int Retractions, int SuffixFailures, bool CloseEverShown);

    private static readonly Regex StartRe = new(
        @"chat\.start arch=(?<arch>\S+) maxTokens=\d+ thinking=(?<think>True|False)", RegexOptions.CultureInvariant);
    private static readonly Regex CompleteRe = new(
        @"chat\.complete tokens=(?<tokens>\d+) .*?tokensPerSec=(?<tps>[\d.]+) finishReason=(?<finish>\S+) assistantOutput=""(?<out>.*)""\s*$",
        RegexOptions.CultureInvariant);
    private static readonly Regex TruncatedRe = new(@"\.\.\.\(\+\d+ chars\)$", RegexOptions.CultureInvariant);

    private static readonly List<ToolFunction> Tools = new()
    {
        Tool("read_file", "path"), Tool("list_files", "path"), Tool("write_file", "path", "content"),
        Tool("shell", "command"), Tool("run_command", "command"), Tool("web_fetch", "url"),
    };

    private static int Main(string[] args)
    {
        var logs = new List<string>();
        var opened = new List<(string File, int From, int To)>();
        string outDir = Path.Combine("artifacts", "nemotron-thinkoff-replay");
        for (int i = 0; i < args.Length; i++)
        {
            if (args[i] == "--out" && i + 1 < args.Length)
                outDir = args[++i];
            else if (args[i] == "--opened" && i + 1 < args.Length)
                opened.Add(ParseRange(args[++i]));
            else
                logs.Add(args[i]);
        }
        if (logs.Count == 0)
        {
            Console.Error.WriteLine(
                "usage: NemotronThinkOffReplay [--out DIR] [--opened LOG:FROM-TO ...] LOG...\n" +
                "  --opened marks requests whose system prompt carried {'reasoning': True}: their\n" +
                "  prompt opened the block although the logged request flag says thinking=False.");
            return 2;
        }

        ChatProtocol protocol = ChatProtocolRegistry.For("nemotron_h")
            ?? throw new InvalidOperationException("nemotron_h has no chat protocol");
        int skipped = 0;
        var replies = new List<Reply>();
        foreach (string log in logs)
            replies.AddRange(ReadReplies(log, opened, ref skipped));

        var problems = new List<string>();
        var rows = new List<object>();
        var holdExtraChars = new List<int>();
        var holdExtraSeconds = new List<double>();
        var strayCloses = new List<(string Where, int At)>();
        int window = WindowOf(protocol);

        foreach (Reply reply in replies)
        {
            string where = $"{Path.GetFileName(reply.Source)}:{reply.Line}";
            bool promptOpened = reply.PromptTail == OpenBlock;
            int close = reply.Text.IndexOf("</think>", StringComparison.Ordinal);
            string kind = promptOpened
                ? (reply.RequestThinking ? "think-on" : "marker-opened")
                : close >= 0 ? "off-stray-close" : "off-ordinary";
            if (kind == "off-stray-close")
                strayCloses.Add((where, close));

            Run? whole = null;
            foreach (int chunk in Chunks)
            {
                // Today's parse: the request flag, no prompt tail. The marker-opened requests
                // are compared with the thinking-on parse, which is what their prompt asked for.
                Run baseline = Drive(NewBaseline(), reply.Text, chunk, promptOpened || reply.RequestThinking, null, false);
                Run held = Drive(protocol.CreateOutputParser!(), reply.Text, chunk, reply.RequestThinking, reply.PromptTail, false);
                Run shown = Drive(protocol.CreateOutputParser!(), reply.Text, chunk, reply.RequestThinking, reply.PromptTail, true);
                if (chunk == 0) whole = held;
                void Fail(string why) => problems.Add($"{where} [{kind}] chunk={chunk}: {why}");

                if (held.Calls != baseline.Calls || shown.Calls != baseline.Calls)
                    Fail($"calls differ: today [{baseline.Calls}] held [{held.Calls}] retracting [{shown.Calls}]");
                if (chunk > 0 && (held.FirstCallAt != baseline.FirstCallAt || shown.FirstCallAt != baseline.FirstCallAt))
                    Fail($"first call at {held.FirstCallAt}/{shown.FirstCallAt}, today at {baseline.FirstCallAt}");
                if (held.Content != shown.Content || held.Thinking != shown.Thinking)
                    Fail("held and retracting parses end differently");
                // The transcript and the non-streaming APIs parse the reply whole; the streams
                // parse it in pieces. They must decide it alike, or the follow-up a page sends
                // back no longer matches the recorded turn.
                if (chunk > 0 && whole != null
                    && (Squash(held.Content) != Squash(whole.Content) || Squash(held.Thinking) != Squash(whole.Thinking)
                        || held.Calls != whole.Calls))
                    Fail("the whole reply and the streamed one are decided differently");
                if (shown.SuffixFailures > 0)
                    Fail("a retraction was not the end of the shown answer");

                switch (kind)
                {
                    case "think-on":
                    case "marker-opened":
                    case "off-ordinary":
                        if (Squash(held.Content) != Squash(baseline.Content) || Squash(held.Thinking) != Squash(baseline.Thinking))
                            Fail("the split changed for a reply that needed no change");
                        if (shown.Retractions > 0)
                            Fail("retracted text from a reply that needed no change");
                        if (chunk > 0 && shown.FirstContentAt != baseline.FirstContentAt)
                            Fail($"retracting stream shows its first answer character at {shown.FirstContentAt}, today at {baseline.FirstContentAt}");
                        if (kind == "off-ordinary" && chunk == 1 && baseline.FirstContentAt > 0)
                        {
                            int extra = held.FirstContentAt - baseline.FirstContentAt;
                            holdExtraChars.Add(extra);
                            double charsPerToken = reply.Tokens > 0 ? (double)reply.Text.Length / reply.Tokens : 4.0;
                            holdExtraSeconds.Add(reply.TokensPerSecond > 0 ? extra / charsPerToken / reply.TokensPerSecond : 0);
                        }
                        break;

                    case "off-stray-close":
                        if (close >= window)
                            break;   // reported as outside the window, below
                        if (held.CloseEverShown || shown.CloseEverShown)
                            Fail("a </think> was shown as answer text");
                        if (held.Calls.Length == 0 && Squash(held.Thinking) != Squash(reply.Text.Substring(0, close)))
                            Fail("the reasoning before the close did not become thinking");
                        if (Squash(reply.Text.Substring(close + 8)) is var tail && tail.StartsWith("<TOOLCALL>[]</TOOLCALL>", StringComparison.Ordinal)
                            && tail.Length == "<TOOLCALL>[]</TOOLCALL>".Length
                            && (Squash(held.Content).Length > 0 || held.Calls.Length > 0))
                            Fail("an empty call list left text or a call behind");
                        break;
                }
            }

            rows.Add(new
            {
                source = where,
                kind,
                requestThinking = reply.RequestThinking,
                promptTail = reply.PromptTail,
                chars = reply.Text.Length,
                closeAt = close,
                finish = reply.Finish,
                visibleAnswer = Clip(whole?.Content),
                thinking = Clip(whole?.Thinking),
                calls = whole?.Calls,
            });
        }

        int inside = strayCloses.Count(c => c.At < window);
        holdExtraChars.Sort();
        holdExtraSeconds.Sort();
        var summary = new StringBuilder();
        summary.AppendLine(CultureInfo.InvariantCulture, $"logs: {string.Join(", ", logs.Select(Path.GetFileName))}");
        summary.AppendLine(FormattableString.Invariant($"replies: {replies.Count} (skipped {skipped} truncated); ")
            + string.Join(", ", rows.GroupBy(r => (string)r.GetType().GetProperty("kind")!.GetValue(r)!)
                .OrderBy(g => g.Key).Select(g => FormattableString.Invariant($"{g.Key} {g.Count()}"))));
        summary.AppendLine(CultureInfo.InvariantCulture, $"stray-reasoning window: {window} characters");
        summary.AppendLine(FormattableString.Invariant($"stray closes inside the window: {inside} of {strayCloses.Count}")
            + (strayCloses.Count > 0
                ? FormattableString.Invariant($" ({100.0 * inside / strayCloses.Count:F0}%); at ")
                    + string.Join(", ", strayCloses.Select(c => FormattableString.Invariant($"{c.Where}@{c.At}")))
                : string.Empty));
        if (holdExtraChars.Count > 0)
        {
            summary.AppendLine(CultureInfo.InvariantCulture,
                $"held streams, ordinary thinking-off replies (n={holdExtraChars.Count}): first answer character later by " +
                $"median {Pct(holdExtraChars, 0.5)} / p90 {Pct(holdExtraChars, 0.9)} / max {holdExtraChars[^1]} characters, " +
                $"about {Pct(holdExtraSeconds, 0.5):F1} / {Pct(holdExtraSeconds, 0.9):F1} / {holdExtraSeconds[^1]:F1} s at each reply's logged decode rate");
        }
        summary.AppendLine(CultureInfo.InvariantCulture, $"problems: {problems.Count}");
        foreach (string problem in problems)
            summary.AppendLine("  " + problem);

        Directory.CreateDirectory(outDir);
        File.WriteAllText(Path.Combine(outDir, "summary.txt"), summary.ToString());
        File.WriteAllText(Path.Combine(outDir, "replies.json"),
            JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
        Console.Write(summary);
        Console.WriteLine($"report: {Path.GetFullPath(outDir)}");
        return problems.Count == 0 ? 0 : 1;
    }

    private static IOutputParser NewBaseline() => new NemotronOutputParser();

    /// <summary>The window the registry's parser was built with, found by probing it: the
    /// longest undecided prefix after which a close still decides the reply as reasoning.</summary>
    private static int WindowOf(ChatProtocol protocol)
    {
        int lo = 0, hi = 1 << 16;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) / 2;
            IOutputParser parser = protocol.CreateOutputParser!();
            parser.Init(false, null);
            parser.SetGenerationPromptSuffix(ClosedBlock);
            ParsedOutput parsed = parser.Add(new string('x', mid) + "</think>answer", true);
            if (parsed.Thinking.Length > 0) lo = mid; else hi = mid - 1;
        }
        return lo + 1;
    }

    private static Run Drive(IOutputParser parser, string text, int chunk, bool thinking, string? promptTail, bool retract)
    {
        parser.Init(thinking, Tools);
        if (retract) parser.AcceptRetractions();
        parser.SetGenerationPromptSuffix(promptTail);
        var answer = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new List<string>();
        int consumed = 0, firstContent = -1, firstCall = -1, retractions = 0, suffixFailures = 0;
        bool closeShown = false;
        bool whole = chunk <= 0;
        IEnumerable<string> pieces = whole ? new[] { text } : Pieces(text, chunk);
        foreach (string piece in pieces.Append(string.Empty))
        {
            bool done = piece.Length == 0 || whole;
            consumed += piece.Length;
            ParsedOutput parsed = parser.Add(piece, done);
            if (parsed.RetractedContent.Length > 0)
            {
                retractions++;
                if (answer.ToString().EndsWith(parsed.RetractedContent, StringComparison.Ordinal))
                    answer.Length -= parsed.RetractedContent.Length;
                else
                    suffixFailures++;
            }
            reasoning.Append(parsed.Thinking);
            answer.Append(parsed.Content);
            if (parsed.ToolCalls is { Count: > 0 })
            {
                calls.AddRange(parsed.ToolCalls.Select(c => c.ToString()));
                if (firstCall < 0) firstCall = consumed;
            }
            if (firstContent < 0 && answer.Length > 0) firstContent = consumed;
            closeShown |= answer.ToString().Contains("</think>", StringComparison.Ordinal);
            if (whole) break;
        }
        return new Run(answer.ToString(), reasoning.ToString(), string.Join("|", calls), firstContent, firstCall,
            retractions, suffixFailures, closeShown);
    }

    private static IEnumerable<Reply> ReadReplies(string log, List<(string File, int From, int To)> opened, ref int skipped)
    {
        var replies = new List<Reply>();
        string? arch = null;
        bool thinking = false;
        int lineNo = 0, startLine = 0;
        foreach (string line in File.ReadLines(log))
        {
            lineNo++;
            Match start = StartRe.Match(line);
            if (start.Success)
            {
                arch = start.Groups["arch"].Value;
                thinking = start.Groups["think"].Value == "True";
                startLine = lineNo;
                continue;
            }
            Match complete = CompleteRe.Match(line);
            if (!complete.Success || arch == null || !arch.StartsWith("nemotron_h", StringComparison.Ordinal))
                continue;
            string escaped = complete.Groups["out"].Value;
            if (TruncatedRe.IsMatch(escaped))
            {
                skipped++;
                continue;
            }
            bool markerOpened = opened.Any(o => SameLog(o.File, log) && startLine >= o.From && startLine <= o.To);
            string tail = thinking || markerOpened ? OpenBlock : ClosedBlock;
            replies.Add(new Reply(log, lineNo, thinking, tail, Unescape(escaped),
                int.Parse(complete.Groups["tokens"].Value, CultureInfo.InvariantCulture),
                double.Parse(complete.Groups["tps"].Value, CultureInfo.InvariantCulture),
                complete.Groups["finish"].Value));
        }
        return replies;
    }

    // A bare file name names every log of that name; a path names that one log.
    private static bool SameLog(string spec, string log)
        => spec.IndexOfAny(new[] { '/', '\\' }) >= 0
            ? string.Equals(Path.GetFullPath(spec), Path.GetFullPath(log), StringComparison.Ordinal)
            : string.Equals(spec, Path.GetFileName(log), StringComparison.Ordinal);

    // The log escapes newlines, carriage returns and tabs and nothing else.
    private static string Unescape(string text)
    {
        var sb = new StringBuilder(text.Length);
        for (int i = 0; i < text.Length; i++)
        {
            if (text[i] == '\\' && i + 1 < text.Length && text[i + 1] is 'n' or 'r' or 't')
            {
                sb.Append(text[i + 1] switch { 'n' => '\n', 'r' => '\r', _ => '\t' });
                i++;
                continue;
            }
            sb.Append(text[i]);
        }
        return sb.ToString();
    }

    private static (string File, int From, int To) ParseRange(string spec)
    {
        int colon = spec.LastIndexOf(':');
        string[] bounds = spec.Substring(colon + 1).Split('-');
        return (spec.Substring(0, colon), int.Parse(bounds[0], CultureInfo.InvariantCulture),
            int.Parse(bounds[^1], CultureInfo.InvariantCulture));
    }

    private static ToolFunction Tool(string name, params string[] parameters) => new()
    {
        Name = name,
        Parameters = parameters.ToDictionary(p => p, _ => new ToolParameter { Type = "string" }),
        Required = parameters.ToList(),
    };

    private static IEnumerable<string> Pieces(string text, int size)
    {
        for (int i = 0; i < text.Length; i += size)
            yield return text.Substring(i, Math.Min(size, text.Length - i));
    }

    private static string Squash(string text) => new(text.Where(c => !char.IsWhiteSpace(c)).ToArray());

    private static string? Clip(string? text) => text == null || text.Length <= 200 ? text : text.Substring(0, 200) + "...";

    private static T Pct<T>(List<T> sorted, double q) => sorted[Math.Min(sorted.Count - 1, (int)(sorted.Count * q))];
}
