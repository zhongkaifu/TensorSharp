// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Replays recorded model output through the engine's RepetitionGuard, token by token, the way
// InferenceEngine asks it: after every emitted token, on the output so far. The output is
// re-tokenized with the model's own GGUF tokenizer (only the tokenizer is loaded, no weights),
// so build this against two trees to compare their guards on the same tokens.
//
//   RepetitionGuardReplay --gguf MODEL.gguf [--runaway LOG.jsonl --match TEXT]
//                         [--outputs LOG.jsonl ...] [--bench TOKENS] [--out RESULT.json]
//
//   --runaway  the one logged output (chat.complete / chat.cancelled) whose line contains
//              --match: where, if anywhere, the guard stops it, with the stop's description.
//   --outputs  every logged output in these TensorSharp.Server / TensorAgent .jsonl logs:
//              how many the guard would stop, and which. Real output that ended normally is
//              the false-positive check.
//   --bench    the guard's cost per token over a TOKENS-long stream that never loops (random
//              ids, then the logged outputs end to end): wall time of the guard calls only.
//
// Output belongs in ignored artifacts/. This records; it does not decide pass/fail.

using System.Diagnostics;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

internal static class Program
{
    private static readonly Regex Output = new(
        @"chat\.(?<kind>complete|cancelled) tokens=(?<tokens>\d+).*?(?:finishReason=(?<finish>\S+) )?assistantOutput=""(?<out>.*)""\s*$",
        RegexOptions.CultureInvariant | RegexOptions.Singleline);

    private static int Main(string[] args)
    {
        string? gguf = null, runaway = null, match = null, outPath = null;
        var outputLogs = new List<string>();
        int bench = 0;
        for (int i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "--gguf": gguf = args[++i]; break;
                case "--runaway": runaway = args[++i]; break;
                case "--match": match = args[++i]; break;
                case "--outputs":
                    while (i + 1 < args.Length && !args[i + 1].StartsWith("--", StringComparison.Ordinal))
                        outputLogs.Add(args[++i]);
                    break;
                case "--bench": bench = int.Parse(args[++i]); break;
                case "--out": outPath = args[++i]; break;
                default:
                    Console.Error.WriteLine($"unknown argument {args[i]}");
                    return 2;
            }
        }
        if (gguf == null)
        {
            Console.Error.WriteLine("usage: RepetitionGuardReplay --gguf MODEL.gguf [--runaway LOG --match TEXT] [--outputs LOG...] [--bench N] [--out FILE]");
            return 2;
        }

        ITokenizer tokenizer;
        using (var file = new GgufFile(gguf))
            tokenizer = ModelBase.CreateTokenizerFromGguf(file);
        var result = new Dictionary<string, object?>
        {
            ["gguf"] = Path.GetFileName(gguf),
            ["guard"] = typeof(RepetitionGuard).Assembly.Location,
        };

        if (runaway != null)
        {
            LoggedOutput? found = ReadOutputs(runaway)
                .FirstOrDefault(o => match == null || o.Line.Contains(match, StringComparison.Ordinal));
            if (found == null)
            {
                Console.Error.WriteLine($"no logged output in {runaway} matches '{match}'");
                return 1;
            }
            (string text, string kind, int loggedTokens) = (found.Text, found.Kind, found.Tokens);
            List<int> tokens = tokenizer.Encode(text, addSpecial: false);
            Replay replay = Run(tokens, tokenizer);
            Console.WriteLine($"runaway: logged {kind} after {loggedTokens} tokens; {text.Length} chars re-tokenized to {tokens.Count} tokens");
            Console.WriteLine(replay.StopAt < 0
                ? $"  guard: no stop in {tokens.Count} tokens"
                : $"  guard: stops at token {replay.StopAt} (char {replay.StopChar}), period {replay.Period}, repeats {replay.Repeats}: {replay.What}");
            Console.WriteLine($"  guard cost: {replay.NsPerToken:F1} ns per token");
            result["runaway"] = new Dictionary<string, object?>
            {
                ["loggedKind"] = kind, ["loggedTokens"] = loggedTokens, ["chars"] = text.Length,
                ["tokens"] = tokens.Count, ["stopAt"] = replay.StopAt, ["stopChar"] = replay.StopChar,
                ["period"] = replay.Period, ["repeats"] = replay.Repeats, ["what"] = replay.What,
                ["tokensSaved"] = replay.StopAt < 0 ? 0 : loggedTokens - replay.StopAt,
                ["nsPerToken"] = replay.NsPerToken,
            };
        }

        var corpus = new List<int>();
        if (outputLogs.Count > 0)
        {
            int outputs = 0, longOutputs = 0;
            long tokensTotal = 0, longTokens = 0;
            var stops = new List<Dictionary<string, object?>>();
            var finishes = new Dictionary<string, int>();
            foreach (string log in outputLogs)
            {
                foreach (LoggedOutput o in ReadOutputs(log))
                {
                    List<int> tokens = tokenizer.Encode(o.Text, addSpecial: false);
                    outputs++;
                    tokensTotal += tokens.Count;
                    // Long enough for the long-block rule (4,096 tokens) to be asked at all.
                    if (tokens.Count >= 4096)
                    {
                        longOutputs++;
                        longTokens += tokens.Count;
                    }
                    finishes[o.Finish] = finishes.GetValueOrDefault(o.Finish) + 1;
                    Replay replay = Run(tokens, tokenizer);
                    if (replay.StopAt >= 0)
                        stops.Add(new()
                        {
                            ["log"] = Path.GetFileName(log), ["line"] = o.LineNumber, ["finish"] = o.Finish,
                            ["kind"] = o.Kind, ["tokens"] = tokens.Count, ["stopAt"] = replay.StopAt,
                            ["period"] = replay.Period, ["repeats"] = replay.Repeats, ["what"] = replay.What,
                        });
                    if (replay.StopAt < 0 && corpus.Count < 4_000_000)
                        corpus.AddRange(tokens);
                }
            }
            Console.WriteLine($"outputs: {outputs} logged outputs, {tokensTotal} tokens ({longOutputs} of them 4,096+ tokens long, {longTokens} tokens); the guard stops {stops.Count}");
            foreach (var stop in stops)
                Console.WriteLine($"  stop: {stop["log"]}:{stop["line"]} finish={stop["finish"]} tokens={stop["tokens"]} at {stop["stopAt"]} period {stop["period"]}: {stop["what"]}");
            result["outputs"] = new Dictionary<string, object?>
            {
                ["logs"] = outputLogs.Select(Path.GetFileName).ToArray(), ["count"] = outputs,
                ["tokens"] = tokensTotal, ["longOutputs"] = longOutputs, ["longTokens"] = longTokens,
                ["loggedFinishReasons"] = finishes, ["stops"] = stops,
            };
        }

        if (bench > 0)
        {
            var random = new Random(1);
            var streams = new List<(string Name, List<int> Tokens)>
            {
                ("random ids", Enumerable.Range(0, bench).Select(_ => random.Next(tokenizer.VocabSize)).ToList()),
            };
            if (corpus.Count > 0)
            {
                var looped = new List<int>(bench);
                while (looped.Count < bench)
                    looped.AddRange(corpus.Take(bench - looped.Count));
                streams.Add(("logged outputs end to end", looped));
            }
            var rows = new List<Dictionary<string, object?>>();
            foreach ((string name, List<int> stream) in streams)
            {
                Replay replay = Run(stream, tokenizer);
                Console.WriteLine($"bench: {name}, {stream.Count} tokens: {replay.NsPerToken:F1} ns per token"
                    + (replay.StopAt >= 0 ? $" (stopped at {replay.StopAt}: {replay.What})" : ", no stop"));
                rows.Add(new() { ["stream"] = name, ["tokens"] = stream.Count, ["nsPerToken"] = replay.NsPerToken, ["stopAt"] = replay.StopAt });
            }
            result["bench"] = rows;
        }

        if (outPath != null)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath))!);
            File.WriteAllText(outPath, JsonSerializer.Serialize(result, new JsonSerializerOptions { WriteIndented = true }));
        }
        return 0;
    }

    private sealed record LoggedOutput(string Line, int LineNumber, string Kind, int Tokens, string Finish, string Text);

    private sealed record Replay(int StopAt, int StopChar, int Period, int Repeats, string What, double NsPerToken);

    /// <summary>The guard after every token, on the output so far, as InferenceEngine.ApplyResults asks it.</summary>
    private static Replay Run(List<int> tokens, ITokenizer tokenizer)
    {
        var output = new List<int>(tokens.Count);
        long ticks = 0;
        for (int i = 0; i < tokens.Count; i++)
        {
            output.Add(tokens[i]);
            long t0 = Stopwatch.GetTimestamp();
            bool looping = RepetitionGuard.IsLooping(output, output.Count, out int period, out int repeats);
            ticks += Stopwatch.GetTimestamp() - t0;
            if (looping)
            {
                string what = RepetitionGuard.Describe(output, output.Count, period, repeats, ids => tokenizer.Decode(ids));
                int chars = tokenizer.Decode(output).Length;
                return new Replay(output.Count, chars, period, repeats, what, Ns(ticks, output.Count));
            }
        }
        return new Replay(-1, -1, 0, 0, string.Empty, Ns(ticks, Math.Max(1, tokens.Count)));
    }

    private static double Ns(long ticks, int tokens) => ticks * 1e9 / Stopwatch.Frequency / tokens;

    /// <summary>Every model output a TensorSharp .jsonl log recorded in full (chat.complete or chat.cancelled).</summary>
    private static IEnumerable<LoggedOutput> ReadOutputs(string log)
    {
        int number = 0;
        foreach (string line in File.ReadLines(log))
        {
            number++;
            if (!line.Contains("assistantOutput=", StringComparison.Ordinal))
                continue;
            string message;
            try
            {
                using JsonDocument doc = JsonDocument.Parse(line);
                message = doc.RootElement.GetProperty("message").GetString() ?? string.Empty;
            }
            catch (JsonException)
            {
                continue;
            }
            Match m = Output.Match(message);
            if (!m.Success)
                continue;
            string raw = m.Groups["out"].Value;
            // A capped log line ends in "...(+N chars)": not the whole output, so not replayable.
            if (Regex.IsMatch(raw, @"\.\.\.\(\+\d+ chars\)$"))
                continue;
            string text = raw.Replace("\\r", "\r", StringComparison.Ordinal)
                .Replace("\\n", "\n", StringComparison.Ordinal)
                .Replace("\\t", "\t", StringComparison.Ordinal);
            string finish = m.Groups["finish"].Success ? m.Groups["finish"].Value : m.Groups["kind"].Value;
            yield return new LoggedOutput(line, number, m.Groups["kind"].Value, int.Parse(m.Groups["tokens"].Value), finish, text);
        }
    }
}
