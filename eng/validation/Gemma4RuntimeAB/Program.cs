// Gemma 4 runtime-migration A/B tool: the legacy Gemma4Model and the shared-runtime Gemma4RuntimeModel in one binary.
//
//   --scenario greedy  --prompts <file, one user prompt a line> --steps N      greedy text, per-step top-5 log-probabilities
//   --scenario nll     --text <raw text> [--prefix 64] [--steps N]             teacher-forced NLL / perplexity (decode path)
//   --scenario bench   [--bench-prefill 512 --bench-decode 128 --bench-runs 3] the CLI's --benchmark procedure
//   --scenario parity  --prompts <file> --steps N                              both families in this process: the legacy argmax
//                                                                              teacher-forced through both, logits compared per step
// Common: --model <gguf> --backend ggml_cuda|cuda|ggml_cpu|cpu|... --family legacy|runtime [--tp N] [--layer-split N]
//         [--mmproj <file>] [--image <file>] [--out report.json]
// Reports are JSON (write them under the ignored artifacts/ or docs/validation/).
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

string Option(string name, string? fallback)
{
    int index = Array.IndexOf(args, name);
    if (index < 0) return fallback!;
    if (index + 1 >= args.Length) throw new ArgumentException($"Missing value for {name}.");
    return args[index + 1];
}

string modelPath = Option("--model", null);
string backendName = Option("--backend", "ggml_cuda");
string scenario = Option("--scenario", "greedy");
string family = Option("--family", "runtime");
string output = Option("--out", $"artifacts/gemma4-ab/{scenario}-{family}.json");
int steps = int.Parse(Option("--steps", "128"));
int tp = int.Parse(Option("--tp", "1"));
int layerSplit = int.Parse(Option("--layer-split", "1"));
string? promptsFile = Option("--prompts", null);
string? textFile = Option("--text", null);
int prefix = int.Parse(Option("--prefix", "64"));
int benchPrefill = int.Parse(Option("--bench-prefill", "512"));
int benchDecode = int.Parse(Option("--bench-decode", "128"));
int benchRuns = int.Parse(Option("--bench-runs", "3"));
bool refill = !args.Contains("--no-refill");
string? draftModel = Option("--draft-model", null);

if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath)) throw new ArgumentException("Pass --model <gemma4 gguf>.");
if (!Enum.TryParse(backendName.Replace("_", string.Empty), ignoreCase: true, out BackendType backend))
    throw new ArgumentException($"Unknown backend '{backendName}'.");
if (family is not ("legacy" or "runtime")) throw new ArgumentException("--family legacy|runtime");

ModelBase Load(string which)
{
    var sw = Stopwatch.StartNew();
    ModelBase m = which == "legacy"
        ? new Gemma4Model(modelPath, backend, tp)
        : Gemma4Architecture.CreateRuntimeModel(modelPath, backend, tp, null, layerSplit, draftModel);
    Console.WriteLine($"[ab] {which}: {m.GetType().Name} loaded in {sw.Elapsed.TotalMilliseconds:F0} ms (kv {m.KvCacheDtype})");
    return m;
}

int Argmax(float[] logits, int vocab)
{
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

(int Id, double LogProb)[] Top(float[] logits, int vocab, int k)
{
    double max = double.NegativeInfinity;
    for (int i = 0; i < vocab; i++) max = Math.Max(max, logits[i]);
    double sum = 0;
    for (int i = 0; i < vocab; i++) sum += Math.Exp(logits[i] - max);
    double lse = max + Math.Log(sum);
    return Enumerable.Range(0, vocab).OrderByDescending(i => logits[i]).Take(k).Select(i => (i, logits[i] - lse)).ToArray();
}

double LogProbAt(float[] logits, int vocab, int id)
{
    double max = double.NegativeInfinity;
    for (int i = 0; i < vocab; i++) max = Math.Max(max, logits[i]);
    double sum = 0;
    for (int i = 0; i < vocab; i++) sum += Math.Exp(logits[i] - max);
    return logits[id] - max - Math.Log(sum);
}

int[] Render(ModelBase m, string user)
{
    var messages = new List<ChatMessage> { new() { Role = "user", Content = user } };
    string rendered = ChatTemplate.RenderFromGgufTemplate(m.Config.ChatTemplate, messages, addGenerationPrompt: true, architecture: m.Config.Architecture);
    return m.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
}

float[] Prefill(ModelBase m, int[] prompt)
{
    m.ResetKVCache();
    return refill ? m.ForwardRefill(prompt) : m.Forward(prompt);
}

string[] Prompts() => File.ReadAllLines(promptsFile ?? throw new ArgumentException("--prompts <file>"))
    .Where(l => !string.IsNullOrWhiteSpace(l)).Select(l => l.Replace("\\n", "\n")).ToArray();

object report;
switch (scenario)
{
    case "greedy":
    {
        ModelBase m = Load(family);
        int vocab = m.Config.VocabSize;
        var results = new List<object>();
        foreach (string p in Prompts())
        {
            int[] prompt = Render(m, p);
            var sw = Stopwatch.StartNew();
            float[] logits = Prefill(m, prompt);
            double prefillMs = sw.Elapsed.TotalMilliseconds;
            var ids = new List<int>();
            var tops = new List<object>();
            sw.Restart();
            for (int i = 0; i < steps; i++)
            {
                var top = Top(logits, vocab, 5);
                tops.Add(top.Select(t => new { t.Id, t.LogProb }).ToArray());
                int next = top[0].Id;
                ids.Add(next);
                if (m.Tokenizer.IsEos(next) || i + 1 == steps) break;
                logits = m.Forward(new[] { next });
            }
            double decodeMs = sw.Elapsed.TotalMilliseconds;
            string text = m.Tokenizer.Decode(ids);
            Console.WriteLine($"[ab] prompt {results.Count}: {prompt.Length} tokens, prefill {prefillMs:F0} ms, {ids.Count} tokens " +
                $"{(ids.Count > 1 ? (ids.Count - 1) / (decodeMs / 1000.0) : 0):F1} tok/s: {text.Replace('\n', ' ')[..Math.Min(160, text.Length)]}");
            results.Add(new { Prompt = p, PromptTokens = prompt.Length, PrefillMs = prefillMs, DecodeMs = decodeMs, Tokens = ids, Text = text, Top = tops });
        }
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Tp = tp, Results = results };
        m.Dispose();
        break;
    }
    case "nll":
    {
        ModelBase m = Load(family);
        int vocab = m.Config.VocabSize;
        int[] all = m.Tokenizer.Encode(File.ReadAllText(textFile ?? throw new ArgumentException("--text <file>")), addSpecial: true).ToArray();
        int scored = Math.Min(steps <= 0 ? int.MaxValue : steps, all.Length - prefix);
        var sw = Stopwatch.StartNew();
        float[] logits = Prefill(m, all[..prefix]);
        double nll = 0;
        int top1 = 0;
        for (int i = 0; i < scored; i++)
        {
            int target = all[prefix + i];
            nll -= LogProbAt(logits, vocab, target);
            if (Argmax(logits, vocab) == target) top1++;
            if (i + 1 < scored) logits = m.Forward(new[] { target });
        }
        double ms = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[ab] nll {family}: {scored} tokens after {prefix}, mean NLL {nll / scored:F6}, ppl {Math.Exp(nll / scored):F5}, top-1 {top1 / (double)scored:P2}, {ms / 1000:F1} s");
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Prefix = prefix, Scored = scored,
            MeanNll = nll / scored, Perplexity = Math.Exp(nll / scored), Top1 = top1 / (double)scored };
        m.Dispose();
        break;
    }
    case "bench":
    {
        ModelBase m = Load(family);
        int vocab = m.Config.VocabSize;
        int span = Math.Max(1, Math.Min(17, vocab)), basis = Math.Max(0, Math.Min(100, vocab - span));
        int[] ids = Enumerable.Range(0, benchPrefill).Select(i => basis + i % span).ToArray();
        var runs = new List<object>();
        for (int run = 0; run < benchRuns; run++)
        {
            m.ResetKVCache();
            var sw = Stopwatch.StartNew();
            float[] logits = refill ? m.ForwardRefill(ids) : m.Forward(ids);
            double prefillMs = sw.Elapsed.TotalMilliseconds;
            int next = Argmax(logits, vocab);
            var produced = new int[benchDecode];
            sw.Restart();
            for (int i = 0; i < benchDecode; i++)
            {
                logits = m.Forward(new[] { next });
                produced[i] = next;
                next = Argmax(logits, vocab);
            }
            double decodeMs = sw.Elapsed.TotalMilliseconds;
            double pp = benchPrefill / (prefillMs / 1000.0), tg = benchDecode / (decodeMs / 1000.0);
            string recorded = m is StandardDecoderModel sd
                ? $"; recordings {sd.DecodeRecordings}, replays {sd.RecordedDecodeReplays}, sliding-window unrecorded {sd.SlidingWindowUnrecordedSteps}" : "";
            Console.WriteLine($"[ab] bench {family} run {run + 1}/{benchRuns}: prefill {benchPrefill} in {prefillMs:F0} ms ({pp:F1} tok/s), decode {benchDecode} in {decodeMs:F0} ms ({tg:F2} tok/s){recorded}");
            runs.Add(new { PrefillMs = prefillMs, PrefillTokPerSec = pp, DecodeMs = decodeMs, DecodeTokPerSec = tg, First = produced.Take(16).ToArray() });
        }
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Tp = tp, Prefill = benchPrefill, Decode = benchDecode, Runs = runs };
        m.Dispose();
        break;
    }
    case "parity":
    {
        ModelBase legacy = Load("legacy");
        ModelBase runtime = Load("runtime");
        int vocab = legacy.Config.VocabSize;
        var results = new List<object>();
        foreach (string p in Prompts())
        {
            int[] prompt = Render(legacy, p);
            float[] a = Prefill(legacy, prompt), b = Prefill(runtime, prompt);
            var rows = new List<object>();
            int agree = 0, n = 0;
            double worstCos = 1, worstRel = 0;
            for (int i = 0; i < steps; i++)
            {
                (double cos, double rel, double maxAbs) = Compare(a, b, vocab);
                int la = Argmax(a, vocab), lb = Argmax(b, vocab);
                var topA = Top(a, vocab, 5);
                double margin = topA[0].LogProb - topA[1].LogProb;
                n++;
                if (la == lb) agree++;
                worstCos = Math.Min(worstCos, cos);
                worstRel = Math.Max(worstRel, rel);
                rows.Add(new { Step = i, Cos = cos, Rel = rel, MaxAbs = maxAbs, Legacy = la, Runtime = lb, LegacyMargin = margin,
                    RuntimeLogProbOfLegacy = LogProbAt(b, vocab, la), LegacyLogProb = topA[0].LogProb });
                if (legacy.Tokenizer.IsEos(la) || i + 1 == steps) break;
                a = legacy.Forward(new[] { la });
                b = runtime.Forward(new[] { la });
            }
            Console.WriteLine($"[ab] parity prompt {results.Count}: {prompt.Length} tokens, {n} steps, argmax agree {agree}/{n}, worst cos {worstCos:F6}, worst rel L2 {worstRel:E3}");
            results.Add(new { Prompt = p, PromptTokens = prompt.Length, Agree = agree, Steps = n, WorstCos = worstCos, WorstRel = worstRel, Rows = rows });
        }
        report = new { Scenario = scenario, Model = modelPath, Backend = backend.ToString(), Results = results };
        legacy.Dispose();
        runtime.Dispose();
        break;
    }
    default:
        throw new ArgumentException($"Unknown scenario '{scenario}'.");
}

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output, JsonSerializer.Serialize(report));
Console.WriteLine($"[ab] wrote {output}");
if (backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlCpu)
    TensorSharp.GGML.GgmlBasicOps.Shutdown();

static (double Cos, double Rel, double MaxAbs) Compare(float[] a, float[] b, int vocab)
{
    double dot = 0, na = 0, nb = 0, diff = 0, maxAbs = 0;
    for (int i = 0; i < vocab; i++)
    {
        dot += (double)a[i] * b[i];
        na += (double)a[i] * a[i];
        nb += (double)b[i] * b[i];
        double d = a[i] - b[i];
        diff += d * d;
        maxAbs = Math.Max(maxAbs, Math.Abs(d));
    }
    return (dot / Math.Sqrt(na * nb), Math.Sqrt(diff / na), maxAbs);
}
