// GPT-OSS validation tool for the shared-runtime GptOssRuntimeModel (it held the runtime to the legacy GptOssModel on real
// checkpoints during the migration, before the legacy model was deleted: a parity scenario teacher-forced the legacy argmax
// through both families in one process).
//
//   --scenario greedy  --prompts <file, one user prompt a line> --steps N      greedy text, per-step top-5 log-probabilities
//   --scenario nll     --text <raw text> [--prefix 64] [--steps N]             teacher-forced NLL / perplexity (decode path)
//   --scenario bench   [--bench-prefill 512 --bench-decode 128 --bench-runs 3] the CLI's --benchmark procedure
//   --scenario spec    --prompts <file> --steps N [--drafter ngram|block] [--draft-model <DFlash gguf>] [--draft-window 3]
//                                                                              plain greedy against greedy speculation
//   --scenario verify  --prompts <file> --steps N [--draft-window 4]           verify windows against the plain steps
//   --scenario logits  --prompts <file> [--steps N]                            each prompt's last-row logits, then N greedy steps'
//                                                                              (raw F32, base64), to compare paths and backends offline
// Common: --model <gguf> --backend ggml_cuda|cuda|ggml_cpu|cpu|... [--tp N] [--layer-split N] [--out report.json]
// TS_N_CPU_MOE / TS_CPU_MOE and KV_CACHE_DTYPE apply as the CLI and server apply them.
// Reports are JSON (write them under the ignored artifacts/ or docs/validation/).
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

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
const string family = "runtime";
string output = Option("--out", $"artifacts/gptoss-ab/{scenario}-{backendName}.json");
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
int draftWindow = int.Parse(Option("--draft-window", "3"));
string drafter = Option("--drafter", "ngram");

if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath)) throw new ArgumentException("Pass --model <gpt-oss gguf>.");
if (!Enum.TryParse(backendName.Replace("_", string.Empty), ignoreCase: true, out BackendType backend))
    throw new ArgumentException($"Unknown backend '{backendName}'.");
MoeCpuOffloadConfig.ConfigureFromEnvironment();   // TS_N_CPU_MOE / TS_CPU_MOE, as the CLI and server apply them
KvCacheDtypeConfig.ConfigureFromEnvironment();   // KV_CACHE_DTYPE

ModelBase Load(string which)
{
    var sw = Stopwatch.StartNew();
    ModelBase m = GptOssArchitecture.CreateRuntimeModel(modelPath, backend, tp, null!, layerSplit, draftModel);
    Console.WriteLine($"[ab] {which}: {m.GetType().Name} loaded in {sw.Elapsed.TotalMilliseconds:F0} ms (kv {m.KvCacheDtype})");
    return m;
}

int Argmax(float[] logits, int vocab)
{
    int best = 0;
    for (int i = 1; i < vocab; i++) if (logits[i] > logits[best]) best = i;
    return best;
}

double RelL2(float[] a, float[] b)
{
    double num = 0, den = 0;
    for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return Math.Sqrt(num / den);
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
        var perToken = new double[scored];
        for (int i = 0; i < scored; i++)
        {
            int target = all[prefix + i];
            perToken[i] = -LogProbAt(logits, vocab, target);
            nll += perToken[i];
            if (Argmax(logits, vocab) == target) top1++;
            if (i + 1 < scored) logits = m.Forward(new[] { target });
        }
        double ms = sw.Elapsed.TotalMilliseconds;
        Console.WriteLine($"[ab] nll {family}: {scored} tokens after {prefix}, mean NLL {nll / scored:F6}, ppl {Math.Exp(nll / scored):F5}, top-1 {top1 / (double)scored:P2}, {ms / 1000:F1} s");
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Prefix = prefix, Scored = scored,
            MeanNll = nll / scored, Perplexity = Math.Exp(nll / scored), Top1 = top1 / (double)scored, PerTokenNll = perToken };
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
    case "logits":
    {
        ModelBase m = Load(family);
        int vocab = m.Config.VocabSize;
        var results = new List<object>();
        foreach (string p in Prompts())
        {
            int[] prompt = Render(m, p);
            float[] logits = Prefill(m, prompt);
            var rows = new List<string>();
            var ids = new List<int>();
            for (int i = 0; i <= Math.Max(0, steps); i++)
            {
                var bytes = new byte[vocab * 4];
                Buffer.BlockCopy(logits, 0, bytes, 0, bytes.Length);
                rows.Add(Convert.ToBase64String(bytes));
                if (i == steps) break;
                int next = Argmax(logits, vocab);
                ids.Add(next);
                logits = m.Forward(new[] { next });
            }
            Console.WriteLine($"[ab] logits {family} prompt {results.Count}: {prompt.Length} tokens, {rows.Count} rows");
            results.Add(new { Prompt = p, PromptTokens = prompt.Length, Tokens = ids, Rows = rows });
        }
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Vocab = vocab, Results = results };
        m.Dispose();
        break;
    }
    case "spec":
    {
        // Plain greedy against greedy speculation (the MTP head, --drafter head; a DFlash / DSpark drafter, --drafter block;
        // or n-gram drafts): the same tokens (a part only at a near-tie), the drafts' acceptance and the decode rate.
        ModelBase m = Load(family);
        if (drafter == "block" && draftModel == null) throw new ArgumentException("--drafter block needs --draft-model <DFlash gguf>.");
        var target = (ISpeculativeModel)m;
        if (target.SpeculationRefusal is { } refusal) throw new InvalidOperationException(refusal);
        int vocab = m.Config.VocabSize;
        var results = new List<object>();
        foreach (string p in Prompts())
        {
            int[] prompt = Render(m, p);
            float[] logits = Prefill(m, prompt);
            var plain = new List<int>();
            var margins = new List<float>();
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < steps; i++)
            {
                int next = Argmax(logits, vocab);
                plain.Add(next);
                float first = float.NegativeInfinity, second = float.NegativeInfinity;
                for (int v = 0; v < vocab; v++) { float x = logits[v]; if (x > first) { second = first; first = x; } else if (x > second) second = x; }
                margins.Add(first - second);
                if (m.Tokenizer.IsEos(next) || i + 1 == steps) break;
                logits = m.Forward(new[] { next });
            }
            double plainMs = sw.Elapsed.TotalMilliseconds;
            var head = (IDraftHead)m;
            ISpeculator speculator = drafter switch
            {
                "block" => new BlockDraftSpeculator(head, head.DraftBlockSize, Math.Min(draftWindow, head.DraftBlockSize)),
                _ => new NGramSpeculator(draftWindow),
            };
            var decoder = new SpeculativeDecoder(target, speculator);
            sw.Restart();
            List<int> ids = decoder.GenerateGreedy(prompt, steps, t => m.Tokenizer.IsEos(t));
            double specMs = sw.Elapsed.TotalMilliseconds - decoder.LastPrefillSeconds * 1000;
            int agree = 0;
            while (agree < Math.Min(plain.Count, ids.Count) && plain[agree] == ids[agree]) agree++;
            double acceptance = decoder.TokensDrafted > 0 ? decoder.TokensAccepted / (double)decoder.TokensDrafted : 0;
            double plainRate = (plain.Count - 1) / (plainMs / 1000.0), specRate = (ids.Count - 1) / (specMs / 1000.0);
            string parted = agree < Math.Min(plain.Count, ids.Count) ? $" (parted where plain's top two were {margins[agree]:F3} apart)" : "";
            Console.WriteLine($"[ab] spec {family} prompt {results.Count}: {prompt.Length} tokens; plain {plain.Count} at {plainRate:F1} tok/s, " +
                $"{drafter} x{draftWindow} {ids.Count} at {specRate:F1} tok/s ({specRate / plainRate:F2}x); agree {agree}/{Math.Min(plain.Count, ids.Count)}{parted}; " +
                $"drafted {decoder.TokensDrafted}, accepted {decoder.TokensAccepted} ({acceptance:P1}), verifies {decoder.VerifySteps}, plain steps {decoder.PlainSteps}");
            results.Add(new { Prompt = p, PromptTokens = prompt.Length, Plain = plain, Speculative = ids, Agree = agree, PlainMs = plainMs, SpecMs = specMs,
                decoder.TokensDrafted, decoder.TokensAccepted, decoder.VerifySteps, decoder.PlainSteps });
        }
        report = new { Scenario = scenario, Family = family, Model = modelPath, Backend = backend.ToString(), Drafter = drafter, Window = draftWindow, Results = results };
        m.Dispose();
        break;
    }
    case "verify":
    {
        // A verify window's rows against the plain steps they replace: plain greedy records each step's logits, then the same
        // tokens run as windows of --draft-window rows (SpecForward, all rows' logits), every window accepted whole, then with
        // every other window rewound by two rows and run again (a rejected draft's rollback).
        ModelBase m = Load(family);
        var target = (ISpeculativeTarget)m;
        int vocab = m.Config.VocabSize, w = draftWindow;
        foreach (string p in Prompts())
        {
            int[] prompt = textFile != null
                ? m.Tokenizer.Encode(File.ReadAllText(textFile), addSpecial: true).Take(prefix).ToArray()
                : Render(m, p);
            float[] logits = Prefill(m, prompt);
            var plainLogits = new List<float[]> { (float[])logits.Clone() };
            var tokens = new List<int>();
            for (int i = 0; i < steps; i++)
            {
                int next = Argmax(logits, vocab);
                tokens.Add(next);
                logits = m.Forward(new[] { next });
                plainLogits.Add((float[])logits.Clone());
            }
            {
                // A verify window (all rows' logits) then plain steps: each plain step's logits against the plain run's.
                Prefill(m, prompt);
                var rowsOut = new float[w * vocab];
                target.SpecForward(tokens.GetRange(0, w).ToArray(), null, rowsOut, allLogitsRows: true);
                target.SpecOnVerifyAccepted(w - 1, w - 1);
                var parts = new List<string>();
                for (int j = 0; j < w; j++)
                    parts.Add($"v{j}:{RelL2(rowsOut.AsSpan(j * vocab, vocab).ToArray(), plainLogits[j + 1]):E1}");
                for (int i = w; i < Math.Min(tokens.Count, w + 6); i++)
                {
                    float[] after = m.Forward(new[] { tokens[i] });
                    parts.Add($"d{i}:{RelL2(after, plainLogits[i + 1]):E1}");
                }
                Console.WriteLine($"[ab] verify-then-decode {family} prompt {prompt.Length} tokens, window {w}: {string.Join(" ", parts)}");
            }
            foreach (bool rewinds in new[] { false, true })
            {
                Prefill(m, prompt);
                var outRows = new float[w * vocab];
                double worst = 0;
                int agree = 0, rows = 0, window = 0;
                for (int i = 0; i + w <= tokens.Count; i += w, window++)
                {
                    int[] chunk = tokens.GetRange(i, w).ToArray();
                    if (rewinds && window % 2 == 1)
                    {
                        // A rejected draft as the speculative decoder settles it: the window keeps its first two rows (the
                        // recurrent states take the state after them), the cache drops the rest, and the rest runs again.
                        int start = target.CacheSeqLen;
                        target.SpecForward(chunk, null, outRows, allLogitsRows: true);
                        target.SpecOnVerifyAccepted(1, w);
                        target.SpecRewindCache(start + 2);
                        var tail = new float[(w - 2) * vocab];
                        target.SpecForward(chunk[2..], null, tail, allLogitsRows: true);
                        target.SpecOnVerifyAccepted(w - 3, w - 2);
                        Array.Copy(tail, 0, outRows, 2 * vocab, tail.Length);
                    }
                    else
                    {
                        target.SpecForward(chunk, null, outRows, allLogitsRows: true);
                        target.SpecOnVerifyAccepted(w - 1, w);
                    }
                    for (int j = 0; j < w; j++)
                    {
                        float[] plain = plainLogits[i + j + 1];
                        var row = outRows.AsSpan(j * vocab, vocab).ToArray();
                        double num = 0, den = 0;
                        for (int v = 0; v < vocab; v++) { double d = row[v] - plain[v]; num += d * d; den += (double)plain[v] * plain[v]; }
                        worst = Math.Max(worst, Math.Sqrt(num / den));
                        if (Argmax(row, vocab) == Argmax(plain, vocab)) agree++;
                        rows++;
                    }
                }
                Console.WriteLine($"[ab] verify {family} prompt {prompt.Length} tokens, windows of {w}{(rewinds ? " with rewinds" : "")}: argmax {agree}/{rows}, worst rel L2 {worst:E3}");
            }
        }
        report = new { Scenario = scenario };
        m.Dispose();
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
