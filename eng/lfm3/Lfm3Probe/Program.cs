// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3 validation probe (TensorSharp side of eng/lfm3's llama.cpp comparisons; outputs belong in ignored artifacts/):
//   ppl    MODEL BACKEND FILE CTX CHUNKS [--per-op]
//                                          perplexity as llama-perplexity computes it: the file's tokens (no BOS, special
//                                          tokens not parsed) in CHUNKS chunks of CTX, each from an empty cache, scoring the
//                                          second half of every chunk (row i predicts token i + 1); --per-op runs the shared
//                                          runtime's per-op path instead of whole-forward graphs
//   tokens MODEL FILE OUT                  the file's token ids as llama-perplexity tokenizes it (one id a line)
//   bench  MODEL BACKEND [PP,...|-] [TG] [DEPTH,...] [--reps N] [--greedy]
//                                          prefill tokens/s of each PP-token prompt and decode tokens/s of TG tokens after
//                                          DEPTH tokens (llama-bench's pp / tg / tg@d), each the median of N runs (3), with
//                                          the decode steps' graph counters (runs, advances in place, builds). As llama-bench,
//                                          decode feeds pseudo-random tokens (the timed loop holds no sampling); --greedy feeds
//                                          each step's host argmax instead. A "RESULT name median min max" line per test.
//   spec   MODEL BACKEND FILE N            N greedy tokens after the file's prompt, plainly and with n-gram speculation (a verify
//                                          window keeps the short-conv inputs and attends under the sliding window; the accept
//                                          count commits a prefix): the two streams must be equal
// BACKEND: ggml_metal | ggml_cpu | cpu | mlx | ggml_cuda | cuda
using System.Diagnostics;
using System.Globalization;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

if (args.Length < 2) return Usage();
BackendType Backend(string name) => name switch
{
    "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "mlx" => BackendType.Mlx,
    "ggml_cuda" => BackendType.GgmlCuda, "cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("unknown backend " + name),
};
KvCacheDtypeConfig.ConfigureFromEnvironment();
switch (args[0])
{
    case "ppl" when args.Length >= 6:
        return Perplexity(args[1], Backend(args[2]), args[3], int.Parse(args[4]), int.Parse(args[5]), args.Contains("--per-op"));
    case "tokens" when args.Length >= 4:
    {
        using var gguf = GgufFile.OpenWithoutSiblingShards(args[1]);
        var tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        File.WriteAllLines(args[3], tokenizer.Encode(File.ReadAllText(args[2]), addSpecial: false).Select(t => t.ToString(CultureInfo.InvariantCulture)));
        return 0;
    }
    case "spec" when args.Length >= 5:
        return Speculation(args[1], Backend(args[2]), args[3], int.Parse(args[4]));
    case "bench" when args.Length >= 3:
    {
        string[] positional = args.Where((a, i) => !a.StartsWith("--", StringComparison.Ordinal) && (i == 0 || args[i - 1] != "--reps")).ToArray();
        int reps = Array.IndexOf(args, "--reps") is int r && r >= 0 && r + 1 < args.Length ? int.Parse(args[r + 1]) : 3;
        return Bench(positional[1], Backend(positional[2]),
            positional.Length > 3 && positional[3] != "-" ? positional[3].Split(',').Select(int.Parse).ToArray() : Array.Empty<int>(),
            positional.Length > 4 ? int.Parse(positional[4]) : 128,
            positional.Length > 5 ? positional[5].Split(',').Select(int.Parse).ToArray() : new[] { 0 },
            reps, args.Contains("--greedy"));
    }
    default:
        return Usage();
}

static int Usage()
{
    Console.Error.WriteLine("usage: Lfm3Probe ppl MODEL BACKEND FILE CTX CHUNKS [--per-op] | spec MODEL BACKEND FILE N | tokens MODEL FILE OUT | bench MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy]");
    return 2;
}

static int Speculation(string path, BackendType backend, string file, int count)
{
    using ModelBase model = ModelBase.Create(path, backend);
    int[] prompt = model.Tokenizer.Encode(File.ReadAllText(file), addSpecial: false).ToArray();
    int vocab = model.Config.VocabSize;
    int ArgMax(float[] l) { int b = 0; for (int v = 1; v < vocab; v++) if (l[v] > l[b]) b = v; return b; }
    var plain = new List<int>();
    var sw = Stopwatch.StartNew();
    model.ResetKVCache();
    float[] logits = model.ForwardRefill(prompt);
    for (int i = 0; i < count; i++)
    {
        plain.Add(ArgMax(logits));
        if (i + 1 < count) logits = model.Forward(new[] { plain[^1] });
    }
    double plainSeconds = sw.Elapsed.TotalSeconds;
    model.ResetKVCache();
    var decoder = new SpeculativeDecoder((ISpeculativeModel)model, new SpeculationOptions
    {
        Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3, MinDraftProb = 0,
    }) { AdaptiveSpeculation = false };
    sw.Restart();
    List<int> drafted = decoder.GenerateGreedy(prompt, count);
    double specSeconds = sw.Elapsed.TotalSeconds;
    int same = 0;
    while (same < Math.Min(plain.Count, drafted.Count) && plain[same] == drafted[same]) same++;
    Console.WriteLine($"prompt={prompt.Length} tokens; plain {plain.Count} in {plainSeconds:F1} s, speculative {drafted.Count} in {specSeconds:F1} s");
    Console.WriteLine($"verify steps {decoder.VerifySteps}, drafted {decoder.TokensDrafted}, accepted {decoder.TokensAccepted}, plain steps {decoder.PlainSteps}");
    Console.WriteLine(same == plain.Count && drafted.Count == plain.Count ? "streams equal" : $"streams part at token {same}");
    return same == plain.Count && drafted.Count == plain.Count ? 0 : 1;
}

static int Perplexity(string path, BackendType backend, string file, int ctx, int chunks, bool perOp)
{
    using ModelBase model = ModelBase.Create(path, backend);
    if (perOp && model is StandardDecoderModel runtime) runtime.GraphLoweringEnabled = false;
    var spec = (ISpeculativeModel)model;
    int[] tokens = model.Tokenizer.Encode(File.ReadAllText(file), addSpecial: false).ToArray();
    int available = tokens.Length / ctx;
    if (chunks <= 0 || chunks > available) chunks = available;
    Console.WriteLine($"tokens={tokens.Length} ctx={ctx} chunks={chunks} backend={backend}{(perOp ? " per-op" : "")}");
    int vocab = model.Config.VocabSize, first = ctx / 2;
    const int block = 128;
    var logits = new float[block * vocab];
    double nll = 0;
    long count = 0;
    var timer = Stopwatch.StartNew();
    for (int c = 0; c < chunks; c++)
    {
        int start = c * ctx;
        model.ResetKVCache();
        model.ForwardRefill(tokens.AsSpan(start, first).ToArray());
        // Rows first .. ctx - 2 predict tokens first + 1 .. ctx - 1 (llama-perplexity's n_ctx - 1 - first rows).
        for (int at = first; at < ctx - 1; at += block)
        {
            int rows = Math.Min(block, ctx - at);
            spec.SpecForward(tokens.AsSpan(start + at, rows).ToArray(), null, logits, allLogitsRows: true);
            for (int r = 0; r < rows && at + r < ctx - 1; r++)
            {
                ReadOnlySpan<float> row = logits.AsSpan(r * vocab, vocab);
                float max = float.NegativeInfinity;
                foreach (float v in row) max = Math.Max(max, v);
                double sum = 0;
                foreach (float v in row) sum += Math.Exp(v - max);
                nll += -(row[tokens[start + at + r + 1]] - max - Math.Log(sum));
                count++;
            }
        }
        Console.Write($"[{c + 1}]{Math.Exp(nll / count):F4},");
    }
    Console.WriteLine();
    Console.WriteLine($"Final estimate: PPL = {Math.Exp(nll / count):F4} over {count} tokens ({timer.Elapsed.TotalSeconds:F1} s)");
    return 0;
}

static int Bench(string path, BackendType backend, int[] prompts, int generate, int[] depths, int reps, bool greedy)
{
    using ModelBase model = ModelBase.Create(path, backend);
    var runtime = model as StandardDecoderModel;
    int vocab = model.Config.VocabSize;
    int[] Tokens(int n, int seed) => Enumerable.Range(0, n).Select(i => 1000 + (int)((uint)(i * 2654435761u + seed) % 60000u)).ToArray();
    static double Median(List<double> v) { v.Sort(); return v[v.Count / 2]; }

    // Warm: kernels load and graphs build once.
    model.ResetKVCache();
    model.ForwardRefill(Tokens(256, 1));
    for (int i = 0; i < 8; i++) model.Forward(new[] { 1234 + i });

    foreach (int n in prompts)
    {
        var rates = new List<double>();
        for (int rep = 0; rep < reps; rep++)
        {
            model.ResetKVCache();
            int[] t = Tokens(n, rep + 7);
            var sw = Stopwatch.StartNew();
            model.ForwardRefill(t);
            rates.Add(n / sw.Elapsed.TotalSeconds);
        }
        Console.WriteLine($"pp{n}: {Median(rates):F1} tok/s  (runs {string.Join(", ", rates.Select(r => r.ToString("F0")))})");
        Console.WriteLine(FormattableString.Invariant($"RESULT pp{n} {Median(rates):F2} {rates.Min():F2} {rates.Max():F2}"));
    }
    foreach (int depth in generate > 0 ? depths : Array.Empty<int>())
    {
        var rates = new List<double>();
        int runs0 = runtime?.GraphRuns ?? 0, advances0 = runtime?.DecodeAdvances ?? 0, builds0 = runtime?.GraphBuilds ?? 0;
        for (int rep = 0; rep < reps; rep++)
        {
            model.ResetKVCache();
            float[] logits = model.ForwardRefill(depth > 0 ? Tokens(depth, rep + 3) : new[] { 1 });
            int[] stream = Tokens(generate, rep + 11);
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < generate; i++)
            {
                int next = stream[i];
                if (greedy)
                {
                    next = 0;
                    for (int v = 1; v < vocab; v++) if (logits[v] > logits[next]) next = v;
                }
                logits = model.Forward(new[] { next });
            }
            rates.Add(generate / sw.Elapsed.TotalSeconds);
        }
        string counters = runtime == null ? "" :
            $"  graph runs +{runtime.GraphRuns - runs0}, decode advances +{runtime.DecodeAdvances - advances0}, graph builds +{runtime.GraphBuilds - builds0}";
        Console.WriteLine($"tg{generate}@d{depth}: {Median(rates):F1} tok/s  (runs {string.Join(", ", rates.Select(r => r.ToString("F1")))}){counters}");
        Console.WriteLine(FormattableString.Invariant($"RESULT tg{generate}@d{depth} {Median(rates):F2} {rates.Min():F2} {rates.Max():F2}"));
    }
    if (runtime != null)
    {
        (long perToken, long rings) = runtime.KvCacheFootprint();
        Console.WriteLine($"kv cache: {perToken} B/token (layers that keep every position) + {rings / 1048576.0:F1} MB of sliding-window rings a sequence");
    }
    return 0;
}
