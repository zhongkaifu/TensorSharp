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
//   bench  MODEL BACKEND [PP,...] [TG] [DEPTH,...]
//                                          prefill tokens/s of each PP-token prompt and decode tokens/s of TG tokens after
//                                          DEPTH tokens (llama-bench's pp / tg / tg@d), each the median of three runs, with
//                                          the decode steps' graph counters (runs, advances in place, builds)
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
    case "bench" when args.Length >= 3:
        return Bench(args[1], Backend(args[2]),
            args.Length > 3 ? args[3].Split(',').Select(int.Parse).ToArray() : new[] { 512, 4096 },
            args.Length > 4 ? int.Parse(args[4]) : 128,
            args.Length > 5 ? args[5].Split(',').Select(int.Parse).ToArray() : new[] { 0 });
    default:
        return Usage();
}

static int Usage()
{
    Console.Error.WriteLine("usage: Lfm3Probe ppl MODEL BACKEND FILE CTX CHUNKS [--per-op] | tokens MODEL FILE OUT | bench MODEL BACKEND [PP,..] [TG] [DEPTH,..]");
    return 2;
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

static int Bench(string path, BackendType backend, int[] prompts, int generate, int[] depths)
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
        for (int rep = 0; rep < 3; rep++)
        {
            model.ResetKVCache();
            int[] t = Tokens(n, rep + 7);
            var sw = Stopwatch.StartNew();
            model.ForwardRefill(t);
            rates.Add(n / sw.Elapsed.TotalSeconds);
        }
        Console.WriteLine($"pp{n}: {Median(rates):F1} tok/s  (runs {string.Join(", ", rates.Select(r => r.ToString("F0")))})");
    }
    foreach (int depth in depths)
    {
        var rates = new List<double>();
        int runs0 = runtime?.GraphRuns ?? 0, advances0 = runtime?.DecodeAdvances ?? 0, builds0 = runtime?.GraphBuilds ?? 0;
        for (int rep = 0; rep < 3; rep++)
        {
            model.ResetKVCache();
            float[] logits = model.ForwardRefill(depth > 0 ? Tokens(depth, rep + 3) : new[] { 1 });
            var sw = Stopwatch.StartNew();
            int next = 1000;
            for (int i = 0; i < generate; i++)
            {
                int best = 0;
                for (int v = 1; v < vocab; v++) if (logits[v] > logits[best]) best = v;
                next = best;
                logits = model.Forward(new[] { next });
            }
            rates.Add(generate / sw.Elapsed.TotalSeconds);
        }
        string counters = runtime == null ? "" :
            $"  graph runs +{runtime.GraphRuns - runs0}, decode advances +{runtime.DecodeAdvances - advances0}, graph builds +{runtime.GraphBuilds - builds0}";
        Console.WriteLine($"tg{generate}@d{depth}: {Median(rates):F1} tok/s  (runs {string.Join(", ", rates.Select(r => r.ToString("F1")))}){counters}");
    }
    return 0;
}
