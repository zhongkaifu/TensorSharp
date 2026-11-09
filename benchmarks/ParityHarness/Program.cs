// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Cross-engine parity + throughput harness. Feeds RAW token ids (so the model
// is isolated from tokenizer/template differences) and compares against
// llama.cpp greedy goldens, measures llama-bench-shaped throughput, and checks
// that batched (continuous-batching) decode reproduces serial decode.
//
// Modes:
//   parity <model.gguf> --ref <golden.json> [backend] [max_new]
//   parity <model.gguf> --bench <backend> <pp1,pp2,...> <tg> [reps] [depth]
//       emits JSON [bench-sample]/[bench-statistics] records and the historic
//       best-rate [bench] lines. Warmup and optional context depth are untimed.
//   parity <model.gguf> --batched <backend> <steps> <promptA> <promptB> [...]
//       prompts are comma-separated token ids; each runs on its own sequence
//       slot serially first, then all together through the fused batched
//       decode; the two must agree token for token.
//   parity <model.gguf> --arena-zero-reuse <backend> [cycles] [dirty_steps]
//                                            [probe_token] [survivor_token]
//       repeatedly replaces one lane of a live two-sequence arena batch with a
//       fresh position-zero holder. Its first logits must match a clean solo
//       position-zero reference even though the physical arena slot is dirty.
//   parity <model.gguf> --retained-continuation <backend> [round1_steps]
//                         [follow_steps] [promptA] [promptB] [suffix]
//       runs two concurrent conversations and then two concurrent exact-prefix
//       follow-ups. Each retained follow-up must report full-prefix reuse and
//       reproduce a retention-disabled full-prefill run token for token. Optional
//       prompts/suffix are comma-separated token ids; text defaults are provided.
//   parity <model.gguf> <tok0,tok1,...> [n_predict] [backend]   raw greedy
//   parity <model.gguf> --raw-step <tok0,tok1,...> [backend]
//       raw logits with the prompt fed one token at a time (decode path)
//   parity <model.gguf> --ppl <text-file> [backend] [n_ctx] [max_chunks] [prefill]
//       teacher-forced perplexity over non-overlapping n_ctx windows,
//       scoring the SECOND half of each window (llama.cpp's
//       `llama-perplexity` protocol: first = n_ctx/2), so the numbers are
//       directly comparable to its "Final estimate: PPL = ..." line.
//       `prefill` runs each window's unscored first half as one batched
//       forward, so the scored tokens attend over a cache the prefill
//       kernels wrote.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

public static class Program
{
    private sealed class RefRecord
    {
        public string prompt { get; set; }
        public int[] prompt_tokens { get; set; }
        public int[] generated_tokens { get; set; }
    }

    private static BackendType ResolveBackend(string s) => s switch
    {
        "ggmlcuda" or "ggml_cuda" => BackendType.GgmlCuda,
        "ggmlmetal" or "ggml_metal" => BackendType.GgmlMetal,
        "ggmlcpu" or "ggml_cpu" => BackendType.GgmlCpu,
        "ggmlvulkan" or "ggml_vulkan" => BackendType.GgmlVulkan,
        "cuda" => BackendType.Cuda,
        _ => BackendType.Cpu,
    };

    private static int ResolveTp()
    {
        string raw = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE");
        return int.TryParse(raw, out int v) && v > 1 ? v : 1;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    public static int Main(string[] args)
    {
        // Match CLI/server startup so raw parity and benchmark runs can exercise
        // the exact cache dtype selected by KV_CACHE_DTYPE (not just the model's
        // automatic F16 default).
        KvCacheDtypeConfig.ConfigureFromEnvironment();

        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: parity <model.gguf> --ref|--bench|--batched|<tokens> ...");
            return 1;
        }

        string modelPath = args[0];
        // The same expert-placement switches the CLI reads (TS_CPU_MOE / TS_N_CPU_MOE),
        // so a model that only fits with its routed experts on the host - DeepSeek V4.1
        // on a Mac - can be measured here too.
        MoeCpuOffloadConfig.ConfigureFromEnvironment();
        // A model the backend refuses to load is an ANSWER, not a crash: the CLI and the
        // server both print one stderr line and exit with HostExitCodes.ModelLoadRefused,
        // and a validation run asking "does ggml_cuda-only Bonsai2 refuse the pure-C# cpu
        // backend cleanly?" reads this harness's exit code. Without this it dumped a stack
        // and aborted (SIGABRT, exit 134), which is indistinguishable from a real crash.
        try
        {
            return Dispatch(modelPath, args);
        }
        catch (Exception ex) when (ModelLoadRefusal.TryDescribe(ex, out string refusal))
        {
            Console.Error.WriteLine(ModelLoadRefusal.FormatErrorLine(refusal));
            return HostExitCodes.ModelLoadRefused;
        }
    }

    private static int Dispatch(string modelPath, string[] args)
    {
        if (args[1] == "--ref") return RunReference(modelPath, args);
        if (args[1] == "--bench") return RunBench(modelPath, args);
        if (args[1] == "--batched") return RunBatched(modelPath, args);
        if (args[1] == "--arena-zero-reuse") return RunArenaZeroReuse(modelPath, args);
        if (args[1] == "--retained-continuation")
            return RunRetainedContinuation(modelPath, args).GetAwaiter().GetResult();
        if (args[1] == "--ppl") return RunPerplexity(modelPath, args);
        if (args[1] == "--raw-step") return RunRawStepped(modelPath, args);
        return RunRaw(modelPath, args);
    }

    /// <summary>
    /// Teacher-forced perplexity, mirroring llama.cpp's `llama-perplexity`
    /// protocol: the text is tokenized once, split into non-overlapping n_ctx
    /// windows, and only the SECOND half of each window is scored (llama.cpp
    /// uses `first = n_ctx/2`) so every scored token has at least n_ctx/2 tokens
    /// of context. The window is fed one token at a time, which is what makes
    /// each position's logits available; that also means this measures the
    /// DECODE kernels, whereas llama.cpp scores a batched prefill.
    /// </summary>
    private static int RunPerplexity(string modelPath, string[] args)
    {
        string textPath = args[2];
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");
        int nCtx = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 512;
        int maxChunks = args.Length > 5 ? int.Parse(args[5], CultureInfo.InvariantCulture) : int.MaxValue;
        bool prefill = args.Length > 6 && args[6] == "prefill";

        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[ppl] model loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");

        List<int> ids = model.Tokenizer.Encode(File.ReadAllText(textPath), addSpecial: true);
        int chunks = Math.Min(maxChunks, ids.Count / nCtx);
        if (chunks <= 0)
        {
            Console.Error.WriteLine($"[ppl] text has {ids.Count} tokens, need at least n_ctx={nCtx}");
            return 1;
        }
        int first = nCtx / 2;
        Console.WriteLine($"[ppl] {ids.Count} tokens, n_ctx={nCtx}, scoring tokens [{first},{nCtx}) of {chunks} chunks" +
            (prefill ? ", context prefilled in one forward" : ""));

        double nllSum = 0.0;
        long scored = 0;
        var swAll = Stopwatch.StartNew();
        for (int c = 0; c < chunks; c++)
        {
            model.ResetKVCache();
            int baseIdx = c * nCtx;
            float[] logits = null;
            int start = 0;
            if (prefill && first > 1)
            {
                // Positions [0, first-1) are context only: their logits are never read.
                model.Forward(ids.GetRange(baseIdx, first - 1).ToArray());
                start = first - 1;
            }
            for (int i = start; i < nCtx - 1; i++)
            {
                logits = model.Forward(new[] { ids[baseIdx + i] });
                if (i + 1 < first)
                    continue;   // context-only positions are not scored

                int target = ids[baseIdx + i + 1];
                // log_softmax(logits)[target], computed in the numerically stable way.
                float max = float.NegativeInfinity;
                for (int v = 0; v < logits.Length; v++)
                    if (logits[v] > max) max = logits[v];
                double sumExp = 0.0;
                for (int v = 0; v < logits.Length; v++)
                    sumExp += Math.Exp(logits[v] - max);
                nllSum += -(logits[target] - max - Math.Log(sumExp));
                scored++;
            }
            double running = Math.Exp(nllSum / Math.Max(1, scored));
            Console.WriteLine($"[ppl] chunk {c + 1}/{chunks}  scored={scored}  running PPL = {running:F4}  ({swAll.Elapsed.TotalSeconds:F0}s)");
        }

        double ppl = Math.Exp(nllSum / Math.Max(1, scored));
        Console.WriteLine($"[ppl] Final estimate: PPL = {ppl:F4}  over {scored} tokens in {chunks} chunks of {nCtx}");
        return 0;
    }

    /// <summary>
    /// Same output as the raw mode, but the prompt is fed ONE TOKEN AT A TIME
    /// (decode kernels) instead of as a single batched prefill. Comparing the two
    /// isolates the engine's own prefill-vs-decode numerical difference from any
    /// difference against another engine: on Blackwell the batched path can
    /// quantize activations to FP4 while the single-token path stays exact.
    /// </summary>
    private static int RunRawStepped(string modelPath, string[] args)
    {
        int[] tokens = args[2].Split(',', StringSplitOptions.RemoveEmptyEntries)
                              .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture)).ToArray();
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        model.ResetKVCache();
        float[] logits = null;
        foreach (int tok in tokens)
            logits = model.Forward(new[] { tok });

        Console.WriteLine($"n_vocab {logits.Length}");
        Console.WriteLine("logits " + string.Join(' ', logits.Select(v => v.ToString("F6", CultureInfo.InvariantCulture))));
        return 0;
    }

    private static int RunRaw(string modelPath, string[] args)
    {
        int[] tokens = args[1].Split(',', StringSplitOptions.RemoveEmptyEntries)
                              .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture)).ToArray();
        int nPredict = args.Length > 2 ? int.Parse(args[2]) : 0;
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        float[] logits = model.Forward(tokens);
        Console.WriteLine($"n_vocab {logits.Length}");
        Console.WriteLine("logits " + string.Join(' ', logits.Select(v => v.ToString("F6", CultureInfo.InvariantCulture))));
        if (nPredict > 0)
        {
            var gen = new List<int>();
            int tok = ArgMax(logits);
            gen.Add(tok);
            for (int i = 1; i < nPredict; i++)
            {
                logits = model.Forward(new[] { tok });
                tok = ArgMax(logits);
                gen.Add(tok);
            }
            Console.WriteLine("generated " + string.Join(' ', gen));
        }
        return 0;
    }

    private static int RunReference(string modelPath, string[] args)
    {
        string refPath = args[2];
        BackendType backend = ResolveBackend(args.Length > 3 ? args[3] : "ggmlcuda");
        int maxNew = args.Length > 4 ? int.Parse(args[4]) : int.MaxValue;

        var records = JsonSerializer.Deserialize<List<RefRecord>>(File.ReadAllText(refPath));
        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[parity] model loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");

        int total = 0, matched = 0;
        foreach (var rec in records)
        {
            if (rec.prompt_tokens == null || rec.prompt_tokens.Length == 0) continue;
            int want = Math.Min(rec.generated_tokens.Length, maxNew);

            model.ResetKVCache();
            var swPrefill = Stopwatch.StartNew();
            float[] logits = model.ForwardRefill(rec.prompt_tokens);
            swPrefill.Stop();

            var produced = new int[want];
            var swDecode = Stopwatch.StartNew();
            for (int i = 0; i < want; i++)
            {
                produced[i] = ArgMax(logits);
                if (i + 1 < want)
                    logits = model.Forward(new[] { produced[i] });
            }
            swDecode.Stop();

            int firstDiff = -1;
            for (int i = 0; i < want; i++)
                if (produced[i] != rec.generated_tokens[i]) { firstDiff = i; break; }

            total++;
            string label = rec.prompt != null && rec.prompt.Length > 44 ? rec.prompt.Substring(0, 44) + "..." : rec.prompt;
            double pp = rec.prompt_tokens.Length / Math.Max(1e-9, swPrefill.Elapsed.TotalSeconds);
            double tg = Math.Max(0, want - 1) / Math.Max(1e-9, swDecode.Elapsed.TotalSeconds);
            if (firstDiff < 0)
            {
                matched++;
                Console.WriteLine($"[MATCH ] {label}  ({want} tokens)  prefill {pp:F1} tok/s  decode {tg:F2} tok/s");
            }
            else
            {
                Console.WriteLine($"[DIFF  ] {label}  diverges at {firstDiff}/{want}  prefill {pp:F1} tok/s  decode {tg:F2} tok/s");
                Console.WriteLine($"          ref: {string.Join(' ', rec.generated_tokens.Take(want))}");
                Console.WriteLine($"          ts : {string.Join(' ', produced)}");
            }
        }
        Console.WriteLine($"[parity] {matched}/{total} prompts reproduce llama.cpp token-for-token");
        return matched == total ? 0 : 2;
    }

    /// <summary>llama-bench-shaped throughput with every measured sample retained.
    /// Uses fixed decode inputs without sampling, untimed warmup, and an optional
    /// pre-existing context. Historic [bench] output still reports the best rate.</summary>
    private static int RunBench(string modelPath, string[] args)
    {
        if (args.Length < 4)
        {
            Console.Error.WriteLine("usage: parity <model.gguf> --bench <backend> <pp1,pp2,...> <tg> [reps] [depth]");
            return 1;
        }
        BackendType backend = ResolveBackend(args[2]);
        int[] ppLens = args[3].Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
        int tg = args.Length > 4 ? int.Parse(args[4]) : 64;
        int reps = args.Length > 5 ? int.Parse(args[5]) : 2;
        int depth = args.Length > 6 ? int.Parse(args[6]) : 0;
        if (ppLens.Length == 0 || ppLens.Any(pp => pp <= 0) || tg < 0 || reps <= 0 || depth < 0)
        {
            Console.Error.WriteLine("prompt lengths and repetitions must be positive; generation length and depth must be nonnegative");
            return 1;
        }

        var sw = Stopwatch.StartNew();
        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        Console.WriteLine($"[bench] loaded in {sw.Elapsed.TotalSeconds:F1}s, backend={backend}, arch={model.Config.Architecture}");
        Console.WriteLine($"[bench] depth={depth} repeats={reps} warmup=untimed decode=fixed-inference");

        var rng = new Random(42);
        int vocab = Math.Max(1000, model.Config.VocabSize - 1000);
        var depthPrompt = new int[depth];
        for (int i = 0; i < depth; i++) depthPrompt[i] = 1000 + rng.Next(vocab - 1000);

        void ResetContext()
        {
            model.ResetKVCache();
            // This setup is outside every measured interval. Unlike llama-bench,
            // which may restore a cached state, we recompute the depth prefix.
            if (depthPrompt.Length > 0)
                model.ForwardRefill(depthPrompt);
        }

        void RecordSample(List<double> rates, int promptTokens, int generationTokens, int run, double seconds)
        {
            double rate = (promptTokens + generationTokens) / seconds;
            rates.Add(rate);
            Console.WriteLine("[bench-sample] " + JsonSerializer.Serialize(new
            {
                n_prompt = promptTokens, n_gen = generationTokens, n_depth = depth,
                run = run + 1, elapsed_seconds = seconds, tokens_per_second = rate,
            }));
        }

        void PrintStatistics(List<double> rates, int promptTokens, int generationTokens)
        {
            double average = rates.Average();
            double[] ordered = rates.OrderBy(rate => rate).ToArray();
            int middle = ordered.Length / 2;
            double median = ordered.Length % 2 == 0 ? (ordered[middle - 1] + ordered[middle]) / 2 : ordered[middle];
            double standardDeviation = rates.Count > 1
                ? Math.Sqrt(rates.Sum(rate => (rate - average) * (rate - average)) / (rates.Count - 1)) : 0;
            Console.WriteLine("[bench-statistics] " + JsonSerializer.Serialize(new
            {
                n_prompt = promptTokens, n_gen = generationTokens, n_depth = depth,
                repetitions = rates.Count, avg_ts = average, stddev_ts = standardDeviation,
                median_ts = median, min_ts = ordered[0], max_ts = ordered[^1], samples_ts = rates,
            }));
        }

        foreach (int pp in ppLens)
        {
            var rates = new List<double>(reps);
            var prompt = new int[pp];
            for (int i = 0; i < pp; i++) prompt[i] = 1000 + rng.Next(vocab - 1000);
            // Match llama-bench's full prompt warmup before measured repetitions.
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            for (int r = 0; r < reps; r++)
            {
                ResetContext();
                var t = Stopwatch.StartNew();
                model.ForwardRefill(prompt);
                t.Stop();
                RecordSample(rates, pp, 0, r, t.Elapsed.TotalSeconds);
            }
            Console.WriteLine($"[bench] pp{pp,-8} {rates.Max(),10:F2} tok/s");
            PrintStatistics(rates, pp, 0);
        }

        if (tg > 0)
        {
            var rates = new List<double>(reps);
            // llama-bench's test_gen feeds random tokens and times
            // llama_decode+synchronize; it does not scan the returned vocabulary
            // for an argmax. Precompute the same style of inputs so this number is
            // model-decode throughput rather than decode plus a 150k-element C#
            // sampler pass on every token.
            var decodeTokens = new int[tg];
            for (int i = 0; i < tg; i++) decodeTokens[i] = 1000 + rng.Next(vocab - 1000);
            var tokenBox = new int[1];

            // Match llama-bench's one-token warm-up followed by memory_clear:
            // the measured generation begins at the explicit context depth,
            // zero by default, rather than after an unrelated prompt.
            model.ResetKVCache();
            tokenBox[0] = decodeTokens[0];
            model.Forward(tokenBox);
            for (int r = 0; r < reps; r++)
            {
                ResetContext();
                var t = Stopwatch.StartNew();
                for (int i = 0; i < tg; i++)
                {
                    tokenBox[0] = decodeTokens[i];
                    model.Forward(tokenBox);
                }
                t.Stop();
                RecordSample(rates, 0, tg, r, t.Elapsed.TotalSeconds);
            }
            Console.WriteLine($"[bench] tg{tg,-8} {rates.Max(),10:F2} tok/s");
            PrintStatistics(rates, 0, tg);
        }
        return 0;
    }

    /// <summary>Continuous-batching equivalence: each prompt decodes serially on
    /// its own sequence slot, then all together through the fused batched-decode
    /// step. Batching changes when the weights are read, not what the model
    /// computes, so the streams must agree token for token.</summary>
    private static int RunBatched(string modelPath, string[] args)
    {
        BackendType backend = ResolveBackend(args.Length > 2 ? args[2] : "ggmlcuda");
        int steps = args.Length > 3 ? int.Parse(args[3]) : 8;
        var prompts = new List<int[]>();
        for (int i = 4; i < args.Length; i++)
            prompts.Add(args[i].Split(',', StringSplitOptions.RemoveEmptyEntries)
                               .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture)).ToArray());
        if (prompts.Count < 2) { Console.Error.WriteLine("need at least two prompts"); return 1; }

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        var seq = model as IBatchedPagedModel;
        if (seq == null || !seq.SupportsPerSequenceFusedForward)
        {
            Console.Error.WriteLine("[batched] model has no per-sequence slots");
            return 1;
        }

        int n = prompts.Count;
        var ids = new string[n];
        for (int i = 0; i < n; i++) ids[i] = "req" + i;

        // --- serial: each sequence decoded on its own slot ---
        var serial = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            seq.BindSequenceCache(ids[i]);
            float[] lg = model.Forward(prompts[i]);
            var outs = new List<int>();
            int tok = ArgMax(lg);
            outs.Add(tok);
            for (int s = 1; s < steps; s++)
            {
                lg = model.Forward(new[] { tok });
                tok = ArgMax(lg);
                outs.Add(tok);
            }
            serial.Add(outs);
            seq.OnSequenceReleased(ids[i]);
        }

        // --- batched: same sequences, one fused step per token ---
        var lastTok = new int[n];
        var pos = new int[n];
        var batched = new List<List<int>>();
        for (int i = 0; i < n; i++)
        {
            seq.BindSequenceCache(ids[i]);
            float[] lg = model.Forward(prompts[i]);
            lastTok[i] = ArgMax(lg);
            pos[i] = prompts[i].Length;
            batched.Add(new List<int> { lastTok[i] });
        }

        var outLogits = new float[n][];
        int fusedSteps = 0, fallbackSteps = 0;
        for (int s = 1; s < steps; s++)
        {
            if (seq.TryForwardBatchedFusedDecode(ids, lastTok, pos, outLogits))
            {
                fusedSteps++;
                for (int i = 0; i < n; i++)
                {
                    lastTok[i] = ArgMax(outLogits[i]);
                    pos[i]++;
                    batched[i].Add(lastTok[i]);
                }
            }
            else
            {
                // round-robin fallback, as the engine would
                fallbackSteps++;
                for (int i = 0; i < n; i++)
                {
                    seq.BindSequenceCache(ids[i]);
                    float[] lg = model.Forward(new[] { lastTok[i] });
                    lastTok[i] = ArgMax(lg);
                    pos[i]++;
                    batched[i].Add(lastTok[i]);
                }
            }
        }
        for (int i = 0; i < n; i++) seq.OnSequenceReleased(ids[i]);

        bool allMatch = true;
        for (int i = 0; i < n; i++)
        {
            bool same = serial[i].SequenceEqual(batched[i]);
            allMatch &= same;
            Console.WriteLine($"[batched] seq{i}: {(same ? "MATCH" : "DIFF")}");
            if (!same)
            {
                Console.WriteLine($"          serial : {string.Join(' ', serial[i])}");
                Console.WriteLine($"          batched: {string.Join(' ', batched[i])}");
            }
        }
        Console.WriteLine($"[batched] fused steps={fusedSteps} fallback steps={fallbackSteps}");
        Console.WriteLine(allMatch ? "[batched] CONCURRENT_MATCH" : "[batched] CONCURRENT_DIFFERS");
        return allMatch ? 0 : 2;
    }

    private sealed class EngineRun
    {
        public InferenceCompletion Completion { get; init; }
        public List<int> Tokens { get; init; }
    }

    private static int[] ParseTokenList(string value)
        => value.Split(',', StringSplitOptions.RemoveEmptyEntries)
                .Select(t => int.Parse(t.Trim(), CultureInfo.InvariantCulture))
                .ToArray();

    private static async Task<EngineRun> DrainEngineRun(InferenceRequestHandle handle)
    {
        var tokens = new List<int>();
        await foreach (int token in handle.Tokens.ReadAllAsync())
            tokens.Add(token);
        return new EngineRun
        {
            Completion = await handle.Completion,
            Tokens = tokens,
        };
    }

    private static async Task<(EngineRun A, EngineRun B)> RunEnginePair(
        InferenceEngine engine,
        SchedulerConfig config,
        string phase,
        IReadOnlyList<int> promptA,
        IReadOnlyList<int> promptB,
        int maxNewTokens)
    {
        // Submit both handles before awaiting either one. The engine drains its
        // multi-writer command queue between scheduler steps and admits these as
        // one two-sequence workload.
        var a = new SequenceState(
            $"retained-parity-{phase}-a", promptA, maxNewTokens,
            config.BlockSize, SamplingConfig.Greedy);
        var b = new SequenceState(
            $"retained-parity-{phase}-b", promptB, maxNewTokens,
            config.BlockSize, SamplingConfig.Greedy);
        InferenceRequestHandle handleA = engine.SubmitRequest(a);
        InferenceRequestHandle handleB = engine.SubmitRequest(b);
        Task<EngineRun> runA = DrainEngineRun(handleA);
        Task<EngineRun> runB = DrainEngineRun(handleB);
        await Task.WhenAll(runA, runB);
        return (await runA, await runB);
    }

    private static void RequireLengthCapped(string label, EngineRun run, int expected)
    {
        if (run.Completion.Status != SequenceStatus.FinishedLengthCapped
            || run.Tokens.Count != expected)
        {
            throw new InvalidOperationException(
                $"{label} stopped before the fixed greedy sample ({run.Completion.FinishReason}, " +
                $"{run.Tokens.Count}/{expected} tokens). Choose prompts that do not emit EOS this early.");
        }
    }

    private static void RequireExactTokens(string label, List<int> expected, List<int> actual)
    {
        int limit = Math.Min(expected.Count, actual.Count);
        int firstDiff = -1;
        for (int i = 0; i < limit; i++)
            if (expected[i] != actual[i]) { firstDiff = i; break; }
        if (firstDiff < 0 && expected.Count != actual.Count)
            firstDiff = limit;
        if (firstDiff < 0)
            return;

        throw new InvalidOperationException(
            $"{label} differs at token {firstDiff}: expected=[{string.Join(' ', expected)}], " +
            $"actual=[{string.Join(' ', actual)}]");
    }

    /// <summary>End-to-end retained-holder regression over the real scheduler and
    /// request lifecycle. First collect a retention-disabled, necessarily
    /// full-prefill golden for two follow-ups. Then repeat the same concurrent
    /// round one with retention enabled and require each exact-extension follow-up
    /// to adopt all prompt+output tokens from its own holder and reproduce the
    /// golden stream exactly.</summary>
    private static async Task<int> RunRetainedContinuation(string modelPath, string[] args)
    {
        BackendType backend = ResolveBackend(args.Length > 2 ? args[2] : "ggmlmetal");
        int round1Steps = args.Length > 3
            ? int.Parse(args[3], CultureInfo.InvariantCulture)
            : 8;
        int followSteps = args.Length > 4
            ? int.Parse(args[4], CultureInfo.InvariantCulture)
            : 8;
        if (round1Steps < 2 || followSteps < 1)
        {
            Console.Error.WriteLine("[retained] round1_steps must be >=2 and follow_steps must be positive");
            return 1;
        }

        string[] optionNames =
        {
            "TS_RETAINED_FUSED_CACHE_MAX",
            "TS_PREFIX_CHECKPOINTS_MAX",
            "TS_SCHED_DISABLE_BATCHED",
            "TS_PER_SEQ_FUSED",
            "TS_BATCHED_FUSED_DECODE",
        };
        var previousOptions = optionNames.ToDictionary(
            name => name,
            Environment.GetEnvironmentVariable,
            StringComparer.Ordinal);
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "0");
        Environment.SetEnvironmentVariable("TS_PREFIX_CHECKPOINTS_MAX", "0");
        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
        Environment.SetEnvironmentVariable("TS_PER_SEQ_FUSED", "1");
        Environment.SetEnvironmentVariable("TS_BATCHED_FUSED_DECODE", "1");

        try
        {
            var load = Stopwatch.StartNew();
            using var model = ModelBase.Create(modelPath, backend, ResolveTp());
            load.Stop();
            if (model is not IBatchedPagedModel fused
                || !fused.SupportsPerSequenceFusedForward
                || !fused.SupportsRetainedFusedCache)
            {
                Console.Error.WriteLine(
                    "[retained] model/backend does not expose per-sequence retained fused holders");
                return 1;
            }

            int[] promptA = args.Length > 5
                ? ParseTokenList(args[5])
                : model.Tokenizer.Encode(
                    "User: Reply with the word apple and then count upward.\nAssistant:",
                    addSpecial: true).ToArray();
            int[] promptB = args.Length > 6
                ? ParseTokenList(args[6])
                : model.Tokenizer.Encode(
                    "User: Reply with the word metal and then count upward.\nAssistant:",
                    addSpecial: true).ToArray();
            int[] suffix = args.Length > 7
                ? ParseTokenList(args[7])
                : model.Tokenizer.Encode("\nContinue:", addSpecial: false).ToArray();
            if (promptA.Length == 0 || promptB.Length == 0 || suffix.Length == 0)
            {
                Console.Error.WriteLine("[retained] prompts and suffix must be non-empty");
                return 1;
            }

            var config = new SchedulerConfig
            {
                MaxNumBatchedTokens = 256,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 32,
                SoloPrefillChunkSize = 256,
                NumBlocks = 512,
                BlockSize = 8,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = 1,
            };
            using var engine = new InferenceEngine(model, config);
            Console.WriteLine(
                $"[retained] loaded in {load.Elapsed.TotalSeconds:F1}s, backend={backend}, " +
                $"prompt={promptA.Length}/{promptB.Length}, suffix={suffix.Length}");

            // Retention-disabled control: the two second-round prompts are fully
            // prefilled, so their streams are the ground truth for this engine.
            var controlRound1 = await RunEnginePair(
                engine, config, "control-r1", promptA, promptB, round1Steps);
            RequireLengthCapped("control round1 A", controlRound1.A, round1Steps);
            RequireLengthCapped("control round1 B", controlRound1.B, round1Steps);

            int[] controlFollowA = promptA.Concat(controlRound1.A.Tokens).Concat(suffix).ToArray();
            int[] controlFollowB = promptB.Concat(controlRound1.B.Tokens).Concat(suffix).ToArray();
            var controlFollow = await RunEnginePair(
                engine, config, "control-r2", controlFollowA, controlFollowB, followSteps);
            RequireLengthCapped("control follow-up A", controlFollow.A, followSteps);
            RequireLengthCapped("control follow-up B", controlFollow.B, followSteps);
            if (controlFollow.A.Completion.PrefixCacheReusedTokens != 0
                || controlFollow.B.Completion.PrefixCacheReusedTokens != 0)
            {
                throw new InvalidOperationException(
                    "retention-disabled control unexpectedly reused a prefix; it is not a full-prefill golden");
            }

            // Repeat round one with holder retention enabled. Its deterministic
            // streams must match the control before they are used as prefixes.
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE_MAX", "4");
            var retainedRound1 = await RunEnginePair(
                engine, config, "enabled-r1", promptA, promptB, round1Steps);
            RequireLengthCapped("retained round1 A", retainedRound1.A, round1Steps);
            RequireLengthCapped("retained round1 B", retainedRound1.B, round1Steps);
            RequireExactTokens("round1 A", controlRound1.A.Tokens, retainedRound1.A.Tokens);
            RequireExactTokens("round1 B", controlRound1.B.Tokens, retainedRound1.B.Tokens);

            int[] retainedFollowA = promptA.Concat(retainedRound1.A.Tokens).Concat(suffix).ToArray();
            int[] retainedFollowB = promptB.Concat(retainedRound1.B.Tokens).Concat(suffix).ToArray();
            var retainedFollow = await RunEnginePair(
                engine, config, "enabled-r2", retainedFollowA, retainedFollowB, followSteps);
            RequireLengthCapped("retained follow-up A", retainedFollow.A, followSteps);
            RequireLengthCapped("retained follow-up B", retainedFollow.B, followSteps);

            int expectedReuseA = promptA.Length + retainedRound1.A.Tokens.Count;
            int expectedReuseB = promptB.Length + retainedRound1.B.Tokens.Count;
            if (retainedFollow.A.Completion.PrefixCacheReusedTokens != expectedReuseA
                || retainedFollow.B.Completion.PrefixCacheReusedTokens != expectedReuseB)
            {
                throw new InvalidOperationException(
                    "retained prefix metric mismatch: " +
                    $"A={retainedFollow.A.Completion.PrefixCacheReusedTokens}/{expectedReuseA}, " +
                    $"B={retainedFollow.B.Completion.PrefixCacheReusedTokens}/{expectedReuseB}");
            }
            RequireExactTokens("follow-up A", controlFollow.A.Tokens, retainedFollow.A.Tokens);
            RequireExactTokens("follow-up B", controlFollow.B.Tokens, retainedFollow.B.Tokens);

            static double TtftMs(EngineRun run) => run.Completion.FirstTokenAt.HasValue
                ? (run.Completion.FirstTokenAt.Value - run.Completion.SubmittedAt).TotalMilliseconds
                : double.NaN;
            Console.WriteLine(
                $"[retained] reuse A={expectedReuseA}/{retainedFollowA.Length}, " +
                $"B={expectedReuseB}/{retainedFollowB.Length}");
            Console.WriteLine(
                $"[retained] follow-up TTFT full={TtftMs(controlFollow.A):F1}/{TtftMs(controlFollow.B):F1}ms " +
                $"retained={TtftMs(retainedFollow.A):F1}/{TtftMs(retainedFollow.B):F1}ms");
            Console.WriteLine("[retained] PASS: both exact-extension streams reused their complete holder and match full-prefill goldens");
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[retained] FAIL: {ex.Message}");
            return 2;
        }
        finally
        {
            foreach (var entry in previousOptions)
                Environment.SetEnvironmentVariable(entry.Key, entry.Value);
        }
    }

    /// <summary>Regression for arena-slot reuse at position zero. A slot's KV
    /// prefix is empty at position zero, but its complete recurrent state must
    /// still be seeded. Keep one sequence resident, repeatedly dirty and replace
    /// the other lane without releasing its holder (which avoids request-pool
    /// invalidation masking the native slot-reuse edge), and compare every fresh
    /// lane's first argmax with a clean solo reference.</summary>
    private static int RunArenaZeroReuse(string modelPath, string[] args)
    {
        BackendType backend = ResolveBackend(args.Length > 2 ? args[2] : "ggmlmetal");
        int cycles = args.Length > 3 ? int.Parse(args[3], CultureInfo.InvariantCulture) : 8;
        int dirtySteps = args.Length > 4 ? int.Parse(args[4], CultureInfo.InvariantCulture) : 8;
        int probeToken = args.Length > 5 ? int.Parse(args[5], CultureInfo.InvariantCulture) : 1;
        int survivorToken = args.Length > 6 ? int.Parse(args[6], CultureInfo.InvariantCulture) : 2;
        if (cycles < 1 || dirtySteps < 1)
        {
            Console.Error.WriteLine("[arena-zero] cycles and dirty_steps must be positive");
            return 1;
        }

        using var model = ModelBase.Create(modelPath, backend, ResolveTp());
        if (model is not IBatchedPagedModel seq || !seq.SupportsPerSequenceFusedForward)
        {
            Console.Error.WriteLine("[arena-zero] model has no per-sequence fused slots");
            return 1;
        }

        // Warm the model's descriptor table and establish the position-zero gold.
        const string referenceId = "arena-zero-reference";
        seq.BindSequenceCache(referenceId);
        int expectedProbeNext = ArgMax(model.Forward(new[] { probeToken }));
        seq.OnSequenceReleased(referenceId);

        const string survivorId = "arena-zero-survivor";
        string replaceId = "arena-zero-probe-0";
        var allIds = new List<string> { survivorId, replaceId };
        seq.BindSequenceCache(replaceId);
        seq.BindSequenceCache(survivorId);

        var ids = new[] { replaceId, survivorId };
        var tokens = new[] { probeToken, survivorToken };
        var positions = new[] { 0, 0 };
        var logits = new float[2][];
        int survivorNext = survivorToken;
        int survivorPos = 0;

        bool StepAndCheckProbe(int cycle, bool checkProbe)
        {
            if (!seq.TryForwardBatchedFusedDecode(ids, tokens, positions, logits))
            {
                Console.Error.WriteLine($"[arena-zero] fused arena declined at cycle {cycle}, positions={positions[0]},{positions[1]}");
                return false;
            }
            int probeNext = ArgMax(logits[0]);
            survivorNext = ArgMax(logits[1]);
            if (checkProbe && probeNext != expectedProbeNext)
            {
                Console.Error.WriteLine($"[arena-zero] cycle {cycle}: probe DIFF, expected {expectedProbeNext}, got {probeNext}");
                return false;
            }
            tokens[0] = probeNext;
            tokens[1] = survivorNext;
            positions[0]++;
            positions[1]++;
            survivorPos = positions[1];
            return true;
        }

        if (!StepAndCheckProbe(0, checkProbe: true)) return 2;
        for (int step = 1; step < dirtySteps; step++)
            if (!StepAndCheckProbe(0, checkProbe: false)) return 2;

        for (int cycle = 1; cycle <= cycles; cycle++)
        {
            // Do not release the displaced holder yet. That would route through
            // the managed reuse invalidation and rebuild the arena, hiding this
            // native dirty-slot reuse condition.
            replaceId = $"arena-zero-probe-{cycle}";
            allIds.Add(replaceId);
            seq.BindSequenceCache(replaceId);
            ids[0] = replaceId;
            tokens[0] = probeToken;
            tokens[1] = survivorNext;
            positions[0] = 0;
            positions[1] = survivorPos;

            if (!StepAndCheckProbe(cycle, checkProbe: true)) return 2;
            for (int step = 1; step < dirtySteps; step++)
                if (!StepAndCheckProbe(cycle, checkProbe: false)) return 2;
            Console.WriteLine($"[arena-zero] cycle {cycle}/{cycles}: MATCH");
        }

        foreach (string id in allIds)
            seq.OnSequenceReleased(id);
        Console.WriteLine($"[arena-zero] PASS ({cycles} dirty-slot replacements, {dirtySteps} steps each)");
        return 0;
    }
}
