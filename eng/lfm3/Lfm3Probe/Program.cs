// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LFM3 validation probe (TensorSharp side of eng/lfm3's llama.cpp comparisons; outputs belong in ignored artifacts/):
//   ppl    MODEL BACKEND FILE CTX CHUNKS [--per-op]   (also prints a digest of every scored logit's bits)
//                                          perplexity as llama-perplexity computes it: the file's tokens (no BOS, special
//                                          tokens not parsed) in CHUNKS chunks of CTX, each from an empty cache, scoring the
//                                          second half of every chunk (row i predicts token i + 1); --per-op runs the shared
//                                          runtime's per-op path instead of whole-forward graphs
//   tokens MODEL FILE OUT                  the file's token ids as llama-perplexity tokenizes it (one id a line)
//   bench  MODEL BACKEND [PP,...|-] [TG] [DEPTH,...] [--reps N] [--greedy]
//                                          prefill tokens/s of each PP-token prompt and decode tokens/s of TG tokens after
//                                          DEPTH tokens (llama-bench's pp / tg / tg@d), each the median of N runs (3; the mean
//                                          of the middle two for an even N, as llama-bench's statistic), with the decode steps'
//                                          graph counters (runs, advances in place, builds). Each test is warmed by one untimed
//                                          run as llama-bench warms its tests (a decode test: TG tokens after a 1-token prompt,
//                                          so the managed lane's JIT has tiered the decode kernels up). As llama-bench, decode
//                                          feeds pseudo-random tokens (the timed loop holds no sampling); --greedy feeds each
//                                          step's host argmax instead. A "RESULT name median min max" line per test, and on
//                                          stderr an "INFO warm-up done" line where the start-up warm-up ends.
//   spec   MODEL BACKEND FILE N            N greedy tokens after the file's prompt, plainly and with n-gram speculation (a verify
//                                          window keeps the short-conv inputs and attends under the sliding window; the accept
//                                          count commits a prefix): the two streams must be equal
//   chat   MODEL REQUESTS OUT              each chat request's prompt token ids as TensorSharp.Server renders a fresh request
//                                          (the GGUF template through KVCachePromptRenderer, the tokenizer's BOS): REQUESTS is
//                                          JSONL {"id", "messages": [{"role", "content"}], "enable_thinking"}, OUT gets
//                                          {"id", "tokens"} a line; eng/lfm3/prompt-parity.py compares them with llama-server's
//                                          and Hugging Face's renders of the same requests
//   batch  MODEL BACKEND [--sequences N,..] [--prompt P] [--warm W] [--steps S] [--reps R] [--greedy] [--ps] [--profile-range]
//                                          the census of batched decode steps (stage B0 of the batched-serving design): for each N,
//                                          N slots take a P-token prompt each (alone), then decode together through
//                                          IUnifiedBatchModel.TryForwardUnified as the server's scheduler drives them: W untimed
//                                          steps, then R runs of S timed steps. Per cell: step time (median, p10, p90), output
//                                          tok/s, the host's share (the step outside the executor's runs) and the run's, graph
//                                          builds / replays / advances / recordings per step, input leaves and bytes, results read
//                                          back, the executor's nodes and kernels, leaves every run copies, managed allocation and
//                                          collections; then each metric's least-squares fit over N (cells from N = 4 up) and the
//                                          growth past N = 4. Tokens are pseudo-random (no sampling in the timed loop); --greedy
//                                          feeds each sequence its argmax (device ids where the executor returns them, as the
//                                          server; the host scan is timed apart as sample_ms). --ps prints the busiest processes
//                                          before each cell. --profile-range brackets each cell's timed steps with the CUDA
//                                          profiler API (nsys --capture-range=cudaProfilerApi: eng/lfm3/bench/step-census.sh).
// BACKEND: ggml_metal | ggml_cpu | cpu | mlx | ggml_cuda | ggml_vulkan | cuda
using System.Diagnostics;
using System.Globalization;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

if (args.Length < 2) return Usage();
BackendType Backend(string name) => name switch
{
    "ggml_metal" => BackendType.GgmlMetal, "ggml_cpu" => BackendType.GgmlCpu, "cpu" => BackendType.Cpu, "mlx" => BackendType.Mlx,
    "ggml_cuda" => BackendType.GgmlCuda, "ggml_vulkan" => BackendType.GgmlVulkan, "cuda" => BackendType.Cuda,
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
    case "chat" when args.Length >= 4:
        return ChatPromptTokens(args[1], args[2], args[3]);
    case "batch" when args.Length >= 3:
        return Batch(args[1], Backend(args[2]), args);
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
    Console.Error.WriteLine("usage: Lfm3Probe ppl MODEL BACKEND FILE CTX CHUNKS [--per-op] | spec MODEL BACKEND FILE N | tokens MODEL FILE OUT | chat MODEL REQUESTS OUT | bench MODEL BACKEND [PP,..|-] [TG] [DEPTH,..] [--reps N] [--greedy]"
        + " | batch MODEL BACKEND [--sequences N,..] [--prompt P] [--warm W] [--steps S] [--reps R] [--greedy] [--ps] [--profile-range]");
    return 2;
}

static int ChatPromptTokens(string path, string requests, string output)
{
    using var gguf = GgufFile.OpenWithoutSiblingShards(path);
    ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
    string template = gguf.GetString("tokenizer.chat_template");
    string architecture = gguf.GetString("general.architecture");
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    using var writer = new StreamWriter(output);
    int count = 0;
    foreach (string line in File.ReadLines(requests))
    {
        if (string.IsNullOrWhiteSpace(line)) continue;
        using JsonDocument doc = JsonDocument.Parse(line);
        JsonElement root = doc.RootElement;
        var messages = root.GetProperty("messages").EnumerateArray()
            .Select(m => new ChatMessage { Role = m.GetProperty("role").GetString(), Content = m.GetProperty("content").GetString() })
            .ToList();
        bool thinking = root.TryGetProperty("enable_thinking", out JsonElement t) && t.GetBoolean();
        List<int> tokens = renderer.RenderToTokens(tokenizer, template, messages, architecture, addGenerationPrompt: true,
            enableThinking: thinking);
        writer.WriteLine(JsonSerializer.Serialize(new { id = root.GetProperty("id").GetRawText(), tokens }));
        count++;
    }
    Console.Error.WriteLine($"rendered {count} requests ({architecture}, template {template?.Length ?? 0} chars)");
    return 0;
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
    ulong digest = 14695981039346656037UL;   // FNV-1a over every scored row's logit bits: equal digests, equal logits
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
                foreach (float v in row)
                    digest = (digest ^ BitConverter.SingleToUInt32Bits(v)) * 1099511628211UL;
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
    Console.WriteLine($"logits digest {digest:X16}, nll {nll:R}");
    return 0;
}

static int Bench(string path, BackendType backend, int[] prompts, int generate, int[] depths, int reps, bool greedy)
{
    using ModelBase model = ModelBase.Create(path, backend);
    var runtime = model as StandardDecoderModel;
    int vocab = model.Config.VocabSize;
    int[] Tokens(int n, int seed) => Enumerable.Range(0, n).Select(i => 1000 + (int)((uint)(i * 2654435761u + seed) % 60000u)).ToArray();
    // The true median: the mean of the middle two for an even count (llama-bench-result.sh's statistics.median). Taking
    // the upper one made a 2-rep figure the faster run, while llama.cpp's was the mean of its two.
    static double Median(List<double> v)
    {
        v.Sort();
        return v.Count % 2 == 1 ? v[v.Count / 2] : (v[v.Count / 2 - 1] + v[v.Count / 2]) / 2;
    }

    // Warm: kernels load and graphs build once, and the host code the prompt and decode steps run is past .NET's first
    // (tier-0) compilation, as in a server that has run a while: 40 prompt steps of up to 512 rows, then 64 decode steps.
    // (A single 256-token prompt left most of the prompt path tier-0 for the first timed prompts: their host work per
    // 512-row step was 2x a warmed process's on the A40 VM.)
    // The warm-up's length depends on MAX_CONTEXT and the lane (a minute on a CPU lane at 6 threads), so tools that profile
    // the tests (bench/cpu-profile.sh, node-profile-diff.py) start from its end: one stderr line when it is done.
    var warmClock = Stopwatch.StartNew();
    int warm = Math.Max(256, Math.Min(5120, model.MaxContextLength - 64));
    for (int w = 0; w < 4; w++)
    {
        model.ResetKVCache();
        model.ForwardRefill(Tokens(warm, w + 101));
    }
    model.ResetKVCache();
    model.ForwardRefill(Tokens(256, 1));
    for (int i = 0; i < 64; i++) model.Forward(new[] { 1234 + i });
    Console.Error.WriteLine(FormattableString.Invariant(
        $"INFO warm-up done in {warmClock.Elapsed.TotalSeconds:F1} s (4 prompts of {warm} tokens, 256 tokens and 64 decode steps)"));

    foreach (int n in prompts)
    {
        // One untimed prompt of this size first, as llama-bench warms every test with its own prompt: a size's first
        // step builds its graphs, and on the managed lane its kernels are still being tiered up by the JIT.
        model.ResetKVCache();
        model.ForwardRefill(Tokens(n, 5));
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
        // One untimed decode run first, as llama-bench warms every test (with one token: its kernels are compiled ahead
        // of time). The managed lane's decode kernels were still being tiered up by the JIT after a start-up warm-up of 8
        // decode steps (the warm-up's length before it took 64), and its first timed run was 25-35 % slower than the rest.
        // At depth 0, so a deep test does not pay its prompt twice.
        model.ResetKVCache();
        model.ForwardRefill(new[] { 1 });
        foreach (int token in Tokens(generate, 9)) model.Forward(new[] { token });
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

static int Batch(string path, BackendType backend, string[] args)
{
    string Option(string name, string fallback)
    {
        int i = Array.IndexOf(args, name);
        return i >= 0 && i + 1 < args.Length ? args[i + 1] : fallback;
    }
    int[] sizes = Option("--sequences", "1,2,4,8,16").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray();
    int prompt = int.Parse(Option("--prompt", "128"), CultureInfo.InvariantCulture);
    int warm = int.Parse(Option("--warm", "32"), CultureInfo.InvariantCulture);
    int steps = int.Parse(Option("--steps", "64"), CultureInfo.InvariantCulture);
    int reps = int.Parse(Option("--reps", "3"), CultureInfo.InvariantCulture);
    bool greedy = args.Contains("--greedy"), ps = args.Contains("--ps"), profileRange = args.Contains("--profile-range");
    if (sizes.Length == 0 || sizes.Any(n => n < 1) || prompt < 1 || warm < 0 || steps < 1 || reps < 1) return Usage();

    using ModelBase model = ModelBase.Create(path, backend);
    if (model is not StandardDecoderModel runtime || model is not IUnifiedBatchModel unified || !unified.SupportsUnifiedBatching)
    {
        Console.Error.WriteLine($"{model.Config.Architecture} on {backend} does not batch over slots (IUnifiedBatchModel)");
        return 1;
    }
    int vocab = model.Config.VocabSize;
    bool deviceIds = greedy && unified.SupportsUnifiedGreedyIds;
    Console.WriteLine(string.Concat(
        FormattableString.Invariant($"batch: {model.Config.Architecture} {Path.GetFileName(path)} on {backend}; prompt {prompt} tokens a sequence (each alone), "),
        FormattableString.Invariant($"{warm} untimed then {reps} x {steps} timed decode steps a cell; tokens "),
        greedy ? deviceIds ? "greedy (device ids)" : "greedy (host argmax)" : "pseudo-random",
        FormattableString.Invariant($"; KV {model.KvCacheDtype}; graph executor {runtime.GraphExecutorForTests?.GetType().Name ?? "none (per-op)"}")));

    int serial = 0;
    int Pseudo(int cell, int sequence, int step)
        => 100 + (int)((uint)((step + 7) * 2654435761u + (uint)sequence * 40503u + (uint)cell * 97u) % (uint)Math.Max(1, Math.Min(vocab, 60000) - 100));

    BatchCell RunCell(int n, int timedReps, int timedSteps, int warmSteps, bool profile)
    {
        int cell = serial++;
        string[] ids = Enumerable.Range(0, n).Select(i => $"batch{cell}-{i}").ToArray();
        int reserve = prompt + warmSteps + timedReps * timedSteps + 8;
        var starts = new int[n];
        var next = new int[n];
        var logits = new float[n][];
        var result = new BatchCell { Sequences = n, LoadBefore = LoadAverage() };
        try
        {
            for (int s = 0; s < n; s++)
            {
                unified.EnsureSequenceSlot(ids[s], reserve);
                int[] tokens = Enumerable.Range(0, prompt).Select(i => Pseudo(cell, s, -1000 - i)).ToArray();
                var one = new float[1][];
                if (!unified.TryForwardUnified(new[] { ids[s] }, new[] { tokens }, new[] { 0 }, one))
                    throw new InvalidOperationException($"the prompt of sequence {s} of {n} declined");
                starts[s] = prompt;
                next[s] = greedy ? ArgMax(one[0], vocab) : Pseudo(cell, s, -1);
            }

            (long Step, long Sample) Step(int t)
            {
                var tokens = new int[n][];
                for (int s = 0; s < n; s++) tokens[s] = new[] { next[s] };
                long t0 = Stopwatch.GetTimestamp();
                bool ok = deviceIds ? unified.TryForwardUnifiedGreedy(ids, tokens, starts, next) : unified.TryForwardUnified(ids, tokens, starts, logits);
                long t1 = Stopwatch.GetTimestamp();
                if (!ok) throw new InvalidOperationException($"a batched decode step of {n} sequences declined");
                for (int s = 0; s < n; s++) starts[s]++;
                if (!deviceIds)
                    for (int s = 0; s < n; s++) next[s] = greedy ? ArgMax(logits[s], vocab) : Pseudo(cell, s, t);
                return (t1 - t0, Stopwatch.GetTimestamp() - t1);
            }

            BatchCounters warmStart = BatchCounters.Take(runtime);
            for (int w = 0; w < warmSteps; w++) Step(-2 - w);
            BatchCounters before = BatchCounters.Take(runtime);
            result.WarmBuilds = before.Builds - warmStart.Builds;
            result.WarmRecordings = before.Recordings - warmStart.Recordings;
            long allocated = GC.GetTotalAllocatedBytes(false);
            int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
            long stepTicks = 0, sampleTicks = 0;
            if (profile) CudaProfilerRange.Start();
            for (int r = 0; r < timedReps; r++)
            {
                var rep = new List<double>(timedSteps);
                for (int t = 0; t < timedSteps; t++)
                {
                    (long a, long b) = Step(r * timedSteps + t);
                    stepTicks += a;
                    sampleTicks += b;
                    rep.Add(a * 1000.0 / Stopwatch.Frequency);
                }
                result.StepMs.AddRange(rep);
                result.RepMedians.Add(Percentile(rep, 0.5));
            }
            if (profile) CudaProfilerRange.Stop();
            BatchCounters after = BatchCounters.Take(runtime);
            double count = timedReps * timedSteps;
            double Ms(long ticks) => ticks * 1000.0 / Stopwatch.Frequency / count;
            StandardDecoderModel.StepCensus c0 = before.Census, c1 = after.Census;
            result.Steps = (int)count;
            result.HostMs = Ms(stepTicks - (c1.RunTicks - c0.RunTicks));
            result.RunMs = Ms(c1.RunTicks - c0.RunTicks);
            result.SampleMs = Ms(sampleTicks);
            result.Per["forwards"] = (after.Forwards - before.Forwards) / count;
            result.Per["runs"] = (c1.Runs - c0.Runs) / count;
            result.Per["builds"] = (after.Builds - before.Builds) / count;
            result.Per["replays"] = (after.Replays - before.Replays) / count;
            result.Per["advances"] = (after.Advances - before.Advances) / count;
            result.Per["recordings"] = (after.Recordings - before.Recordings) / count;
            result.Per["recorded_replays"] = (after.RecordedReplays - before.RecordedReplays) / count;
            result.Per["swa_unrecorded"] = (after.SlidingUnrecorded - before.SlidingUnrecorded) / count;
            result.Per["input_leaves"] = (c1.InputLeaves - c0.InputLeaves) / count;
            result.Per["input_kb"] = (c1.InputBytes - c0.InputBytes) / 1024.0 / count;
            result.Per["output_kb"] = (c1.OutputBytes - c0.OutputBytes) / 1024.0 / count;
            long censusRuns = c1.CensusRuns - c0.CensusRuns;
            result.CensusShare = c1.Runs > c0.Runs ? censusRuns / (double)(c1.Runs - c0.Runs) : 0;
            if (censusRuns > 0)
            {
                result.Per["nodes"] = (c1.Nodes - c0.Nodes) / count;
                result.Per["kernels"] = (c1.Kernels - c0.Kernels) / count;
                result.Per["leaf_uploads"] = (c1.LeafUploads - c0.LeafUploads) / count;
                result.Per["leaf_upload_kb"] = (c1.LeafUploadBytes - c0.LeafUploadBytes) / 1024.0 / count;
                result.Per["leaf_downloads"] = (c1.LeafDownloads - c0.LeafDownloads) / count;
                result.Per["leaf_download_kb"] = (c1.LeafDownloadBytes - c0.LeafDownloadBytes) / 1024.0 / count;
            }
            result.Per["alloc_kb"] = (GC.GetTotalAllocatedBytes(false) - allocated) / 1024.0 / count;
            result.Gc = (GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2);
        }
        finally
        {
            foreach (string id in ids) runtime.OnSequenceReleased(id);
        }
        result.LoadAfter = LoadAverage();
        return result;
    }

    // Start-up warm-up, untimed: graphs of the shapes, kernels loaded, and the host code past .NET's tier-0 compilation.
    var warmClock = Stopwatch.StartNew();
    RunCell(Math.Min(4, sizes.Max()), 1, 32, 16, profile: false);
    Console.Error.WriteLine(FormattableString.Invariant($"INFO warm-up done in {warmClock.Elapsed.TotalSeconds:F1} s"));

    var cells = new List<BatchCell>();
    foreach (int n in sizes)
    {
        if (ps) PrintBusiest();
        BatchCell c = RunCell(n, reps, steps, warm, profileRange);
        cells.Add(c);
        double median = Percentile(c.RepMedians.ToList(), 0.5);
        var inv = CultureInfo.InvariantCulture;
        Console.WriteLine(string.Create(inv, $"CELL n={n} load {c.LoadBefore} -> {c.LoadAfter}"));
        Console.WriteLine(string.Create(inv, $"RESULT batch.n{n}.step_ms {median:F3} {c.RepMedians.Min():F3} {c.RepMedians.Max():F3}"));
        Console.WriteLine(string.Create(inv, $"RESULT batch.n{n}.tok_s {n * 1000.0 / median:F1} {n * 1000.0 / c.RepMedians.Max():F1} {n * 1000.0 / c.RepMedians.Min():F1}"));
        var line = new System.Text.StringBuilder();
        line.Append(string.Create(inv, $"CENSUS n={n} steps={c.Steps} step_ms={Percentile(c.StepMs, 0.5):F3} p10={Percentile(c.StepMs, 0.1):F3} "))
            .Append(string.Create(inv, $"p90={Percentile(c.StepMs, 0.9):F3} mean={c.StepMs.Average():F3} host_ms={c.HostMs:F3} run_ms={c.RunMs:F3} "))
            .Append(string.Create(inv, $"sample_ms={c.SampleMs:F3}"));
        foreach ((string key, double value) in c.Per) line.Append(string.Create(inv, $" {key}={value:0.###}"));
        line.Append(string.Create(inv, $" census_share={c.CensusShare:0.##} gc={c.Gc.G0}/{c.Gc.G1}/{c.Gc.G2} warm_builds={c.WarmBuilds} warm_recordings={c.WarmRecordings}"));
        Console.WriteLine(line.ToString());
    }
    ReportBatchFits(cells);
    return 0;
}

static int ArgMax(float[] logits, int vocab)
{
    int best = 0;
    for (int v = 1; v < vocab; v++) if (logits[v] > logits[best]) best = v;
    return best;
}

// The value at fraction p of the sorted values (linear between neighbours; 0.5 is the true median).
static double Percentile(List<double> values, double p)
{
    if (values.Count == 0) return double.NaN;
    var v = values.ToArray();
    Array.Sort(v);
    double at = p * (v.Length - 1);
    int lo = (int)Math.Floor(at), hi = Math.Min(v.Length - 1, lo + 1);
    return v[lo] + (v[hi] - v[lo]) * (at - lo);
}

static string LoadAverage()
{
    try
    {
        if (OperatingSystem.IsLinux())
            return string.Join("/", File.ReadAllText("/proc/loadavg").Split(' ').Take(3));
        if (OperatingSystem.IsMacOS())
        {
            using var p = Process.Start(new ProcessStartInfo("sysctl", "-n vm.loadavg") { RedirectStandardOutput = true, UseShellExecute = false })!;
            string text = p.StandardOutput.ReadToEnd();
            p.WaitForExit();
            return string.Join("/", text.Trim().Trim('{', '}').Split(' ', StringSplitOptions.RemoveEmptyEntries).Take(3));
        }
    }
    catch (Exception)
    {
    }
    return "?";
}

// The six busiest processes by CPU, before a cell (the machine's other load, recorded with the timings).
static void PrintBusiest()
{
    try
    {
        var info = OperatingSystem.IsMacOS()
            ? new ProcessStartInfo("ps", "-Ao pcpu,rss,pid,comm -r")
            : new ProcessStartInfo("ps", "-eo pcpu,rss,pid,comm --sort=-pcpu");
        info.RedirectStandardOutput = true;
        info.UseShellExecute = false;
        using var p = Process.Start(info)!;
        string[] lines = p.StandardOutput.ReadToEnd().Split('\n', StringSplitOptions.RemoveEmptyEntries);
        p.WaitForExit();
        foreach (string l in lines.Take(7)) Console.WriteLine("PS " + (l.Length > 140 ? l[..140] : l));
    }
    catch (Exception ex)
    {
        Console.WriteLine("PS unavailable: " + ex.Message);
    }
}

// Each metric's least-squares line over the sizes (cells from N = 4 up when there are two such sizes), and the growth from N = 4
// to the largest N: how much of the step's growth was the host's, the executor's run's and the sampling's, and what each added
// sequence brought (kernels, inputs, results). Medians of a size's cells where it ran more than once.
static void ReportBatchFits(List<BatchCell> cells)
{
    var inv = CultureInfo.InvariantCulture;
    var bySize = cells.GroupBy(c => c.Sequences).OrderBy(g => g.Key).ToList();
    if (bySize.Count < 2) return;
    double Med(IEnumerable<double> v) => Percentile(v.ToList(), 0.5);
    var metrics = new List<string> { "step_ms", "host_ms", "run_ms", "sample_ms" };
    foreach (string key in cells.SelectMany(c => c.Per.Keys).Distinct())
        if (!metrics.Contains(key)) metrics.Add(key);
    double Value(BatchCell c, string key) => key switch
    {
        "step_ms" => Percentile(c.StepMs, 0.5),
        "host_ms" => c.HostMs,
        "run_ms" => c.RunMs,
        "sample_ms" => c.SampleMs,
        _ => c.Per.TryGetValue(key, out double v) ? v : double.NaN,
    };
    double At(int n, string key) => Med(bySize.First(g => g.Key == n).Select(c => Value(c, key)));
    int[] xs = bySize.Select(g => g.Key).ToArray();
    int[] fitSizes = xs.Count(n => n >= 4) >= 2 ? xs.Where(n => n >= 4).ToArray() : xs;
    Console.WriteLine($"FIT least squares of each per-step metric against N over N = {string.Join(",", fitSizes)}: value = a + b N");
    foreach (string key in metrics)
    {
        double[] y = fitSizes.Select(n => At(n, key)).ToArray();
        if (y.Any(double.IsNaN)) continue;
        double mx = fitSizes.Average(), my = y.Average(), sxx = 0, sxy = 0;
        for (int i = 0; i < y.Length; i++) { sxx += (fitSizes[i] - mx) * (fitSizes[i] - mx); sxy += (fitSizes[i] - mx) * (y[i] - my); }
        double b = sxx > 0 ? sxy / sxx : 0, a = my - b * mx;
        Console.WriteLine(string.Create(inv, $"FIT {key} a={a:0.####} b={b:0.####} per sequence  ({string.Join(" ", fitSizes.Select((n, i) => $"N{n}={y[i]:0.###}"))})"));
    }
    int top = xs.Max();
    if (!xs.Contains(4) || top <= 4) return;
    double dStep = At(top, "step_ms") - At(4, "step_ms");
    double dHost = At(top, "host_ms") - At(4, "host_ms"), dRun = At(top, "run_ms") - At(4, "run_ms"), dSample = At(top, "sample_ms") - At(4, "sample_ms");
    string Share(double d) => dStep > 0 ? string.Create(inv, $"{100 * d / dStep:F0}%") : "n/a";
    var growth = new System.Text.StringBuilder(string.Create(inv,
        $"GROWTH N=4 -> {top}: step +{dStep:F3} ms (host outside the executor +{dHost:F3} = {Share(dHost)}, executor runs +{dRun:F3} = {Share(dRun)}, sampling +{dSample:F3} = {Share(dSample)})"));
    foreach (string key in new[] { "kernels", "nodes", "input_leaves", "input_kb", "output_kb", "leaf_uploads", "leaf_upload_kb", "alloc_kb" })
    {
        double d = At(top, key) - At(4, key);
        if (!double.IsNaN(d)) growth.Append(string.Create(inv, $"; {key} +{d / (top - 4):0.##} a sequence"));
    }
    Console.WriteLine(growth.ToString());
}

// One size's census (Lfm3Probe batch).
sealed class BatchCell
{
    public int Sequences, Steps;
    public string LoadBefore = "?", LoadAfter = "?";
    public readonly List<double> StepMs = new(), RepMedians = new();
    public double HostMs, RunMs, SampleMs, CensusShare;
    public readonly Dictionary<string, double> Per = new();
    public (int G0, int G1, int G2) Gc;
    public int WarmBuilds, WarmRecordings;
}

// The runtime's counters at one moment.
sealed class BatchCounters
{
    public int Forwards, Builds, Replays, Advances, Recordings, RecordedReplays, SlidingUnrecorded;
    public StandardDecoderModel.StepCensus Census = null!;

    public static BatchCounters Take(StandardDecoderModel m) => new()
    {
        Forwards = m.BatchedForwardCount, Builds = m.GraphBuilds, Replays = m.GraphReplays, Advances = m.DecodeAdvances,
        Recordings = m.DecodeRecordings, RecordedReplays = m.RecordedDecodeReplays, SlidingUnrecorded = m.SlidingWindowUnrecordedSteps,
        Census = m.Census.Clone(),
    };
}

// The CUDA profiler API, which brackets a capture range for nsys --capture-range=cudaProfilerApi (driver API first: one
// library whatever runtime the backends link).
static class CudaProfilerRange
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int Call();

    private static Call? s_start, s_stop;
    private static string? s_api;
    private static bool s_loaded;

    private static void Load()
    {
        if (s_loaded) return;
        s_loaded = true;
        foreach ((string lib, string start, string stop) in new[]
        {
            ("libcuda.so.1", "cuProfilerStart", "cuProfilerStop"), ("nvcuda.dll", "cuProfilerStart", "cuProfilerStop"),
            ("libcudart.so.13", "cudaProfilerStart", "cudaProfilerStop"), ("libcudart.so.12", "cudaProfilerStart", "cudaProfilerStop"),
        })
        {
            if (!NativeLibrary.TryLoad(lib, out IntPtr handle)) continue;
            if (!NativeLibrary.TryGetExport(handle, start, out IntPtr a) || !NativeLibrary.TryGetExport(handle, stop, out IntPtr b)) continue;
            s_start = Marshal.GetDelegateForFunctionPointer<Call>(a);
            s_stop = Marshal.GetDelegateForFunctionPointer<Call>(b);
            s_api = $"{lib} {start}";
            return;
        }
    }

    public static void Start()
    {
        Load();
        if (s_start == null)
        {
            Console.Error.WriteLine("WARN --profile-range: no CUDA profiler API (libcuda.so.1, libcudart) here: no capture range is marked");
            return;
        }
        Console.Error.WriteLine($"INFO profiler range start ({s_api}) = {s_start()}");
    }

    public static void Stop()
    {
        if (s_stop != null) Console.Error.WriteLine($"INFO profiler range stop = {s_stop()}");
    }
}
