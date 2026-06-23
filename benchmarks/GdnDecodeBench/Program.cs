// Standalone GatedDeltaNet/MTP decode benchmark, profilable directly under nsys.
// Mirrors InferenceWeb.Tests.Qwen36MtpTests.Mtp_PerfBench_SpecVsBaseline so the
// numbers are comparable, but as a plain exe (no VSTest host) so external profilers
// can attach to the CUDA work.
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

static int EnvInt(string name, int fallback)
{
    string s = Environment.GetEnvironmentVariable(name);
    return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
}

static int Argmax(float[] v)
{
    int best = 0;
    for (int i = 1; i < v.Length; i++)
        if (v[i] > v[best]) best = i;
    return best;
}

string modelPath = Environment.GetEnvironmentVariable("TS_MTP_MODEL")
    ?? @"C:\Works\models\mtp\Qwen3.6-27B-UD-IQ2_XXS.gguf";
BackendType backend = (Environment.GetEnvironmentVariable("TS_MTP_BACKEND") ?? "cuda").ToLowerInvariant() switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    "cpu" => BackendType.Cpu,
    _ => BackendType.Cuda,
};
int maxNew = EnvInt("TS_MTP_NEW_TOKENS", 48);
int maxDraft = EnvInt("TS_MTP_DRAFT", 8);
string mode = Environment.GetEnvironmentVariable("TS_BENCH_MODE") ?? "both"; // baseline | spec | both

Console.WriteLine($"[gdn-bench] loading {Path.GetFileName(modelPath)} backend={backend} maxNew={maxNew} draft={maxDraft} mode={mode}");
using var model = (Qwen35Model)ModelBase.Create(modelPath, backend);
Console.WriteLine($"[gdn-bench] HasMtp={model.HasMtp}");

if (mode is "concurrent")
{
    await RunConcurrent(model, maxNew);
    return;
}

string prompt = Environment.GetEnvironmentVariable("TS_MTP_PROMPT") ??
                "Write a short story about a robot learning to paint. " +
                "Once upon a time, in a small workshop at the edge of the city,";
int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();

// Warmup.
model.ResetKVCache();
model.ForwardRefill(tokens);
model.Forward(new[] { tokens[^1] });

if (mode is "matmul")
{
    // Isolate the quantized-matmul batch scaling (no recurrent/attention layers).
    foreach (string which in new[] { "gateup", "down", "lmhead" })
    {
        Console.WriteLine($"[gdn-bench] matmul '{which}':");
        foreach (int b in new[] { 1, 2, 4, 8 })
        {
            double ms = model.DebugTimeQuantMatmul(which, b, 50, out int gt, out long inD, out long outD);
            Console.WriteLine($"    B={b}: {ms:F3} ms/call ({ms / b:F3} ms/row)  type={gt} in={inD} out={outD}");
        }
    }
    return;
}

if (mode is "scaling")
{
    // Isolate how a single trunk forward scales with batch width B (the verify
    // window). If GDN/attention amortize, forward(B) should be ~flat per token.
    model.ResetKVCache();
    model.ForwardRefill(tokens);
    int basePos = tokens.Length;
    int[] bs = { 1, 2, 4, 8 };
    foreach (int b in bs)
    {
        // Fresh KV position each trial so attention KV length is comparable.
        int reps = 20;
        var probe = new int[b];
        for (int i = 0; i < b; i++) probe[i] = tokens[i % tokens.Length];
        // Warmup
        model.SpecForward(probe, new float[b * model.Config.HiddenSize], new float[model.Config.VocabSize], allLogitsRows: false);
        var sw2 = Stopwatch.StartNew();
        for (int r = 0; r < reps; r++)
            model.SpecForward(probe, new float[b * model.Config.HiddenSize], new float[model.Config.VocabSize], allLogitsRows: false);
        sw2.Stop();
        double msPer = sw2.Elapsed.TotalMilliseconds / reps;
        Console.WriteLine($"[gdn-bench] forward(B={b}): {msPer:F1} ms/call = {msPer / b:F1} ms/token-in-batch");
    }
    return;
}

List<int> baselineTokens = null;
if (mode is "baseline" or "both")
{
    model.ResetKVCache();
    var baseline = new List<int>();
    float[] logits = model.ForwardRefill(tokens);
    var sw = Stopwatch.StartNew();
    int t = Argmax(logits);
    baseline.Add(t);
    for (int i = 1; i < maxNew; i++)
    {
        logits = model.Forward(new[] { t });
        t = Argmax(logits);
        baseline.Add(t);
    }
    sw.Stop();
    double tps = (maxNew - 1) / sw.Elapsed.TotalSeconds;
    Console.WriteLine($"[gdn-bench] baseline decode {sw.Elapsed.TotalSeconds:F2}s = {tps:F2} tok/s ({1000.0 * sw.Elapsed.TotalSeconds / maxNew:F1} ms/token)");
    string baseText = model.Tokenizer.Decode(baseline).Replace("\n", "\\n");
    Console.WriteLine($"[gdn-bench] baseline text: \"{(baseText.Length > 200 ? baseText.Substring(0, 200) : baseText)}\"");
    baselineTokens = baseline;
}

if (mode is "spec" or "both")
{
    var spec = new MtpSpeculativeDecoder(model, maxDraft);
    string pminEnv = Environment.GetEnvironmentVariable("TS_MTP_PMIN");
    if (!string.IsNullOrEmpty(pminEnv) && float.TryParse(pminEnv, out float pmin))
        spec.MinDraftProb = pmin;
    model.ResetSpecLayerTimings();
    List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);
    double specTps = (specTokens.Count - 1) / spec.LastDecodeSeconds;
    double msPerTick = 1000.0 / Stopwatch.Frequency;
    double decodeMs = spec.LastDecodeSeconds * 1000;
    Console.WriteLine($"[gdn-bench] spec     decode {spec.LastDecodeSeconds:F2}s = {specTps:F2} tok/s ({decodeMs / specTokens.Count:F1} ms/token) " +
        $"accept={spec.AcceptanceRate:P0} verify={spec.VerifySteps} plain={spec.PlainSteps} rollbacks={spec.RollbackSteps}");
    Console.WriteLine($"[gdn-bench] split: attn={model.SpecAttnLayerTicks * msPerTick:F0}ms ({100 * model.SpecAttnLayerTicks * msPerTick / decodeMs:F1}%) " +
        $"gdn={model.SpecRecurrentLayerTicks * msPerTick:F0}ms ({100 * model.SpecRecurrentLayerTicks * msPerTick / decodeMs:F1}%) " +
        $"lmhead={model.SpecLmHeadTicks * msPerTick:F0}ms ({100 * model.SpecLmHeadTicks * msPerTick / decodeMs:F1}%)");
    var st = spec.Stats;
    Console.WriteLine($"[gdn-bench] phases: draft={st.DraftMs:F0}ms ({100 * st.DraftMs / decodeMs:F1}%) " +
        $"verify={st.VerifyMs:F0}ms ({100 * st.VerifyMs / decodeMs:F1}%) " +
        $"snapshot={st.SnapshotMs:F0}ms ({100 * st.SnapshotMs / decodeMs:F1}%) " +
        $"rollback={st.RollbackMs:F0}ms ({100 * st.RollbackMs / decodeMs:F1}%) " +
        $"catchup={st.CatchUpMs:F0}ms ({100 * st.CatchUpMs / decodeMs:F1}%) " +
        $"plain={st.PlainMs:F0}ms ({100 * st.PlainMs / decodeMs:F1}%)");
    Console.WriteLine($"[gdn-bench] counts: drafted={st.TokensDrafted} accepted={st.TokensAccepted} " +
        $"verifySteps={st.VerifySteps} plainSteps={st.PlainSteps} rollbackSteps={st.RollbackSteps} " +
        $"avgWindow={(st.VerifySteps > 0 ? (double)st.TokensDrafted / st.VerifySteps : 0):F2}");
    string specText = model.Tokenizer.Decode(specTokens).Replace("\n", "\\n");
    Console.WriteLine($"[gdn-bench] spec text:     \"{(specText.Length > 200 ? specText.Substring(0, 200) : specText)}\"");
    if (baselineTokens != null)
    {
        int n = Math.Min(baselineTokens.Count, specTokens.Count);
        int diverge = -1;
        for (int i = 0; i < n; i++)
            if (baselineTokens[i] != specTokens[i]) { diverge = i; break; }
        Console.WriteLine(diverge < 0
            ? $"[gdn-bench] correctness: spec == baseline for all {n} compared tokens"
            : $"[gdn-bench] correctness: first divergence at token {diverge}/{n} (FP drift between batched-verify and sequential-decode is expected past near-ties)");
    }
    model.PrintTimingStats();
}

// ---------------------------------------------------------------------------
// Concurrent continuous-batching repro. Submits N requests to the
// InferenceEngine and measures aggregate throughput. Used to reproduce /
// debug the N>=2 true-batched (ForwardBatch) path on ggml_cuda.
//   TS_BATCH_N        number of concurrent requests (default 2)
//   TS_BATCH_TIMEOUT  per-run wall timeout in seconds (default 90)
// ---------------------------------------------------------------------------
static async Task RunConcurrent(Qwen35Model model, int maxNew)
{
    int n = EnvInt("TS_BATCH_N", 2);
    int timeoutSec = EnvInt("TS_BATCH_TIMEOUT", 90);
    int blockSize = 256;

    string[] prompts =
    {
        "Q: What is two plus two? Answer in one short sentence.\nA:",
        "Q: Name three primary colors as a comma separated list.\nA:",
        "Q: Who wrote the play Hamlet? Answer in one sentence.\nA:",
        "Q: What is the boiling point of water at sea level in Celsius?\nA:",
        "Q: Translate 'good morning' into Spanish.\nA:",
        "Q: What is the capital city of France?\nA:",
    };

    var cfg = new SchedulerConfig
    {
        MaxNumBatchedTokens = 4096,
        MaxNumRunningSequences = Math.Max(n, 8),
        MaxPrefillChunkSize = 1024,
        NumBlocks = 256,
        BlockSize = blockSize,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = blockSize,
    };
    using var engine = new InferenceEngine(model, cfg);

    Console.WriteLine($"[gdn-bench] concurrent: N={n} maxNew={maxNew} timeout={timeoutSec}s " +
        $"TS_QWEN35_BATCHED={Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED")} " +
        $"N1FAST={Environment.GetEnvironmentVariable("TS_BATCHED_N1_FAST_PATH")}");

    async Task<(int count, bool timedOut)> RunOne(int i)
    {
        var promptTokens = model.Tokenizer.Encode(prompts[i % prompts.Length], addSpecial: false).ToArray();
        var seq = new SequenceState($"req-{i}", promptTokens, maxNew, blockSize, SamplingConfig.Greedy);
        using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(timeoutSec));
        var handle = engine.SubmitRequest(seq, cts.Token);
        int count = 0;
        bool timedOut = false;
        var outToks = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync(cts.Token))
            { count++; outToks.Add(tok); }
        }
        catch (OperationCanceledException) { timedOut = true; }
        catch { }
        string text = model.Tokenizer.Decode(outToks).Replace("\n", "\\n");
        Console.WriteLine($"[gdn-bench]   req-{i} text: \"{(text.Length > 120 ? text.Substring(0, 120) : text)}\"");
        return (count, timedOut);
    }

    var sw = Stopwatch.StartNew();
    var tasks = new List<Task<(int count, bool timedOut)>>();
    for (int i = 0; i < n; i++) tasks.Add(RunOne(i));
    var results = await Task.WhenAll(tasks);
    sw.Stop();

    int totalOut = 0;
    bool anyTimeout = false;
    for (int i = 0; i < results.Length; i++)
    {
        totalOut += results[i].count;
        anyTimeout |= results[i].timedOut;
        Console.WriteLine($"[gdn-bench]   req-{i}: tokens={results[i].count} timedOut={results[i].timedOut}");
    }
    double aggTps = totalOut / sw.Elapsed.TotalSeconds;
    Console.WriteLine($"[gdn-bench] concurrent N={n}: wall={sw.Elapsed.TotalSeconds:F2}s totalOut={totalOut} " +
        $"aggTps={aggTps:F2} perStream={aggTps / n:F2} anyTimeout={anyTimeout}");
    if (anyTimeout)
        Console.WriteLine("[gdn-bench] !!! TIMEOUT/HANG detected — batched path likely deadlocked.");
}
