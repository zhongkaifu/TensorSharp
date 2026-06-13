// Standalone GatedDeltaNet/MTP decode benchmark, profilable directly under nsys.
// Mirrors InferenceWeb.Tests.Qwen36MtpTests.Mtp_PerfBench_SpecVsBaseline so the
// numbers are comparable, but as a plain exe (no VSTest host) so external profilers
// can attach to the CUDA work.
using System.Diagnostics;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

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

string prompt = "Write a short story about a robot learning to paint. " +
                "Once upon a time, in a small workshop at the edge of the city,";
int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();

// Warmup.
model.ResetKVCache();
model.ForwardRefill(tokens);
model.Forward(new[] { tokens[^1] });

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
}
