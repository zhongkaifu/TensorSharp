using System.Collections;
using System.Diagnostics;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

var options = new Dictionary<string, string>();
for (int i = 0; i < args.Length; i += 2)
{
    if (!args[i].StartsWith("--") || i + 1 == args.Length || !options.TryAdd(args[i][2..], args[i + 1]))
        throw new ArgumentException("Use unique --name value arguments.");
}
string Required(string name) => options.TryGetValue(name, out var value) ? value : throw new ArgumentException("Missing --" + name);
string modelPath = Path.GetFullPath(Required("model")), output = Path.GetFullPath(Required("output"));
string mode = Required("spec-mode");
if (mode is not ("off" or "on" or "ngram" or "unset")) throw new ArgumentException("spec-mode: off, on, ngram, unset");
int context = int.Parse(options.GetValueOrDefault("context", "4096"));
int iterations = int.Parse(options.GetValueOrDefault("iterations", "2"));
if (context < 256 || iterations < 1 || iterations > 10) throw new ArgumentException("Invalid context/iterations.");
if (Directory.Exists(output)) throw new IOException("Use a fresh output directory.");
Directory.CreateDirectory(output);
Environment.SetEnvironmentVariable("TS_SPEC", mode == "unset" ? null : mode == "off" ? "0" : "1");
Environment.SetEnvironmentVariable("TS_SPEC_TYPE", mode == "ngram" ? "ngram" : "auto");
Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
Environment.SetEnvironmentVariable("KV_CACHE_DTYPE", "f16");
var json = new JsonSerializerOptions { WriteIndented = true };
void Write(string name, object data) => File.WriteAllText(Path.Combine(output, name), JsonSerializer.Serialize(data, json));
string Hash(string path) { using var stream = File.OpenRead(path); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }
var rows = new List<object>(); var timings = new List<object>();
Qwen35Model? model = null; Exception? failure = null; bool disposed = false;
object? loaded = null;
int[] prompt = [], teacher = [];
try
{
    var load = Stopwatch.StartNew();
    model = new Qwen35Model(modelPath, BackendType.GgmlCuda);
    load.Stop();
    prompt = model.Tokenizer.Encode("Q: Explain why the Moon has phases.\nA:", addSpecial: false).ToArray();
    teacher = model.Tokenizer.Encode("The Moon reflects sunlight. Its orbit changes the illuminated portion visible from Earth. A full orbit repeats these phases over time.", addSpecial: false).Take(16).ToArray();
    if (teacher.Length != 16 || prompt.Length + teacher.Length > context) throw new InvalidOperationException("Unexpected tokenizer/context.");
    var binding = BindingFlags.NonPublic | BindingFlags.Instance;
    var weightNames = new[] { "_weights", "_quantWeights" }.SelectMany(name =>
        ((IDictionary)typeof(ModelBase).GetField(name, binding)!.GetValue(model)!).Keys.Cast<string>()).Order().ToArray();
    loaded = new {
        load_ms = load.Elapsed.TotalMilliseconds, model.HasDraftHead, model.Config.NumLayers,
        total_layer_count = typeof(Qwen35Model).GetProperty("TotalLayerCount", binding)!.GetValue(model),
        omitted_checkpoint_bytes = typeof(ModelBase).GetProperty("OmittedCheckpointWeightBytes")?.GetValue(model),
        omitted_tensor_count = typeof(ModelBase).GetProperty("OmittedCheckpointWeightCount")?.GetValue(model),
        weight_names = weightNames,
        native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
            .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
            .Select(m => new { path = m.FileName, sha256 = Hash(m.FileName) }).ToArray(),
        managed = new { models_sha256 = Hash(typeof(ModelBase).Assembly.Location), runtime_sha256 = Hash(typeof(SpeculationOptions).Assembly.Location), probe_sha256 = Hash(typeof(Program).Assembly.Location) }
    };
    Write("loaded.json", loaded);
    using var data = File.Create(Path.Combine(output, "rows.f32"));
    void Capture(float[] logits, string stage, int iteration, List<int> history)
    {
        if (logits.Length != model.Config.VocabSize || logits.Any(x => !float.IsFinite(x))) throw new InvalidDataException("Nonfinite/incomplete logits.");
        var bytes = MemoryMarshal.AsBytes(logits.AsSpan());
        rows.Add(new { stage, iteration, input_tokens = history.ToArray(), byte_offset = data.Position, elements = logits.Length,
            sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant(), argmax = Enumerable.Range(0, logits.Length).MaxBy(i => logits[i]) });
        data.Write(bytes); data.Flush();
        Write("rows.json", new { format = "f32le", data_path = Path.Combine(output, "rows.f32"), rows });
    }
    for (int iteration = 0; iteration < iterations; iteration++)
    {
        model.ResetKVCache(); var history = prompt.ToList(); var timer = Stopwatch.StartNew();
        var logits = model.ForwardRefill(prompt); double prefillMs = timer.Elapsed.TotalMilliseconds;
        Capture(logits, "prefill", iteration, history);
        double decodeMs = 0;
        foreach (int token in teacher)
        {
            timer.Restart(); logits = model.Forward(new[] { token }); decodeMs += timer.Elapsed.TotalMilliseconds;
            history.Add(token); Capture(logits, "decode", iteration, history);
        }
        timings.Add(new { iteration, prefill_ms = prefillMs, decode_ms = decodeMs });
    }
    if (mode == "on")
    {
        if (!model.HasDraftHead) throw new InvalidOperationException("Enabled MTP head unavailable.");
        model.ResetKVCache();
        var decoder = new SpeculativeDecoder(model, 2);
        var tokens = decoder.GenerateGreedy(prompt, 16);
        Write("speculative.json", new { tokens, decoder.TokensDrafted, decoder.TokensAccepted, decoder.VerifySteps, decoder.RollbackSteps });
        if (tokens.Count != 16 || decoder.TokensDrafted <= 0) throw new InvalidOperationException("MTP did not execute.");
    }
    if (mode == "ngram" && options.GetValueOrDefault("exercise-ngram", "false") == "true")
    {
        int[] repeatPrompt = model.Tokenizer.Encode("Repeat the following comma-separated sequence without explanation:\n" +
            string.Concat(Enumerable.Repeat("1,2,3,", 20)) + "\nOutput:\n1,2,3,", addSpecial: false).ToArray();
        var decoder = new SpeculativeDecoder(model, new SpeculationOptions {
            Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 2, MaxDraftTokensExplicit = true
        }) { AdaptiveSpeculation = false };
        var tokens = decoder.GenerateGreedy(repeatPrompt, 64);
        Write("ngram.json", new { prompt_tokens = repeatPrompt, tokens, decoder.TokensDrafted, decoder.TokensAccepted,
            decoder.VerifySteps, decoder.RollbackSteps });
        if (tokens.Count != 64 || decoder.TokensDrafted <= 0) throw new InvalidOperationException("N-gram did not exercise draft/verify.");
    }
}
catch (Exception error) { failure = error; }
finally
{
    try { model?.Dispose(); disposed = model != null; }
    catch (Exception error) { failure = failure == null ? error : new AggregateException(failure, error); }
    Write("report.json", new { run_complete = true, passed = failure == null, error = failure?.ToString(), disposed,
        model_path = modelPath, model_bytes = new FileInfo(modelPath).Length, model_sha256 = Hash(modelPath),
        mode, context, iterations, prompt, teacher, loaded, timings,
        environment = new[] { "TS_SPEC", "TS_SPEC_TYPE", "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "TS_GGML_CPU_THREADS", "GGML_CUDA_DISABLE_FUSION", "GGML_CUDA_P2P" }.ToDictionary(n => n, Environment.GetEnvironmentVariable),
        limitations = "Matched-history full-logit and draft execution diagnostic; timings include graph warmup and capture interference, not throughput or independent language quality." });
}
return failure == null ? 0 : 1;
