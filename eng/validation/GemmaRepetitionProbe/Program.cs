// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Encodings.Web;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

var options = new Dictionary<string, string>(StringComparer.Ordinal);
for (int i = 0; i < args.Length; i++)
{
    if (!args[i].StartsWith("--", StringComparison.Ordinal) || i + 1 >= args.Length)
        throw new ArgumentException("Use --name value pairs.");
    options.Add(args[i][2..], args[++i]);
}
string Option(string name, string fallback) => options.GetValueOrDefault(name, fallback);
string path = Path.GetFullPath(options["model"]), output = Path.GetFullPath(options["output"]);
string mode = Option("mode", "direct"), backendName = Option("backend", "ggml_cuda");
string sampling = Option("sampling", "production");
if (sampling is not ("production" or "raw")) throw new ArgumentException("Sampling must be production or raw.");
if (mode == "engine" && sampling == "raw") throw new ArgumentException("Engine mode always respects the model generation contract.");
string prompt = Option("prompt", "请详细介绍最终幻想7");
float temperature = float.Parse(Option("temperature", "0"), System.Globalization.CultureInfo.InvariantCulture);
int topK = int.Parse(Option("top-k", "0")), seed = int.Parse(Option("seed", "17"));
float topP = float.Parse(Option("top-p", "1"), System.Globalization.CultureInfo.InvariantCulture);
if (!float.IsFinite(temperature) || temperature < 0 || topK < 0 || !(topP > 0 && topP <= 1))
    throw new ArgumentException("Invalid sampling parameters.");
if (sampling == "raw" && temperature != 0) throw new ArgumentException("Raw mode is an argmax-only arithmetic diagnostic.");
var samplingConfig = new SamplingConfig { Temperature = temperature, TopK = topK, TopP = topP,
    MinP = 0, Seed = seed, RepetitionPenalty = 1, PresencePenalty = 0, FrequencyPenalty = 0 };
bool thinking = bool.Parse(Option("thinking", "false")), saveLogits = bool.Parse(Option("logits", "false"));
int maxNew = int.Parse(Option("max-new", "1024")), context = int.Parse(Option("context", "2048"));
if (mode is not ("metadata" or "direct" or "engine")) throw new ArgumentException("Unknown mode.");
if (maxNew <= 0 || context <= 0) throw new ArgumentOutOfRangeException("Token limits must be positive.");
if (mode != "direct" && (saveLogits || options.ContainsKey("teacher")))
    throw new ArgumentException("Full logits and teacher replay require --mode direct.");
BackendType backend = backendName switch
{
    "ggml_cuda" => BackendType.GgmlCuda,
    "ggml_cpu" => BackendType.GgmlCpu,
    "cuda" => BackendType.Cuda,
    _ => throw new ArgumentException("Backend must be ggml_cuda, ggml_cpu or cuda."),
};
if (Directory.Exists(output) && Directory.EnumerateFileSystemEntries(output).Any())
    throw new IOException("Use a fresh output directory; prior evidence must not be overwritten.");
Directory.CreateDirectory(output);
var json = new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
void Write(string name, object value) => File.WriteAllText(Path.Combine(output, name), JsonSerializer.Serialize(value, json));
string Hash(string file) { using var stream = File.OpenRead(file); return Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant(); }

ITokenizer tokenizer;
int[] promptTokens;
using (var gguf = new GgufFile(path))
{
    string? architecture = gguf.GetString("general.architecture");
    if (architecture != "gemma4") throw new ArgumentException("This probe requires Gemma4.");
    tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
    string template = gguf.GetString("tokenizer.chat_template") ?? "";
    var messages = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    promptTokens = renderer.RenderToTokens(tokenizer, template, messages, architecture,
        addGenerationPrompt: true, enableThinking: thinking).ToArray();
    string rendered = ChatTemplate.RenderFromGgufTemplate(template, messages,
        addGenerationPrompt: true, architecture: architecture, enableThinking: thinking);
    Write("prompt.json", new
    {
        ModelPath = path, ModelBytes = new FileInfo(path).Length, ModelSha256 = Hash(path),
        Prompt = prompt, Thinking = thinking, ChatTemplate = template, RenderedPrompt = rendered,
        PromptTokens = promptTokens, DecodedPromptTokens = tokenizer.Decode(promptTokens.ToList()),
        TokenizerType = tokenizer.GetType().FullName, tokenizer.BosTokenId,
        LeadingBosCount = promptTokens.TakeWhile(t => t == tokenizer.BosTokenId).Count(),
        EosTokenIds = Enumerable.Range(0, tokenizer.VocabSize).Where(tokenizer.IsEos).ToArray(),
        SuppressedTokens = tokenizer.SuppressedTokenIds.Select(id => new { Id = id, Text = tokenizer.Vocab[id] }).ToArray(),
        Metadata = gguf.Metadata.Where(p => p.Key.StartsWith("general.") || p.Key.StartsWith("gemma4.")
            || p.Key is "tokenizer.ggml.model" or "tokenizer.ggml.add_bos_token" or "tokenizer.ggml.bos_token_id" or "tokenizer.ggml.eos_token_id" or "tokenizer.ggml.suppress_tokens")
            .ToDictionary(p => p.Key, p => p.Value),
        Tensors = gguf.Tensors.Values.Select(t => new { t.Name, Type = t.Type.ToString(), t.Shape,
            Bytes = gguf.GetTensorByteCount(t) }).ToArray(),
    });
    if (promptTokens.Length + maxNew > context) throw new ArgumentException("Prompt plus max-new exceeds context; no context shifting is performed.");
    if (tokenizer.BosTokenId < 0 || promptTokens.TakeWhile(t => t == tokenizer.BosTokenId).Count() != 1
        || !rendered.Contains(prompt, StringComparison.Ordinal))
        throw new InvalidDataException("Prompt did not retain exactly one leading BOS and the requested user text.");
}
if (mode == "metadata") return 0;

Environment.SetEnvironmentVariable("MAX_CONTEXT", context.ToString());
int[]? teacher = null;
if (options.TryGetValue("teacher", out string? teacherPath))
{
    using var document = JsonDocument.Parse(File.ReadAllText(teacherPath));
    var root = document.RootElement;
    if (root.ValueKind != JsonValueKind.Array)
        root = root.TryGetProperty("GeneratedTokens", out var own) ? own : root.GetProperty("tokens");
    teacher = root.EnumerateArray().Select(t => t.GetInt32()).ToArray();
    if (teacher.Length == 0 || teacher.Any(t => t < 0 || t >= tokenizer.VocabSize))
        throw new InvalidDataException("Teacher tokens are empty or outside the checkpoint vocabulary.");
    maxNew = Math.Min(maxNew, teacher.Length);
}
var tokens = new List<int>();
var steps = new List<object>();
var timer = Stopwatch.StartNew();
string stop = "max-new", text = "";
Exception? failure = null;
bool finite = true;
object? native = null;
object? engineCompletion = null;
ModelBase? model = null;
try
{
    model = ModelBase.Create(path, backend);
    native = Process.GetCurrentProcess().Modules.Cast<ProcessModule>()
        .Where(m => Path.GetFileName(m.FileName).Contains("GgmlOps", StringComparison.OrdinalIgnoreCase))
        .Select(m => new { Path = m.FileName, Sha256 = Hash(m.FileName) }).ToArray();
    if (mode == "engine")
    {
        // Exercise the production scheduler, but expose rather than hide loops.
        // No speculative proposals or repetition termination/penalties are used.
        var cfg = new SchedulerConfig { StopRepetition = false, Speculation = SpeculationOptions.Disabled,
            NumBlocks = Math.Max(256, (context + 15) / 16), BlockSize = 16 };
        using var engine = new InferenceEngine(model, cfg);
        var sequence = new SequenceState("gemma-repetition", promptTokens.ToList(), maxNew, engine.PoolStats.blockSize,
            samplingConfig: samplingConfig);
        await foreach (int token in engine.SubmitRequest(sequence).Tokens.ReadAllAsync()) tokens.Add(token);
        if (sequence.Error != null) throw sequence.Error;
        engineCompletion = new { sequence.FinishReason, SequenceOutputTokens = sequence.OutputTokens.ToArray(),
            StopToken = sequence.FinishReason == "eos" && sequence.OutputTokens.Count > 0 ? (int?)sequence.OutputTokens[^1] : null };
        stop = sequence.Status.ToString();
    }
    else
    {
        var sampler = new TokenSampler(samplingConfig, tokenizer.SuppressedTokenIds);
        var greedySampler = new TokenSampler(SamplingConfig.Greedy, tokenizer.SuppressedTokenIds);
        using var full = saveLogits ? new FileStream(Path.Combine(output, "logits.f32"), FileMode.CreateNew) : null;
        for (int step = 0; step < maxNew; step++)
        {
            long started = Stopwatch.GetTimestamp();
            float[] logits = model.Forward(step == 0 ? promptTokens : [tokens[^1]]);
            double milliseconds = Stopwatch.GetElapsedTime(started).TotalMilliseconds;
            if (logits.Length != tokenizer.VocabSize || logits.Any(x => !float.IsFinite(x)))
            {
                finite = false;
                throw new InvalidDataException($"Nonfinite or incorrectly sized logit row at step {step}.");
            }
            full?.Write(MemoryMarshal.AsBytes(logits.AsSpan()));
            int[] top = Enumerable.Range(0, logits.Length).OrderByDescending(i => logits[i]).ThenBy(i => i).Take(20).ToArray();
            int greedy = top[0];
            int production = greedySampler.Sample((float[])logits.Clone(), tokens);
            int chosen = teacher == null ? (sampling == "raw" ? greedy : sampler.Sample((float[])logits.Clone(), tokens)) : teacher[step];
            double max = logits[greedy], denominator = 0, weighted = 0;
            foreach (float value in logits) { double exp = Math.Exp(value - max); denominator += exp; weighted += exp * (value - max); }
            tokens.Add(chosen);
            steps.Add(new { Step = step, ContextPosition = promptTokens.Length + step, Milliseconds = milliseconds,
                Token = chosen, GreedyToken = greedy, TeacherForced = teacher != null,
                ProductionGreedyToken = production,
                ChosenProbability = Math.Exp(logits[chosen] - max) / denominator,
                EntropyNats = Math.Log(denominator) - weighted / denominator,
                Top = top.Select(t => new { Token = t, Text = tokenizer.Decode([t]), Logit = logits[t],
                    Probability = Math.Exp(logits[t] - max) / denominator }).ToArray(),
                RepeatedSuffix = Repetition(tokens),
            });
            // Checkpoint every 32 rows preserves the onset if later native work fails.
            if ((step + 1) % 32 == 0) { Write("steps.json", steps); Write("tokens.json", tokens); }
            if (tokenizer.IsEos(chosen)) { stop = "eos"; break; }
        }
    }
    text = tokenizer.Decode(tokens);
}
catch (Exception ex) { failure = ex; stop = "error"; }
finally
{
    try { model?.Dispose(); }
    catch (Exception cleanup) { failure = failure == null ? cleanup : new AggregateException(failure, cleanup); stop = "cleanup-error"; }
    timer.Stop();
    if (text.Length == 0 && tokens.Count > 0) text = tokenizer.Decode(tokens);
    Write("steps.json", steps);
    Write("tokens.json", tokens);
    File.WriteAllText(Path.Combine(output, "output.txt"), text);
    Write("report.json", new
    {
        Qualification = "Diagnostic only: finite execution and absence of an obvious suffix loop do not establish semantic or numerical correctness. Compare an independent implementation and inspect the answer.",
        Mode = mode, Backend = backendName, Thinking = thinking, PromptTokens = promptTokens,
        GeneratedTokens = tokens, Text = text, Stop = stop, Finite = mode == "direct" && steps.Count > 0 ? (bool?)finite : null,
        Error = failure?.ToString(),
        EngineCompletion = engineCompletion,
        Teacher = options.GetValueOrDefault("teacher"), FullLogitsSaved = saveLogits, VocabSize = tokenizer.VocabSize,
        FinalRepeatedSuffix = Repetition(tokens), DistinctTokens = tokens.Distinct().Count(),
        ElapsedMilliseconds = timer.Elapsed.TotalMilliseconds, Native = native,
        ModelsSha256 = Hash(typeof(ModelBase).Assembly.Location), ProbeSha256 = Hash(typeof(Program).Assembly.Location),
        Sampling = sampling == "raw" ? "raw argmax diagnostic; bypasses model generation exclusions; not a production quality baseline"
            : "production sampling (or exact teacher tokens); GGUF generation exclusions enabled; repetition/presence/frequency penalties disabled; engine repetition stop disabled; speculative decode disabled",
        SamplingParameters = new { Temperature = temperature, TopK = topK, TopP = topP, MinP = 0, Seed = seed,
            RepetitionPenalty = 1, PresencePenalty = 0, FrequencyPenalty = 0 },
        ModelSuppressedTokenIds = tokenizer.SuppressedTokenIds,
        Environment = new[] { "MAX_CONTEXT", "KV_CACHE_DTYPE", "TS_KV_INITIAL_TOKENS", "TS_PREFILL_CHUNK",
            "TS_GGML_SKIP_PRELOAD", "TENSORSHARP_TP_DEGREE", "TENSORSHARP_LAYER_SPLIT_DEGREE", "GGML_CUDA_DISABLE_FUSION", "TS_GEMMA4_TENSOR_DUMP",
            "TS_GEMMA4_DIAGNOSTIC_PAD_LOCAL_KV" }
            .ToDictionary(n => n, Environment.GetEnvironmentVariable),
    });
}
return failure == null ? 0 : 1;

static object? Repetition(IReadOnlyList<int> tokens)
{
    for (int period = 1; period <= 64 && period * 3 <= tokens.Count; period++)
    {
        int matched = period;
        while (matched < tokens.Count && tokens[tokens.Count - 1 - matched] == tokens[tokens.Count - 1 - matched + period]) matched++;
        if (matched >= Math.Max(24, period * 3))
            return new { Period = period, Tokens = matched, Start = tokens.Count - matched,
                Cycles = matched / period, Pattern = tokens.Skip(tokens.Count - period).ToArray() };
    }
    return null;
}
