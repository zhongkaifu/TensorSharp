// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Long-context quality of an arithmetic change, between arms: runs the cases of a case file (eng/validation/
// long-context-quality.py cases) on one loaded model and writes one JSONL record per case and arm. Arms that differ in an
// in-process seam (the CUDA executor's prompt attention: f16 / 8bit / default) run in one process, interleaved case by case;
// arms that differ in what is fixed at load (--kv-cache-dtype) run as separate processes over the same case file, paired by
// case id in the report (eng/validation/long-context-quality.py report). See README.md.
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using TensorSharp.Runtime;
using LongContextQualityProbe;

const string ProbeVersion = "long-context-quality-probe/1";
var options = new Dictionary<string, string>(StringComparer.Ordinal);
var flags = new HashSet<string>(StringComparer.Ordinal) { "resume" };
var valued = new HashSet<string>(StringComparer.Ordinal)
{
    "model", "cases", "output", "backend", "kv-cache-dtype", "arms", "reference", "kinds", "groups", "ids", "max-tokens",
    "prefill-chunk", "score-chunk", "topk", "label",
};
for (int i = 0; i < args.Length; i++)
{
    string a = args[i];
    if (!a.StartsWith("--", StringComparison.Ordinal)) return Usage($"unexpected argument '{a}'");
    string key = a[2..];
    if (flags.Contains(key)) { options[key] = "1"; continue; }
    if (!valued.Contains(key)) return Usage($"unknown option {a}");
    if (i + 1 >= args.Length) return Usage($"{a} needs a value");
    options[key] = args[++i];
}
foreach (string required in new[] { "model", "cases", "output" })
    if (!options.ContainsKey(required)) return Usage($"--{required} is required");

string modelPath = Path.GetFullPath(options["model"]);
string outputPath = Path.GetFullPath(options["output"]);
string backendName = options.GetValueOrDefault("backend", "cuda");
BackendType backend = backendName switch
{
    "cuda" => BackendType.Cuda, "ggml_cuda" => BackendType.GgmlCuda, "ggml_cpu" => BackendType.GgmlCpu, "ggml_metal" => BackendType.GgmlMetal,
    "ggml_vulkan" => BackendType.GgmlVulkan, "mlx" => BackendType.Mlx, "cpu" => BackendType.Cpu,
    _ => throw new ArgumentException($"--backend {backendName}: cuda, ggml_cuda, ggml_cpu, ggml_metal, ggml_vulkan, mlx or cpu"),
};
Arm[] arms = options.GetValueOrDefault("arms", "default").Split(',', StringSplitOptions.RemoveEmptyEntries).Select(Arm.Parse).ToArray();
if (arms.Select(a => a.Name).Distinct().Count() != arms.Length) return Usage("--arms: arm names must differ");
int prefillChunk = Int("prefill-chunk", 2048), scoreChunk = Int("score-chunk", 512), topK = Int("topk", 32);
int maxTokens = Int("max-tokens", int.MaxValue);
if (prefillChunk < 1 || scoreChunk <= 16 || topK < 0) return Usage("--prefill-chunk >= 1, --score-chunk > 16 (a marked chunk), --topk >= 0");
bool resume = options.ContainsKey("resume");

CaseFile caseFile = CaseFile.Load(options["cases"]);
HashSet<string> kinds = Set("kinds"), groups = Set("groups");
Regex idFilter = options.TryGetValue("ids", out string idPattern) ? new Regex(idPattern) : null;
List<Case> selected = caseFile.Cases.Where(c => (kinds == null || kinds.Contains(c.Kind)) && (groups == null || (c.Group != null && groups.Contains(c.Group)))
    && (idFilter == null || idFilter.IsMatch(c.Id))).ToList();
if (selected.Count == 0) return Usage("no case of the case file passes --kinds / --groups / --ids");

// Records already written (--resume skips the ones that ran, ok or failed, and retries the skipped ones, so a first pass under
// --max-tokens can be completed later; without it an existing output is refused, so two sessions never mix silently). The
// report reads the last record of a case and arm.
var done = new HashSet<(string Case, string Arm)>();
if (File.Exists(outputPath) && new FileInfo(outputPath).Length > 0)
{
    if (!resume) return Usage($"{outputPath} exists: pass --resume to continue it, or name a new output");
    foreach (JsonNode r in ReadRecords(outputPath))
        if (r["status"]?.GetValue<string>() is "ok" or "error")
            done.Add((r["case"].GetValue<string>(), r["arm"].GetValue<string>()));
}

// The KL reference: another run's arm (--reference FILE:ARM), else this run's first arm for the others.
string referenceSpec = options.GetValueOrDefault("reference");
Dictionary<string, TopK> fileReference = null;
string klReference = null;
if (referenceSpec != null)
{
    int colon = referenceSpec.LastIndexOf(':');
    if (colon <= 0) return Usage("--reference FILE:ARM");
    string refFile = Path.GetFullPath(referenceSpec[..colon]), refArm = referenceSpec[(colon + 1)..];
    fileReference = LoadTopK(refFile, refArm);
    klReference = $"{refFile}:{refArm}";
    Console.WriteLine($"KL reference: {klReference} ({fileReference.Count} scored cases)");
}
else if (arms.Length > 1) klReference = $"{arms[0].Name} (this run)";
Dictionary<string, TopK> selfReference = referenceSpec == null && arms.Length > 1 && resume && File.Exists(outputPath)
    ? LoadTopK(outputPath, arms[0].Name) : new Dictionary<string, TopK>();

string kvRequested = options.GetValueOrDefault("kv-cache-dtype");
if (kvRequested != null)
{
    if (!KvCacheDtypeConfig.TryParse(kvRequested, out KvCacheDtype dtype)) return Usage($"--kv-cache-dtype {kvRequested}: f32, f16, q8_0 or q4_0");
    KvCacheDtypeConfig.Set(dtype);
}

var loadTimer = Stopwatch.StartNew();
ModelBase loaded;
try { loaded = ModelBase.Create(modelPath, backend); }
catch (Exception ex)
{
    // The configuration cannot run here (e.g. a cache type the executor cannot initialize): one run record says so, so a
    // report never mistakes the arm's absence for a pass.
    Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
    File.AppendAllText(outputPath, new JsonObject
    {
        ["type"] = "run", ["probe"] = ProbeVersion, ["started_utc"] = DateTime.UtcNow.ToString("O"), ["label"] = options.GetValueOrDefault("label"),
        ["model"] = modelPath, ["backend"] = backendName, ["kv_cache_dtype_requested"] = kvRequested, ["status"] = "load_failed",
        ["error"] = Describe(ex),
    }.ToJsonString() + "\n");
    Console.Error.WriteLine($"LongContextQualityProbe: the model did not load ({Describe(ex)}).");
    return 3;
}
using ModelBase model = loaded;
loadTimer.Stop();
var m = new ModelHandle(model);
if (arms.Any(a => a.PromptAttention != "default") && m.Cuda == null)
    return Usage("arms f16 / 8bit / q8act set seams of the CUDA executor: --backend cuda with a shared-decoder model");
if (arms.Any(a => a.PromptAttention == "8bit") && !m.Runs8BitPrompt)
    return Usage("an 8bit arm needs a CUDA image with the 8-bit prompt attention (an sm_120a / sm_121a build on its GPU); this one has none, " +
        "so the arm would run the 16-bit kernel");
ITokenizer tokenizer = model.Tokenizer;

Directory.CreateDirectory(Path.GetDirectoryName(outputPath)!);
using var output = new StreamWriter(outputPath, append: true, new UTF8Encoding(false));
Write(new JsonObject
{
    ["type"] = "run", ["probe"] = ProbeVersion, ["started_utc"] = DateTime.UtcNow.ToString("O"), ["label"] = options.GetValueOrDefault("label"),
    ["model"] = modelPath, ["model_bytes"] = new FileInfo(modelPath).Length, ["architecture"] = model.Config.Architecture,
    ["backend"] = backendName, ["kv_cache_dtype"] = model.KvCacheDtype.ToShortString(), ["kv_cache_dtype_requested"] = kvRequested,
    ["arms"] = new JsonArray(arms.Select(a => (JsonNode)new JsonObject { ["name"] = a.Name, ["prompt_attention"] = a.PromptAttention }).ToArray()),
    ["kl_reference"] = klReference, ["device"] = DeviceName(), ["flash_image_arch"] = m.CudaAllocator?.FlashPrefillKernels?.ImageArch,
    ["runs_8bit_prompt"] = m.CudaAllocator != null ? (bool?)m.Runs8BitPrompt : null, ["free_device_mb"] = Mb(m.FreeDeviceBytes()),
    ["load_s"] = Math.Round(loadTimer.Elapsed.TotalSeconds, 2), ["cases_file"] = caseFile.Path, ["cases_sha256"] = caseFile.Sha256,
    ["selected_cases"] = selected.Count, ["prefill_chunk"] = prefillChunk, ["score_chunk"] = scoreChunk, ["topk"] = topK,
    ["max_tokens"] = maxTokens == int.MaxValue ? null : (int?)maxTokens, ["resumed_records"] = done.Count,
});
Console.WriteLine($"{model.Config.Architecture} on {backendName} (KV {model.KvCacheDtype.ToShortString()}), loaded in {loadTimer.Elapsed.TotalSeconds:F1} s; " +
    $"{selected.Count} cases x {arms.Length} arms ({string.Join(", ", arms.Select(a => $"{a.Name}={a.PromptAttention}"))}) -> {outputPath}");

// The smallest footprint of a kind and arm that ran out of device memory: larger cases of that kind are not run.
var outOfMemory = new Dictionary<(string Kind, string Arm), int>();
int index = 0;
foreach (Case c in selected)
{
    index++;
    Inputs inputs = null;
    foreach (Arm arm in arms)
    {
        if (done.Contains((c.Id, arm.Name))) continue;
        var rec = new JsonObject
        {
            ["type"] = "record", ["case"] = c.Id, ["kind"] = c.Kind, ["group"] = c.Group, ["arm"] = arm.Name,
            ["prompt_attention"] = arm.PromptAttention, ["kv_cache_dtype"] = model.KvCacheDtype.ToShortString(),
        };
        string skip = c.Footprint > maxTokens ? $"not run: {c.Footprint} tokens is above --max-tokens {maxTokens}"
            : outOfMemory.TryGetValue((c.Kind, arm.Name), out int oom) && c.Footprint >= oom
                ? $"not run: a {c.Kind} case of {oom} tokens ran out of device memory in this arm"
            : c.Kind == "layers" && arm.PromptAttention != "8bit"
                ? "not applicable: layers compares the 8-bit prompt attention with the 16-bit kernel on the same rows (an 8bit arm)"
            : null;
        if (skip != null)
        {
            rec["status"] = "skipped";
            rec["reason"] = skip;
            Write(rec);
            Console.WriteLine($"[{index}/{selected.Count}] {c.Id} {arm.Name}: {skip}");
            continue;
        }
        TopK reference = fileReference != null ? fileReference.GetValueOrDefault(c.Id)
            : arm == arms[0] ? null : selfReference.GetValueOrDefault(c.Id);
        if (reference != null || (arm != arms[0] && klReference != null && (c.Kind is "ppl" or "continue"))) rec["kl_reference"] = klReference;
        var wall = Stopwatch.StartNew();
        try
        {
            inputs ??= Inputs.Prepare(c, caseFile, m, tokenizer);
            m.SetArm(arm);
            model.ResetKVCache();
            Counters before = m.Snapshot();
            string summary = c.Kind switch
            {
                "ppl" or "continue" => RunScored(c, inputs, reference, rec, arm == arms[0] && fileReference == null ? selfReference : null),
                "greedy" => RunGreedy(c, inputs, rec),
                "chat" => RunChat(c, inputs, rec),
                "layers" => RunLayers(c, inputs, rec),
                _ => throw new InvalidOperationException(c.Kind),
            };
            Counters after = m.Snapshot();
            rec["status"] = "ok";
            rec["attn8_launches"] = after.Attn8Launches - before.Attn8Launches;
            rec["attn8_band_refusals"] = after.Attn8BandRefusals - before.Attn8BandRefusals;
            rec["marked_prompt_attentions"] = after.MarkedAttentions - before.MarkedAttentions;
            rec["free_device_mb"] = Mb(m.FreeDeviceBytes());
            rec["wall_s"] = Math.Round(wall.Elapsed.TotalSeconds, 3);
            Console.WriteLine($"[{index}/{selected.Count}] {c.Id} {arm.Name}: {summary}, 8-bit launches {rec["attn8_launches"]}" +
                $"{(after.Attn8BandRefusals > before.Attn8BandRefusals ? $", BAND REFUSALS {after.Attn8BandRefusals - before.Attn8BandRefusals}" : "")}, {wall.Elapsed.TotalSeconds:F1} s");
        }
        catch (Exception ex)
        {
            bool oomError = IsOutOfMemory(ex);
            rec["status"] = "error";
            rec["error"] = Describe(ex);
            rec["out_of_memory"] = oomError;
            rec["wall_s"] = Math.Round(wall.Elapsed.TotalSeconds, 3);
            if (oomError) outOfMemory[(c.Kind, arm.Name)] = Math.Min(outOfMemory.GetValueOrDefault((c.Kind, arm.Name), int.MaxValue), c.Footprint);
            Console.WriteLine($"[{index}/{selected.Count}] {c.Id} {arm.Name}: ERROR {Describe(ex)}");
        }
        finally
        {
            m.SetArm(new Arm("reset", "default"));
            if (m.Cuda != null) m.Cuda.PromptAttention8CheckForTests = null;
        }
        Write(rec);
        try { model.ResetKVCache(); }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"The model cannot reset after {c.Id} ({Describe(ex)}): stopping; rerun with --resume.");
            return 3;
        }
    }
    if (fileReference == null) selfReference.Remove(c.Id);
}
Console.WriteLine($"done: {outputPath}");
return 0;

// ---------------------------------------------------------------------------------------------------- case kinds

// ppl: the prefix (tokens - score_tokens) as prompt chunks, then the scored rows as marked prompt chunks of --score-chunk rows
// with every row's logits. continue: the same prefix, then the scored rows one decode row at a time. The prefix's last row
// predicts the first scored token in both.
string RunScored(Case c, Inputs inputs, TopK reference, JsonObject rec, Dictionary<string, TopK> keepAsReference)
{
    int[] tokens = inputs.Tokens;
    int prefix = c.Tokens - c.ScoreTokens, vocab = m.Vocab;
    rec["input_tokens"] = tokens.Length;
    rec["input_sha256"] = inputs.Sha256;
    rec["score_tokens"] = c.ScoreTokens;
    // A reference is another record's distribution over the same scored tokens: one whose input or scored rows differ (a case
    // file rebuilt differently, another tokenizer) is refused, never compared row by row with these. A KL names the exact
    // distribution it was computed against (the sha256 of its topk), which the report checks against its reference arm.
    if (reference != null && (reference.InputSha256 != inputs.Sha256 || reference.Rows != c.ScoreTokens))
    {
        rec["kl_refused"] = $"the reference record scores other tokens (input sha256 {reference.InputSha256 ?? "none"}, {reference.Rows} rows)";
        reference = null;
    }
    if (reference != null) rec["kl_reference_topk_sha256"] = reference.Digest;
    var scores = new RowScores(topK);
    var sw = Stopwatch.StartNew();
    float[] last = m.Prefill(tokens[..prefix], prefillChunk);
    rec["prefill_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    sw.Restart();
    scores.Add(last, tokens[prefix], reference);
    if (c.Kind == "ppl")
    {
        // Scored chunks of --score-chunk rows; a tail of 16 rows or fewer (which would run unmarked, as a decode-shaped step)
        // joins the chunk before it.
        var chunks = new List<(int Pos, int Len)>();
        for (int pos = prefix; pos < c.Tokens - 1; pos += scoreChunk) chunks.Add((pos, Math.Min(scoreChunk, c.Tokens - 1 - pos)));
        if (chunks.Count > 1 && chunks[^1].Len <= 16)
        {
            int tail = chunks[^1].Len;
            chunks.RemoveAt(chunks.Count - 1);
            chunks[^1] = (chunks[^1].Pos, chunks[^1].Len + tail);
        }
        if (chunks[^1].Len <= 16) throw new InvalidDataException($"{c.Id}: ppl scores more than 17 tokens (a scored chunk is a prompt chunk of more than 16 rows)");
        foreach (var (pos, len) in chunks)
        {
            float[] rows = m.ScoredPromptChunk(tokens[pos..(pos + len)]);
            var scored = new RowScores.Row[len];
            int at = scores.Count;
            Parallel.For(0, len, r => scored[r] = RowScores.Score(rows.AsSpan(r * vocab, vocab), tokens[pos + r + 1], topK, reference, at + r));
            foreach (RowScores.Row row in scored) scores.Add(row);
        }
    }
    else
    {
        for (int i = prefix; i < c.Tokens - 1; i++)
            scores.Add(m.Decode(tokens[i]), tokens[i + 1], reference);
    }
    rec["score_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    foreach (var (key, value) in scores.ToJson(withTopK: true)) rec[key] = value?.DeepClone();
    double meanNll = scores.Nll.Average();
    rec["mean_nll"] = Math.Round(meanNll, 6);
    rec["ppl"] = Math.Round(Math.Exp(meanNll), 6);
    if (keepAsReference != null && topK > 0) keepAsReference[c.Id] = TopK.Decode(rec);
    string kl = scores.Kl.Count > 0 ? $", KL {scores.Kl.Average():E2}" : "";
    return $"{c.Kind} {c.Tokens} tokens: PPL {Math.Exp(meanNll):F4} over {scores.Count}{kl}, prefill {rec["prefill_s"]} s, scoring {rec["score_s"]} s";
}

// greedy: the prompt (ending where the case builder cut it), then new_tokens greedy decode rows; every step's top-2 logit gap.
string RunGreedy(Case c, Inputs inputs, JsonObject rec)
{
    rec["input_tokens"] = inputs.Tokens.Length;
    rec["input_sha256"] = inputs.Sha256;
    var sw = Stopwatch.StartNew();
    float[] logits = m.Prefill(inputs.Tokens, prefillChunk);
    rec["prefill_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    sw.Restart();
    var (generated, gaps, stop) = Generate(logits, c.NewTokens);
    rec["decode_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    rec["generated"] = new JsonArray(generated.Select(t => (JsonNode)t).ToArray());
    rec["gaps"] = RowScores.Floats(gaps);
    rec["stop"] = stop;
    rec["text"] = tokenizer.Decode(generated);
    return $"greedy {inputs.Tokens.Length} tokens: {generated.Count} generated ({stop}), prefill {rec["prefill_s"]} s, decode {rec["decode_s"]} s";
}

// chat: the rendered prompt (its document fitted to fit_tokens), then greedy decoding to an end-of-sequence token or max_new_tokens.
string RunChat(Case c, Inputs inputs, JsonObject rec)
{
    rec["prompt_tokens"] = inputs.Tokens.Length;
    rec["prompt_sha256"] = inputs.Sha256;
    rec["fit_tokens"] = c.FitTokens > 0 ? (int?)c.FitTokens : null;
    rec["document_chars"] = inputs.DocumentChars;
    var sw = Stopwatch.StartNew();
    float[] logits = m.Prefill(inputs.Tokens, prefillChunk);
    rec["prefill_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    sw.Restart();
    var (generated, gaps, stop) = Generate(logits, c.MaxNewTokens);
    rec["decode_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    rec["generated"] = new JsonArray(generated.Select(t => (JsonNode)t).ToArray());
    rec["gaps"] = RowScores.Floats(gaps);
    rec["stop"] = stop;
    string text = tokenizer.Decode(generated);
    rec["text"] = text;
    string shown = text.Length > 60 ? text[..60] + "..." : text;
    return $"chat {inputs.Tokens.Length} tokens: {generated.Count} generated ({stop}) \"{shown.ReplaceLineEndings(" ")}\", prefill {rec["prefill_s"]} s";
}

// layers: the 8-bit arm's prompt attention against the 16-bit kernel on the same rows of every marked chunk (the executor's
// diagnostic hook): relative RMS and largest absolute difference a launch, on real activations.
string RunLayers(Case c, Inputs inputs, JsonObject rec)
{
    var checks = new List<(long Call, double Rel, double Max)>();
    m.Cuda.PromptAttention8CheckForTests = (call, rel, max) => checks.Add((call, rel, max));
    var sw = Stopwatch.StartNew();
    m.Prefill(inputs.Tokens, prefillChunk);
    rec["prefill_s"] = Math.Round(sw.Elapsed.TotalSeconds, 3);
    rec["input_tokens"] = inputs.Tokens.Length;
    rec["input_sha256"] = inputs.Sha256;
    rec["checks"] = new JsonArray(checks.Select(x => (JsonNode)new JsonArray(x.Call, Math.Round(x.Rel, 8), Math.Round(x.Max, 6))).ToArray());
    if (checks.Count == 0) return "layers: no 8-bit launch to check";
    var rels = checks.Select(x => x.Rel).OrderBy(x => x).ToArray();
    rec["median_rel_rms"] = Math.Round(rels[rels.Length / 2], 8);
    rec["max_rel_rms"] = Math.Round(rels[^1], 8);
    rec["max_abs"] = Math.Round(checks.Max(x => x.Max), 6);
    return $"layers {inputs.Tokens.Length} tokens: {checks.Count} launches, rel RMS median {rels[rels.Length / 2]:P2}, max {rels[^1]:P2}";
}

(List<int> Generated, List<double> Gaps, string Stop) Generate(float[] logits, int maxNew)
{
    var generated = new List<int>();
    var gaps = new List<double>();
    for (int step = 0; ; step++)
    {
        RowScores.Row row = RowScores.Score(logits, -1, 0, null, 0);
        // Every step's gap, the one that chose end of sequence included (gaps then holds one more entry than generated): where
        // one arm ends and the other goes on, the report reads the divergence's gap there.
        gaps.Add(row.Gap);
        if (tokenizer.IsEos(row.Top1)) return (generated, gaps, "eos");
        generated.Add(row.Top1);
        if (generated.Count >= maxNew) return (generated, gaps, "length");
        logits = m.Decode(row.Top1);
    }
}

// ---------------------------------------------------------------------------------------------------- helpers

void Write(JsonObject record)
{
    output.WriteLine(record.ToJsonString());
    output.Flush();
}

int Int(string key, int fallback) => options.TryGetValue(key, out string v) ? int.Parse(v, CultureInfo.InvariantCulture) : fallback;

HashSet<string> Set(string key) => options.TryGetValue(key, out string v) ? v.Split(',', StringSplitOptions.RemoveEmptyEntries).ToHashSet() : null;

static long? Mb(long bytes) => bytes < 0 ? null : bytes >> 20;

static IEnumerable<JsonNode> ReadRecords(string path)
{
    foreach (string line in File.ReadLines(path))
    {
        if (string.IsNullOrWhiteSpace(line)) continue;
        JsonNode node;
        try { node = JsonNode.Parse(line); }
        catch (JsonException) { continue; }   // a line cut short by a killed run
        if (node?["type"]?.GetValue<string>() == "record") yield return node;
    }
}

static Dictionary<string, TopK> LoadTopK(string path, string arm)
{
    var map = new Dictionary<string, TopK>(StringComparer.Ordinal);
    if (!File.Exists(path)) throw new FileNotFoundException($"reference {path}");
    foreach (JsonNode r in ReadRecords(path))
        if (r["arm"]?.GetValue<string>() == arm && r["status"]?.GetValue<string>() == "ok" && TopK.Decode(r) is { } t)
            map[r["case"].GetValue<string>()] = t;
    return map;
}

static bool IsOutOfMemory(Exception ex)
{
    for (Exception e = ex; e != null; e = e.InnerException)
        if (e is OutOfMemoryException || e.Message.Contains("out of memory", StringComparison.OrdinalIgnoreCase)
            || e.Message.Contains("OUT_OF_MEMORY", StringComparison.Ordinal))
            return true;
    return false;
}

static string Describe(Exception ex)
{
    var sb = new StringBuilder();
    for (Exception e = ex; e != null; e = e.InnerException)
        sb.Append(sb.Length == 0 ? "" : " <- ").Append(e.GetType().Name).Append(": ").Append(e.Message);
    return sb.ToString();
}

string DeviceName()
{
    if (m.CudaAllocator == null) return backendName;
    var name = new byte[256];
    m.CudaAllocator.Context.MakeCurrent();
    CudaDriverApi.cuCtxGetDevice(out int device).ThrowOnError();
    CudaDriverApi.cuDeviceGetName(name, name.Length, device).ThrowOnError();
    return Encoding.UTF8.GetString(name).TrimEnd('\0');
}

static int Usage(string error)
{
    Console.Error.WriteLine($"LongContextQualityProbe: {error}");
    Console.Error.WriteLine("usage: LongContextQualityProbe --model M.gguf --cases CASES.jsonl --output RUN.jsonl [--backend cuda]");
    Console.Error.WriteLine("         [--arms f16,8bit | name=setting,...] [--kv-cache-dtype f16|q8_0] [--reference RUN.jsonl:ARM]");
    Console.Error.WriteLine("         [--kinds ppl,continue,greedy,chat,layers] [--groups retrieval,gsm8k,humaneval] [--ids REGEX]");
    Console.Error.WriteLine("         [--max-tokens N] [--prefill-chunk 2048] [--score-chunk 512] [--topk 32] [--label TEXT] [--resume]");
    Console.Error.WriteLine("Arm settings: default (the build's own), f16 (the 16-bit prompt kernel), 8bit (the 8-bit prompt attention),");
    Console.Error.WriteLine("  q8act (the build's prompt attention; every NVFP4 product on the q8 matvec, so prompt rows keep q8 activations).");
    return 2;
}

/// <summary>A case's model input, built once for all its arms: a corpus window, a prompt ending at a character, or a fitted chat prompt.</summary>
internal sealed class Inputs
{
    public int[] Tokens;
    public string Sha256;
    public int DocumentChars;

    public static Inputs Prepare(Case c, CaseFile file, ModelHandle m, ITokenizer tokenizer)
    {
        var inputs = new Inputs();
        switch (c.Kind)
        {
            case "ppl" or "continue" or "layers": inputs.Tokens = file.Window(c, tokenizer); break;
            case "greedy": inputs.Tokens = file.PromptEndingAt(c, tokenizer); break;
            default: (inputs.Tokens, inputs.DocumentChars) = file.ChatPrompt(c, m); break;
        }
        var bytes = new byte[inputs.Tokens.Length * 4];
        Buffer.BlockCopy(inputs.Tokens, 0, bytes, 0, bytes.Length);
        inputs.Sha256 = Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
        return inputs;
    }
}
