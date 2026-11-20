// Muse-Glimmer KV-cache grow / sliding-window-ring probe.
//
// Drives one Muse-Glimmer model through a prompt and a decode, recording the logits
// of every step, so two processes that differ only in cache configuration can be
// compared position by position (compare.py). See README.md.
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

string Option(string name, string? fallback)
{
    int index = Array.IndexOf(args, name);
    if (index < 0) return fallback!;
    if (index + 1 >= args.Length) throw new ArgumentException($"Missing value for {name}.");
    return args[index + 1];
}

string modelPath = Option("--model", Environment.GetEnvironmentVariable("TS_TEST_MUSE_GLIMMER_MODEL"));
string backendName = Option("--backend", "ggml_metal");
string scenario = Option("--scenario", "decode");
string output = Option("--out", "artifacts/muse-glimmer-kv-grow/report.json");
string? promptFile = Option("--prompt-file", null);
string? systemFile = Option("--system-file", null);
string? teacherFile = Option("--teacher", null);
string? logitsOut = Option("--logits-out", null);
// --logits-ref: another run's --logits-out (the same prompt and teacher): every step's cosine against that run's row is
// recorded in the report (Steps[].Cos), so a long run needs no logits file of its own.
string? logitsRef = Option("--logits-ref", null);
// --logits-ref2: a second reference (Steps[].Cos2), e.g. the other family's presized arm.
string? logitsRef2 = Option("--logits-ref2", null);
// --nll-text: teacher-force a fixed raw text (no chat template) and report its perplexity. The first
// --nll-prefix tokens are prefilled in one forward; every later token is scored from the logits of the
// one-row forward before it (the decode path), so a run crosses the same cache grows a generation does.
string? nllText = Option("--nll-text", null);
int nllPrefix = int.Parse(Option("--nll-prefix", "64"));
int steps = int.Parse(Option("--steps", nllText != null ? "0" : "64"));
// --split: the position a scenario acts at. Inside the prompt it splits the prompt's forward there; at or past the prompt's end
// (truncate, checkpoint, snapshot on the runtime) the scenario acts in the decode, when the cache holds that many positions.
int split = int.Parse(Option("--split", "0"));
// The rows a decode-time truncation forwards past --split before it rewinds there (a live-cache rewind's size).
int rewind = int.Parse(Option("--rewind", "16"));
int topK = int.Parse(Option("--topk", "8"));
int tp = int.Parse(Option("--tp", "1"));
// --layer-split N: whole layers over N GPUs (the runtime family; legacy refuses it, as its serving contract says).
int layerSplit = int.Parse(Option("--layer-split", "1"));
// --family: "legacy" builds what the descriptor builds (MuseGlimmerModel until the switch, docs/design/muse-glimmer-unified.md
// stage S8); "runtime" builds the unified-runtime family through its test seam (MuseGlimmerArchitecture.CreateRuntimeModel).
string family = Option("--family", "legacy");
// --scenario bench: the CLI's --benchmark (greedy, end to end) on either family in this binary: a prompt of
// --bench-prefill tokens cycling 17 ids, then --bench-decode greedy one-row forwards, --bench-runs times.
int benchPrefill = int.Parse(Option("--bench-prefill", "512"));
int benchDecode = int.Parse(Option("--bench-decode", "128"));
int benchRuns = int.Parse(Option("--bench-runs", "2"));
// The runtime family's measured settings, for A/Bs in one binary (docs/design/muse-glimmer-unified.md stage S7); each is set
// through the model's internal test seams before (or right after) it is constructed:
//   --ring-step-rows N|linear  the rows a prompt step writes into a sliding-window ring (rings of align256(window + N) cells;
//                              default 512), or linear caches for the sliding-window layers (no rings, unbounded steps);
//   --gate-up apart|fused      every same-type ffn_gate/ffn_up pair kept apart or fused, in place of the family's rule;
//   --no-prune                 a prompt step's last layer runs its feed-forward over every row (PrunesLastLayerRows off).
// The concurrency reorder of the decode graphs is ggml-metal's own A/B switch, TS_METAL_GRAPH_OPTIMIZE=0.
string? ringStepRows = Option("--ring-step-rows", null);
string? gateUp = Option("--gate-up", null);
bool noPrune = args.Contains("--no-prune");
// --bench-refill: the bench prompt goes through ForwardRefill (the server's long-prompt entry: chunks of TS_PREFILL_CHUNK rows,
// logits for the last token only) instead of one Forward, so a linear-cache arm steps the prompt as a ring arm does.
bool benchRefill = args.Contains("--bench-refill");

string[] scenarios = { "decode", "split", "snapshot", "truncate", "checkpoint", "slot-growth", "bench" };
if (!scenarios.Contains(scenario))
    throw new ArgumentException($"Unknown scenario '{scenario}': use one of {string.Join(", ", scenarios)}.");
if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath))
    throw new ArgumentException("Pass --model <Muse-Glimmer gguf> (or set TS_TEST_MUSE_GLIMMER_MODEL).");
if (!Enum.TryParse(backendName.Replace("_", string.Empty), ignoreCase: true, out BackendType backend))
    throw new ArgumentException($"Unknown backend '{backendName}'.");

if (family != "runtime" && (ringStepRows != null || gateUp != null || noPrune))
    throw new ArgumentException("--ring-step-rows, --gate-up and --no-prune set the runtime family's choices: pass --family runtime.");
if (ringStepRows != null)
    StandardDecoderModel.RingGeometryForTests.Value =
        (ringStepRows == "linear" ? 1 << 24 : int.Parse(ringStepRows), StandardDecoderModel.RingAlignment);
if (gateUp != null)
    MuseGlimmerRuntimeModel.GateUpApartForTests.Value = gateUp switch
    {
        "apart" => true,
        "fused" => false,
        _ => throw new ArgumentException($"--gate-up {gateUp}: use apart or fused."),
    };

var load = Stopwatch.StartNew();
// Released explicitly (and the backend shut down) at the end: ggml-metal asserts at
// process exit if any buffer is still registered, as the server's ProcessExit hook knows.
var model = family switch
{
    "legacy" => ModelBase.Create(modelPath, backend, tpDegree: tp, layerSplitDegree: layerSplit),
    "runtime" => MuseGlimmerArchitecture.CreateRuntimeModel(modelPath, backend, tpDegree: tp, layerSplitDegree: layerSplit),
    _ => throw new ArgumentException($"--family {family}: use legacy or runtime."),
};
double loadMs = load.Elapsed.TotalMilliseconds;
if (model.Config.Architecture != "muse-glimmer")
    throw new InvalidOperationException($"Expected a muse-glimmer model, got {model.Config.Architecture}.");
var runtime = model as StandardDecoderModel;
var slots = model as IBatchedPagedModel;
if (noPrune) runtime!.LastLayerPruningDisabledForTests = true;
Console.WriteLine($"[probe] family={family} ({model.GetType().Name}), loaded in {loadMs:F0} ms");
if (runtime == null && scenario is "checkpoint" or "slot-growth")
    throw new ArgumentException($"--scenario {scenario} runs on the runtime family (per-request slots and checkpoints): pass --family runtime.");

if (scenario == "bench")
{
    RunBench();
    model.Dispose();
    if (backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlCpu)
        TensorSharp.GGML.GgmlBasicOps.Shutdown();
    return;
}

// ---- prompt ------------------------------------------------------------------
int[] prompt;
int[]? teacher = null;
if (nllText != null)
{
    Require(teacherFile == null && promptFile == null && systemFile == null,
        "--nll-text scores its own text: it takes no --teacher, --prompt-file or --system-file.");
    int[] all = model.Tokenizer.Encode(File.ReadAllText(nllText), addSpecial: true).ToArray();
    Require(nllPrefix >= 1 && nllPrefix < all.Length, $"--nll-prefix must be inside the text's {all.Length} tokens.");
    prompt = all[..nllPrefix];
    teacher = all[nllPrefix..];
    steps = steps <= 0 ? teacher.Length : Math.Min(steps, teacher.Length);
}
else
{
    string userText = promptFile != null ? File.ReadAllText(promptFile) : "请详细介绍最终幻想7";
    var messages = new List<ChatMessage>();
    if (systemFile != null)
        messages.Add(new ChatMessage { Role = "system", Content = File.ReadAllText(systemFile) });
    messages.Add(new ChatMessage { Role = "user", Content = userText });
    string rendered = ChatTemplate.RenderFromGgufTemplate(model.Config.ChatTemplate, messages,
        addGenerationPrompt: true, architecture: model.Config.Architecture);
    prompt = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
}

if (teacherFile != null)
{
    using var doc = JsonDocument.Parse(File.ReadAllText(teacherFile));
    teacher = doc.RootElement.GetProperty("Steps").EnumerateArray()
        .Select(s => s.GetProperty("Chosen").GetInt32()).ToArray();
    int[] teacherPrompt = doc.RootElement.GetProperty("PromptTokens").EnumerateArray().Select(e => e.GetInt32()).ToArray();
    if (!teacherPrompt.SequenceEqual(prompt))
        throw new InvalidOperationException("The teacher report was produced from a different prompt.");
    steps = Math.Min(steps, teacher.Length);
}

bool inPrompt = split > 0 && split < prompt.Length;
bool inDecode = runtime != null && split >= prompt.Length && scenario is "snapshot" or "checkpoint" or "truncate";
switch (scenario)
{
    case "split":
        Require(inPrompt, "--split must be inside the prompt.");
        break;
    case "snapshot" when runtime == null:
        Require(split >= 256 && split < prompt.Length, "--split must cover at least one 256-token block and stay inside the prompt.");
        break;
    case "truncate" when runtime == null:
        Require(inPrompt, "--split must be inside the prompt.");
        break;
    case "snapshot" or "checkpoint" or "truncate":
        Require(inPrompt || (inDecode && teacher != null && split < prompt.Length + steps - 1),
            "--split must be inside the prompt, or (teacher-forced) a position the decode reaches.");
        break;
}

// The runtime's scenarios start on the primary cache (it honours MAX_CONTEXT and TS_KV_INITIAL_TOKENS, as legacy's one cache
// does); a restored checkpoint continues in the slot of a new request (`request`); slot-growth steps one request's slot through
// the unified batch entry, as the engine serves it.
int requestSerial = 0;
string? request = null;
bool unified = scenario == "slot-growth";

Console.WriteLine($"[probe] scenario={scenario} family={family} backend={backend} tp={tp} prompt={prompt.Length} tokens steps={steps} split={split} " +
    $"capacity={Capacity()} ringCells={RingCells()} kv={model.KvCacheDtype}");

// ---- prefill -----------------------------------------------------------------
var events = new List<string>();
var prefill = Stopwatch.StartNew();
float[] logits;
model.ResetKVCache();
switch (scenario)
{
    case "decode":
        logits = model.Forward(prompt);
        break;
    case "split":
        // Two forwards: the second one starts below the initial capacity and ends past it.
        model.Forward(prompt[..split]);
        events.Add($"after first forward: seq={model.CacheSeqLen} capacity={Capacity()}");
        logits = model.Forward(prompt[split..]);
        break;
    case "snapshot" when runtime == null:
    {
        // What a pooled radix-cache hit does: extract whole 256-token blocks, start a
        // fresh cache, inject them, and forward only the remainder.
        int blocks = split / 256;
        int reused = blocks * 256;
        model.Forward(prompt);
        long blockBytes = model.ComputeKVBlockByteSize(256);
        var saved = new List<byte[]>();
        for (int b = 0; b < blocks; b++)
        {
            var buf = new byte[blockBytes];
            Require(model.TryExtractKVBlock(b * 256, 256, buf), $"TryExtractKVBlock refused block {b}.");
            saved.Add(buf);
        }
        // Scramble the same rows first: a reset keeps the device copies on the GPU
        // backends, so a restore that wrote nothing would otherwise read them back.
        model.ResetKVCache();
        model.Forward(prompt.Reverse().ToArray());
        model.ResetKVCache();
        for (int b = 0; b < blocks; b++)
            Require(model.TryInjectKVBlock(b * 256, 256, saved[b]), $"TryInjectKVBlock refused block {b}.");
        events.Add($"injected {reused} tokens in {blocks} blocks of {blockBytes} bytes; seq={model.CacheSeqLen}");
        logits = model.Forward(prompt[reused..]);
        break;
    }
    case "truncate" when inPrompt:
        // What a live-cache rewind does: keep the head, drop the tail, re-forward it.
        model.Forward(prompt);
        Rewind(split, prompt[..split]);
        logits = model.Forward(prompt[split..]);
        break;
    case "snapshot" or "checkpoint" when inPrompt:
        model.Forward(prompt[..split]);
        Restore(scenario, prompt[..split]);
        logits = model.Forward(prompt[split..]);
        break;
    case "snapshot" or "checkpoint" or "truncate":
        logits = model.Forward(prompt);   // the scenario acts in the decode
        break;
    case "slot-growth":
    {
        // A request whose slot holds only its prompt and first token (a reservation its generation outgrows): the unified step
        // grows the slot as the decode passes its capacity, keeping its rows (TryForwardSlots -> EnsureSlotCapacity).
        request = "probe-0";
        ((IUnifiedBatchModel)model).EnsureSequenceSlot(request, prompt.Length + 1);
        events.Add($"slot reserved for {prompt.Length + 1} tokens: capacity {Capacity()}");
        logits = Unified(prompt, 0);
        break;
    }
    default:
        throw new InvalidOperationException(scenario);
}
double prefillMs = prefill.Elapsed.TotalMilliseconds;

// ---- decode ------------------------------------------------------------------
var records = new List<StepRecord>(steps);
FileStream? logitsStream = logitsOut != null ? File.Create(logitsOut) : null;
FileStream? refStream = logitsRef != null ? File.OpenRead(logitsRef) : null;
FileStream? refStream2 = logitsRef2 != null ? File.OpenRead(logitsRef2) : null;
int vocab = model.Config.VocabSize;
var refRow = new byte[vocab * 2];
double nllSum = 0;
int lastCapacity = Capacity();
var decode = Stopwatch.StartNew();
for (int i = 0; i < steps; i++)
{
    int position = Length();      // the position the next token will occupy
    var top = TopK(logits, Math.Min(vocab, logits.Length), topK);
    int chosen = teacher != null ? teacher[i] : top[0].Id;
    if (nllText != null) nllSum -= LogSoftmaxAt(logits, Math.Min(vocab, logits.Length), chosen);
    double? cos = refStream != null ? CosineAgainstReference(logits, refStream, logitsRef!) : null;
    double? cos2 = refStream2 != null ? CosineAgainstReference(logits, refStream2, logitsRef2!) : null;
    int capacity = Capacity();
    if (capacity != lastCapacity)
    {
        events.Add($"step {i} (position {position}): capacity {lastCapacity} -> {capacity}");
        lastCapacity = capacity;
    }
    records.Add(new StepRecord(i, position, top[0].Id, chosen, top, capacity, cos, cos2));
    if (logitsStream != null) WriteHalf(logitsStream, logits, vocab);
    if (i + 1 >= steps) break;
    if (inDecode && position + 1 == split)
    {
        // The token at `position` goes in first, so the scenario acts with the cache holding `split` positions.
        logits = model.Forward(new[] { chosen });
        int[] history = prompt.Concat(teacher![..(i + 1)]).ToArray();
        if (scenario == "truncate")
        {
            // A live-cache rewind: rows past `split` forwarded, then dropped; the decode forwards them again one at a time.
            int ahead = Math.Min(rewind, teacher.Length - (i + 1));
            model.Forward(teacher[(i + 1)..(i + 1 + ahead)]);
            Rewind(split, history);
        }
        else
        {
            Restore(scenario, history);
        }
        continue;
    }
    logits = unified ? Unified(new[] { chosen }, position) : model.Forward(new[] { chosen });
}
double decodeMs = decode.Elapsed.TotalMilliseconds;
logitsStream?.Dispose();
refStream?.Dispose();
refStream2?.Dispose();

string text = model.Tokenizer.Decode(records.Select(r => r.Chosen).ToList());
object? nll = nllText == null ? null : new
{
    Text = Path.GetFullPath(nllText),
    Prefix = nllPrefix,
    ScoredTokens = steps,
    MeanNll = steps > 0 ? nllSum / steps : double.NaN,
    Perplexity = steps > 0 ? Math.Exp(nllSum / steps) : double.NaN,
    Top1Agreement = steps > 0 ? records.Count(r => r.Argmax == r.Chosen) / (double)steps : double.NaN,
};
double[] cosines = records.Where(r => r.Cos.HasValue).Select(r => r.Cos!.Value).ToArray();
double[] cosines2 = records.Where(r => r.Cos2.HasValue).Select(r => r.Cos2!.Value).ToArray();
Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output, JsonSerializer.Serialize(new
{
    Scenario = scenario,
    Family = family,
    ModelClass = model.GetType().Name,
    Backend = backend.ToString(),
    TensorParallel = tp,
    Model = Path.GetFullPath(modelPath),
    Environment = new
    {
        MAX_CONTEXT = Environment.GetEnvironmentVariable("MAX_CONTEXT"),
        TS_KV_INITIAL_TOKENS = Environment.GetEnvironmentVariable("TS_KV_INITIAL_TOKENS"),
        TS_MUSE_GLIMMER_SWA_RING = Environment.GetEnvironmentVariable("TS_MUSE_GLIMMER_SWA_RING"),
        KV_CACHE_DTYPE = Environment.GetEnvironmentVariable("KV_CACHE_DTYPE"),
    },
    KvCacheDtype = model.KvCacheDtype.ToString(),
    SwaRows = RingCells(),
    FinalCapacity = Capacity(),
    ModelLoadMilliseconds = loadMs,
    PrefillMilliseconds = prefillMs,
    DecodeMilliseconds = decodeMs,
    DecodeTokensPerSecond = steps > 1 ? (steps - 1) / (decodeMs / 1000.0) : 0,
    Split = split,
    Events = events,
    Nll = nll,
    LogitsReference = logitsRef != null ? Path.GetFullPath(logitsRef) : null,
    MinCosine = cosines.Length > 0 ? cosines.Min() : (double?)null,
    LogitsReference2 = logitsRef2 != null ? Path.GetFullPath(logitsRef2) : null,
    MinCosine2 = cosines2.Length > 0 ? cosines2.Min() : (double?)null,
    PromptTokens = prompt,
    GeneratedText = text,
    Steps = records,
}, new JsonSerializerOptions { WriteIndented = false }));
Console.WriteLine($"[probe] prefill {prefillMs:F0} ms, decode {steps} steps in {decodeMs:F0} ms; wrote {output}");
foreach (string e in events) Console.WriteLine($"[probe] event: {e}");
if (cosines.Length > 0)
    Console.WriteLine($"[probe] cosine against {Path.GetFileName(logitsRef)}: min {cosines.Min():F6} mean {cosines.Average():F6} over {cosines.Length} steps");
if (cosines2.Length > 0)
    Console.WriteLine($"[probe] cosine against {Path.GetFileName(logitsRef2)}: min {cosines2.Min():F6} mean {cosines2.Average():F6} over {cosines2.Length} steps");
if (nllText != null && steps > 0)
    Console.WriteLine($"[probe] nll: {steps} tokens after a {nllPrefix}-token prefix, mean NLL {nllSum / steps:F5}, " +
        $"perplexity {Math.Exp(nllSum / steps):F4}, top-1 agreement {records.Count(r => r.Argmax == r.Chosen) / (double)steps:P2}");
Console.WriteLine($"[probe] text: {Clip(text.Replace('\n', ' '), 400)}");
model.Dispose();
if (backend is BackendType.GgmlMetal or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlCpu)
    TensorSharp.GGML.GgmlBasicOps.Shutdown();

// The CLI's RunBenchmark, greedy end to end: the prefill's argmax seeds the decode; each decode step is a one-row forward
// and its argmax; the decode clock covers both, as the CLI's does.
void RunBench()
{
    int vocab = model.Config.VocabSize;
    int span = Math.Max(1, Math.Min(17, vocab)), basis = Math.Max(0, Math.Min(100, vocab - span));
    int[] ids = Enumerable.Range(0, benchPrefill).Select(i => basis + i % span).ToArray();
    var runs = new List<object>();
    double bestPp = 0, bestTg = 0;
    int[]? firstTokens = null;
    for (int run = 0; run < benchRuns; run++)
    {
        model.ResetKVCache();
        // The host's token embedding (the runtime's lookup and upload of each step's rows; legacy gathers in its graph on Metal
        // and the CPU, so it counts little or nothing there): ModelBase's own embedding clock, read around each phase.
        SetField(model, "_embTicks", 0L);
        var sw = Stopwatch.StartNew();
        float[] logits = benchRefill ? model.ForwardRefill(ids) : model.Forward(ids);
        double prefillMs = sw.Elapsed.TotalMilliseconds;
        double prefillEmbedMs = Field<long>(model, "_embTicks") * 1000.0 / Stopwatch.Frequency;
        int next = TopK(logits, Math.Min(vocab, logits.Length), 1)[0].Id;
        var produced = new int[benchDecode];
        SetField(model, "_embTicks", 0L);
        sw.Restart();
        for (int i = 0; i < benchDecode; i++)
        {
            logits = model.Forward(new[] { next });
            produced[i] = next;
            next = TopK(logits, Math.Min(vocab, logits.Length), 1)[0].Id;
        }
        double decodeMs = sw.Elapsed.TotalMilliseconds;
        double decodeEmbedMs = Field<long>(model, "_embTicks") * 1000.0 / Stopwatch.Frequency;
        double pp = benchPrefill / (prefillMs / 1000.0), tg = benchDecode > 0 ? benchDecode / (decodeMs / 1000.0) : 0;
        bestPp = Math.Max(bestPp, pp);
        bestTg = Math.Max(bestTg, tg);
        firstTokens ??= produced;
        Console.WriteLine($"[probe] bench run {run + 1}/{benchRuns}: prefill {benchPrefill} tokens in {prefillMs:F0} ms ({pp:F1} tok/s; host embedding " +
            $"{prefillEmbedMs:F1} ms); decode {benchDecode} tokens in {decodeMs:F0} ms ({tg:F2} tok/s; host embedding {decodeEmbedMs:F2} ms)");
        runs.Add(new
        {
            PrefillMilliseconds = prefillMs, PrefillTokensPerSecond = pp, DecodeMilliseconds = decodeMs, DecodeTokensPerSecond = tg,
            PrefillEmbeddingMilliseconds = prefillEmbedMs, DecodeEmbeddingMilliseconds = decodeEmbedMs,
        });
    }
    Console.WriteLine($"[probe] bench best: pp{benchPrefill} {bestPp:F1} tok/s, tg{benchDecode} {bestTg:F2} tok/s ({family}, {backend})");
    Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
    File.WriteAllText(output, JsonSerializer.Serialize(new
    {
        Scenario = "bench", Family = family, ModelClass = model.GetType().Name, Backend = backend.ToString(), Model = Path.GetFullPath(modelPath),
        KvCacheDtype = model.KvCacheDtype.ToString(), ModelLoadMilliseconds = loadMs, PrefillTokens = benchPrefill, DecodeTokens = benchDecode,
        Settings = new
        {
            RingCells = RingCells(), MaxSequenceStepRows = runtime?.MaxSequenceStepRows, RingStepRows = ringStepRows, GateUp = gateUp,
            GateUpFusedLayers = (Field<bool[]>(model, "_gateUpFused") ?? Array.Empty<bool>()).Count(f => f),
            PrunesLastLayer = runtime != null ? !noPrune : (bool?)null, BenchRefill = benchRefill,
            TS_METAL_GRAPH_OPTIMIZE = Environment.GetEnvironmentVariable("TS_METAL_GRAPH_OPTIMIZE"),
            TS_PREFILL_CHUNK = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK"),
        },
        BestPrefillTokensPerSecond = bestPp, BestDecodeTokensPerSecond = bestTg, Runs = runs, FirstRunDecodeTokens = firstTokens,
    }));
}

// The cells of the cache the scenario writes: on the runtime the active cache (the primary, or a restored request's slot) or the
// stepped request's slot; legacy's full-attention capacity.
int Capacity()
{
    if (runtime == null) return Field<int>(model, "_kvCacheCapacity");
    return unified ? runtime.SlotCapacityForTests(request!) ?? 0 : runtime.ActiveCacheCapacityForTests;
}

// The sliding-window layers' cells: the runtime's rings (the largest), legacy's ring rows.
int RingCells()
{
    if (runtime == null) return Field<int>(model, "_kvSwaRows");
    int cells = 0;
    for (int l = 0; l < model.Config.NumLayers; l++) cells = Math.Max(cells, runtime.RingCellsForTests(l));
    return cells;
}

int Length() => unified ? runtime!.SlotLengthForTests(request!) ?? 0 : model.CacheSeqLen;

// One unified step of the stepped request: its rows from `start`, the last row's logits. The engine asks for the step's rows
// each step (BatchExecutor's unified prepare): the slot keeps the reservation and grows when a step needs more than it holds.
float[] Unified(int[] tokens, int start)
{
    var batch = (IUnifiedBatchModel)model;
    batch.EnsureSequenceSlot(request!, prompt.Length + 1, start + tokens.Length);
    var outLogits = new float[1][];
    Require(batch.TryForwardUnified(new[] { request! }, new[] { tokens }, new[] { start }, outLogits),
        $"the unified step declined {tokens.Length} rows at {start}");
    return outLogits[0];
}

// A live-cache rewind of the active cache to `target` positions (it holds `history` and the rows past it). Rings that overwrote
// what the next row reads refuse it (the runtime's RingsKeepTruncation): the request then re-prefills its history, as the
// server's turn does.
void Rewind(int target, int[] history)
{
    int cached = model.CacheSeqLen;
    if (model.TryTruncateKVCache(target))
    {
        events.Add($"truncated {cached} -> {model.CacheSeqLen}");
        return;
    }
    events.Add($"truncation {cached} -> {target} refused (the rings overwrote rows the next row reads); re-prefilled {history.Length} tokens");
    model.ResetKVCache();
    model.Forward(history);
}

// What a request served from a cached prefix gets on the runtime: the active cache (`history`'s rows) checkpointed, the source
// moving on with other rows (the checkpoint must be a copy), then the checkpoint cloned into a new request's slot ("checkpoint");
// or the checkpoint written to a checkpoint file, the source and the checkpoint released, other rows forwarded over the released
// cache, and the file read back into a checkpoint that a new request's slot is cloned from ("snapshot": a restarted server's
// persisted prefix).
void Restore(string kind, int[] history)
{
    string key = $"probe-ck-{requestSerial}";
    var sw = Stopwatch.StartNew();
    Require(slots!.TryCheckpointActiveCache(key), $"checkpoint of {model.CacheSeqLen} positions refused");
    double checkpointMs = sw.Elapsed.TotalMilliseconds;
    int[] junk = Enumerable.Range(0, 37).Select(k => history[(k * 7919) % history.Length]).ToArray();
    model.Forward(junk);
    string next = $"probe-{++requestSerial}";
    if (kind == "snapshot")
    {
        using var file = new MemoryStream();
        sw.Restart();
        Require(slots.TryExportRetainedCache(key, file), "checkpoint export refused");
        double exportMs = sw.Elapsed.TotalMilliseconds;
        slots.DiscardRetainedCache(key);
        ReleaseSource();
        model.ResetKVCache();
        model.Forward(history.Reverse().ToArray());
        model.ResetKVCache();
        file.Position = 0;
        sw.Restart();
        Require(slots.TryImportRetainedCache(key, file), "checkpoint import refused");
        events.Add($"checkpoint file of {history.Length} positions: {file.Length} bytes; checkpoint {checkpointMs:F0} ms, " +
            $"export {exportMs:F0} ms, import {sw.Elapsed.TotalMilliseconds:F0} ms");
        Require(slots.TryCloneRetainedCache(key, next), "clone of the imported checkpoint refused");
        slots.DiscardRetainedCache(key);
    }
    else
    {
        if (request != null) slots.RestorePrimaryCache();
        Require(slots.TryCloneRetainedCache(key, next), "clone of the checkpoint refused");
        slots.DiscardRetainedCache(key);
        ReleaseSource();
        events.Add($"checkpoint of {history.Length} positions in {checkpointMs:F0} ms; the source moved on {junk.Length} rows");
    }
    Require(!slots.BindSequenceCache(next), "the clone is the request's slot, not a fresh one");
    request = next;
    Require(model.CacheSeqLen == history.Length, $"the clone holds {model.CacheSeqLen} positions, the checkpoint {history.Length}");
}

// The request the restore replaces gives its cache up: the primary is reset, a slot released (it waits in the pool).
void ReleaseSource()
{
    if (request == null)
    {
        model.ResetKVCache();
        return;
    }
    slots!.RestorePrimaryCache();
    slots.OnSequenceReleased(request);
}

double CosineAgainstReference(float[] values, FileStream reference, string name)
{
    int read = 0;
    while (read < refRow.Length)
    {
        int n = reference.Read(refRow, read, refRow.Length - read);
        if (n == 0) throw new InvalidOperationException($"{name} ends before step {records.Count}.");
        read += n;
    }
    double dot = 0, na = 0, nb = 0;
    for (int v = 0; v < vocab; v++)
    {
        double a = values[v], b = (double)BitConverter.ToHalf(refRow, v * 2);
        dot += a * b;
        na += a * a;
        nb += b * b;
    }
    return dot / Math.Max(Math.Sqrt(na) * Math.Sqrt(nb), 1e-30);
}

static T Field<T>(object target, string name)
{
    for (var type = target.GetType(); type != null; type = type.BaseType)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        if (field != null) return (T)field.GetValue(target)!;
    }
    return default!;
}

static void SetField(object target, string name, object value)
{
    for (var type = target.GetType(); type != null; type = type.BaseType)
    {
        var field = type.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
        if (field != null) { field.SetValue(target, value); return; }
    }
}

static void Require(bool condition, string message)
{
    if (!condition) throw new InvalidOperationException(message);
}

static List<TopEntry> TopK(float[] values, int n, int k)
{
    var best = new List<TopEntry>(k + 1);
    for (int i = 0; i < n; i++)
    {
        float v = values[i];
        if (best.Count == k && v <= best[^1].Logit) continue;
        int at = best.Count;
        while (at > 0 && best[at - 1].Logit < v) at--;
        best.Insert(at, new TopEntry(i, v));
        if (best.Count > k) best.RemoveAt(best.Count - 1);
    }
    return best;
}

// log softmax(values[0..n))[id], in double.
static double LogSoftmaxAt(float[] values, int n, int id)
{
    double max = double.NegativeInfinity;
    for (int i = 0; i < n; i++) if (values[i] > max) max = values[i];
    double sum = 0;
    for (int i = 0; i < n; i++) sum += Math.Exp(values[i] - max);
    return values[id] - max - Math.Log(sum);
}

static void WriteHalf(Stream stream, float[] values, int n)
{
    var bytes = new byte[n * 2];
    for (int i = 0; i < n; i++)
        BitConverter.TryWriteBytes(bytes.AsSpan(i * 2, 2), (System.Half)values[i]);
    stream.Write(bytes);
}

static string Clip(string s, int n) => s.Length <= n ? s : s[..n] + "...";

record TopEntry(int Id, float Logit);
record StepRecord(int Step, int Position, int Argmax, int Chosen, List<TopEntry> Top, int Capacity, double? Cos, double? Cos2);
