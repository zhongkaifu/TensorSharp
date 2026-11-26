// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.GGML;

string? modelPath = null, outputPath = null, referencePath = null;
string backendName = "ggmlcuda";
int iterations = 3, warmup = 1;
int prefixWords = 0;
bool chatDiagnostic = false, compareUnified = false, profile = false, quality = false;
int repeatReads = 0;
int[] widths = [16, 64, 256];
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "--model": modelPath = args[++i]; break;
        case "--backend": backendName = args[++i]; break;
        case "--iterations": iterations = int.Parse(args[++i]); break;
        case "--warmup": warmup = int.Parse(args[++i]); break;
        case "--prefix-words": prefixWords = int.Parse(args[++i]); break;
        case "--chat-diagnostic": chatDiagnostic = true; break;
        case "--compare-unified": compareUnified = true; break;
        case "--repeat-reads": repeatReads = int.Parse(args[++i]); break;
        case "--profile": profile = true; break;
        case "--quality": quality = true; break;
        case "--reference": referencePath = args[++i]; break;
        case "--widths": widths = args[++i].Split(',').Select(int.Parse).ToArray(); break;
        case "--output": outputPath = args[++i]; break;
        default: throw new ArgumentException($"Unknown argument: {args[i]}");
    }
}
int modes = (chatDiagnostic ? 1 : 0) + (compareUnified ? 1 : 0) + (repeatReads > 0 ? 1 : 0) + (quality ? 1 : 0);
if (string.IsNullOrWhiteSpace(modelPath) || iterations < 1 || warmup < 0 || prefixWords < 0 || repeatReads < 0 || modes > 1
    || (referencePath != null && !quality))
    throw new ArgumentException("Usage: JevProbe --model model.gguf --backend ggmlcuda --iterations 3 --warmup 1 --widths 16,64,256 [--prefix-words N] [--chat-diagnostic | --compare-unified | --repeat-reads 4 | --quality [--reference other-backend-quality.json]] [--profile] --output artifacts/jev/projection.json");
BackendType backend = backendName.Replace("_", "").ToLowerInvariant() switch
{
    "cpu" => BackendType.Cpu, "cuda" => BackendType.Cuda, "mlx" => BackendType.Mlx,
    "ggmlcpu" => BackendType.GgmlCpu, "ggmlcuda" => BackendType.GgmlCuda,
    "ggmlmetal" => BackendType.GgmlMetal, "ggmlvulkan" => BackendType.GgmlVulkan,
    _ => throw new ArgumentException($"Unknown backend: {backendName}")
};
var started = DateTimeOffset.UtcNow;
var loadTimer = Stopwatch.StartNew();
using var shutdown = new ProbeShutdown(backend is BackendType.GgmlCpu or BackendType.GgmlCuda
    or BackendType.GgmlMetal or BackendType.GgmlVulkan);
using var model = (DiffusionGemmaModel)ModelBase.Create(Path.GetFullPath(modelPath), backend);
loadTimer.Stop();
if (chatDiagnostic)
{
    int chatResult = RunChatDiagnostic(model, backend, outputPath, started);
    if (profile) model.PrintForwardTiming();
    return chatResult;
}
if (quality)
{
    int qualityResult = RunQuality(model, backend, widths, referencePath, outputPath, started, modelPath!);
    if (profile) model.PrintForwardTiming();
    return qualityResult;
}
string text = "Classify: I love this product and would buy it again.\nReturn a: A for positive, B for negative, C for neutral. Return b: A for a purchase recommendation, B otherwise.";
if (prefixWords > 0) text = string.Concat(Enumerable.Repeat("irrelevant ", prefixWords)) + "\n" + text;
var renderer = new GgufPromptRenderer();
string rendered = renderer.Render(model.Config.ChatTemplate,
    [new ChatMessage { Role = "user", Content = text }], addGenerationPrompt: true,
    architecture: model.Config.Architecture);
int[] prompt = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
int[] labelIds = new[] { "A", "B", "C" }.Select(label => model.Tokenizer.Encode(label, false).Single()).ToArray();
// The timed modes start with one untimed read so the first measured read pays its prompt's
// prefill but not the process's lazy initialization, which it used to charge to whichever case
// (or round) ran first.
double processWarmupMs = compareUnified ? 0 : WarmUpProcess(model, prompt, labelIds);
if (repeatReads > 0)
{
    int code = RunRepeatedReads(model, prompt, labelIds, widths, repeatReads, iterations, backend, outputPath, started,
        modelPath!, processWarmupMs);
    if (profile) model.PrintForwardTiming();
    return code;
}
if (compareUnified)
{
    int code = RunUnifiedComparison(model, prompt, labelIds, widths, backend, outputPath, started, modelPath!);
    if (profile) model.PrintForwardTiming();
    return code;
}
var cases = new List<object>();
double maxError = 0;
bool deterministic = true;
foreach (int width in widths)
{
    if (width < 1 || width > model.CanvasLength) throw new ArgumentOutOfRangeException(nameof(widths));
    foreach (int fieldCount in new[] { 1, 2 })
    {
        var template = model.Tokenizer.Encode("<|channel>thought\n<channel|>", false).ToList();
        var positions = new List<int>();
        var labels = new List<int[]>();
        for (int f = 0; f < fieldCount; f++)
        {
            template.AddRange(model.Tokenizer.Encode($"{(char)('a' + f)}: ", false));
            positions.Add(template.Count);
            template.Add(labelIds[0]);
            template.AddRange(model.Tokenizer.Encode("\n", false));
            labels.Add(f == 0 ? labelIds : [labelIds[1], labelIds[0]]);
        }
        template.AddRange(model.Tokenizer.Encode("<turn|>", false));
        if (template.Count > width)
            throw new ArgumentException($"Width {width} cannot hold the {fieldCount}-field template ({template.Count} tokens).");
        var canvas = new int[width]; // Jev padding token is zero.
        template.CopyTo(canvas);
        int[] original = (int[])canvas.Clone();
        int[] requestedPositions = positions.ToArray();
        int[][] requestedLabels = labels.ToArray();
        model.ClearStructuredCache();
        var timer = Stopwatch.StartNew();
        float[][] first = model.ReadStructured(prompt, canvas, requestedPositions, requestedLabels);
        double coldMs = timer.Elapsed.TotalMilliseconds;
        float[][] reference = model.ReadStructuredReference(prompt, canvas, requestedPositions, requestedLabels);
        double caseError = Difference(first, reference);
        // The first selected/reference calls establish correctness and include lazy initialization.
        // Additional warmups give allocators and GPU clocks time to settle for each matrix shape.
        for (int iteration = 0; iteration < warmup; iteration++)
        {
            float[][] warmedSelected = model.ReadStructured(prompt, canvas, requestedPositions, requestedLabels);
            float[][] warmedReference = model.ReadStructuredReference(prompt, canvas, requestedPositions, requestedLabels);
            caseError = Math.Max(caseError, Difference(warmedSelected, warmedReference));
            deterministic &= Difference(first, warmedSelected) == 0;
        }
        var sparseMs = new List<double>();
        var fullMs = new List<double>();
        for (int iteration = 0; iteration < iterations; iteration++)
        {
            // Alternate ordering to reduce clock/thermal bias. Both paths reuse identical prompt K/V.
            foreach (bool full in iteration % 2 == 0 ? new[] { false, true } : new[] { true, false })
            {
                timer.Restart();
                float[][] result = full
                    ? model.ReadStructuredReference(prompt, canvas, requestedPositions, requestedLabels)
                    : model.ReadStructured(prompt, canvas, requestedPositions, requestedLabels);
                double elapsed = timer.Elapsed.TotalMilliseconds;
                (full ? fullMs : sparseMs).Add(elapsed);
                caseError = Math.Max(caseError, Difference(result, full ? first : reference));
                if (!full) deterministic &= Difference(result, first) == 0;
            }
        }
        if (!canvas.AsSpan().SequenceEqual(original)) throw new InvalidOperationException("Structured inference mutated its seed canvas.");
        maxError = Math.Max(maxError, caseError);
        double sparseMedian = Median(sparseMs), fullMedian = Median(fullMs);
        cases.Add(new
        {
            canvasWidth = width, fields = fieldCount, promptTokens = prompt.Length,
            coldSelectedIncludingPrefillMs = coldMs,
            cachedSelectedMs = sparseMs, cachedFullVocabularyMs = fullMs,
            selectedMedianMs = sparseMedian, fullMedianMs = fullMedian,
            fullOverSelectedSpeedup = fullMedian / sparseMedian,
            selectedLogitElements = fieldCount * labelIds.Length,
            fullLogitElements = (long)width * model.VocabSize,
            maxProbabilityDifference = caseError, probabilities = first
        });
        Console.WriteLine($"width={width} fields={fieldCount}: selected={sparseMedian:F2}ms full={fullMedian:F2}ms " +
            $"speedup={fullMedian / sparseMedian:F3}x maxProbabilityDifference={caseError:G8}");
    }
}
var report = new
{
    startedUtc = started, completedUtc = DateTimeOffset.UtcNow,
    model = Path.GetFullPath(modelPath), modelBytes = new FileInfo(modelPath).Length,
    backend = backend.ToString(), os = Environment.OSVersion.ToString(),
    cpuLogicalProcessors = Environment.ProcessorCount, loadMs = loadTimer.Elapsed.TotalMilliseconds,
    processWarmupReadMs = processWarmupMs,
    iterations, additionalWarmupsPerPath = warmup, deterministic, maxProbabilityDifference = maxError,
    passed = deterministic && maxError <= 0.01,
    limitations = new[]
    {
        "Same-model selected-head versus full-vocabulary TensorSharp reference; not a vLLM implementation comparison.",
        "Warm measurements reuse the prompt K/V; cold selected includes prefill. Both include canvas transformer execution.",
        "One untimed read (processWarmupReadMs) absorbs lazy backend initialization; every cold selected call then pays only its prompt prefill. Additional warmups occur after the cold measurement.",
        "Numerical equivalence and latency only; this probe is not a task-accuracy benchmark.",
        "Different matrix widths may choose different quantized GEMM tilings; maximum probability tolerance is 0.01."
    },
    cases
};
WriteReport(report, outputPath);
if (profile) model.PrintForwardTiming();
return report.passed ? 0 : 1;

// Jev auto sampling re-reads ONE prompt with freshly seeded canvases (JevCompiler.Canvas puts a
// random token at every answer position). The first read of a prompt pays its prefill; with
// prompt-KV caching every later read only runs the canvas. Rounds are separated by
// ClearStructuredCache so each one starts cold. The first read is also what a request with a new
// prompt costs, so it is reported next to the later ones rather than folded into one ratio.
static int RunRepeatedReads(DiffusionGemmaModel model, int[] prompt, int[] labelIds, int[] widths, int reads, int rounds,
    BackendType backend, string? outputPath, DateTimeOffset started, string modelPath, double processWarmupMs)
{
    var samples = new List<object>();
    foreach (int width in widths)
    {
        int[] template = BuildTemplate(model, labelIds, fieldCount: 1, out int[] positions);
        if (template.Length > width) throw new ArgumentException($"Width {width} cannot hold the template ({template.Length} tokens).");
        int[][] labels = [labelIds];
        var firstMs = new List<double>();
        var laterMs = new List<double>();
        for (int round = 0; round < rounds; round++)
        {
            model.ClearStructuredCache();
            for (int read = 0; read < reads; read++)
            {
                var canvas = new int[width];
                template.CopyTo(canvas, 0);
                var random = new Random(1234 + read * 7919);
                foreach (int pos in positions) canvas[pos] = random.Next(model.VocabSize);
                // The printed stage profile (--profile) covers the later reads of the last round only.
                if (round == rounds - 1 && read == 1) model.ResetDiffusionStageTiming();
                var timer = Stopwatch.StartNew();
                model.ReadStructured(prompt, canvas, positions, labels);
                (read == 0 ? firstMs : laterMs).Add(timer.Elapsed.TotalMilliseconds);
            }
        }
        double first = Median(firstMs), later = laterMs.Count > 0 ? Median(laterMs) : double.NaN;
        samples.Add(new { canvasWidth = width, promptTokens = prompt.Length, reads, rounds, firstReadMs = firstMs,
            laterReadMs = laterMs, firstMedianMs = first, laterMedianMs = later, firstOverLater = first / later,
            totalMedianMsPerPrompt = first + (reads - 1) * later });
        Console.WriteLine($"width={width}: first (new-prompt) read {first:F1} ms, later reads {later:F1} ms (x{first / later:F2}), " +
            $"{reads} reads of one prompt = {first + (reads - 1) * later:F0} ms");
    }
    WriteReport(new
    {
        startedUtc = started, completedUtc = DateTimeOffset.UtcNow, mode = "repeat-reads", model = Path.GetFullPath(modelPath),
        backend = backend.ToString(), promptKvCache = model.SupportsPromptKvCache, processWarmupReadMs = processWarmupMs, samples,
    }, outputPath);
    return 0;
}

// Prompt-KV reads against the unified [prompt|canvas] forward on the SAME backend and model instance.
static int RunUnifiedComparison(DiffusionGemmaModel model, int[] prompt, int[] labelIds, int[] widths,
    BackendType backend, string? outputPath, DateTimeOffset started, string modelPath)
{
    if (!model.SupportsPromptKvCache)
        throw new ArgumentException("The unified comparison needs prompt-KV caching (not disabled, and supported by the backend).");
    var samples = new List<object>();
    double worst = 0;
    foreach (int width in widths)
    {
        foreach (int fieldCount in new[] { 1, 2 })
        {
            int[] template = BuildTemplate(model, labelIds, fieldCount, out int[] positions);
            if (template.Length > width) continue;
            var canvas = new int[width];
            template.CopyTo(canvas, 0);
            int[][] labels = positions.Select((_, f) => f == 0 ? labelIds : new[] { labelIds[1], labelIds[0] }).ToArray();
            model.ClearStructuredCache();
            var timer = Stopwatch.StartNew();
            float[][] cached = model.ReadStructured(prompt, canvas, positions, labels);
            double cachedMs = timer.Elapsed.TotalMilliseconds;
            bool saved = model.SupportsPromptKvCache;
            model.SupportsPromptKvCache = false;
            float[][] unified;
            try
            {
                timer.Restart();
                unified = model.ReadStructured(prompt, canvas, positions, labels);
            }
            finally { model.SupportsPromptKvCache = saved; }
            double unifiedMs = timer.Elapsed.TotalMilliseconds;
            double diff = Difference(cached, unified);
            worst = Math.Max(worst, diff);
            samples.Add(new { canvasWidth = width, fields = fieldCount, promptTokens = prompt.Length,
                cachedColdMs = cachedMs, unifiedMs, maxProbabilityDifference = diff, cached, unified });
            Console.WriteLine($"width={width} fields={fieldCount}: prompt-KV vs unified max |dp| = {diff:G6}");
        }
    }
    bool passed = worst <= 1e-4;
    WriteReport(new
    {
        startedUtc = started, completedUtc = DateTimeOffset.UtcNow, mode = "compare-unified", model = Path.GetFullPath(modelPath),
        backend = backend.ToString(), maxProbabilityDifference = worst, passed, samples,
    }, outputPath);
    return passed ? 0 : 1;
}

static double WarmUpProcess(DiffusionGemmaModel model, int[] prompt, int[] labelIds)
{
    int[] template = BuildTemplate(model, labelIds, fieldCount: 1, out int[] positions);
    var canvas = new int[Math.Max(16, template.Length)];
    template.CopyTo(canvas, 0);
    var timer = Stopwatch.StartNew();
    model.ReadStructured(prompt, canvas, positions, [labelIds]);
    double ms = timer.Elapsed.TotalMilliseconds;
    model.ClearStructuredCache();
    model.ResetDiffusionStageTiming();
    return ms;
}

// Task-level quality probe. Jev-style requests - the server's system text, the "id: label"
// answer template after the thought scaffold, the <turn|> terminator, <pad> filler and one random
// token per answer slot (JevCompiler.Canvas with seed 42) - over states whose answers are not in
// doubt. It reports every read's label probabilities and top label; with --reference (the same
// probe run on another backend or build) also the largest probability difference and whether the
// top labels agree. One structured read of this model moves with last-bit differences (Q8
// activations, top-8 routing), so a single case proves little either way; the set is the evidence.
static int RunQuality(DiffusionGemmaModel model, BackendType backend, int[] widths, string? referencePath,
    string? outputPath, DateTimeOffset started, string modelPath)
{
    string[] mood = ["positive", "negative", "neutral"], yesNo = ["yes", "no"], topic = ["science", "sports", "cooking"];
    (string State, (string Id, string Instructions, string[] Labels, string Expected)[] Questions)[] cases =
    [
        ("{\"review\":\"I love this product and would buy it again.\"}",
            [("mood", "What is the sentiment of the review?", mood, "positive"), ("color", "Is a color mentioned?", yesNo, "no")]),
        ("{\"review\":\"The blender broke after two days and support never answered my emails. Terrible.\"}",
            [("mood", "What is the sentiment of the review?", mood, "negative"), ("color", "Is a color mentioned?", yesNo, "no")]),
        ("{\"text\":\"The red car was parked next to the train station.\"}",
            [("color", "Is a color mentioned?", yesNo, "yes"), ("vehicle", "Is a vehicle mentioned?", yesNo, "yes")]),
        ("{\"claim\":\"Paris is the capital of France.\"}",
            [("correct", "Is the claim factually correct?", yesNo, "yes"), ("country", "Is a country mentioned?", yesNo, "yes")]),
        ("{\"claim\":\"Water boils at 10 degrees Celsius at sea level.\"}",
            [("correct", "Is the claim factually correct?", yesNo, "no"), ("topic", "What is the topic?", topic, "science")]),
        ("{\"message\":\"My flight was delayed by six hours and I missed my meeting.\"}",
            [("mood", "What is the sentiment of the message?", mood, "negative"), ("travel", "Is travel mentioned?", yesNo, "yes")]),
        ("{\"message\":\"The meeting is scheduled for Monday at 9am in room 4B.\"}",
            [("time", "Does the message contain a time of day?", yesNo, "yes"), ("complaint", "Is the message a complaint?", yesNo, "no")]),
        ("{\"text\":\"Whisk two eggs with milk, pour into a hot buttered pan and fold when set.\"}",
            [("topic", "What is the topic?", topic, "cooking"), ("complaint", "Is the text a complaint?", yesNo, "no")]),
        ("{\"text\":\"The striker scored twice in the final minutes and the home team won the cup.\"}",
            [("topic", "What is the topic?", topic, "sports"), ("mood", "What is the sentiment of the text?", mood, "positive")]),
    ];
    int eos = model.Tokenizer.LookupToken("<turn|>");
    if (eos < 0) throw new InvalidOperationException("The tokenizer has no <turn|> token.");
    int pad = model.Tokenizer.LookupToken("<pad>");
    if (pad < 0) pad = model.MaskTokenId;
    int[] scaffold = model.Tokenizer.Encode("<|channel>thought\n<channel|>", false).ToArray();
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    JsonElement? reference = referencePath != null ? JsonDocument.Parse(File.ReadAllText(referencePath)).RootElement : null;

    var reads = new List<object>();
    int scored = 0, correct = 0, compared = 0, agreeing = 0;
    double expectedSum = 0, expectedMin = 1, worstDiff = 0, diffSum = 0;
    var timer = new Stopwatch();
    for (int ci = 0; ci < cases.Length; ci++)
    {
        var (state, questions) = cases[ci];
        var system = new System.Text.StringBuilder("Answer a fixed set of questions about the state the user provides. " +
            "Each question lists its allowed answers; reply with exactly one label per question.\n");
        foreach (var q in questions)
        {
            system.Append("\nQuestion ").Append(q.Id).Append(": ").Append(q.Instructions).Append('\n');
            foreach (string label in q.Labels) system.Append("  ").Append(label).Append(": ").Append(label).Append('\n');
        }
        system.Append("\nReply with one line per question, in this order, formatted as \"id: label\".");
        int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
            [new ChatMessage { Role = "system", Content = system.ToString() }, new ChatMessage { Role = "user", Content = state }],
            model.Config.Architecture, addGenerationPrompt: true, enableThinking: false).ToArray();

        // JevCompiler.Resolve: each question's slot is the one template token its labels change.
        int[] Answer(int[] selected) => scaffold.Concat(model.Tokenizer.Encode(
            string.Join("\n", questions.Select((q, i) => q.Id + ": " + q.Labels[selected[i]])), false)).ToArray();
        var selection = new int[questions.Length];
        int[] template = Answer(selection);
        var positions = new int[questions.Length];
        var labelIds = new int[questions.Length][];
        for (int qi = 0; qi < questions.Length; qi++)
        {
            labelIds[qi] = new int[questions[qi].Labels.Length];
            positions[qi] = -1;
            for (int li = 1; li < labelIds[qi].Length; li++)
            {
                selection[qi] = li;
                int[] other = Answer(selection);
                selection[qi] = 0;
                int[] changed = Enumerable.Range(0, template.Length).Where(p => other.Length == template.Length && other[p] != template[p]).ToArray();
                if (other.Length != template.Length || changed.Length != 1 || (positions[qi] >= 0 && positions[qi] != changed[0]))
                    throw new InvalidOperationException($"case {ci} question '{questions[qi].Id}': labels must occupy one shared template token");
                positions[qi] = changed[0];
                labelIds[qi][li] = other[changed[0]];
            }
            labelIds[qi][0] = template[positions[qi]];
        }

        model.ClearStructuredCache();
        foreach (int width in widths)
        {
            if (template.Length + 1 > width) throw new ArgumentException($"Width {width} cannot hold case {ci}'s template ({template.Length + 1} tokens).");
            var canvas = new int[width];
            Array.Fill(canvas, pad);
            template.CopyTo(canvas, 0);
            canvas[template.Length] = eos;
            var random = new Random(42);
            foreach (int pos in positions) canvas[pos] = random.Next(model.VocabSize);
            timer.Restart();
            float[][] probabilities = model.ReadStructured(prompt, canvas, positions, labelIds);
            double ms = timer.Elapsed.TotalMilliseconds;

            var answers = new List<object>();
            for (int qi = 0; qi < questions.Length; qi++)
            {
                var q = questions[qi];
                float[] p = probabilities[qi];
                int top = Array.IndexOf(p, p.Max());
                int expected = Array.IndexOf(q.Labels, q.Expected);
                scored++;
                if (top == expected) correct++;
                expectedSum += p[expected];
                expectedMin = Math.Min(expectedMin, p[expected]);
                double? refDiff = null;
                bool? refAgrees = null;
                if (reference is JsonElement root)
                {
                    JsonElement refRead = root.GetProperty("reads").EnumerateArray()
                        .First(r => r.GetProperty("caseIndex").GetInt32() == ci && r.GetProperty("canvasWidth").GetInt32() == width);
                    float[] rp = refRead.GetProperty("answers")[qi].GetProperty("probabilities").EnumerateArray().Select(e => e.GetSingle()).ToArray();
                    refDiff = p.Zip(rp, (a, b) => (double)Math.Abs(a - b)).Max();
                    refAgrees = top == Array.IndexOf(rp, rp.Max());
                    compared++;
                    if (refAgrees == true) agreeing++;
                    worstDiff = Math.Max(worstDiff, refDiff.Value);
                    diffSum += refDiff.Value;
                }
                answers.Add(new { id = q.Id, labels = q.Labels, probabilities = p, top = q.Labels[top], expected = q.Expected,
                    correct = top == expected, referenceMaxDifference = refDiff, referenceTopAgrees = refAgrees });
            }
            reads.Add(new { caseIndex = ci, state, canvasWidth = width, promptTokens = prompt.Length, readMs = ms, answers });
            Console.WriteLine($"case {ci} width {width}: " + string.Join("  ", questions.Select((q, qi) =>
                $"{q.Id}={q.Labels[Array.IndexOf(probabilities[qi], probabilities[qi].Max())]} ({probabilities[qi].Max():F4})")) + $"  {ms:F0} ms");
        }
    }
    var summary = new
    {
        scoredQuestions = scored, correct, accuracy = (double)correct / Math.Max(1, scored),
        meanExpectedProbability = expectedSum / Math.Max(1, scored), minExpectedProbability = expectedMin,
        reference = referencePath, referenceComparedQuestions = compared, referenceTopAgreement = compared > 0 ? (double)agreeing / compared : (double?)null,
        referenceMaxProbabilityDifference = compared > 0 ? worstDiff : (double?)null,
        referenceMeanMaxProbabilityDifference = compared > 0 ? diffSum / compared : (double?)null,
    };
    Console.WriteLine($"Quality: {correct}/{scored} top labels as expected, mean p(expected) {summary.meanExpectedProbability:F4}, " +
        $"min {expectedMin:F4}" + (compared > 0 ? $"; vs reference: top label agrees {agreeing}/{compared}, max |dp| {worstDiff:F4}, mean {diffSum / compared:F4}" : ""));
    WriteReport(new
    {
        startedUtc = started, completedUtc = DateTimeOffset.UtcNow, mode = "quality", model = Path.GetFullPath(modelPath),
        backend = backend.ToString(), promptKvCache = model.SupportsPromptKvCache,
        seed = 42, widths, summary, reads,
    }, outputPath);
    return 0;
}

static int[] BuildTemplate(DiffusionGemmaModel model, int[] labelIds, int fieldCount, out int[] positions)
{
    var template = model.Tokenizer.Encode("<|channel>thought\n<channel|>", false).ToList();
    var pos = new List<int>();
    for (int f = 0; f < fieldCount; f++)
    {
        template.AddRange(model.Tokenizer.Encode($"{(char)('a' + f)}: ", false));
        pos.Add(template.Count);
        template.Add(labelIds[0]);
        template.AddRange(model.Tokenizer.Encode("\n", false));
    }
    template.AddRange(model.Tokenizer.Encode("<turn|>", false));
    positions = pos.ToArray();
    return template.ToArray();
}

static void WriteReport(object report, string? outputPath)
{
    string json = JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true });
    if (outputPath != null)
    {
        string absolute = Path.GetFullPath(outputPath);
        Directory.CreateDirectory(Path.GetDirectoryName(absolute)!);
        File.WriteAllText(absolute, json);
        Console.WriteLine($"Report: {absolute}");
    }
    else Console.WriteLine(json);
}

static double Difference(float[][] a, float[][] b)
{
    if (a.Length != b.Length) throw new InvalidOperationException("Result row count mismatch.");
    double error = 0;
    for (int r = 0; r < a.Length; r++)
    {
        if (a[r].Length != b[r].Length) throw new InvalidOperationException("Result label count mismatch.");
        if (Math.Abs(a[r].Sum() - 1) > 1e-6) throw new InvalidOperationException("Unnormalized label probabilities.");
        for (int c = 0; c < a[r].Length; c++)
        {
            if (!float.IsFinite(a[r][c]) || !float.IsFinite(b[r][c])) throw new InvalidOperationException("Non-finite probability.");
            error = Math.Max(error, Math.Abs(a[r][c] - b[r][c]));
        }
    }
    return error;
}

static double Median(List<double> values)
{
    double[] sorted = values.Order().ToArray();
    return sorted.Length % 2 == 0
        ? (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2
        : sorted[sorted.Length / 2];
}

static int RunChatDiagnostic(DiffusionGemmaModel model, BackendType backend, string? outputPath,
    DateTimeOffset started)
{
    const string text = "What is two plus two? Reply with only 4.";
    const int seed = 42, maxTokens = 256;
    if (model.CanvasLength != maxTokens)
        throw new ArgumentException("This bounded diagnostic requires the 256-token DiffusionGemma canvas.");
    int maxSteps = int.TryParse(Environment.GetEnvironmentVariable("DIFFUSION_STEPS"), out int configuredSteps)
        && configuredSteps > 0 ? configuredSteps : 48;
    var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
    int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
        [new ChatMessage { Role = "user", Content = text }], model.Config.Architecture,
        addGenerationPrompt: true, enableThinking: false).ToArray();
    var parameters = new DiffusionEbParams { Seed = seed, MaxBlocks = 1, MaxDenoisingSteps = maxSteps };
    var sampler = new DiffusionGemmaSampler(model);
    var directSteps = new List<object>();
    var batchedSteps = new List<object>();
    int[] directCanvas = [], batchedCanvas = [];

    var timer = Stopwatch.StartNew();
    List<int> direct = sampler.Generate(prompt, parameters, (_, step, totalSteps, canvas) =>
    {
        directCanvas = (int[])canvas.Clone();
        directSteps.Add(new { step, totalSteps, canvas = Analyze(canvas, includeAllTokens: false) });
    });
    double directMs = timer.Elapsed.TotalMilliseconds;

    DiffusionSeqState state = model.CreateSeqState();
    List<int> batched;
    bool batchedDone;
    timer.Restart();
    try
    {
        var run = new DiffusionSeqRun(prompt, parameters, state, CancellationToken.None,
            (_, step, totalSteps, canvas) =>
            {
                batchedCanvas = (int[])canvas.Clone();
                batchedSteps.Add(new { step, totalSteps, canvas = Analyze(canvas, includeAllTokens: false) });
            });
        // Exactly one block, matching max_tokens=256 on this model. No retries or seed changes.
        sampler.RunBlockBatched([run]);
        batched = new List<int>(run.Response);
        batchedDone = run.Done;
    }
    finally { model.DisposeSeqState(state); }
    double batchedMs = timer.Elapsed.TotalMilliseconds;
    bool pathsEquivalent = direct.SequenceEqual(batched) && directCanvas.AsSpan().SequenceEqual(batchedCanvas);
    bool nonEmpty = direct.Count > 0 && batched.Count > 0;
    var report = new
    {
        startedUtc = started, completedUtc = DateTimeOffset.UtcNow, diagnostic = "arithmetic-chat-seed42",
        backend = backend.ToString(), text, seed, maxTokens, maxSteps, maxBlocks = 1,
        promptTokenIds = prompt, decodedPrompt = model.Tokenizer.Decode(prompt.ToList()),
        selfConditioning = model.SelfConditioningEnabled, deviceSampling = model.SupportsDeviceSampling,
        direct = new { milliseconds = directMs, committedTokenIds = direct, committedCount = direct.Count,
            decodedCommitted = model.Tokenizer.Decode(direct), finalCanvas = Analyze(directCanvas, true), steps = directSteps },
        batched = new { milliseconds = batchedMs, done = batchedDone, committedTokenIds = batched,
            committedCount = batched.Count, decodedCommitted = model.Tokenizer.Decode(batched),
            finalCanvas = Analyze(batchedCanvas, true), steps = batchedSteps },
        pathsEquivalent, nonEmpty, passed = pathsEquivalent && nonEmpty && batchedDone,
        limitations = new[] {
            "One fixed arithmetic prompt, seed42, one block; no retries or seed search.",
            "Direct sampler and one-request batched scheduler core, using the server's prompt renderer and sampler defaults.",
            "Raw argmax canvases are captured by existing preview callbacks before EOS/repetition trimming and output parsing.",
            "Predicted trim diagnostics mirror the current sampler rule; committedCount records its actual result."
        }
    };
    WriteReport(report, outputPath);
    Console.WriteLine($"Chat diagnostic: direct={direct.Count} tokens, batched={batched.Count} tokens, equivalent={pathsEquivalent}, nonEmpty={nonEmpty}");
    return report.passed ? 0 : 1;

    object Analyze(int[] canvas, bool includeAllTokens)
    {
        int firstEos = Array.FindIndex(canvas, id => model.Tokenizer.IsEos(id));
        int cut = firstEos < 0 ? canvas.Length : firstEos;
        int repetitionStart = -1, repetitionStride = 0, repetitionMatches = 0;
        for (int i = 0; i + 1 < cut; i++)
        {
            for (int stride = 1; stride <= 2; stride++)
            {
                int matches = Matches(i, stride);
                if (matches < 6) continue;
                cut = repetitionStart = i;
                repetitionStride = stride;
                repetitionMatches = matches;
                break;
            }
            if (repetitionStart >= 0) break;
        }
        int[] first32 = canvas.Take(32).ToArray();
        return new
        {
            length = canvas.Length, first32TokenIds = first32, decodedFirst32 = model.Tokenizer.Decode(first32.ToList()),
            firstEosPosition = firstEos, firstEosTokenId = firstEos >= 0 ? (int?)canvas[firstEos] : null,
            leadingStrideOneMatches = Matches(0, 1), leadingStrideTwoMatches = Matches(0, 2),
            repetitionStart, repetitionStride, repetitionMatches, predictedTrimCount = cut,
            predictedTrimCause = repetitionStart >= 0 ? "repetition" : firstEos >= 0 ? "eos" : "canvas-complete",
            allTokenIds = includeAllTokens ? canvas : null,
            decodedBeforeFirstEos = includeAllTokens
                ? model.Tokenizer.Decode(canvas.Take(firstEos < 0 ? canvas.Length : firstEos).ToList()) : null
        };

        int Matches(int start, int stride)
        {
            int count = 0;
            for (int i = start; i + stride < canvas.Length && canvas[i] == canvas[i + stride]; i += stride) count++;
            return count;
        }
    }
}

sealed class ProbeShutdown(bool ggml) : IDisposable
{
    // Dispose after the model, on the inference thread: cached native prefill graphs are
    // thread-local and must release their CUDA buffers before the driver shuts down.
    public void Dispose() { if (ggml) GgmlBasicOps.Shutdown(); }
}
