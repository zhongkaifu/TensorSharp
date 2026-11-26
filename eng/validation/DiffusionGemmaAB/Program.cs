// DiffusionGemma validation tool: one source built against the legacy model and against the shared-runtime port, so the
// two can be held to each other on the real checkpoint (the reports and dumps go under the ignored artifacts/).
//
//   --scenario logits   --prompts <file> [--dump <dir>]       teacher-forced canvas logits: a seeded canvas without
//                                                              self-conditioning, then its argmax canvas with SC from step 1
//   --scenario teacher  --prompts <file> --canvas-json <a generate report> [--dump <dir>]
//                                                              teacher-forced: each prompt's canvas is that report's answer
//                                                              (its first canvas tokens, EOS-padded); the canvas tokens'
//                                                              log-probabilities without and with self-conditioning
//   --scenario generate --prompts <file> [--steps 48 --blocks 2 --seed 0]  the EntropyBound sampler's text and timing
//   --scenario bench    [--bench-prompt 512 --bench-steps 16]  prefill and per-step decode time (host and device sampling)
//   --scenario batched  --prompts <file>                      two sequences batched against each alone
//   --scenario structured --prompts <file>                    ReadStructured against its full-vocabulary reference
//   --scenario compare  --a <dir or report> --b <dir or report>  logits dumps / reports of two runs
// Common: --model <gguf> --backend ggml_cuda|cuda|ggml_cpu|cpu|... [--out report.json] [--system <text>]
using System.Diagnostics;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;

string Option(string name, string? fallback)
{
    int index = Array.IndexOf(args, name);
    if (index < 0) return fallback!;
    if (index + 1 >= args.Length) throw new ArgumentException($"Missing value for {name}.");
    return args[index + 1];
}

string scenario = Option("--scenario", "logits");
string output = Option("--out", $"artifacts/diffusiongemma-ab/{scenario}.json");
var report = new Dictionary<string, object?> { ["scenario"] = scenario };

if (scenario == "compare")
{
    Compare(Option("--a", null), Option("--b", null));
    return;
}

string modelPath = Option("--model", null);
string backendName = Option("--backend", "ggml_cuda");
string? promptsFile = Option("--prompts", null);
string? dumpDir = Option("--dump", null);
string? system = Option("--system", null);
string? canvasJson = Option("--canvas-json", null);
int steps = int.Parse(Option("--steps", "48"));
int blocks = int.Parse(Option("--blocks", "2"));
int seed = int.Parse(Option("--seed", "0"));
int benchPrompt = int.Parse(Option("--bench-prompt", "512"));
int benchSteps = int.Parse(Option("--bench-steps", "16"));
if (string.IsNullOrWhiteSpace(modelPath) || !File.Exists(modelPath)) throw new ArgumentException("Pass --model <diffusion-gemma gguf>.");
if (!Enum.TryParse(backendName.Replace("_", string.Empty), ignoreCase: true, out BackendType backend))
    throw new ArgumentException($"Unknown backend '{backendName}'.");
MoeCpuOffloadConfig.ConfigureFromEnvironment();
KvCacheDtypeConfig.ConfigureFromEnvironment();

var load = Stopwatch.StartNew();
using var model = (DiffusionGemmaModel)ModelBase.Create(Path.GetFullPath(modelPath), backend);
report["model"] = Path.GetFileName(modelPath);
report["backend"] = backend.ToString();
report["type"] = model.GetType().FullName;
report["load_ms"] = load.Elapsed.TotalMilliseconds;
report["canvas"] = model.CanvasLength;
report["device_sampling"] = model.SupportsDeviceSampling;
report["prompt_kv"] = model.SupportsPromptKvCache;
Console.WriteLine($"[ab] {model.GetType().Name} loaded in {load.Elapsed.TotalMilliseconds:F0} ms; device sampling {model.SupportsDeviceSampling}, " +
    $"prompt KV {model.SupportsPromptKvCache}");
int C = model.CanvasLength, vocab = model.VocabSize;
IPromptRenderer renderer = new GgufPromptRenderer();

int[] Render(string text)
{
    var messages = new List<ChatMessage>();
    if (!string.IsNullOrEmpty(system)) messages.Add(new ChatMessage { Role = "system", Content = system });
    messages.Add(new ChatMessage { Role = "user", Content = text });
    string rendered = renderer.Render(model.Config.ChatTemplate, messages, addGenerationPrompt: true, architecture: model.Config.Architecture);
    return model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();
}

string[] Prompts() => File.ReadAllLines(promptsFile ?? throw new ArgumentException("Pass --prompts <file>.")).Where(l => l.Trim().Length > 0).ToArray();

int[] SeededCanvas(int salt)
{
    var rng = new Random(seed * 7919 + salt);
    var canvas = new int[C];
    for (int i = 0; i < C; i++) canvas[i] = rng.Next(vocab);
    return canvas;
}

int[] Argmaxes(float[] logits)
{
    var ids = new int[C];
    for (int p = 0; p < C; p++)
    {
        var row = logits.AsSpan(p * vocab, vocab);
        int best = 0;
        for (int i = 1; i < vocab; i++) if (row[i] > row[best]) best = i;
        ids[p] = best;
    }
    return ids;
}

void Dump(string name, float[] logits)
{
    if (dumpDir == null) return;
    Directory.CreateDirectory(dumpDir);
    using var f = File.Create(Path.Combine(dumpDir, name + ".f32"));
    f.Write(System.Runtime.InteropServices.MemoryMarshal.AsBytes(logits.AsSpan(0, C * vocab)));
}

switch (scenario)
{
    case "logits":
    {
        var cases = new List<object>();
        string[] prompts = Prompts();
        for (int n = 0; n < prompts.Length; n++)
        {
            int[] prompt = Render(prompts[n]);
            var sw = Stopwatch.StartNew();
            model.PrefillPrompt(prompt);
            double prefillMs = sw.Elapsed.TotalMilliseconds;
            int[] canvas = SeededCanvas(n);
            sw.Restart();
            float[] first = (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
            double firstMs = sw.Elapsed.TotalMilliseconds;
            int[] am = Argmaxes(first);
            sw.Restart();
            float[] second = (float[])model.DecodeCanvas(am, first, 1f, 1f / 0.8f).Clone();
            double secondMs = sw.Elapsed.TotalMilliseconds;
            Dump($"p{n}-s1", first);
            Dump($"p{n}-s2", second);
            cases.Add(new { prompt = prompts[n], tokens = prompt.Length, prefill_ms = prefillMs, step1_ms = firstMs, step2_ms = secondMs,
                argmax1 = am, argmax2 = Argmaxes(second), text1 = model.Tokenizer.Decode(am.ToList()) });
            Console.WriteLine($"[ab] logits p{n}: {prompt.Length} tokens, prefill {prefillMs:F0} ms, steps {firstMs:F0}/{secondMs:F0} ms: " +
                model.Tokenizer.Decode(am.Take(24).ToList()).Replace("\n", "\\n"));
        }
        report["cases"] = cases;
        break;
    }
    case "teacher":
    {
        using JsonDocument source = JsonDocument.Parse(File.ReadAllText(canvasJson ?? throw new ArgumentException("Pass --canvas-json <generate report>.")));
        JsonElement[] answers = source.RootElement.GetProperty("cases").EnumerateArray().ToArray();
        var cases = new List<object>();
        string[] prompts = Prompts();
        int eos = model.Tokenizer.EosTokenIds is { Length: > 0 } e ? e[0] : 1;
        for (int n = 0; n < prompts.Length && n < answers.Length; n++)
        {
            int[] prompt = Render(prompts[n]);
            int[] ids = answers[n].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] canvas = Enumerable.Range(0, C).Select(i => i < ids.Length ? ids[i] : eos).ToArray();
            model.PrefillPrompt(prompt);
            float[] plain = (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
            float[] conditioned = (float[])model.DecodeCanvas(canvas, plain, 1f, 1f / 0.8f).Clone();
            Dump($"t{n}-plain", plain);
            Dump($"t{n}-sc", conditioned);
            (double MeanLogp, int Agree) Score(float[] logits)
            {
                double sum = 0;
                int agree = 0;
                for (int pos = 0; pos < C; pos++)
                {
                    var row = logits.AsSpan(pos * vocab, vocab);
                    double max = double.NegativeInfinity;
                    int best = 0;
                    for (int i = 0; i < vocab; i++) if (row[i] > max) { max = row[i]; best = i; }
                    double z = 0;
                    for (int i = 0; i < vocab; i++) z += Math.Exp(row[i] - max);
                    sum += row[canvas[pos]] - max - Math.Log(z);
                    if (best == canvas[pos]) agree++;
                }
                return (sum / C, agree);
            }
            var a = Score(plain);
            var b = Score(conditioned);
            cases.Add(new { prompt = prompts[n], mean_logp_plain = a.MeanLogp, agree_plain = a.Agree, mean_logp_sc = b.MeanLogp, agree_sc = b.Agree });
            Console.WriteLine($"[ab] teacher t{n}: mean log p(canvas) plain {a.MeanLogp:F4} (argmax = canvas {a.Agree}/{C}), " +
                $"with SC {b.MeanLogp:F4} ({b.Agree}/{C})");
        }
        report["cases"] = cases;
        break;
    }
    case "generate":
    {
        var cases = new List<object>();
        foreach (string text in Prompts())
        {
            int[] prompt = Render(text);
            var sampler = new DiffusionGemmaSampler(model);
            int forwards = 0;
            var sw = Stopwatch.StartNew();
            List<int> generated = sampler.Generate(prompt, new DiffusionEbParams { MaxDenoisingSteps = steps, Seed = seed, MaxBlocks = blocks },
                (_, _, _, _) => forwards++);
            double ms = sw.Elapsed.TotalMilliseconds;
            string decoded = model.Tokenizer.Decode(generated);
            cases.Add(new { prompt = text, tokens = prompt.Length, generated = generated.Count, forwards, total_ms = ms, ms_per_step = ms / Math.Max(1, forwards),
                ids = generated, text = decoded });
            Console.WriteLine($"[ab] generate: {generated.Count} tokens, {forwards} steps, {ms:F0} ms ({ms / Math.Max(1, forwards):F1} ms/step)\n{decoded}\n---");
        }
        report["cases"] = cases;
        break;
    }
    case "bench":
    {
        var rng = new Random(seed);
        int[] prompt = new int[benchPrompt];
        prompt[0] = 2;
        for (int i = 1; i < prompt.Length; i++) prompt[i] = 1000 + rng.Next(30000);
        var runs = new List<object>();
        for (int round = 0; round < 3; round++)
        {
            prompt[1] = 1000 + round;   // a fresh prompt each round (no prefix reuse)
            var sw = Stopwatch.StartNew();
            bool cached = model.SupportsPromptKvCache;   // a model without a prompt cache runs [prompt | canvas] every step
            if (cached) model.PrefillPrompt(prompt);
            double prefill = sw.Elapsed.TotalMilliseconds;
            int[] canvas = SeededCanvas(round);
            int[] unified = prompt.Concat(canvas).ToArray();
            var host = new List<double>();
            float[]? prev = null;
            for (int s = 0; s < benchSteps; s++)
            {
                sw.Restart();
                prev = cached ? model.DecodeCanvas(canvas, prev, s == 0 ? 0f : 1f, 1.25f)
                    : model.ForwardCanvas(unified, prompt.Length, prev, s == 0 ? 0f : 1f, 1.25f);
                host.Add(sw.Elapsed.TotalMilliseconds);
            }
            var device = new List<double>();
            if (model.SupportsDeviceSampling)
            {
                int K = Math.Max(1, model.SelfCondTopK);
                int[] am = new int[C], sm = new int[C];
                float[] en = new float[C], u = new float[C];
                int[][] tt = { new int[C * K], new int[C * K] };
                float[][] tp = { new float[C * K], new float[C * K] };
                for (int i = 0; i < C; i++) u[i] = (float)rng.NextDouble();
                for (int s = 0; s < benchSteps; s++)
                {
                    sw.Restart();
                    bool ok = model.DecodeCanvasSampled(canvas, s == 0 ? null : tt[(s + 1) & 1], s == 0 ? null : tp[(s + 1) & 1], s == 0 ? 0f : 1f,
                        1.25f, u, model.SelfCondTopK, am, en, sm, tt[s & 1], tp[s & 1]);
                    device.Add(sw.Elapsed.TotalMilliseconds);
                    if (!ok) { device.Clear(); break; }
                }
            }
            double Median(List<double> v) => v.Count == 0 ? 0 : v.Skip(1).OrderBy(x => x).ElementAt((v.Count - 1) / 2);
            runs.Add(new { prefill_ms = prefill, prefill_tok_s = benchPrompt / (prefill / 1000), host_step_ms = Median(host), device_step_ms = Median(device),
                host_steps = host, device_steps = device });
            Console.WriteLine($"[ab] bench round {round}: prefill {benchPrompt} in {prefill:F0} ms ({benchPrompt / (prefill / 1000):F0} tok/s); " +
                $"step host {Median(host):F1} ms, device {Median(device):F1} ms");
        }
        report["runs"] = runs;
        break;
    }
    case "batched":
    {
        string[] prompts = Prompts().Take(2).ToArray();
        var seqs = prompts.Select(_ => model.CreateSeqState()).ToArray();
        try
        {
            var canvases = new int[seqs.Length][];
            var alone = new float[seqs.Length][];
            JsonElement[]? answers = canvasJson == null ? null
                : JsonDocument.Parse(File.ReadAllText(canvasJson)).RootElement.GetProperty("cases").EnumerateArray().ToArray();
            for (int b = 0; b < seqs.Length; b++)
            {
                model.PrefillSeq(seqs[b], Render(prompts[b]));
                // A teacher canvas (the report's answer) where given, else seeded noise.
                if (answers != null)
                {
                    int[] ids = answers[b].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
                    int eos = model.Tokenizer.EosTokenIds is { Length: > 0 } e ? e[0] : 1;
                    canvases[b] = Enumerable.Range(0, C).Select(i => i < ids.Length ? ids[i] : eos).ToArray();
                }
                else canvases[b] = SeededCanvas(b);
                alone[b] = (float[])model.DecodeCanvasSeq(seqs[b], canvases[b], null, 0f, 1f).Clone();
            }
            var sw = Stopwatch.StartNew();
            float[][] together = model.DecodeCanvasBatched(seqs, canvases, new float[seqs.Length][], new float[seqs.Length], Enumerable.Repeat(1f, seqs.Length).ToArray());
            double ms = sw.Elapsed.TotalMilliseconds;
            var cases = new List<object>();
            for (int b = 0; b < seqs.Length; b++)
            {
                int[] a1 = Argmaxes(alone[b]), a2 = Argmaxes(together[b]);
                cases.Add(new { prompt = prompts[b], argmax_agree = a1.Zip(a2).Count(z => z.First == z.Second) / (double)C, rel_l2 = RelL2(together[b], alone[b]) });
                Console.WriteLine($"[ab] batched b{b}: argmax agree {a1.Zip(a2).Count(z => z.First == z.Second)}/{C}, rel L2 {RelL2(together[b], alone[b]):E2}");
            }
            report["batched_ms"] = ms;
            report["cases"] = cases;
        }
        finally
        {
            foreach (var s in seqs) model.DisposeSeqState(s);
        }
        break;
    }
    case "batchdiag":
    {
        // Sequence A's canvas logits in every arrangement a batched step can give it, against DecodeCanvasSeq alone.
        string[] prompts = Prompts();
        bool sc = model.SelfConditioningEnabled;
        model.SelfConditioningEnabled = false;
        var a = model.CreateSeqState(); var a2 = model.CreateSeqState(); var b = model.CreateSeqState();
        try
        {
            int[] pa = Render(prompts[0]), pb = Render(prompts[1]);
            model.PrefillSeq(a, pa); model.PrefillSeq(a2, pa); model.PrefillSeq(b, pb);
            int[] canvas = Enumerable.Repeat(model.MaskTokenId, C).ToArray();
            float[] reference = (float[])model.DecodeCanvasSeq(a, canvas, null, 0f, 1f).Clone();
            float[] Batch(DiffusionSeqState[] seqs, int at)
                => (float[])model.DecodeCanvasBatched(seqs, seqs.Select(_ => canvas).ToArray(), new float[seqs.Length][], new float[seqs.Length],
                    Enumerable.Repeat(1f, seqs.Length).ToArray())[at].Clone();
            var cases = new (string Name, float[] Logits)[]
            {
                ("seq alone again", (float[])model.DecodeCanvasSeq(a, canvas, null, 0f, 1f).Clone()),
                ("batched [A]", Batch(new[] { a }, 0)),
                ("batched [A, A']", Batch(new[] { a, a2 }, 0)),
                ("batched [A', A] second", Batch(new[] { a2, a }, 1)),
                ("batched [A, B]", Batch(new[] { a, b }, 0)),
                ("batched [B, A] second", Batch(new[] { b, a }, 1)),
            };
            var rows = new List<object>();
            foreach (var (name, logits) in cases)
            {
                double dot = 0, na = 0, nb = 0, max = 0;
                for (int i = 0; i < logits.Length; i++) { dot += (double)logits[i] * reference[i]; na += (double)logits[i] * logits[i]; nb += (double)reference[i] * reference[i]; max = Math.Max(max, Math.Abs(logits[i] - reference[i])); }
                double cos = dot / Math.Sqrt(na * nb);
                rows.Add(new { name, cosine = cos, max_abs = max });
                Console.WriteLine($"[ab] batchdiag {name}: cosine {cos:F6}, max |d| {max:F4} (prompts {pa.Length}/{pb.Length})");
            }
            report["cases"] = rows;
        }
        finally
        {
            model.DisposeSeqState(a); model.DisposeSeqState(a2); model.DisposeSeqState(b);
            model.SelfConditioningEnabled = sc;
        }
        break;
    }
    case "structured":
    {
        var cases = new List<object>();
        foreach (string text in Prompts())
        {
            int[] prompt = Render(text);
            int[] canvas = Enumerable.Repeat(model.MaskTokenId, Math.Min(32, C)).ToArray();
            int[] positions = { 0, 3, 7 };
            int[][] labels = { model.Tokenizer.Encode("Yes", addSpecial: false).Concat(model.Tokenizer.Encode("No", addSpecial: false)).Distinct().ToArray(),
                Enumerable.Range(1000, 8).ToArray(), Enumerable.Range(5000, 4).ToArray() };
            float[][] fast = model.ReadStructured(prompt, canvas, positions, labels);
            float[][] reference = model.ReadStructuredReference(prompt, canvas, positions, labels);
            double max = 0;
            for (int r = 0; r < fast.Length; r++)
                for (int j = 0; j < fast[r].Length; j++) max = Math.Max(max, Math.Abs(fast[r][j] - reference[r][j]));
            cases.Add(new { prompt = text, fast, reference, max_abs = max });
            Console.WriteLine($"[ab] structured: max |fast - reference| {max:E2}");
        }
        report["cases"] = cases;
        break;
    }
    default:
        throw new ArgumentException($"Unknown scenario '{scenario}'.");
}

Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(output))!);
File.WriteAllText(output, JsonSerializer.Serialize(report, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"[ab] wrote {output}");

static double RelL2(float[] a, float[] b)
{
    double num = 0, den = 0;
    for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
    return Math.Sqrt(num / Math.Max(den, 1e-30));
}

static void Compare(string? a, string? b)
{
    if (a == null || b == null) throw new ArgumentException("Pass --a and --b.");
    if (Directory.Exists(a))
    {
        int vocab = 262144;
        foreach (string fa in Directory.GetFiles(a, "*.f32").Order())
        {
            string fb = Path.Combine(b, Path.GetFileName(fa));
            if (!File.Exists(fb)) { Console.WriteLine($"{Path.GetFileName(fa)}: missing in {b}"); continue; }
            float[] x = Read(fa), y = Read(fb);
            int rows = x.Length / vocab, agree = 0;
            double maxLp = 0, sumKl = 0;
            for (int r = 0; r < rows; r++)
            {
                var xr = x.AsSpan(r * vocab, vocab);
                var yr = y.AsSpan(r * vocab, vocab);
                int ax = 0, ay = 0;
                for (int i = 1; i < vocab; i++) { if (xr[i] > xr[ax]) ax = i; if (yr[i] > yr[ay]) ay = i; }
                if (ax == ay) agree++;
                double lx = Lse(xr), ly = Lse(yr), kl = 0;
                for (int i = 0; i < vocab; i++)
                {
                    double px = Math.Exp(xr[i] - lx);
                    if (px > 1e-12) kl += px * ((xr[i] - lx) - (yr[i] - ly));
                }
                sumKl += kl;
                maxLp = Math.Max(maxLp, Math.Abs((xr[ax] - lx) - (yr[ax] - ly)));
            }
            Console.WriteLine($"{Path.GetFileName(fa)}: rel L2 {RelL2(y, x):E3}, argmax {agree}/{rows}, mean KL(a||b) {sumKl / rows:E3}, max |dlogp(top1 a)| {maxLp:F4}");
        }
        return;
    }
    Console.WriteLine("compare takes two --dump directories.");

    static float[] Read(string path)
    {
        byte[] bytes = File.ReadAllBytes(path);
        var f = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
        return f;
    }

    static double Lse(ReadOnlySpan<float> v)
    {
        double m = double.NegativeInfinity;
        foreach (float x in v) m = Math.Max(m, x);
        double s = 0;
        foreach (float x in v) s += Math.Exp(x - m);
        return m + Math.Log(s);
    }
}
