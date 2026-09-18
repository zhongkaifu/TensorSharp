// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// AgentTurnBench: does a multi-token input reach the model as ONE batched forward
// (per chunk), or as one forward per token? It drives the continuous-batching
// engine (InferenceEngine -> ContinuousBatchScheduler -> BatchExecutor), which is
// the path the server and TensorAgent use, with the conversation shapes an agent
// actually produces, and reports per request: prompt tokens, tokens served from
// the cache, engine steps, tokens per prefill step, TTFT, prefill and decode
// rates, and - under speculation - drafted / accepted / verify / plain steps.
//
//   short    a tiny prompt (no agent system prompt)
//   long     an agent system prompt plus a long pasted file
//   tool     a turn, then the SAME conversation extended by a big tool result
//            (read_file), then a short follow-up: the shape of every tool round
//   newchat  two conversations sharing the system prompt (shared-prefix checkpoint)
//   spec     greedy generation plain vs speculative (ngram, and the checkpoint's own
//            drafter when it has one); token differences are recorded for diagnosis
//   json     grammar-constrained (JSON object) generation, plain vs speculative
//   conc     N concurrent requests on one engine, then a solo request after them
//   image    a turn carrying an image (--mmproj + --image): the plain prefill must
//            inject the embeddings, and a hidden-free drafter arms after it
//
// Usage:
//   dotnet run -c Release --project benchmarks/AgentTurnBench -- --model <gguf>
//       [--backend ggml_metal|ggml_cuda|ggml_cpu] [--draft-model <gguf>] [--mmproj <gguf> --image <file>]
//       [--kv f16|q8_0|q4_0]
//       [--chunk 1024] [--max-batched 4096] [--long 4096] [--tool 3000] [--new 32]
//       [--spec-new 192] [--spec-file 600] [--spec-minimal-system] [--spec-engine ngram|auto] [--conc 2,4] [--conc-stagger 400] [--conc-gate] [--scenarios short,long,tool,newchat,spec,json,conc]
//       [--warmup 0] [--measure-passes 1] [--out rows.json] [--verbose]
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace AgentTurnBench;

internal static class Program
{
    public static async Task<int> Main(string[] args)
    {
        Options o = Options.Parse(args);
        if (o == null) return 2;

        BackendType backend = ParseBackend(o.Backend);
        // Match server startup so TS_N_CPU_MOE / TS_CPU_MOE and the host
        // thread limit actually apply to offload benchmark profiles.
        MoeCpuOffloadConfig.ConfigureFromEnvironment();
        if (o.Kv != null)
        {
            if (!KvCacheDtypeConfig.TryParse(o.Kv, out KvCacheDtype dt))
            {
                Console.Error.WriteLine($"unknown --kv '{o.Kv}' (f32, f16, q8_0, q4_0)");
                return 2;
            }
            KvCacheDtypeConfig.Set(dt);
        }

        Console.WriteLine($"[agent-turn-bench] loading {Path.GetFileName(o.Model)} backend={backend} " +
                          $"chunk={o.Chunk} maxBatched={o.MaxBatched} kv={o.Kv ?? "default"}");
        var swLoad = Stopwatch.StartNew();
        // Before the load: a block drafter (DFlash / DSpark) and Nemotron's MTP head
        // join the model at construction; only Gemma 4's assistant head attaches after.
        if (o.DraftModel != null)
            Environment.SetEnvironmentVariable(SpeculationEnvVars.DraftModel, o.DraftModel);
        using ModelBase model = ModelBase.Create(o.Model, backend, draftModelPath: o.DraftModel);
        if (o.DraftModel != null && !SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(model, out string err))
            Console.Error.WriteLine($"[agent-turn-bench] draft head NOT attached: {err}");
        if (o.MmProj != null)
            model.MultimodalInjector.LoadProjectors(o.MmProj);
        model.WarmUpKernels();
        // These managed cache implementations allocate K/V using KvCacheDtype.
        // Opaque native executors need their own effective-storage diagnostics.
        if (model is Gemma4Model or Qwen35Model or GptOssModel)
            Console.WriteLine($"[agent-turn-bench] effective managed KV storage dtype={model.KvCacheDtype.ToShortString()} model_type={model.GetType().Name}");
        Console.WriteLine($"[agent-turn-bench] loaded {model.Config.Architecture} in {swLoad.Elapsed.TotalSeconds:0.0}s; " +
                          $"context={model.MaxContextLength} drafter={DescribeDrafter(model)}");
        if (o.SpecDiagnostic) return SpecParityDiagnostic.Run(model, o);

        // Kernel warm-up does not exercise the scheduler, managed decoding, or
        // background tiered JIT. Optional full passes measure a warmed process
        // without changing the application's runtime compilation settings.
        for (int pass = 0; pass < o.Warmup; pass++)
        {
            Console.WriteLine($"[agent-turn-bench] warm-up pass {pass + 1}/{o.Warmup}");
            var warmup = new Bench(model, o);
            await warmup.RunAsync();
            if (o.Out != null) warmup.WriteJson(o.Out + $".warmup{pass + 1}.json");
            if (warmup.Failures != 0)
            {
                Console.Error.WriteLine($"[agent-turn-bench] FAIL: warm-up had {warmup.Failures} check(s) failed.");
                return 1;
            }
        }

        var samples = new List<object>();
        using var process = Process.GetCurrentProcess();
        for (int pass = 0; pass < o.MeasurePasses; pass++)
        {
            if (o.MeasurePasses > 1)
                Console.WriteLine($"[agent-turn-bench] measured pass {pass + 1}/{o.MeasurePasses}");
            var started = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            var cpuBefore = process.TotalProcessorTime;
            long allocatedBefore = GC.GetTotalAllocatedBytes(precise: false);
            int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
            var elapsed = Stopwatch.StartNew();
            var bench = new Bench(model, o);
            await bench.RunAsync();
            elapsed.Stop();
            samples.Add(new
            {
                Pass = pass + 1,
                StartedUnixMilliseconds = started,
                FinishedUnixMilliseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
                ElapsedMs = elapsed.Elapsed.TotalMilliseconds,
                CpuMs = (process.TotalProcessorTime - cpuBefore).TotalMilliseconds,
                AllocatedBytes = GC.GetTotalAllocatedBytes(precise: false) - allocatedBefore,
                Gen0Collections = GC.CollectionCount(0) - gc0,
                Gen1Collections = GC.CollectionCount(1) - gc1,
                Gen2Collections = GC.CollectionCount(2) - gc2,
                bench.Failures,
            });
            bench.PrintTable();
            if (o.Out != null)
            {
                bench.WriteJson(o.MeasurePasses == 1 ? o.Out : o.Out + $".measure{pass + 1}.json");
                if (o.MeasurePasses > 1)
                    File.WriteAllText(o.Out + ".series.json", JsonSerializer.Serialize(samples,
                        new JsonSerializerOptions { WriteIndented = true }));
            }
            Console.WriteLine(bench.Failures == 0
                ? "[agent-turn-bench] PASS: every multi-token input was forwarded in batched chunks."
                : $"[agent-turn-bench] FAIL: {bench.Failures} check(s) failed (see notes).");
            if (bench.Failures != 0) return 1;
        }
        return 0;
    }

    private static BackendType ParseBackend(string s) => (s ?? "ggml_metal").ToLowerInvariant() switch
    {
        "ggml_metal" => BackendType.GgmlMetal,
        "ggml_cuda" => BackendType.GgmlCuda,
        "ggml_vulkan" => BackendType.GgmlVulkan,
        "ggml_cpu" => BackendType.GgmlCpu,
        "cpu" => BackendType.Cpu,
        "cuda" => BackendType.Cuda,
        "mlx" => BackendType.Mlx,
        _ => throw new ArgumentException($"unknown backend '{s}'"),
    };

    private static string DescribeDrafter(ModelBase model)
    {
        if (model.HasDFlash) return "dflash";
        if (model is IDraftHead h && h.HasDraftHead) return h.DraftHeadKind.ToString().ToLowerInvariant();
        return "none";
    }
}

internal sealed class Options
{
    public string Model;
    public string Backend = "ggml_metal";
    public string DraftModel;
    public string MmProj;
    public string Image;
    public string Kv;
    public int Chunk = 1024;
    public int MaxBatched = 4096;
    public int Long = 4096;
    public int Tool = 3000;
    public int New = 32;
    public int SpecNew = 192;
    public int SpecFile = 600;
    public bool SpecMinimalSystem;
    public bool SpecDiagnostic;
    /// <summary>--spec-diagnostic drafter: "ngram" (default) or "auto" (the checkpoint's own head).</summary>
    public string SpecDiagSpeculator = "ngram";
    /// <summary>--spec-diagnostic user message replacing the file-repeat prompt (no system prompt).</summary>
    public string SpecDiagPrompt;
    /// <summary>--spec-diagnostic keeps following plain greedy past a mismatch and records every row.</summary>
    public bool SpecDiagTeacherForce;
    /// <summary>--spec-diagnostic draft window.</summary>
    public int SpecDiagWindow = 7;
    /// <summary>--spec-diagnostic draws both runs through the JSON-object grammar (json scenario prompt).</summary>
    public bool SpecDiagJson;
    /// <summary>--spec-diagnostic uses the newchat scenario's chat B prompt.</summary>
    public bool SpecDiagNewChat;
    /// <summary>--spec-diagnostic compares one next-token row across every trunk path, then exits.</summary>
    public bool SpecDiagRowCheck;

    /// <summary>Enable speculation on EVERY engine the bench builds ("ngram" or "auto"),
        /// so the concurrent and solo-after-concurrency rows run with it - the way a
        /// server with the setting on would.</summary>
        public string SpecEngine;
        /// <summary>Milliseconds between the submissions of a concurrent round, so
        /// later requests arrive while earlier ones already decode (and speculate).</summary>
        public int ConcStaggerMs;
        /// <summary>Submit a whole concurrent round while the engine's compute gate is
        /// closed and open it afterwards, so the first scheduler step sees every request
        /// (a fixed arrival order instead of racing the engine thread).</summary>
        public bool ConcGate;
    public List<int> Conc = new() { 2, 4 };
    public List<string> Scenarios = new() { "short", "long", "tool", "newchat", "spec", "json", "conc" };
    public string Out;
    public bool Verbose;
    public int Warmup;
    public int MeasurePasses = 1;

    public static Options Parse(string[] args)
    {
        var o = new Options();
        try
        {
            for (int i = 0; i < args.Length; i++)
            {
                string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
                switch (args[i])
                {
                    case "--model": o.Model = Next(); break;
                    case "--backend": o.Backend = Next(); break;
                    case "--draft-model": o.DraftModel = Next(); break;
                    case "--mmproj": o.MmProj = Next(); break;
                    case "--image": o.Image = Next(); break;
                    case "--kv": o.Kv = Next(); break;
                    case "--chunk": o.Chunk = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--max-batched": o.MaxBatched = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--long": o.Long = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--tool": o.Tool = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--new": o.New = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--spec-new": o.SpecNew = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--spec-file": o.SpecFile = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--spec-minimal-system": o.SpecMinimalSystem = true; break;
                    case "--spec-diagnostic": o.SpecDiagnostic = true; break;
                    case "--spec-diagnostic-speculator": o.SpecDiagSpeculator = Next(); break;
                    case "--spec-diagnostic-prompt": o.SpecDiagPrompt = Next(); break;
                    case "--spec-diagnostic-teacher-force": o.SpecDiagTeacherForce = true; break;
                    case "--spec-diagnostic-json": o.SpecDiagJson = true; break;
                    case "--spec-diagnostic-newchat": o.SpecDiagNewChat = true; break;
                    case "--spec-diagnostic-rowcheck": o.SpecDiagRowCheck = true; break;
                    case "--spec-diagnostic-window": o.SpecDiagWindow = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--spec-engine": o.SpecEngine = Next(); break;
                    case "--conc-stagger": o.ConcStaggerMs = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--conc-gate": o.ConcGate = true; break;
                    case "--conc":
                        o.Conc = Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
                            .Select(s => int.Parse(s, CultureInfo.InvariantCulture)).ToList();
                        break;
                    case "--scenarios":
                        o.Scenarios = Next().Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
                        break;
                    case "--out": o.Out = Next(); break;
                    case "--warmup": o.Warmup = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--measure-passes": o.MeasurePasses = int.Parse(Next(), CultureInfo.InvariantCulture); break;
                    case "--verbose": o.Verbose = true; break;
                    default: throw new ArgumentException($"unknown option {args[i]}");
                }
            }
            if (string.IsNullOrEmpty(o.Model)) throw new ArgumentException("--model <gguf> is required");
            if (!File.Exists(o.Model)) throw new ArgumentException($"model not found: {o.Model}");
            if (o.Warmup < 0) throw new ArgumentException("--warmup must be nonnegative");
            if (o.MeasurePasses < 1) throw new ArgumentException("--measure-passes must be positive");
            if (o.ConcGate && o.ConcStaggerMs > 0)
                throw new ArgumentException("--conc-gate submits a round at once and cannot be combined with --conc-stagger");
        }
        catch (ArgumentException ex)
        {
            Console.Error.WriteLine(ex.Message);
            return null;
        }
        return o;
    }
}

/// <summary>One request's measurements.</summary>
internal sealed record Row(
    string Scenario, string Label, int Prompt, int Reused, long Steps, int PrefillSteps, double TokensPerPrefillStep,
    double TtftMs, double PrefillTps, double DecodeTps, int OutTokens, double TotalMs, string Finish,
    long Drafted, long Accepted, long VerifySteps, long PlainSteps, long Rollbacks, string Note)
{
    public int Fresh => Prompt - Reused;
    public List<int> Tokens { get; init; } = new();
    // Actual delivery times from submission, for individual requests only.
    public List<double> TokenTimesMs { get; init; }
    public long StartedUnixMilliseconds { get; init; }
    public List<RequestTimeline> RequestTimelines { get; init; }
    public ConcurrentDecodeMetrics ConcurrentDecode { get; init; }
    /// <summary>Concurrent rows only: true when --conc-gate held the engine until the
    /// whole round was queued, so every run schedules the same batches. Null elsewhere.</summary>
    public bool? ArrivalOrderFixed { get; init; }
    // Boundaries in the flattened token stream for concurrent requests.
    public int[] TokenCounts { get; init; } = Array.Empty<int>();
    public List<string> ExtraNotes { get; } = new();
    public string AllNotes => string.Join(" | ", new[] { Note }.Concat(ExtraNotes).Where(n => !string.IsNullOrEmpty(n)));
}

internal sealed class Bench
{
    private readonly ModelBase _model;
    private readonly Options _o;
    private readonly List<Row> _rows = new();
    private readonly KVCachePromptRenderer _renderer = new(new GgufPromptRenderer());
    private readonly string _arch;
    private readonly string _agentSystem;
    private readonly List<string> _engineLog = new();

    public int Failures { get; private set; }

    public Bench(ModelBase model, Options o)
    {
        _model = model;
        _o = o;
        _arch = model.Config.Architecture;
        _agentSystem = Corpus.AgentSystemPrompt();
    }

    // ---------------------------------------------------------------- scenarios

    public async Task RunAsync()
    {
        foreach (string s in _o.Scenarios)
        {
            Console.WriteLine();
            Console.WriteLine($"==== {s} ====");
            try
            {
                switch (s)
                {
                    case "short": await ShortAsync(); break;
                    case "long": await LongAsync(); break;
                    case "tool": await ToolAsync(null, "tool"); break;
                    case "newchat": await NewChatAsync(); break;
                    case "spec": await SpecAsync(); break;
                    case "json": await JsonAsync(); break;
                    case "conc": await ConcurrentAsync(); break;
                    case "image": await ImageAsync(); break;
                    default: Console.Error.WriteLine($"unknown scenario '{s}'"); break;
                }
            }
            catch (Exception ex)
            {
                Failures++;
                Console.Error.WriteLine($"[agent-turn-bench] scenario {s} threw: {ex}");
            }
        }
    }

    private async Task ShortAsync()
    {
        using var engine = NewEngine();
        List<int> prompt = Render(Corpus.MinimalSystemPrompt, "Say the single word: apple.");
        await RunAsync(engine, "short", "turn 1", prompt, _o.New, SamplingConfig.Greedy, expectBatched: true);
    }

    private async Task LongAsync()
    {
        using var engine = NewEngine();
        // The system prompt and the template cost ~700 tokens; the pasted file fills the rest.
        string file = Corpus.CodeText(_model.Tokenizer, Math.Max(256, _o.Long - 700));
        List<int> prompt = Render(_agentSystem, $"Here is src/Program.cs:\n```csharp\n{file}```\nSummarize it in one sentence.");
        await RunAsync(engine, "long", "turn 1", prompt, _o.New, SamplingConfig.Greedy, expectBatched: true);
    }

    /// <summary>The tool-round shape: a turn, the same conversation extended by a big tool
    /// result, then a short follow-up. Returns the rows so the speculative scenario can
    /// compare its own run of the same shape.</summary>
    private async Task<List<Row>> ToolAsync(SpeculationOptions spec, string scenario)
    {
        using var engine = NewEngine(spec);
        var rows = new List<Row>();
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = _agentSystem },
            new() { Role = "user", Content = "Read the file src/Program.cs with read_file and summarize it." },
        };
        List<int> turn1 = RenderHistory(history, out string boundary1);
        Row r1 = await RunAsync(engine, scenario, "turn 1", turn1, 24, SamplingConfig.Greedy, expectBatched: true);
        rows.Add(r1);

        // What the chat layer sends after the host ran the tool: the whole conversation
        // re-rendered, with the model's RAW output tokens spliced in for its last turn
        // (what the server records per assistant round) and the tool result as the next
        // message. The renderer then reproduces the previous prompt byte for byte, the
        // end-of-turn token included, and only the tool result is new to the cache.
        history.Add(AssistantTurn(r1, boundary1));
        string result = Corpus.CodeText(_model.Tokenizer, _o.Tool);
        history.Add(new ChatMessage
        {
            Role = "user",
            Content = $"[tool result: read_file src/Program.cs]\n{result}[end of tool result]\nNow summarize the file in one sentence.",
        });
        List<int> turn2 = RenderHistory(history, out string boundary2);
        Row r2 = await RunAsync(engine, scenario, $"turn 2 (+{_o.Tool}-token tool result)", turn2, 24, SamplingConfig.Greedy, expectBatched: true);
        rows.Add(r2);

        history.Add(AssistantTurn(r2, boundary2));
        history.Add(new ChatMessage { Role = "user", Content = "Thanks. Reply with the single word: done." });
        List<int> turn3 = RenderHistory(history, out _);
        Row r3 = await RunAsync(engine, scenario, "turn 3 (follow-up)", turn3, 16, SamplingConfig.Greedy, expectBatched: true);
        rows.Add(r3);
        return rows;
    }

    private async Task NewChatAsync()
    {
        List<int> systemOnly = _renderer.RenderToTokens(_model.Tokenizer, _model.Config.ChatTemplate,
            new List<ChatMessage> { new() { Role = "system", Content = _agentSystem } }, _arch, addGenerationPrompt: false);
        string file = Corpus.CodeText(_model.Tokenizer, 200);
        List<int> chatA = Render(_agentSystem, "Say the single word: apple.");
        int shared = Lcp(systemOnly, chatA);
        Console.WriteLine($"    shared system-prompt prefix: {shared} tokens");
        // Chat B starts from a CLONE of the checkpoint - a per-request fused holder,
        // the shape of every new chat in the app - so it is also where speculation
        // has to arm over a holder rather than the linear cache.
        List<int> chatB = Render(_agentSystem, $"Repeat this text exactly:\n```csharp\n{file}```");
        Row plainB = null;
        foreach (var (label, spec) in new[] { ("", SpeculationOptions.Disabled), (" + ngram", SpecOptions(SpeculatorRegistry.NGram)) })
        {
            using var engine = NewEngine(spec);
            await RunAsync(engine, "newchat", "chat A turn 1" + label, chatA, 16, SamplingConfig.Greedy, expectBatched: true, sharedPrefix: shared);
            Row b = await RunAsync(engine, "newchat", "chat B turn 1 (new conversation)" + label, chatB, 96, SamplingConfig.Greedy, expectBatched: true, sharedPrefix: shared);
            if (plainB == null)
                plainB = b;
            else
            {
                CompareStreams(plainB, b, "chat B" + label);
                if (b.VerifySteps == 0)
                    Note(b, "speculation never engaged on the checkpoint clone (see engine log)");
            }
        }
    }

    private async Task SpecAsync()
    {
        string file = Corpus.CodeText(_model.Tokenizer, _o.SpecFile);
        // --spec-minimal-system keeps the whole prompt under a 512-token sliding
        // window, so a divergence that appears only with the agent prompt (which
        // alone wraps the window) can be attributed to the wrapped cache.
        List<int> prompt = Render(_o.SpecMinimalSystem ? Corpus.MinimalSystemPrompt : _agentSystem,
            $"Here is src/Program.cs:\n```csharp\n{file}```\nRepeat the file exactly as given, then add one sentence describing what it does.");

        Row plain;
        using (var engine = NewEngine(SpeculationOptions.Disabled))
            plain = await RunAsync(engine, "spec", "plain greedy", prompt, _o.SpecNew, SamplingConfig.Greedy, expectBatched: true);

        var candidates = new List<(string label, SpeculationOptions opts)>
        {
            ("ngram", SpecOptions(SpeculatorRegistry.NGram)),
        };
        if (_model.HasDFlash || (_model is IDraftHead h && h.HasDraftHead))
            candidates.Add(("draft head (auto)", SpecOptions(SpeculatorRegistry.Auto)));

        foreach (var (label, opts) in candidates)
        {
            using var engine = NewEngine(opts);
            Row spec = await RunAsync(engine, "spec", label, prompt, _o.SpecNew, SamplingConfig.Greedy, expectBatched: true);
            CompareStreams(plain, spec, label);
            if (spec.VerifySteps == 0 && spec.Drafted == 0)
                Note(spec, "speculation never engaged (see engine log)");
        }

        // The tool-round shape under n-gram speculation: the drafter must re-arm after the
        // reused prefix (every turn after the first reuses cache) and the verify windows
        // must go through as batches.
        Console.WriteLine("    -- tool rounds under ngram speculation --");
        await ToolAsync(SpecOptions(SpeculatorRegistry.NGram), "spec+tool");
    }

    private async Task JsonAsync()
    {
        string file = Corpus.CodeText(_model.Tokenizer, 300);
        List<int> prompt = Render(_agentSystem,
            "Return a JSON object with the keys \"name\" (string), \"lines\" (integer) and \"summary\" (string) " +
            $"describing this file. Output only the JSON object.\n```csharp\n{file}```");
        GrammarMaskCache cache = GrammarLibrary.ForJsonObject(_model.Tokenizer);

        SamplingConfig Cfg()
        {
            SamplingConfig c = SamplingConfig.Greedy.Clone();
            c.Grammar = GrammarLibrary.NewConstraint(cache, _model.Tokenizer);
            return c;
        }

        Row plain;
        using (var engine = NewEngine(SpeculationOptions.Disabled))
            plain = await RunAsync(engine, "json", "plain + json grammar", prompt, 192, Cfg(), expectBatched: true);
        CheckJson(plain);

        using (var engine = NewEngine(SpecOptions(SpeculatorRegistry.NGram)))
        {
            Row spec = await RunAsync(engine, "json", "ngram + json grammar", prompt, 192, Cfg(), expectBatched: true);
            CheckJson(spec);
            CompareStreams(plain, spec, "ngram + json grammar");
        }
    }

    private async Task ConcurrentAsync()
    {
        using var engine = NewEngine();
        foreach (int n in _o.Conc)
        {
            var prompts = new List<List<int>>(n);
            for (int i = 0; i < n; i++)
            {
                string file = Corpus.CodeText(_model.Tokenizer, 400, salt: i * 17 + 3);
                prompts.Add(Render(_agentSystem, $"Here is file {i}:\n```csharp\n{file}```\nName its main class."));
            }
            long steps0 = engine.TotalStepsRun;
            long startedUnixMilliseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
            long waveStart = Stopwatch.GetTimestamp();
            var tasks = new List<Task<Run>>(n);
            // SubmitAsync enqueues synchronously, before its first await; with the gate
            // closed the engine thread cannot schedule a step until every request is queued.
            ComputeGate gate = _o.ConcGate ? new ComputeGate() : null;
            gate?.Close();
            engine.ComputeGate = gate;
            for (int i = 0; i < n; i++)
            {
                if (i > 0 && _o.ConcStaggerMs > 0)
                    await Task.Delay(_o.ConcStaggerMs);
                tasks.Add(SubmitAsync(engine, $"conc{n}-{i}", prompts[i], _o.New, SamplingConfig.Greedy, 0, waveStart));
            }
            gate?.Open();
            Run[] runs = await Task.WhenAll(tasks);
            engine.ComputeGate = null;
            double waveMs = Stopwatch.GetElapsedTime(waveStart).TotalMilliseconds;
            long steps = engine.TotalStepsRun - steps0;
            int outTokens = runs.Sum(r => r.Tokens.Count);
            int promptTokens = prompts.Sum(p => p.Count);
            // A request's TTFT starts at its own submission. Staggered arrivals
            // require an absolute wave offset to find the last first delivery.
            var decode = ConcurrentDecodeMetrics.Calculate(runs.Select(r =>
                new ConcurrentDelivery(r.SubmissionOffsetMs, r.TokenTimesMs)).ToArray(), waveMs);
            double maxTtft = runs.Max(r => r.TtftMs);
            double aggregate = decode.TokensPerSecond;
            var timelines = runs.Select(ToTimeline).ToList();
            var speculation = RequestSpeculationCounters.Sum(timelines.Select(r => r.Speculation));
            string note = $"decode aggregate {aggregate:0.0} tok/s from {decode.TokensAfterLastFirst} deliveries strictly after the last first token; " +
                          $"window {decode.WindowMs:0} ms; last first-token wave offset {decode.LastFirstTokenOffsetMs:0} ms; " +
                          $"wall {waveMs:0} ms; max individual request ttft {maxTtft:0} ms; established={decode.Established}";
            var row = new Row("conc", $"{n} concurrent", promptTokens, runs.Sum(r => r.Reused), steps, 0, 0,
                runs.Max(r => r.TtftMs), 0, aggregate, outTokens, waveMs,
                string.Join("/", runs.Select(r => r.Finish)),
                speculation.Drafted, speculation.Accepted, speculation.VerifySteps, speculation.PlainSteps,
                speculation.Rollbacks, note)
            {
                Tokens = runs.SelectMany(r => r.Tokens).ToList(),
                TokenCounts = runs.Select(r => r.Tokens.Count).ToArray(),
                StartedUnixMilliseconds = startedUnixMilliseconds,
                RequestTimelines = timelines,
                ConcurrentDecode = decode,
                ArrivalOrderFixed = _o.ConcGate,
            };
            Add(row);
            foreach (Run r in runs)
                if (RequestCompletionChecks.ConcurrentFailed(r.Tokens.Count, r.Finish, r.Error))
                { Failures++; Console.Error.WriteLine($"    FAIL: {r.Id} produced {r.Tokens.Count} tokens, finish={r.Finish}, error={r.Error}"); }
        }
        // A solo request after the concurrent round must still run batched (and fast).
        List<int> solo = Render(_agentSystem, "After all that, say the single word: solo.");
        await RunAsync(engine, "conc", "solo after concurrency", solo, _o.New, SamplingConfig.Greedy, expectBatched: true);
    }

    /// <summary>Speculation settings for one algorithm; the window and gate come
    /// from the operator's usual TS_SPEC_DRAFT / TS_SPEC_PMIN when set.</summary>
    private static SpeculationOptions SpecOptions(string algorithm)
    {
        SpeculationOptions env = SpeculationOptions.FromEnvironment();
        return new SpeculationOptions
        {
            Enabled = true,
            SpeculatorName = algorithm,
            MaxDraftTokens = env.MaxDraftTokens,
            MaxDraftTokensExplicit = env.MaxDraftTokensExplicit,
            MinDraftProb = env.MinDraftProb,
        };
    }

    private async Task ImageAsync()
    {
        if (_o.Image == null || _o.MmProj == null)
        {
            Console.WriteLine("    skipped: needs --image <file> and --mmproj <gguf>");
            return;
        }
        string file = Corpus.CodeText(_model.Tokenizer, 200);
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = Corpus.MinimalSystemPrompt },
            new()
            {
                Role = "user",
                Content = $"Describe this image in one sentence. Then repeat the following text exactly:\n```csharp\n{file}```",
                ImagePaths = new List<string> { _o.Image },
            },
        };
        Row plain = null;
        foreach (var (label, spec) in new[] { ("plain + image", SpeculationOptions.Disabled), ("ngram + image", SpecOptions(SpeculatorRegistry.NGram)) })
        {
            using var engine = NewEngine(spec);
            // The injector keeps the prepared embeddings per request id, and the
            // engine asks for them under the sequence's id: prepare under the id
            // the request will carry.
            string id = $"image-{_rows.Count}";
            List<int> tokens = RenderHistory(history, out _);
            tokens = _model.MultimodalInjector.ProcessPromptTokens(history, tokens, id);
            Row row = await RunAsync(engine, "image", label, tokens, 160, SamplingConfig.Greedy, expectBatched: true, requestId: id);
            if (plain == null)
                plain = row;
            else
            {
                CompareStreams(plain, row, label);
                if (row.VerifySteps == 0)
                    Note(row, "speculation never engaged after the media prefill (see engine log)");
            }
        }
    }

    // ---------------------------------------------------------------- engine driving

    private InferenceEngine NewEngine(SpeculationOptions spec = null)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = _o.MaxBatched,
            MaxNumRunningSequences = 16,
            MaxPrefillChunkSize = 256,
            SoloPrefillChunkSize = _o.Chunk,
            NumBlocks = EnvInt("TS_SCHED_NUM_BLOCKS", 256),
            BlockSize = EnvInt("TS_SCHED_BLOCK_SIZE", 256),
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 256,
            Speculation = spec ?? (_o.SpecEngine != null ? SpecOptions(_o.SpecEngine == "auto" ? SpeculatorRegistry.Auto : _o.SpecEngine) : SpeculationOptions.Disabled),
        };
        return new InferenceEngine(_model, cfg, new EngineLogger(_engineLog, _o.Verbose));
    }

    private sealed class Run
    {
        public string Id;
        public List<int> Tokens = new();
        public List<double> TokenTimesMs = new();
        public long StartedUnixMilliseconds;
        public double SubmissionOffsetMs;
        public double TtftMs;
        public double TotalMs;
        public int Reused;
        public string Finish;
        public string Error;
        public SpeculationStats Stats;
    }

    private async Task<Run> SubmitAsync(InferenceEngine engine, string id, List<int> prompt, int maxNew,
        SamplingConfig cfg, int sharedPrefix, long waveStart = 0)
    {
        var seq = new SequenceState(id, prompt, maxNew, engine.PoolStats.blockSize, cfg, sharedPrefixTokens: sharedPrefix);
        long submitted = Stopwatch.GetTimestamp();
        var run = new Run
        {
            Id = id, StartedUnixMilliseconds = DateTimeOffset.UtcNow.ToUnixTimeMilliseconds(),
            SubmissionOffsetMs = waveStart == 0 ? 0 : Stopwatch.GetElapsedTime(waveStart, submitted).TotalMilliseconds,
        };
        try
        {
            InferenceRequestHandle handle = engine.SubmitRequest(seq);
            await foreach (int t in handle.Tokens.ReadAllAsync())
            {
                double deliveredMs = Stopwatch.GetElapsedTime(submitted).TotalMilliseconds;
                if (run.Tokens.Count == 0) run.TtftMs = deliveredMs;
                run.Tokens.Add(t);
                run.TokenTimesMs.Add(deliveredMs);
            }
            InferenceCompletion done = await handle.Completion;
            run.Reused = done.PrefixCacheReusedTokens;
            run.Finish = done.FinishReason;
        }
        catch (Exception ex)
        {
            run.Error = ex.Message;
            run.Finish = "error";
        }
        run.TotalMs = Stopwatch.GetElapsedTime(submitted).TotalMilliseconds;
        run.Stats = seq.SpecStats;
        return run;
    }

    private static RequestTimeline ToTimeline(Run run) => new(run.Id, run.StartedUnixMilliseconds,
        run.SubmissionOffsetMs, run.Tokens.Count, run.TokenTimesMs, run.Finish, run.Error,
        new RequestSpeculationCounters(run.Stats?.TokensDrafted ?? 0, run.Stats?.TokensAccepted ?? 0,
            run.Stats?.VerifySteps ?? 0, run.Stats?.PlainSteps ?? 0, run.Stats?.RollbackSteps ?? 0,
            run.Stats?.ParkedSteps ?? 0, run.Stats?.GovernorWins ?? 0, run.Stats?.GovernorLosses ?? 0,
            run.Stats?.GovernorParkedSteps ?? 0));

    private async Task<Row> RunAsync(InferenceEngine engine, string scenario, string label, List<int> prompt, int maxNew,
        SamplingConfig cfg, bool expectBatched, int sharedPrefix = 0, string requestId = null)
    {
        _engineLog.Clear();
        long steps0 = engine.TotalStepsRun;
        Run run = await SubmitAsync(engine, requestId ?? $"{scenario}-{_rows.Count}", prompt, maxNew, cfg, sharedPrefix);
        long steps = engine.TotalStepsRun - steps0;

        int fresh = prompt.Count - run.Reused;
        // One engine step per emitted token, plus the step that produced the EOS the
        // engine does not publish; under speculation the stats say how many steps the
        // decode took (a context that armed but never engaged reports zero of both).
        long decodeSteps = run.Tokens.Count + (run.Finish == "eos" ? 1 : 0);
        if (run.Stats != null && run.Stats.VerifySteps + run.Stats.PlainSteps > 0)
            decodeSteps = run.Stats.VerifySteps + run.Stats.PlainSteps;
        int prefillSteps = (int)Math.Max(0, steps - decodeSteps);
        double perStep = prefillSteps > 0 ? (double)fresh / prefillSteps : 0;
        double prefillTps = run.TtftMs > 0 ? fresh / (run.TtftMs / 1000.0) : 0;
        double decodeMs = run.TotalMs - run.TtftMs;
        double decodeTps = run.Tokens.Count > 1 && decodeMs > 0 ? (run.Tokens.Count - 1) / (decodeMs / 1000.0) : 0;

        var notes = new List<string>();
        if (RequestCompletionChecks.HasError(run.Finish, run.Error))
        {
            Failures++;
            notes.Add("ERROR " + run.Error);
        }
        foreach (string line in _engineLog) notes.Add(line);

        // The claim under test: fresh prompt tokens go through the model in chunks of
        // up to --chunk tokens, one forward per chunk - never one forward per token.
        // Allow two extra chunk boundaries (a shared-prefix cut, a rewound tail).
        if (expectBatched && run.Error == null && fresh > 1)
        {
            int expectedMax = (fresh + _o.Chunk - 1) / _o.Chunk + 2;
            if (prefillSteps > expectedMax)
            {
                Failures++;
                notes.Add($"FAIL: {fresh} fresh tokens took {prefillSteps} prefill steps (expected <= {expectedMax})");
            }
        }

        var row = new Row(scenario, label, prompt.Count, run.Reused, steps, prefillSteps, perStep, run.TtftMs, prefillTps,
            decodeTps, run.Tokens.Count, run.TotalMs, run.Finish,
            run.Stats?.TokensDrafted ?? 0, run.Stats?.TokensAccepted ?? 0, run.Stats?.VerifySteps ?? 0,
            run.Stats?.PlainSteps ?? 0, run.Stats?.RollbackSteps ?? 0, string.Join(" | ", notes))
        { Tokens = run.Tokens, TokenTimesMs = run.TokenTimesMs, StartedUnixMilliseconds = run.StartedUnixMilliseconds,
          RequestTimelines = new List<RequestTimeline> { ToTimeline(run) } };
        Add(row);
        return row;
    }

    private void Add(Row row)
    {
        _rows.Add(row);
        string spec = row.VerifySteps > 0 || row.Drafted > 0
            ? $" spec: drafted {row.Drafted} accepted {row.Accepted} ({(row.Drafted > 0 ? 100.0 * row.Accepted / row.Drafted : 0):0}%) verify {row.VerifySteps} plain {row.PlainSteps} rollbacks {row.Rollbacks}"
            : string.Empty;
        Console.WriteLine($"    {row.Label}: prompt {row.Prompt} reused {row.Reused} fresh {row.Fresh} | steps {row.Steps} " +
                          $"(prefill {row.PrefillSteps} x {row.TokensPerPrefillStep:0} tok) | ttft {row.TtftMs:0} ms " +
                          $"prefill {row.PrefillTps:0} tok/s decode {row.DecodeTps:0.0} tok/s | {row.OutTokens} tokens ({row.Finish}){spec}");
        if (row.Tokens != null && row.Tokens.Count > 0)
            Console.WriteLine($"      text: {Shorten(_model.Tokenizer.Decode(row.Tokens), 110)}");
        if (!string.IsNullOrEmpty(row.Note))
            Console.WriteLine($"      note: {row.Note}");
    }

    private static void Note(Row row, string note)
    {
        row.ExtraNotes.Add(note);
        Console.WriteLine($"      note: {note}");
    }

    private void CompareStreams(Row plain, Row spec, string label)
    {
        int n = Math.Min(plain.Tokens.Count, spec.Tokens.Count);
        int i = 0;
        while (i < n && plain.Tokens[i] == spec.Tokens[i]) i++;
        if (i == plain.Tokens.Count && i == spec.Tokens.Count)
        {
            Note(spec, $"stream identical to plain greedy ({i} tokens)");
            return;
        }
        // Different batch shapes may change rounding and activation quantization.
        // Preserve the mismatch here; --spec-diagnostic measures the distributions
        // and cache transitions instead of assuming that every mismatch is a near-tie.
        Note(spec, $"stream diverges from plain greedy at token {i} of {plain.Tokens.Count}/{spec.Tokens.Count} " +
                   $"(plain '{Shorten(_model.Tokenizer.Decode(plain.Tokens.Skip(i).Take(8).ToList()), 40)}' vs " +
                   $"{label} '{Shorten(_model.Tokenizer.Decode(spec.Tokens.Skip(i).Take(8).ToList()), 40)}')");
    }

    private void CheckJson(Row row)
    {
        string text = _model.Tokenizer.Decode(row.Tokens.Where(t => !_model.Tokenizer.IsEos(t)).ToList()).Trim();
        if (row.Finish != "eos")
        {
            // Cut off by the token budget: the grammar can only be judged on a
            // document the model was allowed to finish.
            Note(row, $"JSON not judged: generation ended by {row.Finish}, not by the grammar's end");
            return;
        }
        try
        {
            using JsonDocument doc = JsonDocument.Parse(text);
            Note(row, $"valid JSON object with {doc.RootElement.EnumerateObject().Count()} keys");
        }
        catch (JsonException ex)
        {
            Failures++;
            Note(row, $"FAIL: output is not valid JSON ({ex.Message}); text: {Shorten(text, 80)}");
        }
    }

    // ---------------------------------------------------------------- helpers

    /// <summary>The assistant round as the server records it: the text for the
    /// template plus the raw tokens and the boundary the renderer splices back.</summary>
    private ChatMessage AssistantTurn(Row row, string promptTrailingWhitespace) => new()
    {
        Role = "assistant",
        Content = _model.Tokenizer.Decode(row.Tokens),
        RawOutputTokens = new List<int>(row.Tokens),
        RawPromptTrailingWhitespace = promptTrailingWhitespace,
        RawGenerationSuffix = KVCachePromptRenderer.GetAssistantGenerationSuffix(_arch, enableThinking: false),
    };

    private List<int> RenderHistory(List<ChatMessage> history, out string generationPromptTrailingWhitespace)
        => _renderer.RenderToTokens(_model.Tokenizer, _model.Config.ChatTemplate, history, _arch,
            addGenerationPrompt: true, out _, out generationPromptTrailingWhitespace);

    private List<int> Render(string system, string user)
    {
        var msgs = new List<ChatMessage>
        {
            new() { Role = "system", Content = system },
            new() { Role = "user", Content = user },
        };
        return _renderer.RenderToTokens(_model.Tokenizer, _model.Config.ChatTemplate, msgs, _arch, addGenerationPrompt: true);
    }

    private static int Lcp(List<int> a, List<int> b)
    {
        int n = Math.Min(a.Count, b.Count - 1);
        int i = 0;
        while (i < n && a[i] == b[i]) i++;
        return i;
    }

    private static int EnvInt(string name, int fallback)
    {
        string s = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
    }

    internal static string Shorten(string s, int max)
    {
        s = (s ?? string.Empty).Replace("\n", "\\n");
        return s.Length <= max ? s : s.Substring(0, max) + "...";
    }

    public void PrintTable()
    {
        Console.WriteLine();
        Console.WriteLine("| scenario | request | prompt | reused | fresh | steps | prefill steps | tok/prefill step | ttft ms | prefill tok/s | decode tok/s | out | drafted | accepted | verify | plain | note |");
        Console.WriteLine("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |");
        foreach (Row r in _rows)
        {
            Console.WriteLine($"| {r.Scenario} | {r.Label} | {r.Prompt} | {r.Reused} | {r.Fresh} | {r.Steps} | {r.PrefillSteps} | " +
                              $"{r.TokensPerPrefillStep:0} | {r.TtftMs:0} | {r.PrefillTps:0} | {r.DecodeTps:0.0} | {r.OutTokens} | " +
                              $"{r.Drafted} | {r.Accepted} | {r.VerifySteps} | {r.PlainSteps} | {r.AllNotes} |");
        }
    }

    public void WriteJson(string path)
    {
        var opts = new JsonSerializerOptions
        {
            WriteIndented = true,
            DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull,
        };
        File.WriteAllText(path, JsonSerializer.Serialize(_rows.Select(r => new
        {
            r.Scenario, r.Label, r.Prompt, r.Reused, r.Fresh, r.Steps, r.PrefillSteps, r.TokensPerPrefillStep,
            r.TtftMs, r.PrefillTps, r.DecodeTps, r.OutTokens, r.TotalMs, r.Finish,
            r.Drafted, r.Accepted, r.VerifySteps, r.PlainSteps, r.Rollbacks, Note = r.AllNotes,
            r.Tokens, r.TokenCounts, r.TokenTimesMs, r.StartedUnixMilliseconds, r.RequestTimelines, r.ConcurrentDecode,
            r.ArrivalOrderFixed,
        }), opts));
        Console.WriteLine($"[agent-turn-bench] rows written to {path}");
    }
}

/// <summary>The engine's own account of what it did with each request: which path,
/// whether a continuation was declined and why, whether speculation armed. Kept per
/// request so the row can carry it.</summary>
internal sealed class EngineLogger : ILogger
{
    private readonly List<string> _sink;
    private readonly bool _verbose;

    public EngineLogger(List<string> sink, bool verbose)
    {
        _sink = sink;
        _verbose = verbose;
    }

    public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;

    public bool IsEnabled(LogLevel level) => level >= LogLevel.Information;

    public void Log<TState>(LogLevel level, EventId eventId, TState state, Exception exception, Func<TState, Exception, string> formatter)
    {
        string message = formatter(state, exception);
        bool interesting = level >= LogLevel.Warning
            || message.Contains("Speculative", StringComparison.Ordinal)
            || message.Contains("speculation", StringComparison.OrdinalIgnoreCase)
            || message.Contains("continuation", StringComparison.OrdinalIgnoreCase)
            || message.Contains("checkpoint", StringComparison.OrdinalIgnoreCase)
            || message.Contains("declined", StringComparison.OrdinalIgnoreCase)
            || message.Contains("fallback", StringComparison.OrdinalIgnoreCase);
        if (interesting)
            _sink.Add($"[{level}] {Bench.Shorten(message, 220)}");
        if (_verbose || level >= LogLevel.Warning)
            Console.Error.WriteLine($"    [engine {level}] {message}");
    }
}

/// <summary>Deterministic agent-shaped text: a system prompt of the kind TensorAgent
/// sends (tool declarations, skill notes) and a source file a read_file call returns.</summary>
internal static class Corpus
{
    public const string MinimalSystemPrompt = "You are a helpful assistant.";

    public static string AgentSystemPrompt()
    {
        var sb = new StringBuilder();
        sb.AppendLine("You are TensorAgent, a careful software engineering assistant running on the user's device.");
        sb.AppendLine("You can call tools. Each call is a JSON object with the tool name and its arguments. Wait for the result before continuing.");
        sb.AppendLine();
        sb.AppendLine("## Tools");
        sb.AppendLine("- read_file(path: string, offset?: integer, limit?: integer): return the numbered lines of a file in the workspace.");
        sb.AppendLine("- write_file(path: string, content: string): create a new file in the workspace; use apply_patch to modify an existing file.");
        sb.AppendLine("- shell(command: string, timeout_seconds?: integer): run a command in the sandboxed workspace shell and return stdout, stderr and the exit code.");
        sb.AppendLine("- apply_patch(patch: string): atomically modify one file or multiple workspace files using anchored V4A hunks; also supports creating, renaming, and deleting files.");
        sb.AppendLine();
        sb.AppendLine("## Rules");
        sb.AppendLine("1. Read a file before editing it. Never guess its contents.");
        sb.AppendLine("2. Use apply_patch for every modification to an existing file, from a single-line edit to changes across multiple files. Use write_file only to create new files.");
        sb.AppendLine("3. After a code change, run the relevant tests or a syntax check with shell and report the result truthfully.");
        sb.AppendLine("4. Keep answers short. Do not repeat the tool output back to the user unless asked.");
        sb.AppendLine("5. If a command fails, show the error and propose one fix at a time.");
        sb.AppendLine();
        sb.AppendLine("## Skills");
        for (int i = 1; i <= 8; i++)
        {
            sb.AppendLine($"- skill-{i:00} ({SkillNames[i % SkillNames.Length]}): {SkillBlurbs[i % SkillBlurbs.Length]} " +
                          $"Invoke it with shell: `python skills/skill-{i:00}/run.py --input <file>`; it writes its report next to the input.");
        }
        sb.AppendLine();
        sb.AppendLine("The workspace root is /workspace. Paths are relative to it. The current date is 2026-09-09.");
        return sb.ToString();
    }

    private static readonly string[] SkillNames =
        { "csv-report", "pdf-extract", "web-research", "unit-test-writer", "doc-summarizer", "image-caption", "regex-builder", "json-validator" };

    private static readonly string[] SkillBlurbs =
    {
        "Turns a CSV table into a short markdown report with totals and outliers.",
        "Extracts text and tables from a PDF and writes them as markdown.",
        "Searches the web for a question and collects sources with quotes.",
        "Generates xUnit tests for a C# class from its public surface.",
        "Summarizes a long document into a one-page brief with headings.",
        "Describes an image and lists the objects it contains.",
        "Builds and explains a regular expression from examples.",
        "Validates a JSON document against a schema and lists every violation.",
    };

    private static readonly string[] CodeLines =
    {
        "using System;",
        "using System.Collections.Generic;",
        "using System.Linq;",
        "namespace Demo.Tools.Module{N}",
        "{",
        "    /// <summary>Accumulates widget measurements for batch {N}.</summary>",
        "    public sealed class Widget{N} : IDisposable",
        "    {",
        "        private readonly List<int> _items = new();",
        "        private bool _disposed;",
        "        public int Count => _items.Count;",
        "        public void Add(int value)",
        "        {",
        "            if (value < 0) throw new ArgumentOutOfRangeException(nameof(value), \"value must be non-negative\");",
        "            _items.Add(value * {N});",
        "        }",
        "        public int Sum() { int s = 0; foreach (int v in _items) s += v; return s; }",
        "        public double Mean() => _items.Count == 0 ? 0.0 : _items.Average();",
        "        public void Dispose() { if (_disposed) return; _items.Clear(); _disposed = true; }",
        "    }",
        "}",
        "// TODO({N}): review the boundary condition above before release {N}",
    };

    /// <summary>A numbered source listing (what read_file returns) of about
    /// <paramref name="targetTokens"/> tokens under this tokenizer.</summary>
    public static string CodeText(ITokenizer tokenizer, int targetTokens, int salt = 0)
    {
        var sb = new StringBuilder();
        int line = 1;
        List<int> tokens;
        while (true)
        {
            for (int i = 0; i < 40; i++, line++)
            {
                string src = CodeLines[(line + salt) % CodeLines.Length].Replace("{N}", (line + salt).ToString(CultureInfo.InvariantCulture));
                sb.Append(line.ToString(CultureInfo.InvariantCulture).PadLeft(4)).Append("  ").Append(src).Append('\n');
            }
            tokens = tokenizer.Encode(sb.ToString(), addSpecial: false);
            if (tokens.Count >= targetTokens) break;
        }
        // Cut at the last whole line at or before the target.
        string text = tokenizer.Decode(tokens.GetRange(0, targetTokens));
        int cut = text.LastIndexOf('\n');
        return cut > 0 ? text.Substring(0, cut + 1) : text;
    }
}
