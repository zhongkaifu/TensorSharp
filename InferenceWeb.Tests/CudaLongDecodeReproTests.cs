// Manual reproduction harness for the "garbled output after a few hundred
// tokens on CUDA" report. Gated behind TS_REPRO_BACKEND so it never runs in the
// normal suite (it loads a multi-GB model and decodes hundreds of steps).
//
// It drives the SAME continuous-batching InferenceEngine the server uses, with
// the real prompt rendered through the GGUF chat template, then decodes the
// output text so genuine garbling can be seen and localized.
//
// Usage (PowerShell):
//   $env:TS_REPRO_MODEL="C:\Works\models\gemma-4-12B-it-qat-q4_0.gguf"
//   $env:TS_REPRO_STEPS="900"
//   $env:TS_REPRO_BACKEND="cuda";  dotnet test --filter FullyQualifiedName~CudaLongDecodeRepro
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class CudaLongDecodeReproTests
{
    private readonly ITestOutputHelper _output;
    public CudaLongDecodeReproTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public async Task GreedyDecode_RealPrompt_RecordText()
    {
        string backendStr = Environment.GetEnvironmentVariable("TS_REPRO_BACKEND");
        if (string.IsNullOrWhiteSpace(backendStr))
        {
            _output.WriteLine("[repro] TS_REPRO_BACKEND not set; skipping.");
            return;
        }

        string modelPath = Environment.GetEnvironmentVariable("TS_REPRO_MODEL")
            ?? @"C:\Works\models\gemma-4-12B-it-qat-q4_0.gguf";
        int steps = int.TryParse(Environment.GetEnvironmentVariable("TS_REPRO_STEPS"), out int s) ? s : 900;
        string outPath = Environment.GetEnvironmentVariable("TS_REPRO_OUT") ?? $"repro_text_{backendStr}.txt";
        string promptText = Environment.GetEnvironmentVariable("TS_REPRO_PROMPT") ?? "请详细介绍最终幻想7";

        BackendType backend = backendStr.ToLowerInvariant() switch
        {
            "cuda" => BackendType.Cuda,
            "cpu" => BackendType.Cpu,
            "ggmlcpu" => BackendType.GgmlCpu,
            "ggmlcuda" => BackendType.GgmlCuda,
            "mlx" => BackendType.Mlx,
            _ => throw new ArgumentException($"unknown backend {backendStr}")
        };

        string mmproj = Environment.GetEnvironmentVariable("TS_REPRO_MMPROJ");
        _output.WriteLine($"[repro] backend={backend} model={modelPath} steps={steps} prompt={promptText} mmproj={mmproj}");
        var model = (Gemma4Model)ModelBase.Create(modelPath, backend);
        try
        {
            if (!string.IsNullOrWhiteSpace(mmproj) && File.Exists(mmproj))
            {
                model.MultimodalInjector.LoadProjectors(mmproj);
                _output.WriteLine($"[repro] loaded mmproj {mmproj}");
            }
            var history = new List<ChatMessage> { new() { Role = "user", Content = promptText } };
            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            List<int> promptTokens = renderer.RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, history, "gemma4", addGenerationPrompt: true);
            _output.WriteLine($"[repro] prompt tokens={promptTokens.Count}: {string.Join(",", promptTokens.Take(40))}");

            var cfg = SchedulerConfig.FromEnvironment();
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            var seq = new SequenceState("repro", promptTokens, maxNewTokens: steps,
                blockSize: cfg.BlockSize, samplingConfig: SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(seq);

            var outToks = new List<int>();
            await foreach (var t in handle.Tokens.ReadAllAsync())
                outToks.Add(t);

            string text = model.Tokenizer.Decode(outToks);
            File.WriteAllText(outPath, text);
            File.WriteAllText(outPath + ".ids", string.Join(",", outToks));

            int distinct = outToks.Distinct().Count();
            int firstLongRepeat = FirstLongRepeat(outToks, 30);
            _output.WriteLine($"[repro] produced {outToks.Count} tokens, distinct={distinct}, firstLongRepeat(step)={firstLongRepeat}");
            _output.WriteLine($"[repro] wrote text to {Path.GetFullPath(outPath)}");
            const int chunk = 300;
            for (int p = 0; p < text.Length; p += chunk)
                _output.WriteLine("[text] " + text.Substring(p, Math.Min(chunk, text.Length - p)));
        }
        finally { model.Dispose(); }
    }

    private static int FirstLongRepeat(List<int> toks, int run)
    {
        int r = 0, last = -1;
        for (int i = 0; i < toks.Count; i++)
        {
            if (toks[i] == last) { if (++r >= run) return i - run + 1; }
            else { r = 0; last = toks[i]; }
        }
        return -1;
    }
}
