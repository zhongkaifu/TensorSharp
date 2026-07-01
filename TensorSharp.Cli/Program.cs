// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Encodings.Web;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli.Logging;
using TensorSharp.Cpu;
using TensorSharp.Runtime;

namespace TensorSharp.Cli
{
    class Program
    {
        private static readonly IPromptRenderer PromptRenderer = new GgufPromptRenderer();
        private static ILogger _log = NullLogger.Instance;

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
            bool showSarah = Array.Exists(args, a => a == "--xzf");
            ConsoleBanner.Print(showSarah);

            var loggingOptions = CliLoggingSetup.ParseFromArgs(args);
            using var loggerFactory = CliLoggingSetup.Build(loggingOptions);
            _log = loggerFactory.CreateLogger("TensorSharp.Cli");
            _log.LogInformation(LogEventIds.CliStarted,
                "tensorsharp-cli started: argv={ArgCount} logLevel={LogLevel} logDir={LogDir} fileLogging={FileLogging} consoleLogging={ConsoleLogging}",
                args.Length, loggingOptions.MinimumLevel, loggingOptions.Directory,
                loggingOptions.FileEnabled, loggingOptions.ConsoleEnabled);

            // The ggml-metal device singleton asserts its residency set is empty
            // when its C++ static destructor runs at process exit; any GGML
            // backend buffer that outlives the run (e.g. the reusable prefill
            // compute buffer) must be freed first. Mirror the server's shutdown
            // wiring so the CLI exits cleanly on the GGML/Metal backend.
            AppDomain.CurrentDomain.ProcessExit += static (_, _) =>
            {
                try { TensorSharp.GGML.GgmlBasicOps.Shutdown(); }
                catch { /* native lib may be absent for non-GGML backends */ }
            };

            try
            {
                MainCore(args);
                _log.LogInformation(LogEventIds.CliCompleted, "tensorsharp-cli completed");
                try { TensorSharp.GGML.GgmlBasicOps.Shutdown(); }
                catch { /* native lib may be absent for non-GGML backends */ }
            }
            catch (Exception ex)
            {
                _log.LogCritical(LogEventIds.CliFailed, ex,
                    "tensorsharp-cli aborted with unhandled exception {ExceptionType}", ex.GetType().Name);
                throw;
            }
        }

        static void MainCore(string[] args)
        {
            // Pick up the KV cache dtype from the KV_CACHE_DTYPE environment variable
            // before parsing CLI args. The --kv-cache-dtype flag below overrides this.
            KvCacheDtypeConfig.ConfigureFromEnvironment();

            string modelPath = null;
            string inputFile = null;
            string outputFile = null;
            string imagePath = null;
            string audioPath = null;
            string videoPath = null;
            string mmProjPath = null;
            int maxTokens = 100;
            bool runTest = false;
            string backendStr = "ggml_cpu";
            string testTemplatesDir = null;
            string inputJsonl = null;
            string multiTurnJsonl = null;
            bool enableThinking = false;
            string toolsFile = null;
            bool dumpPrompt = false;
            bool runBenchmark = false;
            int benchmarkPrefill = 32;
            int benchmarkDecode = 64;
            int benchmarkRuns = 1;
            bool benchmarkChunked = false;
            bool runChunkedPrefillCorrectness = false;
            int correctnessPrefill = 1500;
            int correctnessDecode = 8;
            bool runKvCacheBenchmark = false;
            int kvCacheBenchTurns = 4;
            bool runPagedKvBenchmark = false;
            int pagedKvBenchPrompt = 2048;
            int pagedKvBenchTrials = 3;
            // Cross-session paged KV cache knobs. Each flag is plumbed through to
            // the matching env var so any code that calls
            // PagedKvCacheConfig.FromEnvironment() picks it up. The CLI
            // benchmark still exercises the standalone PagedKvCacheManager.
            bool? pagedKvEnableOverride = null;
            int? pagedKvBlockSizeOverride = null;
            long? pagedKvRamMbOverride = null;
            string pagedKvSsdDirOverride = null;
            long? pagedKvSsdMbOverride = null;
            int? pagedKvQuantBitsOverride = null;
            bool runInteractive = false;
            string systemPrompt = null;
            int warmupInferenceRuns = 0;
            // DiffusionGemma sampler knobs (used only for the diffusion-gemma architecture).
            int diffusionSteps = 48;
            int diffusionSeed = 0;
            int diffusionBlocks = 0;   // 0 => derive from --max-tokens and canvas_length
            // Qwen-Image-Edit knobs.
            string editPrompt = null;
            float cfgScale = 2.5f;   // Qwen-Image-Edit-2511 recommendation; 4.0 over-guides (distorts faces)
            // Qwen-Image-Edit companion GGUFs. The qwen_image DiT GGUF (passed via
            // --model) carries none of these, so the operator can point at them
            // explicitly instead of relying on a same-directory scan / env vars.
            string qwenImageVaePath = null;
            string qwenImageVlPath = null;
            string qwenImageMmprojPath = null;

            var samplingConfig = SamplingConfig.Greedy;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model": modelPath = args[++i]; break;
                    case "--input": inputFile = args[++i]; break;
                    case "--input-jsonl": inputJsonl = args[++i]; break;
                    case "--output": outputFile = args[++i]; break;
                    case "--image": imagePath = args[++i]; break;
                    case "--prompt": editPrompt = args[++i]; break;
                    case "--cfg": cfgScale = float.Parse(args[++i]); break;
                    case "--qwen-image-vae": qwenImageVaePath = args[++i]; break;
                    case "--qwen-image-vl": qwenImageVlPath = args[++i]; break;
                    case "--qwen-image-mmproj": qwenImageMmprojPath = args[++i]; break;
                    case "--audio": audioPath = args[++i]; break;
                    case "--video": videoPath = args[++i]; break;
                    case "--mmproj": mmProjPath = args[++i]; break;
                    case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
                    case "--test": runTest = true; break;
                    case "--backend": backendStr = args[++i].ToLowerInvariant(); break;
                    case "--test-templates": testTemplatesDir = args[++i]; break;
                    case "--think": enableThinking = true; break;
                    case "--tools": toolsFile = args[++i]; break;
                    case "--dump-prompt": dumpPrompt = true; break;
                    case "--multi-turn-jsonl": multiTurnJsonl = args[++i]; break;
                    case "--benchmark": runBenchmark = true; break;
                    case "--bench-prefill": benchmarkPrefill = int.Parse(args[++i]); break;
                    case "--bench-decode": benchmarkDecode = int.Parse(args[++i]); break;
                    case "--bench-runs": benchmarkRuns = int.Parse(args[++i]); break;
                    case "--bench-chunked": benchmarkChunked = true; break;
                    case "--test-chunked-prefill": runChunkedPrefillCorrectness = true; break;
                    case "--correct-prefill": correctnessPrefill = int.Parse(args[++i]); break;
                    case "--correct-decode": correctnessDecode = int.Parse(args[++i]); break;
                    case "--bench-kvcache": runKvCacheBenchmark = true; break;
                    case "--bench-kv-turns": kvCacheBenchTurns = int.Parse(args[++i]); break;
                    case "--paged-bench": runPagedKvBenchmark = true; break;
                    case "--paged-bench-prompt": pagedKvBenchPrompt = int.Parse(args[++i]); break;
                    case "--paged-bench-trials": pagedKvBenchTrials = int.Parse(args[++i]); break;
                    case "--paged-kv":
                    case "--paged-kv-cache":
                        pagedKvEnableOverride = true;
                        break;
                    case "--no-paged-kv":
                    case "--no-paged-kv-cache":
                        pagedKvEnableOverride = false;
                        break;
                    case "--continuous-batching":
                    case "--paged-batching":
                        // Paged-attention continuous batching path. Gates two
                        // env vars: TS_SCHED_DISABLE_BATCHED (scheduler —
                        // falls through to per-seq KV-swap when set) and
                        // TS_QWEN35_BATCHED (Qwen3.5 ForwardBatch gate). Both
                        // default ON; this flag is idempotent with the default
                        // and kept for explicit operator intent or for
                        // overriding a previous --no-continuous-batching.
                        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
                        Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "1");
                        break;
                    case "--no-continuous-batching":
                    case "--no-paged-batching":
                        Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
                        Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "0");
                        break;
                    case "--paged-kv-block-size":
                        pagedKvBlockSizeOverride = int.Parse(args[++i]);
                        break;
                    case "--paged-kv-ram-mb":
                        pagedKvRamMbOverride = long.Parse(args[++i]);
                        break;
                    case "--paged-kv-ssd-dir":
                        pagedKvSsdDirOverride = args[++i];
                        break;
                    case "--paged-kv-ssd-mb":
                        pagedKvSsdMbOverride = long.Parse(args[++i]);
                        break;
                    case "--paged-kv-quant-bits":
                    {
                        string bitsStr = args[++i];
                        if (!int.TryParse(bitsStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out int bitsVal))
                            throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: '{bitsStr}'. Expected 0 (off), 2, 4, or 8.");
                        if (bitsVal != 0 && bitsVal != 2 && bitsVal != 4 && bitsVal != 8)
                            throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: {bitsVal}. Expected 0 (off), 2, 4, or 8.");
                        pagedKvQuantBitsOverride = bitsVal;
                        break;
                    }
                    case "--warmup-runs": warmupInferenceRuns = int.Parse(args[++i]); break;
                    case "--diffusion-steps": diffusionSteps = int.Parse(args[++i]); break;
                    case "--diffusion-seed": diffusionSeed = int.Parse(args[++i]); break;
                    case "--diffusion-blocks": diffusionBlocks = int.Parse(args[++i]); break;
                    case "--kv-cache-dtype":
                        {
                            string kvDtypeStr = args[++i];
                            if (!KvCacheDtypeConfig.TryParse(kvDtypeStr, out KvCacheDtype kvDtype))
                                throw new ArgumentException($"Unknown --kv-cache-dtype value '{kvDtypeStr}'. Valid: f32, f16, q8_0, q4_0.");
                            KvCacheDtypeConfig.Set(kvDtype);
                            break;
                        }
                    case "-i":
                    case "--interactive":
                    case "--chat":
                        runInteractive = true;
                        break;
                    case "--system": systemPrompt = args[++i]; break;
                    case "--system-file":
                        {
                            string spPath = args[++i];
                            if (!File.Exists(spPath))
                                throw new FileNotFoundException($"System prompt file not found: {spPath}", spPath);
                            systemPrompt = File.ReadAllText(spPath);
                        }
                        break;
                    case "--temperature": samplingConfig.Temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--top-k": samplingConfig.TopK = int.Parse(args[++i]); break;
                    case "--top-p": samplingConfig.TopP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--min-p": samplingConfig.MinP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--repeat-penalty": samplingConfig.RepetitionPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--presence-penalty": samplingConfig.PresencePenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--frequency-penalty": samplingConfig.FrequencyPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); break;
                    case "--seed": samplingConfig.Seed = int.Parse(args[++i]); break;
                    case "--stop":
                        samplingConfig.StopSequences ??= new List<string>();
                        samplingConfig.StopSequences.Add(args[++i]);
                        break;
                }
            }

            List<ToolFunction> tools = null;
            if (toolsFile != null)
            {
                if (!File.Exists(toolsFile))
                {
                    _log.LogError(LogEventIds.CliFailed, "Tools file not found: {ToolsFile}", toolsFile);
                    return;
                }
                tools = JsonSerializer.Deserialize<List<ToolFunction>>(File.ReadAllText(toolsFile),
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Loaded {ToolCount} tool definition(s) from {ToolsFile}", tools.Count, toolsFile);
            }

            if (testTemplatesDir != null)
            {
                TestChatTemplates(testTemplatesDir);
                return;
            }

            if (modelPath == null)
            {
                string binDir = AppContext.BaseDirectory;
                string[] candidates = {
                    Path.Combine(binDir, "Qwen3.5-9B-Q8_0.gguf"),
                    Path.Combine(binDir, "Qwen3-4B.fp16.gguf"),
                    "/Users/ZhongkaiFu/Downloads/Qwen3-4B.fp16.gguf",
                };
                modelPath = candidates.FirstOrDefault(File.Exists);
            }

            if (modelPath == null || !File.Exists(modelPath))
            {
                _log.LogError(LogEventIds.CliFailed,
                "Model file not found: {ModelPath}", modelPath ?? "(none)");
                Console.Error.WriteLine("Usage: TensorSharp.Cli --model <path.gguf> [--input <input.txt>] " +
                    "[--input-jsonl <requests.jsonl>] [--image <image.png>] [--output <output.txt>] " +
                    "[--prompt <text>] [--cfg F] [--diffusion-steps N] [--diffusion-seed N] " +
                    "[--qwen-image-vae <vae.gguf>] [--qwen-image-vl <qwen2.5-vl.gguf>] [--qwen-image-mmproj <mmproj.gguf>] " +
                    "[--max-tokens N] [--test] [--backend cpu|cuda|mlx|ggml_cpu|ggml_metal|ggml_cuda] " +
                    "[--interactive] [--system <text>] [--system-file <path>] " +
                    "[--temperature F] [--top-k N] [--top-p F] [--min-p F] " +
                    "[--repeat-penalty F] [--presence-penalty F] [--frequency-penalty F] " +
                    "[--seed N] [--stop <text>] [--think] [--warmup-runs N] " +
                    "[--paged-kv | --no-paged-kv] [--paged-kv-block-size N] [--paged-kv-ram-mb N] " +
                    "[--paged-kv-ssd-dir <path>] [--paged-kv-ssd-mb N] [--paged-kv-quant-bits 0|2|4|8] " +
                    "[--log-level info|debug|trace] [--log-dir <path>] [--log-file off] [--log-console off]");
                return;
            }

            BackendType backend = backendStr switch
            {
                "mlx" or "mlx_metal" or "mlx-metal" => BackendType.Mlx,
                "cpu" => BackendType.Cpu,
                "cuda" or "direct_cuda" or "direct-cuda" => BackendType.Cuda,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cuda" or "ggml-cuda" => BackendType.GgmlCuda,
                _ => throw new ArgumentException($"Unknown backend '{backendStr}'. Use: cpu, cuda, mlx, ggml_cpu, ggml_metal, ggml_cuda"),
            };

            ApplyPagedKvCacheCliOverrides(
                pagedKvEnableOverride, pagedKvBlockSizeOverride,
                pagedKvRamMbOverride, pagedKvSsdDirOverride, pagedKvSsdMbOverride,
                pagedKvQuantBitsOverride);

            // Qwen-Image-Edit: let the operator override the companion GGUFs that
            // QwenImageModel otherwise resolves from a same-directory scan. These
            // are translated into the env vars QwenImageModel reads (the existing
            // override mechanism) and validated here so a typo fails fast instead
            // of silently falling back to the directory scan.
            ApplyQwenImageCompanionOverride("--qwen-image-vae", "TS_QWEN_IMAGE_VAE", qwenImageVaePath);
            ApplyQwenImageCompanionOverride("--qwen-image-vl", "TS_QWEN_IMAGE_TE", qwenImageVlPath);
            ApplyQwenImageCompanionOverride("--qwen-image-mmproj", "TS_QWEN_IMAGE_MMPROJ", qwenImageMmprojPath);

            string requestedDtype = KvCacheDtypeConfig.IsExplicitlySet
                ? KvCacheDtypeConfig.Current.ToShortString()
                : "auto";
            _log.LogInformation(LogEventIds.ModelLoadStarted,
                "Loading model {ModelFile} on backend {Backend} kvCacheDtype={KvCacheDtype} (path={ModelPath})",
                Path.GetFileName(modelPath), backend, requestedDtype, modelPath);
            var modelLoadSw = Stopwatch.StartNew();
            using var model = ModelBase.Create(modelPath, backend);
            modelLoadSw.Stop();
            _log.LogInformation(LogEventIds.ModelLoadCompleted,
                "Loaded model {ModelFile} architecture={Architecture} contextLength={ContextLength} kvCacheDtype={KvCacheDtype} elapsedMs={ElapsedMs:F1}",
                Path.GetFileName(modelPath), model.Config.Architecture ?? "(unknown)",
                model.MaxContextLength, model.KvCacheDtype.ToShortString(),
                modelLoadSw.Elapsed.TotalMilliseconds);

            // Qwen-Image-Edit: prompt + input image -> modified image (no autoregressive path).
            if (model is TensorSharp.Models.QwenImage.QwenImageModel qwenImageModel)
            {
                if (imagePath == null)
                {
                    Console.Error.WriteLine("Qwen-Image-Edit requires --image <input.png>. Optionally --prompt, --output, --diffusion-steps, --cfg, --diffusion-seed.");
                    return;
                }
                string prompt = editPrompt
                    ?? (inputFile != null && File.Exists(inputFile) ? File.ReadAllText(inputFile).Trim() : "");
                string outPath = outputFile ?? "edited.png";
                RunImageEdit(qwenImageModel, imagePath, prompt, outPath, diffusionSteps, cfgScale, diffusionSeed);
                return;
            }

            var warmupSw = Stopwatch.StartNew();
            model.WarmUpKernels();
            warmupSw.Stop();
            _log.LogInformation(LogEventIds.HostConfiguration,
                "Kernel warmup completed in {ElapsedMs:F1} ms", warmupSw.Elapsed.TotalMilliseconds);

            if (mmProjPath != null)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Loading mmproj projector from {MmProj}", mmProjPath);
                model.MultimodalInjector.LoadProjectors(mmProjPath);
            }
            else if (imagePath != null && model.Config.Architecture == "gemma3")
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "mmproj-gemma3-4b-f16.gguf");
                if (File.Exists(autoMmproj))
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Auto-loading vision encoder: {MmProj}", autoMmproj);
                    model.MultimodalInjector.LoadProjectors(autoMmproj);
                }
            }
            else if (imagePath != null && model.Config.Architecture == "mistral3")
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "mistral3-mmproj.gguf");
                if (File.Exists(autoMmproj))
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Auto-loading Mistral3 vision encoder: {MmProj}", autoMmproj);
                    model.MultimodalInjector.LoadProjectors(autoMmproj);
                }
            }
            else if ((imagePath != null || audioPath != null || videoPath != null)
                     && model.Config.Architecture == "gemma4")
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "gemma-4-mmproj-F16.gguf");
                if (File.Exists(autoMmproj))
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Auto-loading multimodal encoder: {MmProj}", autoMmproj);
                    model.MultimodalInjector.LoadProjectors(autoMmproj);
                }
            }
            else if (imagePath != null &&
                     (model.Config.Architecture == "qwen35" ||
                      model.Config.Architecture == "qwen35moe" ||
                      model.Config.Architecture == "qwen3next"))
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "Qwen3.5-mmproj-F16.gguf");
                if (File.Exists(autoMmproj))
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Auto-loading vision encoder: {MmProj}", autoMmproj);
                    model.MultimodalInjector.LoadProjectors(autoMmproj);
                }
            }
            else if ((imagePath != null || audioPath != null || videoPath != null) &&
                     (model.Config.Architecture == "nemotron_h_omni" ||
                      model.Config.Architecture == "nemotron_h" ||
                      model.Config.Architecture == "nemotron_h_moe"))
            {
                string modelDir = Path.GetDirectoryName(modelPath);
                // Look for any mmproj-suffixed companion file in the same directory.
                string autoMmproj = null;
                if (modelDir != null && Directory.Exists(modelDir))
                {
                    foreach (string candidate in Directory.GetFiles(modelDir, "*mmproj*.gguf"))
                    {
                        if (candidate.IndexOf("Nemotron", StringComparison.OrdinalIgnoreCase) >= 0)
                        {
                            autoMmproj = candidate;
                            break;
                        }
                    }
                }
                if (autoMmproj != null && File.Exists(autoMmproj))
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Auto-loading Nemotron vision encoder: {MmProj}", autoMmproj);
                    model.MultimodalInjector.LoadProjectors(autoMmproj);
                }
            }

            if (runTest)
            {
                RunTests(model, maxTokens, outputFile);
                return;
            }

            if (runBenchmark)
            {
                RunBenchmark(model, benchmarkPrefill, benchmarkDecode, benchmarkRuns, benchmarkChunked);
                return;
            }

            if (runChunkedPrefillCorrectness)
            {
                RunChunkedPrefillCorrectness(model, correctnessPrefill, correctnessDecode);
                return;
            }

            if (runKvCacheBenchmark)
            {
                RunKvCacheBenchmark(model, kvCacheBenchTurns, maxTokens, samplingConfig, enableThinking);
                return;
            }

            if (runPagedKvBenchmark)
            {
                RunPagedKvBenchmark(model, pagedKvBenchPrompt, pagedKvBenchTrials);
                return;
            }

            if (multiTurnJsonl != null)
            {
                RunMultiTurnTest(model, multiTurnJsonl, maxTokens, samplingConfig, enableThinking);
                return;
            }

            if (inputJsonl != null)
            {
                RunJsonlBatch(model, inputJsonl, outputFile, maxTokens, samplingConfig, enableThinking);
                return;
            }

            if (runInteractive)
            {
                _log.LogInformation(LogEventIds.CliStarted,
                    "Entering interactive chat mode (model={Model}, backend={Backend}, thinking={Thinking})",
                    Path.GetFileName(modelPath), backend, enableThinking);

                // Apply --system / --system-file by prepending it to the running
                // history before the loop starts; the user can still override
                // it inside the session via the /system command. We forward
                // the model path / backend / mmproj so the session's /info,
                // /model and /backend commands have something concrete to
                // reload against.
                var session = new InteractiveSession(
                    model,
                    modelPath,
                    backend,
                    mmProjPath,
                    PromptRenderer,
                    samplingConfig,
                    tools,
                    enableThinking,
                    maxTokens > 0 ? maxTokens : 512,
                    _log);
                if (!string.IsNullOrEmpty(systemPrompt))
                    session.SetInitialSystemPrompt(systemPrompt);
                session.Run();
                return;
            }

            string rawText;
            // When the user supplies their own prompt via --input, honour it for
            // every modality (text, image, video, audio). Only fall back to a
            // modality-specific default prompt when no input file was given.
            bool hasUserInput = inputFile != null && File.Exists(inputFile);
            if (hasUserInput)
            {
                rawText = File.ReadAllText(inputFile).TrimEnd();
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Loaded input from {InputFile} ({Chars} chars)", inputFile, rawText.Length);
            }
            else
            {
                rawText = "What is 1+1?";
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "No input file specified; using default prompt: \"{Prompt}\"", rawText);
            }

            // DiffusionGemma uses an iterative denoising sampler rather than autoregressive decode.
            if (model is DiffusionGemmaModel diffusionModel)
            {
                RunDiffusion(diffusionModel, rawText, systemPrompt, maxTokens, outputFile,
                    diffusionSteps, diffusionSeed, diffusionBlocks);
                return;
            }

            List<string> imagePaths = null;
            List<string> audioPaths = null;

            if (videoPath != null)
            {
                if (!File.Exists(videoPath))
                {
                    _log.LogError(LogEventIds.CliFailed, "Video file not found: {VideoPath}", videoPath);
                    return;
                }
                _log.LogInformation(LogEventIds.UploadReceived,
                    "Video input: {VideoPath} ({Bytes})",
                    videoPath, LoggingExtensions.FormatBytes(new FileInfo(videoPath).Length));
                imagePaths = MediaHelper.ExtractVideoFrames(videoPath);
                _log.LogInformation(LogEventIds.VideoFrameDownsample,
                    "Extracted {FrameCount} frames from video", imagePaths.Count);
                if (!hasUserInput)
                    rawText = "What is happening in this video? Please describe it.";
            }
            else if (imagePath != null)
            {
                if (!File.Exists(imagePath))
                {
                    _log.LogError(LogEventIds.CliFailed, "Image file not found: {ImagePath}", imagePath);
                    return;
                }
                imagePaths = new List<string> { imagePath };
                if (!hasUserInput)
                    rawText = "What is in this image? Please describe it.";
                _log.LogInformation(LogEventIds.UploadReceived,
                    "Image input: {ImagePath} ({Bytes})",
                    imagePath, LoggingExtensions.FormatBytes(new FileInfo(imagePath).Length));
            }

            if (audioPath != null)
            {
                if (!File.Exists(audioPath))
                {
                    _log.LogError(LogEventIds.CliFailed, "Audio file not found: {AudioPath}", audioPath);
                    return;
                }
                audioPaths = new List<string> { audioPath };
                if (!hasUserInput)
                    rawText = "Listen to this audio and describe what you hear.";
                _log.LogInformation(LogEventIds.UploadReceived,
                    "Audio input: {AudioPath} ({Bytes})",
                    audioPath, LoggingExtensions.FormatBytes(new FileInfo(audioPath).Length));
            }

            if (dumpPrompt)
            {
                var dumpMessages = new List<ChatMessage>
                {
                    new ChatMessage { Role = "user", Content = rawText }
                };
                string rendered = PromptRenderer.Render(
                    model.Config.ChatTemplate, dumpMessages, addGenerationPrompt: true,
                    architecture: model.Config.Architecture, tools: tools, enableThinking: enableThinking);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Dumped rendered prompt ({Chars} chars)", rendered.Length);
                // Prompt dump is a developer tool; emit the rendered text on stdout so
                // it remains easy to pipe/copy regardless of log routing.
                Console.WriteLine("=== Rendered Prompt ===");
                Console.WriteLine(rendered);
                Console.WriteLine($"=== End ({rendered.Length} chars, ends with: {(rendered.Length > 0 ? $"0x{(int)rendered[rendered.Length-1]:X2}" : "empty")}) ===");
                var tokens = model.Tokenizer.Encode(rendered, addSpecial: true);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Tokenized prompt: count={TokenCount} first20=[{First20}] last10=[{Last10}]",
                    tokens.Count,
                    string.Join(", ", tokens.GetRange(0, Math.Min(20, tokens.Count))),
                    string.Join(", ", tokens.GetRange(Math.Max(0, tokens.Count - 10), Math.Min(10, tokens.Count))));
                Console.WriteLine($"Token count: {tokens.Count}");
                Console.WriteLine($"First 20 tokens: [{string.Join(", ", tokens.GetRange(0, Math.Min(20, tokens.Count)))}]");
                Console.WriteLine($"Last 10 tokens: [{string.Join(", ", tokens.GetRange(Math.Max(0, tokens.Count - 10), Math.Min(10, tokens.Count)))}]");
                return;
            }

            // Per-turn upload manifest: include the path AND saved filename of every
            // attachment for this turn so the CLI inference log carries the same
            // upload audit trail as the server's chat.start line.
            string cliTurnUploads = FormatUploadsForCli(imagePaths, audioPaths, videoPath);

            _log.LogInformation(LogEventIds.ChatStarted,
                "cli.inference.start tokensRequested={MaxTokens} thinking={Thinking} tools={ToolCount} input=\"{Input}\" images={ImageCount} audio={AudioCount} video={Video} uploads={Uploads} warmupRuns={WarmupRuns}",
                maxTokens, enableThinking, tools?.Count ?? 0,
                LoggingExtensions.SanitizeForLog(rawText), imagePaths?.Count ?? 0,
                audioPaths?.Count ?? 0, videoPath != null, cliTurnUploads, warmupInferenceRuns);

            // --warmup-runs N : run the full inference path N times silently
            // before the real timed pass.  This forces Metal pipeline JIT
            // (kernel variants for the actual prefill batch size, KV-cache
            // attention shapes, etc.) and allocator pool growth to happen
            // once per shape, so the timed run reflects steady-state speed
            // rather than first-touch compile cost.  Decode tokens are kept
            // small during warmup so the wall cost is bounded.
            for (int w = 0; w < warmupInferenceRuns; w++)
            {
                int warmupDecodeTokens = Math.Min(maxTokens, 4);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "cli.inference.warmup run={Run}/{Total} decodeTokens={DecodeTokens}",
                    w + 1, warmupInferenceRuns, warmupDecodeTokens);
                _ = RunInference(model, rawText, imagePaths, warmupDecodeTokens, audioPaths,
                    isVideo: videoPath != null, samplingConfig: samplingConfig,
                    enableThinking: enableThinking, tools: tools, silent: true);
                model.ResetKVCache();
                model.ResetForwardTiming();
            }

            string result = RunInference(model, rawText, imagePaths, maxTokens, audioPaths,
                isVideo: videoPath != null, samplingConfig: samplingConfig,
                enableThinking: enableThinking, tools: tools);

            _log.LogInformation(LogEventIds.ChatCompleted,
                "cli.inference.complete chars={Chars} preview=\"{Preview}\"",
                result?.Length ?? 0, LoggingExtensions.SanitizeForLog(result ?? string.Empty, maxLength: 480));

            if (outputFile != null)
            {
                File.WriteAllText(outputFile, result);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Output written to {OutputFile} ({Chars} chars)",
                    outputFile, result?.Length ?? 0);
                Console.WriteLine($"Output written to {outputFile}");
            }
            else
            {
                Console.WriteLine("\n=== Generated Output ===");
                Console.WriteLine(result);
            }
        }

        /// <summary>
        /// Simulate multi-turn chat with KV cache reuse, matching the web UI behavior.
        /// Reads a JSONL file where each line is a user turn (just the user message).
        /// Each turn generates a response, uses the output parser to extract content,
        /// then builds the next turn's messages including previous turns.
        /// </summary>
        static void RunMultiTurnTest(ModelBase model, string jsonlPath, int maxTokens,
            SamplingConfig sampling, bool enableThinking)
        {
            if (!File.Exists(jsonlPath))
            {
                _log.LogError(LogEventIds.CliFailed, "Multi-turn jsonl not found: {File}", jsonlPath);
                return;
            }

            string[] lines = File.ReadAllLines(jsonlPath);
            var history = new List<ChatMessage>();
            string arch = model.Config.Architecture;
            int swa = model.Config.SlidingWindow;

            // Conversation cache state - drives KV cache reuse across turns by tracking
            // the canonical token sequence currently in the model and splicing the raw
            // output tokens of past assistant turns directly into the rendered prompt.
            var kvCache = new KVCache();
            var renderer = new KVCachePromptRenderer(PromptRenderer);

            _log.LogInformation(LogEventIds.CliBenchmark,
                "multi-turn test starting: turns={Turns} thinking={Thinking} swa={SWA} arch={Architecture}",
                lines.Length, enableThinking, swa, arch);

            for (int turn = 0; turn < lines.Length; turn++)
            {
                string line = lines[turn].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                string userMsg;
                int turnMaxTokens = maxTokens;
                bool forceReset = false;
                try
                {
                    var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    userMsg = root.TryGetProperty("content", out var c) ? c.GetString() : line;
                    if (root.TryGetProperty("max_tokens", out var mt))
                        turnMaxTokens = mt.GetInt32();
                    if (root.TryGetProperty("force_reset", out var fr))
                        forceReset = fr.GetBoolean();
                }
                catch
                {
                    userMsg = line;
                }

                history.Add(new ChatMessage { Role = "user", Content = userMsg });
                _log.LogInformation(LogEventIds.ChatStarted,
                    "multi-turn turn={Turn}/{TotalTurns} user=\"{User}\"",
                    turn + 1, lines.Length, LoggingExtensions.SanitizeForLog(userMsg));

                if (forceReset)
                {
                    _log.LogInformation(LogEventIds.SessionReset,
                        "multi-turn forcing KV cache reset (per JSONL force_reset flag)");
                    kvCache.Reset();
                    model.ResetKVCache();
                }

                var inputTokens = renderer.RenderToTokens(
                    model.Tokenizer,
                    model.Config.ChatTemplate,
                    history,
                    arch,
                    addGenerationPrompt: true,
                    enableThinking: enableThinking);

                _log.LogInformation(LogEventIds.ChatStarted,
                    "multi-turn prompt tokens={PromptTokens}", inputTokens.Count);

                var sw = Stopwatch.StartNew();
                ReusePlan plan = kvCache.PlanReuse(inputTokens, model.SupportsKVCacheTruncation);
                float[] logits = ApplyReusePlan(model, kvCache, plan, inputTokens);
                double prefillMs = sw.Elapsed.TotalMilliseconds;

                _log.LogInformation(LogEventIds.KvCacheReusePlan,
                    "kv plan={Plan} prefillMs={PrefillMs:F1} description={Description}",
                    plan.Kind, prefillMs, DescribePlan(plan, inputTokens.Count));

                var cfg = sampling ?? SamplingConfig.Greedy;
                var sampler = new TokenSampler(cfg);
                var generatedTokens = new List<int>();
                var sb = new StringBuilder();

                var decodeSw = Stopwatch.StartNew();
                for (int step = 0; step < turnMaxTokens; step++)
                {
                    int nextToken = sampler.Sample(logits, generatedTokens);
                    if (model.Tokenizer.IsEos(nextToken)) break;
                    generatedTokens.Add(nextToken);
                    string decoded = model.Tokenizer.Decode(generatedTokens);
                    sb.Clear();
                    sb.Append(decoded);
                    logits = model.Forward(new[] { nextToken });
                    kvCache.RecordAppend(nextToken, logits);
                }
                double decodeMs = decodeSw.Elapsed.TotalMilliseconds;

                string rawOutput = sb.ToString();

                var parser = OutputParserFactory.Create(arch);
                parser.Init(enableThinking, null);
                var parsed = parser.Add(rawOutput, true);
                string content = parsed.Content ?? "";
                string thinking = parsed.Thinking ?? "";

                if (thinking.Length > 0)
                    _log.LogInformation(LogEventIds.ChatCompleted,
                        "multi-turn thinking ({ThinkingChars} chars): {ThinkingPreview}",
                        thinking.Length, LoggingExtensions.SanitizeForLog(thinking));

                _log.LogInformation(LogEventIds.ChatCompleted,
                    "multi-turn content chars={ContentChars} tokens={Tokens} decodeMs={DecodeMs:F0} tokPerSec={TokensPerSec:F1} preview={ContentPreview}",
                    content.Length, generatedTokens.Count, decodeMs,
                    generatedTokens.Count / (decodeMs / 1000.0),
                    LoggingExtensions.SanitizeForLog(content, maxLength: 480));

                bool hasUnused = rawOutput.Contains("<unused");
                if (hasUnused)
                {
                    _log.LogError(LogEventIds.ChatFailed,
                        "multi-turn output contains <unused> tokens; first 500 chars: {RawPreview}",
                        rawOutput.Substring(0, Math.Min(500, rawOutput.Length)));
                    break;
                }

                // Append the assistant turn to the history with raw output tokens so the
                // NEXT turn's renderer can splice them in instead of re-tokenizing.
                history.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = content,
                    Thinking = thinking,
                    RawOutputTokens = generatedTokens,
                });
            }

            _log.LogInformation(LogEventIds.CliCompleted,
                "multi-turn test completed: {Turns} turns", history.Count / 2);
        }

        /// <summary>
        /// Apply a <see cref="ReusePlan"/> to bring the model's KV state up to date and
        /// return next-token logits. Mirrors the orchestration logic used by ModelService.
        /// </summary>
        static float[] ApplyReusePlan(ModelBase model, KVCache kvCache, ReusePlan plan, List<int> inputTokens)
        {
            switch (plan.Kind)
            {
                case ReusePlanKind.ExactMatch:
                    return plan.CachedLogits;

                case ReusePlanKind.PartialReuse:
                {
                    int reused = plan.ReusedPrefixLength;
                    int suffixLength = plan.TokensToForward;
                    model.TruncateKVCache(reused);
                    kvCache.TruncateTo(reused);

                    var suffix = new int[suffixLength];
                    for (int i = 0; i < suffixLength; i++)
                        suffix[i] = inputTokens[reused + i];
                    float[] logits = model.ForwardRefill(suffix);
                    kvCache.RecordAppend(suffix, logits);
                    return logits;
                }

                case ReusePlanKind.Reset:
                default:
                {
                    model.ResetKVCache();
                    kvCache.Reset();
                    var allTokens = inputTokens.ToArray();
                    float[] logits = model.Forward(allTokens);
                    kvCache.RecordAppend(allTokens, logits);
                    return logits;
                }
            }
        }

        static string DescribePlan(ReusePlan plan, int totalTokens)
        {
            return plan.Kind switch
            {
                ReusePlanKind.ExactMatch => $"Exact match: reusing all {totalTokens} cached tokens (saved 100%)",
                ReusePlanKind.PartialReuse => $"Partial reuse: keeping {plan.ReusedPrefixLength}/{totalTokens} tokens, forwarding {plan.TokensToForward} new (saved {100.0 * plan.ReusedPrefixLength / totalTokens:F0}%)",
                ReusePlanKind.Reset => $"Full reset: forwarding {plan.TokensToForward} tokens",
                _ => "(unknown plan)",
            };
        }

        static void RunJsonlBatch(ModelBase model, string inputJsonlPath, string outputFile, int defaultMaxTokens,
            SamplingConfig defaultSampling, bool enableThinking = false)
        {
            if (!File.Exists(inputJsonlPath))
            {
                _log.LogError(LogEventIds.CliFailed, "JSONL file not found: {File}", inputJsonlPath);
                return;
            }

            string[] lines = File.ReadAllLines(inputJsonlPath);
            var results = new List<string>();
            int total = lines.Length;
            int completed = 0;

            _log.LogInformation(LogEventIds.CliBatchProgress,
                "jsonl batch starting: total={Total} source={Source}",
                total, inputJsonlPath);

            var totalSw = Stopwatch.StartNew();

            for (int lineIdx = 0; lineIdx < lines.Length; lineIdx++)
            {
                string line = lines[lineIdx].Trim();
                if (string.IsNullOrEmpty(line)) continue;

                JsonDocument doc;
                try
                {
                    doc = JsonDocument.Parse(line);
                }
                catch (JsonException ex)
                {
                    _log.LogError(LogEventIds.CliFailed,
                        "jsonl batch line {LineNumber} invalid JSON: {Error}", lineIdx + 1, ex.Message);
                    results.Add(JsonSerializer.Serialize(new { line = lineIdx + 1, error = $"Invalid JSON: {ex.Message}" }));
                    continue;
                }

                var root = doc.RootElement;
                string id = root.TryGetProperty("id", out var idProp) ? idProp.GetString() : $"request_{lineIdx + 1}";

                _log.LogInformation(LogEventIds.CliBatchProgress,
                    "jsonl batch [{Index}/{Total}] processing request: {RequestId}",
                    lineIdx + 1, total, id);

                try
                {
                    var messages = ParseMessages(root);
                    int maxTokens = root.TryGetProperty("max_tokens", out var mt) ? mt.GetInt32() : defaultMaxTokens;
                    var sampling = ParseSamplingFromJson(root, defaultSampling);

                    var imagePaths = ParseStringList(root, "images");
                    var audioPaths = ParseStringList(root, "audios");
                    bool isVideo = root.TryGetProperty("is_video", out var iv) && iv.GetBoolean();

                    model.ResetKVCache();

                    bool reqThinking = enableThinking ||
                        (root.TryGetProperty("enable_thinking", out var etProp) && etProp.GetBoolean());

                    string rendered = PromptRenderer.Render(
                        model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                        architecture: model.Config.Architecture, enableThinking: reqThinking);

                    _log.LogDebug(LogEventIds.ChatStarted,
                        "jsonl batch [{RequestId}] rendered prompt thinking={Thinking} preview={Preview}",
                        id, reqThinking, LoggingExtensions.SanitizeForLog(rendered, maxLength: 320));

                    var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);
                    _log.LogDebug(LogEventIds.ChatStarted,
                        "jsonl batch [{RequestId}] inputTokens={TokenCount} first20=[{First20}]",
                        id, inputTokens.Count, string.Join(", ", inputTokens.Take(20)));

                    var sw = Stopwatch.StartNew();
                    float[] logits = model.Forward(inputTokens.ToArray());
                    double prefillMs = sw.Elapsed.TotalMilliseconds;

                    var cfg = sampling ?? SamplingConfig.Greedy;
                    var sampler = new TokenSampler(cfg);
                    var generatedTokens = new List<int>();
                    var sb = new StringBuilder();

                    for (int step = 0; step < maxTokens; step++)
                    {
                        int nextToken = sampler.Sample(logits, generatedTokens);
                        if (model.Tokenizer.IsEos(nextToken)) break;

                        generatedTokens.Add(nextToken);
                        string decoded = model.Tokenizer.Decode(generatedTokens);
                        sb.Clear();
                        sb.Append(decoded);

                        if (cfg.StopSequences != null)
                        {
                            var (trimmed, shouldStop) = sampler.CheckStopSequences(decoded);
                            if (shouldStop)
                            {
                                sb.Clear();
                                sb.Append(trimmed);
                                break;
                            }
                        }

                        logits = model.Forward(new[] { nextToken });
                    }

                    double totalMs = sw.Elapsed.TotalMilliseconds;
                    string output = sb.ToString();
                    double tokPerSec = generatedTokens.Count / (totalMs / 1000.0);

                    _log.LogInformation(LogEventIds.ChatCompleted,
                        "jsonl batch [{RequestId}] tokens={Tokens} tokPerSec={TokensPerSec:F1} totalMs={TotalMs:F1} output={OutputPreview}",
                        id, generatedTokens.Count, tokPerSec, totalMs,
                        LoggingExtensions.SanitizeForLog(output, maxLength: 320));

                    var resultObj = new Dictionary<string, object>
                    {
                        ["id"] = id,
                        ["output"] = output,
                        ["tokens_generated"] = generatedTokens.Count,
                        ["prefill_ms"] = Math.Round(prefillMs, 2),
                        ["total_ms"] = Math.Round(totalMs, 2),
                        ["tokens_per_second"] = Math.Round(tokPerSec, 2),
                    };
                    results.Add(JsonSerializer.Serialize(resultObj));
                    completed++;
                }
                catch (Exception ex)
                {
                    _log.LogError(LogEventIds.ChatFailed, ex,
                        "jsonl batch line {LineNumber} request {RequestId} failed: {Error}",
                        lineIdx + 1, id, ex.Message);
                    var errorObj = new Dictionary<string, object>
                    {
                        ["id"] = id,
                        ["error"] = ex.Message,
                    };
                    results.Add(JsonSerializer.Serialize(errorObj));
                }
            }

            totalSw.Stop();

            _log.LogInformation(LogEventIds.CliBatchProgress,
                "jsonl batch completed {Completed}/{Total} requests in {ElapsedSec:F1}s",
                completed, total, totalSw.Elapsed.TotalSeconds);

            if (outputFile != null)
            {
                File.WriteAllLines(outputFile, results);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Results written to {OutputFile} ({ResultCount} results)", outputFile, results.Count);
            }
            else
            {
                Console.WriteLine("\n=== Results (JSONL) ===");
                foreach (var r in results)
                    Console.WriteLine(r);
            }
        }

        static List<ChatMessage> ParseMessages(JsonElement root)
        {
            var messages = new List<ChatMessage>();

            if (root.TryGetProperty("messages", out var msgsArr) && msgsArr.ValueKind == JsonValueKind.Array)
            {
                foreach (var msg in msgsArr.EnumerateArray())
                {
                    var cm = new ChatMessage
                    {
                        Role = msg.TryGetProperty("role", out var r) ? r.GetString() : "user",
                        Content = msg.TryGetProperty("content", out var c) ? c.GetString() : "",
                    };
                    if (msg.TryGetProperty("images", out var imgs) && imgs.ValueKind == JsonValueKind.Array)
                        cm.ImagePaths = imgs.EnumerateArray().Select(e => e.GetString()).ToList();
                    if (msg.TryGetProperty("audios", out var auds) && auds.ValueKind == JsonValueKind.Array)
                        cm.AudioPaths = auds.EnumerateArray().Select(e => e.GetString()).ToList();
                    if (msg.TryGetProperty("is_video", out var isv))
                        cm.IsVideo = isv.GetBoolean();
                    messages.Add(cm);
                }
            }
            else if (root.TryGetProperty("prompt", out var prompt))
            {
                messages.Add(new ChatMessage { Role = "user", Content = prompt.GetString() });
            }

            return messages;
        }

        static List<string> ParseStringList(JsonElement root, string key)
        {
            if (!root.TryGetProperty(key, out var arr) || arr.ValueKind != JsonValueKind.Array)
                return null;
            var list = arr.EnumerateArray().Select(e => e.GetString()).Where(s => s != null).ToList();
            return list.Count > 0 ? list : null;
        }

        static SamplingConfig ParseSamplingFromJson(JsonElement root, SamplingConfig fallback)
        {
            bool hasAny = false;
            var cfg = new SamplingConfig
            {
                Temperature = fallback.Temperature,
                TopK = fallback.TopK,
                TopP = fallback.TopP,
                MinP = fallback.MinP,
                RepetitionPenalty = fallback.RepetitionPenalty,
                PresencePenalty = fallback.PresencePenalty,
                FrequencyPenalty = fallback.FrequencyPenalty,
                Seed = fallback.Seed,
                StopSequences = fallback.StopSequences != null ? new List<string>(fallback.StopSequences) : null,
            };

            if (root.TryGetProperty("temperature", out var temp)) { cfg.Temperature = (float)temp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("top_k", out var tk)) { cfg.TopK = tk.GetInt32(); hasAny = true; }
            if (root.TryGetProperty("top_p", out var tp)) { cfg.TopP = (float)tp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("min_p", out var mp)) { cfg.MinP = (float)mp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("repetition_penalty", out var rp)) { cfg.RepetitionPenalty = (float)rp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("presence_penalty", out var pp)) { cfg.PresencePenalty = (float)pp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("frequency_penalty", out var fp)) { cfg.FrequencyPenalty = (float)fp.GetDouble(); hasAny = true; }
            if (root.TryGetProperty("seed", out var sd)) { cfg.Seed = sd.GetInt32(); hasAny = true; }
            if (root.TryGetProperty("stop", out var st) && st.ValueKind == JsonValueKind.Array)
            {
                cfg.StopSequences = st.EnumerateArray().Select(e => e.GetString()).Where(s => s != null).ToList();
                hasAny = true;
            }

            return hasAny ? cfg : fallback;
        }

        static void RunImageEdit(TensorSharp.Models.QwenImage.QwenImageModel model,
            string imagePath, string prompt, string outputPath, int steps, float cfgScale, int seed)
        {
            if (!File.Exists(imagePath))
            {
                Console.Error.WriteLine($"Input image not found: {imagePath}");
                return;
            }
            Console.WriteLine($"=== Qwen-Image-Edit ===");
            Console.WriteLine($"  input  : {imagePath}");
            Console.WriteLine($"  prompt : {prompt}");
            Console.WriteLine($"  steps={steps} cfg={cfgScale} seed={seed} -> {outputPath}");

            var input = TensorSharp.Models.QwenImage.ImageIO.Load(imagePath);
            var p = new TensorSharp.Models.QwenImage.QwenImageParams
            {
                Steps = steps,
                CfgScale = cfgScale,
                Seed = seed,
            };
            var sw = Stopwatch.StartNew();
            var output = model.EditImage(prompt, input, p);
            sw.Stop();
            TensorSharp.Models.QwenImage.ImageIO.SavePng(outputPath, output);
            Console.WriteLine($"Saved {output.Width}x{output.Height} edited image to {outputPath} " +
                $"({sw.Elapsed.TotalSeconds:F1}s, {sw.Elapsed.TotalMilliseconds / Math.Max(1, steps):F0} ms/step)");
        }

        static void RunDiffusion(DiffusionGemmaModel model, string rawText, string systemPrompt,
            int maxTokens, string outputFile, int steps, int seed, int blocks)
        {
            var messages = new List<ChatMessage>();
            if (!string.IsNullOrEmpty(systemPrompt))
                messages.Add(new ChatMessage { Role = "system", Content = systemPrompt });
            messages.Add(new ChatMessage { Role = "user", Content = rawText });

            string rendered = PromptRenderer.Render(
                model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                architecture: model.Config.Architecture);

            var promptTokens = model.Tokenizer.Encode(rendered, addSpecial: true).ToArray();

            int canvas = model.CanvasLength;
            int nBlocks = blocks > 0 ? blocks : Math.Max(1, (Math.Max(1, maxTokens) + canvas - 1) / canvas);

            var p = new DiffusionEbParams
            {
                MaxDenoisingSteps = steps,
                Seed = seed,
                MaxBlocks = nBlocks,
            };

            Console.WriteLine();
            Console.WriteLine("=== DiffusionGemma generation ===");
            Console.WriteLine($"Prompt tokens: {promptTokens.Length}, canvas={canvas}, max_steps={steps}, blocks={nBlocks}, seed={seed}, self_conditioning={model.SelfConditioningEnabled}");

            var sampler = new DiffusionGemmaSampler(model);
            var sw = Stopwatch.StartNew();
            int totalSteps = 0;
            var generated = sampler.Generate(promptTokens, p, (block, step, total, _) =>
            {
                totalSteps++;
                Console.Write($"\rblock {block + 1}/{nBlocks}  step {step + 1}/{total}  (elapsed {sw.Elapsed.TotalSeconds:F1}s)   ");
            });
            sw.Stop();
            Console.WriteLine();

            string output = model.Tokenizer.Decode(generated);
            Console.WriteLine();
            Console.WriteLine("=== Output ===");
            Console.WriteLine(output);
            Console.WriteLine();
            Console.WriteLine("=== Stats ===");
            Console.WriteLine($"Generated {generated.Count} tokens in {totalSteps} forward steps over {nBlocks} block(s).");
            Console.WriteLine($"Total time: {sw.Elapsed.TotalSeconds:F2}s; {sw.Elapsed.TotalMilliseconds / Math.Max(1, totalSteps):F1} ms/step.");
            model.PrintForwardTiming();

            if (outputFile != null)
            {
                File.WriteAllText(outputFile, output);
                Console.WriteLine($"Wrote output to {outputFile}");
            }
        }

        static string RunInference(ModelBase model, string rawText, List<string> imagePaths, int maxTokens,
            List<string> audioPaths = null, bool isVideo = false, SamplingConfig samplingConfig = null,
            bool enableThinking = false, List<ToolFunction> tools = null, bool silent = false)
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = rawText, ImagePaths = imagePaths, AudioPaths = audioPaths, IsVideo = isVideo }
            };

            string rendered = PromptRenderer.Render(
                model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                architecture: model.Config.Architecture,
                tools: tools, enableThinking: enableThinking);

            _log.LogDebug(LogEventIds.ChatStarted,
                "cli.inference rendered prompt chars={Chars} preview={Preview}",
                rendered.Length, LoggingExtensions.SanitizeForLog(rendered, maxLength: 480));

            var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);

            if (imagePaths != null && imagePaths.Count > 0)
            {
                string arch = model.Config.Architecture;

                if (arch == "gemma3")
                {
                    var proc = new Gemma3ImageProcessor();
                    int startId = model.Tokenizer.LookupToken("<start_of_image>");
                    if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
                    int endId = Gemma3ImageProcessor.EndOfImageToken;
                    int nlnlId = Gemma3ImageProcessor.NewlineNewlineToken;
                    int padId = Gemma3ImageProcessor.PadToken;

                    inputTokens = ChatTemplate.ExpandGemma3ImageTokens(inputTokens,
                        startId, endId, nlnlId, padId, proc.TokensPerImage);

                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Gemma3 vision: tokensPerImage={TokensPerImage} start={Start} end={End} totalTokens={TotalTokens}",
                        proc.TokensPerImage, startId, endId, inputTokens.Count);

                    if (model is Gemma3Model g3 && g3.VisionEncoder != null)
                    {
                        _log.LogDebug(LogEventIds.HostConfiguration,
                            "Processing image through Gemma3 vision encoder");
                        float[] pixels = proc.ProcessImage(imagePaths[0]);
                        var visionEmbeddings = g3.VisionEncoder.Encode(pixels);
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Gemma3 vision embeddings: {EmbeddingShape}",
                            $"{visionEmbeddings.Sizes[0]}x{visionEmbeddings.Sizes[1]}");

                        int imageTokenStart = -1;
                        for (int i = 0; i < inputTokens.Count; i++)
                        {
                            if (inputTokens[i] == startId && i + 1 < inputTokens.Count && inputTokens[i + 1] == padId)
                            {
                                imageTokenStart = i + 1;
                                break;
                            }
                        }

                        if (imageTokenStart >= 0)
                        {
                            g3.SetVisionEmbeddings(visionEmbeddings, imageTokenStart);
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Gemma3 vision embeddings injection position={Position}", imageTokenStart);
                        }
                        else
                        {
                            _log.LogWarning(LogEventIds.HostConfiguration,
                                "Gemma3 vision: could not find image placeholder position");
                            visionEmbeddings.Dispose();
                        }
                    }
                    else
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration,
                            "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                    }
                }
                else if (arch == "gemma4")
                {
                    int imageStartId = model.Tokenizer.LookupToken("<|image>");
                    int imageEndId = model.Tokenizer.LookupToken("<image|>");
                    if (imageStartId < 0) imageStartId = 255999;
                    if (imageEndId < 0) imageEndId = 256000;

                    if (model is Gemma4Model g4 && g4.VisionEncoder != null)
                    {
                        // The gemma4uv unified embedder declares its own
                        // image_mean / image_std (mean=0, std=1 -> [0,1]); the
                        // gemma4v SigLIP path keeps the legacy [-1,1] map. Match
                        // the server's ModelMultimodalInjector here so the CLI
                        // preprocesses identically.
                        var proc = g4.VisionEncoder.IsUnified
                            ? new Gemma4ImageProcessor(imageMean: g4.VisionEncoder.ImageMean,
                                imageStd: g4.VisionEncoder.ImageStd)
                            : new Gemma4ImageProcessor();
                        var allVisionEmbeddings = new List<TensorSharp.Tensor>();

                        foreach (var imgP in imagePaths)
                        {
                            var (pixels, imgW, imgH) = proc.ProcessImage(imgP);
                            var visionEmb = g4.VisionEncoder.Encode(pixels, imgW, imgH);
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Gemma4 vision frame: source={Source} resolution={Width}x{Height} embeddings={EmbeddingShape}",
                                imgP, imgW, imgH, $"{visionEmb.Sizes[0]}x{visionEmb.Sizes[1]}");
                            allVisionEmbeddings.Add(visionEmb);
                        }

                        // Expand each <|image> token and register embeddings for injection.
                        // Search forward past already-expanded tokens by tracking a search start.
                        int searchFrom = 0;
                        for (int imgIdx = 0; imgIdx < allVisionEmbeddings.Count; imgIdx++)
                        {
                            var visionEmbeddings = allVisionEmbeddings[imgIdx];
                            int numVisionTokens = (int)visionEmbeddings.Sizes[0];

                            int imageTokenPos = -1;
                            for (int i = searchFrom; i < inputTokens.Count; i++)
                            {
                                if (inputTokens[i] == imageStartId)
                                {
                                    imageTokenPos = i;
                                    break;
                                }
                            }

                            if (imageTokenPos >= 0)
                            {
                                var expanded = new List<int>();
                                for (int i = 0; i < imageTokenPos; i++)
                                    expanded.Add(inputTokens[i]);
                                expanded.Add(imageStartId);
                                for (int i = 0; i < numVisionTokens; i++)
                                    expanded.Add(0);
                                expanded.Add(imageEndId);
                                for (int i = imageTokenPos + 1; i < inputTokens.Count; i++)
                                    expanded.Add(inputTokens[i]);
                                inputTokens = expanded;

                                int insertPos = imageTokenPos + 1;
                                g4.SetVisionEmbeddings(visionEmbeddings, insertPos);
                                _log.LogInformation(LogEventIds.HostConfiguration,
                                    "Gemma4 vision frame {FrameIndex}: {VisionTokens} tokens at position {InsertPos}",
                                    imgIdx, numVisionTokens, insertPos);

                                searchFrom = imageTokenPos + 1 + numVisionTokens + 1;
                            }
                            else
                            {
                                _log.LogWarning(LogEventIds.HostConfiguration,
                                    "Gemma4 vision: no more <|image> tokens for frame {FrameIndex}", imgIdx);
                                visionEmbeddings.Dispose();
                            }
                        }
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Total tokens after Gemma4 image expansion: {TotalTokens}", inputTokens.Count);
                    }
                    else if (imagePaths.Count > 0)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration,
                            "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                    }
                }
                else if (arch == "nemotron_h_omni" || arch == "nemotron_h" || arch == "nemotron_h_moe")
                {
                    int imageTokenId = model.Tokenizer.LookupToken("<image>");
                    int imageStartId = model.Tokenizer.LookupToken("<img>");
                    int imageEndId = model.Tokenizer.LookupToken("</img>");
                    if (imageTokenId < 0) imageTokenId = 18;
                    if (imageStartId < 0) imageStartId = 19;
                    if (imageEndId < 0) imageEndId = 20;

                    if (model is NemotronModel nem && nem.VisionEncoder != null)
                    {
                        var allEmbeddings = new List<TensorSharp.Tensor>();
                        var perImageTileCounts = new List<int[]>();

                        foreach (var imgP in imagePaths)
                        {
                            // VIDEO_MAX_FRAMES already handled upstream by extracting frames.
                            var tiles = nem.ImageProcessor.ProcessImage(imgP);
                            var tileTokens = new int[tiles.Count];
                            var tileEmbeddings = new TensorSharp.Tensor[tiles.Count];

                            for (int t = 0; t < tiles.Count; t++)
                            {
                                var tile = tiles[t];
                                var emb = nem.VisionEncoder.Encode(tile.Pixels, tile.Width, tile.Height);
                                tileEmbeddings[t] = emb;
                                tileTokens[t] = (int)emb.Sizes[0];
                                _log.LogInformation(LogEventIds.HostConfiguration,
                                    "Nemotron vision tile {Index}: source={Width}x{Height} embeddings={EmbeddingShape}",
                                    t, tile.Width, tile.Height, $"{emb.Sizes[0]}x{emb.Sizes[1]}");
                            }

                            perImageTileCounts.Add(tileTokens);
                            allEmbeddings.AddRange(tileEmbeddings);
                        }

                        // Each <image> token in the prompt corresponds to ONE image (chat
                        // template inserts a single <image> per visual). Replace each
                        // <image> with: <img> + (sum of tile token counts) image tokens
                        // + </img>, and queue the embeddings for in-place injection.
                        int searchFrom = 0;
                        int injectedImages = 0;
                        for (int imgIdx = 0; imgIdx < imagePaths.Count; imgIdx++)
                        {
                            int imageTokenPos = -1;
                            for (int i = searchFrom; i < inputTokens.Count; i++)
                            {
                                if (inputTokens[i] == imageTokenId)
                                {
                                    imageTokenPos = i;
                                    break;
                                }
                            }
                            if (imageTokenPos < 0)
                            {
                                _log.LogWarning(LogEventIds.HostConfiguration,
                                    "Nemotron vision: no more <image> tokens for image {Index}", imgIdx);
                                break;
                            }

                            int totalImageTokens = 0;
                            foreach (int t in perImageTileCounts[imgIdx]) totalImageTokens += t;

                            var expanded = new List<int>(inputTokens.Count + totalImageTokens + 2);
                            for (int i = 0; i < imageTokenPos; i++) expanded.Add(inputTokens[i]);
                            expanded.Add(imageStartId);
                            int injectStart = expanded.Count;
                            for (int i = 0; i < totalImageTokens; i++) expanded.Add(imageTokenId);
                            expanded.Add(imageEndId);
                            for (int i = imageTokenPos + 1; i < inputTokens.Count; i++) expanded.Add(inputTokens[i]);
                            inputTokens = expanded;

                            // Each tile's embeddings get injected at consecutive offsets.
                            int tileOffset = injectStart;
                            int tileBase = injectedImages;
                            for (int t = 0; t < perImageTileCounts[imgIdx].Length; t++)
                            {
                                nem.SetVisionEmbeddings(allEmbeddings[tileBase + t], tileOffset);
                                tileOffset += perImageTileCounts[imgIdx][t];
                            }
                            injectedImages += perImageTileCounts[imgIdx].Length;

                            searchFrom = injectStart + totalImageTokens + 1;
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Nemotron vision image {Index}: tiles={Tiles} totalTokens={Tokens} insertPos={Pos}",
                                imgIdx, perImageTileCounts[imgIdx].Length, totalImageTokens, injectStart);
                        }
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Total tokens after Nemotron image expansion: {TotalTokens}", inputTokens.Count);
                    }
                    else if (imagePaths.Count > 0)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration,
                            "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                    }
                }
                else if (arch == "mistral3")
                {
                    if (model is Mistral3Model m3 && m3.VisionEncoder != null)
                    {
                        var proc = new Mistral3ImageProcessor(
                            m3.VisionEncoder.ImageSize,
                            m3.VisionEncoder.PatchSize);

                        int imgTokenId = Mistral3ImageProcessor.ImgTokenId;
                        int imgBreakId = Mistral3ImageProcessor.ImgBreakTokenId;
                        int imgEndId = Mistral3ImageProcessor.ImgEndTokenId;

                        foreach (var imgP in imagePaths)
                        {
                            var (pixels, imgW, imgH) = proc.ProcessImage(imgP);
                            var visionEmb = m3.VisionEncoder.Encode(pixels, imgW, imgH);
                            int numRows = imgH / m3.VisionEncoder.PatchSize / m3.VisionEncoder.SpatialMergeSize;
                            int numCols = imgW / m3.VisionEncoder.PatchSize / m3.VisionEncoder.SpatialMergeSize;

                            int tokenPosition = -1;
                            for (int i = 0; i < inputTokens.Count; i++)
                            {
                                if (inputTokens[i] == imgTokenId)
                                {
                                    tokenPosition = i;
                                    break;
                                }
                            }

                            if (tokenPosition >= 0)
                            {
                                var expanded = new List<int>();
                                for (int i = 0; i < tokenPosition; i++)
                                    expanded.Add(inputTokens[i]);

                                for (int row = 0; row < numRows; row++)
                                {
                                    for (int col = 0; col < numCols; col++)
                                        expanded.Add(imgTokenId);
                                    expanded.Add(row == numRows - 1 ? imgEndId : imgBreakId);
                                }

                                for (int i = tokenPosition + 1; i < inputTokens.Count; i++)
                                    expanded.Add(inputTokens[i]);

                                m3.SetVisionEmbeddings(visionEmb, tokenPosition);
                                inputTokens = expanded;
                                _log.LogInformation(LogEventIds.HostConfiguration,
                                    "Mistral3 vision: rows={Rows} cols={Cols} totalTokens={TotalTokens} position={Position}",
                                    numRows, numCols, numRows * numCols + numRows, tokenPosition);
                            }
                            else
                            {
                                visionEmb.Dispose();
                                _log.LogWarning(LogEventIds.HostConfiguration,
                                    "Mistral3 vision: no [IMG] token found in prompt");
                            }
                        }
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Total tokens after Mistral3 image expansion: {TotalTokens}", inputTokens.Count);
                    }
                    else
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration,
                            "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                    }
                }
                else
                {
                    int imagePadId = model.Tokenizer.LookupToken("<|image_pad|>");
                    if (imagePadId < 0)
                    {
                        _log.LogWarning(LogEventIds.HostConfiguration,
                            "<|image_pad|> token not found in vocabulary");
                    }
                    else
                    {
                        int patchSize = 14;
                        int mergeSize = 2;
                        if (model is Qwen35Model q35m && q35m.VisionEncoder != null)
                        {
                            patchSize = q35m.VisionEncoder.PatchSize;
                            mergeSize = q35m.VisionEncoder.SpatialMergeSize;
                        }
                        var processor = new Qwen35ImageProcessor(patchSize, mergeSize);

                        var tokenCounts = new int[imagePaths.Count];
                        for (int i = 0; i < imagePaths.Count; i++)
                        {
                            var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(imagePaths[i]);
                            tokenCounts[i] = processor.ComputeImageTokenCount(height, width);
                            var (gridH, gridW) = processor.GetPatchGrid(height, width);
                            var (resizedH, resizedW) = processor.SmartResize(height, width);
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Image {Index}: source={Source}x{SourceH} resized={ResizedW}x{ResizedH} grid={GridW}x{GridH} visionTokens={VisionTokens} merged={MergedW}x{MergedH}",
                                i, width, height, resizedW, resizedH, gridW, gridH, tokenCounts[i],
                                gridW / processor.MergeSize, gridH / processor.MergeSize);
                        }

                        inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

                        int visionStartId = model.Tokenizer.LookupToken("<|vision_start|>");
                        int visionEndId = model.Tokenizer.LookupToken("<|vision_end|>");
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Vision token IDs: start={Start} pad={Pad} end={End}",
                            visionStartId, imagePadId, visionEndId);

                        if (model is Qwen35Model q35 && q35.VisionEncoder != null)
                        {
                            var (pixels, resH, resW) = processor.ProcessImage(imagePaths[0]);
                            var visionEmbeddings = q35.VisionEncoder.Encode(pixels, resH, resW);
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Qwen3.5 vision embeddings: resolution={Width}x{Height} shape={EmbeddingShape}",
                                resW, resH, $"{visionEmbeddings.Sizes[0]}x{visionEmbeddings.Sizes[1]}");

                            int imageTokenStart = -1;
                            for (int i = 0; i < inputTokens.Count; i++)
                            {
                                if (inputTokens[i] == imagePadId)
                                {
                                    imageTokenStart = i;
                                    break;
                                }
                            }

                            if (imageTokenStart >= 0)
                            {
                                q35.SetVisionEmbeddings(visionEmbeddings, imageTokenStart);
                                _log.LogInformation(LogEventIds.HostConfiguration,
                                    "Qwen3.5 vision embeddings injection position={Position}", imageTokenStart);
                            }
                            else
                            {
                                _log.LogWarning(LogEventIds.HostConfiguration,
                                    "Qwen3.5 vision: could not find image placeholder position");
                                visionEmbeddings.Dispose();
                            }
                        }
                        else if (!model.HasVisionEncoder())
                        {
                            _log.LogWarning(LogEventIds.HostConfiguration,
                                "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                        }
                    }
                }
            }

            // Audio processing for Nemotron Omni: requires an audio mmproj to be loaded.
            // The current GGUF distribution we tested with does not embed the parakeet
            // weights; in that case we still preprocess the audio to verify the pipeline
            // and warn the user that no inference will run on the embeddings.
            if (audioPaths != null && audioPaths.Count > 0 &&
                (model.Config.Architecture == "nemotron_h_omni" ||
                 model.Config.Architecture == "nemotron_h" ||
                 model.Config.Architecture == "nemotron_h_moe"))
            {
                try
                {
                    float[] samples = NemotronAudioPreprocessor.DecodeAudioFile(audioPaths[0]);
                    var (mel, frames, validFrames) = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(samples);
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Nemotron audio decoded: durationSec={DurationSec:F1} frames={Frames} validFrames={ValidFrames} melShape={Bins}x{Frames2}",
                        (double)samples.Length / NemotronAudioPreprocessor.SampleRate, frames, validFrames,
                        NemotronAudioPreprocessor.MelBins, frames);

                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "Nemotron audio inference is NOT supported by this mmproj (no Parakeet weights present). " +
                        "The audio file was preprocessed for verification only. Pass an audio mmproj to enable real inference.");
                }
                catch (Exception ex)
                {
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "Nemotron audio preprocessing failed: {Error}", ex.Message);
                }
            }

            // Audio processing for Gemma4
            if (audioPaths != null && audioPaths.Count > 0 && model.Config.Architecture == "gemma4")
            {
                int audioStartId = model.Tokenizer.LookupToken("<|audio>");
                int audioEndId = model.Tokenizer.LookupToken("<audio|>");

                if (model is Gemma4Model g4a && g4a.AudioEncoder != null)
                {
                    float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(audioPaths[0]);
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Audio decoded: samples={Samples} durationSec={DurationSec:F1}",
                        samples.Length, (double)samples.Length / 16000);

                    if (samples.Length % 128 != 0)
                    {
                        int padded = samples.Length + (128 - samples.Length % 128);
                        Array.Resize(ref samples, padded);
                    }

                    var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Mel spectrogram computed: frames={Frames}", numFrames);

                    if (melData != null && numFrames > 0)
                    {
                        var audioEmbeddings = g4a.AudioEncoder.Encode(melData, numFrames);
                        int numAudioTokens = (int)audioEmbeddings.Sizes[0];
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Audio embeddings shape={EmbeddingShape}",
                            $"{audioEmbeddings.Sizes[0]}x{audioEmbeddings.Sizes[1]}");

                        int audioTokenPos = -1;
                        for (int i = 0; i < inputTokens.Count; i++)
                        {
                            if (inputTokens[i] == audioStartId)
                            {
                                audioTokenPos = i;
                                break;
                            }
                        }

                        if (audioTokenPos >= 0)
                        {
                            var expanded = new List<int>();
                            for (int i = 0; i < audioTokenPos; i++)
                                expanded.Add(inputTokens[i]);
                            expanded.Add(audioStartId);
                            for (int i = 0; i < numAudioTokens; i++)
                                expanded.Add(0);
                            expanded.Add(audioEndId);
                            for (int i = audioTokenPos + 1; i < inputTokens.Count; i++)
                                expanded.Add(inputTokens[i]);
                            inputTokens = expanded;

                            int insertPos = audioTokenPos + 1;
                            g4a.SetAudioEmbeddings(audioEmbeddings, insertPos);
                            _log.LogInformation(LogEventIds.HostConfiguration,
                                "Gemma4 audio: tokens={Tokens} position={Position} totalTokensAfter={TotalTokens}",
                                numAudioTokens, insertPos, inputTokens.Count);
                        }
                        else
                        {
                            _log.LogWarning(LogEventIds.HostConfiguration,
                                "Gemma4 audio: could not find <|audio> token in prompt");
                            audioEmbeddings.Dispose();
                        }
                    }
                }
                else
                {
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "No audio encoder loaded. Use --mmproj to specify the multimodal GGUF.");
                }
            }

            _log.LogInformation(LogEventIds.ChatStarted,
                "cli.inference inputTokens={InputTokens} preview=[{First30}{TruncationSuffix}]",
                inputTokens.Count,
                string.Join(", ", inputTokens.Take(30)),
                inputTokens.Count > 30 ? $"... ({inputTokens.Count} total)" : string.Empty);

            model.ResetKVCache();

            bool tokenByToken = Environment.GetEnvironmentVariable("TOKEN_BY_TOKEN") == "1";
            float[] logits;
            var prefillSw = Stopwatch.StartNew();
            if (tokenByToken)
            {
                _log.LogInformation(LogEventIds.HostConfiguration, "TOKEN_BY_TOKEN prefill mode enabled");
                logits = null;
                for (int i = 0; i < inputTokens.Count; i++)
                    logits = model.Forward(new[] { inputTokens[i] });
            }
            else
            {
                logits = model.Forward(inputTokens.ToArray());
            }
            prefillSw.Stop();
            double prefillMs = prefillSw.Elapsed.TotalMilliseconds;
            double prefillTps = inputTokens.Count > 0 && prefillMs > 0
                ? inputTokens.Count / (prefillMs / 1000.0)
                : 0.0;
            if (!silent)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "cli.inference prefill complete: tokens={Tokens} ms={Ms:F1} tokensPerSec={Tps:F1}",
                    inputTokens.Count, prefillMs, prefillTps);
            }
            var generatedTokens = new List<int>();

            LogTopLogits(logits, model, "prefill");

            var cfg = samplingConfig ?? SamplingConfig.Greedy;
            var sampler = new TokenSampler(cfg);

            if (!cfg.IsGreedy)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Sampling config: temperature={Temperature} topK={TopK} topP={TopP} minP={MinP} repPen={RepPenalty} presPen={PresPenalty} freqPen={FreqPenalty} seed={Seed}",
                    cfg.Temperature, cfg.TopK, cfg.TopP, cfg.MinP, cfg.RepetitionPenalty,
                    cfg.PresencePenalty, cfg.FrequencyPenalty, cfg.Seed);
            }

            var parser = OutputParserFactory.Create(model.Config.Architecture);
            parser.Init(enableThinking, tools);
            bool useParser = enableThinking || (tools != null && tools.Count > 0) || parser.AlwaysRequired;
            bool showThinking = enableThinking || parser.AlwaysRequired;
            if (useParser)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Output parser={Parser} thinking={Thinking} tools={ToolCount}",
                    parser.GetType().Name, enableThinking, tools?.Count ?? 0);
            }

            string finishReason = "max_tokens";
            var decodeSw = Stopwatch.StartNew();

            // Pipelined greedy decode: when the model supports a device-side
            // argmax + embedding lookup, queue forward N+1 BEFORE syncing
            // forward N's predicted token. This overlaps the LM-head host
            // sync wait with the next forward's first kernels.
            // Used when:
            //   - greedy sampling (no top-K / temperature)
            //   - no stop sequences (the pipeline issues one extra forward
            //     that we'd waste on a mid-stream stop)
            //   - model.SupportsPipelinedGreedy
            // Default ON; opt-out with TS_MLX_PIPELINED_DECODE=0.
            string pipelinedEnv = Environment.GetEnvironmentVariable("TS_MLX_PIPELINED_DECODE");
            bool pipelinedDecodeEnabled =
                !string.Equals(pipelinedEnv, "0", StringComparison.Ordinal)
                && !string.Equals(pipelinedEnv, "false", StringComparison.OrdinalIgnoreCase);
            bool pipelinedGreedy = cfg.IsGreedy
                && (cfg.StopSequences == null || cfg.StopSequences.Count == 0)
                && model.SupportsPipelinedGreedy
                && pipelinedDecodeEnabled;

            if (pipelinedGreedy)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "cli.inference using pipelined greedy decode (TS_MLX_PIPELINED_DECODE=1)");

                // Bootstrap: sample the FIRST decode token from prefill logits.
                int firstToken = sampler.Sample(logits, generatedTokens);
                if (model.Tokenizer.IsEos(firstToken))
                {
                    finishReason = "eos";
                }
                else
                {
                    generatedTokens.Add(firstToken);

                    // Submit decode step that will predict the SECOND decode
                    // token. Returns a [1] int32 device tensor we'll sync later.
                    Tensor pending = model.SubmitGreedyDecodeStep(firstToken);

                    int step = 1;
                    for (; step < maxTokens; step++)
                    {
                        // Issue the NEXT forward (using cached device embedding).
                        // Its argmax + next-embedding queueing runs on GPU while
                        // we host-wait on `pending` below.
                        Tensor next = model.SubmitGreedyDecodeStep(null);

                        // Sync the previously submitted prediction.
                        int tok = pending.GetElementsAsInt(1)[0];
                        pending.Dispose();
                        pending = next;

                        if (model.Tokenizer.IsEos(tok))
                        {
                            pending.Dispose();
                            pending = null;
                            finishReason = "eos";
                            break;
                        }
                        generatedTokens.Add(tok);
                    }

                    if (pending != null)
                    {
                        // Drain the last queued forward; if non-EOS and we still
                        // have room, emit it as the final token.
                        int tok = pending.GetElementsAsInt(1)[0];
                        pending.Dispose();
                        if (step <= maxTokens && !model.Tokenizer.IsEos(tok))
                        {
                            generatedTokens.Add(tok);
                        }
                        else if (model.Tokenizer.IsEos(tok))
                        {
                            finishReason = "eos";
                        }
                    }
                    model.ResetPipelinedGreedyState();
                }
            }
            else
            {
                for (int step = 0; step < maxTokens; step++)
                {
                    int nextToken = sampler.Sample(logits, generatedTokens);
                    _log.LogTrace(LogEventIds.GenerationProgress,
                        "step={Step} token={TokenId} text={TokenText}",
                        step, nextToken, model.Tokenizer.Vocab[nextToken]);

                    if (model.Tokenizer.IsEos(nextToken))
                    {
                        finishReason = "eos";
                        break;
                    }

                    generatedTokens.Add(nextToken);

                    if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                    {
                        string partial = model.Tokenizer.Decode(generatedTokens);
                        var (trimmed, shouldStop) = sampler.CheckStopSequences(partial);
                        if (shouldStop)
                        {
                            decodeSw.Stop();
                            double sdMs = decodeSw.Elapsed.TotalMilliseconds;
                            double sdTps = generatedTokens.Count > 0 && sdMs > 0
                                ? generatedTokens.Count / (sdMs / 1000.0)
                                : 0.0;
                            if (!silent)
                            {
                                _log.LogInformation(LogEventIds.CliBenchmark,
                                    "cli.inference decode complete: tokens={Tokens} ms={Ms:F1} tokensPerSec={Tps:F1}",
                                    generatedTokens.Count, sdMs, sdTps);
                            }
                            finishReason = "stop_sequence";
                            _log.LogInformation(LogEventIds.ChatCompleted,
                                "cli.inference finishReason={FinishReason} tokens={Tokens}",
                                finishReason, generatedTokens.Count);
                            if (useParser)
                            {
                                var finalParsed = parser.Add(trimmed, true);
                                return FormatParsedResult(finalParsed, showThinking);
                            }
                            return trimmed;
                        }
                    }

                    logits = model.Forward(new[] { nextToken });
                    if (step < 3)
                        LogTopLogits(logits, model, $"decode_{step}");
                }
            }
            decodeSw.Stop();
            double decodeMs = decodeSw.Elapsed.TotalMilliseconds;
            double decodeTps = generatedTokens.Count > 0 && decodeMs > 0
                ? generatedTokens.Count / (decodeMs / 1000.0)
                : 0.0;
            if (!silent)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "cli.inference decode complete: tokens={Tokens} ms={Ms:F1} tokensPerSec={Tps:F1}",
                    generatedTokens.Count, decodeMs, decodeTps);

                _log.LogInformation(LogEventIds.ChatCompleted,
                    "cli.inference finishReason={FinishReason} tokens={Tokens}",
                    finishReason, generatedTokens.Count);
                model.PrintTimingStats();
            }
            string decoded = model.Tokenizer.Decode(generatedTokens);

            if (useParser)
            {
                var parsed = parser.Add(decoded, true);
                return FormatParsedResult(parsed, showThinking);
            }
            return decoded;
        }

        // Per-turn upload manifest used by cli.inference.start. Each entry records
        // the saved file path, the saved filename (the path's leaf, which is the
        // unique identifier inside the upload directory) and the kind of media. The
        // shape mirrors TensorSharp.Server.ModelService.SerializeUploadsForLog so
        // operators see a uniform format whether they're inspecting CLI or server
        // logs.
        static string FormatUploadsForCli(List<string> imagePaths, List<string> audioPaths, string videoPath)
        {
            var entries = new List<object>();

            // When a video is supplied, imagePaths actually holds the per-frame images
            // extracted from the video. We tag those frames as "video_frame" and add a
            // separate "video" entry pointing at the original media file so the audit
            // trail captures both the source upload and its frame decomposition.
            bool isVideo = !string.IsNullOrEmpty(videoPath);
            if (isVideo)
            {
                entries.Add(new
                {
                    path = videoPath,
                    name = Path.GetFileName(videoPath),
                    mediaType = "video",
                });
            }

            string imageMediaType = isVideo ? "video_frame" : "image";
            AppendCliUploads(entries, imagePaths, imageMediaType);
            AppendCliUploads(entries, audioPaths, "audio");

            return entries.Count == 0
                ? "[]"
                : JsonSerializer.Serialize(entries, _cliUploadJsonOptions);
        }

        private static void AppendCliUploads(List<object> sink, List<string> paths, string mediaType)
        {
            if (paths == null || paths.Count == 0)
                return;

            foreach (var p in paths)
            {
                if (string.IsNullOrEmpty(p))
                    continue;
                sink.Add(new { path = p, name = Path.GetFileName(p), mediaType });
            }
        }

        // Keeps non-ASCII filenames readable in the log instead of expanding to
        // \uXXXX escapes; control characters are still escaped by JsonSerializer.
        private static readonly JsonSerializerOptions _cliUploadJsonOptions = new()
        {
            DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
            Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping,
        };

        static string FormatParsedResult(ParsedOutput parsed, bool showThinking)
        {
            var sb = new StringBuilder();
            if (showThinking && !string.IsNullOrEmpty(parsed.Thinking))
            {
                sb.AppendLine("\n--- Thinking ---");
                sb.AppendLine(parsed.Thinking.Trim());
                sb.AppendLine("--- End Thinking ---\n");
            }
            if (!string.IsNullOrEmpty(parsed.Content))
            {
                sb.Append(parsed.Content);
            }
            if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
            {
                sb.AppendLine("\n--- Tool Calls ---");
                foreach (var tc in parsed.ToolCalls)
                {
                    sb.AppendLine($"  Function: {tc.Name}");
                    sb.AppendLine($"  Arguments: {JsonSerializer.Serialize(tc.Arguments, new JsonSerializerOptions { WriteIndented = true })}");
                    sb.AppendLine();
                }
                sb.AppendLine("--- End Tool Calls ---");
            }
            return sb.ToString();
        }

        static void LogTopLogits(float[] logits, ModelBase model, string label)
        {
            if (!_log.IsEnabled(LogLevel.Debug))
                return;

            var indexed = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
            var sb = new StringBuilder();
            foreach (var (v, i) in indexed)
                sb.Append($"{i}({model.Tokenizer.Vocab[i]})={v:F4} ");
            _log.LogDebug(LogEventIds.GenerationProgress,
                "topLogits[{Label}] {TopList}", label, sb.ToString().TrimEnd());
        }

        static void RunTests(ModelBase model, int maxTokens, string outputFile)
        {
            _log.LogInformation(LogEventIds.CliBenchmark, "Running verification tests");

            TestTokenizer(model);
            TestChatTemplate();
            TestInferenceWithOllamaComparison(model, maxTokens, outputFile);
        }

        /// <summary>
        /// Standalone inference benchmark: measures pure prefill and decode throughput
        /// without prompt rendering, sampling, or output formatting overhead. Reports the
        /// best (minimum-time) of `runs` independent runs to filter out warm-up artifacts.
        ///
        /// Optionally captures the first decode tokens after each prefill so the same
        /// benchmark can be run twice (e.g. with and without GDN_DISABLE_CHUNKED_PREFILL=1)
        /// and the outputs compared.
        /// </summary>
        static void RunBenchmark(ModelBase model, int prefillTokens, int decodeTokens, int runs, bool chunked = false)
        {
            _log.LogInformation(LogEventIds.CliBenchmark,
                "inference benchmark starting: prefillTokens={PrefillTokens} decodeTokens={DecodeTokens} runs={Runs} chunked={Chunked}",
                prefillTokens, decodeTokens, runs, chunked);

            // Build a synthetic prompt of `prefillTokens` tokens by repeating a stable token.
            // We pick a token id that's safely inside the vocab (not BOS/EOS/special).
            int vocab = model.Config.VocabSize;
            int basisToken = Math.Min(100, vocab - 1);
            int[] prefillIds = new int[prefillTokens];
            for (int i = 0; i < prefillTokens; i++)
                prefillIds[i] = basisToken + (i % 17);

            double bestPrefillMs = double.PositiveInfinity;
            double bestDecodeMs = double.PositiveInfinity;
            double bestPrefillTps = 0;
            double bestDecodeTps = 0;
            double avgPrefillTps = 0;
            double avgDecodeTps = 0;

            int[] firstRunDecodeTokens = null;
            int firstRunPrefillTopToken = -1;

            // Decode kernel selection. The benchmark is greedy by definition
            // (it argmaxes the prefill logits to seed the chain), so when the
            // model exposes the pipelined-greedy device path use it: each
            // decode step queues the next forward's input embedding from a
            // device-side argmax, eliminating the per-token MLX→CPU sync of
            // the full [vocab] logits tensor. Opt out with
            // TS_MLX_PIPELINED_DECODE=0 to force the legacy host-sync path
            // for A/B comparison.
            string pipelinedEnv = Environment.GetEnvironmentVariable("TS_MLX_PIPELINED_DECODE");
            bool usePipelinedGreedy = model.SupportsPipelinedGreedy
                && !string.Equals(pipelinedEnv, "0", StringComparison.Ordinal)
                && !string.Equals(pipelinedEnv, "false", StringComparison.OrdinalIgnoreCase);

            for (int run = 0; run < runs; run++)
            {
                model.ResetKVCache();

                // Prefill timing - choose path based on chunked flag.
                // Forward(): single non-chunked pass, used by --benchmark default.
                // ForwardRefill(): the path the server uses for long prompts; engages
                // the model's internal chunked prefill (with rolling SWA cache, etc.).
                var prefillSw = Stopwatch.StartNew();
                float[] logits = chunked
                    ? model.ForwardRefill(prefillIds)
                    : model.Forward(prefillIds);
                prefillSw.Stop();
                double prefillMs = prefillSw.Elapsed.TotalMilliseconds;
                double prefillTps = prefillTokens / (prefillMs / 1000.0);

                // Decode timing - greedy sampling on a stable token chain
                int next = SampleGreedyFromLogits(logits, vocab);
                if (run == 0)
                {
                    firstRunPrefillTopToken = next;
                    firstRunDecodeTokens = new int[decodeTokens];
                }
                var decodeSw = Stopwatch.StartNew();
                if (usePipelinedGreedy && decodeTokens > 0)
                {
                    // Submit decode step N+1 BEFORE host-syncing token N — overlaps
                    // the MLX LM-head sync wait with the next forward's first
                    // kernels. `pending` is a [1] int32 device tensor.
                    Tensor pending = model.SubmitGreedyDecodeStep(next);
                    int step = 1;
                    for (; step < decodeTokens; step++)
                    {
                        Tensor nextDevice = model.SubmitGreedyDecodeStep(null);
                        int tok = pending.GetElementsAsInt(1)[0];
                        pending.Dispose();
                        pending = nextDevice;
                        if (run == 0)
                            firstRunDecodeTokens[step - 1] = tok;
                    }
                    // Drain the last queued forward.
                    int lastTok = pending.GetElementsAsInt(1)[0];
                    pending.Dispose();
                    if (run == 0)
                        firstRunDecodeTokens[decodeTokens - 1] = lastTok;
                    model.ResetPipelinedGreedyState();
                }
                else
                {
                    for (int i = 0; i < decodeTokens; i++)
                    {
                        logits = model.Forward(new[] { next });
                        next = SampleGreedyFromLogits(logits, vocab);
                        if (run == 0)
                            firstRunDecodeTokens[i] = next;
                    }
                }
                decodeSw.Stop();
                double decodeMs = decodeSw.Elapsed.TotalMilliseconds;
                double decodeTps = decodeTokens / (decodeMs / 1000.0);

                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark run {Run}/{Runs}: prefillMs={PrefillMs:F0} prefillTps={PrefillTps:F1} decodeMs={DecodeMs:F0} decodeTps={DecodeTps:F1} msPerTok={MsPerTok:F1}",
                    run + 1, runs, prefillMs, prefillTps, decodeMs, decodeTps, decodeMs / decodeTokens);

                if (prefillMs < bestPrefillMs)
                {
                    bestPrefillMs = prefillMs;
                    bestPrefillTps = prefillTps;
                }
                if (decodeMs < bestDecodeMs)
                {
                    bestDecodeMs = decodeMs;
                    bestDecodeTps = decodeTps;
                }
                avgPrefillTps += prefillTps;
                avgDecodeTps += decodeTps;
            }

            avgPrefillTps /= runs;
            avgDecodeTps /= runs;

            _log.LogInformation(LogEventIds.CliBenchmark,
                "benchmark summary: bestPrefillMs={BestPrefillMs:F0} bestPrefillTps={BestPrefillTps:F1} bestDecodeMs={BestDecodeMs:F0} bestDecodeTps={BestDecodeTps:F1} bestDecodeMsPerTok={BestDecodeMsPerTok:F2} avgPrefillTps={AvgPrefillTps:F1} avgDecodeTps={AvgDecodeTps:F1}",
                bestPrefillMs, bestPrefillTps, bestDecodeMs, bestDecodeTps,
                bestDecodeMs / decodeTokens, avgPrefillTps, avgDecodeTps);

            if (firstRunDecodeTokens != null)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark sampled tokens (run1): prefillTopToken={Prefill} decode={Decode}",
                    firstRunPrefillTopToken, string.Join(",", firstRunDecodeTokens));
            }
            model.PrintTimingStats();
        }

        static int SampleGreedyFromLogits(float[] logits, int vocab)
        {
            int idx = 0;
            float best = float.NegativeInfinity;
            int n = Math.Min(vocab, logits.Length);
            for (int i = 0; i < n; i++)
            {
                if (logits[i] > best)
                {
                    best = logits[i];
                    idx = i;
                }
            }
            return idx;
        }

        /// <summary>
        /// Correctness check for chunked prefill: prefill the same prompt twice -
        /// once via Forward() (single-pass), once via ForwardRefill() (chunked) -
        /// and compare the next decoded tokens. They must match exactly: any
        /// divergence indicates that chunked prefill (rolling SWA cache, position
        /// handling, sparse window) is not bit-equivalent to non-chunked.
        ///
        /// Useful when iterating on the chunked attention path - quickly catches
        /// regressions like missing previous-window tokens for SWA layers.
        /// </summary>
        static void RunChunkedPrefillCorrectness(ModelBase model, int prefillTokens, int decodeTokens)
        {
            _log.LogInformation(LogEventIds.CliBenchmark,
                "chunked prefill correctness test: prefillTokens={PrefillTokens} decodeTokens={DecodeTokens}",
                prefillTokens, decodeTokens);

            int vocab = model.Config.VocabSize;
            int basisToken = Math.Min(100, vocab - 1);
            int[] prefillIds = new int[prefillTokens];
            for (int i = 0; i < prefillTokens; i++)
                prefillIds[i] = basisToken + (i % 17);

            int[] DecodeWith(System.Func<int[], float[]> prefill)
            {
                model.ResetKVCache();
                var sw = Stopwatch.StartNew();
                float[] logits = prefill(prefillIds);
                sw.Stop();
                int firstSampled = SampleGreedyFromLogits(logits, vocab);
                var decoded = new int[decodeTokens];
                int next = firstSampled;
                var decodeSw = Stopwatch.StartNew();
                for (int i = 0; i < decodeTokens; i++)
                {
                    decoded[i] = next;
                    logits = model.Forward(new[] { next });
                    next = SampleGreedyFromLogits(logits, vocab);
                }
                decodeSw.Stop();
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "  prefillMs={PrefillMs:F0} decodeMs={DecodeMs:F0} firstSampled={First} decoded=[{Decoded}]",
                    sw.Elapsed.TotalMilliseconds, decodeSw.Elapsed.TotalMilliseconds, firstSampled,
                    string.Join(", ", decoded));
                return decoded;
            }

            _log.LogInformation(LogEventIds.CliBenchmark, "Pass 1: Forward (non-chunked)");
            int[] tokensForward = DecodeWith(t => model.Forward(t));

            _log.LogInformation(LogEventIds.CliBenchmark, "Pass 2: ForwardRefill (chunked)");
            int[] tokensRefill = DecodeWith(t => model.ForwardRefill(t));

            int matches = 0;
            for (int i = 0; i < decodeTokens; i++)
                if (tokensForward[i] == tokensRefill[i]) matches++;
                else break;

            bool pass = matches == decodeTokens;
            _log.LogInformation(LogEventIds.CliBenchmark,
                "chunked prefill correctness: matched={Matched}/{Total} {Status} forward=[{F}] refill=[{R}]",
                matches, decodeTokens, pass ? "PASS" : "FAIL",
                string.Join(",", tokensForward), string.Join(",", tokensRefill));
        }

        /// <summary>
        /// Multi-turn first-token-latency benchmark.
        ///
        /// Simulates a conversation of <paramref name="turns"/> user turns. Each turn
        /// generates <paramref name="maxTokens"/> tokens. We measure the prefill time of
        /// each turn under TWO modes back-to-back, on the SAME model and conversation:
        ///
        ///   1. With KV cache reuse (the new behavior): tokens from prior turns are
        ///      kept in the KV cache and only the new (user + generation-prompt + previous
        ///      assistant raw tokens) suffix is forwarded.
        ///   2. Without KV cache reuse: the model's KV cache is fully reset between
        ///      turns and the entire prompt is re-prefilled.
        ///
        /// The interesting metric is the prefill latency PER TURN, since that's what the
        /// user feels as "time to first token". KV cache reuse should bring the per-turn
        /// prefill from O(prompt_so_far) down to O(new_user_message).
        /// </summary>
        static void RunKvCacheBenchmark(ModelBase model, int turns, int maxTokens,
            SamplingConfig sampling, bool enableThinking)
        {
            if (turns < 2)
                turns = 2;

            string arch = model.Config.Architecture;
            _log.LogInformation(LogEventIds.CliBenchmark,
                "kv cache benchmark starting: turns={Turns} decodeBudget={MaxTokens} architecture={Architecture}",
                turns, maxTokens, arch);

            // The user turns are designed so that early turns establish a fairly long
            // running context, then later turns add small follow-up questions. This is
            // the regime where KV cache reuse pays off the most.
            string[] userTurns = new[]
            {
                "Please write a detailed paragraph about the history and evolution of artificial intelligence, covering symbolic AI, expert systems, machine learning and the deep learning revolution.",
                "Could you summarize that into three short bullet points?",
                "Now translate the bullet points into French.",
                "Add one more bullet point about the role of neural networks.",
                "Translate the new bullet point into Spanish.",
                "What was the first bullet point again?",
                "Combine the first two bullet points into one sentence.",
                "Explain what an LLM is in one sentence.",
            };

            int turnLimit = Math.Min(turns, userTurns.Length);

            var samplerCfg = sampling ?? SamplingConfig.Greedy;

            (double[] cached, int[] promptTokensCached) = RunBenchmarkPass(model, arch, userTurns, turnLimit, maxTokens, samplerCfg, enableThinking, useCache: true);
            (double[] noCache, int[] _) = RunBenchmarkPass(model, arch, userTurns, turnLimit, maxTokens, samplerCfg, enableThinking, useCache: false);

            for (int i = 0; i < turnLimit; i++)
            {
                double speedup = cached[i] > 0 ? noCache[i] / cached[i] : 0;
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "kv benchmark turn {Turn}: promptTokens={PromptTokens} withKvMs={WithKvMs:F1} noKvMs={NoKvMs:F1} speedup={Speedup:F2}",
                    i + 1, promptTokensCached[i], cached[i], noCache[i], speedup);
            }

            // Skip turn 1 in the aggregate because both paths do an unavoidable full
            // prefill on the very first turn (no cache to reuse).
            if (turnLimit >= 2)
            {
                double cachedSum = 0;
                double noCacheSum = 0;
                int counted = 0;
                for (int i = 1; i < turnLimit; i++)
                {
                    cachedSum += cached[i];
                    noCacheSum += noCache[i];
                    counted++;
                }
                double avgCached = cachedSum / counted;
                double avgNoCache = noCacheSum / counted;
                double avgSpeedup = avgCached > 0 ? avgNoCache / avgCached : 0;
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "kv benchmark average prefill (turns 2..{TurnLimit}): withKvMs={AvgCached:F1} noKvMs={AvgNoCache:F1} speedup={AvgSpeedup:F2}",
                    turnLimit, avgCached, avgNoCache, avgSpeedup);
            }
        }

        /// <summary>
        /// Run a single benchmark pass through <paramref name="userTurns"/>. Returns the
        /// per-turn prefill latency in milliseconds (one entry per turn) and the per-turn
        /// prompt token counts.
        /// </summary>
        static (double[] prefillMs, int[] promptTokens) RunBenchmarkPass(
            ModelBase model, string arch, string[] userTurns, int turnLimit, int maxTokens,
            SamplingConfig samplerCfg, bool enableThinking, bool useCache)
        {
            _log.LogInformation(LogEventIds.CliBenchmark,
                "benchmark pass: kvCache={KvCacheEnabled}",
                useCache ? "enabled" : "disabled");

            model.ResetKVCache();
            var kvCache = new KVCache();
            var renderer = new KVCachePromptRenderer(PromptRenderer);

            var history = new List<ChatMessage>();
            double[] prefillMs = new double[turnLimit];
            int[] promptTokens = new int[turnLimit];

            for (int turn = 0; turn < turnLimit; turn++)
            {
                history.Add(new ChatMessage { Role = "user", Content = userTurns[turn] });

                // Always render with raw token splicing so the cached path can match.
                var inputTokens = renderer.RenderToTokens(
                    model.Tokenizer,
                    model.Config.ChatTemplate,
                    history,
                    arch,
                    addGenerationPrompt: true,
                    enableThinking: enableThinking);

                promptTokens[turn] = inputTokens.Count;

                if (!useCache)
                {
                    model.ResetKVCache();
                    kvCache.Reset();
                }

                var sw = Stopwatch.StartNew();
                ReusePlan plan = kvCache.PlanReuse(inputTokens, model.SupportsKVCacheTruncation);
                float[] logits = ApplyReusePlan(model, kvCache, plan, inputTokens);
                prefillMs[turn] = sw.Elapsed.TotalMilliseconds;

                // Generate the assistant response so the cached path has realistic raw
                // tokens to splice in for subsequent turns. We use greedy sampling for
                // determinism / reproducibility.
                var sampler = new TokenSampler(samplerCfg);
                var generatedTokens = new List<int>();
                var sb = new StringBuilder();

                for (int step = 0; step < maxTokens; step++)
                {
                    int nextToken = sampler.Sample(logits, generatedTokens);
                    if (model.Tokenizer.IsEos(nextToken)) break;
                    generatedTokens.Add(nextToken);
                    sb.Append(model.Tokenizer.Decode(new List<int> { nextToken }));
                    logits = model.Forward(new[] { nextToken });
                    kvCache.RecordAppend(nextToken, logits);
                }

                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark turn {Turn}: promptTokens={PromptTokens} prefillMs={PrefillMs:F1} decodeTokens={DecodeTokens} plan={Plan}",
                    turn + 1, inputTokens.Count, prefillMs[turn], generatedTokens.Count, plan.Kind);

                // Append the assistant turn so subsequent renders include it.
                var parser = OutputParserFactory.Create(arch);
                parser.Init(enableThinking, null);
                var parsed = parser.Add(sb.ToString(), true);
                history.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = parsed.Content ?? "",
                    Thinking = parsed.Thinking ?? "",
                    RawOutputTokens = generatedTokens,
                });
            }

            return (prefillMs, promptTokens);
        }

        /// <summary>
        /// Paged KV-cache benchmark. Simulates a cross-session scenario: the first
        /// user pays the full prefill cost, then a second user arrives with the
        /// same prompt prefix. Without the paged cache the second user repays the
        /// full cost; with it, most of the prefill is recovered from RAM blocks.
        ///
        /// We measure both halves in-process so the comparison is apples-to-apples
        /// on the same warm-loaded weights. Memory is read from the manager itself
        /// (the bytes it has resident) and from <see cref="GC"/> (managed heap).
        /// </summary>
        static void RunPagedKvBenchmark(ModelBase model, int promptTokens, int trials)
        {
            if (!model.SupportsKVStateSnapshot)
            {
                _log.LogError(LogEventIds.CliBenchmark,
                    "paged-bench: model architecture '{Arch}' does not support KV snapshot. Use Qwen3, Gemma3, GptOss, or Mistral3.",
                    model.Config.Architecture);
                return;
            }
            if (trials <= 0) trials = 1;
            if (promptTokens <= 0) promptTokens = 2048;

            // Pick up --paged-kv-block-size / --paged-kv-ram-mb / --paged-kv-ssd-* (or
            // their env var equivalents) so the bench mirrors what the server
            // would do at runtime instead of using a hard-coded config.
            var envCfg = PagedKvCacheConfig.FromEnvironment();
            int blockSize = envCfg.BlockSize;
            int safeBase = Math.Max(0, 1);
            int vocab = Math.Max(safeBase + 2, model.Config.VocabSize);
            int[] prompt = new int[promptTokens];
            var rng = new Random(unchecked((int)0xCAFEBABE));
            for (int i = 0; i < promptTokens; i++)
                prompt[i] = rng.Next(safeBase, vocab - 1);

            _log.LogInformation(LogEventIds.CliBenchmark,
                "paged-bench starting: promptTokens={Prompt} trials={Trials} blockSize={BlockSize} arch={Arch} kvDtype={Dtype}",
                promptTokens, trials, blockSize, model.Config.Architecture, model.KvCacheDtype);

            // Warm up - resolves Metal JIT, allocator pools, etc., so the first
            // measurement isn't dominated by setup cost.
            model.ResetKVCache();
            model.ForwardRefill(prompt);
            model.ResetKVCache();

            long warmRss = WorkingSetBytes();

            // ===== Baseline: paged cache disabled =====
            var baselineMs = new double[trials];
            for (int t = 0; t < trials; t++)
            {
                model.ResetKVCache();
                var sw = Stopwatch.StartNew();
                model.ForwardRefill(prompt);
                sw.Stop();
                baselineMs[t] = sw.Elapsed.TotalMilliseconds;
                _log.LogDebug(LogEventIds.CliBenchmark,
                    "paged-bench baseline trial {Trial}: {Ms:F1} ms", t + 1, baselineMs[t]);
            }
            long rssAfterBaseline = WorkingSetBytes();

            // ===== With paged cache: prime the store, then measure restore =====
            // Start from the env-resolved config so CLI overrides
            // (--paged-kv-ram-mb, --paged-kv-ssd-dir, ...) carry through. We
            // force Enabled=true here because the bench is, by definition, a
            // measurement of the paged path - regardless of whether the user
            // happened to also pass --paged-kv.
            var pagedConfig = new PagedKvCacheConfig
            {
                Enabled = true,
                BlockSize = blockSize,
                MaxRamBytes = envCfg.MaxRamBytes > 0 ? envCfg.MaxRamBytes : 4L * 1024 * 1024 * 1024,
                SsdDirectory = envCfg.SsdDirectory,
                MaxSsdBytes = envCfg.MaxSsdBytes,
            };
            // Pick up the TurboQuant codec from TS_KV_PAGED_QUANT_BITS so the
            // benchmark exercises the same compression path the server uses.
            // FromEnvironment(model) returns null both when the env var is
            // unset and when the model has recurrent SSM state that
            // quantization would corrupt (Qwen3.5/3.6 GatedDeltaNet,
            // Nemotron Mamba2). See TurboQuantKvCodec docs for details.
            IKvBlockCodec pagedCodec = TurboQuantKvCodec.FromEnvironment(model);
            if (pagedCodec != null)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench codec={Codec} (bitsPerElement={Bits}, kvDtype={Dtype})",
                    pagedCodec.Name, pagedCodec.BitsPerElement, model.KVStateElementType);
            }
            else if (model.RequiresPerBlockCapture &&
                     !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS")))
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench codec=passthrough (model {Arch} has RequiresPerBlockCapture=true; TurboQuant disabled to protect recurrent state)",
                    model.Config.Architecture);
            }
            var pagedManager = new PagedKvCacheManager(pagedConfig, model.KVStateFingerprint,
                Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance, pagedCodec);
            try
            {
                // Prime - what the FIRST user pays. Subsequent users get the speedup.
                // Recurrent models (RequiresPerBlockCapture=true) must be captured at
                // every block boundary because their running state isn't decomposable
                // into per-position slices; we chunk the prefill into block-sized
                // pieces here so each chunk leaves _cacheSeqLen on a block boundary
                // before Capture extracts the layer state.
                model.ResetKVCache();
                var primeSw = Stopwatch.StartNew();
                if (model.RequiresPerBlockCapture)
                {
                    for (int start = 0; start < promptTokens; start += blockSize)
                    {
                        int len = Math.Min(blockSize, promptTokens - start);
                        int[] chunk = new int[len];
                        Array.Copy(prompt, start, chunk, 0, len);
                        model.ForwardRefill(chunk);
                        pagedManager.Capture(model, prompt, start + len);
                    }
                }
                else
                {
                    model.ForwardRefill(prompt);
                    pagedManager.Capture(model, prompt, promptTokens);
                }
                primeSw.Stop();

                var pagedStats = pagedManager.GetStats();

                var pagedMs = new double[trials];
                int restoredTokens = 0;
                float[] postRestoreLogits = null;
                for (int t = 0; t < trials; t++)
                {
                    model.ResetKVCache();
                    var sw = Stopwatch.StartNew();
                    restoredTokens = pagedManager.TryRestorePrefix(model, prompt);
                    if (restoredTokens < promptTokens)
                    {
                        int[] suffix = new int[promptTokens - restoredTokens];
                        Array.Copy(prompt, restoredTokens, suffix, 0, suffix.Length);
                        postRestoreLogits = model.ForwardRefill(suffix);
                    }
                    else
                    {
                        // The restore alone left us no token to forward, so the
                        // model never produced fresh logits this trial. The
                        // manager keeps one trailing block specifically to
                        // avoid this; reaching here means a non-block-aligned
                        // prompt. Fall back to a single-token forward.
                        postRestoreLogits = model.Forward(new[] { prompt[promptTokens - 1] });
                    }
                    sw.Stop();
                    pagedMs[t] = sw.Elapsed.TotalMilliseconds;
                    _log.LogDebug(LogEventIds.CliBenchmark,
                        "paged-bench paged trial {Trial}: {Ms:F1} ms (restored {Restored}/{Total})",
                        t + 1, pagedMs[t], restoredTokens, promptTokens);
                }

                // Quality probe: sample 8 greedy tokens from the last trial's
                // logits so the user can eyeball whether the codec preserved
                // generation behaviour. With passthrough vs int4 vs int8 the
                // token sequences should be identical (or near-identical) if
                // the codec error stays inside the softmax noise floor.
                if (postRestoreLogits != null)
                {
                    int sampleCount = 8;
                    var sampled = new int[sampleCount];
                    int next = SampleGreedyFromLogits(postRestoreLogits, model.Config.VocabSize);
                    for (int i = 0; i < sampleCount; i++)
                    {
                        sampled[i] = next;
                        float[] logits = model.Forward(new[] { next });
                        next = SampleGreedyFromLogits(logits, model.Config.VocabSize);
                    }
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "paged-bench quality probe: sampledTokens=[{Sampled}] (greedy from post-restore logits; same across codecs => quality preserved)",
                        string.Join(",", sampled));
                }

                long rssAfterPaged = WorkingSetBytes();
                double baselineMedian = Median(baselineMs);
                double pagedMedian = Median(pagedMs);
                double speedup = pagedMedian > 0 ? baselineMedian / pagedMedian : 0;
                double pagedTokensPerMs = pagedMedian > 0 ? promptTokens / pagedMedian : 0;
                double baselineTokensPerMs = baselineMedian > 0 ? promptTokens / baselineMedian : 0;

                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench RESULTS arch={Arch} dtype={Dtype} promptTokens={Prompt} blockSize={BlockSize} trials={Trials}",
                    model.Config.Architecture, model.KvCacheDtype, promptTokens, blockSize, trials);
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench prefill ms (median of {Trials}): baseline={Baseline:F1} primingFirst={Prime:F1} pagedRestore={Paged:F1} speedup={Speedup:F2}x",
                    trials, baselineMedian, primeSw.Elapsed.TotalMilliseconds, pagedMedian, speedup);
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench restored={Restored}/{Prompt} tokens ({Pct:F1}% recovered) tokensPerMs baseline={Bppm:F1} paged={Pppm:F1}",
                    restoredTokens, promptTokens, 100.0 * restoredTokens / promptTokens, baselineTokensPerMs, pagedTokensPerMs);
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "paged-bench memory: paged store={PagedMB:F1} MB ({Blocks} blocks) processRSS warm={WarmRssMB:F0} MB afterBaseline={AfterBaseRssMB:F0} MB afterPaged={AfterPagedRssMB:F0} MB delta={DeltaMB:F0} MB",
                    pagedStats.ramBytes / 1024.0 / 1024.0, pagedStats.ramBlocks,
                    warmRss / 1024.0 / 1024.0, rssAfterBaseline / 1024.0 / 1024.0,
                    rssAfterPaged / 1024.0 / 1024.0, (rssAfterPaged - warmRss) / 1024.0 / 1024.0);
            }
            finally
            {
                pagedManager.Dispose();
            }
        }

        /// <summary>
        /// Reflect <c>--paged-kv*</c> CLI overrides onto the env vars that
        /// <see cref="PagedKvCacheConfig.FromEnvironment"/> reads. We funnel
        /// through env vars (instead of a separate config-passing path) so the
        /// in-process benchmark, the production session manager, and any future
        /// reader all see the same configuration without a divergent code path.
        /// </summary>
        static void ApplyPagedKvCacheCliOverrides(
            bool? enable, int? blockSize, long? ramMb, string ssdDir, long? ssdMb, int? quantBits)
        {
            if (enable.HasValue)
                Environment.SetEnvironmentVariable("TS_KV_PAGED_CACHE", enable.Value ? "1" : "0");
            if (blockSize.HasValue)
                Environment.SetEnvironmentVariable("TS_KV_BLOCK_SIZE", blockSize.Value.ToString(CultureInfo.InvariantCulture));
            if (ramMb.HasValue)
                Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_RAM_MB", ramMb.Value.ToString(CultureInfo.InvariantCulture));
            if (ssdDir != null)
                Environment.SetEnvironmentVariable("TS_KV_CACHE_SSD_DIR", ssdDir);
            if (ssdMb.HasValue)
                Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_SSD_MB", ssdMb.Value.ToString(CultureInfo.InvariantCulture));
            if (quantBits.HasValue)
                Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", quantBits.Value.ToString(CultureInfo.InvariantCulture));

            if (enable == true)
            {
                var cfg = PagedKvCacheConfig.FromEnvironment();
                string codecLabel = quantBits.HasValue && quantBits.Value > 0
                    ? $"turboquant-int{quantBits.Value}"
                    : "passthrough";
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "paged-kv enabled via CLI: blockSize={BlockSize} ramMB={RamMB} ssdDir={SsdDir} maxSsdMB={MaxSsdMB} codec={Codec}",
                    cfg.BlockSize, cfg.MaxRamBytes / (1024 * 1024),
                    string.IsNullOrEmpty(cfg.SsdDirectory) ? "(disabled)" : cfg.SsdDirectory,
                    cfg.MaxSsdBytes / (1024 * 1024), codecLabel);
            }
        }

        /// <summary>
        /// Translate a Qwen-Image-Edit companion-model CLI flag into the env var
        /// QwenImageModel reads (its existing override mechanism). Validates the
        /// path exists so a typo fails fast at startup rather than silently
        /// falling back to the same-directory scan and surfacing as a confusing
        /// "companion not found" later.
        /// </summary>
        static void ApplyQwenImageCompanionOverride(string flag, string envVar, string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return;
            if (!File.Exists(path))
                throw new FileNotFoundException($"{flag} file not found: {path}", path);
            Environment.SetEnvironmentVariable(envVar, Path.GetFullPath(path));
            _log.LogInformation(LogEventIds.HostConfiguration,
                "Qwen-Image-Edit companion override {Flag} -> {Path}", flag, Path.GetFullPath(path));
        }

        private static double Median(double[] values)
        {
            if (values == null || values.Length == 0) return 0;
            var copy = (double[])values.Clone();
            Array.Sort(copy);
            int n = copy.Length;
            return n % 2 == 0 ? (copy[n / 2 - 1] + copy[n / 2]) / 2.0 : copy[n / 2];
        }

        private static long WorkingSetBytes()
        {
            try { return Process.GetCurrentProcess().WorkingSet64; }
            catch { return 0; }
        }

        static void TestTokenizer(ModelBase model)
        {
            _log.LogInformation(LogEventIds.CliBenchmark, "tokenizer test starting");

            string[] testInputs = new[]
            {
                "Hello, world!",
                "What is 1+1?",
                "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n",
            };

            foreach (var input in testInputs)
            {
                var tokens = model.Tokenizer.Encode(input, addSpecial: false);
                string decoded = model.Tokenizer.Decode(tokens);
                bool match = decoded == input;
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "tokenizer test input=\"{Input}\" tokens=[{Tokens}] decoded=\"{Decoded}\" roundtripMatch={Match}",
                    LoggingExtensions.SanitizeForLog(input), string.Join(", ", tokens),
                    LoggingExtensions.SanitizeForLog(decoded), match);
            }
        }

        static void TestChatTemplate()
        {
            _log.LogInformation(LogEventIds.CliBenchmark, "chat template test starting");

            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "Hello" }
            };

            string rendered = ChatTemplate.RenderQwen3(messages, true);
            string expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
            _log.LogInformation(LogEventIds.CliBenchmark,
                "chat template test rendered=\"{Rendered}\" expected=\"{Expected}\" match={Match}",
                LoggingExtensions.SanitizeForLog(rendered),
                LoggingExtensions.SanitizeForLog(expected),
                rendered == expected);
        }

        static void TestInferenceWithOllamaComparison(ModelBase model, int maxTokens, string outputFile)
        {
            _log.LogInformation(LogEventIds.CliBenchmark, "inference comparison test starting");
            string testInput = "What is 1+1?";

            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = testInput }
            };
            string rendered = ChatTemplate.RenderQwen3(messages, true);

            var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);
            _log.LogDebug(LogEventIds.CliBenchmark,
                "comparison test inputTokens count={Count} list=[{Tokens}]",
                inputTokens.Count, string.Join(", ", inputTokens));

            model.ResetKVCache();
            float[] logits = model.Forward(inputTokens.ToArray());
            var engineTokens = new List<int>();

            for (int step = 0; step < maxTokens; step++)
            {
                int nextToken = model.SampleGreedy(logits);
                if (model.Tokenizer.IsEos(nextToken)) break;
                engineTokens.Add(nextToken);
                logits = model.Forward(new[] { nextToken });
            }

            string engineText = model.Tokenizer.Decode(engineTokens);
            _log.LogInformation(LogEventIds.CliBenchmark,
                "comparison test engine output: tokens={EngineTokens} text=\"{EngineText}\"",
                engineTokens.Count, LoggingExtensions.SanitizeForLog(engineText));

            _log.LogInformation(LogEventIds.CliBenchmark, "querying ollama for comparison");
            string ollamaResponse = QueryOllama(rendered, maxTokens);
            _log.LogInformation(LogEventIds.CliBenchmark,
                "comparison test ollama output: text=\"{OllamaText}\"",
                LoggingExtensions.SanitizeForLog(ollamaResponse));

            var ollamaTokens = model.Tokenizer.Encode(ollamaResponse, addSpecial: false);
            _log.LogDebug(LogEventIds.CliBenchmark,
                "comparison test engineTokens=[{EngineTokens}] ollamaTokens=[{OllamaTokens}]",
                string.Join(", ", engineTokens), string.Join(", ", ollamaTokens));

            int matchCount = 0;
            int compareLen = Math.Min(engineTokens.Count, ollamaTokens.Count);
            for (int i = 0; i < compareLen; i++)
            {
                if (engineTokens[i] == ollamaTokens[i])
                    matchCount++;
                else
                {
                    _log.LogWarning(LogEventIds.CliBenchmark,
                        "comparison test mismatch at position {Position}: engine={EngineToken}({EngineVocab}) ollama={OllamaToken}({OllamaVocab})",
                        i, engineTokens[i], model.Tokenizer.Vocab[engineTokens[i]],
                        ollamaTokens[i], model.Tokenizer.Vocab[ollamaTokens[i]]);
                    break;
                }
            }
            bool match = engineText == ollamaResponse;
            _log.LogInformation(LogEventIds.CliBenchmark,
                "comparison test result: tokenMatch={MatchCount}/{CompareLen} ({MatchPercent:F1}%) textMatch={TextMatch}",
                matchCount, compareLen,
                compareLen > 0 ? 100.0 * matchCount / compareLen : 0,
                match);

            if (outputFile != null)
            {
                File.WriteAllText(outputFile, $"Engine: {engineText}\nOllama: {ollamaResponse}\nMatch: {match}\n");
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "comparison test output written to {OutputFile}", outputFile);
            }
        }

        static string QueryOllama(string rawPrompt, int maxTokens)
        {
            try
            {
                using var client = new System.Net.Http.HttpClient();
                client.Timeout = TimeSpan.FromSeconds(120);
                string json = System.Text.Json.JsonSerializer.Serialize(new
                {
                    model = "qwen3-fp16-test",
                    prompt = rawPrompt,
                    raw = true,
                    stream = false,
                    options = new
                    {
                        temperature = 0,
                        num_predict = maxTokens,
                        seed = 42
                    }
                });
                var content = new System.Net.Http.StringContent(json, Encoding.UTF8, "application/json");
                var response = client.PostAsync("http://localhost:11434/api/generate", content).Result;
                string body = response.Content.ReadAsStringAsync().Result;
                using var doc = System.Text.Json.JsonDocument.Parse(body);
                return doc.RootElement.GetProperty("response").GetString();
            }
            catch (Exception ex)
            {
                _log.LogError(LogEventIds.CliFailed, ex,
                    "Failed to query ollama: {Error}", ex.Message);
                return "";
            }
        }

        static void TestChatTemplates(string modelDir)
        {
            _log.LogInformation(LogEventIds.CliBenchmark,
                "chat template scan starting: directory={Directory}", modelDir);

            var ggufFiles = Directory.GetFiles(modelDir, "*.gguf")
                .Where(f => !Path.GetFileName(f).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f)
                .ToArray();

            if (ggufFiles.Length == 0)
            {
                _log.LogWarning(LogEventIds.CliBenchmark,
                    "chat template scan: no GGUF files found in {Directory}", modelDir);
                return;
            }

            // Test scenarios
            var singleTurn = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "What is 1+1?" }
            };
            var multiTurn = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "Hello!" },
                new ChatMessage { Role = "assistant", Content = "Hi there! How can I help?" },
                new ChatMessage { Role = "user", Content = "What is the capital of France?" }
            };
            var withSystem = new List<ChatMessage>
            {
                new ChatMessage { Role = "system", Content = "You are a helpful assistant." },
                new ChatMessage { Role = "user", Content = "Tell me a joke." }
            };

            int passed = 0, failed = 0, skipped = 0;

            foreach (string file in ggufFiles)
            {
                string fileName = Path.GetFileName(file);

                try
                {
                    using var gguf = new GgufFile(file);
                    string arch = gguf.GetString("general.architecture");
                    string template = gguf.GetString("tokenizer.chat_template");

                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "chat template scan {File}: architecture={Architecture} templateChars={TemplateChars}",
                        fileName, arch, template?.Length ?? 0);

                    if (template == null)
                    {
                        _log.LogWarning(LogEventIds.CliBenchmark,
                            "chat template scan {File}: SKIP no chat template in GGUF metadata", fileName);
                        skipped++;
                        continue;
                    }

                    var scenarios = new (string Name, List<ChatMessage> Msgs)[]
                    {
                        ("single-turn", singleTurn),
                        ("multi-turn", multiTurn),
                        ("with-system", withSystem),
                    };

                    bool allPassed = true;
                    foreach (var (name, msgs) in scenarios)
                    {
                        // Render with Jinja2
                        string jinja2Result = null;
                        Exception jinja2Error = null;
                        try
                        {
                            var preprocessed = msgs; // no multimodal in this test
                            var jinja = new Jinja2Template(template);
                            var ctx = BuildTemplateTestContext(preprocessed, true);
                            jinja2Result = jinja.Render(ctx);
                        }
                        catch (Exception ex)
                        {
                            jinja2Error = ex;
                        }

                        // Render with hardcoded fallback
                        string hardcodedResult = ChatTemplate.RenderFromGgufTemplate(
                            null, msgs, addGenerationPrompt: true, architecture: arch);

                        if (jinja2Error != null)
                        {
                            _log.LogError(LogEventIds.CliFailed, jinja2Error,
                                "chat template scan {File} [{Scenario}] FAIL Jinja2 error",
                                fileName, name);
                            allPassed = false;
                            continue;
                        }

                        string j2 = jinja2Result?.Trim() ?? "";
                        string hc = hardcodedResult?.Trim() ?? "";
                        bool match = j2 == hc;

                        if (match)
                        {
                            _log.LogInformation(LogEventIds.CliBenchmark,
                                "chat template scan {File} [{Scenario}] PASS chars={Chars}",
                                fileName, name, j2.Length);
                        }
                        else
                        {
                            _log.LogWarning(LogEventIds.CliBenchmark,
                                "chat template scan {File} [{Scenario}] MISMATCH jinja2Chars={J2Chars} hardcodedChars={HcChars} jinja2Sample={J2Sample} hardcodedSample={HcSample}",
                                fileName, name, j2.Length, hc.Length, Escape(j2), Escape(hc));
                            allPassed = false;
                        }
                    }

                    if (allPassed) passed++; else failed++;
                }
                catch (Exception ex)
                {
                    _log.LogError(LogEventIds.CliFailed, ex,
                        "chat template scan {File} ERROR: {Error}", fileName, ex.Message);
                    failed++;
                }
            }

            _log.LogInformation(LogEventIds.CliBenchmark,
                "chat template scan results: passed={Passed} failed={Failed} skipped={Skipped} total={Total}",
                passed, failed, skipped, ggufFiles.Length);
        }

        static Dictionary<string, object> BuildTemplateTestContext(List<ChatMessage> messages, bool addGenerationPrompt)
        {
            var msgList = new List<object>();
            foreach (var m in messages)
            {
                msgList.Add(new Dictionary<string, object>
                {
                    ["role"] = m.Role,
                    ["content"] = m.Content ?? ""
                });
            }
            return new Dictionary<string, object>
            {
                ["messages"] = msgList,
                ["add_generation_prompt"] = addGenerationPrompt,
                ["bos_token"] = "",
                ["eos_token"] = "",
            };
        }

        static string Escape(string s)
        {
            if (s.Length > 200) s = s.Substring(0, 200) + "...";
            return s.Replace("\n", "\\n").Replace("\r", "\\r").Replace("\t", "\\t");
        }
    }
}
