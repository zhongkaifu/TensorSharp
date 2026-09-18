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
using TensorSharp.Cuda;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Cli
{
    partial class Program
    {
        private static readonly IPromptRenderer PromptRenderer = new GgufPromptRenderer();
        private static ILogger _log = NullLogger.Instance;

        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;

            // Merge in options from a --config <file.json> before anything reads
            // argv. File-derived tokens are spliced in ahead of the real
            // command line, so any option also passed on the command line
            // overrides the file (both here and in MainCore parse last-wins).
            try
            {
                args = ConfigFileArgs.Expand(args);
            }
            catch (Exception ex) when (ex is ArgumentException or FileNotFoundException)
            {
                Console.Error.WriteLine("Configuration error: " + ex.Message);
                Environment.ExitCode = HostExitCodes.ConfigurationError;
                return;
            }

            bool showSarah = Array.Exists(args, a => a == "--xzf");
            ConsoleBanner.Print(showSarah);

            // A bare `TensorSharp.Cli` or `--help` shows the full usage page —
            // every option with its description, default, range, and an example —
            // and exits before any logging/model machinery spins up.
            if (args.Length == 0 || CliUsage.IsHelpRequested(args))
            {
                CliUsage.PrintUsage(Console.Out);
                return;
            }

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
                _log.LogInformation(LogEventIds.CliCompleted, "tensorsharp-cli completed exitCode={ExitCode}", Environment.ExitCode);
                try { TensorSharp.GGML.GgmlBasicOps.Shutdown(); }
                catch { /* native lib may be absent for non-GGML backends */ }

                // ggml-vulkan on Linux: the NVIDIA driver's worker threads
                // ("[vkrt] Analysis") race the C++ static destructors that tear
                // the Vulkan instance down and intermittently segfault the
                // process AFTER all work — and the ordered Shutdown above —
                // has completed. Nothing is left to clean up (backends, caches
                // and graphs were freed by Shutdown), so skip the destructors:
                // flush what buffers output and leave through _exit.
                if (OperatingSystem.IsLinux() && SelectedBackend(args) == "ggml_vulkan")
                {
                    loggerFactory.Dispose();
                    Console.Out.Flush();
                    Console.Error.Flush();
                    LibcExit(Environment.ExitCode);
                }
            }
            catch (ArgumentException ex)
            {
                // A bad flag or value (including a removed spelling's "use --X
                // instead" pointer) is the operator's to fix; a stack trace
                // buries the one line they need. The full exception still goes
                // to the log for the rare deep ArgumentException.
                _log.LogError(LogEventIds.CliFailed, ex, "cli.invalid-arguments {Error}", ex.Message);
                Console.Error.WriteLine("Configuration error: " + ex.Message);
                Environment.ExitCode = HostExitCodes.ConfigurationError;
            }
            catch (Exception ex)
            {
                _log.LogCritical(LogEventIds.CliFailed, ex,
                    "tensorsharp-cli aborted with unhandled exception {ExceptionType}", ex.GetType().Name);
                throw;
            }
            finally
            {
                // The CLI process IS the session, so its execution workspace — the
                // working directory, the packages installed into it — is released when
                // the process leaves, whichever of MainCore's many exits it took. Doing
                // this here rather than at one call site is what keeps the one-shot
                // (--input) path from leaking a directory the interactive path cleans up.
                // Files a program PRODUCED were already copied into the artifact store,
                // which outlives this.
                ReleaseCodeWorkspace();
            }
        }

        /// <summary>
        /// This process's code-execution workspace, held statically so
        /// <see cref="Main"/> can release it on every exit path.
        /// </summary>
        private static SessionWorkspaceManager _codeWorkspaces;
        private static SessionWorkspace _codeWorkspace;

        private static void ReleaseCodeWorkspace()
        {
            SessionWorkspaceManager manager = _codeWorkspaces;
            SessionWorkspace workspace = _codeWorkspace;
            _codeWorkspaces = null;
            _codeWorkspace = null;
            if (manager != null && workspace != null)
                manager.Release(workspace.Id);
        }

        /// <summary>The effective --backend value (last one wins, as in MainCore).</summary>
        private static string SelectedBackend(string[] args)
        {
            string backend = null;
            for (int i = 0; i + 1 < args.Length; i++)
            {
                if (args[i] == "--backend")
                    backend = args[i + 1].ToLowerInvariant();
            }
            return backend;
        }

        /// <summary>
        /// Draft path published by <see cref="SpeculativeCliFlags.Apply"/> or by
        /// the process environment. Reading the shared environment contract here
        /// makes both <c>--draft-model PATH</c> and <c>--draft-model=PATH</c> reach
        /// the model factory; the main argument switch only sees the former.
        /// </summary>
        internal static string ResolveConfiguredDraftModelPath()
        {
            string path = Environment.GetEnvironmentVariable(SpeculationEnvVars.DraftModel);
            if (string.IsNullOrWhiteSpace(path))
                path = Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraftModel);
            return string.IsNullOrWhiteSpace(path) ? null : path.Trim();
        }

        [System.Runtime.InteropServices.LibraryImport("libc", EntryPoint = "_exit")]
        private static partial void LibcExit(int status);

        static void MainCore(string[] args)
        {
            // Parsed BEFORE the switch below and REMOVED from the argument list, the
            // same way the server does it: the code-execution flags are owned by
            // CodeExecOptions rather than by this switch, so consuming them here is
            // what keeps the switch from meeting a flag it has no case for. Running
            // code the MODEL wrote is its own decision, separate from --skills-allow-exec.
            // Retired spellings are refused first, against the ORIGINAL line: Parse
            // consumes what it recognises, and the CLI's own switch has no unknown-flag
            // trap at all, so a retired --code-exec-packages would otherwise be silently
            // dropped and the operator would never learn their setting stopped applying.
            if (CodeExecOptions.RejectRemoved(args) is { } removedCodeExecFlag)
            {
                Console.Error.WriteLine(removedCodeExecFlag);
                Environment.ExitCode = HostExitCodes.ConfigurationError;
                return;
            }

            CodeExecOptions codeExecOptions;
            List<string> remainingArgs;
            try
            {
                codeExecOptions = CodeExecOptions.Parse(args, out remainingArgs);
            }
            catch (ArgumentException ex)
            {
                Console.Error.WriteLine(ex.Message);
                Environment.ExitCode = HostExitCodes.ConfigurationError;
                return;
            }
            codeExecOptions.ApplyEnvironment();
            args = remainingArgs.ToArray();

            // Pick up the KV cache dtype from the KV_CACHE_DTYPE environment variable
            // before parsing CLI args. The --kv-cache-dtype flag below overrides this.
            KvCacheDtypeConfig.ConfigureFromEnvironment();

            // Same contract for MoE CPU offload: TS_N_CPU_MOE / TS_CPU_MOE seed the
            // default, --n-cpu-moe / --cpu-moe below override it.
            MoeCpuOffloadConfig.ConfigureFromEnvironment();

            // --spec / --spec-draft / --spec-pmin / --draft-model become the
            // shared TS_SPEC_* settings (with legacy TS_MTP_* mirrors) before anything
            // else runs, because the request has to reach
            // the LOADER and not just the decode loop: glm-dsa pages its NextN
            // block into VRAM (a whole extra 256-expert layer) only when TS_SPEC
            // is already set, and sizes its graph cache from TS_SPEC_DRAFT. Parsing
            // these in the switch below would be too late to matter. Shared with
            // the server so the two hosts cannot drift on names or validation.
            SpeculativeCliFlags.Apply(args);

            string modelPath = null;
            string inputFile = null;
            string pdfPath = null;
            string outputFile = null;
            string imagePath = null;
            var imagePathList = new List<string>();   // every --image in order (multi-image edit)
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
            // Agent Skills. Parsed by the shared reader after the switch so this host and
            // the server accept exactly the same spellings; the locals below only exist
            // for the cases the switch itself has to consume.
            var skillsCliArgs = new List<string>();
            bool dumpPrompt = false;
            bool runBenchmark = false;
            int benchmarkPrefill = 32;
            int benchmarkDecode = 64;
            int benchmarkRuns = 1;
            bool benchmarkChunked = false;
            bool benchmarkFixedTokens = false;
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
            bool noPrefixCache = false;
            // Vulkan GPU selection (multi-GPU hosts, e.g. an integrated Intel GPU
            // next to a discrete NVIDIA one). Plumbed through the env var that
            // GgmlNative reads when the ggml_vulkan backend initializes.
            int? gpuDeviceOverride = null;
            bool listGpus = false;
            int tpDegree = 1;
            int tpNodeId = -1;          // -1 = not set; distributed mode requires >= 0
            string tpPeers = null;      // comma-separated host:port list for distributed TP
            string systemPrompt = null;
            int warmupInferenceRuns = 0;
            // DiffusionGemma sampler knobs (used only for the diffusion-gemma architecture).
            int diffusionSteps = 48;
            int diffusionSeed = 0;
            int imageWidth = 0, imageHeight = 0;   // explicit Qwen-Image-Edit output size (0 = auto/VRAM-clamped)
            int diffusionBlocks = 0;   // 0 => derive from --max-tokens and canvas_length
            // Qwen-Image-Edit knobs.
            string editPrompt = null;
            float cfgScale = 2.5f;   // Qwen-Image-Edit-2511 recommendation; 4.0 over-guides (distorts faces)
            bool cfgScaleSet = false;          // explicit --cfg (image edit passes 0 = auto otherwise)
            bool diffusionStepsSet = false;    // explicit --diffusion-steps (image edit passes 0 = auto otherwise)
            bool diffusionSeedSet = false;     // explicit --diffusion-seed (video draws a random seed otherwise)
            // Qwen-Image-Edit companion GGUFs. The qwen_image DiT GGUF (passed via
            // --model) carries none of these, so the operator can point at them
            // explicitly instead of relying on a same-directory scan / env vars.
            string qwenImageVaePath = null;
            string qwenImageVlPath = null;
            string qwenImageMmprojPath = null;
            string qwenImageLoraPath = null;
            bool offloadCpu = false;
            // Wan text-to-video knobs.
            int videoFrames = 0;       // 0 = model default (33)
            int videoFps = 0;          // 0 = model default (16)
            float flowShift = 0f;      // 0 = auto (8.0 for 1.3B video; 3.0/5.0 otherwise)
            string videoSampler = null;   // null = unipc (the official Wan sampler)
            int cfgCacheStride = 0;       // 0/1 = off: every step runs both CFG passes
            string negativePrompt = null;
            string videoVaePath = null;
            string videoTextEncoderPath = null;
            string videoDit2Path = null;
            string videoAudioVaePath = null;
            string endImagePath = null;
            string videoMode = null;
            var refImagePaths = new List<string>();
            var refVideoPaths = new List<string>();
            var refAudioPaths = new List<string>();
            var refVideoAudioPaths = new List<string>();
            bool videoAudioEnabled = true;
            // SpeculativeCliFlags.Apply already ran, so this captures both value
            // spellings as well as an env-only deployment before ModelBase.Create.
            string draftModelPath = ResolveConfiguredDraftModelPath();
            int specDraftMax = 0;
            float specDraftConfMin = -1f;

            var samplingConfig = SamplingConfig.Greedy;
            // Which sampling knobs the operator set explicitly. Model- and
            // mode-derived defaults must never overwrite these.
            var pinnedSampling = SamplingFields.None;

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model": modelPath = args[++i]; break;
                    case "--input": inputFile = args[++i]; break;
                    case "--pdf": pdfPath = args[++i]; break;
                    case "--input-jsonl": inputJsonl = args[++i]; break;
                    case "--output": outputFile = args[++i]; break;
                    case "--image": imagePath = args[++i]; imagePathList.Add(imagePath); break;
                    case "--prompt": editPrompt = args[++i]; break;
                    case "--cfg": cfgScale = float.Parse(args[++i]); cfgScaleSet = true; break;
                    case "--qwen-image-vae": qwenImageVaePath = args[++i]; break;
                    case "--qwen-image-vl": qwenImageVlPath = args[++i]; break;
                    case "--qwen-image-mmproj": qwenImageMmprojPath = args[++i]; break;
                    case "--qwen-image-lora": qwenImageLoraPath = args[++i]; break;
                    case "--video-frames": videoFrames = int.Parse(args[++i]); break;
                    case "--fps": videoFps = int.Parse(args[++i]); break;
                    case "--flow-shift": flowShift = float.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--sampler": videoSampler = args[++i]; break;
                    case "--cfg-cache-stride": cfgCacheStride = int.Parse(args[++i]); break;
                    case "--negative-prompt": negativePrompt = args[++i]; break;
                    // Companion-network paths. The --wan-* spellings predate the second
                    // video model and stay accepted so existing configs keep working.
                    case "--video-vae": case "--wan-vae": videoVaePath = args[++i]; break;
                    case "--video-text-encoder": case "--video-te": case "--wan-te":
                        videoTextEncoderPath = args[++i]; break;
                    case "--video-dit2": case "--wan-dit2": videoDit2Path = args[++i]; break;
                    case "--audio-vae": videoAudioVaePath = args[++i]; break;
                    case "--end-image": endImagePath = args[++i]; break;
                    case "--video-mode": videoMode = args[++i]; break;
                    case "--ref-image": refImagePaths.Add(args[++i]); break;
                    case "--ref-video": refVideoPaths.Add(args[++i]); break;
                    case "--ref-audio": refAudioPaths.Add(args[++i]); break;
                    case "--ref-video-audio": refVideoAudioPaths.Add(args[++i]); break;
                    case "--no-audio": videoAudioEnabled = false; break;
                    case "--offload-cpu": offloadCpu = true; break;
                    case "--audio": audioPath = args[++i]; break;
                    case "--video": videoPath = args[++i]; break;
                    case "--mmproj": mmProjPath = args[++i]; break;
                    case "--draft-model": draftModelPath = args[++i]; break;
                    case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
                    case "--test": runTest = true; break;
                    case "--backend": backendStr = args[++i].ToLowerInvariant(); break;
                    case "--tp": tpDegree = int.Parse(args[++i]); break;
                    case "--tp-node-id": tpNodeId = int.Parse(args[++i]); break;
                    case "--tp-peers": tpPeers = args[++i]; break;
                    case "--gpu-device":
                    {
                        string gpuStr = args[++i];
                        if (!int.TryParse(gpuStr, NumberStyles.Integer, CultureInfo.InvariantCulture, out int gpuIndex) || gpuIndex < 0)
                            throw new ArgumentException($"Invalid value for --gpu-device: '{gpuStr}'. Expected a non-negative Vulkan device index (see --list-gpus).");
                        gpuDeviceOverride = gpuIndex;
                        break;
                    }
                    case "--list-gpus": listGpus = true; break;
                    case "--test-templates": testTemplatesDir = args[++i]; break;
                    case "--think": enableThinking = true; break;
                    case "--tools": toolsFile = args[++i]; break;
                    // Agent Skills. The values are collected verbatim and handed to
                    // SkillHostOptions.Parse below, which owns the validation and the
                    // env-var layering for both hosts.
                    case "--skills-dir":
                    case "--skill":
                    case "--skills-max-rounds":
                    case "--skills-sandbox":
                        skillsCliArgs.Add(args[i]);
                        skillsCliArgs.Add(args[++i]);
                        break;
                    case "--list-skills":
                    case "--no-skills":
                    case "--skills-no-discovery":
                    case "--skills-allow-exec":
                    case "--skills-allow-network":
                        skillsCliArgs.Add(args[i]);
                        break;
                    case "--dump-prompt": dumpPrompt = true; break;
                    case "--multi-turn-jsonl": multiTurnJsonl = args[++i]; break;
                    case "--benchmark": runBenchmark = true; break;
                    case "--bench-prefill": benchmarkPrefill = int.Parse(args[++i]); break;
                    case "--bench-decode": benchmarkDecode = int.Parse(args[++i]); break;
                    case "--bench-runs": benchmarkRuns = int.Parse(args[++i]); break;
                    case "--bench-chunked": benchmarkChunked = true; break;
                    case "--bench-fixed-tokens": benchmarkFixedTokens = true; break;
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
                    case "--no-prefix-cache":
                        // Spelled the same as the server's, because a config file's keys
                        // ARE flags and the same file is expected to drive either host.
                        noPrefixCache = true;
                        Environment.SetEnvironmentVariable("TS_SCHED_PREFIX_CACHE", "0");
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
                    case "--diffusion-steps": diffusionSteps = int.Parse(args[++i]); diffusionStepsSet = true; break;
                    case "--diffusion-seed": diffusionSeed = int.Parse(args[++i]); diffusionSeedSet = true; break;
                    case "--width": imageWidth = int.Parse(args[++i]); break;
                    case "--height": imageHeight = int.Parse(args[++i]); break;
                    case "--diffusion-blocks": diffusionBlocks = int.Parse(args[++i]); break;
                    case "--kv-cache-dtype":
                        {
                            string kvDtypeStr = args[++i];
                            if (!KvCacheDtypeConfig.TryParse(kvDtypeStr, out KvCacheDtype kvDtype))
                                throw new ArgumentException($"Unknown --kv-cache-dtype value '{kvDtypeStr}'. Valid: f32, f16, q8_0, q4_0.");
                            KvCacheDtypeConfig.Set(kvDtype);
                            break;
                        }
                    case "-ncmoe":
                    case "--n-cpu-moe":
                        {
                            string ncmoeStr = args[++i];
                            if (!MoeCpuOffloadConfig.TryParse(ncmoeStr, out int ncmoeLayers, out bool ncmoeAll))
                                throw new ArgumentException($"Invalid --n-cpu-moe value '{ncmoeStr}'. Expected a non-negative integer or 'all'.");
                            if (ncmoeAll) MoeCpuOffloadConfig.SetAllLayers();
                            else MoeCpuOffloadConfig.SetLayers(ncmoeLayers);
                            break;
                        }
                    case "-cmoe":
                    case "--cpu-moe":
                        MoeCpuOffloadConfig.SetAllLayers();
                        break;
                    case "--cpu-moe-threads":
                        MoeCpuOffloadConfig.SetCpuThreads(int.Parse(args[++i], CultureInfo.InvariantCulture));
                        break;
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
                    case "--temperature": samplingConfig.Temperature = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.Temperature; break;
                    case "--top-k": samplingConfig.TopK = int.Parse(args[++i]); pinnedSampling |= SamplingFields.TopK; break;
                    case "--top-p": samplingConfig.TopP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.TopP; break;
                    case "--min-p": samplingConfig.MinP = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.MinP; break;
                    case "--repeat-penalty": samplingConfig.RepetitionPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.RepetitionPenalty; break;
                    case "--penalty-last-n": samplingConfig.PenaltyLastN = int.Parse(args[++i]); pinnedSampling |= SamplingFields.PenaltyLastN; break;
                    case "--presence-penalty": samplingConfig.PresencePenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.PresencePenalty; break;
                    case "--frequency-penalty": samplingConfig.FrequencyPenalty = float.Parse(args[++i], System.Globalization.CultureInfo.InvariantCulture); pinnedSampling |= SamplingFields.FrequencyPenalty; break;
                    case "--seed": samplingConfig.Seed = int.Parse(args[++i]); break;
                    case "--stop":
                        samplingConfig.StopSequences ??= new List<string>();
                        samplingConfig.StopSequences.Add(args[++i]);
                        break;
                }
            }

            if (listGpus)
            {
                ListVulkanGpus();
                return;
            }

            // Agent Skills, resolved before the model is touched so `--list-skills`
            // works with no `--model` at all and a mistyped `--skills-dir` fails here
            // rather than after a multi-gigabyte load.
            SkillHostOptions skillOptions;
            try
            {
                skillOptions = SkillHostOptions.Parse(skillsCliArgs)
                    .ApplyEnvironmentAndDefaults(AppContext.BaseDirectory);
                if (skillOptions.Enabled)
                {
                    bool onlyDefaultRoot = skillOptions.Roots.Count == 1
                        && string.Equals(
                            skillOptions.Roots[0],
                            Path.Combine(AppContext.BaseDirectory, SkillHostOptions.DefaultDirectoryName),
                            StringComparison.Ordinal);
                    skillOptions.ValidateRoots(createDefault: onlyDefaultRoot);
                }
            }
            catch (ArgumentException ex)
            {
                _log.LogError(LogEventIds.CliFailed, "cli.skills.invalid {Error}", ex.Message);
                Console.Error.WriteLine(ex.Message);
                return;
            }

            SkillRegistry skillRegistry = skillOptions.Enabled
                ? new SkillRegistry(skillOptions.ToRegistryOptions(), _log)
                : null;

            if (skillOptions.ListOnly)
            {
                ListSkills(skillRegistry);
                return;
            }

            // Code execution (--code-exec). The CLI is ONE session for the life of the
            // process, so it gets exactly one workspace: files a program writes and
            // packages it installs stay available to the next call and to a skill's
            // scripts, and the whole thing is released when the process exits. Orphans
            // from a previous run are swept first — a crashed run leaves a directory
            // nothing can reach again.
            ICodeRunner codeRunner = null;
            SessionWorkspace codeWorkspace = null;
            CodeArtifactStore codeArtifacts = null;
            if (codeExecOptions.Enabled)
            {
                if (codeExecOptions.Unconfined)
                {
                    Console.Error.WriteLine(
                        $"{CodeExecOptions.UnconfinedFlag} is set: model-authored commands may fall back to "
                        + "this process's filesystem and network privileges when confinement is unavailable "
                        + "(every run on Windows). Do not use it for a "
                        + "session or machine you do not trust.");
                }
                if (codeExecOptions.AllowNetwork)
                {
                    Console.Error.WriteLine(
                        $"{CodeExecOptions.AllowNetworkFlag} is set: model-authored commands have "
                        + "unrestricted IP network access, including LAN/loopback services and listening "
                        + "sockets, and can send any data the sandbox lets them read. Package/install-domain "
                        + "allow-lists constrain only the host installer, not direct downloads by a command. "
                        + "On macOS, a deliberately detached child may outlive its request; tool results report that gap.");
                }
                codeExecOptions.ScratchDirectory ??= Path.Combine(AppContext.BaseDirectory, "code-scratch");
                codeArtifacts = new CodeArtifactStore(
                    codeExecOptions.ArtifactDirectory
                        ?? Path.Combine(AppContext.BaseDirectory, "code-artifacts"),
                    new CodeArtifactLimits());
                _codeWorkspaces = new SessionWorkspaceManager(codeExecOptions.ScratchDirectory, _log);
                _codeWorkspaces.SweepOrphans();
                codeWorkspace = _codeWorkspaces.GetOrCreate(
                    "cli-" + Environment.ProcessId.ToString(CultureInfo.InvariantCulture));
                _codeWorkspace = codeWorkspace;

                // ArtifactUriPrefix stays null here on purpose: a server hands back a
                // URL, but there is nothing serving one in a terminal, so the pointer a
                // program's output gets is its absolute path on disk — which is what a
                // CLI user can actually open.
                var runner = new ShellRunner(codeExecOptions, _log, codeArtifacts);
                codeRunner = new CodeRunnerAdapter(runner, codeExecOptions);

                if (!runner.CanRun)
                {
                    Console.Error.WriteLine(
                        SkillToolNames.Shell + " is not available: " + runner.UnavailableReason);
                    codeRunner = null;
                }
                else
                {
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "cli.codeexec.ready shell={Shell} sandbox={Sandbox} install={AllowInstall} network={AllowNetwork} tools={Tools} workspace={Workspace}",
                        runner.Shell?.Name ?? "none", runner.Sandbox?.Name ?? "none",
                        codeExecOptions.AllowInstall, codeExecOptions.AllowNetwork,
                        string.Join(",", CodeEnvironment.AvailableTools), codeWorkspace.Root);
                }
            }

            List<Skill> selectedSkills = new List<Skill>();
            if (skillRegistry != null && skillOptions.Selected.Count > 0)
            {
                selectedSkills = new List<Skill>(
                    skillRegistry.Resolve(skillOptions.Selected, out var unknownSkills));
                if (unknownSkills.Count > 0)
                {
                    _log.LogError(LogEventIds.CliFailed,
                        "cli.skills.unknown name={SkillName}", unknownSkills[0]);
                    Console.Error.WriteLine(
                        $"No skill called '{unknownSkills[0]}' was found in {string.Join(", ", skillRegistry.Roots)}.");
                    Console.Error.WriteLine("Run with --list-skills to see what is available.");
                    return;
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
                try
                {
                    tools = ToolFunction.ParseList(File.ReadAllText(toolsFile));
                }
                catch (JsonException ex)
                {
                    _log.LogError(LogEventIds.CliFailed,
                        "Tools file {ToolsFile} is not a usable tool definition list: {Error}", toolsFile, ex.Message);
                    return;
                }
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Loaded {ToolCount} tool definition(s) from {ToolsFile}", tools.Count, toolsFile);
            }

            // Agent Skills. The plan is finalised here, before the model loads, because
            // that is where the registry already is; the CONTEXT budget it needs is
            // filled in after the load (see skillPlan below), since MaxContextLength is
            // not known until then.
            SkillPlan skillPlan = null;
            SkillToolContext skillToolContext = null;
            bool skillScriptsAllowed = skillOptions.AllowScripts;

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
                };
                modelPath = candidates.FirstOrDefault(File.Exists);
            }

            if (modelPath == null || !File.Exists(modelPath))
            {
                _log.LogError(LogEventIds.CliFailed,
                "Model file not found: {ModelPath}", modelPath ?? "(none)");
                // This used to print usage and exit 0, so a script could not tell a
                // missing model from a finished run.
                Console.Error.WriteLine(ModelLoadRefusal.FormatErrorLine(
                    $"model file not found: {modelPath ?? "(no --model given)"}. " +
                    "Usage: TensorSharp.Cli --model <path.gguf> [options]; --help lists every option."));
                Environment.ExitCode = HostExitCodes.ModelLoadRefused;
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
                "ggml_vulkan" or "ggml-vulkan" => BackendType.GgmlVulkan,
                _ => throw new ArgumentException($"Unknown backend '{backendStr}'. Use: cpu, cuda, mlx, ggml_cpu, ggml_metal, ggml_cuda, ggml_vulkan"),
            };

            ApplyPagedKvCacheCliOverrides(
                pagedKvEnableOverride, pagedKvBlockSizeOverride,
                pagedKvRamMbOverride, pagedKvSsdDirOverride, pagedKvSsdMbOverride,
                pagedKvQuantBitsOverride);

            // A NextN block with no LM head of its own borrows the trunk's, which is
            // column-parallel under tensor parallelism — the draft would read one
            // rank's strip of the vocabulary. The loaders refuse it (and say so on
            // stderr), but the refusal is easy to miss in a long load, and the
            // operator has meanwhile lost the VRAM they were budgeting for context.
            // Say it up front, where the two flags were typed.
            if (tpDegree > 1 && SchedulerConfig.FromEnvironment().Speculation.Enabled)
            {
                _log.LogWarning(LogEventIds.HostConfiguration,
                    "--spec with --tp {Degree}: a draft block that borrows the trunk's LM head cannot draft "
                    + "under tensor parallelism (GLM-5.2 is one such checkpoint) and speculation will be refused "
                    + "at load. Drop --tp to speculate, or --no-spec to keep the split.", tpDegree);
            }

            if (gpuDeviceOverride.HasValue)
            {
                if (backend == BackendType.GgmlVulkan)
                {
                    // GgmlNative reads this env var (managed-side) when the Vulkan
                    // backend initializes and pushes the index down to the native
                    // bridge before ggml_backend_vk_init runs.
                    Environment.SetEnvironmentVariable(
                        TensorSharp.GGML.GgmlBasicOps.VulkanDeviceEnvVar,
                        gpuDeviceOverride.Value.ToString(CultureInfo.InvariantCulture));
                    _log.LogInformation(LogEventIds.HostConfiguration,
                        "Vulkan GPU selected via CLI: --gpu-device {DeviceIndex}", gpuDeviceOverride.Value);
                }
                else
                {
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "--gpu-device applies to the ggml_vulkan backend only; ignored for backend {Backend}", backend);
                }
            }

            // Qwen-Image-Edit: let the operator override the companion GGUFs that
            // QwenImageModel otherwise resolves from a same-directory scan. These
            // are translated into the env vars QwenImageModel reads (the existing
            // override mechanism) and validated here so a typo fails fast instead
            // of silently falling back to the directory scan.
            ApplyQwenImageCompanionOverride("--qwen-image-vae", "TS_QWEN_IMAGE_VAE", qwenImageVaePath);
            ApplyQwenImageCompanionOverride("--qwen-image-vl", "TS_QWEN_IMAGE_TE", qwenImageVlPath);
            ApplyQwenImageCompanionOverride("--qwen-image-mmproj", "TS_QWEN_IMAGE_MMPROJ", qwenImageMmprojPath);
            ApplyQwenImageCompanionOverride("--qwen-image-lora", "TS_QWEN_IMAGE_LORA", qwenImageLoraPath);
            // sd.cpp --offload-to-cpu equivalent: force DiT weight streaming from RAM (the
            // pipeline also auto-engages it when the target resolution needs the VRAM).
            if (offloadCpu)
                Environment.SetEnvironmentVariable("TS_QWEN_IMAGE_OFFLOAD_CPU", "1");

            // Video generation: companion overrides. Each path is published under both the
            // generic TS_VIDEO_* name and the historical TS_WAN_* one, so WanVideoModel keeps
            // reading exactly what it always did while new models read the generic names.
            ApplyVideoCompanionOverride("--video-vae", videoVaePath, "TS_VIDEO_VAE", "TS_WAN_VAE");
            ApplyVideoCompanionOverride("--video-text-encoder", videoTextEncoderPath, "TS_VIDEO_TEXT_ENCODER", "TS_WAN_TE");
            ApplyVideoCompanionOverride("--video-dit2", videoDit2Path, "TS_VIDEO_DIT2", "TS_WAN_DIT2");
            ApplyVideoCompanionOverride("--audio-vae", videoAudioVaePath, "TS_VIDEO_AUDIO_VAE");

            if (MoeCpuOffloadConfig.IsEnabled)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "MoE CPU offload active: routed experts of {Layers} stay in system RAM and run on the host ({Threads} threads)",
                    MoeCpuOffloadConfig.Describe(),
                    MoeCpuOffloadConfig.CpuThreads > 0
                        ? MoeCpuOffloadConfig.CpuThreads.ToString(CultureInfo.InvariantCulture)
                        : "auto");
            }

            string requestedDtype = KvCacheDtypeConfig.IsExplicitlySet
                ? KvCacheDtypeConfig.Current.ToShortString()
                : "auto";
            _log.LogInformation(LogEventIds.ModelLoadStarted,
                "Loading model {ModelFile} on backend {Backend} kvCacheDtype={KvCacheDtype} (path={ModelPath})",
                Path.GetFileName(modelPath), backend, requestedDtype, modelPath);
            var modelLoadSw = Stopwatch.StartNew();

            // Build a distributed TP group when --tp-node-id and --tp-peers are provided.
            ITensorParallelGroup tpGroup = null;
            if (tpNodeId >= 0 && !string.IsNullOrEmpty(tpPeers))
            {
                var peerEndpoints = TensorSharp.Distributed.DistributedTpConfig.ParsePeers(tpPeers);
                int localDegree = tpDegree > 1 ? tpDegree : 1;
                // The on-node group has to match the backend: direct CUDA drives
                // CudaAllocators, the ggml backends drive per-rank ggml backends.
                tpGroup = backend is BackendType.GgmlCuda or BackendType.GgmlVulkan
                    ? new TensorSharp.Distributed.DistributedTensorParallelGroup(
                        ModelBase.CreateGgmlLocalTpGroup(backend, localDegree), tpNodeId, peerEndpoints)
                    : new TensorSharp.Distributed.DistributedTensorParallelGroup(localDegree, tpNodeId, peerEndpoints);
                tpDegree = localDegree;
            }

            ModelBase createdModel;
            try
            {
                createdModel = ModelBase.Create(modelPath, backend, tpDegree, tpGroup, draftModelPath);
            }
            catch (Exception ex) when (ModelLoadRefusal.TryDescribe(ex, out string loadRefusal))
            {
                // A refused load (not enough VRAM, an unsupported KV dtype or --tp layout,
                // a missing or invalid file) is the operator's to fix: one line and
                // HostExitCodes.ModelLoadRefused rather than an unhandled exception and
                // abort(). The stack trace stays available at --log-level debug. Main's
                // normal exit path then shuts the ggml backend down.
                _log.LogError(LogEventIds.ModelLoadFailed,
                    "Model load refused: {ModelFile} on backend {Backend}: {Reason}",
                    Path.GetFileName(modelPath), backend, loadRefusal);
                _log.LogDebug(LogEventIds.ModelLoadFailed, ex, "Model load refused: {ModelFile}", Path.GetFileName(modelPath));
                tpGroup?.Dispose();
                Console.Error.WriteLine(ModelLoadRefusal.FormatErrorLine(loadRefusal));
                Environment.ExitCode = HostExitCodes.ModelLoadRefused;
                return;
            }
            using var model = createdModel;

            // Speculator weights that ship as their own file (Gemma 4's
            // gemma4-assistant draft head, named with --draft-model) attach
            // to the target here, through the same loader the server uses. A
            // drafter is an optimization: failing to attach one warns and falls
            // back to plain decoding rather than failing the run.
            if (!SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(model, out string draftHeadError))
            {
                _log.LogWarning(LogEventIds.HostConfiguration,
                    "{Error} Speculative decoding will serve standard decoding instead.", draftHeadError);
            }

            modelLoadSw.Stop();
            _log.LogInformation(LogEventIds.ModelLoadCompleted,
                "Loaded model {ModelFile} architecture={Architecture} contextLength={ContextLength} kvCacheDtype={KvCacheDtype} elapsedMs={ElapsedMs:F1}",
                Path.GetFileName(modelPath), model.Config.Architecture ?? "(unknown)",
                model.MaxContextLength, model.KvCacheDtype.ToShortString(),
                modelLoadSw.Elapsed.TotalMilliseconds);

            // Qwen-Image-Edit: prompt + input image(s) -> modified image (no autoregressive path).
            // Repeat --image for multi-image edits (e.g. --image model.png --image dress.png);
            // the first image drives the output geometry, the prompt can reference each as
            // "Picture 1", "Picture 2", ... in listed order.
            if (model is TensorSharp.Models.QwenImage.QwenImageModel qwenImageModel)
            {
                if (imagePathList.Count == 0)
                {
                    Console.Error.WriteLine("Qwen-Image-Edit requires --image <input.png> (repeatable for multi-image edits). Optionally --prompt, --output, --diffusion-steps, --cfg, --diffusion-seed.");
                    return;
                }
                string prompt = editPrompt
                    ?? (inputFile != null && File.Exists(inputFile) ? File.ReadAllText(inputFile).Trim() : "");
                string outPath = outputFile ?? "edited.png";
                RunImageEdit(qwenImageModel, imagePathList, prompt, outPath, diffusionStepsSet ? diffusionSteps : 0, cfgScaleSet ? cfgScale : 0f, diffusionSeed, imageWidth, imageHeight);
                return;
            }

            // Video generation: prompt (+ optional conditioning) -> MP4, plus a sidecar WAV
            // on models that generate audio jointly. No autoregressive path.
            if (model is TensorSharp.Models.Video.IVideoGenerationModel videoModel)
            {
                string prompt = editPrompt
                    ?? (inputFile != null && File.Exists(inputFile) ? File.ReadAllText(inputFile).Trim() : null);
                if (string.IsNullOrWhiteSpace(prompt))
                {
                    Console.Error.WriteLine("Video generation requires --prompt \"<description>\" (or --input prompt.txt). " +
                        "Optionally --image first_frame.png, --end-image last_frame.png, " +
                        "--ref-image/--ref-video/--ref-audio (reference-conditioned models), " +
                        "--output out.mp4, --width, --height, --video-frames, --fps, --diffusion-steps, " +
                        "--cfg, --flow-shift, --negative-prompt, --diffusion-seed, --cfg-cache-stride, --no-audio.");
                    return;
                }
                RunVideoGeneration(videoModel, prompt, outputFile ?? "video.mp4",
                    imageWidth, imageHeight, videoFrames,
                    diffusionStepsSet ? diffusionSteps : 0, cfgScaleSet ? cfgScale : 0f,
                    diffusionSeedSet ? diffusionSeed : -1, flowShift, videoFps, negativePrompt,
                    videoSampler, imagePath, cfgCacheStride,
                    endImagePath, refImagePaths, refVideoPaths, refAudioPaths, videoAudioEnabled,
                    videoMode, refVideoAudioPaths);
                return;
            }

            var warmupSw = Stopwatch.StartNew();
            model.WarmUpKernels();
            warmupSw.Stop();
            _log.LogInformation(LogEventIds.HostConfiguration,
                "Kernel warmup completed in {ElapsedMs:F1} ms", warmupSw.Elapsed.TotalMilliseconds);

            // Multi-node tensor parallelism: warmup ran identically (in lockstep)
            // on every node. From here on only the driver (node 0) runs the
            // generation logic and owns sampling/IO; the other nodes become
            // workers that mirror the driver's forward passes so their weight
            // shards contribute to every AllReduce. Worker nodes never reach the
            // interactive/batch code below — they loop until the driver exits.
            if (tpGroup != null && tpGroup.NodeCount > 1)
            {
                if (model.IsDistributedWorker)
                {
                    model.RunDistributedWorkerLoop();
                    return;
                }
                model.BeginDistributedDriver();
            }

            if (mmProjPath != null)
            {
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "Loading mmproj projector from {MmProj}", mmProjPath);
                model.MultimodalInjector.LoadProjectors(mmProjPath);
            }
            else if (imagePath != null || audioPath != null || videoPath != null)
            {
                // No --mmproj: look for the family's companion projector beside the
                // model. Whether to look at all is a capability question (can this model
                // consume image or audio embeddings?), and WHICH file to look for is the
                // architecture's own business - both answered without naming a single
                // architecture here. A text-only model declares neither capability and
                // falls straight through.
                bool wantsVision = imagePath != null && model is IVisionCapableModel;
                bool wantsAudio = (audioPath != null || videoPath != null) && model is IAudioCapableModel;
                if (wantsVision || wantsAudio)
                {
                    string autoMmproj = ModelArchitectureRegistry.FindCompanionProjector(
                        model.Config.Architecture, modelPath);
                    if (autoMmproj != null)
                    {
                        _log.LogInformation(LogEventIds.HostConfiguration,
                            "Auto-loading multimodal encoder: {MmProj}", autoMmproj);
                        model.MultimodalInjector.LoadProjectors(autoMmproj);
                    }
                }
            }

            if (runTest)
            {
                RunTests(model, maxTokens, outputFile);
                return;
            }

            if (runBenchmark)
            {
                RunBenchmark(
                    model, benchmarkPrefill, benchmarkDecode, benchmarkRuns,
                    benchmarkChunked, benchmarkFixedTokens);
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
                RunMultiTurnTest(model, multiTurnJsonl, maxTokens, samplingConfig, enableThinking,
                    specDraftMax, specDraftConfMin);
                return;
            }

            if (inputJsonl != null)
            {
                RunJsonlBatch(model, inputJsonl, outputFile, maxTokens, samplingConfig, enableThinking,
                    specDraftMax, specDraftConfMin);
                return;
            }

            if (runInteractive)
            {
                _log.LogInformation(LogEventIds.CliStarted,
                    "Entering interactive chat mode (model={Model}, backend={Backend}, thinking={Thinking})",
                    Path.GetFileName(modelPath), backend, enableThinking);

                // Interactive chat must not decode greedily. An unpenalised argmax
                // over thousands of steps is the textbook neural-text-degeneration
                // setting: locally fluent for a few hundred tokens, then the argmax
                // map reaches a fixed cycle and repeats the same phrase forever.
                // (The 512-token default budget used to hide it; --max-tokens 20000
                // removes that backstop.) Batch / benchmark entry points below keep
                // SamplingConfig.Greedy so they stay bit-reproducible.
                ResolveChatSamplingDefaults(samplingConfig, ref pinnedSampling, model);

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
                    _log,
                    specDraftMax,
                    specDraftConfMin,
                    skillRegistry,
                    skillOptions,
                    selectedSkills,
                    codeRunner,
                    codeWorkspace);
                if (!string.IsNullOrEmpty(systemPrompt))
                    session.SetInitialSystemPrompt(systemPrompt);
                session.PrefixCacheEnabled = !noPrefixCache;
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

            // PDF document input. A born-digital PDF has a selectable text layer: extract
            // it and inline it into the user message so the model reasons over it through
            // the normal (already optimized) prefill path, preserving every extracted
            // page. A scanned / image-only PDF has NO text layer, so instead we recover
            // its page images and feed them to a vision model (mirroring the video -> frames
            // path); if no vision model is loaded we fail with a clear, actionable message
            // rather than silently sending an empty document. When --input is also given it
            // becomes the instruction over the document; otherwise a default is used.
            List<string> pdfPageImages = null;
            if (pdfPath != null)
            {
                if (!File.Exists(pdfPath))
                {
                    _log.LogError(LogEventIds.CliFailed, "PDF file not found: {PdfPath}", pdfPath);
                    return;
                }

                var pdfSw = Stopwatch.StartNew();
                TensorSharp.Models.PdfTextResult pdf;
                try
                {
                    pdf = TensorSharp.Models.PdfTextExtractor.ExtractFromFile(pdfPath, ResolvePdfMaxPages());
                }
                catch (Exception ex)
                {
                    _log.LogError(LogEventIds.CliFailed, ex,
                        "Failed to extract text from PDF {PdfPath}: {Error}", pdfPath, ex.Message);
                    Console.Error.WriteLine("Could not read the PDF: " + ex.Message);
                    return;
                }
                pdfSw.Stop();
                string pdfName = Path.GetFileName(pdfPath);

                if (!pdf.LooksTextless)
                {
                    // Born-digital PDF: inline the extracted text.
                    string docText = pdf.Text ?? string.Empty;
                    bool allPagesExtracted = pdf.ExtractedPageCount == pdf.PageCount;

                    string instruction = hasUserInput
                        ? rawText
                        : "Please analyze the attached PDF document and summarize its content.";
                    rawText = $"[File: {pdfName}]\n{docText}\n[End of file]\n\n{instruction}";

                    _log.LogInformation(LogEventIds.UploadReceived,
                        "PDF input (untruncated text): {PdfPath} pages={Pages} extractedPages={ExtractedPages} complete={Complete} chars={Chars} extractMs={ExtractMs:F1}",
                        pdfPath, pdf.PageCount, pdf.ExtractedPageCount, allPagesExtracted, docText.Length,
                        pdfSw.Elapsed.TotalMilliseconds);
                    if (allPagesExtracted)
                    {
                        Console.WriteLine(
                            $"Loaded PDF in full: {pdfName} ({pdf.ExtractedPageCount}/{pdf.PageCount} pages, {docText.Length} chars)");
                    }
                    else
                    {
                        Console.Error.WriteLine(
                            $"Warning: only {pdf.ExtractedPageCount}/{pdf.PageCount} PDF pages could be read; " +
                            "the extracted pages were preserved without token truncation.");
                    }
                }
                else
                {
                    // Scanned / image-only PDF: needs a vision model to read the page images.
                    bool visionAvailable = mmProjPath != null || model.HasVisionEncoder();
                    if (!visionAvailable)
                    {
                        _log.LogError(LogEventIds.CliFailed,
                            "PDF has no text layer and no vision model is loaded: {PdfPath} pages={Pages}",
                            pdfPath, pdf.PageCount);
                        Console.Error.WriteLine(
                            $"PDF \"{pdfName}\" has no selectable text (it appears to be scanned or image-only). " +
                            "Re-run with a vision-capable model and its projector, e.g. --mmproj <projector.gguf>, " +
                            "so its pages can be read as images.");
                        return;
                    }

                    string pdfImgDir = Path.Combine(Path.GetTempPath(), $"pdfimg_{Guid.NewGuid():N}");
                    TensorSharp.Models.PdfImageResult imgRes;
                    try
                    {
                        imgRes = TensorSharp.Models.PdfPageImageExtractor.ExtractPageImages(
                            pdfPath, pdfImgDir, ResolvePdfMaxPages(), pdfName);
                    }
                    catch (Exception ex)
                    {
                        _log.LogError(LogEventIds.CliFailed, ex,
                            "Failed to extract page images from PDF {PdfPath}: {Error}", pdfPath, ex.Message);
                        Console.Error.WriteLine("Could not read the PDF: " + ex.Message);
                        return;
                    }

                    if (imgRes.ImagePaths.Count == 0)
                    {
                        _log.LogError(LogEventIds.CliFailed,
                            "PDF yielded neither text nor images: {PdfPath}", pdfPath);
                        Console.Error.WriteLine($"Could not extract any text or images from \"{pdfName}\".");
                        return;
                    }

                    pdfPageImages = new List<string>(imgRes.ImagePaths);
                    if (!hasUserInput)
                        rawText = "Please analyze and interpret the attached document pages.";

                    _log.LogInformation(LogEventIds.UploadReceived,
                        "PDF input (images): {PdfPath} pages={Pages} images={Images} extractMs={ExtractMs:F1}",
                        pdfPath, pdf.PageCount, imgRes.ImagePaths.Count, pdfSw.Elapsed.TotalMilliseconds);
                    Console.WriteLine($"Loaded PDF as images: {pdfName} " +
                        $"({imgRes.ExtractedPageCount}/{pdf.PageCount} pages -> {imgRes.ImagePaths.Count} image(s) for the vision model)");
                }
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
                // Every --image in order; imagePath alone would keep only the last.
                imagePaths = imagePathList.Count > 0
                    ? new List<string>(imagePathList)
                    : new List<string> { imagePath };
                foreach (string ip in imagePaths)
                {
                    if (!File.Exists(ip))
                    {
                        _log.LogError(LogEventIds.CliFailed, "Image file not found: {ImagePath}", ip);
                        return;
                    }
                }
                if (!hasUserInput)
                    rawText = imagePaths.Count > 1
                        ? "What is in these images? Please describe each."
                        : "What is in this image? Please describe it.";
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
                // Same gate as the server's parsers: a family with no audio tower
                // (DeepSeek V4.1, Nemotron-H) is refused here rather than decoding
                // the clip and then generating as if none had been given.
                if (AudioInputSupport.UnsupportedReasonFor(model) is string audioError)
                {
                    _log.LogError(LogEventIds.CliFailed, "--audio rejected: {Reason}", audioError);
                    return;
                }
                audioPaths = new List<string> { audioPath };
                if (!hasUserInput)
                    rawText = "Listen to this audio and describe what you hear.";
                _log.LogInformation(LogEventIds.UploadReceived,
                    "Audio input: {AudioPath} ({Bytes})",
                    audioPath, LoggingExtensions.FormatBytes(new FileInfo(audioPath).Length));
            }

            // A scanned / image-only PDF was recovered as page images above; attach them as
            // image inputs so the vision model reads the document. Page images take
            // precedence over any --image/--video (combining those with --pdf is unusual).
            if (pdfPageImages != null && pdfPageImages.Count > 0)
                imagePaths = pdfPageImages;

            // Now that the model is loaded its context length is known, so the skills
            // block can be budgeted against it rather than against a guess, and the
            // family's ability to carry tool declarations can be consulted: Mistral 3
            // discards them, so there the instructions are written into the
            // prompt up front instead of being fetched on demand.
            // The operator's OWN --tools, before anything of ours is merged in. The loop
            // needs the two apart: a name in neither list belongs to nobody, and telling
            // the model so beats ending its turn on a guess.
            List<ToolFunction> clientTools = tools != null ? new List<ToolFunction>(tools) : null;

            if (skillRegistry != null && (selectedSkills.Count > 0 || (skillOptions.Discovery && skillRegistry.Skills.Count > 0)))
            {
                var skillCaps = SkillCapabilities.For(model.Config.Architecture);
                var catalogSkills = skillOptions.Discovery
                    ? skillRegistry.Skills
                    : (IReadOnlyList<Skill>)Array.Empty<Skill>();

                skillPlan = SkillPrompt.Plan(selectedSkills, catalogSkills, new SkillPromptOptions
                {
                    ContextTokens = model.MaxContextLength,
                    ToolsAvailable = skillCaps.ToolsRendered,
                });

                if (!skillPlan.IsEmpty)
                {
                    systemPrompt = string.IsNullOrWhiteSpace(systemPrompt)
                        ? skillPlan.Instructions
                        : systemPrompt.TrimEnd() + "\n\n" + skillPlan.Instructions;

                    if (skillCaps.ToolsRendered)
                    {
                        tools = SkillTools.Merge(tools, skillScriptsAllowed, out var shadowedTools);
                        foreach (string shadowed in shadowedTools)
                        {
                            _log.LogWarning(LogEventIds.HostConfiguration,
                                "cli.skills.tool-shadowed name={ToolName} - your --tools definition wins", shadowed);
                        }
                        if (codeRunner != null)
                            tools = AppendCodeTool(tools, codeRunner, codeWorkspace);

                        skillToolContext = new SkillToolContext(new List<Skill>(skillPlan.Reachable))
                        {
                            ScriptRunner = skillScriptsAllowed
                                ? new SkillScriptRunner(ToScriptRunnerOptions(
                                    skillOptions, codeWorkspace, codeRunner), _log)
                                : null,
                            CodeRunner = codeRunner,
                            Workspace = codeWorkspace,
                        };
                    }

                    _log.LogInformation(LogEventIds.SkillSelected,
                        "cli.skills.ready selected={Selected} announced={Announced} inlined={Inlined} catalog={Catalog} tools={ToolsOffered} promptTokens~{PromptTokens}",
                        selectedSkills.Count == 0 ? "(none)" : string.Join(",", selectedSkills.Select(sk => sk.Id)),
                        skillPlan.Deferred.Count, skillPlan.Inlined.Count, skillPlan.Catalog.Count, skillCaps.ToolsRendered,
                        skillPlan.ApproximateTokens);
                }
                else
                {
                    skillPlan = null;
                }
            }

            // Code execution does not require skills: --code-exec with no skill selected
            // (and no registry at all) must still offer the shell tool, the way the server's
            // code-only plan does. Without this the flag looked accepted and did nothing.
            if (codeRunner != null && skillToolContext == null
                && SkillCapabilities.For(model.Config.Architecture).ToolsRendered)
            {
                tools = AppendCodeTool(tools, codeRunner, codeWorkspace);
                skillToolContext = new SkillToolContext(new List<Skill>())
                {
                    CodeRunner = codeRunner,
                    Workspace = codeWorkspace,
                };
            }

            // The editing rules, after every path that may have declared the code tools —
            // with skills and without. They were injected only by the SERVER's plan, so a
            // one-shot CLI run was declared all five code tools and told nothing about
            // which to reach for, which is the exact condition the measurement blamed for
            // models re-typing whole files.
            if (CodeSystemBlock(tools) is { Length: > 0 } editingRules)
            {
                systemPrompt = string.IsNullOrWhiteSpace(systemPrompt)
                    ? editingRules
                    : systemPrompt.TrimEnd() + "\n\n" + editingRules;
            }

            if (dumpPrompt)
            {
                var dumpMessages = new List<ChatMessage>();
                if (!string.IsNullOrWhiteSpace(systemPrompt))
                    dumpMessages.Add(new ChatMessage { Role = "system", Content = systemPrompt });
                dumpMessages.Add(new ChatMessage { Role = "user", Content = rawText });
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
            using var inference = new CliInferenceSession(model,
                SchedulerConfig.FromEnvironment().WithSpeculation(
                    SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin)), _log);
            for (int w = 0; w < warmupInferenceRuns; w++)
            {
                int warmupDecodeTokens = Math.Min(maxTokens, 4);
                _log.LogInformation(LogEventIds.HostConfiguration,
                    "cli.inference.warmup run={Run}/{Total} decodeTokens={DecodeTokens}",
                    w + 1, warmupInferenceRuns, warmupDecodeTokens);
                _ = RunInference(model, rawText, imagePaths, warmupDecodeTokens, audioPaths,
                    isVideo: videoPath != null, samplingConfig: samplingConfig,
                    enableThinking: enableThinking, tools: tools, silent: true,
                    preserveAllInput: pdfPath != null,
                    specDraftMax: specDraftMax, specDraftConfMin: specDraftConfMin,
                    systemPrompt: systemPrompt, inference: inference);
                inference.StartNewConversation();
                model.ResetForwardTiming();
            }

            // With skills in play the model may answer by asking to read one first. Those
            // reads are served here, in process, and the loop re-enters — so a single-shot
            // run still ends in an answer rather than a printed tool call the user would
            // have to service by hand.
            string result;
            if (skillToolContext != null)
            {
                result = RunInferenceWithSkills(
                    model, rawText, imagePaths, maxTokens, audioPaths, videoPath != null,
                    samplingConfig, enableThinking, tools, pdfPath != null,
                    specDraftMax, specDraftConfMin, systemPrompt,
                    skillToolContext, skillOptions.RoundsFor(skillToolContext.CodeRunner is { CanRun: true }),
                    SkillCapabilities.For(model.Config.Architecture).ToolResultsRendered,
                    clientTools, inference);
            }
            else
            {
                result = RunInference(model, rawText, imagePaths, maxTokens, audioPaths,
                    isVideo: videoPath != null, samplingConfig: samplingConfig,
                    enableThinking: enableThinking, tools: tools,
                    preserveAllInput: pdfPath != null,
                    specDraftMax: specDraftMax, specDraftConfMin: specDraftConfMin,
                    systemPrompt: systemPrompt, inference: inference);
            }

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
            SamplingConfig sampling, bool enableThinking,
            int specDraftMax = 0, float specDraftConfMin = -1f)
        {
            if (!File.Exists(jsonlPath))
            {
                _log.LogError(LogEventIds.CliFailed, "Multi-turn jsonl not found: {File}", jsonlPath);
                return;
            }

            var specSettings = SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin);

            using var inference = new CliInferenceSession(model,
                SchedulerConfig.FromEnvironment().WithSpeculation(specSettings), _log);

            string[] lines = File.ReadAllLines(jsonlPath);
            var history = new List<ChatMessage>();
            string arch = model.Config.Architecture;
            int swa = model.Config.SlidingWindow;

            // Preserve raw assistant token splicing while the engine owns KV state.
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
                List<string> turnImages = null;
                try
                {
                    var doc = JsonDocument.Parse(line);
                    var root = doc.RootElement;
                    // "content" is the documented key; accept "user" as an alias
                    // because the object shape invites it. Falling through to the
                    // raw line for a well-formed object silently feeds the model
                    // the JSON itself, which reads as a plausible-but-wrong turn
                    // and makes the run look like a model failure.
                    userMsg =
                        root.TryGetProperty("content", out var c) ? c.GetString() :
                        root.TryGetProperty("user", out var u) ? u.GetString() : line;
                    if (root.TryGetProperty("max_tokens", out var mt))
                        turnMaxTokens = mt.GetInt32();
                    if (root.TryGetProperty("force_reset", out var fr))
                        forceReset = fr.GetBoolean();
                    if (root.TryGetProperty("images", out var imgs) && imgs.ValueKind == JsonValueKind.Array)
                    {
                        turnImages = new List<string>();
                        foreach (var im in imgs.EnumerateArray())
                        {
                            string ip = im.GetString();
                            if (!string.IsNullOrEmpty(ip)) turnImages.Add(ip);
                        }
                    }
                }
                catch
                {
                    userMsg = line;
                }

                history.Add(new ChatMessage { Role = "user", Content = userMsg, ImagePaths = turnImages });
                _log.LogInformation(LogEventIds.ChatStarted,
                    "multi-turn turn={Turn}/{TotalTurns} user=\"{User}\"",
                    turn + 1, lines.Length, LoggingExtensions.SanitizeForLog(userMsg));

                if (forceReset)
                {
                    _log.LogInformation(LogEventIds.SessionReset,
                        "multi-turn forcing KV cache reset (per JSONL force_reset flag)");
                    inference.Reset();
                }

                var inputTokens = renderer.RenderToTokens(
                    model.Tokenizer,
                    model.Config.ChatTemplate,
                    history,
                    arch,
                    addGenerationPrompt: true,
                    out _,
                    out string generationPromptTrailingWhitespace,
                    enableThinking: enableThinking);

                string requestId = $"cli-multiturn-{Guid.NewGuid():N}";
                CliInferenceSession.Result result;
                try
                {
                    inputTokens = model.MultimodalInjector.ProcessPromptTokens(history, inputTokens, requestId);
                    result = inference.Generate(inputTokens, turnMaxTokens, sampling ?? SamplingConfig.Greedy,
                        requestId: requestId, mediaSpans: model.MultimodalInjector.GetPreparedMediaSpans(requestId));
                }
                finally
                {
                    model.MultimodalInjector.ClearPreparedPromptState(requestId);
                }
                var generatedTokens = result.Tokens;
                double prefillMs = result.PrefillMs;
                double decodeMs = result.DecodeMs;
                _log.LogInformation(LogEventIds.KvCacheReusePlan,
                    "radix prefix reused={ReusedTokens}/{PromptTokens} prefillMs={PrefillMs:F1}",
                    result.Completion.PrefixCacheReusedTokens, inputTokens.Count, prefillMs);
                string rawOutput = model.Tokenizer.Decode(generatedTokens);

                var parser = CliOutputParser.Create(arch, enableThinking, null,
                    model.Tokenizer, inputTokens);
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
                    RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
                });
            }

            _log.LogInformation(LogEventIds.CliCompleted,
                "multi-turn test completed: {Turns} turns", history.Count / 2);
        }

        static void RunJsonlBatch(ModelBase model, string inputJsonlPath, string outputFile, int defaultMaxTokens,
            SamplingConfig defaultSampling, bool enableThinking = false,
            int specDraftMax = 0, float specDraftConfMin = -1f)
        {
            if (!File.Exists(inputJsonlPath))
            {
                _log.LogError(LogEventIds.CliFailed, "JSONL file not found: {File}", inputJsonlPath);
                return;
            }

            var specSettings = SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin);
            using var inference = new CliInferenceSession(model,
                SchedulerConfig.FromEnvironment().WithSpeculation(specSettings), _log);

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
                    // Each JSONL line is an independent conversation. Only its
                    // declared system prefix may reuse another line's cache.
                    inference.StartNewConversation();
                    var messages = ParseMessages(root);
                    int maxTokens = root.TryGetProperty("max_tokens", out var mt) ? mt.GetInt32() : defaultMaxTokens;
                    var sampling = ParseSamplingFromJson(root, defaultSampling);

                    var imagePaths = ParseStringList(root, "images");
                    var audioPaths = ParseStringList(root, "audios");
                    bool isVideo = root.TryGetProperty("is_video", out var iv) && iv.GetBoolean();

                    var lastUser = messages.LastOrDefault(m => m.Role == "user");
                    if (lastUser != null)
                    {
                        lastUser.ImagePaths = imagePaths ?? lastUser.ImagePaths;
                        lastUser.AudioPaths = audioPaths ?? lastUser.AudioPaths;
                        lastUser.IsVideo |= isVideo;
                    }

                    bool reqThinking = enableThinking ||
                        (root.TryGetProperty("enable_thinking", out var etProp) && etProp.GetBoolean());

                    string rendered = PromptRenderer.Render(
                        model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                        architecture: model.Config.Architecture, enableThinking: reqThinking);

                    _log.LogDebug(LogEventIds.ChatStarted,
                        "jsonl batch [{RequestId}] rendered prompt thinking={Thinking} preview={Preview}",
                        id, reqThinking, LoggingExtensions.SanitizeForLog(rendered, maxLength: 320));

                    var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);
                    var sharedPrefix = ComputeSharedPrefix(model, messages, reqThinking);
                    _log.LogDebug(LogEventIds.ChatStarted,
                        "jsonl batch [{RequestId}] inputTokens={TokenCount} first20=[{First20}]",
                        id, inputTokens.Count, string.Join(", ", inputTokens.Take(20)));

                    var cfg = sampling ?? SamplingConfig.Greedy;
                    string requestId = $"cli-batch-{Guid.NewGuid():N}";
                    CliInferenceSession.Result result;
                    try
                    {
                        inputTokens = model.MultimodalInjector.ProcessPromptTokens(messages, inputTokens, requestId);
                        var stopSampler = new TokenSampler(cfg);
                        var streamed = new List<int>();
                        result = inference.Generate(inputTokens, maxTokens, cfg, token =>
                        {
                            streamed.Add(token);
                            return cfg.StopSequences == null || cfg.StopSequences.Count == 0
                                || !stopSampler.CheckStopSequences(model.Tokenizer.Decode(streamed)).shouldStop;
                        }, requestId: requestId, mediaSpans: model.MultimodalInjector.GetPreparedMediaSpans(requestId),
                            sharedPrefixTokens: CliSharedPrefix.MatchingLength(sharedPrefix, inputTokens));
                    }
                    finally
                    {
                        model.MultimodalInjector.ClearPreparedPromptState(requestId);
                    }
                    var generatedTokens = result.Tokens;
                    double prefillMs = result.PrefillMs;
                    double totalMs = result.TotalMs;
                    string output = model.Tokenizer.Decode(generatedTokens);
                    if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                        (output, _) = new TokenSampler(cfg).CheckStopSequences(output);
                    double tokPerSec = generatedTokens.Count / Math.Max(totalMs / 1000.0, 1e-9);

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

        /// <summary>
        /// Optional cap on the number of PDF pages read, from the <c>TS_PDF_MAX_PAGES</c>
        /// environment variable. Returns 0 (all pages) when unset or invalid.
        /// </summary>
        static int ResolvePdfMaxPages()
        {
            string raw = Environment.GetEnvironmentVariable("TS_PDF_MAX_PAGES");
            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int v) && v > 0)
                return v;
            return 0;
        }

        /// <summary>
        /// Resolve the sampling chain an interactive chat turn should use, in the
        /// same precedence order llama.cpp uses (common/common.cpp): built-in chat
        /// defaults, then the model's own <c>general.sampling.*</c> recommendations
        /// from the GGUF, then whatever the operator pinned on the command line.
        ///
        /// The CLI previously handed <see cref="SamplingConfig.Greedy"/> straight to
        /// the chat loop. Greedy decoding with every penalty disabled is why long
        /// answers degenerated into an endless repetition of the same phrase: the
        /// argmax map has fixed cycles, and nothing in the decode loop can leave one.
        /// Passing <c>--temperature 0</c> restores the old behaviour (and with it the
        /// argmax-keyed fast paths: pipelined greedy decode and MTP block-speculative
        /// decoding, both of which require a pure-argmax sampler).
        /// </summary>
        internal static void ResolveChatSamplingDefaults(
            SamplingConfig cfg, ref SamplingFields pinned, IModelArchitecture model)
        {
            if (cfg == null) return;

            // 1. Chat defaults for everything the operator did not pin. These match
            //    llama.cpp's default sampler chain (common/common.h) rather than the
            //    CLI's historical greedy config.
            if (!pinned.HasFlag(SamplingFields.Temperature)) cfg.Temperature = 0.8f;
            if (!pinned.HasFlag(SamplingFields.TopK)) cfg.TopK = 40;
            if (!pinned.HasFlag(SamplingFields.TopP)) cfg.TopP = 0.95f;
            if (!pinned.HasFlag(SamplingFields.MinP)) cfg.MinP = 0.05f;
            if (!pinned.HasFlag(SamplingFields.PenaltyLastN)) cfg.PenaltyLastN = 64;
            // NOTE: RepetitionPenalty is deliberately NOT defaulted here. The chat
            // config is seeded from SamplingConfig.Greedy, which sets it to 1.0
            // (disabled), and PenaltyLastN=64 above therefore penalises nothing -
            // which looks like an oversight but is exactly llama.cpp's default pair
            // (common/common.h: penalty_last_n=64, penalty_repeat=1.0), and matching
            // that chain is this method's stated contract. A 1.1 default was tried
            // while chasing an endless-repetition report on Qwen3.8-Flash-Next; the
            // real cause turned out to be a mid-sequence fused-path fallback that
            // reset the recurrent state (see Qwen4ExpModel.WarnIfQsaBudgetExceeded),
            // and once that was fixed the seed that looped no longer did. Operators
            // who want a penalty pass --repeat-penalty; a GGUF can also ask for one
            // via general.sampling.penalty_repeat, applied just below.

            // 2. The model's own recommendation wins over our generic defaults.
            string fromModel = model?.Config?.RecommendedSampling?.ApplyTo(cfg, pinned) ?? string.Empty;

            string summary =
                $"temp={cfg.Temperature.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"top_k={cfg.TopK} " +
                $"top_p={cfg.TopP.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"min_p={cfg.MinP.ToString("0.###", CultureInfo.InvariantCulture)} " +
                $"repeat_penalty={cfg.RepetitionPenalty.ToString("0.###", CultureInfo.InvariantCulture)}";
            Console.WriteLine(fromModel.Length > 0
                ? $"  Chat sampling: {summary} (GGUF general.sampling.*: {fromModel})"
                : $"  Chat sampling: {summary}");
            if (cfg.IsGreedy)
                Console.WriteLine("  NOTE: sampling resolved to greedy; long answers can repeat endlessly.");
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
            IReadOnlyList<string> imagePaths, string prompt, string outputPath, int steps, float cfgScale, int seed,
            int width = 0, int height = 0)
        {
            foreach (var path in imagePaths)
            {
                if (!File.Exists(path))
                {
                    Console.Error.WriteLine($"Input image not found: {path}");
                    return;
                }
            }
            Console.WriteLine($"=== Qwen-Image-Edit ===");
            for (int i = 0; i < imagePaths.Count; i++)
                Console.WriteLine($"  input{(imagePaths.Count > 1 ? $" {i + 1}" : "  ")}: {imagePaths[i]}");
            Console.WriteLine($"  prompt : {prompt}");
            Console.WriteLine($"  steps={steps} cfg={cfgScale} seed={seed} -> {outputPath}");

            var inputs = new List<TensorSharp.Models.QwenImage.RgbImage>();
            foreach (var path in imagePaths)
                inputs.Add(TensorSharp.Models.QwenImage.ImageIO.Load(path));
            var p = new TensorSharp.Models.QwenImage.QwenImageParams
            {
                Steps = steps,
                CfgScale = cfgScale,
                Seed = seed,
                Width = width,
                Height = height,
            };
            if (width > 0 && height > 0)
                Console.WriteLine($"  explicit output size {width}x{height} (bypasses the VRAM area clamp)");
            var sw = Stopwatch.StartNew();
            var output = model.EditImage(prompt, inputs, p);
            sw.Stop();
            TensorSharp.Models.QwenImage.ImageIO.SavePng(outputPath, output);
            Console.WriteLine($"Saved {output.Width}x{output.Height} edited image to {outputPath} " +
                $"({sw.Elapsed.TotalSeconds:F1}s, {sw.Elapsed.TotalMilliseconds / Math.Max(1, steps):F0} ms/step)");
        }

        static void RunVideoGeneration(TensorSharp.Models.Video.IVideoGenerationModel model,
            string prompt, string outputPath, int width, int height, int frames,
            int steps, float cfgScale, int seed, float flowShift, int fps, string negativePrompt,
            string sampler = null, string imagePath = null, int cfgCacheStride = 0,
            string endImagePath = null, IList<string> refImages = null, IList<string> refVideos = null,
            IList<string> refAudios = null, bool generateAudio = true, string videoMode = null,
            IList<string> refVideoAudios = null)
        {
            Console.WriteLine(imagePath != null ? "=== Image-to-Video ===" : "=== Text-to-Video ===");
            Console.WriteLine($"  prompt : {prompt}");
            if (imagePath != null)
            {
                if (!File.Exists(imagePath))
                {
                    Console.Error.WriteLine($"Conditioning image not found: {imagePath}");
                    return;
                }
                Console.WriteLine($"  image  : {imagePath}");
            }
            if (endImagePath != null)
            {
                if (!File.Exists(endImagePath))
                {
                    Console.Error.WriteLine($"End-frame conditioning image not found: {endImagePath}");
                    return;
                }
                Console.WriteLine($"  end    : {endImagePath}");
            }
            foreach (var r in refImages ?? (IList<string>)Array.Empty<string>())
                Console.WriteLine($"  ref-img: {r}");
            foreach (var r in refVideos ?? (IList<string>)Array.Empty<string>())
                Console.WriteLine($"  ref-vid: {r}");
            foreach (var r in refAudios ?? (IList<string>)Array.Empty<string>())
                Console.WriteLine($"  ref-aud: {r}");
            foreach (var r in refVideoAudios ?? (IList<string>)Array.Empty<string>())
                Console.WriteLine($"  ref-vaud: {r}");
            Console.WriteLine($"  -> {outputPath}");

            var p = new TensorSharp.Models.Video.VideoGenerationParams
            {
                Width = width,
                Height = height,
                Frames = frames,
                Steps = steps,
                CfgScale = cfgScale,
                Seed = seed,
                FlowShift = flowShift,
                Fps = fps,
                NegativePrompt = negativePrompt,
                Sampler = sampler,
                ImagePath = imagePath,
                CfgCacheStride = cfgCacheStride,
                EndImagePath = endImagePath,
                ReferenceImagePaths = refImages,
                ReferenceVideoPaths = refVideos,
                ReferenceAudioPaths = refAudios,
                ReferenceVideoAudioPaths = refVideoAudios,
                GenerateAudio = generateAudio,
                Mode = videoMode,
            };
            var sw = Stopwatch.StartNew();
            var video = model.GenerateVideo(prompt, p);
            sw.Stop();
            string codec;
            if (outputPath.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            {
                // Single-frame (text-to-image) convenience: --output out.png writes the
                // first frame as a PNG (additional frames as out_00001.png, ...).
                TensorSharp.Models.QwenImage.ImageIO.SavePng(outputPath, video.Frames[0]);
                for (int i = 1; i < video.Frames.Length; i++)
                    TensorSharp.Models.QwenImage.ImageIO.SavePng(
                        Path.ChangeExtension(outputPath, null) + $"_{i:D5}.png", video.Frames[i]);
                codec = "png";
            }
            else
            {
                codec = TensorSharp.Models.WanVideo.VideoIO.SaveMp4(outputPath, video.Frames, video.Fps);
            }
            Console.WriteLine($"Saved {video.Frames[0].Width}x{video.Frames[0].Height} x {video.Frames.Length} frames " +
                $"({codec}, {video.Fps} fps, seed {video.Seed}) to {outputPath} in {sw.Elapsed.TotalSeconds:F1}s");

            // Models that generate audio jointly return a track alongside the frames. It is
            // written as a sidecar WAV rather than muxed: muxing needs an encoder we cannot
            // assume is installed, and a sidecar is trivially muxable later with
            //   ffmpeg -i out.mp4 -i out.wav -c:v copy -c:a aac out_av.mp4
            if (video.Audio is { ChannelCount: > 0, SampleCount: > 0 })
            {
                string wavPath = Path.ChangeExtension(outputPath, ".wav");
                TensorSharp.Models.Video.WavWriter.Write(wavPath, video.Audio.Channels, video.Audio.SampleRate);
                Console.WriteLine($"Saved {video.Audio.ChannelCount}-channel " +
                    $"{video.Audio.SampleRate} Hz audio ({video.Audio.DurationSeconds:F2}s) to {wavPath}");
            }
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

        /// <summary>
        /// Run skill/tool rounds on one engine, reusing their shared radix prefix and
        /// splicing assistant token IDs into each subsequent prompt.
        /// </summary>
        static string RunInferenceWithSkills(
            ModelBase model, string rawText, List<string> imagePaths, int maxTokens,
            List<string> audioPaths, bool isVideo, SamplingConfig samplingConfig,
            bool enableThinking, List<ToolFunction> tools, bool preserveAllInput,
            int specDraftMax, float specDraftConfMin, string systemPrompt,
            SkillToolContext skillContext, int maxRounds, bool toolResultsRendered,
            List<ToolFunction> clientTools = null, CliInferenceSession inference = null)
        {
            var specSettings = SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin);
            using var ownedInference = inference == null
                ? new CliInferenceSession(model,
                    SchedulerConfig.FromEnvironment().WithSpeculation(specSettings), _log) : null;
            inference ??= ownedInference;
            var priorTurns = new List<ChatMessage>();
            string result = string.Empty;

            for (int round = 1; round <= Math.Max(1, maxRounds); round++)
            {
                ParsedOutput parsed = null;
                List<int> rawTokens = null;
                string rawTrailingWhitespace = null;
                result = RunInference(model, rawText, imagePaths, maxTokens, audioPaths,
                    isVideo: isVideo, samplingConfig: samplingConfig,
                    enableThinking: enableThinking, tools: tools,
                    preserveAllInput: preserveAllInput,
                    specDraftMax: specDraftMax, specDraftConfMin: specDraftConfMin,
                    systemPrompt: systemPrompt,
                    priorTurns: priorTurns.Count > 0 ? priorTurns : null,
                    onParsed: p => parsed = p, inference: inference,
                    onGenerated: (tokens, whitespace) => { rawTokens = tokens; rawTrailingWhitespace = whitespace; });

                // Three ways, so a name nobody declared is answered here rather than
                // dropped: returning at this point handed the operator whatever prose
                // preceded the call, which for a reasoning model is nothing at all.
                SkillTools.Partition(
                    parsed?.ToolCalls, clientTools,
                    out var builtInCalls, out _, out var unknownCalls);
                if (builtInCalls.Count == 0 && unknownCalls.Count == 0)
                    return result;

                var calls = parsed.ToolCalls;

                priorTurns.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = parsed.Content ?? string.Empty,
                    Thinking = string.IsNullOrEmpty(parsed.Thinking) ? null : parsed.Thinking,
                    ToolCalls = new List<ToolCall>(calls),
                    RawOutputTokens = rawTokens,
                    RawPromptTrailingWhitespace = rawTrailingWhitespace,
                });

                foreach (var call in unknownCalls)
                {
                    _log.LogWarning(LogEventIds.SkillToolInvoked,
                        "cli.skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                        round, call.Name ?? "-", "-", "-", false, 0);
                    Console.Error.WriteLine($"[tool call] {call.Name} (no such tool)");
                    priorTurns.Add(BuildSkillResultMessage(
                        toolResultsRendered,
                        SkillTools.DescribeUnknownTool(
                            call.Name,
                            tools,
                            skillContext?.Reachable.Select(skill => skill.Id).ToList()),
                        call.Name));
                }

                foreach (var call in builtInCalls)
                {
                    var toolResult = SkillTools.Execute(call, skillContext);
                    _log.LogInformation(LogEventIds.SkillToolInvoked,
                        "cli.skills.tool round={Round} tool={Tool} skill={SkillId} path={Path} ok={Ok} bytes={Bytes}",
                        round, call.Name, toolResult.SkillId ?? "-", toolResult.ResourcePath ?? "-",
                        toolResult.Ok, toolResult.Content?.Length ?? 0);
                    Console.Error.WriteLine(
                        $"[skill] {call.Name} {toolResult.SkillId ?? "?"} {toolResult.ResourcePath ?? string.Empty}"
                        + (toolResult.Ok ? string.Empty : " (failed)"));
                    priorTurns.Add(BuildSkillResultMessage(toolResultsRendered, toolResult.Content, call.Name));
                }

            }

            _log.LogWarning(LogEventIds.SkillLoopCapped,
                "cli.skills.loop.capped rounds={Rounds}", maxRounds);
            priorTurns.Add(BuildSkillResultMessage(toolResultsRendered,
                "Error: the limit on skill lookups for this turn has been reached. Answer now using what you "
                + "have already read, and say which part you could not check.", null));
            return RunInference(model, rawText, imagePaths, maxTokens, audioPaths,
                isVideo: isVideo, samplingConfig: samplingConfig,
                enableThinking: enableThinking, tools: tools,
                preserveAllInput: preserveAllInput,
                specDraftMax: specDraftMax, specDraftConfMin: specDraftConfMin,
                systemPrompt: systemPrompt, priorTurns: priorTurns, inference: inference);
        }

        /// <summary>
        /// Wrap a skill tool result in the message shape this family renders. Mistral 3
        /// drops <c>role: "tool"</c> messages outright, so there the result is fed back
        /// as a user turn rather than vanishing from the prompt.
        /// </summary>
        static ChatMessage BuildSkillResultMessage(bool toolResultsRendered, string content, string tool)
        {
            if (toolResultsRendered)
                return new ChatMessage { Role = "tool", Content = content };

            string prefix = tool == null
                ? "Result of the skill lookup you requested:"
                : $"Result of your {tool} call:";
            return new ChatMessage { Role = "user", Content = prefix + "\n\n" + content };
        }

        /// <param name="systemPrompt">
        /// Rendered as a leading system turn. Until Agent Skills needed a system channel
        /// here, this path built a ONE-MESSAGE list containing only the user turn, so
        /// <c>--system</c> and <c>--system-file</c> were silently discarded on every
        /// single-shot run and only took effect in interactive chat.
        /// </param>
        /// <param name="priorTurns">
        /// Turns to append after the user message — the assistant/tool exchanges an
        /// Agent Skills lookup has already produced. Null for an ordinary single-shot run.
        /// </param>
        /// <param name="onParsed">
        /// Receives the final parse, so a caller can inspect tool calls without this
        /// method's string return type changing under every existing caller.
        /// </param>
        static string RunInference(ModelBase model, string rawText, List<string> imagePaths, int maxTokens,
            List<string> audioPaths = null, bool isVideo = false, SamplingConfig samplingConfig = null,
            bool enableThinking = false, List<ToolFunction> tools = null, bool silent = false,
            bool preserveAllInput = false, int specDraftMax = 0, float specDraftConfMin = -1f,
            string systemPrompt = null, List<ChatMessage> priorTurns = null,
            Action<ParsedOutput> onParsed = null, CliInferenceSession inference = null,
            Action<List<int>, string> onGenerated = null)
        {
            var messages = new List<ChatMessage>();
            if (!string.IsNullOrWhiteSpace(systemPrompt))
                messages.Add(new ChatMessage { Role = "system", Content = systemPrompt });
            messages.Add(new ChatMessage { Role = "user", Content = rawText, ImagePaths = imagePaths, AudioPaths = audioPaths, IsVideo = isVideo });
            if (priorTurns != null)
                messages.AddRange(priorTurns);

            var inputTokens = new KVCachePromptRenderer(PromptRenderer).RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, messages, model.Config.Architecture,
                addGenerationPrompt: true, out _, out string trailingWhitespace,
                tools: tools, enableThinking: enableThinking);
            var sharedPrefix = ComputeSharedPrefix(model, messages, enableThinking, tools);
            _log.LogDebug(LogEventIds.ChatStarted,
                "cli.inference prompt tokens={Tokens}", inputTokens.Count);

            string requestId = $"cli-inference-{Guid.NewGuid():N}";
            var settings = SpeculativeDecodingOptions.Resolve(specDraftMax, specDraftConfMin);
            using var ownedInference = inference == null
                ? new CliInferenceSession(model, SchedulerConfig.FromEnvironment().WithSpeculation(settings), _log)
                : null;
            inference ??= ownedInference;
            try
            {
                if (imagePaths is { Count: > 0 } && !model.HasVisionEncoder())
                    _log.LogWarning(LogEventIds.HostConfiguration,
                        "No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                if (audioPaths is { Count: > 0 } && model is not IAudioCapableModel)
                    _log.LogWarning(LogEventIds.HostConfiguration, "This model has no audio path; the audio input will be ignored.");
                inputTokens = model.MultimodalInjector.ProcessPromptTokens(messages, inputTokens, requestId);
                if (preserveAllInput && model.MaxContextLength > 0
                    && (long)inputTokens.Count + maxTokens > model.MaxContextLength)
                    throw new InvalidOperationException(
                        $"The complete PDF requires {inputTokens.Count} prompt tokens plus a " +
                        $"{maxTokens}-token generation reserve, but the loaded model supports " +
                        $"{model.MaxContextLength} context tokens. No document content was truncated. " +
                        "Reduce --max-tokens, use a shorter PDF, or choose a model with a larger context window.");

                var cfg = samplingConfig ?? SamplingConfig.Greedy;
                var sampler = new TokenSampler(cfg);
                var streamed = new List<int>();
                string trimmedAtStop = null;
                bool Emit(int token)
                {
                    streamed.Add(token);
                    if (cfg.StopSequences == null || cfg.StopSequences.Count == 0)
                        return true;
                    var (trimmed, shouldStop) = sampler.CheckStopSequences(model.Tokenizer.Decode(streamed));
                    if (shouldStop)
                        trimmedAtStop = trimmed;
                    return !shouldStop;
                }
                var result = inference.Generate(inputTokens, maxTokens, cfg, Emit,
                    requestId: requestId, mediaSpans: model.MultimodalInjector.GetPreparedMediaSpans(requestId),
                    sharedPrefixTokens: CliSharedPrefix.MatchingLength(sharedPrefix, inputTokens));
                onGenerated?.Invoke(result.Tokens, trailingWhitespace);
                if (!silent)
                {
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "cli.inference prefill complete: tokens={Tokens} ms={Ms:F1} radixReused={ReusedTokens}",
                        inputTokens.Count, result.PrefillMs, result.Completion.PrefixCacheReusedTokens);
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "cli.inference decode complete: tokens={Tokens} ms={Ms:F1} tokensPerSec={Tps:F1}",
                        result.Tokens.Count, result.DecodeMs, result.Tokens.Count / Math.Max(result.DecodeMs / 1000.0, 1e-9));
                    _log.LogInformation(LogEventIds.ChatCompleted,
                        "cli.inference finishReason={FinishReason} tokens={Tokens}",
                        trimmedAtStop != null ? "stop_sequence" : result.Completion.FinishReason, result.Tokens.Count);
                    model.PrintTimingStats();
                }
                string decoded = trimmedAtStop ?? model.Tokenizer.Decode(result.Tokens);
                var parser = CliOutputParser.Create(model.Config.Architecture, enableThinking, tools,
                    model.Tokenizer, inputTokens);
                bool useParser = enableThinking || (tools != null && tools.Count > 0) || parser.AlwaysRequired;
                var parsed = useParser ? parser.Add(decoded, true) : new ParsedOutput { Content = decoded };
                onParsed?.Invoke(parsed);
                return useParser ? FormatParsedResult(parsed, enableThinking || parser.AlwaysRequired) : decoded;
            }
            finally
            {
                model.MultimodalInjector.ClearPreparedPromptState(requestId);
            }
        }

        private static List<int> ComputeSharedPrefix(ModelBase model, IReadOnlyList<ChatMessage> messages,
            bool enableThinking, List<ToolFunction> tools = null)
        {
            string system = messages.Count > 0 && messages[0].Role == "system" ? messages[0].Content : null;
            try
            {
                return CliSharedPrefix.Compute(system, tools is { Count: > 0 }, (probe, generationPrompt) =>
                    model.Tokenizer.Encode(PromptRenderer.Render(model.Config.ChatTemplate,
                        new List<ChatMessage>(probe), addGenerationPrompt: generationPrompt,
                        architecture: model.Config.Architecture, tools: tools, enableThinking: enableThinking),
                        addSpecial: true));
            }
            catch (Exception ex)
            {
                _log.LogDebug(LogEventIds.KvCacheReusePlan, ex,
                    "Could not establish a shared system prefix; keeping this request private");
                return new List<int>();
            }
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

        static void RunTests(ModelBase model, int maxTokens, string outputFile)
        {
            _log.LogInformation(LogEventIds.CliBenchmark, "Running verification tests");
            TestTokenizer(model);
            TestChatTemplate();
            TestInferenceWithOllamaComparison(model, maxTokens, outputFile);
        }

        /// <summary>
        /// Standalone inference benchmark: measures prefill and decode throughput without
        /// prompt rendering or output formatting overhead. The default greedy mode reports
        /// model-forward and host argmax time separately while retaining the end-to-end
        /// decode measurement. Fixed-token mode feeds a predetermined token stream so its
        /// timed decode is directly comparable to inference-only tools such as llama-bench;
        /// an untimed greedy chain still verifies deterministic model output.
        ///
        /// Optionally captures the first decode tokens after each prefill so the same
        /// benchmark can be run twice (e.g. with and without GDN_DISABLE_CHUNKED_PREFILL=1)
        /// and the outputs compared.
        /// </summary>
        static void RunBenchmark(
            ModelBase model,
            int prefillTokens,
            int decodeTokens,
            int runs,
            bool chunked = false,
            bool fixedTokens = false)
        {
            if (prefillTokens < 1)
                throw new ArgumentOutOfRangeException(nameof(prefillTokens), "Benchmark prefill tokens must be at least 1.");
            // 0 is legal and means "prefill only" — the llama-bench `pp<N>` shape.
            // TensorSharp.TestMatrix's pp512 / pp2048 cells pass exactly that, and
            // rejecting it made every prefill-only cell in the shipped default
            // feature set abort before the first forward.
            if (decodeTokens < 0)
                throw new ArgumentOutOfRangeException(nameof(decodeTokens), "Benchmark decode tokens cannot be negative.");
            if (runs < 1)
                throw new ArgumentOutOfRangeException(nameof(runs), "Benchmark runs must be at least 1.");

            string decodeMode = fixedTokens ? "fixed-inference" : "greedy-e2e";
            _log.LogInformation(LogEventIds.CliBenchmark,
                "inference benchmark starting: prefillTokens={PrefillTokens} decodeTokens={DecodeTokens} runs={Runs} chunked={Chunked} decodeMode={DecodeMode}",
                prefillTokens, decodeTokens, runs, chunked, decodeMode);

            // Build a synthetic prompt of `prefillTokens` tokens by repeating a stable token.
            // We pick a token id that's safely inside the vocab (not BOS/EOS/special).
            int vocab = model.Config.VocabSize;
            int syntheticSpan = Math.Max(1, Math.Min(17, vocab));
            int basisToken = Math.Max(0, Math.Min(100, vocab - syntheticSpan));
            int[] prefillIds = new int[prefillTokens];
            for (int i = 0; i < prefillTokens; i++)
                prefillIds[i] = basisToken + (i % syntheticSpan);

            // llama-bench advances decode with generated random token IDs rather
            // than scanning the vocabulary after every step. This deterministic
            // equivalent keeps token preparation out of the timed model call and
            // makes repeated TensorSharp runs directly comparable.
            int[] fixedDecodeIds = null;
            if (fixedTokens)
            {
                fixedDecodeIds = new int[decodeTokens];
                for (int i = 0; i < decodeTokens; i++)
                    fixedDecodeIds[i] = basisToken + ((prefillTokens + i) % syntheticSpan);
            }

            double bestPrefillMs = double.PositiveInfinity;
            double bestDecodeMs = double.PositiveInfinity;
            double bestPrefillTps = 0;
            double bestDecodeTps = 0;
            double bestDecodeModelMs = double.NaN;
            double bestDecodeModelTps = double.NaN;
            double bestDecodeSamplingMs = double.NaN;
            double avgPrefillTps = 0;
            double avgDecodeTps = 0;
            double avgDecodeModelTps = 0;
            int decodeModelTimingRuns = 0;

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

                int next = -1;
                if (!fixedTokens && decodeTokens > 0)
                {
                    // Seed sampling occurs after prefill and before the decode
                    // stopwatch, matching the benchmark's historic behavior.
                    next = SampleGreedyFromLogits(logits, vocab);
                    if (run == 0)
                    {
                        firstRunPrefillTopToken = next;
                        firstRunDecodeTokens = new int[decodeTokens];
                    }
                }

                // The outer stopwatch retains the benchmark's end-to-end metric.
                // Per-operation timestamps expose how much of it is actual model
                // execution versus the managed O(vocab) greedy scan.
                long decodeModelTicks = 0;
                long decodeSamplingTicks = 0;
                bool decodeBreakdownAvailable = fixedTokens || !usePipelinedGreedy;
                int[] decodeInput = new int[1];
                var decodeSw = Stopwatch.StartNew();
                if (decodeTokens == 0)
                {
                    // Prefill-only (pp<N>): nothing to decode. Skipping the whole
                    // block matters beyond saving work — the pipelined branch would
                    // submit a step for the unsampled `next = -1` and then index
                    // firstRunDecodeTokens[-1], which is null here because it is
                    // only allocated when there are tokens to record.
                }
                else if (fixedTokens)
                {
                    for (int i = 0; i < decodeTokens; i++)
                    {
                        decodeInput[0] = fixedDecodeIds[i];
                        long modelStart = Stopwatch.GetTimestamp();
                        logits = model.Forward(decodeInput);
                        decodeModelTicks += Stopwatch.GetTimestamp() - modelStart;
                    }
                }
                else if (usePipelinedGreedy)
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
                        decodeInput[0] = next;
                        long modelStart = Stopwatch.GetTimestamp();
                        logits = model.Forward(decodeInput);
                        decodeModelTicks += Stopwatch.GetTimestamp() - modelStart;

                        long samplingStart = Stopwatch.GetTimestamp();
                        next = SampleGreedyFromLogits(logits, vocab);
                        decodeSamplingTicks += Stopwatch.GetTimestamp() - samplingStart;
                        if (run == 0)
                            firstRunDecodeTokens[i] = next;
                    }
                }
                decodeSw.Stop();
                double decodeMs = decodeSw.Elapsed.TotalMilliseconds;
                // Prefill-only runs report NaN rather than 0 for the decode metrics:
                // 0 tok/s would read as "measured, and catastrophically slow" in the
                // report, while NaN is unambiguously "not measured here".
                double decodeTps = decodeTokens > 0 ? decodeTokens / (decodeMs / 1000.0) : double.NaN;
                double decodeMsPerTok = decodeTokens > 0 ? decodeMs / decodeTokens : double.NaN;
                double decodeModelMs = decodeModelTicks * 1000.0 / Stopwatch.Frequency;
                double decodeSamplingMs = decodeSamplingTicks * 1000.0 / Stopwatch.Frequency;
                double decodeModelTps = decodeBreakdownAvailable && decodeTokens > 0
                    ? decodeTokens / (decodeModelMs / 1000.0)
                    : double.NaN;

                if (decodeBreakdownAvailable)
                {
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "benchmark run {Run}/{Runs}: prefillMs={PrefillMs:F0} prefillTps={PrefillTps:F1} decodeMs={DecodeMs:F0} decodeTps={DecodeTps:F1} msPerTok={MsPerTok:F1} decodeModelMs={DecodeModelMs:F0} decodeModelTps={DecodeModelTps:F1} greedySampleMs={GreedySampleMs:F1} decodeMode={DecodeMode}",
                        run + 1, runs, prefillMs, prefillTps, decodeMs, decodeTps,
                        decodeMsPerTok, decodeModelMs, decodeModelTps,
                        decodeSamplingMs, decodeMode);
                }
                else
                {
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "benchmark run {Run}/{Runs}: prefillMs={PrefillMs:F0} prefillTps={PrefillTps:F1} decodeMs={DecodeMs:F0} decodeTps={DecodeTps:F1} msPerTok={MsPerTok:F1} decodeBreakdown=pipelined-overlap decodeMode={DecodeMode}",
                        run + 1, runs, prefillMs, prefillTps, decodeMs, decodeTps,
                        decodeMsPerTok, decodeMode);
                }

                if (prefillMs < bestPrefillMs)
                {
                    bestPrefillMs = prefillMs;
                    bestPrefillTps = prefillTps;
                }
                if (decodeMs < bestDecodeMs)
                {
                    bestDecodeMs = decodeMs;
                    bestDecodeTps = decodeTps;
                    bestDecodeModelMs = decodeBreakdownAvailable ? decodeModelMs : double.NaN;
                    bestDecodeModelTps = decodeBreakdownAvailable ? decodeModelTps : double.NaN;
                    bestDecodeSamplingMs = decodeBreakdownAvailable ? decodeSamplingMs : double.NaN;
                }
                avgPrefillTps += prefillTps;
                avgDecodeTps += decodeTps;
                if (decodeBreakdownAvailable)
                {
                    avgDecodeModelTps += decodeModelTps;
                    decodeModelTimingRuns++;
                }
            }

            avgPrefillTps /= runs;
            avgDecodeTps /= runs;
            if (decodeModelTimingRuns > 0)
                avgDecodeModelTps /= decodeModelTimingRuns;

            if (decodeModelTimingRuns > 0)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark summary: bestPrefillMs={BestPrefillMs:F0} bestPrefillTps={BestPrefillTps:F1} bestDecodeMs={BestDecodeMs:F0} bestDecodeTps={BestDecodeTps:F1} bestDecodeMsPerTok={BestDecodeMsPerTok:F2} bestDecodeModelMs={BestDecodeModelMs:F0} bestDecodeModelTps={BestDecodeModelTps:F1} bestGreedySampleMs={BestGreedySampleMs:F1} avgPrefillTps={AvgPrefillTps:F1} avgDecodeTps={AvgDecodeTps:F1} avgDecodeModelTps={AvgDecodeModelTps:F1} decodeMode={DecodeMode}",
                    bestPrefillMs, bestPrefillTps, bestDecodeMs, bestDecodeTps,
                    decodeTokens > 0 ? bestDecodeMs / decodeTokens : double.NaN, bestDecodeModelMs, bestDecodeModelTps,
                    bestDecodeSamplingMs, avgPrefillTps, avgDecodeTps,
                    avgDecodeModelTps, decodeMode);
            }
            else
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark summary: bestPrefillMs={BestPrefillMs:F0} bestPrefillTps={BestPrefillTps:F1} bestDecodeMs={BestDecodeMs:F0} bestDecodeTps={BestDecodeTps:F1} bestDecodeMsPerTok={BestDecodeMsPerTok:F2} avgPrefillTps={AvgPrefillTps:F1} avgDecodeTps={AvgDecodeTps:F1} decodeBreakdown=pipelined-overlap decodeMode={DecodeMode}",
                    bestPrefillMs, bestPrefillTps, bestDecodeMs, bestDecodeTps,
                    decodeTokens > 0 ? bestDecodeMs / decodeTokens : double.NaN, avgPrefillTps, avgDecodeTps, decodeMode);
            }

            if (!fixedTokens && firstRunDecodeTokens != null)
            {
                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark sampled tokens (run1): prefillTopToken={Prefill} decode={Decode}",
                    firstRunPrefillTopToken, string.Join(",", firstRunDecodeTokens));
            }
            model.PrintTimingStats();

            if (fixedTokens)
            {
                // Fixed-token timing intentionally does not inspect logits. Run
                // the historic greedy chain once, outside all benchmark
                // stopwatches, so regressions remain visible and comparable.
                model.ResetKVCache();
                float[] correctnessLogits = chunked
                    ? model.ForwardRefill(prefillIds)
                    : model.Forward(prefillIds);
                firstRunPrefillTopToken = SampleGreedyFromLogits(correctnessLogits, vocab);
                firstRunDecodeTokens = RunUntimedGreedyDecode(
                    model, firstRunPrefillTopToken, decodeTokens, vocab, usePipelinedGreedy);

                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark sampled tokens (untimed correctness): prefillTopToken={Prefill} decode={Decode}",
                    firstRunPrefillTopToken, string.Join(",", firstRunDecodeTokens));
            }
        }

        static int[] RunUntimedGreedyDecode(
            ModelBase model,
            int firstToken,
            int decodeTokens,
            int vocab,
            bool usePipelinedGreedy)
        {
            int[] sampledTokens = new int[decodeTokens];
            if (usePipelinedGreedy)
            {
                Tensor pending = model.SubmitGreedyDecodeStep(firstToken);
                int step = 1;
                for (; step < decodeTokens; step++)
                {
                    Tensor nextDevice = model.SubmitGreedyDecodeStep(null);
                    sampledTokens[step - 1] = pending.GetElementsAsInt(1)[0];
                    pending.Dispose();
                    pending = nextDevice;
                }

                sampledTokens[decodeTokens - 1] = pending.GetElementsAsInt(1)[0];
                pending.Dispose();
                model.ResetPipelinedGreedyState();
                return sampledTokens;
            }

            int next = firstToken;
            int[] decodeInput = new int[1];
            for (int i = 0; i < decodeTokens; i++)
            {
                decodeInput[0] = next;
                float[] logits = model.Forward(decodeInput);
                next = SampleGreedyFromLogits(logits, vocab);
                sampledTokens[i] = next;
            }
            return sampledTokens;
        }

        /// <summary>
        /// True when <see cref="TokenSampler.Sample"/> will reduce to a plain
        /// argmax over the raw logits — which is exactly what the model's
        /// pipelined device argmax computes.
        ///
        /// <see cref="SamplingConfig.IsGreedy"/> is not the right test here: it
        /// also demands topK &lt;= 0 / topP &gt;= 1 / minP &lt;= 0, but the
        /// sampler never reaches those stages once temperature &lt;= 0, so the
        /// very common "--top-k 1" spelling of greedy decoding was silently
        /// falling off the pipelined path. Conversely IsGreedy ignores the
        /// penalty and grammar knobs, which DO change the greedy branch's
        /// result (ArgmaxWithPenaltiesInPlace / grammar masking) and therefore
        /// have to disqualify the device argmax.
        /// </summary>
        static bool IsArgmaxDecode(SamplingConfig cfg)
        {
            return cfg.Temperature <= 0f
                && cfg.RepetitionPenalty == 1f
                && cfg.PresencePenalty == 0f
                && cfg.FrequencyPenalty == 0f
                && cfg.Grammar == null
                && (cfg.FirstTokenAllowList == null || cfg.FirstTokenAllowList.Count == 0);
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

            using var inference = new CliInferenceSession(model, SchedulerConfig.FromEnvironment(), _log);
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
                    out _,
                    out string generationPromptTrailingWhitespace,
                    enableThinking: enableThinking);

                promptTokens[turn] = inputTokens.Count;

                if (!useCache)
                    inference.Reset();
                var result = inference.Generate(inputTokens, maxTokens, samplerCfg, enablePrefixCache: useCache);
                prefillMs[turn] = result.PrefillMs;
                var generatedTokens = result.Tokens;

                _log.LogInformation(LogEventIds.CliBenchmark,
                    "benchmark turn {Turn}: promptTokens={PromptTokens} prefillMs={PrefillMs:F1} decodeTokens={DecodeTokens} radixReused={ReusedTokens}",
                    turn + 1, inputTokens.Count, prefillMs[turn], generatedTokens.Count,
                    result.Completion.PrefixCacheReusedTokens);

                // Append the assistant turn so subsequent renders include it.
                var parser = CliOutputParser.Create(arch, enableThinking, null,
                    model.Tokenizer, inputTokens);
                var parsed = parser.Add(model.Tokenizer.Decode(generatedTokens), true);
                history.Add(new ChatMessage
                {
                    Role = "assistant",
                    Content = parsed.Content ?? "",
                    Thinking = parsed.Thinking ?? "",
                    RawOutputTokens = generatedTokens,
                    RawPromptTrailingWhitespace = generationPromptTrailingWhitespace,
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
                    "paged-bench: model architecture '{Arch}' does not support KV snapshot. Use GptOss or Mistral3.",
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
        /// <summary>
        /// Print the Vulkan devices ggml-vulkan can see (index + adapter name) so the
        /// operator knows what to pass to <c>--gpu-device</c> on multi-GPU hosts.
        /// Enumerating spins up the Vulkan instance but no backend/device state.
        /// </summary>
        /// <summary>
        /// Print the registered skills and exit. Console-only, like
        /// <see cref="ListVulkanGpus"/>, and reached before any model is loaded so it
        /// answers "what do I have?" without a GGUF on disk.
        ///
        /// <para>
        /// The load ERRORS are printed too, and that is the point of the command: a
        /// skill whose SKILL.md will not parse is otherwise simply absent from every
        /// list, which is the hardest kind of problem for its author to diagnose.
        /// </para>
        /// </summary>
        /// <summary>
        /// Splice the code-execution declarations into the tool list. A caller's own tool
        /// of the same name wins: it is theirs and they can service it.
        /// </summary>
        /// <param name="workspace">
        /// The session's workspace, or null when this host keeps nothing between calls.
        /// It is passed rather than assumed: this took the <c>DeclareTools()</c> overload
        /// whose <c>persists</c> defaults to true, so a CLI without a workspace would have
        /// been offered the file tools and the patcher — every call to which refuses,
        /// because all three need a directory that outlives the call.
        /// </param>
        /// <summary>
        /// The six editing rules for the tools that were actually declared, or the empty
        /// string. Read off the finished tool list rather than from the options that
        /// produced it, so the block can never name a tool the model was not given.
        /// </summary>
        internal static string CodeSystemBlock(IReadOnlyList<ToolFunction>? tools)
        {
            if (tools == null)
                return string.Empty;

            bool Declared(string name) =>
                tools.Any(t => string.Equals(t?.Name, name, StringComparison.Ordinal));

            return CodePrompt.Block(
                fileTools: Declared(SkillToolNames.ReadFile) && Declared(SkillToolNames.WriteFile),
                hasPatch: Declared(SkillToolNames.ApplyPatch));
        }

        internal static List<ToolFunction> AppendCodeTool(
            List<ToolFunction> tools, ICodeRunner runner, SessionWorkspace? workspace = null)
        {
            var merged = tools != null ? new List<ToolFunction>(tools) : new List<ToolFunction>();

            // Shadowing is decided PER NAME. An early return on a clash with one of them
            // would also withhold the other, which a caller never asked for: a client that
            // happens to own a tool called 'shell' must not silently lose apply_patch.
            foreach (ToolFunction declaration in runner.DeclareTools(persists: workspace != null))
            {
                if (!merged.Any(t => string.Equals(t?.Name, declaration.Name, StringComparison.OrdinalIgnoreCase)))
                    merged.Add(declaration);
            }
            return merged;
        }

        /// <summary>
        /// The skill-script runner's options, extended with the session workspace and the
        /// package installer. Both come from code execution: with them a skill's bundled
        /// script runs in the same directory as everything else and gets its missing
        /// dependencies installed automatically; without <c>--code-exec</c> the script
        /// keeps the plain per-call scratch it always had.
        /// </summary>
        internal static SkillScriptRunnerOptions ToScriptRunnerOptions(
            SkillHostOptions skillOptions, SessionWorkspace workspace, ICodeRunner installer)
        {
            SkillScriptRunnerOptions options = skillOptions.ToScriptRunnerOptions();
            if (workspace == null)
                return options;

            return new SkillScriptRunnerOptions
            {
                Sandbox = options.Sandbox,
                AllowNetwork = options.AllowNetwork,
                Workspace = workspace,
                PackageInstaller = installer,
            };
        }

        static void ListSkills(SkillRegistry registry)
        {
            if (registry == null)
            {
                Console.WriteLine("Agent skills are disabled (--no-skills / TS_NO_SKILLS).");
                return;
            }

            Console.WriteLine($"Skill roots: {string.Join(", ", registry.Roots)}");
            Console.WriteLine();

            if (registry.Skills.Count == 0)
            {
                Console.WriteLine("No skills found.");
                Console.WriteLine();
                Console.WriteLine("A skill is a directory containing SKILL.md. Put one under a root above, or");
                Console.WriteLine("point at your own with --skills-dir <path>.");
            }
            else
            {
                Console.WriteLine($"{registry.Skills.Count} skill(s):");
                Console.WriteLine();
                foreach (var skill in registry.Skills)
                {
                    Console.WriteLine($"  {skill.Id}");
                    Console.WriteLine($"      {Wrap(skill.Description, 92, 6)}");
                    int bundled = 0;
                    foreach (var _ in skill.BundledFiles)
                        bundled++;
                    Console.WriteLine(
                        $"      {bundled} bundled file(s), {SkillTextBudget.FormatBytes(skill.TotalBytes)}, "
                        + $"~{skill.Manifest.ApproximateBodyTokens} tokens of instructions");
                    foreach (string warning in skill.Manifest.Warnings)
                        Console.WriteLine($"      warning: {warning}");
                    Console.WriteLine();
                }
            }

            if (registry.Errors.Count > 0)
            {
                Console.WriteLine($"{registry.Errors.Count} directory/directories could not be loaded:");
                foreach (var error in registry.Errors)
                    Console.WriteLine($"  {error.Path}: {error.Message}");
                Console.WriteLine();
            }

            Console.WriteLine("Use one with:  --skill <name>");
        }

        /// <summary>Wrap text to <paramref name="width"/> columns, indenting continuations.</summary>
        static string Wrap(string text, int width, int indent)
        {
            if (string.IsNullOrEmpty(text))
                return string.Empty;

            var sb = new StringBuilder();
            int column = 0;
            string pad = new string(' ', indent);
            foreach (string word in text.Split(' ', StringSplitOptions.RemoveEmptyEntries))
            {
                if (column > 0 && column + 1 + word.Length > width)
                {
                    sb.Append('\n').Append(pad);
                    column = 0;
                }
                else if (column > 0)
                {
                    sb.Append(' ');
                    column++;
                }
                sb.Append(word);
                column += word.Length;
            }
            return sb.ToString();
        }

        static void ListVulkanGpus()
        {
            int count = TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceCount();
            if (count <= 0)
            {
                Console.WriteLine("No Vulkan devices found. Ensure the native GGML bridge is built with Vulkan support " +
                    "(TensorSharp.GGML.Native/build-windows.ps1 --vulkan) and a Vulkan driver is installed.");
                return;
            }

            Console.WriteLine($"Vulkan devices ({count}):");
            for (int i = 0; i < count; i++)
            {
                Console.WriteLine($"  {i}: {TensorSharp.GGML.GgmlBasicOps.GetVulkanDeviceDescription(i) ?? "(unknown)"}");
            }
            Console.WriteLine("Select one with: --backend ggml_vulkan --gpu-device <index>");
        }

        // Publish a companion-network path under every env var that consumes it. Video
        // models read a generic TS_VIDEO_* name; Wan predates that and reads TS_WAN_*, so
        // both are set and neither model needs to know about the other's naming.
        static void ApplyVideoCompanionOverride(string flag, string path, params string[] envVars)
        {
            if (string.IsNullOrWhiteSpace(path))
                return;
            if (!File.Exists(path))
                throw new FileNotFoundException($"{flag} file not found: {path}", path);
            string full = Path.GetFullPath(path);
            foreach (string envVar in envVars)
                Environment.SetEnvironmentVariable(envVar, full);
            _log.LogInformation(LogEventIds.HostConfiguration,
                "Video companion override {Flag} -> {Path}", flag, full);
        }

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

            string rendered = ChatTemplate.RenderChatMl(messages, true);
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
            string rendered = ChatTemplate.RenderChatMl(messages, true);

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
                    model = "tensorsharp-test",
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
