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
using System.IO;
using System.Linq;
using System.Text;
using System.Text.Json;
using InferenceEngine;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.OutputEncoding = Encoding.UTF8;
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
            bool enableThinking = false;
            string toolsFile = null;

            var samplingConfig = new SamplingConfig();

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model": modelPath = args[++i]; break;
                    case "--input": inputFile = args[++i]; break;
                    case "--input-jsonl": inputJsonl = args[++i]; break;
                    case "--output": outputFile = args[++i]; break;
                    case "--image": imagePath = args[++i]; break;
                    case "--audio": audioPath = args[++i]; break;
                    case "--video": videoPath = args[++i]; break;
                    case "--mmproj": mmProjPath = args[++i]; break;
                    case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
                    case "--test": runTest = true; break;
                    case "--backend": backendStr = args[++i].ToLowerInvariant(); break;
                    case "--test-templates": testTemplatesDir = args[++i]; break;
                    case "--think": enableThinking = true; break;
                    case "--tools": toolsFile = args[++i]; break;
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
                    Console.Error.WriteLine($"Tools file not found: {toolsFile}");
                    return;
                }
                tools = JsonSerializer.Deserialize<List<ToolFunction>>(File.ReadAllText(toolsFile),
                    new JsonSerializerOptions { PropertyNameCaseInsensitive = true });
                Console.WriteLine($"Loaded {tools.Count} tool definition(s) from {toolsFile}");
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
                Console.Error.WriteLine($"Model file not found: {modelPath ?? "(none)"}");
                Console.Error.WriteLine("Usage: InferenceConsole --model <path.gguf> [--input <input.txt>] " +
                    "[--input-jsonl <requests.jsonl>] [--image <image.png>] [--output <output.txt>] " +
                    "[--max-tokens N] [--test] [--backend cpu|ggml_cpu|ggml_metal|ggml_cuda]");
                return;
            }

            BackendType backend = backendStr switch
            {
                "cpu" => BackendType.Cpu,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_metal" => BackendType.GgmlMetal,
                "cuda" or "ggml_cuda" => BackendType.GgmlCuda,
                _ => throw new ArgumentException($"Unknown backend '{backendStr}'. Use: cpu, ggml_cpu, ggml_metal, ggml_cuda"),
            };

            using var model = ModelBase.Create(modelPath, backend);

            if (mmProjPath != null && model is Gemma3Model gemma3WithVision)
            {
                gemma3WithVision.LoadVisionEncoder(mmProjPath);
            }
            else if (mmProjPath != null && model is Gemma4Model gemma4WithVision)
            {
                gemma4WithVision.LoadVisionEncoder(mmProjPath);
                if (audioPath != null)
                    gemma4WithVision.LoadAudioEncoder(mmProjPath);
            }
            else if (mmProjPath != null && model is Qwen35Model qwen35WithVision)
            {
                qwen35WithVision.LoadVisionEncoder(mmProjPath);
            }
            else if (imagePath != null && model.Config.Architecture == "gemma3")
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "mmproj-gemma3-4b-f16.gguf");
                if (File.Exists(autoMmproj) && model is Gemma3Model g3auto)
                {
                    Console.WriteLine($"Auto-loading vision encoder: {autoMmproj}");
                    g3auto.LoadVisionEncoder(autoMmproj);
                }
            }
            else if ((imagePath != null || audioPath != null || videoPath != null)
                     && model.Config.Architecture == "gemma4")
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "gemma-4-mmproj-F16.gguf");
                if (File.Exists(autoMmproj) && model is Gemma4Model g4auto)
                {
                    Console.WriteLine($"Auto-loading multimodal encoder: {autoMmproj}");
                    if (imagePath != null || videoPath != null)
                        g4auto.LoadVisionEncoder(autoMmproj);
                    if (audioPath != null)
                        g4auto.LoadAudioEncoder(autoMmproj);
                }
            }
            else if (imagePath != null && model is Qwen35Model q35auto)
            {
                string autoMmproj = Path.Combine(Path.GetDirectoryName(modelPath), "Qwen3.5-mmproj-F16.gguf");
                if (File.Exists(autoMmproj))
                {
                    Console.WriteLine($"Auto-loading vision encoder: {autoMmproj}");
                    q35auto.LoadVisionEncoder(autoMmproj);
                }
            }

            if (runTest)
            {
                RunTests(model, maxTokens, outputFile);
                return;
            }

            if (inputJsonl != null)
            {
                RunJsonlBatch(model, inputJsonl, outputFile, maxTokens, samplingConfig);
                return;
            }

            string rawText;
            if (inputFile != null && File.Exists(inputFile))
            {
                rawText = File.ReadAllText(inputFile).TrimEnd();
            }
            else
            {
                rawText = "What is 1+1?";
                Console.WriteLine($"No input file specified, using default: \"{rawText}\"");
            }

            List<string> imagePaths = null;
            List<string> audioPaths = null;

            if (videoPath != null)
            {
                if (!File.Exists(videoPath))
                {
                    Console.Error.WriteLine($"Video file not found: {videoPath}");
                    return;
                }
                Console.WriteLine($"Video: {videoPath} ({new FileInfo(videoPath).Length / 1024} KB)");
                imagePaths = MediaHelper.ExtractVideoFrames(videoPath);
                Console.WriteLine($"Extracted {imagePaths.Count} frames from video");
                rawText = "What is happening in this video? Please describe it.";
            }
            else if (imagePath != null)
            {
                if (!File.Exists(imagePath))
                {
                    Console.Error.WriteLine($"Image file not found: {imagePath}");
                    return;
                }
                imagePaths = new List<string> { imagePath };
                rawText = "What is in this image? Please describe it.";
                Console.WriteLine($"Image: {imagePath} ({new FileInfo(imagePath).Length / 1024} KB)");
            }

            if (audioPath != null)
            {
                if (!File.Exists(audioPath))
                {
                    Console.Error.WriteLine($"Audio file not found: {audioPath}");
                    return;
                }
                audioPaths = new List<string> { audioPath };
                rawText = "Listen to this audio and describe what you hear.";
                Console.WriteLine($"Audio: {audioPath} ({new FileInfo(audioPath).Length / 1024} KB)");
            }

            string result = RunInference(model, rawText, imagePaths, maxTokens, audioPaths,
                isVideo: videoPath != null, samplingConfig: samplingConfig,
                enableThinking: enableThinking, tools: tools);

            if (outputFile != null)
            {
                File.WriteAllText(outputFile, result);
                Console.WriteLine($"Output written to {outputFile}");
            }
            else
            {
                Console.WriteLine("\n=== Generated Output ===");
                Console.WriteLine(result);
            }
        }

        static void RunJsonlBatch(ModelBase model, string inputJsonlPath, string outputFile, int defaultMaxTokens,
            SamplingConfig defaultSampling)
        {
            if (!File.Exists(inputJsonlPath))
            {
                Console.Error.WriteLine($"JSONL file not found: {inputJsonlPath}");
                return;
            }

            string[] lines = File.ReadAllLines(inputJsonlPath);
            var results = new List<string>();
            int total = lines.Length;
            int completed = 0;

            Console.WriteLine($"Processing {total} requests from {inputJsonlPath}");
            Console.WriteLine(new string('=', 60));

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
                    Console.Error.WriteLine($"[Line {lineIdx + 1}] Invalid JSON: {ex.Message}");
                    results.Add(JsonSerializer.Serialize(new { line = lineIdx + 1, error = $"Invalid JSON: {ex.Message}" }));
                    continue;
                }

                var root = doc.RootElement;
                string id = root.TryGetProperty("id", out var idProp) ? idProp.GetString() : $"request_{lineIdx + 1}";

                Console.WriteLine($"\n[{lineIdx + 1}/{total}] Processing request: {id}");

                try
                {
                    var messages = ParseMessages(root);
                    int maxTokens = root.TryGetProperty("max_tokens", out var mt) ? mt.GetInt32() : defaultMaxTokens;
                    var sampling = ParseSamplingFromJson(root, defaultSampling);

                    var imagePaths = ParseStringList(root, "images");
                    var audioPaths = ParseStringList(root, "audios");
                    bool isVideo = root.TryGetProperty("is_video", out var iv) && iv.GetBoolean();

                    model.ResetKVCache();

                    string rendered = ChatTemplate.RenderFromGgufTemplate(
                        model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                        architecture: model.Config.Architecture);

                    Console.WriteLine($"Rendered prompt:\n---\n{rendered}\n---");

                    var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);
                    Console.WriteLine($"Input tokens ({inputTokens.Count}): [{string.Join(", ", inputTokens.Take(20))}{(inputTokens.Count > 20 ? ", ..." : "")}]");

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

                    Console.WriteLine($"Output ({generatedTokens.Count} tokens, {tokPerSec:F1} tok/s): {output}");

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
                    Console.Error.WriteLine($"[Line {lineIdx + 1}] Error: {ex.Message}");
                    var errorObj = new Dictionary<string, object>
                    {
                        ["id"] = id,
                        ["error"] = ex.Message,
                    };
                    results.Add(JsonSerializer.Serialize(errorObj));
                }
            }

            totalSw.Stop();

            Console.WriteLine(new string('=', 60));
            Console.WriteLine($"Completed {completed}/{total} requests in {totalSw.Elapsed.TotalSeconds:F1}s");

            if (outputFile != null)
            {
                File.WriteAllLines(outputFile, results);
                Console.WriteLine($"Results written to {outputFile}");
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

        static string RunInference(ModelBase model, string rawText, List<string> imagePaths, int maxTokens,
            List<string> audioPaths = null, bool isVideo = false, SamplingConfig samplingConfig = null,
            bool enableThinking = false, List<ToolFunction> tools = null)
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = rawText, ImagePaths = imagePaths, AudioPaths = audioPaths, IsVideo = isVideo }
            };

            string rendered = ChatTemplate.RenderFromGgufTemplate(
                model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                architecture: model.Config.Architecture,
                tools: tools, enableThinking: enableThinking);

            Console.WriteLine($"Rendered prompt:\n---\n{rendered}\n---");

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

                    Console.WriteLine($"Gemma3 vision: {proc.TokensPerImage} tokens per image, " +
                        $"start={startId}, end={endId}");
                    Console.WriteLine($"Total tokens after image expansion: {inputTokens.Count}");

                    if (model is Gemma3Model g3 && g3.VisionEncoder != null)
                    {
                        Console.Write("Processing image through vision encoder...");
                        float[] pixels = proc.ProcessImage(imagePaths[0]);
                        Console.Write(" pixels ready...");
                        var visionEmbeddings = g3.VisionEncoder.Encode(pixels);
                        Console.WriteLine($" done ({visionEmbeddings.Sizes[0]}x{visionEmbeddings.Sizes[1]})");

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
                            Console.WriteLine($"Vision embeddings will be injected at token position {imageTokenStart}");
                        }
                        else
                        {
                            Console.WriteLine("Warning: Could not find image placeholder position");
                            visionEmbeddings.Dispose();
                        }
                    }
                    else
                    {
                        Console.WriteLine("Note: No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
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
                        var proc = new Gemma4ImageProcessor();
                        var allVisionEmbeddings = new List<TensorSharp.Tensor>();

                        foreach (var imgP in imagePaths)
                        {
                            Console.Write($"Processing frame through Gemma4 vision encoder...");
                            var (pixels, imgW, imgH) = proc.ProcessImage(imgP);
                            Console.Write($" pixels ready ({imgW}x{imgH})...");
                            var visionEmb = g4.VisionEncoder.Encode(pixels, imgW, imgH);
                            Console.WriteLine($" done ({visionEmb.Sizes[0]}x{visionEmb.Sizes[1]})");
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
                                Console.WriteLine($"Gemma4 vision frame {imgIdx}: {numVisionTokens} tokens at pos {insertPos}");

                                searchFrom = imageTokenPos + 1 + numVisionTokens + 1;
                            }
                            else
                            {
                                Console.WriteLine($"Warning: No more <|image> tokens for frame {imgIdx}");
                                visionEmbeddings.Dispose();
                            }
                        }
                        Console.WriteLine($"Total tokens after image expansion: {inputTokens.Count}");
                    }
                    else if (imagePaths.Count > 0)
                    {
                        Console.WriteLine("Note: No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                    }
                }
                else
                {
                    int imagePadId = model.Tokenizer.LookupToken("<|image_pad|>");
                    if (imagePadId < 0)
                    {
                        Console.Error.WriteLine("Warning: <|image_pad|> token not found in vocabulary");
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
                            var (width, height) = Qwen35ImageProcessor.ReadPngDimensions(imagePaths[i]);
                            tokenCounts[i] = processor.ComputeImageTokenCount(height, width);
                            var (gridH, gridW) = processor.GetPatchGrid(height, width);
                            var (resizedH, resizedW) = processor.SmartResize(height, width);
                            Console.WriteLine($"Image {i}: {width}x{height} -> resize {resizedW}x{resizedH} -> " +
                                $"grid {gridW}x{gridH} -> {tokenCounts[i]} vision tokens " +
                                $"(merged {gridW / processor.MergeSize}x{gridH / processor.MergeSize})");
                        }

                        inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

                        int visionStartId = model.Tokenizer.LookupToken("<|vision_start|>");
                        int visionEndId = model.Tokenizer.LookupToken("<|vision_end|>");
                        Console.WriteLine($"Vision token IDs: start={visionStartId}, pad={imagePadId}, end={visionEndId}");

                        if (model is Qwen35Model q35 && q35.VisionEncoder != null)
                        {
                            Console.Write("Processing image through Qwen3.5 vision encoder...");
                            var (pixels, resH, resW) = processor.ProcessImage(imagePaths[0]);
                            Console.Write($" pixels ready ({resW}x{resH})...");
                            var visionEmbeddings = q35.VisionEncoder.Encode(pixels, resH, resW);
                            Console.WriteLine($" done ({visionEmbeddings.Sizes[0]}x{visionEmbeddings.Sizes[1]})");

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
                                Console.WriteLine($"Vision embeddings will be injected at token position {imageTokenStart}");
                            }
                            else
                            {
                                Console.WriteLine("Warning: Could not find image placeholder position");
                                visionEmbeddings.Dispose();
                            }
                        }
                        else if (!model.HasVisionEncoder())
                        {
                            Console.WriteLine("Note: No vision encoder loaded. Use --mmproj to specify the vision encoder GGUF.");
                        }
                    }
                }
            }

            // Audio processing for Gemma4
            if (audioPaths != null && audioPaths.Count > 0 && model.Config.Architecture == "gemma4")
            {
                int audioStartId = model.Tokenizer.LookupToken("<|audio>");
                int audioEndId = model.Tokenizer.LookupToken("<audio|>");

                if (model is Gemma4Model g4a && g4a.AudioEncoder != null)
                {
                    Console.Write("Decoding audio...");
                    float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(audioPaths[0]);
                    Console.WriteLine($" {samples.Length} samples ({(double)samples.Length / 16000:F1}s)");

                    if (samples.Length % 128 != 0)
                    {
                        int padded = samples.Length + (128 - samples.Length % 128);
                        Array.Resize(ref samples, padded);
                    }

                    Console.Write("Computing mel spectrogram...");
                    var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
                    Console.WriteLine($" {numFrames} frames");

                    if (melData != null && numFrames > 0)
                    {
                        var audioEmbeddings = g4a.AudioEncoder.Encode(melData, numFrames);
                        int numAudioTokens = (int)audioEmbeddings.Sizes[0];
                        Console.WriteLine($"Audio embeddings: [{audioEmbeddings.Sizes[0]}x{audioEmbeddings.Sizes[1]}]");

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
                            Console.WriteLine($"Gemma4 audio: {numAudioTokens} tokens at position {insertPos}");
                            Console.WriteLine($"Total tokens after audio expansion: {inputTokens.Count}");
                        }
                        else
                        {
                            Console.WriteLine("Warning: Could not find <|audio> token in prompt");
                            audioEmbeddings.Dispose();
                        }
                    }
                }
                else
                {
                    Console.WriteLine("Note: No audio encoder loaded. Use --mmproj to specify the multimodal GGUF.");
                }
            }

            Console.WriteLine($"Input tokens ({inputTokens.Count}): [{string.Join(", ", inputTokens.Take(30))}" +
                (inputTokens.Count > 30 ? $"... ({inputTokens.Count} total)" : "") + "]");

            model.ResetKVCache();

            bool tokenByToken = Environment.GetEnvironmentVariable("TOKEN_BY_TOKEN") == "1";
            float[] logits;
            if (tokenByToken)
            {
                Console.WriteLine("*** TOKEN-BY-TOKEN MODE ***");
                logits = null;
                for (int i = 0; i < inputTokens.Count; i++)
                    logits = model.Forward(new[] { inputTokens[i] });
            }
            else
            {
                logits = model.Forward(inputTokens.ToArray());
            }
            var generatedTokens = new List<int>();

            PrintTopLogits(logits, model, "prefill");

            var cfg = samplingConfig ?? SamplingConfig.Greedy;
            var sampler = new TokenSampler(cfg);

            if (!cfg.IsGreedy)
            {
                Console.WriteLine($"Sampling: temp={cfg.Temperature}, top_k={cfg.TopK}, top_p={cfg.TopP}, " +
                    $"min_p={cfg.MinP}, rep_pen={cfg.RepetitionPenalty}, pres_pen={cfg.PresencePenalty}, " +
                    $"freq_pen={cfg.FrequencyPenalty}, seed={cfg.Seed}");
            }

            var parser = OutputParserFactory.Create(model.Config.Architecture);
            parser.Init(enableThinking, tools);
            bool useParser = enableThinking || (tools != null && tools.Count > 0);
            if (useParser)
            {
                Console.WriteLine($"Output parser: {parser.GetType().Name} (thinking={enableThinking}, tools={tools?.Count ?? 0})");
            }

            for (int step = 0; step < maxTokens; step++)
            {
                int nextToken = sampler.Sample(logits, generatedTokens);
                Console.Write($"[{nextToken}:{model.Tokenizer.Vocab[nextToken]}]");

                if (model.Tokenizer.IsEos(nextToken))
                {
                    Console.WriteLine(" <EOS>");
                    break;
                }

                generatedTokens.Add(nextToken);

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    string partial = model.Tokenizer.Decode(generatedTokens);
                    var (trimmed, shouldStop) = sampler.CheckStopSequences(partial);
                    if (shouldStop)
                    {
                        Console.WriteLine(" <STOP>");
                        if (useParser)
                        {
                            var finalParsed = parser.Add(trimmed, true);
                            return FormatParsedResult(finalParsed, enableThinking);
                        }
                        return trimmed;
                    }
                }

                logits = model.Forward(new[] { nextToken });
                if (step < 3)
                    PrintTopLogits(logits, model, $"decode_{step}");
            }

            Console.WriteLine();
            model.PrintTimingStats();
            string decoded = model.Tokenizer.Decode(generatedTokens);

            if (useParser)
            {
                var parsed = parser.Add(decoded, true);
                return FormatParsedResult(parsed, enableThinking);
            }
            return decoded;
        }

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

        

        static unsafe TensorSharp.Tensor ConcatenateEmbeddings(List<TensorSharp.Tensor> embeddings,
            int totalTokens, int embDim)
        {
            var allocator = embeddings[0].Allocator;
            var result = new TensorSharp.Tensor(allocator, DType.Float32, totalTokens, embDim);

            float* dstPtr = GetTensorPtr(result);
            int offset = 0;
            foreach (var emb in embeddings)
            {
                int tokens = (int)emb.Sizes[0];
                float* srcPtr = GetTensorPtr(emb);
                long bytes = (long)tokens * embDim * sizeof(float);
                Buffer.MemoryCopy(srcPtr, dstPtr + offset * embDim, bytes, bytes);
                offset += tokens;
                emb.Dispose();
            }
            return result;
        }

        static unsafe float* GetTensorPtr(TensorSharp.Tensor t)
        {
            if (t.Storage is TensorSharp.GGML.GgmlStorage gs)
                return (float*)gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return (float*)cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException();
        }

        static void PrintTopLogits(float[] logits, ModelBase model, string label)
        {
            var indexed = logits.Select((v, i) => (v, i)).OrderByDescending(x => x.v).Take(10).ToArray();
            Console.Write($"  Top logits [{label}]: ");
            foreach (var (v, i) in indexed)
                Console.Write($"{i}({model.Tokenizer.Vocab[i]})={v:F4} ");
            Console.WriteLine();
        }

        static void RunTests(ModelBase model, int maxTokens, string outputFile)
        {
            Console.WriteLine("=== Running Verification Tests ===\n");

            TestTokenizer(model);
            TestChatTemplate(model);
            TestInferenceWithOllamaComparison(model, maxTokens, outputFile);
        }

        static void TestTokenizer(ModelBase model)
        {
            Console.WriteLine("--- Tokenizer Test ---");

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
                Console.WriteLine($"  Input: \"{input}\"");
                Console.WriteLine($"  Tokens: [{string.Join(", ", tokens)}]");
                Console.WriteLine($"  Decoded: \"{decoded}\"");
                Console.WriteLine($"  Roundtrip match: {match}");
                Console.WriteLine();
            }
        }

        static void TestChatTemplate(ModelBase model)
        {
            Console.WriteLine("--- Chat Template Test ---");

            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = "Hello" }
            };

            string rendered = ChatTemplate.RenderQwen3(messages, true);
            Console.WriteLine($"  Rendered: \"{rendered}\"");
            string expected = "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n";
            Console.WriteLine($"  Expected: \"{expected}\"");
            Console.WriteLine($"  Match: {rendered == expected}");
            Console.WriteLine();
        }

        static void TestInferenceWithOllamaComparison(ModelBase model, int maxTokens, string outputFile)
        {
            Console.WriteLine("--- Inference Comparison Test ---");
            string testInput = "What is 1+1?";

            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = testInput }
            };
            string rendered = ChatTemplate.RenderQwen3(messages, true);

            var inputTokens = model.Tokenizer.Encode(rendered, addSpecial: true);
            Console.WriteLine($"Input tokens ({inputTokens.Count}): [{string.Join(", ", inputTokens)}]");

            model.ResetKVCache();
            float[] logits = model.Forward(inputTokens.ToArray());
            var engineTokens = new List<int>();

            Console.Write("Engine tokens: ");
            for (int step = 0; step < maxTokens; step++)
            {
                int nextToken = model.SampleGreedy(logits);
                Console.Write($"{nextToken} ");

                if (model.Tokenizer.IsEos(nextToken)) break;
                engineTokens.Add(nextToken);
                logits = model.Forward(new[] { nextToken });
            }
            Console.WriteLine();

            string engineText = model.Tokenizer.Decode(engineTokens);
            Console.WriteLine($"Engine output: \"{engineText}\"");

            Console.WriteLine("\nQuerying ollama for comparison...");
            string ollamaResponse = QueryOllama(rendered, maxTokens);
            Console.WriteLine($"Ollama output: \"{ollamaResponse}\"");

            var ollamaTokens = model.Tokenizer.Encode(ollamaResponse, addSpecial: false);
            Console.WriteLine($"\nEngine tokens: [{string.Join(", ", engineTokens)}]");
            Console.WriteLine($"Ollama tokens: [{string.Join(", ", ollamaTokens)}]");

            int matchCount = 0;
            int compareLen = Math.Min(engineTokens.Count, ollamaTokens.Count);
            for (int i = 0; i < compareLen; i++)
            {
                if (engineTokens[i] == ollamaTokens[i])
                    matchCount++;
                else
                {
                    Console.WriteLine($"MISMATCH at position {i}: engine={engineTokens[i]}({model.Tokenizer.Vocab[engineTokens[i]]}) vs ollama={ollamaTokens[i]}({model.Tokenizer.Vocab[ollamaTokens[i]]})");
                    break;
                }
            }
            Console.WriteLine($"\nToken match: {matchCount}/{compareLen} ({(compareLen > 0 ? 100.0 * matchCount / compareLen : 0):F1}%)");
            bool match = engineText == ollamaResponse;
            Console.WriteLine($"Text match: {match}");

            if (outputFile != null)
            {
                File.WriteAllText(outputFile, $"Engine: {engineText}\nOllama: {ollamaResponse}\nMatch: {match}\n");
                Console.WriteLine($"Output written to: {outputFile}");
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
                Console.WriteLine($"Failed to query ollama: {ex.Message}");
                return "";
            }
        }

        static void TestChatTemplates(string modelDir)
        {
            Console.WriteLine($"=== Chat Template Test ===");
            Console.WriteLine($"Scanning: {modelDir}\n");

            var ggufFiles = Directory.GetFiles(modelDir, "*.gguf")
                .Where(f => !Path.GetFileName(f).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                .OrderBy(f => f)
                .ToArray();

            if (ggufFiles.Length == 0)
            {
                Console.WriteLine("No GGUF model files found.");
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
                Console.WriteLine($"--- {fileName} ---");

                try
                {
                    using var gguf = new GgufFile(file);
                    string arch = gguf.GetString("general.architecture");
                    string template = gguf.GetString("tokenizer.chat_template");

                    Console.WriteLine($"  Architecture: {arch}");
                    Console.WriteLine($"  Template: {(template != null ? $"{template.Length} chars" : "(none)")}");

                    if (template == null)
                    {
                        Console.WriteLine($"  SKIP: No chat template in GGUF metadata\n");
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
                            Console.WriteLine($"  [{name}] FAIL - Jinja2 error: {jinja2Error.Message}");
                            allPassed = false;
                            continue;
                        }

                        string j2 = jinja2Result?.Trim() ?? "";
                        string hc = hardcodedResult?.Trim() ?? "";
                        bool match = j2 == hc;

                        if (match)
                        {
                            Console.WriteLine($"  [{name}] PASS - Jinja2 matches hardcoded ({j2.Length} chars)");
                        }
                        else
                        {
                            Console.WriteLine($"  [{name}] MISMATCH");
                            Console.WriteLine($"    Jinja2    ({j2.Length} chars): {Escape(j2)}");
                            Console.WriteLine($"    Hardcoded ({hc.Length} chars): {Escape(hc)}");
                            allPassed = false;
                        }
                    }

                    if (allPassed) passed++; else failed++;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  ERROR: {ex.Message}");
                    failed++;
                }

                Console.WriteLine();
            }

            Console.WriteLine($"=== Results: {passed} passed, {failed} failed, {skipped} skipped out of {ggufFiles.Length} models ===");
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
