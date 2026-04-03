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
using System.IO;
using System.Linq;
using System.Text;
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

            for (int i = 0; i < args.Length; i++)
            {
                switch (args[i])
                {
                    case "--model": modelPath = args[++i]; break;
                    case "--input": inputFile = args[++i]; break;
                    case "--output": outputFile = args[++i]; break;
                    case "--image": imagePath = args[++i]; break;
                    case "--audio": audioPath = args[++i]; break;
                    case "--video": videoPath = args[++i]; break;
                    case "--mmproj": mmProjPath = args[++i]; break;
                    case "--max-tokens": maxTokens = int.Parse(args[++i]); break;
                    case "--test": runTest = true; break;
                    case "--backend": backendStr = args[++i].ToLowerInvariant(); break;
                }
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
                    "[--image <image.png>] [--output <output.txt>] [--max-tokens N] [--test] " +
                    "[--backend cpu|ggml_cpu|ggml_metal]");
                return;
            }

            BackendType backend = backendStr switch
            {
                "cpu" => BackendType.Cpu,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_metal" => BackendType.GgmlMetal,
                _ => throw new ArgumentException($"Unknown backend '{backendStr}'. Use: cpu, ggml_cpu, ggml_metal"),
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

            string result = RunInference(model, rawText, imagePaths, maxTokens, audioPaths, isVideo: videoPath != null);

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

        static string RunInference(ModelBase model, string rawText, List<string> imagePaths, int maxTokens,
            List<string> audioPaths = null, bool isVideo = false)
        {
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = rawText, ImagePaths = imagePaths, AudioPaths = audioPaths, IsVideo = isVideo }
            };

            string rendered = ChatTemplate.RenderFromGgufTemplate(
                model.Config.ChatTemplate, messages, addGenerationPrompt: true,
                architecture: model.Config.Architecture);

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

            for (int step = 0; step < maxTokens; step++)
            {
                int nextToken = model.SampleGreedy(logits);
                Console.Write($"[{nextToken}:{model.Tokenizer.Vocab[nextToken]}]");

                if (model.Tokenizer.IsEos(nextToken))
                {
                    Console.WriteLine(" <EOS>");
                    break;
                }

                generatedTokens.Add(nextToken);

                logits = model.Forward(new[] { nextToken });
                if (step < 3)
                    PrintTopLogits(logits, model, $"decode_{step}");
            }

            Console.WriteLine();
            model.PrintTimingStats();
            string decoded = model.Tokenizer.Decode(generatedTokens);
            return decoded;
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
    }
}
