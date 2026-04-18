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
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;

namespace TensorSharp.Server
{
    public class ModelService : IDisposable
    {
        private readonly IPromptRenderer _promptRenderer = new GgufPromptRenderer();
        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        private List<int> _cachedTokens;
        private float[] _cachedNextLogits;

        public bool IsLoaded => _model != null;
        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string LoadedModelPath => _loadedModelPath;
        public string LoadedMmProjName => _loadedMmProjPath != null ? Path.GetFileName(_loadedMmProjPath) : null;
        public string LoadedBackend => _model != null ? BackendCatalog.ToBackendValue(_backend) : null;
        public string Architecture => _model?.Config?.Architecture;
        public ModelBase Model => _model;

        /// <summary>
        /// Check if the specified model is already loaded (no locking needed, just a name comparison).
        /// </summary>
        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
        }

        public void InvalidateKVCache()
        {
            _cachedTokens = null;
            _cachedNextLogits = null;
            _model?.ResetKVCache();
        }

        /// <summary>
        /// Load a model. Must be called within the InferenceQueue to prevent concurrent access.
        /// When mmProjPath is null, auto-detection is used. Pass empty string to skip mmproj loading.
        /// </summary>
        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            _cachedTokens = null;
            _cachedNextLogits = null;

            _backend = BackendCatalog.Canonicalize(backendStr) switch
            {
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_cuda" => BackendType.GgmlCuda,
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
            };

            _model = ModelBase.Create(modelPath, _backend);
            _loadedModelPath = modelPath;

            if (mmProjPath == null)
                mmProjPath = AutoDetectMmProj(modelPath);

            if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
            {
                LoadEncoders(mmProjPath);
                _loadedMmProjPath = mmProjPath;
            }
        }

        private string AutoDetectMmProj(string modelPath)
        {
            string dir = Path.GetDirectoryName(modelPath);
            if (dir == null) return null;

            string arch = _model?.Config?.Architecture;
            string[] candidates = arch switch
            {
                "gemma4" => new[] { "gemma-4-mmproj-F16.gguf" },
                "gemma3" => new[] { "mmproj-gemma3-4b-f16.gguf" },
                "qwen35" or "qwen35moe" or "qwen3next" => new[] { "Qwen3.5-mmproj-F16.gguf" },
                _ => Array.Empty<string>()
            };

            foreach (var c in candidates)
            {
                string full = Path.Combine(dir, c);
                if (File.Exists(full)) return full;
            }

            foreach (var f in Directory.GetFiles(dir, "*mmproj*"))
                return f;

            return null;
        }

        private void LoadEncoders(string mmProjPath)
        {
            _model?.MultimodalInjector.LoadProjectors(mmProjPath);
        }

        /// <summary>
        /// Stream chat inference tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// Reuses the KV cache from the previous turn when the rendered text prefix matches.
        /// </summary>
        public async IAsyncEnumerable<string> ChatStreamAsync(
            List<ChatMessage> history,
            int maxTokens,
            [EnumeratorCancellation] CancellationToken cancellationToken,
            SamplingConfig samplingConfig = null,
            List<ToolFunction> tools = null, bool enableThinking = false)
        {
            string arch = _model.Config.Architecture;
            var preparedHistory = PrepareHistoryForInference(history, arch);
            string rendered = _promptRenderer.Render(
                _model.Config.ChatTemplate, preparedHistory, addGenerationPrompt: true,
                architecture: arch, tools: tools, enableThinking: enableThinking);

            Console.Error.WriteLine($"[Prompt] arch={arch}, length={rendered.Length}, first 500 chars:");
            Console.Error.WriteLine(rendered.Length > 500 ? rendered.Substring(0, 500) + "..." : rendered);

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(preparedHistory, inputTokens);
            inputTokens = TruncatePromptToContext(inputTokens, maxTokens);

            float[] logits;
            (_, logits) = PrepareChatPrompt(inputTokens);

            var generatedTokens = new List<int>();
            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var rawBytes = new List<byte>();
            int prevCharLen = 0;
            bool cachedLogitsValid = true;

            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested)
                    break;

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken))
                    break;

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes);
                int validLen = FindValidUtf8Length(rawBytes);
                string decoded = Encoding.UTF8.GetString(rawBytes.GetRange(0, validLen).ToArray());
                string piece = prevCharLen < decoded.Length ? decoded.Substring(prevCharLen) : "";
                prevCharLen = decoded.Length;

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        cachedLogitsValid = false;
                        break;
                    }
                }

                if (piece.Length > 0)
                    yield return piece;

                logits = _model.Forward(new[] { nextToken });
            }

            SaveTokenCache(inputTokens, generatedTokens, logits, cachedLogitsValid);
        }

        private float[] ForwardPromptPrefill(IList<int> tokens, bool allowChunking = true)
        {
            if (tokens == null || tokens.Count == 0)
                throw new ArgumentException("Prompt token list cannot be null or empty.", nameof(tokens));

            if (!allowChunking)
                return _model.ForwardRefill(CopyTokenRange(tokens, 0, tokens.Count));

            int chunkSize = ResolvePrefillChunkSize(_backend, tokens.Count);
            if (chunkSize >= tokens.Count)
                return _model.ForwardRefill(CopyTokenRange(tokens, 0, tokens.Count));

            Console.WriteLine($"[Prompt] Chunking prefill: {tokens.Count} tokens in blocks of {chunkSize} on {LoadedBackend ?? _backend.ToString()}");

            float[] logits = null;
            for (int start = 0; start < tokens.Count; start += chunkSize)
            {
                int length = Math.Min(chunkSize, tokens.Count - start);
                logits = _model.ForwardRefill(CopyTokenRange(tokens, start, length));
            }

            return logits;
        }

        internal static int ResolvePrefillChunkSize(BackendType backend, int tokenCount)
        {
            if (tokenCount <= 0)
                return 0;

            return backend == BackendType.GgmlCuda
                ? Math.Min(tokenCount, 5120)
                : tokenCount;
        }

        internal static int ResolveReusablePrefixForInference(int reusablePrefix, int inputTokenCount, bool hasExactCachedLogits)
        {
            if (reusablePrefix <= 0 || inputTokenCount <= 0)
                return 0;

            if (reusablePrefix < inputTokenCount)
                return reusablePrefix;

            if (hasExactCachedLogits)
                return inputTokenCount;

            return inputTokenCount > 1 ? inputTokenCount - 1 : 0;
        }

        private List<int> TruncatePromptToContext(List<int> inputTokens, int maxTokens)
        {
            int maxCtx = _model.MaxContextLength;
            if (maxCtx <= 0 || inputTokens == null || inputTokens.Count + maxTokens <= maxCtx)
                return inputTokens;

            int available = maxCtx - maxTokens;
            if (available < 1)
            {
                throw new InvalidOperationException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            int trimStart = inputTokens.Count - available;
            trimStart = _model.MultimodalInjector.ClampTrimStart(trimStart);
            int kept = inputTokens.Count - trimStart;
            if (kept < 1)
            {
                throw new InvalidOperationException(
                    $"Prompt ({inputTokens.Count} tokens) exceeds the model's context limit ({maxCtx} tokens). " +
                    "Please shorten the input or reduce attached file size.");
            }

            Console.WriteLine($"[Context] Truncating prompt from {inputTokens.Count} to {kept} tokens (context limit {maxCtx}, reserving {maxTokens} for generation)");
            _model.MultimodalInjector.TrimPreparedPrompt(trimStart);
            _cachedTokens = null;
            _cachedNextLogits = null;
            return inputTokens.GetRange(trimStart, kept);
        }

        private static int[] CopyTokenRange(IList<int> tokens, int start, int length)
        {
            var result = new int[length];
            for (int i = 0; i < length; i++)
                result[i] = tokens[start + i];
            return result;
        }

        private List<int> ProcessMultimodalHistory(List<ChatMessage> history, List<int> inputTokens, string arch)
        {
            if (!HasMultimodalContent(history))
                return inputTokens;

            if (arch == "gemma4")
                return ProcessGemma4History(history, inputTokens);
            if (arch == "gemma3")
                return ProcessGemma3History(history, inputTokens);
            if (_model is Qwen35Model)
                return ProcessQwen35History(history, inputTokens);
            return inputTokens;
        }

        private List<int> ProcessGemma4History(List<ChatMessage> history, List<int> inputTokens)
        {
            if (_model is not Gemma4Model g4)
                return inputTokens;

            int imageStartId = _model.Tokenizer.LookupToken("<|image>");
            int imageEndId = _model.Tokenizer.LookupToken("<image|>");
            if (imageStartId < 0) imageStartId = 255999;
            if (imageEndId < 0) imageEndId = 256000;

            int audioStartId = _model.Tokenizer.LookupToken("<|audio>");
            int audioEndId = _model.Tokenizer.LookupToken("<audio|>");

            var imageProcessor = g4.VisionEncoder != null ? new Gemma4ImageProcessor() : null;
            int searchFrom = 0;

            foreach (var msg in history)
            {
                if (msg.ImagePaths != null && g4.VisionEncoder != null)
                {
                    foreach (var imgPath in msg.ImagePaths)
                    {
                        var (pixels, imgW, imgH) = imageProcessor.ProcessImage(imgPath);
                        var emb = g4.VisionEncoder.Encode(pixels, imgW, imgH);
                        int numTokens = (int)emb.Sizes[0];
                        int pos = FindTokenPosition(inputTokens, imageStartId, searchFrom);

                        if (pos >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, pos, imageStartId, numTokens, imageEndId);
                            g4.SetVisionEmbeddings(emb, pos + 1);
                            searchFrom = pos + numTokens + 2;
                        }
                        else
                        {
                            emb.Dispose();
                        }
                    }
                }

                if (msg.AudioPaths != null && g4.AudioEncoder != null && audioStartId >= 0 && audioEndId >= 0)
                {
                    foreach (var audioPath in msg.AudioPaths)
                    {
                        float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(audioPath);
                        if (samples.Length % 128 != 0)
                        {
                            int padded = samples.Length + (128 - samples.Length % 128);
                            Array.Resize(ref samples, padded);
                        }

                        var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
                        if (melData == null || numFrames == 0)
                            continue;

                        var audioEmb = g4.AudioEncoder.Encode(melData, numFrames);
                        int numAudioTokens = (int)audioEmb.Sizes[0];
                        int pos = FindTokenPosition(inputTokens, audioStartId, searchFrom);

                        if (pos >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, pos, audioStartId, numAudioTokens, audioEndId);
                            g4.SetAudioEmbeddings(audioEmb, pos + 1);
                            searchFrom = pos + numAudioTokens + 2;
                        }
                        else
                        {
                            audioEmb.Dispose();
                        }
                    }
                }
            }

            return inputTokens;
        }

        private List<int> ProcessGemma3History(List<ChatMessage> history, List<int> inputTokens)
        {
            if (_model is not Gemma3Model g3 || g3.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            var processor = new Gemma3ImageProcessor();
            int startId = _model.Tokenizer.LookupToken("<start_of_image>");
            if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
            int endId = Gemma3ImageProcessor.EndOfImageToken;
            int nlnlId = Gemma3ImageProcessor.NewlineNewlineToken;
            int padId = Gemma3ImageProcessor.PadToken;

            inputTokens = ChatTemplate.ExpandGemma3ImageTokens(inputTokens,
                startId, endId, nlnlId, padId, processor.TokensPerImage);

            int searchFrom = 0;
            foreach (var imgPath in imagePaths)
            {
                float[] pixels = processor.ProcessImage(imgPath);
                var emb = g3.VisionEncoder.Encode(pixels);
                int tokenStart = FindGemma3ImageInsertPosition(inputTokens, startId, padId, searchFrom);

                if (tokenStart >= 0)
                {
                    g3.SetVisionEmbeddings(emb, tokenStart);
                    searchFrom = tokenStart + processor.TokensPerImage + 2;
                }
                else
                {
                    emb.Dispose();
                }
            }

            return inputTokens;
        }

        private List<int> ProcessQwen35History(List<ChatMessage> history, List<int> inputTokens)
        {
            if (_model is not Qwen35Model q35 || q35.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            int imagePadId = _model.Tokenizer.LookupToken("<|image_pad|>");
            if (imagePadId < 0)
                return inputTokens;

            var processor = new Qwen35ImageProcessor(q35.VisionEncoder.PatchSize, q35.VisionEncoder.SpatialMergeSize);
            var tokenCounts = new int[imagePaths.Count];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                var (w, h) = Qwen35ImageProcessor.ReadImageDimensions(imagePaths[i]);
                tokenCounts[i] = processor.ComputeImageTokenCount(h, w);
            }

            inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

            int searchFrom = 0;
            for (int i = 0; i < imagePaths.Count; i++)
            {
                var (px, resizedH, resizedW) = processor.ProcessImage(imagePaths[i]);
                var emb = q35.VisionEncoder.Encode(px, resizedH, resizedW);
                int tokenStart = FindTokenPosition(inputTokens, imagePadId, searchFrom);

                if (tokenStart >= 0)
                {
                    q35.SetVisionEmbeddings(emb, tokenStart);
                    searchFrom = tokenStart + tokenCounts[i];
                }
                else
                {
                    emb.Dispose();
                }
            }

            return inputTokens;
        }

        private static List<int> ExpandSingleTokenPlaceholder(
            List<int> inputTokens, int tokenPosition, int startTokenId, int expandedTokenCount, int endTokenId)
        {
            var expanded = new List<int>(inputTokens.Count + expandedTokenCount + 1);
            for (int i = 0; i < tokenPosition; i++)
                expanded.Add(inputTokens[i]);
            expanded.Add(startTokenId);
            for (int i = 0; i < expandedTokenCount; i++)
                expanded.Add(0);
            expanded.Add(endTokenId);
            for (int i = tokenPosition + 1; i < inputTokens.Count; i++)
                expanded.Add(inputTokens[i]);
            return expanded;
        }

        private static int FindTokenPosition(List<int> tokens, int tokenId, int searchFrom)
        {
            for (int i = Math.Max(0, searchFrom); i < tokens.Count; i++)
            {
                if (tokens[i] == tokenId)
                    return i;
            }
            return -1;
        }

        private static int FindGemma3ImageInsertPosition(List<int> tokens, int startTokenId, int padTokenId, int searchFrom)
        {
            for (int i = Math.Max(0, searchFrom); i + 1 < tokens.Count; i++)
            {
                if (tokens[i] == startTokenId && tokens[i + 1] == padTokenId)
                    return i + 1;
            }
            return -1;
        }

        /// <summary>
        /// Ensure the specified model is loaded. Skips loading if already loaded.
        /// Must be called within the InferenceQueue to prevent concurrent access.
        /// Returns false if the model file cannot be found.
        /// </summary>
        public bool EnsureModelLoaded(string modelName, string modelDir, string defaultBackend)
        {
            if (IsModelAlreadyLoaded(modelName))
                return true;

            string modelPath = Path.Combine(modelDir, modelName);
            if (!File.Exists(modelPath))
            {
                var match = Directory.GetFiles(modelDir, "*.gguf")
                    .FirstOrDefault(f => Path.GetFileNameWithoutExtension(f)
                        .Equals(modelName, StringComparison.OrdinalIgnoreCase));
                if (match != null) modelPath = match;
                else
                {
                    match = Directory.GetFiles(modelDir, "*.gguf")
                        .FirstOrDefault(f => Path.GetFileName(f)
                            .Equals(modelName, StringComparison.OrdinalIgnoreCase));
                    if (match != null) modelPath = match;
                    else return false;
                }
            }

            LoadModel(modelPath, null, defaultBackend);
            return _model != null;
        }

        /// <summary>
        /// Stream generate tokens. Must be called within the InferenceQueue to prevent concurrent access.
        /// </summary>
        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, long totalNs, long promptNs, long evalNs)>
            GenerateStreamAsync(
                string prompt,
                List<string> imagePaths,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null)
        {
            string arch = _model.Config.Architecture;
            var messages = new List<ChatMessage>
            {
                new ChatMessage { Role = "user", Content = prompt, ImagePaths = imagePaths }
            };

            var preparedMessages = PrepareHistoryForInference(messages, arch);
            string rendered = _promptRenderer.Render(
                _model.Config.ChatTemplate, preparedMessages, addGenerationPrompt: true,
                architecture: arch);

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(preparedMessages, inputTokens);
            inputTokens = TruncatePromptToContext(inputTokens, maxTokens);

            InvalidateKVCache();

            var sw = Stopwatch.StartNew();
            bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
            float[] logits = ForwardPromptPrefill(inputTokens, allowChunking: !queuedPromptEmbeddings);
            long promptNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            int promptTokenCount = inputTokens.Count;

            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var generatedTokens = new List<int>();
            var rawBytes2 = new List<byte>();
            int prevCharLen2 = 0;

            var evalSw = Stopwatch.StartNew();
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested) break;

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken)) break;

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes2);
                int validLen2 = FindValidUtf8Length(rawBytes2);
                string decoded = Encoding.UTF8.GetString(rawBytes2.GetRange(0, validLen2).ToArray());
                string piece = prevCharLen2 < decoded.Length ? decoded.Substring(prevCharLen2) : "";
                prevCharLen2 = decoded.Length;

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop) break;
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0);
                logits = _model.Forward(new[] { nextToken });
            }

            long evalNs = evalSw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            long totalNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

            yield return ("", true, promptTokenCount, generatedTokens.Count, totalNs, promptNs, evalNs);
        }

        /// <summary>
        /// Stream chat inference tokens with timing metrics. Must be called within the InferenceQueue.
        /// Reuses the KV cache from the previous turn when the rendered text prefix matches.
        /// </summary>
        public async IAsyncEnumerable<(string piece, bool done, int promptTokens, int evalTokens, long totalNs, long promptNs, long evalNs)>
            ChatStreamWithMetricsAsync(
                List<ChatMessage> history,
                int maxTokens,
                [EnumeratorCancellation] CancellationToken cancellationToken,
                SamplingConfig samplingConfig = null,
                List<ToolFunction> tools = null, bool enableThinking = false)
        {
            string arch = _model.Config.Architecture;
            var preparedHistory = PrepareHistoryForInference(history, arch);
            string rendered = _promptRenderer.Render(
                _model.Config.ChatTemplate, preparedHistory, addGenerationPrompt: true,
                architecture: arch, tools: tools, enableThinking: enableThinking);

            Console.Error.WriteLine($"[Prompt] arch={arch}, length={rendered.Length}, sampling: temp={samplingConfig?.Temperature ?? 0.8f}, top_k={samplingConfig?.TopK ?? 40}, top_p={samplingConfig?.TopP ?? 0.9f}");
            Console.Error.WriteLine($"[Prompt] first 500 chars: {(rendered.Length > 500 ? rendered.Substring(0, 500) + "..." : rendered)}");

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            inputTokens = _model.MultimodalInjector.ProcessPromptTokens(preparedHistory, inputTokens);
            inputTokens = TruncatePromptToContext(inputTokens, maxTokens);

            int promptTokenCount;
            var sw = Stopwatch.StartNew();
            float[] logits;
            (promptTokenCount, logits) = PrepareChatPrompt(inputTokens);

            long promptNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var generatedTokens = new List<int>();
            var rawBytes3 = new List<byte>();
            int prevCharLen3 = 0;
            bool cachedLogitsValid = true;

            var evalSw = Stopwatch.StartNew();
            for (int step = 0; step < maxTokens; step++)
            {
                if (cancellationToken.IsCancellationRequested) break;

                int nextToken = sampler.Sample(logits, generatedTokens);
                if (_model.Tokenizer.IsEos(nextToken)) break;

                generatedTokens.Add(nextToken);
                _model.Tokenizer.AppendTokenBytes(nextToken, rawBytes3);
                int validLen3 = FindValidUtf8Length(rawBytes3);
                string decoded = Encoding.UTF8.GetString(rawBytes3.GetRange(0, validLen3).ToArray());
                string piece = prevCharLen3 < decoded.Length ? decoded.Substring(prevCharLen3) : "";
                prevCharLen3 = decoded.Length;

                if (cfg.StopSequences != null && cfg.StopSequences.Count > 0)
                {
                    var (_, shouldStop) = sampler.CheckStopSequences(decoded);
                    if (shouldStop)
                    {
                        cachedLogitsValid = false;
                        break;
                    }
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0);
                logits = _model.Forward(new[] { nextToken });
            }

            long evalNs = evalSw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            long totalNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

            SaveTokenCache(inputTokens, generatedTokens, logits, cachedLogitsValid);

            yield return ("", true, promptTokenCount, generatedTokens.Count, totalNs, promptNs, evalNs);
        }

        /// <summary>
        /// Find the length of the longest prefix of the byte buffer that forms valid UTF-8.
        /// Strips any trailing incomplete multi-byte sequence.
        /// </summary>
        private static int FindValidUtf8Length(List<byte> bytes)
        {
            int len = bytes.Count;
            if (len == 0) return 0;

            for (int i = 1; i <= Math.Min(4, len); i++)
            {
                byte b = bytes[len - i];
                if ((b & 0x80) == 0) return len;
                if ((b & 0xE0) == 0xC0) return (i >= 2) ? len : len - i;
                if ((b & 0xF0) == 0xE0) return (i >= 3) ? len : len - i;
                if ((b & 0xF8) == 0xF0) return (i >= 4) ? len : len - i;
                if ((b & 0xC0) == 0x80) continue;
                return len;
            }
            return len;
        }

        /// <summary>
        /// Compute a usable common prefix length between cached tokens and new input.
        /// Returns 0 (meaning full reset) when:
        ///   - No cache exists
        ///   - The model doesn't support KV cache truncation (e.g. Qwen3.5 recurrent layers)
        ///   - The savings would be less than 10% (not worth the risk)
        ///   - The common prefix is shorter than 4 tokens
        ///
        /// For models with sliding window attention (SWA), KV cache truncation is
        /// disabled because the prefill path reads the SWA ring buffer linearly but
        /// the entries are not in positional order after truncation, which corrupts
        /// attention output and causes degenerate tokens.
        /// </summary>
        private int ComputeUsablePrefix(List<int> inputTokens)
        {
            int reusablePrefix = _model?.KVCachePolicy.ComputeReusablePrefix(_model, _cachedTokens, inputTokens, hasMultimodal: false) ?? 0;
            return _model?.MultimodalInjector.ClampReusablePrefix(reusablePrefix) ?? reusablePrefix;
        }

        /// <summary>
        /// Save the full token sequence (prompt + generated) so the next turn can find
        /// the longest common prefix and reuse the KV cache up to that point.
        /// Works correctly for all models including thinking/Harmony because it compares
        /// actual token IDs rather than re-rendered text.
        /// </summary>
        private (int forwardedTokenCount, float[] logits) PrepareChatPrompt(List<int> inputTokens)
        {
            if (TryReuseFullPromptLogits(inputTokens, out float[] cachedLogits))
            {
                Console.WriteLine($"[KV cache] Reusing {inputTokens.Count} cached tokens, forwarding 0 new tokens (saved 100%)");
                return (0, cachedLogits);
            }

            int commonPrefix = ComputeUsablePrefix(inputTokens);
            commonPrefix = ResolveReusablePrefixForInference(commonPrefix, inputTokens.Count, false);
            commonPrefix = _model.MultimodalInjector.ClampReusablePrefix(commonPrefix);

            if (commonPrefix > 0)
            {
                _model.TruncateKVCache(commonPrefix);
                var suffixTokens = inputTokens.GetRange(commonPrefix, inputTokens.Count - commonPrefix).ToArray();
                bool queuedPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(commonPrefix);
                Console.WriteLine($"[KV cache] Reusing {commonPrefix} cached tokens, forwarding {suffixTokens.Length} new tokens (saved {100.0 * commonPrefix / inputTokens.Count:F0}%)");
                return (suffixTokens.Length, ForwardPromptPrefill(suffixTokens, allowChunking: !queuedPromptEmbeddings));
            }

            Console.Error.WriteLine($"[Prompt] Encoded to {inputTokens.Count} tokens (full prompt)");
            if (_cachedTokens != null)
                Console.WriteLine("[KV cache] Reset (no usable common prefix)");

            _model.ResetKVCache();
            bool queuedFullPromptEmbeddings = _model.MultimodalInjector.QueuePromptEmbeddings(0);
            return (inputTokens.Count, ForwardPromptPrefill(inputTokens, allowChunking: !queuedFullPromptEmbeddings));
        }

        private bool TryReuseFullPromptLogits(List<int> inputTokens, out float[] logits)
        {
            logits = null;
            if (_cachedTokens == null || _cachedNextLogits == null || inputTokens == null || _cachedTokens.Count != inputTokens.Count)
                return false;

            for (int i = 0; i < inputTokens.Count; i++)
            {
                if (_cachedTokens[i] != inputTokens[i])
                    return false;
            }

            logits = (float[])_cachedNextLogits.Clone();
            return true;
        }

        private void SaveTokenCache(List<int> promptTokens, List<int> generatedTokens, float[] nextLogits, bool nextLogitsValid)
        {
            _cachedTokens = new List<int>(promptTokens.Count + generatedTokens.Count);
            _cachedTokens.AddRange(promptTokens);
            _cachedTokens.AddRange(generatedTokens);
            _cachedNextLogits = nextLogitsValid && nextLogits != null ? (float[])nextLogits.Clone() : null;
        }

        /// <summary>
        /// Find the length of the longest common prefix between the cached token sequence
        /// and the new input tokens. Returns 0 if there is no cache or no common prefix.
        /// </summary>
        internal static int FindTokenPrefixLength(List<int> cached, List<int> newTokens)
        {
            if (cached == null || cached.Count == 0 || newTokens == null || newTokens.Count == 0)
                return 0;

            int maxLen = Math.Min(cached.Count, newTokens.Count);
            int common = 0;
            for (int i = 0; i < maxLen; i++)
            {
                if (cached[i] != newTokens[i])
                    break;
                common++;
            }

            if (common == 0 || common >= newTokens.Count)
                return 0;

            return common;
        }

        internal static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            if (arch != "gemma4" || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampled = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames)
                .Select(i => msg.ImagePaths[i])
                .ToList();

            Console.WriteLine($"[video] Downsampled {msg.ImagePaths.Count} frames to {sampled.Count} evenly spaced frames for Gemma4 prefill stability.");

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                Thinking = msg.Thinking
            };
        }

        internal static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        internal static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        internal static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        public List<string> ScanModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(f => !IsMmProjFile(f))
                .OrderBy(f => f)
                .ToList();
        }

        public List<string> ScanMmProjModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .Where(IsMmProjFile)
                .OrderBy(f => f)
                .ToList();
        }

        private static bool IsMmProjFile(string fileName)
        {
            return fileName.IndexOf("mmproj", StringComparison.OrdinalIgnoreCase) >= 0;
        }

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _cachedTokens = null;
            _cachedNextLogits = null;
        }
    }
}

