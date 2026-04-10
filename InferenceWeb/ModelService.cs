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
using InferenceEngine;
using TensorSharp;
using TensorSharp.Cpu;

namespace InferenceWeb
{
    public class ModelService : IDisposable
    {
        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        private List<int> _cachedTokens;

        public bool IsLoaded => _model != null;
        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string LoadedModelPath => _loadedModelPath;
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
            _model?.ResetKVCache();
        }

        /// <summary>
        /// Load a model. Must be called within the InferenceQueue to prevent concurrent access.
        /// </summary>
        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            _cachedTokens = null;

            _backend = backendStr switch
            {
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "cuda" or "ggml_cuda" => BackendType.GgmlCuda,
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
            };

            _model = ModelBase.Create(modelPath, _backend);
            _loadedModelPath = modelPath;

            if (mmProjPath == null)
                mmProjPath = AutoDetectMmProj(modelPath);

            if (mmProjPath != null && File.Exists(mmProjPath))
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
            switch (_model)
            {
                case Gemma4Model g4:
                    g4.LoadVisionEncoder(mmProjPath);
                    g4.LoadAudioEncoder(mmProjPath);
                    break;
                case Gemma3Model g3:
                    g3.LoadVisionEncoder(mmProjPath);
                    break;
                case Qwen35Model q35:
                    q35.LoadVisionEncoder(mmProjPath);
                    break;
            }
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
            string rendered = ChatTemplate.RenderFromGgufTemplate(
                _model.Config.ChatTemplate, preparedHistory, addGenerationPrompt: true,
                architecture: arch, tools: tools, enableThinking: enableThinking);

            Console.Error.WriteLine($"[Prompt] arch={arch}, length={rendered.Length}, first 500 chars:");
            Console.Error.WriteLine(rendered.Length > 500 ? rendered.Substring(0, 500) + "..." : rendered);

            var lastMsg = preparedHistory.LastOrDefault(m => m.Role == "user");
            bool hasMultimodal = HasMultimodalContent(lastMsg);

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            if (lastMsg != null)
                inputTokens = ProcessMultimodal(lastMsg, inputTokens, arch);

            float[] logits;
            int commonPrefix = ComputeUsablePrefix(inputTokens, hasMultimodal);

            if (commonPrefix > 0)
            {
                _model.TruncateKVCache(commonPrefix);
                var suffixTokens = inputTokens.GetRange(commonPrefix, inputTokens.Count - commonPrefix).ToArray();
                Console.WriteLine($"[KV cache] Reusing {commonPrefix} cached tokens, forwarding {suffixTokens.Length} new tokens (saved {100.0 * commonPrefix / inputTokens.Count:F0}%)");
                logits = _model.Forward(suffixTokens);
            }
            else
            {
                Console.Error.WriteLine($"[Prompt] Encoded to {inputTokens.Count} tokens (full prompt)");
                if (_cachedTokens != null)
                    Console.WriteLine("[KV cache] Reset (no usable common prefix)");
                _model.ResetKVCache();
                logits = _model.Forward(inputTokens.ToArray());
            }

            var generatedTokens = new List<int>();
            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var rawBytes = new List<byte>();
            int prevCharLen = 0;

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
                        break;
                }

                if (piece.Length > 0)
                    yield return piece;

                logits = _model.Forward(new[] { nextToken });
            }

            SaveTokenCache(inputTokens, generatedTokens);
        }

        private List<int> ProcessMultimodal(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (msg.ImagePaths != null && msg.ImagePaths.Count > 0)
                inputTokens = ProcessImages(msg, inputTokens, arch);

            if (msg.AudioPaths != null && msg.AudioPaths.Count > 0)
                inputTokens = ProcessAudio(msg, inputTokens, arch);

            return inputTokens;
        }

        private List<int> ProcessImages(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (arch == "gemma4" && _model is Gemma4Model g4 && g4.VisionEncoder != null)
            {
                int imageStartId = _model.Tokenizer.LookupToken("<|image>");
                int imageEndId = _model.Tokenizer.LookupToken("<image|>");
                if (imageStartId < 0) imageStartId = 255999;
                if (imageEndId < 0) imageEndId = 256000;

                var proc = new Gemma4ImageProcessor();
                int searchFrom = 0;
                foreach (var imgPath in msg.ImagePaths)
                {
                    var (pixels, imgW, imgH) = proc.ProcessImage(imgPath);
                    var emb = g4.VisionEncoder.Encode(pixels, imgW, imgH);
                    int numTokens = (int)emb.Sizes[0];

                    int pos = -1;
                    for (int i = searchFrom; i < inputTokens.Count; i++)
                    {
                        if (inputTokens[i] == imageStartId) { pos = i; break; }
                    }

                    if (pos >= 0)
                    {
                        var expanded = new List<int>();
                        for (int i = 0; i < pos; i++) expanded.Add(inputTokens[i]);
                        expanded.Add(imageStartId);
                        for (int i = 0; i < numTokens; i++) expanded.Add(0);
                        expanded.Add(imageEndId);
                        for (int i = pos + 1; i < inputTokens.Count; i++) expanded.Add(inputTokens[i]);
                        inputTokens = expanded;

                        g4.SetVisionEmbeddings(emb, pos + 1);
                        searchFrom = pos + 1 + numTokens + 1;
                    }
                    else
                    {
                        emb.Dispose();
                    }
                }
            }
            else if (arch == "gemma3" && _model is Gemma3Model g3 && g3.VisionEncoder != null)
            {
                var proc = new Gemma3ImageProcessor();
                int startId = _model.Tokenizer.LookupToken("<start_of_image>");
                if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
                int endId = Gemma3ImageProcessor.EndOfImageToken;
                int nlnlId = Gemma3ImageProcessor.NewlineNewlineToken;
                int padId = Gemma3ImageProcessor.PadToken;

                inputTokens = ChatTemplate.ExpandGemma3ImageTokens(inputTokens,
                    startId, endId, nlnlId, padId, proc.TokensPerImage);

                float[] pixels = proc.ProcessImage(msg.ImagePaths[0]);
                var emb = g3.VisionEncoder.Encode(pixels);

                int tokenStart = -1;
                for (int i = 0; i < inputTokens.Count; i++)
                {
                    if (inputTokens[i] == startId && i + 1 < inputTokens.Count && inputTokens[i + 1] == padId)
                    { tokenStart = i + 1; break; }
                }
                if (tokenStart >= 0) g3.SetVisionEmbeddings(emb, tokenStart);
                else emb.Dispose();
            }
            else if (_model is Qwen35Model q35 && q35.VisionEncoder != null)
            {
                int imagePadId = _model.Tokenizer.LookupToken("<|image_pad|>");
                if (imagePadId >= 0)
                {
                    var processor = new Qwen35ImageProcessor(q35.VisionEncoder.PatchSize, q35.VisionEncoder.SpatialMergeSize);
                    var tokenCounts = new int[msg.ImagePaths.Count];
                    for (int i = 0; i < msg.ImagePaths.Count; i++)
                    {
                        var (w, h) = Qwen35ImageProcessor.ReadImageDimensions(msg.ImagePaths[i]);
                        tokenCounts[i] = processor.ComputeImageTokenCount(h, w);
                    }
                    inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

                    var (px, rH, rW) = processor.ProcessImage(msg.ImagePaths[0]);
                    var emb = q35.VisionEncoder.Encode(px, rH, rW);
                    int tokenStart = inputTokens.IndexOf(imagePadId);
                    if (tokenStart >= 0) q35.SetVisionEmbeddings(emb, tokenStart);
                    else emb.Dispose();
                }
            }
            return inputTokens;
        }

        private List<int> ProcessAudio(ChatMessage msg, List<int> inputTokens, string arch)
        {
            if (arch != "gemma4" || _model is not Gemma4Model g4a || g4a.AudioEncoder == null)
                return inputTokens;

            int audioStartId = _model.Tokenizer.LookupToken("<|audio>");
            int audioEndId = _model.Tokenizer.LookupToken("<audio|>");

            float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(msg.AudioPaths[0]);
            if (samples.Length % 128 != 0)
            {
                int padded = samples.Length + (128 - samples.Length % 128);
                Array.Resize(ref samples, padded);
            }

            var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
            if (melData == null || numFrames == 0) return inputTokens;

            var audioEmb = g4a.AudioEncoder.Encode(melData, numFrames);
            int numAudioTokens = (int)audioEmb.Sizes[0];

            int audioPos = -1;
            for (int i = 0; i < inputTokens.Count; i++)
            {
                if (inputTokens[i] == audioStartId) { audioPos = i; break; }
            }

            if (audioPos >= 0)
            {
                var expanded = new List<int>();
                for (int i = 0; i < audioPos; i++) expanded.Add(inputTokens[i]);
                expanded.Add(audioStartId);
                for (int i = 0; i < numAudioTokens; i++) expanded.Add(0);
                expanded.Add(audioEndId);
                for (int i = audioPos + 1; i < inputTokens.Count; i++) expanded.Add(inputTokens[i]);
                inputTokens = expanded;
                g4a.SetAudioEmbeddings(audioEmb, audioPos + 1);
            }
            else
            {
                audioEmb.Dispose();
            }

            return inputTokens;
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
            string rendered = ChatTemplate.RenderFromGgufTemplate(
                _model.Config.ChatTemplate, preparedMessages, addGenerationPrompt: true,
                architecture: arch);

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            var lastMsg = preparedMessages[0];
            if (lastMsg.ImagePaths != null && lastMsg.ImagePaths.Count > 0)
                inputTokens = ProcessMultimodal(lastMsg, inputTokens, arch);

            InvalidateKVCache();

            var sw = Stopwatch.StartNew();
            float[] logits = _model.Forward(inputTokens.ToArray());
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
            string rendered = ChatTemplate.RenderFromGgufTemplate(
                _model.Config.ChatTemplate, preparedHistory, addGenerationPrompt: true,
                architecture: arch, tools: tools, enableThinking: enableThinking);

            Console.Error.WriteLine($"[Prompt] arch={arch}, length={rendered.Length}, sampling: temp={samplingConfig?.Temperature ?? 0.8f}, top_k={samplingConfig?.TopK ?? 40}, top_p={samplingConfig?.TopP ?? 0.9f}");
            Console.Error.WriteLine($"[Prompt] first 500 chars: {(rendered.Length > 500 ? rendered.Substring(0, 500) + "..." : rendered)}");

            var lastMsg = preparedHistory.LastOrDefault(m => m.Role == "user");
            bool hasMultimodal = HasMultimodalContent(lastMsg);

            var inputTokens = _model.Tokenizer.Encode(rendered, addSpecial: true);
            if (lastMsg != null)
                inputTokens = ProcessMultimodal(lastMsg, inputTokens, arch);

            int promptTokenCount;
            var sw = Stopwatch.StartNew();
            float[] logits;
            int commonPrefix = ComputeUsablePrefix(inputTokens, hasMultimodal);

            if (commonPrefix > 0)
            {
                _model.TruncateKVCache(commonPrefix);
                var suffixTokens = inputTokens.GetRange(commonPrefix, inputTokens.Count - commonPrefix).ToArray();
                Console.WriteLine($"[KV cache] Reusing {commonPrefix} cached tokens, forwarding {suffixTokens.Length} new tokens (saved {100.0 * commonPrefix / inputTokens.Count:F0}%)");
                logits = _model.Forward(suffixTokens);
                promptTokenCount = suffixTokens.Length;
            }
            else
            {
                Console.Error.WriteLine($"[Prompt] Encoded to {inputTokens.Count} tokens (full prompt)");
                if (_cachedTokens != null)
                    Console.WriteLine("[KV cache] Reset (no usable common prefix)");
                _model.ResetKVCache();
                logits = _model.Forward(inputTokens.ToArray());
                promptTokenCount = inputTokens.Count;
            }

            long promptNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

            var cfg = samplingConfig ?? SamplingConfig.Default;
            var sampler = new TokenSampler(cfg);
            var generatedTokens = new List<int>();
            var rawBytes3 = new List<byte>();
            int prevCharLen3 = 0;

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
                    if (shouldStop) break;
                }

                if (piece.Length > 0)
                    yield return (piece, false, 0, 0, 0, 0, 0);
                logits = _model.Forward(new[] { nextToken });
            }

            long evalNs = evalSw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);
            long totalNs = sw.ElapsedTicks * (1_000_000_000L / Stopwatch.Frequency);

            SaveTokenCache(inputTokens, generatedTokens);

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
        ///   - No cache exists or multimodal content is present
        ///   - The model doesn't support KV cache truncation (e.g. Qwen3.5 recurrent layers)
        ///   - The savings would be less than 10% (not worth the risk)
        ///   - The common prefix is shorter than 4 tokens
        /// </summary>
        private int ComputeUsablePrefix(List<int> inputTokens, bool hasMultimodal)
        {
            if (hasMultimodal || !_model.SupportsKVCacheTruncation)
                return 0;

            int raw = FindTokenPrefixLength(_cachedTokens, inputTokens);
            if (raw <= 0)
                return 0;

            int suffixLen = inputTokens.Count - raw;

            if (raw < 4)
            {
                Console.WriteLine($"[KV cache] Common prefix too short ({raw} tokens), doing full reset");
                return 0;
            }

            double savingsRatio = (double)raw / inputTokens.Count;
            if (savingsRatio < 0.10)
            {
                Console.WriteLine($"[KV cache] Savings too small ({raw}/{inputTokens.Count} = {100 * savingsRatio:F0}%), doing full reset");
                return 0;
            }

            if (_cachedTokens != null && raw < _cachedTokens.Count)
            {
                string cachedTokStr = _cachedTokens.Count > raw ? _cachedTokens[raw].ToString() : "N/A";
                string newTokStr = inputTokens.Count > raw ? inputTokens[raw].ToString() : "N/A";
                Console.WriteLine($"[KV cache] Divergence at index {raw}: cached={cachedTokStr}, new={newTokStr} (cached total={_cachedTokens.Count}, new total={inputTokens.Count})");
            }

            return raw;
        }

        /// <summary>
        /// Save the full token sequence (prompt + generated) so the next turn can find
        /// the longest common prefix and reuse the KV cache up to that point.
        /// Works correctly for all models including thinking/Harmony because it compares
        /// actual token IDs rather than re-rendered text.
        /// </summary>
        private void SaveTokenCache(List<int> promptTokens, List<int> generatedTokens)
        {
            _cachedTokens = new List<int>(promptTokens.Count + generatedTokens.Count);
            _cachedTokens.AddRange(promptTokens);
            _cachedTokens.AddRange(generatedTokens);
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

        private static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
        {
            if (history == null || history.Count == 0)
                return history;

            int lastUserIdx = history.FindLastIndex(m => m.Role == "user");
            if (lastUserIdx < 0)
                return history;

            var normalized = NormalizeMessageForInference(history[lastUserIdx], arch);
            if (ReferenceEquals(normalized, history[lastUserIdx]))
                return history;

            var prepared = new List<ChatMessage>(history);
            prepared[lastUserIdx] = normalized;
            return prepared;
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

        private static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        public List<string> ScanModels(string directory)
        {
            if (!Directory.Exists(directory)) return new List<string>();
            return Directory.GetFiles(directory, "*.gguf")
                .Select(Path.GetFileName)
                .OrderBy(f => f)
                .ToList();
        }

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
        }
    }
}
