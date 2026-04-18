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
using TensorSharp;

namespace TensorSharp.Models
{
    internal sealed class ModelMultimodalInjector : IMultimodalInjector, IDisposable
    {
        private readonly ModelBase _model;
        private readonly Dictionary<string, CachedEmbedding> _visionCache = new(StringComparer.OrdinalIgnoreCase);
        private readonly Dictionary<string, CachedEmbedding> _audioCache = new(StringComparer.OrdinalIgnoreCase);
        private readonly List<PreparedEmbeddingSpan> _preparedVisionEmbeddings = new();
        private readonly List<PreparedEmbeddingSpan> _preparedAudioEmbeddings = new();

        private sealed class CachedEmbedding : IDisposable
        {
            public CachedEmbedding(
                string fullPath,
                long fileSize,
                long lastWriteUtcTicks,
                Tensor embeddings,
                int tokenCount,
                int extra0 = 0,
                int extra1 = 0)
            {
                FullPath = fullPath;
                FileSize = fileSize;
                LastWriteUtcTicks = lastWriteUtcTicks;
                Embeddings = embeddings;
                TokenCount = tokenCount;
                Extra0 = extra0;
                Extra1 = extra1;
            }

            public string FullPath { get; }
            public long FileSize { get; }
            public long LastWriteUtcTicks { get; }
            public Tensor Embeddings { get; }
            public int TokenCount { get; }
            public int Extra0 { get; }
            public int Extra1 { get; }

            public bool Matches(long fileSize, long lastWriteUtcTicks) =>
                FileSize == fileSize && LastWriteUtcTicks == lastWriteUtcTicks;

            public void Dispose()
            {
                Embeddings?.Dispose();
            }
        }

        private sealed class PreparedEmbeddingSpan
        {
            public PreparedEmbeddingSpan(
                CachedEmbedding cacheEntry,
                int insertPosition,
                int promptTokenStart,
                int promptTokenEndExclusive)
            {
                CacheEntry = cacheEntry;
                InsertPosition = insertPosition;
                PromptTokenStart = promptTokenStart;
                PromptTokenEndExclusive = promptTokenEndExclusive;
            }

            public CachedEmbedding CacheEntry { get; }
            public int InsertPosition { get; set; }
            public int PromptTokenStart { get; set; }
            public int PromptTokenEndExclusive { get; set; }
            public int EndPosition => InsertPosition + CacheEntry.TokenCount;
        }

        public ModelMultimodalInjector(ModelBase model)
        {
            _model = model;
        }

        public void LoadProjectors(string mmProjPath)
        {
            if (string.IsNullOrWhiteSpace(mmProjPath))
                return;

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
                case Mistral3Model m3:
                    m3.LoadVisionEncoder(mmProjPath);
                    break;
            }
        }

        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens)
        {
            ClearPreparedPromptState();

            if (history == null || history.Count == 0 || inputTokens == null || inputTokens.Count == 0)
                return inputTokens;

            if (_model is Gemma4Model g4)
                return ProcessGemma4History(g4, history, inputTokens);
            if (_model is Gemma3Model g3)
                return ProcessGemma3History(g3, history, inputTokens);
            if (_model is Qwen35Model q35)
                return ProcessQwen35History(q35, history, inputTokens);
            if (_model is Mistral3Model m3)
                return ProcessMistral3History(m3, history, inputTokens);

            return inputTokens;
        }

        public int ClampReusablePrefix(int reusablePrefixTokenCount)
        {
            int clamped = ClampReusablePrefix(reusablePrefixTokenCount, _preparedVisionEmbeddings);
            clamped = ClampReusablePrefix(clamped, _preparedAudioEmbeddings);
            return clamped;
        }

        public int ClampTrimStart(int trimStartTokenCount)
        {
            int clamped = ClampTrimStart(trimStartTokenCount, _preparedVisionEmbeddings);
            clamped = ClampTrimStart(clamped, _preparedAudioEmbeddings);
            return clamped;
        }

        public void TrimPreparedPrompt(int trimStartTokenCount)
        {
            TrimPreparedPrompt(_preparedVisionEmbeddings, trimStartTokenCount);
            TrimPreparedPrompt(_preparedAudioEmbeddings, trimStartTokenCount);
        }

        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount)
        {
            bool queued = QueuePreparedVisionEmbeddings(reusablePrefixTokenCount);
            queued |= QueuePreparedAudioEmbeddings(reusablePrefixTokenCount);
            return queued;
        }

        private List<int> ProcessGemma4History(Gemma4Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            int imageStartId = _model.Tokenizer.LookupToken("<|image>");
            int imageEndId = _model.Tokenizer.LookupToken("<image|>");
            if (imageStartId < 0) imageStartId = 255999;
            if (imageEndId < 0) imageEndId = 256000;

            int audioStartId = _model.Tokenizer.LookupToken("<|audio>");
            int audioEndId = _model.Tokenizer.LookupToken("<audio|>");

            var imageProcessor = model.VisionEncoder != null ? new Gemma4ImageProcessor() : null;
            int searchFrom = 0;

            foreach (var message in history)
            {
                if (message.ImagePaths != null && model.VisionEncoder != null)
                {
                    foreach (var imagePath in message.ImagePaths)
                    {
                        CachedEmbedding cached = GetOrCreateGemma4VisionEmbedding(model, imageProcessor, imagePath);
                        int tokenPosition = FindTokenPosition(inputTokens, imageStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, imageStartId, cached.TokenCount, imageEndId);
                            _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                                cached,
                                tokenPosition + 1,
                                tokenPosition,
                                tokenPosition + cached.TokenCount + 2));
                            searchFrom = tokenPosition + cached.TokenCount + 2;
                        }
                    }
                }

                if (message.AudioPaths != null && model.AudioEncoder != null && audioStartId >= 0 && audioEndId >= 0)
                {
                    foreach (var audioPath in message.AudioPaths)
                    {
                        CachedEmbedding cached = GetOrCreateGemma4AudioEmbedding(model, audioPath);
                        int tokenPosition = FindTokenPosition(inputTokens, audioStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, audioStartId, cached.TokenCount, audioEndId);
                            _preparedAudioEmbeddings.Add(new PreparedEmbeddingSpan(
                                cached,
                                tokenPosition + 1,
                                tokenPosition,
                                tokenPosition + cached.TokenCount + 2));
                            searchFrom = tokenPosition + cached.TokenCount + 2;
                        }
                    }
                }
            }

            return inputTokens;
        }

        private List<int> ProcessGemma3History(Gemma3Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            var processor = new Gemma3ImageProcessor();
            int startId = _model.Tokenizer.LookupToken("<start_of_image>");
            if (startId < 0) startId = Gemma3ImageProcessor.StartOfImageToken;
            int endId = Gemma3ImageProcessor.EndOfImageToken;
            int newlineId = Gemma3ImageProcessor.NewlineNewlineToken;
            int padId = Gemma3ImageProcessor.PadToken;

            inputTokens = ChatTemplate.ExpandGemma3ImageTokens(
                inputTokens,
                startId,
                endId,
                newlineId,
                padId,
                processor.TokensPerImage);

            int searchFrom = 0;
            foreach (var imagePath in imagePaths)
            {
                CachedEmbedding cached = GetOrCreateGemma3VisionEmbedding(model, processor, imagePath);
                int tokenStart = FindGemma3ImageInsertPosition(inputTokens, startId, padId, searchFrom);

                if (tokenStart >= 0)
                {
                    _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                        cached,
                        tokenStart,
                        tokenStart - 2,
                        tokenStart + cached.TokenCount + 2));
                    searchFrom = tokenStart + processor.TokensPerImage + 2;
                }
            }

            return inputTokens;
        }

        private List<int> ProcessQwen35History(Qwen35Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            int imagePadId = _model.Tokenizer.LookupToken("<|image_pad|>");
            if (imagePadId < 0)
                return inputTokens;

            var processor = new Qwen35ImageProcessor(model.VisionEncoder.PatchSize, model.VisionEncoder.SpatialMergeSize);
            var cachedEmbeddings = new CachedEmbedding[imagePaths.Count];
            var tokenCounts = new int[imagePaths.Count];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                cachedEmbeddings[i] = GetOrCreateQwen35VisionEmbedding(model, processor, imagePaths[i]);
                tokenCounts[i] = cachedEmbeddings[i].TokenCount;
            }

            inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

            int searchFrom = 0;
            for (int i = 0; i < imagePaths.Count; i++)
            {
                int tokenStart = FindTokenPosition(inputTokens, imagePadId, searchFrom);

                if (tokenStart >= 0)
                {
                    _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                        cachedEmbeddings[i],
                        tokenStart,
                        tokenStart,
                        tokenStart + tokenCounts[i]));
                    searchFrom = tokenStart + tokenCounts[i];
                }
            }

            return inputTokens;
        }

        private List<int> ProcessMistral3History(Mistral3Model model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            var imagePaths = GetImagePathsInPromptOrder(history);
            if (imagePaths.Count == 0)
                return inputTokens;

            var processor = new Mistral3ImageProcessor(
                model.VisionEncoder.ImageSize,
                model.VisionEncoder.PatchSize);

            int searchFrom = 0;
            foreach (var imagePath in imagePaths)
            {
                CachedEmbedding cached = GetOrCreateMistral3VisionEmbedding(model, processor, imagePath);
                int numRows = cached.Extra0;
                int numCols = cached.Extra1;

                int tokenPosition = FindTokenPosition(inputTokens, Mistral3ImageProcessor.ImgTokenId, searchFrom);
                if (tokenPosition < 0)
                    continue;

                var expanded = new List<int>(inputTokens.Count + numRows * numCols + numRows);
                for (int i = 0; i < tokenPosition; i++)
                    expanded.Add(inputTokens[i]);

                for (int row = 0; row < numRows; row++)
                {
                    for (int col = 0; col < numCols; col++)
                        expanded.Add(Mistral3ImageProcessor.ImgTokenId);

                    expanded.Add(row == numRows - 1
                        ? Mistral3ImageProcessor.ImgEndTokenId
                        : Mistral3ImageProcessor.ImgBreakTokenId);
                }

                for (int i = tokenPosition + 1; i < inputTokens.Count; i++)
                    expanded.Add(inputTokens[i]);

                _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                    cached,
                    tokenPosition,
                    tokenPosition,
                    tokenPosition + numRows * numCols + numRows));

                inputTokens = expanded;
                searchFrom = tokenPosition + numRows * numCols + numRows;
            }

            return inputTokens;
        }

        private CachedEmbedding GetOrCreateGemma4VisionEmbedding(
            Gemma4Model model,
            Gemma4ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, imageWidth, imageHeight) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, imageWidth, imageHeight);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        private CachedEmbedding GetOrCreateGemma4AudioEmbedding(Gemma4Model model, string audioPath)
        {
            return GetOrCreateCachedEmbedding(_audioCache, audioPath, fullPath =>
            {
                float[] samples = Gemma4AudioPreprocessor.DecodeAudioFile(fullPath);
                if (samples.Length % 128 != 0)
                {
                    int padded = samples.Length + (128 - samples.Length % 128);
                    Array.Resize(ref samples, padded);
                }

                var (melData, numFrames) = Gemma4AudioPreprocessor.ComputeMelSpectrogram(samples);
                if (melData == null || numFrames == 0)
                    throw new InvalidOperationException($"Audio file '{fullPath}' did not produce a valid mel spectrogram.");

                Tensor embeddings = model.AudioEncoder.Encode(melData, numFrames);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        private CachedEmbedding GetOrCreateGemma3VisionEmbedding(
            Gemma3Model model,
            Gemma3ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                float[] pixels = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        private CachedEmbedding GetOrCreateQwen35VisionEmbedding(
            Qwen35Model model,
            Qwen35ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, resizedHeight, resizedWidth) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, resizedHeight, resizedWidth);
                return CreateCachedEmbedding(fullPath, embeddings);
            });
        }

        private CachedEmbedding GetOrCreateMistral3VisionEmbedding(
            Mistral3Model model,
            Mistral3ImageProcessor processor,
            string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var (pixels, imageWidth, imageHeight) = processor.ProcessImage(fullPath);
                Tensor embeddings = model.VisionEncoder.Encode(pixels, imageWidth, imageHeight);
                int numRows = imageHeight / model.VisionEncoder.PatchSize / model.VisionEncoder.SpatialMergeSize;
                int numCols = imageWidth / model.VisionEncoder.PatchSize / model.VisionEncoder.SpatialMergeSize;
                return CreateCachedEmbedding(fullPath, embeddings, numRows, numCols);
            });
        }

        private CachedEmbedding GetOrCreateCachedEmbedding(
            Dictionary<string, CachedEmbedding> cache,
            string path,
            Func<string, CachedEmbedding> factory)
        {
            string fullPath = NormalizePath(path);
            GetMediaVersion(fullPath, out long fileSize, out long lastWriteUtcTicks);

            if (cache.TryGetValue(fullPath, out var cached) && cached.Matches(fileSize, lastWriteUtcTicks))
                return cached;

            cached?.Dispose();
            CachedEmbedding fresh = factory(fullPath);
            cache[fullPath] = fresh;
            return fresh;
        }

        private static CachedEmbedding CreateCachedEmbedding(string fullPath, Tensor embeddings, int extra0 = 0, int extra1 = 0)
        {
            GetMediaVersion(fullPath, out long fileSize, out long lastWriteUtcTicks);
            return new CachedEmbedding(
                fullPath,
                fileSize,
                lastWriteUtcTicks,
                embeddings,
                (int)embeddings.Sizes[0],
                extra0,
                extra1);
        }

        private bool QueuePreparedVisionEmbeddings(int reusablePrefixTokenCount)
        {
            if (_preparedVisionEmbeddings.Count == 0)
                return false;

            bool queued = false;

            switch (_model)
            {
                case Gemma4Model g4:
                    foreach (var span in _preparedVisionEmbeddings)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g4.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Gemma3Model g3:
                    foreach (var span in _preparedVisionEmbeddings)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Qwen35Model q35:
                    foreach (var span in _preparedVisionEmbeddings)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        q35.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Mistral3Model m3:
                    foreach (var span in _preparedVisionEmbeddings)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        m3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
            }

            return queued;
        }

        private bool QueuePreparedAudioEmbeddings(int reusablePrefixTokenCount)
        {
            if (_preparedAudioEmbeddings.Count == 0 || _model is not Gemma4Model g4)
                return false;

            bool queued = false;
            foreach (var span in _preparedAudioEmbeddings)
            {
                if (span.EndPosition <= reusablePrefixTokenCount)
                    continue;

                g4.SetAudioEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                queued = true;
            }

            return queued;
        }

        private static int ClampReusablePrefix(int prefixTokenCount, List<PreparedEmbeddingSpan> spans)
        {
            if (prefixTokenCount <= 0 || spans.Count == 0)
                return prefixTokenCount;

            int clamped = prefixTokenCount;
            foreach (var span in spans)
            {
                if (clamped > span.InsertPosition && clamped < span.EndPosition)
                    clamped = Math.Min(clamped, span.InsertPosition);
            }

            return clamped;
        }

        private static int ClampTrimStart(int trimStartTokenCount, List<PreparedEmbeddingSpan> spans)
        {
            if (trimStartTokenCount <= 0 || spans.Count == 0)
                return trimStartTokenCount;

            int clamped = trimStartTokenCount;
            foreach (var span in spans)
            {
                if (clamped > span.PromptTokenStart && clamped < span.PromptTokenEndExclusive)
                    clamped = Math.Max(clamped, span.PromptTokenEndExclusive);
            }

            return clamped;
        }

        private static void TrimPreparedPrompt(List<PreparedEmbeddingSpan> spans, int trimStartTokenCount)
        {
            if (trimStartTokenCount <= 0 || spans.Count == 0)
                return;

            for (int i = spans.Count - 1; i >= 0; i--)
            {
                PreparedEmbeddingSpan span = spans[i];
                if (span.PromptTokenEndExclusive <= trimStartTokenCount)
                {
                    spans.RemoveAt(i);
                    continue;
                }

                span.InsertPosition -= trimStartTokenCount;
                span.PromptTokenStart -= trimStartTokenCount;
                span.PromptTokenEndExclusive -= trimStartTokenCount;
            }
        }

        private void ClearPreparedPromptState()
        {
            _preparedVisionEmbeddings.Clear();
            _preparedAudioEmbeddings.Clear();
        }

        private static void GetMediaVersion(string fullPath, out long fileSize, out long lastWriteUtcTicks)
        {
            if (File.Exists(fullPath))
            {
                var fileInfo = new FileInfo(fullPath);
                fileSize = fileInfo.Length;
                lastWriteUtcTicks = fileInfo.LastWriteTimeUtc.Ticks;
                return;
            }

            fileSize = -1;
            lastWriteUtcTicks = 0;
        }

        private static string NormalizePath(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                return path ?? string.Empty;

            return Path.GetFullPath(path);
        }

        private static Tensor CloneTensor(Tensor source)
        {
            var clone = new Tensor(source.Allocator, source.ElementType, (long[])source.Sizes.Clone());
            Ops.Copy(clone, source);
            return clone;
        }

        private static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var message in history)
            {
                if (message.ImagePaths == null)
                    continue;

                foreach (var path in message.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
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

        public void Dispose()
        {
            ClearPreparedPromptState();

            foreach (var cached in _visionCache.Values)
                cached.Dispose();
            _visionCache.Clear();

            foreach (var cached in _audioCache.Values)
                cached.Dispose();
            _audioCache.Clear();
        }
    }
}
