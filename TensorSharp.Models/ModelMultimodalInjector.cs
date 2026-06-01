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

        // Per-request buckets. "" is the default bucket used by direct
        // single-threaded callers (for example InteractiveSession);
        // engine-path callers pass a unique requestId so concurrent requests
        // don't clobber each other's prepared embeddings. Mutations are guarded
        // by _bucketLock because the engine's per-seq Forward (driven by the
        // worker thread inside BatchExecutor) can race the request-thread's
        // ProcessPromptTokens for a different request.
        private readonly object _bucketLock = new();
        private readonly Dictionary<string, List<PreparedEmbeddingSpan>> _visionByRequest = new();
        private readonly Dictionary<string, List<PreparedEmbeddingSpan>> _audioByRequest = new();

        // Per-request flat [T0,H0,W0, T1,H1,W1, ...] position table for
        // interleaved MRoPE-using models (Qwen3.5). Populated by Process*History
        // when an image is in the prompt, then sliced and pushed to the model
        // alongside vision-embedding queueing so prefill RoPE can apply the
        // right per-axis positions to the right rotary dims. Null/missing entry
        // means the request is text-only and standard scalar RoPE is fine.
        private readonly Dictionary<string, int[]> _mropePositionsByRequest = new();

        // "Live view" into the bucket for the request currently being processed
        // by ProcessPromptTokens. The model-specific Process*History helpers
        // append to these. Reset to the default bucket between requests.
        private List<PreparedEmbeddingSpan> _preparedVisionEmbeddings;
        private List<PreparedEmbeddingSpan> _preparedAudioEmbeddings;
        // RequestId currently being processed. Set at ProcessPromptTokens entry
        // so the model-specific Process*History helpers (which don't take a
        // requestId arg) can attach per-request state like the MRoPE position
        // table to the right bucket. Single-threaded under the chat pipeline's
        // GpuComputeLock so no races.
        private string _currentRequestId;

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
            _preparedVisionEmbeddings = GetOrCreateBucket(_visionByRequest, "");
            _preparedAudioEmbeddings = GetOrCreateBucket(_audioByRequest, "");
        }

        private static string NormalizeRequestId(string requestId) => requestId ?? "";

        private List<PreparedEmbeddingSpan> GetOrCreateBucket(
            Dictionary<string, List<PreparedEmbeddingSpan>> buckets, string requestId)
        {
            lock (_bucketLock)
            {
                if (!buckets.TryGetValue(requestId, out var list))
                {
                    list = new List<PreparedEmbeddingSpan>();
                    buckets[requestId] = list;
                }
                return list;
            }
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
                case NemotronModel nem:
                    nem.LoadVisionEncoder(mmProjPath);
                    break;
            }
        }

        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            _preparedVisionEmbeddings = GetOrCreateBucket(_visionByRequest, key);
            _preparedAudioEmbeddings = GetOrCreateBucket(_audioByRequest, key);
            _currentRequestId = key;

            _preparedVisionEmbeddings.Clear();
            _preparedAudioEmbeddings.Clear();

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
            if (_model is NemotronModel nem)
                return ProcessNemotronHistory(nem, history, inputTokens);

            return inputTokens;
        }

        public int ClampReusablePrefix(int reusablePrefixTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            int clamped = ClampReusablePrefix(reusablePrefixTokenCount, visionBucket);
            clamped = ClampReusablePrefix(clamped, audioBucket);
            return clamped;
        }

        public int ClampTrimStart(int trimStartTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            int clamped = ClampTrimStart(trimStartTokenCount, visionBucket);
            clamped = ClampTrimStart(clamped, audioBucket);
            return clamped;
        }

        public void TrimPreparedPrompt(int trimStartTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            TrimPreparedPrompt(GetOrCreateBucket(_visionByRequest, key), trimStartTokenCount);
            TrimPreparedPrompt(GetOrCreateBucket(_audioByRequest, key), trimStartTokenCount);
        }

        public bool QueuePromptEmbeddings(int reusablePrefixTokenCount, string requestId = null)
        {
            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            bool queued = QueuePreparedVisionEmbeddings(visionBucket, reusablePrefixTokenCount);
            queued |= QueuePreparedAudioEmbeddings(audioBucket, reusablePrefixTokenCount);
            return queued;
        }

        public bool QueuePromptEmbeddingsForSlice(int promptStartToken, int tokenCount, string requestId = null)
        {
            if (tokenCount <= 0)
                return false;
            if (promptStartToken < 0)
                throw new ArgumentOutOfRangeException(nameof(promptStartToken));

            long promptEndToken = (long)promptStartToken + tokenCount;
            if (promptEndToken > int.MaxValue)
                promptEndToken = int.MaxValue;

            string key = NormalizeRequestId(requestId);
            var visionBucket = GetOrCreateBucket(_visionByRequest, key);
            var audioBucket = GetOrCreateBucket(_audioByRequest, key);
            bool queued = QueuePreparedVisionEmbeddingsForSlice(visionBucket, promptStartToken, (int)promptEndToken);
            queued |= QueuePreparedAudioEmbeddingsForSlice(audioBucket, promptStartToken, (int)promptEndToken);

            // Also push the matching slice of MRoPE positions onto the model
            // so the upcoming Forward call can apply interleaved per-axis
            // rotations to image-region rotary dims. Text-only requests skip
            // this (TryGet returns null) and the model uses scalar positions.
            int[] mropeSlice = TryGetMRoPEPositionsForSlice(requestId, promptStartToken, tokenCount);
            if (mropeSlice != null && _model is Qwen35Model q35)
            {
                q35.SetMRoPEPositions(mropeSlice);
                queued = true;
            }
            return queued;
        }

        /// <summary>Store the flat (T,H,W) position table for a request. Length
        /// must equal 3 * promptTokenCount. Pass null to clear.</summary>
        internal void SetMRoPEPositions(string requestId, int[] flatThw)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (flatThw == null) _mropePositionsByRequest.Remove(key);
                else _mropePositionsByRequest[key] = flatThw;
            }
        }

        /// <summary>Slice the request's MRoPE position table for the prompt range
        /// [promptStartToken, promptStartToken + tokenCount). Returns null if the
        /// request has no MRoPE positions (text-only request).</summary>
        internal int[] TryGetMRoPEPositionsForSlice(string requestId, int promptStartToken, int tokenCount)
        {
            if (tokenCount <= 0) return null;
            string key = NormalizeRequestId(requestId);
            int[] full;
            lock (_bucketLock)
            {
                if (!_mropePositionsByRequest.TryGetValue(key, out full) || full == null)
                    return null;
            }
            int total = full.Length / 3;
            if (promptStartToken >= total) return null;
            int end = Math.Min(promptStartToken + tokenCount, total);
            int len = end - promptStartToken;
            if (len <= 0) return null;
            int[] slice = new int[len * 3];
            Buffer.BlockCopy(full, promptStartToken * 3 * sizeof(int), slice, 0, len * 3 * sizeof(int));
            return slice;
        }

        public bool HasPendingEmbeddings(string requestId)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (_visionByRequest.TryGetValue(key, out var vision) && vision.Count > 0)
                    return true;
                if (_audioByRequest.TryGetValue(key, out var audio) && audio.Count > 0)
                    return true;
                return false;
            }
        }

        public void ClearPreparedPromptState(string requestId)
        {
            string key = NormalizeRequestId(requestId);
            lock (_bucketLock)
            {
                if (_visionByRequest.TryGetValue(key, out var vision))
                    vision.Clear();
                if (_audioByRequest.TryGetValue(key, out var audio))
                    audio.Clear();
                _mropePositionsByRequest.Remove(key);
                if (key.Length > 0)
                {
                    // Drop the buckets entirely so a finished request doesn't leak
                    // dictionary entries. The default bucket ("") stays around.
                    _visionByRequest.Remove(key);
                    _audioByRequest.Remove(key);
                }
            }
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

            // Build the per-token (T,H,W) MRoPE position table for the entire
            // expanded prompt. vLLM Qwen3.5 (MRotaryEmbedding.get_input_positions)
            // assigns positions like this:
            //  - text tokens get (k, k, k) where k is the running scalar position
            //  - image tokens at merged grid coords (h, w) get
            //      (text_pos, text_pos + h, text_pos + w)
            //    where text_pos is the running scalar at the image's start;
            //    after the image, the running scalar resumes at
            //      max(H, W) + text_pos
            //    so the next text token gets (next_k, next_k, next_k) with no
            //    overlap. For static images T axis stays at text_pos.
            int total = inputTokens.Count;
            int[] thw = new int[3 * total];
            int searchFrom = 0;
            int textPos = 0;
            int writeIdx = 0;
            int imgIdx = 0;
            while (writeIdx < total)
            {
                int imgStart = (imgIdx < imagePaths.Count)
                    ? FindTokenPosition(inputTokens, imagePadId, searchFrom)
                    : -1;
                int textEnd = imgStart >= 0 ? imgStart : total;

                // text run [writeIdx, textEnd) - collapse all three axes
                for (int t = writeIdx; t < textEnd; t++)
                {
                    thw[3 * t + 0] = textPos;
                    thw[3 * t + 1] = textPos;
                    thw[3 * t + 2] = textPos;
                    textPos++;
                }

                if (imgStart < 0) break;

                int mergedH = cachedEmbeddings[imgIdx].Extra0;
                int mergedW = cachedEmbeddings[imgIdx].Extra1;
                int imgTokenCount = tokenCounts[imgIdx];
                if (mergedH * mergedW != imgTokenCount)
                {
                    Console.WriteLine($"[qwen35-mrope] image {imgIdx} grid {mergedH}x{mergedW}={mergedH * mergedW} " +
                                      $"≠ token count {imgTokenCount}; falling back to text-only positions");
                    for (int t = imgStart; t < imgStart + imgTokenCount; t++)
                    {
                        thw[3 * t + 0] = textPos;
                        thw[3 * t + 1] = textPos;
                        thw[3 * t + 2] = textPos;
                        textPos++;
                    }
                }
                else
                {
                    int imgBase = textPos;
                    for (int h = 0; h < mergedH; h++)
                    {
                        for (int w = 0; w < mergedW; w++)
                        {
                            int t = imgStart + h * mergedW + w;
                            thw[3 * t + 0] = imgBase;        // T axis: constant for a single image
                            thw[3 * t + 1] = imgBase + h;    // H axis
                            thw[3 * t + 2] = imgBase + w;    // W axis
                        }
                    }
                    // After the image, the running scalar jumps past the
                    // image's max H/W so subsequent text tokens don't alias
                    // image positions.
                    textPos = imgBase + Math.Max(mergedH, mergedW);
                }

                _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                    cachedEmbeddings[imgIdx],
                    imgStart,
                    imgStart,
                    imgStart + imgTokenCount));

                writeIdx = imgStart + imgTokenCount;
                searchFrom = imgStart + imgTokenCount;
                imgIdx++;
            }

            // Stash on the injector so QueuePromptEmbeddingsForSlice can push
            // the right slice into the model just before each Forward call.
            string key = NormalizeRequestId(_currentRequestId);
            lock (_bucketLock)
            {
                _mropePositionsByRequest[key] = thw;
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

        private List<int> ProcessNemotronHistory(NemotronModel model, List<ChatMessage> history, List<int> inputTokens)
        {
            if (model.VisionEncoder == null)
                return inputTokens;

            int imageTokenId = _model.Tokenizer.LookupToken("<image>");
            int imageStartId = _model.Tokenizer.LookupToken("<img>");
            int imageEndId = _model.Tokenizer.LookupToken("</img>");
            if (imageTokenId < 0) imageTokenId = 18;
            if (imageStartId < 0) imageStartId = 19;
            if (imageEndId < 0) imageEndId = 20;

            // Audio tokens are emitted by the chat template per-message but we can only
            // run the encoder if the mmproj actually shipped Parakeet weights. Detect
            // upfront and skip silently otherwise (the CLI logs a warning).
            int audioTokenId = _model.Tokenizer.LookupToken("<so_embedding>");
            if (audioTokenId < 0) audioTokenId = 27;

            int searchFrom = 0;
            foreach (var message in history)
            {
                if (message.ImagePaths != null && message.ImagePaths.Count > 0)
                {
                    foreach (var imagePath in message.ImagePaths)
                    {
                        if (string.IsNullOrEmpty(imagePath))
                            continue;

                        CachedEmbedding cached = GetOrCreateNemotronVisionEmbedding(model, imagePath);
                        int tokenPosition = FindTokenPosition(inputTokens, imageTokenId, searchFrom);
                        if (tokenPosition < 0)
                            continue;

                        inputTokens = ExpandSingleTokenPlaceholder(
                            inputTokens, tokenPosition, imageStartId, cached.TokenCount, imageEndId);

                        // Insertion point is right after the start sentinel token.
                        _preparedVisionEmbeddings.Add(new PreparedEmbeddingSpan(
                            cached,
                            tokenPosition + 1,
                            tokenPosition,
                            tokenPosition + cached.TokenCount + 2));

                        searchFrom = tokenPosition + cached.TokenCount + 2;
                    }
                }

                // Audio path: the chat template emits a `<so_embedding>` per uploaded
                // audio file so the model "sees" the modality, but real inference is
                // gated on a Parakeet audio mmproj that this distribution does not ship.
                // The test data still gets preprocessed in the CLI for verification.
            }

            return inputTokens;
        }

        private CachedEmbedding GetOrCreateNemotronVisionEmbedding(NemotronModel model, string imagePath)
        {
            return GetOrCreateCachedEmbedding(_visionCache, imagePath, fullPath =>
            {
                var processor = model.ImageProcessor;
                var tiles = processor.ProcessImage(fullPath);
                if (tiles.Count == 0)
                    throw new InvalidOperationException($"Image '{fullPath}' produced zero vision tiles.");

                // Encode each tile and concatenate into a single [totalTokens, hidden] tensor
                // so a single PreparedEmbeddingSpan covers the whole image.
                var tileEmbeddings = new Tensor[tiles.Count];
                int totalTokens = 0;
                int hidden = 0;
                try
                {
                    for (int i = 0; i < tiles.Count; i++)
                    {
                        var tile = tiles[i];
                        tileEmbeddings[i] = model.VisionEncoder.Encode(tile.Pixels, tile.Width, tile.Height);
                        totalTokens += (int)tileEmbeddings[i].Sizes[0];
                        if (i == 0)
                            hidden = (int)tileEmbeddings[i].Sizes[1];
                    }

                    var concatenated = new Tensor(tileEmbeddings[0].Allocator, DType.Float32, totalTokens, hidden);
                    int offset = 0;
                    for (int i = 0; i < tileEmbeddings.Length; i++)
                    {
                        int rows = (int)tileEmbeddings[i].Sizes[0];
                        using var slice = concatenated.Narrow(0, offset, rows);
                        Ops.Copy(slice, tileEmbeddings[i]);
                        offset += rows;
                    }

                    return CreateCachedEmbedding(fullPath, concatenated);
                }
                finally
                {
                    foreach (var t in tileEmbeddings) t?.Dispose();
                }
            });
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
                int mergedH = resizedHeight / processor.PatchSize / processor.MergeSize;
                int mergedW = resizedWidth / processor.PatchSize / processor.MergeSize;
                return CreateCachedEmbedding(fullPath, embeddings, mergedH, mergedW);
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

        private bool QueuePreparedVisionEmbeddings(List<PreparedEmbeddingSpan> bucket, int reusablePrefixTokenCount)
        {
            if (bucket.Count == 0)
                return false;

            bool queued = false;

            switch (_model)
            {
                case Gemma4Model g4:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g4.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Gemma3Model g3:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        g3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Qwen35Model q35:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        q35.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case Mistral3Model m3:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        m3.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
                case NemotronModel nem:
                    foreach (var span in bucket)
                    {
                        if (span.EndPosition <= reusablePrefixTokenCount)
                            continue;

                        nem.SetVisionEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                        queued = true;
                    }
                    break;
            }

            return queued;
        }

        private bool QueuePreparedAudioEmbeddings(List<PreparedEmbeddingSpan> bucket, int reusablePrefixTokenCount)
        {
            if (bucket.Count == 0 || _model is not Gemma4Model g4)
                return false;

            bool queued = false;
            foreach (var span in bucket)
            {
                if (span.EndPosition <= reusablePrefixTokenCount)
                    continue;

                g4.SetAudioEmbeddings(CloneTensor(span.CacheEntry.Embeddings), span.InsertPosition - reusablePrefixTokenCount);
                queued = true;
            }

            return queued;
        }

        private bool QueuePreparedVisionEmbeddingsForSlice(List<PreparedEmbeddingSpan> bucket, int promptStartToken, int promptEndToken)
        {
            if (bucket.Count == 0)
                return false;

            bool queued = false;

            switch (_model)
            {
                case Gemma4Model g4:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        g4.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Gemma3Model g3:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        g3.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Qwen35Model q35:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        q35.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case Mistral3Model m3:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        m3.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
                case NemotronModel nem:
                    foreach (var span in bucket)
                    {
                        if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                                out Tensor embeddings, out int insertPosition))
                            continue;

                        nem.SetVisionEmbeddings(embeddings, insertPosition);
                        queued = true;
                    }
                    break;
            }

            return queued;
        }

        private bool QueuePreparedAudioEmbeddingsForSlice(List<PreparedEmbeddingSpan> bucket, int promptStartToken, int promptEndToken)
        {
            if (bucket.Count == 0 || _model is not Gemma4Model g4)
                return false;

            bool queued = false;
            foreach (var span in bucket)
            {
                if (!TryCloneOverlappingEmbeddingRows(span, promptStartToken, promptEndToken,
                        out Tensor embeddings, out int insertPosition))
                    continue;

                g4.SetAudioEmbeddings(embeddings, insertPosition);
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

        private void ClearAllPreparedPromptState()
        {
            lock (_bucketLock)
            {
                foreach (var bucket in _visionByRequest.Values) bucket.Clear();
                foreach (var bucket in _audioByRequest.Values) bucket.Clear();
            }
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
            var clone = new Tensor(source.Allocator, source.ElementType, source.Sizes);
            Ops.Copy(clone, source);
            return clone;
        }

        private static bool TryCloneOverlappingEmbeddingRows(
            PreparedEmbeddingSpan span,
            int promptStartToken,
            int promptEndToken,
            out Tensor embeddings,
            out int insertPosition)
        {
            embeddings = null;
            insertPosition = 0;

            int overlapStart = Math.Max(promptStartToken, span.InsertPosition);
            int overlapEnd = Math.Min(promptEndToken, span.EndPosition);
            if (overlapStart >= overlapEnd)
                return false;

            int sourceStart = overlapStart - span.InsertPosition;
            int rowCount = overlapEnd - overlapStart;
            insertPosition = overlapStart - promptStartToken;
            embeddings = CloneTensorRows(span.CacheEntry.Embeddings, sourceStart, rowCount);
            return true;
        }

        private static Tensor CloneTensorRows(Tensor source, int startRow, int rowCount)
        {
            if (startRow == 0 && rowCount == source.Sizes[0])
                return CloneTensor(source);

            using var rows = source.Narrow(0, startRow, rowCount);
            var clone = new Tensor(source.Allocator, source.ElementType, rows.Sizes);
            Ops.Copy(clone, rows);
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
            ClearAllPreparedPromptState();

            foreach (var cached in _visionCache.Values)
                cached.Dispose();
            _visionCache.Clear();

            foreach (var cached in _audioCache.Values)
                cached.Dispose();
            _audioCache.Clear();
        }
    }
}
