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
using TensorSharp;

namespace TensorSharp.Models
{
    internal sealed class ModelMultimodalInjector : IMultimodalInjector
    {
        private readonly ModelBase _model;

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
            }
        }

        public List<int> ProcessPromptTokens(List<ChatMessage> history, List<int> inputTokens)
        {
            if (history == null || history.Count == 0 || inputTokens == null || inputTokens.Count == 0)
                return inputTokens;

            if (_model is Gemma4Model g4)
                return ProcessGemma4History(g4, history, inputTokens);
            if (_model is Gemma3Model g3)
                return ProcessGemma3History(g3, history, inputTokens);
            if (_model is Qwen35Model q35)
                return ProcessQwen35History(q35, history, inputTokens);

            return inputTokens;
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
                        var (pixels, imageWidth, imageHeight) = imageProcessor.ProcessImage(imagePath);
                        var embeddings = model.VisionEncoder.Encode(pixels, imageWidth, imageHeight);
                        int tokenCount = (int)embeddings.Sizes[0];
                        int tokenPosition = FindTokenPosition(inputTokens, imageStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, imageStartId, tokenCount, imageEndId);
                            model.SetVisionEmbeddings(embeddings, tokenPosition + 1);
                            searchFrom = tokenPosition + tokenCount + 2;
                        }
                        else
                        {
                            embeddings.Dispose();
                        }
                    }
                }

                if (message.AudioPaths != null && model.AudioEncoder != null && audioStartId >= 0 && audioEndId >= 0)
                {
                    foreach (var audioPath in message.AudioPaths)
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

                        var embeddings = model.AudioEncoder.Encode(melData, numFrames);
                        int tokenCount = (int)embeddings.Sizes[0];
                        int tokenPosition = FindTokenPosition(inputTokens, audioStartId, searchFrom);

                        if (tokenPosition >= 0)
                        {
                            inputTokens = ExpandSingleTokenPlaceholder(inputTokens, tokenPosition, audioStartId, tokenCount, audioEndId);
                            model.SetAudioEmbeddings(embeddings, tokenPosition + 1);
                            searchFrom = tokenPosition + tokenCount + 2;
                        }
                        else
                        {
                            embeddings.Dispose();
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
                float[] pixels = processor.ProcessImage(imagePath);
                var embeddings = model.VisionEncoder.Encode(pixels);
                int tokenStart = FindGemma3ImageInsertPosition(inputTokens, startId, padId, searchFrom);

                if (tokenStart >= 0)
                {
                    model.SetVisionEmbeddings(embeddings, tokenStart);
                    searchFrom = tokenStart + processor.TokensPerImage + 2;
                }
                else
                {
                    embeddings.Dispose();
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
            var tokenCounts = new int[imagePaths.Count];
            for (int i = 0; i < imagePaths.Count; i++)
            {
                var (width, height) = Qwen35ImageProcessor.ReadImageDimensions(imagePaths[i]);
                tokenCounts[i] = processor.ComputeImageTokenCount(height, width);
            }

            inputTokens = ChatTemplate.ExpandImageTokens(inputTokens, imagePadId, tokenCounts);

            int searchFrom = 0;
            for (int i = 0; i < imagePaths.Count; i++)
            {
                var (pixels, resizedHeight, resizedWidth) = processor.ProcessImage(imagePaths[i]);
                var embeddings = model.VisionEncoder.Encode(pixels, resizedHeight, resizedWidth);
                int tokenStart = FindTokenPosition(inputTokens, imagePadId, searchFrom);

                if (tokenStart >= 0)
                {
                    model.SetVisionEmbeddings(embeddings, tokenStart);
                    searchFrom = tokenStart + tokenCounts[i];
                }
                else
                {
                    embeddings.Dispose();
                }
            }

            return inputTokens;
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
    }
}

