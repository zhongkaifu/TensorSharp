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
using TensorSharp.Core;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        // Qwen3.8-Flash-Next ships the same qwen3vl_merger vision tower as
        // Qwen3.5-VL - identical tensor names, spatial merge, projector MLP and
        // (in this checkpoint) no active deepstack layers - so the proven Qwen3.5
        // encoder runs it as is.
        public Qwen35VisionEncoder VisionEncoder { get; private set; }

        // The GDN recurrence, the PLE conv history and the n-gram history cannot be
        // rewound to an earlier position, so a cached prefix is only reusable when
        // the new prompt EXTENDS it exactly - same contract as Qwen3.5. The base
        // default of true would let a reuse plan truncate mid-conversation and
        // silently continue from unrewindable state.
        public override bool SupportsKVCacheTruncation => false;

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
            VisionEncoder.SetHostModel(this);
        }

        private readonly List<(Tensor Embeddings, int StartPosition)> _visionEmbeddingsList = new();

        public void SetVisionEmbeddings(Tensor visionEmbeddings, int startPosition)
        {
            _visionEmbeddingsList.Add((visionEmbeddings, startPosition));
        }

        /// <summary>
        /// Replace the text embeddings of the image-pad placeholder tokens with the
        /// projected vision embeddings. The placeholder token IDS stay in the token
        /// array, which is what the PLE hash wants: the reference hashes input_ids,
        /// where image positions hold the placeholder (ple.image_token_id), and this
        /// checkpoint's placeholder IS that token.
        /// </summary>
        private unsafe void InjectVisionEmbeddings(Tensor textEmbeddings, int seqLen)
        {
            if (_visionEmbeddingsList.Count == 0)
                return;

            float* textPtr = GetFloatPtr(textEmbeddings);
            int dim = Config.HiddenSize;
            foreach (var (visionEmbeddings, startPos) in _visionEmbeddingsList)
            {
                if (visionEmbeddings == null || startPos < 0)
                    continue;

                int numVisionTokens = (int)visionEmbeddings.Sizes[0];
                int projDim = (int)visionEmbeddings.Sizes[1];

                if (projDim != dim || startPos + numVisionTokens > seqLen)
                {
                    Console.WriteLine($"Warning: qwen4exp vision span [{startPos}, +{numVisionTokens}) x {projDim} " +
                                      $"does not fit [T={seqLen}, dim={dim}]; skipping injection.");
                    visionEmbeddings.Dispose();
                    continue;
                }

                float* visPtr = GetFloatPtr(visionEmbeddings);
                long bytes = (long)numVisionTokens * dim * sizeof(float);
                Buffer.MemoryCopy(visPtr, textPtr + (long)startPos * dim, bytes, bytes);
                visionEmbeddings.Dispose();
            }

            _visionEmbeddingsList.Clear();
            InvalidateTensorDeviceCache(textEmbeddings);
        }

        // The per-token (T,H,W) IMRoPE position table for the upcoming forward,
        // flat [3 * seqLen], pushed by the multimodal injector just before each
        // prefill slice that overlaps an image. Null for text-only forwards, and
        // for decode steps - which continue on scalar cache positions, exactly as
        // llama.cpp's mtmd path does.
        private int[] _pendingMRoPEPositions;
        private int[] _mropeSections;
        private int[] _mropePosPinned;      // pinned copy the span kernel reads
        private int[] _mropeSectionsPinned;

        public void SetMRoPEPositions(int[] flatThw)
        {
            _pendingMRoPEPositions = flatThw;
        }

        // How far the rotary position stream lags the KV cache index. An HxW image
        // occupies H*W cache rows but only max(H,W) positions (IMRoPE compaction),
        // so after every image the text positions fall behind the cache. llama.cpp's
        // mtmd advances n_past by the compacted span the same way. Text-only
        // forwards and decode subtract this gap from their scalar positions so the
        // stream stays continuous across turns.
        private int _mropeCacheGap;

        private void UpdateMropeGap(int startPos, int seqLen)
        {
            if (_pendingMRoPEPositions == null || _pendingMRoPEPositions.Length < 3 * seqLen || seqLen <= 0)
                return;
            // The last prompt token is text, so its T component is the scalar
            // position stream; the next token continues at that + 1.
            int lastT = _pendingMRoPEPositions[3 * (seqLen - 1)];
            // Video time coordinates can also lead the cache index. Preserve the
            // signed offset so subsequent scalar text continues at lastT + 1.
            _mropeCacheGap = checked((int)((long)startPos + seqLen - 1 - lastT));
        }

        private bool EnsureMropeSections()
        {
            if (_mropeSectionsPinned != null) return true;
            if (_mropeSections == null)
            {
                _mropeSections = _gguf.GetInt32Array($"{Config.Architecture}.rope.dimension_sections");
                if (_mropeSections == null || _mropeSections.Length < 4)
                    return false;
            }
            var pinned = GC.AllocateArray<int>(4, pinned: true);
            for (int i = 0; i < 4; i++) pinned[i] = _mropeSections[i];
            _mropeSectionsPinned = pinned;
            return true;
        }
    }
}
