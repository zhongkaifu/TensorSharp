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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Vision encoder for the NVIDIA Nemotron 3 Nano Omni model.
    /// Architecture matches the ollama <c>nemotronh.VisionModel</c> reference (RADIO/CLIP-style ViT):
    ///   - Linear "patch embedding" (weights: [hidden, channels*patch*patch], bias optional)
    ///   - Optional position embedding stored as a [hidden, posTokens] grid that is bilinearly
    ///     resized via align-corners=false when the source patch grid differs from the cached
    ///     one. Resized embeddings are cached per (gridW,gridH).
    ///   - Optional class embedding [hidden, numPrefixTokens] prepended to the sequence
    ///   - 32 encoder blocks: LayerNorm → fused-QKV self attention → residual → LayerNorm
    ///     → up linear → GELU → down linear → residual
    ///   - After the encoder strip the class tokens, pixel-shuffle by <c>scaleFactor</c>,
    ///     then RMSNorm → Linear → ReLU-squared → Linear projector that emits embeddings
    ///     in the LM hidden dim.
    /// </summary>
    public sealed class NemotronVisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        // Cooperative GpuComputeLock yielding (see Gemma4VisionEncoder).
        private ModelBase _hostModel;
        public void SetHostModel(ModelBase model) => _hostModel = model;

        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _blockCount;
        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _numChannels;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _scaleFactor;
        private readonly bool _useGelu;

        private readonly int _numClassTokens;
        private readonly int _positionTokens;
        private readonly int _positionSourceSide;
        private readonly bool _hasPositionEmbed;

        private readonly ConcurrentDictionary<(int W, int H), float[]> _resizedPositionEmbeddings = new();
        private readonly NemotronImageProcessor _imageProcessor;

        public int HiddenSize => _hiddenSize;
        public int PatchSize => _patchSize;
        public int ImageSize => _imageSize;
        public int NumChannels => _numChannels;
        public int ProjectionDim => _projectionDim;
        public int ScaleFactor => _scaleFactor;
        public NemotronImageProcessor ImageProcessor => _imageProcessor;

        public NemotronVisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            using var gguf = new GgufFile(mmProjPath);

            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1280);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 5120);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _headDim = _hiddenSize / Math.Max(1, _numHeads);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 32);
            _imageSize = (int)gguf.GetUint32("clip.vision.image_size", 512);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 16);
            _numChannels = (int)gguf.GetUint32("clip.vision.num_channels", 3);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 2688);
            _scaleFactor = (int)gguf.GetUint32("clip.vision.projector.scale_factor", 2);
            _useGelu = gguf.GetBool("clip.use_gelu", true);

            Console.WriteLine($"Nemotron Vision: hidden={_hiddenSize}, intermediate={_intermediateSize}, " +
                $"heads={_numHeads}, headDim={_headDim}, blocks={_blockCount}, " +
                $"image={_imageSize}, patch={_patchSize}, projDim={_projectionDim}, scale={_scaleFactor}, gelu={_useGelu}");

            LoadWeights(gguf);

            if (_weights.TryGetValue("v.class_embd", out var classEmb))
            {
                _numClassTokens = (int)classEmb.Sizes[0];
                Console.WriteLine($"  Class tokens: {_numClassTokens}");
            }

            if (_weights.TryGetValue("v.position_embd.weight", out var posEmb))
            {
                _hasPositionEmbed = true;
                _positionTokens = (int)posEmb.Sizes[posEmb.Sizes.Length - 2];
                _positionSourceSide = (int)Math.Round(Math.Sqrt(_positionTokens));
                Console.WriteLine($"  Position embed: {_positionTokens} tokens ({_positionSourceSide}x{_positionSourceSide})");
            }

            float[] mean = gguf.GetFloatArray("clip.vision.image_mean");
            float[] std = gguf.GetFloatArray("clip.vision.image_std");
            uint maxTiles = gguf.GetUint32("clip.vision.max_tiles", 12);
            uint minPatches = gguf.GetUint32("clip.vision.min_num_patches", 0);
            uint maxPatches = gguf.GetUint32("clip.vision.max_num_patches", 0);
            bool useThumb = gguf.GetBool("clip.vision.use_thumbnail", true);
            _imageProcessor = new NemotronImageProcessor(
                imageSize: _imageSize,
                patchSize: _patchSize,
                numChannels: _numChannels,
                maxTiles: (int)maxTiles,
                minNumPatches: (int)minPatches,
                maxNumPatches: (int)maxPatches,
                useThumbnail: useThumb,
                projectorScaleFactor: _scaleFactor,
                imageMean: mean,
                imageStd: std);
            if (_imageProcessor.MaxTiles != (int)maxTiles)
            {
                Console.WriteLine($"  Image tiling cap: maxTiles={_imageProcessor.MaxTiles} " +
                    $"(model={maxTiles}; set TS_NEMOTRON_IMAGE_MAX_TILES={maxTiles} for full-resolution tiling)");
            }
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading Nemotron vision encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                if (!info.Name.StartsWith("v.") && !info.Name.StartsWith("mm."))
                    continue;

                long numElements = info.NumElements;
                float[] f32 = new float[numElements];
                byte[] raw = gguf.ReadTensorData(info);
                if (info.Type == GgmlTensorType.F32)
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                long[] tsShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    tsShape[i] = (long)info.Shape[info.Shape.Length - 1 - i];

                // Conv-style patch embedding weights ship as 4D [P_x, P_y, C, hidden]
                // (GGUF order); after reversing to TS order we get [hidden, C, P_y, P_x].
                // Flatten the trailing C*P_y*P_x dims so the embedding can run as a
                // standard 2D linear (input: [numPatches, C*P*P], output: [numPatches, hidden]).
                if (info.Name == "v.patch_embd.weight" && tsShape.Length == 4)
                {
                    long hidden = tsShape[0];
                    long inDim = tsShape[1] * tsShape[2] * tsShape[3];
                    tsShape = new long[] { hidden, inDim };
                }
                // Position embeddings ship as 3D [posTokens, hidden, 1] in GGUF (or
                // [1, posTokens, hidden] in our reversed order with a singleton). Flatten
                // to [posTokens, hidden] for cleaner downstream code.
                else if (info.Name == "v.position_embd.weight" && tsShape.Length == 3)
                {
                    long hidden = tsShape[0] == 1 ? tsShape[2] : tsShape[2];
                    long tokens = tsShape[0] == 1 ? tsShape[1] : tsShape[0];
                    if (tsShape[0] == 1) tsShape = new long[] { tokens, hidden };
                    else if (tsShape[2] == 1) tsShape = new long[] { tokens, hidden };
                }

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>
        /// Encode one preprocessed tile into projected embeddings ready for the LM.
        /// </summary>
        /// <param name="pixelValues">Channel-first [C, H, W] float32 pixel data.</param>
        /// <param name="imageWidth">Tile width in pixels (multiple of patch size).</param>
        /// <param name="imageHeight">Tile height in pixels (multiple of patch size).</param>
        /// <returns>Tensor [numTokens, projectionDim] where numTokens =
        /// (W/patch/scale) * (H/patch/scale).</returns>
        public unsafe Tensor Encode(float[] pixelValues, int imageWidth, int imageHeight)
        {
            int patchesX = imageWidth / _patchSize;
            int patchesY = imageHeight / _patchSize;
            int numPatches = patchesX * patchesY;
            int patchDim = _numChannels * _patchSize * _patchSize;

            // 1. Pack patches into [numPatches, patchDim] then linearly project.
            float[] packed = NemotronImageProcessor.PackPatchesCHW(
                pixelValues, imageWidth, imageHeight, _numChannels, _patchSize);

            using var packedTensor = new Tensor(_allocator, DType.Float32, numPatches, patchDim);
            packedTensor.SetElementsAsFloat(packed);

            Tensor hidden = LinearForward(packedTensor, "v.patch_embd.weight",
                "v.patch_embd.bias", _hiddenSize);

            // 2. Add (resized) position embeddings.
            if (_hasPositionEmbed)
                AddPositionEmbeddings(hidden, patchesX, patchesY, numPatches);

            // 3. Concatenate class tokens at the front.
            int totalTokens = numPatches;
            if (_numClassTokens > 0)
            {
                hidden = ConcatClassTokens(hidden, numPatches);
                totalTokens = numPatches + _numClassTokens;
            }

            // 4. Run encoder blocks.
            for (int i = 0; i < _blockCount; i++)
            {
                hidden = EncoderBlock(hidden, i, totalTokens);
                // Yield GpuComputeLock between encoder blocks (see
                // Gemma4VisionEncoder).
                _hostModel?.YieldGpuComputeLock();
            }

            // 5. Drop the class tokens.
            if (_numClassTokens > 0)
            {
                using var sliced = hidden.Narrow(0, _numClassTokens, numPatches);
                Tensor patchOnly = Ops.NewContiguous(sliced);
                hidden.Dispose();
                hidden = patchOnly;
            }

            // 6. Pixel-shuffle, RMSNorm, MLP projector.
            Tensor projected = MultiModalProject(hidden, patchesX, patchesY);
            hidden.Dispose();
            return projected;
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int totalTokens)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var ln1 = LayerNormOp(hidden, $"{prefix}.ln1");
            using var attnOut = SelfAttention(ln1, prefix, totalTokens);
            Ops.Add(hidden, hidden, attnOut);

            using var ln2 = LayerNormOp(hidden, $"{prefix}.ln2");
            using var mlpOut = MlpForward(ln2, prefix);
            Ops.Add(hidden, hidden, mlpOut);

            return hidden;
        }

        private unsafe Tensor SelfAttention(Tensor input, string prefix, int totalTokens)
        {
            string fusedName = $"{prefix}.attn_qkv.weight";
            string fusedBias = $"{prefix}.attn_qkv.bias";

            // Fused projection: out = X @ W_qkv^T + bias_qkv, layout [Q | K | V].
            int qkvDim = _hiddenSize * 3;
            using var qkv = LinearForward(input, fusedName, fusedBias, qkvDim);

            using var qPart = qkv.Narrow(1, 0, _hiddenSize);
            using var kPart = qkv.Narrow(1, _hiddenSize, _hiddenSize);
            using var vPart = qkv.Narrow(1, 2 * _hiddenSize, _hiddenSize);

            using var qContig = Ops.NewContiguous(qPart);
            using var kContig = Ops.NewContiguous(kPart);
            using var vContig = Ops.NewContiguous(vPart);

            using var qH = qContig.View(totalTokens, _numHeads, _headDim);
            using var kH = kContig.View(totalTokens, _numHeads, _headDim);
            using var vH = vContig.View(totalTokens, _numHeads, _headDim);

            float scale = 1.0f / MathF.Sqrt(_headDim);

            if (_useNativeAttention)
            {
                using var q4 = qH.View(1, totalTokens, _numHeads, _headDim);
                using var k4 = kH.View(1, totalTokens, _numHeads, _headDim);
                using var v4 = vH.View(1, totalTokens, _numHeads, _headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);
                using var flat = attn4.View(totalTokens, _hiddenSize);
                return LinearForward(flat, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias", _hiddenSize);
            }

            using var qHeadsT = qH.Transpose(0, 1);
            using var kHeadsT = kH.Transpose(0, 1);
            using var vHeadsT = vH.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qHeadsT);
            using var kHeads = Ops.NewContiguous(kHeadsT);
            using var vHeads = Ops.NewContiguous(vHeadsT);

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, totalTokens, totalTokens);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOut = new Tensor(_allocator, DType.Float32, _numHeads, totalTokens, _headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOut.Transpose(0, 1);
            using var contig = Ops.NewContiguous(transposed);
            using var flatA = contig.View(totalTokens, _hiddenSize);
            attnOut.Dispose();
            return LinearForward(flatA, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias", _hiddenSize);
        }

        private Tensor MlpForward(Tensor input, string prefix)
        {
            using var up = LinearForward(input, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias", _intermediateSize);
            if (_useGelu)
                Ops.GELU(up, up);
            else
                Ops.Relu(up, up);
            return LinearForward(up, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias", _hiddenSize);
        }

        private Tensor MultiModalProject(Tensor visionOutputs, int patchesX, int patchesY)
        {
            int scale = Math.Max(1, _scaleFactor);

            // Pixel shuffle: [hidden, W*H] -> [hidden*scale*scale, (W/scale)*(H/scale)]
            // We follow the v2 packing order from the ollama reference: reshape into
            // (channels*scale, W/scale, H, 1), then permute to swap rows/cols within
            // the scale block, finally reshape into the merged tokens.
            using var merged = PixelShuffleVisionOutputs(visionOutputs, patchesX, patchesY, scale);

            // RMSNorm then 2-layer MLP with ReLU-squared activation.
            using var normed = RMSNormOp(merged, "mm.model.mlp.0.weight");
            using var proj1 = LinearForward(normed, "mm.model.mlp.1.weight", null, GetWeightOutDim("mm.model.mlp.1.weight"));
            ReluSquaredInPlace(proj1);
            return LinearForward(proj1, "mm.model.mlp.3.weight", null, GetWeightOutDim("mm.model.mlp.3.weight"));
        }

        /// <summary>
        /// Pixel-shuffle that matches the ollama reference v2 packing order. The
        /// vision encoder output is viewed as [numPatches=patchesY*patchesX, hidden].
        /// We produce [outTokens, hidden*scale*scale] where each output row gathers
        /// a scale x scale block of source patches in (sy,sx) order — matching the
        /// reference projector's `pixelShuffleVisionOutputs` packing exactly.
        ///
        /// Implemented entirely with Tensor ops (View/Permute/Contiguous) so the
        /// data stays on the active backend (Metal/CUDA/CPU) and avoids a costly
        /// host roundtrip on GGML allocators.
        /// </summary>
        private Tensor PixelShuffleVisionOutputs(Tensor visionOutputs, int patchesX, int patchesY, int scale)
        {
            int hidden = (int)visionOutputs.Sizes[1];
            int outX = patchesX / scale;
            int outY = patchesY / scale;
            int outTokens = outX * outY;
            int outHidden = hidden * scale * scale;

            // Input layout (TS row-major): [patchesY, patchesX, hidden].
            using var view5d = visionOutputs.View(outY, scale, outX, scale, hidden);

            // Reorder so spatial blocks (sy, sx) become inner-most:
            //   from [outY, sy, outX, sx, hidden] to [outY, outX, sy, sx, hidden]
            using var permuted = view5d.Permute(0, 2, 1, 3, 4);
            using var contig = Ops.NewContiguous(permuted);

            return contig.View(outTokens, outHidden);
        }

        /// <summary>
        /// Add positional embeddings, lazily resizing the [posTokens, hidden] table to the
        /// current patch grid via PyTorch align-corners=false bilinear interpolation when
        /// the source side does not match.
        /// </summary>
        private void AddPositionEmbeddings(Tensor hidden, int patchesW, int patchesH, int numPatches)
        {
            if (patchesW == _positionSourceSide && patchesH == _positionSourceSide
                && _positionTokens >= numPatches)
            {
                var posTensor = _weights["v.position_embd.weight"];
                if (_positionTokens == numPatches)
                {
                    Ops.Add(hidden, hidden, posTensor);
                }
                else
                {
                    using var sliced = posTensor.Narrow(0, 0, numPatches);
                    using var contig = Ops.NewContiguous(sliced);
                    Ops.Add(hidden, hidden, contig);
                }
                return;
            }

            float[] resized = _resizedPositionEmbeddings.GetOrAdd((patchesW, patchesH), key =>
            {
                var posTensor = _weights["v.position_embd.weight"];
                float[] values = posTensor.GetElementsAsFloat((int)posTensor.ElementCount());
                return ResizePositionEmbedding(values, _hiddenSize,
                    _positionSourceSide, _positionSourceSide, key.W, key.H);
            });

            using var posDevice = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            posDevice.SetElementsAsFloat(resized);
            Ops.Add(hidden, hidden, posDevice);
        }

        /// <summary>
        /// PyTorch align-corners=false bilinear resize for position embeddings, exactly
        /// matching the reference implementation. Source layout: [hidden, srcW*srcH] with
        /// hidden being the inner stride. Output layout: [hidden, dstW*dstH] in the same
        /// hidden-inner-stride layout, but after the transpose Ollama does we end up with
        /// [dstW*dstH, hidden] which is what callers add to <c>hidden</c>.
        /// </summary>
        private static float[] ResizePositionEmbedding(float[] values, int hidden,
            int sourceWidth, int sourceHeight, int targetWidth, int targetHeight)
        {
            // Source values come in as [hidden, sourceTokens] flattened with hidden as
            // the outer dim per token (i.e. token-major when read as [tokens, hidden]).
            // The ollama reference call site permutes the embedding to (hidden, src, src)
            // before interpolation. Since our cache returns the data in token-major
            // [tokens, hidden] order (each token of length hidden contiguous), interpolate
            // across the spatial grid keeping hidden as inner.
            float[] outArr = new float[hidden * targetWidth * targetHeight];
            double scaleX = (double)sourceWidth / targetWidth;
            double scaleY = (double)sourceHeight / targetHeight;

            Parallel.For(0, targetHeight, oy =>
            {
                double srcY = scaleY * (oy + 0.5) - 0.5;
                int y0 = (int)Math.Floor(srcY);
                int y1 = Math.Min(y0 + 1, sourceHeight - 1);
                float wy = (float)(srcY - y0);
                y0 = Math.Max(y0, 0);

                for (int ox = 0; ox < targetWidth; ox++)
                {
                    double srcX = scaleX * (ox + 0.5) - 0.5;
                    int x0 = (int)Math.Floor(srcX);
                    int x1 = Math.Min(x0 + 1, sourceWidth - 1);
                    float wx = (float)(srcX - x0);
                    x0 = Math.Max(x0, 0);

                    int t00 = (y0 * sourceWidth + x0) * hidden;
                    int t01 = (y0 * sourceWidth + x1) * hidden;
                    int t10 = (y1 * sourceWidth + x0) * hidden;
                    int t11 = (y1 * sourceWidth + x1) * hidden;
                    int dst = (oy * targetWidth + ox) * hidden;
                    for (int h = 0; h < hidden; h++)
                    {
                        float v00 = values[t00 + h];
                        float v01 = values[t01 + h];
                        float v10 = values[t10 + h];
                        float v11 = values[t11 + h];
                        float top = v00 + (v01 - v00) * wx;
                        float bot = v10 + (v11 - v10) * wx;
                        outArr[dst + h] = top + (bot - top) * wy;
                    }
                }
            });

            return outArr;
        }

        private Tensor ConcatClassTokens(Tensor hidden, int numPatches)
        {
            int totalTokens = numPatches + _numClassTokens;
            var result = new Tensor(_allocator, DType.Float32, totalTokens, _hiddenSize);
            var classEmb = _weights["v.class_embd"];

            using (var clsTarget = result.Narrow(0, 0, _numClassTokens))
                Ops.Copy(clsTarget, classEmb);
            using (var patchTarget = result.Narrow(0, _numClassTokens, numPatches))
                Ops.Copy(patchTarget, hidden);

            hidden.Dispose();
            return result;
        }

        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            if (!_weights.TryGetValue(weightName, out var w))
                return Ops.NewContiguous(input);
            return Ops.RMSNorm(null, input, w, null, 1e-5f);
        }

        private Tensor LayerNormOp(Tensor input, string weightPrefix)
        {
            string gName = weightPrefix + ".weight";
            string bName = weightPrefix + ".bias";
            if (!_weights.TryGetValue(gName, out var gamma))
                return Ops.NewContiguous(input);
            _weights.TryGetValue(bName, out var beta);
            return Ops.LayerNorm(null, input, gamma, beta, _eps);
        }

        private Tensor LinearForward(Tensor input, string weightName, string biasName, int outDim)
        {
            if (!_weights.TryGetValue(weightName, out _))
                return null;

            int seqLen = (int)input.Sizes[0];
            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            Ops.Addmm(result, 0, result, 1.0f, src, GetOrCreateTransposedWeight(weightName));

            if (biasName != null && _weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            contiguousInput?.Dispose();
            return result;
        }

        private int GetWeightOutDim(string weightName)
        {
            if (_weights.TryGetValue(weightName, out var w))
                return (int)w.Sizes[0];
            return -1;
        }

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;
            using var weightT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ReluSquaredInPlace(Tensor t) =>
            TensorComputePrimitives.ReluSquaredInPlace(t);

        public void Dispose()
        {
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
        }
    }
}
