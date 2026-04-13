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
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Pixtral-style vision encoder for Mistral 3.
    /// Architecture:
    /// - Conv2D patch embedding
    /// - RMSNorm on patch embeddings (encoder_norm)
    /// - 2D RoPE positional embeddings (computed on-the-fly)
    /// - Transformer blocks with RMSNorm, SiLU-gated MLP
    /// - Patch merger with spatial merge
    /// - Multi-modal projector: RMSNorm → PatchMerger → Linear → GELU → Linear
    /// </summary>
    public class Mistral3VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, QuantizedWeight> _quantWeights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _numHeads;
        private readonly int _headDim;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly float _visionRopeBase;
        private readonly int _spatialMergeSize;

        // Multi-modal projector config
        private readonly float _textEps;

        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;
        public int ImageSize => _imageSize;

        public Mistral3VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("vision.image_size",
                          (uint)gguf.GetUint32("clip.vision.image_size", 1540));
            _patchSize = (int)gguf.GetUint32("vision.patch_size",
                          (uint)gguf.GetUint32("clip.vision.patch_size", 14));
            _hiddenSize = (int)gguf.GetUint32("vision.embedding_length",
                           (uint)gguf.GetUint32("clip.vision.embedding_length", 1024));
            _numHeads = (int)gguf.GetUint32("vision.attention.head_count",
                         (uint)gguf.GetUint32("clip.vision.attention.head_count", 16));
            _headDim = (int)gguf.GetUint32("vision.attention.key_length",
                        (uint)(_hiddenSize / _numHeads));
            _blockCount = (int)gguf.GetUint32("vision.block_count",
                           (uint)gguf.GetUint32("clip.vision.block_count", 24));
            _eps = gguf.GetFloat32("vision.attention.layer_norm_epsilon",
                   gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f));
            _visionRopeBase = gguf.GetFloat32("vision.rope.freq_base", 10000.0f);
            _spatialMergeSize = (int)gguf.GetUint32("spatial_merge_size", 2);
            _textEps = gguf.GetFloat32("text_config.rms_norm_eps", 1e-5f);

            Console.WriteLine($"Mistral3 Vision: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, heads={_numHeads}, headDim={_headDim}, " +
                $"blocks={_blockCount}, ropeBase={_visionRopeBase}, mergeSize={_spatialMergeSize}");

            LoadWeights(gguf);
            gguf.Dispose();
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading Mistral3 vision encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                long numElements = info.NumElements;

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                if (info.Type == GgmlTensorType.F32 || info.Shape.Length < 2)
                {
                    float[] f32 = new float[numElements];
                    byte[] raw = gguf.ReadTensorData(info);
                    if (info.Type == GgmlTensorType.F32)
                        Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                    else
                        NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    tensor.SetElementsAsFloat(f32);
                    _weights[info.Name] = tensor;
                }
                else
                {
                    byte[] raw = gguf.ReadTensorData(info);
                    float[] f32 = new float[numElements];
                    if (info.Type == GgmlTensorType.F32)
                        Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                    else
                        NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    tensor.SetElementsAsFloat(f32);
                    _weights[info.Name] = tensor;
                }
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>
        /// Encode an image into vision embeddings ready for the text model.
        /// Input: normalized pixel data, image dimensions.
        /// Output: Tensor of shape [numOutputTokens, textHiddenSize].
        /// </summary>
        public unsafe Tensor Encode(float[] pixelValues, int imageWidth, int imageHeight)
        {
            int numPatchesW = imageWidth / _patchSize;
            int numPatchesH = imageHeight / _patchSize;
            int numPatches = numPatchesW * numPatchesH;

            // Patch embedding via Conv2D
            var hidden = PatchEmbed(pixelValues, imageWidth, imageHeight, numPatchesW, numPatchesH);

            // Encoder norm
            using var normed = RMSNormOp(hidden, "v.encoder_norm.weight");
            hidden.Dispose();
            hidden = Ops.NewContiguous(normed);

            // 2D RoPE positional embeddings
            var (cos, sin) = Compute2DRoPE(numPatchesW, numPatchesH);

            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, cos, sin);
            }
            Console.WriteLine(" done");

            cos.Dispose();
            sin.Dispose();

            // Multi-modal projector
            var projected = MultiModalProject(hidden, numPatchesW, numPatchesH);
            hidden.Dispose();

            return projected;
        }

        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgW, int imgH,
            int patchesW, int patchesH)
        {
            int numPatches = patchesW * patchesH;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            var convWeight = _weights["v.patch_conv.weight"];
            float* wPtr = GetFloatPtr(convWeight);
            float* biasPtr = _weights.ContainsKey("v.patch_conv.bias")
                ? GetFloatPtr(_weights["v.patch_conv.bias"]) : null;

            int C = 3;
            int P = _patchSize;

            for (int py = 0; py < patchesH; py++)
            {
                for (int px = 0; px < patchesW; px++)
                {
                    int patchIdx = py * patchesW + px;
                    float* outPatch = dst + patchIdx * _hiddenSize;

                    for (int f = 0; f < _hiddenSize; f++)
                    {
                        float sum = biasPtr != null ? biasPtr[f] : 0f;

                        for (int c = 0; c < C; c++)
                        {
                            for (int ky = 0; ky < P; ky++)
                            {
                                for (int kx = 0; kx < P; kx++)
                                {
                                    int imgY = py * P + ky;
                                    int imgX = px * P + kx;
                                    float pixel = pixelValues[c * imgH * imgW + imgY * imgW + imgX];
                                    int wIdx = f * C * P * P + c * P * P + ky * P + kx;
                                    sum += pixel * wPtr[wIdx];
                                }
                            }
                        }
                        outPatch[f] = sum;
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Compute 2D RoPE embeddings for the vision transformer.
        /// Returns (cos, sin) tensors of shape [headDim, 1, numPatches].
        /// </summary>
        private (Tensor cos, Tensor sin) Compute2DRoPE(int patchesW, int patchesH)
        {
            int maxPatchesPerSide = _imageSize / _patchSize;
            int numPatches = patchesW * patchesH;
            int frequencies = _headDim / 2;

            float[] freqsHeight = new float[frequencies / 2 * maxPatchesPerSide];
            float[] freqsWidth = new float[frequencies / 2 * maxPatchesPerSide];

            for (int i = 0; i < frequencies; i++)
            {
                for (int j = 0; j < maxPatchesPerSide; j++)
                {
                    float frequency = (float)(j / Math.Pow(_visionRopeBase, (double)i * 2 / _headDim));
                    if (i % 2 == 0)
                        freqsHeight[i / 2 * maxPatchesPerSide + j] = frequency;
                    else
                        freqsWidth[i / 2 * maxPatchesPerSide + j] = frequency;
                }
            }

            // Build per-position inverse frequencies
            float[] invFreqs = new float[frequencies * numPatches];
            for (int h = 0; h < patchesH; h++)
            {
                for (int w = 0; w < patchesW; w++)
                {
                    int patchIdx = h * patchesW + w;
                    for (int f = 0; f < frequencies / 2; f++)
                    {
                        invFreqs[f * numPatches + patchIdx] = freqsHeight[f * maxPatchesPerSide + h];
                        invFreqs[(f + frequencies / 2) * numPatches + patchIdx] = freqsWidth[f * maxPatchesPerSide + w];
                    }
                }
            }

            // Duplicate for cos+sin pairs
            float[] fullFreqs = new float[_headDim * numPatches];
            Array.Copy(invFreqs, 0, fullFreqs, 0, frequencies * numPatches);
            Array.Copy(invFreqs, 0, fullFreqs, frequencies * numPatches, frequencies * numPatches);

            // Compute cos and sin
            float[] cosVals = new float[_headDim * numPatches];
            float[] sinVals = new float[_headDim * numPatches];
            for (int i = 0; i < fullFreqs.Length; i++)
            {
                cosVals[i] = MathF.Cos(fullFreqs[i]);
                sinVals[i] = MathF.Sin(fullFreqs[i]);
            }

            // Reshape to [headDim, 1, numPatches]
            var cosTensor = new Tensor(_allocator, DType.Float32, numPatches, 1, _headDim);
            cosTensor.SetElementsAsFloat(cosVals);
            var sinTensor = new Tensor(_allocator, DType.Float32, numPatches, 1, _headDim);
            sinTensor.SetElementsAsFloat(sinVals);

            return (cosTensor, sinTensor);
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches,
            Tensor cos, Tensor sin)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var normed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");
            using var attnOut = VisionSelfAttention(normed, prefix, numPatches, cos, sin);

            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();

            using var normed2 = RMSNormOp(attnOut, $"{prefix}.ffn_norm.weight");
            using var mlpOut = VisionMLP(normed2, prefix);

            var result = new Tensor(_allocator, DType.Float32, attnOut.Sizes);
            Ops.Add(result, attnOut, mlpOut);

            return result;
        }

        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches,
            Tensor cos, Tensor sin)
        {
            using var q = LinearForward(input, $"{prefix}.attn_q.weight");
            using var k = LinearForward(input, $"{prefix}.attn_k.weight");
            using var v = LinearForward(input, $"{prefix}.attn_v.weight");

            // Reshape to [numPatches, numHeads, headDim]
            using var qR = q.View(numPatches, _numHeads, _headDim);
            using var kR = k.View(numPatches, _numHeads, _headDim);
            using var vR = v.View(numPatches, _numHeads, _headDim);

            // Apply 2D RoPE
            var qRoped = ApplyVisionRoPE(qR, cos, sin, numPatches);
            var kRoped = ApplyVisionRoPE(kR, cos, sin, numPatches);

            float scale = 1f / MathF.Sqrt(_headDim);

            if (_useNativeAttention)
            {
                using var q4 = qRoped.View(1, numPatches, _numHeads, _headDim);
                using var k4 = kRoped.View(1, numPatches, _numHeads, _headDim);
                using var v4 = vR.View(1, numPatches, _numHeads, _headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);
                qRoped.Dispose();
                kRoped.Dispose();
                using var flat = attn4.View(numPatches, _hiddenSize);
                return LinearForward(flat, $"{prefix}.attn_output.weight");
            }

            // Manual attention path
            using var qT0 = qRoped.Transpose(0, 1);
            using var kT0 = kRoped.Transpose(0, 1);
            using var vT0 = vR.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);
            qRoped.Dispose();
            kRoped.Dispose();

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, _headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1.0f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return LinearForward(flatContig, $"{prefix}.attn_output.weight");
        }

        /// <summary>
        /// Apply rotary position embeddings (2D RoPE) for vision.
        /// Uses rotate_half style: [-x1, x0] * sin + [x0, x1] * cos
        /// </summary>
        private unsafe Tensor ApplyVisionRoPE(Tensor input, Tensor cos, Tensor sin, int numPatches)
        {
            // input: [numPatches, numHeads, headDim]
            var result = new Tensor(_allocator, DType.Float32, input.Sizes);
            float* inPtr = GetFloatPtr(input);
            float* outPtr = GetFloatPtr(result);
            float* cosPtr = GetFloatPtr(cos);
            float* sinPtr = GetFloatPtr(sin);

            int halfDim = _headDim / 2;

            for (int p = 0; p < numPatches; p++)
            {
                for (int h = 0; h < _numHeads; h++)
                {
                    float* inHead = inPtr + (long)p * _numHeads * _headDim + h * _headDim;
                    float* outHead = outPtr + (long)p * _numHeads * _headDim + h * _headDim;

                    for (int d = 0; d < halfDim; d++)
                    {
                        float x0 = inHead[d];
                        float x1 = inHead[d + halfDim];
                        float c = cosPtr[p * _headDim + d];
                        float s = sinPtr[p * _headDim + d];

                        // rotate_half: cos*x - sin*rotate_half(x)
                        outHead[d] = x0 * c - x1 * s;
                        outHead[d + halfDim] = x1 * c + x0 * s;
                    }
                }
            }

            return result;
        }

        private Tensor VisionMLP(Tensor input, string prefix)
        {
            using var gate = LinearForward(input, $"{prefix}.ffn_gate.weight");
            using var up = LinearForward(input, $"{prefix}.ffn_up.weight");
            Ops.SiLUMul(gate, gate, up);
            return LinearForward(gate, $"{prefix}.ffn_down.weight");
        }

        /// <summary>
        /// Multi-modal projector: vision → text space.
        /// Steps: RMSNorm → PatchMerger → Linear1 → GELU → Linear2
        /// </summary>
        private unsafe Tensor MultiModalProject(Tensor visionOutput, int patchesW, int patchesH)
        {
            int numPatches = patchesW * patchesH;

            // RMSNorm
            using var normed = RMSNormOp(visionOutput, "mm.norm.weight");

            // Patch merger: merge spatialMergeSize x spatialMergeSize patches
            int mergedW = patchesW / _spatialMergeSize;
            int mergedH = patchesH / _spatialMergeSize;
            int mergedPatches = mergedW * mergedH;
            int mergeInputDim = _hiddenSize * _spatialMergeSize * _spatialMergeSize;

            var mergeInput = new Tensor(_allocator, DType.Float32, mergedPatches, mergeInputDim);
            float* srcPtr = GetFloatPtr(normed);
            float* dstPtr = GetFloatPtr(mergeInput);

            for (int my = 0; my < mergedH; my++)
            {
                for (int mx = 0; mx < mergedW; mx++)
                {
                    int outIdx = my * mergedW + mx;
                    float* outRow = dstPtr + (long)outIdx * mergeInputDim;
                    int fillOffset = 0;

                    for (int sy = 0; sy < _spatialMergeSize; sy++)
                    {
                        for (int sx = 0; sx < _spatialMergeSize; sx++)
                        {
                            int srcY = my * _spatialMergeSize + sy;
                            int srcX = mx * _spatialMergeSize + sx;
                            int srcIdx = srcY * patchesW + srcX;
                            float* srcRow = srcPtr + (long)srcIdx * _hiddenSize;

                            Buffer.MemoryCopy(srcRow, outRow + fillOffset,
                                _hiddenSize * sizeof(float), _hiddenSize * sizeof(float));
                            fillOffset += _hiddenSize;
                        }
                    }
                }
            }

            // Patch merger linear
            using var merged = LinearForward(mergeInput, "mm.patch_merger.merging_layer.weight");
            mergeInput.Dispose();

            // Linear1 → GELU → Linear2
            using var proj1 = LinearForward(merged, "mm.linear_1.weight");
            Ops.GELU(proj1, proj1);
            var proj2 = LinearForward(proj1, "mm.linear_2.weight");

            Console.WriteLine($"Vision projector: {numPatches} patches → {mergedPatches} merged tokens " +
                $"({(int)proj2.Sizes[0]}x{(int)proj2.Sizes[1]})");

            return proj2;
        }

        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            if (!_weights.ContainsKey(weightName))
                return Ops.NewContiguous(input);
            return Ops.RMSNorm(null, input, _weights[weightName], null, _eps);
        }

        private Tensor LinearForward(Tensor input, string weightName)
        {
            if (!_weights.ContainsKey(weightName))
                return null;

            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;
            Ops.Addmm(result, 0, result, 1.0f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();
            return result;
        }

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        private static unsafe float* GetFloatPtr(Tensor t)
        {
            if (t.Storage is GgmlStorage gs)
                return (float*)gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return (float*)cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        public void Dispose()
        {
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            foreach (var qw in _quantWeights.Values)
                qw.Dispose();
            _quantWeights.Clear();
        }
    }
}
