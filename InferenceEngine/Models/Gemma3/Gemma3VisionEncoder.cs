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

namespace InferenceEngine
{
    public class Gemma3VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly IAllocator _allocator;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _tokensPerImage;

        public int ProjectionDim => _projectionDim;
        public int TokensPerImage => _tokensPerImage;

        public Gemma3VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("clip.vision.image_size", 896);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1152);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 4304);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 27);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 2560);
            _tokensPerImage = 256;

            Console.WriteLine($"Vision encoder: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, intermediate={_intermediateSize}, heads={_numHeads}, " +
                $"blocks={_blockCount}, projDim={_projectionDim}");

            LoadWeights(gguf);
            gguf.Dispose();
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading vision encoder weights...");
            int count = 0;
            foreach (var kv in gguf.Tensors)
            {
                var info = kv.Value;
                byte[] raw = gguf.ReadTensorData(info);

                long numElements = info.NumElements;
                float[] f32 = new float[numElements];

                if (info.Type == GgmlTensorType.F32)
                {
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                }
                else
                {
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);
                }

                long[] ggufShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    ggufShape[i] = (long)info.Shape[i];

                long[] tsShape = new long[ggufShape.Length];
                for (int i = 0; i < ggufShape.Length; i++)
                    tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>
        /// Encode an image into vision embeddings ready to be injected into the text model.
        /// Input: pixelValues float array of shape [channels * imageSize * imageSize] normalized.
        /// Output: Tensor of shape [tokensPerImage, projectionDim].
        /// </summary>
        public unsafe Tensor Encode(float[] pixelValues)
        {
            int numPatches = (_imageSize / _patchSize) * (_imageSize / _patchSize);
            int patchesPerSide = _imageSize / _patchSize;
            int headDim = _hiddenSize / _numHeads;

            bool debug = Environment.GetEnvironmentVariable("DUMP_VISION") == "1";

            var hidden = PatchEmbed(pixelValues, patchesPerSide);
            if (debug) DumpTensor(hidden, "After PatchEmbed", numPatches);

            AddPositionEmbedding(hidden, numPatches);
            if (debug) DumpTensor(hidden, "After PosEmbed", numPatches);

            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, headDim);
                if (debug && (i == 0 || i == _blockCount - 1))
                    DumpTensor(hidden, $"After block {i}", numPatches);
            }
            Console.WriteLine(" done");

            var postNormed = LayerNormOp(hidden, "v.post_ln.weight", "v.post_ln.bias");
            hidden.Dispose();
            if (debug) DumpTensor(postNormed, "After PostLN", numPatches);

            var projected = MultiModalProject(postNormed, patchesPerSide, numPatches);
            postNormed.Dispose();
            if (debug) DumpTensor(projected, "Final projected", (int)projected.Sizes[0]);

            return projected;
        }

        /// <summary>
        /// Conv2D patch embedding: [3, imageSize, imageSize] -> [numPatches, hiddenSize]
        /// Uses the v.patch_embd.weight [patchSize, patchSize, 3, hiddenSize] convolution kernel.
        /// </summary>
        private unsafe Tensor PatchEmbed(float[] pixelValues, int patchesPerSide)
        {
            int numPatches = patchesPerSide * patchesPerSide;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            var convWeight = _weights["v.patch_embd.weight"];
            float* wPtr = GetFloatPtr(convWeight);
            float* biasPtr = _weights.ContainsKey("v.patch_embd.bias")
                ? GetFloatPtr(_weights["v.patch_embd.bias"]) : null;

            int C = 3;
            int P = _patchSize;

            for (int py = 0; py < patchesPerSide; py++)
            {
                for (int px = 0; px < patchesPerSide; px++)
                {
                    int patchIdx = py * patchesPerSide + px;
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
                                    float pixel = pixelValues[c * _imageSize * _imageSize + imgY * _imageSize + imgX];

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

        private void AddPositionEmbedding(Tensor hidden, int numPatches)
        {
            var posEmbd = _weights["v.position_embd.weight"];
            Ops.Add(hidden, hidden, posEmbd);
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var ln1 = LayerNormOp(hidden, $"{prefix}.ln1.weight", $"{prefix}.ln1.bias");

            using var attnOut = VisionSelfAttention(ln1, prefix, numPatches, headDim);

            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();

            using var ln2 = LayerNormOp(attnOut, $"{prefix}.ln2.weight", $"{prefix}.ln2.bias");

            using var mlpOut = VisionMLP(ln2, prefix);

            var result = new Tensor(_allocator, DType.Float32, attnOut.Sizes);
            Ops.Add(result, attnOut, mlpOut);

            return result;
        }

        private Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches, int headDim)
        {
            using var q = LinearForwardWithBias(input, $"{prefix}.attn_q.weight", $"{prefix}.attn_q.bias");
            using var k = LinearForwardWithBias(input, $"{prefix}.attn_k.weight", $"{prefix}.attn_k.bias");
            using var v = LinearForwardWithBias(input, $"{prefix}.attn_v.weight", $"{prefix}.attn_v.bias");

            float scale = 1f / MathF.Sqrt(headDim);

            // Reshape [numPatches, hiddenSize] -> [numHeads, numPatches, headDim]
            using var qReshaped = q.View(numPatches, _numHeads, headDim);
            using var kReshaped = k.View(numPatches, _numHeads, headDim);
            using var vReshaped = v.View(numPatches, _numHeads, headDim);

            using var qT0 = qReshaped.Transpose(0, 1);
            using var kT0 = kReshaped.Transpose(0, 1);
            using var vT0 = vReshaped.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);

            // Batched Q @ K^T -> [numHeads, numPatches, numPatches]
            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);

            Ops.Softmax(scores, scores);

            // Batched softmax @ V -> [numHeads, numPatches, headDim]
            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1.0f, scores, vHeads);
            scores.Dispose();

            // Reshape back: [numHeads, numPatches, headDim] -> [numPatches, hiddenSize]
            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flat = contiguous.View(numPatches, _hiddenSize);
            using var flatContig = Ops.NewContiguous(flat);
            attnOutput.Dispose();

            return LinearForwardWithBias(flatContig, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
        }

        private unsafe Tensor VisionMLP(Tensor input, string prefix)
        {
            using var fc1Out = LinearForwardWithBias(input, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");

            ApplyGELU(fc1Out);

            return LinearForwardWithBias(fc1Out, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
        }

        private unsafe void ApplyGELU(Tensor t)
        {
            float* ptr = GetFloatPtr(t);
            int count = (int)t.ElementCount();
            for (int i = 0; i < count; i++)
            {
                double x = ptr[i];
                double cdf = 0.5 * (1.0 + Math.Tanh(Math.Sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
                ptr[i] = (float)(x * cdf);
            }
        }

        /// <summary>
        /// Multi-modal projector: vision output → text space.
        /// Steps: reshape to 2D grid → average pool → RMSNorm → linear projection.
        /// </summary>
        private unsafe Tensor MultiModalProject(Tensor visionOutput, int patchesPerSide, int numPatches)
        {
            int kernelSize = patchesPerSide / (int)MathF.Sqrt(_tokensPerImage);
            int pooledSide = patchesPerSide / kernelSize;
            int pooledTokens = pooledSide * pooledSide;

            var pooled = new Tensor(_allocator, DType.Float32, pooledTokens, _hiddenSize);
            float* srcPtr = GetFloatPtr(visionOutput);
            float* dstPtr = GetFloatPtr(pooled);

            for (int py = 0; py < pooledSide; py++)
            {
                for (int px = 0; px < pooledSide; px++)
                {
                    int outIdx = py * pooledSide + px;
                    float* outRow = dstPtr + outIdx * _hiddenSize;

                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] = 0;

                    int count = 0;
                    for (int ky = 0; ky < kernelSize; ky++)
                    {
                        for (int kx = 0; kx < kernelSize; kx++)
                        {
                            int srcY = py * kernelSize + ky;
                            int srcX = px * kernelSize + kx;
                            int srcIdx = srcY * patchesPerSide + srcX;
                            float* srcRow = srcPtr + srcIdx * _hiddenSize;
                            for (int d = 0; d < _hiddenSize; d++)
                                outRow[d] += srcRow[d];
                            count++;
                        }
                    }

                    float invCount = 1f / count;
                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] *= invCount;
                }
            }

            using var normed = RMSNormOp(pooled, "mm.soft_emb_norm.weight");
            pooled.Dispose();

            var projected = LinearProjection(normed, "mm.input_projection.weight");
            return projected;
        }

        /// <summary>
        /// Linear projection for mm.input_projection: y = x @ W (no bias, no transpose).
        /// The mm.input_projection.weight is stored as [projDim, hiddenSize] in GGUF.
        /// After loading, TensorSharp shape is [hiddenSize, projDim] (reversed).
        /// In Ollama, this weight is transposed before Mulmat (GGML convention), which
        /// effectively computes y = x @ W where W has TensorSharp shape [hiddenSize, projDim].
        /// </summary>
        private Tensor LinearProjection(Tensor input, string weightName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[1];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Ops.Addmm(result, 0, result, 1.0f, input, weight);
            return result;
        }

        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            using var wT = weight.Transpose();
            Ops.Addmm(result, 0, result, 1.0f, src, wT);

            contiguousInput?.Dispose();

            if (_weights.TryGetValue(biasName, out var bias))
            {
                float* rPtr = GetFloatPtr(result);
                float* bPtr = GetFloatPtr(bias);
                for (int s = 0; s < seqLen; s++)
                {
                    float* row = rPtr + s * outDim;
                    for (int d = 0; d < outDim; d++)
                        row[d] += bPtr[d];
                }
            }

            return result;
        }

        private unsafe Tensor LayerNormOp(Tensor input, string weightName, string biasName)
        {
            int rows = (int)input.Sizes[0];
            int dim = (int)input.Sizes[1];
            var result = new Tensor(_allocator, DType.Float32, rows, dim);

            float* src = GetFloatPtr(input);
            float* dst = GetFloatPtr(result);
            float* w = GetFloatPtr(_weights[weightName]);
            float* b = _weights.ContainsKey(biasName) ? GetFloatPtr(_weights[biasName]) : null;

            for (int r = 0; r < rows; r++)
            {
                float* srcRow = src + r * dim;
                float* dstRow = dst + r * dim;

                float mean = 0;
                for (int i = 0; i < dim; i++)
                    mean += srcRow[i];
                mean /= dim;

                float variance = 0;
                for (int i = 0; i < dim; i++)
                {
                    float diff = srcRow[i] - mean;
                    variance += diff * diff;
                }
                variance /= dim;

                float invStd = 1f / MathF.Sqrt(variance + _eps);
                for (int i = 0; i < dim; i++)
                {
                    float normalized = (srcRow[i] - mean) * invStd;
                    dstRow[i] = w[i] * normalized + (b != null ? b[i] : 0f);
                }
            }

            return result;
        }

        private unsafe Tensor RMSNormOp(Tensor input, string weightName)
        {
            int rows = (int)input.Sizes[0];
            int dim = (int)input.Sizes[1];
            var result = new Tensor(_allocator, DType.Float32, rows, dim);

            float* src = GetFloatPtr(input);
            float* dst = GetFloatPtr(result);
            float* w = GetFloatPtr(_weights[weightName]);

            for (int r = 0; r < rows; r++)
            {
                float* srcRow = src + r * dim;
                float* dstRow = dst + r * dim;

                float sumSq = 0;
                for (int i = 0; i < dim; i++)
                    sumSq += srcRow[i] * srcRow[i];
                float rms = 1f / MathF.Sqrt(sumSq / dim + _eps);
                for (int i = 0; i < dim; i++)
                    dstRow[i] = w[i] * srcRow[i] * rms;
            }

            return result;
        }

        private unsafe void DumpTensor(Tensor t, string label, int numRows)
        {
            float* ptr = GetFloatPtr(t);
            int dim = (int)t.Sizes[1];
            Console.Write($"\n  {label} [{numRows}x{dim}]: row0=[");
            for (int i = 0; i < Math.Min(5, dim); i++)
                Console.Write($"{ptr[i]:F6}{(i < 4 ? ", " : "")}");
            Console.Write($"] last_row=[");
            float* lastRow = ptr + (numRows - 1) * dim;
            for (int i = 0; i < Math.Min(5, dim); i++)
                Console.Write($"{lastRow[i]:F6}{(i < 4 ? ", " : "")}");
            float norm0 = 0, normLast = 0;
            for (int i = 0; i < dim; i++) { norm0 += ptr[i] * ptr[i]; normLast += lastRow[i] * lastRow[i]; }
            Console.WriteLine($"] norm0={MathF.Sqrt(norm0):F4} normLast={MathF.Sqrt(normLast):F4}");
        }

        private static unsafe float* GetFloatPtr(Tensor t)
        {
            if (t.Storage is TensorSharp.GGML.GgmlStorage gs)
                return (float*)gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return (float*)cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        public void Dispose()
        {
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
        }
    }
}
