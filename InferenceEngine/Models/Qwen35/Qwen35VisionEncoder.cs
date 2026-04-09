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

namespace InferenceEngine
{
    public class Qwen35VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly Dictionary<long, Tensor> _positionEmbeddingCache = new();
        private readonly Dictionary<long, RopeCache> _ropeCache = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;

        private readonly int _imageSize;
        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _spatialMergeSize;
        private readonly int _gridPerSide;
        private readonly float _ropeTheta;

        private sealed class RopeCache
        {
            public required float[] CosTable { get; init; }
            public required float[] SinTable { get; init; }
        }

        public int ProjectionDim => _projectionDim;
        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;

        public Qwen35VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            var gguf = new GgufFile(mmProjPath);

            _imageSize = (int)gguf.GetUint32("clip.vision.image_size", 768);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 16);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1152);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 4304);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 27);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 4096);
            _spatialMergeSize = (int)gguf.GetUint32("clip.vision.spatial_merge_size", 2);
            _ropeTheta = gguf.GetFloat32("clip.vision.rope.freq_base", 10000f);
            _gridPerSide = _imageSize / _patchSize;

            Console.WriteLine($"Qwen3.5 Vision encoder: imageSize={_imageSize}, patchSize={_patchSize}, " +
                $"hidden={_hiddenSize}, intermediate={_intermediateSize}, heads={_numHeads}, " +
                $"blocks={_blockCount}, projDim={_projectionDim}, mergeSize={_spatialMergeSize}, " +
                $"gridPerSide={_gridPerSide}, ropeTheta={_ropeTheta}");

            LoadWeights(gguf);
            CombineTemporalPatchWeights();
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
                    Buffer.BlockCopy(raw, 0, f32, 0, raw.Length);
                else
                    NativeDequant.DequantizeToFloat32((int)info.Type, raw, 0, f32, 0, numElements);

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

        private unsafe void CombineTemporalPatchWeights()
        {
            if (!_weights.ContainsKey("v.patch_embd.weight") ||
                !_weights.ContainsKey("v.patch_embd.weight.1"))
                return;

            var w0 = _weights["v.patch_embd.weight"];
            var w1 = _weights["v.patch_embd.weight.1"];
            var combined = new Tensor(_allocator, DType.Float32, w0.Sizes);
            Ops.Add(combined, w0, w1);
            _weights["v.patch_embd.combined"] = combined;
        }

        /// <summary>
        /// Encode an image into vision embeddings ready to be injected into the text model.
        /// Input: pixelValues float array in channel-first [C, H, W], resized dimensions.
        /// Output: Tensor of shape [numMergedTokens, projectionDim].
        /// </summary>
        public unsafe Tensor Encode(float[] pixelValues, int resizedH, int resizedW)
        {
            int gridH = resizedH / _patchSize;
            int gridW = resizedW / _patchSize;
            int numPatches = gridH * gridW;
            int headDim = _hiddenSize / _numHeads;
            int halfDim = headDim / 2;

            bool debug = Environment.GetEnvironmentVariable("DUMP_VISION") == "1";

            // 1. Patch embedding (Conv2D, raster order)
            var hidden = PatchEmbed(pixelValues, resizedH, resizedW, gridH, gridW);
            if (debug) DumpTensor(hidden, "After PatchEmbed (raster)", numPatches);

            // 2. Position embedding (bilinear interpolation, raster order)
            AddPositionEmbedding(hidden, gridH, gridW);
            if (debug) DumpTensor(hidden, "After PosEmbed (raster)", numPatches);

            // 3. Reorder from raster to block order
            var blockOrdered = ReorderToBlockOrder(hidden, gridH, gridW);
            hidden.Dispose();
            if (debug) DumpTensor(blockOrdered, "After BlockReorder", numPatches);

            // 4. Build block-order grid coordinate arrays for RoPE
            RopeCache ropeCache = GetOrCreateRopeCache(gridH, gridW, numPatches, halfDim);

            // 6. Encoder blocks
            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                blockOrdered = EncoderBlock(blockOrdered, i, numPatches, headDim, halfDim,
                    ropeCache.CosTable, ropeCache.SinTable);
                if (debug && (i == 0 || i == _blockCount - 1))
                    DumpTensor(blockOrdered, $"After block {i}", numPatches);
            }
            Console.WriteLine(" done");

            // 7. Post-LayerNorm
            var postNormed = LayerNormOp(blockOrdered, "v.post_ln.weight", "v.post_ln.bias");
            blockOrdered.Dispose();
            if (debug) DumpTensor(postNormed, "After PostLN", numPatches);

            // 8. Spatial merge + projection
            int mergedPatches = numPatches / (_spatialMergeSize * _spatialMergeSize);
            int mergedDim = _hiddenSize * _spatialMergeSize * _spatialMergeSize;

            using var merged = postNormed.View(mergedPatches, mergedDim);
            var mergedContig = Ops.NewContiguous(merged);
            postNormed.Dispose();

            using var fc1 = LinearForwardWithBias(mergedContig, "mm.0.weight", "mm.0.bias");
            mergedContig.Dispose();
            Ops.GELU(fc1, fc1);

            var projected = LinearForwardWithBias(fc1, "mm.2.weight", "mm.2.bias");
            if (debug) DumpTensor(projected, "Final projected", mergedPatches);

            return projected;
        }

        /// <summary>
        /// Conv2D patch embedding with combined temporal weights.
        /// Output in raster (row-major) order: [numPatches, hiddenSize].
        /// </summary>
        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgH, int imgW, int gridH, int gridW)
        {
            int numPatches = gridH * gridW;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            string wName = _weights.ContainsKey("v.patch_embd.combined")
                ? "v.patch_embd.combined" : "v.patch_embd.weight";
            var convWeight = _weights[wName];
            float* wPtr = GetFloatPtr(convWeight);
            float* biasPtr = _weights.ContainsKey("v.patch_embd.bias")
                ? GetFloatPtr(_weights["v.patch_embd.bias"]) : null;

            int C = 3;
            int P = _patchSize;

            for (int py = 0; py < gridH; py++)
            {
                for (int px = 0; px < gridW; px++)
                {
                    int patchIdx = py * gridW + px;
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
        /// Add bilinearly-interpolated position embeddings (computed in raster order).
        /// </summary>
        private void AddPositionEmbedding(Tensor hidden, int gridH, int gridW)
        {
            Ops.Add(hidden, hidden, GetOrCreatePositionEmbedding(gridH, gridW));
        }

        /// <summary>
        /// Reorder patches from raster (h, w) order to spatial-merge block order
        /// (2x2 groups iterated in raster order, within each group: mh, mw).
        /// </summary>
        private unsafe Tensor ReorderToBlockOrder(Tensor input, int gridH, int gridW)
        {
            int numPatches = gridH * gridW;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* srcPtr = GetFloatPtr(input);
            float* dstPtr = GetFloatPtr(result);
            int bytes = _hiddenSize * sizeof(float);

            int dstIdx = 0;
            for (int bh = 0; bh < gridH; bh += _spatialMergeSize)
            {
                for (int bw = 0; bw < gridW; bw += _spatialMergeSize)
                {
                    for (int mh = 0; mh < _spatialMergeSize; mh++)
                    {
                        for (int mw = 0; mw < _spatialMergeSize; mw++)
                        {
                            int srcRow = (bh + mh) * gridW + (bw + mw);
                            Buffer.MemoryCopy(
                                srcPtr + srcRow * _hiddenSize,
                                dstPtr + dstIdx * _hiddenSize,
                                bytes, bytes);
                            dstIdx++;
                        }
                    }
                }
            }

            return result;
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim,
            int halfDim, float[] cosTable, float[] sinTable)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var ln1 = LayerNormOp(hidden, $"{prefix}.ln1.weight", $"{prefix}.ln1.bias");
            using var attnOut = VisionSelfAttention(ln1, prefix, numPatches, headDim, halfDim,
                cosTable, sinTable);

            Ops.Add(attnOut, attnOut, hidden);
            hidden.Dispose();

            using var ln2 = LayerNormOp(attnOut, $"{prefix}.ln2.weight", $"{prefix}.ln2.bias");
            using var mlpOut = VisionMLP(ln2, prefix);

            var result = new Tensor(_allocator, DType.Float32, attnOut.Sizes);
            Ops.Add(result, attnOut, mlpOut);

            return result;
        }

        /// <summary>
        /// Vision self-attention with fused QKV and RoPE.
        /// </summary>
        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches,
            int headDim, int halfDim, float[] cosTable, float[] sinTable)
        {
            using var qkv = LinearForwardWithBias(input, $"{prefix}.attn_qkv.weight", $"{prefix}.attn_qkv.bias");

            // Split fused QKV: [numPatches, 3*hiddenSize] -> Q, K, V
            using var qView = qkv.Narrow(1, 0, _hiddenSize);
            using var kView = qkv.Narrow(1, _hiddenSize, _hiddenSize);
            using var vView = qkv.Narrow(1, 2 * _hiddenSize, _hiddenSize);
            var q = Ops.NewContiguous(qView);
            var k = Ops.NewContiguous(kView);
            var v = Ops.NewContiguous(vView);

            // Apply RoPE to Q and K
            ApplyVisionRoPE(q, numPatches, headDim, halfDim, cosTable, sinTable);
            ApplyVisionRoPE(k, numPatches, headDim, halfDim, cosTable, sinTable);

            float scale = 1f / MathF.Sqrt(headDim);

            if (_useNativeAttention)
            {
                using var q4 = q.View(1, numPatches, _numHeads, headDim);
                using var k4 = k.View(1, numPatches, _numHeads, headDim);
                using var v4 = v.View(1, numPatches, _numHeads, headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, scale);
                using var flat = attn4.View(numPatches, _hiddenSize);
                q.Dispose();
                k.Dispose();
                v.Dispose();
                return LinearForwardWithBias(flat, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
            }

            using var qR = q.View(numPatches, _numHeads, headDim);
            using var kR = k.View(numPatches, _numHeads, headDim);
            using var vR = v.View(numPatches, _numHeads, headDim);

            using var qT0 = qR.Transpose(0, 1);
            using var kT0 = kR.Transpose(0, 1);
            using var vT0 = vR.Transpose(0, 1);
            using var qHeads = Ops.NewContiguous(qT0);
            using var kHeads = Ops.NewContiguous(kT0);
            using var vHeads = Ops.NewContiguous(vT0);
            q.Dispose();
            k.Dispose();
            v.Dispose();

            using var kT = kHeads.Transpose(1, 2);
            var scores = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, numPatches);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1.0f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return LinearForwardWithBias(flatContig, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
        }

        /// <summary>
        /// Apply NeoX-style RoPE with 2D spatial position encoding.
        /// data: [numPatches, numHeads * headDim], cosTable/sinTable: [numPatches * halfDim].
        /// </summary>
        private unsafe void ApplyVisionRoPE(Tensor data, int numPatches, int headDim, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            float* ptr = GetFloatPtr(data);

            for (int p = 0; p < numPatches; p++)
            {
                int cosBase = p * halfDim;
                for (int h = 0; h < _numHeads; h++)
                {
                    float* head = ptr + (p * _numHeads + h) * headDim;
                    for (int d = 0; d < halfDim; d++)
                    {
                        float x0 = head[d];
                        float x1 = head[d + halfDim];
                        float cos = cosTable[cosBase + d];
                        float sin = sinTable[cosBase + d];
                        head[d] = x0 * cos - x1 * sin;
                        head[d + halfDim] = x0 * sin + x1 * cos;
                    }
                }
            }
        }

        private Tensor VisionMLP(Tensor input, string prefix)
        {
            using var fc1Out = LinearForwardWithBias(input, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
            Ops.GELU(fc1Out, fc1Out);
            return LinearForwardWithBias(fc1Out, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
        }

        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            Ops.Addmm(result, 0, result, 1.0f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();

            if (_weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        private Tensor LayerNormOp(Tensor input, string weightName, string biasName)
        {
            _weights.TryGetValue(biasName, out var bias);
            return Ops.LayerNorm(null, input, _weights[weightName], bias, _eps);
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

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            using var weightViewT = _weights[weightName].Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
        }

        private unsafe Tensor GetOrCreatePositionEmbedding(int gridH, int gridW)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (_positionEmbeddingCache.TryGetValue(key, out var cached))
                return cached;

            int numPatches = gridH * gridW;
            cached = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* posPtr = GetFloatPtr(_weights["v.position_embd.weight"]);
            float* dstPtr = GetFloatPtr(cached);

            float stepH = gridH > 1 ? (float)(_gridPerSide - 1) / (gridH - 1) : 0f;
            float stepW = gridW > 1 ? (float)(_gridPerSide - 1) / (gridW - 1) : 0f;

            for (int h = 0; h < gridH; h++)
            {
                for (int w = 0; w < gridW; w++)
                {
                    float y = h * stepH;
                    float x = w * stepW;

                    int fy = (int)y;
                    int fx = (int)x;
                    int cy = Math.Min(fy + 1, _gridPerSide - 1);
                    int cx = Math.Min(fx + 1, _gridPerSide - 1);
                    float dy = y - fy;
                    float dx = x - fx;

                    float w00 = (1 - dy) * (1 - dx);
                    float w01 = (1 - dy) * dx;
                    float w10 = dy * (1 - dx);
                    float w11 = dy * dx;

                    int idx00 = fy * _gridPerSide + fx;
                    int idx01 = fy * _gridPerSide + cx;
                    int idx10 = cy * _gridPerSide + fx;
                    int idx11 = cy * _gridPerSide + cx;

                    int patchIdx = h * gridW + w;
                    float* dstRow = dstPtr + patchIdx * _hiddenSize;
                    float* p00 = posPtr + idx00 * _hiddenSize;
                    float* p01 = posPtr + idx01 * _hiddenSize;
                    float* p10 = posPtr + idx10 * _hiddenSize;
                    float* p11 = posPtr + idx11 * _hiddenSize;

                    for (int d = 0; d < _hiddenSize; d++)
                        dstRow[d] = w00 * p00[d] + w01 * p01[d] + w10 * p10[d] + w11 * p11[d];
                }
            }

            _positionEmbeddingCache[key] = cached;
            return cached;
        }

        private RopeCache GetOrCreateRopeCache(int gridH, int gridW, int numPatches, int halfDim)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (_ropeCache.TryGetValue(key, out var cache))
                return cache;

            int[] gridY = new int[numPatches];
            int[] gridX = new int[numPatches];
            int idx = 0;
            for (int bh = 0; bh < gridH; bh += _spatialMergeSize)
            {
                for (int bw = 0; bw < gridW; bw += _spatialMergeSize)
                {
                    for (int mh = 0; mh < _spatialMergeSize; mh++)
                    {
                        for (int mw = 0; mw < _spatialMergeSize; mw++)
                        {
                            gridY[idx] = bh + mh;
                            gridX[idx] = bw + mw;
                            idx++;
                        }
                    }
                }
            }

            int numBands = halfDim / 2;
            float[] cosTable = new float[numPatches * halfDim];
            float[] sinTable = new float[numPatches * halfDim];
            float[] invFreqs = new float[numBands];
            for (int j = 0; j < numBands; j++)
                invFreqs[j] = 1f / MathF.Pow(_ropeTheta, (2f * j) / halfDim);

            for (int p = 0; p < numPatches; p++)
            {
                int baseIdx = p * halfDim;
                for (int j = 0; j < numBands; j++)
                {
                    float angleY = gridY[p] * invFreqs[j];
                    float angleX = gridX[p] * invFreqs[j];

                    cosTable[baseIdx + j * 2] = MathF.Cos(angleY);
                    sinTable[baseIdx + j * 2] = MathF.Sin(angleY);
                    cosTable[baseIdx + j * 2 + 1] = MathF.Cos(angleX);
                    sinTable[baseIdx + j * 2 + 1] = MathF.Sin(angleX);
                }
            }

            cache = new RopeCache { CosTable = cosTable, SinTable = sinTable };
            _ropeCache[key] = cache;
            return cache;
        }

        public void Dispose()
        {
            foreach (var w in _positionEmbeddingCache.Values)
                w.Dispose();
            _positionEmbeddingCache.Clear();
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            _ropeCache.Clear();
        }
    }
}
