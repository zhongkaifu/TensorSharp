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
    public class Gemma4VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;

        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _patchSize;
        private readonly int _nMerge;
        private readonly float _ropeTheta;

        private struct ClampParams
        {
            public float InMin, InMax, OutMin, OutMax;
            public bool HasClamp;
        }

        private readonly Dictionary<string, ClampParams> _clampParams = new();
        private readonly Dictionary<long, Rope2DCache> _ropeCache = new();
        private Tensor _onesForNorm;

        private sealed class Rope2DCache
        {
            public required int[] PosX { get; init; }
            public required int[] PosY { get; init; }
            public required float[] CosX { get; init; }
            public required float[] SinX { get; init; }
            public required float[] CosY { get; init; }
            public required float[] SinY { get; init; }
        }

        public int ProjectionDim => _projectionDim;

        public Gemma4VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            var gguf = new GgufFile(mmProjPath);

            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 768);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 3072);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 12);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 16);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-6f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 2560);
            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 16);
            _nMerge = (int)gguf.GetUint32("clip.vision.projector_scale_factor", 0);
            if (_nMerge == 0) _nMerge = 3;
            _ropeTheta = 100f;

            Console.WriteLine($"Vision encoder: hidden={_hiddenSize}, intermediate={_intermediateSize}, " +
                $"heads={_numHeads}, blocks={_blockCount}, projDim={_projectionDim}, " +
                $"patch={_patchSize}, nMerge={_nMerge}");

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
                if (!info.Name.StartsWith("v.") && !info.Name.StartsWith("mm.input_projection"))
                    continue;

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

                if (info.Name.Contains("input_min") || info.Name.Contains("input_max") ||
                    info.Name.Contains("output_min") || info.Name.Contains("output_max"))
                {
                    string linearKey = info.Name.Substring(0, info.Name.LastIndexOf('.'));
                    if (!_clampParams.ContainsKey(linearKey))
                        _clampParams[linearKey] = new ClampParams
                        {
                            InMin = float.MinValue, InMax = float.MaxValue,
                            OutMin = float.MinValue, OutMax = float.MaxValue,
                            HasClamp = false
                        };

                    var cp = _clampParams[linearKey];
                    cp.HasClamp = true;
                    if (info.Name.EndsWith("input_min")) cp.InMin = f32[0];
                    else if (info.Name.EndsWith("input_max")) cp.InMax = f32[0];
                    else if (info.Name.EndsWith("output_min")) cp.OutMin = f32[0];
                    else if (info.Name.EndsWith("output_max")) cp.OutMax = f32[0];
                    _clampParams[linearKey] = cp;
                }
            }
            Console.WriteLine($" done ({count} tensors, {_clampParams.Count} clampable linears)");
        }

        public unsafe Tensor Encode(float[] pixelValues, int imgWidth, int imgHeight)
        {
            int patchesX = imgWidth / _patchSize;
            int patchesY = imgHeight / _patchSize;
            int numPatches = patchesX * patchesY;
            int headDim = _hiddenSize / _numHeads;
            Rope2DCache ropeCache = GetOrCreateRopeCache(patchesX, patchesY, headDim);

            var hidden = PatchEmbed(pixelValues, imgWidth, imgHeight, patchesX, patchesY);
            AddPositionEmbedding2D(hidden, ropeCache, numPatches);

            for (int i = 0; i < _blockCount; i++)
            {
                Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                hidden = EncoderBlock(hidden, i, numPatches, headDim, ropeCache);
            }
            Console.WriteLine(" done");

            var projected = PoolAndProject(hidden, patchesX, patchesY, numPatches);
            hidden.Dispose();

            return projected;
        }

        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgW, int imgH, int patchesX, int patchesY)
        {
            int numPatches = patchesX * patchesY;
            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            float* dst = GetFloatPtr(result);

            var convWeight = _weights["v.patch_embd.weight"];
            float* wPtr = GetFloatPtr(convWeight);

            int C = 3, P = _patchSize;

            for (int py = 0; py < patchesY; py++)
            {
                for (int px = 0; px < patchesX; px++)
                {
                    int patchIdx = py * patchesX + px;
                    float* outPatch = dst + patchIdx * _hiddenSize;

                    for (int f = 0; f < _hiddenSize; f++)
                    {
                        float sum = 0f;
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

        private unsafe void AddPositionEmbedding2D(Tensor hidden, Rope2DCache ropeCache, int numPatches)
        {
            var posEmbd = _weights["v.position_embd.weight"];
            int maxPos = (int)posEmbd.Sizes[1];
            float* posPtr = GetFloatPtr(posEmbd);
            float* xTable = posPtr;
            float* yTable = posPtr + maxPos * _hiddenSize;
            float* dstPtr = GetFloatPtr(hidden);

            for (int p = 0; p < numPatches; p++)
            {
                float* dstRow = dstPtr + p * _hiddenSize;
                float* xRow = xTable + ropeCache.PosX[p] * _hiddenSize;
                float* yRow = yTable + ropeCache.PosY[p] * _hiddenSize;
                for (int d = 0; d < _hiddenSize; d++)
                    dstRow[d] += xRow[d] + yRow[d];
            }
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim,
            Rope2DCache ropeCache)
        {
            string prefix = $"v.blk.{blockIdx}";

            using var attnNormed = RMSNormOp(hidden, $"{prefix}.ln1.weight");
            using var attnOut = VisionSelfAttention(attnNormed, prefix, numPatches, headDim,
                ropeCache);
            using var postAttnNormed = RMSNormOp(attnOut, $"{prefix}.attn_post_norm.weight");

            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            using var ffnNormed = RMSNormOp(postAttnNormed, $"{prefix}.ln2.weight");
            using var mlpOut = VisionMLP(ffnNormed, prefix);
            using var postFfnNormed = RMSNormOp(mlpOut, $"{prefix}.ffn_post_norm.weight");

            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);

            return result;
        }

        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches, int headDim,
            Rope2DCache ropeCache)
        {
            var q = ClippableLinear(input, $"{prefix}.attn_q");
            var k = ClippableLinear(input, $"{prefix}.attn_k");
            var v = ClippableLinear(input, $"{prefix}.attn_v");

            ApplyPerHeadRMSNorm(q, _weights[$"{prefix}.attn_q_norm.weight"], numPatches, headDim);
            ApplyPerHeadRMSNorm(k, _weights[$"{prefix}.attn_k_norm.weight"], numPatches, headDim);
            ApplyUnweightedRMSNorm(v, _numHeads * numPatches, headDim);

            Apply2DRoPE(q, ropeCache, numPatches, headDim);
            Apply2DRoPE(k, ropeCache, numPatches, headDim);

            if (_useNativeAttention)
            {
                using var q4 = q.View(1, numPatches, _numHeads, headDim);
                using var k4 = k.View(1, numPatches, _numHeads, headDim);
                using var v4 = v.View(1, numPatches, _numHeads, headDim);
                using var attn4 = Ops.ScaledDotProductAttention(null, q4, k4, v4, null, 1f);
                using var flat = attn4.View(numPatches, _hiddenSize);
                q.Dispose();
                k.Dispose();
                v.Dispose();
                return ClippableLinear(flat, $"{prefix}.attn_out");
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
            Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
            Ops.Softmax(scores, scores);

            var attnOutput = new Tensor(_allocator, DType.Float32, _numHeads, numPatches, headDim);
            Ops.AddmmBatch(attnOutput, 0, attnOutput, 1f, scores, vHeads);
            scores.Dispose();

            using var transposed = attnOutput.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            using var flatContig = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return ClippableLinear(flatContig, $"{prefix}.attn_out");
        }

        private unsafe void Apply2DRoPE(Tensor data, Rope2DCache ropeCache, int numPatches, int headDim)
        {
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;
            int quarterDim = halfDim / 2;

            for (int p = 0; p < numPatches; p++)
            {
                int ropeBase = p * quarterDim;
                for (int h = 0; h < _numHeads; h++)
                {
                    float* head = ptr + ((long)p * _numHeads + h) * headDim;
                    for (int j = 0; j < quarterDim; j++)
                    {
                        float cos = ropeCache.CosX[ropeBase + j];
                        float sin = ropeCache.SinX[ropeBase + j];
                        float x0 = head[j];
                        float x1 = head[j + quarterDim];
                        head[j] = x0 * cos - x1 * sin;
                        head[j + quarterDim] = x0 * sin + x1 * cos;
                    }

                    for (int j = 0; j < quarterDim; j++)
                    {
                        float cos = ropeCache.CosY[ropeBase + j];
                        float sin = ropeCache.SinY[ropeBase + j];
                        float x0 = head[halfDim + j];
                        float x1 = head[halfDim + j + quarterDim];
                        head[halfDim + j] = x0 * cos - x1 * sin;
                        head[halfDim + j + quarterDim] = x0 * sin + x1 * cos;
                    }
                }
            }
        }

        private void ApplyPerHeadRMSNorm(Tensor data, Tensor normWeight, int numPatches, int headDim)
        {
            int total = _numHeads * numPatches;
            using var reshaped = data.View(total, headDim);
            Ops.RMSNorm(reshaped, reshaped, normWeight, null, _eps);
        }

        private void ApplyUnweightedRMSNorm(Tensor data, int numVectors, int dim)
        {
            if (_onesForNorm == null || (int)_onesForNorm.Sizes[0] != dim)
            {
                _onesForNorm?.Dispose();
                _onesForNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForNorm, 1f);
            }
            using var reshaped = data.View(numVectors, dim);
            Ops.RMSNorm(reshaped, reshaped, _onesForNorm, null, _eps);
        }

        private unsafe Tensor VisionMLP(Tensor input, string prefix)
        {
            var gate = ClippableLinear(input, $"{prefix}.ffn_gate");
            var up = ClippableLinear(input, $"{prefix}.ffn_up");

            // QuickGELU: x * sigmoid(1.702 * x)
            ApplyQuickGELUMul(gate, up);
            up.Dispose();

            var down = ClippableLinear(gate, $"{prefix}.ffn_down");
            gate.Dispose();
            return down;
        }

        private void ApplyQuickGELUMul(Tensor gate, Tensor up)
        {
            // QuickGELU(x) * up = x * sigmoid(1.702 * x) * up
            using var scaled = Ops.Mul(null, gate, 1.702f);
            Ops.SigmoidMul(gate, gate, scaled);
            Ops.Mul(gate, gate, up);
        }

        private unsafe Tensor ClippableLinear(Tensor input, string prefix)
        {
            string weightName = $"{prefix}.weight";
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            bool hasClamp = _clampParams.TryGetValue(prefix, out var cp) && cp.HasClamp;

            if (hasClamp)
                Clamp(src, cp.InMin, cp.InMax);

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Ops.Addmm(result, 0, result, 1f, src, GetOrCreateTransposedWeight(weightName));

            contiguousInput?.Dispose();

            if (hasClamp)
                Clamp(result, cp.OutMin, cp.OutMax);

            return result;
        }

        private unsafe void Clamp(Tensor t, float min, float max)
        {
            float* ptr = GetFloatPtr(t);
            int count = (int)t.ElementCount();
            for (int i = 0; i < count; i++)
            {
                if (ptr[i] < min) ptr[i] = min;
                else if (ptr[i] > max) ptr[i] = max;
            }
        }

        private unsafe Tensor PoolAndProject(Tensor visionOutput, int patchesX, int patchesY, int numPatches)
        {
            int mergedX = patchesX / _nMerge;
            int mergedY = patchesY / _nMerge;
            int mergedPatches = mergedX * mergedY;

            var pooled = new Tensor(_allocator, DType.Float32, mergedPatches, _hiddenSize);
            float* srcPtr = GetFloatPtr(visionOutput);
            float* dstPtr = GetFloatPtr(pooled);

            for (int py = 0; py < mergedY; py++)
            {
                for (int px = 0; px < mergedX; px++)
                {
                    int outIdx = py * mergedX + px;
                    float* outRow = dstPtr + outIdx * _hiddenSize;
                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] = 0;

                    int count = 0;
                    for (int ky = 0; ky < _nMerge; ky++)
                    {
                        for (int kx = 0; kx < _nMerge; kx++)
                        {
                            int srcY = py * _nMerge + ky;
                            int srcX = px * _nMerge + kx;
                            if (srcY < patchesY && srcX < patchesX)
                            {
                                int srcIdx = srcY * patchesX + srcX;
                                float* srcRow = srcPtr + srcIdx * _hiddenSize;
                                for (int d = 0; d < _hiddenSize; d++)
                                    outRow[d] += srcRow[d];
                                count++;
                            }
                        }
                    }

                    float invCount = 1f / count;
                    for (int d = 0; d < _hiddenSize; d++)
                        outRow[d] *= invCount;
                }
            }

            // Scale by sqrt(hiddenSize)
            float scale = MathF.Sqrt(_hiddenSize);
            Ops.Mul(pooled, pooled, scale);

            // Project to text dimension + unweighted RMSNorm
            var projected = LinearProjection(pooled, "mm.input_projection.weight");
            pooled.Dispose();

            ApplyUnweightedRMSNorm(projected, mergedPatches, _projectionDim);

            return projected;
        }

        private Tensor LinearProjection(Tensor input, string weightName)
        {
            var weight = _weights[weightName];
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weight.Sizes[0];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Ops.Addmm(result, 0, result, 1f, input, GetOrCreateTransposedWeight(weightName));
            return result;
        }

        private Tensor RMSNormOp(Tensor input, string weightName)
        {
            var alpha = _weights[weightName];
            return Ops.RMSNorm(null, input, alpha, null, _eps);
        }

        private Tensor CreateIntTensor(int[] data, params long[] sizes)
        {
            var tensor = new Tensor(_allocator, DType.Int32, sizes);
            tensor.SetElementsAsInt(data);
            return tensor;
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

        private Rope2DCache GetOrCreateRopeCache(int patchesX, int patchesY, int headDim)
        {
            long key = ((long)patchesX << 32) | (uint)patchesY;
            if (_ropeCache.TryGetValue(key, out var cache))
                return cache;

            int numPatches = patchesX * patchesY;
            int halfDim = headDim / 2;
            int quarterDim = halfDim / 2;
            int[] posX = new int[numPatches];
            int[] posY = new int[numPatches];
            float[] cosX = new float[numPatches * quarterDim];
            float[] sinX = new float[numPatches * quarterDim];
            float[] cosY = new float[numPatches * quarterDim];
            float[] sinY = new float[numPatches * quarterDim];
            float[] invFreq = new float[quarterDim];

            for (int j = 0; j < quarterDim; j++)
                invFreq[j] = (float)(1.0 / Math.Pow(_ropeTheta, 2.0 * j / halfDim));

            for (int p = 0; p < numPatches; p++)
            {
                int x = p % patchesX;
                int y = p / patchesX;
                posX[p] = x;
                posY[p] = y;

                int baseIdx = p * quarterDim;
                for (int j = 0; j < quarterDim; j++)
                {
                    float angleX = x * invFreq[j];
                    float angleY = y * invFreq[j];
                    cosX[baseIdx + j] = MathF.Cos(angleX);
                    sinX[baseIdx + j] = MathF.Sin(angleX);
                    cosY[baseIdx + j] = MathF.Cos(angleY);
                    sinY[baseIdx + j] = MathF.Sin(angleY);
                }
            }

            cache = new Rope2DCache
            {
                PosX = posX,
                PosY = posY,
                CosX = cosX,
                SinX = sinX,
                CosY = cosY,
                SinY = sinY,
            };
            _ropeCache[key] = cache;
            return cache;
        }

        public void Dispose()
        {
            _onesForNorm?.Dispose();
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

