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
using System.Numerics;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public class Gemma4VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        // Optional ModelBase reference for cooperative GpuComputeLock yielding.
        // When set, Encode() releases the lock between encoder blocks so the
        // engine worker can run one inference step per block — keeping
        // concurrent decode requests responsive during long image encodes.
        private ModelBase _hostModel;

        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _patchSize;
        private readonly int _nMerge;
        private readonly float _ropeTheta;

        // "gemma4uv" (Gemma 4 unified vision embedder, e.g. Gemma-4-12B) uses a
        // completely different, block-less vision path: a single large-patch
        // conv (patch_size * n_merge) followed by three pytorch LayerNorms,
        // learned 2D positional embeddings, an unweighted RMSNorm and a linear
        // projection. The classic "gemma4v" path (SigLIP transformer with
        // v.blk.N.* blocks, e.g. Gemma-4-E4B) keeps the original encode flow.
        private readonly bool _isUnified;
        private readonly string _projectorType;
        // image_mean / image_std read from the mmproj. The unified embedder is
        // trained on raw [0,1] pixels (mean=0, std=1); the SigLIP path keeps the
        // legacy [-1,1] preprocessing for backward compatibility.
        private readonly float[] _imageMean;
        private readonly float[] _imageStd;

        // pytorch nn.LayerNorm default epsilon used by the unified embedder.
        private const float UnifiedLayerNormEps = 1e-5f;

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

        /// <summary>True when this mmproj is a Gemma 4 "unified" vision embedder
        /// (projector_type "gemma4uv"), which uses the block-less encode path.</summary>
        public bool IsUnified => _isUnified;

        /// <summary>Per-channel image mean read from the mmproj (or null).</summary>
        public float[] ImageMean => _imageMean;

        /// <summary>Per-channel image std read from the mmproj (or null).</summary>
        public float[] ImageStd => _imageStd;

        /// <summary>Attach the model that owns this encoder so the per-block
        /// loop in <see cref="Encode"/> can yield the GPU compute lock between
        /// blocks. Set once after construction.</summary>
        public void SetHostModel(ModelBase model) => _hostModel = model;

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

            _projectorType = gguf.GetString("clip.vision.projector_type", "gemma4v") ?? "gemma4v";
            _isUnified = string.Equals(_projectorType, "gemma4uv", StringComparison.Ordinal);
            _imageMean = gguf.GetFloatArray("clip.vision.image_mean");
            _imageStd = gguf.GetFloatArray("clip.vision.image_std");

            Console.WriteLine($"Vision encoder: type={_projectorType}, hidden={_hiddenSize}, " +
                $"intermediate={_intermediateSize}, heads={_numHeads}, blocks={_blockCount}, " +
                $"projDim={_projectionDim}, patch={_patchSize}, nMerge={_nMerge}" +
                (_isUnified ? $", unifiedPatch={_patchSize * _nMerge}" : string.Empty));

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
            if (_isUnified)
                return EncodeUnified(pixelValues, imgWidth, imgHeight);

            int patchesX = imgWidth / _patchSize;
            int patchesY = imgHeight / _patchSize;
            int numPatches = patchesX * patchesY;
            int headDim = _hiddenSize / _numHeads;
            Rope2DCache ropeCache = GetOrCreateRopeCache(patchesX, patchesY, headDim);

            bool vtime = Environment.GetEnvironmentVariable("TS_VTIME") == "1";
            double msPerTick = 1000.0 / System.Diagnostics.Stopwatch.Frequency;
            long t0 = System.Diagnostics.Stopwatch.GetTimestamp();

            var hidden = PatchEmbed(pixelValues, imgWidth, imgHeight, patchesX, patchesY);
            AddPositionEmbedding2D(hidden, ropeCache, numPatches);
            long tPatch = System.Diagnostics.Stopwatch.GetTimestamp();

            for (int i = 0; i < _blockCount; i++)
            {
                if (!vtime) Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                long tb0 = vtime ? System.Diagnostics.Stopwatch.GetTimestamp() : 0;
                hidden = EncoderBlock(hidden, i, numPatches, headDim, ropeCache);
                if (vtime)
                    Console.WriteLine($"  [gemma4v] block {i}: " +
                        $"{(System.Diagnostics.Stopwatch.GetTimestamp() - tb0) * msPerTick:F1}ms");
                // Yield the GPU compute lock between encoder blocks so the
                // engine worker can run an inference step. Without this,
                // a single image encode (16–32 blocks, 100ms–2s+ total)
                // freezes every other in-flight chat/decode request for
                // its entire duration; with this, each yield admits one
                // engine step (≈50–200ms), interleaving encoder progress
                // with inference progress. No-op when the encoder is
                // running outside a GpuComputeLock scope.
                _hostModel?.YieldGpuComputeLock();
            }
            if (!vtime) Console.WriteLine(" done");
            long tBlocks = System.Diagnostics.Stopwatch.GetTimestamp();

            var projected = PoolAndProject(hidden, patchesX, patchesY, numPatches);
            hidden.Dispose();
            long tProj = System.Diagnostics.Stopwatch.GetTimestamp();

            if (vtime)
                Console.WriteLine($"  [gemma4v] patchEmbed+pos={ (tPatch - t0) * msPerTick:F0}ms " +
                    $"blocks={ (tBlocks - tPatch) * msPerTick:F0}ms proj={ (tProj - tBlocks) * msPerTick:F0}ms " +
                    $"({numPatches} patches, {_blockCount} blocks)");

            return projected;
        }

        /// <summary>
        /// Encode path for the Gemma 4 "unified" vision embedder (projector_type
        /// "gemma4uv", used by Gemma-4-12B). Mirrors llama.cpp's
        /// clip_graph_gemma4uv::build():
        ///   im2col(patch = patch_size * n_merge) -> LayerNorm(patch_norm_1)
        ///   -> patch_embd matmul + bias -> LayerNorm(patch_norm_2)
        ///   -> + 2D learned position embeddings -> LayerNorm(patch_norm_3 / pos_norm)
        ///   -> unweighted RMSNorm -> linear projection to the text embedding dim.
        /// There are no transformer blocks (clip.vision.block_count == 0).
        /// </summary>
        private unsafe Tensor EncodeUnified(float[] pixelValues, int imgWidth, int imgHeight)
        {
            // The unified variant folds the n_merge "token merging" directly into
            // a larger conv patch, so the effective patch is patch_size * n_merge
            // (e.g. 16 * 3 = 48) and there is no separate pooling step.
            int P = _patchSize * _nMerge;
            const int C = 3;
            int patchDim = P * P * C;          // e.g. 48 * 48 * 3 = 6912
            int patchesX = imgWidth / P;
            int patchesY = imgHeight / P;
            int numPatches = patchesX * patchesY;

            // 1. im2col: each patch becomes a flat row ordered [c][ky][kx] to match
            //    the converted v.patch_embd.weight layout (same ordering as the
            //    SigLIP PatchEmbed above).
            var cols = new Tensor(_allocator, DType.Float32, numPatches, patchDim);
            float* colsPtr = GetFloatPtr(cols);
            fixed (float* pixSrc = pixelValues)
            {
                long colsPtrL = (long)colsPtr, pixSrcL = (long)pixSrc;
                // Each (c, ky) source span of P pixels is contiguous in kx, so the
                // im2col packing is a parallel set of MemoryCopy row writes.
                Parallel.For(0, patchesY, py =>
                {
                    float* dst = (float*)colsPtrL;
                    float* pix = (float*)pixSrcL;
                    for (int px = 0; px < patchesX; px++)
                    {
                        int patchIdx = py * patchesX + px;
                        float* row = dst + (long)patchIdx * patchDim;
                        for (int c = 0; c < C; c++)
                        {
                            long imgChannelOffset = (long)c * imgHeight * imgWidth;
                            long outChannelOffset = (long)c * P * P;
                            for (int ky = 0; ky < P; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(py * P + ky) * imgWidth + px * P;
                                Buffer.MemoryCopy(pix + srcOffset, row + outChannelOffset + (long)ky * P,
                                    P * sizeof(float), P * sizeof(float));
                            }
                        }
                    }
                });
            }

            // 2. patch_norm_1: pytorch LayerNorm over the patch vector.
            Ops.LayerNorm(cols, cols, _weights["v.patch_norm.1.weight"],
                _weights["v.patch_norm.1.bias"], UnifiedLayerNormEps);

            // 3. patch embedding (linear) + bias -> [numPatches, hiddenSize].
            var hidden = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            Ops.Addmm(hidden, 0, hidden, 1f, cols, GetOrCreateTransposedWeight("v.patch_embd.weight"));
            cols.Dispose();
            AddBiasInPlace(hidden, _weights["v.patch_embd.bias"], numPatches, _hiddenSize);

            // 4. patch_norm_2: LayerNorm over the embedding dim.
            Ops.LayerNorm(hidden, hidden, _weights["v.patch_norm.2.weight"],
                _weights["v.patch_norm.2.bias"], UnifiedLayerNormEps);

            // 5. add learned 2D (x, y) position embeddings.
            AddUnifiedPositionEmbedding(hidden, patchesX, numPatches);

            // 6. patch_norm_3 (pos_norm): LayerNorm over the embedding dim.
            Ops.LayerNorm(hidden, hidden, _weights["v.patch_norm.3.weight"],
                _weights["v.patch_norm.3.bias"], UnifiedLayerNormEps);

            // 7. embedding_pre_projection_norm: unweighted RMSNorm (eps = hparams.eps).
            ApplyUnweightedRMSNorm(hidden, numPatches, _hiddenSize);

            // 8. project to the text embedding dimension.
            var projected = LinearProjection(hidden, "mm.input_projection.weight");
            hidden.Dispose();

            return projected;
        }

        private unsafe void AddBiasInPlace(Tensor t, Tensor bias, int rows, int cols)
        {
            float* p = GetFloatPtr(t);
            float* b = GetFloatPtr(bias);
            int vLen = Vector<float>.Count;
            long pL = (long)p, bL = (long)b;
            Parallel.For(0, rows, r =>
            {
                float* row = (float*)pL + (long)r * cols;
                float* bias2 = (float*)bL;
                int d = 0;
                for (; d <= cols - vLen; d += vLen)
                    TensorComputePrimitives.StoreVector(row + d,
                        TensorComputePrimitives.LoadVector(row + d)
                        + TensorComputePrimitives.LoadVector(bias2 + d));
                for (; d < cols; d++)
                    row[d] += bias2[d];
            });
        }

        private unsafe void AddUnifiedPositionEmbedding(Tensor hidden, int patchesX, int numPatches)
        {
            // v.position_embd.weight is stored as two stacked lookup tables
            // (x then y), each [maxPos, hiddenSize]; pos_x = patch % cols,
            // pos_y = patch / cols.
            var posEmbd = _weights["v.position_embd.weight"];
            int maxPos = (int)posEmbd.Sizes[1];
            float* posPtr = GetFloatPtr(posEmbd);
            float* xTable = posPtr;
            float* yTable = posPtr + (long)maxPos * _hiddenSize;
            float* dstPtr = GetFloatPtr(hidden);
            int hiddenSize = _hiddenSize;
            int vLen = Vector<float>.Count;
            long dstPtrL = (long)dstPtr, xTableL = (long)xTable, yTableL = (long)yTable;

            Parallel.For(0, numPatches, p =>
            {
                int x = p % patchesX;
                int y = p / patchesX;
                float* dstRow = (float*)dstPtrL + (long)p * hiddenSize;
                float* xRow = (float*)xTableL + (long)x * hiddenSize;
                float* yRow = (float*)yTableL + (long)y * hiddenSize;
                int d = 0;
                for (; d <= hiddenSize - vLen; d += vLen)
                {
                    var acc = TensorComputePrimitives.LoadVector(dstRow + d)
                            + TensorComputePrimitives.LoadVector(xRow + d)
                            + TensorComputePrimitives.LoadVector(yRow + d);
                    TensorComputePrimitives.StoreVector(dstRow + d, acc);
                }
                for (; d < hiddenSize; d++)
                    dstRow[d] += xRow[d] + yRow[d];
            });
        }

        // Patch embedding (Conv2D, stride = kernel = patch_size) reformulated as
        // im2col + GEMM, matching llama.cpp's ggml_conv_2d and the already-optimized
        // Qwen35 path. The per-patch im2col packing is parallelised with SIMD row
        // copies; the heavy compute becomes a single matmul that the backend
        // (CPU BLAS or GPU) handles efficiently, replacing the original
        // single-threaded scalar quintuple loop.
        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgW, int imgH, int patchesX, int patchesY)
        {
            int numPatches = patchesX * patchesY;
            int C = 3, P = _patchSize;
            int patchStride = C * P * P;

            // v.patch_embd.weight post-load is [hiddenSize, C, P, P] contiguous in
            // [f][c][ky][kx] order — exactly the im2col row ordering below — so it
            // views directly as a [hiddenSize, patchStride] matmul matrix.
            Tensor weightT = GetOrCreatePatchEmbedTransposed(patchStride);

            var im2col = new Tensor(_allocator, DType.Float32, numPatches, patchStride);
            float* im2colPtr = GetFloatPtr(im2col);

            fixed (float* pixSrc = pixelValues)
            {
                long pixSrcL = (long)pixSrc;
                Parallel.For(0, patchesY, py =>
                {
                    float* pix = (float*)pixSrcL;
                    for (int px = 0; px < patchesX; px++)
                    {
                        int patchIdx = py * patchesX + px;
                        float* outRow = im2colPtr + (long)patchIdx * patchStride;
                        int yBase = py * P;
                        int xBase = px * P;
                        for (int c = 0; c < C; c++)
                        {
                            long imgChannelOffset = (long)c * imgH * imgW;
                            long outChannelOffset = (long)c * P * P;
                            for (int ky = 0; ky < P; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(yBase + ky) * imgW + xBase;
                                long dstOffset = outChannelOffset + (long)ky * P;
                                Buffer.MemoryCopy(pix + srcOffset, outRow + dstOffset,
                                    P * sizeof(float), P * sizeof(float));
                            }
                        }
                    }
                });
            }

            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            Ops.Addmm(result, 0, result, 1.0f, im2col, weightT);
            im2col.Dispose();
            return result;
        }

        // Cache the transposed [patchStride, hiddenSize] patch-embed weight used by
        // the im2col GEMM. v.patch_embd.weight is already contiguous as
        // [hiddenSize, patchStride], so we transpose once.
        private Tensor GetOrCreatePatchEmbedTransposed(int patchStride)
        {
            const string key = "v.patch_embd.weight.2d.T";
            if (_transposedWeights.TryGetValue(key, out var cached))
                return cached;

            var convWeight = _weights["v.patch_embd.weight"];
            int outDim = (int)convWeight.Sizes[0];
            using var weight2D = convWeight.View(outDim, patchStride);
            using var weightViewT = weight2D.Transpose();
            var result = Ops.NewContiguous(weightViewT);
            _transposedWeights[key] = result;
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
            int hiddenSize = _hiddenSize;
            int vLen = Vector<float>.Count;

            long dstPtrL = (long)dstPtr, xTableL = (long)xTable, yTableL = (long)yTable;
            var posX = ropeCache.PosX;
            var posY = ropeCache.PosY;
            Parallel.For(0, numPatches, p =>
            {
                float* dstRow = (float*)dstPtrL + (long)p * hiddenSize;
                float* xRow = (float*)xTableL + (long)posX[p] * hiddenSize;
                float* yRow = (float*)yTableL + (long)posY[p] * hiddenSize;
                int d = 0;
                for (; d <= hiddenSize - vLen; d += vLen)
                {
                    var acc = TensorComputePrimitives.LoadVector(dstRow + d)
                            + TensorComputePrimitives.LoadVector(xRow + d)
                            + TensorComputePrimitives.LoadVector(yRow + d);
                    TensorComputePrimitives.StoreVector(dstRow + d, acc);
                }
                for (; d < hiddenSize; d++)
                    dstRow[d] += xRow[d] + yRow[d];
            });
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim,
            Rope2DCache ropeCache)
        {
            string prefix = $"v.blk.{blockIdx}";

            // Fused fast path: run the whole block (attention + gated MLP) as one
            // on-device GGML graph, keeping every intermediate on the GPU instead
            // of round-tripping each of the ~30 sub-ops through host memory. Falls
            // back to the per-op path on any failure (e.g. flash-attn unsupported
            // for this head_dim/backend). No-op unless the encoder runs on a GGML
            // allocator (_useNativeAttention).
            if (_useNativeAttention && _fusedBlockEnabled
                && TryFusedEncoderBlock(hidden, prefix, numPatches, headDim, ropeCache))
                return hidden;

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

            string scaleKey = $"v.blk.{blockIdx}.out_scale.weight";
            if (_weights.TryGetValue(scaleKey, out var scaleTensor))
                Ops.Mul(result, result, scaleTensor);

            return result;
        }

        // TS_GEMMA4V_FUSED=0 disables the fused single-graph block (for A/B testing).
        private readonly bool _fusedBlockEnabled =
            Environment.GetEnvironmentVariable("TS_GEMMA4V_FUSED") != "0";

        private static readonly string[] _clampLinearSuffixes =
            { "attn_q", "attn_k", "attn_v", "attn_out", "ffn_gate", "ffn_up", "ffn_down" };

        /// <summary>
        /// Run the entire SigLIP encoder block through the fused native GGML kernel
        /// (one graph dispatch). Modifies <paramref name="hidden"/> in place and
        /// returns true on success; returns false (leaving hidden untouched) when a
        /// required weight is missing, an out_scale is present (unsupported by the
        /// kernel), or the native call throws — the caller then uses the per-op path.
        /// </summary>
        private bool TryFusedEncoderBlock(Tensor hidden, string prefix, int numPatches,
            int headDim, Rope2DCache ropeCache)
        {
            // The fused kernel doesn't implement the optional per-channel out_scale.
            if (_weights.ContainsKey($"{prefix}.out_scale.weight"))
                return false;

            if (!_weights.TryGetValue($"{prefix}.ln1.weight", out var ln1)
                || !_weights.TryGetValue($"{prefix}.attn_q.weight", out var qW)
                || !_weights.TryGetValue($"{prefix}.attn_k.weight", out var kW)
                || !_weights.TryGetValue($"{prefix}.attn_v.weight", out var vW)
                || !_weights.TryGetValue($"{prefix}.attn_q_norm.weight", out var qNorm)
                || !_weights.TryGetValue($"{prefix}.attn_k_norm.weight", out var kNorm)
                || !_weights.TryGetValue($"{prefix}.attn_post_norm.weight", out var apn)
                || !_weights.TryGetValue($"{prefix}.attn_out.weight", out var outW)
                || !_weights.TryGetValue($"{prefix}.ln2.weight", out var ln2)
                || !_weights.TryGetValue($"{prefix}.ffn_gate.weight", out var gateW)
                || !_weights.TryGetValue($"{prefix}.ffn_up.weight", out var upW)
                || !_weights.TryGetValue($"{prefix}.ffn_down.weight", out var downW)
                || !_weights.TryGetValue($"{prefix}.ffn_post_norm.weight", out var fpn))
                return false;

            float[] clamps = BuildClampArray(prefix);
            try
            {
                GgmlBasicOps.FusedGemma4VisionBlock(hidden, _eps, ln1,
                    qW, kW, vW, qNorm, kNorm, apn, outW,
                    ropeCache.CosX, ropeCache.SinX, ropeCache.CosY, ropeCache.SinY,
                    ln2, gateW, upW, downW, fpn,
                    clamps, numPatches, _numHeads, headDim);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // Pack the per-linear QAT activation clamps into the 28-float layout the
        // fused kernel expects: {q,k,v,out,gate,up,down} x {inMin,inMax,outMin,outMax}.
        // Absent clamps use ±float.MaxValue (the kernel treats |bound|>=3e38 as "none").
        private float[] BuildClampArray(string prefix)
        {
            var c = new float[28];
            for (int i = 0; i < _clampLinearSuffixes.Length; i++)
            {
                int b = i * 4;
                if (_clampParams.TryGetValue($"{prefix}.{_clampLinearSuffixes[i]}", out var cp) && cp.HasClamp)
                {
                    c[b] = cp.InMin; c[b + 1] = cp.InMax; c[b + 2] = cp.OutMin; c[b + 3] = cp.OutMax;
                }
                else
                {
                    c[b] = float.MinValue; c[b + 1] = float.MaxValue;
                    c[b + 2] = float.MinValue; c[b + 3] = float.MaxValue;
                }
            }
            return c;
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

        // 2D RoPE: split the head dim into an X-rotated first half and a
        // Y-rotated second half. Parallelised across patches with SIMD over the
        // (quarter-dim) rotation pairs.
        private unsafe void Apply2DRoPE(Tensor data, Rope2DCache ropeCache, int numPatches, int headDim)
        {
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;
            int quarterDim = halfDim / 2;
            int numHeads = _numHeads;
            int vLen = Vector<float>.Count;

            long ptrL = (long)ptr;
            fixed (float* cosX = ropeCache.CosX, sinX = ropeCache.SinX,
                          cosY = ropeCache.CosY, sinY = ropeCache.SinY)
            {
                long cosXL = (long)cosX, sinXL = (long)sinX, cosYL = (long)cosY, sinYL = (long)sinY;
                Parallel.For(0, numPatches, p =>
                {
                    float* dataPtr = (float*)ptrL;
                    int ropeBase = p * quarterDim;
                    float* cX = (float*)cosXL + ropeBase, sX = (float*)sinXL + ropeBase;
                    float* cY = (float*)cosYL + ropeBase, sY = (float*)sinYL + ropeBase;

                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = dataPtr + ((long)p * numHeads + h) * headDim;
                        float* headY = head + halfDim;
                        int j = 0;
                        for (; j <= quarterDim - vLen; j += vLen)
                        {
                            var x0 = TensorComputePrimitives.LoadVector(head + j);
                            var x1 = TensorComputePrimitives.LoadVector(head + j + quarterDim);
                            var cv = TensorComputePrimitives.LoadVector(cX + j);
                            var sv = TensorComputePrimitives.LoadVector(sX + j);
                            TensorComputePrimitives.StoreVector(head + j, x0 * cv - x1 * sv);
                            TensorComputePrimitives.StoreVector(head + j + quarterDim, x0 * sv + x1 * cv);

                            var y0 = TensorComputePrimitives.LoadVector(headY + j);
                            var y1 = TensorComputePrimitives.LoadVector(headY + j + quarterDim);
                            var cvy = TensorComputePrimitives.LoadVector(cY + j);
                            var svy = TensorComputePrimitives.LoadVector(sY + j);
                            TensorComputePrimitives.StoreVector(headY + j, y0 * cvy - y1 * svy);
                            TensorComputePrimitives.StoreVector(headY + j + quarterDim, y0 * svy + y1 * cvy);
                        }
                        for (; j < quarterDim; j++)
                        {
                            float x0 = head[j], x1 = head[j + quarterDim];
                            head[j] = x0 * cX[j] - x1 * sX[j];
                            head[j + quarterDim] = x0 * sX[j] + x1 * cX[j];
                            float y0 = headY[j], y1 = headY[j + quarterDim];
                            headY[j] = y0 * cY[j] - y1 * sY[j];
                            headY[j + quarterDim] = y0 * sY[j] + y1 * cY[j];
                        }
                    }
                });
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
            int hiddenSize = _hiddenSize;
            int nMerge = _nMerge;
            int vLen = Vector<float>.Count;

            long srcPtrL = (long)srcPtr, dstPtrL = (long)dstPtr;
            Parallel.For(0, mergedY, py =>
            {
                float* src = (float*)srcPtrL;
                float* dst = (float*)dstPtrL;
                for (int px = 0; px < mergedX; px++)
                {
                    int outIdx = py * mergedX + px;
                    float* outRow = dst + (long)outIdx * hiddenSize;
                    for (int d = 0; d < hiddenSize; d++)
                        outRow[d] = 0;

                    int count = 0;
                    for (int ky = 0; ky < nMerge; ky++)
                    {
                        for (int kx = 0; kx < nMerge; kx++)
                        {
                            int srcY = py * nMerge + ky;
                            int srcX = px * nMerge + kx;
                            if (srcY < patchesY && srcX < patchesX)
                            {
                                float* srcRow = src + (long)(srcY * patchesX + srcX) * hiddenSize;
                                int d = 0;
                                for (; d <= hiddenSize - vLen; d += vLen)
                                    TensorComputePrimitives.StoreVector(outRow + d,
                                        TensorComputePrimitives.LoadVector(outRow + d)
                                        + TensorComputePrimitives.LoadVector(srcRow + d));
                                for (; d < hiddenSize; d++)
                                    outRow[d] += srcRow[d];
                                count++;
                            }
                        }
                    }

                    float invCount = 1f / count;
                    var invVec = new Vector<float>(invCount);
                    int e = 0;
                    for (; e <= hiddenSize - vLen; e += vLen)
                        TensorComputePrimitives.StoreVector(outRow + e,
                            TensorComputePrimitives.LoadVector(outRow + e) * invVec);
                    for (; e < hiddenSize; e++)
                        outRow[e] *= invCount;
                }
            });

            // Scale by sqrt(hiddenSize)
            float scale = MathF.Sqrt(_hiddenSize);
            Ops.Mul(pooled, pooled, scale);

            // Vision standardization before projection (matches Ollama)
            if (_weights.TryGetValue("v.std_bias", out var stdBias) &&
                _weights.TryGetValue("v.std_scale", out var stdScale))
            {
                Ops.Sub(pooled, pooled, stdBias);
                Ops.Mul(pooled, pooled, stdScale);
            }

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

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

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

