// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// GLM-5.3-Flash (glm5next) vision tower: the GLM-OCR ViT plus the GLM4V-style
// merger, mirroring llama.cpp's clip_graph_glm5next / clip_graph_glm4v.
//
// Structure per block (RMS norms, no biases on the norms):
//   x += attn_out( SDPA( rope(q_norm(q)), rope(k_norm(k)), v ) )   [fused qkv +bias]
//   x += ffn_down( clampsilu(ffn_gate(x')) * clamp(ffn_up(x')) )   [all +bias]
// with per-head RMS q/k norms (weight [head_dim]), NeoX 2D vision RoPE in the
// same blocked Y|X layout as Qwen3-VL, and the SwiGLU clamp (gate <= L,
// up in [-L, L], L = clip.vision.swiglu_limit) that also applies to the merger.
//
// No learned position embeddings and no post-conv norm (GLM-OCR); the dual
// patch-embed convs collapse into one by adding their kernels.
//
// Merger: RMS post_ln -> 2x2 conv patch merger (run as a linear over the
// block-ordered 2x2 groups with a re-laid-out weight) -> FC (no bias) ->
// LayerNorm -> erf-GELU -> SwiGLU-clamp FFN -> [nTokens, 4096].
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public class GlmNextVisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly Dictionary<long, RopeCache> _ropeCache = new();
        private readonly Dictionary<long, int[]> _blockOrderCache = new();
        private readonly IAllocator _allocator;

        private readonly int _patchSize;
        private readonly int _hiddenSize;
        private readonly int _intermediateSize;
        private readonly int _numHeads;
        private readonly int _blockCount;
        private readonly float _eps;
        private readonly int _projectionDim;
        private readonly int _spatialMergeSize;
        private readonly float _ropeTheta;
        private readonly float _swigluLimit;
        private readonly string[] _blockPrefixes;

        private sealed class RopeCache
        {
            public required float[] CosTable { get; init; }
            public required float[] SinTable { get; init; }
        }

        public int ProjectionDim => _projectionDim;
        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;

        public GlmNextVisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            var gguf = new GgufFile(mmProjPath);

            string projector = gguf.GetString("clip.projector_type") ?? "";
            if (!GlmDsaArchitecture.IsGlm5Next(projector))
                Console.WriteLine($"Warning: mmproj projector_type is '{projector}', expected 'glm5next'.");

            _patchSize = (int)gguf.GetUint32("clip.vision.patch_size", 14);
            _hiddenSize = (int)gguf.GetUint32("clip.vision.embedding_length", 1024);
            _intermediateSize = (int)gguf.GetUint32("clip.vision.feed_forward_length", 4096);
            _numHeads = (int)gguf.GetUint32("clip.vision.attention.head_count", 16);
            _blockCount = (int)gguf.GetUint32("clip.vision.block_count", 24);
            _eps = gguf.GetFloat32("clip.vision.attention.layer_norm_epsilon", 1e-5f);
            _projectionDim = (int)gguf.GetUint32("clip.vision.projection_dim", 4096);
            _spatialMergeSize = (int)gguf.GetUint32("clip.vision.spatial_merge_size", 2);
            _ropeTheta = gguf.GetFloat32("clip.vision.rope.freq_base", 10000f);
            _swigluLimit = gguf.GetFloat32("clip.vision.swiglu_limit", 10f);

            Console.WriteLine($"GLM-5.3 vision encoder: patchSize={_patchSize}, hidden={_hiddenSize}, " +
                $"intermediate={_intermediateSize}, heads={_numHeads}, blocks={_blockCount}, " +
                $"projDim={_projectionDim}, mergeSize={_spatialMergeSize}, ropeTheta={_ropeTheta}, " +
                $"swigluLimit={_swigluLimit}, eps={_eps}");

            _blockPrefixes = new string[_blockCount];
            for (int i = 0; i < _blockCount; i++)
                _blockPrefixes[i] = $"v.blk.{i}";

            LoadWeights(gguf);
            CombinePatchEmbedWeights();
            BuildMergerLinearWeight();
            gguf.Dispose();
        }

        private void LoadWeights(GgufFile gguf)
        {
            Console.Write("Loading GLM-5.3 vision weights...");
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

                long[] tsShape = new long[info.Shape.Length];
                for (int i = 0; i < info.Shape.Length; i++)
                    tsShape[i] = (long)info.Shape[info.Shape.Length - 1 - i];

                var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                tensor.SetElementsAsFloat(f32);
                _weights[info.Name] = tensor;
                count++;
            }
            Console.WriteLine($" done ({count} tensors)");
        }

        /// <summary>The dual patch-embed convs read the SAME input, so their sum
        /// collapses into a single conv with added kernels.</summary>
        private void CombinePatchEmbedWeights()
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
        /// Re-lay the [2, 2, hidden, proj] patch-merger conv kernel as a linear
        /// weight over the block-ordered 2x2 groups. A merged row is
        /// [patch0 | patch1 | patch2 | patch3] with patches in (ky, kx) row-major
        /// order and each patch's channels contiguous; the conv kernel is
        /// (kx fastest, then ky, then channel) per output, so
        ///   W2[o][(ky*2+kx)*hidden + c] = W[o][c*4 + ky*2 + kx].
        /// </summary>
        private unsafe void BuildMergerLinearWeight()
        {
            if (!_weights.TryGetValue("mm.patch_merger.weight", out var conv))
                return;

            int merge = _spatialMergeSize;
            int cells = merge * merge;
            int hidden = _hiddenSize;
            int proj = (int)conv.Sizes[0];
            int inDim = cells * hidden;

            var linear = new Tensor(_allocator, DType.Float32, proj, inDim);
            float* src = TensorComputePrimitives.GetFloatPointer(conv);
            float* dst = TensorComputePrimitives.GetFloatPointer(linear);
            long srcL = (long)src, dstL = (long)dst;
            Parallel.For(0, proj, o =>
            {
                float* s = (float*)srcL + (long)o * inDim;
                float* d = (float*)dstL + (long)o * inDim;
                for (int c = 0; c < hidden; c++)
                    for (int p = 0; p < cells; p++)
                        d[p * hidden + c] = s[(long)c * cells + p];
            });

            _weights["mm.patch_merger.linear"] = linear;
        }

        /// <summary>
        /// Encode a preprocessed image (channel-first [3, H, W] floats) into
        /// projected embeddings [numMergedTokens, projectionDim].
        /// </summary>
        public unsafe Tensor Encode(float[] pixelValues, int resizedH, int resizedW)
        {
            long encodeStart = Stopwatch.GetTimestamp();
            int gridH = resizedH / _patchSize;
            int gridW = resizedW / _patchSize;
            int numPatches = gridH * gridW;
            int headDim = _hiddenSize / _numHeads;
            int halfDim = headDim / 2;

            // 1. Patch embedding straight into spatial-merge block order.
            var hidden = PatchEmbed(pixelValues, resizedH, resizedW, gridH, gridW);

            // 2. RoPE tables in the same block order (no learned position embd).
            RopeCache rope = GetOrCreateRopeCache(gridH, gridW, numPatches, halfDim);

            // 3. Encoder blocks. Fast path: all 24 blocks as ONE device-resident
            // GGML graph (weights cached on the device across encodes); the
            // managed per-block loop stays as the reference and fallback.
            bool fused = TryWholeEncoderFused(hidden, numPatches, headDim, halfDim,
                rope.CosTable, rope.SinTable);
            if (!fused)
            {
                for (int i = 0; i < _blockCount; i++)
                {
                    Console.Write($"\r  GLM vision block {i + 1}/{_blockCount}...");
                    EncoderBlock(hidden, i, numPatches, headDim, halfDim, rope.CosTable, rope.SinTable);
                }
                Console.WriteLine(" done");
            }

            // 4. RMS post_ln.
            var postNormed = RmsNormOp(hidden, "v.post_ln.weight");
            hidden.Dispose();

            // 5. Merger: 2x2 conv as a linear over the block-ordered groups.
            int mergedTokens = numPatches / (_spatialMergeSize * _spatialMergeSize);
            int mergedDim = _hiddenSize * _spatialMergeSize * _spatialMergeSize;

            using var mergedView = postNormed.View(mergedTokens, mergedDim);
            var merged = Ops.NewContiguous(mergedView);
            postNormed.Dispose();

            var pooled = LinearForwardWithBias(merged, "mm.patch_merger.linear", "mm.patch_merger.bias");
            merged.Dispose();

            // 6. FC projector: fc (no bias) -> LayerNorm -> erf-GELU.
            var fc = LinearForwardWithBias(pooled, "mm.model.fc.weight", null);
            pooled.Dispose();
            var fcn = Ops.LayerNorm(null, fc, _weights["mm.post_norm.weight"], _weights["mm.post_norm.bias"], 1e-5f);
            fc.Dispose();
            GeluErf(fcn);

            // 7. SwiGLU-clamp FFN projector.
            var gate = LinearForwardWithBias(fcn, "mm.gate.weight", null);
            var up = LinearForwardWithBias(fcn, "mm.up.weight", null);
            fcn.Dispose();
            ClampSiluMul(gate, up);
            up.Dispose();
            var projected = LinearForwardWithBias(gate, "mm.down.weight", null);
            gate.Dispose();

            double totalMs = (Stopwatch.GetTimestamp() - encodeStart) * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"  GLM vision encode: {totalMs:F0} ms, {numPatches} patches -> {mergedTokens} tokens");
            return projected;
        }

        // ---- patch embedding ------------------------------------------------

        private unsafe Tensor PatchEmbed(float[] pixelValues, int imgH, int imgW, int gridH, int gridW)
        {
            int numPatches = gridH * gridW;
            int C = 3;
            int P = _patchSize;
            int patchStride = C * P * P;

            string wName = _weights.ContainsKey("v.patch_embd.combined")
                ? "v.patch_embd.combined" : "v.patch_embd.weight";
            var convWeight = _weights[wName];
            Tensor weightT = GetOrCreatePatchEmbedTransposed(convWeight, wName, patchStride);

            int[] blockOrder = GetOrCreateBlockOrder(gridH, gridW);
            var im2col = new Tensor(_allocator, DType.Float32, numPatches, patchStride);
            float* im2colPtr = TensorComputePrimitives.GetFloatPointer(im2col);

            fixed (float* pixSrc = pixelValues)
            fixed (int* orderSrc = blockOrder)
            {
                long pixSrcL = (long)pixSrc;
                long orderSrcL = (long)orderSrc;
                long im2colL = (long)im2colPtr;
                Parallel.For(0, gridH, brow =>
                {
                    float* pix = (float*)pixSrcL;
                    int* order = (int*)orderSrcL;
                    float* dst = (float*)im2colL;
                    for (int col = 0; col < gridW; col++)
                    {
                        int destIdx = brow * gridW + col;
                        int rasterIdx = order[destIdx];
                        int py = rasterIdx / gridW;
                        int px = rasterIdx - py * gridW;
                        float* outRow = dst + (long)destIdx * patchStride;
                        int yBase = py * P;
                        int xBase = px * P;
                        for (int c = 0; c < 3; c++)
                        {
                            long imgChannelOffset = (long)c * imgH * imgW;
                            long outChannelOffset = (long)c * P * P;
                            for (int ky = 0; ky < P; ky++)
                            {
                                long srcOffset = imgChannelOffset + (long)(yBase + ky) * imgW + xBase;
                                Buffer.MemoryCopy(pix + srcOffset, outRow + outChannelOffset + (long)ky * P,
                                    P * sizeof(float), P * sizeof(float));
                            }
                        }
                    }
                });
            }

            var result = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            Ops.Addmm(result, 0, result, 1.0f, im2col, weightT);
            im2col.Dispose();

            if (_weights.TryGetValue("v.patch_embd.bias", out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        private unsafe Tensor GetOrCreatePatchEmbedTransposed(Tensor convWeight, string weightName, int patchStride)
        {
            string key = weightName + ".2d.T";
            if (_transposedWeights.TryGetValue(key, out var cached))
                return cached;

            int outDim = (int)convWeight.Sizes[0];
            using var flat = convWeight.View(outDim, patchStride);
            using var t = flat.Transpose();
            var result = Ops.NewContiguous(t);
            _transposedWeights[key] = result;
            return result;
        }

        private int[] GetOrCreateBlockOrder(int gridH, int gridW)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (_blockOrderCache.TryGetValue(key, out var cached))
                return cached;

            int merge = _spatialMergeSize;
            var order = new int[gridH * gridW];
            int idx = 0;
            for (int bh = 0; bh < gridH; bh += merge)
                for (int bw = 0; bw < gridW; bw += merge)
                    for (int mh = 0; mh < merge; mh++)
                        for (int mw = 0; mw < merge; mw++)
                            order[idx++] = (bh + mh) * gridW + (bw + mw);

            _blockOrderCache[key] = order;
            return order;
        }

        private bool _fusedEncoderUnavailable;

        /// <summary>Run all encoder blocks as one native GGML graph. Returns false
        /// (and never retries) when the native path is unavailable.</summary>
        private bool TryWholeEncoderFused(Tensor hidden, int numPatches, int headDim, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            if (_fusedEncoderUnavailable || _allocator is not TensorSharp.GGML.GgmlAllocator)
                return false;

            var ln1W = new Tensor[_blockCount];
            var qkvW = new Tensor[_blockCount];
            var qkvB = new Tensor[_blockCount];
            var qnW = new Tensor[_blockCount];
            var knW = new Tensor[_blockCount];
            var outW = new Tensor[_blockCount];
            var outB = new Tensor[_blockCount];
            var ln2W = new Tensor[_blockCount];
            var gateW = new Tensor[_blockCount];
            var gateB = new Tensor[_blockCount];
            var upW = new Tensor[_blockCount];
            var upB = new Tensor[_blockCount];
            var downW = new Tensor[_blockCount];
            var downB = new Tensor[_blockCount];

            for (int i = 0; i < _blockCount; i++)
            {
                string p = _blockPrefixes[i];
                if (!_weights.TryGetValue($"{p}.ln1.weight", out ln1W[i]) ||
                    !_weights.TryGetValue($"{p}.attn_qkv.weight", out qkvW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_qkv.bias", out qkvB[i]) ||
                    !_weights.TryGetValue($"{p}.attn_q_norm.weight", out qnW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_k_norm.weight", out knW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_out.weight", out outW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_out.bias", out outB[i]) ||
                    !_weights.TryGetValue($"{p}.ln2.weight", out ln2W[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_gate.weight", out gateW[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_gate.bias", out gateB[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_up.weight", out upW[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_up.bias", out upB[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_down.weight", out downW[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_down.bias", out downB[i]))
                {
                    _fusedEncoderUnavailable = true;
                    return false;
                }
            }

            try
            {
                bool ok = TensorSharp.GGML.GgmlBasicOps.GlmVisionEncoder(hidden, _eps,
                    1f / MathF.Sqrt(headDim), _swigluLimit,
                    numPatches, _numHeads, headDim, halfDim, cosTable, sinTable,
                    ln1W, qkvW, qkvB, qnW, knW, outW, outB, ln2W,
                    gateW, gateB, upW, upB, downW, downB);
                if (!ok)
                    _fusedEncoderUnavailable = true;
                return ok;
            }
            catch (Exception ex)
            {
                Console.WriteLine($"GLM fused vision encoder unavailable ({ex.Message}); using the per-block path.");
                _fusedEncoderUnavailable = true;
                return false;
            }
        }

        // ---- encoder block --------------------------------------------------

        private unsafe void EncoderBlock(Tensor hidden, int blockIdx, int numPatches,
            int headDim, int halfDim, float[] cosTable, float[] sinTable)
        {
            string prefix = _blockPrefixes[blockIdx];

            // attention
            using (var ln1 = RmsNormOp(hidden, $"{prefix}.ln1.weight"))
            using (var attnOut = SelfAttention(ln1, prefix, numPatches, headDim, halfDim, cosTable, sinTable))
            {
                Ops.Add(hidden, hidden, attnOut);
            }

            // SwiGLU-clamp MLP
            using var ln2 = RmsNormOp(hidden, $"{prefix}.ln2.weight");
            using var gate = LinearForwardWithBias(ln2, $"{prefix}.ffn_gate.weight", $"{prefix}.ffn_gate.bias");
            using var up = LinearForwardWithBias(ln2, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
            ClampSiluMul(gate, up);
            using var down = LinearForwardWithBias(gate, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
            Ops.Add(hidden, hidden, down);
        }

        private unsafe Tensor SelfAttention(Tensor input, string prefix, int numPatches,
            int headDim, int halfDim, float[] cosTable, float[] sinTable)
        {
            using var qkv = LinearForwardWithBias(input, $"{prefix}.attn_qkv.weight", $"{prefix}.attn_qkv.bias");

            int hiddenSize = _hiddenSize;
            int tripleHidden = 3 * hiddenSize;
            var q = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);
            var k = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);
            var v = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);

            float* qkvPtr = TensorComputePrimitives.GetFloatPointer(qkv);
            float* qPtr = TensorComputePrimitives.GetFloatPointer(q);
            float* kPtr = TensorComputePrimitives.GetFloatPointer(k);
            float* vPtr = TensorComputePrimitives.GetFloatPointer(v);
            long rowBytes = (long)hiddenSize * sizeof(float);
            long qkvL = (long)qkvPtr, qL = (long)qPtr, kL = (long)kPtr, vL = (long)vPtr;

            Parallel.For(0, numPatches, p =>
            {
                float* src = (float*)qkvL + (long)p * tripleHidden;
                Buffer.MemoryCopy(src, (float*)qL + (long)p * hiddenSize, rowBytes, rowBytes);
                Buffer.MemoryCopy(src + hiddenSize, (float*)kL + (long)p * hiddenSize, rowBytes, rowBytes);
                Buffer.MemoryCopy(src + 2 * hiddenSize, (float*)vL + (long)p * hiddenSize, rowBytes, rowBytes);
            });

            // per-head RMS q/k norms (weight [headDim], one weight shared by
            // every head), BEFORE the rope - transformers applies them to the
            // projected heads.
            ApplyHeadRmsNorm(q, numPatches, headDim, $"{prefix}.attn_q_norm.weight");
            ApplyHeadRmsNorm(k, numPatches, headDim, $"{prefix}.attn_k_norm.weight");

            ApplyVisionRoPE(q, numPatches, headDim, halfDim, cosTable, sinTable);
            ApplyVisionRoPE(k, numPatches, headDim, halfDim, cosTable, sinTable);

            float scale = 1f / MathF.Sqrt(headDim);

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
            using var flat = contiguous.View(numPatches, _hiddenSize);
            attnOutput.Dispose();

            return LinearForwardWithBias(flat, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
        }

        /// <summary>RMS-normalize each head's slice of every row against a shared
        /// [headDim] weight, in place.</summary>
        private unsafe void ApplyHeadRmsNorm(Tensor data, int numPatches, int headDim, string weightName)
        {
            if (!_weights.TryGetValue(weightName, out var w))
                return;
            float* ptr = TensorComputePrimitives.GetFloatPointer(data);
            float* wp = TensorComputePrimitives.GetFloatPointer(w);
            int numHeads = _numHeads;
            float eps = _eps;
            long ptrL = (long)ptr, wpL = (long)wp;

            Parallel.For(0, numPatches, p =>
            {
                float* dp = (float*)ptrL + (long)p * numHeads * headDim;
                float* wl = (float*)wpL;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = dp + (long)h * headDim;
                    double ss = 0;
                    for (int d = 0; d < headDim; d++)
                        ss += (double)head[d] * head[d];
                    float inv = 1.0f / MathF.Sqrt((float)(ss / headDim) + eps);
                    for (int d = 0; d < headDim; d++)
                        head[d] = head[d] * inv * wl[d];
                }
            });
        }

        private unsafe void ApplyVisionRoPE(Tensor data, int numPatches, int headDim, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            float* ptr = TensorComputePrimitives.GetFloatPointer(data);
            int numHeads = _numHeads;
            int vLen = Vector<float>.Count;

            fixed (float* cosPtr = cosTable, sinPtr = sinTable)
            {
                long ptrL = (long)ptr, cosPtrL = (long)cosPtr, sinPtrL = (long)sinPtr;
                Parallel.For(0, numPatches, p =>
                {
                    float* dataPtr = (float*)ptrL;
                    float* cTbl = (float*)cosPtrL;
                    float* sTbl = (float*)sinPtrL;
                    int cosBase = p * halfDim;

                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = dataPtr + ((long)p * numHeads + h) * headDim;
                        float* head1 = head + halfDim;
                        float* cosRow = cTbl + cosBase;
                        float* sinRow = sTbl + cosBase;

                        int d = 0;
                        for (; d <= halfDim - vLen; d += vLen)
                        {
                            var x0 = TensorComputePrimitives.LoadVector(head + d);
                            var x1 = TensorComputePrimitives.LoadVector(head1 + d);
                            var cv = TensorComputePrimitives.LoadVector(cosRow + d);
                            var sv = TensorComputePrimitives.LoadVector(sinRow + d);
                            TensorComputePrimitives.StoreVector(head + d, x0 * cv - x1 * sv);
                            TensorComputePrimitives.StoreVector(head1 + d, x0 * sv + x1 * cv);
                        }
                        for (; d < halfDim; d++)
                        {
                            float x0 = head[d];
                            float x1 = head1[d];
                            head[d] = x0 * cosRow[d] - x1 * sinRow[d];
                            head1[d] = x0 * sinRow[d] + x1 * cosRow[d];
                        }
                    }
                });
            }
        }

        /// <summary>gate = silu(min(gate, L)) * clamp(up, -L, L), in place into gate.</summary>
        private unsafe void ClampSiluMul(Tensor gate, Tensor up)
        {
            long total = gate.ElementCount();
            int rows = (int)gate.Sizes[0];
            long cols = total / rows;
            float* gp = TensorComputePrimitives.GetFloatPointer(gate);
            float* upp = TensorComputePrimitives.GetFloatPointer(up);
            float limit = _swigluLimit;
            long gL = (long)gp, uL = (long)upp;

            Parallel.For(0, rows, r =>
            {
                float* g = (float*)gL + r * cols;
                float* u = (float*)uL + r * cols;
                for (long i = 0; i < cols; i++)
                {
                    float gv = g[i];
                    if (limit > 0f && gv > limit) gv = limit;
                    float uv = u[i];
                    if (limit > 0f)
                    {
                        if (uv > limit) uv = limit;
                        else if (uv < -limit) uv = -limit;
                    }
                    float silu = gv / (1.0f + MathF.Exp(-gv));
                    g[i] = silu * uv;
                }
            });
        }

        private static float Erf(float x)
        {
            // Abramowitz & Stegun 7.1.26 in double precision: max abs error 1.5e-7.
            double t = 1.0 / (1.0 + 0.3275911 * Math.Abs(x));
            double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
                              + 0.254829592) * t * Math.Exp(-(double)x * x);
            return (float)(x >= 0 ? y : -y);
        }

        private unsafe void GeluErf(Tensor t)
        {
            long total = t.ElementCount();
            int rows = (int)t.Sizes[0];
            long cols = total / rows;
            float* ptr = TensorComputePrimitives.GetFloatPointer(t);
            long ptrL = (long)ptr;
            float invSqrt2 = 1.0f / MathF.Sqrt(2.0f);

            Parallel.For(0, rows, r =>
            {
                float* row = (float*)ptrL + r * cols;
                for (long i = 0; i < cols; i++)
                    row[i] = 0.5f * row[i] * (1.0f + Erf(row[i] * invSqrt2));
            });
        }

        private Tensor RmsNormOp(Tensor input, string weightName)
        {
            return Ops.RMSNorm(null, input, _weights[weightName], null, _eps);
        }

        private Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            Tensor weightT = GetOrCreateTransposedWeight(weightName);
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weightT.Sizes[1];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;
            Ops.Addmm(result, 0, result, 1.0f, src, weightT);
            contiguousInput?.Dispose();

            if (biasName != null && _weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        private Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            Tensor weight = _weights[weightName];
            using var weightViewT = weight.Transpose();
            transposed = Ops.NewContiguous(weightViewT);
            _transposedWeights[weightName] = transposed;
            return transposed;
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
                for (int bw = 0; bw < gridW; bw += _spatialMergeSize)
                    for (int mh = 0; mh < _spatialMergeSize; mh++)
                        for (int mw = 0; mw < _spatialMergeSize; mw++)
                        {
                            gridY[idx] = bh + mh;
                            gridX[idx] = bw + mw;
                            idx++;
                        }

            // Blocked frequency layout, exactly as Qwen3-VL / ggml VISION rope:
            // the first halfDim/2 rotary pairs encode the ROW position, the next
            // halfDim/2 the COLUMN position, each over theta^(2j/halfDim) bands.
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
                    cosTable[baseIdx + j] = MathF.Cos(angleY);
                    sinTable[baseIdx + j] = MathF.Sin(angleY);
                    cosTable[baseIdx + numBands + j] = MathF.Cos(angleX);
                    sinTable[baseIdx + numBands + j] = MathF.Sin(angleX);
                }
            }

            cache = new RopeCache { CosTable = cosTable, SinTable = sinTable };
            _ropeCache[key] = cache;
            return cache;
        }

        public void Dispose()
        {
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            _ropeCache.Clear();
            _blockOrderCache.Clear();
        }
    }
}
