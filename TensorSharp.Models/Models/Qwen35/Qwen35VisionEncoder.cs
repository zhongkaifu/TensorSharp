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
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    public class Qwen35VisionEncoder : IDisposable
    {
        private readonly Dictionary<string, Tensor> _weights = new();
        private readonly Dictionary<string, Tensor> _transposedWeights = new();
        private readonly Dictionary<long, Tensor> _positionEmbeddingCache = new();
        private readonly Dictionary<long, RopeCache> _ropeCache = new();
        private readonly Dictionary<long, int[]> _blockOrderCache = new();
        // Device-resident RoPE tables (direct-CUDA path only), keyed by geometry.
        private readonly Dictionary<long, (Tensor Cos, Tensor Sin)> _ropeDeviceCache = new();
        private readonly IAllocator _allocator;
        private readonly bool _useNativeAttention;
        // Direct-CUDA backend. Tensor data lives in device memory and every raw
        // host pointer checkout (TensorComputePrimitives.GetFloatPointer) forces a
        // synchronous DtoH copy plus a re-upload on the next kernel, so the encoder
        // must stay on tensor ops end to end. See CudaSelfAttention.
        private readonly bool _cudaDirect;
        // Optional ModelBase reference for cooperative GpuComputeLock
        // yielding between encoder blocks. See Gemma4VisionEncoder.
        private ModelBase _hostModel;
        public void SetHostModel(ModelBase model) => _hostModel = model;

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
        private readonly string[] _blockPrefixes;

        private sealed class RopeCache
        {
            public required float[] CosTable { get; init; }
            public required float[] SinTable { get; init; }
        }

        public int ProjectionDim => _projectionDim;
        public int PatchSize => _patchSize;
        public int SpatialMergeSize => _spatialMergeSize;

        /// <summary>
        /// Frames the patch embedding merges into one temporal patch: 2 when the
        /// projector ships the second conv slice (<c>v.patch_embd.weight.1</c>), as
        /// every Qwen-VL tower does, else 1. A still image fills both slices with
        /// itself (the combined weight); a video pair fills them with consecutive
        /// frames through <see cref="Encode(float[], float[], int, int)"/>.
        /// </summary>
        public int TemporalPatchSize => _weights.ContainsKey("v.patch_embd.weight.1") ? 2 : 1;

        public Qwen35VisionEncoder(string mmProjPath, IAllocator allocator)
        {
            _allocator = allocator;
            _useNativeAttention = allocator is GgmlAllocator;
            _cudaDirect = allocator is TensorSharp.Cuda.CudaAllocator;
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

            _blockPrefixes = new string[_blockCount];
            for (int i = 0; i < _blockCount; i++)
                _blockPrefixes[i] = $"v.blk.{i}";

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
        public Tensor Encode(float[] pixelValues, int resizedH, int resizedW)
            => EncodeCore(pixelValues, null, resizedH, resizedW);

        /// <summary>
        /// Encode one temporal patch of a video: two consecutive sampled frames, both
        /// channel-first [C, H, W] at the same resized size, merged by the patch
        /// embedding's two temporal conv slices exactly as the Qwen-VL processor stacks
        /// them (frame order matters: the first frame meets <c>v.patch_embd.weight</c>,
        /// the second <c>v.patch_embd.weight.1</c>). Everything after the patch
        /// embedding is the still-image path, which is the reference behaviour: the
        /// tower attends within one temporal patch only. Output:
        /// [numMergedTokens, projectionDim], the same token count as one still frame.
        /// </summary>
        public Tensor Encode(float[] firstFrame, float[] secondFrame, int resizedH, int resizedW)
        {
            ArgumentNullException.ThrowIfNull(firstFrame);
            ArgumentNullException.ThrowIfNull(secondFrame);
            if (TemporalPatchSize != 2)
                throw new NotSupportedException(
                    "This vision projector has no second temporal patch slice (v.patch_embd.weight.1), " +
                    "so it cannot merge a pair of video frames.");
            if (firstFrame.Length != secondFrame.Length)
                throw new ArgumentException("Both frames of a temporal patch must have the same resized size.");
            return EncodeCore(firstFrame, secondFrame, resizedH, resizedW);
        }

        private unsafe Tensor EncodeCore(float[] pixelValues, float[] secondFrame, int resizedH, int resizedW)
        {
            ArgumentNullException.ThrowIfNull(pixelValues);
            int factor = checked(_patchSize * _spatialMergeSize);
            if (resizedH <= 0 || resizedW <= 0 || resizedH % factor != 0 || resizedW % factor != 0)
                throw new ArgumentException($"Vision dimensions must be positive multiples of {factor}.");
            long expectedPixels = checked(3L * resizedH * resizedW);
            if (pixelValues.LongLength != expectedPixels ||
                (secondFrame != null && secondFrame.LongLength != expectedPixels))
                throw new ArgumentException("Each frame must contain exactly 3 * height * width channel-first values.");
            long encodeStart = Stopwatch.GetTimestamp();
            int gridH = resizedH / _patchSize;
            int gridW = resizedW / _patchSize;
            int numPatches = gridH * gridW;
            int headDim = _hiddenSize / _numHeads;
            int halfDim = headDim / 2;


            // 1. Patch embedding (Conv2D). The im2col rows are emitted directly in
            //    spatial-merge block order, so the separate raster->block reorder
            //    pass (a full [numPatches, hiddenSize] copy, and on a device
            //    allocator a full round trip) is not needed. The position
            //    embedding below is built in the same order, so the sum is exactly
            //    the raster pipeline's result with its rows permuted.
            long t0 = Stopwatch.GetTimestamp();
            var blockOrdered = PatchEmbed(pixelValues, secondFrame, resizedH, resizedW, gridH, gridW);
            long patchEmbedTicks = Stopwatch.GetTimestamp() - t0;
            Trace("patchEmbed", blockOrdered);

            // 2. Position embedding (bilinear interpolation, block order)
            AddPositionEmbedding(blockOrdered, gridH, gridW);
            Trace("blockOrder", blockOrdered);

            // 3. Build block-order grid coordinate arrays for RoPE
            RopeCache ropeCache = GetOrCreateRopeCache(gridH, gridW, numPatches, halfDim);

            // 4. Encoder blocks
            long blocksStart = Stopwatch.GetTimestamp();
            // Fast path: run ALL blocks as one device-resident GGML graph (one
            // sync, residual kept on-device) instead of 2 synchronous PCIe
            // round-tripping dispatches per block. Falls back to the per-block
            // loop on any failure. CPU-allocator path keeps the per-block loop.
            bool wholeEncoderDone = false;
            if (_useNativeAttention && s_wholeEncoderFusedEnabled)
            {
                wholeEncoderDone = TryWholeEncoderFused(blockOrdered, numPatches, headDim, halfDim,
                    ropeCache.CosTable, ropeCache.SinTable);
            }
            if (!wholeEncoderDone)
            {
                for (int i = 0; i < _blockCount; i++)
                {
                    if (!s_traceEnabled)
                        Console.Write($"\r  Vision encoder block {i + 1}/{_blockCount}...");
                    blockOrdered = EncoderBlock(blockOrdered, i, numPatches, headDim, halfDim,
                        ropeCache.CosTable, ropeCache.SinTable);
                    Trace($"block{i}", blockOrdered);
                    // Flush MLX's lazy graph at every block boundary. Without this
                    // the [numHeads, numPatches, numPatches] attention-scores
                    // intermediate (~340 MB at 768x768, multiple GB for larger
                    // images) from every previous block stays referenced as a
                    // graph input until the next forced host read in
                    // VisionSelfAttention. Evaluating them all together exceeds
                    // the Metal command-buffer budget and crashes with
                    // kIOGPUCommandBufferCallbackErrorOutOfMemory. AsyncEval kicks
                    // the GPU so block N's intermediates are freed while the host
                    // queues block N+1.
                    MlxFusedOps.TryAsyncEvaluate(blockOrdered);
                    // Yield GpuComputeLock between encoder blocks so concurrent
                    // decode requests on the engine worker stay responsive.
                    _hostModel?.YieldGpuComputeLock();
                }
            }
            // Drain before splitting the phase timings; direct-CUDA kernels are
            // queued, not awaited, so an unsynchronized "blocks" number would just
            // measure launch overhead.
            if (_cudaDirect && _allocator is TensorSharp.Cuda.CudaAllocator blocksAllocator)
                blocksAllocator.Synchronize();
            long blocksTicks = Stopwatch.GetTimestamp() - blocksStart;
            Console.WriteLine(" done");

            // 5. Post-LayerNorm
            var postNormed = LayerNormOp(blockOrdered, "v.post_ln.weight", "v.post_ln.bias");
            blockOrdered.Dispose();

            // 6. Spatial merge + projection
            long projStart = Stopwatch.GetTimestamp();
            int mergedPatches = numPatches / (_spatialMergeSize * _spatialMergeSize);
            int mergedDim = _hiddenSize * _spatialMergeSize * _spatialMergeSize;

            using var merged = postNormed.View(mergedPatches, mergedDim);
            var mergedContig = Ops.NewContiguous(merged);
            postNormed.Dispose();

            using var fc1 = LinearForwardWithBias(mergedContig, "mm.0.weight", "mm.0.bias");
            mergedContig.Dispose();
            Ops.GELU(fc1, fc1);

            var projected = LinearForwardWithBias(fc1, "mm.2.weight", "mm.2.bias");

            // The direct-CUDA path only *queues* its kernels, so without this the
            // numbers below would report launch cost (tens of ms) rather than
            // encode time. The caller consumes the embeddings immediately, so the
            // work has to complete here regardless.
            if (_cudaDirect && _allocator is TensorSharp.Cuda.CudaAllocator cudaAllocator)
                cudaAllocator.Synchronize();
            long projTicks = Stopwatch.GetTimestamp() - projStart;

            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double totalMs = (Stopwatch.GetTimestamp() - encodeStart) * msPerTick;
            Console.WriteLine($"  Vision encode: {totalMs:F0} ms total " +
                $"(patchEmbed {patchEmbedTicks * msPerTick:F0} ms, " +
                $"blocks {blocksTicks * msPerTick:F0} ms, " +
                $"proj {projTicks * msPerTick:F0} ms), " +
                $"{numPatches} patches -> {mergedPatches} tokens");

            return projected;
        }

        /// <summary>
        /// Conv2D patch embedding with combined temporal weights.
        /// Output in spatial-merge block order: [numPatches, hiddenSize].
        ///
        /// Reformulated as a GEMM (im2col + matmul) so the heavy compute can run on the GPU
        /// when available. The im2col packing itself is parallelised on the CPU using SIMD
        /// per-channel row copies. This is dramatically faster than the original quintuple
        /// nested loop (single-threaded scalar) implementation, especially on Apple Silicon
        /// where matmul saturates the unified-memory bandwidth.
        ///
        /// The im2col rows are emitted in block order (see
        /// <see cref="GetOrCreateBlockOrder"/>) rather than raster order, so the
        /// encoder never needs a separate reorder pass over the embedded patches.
        /// </summary>
        private unsafe Tensor PatchEmbed(float[] pixelValues, float[] secondFrame, int imgH, int imgW, int gridH, int gridW)
        {
            int numPatches = gridH * gridW;
            int C = 3;
            int P = _patchSize;
            int patchStride = C * P * P;
            // A video pair is one im2col row of BOTH frames' patches side by side
            // ([frame0 C*P*P | frame1 C*P*P]) against the two temporal conv slices
            // concatenated the same way, which is conv3d with kernel (2, P, P) over
            // the stacked frames. A still image keeps the summed (combined) slice.
            int frames = secondFrame != null ? 2 : 1;
            int rowStride = frames * patchStride;

            string wName = secondFrame != null
                ? "v.patch_embd.temporal"
                : _weights.ContainsKey("v.patch_embd.combined") ? "v.patch_embd.combined" : "v.patch_embd.weight";
            Tensor convWeight = secondFrame != null
                ? GetOrCreateTemporalPatchWeight(patchStride)
                : _weights[wName];
            // convWeight shape (post-load reverse): [hiddenSize, C, P, P]. We view it as a
            // 2D matrix [hiddenSize, patchStride] for the matmul. This relies on the weight
            // being stored row-major in C * P * P element order per output channel, which is
            // how PyTorch / GGUF lay out conv2d weights.
            string biasName = "v.patch_embd.bias";

            Tensor weightView2D = GetOrCreatePatchEmbedWeight2D(convWeight, wName, rowStride);
            Tensor weightT = GetOrCreatePatchEmbedTransposed(weightView2D, wName);

            // Build im2col matrix [numPatches, rowStride] in parallel.
            int[] blockOrder = GetOrCreateBlockOrder(gridH, gridW);
            var im2col = new Tensor(_allocator, DType.Float32, numPatches, rowStride);
            using (var staging = new HostStaging(im2col, _cudaDirect))
            {
                float* im2colPtr = staging.Ptr;

                // Capture pointer locally since pixelValues is a managed array - we pin once
                // via fixed at the top so the inner Parallel.For lambda can share the address.
                fixed (float* pixSrc = pixelValues)
                fixed (float* pixSrc2 = secondFrame)
                fixed (int* orderSrc = blockOrder)
                {
                    long pixSrcL = (long)pixSrc;
                    long pixSrc2L = (long)pixSrc2;
                    long orderSrcL = (long)orderSrc;
                    Parallel.For(0, gridH, brow =>
                    {
                        int* order = (int*)orderSrcL;
                        for (int col = 0; col < gridW; col++)
                        {
                            // Destination row is block-ordered; the raster patch it
                            // pulls from is order[destIdx].
                            int destIdx = brow * gridW + col;
                            int rasterIdx = order[destIdx];
                            int py = rasterIdx / gridW;
                            int px = rasterIdx - py * gridW;
                            float* outRow = im2colPtr + (long)destIdx * rowStride;
                            int yBase = py * P;
                            int xBase = px * P;

                            for (int f = 0; f < frames; f++)
                            {
                                float* pixSrcLocal = (float*)(f == 0 ? pixSrcL : pixSrc2L);
                                float* frameRow = outRow + (long)f * patchStride;
                                for (int c = 0; c < C; c++)
                                {
                                    long imgChannelOffset = (long)c * imgH * imgW;
                                    long outChannelOffset = (long)c * P * P;
                                    for (int ky = 0; ky < P; ky++)
                                    {
                                        int imgY = yBase + ky;
                                        long srcOffset = imgChannelOffset + (long)imgY * imgW + xBase;
                                        long dstOffset = outChannelOffset + (long)ky * P;
                                        Buffer.MemoryCopy(pixSrcLocal + srcOffset, frameRow + dstOffset,
                                            P * sizeof(float), P * sizeof(float));
                                    }
                                }
                            }
                        }
                    });
                }
            }

            // result = im2col @ weight^T  (+ bias if present)
            int outDim = (int)weightView2D.Sizes[0];
            var result = new Tensor(_allocator, DType.Float32, numPatches, outDim);
            Ops.Addmm(result, 0, result, 1.0f, im2col, weightT);
            im2col.Dispose();

            if (_weights.TryGetValue(biasName, out var bias))
                Ops.Add(result, result, bias);

            return result;
        }

        /// <summary>
        /// Host write buffer for a tensor that may live in device memory. On the
        /// direct-CUDA backend, checking out a raw host pointer (PtrAtElement)
        /// first synchronously copies the tensor's uninitialized device bytes back
        /// to the host; for a write-only fill that copy is pure waste. Staging in a
        /// pinned array and pushing it with a single bulk CopyToStorage skips it.
        /// On host-resident allocators the tensor's own buffer is used directly.
        /// </summary>
        private sealed unsafe class HostStaging : IDisposable
        {
            private readonly Tensor _tensor;
            private readonly float[] _buffer;
            private System.Runtime.InteropServices.GCHandle _handle;

            public float* Ptr { get; }

            public HostStaging(Tensor tensor, bool stageOnHost)
            {
                _tensor = tensor;
                if (!stageOnHost)
                {
                    Ptr = TensorComputePrimitives.GetFloatPointer(tensor);
                    return;
                }

                _buffer = new float[tensor.ElementCount()];
                _handle = System.Runtime.InteropServices.GCHandle.Alloc(
                    _buffer, System.Runtime.InteropServices.GCHandleType.Pinned);
                Ptr = (float*)_handle.AddrOfPinnedObject();
            }

            public void Dispose()
            {
                if (_buffer == null)
                    return;

                _tensor.Storage.CopyToStorage(_tensor.StorageOffset, _handle.AddrOfPinnedObject(),
                    _tensor.ElementCount() * sizeof(float));
                _handle.Free();
            }
        }

        /// <summary>
        /// The two temporal conv slices side by side as one [hiddenSize, 2 * C * P * P]
        /// matrix, the weight a video pair's im2col row multiplies. Built once.
        /// </summary>
        private Tensor GetOrCreateTemporalPatchWeight(int patchStride)
        {
            const string key = "v.patch_embd.temporal";
            if (_weights.TryGetValue(key, out var cached))
                return cached;

            var w0 = _weights["v.patch_embd.weight"];
            var w1 = _weights["v.patch_embd.weight.1"];
            int outDim = (int)w0.Sizes[0];
            float[] a = w0.GetElementsAsFloat(outDim * patchStride);
            float[] b = w1.GetElementsAsFloat(outDim * patchStride);
            float[] rows = new float[(long)outDim * 2 * patchStride];
            for (int o = 0; o < outDim; o++)
            {
                Array.Copy(a, (long)o * patchStride, rows, (long)o * 2 * patchStride, patchStride);
                Array.Copy(b, (long)o * patchStride, rows, (long)o * 2 * patchStride + patchStride, patchStride);
            }
            var temporal = new Tensor(_allocator, DType.Float32, outDim, 2 * patchStride);
            temporal.SetElementsAsFloat(rows);
            _weights[key] = temporal;
            return temporal;
        }

        private Tensor GetOrCreatePatchEmbedWeight2D(Tensor convWeight, string weightName, int patchStride)
        {
            string key = weightName + ".2d";
            if (_transposedWeights.TryGetValue(key, out var cached))
                return cached;

            int outDim = (int)convWeight.Sizes[0];
            // Reshape the 4D conv weight [outDim, C, P, P] into a 2D matrix [outDim, patchStride]
            // through a contiguous copy. This is a one-time allocation.
            Tensor flat = new Tensor(_allocator, DType.Float32, outDim, patchStride);
            unsafe
            {
                float* src = GetFloatPtr(convWeight);
                float* dst = GetFloatPtr(flat);
                long bytes = (long)outDim * patchStride * sizeof(float);
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            }

            _transposedWeights[key] = flat;
            return flat;
        }

        private Tensor GetOrCreatePatchEmbedTransposed(Tensor weight2D, string weightName)
        {
            string key = weightName + ".2d.T";
            if (_transposedWeights.TryGetValue(key, out var cached))
                return cached;

            Tensor result;
            if (_cudaDirect)
            {
                result = HostTranspose2D(weight2D);
            }
            else
            {
                using var t = weight2D.Transpose();
                result = Ops.NewContiguous(t);
            }

            _transposedWeights[key] = result;
            return result;
        }

        /// <summary>
        /// Add bilinearly-interpolated position embeddings (in block order, matching
        /// the patch-embed output).
        /// </summary>
        private void AddPositionEmbedding(Tensor hidden, int gridH, int gridW)
        {
            Ops.Add(hidden, hidden, GetOrCreatePositionEmbedding(gridH, gridW));
        }

        /// <summary>
        /// Permutation from spatial-merge block order to raster (h, w) order:
        /// blockOrder[i] is the raster patch index that lands at block-order row i.
        /// Blocks are mergeSize x mergeSize groups iterated in raster order; within
        /// a group the order is (mh, mw). This is the same enumeration the RoPE
        /// coordinate tables use, so patch-embed rows, position embeddings and RoPE
        /// positions all agree.
        /// </summary>
        private int[] GetOrCreateBlockOrder(int gridH, int gridW)
        {
            long key = ((long)gridH << 32) | (uint)gridW;
            if (_blockOrderCache.TryGetValue(key, out var cached))
                return cached;

            int merge = _spatialMergeSize;
            var order = new int[gridH * gridW];
            int idx = 0;
            for (int bh = 0; bh < gridH; bh += merge)
            {
                for (int bw = 0; bw < gridW; bw += merge)
                {
                    for (int mh = 0; mh < merge; mh++)
                    {
                        for (int mw = 0; mw < merge; mw++)
                            order[idx++] = (bh + mh) * gridW + (bw + mw);
                    }
                }
            }

            _blockOrderCache[key] = order;
            return order;
        }

        // TS_QWEN35_VENC_FUSED=0 forces the per-block path (A/B + safety kill-switch).
        private static readonly bool s_wholeEncoderFusedEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_VENC_FUSED") != "0";

        // TS_QWEN35_VENC_FUSED_ATTN=0 bypasses the fused native attention
        // subgraph in the per-block path (keeping the fused MLP), forcing the
        // managed split + RoPE + Ops.ScaledDotProductAttention chain. A/B
        // switch for isolating native attention-kernel issues; this is how
        // the head_dim-72 CUDA flash-attn precision bug was pinned down.
        private static readonly bool s_fusedAttnEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_VENC_FUSED_ATTN") != "0";

        // TS_QWEN35_VENC_TRACE=1 prints a checksum of the residual stream at every
        // encoder stage. Used to localize a numeric divergence between backends
        // (run the same image through two allocators and diff the first stage whose
        // checksum differs). Forces a host read, so it is debug-only.
        private static readonly bool s_traceEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_VENC_TRACE") == "1";

        private static void Trace(string label, Tensor t)
        {
            if (!s_traceEnabled || t == null)
                return;

            using var contig = t.IsContiguous() ? null : Ops.NewContiguous(t);
            Tensor src = contig ?? t;
            int n = (int)src.ElementCount();
            float[] data = src.GetElementsAsFloat(n);
            double sum = 0, sumsq = 0;
            float min = float.MaxValue, max = float.MinValue;
            for (int i = 0; i < n; i++)
            {
                float v = data[i];
                sum += v;
                sumsq += (double)v * v;
                if (v < min) min = v;
                if (v > max) max = v;
            }
            Console.WriteLine($"[venc-trace] {label,-28} shape={string.Join("x", src.Sizes.ToArray())} " +
                $"sum={sum:F4} sumsq={sumsq:F4} min={min:F5} max={max:F5} first={data[0]:F6},{data[1]:F6},{data[2]:F6}");
        }

        // One-shot degrade notices: these fused paths used to fall back silently,
        // leaving image prefill on the per-op chain with no trace of why it was slow.
        private bool _wholeEncoderFusedWarned;
        private bool _fusedVisionAttnWarned;
        private bool _fusedVisionMlpWarned;

        /// <summary>
        /// Run every encoder block as ONE device-resident GGML graph
        /// (<see cref="GgmlBasicOps.Qwen35VisionEncoder"/>). Collects the per-block
        /// F32 weights into block-major arrays and dispatches once. Returns false
        /// (→ caller runs the per-block loop) if any weight is missing or the native
        /// path declines.
        /// </summary>
        private bool TryWholeEncoderFused(Tensor hidden, int numPatches, int headDim, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            int n = _blockCount;
            var ln1W = new Tensor[n]; var ln1B = new Tensor[n];
            var qkvW = new Tensor[n]; var qkvB = new Tensor[n];
            var outW = new Tensor[n]; var outB = new Tensor[n];
            var ln2W = new Tensor[n]; var ln2B = new Tensor[n];
            var upW = new Tensor[n]; var upB = new Tensor[n];
            var downW = new Tensor[n]; var downB = new Tensor[n];

            for (int i = 0; i < n; i++)
            {
                string p = _blockPrefixes[i];
                if (!_weights.TryGetValue($"{p}.ln1.weight", out ln1W[i]) ||
                    !_weights.TryGetValue($"{p}.ln1.bias", out ln1B[i]) ||
                    !_weights.TryGetValue($"{p}.attn_qkv.weight", out qkvW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_qkv.bias", out qkvB[i]) ||
                    !_weights.TryGetValue($"{p}.attn_out.weight", out outW[i]) ||
                    !_weights.TryGetValue($"{p}.attn_out.bias", out outB[i]) ||
                    !_weights.TryGetValue($"{p}.ln2.weight", out ln2W[i]) ||
                    !_weights.TryGetValue($"{p}.ln2.bias", out ln2B[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_up.weight", out upW[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_up.bias", out upB[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_down.weight", out downW[i]) ||
                    !_weights.TryGetValue($"{p}.ffn_down.bias", out downB[i]))
                    return false;
            }

            // No VRAM-fit guard any more. It existed for the pre-flash graph that
            // materialized an O(numPatches^2) score tensor (~4 GB at 7920 patches,
            // 95 s WDDM-thrash next to a resident 12 GB model). Attention now runs
            // through ggml flash_attn_ext (the vendored ggml-cuda has a head_dim=72
            // kernel), so the per-encode budget is just a handful of
            // [3*hidden, numPatches] activations (~0.3 GB at 7920 patches). The
            // per-block fallback can never win post-flash: it caches the SAME
            // ~1.6 GB of block weights resident (bind_w cacheable buffers), peaks
            // on the same per-tensor activations, and is 5-15x slower (measured
            // 2.7 s fused vs 47 s per-block at 7920 patches with 0.4 GB free —
            // WDDM absorbs the scratch fine). TS_QWEN35_VENC_FUSED=0 remains the
            // kill-switch that forces the per-block path.

            float attnScale = 1f / MathF.Sqrt(headDim);
            try
            {
                bool ok = GgmlBasicOps.Qwen35VisionEncoder(hidden, _eps, attnScale,
                    numPatches, _numHeads, headDim, halfDim, cosTable, sinTable,
                    ln1W, ln1B, qkvW, qkvB, outW, outB, ln2W, ln2B, upW, upB, downW, downB);
                if (ok)
                    _hostModel?.YieldGpuComputeLock();
                return ok;
            }
            catch (Exception e)
            {
                if (!_wholeEncoderFusedWarned)
                {
                    _wholeEncoderFusedWarned = true;
                    Console.Error.WriteLine(
                        $"[qwen35-vision] whole-encoder fused graph failed ({e.Message}); " +
                        "falling back to the per-block encoder (5-15x slower image prefill). Reported once.");
                }
                return false;
            }
        }

        private Tensor EncoderBlock(Tensor hidden, int blockIdx, int numPatches, int headDim,
            int halfDim, float[] cosTable, float[] sinTable)
        {
            string prefix = _blockPrefixes[blockIdx];
            float attnScale = 1f / MathF.Sqrt(headDim);

            // Fully fused attention path: LN + QKV + RoPE + SDPA + out + residual in one dispatch.
            bool fusedAttn = false;
            if (_useNativeAttention && s_fusedAttnEnabled
                && _weights.TryGetValue($"{prefix}.ln1.weight", out var ln1W)
                && _weights.TryGetValue($"{prefix}.ln1.bias", out var ln1B)
                && _weights.TryGetValue($"{prefix}.attn_qkv.weight", out var qkvW)
                && _weights.TryGetValue($"{prefix}.attn_qkv.bias", out var qkvB)
                && _weights.TryGetValue($"{prefix}.attn_out.weight", out var outW)
                && _weights.TryGetValue($"{prefix}.attn_out.bias", out var outB))
            {
                try
                {
                    GgmlBasicOps.FusedVisionAttention(hidden, ln1W, ln1B, _eps,
                        qkvW, qkvB, outW, outB,
                        cosTable, sinTable, numPatches, _numHeads, headDim, halfDim, attnScale);
                    fusedAttn = true;
                }
                catch (Exception e)
                {
                    fusedAttn = false;
                    if (!_fusedVisionAttnWarned)
                    {
                        _fusedVisionAttnWarned = true;
                        Console.Error.WriteLine(
                            $"[qwen35-vision] FusedVisionAttention failed ({e.Message}); " +
                            "using the managed split + RoPE + SDPA chain instead " +
                            "(significantly slower image prefill). Reported once.");
                    }
                }
            }

            if (!fusedAttn)
            {
                var ln1 = LayerNormOp(hidden, $"{prefix}.ln1.weight", $"{prefix}.ln1.bias");
                Trace($"{prefix}.ln1", ln1);
                var attnOut = VisionSelfAttention(ln1, prefix, numPatches, headDim, halfDim,
                    cosTable, sinTable);
                ln1.Dispose();
                Trace($"{prefix}.attnOut", attnOut);
                Ops.Add(hidden, attnOut, hidden);
                attnOut.Dispose();
            }
            Trace($"{prefix}.postAttn", hidden);

            // Fused MLP path: LayerNorm + up + GELU + down + residual in one GPU dispatch.
            if (_useNativeAttention
                && _weights.TryGetValue($"{prefix}.ln2.weight", out var ln2W)
                && _weights.TryGetValue($"{prefix}.ln2.bias", out var ln2B)
                && _weights.TryGetValue($"{prefix}.ffn_up.weight", out var upW)
                && _weights.TryGetValue($"{prefix}.ffn_up.bias", out var upB)
                && _weights.TryGetValue($"{prefix}.ffn_down.weight", out var downW)
                && _weights.TryGetValue($"{prefix}.ffn_down.bias", out var downB))
            {
                try
                {
                    GgmlBasicOps.FusedVisionMLP(hidden, ln2W, ln2B, _eps, upW, upB, downW, downB);
                    return hidden;
                }
                catch (Exception e)
                {
                    // Fall back to unfused path on failure.
                    if (!_fusedVisionMlpWarned)
                    {
                        _fusedVisionMlpWarned = true;
                        Console.Error.WriteLine(
                            $"[qwen35-vision] FusedVisionMLP failed ({e.Message}); " +
                            "using the unfused LayerNorm + up/GELU/down chain instead " +
                            "(significantly slower image prefill). Reported once.");
                    }
                }
            }

            var ln2 = LayerNormOp(hidden, $"{prefix}.ln2.weight", $"{prefix}.ln2.bias");
            var mlpOut = VisionMLP(ln2, prefix);
            ln2.Dispose();

            Ops.Add(hidden, hidden, mlpOut);
            mlpOut.Dispose();

            return hidden;
        }

        /// <summary>
        /// Direct-CUDA self-attention: QKV split, RoPE, bidirectional attention and
        /// the output projection all run as device kernels, so nothing in the block
        /// touches a host pointer.
        ///
        /// The generic path below cannot be used here. Its RoPE and Q/K/V split go
        /// through raw host pointers, which on CudaStorage means a synchronous DtoH
        /// copy of the whole activation plus a re-upload on the next kernel — twice
        /// per encoder block — and its fallback attention materializes an
        /// O(numPatches^2) score tensor per head.
        /// </summary>
        private Tensor CudaSelfAttention(Tensor qkv, string prefix, int numPatches,
            int headDim, int halfDim, float[] cosTable, float[] sinTable)
        {
            Tensor q, k, v;
            using (var qView = qkv.Narrow(1, 0, _hiddenSize))
            using (var kView = qkv.Narrow(1, _hiddenSize, _hiddenSize))
            using (var vView = qkv.Narrow(1, 2 * _hiddenSize, _hiddenSize))
            {
                q = Ops.NewContiguous(qView);
                k = Ops.NewContiguous(kView);
                v = Ops.NewContiguous(vView);
            }

            try
            {
                var rope = GetOrCreateRopeDeviceTables(numPatches, halfDim, cosTable, sinTable);
                bool roped =
                    TensorSharp.Cuda.CudaFusedOps.TryNeoxRopeFlat(q, rope.Cos, rope.Sin, _numHeads, numPatches, headDim, halfDim) &&
                    TensorSharp.Cuda.CudaFusedOps.TryNeoxRopeFlat(k, rope.Cos, rope.Sin, _numHeads, numPatches, headDim, halfDim);
                if (!roped)
                {
                    ApplyVisionRoPE(q, numPatches, headDim, halfDim, cosTable, sinTable);
                    ApplyVisionRoPE(k, numPatches, headDim, halfDim, cosTable, sinTable);
                }

                float scale = 1f / MathF.Sqrt(headDim);
                var attn = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
                try
                {
                    if (!TensorSharp.Cuda.CudaFusedOps.TryVisionAttention(
                            attn, q, k, v, _numHeads, numPatches, headDim, scale))
                    {
                        // No specialized kernel for this head dim: the generic SDPA
                        // kernel takes the same [1, seq, heads, dim] layout.
                        using var q4 = q.View(1, numPatches, _numHeads, headDim);
                        using var k4 = k.View(1, numPatches, _numHeads, headDim);
                        using var v4 = v.View(1, numPatches, _numHeads, headDim);
                        using var out4 = attn.View(1, numPatches, _numHeads, headDim);
                        Ops.ScaledDotProductAttention(out4, q4, k4, v4, null, scale);
                    }

                    return LinearForwardWithBias(attn, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
                }
                finally
                {
                    attn.Dispose();
                }
            }
            finally
            {
                q.Dispose();
                k.Dispose();
                v.Dispose();
            }
        }

        /// <summary>
        /// Upload the block-order RoPE cos/sin tables once per image geometry so the
        /// per-block RoPE runs from device memory.
        /// </summary>
        private (Tensor Cos, Tensor Sin) GetOrCreateRopeDeviceTables(int numPatches, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            long key = ((long)numPatches << 32) | (uint)halfDim;
            if (!_ropeDeviceCache.TryGetValue(key, out var tables))
            {
                var cos = new Tensor(_allocator, DType.Float32, numPatches, halfDim);
                var sin = new Tensor(_allocator, DType.Float32, numPatches, halfDim);
                cos.SetElementsAsFloat(cosTable);
                sin.SetElementsAsFloat(sinTable);
                tables = (cos, sin);
                _ropeDeviceCache[key] = tables;
            }

            return tables;
        }

        /// <summary>
        /// Vision self-attention with fused QKV and RoPE.
        ///
        /// When on GPU (native attention), uses Narrow+NewContiguous to keep data on-device
        /// and avoid CPU<->GPU round-trips. When on CPU, uses a fused parallel
        /// Buffer.MemoryCopy pass to split Q/K/V in one sweep, eliminating three separate
        /// Narrow+NewContiguous allocations.
        /// </summary>
        private unsafe Tensor VisionSelfAttention(Tensor input, string prefix, int numPatches,
            int headDim, int halfDim, float[] cosTable, float[] sinTable)
        {
            using var qkv = LinearForwardWithBias(input, $"{prefix}.attn_qkv.weight", $"{prefix}.attn_qkv.bias");

            Tensor q, k, v;

            if (_cudaDirect)
                return CudaSelfAttention(qkv, prefix, numPatches, headDim, halfDim, cosTable, sinTable);

            if (_useNativeAttention)
            {
                // GPU path: split via tensor ops to keep data on device.
                using var qView = qkv.Narrow(1, 0, _hiddenSize);
                using var kView = qkv.Narrow(1, _hiddenSize, _hiddenSize);
                using var vView = qkv.Narrow(1, 2 * _hiddenSize, _hiddenSize);
                q = Ops.NewContiguous(qView);
                k = Ops.NewContiguous(kView);
                v = Ops.NewContiguous(vView);
            }
            else
            {
                // CPU path: fused parallel split avoids 3 separate alloc+copy passes.
                int hiddenSize = _hiddenSize;
                int tripleHidden = 3 * hiddenSize;
                q = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);
                k = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);
                v = new Tensor(_allocator, DType.Float32, numPatches, hiddenSize);

                float* qkvPtr = GetFloatPtr(qkv);
                float* qPtr = GetFloatPtr(q);
                float* kPtr = GetFloatPtr(k);
                float* vPtr = GetFloatPtr(v);
                long rowBytes = (long)hiddenSize * sizeof(float);

                Parallel.For(0, numPatches, p =>
                {
                    float* src = qkvPtr + (long)p * tripleHidden;
                    Buffer.MemoryCopy(src, qPtr + (long)p * hiddenSize, rowBytes, rowBytes);
                    Buffer.MemoryCopy(src + hiddenSize, kPtr + (long)p * hiddenSize, rowBytes, rowBytes);
                    Buffer.MemoryCopy(src + 2 * hiddenSize, vPtr + (long)p * hiddenSize, rowBytes, rowBytes);
                });
            }

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

            // For large images (high numPatches) the [numHeads, numPatches,
            // numPatches] scores tensor exceeds the GPU's per-command-buffer
            // working memory and trips OOM (~4 GB at numPatches=8000+). Chunk
            // the query dimension whenever scores would exceed
            // AttentionChunkBudgetBytes so peak working memory stays bounded.
            int chunkSize = ComputeAttentionChunkSize(numPatches);
            if (chunkSize >= numPatches)
            {
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

            // Chunked path: stream Q in chunks. attnFinal is allocated in
            // seq-major [numPatches, _hiddenSize] layout so each chunk's rows
            // map to a contiguous Narrow(0, qOff, qLen) slice that we can write
            // into directly without a second transpose at the end.
            var attnFinal = new Tensor(_allocator, DType.Float32, numPatches, _hiddenSize);
            try
            {
                for (int qOff = 0; qOff < numPatches; qOff += chunkSize)
                {
                    int qLen = Math.Min(chunkSize, numPatches - qOff);

                    using var qChunkView = qHeads.Narrow(1, qOff, qLen);
                    using var qChunkContig = Ops.NewContiguous(qChunkView);

                    using var scoresChunk = new Tensor(_allocator, DType.Float32, _numHeads, qLen, numPatches);
                    Ops.AddmmBatch(scoresChunk, 0, scoresChunk, scale, qChunkContig, kT);
                    Ops.Softmax(scoresChunk, scoresChunk);

                    using var outChunkHeadFirst = new Tensor(_allocator, DType.Float32, _numHeads, qLen, headDim);
                    Ops.AddmmBatch(outChunkHeadFirst, 0, outChunkHeadFirst, 1.0f, scoresChunk, vHeads);

                    using var outChunkSeqMajor = outChunkHeadFirst.Transpose(0, 1);
                    using var outChunkContig = Ops.NewContiguous(outChunkSeqMajor);
                    using var outChunkFlat = outChunkContig.View(qLen, _hiddenSize);

                    using var attnSlice = attnFinal.Narrow(0, qOff, qLen);
                    Ops.Copy(attnSlice, outChunkFlat);

                    // Drain each chunk's graph before queuing the next so the
                    // scoresChunk allocation cycles in/out of GPU memory rather
                    // than accumulating across chunks.
                    MlxFusedOps.TryAsyncEvaluate(attnFinal);
                }

                return LinearForwardWithBias(attnFinal, $"{prefix}.attn_out.weight", $"{prefix}.attn_out.bias");
            }
            finally
            {
                attnFinal.Dispose();
            }
        }

        // Bound the [numHeads, chunkSize, numPatches] scores intermediate to
        // ~AttentionChunkBudgetBytes so a single attention compute fits in the
        // Metal command-buffer working set on Apple Silicon. Returns numPatches
        // unchanged when the un-chunked path already fits — keeps the existing
        // fast path active for typical image sizes.
        private const long AttentionChunkBudgetBytes = 256L * 1024 * 1024;
        private int ComputeAttentionChunkSize(int numPatches)
        {
            long perRowBytes = (long)_numHeads * numPatches * sizeof(float);
            if (perRowBytes <= 0)
                return numPatches;
            long maxRows = AttentionChunkBudgetBytes / perRowBytes;
            if (maxRows >= numPatches)
                return numPatches;
            // Keep chunks large enough that matmul stays GEMM-efficient.
            return (int)Math.Max(64, Math.Min(numPatches, maxRows));
        }

        /// <summary>
        /// Apply NeoX-style RoPE with 2D spatial position encoding.
        /// data: [numPatches, numHeads * headDim], cosTable/sinTable: [numPatches * halfDim].
        ///
        /// Parallelized across patches (independent) with SIMD vectorization along the
        /// halfDim dimension. For a typical 768px image with 2304 patches, 16 heads, and
        /// halfDim=36, this processes ~1.3M element pairs. The parallel+SIMD path is ~8-12x
        /// faster than the scalar triple-nested loop on Apple M-series.
        /// </summary>
        private unsafe void ApplyVisionRoPE(Tensor data, int numPatches, int headDim, int halfDim,
            float[] cosTable, float[] sinTable)
        {
            float* ptr = GetFloatPtr(data);
            int numHeads = _numHeads;
            int vLen = Vector<float>.Count;

            fixed (float* cosPtr = cosTable, sinPtr = sinTable)
            {
                long ptrL = (long)ptr;
                long cosPtrL = (long)cosPtr;
                long sinPtrL = (long)sinPtr;

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

        private Tensor VisionMLP(Tensor input, string prefix)
        {
            using var fc1Out = LinearForwardWithBias(input, $"{prefix}.ffn_up.weight", $"{prefix}.ffn_up.bias");
            Ops.GELU(fc1Out, fc1Out);
            return LinearForwardWithBias(fc1Out, $"{prefix}.ffn_down.weight", $"{prefix}.ffn_down.bias");
        }

        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            // Derived from the transposed copy, not _weights[weightName]: on the
            // direct-CUDA path the untransposed original is released once the
            // transpose exists (see GetOrCreateTransposedWeight).
            Tensor weightT = GetOrCreateTransposedWeight(weightName);
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)weightT.Sizes[1];

            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);

            Tensor contiguousInput = input.IsContiguous() ? null : Ops.NewContiguous(input);
            Tensor src = contiguousInput ?? input;

            Ops.Addmm(result, 0, result, 1.0f, src, weightT);

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

        private static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        /// <summary>
        /// Transpose a 2D weight through host memory and upload the result.
        /// NewContiguous over a transposed view has no CUDA kernel (the inner
        /// stride is not 1), so it lands in the CUDA CPU fallback, whose
        /// CopyLogical walks every element through the scalar
        /// Get/SetElementAsFloat accessors. Across the 27 blocks' QKV / out / FFN
        /// weights that is ~390M scalar round trips — 38 s of first-encode latency.
        /// </summary>
        private unsafe Tensor HostTranspose2D(Tensor weight)
        {
            int rows = (int)weight.Sizes[0];
            int cols = (int)weight.Sizes[1];
            float[] src = weight.GetElementsAsFloat(rows * cols);
            var transposed = new Tensor(_allocator, DType.Float32, cols, rows);
            using (var staging = new HostStaging(transposed, true))
            {
                float* dst = staging.Ptr;
                fixed (float* srcPtr = src)
                {
                    long srcPtrL = (long)srcPtr;
                    long dstPtrL = (long)dst;
                    Parallel.For(0, cols, c =>
                    {
                        float* s = (float*)srcPtrL;
                        float* d = (float*)dstPtrL + (long)c * rows;
                        for (int r = 0; r < rows; r++)
                            d[r] = s[(long)r * cols + c];
                    });
                }
            }

            return transposed;
        }

        private unsafe Tensor GetOrCreateTransposedWeight(string weightName)
        {
            if (_transposedWeights.TryGetValue(weightName, out var transposed))
                return transposed;

            Tensor weight = _weights[weightName];
            if (_cudaDirect && weight.DimensionCount == 2)
            {
                transposed = HostTranspose2D(weight);
                _transposedWeights[weightName] = transposed;
                // Release the untransposed original: nothing on this path reads it
                // again (LinearForwardWithBias takes its output dim from the
                // transpose, and the GGML fused block paths are gated off). The
                // 27 blocks' QKV / out / FFN weights are ~1.6 GB of F32, and
                // keeping both copies resident next to a multi-GB language model
                // is what pushes a 16 GB card into WDDM shared-memory spillover —
                // where the encoder's ~350 MB of per-block activations run 6x
                // slower and each successive image degrades further.
                _weights.Remove(weightName);
                weight.Dispose();
                return transposed;
            }

            using var weightViewT = weight.Transpose();
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
            using var staging = new HostStaging(cached, _cudaDirect);
            float* dstPtr = staging.Ptr;

            float stepH = gridH > 1 ? (float)(_gridPerSide - 1) / (gridH - 1) : 0f;
            float stepW = gridW > 1 ? (float)(_gridPerSide - 1) / (gridW - 1) : 0f;
            int hiddenSize = _hiddenSize;
            int gridPerSide = _gridPerSide;
            int vLen = Vector<float>.Count;

            // Written in block order so the table lines up with the block-ordered
            // patch-embed rows (see GetOrCreateBlockOrder / PatchEmbed).
            int[] blockOrder = GetOrCreateBlockOrder(gridH, gridW);

            long posPtrL = (long)posPtr;
            long dstPtrL = (long)dstPtr;

            fixed (int* orderPtr = blockOrder)
            {
            long orderPtrL = (long)orderPtr;
            Parallel.For(0, gridH, brow =>
            {
                float* pos = (float*)posPtrL;
                float* dst = (float*)dstPtrL;
                int* order = (int*)orderPtrL;

                for (int bcol = 0; bcol < gridW; bcol++)
                {
                    int destIdx = brow * gridW + bcol;
                    int rasterIdx = order[destIdx];
                    int h = rasterIdx / gridW;
                    int w = rasterIdx - h * gridW;
                    float y = h * stepH;
                    float x = w * stepW;

                    int fy = (int)y;
                    int fx = (int)x;
                    int cy = Math.Min(fy + 1, gridPerSide - 1);
                    int cx = Math.Min(fx + 1, gridPerSide - 1);
                    float dy = y - fy;
                    float dx = x - fx;

                    float wt00 = (1 - dy) * (1 - dx);
                    float wt01 = (1 - dy) * dx;
                    float wt10 = dy * (1 - dx);
                    float wt11 = dy * dx;

                    int idx00 = fy * gridPerSide + fx;
                    int idx01 = fy * gridPerSide + cx;
                    int idx10 = cy * gridPerSide + fx;
                    int idx11 = cy * gridPerSide + cx;

                    float* dstRow = dst + (long)destIdx * hiddenSize;
                    float* p00 = pos + (long)idx00 * hiddenSize;
                    float* p01 = pos + (long)idx01 * hiddenSize;
                    float* p10 = pos + (long)idx10 * hiddenSize;
                    float* p11 = pos + (long)idx11 * hiddenSize;

                    var v00 = new Vector<float>(wt00);
                    var v01 = new Vector<float>(wt01);
                    var v10 = new Vector<float>(wt10);
                    var v11 = new Vector<float>(wt11);

                    int d = 0;
                    for (; d <= hiddenSize - vLen; d += vLen)
                    {
                        var a = TensorComputePrimitives.LoadVector(p00 + d);
                        var b = TensorComputePrimitives.LoadVector(p01 + d);
                        var c = TensorComputePrimitives.LoadVector(p10 + d);
                        var e = TensorComputePrimitives.LoadVector(p11 + d);
                        var r = a * v00 + b * v01 + c * v10 + e * v11;
                        TensorComputePrimitives.StoreVector(dstRow + d, r);
                    }
                    for (; d < hiddenSize; d++)
                        dstRow[d] = wt00 * p00[d] + wt01 * p01[d] + wt10 * p10[d] + wt11 * p11[d];
                }
            });
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

                    // Qwen3-VL vision M-RoPE uses a BLOCKED frequency layout: the
                    // first half of the rotary pairs (indices 0..numBands-1) encode
                    // the row (Y/H) position, the second half (numBands..halfDim-1)
                    // encode the column (X/W) position. This matches ggml's
                    // GGML_ROPE_TYPE_VISION (first sections[0] dims use p_t=row, the
                    // next use p_h=col) and HF/vLLM rot_pos_emb, where
                    // cos[pos_ids].flatten() concatenates the H-frequency block then
                    // the W-frequency block. An interleaved [Y0,X0,Y1,X1,...] layout
                    // rotates the wrong head dimensions and scrambles the spatial
                    // encoding (coarse gist survives, fine reading/OCR fails).
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
            foreach (var w in _positionEmbeddingCache.Values)
                w.Dispose();
            _positionEmbeddingCache.Clear();
            foreach (var w in _transposedWeights.Values)
                w.Dispose();
            _transposedWeights.Clear();
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();
            foreach (var (cos, sin) in _ropeDeviceCache.Values)
            {
                cos.Dispose();
                sin.Dispose();
            }
            _ropeDeviceCache.Clear();
            _ropeCache.Clear();
            _blockOrderCache.Clear();
        }
    }
}
