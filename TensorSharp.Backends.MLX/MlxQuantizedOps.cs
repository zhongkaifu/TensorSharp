using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;

namespace TensorSharp.MLX
{
    public static class MlxQuantizedOps
    {
        private const int Q8_0BlockElements = 32;
        private const int Q8_0BlockBytes = 34;
        private const int Q4BlockElements = 32;
        private const int Q4_0BlockBytes = 18;
        private const int Q4_1BlockBytes = 20;
        private const int MlxAffineQ4GroupSize = 32;
        private const int MlxAffineQ4Bits = 4;
        private const int Q5BlockElements = 32;
        private const int Q5_0BlockBytes = 22;
        private const int Q5_1BlockBytes = 24;
        private const int MlxAffineQ5GroupSize = 32;
        private const int MlxAffineQ5Bits = 5;
        private const int QK_K = 256;
        private const int KScaleSize = 12;
        private const int Q4_KBlockBytes = 4 + KScaleSize + QK_K / 2;
        private const int Q5_KBlockBytes = 4 + KScaleSize + QK_K / 8 + QK_K / 2;
        private const int Q6_KBlockBytes = QK_K / 2 + QK_K / 4 + QK_K / 16 + 2;
        private const int IQ2_XXSBlockBytes = 2 + QK_K / 8 * 2;
        private const int IQ2_SBlockBytes = 2 + QK_K / 4 + QK_K / 16;
        private const int IQ3_SBlockBytes = 2 + 13 * (QK_K / 32) + QK_K / 64;
        private const int IQ4_XSBlockBytes = 2 + 2 + QK_K / 64 + QK_K / 2;
        // IQ4_NL: 32 elements per block (QK4_NL), 18 bytes = 2-byte F16 d + 16-byte qs.
        private const int IQ4_NLBlockElements = 32;
        private const int IQ4_NLBlockBytes = 18;
        private const int MlxAffineQ8GroupSize = 32;
        private const int MlxAffineQ8Bits = 8;
        private const int MlxAffineQ6GroupSize = 16;
        private const int MlxAffineQ6Bits = 6;
        private const int Mxfp4BlockElements = 32;
        private const int Mxfp4BlockBytes = 1 + Mxfp4BlockElements / 2;
        private const int MlxMxfp4GroupSize = 32;
        private const int MlxMxfp4Bits = 4;
        private const string MlxAffineMode = "affine";
        private const string MlxMxfp4Mode = "mxfp4";

        private static readonly object Sync = new();
        private static readonly Dictionary<CacheKey, LinkedListNode<DeviceWeight>> Cache = new();
        // LRU ordering for offloadable (MoE expert) entries only. Pinned entries
        // sit in `Cache` but never participate in eviction and are never linked
        // into this list. Front of list = most recently used.
        private static readonly LinkedList<DeviceWeight> OffloadLru = new();
        private static long _offloadResidentBytes;

        private sealed class DeviceWeight : IDisposable
        {
            public MlxNative.MlxArray Weight;
            public MlxNative.MlxArray Scales;
            public MlxNative.MlxArray Biases;
            public int DeviceId;
            public int GgmlType;
            public long Ne0;
            public long Ne1;
            public long RawBytes;
            public int GroupSize;
            public int Bits;
            public string Mode = MlxAffineMode;
            public CacheKey Key;
            public IntPtr HostData;
            // True when the entry was created against an offloadable cache key
            // (a MoE expert weight). Offloadable entries participate in the LRU
            // and may be evicted; non-offloadable entries are pinned in the
            // cache forever.
            public bool Offloadable;

            public void Dispose()
            {
                MlxNative.FreeArray(Weight);
                MlxNative.FreeArray(Scales);
                MlxNative.FreeArray(Biases);
                Weight = default;
                Scales = default;
                Biases = default;
            }
        }

        // Formats that MLX should keep compressed while loading. Only the subset in
        // CanPreloadQuantizedType has a native MLX matmul representation today; the
        // remaining formats stay as GGUF file-backed views and use the row-dequant
        // fallback from ModelBase. This avoids expanding large IQ/UD models to full
        // Float32 residency during load.
        public static bool SupportsQuantizedType(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F16 => true,
                GgmlTensorType.BF16 => true,
                GgmlTensorType.F32 => true,
                GgmlTensorType.Q2_K => true,
                GgmlTensorType.Q3_K => true,
                GgmlTensorType.Q8_0 => true,
                GgmlTensorType.Q8_1 => true,
                GgmlTensorType.Q4_K => true,
                GgmlTensorType.Q5_K => true,
                GgmlTensorType.Q6_K => true,
                GgmlTensorType.Q8_K => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.IQ2_XXS => true,
                GgmlTensorType.IQ2_XS => true,
                GgmlTensorType.IQ3_XXS => true,
                GgmlTensorType.IQ1_S => true,
                GgmlTensorType.IQ4_NL => true,
                GgmlTensorType.IQ3_S => true,
                GgmlTensorType.IQ2_S => true,
                GgmlTensorType.IQ4_XS => true,
                GgmlTensorType.IQ1_M => true,
                GgmlTensorType.TQ1_0 => true,
                GgmlTensorType.TQ2_0 => true,
                GgmlTensorType.MXFP4 => true,
                _ => false,
            };
        }

        public static bool SupportsQuantizedType(int type)
        {
            return Enum.IsDefined(typeof(GgmlTensorType), type) && SupportsQuantizedType((GgmlTensorType)type);
        }

        // Opt-out for the MLX IQ4_NL preload + matmul path. Default ON. Set
        // TS_MLX_IQ4NL_GPU=0 to disable, which forces IQ4_NL weights to
        // stay file-backed (no MLX preload, no ReleaseHostData) so the C#
        // `ManagedQuantizedOps` CPU matmul path remains usable. Useful
        // when the workload is prefill-heavy (short generations): the
        // GPU kernel currently issues one threadgroup per (out_col, row)
        // and prefill regresses by ~40-50 % vs the CPU baseline on
        // Nemotron-H 30B / IQ2_XXS-UD, where the MoE expert weights are
        // IQ4_NL. Decode (rows=1, ~3× faster on GPU) is unaffected by
        // this trade since it's outside the regressed regime.
        private static bool Iq4NlGpuEnabled()
        {
            return !string.Equals(
                Environment.GetEnvironmentVariable("TS_MLX_IQ4NL_GPU"),
                "0", StringComparison.Ordinal);
        }

        public static bool CanPreloadQuantizedType(int type)
        {
            return type == (int)GgmlTensorType.Q4_0 ||
                type == (int)GgmlTensorType.Q4_1 ||
                type == (int)GgmlTensorType.Q4_K ||
                type == (int)GgmlTensorType.Q5_0 ||
                type == (int)GgmlTensorType.Q5_1 ||
                type == (int)GgmlTensorType.Q5_K ||
                type == (int)GgmlTensorType.Q6_K ||
                type == (int)GgmlTensorType.Q8_0 ||
                type == (int)GgmlTensorType.IQ2_XXS ||
                type == (int)GgmlTensorType.IQ2_S ||
                type == (int)GgmlTensorType.IQ3_S ||
                type == (int)GgmlTensorType.IQ4_XS ||
                (type == (int)GgmlTensorType.IQ4_NL && Iq4NlGpuEnabled()) ||
                type == (int)GgmlTensorType.MXFP4;
        }

        /// <summary>
        /// True when the MLX preload of this quant type allocates fresh MLX
        /// memory (repacks the GGUF bytes into an MLX-managed buffer), as
        /// opposed to a zero-copy wrap of the file-backed host pointer. The
        /// offload LRU only delivers a real memory-residency benefit for
        /// "repack" types — for "raw wrap" types the baseline preload path
        /// already runs <see cref="ReleaseHostData"/> with its madvise hint
        /// once at startup, and the OS evicts page-cache pages without
        /// further help. Wrapping those types in the LRU just adds churn.
        /// </summary>
        public static bool PreloadDuplicatesHostMemory(int ggmlType)
        {
            switch (ggmlType)
            {
                case (int)GgmlTensorType.Q4_0:
                case (int)GgmlTensorType.Q4_1:
                case (int)GgmlTensorType.Q5_0:
                case (int)GgmlTensorType.Q5_1:
                case (int)GgmlTensorType.Q8_0:
                case (int)GgmlTensorType.MXFP4:
                    return true;
                case (int)GgmlTensorType.Q5_K:
                    // Q5_K is repack-only when the raw kernel is disabled
                    // (TS_MLX_Q5K_RAW=0). Default is raw=true → zero-copy.
                    return !UseRawQ5KKernel();
                default:
                    return false;
            }
        }

        public static void PreloadQuantizedWeight(
            MlxAllocator allocator,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (allocator == null)
                throw new ArgumentNullException(nameof(allocator));
            if (!CanPreloadQuantizedType(ggmlType))
                return;

            EnsureWeight(allocator.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
        }

        // Returns true if a batched-MoE matmul kernel exists for this
        // ggml type. Currently only IQ2_XXS — extend as more custom
        // batched kernels are written.
        //
        // NOTE: An IQ4_NL batched-MoE kernel (`Iq4NlMoeMatmulBatched` /
        // `Iq4NlMoeMatmulBatchedRowed` in MlxNative.cs) is written but
        // currently SEGFAULTS Metal at runtime. The kernels assume a
        // stacked weight layout of `[numExperts, outDim, blocksPerRow ×
        // 18 bytes]` which matches what `_layerStackedUp[layer].Data`
        // exposes for IQ2_XXS Qwen 3.5, but may not be how Nemotron-H
        // stacks its `ffn_up_exps.weight` / `ffn_down_exps.weight` GGUF
        // tensors. Left disabled here so the kernel code is preserved
        // for future debugging — flip the IQ4_NL check on again once the
        // stacked layout is verified and the kernel either passes the
        // bounds check or is rewritten to match the actual GGUF layout.
        public static bool SupportsBatchedMoeMatmul(int ggmlType)
        {
            return ggmlType == (int)GgmlTensorType.IQ2_XXS;
        }

        // Batched IQ2_XXS MoE matmul. Replaces K separate per-expert
        // matmul dispatches with a single Metal kernel.
        //
        // - For gate / up projections: pass sharedInput=true. `input` is
        //   the [1, inDim] decode token; the kernel applies each expert's
        //   weight to the same row.
        // - For down projection: pass sharedInput=false. `input` is
        //   [K, inDim] (the per-expert SwiGLU output); row k uses expert
        //   expertIndices[k]'s weight.
        //
        // `result` must be a contiguous [K, outDim] Float32 MLX tensor.
        // `expertIndices` must be a contiguous [K] Int32 MLX tensor.
        /// <summary>
        /// Fused gate + up + SiLUMul for the MoE FFN decode path. Replaces
        /// {gate matmul, up matmul, SiLUMul} 3 kernels with one. Inputs are
        /// the shared decode-token row and two stacked weight tensors
        /// (gate_stack, up_stack), one per expert. Output is the post-SiLU
        /// product [K, intermediate], ready as input for the down matmul.
        /// </summary>
        public static bool TryMoeFusedGateUpSilu(
            Tensor result,
            Tensor input,
            Tensor expertIndices,
            IntPtr stackedGateCacheKey,
            IntPtr stackedGateHostData,
            long gateTotalRawBytes,
            IntPtr stackedUpCacheKey,
            IntPtr stackedUpHostData,
            long upTotalRawBytes,
            int ggmlType,
            long perExpertNe0,
            long perExpertNe1,
            int numExperts)
        {
            // Only IQ2_XXS implemented for now.
            if (ggmlType != (int)GgmlTensorType.IQ2_XXS) return false;
            if (result == null || input == null || expertIndices == null) return false;
            if (result.Storage is not MlxStorage resultStorage) return false;
            if (input.Storage is not MlxStorage inputStorage) return false;
            if (expertIndices.Storage is not MlxStorage indicesStorage) return false;
            if (result.ElementType != DType.Float32 || input.ElementType != DType.Float32
                || expertIndices.ElementType != DType.Int32)
                return false;
            if (result.DimensionCount != 2 || input.DimensionCount != 2 || expertIndices.DimensionCount != 1)
                return false;
            int K = (int)expertIndices.Sizes[0];
            int outDim = (int)perExpertNe1;
            int inDim = (int)perExpertNe0;
            if (result.Sizes[0] != K || result.Sizes[1] != outDim) return false;
            if (input.Sizes[0] != 1 || input.Sizes[1] != inDim) return false;
            if (inDim % 256 != 0) return false;

            DeviceWeight gate = EnsureWeight(
                resultStorage.DeviceId,
                stackedGateCacheKey,
                stackedGateHostData,
                ggmlType,
                perExpertNe0,
                numExperts * perExpertNe1,
                gateTotalRawBytes);
            DeviceWeight up = EnsureWeight(
                resultStorage.DeviceId,
                stackedUpCacheKey,
                stackedUpHostData,
                ggmlType,
                perExpertNe0,
                numExperts * perExpertNe1,
                upTotalRawBytes);

            MlxNative.MlxArray inputView = default;
            MlxNative.MlxArray indicesView = default;
            MlxNative.MlxArray output = default;
            try
            {
                inputView = inputStorage.CreateArrayView(input);
                indicesView = indicesStorage.CreateArrayView(expertIndices);
                output = MlxNative.Iq2XxsMoeMatmulBatchedFusedGateUpSilu(
                    inputView, gate.Weight, up.Weight, indicesView, K, inDim, outDim);
                SetDeviceResult(result, output);
                output = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(inputView);
                MlxNative.FreeArray(indicesView);
                MlxNative.FreeArray(output);
            }
        }

        public static bool TryMoeMatmulBatched(
            Tensor result,
            Tensor input,
            Tensor expertIndices,
            IntPtr stackedCacheKey,
            IntPtr stackedHostData,
            int ggmlType,
            long perExpertNe0,
            long perExpertNe1,
            int numExperts,
            long totalRawBytes,
            bool sharedInput)
        {
            if (!SupportsBatchedMoeMatmul(ggmlType)) return false;
            if (result == null || input == null || expertIndices == null) return false;
            if (result.Storage is not MlxStorage resultStorage) return false;
            if (input.Storage is not MlxStorage inputStorage) return false;
            if (expertIndices.Storage is not MlxStorage indicesStorage) return false;
            if (result.ElementType != DType.Float32 || input.ElementType != DType.Float32
                || expertIndices.ElementType != DType.Int32)
                return false;
            if (result.DimensionCount != 2 || input.DimensionCount != 2 || expertIndices.DimensionCount != 1)
                return false;
            int K = (int)expertIndices.Sizes[0];
            int outDim = (int)perExpertNe1;
            int inDim = (int)perExpertNe0;
            if (result.Sizes[0] != K || result.Sizes[1] != outDim) return false;
            int expectedInRows = sharedInput ? 1 : K;
            if (input.Sizes[0] != expectedInRows || input.Sizes[1] != inDim) return false;
            // Block-size alignment varies by quant type: IQ4_NL = 32-element
            // blocks (so inDim % 32 == 0), all others use 256-element blocks.
            int blockAlignment = ggmlType == (int)GgmlTensorType.IQ4_NL ? 32 : 256;
            if (inDim % blockAlignment != 0) return false;

            // Ensure the stacked weight is uploaded as ONE MLX array.
            // Cache key is the stacked-bytes host pointer (unique per
            // layer × kind). We reuse the existing per-(layer, kind)
            // DeviceWeight cache by passing the stacked dimensions:
            // ne0 = perExpertNe0, ne1 = numExperts * perExpertNe1, so
            // the cached MLX array is exactly the stacked uint8 buffer.
            DeviceWeight weight = EnsureWeight(
                resultStorage.DeviceId,
                stackedCacheKey,
                stackedHostData,
                ggmlType,
                perExpertNe0,
                numExperts * perExpertNe1,
                totalRawBytes);

            MlxNative.MlxArray inputView = default;
            MlxNative.MlxArray indicesView = default;
            MlxNative.MlxArray output = default;
            try
            {
                inputView = inputStorage.CreateArrayView(input);
                indicesView = indicesStorage.CreateArrayView(expertIndices);
                if (ggmlType == (int)GgmlTensorType.IQ4_NL)
                {
                    output = sharedInput
                        ? MlxNative.Iq4NlMoeMatmulBatched(inputView, weight.Weight, indicesView, K, inDim, outDim)
                        : MlxNative.Iq4NlMoeMatmulBatchedRowed(inputView, weight.Weight, indicesView, K, inDim, outDim);
                }
                else
                {
                    output = sharedInput
                        ? MlxNative.Iq2XxsMoeMatmulBatched(inputView, weight.Weight, indicesView, K, inDim, outDim)
                        : MlxNative.Iq2XxsMoeMatmulBatchedRowed(inputView, weight.Weight, indicesView, K, inDim, outDim);
                }
                SetDeviceResult(result, output);
                output = default;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                MlxNative.FreeArray(inputView);
                MlxNative.FreeArray(indicesView);
                MlxNative.FreeArray(output);
            }
        }

        public static void ReleaseQuantizedWeight(MlxAllocator allocator, IntPtr cacheKey)
        {
            if (allocator == null)
                return;
            ReleaseQuantizedWeight(allocator.DeviceId, cacheKey);
        }

        internal static void ReleaseQuantizedWeight(int deviceId, IntPtr cacheKey)
        {
            if (cacheKey == IntPtr.Zero)
                return;

            var key = new CacheKey(deviceId, cacheKey);
            lock (Sync)
            {
                if (!Cache.TryGetValue(key, out LinkedListNode<DeviceWeight> node))
                    return;

                EvictNodeLocked(node);
                Cache.Remove(key);
            }
        }

        internal static void ClearDeviceCache(int deviceId)
        {
            lock (Sync)
            {
                List<CacheKey> remove = new();
                foreach (var kv in Cache)
                {
                    if (kv.Key.DeviceId == deviceId)
                    {
                        EvictNodeLocked(kv.Value);
                        remove.Add(kv.Key);
                    }
                }

                foreach (CacheKey key in remove)
                    Cache.Remove(key);
            }
        }

        /// <summary>
        /// Frees the MLX arrays for the entry and unlinks it from the LRU list.
        /// For offloadable entries, also advises the OS that the underlying
        /// host-backed mmap pages can be reclaimed (matching the
        /// madvise(MADV_DONTNEED) that the baseline preload path applies via
        /// QuantizedWeight.ReleaseHostData → AdviseExternalViewCanBePagedOut).
        /// Caller is responsible for removing the cache dictionary entry.
        /// Must be called under <see cref="Sync"/>.
        /// </summary>
        private static void EvictNodeLocked(LinkedListNode<DeviceWeight> node)
        {
            DeviceWeight entry = node.Value;
            bool wasOffloadable = entry.Offloadable && node.List != null;
            if (wasOffloadable)
            {
                _offloadResidentBytes -= entry.RawBytes;
                OffloadLru.Remove(node);
            }
            entry.Dispose();
            if (wasOffloadable && entry.HostData != IntPtr.Zero)
                MoeExpertOffload.AdvisePagesNotNeeded(entry.HostData, entry.RawBytes);
        }

        private static bool UseRawQ5KKernel()
        {
            return !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_Q5K_RAW"), "0", StringComparison.Ordinal);
        }

        public static bool TryAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (!CanPreloadQuantizedType(ggmlType))
                return false;
            if (!TryValidateMatmul(result, input, ne0, ne1, out MlxStorage resultStorage, out MlxStorage inputStorage, out int rows))
                return false;

            DeviceWeight weight = EnsureWeight(resultStorage.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            return MlxWorker.Shared.Invoke(() => RunAddmmQuantizedToFloat32(
                result, input, weight, ggmlType, ne0, ne1, inputStorage, rows));
        }

        private static bool RunAddmmQuantizedToFloat32(
            Tensor result, Tensor input, DeviceWeight weight, int ggmlType,
            long ne0, long ne1, MlxStorage inputStorage, int rows)
        {
            MlxNative.MlxArray inputView = default;
            MlxNative.MlxArray output = default;
            MlxNative.MlxArray contiguous = default;
            try
            {
                inputView = inputStorage.CreateArrayView(input);
                if (ggmlType == (int)GgmlTensorType.IQ4_XS)
                {
                    output = MlxNative.Iq4XsMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ4_NL)
                {
                    // Per-call IQ4_NL matmul. Without this dispatch the call
                    // would fall through to MLX's generic QuantizedMatmul, but
                    // that wouldn't apply because IQ4_NL doesn't have an
                    // MLX-native repacked weight — we wrap the GGUF mmap as a
                    // raw uchar array and the Iq4NlMatmul Metal kernel
                    // dequantises on the fly. The kernel handles `rows >= 1`
                    // via the (256, outDim, rows) grid; multi-row prefill is
                    // not as well-tuned as decode (no per-row weight reuse
                    // yet) but always returns the correct result, which is
                    // what matters once `ReleaseHostData` has run and the
                    // CPU fallback is no longer reachable.
                    output = MlxNative.Iq4NlMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ2_XXS)
                {
                    output = MlxNative.Iq2XxsMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ2_S)
                {
                    output = MlxNative.Iq2SMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ3_S)
                {
                    output = MlxNative.Iq3SMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q4_K && string.Equals(weight.Mode, "q4_k", StringComparison.Ordinal))
                {
                    output = MlxNative.Q4KMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q5_K && string.Equals(weight.Mode, "q5_k", StringComparison.Ordinal))
                {
                    output = MlxNative.Q5KMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q6_K)
                {
                    output = MlxNative.Q6KMatmul(inputView, weight.Weight, rows, (int)ne0, (int)ne1);
                    SetDeviceResult(result, output);
                    output = default;
                }
                else
                {
                    // Phase 6c fast path: Q8_0 decode (rows == 1) gets a
                    // custom simdgroup-optimized kernel that's competitive
                    // with — sometimes faster than — mlx_quantized_matmul.
                    // Useful for the LM head, attention output projection,
                    // PLE matmuls, and any other Q8 matmul that doesn't
                    // already have a fused norm or add wrapper.
                    if (rows == 1
                        && ggmlType == (int)GgmlTensorType.Q8_0
                        && (int)ne0 % 32 == 0
                        && weight.Biases.IsValid
                        && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_Q8_MATMUL"), "0", StringComparison.Ordinal))
                    {
                        try
                        {
                            output = MlxNative.Q8Matmul(
                                inputView, weight.Weight, weight.Scales, weight.Biases,
                                (int)ne0, (int)ne1, (int)ne0 / 32);
                            SetDeviceResult(result, output);
                            output = default;
                            return true;
                        }
                        catch
                        {
                            MlxNative.FreeArray(output);
                            output = default;
                            // Fall through to MLX's built-in path.
                        }
                    }

                    output = MlxNative.QuantizedMatmul(
                        inputView,
                        weight.Weight,
                        weight.Scales,
                        weight.Biases,
                        transpose: true,
                        weight.GroupSize,
                        weight.Bits,
                        weight.Mode);
                    // mlx_quantized_matmul returns a row-major contiguous
                    // [rows, outDim] array for the standard 2D matmul case
                    // used by Forward/Decode. Skip the explicit
                    // mlx_contiguous call (saves ~84 ops/token on Gemma 4
                    // decode at ~5 quantized matmuls per layer × 42 layers).
                    // TS_MLX_QUANT_MATMUL_CONTIG=1 re-enables the defensive
                    // copy for backends/shapes where the output may be
                    // strided.
                    if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QUANT_MATMUL_CONTIG"), "1", StringComparison.Ordinal))
                    {
                        contiguous = MlxNative.Contiguous(output);
                        SetDeviceResult(result, contiguous);
                        contiguous = default;
                    }
                    else
                    {
                        SetDeviceResult(result, output);
                        output = default;
                    }
                }
                return true;
            }
            finally
            {
                MlxNative.FreeArray(inputView);
                MlxNative.FreeArray(output);
                MlxNative.FreeArray(contiguous);
            }
        }

        public static bool TryRmsNormAddmmQuantizedToFloat32(
            Tensor result,
            Tensor input,
            Tensor normWeight,
            float eps,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (!CanPreloadQuantizedType(ggmlType))
                return false;
            if (!TryValidateMatmul(result, input, ne0, ne1, out MlxStorage resultStorage, out MlxStorage inputStorage, out int rows))
                return false;
            if (normWeight?.Storage is not MlxStorage normStorage
                || normWeight.ElementType != DType.Float32
                || normWeight.DimensionCount != 1
                || normWeight.Sizes[0] != ne0)
            {
                return false;
            }

            DeviceWeight weight = EnsureWeight(resultStorage.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            // Batch the whole norm+matmul sub-graph in a single worker call.
            // Without this wrapper each of the ~5 MLX calls below pays a
            // ~25 µs queue round-trip; for Qwen3.5 decode (64 layers × this
            // path twice on hybrid attn+GDN models) that's ~16 ms / token
            // of pure C#-↔-worker synchronization overhead. With the wrapper
            // the sub-calls run inline (IsOnWorkerThread short-circuits the
            // queue), collapsing to a single hand-off.
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray normView = default;
                MlxNative.MlxArray normed = default;
                MlxNative.MlxArray output = default;
                MlxNative.MlxArray contiguous = default;
                MlxNative.MlxArray fused = default;
                try
                {
                    inputView = inputStorage.CreateArrayView(input);
                    normView = normStorage.CreateArrayView(normWeight);

                    // Phase 6 fast path: Q8_0 path (Gemma 4 / Q8_0 GGUFs)
                    // gets a single custom Metal kernel that does
                    // RMSNorm(input) + matmul together using simdgroup
                    // intrinsics. Saves one Metal dispatch per layer ×
                    // ~84 norm+matmul calls / token (QKV proj + gate_up).
                    if (rows == 1
                        && ggmlType == (int)GgmlTensorType.Q8_0
                        && (int)ne0 % 32 == 0
                        && weight.Biases.IsValid
                        && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_Q8_RMSNORM_MATMUL"), "0", StringComparison.Ordinal))
                    {
                        try
                        {
                            fused = MlxNative.Q8RmsNormMatmul(
                                inputView, normView, weight.Weight, weight.Scales, weight.Biases,
                                eps, (int)ne0, (int)ne1, (int)ne0 / 32);
                            SetDeviceResult(result, fused);
                            fused = default;
                            return true;
                        }
                        catch
                        {
                            MlxNative.FreeArray(fused);
                            fused = default;
                            // Fall through to legacy path.
                        }
                    }

                    normed = MlxNative.FastRmsNorm(inputView, normView, eps);
                    output = RunMatmul(normed, weight, ggmlType, rows, (int)ne0, (int)ne1);
                    if (MatmulOutputNeedsContiguous())
                    {
                        contiguous = MlxNative.Contiguous(output);
                        SetDeviceResult(result, contiguous);
                        contiguous = default;
                    }
                    else
                    {
                        SetDeviceResult(result, output);
                        output = default;
                    }
                    return true;
                }
                finally
                {
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(normView);
                    MlxNative.FreeArray(normed);
                    MlxNative.FreeArray(output);
                    MlxNative.FreeArray(contiguous);
                    MlxNative.FreeArray(fused);
                }
            });
        }

        public static bool TryAddmmQuantizedAddToFloat32(
            Tensor residual,
            Tensor input,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (!CanPreloadQuantizedType(ggmlType))
                return false;
            if (!TryValidateMatmul(residual, input, ne0, ne1, out MlxStorage residualStorage, out MlxStorage inputStorage, out int rows))
                return false;

            DeviceWeight weight = EnsureWeight(residualStorage.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            // Batch matmul + residual add in a single worker call so the
            // ~5 internal MLX calls collapse to one queue round-trip.
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray residualView = default;
                MlxNative.MlxArray matmul = default;
                MlxNative.MlxArray added = default;
                MlxNative.MlxArray fused = default;
                try
                {
                    inputView = inputStorage.CreateArrayView(input);
                    residualView = residualStorage.CreateArrayView(residual);

                    // Phase 6 fast path: Q8_0 (the Gemma 4 / Q8_0 GGUF
                    // path) gets a single custom Metal kernel that does
                    // matmul + residual add together. Saves one Metal
                    // dispatch per layer × 42 layers / token. Falls back
                    // to the legacy 2-op path on any unsupported shape.
                    if (rows == 1
                        && ggmlType == (int)GgmlTensorType.Q8_0
                        && (int)ne0 % 32 == 0
                        && weight.Biases.IsValid
                        && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_Q8_ADDMM_ADD"), "0", StringComparison.Ordinal))
                    {
                        try
                        {
                            fused = MlxNative.Q8AddmmAdd(
                                inputView, weight.Weight, weight.Scales, weight.Biases,
                                residualView, (int)ne0, (int)ne1, (int)ne0 / 32);
                            SetDeviceResult(residual, fused);
                            fused = default;
                            return true;
                        }
                        catch
                        {
                            MlxNative.FreeArray(fused);
                            fused = default;
                            // Fall through to legacy path.
                        }
                    }

                    matmul = RunMatmul(inputView, weight, ggmlType, rows, (int)ne0, (int)ne1);
                    added = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, residualView, matmul);
                    SetDeviceResult(residual, added);
                    added = default;
                    return true;
                }
                finally
                {
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(residualView);
                    MlxNative.FreeArray(matmul);
                    MlxNative.FreeArray(added);
                    MlxNative.FreeArray(fused);
                }
            });
        }

        // Lazily-compiled closure for the Gemma 4 dense decode FFN (and any
        // model that happens to share the same shape: input [1, hidden],
        // gate_up Q8_0 weight [hidden, 2·intermediate], gelu-mul split,
        // down Q8_0 weight [intermediate, hidden], residual add). The
        // closure captures the {eps, mode, groupSize, bits, halfDim} tuple
        // — see <see cref="FusedFFNCacheKey"/>. A different model with
        // different intermediate size or quant params would just compile
        // its own slot. shapeless=true so MLX specializes per shape
        // internally.
        private sealed class FusedFFNClosureSlot
        {
            public MlxNative.CompiledClosure Closure;
        }
        private static readonly Dictionary<FusedFFNCacheKey, FusedFFNClosureSlot> s_FusedFFNClosures = new();

        private readonly struct FusedFFNCacheKey : IEquatable<FusedFFNCacheKey>
        {
            public readonly int GgmlType;
            public readonly int HalfDim;
            public readonly int GroupSize;
            public readonly int Bits;
            public readonly string Mode;
            public readonly int EpsBits;

            public FusedFFNCacheKey(int ggmlType, int halfDim, int groupSize, int bits, string mode, float eps)
            {
                GgmlType = ggmlType;
                HalfDim = halfDim;
                GroupSize = groupSize;
                Bits = bits;
                Mode = mode ?? "affine";
                EpsBits = BitConverter.SingleToInt32Bits(eps);
            }

            public bool Equals(FusedFFNCacheKey other) =>
                GgmlType == other.GgmlType && HalfDim == other.HalfDim
                && GroupSize == other.GroupSize && Bits == other.Bits
                && string.Equals(Mode, other.Mode, StringComparison.Ordinal)
                && EpsBits == other.EpsBits;

            public override bool Equals(object obj) => obj is FusedFFNCacheKey o && Equals(o);
            public override int GetHashCode() => HashCode.Combine(GgmlType, HalfDim, GroupSize, Bits, Mode, EpsBits);
        }

        /// <summary>
        /// Fused decode-step FFN via <c>mlx_compile</c>. Matches Gemma 4's
        /// dense MLP block shape:
        /// <code>
        ///   normed_pre   = rmsnorm(hidden, ffn_norm_w, eps)
        ///   gate_up      = normed_pre @ gate_up_w
        ///   activated    = gelu(gate_up[:, :half]) * gate_up[:, half:]
        ///   down_out     = activated @ down_w
        ///   normed_post  = rmsnorm(down_out, post_ffn_norm_w, eps)
        ///   result       = residual + normed_post
        /// </code>
        /// Replaces four separate <c>MlxWorker.Invoke</c> dispatches
        /// (norm+gate_up matmul + GeluMulSplit + down matmul + post-norm-add)
        /// with one <c>mlx_closure_apply</c> so MLX pays the per-op
        /// graph-build cost only once at compile time.
        ///
        /// Scoped to Q8_0 weights so the apply-time argument list stays a
        /// fixed shape (six matmul-component arrays + three norm/IO arrays).
        /// Returns false for any other quant type so the caller falls back
        /// to the existing per-op path.
        /// </summary>
        public static bool TryFusedGemma4DenseFFNDecode(
            Tensor residual,
            Tensor hidden,
            Tensor preNormWeight, float eps,
            IntPtr gateUpCacheKey, IntPtr gateUpHostData, int gateUpType, long gateUpNe0, long gateUpNe1, long gateUpBytes,
            IntPtr downCacheKey, IntPtr downHostData, int downType, long downNe0, long downNe1, long downBytes,
            Tensor postNormWeight,
            int halfDim)
        {
            if (gateUpType != (int)GgmlTensorType.Q8_0 || downType != (int)GgmlTensorType.Q8_0)
                return false;
            if (!CanPreloadQuantizedType(gateUpType) || !CanPreloadQuantizedType(downType))
                return false;
            if (halfDim <= 0) return false;

            if (residual == null || hidden == null || preNormWeight == null || postNormWeight == null) return false;
            if (residual.Storage is not MlxStorage residualStorage) return false;
            if (hidden.Storage is not MlxStorage hiddenStorage) return false;
            if (preNormWeight.Storage is not MlxStorage preNormStorage) return false;
            if (postNormWeight.Storage is not MlxStorage postNormStorage) return false;
            if (residual.ElementType != DType.Float32 || hidden.ElementType != DType.Float32
                || preNormWeight.ElementType != DType.Float32 || postNormWeight.ElementType != DType.Float32)
                return false;
            if (hidden.DimensionCount != 2 || hidden.Sizes[0] != 1 || hidden.Sizes[1] != gateUpNe0)
                return false;
            if (residual.DimensionCount != 2 || residual.Sizes[0] != 1 || residual.Sizes[1] != gateUpNe0)
                return false;
            if (preNormWeight.DimensionCount != 1 || preNormWeight.Sizes[0] != gateUpNe0)
                return false;
            if (postNormWeight.DimensionCount != 1 || postNormWeight.Sizes[0] != gateUpNe0)
                return false;
            if (gateUpNe1 != 2L * halfDim || downNe0 != halfDim || downNe1 != gateUpNe0)
                return false;

            DeviceWeight gateUpDw = EnsureWeight(residualStorage.DeviceId,
                gateUpCacheKey, gateUpHostData, gateUpType, gateUpNe0, gateUpNe1, gateUpBytes);
            DeviceWeight downDw = EnsureWeight(residualStorage.DeviceId,
                downCacheKey, downHostData, downType, downNe0, downNe1, downBytes);

            var key = new FusedFFNCacheKey(gateUpType, halfDim, gateUpDw.GroupSize, gateUpDw.Bits, gateUpDw.Mode, eps);
            MlxNative.CompiledClosure closure = EnsureFusedFFNClosure(key);

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray hiddenView = default;
                MlxNative.MlxArray preNormView = default;
                MlxNative.MlxArray postNormView = default;
                MlxNative.MlxArray residualView = default;
                MlxNative.MlxArray output = default;
                MlxNative.MlxArray[] outputs = null;
                try
                {
                    hiddenView = hiddenStorage.CreateArrayView(hidden);
                    preNormView = preNormStorage.CreateArrayView(preNormWeight);
                    postNormView = postNormStorage.CreateArrayView(postNormWeight);
                    residualView = residualStorage.CreateArrayView(residual);

                    outputs = MlxNative.ApplyClosure(closure, new[]
                    {
                        hiddenView,
                        preNormView,
                        gateUpDw.Weight,
                        gateUpDw.Scales,
                        gateUpDw.Biases,
                        downDw.Weight,
                        downDw.Scales,
                        downDw.Biases,
                        postNormView,
                        residualView,
                    });

                    if (outputs == null || outputs.Length == 0) return false;
                    output = outputs[0];
                    outputs[0] = default;
                    SetDeviceResult(residual, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(hiddenView);
                    MlxNative.FreeArray(preNormView);
                    MlxNative.FreeArray(postNormView);
                    MlxNative.FreeArray(residualView);
                    MlxNative.FreeArray(output);
                    if (outputs != null)
                    {
                        for (int i = 0; i < outputs.Length; i++)
                            MlxNative.FreeArray(outputs[i]);
                    }
                }
            });
        }

        private static MlxNative.CompiledClosure EnsureFusedFFNClosure(FusedFFNCacheKey key)
        {
            lock (s_FusedFFNClosures)
            {
                if (s_FusedFFNClosures.TryGetValue(key, out var existing) && existing.Closure != null)
                    return existing.Closure;
            }

            // Capture key fields as locals so the closure body uses them as
            // compile-time constants rather than going through hash lookups.
            int halfDim = key.HalfDim;
            int groupSize = key.GroupSize;
            int bits = key.Bits;
            string mode = key.Mode;
            float eps = BitConverter.Int32BitsToSingle(key.EpsBits);

            var closure = MlxNative.NewClosure(inputs =>
            {
                // Apply-time arg layout: hidden, pre_norm_w,
                // gate_up_{weight,scales,biases}, down_{...},
                // post_norm_w, residual.
                MlxNative.MlxArray hidden = inputs[0];
                MlxNative.MlxArray preNormW = inputs[1];
                MlxNative.MlxArray gateUpW = inputs[2];
                MlxNative.MlxArray gateUpS = inputs[3];
                MlxNative.MlxArray gateUpB = inputs[4];
                MlxNative.MlxArray downW = inputs[5];
                MlxNative.MlxArray downS = inputs[6];
                MlxNative.MlxArray downB = inputs[7];
                MlxNative.MlxArray postNormW = inputs[8];
                MlxNative.MlxArray residual = inputs[9];

                MlxNative.MlxArray normedPre = default;
                MlxNative.MlxArray gateUp = default;
                MlxNative.MlxArray activated = default;
                MlxNative.MlxArray downOut = default;
                MlxNative.MlxArray normedPost = default;
                try
                {
                    normedPre = MlxNative.FastRmsNorm(hidden, preNormW, eps);
                    gateUp = MlxNative.QuantizedMatmul(
                        normedPre, gateUpW, gateUpS, gateUpB,
                        transpose: true, groupSize, bits, mode);
                    activated = MlxNative.GeluMulSplit(gateUp, rows: 1, halfDim: halfDim);
                    downOut = MlxNative.QuantizedMatmul(
                        activated, downW, downS, downB,
                        transpose: true, groupSize, bits, mode);
                    normedPost = MlxNative.FastRmsNorm(downOut, postNormW, eps);
                    MlxNative.MlxArray result = MlxNative.Binary(
                        MlxNative.MlxBinaryOp.Add, residual, normedPost);
                    return new[] { result };
                }
                finally
                {
                    MlxNative.FreeArray(normedPre);
                    MlxNative.FreeArray(gateUp);
                    MlxNative.FreeArray(activated);
                    MlxNative.FreeArray(downOut);
                    MlxNative.FreeArray(normedPost);
                }
            }, shapeless: true);

            lock (s_FusedFFNClosures)
            {
                if (s_FusedFFNClosures.TryGetValue(key, out var existing) && existing.Closure != null)
                {
                    // Someone else compiled the same closure while we were
                    // tracing. Free ours and use theirs.
                    MlxNative.FreeCompiledClosure(closure);
                    return existing.Closure;
                }
                s_FusedFFNClosures[key] = new FusedFFNClosureSlot { Closure = closure };
                return closure;
            }
        }

        /// <summary>
        /// Phase 6h fused decode-step Q8 matmul + GELU-tanh + per-element
        /// multiply. Used by Gemma 4's PLE inp_gate stage:
        /// <c>result = gelu(input @ weight) * gate</c>. Replaces a (Q8
        /// matmul + Ops.GELUMul) pair with one custom Metal dispatch.
        /// </summary>
        public static bool TryFusedQ8MatmulGeluMul(
            Tensor result, Tensor input, Tensor gate,
            IntPtr cacheKey, IntPtr hostData, int ggmlType,
            long ne0, long ne1, long rawBytes)
        {
            if (ggmlType != (int)GgmlTensorType.Q8_0) return false;
            if (!CanPreloadQuantizedType(ggmlType)) return false;
            if (!TryValidateMatmul(result, input, ne0, ne1, out MlxStorage resultStorage, out MlxStorage inputStorage, out int rows))
                return false;
            if (rows != 1) return false;
            if (gate == null || gate.Storage is not MlxStorage gateStorage) return false;
            if (gate.ElementType != DType.Float32) return false;
            if (gate.ElementCount() != ne1) return false;
            if ((int)ne0 % 32 != 0) return false;

            DeviceWeight weight = EnsureWeight(resultStorage.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            if (!weight.Biases.IsValid) return false;

            return MlxWorker.Shared.Invoke(() =>
            {
                MlxNative.MlxArray inputView = default;
                MlxNative.MlxArray gateView = default;
                MlxNative.MlxArray output = default;
                try
                {
                    inputView = inputStorage.CreateArrayView(input);
                    gateView = gateStorage.CreateArrayView(gate);
                    output = MlxNative.Q8MatmulGeluMul(
                        inputView, weight.Weight, weight.Scales, weight.Biases,
                        gateView, (int)ne0, (int)ne1, (int)ne0 / 32);
                    SetDeviceResult(result, output);
                    output = default;
                    return true;
                }
                catch
                {
                    return false;
                }
                finally
                {
                    MlxNative.FreeArray(inputView);
                    MlxNative.FreeArray(gateView);
                    MlxNative.FreeArray(output);
                }
            });
        }

        public static bool TryGetRowsQuantizedToFloat32(
            Tensor result,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            Tensor indices)
        {
            if (!CanPreloadQuantizedType(ggmlType))
                return false;
            if (!TryValidateGetRows(result, indices, ne0, ne1, out MlxStorage resultStorage, out MlxStorage indicesStorage))
                return false;

            DeviceWeight weight = EnsureWeight(resultStorage.DeviceId, cacheKey, hostData, ggmlType, ne0, ne1, rawBytes);
            MlxNative.MlxArray indicesView = default;
            MlxNative.MlxArray selectedWeight = default;
            MlxNative.MlxArray selectedScales = default;
            MlxNative.MlxArray selectedBiases = default;
            MlxNative.MlxArray dequantized = default;
            MlxNative.MlxArray contiguous = default;
            try
            {
                indicesView = indicesStorage.CreateArrayView(indices);
                if (ggmlType == (int)GgmlTensorType.IQ4_XS)
                {
                    dequantized = MlxNative.Iq4XsGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ2_XXS)
                {
                    dequantized = MlxNative.Iq2XxsGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ2_S)
                {
                    dequantized = MlxNative.Iq2SGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.IQ3_S)
                {
                    dequantized = MlxNative.Iq3SGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q4_K && string.Equals(weight.Mode, "q4_k", StringComparison.Ordinal))
                {
                    dequantized = MlxNative.Q4KGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q5_K && string.Equals(weight.Mode, "q5_k", StringComparison.Ordinal))
                {
                    dequantized = MlxNative.Q5KGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else if (ggmlType == (int)GgmlTensorType.Q6_K)
                {
                    dequantized = MlxNative.Q6KGetRows(weight.Weight, indicesView, (int)indices.Sizes[0], (int)ne0);
                    SetDeviceResult(result, dequantized);
                    dequantized = default;
                }
                else
                {
                    selectedWeight = MlxNative.TakeAxis(weight.Weight, indicesView, 0);
                    selectedScales = MlxNative.TakeAxis(weight.Scales, indicesView, 0);
                    selectedBiases = weight.Biases.IsValid
                        ? MlxNative.TakeAxis(weight.Biases, indicesView, 0)
                        : default;
                    dequantized = MlxNative.Dequantize(
                        selectedWeight,
                        selectedScales,
                        selectedBiases,
                        weight.GroupSize,
                        weight.Bits,
                        weight.Mode,
                        DType.Float32);
                    contiguous = MlxNative.Contiguous(dequantized);
                    SetDeviceResult(result, contiguous);
                    contiguous = default;
                }
                return true;
            }
            finally
            {
                MlxNative.FreeArray(indicesView);
                MlxNative.FreeArray(selectedWeight);
                MlxNative.FreeArray(selectedScales);
                MlxNative.FreeArray(selectedBiases);
                MlxNative.FreeArray(dequantized);
                MlxNative.FreeArray(contiguous);
            }
        }

        private static bool TryValidateMatmul(
            Tensor result,
            Tensor input,
            long ne0,
            long ne1,
            out MlxStorage resultStorage,
            out MlxStorage inputStorage,
            out int rows)
        {
            resultStorage = null;
            inputStorage = null;
            rows = 0;

            if (result == null || input == null)
                return false;
            if (result.Storage is not MlxStorage rs || input.Storage is not MlxStorage ins)
                return false;
            if (result.ElementType != DType.Float32 || input.ElementType != DType.Float32)
                return false;
            if (!result.IsContiguous() || !input.IsContiguous())
                return false;
            if (result.DimensionCount != 2 || input.DimensionCount != 2)
                return false;
            if (ne0 > int.MaxValue || ne1 > int.MaxValue || input.Sizes[0] > int.MaxValue)
                return false;
            if (input.Sizes[1] != ne0 || result.Sizes[0] != input.Sizes[0] || result.Sizes[1] != ne1)
                return false;

            resultStorage = rs;
            inputStorage = ins;
            rows = (int)input.Sizes[0];
            return rows >= 0;
        }

        private static MlxNative.MlxArray RunMatmul(
            MlxNative.MlxArray input,
            DeviceWeight weight,
            int ggmlType,
            int rows,
            int inDim,
            int outDim)
        {
            if (ggmlType == (int)GgmlTensorType.IQ4_XS)
                return MlxNative.Iq4XsMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.IQ2_XXS)
                return MlxNative.Iq2XxsMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.IQ2_S)
                return MlxNative.Iq2SMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.IQ3_S)
                return MlxNative.Iq3SMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.Q4_K && string.Equals(weight.Mode, "q4_k", StringComparison.Ordinal))
                return MlxNative.Q4KMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.Q5_K && string.Equals(weight.Mode, "q5_k", StringComparison.Ordinal))
                return MlxNative.Q5KMatmul(input, weight.Weight, rows, inDim, outDim);
            if (ggmlType == (int)GgmlTensorType.Q6_K)
                return MlxNative.Q6KMatmul(input, weight.Weight, rows, inDim, outDim);

            return MlxNative.QuantizedMatmul(
                input,
                weight.Weight,
                weight.Scales,
                weight.Biases,
                transpose: true,
                weight.GroupSize,
                weight.Bits,
                weight.Mode);
        }

        private static bool MatmulOutputNeedsContiguous()
        {
            // mlx_quantized_matmul produces a row-major contiguous output for
            // the standard 2D matmul case (input is [rows, inDim], weight is
            // [outDim, inDim] with transpose=true → output [rows, outDim]) —
            // so the explicit mlx_contiguous call after the matmul is
            // redundant for ALL types that go through it. The previously-
            // listed "custom-kernel" types (IQ4_XS, Q4_K with q4_k mode, ...)
            // were already known to skip it; the generic fallback path now
            // skips it too unless TS_MLX_QUANT_MATMUL_CONTIG=1 forces the
            // defensive copy back on (kept for cases where a future MLX
            // version changes the output layout for some quantization mode).
            if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QUANT_MATMUL_CONTIG"), "1", StringComparison.Ordinal))
                return true;
            return false;
        }

        private static bool TryValidateGetRows(
            Tensor result,
            Tensor indices,
            long ne0,
            long ne1,
            out MlxStorage resultStorage,
            out MlxStorage indicesStorage)
        {
            resultStorage = null;
            indicesStorage = null;

            if (result == null || indices == null)
                return false;
            if (result.Storage is not MlxStorage rs || indices.Storage is not MlxStorage ix)
                return false;
            if (result.ElementType != DType.Float32 || indices.ElementType != DType.Int32)
                return false;
            if (!result.IsContiguous() || !indices.IsContiguous())
                return false;
            if (result.DimensionCount != 2 || indices.DimensionCount != 1)
                return false;
            if (ne0 > int.MaxValue || ne1 > int.MaxValue)
                return false;
            if (result.Sizes[1] != ne0 || result.Sizes[0] != indices.Sizes[0])
                return false;

            resultStorage = rs;
            indicesStorage = ix;
            return true;
        }

        private static DeviceWeight EnsureWeight(
            int deviceId,
            IntPtr cacheKey,
            IntPtr hostData,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes)
        {
            if (cacheKey == IntPtr.Zero)
                cacheKey = hostData;
            if (cacheKey == IntPtr.Zero)
                throw new ArgumentException("MLX quantized weight cache key cannot be zero.", nameof(cacheKey));

            var key = new CacheKey(deviceId, cacheKey);
            lock (Sync)
            {
                if (Cache.TryGetValue(key, out LinkedListNode<DeviceWeight> existing))
                {
                    if (existing.Value.Offloadable && existing.List != null)
                    {
                        // Touch LRU — move to front so this entry isn't the next
                        // eviction candidate.
                        OffloadLru.Remove(existing);
                        OffloadLru.AddFirst(existing);
                    }
                    return existing.Value;
                }

                if (hostData == IntPtr.Zero)
                    throw new InvalidOperationException("Quantized weight is not preloaded on this MLX device and no host data was provided.");

                bool offloadable = MoeExpertOffload.IsEnabled && MoeExpertOffload.IsOffloadable(cacheKey);

                // Make room before uploading if this is an offloadable entry
                // and the LRU is already over budget. Eviction frees the old
                // MLX arrays via the FIFO-ordered worker, so any kernel still
                // using them completes before the free executes.
                if (offloadable && MoeExpertOffload.MaxCacheBytes > 0)
                {
                    long limit = MoeExpertOffload.MaxCacheBytes;
                    while (_offloadResidentBytes + rawBytes > limit && OffloadLru.Last != null)
                    {
                        LinkedListNode<DeviceWeight> victim = OffloadLru.Last;
                        Cache.Remove(victim.Value.Key);
                        EvictNodeLocked(victim);
                    }
                }

                DeviceWeight entry = ggmlType switch
                {
                    (int)GgmlTensorType.Q4_0 => CreateQ4Weight(deviceId, hostData, ne0, ne1, rawBytes, hasExplicitBias: false),
                    (int)GgmlTensorType.Q4_1 => CreateQ4Weight(deviceId, hostData, ne0, ne1, rawBytes, hasExplicitBias: true),
                    (int)GgmlTensorType.Q4_K => CreateQ4KRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.Q5_0 => CreateQ5Weight(deviceId, hostData, ne0, ne1, rawBytes, hasExplicitBias: false),
                    (int)GgmlTensorType.Q5_1 => CreateQ5Weight(deviceId, hostData, ne0, ne1, rawBytes, hasExplicitBias: true),
                    (int)GgmlTensorType.Q5_K => UseRawQ5KKernel()
                        ? CreateQ5KRawWeight(deviceId, hostData, ne0, ne1, rawBytes)
                        : CreateQ5KWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.Q6_K => CreateQ6KRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.Q8_0 => CreateQ8Weight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.IQ2_XXS => CreateIq2XxsRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.IQ2_S => CreateIq2SRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.IQ3_S => CreateIq3SRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.IQ4_XS => CreateIq4XsRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.IQ4_NL => CreateIq4NlRawWeight(deviceId, hostData, ne0, ne1, rawBytes),
                    (int)GgmlTensorType.MXFP4 => CreateMxfp4Weight(deviceId, hostData, ne0, ne1, rawBytes),
                    _ => throw new NotSupportedException($"MLX quantized preload does not support GGML tensor type {(GgmlTensorType)ggmlType}."),
                };
                entry.Key = key;
                entry.HostData = hostData;
                entry.Offloadable = offloadable;

                LinkedListNode<DeviceWeight> node = new LinkedListNode<DeviceWeight>(entry);
                if (offloadable)
                {
                    OffloadLru.AddFirst(node);
                    _offloadResidentBytes += rawBytes;
                }
                Cache.Add(key, node);
                return entry;
            }
        }

        private static unsafe DeviceWeight CreateQ4Weight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes, bool hasExplicitBias)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % Q4BlockElements != 0)
                throw new NotSupportedException($"Q4 MLX preload requires positive dimensions and input dim aligned to {Q4BlockElements}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int blocksPerRow = inDim / Q4BlockElements;
            int blockBytes = hasExplicitBias ? Q4_1BlockBytes : Q4_0BlockBytes;
            long expectedBytes = (long)outDim * blocksPerRow * blockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q4 raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            int packedWeightBytes = checked(outDim * (inDim / 2));
            int scaleCount = checked(outDim * blocksPerRow);
            byte[] packedWeights = new byte[packedWeightBytes];
            System.Half[] scales = new System.Half[scaleCount];
            System.Half[] biases = new System.Half[scaleCount];

            byte* src = (byte*)hostData.ToPointer();
            for (int row = 0; row < outDim; row++)
            {
                for (int block = 0; block < blocksPerRow; block++)
                {
                    long blockIndex = (long)row * blocksPerRow + block;
                    byte* blockPtr = src + blockIndex * blockBytes;
                    ushort scaleBits = (ushort)(blockPtr[0] | (blockPtr[1] << 8));
                    System.Half scale = BitConverter.UInt16BitsToHalf(scaleBits);
                    scales[blockIndex] = scale;
                    biases[blockIndex] = hasExplicitBias
                        ? BitConverter.UInt16BitsToHalf((ushort)(blockPtr[2] | (blockPtr[3] << 8)))
                        : (System.Half)(-8.0f * (float)scale);

                    int quantOffset = hasExplicitBias ? 4 : 2;
                    long dstOffset = (long)row * (inDim / 2) + block * (Q4BlockElements / 2);
                    PackQ4Block(blockPtr + quantOffset, packedWeights, dstOffset);
                }
            }

            return CreateDeviceWeight(
                deviceId,
                hasExplicitBias ? (int)GgmlTensorType.Q4_1 : (int)GgmlTensorType.Q4_0,
                ne0,
                ne1,
                rawBytes,
                packedWeights,
                scales,
                biases,
                new[] { outDim, inDim / 8 },
                new[] { outDim, blocksPerRow },
                MlxAffineQ4GroupSize,
                MlxAffineQ4Bits);
        }

        private static unsafe DeviceWeight CreateQ5Weight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes, bool hasExplicitBias)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % Q5BlockElements != 0)
                throw new NotSupportedException($"Q5 MLX preload requires positive dimensions and input dim aligned to {Q5BlockElements}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int blocksPerRow = inDim / Q5BlockElements;
            int blockBytes = hasExplicitBias ? Q5_1BlockBytes : Q5_0BlockBytes;
            long expectedBytes = (long)outDim * blocksPerRow * blockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q5 raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            int packedWeightBytes = checked(outDim * blocksPerRow * 20);
            int scaleCount = checked(outDim * blocksPerRow);
            byte[] packedWeights = new byte[packedWeightBytes];
            System.Half[] scales = new System.Half[scaleCount];
            System.Half[] biases = new System.Half[scaleCount];
            byte[] values = new byte[Q5BlockElements];

            byte* src = (byte*)hostData.ToPointer();
            for (int row = 0; row < outDim; row++)
            {
                for (int block = 0; block < blocksPerRow; block++)
                {
                    long blockIndex = (long)row * blocksPerRow + block;
                    byte* blockPtr = src + blockIndex * blockBytes;
                    ushort scaleBits = (ushort)(blockPtr[0] | (blockPtr[1] << 8));
                    System.Half scale = BitConverter.UInt16BitsToHalf(scaleBits);
                    scales[blockIndex] = scale;
                    biases[blockIndex] = hasExplicitBias
                        ? BitConverter.UInt16BitsToHalf((ushort)(blockPtr[2] | (blockPtr[3] << 8)))
                        : (System.Half)(-16.0f * (float)scale);

                    int highBitOffset = hasExplicitBias ? 4 : 2;
                    int quantOffset = hasExplicitBias ? 8 : 6;
                    uint highBits =
                        (uint)blockPtr[highBitOffset] |
                        ((uint)blockPtr[highBitOffset + 1] << 8) |
                        ((uint)blockPtr[highBitOffset + 2] << 16) |
                        ((uint)blockPtr[highBitOffset + 3] << 24);
                    for (int j = 0; j < Q5BlockElements / 2; j++)
                    {
                        byte packed = blockPtr[quantOffset + j];
                        values[j] = (byte)((uint)(packed & 0x0F) | (((highBits >> j) & 1u) << 4));
                        values[j + Q5BlockElements / 2] = (byte)((uint)((packed >> 4) & 0x0F) | (((highBits >> (j + 16)) & 1u) << 4));
                    }

                    long dstOffset = (long)row * blocksPerRow * 20 + block * 20;
                    PackUnsignedBits(values, MlxAffineQ5Bits, packedWeights, dstOffset);
                }
            }

            return CreateDeviceWeight(
                deviceId,
                hasExplicitBias ? (int)GgmlTensorType.Q5_1 : (int)GgmlTensorType.Q5_0,
                ne0,
                ne1,
                rawBytes,
                packedWeights,
                scales,
                biases,
                new[] { outDim, blocksPerRow * 5 },
                new[] { outDim, blocksPerRow },
                MlxAffineQ5GroupSize,
                MlxAffineQ5Bits);
        }

        private static unsafe DeviceWeight CreateQ4KWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % QK_K != 0)
                throw new NotSupportedException($"Q4_K MLX preload requires positive dimensions and input dim aligned to {QK_K}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int superBlocksPerRow = inDim / QK_K;
            long expectedBytes = (long)outDim * superBlocksPerRow * Q4_KBlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q4_K raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            int groupsPerRow = superBlocksPerRow * 8;
            byte[] packedWeights = new byte[checked(outDim * (inDim / 2))];
            System.Half[] scales = new System.Half[checked(outDim * groupsPerRow)];
            System.Half[] biases = new System.Half[scales.Length];
            byte[] values = new byte[32];

            byte* src = (byte*)hostData.ToPointer();
            for (int row = 0; row < outDim; row++)
            {
                for (int sb = 0; sb < superBlocksPerRow; sb++)
                {
                    byte* block = src + ((long)row * superBlocksPerRow + sb) * Q4_KBlockBytes;
                    float d = (float)BitConverter.UInt16BitsToHalf((ushort)(block[0] | (block[1] << 8)));
                    float min = (float)BitConverter.UInt16BitsToHalf((ushort)(block[2] | (block[3] << 8)));
                    byte* packedScales = block + 4;
                    byte* q = block + 4 + KScaleSize;
                    for (int group = 0; group < 8; group++)
                    {
                        GetScaleMinK4(group, packedScales, out byte scaleByte, out byte minByte);
                        int pairIndex = group / 2;
                        bool highNibble = (group & 1) != 0;
                        byte* qGroup = q + pairIndex * 32;
                        for (int i = 0; i < 32; i++)
                            values[i] = highNibble ? (byte)(qGroup[i] >> 4) : (byte)(qGroup[i] & 0x0F);

                        int scaleIndex = row * groupsPerRow + sb * 8 + group;
                        scales[scaleIndex] = (System.Half)(d * scaleByte);
                        biases[scaleIndex] = (System.Half)(-min * minByte);
                        long dstOffset = (long)row * (inDim / 2) + sb * (QK_K / 2) + group * 16;
                        PackUnsignedBits(values, MlxAffineQ4Bits, packedWeights, dstOffset);
                    }
                }
            }

            return CreateDeviceWeight(
                deviceId,
                (int)GgmlTensorType.Q4_K,
                ne0,
                ne1,
                rawBytes,
                packedWeights,
                scales,
                biases,
                new[] { outDim, inDim / 8 },
                new[] { outDim, groupsPerRow },
                MlxAffineQ4GroupSize,
                MlxAffineQ4Bits);
        }

        private static unsafe DeviceWeight CreateQ5KWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % QK_K != 0)
                throw new NotSupportedException($"Q5_K MLX preload requires positive dimensions and input dim aligned to {QK_K}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int superBlocksPerRow = inDim / QK_K;
            long expectedBytes = (long)outDim * superBlocksPerRow * Q5_KBlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q5_K raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            int groupsPerRow = superBlocksPerRow * 8;
            byte[] packedWeights = new byte[checked(outDim * superBlocksPerRow * 160)];
            System.Half[] scales = new System.Half[checked(outDim * groupsPerRow)];
            System.Half[] biases = new System.Half[scales.Length];
            byte[] values = new byte[32];

            byte* src = (byte*)hostData.ToPointer();
            for (int row = 0; row < outDim; row++)
            {
                for (int sb = 0; sb < superBlocksPerRow; sb++)
                {
                    byte* block = src + ((long)row * superBlocksPerRow + sb) * Q5_KBlockBytes;
                    float d = (float)BitConverter.UInt16BitsToHalf((ushort)(block[0] | (block[1] << 8)));
                    float min = (float)BitConverter.UInt16BitsToHalf((ushort)(block[2] | (block[3] << 8)));
                    byte* packedScales = block + 4;
                    byte* qh = block + 4 + KScaleSize;
                    byte* ql = qh + QK_K / 8;
                    for (int group = 0; group < 8; group++)
                    {
                        GetScaleMinK4(group, packedScales, out byte scaleByte, out byte minByte);
                        int pairIndex = group / 2;
                        bool highNibble = (group & 1) != 0;
                        byte* qlGroup = ql + pairIndex * 32;
                        for (int i = 0; i < 32; i++)
                        {
                            int lo4 = highNibble ? qlGroup[i] >> 4 : qlGroup[i] & 0x0F;
                            int bit5 = (qh[i] >> group) & 1;
                            values[i] = (byte)(lo4 | (bit5 << 4));
                        }

                        int scaleIndex = row * groupsPerRow + sb * 8 + group;
                        scales[scaleIndex] = (System.Half)(d * scaleByte);
                        biases[scaleIndex] = (System.Half)(-min * minByte);
                        long dstOffset = (long)row * superBlocksPerRow * 160 + sb * 160 + group * 20;
                        PackUnsignedBits(values, MlxAffineQ5Bits, packedWeights, dstOffset);
                    }
                }
            }

            return CreateDeviceWeight(
                deviceId,
                (int)GgmlTensorType.Q5_K,
                ne0,
                ne1,
                rawBytes,
                packedWeights,
                scales,
                biases,
                new[] { outDim, superBlocksPerRow * 40 },
                new[] { outDim, groupsPerRow },
                MlxAffineQ5GroupSize,
                MlxAffineQ5Bits);
        }

        private static unsafe DeviceWeight CreateQ6KWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % QK_K != 0)
                throw new NotSupportedException($"Q6_K MLX preload requires positive dimensions and input dim aligned to {QK_K}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int superBlocksPerRow = inDim / QK_K;
            long expectedBytes = (long)outDim * superBlocksPerRow * Q6_KBlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q6_K raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            int groupsPerRow = superBlocksPerRow * 16;
            long packedWeightBytes = (long)outDim * inDim * MlxAffineQ6Bits / 8;
            long scaleCount = (long)outDim * groupsPerRow;
            IntPtr packedWeightsBuffer = IntPtr.Zero;
            IntPtr scalesBuffer = IntPtr.Zero;
            IntPtr biasesBuffer = IntPtr.Zero;

            try
            {
                packedWeightsBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)packedWeightBytes, 64);
                scalesBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)(scaleCount * sizeof(float)), 64);
                biasesBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)(scaleCount * sizeof(float)), 64);
                if (packedWeightsBuffer == IntPtr.Zero || scalesBuffer == IntPtr.Zero || biasesBuffer == IntPtr.Zero)
                    throw new OutOfMemoryException("Unable to allocate MLX Q6_K staging buffers.");
                NativeMemory.Clear(packedWeightsBuffer.ToPointer(), (nuint)packedWeightBytes);

                byte* src = (byte*)hostData.ToPointer();
                byte* packedWeights = (byte*)packedWeightsBuffer.ToPointer();
                float* scalesOut = (float*)scalesBuffer.ToPointer();
                float* biasesOut = (float*)biasesBuffer.ToPointer();
                byte* values = stackalloc byte[16];
                for (int row = 0; row < outDim; row++)
                {
                    for (int sb = 0; sb < superBlocksPerRow; sb++)
                    {
                        byte* block = src + ((long)row * superBlocksPerRow + sb) * Q6_KBlockBytes;
                        byte* ql = block;
                        byte* qh = ql + QK_K / 2;
                        sbyte* blockScales = (sbyte*)(qh + QK_K / 4);
                        float d = (float)BitConverter.UInt16BitsToHalf((ushort)(((byte*)blockScales)[QK_K / 16] | (((byte*)blockScales)[QK_K / 16 + 1] << 8)));

                        for (int sub = 0; sub < 16; sub++)
                        {
                            int half = sub / 8;
                            int sh = sub % 8;
                            int qlOffset = half * 64 + (sh % 4) * 16;
                            bool isUpper = sh >= 4;
                            int qhOffset = half * 32 + (sh % 2) * 16;
                            int qhShift = (sh / 2) * 2;
                            for (int i = 0; i < 16; i++)
                            {
                                int lo4 = isUpper ? (ql[qlOffset + i] >> 4) & 0x0F : ql[qlOffset + i] & 0x0F;
                                int hi2 = (qh[qhOffset + i] >> qhShift) & 0x03;
                                values[i] = (byte)(lo4 | (hi2 << 4));
                            }

                            float scale = d * blockScales[sub];
                            long scaleIndex = (long)row * groupsPerRow + sb * 16 + sub;
                            scalesOut[scaleIndex] = scale;
                            biasesOut[scaleIndex] = -32.0f * scale;
                            long dstOffset = (long)row * (inDim * MlxAffineQ6Bits / 8) + sb * 192 + sub * 12;
                            PackUnsignedBits(values, 16, MlxAffineQ6Bits, packedWeights, dstOffset);
                        }
                    }
                }

                return CreateDeviceWeightFromHostBuffers(
                    deviceId,
                    (int)GgmlTensorType.Q6_K,
                    ne0,
                    ne1,
                    rawBytes,
                    packedWeightsBuffer,
                    scalesBuffer,
                    biasesBuffer,
                    new[] { outDim, superBlocksPerRow * 48 },
                    new[] { outDim, groupsPerRow },
                    MlxAffineQ6GroupSize,
                    MlxAffineQ6Bits,
                    scaleDType: DType.Float32,
                    biasDType: DType.Float32);
            }
            finally
            {
                if (packedWeightsBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(packedWeightsBuffer.ToPointer());
                if (scalesBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(scalesBuffer.ToPointer());
                if (biasesBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(biasesBuffer.ToPointer());
            }
        }

        private static DeviceWeight CreateQ4KRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.Q4_K,
                hostData,
                ne0,
                ne1,
                rawBytes,
                Q4_KBlockBytes,
                "Q4_K",
                mode: "q4_k");
        }

        private static DeviceWeight CreateQ5KRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.Q5_K,
                hostData,
                ne0,
                ne1,
                rawBytes,
                Q5_KBlockBytes,
                "Q5_K",
                mode: "q5_k");
        }

        private static DeviceWeight CreateQ6KRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.Q6_K,
                hostData,
                ne0,
                ne1,
                rawBytes,
                Q6_KBlockBytes,
                "Q6_K",
                mode: "q6_k");
        }

        private static DeviceWeight CreateIq2XxsRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.IQ2_XXS,
                hostData,
                ne0,
                ne1,
                rawBytes,
                IQ2_XXSBlockBytes,
                "IQ2_XXS",
                mode: "iq2_xxs");
        }

        private static DeviceWeight CreateIq2SRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.IQ2_S,
                hostData,
                ne0,
                ne1,
                rawBytes,
                IQ2_SBlockBytes,
                "IQ2_S",
                mode: "iq2_s");
        }

        private static DeviceWeight CreateIq3SRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            return CreateRawKWeight(
                deviceId,
                (int)GgmlTensorType.IQ3_S,
                hostData,
                ne0,
                ne1,
                rawBytes,
                IQ3_SBlockBytes,
                "IQ3_S",
                mode: "iq3_s");
        }

        private static DeviceWeight CreateRawKWeight(
            int deviceId,
            int ggmlType,
            IntPtr hostData,
            long ne0,
            long ne1,
            long rawBytes,
            int blockBytes,
            string label,
            string mode)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % QK_K != 0)
                throw new NotSupportedException($"{label} MLX preload requires positive dimensions and input dim aligned to {QK_K}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            long blocksPerRow = inDim / QK_K;
            long expectedBytes = outDim * blocksPerRow * blockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"{label} raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));
            if (expectedBytes > int.MaxValue)
                throw new NotSupportedException($"{label} raw MLX array exceeds the current Int32 shape limit: {expectedBytes} bytes.");

            MlxNative.MlxArray rawWeight = default;
            try
            {
                // Zero-copy wrap of the GGUF mmap region as an MLX uchar
                // array. Apple Silicon Metal accepts the host pointer via
                // MTLBuffer's no-copy path (shared memory mode), so we save
                // a 14 GB+ malloc+memcpy at model load and avoid carrying a
                // duplicate of the weights in RAM. Caller must keep the
                // GGUF mmap alive for the lifetime of the MLX array.
                rawWeight = MlxNative.NewArrayFromHostNoCopy(hostData, new[] { (int)expectedBytes }, DType.UInt8);
                var entry = new DeviceWeight
                {
                    Weight = rawWeight,
                    Scales = default,
                    Biases = default,
                    DeviceId = deviceId,
                    GgmlType = ggmlType,
                    Ne0 = ne0,
                    Ne1 = ne1,
                    RawBytes = rawBytes,
                    GroupSize = QK_K,
                    Bits = ggmlType == (int)GgmlTensorType.Q6_K ? 6
                        : ggmlType == (int)GgmlTensorType.Q4_K ? 4
                        : ggmlType == (int)GgmlTensorType.IQ3_S ? 3
                        : ggmlType == (int)GgmlTensorType.IQ2_XXS ? 2
                        : ggmlType == (int)GgmlTensorType.IQ2_S ? 2
                        : 5,
                    Mode = mode,
                };
                rawWeight = default;
                return entry;
            }
            finally
            {
                MlxNative.FreeArray(rawWeight);
            }
        }

        private static unsafe DeviceWeight CreateQ8Weight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % Q8_0BlockElements != 0)
                throw new NotSupportedException($"Q8_0 MLX preload requires positive dimensions and input dim aligned to {Q8_0BlockElements}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int blocksPerRow = inDim / Q8_0BlockElements;
            long expectedBytes = (long)outDim * blocksPerRow * Q8_0BlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"Q8_0 raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            long packedWeightBytes = (long)outDim * inDim;
            long scaleCount = (long)outDim * blocksPerRow;
            IntPtr packedWeightsBuffer = IntPtr.Zero;
            IntPtr scalesBuffer = IntPtr.Zero;
            IntPtr biasesBuffer = IntPtr.Zero;

            try
            {
                packedWeightsBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)packedWeightBytes, 64);
                scalesBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)(scaleCount * sizeof(ushort)), 64);
                biasesBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)(scaleCount * sizeof(ushort)), 64);
                if (packedWeightsBuffer == IntPtr.Zero || scalesBuffer == IntPtr.Zero || biasesBuffer == IntPtr.Zero)
                    throw new OutOfMemoryException("Unable to allocate MLX Q8_0 staging buffers.");

                byte* src = (byte*)hostData.ToPointer();
                byte* packedWeights = (byte*)packedWeightsBuffer.ToPointer();
                System.Half* scales = (System.Half*)scalesBuffer.ToPointer();
                System.Half* biases = (System.Half*)biasesBuffer.ToPointer();
                for (int row = 0; row < outDim; row++)
                {
                    for (int block = 0; block < blocksPerRow; block++)
                    {
                        long blockIndex = (long)row * blocksPerRow + block;
                        byte* blockPtr = src + blockIndex * Q8_0BlockBytes;
                        ushort scaleBits = (ushort)(blockPtr[0] | (blockPtr[1] << 8));
                        System.Half scale = BitConverter.UInt16BitsToHalf(scaleBits);
                        scales[blockIndex] = scale;
                        biases[blockIndex] = (System.Half)(-128.0f * (float)scale);

                        long dstOffset = (long)row * inDim + block * Q8_0BlockElements;
                        for (int j = 0; j < Q8_0BlockElements; j++)
                            packedWeights[dstOffset + j] = (byte)(blockPtr[2 + j] ^ 0x80);
                    }
                }

                return CreateDeviceWeightFromHostBuffers(
                    deviceId,
                    (int)GgmlTensorType.Q8_0,
                    ne0,
                    ne1,
                    rawBytes,
                    packedWeightsBuffer,
                    scalesBuffer,
                    biasesBuffer,
                    new[] { outDim, inDim / 4 },
                    new[] { outDim, blocksPerRow },
                    MlxAffineQ8GroupSize,
                    MlxAffineQ8Bits);
            }
            finally
            {
                if (packedWeightsBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(packedWeightsBuffer.ToPointer());
                if (scalesBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(scalesBuffer.ToPointer());
                if (biasesBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(biasesBuffer.ToPointer());
            }
        }

        private static DeviceWeight CreateIq4XsRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % QK_K != 0)
                throw new NotSupportedException($"IQ4_XS MLX preload requires positive dimensions and input dim aligned to {QK_K}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            long blocksPerRow = inDim / QK_K;
            long expectedBytes = outDim * blocksPerRow * IQ4_XSBlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"IQ4_XS raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));
            if (expectedBytes > int.MaxValue)
                throw new NotSupportedException($"IQ4_XS raw MLX array exceeds the current Int32 shape limit: {expectedBytes} bytes.");

            MlxNative.MlxArray rawWeight = default;
            try
            {
                // Zero-copy wrap: see CreateRawKWeight for the rationale.
                rawWeight = MlxNative.NewArrayFromHostNoCopy(hostData, new[] { (int)expectedBytes }, DType.UInt8);
                var entry = new DeviceWeight
                {
                    Weight = rawWeight,
                    Scales = default,
                    Biases = default,
                    DeviceId = deviceId,
                    GgmlType = (int)GgmlTensorType.IQ4_XS,
                    Ne0 = ne0,
                    Ne1 = ne1,
                    RawBytes = rawBytes,
                    GroupSize = QK_K,
                    Bits = 4,
                    Mode = "iq4_xs",
                };
                rawWeight = default;
                return entry;
            }
            finally
            {
                MlxNative.FreeArray(rawWeight);
            }
        }

        // Wrap an IQ4_NL GGUF row group as a single MLX uchar array (one big
        // [outDim * blocksPerRow * 18]-byte buffer) and hand it to
        // `MlxNative.Iq4NlMatmul`. The buffer is created from the host mmap
        // pointer with no copy — Apple Silicon unified memory makes this a
        // zero-cost wrap, and the OS keeps the pages resident for the
        // lifetime of the MlxArray reference.
        private static DeviceWeight CreateIq4NlRawWeight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % IQ4_NLBlockElements != 0)
                throw new NotSupportedException($"IQ4_NL MLX preload requires positive dimensions and input dim aligned to {IQ4_NLBlockElements}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            long blocksPerRow = inDim / IQ4_NLBlockElements;
            long expectedBytes = outDim * blocksPerRow * IQ4_NLBlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"IQ4_NL raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));
            if (expectedBytes > int.MaxValue)
                throw new NotSupportedException($"IQ4_NL raw MLX array exceeds the current Int32 shape limit: {expectedBytes} bytes.");

            MlxNative.MlxArray rawWeight = default;
            try
            {
                // Zero-copy wrap: see CreateRawKWeight for the rationale.
                rawWeight = MlxNative.NewArrayFromHostNoCopy(hostData, new[] { (int)expectedBytes }, DType.UInt8);
                var entry = new DeviceWeight
                {
                    Weight = rawWeight,
                    Scales = default,
                    Biases = default,
                    DeviceId = deviceId,
                    GgmlType = (int)GgmlTensorType.IQ4_NL,
                    Ne0 = ne0,
                    Ne1 = ne1,
                    RawBytes = rawBytes,
                    GroupSize = IQ4_NLBlockElements,
                    Bits = 4,
                    Mode = "iq4_nl",
                };
                rawWeight = default;
                return entry;
            }
            finally
            {
                MlxNative.FreeArray(rawWeight);
            }
        }

        private static unsafe DeviceWeight CreateMxfp4Weight(int deviceId, IntPtr hostData, long ne0, long ne1, long rawBytes)
        {
            if (ne0 <= 0 || ne1 <= 0 || ne0 > int.MaxValue || ne1 > int.MaxValue || ne0 % Mxfp4BlockElements != 0)
                throw new NotSupportedException($"MXFP4 MLX preload requires positive dimensions and input dim aligned to {Mxfp4BlockElements}, got [{ne0}, {ne1}].");

            int inDim = checked((int)ne0);
            int outDim = checked((int)ne1);
            int blocksPerRow = inDim / Mxfp4BlockElements;
            long expectedBytes = (long)outDim * blocksPerRow * Mxfp4BlockBytes;
            if (rawBytes < expectedBytes)
                throw new ArgumentException($"MXFP4 raw buffer is too small: expected at least {expectedBytes} bytes, got {rawBytes}.", nameof(rawBytes));

            long packedWeightBytes = (long)outDim * (inDim / 2);
            long scaleBytes = (long)outDim * blocksPerRow;
            IntPtr packedWeightsBuffer = IntPtr.Zero;
            IntPtr scalesBuffer = IntPtr.Zero;

            try
            {
                packedWeightsBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)packedWeightBytes, 64);
                scalesBuffer = (IntPtr)NativeMemory.AlignedAlloc((nuint)scaleBytes, 64);
                if (packedWeightsBuffer == IntPtr.Zero || scalesBuffer == IntPtr.Zero)
                    throw new OutOfMemoryException("Unable to allocate MLX MXFP4 staging buffers.");

                byte* src = (byte*)hostData.ToPointer();
                byte* packedWeights = (byte*)packedWeightsBuffer.ToPointer();
                byte* scales = (byte*)scalesBuffer.ToPointer();
                for (int row = 0; row < outDim; row++)
                {
                    for (int block = 0; block < blocksPerRow; block++)
                    {
                        long blockIndex = (long)row * blocksPerRow + block;
                        byte* blockPtr = src + blockIndex * Mxfp4BlockBytes;
                        scales[blockIndex] = blockPtr[0];
                        PackQ4Block(
                            blockPtr + 1,
                            packedWeights,
                            (long)row * (inDim / 2) + block * (Mxfp4BlockElements / 2));
                    }
                }

                return CreateDeviceWeightFromHostBuffers(
                    deviceId,
                    (int)GgmlTensorType.MXFP4,
                    ne0,
                    ne1,
                    rawBytes,
                    packedWeightsBuffer,
                    scalesBuffer,
                    IntPtr.Zero,
                    new[] { outDim, inDim / 8 },
                    new[] { outDim, blocksPerRow },
                    MlxMxfp4GroupSize,
                    MlxMxfp4Bits,
                    scaleDType: DType.UInt8,
                    hasBias: false,
                    mode: MlxMxfp4Mode);
            }
            finally
            {
                if (packedWeightsBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(packedWeightsBuffer.ToPointer());
                if (scalesBuffer != IntPtr.Zero)
                    NativeMemory.AlignedFree(scalesBuffer.ToPointer());
            }
        }

        private static unsafe void PackQ4Block(byte* source, byte[] destination, long destinationOffset)
        {
            for (int i = 0; i < 8; i++)
            {
                byte first = (byte)(source[i * 2] & 0x0F);
                byte second = (byte)(source[i * 2 + 1] & 0x0F);
                destination[destinationOffset + i] = (byte)(first | (second << 4));
            }

            for (int i = 0; i < 8; i++)
            {
                byte first = (byte)((source[i * 2] >> 4) & 0x0F);
                byte second = (byte)((source[i * 2 + 1] >> 4) & 0x0F);
                destination[destinationOffset + 8 + i] = (byte)(first | (second << 4));
            }
        }

        private static unsafe void PackQ4Block(byte* source, byte* destination, long destinationOffset)
        {
            for (int i = 0; i < 8; i++)
            {
                byte first = (byte)(source[i * 2] & 0x0F);
                byte second = (byte)(source[i * 2 + 1] & 0x0F);
                destination[destinationOffset + i] = (byte)(first | (second << 4));
            }

            for (int i = 0; i < 8; i++)
            {
                byte first = (byte)((source[i * 2] >> 4) & 0x0F);
                byte second = (byte)((source[i * 2 + 1] >> 4) & 0x0F);
                destination[destinationOffset + 8 + i] = (byte)(first | (second << 4));
            }
        }

        private static void PackUnsignedBits(byte[] values, int bits, byte[] destination, long destinationOffset)
        {
            int bitPosition = 0;
            int mask = (1 << bits) - 1;
            for (int i = 0; i < values.Length; i++)
            {
                int value = values[i] & mask;
                int byteIndex = (int)(destinationOffset + bitPosition / 8);
                int shift = bitPosition & 7;
                uint shifted = (uint)(value << shift);
                destination[byteIndex] |= (byte)shifted;
                if (shifted > 0xFF)
                    destination[byteIndex + 1] |= (byte)(shifted >> 8);
                if (shifted > 0xFFFF)
                    destination[byteIndex + 2] |= (byte)(shifted >> 16);
                bitPosition += bits;
            }
        }

        private static unsafe void PackUnsignedBits(byte* values, int valueCount, int bits, byte* destination, long destinationOffset)
        {
            int bitPosition = 0;
            int mask = (1 << bits) - 1;
            for (int i = 0; i < valueCount; i++)
            {
                int value = values[i] & mask;
                int byteIndex = (int)(destinationOffset + bitPosition / 8);
                int shift = bitPosition & 7;
                uint shifted = (uint)(value << shift);
                destination[byteIndex] |= (byte)shifted;
                if (shifted > 0xFF)
                    destination[byteIndex + 1] |= (byte)(shifted >> 8);
                if (shifted > 0xFFFF)
                    destination[byteIndex + 2] |= (byte)(shifted >> 16);
                bitPosition += bits;
            }
        }

        private static unsafe void GetScaleMinK4(int index, byte* packed, out byte scale, out byte min)
        {
            if (index < 4)
            {
                scale = (byte)(packed[index] & 63);
                min = (byte)(packed[index + 4] & 63);
                return;
            }

            scale = (byte)((packed[index + 4] & 0x0F) | ((packed[index - 4] >> 6) << 4));
            min = (byte)((packed[index + 4] >> 4) | ((packed[index] >> 6) << 4));
        }

        private static unsafe DeviceWeight CreateDeviceWeight(
            int deviceId,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            byte[] packedWeights,
            System.Half[] scales,
            System.Half[] biases,
            int[] weightShape,
            int[] scaleShape,
            int groupSize,
            int bits)
        {
            fixed (byte* packedPtr = packedWeights)
            fixed (System.Half* scalePtr = scales)
            fixed (System.Half* biasPtr = biases)
            {
                return CreateDeviceWeightFromHostBuffers(
                    deviceId,
                    ggmlType,
                    ne0,
                    ne1,
                    rawBytes,
                    (IntPtr)packedPtr,
                    (IntPtr)scalePtr,
                    (IntPtr)biasPtr,
                    weightShape,
                    scaleShape,
                    groupSize,
                    bits,
                    scaleDType: DType.Float16,
                    hasBias: true,
                    mode: MlxAffineMode);
            }
        }

        private static DeviceWeight CreateDeviceWeightFromHostBuffers(
            int deviceId,
            int ggmlType,
            long ne0,
            long ne1,
            long rawBytes,
            IntPtr packedWeights,
            IntPtr scales,
            IntPtr biases,
            int[] weightShape,
            int[] scaleShape,
            int groupSize,
            int bits,
            DType scaleDType = DType.Float16,
            DType biasDType = DType.Float16,
            bool hasBias = true,
            string mode = MlxAffineMode)
        {
            MlxNative.MlxArray weight = default;
            MlxNative.MlxArray scaleArray = default;
            MlxNative.MlxArray biasArray = default;
            try
            {
                weight = MlxNative.NewArrayFromHostUInt32(packedWeights, weightShape);
                scaleArray = MlxNative.NewArrayFromHost(scales, scaleShape, scaleDType);
                if (hasBias)
                    biasArray = MlxNative.NewArrayFromHost(biases, scaleShape, biasDType);
                MlxNative.Eval(weight);
                MlxNative.Eval(scaleArray);
                if (biasArray.IsValid)
                    MlxNative.Eval(biasArray);
                var entry = new DeviceWeight
                {
                    Weight = weight,
                    Scales = scaleArray,
                    Biases = biasArray,
                    DeviceId = deviceId,
                    GgmlType = ggmlType,
                    Ne0 = ne0,
                    Ne1 = ne1,
                    RawBytes = rawBytes,
                    GroupSize = groupSize,
                    Bits = bits,
                    Mode = mode,
                };
                weight = default;
                scaleArray = default;
                biasArray = default;
                return entry;
            }
            finally
            {
                MlxNative.FreeArray(weight);
                MlxNative.FreeArray(scaleArray);
                MlxNative.FreeArray(biasArray);
            }
        }

        private static void SetDeviceResult(Tensor tensor, MlxNative.MlxArray output)
        {
            MlxStorage storage = (MlxStorage)tensor.Storage;
            if (tensor.StorageOffset == 0 && tensor.Storage.ElementCount == tensor.ElementCount())
            {
                storage.ReplaceDeviceArray(output);
            }
            else
            {
                storage.UpdateDeviceSlice(tensor, output);
                MlxNative.FreeArray(output);
            }
        }

        private readonly struct CacheKey : IEquatable<CacheKey>
        {
            public readonly int DeviceId;
            private readonly IntPtr value;

            public CacheKey(int deviceId, IntPtr value)
            {
                DeviceId = deviceId;
                this.value = value;
            }

            public bool Equals(CacheKey other) => DeviceId == other.DeviceId && value == other.value;
            public override bool Equals(object obj) => obj is CacheKey other && Equals(other);
            public override int GetHashCode() => HashCode.Combine(DeviceId, value);
        }
    }
}
