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
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{

    // Reading weights out of the GGUF and getting them where the chosen backend
    // wants them: host quant buffers, CUDA / MLX / Metal / direct-CUDA device
    // preloads, gate+up fusion and requantization. All of it runs once, at load.
    public abstract partial class ModelBase
    {
        protected void LoadWeights()
        {
            // Parallel page-cache warm-up first: everything below (serial
            // F32/dequant reads, mmap faults from the sharding/upload threads)
            // otherwise reads the file at one-or-two-stream speed, which is the
            // whole cold-load time on network-backed model storage.
            ReadBonsaiMetadata();
            _gguf.PrefaultFileCache();
            Console.Write("Loading model weights...");
            int countF32 = 0;
            int countQuant = 0;
            long totalQuantBytes = 0;
            long totalF32Bytes = 0;
            long mappedQuantBytes = 0;
            bool tryMmap = CanUseFileMappedQuantizedWeights;
            foreach (var kv in _gguf.Tensors)
            {
                var info = kv.Value;
                long byteCount = _gguf.GetTensorByteCount(info);

                if (info.Type is GgmlTensorType.PQ2_0 or GgmlTensorType.PTQ1_0)
                {
                    QuantizedWeight converted = LoadBonsaiQuantizedWeight(info);
                    _quantWeights[info.Name] = converted;
                    countQuant++;
                    totalQuantBytes += converted.RawBytes;
                    continue;
                }

                if (IsQuantizedLinearWeight(info))
                {
                    if (IsGgmlBackend)
                        EnsureQuantBackendAvailable();

                    long ne0 = (long)info.Shape[0];
                    long ne1 = (long)info.Shape[1];

                    if (info.Shape.Length == 3 && info.Name.Contains("_exps."))
                    {
                        // 3D MoE expert tensor: split into per-expert 2D quantized weights.
                        // Also build a single stacked-along-experts view that the fused
                        // MoE prefill kernel can hand to ggml_mul_mat_id directly.
                        int numExperts = (int)info.Shape[2];
                        long perExpertBytes = byteCount / numExperts;
                        string baseName = info.Name;
                        if (baseName.EndsWith(".weight"))
                            baseName = baseName.Substring(0, baseName.Length - 7);

                        if (tryMmap && _gguf.TryGetTensorDataPointer(info, out IntPtr mappedTensorPtr))
                        {
                            for (int e = 0; e < numExperts; e++)
                            {
                                IntPtr expertPtr = new IntPtr(mappedTensorPtr.ToInt64() + e * perExpertBytes);
                                string expertName = $"{baseName}.{e}.weight";
                                _quantWeights[expertName] = QuantizedWeight.CreateExternalView(
                                    expertPtr, perExpertBytes, (int)info.Type, ne0, ne1, _gguf);
                                _stackedExpertMemberNames.Add(expertName);
                            }
                            // Free zero-cost stacked view: same bytes the per-expert
                            // views point into, owner is the GgufFile mmap.
                            _stackedExpertWeights[info.Name] = new StackedExpertWeights(
                                mappedTensorPtr, (int)info.Type, ne0, ne1, numExperts,
                                byteCount, isExternalView: true, ownerToken: _gguf,
                                ownedBuffer: IntPtr.Zero);
                            mappedQuantBytes += byteCount;
                        }
                        else
                        {
                            // Non-mmap path: keep the bulk buffer alive as the
                            // owning storage, and make per-expert views into it
                            // instead of memcpy'ing into per-expert buffers. This
                            // lets us expose a stacked-experts view for free at
                            // the cost of an extra strong reference held by the
                            // stacked weight (no memory duplication).
                            IntPtr bulkPtr = QuantizedWeight.AllocateBuffer(byteCount);
                            _gguf.ReadTensorDataToNative(info, bulkPtr, byteCount);

                            var stacked = new StackedExpertWeights(
                                bulkPtr, (int)info.Type, ne0, ne1, numExperts,
                                byteCount, isExternalView: false, ownerToken: null,
                                ownedBuffer: bulkPtr);
                            _stackedExpertWeights[info.Name] = stacked;

                            for (int e = 0; e < numExperts; e++)
                            {
                                IntPtr expertPtr = new IntPtr(bulkPtr.ToInt64() + e * perExpertBytes);
                                string expertName = $"{baseName}.{e}.weight";
                                _quantWeights[expertName] = QuantizedWeight.CreateExternalView(
                                    expertPtr, perExpertBytes, (int)info.Type, ne0, ne1, stacked);
                                _stackedExpertMemberNames.Add(expertName);
                            }
                        }
                        countQuant += numExperts;
                        totalQuantBytes += byteCount;
                    }
                    else
                    {
                        if (tryMmap && _gguf.TryGetTensorDataPointer(info, out IntPtr mappedTensorPtr))
                        {
                            _quantWeights[info.Name] = QuantizedWeight.CreateExternalView(
                                mappedTensorPtr, byteCount, (int)info.Type, ne0, ne1, _gguf);
                            mappedQuantBytes += byteCount;
                        }
                        else
                        {
                            IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                            _gguf.ReadTensorDataToNative(info, ptr, byteCount);
                            _quantWeights[info.Name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, ne0, ne1);
                        }
                        countQuant++;
                        totalQuantBytes += byteCount;
                    }
                }
                else
                {
                    long numElements = info.NumElements;

                    long[] ggufShape = new long[info.Shape.Length];
                    for (int i = 0; i < info.Shape.Length; i++)
                        ggufShape[i] = (long)info.Shape[i];

                    long[] tsShape = new long[ggufShape.Length];
                    for (int i = 0; i < ggufShape.Length; i++)
                        tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    IntPtr destPtr = GetStoragePtr(tensor);

                    if (info.Type == GgmlTensorType.F32)
                    {
                        _gguf.ReadTensorDataToFloat32Native(info, destPtr, numElements);
                    }
                    else
                    {
                        IntPtr tempPtr = QuantizedWeight.AllocateBuffer(byteCount);
                        try
                        {
                            _gguf.ReadTensorDataToNative(info, tempPtr, byteCount);
                            NativeDequant.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
                        }
                        finally { QuantizedWeight.FreeBuffer(tempPtr); }
                    }

                    _weights[info.Name] = tensor;

                    countF32++;
                    totalF32Bytes += numElements * 4;
                }
            }
            Console.WriteLine($" done ({countF32} F32 tensors, {countQuant} quantized tensors)");
            AttachSidecarWeightScales();
            if (countQuant > 0)
            {
                if (mappedQuantBytes > 0)
                    Console.WriteLine($"  Quantized: {totalQuantBytes / 1024 / 1024} MB ({mappedQuantBytes / 1024 / 1024} MB file-backed), F32: {totalF32Bytes / 1024 / 1024} MB");
                else
                    Console.WriteLine($"  Quantized: {totalQuantBytes / 1024 / 1024} MB, F32: {totalF32Bytes / 1024 / 1024} MB");
            }
        }

        protected void PrepareCudaQuantizedWeightsForInference()
        {
            if (_backend == BackendType.Mlx)
            {
                PrepareMlxQuantizedWeightsForInference();
                return;
            }

            if (_backend == BackendType.Cuda)
            {
                PrepareDirectCudaQuantizedWeightsForInference();
                return;
            }

            if (_backend == BackendType.GgmlMetal)
            {
                PrepareGgmlMetalQuantizedWeightsForInference();
                return;
            }

            // GgmlCuda and GgmlVulkan share this path: the preload below goes through
            // the backend-agnostic GGML device buffer API (TSGgml_PreloadQuantizedWeight),
            // which gives both discrete-GPU backends device-resident weights.
            if ((_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlVulkan) ||
                _cudaQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            EnsureQuantBackendAvailable();

            long preloadedBytes = 0;
            int preloadedCount = 0;
            int mappedHostViews = 0;

            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                if (qw.HasExternalHostView)
                    mappedHostViews++;
            }

            int activeRank = 0;
            foreach (var kv in _quantWeights)
            {
                string weightName = kv.Key;
                QuantizedWeight qw = kv.Value;

                if (!qw.HasHostData)
                    continue;

                // Skip weights the model serves device-resident by another route
                // (e.g. MoE per-expert split views that are covered by the stacked
                // expert device buffer). Preloading them here would put a second,
                // redundant copy of every expert byte in VRAM. The host view is
                // left intact so the stacked path / any per-op fallback can still
                // reach the bytes (and lazily upload on demand if ever needed).
                if (!ShouldPreloadCudaQuantWeightToDevice(weightName))
                    continue;

                // LAYER SPLIT: upload this weight to the GPU that owns its layer.
                // Exactly one rank, because ReleaseHostData() below frees the host
                // copy - a second preload elsewhere would upload from freed memory.
                // Weights that are NOT preloaded (the stacked experts, vetoed by
                // ShouldPreloadCudaQuantWeightToDevice) keep their host views and are
                // bound lazily by the native binder on whichever rank is active when
                // their layer runs, so they distribute across the GPUs for free.
                if (LayerSplitDegree > 1)
                {
                    int rank = PreloadRankForWeight(weightName);
                    if (rank != activeRank)
                    {
                        GgmlBasicOps.SetActiveRank(rank);
                        activeRank = rank;
                    }
                }

                // llama.cpp keeps token_embd on the host (its CPU_Mapped model
                // buffer): embedding lookup is a row gather, and when the quant
                // type has no device get_rows kernel Embedding() always serves it
                // from the retained host copy, so a device copy would be pure
                // VRAM waste (521 MB for Qwen3.6-27B's 248320x5120 Q3_K table).
                // Tied-output models matmul against token_embd through its device
                // cache key, so the skip requires a separate output.weight.
                if (string.Equals(weightName, "token_embd.weight", StringComparison.Ordinal)
                    && !CanUseGgmlQuantizedGetRows(qw.GgmlType)
                    && (_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")))
                    continue;

                IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                {
                    // The device cannot hold this weight in a single backend buffer
                    // (e.g. ggml-vulkan's per-buffer maxBufferSize cap; WSL's dzn
                    // Vulkan layer caps it under 3 GB, below Gemma E4B's ~2.9 GB
                    // Q8_0 per_layer_token_embd). Keep the host copy and let the
                    // model's host-gather fallbacks serve it.
                    qw.MarkDevicePreloadTooLarge();
                    Console.WriteLine(
                        $"  {weightName}: {qw.RawBytes / 1024 / 1024} MB exceeds the {_backend} device's single-buffer limit; keeping host copy (device lookups fall back to host).");
                    continue;
                }
                preloadedBytes += qw.RawBytes;
                preloadedCount++;

                if (!ShouldRetainCudaHostQuantWeight(weightName))
                {
                    bool wasMappedView = qw.HasExternalHostView;
                    qw.ReleaseHostData();

                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            if (activeRank != 0)
                GgmlBasicOps.SetActiveRank(0);

            if (mappedHostViews == 0)
                _gguf?.Dispose();
            _cudaQuantWeightsPrepared = true;

            if (preloadedCount > 0)
                Console.WriteLine($"  Device-resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors");
        }

        private void PrepareMlxQuantizedWeightsForInference()
        {
            if (_mlxQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            if (_allocator is not MlxAllocator mlxAllocator)
                return;

            long fallbackBytes = MlxHostFallbackQuantizedBytes();
            long nativeBytes = MlxNativePreloadableQuantizedBytes();
            if (fallbackBytes > 0)
            {
                Console.WriteLine(
                    $"  MLX eager quantized preload: {nativeBytes / 1024 / 1024} MB native-capable weights will be device-resident; " +
                    $"{fallbackBytes / 1024 / 1024} MB fallback quantized weights remain file-backed.");
            }

            bool offloadEnabled = MoeExpertOffload.IsEnabled;
            long preloadedBytes = 0;
            int preloadedCount = 0;
            long deferredBytes = 0;
            int deferredCount = 0;
            long zeroCopyExpertBytes = 0;
            int zeroCopyExpertCount = 0;
            long fallbackExpertBytes = 0;
            int fallbackExpertCount = 0;
            int mappedHostViews = 0;
            long regroupedQ6KBytes = 0;
            int regroupedQ6KCount = 0;
            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                if (qw.HasExternalHostView)
                    mappedHostViews++;
            }

            foreach (var kv in _quantWeights)
            {
                string weightName = kv.Key;
                QuantizedWeight qw = kv.Value;
                if (!qw.HasHostData)
                    continue;

                // Skip weights the model serves device-resident by another route
                // (e.g. MoE per-expert views covered by a stacked-experts MLX
                // weight built for mlx_gather_qmm). Preloading them here would
                // put a second, redundant copy of every expert byte in unified
                // memory. The host view is left intact so the per-expert
                // fallback can still lazily upload on first use if the batched
                // path ever refuses at runtime.
                if (!ShouldPreloadMlxQuantWeightToDevice(weightName, qw))
                    continue;

                bool isExpert = offloadEnabled && MoeExpertOffload.IsExpertWeightName(weightName);
                bool canPreload = MlxQuantizedOps.CanPreloadQuantizedType(qw.GgmlType);
                bool preloadCopies = canPreload && MlxQuantizedOps.PreloadDuplicatesHostMemory(qw.GgmlType, qw.RawBytes);

                if (isExpert && !canPreload)
                {
                    // Host-fallback expert (e.g. IQ1_S / IQ2_XS / IQ1_M in
                    // Nemotron's UD-IQ2_XXS): matmul runs the host-side
                    // dequant path and never enters the MLX cache. Track for
                    // accounting only.
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    MoeExpertOffload.RegisterOffloadable(cacheKey);
                    if (qw.HasExternalHostView)
                        MoeExpertOffload.AdvisePagesNotNeeded(qw.Data, qw.RawBytes);
                    fallbackExpertBytes += qw.RawBytes;
                    fallbackExpertCount++;
                    continue;
                }

                if (isExpert && canPreload && preloadCopies)
                {
                    // Repack-kernel expert (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
                    // MXFP4, or Q5_K with TS_MLX_Q5K_RAW=0). The MLX preload
                    // would allocate fresh MLX-managed memory and double the
                    // residency cost; offload bypasses that by deferring the
                    // upload to first use and bounding total residency via the
                    // LRU. This is where the offload mechanism produces the
                    // largest measured memory savings.
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    MoeExpertOffload.RegisterOffloadable(cacheKey);
                    if (qw.HasExternalHostView)
                        MoeExpertOffload.AdvisePagesNotNeeded(qw.Data, qw.RawBytes);
                    deferredBytes += qw.RawBytes;
                    deferredCount++;
                    continue;
                }

                if (isExpert && canPreload && !preloadCopies)
                {
                    // Raw-wrap kernel expert (Q4_K / Q6_K, IQ2_XXS / IQ2_S /
                    // IQ3_S / IQ4_XS, or Q5_K when raw mode is enabled). The
                    // MLX preload does NOT allocate fresh memory — it just
                    // wraps the GGUF mmap pointer as an MLX array. The
                    // baseline preload path's qw.ReleaseHostData() call after
                    // upload already issues madvise(DONTNEED) on the mmap
                    // region, letting the OS evict page-cache pages between
                    // accesses. Routing these experts through the offload LRU
                    // instead would just churn MlxArray wrappers without any
                    // memory-residency win, and on Apple Silicon makes
                    // measured RSS WORSE because lazy wrappers prevent the
                    // OS from settling its page-cache eviction policy.
                    //
                    // → Fall through to the baseline-preload path below.
                    zeroCopyExpertBytes += qw.RawBytes;
                    zeroCopyExpertCount++;
                }

                if (!canPreload)
                    continue;

                IntPtr preloadKey = qw.EnsureDeviceCacheKey();
                MlxQuantizedOps.PreloadQuantizedWeight(
                    mlxAllocator,
                    preloadKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);

                preloadedBytes += qw.RawBytes;
                preloadedCount++;
                if (qw.GgmlType == (int)GgmlTensorType.Q6_K && MlxQuantizedOps.Q6KUsesAffine8(qw.RawBytes))
                {
                    regroupedQ6KBytes += qw.RawBytes;
                    regroupedQ6KCount++;
                }

                // Repack quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/MXFP4/Q5_K-repack)
                // were materialised into a fresh MLX-allocator MTLBuffer in
                // the preload above. The original GGUF/host bytes are now
                // redundant — releasing them frees the source view and
                // (when external) lets the OS reclaim those mmap pages.
                //
                // Raw-wrap quants (Q4_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S,
                // IQ4_XS, IQ4_NL, Q5_K-raw) are wrapped zero-copy via
                // mlx_array_new_data_managed → MTLBuffer-with-bytes-no-copy
                // pointing at the GGUF mmap. They MUST keep that mmap
                // alive — calling ReleaseHostData here would (a) lose the
                // host pointer that MLX is reading from, (b) invoke
                // madvise(MADV_DONTNEED) on still-active model pages,
                // forcing the kernel to re-read them from disk on every
                // forward pass.
                bool wasMappedView = qw.HasExternalHostView;
                if (preloadCopies)
                {
                    qw.ReleaseHostData();
                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            // Stacked-experts views are lazily uploaded by the batched-MoE matmul
            // path (no explicit preload). Register them as offloadable so any
            // repack-kernel batched-MoE uploads are governed by the LRU. For
            // raw-wrap kernel stacked views (the common case — IQ2_XXS, Q4_K
            // etc.) the LRU does no harm because no MLX-allocator memory is
            // duplicated, and the registration is essentially a no-op there.
            if (offloadEnabled)
            {
                foreach (var stacked in _stackedExpertWeights.Values)
                    MoeExpertOffload.RegisterOffloadable(stacked.Data);
            }

            if (regroupedQ6KCount > 0)
            {
                // Not a lossless reinterpretation like the other repacks; say so.
                Console.WriteLine(
                    $"  Q6_K: {regroupedQ6KCount} tensors ({regroupedQ6KBytes / 1024 / 1024} MB) regrouped to MLX 8-bit affine " +
                    "for MLX's quantized kernels, within half an 8-bit step per weight (TS_MLX_Q6K_AFFINE8=1, or past the raw kernels' 2 GB limit; otherwise Q6_K stays exact).");
            }

            _mlxQuantWeightsPrepared = true;
            // Keep the GGUF mmap alive whenever any quantized weight still has a
            // file-backed view — both the existing fallback path (unpreloadable
            // types) AND the offload path (expert weights with retained host
            // pointers) need it to remain mapped.
            if (mappedHostViews == 0 && preloadedCount > 0)
                _gguf?.Dispose();
            else if (_gguf != null && string.Equals(
                Environment.GetEnvironmentVariable("TS_MLX_MLOCK_GGUF") ?? "1", "1", StringComparison.Ordinal))
            {
                // Pin the GGUF mmap region in physical RAM. Without this,
                // macOS treats file-backed pages as evictable and the kernel
                // throws model weights into the page cache between forward
                // passes — every subsequent layer page-faults them back from
                // disk and inference collapses to ~0.3 tok/s.
                //
                // mlx_set_wired_limit only governs MLX-allocator MTLBuffer
                // residency, not arbitrary mmap'd pages, so MTLBuffer-backed
                // zero-copy wrappers (CreateIq4XsRawWeight etc.) need this
                // explicit mlock too. Opt out via TS_MLX_MLOCK_GGUF=0.
                // Only the tensors MLX still reads from the file: those preloaded
                // into device buffers had their pages released above, and locking
                // the whole file would fault them back in and wire them twice.
                var retained = new List<(IntPtr, long)>();
                foreach (QuantizedWeight weight in _quantWeights.Values)
                {
                    if (weight.HasExternalHostView)
                        retained.Add((weight.Data, weight.RawBytes));
                }
                foreach (StackedExpertWeights stacked in _stackedExpertWeights.Values)
                {
                    if (stacked.IsExternalView)
                        retained.Add((stacked.Data, stacked.TotalRawBytes));
                }
                bool locked = _gguf.TryLockRanges(retained);
                if (locked)
                {
                    Console.WriteLine(
                        $"  GGUF mmap pinned via mlock: {_gguf.LockedRangeBytes / 1024 / 1024} MB still read from the file " +
                        "(set TS_MLX_MLOCK_GGUF=0 to disable).");
                }
                else
                {
                    Console.WriteLine(
                        $"  GGUF mlock failed (errno={_gguf.LastLockError}); inference may swap under memory pressure. " +
                        "Set TS_MLX_MLOCK_GGUF=0 to suppress this message.");
                }
            }

            if (preloadedCount > 0 || deferredCount > 0 || zeroCopyExpertCount > 0 || fallbackExpertCount > 0)
            {
                var snapshot = mlxAllocator.GetMemorySnapshot();
                Console.WriteLine(
                    $"  MLX resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors " +
                    $"(active {snapshot.ActiveBytes / 1024 / 1024} MB, cache {snapshot.CacheBytes / 1024 / 1024} MB, peak {snapshot.PeakBytes / 1024 / 1024} MB)");
                if (deferredCount > 0 || zeroCopyExpertCount > 0 || fallbackExpertCount > 0)
                {
                    long capMb = MoeExpertOffload.MaxCacheBytes / 1024 / 1024;
                    long totalExpertMb = (deferredBytes + zeroCopyExpertBytes + fallbackExpertBytes) / 1024 / 1024;
                    int totalExpertCount = deferredCount + zeroCopyExpertCount + fallbackExpertCount;
                    Console.WriteLine(
                        $"  MoE expert weights detected: {totalExpertMb} MB across {totalExpertCount} tensors " +
                        $"(TS_MLX_EXPERT_OFFLOAD_MB={(offloadEnabled ? capMb.ToString() : "0")})");
                    if (deferredCount > 0)
                    {
                        Console.WriteLine(
                            $"    Offload-LRU: {deferredBytes / 1024 / 1024} MB / {deferredCount} tensors are " +
                            $"repack-kernel quants (LRU bounds MLX-allocator residency to ~{capMb} MB).");
                    }
                    if (zeroCopyExpertCount > 0)
                    {
                        Console.WriteLine(
                            $"    Zero-copy preload: {zeroCopyExpertBytes / 1024 / 1024} MB / {zeroCopyExpertCount} tensors are " +
                            $"raw-wrap kernel quants (no MLX allocator copy; baseline madvise upfront, OS page-cache evicts cold pages).");
                    }
                    if (fallbackExpertCount > 0)
                    {
                        Console.WriteLine(
                            $"    Host fallback: {fallbackExpertBytes / 1024 / 1024} MB / {fallbackExpertCount} tensors use " +
                            $"unpreloadable quant types (matmul runs via host-side dequant; OS page cache governs residency).");
                    }
                }
                MlxBackend.ClearCache();
            }
        }

        // GGML_METAL doesn't perform an eager device upload — weights are
        // wrapped as MTLBuffer pointers around the GGUF mmap via
        // ggml_backend_dev_buffer_from_host_ptr, so they already live in
        // unified memory at zero extra bytes. The wrapper itself, cached
        // in the native g_host_buffer_cache, can still keep Metal's claim
        // on those pages and prevent the OS from paging them out. When
        // TS_MLX_EXPERT_OFFLOAD_MB is set, we register expert host pointers
        // with the native cache so it LRU-bounds their MTLBuffer wrappers
        // and frees the oldest ones when the budget is exceeded.
        private void PrepareGgmlMetalQuantizedWeightsForInference()
        {
            if (_quantWeights.Count == 0)
                return;

            EnsureQuantBackendAvailable();

            if (!MoeExpertOffload.IsEnabled)
                return;

            long offloadedBytes = 0;
            int offloadedCount = 0;
            foreach (var kv in _quantWeights)
            {
                QuantizedWeight qw = kv.Value;
                if (!qw.HasHostData)
                    continue;
                if (!MoeExpertOffload.IsExpertWeightName(kv.Key))
                    continue;
                GgmlBasicOps.RegisterOffloadable(qw.Data);
                offloadedBytes += qw.RawBytes;
                offloadedCount++;
            }

            // The native MoE FFN kernels look up each expert weight via
            // try_get_cacheable_tensor_buffer keyed by `data` — the GGUF
            // mmap pointer. The stacked-experts view points at the SAME
            // bytes (its Data is the start of the 3D GGUF tensor, which is
            // also the first per-expert tile's address), so the per-expert
            // RegisterOffloadable above already covers it. We do not
            // register stacked.Data separately because doing so would
            // double-count the resident bytes.

            if (offloadedCount > 0)
            {
                GgmlBasicOps.SetOffloadableBudget(MoeExpertOffload.MaxCacheBytes);
                long capMb = MoeExpertOffload.MaxCacheBytes / 1024 / 1024;
                Console.WriteLine(
                    $"  GGML_METAL MoE expert offload: {offloadedBytes / 1024 / 1024} MB across {offloadedCount} tensors registered " +
                    $"(LRU cap {capMb} MB, set TS_MLX_EXPERT_OFFLOAD_MB=0 to disable)");
            }
        }

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): logs dedicated-VRAM free/used at
        /// <paramref name="label"/> when the active allocator is the direct-CUDA one.</summary>
        internal static void LogCudaVram(IAllocator allocator, string label)
        {
            if (allocator is CudaAllocator cuda)
                cuda.LogVram(label);
        }

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): logs the model allocator's
        /// dedicated-VRAM free/used at <paramref name="label"/>.</summary>
        public void LogVramSnapshot(string label) => LogCudaVram(_allocator, label);

        private void PrepareDirectCudaQuantizedWeightsForInference()
        {
            if (_cudaQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            if (_allocator is not CudaAllocator cudaAllocator)
                return;

            cudaAllocator.LogVram("before direct-CUDA quant weight preload");

            // When CUDA kernels are unavailable (PTX load failed), device-side
            // quantized matmul/embedding will fail and every op falls back to
            // the CPU dequant path.  Keep all host data alive in that case.
            bool kernelsAvailable = CudaQuantizedOps.AreKernelsAvailable(cudaAllocator);

            long preloadedBytes = 0;
            int preloadedCount = 0;
            int mappedHostViews = 0;
            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                if (qw.HasExternalHostView)
                    mappedHostViews++;
            }

            foreach (var kv in _quantWeights)
            {
                var qw = kv.Value;
                if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                    continue;

                IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                CudaQuantizedOps.PreloadQuantizedWeight(
                    cudaAllocator,
                    cacheKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);
                preloadedBytes += qw.RawBytes;
                preloadedCount++;

                bool wasMappedView = qw.HasExternalHostView;
                if (kernelsAvailable && !ShouldRetainCudaHostQuantWeight(kv.Key))
                {
                    qw.ReleaseHostData();
                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            _cudaQuantWeightsPrepared = true;
            if (mappedHostViews == 0)
                _gguf?.Dispose();

            if (preloadedCount > 0)
                Console.WriteLine($"  Direct CUDA resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors (host copies released)");

            cudaAllocator.LogVram("after direct-CUDA quant weight preload");
        }

        // TS_GGML_RETAIN_HOST_WEIGHTS=1 keeps every quantized weight's host copy
        // alive after the device preload instead of releasing it. Costs the model's
        // full host footprint in RAM; diagnostic/workaround knob for any native
        // path that still reads weight bytes through the original host pointer
        // after preload (symptom: memcpy access violation on first forward).
        private static readonly bool s_retainAllHostQuantWeights =
            Environment.GetEnvironmentVariable("TS_GGML_RETAIN_HOST_WEIGHTS") == "1";

        protected virtual bool ShouldRetainCudaHostQuantWeight(string weightName)
        {
            return s_retainAllHostQuantWeights ||
                string.Equals(weightName, "token_embd.weight", StringComparison.Ordinal) ||
                string.Equals(weightName, "per_layer_token_embd.weight", StringComparison.Ordinal);
        }

        /// <summary>
        /// Whether <paramref name="weightName"/> should get its own device-resident
        /// copy during <see cref="PrepareCudaQuantizedWeightsForInference"/> (the
        /// <c>ggml_cuda</c> backend). Defaults to true. Models whose CUDA decode and
        /// prefill paths serve MoE experts exclusively through the stacked-expert
        /// device buffer override this to return false for the per-expert split
        /// views, avoiding a second full copy of the experts in VRAM.
        ///
        /// Overrides MUST keep the <see cref="MoeCpuOffloadConfig"/> term: a routed
        /// expert belonging to a <c>--n-cpu-moe</c> layer is multiplied on the host
        /// and uploading it would spend exactly the VRAM the flag exists to save.
        /// </summary>
        /// <summary>
        /// GPU that should hold <paramref name="weightName"/> under a layer split.
        /// Default 0 (everything on the first GPU); a model that splits overrides
        /// this to return the rank owning the weight's layer. Only consulted when
        /// <see cref="LayerSplitDegree"/> &gt; 1.
        /// </summary>
        protected virtual int PreloadRankForWeight(string weightName) => 0;

        protected virtual bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !MoeCpuOffloadConfig.IsOffloadedExpertWeightName(weightName);

        /// <summary>
        /// Per-weight veto for the eager MLX quantized preload
        /// (<see cref="PrepareMlxQuantizedWeightsForInference"/>). Models whose
        /// MLX forward serves a weight device-resident by another route (e.g.
        /// routed experts through a stacked-experts <c>mlx_gather_qmm</c>
        /// weight) override this to return false for those names, avoiding a
        /// second full copy of the bytes in unified memory. Skipped weights
        /// keep their host data, so any per-op fallback still lazily uploads
        /// them on first use.
        /// </summary>
        protected virtual bool ShouldPreloadMlxQuantWeightToDevice(string weightName, QuantizedWeight weight)
            => true;

        protected bool CanUseGgmlQuantizedGetRows(int ggmlType)
        {
            if (!IsGgmlBackend)
                return false;

            // Q1_0 is deliberately gathered from the mmap on the host. It is only
            // used by embedding tables, so this moves a few KiB per prompt while
            // avoiding backend-specific get_rows coverage gaps for the new format.
            // The large Q1_0 projection matrices still stay device-resident and use
            // the optimized mul_mat kernels.
            if ((GgmlTensorType)ggmlType == GgmlTensorType.Q1_0)
                return false;

            if (_backend != BackendType.GgmlCuda)
                return true;

            // ggml-cuda's get_rows kernel only implements the legacy round-number
            // quant types (see ExternalProjects/ggml/src/ggml-cuda/getrows.cu:
            // ggml_cuda_get_rows_switch_src0_type). k-quants such as Q6_K are NOT
            // supported and abort at runtime, so they must fall back to the host
            // dequant path (PopulateQuantizedRows). Keep this list in sync with the
            // upstream kernel's supported src0 types.
            return ((GgmlTensorType)ggmlType) switch
            {
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                _ => false,
            };
        }

        /// <summary>
        /// Bytes of anonymous memory NOT allocated because a fusion that needed a copy
        /// was declined. Reported once per load so the trade is visible rather than
        /// silent; see <see cref="AllowWeightFusionCopies"/>.
        /// </summary>
        protected long DeclinedFusionCopyBytes { get; private set; }

        /// <summary>Fusions declined because joining the sources would have required a copy.</summary>
        protected int DeclinedFusionCopyCount { get; private set; }

        /// <summary>
        /// Fuse unconditionally -- free view when the sources are adjacent, anonymous
        /// copy when they are not. For the few consumers that have NO separate-weights
        /// path and would simply fail without the fused tensor (gpt-oss's per-expert
        /// gate_up, which ExpertFFN and the fused decode arrays index by name). Every
        /// other site must use <see cref="TryCreateFusedQuantizedWeight"/>, which
        /// declines the copy where memory matters more than the saved matmul.
        /// </summary>
        protected static QuantizedWeight CreateFusedQuantizedWeightRequired(params QuantizedWeight[] weights)
        {
            if (QuantizedWeight.TryCreateConcatenatedView(out QuantizedWeight view, weights))
                return view;
            return QuantizedWeight.ConcatOrCreateCopy(weights);
        }

        /// <summary>
        /// Fuse for a caller that has NO separate-weights fallback: always produces a
        /// tensor, by view when the sources are adjacent and by copy when they are not.
        /// This is the default because declining without a fallback leaves the consumer
        /// with no weight at all -- see the SupportsSplitGateUpFfn warning below, which
        /// is what that failure looks like.
        /// </summary>
        protected bool TryCreateFusedQuantizedWeight(out QuantizedWeight fused, params QuantizedWeight[] weights)
            => TryCreateFusedQuantizedWeight(separatePathAvailable: false, out fused, weights);

        /// <summary>
        /// Fuse, optionally declining the anonymous copy when the caller can run the
        /// sources separately instead.
        /// </summary>
        /// <param name="separatePathAvailable">
        /// True only where the caller has been VERIFIED to work with the fused tensor
        /// absent -- SeparateQkv in the fused decode kernels, SupportsSplitGateUpFfn for
        /// the FFN, the four source weights for the recurrent pack. Where it is false the
        /// copy is taken regardless of policy, because the alternative is a model that
        /// loads and then generates nonsense.
        /// </param>
        protected bool TryCreateFusedQuantizedWeight(
            bool separatePathAvailable, out QuantizedWeight fused, params QuantizedWeight[] weights)
        {
            if (CanUseFileMappedQuantizedWeights && QuantizedWeight.TryCreateConcatenatedView(out fused, weights))
                return true;

            // The free view was not available -- the sources are not adjacent in the
            // mapping, or one of them is itself a copy. Building the fused tensor now
            // means duplicating already-mapped bytes into anonymous memory, which on a
            // phone is charged against the jetsam limit. Decline and let the caller use
            // its separate-weights path.
            if (separatePathAvailable && !AllowWeightFusionCopies)
            {
                long bytes = 0;
                for (int i = 0; i < weights.Length; i++)
                    bytes += weights[i].RawBytes;
                DeclinedFusionCopyBytes += bytes;
                DeclinedFusionCopyCount++;
                fused = null;
                return false;
            }

            fused = QuantizedWeight.ConcatOrCreateCopy(weights);
            return true;
        }

        /// <summary>
        /// Print the fusion copies this load declined, once, after all fusion has run.
        /// Silent when none were declined (the desktop default).
        /// </summary>
        private bool _reportedDeclinedFusion;

        protected void ReportDeclinedFusionCopies()
        {
            // Once per load, whichever fusion pass happens to run last for this family.
            if (_reportedDeclinedFusion || DeclinedFusionCopyCount <= 0)
                return;
            _reportedDeclinedFusion = true;
            Console.WriteLine(
                $"  Kept separate to save memory: {DeclinedFusionCopyCount} fusions, " +
                $"{DeclinedFusionCopyBytes / (1024.0 * 1024.0):0} MB of anonymous copies avoided " +
                "(TS_WEIGHT_FUSION_COPIES=1 to fuse anyway)");
        }

        protected bool HasMlxHostFallbackQuantizedWeights()
        {
            if (_backend != BackendType.Mlx)
                return false;

            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (!MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    return true;
            }

            return false;
        }

        protected long MlxHostFallbackQuantizedBytes()
        {
            if (_backend != BackendType.Mlx)
                return 0;

            long bytes = 0;
            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (!MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    bytes += weight.RawBytes;
            }

            return bytes;
        }

        protected long MlxNativePreloadableQuantizedBytes()
        {
            if (_backend != BackendType.Mlx)
                return 0;

            long bytes = 0;
            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    bytes += weight.RawBytes;
            }

            return bytes;
        }

        protected unsafe void PopulateQuantizedRows(Tensor result, QuantizedWeight weight, int[] rowIndices)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (rowIndices == null)
                throw new ArgumentNullException(nameof(rowIndices));
            if (!weight.HasHostData)
                throw new InvalidOperationException("Quantized row lookup requires host-side weight data.");

            int dim = (int)weight.Ne0;
            if (result.DimensionCount != 2 || result.ElementType != DType.Float32 ||
                result.Sizes[0] != rowIndices.Length || result.Sizes[1] != dim)
            {
                throw new ArgumentException("Result tensor shape must be [rowIndices.Length, weight.Ne0].", nameof(result));
            }

            long rowBytes = NativeDequant.RowSize(weight.GgmlType, weight.Ne0);
            byte* basePtr = (byte*)weight.Data.ToPointer();
            float* dst = GetFloatPtr(result);
            for (int i = 0; i < rowIndices.Length; i++)
            {
                byte* rowPtr = basePtr + (long)rowIndices[i] * rowBytes;
                NativeDequant.DequantizeToFloat32Native(
                    weight.GgmlType,
                    (IntPtr)rowPtr,
                    (IntPtr)(dst + (long)i * dim),
                    dim);
                weight.InverseBonsaiEmbeddingRow(new Span<float>(dst + (long)i * dim, dim));
            }

            InvalidateTensorDeviceCache(result);
        }

        /// <summary>
        /// True when this model's FFN can run <c>ffn_gate</c> and <c>ffn_up</c> as
        /// two separate projections - i.e. when <see cref="FuseGateUpWeights"/> is
        /// allowed to leave a layer unfused.
        ///
        /// Default false, deliberately: most families look up
        /// <c>blk.N.ffn_gate_up.weight</c> unconditionally and would either throw
        /// or (worse) bind a null weight and produce silent garbage. Mixed-IQ "UD"
        /// GGUFs make that reachable, so a family that has not implemented the
        /// split path must say so at load time rather than fail later.
        /// </summary>
        /// <summary>Non-trivial per-tensor sidecar ".scale" tensors were found
        /// and attached to their QuantizedWeights at load time.</summary>
        protected bool HasSidecarWeightScales { get; private set; }

        /// <summary>
        /// llama.cpp-style NVFP4 sidecars: a 1-element F32 tensor named
        /// "&lt;base&gt;.scale" (converted from HF weight_scale_2) multiplies the
        /// matmul output of "&lt;base&gt;.weight". Attach the value to the weight so
        /// every consumer can apply it. ".input_scale" sidecars are calibration
        /// metadata that llama.cpp also ignores at inference. Vector-valued
        /// ".scale" tensors (per-expert / per-dim, e.g. Gemma4's) are left in
        /// _weights for their model-specific consumers.
        /// </summary>
        private void AttachSidecarWeightScales()
        {
            int attached = 0;
            foreach (var kv in _quantWeights)
            {
                if (!kv.Key.EndsWith(".weight", StringComparison.Ordinal))
                    continue;
                string scaleKey = kv.Key.Substring(0, kv.Key.Length - ".weight".Length) + ".scale";
                if (_weights.TryGetValue(scaleKey, out var st) && st.ElementCount() == 1)
                {
                    float v = st.GetElementsAsFloat(1)[0];
                    if (v != 1.0f)
                    {
                        kv.Value.Scale = v;
                        attached++;
                    }
                }
            }
            if (attached > 0)
            {
                HasSidecarWeightScales = true;
                Console.WriteLine($"  Per-tensor weight scales: {attached} sidecar .scale tensors attached (NVFP4 scale2)");
                if (IsTensorParallel && !SupportsTensorParallelWeightScales)
                    throw new NotSupportedException(
                        "This GGUF carries per-tensor weight-scale sidecars (.scale), which the " +
                        "tensor-parallel path for this architecture does not apply yet. " +
                        "Run it without --tp, or use a GGUF whose scales are folded into " +
                        "the quantized blocks.");
            }
        }

        /// <summary>
        /// Whether this architecture's tensor-parallel paths apply per-tensor
        /// sidecar weight scales (<see cref="QuantizedWeight.Scale"/>). The scalar
        /// itself is shard-invariant - it does not depend on the output row, and it
        /// distributes over the row-parallel AllReduce - so this is purely about
        /// whether every per-rank matmul in the family has been wired to apply it.
        /// </summary>
        protected virtual bool SupportsTensorParallelWeightScales => false;

        protected virtual bool SupportsSplitGateUpFfn => false;

        // On MLX a mixed-quant gate/up pair (IQ4_XS ffn_gate + Q5_K ffn_up in Qwen3.8
        // UD quants) stays two weights in their own formats instead of the gate being
        // requantized to Q5_K and fused. MLX runs Q5_K as 6-bit affine, so the fused
        // gate read 6 bits a weight where IQ4_XS reads 4.25, and the requantization
        // rounded the gate a second time. Qwen3.8-27B UD-Q4_K_XL on an M5 Pro, fused ->
        // split: decode 13.1 -> 13.9 tok/s, prefill 443 -> 432 (pp512) and 444 -> 454
        // (pp4096) tok/s, resident weights 17.9 -> 17.1 GB. Needs a family that can run
        // the pair as two matmuls (SupportsSplitGateUpFfn) and has a half-precision split
        // prefill FFN on MLX (SplitsMixedGateUpOnMlx; without one, prefill lost ~7%).
        // TS_MLX_MIXED_GATE_UP_SPLIT=0 fuses them as on the other backends.
        protected virtual bool SplitsMixedGateUpOnMlx => false;

        private bool KeepMixedGateUpSplitOnMlx =>
            _backend == BackendType.Mlx && SupportsSplitGateUpFfn && SplitsMixedGateUpOnMlx
            && !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_MIXED_GATE_UP_SPLIT"), "0", StringComparison.Ordinal);

        protected unsafe void FuseGateUpWeights(int numLayers = 0)
        {
            if (numLayers <= 0)
                numLayers = Config.NumLayers;
            int fused = 0;
            int requantized = 0;
            // "layer:gateType+upType" - the type pair is the only place the
            // mismatch is ever surfaced, and it is what a future GGUF tripping a
            // DIFFERENT combination has to be diagnosed from.
            var splitLayers = new List<string>();
            var requantLayers = new List<string>();
            var keptMixedLayers = new List<string>();
            for (int l = 0; l < numLayers; l++)
            {
                string gateName = $"blk.{l}.ffn_gate.weight";
                string upName = $"blk.{l}.ffn_up.weight";
                string guName = $"blk.{l}.ffn_gate_up.weight";

                if (BonsaiHadamard != null && !BonsaiHadamard.CanFuseProjections(gateName, upName))
                    continue;

                if (_quantWeights.TryGetValue(gateName, out var gw) &&
                    _quantWeights.TryGetValue(upName, out var uw) &&
                    gw.Ne0 == uw.Ne0)
                {
                    // Mixed-quant "UD"/dynamic GGUFs (e.g. Qwen3.8 UD quants, where
                    // ffn_gate is IQ4_XS but ffn_up is Q5_K) store gate and up in
                    // different types, which a single fused tensor can't represent.
                    // Requantize the lower-fidelity side into the higher-fidelity
                    // type first, then fuse as usual.
                    if (gw.Scale != uw.Scale)
                    {
                        // Per-tensor sidecar scales differ: one fused tensor would
                        // need a single scalar for both halves. Keep the pair split;
                        // the split-FFN path applies each side's own scale.
                        splitLayers.Add($"{l}:scale {gw.Scale}!={uw.Scale}");
                        continue;
                    }

                    QuantizedWeight gateSrc = gw, upSrc = uw, requant = null;
                    if (gw.GgmlType != uw.GgmlType && KeepMixedGateUpSplitOnMlx)
                    {
                        keptMixedLayers.Add(
                            $"{l}:{(Runtime.GgmlTensorType)(uint)gw.GgmlType}+" +
                            $"{(Runtime.GgmlTensorType)(uint)uw.GgmlType}");
                        continue;
                    }
                    if (gw.GgmlType != uw.GgmlType)
                    {
                        requant = TryRequantizeForFusion(gw, uw, out bool requantIsGate);
                        if (requant == null)
                        {
                            // ggml refuses to quantize INTO IQ2_XXS / IQ2_XS /
                            // IQ1_S without an importance matrix
                            // (ggml_quantize_requires_imatrix), so a layer whose gate
                            // and up are both such types cannot be brought to a
                            // common type at load time. Collect and report ONCE below
                            // - ten per-layer WARNINGs read like ten problems - and
                            // let the report say whether this family can actually run
                            // the two projections separately (SupportsSplitGateUpFfn).
                            splitLayers.Add(
                                $"{l}:{(Runtime.GgmlTensorType)(uint)gw.GgmlType}+" +
                                $"{(Runtime.GgmlTensorType)(uint)uw.GgmlType}");
                            continue;
                        }
                        if (requantIsGate) gateSrc = requant; else upSrc = requant;
                        requantized++;
                        requantLayers.Add(
                            $"{l}:{(Runtime.GgmlTensorType)(uint)gw.GgmlType}+" +
                            $"{(Runtime.GgmlTensorType)(uint)uw.GgmlType}->" +
                            $"{(Runtime.GgmlTensorType)(uint)requant.GgmlType}");
                    }

                    // Where fusion IS possible it must produce a tensor at guName.
                    // (It is not always possible - see the split path above - and the
                    // FFN of every model that can load such a GGUF handles a missing
                    // guName by running gate and up separately.) If MLX view-fusion fails
                    // (gate/up not contiguous in the GGUF file), fall back to a
                    // copy. Cost is bounded — 2 tensors × per-layer, host memory
                    // released after the MLX device upload.
                    if (!TryCreateFusedQuantizedWeight(
                            SupportsSplitGateUpFfn, out QuantizedWeight fusedWeight, gateSrc, upSrc))
                    {
                        // Only reachable when the copy was declined for memory (see
                        // AllowWeightFusionCopies). Every family that reaches here runs
                        // gate and up as two matmuls when guName is absent, so drop the
                        // requantized temporary and leave the mapped sources in place.
                        if (requant != null)
                            requant.Dispose();
                        continue;
                    }

                    fusedWeight.Scale = gw.Scale;
                    _quantWeights[guName] = fusedWeight;
                    _quantWeights.Remove(gateName); gw.Dispose();
                    _quantWeights.Remove(upName); uw.Dispose();
                    if (requant != null && !ReferenceEquals(requant, fusedWeight))
                        requant.Dispose();
                    fused++;
                }
                else if (_weights.TryGetValue(gateName, out var gf) &&
                         _weights.TryGetValue(upName, out var uf))
                {
                    int gateDim = (int)gf.Sizes[0], upDim = (int)uf.Sizes[0];
                    int inDim = (int)gf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, gateDim + upDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, gateDim)) Ops.Copy(s0, gf);
                    using (var s1 = fusedTensor.Narrow(0, gateDim, upDim)) Ops.Copy(s1, uf);
                    _weights[guName] = fusedTensor;
                    _weights.Remove(gateName); gf.Dispose();
                    _weights.Remove(upName); uf.Dispose();
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine(requantized > 0
                    ? $"  Fused projections: {fused} Gate+Up ({requantized} mixed-quant layers requantized to a common type)"
                    : $"  Fused projections: {fused} Gate+Up");
            if (requantLayers.Count > 0)
            {
                // A dequantize+requantize of already-lossy weights is a real
                // (small) quality and VRAM change. It used to happen silently.
                Console.WriteLine(
                    $"    Requantized to fuse: {string.Join(", ", requantLayers)}");
            }
            if (splitLayers.Count > 0)
            {
                // ggml refuses to quantize INTO IQ2_XXS / IQ2_XS / IQ1_S without an
                // importance matrix, so a layer whose gate and up are BOTH such
                // types cannot be brought to a common type at load time.
                if (SupportsSplitGateUpFfn)
                {
                    Console.WriteLine(
                        $"  Split projections: {splitLayers.Count} of {numLayers} layers keep separate " +
                        "ffn_gate/ffn_up (mixed IQ quant types that would need an importance matrix to " +
                        "requantize). This model runs them as two matmuls instead of one, with identical " +
                        "output - no action needed.");
                }
                else
                {
                    Console.Error.WriteLine(
                        $"  WARNING: {splitLayers.Count} of {numLayers} layers have mixed-IQ ffn_gate/ffn_up that " +
                        "cannot be fused (requantizing into IQ2_XXS/IQ2_XS/IQ1_S needs an importance matrix), and " +
                        $"this architecture ({Config?.Architecture ?? "unknown"}) has no split-FFN path. Those layers " +
                        "have no usable FFN weight and generation will be wrong or will fail. Use a GGUF whose " +
                        "ffn_gate and ffn_up share a quant type - most non-UD quants do.");
                }
                Console.WriteLine($"    Layers: {string.Join(", ", splitLayers)}");
            }
            if (keptMixedLayers.Count > 0)
            {
                Console.WriteLine(
                    $"  Split projections: {keptMixedLayers.Count} of {numLayers} mixed-quant ffn_gate/ffn_up pairs run " +
                    "as two matmuls in their own types on MLX (no requantization; TS_MLX_MIXED_GATE_UP_SPLIT=0 fuses them).");
            }
            ReportDeclinedFusionCopies();
        }

        /// <summary>
        /// Produce a copy of the lower-fidelity side of a mixed-type gate/up pair,
        /// requantized to the other side's type so the pair can be fused. Prefers
        /// upcasting (smaller row size → larger); tries the opposite direction when
        /// the preferred target can't be produced without an importance matrix.
        /// Returns null when neither direction is possible.
        /// </summary>
        private unsafe QuantizedWeight TryRequantizeForFusion(QuantizedWeight gw, QuantizedWeight uw, out bool requantIsGate)
        {
            requantIsGate = false;
            if (!gw.HasHostData || !uw.HasHostData || gw.Ne0 != uw.Ne0)
                return null;

            long gRow = NativeDequant.RowSize(gw.GgmlType, gw.Ne0);
            long uRow = NativeDequant.RowSize(uw.GgmlType, uw.Ne0);
            QuantizedWeight lower = gRow <= uRow ? gw : uw;
            QuantizedWeight higher = gRow <= uRow ? uw : gw;

            QuantizedWeight result = TryRequantizeWeight(lower, higher.GgmlType);
            if (result != null)
            {
                requantIsGate = ReferenceEquals(lower, gw);
                return result;
            }

            result = TryRequantizeWeight(higher, lower.GgmlType);
            if (result != null)
            {
                requantIsGate = ReferenceEquals(higher, gw);
                return result;
            }

            return null;
        }

        /// <summary>
        /// Dequantize a weight row-chunk-wise to FP32 and requantize it into
        /// <paramref name="targetType"/>. Returns null when the conversion is not
        /// possible (imatrix-only target, or no native quantize available).
        /// </summary>
        private unsafe QuantizedWeight TryRequantizeWeight(QuantizedWeight src, int targetType)
        {
            try
            {
                long ne0 = src.Ne0, ne1 = src.Ne1;
                long srcRow = NativeDequant.RowSize(src.GgmlType, ne0);
                long dstRow = NativeDequant.RowSize(targetType, ne0);
                byte[] dstBuf = new byte[checked(dstRow * ne1)];
                const int ChunkRows = 512;
                int numChunks = (int)((ne1 + ChunkRows - 1) / ChunkRows);
                IntPtr srcBase = src.Data;
                bool failed = false;
                GCHandle hDst = GCHandle.Alloc(dstBuf, GCHandleType.Pinned);
                try
                {
                    IntPtr dstBase = hDst.AddrOfPinnedObject();
                    Parallel.For(0, numChunks,
                        () => new float[(long)ChunkRows * ne0],
                        (ci, state, f32) =>
                        {
                            if (Volatile.Read(ref failed))
                            {
                                state.Stop();
                                return f32;
                            }
                            long r = (long)ci * ChunkRows;
                            long rows = Math.Min(ChunkRows, ne1 - r);
                            fixed (float* pF32 = f32)
                            {
                                NativeDequant.DequantizeToFloat32Native(src.GgmlType,
                                    (IntPtr)((byte*)srcBase.ToPointer() + r * srcRow), (IntPtr)pF32, rows * ne0);
                                long written = GgmlGgufTensorDequant.QuantizeFloat32RowsOrZero(targetType,
                                    (IntPtr)pF32, (IntPtr)((byte*)dstBase.ToPointer() + r * dstRow), rows, ne0);
                                if (written != rows * dstRow)
                                {
                                    Volatile.Write(ref failed, true);
                                    state.Stop();
                                }
                            }
                            return f32;
                        },
                        _ => { });
                }
                finally
                {
                    hDst.Free();
                }

                if (failed)
                    return null;

                return new QuantizedWeight(dstBuf, targetType, ne0, ne1);
            }
            catch (Exception ex) when (IsRequantizeUnavailable(ex))
            {
                return null;
            }
        }

        private static bool IsRequantizeUnavailable(Exception ex)
        {
            if (ex is AggregateException agg)
                return agg.InnerExceptions.Count > 0 && agg.InnerExceptions.All(IsRequantizeUnavailable);
            return ex is DllNotFoundException or EntryPointNotFoundException or NotSupportedException;
        }

    }
}
