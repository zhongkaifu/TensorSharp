// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Native whole-model execution for Muse Spark on the GGML GPU backends.
//
// The 816B checkpoint is ~6 GiB per layer, so it is spread across every GPU with
// the leading layers' experts in system RAM. The per-op managed forward spends
// most of its time in host round trips at that scale; the native executor
// (TensorSharp.GGML.Native/ggml_ops_muse_spark_exec.cpp) loads the split GGUF
// itself, owns every sequence slot's K/V rows on the devices that read them, and
// submits ONE graph per ubatch. --layer-split N spreads whole layers over N
// GPUs; --tp N on ggml_cuda splits every layer over N GPUs through ggml's meta
// backend (llama.cpp's -sm tensor mechanism). The managed per-op path remains
// the reference implementation: TS_MUSE_SPARK_NATIVE=0 selects it on a GGML
// backend (also for --tp), and TS_MUSE_SPARK_NATIVE=1 opts the GGML CPU backend
// into the native executor.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models;

public partial class MuseSparkModel : ISoloPrefillBudgetModel
{
    private IntPtr _native;
    private readonly object _nativeSync = new();
    private int _nativeUbatch;
    private int _nativePipelineUbatch;   // pipelined prefill micro-batch rows (adaptive: their upper bound); 0 when off
    private int _nativeTpDegree;         // GPUs the native executor splits every layer over; 0 for a layer split

    /// <summary>A lone prompt's prefill step. Each native call fills and drains the GPU pipeline
    /// once: on 10x A40 one 8192-token call keeps about 90% of a 16K call's throughput (a 4K call
    /// about 70%), and a request that arrives meanwhile waits for one step of about 6 s rather
    /// than 10 s. A remainder of at most a quarter step joins the step before it, so no short
    /// call fills and drains the pipeline alone. 0 (the scheduler's own step) when the call would
    /// not run pipelined: pipelined prefill off, or a context whose masks exceed its budget.</summary>
    public int PreferredSoloPrefillTokens(int start, int remaining)
    {
        // (The scheduler keeps its own step when the step would carry image rows.)
        if (_nativePipelineUbatch <= 0) return 0;
        lock (_nativeSync)
        {
            if (_native == IntPtr.Zero) return 0;
            IntPtr handle = _native;
            return SoloPrefillStep(start, remaining, (p0, n) => GgmlMuseSparkNative.PipelinePlan(handle, p0, n));
        }
    }

    /// <summary>The lone-prompt step rule: <see cref="SoloPrefillTokens"/> tokens, a remainder of at most
    /// a quarter step joined to the step before it, and 0 when <paramref name="pipelinePlan"/> says the call
    /// would not run pipelined.</summary>
    internal static int SoloPrefillStep(int start, int remaining, Func<long, long, int> pipelinePlan)
    {
        if (remaining <= 0 || start < 0) return 0;
        int step = Math.Min(remaining, SoloPrefillTokens);
        int rest = remaining - step;
        if (rest > 0 && rest <= SoloPrefillTokens / 4) step = remaining;
        return pipelinePlan(start, step) > 0 ? step : 0;
    }

    internal const int SoloPrefillTokens = 8192;

    /// <summary>True when the native whole-model executor drives this instance.</summary>
    internal bool UsesNativeExecutor => _native != IntPtr.Zero;

    // requestId -> native slot (guarded by _nativeSync)
    private Dictionary<string, int> _slotByRequest;
    private int _primarySlot;
    private string _activeSlotKey;
    private bool _batchedChunkingLatched;

    private static bool NativeRequested(BackendType backend, int tpDegree, ITensorParallelGroup tpGroup)
    {
        if (tpGroup?.IsActive == true) return false;
        string flag = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_NATIVE");
        if (string.Equals(flag, "0", StringComparison.Ordinal)) return false;
        // Native tensor parallelism runs on ggml's meta backend over CUDA GPUs.
        if (tpDegree > 1) return backend == BackendType.GgmlCuda;
        return backend switch
        {
            BackendType.GgmlCuda or BackendType.GgmlVulkan => true,
            BackendType.GgmlCpu => string.Equals(flag, "1", StringComparison.Ordinal),
            _ => false,
        };
    }

    private static string NativeBackendName(BackendType backend) => backend switch
    {
        BackendType.GgmlCuda => "CUDA",
        BackendType.GgmlVulkan => "Vulkan",
        BackendType.GgmlMetal => "Metal",
        BackendType.GgmlCpu => "CPU",
        _ => null,
    };

    private static int ParseEnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;

    /// <summary>GPUs the native executor can span (MAX_GPUS in ggml_ops_muse_spark_exec.cpp).</summary>
    internal const int NativeMaxGpus = 16;

    /// <summary>The GPU count the native executor is asked for: <c>--layer-split N</c>, or more when
    /// TS_MUSE_SPARK_LAYER_SPLIT lists more per-GPU counts. A count it cannot honour is a load refusal
    /// rather than a run on fewer GPUs. <paramref name="available"/> is null on the CPU backend.</summary>
    internal static int ResolveNativeGpuCount(int layerSplitDegree, string explicitSplit, int? available, string backendName)
    {
        int requested = Math.Max(1, layerSplitDegree);
        if (!string.IsNullOrWhiteSpace(explicitSplit))
            requested = Math.Max(requested, explicitSplit.Split(',', StringSplitOptions.RemoveEmptyEntries).Length);
        if (available is int visible)
        {
            if (requested > NativeMaxGpus)
                throw new ModelLoadRefusedException(
                    $"[muse-spark] {requested} GPUs requested; the native executor spans at most {NativeMaxGpus}.");
            if (visible < requested)
                throw new ModelLoadRefusedException(
                    $"[muse-spark] {requested} GPU(s) requested but the GGML {backendName} backend sees only {visible}.");
        }
        return requested;
    }

    /// <summary>Per-op-path settings the native executor does not read. Under --tp they used to tune the
    /// managed tensor-parallel path, so they are named when ignored rather than dropped silently.</summary>
    internal static readonly string[] PerOpOnlyOptionNames =
    [
        "TS_MUSE_SPARK_PHASE_TIMING", "TS_MUSE_SPARK_FUSE_QK_PREP", "TS_MUSE_SPARK_REUSE_NORM_SCRATCH",
        "TS_MUSE_SPARK_KV_UPLOAD_2D", "TS_MUSE_SPARK_FUSE_RESIDUALS", "TS_MUSE_SPARK_TP_HOST_POOL",
        "TS_MUSE_SPARK_PREFILL_CHUNK", "TS_MUSE_SPARK_SCRATCH_RESERVE_MIB", "TS_MUSE_SPARK_RELEASE_GPU_SOURCE_PAGES",
    ];

    private void InitNativeExecutor(string ggufPath, BackendType backend, int layerSplitDegree, int tpDegree)
    {
        string split = Environment.GetEnvironmentVariable("TS_MUSE_SPARK_LAYER_SPLIT");
        int? available = null;
        if (backend is BackendType.GgmlCuda or BackendType.GgmlVulkan)
            available = GgmlBasicOps.GetGpuDeviceCount(backend == BackendType.GgmlCuda ? GgmlBackendType.Cuda : GgmlBackendType.Vulkan);
        bool tensorParallel = tpDegree > 1;
        if (tensorParallel)
        {
            if (!string.IsNullOrWhiteSpace(split))
                throw new ModelLoadRefusedException(
                    "[muse-spark] --tp cannot be combined with TS_MUSE_SPARK_LAYER_SPLIT (a layer split); use one or the other.");
            if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEVICES")))
                throw new ModelLoadRefusedException(
                    "[muse-spark] the native executor's --tp uses the first N CUDA GPUs and does not read TENSORSHARP_TP_DEVICES; " +
                    "select GPUs with CUDA_VISIBLE_DEVICES, or set TS_MUSE_SPARK_NATIVE=0 for the per-op path that reads it.");
            if (tpDegree > Config.NumKVHeads)
                throw new ModelLoadRefusedException(
                    $"[muse-spark] --tp {tpDegree} exceeds the model's {Config.NumKVHeads} KV heads; every GPU needs at least one.");
            string[] ignored = PerOpOnlyOptionNames.Where(n => !string.IsNullOrEmpty(Environment.GetEnvironmentVariable(n))).ToArray();
            if (ignored.Length > 0)
                Console.Error.WriteLine($"[muse-spark] ignoring {string.Join(", ", ignored)}: only the per-op path reads " +
                    "them (TS_MUSE_SPARK_NATIVE=0); the native tensor-parallel executor runs without them.");
        }
        int requested = tensorParallel
            ? ResolveNativeGpuCount(tpDegree, null, available, NativeBackendName(backend))
            : ResolveNativeGpuCount(layerSplitDegree, split, available, NativeBackendName(backend));

        int nUbatch = ParseEnvInt("TS_MUSE_SPARK_UBATCH", 2048);
        int nThreads = ParseEnvInt("TS_MUSE_SPARK_THREADS", 0);
        int cpuMoe = !MoeOffload.IsExplicitlySet ? 0
            : MoeOffload.AllLayers ? Config.NumLayers : MoeOffload.CpuMoeLayers;
        if (Environment.GetEnvironmentVariable("TS_MUSE_SPARK_CPU_MOE") is string auto &&
            string.Equals(auto, "auto", StringComparison.OrdinalIgnoreCase))
            cpuMoe = -1;
        int maxContext = ResolveConfiguredContextLength();
        bool hardLimit = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT"));
        ForwardNativeOptions();

        IntPtr handle = GgmlMuseSparkNative.LoadModel(ggufPath, requested, maxContext, nUbatch, nThreads, cpuMoe,
            NativeBackendName(backend), hardLimit, tensorParallel ? null : split, tensorParallel ? tpDegree : 0);
        if (handle == IntPtr.Zero)
            throw NativeLoadRefused("muse-spark", ggufPath,
                "TS_MUSE_SPARK_NATIVE=0 selects the per-op path instead of the native executor.");
        _native = handle;
        _nativeUbatch = nUbatch;
        _maxContextLength = GgmlMuseSparkNative.CtxSize(handle);
        int vocab = GgmlMuseSparkNative.VocabSize(handle);
        if (vocab > 0) Config.VocabSize = vocab;
        _logitsBuffer = new float[Config.VocabSize];
        _primarySlot = 0;
        long[] stats = GgmlMuseSparkNative.GetStats(handle);
        _nativePipelineUbatch = stats != null ? (int)stats[9] : 0;
        _nativeTpDegree = stats != null ? (int)stats[11] : 0;
        if (tensorParallel && _nativeTpDegree != tpDegree)
            throw new ModelLoadRefusedException(
                $"[muse-spark] --tp {tpDegree} was requested but the native executor reports tensor parallelism over {_nativeTpDegree} GPU(s).");
        string hostLayers = stats != null ? stats[3].ToString() : cpuMoe < 0 ? "auto" : cpuMoe.ToString();
        if (tensorParallel)
            Console.WriteLine($"Muse Spark native executor: {Config.NumLayers} layers tensor-parallel over {_nativeTpDegree} GPUs, " +
                $"context {_maxContextLength}, ubatch {nUbatch}.");
        else
            Console.WriteLine($"Muse Spark native executor: {Config.NumLayers} layers over " +
                $"{(stats != null ? stats[4].ToString() : requested.ToString())} GPU(s), " +
                $"context {_maxContextLength}, ubatch {nUbatch}, layers with host experts {hostLayers}.");
    }

    /// <summary>Native executor counters (<see cref="GgmlMuseSparkNative.GetStats"/>); null without the executor.</summary>
    internal long[] NativeStatsForTests()
    {
        lock (_nativeSync) return _native == IntPtr.Zero ? null : GgmlMuseSparkNative.GetStats(_native);
    }

    /// <summary>Settings the native executor reads. Process-environment values set from .NET never reach
    /// native getenv on Linux, so the managed view is forwarded explicitly (and a cleared value clears it).</summary>
    internal static readonly string[] NativeOptionNames =
    [
        "TS_MUSE_SPARK_PLAN_SLOTS", "TS_MUSE_SPARK_VRAM_RESERVE_MB", "TS_MUSE_SPARK_DEV0_RESERVE_MB",
        "TS_MUSE_SPARK_SWA_REWIND", "TS_MUSE_SPARK_SWA_FULL", "TS_MUSE_SPARK_FUSE_QKVG", "TS_MUSE_SPARK_KV_PAD",
        "TS_MUSE_SPARK_BD_KV_PAD", "TS_MUSE_SPARK_LOAD_CHUNK_MB", "TS_MUSE_SPARK_LOAD_THREADS",
        "TS_MUSE_SPARK_OP_OFFLOAD", "TS_MUSE_SPARK_GRAPH_CACHE", "TS_MUSE_SPARK_NODES_PER_LAYER",
        "TS_MUSE_SPARK_EXEC_DEBUG", "TS_MUSE_SPARK_WARM_HOST", "TS_MUSE_SPARK_DROP_UPLOAD_CACHE",
        "TS_MUSE_SPARK_SLOT_HEADROOM_MB", "TS_MUSE_SPARK_CPU_REPACK", "TS_MUSE_SPARK_PROFILE", "TS_CPU_MOE_THREADS",
        "TS_MUSE_SPARK_HOST_NUMA", "TS_MUSE_SPARK_NUMA_BATCH_GB", "TS_MUSE_SPARK_SLOT_INIT_ROWS",
        "TS_MUSE_SPARK_PLAN_CTX_TOKENS", "TS_MUSE_SPARK_HOST_MOE_TOKENS", "TS_MUSE_SPARK_ADMIT_GRAPH_MB",
        "TS_MUSE_SPARK_WARM_POOLS", "TS_MUSE_SPARK_PIPELINE", "TS_MUSE_SPARK_PIPELINE_UBATCH",
    ];

    private static void ForwardNativeOptions()
    {
        foreach (string name in NativeOptionNames)
            GgmlMuseSparkNative.SetOption(name, Environment.GetEnvironmentVariable(name));
    }

    /// <summary>Pending image rows overlapping this call become native overrides (row index relative to the call).</summary>
    private void QueueNativeVisionRows(int start, int count)
    {
        int end = start + count;
        for (int i = 0; i < _pendingVisionEmbeddingsList.Count;)
        {
            var (embeddings, position) = _pendingVisionEmbeddingsList[i];
            int spanEnd = checked(position + (int)embeddings.Sizes[0]);
            int from = Math.Max(position, start), to = Math.Min(spanEnd, end);
            if (from < to)
            {
                using var rows = embeddings.Narrow(0, from - position, to - from);
                float[] data = rows.GetElementsAsFloat(checked((to - from) * Config.HiddenSize));
                if (!GgmlMuseSparkNative.QueueVisionRows(_native, data, to - from, from - start))
                    throw new InvalidOperationException("Muse Spark native executor refused image rows.");
            }
            if (spanEnd <= end)
            {
                embeddings.Dispose();
                _pendingVisionEmbeddingsList.RemoveAt(i);
            }
            else
                ++i;
        }
    }

    private float[] ForwardNative(int[] tokens, bool allRows = false, float[] logitsOut = null)
    {
        lock (_nativeSync)
        {
            int rows = allRows ? tokens.Length : 1;
            float[] output = logitsOut;
            if (output == null)
            {
                if (!allRows && (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize))
                    _logitsBuffer = new float[Config.VocabSize];
                output = allRows ? new float[checked(rows * Config.VocabSize)] : _logitsBuffer;
            }
            int start = GgmlMuseSparkNative.NPast(_native);
            if ((long)start + tokens.Length > _maxContextLength)
                throw new InvalidOperationException("Muse Spark context length exceeded.");
            BeforeGraphCall();
            bool ok;
            try
            {
                if (HasPendingVisionEmbeddings) QueueNativeVisionRows(start, tokens.Length);
                _forwardSw.Start();
                ok = GgmlMuseSparkNative.Forward(_native, tokens, output, allRows);
            }
            finally
            {
                _forwardSw.Stop();
                _forwardCount++;
                AfterGraphCall();
            }
            if (!ok)
            {
                GgmlMuseSparkNative.ClearVisionRows(_native);
                throw new InvalidOperationException("Muse Spark native forward failed (see stderr).");
            }
            _cacheSeqLen = GgmlMuseSparkNative.NPast(_native);
            return output;
        }
    }

    private void ResetNative()
    {
        lock (_nativeSync)
        {
            if (!GgmlMuseSparkNative.ResetChecked(_native))
                throw new InvalidOperationException("Muse Spark native reset failed.");
            _cacheSeqLen = 0;
        }
    }

    private bool RewindNative(int tokenCount)
    {
        lock (_nativeSync)
        {
            if (!GgmlMuseSparkNative.Rewind(_native, tokenCount)) return false;
            _cacheSeqLen = tokenCount;
            return true;
        }
    }

    private void WarmUpNative()
    {
        // One short forward plus a decode step allocates the scheduler's graph
        // buffers and captures the decode graphs before the first request.
        try
        {
            int bos = Tokenizer?.BosTokenId ?? 0;
            ForwardNative(new[] { bos, bos });
            ForwardNative(new[] { bos });
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[muse-spark] warmup forward failed: {ex.Message}");
        }
        finally
        {
            ResetNative();
        }
    }

    private void DisposeNative()
    {
        lock (_nativeSync)
        {
            if (_native != IntPtr.Zero)
            {
                GgmlMuseSparkNative.Free(_native);
                _native = IntPtr.Zero;
            }
        }
    }

    // ------------------------------------------------------------------ slots

    private readonly struct NativeSlotStore : SlotRetention.ISlotStore
    {
        private readonly IntPtr _h;
        public NativeSlotStore(IntPtr handle) => _h = handle;
        public int Alloc() => GgmlMuseSparkNative.SlotAlloc(_h);
        public bool Select(int slot) => GgmlMuseSparkNative.SetActiveSlot(_h, slot);
        public bool Free(int slot) => GgmlMuseSparkNative.SlotFree(_h, slot);
        public int ActiveHead() => GgmlMuseSparkNative.NPast(_h);
        public bool RewindActive(int tokens) => GgmlMuseSparkNative.Rewind(_h, tokens);
        public bool ResetActive() => GgmlMuseSparkNative.ResetChecked(_h);
        public bool SetReclaimable(int[] slots) => GgmlMuseSparkNative.SetReclaimableSlots(_h, slots);
        public int TakeReclaimed(int[] buffer) => GgmlMuseSparkNative.TakeReclaimedSlots(_h, buffer);
    }

    public bool SupportsPerSequenceFusedForward => UsesNativeExecutor;

    public bool HasFusedSequenceCache(string requestId)
    {
        lock (_nativeSync)
            return requestId != null && _slotByRequest != null && _slotByRequest.ContainsKey(requestId);
    }

    public bool BindSequenceCache(string requestId)
    {
        if (string.IsNullOrEmpty(requestId)) throw new ArgumentException("RequestId required", nameof(requestId));
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor) throw new InvalidOperationException("Per-request slots require the native Muse Spark executor.");
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            bool fresh = false;
            if (!_slotByRequest.TryGetValue(requestId, out int slot))
            {
                slot = store.Alloc();
                if (slot < 0) slot = SlotRetention.TakeIdlePrimary(_slotByRequest, ref _primarySlot, _activeSlotKey, store);
                if (slot < 0 && _retainedSlots is { Count: > 0 }) slot = AllocSlotReclaimingRetained();
                if (slot < 0)
                    throw new SequenceSlotUnavailableException("Muse Spark has no device memory for another sequence slot.");
                _slotByRequest[requestId] = slot;
                fresh = true;
            }
            if (!store.Select(slot))
                throw new InvalidOperationException($"Muse Spark slot {slot} missing for request {requestId}.");
            _activeSlotKey = requestId;
            _cacheSeqLen = store.ActiveHead();
            return fresh;
        }
    }

    public void AdoptPrimaryCacheToFused(string requestId)
    {
        if (string.IsNullOrEmpty(requestId)) return;
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor) return;
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            if (_activeSlotKey != null || _slotByRequest.ContainsKey(requestId)) return;
            _slotByRequest[requestId] = _primarySlot;
            _activeSlotKey = requestId;
            _primarySlot = -1;
        }
    }

    /// <summary>A new slot, freeing retained slots while memory has no room for one: donated end states
    /// (each serves at most one later request) go first, oldest first; copy-captured checkpoints (which
    /// every later conversation sharing the prefix clones) only when nothing else is left.</summary>
    private int AllocSlotReclaimingRetained()
    {
        var store = new NativeSlotStore(_native);
        int slot = store.Alloc();
        while (slot < 0 && _retainedSlots is { Count: > 0 })
        {
            string key = ReclaimOrder().FirstOrDefault();
            if (key == null || !store.Free(_retainedSlots[key].Slot)) break;
            _retainedSlots.Remove(key);
            _checkpointKeys.Remove(key);
            _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
            slot = store.Alloc();
        }
        return slot;
    }

    /// <summary>Retained payload keys in the order memory takes them back.</summary>
    private IEnumerable<string> ReclaimOrder()
    {
        if (_retainedSlots == null) yield break;
        foreach (string k in _retainedSlots.Keys) if (!_checkpointKeys.Contains(k)) yield return k;
        foreach (string k in _retainedSlots.Keys) if (_checkpointKeys.Contains(k)) yield return k;
    }

    public void RestorePrimaryCache()
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || _activeSlotKey == null) return;
            if (_primarySlot < 0)
            {
                _primarySlot = AllocSlotReclaimingRetained();
                if (_primarySlot < 0)
                    throw new SequenceSlotUnavailableException("Muse Spark primary-slot allocation failed (device memory exhausted?).");
            }
            var store = new NativeSlotStore(_native);
            if (!store.Select(_primarySlot))
                throw new InvalidOperationException($"Muse Spark primary slot {_primarySlot} missing.");
            _activeSlotKey = null;
            _cacheSeqLen = store.ActiveHead();
        }
    }

    public void OnSequenceReleased(string requestId)
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || _slotByRequest == null || string.IsNullOrEmpty(requestId) ||
                !_slotByRequest.TryGetValue(requestId, out int slot))
                return;
            var store = new NativeSlotStore(_native);
            if (string.Equals(_activeSlotKey, requestId, StringComparison.Ordinal))
            {
                if (_primarySlot < 0)
                {
                    if (SlotRetention.ReleaseActiveSlotAsPrimary(_slotByRequest, requestId, ref _primarySlot, ref _activeSlotKey, store))
                        _cacheSeqLen = 0;
                    return;
                }
                RestorePrimaryCache();
            }
            _slotByRequest.Remove(requestId);
            store.Free(slot);
        }
    }

    public string BatchedFusedDecodeDeclineReason => null;

    public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || _slotByRequest == null) return false;
            int n = requestIds.Count;
            if (n < 2) return false;
            var slots = new int[n];
            for (int i = 0; i < n; i++)
                if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i])) return false;
            int vocab = Config.VocabSize;
            if ((long)n * vocab > int.MaxValue) return false;
            var flat = new float[n * vocab];
            BeforeGraphCall();
            try
            {
                _forwardSw.Start();
                if (!ForwardBatchedDecodeChunks(slots, tokens, positions, flat, vocab)) return false;
            }
            finally
            {
                _forwardSw.Stop();
                _forwardCount++;
                AfterGraphCall();
            }
            for (int i = 0; i < n; i++)
            {
                var row = new float[vocab];
                Array.Copy(flat, (long)i * vocab, row, 0, vocab);
                outLogits[i] = row;
            }
            // The active slot may have advanced as one of the batch.
            _cacheSeqLen = GgmlMuseSparkNative.NPast(_native);
            return true;
        }
    }

    /// <summary>Concurrent prompt chunks in shared graphs: each native call takes at most one ubatch of
    /// tokens, drawn from every sequence that still has some, so arrivals prefill together.</summary>
    public bool TryForwardBatchedFusedPrefill(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens,
        int[] positions, float[][] outLogits)
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || requestIds == null || tokens == null || positions == null) return false;
            int n = requestIds.Count;
            if (n < 2 || tokens.Count != n || positions.Length != n || outLogits.Length < n || n > 16) return false;
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            var slots = new int[n];
            for (int i = 0; i < n; i++)
            {
                if (requestIds[i] == null || tokens[i] == null || tokens[i].Length == 0) return false;
                if (!_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                {
                    if (positions[i] != 0) return false;
                    int slot = store.Alloc();
                    if (slot < 0 && _retainedSlots is { Count: > 0 }) slot = AllocSlotReclaimingRetained();
                    if (slot < 0) return false;   // the per-sequence loop reports the missing slot
                    _slotByRequest[requestIds[i]] = slot;
                    slots[i] = slot;
                }
                if (GgmlMuseSparkNative.SlotNPast(_native, slots[i]) != positions[i]) return false;
            }

            int ubatch = Math.Max(n, GgmlMuseSparkNative.Ubatch(_native));
            int vocab = Config.VocabSize;
            var done = new int[n];
            bool progressed = false;
            BeforeGraphCall();
            try
            {
                while (true)
                {
                    var callSlots = new List<int>(n); var callLens = new List<int>(n);
                    var callPos = new List<int>(n); var callIdx = new List<int>(n);
                    var flat = new List<int>(ubatch);
                    int budget = ubatch;
                    for (int i = 0; i < n && budget > 0; i++)
                    {
                        int left = tokens[i].Length - done[i];
                        if (left <= 0) continue;
                        int take = Math.Min(left, budget);
                        callSlots.Add(slots[i]); callLens.Add(take); callPos.Add(positions[i] + done[i]); callIdx.Add(i);
                        for (int t = 0; t < take; t++) flat.Add(tokens[i][done[i] + t]);
                        budget -= take;
                    }
                    if (callSlots.Count == 0) break;
                    var logits = new float[checked(callSlots.Count * vocab)];
                    _forwardSw.Start();
                    bool ok = GgmlMuseSparkNative.ForwardSegments(_native, callSlots.ToArray(), callLens.ToArray(),
                        flat.ToArray(), callPos.ToArray(), logits);
                    _forwardSw.Stop();
                    _forwardCount++;
                    if (!ok)
                    {
                        if (!progressed) return false;
                        throw new InvalidOperationException("Muse Spark batched prefill failed after partial progress (see stderr).");
                    }
                    progressed = true;
                    for (int c = 0; c < callIdx.Count; c++)
                    {
                        int i = callIdx[c];
                        done[i] += callLens[c];
                        if (done[i] == tokens[i].Length)
                        {
                            var row = new float[vocab];
                            Array.Copy(logits, (long)c * vocab, row, 0, vocab);
                            outLogits[i] = row;
                        }
                    }
                }
            }
            finally
            {
                AfterGraphCall();
            }
            _cacheSeqLen = GgmlMuseSparkNative.NPast(_native);
            return true;
        }
    }

    private bool ForwardBatchedDecodeChunks(int[] slots, int[] tokens, int[] positions, float[] flat, int vocab)
    {
        const int MaxPerCall = 16;
        int n = slots.Length;
        if (n <= MaxPerCall) return GgmlMuseSparkNative.ForwardBatchedDecode(_native, slots, tokens, positions, flat);
        if (_batchedChunkingLatched) return false;
        int chunks = (n + MaxPerCall - 1) / MaxPerCall;
        int baseSize = n / chunks, rem = n % chunks, off = 0;
        for (int c = 0; c < chunks; c++)
        {
            int len = baseSize + (c < rem ? 1 : 0);
            var cs = slots.AsSpan(off, len).ToArray();
            var ct = tokens.AsSpan(off, len).ToArray();
            var cp = positions.AsSpan(off, len).ToArray();
            var cf = new float[len * vocab];
            if (!GgmlMuseSparkNative.ForwardBatchedDecode(_native, cs, ct, cp, cf))
            {
                if (c > 0)
                {
                    _batchedChunkingLatched = true;
                    Console.Error.WriteLine($"[muse-spark batched-decode] chunk {c + 1}/{chunks} failed mid-step; chunked batching disabled");
                }
                return false;
            }
            Array.Copy(cf, 0, flat, (long)off * vocab, (long)len * vocab);
            off += len;
        }
        return true;
    }

    // ------------------------------------------------------------------ prefix-cache donation

    private OrderedDictionary<string, SlotRetention.RetainedSlot> _retainedSlots;
    // Retained payloads made by copying a cache (shared-prefix checkpoints): kept longest under memory pressure.
    private readonly HashSet<string> _checkpointKeys = new(StringComparer.Ordinal);
    private int[] _reclaimableSynced = Array.Empty<int>();
    private readonly int[] _reclaimedIds = new int[16];
    private IPrefixPayloadSink _prefixCacheSink;

    private void BeforeGraphCall()
    {
        if (!UsesNativeExecutor) return;
        // The native side frees these, in this order, when a graph does not fit.
        int[] order = ReclaimOrder().Select(k => _retainedSlots[k].Slot).ToArray();
        if (!order.AsSpan().SequenceEqual(_reclaimableSynced))
        {
            if (!GgmlMuseSparkNative.SetReclaimableSlots(_native, order))
                throw new InvalidOperationException("The executor refused the list of retained slots.");
            _reclaimableSynced = order;
        }
    }

    private void AfterGraphCall()
    {
        if (!UsesNativeExecutor) return;
        var reclaimed = new List<string>(0);
        SlotRetention.DrainReclaimed(_retainedSlots, new NativeSlotStore(_native), _reclaimedIds, reclaimed);
        foreach (string key in reclaimed)
        {
            _checkpointKeys.Remove(key);
            _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
        }
    }

    private PrefixCacheCapabilities NativePrefixCacheCapabilities() => new()
    {
        Class = FamilyClass.N,
        NamespaceFingerprint = KVStateFingerprint,
        // A slot copy is a device-to-device copy of its rows, so the prompt every
        // conversation shares (system prompt, tool schemas) can be checkpointed
        // once and cloned into each new conversation instead of re-prefilled.
        EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true,
        PrimaryResident = true,
        MinRetainTokens = 32,
        // Global layers rewind exactly; sliding layers keep a ring that holds a
        // bounded margin beyond their window (any depth before the ring wraps).
        // The native rewind accepts or refuses at execution time, and a refusal
        // re-prefills, so the model decides rather than a fixed planner cap.
        Truncation = TruncationKind.ModelDecides,
        TruncationGranularity = 1,
        RewindCapTokens = int.MaxValue,
        Pages = PageSupport.None,
        ReuseAcrossMediaSpan = false,
        Persistable = false,
        MaxRetainedNativeSlots = 0,
    };

    private bool NativeTryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        lock (_nativeSync)
        {
            if (_slotByRequest == null) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            if (!SlotRetention.RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, true,
                    ref _primarySlot, ref _activeSlotKey, store))
                return false;
            _cacheSeqLen = store.ActiveHead();
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    private bool NativeTryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        lock (_nativeSync)
        {
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            if (!SlotRetention.ConvertPrimarySlot(_retainedSlots, payloadKey, length, ref _primarySlot, ref _activeSlotKey, store))
                return false;
            _cacheSeqLen = store.ActiveHead();
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    /// <summary>The native slot holding <paramref name="requestId"/>'s cache: its own slot when it has one
    /// (also when it is not the active one, as after a batched step), else the active slot.</summary>
    private int SlotOfRequestOrActive(string requestId)
    {
        if (requestId != null && _slotByRequest != null && _slotByRequest.TryGetValue(requestId, out int slot)) return slot;
        if (_activeSlotKey != null && _slotByRequest != null && _slotByRequest.TryGetValue(_activeSlotKey, out slot)) return slot;
        return _primarySlot;
    }

    /// <summary>Copy a slot into a fresh one, reclaiming retained slots for memory when needed; -1 on failure.</summary>
    private int CopyIntoNewSlot(int source)
    {
        if (source < 0) return -1;
        int fresh = GgmlMuseSparkNative.SlotAlloc(_native);
        if (fresh < 0 && _retainedSlots is { Count: > 0 }) fresh = AllocSlotReclaimingRetained();
        if (fresh < 0) return -1;
        if (!GgmlMuseSparkNative.SlotCopy(_native, source, fresh))
        {
            GgmlMuseSparkNative.SlotFree(_native, fresh);
            return -1;
        }
        return fresh;
    }

    private bool NativeTryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        lock (_nativeSync)
        {
            if (string.IsNullOrEmpty(payloadKey)) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            if (_retainedSlots.ContainsKey(payloadKey)) return false;
            int source = SlotOfRequestOrActive(requestId);
            int tokens = source >= 0 ? GgmlMuseSparkNative.SlotNPast(_native, source) : -1;
            if (tokens <= 0) return false;
            int copy = CopyIntoNewSlot(source);
            if (copy < 0) return false;
            _retainedSlots.Add(payloadKey, new SlotRetention.RetainedSlot(copy, tokens));
            _checkpointKeys.Add(payloadKey);
            footprint = NativeFootprint(_retainedSlots[payloadKey]);
            return true;
        }
    }

    private bool NativeTryMaterialize(in MaterializeRequest request)
    {
        if (request.Op == MaterializeOp.Clone)
        {
            lock (_nativeSync)
            {
                if (_retainedSlots == null || !_retainedSlots.TryGetValue(request.PayloadKey, out var payload)
                    || payload.Tokens != request.PayloadTokens || request.TargetTokens < 0 || request.TargetTokens > payload.Tokens
                    || string.IsNullOrEmpty(request.TargetRequestId))
                    return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (_slotByRequest.ContainsKey(request.TargetRequestId)) return false;
                int copy = CopyIntoNewSlot(payload.Slot);
                if (copy < 0) return false;
                // The clone is unbound; the executor truncates it to TargetTokens at the first bind.
                _slotByRequest.Add(request.TargetRequestId, copy);
                return true;
            }
        }
        if (request.Op != MaterializeOp.Donate) return false;
        lock (_nativeSync)
        {
            if (!SlotRetention.CanDonate(_retainedSlots, request.PayloadKey, request.PayloadTokens, request.TargetTokens, true))
                return false;
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            if (!SlotRetention.DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _))
                return false;
            _checkpointKeys.Remove(request.PayloadKey);
            return true;
        }
    }

    private bool NativeTryReturnDonation(string requestId, string payloadKey)
    {
        lock (_nativeSync)
        {
            if (_slotByRequest == null) return false;
            _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
            return SlotRetention.ReturnSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, ref _primarySlot,
                ref _activeSlotKey, new NativeSlotStore(_native));
        }
    }

    private PayloadFootprint NativeFootprint(SlotRetention.RetainedSlot entry) => new(entry.Tokens, _maxContextLength, default, PositionDelta: 0);
}
