// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Native whole-model execution for MiniMax-M3.1 on ggml_cuda (and ggml_cpu as a
// correctness path).
//
// The UD-Q4_K_XL checkpoint is ~232 GiB, so it is spread across GPUs with
// --layer-split N, optionally with the leading MoE layers' experts in system RAM
// (--n-cpu-moe), or split layer by layer with --tp N (ggml's meta backend over
// the first N CUDA GPUs: attention and caches mirrored, the FFN split; see
// ggml_ops_minimax_m3_tp.h). The native executor (TensorSharp.GGML.Native/ggml_ops_minimax_m3_exec.cpp)
// loads the split GGUF itself, owns every sequence slot's K, V and index-key rows on
// the devices that read them, runs MiniMax Sparse Attention per sequence (positions
// equal cache rows, so a slot is never shifted), and submits ONE graph per ubatch.
// The ABI and the slot, batching and reclaim contracts are Muse Spark's.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models;

public partial class MiniMaxM3Model : IBatchedPagedModel, ISoloPrefillBudgetModel
{
    private IntPtr _native;
    private readonly object _nativeSync = new();
    private int _nativeUbatch;
    private int _nativePipelineUbatch;   // pipelined prefill micro-batch rows (adaptive: their upper bound); 0 when off
    private int _nativeTpDegree;         // GPUs the native executor splits every layer over; 0 for a layer split

    /// <summary>True when the native whole-model executor drives this instance.</summary>
    internal bool UsesNativeExecutor => _native != IntPtr.Zero;

    /// <summary>Backends the native executor runs on: ggml_cuda serves, ggml_cpu is a correctness path.</summary>
    internal static bool NativeRequested(BackendType backend) => backend is BackendType.GgmlCuda or BackendType.GgmlCpu;

    private static string NativeBackendName(BackendType backend) => backend switch
    {
        BackendType.GgmlCuda => "CUDA",
        BackendType.GgmlCpu => "CPU",
        _ => null,
    };

    private static int ParseEnvInt(string name, int fallback)
        => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v > 0 ? v : fallback;

    /// <summary>GPUs the native executor can span (MAX_GPUS of the Muse Spark executor it shares its layout with).</summary>
    internal const int NativeMaxGpus = 16;

    /// <summary>The GPU count the native executor is asked for: <c>--layer-split N</c>, or more when
    /// TS_MINIMAX_M3_LAYER_SPLIT lists more per-GPU counts. A count it cannot honour is a load refusal
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
                    $"[minimax-m3] {requested} GPUs requested; the native executor spans at most {NativeMaxGpus}.");
            if (visible < requested)
                throw new ModelLoadRefusedException(
                    $"[minimax-m3] {requested} GPU(s) requested but the GGML {backendName} backend sees only {visible}.");
        }
        return requested;
    }

    /// <summary>The <c>--n-cpu-moe</c> layer count handed to the executor (-1: the fewest that fit,
    /// from TS_MINIMAX_M3_CPU_MOE=auto). Layer indices count from 0, as in llama.cpp, so the leading
    /// dense layers (which have no experts) are part of the count; a count that reaches no MoE layer
    /// is said rather than silently doing nothing.</summary>
    internal static int ResolveCpuMoeLayers(int numLayers, int denseLead, string autoSetting)
    {
        if (autoSetting is string auto && string.Equals(auto.Trim(), "auto", StringComparison.OrdinalIgnoreCase))
            return -1;
        int cpuMoe = !MoeCpuOffloadConfig.Current.IsExplicitlySet ? 0
            : MoeCpuOffloadConfig.Current.AllLayers ? numLayers : Math.Min(numLayers, MoeCpuOffloadConfig.Current.CpuMoeLayers);
        if (cpuMoe > 0 && cpuMoe <= denseLead)
            Console.Error.WriteLine($"[minimax-m3] --n-cpu-moe {cpuMoe} moves no experts: layers 0..{denseLead - 1} are " +
                $"dense, so the first routed experts are in layer {denseLead}. Use --n-cpu-moe {denseLead + 1} or more.");
        return cpuMoe;
    }

    private void InitNativeExecutor(string ggufPath, BackendType backend, int layerSplitDegree, int tpDegree)
    {
        string split = Environment.GetEnvironmentVariable("TS_MINIMAX_M3_LAYER_SPLIT");
        int? available = null;
        if (backend == BackendType.GgmlCuda)
            available = GgmlBasicOps.GetGpuDeviceCount(GgmlBackendType.Cuda);
        bool tensorParallel = tpDegree > 1;
        if (tensorParallel)
        {
            if (!string.IsNullOrWhiteSpace(split))
                throw new ModelLoadRefusedException(
                    "[minimax-m3] --tp cannot be combined with TS_MINIMAX_M3_LAYER_SPLIT (a layer split); use one or the other.");
            if (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEVICES")))
                throw new ModelLoadRefusedException(
                    "[minimax-m3] the native executor's --tp uses the first N CUDA GPUs and does not read " +
                    "TENSORSHARP_TP_DEVICES; select GPUs with CUDA_VISIBLE_DEVICES.");
        }
        int requested = tensorParallel
            ? ResolveNativeGpuCount(tpDegree, null, available, NativeBackendName(backend))
            : ResolveNativeGpuCount(layerSplitDegree, split, available, NativeBackendName(backend));

        int nUbatch = ParseEnvInt("TS_MINIMAX_M3_UBATCH", 2048);
        int nThreads = ParseEnvInt("TS_MINIMAX_M3_THREADS", 0);
        int cpuMoe = ResolveCpuMoeLayers(Config.NumLayers, _denseLead, Environment.GetEnvironmentVariable("TS_MINIMAX_M3_CPU_MOE"));
        int maxContext = ResolveConfiguredContextLength();
        bool hardLimit = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT"));

        IntPtr handle;
        try
        {
            ForwardNativeOptions();
            handle = _dsparkPath == null
                ? GgmlMiniMaxM3Native.LoadModel(ggufPath, requested, maxContext, nUbatch, nThreads, cpuMoe,
                    NativeBackendName(backend), hardLimit, tensorParallel ? null : split, tensorParallel ? tpDegree : 0)
                : GgmlMiniMaxM3DsparkNative.LoadModel(ggufPath, requested, maxContext, nUbatch, nThreads, cpuMoe,
                    NativeBackendName(backend), hardLimit, split, 0, _dsparkPath);
        }
        catch (Exception ex) when (ex is EntryPointNotFoundException or DllNotFoundException)
        {
            throw new ModelLoadRefusedException(
                "[minimax-m3] the loaded GgmlOps native library has no MiniMax-M3 executor " +
                $"(TSGgml_MiniMaxM3*: {ex.GetType().Name}); rebuild TensorSharp.GGML.Native with " +
                "ggml_ops_minimax_m3_exec.cpp.");
        }
        if (handle == IntPtr.Zero)
            throw NativeLoadRefused("minimax-m3", ggufPath);
        _native = handle;
        _nativeUbatch = Math.Max(1, GgmlMiniMaxM3Native.Ubatch(handle));
        int hidden = GgmlMiniMaxM3Native.HiddenSize(handle);
        if (hidden != Config.HiddenSize)
            throw new ModelLoadRefusedException(
                $"[minimax-m3] the native executor reports hidden size {hidden}; the GGUF metadata says {Config.HiddenSize}.");
        _maxContextLength = GgmlMiniMaxM3Native.CtxSize(handle);
        int vocab = GgmlMiniMaxM3Native.VocabSize(handle);
        if (vocab > 0) Config.VocabSize = vocab;
        _logitsBuffer = new float[Config.VocabSize];
        _primarySlot = 0;
        long[] stats = GgmlMiniMaxM3Native.GetStats(handle);
        _nativePipelineUbatch = stats != null ? (int)stats[9] : 0;
        _nativeTpDegree = stats != null ? (int)stats[11] : 0;
        if (tensorParallel && _nativeTpDegree != tpDegree)
            throw new ModelLoadRefusedException(
                $"[minimax-m3] --tp {tpDegree} was requested but the native executor reports tensor parallelism over {_nativeTpDegree} GPU(s).");
        if (!tensorParallel && _nativeTpDegree > 1)
            throw new ModelLoadRefusedException(
                $"[minimax-m3] the native executor reports tensor parallelism over {_nativeTpDegree} GPUs, which was not requested.");
        // The cache namespace names the sparse layers; it must describe what the executor runs.
        int sparseLayers = _isMsaLayer.Count(m => m);
        if (stats != null && stats.Length > GgmlMiniMaxM3Native.StatSparseLayers &&
            stats[GgmlMiniMaxM3Native.StatSparseLayers] != sparseLayers)
            throw new ModelLoadRefusedException(
                $"[minimax-m3] the native executor runs {stats[GgmlMiniMaxM3Native.StatSparseLayers]} sparse-attention " +
                $"layers; the checkpoint's indexer tensors give {sparseLayers}.");
        InitDsparkHead();
        string hostLayers = stats != null ? stats[3].ToString() : cpuMoe < 0 ? "auto" : cpuMoe.ToString();
        if (tensorParallel)
            Console.WriteLine($"MiniMax-M3.1 native executor: {Config.NumLayers} layers tensor-parallel over {_nativeTpDegree} GPUs " +
                $"(attention mirrored, FFN split), context {_maxContextLength}, ubatch {_nativeUbatch}.");
        else
            Console.WriteLine($"MiniMax-M3.1 native executor: {Config.NumLayers} layers over " +
                $"{(stats != null ? stats[4].ToString() : requested.ToString())} {(backend == BackendType.GgmlCpu ? "CPU device" : "GPU(s)")}, " +
                $"context {_maxContextLength}, ubatch {_nativeUbatch}, layers with host experts {hostLayers}.");
    }

    /// <summary>Native executor counters (<see cref="GgmlMiniMaxM3Native.GetStats"/>); null without the executor.</summary>
    internal long[] NativeStatsForTests()
    {
        lock (_nativeSync) return _native == IntPtr.Zero ? null : GgmlMiniMaxM3Native.GetStats(_native);
    }

    /// <summary>Settings the native executor reads. Process-environment values set from .NET never reach
    /// native getenv on Linux, so the managed view is forwarded explicitly (and a cleared value clears it).
    /// TS_MINIMAX_M3_LAYER_SPLIT, _UBATCH, _THREADS and _CPU_MOE travel as load arguments instead
    /// (<see cref="NativeLoadArgumentNames"/>). MiniMaxM3ModelTests checks both directions against the
    /// native sources.</summary>
    internal static readonly string[] NativeOptionNames =
    [
        MiniMaxM3Architecture.MsaModeVariable, "TS_MINIMAX_M3_MSA_CHUNK_MB", "TS_MINIMAX_M3_INDEXER_F16",
        "TS_MINIMAX_M3_FUSE_QKV", "TS_MINIMAX_M3_VRAM_RESERVE_MB", "TS_MINIMAX_M3_DEV0_RESERVE_MB",
        "TS_MINIMAX_M3_KV_PAD", "TS_MINIMAX_M3_BD_KV_PAD", "TS_MINIMAX_M3_LOAD_CHUNK_MB", "TS_MINIMAX_M3_LOAD_THREADS",
        "TS_MINIMAX_M3_OP_OFFLOAD", "TS_MINIMAX_M3_GRAPH_CACHE", "TS_MINIMAX_M3_NODES_PER_LAYER",
        "TS_MINIMAX_M3_EXEC_DEBUG", "TS_MINIMAX_M3_WARM_HOST", "TS_MINIMAX_M3_DROP_UPLOAD_CACHE",
        "TS_MINIMAX_M3_SLOT_HEADROOM_MB", "TS_MINIMAX_M3_CPU_REPACK", "TS_MINIMAX_M3_PROFILE", "TS_CPU_MOE_THREADS",
        "TS_MINIMAX_M3_HOST_NUMA", "TS_MINIMAX_M3_NUMA_BATCH_GB", "TS_MINIMAX_M3_SLOT_INIT_ROWS",
        "TS_MINIMAX_M3_PLAN_CTX_TOKENS", "TS_MINIMAX_M3_HOST_MOE_TOKENS", "TS_MINIMAX_M3_ADMIT_GRAPH_MB",
        "TS_MINIMAX_M3_WARM_POOLS", "TS_MINIMAX_M3_PIPELINE", "TS_MINIMAX_M3_PIPELINE_UBATCH",
        "TS_MINIMAX_M3_DSPARK_GRAPHS",
    ];

    /// <summary>Settings the native sources also read but the managed side hands to the load call itself
    /// (TS_MINIMAX_M3_LAYER_SPLIT: the layer_split argument, which takes precedence over the environment).</summary>
    internal static readonly string[] NativeLoadArgumentNames = ["TS_MINIMAX_M3_LAYER_SPLIT"];

    private static void ForwardNativeOptions()
    {
        foreach (string name in NativeOptionNames)
            GgmlMiniMaxM3Native.SetOption(name, Environment.GetEnvironmentVariable(name));
    }

    // ------------------------------------------------------------------ image rows

    // Projected image rows waiting for the forward call that covers their positions:
    // (rows [count x hidden] row-major, count, absolute position of the first row).
    private readonly List<(float[] Rows, int Count, int Position)> _pendingVisionRows = new();

    internal bool HasPendingVisionRows
    {
        get
        {
            lock (_pendingVisionRows) return _pendingVisionRows.Count != 0;
        }
    }

    /// <summary>
    /// Queue <paramref name="nRows"/> projected image rows (<c>[nRows, hidden]</c>, row-major) to
    /// replace the token embeddings at absolute positions <paramref name="absolutePosition"/> ..
    /// <paramref name="absolutePosition"/> + <paramref name="nRows"/> - 1 of the active sequence.
    /// The rows are handed to the executor by the forward call that covers them, so a span may
    /// arrive across several prefill chunks; a reset or rewind of the sequence drops them.
    /// </summary>
    internal void QueueNativeVisionRows(float[] rowsRowMajor, int nRows, int absolutePosition)
    {
        ArgumentNullException.ThrowIfNull(rowsRowMajor);
        if (nRows <= 0) throw new ArgumentOutOfRangeException(nameof(nRows));
        if (absolutePosition < 0) throw new ArgumentOutOfRangeException(nameof(absolutePosition));
        if ((long)rowsRowMajor.Length != (long)nRows * Config.HiddenSize)
            throw new ArgumentException($"Image rows must be [{nRows}, {Config.HiddenSize}].", nameof(rowsRowMajor));
        _ = checked(absolutePosition + nRows);
        lock (_nativeSync)
        {
            if (absolutePosition < _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(absolutePosition),
                    $"Image rows at position {absolutePosition} precede the sequence head ({_cacheSeqLen}).");
            lock (_pendingVisionRows) _pendingVisionRows.Add((rowsRowMajor, nRows, absolutePosition));
        }
    }

    private void ClearPendingVisionRows()
    {
        lock (_pendingVisionRows) _pendingVisionRows.Clear();
    }

    /// <summary>Hand the executor the pending rows a call over positions [start, start + count) covers
    /// (at their absolute positions; the executor drops rows outside the call) and drop the spans that
    /// end inside it.</summary>
    private void QueueOverlappingVisionRows(int start, int count)
    {
        int end = start + count;
        int hidden = Config.HiddenSize;
        lock (_pendingVisionRows)
        {
            for (int i = 0; i < _pendingVisionRows.Count;)
            {
                var (rows, n, position) = _pendingVisionRows[i];
                int spanEnd = position + n;
                int from = Math.Max(position, start), to = Math.Min(spanEnd, end);
                if (from < to)
                {
                    var slice = rows.AsSpan(checked((from - position) * hidden), checked((to - from) * hidden));
                    if (!GgmlMiniMaxM3Native.QueueVisionRows(_native, slice, to - from, from))
                        throw new InvalidOperationException("MiniMax-M3.1 native executor refused image rows.");
                }
                if (spanEnd <= end) _pendingVisionRows.RemoveAt(i);
                else ++i;
            }
        }
    }

    // ------------------------------------------------------------------ forward

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
            int start = GgmlMiniMaxM3Native.NPast(_native);
            if ((long)start + tokens.Length > _maxContextLength)
                throw new InvalidOperationException("MiniMax-M3.1 context length exceeded.");
            BeforeGraphCall();
            bool ok;
            try
            {
                if (HasPendingVisionRows) QueueOverlappingVisionRows(start, tokens.Length);
                _forwardSw.Start();
                ok = GgmlMiniMaxM3Native.Forward(_native, tokens, output, allRows);
            }
            finally
            {
                _forwardSw.Stop();
                _forwardCount++;
                AfterGraphCall();
            }
            if (!ok)
            {
                GgmlMiniMaxM3Native.ClearVisionRows(_native);
                throw new InvalidOperationException("MiniMax-M3.1 native forward failed (see stderr).");
            }
            _cacheSeqLen = GgmlMiniMaxM3Native.NPast(_native);
            return output;
        }
    }

    private void ResetNative()
    {
        lock (_nativeSync)
        {
            if (!GgmlMiniMaxM3Native.ResetChecked(_native))
                throw new InvalidOperationException("MiniMax-M3.1 native reset failed.");
            _cacheSeqLen = 0;
        }
    }

    private bool RewindNative(int tokenCount)
    {
        lock (_nativeSync)
        {
            if (!GgmlMiniMaxM3Native.Rewind(_native, tokenCount)) return false;
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
            Console.Error.WriteLine($"[minimax-m3] warmup forward failed: {ex.Message}");
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
                GgmlMiniMaxM3Native.Free(_native);
                _native = IntPtr.Zero;
            }
        }
    }

    // ------------------------------------------------------------------ lone-prompt prefill steps

    /// <summary>A lone prompt's prefill step when the executor pipelines a layer split's micro-batches
    /// (Muse Spark's rule, <see cref="SoloPrefillStep"/>); 0 (the scheduler's own step) otherwise.</summary>
    public int PreferredSoloPrefillTokens(int start, int remaining)
    {
        // (The scheduler keeps its own step when the step would carry image rows.)
        if (_nativePipelineUbatch <= 0) return 0;
        lock (_nativeSync)
        {
            if (_native == IntPtr.Zero) return 0;
            IntPtr handle = _native;
            return SoloPrefillStep(start, remaining, (p0, n) => GgmlMiniMaxM3Native.PipelinePlan(handle, p0, n));
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

    /// <summary>A shared-prefix boundary at most this far past a reused prefix gets no prefill step of its
    /// own (<see cref="ISoloPrefillBudgetModel.CheckpointGapTokens"/>): any forward pass of this model streams
    /// most of its routed experts through every GPU, so the 5-token step a chat turn spent reaching a
    /// boundary just past its reused prefix cost a pass of its own before the turn's real prefill.</summary>
    public int CheckpointGapTokens => 64;

    // ------------------------------------------------------------------ slots

    // requestId -> native slot (guarded by _nativeSync)
    private Dictionary<string, int> _slotByRequest;
    private int _primarySlot;
    private string _activeSlotKey;
    private bool _batchedChunkingLatched;

    private readonly struct NativeSlotStore : SlotRetention.ISlotStore
    {
        private readonly IntPtr _h;
        public NativeSlotStore(IntPtr handle) => _h = handle;
        public int Alloc() => GgmlMiniMaxM3Native.SlotAlloc(_h);
        public bool Select(int slot) => GgmlMiniMaxM3Native.SetActiveSlot(_h, slot);
        public bool Free(int slot) => GgmlMiniMaxM3Native.SlotFree(_h, slot);
        public int ActiveHead() => GgmlMiniMaxM3Native.NPast(_h);
        public bool RewindActive(int tokens) => GgmlMiniMaxM3Native.Rewind(_h, tokens);
        public bool ResetActive() => GgmlMiniMaxM3Native.ResetChecked(_h);
        public bool SetReclaimable(int[] slots) => GgmlMiniMaxM3Native.SetReclaimableSlots(_h, slots);
        public int TakeReclaimed(int[] buffer) => GgmlMiniMaxM3Native.TakeReclaimedSlots(_h, buffer);
    }

    // The executor serves concurrency through its own sequence slots; there is no paged path.
    public bool BatchedForwardAvailable => false;

    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        => throw new NotSupportedException("MiniMax-M3.1 serves concurrent sequences through its native executor's " +
            "per-sequence slots, not the paged batch path.");

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
            if (!UsesNativeExecutor) throw new InvalidOperationException("Per-request slots require the native MiniMax-M3.1 executor.");
            _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
            var store = new NativeSlotStore(_native);
            bool fresh = false;
            if (!_slotByRequest.TryGetValue(requestId, out int slot))
            {
                slot = store.Alloc();
                if (slot < 0) slot = SlotRetention.TakeIdlePrimary(_slotByRequest, ref _primarySlot, _activeSlotKey, store);
                if (slot < 0 && _retainedSlots is { Count: > 0 }) slot = AllocSlotReclaimingRetained();
                if (slot < 0)
                    throw new SequenceSlotUnavailableException("MiniMax-M3.1 has no device memory for another sequence slot.");
                _slotByRequest[requestId] = slot;
                fresh = true;
            }
            if (!store.Select(slot))
                throw new InvalidOperationException($"MiniMax-M3.1 slot {slot} missing for request {requestId}.");
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
    /// every later conversation sharing the prefix clones) only when nothing else is left. The retained
    /// slot <paramref name="keep"/> (a clone's source) is never freed.</summary>
    private int AllocSlotReclaimingRetained(int keep = -1)
    {
        var store = new NativeSlotStore(_native);
        int slot = store.Alloc();
        while (slot < 0 && _retainedSlots is { Count: > 0 })
        {
            string key = ReclaimOrder().FirstOrDefault(k => _retainedSlots[k].Slot != keep);
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
                    throw new SequenceSlotUnavailableException("MiniMax-M3.1 primary-slot allocation failed (device memory exhausted?).");
            }
            var store = new NativeSlotStore(_native);
            if (!store.Select(_primarySlot))
                throw new InvalidOperationException($"MiniMax-M3.1 primary slot {_primarySlot} missing.");
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

    /// <summary>Image rows are queued against the active sequence's positions and only a single-sequence
    /// forward hands them to the executor, so the batched paths step aside while any are pending.</summary>
    private const string PendingImageRowsDecline =
        "image rows are queued for the active sequence's next prefill; a batched graph would not place them";

    public string BatchedFusedDecodeDeclineReason => HasPendingVisionRows ? PendingImageRowsDecline : null;

    public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || _slotByRequest == null || HasPendingVisionRows) return false;
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
            _cacheSeqLen = GgmlMiniMaxM3Native.NPast(_native);
            return true;
        }
    }

    /// <summary>The batched prefill rewinds a cloned slot to the prefix its request reuses (exact here: the
    /// cache keeps every row, so a rewind only moves the head).</summary>
    public bool BatchedPrefillRewindsHolders => UsesNativeExecutor;

    /// <summary>Rewind a slot other than (or equal to) the active one; the active slot stays active.</summary>
    private bool RewindNativeSlot(int slot, int rows)
    {
        int active = SlotOfRequestOrActive(null);
        if (!GgmlMiniMaxM3Native.SetActiveSlot(_native, slot)) return false;
        bool ok = GgmlMiniMaxM3Native.Rewind(_native, rows);
        if (active >= 0) GgmlMiniMaxM3Native.SetActiveSlot(_native, active);
        return ok;
    }

    /// <summary>Concurrent prompt chunks in shared graphs: each native call takes at most one ubatch of
    /// tokens, drawn from every sequence that still has some, so arrivals prefill together.</summary>
    public bool TryForwardBatchedFusedPrefill(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens,
        int[] positions, float[][] outLogits)
    {
        lock (_nativeSync)
        {
            if (!UsesNativeExecutor || requestIds == null || tokens == null || positions == null || HasPendingVisionRows)
                return false;
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
                // Positions are cache rows: a segment must continue its slot exactly. A slot cloned
                // from a longer cached state still holds rows past the reused prefix
                // (BatchedPrefillRewindsHolders); every rewind of this cache is exact.
                int head = GgmlMiniMaxM3Native.SlotNPast(_native, slots[i]);
                if (head > positions[i] && !RewindNativeSlot(slots[i], positions[i])) return false;
                if (head < positions[i]) return false;
            }

            int ubatch = Math.Max(n, GgmlMiniMaxM3Native.Ubatch(_native));
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
                    bool ok = GgmlMiniMaxM3Native.ForwardSegments(_native, callSlots.ToArray(), callLens.ToArray(),
                        flat.ToArray(), callPos.ToArray(), logits);
                    _forwardSw.Stop();
                    _forwardCount++;
                    if (!ok)
                    {
                        if (!progressed) return false;
                        throw new InvalidOperationException("MiniMax-M3.1 batched prefill failed after partial progress (see stderr).");
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
            _cacheSeqLen = GgmlMiniMaxM3Native.NPast(_native);
            return true;
        }
    }

    private bool ForwardBatchedDecodeChunks(int[] slots, int[] tokens, int[] positions, float[] flat, int vocab)
    {
        const int MaxPerCall = 16;
        int n = slots.Length;
        if (n <= MaxPerCall) return GgmlMiniMaxM3Native.ForwardBatchedDecode(_native, slots, tokens, positions, flat);
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
            if (!GgmlMiniMaxM3Native.ForwardBatchedDecode(_native, cs, ct, cp, cf))
            {
                if (c > 0)
                {
                    // The earlier chunks already wrote this step's token into their slots. False means "nothing
                    // advanced" (the executor then serves every sequence serially from its position), so their
                    // heads go back to the step's positions first (exact: every row is kept); a slot that cannot
                    // be rewound fails the step instead of having its token forwarded twice.
                    _batchedChunkingLatched = true;
                    for (int i = 0; i < off; i++)
                        if (GgmlMiniMaxM3Native.SlotNPast(_native, slots[i]) != positions[i] && !RewindNativeSlot(slots[i], positions[i]))
                            throw new InvalidOperationException(
                                $"MiniMax-M3.1 batched decode failed in chunk {c + 1}/{chunks} after earlier chunks advanced, " +
                                $"and slot {slots[i]} could not be rewound to position {positions[i]} (see stderr).");
                    Console.Error.WriteLine($"[minimax-m3 batched-decode] chunk {c + 1}/{chunks} failed mid-step; the {off} sequences of " +
                        "the earlier chunks were rewound and this step runs serially; chunked batching disabled");
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
            if (!GgmlMiniMaxM3Native.SetReclaimableSlots(_native, order))
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
}
