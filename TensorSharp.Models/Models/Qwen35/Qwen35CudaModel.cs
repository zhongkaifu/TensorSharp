// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.5 / Qwen3.8 dense on --backend cuda: the direct-CUDA engine (TensorSharp.Backends.Cuda/
// Qwen35). The model's managed state stays on a host allocator; the engine owns every weight and
// cache. Concurrent requests run on the engine's sequence slots; a finished conversation's slot is
// retained for its next turn (donate-only prefix reuse, as Qwen4Exp's). With a DFlash2 drafter
// (--draft-model) every decoding request speculates through the batched verify: the drafter keeps
// a KV ring inside each slot that the engine refills on every commit, so no hidden state ever
// leaves the device.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public sealed partial class Qwen35CudaModel : ModelBase, IBatchedPagedModel, IPrefixCacheModel, IPrefixCacheModelDiagnostics,
        ISpeculativeModel, IBatchedSpeculativeTarget, IMixedStepPrefillTarget, IVisionCapableModel, IMRoPEPositionSink, IMultimodalPromptExpander
    {
        private const string Tag = "q35-cuda";
        private Qwen35CudaExecutor _exec;
        private readonly object _sync = new object();
        private readonly bool _reclaimVisionPool = Environment.GetEnvironmentVariable("TS_Q35_RECLAIM_VISION_POOL") == "1";
        private long _visionPoolPressureTrimCount;

        // requestId -> engine slot. Guarded by _sync.
        private Dictionary<string, int> _slotByRequest;
        // The slot serving the single-stream (N==1) path; -1 while a request holds it.
        private int _primarySlot;
        // Request whose slot is currently active, or null when the primary is.
        private string _activeSlotKey;
        // payload key -> retained slot, oldest first (the order memory takes them back in).
        private OrderedDictionary<string, SlotRetention.RetainedSlot> _retainedSlots;
        // payload key -> retained slot moved to pinned host memory, oldest first; a hit brings it back.
        private readonly OrderedDictionary<string, HostPayload> _hostRetained = new(StringComparer.Ordinal);
        private readonly record struct HostPayload(TensorSharp.Cuda.Q35CudaEngine.HostSlot Host, int Tokens, bool Checkpoint);
        // Pinned host memory retained slots may take (0: none; they are dropped when the device is full).
        private long _hostBudgetBytes;
        // The host payload being brought back, which making room for it must not drop.
        private string _promoting;
        // Retained slots whose host copy is in flight on the copy stream, oldest first: they stay
        // retained device slots until the copy completes.
        private readonly OrderedDictionary<string, Retiring> _retiring = new(StringComparer.Ordinal);
        private readonly record struct Retiring(TensorSharp.Cuda.Q35CudaEngine.HostSlot Host, IntPtr Done);
        /// <summary>Free device memory kept beyond the retain reserve by moving retained slots to host
        /// memory ahead of need, off the critical path.</summary>
        private const long RetireHeadroomBytes = 2L << 30;
        private IPrefixPayloadSink _prefixCacheSink;
        // The request whose drafter state DraftBlock uses (BindDraftState), and the last verify's window.
        private string _draftRequest;
        private int _verifyWindow;
        // Rows of the single-sequence verify awaiting SpecOnVerifyAccepted (0: none).
        private int _soloVerifyRows;
        private int[] _soloDrawn = Array.Empty<int>();

        /// <summary>
        /// True when this checkpoint runs on the direct-CUDA engine: a dense qwen35 (no experts)
        /// whose projections are unscaled NVFP4. TS_Q35_NATIVE=0 keeps the managed path.
        /// </summary>
        internal static bool CanServe(GgufFile probe)
        {
            if (Environment.GetEnvironmentVariable("TS_Q35_NATIVE") == "0" || probe == null)
                return false;
            if (probe.GetString("general.architecture") != "qwen35" || probe.GetUint32("qwen35.expert_count", 0) != 0)
                return false;
            if (!probe.Tensors.TryGetValue("blk.0.ffn_gate.weight", out var gate) || gate.Type != GgmlTensorType.NVFP4
                || probe.Tensors.ContainsKey("blk.0.ffn_gate.scale"))
                return false;
            // Elsewhere the managed path serves the model, as it did before this engine.
            if (!TensorSharp.Cuda.Q35CudaEngine.IsSupported(out string reason))
            {
                Console.Error.WriteLine($"[{Tag}] the direct-CUDA engine is unavailable here ({reason}); using the managed path");
                return false;
            }
            return true;
        }

        public Qwen35CudaModel(string ggufPath, string draftGgufPath = null)
            : base(ggufPath, BackendType.Cpu, 1, null)
        {
            Config = new ModelConfig { Architecture = "qwen35" };
            try
            {
                ParseBaseConfig();
                ParseTokenizer();
                int maxContext = ResolveConfiguredContextLength();
                // Every slot holds its whole context: keep a practical default unless MAX_CONTEXT names one.
                if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT")))
                    maxContext = Math.Min(maxContext, 65536);
                int nUbatch = ParseEnvInt("TS_Q35_UBATCH", 2048);
                _exec = new Qwen35CudaExecutor(ggufPath, maxContext, nUbatch, Qwen35Model.ResolveQwen35DFlashPath(draftGgufPath));
                InitializeEngineRuntime();
                Console.WriteLine($"Model: qwen35 on the direct-CUDA engine, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                    $"Vocab={Config.VocabSize}, n_ctx={_maxContextLength}, drafter={(_exec.Draft != null ? "DFlash2" : "none")}");
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>Pinned host memory for retained slots: TS_Q35_HOST_CACHE_MB, else half the
        /// memory the runtime may use (at most 32 GB).</summary>
        private static long ResolveHostBudgetBytes()
        {
            string raw = Environment.GetEnvironmentVariable("TS_Q35_HOST_CACHE_MB");
            if (long.TryParse(raw, out long mb) && mb >= 0)
                return mb << 20;
            long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
            return available > 0 ? Math.Min(available / 2, 32L << 30) : 0;
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        /// <summary>A delta recurrence cannot be rewound: a cached prefix is reusable only when the new
        /// prompt extends it exactly.</summary>
        public override bool SupportsKVCacheTruncation => false;

        public override string KVStateFingerprint =>
            $"qwen35|exec=cuda|L={Config.NumLayers}|H={Config.HiddenSize}|V={Config.VocabSize}|ctx={_maxContextLength}" +
            (_ninferCacheIdentity == null ? string.Empty : "|ninfer=" + _ninferCacheIdentity) +
            (_exec?.Engine?.KVStateNamespaceSuffix ?? string.Empty);

        /// <summary>Host metadata diagnostics only, immutable at native engine construction.</summary>
        public bool ServingTraceEnabled => _exec?.Engine?.ServingTraceEnabled == true;

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_sync)
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                ForwardWithMedia(tokens, _logitsBuffer);
                return _logitsBuffer;
            }
        }

        /// <summary>A forward of the active slot that consumes the queued image rows and (T,H,W)
        /// position table, if any (under _sync).</summary>
        private void ForwardWithMedia(int[] tokens, float[] logitsOut)
        {
            int[] ropePos = RopeComponents(_pendingMRoPE, tokens.Length);
            _pendingMRoPE = null;
            var spans = _pendingVision.Count > 0 ? _pendingVision.ToArray() : null;
            _pendingVision.Clear();
            try
            {
                _exec.Engine.TraceBindSequence(_activeSlotKey, _exec.Engine.Active);
                _exec.Engine.Forward(tokens, logitsOut, ropePos, spans);
            }
            finally
            {
                // The rows were copied on the engine's stream: let it finish before the tower reuses them.
                if (_pendingVisionTensors.Count > 0)
                {
                    _exec.Engine.Synchronize();
                    foreach (var t in _pendingVisionTensors)
                        t.Dispose();
                    _pendingVisionTensors.Clear();
                }
            }
            _cacheSeqLen = _exec.NPast;
        }

        /// <summary>The first <paramref name="n"/> token-major (T, H, W) rows of a queued M-RoPE table as
        /// the engine's [4][n] components (the fourth unused); null without a table for them.</summary>
        private static int[] RopeComponents(int[] table, int n)
        {
            if (table == null || table.Length < 3 * n)
                return null;
            var ropePos = new int[4 * n];
            for (int i = 0; i < n; i++)
            {
                ropePos[i] = table[3 * i];
                ropePos[n + i] = table[3 * i + 1];
                ropePos[2 * n + i] = table[3 * i + 2];
            }
            return ropePos;
        }

        // ---------------------------------------------------------------- vision

        private Qwen35VisionEncoder _vision;
        private IAllocator _visionAllocator;
        private readonly List<TensorSharp.Cuda.Q35CudaEngine.EmbeddingSpan> _pendingVision = new();
        // Device tensors the queued spans read from, released after the forward that consumes them.
        private readonly List<Tensor> _pendingVisionTensors = new();
        private int[] _pendingMRoPE;

        public bool IsVisionEncoderLoaded => _vision != null;

        // The tower runs on its own allocator and stream, the engine on its own: an encode can
        // overlap engine steps (images serialize among themselves).
        private readonly object _mediaLock = new();
        public override object MediaPreparationLock => _mediaLock;

        /// <summary>The Qwen-VL tower runs on the direct-CUDA tensor backend in the engine's context;
        /// its rows enter the engine's forward as embedding overrides copied device to device.</summary>
        public void LoadVisionEncoder(string mmProjPath)
        {
            // The NInfer route takes the same Qwen3.8 vision tower from the mmproj GGUF; its rows
            // replace the image placeholders' FP8 embeddings like on the GGUF route.
            _visionAllocator ??= new TensorSharp.Cuda.CudaAllocator(0);
            _vision = new Qwen35VisionEncoder(mmProjPath, _visionAllocator);
            _vision.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            if (embeddings == null)
                return;
            bool kept = false;
            try
            {
                int rows = (int)embeddings.Sizes[0], width = (int)embeddings.Sizes[1];
                if (width != Config.HiddenSize)
                    throw new InvalidOperationException($"qwen35 vision rows are {width} wide; the model takes {Config.HiddenSize}.");
                if (embeddings.Storage is TensorSharp.Cuda.CudaStorage cuda && embeddings.IsContiguous()
                    && embeddings.ElementType == DType.Float32)
                {
                    // The tower synchronized its stream when the encode finished.
                    IntPtr device = cuda.DevicePtrForRead(embeddings.StorageOffset);
                    lock (_sync)
                    {
                        _pendingVision.Add(new TensorSharp.Cuda.Q35CudaEngine.EmbeddingSpan(null, rows, insertPosition, device));
                        _pendingVisionTensors.Add(embeddings);
                    }
                    kept = true;
                    return;
                }
                float[] data = embeddings.GetElementsAsFloat(rows * width);
                lock (_sync)
                    _pendingVision.Add(new TensorSharp.Cuda.Q35CudaEngine.EmbeddingSpan(data, rows, insertPosition));
            }
            finally
            {
                if (!kept)
                    embeddings.Dispose();
            }
        }

        public void SetMRoPEPositions(int[] flatThw)
        {
            lock (_sync)
                _pendingMRoPE = flatThw;
        }

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessQwenVLHistory(_vision, history, inputTokens);

        protected override void ResetKVCacheCore()
        {
            lock (_sync)
            {
                if (!_exec.ResetChecked())
                    throw new InvalidOperationException("qwen35 reset failed; the slot remains unusable.");
                _cacheSeqLen = 0;
            }
        }

        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            lock (_sync)
            {
                if (!_exec.Rewind(tokenCount))
                    return false;
                _cacheSeqLen = tokenCount;
                return true;
            }
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!TryTruncateKVCacheCore(tokenCount))
                throw new InvalidOperationException(
                    $"qwen35 cannot truncate its caches to {tokenCount} tokens (head at {_cacheSeqLen}); its " +
                    "recurrent state only restarts from 0. Use TryTruncateKVCache and re-prefill when it declines.");
        }

        public override void WarmUpKernels()
        {
            try
            {
                int bos = Tokenizer?.BosTokenId ?? -1;
                ForwardCore(new[] { bos >= 0 ? bos : 0 });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[{Tag}] warmup forward failed: {ex.Message}");
            }
            finally
            {
                ResetKVCacheCore();
            }
            // The tower's first encode pays ~3.5 s of one-time library and kernel setup: take it
            // here, on a blank screenshot-sized image, instead of in the first image request.
            if (_vision != null)
            {
                try
                {
                    const int h = 768, w = 1344;
                    _vision.Encode(new float[3 * h * w], h, w)?.Dispose();
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[{Tag}] vision warmup failed: {ex.Message}");
                }
            }
        }

        public override void Dispose()
        {
            lock (_sync)
            {
                _vision?.Dispose();
                _vision = null;
                _exec?.Dispose();
                _exec = null;
            }
            base.Dispose();
        }

        // ---------------------------------------------------------------- sequence slots

        private readonly struct Store : SlotRetention.ISlotStore
        {
            private readonly Qwen35CudaExecutor _exec;
            public Store(Qwen35CudaExecutor exec) => _exec = exec;
            public int Alloc() => _exec.SlotAlloc();
            public bool Select(int slot) => _exec.SetActiveSlot(slot);
            public bool Free(int slot) => _exec.SlotFree(slot);
            public int ActiveHead() => _exec.NPast;
            public bool RewindActive(int tokens) => _exec.Rewind(tokens);
            public bool ResetActive() => _exec.ResetChecked();
            // The engine sizes its scratch at load, so no step ever needs a retained slot's memory.
            public bool SetReclaimable(int[] slots) => true;
            public int TakeReclaimed(int[] buffer) => 0;
        }

        public bool BatchedForwardAvailable => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException("qwen35 on the direct-CUDA engine serves concurrency through sequence slots.");

        public bool SupportsPerSequenceFusedForward => _exec != null;

        public bool HasFusedSequenceCache(string requestId)
        {
            lock (_sync)
                return requestId != null && _slotByRequest != null && _slotByRequest.ContainsKey(requestId);
        }

        public bool BindSequenceCache(string requestId)
        {
            if (string.IsNullOrEmpty(requestId))
                throw new ArgumentException("RequestId required", nameof(requestId));
            lock (_sync)
            {
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                bool fresh = false;
                if (!_slotByRequest.TryGetValue(requestId, out int slot))
                {
                    // Memory full: the idle primary first, then a retained conversation's slot.
                    slot = _exec.SlotAlloc();
                    if (slot < 0) slot = SlotRetention.TakeIdlePrimary(_slotByRequest, ref _primarySlot, _activeSlotKey, new Store(_exec));
                    if (slot < 0 && _retainedSlots is { Count: > 0 }) slot = AllocSlotReclaimingRetained();
                    if (slot < 0)
                        throw new SequenceSlotUnavailableException("qwen35 has no device memory for another sequence slot.");
                    _slotByRequest[requestId] = slot;
                    fresh = true;
                }
                if (!_exec.SetActiveSlot(slot))
                    throw new InvalidOperationException($"qwen35 slot {slot} missing for request {requestId}.");
                _activeSlotKey = requestId;
                _exec.Engine.TraceBindSequence(requestId, _exec.Engine.Active);
                _cacheSeqLen = _exec.NPast;
                return fresh;
            }
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (string.IsNullOrEmpty(requestId)) return;
            lock (_sync)
            {
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (_activeSlotKey != null || _slotByRequest.ContainsKey(requestId)) return;
                _slotByRequest[requestId] = _primarySlot;
                _exec.Engine.TraceBindSequence(requestId, _exec.Slot(_primarySlot));
                _activeSlotKey = requestId;
                _primarySlot = -1;
            }
        }

        // Payloads captured as copies (prompt checkpoints, shared prefixes). Memory takes back
        // retained end states first: a finished request's slot serves only an exact extension
        // of its conversation, while a checkpoint serves every request that repeats its prefix,
        // and a plain oldest-first order freed the checkpoint as soon as end states piled up
        // behind it.
        private readonly HashSet<string> _checkpointKeys = new(StringComparer.Ordinal);
        private static readonly bool RetainDebug = Environment.GetEnvironmentVariable("TS_PREFIX_RELEASE_DEBUG") == "1";

        /// <summary>The retained slot device memory takes next, not already on its way to host
        /// memory: leaves before prefixes (see <see cref="MakeHostRoom"/>), and within each the
        /// oldest end state, else the oldest checkpoint.</summary>
        private string NextReclaimable()
        {
            if (_retainedSlots == null)
                return null;
            bool Leaf(string k) => !IsContinued(k, _exec.Engine.PrefixKey(_exec.Slot(_retainedSlots[k].Slot), _retainedSlots[k].Tokens),
                _retainedSlots[k].Tokens);
            var free = _retainedSlots.Keys.Where(k => !_retiring.ContainsKey(k)).ToList();
            return free.FirstOrDefault(k => !_checkpointKeys.Contains(k) && Leaf(k))
                ?? free.FirstOrDefault(Leaf)
                ?? free.FirstOrDefault(k => !_checkpointKeys.Contains(k))
                ?? free.FirstOrDefault();
        }

        /// <summary>Retained slots whose host copy finished become host payloads and free their
        /// device slots; <paramref name="wait"/> waits for the oldest copy first. True when a slot
        /// was freed (under _sync).</summary>
        private bool HarvestRetired(bool wait)
        {
            bool freed = false;
            var engine = _exec.Engine;
            while (_retiring.Count > 0)
            {
                string key = _retiring.Keys.First();
                var r = _retiring[key];
                if (wait && !freed)
                    engine.WaitCopy(r.Done);
                else if (!engine.CopyDone(r.Done))
                    break;
                _retiring.Remove(key);
                var entry = _retainedSlots[key];
                bool checkpoint = _checkpointKeys.Contains(key);
                if (!_exec.SlotFree(entry.Slot))
                {
                    engine.FreeHostSlot(r.Host);
                    continue;
                }
                _retainedSlots.Remove(key);
                _checkpointKeys.Remove(key);
                _hostRetained[key] = new HostPayload(r.Host, entry.Tokens, checkpoint);
                freed = true;
            }
            return freed;
        }

        /// <summary>Stops moving <paramref name="key"/> to host memory (it is about to be used or
        /// released): waits for its copy and drops it (under _sync).</summary>
        private void CancelRetire(string key)
        {
            if (key == null || !_retiring.Remove(key, out var r))
                return;
            _exec.Engine.WaitCopy(r.Done);
            _exec.Engine.FreeHostSlot(r.Host);
        }

        /// <summary>Starts moving retained slots to host memory while free device memory is short of
        /// the retain reserve plus <see cref="RetireHeadroomBytes"/>, so later allocations find room
        /// without copying on the critical path (under _sync).</summary>
        private void MaintainDeviceHeadroom()
        {
            if (_hostBudgetBytes <= 0 || _exec == null)
                return;
            HarvestRetired(wait: false);
            var engine = _exec.Engine;
            long free = engine.FreeDeviceBytes();
            long pending = 0;
            while (free + pending < RetainReserveBytes + RetireHeadroomBytes)
            {
                string key = NextReclaimable();
                if (key == null)
                    break;
                var slot = _exec.Slot(_retainedSlots[key].Slot);
                if (!MakeHostRoom(engine.HostBytesToCopy(slot)))
                    break;
                var host = engine.BeginCopyToHost(slot, out IntPtr done);
                _retiring[key] = new Retiring(host, done);
                if (RetainDebug)
                    Console.Error.WriteLine($"[q35-retire] {key} tokens={_retainedSlots[key].Tokens} checkpoint={_checkpointKeys.Contains(key)} " +
                        $"free={free >> 20}MiB host={_hostRetained.Count} ({engine.HostBytesInUse >> 20}MiB)");
                pending += engine.SlotStateBytes + engine.SlotKvBytes(slot);
            }
        }

        /// <summary>Called under _sync when the engine needs device memory. Media preparation can
        /// hold _mediaLock while waiting for _sync, so never wait for that lock here. Dispose also
        /// holds _sync: an allocator selected below cannot be disposed by the model during a trim.</summary>
        private bool ReclaimDeviceMemory()
        {
            if (_reclaimVisionPool && _vision != null && _visionAllocator is TensorSharp.Cuda.CudaAllocator allocator)
            {
                bool freed = TryTrimIdleVisionPool(_mediaLock, () => allocator.GetStats().CachedBytes,
                    allocator.TrimPool, out long before, out long after);
                if (before > 0)
                {
                    ++_visionPoolPressureTrimCount;
                    Console.Error.WriteLine($"[q35-vision-pool-reclaim] trim={_visionPoolPressureTrimCount} " +
                        $"cached_before={before} cached_after={after} cached_reduction={before - after}");
                }
                if (freed)
                    return true;
            }
            return ReclaimOneRetained();
        }

        /// <summary>Only trim returned, idle pool blocks. The allocator synchronizes its own
        /// stream before draining; live weights, embeddings and graph-owned blocks stay owned.
        /// A late finalizer may return another block, so only an observed cache reduction counts
        /// as progress. Stats/trim errors propagate, with the media lock released in every case.</summary>
        internal static bool TryTrimIdleVisionPool(object mediaLock, Func<long> cachedBytes, Action trim,
            out long before, out long after)
        {
            before = after = 0;
            if (!Monitor.TryEnter(mediaLock, 0))
                return false;
            try
            {
                before = cachedBytes();
                after = before;
                if (before <= 0)
                    return false;
                trim();
                after = cachedBytes();
                return after < before;
            }
            finally
            {
                Monitor.Exit(mediaLock);
            }
        }

        /// <summary>Frees one retained slot's device memory (called under _sync when an allocation
        /// needs it now): a slot already on its way to host memory first, else the oldest end state,
        /// else the oldest checkpoint, moved to pinned host memory while the host budget has room for
        /// it, else dropped.</summary>
        private bool ReclaimOneRetained()
        {
            if (_retiring.Count > 0 && HarvestRetired(wait: true))
                return true;
            string key = NextReclaimable();
            if (key == null)
                return false;
            var entry = _retainedSlots[key];
            bool checkpoint = _checkpointKeys.Contains(key);
            var engine = _exec.Engine;
            if (RetainDebug)
                Console.Error.WriteLine($"[q35-reclaim] {key} tokens={entry.Tokens} checkpoint={checkpoint} " +
                    $"retained={_retainedSlots.Count} host={_hostRetained.Count} ({engine.HostBytesInUse >> 20}MiB) free={engine.FreeDeviceBytes() >> 20}MiB " +
                    $"pin={engine.HostPinMs:F0}ms copy={engine.HostCopyMs:F0}ms");
            TensorSharp.Cuda.Q35CudaEngine.HostSlot host = null;
            if (_hostBudgetBytes > 0 && MakeHostRoom(engine.HostBytesToCopy(_exec.Slot(entry.Slot))))
            {
                try
                {
                    host = engine.CopyToHost(_exec.Slot(entry.Slot));
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[q35-cuda] moving a retained slot to host memory failed: {ex.Message}");
                    host = null;
                }
            }
            if (!_exec.SlotFree(entry.Slot))
            {
                if (host != null)
                    engine.FreeHostSlot(host);
                return false;
            }
            _retainedSlots.Remove(key);
            _checkpointKeys.Remove(key);
            if (host != null)
                _hostRetained[key] = new HostPayload(host, entry.Tokens, checkpoint);
            else
                _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
            return true;
        }

        /// <summary>Makes room for <paramref name="bytes"/> more in the host budget, dropping host
        /// payloads leaf first: the oldest that no longer retained payload continues (a prefix stays
        /// while anything built on it is kept). False when only prefixes are left (under _sync).</summary>
        private bool MakeHostRoom(long bytes)
        {
            var engine = _exec.Engine;
            if (bytes > _hostBudgetBytes)
                return false;
            while (engine.HostBytesInUse + bytes > _hostBudgetBytes)
            {
                string victim = null;
                foreach (var (key, payload) in _hostRetained)
                    if (!string.Equals(key, _promoting, StringComparison.Ordinal) && !IsContinued(key, payload.Host, payload.Tokens))
                    {
                        victim = key;
                        break;
                    }
                if (victim == null)
                    return false;
                DropHostPayload(victim, InvalidationReason.NativeSlotReclaimed);
            }
            return true;
        }

        /// <summary>Whether a longer retained payload (on host or device) continues the host payload
        /// <paramref name="key"/>: it shares the payload's last full attention block.</summary>
        private bool IsContinued(string key, TensorSharp.Cuda.Q35CudaEngine.HostSlot host, int tokens)
            => IsContinued(key, TensorSharp.Cuda.Q35CudaEngine.PrefixKey(host, tokens), tokens);

        private bool IsContinued(string key, long prefix, int tokens)
        {
            if (prefix == 0)
                return false;
            foreach (var (other, payload) in _hostRetained)
                if (payload.Tokens > tokens && !string.Equals(other, key, StringComparison.Ordinal)
                    && TensorSharp.Cuda.Q35CudaEngine.PrefixKey(payload.Host, tokens) == prefix)
                    return true;
            if (_retainedSlots != null)
                foreach (var (other, entry) in _retainedSlots)
                    if (entry.Tokens > tokens && !string.Equals(other, key, StringComparison.Ordinal)
                        && _exec.Engine.PrefixKey(_exec.Slot(entry.Slot), tokens) == prefix)
                        return true;
            return false;
        }

        private void DropHostPayload(string key, InvalidationReason? reason)
        {
            if (!_hostRetained.Remove(key, out var payload))
                return;
            if (RetainDebug && reason != null)
                Console.Error.WriteLine($"[q35-hostdrop] {key} tokens={payload.Tokens} checkpoint={payload.Checkpoint} " +
                    $"continued={IsContinued(key, payload.Host, payload.Tokens)} host={_hostRetained.Count} ({_exec.Engine.HostBytesInUse >> 20}MiB)");
            _exec.Engine.FreeHostSlot(payload.Host);
            if (reason is InvalidationReason why)
                _prefixCacheSink?.OnPayloadInvalidated(key, why);
        }

        /// <summary>Brings a host payload back into a retained device slot (under _sync); false when
        /// device memory has no slot for it.</summary>
        private bool PromoteHostPayload(string key)
        {
            if (!_hostRetained.TryGetValue(key, out var payload))
                return false;
            _promoting = key;
            try
            {
                int slot = AllocSlotReclaimingRetained();
                if (slot < 0)
                    return false;
                try
                {
                    long t0 = System.Diagnostics.Stopwatch.GetTimestamp();
                    _exec.Engine.CopyFromHost(payload.Host, _exec.Slot(slot));
                    if (RetainDebug)
                        Console.Error.WriteLine($"[q35-promote] {key} tokens={payload.Tokens} checkpoint={payload.Checkpoint} " +
                            $"{System.Diagnostics.Stopwatch.GetElapsedTime(t0).TotalMilliseconds:F0}ms host={_hostRetained.Count}");
                }
                catch
                {
                    _exec.SlotFree(slot);
                    throw;
                }
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                _retainedSlots[key] = new SlotRetention.RetainedSlot(slot, payload.Tokens);
                if (payload.Checkpoint)
                    _checkpointKeys.Add(key);
                DropHostPayload(key, reason: null);
                return true;
            }
            finally
            {
                _promoting = null;
            }
        }

        private bool IsHostPayload(string key, int tokens, int target)
            => key != null && _hostRetained.TryGetValue(key, out var p) && p.Tokens == tokens && target == tokens;

        private int AllocSlotReclaimingRetained()
        {
            int slot = _exec.SlotAlloc();
            while (slot < 0 && ReclaimOneRetained())
                slot = _exec.SlotAlloc();
            return slot;
        }

        public void RestorePrimaryCache()
        {
            lock (_sync)
            {
                if (_activeSlotKey == null) return;
                if (_primarySlot < 0)
                {
                    _primarySlot = AllocSlotReclaimingRetained();
                    if (_primarySlot < 0)
                        throw new SequenceSlotUnavailableException("qwen35 primary-slot allocation failed (device memory exhausted?).");
                }
                if (!_exec.SetActiveSlot(_primarySlot))
                    throw new InvalidOperationException($"qwen35 primary slot {_primarySlot} missing.");
                _activeSlotKey = null;
                _exec.Engine.TraceBindSequence(null, _exec.Engine.Active);
                _cacheSeqLen = _exec.NPast;
            }
        }

        public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            lock (_sync)
            {
                if (_slotByRequest == null) return false;
                int n = requestIds.Count;
                if (n < 2 || n > TensorSharp.Cuda.Q35CudaEngine.MaxSeqs) return false;
                var slots = new int[n];
                for (int i = 0; i < n; i++)
                    if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                        return false;
                int vocab = Config.VocabSize;
                var flat = new float[n * vocab];
                if (_exec.Engine.ServingTraceEnabled)
                    for (int i = 0; i < n; i++)
                        _exec.Engine.TraceBindSequence(requestIds[i], _exec.Slot(slots[i]));
                if (!_exec.ForwardBatchedDecode(slots, tokens, positions, flat)) return false;
                for (int i = 0; i < n; i++)
                {
                    var row = new float[vocab];
                    Array.Copy(flat, (long)i * vocab, row, 0, vocab);
                    outLogits[i] = row;
                }
                _cacheSeqLen = _exec.NPast;
                return true;
            }
        }

        /// <summary>
        /// Concurrent prompt chunks in shared forwards (padded to the longest chunk of each group),
        /// so prefilling sequences share every weight read; a request without a slot gets a fresh one.
        /// Declines, with nothing advanced, when a slot is missing room or sits elsewhere.
        /// </summary>
        public bool TryForwardBatchedFusedPrefill(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> tokens, int[] positions,
            float[][] outLogits)
        {
            lock (_sync)
            {
                int n = requestIds.Count;
                if (n < 2 || _exec == null)
                    return false;
                for (int i = 0; i < n; i++)
                    if (tokens[i] == null || tokens[i].Length == 0 || tokens[i].Length > _exec.Engine.UBatch
                        || positions[i] + tokens[i].Length > _maxContextLength)
                        return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                var slots = new TensorSharp.Cuda.Q35CudaEngine.Slot[n];
                var fresh = new List<string>();
                for (int i = 0; i < n; i++)
                {
                    if (!_slotByRequest.TryGetValue(requestIds[i], out int id))
                    {
                        if (positions[i] != 0)
                            return Undo(fresh);
                        id = AllocSlotReclaimingRetained();
                        if (id < 0)
                            return Undo(fresh);
                        _slotByRequest[requestIds[i]] = id;
                        fresh.Add(requestIds[i]);
                    }
                    slots[i] = _exec.Slot(id);
                    if (slots[i] == null || slots[i].NPast != positions[i])
                        return Undo(fresh);
                }
                if (_exec.Engine.ServingTraceEnabled)
                    for (int i = 0; i < n; i++)
                        _exec.Engine.TraceBindSequence(requestIds[i], slots[i]);
                _exec.Engine.PrefillBatch(slots, tokens.ToArray(), outLogits);
                _cacheSeqLen = _exec.NPast;
                return true;
            }

            bool Undo(List<string> created)
            {
                foreach (string id in created)
                    if (_slotByRequest.Remove(id, out int slot))
                        _exec.SlotFree(slot);
                return false;
            }
        }

        public void OnSequenceReleased(string requestId)
        {
            lock (_sync)
            {
                if (_slotByRequest == null || string.IsNullOrEmpty(requestId) || !_slotByRequest.TryGetValue(requestId, out int slot))
                    return;
                if (string.Equals(_activeSlotKey, requestId, StringComparison.Ordinal))
                {
                    if (_primarySlot < 0)
                    {
                        // A request holds the primary: this slot becomes it.
                        if (SlotRetention.ReleaseActiveSlotAsPrimary(_slotByRequest, requestId, ref _primarySlot, ref _activeSlotKey, new Store(_exec)))
                            _cacheSeqLen = 0;
                        return;
                    }
                    RestorePrimaryCache();
                }
                _slotByRequest.Remove(requestId);
                _exec.SlotFree(slot);
            }
        }

        /// <summary>Engine slots of <paramref name="requestIds"/>, each at its expected position; null
        /// when any request has no slot or sits elsewhere.</summary>
        private TensorSharp.Cuda.Q35CudaEngine.Slot[] SlotsOf(IReadOnlyList<string> requestIds, int[] positions)
        {
            if (_slotByRequest == null)
                return null;
            var slots = new TensorSharp.Cuda.Q35CudaEngine.Slot[requestIds.Count];
            for (int i = 0; i < slots.Length; i++)
            {
                if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out int id))
                    return null;
                slots[i] = _exec.Slot(id);
                if (slots[i] == null || (positions != null && slots[i].NPast != positions[i]))
                    return null;
                _exec.Engine.TraceBindSequence(requestIds[i], slots[i]);
            }
            return slots;
        }

        // ---------------------------------------------------------------- speculation

        private bool HasDrafter => _exec?.Engine?.HasDrafter == true;
        private int DraftWidth => HasDrafter ? _exec.Engine.DraftBlockSize - 1 : 0;

        public bool SpeculationProfitable => HasDrafter;
        private static bool NativeGreedyVerifyEnabled => Environment.GetEnvironmentVariable("TS_Q35_GREEDY_VERIFY") == "1";
        public bool SpecPrefersFixedCostGovernor => _exec?.HasEmbeddedNinferDrafter == true
            && _exec.Engine.NinferFixedDraftGovernorQualified && NativeGreedyVerifyEnabled;
        public bool SpecTrunkFollowsBoundCache => true;
        /// <summary>The drafter's taps never leave the device: the executor's feature rows are a
        /// one-float placeholder.</summary>
        public int SpecFeatureSize => 1;
        /// <summary>Every forwarded and committed row feeds the slot's drafter ring on the device.</summary>
        public bool SpecCarryOnDevice => true;
        public int SpecMaxDraftTokens => DraftWidth;
        /// <summary>A verify leaves the recurrent state alone and SpecOnVerifyAccepted commits the
        /// accepted rows, so a partial acceptance is only a position check.</summary>
        public bool SpecVerifyPersistsAcceptedKv => true;

        /// <summary>
        /// Several rows with every row's logits: a single-sequence verify on the active slot that
        /// writes the attention caches but advances nothing until <see cref="SpecOnVerifyAccepted"/>.
        /// Otherwise an ordinary forward.
        /// </summary>
        public void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            lock (_sync)
            {
                if (allLogitsRows && tokens.Length > 1)
                {
                    if (tokens.Length > TensorSharp.Cuda.Q35CudaEngine.MaxVerifyRows)
                        throw new NotSupportedException($"qwen35 verifies at most {TensorSharp.Cuda.Q35CudaEngine.MaxVerifyRows} rows.");
                    if (_soloDrawn.Length < tokens.Length)
                        _soloDrawn = new int[tokens.Length];
                    _exec.Engine.Verify(new[] { _exec.Engine.Active }, tokens.Length, tokens, _soloDrawn, logitsOut);
                    _soloVerifyRows = tokens.Length;
                }
                else
                    ForwardWithMedia(tokens, logitsOut);
            }
            if (hAllOut != null && hAllOut.Length > 0)
                hAllOut[0] = 0f;
        }

        /// <summary>Token-only version of the solo verify; explicit opt-in until
        /// full sampler/state and model quality gates have passed on the device.</summary>
        public bool TrySpecForwardGreedyVerify(int[] tokens, float[] hAllOut, int[] drawnOut)
        {
            if (!NativeGreedyVerifyEnabled
                || tokens.Length < 2 || tokens.Length > TensorSharp.Cuda.Q35CudaEngine.MaxVerifyRows)
                return false;
            if (drawnOut == null || drawnOut.Length < tokens.Length)
                throw new ArgumentException("A greedy verify needs one output token per row.", nameof(drawnOut));
            lock (_sync)
            {
                _exec.Engine.Verify(new[] { _exec.Engine.Active }, tokens.Length, tokens, drawnOut);
                _soloVerifyRows = tokens.Length;
            }
            if (hAllOut != null && hAllOut.Length > 0)
                hAllOut[0] = 0f;
            return true;
        }

        public void SpecOnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            lock (_sync)
            {
                if (_soloVerifyRows == 0)
                    return;
                _exec.Engine.Commit(new[] { _exec.Engine.Active }, _soloVerifyRows, new[] { acceptedRows + 1 });
                _soloVerifyRows = 0;
                _cacheSeqLen = _exec.NPast;
            }
        }

        public void SpecEnsureCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"qwen35 slots hold {_maxContextLength} tokens; {requiredSeqLen} requested.");
        }

        public void SpecSnapshotRecurrentState() { }

        public void SpecRestoreRecurrentState()
            => throw new NotSupportedException("qwen35 on the direct-CUDA engine keeps its recurrent state until a verify commits.");

        public void SpecRewindCache(int length)
        {
            if (length != _exec.NPast)
                throw new NotSupportedException($"qwen35 on the direct-CUDA engine cannot rewind from {_exec.NPast} to {length}.");
        }

        public DraftHeadKind DraftHeadKind => HasDrafter ? DraftHeadKind.Block : DraftHeadKind.None;
        public int DraftBlockSize => DraftWidth;
        public bool DraftSelfCatchUp => true;
        public int DraftContextRows => HasDrafter ? _exec.Engine.DraftRingRows : int.MaxValue;
        public bool DraftHeadResumesAfterGap => true;

        public int DraftBlock(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            lock (_sync)
            {
                var slots = _draftRequest != null ? SlotsOf(new[] { _draftRequest }, new[] { position }) : null;
                var slot = slots?[0] ?? (_exec.NPast == position ? _exec.Engine.Active : null);
                if (slot == null)
                    return 0;
                int g = DraftWidth;
                var d = new[] { new int[g] };
                var c = new[] { new float[g] };
                _exec.Engine.Draft(new[] { slot }, new[] { lastToken }, d, c);
                int n = Math.Min(g, draftOut.Length);
                Array.Copy(d[0], draftOut, n);
                if (confOut != null)
                    Array.Copy(c[0], confOut, Math.Min(n, confOut.Length));
                return n;
            }
        }

        /// <summary>The engine writes every committed row into its slot's drafter ring itself.</summary>
        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos) { }

        public int BatchedSpecWindow(int sequences)
        {
            if (!HasDrafter || sequences < 1 || sequences > TensorSharp.Cuda.Q35CudaEngine.MaxSeqs)
                return 0;
            if (_exec.Engine.HasNinferDrafter && sequences > _exec.Engine.NinferDraftMaximumLanes)
                return 0;
            return Math.Min(DraftWidth + 1, _exec.Engine.NinferExactVerifyMaximumRows / sequences);
        }

        public int FusedCacheSeqLen(string requestId)
        {
            lock (_sync)
                return requestId != null && _slotByRequest != null && _slotByRequest.TryGetValue(requestId, out int id)
                    && _exec.SlotHead(id, out int head) ? head : -1;
        }

        public int FusedCacheFreeRows(string requestId)
        {
            int head = FusedCacheSeqLen(requestId);
            return head < 0 ? 0 : _maxContextLength - head;
        }

        public void BindDraftState(string requestId) => _draftRequest = requestId;

        public void ReleaseDraftState(string requestId)
        {
            if (string.Equals(_draftRequest, requestId, StringComparison.Ordinal))
                _draftRequest = null;
        }

        private sealed class NoScope : IDisposable
        {
            public static readonly NoScope Instance = new();
            public void Dispose() { }
        }

        public IDisposable BeginBatchedSpeculation() => NoScope.Instance;

        public int BatchDraftBlockWidth => DraftWidth;

        public bool TryDraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
            int[][] tokensOut, float[][] confOut)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, positions);
                if (slots == null || !HasDrafter)
                    return false;
                if (_exec.Engine.HasNinferDrafter && slots.Length > _exec.Engine.NinferDraftMaximumLanes)
                    return false;
                _exec.Engine.Draft(slots, anchors, tokensOut, confOut);
                return true;
            }
        }

        public bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, int[] drawn, float[] features)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, positions);
                if (slots == null || window < 1 || (long)slots.Length * window > _exec.Engine.NinferExactVerifyMaximumRows)
                    return false;
                foreach (var s in slots)
                    if (s.NPast + window > _maxContextLength)
                        return false;
                if (TryMixedVerify(requestIds, slots, window, tokens, drawn))
                    _cacheSeqLen = _exec.NPast;
                else
                    _exec.Engine.Verify(slots, window, tokens, drawn);
                _verifyWindow = window;
                return true;
            }
        }

        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            lock (_sync)
            {
                var slots = SlotsOf(requestIds, positions)
                    ?? throw new InvalidOperationException("qwen35 verify commit: a request lost its slot between verify and commit.");
                _exec.Engine.Commit(slots, _verifyWindow, kept);
                _cacheSeqLen = _exec.NPast;
            }
        }

        // ---------------------------------------------------------------- mixed step

        // TS_Q35_MIXED_STEP=0 runs a step's prompt chunk and its verify as separate forwards.
        private static readonly bool MixedStep = Environment.GetEnvironmentVariable("TS_Q35_MIXED_STEP") != "0";
        // The mixed step also fuses the chunks and verify windows of up to 8 rows that would take the GEMV's
        // 8-bit activations on their own: in the mixed forward they share the chunk's FP4 GEMM, as batched
        // verifies past 8 rows always do (and as Velox's mixed step does), instead of a forward of their own.
        // Qwen3.8-27B replay: image_replace c=2 1.05 -> 0.95 of Velox, incremental c=4 1.01 -> 0.91, with the
        // same validation failures. TS_Q35_MIXED_FP4_VERIFY=0 keeps them on the GEMV.
        private static readonly bool MixedFp4Verify = Environment.GetEnvironmentVariable("TS_Q35_MIXED_FP4_VERIFY") != "0";

        private static bool MixedNumericsAllowed(int chunkRows, int verifyRows) =>
            MixedFp4Verify || TensorSharp.Cuda.Q35CudaEngine.MixedKeepsNumerics(chunkRows, verifyRows);
        private const int MixedTileRows = 128;

        /// <summary>A prompt chunk waiting to ride the next batched verify, with the media queued for
        /// it; Ran and Logits once a verify carried it, Failure when that forward threw.</summary>
        private sealed class MixedChunk
        {
            public string RequestId;
            public int[] Tokens;
            public int Position;
            public bool Whole;
            public int[] MRoPE;
            public TensorSharp.Cuda.Q35CudaEngine.EmbeddingSpan[] Spans;
            public Tensor[] Tensors;
            public int Ran;
            public float[] Logits;
            public Exception Failure;
        }

        private MixedChunk _mixed;

        /// <summary>A decode round that carries a whole 2048-token chunk takes ~220 ms, so a request
        /// that only decodes beside a cold prompt waits that long per round; 1024 halves it at a
        /// small cost to the prompt (Qwen3.8-27B replay: image_replace c=2 1.02 -> 0.97 of Velox,
        /// c=4 0.99 -> 0.94, the rest unchanged within noise).</summary>
        public int PreferredMixedStepPrefillTokens => MixedStep && _ninferCacheIdentity == null ? 1024 : 0;

        public bool StageMixedPrefill(string requestId, int[] tokens, int position, bool whole)
        {
            lock (_sync)
            {
                EndMixed();
                // The media queued for the chunk's slice is the chunk's from here on.
                _mixed = new MixedChunk
                {
                    RequestId = requestId,
                    Tokens = tokens,
                    Position = position,
                    Whole = whole,
                    MRoPE = _pendingMRoPE,
                    Spans = _pendingVision.ToArray(),
                    Tensors = _pendingVisionTensors.ToArray(),
                };
                _pendingMRoPE = null;
                _pendingVision.Clear();
                _pendingVisionTensors.Clear();
                // A chunk of a few tokens (a prompt's tail) takes the GEMV's 8-bit activations on its own:
                // see MixedFp4Verify.
                if (_ninferCacheIdentity != null || !MixedStep || requestId == null || tokens == null
                    || !MixedNumericsAllowed(tokens.Length, int.MaxValue) || tokens.Length > _exec.Engine.UBatch
                    || position + tokens.Length > _maxContextLength || SlotsOf(new[] { requestId }, new[] { position }) == null)
                {
                    EndMixed();
                    return false;
                }
                return true;
            }
        }

        public bool TakeMixedPrefill(string requestId, out int tokensRun, out float[] logits)
        {
            lock (_sync)
            {
                var chunk = _mixed;
                EndMixed();
                tokensRun = 0;
                logits = null;
                if (chunk == null || !string.Equals(chunk.RequestId, requestId, StringComparison.Ordinal))
                    return false;
                if (chunk.Failure != null)
                    throw new InvalidOperationException($"qwen35 mixed step for {requestId} failed.", chunk.Failure);
                if (chunk.Logits == null)
                    return false;
                tokensRun = chunk.Ran;
                logits = chunk.Logits;
                return true;
            }
        }

        /// <summary>Drops the staged chunk and the media rows it holds (under _sync).</summary>
        private void EndMixed()
        {
            var chunk = _mixed;
            _mixed = null;
            if (chunk?.Tensors is { Length: > 0 })
            {
                // A forward may still read them on the engine's stream.
                _exec.Engine.Synchronize();
                foreach (var t in chunk.Tensors)
                    t.Dispose();
                chunk.Tensors = null;
            }
        }

        /// <summary>The staged chunk and a verify of <paramref name="slots"/> in one forward; false,
        /// with nothing run, when there is none or it cannot join (under _sync).</summary>
        private bool TryMixedVerify(IReadOnlyList<string> requestIds, TensorSharp.Cuda.Q35CudaEngine.Slot[] slots, int window, int[] tokens,
            int[] drawn)
        {
            var chunk = _mixed;
            if (chunk == null || chunk.Logits != null || chunk.Failure != null)
                return false;
            foreach (string id in requestIds)
                if (string.Equals(id, chunk.RequestId, StringComparison.Ordinal))
                    return false;
            var prefill = SlotsOf(new[] { chunk.RequestId }, new[] { chunk.Position })?[0];
            if (prefill == null || Array.IndexOf(slots, prefill) >= 0)
                return false;
            int take = MixedChunkRows(chunk.Tokens.Length, slots.Length * window, chunk.Whole);
            // A lone sequence's verify takes the GEMV's 8-bit activations on its own: see MixedFp4Verify.
            if (!MixedNumericsAllowed(take, slots.Length * window))
                return false;
            var run = take == chunk.Tokens.Length ? chunk.Tokens : chunk.Tokens[..take];
            var logits = new float[Config.VocabSize];
            try
            {
                _exec.Engine.MixedVerify(prefill, run, logits, slots, window, tokens, drawn, RopeComponents(chunk.MRoPE, take), chunk.Spans);
            }
            catch (Exception ex)
            {
                chunk.Failure = ex;
                throw;
            }
            chunk.Ran = take;
            chunk.Logits = logits;
            return true;
        }

        /// <summary>The chunk rows to run beside <paramref name="verifyRows"/> verify rows: all of them
        /// when the chunk must run whole or the verify rows fit its last 128-row GEMM tile; else the
        /// rows that keep its tiles (the rest leads the next chunk).</summary>
        private int MixedChunkRows(int rows, int verifyRows, bool whole)
        {
            int cap = _exec.Engine.MixedRowCapacity - verifyRows;
            int tiles = (rows + MixedTileRows - 1) / MixedTileRows;
            if (rows <= cap && (whole || rows + verifyRows <= tiles * MixedTileRows))
                return rows;
            return Math.Min(Math.Min(rows, cap), tiles * MixedTileRows - verifyRows);
        }

        // ---------------------------------------------------------------- IPrefixCacheModel

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new PrefixCacheCapabilities
        {
            Class = FamilyClass.N,
            NamespaceFingerprint = KVStateFingerprint,
            // A checkpoint is a copy of the request's slot (recurrent states, drafter ring and the
            // attention rows so far): the only way a prompt's state survives the decode that
            // follows it, since the recurrence cannot be rewound.
            EndState = EndStateSupport.CopyAndDonate,
            CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = true,
            PrimaryResident = true,
            MinRetainTokens = 32,
            // The recurrence cannot be rewound: only an exact extension reuses a slot.
            Truncation = TruncationKind.None,
            TruncationGranularity = 1,
            RewindCapTokens = 16,
            Pages = PageSupport.None,
            // A slot carries its M-RoPE delta (copied with every checkpoint), so a cached prefix
            // that ends past an image resumes at the positions a full prefill would use.
            ReuseAcrossMediaSpan = true,
            // Checkpoints share the attention blocks: one past the media costs only the recurrent state.
            CheckpointPromptTailWithMedia = true,
            CountCapsByBytes = true,
            Persistable = false,
            MaxRetainedNativeSlots = 0,
        };

        public void AttachPrefixCache(IPrefixPayloadSink sink)
        {
            lock (_sync)
                _prefixCacheSink = sink ?? throw new ArgumentNullException(nameof(sink));
        }

        public void DetachPrefixCache()
        {
            lock (_sync)
                _prefixCacheSink = null;
        }

        public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey))
                return false;
            lock (_sync)
            {
                // The request's own slot when it has one (a batched prefill leaves whichever slot was
                // active selected), else the primary it runs on.
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                if (_retainedSlots.ContainsKey(payloadKey))
                    return false;
                var from = _slotByRequest != null && _slotByRequest.TryGetValue(requestId, out int own)
                    ? _exec.Slot(own) : _exec.Engine.Active;
                int dst = AllocSlotReclaimingRetained();
                if (dst < 0)
                    return false;
                _exec.Engine.CopySlot(from, _exec.Slot(dst), from.NPast);
                _retainedSlots[payloadKey] = new SlotRetention.RetainedSlot(dst, from.NPast);
                _checkpointKeys.Add(payloadKey);
                if (RetainDebug)
                    Console.Error.WriteLine($"[q35-capture] {payloadKey} tokens={from.NPast} request={requestId}");
                footprint = Footprint(payloadKey);
                MaintainDeviceHeadroom();
                return true;
            }
        }

        /// <summary>A copy of a retained payload as <paramref name="requestId"/>'s slot (its own,
        /// overwritten, or a new one). The payload stays cached; while the copy's slot is found it
        /// is out of the reclaim order, and it comes back as the newest entry.</summary>
        private bool CloneRetained(string payloadKey, string requestId)
        {
            if (!_retainedSlots.Remove(payloadKey, out var payload))
                return false;
            try
            {
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (!_slotByRequest.TryGetValue(requestId, out int target))
                {
                    target = AllocSlotReclaimingRetained();
                    if (target < 0)
                        return false;
                    _slotByRequest[requestId] = target;
                }
                var dst = _exec.Slot(target);
                _exec.Engine.ResetSlot(dst);
                _exec.Engine.CopySlot(_exec.Slot(payload.Slot), dst, payload.Tokens);
                if (ReferenceEquals(dst, _exec.Engine.Active))
                    _cacheSeqLen = dst.NPast;
                return true;
            }
            finally
            {
                _retainedSlots[payloadKey] = payload;
            }
        }

        public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_sync)
            {
                if (_slotByRequest == null) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                if (!SlotRetention.RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, canRewind: false,
                        ref _primarySlot, ref _activeSlotKey, new Store(_exec)))
                    return false;
                _cacheSeqLen = _exec.NPast;
                footprint = Footprint(payloadKey);
                MaintainDeviceHeadroom();
                return true;
            }
        }

        public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_sync)
            {
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                if (!SlotRetention.ConvertPrimarySlot(_retainedSlots, payloadKey, length, ref _primarySlot, ref _activeSlotKey, new Store(_exec)))
                    return false;
                _cacheSeqLen = _exec.NPast;
                footprint = Footprint(payloadKey);
                MaintainDeviceHeadroom();
                return true;
            }
        }

        public bool TryMaterialize(in MaterializeRequest request)
        {
            if (string.IsNullOrEmpty(request.PayloadKey) || string.IsNullOrEmpty(request.TargetRequestId))
                return false;
            lock (_sync)
            {
                CancelRetire(request.PayloadKey);
                if (IsHostPayload(request.PayloadKey, request.PayloadTokens, request.TargetTokens)
                    && !PromoteHostPayload(request.PayloadKey))
                    return false;
                if (!SlotRetention.CanDonate(_retainedSlots, request.PayloadKey, request.PayloadTokens, request.TargetTokens, canRewind: false))
                    return false;
                if (request.Op == MaterializeOp.Clone)
                    return CloneRetained(request.PayloadKey, request.TargetRequestId);
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                if (!SlotRetention.DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _))
                    return false;
                _checkpointKeys.Remove(request.PayloadKey);
                return true;
            }
        }

        public bool TryReturnDonation(string requestId, string payloadKey)
        {
            lock (_sync)
            {
                if (_slotByRequest == null) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                return SlotRetention.ReturnSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, ref _primarySlot, ref _activeSlotKey,
                    new Store(_exec));
            }
        }

        public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        {
            lock (_sync)
                return SlotRetention.CanDonate(_retainedSlots, payloadKey, payloadTokens, targetTokens, canRewind: false)
                    || IsHostPayload(payloadKey, payloadTokens, targetTokens);
        }

        public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_sync)
            {
                foreach (string key in payloadKeys)
                    CancelRetire(key);
                SlotRetention.ReleaseSlots(_retainedSlots, payloadKeys, new Store(_exec));
                foreach (string key in payloadKeys)
                {
                    _checkpointKeys.Remove(key);
                    DropHostPayload(key, reason: null);
                }
            }
        }

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_sync)
            {
                if (payloadKey != null && _hostRetained.TryGetValue(payloadKey, out var host))
                    return new PayloadFootprint(host.Tokens, host.Tokens, new ResourceVector { StateSnapshot = _exec.Engine.SlotStateBytes },
                        PositionDelta: host.Host.RopeDelta);
                return payloadKey != null && _retainedSlots != null && _retainedSlots.ContainsKey(payloadKey)
                    ? Footprint(payloadKey) : default;
            }
        }

        /// <summary>A retained slot's device cost: its recurrent states and drafter ring, plus its
        /// attention blocks for an end state (a checkpoint's blocks are shared with the sequence it
        /// was taken from). With host memory behind the device, where a retained slot lives is the
        /// model's cache to manage and its blocks are mostly shared with its relatives (counting
        /// them per payload evicted whole conversations at a fraction of the real use): only the
        /// states are charged, against device and host room together.</summary>
        private PayloadFootprint Footprint(string payloadKey)
        {
            var entry = _retainedSlots[payloadKey];
            var slot = _exec.Slot(entry.Slot);
            var bytes = new ResourceVector
            {
                StateSnapshot = _exec.Engine.SlotStateBytes,
                DeviceKv = _hostBudgetBytes > 0 || _checkpointKeys.Contains(payloadKey) ? 0 : _exec.Engine.SlotKvBytes(slot),
            };
            return new PayloadFootprint(entry.Tokens, Math.Max(entry.Tokens, slot.Capacity), bytes, PositionDelta: slot.RopeDelta);
        }

        public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => default;
        public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
        public bool TryExport(string payloadKey, System.IO.Stream destination) => false;

        public bool TryImport(string payloadKey, int tokens, System.IO.Stream source, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public bool TryBeginImport(int tokens, out object importTicket)
        {
            importTicket = null;
            return false;
        }

        public bool RunImportRead(object importTicket, System.IO.Stream source) => false;

        public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
        {
            footprint = default;
            return false;
        }

        public void AbortImport(object importTicket) { }

        /// <summary>Headroom for retained slots: device memory beyond room for running sequences to
        /// grow, plus the host budget not yet used (memory moves retained slots there) short of a
        /// margin, so the coordinator evicts by its order while slots still fit on the host
        /// instead of the device dropping them when both are full.</summary>
        public long QuerySpareBytes(ResourceClass cls)
        {
            if (cls is not (ResourceClass.DeviceKv or ResourceClass.StateSnapshot or ResourceClass.NativeSlot))
                return QueryPrefixCacheSpareBytes(cls);
            lock (_sync)
                return _exec == null ? -1 : Math.Max(0, _exec.Engine.FreeDeviceBytes() - RetainReserveBytes)
                    + Math.Max(0, _hostBudgetBytes - _exec.Engine.HostBytesInUse - HostMarginBytes(_hostBudgetBytes));
        }

        /// <summary>Host budget kept free of retained slots for the next moves to host memory.</summary>
        private static long HostMarginBytes(long budget) => Math.Min(budget / 4, 4L << 30);

        /// <summary>Device memory kept clear of retained slots for running sequences' growth.</summary>
        private const long RetainReserveBytes = 3L << 30;

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys
        {
            get
            {
                lock (_sync)
                    return (_retainedSlots?.Keys ?? Enumerable.Empty<string>()).Concat(_hostRetained.Keys).ToArray();
            }
        }

        public int PrivateHolderCount
        {
            get
            {
                lock (_sync)
                    return _slotByRequest?.Count ?? 0;
            }
        }

        public int PrimaryCacheLength
        {
            get
            {
                lock (_sync)
                    return _activeSlotKey == null && _primarySlot >= 0 ? _exec.NPast : 0;
            }
        }
    }
}
