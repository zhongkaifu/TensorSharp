// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.8-Flash-Next on --backend cuda: the direct-CUDA whole-model engine
// (TensorSharp.Backends.Cuda/Qwen4Exp), layer-split across the visible GPUs. The model's managed
// state stays on a host allocator; the engine owns every weight and cache. Concurrent requests
// run on the engine's sequence slots, decoded together in batched steps, and a finished
// conversation's slot is retained for its next turn (donate-only prefix reuse, as GLM's).
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Models
{
    public sealed class Qwen4ExpCudaModel : ModelBase, IBatchedPagedModel, IPrefixCacheModel, IPrefixCacheModelDiagnostics,
        IServingTelemetrySource
    {
        private Qwen4ExpCudaExecutor _exec;
        private readonly object _sync = new object();

        // requestId -> engine slot. Guarded by _sync.
        private Dictionary<string, int> _slotByRequest;
        // The slot serving the single-stream (N==1) path; -1 while a request holds it.
        private int _primarySlot;
        // Request whose slot is currently active, or null when the primary is.
        private string _activeSlotKey;
        // payload key -> retained slot, oldest first (the order memory takes them back in).
        private OrderedDictionary<string, SlotRetention.RetainedSlot> _retainedSlots;
        private IPrefixPayloadSink _prefixCacheSink;
        private bool _batchedChunkingLatched;
        // A batched decode step's logits: one flat buffer for the engine's rows, and two arrays per slot used in
        // turn, so a step allocates nothing. A sequence's LastLogits (the array its step wrote) is read by the next
        // step's sampler before that step's forward writes the slot's other array.
        private float[] _batchFlat;
        private float[][][] _slotLogits;
        private int[] _slotLogitsTurn;

        public Qwen4ExpCudaModel(string ggufPath, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            int layerSplitDegree = 1, string draftGgufPath = null)
            : base(ggufPath, Validate(tpDegree, tpGroup, layerSplitDegree), 1, null)
        {
            Config = new ModelConfig { Architecture = Qwen4ExpModel.ArchitectureId };
            try
            {
                ParseBaseConfig();
                ParseTokenizer();
                if (!string.IsNullOrWhiteSpace(draftGgufPath))
                    throw new NotSupportedException(
                        "Qwen3.8-Flash-Next's MTP draft head is not built on --backend cuda yet; drop --draft-model " +
                        "or run --backend ggml_cuda.");
                // The direct-CUDA engine has no host-expert seam; the GGML token span does.
                MoeCpuOffloadConfig.WarnUnsupportedBackend(Qwen4ExpModel.ArchitectureId, "cuda");
                int maxContext = ResolveConfiguredContextLength();
                // The checkpoint advertises 262144 tokens: ~7 GiB of attention and indexer rows per
                // sequence slot. Keep a practical default unless MAX_CONTEXT names one.
                if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT")))
                    maxContext = Math.Min(maxContext, 65536);
                int nUbatch = ParseEnvInt("TS_Q4E_UBATCH", 512);
                _exec = new Qwen4ExpCudaExecutor(ggufPath, maxContext, nUbatch, Math.Max(1, layerSplitDegree));
                _exec.ReclaimRetainedSlot = ReclaimOldestRetainedSlot;
                _exec.StageObserver = RecordStage;
                _primarySlot = _exec.ActiveSlot;
                _maxContextLength = _exec.ContextSize;
                Config.VocabSize = _exec.VocabSize;
                _logitsBuffer = new float[Config.VocabSize];
                Console.WriteLine($"Model: {Qwen4ExpModel.ArchitectureId} on the direct-CUDA engine, Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Vocab={Config.VocabSize}, n_ctx={_maxContextLength}");
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>The engine places whole layers itself; the base class only needs a host allocator.</summary>
        private static BackendType Validate(int tpDegree, ITensorParallelGroup tpGroup, int layerSplitDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (layerSplitDegree < 1) throw new ArgumentOutOfRangeException(nameof(layerSplitDegree));
            if (tpGroup != null || tpDegree > 1)
                throw new NotSupportedException(
                    "Qwen3.8-Flash-Next on --backend cuda places whole layers per GPU; use --layer-split N instead of --tp.");
            return BackendType.Cpu;
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        /// <summary>The engine behind this model, for tests that probe its stages.</summary>
        internal TensorSharp.Cuda.Q4eCudaEngine Engine => _exec?.Engine;

        /// <summary>A delta recurrence cannot run backwards, but a slot keeps a checkpoint of its states
        /// at its last prompt boundary: it rewinds to any position from there to its head (the engine
        /// replays its own tokens past the checkpoint), or to 0. Anything else is refused, so callers
        /// use the refusable <see cref="ModelBase.TryTruncateKVCache"/>.</summary>
        public override bool SupportsKVCacheTruncation => _exec != null;

        public override bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
        {
            lock (_sync)
                return _exec != null && _exec.CanRewind(_exec.ActiveSlot, cachedTokenCount, targetTokenCount);
        }

        public override string KVStateFingerprint =>
            $"qwen4exp|exec=cuda|L={Config.NumLayers}|H={Config.HiddenSize}|V={Config.VocabSize}|ctx={_maxContextLength}";

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_sync)
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                _exec.Forward(tokens, _logitsBuffer);
                _cacheSeqLen = _exec.NPast;
                return _logitsBuffer;
            }
        }

        protected override void ResetKVCacheCore()
        {
            lock (_sync)
            {
                if (!_exec.ResetChecked())
                    throw new InvalidOperationException("qwen4exp reset failed; the slot remains unusable.");
                _cacheSeqLen = 0;
            }
        }

        /// <summary>The engine moves a slot only to its own head or to 0; anything else is refused
        /// and reported, never swallowed.</summary>
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
                    $"qwen4exp cannot truncate its caches to {tokenCount} tokens (head at {_cacheSeqLen}); its " +
                    "recurrent state only restarts from 0. Use TryTruncateKVCache and re-prefill when it declines.");
        }

        public override void WarmUpKernels()
        {
            // One tiny forward loads every kernel module and sizes the matmul scratch, so the first
            // real request does not pay for it mid-prompt.
            try
            {
                ForwardCore(new[] { Tokenizer?.BosTokenId ?? 0 });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[q4e-cuda] warmup forward failed: {ex.Message}");
            }
            finally
            {
                ResetKVCacheCore();
            }
        }

        public override void Dispose()
        {
            lock (_sync)
            {
                _exec?.Dispose();
                _exec = null;
            }
            base.Dispose();
        }

        // ---------------------------------------------------------------- sequence slots

        private readonly struct Store : SlotRetention.ISlotStore
        {
            private readonly Qwen4ExpCudaExecutor _exec;
            public Store(Qwen4ExpCudaExecutor exec) => _exec = exec;
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
            => throw new NotSupportedException("qwen4exp on the direct-CUDA engine serves concurrency through sequence slots.");

        public bool SupportsPerSequenceFusedForward => _exec != null;

        /// <summary>Each request's caches are an engine slot sized for the whole context, so the
        /// scheduler's block pool only tracks reservations: sized for one context in all, it capped
        /// every request's tokens together at one context (two 100K-token agent turns at
        /// MAX_CONTEXT=262144) while the GPUs had room for more slots. A full device answers
        /// <see cref="SequenceSlotUnavailableException"/>, and the request waits for a slot.</summary>
        public bool UsesNativeSequenceSlots => _exec != null;

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
                        throw new SequenceSlotUnavailableException("qwen4exp has no device memory for another sequence slot.");
                    _slotByRequest[requestId] = slot;
                    fresh = true;
                }
                if (!_exec.SetActiveSlot(slot))
                    throw new InvalidOperationException($"qwen4exp slot {slot} missing for request {requestId}.");
                _activeSlotKey = requestId;
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
                _activeSlotKey = requestId;
                _primarySlot = -1;
            }
        }

        /// <summary>A running slot that cannot grow takes memory back from the oldest retained
        /// conversation, as a new slot does (<see cref="AllocSlotReclaimingRetained"/>).</summary>
        private bool ReclaimOldestRetainedSlot()
        {
            lock (_sync)
            {
                if (_retainedSlots == null || _retainedSlots.Count == 0) return false;
                string oldest = _retainedSlots.Keys.First();
                SlotRetention.ReleaseSlots(_retainedSlots, new[] { oldest }, new Store(_exec));
                System.Threading.Interlocked.Increment(ref _reclaimsForGrowth);
                _prefixCacheSink?.OnPayloadInvalidated(oldest, InvalidationReason.NativeSlotReclaimed);
                return true;
            }
        }

        private int AllocSlotReclaimingRetained()
        {
            var reclaimed = new List<string>(0);
            int slot = SlotRetention.AllocReclaiming(_retainedSlots, new Store(_exec), reclaimed);
            System.Threading.Interlocked.Add(ref _reclaimsForNewSlot, reclaimed.Count);
            foreach (string key in reclaimed)
                _prefixCacheSink?.OnPayloadInvalidated(key, InvalidationReason.NativeSlotReclaimed);
            return slot;
        }

        // ---------------------------------------------------------------- serving telemetry

        private long _reclaimsForNewSlot, _reclaimsForGrowth;
        private static readonly KeyValuePair<string, object> s_architectureTag =
            new(TelemetryAttributes.ModelArchitecture, Qwen4ExpModel.ArchitectureId);

        /// <summary>The engine's counters and this model's slot ownership, for TensorSharp's
        /// instruments. Runs on the metrics thread and never takes <c>_sync</c>: a forward holds
        /// it for its whole step, so under load a collection would wait behind GPU work, or,
        /// skipping when busy, never see a running slot. The two ownership counts are read
        /// racily instead; <c>Count</c> is a field read on both collections, so a read during an
        /// update is at worst one off for that collection, which a gauge tolerates.</summary>
        public ServingTelemetrySnapshot ReadServingTelemetry()
        {
            var exec = _exec;
            if (exec == null)
                return default;
            // Slots bound to a request (one that adopted the primary included).
            int running = System.Threading.Volatile.Read(ref _slotByRequest)?.Count ?? 0;
            int retained = System.Threading.Volatile.Read(ref _retainedSlots)?.Count ?? 0;
            var c = exec.ReadTelemetry();
            return new ServingTelemetrySnapshot
            {
                RunningSlots = running,
                RetainedSlots = retained,
                SlotBytes = c.SlotBytes,
                SlotCapacityTokens = c.SlotCapacityTokens,
                SlotGrows = c.Grows,
                SlotGrowFailures = c.GrowFailures,
                ReclaimsForNewSlot = System.Threading.Interlocked.Read(ref _reclaimsForNewSlot),
                ReclaimsForGrowth = System.Threading.Interlocked.Read(ref _reclaimsForGrowth),
                RewindsToCheckpoint = c.RewindsToCheckpoint,
                RewindsWithReplay = c.RewindsWithReplay,
                RewindsToZero = c.RewindsToZero,
                RewindsRefused = c.RewindsRefused,
                ReplayedTokens = c.ReplayedTokens,
                SoloGraphCaptures = c.SoloGraphCaptures,
                SoloGraphLaunches = c.SoloGraphLaunches,
                BatchedGraphCaptures = c.BatchedGraphCaptures,
                BatchedGraphLaunches = c.BatchedGraphLaunches,
                GraphCaptureFailures = c.GraphCaptureFailures,
                ForwardIssueSeconds = c.ForwardIssueSeconds,
                ForwardWaitSeconds = c.ForwardWaitSeconds,
            };
        }

        /// <summary>One sampled forward's stage (TS_Q4E_PERF_SAMPLE), as tensorsharp.engine.stage.duration.</summary>
        private static void RecordStage(string stage, double seconds, int rows, bool decode)
        {
            var histogram = TensorSharpMetrics.EngineStageDuration;
            if (!histogram.Enabled)
                return;
            histogram.Record(seconds, s_architectureTag,
                new KeyValuePair<string, object>(TelemetryAttributes.EngineStage, stage),
                new KeyValuePair<string, object>(TelemetryAttributes.StepKind, decode ? "decode" : "prefill"),
                new KeyValuePair<string, object>(TelemetryAttributes.RowsBucket, TelemetryAttributes.RowsBucketOf(rows)));
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
                        throw new SequenceSlotUnavailableException("qwen4exp primary-slot allocation failed (device memory exhausted?).");
                }
                if (!_exec.SetActiveSlot(_primarySlot))
                    throw new InvalidOperationException($"qwen4exp primary slot {_primarySlot} missing.");
                _activeSlotKey = null;
                _cacheSeqLen = _exec.NPast;
            }
        }

        public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
        {
            lock (_sync)
            {
                if (_slotByRequest == null) return false;
                int n = requestIds.Count;
                if (n < 2) return false;
                var slots = new int[n];
                for (int i = 0; i < n; i++)
                    if (requestIds[i] == null || !_slotByRequest.TryGetValue(requestIds[i], out slots[i]))
                        return false;
                int vocab = Config.VocabSize;
                if ((long)n * vocab > int.MaxValue) return false;
                if (_batchFlat == null || _batchFlat.Length < n * vocab)
                    _batchFlat = new float[n * vocab];
                var flat = _batchFlat;
                if (!ForwardBatchedDecodeChunks(slots, tokens, positions, flat, vocab)) return false;
                int maxSlot = 0;
                for (int i = 0; i < n; i++)
                    maxSlot = Math.Max(maxSlot, slots[i]);
                if (_slotLogits == null || _slotLogits.Length <= maxSlot)
                {
                    Array.Resize(ref _slotLogits, maxSlot + 1);
                    Array.Resize(ref _slotLogitsTurn, maxSlot + 1);
                }
                var rows = new float[n][];
                for (int i = 0; i < n; i++)
                {
                    int slot = slots[i];
                    var pair = _slotLogits[slot] ??= new float[2][];
                    int turn = _slotLogitsTurn[slot] ^= 1;
                    rows[i] = pair[turn] != null && pair[turn].Length == vocab ? pair[turn] : (pair[turn] = new float[vocab]);
                }
                // 1 MB a row: the copies run side by side.
                if (n >= 4)
                    System.Threading.Tasks.Parallel.For(0, n, i => Array.Copy(flat, (long)i * vocab, rows[i], 0, vocab));
                else
                    for (int i = 0; i < n; i++)
                        Array.Copy(flat, (long)i * vocab, rows[i], 0, vocab);
                for (int i = 0; i < n; i++)
                    outLogits[i] = rows[i];
                return true;
            }
        }

        /// <summary>One batched decode step in windows of at most 16 sequences, none of them 1. A failure
        /// after the first window would leave earlier slots advanced while the engine retries the whole
        /// step, so any mid-step failure latches chunking off and declines.</summary>
        private bool ForwardBatchedDecodeChunks(int[] slots, int[] tokens, int[] positions, float[] flat, int vocab)
        {
            int max = TensorSharp.Cuda.Q4eCudaEngine.MaxBatchedDecodeRows;
            int n = slots.Length;
            if (n <= max)
                return _exec.ForwardBatchedDecode(slots, tokens, positions, flat);
            if (_batchedChunkingLatched) return false;
            int chunks = (n + max - 1) / max;
            int baseSize = n / chunks, rem = n % chunks, off = 0;
            for (int c = 0; c < chunks; c++)
            {
                int len = baseSize + (c < rem ? 1 : 0);
                var cs = new int[len]; var ct = new int[len]; var cp = new int[len];
                Array.Copy(slots, off, cs, 0, len);
                Array.Copy(tokens, off, ct, 0, len);
                Array.Copy(positions, off, cp, 0, len);
                var cf = new float[len * vocab];
                if (!_exec.ForwardBatchedDecode(cs, ct, cp, cf))
                {
                    if (c > 0)
                    {
                        _batchedChunkingLatched = true;
                        Console.Error.WriteLine($"[q4e-cuda batched-decode] chunk {c + 1}/{chunks} failed mid-step; chunked batching disabled");
                    }
                    return false;
                }
                Array.Copy(cf, 0, flat, (long)off * vocab, (long)len * vocab);
                off += len;
            }
            return true;
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

        // ---------------------------------------------------------------- IPrefixCacheModel

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new PrefixCacheCapabilities
        {
            Class = FamilyClass.N,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.DonateOnly,
            CanCaptureCopy = false,
            AdoptPrimaryOnDisplacement = true,
            PrimaryResident = true,
            MinRetainTokens = 32,
            // The slot decides (Q4eCudaEngine.CanRewind): from the checkpoint it took at its last
            // prompt boundary up to its head. A turn whose prompt re-renders the previous reply
            // matches the slot only partway into that reply; the slot goes back to the boundary and
            // replays its own tokens up to the match, instead of the whole prompt prefilling again.
            Truncation = TruncationKind.ModelDecides,
            TruncationGranularity = 1,
            RewindCapTokens = int.MaxValue,
            // Only a multi-token forward records the checkpoint: forwarding at least two prompt
            // tokens after any reuse leaves the next turn a boundary to rewind to (as DeepSeek V4.1).
            MinTailPrefillTokens = 2,
            Pages = PageSupport.None,
            ReuseAcrossMediaSpan = false,
            Persistable = false,
            // Each running request retains its slot for its conversation's next turn, and memory
            // takes them back oldest first when a new slot needs it.
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
            return false;
        }

        public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            footprint = default;
            lock (_sync)
            {
                if (_slotByRequest == null) return false;
                _retainedSlots ??= new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal);
                // A length below the head rewinds first, which the engine refuses unless it can.
                if (!SlotRetention.RetainSlot(_slotByRequest, _retainedSlots, requestId, payloadKey, length, canRewind: true,
                        ref _primarySlot, ref _activeSlotKey, new Store(_exec)))
                    return false;
                _cacheSeqLen = _exec.NPast;   // the selection may have moved to the primary
                footprint = Footprint(_retainedSlots[payloadKey]);
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
                footprint = Footprint(_retainedSlots[payloadKey]);
                return true;
            }
        }

        public bool TryMaterialize(in MaterializeRequest request)
        {
            if (request.Op != MaterializeOp.Donate) return false;   // slots are moved, never copied
            lock (_sync)
            {
                // A target below the payload is the executor's truncation at the first bind.
                if (!CanDonateLocked(request.PayloadKey, request.PayloadTokens, request.TargetTokens))
                    return false;
                _slotByRequest ??= new Dictionary<string, int>(StringComparer.Ordinal);
                return SlotRetention.DonateSlot(_slotByRequest, _retainedSlots, request.PayloadKey, request.TargetRequestId, out _);
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
                return CanDonateLocked(payloadKey, payloadTokens, targetTokens);
        }

        /// <summary>A retained slot holding <paramref name="payloadTokens"/> serves a request that keeps
        /// <paramref name="targetTokens"/> of them when the slot can rewind there (its checkpoint).</summary>
        private bool CanDonateLocked(string payloadKey, int payloadTokens, int targetTokens)
        {
            if (!SlotRetention.CanDonate(_retainedSlots, payloadKey, payloadTokens, targetTokens, canRewind: true))
                return false;
            return targetTokens == payloadTokens
                || _exec.CanRewind(_retainedSlots[payloadKey].Slot, payloadTokens, targetTokens);
        }

        /// <summary>The idle primary slot decides, as for a retained one (see <see cref="CanDonateLocked"/>).</summary>
        public bool CanRewindPrimary(int cachedTokens, int targetTokens)
        {
            lock (_sync)
                return _exec != null && _activeSlotKey == null && _primarySlot >= 0
                    && _exec.CanRewind(_primarySlot, cachedTokens, targetTokens);
        }

        public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        {
            lock (_sync)
                SlotRetention.ReleaseSlots(_retainedSlots, payloadKeys, new Store(_exec));
        }

        public PayloadFootprint MeasureEndState(string payloadKey)
        {
            lock (_sync)
                return payloadKey != null && _retainedSlots != null && _retainedSlots.TryGetValue(payloadKey, out var entry)
                    ? Footprint(entry) : default;
        }

        private PayloadFootprint Footprint(SlotRetention.RetainedSlot entry) => new(entry.Tokens, _maxContextLength, default, PositionDelta: 0);

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

        public long QuerySpareBytes(ResourceClass cls) => QueryPrefixCacheSpareBytes(cls);

        // ---------------------------------------------------------------- diagnostics

        public IReadOnlyCollection<string> RetainedPayloadKeys
        {
            get
            {
                lock (_sync)
                    return _retainedSlots == null ? Array.Empty<string>() : _retainedSlots.Keys.ToArray();
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
