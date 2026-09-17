// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Retained-prefix reuse for Qwen3.8-Flash-Next (the qwen4exp analogue of
// Qwen35Model.PerSeqCache's retained set and DeepSeek4Model.RetainedCache):
//
//   * a finished conversation's whole holder is RETAINED and re-keyed for the
//     turn that extends it exactly (nothing moves: the native state entries
//     keyed on the holder's host pointers, its captured graphs and the draft
//     head's private K/V all stay where they are);
//   * the state at the end of the prompt every chat shares is CHECKPOINTED as a
//     host-authoritative deep copy and CLONED into each new chat;
//   * both count against one byte budget (TS_Q4E_RETAINED_CACHE_MB, clamped by
//     the measured memory headroom), with the oldest retained conversation
//     evicted first and a decline when nothing more can go.
//
// Reuse is exact-prefix ONLY. The GatedDeltaNet recurrence and the PLE conv
// history have no per-position state to rewind to and the QSA indexer cache is
// only ever extended, so a holder whose tokens the new prompt does not
// reproduce to the last one is not a continuation of it; the model advertises
// no K/V truncation and refuses every partial match (IExactFusedCacheReuse).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp.Core;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel : IExactFusedCacheReuse
    {
        // Retained conversations and shared-prefix checkpoints, by key. Never
        // contains the active holder.
        private Dictionary<string, Qwen4ExpKvCacheHolder> _retainedFusedHolders;
        private long _retainedSerial;
        private readonly bool _retainedCacheEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_Q4E_RETAINED_CACHE"), "0", StringComparison.Ordinal);
        private readonly long _retainedCacheBudgetBytes = RetainedCacheBudgetFromEnvironment();
        private bool _retainedBudgetWarned;
        private bool _retainedLayoutWarned;

        /// <summary><c>TS_Q4E_RETAINED_CACHE_MB</c>: how many megabytes of retained
        /// state (finished conversations plus shared-prefix checkpoints, attention
        /// K/V + QSA keys + GDN/PLE state + the draft head's private K/V) this model
        /// keeps resident. Default 4096. An unparsable or zero value declines every
        /// retention, which the scheduler reports as a re-prefill.</summary>
        private static long RetainedCacheBudgetFromEnvironment()
        {
            string text = Environment.GetEnvironmentVariable("TS_Q4E_RETAINED_CACHE_MB");
            if (string.IsNullOrEmpty(text)) return 4096L * 1024 * 1024;
            return long.TryParse(text, out long mb) && mb >= 0 && mb <= long.MaxValue / (1024 * 1024)
                ? mb * 1024 * 1024 : 0;
        }

        /// <summary>The complete GGML token-span path, where every piece of per-sequence
        /// state (GDN conv+ssm, PLE conv, QSA raw keys, attention K/V) is device
        /// resident and keyed by the holder. The per-layer and op-by-op fallbacks
        /// keep parts of that state in shared scratch that no holder owns.</summary>
        private bool CompleteSpanPathAvailable =>
            IsGgmlBackend && _tokenGraphEnabled && !_tokenGraphUnsupported
            && _spanAttnEnabled && !_fusedGateUpExperts
            && _fusedFfnEnabled && !_fusedFfnUnsupported
            && _fusedGdnEnabled && !_fusedGdnUnsupported
            && _fusedAttnEnabled && !_fusedAttnUnsupported
            && _gdnMaxLayers < 0 && !_gdnVerify;

        /// <summary>The native GDN state entry stores the delta state as
        /// <c>head_v_dim * head_v_dim * n_v_heads</c> floats; the host seed tensor
        /// is <c>[n_v_heads, head_v_dim, head_k_dim]</c>. A copy reads the entry
        /// back through that layout, so the two must agree.</summary>
        private bool GdnStateLayoutMatchesNative
        {
            get
            {
                bool ok = _headKDim == _headVDim;
                if (!ok && !_retainedLayoutWarned)
                {
                    _retainedLayoutWarned = true;
                    Console.Error.WriteLine(
                        $"[q4e retained] disabled: GDN head_k_dim {_headKDim} != head_v_dim {_headVDim}, so the native delta-state layout cannot be copied exactly.");
                }
                return ok;
            }
        }

        /// <summary>Retention and rebinding move nothing, so they work wherever the
        /// complete span path does, including a layer split.</summary>
        public bool SupportsRetainedFusedCache =>
            _retainedCacheEnabled && _kCache != null && CompleteSpanPathAvailable && GdnStateLayoutMatchesNative;

        public bool SupportsExactFusedCacheReuse => SupportsRetainedFusedCache;

        /// <summary>A checkpoint downloads each layer's state from its owning
        /// device, including a layer split. Tensor-parallel state layouts are
        /// separate and remain unsupported.</summary>
        public bool SupportsPrefixCheckpoints =>
            SupportsRetainedFusedCache && !IsTensorParallel;

        // ---- exact-prefix admission checks (IExactFusedCacheReuse) ----

        public bool CanReuseLivePrefix(int cachedTokenCount, int targetTokenCount)
        {
            if (!SupportsExactFusedCacheReuse || cachedTokenCount < 0 || targetTokenCount != cachedTokenCount)
                return false;
            // The live cache the scheduler describes is the primary one; when a
            // fused holder is checked out, the primary's own health is what matters.
            bool failed = _activeFusedKey == null ? _specStateFailed : (_primaryHolder?.SpecStateFailed ?? false);
            return !failed;
        }

        public bool CanReuseRetainedPrefix(string retainedKey, int cachedTokenCount, int targetTokenCount)
        {
            if (!SupportsExactFusedCacheReuse || string.IsNullOrEmpty(retainedKey) || _retainedFusedHolders == null)
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedKey, out var holder))
                return false;
            return !holder.Retired && !holder.SpecStateFailed
                && targetTokenCount == cachedTokenCount && cachedTokenCount == holder.CacheSeqLen;
        }

        // ---- retain / rebind / discard ----

        /// <summary>Move a cleanly finished request's complete holder out of the
        /// active set. Its native state entries, captured graphs and draft-head
        /// state stay registered under the holder's own pointers, so the turn that
        /// continues it pays nothing. Declined (the holder is then released as
        /// usual) for a failed conversation, an empty one, a repeated key, or when
        /// the retention budget cannot take it even after evicting older
        /// conversations.</summary>
        public bool RetainSequenceCache(string requestId) => RetainSequenceCacheAs(requestId, requestId);

        /// <summary>The key-parameterised form of <see cref="RetainSequenceCache"/>: the finished
        /// holder of <paramref name="requestId"/> is retained under <paramref name="key"/>
        /// (the prefix cache's tree-minted payload key, or the request id itself).</summary>
        public bool RetainSequenceCacheAs(string requestId, string key)
        {
            if (!SupportsRetainedFusedCache || _fusedHolders == null || string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key))
                return false;
            if (!_fusedHolders.TryGetValue(requestId, out var holder))
                return false;
            if (_retainedFusedHolders != null && _retainedFusedHolders.ContainsKey(key))
                return false;
            bool active = string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal);
            // The dictionary entry is stale for the active holder: growth or a
            // re-seed may have replaced fields since it was loaded.
            if (active) holder = SnapshotActiveCache();
            if (holder.SpecStateFailed || holder.Retired || holder.CacheSeqLen <= 0)
                return false;

            _retainedFusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));
            long bytes = RetainedHolderBytes(holder);
            if (!EnsureRetentionBudget(bytes, "retaining conversation " + requestId))
                return false;

            if (active)
            {
                _activeFusedKey = null;
                if (_primaryHolder != null)
                {
                    LoadCacheHolder(_primaryHolder);
                    _primaryHolder = null;
                }
            }
            holder.RetainedBytes = bytes;
            holder.RetainedSerial = ++_retainedSerial;
            _retainedFusedHolders.Add(key, holder);
            _fusedHolders.Remove(requestId);
            return true;
        }

        /// <summary>Re-key a retained conversation for the request that extends it
        /// exactly. The next <see cref="BindSequenceCache"/> finds it non-fresh and
        /// forwards only the new suffix. The scheduler only asks for a holder that
        /// <see cref="CanReuseRetainedPrefix"/> approved; a checkpoint is cloned,
        /// never moved.</summary>
        public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
        {
            if (_retainedFusedHolders == null || string.IsNullOrEmpty(retainedRequestId) || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedRequestId, out var holder))
                return false;
            // A checkpoint serves every new chat and is cloned, never moved, while this model
            // keeps its own retention. Once the prefix cache owns it, the tree decides what may
            // be donated (a scoped capture, never a public one).
            if (holder.Retired || (holder.IsCheckpoint && _prefixCacheSink == null) || holder.SpecStateFailed)
                return false;
            _fusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId) || string.Equals(_activeFusedKey, newRequestId, StringComparison.Ordinal))
                return false;
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            _fusedHolders.Add(newRequestId, holder);
            _retainedFusedHolders.Remove(retainedRequestId);
            holder.RetainedBytes = 0;
            return true;
        }

        /// <summary>Free a retained holder (LRU eviction, budget eviction, a
        /// declined clone, shutdown): its native state entries and every cached
        /// graph that binds them, its tensors, and the draft head's private state.</summary>
        public void DiscardRetainedCache(string requestId)
        {
            if (_retainedFusedHolders == null || string.IsNullOrEmpty(requestId))
                return;
            if (!_retainedFusedHolders.TryGetValue(requestId, out var holder))
                return;
            holder.Retired = true;
            _retainedFusedHolders.Remove(requestId);
            DisposeHolder(holder);
            ReleaseSlotBase(holder.SlotBase);
        }

        // ---- shared-prefix checkpoints ----

        /// <summary>Deep-copy the ACTIVE cache (primary or checked-out holder) into the
        /// retained set as a checkpoint. Every device-resident piece is downloaded
        /// first, so the copy is host-authoritative and its first forward seeds fresh
        /// native entries from it.</summary>
        public bool TryCheckpointActiveCache(string key)
        {
            if (!SupportsPrefixCheckpoints || string.IsNullOrEmpty(key) || _isRecurrent == null)
                return false;
            if (_specStateFailed || _cacheSeqLen <= 0)
                return false;
            _retainedFusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);
            if (_retainedFusedHolders.ContainsKey(key) || (_fusedHolders != null && _fusedHolders.ContainsKey(key)))
                return false;
            _retainedFusedHolders.EnsureCapacity(checked(_retainedFusedHolders.Count + 1));

            // Attention K/V and the QSA raw keys are read from the host mirrors: bring
            // the device copies back first (EnsureCacheCapacity's own precondition).
            EnsureKvCacheHostSynchronized();
            var source = SnapshotActiveCache();
            // The copy is sized to what the source holds, never more than the source.
            if (!EnsureRetentionBudget(RetainedHolderBytes(source), "checkpointing " + key))
                return false;
            var copy = DeepCopyHolder(source);
            bool published = false;
            try
            {
                copy.IsCheckpoint = true;
                copy.RetainedBytes = RetainedHolderBytes(copy);
                copy.RetainedSerial = ++_retainedSerial;
                _retainedFusedHolders.Add(key, copy);
                published = true;
            }
            finally
            {
                if (!published)
                {
                    DisposeHolder(copy);
                    ReleaseSlotBase(copy.SlotBase);
                }
            }
            return true;
        }

        /// <summary>Deep-copy the retained holder <paramref name="retainedKey"/> into a
        /// fresh active holder for <paramref name="newRequestId"/>; the retained one is
        /// untouched, so the next chat can clone it too. A retained conversation can be
        /// cloned as well: its device entries are downloaded like a checkpoint's were.</summary>
        public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
        {
            if (!SupportsPrefixCheckpoints || _retainedFusedHolders == null
                || string.IsNullOrEmpty(retainedKey) || string.IsNullOrEmpty(newRequestId))
                return false;
            if (!_retainedFusedHolders.TryGetValue(retainedKey, out var source))
                return false;
            if (source.Retired || source.SpecStateFailed)
                return false;
            _fusedHolders ??= new Dictionary<string, Qwen4ExpKvCacheHolder>(StringComparer.Ordinal);
            if (_fusedHolders.ContainsKey(newRequestId) || string.Equals(_activeFusedKey, newRequestId, StringComparison.Ordinal))
                return false;
            _fusedHolders.EnsureCapacity(checked(_fusedHolders.Count + 1));
            var copy = DeepCopyHolder(source);
            bool published = false;
            try
            {
                _fusedHolders.Add(newRequestId, copy);
                published = true;
            }
            finally
            {
                if (!published)
                {
                    DisposeHolder(copy);
                    ReleaseSlotBase(copy.SlotBase);
                }
            }
            return true;
        }

        // ---- budget ----

        /// <summary>Whether a holder of <paramref name="holderBytes"/> may join
        /// <paramref name="retainedBytes"/> of already retained state under
        /// <paramref name="budgetBytes"/>, further clamped to half of the measured
        /// memory headroom when one is known (the same half-spare rule Qwen 3.5
        /// applies to its idle holders). A zero budget declines everything.</summary>
        internal static bool CanRetainWithinBudget(long holderBytes, long retainedBytes, long budgetBytes, long? spareBytes)
        {
            if (budgetBytes <= 0 || holderBytes < 0 || retainedBytes < 0) return false;
            long limit = budgetBytes;
            if (spareBytes.HasValue) limit = Math.Min(limit, Math.Max(0, spareBytes.Value) / 2);
            return holderBytes <= limit && retainedBytes <= limit - holderBytes;
        }

        /// <summary>Make room for <paramref name="bytes"/> by evicting the oldest
        /// retained conversations (never a checkpoint: those are bounded by the
        /// scheduler's own budget and each serves every new chat). False when the
        /// budget still cannot take it, reported once.</summary>
        private bool EnsureRetentionBudget(long bytes, string what)
        {
            long? spare = GetCacheMemorySpareBytes();
            if (_prefixCacheSink != null)
            {
                // Refuse-and-report (DEC-23): the prefix cache owns eviction, so a holder that does
                // not fit beside what is retained is refused; nothing retained is evicted to fit it.
                if (CanRetainWithinBudget(bytes, RetainedBytesTotal(), _retainedCacheBudgetBytes, spare))
                    return true;
                if (!_retainedBudgetWarned)
                {
                    _retainedBudgetWarned = true;
                    Console.Error.WriteLine(
                        $"[q4e retained] refused {what}: {bytes / 1048576.0:F1} MB does not fit the retention budget " +
                        $"(TS_Q4E_RETAINED_CACHE_MB={_retainedCacheBudgetBytes / 1048576} MB, retained {RetainedBytesTotal() / 1048576.0:F1} MB" +
                        (spare.HasValue ? $", headroom {spare.Value / 1048576.0:F0} MB" : "") + ") and the prefix cache owns eviction. Reported once.");
                }
                return false;
            }
            // Evict nothing for a holder that could not fit beside the checkpoints
            // even with every conversation gone (a zero budget, a holder larger
            // than the budget): a decline must not cost the conversations kept.
            bool feasible = CanRetainWithinBudget(bytes, RetainedBytesTotal(checkpointsOnly: true), _retainedCacheBudgetBytes, spare);
            while (!feasible || !CanRetainWithinBudget(bytes, RetainedBytesTotal(), _retainedCacheBudgetBytes, spare))
            {
                string victim = feasible ? OldestRetainedConversation() : null;
                if (victim == null)
                {
                    if (!_retainedBudgetWarned)
                    {
                        _retainedBudgetWarned = true;
                        Console.Error.WriteLine(
                            $"[q4e retained] declined {what}: {bytes / 1048576.0:F1} MB does not fit the retention budget " +
                            $"(TS_Q4E_RETAINED_CACHE_MB={_retainedCacheBudgetBytes / 1048576} MB, retained {RetainedBytesTotal() / 1048576.0:F1} MB" +
                            (spare.HasValue ? $", headroom {spare.Value / 1048576.0:F0} MB" : "") + "); that request re-prefills. Reported once.");
                    }
                    return false;
                }
                Console.Error.WriteLine($"[q4e retained] evicting {victim} ({_retainedFusedHolders[victim].RetainedBytes / 1048576.0:F1} MB) for {what}");
                DiscardRetainedCache(victim);
            }
            return true;
        }

        private string OldestRetainedConversation()
        {
            string oldest = null;
            long serial = long.MaxValue;
            if (_retainedFusedHolders == null) return null;
            foreach (var kv in _retainedFusedHolders)
            {
                if (kv.Value.IsCheckpoint || kv.Value.RetainedSerial >= serial) continue;
                oldest = kv.Key;
                serial = kv.Value.RetainedSerial;
            }
            return oldest;
        }

        private long RetainedBytesTotal(bool checkpointsOnly = false)
        {
            long total = 0;
            if (_retainedFusedHolders != null)
                foreach (var holder in _retainedFusedHolders.Values)
                    if (!checkpointsOnly || holder.IsCheckpoint)
                        total = checked(total + holder.RetainedBytes);
            return total;
        }

        /// <summary>Bytes a holder keeps resident while retained: its attention K/V,
        /// QSA raw-key caches and position history, the GDN conv + delta state and
        /// host ring, the PLE conv history, and the draft head's private K/V.</summary>
        private long RetainedHolderBytes(Qwen4ExpKvCacheHolder holder)
        {
            long bytes = 0;
            var storages = new HashSet<Storage>();
            foreach (Tensor[] set in new[] { holder.K, holder.V, holder.IdxK, holder.GdnConvStateT, holder.GdnStateT })
                if (set != null)
                    foreach (Tensor t in set)
                        if (t != null && storages.Add(t.Storage))
                            bytes = checked(bytes + t.Storage.ByteLength);
            if (holder.GdnConvState != null)
                foreach (float[] ring in holder.GdnConvState)
                    if (ring != null) bytes = checked(bytes + (long)ring.Length * sizeof(float));
            if (holder.PleConvState != null) bytes = checked(bytes + (long)holder.PleConvState.Length * sizeof(float));
            if (holder.QsaPositions != null) bytes = checked(bytes + (long)holder.QsaPositions.Length * sizeof(int));
            bytes = checked(bytes + MtpStateBytes((object)holder.GdnConvStateT ?? holder.K));
            return bytes;
        }

        /// <summary>Headroom for retained holders: both the host allocation and a GPU
        /// mirror have to fit (see <c>Qwen35Model.GetCacheMemorySpareBytes</c>, which
        /// this mirrors). Unknown budgets leave the configured byte budget alone.</summary>
        protected virtual long? GetCacheMemorySpareBytes()
        {
            long? spare = GpuMemoryBudget.TryGetReservationSpareBytes(_backend, out long deviceSpare)
                ? deviceSpare : null;
            try
            {
                long available = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
                if (available > 0)
                {
                    using var process = Process.GetCurrentProcess();
                    long reserve = Math.Max(GpuMemoryBudget.MinHeadroomBytes, available / 16);
                    long hostSpare = Math.Max(0, available - Math.Min(available, process.WorkingSet64) - reserve);
                    spare = spare.HasValue ? Math.Min(spare.Value, hostSpare) : hostSpare;
                }
            }
            catch (InvalidOperationException) { }
            catch (System.ComponentModel.Win32Exception) { }
            catch (NotSupportedException) { }
            return spare;
        }

        /// <summary>Under memory pressure, free every retained conversation. The
        /// scheduler's metadata for them then fails <see cref="CanReuseRetainedPrefix"/>
        /// and those turns re-prefill. Checkpoints stay: each serves every new chat.</summary>
        public override void TrimIdleMemory()
        {
            if (_retainedFusedHolders != null)
            {
                var victims = new List<string>();
                foreach (var kv in _retainedFusedHolders)
                    if (!kv.Value.IsCheckpoint) victims.Add(kv.Key);
                foreach (string key in victims) DiscardRetainedCache(key);
                // The prefix cache learns of every holder freed behind its back (DEC-23).
                if (_prefixCacheSink != null)
                    foreach (string key in victims) _prefixCacheSink.OnPayloadInvalidated(key, InvalidationReason.TrimmedByModel);
                if (victims.Count > 0)
                    Console.WriteLine($"[memory] Qwen4Exp: freed {victims.Count} retained conversation holder(s)");
            }
            base.TrimIdleMemory();
        }

        // ---- test-facing accessors ----

        internal int RetainedCacheCount => _retainedFusedHolders?.Count ?? 0;
        internal bool IsRetainedCheckpoint(string key)
            => _retainedFusedHolders != null && _retainedFusedHolders.TryGetValue(key, out var h) && h.IsCheckpoint;
        internal long RetainedCacheBytes(string key)
            => _retainedFusedHolders != null && _retainedFusedHolders.TryGetValue(key, out var h) ? h.RetainedBytes : -1;

        // ---- the deep copy ----

        private static int CopyCapacityFor(int rows, int maxContext)
        {
            long cap = Math.Max(256, ((long)rows + 255) / 256 * 256);
            return (int)Math.Min(Math.Max(1, maxContext), cap);
        }

        /// <summary>Bring a NON-active holder's attention K/V and QSA raw keys back to
        /// their host mirrors, as <see cref="EnsureKvCacheHostSynchronized"/> does for
        /// the active one. Retained holders are never active.</summary>
        private void SyncHolderKvToHost(Qwen4ExpKvCacheHolder holder)
        {
            if (!holder.KvHostStale || !IsGgmlBackend || holder.K == null) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (holder.K[l] != null) SyncTensorHostCache(holder.K[l]);
                if (holder.V[l] != null) SyncTensorHostCache(holder.V[l]);
                if (holder.IdxK != null && holder.IdxK[l] != null)
                {
                    var cache = holder.IdxK[l];
                    cache.Storage.EnsureHostReadable();
                    IntPtr pointer = TensorComputePrimitives.GetStoragePointer(cache);
                    if (!TryExportSeqState(pointer, pointer, cache.Storage.ByteLength, DeviceForLayer(l), holder.DeviceStateSeeded, "QSA raw keys"))
                        throw new InvalidOperationException("qwen4exp retained: could not export the indexer cache of a retained holder.");
                    InvalidateTensorDeviceCache(cache);
                }
            }
            holder.KvHostStale = false;
        }

        /// <summary>Copy one native per-sequence state entry (keyed by its host seed
        /// pointer) to <paramref name="destination"/>. False when the entry
        /// was never seeded - then the host seed IS what the next forward
        /// uploads, and the caller copies that instead. A seeded holder's host
        /// bytes are stale; failure to export must abort the clone.</summary>
        private bool TryExportSeqState(IntPtr key, IntPtr destination, long bytes, int device, bool seeded, string what)
        {
            if (!seeded) return false;
            if (GgmlBasicOps.Qwen4ExpCopyQsaCache(key, destination, bytes, device))
                return true;
            throw new InvalidOperationException(
                $"qwen4exp retained: could not export authoritative {what}: {GgmlEmbeddingNative.LastError("(no native error)")}");
        }

        private static unsafe void ZeroHostStorage(Tensor t)
        {
            t.Storage.EnsureHostReadable();
            NativeMemory.Clear(TensorComputePrimitives.GetStoragePointer(t).ToPointer(), checked((nuint)t.Storage.ByteLength));
        }

        /// <summary>An independent, host-authoritative copy of <paramref name="source"/>:
        /// fresh tensors sized to the rows it holds, the attention K/V rows and QSA
        /// raw keys copied from the (synchronized) host mirrors, the GDN conv + delta
        /// state and the PLE conv history downloaded from the device entries the
        /// source seeded (or its host seeds when it never forwarded), the QSA
        /// position history, the PLE n-gram window and the multimodal position gap
        /// copied, and the draft head's private state cloned. Its first forward
        /// seeds its own native entries and builds its own graphs.</summary>
        private unsafe Qwen4ExpKvCacheHolder DeepCopyHolder(Qwen4ExpKvCacheHolder source)
        {
            if (source.K == null) throw new InvalidOperationException("qwen4exp retained: the source holder has no caches.");
            bool sourceIsActive = ReferenceEquals(source.K, _kCache);
            if (sourceIsActive) EnsureKvCacheHostSynchronized();
            else SyncHolderKvToHost(source);

            int rows = Math.Max(0, Math.Min(source.CacheSeqLen, source.KvCapacity));
            int cap = Math.Min(source.KvCapacity, CopyCapacityFor(rows, _maxContextLength));
            var dst = AllocateHolder(cap);
            bool complete = false;
            try
            {
                int nLayer = Config.NumLayers;
                DType kvDtype = _kvCacheDtype.ToDType();
                int kvElem = kvDtype == DType.Float16 ? 2 : 4;
                for (int l = 0; l < nLayer; l++)
                {
                    if (_isRecurrent[l])
                    {
                        CopyGdnLayerState(source, dst, l);
                        continue;
                    }
                    // Attention K/V: the copy's rows past `rows` must be finite (the
                    // fused window reads them masked), and a pooled allocation
                    // guarantees nothing, so zero the whole storage first.
                    ZeroHostStorage(dst.K[l]);
                    ZeroHostStorage(dst.V[l]);
                    CopyCacheRows(source.K[l], dst.K[l], rows);
                    CopyCacheRows(source.V[l], dst.V[l], rows);
                    InvalidateTensorDeviceCache(dst.K[l]);
                    InvalidateTensorDeviceCache(dst.V[l]);
                    if (source.IdxK != null && source.IdxK[l] != null && dst.IdxK[l] != null)
                    {
                        // One-head raw-key cache with a contiguous live prefix; the
                        // destination was zeroed by InitializeQsaCache.
                        long liveBytes = checked((long)rows * _indexerHeadDim * kvElem);
                        if (liveBytes > dst.IdxK[l].Storage.ByteLength || liveBytes > source.IdxK[l].Storage.ByteLength)
                            throw new InvalidOperationException("qwen4exp retained: QSA cache rows exceed the copy's capacity.");
                        source.IdxK[l].Storage.EnsureHostReadable();
                        dst.IdxK[l].Storage.EnsureHostReadable();
                        Buffer.MemoryCopy(TensorComputePrimitives.GetStoragePointer(source.IdxK[l]).ToPointer(),
                            TensorComputePrimitives.GetStoragePointer(dst.IdxK[l]).ToPointer(), dst.IdxK[l].Storage.ByteLength, liveBytes);
                        InvalidateTensorDeviceCache(dst.IdxK[l]);
                    }
                }

                // PLE conv history (device entry keyed by the pinned host array).
                if (dst.PleConvState != null && source.PleConvState != null)
                {
                    if (dst.PleConvState.Length != source.PleConvState.Length)
                        throw new InvalidOperationException("qwen4exp retained: PLE conv history sizes differ.");
                    IntPtr key = Marshal.UnsafeAddrOfPinnedArrayElement(source.PleConvState, 0);
                    IntPtr to = Marshal.UnsafeAddrOfPinnedArrayElement(dst.PleConvState, 0);
                    if (!TryExportSeqState(key, to, (long)source.PleConvState.Length * sizeof(float),
                            DeviceForLayer(Math.Max(0, _pleLayerIndex)), source.DeviceStateSeeded, "PLE conv history"))
                        Array.Copy(source.PleConvState, dst.PleConvState, source.PleConvState.Length);
                }
                dst.PleHistory = source.PleHistory != null ? new List<int>(source.PleHistory) : new List<int>();
                dst.PleNextPos = source.PleNextPos;

                // QSA position history: (T,H,W) per cached cell, pinned like the source.
                if (source.QsaPositions != null && source.QsaPositionCount > 0)
                {
                    int count = source.QsaPositionCount;
                    if ((long)3 * count > source.QsaPositions.LongLength)
                        throw new InvalidOperationException("qwen4exp retained: QSA position history is shorter than its count.");
                    int length = checked(3 * Math.Max(cap, count));
                    dst.QsaPositions = GC.AllocateArray<int>(length, pinned: true);
                    Array.Copy(source.QsaPositions, dst.QsaPositions, checked(3 * count));
                    dst.QsaPositionCount = count;
                }

                dst.CacheSeqLen = source.CacheSeqLen;
                dst.MropeCacheGap = source.MropeCacheGap;
                dst.KvHostStale = false;
                dst.SpecStateFailed = false;
                dst.DeviceStateSeeded = false;

                CloneMtpState((object)source.GdnConvStateT ?? source.K, dst.GdnConvStateT);
                complete = true;
                return dst;
            }
            finally
            {
                if (!complete)
                {
                    DisposeHolder(dst);
                    ReleaseSlotBase(dst.SlotBase);
                }
            }
        }

        /// <summary>One recurrent layer: the native entry keyed by the source's conv
        /// seed pointer holds <c>[conv | pad to 256 | delta]</c>; split it into the
        /// copy's two host seed tensors. A never-forwarded source is copied from its
        /// host seeds. The op-by-op host ring travels too.</summary>
        private unsafe void CopyGdnLayerState(Qwen4ExpKvCacheHolder source, Qwen4ExpKvCacheHolder dst, int l)
        {
            Tensor srcConv = source.GdnConvStateT?[l], srcSsm = source.GdnStateT?[l];
            Tensor dstConv = dst.GdnConvStateT[l], dstSsm = dst.GdnStateT[l];
            long convBytes = dstConv.Storage.ByteLength;
            long ssmBytes = dstSsm.Storage.ByteLength;
            long ssmOff = (convBytes + 255) & ~255L;
            bool exported = false;
            if (srcConv != null && srcSsm != null)
            {
                if (srcConv.Storage.ByteLength != convBytes || srcSsm.Storage.ByteLength != ssmBytes)
                    throw new InvalidOperationException("qwen4exp retained: GDN state tensors differ in size.");
                if ((long)_headVDim * _headVDim * _numVHeads * sizeof(float) != ssmBytes)
                    throw new InvalidOperationException("qwen4exp retained: the GDN delta-state layout does not match the native entry.");
                dstConv.Storage.EnsureHostReadable();
                dstSsm.Storage.EnsureHostReadable();
                long entryBytes = checked(ssmOff + ssmBytes);
                byte[] staging = GC.AllocateUninitializedArray<byte>(checked((int)entryBytes), pinned: true);
                IntPtr stagingPtr = Marshal.UnsafeAddrOfPinnedArrayElement(staging, 0);
                if (TryExportSeqState((IntPtr)GetFloatPtr(srcConv), stagingPtr, entryBytes, DeviceForLayer(l),
                        source.DeviceStateSeeded, $"GDN state of layer {l}"))
                {
                    fixed (byte* s = staging)
                    {
                        Buffer.MemoryCopy(s, GetFloatPtr(dstConv), convBytes, convBytes);
                        Buffer.MemoryCopy(s + ssmOff, GetFloatPtr(dstSsm), ssmBytes, ssmBytes);
                    }
                    exported = true;
                }
                if (!exported)
                {
                    srcConv.Storage.EnsureHostReadable();
                    srcSsm.Storage.EnsureHostReadable();
                    Buffer.MemoryCopy(GetFloatPtr(srcConv), GetFloatPtr(dstConv), convBytes, convBytes);
                    Buffer.MemoryCopy(GetFloatPtr(srcSsm), GetFloatPtr(dstSsm), ssmBytes, ssmBytes);
                }
                InvalidateTensorDeviceCache(dstConv);
                InvalidateTensorDeviceCache(dstSsm);
            }
            if (source.GdnConvState?[l] != null && dst.GdnConvState[l] != null)
                Array.Copy(source.GdnConvState[l], dst.GdnConvState[l], Math.Min(source.GdnConvState[l].Length, dst.GdnConvState[l].Length));
            if (source.GdnConvWriteIdx != null && dst.GdnConvWriteIdx != null)
                dst.GdnConvWriteIdx[l] = source.GdnConvWriteIdx[l];
        }
    }
}
