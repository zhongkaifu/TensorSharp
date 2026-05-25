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
using System.Threading;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Cross-session paged KV-cache orchestrator. Splits the canonical token
    /// sequence into fixed-size blocks, hashes each block chained with its parent,
    /// and persists the corresponding K/V byte slices in an in-memory tier with
    /// an optional SSD spill tier. Prefill paths consult the manager to recover
    /// state that was produced for a different session whose prompt happens to
    /// share a prefix - turning the second user's prefill cost from O(n) into
    /// O(block_size * suffix).
    ///
    /// The manager is intentionally a thin coordinator: byte-level serialization
    /// lives on the model (which understands its own per-layer K/V layout) and
    /// storage / eviction lives on the underlying <see cref="PagedKvBlockStore"/>
    /// and optional <see cref="SsdKvBlockTier"/>. The class is safe to access from
    /// multiple inference threads provided the model itself is only used by one
    /// thread at a time (which is already the invariant inside ModelService).
    /// </summary>
    public sealed class PagedKvCacheManager : IDisposable
    {
        private readonly PagedKvCacheConfig _config;
        private readonly ILogger _logger;
        private readonly PagedKvBlockStore _ramTier;
        private readonly SsdKvBlockTier _ssdTier;
        private readonly string _fingerprint;
        private readonly IKvBlockCodec _codec;
        private long _hitTokens;
        private long _missTokens;
        private long _capturedBlocks;

        /// <summary>Construct a disabled manager (all operations are no-ops).</summary>
        public static PagedKvCacheManager Disabled() => new(new PagedKvCacheConfig { Enabled = false }, fingerprint: string.Empty, logger: null);

        public PagedKvCacheManager(PagedKvCacheConfig config, string fingerprint, ILogger logger)
            : this(config, fingerprint, logger, codec: null)
        {
        }

        /// <summary>
        /// Construct a paged manager that re-encodes captured blocks through
        /// <paramref name="codec"/> before placing them into the in-memory tier
        /// (and consequently spilling encoded bytes to SSD). The decode happens
        /// transparently on <see cref="TryRestorePrefix"/> so the model only
        /// ever sees raw bytes in its own KV layout. Pass <c>null</c> to use
        /// the historical passthrough behaviour.
        /// </summary>
        public PagedKvCacheManager(PagedKvCacheConfig config, string fingerprint, ILogger logger, IKvBlockCodec codec)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _fingerprint = fingerprint ?? string.Empty;
            _logger = logger ?? NullLogger.Instance;
            _codec = codec;
            if (!config.Enabled)
                return;

            if (!string.IsNullOrWhiteSpace(config.SsdDirectory))
            {
                try
                {
                    _ssdTier = new SsdKvBlockTier(config.SsdDirectory, config.MaxSsdBytes, _fingerprint, _logger);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(LogEventIds.PagedKvCacheTierInit,
                        ex, "Failed to initialize SSD KV cache tier at {Dir}; SSD spill disabled.", config.SsdDirectory);
                    _ssdTier = null;
                }
            }

            Action<KvBlockHash, byte[]> spillCallback = _ssdTier != null
                ? (h, b) => _ssdTier.EnqueueWrite(h, b)
                : null;
            _ramTier = new PagedKvBlockStore(config.MaxRamBytes, spillCallback);
        }

        public bool IsEnabled => _config.Enabled && _ramTier != null;
        public int BlockSize => _config.BlockSize;
        public string Fingerprint => _fingerprint;
        public IKvBlockCodec Codec => _codec;

        public PagedKvCacheStats GetStats() => new(
            ramBytes: _ramTier?.ResidentBytes ?? 0,
            ramBlocks: _ramTier?.Count ?? 0,
            ramMaxBytes: _ramTier?.MaxBytes ?? 0,
            ssdBytes: _ssdTier?.ResidentBytes ?? 0,
            ssdBlocks: _ssdTier?.Count ?? 0,
            hitTokens: Interlocked.Read(ref _hitTokens),
            missTokens: Interlocked.Read(ref _missTokens),
            capturedBlocks: Interlocked.Read(ref _capturedBlocks));

        /// <summary>
        /// Reset all in-memory and on-disk state. Used when the underlying model is
        /// unloaded or its KV-state fingerprint changes.
        /// </summary>
        public void Clear()
        {
            _ramTier?.Clear();
            _ssdTier?.Clear();
            Interlocked.Exchange(ref _hitTokens, 0);
            Interlocked.Exchange(ref _missTokens, 0);
            Interlocked.Exchange(ref _capturedBlocks, 0);
        }

        public void Dispose() => _ssdTier?.Dispose();

        /// <summary>
        /// Return how many leading blocks of <paramref name="inputTokens"/> are
        /// currently retrievable from the in-memory tier (the SSD tier is checked
        /// lazily inside <see cref="TryRestorePrefix"/>). Used by the session
        /// manager to decide whether paging beats the in-session reuse path.
        /// </summary>
        public int CountAvailableBlocks(IModelArchitecture model, IReadOnlyList<int> inputTokens)
        {
            if (!IsEnabled || model == null || inputTokens == null || inputTokens.Count == 0)
                return 0;
            if (!model.SupportsKVStateSnapshot)
                return 0;
            if (!string.Equals(model.KVStateFingerprint, _fingerprint, StringComparison.Ordinal))
                return 0;

            int blockSize = _config.BlockSize;
            if (inputTokens.Count < blockSize)
                return 0;

            var hashes = KvBlockHasher.ComputeBlockHashes(inputTokens, blockSize, _fingerprint);
            int available = 0;
            for (int b = 0; b < hashes.Count; b++)
            {
                if (!_ramTier.Contains(hashes[b]))
                    break;
                available++;
            }
            return available;
        }

        /// <summary>
        /// Try to inject as long a block-aligned prefix of <paramref name="inputTokens"/>
        /// as possible into the model's KV cache. Returns the number of tokens
        /// restored (always a multiple of <see cref="BlockSize"/>). The model must
        /// be in a reset state (no existing KV content) before this call. After
        /// restoration the caller must update its <see cref="KVCache"/> token list
        /// to reflect the restored prefix.
        /// </summary>
        public int TryRestorePrefix(IModelArchitecture model, IReadOnlyList<int> inputTokens)
        {
            if (!IsEnabled || model == null || inputTokens == null || inputTokens.Count == 0)
                return 0;
            if (!model.SupportsKVStateSnapshot)
                return 0;
            if (!string.Equals(model.KVStateFingerprint, _fingerprint, StringComparison.Ordinal))
                return 0;

            int blockSize = _config.BlockSize;
            int fullBlocks = inputTokens.Count / blockSize;
            if (fullBlocks == 0)
                return 0;

            // Reserve one trailing block for the actual forward call so the model
            // produces fresh logits. Without this guarantee the caller would need
            // to also recover the logits buffer, which is not currently snapshotted.
            int restoreBudgetBlocks = fullBlocks;
            if (inputTokens.Count == fullBlocks * blockSize && restoreBudgetBlocks > 0)
                restoreBudgetBlocks--;
            if (restoreBudgetBlocks == 0)
                return 0;

            var hashes = KvBlockHasher.ComputeBlockHashes(inputTokens, blockSize, _fingerprint);
            int restored = 0;
            long expectedRaw = model.ComputeKVBlockByteSize(blockSize);
            if (expectedRaw <= 0)
                return 0;

            // Reuse a single decode scratch buffer across the loop; freeing
            // and reallocating per-block would thrash the LOH for typical
            // block sizes (>= 85 KB).
            byte[] decodeScratch = _codec != null ? new byte[expectedRaw] : null;

            for (int b = 0; b < restoreBudgetBlocks && b < hashes.Count; b++)
            {
                if (!TryFetch(hashes[b], out byte[] payload))
                    break;

                ReadOnlySpan<byte> rawSpan;
                if (_codec != null)
                {
                    if (!_codec.TryDecode(payload, decodeScratch))
                    {
                        _logger.LogWarning(LogEventIds.PagedKvCacheRestoreSkip,
                            "Skipping paged restore at block {BlockIndex}: codec {Codec} rejected payload of {ActualBytes} bytes",
                            b, _codec.Name, payload.LongLength);
                        break;
                    }
                    rawSpan = decodeScratch;
                }
                else
                {
                    if (payload.LongLength != expectedRaw)
                    {
                        _logger.LogWarning(LogEventIds.PagedKvCacheRestoreSkip,
                            "Skipping paged restore at block {BlockIndex}: cached payload size {ActualBytes} does not match expected {ExpectedBytes}",
                            b, payload.LongLength, expectedRaw);
                        break;
                    }
                    rawSpan = payload;
                }

                if (!model.TryInjectKVBlock(restored, blockSize, rawSpan))
                {
                    _logger.LogWarning(LogEventIds.PagedKvCacheRestoreSkip,
                        "Model rejected paged restore at block {BlockIndex}; aborting restore.", b);
                    break;
                }
                restored += blockSize;
            }

            if (restored > 0)
            {
                Interlocked.Add(ref _hitTokens, restored);
                Interlocked.Add(ref _missTokens, inputTokens.Count - restored);
                _logger.LogDebug(LogEventIds.PagedKvCacheRestore,
                    "kv.paged restored {Restored}/{Total} tokens ({Blocks} blocks)",
                    restored, inputTokens.Count, restored / blockSize);
            }
            else
            {
                Interlocked.Add(ref _missTokens, inputTokens.Count);
            }

            return restored;
        }

        /// <summary>
        /// Snapshot the model's KV state for the first <paramref name="upToTokens"/>
        /// tokens (rounded down to a block boundary) into the in-memory tier, using
        /// the hashes derived from <paramref name="tokens"/>. Idempotent: blocks
        /// whose hash already exists in the tier are not re-extracted, so calling
        /// this after every prefill is cheap.
        /// </summary>
        public void Capture(IModelArchitecture model, IReadOnlyList<int> tokens, int upToTokens)
        {
            if (!IsEnabled || model == null || tokens == null || upToTokens <= 0)
                return;
            if (!model.SupportsKVStateSnapshot)
                return;
            if (!string.Equals(model.KVStateFingerprint, _fingerprint, StringComparison.Ordinal))
                return;

            int blockSize = _config.BlockSize;
            int captureBlocks = Math.Min(upToTokens, tokens.Count) / blockSize;
            if (captureBlocks == 0)
                return;

            var hashes = KvBlockHasher.ComputeBlockHashes(tokens, blockSize, _fingerprint);
            long blockBytes = model.ComputeKVBlockByteSize(blockSize);
            if (blockBytes <= 0)
                return;

            int newlyCaptured = 0;
            for (int b = 0; b < captureBlocks && b < hashes.Count; b++)
            {
                if (_ramTier.Contains(hashes[b]))
                    continue;

                byte[] payload = new byte[blockBytes];
                if (!model.TryExtractKVBlock(b * blockSize, blockSize, payload))
                {
                    _logger.LogDebug(LogEventIds.PagedKvCacheCaptureSkip,
                        "Model declined to extract KV block {BlockIndex}; stopping capture.", b);
                    break;
                }

                // Run the captured raw bytes through the codec when one is
                // present. The store accounts for residency in encoded bytes
                // so a 4-bit codec quadruples the effective capacity of the
                // RAM tier without any other code path changing.
                byte[] storedPayload = _codec != null ? _codec.Encode(payload) : payload;
                _ramTier.Put(hashes[b], storedPayload);
                newlyCaptured++;
            }

            if (newlyCaptured > 0)
            {
                Interlocked.Add(ref _capturedBlocks, newlyCaptured);
                _logger.LogDebug(LogEventIds.PagedKvCacheCapture,
                    "kv.paged captured {Blocks} new blocks ({UpTo} tokens prepared)",
                    newlyCaptured, captureBlocks * blockSize);
            }
        }

        private bool TryFetch(KvBlockHash hash, out byte[] payload)
        {
            if (_ramTier.TryGet(hash, out payload))
                return true;
            if (_ssdTier != null && _ssdTier.TryRead(hash, out payload))
            {
                // Promote disk hit back into RAM so subsequent lookups are fast.
                _ramTier.Put(hash, payload);
                return true;
            }
            payload = null;
            return false;
        }
    }

    /// <summary>Lightweight snapshot of cache state for diagnostics / logging.</summary>
    public readonly record struct PagedKvCacheStats(
        long ramBytes,
        int ramBlocks,
        long ramMaxBytes,
        long ssdBytes,
        int ssdBlocks,
        long hitTokens,
        long missTokens,
        long capturedBlocks);
}
