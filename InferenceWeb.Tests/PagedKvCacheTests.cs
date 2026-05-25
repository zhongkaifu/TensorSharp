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
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for the paged KV-cache machinery. These tests exercise the runtime
/// layer in isolation - they do not load a real model. A <see cref="FakeArchitecture"/>
/// stub stands in for <see cref="IModelArchitecture"/> and reports a deterministic
/// fingerprint plus a byte payload sized to a configurable layout, so we can verify
/// hashing, restoration, capture, and the in-memory LRU end-to-end.
/// </summary>
public class PagedKvCacheTests
{
    private const int BlockSize = 16;
    private const int NumLayers = 4;
    private const int NumKVHeads = 2;
    private const int HeadDim = 32;

    [Fact]
    public void BlockHasher_ProducesDeterministicChainHashes()
    {
        var tokens = Enumerable.Range(0, BlockSize * 3).ToList();
        var first = KvBlockHasher.ComputeBlockHashes(tokens, BlockSize, "model-A");
        var second = KvBlockHasher.ComputeBlockHashes(tokens, BlockSize, "model-A");

        Assert.Equal(3, first.Count);
        Assert.Equal(first, second);
    }

    [Fact]
    public void BlockHasher_DiffersAcrossFingerprints()
    {
        var tokens = Enumerable.Range(0, BlockSize * 2).ToList();
        var a = KvBlockHasher.ComputeBlockHashes(tokens, BlockSize, "model-A");
        var b = KvBlockHasher.ComputeBlockHashes(tokens, BlockSize, "model-B");
        Assert.Equal(a.Count, b.Count);
        Assert.NotEqual(a[0], b[0]);
    }

    [Fact]
    public void BlockHasher_ChainsParent_DivergesAfterFirstMismatch()
    {
        var commonPrefix = Enumerable.Range(0, BlockSize).ToList();
        var seqA = new List<int>(commonPrefix);
        seqA.AddRange(Enumerable.Repeat(100, BlockSize));
        var seqB = new List<int>(commonPrefix);
        seqB.AddRange(Enumerable.Repeat(200, BlockSize));

        var hashesA = KvBlockHasher.ComputeBlockHashes(seqA, BlockSize, "fp");
        var hashesB = KvBlockHasher.ComputeBlockHashes(seqB, BlockSize, "fp");

        Assert.Equal(hashesA[0], hashesB[0]);
        Assert.NotEqual(hashesA[1], hashesB[1]);
    }

    [Fact]
    public void BlockHasher_TrailingPartialBlock_IsIgnored()
    {
        var tokens = Enumerable.Range(0, BlockSize * 2 + 5).ToList();
        var hashes = KvBlockHasher.ComputeBlockHashes(tokens, BlockSize, "fp");
        Assert.Equal(2, hashes.Count);
    }

    [Fact]
    public void Manager_Disabled_NoOpsAndProducesEmptyStats()
    {
        var manager = PagedKvCacheManager.Disabled();
        Assert.False(manager.IsEnabled);
        Assert.Equal(0, manager.TryRestorePrefix(new FakeArchitecture("fp"), new[] { 1, 2 }));
        manager.Capture(new FakeArchitecture("fp"), new[] { 1, 2 }, 2); // must not throw
        var stats = manager.GetStats();
        Assert.Equal(0, stats.ramBlocks);
    }

    [Fact]
    public void Manager_CaptureThenRestore_RoundTripsBytes()
    {
        var manager = NewEnabledManager("fp-1");
        var model = new FakeArchitecture("fp-1");
        var tokens = Enumerable.Range(1, BlockSize * 2).ToArray();

        // Capture from "session A"
        model.InjectFullState(tokens, tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        // Fresh model state ("session B") and restore
        var modelB = new FakeArchitecture("fp-1");
        int restored = manager.TryRestorePrefix(modelB, tokens);

        // Last block is reserved for the forward pass so the model still produces logits.
        Assert.Equal(BlockSize, restored);
        Assert.Equal(BlockSize, modelB.CurrentSeqLen);
        Assert.True(modelB.BytesEqualPrefix(model, BlockSize));
    }

    [Fact]
    public void Manager_RestoreLastFullBlock_HeldBackForForwardLogits()
    {
        var manager = NewEnabledManager("fp-2");
        var model = new FakeArchitecture("fp-2");
        var tokens = Enumerable.Range(1, BlockSize * 3).ToArray();
        model.InjectFullState(tokens, tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        var modelB = new FakeArchitecture("fp-2");
        int restored = manager.TryRestorePrefix(modelB, tokens);

        // 3 blocks captured; 2 restored so the third can re-compute fresh logits.
        Assert.Equal(BlockSize * 2, restored);
    }

    [Fact]
    public void Manager_FingerprintMismatch_RejectsRestore()
    {
        var manager = NewEnabledManager("fp-x");
        var model = new FakeArchitecture("fp-x");
        var tokens = Enumerable.Range(1, BlockSize * 2).ToArray();
        model.InjectFullState(tokens, tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        var foreign = new FakeArchitecture("fp-DIFFERENT");
        int restored = manager.TryRestorePrefix(foreign, tokens);
        Assert.Equal(0, restored);
    }

    [Fact]
    public void Manager_CountAvailableBlocks_OnlyCountsContiguousPrefix()
    {
        var manager = NewEnabledManager("fp-3");
        var model = new FakeArchitecture("fp-3");
        var tokens = Enumerable.Range(1, BlockSize * 4).ToArray();

        // Capture only the first 3 blocks
        model.InjectFullState(tokens, BlockSize * 3);
        manager.Capture(model, tokens, BlockSize * 3);

        // A request for the full 4-block sequence finds 3 leading blocks
        Assert.Equal(3, manager.CountAvailableBlocks(new FakeArchitecture("fp-3"), tokens));

        // A different prompt that diverges in block 2 finds only 1 matching block
        var diverging = tokens.ToArray();
        diverging[BlockSize + 1] = 9999;
        Assert.Equal(1, manager.CountAvailableBlocks(new FakeArchitecture("fp-3"), diverging));
    }

    [Fact]
    public void Manager_PromotesBlocksFromSsdToRam()
    {
        string ssdDir = Path.Combine(Path.GetTempPath(), "tensorsharp-paged-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            // A tiny RAM tier (only 1 block worth) forces immediate spill on the 2nd Put.
            var config = new PagedKvCacheConfig
            {
                Enabled = true,
                BlockSize = BlockSize,
                MaxRamBytes = BytesPerBlock(),
                SsdDirectory = ssdDir,
                MaxSsdBytes = 64L * 1024 * 1024,
            };
            var manager = new PagedKvCacheManager(config, "fp-ssd", NullLogger.Instance);
            try
            {
                var model = new FakeArchitecture("fp-ssd");
                var tokens = Enumerable.Range(1, BlockSize * 4).ToArray();
                model.InjectFullState(tokens, tokens.Length);
                manager.Capture(model, tokens, tokens.Length);

                // Force the SSD writer to drain so its index is hot when we restore.
                manager.Dispose();
                manager = new PagedKvCacheManager(config, "fp-ssd", NullLogger.Instance);

                var modelB = new FakeArchitecture("fp-ssd");
                int restored = manager.TryRestorePrefix(modelB, tokens);
                Assert.True(restored >= BlockSize);
                Assert.True(modelB.BytesEqualPrefix(model, restored));
            }
            finally
            {
                manager.Dispose();
            }
        }
        finally
        {
            try { if (Directory.Exists(ssdDir)) Directory.Delete(ssdDir, recursive: true); } catch { }
        }
    }

    [Fact]
    public void Manager_RamTier_EvictsLruWhenOverBudget()
    {
        // Two-block RAM budget. The third unique block must evict the LRU entry.
        var config = new PagedKvCacheConfig
        {
            Enabled = true,
            BlockSize = BlockSize,
            MaxRamBytes = BytesPerBlock() * 2,
        };
        var manager = new PagedKvCacheManager(config, "fp-evict", NullLogger.Instance);
        try
        {
            var model = new FakeArchitecture("fp-evict");

            var seqA = Enumerable.Range(1, BlockSize).ToArray();
            var seqB = Enumerable.Range(1000, BlockSize).ToArray();
            var seqC = Enumerable.Range(2000, BlockSize).ToArray();

            void CaptureOne(int[] tokens)
            {
                var fresh = new FakeArchitecture("fp-evict");
                fresh.InjectFullState(tokens, tokens.Length);
                manager.Capture(fresh, tokens, tokens.Length);
            }

            CaptureOne(seqA);
            CaptureOne(seqB);
            CaptureOne(seqC);

            var stats = manager.GetStats();
            Assert.True(stats.ramBlocks <= 2,
                $"Expected at most 2 blocks resident under {BytesPerBlock() * 2}-byte cap, got {stats.ramBlocks}");
            Assert.True(stats.ramBytes <= config.MaxRamBytes);
        }
        finally
        {
            manager.Dispose();
        }
    }

    [Fact]
    public void Manager_ModelReportsUnsupported_NoCaptureAndNoRestore()
    {
        var manager = NewEnabledManager("fp-unsup");
        var unsupportedModel = new FakeArchitecture("fp-unsup", supportsSnapshot: false);
        var tokens = Enumerable.Range(1, BlockSize * 2).ToArray();

        manager.Capture(unsupportedModel, tokens, tokens.Length);
        Assert.Equal(0, manager.GetStats().ramBlocks);

        int restored = manager.TryRestorePrefix(unsupportedModel, tokens);
        Assert.Equal(0, restored);
    }

    private static PagedKvCacheManager NewEnabledManager(string fingerprint)
    {
        var config = new PagedKvCacheConfig
        {
            Enabled = true,
            BlockSize = BlockSize,
            MaxRamBytes = 256L * 1024 * 1024,
        };
        return new PagedKvCacheManager(config, fingerprint, NullLogger.Instance);
    }

    private static long BytesPerBlock() =>
        // 2 (K, V) * NumLayers * NumKVHeads * BlockSize * HeadDim, F32 (4 bytes).
        2L * NumLayers * NumKVHeads * BlockSize * HeadDim * 4;

    /// <summary>
    /// Minimal <see cref="IModelArchitecture"/> stub. Its "KV state" is a single byte
    /// buffer sized as if the model had NumLayers layers each with K and V tensors
    /// of shape (NumKVHeads, capacity, HeadDim) in F32. The contents are derived
    /// from token IDs so equal prefixes produce equal bytes - which lets the round-
    /// trip tests verify that the captured bytes were stored / restored correctly.
    /// </summary>
    private sealed class FakeArchitecture : IModelArchitecture
    {
        private readonly string _fingerprint;
        private readonly bool _supportsSnapshot;
        private byte[] _state = Array.Empty<byte>();
        private int _cacheSeqLen;

        public FakeArchitecture(string fingerprint, bool supportsSnapshot = true)
        {
            _fingerprint = fingerprint;
            _supportsSnapshot = supportsSnapshot;
        }

        public ModelConfig Config { get; } = new ModelConfig();
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public int CurrentSeqLen => _cacheSeqLen;

        public float[] Forward(int[] tokens) => Array.Empty<float>();
        public void ResetKVCache()
        {
            _cacheSeqLen = 0;
            // We keep _state allocated for the test - injection rewrites it.
        }
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }

        public bool SupportsKVStateSnapshot => _supportsSnapshot;
        public string KVStateFingerprint => _fingerprint;

        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!_supportsSnapshot)
                return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;

            long startByte = ComputeKVBlockByteSize(startToken);
            new ReadOnlySpan<byte>(_state, (int)startByte, (int)expected).CopyTo(destination);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!_supportsSnapshot)
                return false;
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;

            long needed = ComputeKVBlockByteSize(destToken + tokenCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            long startByte = ComputeKVBlockByteSize(destToken);
            source.CopyTo(new Span<byte>(_state, (int)startByte, (int)expected));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        // Test helper: fill _state with content derived from token IDs.
        internal void InjectFullState(int[] tokens, int upToTokens)
        {
            long bytes = ComputeKVBlockByteSize(upToTokens);
            _state = new byte[bytes];
            // Per-token deterministic fill so we can assert byte equality on restore.
            for (int t = 0; t < upToTokens; t++)
            {
                long start = ComputeKVBlockByteSize(t);
                long end = ComputeKVBlockByteSize(t + 1);
                byte v = (byte)(tokens[t] & 0xFF);
                for (long i = start; i < end; i++)
                    _state[i] = v;
            }
            _cacheSeqLen = upToTokens;
        }

        internal bool BytesEqualPrefix(FakeArchitecture other, int tokenCount)
        {
            long bytes = ComputeKVBlockByteSize(tokenCount);
            if (_state.Length < bytes || other._state.Length < bytes) return false;
            for (long i = 0; i < bytes; i++)
                if (_state[i] != other._state[i]) return false;
            return true;
        }
    }
}
