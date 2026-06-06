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
using System.Buffers.Binary;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Integration tests that combine <see cref="PagedKvCacheManager"/> with the
/// <see cref="TurboQuantKvCodec"/> so we can verify the manager actually applies
/// the codec on capture and reverses it on restore. We use a model stub whose
/// "KV state" is a fp32 buffer of plausible activation values so the precision
/// envelope of the codec is the only source of error.
/// </summary>
public class PagedKvCacheCodecTests
{
    private const int BlockSize = 16;
    private const int NumLayers = 4;
    private const int NumKVHeads = 2;
    private const int HeadDim = 32;

    [Fact]
    public void PagedManager_WithInt4Codec_ShrinksResidentBytes()
    {
        var manager = NewEnabledManager("fp-int4", bits: 4);
        var model = new FloatFakeArchitecture("fp-int4", seed: 1);
        var tokens = Enumerable.Range(1, BlockSize * 3).ToArray();
        model.InjectFullState(tokens.Length);

        manager.Capture(model, tokens, tokens.Length);
        var stats = manager.GetStats();

        long rawBytes = model.ComputeKVBlockByteSize(BlockSize) * stats.ramBlocks;
        Assert.True(stats.ramBlocks > 0);
        Assert.True(stats.ramBytes < rawBytes,
            $"int4 codec should reduce resident bytes: encoded {stats.ramBytes} vs raw {rawBytes}");
        // Sanity check the compression ratio - int4 on fp32 should give ~5x.
        Assert.True(stats.ramBytes * 4 < rawBytes,
            $"int4 should compress fp32 by >=4x: encoded {stats.ramBytes} vs raw {rawBytes}");
    }

    [Fact]
    public void PagedManager_WithInt4Codec_RestoresWithinTolerance()
    {
        var manager = NewEnabledManager("fp-int4-rt", bits: 4);
        var model = new FloatFakeArchitecture("fp-int4-rt", seed: 7);
        var tokens = Enumerable.Range(1, BlockSize * 3).ToArray();
        model.InjectFullState(tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        var modelB = new FloatFakeArchitecture("fp-int4-rt", seed: 7);
        int restored = manager.TryRestorePrefix(modelB, tokens);
        Assert.True(restored >= BlockSize,
            $"Expected at least one restored block, got {restored}");

        // Bytes won't match exactly through int4 - but the underlying floats
        // should be within the int4 precision envelope (per-32 fp16-scaled
        // signed-4 quantization gives ~10% relative error in the worst case).
        float maxRelErr = model.MaxRelativeFloatErrorPrefix(modelB, restored);
        Assert.True(maxRelErr < 0.30f,
            $"int4 round-trip relative error {maxRelErr:G4} above tolerance");
    }

    [Fact]
    public void PagedManager_WithInt8Codec_RestoresWithTighterTolerance()
    {
        var manager = NewEnabledManager("fp-int8-rt", bits: 8);
        var model = new FloatFakeArchitecture("fp-int8-rt", seed: 11);
        var tokens = Enumerable.Range(1, BlockSize * 3).ToArray();
        model.InjectFullState(tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        var modelB = new FloatFakeArchitecture("fp-int8-rt", seed: 11);
        int restored = manager.TryRestorePrefix(modelB, tokens);
        Assert.True(restored >= BlockSize);

        float maxRelErr = model.MaxRelativeFloatErrorPrefix(modelB, restored);
        // int8 gives ~0.5% worst-case error; allow some slack for fp16 scale
        // quantization on top.
        Assert.True(maxRelErr < 0.05f,
            $"int8 round-trip relative error {maxRelErr:G4} above tolerance");
    }

    [Fact]
    public void PagedManager_WithoutCodec_BytesRoundTripExactly()
    {
        // Sanity check: passing null codec preserves the historical
        // bit-identical behaviour.
        var config = new PagedKvCacheConfig
        {
            Enabled = true,
            BlockSize = BlockSize,
            MaxRamBytes = 256L * 1024 * 1024,
        };
        var manager = new PagedKvCacheManager(config, "fp-raw", NullLogger.Instance, codec: null);
        var model = new FloatFakeArchitecture("fp-raw", seed: 13);
        var tokens = Enumerable.Range(1, BlockSize * 2).ToArray();
        model.InjectFullState(tokens.Length);
        manager.Capture(model, tokens, tokens.Length);

        var modelB = new FloatFakeArchitecture("fp-raw", seed: 13);
        int restored = manager.TryRestorePrefix(modelB, tokens);
        Assert.Equal(BlockSize, restored);
        Assert.True(model.BytesEqualPrefix(modelB, BlockSize));
    }

    private static PagedKvCacheManager NewEnabledManager(string fingerprint, int bits)
    {
        var config = new PagedKvCacheConfig
        {
            Enabled = true,
            BlockSize = BlockSize,
            MaxRamBytes = 256L * 1024 * 1024,
        };
        var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits);
        return new PagedKvCacheManager(config, fingerprint, NullLogger.Instance, codec);
    }

    /// <summary>
    /// Fake architecture whose KV state is a fp32 buffer filled with plausible
    /// post-RoPE activation values (Gaussian noise scaled to [-4, 4]). The
    /// per-token slice is large enough to span multiple 32-element codec
    /// groups so we exercise the boundary handling. Using realistic floats
    /// is critical: <see cref="PagedKvCacheTests.FakeArchitecture"/> stores
    /// constant-byte fills which round to fp16 denormals and collapse a
    /// quantizing codec to scale=0, producing all-zero reconstructions.
    /// </summary>
    private sealed class FloatFakeArchitecture : IModelArchitecture
    {
        private readonly string _fingerprint;
        private readonly int _seed;
        private float[] _floats = Array.Empty<float>();
        private int _cacheSeqLen;

        public FloatFakeArchitecture(string fingerprint, int seed)
        {
            _fingerprint = fingerprint;
            _seed = seed;
        }

        public ModelConfig Config { get; } = new ModelConfig();
        public ITokenizer Tokenizer => null;
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;

        public float[] Forward(int[] tokens) => Array.Empty<float>();
        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }

        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fingerprint;
        public KvCodecElementType KVStateElementType => KvCodecElementType.Float32;

        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);

        private static int FloatsPerToken => 2 * NumLayers * NumKVHeads * HeadDim;

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;

            int startFloat = startToken * FloatsPerToken;
            int floatCount = tokenCount * FloatsPerToken;
            for (int i = 0; i < floatCount; i++)
                BinaryPrimitives.WriteSingleLittleEndian(
                    destination.Slice(i * 4, 4), _floats[startFloat + i]);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;

            int needed = (destToken + tokenCount) * FloatsPerToken;
            if (_floats.Length < needed) Array.Resize(ref _floats, needed);
            int startFloat = destToken * FloatsPerToken;
            int floatCount = tokenCount * FloatsPerToken;
            for (int i = 0; i < floatCount; i++)
                _floats[startFloat + i] = BinaryPrimitives.ReadSingleLittleEndian(
                    source.Slice(i * 4, 4));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        // Test helper: deterministically fill the KV buffer with realistic
        // values derived from the seed and the token index so different
        // instances with the same seed see identical floats.
        internal void InjectFullState(int upToTokens)
        {
            int floatCount = upToTokens * FloatsPerToken;
            _floats = new float[floatCount];
            var rng = new Random(_seed);
            for (int i = 0; i < floatCount; i++)
            {
                // Box-Muller-ish (random in [-1, 1]) scaled to [-4, 4].
                float v = ((float)rng.NextDouble() * 2f - 1f) * 4f;
                _floats[i] = v;
            }
            _cacheSeqLen = upToTokens;
        }

        internal bool BytesEqualPrefix(FloatFakeArchitecture other, int tokenCount)
        {
            int floatCount = tokenCount * FloatsPerToken;
            if (_floats.Length < floatCount || other._floats.Length < floatCount) return false;
            for (int i = 0; i < floatCount; i++)
                if (_floats[i] != other._floats[i]) return false;
            return true;
        }

        internal float MaxRelativeFloatErrorPrefix(FloatFakeArchitecture other, int tokenCount)
        {
            int floatCount = tokenCount * FloatsPerToken;
            float maxRel = 0;
            // Per-group reference magnitude so we measure error against the
            // group's own scale (matches how the codec quantizes).
            const int groupSize = 32;
            for (int g = 0; g * groupSize < floatCount; g++)
            {
                int baseIdx = g * groupSize;
                float refMag = 0;
                for (int i = 0; i < groupSize && baseIdx + i < floatCount; i++)
                {
                    float a = Math.Abs(_floats[baseIdx + i]);
                    if (a > refMag) refMag = a;
                }
                if (refMag < 1e-6f) refMag = 1f;

                for (int i = 0; i < groupSize && baseIdx + i < floatCount; i++)
                {
                    float err = Math.Abs(_floats[baseIdx + i] - other._floats[baseIdx + i]) / refMag;
                    if (err > maxRel) maxRel = err;
                }
            }
            return maxRel;
        }
    }
}
