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
using System.Collections.Generic;
using System.Diagnostics;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Micro-benchmarks for the TurboQuant KV codec and the paged-tier integration.
/// Implemented as XUnit Facts so they run under <c>dotnet test</c> alongside the
/// correctness suite; numbers are printed to stdout (look for the
/// <c>[TurboQuant.*]</c> banner in the test output). The "block size" here
/// matches a realistic per-layer / per-head K/V slice for a 16-layer, 8-KV-head,
/// 128-headDim model with a 256-token block (the omlx-default block size) so
/// the timings are roughly representative of a production paged tier rather
/// than the toy sizes used by the correctness tests.
/// </summary>
public class TurboQuantKvCodecBenchmark
{
    // 16 layers × 8 KV heads × 256 tokens × 128 headDim × 2 (K+V) = 8,388,608 elements
    // ≈ 32 MB per block at fp32, 16 MB at fp16 — matches a single paged block on a
    // mid-sized model.
    private const int NumLayers = 16;
    private const int NumKVHeads = 8;
    private const int BlockTokens = 256;
    private const int HeadDim = 128;

    private static int ElementsPerBlock => 2 * NumLayers * NumKVHeads * BlockTokens * HeadDim;

    [Fact]
    public void Codec_Float32_Encode_Decode_Throughput_AllBitWidths()
    {
        const int warmup = 1;
        const int iters = 3;

        float[] values = SampleFloats(seed: 100, count: ElementsPerBlock, range: 4f);
        byte[] raw = FloatsToBytes(values);
        long rawBytes = raw.LongLength;

        Console.WriteLine($"[TurboQuant.fp32] Block: {ElementsPerBlock:N0} elements, " +
                          $"{rawBytes / 1024.0 / 1024.0:F2} MB raw");

        BenchOne(KvCodecElementType.Float32, raw, 4, warmup, iters, rawBytes);
        BenchOne(KvCodecElementType.Float32, raw, 8, warmup, iters, rawBytes);
    }

    [Fact]
    public void Codec_Float16_Encode_Decode_Throughput_AllBitWidths()
    {
        const int warmup = 1;
        const int iters = 3;

        float[] values = SampleFloats(seed: 200, count: ElementsPerBlock, range: 4f);
        byte[] raw = HalvesToBytes(values);
        long rawBytes = raw.LongLength;

        Console.WriteLine($"[TurboQuant.fp16] Block: {ElementsPerBlock:N0} elements, " +
                          $"{rawBytes / 1024.0 / 1024.0:F2} MB raw");

        BenchOne(KvCodecElementType.Float16, raw, 4, warmup, iters, rawBytes);
        BenchOne(KvCodecElementType.Float16, raw, 8, warmup, iters, rawBytes);
    }

    [Fact]
    public void Codec_Reconstruction_Precision_Envelope_Across_Distributions()
    {
        // Sweep three input distributions: uniform, Gaussian-like (sum of 8
        // uniforms), and bimodal (cluster at ±1). Real K/V activations are
        // closer to the Gaussian case; reporting all three lets a reader see
        // how each shapes the worst-case error.
        var distributions = new (string Name, Func<int, float[]> Sampler)[]
        {
            ("uniform[-4,4]", n => SampleFloats(seed: 7, count: n, range: 4f)),
            ("gaussian-like", n => SampleGaussianLike(seed: 11, count: n, scale: 1.5f)),
            ("bimodal±1",     n => SampleBimodal(seed: 13, count: n, mode: 1.0f, jitter: 0.2f)),
        };

        Console.WriteLine("[TurboQuant.precision] dtype=fp32 elements=" + (32 * 1024));
        foreach (var (name, sampler) in distributions)
        {
            float[] values = sampler(32 * 1024);
            byte[] raw = FloatsToBytes(values);

            foreach (int bits in new[] { 4, 8 })
            {
                var codec = new TurboQuantKvCodec(KvCodecElementType.Float32, bits);
                byte[] encoded = codec.Encode(raw);
                var decoded = new byte[raw.Length];
                Assert.True(codec.TryDecode(encoded, decoded));

                float[] roundTripped = BytesToFloats(decoded);
                var stats = ComputeErrorStats(values, roundTripped);
                Console.WriteLine($"  {name,-15} int{bits}: " +
                                  $"rms={stats.Rms:G4}  max={stats.Max:G4}  " +
                                  $"signal-rms={stats.SignalRms:G4}  " +
                                  $"snr={stats.SnrDb:F1}dB  " +
                                  $"size={encoded.Length / 1024:N0}KB / {raw.Length / 1024:N0}KB " +
                                  $"({(double)raw.Length / encoded.Length:F2}x)");
            }
        }
    }

    [Fact]
    public void PagedManager_With_Codec_Endtoend_Throughput_Vs_Passthrough()
    {
        const int blocksToCapture = 4;

        // Smaller block size so the test finishes quickly while still hitting
        // the codec's hot path multiple times per call.
        int blockTokens = BlockTokens;
        int floatsPerToken = 2 * NumLayers * NumKVHeads * HeadDim;
        int totalTokens = blockTokens * blocksToCapture;

        // Build a model stub with realistic floats.
        var model = new BenchFakeArchitecture("bench-fp", seed: 17, floatsPerToken);
        var tokens = new int[totalTokens];
        for (int i = 0; i < totalTokens; i++) tokens[i] = i + 1;
        model.Fill(totalTokens);

        var pagedConfig = new PagedKvCacheConfig
        {
            Enabled = true,
            BlockSize = blockTokens,
            MaxRamBytes = 4L * 1024 * 1024 * 1024,
        };

        long passthroughBytes = MeasureCaptureBytes(pagedConfig, codec: null, model, tokens, totalTokens, out double passthroughMs);
        long int4Bytes = MeasureCaptureBytes(pagedConfig,
            codec: new TurboQuantKvCodec(KvCodecElementType.Float32, 4),
            model, tokens, totalTokens, out double int4Ms);
        long int8Bytes = MeasureCaptureBytes(pagedConfig,
            codec: new TurboQuantKvCodec(KvCodecElementType.Float32, 8),
            model, tokens, totalTokens, out double int8Ms);

        Console.WriteLine($"[TurboQuant.paged] blocks={blocksToCapture} bytesPerBlockRaw={model.ComputeKVBlockByteSize(blockTokens) / 1024.0 / 1024.0:F2} MB");
        Console.WriteLine($"  passthrough   : {passthroughBytes / 1024 / 1024:N0} MB resident, capture {passthroughMs:F1} ms");
        Console.WriteLine($"  int4 codec    : {int4Bytes / 1024 / 1024:N0} MB resident, capture {int4Ms:F1} ms ({(double)passthroughBytes / int4Bytes:F2}x smaller)");
        Console.WriteLine($"  int8 codec    : {int8Bytes / 1024 / 1024:N0} MB resident, capture {int8Ms:F1} ms ({(double)passthroughBytes / int8Bytes:F2}x smaller)");

        Assert.True(int4Bytes * 4 < passthroughBytes,
            $"int4 codec must reduce resident bytes by >=4x (passthrough={passthroughBytes}, int4={int4Bytes})");
        Assert.True(int8Bytes * 2 < passthroughBytes,
            $"int8 codec must reduce resident bytes by >=2x (passthrough={passthroughBytes}, int8={int8Bytes})");
    }

    // ---- Helpers ----

    private static void BenchOne(KvCodecElementType dtype, byte[] raw, int bits, int warmup, int iters, long rawBytes)
    {
        var codec = new TurboQuantKvCodec(dtype, bits);

        // Warm up
        byte[] encoded = null;
        var decoded = new byte[raw.Length];
        for (int w = 0; w < warmup; w++)
        {
            encoded = codec.Encode(raw);
            codec.TryDecode(encoded, decoded);
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iters; i++)
            encoded = codec.Encode(raw);
        double encMs = sw.Elapsed.TotalMilliseconds / iters;

        sw.Restart();
        for (int i = 0; i < iters; i++)
            codec.TryDecode(encoded, decoded);
        double decMs = sw.Elapsed.TotalMilliseconds / iters;

        double encMBs = (rawBytes / 1024.0 / 1024.0) / (encMs / 1000.0);
        double decMBs = (rawBytes / 1024.0 / 1024.0) / (decMs / 1000.0);
        double ratio = (double)raw.Length / encoded.Length;

        Console.WriteLine($"  int{bits,-2}: encoded={encoded.Length / 1024.0 / 1024.0:F2}MB " +
                          $"({ratio:F2}x compression)  " +
                          $"enc={encMs:F1}ms ({encMBs:F0} MB/s)  " +
                          $"dec={decMs:F1}ms ({decMBs:F0} MB/s)");
    }

    private static long MeasureCaptureBytes(
        PagedKvCacheConfig cfg, IKvBlockCodec codec, BenchFakeArchitecture model,
        int[] tokens, int upToTokens, out double captureMs)
    {
        var manager = new PagedKvCacheManager(cfg, "bench-fp", NullLogger.Instance, codec);
        try
        {
            var sw = Stopwatch.StartNew();
            manager.Capture(model, tokens, upToTokens);
            captureMs = sw.Elapsed.TotalMilliseconds;
            return manager.GetStats().ramBytes;
        }
        finally
        {
            manager.Dispose();
        }
    }

    private static float[] SampleFloats(int seed, int count, float range)
    {
        var rng = new Random(seed);
        var buf = new float[count];
        for (int i = 0; i < count; i++)
            buf[i] = ((float)rng.NextDouble() * 2f - 1f) * range;
        return buf;
    }

    private static float[] SampleGaussianLike(int seed, int count, float scale)
    {
        // Sum of 8 uniforms ≈ Gaussian by CLT; scale to roughly N(0, scale).
        var rng = new Random(seed);
        var buf = new float[count];
        for (int i = 0; i < count; i++)
        {
            float sum = 0;
            for (int j = 0; j < 8; j++) sum += (float)rng.NextDouble();
            buf[i] = (sum - 4f) * scale * 0.5f;
        }
        return buf;
    }

    private static float[] SampleBimodal(int seed, int count, float mode, float jitter)
    {
        var rng = new Random(seed);
        var buf = new float[count];
        for (int i = 0; i < count; i++)
        {
            float sign = rng.Next(2) == 0 ? -1f : 1f;
            float noise = ((float)rng.NextDouble() * 2f - 1f) * jitter;
            buf[i] = sign * mode + noise;
        }
        return buf;
    }

    private static byte[] FloatsToBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int i = 0; i < values.Length; i++)
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), values[i]);
        return bytes;
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        float[] values = new float[bytes.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4, 4));
        return values;
    }

    private static byte[] HalvesToBytes(float[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int i = 0; i < values.Length; i++)
        {
            ushort bits = System.BitConverter.HalfToUInt16Bits((System.Half)values[i]);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2, 2), bits);
        }
        return bytes;
    }

    private record struct ErrorStats(double Rms, double Max, double SignalRms, double SnrDb);

    private static ErrorStats ComputeErrorStats(float[] expected, float[] actual)
    {
        double sumSqErr = 0, sumSqSig = 0, maxErr = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double err = expected[i] - actual[i];
            double absErr = Math.Abs(err);
            if (absErr > maxErr) maxErr = absErr;
            sumSqErr += err * err;
            sumSqSig += (double)expected[i] * expected[i];
        }
        double rms = Math.Sqrt(sumSqErr / expected.Length);
        double sigRms = Math.Sqrt(sumSqSig / expected.Length);
        double snr = sigRms > 0 && rms > 0 ? 20 * Math.Log10(sigRms / rms) : double.PositiveInfinity;
        return new ErrorStats(rms, maxErr, sigRms, snr);
    }

    private sealed class BenchFakeArchitecture : IModelArchitecture
    {
        private readonly string _fingerprint;
        private readonly int _seed;
        private readonly int _floatsPerToken;
        private float[] _floats = Array.Empty<float>();
        private int _cacheSeqLen;

        public BenchFakeArchitecture(string fingerprint, int seed, int floatsPerToken)
        {
            _fingerprint = fingerprint;
            _seed = seed;
            _floatsPerToken = floatsPerToken;
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
            => (long)tokenCount * _floatsPerToken * sizeof(float);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;

            int startFloat = startToken * _floatsPerToken;
            int floatCount = tokenCount * _floatsPerToken;
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

            int needed = (destToken + tokenCount) * _floatsPerToken;
            if (_floats.Length < needed) Array.Resize(ref _floats, needed);
            int startFloat = destToken * _floatsPerToken;
            int floatCount = tokenCount * _floatsPerToken;
            for (int i = 0; i < floatCount; i++)
                _floats[startFloat + i] = BinaryPrimitives.ReadSingleLittleEndian(
                    source.Slice(i * 4, 4));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        public void Fill(int upToTokens)
        {
            int floatCount = upToTokens * _floatsPerToken;
            _floats = new float[floatCount];
            var rng = new Random(_seed);
            for (int i = 0; i < floatCount; i++)
                _floats[i] = ((float)rng.NextDouble() * 2f - 1f) * 4f;
            _cacheSeqLen = upToTokens;
        }
    }
}
