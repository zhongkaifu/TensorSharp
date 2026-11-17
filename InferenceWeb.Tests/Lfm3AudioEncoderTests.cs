// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3-Omni's audio encoder (Lfm3AudioEncoder) on the real mmproj (TS_LFM3_MMPROJ: mmproj-LFM3-Omni-2B-Q8_0.gguf or
/// -F16.gguf) against llama.cpp's mtmd rows for the same clips (InferenceWeb.Tests/Fixtures/Lfm3Audio, from
/// eng/lfm3/audio-in), on every executor this machine runs: GGML metal and cpu (one per process: TS_TEST_GGML_BACKEND), the
/// managed CPU and MLX; plus chunked and streamed encoding against the whole clip, and the Metal encode time.
/// </summary>
public sealed class Lfm3AudioEncoderTests
{
    public enum Lane { Managed, Reference, GgmlCpu, GgmlMetal, Mlx }

    /// <summary>The tensor executor with none of a backend's kernels, on the CPU: the reference decompositions direct CUDA
    /// runs for the network ops it has no kernel of (TensorGraphExecutor.Network.cs), here with the products on the host.</summary>
    private sealed class ReferenceExecutor : TensorGraphExecutor
    {
        public ReferenceExecutor(IAllocator allocator) : base(allocator) { }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            int rows = (int)x.Sizes[0];
            float[] host = x.GetElementsAsFloat((int)x.ElementCount());
            var y = new float[rows * weight.Ne1];
            HostQuantizedProduct(weight, host, rows, y);
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            result.SetElementsAsFloat(y);
            return result;
        }
    }

    private readonly ITestOutputHelper _output;

    public Lfm3AudioEncoderTests(ITestOutputHelper output) => _output = output;

    /// <summary>The encoder on one lane's allocator.</summary>
    private sealed class LaneEncoder : IDisposable
    {
        private readonly IDisposable _owner;
        private readonly bool _ggml;

        public LaneEncoder(Lane lane)
        {
            IAllocator allocator;
            BackendType backend;
            IGraphExecutor executor = null;
            switch (lane)
            {
                case Lane.Managed:
                case Lane.Reference:
                    allocator = new CpuAllocator(BlasEnum.DotNet);
                    backend = BackendType.Cpu;
                    if (lane == Lane.Reference) executor = new ReferenceExecutor(allocator);
                    break;
                case Lane.Mlx:
                    var mlx = new MlxAllocator();
                    allocator = mlx;
                    _owner = mlx;
                    backend = BackendType.Mlx;
                    break;
                default:
                    allocator = new GgmlAllocator(new GgmlContext([0], lane == Lane.GgmlMetal ? GgmlBackendType.Metal : GgmlBackendType.Cpu), 0);
                    backend = lane == Lane.GgmlMetal ? BackendType.GgmlMetal : BackendType.GgmlCpu;
                    _ggml = true;
                    break;
            }
            using (var gguf = new GgufFile(Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ")))
                Encoder = Lfm3AudioEncoder.Load(gguf, allocator, backend, executor);
            Assert.NotNull(Encoder);
            // The lane runs on the executor it names.
            switch (lane)
            {
                case Lane.Managed: Assert.IsType<ManagedGraphExecutor>(Encoder.Executor); break;
                case Lane.Reference: Assert.IsType<ReferenceExecutor>(Encoder.Executor); break;
                case Lane.Mlx: Assert.IsType<MlxGraphExecutor>(Encoder.Executor); break;
                default:
                    Assert.Equal(lane == Lane.GgmlMetal ? GgmlBackendType.Metal : GgmlBackendType.Cpu, Assert.IsType<GgmlGraphExecutor>(Encoder.Executor).Backend);
                    break;
            }
        }

        public Lfm3AudioEncoder Encoder { get; }

        public void Dispose()
        {
            Encoder.Dispose();
            _owner?.Dispose();
            if (_ggml)
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
        }
    }

    // GGML metal: llama.cpp's own backend and precision (F16 K/V in the windowed attention). Where the two differ is the
    // pointwise convs: llama.cpp runs them as F32 direct convolutions, this program as products, which ggml-metal's
    // matrix-matrix kernel (and MLX's) computes over half-precision tiles; rows then differ by ~1e-3 relative. The CPU lanes
    // multiply block weights against activations quantized to 8 bits (ggml-cpu and the managed kernels alike), a few
    // percent over 17 layers.
    [Lfm3AudioFact(Lane.GgmlMetal)]
    public void MetalRowsMatchLlamaCpp() => Compare(Lane.GgmlMetal, Lfm3AudioSignals.Names, minCosine: 0.9999, normTolerance: 1e-2);

    [Lfm3AudioFact(Lane.Mlx)]
    public void MlxRowsMatchLlamaCpp() => Compare(Lane.Mlx, Lfm3AudioSignals.Names, minCosine: 0.9999, normTolerance: 1e-2);

    [Lfm3AudioFact(Lane.GgmlCpu)]
    public void GgmlCpuRowsMatchLlamaCpp() => Compare(Lane.GgmlCpu, new[] { "chirp_1s", "noise_odd", "tiny" }, minCosine: 0.995, normTolerance: 3e-2);

    [Lfm3AudioFact(Lane.Managed)]
    public void ManagedCpuRowsMatchLlamaCpp() => Compare(Lane.Managed, new[] { "chirp_1s", "noise_odd", "tiny" }, minCosine: 0.995, normTolerance: 3e-2);

    /// <summary>Direct CUDA's decompositions (validated here on the CPU; no CUDA device ran them).</summary>
    [Lfm3AudioFact(Lane.Reference)]
    public void ReferenceDecompositionRowsMatchLlamaCpp() => Compare(Lane.Reference, new[] { "chirp_1s", "tiny" }, minCosine: 0.995, normTolerance: 3e-2);

    /// <summary>Each clip's rows against the fixture: the row count; three rows' first 128 values (cosine); every row's L2
    /// norm (relative difference).</summary>
    private void Compare(Lane lane, IEnumerable<string> clips, double minCosine, double normTolerance)
    {
        string mmproj = Path.GetFileName(Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ"));
        JsonElement fixture = Lfm3AudioFrontEndTests.Fixture();
        using var lanes = new LaneEncoder(lane);
        Lfm3AudioEncoder encoder = lanes.Encoder;
        foreach (string clip in clips)
        {
            JsonElement reference = fixture.GetProperty(clip);
            Assert.True(reference.GetProperty("outputs").TryGetProperty(mmproj, out JsonElement expected),
                $"the fixture has no llama.cpp rows for {mmproj} (it has {string.Join(", ", reference.GetProperty("outputs").EnumerateObject().Select(p => p.Name))})");
            float[] pcm = Lfm3AudioSignals.Make(clip, Lfm3AudioFrontEndTests.RepoRoot());
            var sw = Stopwatch.StartNew();
            float[] rows = encoder.Encode(pcm);
            double ms = sw.Elapsed.TotalMilliseconds;
            int width = encoder.OutputWidth, n = rows.Length / width;
            Assert.Equal(reference.GetProperty("rows").GetInt32(), n);

            double worstCos = 1;
            int[] index = expected.GetProperty("row_index").EnumerateArray().Select(v => v.GetInt32()).ToArray();
            JsonElement[] heads = expected.GetProperty("row_head").EnumerateArray().ToArray();
            for (int i = 0; i < index.Length; i++)
            {
                float[] head = Lfm3AudioFrontEndTests.Floats(heads[i]);
                worstCos = Math.Min(worstCos, Cosine(rows.AsSpan(index[i] * width, head.Length), head));
            }
            float[] norms = Lfm3AudioFrontEndTests.Floats(expected.GetProperty("row_norm"));
            double worstNorm = 0;
            for (int r = 0; r < n; r++)
            {
                double s = 0;
                for (int j = 0; j < width; j++) s += (double)rows[r * width + j] * rows[r * width + j];
                worstNorm = Math.Max(worstNorm, Math.Abs(Math.Sqrt(s) - norms[r]) / norms[r]);
            }
            string line = $"{lane} {mmproj} {clip}: {n} rows in {ms:F1} ms; sampled rows min cosine {worstCos:F6}, row norms max rel diff {worstNorm:G3}";
            _output.WriteLine(line);
            Assert.True(worstCos >= minCosine && worstNorm <= normTolerance, line);
        }
    }

    private static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double ab = 0, aa = 0, bb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            ab += (double)a[i] * b[i];
            aa += (double)a[i] * a[i];
            bb += (double)b[i] * b[i];
        }
        return ab / Math.Sqrt(aa * bb);
    }

    /// <summary>
    /// The encoder streams exactly: the whole clip in one program, the same clip in programs of 64 frames (carried state), a
    /// live stream fed 0.1 s at a time encoding every 32 frames, and a clip after another clip (a fresh state) all give the
    /// same rows - bit for bit on the managed CPU with Q8_0 weights (its F16 products block by rows: ~5e-7). The GPU kernels do:
    /// ggml-metal (and MLX) multiply few rows with float matrix-vector kernels and many with half-precision tiles, so a
    /// program of 8 rows and one of 125 round differently (~1e-3 relative), as llama.cpp's own streamed and whole-clip
    /// encodes do.
    /// </summary>
    [Lfm3AudioFact(Lane.GgmlMetal)]
    public void MetalChunkedAndStreamedRowsEqualTheWholeClip() => ChunkedAndStreamed(Lane.GgmlMetal, "speech_10s", 3e-3);

    [Lfm3AudioFact(Lane.Managed)]
    public void ManagedChunkedAndStreamedRowsEqualTheWholeClip() => ChunkedAndStreamed(Lane.Managed, "chirp_1s", 1e-5);

    [Lfm3AudioFact(Lane.Reference)]
    public void ReferenceDecompositionChunkedAndStreamedRowsEqualTheWholeClip() => ChunkedAndStreamed(Lane.Reference, "chirp_1s", 1e-5);

    [Lfm3AudioFact(Lane.Mlx)]
    public void MlxChunkedAndStreamedRowsEqualTheWholeClip() => ChunkedAndStreamed(Lane.Mlx, "speech_6s", 3e-3);

    private void ChunkedAndStreamed(Lane lane, string clip, double tolerance)
    {
        using var lanes = new LaneEncoder(lane);
        Lfm3AudioEncoder encoder = lanes.Encoder;
        string root = Lfm3AudioFrontEndTests.RepoRoot();
        float[] pcm = Lfm3AudioSignals.Make(clip, root);
        float[] whole = encoder.Encode(pcm);

        encoder.MaxChunkFrames = 64;
        float[] chunked = encoder.Encode(pcm);
        encoder.MaxChunkFrames = 3000;

        float[] Stream(int stepFrames)
        {
            using Lfm3AudioStream stream = encoder.OpenAudioStream(stepFrames);
            var rows = new List<float>();
            for (int at = 0; at < pcm.Length; at += 1600)
            {
                float[] got = stream.Push(pcm.AsSpan(at, Math.Min(1600, pcm.Length - at)));
                Assert.Equal(0, got.Length % encoder.OutputWidth);
                rows.AddRange(got);
            }
            rows.AddRange(stream.Finish());
            Assert.Equal(rows.Count / encoder.OutputWidth, stream.Rows);
            Assert.Throws<InvalidOperationException>(() => stream.Encoder.Encode(new float[128 * 8], 8, final: false));
            return rows.ToArray();
        }
        float[] streamed = Stream(32);
        // A second stream (new state tensors, possibly where the first one's were) starts clean too.
        float[] streamedAgain = Stream(96);

        // Another clip first, then this one: the second encode starts from a fresh state.
        encoder.Encode(Lfm3AudioSignals.Make("music_10s", root));
        float[] again = encoder.Encode(pcm);

        double chunkedErr = RmsRel(chunked, whole), streamedErr = RmsRel(streamed, whole), streamed2Err = RmsRel(streamedAgain, whole);
        double againErr = RmsRel(again, whole);
        string line = $"{lane} {clip}: {whole.Length / encoder.OutputWidth} rows; 64-frame programs rms rel {chunkedErr:G3}, " +
            $"live stream (32-frame steps) {streamedErr:G3}, a second stream (96-frame steps) {streamed2Err:G3}, after another clip {againErr:G3}";
        _output.WriteLine(line);
        Assert.Equal(whole.Length, chunked.Length);
        Assert.True(chunkedErr <= tolerance && streamedErr <= tolerance && streamed2Err <= tolerance && againErr <= tolerance, line);
        Assert.Throws<ArgumentException>(() =>
        {
            using var s = encoder.OpenStream();
            s.Encode(new float[128 * 12], 12, final: false);
        });
    }

    private static double RmsRel(float[] actual, float[] reference)
    {
        Assert.Equal(reference.Length, actual.Length);
        double dd = 0, rr = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            dd += d * d;
            rr += (double)reference[i] * reference[i];
        }
        return Math.Sqrt(dd / rr);
    }

    /// <summary>The encoder's time on Metal for 1 s, 10 s and 30 s of speech (front end and encoder; first run, then the
    /// median of five), against which llama.cpp's mtmd encode of the same clips (eng/lfm3/audio-in) is the reference.</summary>
    [Lfm3AudioFact(Lane.GgmlMetal)]
    public void MetalEncodeSpeed()
    {
        using var lanes = new LaneEncoder(Lane.GgmlMetal);
        Lfm3AudioEncoder encoder = lanes.Encoder;
        string root = Lfm3AudioFrontEndTests.RepoRoot();
        foreach (string clip in new[] { "chirp_1s", "speech_10s", "speech_30s" })
        {
            float[] pcm = Lfm3AudioSignals.Make(clip, root);
            var sw = Stopwatch.StartNew();
            encoder.Encode(pcm);
            double first = sw.Elapsed.TotalMilliseconds;
            var front = new double[5];
            var enc = new double[5];
            for (int i = 0; i < 5; i++)
            {
                sw.Restart();
                float[] features = Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
                front[i] = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                encoder.EncodeFeatures(features, frames);
                enc[i] = sw.Elapsed.TotalMilliseconds;
            }
            Array.Sort(front);
            Array.Sort(enc);
            string line = $"lfm3 audio encoder Metal {clip} ({pcm.Length / 16000.0:F1} s, {Lfm3AudioPreprocessor.RowCountForSamples(pcm.Length)} rows): " +
                $"first run {first:F1} ms; front end median {front[2]:F2} ms, encoder median {enc[2]:F2} ms (min {enc[0]:F2})";
            _output.WriteLine(line);
            Console.WriteLine(line);
        }
        _output.WriteLine($"programs: {encoder.Graph.Builds} builds, {encoder.Graph.Replays} replays");
    }

    [Fact]
    public void PromptHelpers()
    {
        Assert.Equal("Perform ASR.", Lfm3AudioPrompt.AsrSystemPrompt());
        Assert.Equal("Perform ASR in English.", Lfm3AudioPrompt.AsrSystemPrompt("en"));
        Assert.Equal("Perform ASR in Chinese.", Lfm3AudioPrompt.AsrSystemPrompt("ZH"));
        Assert.Equal("Perform ASR in Klingon.", Lfm3AudioPrompt.AsrSystemPrompt("Klingon"));
        Assert.Equal(18, Lfm3AudioPrompt.Languages.Count);
        Assert.Equal(15, Lfm3AudioPrompt.PositionsForSamples(16000));
        Assert.Equal(377, Lfm3AudioPrompt.PositionsForSamples(480000));
    }

    /// <summary>The soundtrack split of llama.cpp's mtmd helper on worked cases: whole groups at 2 fps from a 25 fps video, a
    /// short last group, a track longer than the video (trimmed) and one shorter (later groups empty).</summary>
    [Fact]
    public void SoundtrackSplitFollowsLlamaCpp()
    {
        // 25 fps, 100 source frames (4 s), sampled at 2 fps: indices 0, 12, 25, 37, 50, 62, 75, 87 -> 4 groups of 2.
        int[] indices = Enumerable.Range(0, 8).Select(i => Math.Min(99, (int)(i * 25.0 / 2))).ToArray();
        Assert.Equal(new[] { 0, 12, 25, 37, 50, 62, 75, 87 }, indices);
        var pieces = Lfm3AudioPrompt.SplitSoundtrack(indices, 25, 100, 70000, out long trimmed);
        Assert.Equal(64000, trimmed);                                    // 4 s of the 4.375 s track
        Assert.Equal(new long[] { 0, 16000, 32000, 48000 }, pieces.Select(p => p.SampleStart));
        Assert.Equal(new long[] { 16000, 32000, 48000, 64000 }, pieces.Select(p => p.SampleEnd));
        // 64000 samples -> 399 frames -> 50 rows; cuts at round(16000 / 1280) = 12 (12.5, ties to even), 25, round(37.5) = 38.
        Assert.Equal(new[] { 0, 12, 25, 38 }, pieces.Select(p => p.RowStart));
        Assert.Equal(new[] { 12, 25, 38, 50 }, pieces.Select(p => p.RowEnd));

        // Five sampled frames: the last group holds one frame and takes the rest of the track.
        pieces = Lfm3AudioPrompt.SplitSoundtrack(indices.Take(5).ToArray(), 25, 100, 64000, out trimmed);
        Assert.Equal(3, pieces.Count);
        Assert.Equal(64000, pieces[^1].SampleEnd);
        Assert.Equal(50, pieces[^1].RowEnd);

        // A track that ends after 1.5 s: groups past it have no audio and get no piece.
        pieces = Lfm3AudioPrompt.SplitSoundtrack(indices, 25, 100, 24000, out trimmed);
        Assert.Equal(24000, trimmed);
        Assert.Equal(new[] { 0, 1 }, pieces.Select(p => p.Group));
        Assert.Equal(new long[] { 0, 16000 }, pieces.Select(p => p.SampleStart));
        Assert.Equal(24000, pieces[^1].SampleEnd);
        Assert.Equal(Lfm3AudioPreprocessor.RowCountForSamples(24000), pieces[^1].RowEnd);

        Assert.Empty(Lfm3AudioPrompt.SplitSoundtrack(indices, 25, 100, 0, out _));
    }
}

/// <summary>[Fact] over the LFM3-Omni mmproj (TS_LFM3_MMPROJ) on one lane: GGML lanes run in the process that pins their
/// backend (TS_TEST_GGML_BACKEND), MLX where its native library loads.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class Lfm3AudioFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue { get; }

    public Lfm3AudioFactAttribute(Lfm3AudioEncoderTests.Lane lane)
    {
        Skip = TestGates.ModelSkip("TS_LFM3_MMPROJ");
        switch (lane)
        {
            case Lfm3AudioEncoderTests.Lane.Mlx:
                Skip ??= TestGates.MlxSkip;
                RequiresValue = "Models,Mlx";
                break;
            case Lfm3AudioEncoderTests.Lane.GgmlCpu:
                Skip ??= TestGates.GgmlPinSkip(BackendType.GgmlCpu);
                RequiresValue = "Models";
                break;
            case Lfm3AudioEncoderTests.Lane.GgmlMetal:
                Skip ??= TestGates.GgmlPinSkip(BackendType.GgmlMetal);
                RequiresValue = "Models";
                break;
            default:
                RequiresValue = "Models";
                break;
        }
    }
}
