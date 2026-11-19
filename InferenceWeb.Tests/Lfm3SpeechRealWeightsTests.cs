// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;
using static InferenceWeb.Tests.NetworkGraphTests;

namespace InferenceWeb.Tests;

/// <summary>[Fact] needing the LFM3-Omni mmproj (TS_LFM3_MMPROJ) and the MLX native backend.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class Lfm3SpeechMlxFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Mlx,Models";

    public Lfm3SpeechMlxFactAttribute() => Skip = TestGates.ModelSkip("TS_LFM3_MMPROJ") ?? TestGates.MlxSkip;
}

/// <summary>
/// LFM3-Omni speech output on the real mmproj (TS_LFM3_MMPROJ: mmproj-LFM3-Omni-2B-Q8_0.gguf or -F16.gguf) against
/// llama.cpp's own run of it (Fixtures/Lfm3Speech/llamacpp-reference-*.json, made by eng/lfm3/speech-ref.sh: llama.cpp's
/// TTS pipeline on a fixed text and seed with the LFM3-Omni-2B Q8_0 LM, its hidden rows recorded):
/// <list type="bullet">
/// <item>the depthformer's logits of every pass of the first frames (given llama.cpp's hidden rows and the codes it had
/// revealed), as statistics, top codes and every 41st logit;</item>
/// <item>the frames the sampler draws from those rows with the same seed (the same codes);</item>
/// <item>the LM feedback rows;</item>
/// <item>the codec's PCM of all of llama.cpp's frames, decoded one frame a call (streaming), as per-80 ms RMS and every 61st
/// sample.</item>
/// </list>
/// Every Mac lane: GGML metal and MLX over the whole utterance, GGML cpu and the managed CPU over its first frames.
/// </summary>
public sealed class Lfm3SpeechRealWeightsTests
{
    private const int Cb = Lfm3SpeechConfig.Codebooks, V = Lfm3SpeechConfig.CodebookSize;
    private readonly ITestOutputHelper _output;

    public Lfm3SpeechRealWeightsTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlMetal)]
    public void GgmlMetalMatchesLlamaCpp() => Compare(Lane.GgmlMetal, codecFrames: int.MaxValue);

    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlCpu)]
    public void GgmlCpuMatchesLlamaCpp() => Compare(Lane.GgmlCpu, codecFrames: 12);

    [ModelFact("TS_LFM3_MMPROJ")]
    public void ManagedCpuMatchesLlamaCpp() => Compare(Lane.Managed, codecFrames: 12);

    [Lfm3SpeechMlxFact]
    public void MlxMatchesLlamaCpp() => Compare(Lane.Mlx, codecFrames: int.MaxValue);

    /// <summary>The GPU lanes over the whole utterance, against llama.cpp's Metal run (the fixture).</summary>
    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlCuda)]
    public void GgmlCudaMatchesLlamaCpp() => Compare(Lane.GgmlCuda, codecFrames: int.MaxValue);

    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlVulkan)]
    public void GgmlVulkanMatchesLlamaCpp() => Compare(Lane.GgmlVulkan, codecFrames: int.MaxValue);

    [CudaFact("TS_LFM3_MMPROJ")]
    public void DirectCudaMatchesLlamaCpp() => Compare(Lane.Cuda, codecFrames: int.MaxValue);

    /// <summary>Per-frame time on ggml-metal (one frame = 80 ms of audio): the sampler's three depthformer passes, the
    /// feedback row and the codec's frame (graph + host ISTFT), against llama.cpp's medians of the same pieces on the same
    /// Mac (the fixture's timing_ms; llama.cpp computes every codebook's logits and the projection in each pass).</summary>
    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlMetal)]
    public void GgmlMetalSpeed() => Speed(Lane.GgmlMetal);

    [Lfm3SpeechMlxFact]
    public void MlxSpeed() => Speed(Lane.Mlx);

    private static (GgufFile Gguf, JsonElement Fixture, string Name) Open()
    {
        string path = Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ");
        var gguf = new GgufFile(path);
        // llama.cpp's run on this machine's GPU (eng/lfm3/speech-ref.sh's fixture.json, the same mmproj): the GPU lanes then
        // compare with llama.cpp on the same device and backend family instead of its Metal run.
        string local = Environment.GetEnvironmentVariable("TS_LFM3_SPEECH_FIXTURE");
        if (!string.IsNullOrEmpty(local))
            return (gguf, JsonDocument.Parse(File.ReadAllText(local)).RootElement.Clone(), Path.GetFileName(Path.GetDirectoryName(Path.GetFullPath(local))) + "/" + Path.GetFileName(local));
        GgmlTensorType type = gguf.Tensors["a.gen.code.proj_in.weight"].Type;
        string name = type switch
        {
            GgmlTensorType.Q8_0 => "llamacpp-reference-q8_0.json",
            GgmlTensorType.F16 => "llamacpp-reference-f16.json",
            _ => throw new InvalidOperationException($"no llama.cpp reference for a {type} mmproj ({path})"),
        };
        return (gguf, Lfm3SpeechOutputTests.Fixture(name), name);
    }

    private static float[] Hidden(JsonElement fixture, out int frames)
    {
        frames = fixture.GetProperty("hidden_frames").GetInt32();
        byte[] bytes = Convert.FromBase64String(fixture.GetProperty("hidden_f32_b64").GetString());
        var rows = new float[bytes.Length / 4];
        Buffer.BlockCopy(bytes, 0, rows, 0, bytes.Length);
        return rows;
    }

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    private static float[] Floats(JsonElement e) => e.EnumerateArray().Select(x => x.GetSingle()).ToArray();

    private void Compare(Lane laneKind, int codecFrames)
    {
        (GgufFile gguf, JsonElement fixture, string fixtureName) = Open();
        using (gguf)
        using (var lane = new LaneContext(laneKind))
        using (Lfm3SpeechOutput speech = Lfm3SpeechOutput.Load(gguf, lane.Allocator, lane.Backend))
        {
            int hiddenSize = speech.Config.HiddenSize;
            float[] hidden = Hidden(fixture, out int hiddenFrames);
            int[] allCodes = Ints(fixture.GetProperty("codes"));
            string tag = $"{laneKind} ({System.IO.Path.GetFileName(Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ"))} vs {fixtureName})";

            // ---- the depthformer, pass by pass, given llama.cpp's hidden rows and revealed codes ----
            double worstSample = 0, worstLse = 0, worstRms = 0;
            int top1 = 0, logitRows = 0;
            var logits = new float[4 * V];
            JsonElement passes = fixture.GetProperty("passes");
            for (int f = 0; f < hiddenFrames; f++)
            {
                JsonElement frame = passes[f];
                for (int p = 0; p < 3; p++)
                {
                    JsonElement pass = frame.GetProperty("pass")[p];
                    int[] known = Ints(pass.GetProperty("known"));
                    JsonElement[] stats = pass.GetProperty("logits").EnumerateArray().ToArray();
                    int[] pick = stats.Select(s => s.GetProperty("cb").GetInt32()).ToArray();
                    speech.Depthformer.Pass(p == 0 ? hidden.AsSpan(f * hiddenSize, hiddenSize) : ReadOnlySpan<float>.Empty, known, pick, logits);
                    for (int j = 0; j < pick.Length; j++)
                    {
                        ReadOnlySpan<float> l = logits.AsSpan(j * V, V);
                        JsonElement s = stats[j];
                        int stride = s.GetProperty("stride").GetInt32();
                        float[] sample = Floats(s.GetProperty("sample"));
                        double scale = sample.Max(v => Math.Abs((double)v));
                        for (int i = 0; i < sample.Length; i++) worstSample = Math.Max(worstSample, Math.Abs(l[i * stride] - sample[i]) / scale);
                        double max = double.NegativeInfinity, sum = 0, ss = 0;
                        int arg = 0;
                        for (int v = 0; v < V; v++)
                        {
                            if (l[v] > max) { max = l[v]; arg = v; }
                            ss += (double)l[v] * l[v];
                        }
                        for (int v = 0; v < V; v++) sum += Math.Exp(l[v] - max);
                        worstLse = Math.Max(worstLse, Math.Abs(max + Math.Log(sum) - s.GetProperty("lse").GetDouble()));
                        worstRms = Math.Max(worstRms, Math.Abs(Math.Sqrt(ss / V) / s.GetProperty("rms").GetDouble() - 1));
                        top1 += arg == s.GetProperty("top")[0][0].GetInt32() ? 1 : 0;
                        logitRows++;
                    }
                }
            }

            // ---- the sampler: llama.cpp's seed over its hidden rows gives its codes ----
            Lfm3FrameSampler sampler = speech.CreateSampler(fixture.GetProperty("seed").GetUInt32());
            int codesEqual = 0;
            for (int f = 0; f < hiddenFrames; f++)
            {
                Lfm3SpeechFrame frame = sampler.Sample(hidden.AsSpan(f * hiddenSize, hiddenSize));
                for (int c = 0; c < Cb; c++) codesEqual += frame.Codes[c] == allCodes[f * Cb + c] ? 1 : 0;
            }

            // ---- the feedback rows ----
            double worstFeedback = 0;
            JsonElement feedback = fixture.GetProperty("feedback");
            for (int f = 0; f < feedback.GetArrayLength(); f++)
            {
                float[] row = speech.FeedbackRow(allCodes.AsSpan(f * Cb, Cb));
                float[] first = Floats(feedback[f].GetProperty("first"));
                for (int i = 0; i < first.Length; i++) worstFeedback = Math.Max(worstFeedback, Math.Abs(row[i] - first[i]));
                double sum = row.Sum(v => (double)v);
                worstFeedback = Math.Max(worstFeedback, Math.Abs(sum - feedback[f].GetProperty("sum").GetDouble()) / hiddenSize);
            }

            // ---- the codec: llama.cpp's frames, one a call ----
            int audioFrames = fixture.GetProperty("audio_frames").GetInt32();
            int frames = Math.Min(codecFrames, audioFrames);
            var pcm = new List<float>();
            var sw = Stopwatch.StartNew();
            using (Lfm3CodecStream stream = speech.CreateCodecStream())
            {
                for (int f = 0; f < frames; f++) stream.Decode(allCodes.AsSpan(f * Cb, Cb), pcm);
                if (frames == audioFrames) stream.Flush(pcm);
            }
            double codecMs = sw.Elapsed.TotalMilliseconds;
            JsonElement reference = fixture.GetProperty("pcm");
            if (frames == audioFrames) Assert.Equal(reference.GetProperty("samples").GetInt32(), pcm.Count);
            else Assert.Equal(frames * speech.Config.SamplesPerFrame - 480, pcm.Count);
            int pcmStride = reference.GetProperty("stride").GetInt32();
            float[] refSample = Floats(reference.GetProperty("sample"));
            float maxAbs = reference.GetProperty("max_abs").GetSingle();
            double err2 = 0, ref2 = 0, worstPcm = 0;
            int compared = 0;
            for (int i = 0; i < refSample.Length && i * pcmStride < pcm.Count; i++, compared++)
            {
                double d = pcm[i * pcmStride] - refSample[i];
                err2 += d * d;
                ref2 += (double)refSample[i] * refSample[i];
                worstPcm = Math.Max(worstPcm, Math.Abs(d) / maxAbs);
            }
            double snr = 10 * Math.Log10(ref2 / Math.Max(err2, 1e-30));
            float[] frameRms = Floats(reference.GetProperty("frame_rms"));
            double worstFrameRms = 0;
            for (int f = 0; (f + 1) * 1920 <= pcm.Count && f < frameRms.Length; f++)
            {
                double ss = 0;
                for (int i = f * 1920; i < (f + 1) * 1920; i++) ss += (double)pcm[i] * pcm[i];
                worstFrameRms = Math.Max(worstFrameRms, Math.Abs(Math.Sqrt(ss / 1920) - frameRms[f]) / Math.Max(frameRms[f], 1e-3f));
            }

            string line = $"{tag}: depthformer over {logitRows} codebook passes: max |d| / max|logit| {worstSample:G3}, logsumexp {worstLse:G3}, " +
                $"rms rel {worstRms:G3}, argmax {top1}/{logitRows}; sampled codes {codesEqual}/{hiddenFrames * Cb} equal; feedback max |d| {worstFeedback:G3}; " +
                $"codec {frames} frames ({codecMs:F0} ms): every {pcmStride}th sample max |d| / max|pcm| {worstPcm:G3}, SNR {snr:F1} dB over {compared}, " +
                $"80 ms RMS max rel {worstFrameRms:G3}; depthformer builds {speech.Depthformer.Graph.Builds}, replays {speech.Depthformer.Graph.Replays}";
            _output.WriteLine(line);
            Console.WriteLine(line);

            // Tolerances (observed on the M5 Pro in brackets). ggml-metal runs llama.cpp's kernels on the same graph: its logits
            // equal llama.cpp's Metal run bit for bit [0], its PCM within float rounding [~122 dB]. MLX rounds products through
            // F16 operands [1-2e-3, 53-60 dB]. The CPU lanes match an F16 mmproj closely [4e-4, 113 dB] and quantize the
            // activations of Q8_0 products to 8 bits (as ggml-cpu does) [2e-2, 33 dB] where llama.cpp's reference ran on Metal.
            // A40, against the Metal fixture: ggml-vulkan as MLX [4.7e-3, 0.039]; ggml-cuda and direct CUDA quantize Q8_0
            // products' activations (q8_1) as the CPU lanes do [2.0-2.5e-2, lse 0.13-0.18, 30.6-31.2 dB, codes 35-37 of 48].
            bool q8 = gguf.Tensors["a.gen.code.proj_in.weight"].Type == GgmlTensorType.Q8_0;
            bool q8Activations = laneKind is Lane.Managed or Lane.GgmlCpu or Lane.GgmlCuda or Lane.Cuda;
            (double logit, double lse, double minSnr) = laneKind switch
            {
                Lane.GgmlMetal => (2e-3, 0.02, 60.0),
                Lane.Mlx or Lane.GgmlVulkan => (5e-3, 0.05, 40.0),
                _ when q8Activations && q8 => (8e-2, 0.3, 25.0),
                Lane.GgmlCuda or Lane.Cuda => (5e-3, 0.05, 40.0),
                _ => (2e-3, 0.02, 60.0),
            };
            bool cpu = q8Activations;
            Assert.True(worstSample <= logit, line);
            Assert.True(worstLse <= lse, line);
            Assert.True(top1 >= logitRows - (cpu && q8 ? 4 : 1), line);
            // A code drawn near a boundary of the cumulative mass flips with a logit's last digits, and a flipped code changes the
            // later passes of its frame and the penalties of the next frames: the draws themselves are checked exactly by
            // Lfm3SpeechOutputTests (llama.cpp's random stream, its sampler over this lane's logits).
            Assert.True(codesEqual >= hiddenFrames * Cb * (cpu && q8 ? 0.5 : 0.75), line);
            Assert.True(snr >= minSnr, line);
            Assert.True(worstFeedback <= 1e-4, line);
        }
    }

    private void Speed(Lane laneKind)
    {
        (GgufFile gguf, JsonElement fixture, string fixtureName) = Open();
        using (gguf)
        using (var lane = new LaneContext(laneKind))
        using (Lfm3SpeechOutput speech = Lfm3SpeechOutput.Load(gguf, lane.Allocator, lane.Backend))
        {
            int hiddenSize = speech.Config.HiddenSize;
            float[] hidden = Hidden(fixture, out int hiddenFrames);
            int[] allCodes = Ints(fixture.GetProperty("codes"));
            int audioFrames = fixture.GetProperty("audio_frames").GetInt32();
            Lfm3FrameSampler sampler = speech.CreateSampler(fixture.GetProperty("seed").GetUInt32());
            const int Reps = 40;
            var sample = new double[Reps];
            var feedback = new double[Reps];
            var codec = new double[Reps];
            var sw = new Stopwatch();
            using Lfm3CodecStream stream = speech.CreateCodecStream();
            var pcm = new List<float>();
            for (int r = -4; r < Reps; r++)
            {
                int f = (r + 4) % hiddenFrames;
                sw.Restart();
                Lfm3SpeechFrame frame = sampler.Sample(hidden.AsSpan(f * hiddenSize, hiddenSize));
                double t0 = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                speech.FeedbackRow(frame.IsEndOfAudio ? allCodes.AsSpan(0, Cb) : frame.Codes);
                double t1 = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                stream.Decode(allCodes.AsSpan(((r + 4) % audioFrames) * Cb, Cb), pcm);
                double t2 = sw.Elapsed.TotalMilliseconds;
                if (r < 0) continue;   // warm-up: the programs build
                sample[r] = t0;
                feedback[r] = t1;
                codec[r] = t2;
            }
            // The head's choice: logits of the 4 codebooks a pass samples (one MulMatId), against all 12 (llama.cpp's).
            var four = new double[Reps];
            var twelve = new double[Reps];
            var l = new float[Cb * V];
            int[] codes = Enumerable.Repeat(-1, Cb).ToArray();
            int[] pick4 = { 0, 5, 7, 11 }, all = Enumerable.Range(0, Cb).ToArray();
            speech.Depthformer.Pass(hidden.AsSpan(0, hiddenSize), codes, pick4, l);
            speech.Depthformer.Pass(ReadOnlySpan<float>.Empty, codes, all, l);
            for (int r = 0; r < Reps; r++)
            {
                sw.Restart();
                speech.Depthformer.Pass(ReadOnlySpan<float>.Empty, codes, pick4, l);
                four[r] = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                speech.Depthformer.Pass(ReadOnlySpan<float>.Empty, codes, all, l);
                twelve[r] = sw.Elapsed.TotalMilliseconds;
            }
            static double Median(double[] v) => v.OrderBy(x => x).ElementAt(v.Length / 2);
            JsonElement t = fixture.GetProperty("timing_ms");
            double llamaFrame = 3 * t.GetProperty("depthformer_pass").GetDouble() + t.GetProperty("feedback").GetDouble() + t.GetProperty("codec_frame").GetDouble();
            double ours = Median(sample) + Median(feedback) + Median(codec);
            string line = $"lfm3 speech per 80 ms frame on {laneKind} ({fixtureName}): sampler (3 depthformer passes + MaskGIT) {Median(sample):F2} ms, " +
                $"feedback row {Median(feedback):F3} ms, codec frame {Median(codec):F2} ms -> {ours:F2} ms ({80 / ours:F1}x real time); " +
                $"llama.cpp (same Mac, fixture) {3 * t.GetProperty("depthformer_pass").GetDouble():F2} + {t.GetProperty("feedback").GetDouble():F3} + " +
                $"{t.GetProperty("codec_frame").GetDouble():F2} = {llamaFrame:F2} ms; a later pass with the head over 4 codebooks {Median(four):F2} ms, over 12 " +
                $"{Median(twelve):F2} ms; depthformer builds {speech.Depthformer.Graph.Builds}, codec builds {stream.Graph.Builds}";
            _output.WriteLine(line);
            Console.WriteLine(line);
            // ggml-metal: within half the real-time budget even on a loaded machine (3.5-6.6 ms observed). MLX: real time; its
            // runs are not replayed and each crosses the MLX worker thread, a fixed cost that grows with the machine's load
            // (5.6-13.5 ms a frame at load average 6-75, 39-57 ms at ~100).
            Assert.True(ours < (laneKind == Lane.Mlx ? 80 : 40), line);
            if (speech.Depthformer.Graph.Executor.Capabilities.Replays) Assert.True(stream.Graph.Builds <= 2, line);
        }
    }
}
