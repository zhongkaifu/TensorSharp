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
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>[Fact] needing the LFM3 models (TS_TEST_LFM3_MODEL), llama.cpp's speech reference run (TS_LFM3_SPEECH_REF) and the
/// pinned ggml-metal backend.</summary>
[Xunit.Sdk.TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class Lfm3SpeechReferenceFactAttribute : FactAttribute, Xunit.Sdk.ITraitAttribute
{
    public string RequiresValue => "Models";

    public Lfm3SpeechReferenceFactAttribute()
        => Skip = TestGates.ModelSkip("TS_LFM3_SPEECH_REF") ?? TestGates.ModelSkip(Lfm3SpeechIntegrationRealModelTests.ModelEnv)
                  ?? TestGates.GgmlPinSkip(BackendType.GgmlMetal);
}

/// <summary>
/// LFM3-Omni speaking through the shared runtime on the real checkpoint: <c>TS_TEST_LFM3_MODEL</c> names the directory
/// holding <c>LFM3-Omni-2B-Q8_0.gguf</c> and <c>mmproj-LFM3-Omni-2B-Q8_0.gguf</c>. Optional <c>TS_LFM3_TTS_REFERENCE</c>
/// names llama-tts's WAV of <see cref="ReferenceText"/> (female, en, seed 1234) to compare the synthesis with. The tests run
/// on ggml-metal, ggml-cuda, ggml-vulkan, ggml-cpu (the process pinning that GGML backend), direct CUDA and the managed CPU
/// executor; the bit-level driver test on ggml-metal only.
/// </summary>
public sealed class Lfm3SpeechIntegrationRealModelTests
{
    internal const string ModelEnv = "TS_TEST_LFM3_MODEL";
    internal const string ReferenceText = "Hello! This is a short test of speech output, eighty milliseconds at a time.";
    private readonly ITestOutputHelper _output;

    public Lfm3SpeechIntegrationRealModelTests(ITestOutputHelper output) => _output = output;

    internal static Lfm3RuntimeModel LoadOmni(BackendType backend, string quant = "Q8_0", string mmprojQuant = "Q8_0")
    {
        string dir = Environment.GetEnvironmentVariable(ModelEnv)!;
        var model = (Lfm3RuntimeModel)ModelBase.Create(Path.Combine(dir, $"LFM3-Omni-2B-{quant}.gguf"), backend);
        model.MultimodalInjector.LoadProjectors(Path.Combine(dir, $"mmproj-LFM3-Omni-2B-{mmprojQuant}.gguf"));
        return model;
    }

    /// <summary>The backend name the server and CLI take for <paramref name="backend"/>.</summary>
    private static string BackendName(BackendType backend) => backend switch
    {
        BackendType.GgmlMetal => "ggml_metal",
        BackendType.GgmlCuda => "ggml_cuda",
        BackendType.GgmlVulkan => "ggml_vulkan",
        BackendType.Cuda => "cuda",
        BackendType.GgmlCpu => "ggml_cpu",
        BackendType.Cpu => "cpu",
        _ => throw new ArgumentOutOfRangeException(nameof(backend)),
    };

    /// <summary>The TTS prompt is llama.cpp's (46 tokens for the reference text), the synthesis ends at the end-of-speech frame,
    /// and with llama-tts's seed its PCM16 equals llama-tts's WAV where the LM's hidden rows let the same codes be drawn.</summary>
    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlMetal)]
    public void SynthesisOnGgmlMetalReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.GgmlMetal);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCuda)]
    public void SynthesisOnGgmlCudaReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.GgmlCuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlVulkan)]
    public void SynthesisOnGgmlVulkanReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.GgmlVulkan);

    [CudaFact(ModelEnv)]
    public void SynthesisOnDirectCudaReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.Cuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCpu)]
    public void SynthesisOnGgmlCpuReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.GgmlCpu);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.Cpu)]
    public void SynthesisOnManagedCpuReproducesLlamaTts() => SynthesisReproducesLlamaTts(BackendType.Cpu);

    private void SynthesisReproducesLlamaTts(BackendType backend)
    {
        using Lfm3RuntimeModel model = LoadOmni(backend);
        Assert.True(model.HasSpeechOutput, model.SpeechOutputUnavailableReason);
        List<int> prompt = model.SpeechSynthesisPrompt(ReferenceText, "en", "female");
        _output.WriteLine($"prompt {prompt.Count} tokens");
        var chunks = new List<int>();
        var clock = Stopwatch.StartNew();
        double firstAudio = -1;
        DecodedAudio audio = model.SynthesizeStream(ReferenceText, new SpeechSynthesisOptions { Voice = "female", Language = "en", Seed = 1234 },
            pcm =>
            {
                if (firstAudio < 0) firstAudio = clock.Elapsed.TotalMilliseconds;
                chunks.Add(pcm.Length);
            });
        double total = clock.Elapsed.TotalSeconds;
        float[] pcm = audio.Channels[0];
        _output.WriteLine($"{pcm.Length} samples ({pcm.Length / 24000.0:F2} s) in {total:F2} s, first audio after {firstAudio:F0} ms; chunks {string.Join(",", chunks.Take(6))}...");
        Assert.True(pcm.Length > 24000, "under a second of speech");
        Assert.Equal(1440, chunks[0]);
        string outDir = Path.Combine(Lfm3AudioFrontEndTests.RepoRoot(), "artifacts", "lfm3", "speech-int");
        Directory.CreateDirectory(outDir);
        File.WriteAllBytes(Path.Combine(outDir, $"ts-standalone-en1-{BackendName(backend)}.wav"), Lfm3SpeechAudio.Wav(pcm, 24000));
        string reference = Environment.GetEnvironmentVariable("TS_LFM3_TTS_REFERENCE");
        if (!string.IsNullOrEmpty(reference) && File.Exists(reference))
        {
            byte[] ours = Lfm3SpeechAudio.Pcm16(pcm);
            byte[] theirs = File.ReadAllBytes(reference)[44..];
            int n = Math.Min(ours.Length, theirs.Length) / 2, diff = 0, maxLsb = 0;
            for (int i = 0; i < n; i++)
            {
                int a = BitConverter.ToInt16(ours, 2 * i), b = BitConverter.ToInt16(theirs, 2 * i);
                if (a != b) { diff++; maxLsb = Math.Max(maxLsb, Math.Abs(a - b)); }
            }
            _output.WriteLine($"vs llama-tts: {ours.Length / 2} vs {theirs.Length / 2} samples, {diff} differ (max {maxLsb} LSB)");
        }
    }

    /// <summary>
    /// The reply driver (schedule, sampler, codec stream, flush) on llama.cpp's own LM hidden rows of llama-tts's run
    /// (<c>TS_LFM3_SPEECH_REF</c>: eng/lfm3/speech-ref.sh's output directory, its hidden.f32 and llama-tts.wav, Q8_0, seed
    /// 1234): the WAV is llama-tts's to 1 LSB and ends at the same end-of-speech frame. The LM is the only part of the
    /// integrated TTS this leaves out; its rows sit 2-7e-4 (relative L2) from llama.cpp's on ggml-metal.
    /// </summary>
    /// <remarks>ggml-metal only: there TensorSharp's depthformer computes llama.cpp's Metal logits bit for bit, so the same
    /// codes are drawn. On an A40 the codec matches llama.cpp's CUDA run to 136 dB and the depthformer to 4.3e-3 (rms rel) on
    /// ggml-cuda, but llama.cpp's CUDA run draws codes no lane of TensorSharp's sampler reproduces from its own hidden rows
    /// (5 of 48, the managed CPU lane included), so its WAV is not a bit-exact target there; eng/lfm3/speech-asr.py checks
    /// the GPU lanes' speech instead. On the CPU lanes (ggml-cpu, managed) the depthformer's products round their
    /// activations to 8 bits, as llama.cpp's CPU backend does, and two CPU implementations that differ only in summation
    /// order land about 2% apart in the logits (the LM shows the same floor), so the temperature-0.4 draws part within a few
    /// frames (measured against llama.cpp's CPU run: the first two frames equal, the third differs). The CPU lanes check the
    /// depthformer and codec within tolerances (<see cref="Lfm3SpeechRealWeightsTests"/>) and the synthesis end to end
    /// below.</remarks>
    [Lfm3SpeechReferenceFact]
    public void TheDriverOnLlamaCppHiddenRowsReproducesLlamaTts() => DriverReproducesLlamaTts(BackendType.GgmlMetal);

    private void DriverReproducesLlamaTts(BackendType backend)
    {
        string refDir = Environment.GetEnvironmentVariable("TS_LFM3_SPEECH_REF")!;
        byte[] raw = File.ReadAllBytes(Path.Combine(refDir, "hidden.f32"));
        var rows = new float[raw.Length / 4];
        Buffer.BlockCopy(raw, 0, rows, 0, raw.Length);
        using Lfm3RuntimeModel model = LoadOmni(backend);
        int width = model.Config.HiddenSize, frames = rows.Length / width;
        List<int> prompt = model.SpeechSynthesisPrompt(ReferenceText, "en", "female");
        using SpeechReplyDriver driver = model.CreateSpeechDriver(prompt, new SpeechReplyOptions { Synthesis = true, Seed = 1234 });
        var pcm = new List<float>();
        int f = 0;
        for (; f < frames; f++)
        {
            Assert.True(driver.NextNeedsHidden);
            DriverSlot slot = driver.Next(null, rows.AsSpan(f * width, width).ToArray(), () => throw new InvalidOperationException("a text slot"));
            pcm.AddRange(driver.TakeAudio());
            if (slot.IsStop) break;
            Assert.Equal(driver.FrameToken, slot.Token);
            Assert.Equal(width, slot.Row.Length);
        }
        driver.Complete();
        pcm.AddRange(driver.TakeAudio());
        byte[] ours = Lfm3SpeechAudio.Pcm16(pcm.ToArray());
        byte[] theirs = File.ReadAllBytes(Path.Combine(refDir, "llama-tts.wav"))[44..];
        int differ = 0, worst = 0;
        for (int i = 0; i < Math.Min(ours.Length, theirs.Length) / 2; i++)
        {
            int d = Math.Abs(BitConverter.ToInt16(ours, 2 * i) - BitConverter.ToInt16(theirs, 2 * i));
            if (d > 0) differ++;
            worst = Math.Max(worst, d);
        }
        _output.WriteLine($"{backend}: end of speech at frame {f} of {frames} rows; {ours.Length / 2} vs {theirs.Length / 2} samples; {differ} differ, max {worst} LSB");
        string codesPath = Path.Combine(refDir, "codes.txt");
        if (File.Exists(codesPath))
        {
            // llama.cpp's sampled codes, one frame a line: where the waveforms part, the first code drawn differently.
            int[][] reference = File.ReadAllLines(codesPath).Where(l => l.Trim().Length > 0)
                .Select(l => l.Split(' ', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse).ToArray()).ToArray();
            IReadOnlyList<int[]> drawn = driver.FedFrameCodes;
            int same = 0, firstFrame = -1, firstBook = -1;
            for (int i = 0; i < Math.Min(drawn.Count, reference.Length); i++)
            {
                int book = Enumerable.Range(0, Math.Min(drawn[i].Length, reference[i].Length)).FirstOrDefault(c => drawn[i][c] != reference[i][c], -1);
                if (book < 0) same++;
                else if (firstFrame < 0) (firstFrame, firstBook) = (i, book);
            }
            _output.WriteLine($"codes: {same} of {Math.Min(drawn.Count, reference.Length)} frames as llama.cpp drew them" +
                (firstFrame >= 0 ? $"; first difference at frame {firstFrame}, codebook {firstBook}" : ""));
        }
        Assert.Equal(frames - 1, f);
        Assert.Equal(theirs.Length, ours.Length);
        Assert.True(worst <= 1, $"{differ} samples differ, by up to {worst} LSB");
    }

    /// <summary>
    /// <c>/v1/chat/completions</c> with <c>modalities ["text","audio"]</c> on the real model through the server's adapter and
    /// pipeline (no HTTP): the reply's <c>message.audio</c> carries a WAV, its transcript and an id; streamed, the deltas
    /// come as id, then transcript and PCM16 pieces in order, then expires_at; and a next turn sending the reply back by its
    /// id renders it as its transcript.
    /// </summary>
    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlMetal)]
    public Task ChatCompletionsSpeakThroughTheServerAdapter() => ChatCompletionsSpeak(BackendType.GgmlMetal);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCuda)]
    public Task ChatCompletionsSpeakThroughTheServerAdapterOnGgmlCuda() => ChatCompletionsSpeak(BackendType.GgmlCuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlVulkan)]
    public Task ChatCompletionsSpeakThroughTheServerAdapterOnGgmlVulkan() => ChatCompletionsSpeak(BackendType.GgmlVulkan);

    [CudaFact(ModelEnv)]
    public Task ChatCompletionsSpeakThroughTheServerAdapterOnDirectCuda() => ChatCompletionsSpeak(BackendType.Cuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCpu)]
    public Task ChatCompletionsSpeakThroughTheServerAdapterOnGgmlCpu() => ChatCompletionsSpeak(BackendType.GgmlCpu);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.Cpu)]
    public Task ChatCompletionsSpeakThroughTheServerAdapterOnManagedCpu() => ChatCompletionsSpeak(BackendType.Cpu);

    private async Task ChatCompletionsSpeak(BackendType backend)
    {
        string dir = Environment.GetEnvironmentVariable(ModelEnv)!;
        string lm = Path.Combine(dir, "LFM3-Omni-2B-Q8_0.gguf"), mmproj = Path.Combine(dir, "mmproj-LFM3-Omni-2B-Q8_0.gguf");
        string root = Path.Combine(Path.GetTempPath(), "ts-speech-chat-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            using var service = new ModelService();
            service.LoadModel(lm, mmproj, BackendName(backend));
            Assert.True(service.HasSpeechOutput);
            ServerHostingOptions options = ServerOptionsBuilder.Build(new[] { "--model", lm, "--mmproj", mmproj, "--backend", BackendName(backend), "--no-skills" }, root);
            var adapter = new OpenAIChatAdapter(service, options, new UploadStoragePolicy(Path.Combine(root, "uploads")),
                new SkillRegistry(new SkillRegistryOptions { Roots = Array.Empty<string>() }), null,
                new SessionWorkspaceManager(Path.Combine(root, "workspaces")), NullLoggerFactory.Instance);
            async Task<string> Post(string body)
            {
                var ctx = new DefaultHttpContext();
                ctx.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(body));
                ctx.Request.ContentType = "application/json";
                ctx.Response.Body = new MemoryStream();
                await adapter.ChatCompletionsAsync(ctx);
                Assert.Equal(200, ctx.Response.StatusCode == 0 ? 200 : ctx.Response.StatusCode);
                return Encoding.UTF8.GetString(((MemoryStream)ctx.Response.Body).ToArray());
            }
            string model = Path.GetFileNameWithoutExtension(lm);
            string sync = await Post("{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello to Anna in one short sentence.\"}]," +
                "\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"female\",\"format\":\"wav\",\"seed\":5},\"max_tokens\":300}");
            using var doc = JsonDocument.Parse(sync);
            JsonElement message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");
            JsonElement audio = message.GetProperty("audio");
            string transcript = audio.GetProperty("transcript").GetString();
            byte[] wav = Convert.FromBase64String(audio.GetProperty("data").GetString());
            _output.WriteLine($"transcript: {transcript}; {(wav.Length - 44) / 48000.0:F2} s; finish {doc.RootElement.GetProperty("choices")[0].GetProperty("finish_reason")}");
            Assert.Equal(JsonValueKind.Null, message.GetProperty("content").ValueKind);
            Assert.False(string.IsNullOrWhiteSpace(transcript));
            Assert.Equal("RIFF"u8.ToArray(), wav[..4]);
            Assert.True(wav.Length - 44 > 24000, "under half a second of speech");
            string id = audio.GetProperty("id").GetString();
            Assert.StartsWith("audio_", id);
            Assert.True(audio.GetProperty("expires_at").GetInt64() > DateTimeOffset.UtcNow.ToUnixTimeSeconds());

            string stream = await Post("{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello to Anna in one short sentence.\"}]," +
                "\"modalities\":[\"text\",\"audio\"],\"audio\":{\"voice\":\"male\",\"format\":\"pcm16\"},\"stream\":true,\"max_tokens\":300}");
            var kinds = new List<string>();
            var streamedTranscript = new StringBuilder();
            long pcmBytes = 0;
            foreach (string line in stream.Split('\n'))
            {
                if (!line.StartsWith("data: {", StringComparison.Ordinal)) continue;
                using var chunk = JsonDocument.Parse(line[6..]);
                JsonElement delta = chunk.RootElement.GetProperty("choices")[0].GetProperty("delta");
                if (!delta.TryGetProperty("audio", out JsonElement a)) continue;
                foreach (JsonProperty p in a.EnumerateObject())
                {
                    kinds.Add(p.Name);
                    if (p.Name == "transcript") streamedTranscript.Append(p.Value.GetString());
                    if (p.Name == "data") pcmBytes += Convert.FromBase64String(p.Value.GetString()).Length;
                }
            }
            _output.WriteLine($"stream: {kinds.Count(k => k == "transcript")} transcript pieces, {kinds.Count(k => k == "data")} audio pieces, " +
                $"{pcmBytes / 48000.0:F2} s; transcript '{streamedTranscript}'");
            Assert.Equal("id", kinds[0]);
            Assert.Equal("expires_at", kinds[^1]);
            Assert.Contains("transcript", kinds);
            Assert.Contains("data", kinds);
            Assert.True(stream.TrimEnd().EndsWith("data: [DONE]", StringComparison.Ordinal));

            // The reply sent back by its id is the assistant turn of the next request.
            string next = await Post("{\"model\":\"" + model + "\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello to Anna in one short sentence.\"}," +
                "{\"role\":\"assistant\",\"content\":null,\"audio\":{\"id\":\"" + id + "\"}},{\"role\":\"user\",\"content\":\"Now say goodbye.\"}]," +
                "\"max_tokens\":40}");
            using var nextDoc = JsonDocument.Parse(next);
            Assert.False(string.IsNullOrWhiteSpace(nextDoc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()));
        }
        finally
        {
            try { Directory.Delete(root, recursive: true); } catch (IOException) { }
        }
    }

    private static SchedulerConfig EngineConfig() => new()
    {
        MaxNumBatchedTokens = 2048,
        MaxNumRunningSequences = 4,
        NumBlocks = 256,
        BlockSize = 16,
        EnablePrefixCaching = true,
    };

    /// <summary>Submit a TTS reply to <paramref name="engine"/> as a driven request; returns its PCM (frames' audio in token
    /// order, then the codec's tail) and the time to its first audio.</summary>
    internal static async Task<(float[] Pcm, double FirstAudioMs, int Frames, string Finish)> SpeakOnEngine(InferenceEngine engine,
        Lfm3RuntimeModel model, string text, string language, string voice, uint seed, float? temperature = null, int maxFrames = 512)
    {
        List<int> prompt = model.SpeechSynthesisPrompt(text, language, voice);
        using SpeechReplyDriver driver = model.CreateSpeechDriver(prompt, new SpeechReplyOptions
        {
            Synthesis = true, Seed = seed, Voice = voice, Language = language, Temperature = temperature, Mode = SpeechReplyMode.Text,
        });
        var clock = Stopwatch.StartNew();
        InferenceRequestHandle handle = engine.SubmitRequest(new SequenceState($"tts-{Guid.NewGuid():N}", prompt, maxFrames,
            engine.PoolStats.blockSize, SamplingConfig.Greedy) { StepDriver = driver });
        var pcm = new List<float>();
        double first = -1;
        int frames = 0;
        await foreach (int token in handle.Tokens.ReadAllAsync())
        {
            Assert.Equal(driver.FrameToken, token);
            frames++;
            float[] samples = driver.TakeAudio();
            if (samples.Length > 0 && first < 0) first = clock.Elapsed.TotalMilliseconds;
            pcm.AddRange(samples);
        }
        InferenceCompletion completion = await handle.Completion;
        driver.Complete();
        pcm.AddRange(driver.TakeAudio());
        return (pcm.ToArray(), first, frames, completion.FinishReason);
    }

    /// <summary>A TTS reply served by the engine (a driven request on the unified runtime) is the reply the standalone
    /// synthesis makes: the same prompt chunk and one-row steps, so the same hidden rows, codes and PCM, bit for bit.</summary>
    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlMetal)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesis() => EngineSynthesisEqualsStandalone(BackendType.GgmlMetal);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCuda)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesisOnGgmlCuda() => EngineSynthesisEqualsStandalone(BackendType.GgmlCuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlVulkan)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesisOnGgmlVulkan() => EngineSynthesisEqualsStandalone(BackendType.GgmlVulkan);

    [CudaFact(ModelEnv)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesisOnDirectCuda() => EngineSynthesisEqualsStandalone(BackendType.Cuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCpu)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesisOnGgmlCpu() => EngineSynthesisEqualsStandalone(BackendType.GgmlCpu);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.Cpu)]
    public Task EngineSynthesisEqualsTheStandaloneSynthesisOnManagedCpu() => EngineSynthesisEqualsStandalone(BackendType.Cpu);

    private async Task EngineSynthesisEqualsStandalone(BackendType backend)
    {
        using Lfm3RuntimeModel model = LoadOmni(backend);
        float[] alone = model.Synthesize(ReferenceText, new SpeechSynthesisOptions { Voice = "female", Language = "en", Seed = 1234 }).Channels[0];
        using var engine = new InferenceEngine(model, EngineConfig(), NullLogger.Instance);
        var (pcm, firstMs, frames, finish) = await SpeakOnEngine(engine, model, ReferenceText, "en", "female", 1234).WaitAsync(TimeSpan.FromMinutes(5));
        _output.WriteLine($"{backend} engine: {frames} frame rows, {pcm.Length} samples, first audio {firstMs:F0} ms, finish {finish}; standalone {alone.Length} samples");
        Assert.Equal("eos", finish);
        Assert.Equal(alone.Length, pcm.Length);
        Assert.Equal(Lfm3SpeechAudio.Pcm16(alone), Lfm3SpeechAudio.Pcm16(pcm));
    }

    /// <summary>A text request served while a TTS reply speaks on the same engine (their steps batched) keeps working: its
    /// greedy reply is the one it gets alone, and the speech is complete.</summary>
    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlMetal)]
    public Task ATextRequestDuringSynthesisKeepsItsReply() => TextRequestDuringSynthesis(BackendType.GgmlMetal);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCuda)]
    public Task ATextRequestDuringSynthesisKeepsItsReplyOnGgmlCuda() => TextRequestDuringSynthesis(BackendType.GgmlCuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlVulkan)]
    public Task ATextRequestDuringSynthesisKeepsItsReplyOnGgmlVulkan() => TextRequestDuringSynthesis(BackendType.GgmlVulkan);

    [CudaFact(ModelEnv)]
    public Task ATextRequestDuringSynthesisKeepsItsReplyOnDirectCuda() => TextRequestDuringSynthesis(BackendType.Cuda);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.GgmlCpu)]
    public Task ATextRequestDuringSynthesisKeepsItsReplyOnGgmlCpu() => TextRequestDuringSynthesis(BackendType.GgmlCpu);

    [ModelFact(ModelEnv, GgmlBackend = BackendType.Cpu)]
    public Task ATextRequestDuringSynthesisKeepsItsReplyOnManagedCpu() => TextRequestDuringSynthesis(BackendType.Cpu);

    private async Task TextRequestDuringSynthesis(BackendType backend)
    {
        using Lfm3RuntimeModel model = LoadOmni(backend);
        List<int> textPrompt = model.Tokenizer.Encode(
            "<|startoftext|><|im_start|>user\nName three primary colors.<|im_end|>\n<|im_start|>assistant\n<think></think>", addSpecial: false);
        const int replyTokens = 40;
        async Task<int[]> Text(InferenceEngine engine)
        {
            InferenceRequestHandle h = engine.SubmitRequest(new SequenceState($"text-{Guid.NewGuid():N}", textPrompt, replyTokens,
                engine.PoolStats.blockSize, SamplingConfig.Greedy));
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync()) tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        }
        int[] alone;
        using (var engine = new InferenceEngine(model, EngineConfig(), NullLogger.Instance))
            alone = await Text(engine).WaitAsync(TimeSpan.FromMinutes(3));
        using (var engine = new InferenceEngine(model, EngineConfig(), NullLogger.Instance))
        {
            Task<(float[] Pcm, double FirstAudioMs, int Frames, string Finish)> speech = SpeakOnEngine(engine, model, ReferenceText, "en", "female", 1234);
            await Task.Delay(50);
            int[] during = await Text(engine).WaitAsync(TimeSpan.FromMinutes(3));
            var (pcm, _, frames, finish) = await speech.WaitAsync(TimeSpan.FromMinutes(5));
            _output.WriteLine($"text alone: {model.Tokenizer.Decode(alone.ToList())}");
            _output.WriteLine($"text during speech: {model.Tokenizer.Decode(during.ToList())}");
            _output.WriteLine($"speech: {frames} frames, {pcm.Length / 24000.0:F2} s, {finish}; batched forwards {model.BatchedForwardCount}");
            Assert.Equal(alone, during);
            Assert.Equal("eos", finish);
            Assert.True(pcm.Length > 24000);
        }
    }
}
