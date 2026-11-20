// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Reuse past an image on Muse-Glimmer's unified-runtime family (MuseGlimmerRuntimeModel, stage S4 of
// docs/design/muse-glimmer-unified.md), which declares reuse across media spans where legacy reused nothing past the
// first image. Synthetic weights (MuseGlimmerSyntheticModelBuilder with the image tokens, the vision tower of
// MuseGlimmerVisionSynthetic at the model's width), as Qwen35ImageFollowUpExactnessTests does with real ones:
//  A. The engine with the prefix cache: turn 2, a follow-up about turn 1's image, reuses the KV through the image and
//     the whole previous turn, and its greedy tokens are a cold engine's; a conversation with another image reuses at
//     most the text before that image's span and answers as a cold engine does. Two lengths: turn 1 inside the
//     runtime's 768-cell rings (window 64 plus a 512-row step, aligned to 256), and past them, where the rings have
//     wrapped before the image. Where a backend's prompt chunks round apart from a whole prompt's, a stream may leave the
//     cold one only at a near-tie of the cold run (NearTie, MuseGlimmerRuntimeServingTests' rule).
//  B. The model directly, where logits are observable: a turn built by prefilling turn 1 (its image rows queued),
//     DECODING the reply and prefilling the follow-up, against a cold prefill of the whole turn 2 with the same rows -
//     the logits of the first token and of each following greedy step. A text-only control (the same tokens, no image
//     rows) measures what the decode and prompt kernels differ by with no image involved; reuse past the image is exact
//     when it adds nothing to that.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerImageFollowUpExactnessTests(ITestOutputHelper output) : IDisposable
{
    /// <summary>How far the image conversation's worst logit difference may exceed the text control's (Qwen 3.5's bound).</summary>
    private const double ControlRatio = 4.0;

    /// <summary>The floor under that: differences this small relative to the largest logit are rounding, whatever the
    /// control measured (an executor whose decode and prompt kernels agree to the bit leaves a control of 0).</summary>
    private const double RoundingFloor = 1e-5;

    private const int NewTokens = 12;

    /// <summary>
    /// The relative top-two margin (a share of the largest logit) at or under which a cold run's greedy step is a near-tie that
    /// a reused stream may resolve the other way, on every backend but the managed one (whose prompt chunks are a whole
    /// prompt's bits): MuseGlimmerRuntimeServingTests' rule for streams whose prompts ran in other chunks. ggml_cuda's prompt
    /// products and attention pick their kernels by the chunk's rows, so a prompt resumed after 288 reused tokens rounds apart
    /// from a whole one. Measured on an RTX PRO 4500 Blackwell (model-direct, the "within the rings" prompt): 1.6e-3 of the
    /// largest logit with the image's rows in the resumed chunk, 1.9e-3 with text there instead, 0 on the managed backend;
    /// ggml_vulkan 2.1e-3 and 1.9e-3. The other image's first greedy step is a 1.0e-3 tie (tokens 243 and 76) on this
    /// fixture, which ggml_cuda's resumed chunk resolves the other way.
    /// </summary>
    private const double NearTie = 1e-2;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-mg-followup-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    private (string Model, string Mmproj) Files()
    {
        Directory.CreateDirectory(_dir);
        string model = Path.Combine(_dir, "glimmer-synthetic.gguf");
        if (!File.Exists(model))
        {
            new MuseGlimmerSyntheticModelBuilder
            {
                BlockCount = 8, SlidingWindow = 64, LogitScale = 0.19612f, Softcap = 20f, HeadScale = 3f, BlockOutputScale = 0.004f,
                ImageTokens = true,
            }.Write(model);
            MuseGlimmerVisionSynthetic.Write(Path.Combine(_dir, "mmproj-synthetic.gguf"), projection: MuseGlimmerSyntheticModelBuilder.Hidden);
        }
        return (model, Path.Combine(_dir, "mmproj-synthetic.gguf"));
    }

    private StandardDecoderModel Load(BackendType backend)
    {
        var (model, mmproj) = Files();
        var m = Assert.IsAssignableFrom<StandardDecoderModel>(MuseGlimmerArchitecture.CreateRuntimeModel(model, backend));
        ((IVisionCapableModel)m).LoadVisionEncoder(mmproj);
        Assert.True(((IVisionCapableModel)m).IsVisionEncoderLoaded);
        return m;
    }

    /// <summary>A 168 x 112 picture: a 6 x 4 merged grid (24 rows), 12 x 8 patches in six windows of the synthetic tower.</summary>
    private string Picture(string name, int seed)
    {
        const int width = 168, height = 112;
        var rgb = new byte[width * height * 3];
        var rng = new Random(seed);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                int p = (y * width + x) * 3;
                rgb[p] = (byte)(x * 255 / width);
                rgb[p + 1] = (byte)(y * 255 / height);
                rgb[p + 2] = (byte)((seed * 70) & 255);
            }
        for (int b = 0; b < 4; b++)
        {
            int bx = rng.Next(width - 30), by = rng.Next(height - 30);
            byte r = (byte)rng.Next(256), g = (byte)rng.Next(256), bl = (byte)rng.Next(256);
            for (int y = by; y < by + 30; y++)
                for (int x = bx; x < bx + 30; x++)
                {
                    int p = (y * width + x) * 3;
                    (rgb[p], rgb[p + 1], rgb[p + 2]) = (r, g, bl);
                }
        }
        string path = Path.Combine(_dir, name);
        using var image = new ImageMagick.MagickImage(rgb, new ImageMagick.PixelReadSettings((uint)width, (uint)height, ImageMagick.StorageType.Char, "RGB"));
        image.Write(path, ImageMagick.MagickFormat.Png);
        return path;
    }

    /// <summary>The system text of the two lengths: turn 1 within the rings (and longer than a 256-token cache block), or
    /// past them.</summary>
    private static string System(bool longTurn)
    {
        string unit = "You are a careful assistant who describes pictures plainly and answers in short sentences. ";
        return string.Concat(Enumerable.Repeat(unit, longTurn ? 10 : 3));
    }

    private static List<int> Text(ModelBase model, string s) => model.Tokenizer.Encode(s, addSpecial: false);

    private static (List<int> Tokens, List<ChatMessage> History) Turn1(ModelBase model, bool longTurn, string image)
    {
        var t = Text(model, "system: " + System(longTurn) + "\nuser: ");
        t.Add(model.Tokenizer.LookupToken("<|patch|>"));
        t.AddRange(Text(model, " Describe the picture.\nassistant:"));
        return (t, new List<ChatMessage> { new() { Role = "user", Content = "Describe the picture.", ImagePaths = new List<string> { image } } });
    }

    private static List<int> Turn2(ModelBase model, List<int> previous, List<int> reply)
    {
        var t = new List<int>(previous);
        t.AddRange(reply);
        t.AddRange(Text(model, "\nuser: Which colour covers most of it?\nassistant:"));
        return t;
    }

    private static (List<int> Tokens, IReadOnlyList<PromptMediaSpan> Spans) Prepare(ModelBase model, List<ChatMessage> history,
        List<int> tokens, string requestId)
    {
        lock (model.GpuComputeLock)
        {
            List<int> expanded = model.MultimodalInjector.ProcessPromptTokens(history, new List<int>(tokens), requestId);
            return (expanded, model.MultimodalInjector.GetPreparedMediaSpans(requestId));
        }
    }

    // ================================================================ A. the engine

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnTheManagedBackend(bool longTurn) => EngineReuse(BackendType.Cpu, longTurn);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnGgmlCpu(bool longTurn) => EngineReuse(BackendType.GgmlCpu, longTurn);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnGgmlMetal(bool longTurn) => EngineReuse(BackendType.GgmlMetal, longTurn);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnGgmlCuda(bool longTurn) => EngineReuse(BackendType.GgmlCuda, longTurn);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnGgmlVulkan(bool longTurn) => EngineReuse(BackendType.GgmlVulkan, longTurn);

    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnMlx(bool longTurn) => EngineReuse(BackendType.Mlx, longTurn);

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public Task TheEngineReusesPastTheImageOnDirectCuda(bool longTurn) => EngineReuse(BackendType.Cuda, longTurn);

    private async Task EngineReuse(BackendType backend, bool longTurn)
    {
        using StandardDecoderModel model = Load(backend);
        string imageA = Picture("a.png", 21), imageB = Picture("b.png", 22);
        var config = new SchedulerConfig
        {
            MaxNumBatchedTokens = 4096, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 1024, SoloPrefillChunkSize = 1024,
            NumBlocks = 64, BlockSize = 256, EnablePrefixCaching = true, DecodeQuantumTokens = 256,
        };

        async Task<(int Reused, List<int> Output, int ImageStart, int ImageEnd, List<int> Prompt)> Generate(InferenceEngine engine,
            List<int> tokens, List<ChatMessage> history, string id)
        {
            var (expanded, spans) = Prepare(model, history, tokens, id);
            try
            {
                var seq = new SequenceState(id, new List<int>(expanded), NewTokens, 256, SamplingConfig.Greedy, mediaSpans: spans, cacheScope: "mg-followup");
                var handle = engine.SubmitRequest(seq);
                var produced = new List<int>();
                await foreach (int t in handle.Tokens.ReadAllAsync())
                    produced.Add(t);
                var completion = await handle.Completion;
                PromptMediaSpan image = Assert.Single(spans);
                return (completion.PrefixCacheReusedTokens, produced, image.Start, image.End, expanded);
            }
            finally
            {
                model.MultimodalInjector.ClearPreparedPromptState(id);
            }
        }

        var a1 = Turn1(model, longTurn, imageA);
        var b1 = Turn1(model, longTurn, imageB);
        List<int> outA1, outA2, outB1, promptA2, promptB1;
        int reusedA2, reusedB1, imageStart, imageEnd, promptA1;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
        {
            var r1 = await Generate(engine, a1.Tokens, a1.History, "a1");
            (outA1, imageStart, imageEnd, promptA1) = (r1.Output, r1.ImageStart, r1.ImageEnd, r1.Prompt.Count);
            var r2 = await Generate(engine, Turn2(model, a1.Tokens, outA1), a1.History, "a2");
            (reusedA2, outA2, promptA2) = (r2.Reused, r2.Output, r2.Prompt);
            var r3 = await Generate(engine, b1.Tokens, b1.History, "b1");
            (reusedB1, outB1, promptB1) = (r3.Reused, r3.Output, r3.Prompt);
        }
        output.WriteLine($"[{backend}, {(longTurn ? "past the rings" : "within the rings")}] image span [{imageStart}, {imageEnd}); " +
            $"turn 1: {promptA1} prompt + {outA1.Count} generated; turn 2 reused {reusedA2}; another image reused {reusedB1}");

        List<int> coldA2, coldB1;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
            coldA2 = (await Generate(engine, Turn2(model, a1.Tokens, outA1), a1.History, "cold-a2")).Output;
        using (var engine = new InferenceEngine(model, config, NullLogger.Instance))
            coldB1 = (await Generate(engine, b1.Tokens, b1.History, "cold-b1")).Output;
        output.WriteLine($"  A2 {string.Join(",", outA2)} | cold {string.Join(",", coldA2)}");
        output.WriteLine($"  B1 {string.Join(",", outB1)} | cold {string.Join(",", coldB1)}");

        Assert.Equal(24, imageEnd - imageStart - 2);
        Assert.True(!longTurn || imageStart > 768, $"the long turn's image starts at {imageStart}, inside the 768-cell rings");
        Assert.True(reusedA2 >= imageEnd, $"the follow-up reused {reusedA2} tokens, stopping before the image ends at {imageEnd}");
        Assert.True(reusedA2 >= promptA1 + outA1.Count - 2,
            $"the follow-up reused {reusedA2} of the {promptA1 + outA1.Count - 1} tokens turn 1 left");
        Assert.True(reusedB1 <= imageStart, $"another image reused {reusedB1} tokens, past its span's start {imageStart}");
        AssertSameStream(backend, coldA2, outA2, promptA2, imageStart + 1, imageA, "the follow-up");
        AssertSameStream(backend, coldB1, outB1, promptB1, imageStart + 1, imageB, "another image");
    }

    /// <summary>The reused stream is the cold engine's, or (on every backend but the managed one) leaves it only where the
    /// cold run's greedy step is a near-tie (<see cref="NearTie"/>), read on a fresh model from one forward of the whole
    /// prompt with the image's rows.</summary>
    private void AssertSameStream(BackendType backend, List<int> cold, List<int> reused, List<int> prompt, int rowsAt, string image, string what)
    {
        int at = Enumerable.Range(0, Math.Min(cold.Count, reused.Count)).FirstOrDefault(t => cold[t] != reused[t], -1);
        if (at < 0 && cold.Count == reused.Count) return;
        Assert.True(at >= 0 && backend != BackendType.Cpu,
            $"{what}: the stream {string.Join(",", reused)} differs from a cold engine's {string.Join(",", cold)}");
        double margin = ColdMargin(backend, prompt, rowsAt, image, cold, at);
        output.WriteLine($"  {what}: leaves the cold stream at token {at} ({reused[at]} for {cold[at]}), where the cold run's top-two " +
            $"margin is {margin:E2} of the largest logit (near-tie bound {NearTie:E0})");
        Assert.True(margin <= NearTie, $"{what}: the stream leaves a cold engine's at token {at} ({reused[at]} for {cold[at]}), " +
            $"where the cold run's top-two margin is {margin:E2} of the largest logit, past the near-tie bound {NearTie:E0}");
    }

    /// <summary>The relative top-two margin of the cold run's greedy step <paramref name="at"/>: a fresh model's forward of the
    /// whole <paramref name="prompt"/> with the image's rows at <paramref name="rowsAt"/>, then a token of
    /// <paramref name="cold"/> a step.</summary>
    private double ColdMargin(BackendType backend, List<int> prompt, int rowsAt, string image, List<int> cold, int at)
    {
        using StandardDecoderModel model = Load(backend);
        var tower = ((MuseGlimmerRuntimeModel)model).VisionTower;
        var px = tower.ImageProcessor.ProcessImage(image);
        float[] rows = tower.EncodeToHost(px.pixels, px.width, px.height, out int count);
        var t = new Tensor(tower.Weights.Allocator, DType.Float32, count, model.Config.HiddenSize);
        t.SetElementsAsFloat(rows);
        model.ResetKVCache();
        ((IVisionCapableModel)model).SetVisionEmbeddings(t, rowsAt);
        float[] logits = model.Forward(prompt.ToArray());
        for (int i = 0; i < at; i++) logits = model.Forward(new[] { cold[i] });
        float first = float.NegativeInfinity, second = float.NegativeInfinity, scale = 0;
        for (int v = 0; v < model.Config.VocabSize; v++)
        {
            scale = Math.Max(scale, Math.Abs(logits[v]));
            if (logits[v] > first) { second = first; first = logits[v]; }
            else if (logits[v] > second) second = logits[v];
        }
        return (first - second) / Math.Max(scale, 1e-30f);
    }

    // ================================================================ B. the model, logits

    public static TheoryData<bool, string> LengthsAndPaths()
    {
        var data = new TheoryData<bool, string>();
        foreach (bool l in new[] { false, true })
            foreach (string p in new[] { "graphs", "per-op" })
                data.Add(l, p);
        return data;
    }

    [Theory]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnTheManagedBackend(bool longTurn, string path) => Direct(BackendType.Cpu, longTurn, path);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnGgmlCpu(bool longTurn, string path) => Direct(BackendType.GgmlCpu, longTurn, path);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnGgmlMetal(bool longTurn, string path) => Direct(BackendType.GgmlMetal, longTurn, path);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnGgmlCuda(bool longTurn, string path) => Direct(BackendType.GgmlCuda, longTurn, path);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnGgmlVulkan(bool longTurn, string path) => Direct(BackendType.GgmlVulkan, longTurn, path);

    [MlxTheory]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnMlx(bool longTurn, string path) => Direct(BackendType.Mlx, longTurn, path);

    [CudaTheory]
    [MemberData(nameof(LengthsAndPaths))]
    public void AFollowUpAfterAnImageIsAColdPrefillsOnDirectCuda(bool longTurn, string path) => Direct(BackendType.Cuda, longTurn, path);

    private void Direct(BackendType backend, bool longTurn, string path)
    {
        using StandardDecoderModel model = Load(backend);
        model.GraphLoweringEnabled = path == "graphs";
        var tower = ((MuseGlimmerRuntimeModel)model).VisionTower;
        var (turn1, history) = Turn1(model, longTurn, Picture("a.png", 21));
        var (expanded1, spans) = Prepare(model, history, turn1, "direct");
        model.MultimodalInjector.ClearPreparedPromptState("direct");
        PromptMediaSpan span = Assert.Single(spans);
        int rowsAt = span.Start + 1, rowCount = span.End - span.Start - 2;
        var px = tower.ImageProcessor.ProcessImage(history[0].ImagePaths[0]);
        float[] rows = tower.EncodeToHost(px.pixels, px.width, px.height, out int tokens);
        Assert.Equal(rowCount, tokens);
        int vocab = model.Config.VocabSize;

        Tensor Rows()
        {
            var t = new Tensor(tower.Weights.Allocator, DType.Float32, tokens, model.Config.HiddenSize);
            t.SetElementsAsFloat(rows);
            return t;
        }

        // Turn 1 and its reply, decoded.
        model.ResetKVCache();
        ((IVisionCapableModel)model).SetVisionEmbeddings(Rows(), rowsAt);
        float[] logits = model.Forward(expanded1.ToArray());
        var reply = new List<int>();
        for (int i = 0; i < NewTokens; i++)
        {
            int next = MuseGlimmerParity.ArgMax(logits, vocab);
            reply.Add(next);
            logits = model.Forward(new[] { next });
        }
        List<int> turn2 = Turn2(model, expanded1, reply);
        int[] suffix = turn2.Skip(expanded1.Count + reply.Count).ToArray();

        // Reused: the follow-up prefilled after the decoded reply; cold: the whole turn 2 at once. Each with and without the
        // image rows (the text control), each then decoded greedily in lockstep (the reused stream's tokens fed to both).
        (double Worst, bool SameTokens) Compare(bool image)
        {
            model.ResetKVCache();
            if (image) ((IVisionCapableModel)model).SetVisionEmbeddings(Rows(), rowsAt);
            model.Forward(expanded1.ToArray());
            foreach (int t in reply) model.Forward(new[] { t });
            var reused = new List<float[]> { (float[])model.Forward(suffix).Clone() };
            var tokensOut = new List<int>();
            for (int i = 0; i < NewTokens; i++)
            {
                int next = MuseGlimmerParity.ArgMax(reused[^1], vocab);
                tokensOut.Add(next);
                reused.Add((float[])model.Forward(new[] { next }).Clone());
            }
            model.ResetKVCache();
            if (image) ((IVisionCapableModel)model).SetVisionEmbeddings(Rows(), rowsAt);
            var cold = new List<float[]> { (float[])model.Forward(turn2.ToArray()).Clone() };
            bool same = true;
            for (int i = 0; i < NewTokens; i++)
            {
                same &= MuseGlimmerParity.ArgMax(cold[^1], vocab) == tokensOut[i];
                cold.Add((float[])model.Forward(new[] { tokensOut[i] }).Clone());
            }
            double worst = 0;
            for (int s = 0; s < reused.Count; s++)
            {
                double d = 0, m = 0;
                for (int v = 0; v < vocab; v++)
                {
                    d = Math.Max(d, Math.Abs((double)reused[s][v] - cold[s][v]));
                    m = Math.Max(m, Math.Abs(cold[s][v]));
                }
                worst = Math.Max(worst, d / m);
            }
            return (worst, same);
        }

        var withImage = Compare(image: true);
        var control = Compare(image: false);
        double allowed = Math.Max(ControlRatio * control.Worst, RoundingFloor);
        output.WriteLine($"[{backend} {path}, {(longTurn ? "past the rings" : "within the rings")}] image at [{span.Start}, {span.End}), " +
            $"turn 2 {turn2.Count} tokens: reused vs cold worst |dlogit| / max |logit| {withImage.Worst:E2} (text control {control.Worst:E2}, " +
            $"allowed {allowed:E2}); greedy tokens equal: image {withImage.SameTokens}, control {control.SameTokens}");
        Assert.True(withImage.Worst <= allowed, $"reuse past the image is {withImage.Worst:E3} from a cold prefill (allowed {allowed:E3})");
        Assert.True(withImage.SameTokens, "the reused and the cold streams' greedy tokens differ");
    }
}
