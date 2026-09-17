// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Is an image (or audio) turn prefilled after a reused prefix the same computation
// as a cold prefill of the whole prompt?
//
// The conversation is [user text, assistant answer, user: "describe this" + media].
// Cold: one forward over the whole expanded prompt from position 0. Reuse: a
// forward over the text the previous turn left in the cache, then a forward over
// the rest - the media chunk at start position P, exactly what the engine runs when
// it continues a conversation's cache into an image turn. Both then decode
// greedily. The prefill logits must agree to floating-point noise and the greedy
// tokens must be identical.
//
// Two prefix lengths: inside Gemma 4's sliding window (512 tokens on E4B), and past it, where
// the local layers' ring has wrapped and the chunk attends a window gathered from
// the ring. Each runs on the fused whole-model prefill (which must actually serve
// the media chunk at P) and on the per-op multimodal path (TS_G4_MM_PREFILL=0),
// each path compared with its own cold prefill.
//
// The fused kernels are bitwise reproducible, so their greedy streams must match
// token for token. The per-op path is not reproducible even against itself: four
// identical cold prefills in one process (E4B, Metal) left the runner-up 0.045 to
// 0.274 logits behind at decode step 16, and its cold stream flipped there between
// runs (E2B: at step 19). So the per-op rows check the prefill logits only - a wrong
// soft-token mask moved them by 3.2 on E4B, the noise by 0.02 - and report the
// streams.
//
//   TS_TEST_MODEL_DIR=~/work/models/gemma-4-E4B TS_TEST_GGML_BACKEND=metal
//   TS_TEST_MODEL_DIR=~/work/models/gemma-4-E2B TS_TEST_GGML_BACKEND=metal
//   (also gemma-4-12b, and gemma-4-26b for the all-MoE kernel; each directory holds
//    one model, since the MTP head GGUF matches the same name)
//   (the directory also holds the model's mmproj GGUF; TS_TEST_MEDIA_DIR holds
//    image.png and sample.wav, default ~/work/models/testmedia)
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4MediaAfterReusedPrefixExactnessTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string ModelNames = "gemma-4-e4b|gemma-4-e2b|gemma-4-12b|gemma-4-26b";
    private const int DecodeTokens = 24;
    private readonly ITestOutputHelper _output;

    public Gemma4MediaAfterReusedPrefixExactnessTests(ITestOutputHelper output) { _output = output; }

    [ModelTheory(EnvModelDir, ModelNames)]
    [InlineData("image", false, true)]    // prefix inside the window, fused prefill
    [InlineData("image", true, true)]     // prefix past the window (wrapped ring), fused prefill
    [InlineData("image", false, false)]   // inside the window, per-op multimodal path
    [InlineData("image", true, false)]    // past the window, per-op multimodal path
    [InlineData("audio", true, true)]     // audio uses the same soft-token mask
    public void MediaTurnAfterAReusedPrefix_MatchesAColdPrefill(string media, bool longPrefix, bool fused)
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = dir == null ? null : TestGates.FindGguf(dir, ModelNames);
        if (modelPath == null) { _output.WriteLine("no Gemma 4 model; skipping"); return; }
        string mmproj = Directory.GetFiles(Path.GetDirectoryName(modelPath)!, "*.gguf")
            .Where(p => Path.GetFileName(p).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
            .OrderBy(p => new FileInfo(p).Length)
            .FirstOrDefault();
        if (mmproj == null) { _output.WriteLine($"no mmproj GGUF next to {modelPath}; skipping"); return; }
        string mediaDir = Environment.GetEnvironmentVariable("TS_TEST_MEDIA_DIR")
            ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "work", "models", "testmedia");
        string mediaPath = Path.Combine(mediaDir, media == "audio" ? "sample.wav" : "image.png");
        if (!File.Exists(mediaPath)) { _output.WriteLine($"no {mediaPath}; skipping"); return; }

        BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
            .Trim().ToLowerInvariant() switch
        {
            "metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.GgmlCuda,
            _ => BackendType.GgmlCpu,
        };

        using var model = (Gemma4Model)ModelBase.Create(modelPath, backend);
        model.MultimodalInjector.LoadProjectors(mmproj);
        if (media == "audio" && model.AudioEncoder == null) { _output.WriteLine($"{mmproj} has no audio encoder; skipping"); return; }
        if (media == "image" && model.VisionEncoder == null) { _output.WriteLine($"{mmproj} has no vision encoder; skipping"); return; }
        model.FusedMediaPrefillEnabled = fused;

        // ---- the prompt: a finished text turn, then a media turn.
        var firstTurn = new List<ChatMessage>
        {
            new() { Role = "user", Content = FirstQuestion(longPrefix) },
            new() { Role = "assistant", Content = "Noted. I have read the whole inventory and can answer questions about any room." },
        };
        var history = new List<ChatMessage>(firstTurn)
        {
            new()
            {
                Role = "user",
                Content = media == "audio" ? "What is said in this recording? Answer in two sentences." : "Describe this image in two sentences.",
                ImagePaths = media == "image" ? new List<string> { mediaPath } : null,
                AudioPaths = media == "audio" ? new List<string> { mediaPath } : null,
            },
        };
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> rendered = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history, "gemma4", addGenerationPrompt: true);
        List<int> previous = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, firstTurn, "gemma4", addGenerationPrompt: false);

        const string requestId = "media-after-reuse";
        var injector = model.MultimodalInjector;
        int[] prompt = injector.ProcessPromptTokens(history, rendered, requestId).ToArray();
        IReadOnlyList<PromptMediaSpan> spans = injector.GetPreparedMediaSpans(requestId);
        Assert.Single(spans);
        int mediaStart = spans[0].Start;

        // The reused prefix is what the previous turn left: the text both renders
        // share, which ends before the media.
        int reused = 0;
        while (reused < previous.Count && reused < prompt.Length && previous[reused] == prompt[reused]) reused++;
        reused = Math.Min(reused, mediaStart);
        int window = model.MaxReusablePrefixTokens;   // the sliding window
        _output.WriteLine($"model={Path.GetFileName(modelPath)} backend={backend} media={media} fused={fused} " +
                          $"prompt={prompt.Length} reused={reused} span=[{spans[0].Start},{spans[0].End})");
        Assert.True(longPrefix ? reused > window : reused + (prompt.Length - reused) <= window,
            $"the test prompt does not exercise the intended case (reused {reused}, prompt {prompt.Length})");

        // ---- cold: the whole prompt from position 0.
        model.ResetKVCache();
        Assert.True(injector.QueuePromptEmbeddings(0, requestId));
        int coldChunks = model.FusedMediaPrefillChunks;
        float[] coldLogits = (float[])model.Forward(prompt).Clone();
        bool coldFused = model.FusedMediaPrefillChunks > coldChunks;
        List<int> coldTokens = Decode(model, coldLogits, out List<float> coldMargins);

        // ---- reuse: the previous turn's text, then the media chunk at position P.
        model.ResetKVCache();
        model.Forward(prompt.Take(reused).ToArray());
        Assert.True(injector.QueuePromptEmbeddingsForSlice(reused, prompt.Length - reused, requestId));
        int afterPrefix = model.FusedMediaPrefillChunksAfterPrefix;
        float[] reuseLogits = (float[])model.Forward(prompt.Skip(reused).ToArray()).Clone();
        bool reuseFused = model.FusedMediaPrefillChunksAfterPrefix > afterPrefix;
        List<int> reuseTokens = Decode(model, reuseLogits, out _);
        injector.ClearPreparedPromptState(requestId);

        float diff = MaxAbsDiff(coldLogits, reuseLogits);
        float scale = Math.Max(1e-6f, coldLogits.Max(Math.Abs));
        _output.WriteLine($"fused media chunk: cold={coldFused} reuse={reuseFused}");
        _output.WriteLine($"prefill logits max|diff| {diff:E2} (logit scale {scale:F1})");
        _output.WriteLine($"cold:  {Escape(model.Tokenizer.Decode(coldTokens))}");
        _output.WriteLine($"reuse: {Escape(model.Tokenizer.Decode(reuseTokens))}");
        int firstDiff = Enumerable.Range(0, DecodeTokens).FirstOrDefault(i => coldTokens[i] != reuseTokens[i], -1);
        if (firstDiff >= 0)
            _output.WriteLine($"first different token at {firstDiff}: the cold top-2 logit margin there was {coldMargins[firstDiff]:F4} " +
                              $"(smallest margin up to it {coldMargins.Take(firstDiff + 1).Min():F4})");

        Assert.Equal(fused, coldFused);
        Assert.Equal(fused, reuseFused);
        Assert.True(diff <= 0.005f * scale,
            $"the media turn's logits after a {reused}-token reused prefix differ from a cold prefill by {diff:E2} " +
            $"(scale {scale:F1})");
        if (fused)
            Assert.Equal(coldTokens, reuseTokens);
    }

    private static string FirstQuestion(bool longPrefix)
    {
        if (!longPrefix)
            return "Here is a short inventory. Room 1 holds a red chair. Room 2 holds a blue lamp. " +
                   "Room 3 holds a green rug. Please remember it.";
        string[] colours = { "red", "blue", "green", "amber", "violet", "silver", "ochre" };
        string[] things = { "chair", "lamp", "rug", "clock", "kettle", "mirror", "bookcase", "piano", "vase", "desk", "globe" };
        var sb = new StringBuilder("Here is the inventory of a large house. Please remember it.\n");
        for (int room = 1; room <= 60; room++)
            sb.Append($"Room {room} holds a {colours[room % colours.Length]} {things[room % things.Length]} " +
                      $"bought in {1900 + room * 3} for {room * 17} dollars.\n");
        return sb.ToString();
    }

    /// <summary>Greedy decode; <paramref name="margins"/>[i] is how far token i's logit
    /// led the runner-up, which tells a near-tie from a real difference.</summary>
    private static List<int> Decode(ModelBase model, float[] logits, out List<float> margins)
    {
        var tokens = new List<int>();
        margins = new List<float>();
        for (int i = 0; i < DecodeTokens; i++)
        {
            int t = Argmax(logits, out float margin);
            tokens.Add(t);
            margins.Add(margin);
            if (i + 1 < DecodeTokens)
                logits = model.Forward(new[] { t });
        }
        return tokens;
    }

    private static int Argmax(float[] v, out float margin)
    {
        int best = 0, second = -1;
        for (int i = 1; i < v.Length; i++)
        {
            if (v[i] > v[best]) { second = best; best = i; }
            else if (second < 0 || v[i] > v[second]) second = i;
        }
        margin = second < 0 ? float.PositiveInfinity : v[best] - v[second];
        return best;
    }

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static string Escape(string s) => s.Replace("\n", "\\n");
}
