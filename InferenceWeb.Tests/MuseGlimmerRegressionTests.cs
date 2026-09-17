// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Regression tests for two bugs found while bringing up Muse-Glimmer. Both were
// SILENT: no exception, no log, just wrong output. Neither needs a GGUF, so they
// run unconditionally.
//
//   1. BpeTokenizer had no "llama4" pre-tokenizer case and fell through to the
//      default pattern, whose \p{N} splits every digit individually. "17"
//      tokenized as '1','7' instead of the single "17" token, so every numeric
//      prompt diverged from llama.cpp. llama.cpp maps tokenizer.ggml.pre in
//      {gpt-4o, llama4, kanana2, talkie} to one regex (llama-vocab.cpp:2293-2299).
//
//   2. ChatTemplate.InjectMultimodalTokens had no muse-glimmer branch, so the
//      <|patch|> image marker never reached the rendered prompt. The vision
//      encoder still ran and produced correct embeddings; they were then dropped
//      because the host found no placeholder to expand, and the model answered
//      "I don't see an image attached to your message."
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class MuseGlimmerRegressionTests
{
    [Fact]
    public void ImageGeometry_IosPortrait_UsesBoundedReferenceGrid()
    {
        var processor = new MuseGlimmerImageProcessor(
            patchSize: 14,
            mergeSize: 2,
            maxTokens: 4096);

        var (width, height) = processor.ComputeTargetSize(
            imgWidth: 3808,
            imgHeight: 5712);

        Assert.Equal(1456, width);
        Assert.Equal(2184, height);

        int rawPatchCount = (width / processor.PatchSize) * (height / processor.PatchSize);
        Assert.Equal(16_224, rawPatchCount);
        Assert.Equal(4_056, processor.ComputeTokenCount(3808, 5712));
    }

    // ---------------------------------------------------------------------
    // Bug 1: llama4 pre-tokenizer
    // ---------------------------------------------------------------------

    [Theory]
    [InlineData("llama4")]
    [InlineData("kanana2")]
    [InlineData("talkie")]
    public void PreTokenizer_Gpt4oFamily_SharesTheGpt4oPattern(string pre)
    {
        string expected = BpeTokenizer.ResolvePreTokenizerPattern("gpt-4o");
        string actual = BpeTokenizer.ResolvePreTokenizerPattern(pre);
        Assert.Equal(expected, actual);
    }

    [Theory]
    [InlineData("glm4")]
    [InlineData("chatglm-bpe")]
    public void PreTokenizer_Glm4_KeepsDigitRunsOfUpToThree(string pre)
    {
        // llama.cpp LLAMA_VOCAB_PRE_TYPE_CHATGLM4 (GLM-4, GLM-5.2, GLM-5.3, GLM-5.3-Flash).
        var regex = new Regex(BpeTokenizer.ResolvePreTokenizerPattern(pre));
        Assert.Equal(new[] { "INV", "-", "472" }, Split(regex, "INV-472"));
        Assert.Equal(new[] { "silver", "-", "482", "1" }, Split(regex, "silver-4821"));
        Assert.Equal(new[] { " is", " 17", " +", " 25" }.Select(x => x.Trim()).ToArray(),
            Split(regex, " is 17 + 25").Select(x => x.Trim()).ToArray());
    }

    [Theory]
    [InlineData("gpt-4o")]
    [InlineData("llama4")]
    public void PreTokenizer_Gpt4oFamily_KeepsMultiDigitRunsTogether(string pre)
    {
        var regex = new Regex(BpeTokenizer.ResolvePreTokenizerPattern(pre));

        // \p{N}{1,3}: runs of up to three digits stay in one pre-token.
        Assert.Equal(new[] { "17" }, Split(regex, "17"));
        Assert.Equal(new[] { "23" }, Split(regex, "23"));
        Assert.Equal(new[] { "391" }, Split(regex, "391"));

        // Four digits split 3 + 1, exactly as llama.cpp does.
        Assert.Equal(new[] { "202", "5" }, Split(regex, "2025"));

        // The prompt that originally exposed the bug.
        var pieces = Split(regex, "Q: What is 17 * 23?");
        Assert.Contains("17", pieces);
        Assert.Contains("23", pieces);
        Assert.DoesNotContain(pieces, p => p == "1" || p == "7" || p == "2" || p == "3");
    }

    [Fact]
    public void PreTokenizer_DefaultPattern_StillSplitsEveryDigit()
    {
        // Documents the behaviour the llama4 vocabs were accidentally getting.
        // If this ever changes, the fix above needs revisiting rather than silently
        // becoming a no-op.
        var fallback = new Regex(BpeTokenizer.ResolvePreTokenizerPattern("some-unknown-pre"));
        Assert.Equal(new[] { "1", "7" }, Split(fallback, "17"));
    }

    private static string[] Split(Regex regex, string text)
        => regex.Matches(text).Select(m => m.Value).Where(v => v.Trim().Length > 0).ToArray();

    // ---------------------------------------------------------------------
    // Bug 2: muse-glimmer image marker
    // ---------------------------------------------------------------------

    [Fact]
    public void InjectMultimodalTokens_MuseGlimmer_EmitsPatchMarkerPerImage()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Describe this image.", ImagePaths = new List<string> { "a.png" } },
        };

        var injected = ChatTemplate.InjectMultimodalTokens(messages, "muse-glimmer");

        Assert.Single(injected);
        Assert.Equal("<|patch|>Describe this image.", injected[0].Content);
        // The paths must survive so the host can still encode them.
        Assert.Equal(new[] { "a.png" }, injected[0].ImagePaths.ToArray());
    }

    [Fact]
    public void InjectMultimodalTokens_MuseGlimmer_EmitsOneMarkerPerImage()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Compare.", ImagePaths = new List<string> { "a.png", "b.png", "c.png" } },
        };

        var injected = ChatTemplate.InjectMultimodalTokens(messages, "muse-glimmer");

        Assert.Equal("<|patch|><|patch|><|patch|>Compare.", injected[0].Content);
    }

    [Fact]
    public void InjectMultimodalTokens_MuseGlimmer_UsesVideoMarkerForFrames()
    {
        var messages = new List<ChatMessage>
        {
            new()
            {
                Role = "user",
                Content = "What happens?",
                ImagePaths = new List<string> { "f0.png", "f1.png" },
                IsVideo = true,
            },
        };

        var injected = ChatTemplate.InjectMultimodalTokens(messages, "muse-glimmer");

        Assert.Equal("<|video|>What happens?", injected[0].Content);
    }

    [Fact]
    public void InjectMultimodalTokens_MuseGlimmer_UnderscoreSpellingAlsoWorks()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Hi.", ImagePaths = new List<string> { "a.png" } },
        };

        Assert.Equal("<|patch|>Hi.",
            ChatTemplate.InjectMultimodalTokens(messages, "muse_glimmer")[0].Content);
    }

    [Fact]
    public void InjectMultimodalTokens_MuseGlimmer_TextOnlyMessageIsUntouched()
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "No image here." },
        };

        var injected = ChatTemplate.InjectMultimodalTokens(messages, "muse-glimmer");

        Assert.Equal("No image here.", injected[0].Content);
        Assert.Same(messages[0], injected[0]);
    }

    /// <summary>
    /// The failure mode this guards: an architecture with images but no branch in
    /// InjectMultimodalTokens silently produces a prompt with no placeholder, so
    /// the encoded image embeddings are dropped. Every architecture whose model
    /// class exposes a vision encoder must appear here.
    /// </summary>
    [Theory]
    [InlineData("gemma4", "<|image>")]
    [InlineData("qwen35", "<|vision_start|><|image_pad|><|vision_end|>")]
    [InlineData("mistral3", "[IMG]")]
    [InlineData("muse-glimmer", "<|patch|>")]
    public void InjectMultimodalTokens_EveryVisionArchEmitsAMarker(string architecture, string expectedMarker)
    {
        var messages = new List<ChatMessage>
        {
            new() { Role = "user", Content = "x", ImagePaths = new List<string> { "a.png" } },
        };

        var injected = ChatTemplate.InjectMultimodalTokens(messages, architecture);

        Assert.Equal(expectedMarker + "x", injected[0].Content);
    }
}
