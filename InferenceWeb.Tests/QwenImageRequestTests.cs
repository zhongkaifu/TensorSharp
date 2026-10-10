// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Chat;
using TensorSharp.Server.ProtocolAdapters;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class QwenImageRequestTests
{
    private static JsonElement Json(string text) => JsonDocument.Parse(text).RootElement.Clone();

    [Fact]
    public void ImageParameters_PreserveGeometrySamplingAndNegativePrompt()
    {
        var p = WebUiChatService.ParseImageParameters(Json("""
            {"width":1024,"height":768,"steps":40,"cfg":6,"seed":9223372036854775807,
             "negativePrompt":"blur","targetArea":786432}
            """));
        Assert.Equal(1024, p.Width);
        Assert.Equal(768, p.Height);
        Assert.Equal(40, p.Steps);
        Assert.Equal(6f, p.CfgScale);
        Assert.Equal(long.MaxValue, p.Seed);
        Assert.Equal("blur", p.NegativePrompt);
        Assert.Equal(786432, p.TargetArea);
    }

    [Theory]
    [InlineData("{\"width\":1008,\"height\":1024}")]
    [InlineData("{\"width\":1024}")]
    [InlineData("{\"width\":-32,\"height\":32}")]
    [InlineData("{\"steps\":-1}")]
    [InlineData("{\"cfg\":-1}")]
    [InlineData("{\"cfg\":1e100}")]
    [InlineData("{\"width\":\"1024\"}")]
    [InlineData("{\"targetArea\":0}")]
    [InlineData("[]")]
    public void Image21Parameters_RejectInvalidRequestsBeforeInference(string json)
    {
        var error = Assert.Throws<WebUiRequestRejectedException>(() =>
            WebUiChatService.ParseImageParameters(Json(json)));
        Assert.Equal(400, error.StatusCode);
    }

    [Fact]
    public void ImageParameters_UnspecifiedSamplingIsResolvedByModel()
    {
        var p = WebUiChatService.ParseImageParameters(Json("{}"));
        Assert.Equal(0, p.Steps);
        Assert.Equal(0f, p.CfgScale);
        Assert.Equal(0, p.Width);
        Assert.Equal(0, p.Height);
    }

    [Fact]
    public void ImageParameters_OmittedAreaUsesTheModelsNativeResolution()
    {
        // Both plain and streaming routes share this parser; the Web UI omits geometry.
        var p = WebUiChatService.ParseImageParameters(Json("{}"));
        Assert.Equal(4194304, p.TargetArea);
        Assert.Equal(0, p.Width);
        Assert.Equal(0, p.Height);
    }

    [Fact]
    public void ImageParameters_ExplicitDraftAreaOverridesModelDefaults()
    {
        var p = WebUiChatService.ParseImageParameters(Json("{\"targetArea\":1048576}"));
        Assert.Equal(1048576, p.TargetArea);
        Assert.Equal(1048576, p.ResolveTargetArea());
    }

    [Theory]
    [InlineData("{}")]
    [InlineData("{\"prompt\":\" \"}")]
    [InlineData("{\"prompt\":5}")]
    [InlineData("{\"prompt\":\"cat\",\"imagePaths\":[\"ref.png\"]}")]
    [InlineData("{\"prompt\":\"cat\",\"imagePath\":\"ref.png\"}")]
    public void GeneratePrompt_RejectsMissingPromptAndAccidentalReferenceImages(string json)
    {
        var error = Assert.Throws<WebUiRequestRejectedException>(() =>
            WebUiChatService.ParseImagePrompt(Json(json), generate: true));
        Assert.Equal(400, error.StatusCode);
    }

    [Fact]
    public void GeneratePrompt_AllowsAnEmptyAttachmentArrayFromTheWebUi()
    {
        Assert.Equal("cat", WebUiChatService.ParseImagePrompt(Json("{\"prompt\":\"cat\",\"imagePaths\":[]}"), generate: true));
    }

    [Fact]
    public void EditPrompt_StillAllowsAnEmptyInstruction()
    {
        Assert.Equal("", WebUiChatService.ParseImagePrompt(Json("{}"), generate: false));
    }

    [Fact]
    public void KeepSourceSize_IsReadAndOffUnlessSent()
    {
        Assert.False(WebUiChatService.ParseImageParameters(Json("{}")).KeepSourceSize);
        Assert.False(WebUiChatService.ParseImageParameters(Json("{\"keepSourceSize\":false}")).KeepSourceSize);
        // TensorAgent's edit body: its area stays the budget the edit samples within.
        var p = WebUiChatService.ParseImageParameters(Json("{\"keepSourceSize\":true,\"targetArea\":1048576}"));
        Assert.True(p.KeepSourceSize);
        Assert.Equal(1048576, p.TargetArea);
    }

    [Theory]
    [InlineData("{\"keepSourceSize\":\"true\"}")]
    [InlineData("{\"keepSourceSize\":1}")]
    [InlineData("{\"keepSourceSize\":null}")]
    [InlineData("{\"keepSourceSize\":true,\"width\":1024,\"height\":768}")]
    public void KeepSourceSize_RefusesNonBooleansAndAnExplicitSize(string json)
    {
        var error = Assert.Throws<WebUiRequestRejectedException>(() =>
            WebUiChatService.ParseImageParameters(Json(json)));
        Assert.Equal(400, error.StatusCode);
    }

    [Fact]
    public void KeepSourceSize_IsAnEditOption()
    {
        var error = Assert.Throws<WebUiRequestRejectedException>(() =>
            WebUiChatService.ParseImagePrompt(Json("{\"prompt\":\"cat\",\"keepSourceSize\":true}"), generate: true));
        Assert.Equal(400, error.StatusCode);
        Assert.Contains("/api/image-edit", error.Message, StringComparison.Ordinal);
        Assert.Equal("cat", WebUiChatService.ParseImagePrompt(Json("{\"prompt\":\"cat\",\"keepSourceSize\":false}"), generate: true));
        Assert.Equal("cat", WebUiChatService.ParseImagePrompt(Json("{\"prompt\":\"cat\",\"keepSourceSize\":true}"), generate: false));
    }
}
