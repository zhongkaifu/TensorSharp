// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Globalization;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Primitives;
using TensorSharp.Chat;
using TensorSharp.Models.QwenImage;
using TensorSharp.Server.ProtocolAdapters;

namespace InferenceWeb.Tests;

public sealed class QwenImageMaskRequestTests
{
    private static JsonElement Json(string text) => JsonSerializer.Deserialize<JsonElement>(text);

    [Fact]
    public void MaskParameters_DefaultToWhiteEditsAndKeepAllExplicitOptions()
    {
        Assert.Equal(QwenImageMaskMode.Grayscale, WebUiChatService.ParseImageParameters(Json("{}")).MaskMode);
        var p = WebUiChatService.ParseImageParameters(Json("""
            {"maskPath":"mask.png","maskMode":"alpha","maskInvert":true,"maskFeather":8,"maskCrop":true,"maskCropPadding":96}
            """));
        Assert.Equal(QwenImageMaskMode.Alpha, p.MaskMode);
        Assert.True(p.MaskInvert);
        Assert.Equal(8, p.MaskFeather);
        Assert.True(p.MaskCrop);
        Assert.Equal(96, p.MaskCropPadding);
    }

    [Theory]
    [InlineData("{\"maskMode\":\"edit\"}")]
    [InlineData("{\"maskMode\":null}")]
    [InlineData("{\"maskMode\":1}")]
    [InlineData("{\"maskFeather\":-1}")]
    [InlineData("{\"maskFeather\":1025}")]
    [InlineData("{\"maskFeather\":1.5}")]
    [InlineData("{\"maskCropPadding\":-1}")]
    [InlineData("{\"maskCropPadding\":16385}")]
    [InlineData("{\"maskCrop\":\"true\"}")]
    [InlineData("{\"maskInvert\":1}")]
    public void InvalidMaskOptionsAreRejected(string json) =>
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiChatService.ParseImageParameters(Json(json))).StatusCode);

    [Theory]
    [InlineData("maskPath", "\"mask.png\"")]
    [InlineData("maskMode", "\"grayscale\"")]
    [InlineData("maskFeather", "0")]
    [InlineData("maskInvert", "false")]
    [InlineData("maskCrop", "false")]
    [InlineData("maskCropPadding", "64")]
    public void MaskFieldsRequireMaskAndAreRejectedForTextToImage(string name, string value)
    {
        var body = Json("{\"prompt\":\"cat\",\"" + name + "\":" + value + "}");
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiChatService.ValidateMaskPresence(body, false)).StatusCode);
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiChatService.ParseImagePrompt(body, true)).StatusCode);
    }

    private static FormFile File(string field, byte[]? bytes = null)
    {
        bytes ??= [1, 2, 3];
        return new FormFile(new MemoryStream(bytes), 0, bytes.Length, field, field + ".png");
    }

    private static FormCollection Form(Dictionary<string, StringValues>? fields = null, params IFormFile[] files)
    {
        var collection = new FormFileCollection();
        foreach (var file in files) collection.Add(file);
        return new FormCollection(fields ?? new(), collection);
    }

    [Theory]
    [InlineData("image")]
    [InlineData("image[]")]
    [InlineData("legacy-file")]
    public void MultipartSeparatesMaskFromReferenceImagesAndPreservesOrder(string field)
    {
        var first = File(field);
        var second = File(field);
        var mask = File("mask");
        var parsed = WebUiAdapter.ParseImageEditForm(Form(new() { ["maskMode"] = "alpha", ["maskCrop"] = "true" }, mask, first, second), 1024);
        Assert.Equal(new[] { first, second }, parsed.Images);
        Assert.Same(mask, parsed.Mask);
        Assert.Equal("alpha", parsed.Parameters.GetProperty("maskMode").GetString());
        Assert.True(parsed.Parameters.GetProperty("maskCrop").GetBoolean());
    }

    [Fact]
    public void MultipartMaskAloneCannotBecomeTheInputImage() =>
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(Form(null, File("mask")), 1024)).StatusCode);

    [Fact]
    public void MultipartRejectsMultipleMasks() =>
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(Form(null, File("image"), File("mask"), File("mask")), 1024)).StatusCode);

    [Theory]
    [InlineData("cfg", "NaN")]
    [InlineData("cfg", "1,5")]
    [InlineData("seed", "9223372036854775808")]
    [InlineData("steps", "potato")]
    [InlineData("steps", "-1")]
    [InlineData("targetArea", "0")]
    [InlineData("maskFeather", "1.1")]
    [InlineData("maskCrop", "1")]
    public void MultipartRejectsMalformedValuesInsteadOfSilentlyUsingDefaults(string field, string value) =>
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(new() { [field] = value }, File("image"), File("mask")), 1024)).StatusCode);

    [Fact]
    public void MultipartKeepSourceSizeIsAJsonBoolean()
    {
        var parsed = WebUiAdapter.ParseImageEditForm(Form(new() { ["keepSourceSize"] = "true" }, File("image")), 1024);
        Assert.True(WebUiChatService.ParseImageParameters(parsed.Parameters).KeepSourceSize);
        foreach (string value in new[] { "yes", "1" })
            Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
                Form(new() { ["keepSourceSize"] = value }, File("image")), 1024)).StatusCode);
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(new() { ["keepSourceSize"] = "true", ["width"] = "1024", ["height"] = "768" }, File("image")), 1024)).StatusCode);
    }

    [Fact]
    public void MultipartNumbersAreInvariantAndKeepInt64Seeds()
    {
        var old = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = CultureInfo.GetCultureInfo("fr-FR");
            var parsed = WebUiAdapter.ParseImageEditForm(Form(new() { ["cfg"] = "1.5", ["seed"] = long.MaxValue.ToString(CultureInfo.InvariantCulture) }, File("image")), 1024);
            var p = WebUiChatService.ParseImageParameters(parsed.Parameters);
            Assert.Equal(1.5f, p.CfgScale);
            Assert.Equal(long.MaxValue, p.Seed);
        }
        finally { CultureInfo.CurrentCulture = old; }
    }

    [Fact]
    public void MultipartEnforcesLimitsOnMask()
    {
        Assert.Equal(413, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(null, File("image", [1]), File("mask", [1, 2, 3])), 2)).StatusCode);
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(null, File("image"), File("mask", [])), 1024)).StatusCode);
    }

    [Fact]
    public void MultipartRejectsMaskMetadataWithoutMaskFile() =>
        Assert.Equal(400, Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(new() { ["maskMode"] = "grayscale" }, File("image")), 1024)).StatusCode);

    [Fact]
    public void MultipartRejectsDuplicateScalarAndTextMaskParts()
    {
        Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(new() { ["maskMode"] = new StringValues(["alpha", "grayscale"]) }, File("image"), File("mask")), 1024));
        Assert.Throws<WebUiRequestRejectedException>(() => WebUiAdapter.ParseImageEditForm(
            Form(new() { ["mask"] = "mask.png" }, File("image")), 1024));
    }
}
