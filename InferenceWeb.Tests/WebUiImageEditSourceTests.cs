// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
namespace InferenceWeb.Tests;

/// <summary>
/// The server Web UI's image request (<c>runImageEdit</c>). Its edits keep their source
/// picture's size: sized from the area alone, an edit of a selection edit's full-size result
/// came back smaller than that result.
/// </summary>
public sealed class WebUiImageEditSourceTests
{
    private static string ReadRunImageEdit()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html");
        Assert.True(File.Exists(path), $"The TensorSharp.Server Web UI was not copied to the test output: {path}");
        string html = File.ReadAllText(path);
        int start = html.IndexOf("async function runImageEdit", StringComparison.Ordinal);
        int end = html.IndexOf("function addImageEditActions", start, StringComparison.Ordinal);
        Assert.True(start >= 0 && end > start, "Could not isolate runImageEdit in the bundled Web UI");
        return html.Substring(start, end - start);
    }

    [Fact]
    public void AnEditAsksToKeepItsSourceSizeAndAPictureFromWordsDoesNot()
    {
        string run = ReadRunImageEdit();
        // One body serves both routes; JSON.stringify leaves an undefined field out, so a picture
        // from words sends no keepSourceSize, which /api/image-generate would refuse.
        Assert.Contains("editing ? '/api/image-edit/stream' : '/api/image-generate/stream'", run, StringComparison.Ordinal);
        Assert.Contains("keepSourceSize: editing || undefined", run, StringComparison.Ordinal);
        // No size of its own: the server's default area stays the budget an edit samples within.
        Assert.DoesNotContain("width:", run, StringComparison.Ordinal);
        Assert.DoesNotContain("targetArea", run, StringComparison.Ordinal);
    }

    /// <summary>
    /// Any image model the server reports (/api/models' <c>image</c> block: MiniMax-H3-Image as well as Qwen-Image-2.1)
    /// sends a message through the image routes, and offers and sends a selection when it takes masks; an older server
    /// with no block still has Qwen-Image's architecture name.
    /// </summary>
    [Fact]
    public void ImageModelsAreKnownByTheReportedCapabilityNotAnArchitectureName()
    {
        string html = File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html"));
        Assert.Contains("currentImageCaps = data.image || null;", html, StringComparison.Ordinal);
        Assert.Contains("return !!currentImageCaps || currentArchitecture === 'qwen_image' || currentArchitecture === 'qwen-image';", html, StringComparison.Ordinal);
        Assert.Contains("return currentImageCaps ? !!currentImageCaps.supportsMask : isImageModel();", html, StringComparison.Ordinal);
        Assert.Contains("if (isImageModel() && _lastUser) {", html, StringComparison.Ordinal);
        Assert.Contains("const selectableImage = att.mediaType === 'image' && imageTakesMasks();", html, StringComparison.Ordinal);
        Assert.Contains("editSource.maskPath && imageTakesMasks()) {", html, StringComparison.Ordinal);
        Assert.DoesNotContain("isQwenImageModel", html, StringComparison.Ordinal);
    }
}
