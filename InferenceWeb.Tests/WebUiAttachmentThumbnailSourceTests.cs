// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text.RegularExpressions;

namespace InferenceWeb.Tests;

/// <summary>
/// Guards the composer's pending-attachment strip in the bundled desktop page: an
/// uploaded image, video or scanned PDF shows as a thumbnail, and the client's own file name never reaches
/// innerHTML. Like the other Web UI source tests, this inspects the exact copy shipped
/// beside the server rather than adding a JavaScript runtime.
/// </summary>
public sealed class WebUiAttachmentThumbnailSourceTests
{
    private static string ReadWebUi()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html");
        Assert.True(File.Exists(path), $"The TensorSharp.Server Web UI was not copied to the test output: {path}");
        return File.ReadAllText(path);
    }

    private static string Between(string source, string startMarker, string endMarker)
    {
        int start = source.IndexOf(startMarker, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Could not find Web UI marker: {startMarker}");
        int end = source.IndexOf(endMarker, start + startMarker.Length, StringComparison.Ordinal);
        Assert.True(end > start, $"Could not find Web UI marker after {startMarker}: {endMarker}");
        return source.Substring(start, end - start);
    }

    [Fact]
    public void PendingVisualMedia_RendersAsThumbnailFromTheDisplayableUrl()
    {
        string html = ReadWebUi();
        string render = Between(html, "function renderAttachments()", "function buildAttachmentChip(");
        string thumb = Between(html, "function buildAttachmentThumb(", "function removeAttachment(");

        // uploadUrlForAttachment prefers previewUrl, the PNG the server makes for HEIC/HEIF.
        Assert.Matches(new Regex(@"mediaType === 'image'\)[\s\S]{0,400}?thumb = \{ url: uploadUrlForAttachment\(att\)"), render);
        // A clip shows its first extracted frame; a scanned PDF its first page image,
        // with the page count kept visible on the tile as a bare count that fits it.
        Assert.Matches(new Regex(@"mediaType === 'video'\)[\s\S]{0,200}?thumb = \{ url: \(att\.frameUrls && att\.frameUrls\[0\]\)"), render);
        Assert.Matches(new Regex(@"mediaType === 'pdf' && att\.renderedAsImages\)[\s\S]{0,700}?thumb = \{ url: \(att\.frameUrls && att\.frameUrls\[0\]\) \|\| '', kind: 'PDF', badge: pages,[\s\S]{0,120}?page: true"), render);
        Assert.Contains(".attachment-thumb.page img { object-position: left top; }", html, StringComparison.Ordinal);
        // Every kind with a descriptor gets a tile; only a missing or failed image falls back.
        Assert.Matches(new Regex(@"appendChild\(thumb && thumb\.url && !att\.thumbFailed\s*\?\s*buildAttachmentThumb\(att, idx, name, thumb, chip\)"), render);
        // "Picture N" counts still images only: that is the order the edit request carries.
        Assert.Contains("const totalImages = pendingAttachments.filter(a => a.mediaType === 'image').length;", render, StringComparison.Ordinal);
        Assert.Contains("img.src = thumb.url;", thumb, StringComparison.Ordinal);
        // A thumbnail that cannot be drawn degrades to the named chip, not an empty tile,
        // and is remembered so a re-render neither refetches it nor flashes a tile.
        Assert.Matches(new Regex(@"addEventListener\('error'[\s\S]{0,200}?att\.thumbFailed = true;[\s\S]{0,80}?replaceWith\(chip\(\)\)"), thumb);
    }

    [Fact]
    public void AttachmentNames_AreNeverInterpolatedIntoInnerHtml()
    {
        string renderers = Between(ReadWebUi(), "function renderAttachments()", "function removeAttachment(");

        // The only innerHTML write left is the strip reset.
        Assert.Single(Regex.Matches(renderers, "innerHTML"));
        Assert.Contains("attachmentsDiv.innerHTML = '';", renderers, StringComparison.Ordinal);
        Assert.Contains("text.textContent = icon + ' ' + label;", renderers, StringComparison.Ordinal);
    }
}
