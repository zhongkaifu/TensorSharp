// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

public sealed class ImageDownloadTests : IDisposable
{
    private const string PageUrl = "http://127.0.0.1:43210/?token=launch";
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-image-download-" + Guid.NewGuid().ToString("N"));
    private readonly string _uploads;

    public ImageDownloadTests()
    {
        _uploads = Path.Combine(_root, "uploads");
        Directory.CreateDirectory(_uploads);
        File.WriteAllBytes(Path.Combine(_uploads, "generated.png"), MediaFixtures.RedCircleOnWhitePng(16));
        File.WriteAllText(Path.Combine(_uploads, "settings.json"), "{}");
        File.WriteAllBytes(Path.Combine(_root, "outside.png"), MediaFixtures.RedCircleOnWhitePng(16));
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { }
    }

    [Theory]
    [InlineData("/uploads/generated.png")]
    [InlineData("/uploads/generated.png?download=1#image")]
    [InlineData("http://127.0.0.1:43210/uploads/generated.png")]
    public void GeneratedResultsResolveToTheirOriginalImageFile(string url)
    {
        Assert.True(ImageDownload.TryResolve(_uploads, url, PageUrl, out string? full));
        Assert.Equal(Path.Combine(_uploads, "generated.png"), full);
    }

    [Fact]
    public void EncodedImageFilenamesAreDecodedExactlyOnce()
    {
        const string name = "edited photo #1 100%.png";
        File.Copy(Path.Combine(_uploads, "generated.png"), Path.Combine(_uploads, name));

        Assert.True(ImageDownload.TryResolve(_uploads, "/uploads/" + Uri.EscapeDataString(name), PageUrl, out string? full));
        Assert.Equal(Path.Combine(_uploads, name), full);
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("/uploads/")]
    [InlineData("/uploads/missing.png")]
    [InlineData("/uploads/settings.json")]
    [InlineData("/uploads/../outside.png")]
    [InlineData("/uploads/%2e%2e%2foutside.png")]
    [InlineData("/uploads/%2e%2e%5coutside.png")]
    [InlineData("/uploads/generated.png%00.png")]
    [InlineData("/uploads/generated.png:stream.png")]
    [InlineData("/uploads/subdirectory/generated.png")]
    [InlineData("/api/code/artifacts/run/image.png")]
    [InlineData("//127.0.0.1:43210/uploads/generated.png")]
    [InlineData("http://127.0.0.1:43211/uploads/generated.png")]
    [InlineData("http://example.com/uploads/generated.png")]
    [InlineData("https://127.0.0.1:43210/uploads/generated.png")]
    [InlineData("http://user@127.0.0.1:43210/uploads/generated.png")]
    [InlineData("data:image/png;base64,AAAA")]
    public void ThePageCannotSelectMissingFilesOtherOriginsOrFilesOutsideUploads(string? url)
    {
        Assert.False(ImageDownload.TryResolve(_uploads, url, PageUrl, out string? full));
        Assert.Null(full);
    }

    [SkippableFact]
    public void ASymbolicLinkCannotExportAFileOutsideUploads()
    {
        try { File.CreateSymbolicLink(Path.Combine(_uploads, "linked.png"), Path.Combine(_root, "outside.png")); }
        catch (Exception ex) when (ex is UnauthorizedAccessException or PlatformNotSupportedException or IOException)
        {
            Skip.If(true, "Creating a symbolic link is unavailable: " + ex.Message);
        }

        Assert.False(ImageDownload.TryResolve(_uploads, "/uploads/linked.png", PageUrl, out string? full));
        Assert.Null(full);
    }
}
