using System;
using System.IO;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.FileProviders;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for <see cref="UploadContentPolicy"/> and <see cref="UploadStaticFiles"/>: the upload extension allow-list
/// and the content types /uploads serves back. Uploaded files are attacker
/// content, so an uploaded .html/.svg must never come back with a content type
/// a browser will execute in the server's origin.
/// </summary>
public class UploadContentPolicyTests
{
    [Theory]
    [InlineData(".png", "image")]
    [InlineData(".jpeg", "image")]
    [InlineData(".heic", "image")]
    [InlineData(".mp4", "video")]
    [InlineData(".mp3", "audio")]
    [InlineData(".pdf", "pdf")]
    [InlineData(".docx", "document")]
    [InlineData(".xlsx", "document")]
    [InlineData(".pptx", "document")]
    [InlineData(".txt", "text")]
    [InlineData(".html", "text")]
    [InlineData(".cs", "text")]
    public void Classify_KnownExtensions_KeepTheirMediaType(string ext, string expected)
    {
        Assert.Equal(expected, UploadContentPolicy.Classify(ext));
    }

    [Theory]
    [InlineData(".svg")]
    [InlineData(".exe")]
    [InlineData(".xhtml")]
    [InlineData("")]
    public void Classify_UnlistedExtensions_AreUnknown(string ext)
    {
        Assert.Equal("unknown", UploadContentPolicy.Classify(ext));
    }

    [Theory]
    [InlineData(".png", "image/png")]
    [InlineData(".jpg", "image/jpeg")]
    [InlineData(".heic", "image/heic")]
    [InlineData(".heif", "image/heif")]
    [InlineData(".mp4", "video/mp4")]
    [InlineData(".webm", "video/webm")]
    [InlineData(".wav", "audio/wav")]
    [InlineData(".pdf", "application/pdf")]
    public void Serve_MediaExtensions_KeepRealContentTypes(string ext, string expected)
    {
        var provider = UploadStaticFiles.BuildServeContentTypes();
        Assert.True(provider.TryGetContentType("f" + ext, out string contentType));
        Assert.Equal(expected, contentType);
    }

    [Theory]
    [InlineData(".html")]
    [InlineData(".js")]
    [InlineData(".css")]
    [InlineData(".xml")]
    [InlineData(".md")]
    [InlineData(".json")]
    [InlineData(".txt")]
    public void Serve_TextExtensions_AlwaysComeBackAsPlainText(string ext)
    {
        var provider = UploadStaticFiles.BuildServeContentTypes();
        Assert.True(provider.TryGetContentType("f" + ext, out string contentType));
        Assert.Equal("text/plain; charset=utf-8", contentType);
    }

    [Theory]
    [InlineData("f.svg")]
    [InlineData("f.exe")]
    [InlineData("f.xhtml")]
    [InlineData("f")]
    public void Serve_UnlistedExtensions_HaveNoContentType(string fileName)
    {
        var provider = UploadStaticFiles.BuildServeContentTypes();
        Assert.False(provider.TryGetContentType(fileName, out _));
    }

    [Fact]
    public void Serve_CoversEveryAcceptedUploadExtension()
    {
        var provider = UploadStaticFiles.BuildServeContentTypes();
        foreach (string ext in UploadContentPolicy.SupportedExtensions)
            Assert.True(provider.TryGetContentType("f" + ext, out _),
                $"accepted upload extension {ext} has no serve content type");
    }

    [Fact]
    public void StaticFileOptions_MountUploadsWithNosniffAndNoUnknownTypes()
    {
        string dir = Directory.CreateTempSubdirectory("ts-uploads-test").FullName;
        try
        {
            var options = UploadStaticFiles.BuildStaticFileOptions(dir);

            Assert.Equal("/uploads", options.RequestPath.Value);
            Assert.False(options.ServeUnknownFileTypes);

            var responseContext = new StaticFileResponseContext(
                new DefaultHttpContext(), new NotFoundFileInfo("f.png"));
            options.OnPrepareResponse(responseContext);
            Assert.Equal("nosniff",
                responseContext.Context.Response.Headers["X-Content-Type-Options"]);
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }
}
