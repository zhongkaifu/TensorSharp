// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Jev;

namespace InferenceWeb.Tests;

/// <summary>
/// The image half of the <c>/v1/systemone</c> contract: what an <c>images</c> entry may be, what
/// it is refused for, and how decoded bytes reach media storage. No model or GPU is involved.
/// </summary>
public sealed class JevImageInputTests : IDisposable
{
    // A 2x2 red PNG, small enough to inline and real enough to sniff.
    private const string Png =
        "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM8ISfHwMDAxAAGAA0EAQijE05aAAAAAElFTkSuQmCC";

    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-jev-images-" + Guid.NewGuid().ToString("N"));

    public JevImageInputTests() => Directory.CreateDirectory(_directory);
    public void Dispose()
    {
        try { Directory.Delete(_directory, recursive: true); } catch { /* best effort */ }
    }

    private static JevRequest Parse(string images)
    {
        using var document = JsonDocument.Parse("{\"state\":\"a picture of a receipt\",\"images\":" + images +
            ",\"questions\":{\"legible\":{\"type\":\"noul\",\"instructions\":\"Is it legible?\"}}}");
        return JevRequest.Parse(document.RootElement);
    }

    [Theory]
    [InlineData("\"" + Png + "\"")]
    [InlineData("\"data:image/png;base64," + Png + "\"")]
    [InlineData("\"DATA:IMAGE/PNG;BASE64," + Png + "\"")]
    public void AcceptsBareBase64AndDataUrls(string entry)
    {
        var image = Assert.Single(Parse("[" + entry + "]").Images);
        Assert.Equal("png", image.Format);
        Assert.Equal(Convert.FromBase64String(Png), image.Bytes);
    }

    [Fact]
    public void KeepsRequestOrderAndCountsAsTextOnlyWhenAbsent()
    {
        var request = Parse($"[\"{Png}\", \"data:image/jpeg;base64,{Convert.ToBase64String([0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4])}\"]");
        Assert.Equal(["png", "jpeg"], request.Images.Select(i => i.Format));
        Assert.Empty(Parse("[]").Images);
        using var none = JsonDocument.Parse("""{"state":"x","questions":{"a":{"type":"noul"}}}""");
        Assert.Empty(JevRequest.Parse(none.RootElement).Images);
    }

    [Theory]
    // Remote fetches and filesystem reads are the two ways an inference endpoint turns into a
    // request forger / file reader, so both are named refusals rather than decode failures.
    [InlineData("[\"https://example.com/receipt.png\"]", "not fetched")]
    [InlineData("[\"file:///etc/passwd\"]", "file paths")]
    [InlineData("[\"/var/data/receipt.png\"]", "file paths")]
    [InlineData("[\"C:\\\\data\\\\receipt.png\"]", "file paths")]
    [InlineData("[\"data:image/png,notbase64\"]", "base64 data: URL")]
    [InlineData("[\"data:image/png;base64\"]", "malformed data: URL")]
    [InlineData("[\"!!!not base64!!!\"]", "valid base64")]
    [InlineData("[\"\"]", "must not be empty")]
    [InlineData("[123]", "base64 string")]
    [InlineData("[\"AAAAAAAAAAAAAAAAAAAAAAAA\"]", "not a recognized image")]
    [InlineData("\"" + Png + "\"", "images must be an array")]
    public void RefusesEverythingThatIsNotInlineImageBytes(string images, string reason)
        => Assert.Contains(reason, Assert.Throws<JevValidationException>(() => Parse(images)).Message);

    [Fact]
    public void BoundsImageCountPerRequest()
    {
        string many = string.Join(",", Enumerable.Repeat($"\"{Png}\"", JevRequest.MaxImages));
        Assert.Equal(JevRequest.MaxImages, Parse("[" + many + "]").Images.Length);
        Assert.Contains("at most", Assert.Throws<JevValidationException>(
            () => Parse("[" + many + $",\"{Png}\"]")).Message);
    }

    [Fact]
    public void MaterializesContentAddressedSoAResentImageIsOneFile()
    {
        var storage = new UploadStoragePolicy(_directory);
        byte[] other = Convert.FromBase64String(Png);
        other[^1] ^= 0xFF;
        string[] paths = JevImageInput.Materialize(
            [new JevImage(Convert.FromBase64String(Png), "png"), new JevImage(other, "png"),
             new JevImage(Convert.FromBase64String(Png), "png")],
            storage);

        Assert.Equal(paths[0], paths[2]);
        Assert.NotEqual(paths[0], paths[1]);
        Assert.Equal(2, Directory.GetFiles(_directory).Length);
        Assert.All(paths, path => Assert.EndsWith(".png", path));
        Assert.Equal(Convert.FromBase64String(Png), File.ReadAllBytes(paths[0]));
    }

    [Fact]
    public void RefusesAnOversizedImageAheadOfTheVisionTower()
    {
        var storage = new UploadStoragePolicy(_directory, maxFileBytes: 8);
        Assert.Equal(413, Assert.Throws<UploadLimitExceededException>(
            () => JevImageInput.Materialize([new JevImage(Convert.FromBase64String(Png), "png")], storage)).StatusCode);
    }

    [Fact]
    public void SystemTextTellsTheModelTheStateCarriesImages()
    {
        var text = Parse($"[\"{Png}\"]");
        Assert.Contains("begins with one image", JevCompiler.SystemText(text, text.Questions, chunked: false));
        var two = Parse($"[\"{Png}\",\"data:image/gif;base64,{Convert.ToBase64String("GIF89a"u8.ToArray())}\"]");
        Assert.Contains("begins with 2 images", JevCompiler.SystemText(two, two.Questions, chunked: false));
        using var plain = JsonDocument.Parse("""{"state":"x","questions":{"a":{"type":"noul"}}}""");
        var request = JevRequest.Parse(plain.RootElement);
        Assert.DoesNotContain("image", JevCompiler.SystemText(request, request.Questions, chunked: false));
    }

    [Theory]
    [InlineData("audio")]
    [InlineData("video")]
    public void NamesTheSupportedArrayForSingularMediaFields(string field)
    {
        using var document = JsonDocument.Parse(
            "{\"state\":\"x\",\"questions\":{\"a\":{\"type\":\"noul\"}},\"" + field + "\":[\"anything\"]}");
        Assert.Contains("attachment array",
            Assert.Throws<JevValidationException>(() => JevRequest.Parse(document.RootElement)).Message);
    }
}
