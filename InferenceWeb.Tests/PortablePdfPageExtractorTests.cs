using System.Buffers.Binary;
using System.Text;
using TensorSharp.Knowledge;
using TensorSharp.Models.Media;
using TensorSharp.Server.Knowledge;
using UglyToad.PdfPig;

namespace InferenceWeb.Tests;

public sealed class PortablePdfPageExtractorTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "portable-pdf-test-" + Guid.NewGuid().ToString("N"));
    public PortablePdfPageExtractorTests() => Directory.CreateDirectory(_directory);

    [Theory]
    [InlineData("tesseract 4.1.1", false)]
    [InlineData("tesseract 5.0.0", true)]
    [InlineData("tesseract 5.5.1-rc1", true)]
    [InlineData("tesseract 10.2.3", true)]
    [InlineData("tesseract v5.5.0.20241111", true)]
    [InlineData("tesseract v10.0.0", true)]
    [InlineData("tesseract v4.1.1", false)]
    [InlineData("tesseract 2147483648.0", false)]
    [InlineData("pretend tesseract 5.0", false)]
    [InlineData("tesseract 50junk", false)]
    public void TesseractVersionRequiresAtLeastFiveAndAllowsFutureMajorVersions(string version, bool supported)
        => Assert.Equal(supported, PortablePdfPageExtractor.IsSupportedVersion(version));

    [Theory]
    [InlineData("../../secret")]
    [InlineData("eng+../../secret")]
    [InlineData("--tessdata-dir")]
    [InlineData("eng\n--oem 0")]
    [InlineData("eng++chi_sim")]
    [InlineData("")]
    public void LanguageNamesCannotContainPathsOrArguments(string languages)
        => Assert.Throws<ArgumentException>(() => new PortablePdfPageExtractor(new() { Languages = languages }));

    [Theory]
    [InlineData("eng+chi_sim")]
    [InlineData("script/HanS+eng")]
    public void NamedLanguageAndScriptPacksAreAccepted(string languages)
        => Assert.NotNull(new PortablePdfPageExtractor(new() { Languages = languages }));

    [Fact]
    public void DefaultLanguagesUseInstalledPacksAndMissingRequestedDataFailsClearly()
    {
        const string listing = "List of available languages in \"/test/tessdata\" (4):\nchi_sim\neng\nosd\nfra\n../../secret\n";
        Assert.Equal(new[] { "eng", "chi_sim" }, PortablePdfPageExtractor.SelectLanguages(listing, null));
        Assert.Equal(new[] { "fra" }, PortablePdfPageExtractor.SelectLanguages("osd\nfra\n", null));
        Assert.Throws<NotSupportedException>(() => PortablePdfPageExtractor.SelectLanguages(listing, "eng+chi_tra"));
        Assert.Throws<NotSupportedException>(() => PortablePdfPageExtractor.SelectLanguages("osd\n../../secret\n", null));
    }

    [Fact]
    public async Task FingerprintTracksInstalledVerticalCompanionsButNotUnrelatedLanguages()
    {
        await File.WriteAllTextAsync(Path.Combine(_directory, "chi_sim.traineddata"), "primary test bytes");
        string vertical = Path.Combine(_directory, "chi_sim_vert.traineddata");
        await File.WriteAllTextAsync(vertical, "vertical test bytes");
        var before = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "chi_sim" });
        Assert.Equal(2, before.Files.Length);
        await File.WriteAllTextAsync(Path.Combine(_directory, "fra.traineddata"), "unrelated test bytes");
        var unrelated = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "chi_sim" });
        Assert.Equal(before.Sha256, unrelated.Sha256);
        Assert.All(before.Files, file => Assert.True(file.IsCurrent()));
        await File.WriteAllTextAsync(vertical, "different vertical model test bytes");
        Assert.Contains(before.Files, file => !file.IsCurrent());
        var changed = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "chi_sim" });
        Assert.NotEqual(before.Sha256, changed.Sha256);
    }

    [Fact]
    public async Task InstallingOrRemovingVerticalCompanionInvalidatesPreviouslyCapturedIdentity()
    {
        string primary = Path.Combine(_directory, "jpn.traineddata"), vertical = Path.Combine(_directory, "jpn_vert.traineddata");
        await File.WriteAllTextAsync(primary, "Japanese primary test bytes");
        var absent = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "jpn" });
        Assert.Contains(absent.Files, file => file.Path == vertical && file.Length == -1 && file.IsCurrent());
        await File.WriteAllTextAsync(vertical, "Japanese vertical test bytes");
        Assert.Contains(absent.Files, file => !file.IsCurrent());
        var installed = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "jpn" });
        Assert.NotEqual(absent.Sha256, installed.Sha256);
        File.Delete(vertical);
        Assert.Contains(installed.Files, file => !file.IsCurrent());
        var removed = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "jpn" });
        Assert.Equal(absent.Sha256, removed.Sha256);
    }

    [Fact]
    public async Task ScriptCompanionsAreTrackedAndExplicitVerticalLanguagesRemainRequired()
    {
        Directory.CreateDirectory(Path.Combine(_directory, "script"));
        await File.WriteAllTextAsync(Path.Combine(_directory, "script", "HanS.traineddata"), "primary script test bytes");
        var absent = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "script/HanS" });
        Assert.Contains(absent.Files, file => file.Path.EndsWith("HanS_vert.traineddata", StringComparison.Ordinal) && file.Length == -1);
        await Assert.ThrowsAsync<NotSupportedException>(() => PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "script/HanS", "script/HanS_vert" }));
        await File.WriteAllTextAsync(Path.Combine(_directory, "script", "HanS_vert.traineddata"), "vertical script test bytes");
        var explicitVertical = await PortablePdfPageExtractor.FingerprintLanguageDataAsync(_directory, new[] { "script/HanS", "script/HanS_vert" });
        Assert.Equal(2, explicitVertical.Files.Length); // No duplicate or nonexistent _vert_vert dependency.
        Assert.DoesNotContain(explicitVertical.Files, file => file.Path.Contains("_vert_vert", StringComparison.Ordinal));
    }

    [Fact]
    public async Task MissingLanguageDirectoryFailsBeforeStartingUtilities()
    {
        var provider = new PortablePdfPageExtractor(new()
        {
            PdfToPpmExecutable = Environment.ProcessPath, TesseractExecutable = Environment.ProcessPath,
            TessdataDirectory = Path.Combine(_directory, "missing-language-data")
        });
        var error = await Assert.ThrowsAsync<NotSupportedException>(() => provider.ExtractPageAsync(Context(CopyFixture("scanned-four-pages.pdf"))));
        Assert.Contains("language-data directory", error.Message);
    }

    [Fact]
    public async Task ExplicitlyDisabledProviderDoesNotOpenFilesOrRequireUtilities()
    {
        var provider = new PdfKnowledgePageExtractor(new()
        {
            Provider = "none", PdfToPpmExecutable = "missing-renderer", TesseractExecutable = "missing-ocr"
        });
        var result = await provider.ExtractPageAsync(Context("must-not-be-opened.pdf"));
        Assert.Null(result.Text);
        Assert.Contains(result.Warnings, message => message.Contains("disabled", StringComparison.OrdinalIgnoreCase));
    }

    [ApplePdfOcrFact]
    public async Task AutomaticAppleProviderWorksWithoutPortableDependencies()
    {
        var provider = new PdfKnowledgePageExtractor(new()
        {
            Provider = "auto", PdfToPpmExecutable = "missing-renderer", TesseractExecutable = "missing-ocr"
        });
        var result = await provider.ExtractPageAsync(Context(CopyFixture("scanned-four-pages.pdf")));
        Assert.Contains("CEDAR-4821", result.Text);
        Assert.StartsWith("apple-vision", result.ExtractorId);
    }

    [Fact]
    public void ProviderCeilingDoesNotSilentlyOverrideTheMaximumCoreDeadline()
    {
        Assert.Equal(600, new PdfOcrOptions().TimeoutSeconds);
        Assert.Throws<ArgumentOutOfRangeException>(() => new PortablePdfPageExtractor(new() { TimeoutSeconds = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PortablePdfPageExtractor(new() { TimeoutSeconds = 601 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PortablePdfPageExtractor(new() { MaxInputBytes = 0 }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new PortablePdfPageExtractor(new() { MaxEmbeddedImagePixels = long.MaxValue }));
    }

    [Fact]
    public void MalformedUtf8IsARecoverablePageErrorAndValidChineseIsPreserved()
    {
        Assert.Throws<InvalidDataException>(() => PortablePdfPageExtractor.DecodeUtf8(new byte[] { 0xC0, 0xAF }));
        Assert.Equal("竹林项目 CEDAR-4821", PortablePdfPageExtractor.DecodeUtf8(Encoding.UTF8.GetBytes("竹林项目 CEDAR-4821")));
    }

    [Fact]
    public void PageSizingKeepsTheWholeRotatedAspectRatioWithinPixelBudget()
    {
        Assert.Equal((2499, 1665), PortablePdfPageExtractor.CalculateRasterSize(600, 400, false, 300, 16_777_216));
        Assert.Equal((1665, 2499), PortablePdfPageExtractor.CalculateRasterSize(600, 400, true, 300, 16_777_216));
        foreach (bool rotated in new[] { false, true })
        {
            var size = PortablePdfPageExtractor.CalculateRasterSize(600, 400, rotated, 300, 65_536);
            Assert.InRange((long)size.Width * size.Height, 63_000, 65_536);
            Assert.InRange((double)size.Width / size.Height, rotated ? .66 : 1.49, rotated ? .68 : 1.51);
        }
        Assert.Throws<InvalidDataException>(() => PortablePdfPageExtractor.CalculateRasterSize(double.NaN, 400, false, 300, 65_536));
        Assert.Throws<InvalidDataException>(() => PortablePdfPageExtractor.CalculateRasterSize(-1, 400, false, 300, 65_536));
    }

    [Fact]
    public void PngDimensionsAndHeaderCannotBypassRasterBounds()
    {
        byte[] png = MakePngHeader(200, 300);
        PortablePdfPageExtractor.ValidatePng(png, 65_536, 200, 300);
        Assert.Throws<KnowledgeLimitException>(() => PortablePdfPageExtractor.ValidatePng(png, 59_999, 200, 300));
        Assert.Throws<KnowledgeLimitException>(() => PortablePdfPageExtractor.ValidatePng(png, 65_536, 199, 300));
        Assert.Throws<KnowledgeLimitException>(() => PortablePdfPageExtractor.ValidatePng(MakePngHeader(uint.MaxValue, uint.MaxValue), 65_536, int.MaxValue, int.MaxValue));
        Assert.Throws<KnowledgeLimitException>(() => PortablePdfPageExtractor.ValidatePng(MakePngHeader(0, 1), 65_536, 200, 300));
        png[11] = 12; // Invalid IHDR chunk length, even though the dimensions look valid.
        Assert.Throws<InvalidDataException>(() => PortablePdfPageExtractor.ValidatePng(png, 65_536, 200, 300));
    }

    [Fact]
    public async Task EmptyOrOversizedFilesAreRejectedBeforeDependencyDiscovery()
    {
        string path = Path.Combine(_directory, "input.pdf");
        await File.WriteAllBytesAsync(path, Array.Empty<byte>());
        var provider = new PortablePdfPageExtractor(new() { PdfToPpmExecutable = Path.Combine(_directory, "does-not-exist"), MaxInputBytes = 2 });
        await Assert.ThrowsAsync<InvalidDataException>(() => provider.ExtractPageAsync(Context(path)));
        await File.WriteAllBytesAsync(path, new byte[] { 1, 2, 3 });
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => provider.ExtractPageAsync(Context(path)));
    }

    [Fact]
    public async Task DirectRendererRejectsAggregateImagesBeforeLaunchingTheUtility()
    {
        string path = CopyFixture("mixed-two-pages.pdf");
        using var pdf = PdfDocument.Open(path);
        long[] pixels = pdf.GetPage(2).GetImages().Select(image => (long)image.WidthInSamples * image.HeightInSamples).ToArray();
        Assert.Equal(2, pixels.Length);
        long maximum = pixels.Max();
        Assert.True(pixels.Sum() > maximum);
        var provider = new PortablePdfPageExtractor(new()
        {
            // A real existing executable satisfies resolution. The test must fail before
            // launching it, because each image is legal but the sum exceeds the budget.
            PdfToPpmExecutable = Environment.ProcessPath,
            MaxEmbeddedImagePixels = maximum
        });
        await Assert.ThrowsAsync<KnowledgeLimitException>(() => provider.RenderPageAsync(Context(path, 2)));
    }

    [Fact]
    public void MissingExplicitExecutableDoesNotFallBackToAnUnrelatedPathEntry()
        => Assert.Throws<NotSupportedException>(() => PortablePdfPageExtractor.ResolveExecutable(
            Path.Combine(_directory, "missing-pdftoppm"), "pdftoppm"));

    [PopplerPdfFact]
    public async Task ActualPopplerPreservesCropCornersAndOrientationAtFullAndLimitedResolution()
    {
        string path = CopyFixture("crop-rotation.pdf");
        var provider = new PortablePdfPageExtractor(new()
        {
            PdfToPpmExecutable = Environment.GetEnvironmentVariable("TS_KNOWLEDGE_PDFTOPPM_PATH")
        });
        foreach (int pixels in new[] { 16_777_216, 65_536 })
        for (int page = 1; page <= 4; page++)
        {
            byte[] png = await provider.RenderPageAsync(Context(path, page, pixels));
            byte[] rgba = ManagedMediaProvider.Instance.DecodeRgba(png, out int width, out int height);
            Assert.True(page % 2 == 1 ? width > height : height > width);
            Assert.InRange((long)width * height, 1, pixels);
            string[] expected = page switch
            {
                1 => ["blue", "yellow", "red", "green"],
                2 => ["red", "blue", "green", "yellow"],
                3 => ["green", "red", "yellow", "blue"],
                _ => ["yellow", "green", "blue", "red"]
            };
            (int X, int Y)[] corners = [(width / 20, height / 20), (width * 19 / 20, height / 20),
                (width / 20, height * 19 / 20), (width * 19 / 20, height * 19 / 20)];
            for (int i = 0; i < corners.Length; i++)
            {
                byte[] color = expected[i] switch
                {
                    "red" => [255, 0, 0, 255], "green" => [0, 255, 0, 255],
                    "blue" => [0, 0, 255, 255], _ => [255, 255, 0, 255]
                };
                Assert.Equal(color, rgba.AsSpan((corners[i].Y * width + corners[i].X) * 4, 4).ToArray());
            }
        }
    }

    private static KnowledgePdfPageContext Context(string path, int page = 1, int pixels = 16_777_216)
        => new(path, Path.GetFileName(path), page, 20000, pixels);
    private static byte[] MakePngHeader(uint width, uint height)
    {
        byte[] png = new byte[33];
        new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 }.CopyTo(png, 0);
        BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(8), 13);
        "IHDR"u8.CopyTo(png.AsSpan(12));
        BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(16), width);
        BinaryPrimitives.WriteUInt32BigEndian(png.AsSpan(20), height);
        return png;
    }
    private string CopyFixture(string fileName)
    {
        string path = Path.Combine(_directory, fileName);
        using var source = typeof(PortablePdfPageExtractorTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Knowledge." + fileName)!;
        using var target = File.Create(path);
        source.CopyTo(target);
        return path;
    }
    public void Dispose() => PortablePdfScratchDirectory.Delete(_directory);
}

internal sealed class PopplerPdfFactAttribute : FactAttribute
{
    public PopplerPdfFactAttribute()
    {
        try { PortablePdfPageExtractor.ResolveExecutable(Environment.GetEnvironmentVariable("TS_KNOWLEDGE_PDFTOPPM_PATH"), "pdftoppm"); }
        catch (NotSupportedException) { Skip = "Requires installed Poppler pdftoppm; no renderer coverage is claimed when unavailable."; }
    }
}
