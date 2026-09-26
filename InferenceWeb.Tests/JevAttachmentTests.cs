// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.IO.Compression;
using System.Text;
using System.Text.Json;
using TensorSharp.Models.Media;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Jev;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace InferenceWeb.Tests;

[CollectionDefinition("Jev attachment media", DisableParallelization = true)]
public sealed class JevAttachmentMediaCollection;

/// <summary>Discovery-time gate, so hosts without symlink permission report skipped coverage.</summary>
public sealed class JevSymlinkFactAttribute : FactAttribute
{
    private static readonly Lazy<string?> SkipReason = new(() =>
    {
        string directory = Path.Combine(Path.GetTempPath(), "ts-jev-symlink-probe-" + Guid.NewGuid().ToString("N"));
        try
        {
            Directory.CreateDirectory(directory);
            string target = Path.Combine(directory, "target");
            File.WriteAllText(target, "probe");
            File.CreateSymbolicLink(Path.Combine(directory, "link"), target);
            return null;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or PlatformNotSupportedException)
        { return "Requires permission and filesystem support for creating symbolic links."; }
        finally { try { Directory.Delete(directory, recursive: true); } catch (IOException) { } }
    });
    public JevSymlinkFactAttribute() => Skip = SkipReason.Value;
}

[Collection("Jev attachment media")]
public sealed class JevAttachmentTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-jev-attachments-" + Guid.NewGuid().ToString("N"));
    private UploadStoragePolicy Storage => new(_directory);
    public JevAttachmentTests() => Directory.CreateDirectory(_directory);
    public void Dispose() => Directory.Delete(_directory, recursive: true);

    private static JevRequest Parse(object value)
    {
        using var json = JsonDocument.Parse(JsonSerializer.Serialize(value));
        return JevRequest.Parse(json.RootElement);
    }
    private static object Questions => new { q = new { type = "noul", instructions = "Does the customer need a refund?" } };
    private static JevRequest Inline(string name, byte[] data, string field = "files") => Parse(new Dictionary<string, object>
    { ["state"] = "Decide from the supplied evidence.", ["questions"] = Questions, [field] = new[] { new { name, data = Convert.ToBase64String(data) } } });

    [Fact]
    public async Task InlineUtf8AndUploadedReferenceProduceSameEvidenceAndReuseCache()
    {
        byte[] bytes = Encoding.UTF8.GetBytes("Customer says: refund my duplicate charge. 中文 evidence.");
        var first = await JevAttachmentPreparer.PrepareAsync(Inline("ticket.txt", bytes), Storage);
        string file = Path.GetFileName(Assert.Single(Directory.GetFiles(_directory)));
        var uploaded = Parse(new { state = "Decide from the supplied evidence.", questions = Questions, files = new[] { new { file, name = "ticket.txt" } } });
        var second = await JevAttachmentPreparer.PrepareAsync(uploaded, Storage);
        Assert.Equal(first.State, second.State);
        Assert.Contains("refund my duplicate charge. 中文", second.State);
        Assert.True(Assert.Single(second.Diagnostics).CacheHit);
        Assert.Empty(second.ImagePaths);
        Assert.Equal(bytes.Length, new FileInfo(Path.Combine(_directory, file)).Length);
    }

    [Theory]
    [InlineData("../secret.txt")]
    [InlineData("/etc/passwd")]
    [InlineData("https://example.com/file.txt")]
    [InlineData("file:///tmp/file.txt")]
    [InlineData("C:\\secret.txt")]
    public void UploadedReferencesRejectPathsAndUrls(string file)
        => Assert.Contains("bare upload filename", Assert.Throws<JevValidationException>(() => Parse(new
        { state = "x", questions = Questions, files = new[] { new { file } } })).Message);

    [Theory]
    [InlineData("files", "archive.zip", "unsupported file type")]
    [InlineData("documents", "movie.mp4", "does not match")]
    [InlineData("videos", "ticket.txt", "does not match")]
    [InlineData("audios", "ticket.txt", "does not match")]
    public void RefusesUnknownAndMismatchedTypes(string field, string name, string reason)
        => Assert.Contains(reason, Assert.Throws<JevValidationException>(() => Inline(name, "data"u8.ToArray(), field)).Message);

    [Fact]
    public void RejectsAmbiguousAndDuplicateSources()
    {
        Assert.Contains("exactly one", Assert.Throws<JevValidationException>(() => Parse(new
        { state = "x", questions = Questions, files = new[] { new { name = "x.txt", data = "eA==", file = "x.txt" } } })).Message);
        using var json = JsonDocument.Parse("""{"state":"x","questions":{"q":{"type":"noul"}},"files":[{"name":"x.txt","data":"eA==","data":"eA=="}]}""");
        Assert.Contains("duplicate", Assert.Throws<JevValidationException>(() => JevRequest.Parse(json.RootElement)).Message);
    }

    [Fact]
    public async Task MissingUploadsNeverReachExtraction()
    {
        var request = Parse(new { state = "x", questions = Questions, files = new[] { new { file = "missing.txt" } } });
        Assert.Contains("missing", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage))).Message);
    }

    [JevSymlinkFact]
    public async Task SymlinkUploadsNeverReachExtraction()
    {
        File.WriteAllText(Path.Combine(_directory, "source.txt"), "refund");
        File.CreateSymbolicLink(Path.Combine(_directory, "link.txt"), Path.Combine(_directory, "source.txt"));
        var request = Parse(new { state = "x", questions = Questions, files = new[] { new { file = "link.txt" } } });
        Assert.Contains("regular uploaded file", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage))).Message);
    }

    [Fact]
    public async Task TextLimitsRejectInsteadOfTruncatingAndIncludeBomBoundary()
    {
        foreach (string prefix in new[] { "", "\uFEFF" })
        {
            var request = Inline("long.txt", Encoding.UTF8.GetBytes(prefix + new string('x', JevAttachmentPreparer.MaxTextCharacters + 1)));
            Assert.Contains("exceeds", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage))).Message);
        }
    }

    [Fact]
    public async Task AggregateTextAndSourceByteLimitsApplyToUploadedReferences()
    {
        File.WriteAllText(Path.Combine(_directory, "large.txt"), new string('x', JevAttachmentPreparer.MaxTextCharacters));
        var request = Parse(new { state = "x", questions = Questions, files = Enumerable.Repeat(new { file = "large.txt" }, 3).ToArray() });
        Assert.Contains("combined extracted text", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage))).Message);
        using (var stream = File.Create(Path.Combine(_directory, "bytes.txt"))) stream.SetLength(JevAttachmentInput.MaxAttachmentBytes + 1L);
        request = Parse(new { state = "x", questions = Questions, files = new[] { new { file = "bytes.txt" } } });
        Assert.Contains("must contain", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage))).Message);
    }

    [Theory]
    [InlineData(new byte[] { 0xFF, 0xFE, 0x00 })]
    [InlineData(new byte[] { 0x68, 0x00, 0x69 })]
    public async Task RejectsBinaryAndInvalidUtf8(byte[] bytes)
        => await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("bad.txt", bytes), Storage));

    [Fact]
    public async Task OfficeExtractionPreservesRunsSlidesAndSharedStrings()
    {
        byte[] docx = Zip(("word/document.xml", "<document><p><r><t>Re</t></r><r><t>fund</t></r></p><p><t>duplicate charge</t></p></document>"));
        Assert.Contains("Refund\nduplicate charge", (await JevAttachmentPreparer.PrepareAsync(Inline("ticket.docx", docx), Storage)).State);
        byte[] pptx = Zip(("ppt/slides/slide1.xml", "<slide><p><t>Second</t></p></slide>"), ("ppt/slides/slide2.xml", "<slide><p><t>First</t></p></slide>"),
            ("ppt/presentation.xml", "<presentation xmlns:r='urn:rels'><sldIdLst><sldId id='257' r:id='rId2'/><sldId id='256' r:id='rId1'/></sldIdLst></presentation>"),
            ("ppt/_rels/presentation.xml.rels", "<Relationships><Relationship Id='rId1' Target='slides/slide1.xml'/><Relationship Id='rId2' Target='slides/slide2.xml'/></Relationships>"));
        string slides = (await JevAttachmentPreparer.PrepareAsync(Inline("deck.pptx", pptx), Storage)).State;
        Assert.True(slides.IndexOf("First", StringComparison.Ordinal) < slides.IndexOf("Second", StringComparison.Ordinal));
        byte[] xlsx = Zip(("xl/workbook.xml", "<workbook xmlns:r='urn:rels'><sheets><sheet name='Tickets' r:id='rId1'/></sheets></workbook>"),
            ("xl/_rels/workbook.xml.rels", "<Relationships><Relationship Id='rId1' Target='worksheets/sheet1.xml'/></Relationships>"),
            ("xl/sharedStrings.xml", "<sst><si><t>refund</t></si></sst>"),
            ("xl/worksheets/sheet1.xml", "<worksheet><sheetData><row><c r='A1' t='s'><v>0</v></c><c r='B1'><f>1+2</f><v>3</v></c></row></sheetData></worksheet>"));
        string sheet = (await JevAttachmentPreparer.PrepareAsync(Inline("tickets.xlsx", xlsx), Storage)).State;
        Assert.Contains("Sheet Tickets:\nA1=refund\tB1=3", sheet);
    }

    [Fact]
    public async Task OfficeRejectsExpandedArchivesAndExternalEntities()
    {
        byte[] bomb = Zip(("word/document.xml", new string('x', (int)JevAttachmentOfficeText.MaxExpandedBytes + 1)));
        Assert.Contains("expanded bytes", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("bomb.docx", bomb), Storage))).Message);
        byte[] entity = Zip(("word/document.xml", "<!DOCTYPE x [<!ENTITY payload SYSTEM 'file:///etc/passwd'>]><document><p><t>&payload;</t></p></document>"));
        await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("entity.docx", entity), Storage));
    }

    [Fact]
    public async Task OfficeKeepsExplicitLineBreaksAndIgnoresUnreferencedParts()
    {
        byte[] pptx = Zip(("ppt/presentation.xml", "<presentation xmlns:r='urn:rels'><sldIdLst><sldId r:id='r1'/></sldIdLst></presentation>"),
            ("ppt/_rels/presentation.xml.rels", "<Relationships><Relationship Id='r1' Target='slides/slide1.xml'/></Relationships>"),
            ("ppt/slides/slide1.xml", "<slide><p><t>not</t><br/><t>approved</t></p></slide>"));
        Assert.Contains("not\napproved", (await JevAttachmentPreparer.PrepareAsync(Inline("break.pptx", pptx), Storage)).State);
        byte[] docx = Zip(("word/document.xml", "<document xmlns:r='urn:rels'><p><t>Refund requested.</t></p><sectPr><headerReference r:id='h1'/></sectPr><footnoteReference id='1'/></document>"),
            ("word/_rels/document.xml.rels", "<Relationships><Relationship Id='h1' Target='header2.xml'/></Relationships>"),
            ("word/header2.xml", "<header><p><t>Active policy.</t></p></header>"),
            ("word/header1.xml", "<header><p><t>ORPHAN HEADER</t></p></header>"),
            ("word/footer1.xml", "<footer><p><t>ORPHAN FOOTER</t></p></footer>"),
            ("word/footnotes.xml", "<footnotes><footnote id='1'><p><t>Active footnote.</t></p></footnote><footnote id='2'><p><t>ORPHAN FOOTNOTE</t></p></footnote></footnotes>"));
        string state = (await JevAttachmentPreparer.PrepareAsync(Inline("headers.docx", docx), Storage)).State;
        Assert.Contains("Active policy.", state);
        Assert.Contains("Active footnote.", state);
        Assert.DoesNotContain("ORPHAN", state);
    }

    [Theory]
    [InlineData("<Relationship Target='slides/slide1.xml'/>")]
    [InlineData("<Relationship Id='rId1'/>")]
    [InlineData("<Relationship Id='rId1' Target='slides/slide1.xml'/><Relationship Id='rId1' Target='slides/slide2.xml'/>")]
    public async Task OfficeMalformedRelationshipsBecomeValidationErrors(string relationship)
    {
        byte[] pptx = Zip(("ppt/presentation.xml", "<presentation/>"),
            ("ppt/_rels/presentation.xml.rels", "<Relationships>" + relationship + "</Relationships>"));
        Assert.Contains("relationships", (await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("bad.pptx", pptx), Storage))).Message);
    }

    [Fact]
    public async Task PdfTextAndScannedPagesProduceCompleteEvidenceWithCoverageNotes()
    {
        var text = await JevAttachmentPreparer.PrepareAsync(Inline("ticket.pdf", Pdf(2, scanned: false), "documents"), Storage);
        Assert.Contains("Customer requests a refund for the duplicate charge.", text.State);
        Assert.Empty(text.ImagePaths);
        Assert.Contains("text layer", Assert.Single(text.Diagnostics).Warning);
        var scan = await JevAttachmentPreparer.PrepareAsync(Inline("scan.pdf", Pdf(2, scanned: true), "documents"), Storage);
        Assert.Equal(2, scan.ImagePaths.Length);
        Assert.Contains("Image 1 (page 1)", scan.State);
        Assert.Contains("Image 2 (page 2)", scan.State);
        Assert.Contains("largest embedded image", Assert.Single(scan.Diagnostics).Warning);
    }

    [Fact]
    public async Task PdfRejectsPageAndImageOverflowsAndUnrenderablePages()
    {
        await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("long.pdf", Pdf(JevAttachmentPreparer.MaxPdfPages + 1, scanned: false)), Storage));
        await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("scan.pdf", Pdf(JevRequest.MaxImages + 1, scanned: true)), Storage));
        var blank = new PdfDocumentBuilder();
        blank.AddPage(PageSize.A4);
        await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("blank.pdf", blank.Build()), Storage));
        await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("broken.pdf", "%PDF-broken"u8.ToArray()), Storage));
    }

    [Fact]
    public async Task AudioRequiresTranscriberAndPassesCancellationAndActualStoredBytes()
    {
        var request = Inline("ticket.wav", "RIFFfake test wave"u8.ToArray(), "audios");
        await Assert.ThrowsAsync<JevModelUnavailableException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage));
        using var cancellation = new CancellationTokenSource();
        var prepared = await JevAttachmentPreparer.PrepareAsync(request, Storage, (path, token) =>
        {
            Assert.Equal(cancellation.Token, token);
            Assert.Equal("RIFFfake test wave", File.ReadAllText(path));
            return Task.FromResult("Please refund the duplicate charge.");
        }, cancellation.Token);
        Assert.Contains("Please refund", prepared.State);
        Assert.Contains("non-speech", Assert.Single(prepared.Diagnostics).Warning);
        cancellation.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => JevAttachmentPreparer.PrepareAsync(request, Storage, cancellationToken: cancellation.Token));
    }

    [Fact]
    public async Task VideoSamplesBoundedOrderedTimestampedFramesAndCachesDecoding()
    {
        var previous = MediaCodecs.Video;
        var decoder = new FakeVideoDecoder();
        MediaCodecs.Video = decoder;
        try
        {
            var request = Inline("clip.mp4", "video fixture"u8.ToArray(), "videos");
            var storage = Storage;
            var first = await JevAttachmentPreparer.PrepareAsync(request, storage);
            Assert.Equal(4, first.ImagePaths.Length);
            Assert.Contains("Image 1 at 0 seconds", first.State);
            Assert.Contains("Image 4 at 9 seconds", first.State);
            Assert.True(Assert.Single(first.Diagnostics).Sampled);
            Assert.Equal(new[] { 0, 30, 60, 90 }, decoder.Requested);
            long used = storage.UsedBytes;
            var second = await JevAttachmentPreparer.PrepareAsync(request, storage);
            Assert.True(Assert.Single(second.Diagnostics).CacheHit);
            Assert.Equal(1, decoder.ReadCalls);
            Assert.Equal(used, storage.UsedBytes);
            File.Delete(first.ImagePaths[0]);
            var third = await JevAttachmentPreparer.PrepareAsync(request, Storage);
            Assert.False(Assert.Single(third.Diagnostics).CacheHit);
            Assert.Equal(2, decoder.ReadCalls);
        }
        finally { MediaCodecs.Video = previous; }
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task CorruptAndPartialVideoDecodesBecomeValidationErrors(bool throwOnProbe)
    {
        var previous = MediaCodecs.Video;
        MediaCodecs.Video = new FakeVideoDecoder { ThrowOnProbe = throwOnProbe, PartialRead = !throwOnProbe };
        try
        {
            await Assert.ThrowsAsync<JevValidationException>(() => JevAttachmentPreparer.PrepareAsync(Inline("bad.mp4", "bad video fixture"u8.ToArray(), "videos"), Storage));
        }
        finally { MediaCodecs.Video = previous; }
    }

    private sealed class FakeVideoDecoder : IVideoDecoder
    {
        internal int ReadCalls;
        internal int[] Requested = [];
        internal bool ThrowOnProbe, PartialRead;
        public VideoInfo Probe(string path) => ThrowOnProbe ? throw new Exception("corrupt codec data") : new(10, 100, 2, 2);
        public void ReadFrames(string path, IReadOnlyList<int> frameIndices, FrameCallback onFrame)
        {
            ReadCalls++;
            Requested = frameIndices.ToArray();
            foreach (int index in PartialRead ? frameIndices.Take(1) : frameIndices) onFrame(index, Enumerable.Repeat((byte)index, 12).ToArray(), 2, 2, 6, PixelLayout.Rgb);
        }
    }

    private static byte[] Zip(params (string Name, string Content)[] entries)
    {
        using var buffer = new MemoryStream();
        using (var archive = new ZipArchive(buffer, ZipArchiveMode.Create, leaveOpen: true))
            foreach (var (name, content) in entries)
            {
                using var writer = new StreamWriter(archive.CreateEntry(name).Open(), new UTF8Encoding(false));
                writer.Write(content);
            }
        return buffer.ToArray();
    }

    private static byte[] Pdf(int pages, bool scanned)
    {
        var builder = new PdfDocumentBuilder();
        var font = builder.AddStandard14Font(Standard14Font.Helvetica);
        byte[] png = Convert.FromBase64String("iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGM8ISfHwMDAxAAGAA0EAQijE05aAAAAAElFTkSuQmCC");
        for (int i = 0; i < pages; i++)
        {
            var page = builder.AddPage(PageSize.A4);
            if (scanned) page.AddPng(png, new PdfRectangle(40, 40, 200, 200));
            else page.AddText("Customer requests a refund for the duplicate charge.", 12m, new PdfPoint(25, 800), font);
        }
        return builder.Build();
    }
}
