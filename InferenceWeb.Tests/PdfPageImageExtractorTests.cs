using System;
using System.IO;
using System.IO.Compression;
using TensorSharp.Models;
using TensorSharp.Server.ProtocolAdapters;
using UglyToad.PdfPig.Content;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace InferenceWeb.Tests;

public class PdfPageImageExtractorTests
{
    [Fact]
    public void IncompletePageImageWarning_IsNullOnlyWhenEveryPageIsAvailable()
    {
        Assert.Null(WebUiAdapter.BuildIncompletePdfImageWarning(134, 134));

        string warning = WebUiAdapter.BuildIncompletePdfImageWarning(133, 134);
        Assert.Contains("133 of 134", warning);
        Assert.Contains("will not be sent", warning);
        Assert.Contains("TS_PDF_MAX_PAGES", warning);
    }

    // Builds an image-only PDF: N pages, each holding one embedded PNG and no text — the
    // shape of a scanned document or a slide deck exported as page images.
    private static byte[] BuildImageOnlyPdf(int pages, int w = 96, int h = 64)
    {
        byte[] png = MakeSolidPng(w, h);
        var builder = new PdfDocumentBuilder();
        for (int i = 0; i < pages; i++)
        {
            PdfPageBuilder page = builder.AddPage(PageSize.A4);
            page.AddPng(png, new PdfRectangle(40, 40, 40 + w, 40 + h));
        }
        return builder.Build();
    }

    // Builds a PDF whose pages carry an embedded image AND real text (born-digital).
    private static byte[] BuildImageAndTextPdf(string text, int w = 96, int h = 64)
    {
        byte[] png = MakeSolidPng(w, h);
        var builder = new PdfDocumentBuilder();
        PdfDocumentBuilder.AddedFont font = builder.AddStandard14Font(Standard14Font.Helvetica);
        PdfPageBuilder page = builder.AddPage(PageSize.A4);
        page.AddPng(png, new PdfRectangle(40, 600, 40 + w, 600 + h));
        page.AddText(text, 12m, new PdfPoint(40, 500), font);
        return builder.Build();
    }

    [Fact]
    public void ImageOnlyPdf_LooksTextless()
    {
        byte[] pdf = BuildImageOnlyPdf(3);

        PdfTextResult text = PdfTextExtractor.ExtractFromBytes(pdf);

        Assert.Equal(3, text.PageCount);
        Assert.True(text.LooksTextless, "an image-only PDF should be detected as textless");
        Assert.Equal(0, text.NonWhitespaceCharCount);
    }

    [Fact]
    public void ExtractPageImages_RecoversOnePngPerPage()
    {
        byte[] pdf = BuildImageOnlyPdf(3);
        string dir = Path.Combine(Path.GetTempPath(), $"ts-pdfimg-{Guid.NewGuid():N}");
        try
        {
            PdfImageResult res = PdfPageImageExtractor.ExtractPageImagesFromBytes(pdf, dir, namePrefix: "doc");

            Assert.Equal(3, res.PageCount);
            Assert.Equal(3, res.ExtractedPageCount);
            Assert.Equal(3, res.ImagePaths.Count);
            foreach (string p in res.ImagePaths)
            {
                Assert.True(File.Exists(p), $"expected saved image {p}");
                byte[] bytes = File.ReadAllBytes(p);
                Assert.True(bytes.Length > 8);
                // PNG signature.
                Assert.Equal(0x89, bytes[0]);
                Assert.Equal(0x50, bytes[1]); // 'P'
                Assert.Equal(0x4E, bytes[2]); // 'N'
                Assert.Equal(0x47, bytes[3]); // 'G'
            }
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void ExtractPageImages_HonorsMaxPagesCap()
    {
        byte[] pdf = BuildImageOnlyPdf(5);
        string dir = Path.Combine(Path.GetTempPath(), $"ts-pdfimg-{Guid.NewGuid():N}");
        try
        {
            PdfImageResult res = PdfPageImageExtractor.ExtractPageImagesFromBytes(pdf, dir, maxPages: 2, namePrefix: "doc");

            Assert.Equal(5, res.PageCount);           // total in the document
            Assert.Equal(2, res.ExtractedPageCount);  // capped
            Assert.Equal(2, res.ImagePaths.Count);
        }
        finally
        {
            if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true);
        }
    }

    [Fact]
    public void RasterBudgetRejectsBeforeWritingAndCancellationIsPreserved()
    {
        byte[] pdf = BuildImageOnlyPdf(1, w: 96, h: 64);
        string dir = Path.Combine(Path.GetTempPath(), $"ts-pdfimg-{Guid.NewGuid():N}");
        try
        {
            var error = Assert.Throws<InvalidDataException>(() =>
                PdfPageImageExtractor.ExtractPageImagesFromBytes(pdf, dir, 0, null, null, maxImagePixels: 96 * 64 - 1, cancellationToken: default));
            Assert.Contains("pixels", error.Message);
            Assert.Empty(Directory.GetFiles(dir));
            var result = PdfPageImageExtractor.ExtractPageImagesFromBytes(pdf, dir, 0, null, null, maxImagePixels: 96 * 64, cancellationToken: default);
            Assert.Single(result.ImagePaths);
            using var cancellation = new System.Threading.CancellationTokenSource();
            cancellation.Cancel();
            Assert.ThrowsAny<OperationCanceledException>(() =>
                PdfPageImageExtractor.ExtractPageImagesFromBytes(pdf, dir, 0, null, null, 0, cancellationToken: cancellation.Token));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void ImageAndTextPdf_IsNotTextless()
    {
        byte[] pdf = BuildImageAndTextPdf("BornDigitalWithFigure");

        PdfTextResult text = PdfTextExtractor.ExtractFromBytes(pdf);

        Assert.False(text.LooksTextless, "a PDF with a real text layer must use the text path");
        Assert.Contains("BornDigitalWithFigure", text.Text.Replace(" ", string.Empty));
    }

    [Theory]
    [InlineData(0, 1, true)]     // no text, 1 page -> textless
    [InlineData(3, 5, true)]     // 3 non-ws chars over 5 pages -> textless (stray page numbers)
    [InlineData(400, 1, false)]  // dense page -> has text
    [InlineData(50, 2, false)]   // 25 chars/page -> has text
    public void LooksTextless_ThresholdScalesWithPageCount(int nonWhitespace, int pageCount, bool expected)
    {
        var r = new PdfTextResult { PageCount = pageCount, NonWhitespaceCharCount = nonWhitespace };
        Assert.Equal(expected, r.LooksTextless);
    }

    // ---- minimal self-contained PNG encoder (solid gray RGBA) --------------

    private static byte[] MakeSolidPng(int w, int h)
    {
        int stride = 1 + w * 4;
        byte[] raw = new byte[h * stride];
        for (int y = 0; y < h; y++)
        {
            raw[y * stride] = 0; // filter: none
            for (int x = 0; x < w; x++)
            {
                int o = y * stride + 1 + x * 4;
                raw[o] = 120; raw[o + 1] = 130; raw[o + 2] = 140; raw[o + 3] = 255;
            }
        }

        byte[] idat;
        using (var ms = new MemoryStream())
        {
            ms.WriteByte(0x78); ms.WriteByte(0x01); // zlib header
            using (var d = new DeflateStream(ms, CompressionLevel.Fastest, true))
                d.Write(raw, 0, raw.Length);
            uint a = 1, b = 0;
            foreach (byte bb in raw) { a = (a + bb) % 65521; b = (b + a) % 65521; }
            uint adler = (b << 16) | a;
            ms.WriteByte((byte)(adler >> 24)); ms.WriteByte((byte)(adler >> 16));
            ms.WriteByte((byte)(adler >> 8)); ms.WriteByte((byte)adler);
            idat = ms.ToArray();
        }

        using var outMs = new MemoryStream();
        outMs.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A }, 0, 8);
        WriteChunk(outMs, "IHDR", Ihdr(w, h));
        WriteChunk(outMs, "IDAT", idat);
        WriteChunk(outMs, "IEND", Array.Empty<byte>());
        return outMs.ToArray();
    }

    private static byte[] Ihdr(int w, int h)
    {
        byte[] d = new byte[13];
        d[0] = (byte)(w >> 24); d[1] = (byte)(w >> 16); d[2] = (byte)(w >> 8); d[3] = (byte)w;
        d[4] = (byte)(h >> 24); d[5] = (byte)(h >> 16); d[6] = (byte)(h >> 8); d[7] = (byte)h;
        d[8] = 8;  // bit depth
        d[9] = 6;  // color type RGBA
        return d;
    }

    private static void WriteChunk(Stream s, string type, byte[] data)
    {
        byte[] len = { (byte)(data.Length >> 24), (byte)(data.Length >> 16), (byte)(data.Length >> 8), (byte)data.Length };
        byte[] typeBuf = System.Text.Encoding.ASCII.GetBytes(type);
        s.Write(len, 0, 4);
        s.Write(typeBuf, 0, 4);
        if (data.Length > 0) s.Write(data, 0, data.Length);
        uint crc = Crc32(typeBuf, data);
        byte[] crcBuf = { (byte)(crc >> 24), (byte)(crc >> 16), (byte)(crc >> 8), (byte)crc };
        s.Write(crcBuf, 0, 4);
    }

    private static readonly uint[] CrcTable = BuildCrcTable();
    private static uint[] BuildCrcTable()
    {
        var t = new uint[256];
        for (uint n = 0; n < 256; n++)
        {
            uint c = n;
            for (int k = 0; k < 8; k++)
                c = (c & 1) != 0 ? 0xEDB88320 ^ (c >> 1) : c >> 1;
            t[n] = c;
        }
        return t;
    }

    private static uint Crc32(byte[] type, byte[] data)
    {
        uint crc = 0xFFFFFFFF;
        foreach (byte b in type) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        foreach (byte b in data) crc = CrcTable[(crc ^ b) & 0xFF] ^ (crc >> 8);
        return crc ^ 0xFFFFFFFF;
    }
}
