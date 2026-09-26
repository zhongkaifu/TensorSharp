// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace TensorSharp.Models
{
    /// <summary>Page images extracted from a PDF, ready to hand to a vision model.</summary>
    public sealed class PdfImageResult
    {
        /// <summary>Absolute paths of the saved page images, in page order.</summary>
        public IReadOnlyList<string> ImagePaths { get; init; }

        /// <summary>Total number of pages in the source document.</summary>
        public int PageCount { get; init; }

        /// <summary>Number of pages that yielded at least one saved image.</summary>
        public int ExtractedPageCount { get; init; }
    }

    /// <summary>
    /// Extracts the embedded page images from a PDF and writes them out as PNG files.
    ///
    /// This is the ingestion path for scanned / image-only PDFs — documents whose pages
    /// carry no selectable text layer (see <see cref="PdfTextResult.LooksTextless"/>). For
    /// those, each page is typically a single full-page raster (a scan or an exported
    /// slide), so we take the largest image on each page as that page's picture and let a
    /// vision model read it — mirroring how the engine already turns a video into frames.
    ///
    /// Backed by PdfPig's <c>Page.GetImages()</c> + <c>IPdfImage.TryGetPng</c>, so it needs
    /// no external rasterizer (no PDFium / Ghostscript). Note: it recovers <em>embedded</em>
    /// images, not a rendered raster of vector/text content, which is exactly what an
    /// image-only PDF is made of.
    /// </summary>
    public static class PdfPageImageExtractor
    {
        /// <summary>Extracts page images from a PDF on disk. See <see cref="ExtractPageImagesFromBytes"/>.</summary>
        public static PdfImageResult ExtractPageImages(
            string pdfPath, string outputDirectory, int maxPages = 0, string namePrefix = null, string password = null)
            => ExtractPageImages(pdfPath, outputDirectory, maxPages, namePrefix, password, 0, default);

        /// <summary>Extract page images with an optional raster budget (zero is unlimited),
        /// checked before decoding, and cancellation between pages/images.</summary>
        public static PdfImageResult ExtractPageImages(
            string pdfPath, string outputDirectory, int maxPages, string namePrefix, string password,
            long maxImagePixels, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (string.IsNullOrEmpty(pdfPath))
                throw new ArgumentNullException(nameof(pdfPath));
            if (!File.Exists(pdfPath))
                throw new FileNotFoundException("PDF file not found.", pdfPath);

            byte[] bytes = File.ReadAllBytes(pdfPath);
            return ExtractPageImagesFromBytes(bytes, outputDirectory, maxPages, namePrefix, password, maxImagePixels, cancellationToken);
        }

        /// <summary>
        /// Extracts one image per page (the largest embedded image on each page) and writes
        /// each as a PNG into <paramref name="outputDirectory"/>.
        /// </summary>
        /// <param name="pdfBytes">The raw PDF file contents.</param>
        /// <param name="outputDirectory">Directory the PNGs are written to (created if missing).</param>
        /// <param name="maxPages">Optional cap on pages processed (<c>&lt;= 0</c> = all pages).</param>
        /// <param name="namePrefix">Optional base name for the emitted files (sanitized).</param>
        /// <param name="password">Optional password for an encrypted PDF.</param>
        /// <exception cref="InvalidDataException">The bytes are not a usable PDF.</exception>
        public static PdfImageResult ExtractPageImagesFromBytes(
            byte[] pdfBytes, string outputDirectory, int maxPages = 0, string namePrefix = null, string password = null)
            => ExtractPageImagesFromBytes(pdfBytes, outputDirectory, maxPages, namePrefix, password, 0, default);

        /// <summary>Byte-array extraction with a raster budget checked before decoding and
        /// cancellation between pages/images. The legacy overload retains its unlimited budget.</summary>
        public static PdfImageResult ExtractPageImagesFromBytes(
            byte[] pdfBytes, string outputDirectory, int maxPages, string namePrefix, string password,
            long maxImagePixels, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (maxImagePixels < 0) throw new ArgumentOutOfRangeException(nameof(maxImagePixels));
            if (pdfBytes == null || pdfBytes.Length == 0)
                throw new ArgumentException("Empty PDF data.", nameof(pdfBytes));
            if (string.IsNullOrEmpty(outputDirectory))
                throw new ArgumentNullException(nameof(outputDirectory));

            Directory.CreateDirectory(outputDirectory);
            string prefix = SanitizeName(namePrefix);

            var options = new ParsingOptions { UseLenientParsing = true, SkipMissingFonts = true };
            if (!string.IsNullOrEmpty(password))
                options.Password = password;

            PdfDocument document;
            try
            {
                document = PdfDocument.Open(pdfBytes, options);
            }
            catch (Exception ex)
            {
                throw new InvalidDataException(
                    "Could not open the PDF (it may be corrupt or password-protected): " + ex.Message, ex);
            }

            var paths = new List<string>();
            using (document)
            {
                int total = document.NumberOfPages;
                int limit = maxPages > 0 ? Math.Min(maxPages, total) : total;
                for (int i = 1; i <= limit; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    Page page;
                    try { page = document.GetPage(i); }
                    catch { continue; }

                    if (TrySaveLargestImage(page, outputDirectory, prefix, i, maxImagePixels, cancellationToken, out string savedPath))
                        paths.Add(savedPath);
                }

                return new PdfImageResult
                {
                    ImagePaths = paths,
                    PageCount = total,
                    ExtractedPageCount = paths.Count,
                };
            }
        }

        // Picks the largest embedded image on a page (by pixel area) and writes it as PNG.
        // Falls back to the next-largest if the largest can't be decoded to PNG, so a page
        // isn't lost to one odd image encoding.
        private static bool TrySaveLargestImage(
            Page page, string outputDirectory, string prefix, int pageNumber,
            long maxImagePixels, CancellationToken cancellationToken, out string savedPath)
        {
            savedPath = null;

            var images = new List<IPdfImage>();
            try
            {
                foreach (IPdfImage img in page.GetImages())
                    images.Add(img);
            }
            catch
            {
                return false;
            }
            if (images.Count == 0)
                return false;

            // Largest first: the dominant image on a text-less page is the page itself.
            images.Sort((a, b) =>
                ((long)b.WidthInSamples * b.HeightInSamples).CompareTo((long)a.WidthInSamples * a.HeightInSamples));

            foreach (IPdfImage img in images)
            {
                cancellationToken.ThrowIfCancellationRequested();
                // Check the declared raster dimensions before decompressing a potentially
                // tiny, highly-compressed image stream into a huge pixel allocation. Reject
                // instead of falling back to a thumbnail that would omit the page evidence.
                if (maxImagePixels > 0 && (img.WidthInSamples <= 0 || img.HeightInSamples <= 0 ||
                    (long)img.WidthInSamples * img.HeightInSamples > maxImagePixels))
                    throw new InvalidDataException($"PDF page {pageNumber} contains an image exceeding {maxImagePixels} pixels.");
                byte[] png;
                try
                {
                    if (!img.TryGetPng(out png) || png == null || png.Length == 0)
                        continue;
                }
                catch
                {
                    continue;
                }

                cancellationToken.ThrowIfCancellationRequested();

                string fileName = $"{prefix}_p{pageNumber:D3}.png";
                string path = Path.Combine(outputDirectory, fileName);
                File.WriteAllBytes(path, png);
                savedPath = path;
                return true;
            }

            return false;
        }

        private static string SanitizeName(string name)
        {
            if (string.IsNullOrWhiteSpace(name))
                return "pdfpage";

            name = Path.GetFileNameWithoutExtension(name);
            var sb = new StringBuilder(name.Length);
            foreach (char c in name)
                sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');

            string cleaned = sb.ToString().Trim('_');
            return cleaned.Length == 0 ? "pdfpage" : cleaned;
        }
    }
}
