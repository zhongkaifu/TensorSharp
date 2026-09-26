// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;

namespace TensorSharp.Server.Jev;

internal sealed record JevAttachmentDiagnostic(string Name, string Kind, int TextCharacters,
    int ImageCount, bool Sampled, bool CacheHit, string? Warning);

internal sealed record JevPreparedAttachments(string State, string[] ImagePaths,
    JevAttachmentDiagnostic[] Diagnostics, double PreprocessingMs);

/// <summary>Bounded attachment ingestion for the image/text Jev checkpoint. Audio is supplied
/// by the host's transcription service; it is never represented as native audio tokens.</summary>
internal static class JevAttachmentPreparer
{
    internal const int MaxTextCharacters = 32768;
    internal const int MaxTotalTextCharacters = 65536;
    internal const int MaxPdfPages = 32;
    internal const int MaxVideoFrames = 4;
    internal const double MaxVideoSeconds = 600;
    private const int CacheCapacity = 128;
    private static readonly object CacheLock = new();
    private static readonly Dictionary<string, Extraction> Cache = new(StringComparer.Ordinal);
    private static readonly Queue<string> CacheOrder = new();
    private sealed record Extraction(string Text, string[] Images, double[]? Timestamps = null,
        bool Sampled = false, string? Warning = null);

    internal static async Task<JevPreparedAttachments> PrepareAsync(JevRequest request,
        UploadStoragePolicy storage, Func<string, CancellationToken, Task<string>>? transcribeAudio = null,
        CancellationToken cancellationToken = default)
    {
        var watch = Stopwatch.StartNew();
        cancellationToken.ThrowIfCancellationRequested();
        Directory.CreateDirectory(storage.DirectoryPath);
        // Validate references and total byte budget before writing or invoking any decoder.
        long totalBytes = request.Images.Sum(image => (long)image.Bytes.Length);
        var sources = new List<(JevAttachment Attachment, string? Path)>();
        foreach (var attachment in request.Attachments)
        {
            string? path = attachment.File == null ? null : ResolveUpload(attachment.File, storage);
            long bytes;
            try { bytes = attachment.Bytes?.LongLength ?? new FileInfo(path!).Length; }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            { throw Error(attachment, "uploaded file is missing or unreadable"); }
            if (bytes <= 0 || bytes > JevAttachmentInput.MaxAttachmentBytes)
                throw Error(attachment, $"must contain 1 to {JevAttachmentInput.MaxAttachmentBytes} bytes");
            if (bytes > storage.MaxFileBytes)
                throw new UploadLimitExceededException($"Attachment '{attachment.Name}' exceeds the server's per-file upload limit.", 413);
            totalBytes += bytes;
            if (totalBytes > JevAttachmentInput.MaxTotalBytes)
                throw Error(attachment, $"combined attachments exceed {JevAttachmentInput.MaxTotalBytes} bytes");
            sources.Add((attachment, path));
        }
        if (totalBytes > JevAttachmentInput.MaxTotalBytes)
            throw new JevValidationException($"combined attachments exceed {JevAttachmentInput.MaxTotalBytes} bytes");
        var images = new List<string>(JevImageInput.Materialize(request.Images, storage));
        var state = new StringBuilder(request.State);
        var diagnostics = new List<JevAttachmentDiagnostic>();
        int textCharacters = 0;
        foreach (var (attachment, uploadedPath) in sources)
        {
            cancellationToken.ThrowIfCancellationRequested();
            string path = uploadedPath ?? ChatMessageParser.WriteContentAddressed(attachment.Bytes!, attachment.Extension, storage);
            int remainingImages = JevRequest.MaxImages - images.Count;
            bool cached = false;
            Extraction extracted;
            try
            {
                if (attachment.Kind == "audio")
                {
                    if (transcribeAudio == null)
                        throw new JevModelUnavailableException("Audio attachments require a configured transcription service (TS_JEV_TRANSCRIPTION_URL) or host JevAudioTranscriber callback.");
                    string transcript = await transcribeAudio(path, cancellationToken).ConfigureAwait(false);
                    if (string.IsNullOrWhiteSpace(transcript)) throw Error(attachment, "audio transcription returned no text");
                    CheckTextLength(transcript);
                    extracted = new(transcript, [], Warning: "Audio is represented by a transcript; non-speech sounds and speaker identity are not interpreted.");
                }
                else
                {
                    string key;
                    using (var stream = File.OpenRead(path))
                        key = Path.GetFullPath(storage.DirectoryPath) + "|" + Convert.ToHexString(SHA256.HashData(stream)) +
                            "|" + attachment.Extension + "|" + remainingImages;
                    lock (CacheLock)
                        cached = Cache.TryGetValue(key, out extracted!) && extracted.Images.All(File.Exists);
                    if (!cached)
                    {
                        extracted = await Task.Run(() => Extract(attachment, path, remainingImages, storage, cancellationToken), cancellationToken).ConfigureAwait(false);
                        lock (CacheLock)
                        {
                            if (!Cache.ContainsKey(key)) CacheOrder.Enqueue(key);
                            Cache[key] = extracted;
                            while (CacheOrder.Count > CacheCapacity) Cache.Remove(CacheOrder.Dequeue());
                        }
                    }
                    else
                        foreach (string image in extracted.Images) File.SetLastWriteTimeUtc(image, DateTime.UtcNow);
                }
            }
            catch (Exception ex) when (ex is InvalidDataException or IOException or UnauthorizedAccessException or DecoderFallbackException or System.Xml.XmlException or NotSupportedException or InvalidOperationException &&
                ex is not JevModelUnavailableException)
            {
                throw Error(attachment, "could not prepare attachment: " + ex.Message);
            }
            cancellationToken.ThrowIfCancellationRequested();
            textCharacters += extracted.Text.Length;
            if (textCharacters > MaxTotalTextCharacters)
                throw Error(attachment, $"combined extracted text exceeds {MaxTotalTextCharacters} characters; split the request");
            int firstImage = images.Count + 1;
            images.AddRange(extracted.Images);
            state.Append("\n\nAttachment ").Append(JsonSerializer.Serialize(attachment.Name))
                .Append(" (").Append(attachment.Kind).Append("):\n");
            if (extracted.Images.Length > 0)
            {
                for (int i = 0; i < extracted.Images.Length; i++)
                {
                    state.Append("Image ").Append(firstImage + i);
                    if (extracted.Timestamps != null)
                        state.Append(" at ").Append(extracted.Timestamps[i].ToString("0.###", CultureInfo.InvariantCulture)).Append(" seconds");
                    else if (attachment.Kind == "pdf") state.Append(" (page ").Append(i + 1).Append(')');
                    state.AppendLine();
                }
            }
            if (extracted.Warning != null) state.Append("Coverage: ").AppendLine(extracted.Warning);
            state.Append(extracted.Text);
            diagnostics.Add(new(attachment.Name, attachment.Kind, extracted.Text.Length, extracted.Images.Length,
                extracted.Sampled, cached, extracted.Warning));
        }
        return new(state.ToString(), images.ToArray(), diagnostics.ToArray(), watch.Elapsed.TotalMilliseconds);
    }

    private static string ResolveUpload(string reference, UploadStoragePolicy storage)
    {
        if (!JevAttachmentInput.IsBareName(reference))
            throw new JevValidationException("file must be a bare upload filename; paths and URLs are not accepted");
        string path = Path.Combine(storage.DirectoryPath, reference);
        try
        {
            var info = new FileInfo(path);
            if (!info.Exists || (info.Attributes & (FileAttributes.ReparsePoint | FileAttributes.Directory)) != 0 || info.LinkTarget != null)
                throw new JevValidationException($"file '{reference}' is missing, expired, or not a regular uploaded file");
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        { throw new JevValidationException($"file '{reference}' is missing, expired, or unreadable"); }
        return path;
    }

    private static Extraction Extract(JevAttachment attachment, string path, int remainingImages,
        UploadStoragePolicy storage, CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        if (attachment.Kind == "text")
        {
            // Strict UTF-8 prevents binary files and encoding errors from becoming plausible
            // replacement-character prose. The read is bounded independently of source bytes.
            using var reader = new StreamReader(path, new UTF8Encoding(false, true), detectEncodingFromByteOrderMarks: false);
            var chars = new char[MaxTextCharacters + 2];
            int length = reader.ReadBlock(chars, 0, chars.Length);
            string text = new(chars, 0, length);
            if (text.Length > 0 && text[0] == '\uFEFF') text = text[1..];
            CheckTextLength(text);
            if (text.Any(c => char.IsControl(c) && c is not ('\n' or '\r' or '\t')))
                throw new InvalidDataException("text files must contain UTF-8 text without binary control characters");
            if (string.IsNullOrWhiteSpace(text)) throw new InvalidDataException("attachment contains no readable text");
            return new(text, []);
        }
        if (attachment.Kind == "document")
            return new(JevAttachmentOfficeText.Extract(path, attachment.Extension, MaxTextCharacters, cancellationToken), [],
                Warning: "Office extraction includes text and cached cell values, without embedded images, chart rendering, formatting or formula recalculation.");
        if (attachment.Kind == "image")
        {
            RequireImages(remainingImages);
            if (new FileInfo(path).Length > JevImageInput.MaxImageBytes)
                throw new InvalidDataException($"image exceeds {JevImageInput.MaxImageBytes} bytes");
            byte[] bytes = File.ReadAllBytes(path);
            string? format = JevImageInput.SniffFormat(bytes);
            if (format == null) throw new InvalidDataException("unrecognized image contents");
            return new("", JevImageInput.Materialize([new JevImage(bytes, format)], storage));
        }
        if (attachment.Kind == "pdf")
        {
            var pdf = PdfTextExtractor.ExtractFromFile(path, MaxPdfPages, password: null, MaxTextCharacters + 1);
            if (pdf.PageCount > MaxPdfPages || pdf.ExtractedPageCount != pdf.PageCount || pdf.TextTruncated)
                throw new InvalidDataException($"PDF could not be read completely within {MaxPdfPages} pages and {MaxTextCharacters} characters; split or repair the document");
            CheckTextLength(pdf.Text);
            if (!pdf.LooksTextless) return new(pdf.Text, [], Warning: "PDF text layer only; embedded figures and layout are not rendered.");
            RequireImages(remainingImages);
            if (pdf.PageCount > remainingImages)
                throw new InvalidDataException($"scanned PDF has {pdf.PageCount} pages but only {remainingImages} image slots remain");
            string stage = MakeStageDirectory();
            try
            {
                var pages = PdfPageImageExtractor.ExtractPageImages(path, stage, remainingImages, namePrefix: null, password: null,
                    maxImagePixels: 16_777_216, cancellationToken: cancellationToken);
                if (pages.ExtractedPageCount != pages.PageCount)
                {
                    // Short but valid text documents need not have embedded page images.
                    if (pdf.NonWhitespaceCharCount > 0 && pages.ExtractedPageCount == 0)
                        return new(pdf.Text, [], Warning: "PDF text layer only; embedded figures and layout are not rendered.");
                    throw new InvalidDataException("scanned PDF could not yield one complete embedded image per page; render pages as images and retry");
                }
                return new(pdf.Text, StoreImages(pages.ImagePaths, storage),
                    Warning: "Scanned PDF uses the largest embedded image on each page; vector graphics and composited page layouts are not rendered.");
            }
            finally { Directory.Delete(stage, recursive: true); }
        }
        if (attachment.Kind == "video")
        {
            RequireImages(remainingImages);
            var info = DecodeVideo(() => MediaCodecs.Video.Probe(path));
            double duration = info.FrameCount / info.Fps;
            if (!double.IsFinite(duration) || duration <= 0 || duration > MaxVideoSeconds || info.FrameCount > 1_000_000 ||
                info.Width <= 0 || info.Height <= 0 || (long)info.Width * info.Height > 16_777_216)
                throw new InvalidDataException($"video must have valid dimensions (at most 16 megapixels) and duration of at most {MaxVideoSeconds} seconds");
            string stage = MakeStageDirectory();
            try
            {
                int limit = Math.Min(remainingImages, MaxVideoFrames);
                var frames = DecodeVideo(() => MediaHelper.ExtractVideoFramesWithTimestamps(path, stage, "jev", limit, fps: 1));
                cancellationToken.ThrowIfCancellationRequested();
                int expected = Math.Min(limit, (int)Math.Ceiling(info.FrameCount / Math.Max(1, Math.Round(info.Fps))));
                if (frames.Paths.Count != expected) throw new InvalidDataException("video could not decode all sampled frames");
                return new("", StoreImages(frames.Paths, storage), frames.Timestamps.ToArray(), Sampled: true,
                    Warning: "Video is represented by sampled frames with approximate timestamps; unsampled motion and the soundtrack are not analyzed.");
            }
            finally { Directory.Delete(stage, recursive: true); }
        }
        throw new InvalidDataException($"unsupported attachment kind '{attachment.Kind}'");
    }

    private static string[] StoreImages(IReadOnlyList<string> paths, UploadStoragePolicy storage)
        => paths.Select(path => ChatMessageParser.WriteContentAddressed(File.ReadAllBytes(path), ".png", storage)).ToArray();

    private static T DecodeVideo<T>(Func<T> decode)
    {
        try { return decode(); }
        // Providers include native codec exceptions that do not inherit IOException.
        catch (Exception ex) when (ex is not (OperationCanceledException or OutOfMemoryException))
        { throw new InvalidDataException("video decoder could not read the clip: " + ex.Message, ex); }
    }

    private static string MakeStageDirectory()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-jev-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(path);
        return path;
    }

    private static void CheckTextLength(string text)
    {
        if (text.Length > MaxTextCharacters)
            throw new InvalidDataException($"extracted text exceeds {MaxTextCharacters} characters; split the attachment");
    }

    private static void RequireImages(int remainingImages)
    {
        if (remainingImages <= 0) throw new InvalidDataException($"attachments exceed the combined limit of {JevRequest.MaxImages} images/frames/pages");
    }

    private static JevValidationException Error(JevAttachment attachment, string message) => new($"attachment '{attachment.Name}': {message}");
}
