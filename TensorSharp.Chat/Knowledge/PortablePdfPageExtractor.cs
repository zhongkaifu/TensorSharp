using System;
using System.Buffers.Binary;
using System.Collections.Generic;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Knowledge;
using UglyToad.PdfPig;

namespace TensorSharp.Server.Knowledge;

/// <summary>Full-page local OCR through local Poppler and Tesseract 5.
/// Uses argument arrays and bounded pipes, never a shell. Apple mobile hosts use Vision instead.</summary>
public sealed class PortablePdfPageExtractor : IKnowledgePdfPageExtractor
{
    private readonly PdfOcrOptions _options;
    private readonly IReadOnlyDictionary<string, string> _processEnvironment;
    private readonly SemaphoreSlim _initialization = new(1, 1);
    private Capabilities _capabilities;
    internal sealed record FileStamp(string Path, long Length, long LastWriteTicks)
    {
        internal static FileStamp Read(string path)
        {
            var file = new FileInfo(path);
            return new(path, file.Exists ? file.Length : -1, file.Exists ? file.LastWriteTimeUtc.Ticks : 0);
        }
        internal bool IsCurrent() => this == Read(Path);
    }
    internal sealed record LanguageDataFingerprint(string Sha256, FileStamp[] Files);
    private sealed record Capabilities(string Renderer, string Ocr, string Languages, string DataDirectory,
        string Identity, FileStamp[] Files)
    {
        internal bool IsCurrent() => Files.All(file => file.IsCurrent());
    }
    private static readonly UTF8Encoding StrictUtf8 = new(false, true);
    private static readonly IReadOnlyDictionary<string, string> ProcessEnvironment = new Dictionary<string, string>
    {
        ["OMP_THREAD_LIMIT"] = "1", ["LC_ALL"] = "C", ["LANG"] = "C"
    };
    public static bool IsSupported => OperatingSystem.IsWindows() || OperatingSystem.IsLinux()
        || OperatingSystem.IsMacOS() || OperatingSystem.IsFreeBSD();

    public PortablePdfPageExtractor(PdfOcrOptions options = null)
    {
        _options = options ?? PdfOcrOptions.FromEnvironment();
        var environment = _options.ProcessEnvironment == null ? new Dictionary<string, string>()
            : new Dictionary<string, string>(_options.ProcessEnvironment);
        foreach (var pair in ProcessEnvironment) environment[pair.Key] = pair.Value;
        _processEnvironment = environment;
        if (_options.Dpi is < 72 or > 600) throw new ArgumentOutOfRangeException(nameof(options), "OCR DPI must be between 72 and 600.");
        if (_options.TimeoutSeconds is < 1 or > 600) throw new ArgumentOutOfRangeException(nameof(options), "OCR timeout must be between 1 and 600 seconds.");
        if (_options.MaxInputBytes is < 1 or > 1024L * 1024 * 1024) throw new ArgumentOutOfRangeException(nameof(options), "OCR input byte limit must be between 1 byte and 1 GiB.");
        if (_options.MaxEmbeddedImagePixels is < 1 or > 256_000_000) throw new ArgumentOutOfRangeException(nameof(options), "OCR embedded image pixel limit must be between 1 and 256000000.");
        if (_options.Languages != null) ValidateLanguages(_options.Languages);
    }

    public async Task<KnowledgePdfPageExtraction> ExtractPageAsync(KnowledgePdfPageContext context,
        CancellationToken cancellationToken = default)
    {
        Validate(context);
        if (!IsSupported) throw new PlatformNotSupportedException("The process PDF OCR backend requires a desktop platform; Apple mobile uses Vision.");
        cancellationToken.ThrowIfCancellationRequested();
        // The core normally provides a per-page deadline. Bound direct SDK use too.
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(TimeSpan.FromSeconds(_options.TimeoutSeconds));
        var ct = deadline.Token;
        string scratch = CreateScratch();
        Exception? primaryFailure = null;
        try
        {
            PreflightFile(context.FilePath);
            var capabilities = await GetCapabilitiesAsync(scratch, ct).ConfigureAwait(false);
            byte[] image = await RenderAsync(context, capabilities.Renderer, scratch, ct).ConfigureAwait(false);
            var args = new List<string> { "stdin", "stdout", "-l", capabilities.Languages, "--oem", "1", "--psm", "3", "--dpi", _options.Dpi.ToString(CultureInfo.InvariantCulture) };
            if (capabilities.DataDirectory != null) { args.Add("--tessdata-dir"); args.Add(capabilities.DataDirectory); }
            var result = await RunAsync(capabilities.Ocr, args, scratch, checked(context.MaxCharacters * 4 + 4), ct, image).ConfigureAwait(false);
            EnsureSuccess(result, "Tesseract");
            if (!capabilities.IsCurrent())
                throw new InvalidDataException("The PDF OCR installation changed during recognition; retry indexing.");
            string text = DecodeUtf8(result.StandardOutput).Trim();
            if (text.Length > context.MaxCharacters) throw new KnowledgeLimitException("PDF OCR text exceeds the remaining extraction limit; split the document.");
            return new KnowledgePdfPageExtraction { Text = text, ExtractorId = capabilities.Identity,
                Warnings = string.IsNullOrWhiteSpace(result.StandardError) ? Array.Empty<string>()
                    : new[] { "Tesseract diagnostic: " + LimitDiagnostic(result.StandardError) } };
        }
        catch (Win32Exception error)
        {
            primaryFailure = new NotSupportedException("PDF OCR requires installed Poppler pdftoppm and Tesseract 5 executables. Configure their paths and reindex.", error);
            throw primaryFailure;
        }
        catch (Exception error) { primaryFailure = error; throw; }
        finally { await PortablePdfScratchDirectory.DeletePreservingFailureAsync(scratch, primaryFailure).ConfigureAwait(false); }
    }

    /// <summary>Render a complete cropped/rotated page. Useful for previews and independent raster validation.</summary>
    public async Task<byte[]> RenderPageAsync(KnowledgePdfPageContext context, CancellationToken cancellationToken = default)
    {
        Validate(context);
        if (!IsSupported) throw new PlatformNotSupportedException("Portable PDF rendering requires a desktop platform.");
        cancellationToken.ThrowIfCancellationRequested();
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        deadline.CancelAfter(TimeSpan.FromSeconds(_options.TimeoutSeconds));
        string scratch = CreateScratch();
        Exception? primaryFailure = null;
        try { return await RenderAsync(context, ResolveExecutable(_options.PdfToPpmExecutable, "pdftoppm"), scratch, deadline.Token).ConfigureAwait(false); }
        catch (Exception error) { primaryFailure = error; throw; }
        finally { await PortablePdfScratchDirectory.DeletePreservingFailureAsync(scratch, primaryFailure).ConfigureAwait(false); }
    }

    private async Task<byte[]> RenderAsync(KnowledgePdfPageContext context, string renderer, string scratch, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        PreflightFile(context.FilePath);
        using var input = File.OpenRead(context.FilePath);
        using var pdf = PdfDocument.Open(input);
        if (context.PageNumber > pdf.NumberOfPages) throw new ArgumentOutOfRangeException(nameof(context.PageNumber));
        var page = pdf.GetPage(context.PageNumber);
        var box = page.CropBox.Bounds;
        var media = page.MediaBox.Bounds;
        // PDF crop boxes are clipped to the media box. Use that visible intersection
        // for aspect ratio and limits, matching Poppler's -cropbox behavior.
        double width = Math.Min(box.Right, media.Right) - Math.Max(box.Left, media.Left);
        double height = Math.Min(box.Top, media.Top) - Math.Max(box.Bottom, media.Bottom);
        long pixels = 0;
        int imageCount = 0;
        foreach (var image in page.GetImages())
        {
            ct.ThrowIfCancellationRequested();
            long imagePixels = (long)image.WidthInSamples * image.HeightInSamples;
            if (++imageCount > 4096 || image.WidthInSamples <= 0 || image.HeightInSamples <= 0
                || imagePixels > _options.MaxEmbeddedImagePixels - pixels)
                throw new KnowledgeLimitException("PDF embedded images exceed the per-page aggregate pixel or count limit.");
            pixels += imagePixels;
        }
        var (w, h) = CalculateRasterSize(width, height, page.Rotation.SwapsAxis, _options.Dpi, context.MaxPixels);
        string number = context.PageNumber.ToString(CultureInfo.InvariantCulture);
        var args = new[] { "-f", number, "-l", number, "-singlefile", "-cropbox", "-scale-dimension-before-rotation", "-r", _options.Dpi.ToString(CultureInfo.InvariantCulture),
            "-scale-to-x", w.ToString(CultureInfo.InvariantCulture), "-scale-to-y", h.ToString(CultureInfo.InvariantCulture),
            "-png", Path.GetFullPath(context.FilePath) };
        // A PNG's incompressible RGBA representation plus bounded encoding overhead.
        var result = await RunAsync(renderer, args, scratch, checked(context.MaxPixels * 4 + 1_048_576), ct).ConfigureAwait(false);
        EnsureSuccess(result, "Poppler");
        ValidatePng(result.StandardOutput, context.MaxPixels, w + 1, h + 1);
        return result.StandardOutput;
    }

    internal static (int Width, int Height) CalculateRasterSize(double width, double height,
        bool swapsAxis, int dpi, int maxPixels)
    {
        if (swapsAxis) (width, height) = (height, width);
        if (!double.IsFinite(width) || !double.IsFinite(height) || width <= 0 || height <= 0 || width > 1_000_000 || height > 1_000_000)
            throw new InvalidDataException("The PDF page has invalid crop dimensions.");
        double scale = Math.Min(dpi / 72.0, Math.Sqrt(maxPixels / (width * height)));
        // Poppler may round up raster edges. Reserve one pixel in each dimension.
        int w = Math.Max(1, (int)Math.Floor(width * scale) - 1), h = Math.Max(1, (int)Math.Floor(height * scale) - 1);
        if (w > 32768 || h > 32768) throw new KnowledgeLimitException("The PDF page dimensions exceed the raster limit.");
        return (w, h);
    }

    internal static void ValidatePng(byte[] png, int maxPixels, int maxWidth, int maxHeight)
    {
        ReadOnlySpan<byte> signature = new byte[] { 137, 80, 78, 71, 13, 10, 26, 10 };
        if (png.Length < 33 || !png.AsSpan(0, 8).SequenceEqual(signature)
            || BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(8, 4)) != 13
            || !png.AsSpan(12, 4).SequenceEqual("IHDR"u8))
            throw new InvalidDataException("The PDF renderer did not return a PNG raster.");
        uint width = BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(16, 4)), height = BinaryPrimitives.ReadUInt32BigEndian(png.AsSpan(20, 4));
        if (width == 0 || height == 0 || width > maxWidth || height > maxHeight || (ulong)width * height > (ulong)maxPixels)
            throw new KnowledgeLimitException("The PDF renderer exceeded the requested raster dimensions.");
    }

    private async Task<Capabilities> GetCapabilitiesAsync(string scratch, CancellationToken ct)
    {
        if (_capabilities?.IsCurrent() == true) return _capabilities;
        await _initialization.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (_capabilities?.IsCurrent() == true) return _capabilities;
            string renderer = ResolveExecutable(_options.PdfToPpmExecutable, "pdftoppm"), ocr = ResolveExecutable(_options.TesseractExecutable, "tesseract");
            string data = string.IsNullOrWhiteSpace(_options.TessdataDirectory) ? null : Path.GetFullPath(_options.TessdataDirectory);
            if (data != null && !Directory.Exists(data)) throw new NotSupportedException("The configured Tesseract language-data directory does not exist.");
            var renderVersion = await RunAsync(renderer, new[] { "-v" }, scratch, 16384, ct).ConfigureAwait(false);
            EnsureSuccess(renderVersion, "Poppler");
            var ocrVersion = await RunAsync(ocr, new[] { "--version" }, scratch, 16384, ct).ConfigureAwait(false);
            EnsureSuccess(ocrVersion, "Tesseract");
            string version = FirstLine(DecodeUtf8(ocrVersion.StandardOutput) + ocrVersion.StandardError);
            if (!IsSupportedVersion(version))
                throw new NotSupportedException("PDF OCR requires Tesseract 5 or later with the LSTM engine.");
            var languageArgs = new List<string> { "--list-langs" };
            if (data != null) { languageArgs.Add("--tessdata-dir"); languageArgs.Add(data); }
            var languageResult = await RunAsync(ocr, languageArgs, scratch, 65536, ct).ConfigureAwait(false);
            EnsureSuccess(languageResult, "Tesseract language discovery");
            string listing = DecodeUtf8(languageResult.StandardOutput);
            string[] selected = SelectLanguages(listing, _options.Languages);
            if (data == null)
            {
                int start = listing.IndexOf('"'), end = start < 0 ? -1 : listing.IndexOf('"', start + 1);
                if (end > start) data = listing[(start + 1)..end];
            }
            string languageIdentity = "unresolved-data-path";
            var files = new List<FileStamp> { FileStamp.Read(renderer), FileStamp.Read(ocr) };
            if (!string.IsNullOrEmpty(data) && Directory.Exists(data))
            {
                var fingerprint = await FingerprintLanguageDataAsync(data, selected, ct).ConfigureAwait(false);
                languageIdentity = fingerprint.Sha256;
                files.AddRange(fingerprint.Files);
            }
            string poppler = FirstLine(DecodeUtf8(renderVersion.StandardOutput) + renderVersion.StandardError);
            string languages = string.Join('+', selected);
            _capabilities = new(renderer, ocr, languages, data,
                $"poppler-tesseract-v1;{poppler};{version};dpi={_options.Dpi};languages={languages};data-sha256={languageIdentity}", files.ToArray());
            return _capabilities;
        }
        finally { _initialization.Release(); }
    }

    internal static async Task<LanguageDataFingerprint> FingerprintLanguageDataAsync(string data,
        IReadOnlyList<string> selected, CancellationToken ct = default)
    {
        using var hash = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
        var files = new List<FileStamp>();
        var required = selected.ToHashSet(StringComparer.Ordinal);
        // Tesseract language configs can load the corresponding vertical model even
        // when it was not named in -l. Keep the requested languages unchanged, but
        // conservatively fingerprint each installed companion, including script packs.
        var names = selected.Concat(selected.Where(name => !name.EndsWith("_vert", StringComparison.Ordinal))
            .Select(name => name + "_vert")).Distinct(StringComparer.Ordinal);
        foreach (string language in names)
        {
            ct.ThrowIfCancellationRequested();
            string path = Path.Combine(data, language.Replace('/', Path.DirectorySeparatorChar) + ".traineddata");
            FileStamp stamp = FileStamp.Read(path);
            // Retain absence too, so installing a previously missing companion
            // invalidates cached capabilities before the next page is recognized.
            files.Add(stamp);
            if (stamp.Length < 0)
            {
                if (required.Contains(language))
                    throw new NotSupportedException("A selected Tesseract traineddata file is missing; restore the language installation and reindex.");
                continue;
            }
            hash.AppendData(Encoding.UTF8.GetBytes(language));
            await using var stream = File.OpenRead(path);
            hash.AppendData(await SHA256.HashDataAsync(stream, ct).ConfigureAwait(false));
            if (!stamp.IsCurrent()) throw new InvalidDataException("Tesseract language data changed during initialization; retry indexing.");
        }
        return new(Convert.ToHexStringLower(hash.GetHashAndReset()), files.ToArray());
    }

    private Task<PortablePdfProcessResult> RunAsync(string executable, IReadOnlyList<string> args, string scratch,
        int limit, CancellationToken ct, ReadOnlyMemory<byte> input = default)
        => PortablePdfProcessRunner.RunAsync(executable, args, scratch, limit, ct, input, _processEnvironment);

    internal static string ResolveExecutable(string configured, string name)
    {
        string suffix = OperatingSystem.IsWindows() ? ".exe" : "";
        if (!string.IsNullOrWhiteSpace(configured))
        {
            string explicitPath = Path.GetFullPath(configured);
            if (!File.Exists(explicitPath)) throw new NotSupportedException($"Configured {name} executable does not exist.");
            return explicitPath;
        }
        // Resolve to an absolute PATH entry: never search the current/source directory implicitly.
        foreach (string directory in (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator))
        {
            string dir = directory.Trim().Trim('"');
            if (!Path.IsPathFullyQualified(dir)) continue;
            string candidate = Path.Combine(dir, name + suffix);
            if (File.Exists(candidate)) return candidate;
        }
        throw new NotSupportedException($"PDF OCR needs {name}{suffix} installed on PATH, or its configured executable path. Install Poppler and Tesseract 5, then reindex.");
    }

    private static bool IsLanguageName(string value) => value.Length <= 80 && Regex.IsMatch(value, @"^[A-Za-z0-9_]+(?:/[A-Za-z0-9_]+)?$", RegexOptions.CultureInvariant);
    internal static string[] SelectLanguages(string listing, string requested)
    {
        var available = listing.Split('\n').Select(s => s.Trim()).Where(IsLanguageName).ToHashSet(StringComparer.Ordinal);
        string[] selected = requested?.Split('+') ?? new[] { "eng", "chi_sim", "chi_tra" }.Where(available.Contains).ToArray();
        if (selected.Length == 0) selected = available.Where(s => s != "osd").OrderBy(s => s, StringComparer.Ordinal).Take(1).ToArray();
        if (selected.Length == 0 || selected.Any(s => !available.Contains(s)))
            throw new NotSupportedException("The requested Tesseract language packs are not installed. Install the selected traineddata files and reindex.");
        return selected;
    }
    internal static bool IsSupportedVersion(string version)
    {
        Match match = Regex.Match(version, @"^tesseract v?([0-9]+)\.", RegexOptions.CultureInvariant);
        return match.Success && int.TryParse(match.Groups[1].Value, NumberStyles.None, CultureInfo.InvariantCulture, out int major) && major >= 5;
    }
    internal static string DecodeUtf8(byte[] bytes)
    {
        try { return StrictUtf8.GetString(bytes); }
        catch (DecoderFallbackException error) { throw new InvalidDataException("The PDF utility returned malformed UTF-8 text.", error); }
    }
    private void PreflightFile(string path)
    {
        long length = new FileInfo(path).Length;
        if (length == 0) throw new InvalidDataException("The source PDF is empty.");
        if (length > _options.MaxInputBytes) throw new KnowledgeLimitException("The PDF exceeds the configured OCR input byte limit.");
    }
    private static void ValidateLanguages(string value)
    {
        string[] names = value.Split('+');
        if (value.Length > 240 || names.Length > 16 || names.Any(s => !IsLanguageName(s)))
            throw new ArgumentException("OCR languages must be traineddata names separated by '+', such as eng+chi_sim.", nameof(value));
    }
    private static void Validate(KnowledgePdfPageContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        ArgumentException.ThrowIfNullOrWhiteSpace(context.FilePath);
        if (context.PageNumber < 1) throw new ArgumentOutOfRangeException(nameof(context.PageNumber));
        if (context.MaxCharacters is < 1 or > 2_000_000) throw new KnowledgeLimitException("PDF OCR character budget must be between 1 and 2000000.");
        if (context.MaxPixels is < 65536 or > 64_000_000) throw new ArgumentOutOfRangeException(nameof(context.MaxPixels));
    }
    private static string CreateScratch()
    {
        string path = Path.Combine(Path.GetTempPath(), "tensorsharp-pdf-ocr-" + Guid.NewGuid().ToString("N"));
        if (OperatingSystem.IsWindows()) Directory.CreateDirectory(path);
        else Directory.CreateDirectory(path, UnixFileMode.UserRead | UnixFileMode.UserWrite | UnixFileMode.UserExecute);
        return path;
    }
    private static void EnsureSuccess(PortablePdfProcessResult result, string tool)
    {
        if (result.ExitCode != 0) throw new InvalidDataException($"{tool} failed (exit {result.ExitCode}): {LimitDiagnostic(result.StandardError)}");
    }
    private static string FirstLine(string text) => LimitDiagnostic(text.Split('\n')[0].Trim(), 120);
    private static string LimitDiagnostic(string text, int length = 1000) => string.Concat(text.Take(length).Select(c => char.IsControl(c) ? ' ' : c));
}
