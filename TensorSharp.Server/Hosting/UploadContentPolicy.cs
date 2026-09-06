using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.FileProviders;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Single source of truth for which upload extensions the server accepts
    /// and what content type <c>/uploads</c> serves each one back with.
    /// Uploaded files are untrusted, so every text/code extension is served as
    /// <c>text/plain</c> — an uploaded .html page must never execute in the
    /// server's origin — and unlisted extensions are rejected at upload and
    /// unmapped (404) when served.
    /// </summary>
    public static class UploadContentPolicy
    {
        private const string TextPlain = "text/plain; charset=utf-8";

        // Value = (media type reported to the Web UI, content type served on /uploads).
        private static readonly Dictionary<string, (string MediaType, string ContentType)> Extensions =
            new(StringComparer.OrdinalIgnoreCase)
            {
                [".png"] = ("image", "image/png"),
                [".jpg"] = ("image", "image/jpeg"),
                [".jpeg"] = ("image", "image/jpeg"),
                [".gif"] = ("image", "image/gif"),
                [".webp"] = ("image", "image/webp"),
                [".bmp"] = ("image", "image/bmp"),
                // HEIC/HEIF (iPhone photos): browsers can't render them in <img> —
                // the Web UI shows the server-generated PNG previewUrl — but the
                // original must stay downloadable, and the default provider has no
                // mapping for either extension.
                [".heic"] = ("image", "image/heic"),
                [".heif"] = ("image", "image/heif"),

                [".mp4"] = ("video", "video/mp4"),
                [".mov"] = ("video", "video/quicktime"),
                [".avi"] = ("video", "video/x-msvideo"),
                [".mkv"] = ("video", "video/x-matroska"),
                [".webm"] = ("video", "video/webm"),

                [".mp3"] = ("audio", "audio/mpeg"),
                [".wav"] = ("audio", "audio/wav"),
                [".ogg"] = ("audio", "audio/ogg"),
                [".flac"] = ("audio", "audio/flac"),
                [".m4a"] = ("audio", "audio/mp4"),

                [".pdf"] = ("pdf", "application/pdf"),

                [".txt"] = ("text", TextPlain),
                [".csv"] = ("text", TextPlain),
                [".json"] = ("text", TextPlain),
                [".xml"] = ("text", TextPlain),
                [".md"] = ("text", TextPlain),
                [".log"] = ("text", TextPlain),
                [".py"] = ("text", TextPlain),
                [".js"] = ("text", TextPlain),
                [".ts"] = ("text", TextPlain),
                [".cs"] = ("text", TextPlain),
                [".java"] = ("text", TextPlain),
                [".cpp"] = ("text", TextPlain),
                [".c"] = ("text", TextPlain),
                [".h"] = ("text", TextPlain),
                [".html"] = ("text", TextPlain),
                [".css"] = ("text", TextPlain),
                [".yaml"] = ("text", TextPlain),
                [".yml"] = ("text", TextPlain),
                [".toml"] = ("text", TextPlain),
                [".ini"] = ("text", TextPlain),
                [".cfg"] = ("text", TextPlain),
                [".sh"] = ("text", TextPlain),
                [".bat"] = ("text", TextPlain),
                [".ps1"] = ("text", TextPlain),
                [".rb"] = ("text", TextPlain),
                [".go"] = ("text", TextPlain),
                [".rs"] = ("text", TextPlain),
                [".swift"] = ("text", TextPlain),
                [".kt"] = ("text", TextPlain),
                [".sql"] = ("text", TextPlain),
                [".r"] = ("text", TextPlain),
                [".m"] = ("text", TextPlain),
                [".tex"] = ("text", TextPlain),
                [".rtf"] = ("text", TextPlain),
            };

        internal static IEnumerable<string> SupportedExtensions => Extensions.Keys;

        /// <summary>
        /// Media type for an upload extension: image, video, audio, pdf, text,
        /// or "unknown" for anything unlisted. Unknown uploads are rejected.
        /// </summary>
        internal static string Classify(string ext) =>
            Extensions.TryGetValue(ext, out var entry) ? entry.MediaType : "unknown";

        internal static IContentTypeProvider BuildServeContentTypes() =>
            new FileExtensionContentTypeProvider(Extensions.ToDictionary(
                kv => kv.Key, kv => kv.Value.ContentType, StringComparer.OrdinalIgnoreCase));

        public static StaticFileOptions BuildStaticFileOptions(string uploadDirectory) => new()
        {
            FileProvider = new PhysicalFileProvider(uploadDirectory),
            RequestPath = "/uploads",
            ContentTypeProvider = BuildServeContentTypes(),
            OnPrepareResponse = ctx =>
                ctx.Context.Response.Headers["X-Content-Type-Options"] = "nosniff",
        };
    }
}
