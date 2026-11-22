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
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;

namespace TensorSharp.AgentHost.Skills
{
    /// <summary>Where a skill came from, which decides whether it may be deleted through the API.</summary>
    public enum SkillOrigin
    {
        /// <summary>Found by scanning a directory the operator configured. Removing it
        /// through the management API would delete the operator's own file tree, so it
        /// is refused.</summary>
        Discovered,
        /// <summary>Installed at runtime — uploaded as a ZIP or copied in from a
        /// directory — into the managed skills root. Owned by TensorSharp, so it can be
        /// deleted again.</summary>
        Installed,
    }

    /// <summary>
    /// What kind of content a bundled file holds, following the directory
    /// conventions in the specification. This is a hint used to order the file
    /// listing the model sees, never a restriction: the specification says a skill
    /// "may contain any files and directories beyond the required SKILL.md", and
    /// several published skills keep documentation in <c>reference/</c>,
    /// <c>examples/</c>, per-language folders (<c>python/</c>, <c>csharp/</c>) or at
    /// the skill root, so anything that only exposed <c>scripts/</c>,
    /// <c>references/</c> and <c>assets/</c> would hide most of them.
    /// </summary>
    public enum SkillFileKind
    {
        /// <summary>The skill's own <c>SKILL.md</c>.</summary>
        Manifest,
        /// <summary>Under <c>scripts/</c>, or an executable file type anywhere.</summary>
        Script,
        /// <summary>Under <c>references/</c> or <c>reference/</c>, or a Markdown file elsewhere.</summary>
        Reference,
        /// <summary>Under <c>assets/</c>, or a binary resource anywhere.</summary>
        Asset,
        /// <summary>Anything else the skill ships.</summary>
        Other,
    }

    /// <summary>One file inside a skill directory.</summary>
    /// <param name="Path">Path relative to the skill root, always with forward slashes.</param>
    /// <param name="Bytes">Size on disk.</param>
    /// <param name="Kind">The convention bucket this file falls in.</param>
    /// <param name="IsText">
    /// True when the file can be handed to the model as text. Decided by extension
    /// and, for unknown extensions, by sniffing the first block for NUL bytes —
    /// a skill that ships a <c>.pdf</c> or a font must not have it spliced into a
    /// prompt as mojibake.
    /// </param>
    public readonly record struct SkillFile(string Path, long Bytes, SkillFileKind Kind, bool IsText);

    /// <summary>
    /// Where a skill installed from the public skill hub came from: the exact package, the
    /// immutable Git commit it was installed at, and when. Recorded by the registry at
    /// install time and shown wherever the skill is listed, so a user can always see that a
    /// skill is third-party and which revision of whose repository it is.
    /// </summary>
    /// <param name="Package">The hub selector, <c>owner/repository@skill-name</c>.</param>
    /// <param name="Url">The skill's directory on GitHub at <paramref name="Revision"/>.</param>
    /// <param name="Revision">The 40-character commit the files were taken from.</param>
    /// <param name="InstalledUtc">When this host installed it.</param>
    /// <param name="RenamedFrom">
    /// The upstream name when it was installed under another one because a different skill
    /// here already has that name; null when the names agree.
    /// </param>
    public sealed record SkillSource(
        string Package, string Url, string Revision, DateTimeOffset InstalledUtc, string? RenamedFrom = null)
    {
        /// <summary>One line for a person: "owner/repo@name from https://github.com/... @ 1a2b3c4".</summary>
        public string Describe() => Package + " from " + Url + " @ " + Revision;
    }

    /// <summary>
    /// One skill loaded from disk: its parsed <c>SKILL.md</c>, the directory that
    /// holds it, and an index of the files it ships.
    ///
    /// <para>
    /// A <see cref="Skill"/> is immutable and cheap to hold: the file index is
    /// built once at load and the <c>SKILL.md</c> body is already in memory (it is
    /// the "instructions" tier of progressive disclosure and is needed whenever the
    /// skill is activated), while every other file is read on demand through
    /// <see cref="TryReadResource"/> so a skill shipping a 40 MB asset costs
    /// nothing until something asks for it.
    /// </para>
    /// </summary>
    public sealed class Skill
    {
        internal Skill(
            string id,
            SkillManifest manifest,
            string rootDirectory,
            SkillOrigin origin,
            string? discoveredUnder,
            IReadOnlyList<SkillFile> files,
            DateTime modifiedUtc)
        {
            Id = id;
            Manifest = manifest;
            RootDirectory = rootDirectory;
            Origin = origin;
            DiscoveredUnder = discoveredUnder;
            Files = files;
            ModifiedUtc = modifiedUtc;
            TotalBytes = files.Sum(f => f.Bytes);
        }

        /// <summary>
        /// The identifier this skill is selected by. Normally
        /// <see cref="SkillManifest.Name"/>; see <see cref="SkillRegistry"/> for what
        /// happens when two roots ship a skill of the same name.
        /// </summary>
        public string Id { get; }

        /// <summary>The parsed <c>SKILL.md</c>.</summary>
        public SkillManifest Manifest { get; }

        /// <summary>The skill's own directory. Every file access is confined to it.</summary>
        public string RootDirectory { get; }

        /// <summary>Whether this skill is owned by TensorSharp or by the operator.</summary>
        public SkillOrigin Origin { get; }

        /// <summary>The configured root this skill was found under, or null when installed directly.</summary>
        public string? DiscoveredUnder { get; }

        /// <summary>Every file in the skill directory, sorted by path.</summary>
        public IReadOnlyList<SkillFile> Files { get; }

        /// <summary>Total bytes on disk.</summary>
        public long TotalBytes { get; }

        /// <summary>
        /// Provenance of a skill installed from the skill hub; null for every other skill
        /// (bundled, operator-provided, or uploaded by the user). A hub skill's files are
        /// third-party content: they reach the model only as tool output, never as part of
        /// the system prompt (<see cref="SkillPrompt"/>).
        /// </summary>
        public SkillSource? Source { get; internal set; }

        /// <summary>The newest write time in the skill directory, used for cache invalidation and listings.</summary>
        public DateTime ModifiedUtc { get; }

        /// <summary>Convenience: the skill's name as declared, which is also its <see cref="Id"/> unless it collided.</summary>
        public string Name => Manifest.Name;

        /// <summary>Convenience: the one-line description shown in the catalog.</summary>
        public string Description => Manifest.Description;

        /// <summary>
        /// The files worth advertising to the model when it activates this skill,
        /// excluding <c>SKILL.md</c> itself (already in front of it) and licence
        /// boilerplate (never task-relevant, and present in most published skills).
        /// </summary>
        public IEnumerable<SkillFile> BundledFiles =>
            Files.Where(f => f.Kind != SkillFileKind.Manifest && !IsLicenseFile(f.Path));

        /// <summary>
        /// Read one of the skill's files as text.
        /// </summary>
        /// <param name="relativePath">
        /// A path relative to the skill root, as a <c>SKILL.md</c> body writes it.
        /// Resolved through <see cref="SkillPathGuard"/>, so nothing outside the skill
        /// directory is reachable however the argument is spelled.
        /// </param>
        /// <param name="maxBytes">
        /// Hard cap on how much is returned. A skill may ship a reference file far
        /// larger than the context window, and returning all of it would push the
        /// conversation over the limit; the caller is told what was cut through
        /// <paramref name="result"/>.
        /// </param>
        /// <param name="offsetBytes">
        /// Where to start, so a caller that hit the cap can ask for the next page.
        /// </param>
        /// <param name="result">The file's text and what was truncated.</param>
        /// <param name="error">Why the read failed, or null.</param>
        public bool TryReadResource(
            string? relativePath,
            int maxBytes,
            long offsetBytes,
            out SkillResourceContent result,
            out string? error)
        {
            result = default;

            if (!SkillPathGuard.TryResolveExistingFile(RootDirectory, relativePath, out string? fullPath, out error))
                return false;

            string normalized = SkillPathGuard.ToSkillRelative(RootDirectory, fullPath!);

            FileInfo info;
            try
            {
                info = new FileInfo(fullPath!);
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{normalized}' could not be opened: {ex.Message}";
                return false;
            }

            if (!IsTextFile(fullPath!, normalized))
            {
                error = $"'{normalized}' is a binary file ({SkillTextBudget.FormatBytes(info.Length)}) and cannot be read as text; "
                      + "run it or reference it by path instead.";
                return false;
            }

            if (offsetBytes < 0)
                offsetBytes = 0;
            if (offsetBytes >= info.Length && info.Length > 0)
            {
                error = $"'{normalized}' is {info.Length} bytes; offset {offsetBytes} is past the end.";
                return false;
            }

            try
            {
                using FileStream stream = new(fullPath!, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                if (offsetBytes > 0)
                    stream.Seek(offsetBytes, SeekOrigin.Begin);

                int budget = Math.Max(256, maxBytes);
                byte[] buffer = new byte[budget];
                int read = stream.ReadAtLeast(buffer, budget, throwOnEndOfStream: false);

                // Never split a UTF-8 sequence across a page boundary: the tail bytes
                // would decode to U+FFFD here and the next page would start mid-glyph.
                int usable = read < budget ? read : TrimToUtf8Boundary(buffer, read);
                string text = Encoding.UTF8.GetString(buffer, 0, usable);

                long nextOffset = offsetBytes + usable;
                result = new SkillResourceContent(
                    normalized,
                    text,
                    info.Length,
                    offsetBytes,
                    nextOffset,
                    Truncated: nextOffset < info.Length);
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                error = $"'{normalized}' could not be read: {ex.Message}";
                return false;
            }
        }

        /// <summary>
        /// True when a file is safe to hand to the model as text. Extensions decide
        /// the common cases outright; an unrecognised extension is sniffed, because a
        /// skill may reasonably ship an extension-less <c>Makefile</c> or <c>LICENSE</c>
        /// and refusing those would be worse than reading them.
        /// </summary>
        private static bool IsTextFile(string fullPath, string relativePath)
        {
            string ext = Path.GetExtension(relativePath);
            if (TextExtensions.Contains(ext))
                return true;
            if (BinaryExtensions.Contains(ext))
                return false;

            try
            {
                using FileStream stream = new(fullPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
                Span<byte> probe = stackalloc byte[512];
                int read = stream.ReadAtLeast(probe, probe.Length, throwOnEndOfStream: false);
                for (int i = 0; i < read; i++)
                {
                    if (probe[i] == 0)
                        return false;
                }
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return false;
            }
        }

        /// <summary>
        /// Walk back from <paramref name="length"/> to the last byte that starts a
        /// complete UTF-8 sequence, so a page never ends inside one.
        /// </summary>
        private static int TrimToUtf8Boundary(byte[] buffer, int length)
        {
            // A continuation byte is 10xxxxxx; at most 3 of them can precede the lead.
            int limit = Math.Max(0, length - 4);
            for (int i = length - 1; i >= limit; i--)
            {
                byte b = buffer[i];
                if ((b & 0b1100_0000) == 0b1000_0000)
                    continue;   // continuation byte, keep walking back to its lead

                int sequenceLength =
                    (b & 0b1000_0000) == 0 ? 1 :
                    (b & 0b1110_0000) == 0b1100_0000 ? 2 :
                    (b & 0b1111_0000) == 0b1110_0000 ? 3 :
                    (b & 0b1111_1000) == 0b1111_0000 ? 4 : 1;

                return i + sequenceLength <= length ? length : i;
            }
            return length;
        }

        internal static SkillFileKind ClassifyFile(string relativePath)
        {
            if (string.Equals(relativePath, SkillManifestParser.SkillFileName, StringComparison.OrdinalIgnoreCase))
                return SkillFileKind.Manifest;

            int slash = relativePath.IndexOf('/');
            string top = slash < 0 ? string.Empty : relativePath.Substring(0, slash);
            switch (top.ToLowerInvariant())
            {
                case "scripts":
                case "script":
                case "bin":
                    return SkillFileKind.Script;
                case "references":
                case "reference":
                case "docs":
                case "examples":
                    return SkillFileKind.Reference;
                case "assets":
                case "templates":
                case "themes":
                    return SkillFileKind.Asset;
            }

            string ext = Path.GetExtension(relativePath).ToLowerInvariant();
            return ext switch
            {
                ".py" or ".sh" or ".bash" or ".zsh" or ".ps1" or ".js" or ".mjs" or ".ts" or ".rb" or ".pl"
                    => SkillFileKind.Script,
                ".md" or ".markdown" or ".txt" or ".rst" or ".adoc"
                    => SkillFileKind.Reference,
                ".png" or ".jpg" or ".jpeg" or ".gif" or ".svg" or ".pdf" or ".woff" or ".woff2" or ".ttf" or ".otf"
                    => SkillFileKind.Asset,
                _ => SkillFileKind.Other,
            };
        }

        internal static bool IsLicenseFile(string relativePath)
        {
            string name = Path.GetFileNameWithoutExtension(relativePath);
            return relativePath.IndexOf('/') < 0
                && (name.Equals("LICENSE", StringComparison.OrdinalIgnoreCase)
                    || name.Equals("LICENCE", StringComparison.OrdinalIgnoreCase)
                    || name.Equals("COPYING", StringComparison.OrdinalIgnoreCase)
                    || name.Equals("NOTICE", StringComparison.OrdinalIgnoreCase));
        }

        internal static bool IsTextExtension(string extension) => TextExtensions.Contains(extension);

        private static readonly HashSet<string> TextExtensions = new(StringComparer.OrdinalIgnoreCase)
        {
            ".md", ".markdown", ".txt", ".rst", ".adoc", ".json", ".jsonc", ".yaml", ".yml", ".toml", ".ini",
            ".cfg", ".conf", ".csv", ".tsv", ".xml", ".html", ".htm", ".css", ".scss", ".svg",
            ".py", ".pyi", ".sh", ".bash", ".zsh", ".fish", ".ps1", ".bat", ".cmd",
            ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".rb", ".pl", ".php", ".lua", ".r",
            ".c", ".h", ".cc", ".cpp", ".hpp", ".cs", ".java", ".kt", ".go", ".rs", ".swift", ".m", ".mm",
            ".sql", ".graphql", ".proto", ".env", ".gitignore", ".editorconfig", ".dockerfile", ".make", ".mk",
        };

        private static readonly HashSet<string> BinaryExtensions = new(StringComparer.OrdinalIgnoreCase)
        {
            ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".tif", ".tiff", ".heic",
            ".pdf", ".zip", ".gz", ".tgz", ".bz2", ".xz", ".7z", ".rar", ".tar",
            ".woff", ".woff2", ".ttf", ".otf", ".eot",
            ".mp3", ".wav", ".ogg", ".flac", ".mp4", ".mov", ".avi", ".webm", ".mkv",
            ".so", ".dll", ".dylib", ".exe", ".bin", ".o", ".a", ".lib", ".pyc", ".class", ".wasm",
            ".xlsx", ".xlsm", ".docx", ".pptx", ".odt", ".ods", ".odp",
        };
    }

    /// <summary>
    /// One page of a skill file's text.
    /// </summary>
    /// <param name="Path">The file's path relative to the skill root.</param>
    /// <param name="Text">The decoded page.</param>
    /// <param name="TotalBytes">The file's full size on disk.</param>
    /// <param name="OffsetBytes">Where this page started.</param>
    /// <param name="NextOffsetBytes">
    /// Where the next page starts. Handed back to the model as the cursor for a
    /// follow-up read, so it can finish a long reference file rather than acting on
    /// half of it.
    /// </param>
    /// <param name="Truncated">True when more of the file remains.</param>
    public readonly record struct SkillResourceContent(
        string Path,
        string Text,
        long TotalBytes,
        long OffsetBytes,
        long NextOffsetBytes,
        bool Truncated)
    {
        /// <summary>A one-line human summary, used in logs and in the tool result footer.</summary>
        public string Describe() => Truncated
            ? string.Create(CultureInfo.InvariantCulture,
                $"{Path} (bytes {OffsetBytes}-{NextOffsetBytes} of {TotalBytes}; more remains)")
            : string.Create(CultureInfo.InvariantCulture, $"{Path} ({TotalBytes} bytes)");
    }
}
