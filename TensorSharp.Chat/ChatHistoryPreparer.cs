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
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorSharp.Server
{
    internal static class ChatHistoryPreparer
    {
        private const string FileBackedCsvPrefix = "[Attached CSV available to tools:";

        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch)
            => PrepareHistoryForInference(history, arch, NullLogger.Instance);

        public static List<ChatMessage> PrepareHistoryForInference(List<ChatMessage> history, string arch, ILogger logger)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                var normalized = NormalizeMessageForInference(history[i], arch, logger);
                if (ReferenceEquals(normalized, history[i]))
                    continue;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = normalized;
            }

            return prepared ?? history;
        }

        /// <summary>
        /// Replace browser-inlined CSV rows with a compact reference when the named
        /// uploads have already been staged into the tool workspace. Every row remains
        /// available without spending the model context on the whole table.
        /// </summary>
        /// <remarks>
        /// This runs on the server rather than only in the page. Saved conversations
        /// created by older builds already contain the large <c>[File: ...]</c>
        /// envelope and would otherwise reproduce the overflow whenever reopened.
        /// Non-CSV documents and files that were not successfully staged retain their
        /// existing inline behavior; data is never silently discarded when no tool can
        /// read the file. The returned inference-only clone no longer marks the file as
        /// inlined text, so ordinary history trimming can resume on later turns. The
        /// original request and its attachment metadata remain untouched for persistence
        /// and audit logging.
        /// </remarks>
        internal static List<ChatMessage> UseFileBackedCsvAttachments(
            List<ChatMessage> history,
            IReadOnlyDictionary<string, string> stagedFiles)
        {
            if (history == null || history.Count == 0 ||
                stagedFiles == null || stagedFiles.Count == 0)
            {
                return history;
            }

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                ChatMessage message = history[i];
                if (message?.TextFilePaths is not { Count: > 0 } ||
                    string.Equals(message.Role, "user", StringComparison.OrdinalIgnoreCase) == false)
                {
                    continue;
                }

                List<AttachedTextFile> attached = AttachedTextFiles(message);
                List<AttachedTextFile> csvFiles = attached.Where(file => file.IsCsv).ToList();
                if (csvFiles.Count == 0 ||
                    csvFiles.Any(file => !stagedFiles.ContainsKey(file.Path)))
                    continue;

                string content = message.Content ?? string.Empty;
                if (content.StartsWith(FileBackedCsvPrefix, StringComparison.Ordinal))
                    continue;

                List<string> referencedCsvNames = csvFiles
                    .Select(file => stagedFiles[file.Path])
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .ToList();
                var csvDisplayNames = new HashSet<string>(
                    csvFiles.Select(file => file.Name), StringComparer.OrdinalIgnoreCase);
                var csvPaths = new HashSet<string>(
                    csvFiles.Select(file => file.Path), StringComparer.Ordinal);

                // Both bundled pages put inlined text envelopes before the user's own
                // words. Parse at most the structured attachment count and remove only
                // staged CSV envelopes; a Markdown/PDF attached beside the table stays
                // inline and keeps its document-preservation semantics.
                if (content.StartsWith("[File: ", StringComparison.Ordinal) &&
                    TrySplitLeadingFileEnvelopes(
                        content, attached.Count, out List<FileEnvelope> envelopes, out string trailing))
                {
                    // Old clients did not send TextFileNames. When every structured
                    // text attachment has an envelope, their documented ordering gives
                    // us a safe fallback for a header that uses the friendly name while
                    // the path carries a generated storage name.
                    bool positionalMapping = envelopes.Count == attached.Count;
                    var kept = new List<string>();
                    for (int envelopeIndex = 0; envelopeIndex < envelopes.Count; envelopeIndex++)
                    {
                        FileEnvelope envelope = envelopes[envelopeIndex];
                        bool isCsv = csvDisplayNames.Contains(envelope.Name)
                            || (positionalMapping && attached[envelopeIndex].IsCsv);
                        if (!isCsv)
                            kept.Add(envelope.Content);
                    }
                    if (trailing.Length > 0)
                        kept.Add(trailing);
                    content = string.Join("\n\n", kept);
                }

                string names = string.Join(", ", referencedCsvNames.Select(QuoteFileName));
                string noun = referencedCsvNames.Count == 1 ? "file is" : "files are";
                string reference = FileBackedCsvPrefix + " " + names + "]\n"
                    + "The complete attached " + noun + " in the working directory. Before answering about "
                    + "the data, you must inspect it with read_file, shell, or an applicable table-analysis skill. "
                    + "Compute over all rows when reporting statistics; do not infer from the filename or a row "
                    + "preview. The rows are intentionally kept out of this prompt so a large table does not "
                    + "consume the model context.\n"
                    + "[End of attached CSV reference]";

                ChatMessage copy = CloneShallow(message);
                copy.Content = content.Length == 0 ? reference : reference + "\n\n" + content;
                // Offsets into the old, inlined body no longer name the same bytes.
                copy.ContentCacheBreakpoints = null;
                // These fields mean "the bytes are inline in Content". Remove the CSV
                // entries after externalising them, while retaining any companion prose
                // document that is still inline. Otherwise the overflow guard preserves
                // all prior turns forever. AttachmentPaths remains as provenance; the
                // tool plan already owns the CodeInputFile instances.
                RemoveFileBackedCsvMetadata(message, copy, csvPaths);
                copy.HasFileBackedTextAttachments = false;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = copy;
            }

            return prepared ?? history;
        }

        /// <summary>
        /// Put metadata-only CSV uploads back into the old full-inline shape when this
        /// request cannot offer a readable tool workspace. This is a compatibility and
        /// safety fallback: a small CSV still works with a non-tool model, while a large
        /// one reaches the normal attached-document context check and is rejected
        /// honestly instead of inviting an answer about rows the model never received.
        /// </summary>
        internal static List<ChatMessage> RestoreUnstagedFileBackedCsvAttachments(
            List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return history;

            List<ChatMessage> prepared = null;
            for (int i = 0; i < history.Count; i++)
            {
                ChatMessage message = history[i];
                if (message?.HasFileBackedTextAttachments != true ||
                    message.TextFilePaths is not { Count: > 0 })
                {
                    continue;
                }

                List<AttachedTextFile> csvFiles = AttachedTextFiles(message)
                    .Where(file => file.IsCsv)
                    .ToList();
                if (csvFiles.Count == 0)
                    continue;

                var envelopes = new List<string>();
                // The display name is not an identity. Two uploads from different
                // directories may both be called responses.csv, and the tool workspace
                // deliberately refuses to collapse those onto one file. Preserve both
                // bodies in this no-tools fallback; only an exact repeated source path
                // is redundant.
                var seenPaths = new HashSet<string>(StringComparer.Ordinal);
                foreach (AttachedTextFile file in csvFiles)
                {
                    if (!seenPaths.Add(file.Path))
                        continue;
                    string fullText = TextUploadHelper.PreserveFullText(File.ReadAllText(file.Path));
                    envelopes.Add("[File: " + file.Name + "]\n" + fullText + "\n[End of file]");
                }

                ChatMessage copy = CloneShallow(message);
                string prefix = string.Join("\n\n", envelopes);
                copy.Content = string.IsNullOrEmpty(message.Content)
                    ? prefix
                    : prefix + "\n\n" + message.Content;
                copy.ContentCacheBreakpoints = null;
                copy.HasFileBackedTextAttachments = false;

                prepared ??= new List<ChatMessage>(history);
                prepared[i] = copy;
            }

            return prepared ?? history;
        }

        private readonly record struct AttachedTextFile(string Path, string Name, bool IsCsv);

        private readonly record struct FileEnvelope(string Name, string Content);

        private static List<AttachedTextFile> AttachedTextFiles(ChatMessage message)
        {
            var result = new List<AttachedTextFile>();
            for (int i = 0; i < message.TextFilePaths.Count; i++)
            {
                string path = message.TextFilePaths[i] ?? string.Empty;
                string supplied = message.TextFileNames != null && i < message.TextFileNames.Count
                    ? message.TextFileNames[i]
                    : null;
                string name = Path.GetFileName(string.IsNullOrWhiteSpace(supplied) ? path : supplied);
                if (name.Length == 0)
                    continue;
                bool isCsv = string.Equals(Path.GetExtension(name), ".csv", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(Path.GetExtension(path), ".csv", StringComparison.OrdinalIgnoreCase);
                result.Add(new AttachedTextFile(path, name, isCsv));
            }
            return result;
        }

        private static void RemoveFileBackedCsvMetadata(
            ChatMessage source, ChatMessage copy, IReadOnlySet<string> csvPaths)
        {
            var paths = new List<string>();
            List<string> names = source.TextFileNames != null ? new List<string>() : null;
            for (int i = 0; i < source.TextFilePaths.Count; i++)
            {
                string path = source.TextFilePaths[i] ?? string.Empty;
                string supplied = source.TextFileNames != null && i < source.TextFileNames.Count
                    ? source.TextFileNames[i]
                    : null;
                if (csvPaths.Contains(path))
                    continue;

                paths.Add(path);
                names?.Add(string.IsNullOrWhiteSpace(supplied) ? path : supplied);
            }

            copy.TextFilePaths = paths.Count > 0 ? paths : null;
            copy.TextFileNames = paths.Count > 0 ? names : null;
        }

        private static bool TrySplitLeadingFileEnvelopes(
            string content,
            int maximumEnvelopeCount,
            out List<FileEnvelope> envelopes,
            out string remainder)
        {
            const string startMarker = "[File: ";
            const string endMarker = "\n[End of file]";
            envelopes = new List<FileEnvelope>();
            int cursor = 0;

            while (envelopes.Count < maximumEnvelopeCount)
            {
                while (cursor < content.Length && (content[cursor] == '\r' || content[cursor] == '\n'))
                    cursor++;

                if (!content.AsSpan(cursor).StartsWith(startMarker, StringComparison.Ordinal))
                    break;

                int headerEnd = content.IndexOf('\n', cursor);
                int nameEnd = headerEnd < 0
                    ? -1
                    : content.IndexOf(']', cursor + startMarker.Length);
                int envelopeEnd = headerEnd < 0
                    ? -1
                    : content.IndexOf(endMarker, headerEnd, StringComparison.Ordinal);
                if (nameEnd < 0 || nameEnd > headerEnd || envelopeEnd < 0)
                {
                    remainder = content;
                    envelopes.Clear();
                    return false;
                }

                string name = Path.GetFileName(
                    content.Substring(cursor + startMarker.Length, nameEnd - cursor - startMarker.Length));
                int afterEnvelope = envelopeEnd + endMarker.Length;
                envelopes.Add(new FileEnvelope(name, content.Substring(cursor, afterEnvelope - cursor)));
                cursor = afterEnvelope;
            }

            remainder = content[cursor..].TrimStart('\r', '\n');
            return true;
        }

        private static string QuoteFileName(string name)
        {
            // A file name belongs in prose, not in prompt structure. Strip control
            // characters so a crafted name cannot manufacture another instruction line.
            string safe = new(name.Where(c => !char.IsControl(c)).ToArray());
            return "'" + safe.Replace("'", "’", StringComparison.Ordinal) + "'";
        }

        public static bool HasMultimodalContent(ChatMessage msg)
        {
            if (msg == null) return false;
            return (msg.ImagePaths != null && msg.ImagePaths.Count > 0) ||
                   (msg.AudioPaths != null && msg.AudioPaths.Count > 0);
        }

        public static bool HasMultimodalContent(List<ChatMessage> history)
        {
            if (history == null || history.Count == 0)
                return false;

            return history.Any(HasMultimodalContent);
        }

        public static List<string> GetImagePathsInPromptOrder(List<ChatMessage> history)
        {
            var imagePaths = new List<string>();
            if (history == null)
                return imagePaths;

            foreach (var msg in history)
            {
                if (msg.ImagePaths == null)
                    continue;

                foreach (var path in msg.ImagePaths)
                {
                    if (!string.IsNullOrEmpty(path))
                        imagePaths.Add(path);
                }
            }

            return imagePaths;
        }

        private static ChatMessage NormalizeMessageForInference(ChatMessage msg, string arch, ILogger logger)
        {
            int maxVideoFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            // maxVideoFrames <= 0 means "no cap" (pure time-based extraction); leave history untouched.
            // Whether a family expands a video into per-frame images (and so needs the
            // cap) is declared with the rest of its chat protocol, not matched on here.
            bool capsFrames = ChatProtocolRegistry.For(arch)?.CapsVideoFrames ?? false;
            if (!capsFrames || maxVideoFrames <= 0 || !msg.IsVideo || msg.ImagePaths == null || msg.ImagePaths.Count <= maxVideoFrames)
                return msg;

            var sampledIndices = MediaHelper.SelectEvenlySpacedIndices(msg.ImagePaths.Count, maxVideoFrames);
            var sampled = sampledIndices.Select(i => msg.ImagePaths[i]).ToList();

            // A Warning, not an Information: frames the user sent are being thrown away,
            // and the answer may miss what happened between the kept ones.
            (logger ?? NullLogger.Instance).LogWarning(LogEventIds.VideoFrameDownsample,
                "video.downsample originalFrames={OriginalFrames} sampledFrames={SampledFrames} architecture={Architecture}: " +
                "the video exceeds the per-message frame cap, so only the sampled, evenly spaced frames reach the model " +
                "and detail between them is lost. Raise VIDEO_MAX_FRAMES to keep more.",
                msg.ImagePaths.Count, sampled.Count, arch);

            return new ChatMessage
            {
                Role = msg.Role,
                Content = msg.Content,
                ImagePaths = sampled,
                ImageTimestamps = msg.ImageTimestamps?.Count == msg.ImagePaths.Count
                    ? sampledIndices.Select(i => msg.ImageTimestamps[i]).ToList() : null,
                AudioPaths = msg.AudioPaths != null ? new List<string>(msg.AudioPaths) : null,
                TextFilePaths = msg.TextFilePaths != null ? new List<string>(msg.TextFilePaths) : null,
                TextFileNames = msg.TextFileNames != null ? new List<string>(msg.TextFileNames) : null,
                HasFileBackedTextAttachments = msg.HasFileBackedTextAttachments,
                IsVideo = msg.IsVideo,
                ToolCalls = msg.ToolCalls,
                ToolCallId = msg.ToolCallId,
                Thinking = msg.Thinking,
                RawOutputTokens = msg.RawOutputTokens,
                RawPromptTrailingWhitespace = msg.RawPromptTrailingWhitespace,
                RawGenerationSuffix = msg.RawGenerationSuffix,
                CacheControl = msg.CacheControl,
                ContentCacheBreakpoints = msg.ContentCacheBreakpoints != null
                    ? new List<int>(msg.ContentCacheBreakpoints)
                    : null,
            };
        }

        private static ChatMessage CloneShallow(ChatMessage src)
        {
            return new ChatMessage
            {
                Role = src.Role,
                Content = src.Content,
                ImagePaths = src.ImagePaths,
                ImageTimestamps = src.ImageTimestamps,
                AudioPaths = src.AudioPaths,
                TextFilePaths = src.TextFilePaths,
                TextFileNames = src.TextFileNames,
                HasFileBackedTextAttachments = src.HasFileBackedTextAttachments,
                AttachmentPaths = src.AttachmentPaths,
                AttachmentNames = src.AttachmentNames,
                IsVideo = src.IsVideo,
                ToolCalls = src.ToolCalls,
                ToolCallId = src.ToolCallId,
                Thinking = src.Thinking,
                RawOutputTokens = src.RawOutputTokens,
                RawPromptTrailingWhitespace = src.RawPromptTrailingWhitespace,
                RawGenerationSuffix = src.RawGenerationSuffix,
                CacheControl = src.CacheControl != null
                    ? new CacheControlMarker { Type = src.CacheControl.Type }
                    : null,
                ContentCacheBreakpoints = src.ContentCacheBreakpoints != null
                    ? new List<int>(src.ContentCacheBreakpoints)
                    : null,
            };
        }
    }
}
