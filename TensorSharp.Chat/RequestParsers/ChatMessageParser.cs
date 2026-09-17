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
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Pure parsers that translate request bodies for each protocol into the
    /// shared <see cref="ChatMessage"/> shape consumed by <see cref="ModelService"/>.
    ///
    /// The Web UI sends already-uploaded file paths inline, while the Ollama and
    /// OpenAI flavours embed binary content as base64. Whenever the parser
    /// materialises new bytes to disk it writes them into the shared upload
    /// directory so subsequent attachments (and ChatStream's media injector)
    /// resolve the same way regardless of how the request arrived.
    /// </summary>
    internal static partial class ChatMessageParser
    {
        /// <summary>
        /// Parse the messages array from the Web UI's <c>/api/chat</c> body.
        /// The UI references previously uploaded files by absolute path, so we
        /// don't decode any binary content here.
        /// </summary>
        public static List<ChatMessage> ParseWebUi(JsonElement messagesEl)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.GetProperty("role").GetString(),
                    Content = msgEl.GetProperty("content").GetString()
                };

                msg.ImagePaths = StringList(msgEl, "imagePaths");
                msg.AudioPaths = StringList(msgEl, "audioPaths");

                // Text uploads usually inline their content into msg.Content. Large CSV
                // tables use the explicit fileBacked attachment marker instead; either
                // way, the paths identify the upload for staging and audit logging.
                msg.TextFilePaths = StringList(msgEl, "textFilePaths");

                // The names the user knows those files by ("report.md" rather than the
                // stored GUID), same order as textFilePaths. Optional: older clients
                // don't send it, and code execution then stages under the stored name.
                msg.TextFileNames = StringList(msgEl, "textFileNames");

                if (msgEl.TryGetProperty("isVideo", out var iv)
                    && (iv.ValueKind == JsonValueKind.True || iv.ValueKind == JsonValueKind.False))
                {
                    msg.IsVideo = iv.GetBoolean();
                }

                ReadAttachments(msgEl, msg);

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// One optional array of strings out of a message, or null.
        ///
        /// <para>
        /// The <c>ValueKind</c> check is the whole point, and it was missing from four
        /// of these five fields. A message may carry <c>"imagePaths": null</c> — the
        /// saved-conversation payload serializes every unset list that way — and
        /// <see cref="JsonElement.GetArrayLength"/> on a null THROWS, so the request
        /// died with a 500 inside the parser. What that looked like from the phone was a
        /// chat that worked until it was reopened and then refused every message,
        /// because the page sends the history it was handed back.
        /// </para>
        /// </summary>
        private static List<string> StringList(JsonElement msgEl, string name)
        {
            if (!msgEl.TryGetProperty(name, out JsonElement list)
                || list.ValueKind != JsonValueKind.Array
                || list.GetArrayLength() == 0)
            {
                return null;
            }
            return list.EnumerateArray().Select(e => e.GetString()).ToList();
        }

        /// <summary>
        /// The files the user attached to one message, as (stored name, display name)
        /// pairs, out of the client's <c>attachments</c> array.
        ///
        /// <para>
        /// A client that does not send one — the desktop page, and any older build —
        /// still has its text uploads staged, because <c>textFilePaths</c> is read as the
        /// fallback. What such a client cannot have is an image staged as a file, which
        /// is the whole reason the array exists: the paths are already in the body twice
        /// over, but only this one says which of them the USER attached (rather than a
        /// frame the server extracted) and what they call it.
        /// </para>
        /// </summary>
        private static void ReadAttachments(JsonElement msgEl, ChatMessage msg)
        {
            if (msgEl.TryGetProperty("attachments", out var attachments)
                && attachments.ValueKind == JsonValueKind.Array
                && attachments.GetArrayLength() > 0)
            {
                var paths = new List<string>();
                var names = new List<string>();
                foreach (var attachment in attachments.EnumerateArray())
                {
                    if (attachment.ValueKind != JsonValueKind.Object)
                        continue;
                    string file = attachment.TryGetProperty("file", out var f) ? f.GetString() : null;
                    if (string.IsNullOrWhiteSpace(file))
                        continue;
                    string name = attachment.TryGetProperty("fileName", out var n) ? n.GetString() : null;
                    string mediaType = attachment.TryGetProperty("mediaType", out var mt)
                        ? mt.GetString()
                        : null;
                    bool fileBacked = attachment.TryGetProperty("fileBacked", out var fb)
                        && fb.ValueKind == JsonValueKind.True;
                    if (fileBacked &&
                        string.Equals(mediaType, "text", StringComparison.OrdinalIgnoreCase) &&
                        (string.Equals(Path.GetExtension(file), ".csv", StringComparison.OrdinalIgnoreCase) ||
                         string.Equals(Path.GetExtension(name), ".csv", StringComparison.OrdinalIgnoreCase)))
                    {
                        msg.HasFileBackedTextAttachments = true;
                    }
                    paths.Add(file);
                    names.Add(string.IsNullOrWhiteSpace(name) ? file : name);
                }

                if (paths.Count > 0)
                {
                    msg.AttachmentPaths = paths;
                    msg.AttachmentNames = names;
                    return;
                }
            }

            if (msg.TextFilePaths is { Count: > 0 })
            {
                msg.AttachmentPaths = new List<string>(msg.TextFilePaths);
                msg.AttachmentNames = msg.TextFileNames != null
                    ? new List<string>(msg.TextFileNames)
                    : new List<string>(msg.TextFilePaths);
            }
        }

        /// <summary>
        /// Resolve every client-supplied attachment reference (imagePaths /
        /// audioPaths / textFilePaths / attachments) to a full path inside the upload
        /// directory, rewriting the lists in place. The Web UI sends the bare
        /// server filenames returned by <c>/api/upload</c>; absolute paths from
        /// older clients are still accepted when they resolve inside the upload
        /// directory. Returns an error message, or null when everything
        /// resolved.
        /// </summary>
        public static string ResolveAttachmentPaths(List<ChatMessage> messages, string uploadRoot)
        {
            if (messages == null)
                return null;

            foreach (var msg in messages)
            {
                string error = ResolvePathList(msg?.ImagePaths, uploadRoot)
                    ?? ResolvePathList(msg?.AudioPaths, uploadRoot)
                    ?? ResolvePathList(msg?.TextFilePaths, uploadRoot)
                    ?? ResolvePathList(msg?.AttachmentPaths, uploadRoot);
                if (error != null)
                    return error;
            }
            return null;
        }

        private static string ResolvePathList(List<string> paths, string uploadRoot)
        {
            if (paths == null)
                return null;

            for (int i = 0; i < paths.Count; i++)
            {
                if (!UploadFileReference.TryResolve(uploadRoot, paths[i], out string full))
                    return "Attachment path must reference a previously uploaded file.";
                paths[i] = full;
            }
            return null;
        }

        /// <summary>
        /// Parse Ollama's messages array. Ollama embeds images per-message as a
        /// base64 array under <c>"images"</c>; we materialise each one as a PNG
        /// in the upload directory and reference them by absolute path.
        /// </summary>
        public static List<ChatMessage> ParseOllama(JsonElement messagesEl, UploadStoragePolicy uploads)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user",
                    Content = msgEl.TryGetProperty("content", out var c) ? c.GetString() : ""
                };

                if (msgEl.TryGetProperty("images", out var imgs) && imgs.ValueKind == JsonValueKind.Array)
                {
                    msg.ImagePaths = new List<string>();
                    foreach (var imgEl in imgs.EnumerateArray())
                    {
                        string b64 = imgEl.GetString();
                        if (string.IsNullOrEmpty(b64))
                            continue;

                        string path = WriteBase64Image(b64, uploads);
                        msg.ImagePaths.Add(path);
                    }
                }

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Parse OpenAI's messages array which uses an "input parts" structure:
        /// either a plain string content, or an array of parts where each part
        /// is either text or an image (data URL or external URL).
        /// </summary>
        public static List<ChatMessage> ParseOpenAI(JsonElement messagesEl, UploadStoragePolicy uploads, ILogger logger = null,
            string architecture = null, bool audioEncoderLoaded = false)
        {
            // Scan the complete request before writing any uploads: an image
            // preceding unsupported audio or an invalid image must not leave
            // partial files, or silently disappear from the model's input.
            // The audio gate covers every family without an audio tower
            // (DeepSeek V4.1, Nemotron-H); the image checks are V4.1's own.
            string audioError = ChatGenerationPipeline.AudioInputErrorFor(architecture, audioEncoderLoaded);
            bool deepSeek41 = string.Equals(architecture, "deepseek41", StringComparison.OrdinalIgnoreCase);
            if (audioError != null || deepSeek41)
            {
                foreach (JsonElement message in messagesEl.EnumerateArray())
                    if (message.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.Array)
                        foreach (JsonElement part in content.EnumerateArray())
                        {
                            if (part.TryGetProperty("type", out JsonElement type) &&
                                type.ValueKind == JsonValueKind.String)
                            {
                                if (audioError != null && type.GetString() is "input_audio" or "audio_url")
                                    throw new JsonException(audioError);
                                if (deepSeek41 && type.GetString() == "image_url")
                                    ValidateDeepSeek41ImageUrl(part);
                            }
                        }
            }

            var messages = new List<ChatMessage>();
            int droppedImages = 0;
            int droppedAudio = 0;
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user"
                };

                if (CacheControlParser.TryParse(msgEl, out var msgMarker))
                    msg.CacheControl = msgMarker;

                if (msgEl.TryGetProperty("content", out var contentEl))
                {
                    if (contentEl.ValueKind == JsonValueKind.String)
                    {
                        msg.Content = contentEl.GetString();
                    }
                    else if (contentEl.ValueKind == JsonValueKind.Array)
                    {
                        var textParts = new List<string>();
                        msg.ImagePaths = new List<string>();
                        msg.AudioPaths = new List<string>();
                        // Running length of the string.Join("\n", textParts) built
                        // below, so a part's cache_control marker can be recorded at
                        // the offset where that part ends rather than at the end of
                        // the whole message.
                        int joinedLength = 0;

                        foreach (var part in contentEl.EnumerateArray())
                        {
                            string type = part.TryGetProperty("type", out var t) ? t.GetString() : "";
                            if (type == "text" && part.TryGetProperty("text", out var txt))
                            {
                                string text = txt.GetString() ?? string.Empty;
                                if (textParts.Count > 0) joinedLength++; // the "\n" separator
                                joinedLength += text.Length;
                                textParts.Add(text);
                                if (CacheControlParser.TryParse(part, out _))
                                    msg.AddContentCacheBreakpoint(joinedLength);
                            }
                            else if (type == "image_url" && part.TryGetProperty("image_url", out var imgUrl))
                            {
                                string url = imgUrl.TryGetProperty("url", out var u) ? u.GetString() : "";
                                int commaIdx = !string.IsNullOrEmpty(url) && url.StartsWith("data:")
                                    ? url.IndexOf(',')
                                    : -1;
                                if (commaIdx > 0)
                                {
                                    string b64 = url.Substring(commaIdx + 1);
                                    string path = WriteBase64Image(b64, uploads);
                                    msg.ImagePaths.Add(path);
                                    msg.ImageTimestamps?.Add(null);
                                }
                                else if (!string.IsNullOrEmpty(url))
                                {
                                    // http(s) or a malformed data: URI - never fetched or
                                    // decoded; counted so the drop is reported below.
                                    droppedImages++;
                                }
                            }
                            else if (type == "video_url" && part.TryGetProperty("video_url", out var videoUrl))
                            {
                                if (ChatProtocolRegistry.For(architecture)?.CapsVideoFrames != true)
                                    throw new JsonException("video_url frame sampling is not supported by this model's chat protocol.");
                                AppendSampledVideo(msg, videoUrl, uploads);
                            }
                            else if (type == "input_audio" && part.TryGetProperty("input_audio", out var audioEl))
                            {
                                // OpenAI audio part: {"data": "<base64>", "format": "wav"|"mp3"}.
                                string path = WriteBase64Audio(
                                    audioEl.TryGetProperty("data", out var aData) ? aData.GetString() : null,
                                    audioEl.TryGetProperty("format", out var aFmt) ? aFmt.GetString() : null,
                                    uploads);
                                if (path != null)
                                    msg.AudioPaths.Add(path);
                                else
                                    droppedAudio++;     // empty or undecodable base64 content
                            }
                            else if (type == "audio_url" && part.TryGetProperty("audio_url", out var audioUrl))
                            {
                                // Some clients mirror the image_url shape for audio.
                                string url = audioUrl.ValueKind == JsonValueKind.String
                                    ? audioUrl.GetString()
                                    : (audioUrl.TryGetProperty("url", out var au) ? au.GetString() : null);
                                string path = WriteBase64AudioDataUri(url, uploads);
                                if (path != null)
                                    msg.AudioPaths.Add(path);
                                else if (!string.IsNullOrEmpty(url))
                                    droppedAudio++;     // http(s) or undecodable data: URI
                            }
                        }

                        msg.Content = string.Join("\n", textParts);
                        if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
                        if (msg.AudioPaths.Count == 0) msg.AudioPaths = null;
                    }
                }

                // Message-level base64 audio array (the same shorthand Ollama uses
                // for images), accepted alongside the OpenAI content-part form.
                AppendMessageLevelAudios(msgEl, msg, uploads);

                ReadOpenAIHistory(msgEl, msg);

                messages.Add(msg);
            }

            if (droppedImages > 0)
            {
                logger?.LogWarning(LogEventIds.RequestContentDropped,
                    "Discarded {Count} image_url part(s): only data: URIs are supported (remote http(s) images are " +
                    "not fetched). The model answers without seeing those images.",
                    droppedImages);
            }
            if (droppedAudio > 0)
            {
                logger?.LogWarning(LogEventIds.RequestContentDropped,
                    "Discarded {Count} audio part(s): only base64 / data: URI audio is supported (remote URLs are not " +
                    "fetched), and undecodable content is skipped. The model answers without hearing that audio.",
                    droppedAudio);
            }
            return messages;
        }

        private static void ValidateDeepSeek41ImageUrl(JsonElement part)
        {
            if (!part.TryGetProperty("image_url", out JsonElement image) ||
                image.ValueKind != JsonValueKind.Object ||
                !image.TryGetProperty("url", out JsonElement value) ||
                value.ValueKind != JsonValueKind.String)
                throw new JsonException("DeepSeek V4.1 image_url must contain a base64 image data URI in url.");

            ValidateDeepSeek41ImageDataUri(value.GetString());
        }

        private static void ValidateDeepSeek41ImageDataUri(string url)
        {
            int comma = !string.IsNullOrEmpty(url) && url.StartsWith("data:", StringComparison.Ordinal)
                ? url.IndexOf(',') : -1;
            if (comma <= 0)
                throw new JsonException("DeepSeek V4.1 image_url supports base64 image data URIs; remote URLs are not fetched.");

            // Preserve the existing data-URI header and base64 whitespace
            // handling. This scan validates without allocating decoded bytes;
            // WriteBase64Image performs the one decode after all parts pass.
            if (!System.Buffers.Text.Base64.IsValid(url.AsSpan(comma + 1), out int decodedLength))
                throw new JsonException("DeepSeek V4.1 image_url contains invalid base64.");
            if (decodedLength == 0)
                throw new JsonException("DeepSeek V4.1 image_url contains an empty image.");
        }

        private static void ReadOpenAIHistory(JsonElement source, ChatMessage message)
        {
            if (source.TryGetProperty("tool_call_id", out var resultId) && resultId.ValueKind == JsonValueKind.String)
                message.ToolCallId = resultId.GetString();
            if ((source.TryGetProperty("reasoning_content", out var reasoning) ||
                 source.TryGetProperty("reasoning", out reasoning)) && reasoning.ValueKind == JsonValueKind.String)
                message.Thinking = reasoning.GetString();

            if (!source.TryGetProperty("tool_calls", out var calls) || calls.ValueKind == JsonValueKind.Null)
                return;
            if (calls.ValueKind != JsonValueKind.Array)
                throw new JsonException("tool_calls must be an array.");

            message.ToolCalls = new List<ToolCall>();
            foreach (JsonElement call in calls.EnumerateArray())
            {
                if (call.ValueKind != JsonValueKind.Object || !call.TryGetProperty("function", out var function) ||
                    function.ValueKind != JsonValueKind.Object ||
                    !function.TryGetProperty("name", out var name) || name.ValueKind != JsonValueKind.String)
                    throw new JsonException("Each tool call requires a function name.");

                Dictionary<string, object> arguments = new();
                if (function.TryGetProperty("arguments", out var value))
                {
                    string json = value.ValueKind == JsonValueKind.String ? value.GetString() : value.GetRawText();
                    arguments = JsonSerializer.Deserialize<Dictionary<string, object>>(json)
                        ?? throw new JsonException("Tool arguments must be a JSON object.");
                }
                message.ToolCalls.Add(new ToolCall
                {
                    Id = call.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.String ? id.GetString() : null,
                    Name = name.GetString(),
                    Arguments = arguments,
                    Index = message.ToolCalls.Count,
                });
            }
        }

        /// <summary>
        /// Parse the Responses API <c>input</c> field, which is either a plain
        /// string (shorthand for a single user turn) or an array of message
        /// items: <c>{role, content: string | [{type:"input_text"|"output_text", text} |
        /// {type:"input_image", image_url}]}</c>. An optional <c>instructions</c>
        /// string is prepended as a system message, matching how the real API
        /// folds it into the model's system prompt.
        /// </summary>
        public static List<ChatMessage> ParseResponsesInput(JsonElement inputEl, string instructions, UploadStoragePolicy uploads, ILogger logger = null,
            string architecture = null, bool audioEncoderLoaded = false)
        {
            string audioError = ChatGenerationPipeline.AudioInputErrorFor(architecture, audioEncoderLoaded);
            bool deepSeek41 = string.Equals(architecture, "deepseek41", StringComparison.OrdinalIgnoreCase);
            if ((audioError != null || deepSeek41) && inputEl.ValueKind == JsonValueKind.Array)
            {
                // Validate every supported message before any media is written.
                // Check the audio type before decoding its payload: missing or
                // malformed unsupported audio must not silently disappear.
                foreach (JsonElement item in inputEl.EnumerateArray())
                {
                    if (item.TryGetProperty("type", out JsonElement itemType) &&
                        (itemType.ValueKind != JsonValueKind.String || itemType.GetString() != "message"))
                        continue;
                    if (item.TryGetProperty("content", out JsonElement content) && content.ValueKind == JsonValueKind.Array)
                        foreach (JsonElement part in content.EnumerateArray())
                            if (part.TryGetProperty("type", out JsonElement type) &&
                                type.ValueKind == JsonValueKind.String)
                            {
                                if (audioError != null && type.GetString() is "input_audio" or "audio_url")
                                    throw new JsonException(audioError);
                                if (deepSeek41 && type.GetString() == "input_image")
                                {
                                    if (!part.TryGetProperty("image_url", out JsonElement url) || url.ValueKind != JsonValueKind.String)
                                        throw new JsonException("DeepSeek V4.1 input_image.image_url must contain a base64 image data URI.");
                                    ValidateDeepSeek41ImageDataUri(url.GetString());
                                }
                            }
                }
            }

            var messages = new List<ChatMessage>();
            Dictionary<string, int> skippedItems = null;
            int droppedImages = 0;

            if (!string.IsNullOrEmpty(instructions))
                messages.Add(new ChatMessage { Role = "system", Content = instructions });

            if (inputEl.ValueKind == JsonValueKind.String)
            {
                messages.Add(new ChatMessage { Role = "user", Content = inputEl.GetString() ?? "" });
                return messages;
            }

            if (inputEl.ValueKind != JsonValueKind.Array)
                return messages;

            foreach (var itemEl in inputEl.EnumerateArray())
            {
                // Only message-shaped items are supported; other item types
                // (function_call, function_call_output, reasoning, ...) are
                // ignored for this MVP surface - counted so the drop is
                // reported below rather than happening in silence.
                string itemType = itemEl.TryGetProperty("type", out var t) ? t.GetString() : "message";
                if (itemType != "message")
                {
                    skippedItems ??= new Dictionary<string, int>(StringComparer.Ordinal);
                    string key = itemType ?? "(none)";
                    skippedItems.TryGetValue(key, out int seen);
                    skippedItems[key] = seen + 1;
                    continue;
                }

                var msg = new ChatMessage
                {
                    Role = itemEl.TryGetProperty("role", out var r) ? r.GetString() : "user",
                };

                if (CacheControlParser.TryParse(itemEl, out var itemMarker))
                    msg.CacheControl = itemMarker;

                if (!itemEl.TryGetProperty("content", out var contentEl))
                {
                    messages.Add(msg);
                    continue;
                }

                if (contentEl.ValueKind == JsonValueKind.String)
                {
                    msg.Content = contentEl.GetString();
                    messages.Add(msg);
                    continue;
                }

                if (contentEl.ValueKind == JsonValueKind.Array)
                {
                    var textParts = new List<string>();
                    msg.ImagePaths = new List<string>();
                    msg.AudioPaths = new List<string>();
                    // See the equivalent loop in ParseOpenAI: the offset of a
                    // part-scoped marker is the running length of the join.
                    int joinedLength = 0;

                    foreach (var part in contentEl.EnumerateArray())
                    {
                        string partType = part.TryGetProperty("type", out var pt) ? pt.GetString() : "";
                        if ((partType == "input_text" || partType == "output_text") &&
                            part.TryGetProperty("text", out var txt))
                        {
                            string text = txt.GetString() ?? string.Empty;
                            if (textParts.Count > 0) joinedLength++; // the "\n" separator
                            joinedLength += text.Length;
                            textParts.Add(text);
                            if (CacheControlParser.TryParse(part, out _))
                                msg.AddContentCacheBreakpoint(joinedLength);
                        }
                        else if (partType == "input_audio" && part.TryGetProperty("input_audio", out var audioEl))
                        {
                            string audioPath = WriteBase64Audio(
                                audioEl.TryGetProperty("data", out var aData) ? aData.GetString() : null,
                                audioEl.TryGetProperty("format", out var aFmt) ? aFmt.GetString() : null,
                                uploads);
                            if (audioPath != null)
                                msg.AudioPaths.Add(audioPath);
                        }
                        else if (partType == "input_image")
                        {
                            string url = part.TryGetProperty("image_url", out var u)
                                ? (u.ValueKind == JsonValueKind.String ? u.GetString() : null)
                                : null;
                            int commaIdx = !string.IsNullOrEmpty(url) && url.StartsWith("data:")
                                ? url.IndexOf(',')
                                : -1;
                            if (commaIdx > 0)
                            {
                                string b64 = url.Substring(commaIdx + 1);
                                msg.ImagePaths.Add(WriteBase64Image(b64, uploads));
                            }
                            else if (!string.IsNullOrEmpty(url))
                            {
                                // http(s) or a malformed data: URI - never fetched or
                                // decoded; counted so the drop is reported below.
                                droppedImages++;
                            }
                        }
                    }

                    msg.Content = string.Join("\n", textParts);
                    if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
                    if (msg.AudioPaths.Count == 0) msg.AudioPaths = null;
                }

                messages.Add(msg);
            }

            if (skippedItems != null)
            {
                logger?.LogWarning(LogEventIds.RequestContentDropped,
                    "/v1/responses input: skipped {Count} unsupported input item(s) (types: {Types}); only 'message' " +
                    "items are used, so their content never reaches the model.",
                    skippedItems.Values.Sum(),
                    string.Join(", ", skippedItems.Select(kv => kv.Key + " x" + kv.Value)));
            }
            if (droppedImages > 0)
            {
                logger?.LogWarning(LogEventIds.RequestContentDropped,
                    "/v1/responses input: discarded {Count} input_image part(s): only data: URIs are supported " +
                    "(remote http(s) images are not fetched). The model answers without seeing those images.",
                    droppedImages);
            }
            return messages;
        }

        /// <summary>
        /// Decode the top-level <c>"images"</c> base64 array used by Ollama's
        /// <c>/api/generate</c>. Returns null when no images are present so the
        /// downstream code path can short-circuit cleanly.
        /// </summary>
        public static List<string>? DecodeBase64Images(JsonElement body, UploadStoragePolicy uploads)
        {
            if (!body.TryGetProperty("images", out var imgs) || imgs.ValueKind != JsonValueKind.Array)
                return null;

            var paths = new List<string>();
            foreach (var imgEl in imgs.EnumerateArray())
            {
                string b64 = imgEl.GetString();
                if (string.IsNullOrEmpty(b64))
                    continue;

                paths.Add(WriteBase64Image(b64, uploads));
            }
            return paths.Count > 0 ? paths : null;
        }

        private static string WriteBase64Image(string base64, UploadStoragePolicy uploads)
        {
            byte[] imgData = Convert.FromBase64String(base64);
            return WriteContentAddressed(imgData, ".png", uploads);
        }

        /// <summary>
        /// Store a decoded attachment under the SHA-256 of its bytes and return the path.
        ///
        /// <para>
        /// API clients resend every image and audio clip of the conversation with each
        /// turn. Written under a fresh random name each time, one picture became a new
        /// file per turn (the upload directory grew without bound) and, since media was
        /// identified by path, a new picture as far as prompt reuse and the embedding
        /// cache could tell: every turn after an image re-prefilled from the image and
        /// re-ran the vision encoder. The same bytes now land on the same file, written
        /// once; a repeat only refreshes its write time so the TTL sweep keeps a file a
        /// conversation still sends.
        /// </para>
        /// </summary>
        internal static string WriteContentAddressed(byte[] data, string extension, UploadStoragePolicy uploads)
        {
            string path = Path.Combine(uploads.DirectoryPath, MediaContentId.OfBytes(data) + extension);
            if (TryReuseStoredCopy(path, data.Length))
                return path;

            uploads.ReserveClientWriteOrThrow(data.Length);
            // Written aside and moved into place, so a reader never sees a partial file
            // under a content name and two concurrent writers of the same bytes agree.
            string temp = path + "." + Guid.NewGuid().ToString("N") + ".partial";
            try
            {
                File.WriteAllBytes(temp, data);
                File.Move(temp, path, overwrite: false);
            }
            catch (IOException) when (TryReuseStoredCopy(path, data.Length))
            {
                // Another request stored the same bytes first.
                TryDelete(temp);
                uploads.Release(data.Length);
            }
            catch
            {
                TryDelete(temp);
                uploads.Release(data.Length);
                throw;
            }
            return path;
        }

        private static bool TryReuseStoredCopy(string path, long length)
        {
            try
            {
                var existing = new FileInfo(path);
                if (!existing.Exists || existing.Length != length)
                    return false;
                existing.LastWriteTimeUtc = DateTime.UtcNow;
                return true;
            }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
            {
                return File.Exists(path);
            }
        }

        private static void TryDelete(string path)
        {
            try { File.Delete(path); }
            catch (Exception ex) when (ex is IOException or UnauthorizedAccessException) { }
        }

        /// <summary>
        /// Materialise a base64 audio attachment in the upload directory. The
        /// audio preprocessor picks its decoder from the file extension, so the
        /// declared <paramref name="format"/> (OpenAI sends "wav" / "mp3") is
        /// mapped to one — falling back to sniffing the container header when it
        /// is missing or unrecognised. Returns null for empty/invalid content.
        /// </summary>
        private static string WriteBase64Audio(string base64, string format, UploadStoragePolicy uploads)
        {
            if (string.IsNullOrWhiteSpace(base64))
                return null;

            // Tolerate a full data URI in the data field.
            int commaIdx = base64.StartsWith("data:") ? base64.IndexOf(',') : -1;
            if (commaIdx > 0)
            {
                if (string.IsNullOrEmpty(format))
                    format = MimeToAudioFormat(base64.Substring(5, commaIdx - 5));
                base64 = base64.Substring(commaIdx + 1);
            }

            byte[] data;
            try
            {
                data = Convert.FromBase64String(base64);
            }
            catch (FormatException)
            {
                return null;
            }
            if (data.Length == 0)
                return null;

            return WriteContentAddressed(data, AudioExtension(format, data), uploads);
        }

        private static string WriteBase64AudioDataUri(string url, UploadStoragePolicy uploads)
        {
            if (string.IsNullOrEmpty(url) || !url.StartsWith("data:"))
                return null;
            return WriteBase64Audio(url, null, uploads);
        }

        private static string MimeToAudioFormat(string mimeAndParams)
        {
            int semi = mimeAndParams.IndexOf(';');
            string mime = semi >= 0 ? mimeAndParams.Substring(0, semi) : mimeAndParams;
            int slash = mime.IndexOf('/');
            return slash >= 0 ? mime.Substring(slash + 1) : mime;
        }

        private static string AudioExtension(string format, byte[] data)
        {
            switch ((format ?? string.Empty).Trim().TrimStart('.').ToLowerInvariant())
            {
                case "wav":
                case "wave":
                case "x-wav":
                case "pcm":
                    return ".wav";
                case "mp3":
                case "mpeg":
                case "mpga":
                    return ".mp3";
                case "ogg":
                case "oga":
                case "vorbis":
                    return ".ogg";
            }

            if (data.Length >= 12 && data[0] == 'R' && data[1] == 'I' && data[2] == 'F' && data[3] == 'F'
                && data[8] == 'W' && data[9] == 'A' && data[10] == 'V' && data[11] == 'E')
                return ".wav";
            if (data.Length >= 4 && data[0] == 'O' && data[1] == 'g' && data[2] == 'g' && data[3] == 'S')
                return ".ogg";
            return ".mp3";                 // ID3-tagged or bare MPEG frames
        }

        /// <summary>
        /// Read a message-level <c>"audios"</c> array of base64 clips (optionally
        /// data URIs) and append them to the message's audio attachments.
        /// </summary>
        private static void AppendMessageLevelAudios(JsonElement msgEl, ChatMessage msg, UploadStoragePolicy uploads)
        {
            if (!msgEl.TryGetProperty("audios", out var auds) || auds.ValueKind != JsonValueKind.Array)
                return;

            foreach (var audEl in auds.EnumerateArray())
            {
                string b64 = audEl.ValueKind == JsonValueKind.String
                    ? audEl.GetString()
                    : (audEl.TryGetProperty("data", out var d) ? d.GetString() : null);
                string format = audEl.ValueKind == JsonValueKind.Object &&
                                audEl.TryGetProperty("format", out var f) ? f.GetString() : null;
                string path = WriteBase64Audio(b64, format, uploads);
                if (path == null)
                    continue;
                (msg.AudioPaths ??= new List<string>()).Add(path);
            }
        }
    }
}
