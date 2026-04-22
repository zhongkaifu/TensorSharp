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
using TensorSharp.Models;

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
    internal static class ChatMessageParser
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

                if (msgEl.TryGetProperty("imagePaths", out var imgs) && imgs.GetArrayLength() > 0)
                    msg.ImagePaths = imgs.EnumerateArray().Select(e => e.GetString()).ToList();

                if (msgEl.TryGetProperty("audioPaths", out var auds) && auds.GetArrayLength() > 0)
                    msg.AudioPaths = auds.EnumerateArray().Select(e => e.GetString()).ToList();

                // Text uploads inline their content into msg.Content; the original file
                // paths are surfaced separately so the per-turn audit log can record
                // which uploaded files belonged to this message.
                if (msgEl.TryGetProperty("textFilePaths", out var texts) && texts.GetArrayLength() > 0)
                    msg.TextFilePaths = texts.EnumerateArray().Select(e => e.GetString()).ToList();

                if (msgEl.TryGetProperty("isVideo", out var iv))
                    msg.IsVideo = iv.GetBoolean();

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Parse Ollama's messages array. Ollama embeds images per-message as a
        /// base64 array under <c>"images"</c>; we materialise each one as a PNG
        /// in the upload directory and reference them by absolute path.
        /// </summary>
        public static List<ChatMessage> ParseOllama(JsonElement messagesEl, string uploadDir)
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

                        string path = WriteBase64Image(b64, uploadDir);
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
        public static List<ChatMessage> ParseOpenAI(JsonElement messagesEl, string uploadDir)
        {
            var messages = new List<ChatMessage>();
            foreach (var msgEl in messagesEl.EnumerateArray())
            {
                var msg = new ChatMessage
                {
                    Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user"
                };

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

                        foreach (var part in contentEl.EnumerateArray())
                        {
                            string type = part.TryGetProperty("type", out var t) ? t.GetString() : "";
                            if (type == "text" && part.TryGetProperty("text", out var txt))
                            {
                                textParts.Add(txt.GetString());
                            }
                            else if (type == "image_url" && part.TryGetProperty("image_url", out var imgUrl))
                            {
                                string url = imgUrl.TryGetProperty("url", out var u) ? u.GetString() : "";
                                if (!string.IsNullOrEmpty(url) && url.StartsWith("data:"))
                                {
                                    int commaIdx = url.IndexOf(',');
                                    if (commaIdx > 0)
                                    {
                                        string b64 = url.Substring(commaIdx + 1);
                                        string path = WriteBase64Image(b64, uploadDir);
                                        msg.ImagePaths.Add(path);
                                    }
                                }
                            }
                        }

                        msg.Content = string.Join("\n", textParts);
                        if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
                    }
                }

                messages.Add(msg);
            }
            return messages;
        }

        /// <summary>
        /// Decode the top-level <c>"images"</c> base64 array used by Ollama's
        /// <c>/api/generate</c>. Returns null when no images are present so the
        /// downstream code path can short-circuit cleanly.
        /// </summary>
        public static List<string> DecodeBase64Images(JsonElement body, string uploadDir)
        {
            if (!body.TryGetProperty("images", out var imgs) || imgs.ValueKind != JsonValueKind.Array)
                return null;

            var paths = new List<string>();
            foreach (var imgEl in imgs.EnumerateArray())
            {
                string b64 = imgEl.GetString();
                if (string.IsNullOrEmpty(b64))
                    continue;

                paths.Add(WriteBase64Image(b64, uploadDir));
            }
            return paths.Count > 0 ? paths : null;
        }

        private static string WriteBase64Image(string base64, string uploadDir)
        {
            byte[] imgData = Convert.FromBase64String(base64);
            string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}.png");
            File.WriteAllBytes(path, imgData);
            return path;
        }
    }
}
