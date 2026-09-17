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
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;

namespace TensorSharp.Server
{
    /// <summary>
    /// What the server sent the client for one generated assistant turn: the content,
    /// tool calls and reasoning its output parser separated out of the raw text, plus the
    /// raw text itself (what a client that runs no parser received).
    /// </summary>
    internal sealed record EmittedAssistantTurn(
        string Content,
        IReadOnlyList<ToolCall> ToolCalls,
        string Thinking,
        string RawText,
        bool Cancelled);

    /// <summary>The render history <see cref="ConversationTranscriptStore.Augment"/> built,
    /// and the conversation it proved the request continues (null when none).</summary>
    internal readonly record struct TranscriptAugmentation(
        List<ChatMessage> History,
        string InheritedScope,
        int SplicedTurns);

    /// <summary>
    /// The raw output tokens of assistant turns this server generated, kept so the next
    /// request of the same conversation renders exactly the tokens the model's cache
    /// holds (Qwen 3.5's re-render of its own answer is not token-identical to what it
    /// generated, so without them every turn re-prefills its history).
    ///
    /// <para>
    /// It replaced one <c>TrackedHistory</c> list per session. That list was the bug:
    /// every OpenAI Chat, Responses and Ollama request shared one session, as did every
    /// Web UI request without a sessionId, and the splice replaced a client's assistant
    /// message with the tracked raw tokens WITHOUT comparing content. A client whose
    /// history said the assistant answered "Zeppelin" was rendered with another client's
    /// "Quintessence" and answered with it; a Tokyo tool call was rendered as another
    /// conversation's Paris call. And because the last request to finish overwrote the
    /// list, concurrent conversations kept destroying each other's reuse.
    /// </para>
    /// <para>
    /// Now each generated turn is its own record, keyed by the content-hash chain of the
    /// client-visible history that preceded it (roles, content, tool calls and media by
    /// content), so concurrent conversations never overwrite each other. A record is
    /// spliced only when the client's assistant message equals what the server EMITTED
    /// for those tokens (the parsed content and tool calls, or the raw text), compared
    /// ignoring whitespace: an assistant message a client wrote or edited renders from
    /// its own text. Matching a record is also the possession proof that the request
    /// continues that record's conversation, which is how a stateless API request
    /// inherits its conversation's cache scope.
    /// </para>
    /// <para>Thread-safe; bounded by record count and raw-token total (LRU).</para>
    /// </summary>
    internal sealed class ConversationTranscriptStore
    {
        /// <summary>Distinct conversations allowed to share one chain position (the same
        /// history followed by different answers, or the same answer in two scopes).</summary>
        private const int MaxRecordsPerChain = 4;

        private sealed class TurnRecord
        {
            public List<ChatMessage> Replacement;   // the generated message(s), raw tokens attached
            public EmittedAssistantTurn Emitted;
            public string Scope;
            public long Tokens;
        }

        private sealed class ChainEntry
        {
            public string Key;
            public readonly List<TurnRecord> Records = new();   // newest first
            public LinkedListNode<ChainEntry> Node;
        }

        private readonly object _lock = new();
        private readonly Dictionary<string, ChainEntry> _chains = new(StringComparer.Ordinal);
        private readonly LinkedList<ChainEntry> _lru = new();   // most recent at the tail
        private readonly int _maxChains;
        private readonly long _maxTokens;
        private long _tokens;
        private List<ChatMessage> _latestTranscript = new();

        public ConversationTranscriptStore(int maxChains, long maxTokens)
        {
            _maxChains = Math.Max(1, maxChains);
            _maxTokens = Math.Max(1, maxTokens);
        }

        /// <summary>Number of chain positions holding at least one recorded turn.</summary>
        public int Count
        {
            get { lock (_lock) return _chains.Count; }
        }

        /// <summary>The render history of the most recently recorded turn plus that turn,
        /// for diagnostics and the legacy <c>ModelService.TrackedHistory</c> view.</summary>
        public IReadOnlyList<ChatMessage> LatestTranscript
        {
            get { lock (_lock) return _latestTranscript.AsReadOnly(); }
        }

        public void Clear()
        {
            lock (_lock)
            {
                _chains.Clear();
                _lru.Clear();
                _tokens = 0;
                _latestTranscript = new List<ChatMessage>();
            }
        }

        /// <summary>
        /// Rebuild <paramref name="incoming"/> so each assistant turn this server generated
        /// - and the client sent back unchanged - renders from its recorded raw tokens.
        /// A turn that ran the in-process tool loop is recorded as its whole transcript
        /// (assistant round, tool result, ..., final round) and expands in place of the one
        /// clean assistant message the client sends for it. Messages that already carry
        /// raw tokens (the tool loop's own rounds) are left as they are.
        /// </summary>
        public TranscriptAugmentation Augment(IReadOnlyList<ChatMessage> incoming)
        {
            if (incoming == null)
                return new TranscriptAugmentation(null, null, 0);

            var result = new List<ChatMessage>(incoming.Count);
            int visibleEnd = VisibleEnd(incoming);
            byte[] chain = new byte[32];
            string inheritedScope = null;
            int spliced = 0;
            int lastAssistant = -1;
            for (int i = 0; i < visibleEnd; i++)
                if (incoming[i].Role == "assistant")
                    lastAssistant = i;

            lock (_lock)
            {
                for (int i = 0; i < incoming.Count; i++)
                {
                    ChatMessage src = incoming[i];
                    if (i >= visibleEnd || src.Role != "assistant" || src.RawOutputTokens is { Count: > 0 })
                    {
                        result.Add(src);
                        if (i < visibleEnd)
                            chain = Advance(chain, src);
                        continue;
                    }

                    TurnRecord match = null;
                    if (_chains.TryGetValue(Convert.ToHexString(chain), out ChainEntry entry))
                    {
                        foreach (TurnRecord candidate in entry.Records)
                        {
                            if (EmittedMatches(src, candidate))
                            {
                                match = candidate;
                                break;
                            }
                        }
                        if (match != null)
                            Touch(entry);
                    }

                    string nextRole = i + 1 < incoming.Count ? incoming[i + 1].Role : null;
                    if (match == null || (match.Replacement.Count > 1 && nextRole == "tool"))
                    {
                        // A client that sends its own tool messages after this one carries
                        // the transcript itself; expanding would insert the rounds twice.
                        result.Add(src);
                    }
                    else if (match.Replacement.Count == 1)
                    {
                        result.Add(SpliceOnto(src, match.Replacement[0]));
                        spliced++;
                        if (i == lastAssistant)
                            inheritedScope = match.Scope;
                    }
                    else
                    {
                        var expanded = new List<ChatMessage>(match.Replacement.Count);
                        foreach (ChatMessage m in match.Replacement)
                            expanded.Add(CloneShallow(m));
                        PreserveCollapsedCacheMarkers(src, expanded);
                        result.AddRange(expanded);
                        spliced++;
                        if (i == lastAssistant)
                            inheritedScope = match.Scope;
                    }

                    // The chain continues over what the CLIENT sent, so the next turn's
                    // key is computable from its request alone.
                    chain = Advance(chain, src);
                }
            }

            return new TranscriptAugmentation(result, inheritedScope, spliced);
        }

        /// <summary>
        /// Record the turn just generated after <paramref name="history"/> (the request's
        /// history as prepared, before any splice): <paramref name="generated"/> carries the
        /// raw tokens, <paramref name="emitted"/> what the client was sent for them. When
        /// the history ends with the in-process tool loop's own rounds, the record is that
        /// whole transcript, keyed by the client-visible history before it. Earlier records
        /// of other conversations are never touched; a newer record of the same scope at
        /// the same position replaces the older one (a retry or regenerate).
        /// </summary>
        public void Record(
            IReadOnlyList<ChatMessage> history,
            ChatMessage generated,
            EmittedAssistantTurn emitted,
            string scope,
            IReadOnlyList<ChatMessage> renderHistoryForDiagnostics = null)
        {
            if (generated?.RawOutputTokens is not { Count: > 0 } || emitted == null)
                return;
            history ??= Array.Empty<ChatMessage>();
            int visibleEnd = VisibleEnd(history);

            byte[] chain = new byte[32];
            for (int i = 0; i < visibleEnd; i++)
                chain = Advance(chain, history[i]);
            string key = Convert.ToHexString(chain);

            var replacement = new List<ChatMessage>(history.Count - visibleEnd + 1);
            var emittedContent = new StringBuilder();
            long tokens = generated.RawOutputTokens.Count;
            for (int i = visibleEnd; i < history.Count; i++)
            {
                ChatMessage internalMessage = CloneShallow(history[i]);
                replacement.Add(internalMessage);
                if (internalMessage.Role == "assistant")
                {
                    emittedContent.Append(internalMessage.Content);
                    tokens += internalMessage.RawOutputTokens?.Count ?? 0;
                }
            }
            replacement.Add(generated);
            if (visibleEnd < history.Count)
            {
                emittedContent.Append(emitted.Content);
                emitted = emitted with { Content = emittedContent.ToString(), RawText = null };
            }

            var record = new TurnRecord { Replacement = replacement, Emitted = emitted, Scope = scope, Tokens = tokens };
            lock (_lock)
            {
                if (!_chains.TryGetValue(key, out ChainEntry entry))
                {
                    entry = new ChainEntry { Key = key };
                    entry.Node = _lru.AddLast(entry);
                    _chains[key] = entry;
                }
                else
                {
                    Touch(entry);
                }

                for (int r = entry.Records.Count - 1; r >= 0; r--)
                {
                    TurnRecord old = entry.Records[r];
                    if (string.Equals(old.Scope, scope, StringComparison.Ordinal)
                        || SameEmitted(old.Emitted, record.Emitted))
                    {
                        _tokens -= old.Tokens;
                        entry.Records.RemoveAt(r);
                    }
                }
                entry.Records.Insert(0, record);
                _tokens += record.Tokens;
                while (entry.Records.Count > MaxRecordsPerChain)
                {
                    _tokens -= entry.Records[^1].Tokens;
                    entry.Records.RemoveAt(entry.Records.Count - 1);
                }

                // Oldest conversation positions go first; the one just written stays.
                while ((_chains.Count > _maxChains || _tokens > _maxTokens) && _lru.First != entry.Node)
                {
                    ChainEntry victim = _lru.First.Value;
                    _lru.RemoveFirst();
                    _chains.Remove(victim.Key);
                    foreach (TurnRecord gone in victim.Records)
                        _tokens -= gone.Tokens;
                }

                var latest = new List<ChatMessage>();
                if (renderHistoryForDiagnostics != null)
                    latest.AddRange(renderHistoryForDiagnostics);
                latest.Add(generated);
                _latestTranscript = latest;
            }
        }

        private void Touch(ChainEntry entry)
        {
            _lru.Remove(entry.Node);
            _lru.AddLast(entry.Node);
        }

        /// <summary>Where the client-visible history ends: the first message carrying raw
        /// tokens starts the in-process tool loop's own rounds, which no client sent.</summary>
        private static int VisibleEnd(IReadOnlyList<ChatMessage> history)
        {
            for (int i = 0; i < history.Count; i++)
                if (history[i].RawOutputTokens is { Count: > 0 })
                    return i;
            return history.Count;
        }

        // ---- content-hash chain --------------------------------------------------

        private static byte[] Advance(byte[] chain, ChatMessage message)
        {
            using var sha = IncrementalHash.CreateHash(HashAlgorithmName.SHA256);
            sha.AppendData(chain);
            Append(sha, message.Role);
            Append(sha, message.Content);
            if (message.ToolCalls != null)
            {
                foreach (ToolCall call in message.ToolCalls)
                {
                    Append(sha, "tc");
                    Append(sha, call.Name);
                    Append(sha, CanonicalArguments(call.Arguments));
                }
            }
            AppendMedia(sha, "img", message.ImagePaths);
            AppendMedia(sha, "aud", message.AudioPaths);
            if (message.IsVideo)
                Append(sha, "video");
            if (message.ImageTimestamps != null)
                foreach (double? t in message.ImageTimestamps)
                    Append(sha, t?.ToString("R", System.Globalization.CultureInfo.InvariantCulture) ?? "-");
            return sha.GetHashAndReset();
        }

        private static void AppendMedia(IncrementalHash sha, string kind, List<string> paths)
        {
            if (paths == null) return;
            foreach (string path in paths)
            {
                Append(sha, kind);
                // By content: a resent base64 image is the same image whatever file it
                // was stored under this time.
                Append(sha, MediaContentId.OfFile(path) ?? "path:" + path);
            }
        }

        private static void Append(IncrementalHash sha, string value)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(value ?? string.Empty);
            Span<byte> length = stackalloc byte[4];
            System.Buffers.Binary.BinaryPrimitives.WriteInt32LittleEndian(length, value == null ? -1 : bytes.Length);
            sha.AppendData(length);
            sha.AppendData(bytes);
        }

        // ---- emitted-form comparison --------------------------------------------

        private static bool EmittedMatches(ChatMessage client, TurnRecord record)
        {
            EmittedAssistantTurn emitted = record.Emitted;
            if (!ToolCallsEqual(client.ToolCalls, emitted.ToolCalls))
                return false;
            string clientThinking = Squash(client.Thinking);
            string emittedThinking = Squash(emitted.Thinking);
            if (clientThinking.Length > 0 && emittedThinking.Length > 0
                && !string.Equals(clientThinking, emittedThinking, StringComparison.Ordinal))
                return false;

            string clientContent = Squash(client.Content);
            string emittedContent = Squash(emitted.Content);
            if (string.Equals(clientContent, emittedContent, StringComparison.Ordinal))
                return true;
            if (emitted.RawText != null
                && string.Equals(clientContent, Squash(emitted.RawText), StringComparison.Ordinal))
                return true;

            // A stopped turn: the client kept what streamed before the stop, which can be
            // a few characters short of what the parser flushes at the end.
            if (emitted.Cancelled && clientContent.Length > 0
                && emittedContent.StartsWith(clientContent, StringComparison.Ordinal)
                && emittedContent.Length - clientContent.Length <= 64
                && clientContent.Length * 2 >= emittedContent.Length)
                return true;

            // A tool-loop transcript: the loop may append its own note after the model's
            // last round ("\n\n" + a verified artifact link, a repetition notice). The
            // generated part must be all there; the host's note is not model output.
            if (record.Replacement.Count > 1 && emittedContent.Length > 0
                && clientContent.StartsWith(emittedContent, StringComparison.Ordinal))
                return HostNoteFollows(client.Content ?? string.Empty, emittedContent.Length);

            return false;
        }

        /// <summary>After the first <paramref name="nonWhitespace"/> non-whitespace
        /// characters of <paramref name="content"/>, the rest begins with a blank line.</summary>
        private static bool HostNoteFollows(string content, int nonWhitespace)
        {
            int seen = 0, i = 0;
            for (; i < content.Length && seen < nonWhitespace; i++)
                if (!char.IsWhiteSpace(content[i]))
                    seen++;
            int newlines = 0;
            for (; i < content.Length && char.IsWhiteSpace(content[i]); i++)
                if (content[i] == '\n')
                    newlines++;
            return newlines >= 2;
        }

        private static bool SameEmitted(EmittedAssistantTurn a, EmittedAssistantTurn b)
            => string.Equals(Squash(a.Content), Squash(b.Content), StringComparison.Ordinal)
                && ToolCallsEqual(a.ToolCalls, b.ToolCalls);

        /// <summary>Whitespace is not content: an adapter or client that trims, or
        /// re-flows line endings, still sent back what it was given.</summary>
        private static string Squash(string value)
        {
            if (string.IsNullOrEmpty(value))
                return string.Empty;
            var sb = new StringBuilder(value.Length);
            foreach (char c in value)
                if (!char.IsWhiteSpace(c))
                    sb.Append(c);
            return sb.ToString();
        }

        private static bool ToolCallsEqual(IReadOnlyList<ToolCall> a, IReadOnlyList<ToolCall> b)
        {
            int ca = a?.Count ?? 0, cb = b?.Count ?? 0;
            if (ca != cb) return false;
            for (int i = 0; i < ca; i++)
            {
                if (!string.Equals(a[i].Name, b[i].Name, StringComparison.Ordinal))
                    return false;
                if (!string.Equals(CanonicalArguments(a[i].Arguments), CanonicalArguments(b[i].Arguments), StringComparison.Ordinal))
                    return false;
            }
            return true;
        }

        /// <summary>Tool arguments as JSON with object keys sorted, so the same call
        /// compares equal whichever side serialized it.</summary>
        internal static string CanonicalArguments(Dictionary<string, object> arguments)
        {
            if (arguments == null || arguments.Count == 0)
                return "{}";
            try
            {
                using var doc = JsonDocument.Parse(JsonSerializer.Serialize(arguments));
                var sb = new StringBuilder();
                WriteCanonical(doc.RootElement, sb);
                return sb.ToString();
            }
            catch (Exception ex) when (ex is JsonException or NotSupportedException)
            {
                return JsonSerializer.Serialize(arguments);
            }
        }

        private static void WriteCanonical(JsonElement element, StringBuilder sb)
        {
            switch (element.ValueKind)
            {
                case JsonValueKind.Object:
                    var props = new List<JsonProperty>();
                    foreach (JsonProperty p in element.EnumerateObject())
                        props.Add(p);
                    props.Sort((x, y) => string.CompareOrdinal(x.Name, y.Name));
                    sb.Append('{');
                    for (int i = 0; i < props.Count; i++)
                    {
                        if (i > 0) sb.Append(',');
                        sb.Append(JsonSerializer.Serialize(props[i].Name)).Append(':');
                        WriteCanonical(props[i].Value, sb);
                    }
                    sb.Append('}');
                    break;
                case JsonValueKind.Array:
                    sb.Append('[');
                    int n = 0;
                    foreach (JsonElement item in element.EnumerateArray())
                    {
                        if (n++ > 0) sb.Append(',');
                        WriteCanonical(item, sb);
                    }
                    sb.Append(']');
                    break;
                case JsonValueKind.Number:
                    sb.Append(element.TryGetDouble(out double d)
                        ? d.ToString("R", System.Globalization.CultureInfo.InvariantCulture)
                        : element.GetRawText());
                    break;
                default:
                    sb.Append(element.GetRawText());
                    break;
            }
        }

        // ---- splice / expansion ---------------------------------------------------

        private static ChatMessage SpliceOnto(ChatMessage src, ChatMessage recorded) => new()
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
            RawOutputTokens = recorded.RawOutputTokens,
            RawPromptTrailingWhitespace = recorded.RawPromptTrailingWhitespace,
            RawGenerationSuffix = recorded.RawGenerationSuffix,
            CacheControl = src.CacheControl,
            ContentCacheBreakpoints = src.ContentCacheBreakpoints,
        };

        /// <summary>Map cache markers from a client-visible assistant message onto the
        /// recorded assistant/tool transcript that replaces it. A message-level marker
        /// belongs after the whole expansion. Content offsets are mapped over the
        /// intermediate rounds' content; an offset that reaches the final raw round marks
        /// its start conservatively, because parser-stripped thinking text makes the exact
        /// offset unknowable.</summary>
        private static void PreserveCollapsedCacheMarkers(ChatMessage source, List<ChatMessage> expanded)
        {
            if (expanded.Count == 0) return;

            int lastAssistant = expanded.FindLastIndex(m => m.Role == "assistant");
            if (lastAssistant < 0) return;

            if (source.CacheControl != null)
            {
                expanded[lastAssistant].CacheControl = new CacheControlMarker
                {
                    Type = source.CacheControl.Type,
                };
            }

            if (source.ContentCacheBreakpoints == null || source.ContentCacheBreakpoints.Count == 0)
                return;

            foreach (int rawOffset in source.ContentCacheBreakpoints)
            {
                int remaining = Math.Max(0, rawOffset);
                bool mapped = false;
                for (int i = 0; i < lastAssistant; i++)
                {
                    if (expanded[i].Role != "assistant") continue;
                    int length = expanded[i].Content?.Length ?? 0;
                    if (remaining <= length)
                    {
                        AddMappedContentCacheBreakpoint(expanded[i], remaining);
                        mapped = true;
                        break;
                    }
                    remaining -= length;
                }

                if (!mapped)
                    AddMappedContentCacheBreakpoint(expanded[lastAssistant], 0);
            }
        }

        private static void AddMappedContentCacheBreakpoint(ChatMessage message, int offset)
        {
            message.AddContentCacheBreakpoint(offset);
            message.ContentCacheBreakpoints.Sort();
        }

        internal static ChatMessage CloneShallow(ChatMessage src) => new()
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
