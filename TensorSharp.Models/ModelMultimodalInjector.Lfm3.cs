// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

using TensorSharp.Models.Architecture;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    internal sealed partial class ModelMultimodalInjector
    {
        /// <summary>
        /// Encoded LFM3 videos by identity. A video's rows are cache entries of their own - one per frame pair in the
        /// video-frame cache, one per soundtrack piece in the audio cache - which the byte budget may evict separately; a
        /// record whose entries are not all still cached is encoded again.
        /// </summary>
        private readonly Dictionary<string, Lfm3VideoRecord> _lfm3Videos = new(StringComparer.Ordinal);

        /// <summary>A video's labels (one per frame pair), its pairs' cache keys and, per pair, the key of the soundtrack
        /// piece that follows it (null where none).</summary>
        private sealed record Lfm3VideoRecord(string[] Labels, string[] GroupKeys, string[] PieceKeys);

        /// <summary>
        /// LFM3 / LFM3-Omni: the chat template renders each media item of a message (<see cref="Lfm3MediaLayout"/>) as
        /// <c>&lt;|vision_start|&gt;</c> + a placeholder + <c>&lt;|vision_end|&gt;</c>, and only the placeholder is replaced,
        /// which gives llama.cpp's mtmd layout (<c>mtmd.cpp</c>: Omni wraps images, videos and audio in the vision markers)
        /// and HF's:
        /// <list type="bullet">
        /// <item><c>&lt;|image|&gt;</c> -> the image's rows;</item>
        /// <item><c>&lt;|audio|&gt;</c> -> the clip's rows (one per 80 ms);</item>
        /// <item><c>&lt;|video|&gt;</c> -> per frame pair <c>"&lt;t.t seconds&gt;"</c> + <c>&lt;|vision_start|&gt;</c> + the
        /// pair's rows + <c>&lt;|vision_end|&gt;</c>, then (Omni, a file with sound) that pair's soundtrack rows, bare.</item>
        /// </list>
        /// Rows take plain 1-D positions. Each row block is one prepared span with a content identity: an image or a clip by
        /// its bytes, a frame pair by its video's bytes (or its frames') and its index, a soundtrack piece by its video and
        /// rows. Media without the tower it needs is refused here, the last line behind the request gates.
        /// </summary>
        internal List<int> ProcessLfm3History(Lfm3RuntimeModel model, List<ChatMessage> history, List<int> inputTokens)
        {
            var items = new List<(ChatMessage Message, Lfm3MediaItem Item)>();
            if (history != null)
                foreach (ChatMessage message in history)
                    foreach (Lfm3MediaItem item in Lfm3MediaLayout.Items(message))
                        items.Add((message, item));
            if (items.Count == 0)
                return PlaceLfm3SpeechRows(model, history, inputTokens);

            if (model.VisionEncoder == null && items.Exists(i => i.Item.Kind != Lfm3MediaKind.Audio))
                throw new NotSupportedException(Lfm3RuntimeModel.VisionInputUnsupportedMessage);
            if (model.AudioEncoder == null && items.Exists(i => i.Item.Kind == Lfm3MediaKind.Audio))
                throw new NotSupportedException(AudioInputSupport.Lfm3Message);

            int[] placeholders = FindLfm3Placeholders(inputTokens, items);
            var output = new List<int>(inputTokens.Count + 1024);
            int copied = 0;
            for (int k = 0; k < items.Count; k++)
            {
                int at = placeholders[k];
                for (int i = copied; i < at; i++)
                    output.Add(inputTokens[i]);
                copied = at + 1;
                (ChatMessage message, Lfm3MediaItem item) = items[k];
                switch (item.Kind)
                {
                    case Lfm3MediaKind.Image:
                    {
                        string path = message.ImagePaths[item.Index];
                        WarmContentId(model, path);
                        CachedEmbedding image = GetOrCreateCachedEmbedding(_visionCache, path,
                            fullPath => CreateCachedEmbedding(fullPath, model.EncodeImageFile(fullPath)));
                        AppendLfm3Rows(output, image, Lfm3VisionEncoder.ImageToken, audio: false, marked: true);
                        break;
                    }
                    case Lfm3MediaKind.Audio:
                    {
                        string path = message.AudioPaths[item.Index];
                        WarmContentId(model, path);
                        CachedEmbedding clip = GetOrCreateCachedEmbedding(_audioCache, path,
                            fullPath => CreateCachedEmbedding(fullPath, model.EncodeAudioFile(fullPath)));
                        AppendLfm3Rows(output, clip, Lfm3AudioToken, audio: true, marked: true);
                        break;
                    }
                    default:
                        AppendLfm3Video(model, message, item, output);
                        break;
                }
            }
            for (int i = copied; i < inputTokens.Count; i++)
                output.Add(inputTokens[i]);
            return PlaceLfm3SpeechRows(model, history, output);
        }

        /// <summary>
        /// The speech of past spoken turns (<see cref="ChatMessage.SpeechFrames"/>): their spliced output holds one
        /// <c>&lt;|audio_out|&gt;</c> per frame the reply decoded, in order, and each gets that frame's feedback row
        /// (<see cref="Lfm3SpeechOutput.FeedbackRow(ReadOnlySpan{int})"/>, the row generation decoded), so the turn re-renders
        /// as what generation produced (HF's processor gives a past speech turn's placeholders the same codebook sums). Each
        /// run of placeholders is one span whose content identity is its codes, for prompt reuse. A prompt whose placeholders
        /// and frames disagree is refused rather than given rows of the wrong frames.
        /// </summary>
        private List<int> PlaceLfm3SpeechRows(Lfm3RuntimeModel model, List<ChatMessage> history, List<int> tokens)
        {
            var frames = new List<int[]>();
            if (history != null)
                foreach (ChatMessage message in history)
                    if (message?.SpeechFrames is { Count: > 0 } spoken)
                        frames.AddRange(spoken);
            if (frames.Count == 0)
                return tokens;
            int audioOut = model.SpeechTokens?.AudioOut ?? -1;
            int placeholders = 0;
            if (audioOut >= 0)
                foreach (int t in tokens)
                    if (t == audioOut) placeholders++;
            Lfm3SpeechOutput speech = model.SpeechOutput ?? throw new NotSupportedException(
                "lfm3: the history holds a spoken reply, whose speech re-renders through the speech decoder of the mmproj; " +
                "load the LFM3-Omni mmproj with speech output, or send the reply's transcript as its content");
            if (placeholders != frames.Count)
                throw new InvalidOperationException(
                    $"lfm3: the prompt holds {placeholders} <|audio_out|> placeholder(s) for {frames.Count} speech frame(s) of " +
                    "past spoken replies; the rendered prompt and the replies disagree.");
            int next = 0;
            for (int i = 0; i < tokens.Count;)
            {
                if (tokens[i] != audioOut) { i++; continue; }
                int start = i;
                while (i < tokens.Count && tokens[i] == audioOut) i++;
                CachedEmbedding rows = GetOrCreateLfm3SpeechRows(model, speech, frames, next, i - start);
                _preparedAudioEmbeddings.Add(new PreparedEmbeddingSpan(rows, start, start, i));
                next += i - start;
            }
            return tokens;
        }

        /// <summary>Feedback rows of frames [<paramref name="first"/>, + <paramref name="count"/>) as a cached embedding keyed by
        /// their codes.</summary>
        private CachedEmbedding GetOrCreateLfm3SpeechRows(Lfm3RuntimeModel model, Lfm3SpeechOutput speech, List<int[]> frames,
            int first, int count)
        {
            var codes = new List<byte>(count * Lfm3SpeechConfig.Codebooks * 4);
            for (int f = first; f < first + count; f++)
                foreach (int c in frames[f])
                    codes.AddRange(BitConverter.GetBytes(c));
            string key = "lfm3-speech:" + Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(codes.ToArray()));
            if (_audioCache.TryGetValue(key, out CachedEmbedding cached))
            {
                TouchAndPin(cached);
                return cached;
            }
            int width = speech.Config.HiddenSize;
            var values = new float[(long)count * width];
            for (int f = 0; f < count; f++)
                speech.FeedbackRow(frames[first + f], values.AsSpan(f * width, width));
            var t = new Tensor(model.MediaAllocator, DType.Float32, count, width);
            t.SetElementsAsFloat(values);
            return StoreLfm3Entry(_audioCache, key, key, t);
        }

        /// <summary><c>&lt;|audio|&gt;</c>: the template's placeholder of an audio part, and the slot token of audio rows.</summary>
        internal const int Lfm3AudioToken = 125031;

        /// <summary>
        /// Where each item's placeholder sits: the template's spans, in the order the layout lists the items, each inside
        /// its markers. A prompt that disagrees with the attachments (a placeholder of another kind, a count that differs,
        /// a placeholder typed into the text) is refused rather than injected into the wrong tokens.
        /// </summary>
        private static int[] FindLfm3Placeholders(List<int> tokens, List<(ChatMessage Message, Lfm3MediaItem Item)> items)
        {
            var found = new List<int>(items.Count);
            for (int i = 0; i < tokens.Count; i++)
                if (tokens[i] is Lfm3VisionEncoder.ImageToken or Lfm3VisionEncoder.VideoToken or Lfm3AudioToken)
                    found.Add(i);
            if (found.Count != items.Count)
                throw new InvalidOperationException(
                    $"lfm3: the prompt renders {found.Count} media placeholder(s) for {items.Count} attachment(s) " +
                    $"({Describe(items)}); the rendered prompt and the message attachments disagree.");
            for (int k = 0; k < found.Count; k++)
            {
                int at = found[k];
                int expected = items[k].Item.Kind switch
                {
                    Lfm3MediaKind.Image => Lfm3VisionEncoder.ImageToken,
                    Lfm3MediaKind.Video => Lfm3VisionEncoder.VideoToken,
                    _ => Lfm3AudioToken,
                };
                if (tokens[at] != expected)
                    throw new InvalidOperationException(
                        $"lfm3: media placeholder {k} is token {tokens[at]} where attachment {k} ({items[k].Item.Kind}) needs {expected}.");
                if (at == 0 || tokens[at - 1] != Lfm3VisionEncoder.VisionStartToken
                    || at + 1 >= tokens.Count || tokens[at + 1] != Lfm3VisionEncoder.VisionEndToken)
                    throw new InvalidOperationException(
                        $"lfm3: media placeholder {k} ({items[k].Item.Kind}) is not inside <|vision_start|> ... <|vision_end|>, " +
                        "the span the chat template renders around it.");
            }
            return found.ToArray();
        }

        private static string Describe(List<(ChatMessage Message, Lfm3MediaItem Item)> items)
        {
            var sb = new StringBuilder();
            foreach (var (_, item) in items)
                sb.Append(sb.Length > 0 ? ", " : "").Append(item.Kind.ToString().ToLowerInvariant());
            return sb.ToString();
        }

        /// <summary>Hash the file before the cache looks it up, off the GPU lock (the hash is memoized by path, size and
        /// write time, so the lookup under the lock finds it).</summary>
        private static void WarmContentId(ModelBase model, string path)
            => model.RunOutsideGpuComputeLock(() => MediaContentId.OfFile(path));

        /// <summary>Lay <paramref name="rows"/>' slots at the end of <paramref name="output"/> and prepare its span. A
        /// <paramref name="marked"/> block sits between <c>&lt;|vision_start|&gt;</c> (already written) and
        /// <c>&lt;|vision_end|&gt;</c> (written next), and its prompt extent covers both, so a trim or a reuse boundary
        /// never keeps one marker without its rows; a bare block (a soundtrack piece) covers its rows.</summary>
        private void AppendLfm3Rows(List<int> output, CachedEmbedding rows, int slotToken, bool audio, bool marked)
        {
            int insert = output.Count;
            for (int r = 0; r < rows.TokenCount; r++)
                output.Add(slotToken);
            var span = marked
                ? new PreparedEmbeddingSpan(rows, insert, insert - 1, insert + rows.TokenCount + 1)
                : new PreparedEmbeddingSpan(rows, insert, insert, insert + rows.TokenCount);
            (audio ? _preparedAudioEmbeddings : _preparedVisionEmbeddings).Add(span);
        }

        /// <summary>One video's inner layout in place of its <c>&lt;|video|&gt;</c>: per pair the label, the marked rows, then
        /// the soundtrack piece that follows it.</summary>
        private void AppendLfm3Video(Lfm3RuntimeModel model, ChatMessage message, Lfm3MediaItem item, List<int> output)
        {
            (string[] labels, CachedEmbedding[] groups, CachedEmbedding[] pieces) = item.VideoFile != null
                ? GetOrCreateLfm3FileVideo(model, item.VideoFile)
                : GetOrCreateLfm3FrameVideo(model, message, item);
            for (int g = 0; g < groups.Length; g++)
            {
                output.AddRange(_model.Tokenizer.Encode(labels[g], addSpecial: false));
                output.Add(Lfm3VisionEncoder.VisionStartToken);
                AppendLfm3Rows(output, groups[g], Lfm3VisionEncoder.ImageToken, audio: false, marked: true);
                output.Add(Lfm3VisionEncoder.VisionEndToken);
                if (pieces[g] != null)
                    AppendLfm3Rows(output, pieces[g], Lfm3AudioToken, audio: true, marked: false);
            }
        }

        /// <summary>A video file's pairs and soundtrack pieces, cached by the file's bytes and the frame cap (the sampling
        /// follows from them) and whether the soundtrack is heard.</summary>
        private (string[] Labels, CachedEmbedding[] Groups, CachedEmbedding[] Pieces) GetOrCreateLfm3FileVideo(
            Lfm3RuntimeModel model, string videoPath)
        {
            string fullPath = NormalizePath(videoPath);
            string hash = model.RunOutsideGpuComputeLock(() => MediaContentId.OfFile(fullPath))
                ?? throw new System.IO.FileNotFoundException($"lfm3 video: cannot read '{videoPath}'", videoPath);
            string id = string.Create(CultureInfo.InvariantCulture,
                $"lfm3v:{hash}:{model.VisionEncoder.MaxVideoFrames}:{(model.AudioEncoder != null ? "a" : "-")}");
            if (TryGetLfm3Video(id, out var cached))
                return cached;

            Lfm3EncodedVideo encoded = model.EncodeVideoFile(fullPath);
            Lfm3VideoEncoding video = encoded.Video;
            int n = video.Groups.Count;
            var pieceRows = new (int Start, int End)[n];
            foreach (Lfm3AudioPrompt.SoundtrackPiece piece in encoded.Pieces)
                if (piece.Rows > 0 && piece.Group < n)
                    pieceRows[piece.Group] = (piece.RowStart, piece.RowEnd);
            return StoreLfm3Video(model.MediaAllocator, id, fullPath, video, g => pieceRows[g].End > pieceRows[g].Start
                ? SliceRows(model.MediaAllocator, encoded.SoundtrackRows, encoded.RowWidth, pieceRows[g].Start, pieceRows[g].End)
                : null);
        }

        /// <summary>A video of host-sampled frames, cached by the frames' bytes and times.</summary>
        private (string[] Labels, CachedEmbedding[] Groups, CachedEmbedding[] Pieces) GetOrCreateLfm3FrameVideo(
            Lfm3RuntimeModel model, ChatMessage message, Lfm3MediaItem item)
        {
            IReadOnlyList<int> frames = item.Frames;
            var paths = new string[frames.Count];
            for (int i = 0; i < paths.Length; i++)
                paths[i] = NormalizePath(message.ImagePaths[frames[i]]);
            IReadOnlyList<double> seconds = item.FrameSeconds ?? UntimedFrameSeconds(paths.Length);
            string id = model.RunOutsideGpuComputeLock(() =>
            {
                var sb = new StringBuilder("lfm3vf");
                for (int i = 0; i < paths.Length; i++)
                    sb.Append(':').Append(MediaContentId.OfFile(paths[i]) ?? "path:" + paths[i])
                      .Append('@').Append(seconds[i].ToString("R", CultureInfo.InvariantCulture));
                return "lfm3vf:" + MediaContentId.OfBytes(Encoding.UTF8.GetBytes(sb.ToString()));
            });
            if (TryGetLfm3Video(id, out var cached))
                return cached;

            Lfm3VideoEncoding video = model.EncodeVideoFrames(paths, seconds);
            return StoreLfm3Video(model.MediaAllocator, id, paths[0], video, _ => null);
        }

        /// <summary>
        /// Times for frames a host sampled without recording them (the CLI's and the Web UI's extraction at
        /// <c>VIDEO_SAMPLE_FPS</c>, 1 fps by default): frame i at i / fps. Approximate - the extraction steps a whole number
        /// of source frames - and only the pairing and the labels depend on it; a video file gives the exact ones.
        /// </summary>
        private static double[] UntimedFrameSeconds(int count)
        {
            double fps = MediaHelper.GetConfiguredVideoSampleFps();
            var seconds = new double[count];
            for (int i = 0; i < count; i++)
                seconds[i] = i / fps;
            return seconds;
        }

        /// <summary>A cached video whose every entry is still in the caches (each touched and pinned for this
        /// preparation), or false.</summary>
        private bool TryGetLfm3Video(string id, out (string[] Labels, CachedEmbedding[] Groups, CachedEmbedding[] Pieces) video)
        {
            video = default;
            if (!_lfm3Videos.TryGetValue(id, out Lfm3VideoRecord record))
                return false;
            var groups = new CachedEmbedding[record.GroupKeys.Length];
            var pieces = new CachedEmbedding[record.GroupKeys.Length];
            for (int g = 0; g < groups.Length; g++)
            {
                if (!_videoFrameCache.TryGetValue(record.GroupKeys[g], out groups[g]))
                {
                    _lfm3Videos.Remove(id);
                    return false;
                }
                if (record.PieceKeys[g] != null && !_audioCache.TryGetValue(record.PieceKeys[g], out pieces[g]))
                {
                    _lfm3Videos.Remove(id);
                    return false;
                }
            }
            for (int g = 0; g < groups.Length; g++)
            {
                TouchAndPin(groups[g]);
                if (pieces[g] != null) TouchAndPin(pieces[g]);
            }
            video = (record.Labels, groups, pieces);
            return true;
        }

        /// <summary>Cache a freshly encoded video under <paramref name="id"/>: a pair's rows under <c>id:g</c>, the piece of
        /// soundtrack after pair g (<paramref name="pieceAfter"/>, null for none) under <c>id:s:g</c>. A video another
        /// preparation cached while this one encoded (the encoder yields the GPU lock) wins, as for single media.</summary>
        private (string[] Labels, CachedEmbedding[] Groups, CachedEmbedding[] Pieces) StoreLfm3Video(
            IAllocator allocator, string id, string fullPath, Lfm3VideoEncoding video, Func<int, Tensor> pieceAfter)
        {
            int n = video.Groups.Count;
            var labels = new string[n];
            var groups = new CachedEmbedding[n];
            var pieces = new CachedEmbedding[n];
            var tensors = new Tensor[n];
            var pieceTensors = new Tensor[n];
            try
            {
                for (int g = 0; g < n; g++)
                {
                    labels[g] = video.Plan.Groups[g].Label;
                    tensors[g] = video.Groups[g].ToTensor(allocator);
                    pieceTensors[g] = pieceAfter(g);
                }
            }
            catch
            {
                foreach (Tensor t in tensors) t?.Dispose();
                foreach (Tensor t in pieceTensors) t?.Dispose();
                throw;
            }

            if (TryGetLfm3Video(id, out var raced))
            {
                foreach (Tensor t in tensors) t?.Dispose();
                foreach (Tensor t in pieceTensors) t?.Dispose();
                return raced;
            }

            var groupKeys = new string[n];
            var pieceKeys = new string[n];
            for (int g = 0; g < n; g++)
            {
                groupKeys[g] = id + ":" + g.ToString(CultureInfo.InvariantCulture);
                groups[g] = StoreLfm3Entry(_videoFrameCache, groupKeys[g], fullPath, tensors[g]);
                if (pieceTensors[g] != null)
                {
                    pieceKeys[g] = id + ":s:" + g.ToString(CultureInfo.InvariantCulture);
                    pieces[g] = StoreLfm3Entry(_audioCache, pieceKeys[g], fullPath, pieceTensors[g]);
                }
            }
            _lfm3Videos[id] = new Lfm3VideoRecord(labels, groupKeys, pieceKeys);
            return (labels, groups, pieces);
        }

        private CachedEmbedding StoreLfm3Entry(Dictionary<string, CachedEmbedding> cache, string key, string fullPath, Tensor rows)
        {
            CachedEmbedding entry = CreateCachedEmbedding(fullPath, rows);
            entry.ContentId = key;
            AddCachedEmbedding(cache, key, entry);
            return entry;
        }

        /// <summary>Rows [<paramref name="start"/>, <paramref name="end"/>) of a row-major [rows][<paramref name="width"/>]
        /// array as a tensor on the model's allocator.</summary>
        private static Tensor SliceRows(IAllocator allocator, float[] rows, int width, int start, int end)
        {
            var slice = new float[(long)(end - start) * width];
            Array.Copy(rows, (long)start * width, slice, 0, slice.LongLength);
            var t = new Tensor(allocator, DType.Float32, end - start, width);
            t.SetElementsAsFloat(slice);
            return t;
        }
    }
}
