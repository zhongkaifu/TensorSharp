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
using System.Buffers.Binary;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Media.Desktop
{
    /// <summary>
    /// Audio decode on desktop hosts through ffmpeg (on PATH, or <c>TS_FFMPEG</c>): the formats the managed WAV / MP3 / Ogg
    /// readers do not cover (.flac, .m4a / .aac, .opus / .webm, .aiff, ...) and the soundtrack of a video (.mp4, .mov, .mkv,
    /// .webm: its first audio stream), which is what LFM3-Omni hears beside a video's frames. As the
    /// <see cref="IAudioDecoder"/> contract requires, the PCM keeps the stream's own rate and channels (ffmpeg converts the
    /// sample format to F32 only); each model resamples with its own code.
    /// </summary>
    public sealed partial class DesktopMediaProvider : IAudioDecoder
    {
        /// <summary>The longest decode ffmpeg gets (an hour of audio decodes in seconds).</summary>
        internal static readonly TimeSpan AudioDecodeTimeout = TimeSpan.FromMinutes(5);

        /// <summary>
        /// The demuxers ffmpeg may open a request's file with: audio files (WAV and Wave64, MP3, Ogg, FLAC, raw AAC, AIFF,
        /// Core Audio, WavPack, Sun AU) and the video containers whose soundtrack a model hears (QuickTime / MP4 / M4A,
        /// Matroska / WebM, AVI, MPEG-TS). No playlist or script demuxer (hls, concat, ffconcat, dash, image sequences):
        /// those open further files or URLs named inside the upload. Read with <c>-protocol_whitelist file,pipe</c>, so
        /// nothing but the file itself is opened in any case (TensorSharp.Studio reads uploads the same way).
        /// </summary>
        internal const string AudioFormats = "wav,w64,mp3,ogg,flac,aac,aiff,caf,wv,au,mov,matroska,avi,mpegts";

        public DecodedAudio Decode(string path) => Decode(path, AudioDecodeTimeout);

        /// <summary><see cref="Decode(string)"/> with ffmpeg given at most <paramref name="timeout"/>.</summary>
        internal DecodedAudio Decode(string path, TimeSpan timeout)
        {
            if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
            if (!File.Exists(path)) throw new FileNotFoundException($"audio file not found: {path}", path);
            string ffmpeg = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG") ?? throw new NotSupportedException(
                $"audio format '{Path.GetExtension(path)}' needs ffmpeg on this host (the managed decoders read .wav, .mp3 and .ogg): " +
                "install ffmpeg (https://ffmpeg.org) on PATH or set TS_FFMPEG to the ffmpeg executable.");
            var start = new ProcessStartInfo(ffmpeg)
            {
                UseShellExecute = false,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true,
            };
            // The first audio stream as F32 WAV on stdout; no metadata chunks, the stream's rate and channel count. The input
            // is the file alone, through the demuxers of AudioFormats.
            foreach (string a in new[] { "-nostdin", "-hide_banner", "-loglevel", "error",
                         "-protocol_whitelist", "file,pipe", "-format_whitelist", AudioFormats, "-i", path, "-map", "0:a:0", "-vn",
                         "-map_metadata", "-1", "-fflags", "+bitexact", "-flags:a", "+bitexact", "-c:a", "pcm_f32le", "-f", "wav", "pipe:1" })
                start.ArgumentList.Add(a);
            using var process = new Process { StartInfo = start };
            try { process.Start(); }
            catch (Exception ex) when (ex is System.ComponentModel.Win32Exception or InvalidOperationException)
            {
                throw new NotSupportedException($"could not start ffmpeg ('{ffmpeg}') to decode '{Path.GetFileName(path)}': {ex.Message}", ex);
            }
            using var deadline = new CancellationTokenSource(timeout);
            Task<string> errors = process.StandardError.ReadToEndAsync(deadline.Token);
            var wav = new MemoryStream();
            try
            {
                process.StandardOutput.BaseStream.CopyToAsync(wav, deadline.Token).GetAwaiter().GetResult();
                process.WaitForExitAsync(deadline.Token).GetAwaiter().GetResult();
            }
            catch (OperationCanceledException ex)
            {
                try { process.Kill(entireProcessTree: true); } catch (InvalidOperationException) { }
                throw new MediaInputException(
                    $"'{Path.GetFileName(path)}' could not be decoded: ffmpeg did not finish within {timeout.TotalSeconds:0.###} s",
                    new TimeoutException(ex.Message, ex));
            }
            // ffmpeg names its input by the path it was given: the reply names the file only.
            string error = SummarizeFfmpegError(MediaInputException.Scrub(errors.GetAwaiter().GetResult(), new[] { path }));
            if (process.ExitCode != 0)
            {
                if (error.Contains("matches no streams", StringComparison.Ordinal))
                    throw new MediaInputException($"'{Path.GetFileName(path)}' has no audio stream");
                if (error.Contains("not on whitelist", StringComparison.Ordinal))
                    throw new MediaInputException($"'{Path.GetFileName(path)}' is not an audio or video file this host decodes " +
                        $"(ffmpeg reads it only as one of: {AudioFormats}): {error}");
                throw new MediaInputException($"ffmpeg could not decode '{Path.GetFileName(path)}' (exit {process.ExitCode}): {error}");
            }
            return ParseStreamedWav(wav.GetBuffer().AsSpan(0, (int)wav.Length), Path.GetFileName(path));
        }

        /// <summary>
        /// ffmpeg's error log as a reason a person can read: the "[demuxer @ 0x...]" tags dropped, and the lines that only say
        /// its threads stopped (every failure ends with several) left out; at most 400 characters.
        /// </summary>
        internal static string SummarizeFfmpegError(string log)
        {
            var kept = new System.Collections.Generic.List<string>();
            foreach (string raw in (log ?? string.Empty).Split('\n'))
            {
                string line = System.Text.RegularExpressions.Regex.Replace(raw.Trim(), @"^(\[[^\]]*@ 0x[0-9a-fA-F]+\]\s*)+", "");
                if (line.Length == 0 || line.StartsWith("Last message repeated", StringComparison.Ordinal)
                    || line.Contains("Terminating thread with return code", StringComparison.Ordinal)
                    || line.Contains("Task finished with error code", StringComparison.Ordinal)
                    || line.Contains("Could not open encoder before EOF", StringComparison.Ordinal)
                    || line.StartsWith("Nothing was written into output file", StringComparison.Ordinal)
                    || kept.Contains(line))
                    continue;
                kept.Add(line);
            }
            string text = kept.Count > 0 ? string.Join("; ", kept) : (log ?? string.Empty).Trim();
            return text.Length <= 400 ? text : text[..397] + "...";
        }

        /// <summary>
        /// Planar PCM from the F32 WAV ffmpeg streams to a pipe: it cannot seek back to fill in the RIFF and data sizes, so
        /// they may read 0 or 0xFFFFFFFF; the data chunk then runs to the end of the stream.
        /// </summary>
        internal static DecodedAudio ParseStreamedWav(ReadOnlySpan<byte> wav, string name)
        {
            if (wav.Length < 12 || !wav[..4].SequenceEqual("RIFF"u8) || !wav.Slice(8, 4).SequenceEqual("WAVE"u8))
                throw new InvalidDataException($"ffmpeg returned no WAV stream for '{name}'");
            int at = 12, channels = 0, rate = 0, format = 0, bits = 0;
            while (at + 8 <= wav.Length)
            {
                ReadOnlySpan<byte> id = wav.Slice(at, 4);
                uint size = BinaryPrimitives.ReadUInt32LittleEndian(wav.Slice(at + 4, 4));
                int body = at + 8;
                if (id.SequenceEqual("fmt "u8))
                {
                    if (size < 16 || body + 16 > wav.Length) throw new InvalidDataException($"a short fmt chunk in ffmpeg's WAV of '{name}'");
                    format = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body, 2));
                    channels = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body + 2, 2));
                    rate = BinaryPrimitives.ReadInt32LittleEndian(wav.Slice(body + 4, 4));
                    bits = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body + 14, 2));
                    if (format == 0xFFFE && size >= 40) format = BinaryPrimitives.ReadUInt16LittleEndian(wav.Slice(body + 24, 2));
                }
                else if (id.SequenceEqual("data"u8))
                {
                    if (format != 3 || bits != 32 || channels < 1 || rate < 1)
                        throw new InvalidDataException($"ffmpeg's WAV of '{name}' is format {format}, {bits} bits, {channels} channels at {rate} Hz, not F32");
                    long available = wav.Length - body;
                    long bytes = size == 0 || size == uint.MaxValue || size > available ? available : size;
                    int frames = (int)(bytes / (4L * channels));
                    if (frames == 0) throw new MediaInputException($"'{name}' decodes to no audio samples");
                    var planes = new float[channels][];
                    for (int c = 0; c < channels; c++) planes[c] = new float[frames];
                    for (int i = 0; i < frames; i++)
                        for (int c = 0; c < channels; c++)
                            planes[c][i] = BinaryPrimitives.ReadSingleLittleEndian(wav.Slice(body + 4 * (i * channels + c), 4));
                    return new DecodedAudio { Channels = planes, SampleRate = rate };
                }
                if (size == uint.MaxValue) break;
                at = body + (int)size + (int)(size & 1);
            }
            throw new InvalidDataException($"ffmpeg's WAV of '{name}' has no data chunk");
        }
    }
}
