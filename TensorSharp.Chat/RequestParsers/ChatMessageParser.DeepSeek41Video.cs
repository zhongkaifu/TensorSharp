// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TensorSharp.Models.Media;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    internal static partial class ChatMessageParser
    {
        /// <summary>
        /// An OpenAI chat <c>video_url</c> part's object: <c>{"url": "data:video/...;base64,...", "fps"?, "max_frames"?}</c>.
        /// </summary>
        private static void AppendSampledVideo(ChatMessage message, JsonElement video,
            UploadStoragePolicy uploads, bool sampleFrames)
        {
            if (video.ValueKind != JsonValueKind.Object || !video.TryGetProperty("url", out var urlElement) ||
                urlElement.ValueKind != JsonValueKind.String)
                throw new JsonException("video_url must contain a base64 video data URI in url.");
            AppendSampledVideo(message, urlElement.GetString(), video, uploads, sampleFrames);
        }

        /// <summary>
        /// A Responses <c>input_video</c> part: <c>{"type": "input_video", "input_video": {"url": "data:video/...;base64,...",
        /// "fps"?, "max_frames"?}}</c> - the embeddings API's <c>input_video</c> (and llama.cpp's), with <c>video_url</c>'s
        /// sampling options.
        /// </summary>
        private static void AppendResponsesVideo(ChatMessage message, JsonElement part, UploadStoragePolicy uploads,
            bool sampleFrames)
        {
            if (!part.TryGetProperty("input_video", out var video) || video.ValueKind != JsonValueKind.Object
                || !video.TryGetProperty("url", out var urlElement) || urlElement.ValueKind != JsonValueKind.String)
                throw new JsonException("input_video must contain a base64 video data URI in input_video.url.");
            AppendSampledVideo(message, urlElement.GetString(), video, uploads, sampleFrames);
        }

        /// <summary>
        /// Save the clip of a data URI to the upload directory and name it in <see cref="ChatMessage.VideoFilePaths"/>; with
        /// <paramref name="sampleFrames"/>, also sample its frames into <see cref="ChatMessage.ImagePaths"/> (timed,
        /// <see cref="ChatMessage.ImageTimestamps"/>) at <paramref name="options"/>' <c>fps</c> / <c>max_frames</c>. A
        /// family that reads the clip itself (<see cref="ChatProtocol.ReadsVideoFiles"/>: LFM3-Omni's own 2 fps frames and
        /// the soundtrack) takes no host frames: the clip is only probed, so an unreadable one is still refused here.
        /// </summary>
        private static void AppendSampledVideo(ChatMessage message, string url, JsonElement options,
            UploadStoragePolicy uploads, bool sampleFrames)
        {
            int comma = url?.IndexOf(',') ?? -1;
            if (comma <= 0 || !url.StartsWith("data:video/", StringComparison.Ordinal) ||
                !url.AsSpan(0, comma).EndsWith(";base64", StringComparison.Ordinal))
                throw new JsonException("video_url supports base64 video data URIs; remote URLs are not fetched.");
            string extension = url.Substring(5, comma - 5 - ";base64".Length) switch
            {
                "video/mp4" => ".mp4",
                "video/webm" => ".webm",
                "video/quicktime" => ".mov",
                _ => throw new JsonException("video_url supports video/mp4, video/webm and video/quicktime data URIs.")
            };
            double fps = MediaHelper.GetConfiguredVideoSampleFps();
            int maxFrames = MediaHelper.GetConfiguredMaxVideoFrames();
            if (maxFrames <= 0) maxFrames = 16;
            if (options.TryGetProperty("fps", out var fpsElement) &&
                (fpsElement.ValueKind != JsonValueKind.Number || !fpsElement.TryGetDouble(out fps)))
                throw new JsonException("video_url.fps must be a number.");
            if (options.TryGetProperty("max_frames", out var countElement) &&
                (countElement.ValueKind != JsonValueKind.Number || !countElement.TryGetInt32(out maxFrames)))
                throw new JsonException("video_url.max_frames must be an integer.");
            if (!double.IsFinite(fps) || fps <= 0 || fps > 60 || maxFrames <= 0 || maxFrames > 64)
                throw new JsonException("video_url requires fps in (0, 60] and max_frames in [1, 64].");

            byte[] data;
            try { data = Convert.FromBase64String(url.Substring(comma + 1)); }
            catch (FormatException ex) { throw new JsonException("video_url contains invalid base64.", ex); }
            if (data.Length == 0)
                throw new JsonException("video_url contains an empty video.");
            uploads.ReserveClientWriteOrThrow(data.Length);
            string prefix = Guid.NewGuid().ToString("N");
            string path = Path.Combine(uploads.DirectoryPath, prefix + extension);
            try
            {
                Directory.CreateDirectory(uploads.DirectoryPath);
                File.WriteAllBytes(path, data);
                if (!sampleFrames)
                {
                    VideoInfo info = MediaCodecs.Video.Probe(path);
                    if (info.FrameCount < 1 || !(info.Fps > 0))
                        throw new InvalidDataException($"the clip reports {info.FrameCount} frames at {info.Fps} fps.");
                    message.IsVideo = true;
                    (message.VideoFilePaths ??= new List<string>()).Add(path);
                    // The message is now a video one: its stills, before and after this part, keep an explicit
                    // no-time entry so no reader takes them for frames.
                    message.ImageTimestamps ??= new List<double?>(new double?[message.ImagePaths.Count]);
                    return;
                }
                var (frames, timestamps) = MediaHelper.ExtractVideoFramesWithTimestamps(
                    path, uploads.DirectoryPath, prefix, maxFrames, fps);
                if (frames.Count == 0 || frames.Count != timestamps.Count)
                    throw new InvalidDataException("No video frames were decoded.");
                uploads.RecordFiles(frames);
                message.ImageTimestamps ??= new List<double?>(new double?[message.ImagePaths.Count]);
                message.ImagePaths.AddRange(frames);
                foreach (double time in timestamps) message.ImageTimestamps.Add(time);
                message.IsVideo = true;
                (message.VideoFilePaths ??= new List<string>()).Add(path);
            }
            catch (Exception ex)
            {
                foreach (string file in Directory.Exists(uploads.DirectoryPath)
                    ? Directory.EnumerateFiles(uploads.DirectoryPath, prefix + "*") : Array.Empty<string>())
                {
                    try { File.Delete(file); }
                    catch (Exception cleanup) when (cleanup is IOException or UnauthorizedAccessException) { }
                }
                uploads.Release(data.Length);
                throw new JsonException("Could not read video_url: " + ex.Message, ex);
            }
        }
    }
}
