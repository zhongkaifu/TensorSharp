// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text.Json;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    internal static partial class ChatMessageParser
    {
        /// <summary>
        /// An OpenAI chat <c>video_url</c> part's object: <c>{"url": "data:video/...;base64,...", "fps"?, "max_frames"?}</c>.
        /// </summary>
        private static void AppendSampledVideo(ChatMessage message, JsonElement video,
            UploadStoragePolicy uploads)
        {
            if (video.ValueKind != JsonValueKind.Object || !video.TryGetProperty("url", out var urlElement) ||
                urlElement.ValueKind != JsonValueKind.String)
                throw new JsonException("video_url must contain a base64 video data URI in url.");
            AppendSampledVideo(message, urlElement.GetString(), video, uploads);
        }

        /// <summary>
        /// A Responses <c>input_video</c> part: <c>{"type": "input_video", "video_url": "data:video/...;base64,...",
        /// "fps"?, "max_frames"?}</c> - the shape of <c>input_image</c>, with <c>video_url</c>'s sampling options.
        /// </summary>
        private static void AppendResponsesVideo(ChatMessage message, JsonElement part, UploadStoragePolicy uploads)
        {
            if (!part.TryGetProperty("video_url", out var urlElement) || urlElement.ValueKind != JsonValueKind.String)
                throw new JsonException("input_video.video_url must contain a base64 video data URI.");
            AppendSampledVideo(message, urlElement.GetString(), part, uploads);
        }

        /// <summary>
        /// Save the clip of a data URI to the upload directory, sample its frames into <see cref="ChatMessage.ImagePaths"/>
        /// (timed, <see cref="ChatMessage.ImageTimestamps"/>) at <paramref name="options"/>' <c>fps</c> / <c>max_frames</c>,
        /// and name the saved clip in <see cref="ChatMessage.VideoFilePaths"/>, so a family that samples a video itself
        /// (LFM3-Omni: its own 2 fps frames and the soundtrack) reads the file instead of the frames.
        /// </summary>
        private static void AppendSampledVideo(ChatMessage message, string url, JsonElement options,
            UploadStoragePolicy uploads)
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
