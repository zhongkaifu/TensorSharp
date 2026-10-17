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
using System.IO;
using System.Text.Json;
using System.Collections.Generic;
using TensorSharp.Models.Video;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.RequestParsers
{
    /// <summary>
    /// Parses the parameters shared by all video-generation endpoints, for any video
    /// model. Startup <c>--video-frames</c>/<c>--fps</c> values seed the request;
    /// explicitly supplied JSON fields then override those defaults.
    /// <para>Fields added for joint audio-video and reference-conditioned models are
    /// accepted in both camelCase and snake_case, since the OpenAI-shaped route uses
    /// snake_case while the web UI uses camelCase.</para>
    /// </summary>
    internal static class VideoGenerationParamsParser
    {
        /// <summary>The host defaults shared by chat and Studio, before explicit request overrides.</summary>
        internal static VideoGenerationParams CreateDefaults(ServerHostingOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            return new VideoGenerationParams
            {
                Frames = options.DefaultVideoFrames,
                Fps = options.DefaultVideoFps,
                Width = options.DefaultVideoWidth,
                Height = options.DefaultVideoHeight,
                CanvasPolicy = options.VideoCanvasPolicy,
                Steps = options.DefaultVideoSteps,
                Mode = options.DefaultVideoMode,
            };
        }

        public static VideoGenerationParams Parse(
            JsonElement root,
            ServerHostingOptions options,
            out string error)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));

            error = null;
            var p = CreateDefaults(options);

            if (root.TryGetProperty("width", out var w) && w.TryGetInt32(out int wi)) p.Width = wi;
            if (root.TryGetProperty("height", out var h) && h.TryGetInt32(out int hi)) p.Height = hi;
            if (root.TryGetProperty("frames", out var f) && f.TryGetInt32(out int fi)) p.Frames = fi;
            if (root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si)) p.Steps = si;
            if (root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv)) p.CfgScale = cv;
            if (root.TryGetProperty("cfg2", out var cf2) && cf2.TryGetSingle(out float cv2)) p.CfgScale2 = cv2;
            if (root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv)) p.Seed = sv;
            if (root.TryGetProperty("fps", out var fp) && fp.TryGetInt32(out int fv)) p.Fps = fv;
            if (root.TryGetProperty("flowShift", out var fs) && fs.TryGetSingle(out float fsv)) p.FlowShift = fsv;
            if (root.TryGetProperty("negativePrompt", out var np) && np.ValueKind == JsonValueKind.String)
                p.NegativePrompt = np.GetString();
            if (root.TryGetProperty("sampler", out var sm) && sm.ValueKind == JsonValueKind.String)
                p.Sampler = sm.GetString();
            // Opt-in guidance cache: run the unconditional pass on one step in N.
            if (root.TryGetProperty("cfgCacheStride", out var cc) && cc.TryGetInt32(out int ccv))
                p.CfgCacheStride = ccv;

            // How to interpret supplied images (t2v / i2v / fl2v / ref); null = infer.
            if (TryGetEither(root, "videoMode", "video_mode", out var vm)
                && vm.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(vm.GetString()))
                p.Mode = vm.GetString();

            // Decode the joint audio track (models that have one). Video-only models ignore it.
            if (TryGetEither(root, "generateAudio", "generate_audio", out var ga)
                && (ga.ValueKind == JsonValueKind.True || ga.ValueKind == JsonValueKind.False))
                p.GenerateAudio = ga.GetBoolean();

            // Conditioning image for Wan 2.2 image-to-video: either a previously uploaded
            // file ("imagePath" — the bare server filename from /api/upload, or an
            // absolute path inside the upload directory) or inline base64 ("image", API
            // flow; a data:...;base64, prefix is accepted).
            if (TryGetEither(root, "imagePath", "image_path", out var ip) && ip.ValueKind == JsonValueKind.String
                && !string.IsNullOrWhiteSpace(ip.GetString()))
            {
                if (!UploadFileReference.TryResolve(options.UploadDirectory, ip.GetString(), out string full)
                    || !File.Exists(full))
                {
                    error = "imagePath must reference a previously uploaded file.";
                    return p;
                }
                p.ImageBytes = File.ReadAllBytes(full);
            }
            else if (root.TryGetProperty("image", out var ib) && ib.ValueKind == JsonValueKind.String
                     && !string.IsNullOrWhiteSpace(ib.GetString()))
            {
                string b64 = ib.GetString();
                int comma = b64.IndexOf(',');
                if (b64.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && comma > 0)
                    b64 = b64[(comma + 1)..];
                try { p.ImageBytes = Convert.FromBase64String(b64); }
                catch (FormatException) { error = "image must be base64-encoded (optionally a data: URL)."; }
            }

            // Last-frame conditioning, and reference images/videos/audio. Every one of these
            // is a server-side path, so each goes through the same upload-root confinement
            // check as imagePath — a request must not be able to name a file outside it.
            if (TryGetEither(root, "endImage", "end_image", out var ei) && ei.ValueKind == JsonValueKind.String
                && !string.IsNullOrWhiteSpace(ei.GetString()))
            {
                if (!TryResolveUpload(options, ei.GetString(), out string endFull))
                {
                    error = "endImage must reference a previously uploaded file.";
                    return p;
                }
                p.EndImagePath = endFull;
            }

            p.ReferenceImagePaths = ParseReferenceList(
                root, "referenceImages", "reference_images", options, "referenceImages", ref error);
            if (error != null) return p;
            p.ReferenceVideoPaths = ParseReferenceList(
                root, "referenceVideos", "reference_videos", options, "referenceVideos", ref error);
            if (error != null) return p;
            p.ReferenceAudioPaths = ParseReferenceList(
                root, "referenceAudios", "reference_audios", options, "referenceAudios", ref error);
            // Paired BY INDEX with referenceVideos: entry i is clip i's soundtrack.
            p.ReferenceVideoAudioPaths = ParseReferenceList(
                root, "referenceVideoAudios", "reference_video_audios", options,
                "referenceVideoAudios", ref error);
            if (error != null) return p;

            p.ImageConditions = ParseImageConditions(root, options, ref error);

            return p;
        }

        private static List<VideoImageCondition> ParseImageConditions(
            JsonElement root, ServerHostingOptions options, ref string error)
        {
            if (!TryGetEither(root, "imageConditions", "image_conditions", out var conditions))
                return null;
            if (conditions.ValueKind != JsonValueKind.Array)
            {
                error = "imageConditions must be an array of image, frameIndex and strength objects.";
                return null;
            }
            var result = new List<VideoImageCondition>();
            var frames = new HashSet<int>();
            foreach (var item in conditions.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.Object)
                {
                    error = "imageConditions entries must be objects.";
                    return null;
                }
                var condition = new VideoImageCondition();
                if (!(TryGetEither(item, "frameIndex", "frame_index", out var frame)
                        || TryGetEither(item, "frameIdx", "frame_idx", out frame))
                    || frame.ValueKind != JsonValueKind.Number || !frame.TryGetInt32(out int index)
                    || index < -1)
                {
                    error = "imageConditions frameIndex must be a non-negative integer, or -1 for the last frame.";
                    return null;
                }
                condition.FrameIndex = index;
                if (!frames.Add(index))
                {
                    error = "imageConditions must not contain duplicate frameIndex values.";
                    return null;
                }
                if (item.TryGetProperty("strength", out var strength))
                {
                    if (strength.ValueKind != JsonValueKind.Number || !strength.TryGetSingle(out float value)
                        || !float.IsFinite(value) || value < 0 || value > 1)
                    {
                        error = "imageConditions strength must be a finite number between 0 and 1.";
                        return null;
                    }
                    condition.Strength = value;
                }
                if (TryGetEither(item, "imagePath", "image_path", out var path))
                {
                    if (path.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(path.GetString())
                        || !TryResolveUpload(options, path.GetString(), out string full))
                    {
                        error = "imageConditions imagePath must reference a previously uploaded file.";
                        return null;
                    }
                    condition.ImagePath = full;
                }
                else if (item.TryGetProperty("image", out var encoded) && encoded.ValueKind == JsonValueKind.String)
                {
                    string base64 = encoded.GetString();
                    int comma = base64.IndexOf(',');
                    if (base64.StartsWith("data:", StringComparison.OrdinalIgnoreCase) && comma > 0)
                        base64 = base64[(comma + 1)..];
                    try { condition.ImageBytes = Convert.FromBase64String(base64); }
                    catch (FormatException)
                    {
                        error = "imageConditions image must be base64-encoded (optionally a data: URL).";
                        return null;
                    }
                    if (condition.ImageBytes.Length == 0)
                    {
                        error = "imageConditions image must contain an encoded image.";
                        return null;
                    }
                }
                else
                {
                    error = "imageConditions entries require imagePath or a base64-encoded image.";
                    return null;
                }
                result.Add(condition);
            }
            return result.Count == 0 ? null : result;
        }

        // Accept either spelling of a field name; camelCase wins when both are present.
        private static bool TryGetEither(JsonElement root, string camel, string snake, out JsonElement value)
            => root.TryGetProperty(camel, out value) || root.TryGetProperty(snake, out value);

        private static bool TryResolveUpload(ServerHostingOptions options, string reference, out string full)
            => UploadFileReference.TryResolve(options.UploadDirectory, reference, out full) && File.Exists(full);

        private static List<string> ParseReferenceList(
            JsonElement root, string camel, string snake,
            ServerHostingOptions options, string fieldName, ref string error)
        {
            if (!TryGetEither(root, camel, snake, out var arr) || arr.ValueKind != JsonValueKind.Array)
                return null;

            var result = new List<string>();
            foreach (var item in arr.EnumerateArray())
            {
                if (item.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(item.GetString()))
                {
                    error = $"{fieldName} entries must be non-empty strings.";
                    return null;
                }
                if (!TryResolveUpload(options, item.GetString(), out string full))
                {
                    error = $"{fieldName} entries must reference previously uploaded files.";
                    return null;
                }
                result.Add(full);
            }
            return result.Count == 0 ? null : result;
        }
    }
}
