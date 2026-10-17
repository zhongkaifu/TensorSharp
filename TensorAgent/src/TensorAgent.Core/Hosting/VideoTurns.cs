// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorAgent.Core.Localization;
using TensorSharp.Chat;
using TensorSharp.Models.Video;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// A chat turn when the loaded model makes video clips (MiniMax-H3 or Wan).
///
/// <para>
/// The same arrangement as <see cref="ImageTurns"/>: the page's one route for a turn,
/// <c>/api/chat</c>, runs under the turn manager and the conversation recorder, so a clip
/// that takes minutes survives the page being hidden and is in the saved chat when it is
/// done. The newest user message is the request. Its words are the description; its photos,
/// clips and sound files become whatever the loaded checkpoint can be given - keyframes for
/// the first-and-last-frame checkpoint, references for the reference one - and a
/// combination it cannot use is refused here, in the app's own words, before the model
/// starts.
/// </para>
/// <para>
/// The video service's frames are translated into the chat stream's vocabulary:
/// <c>video_step</c> with a <c>video_phase</c> while the clip is made (and <c>video_loras</c>,
/// the LoRA plug-ins it is made with, when there are any), one <c>videoUrl</c> for the finished
/// clip (with <c>audioUrl</c> only when its sound is a separate file the page must play
/// itself), and the usual <c>done</c> with the session, which is what the recorder keys the
/// transcript on.
/// </para>
/// </summary>
public static class VideoTurns
{
    /// <summary>
    /// MiniMax-H3's preview length, on its 17k+5 grid: 22 is 0.92 s at 24 fps, the length the
    /// shipped MiniMax-H3 configs make too. The page offers no length setting, so every
    /// turn gets this. MEASURED on an M5 Pro (ggml_metal, 640x384, 20 steps, CLI): 22 frames
    /// take 148 s, 39 frames 279 s, and 56 frames about 7 minutes, since every step attends
    /// over the whole clip.
    ///
    /// <para>The step count is left to the model: 20, the reference's own default, or the
    /// count a MiniMax-H3 Turbo plug-in turned on under LoRAs fixes (8 or 4, with its own
    /// schedule; see <c>config/lora/minimax-h3-fl2v-turbo-*.json</c>). A clip's request never
    /// carries steps, a flow shift or a sampler for H3: a plug-in's contract fixes them and
    /// refuses any other value, and without one the model's own defaults apply.</para>
    /// </summary>
    public const int DefaultFrames = 22;

    /// <summary>Wan uses a 4k+1 frame grid; keep its default short too.</summary>
    public const int DefaultWanFrames = 21;

    /// <summary>
    /// Run the turn <paramref name="body"/> describes and yield its chat frames.
    /// </summary>
    /// <param name="chat">The chat service, with a video model loaded.</param>
    /// <param name="body">The <c>/api/chat</c> request the page sent.</param>
    /// <param name="cancellationToken">Ends the turn; the clip is abandoned.</param>
    /// <param name="prepare">Asked once, after the request is read and recorded, for the LoRA
    /// plug-ins the clip is made with (always with false: a clip edits no photo). The video
    /// service applies them under the lock the clip runs in; a refusal ends the turn with its
    /// reason. Null for a host that chooses none, whose clips keep the model's own set.</param>
    public static async IAsyncEnumerable<object> StreamAsync(
        WebUiChatService chat, JsonElement body, [EnumeratorCancellation] CancellationToken cancellationToken,
        Func<bool, ImageTurns.Preparation>? prepare = null)
    {
        ArgumentNullException.ThrowIfNull(chat);

        string? sessionId = body.ValueKind == JsonValueKind.Object
            && body.TryGetProperty("sessionId", out JsonElement id) && id.ValueKind == JsonValueKind.String
                ? id.GetString()
                : null;
        // Held for the whole turn, as a text turn holds it: a shared item being sent here
        // must not be discarded underneath it. A refusal is thrown before the first frame,
        // which is what turns it into a status code rather than a stream.
        using IDisposable? lease = chat.AcquireChatRequestLease?.Invoke(body);

        IVideoGenerationModel? model = chat.LoadedVideoModel;
        VideoRequest request = model is null
            ? VideoRequest.Refused(Loc.T("host.video.notVideoModel"))
            : Read(body, model);
        if (request.Error is not null)
        {
            yield return new { done = true, error = request.Error, sessionId };
            yield break;
        }

        // Recorded exactly as a text turn is (see ImageTurns).
        if (!string.IsNullOrEmpty(sessionId))
            chat.OnChatRequest?.Invoke(sessionId, body);

        // As a picture asks after its plan: the clip is made with the plug-ins chosen now, or
        // not at all, rather than without the ones the user turned on.
        ImageTurns.Preparation? prepared = prepare?.Invoke(false);
        if (prepared?.Error is { } refusal)
        {
            yield return new { done = true, error = refusal, sessionId };
            yield break;
        }

        using JsonDocument payload = JsonDocument.Parse(JsonSerializer.Serialize(request.Payload));
        JsonElement service = payload.RootElement.Clone();
        IAsyncEnumerable<object> frames = prepared is null
            ? chat.VideoGenerateStreamAsync(service, cancellationToken)
            : chat.VideoGenerateStreamAsync(service, prepared.Specs, cancellationToken);
        await foreach (object frame in Translate(frames, sessionId, cancellationToken, prepared?.Loras).ConfigureAwait(false))
            yield return frame;
    }

    /// <summary>The video service's frames, as the chat stream's. A step frame names the LoRA
    /// plug-ins in use (<c>video_loras</c>) when there are any.</summary>
    internal static async IAsyncEnumerable<object> Translate(
        IAsyncEnumerable<object> frames, string? sessionId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        IReadOnlyList<string>? loras = null)
    {
        string? lastPhase = null;
        int lastStep = -1;
        bool lastTimed = false;
        await foreach (object frame in frames.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(frame));
            JsonElement f = document.RootElement;
            if (f.TryGetProperty("done", out _))
            {
                if (f.TryGetProperty("error", out JsonElement error) && error.ValueKind == JsonValueKind.String)
                {
                    yield return new { done = true, error = error.GetString(), sessionId };
                    yield break;
                }
                string? audioUrl = Text(f, "audioUrl");
                bool muxed = f.TryGetProperty("audioMuxed", out JsonElement m) && m.ValueKind == JsonValueKind.True;
                yield return new
                {
                    videoUrl = Text(f, "url"),
                    // Only a soundtrack the page has to play beside the clip; one inside
                    // the MP4 plays with it.
                    audioUrl = muxed ? null : audioUrl,
                    width = Number(f, "width"),
                    height = Number(f, "height"),
                    frames = Number(f, "frames"),
                    fps = Number(f, "fps"),
                    seed = f.TryGetProperty("seed", out JsonElement s) && s.TryGetInt64(out long seed) ? seed : 0,
                    hasAudio = audioUrl is { Length: > 0 },
                };
                yield return new
                {
                    done = true,
                    sessionId,
                    tokenCount = 0,
                    elapsed = Seconds(f, "elapsedSeconds"),
                    tokPerSec = 0.0,
                    truncated = false,
                };
                yield break;
            }

            // The service reports every denoising step twice: first from the step hook,
            // with no phase and no timing, then as the "denoise" phase with the elapsed
            // time and an estimate. A repeat is dropped unless it is the one that brings
            // the timing, and a VAE that reports each of its tiles is one phase, not six.
            // The pipeline reports "done" when the frames are made; writing the MP4 comes
            // after that, so the page hears it as the clip being saved.
            string phase = Text(f, "phase") switch
            {
                null => "denoise",
                "done" => "encode",
                string named => named,
            };
            int step = Number(f, "step");
            double elapsed = Seconds(f, "elapsedSeconds");
            if (phase == lastPhase && step == lastStep && (lastTimed || elapsed <= 0))
                continue;
            lastPhase = phase;
            lastStep = step;
            lastTimed = elapsed > 0;
            double etaSeconds = f.TryGetProperty("etaSeconds", out JsonElement e) && e.TryGetDouble(out double eta) ? eta : -1;
            if (loras is { Count: > 0 })
            {
                yield return new
                {
                    video_step = step,
                    video_steps = Number(f, "total"),
                    video_phase = phase,
                    elapsed,
                    eta = etaSeconds,
                    video_loras = loras,
                };
                continue;
            }
            yield return new
            {
                video_step = step,
                video_steps = Number(f, "total"),
                video_phase = phase,
                elapsed,
                eta = etaSeconds,
            };
        }

        // The service ends without a terminal frame only when the turn was stopped.
        yield return new { done = true, aborted = true, sessionId };
    }

    /// <summary>
    /// What the newest user message asks the loaded model for, as the video service's body,
    /// or why it cannot.
    /// </summary>
    internal static VideoRequest Read(JsonElement body, IVideoGenerationModel model)
    {
        if (body.ValueKind != JsonValueKind.Object
            || !body.TryGetProperty("messages", out JsonElement messages)
            || messages.ValueKind != JsonValueKind.Array)
            return VideoRequest.Refused(Describe);

        JsonElement? last = null;
        foreach (JsonElement message in messages.EnumerateArray())
        {
            if (message.ValueKind == JsonValueKind.Object
                && message.TryGetProperty("role", out JsonElement role)
                && role.ValueKind == JsonValueKind.String
                && role.GetString() == "user")
                last = message;
        }
        if (last is not { } user)
            return VideoRequest.Refused(Describe);

        string prompt = Text(user, "content")?.Trim() ?? string.Empty;
        if (prompt.Length == 0)
            return VideoRequest.Refused(Describe);
        // The page puts a document's text in front of the message, where it would become the
        // clip's script; a video model takes words, pictures, clips and sounds, not files.
        if (Paths(user, "textFilePaths").Length > 0)
            return VideoRequest.Refused(Loc.T("host.video.noDocuments"));

        // Stills only for photos: a clip's sampled frames are in imagePaths as well, and
        // they are not a picture the user chose.
        string[] photos = Paths(user, "stillImagePaths");
        string[] clips = Paths(user, "videoFilePaths");
        string[] sounds = Paths(user, "audioPaths");

        if (!model.SupportsKeyframeConditioning
            && ((user.TryGetProperty("imageConditions", out JsonElement unsupported) && unsupported.ValueKind != JsonValueKind.Null)
                || (user.TryGetProperty("image_conditions", out unsupported) && unsupported.ValueKind != JsonValueKind.Null)))
            return VideoRequest.Refused(Loc.T("host.video.keyframesUnavailable"));

        var payload = new Dictionary<string, object>
        {
            ["prompt"] = prompt,
        };
        // Frame grids belong to the model: new families must not inherit H3's
        // 17k+5 length merely because they are not Wan. An unspecified preference
        // leaves the request unset so the model can apply its own valid default.
        if (model.DefaultPreviewFrames > 0)
            payload["frames"] = model.DefaultPreviewFrames;

        if (model.VideoModelFamily == "ltx-2.5")
        {
            string? requestedMode = Text(user, "videoMode") ?? Text(user, "video_mode")
                ?? Text(body, "videoMode") ?? Text(body, "video_mode");
            if (string.Equals(requestedMode, "ref", StringComparison.OrdinalIgnoreCase) && !model.SupportsReferenceConditioning)
                return VideoRequest.Refused(Loc.T("host.video.ingredientsUnavailable"));
            // The published distilled recipe must override generic video defaults.
            payload["steps"] = 8;
            payload["cfg"] = 1f;
            payload["sampler"] = "euler_ancestral";
            payload["flowShift"] = 0f;
            // Size is left to the model (1024x1024): LTX 2.5 samples its first pass at
            // half the output size, and a 512x512 output smears hands and faces.
            payload["fps"] = 24;
            payload["videoMode"] = "t2va";
            payload["generateAudio"] = true;
            if (user.TryGetProperty("videoFrames", out JsonElement clipFrames))
            {
                if (clipFrames.ValueKind != JsonValueKind.Number || !clipFrames.TryGetInt32(out int requestedFrames)
                    || requestedFrames < 1 || (requestedFrames - 1) % 8 != 0)
                    return VideoRequest.Refused(Loc.T("host.video.invalidFrames"));
                payload["frames"] = requestedFrames;
            }
            if (string.Equals(requestedMode, "ref", StringComparison.OrdinalIgnoreCase))
            {
                if (clips.Length > 0 || sounds.Length > 0)
                    return VideoRequest.Refused(Loc.T("host.video.photosOnly"));
                if (photos.Length > model.MaxReferenceImages)
                    return VideoRequest.Refused(Loc.T("host.video.tooManyReferences", ("max", model.MaxReferenceImages), ("count", photos.Length)));
                payload["width"] = 768; payload["height"] = 448; payload["frames"] = 121;
                payload["videoMode"] = "ref"; payload["referenceImages"] = photos;
                return VideoRequest.For(payload);
            }
        }

        if (model.SupportsKeyframeConditioning)
        {
            // LTX images describe moments in the output timeline. They are not
            // identity references, and a clip/audio adapter needs separate weights.
            if (clips.Length > 0 || sounds.Length > 0)
                return VideoRequest.Refused(Loc.T("host.video.photosOnly"));
            if (user.TryGetProperty("imageConditions", out JsonElement conditions)
                || user.TryGetProperty("image_conditions", out conditions))
            {
                if (conditions.ValueKind == JsonValueKind.Array && model.MaxReferenceImages > 0
                    && conditions.GetArrayLength() > model.MaxReferenceImages)
                    return VideoRequest.Refused(Loc.T("host.video.tooManyReferences", ("max", model.MaxReferenceImages), ("count", conditions.GetArrayLength())));
                payload["imageConditions"] = conditions.Clone();
                payload["videoMode"] = "keyframes";
                return VideoRequest.For(payload);
            }
            int limit = model.MaxReferenceImages;
            if (limit > 0 && photos.Length > limit)
                return VideoRequest.Refused(Loc.T("host.video.tooManyReferences", ("max", limit), ("count", photos.Length)));
            if (photos.Length > 2)
            {
                int frames = payload.TryGetValue("frames", out object selectedFrames) ? (int)selectedFrames : model.DefaultPreviewFrames;
                if (frames < photos.Length)
                    return VideoRequest.Refused(Loc.T("host.video.tooManyReferences", ("max", frames), ("count", photos.Length)));
                payload["imageConditions"] = photos.Select((path, index) => new
                {
                    imagePath = path,
                    frameIndex = (int)Math.Round((double)index * (frames - 1) / (photos.Length - 1), MidpointRounding.AwayFromZero),
                    strength = 1f,
                }).ToArray();
                payload["videoMode"] = "keyframes";
            }
            else
            {
                if (photos.Length > 0) payload["imagePath"] = photos[0];
                if (photos.Length > 1) payload["endImage"] = photos[1];
                if (photos.Length > 0) payload["videoMode"] = photos.Length == 1 ? "i2v" : "fl2v";
            }
            return VideoRequest.For(payload);
        }

        if (model.SupportsReferenceConditioning)
        {
            int count = photos.Length + clips.Length + sounds.Length;
            int max = model.MaxReferenceImages;
            if (max > 0 && count > max)
                return VideoRequest.Refused(Loc.T("host.video.tooManyReferences", ("max", max), ("count", count)));
            if (photos.Length > 0)
                payload["referenceImages"] = photos;
            if (clips.Length > 0)
                payload["referenceVideos"] = clips;
            if (sounds.Length > 0)
                payload["referenceAudios"] = sounds;
            // This turn decides how its attachments are used. An inherited startup
            // mode (for example i2v) must not reinterpret a reference as a first frame.
            if (count > 0)
                payload["videoMode"] = "ref";
            return VideoRequest.For(payload);
        }

        if (clips.Length > 0 || sounds.Length > 0)
        {
            return VideoRequest.Refused(model.VideoModelFamily == "minimax-h3"
                ? Loc.T("host.video.photosOnlyUseReferences")
                : Loc.T("host.video.photosOnly"));
        }

        int photoLimit = !model.SupportsImageConditioning ? 0 : model.SupportsEndImageConditioning ? 2 : 1;
        if (photos.Length > photoLimit)
        {
            return VideoRequest.Refused(photoLimit switch
            {
                0 => Loc.T("host.video.descriptionOnly"),
                1 => Loc.T("host.video.onePhoto"),
                _ => Loc.T("host.video.oneOrTwoPhotos"),
            });
        }
        if (photos.Length > 0)
            payload["imagePath"] = photos[0];
        if (photos.Length > 1)
            payload["endImage"] = photos[1];
        if (photos.Length > 0)
            payload["videoMode"] = photos.Length > 1 ? "fl2v" : "i2v";
        return VideoRequest.For(payload);
    }

    private static string Describe => Loc.T("host.video.describe");

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static int Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.TryGetInt32(out int n) ? n : 0;

    private static double Seconds(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.TryGetDouble(out double seconds) ? seconds : 0;

    private static string[] Paths(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray()
                .Where(p => p.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(p.GetString()))
                .Select(p => p.GetString()!)
                .ToArray()
            : Array.Empty<string>();

    /// <summary>The service body for a clip, or the reason there is none.</summary>
    internal sealed record VideoRequest(Dictionary<string, object>? Payload, string? Error)
    {
        public static VideoRequest For(Dictionary<string, object> payload) => new(payload, null);
        public static VideoRequest Refused(string error) => new(null, error);
    }
}
