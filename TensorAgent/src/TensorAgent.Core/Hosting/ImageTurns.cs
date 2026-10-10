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
using TensorSharp.Runtime;

namespace TensorAgent.Core.Hosting;

/// <summary>
/// A chat turn when the loaded model makes pictures instead of text.
///
/// <para>
/// The page has one route for a turn, <c>/api/chat</c>, and on this host that route runs
/// under the turn manager and the conversation recorder: the turn survives the page being
/// hidden, and what it produced is written down when it ends. An image model answers the
/// same route rather than a second one the page would have to drive on its own.
/// </para>
/// <para>
/// The newest user message is the request, read against the conversation it ends (see
/// <see cref="PlanAsync(ImageConversation, Planner, CancellationToken)"/>). Photos attached
/// to it are what it edits. Otherwise it may change a picture already in the chat, ask for
/// a new one, or ask for another version of the last one; which, when the conversation's
/// shape does not settle it, the image model's own language model is asked. The turn used to
/// read the newest message alone, so "make it brighter" after a picture drew a new picture
/// of those words.
/// </para>
/// <para>
/// The image service's frames are translated into the chat stream's vocabulary:
/// <c>image_plan</c> once the turn knows what it will do, <c>image_step</c> while the
/// picture denoises (some carry a small <c>preview</c>), one <c>imageUrl</c> for the
/// finished picture with what it was made from, and the usual <c>done</c> with the session,
/// which is what the recorder keys the transcript on. A turn that is unsure what was meant
/// answers with a question and <c>image_choice</c> instead of a picture.
/// </para>
/// </summary>
public static partial class ImageTurns
{
    /// <summary>
    /// The output area a turn asks for: 1024 x 1024 for a picture made from words. An edit
    /// returns its source picture's own size and samples at this area at most, at the
    /// picture's shape (see <see cref="ImagePlan.Payload"/>). The model's native area is 2048
    /// x 2048, which has four times the image tokens and costs several times as long per
    /// step; the page offers no size setting, so this is the budget every turn gets.
    /// </summary>
    public const long DefaultTargetArea = 1024L * 1024;

    /// <summary>
    /// Where a chat turn's frames come from: the image service when the loaded model
    /// makes pictures, the video service (<see cref="VideoTurns"/>) when it makes clips,
    /// the chat service otherwise.
    ///
    /// <para>
    /// One decision for every caller that answers <c>/api/chat</c> — the route's own
    /// default and the app host's GPU gate, which supplies its frames in place of that
    /// default. The first version of this decided inside the route, the app passed its
    /// gate, and a picture request in the app went to the text pipeline and failed on its
    /// first frame; every unit test passed, because none of them ran the app's wiring.
    /// </para>
    /// </summary>
    /// <param name="prepare">Asked before a picture is started, with whether it is an edit: the
    /// LoRA plug-ins it is made with (see <see cref="Preparation"/>). Null for a host that chooses
    /// none, whose pictures keep whatever set the model was loaded with.</param>
    /// <param name="planner">How a picture turn reads the conversation (see <see cref="Planner"/>).
    /// Required rather than defaulted for the same reason this method exists: a caller that left
    /// it out would quietly plan every turn from the newest message alone again.</param>
    public static IAsyncEnumerable<object> FramesFor(
        WebUiChatService chat, JsonElement body, CancellationToken cancellationToken,
        Func<bool, Preparation>? prepare, Planner planner)
    {
        ArgumentNullException.ThrowIfNull(chat);
        ArgumentNullException.ThrowIfNull(planner);
        return chat.LoadedModelMakesImages ? StreamAsync(chat, body, planner, cancellationToken, prepare)
            : chat.LoadedModelMakesVideo ? VideoTurns.StreamAsync(chat, body, cancellationToken)
            : chat.ChatStreamAsync(body, cancellationToken);
    }

    /// <summary>
    /// The LoRA plug-ins a picture is made with: the engine's specs, applied under the same lock
    /// as the run, and their names (the page says what it is drawing with). Or why the picture
    /// cannot be made as the user asked, which ends the turn with that message rather than with
    /// a picture made without the plug-ins they turned on.
    /// </summary>
    public sealed record Preparation(string? Error, IReadOnlyList<LoraSpec> Specs, IReadOnlyList<string> Loras)
    {
        public static Preparation Ready(IReadOnlyList<LoraSpec> specs, IReadOnlyList<string> loras) => new(null, specs, loras);
        public static Preparation Refused(string error) => new(error, Array.Empty<LoraSpec>(), Array.Empty<string>());
    }

    /// <summary>
    /// Run the turn <paramref name="body"/> describes and yield its chat frames.
    /// </summary>
    /// <param name="chat">The chat service, with an image model loaded.</param>
    /// <param name="body">The <c>/api/chat</c> request the page sent.</param>
    /// <param name="planner">See <see cref="FramesFor"/>.</param>
    /// <param name="cancellationToken">Ends the turn; the picture is abandoned.</param>
    /// <param name="prepare">See <see cref="FramesFor"/>.</param>
    public static async IAsyncEnumerable<object> StreamAsync(
        WebUiChatService chat, JsonElement body, Planner planner,
        [EnumeratorCancellation] CancellationToken cancellationToken,
        Func<bool, Preparation>? prepare = null)
    {
        ArgumentNullException.ThrowIfNull(chat);
        ArgumentNullException.ThrowIfNull(planner);

        string? sessionId = body.ValueKind == JsonValueKind.Object
            && body.TryGetProperty("sessionId", out JsonElement id) && id.ValueKind == JsonValueKind.String
                ? id.GetString()
                : null;
        // Held for the whole turn, as a text turn holds it: a shared photo being sent here
        // must not be discarded underneath the edit. A refusal is thrown before the first
        // frame, which is what turns it into a status code rather than a stream.
        using IDisposable? lease = chat.AcquireChatRequestLease?.Invoke(body);
        ImageConversation? conversation = ReadConversation(body, planner.UploadDirectory);
        if (conversation is null)
        {
            yield return new
            {
                done = true,
                error = Loc.T("host.image.describe"),
                sessionId,
            };
            yield break;
        }

        // Recorded exactly as a text turn is: the host's handler writes the user's
        // message into the conversation and settles a shared item the turn consumed.
        if (!string.IsNullOrEmpty(sessionId))
            chat.OnChatRequest?.Invoke(sessionId, body);

        // Cancellation and faults from the image model's answer are not caught: a stopped
        // turn is a stopped turn, and the app's GPU gate tells a damaged engine from a refusal
        // only by what reaches it (AgentAppHost.GatedChatFrames).
        ImagePlan plan = await PlanAsync(conversation, planner, cancellationToken).ConfigureAwait(false);
        if (plan.Choices is { } choices)
        {
            // The question is the answer's text, so the transcript and a text model reading
            // this chat later see what the user saw; the buttons ride on their own frame.
            yield return new { token = Loc.T("host.image.choose") };
            yield return new { image_choice = choices.Select(c => new { intent = c.Intent, source = c.Source }).ToArray() };
            yield return new { done = true, sessionId };
            yield break;
        }

        // After the plan, because the plug-ins depend on it: one made only for edits is not
        // applied to a picture drawn from words.
        Preparation? prepared = prepare?.Invoke(plan.Editing);
        if (prepared?.Error is { } refusal)
        {
            yield return new { done = true, error = refusal, sessionId };
            yield break;
        }

        // Before the first step, so the page can say what it is doing while it does it. A turn
        // the GPU gate runs again plans again and sends this again; the page keeps the last.
        yield return new
        {
            image_plan = plan.Kind,
            image_sources = plan.Sources.Select(UploadUrl).ToArray(),
            image_prompt = plan.Prompt,
            image_plan_reason = plan.Reason,
        };

        using JsonDocument payload = JsonDocument.Parse(JsonSerializer.Serialize(plan.Payload));
        JsonElement service = payload.RootElement.Clone();
        IAsyncEnumerable<object> frames = (plan.Editing, prepared) switch
        {
            (true, null) => chat.ImageEditStreamAsync(service, cancellationToken),
            (false, null) => chat.ImageGenerateStreamAsync(service, cancellationToken),
            (true, _) => chat.ImageEditStreamAsync(service, prepared.Specs, cancellationToken),
            (false, _) => chat.ImageGenerateStreamAsync(service, prepared.Specs, cancellationToken),
        };

        await foreach (object frame in Translate(frames, sessionId, cancellationToken, prepared?.Loras, plan).ConfigureAwait(false))
            yield return frame;
    }

    /// <summary>The image service's frames, as the chat stream's. A step frame names the LoRA
    /// plug-ins in use (<c>image_loras</c>) when there are any; the picture's frame carries what
    /// <paramref name="plan"/> made it from, which the recorders write down beside it.</summary>
    internal static async IAsyncEnumerable<object> Translate(
        IAsyncEnumerable<object> frames, string? sessionId,
        [EnumeratorCancellation] CancellationToken cancellationToken = default,
        IReadOnlyList<string>? loras = null, ImagePlan? plan = null)
    {
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
                if (plan is null)
                {
                    yield return new { imageUrl = Text(f, "url"), width = Number(f, "width"), height = Number(f, "height") };
                }
                else
                {
                    yield return new
                    {
                        imageUrl = Text(f, "url"),
                        width = Number(f, "width"),
                        height = Number(f, "height"),
                        imagePlan = plan.Kind,
                        imagePlanReason = plan.Reason,
                        imageSources = plan.Sources,
                        imagePrompt = plan.Prompt,
                        imageSeed = plan.Seed,
                        imageMask = plan.Mask.Count == 0 ? null : plan.Mask,
                    };
                }
                yield return new
                {
                    done = true,
                    sessionId,
                    tokenCount = 0,
                    elapsed = f.TryGetProperty("elapsedSeconds", out JsonElement s) && s.TryGetDouble(out double seconds) ? seconds : 0,
                    tokPerSec = 0.0,
                    truncated = false,
                };
                yield break;
            }

            if (loras is { Count: > 0 })
            {
                yield return new
                {
                    image_step = Number(f, "step"),
                    image_steps = Number(f, "total"),
                    preview = Text(f, "image"),
                    image_loras = loras,
                };
                continue;
            }
            yield return new
            {
                image_step = Number(f, "step"),
                image_steps = Number(f, "total"),
                preview = Text(f, "image"),
            };
        }

        // The service ends without a terminal frame only when the turn was stopped.
        yield return new { done = true, aborted = true, sessionId };
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;

    private static int Number(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.TryGetInt32(out int n) ? n : 0;

    private static string[] Paths(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.Array
            ? value.EnumerateArray()
                .Where(p => p.ValueKind == JsonValueKind.String && !string.IsNullOrWhiteSpace(p.GetString()))
                .Select(p => p.GetString()!)
                .ToArray()
            : Array.Empty<string>();
}
