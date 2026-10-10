// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;

namespace TensorAgent.Core.Sessions;

/// <summary>
/// What an image model's turn writes down beside what the page shows (see
/// <see cref="Hosting.ImageTurns"/>): for a picture, what it was made from; for a turn that
/// asked instead of guessing, the readings it offered.
///
/// <para>
/// The next turn reads the first back to tell which picture a follow-up is about and to make
/// a picture again, so it is taken from the frames this side sent, by the same readers that
/// take the picture itself, rather than left to the page to copy.
/// </para>
/// </summary>
public sealed record ImageTurnRecord(
    string? Plan, IReadOnlyList<string>? Sources, string? Prompt, long? Seed, StoredImageMask? Mask,
    IReadOnlyList<StoredImageChoice>? Choices)
{
    /// <summary>What decided <see cref="Plan"/> (see <see cref="StoredMessage.ImagePlanReason"/>).</summary>
    public string? Reason { get; init; }

    /// <summary>
    /// The record a frame carries: a picture's frame (<c>imageUrl</c> with <c>imagePlan</c>
    /// and the rest) or a question's (<c>image_choice</c>). Null for any other frame.
    /// </summary>
    public static ImageTurnRecord? FromFrame(JsonElement frame)
    {
        if (frame.ValueKind != JsonValueKind.Object)
            return null;

        if (frame.TryGetProperty("image_choice", out JsonElement offered) && offered.ValueKind == JsonValueKind.Array)
        {
            var choices = new List<StoredImageChoice>();
            foreach (JsonElement choice in offered.EnumerateArray())
            {
                if (choice.ValueKind != JsonValueKind.Object || Text(choice, "intent") is not { Length: > 0 } intent)
                    continue;
                choices.Add(new StoredImageChoice { Intent = intent, Source = Text(choice, "source") });
            }
            return choices.Count == 0 ? null : new ImageTurnRecord(null, null, null, null, null, choices);
        }

        if (!frame.TryGetProperty("imageUrl", out _) || Text(frame, "imagePlan") is not { Length: > 0 } plan)
            return null;
        var sources = new List<string>();
        if (frame.TryGetProperty("imageSources", out JsonElement list) && list.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement source in list.EnumerateArray())
                if (source.ValueKind == JsonValueKind.String && source.GetString() is { Length: > 0 } name)
                    sources.Add(name);
        }
        long? seed = frame.TryGetProperty("imageSeed", out JsonElement s) && s.ValueKind == JsonValueKind.Number
            && s.TryGetInt64(out long value) ? value : null;
        return new ImageTurnRecord(plan, sources, Text(frame, "imagePrompt"), seed, MaskOf(frame), null)
        {
            Reason = Text(frame, "imagePlanReason"),
        };
    }

    /// <summary>What a saved message's picture was made from, or null when nothing recorded it.</summary>
    public static ImageTurnRecord? MadeFrom(StoredMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        return message.ImagePlan is null
            ? null
            : new ImageTurnRecord(message.ImagePlan, message.ImageSources, message.ImagePrompt, message.ImageSeed, message.ImageMask, null)
            {
                Reason = message.ImagePlanReason,
            };
    }

    /// <summary>Write this onto <paramref name="message"/>; what the record does not carry is left as it is.</summary>
    public void ApplyTo(StoredMessage message)
    {
        ArgumentNullException.ThrowIfNull(message);
        if (Plan is not null)
        {
            message.ImagePlan = Plan;
            message.ImagePlanReason = Reason;
            message.ImageSources = Sources is null ? null : new List<string>(Sources);
            message.ImagePrompt = Prompt;
            message.ImageSeed = Seed;
            message.ImageMask = Mask;
        }
        if (Choices is not null)
            message.ImageChoices = new List<StoredImageChoice>(Choices);
    }

    /// <summary>
    /// The selection off the frame. Read on its own, because the picture it came with must be
    /// written down even if the selection cannot be: the readers that call this drop a whole
    /// frame that does not parse.
    /// </summary>
    private static StoredImageMask? MaskOf(JsonElement frame)
    {
        if (!frame.TryGetProperty("imageMask", out JsonElement mask) || mask.ValueKind != JsonValueKind.Object)
            return null;
        try
        {
            return mask.Deserialize<StoredImageMask>(ConversationStore.Json);
        }
        catch (JsonException)
        {
            return null;
        }
    }

    private static string? Text(JsonElement element, string name) =>
        element.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
}
