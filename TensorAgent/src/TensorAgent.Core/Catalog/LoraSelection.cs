// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorAgent.Core.Localization;
using TensorAgent.Core.Settings;
using TensorSharp.Runtime;

namespace TensorAgent.Core.Catalog;

/// <summary>The engine's specs for the plug-ins in use, with their names for the page.</summary>
/// <param name="Specs">What the image service is handed for the picture: the speed plug-in
/// first, then the others in the order they were turned on.</param>
/// <param name="Names">Their display names, in the same order.</param>
/// <param name="SatOut">Why chosen plug-ins are not applied to this picture, or null.</param>
public sealed record LoraPlan(IReadOnlyList<LoraSpec> Specs, IReadOnlyList<string> Names, string? SatOut = null)
{
    public static LoraPlan None { get; } = new(Array.Empty<LoraSpec>(), Array.Empty<string>());
}

/// <summary>
/// The rules a choice of LoRA plug-ins obeys, in one place for the route that saves a choice
/// and for the picture that applies it: known plug-ins of the model in use, installed, and at
/// most one speed plug-in, because two sampling schedules cannot both apply.
/// </summary>
public static class LoraSelection
{
    /// <summary>
    /// Check a choice the user just made and return it as it will be saved: duplicates
    /// dropped, strengths inside <see cref="LoraCatalog.MinStrength"/>..<see cref="LoraCatalog.MaxStrength"/>,
    /// a speed plug-in at its own strength. The page sends the whole choice with every change,
    /// so only what the change adds has to be downloaded: a plug-in that was already on stays
    /// on when its files have gone (the picture says so, and the sheet can turn it off).
    /// Ids this build does not know that were already saved are kept after the user's choice,
    /// whether or not the request repeats them, because a newer build sharing the settings
    /// chose them and this build cannot show them to be turned off.
    /// </summary>
    /// <returns>The choice to save, or null with <paramref name="error"/> set.</returns>
    public static List<ImageLoraChoice>? Validate(
        IEnumerable<ImageLoraChoice> requested, IReadOnlyList<ImageLoraChoice> saved, LoraStore store, out string? error)
    {
        ArgumentNullException.ThrowIfNull(requested);
        ArgumentNullException.ThrowIfNull(saved);
        ArgumentNullException.ThrowIfNull(store);
        error = null;
        var result = new List<ImageLoraChoice>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var wasOn = new HashSet<string>(
            saved.Where(c => c is not null && !string.IsNullOrWhiteSpace(c.Id)).Select(c => c.Id), StringComparer.OrdinalIgnoreCase);
        CatalogLora? speed = null;
        foreach (ImageLoraChoice choice in requested)
        {
            if (choice is null || string.IsNullOrWhiteSpace(choice.Id) || !seen.Add(choice.Id))
                continue;
            if (LoraCatalog.Find(choice.Id) is not { } lora)
            {
                // A newer build's plug-in, sent back with the rest: kept by the loop below.
                if (wasOn.Contains(choice.Id))
                {
                    seen.Remove(choice.Id);
                    continue;
                }
                error = Loc.T("host.loras.unknown", ("id", choice.Id));
                return null;
            }
            if (!wasOn.Contains(lora.Id) && !store.IsInstalled(lora))
            {
                error = Loc.T("host.loras.notDownloaded", ("lora", lora.DisplayName));
                return null;
            }
            if (lora.Kind == LoraKind.Speed)
            {
                if (speed is not null)
                {
                    error = Loc.T("host.loras.twoSpeeds", ("first", speed.DisplayName), ("second", lora.DisplayName));
                    return null;
                }
                speed = lora;
            }
            if (!float.IsFinite(choice.Strength) && lora.StrengthAdjustable)
            {
                error = Loc.T("host.loras.strengthNotNumber", ("lora", lora.DisplayName));
                return null;
            }
            float strength = lora.StrengthAdjustable
                ? Math.Clamp(choice.Strength, LoraCatalog.MinStrength, LoraCatalog.MaxStrength)
                : lora.DefaultStrength;
            result.Add(new ImageLoraChoice(lora.Id, strength));
        }
        foreach (ImageLoraChoice kept in saved)
        {
            if (kept is not null && !string.IsNullOrWhiteSpace(kept.Id) && LoraCatalog.Find(kept.Id) is null && seen.Add(kept.Id))
                result.Add(kept);
        }
        return result;
    }

    /// <summary>
    /// What the saved choice asks the engine to apply to <paramref name="modelId"/>'s next picture.
    /// A plug-in for another model, or one this build does not know, is skipped, and so is one
    /// made only for edits when the picture is made from words (the sheet says so, and the
    /// picture's progress names what it is drawn with). A plug-in chosen for another image model
    /// sits out this one's pictures and the plan says so (<see cref="LoraPlan.SatOut"/>): a speed
    /// plug-in on a Qwen-Image 2.1 Turbo entry, which is step-distilled already, and any plug-in
    /// not validated on the entry, which <see cref="CatalogLora.AlsoFor"/> lists. A known plug-in
    /// that is no longer installed, or a hand-edited choice with two speed plug-ins, is an error
    /// rather than a silent omission: the user would otherwise get a picture without the plug-in
    /// they turned on, with nothing to say so.
    /// </summary>
    /// <param name="editing">Whether the picture edits an attached photo.</param>
    /// <returns>The plan, or null with <paramref name="error"/> set.</returns>
    public static LoraPlan? Plan(
        IReadOnlyList<ImageLoraChoice> choices, string? modelId, LoraStore store, bool editing, out string? error)
    {
        ArgumentNullException.ThrowIfNull(choices);
        ArgumentNullException.ThrowIfNull(store);
        error = null;
        var chosen = new List<(CatalogLora Lora, float Strength)>();
        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        CatalogModel? model = ModelCatalog.Find(modelId ?? string.Empty);
        var satOut = new List<string>();
        foreach (ImageLoraChoice choice in choices)
        {
            if (choice is null || !seen.Add(choice.Id ?? string.Empty))
                continue;
            if (LoraCatalog.Find(choice.Id!) is not { } lora)
                continue;
            if (!lora.AppliesTo(modelId))
            {
                // Another image model's plug-in: the picture is made without it, and says why.
                if (model is { IsImageGenerator: true })
                    satOut.Add(lora.Kind == LoraKind.Speed && model.ImageVariant == QwenImageVariant.Turbo
                        ? $"{lora.DisplayName} sits out: {model.DisplayName} is step-distilled already"
                        : $"{lora.DisplayName} sits out: not validated on {model.DisplayName}");
                continue;
            }
            if (lora.NeedsPhoto && !editing)
                continue;
            if (!store.IsInstalled(lora))
            {
                error = Loc.T("host.loras.filesMissing", ("lora", lora.DisplayName));
                return null;
            }
            chosen.Add((lora, choice.Strength));
        }

        CatalogLora[] speeds = chosen.Where(c => c.Lora.Kind == LoraKind.Speed).Select(c => c.Lora).ToArray();
        if (speeds.Length > 1)
        {
            error = Loc.T("host.loras.twoSpeedsSaved", ("first", speeds[0].DisplayName), ("second", speeds[1].DisplayName));
            return null;
        }
        // A task that works only on the model's own schedule keeps it: the speed plug-in sits
        // out this picture, which the page names in its progress and the host logs.
        if (speeds.Length == 1 && chosen.FirstOrDefault(c => c.Lora.NeedsModelSteps).Lora is { } task)
        {
            satOut.Add($"{speeds[0].DisplayName} sits out: {task.DisplayName} works only at the model's own steps");
            chosen.RemoveAll(c => c.Lora.Kind == LoraKind.Speed);
        }
        string? why = satOut.Count == 0 ? null : string.Join("; ", satOut);
        if (chosen.Count == 0)
            return why is null ? LoraPlan.None : LoraPlan.None with { SatOut = why };

        var ordered = chosen.Where(c => c.Lora.Kind == LoraKind.Speed).Concat(chosen.Where(c => c.Lora.Kind != LoraKind.Speed)).ToList();
        float StrengthOf((CatalogLora Lora, float Strength) c) => c.Lora.StrengthAdjustable && float.IsFinite(c.Strength)
            ? Math.Clamp(c.Strength, LoraCatalog.MinStrength, LoraCatalog.MaxStrength)
            : c.Lora.DefaultStrength;
        return new LoraPlan(
            ordered.Select(c => store.SpecFor(c.Lora, StrengthOf(c))).ToArray(),
            ordered.Select(c => c.Lora.DisplayName).ToArray(),
            why);
    }
}
