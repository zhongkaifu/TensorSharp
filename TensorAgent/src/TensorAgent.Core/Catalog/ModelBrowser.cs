// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Text;
using System.Text.RegularExpressions;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Catalog;

public enum ModelBrowseFilter { All, Compatible, Downloaded }

public sealed record ModelFamilyGroup(string Id, string DisplayName, IReadOnlyList<CatalogModel> Models);

/// <summary>Device-aware browsing and search over the catalog, without changing model or
/// companion identities. Compatibility here is the catalog's memory tier; callers must
/// still use their usual download and activation checks.</summary>
public static class ModelBrowser
{
    public static string FamilyId(CatalogModel model) => Family(model.Family).Id;
    public static string FamilyName(CatalogModel model) => Family(model.Family).Name;

    /// <summary>Groups matching entries under stable family IDs. Selected, downloaded and
    /// compatible entries lead within each family; family order stays alphabetical as the
    /// selection changes. Notes are deliberately not indexed:
    /// a note about an unsupported feature must not advertise that feature in search.</summary>
    public static IReadOnlyList<ModelFamilyGroup> Browse(
        IEnumerable<CatalogModel> models, string? query, ModelBrowseFilter filter,
        int deviceMemoryGB, IReadOnlySet<string> installedIds, string? selectedId)
        => Search(models, query, filter, deviceMemoryGB, installedIds, selectedId)
            .GroupBy(FamilyId, StringComparer.Ordinal)
            .Select(group => new ModelFamilyGroup(group.Key, FamilyName(group.First()), group.ToArray()))
            .OrderBy(group => group.DisplayName, Comparer<string>.Create(NaturalCompare)).ToArray();

    /// <summary>Flat results in relevance order, with selected, downloaded and compatible
    /// entries breaking ties. Exact model names lead descriptive or companion matches.</summary>
    public static IReadOnlyList<CatalogModel> Search(
        IEnumerable<CatalogModel> models, string? query, ModelBrowseFilter filter,
        int deviceMemoryGB, IReadOnlySet<string> installedIds, string? selectedId)
    {
        ArgumentNullException.ThrowIfNull(models);
        ArgumentNullException.ThrowIfNull(installedIds);
        // Keep a spaced quantization together, so the "m" in "q4 k m" cannot
        // accidentally match "memory" on a Q4_K_S entry.
        string prepared = Regex.Replace(query ?? string.Empty,
            @"\b((?:i?q|ptq|pq|mxfp)\d+)[\s_-]+(k[\s_-]+(?:m|s|l|xl)|xxs|xs|nl|m|s|0)\b",
            match => Compact(match.Value), RegexOptions.IgnoreCase | RegexOptions.CultureInvariant);
        string[] terms = prepared.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)
            .Select(Compact).Where(term => term.Length > 0).Distinct(StringComparer.Ordinal).ToArray();
        string phrase = Compact(query ?? string.Empty);

        var matches = new List<Match>();
        foreach (CatalogModel model in models)
        {
            bool installed = installedIds.Contains(model.Id);
            bool compatible = model.MinDeviceMemoryGB <= deviceMemoryGB;
            if (filter == ModelBrowseFilter.Compatible && !compatible
                || filter == ModelBrowseFilter.Downloaded && !installed)
                continue;
            int relevance = terms.Length == 0 ? 0 : Relevance(model, terms, phrase);
            if (relevance < 0)
                continue;
            matches.Add(new Match(model, relevance,
                string.Equals(model.Id, selectedId, StringComparison.OrdinalIgnoreCase), installed, compatible));
        }

        matches.Sort(Compare);
        return matches.Select(match => match.Model).ToArray();
    }

    private static (string Id, string Name) Family(CatalogFamily family) => family switch
    {
        CatalogFamily.Gemma4 => ("gemma", "Gemma"),
        CatalogFamily.Qwen35 or CatalogFamily.Qwen36 or CatalogFamily.Qwen38 or CatalogFamily.Qwen38FlashNext
            => ("qwen", "Qwen"),
        CatalogFamily.QwenImage => ("qwen-image", "Qwen Image"),
        CatalogFamily.GptOss => ("gpt-oss", "GPT-OSS"),
        CatalogFamily.Bonsai => ("bonsai", "Bonsai"),
        CatalogFamily.MuseGlimmer => ("muse-glimmer", "Muse-Glimmer"),
        CatalogFamily.MiniMaxH3 => ("minimax-h3", "MiniMax H3"),
        CatalogFamily.DeepSeek4 or CatalogFamily.DeepSeek41 => ("deepseek", "DeepSeek"),
        CatalogFamily.Glm5 => ("glm", "GLM"),
        CatalogFamily.Nemotron => ("nemotron", "Nemotron"),
        CatalogFamily.Mistral3 => ("mistral", "Mistral"),
        CatalogFamily.HunyuanDense => ("hunyuan", "Hunyuan"),
        CatalogFamily.DiffusionGemma => ("diffusiongemma", "DiffusionGemma"),
        CatalogFamily.Wan => ("wan", "Wan"),
        _ => (family.ToString().ToLowerInvariant(), family.ToString()),
    };

    private static int Relevance(CatalogModel model, string[] terms, string phrase)
    {
        string name = Compact(model.DisplayName);
        string id = Compact(model.Id);
        string[] identity = [name, id];
        string[] details = [.. identity, Compact(FamilyName(model)), Compact(model.Parameters),
            Compact(model.Quantization), Compact($"{model.MinDeviceMemoryGB} GB RAM memory")];
        string[] fields = [.. details, .. Capabilities(model).Select(Compact),
            .. model.Files.Where(file => file.Role != CatalogFileRole.Weights
                && file.Role != CatalogFileRole.WeightsShard).Select(file => Compact(file.FileName))];

        static bool ContainsEvery(string[] fields, string[] terms) =>
            terms.All(term => fields.Any(field => field.Contains(term, StringComparison.Ordinal)));

        if (!ContainsEvery(fields, terms)) return -1;
        if (name == phrase || id == phrase) return 0;
        if (name.StartsWith(phrase, StringComparison.Ordinal) || id.StartsWith(phrase, StringComparison.Ordinal)) return 1;
        if (ContainsEvery(identity, terms)) return 2;
        if (ContainsEvery(details, terms)) return 3;
        return 4;
    }

    private static IEnumerable<string> Capabilities(CatalogModel model)
    {
        yield return "text";
        if ((model.Modalities & (CatalogModalities.ImageOutput | CatalogModalities.VideoOutput | CatalogModalities.AudioOutput)) == 0)
            yield return "chat";
        yield return Loc.T("models.row.input.text");
        if (model.Kind == CatalogArchitectureKind.MixtureOfExperts)
        {
            yield return "moe mixture of experts";
            yield return Loc.T("models.row.mixtureOfExperts");
        }
        else if (model.Kind == CatalogArchitectureKind.Dense) yield return "dense";
        else if (model.Kind == CatalogArchitectureKind.Diffusion) yield return "diffusion";
        if (model.SupportsThinking)
        {
            yield return "thinking reasoning";
            yield return Loc.T("models.row.thinks");
        }
        if (model.Modalities.HasFlag(CatalogModalities.Image))
        {
            yield return "vision image images photo photos multimodal";
            yield return Loc.T("models.row.input.images");
            yield return Loc.T("models.action.addVision");
        }
        if (model.Modalities.HasFlag(CatalogModalities.Audio))
        {
            yield return "audio sound speech multimodal";
            yield return Loc.T("models.row.input.audio");
        }
        if (model.Modalities.HasFlag(CatalogModalities.Video))
        {
            yield return "video clips multimodal";
            yield return Loc.T("models.row.input.video");
        }
        if (model.Modalities.HasFlag(CatalogModalities.ImageOutput))
        {
            yield return "image generation pictures";
            yield return Loc.T("models.row.makes.pictures");
        }
        if (model.Modalities.HasFlag(CatalogModalities.VideoOutput))
        {
            yield return "video generation clips";
            yield return Loc.T("models.row.makes.clips");
        }
        if (model.Modalities.HasFlag(CatalogModalities.AudioOutput))
        {
            yield return "audio generation sound soundtrack";
            yield return Loc.T("models.row.makes.clipsWithSound");
        }
        if (model.Files.Any(file => file.Role is CatalogFileRole.Projector or CatalogFileRole.VisionProjector))
            yield return "mmproj projector";
        if (model.Files.Any(file => file.Role == CatalogFileRole.Draft))
        {
            yield return "draft speculative decoding";
            yield return Loc.T("models.action.loadDraft");
        }
    }

    // Ignore separators inside a search term, retaining decimals so a version query
    // "3.5" does not match "35B". FormD makes café/cafe equal.
    private static string Compact(string value)
    {
        var result = new StringBuilder(value.Length);
        string decomposed = value.Normalize(NormalizationForm.FormD);
        for (int i = 0; i < decomposed.Length; i++)
        {
            char character = decomposed[i];
            if (char.IsLetterOrDigit(character)) result.Append(char.ToLowerInvariant(character));
            else if (character is '.' or ',' && i > 0 && i + 1 < decomposed.Length
                && char.IsDigit(decomposed[i - 1]) && char.IsDigit(decomposed[i + 1])) result.Append('.');
        }
        return result.ToString();
    }

    private sealed record Match(CatalogModel Model, int Relevance, bool Selected, bool Installed, bool Compatible);

    private static int Compare(Match left, Match right)
    {
        int result = left.Relevance.CompareTo(right.Relevance);
        if (result == 0) result = right.Selected.CompareTo(left.Selected);
        if (result == 0) result = right.Installed.CompareTo(left.Installed);
        if (result == 0) result = right.Compatible.CompareTo(left.Compatible);
        if (result == 0) result = NaturalCompare(FamilyName(left.Model), FamilyName(right.Model));
        if (result == 0) result = NaturalCompare(left.Model.DisplayName, right.Model.DisplayName);
        if (result == 0) result = NaturalCompare(left.Model.Quantization, right.Model.Quantization);
        if (result == 0) result = string.Compare(left.Model.Id, right.Model.Id, StringComparison.Ordinal);
        return result;
    }

    private static int NaturalCompare(string left, string right)
    {
        int l = 0, r = 0;
        while (l < left.Length && r < right.Length)
        {
            if (char.IsAsciiDigit(left[l]) && char.IsAsciiDigit(right[r]))
            {
                int leftStart = l, rightStart = r;
                while (l < left.Length && char.IsAsciiDigit(left[l])) l++;
                while (r < right.Length && char.IsAsciiDigit(right[r])) r++;
                while (leftStart < l - 1 && left[leftStart] == '0') leftStart++;
                while (rightStart < r - 1 && right[rightStart] == '0') rightStart++;
                int length = (l - leftStart).CompareTo(r - rightStart);
                if (length != 0) return length;
                int digits = left.AsSpan(leftStart, l - leftStart).CompareTo(right.AsSpan(rightStart, r - rightStart),
                    StringComparison.Ordinal);
                if (digits != 0) return digits;
            }
            else
            {
                int character = char.ToUpperInvariant(left[l++]).CompareTo(char.ToUpperInvariant(right[r++]));
                if (character != 0) return character;
            }
        }
        return (left.Length - l).CompareTo(right.Length - r);
    }
}
