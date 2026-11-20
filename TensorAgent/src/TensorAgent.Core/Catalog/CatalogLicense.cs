// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorAgent.Core.Localization;

namespace TensorAgent.Core.Catalog;

/// <summary>
/// The terms a user has to accept before a catalog entry's first download (<see cref="CatalogModel.LicenseTerms"/>).
///
/// <para>
/// Most entries carry only a license name (<see cref="CatalogModel.License"/>), shown on the row. An entry
/// whose license limits who may use it where -- MiniMax-H3-Image's excludes the United States, the European
/// Union, the United Kingdom and the Republic of Korea, and asks for MiniMax's authorization past USD 1 million
/// of revenue or for a hosted service -- also declares these terms, and nothing of it downloads until they are
/// accepted (<see cref="LicenseAcceptanceStore"/>, checked by <see cref="ModelStore.DownloadAsync"/>).
/// </para>
/// <para>
/// The license text ships with the app (<see cref="TextResource"/>), verbatim, so the full terms can be read
/// before anything is fetched and the revision names exactly what was shown: it is the SHA-256 of that text.
/// A publisher that changes its license changes the text the catalog ships, so the revision changes and an
/// acceptance of the old one no longer counts.
/// </para>
/// </summary>
public sealed record CatalogLicenseTerms
{
    private const string ResourcePrefix = "TensorAgent.Core.Catalog.Licenses.";
    private readonly string _title = string.Empty;
    private readonly string _summary = string.Empty;

    /// <summary>Stable id of the license, the same across its revisions.</summary>
    public required string Id { get; init; }

    /// <summary>Lower-case SHA-256 of the license text (<see cref="TextResource"/>). An acceptance is of this
    /// revision only.</summary>
    public required string Revision { get; init; }

    /// <summary>The license's name, in the interface language (a catalog key, see <see cref="CatalogText"/>).</summary>
    public required string Title { get => CatalogText.Read(_title); init => _title = value; }

    /// <summary>What the terms say that a user most needs to know before downloading -- where it may be used,
    /// when MiniMax's own authorization is needed, the use restrictions -- in the interface language. A summary,
    /// never the terms themselves: those are the text.</summary>
    public required string Summary { get => CatalogText.Read(_summary); init => _summary = value; }

    /// <summary>The license at its publisher, at the commit its text was read from.</summary>
    public required string Url { get; init; }

    /// <summary>The file name of the verbatim license text embedded in this assembly
    /// (<c>Catalog/Licenses/</c>).</summary>
    public required string TextResource { get; init; }

    /// <summary>The license text, verbatim, in the language its publisher wrote it in.</summary>
    public string ReadText()
    {
        using Stream stream = OpenText();
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    }

    /// <summary>The SHA-256 of the embedded text: what <see cref="Revision"/> must say.</summary>
    public string TextSha256()
    {
        using Stream stream = OpenText();
        return Convert.ToHexStringLower(SHA256.HashData(stream));
    }

    private Stream OpenText() =>
        typeof(CatalogLicenseTerms).Assembly.GetManifestResourceStream(ResourcePrefix + TextResource)
        ?? throw new FileNotFoundException($"The license text {TextResource} is not embedded in this build.", TextResource);
}

/// <summary>One acceptance: who accepted which revision of an entry's license, and when.</summary>
public sealed record LicenseAcceptance(
    [property: JsonPropertyName("model")] string ModelId,
    [property: JsonPropertyName("license")] string LicenseId,
    [property: JsonPropertyName("revision")] string Revision,
    [property: JsonPropertyName("acceptedAt")] DateTimeOffset AcceptedAt);

/// <summary>A download of an entry whose license has not been accepted, refused before anything is fetched.</summary>
public sealed class LicenseAcceptanceRequiredException : InvalidOperationException
{
    public LicenseAcceptanceRequiredException(CatalogModel model)
        : base(Loc.T("host.license.required", ("model", model.DisplayName), ("license", model.LicenseTerms!.Title)))
    {
        Model = model;
    }

    /// <summary>The entry whose download was refused.</summary>
    public CatalogModel Model { get; }
}

/// <summary>A download of a LoRA plug-in under its base model's license, refused before anything is fetched: the base
/// entry's terms (<see cref="Model"/>, the first entry it applies to with them) have not been accepted.</summary>
public sealed class LoraLicenseAcceptanceRequiredException : InvalidOperationException
{
    public LoraLicenseAcceptanceRequiredException(CatalogLora lora, CatalogModel? model)
        : base(Loc.T("host.license.loraRequired", ("lora", lora.DisplayName), ("license", lora.LicenseTerms!.Title),
            ("model", model?.DisplayName ?? lora.BaseModelId)))
    {
        Lora = lora;
        Model = model;
    }

    /// <summary>The plug-in whose download was refused.</summary>
    public CatalogLora Lora { get; }

    /// <summary>The entry whose terms accepting covers it (null when the catalog holds none).</summary>
    public CatalogModel? Model { get; }
}

/// <summary>
/// The license acceptances this installation has recorded, one file in the app's data folder (not its cache:
/// what the user agreed to is not something a cache purge may forget).
///
/// <para>
/// Keyed by the catalog entry and the license revision (<see cref="CatalogLicenseTerms.Revision"/>). Accepting
/// one entry's license does not accept another's, even under the same license, because each download asks;
/// and a new revision of the text asks again. A decline removes what this entry's current revision had, so the
/// file always holds the user's last answer.
/// </para>
/// </summary>
public sealed class LicenseAcceptanceStore
{
    private static readonly JsonSerializerOptions Json = new() { WriteIndented = true };
    private readonly object _lock = new();
    private readonly TimeProvider _clock;

    public LicenseAcceptanceStore(string path, TimeProvider? clock = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        Path = System.IO.Path.GetFullPath(path);
        _clock = clock ?? TimeProvider.System;
    }

    /// <summary>The file the acceptances are kept in.</summary>
    public string Path { get; }

    /// <summary>Every acceptance on record, oldest first.</summary>
    public IReadOnlyList<LicenseAcceptance> All
    {
        get { lock (_lock) return Load(); }
    }

    /// <summary>Whether <paramref name="model"/> may download: it declares no terms, or its current revision was
    /// accepted for this entry.</summary>
    public bool IsAccepted(CatalogModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.LicenseTerms is not { } terms)
            return true;
        lock (_lock)
            return Load().Any(a => Matches(a, model) && string.Equals(a.Revision, terms.Revision, StringComparison.Ordinal));
    }

    /// <summary>
    /// Whether <paramref name="lora"/> may download: it declares no terms, or their current revision was accepted for
    /// one of the <paramref name="catalog"/> entries it applies to that carry the same terms (<see cref="CatalogLora.LicenseTerms"/>).
    /// </summary>
    public bool IsAccepted(CatalogLora lora, IEnumerable<CatalogModel> catalog)
    {
        ArgumentNullException.ThrowIfNull(lora);
        ArgumentNullException.ThrowIfNull(catalog);
        if (lora.LicenseTerms is not { } terms)
            return true;
        return BaseEntries(lora, catalog).Any(IsAccepted);
    }

    /// <summary>The entries <paramref name="lora"/> applies to whose terms are its own (same license, same revision): the
    /// ones whose acceptance covers it.</summary>
    public static IEnumerable<CatalogModel> BaseEntries(CatalogLora lora, IEnumerable<CatalogModel> catalog) =>
        lora.LicenseTerms is not { } terms ? Enumerable.Empty<CatalogModel>()
            : catalog.Where(m => lora.AppliesTo(m.Id) && m.LicenseTerms is { } t
                                 && string.Equals(t.Id, terms.Id, StringComparison.Ordinal)
                                 && string.Equals(t.Revision, terms.Revision, StringComparison.Ordinal));

    /// <summary>Whether this entry's license was accepted at some revision other than the current one: the
    /// terms have changed since, which is worth saying when they are shown again.</summary>
    public bool AcceptedAnEarlierRevision(CatalogModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.LicenseTerms is not { } terms)
            return false;
        lock (_lock)
            return Load().Any(a => Matches(a, model) && !string.Equals(a.Revision, terms.Revision, StringComparison.Ordinal));
    }

    /// <summary>
    /// Record that the user accepted <paramref name="revision"/> of <paramref name="model"/>'s license. The
    /// revision is the one the caller showed; one that is not the entry's current revision is refused (the text
    /// changed under the screen that asked), so an acceptance is never stored for terms nobody saw.
    /// </summary>
    public LicenseAcceptance Accept(CatalogModel model, string revision)
    {
        ArgumentNullException.ThrowIfNull(model);
        CatalogLicenseTerms terms = model.LicenseTerms
            ?? throw new InvalidOperationException(Loc.T("host.license.notRequired", ("model", model.DisplayName)));
        if (!string.Equals(revision, terms.Revision, StringComparison.Ordinal))
            throw new InvalidOperationException(Loc.T("host.license.changed", ("model", model.DisplayName)));
        lock (_lock)
        {
            List<LicenseAcceptance> all = Load();
            var accepted = new LicenseAcceptance(model.Id, terms.Id, terms.Revision, _clock.GetUtcNow());
            all.RemoveAll(a => Matches(a, model) && string.Equals(a.Revision, terms.Revision, StringComparison.Ordinal));
            all.Add(accepted);
            Save(all);
            return accepted;
        }
    }

    /// <summary>Record that the user declined <paramref name="model"/>'s current terms: an acceptance of them,
    /// if one was on record, is withdrawn, and the next download asks again.</summary>
    /// <returns>Whether an acceptance was withdrawn.</returns>
    public bool Decline(CatalogModel model)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.LicenseTerms is not { } terms)
            return false;
        lock (_lock)
        {
            List<LicenseAcceptance> all = Load();
            int removed = all.RemoveAll(a => Matches(a, model) && string.Equals(a.Revision, terms.Revision, StringComparison.Ordinal));
            if (removed > 0)
                Save(all);
            return removed > 0;
        }
    }

    private static bool Matches(LicenseAcceptance acceptance, CatalogModel model) =>
        string.Equals(acceptance.ModelId, model.Id, StringComparison.OrdinalIgnoreCase)
        && string.Equals(acceptance.LicenseId, model.LicenseTerms!.Id, StringComparison.Ordinal);

    private List<LicenseAcceptance> Load()
    {
        if (!File.Exists(Path))
            return new List<LicenseAcceptance>();
        try
        {
            using FileStream stream = File.OpenRead(Path);
            return JsonSerializer.Deserialize<Record>(stream, Json)?.Acceptances?.Where(a => a is not null).ToList()
                ?? new List<LicenseAcceptance>();
        }
        catch (JsonException)
        {
            // A file nobody can read accepts nothing: every gated download asks again, which is the safe way
            // to be wrong. The next acceptance writes a good file over it.
            return new List<LicenseAcceptance>();
        }
    }

    private void Save(List<LicenseAcceptance> all)
    {
        string? directory = System.IO.Path.GetDirectoryName(Path);
        if (!string.IsNullOrEmpty(directory))
            Directory.CreateDirectory(directory);
        string tmp = Path + ".tmp";
        using (FileStream stream = File.Create(tmp))
            JsonSerializer.Serialize(stream, new Record { Acceptances = all }, Json);
        File.Move(tmp, Path, overwrite: true);
    }

    private sealed class Record
    {
        [JsonPropertyName("acceptances")]
        public List<LicenseAcceptance>? Acceptances { get; set; }
    }
}
