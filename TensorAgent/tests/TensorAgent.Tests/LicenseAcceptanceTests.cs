// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Localization;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;
using TensorAgent.Sharing.Localization;

namespace TensorAgent.Tests;

/// <summary>
/// A catalog entry whose license must be accepted before it downloads (decision D9: MiniMax-H3-Image is listed,
/// and its terms are shown and accepted at download). The store of answers, the gate every download passes, a
/// license whose text changes, a decline, and the routes the desktop host serves. Every other entry downloads
/// as before.
/// </summary>
public sealed class LicenseAcceptanceTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-license-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _owned = new();

    public LicenseAcceptanceTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        foreach (IDisposable owned in _owned)
        {
            try { owned.Dispose(); } catch (Exception) { /* scratch */ }
        }
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    private T Own<T>(T value) where T : IDisposable { _owned.Add(value); return value; }

    private static CatalogLicenseTerms Terms => ModelCatalog.MiniMaxH3ImageTerms;

    private static CatalogModel H3Image => ModelCatalog.Find("minimax-h3-image-q4km")!;

    private LicenseAcceptanceStore Acceptances(string name = "license-acceptances.json", TimeProvider? clock = null) =>
        new(Path.Combine(_root, name), clock);

    /// <summary>A one-file entry served by <paramref name="server"/>, with or without license terms.</summary>
    private static CatalogModel Entry(string id, RangeServer server, CatalogLicenseTerms? terms) => new()
    {
        Id = id,
        DisplayName = id,
        Family = CatalogFamily.Gemma4,
        Kind = CatalogArchitectureKind.Dense,
        Parameters = "test",
        Quantization = "test",
        Files = new[]
        {
            new CatalogFile(CatalogFileRole.Weights, id + ".gguf", server.Url,
                server.Body.Length, Convert.ToHexStringLower(SHA256.HashData(server.Body))),
        },
        Modalities = CatalogModalities.Text,
        MinDeviceMemoryGB = 1,
        ContextLength = 128,
        KvCacheDtype = "f16",
        Sampling = new CatalogSampling(1, 1, 1, 0),
        License = "test",
        LicenseTerms = terms,
    };

    private static byte[] Body(int size)
    {
        var bytes = new byte[size];
        new Random(11).NextBytes(bytes);
        return bytes;
    }

    private ModelStore Store(string name, LicenseAcceptanceStore? licenses, ResumableDownloader? downloader = null)
    {
        string root = Path.Combine(_root, name);
        Directory.CreateDirectory(root);
        return new ModelStore(root, downloader) { Licenses = licenses };
    }

    private static async Task<ModelDownloadStatus> WaitForEnd(ModelDownloadManager downloads, string id)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (clock.Elapsed < TimeSpan.FromSeconds(60))
        {
            if (downloads.StatusOf(id) is { IsRunning: false } status)
                return status;
            await Task.Delay(20);
        }
        throw new TimeoutException($"{id} did not finish");
    }

    // ---- the terms ----------------------------------------------------------------------------------------

    [Fact]
    public void TheRevisionIsTheShaOfTheLicenseTextTheAppShows()
    {
        Assert.Equal("0ec3bb5e6d9e885fa2aaf15e84689e50fd51f42b53fa24d3a64ed9d8c76d423b", Terms.Revision);
        Assert.Equal(Terms.Revision, Terms.TextSha256());
        string text = Terms.ReadText();
        Assert.StartsWith("MiniMax-H3-Image COMMUNITY LICENSE AGREEMENT", text, StringComparison.Ordinal);
        Assert.Contains("“Excluded Territories” means the European Union, the United Kingdom, the Republic of Korea and the United States of America.",
            text, StringComparison.Ordinal);
        Assert.Contains("Annualized Revenue of you and/or your affiliates exceeds 1 Million USD, or, you provide Hosted Service(s)",
            text, StringComparison.Ordinal);
        Assert.Matches(@"^https://huggingface\.co/MiniMaxAI/MiniMax-H3-Image/blob/[0-9a-f]{40}/LICENSE$", Terms.Url);
    }

    /// <summary>The summary says what the terms exclude in every language (read from the tables, so the
    /// interface language of the other tests is never touched).</summary>
    [Fact]
    public void TheSummaryNamesTheTerritoryRevenueAndHostedServiceTermsInEveryLanguage()
    {
        string english = Loc.Tables.LoadFiles("en")["catalog"]["catalog.terms.minimaxH3Image.summary"];
        foreach (string fragment in new[]
                 {
                     "European Union", "United Kingdom", "Republic of Korea", "United States", "USD 1 million",
                     "hosted service", "MiniMax-H3-Image", "improve another AI model", "Hong Kong",
                 })
            Assert.Contains(fragment, english, StringComparison.Ordinal);

        IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> en = Loc.Tables.LoadFiles("en");
        foreach (UiLanguage language in UiLanguages.Supported)
        {
            IReadOnlyDictionary<string, IReadOnlyDictionary<string, string>> tables = Loc.Tables.LoadFiles(language.Tag);
            string summary = tables["catalog"]["catalog.terms.minimaxH3Image.summary"];
            Assert.Contains("MiniMax-H3-Image", summary, StringComparison.Ordinal);
            Assert.Contains("NOTICE", summary, StringComparison.Ordinal);
            Assert.Contains("1", summary, StringComparison.Ordinal);
            Assert.Equal(7, summary.Split('\n').Length);
            if (language.Tag == "en")
                continue;
            // Translated, not copied: the summary and the buttons read in the language.
            Assert.NotEqual(english, summary);
            foreach (string key in new[] { "models.license.accept", "models.license.decline", "models.license.intro" })
                Assert.NotEqual(en["models"][key], tables["models"][key]);
        }
    }

    [Fact]
    public void OnlyTheMiniMaxH3ImageEntriesAskAndNoOtherEntryChanges()
    {
        Assert.Equal(new[] { "minimax-h3-image-q4km", "minimax-h3-image-q8-0" },
            ModelCatalog.BuiltIn.Where(m => m.LicenseTerms is not null).Select(m => m.Id));
        // A store that keeps no answers at all still downloads every other entry.
        var store = new ModelStore(Path.Combine(_root, "plain"));
        foreach (CatalogModel model in ModelCatalog.BuiltIn.Where(m => m.LicenseTerms is null))
        {
            Assert.False(store.NeedsLicenseAcceptance(model), model.Id);
            store.EnsureLicenseAccepted(model);
        }
        Assert.True(store.NeedsLicenseAcceptance(H3Image));
    }

    // ---- the store of answers -----------------------------------------------------------------------------

    [Fact]
    public void AnAcceptanceIsOfOneEntryAndOneRevisionAndOutlivesTheProcess()
    {
        var clock = new FixedClock(new DateTimeOffset(2026, 11, 19, 12, 0, 0, TimeSpan.Zero));
        LicenseAcceptanceStore acceptances = Acceptances(clock: clock);
        CatalogModel q8 = ModelCatalog.Find("minimax-h3-image-q8-0")!;
        Assert.False(acceptances.IsAccepted(H3Image));

        LicenseAcceptance accepted = acceptances.Accept(H3Image, Terms.Revision);

        Assert.Equal(new LicenseAcceptance(H3Image.Id, Terms.Id, Terms.Revision, clock.GetUtcNow()), accepted);
        Assert.True(acceptances.IsAccepted(H3Image));
        // The other entry under the same license asks for itself.
        Assert.False(acceptances.IsAccepted(q8));
        // Read back by another instance, as the next launch does.
        LicenseAcceptanceStore relaunched = Acceptances();
        Assert.True(relaunched.IsAccepted(H3Image));
        Assert.Equal(new[] { accepted }, relaunched.All);
        // Accepting again replaces the record rather than adding one.
        relaunched.Accept(H3Image, Terms.Revision);
        Assert.Single(Acceptances().All);
    }

    [Fact]
    public void ANewRevisionOfTheTextAsksAgain()
    {
        LicenseAcceptanceStore acceptances = Acceptances();
        acceptances.Accept(H3Image, Terms.Revision);
        CatalogModel revised = H3Image with { LicenseTerms = Terms with { Revision = new string('a', 64) } };

        Assert.False(acceptances.IsAccepted(revised));
        Assert.True(acceptances.AcceptedAnEarlierRevision(revised));
        Assert.False(acceptances.AcceptedAnEarlierRevision(H3Image));
        // The old revision cannot be accepted for the new terms: nobody was shown those.
        var refused = Assert.Throws<InvalidOperationException>(() => acceptances.Accept(revised, Terms.Revision));
        Assert.Equal(Loc.T("host.license.changed", ("model", revised.DisplayName)), refused.Message);
        Assert.False(acceptances.IsAccepted(revised));

        acceptances.Accept(revised, revised.LicenseTerms!.Revision);
        Assert.True(acceptances.IsAccepted(revised));
        Assert.Equal(2, acceptances.All.Count);
    }

    [Fact]
    public void ADeclineRecordsNoAcceptanceAndWithdrawsOne()
    {
        LicenseAcceptanceStore acceptances = Acceptances();
        Assert.False(acceptances.Decline(H3Image));
        Assert.False(File.Exists(acceptances.Path));

        acceptances.Accept(H3Image, Terms.Revision);
        Assert.True(acceptances.Decline(H3Image));

        Assert.False(acceptances.IsAccepted(H3Image));
        Assert.False(Acceptances().IsAccepted(H3Image));
        Assert.Empty(Acceptances().All);
    }

    [Fact]
    public void AFileNobodyCanReadAcceptsNothing()
    {
        LicenseAcceptanceStore acceptances = Acceptances();
        File.WriteAllText(acceptances.Path, "{ not json");
        Assert.False(acceptances.IsAccepted(H3Image));
        acceptances.Accept(H3Image, Terms.Revision);
        Assert.True(Acceptances().IsAccepted(H3Image));
    }

    [Fact]
    public void AnEntryWithoutTermsHasNothingToAccept()
    {
        CatalogModel plain = ModelCatalog.Find("qwen-image-2.1-q4km")!;
        LicenseAcceptanceStore acceptances = Acceptances();
        Assert.True(acceptances.IsAccepted(plain));
        Assert.Throws<InvalidOperationException>(() => acceptances.Accept(plain, Terms.Revision));
        Assert.False(acceptances.Decline(plain));
    }

    // ---- the gate -----------------------------------------------------------------------------------------

    [Fact]
    public async Task NothingIsFetchedUntilTheTermsAreAccepted()
    {
        using var server = new RangeServer(Body(256 * 1024));
        LicenseAcceptanceStore acceptances = Acceptances();
        ModelStore store = Store("gate", acceptances);
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("gated", server, Terms);

        var refused = Assert.Throws<LicenseAcceptanceRequiredException>(() => downloads.Start(model));
        Assert.Same(model, refused.Model);
        Assert.Equal(Loc.T("host.license.required", ("model", "gated"), ("license", Terms.Title)), refused.Message);
        Assert.Null(downloads.StatusOf(model.Id));
        // The store refuses it too, for a caller that goes around the manager.
        await Assert.ThrowsAsync<LicenseAcceptanceRequiredException>(() => store.DownloadAsync(model, null, CancellationToken.None));
        Assert.Equal(0, server.Requests);
        Assert.False(Directory.Exists(store.DirectoryFor(model)));

        acceptances.Accept(model, Terms.Revision);
        downloads.Start(model);
        Assert.Equal(DownloadState.Completed, (await WaitForEnd(downloads, model.Id)).State);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
    }

    [Fact]
    public async Task AStoreThatKeepsNoAnswersNeverDownloadsAGatedEntryAndEveryOtherAsBefore()
    {
        using var server = new RangeServer(Body(128 * 1024));
        ModelStore store = Store("none", licenses: null);
        await Assert.ThrowsAsync<LicenseAcceptanceRequiredException>(
            () => store.DownloadAsync(Entry("gated", server, Terms), null, CancellationToken.None));
        Assert.Equal(0, server.Requests);

        CatalogModel plain = Entry("plain", server, terms: null);
        await store.DownloadAsync(plain, null, CancellationToken.None);
        Assert.Equal(InstallState.Installed, store.StateOf(plain));
    }

    [Fact]
    public async Task ANewRevisionStopsTheDownloadUntilItIsAccepted()
    {
        using var server = new RangeServer(Body(64 * 1024));
        LicenseAcceptanceStore acceptances = Acceptances();
        ModelStore store = Store("revised", acceptances);
        CatalogModel model = Entry("gated", server, Terms);
        acceptances.Accept(model, Terms.Revision);
        CatalogModel revised = model with { LicenseTerms = Terms with { Revision = new string('b', 64) } };

        Assert.True(store.NeedsLicenseAcceptance(revised));
        await Assert.ThrowsAsync<LicenseAcceptanceRequiredException>(() => store.DownloadAsync(revised, null, CancellationToken.None));
        Assert.Equal(0, server.Requests);
    }

    [Fact]
    public async Task ADeclinedDownloadIsNotResumedWhenTheAppComesBack()
    {
        using var server = new RangeServer(Body(512 * 1024)) { CutAfterBytes = 64 * 1024, CutResponses = 1 };
        LicenseAcceptanceStore acceptances = Acceptances();
        ModelStore store = Store("resume", acceptances, new ResumableDownloader(maxAttempts: 1));
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("gated", server, Terms);
        acceptances.Accept(model, Terms.Revision);

        downloads.Start(model);
        Assert.Equal(DownloadState.Failed, (await WaitForEnd(downloads, model.Id)).State);
        acceptances.Decline(model);

        Assert.Empty(downloads.ResumeInterrupted(id => id == model.Id ? model : null));

        acceptances.Accept(model, Terms.Revision);
        Assert.Equal(new[] { model.Id }, downloads.ResumeInterrupted(id => id == model.Id ? model : null));
        Assert.Equal(DownloadState.Completed, (await WaitForEnd(downloads, model.Id)).State);
    }

    // ---- a LoRA plug-in under its base model's license ------------------------------------------------------

    private static CatalogLora Pdmd => LoraCatalog.Find("minimax-h3-image-pdmd-2step")!;

    /// <summary>The pDMD plug-in with one file <paramref name="server"/> serves, and the given terms.</summary>
    private static CatalogLora Plugin(RangeServer server, CatalogLicenseTerms? terms) => Pdmd with
    {
        Id = "gated-plug-in",
        LicenseTerms = terms,
        Files = new[]
        {
            new LoraFile("weights.safetensors", server.Url, server.Body.Length, Convert.ToHexStringLower(SHA256.HashData(server.Body))),
        },
    };

    private LoraStore Loras(string name, LicenseAcceptanceStore? licenses) =>
        new(Path.Combine(_root, name)) { Licenses = licenses };

    /// <summary>The pDMD plug-in is a Model Derivative of MiniMax-H3-Image under its Community License: it carries the base
    /// entries' terms (the same text and revision), and no other plug-in carries any.</summary>
    [Fact]
    public void ThePdmdPlugInCarriesItsBaseEntriesTermsAndNoOtherPlugInAsks()
    {
        Assert.Same(Terms, Pdmd.LicenseTerms);
        Assert.Equal(new[] { "minimax-h3-image-q4km", "minimax-h3-image-q8-0" },
            LicenseAcceptanceStore.BaseEntries(Pdmd, ModelCatalog.BuiltIn).Select(m => m.Id).ToArray());
        Assert.All(LoraCatalog.BuiltIn.Where(l => l.Id != Pdmd.Id), l => Assert.Null(l.LicenseTerms));
        var store = new LoraStore(Path.Combine(_root, "plain"));
        Assert.All(LoraCatalog.BuiltIn.Where(l => l.Id != Pdmd.Id), l => Assert.False(store.NeedsLicenseAcceptance(l)));
    }

    /// <summary>Nothing of the plug-in is fetched until its base model's terms are accepted; an acceptance recorded for any
    /// entry it applies to (here the Q8_0 entry it is AlsoFor) covers it, as it covers the base model's own download.</summary>
    [Fact]
    public async Task APlugInUnderItsBaseModelsLicenseFetchesNothingUntilABaseEntryAcceptedThatRevision()
    {
        using var server = new RangeServer(Body(64 * 1024));
        LicenseAcceptanceStore acceptances = Acceptances();
        LoraStore store = Loras("lora-gate", acceptances);
        CatalogLora plugin = Plugin(server, Terms);

        Assert.True(store.NeedsLicenseAcceptance(plugin));
        var refused = await Assert.ThrowsAsync<LoraLicenseAcceptanceRequiredException>(
            () => store.DownloadAsync(plugin, null, CancellationToken.None));
        Assert.Same(plugin, refused.Lora);
        Assert.Equal("minimax-h3-image-q4km", refused.Model?.Id);
        Assert.Equal(Loc.T("host.license.loraRequired", ("lora", plugin.DisplayName), ("license", Terms.Title),
            ("model", refused.Model!.DisplayName)), refused.Message);
        Assert.Equal(0, server.Requests);
        Assert.False(Directory.Exists(store.DirectoryFor(plugin)));

        acceptances.Accept(ModelCatalog.Find("minimax-h3-image-q8-0")!, Terms.Revision);
        Assert.False(store.NeedsLicenseAcceptance(plugin));
        await store.DownloadAsync(plugin, null, CancellationToken.None);
        Assert.Equal(InstallState.Installed, store.StateOf(plugin));
    }

    /// <summary>A decline of the base entry's terms withdraws the cover, and an acceptance of another revision never gives it:
    /// a plug-in whose terms are not the ones a base entry accepted (a changed text) asks again.</summary>
    [Fact]
    public async Task ADeclineOrAnotherRevisionOfTheBaseTermsDoesNotCoverThePlugIn()
    {
        using var server = new RangeServer(Body(32 * 1024));
        LicenseAcceptanceStore acceptances = Acceptances();
        LoraStore store = Loras("lora-revision", acceptances);
        acceptances.Accept(H3Image, Terms.Revision);
        CatalogLora changed = Plugin(server, Terms with { Revision = new string('0', 64) });
        Assert.True(store.NeedsLicenseAcceptance(changed));
        await Assert.ThrowsAsync<LoraLicenseAcceptanceRequiredException>(() => store.DownloadAsync(changed, null, CancellationToken.None));
        Assert.Equal(0, server.Requests);

        CatalogLora plugin = Plugin(server, Terms);
        Assert.False(store.NeedsLicenseAcceptance(plugin));
        Assert.True(acceptances.Decline(H3Image));
        Assert.True(store.NeedsLicenseAcceptance(plugin));
        await Assert.ThrowsAsync<LoraLicenseAcceptanceRequiredException>(() => store.DownloadAsync(plugin, null, CancellationToken.None));
        Assert.Equal(0, server.Requests);
    }

    /// <summary>A store nobody gave the user's answers never downloads a gated plug-in (fail closed); a plug-in under its own
    /// license downloads as before.</summary>
    [Fact]
    public async Task AStoreThatKeepsNoAnswersNeverDownloadsAGatedPlugInAndEveryOtherAsBefore()
    {
        using var server = new RangeServer(Body(32 * 1024));
        LoraStore store = Loras("lora-none", licenses: null);
        await Assert.ThrowsAsync<LoraLicenseAcceptanceRequiredException>(
            () => store.DownloadAsync(Plugin(server, Terms), null, CancellationToken.None));
        Assert.Equal(0, server.Requests);
        CatalogLora open = Plugin(server, terms: null);
        await store.DownloadAsync(open, null, CancellationToken.None);
        Assert.Equal(InstallState.Installed, store.StateOf(open));
    }

    // ---- the desktop host's routes ------------------------------------------------------------------------

    private (HttpClient Client, ModelStore Store, ModelDownloadManager Downloads) Host(
        string name, CatalogModel[] catalog, LicenseAcceptanceStore acceptances)
    {
        ModelStore store = Store(name, acceptances);
        ModelDownloadManager downloads = Own(new ModelDownloadManager(store, NullLogger.Instance));
        LoopbackServer loopback = Own(new LoopbackServer(NullLogger.Instance));
        loopback.MapAgent(catalog, store,
            new ConversationStore(Path.Combine(_root, name + "-conv")),
            new SettingsStore(Path.Combine(_root, name + "-settings.json")),
            downloads: downloads);
        loopback.Start();
        HttpClient client = Own(new HttpClient { BaseAddress = new Uri(loopback.BaseUrl) });
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={loopback.Token}");
        return (client, store, downloads);
    }

    private static async Task<(HttpStatusCode Status, JsonElement Body)> Send(HttpClient client, HttpMethod method, string path, object? body = null)
    {
        using var request = new HttpRequestMessage(method, path);
        if (body is not null)
            request.Content = JsonContent.Create(body);
        using HttpResponseMessage response = await client.SendAsync(request);
        string text = await response.Content.ReadAsStringAsync();
        if (response.Content.Headers.ContentType?.MediaType == "text/event-stream")
        {
            string last = text.Split('\n').Last(l => l.StartsWith("data: ", StringComparison.Ordinal));
            return (response.StatusCode, JsonSerializer.Deserialize<JsonElement>(last[6..]));
        }
        return (response.StatusCode, JsonSerializer.Deserialize<JsonElement>(text));
    }

    [Fact]
    public async Task TheDownloadRouteShowsTheTermsAndStartsOnlyAfterAnAcceptanceOfTheirRevision()
    {
        using var server = new RangeServer(Body(256 * 1024));
        LicenseAcceptanceStore acceptances = Acceptances();
        CatalogModel model = Entry("gated", server, Terms);
        CatalogModel plain = Entry("plain", server, terms: null);
        (HttpClient client, ModelStore store, ModelDownloadManager downloads) = Host("route", new[] { model, plain }, acceptances);

        // Asked to download: the terms, not a download.
        (HttpStatusCode status, JsonElement body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/download");
        Assert.Equal(HttpStatusCode.Conflict, status);
        Assert.True(body.GetProperty("licenseRequired").GetBoolean());
        JsonElement license = body.GetProperty("license");
        Assert.Equal(Terms.Revision, license.GetProperty("revision").GetString());
        Assert.Equal(Terms.Summary, license.GetProperty("summary").GetString());
        Assert.Equal(Terms.Url, license.GetProperty("url").GetString());
        Assert.False(license.GetProperty("accepted").GetBoolean());
        Assert.Null(downloads.StatusOf(model.Id));

        // The full text is there to read.
        (status, body) = await Send(client, HttpMethod.Get, $"/api/agent/catalog/{model.Id}/license");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(Terms.ReadText(), body.GetProperty("text").GetString());

        // Declined: still nothing.
        (status, body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/license",
            new { revision = Terms.Revision, accept = false });
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.False(body.GetProperty("accepted").GetBoolean());
        (status, _) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/download");
        Assert.Equal(HttpStatusCode.Conflict, status);

        // An acceptance of another revision than the one the host has: refused, nothing recorded.
        (status, body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/license",
            new { revision = new string('c', 64), accept = true });
        Assert.Equal(HttpStatusCode.Conflict, status);
        Assert.Equal(Loc.T("host.license.changed", ("model", model.DisplayName)), body.GetProperty("error").GetString());
        Assert.False(acceptances.IsAccepted(model));
        Assert.Equal(0, server.Requests);

        // Accepted: recorded for that revision, and the download runs.
        (status, body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/license",
            new { revision = Terms.Revision, accept = true });
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.True(body.GetProperty("accepted").GetBoolean());
        Assert.True(acceptances.IsAccepted(model));
        (status, body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/download");
        Assert.Equal(HttpStatusCode.OK, status);
        Assert.Equal(DownloadState.Completed, (await WaitForEnd(downloads, model.Id)).State);
        Assert.Equal(InstallState.Installed, store.StateOf(model));

        (_, body) = await Send(client, HttpMethod.Get, $"/api/agent/catalog/{model.Id}");
        Assert.True(body.GetProperty("licenseTerms").GetProperty("accepted").GetBoolean());
        Assert.False(body.GetProperty("licenseTerms").GetProperty("changedSinceAccepted").GetBoolean());

        // An entry with no terms: none described, none to accept, its download as before.
        (_, body) = await Send(client, HttpMethod.Get, $"/api/agent/catalog/{plain.Id}");
        Assert.Equal(JsonValueKind.Null, body.GetProperty("licenseTerms").ValueKind);
        (status, _) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{plain.Id}/license", new { accept = true });
        Assert.Equal(HttpStatusCode.NotFound, status);
        (status, _) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{plain.Id}/download");
        Assert.Equal(HttpStatusCode.OK, status);
    }

    [Fact]
    public async Task ADeclineStopsTheRunningDownloadAndTheNextOneAsksAgain()
    {
        using var server = new RangeServer(Body(8 * 1024 * 1024)) { DelayPerChunkMs = 30 };
        LicenseAcceptanceStore acceptances = Acceptances();
        CatalogModel model = Entry("gated", server, Terms);
        (HttpClient client, _, ModelDownloadManager downloads) = Host("decline", new[] { model }, acceptances);
        acceptances.Accept(model, Terms.Revision);
        downloads.Start(model);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (downloads.StatusOf(model.Id)?.Progress.BytesReceived is not > 0 && clock.Elapsed < TimeSpan.FromSeconds(30))
            await Task.Delay(20);

        (HttpStatusCode status, JsonElement body) = await Send(client, HttpMethod.Post, $"/api/agent/catalog/{model.Id}/license",
            new { revision = Terms.Revision, accept = false });

        Assert.Equal(HttpStatusCode.OK, status);
        Assert.True(body.GetProperty("cancelled").GetBoolean());
        Assert.Equal(DownloadState.Cancelled, (await WaitForEnd(downloads, model.Id)).State);
        Assert.False(acceptances.IsAccepted(model));
        Assert.Throws<LicenseAcceptanceRequiredException>(() => downloads.Start(model));
    }

    [Fact]
    public async Task TheCatalogRouteDescribesTheTermsOfTheEntriesThatHaveThem()
    {
        LicenseAcceptanceStore acceptances = Acceptances();
        (HttpClient client, _, _) = Host("catalog", ModelCatalog.BuiltIn.ToArray(), acceptances);
        acceptances.Accept(H3Image, Terms.Revision);

        (_, JsonElement body) = await Send(client, HttpMethod.Get, "/api/agent/catalog");
        var described = body.GetProperty("models").EnumerateArray()
            .ToDictionary(m => m.GetProperty("id").GetString()!, m => m.GetProperty("licenseTerms"));
        Assert.Equal(new[] { "minimax-h3-image-q4km", "minimax-h3-image-q8-0" },
            described.Where(p => p.Value.ValueKind != JsonValueKind.Null).Select(p => p.Key).Order(StringComparer.Ordinal));
        Assert.True(described["minimax-h3-image-q4km"].GetProperty("accepted").GetBoolean());
        Assert.False(described["minimax-h3-image-q8-0"].GetProperty("accepted").GetBoolean());
        Assert.Equal(Terms.Title, described["minimax-h3-image-q8-0"].GetProperty("title").GetString());
    }

    /// <summary>
    /// The Models page asks before each of its download buttons starts a transfer (the native page cannot be
    /// driven here, so its source is held to it): every <c>Downloads.Start</c> there is preceded by the question.
    /// </summary>
    [Fact]
    public void TheModelsPageAsksBeforeEveryDownloadItStarts()
    {
        string page = File.ReadAllText(Path.Combine(RepoRoot(), "TensorAgent", "src", "TensorAgent.Maui", "Pages", "ModelsPage.cs"));
        string[] lines = page.Split('\n');
        int starts = 0;
        for (int i = 0; i < lines.Length; i++)
        {
            if (!lines[i].Contains("_app.Downloads.Start(", StringComparison.Ordinal))
                continue;
            starts++;
            // The question comes first, in the same handler: the closest earlier line that names either.
            int asked = Array.FindLastIndex(lines, i, l => l.Contains("LicenseAcceptedAsync(row.Model)", StringComparison.Ordinal));
            int handler = Array.FindLastIndex(lines, i, l => l.Contains("private ", StringComparison.Ordinal) && l.Contains('('));
            bool download = lines[handler].Contains("void Download(", StringComparison.Ordinal);
            Assert.True(download || asked > handler, $"ModelsPage.cs:{i + 1} starts a download its handler did not ask for");
        }
        Assert.Equal(4, starts);
        // And the ordinary Download button's handler asks before it calls Download(row).
        Assert.Matches(@"if \(!await LicenseAcceptedAsync\(row\.Model\)\)\s+return;\s+Download\(row\);", page);
    }

    private static string RepoRoot()
    {
        var directory = new DirectoryInfo(AppContext.BaseDirectory);
        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "TensorSharp.slnx")))
            directory = directory.Parent;
        return directory!.FullName;
    }

    private sealed class FixedClock(DateTimeOffset now) : TimeProvider
    {
        public override DateTimeOffset GetUtcNow() => now;
    }
}
