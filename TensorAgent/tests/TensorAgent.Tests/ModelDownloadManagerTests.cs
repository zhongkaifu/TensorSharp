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
using System.Security.Cryptography;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Catalog;
using TensorAgent.Core.Downloads;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// The one property a model download has to have on a phone: it belongs to the APP,
/// not to whatever screen or HTTP request happened to start it.
///
/// <para>
/// This is the regression these tests exist for. The transfer used to run on the
/// Models page's own cancellation source and, over HTTP, on the request's — so going
/// back to the chat, or letting the progress stream drop, killed a five-gigabyte
/// download in the middle. Nothing failed and nothing said so; the user came back to
/// a progress bar that had simply stopped. Everything below is written against a real
/// HTTP transfer through <see cref="RangeServer"/>, because the failure was about who
/// holds the socket.
/// </para>
/// </summary>
public sealed class ModelDownloadManagerTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-dlm-" + Guid.NewGuid().ToString("N"));
    private readonly List<IDisposable> _owned = new();

    public ModelDownloadManagerTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        foreach (IDisposable owned in _owned)
        {
            try { owned.Dispose(); } catch (Exception) { /* scratch */ }
        }
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    private T Own<T>(T value) where T : IDisposable { _owned.Add(value); return value; }

    private static byte[] Body(int size)
    {
        var bytes = new byte[size];
        new Random(7).NextBytes(bytes);
        return bytes;
    }

    private static CatalogModel Entry(string id, RangeServer server) => new()
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
    };

    private ModelStore Store(string name)
    {
        string root = Path.Combine(_root, name);
        Directory.CreateDirectory(root);
        return new ModelStore(root);
    }

    // ---- the manager ---------------------------------------------------------------

    [Fact]
    public async Task ADownloadRunsToTheEndWithNobodyWatchingIt()
    {
        byte[] body = Body(2 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 5 };
        ModelStore store = Store("a");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        downloads.Start(model);
        // Nothing subscribes, nothing awaits: exactly the situation the user creates
        // by tapping Download and immediately leaving the page.
        ModelDownloadStatus status = await WaitForEnd(downloads, model.Id);

        Assert.Equal(DownloadState.Completed, status.State);
        Assert.Null(status.Error);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
        Assert.Equal(body, await File.ReadAllBytesAsync(store.PathFor(model, model.Weights)));
    }

    [Fact]
    public async Task AWatcherThatWalksAwayDoesNotTakeTheDownloadWithIt()
    {
        byte[] body = Body(3 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 8 };
        ModelStore store = Store("b");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        downloads.Start(model);

        // Watch just long enough to see it running, then cancel the WATCH — which is
        // what a closed page does — and check the transfer finishes anyway.
        using (var watching = new CancellationTokenSource())
        {
            int seen = 0;
            await foreach (ModelDownloadStatus _ in downloads.WatchAsync(model.Id, watching.Token))
            {
                if (++seen >= 2)
                    break;
            }
            await watching.CancelAsync();
        }

        ModelDownloadStatus status = await WaitForEnd(downloads, model.Id);
        Assert.Equal(DownloadState.Completed, status.State);
        Assert.Equal(InstallState.Installed, store.StateOf(model));
    }

    [Fact]
    public async Task StartingTheSameModelTwiceJoinsTheRunningTransferRatherThanRacingIt()
    {
        byte[] body = Body(4 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 10 };
        ModelStore store = Store("c");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        downloads.Start(model);
        downloads.Start(model);
        downloads.Start(model);

        ModelDownloadStatus status = await WaitForEnd(downloads, model.Id);
        Assert.Equal(DownloadState.Completed, status.State);
        // One transfer, not three writing into the same .part.
        Assert.Equal(1, server.Requests);
        Assert.Equal(body, await File.ReadAllBytesAsync(store.PathFor(model, model.Weights)));
    }

    [Fact]
    public async Task CancellingKeepsWhatWasFetchedAndResumingFinishesIt()
    {
        byte[] body = Body(6 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 25 };
        ModelStore store = Store("d");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);
        string part = ResumableDownloader.PartPath(store.PathFor(model, model.Weights));

        downloads.Start(model);
        await WaitUntil(() => File.Exists(part) && new FileInfo(part).Length > 256 * 1024);
        Assert.True(downloads.Cancel(model.Id));

        ModelDownloadStatus stopped = await WaitForEnd(downloads, model.Id);
        Assert.Equal(DownloadState.Cancelled, stopped.State);
        long kept = new FileInfo(part).Length;
        Assert.True(kept > 0, "the part file was thrown away, so resuming would start from zero");

        downloads.Start(model);
        ModelDownloadStatus finished = await WaitForEnd(downloads, model.Id);
        Assert.Equal(DownloadState.Completed, finished.State);
        Assert.Equal(body, await File.ReadAllBytesAsync(store.PathFor(model, model.Weights)));
        // The second request asked for the rest, not the whole file again.
        Assert.Contains(server.RangeHeaders, header => header.StartsWith("bytes=", StringComparison.Ordinal));
    }

    [Fact]
    public async Task AFailedDownloadIsResumedAndACancelledOneIsLeftAlone()
    {
        byte[] body = Body(1024 * 1024);
        // One truncated response, and only one attempt, so the job ends Failed.
        using var server = new RangeServer(body) { CutAfterBytes = 200_000, CutResponses = 1 };
        ModelStore store = Store("e");
        using var downloads = new ModelDownloadManager(
            store, NullLogger.Instance);
        CatalogModel failing = Entry("m", server);

        // The store's downloader retries internally, so a single cut heals itself;
        // to observe a Failed job the URL has to be one that cannot work at all.
        CatalogModel broken = failing with
        {
            Id = "broken",
            Files = new[] { failing.Weights with { Url = "http://127.0.0.1:1/none.gguf" } },
        };
        downloads.Start(broken);
        ModelDownloadStatus failed = await WaitForEnd(downloads, broken.Id);
        Assert.Equal(DownloadState.Failed, failed.State);
        Assert.False(string.IsNullOrWhiteSpace(failed.Error));

        // Now point the same id at a URL that works, and let the foreground resume
        // path restart it — this is what happens when the app comes back after iOS
        // suspended it mid-transfer.
        CatalogModel fixedUp = broken with { Files = new[] { failing.Weights with { FileName = "broken.gguf" } } };
        IReadOnlyList<string> resumed = downloads.ResumeInterrupted(
            id => string.Equals(id, broken.Id, StringComparison.Ordinal) ? fixedUp : null);
        Assert.Equal(new[] { broken.Id }, resumed);
        Assert.Equal(DownloadState.Completed, (await WaitForEnd(downloads, broken.Id)).State);

        // A download the USER stopped is not restarted behind their back.
        CatalogModel other = Entry("stopped", new RangeServer(body));
        downloads.Start(other with { Files = new[] { other.Weights with { Url = "http://127.0.0.1:1/none.gguf" } } });
        downloads.Cancel(other.Id);
        await WaitForEnd(downloads, other.Id);
        Assert.DoesNotContain(other.Id, downloads.ResumeInterrupted(_ => other));
    }

    [Fact]
    public async Task AProjectorOnlyFailureResumesTheProjectorWhenGlobalOptionalsAreOff()
    {
        byte[] weights = Body(128 * 1024);
        byte[] projector = Body(512 * 1024);
        using var weightsServer = new RangeServer(weights);
        using var projectorServer = new RangeServer(projector);
        string storeRoot = Path.Combine(_root, "projector-resume");
        var store = new ModelStore(storeRoot, new ResumableDownloader(maxAttempts: 1));
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);

        CatalogModel textEntry = Entry("vision", weightsServer);
        CatalogFile projectorFile = new(
            CatalogFileRole.Projector, "mmproj.gguf", "http://127.0.0.1:1/mmproj.gguf",
            projector.Length, Convert.ToHexStringLower(SHA256.HashData(projector)), Optional: true);
        CatalogModel broken = textEntry with
        {
            Files = new[] { textEntry.Weights, projectorFile },
            Modalities = CatalogModalities.Image,
        };

        // The required weights are already usable. This is the state in which the
        // Models page exposes Add vision even though Download optional files is off.
        Directory.CreateDirectory(store.DirectoryFor(broken));
        await File.WriteAllBytesAsync(store.PathFor(broken, broken.Weights), weights);

        downloads.Start(broken, new[] { CatalogFileRole.Projector });
        ModelDownloadStatus failed = await WaitForEnd(downloads, broken.Id);
        Assert.Equal(DownloadState.Failed, failed.State);
        Assert.True(failed.RequestsOnly(CatalogFileRole.Projector));

        CatalogModel reachable = broken with
        {
            Files = new[] { broken.Weights, projectorFile with { Url = projectorServer.Url } },
        };

        // No optional-role argument is supplied here: foreground resume must remember
        // the explicit Add vision request, rather than re-reading the global setting
        // and completing immediately after noticing that the weights already exist.
        Assert.Equal(new[] { broken.Id }, downloads.ResumeInterrupted(
            id => string.Equals(id, broken.Id, StringComparison.Ordinal) ? reachable : null));

        ModelDownloadStatus finished = await WaitForEnd(downloads, broken.Id);
        Assert.Equal(DownloadState.Completed, finished.State);
        Assert.True(finished.RequestsOnly(CatalogFileRole.Projector));
        Assert.Equal(projector,
            await File.ReadAllBytesAsync(store.PathFor(reachable, reachable.Projector!)));
        Assert.Equal(0, weightsServer.Requests);
        Assert.Equal(1, projectorServer.Requests);
    }

    [Fact]
    public async Task TheBusySignalRisesOnTheFirstTransferAndFallsOnTheLast()
    {
        byte[] body = Body(1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 5 };
        ModelStore store = Store("f");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);

        var signals = new List<bool>();
        downloads.BusyChanged += busy => { lock (signals) signals.Add(busy); };

        CatalogModel first = Entry("one", server);
        CatalogModel second = Entry("two", server) with { Files = Entry("two", server).Files };
        downloads.Start(first);
        downloads.Start(second);
        await WaitForEnd(downloads, first.Id);
        await WaitForEnd(downloads, second.Id);
        await WaitUntil(() => !downloads.IsBusy);

        // Exactly one rise and one fall, whatever the interleaving: the iOS side holds
        // a background-task assertion between them, and a second Begin whose End is
        // lost is a termination rather than a warning.
        lock (signals)
        {
            Assert.Equal(new[] { true, false }, signals);
        }
    }

    [Theory]
    [InlineData(CatalogFileRole.Projector, CatalogFileRole.Draft)]
    [InlineData(CatalogFileRole.Draft, CatalogFileRole.Projector)]
    public async Task AnInstalledModelCanDownloadEachCompanionIndependently(
        CatalogFileRole requested, CatalogFileRole other)
    {
        byte[] weights = Body(128 * 1024);
        byte[] companion = Body(256 * 1024);
        using var weightsServer = new RangeServer(weights);
        using var companionServer = new RangeServer(companion);
        using var otherServer = new RangeServer(Body(512 * 1024));
        CatalogModel original = Entry("companions", weightsServer);
        CatalogFile chosen = new(requested, "chosen.gguf", companionServer.Url,
            companion.Length, Convert.ToHexStringLower(SHA256.HashData(companion)), Optional: true);
        CatalogFile unchosen = new(other, "unchosen.gguf", otherServer.Url,
            otherServer.Body.Length, Convert.ToHexStringLower(SHA256.HashData(otherServer.Body)), Optional: true);
        CatalogModel model = original with { Files = new[] { original.Weights, chosen, unchosen } };
        ModelStore store = Store("independent");
        Directory.CreateDirectory(store.DirectoryFor(model));
        await File.WriteAllBytesAsync(store.PathFor(model, model.Weights), weights);
        // The action's byte estimate includes an interrupted companion transfer.
        await File.WriteAllBytesAsync(ResumableDownloader.PartPath(store.PathFor(model, chosen)), companion[..1024]);
        Assert.Equal(companion.Length - 1024, store.RemainingBytes(model, new[] { requested }));
        Assert.Equal(InstallState.Installed, store.StateOf(model));

        using var downloads = new ModelDownloadManager(store);
        downloads.Start(model, new[] { requested });
        ModelDownloadStatus completed = await WaitForEnd(downloads, model.Id);

        Assert.Equal(DownloadState.Completed, completed.State);
        Assert.True(completed.RequestsOnly(requested));
        Assert.Equal(companion, await File.ReadAllBytesAsync(store.CompanionPath(model, requested)!));
        Assert.Null(store.CompanionPath(model, other));
        Assert.Equal(0, store.RemainingBytes(model, new[] { requested }));
        Assert.Equal(0, weightsServer.Requests);
        Assert.Equal(0, otherServer.Requests);
        Assert.Equal(1, companionServer.Requests);
        Assert.Contains("bytes=1024-", companionServer.RangeHeaders);
    }

    // ---- the route over it ----------------------------------------------------------

    [Fact]
    public async Task TheDownloadRouteStreamsProgressAndSurvivesTheReaderHangingUp()
    {
        byte[] body = Body(4 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 12 };
        ModelStore store = Store("g");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        using LoopbackServer loopback = Own(new LoopbackServer(NullLogger.Instance));
        loopback.MapAgent(
            new[] { model }, store,
            new ConversationStore(Path.Combine(_root, "g-conv")),
            new SettingsStore(Path.Combine(_root, "g-settings.json")),
            downloads: downloads);
        loopback.Start();

        using var client = new HttpClient { BaseAddress = new Uri(loopback.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={loopback.Token}");

        // Read a couple of frames, then drop the connection exactly as a page that
        // navigates away does.
        var frames = new List<JsonElement>();
        using (var request = new HttpRequestMessage(HttpMethod.Post, $"/api/agent/catalog/{model.Id}/download"))
        using (HttpResponseMessage response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead))
        {
            Assert.True(response.IsSuccessStatusCode);
            await using Stream stream = await response.Content.ReadAsStreamAsync();
            using var reader = new StreamReader(stream);
            while (await reader.ReadLineAsync() is { } line)
            {
                if (!line.StartsWith("data: ", StringComparison.Ordinal))
                    continue;
                frames.Add(JsonSerializer.Deserialize<JsonElement>(line[6..]));
                if (frames.Count >= 2)
                    break;
            }
        }

        Assert.NotEmpty(frames);
        Assert.True(frames[0].TryGetProperty("received", out _), "the first frame is not a progress frame");

        ModelDownloadStatus status = await WaitForEnd(downloads, model.Id);
        Assert.Equal(DownloadState.Completed, status.State);
        Assert.Equal(body, await File.ReadAllBytesAsync(store.PathFor(model, model.Weights)));

        // And the app can be asked what it did, which is how a reopened page finds the
        // job it stopped watching.
        JsonElement listed = JsonSerializer.Deserialize<JsonElement>(
            await client.GetStringAsync("/api/agent/downloads"));
        Assert.Equal("Completed", listed.GetProperty("downloads")[0].GetProperty("state").GetString());
    }

    [Fact]
    public async Task TheCancelRouteIsTheOnlyThingThatStopsADownload()
    {
        byte[] body = Body(8 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 30 };
        ModelStore store = Store("h");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        using LoopbackServer loopback = Own(new LoopbackServer(NullLogger.Instance));
        loopback.MapAgent(
            new[] { model }, store,
            new ConversationStore(Path.Combine(_root, "h-conv")),
            new SettingsStore(Path.Combine(_root, "h-settings.json")),
            downloads: downloads);
        loopback.Start();

        using var client = new HttpClient { BaseAddress = new Uri(loopback.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={loopback.Token}");

        downloads.Start(model);
        await WaitUntil(() => downloads.StatusOf(model.Id)?.Progress.BytesReceived > 0);

        JsonElement cancelled = JsonSerializer.Deserialize<JsonElement>(
            await (await client.PostAsync($"/api/agent/catalog/{model.Id}/download/cancel", null))
                .Content.ReadAsStringAsync());
        Assert.True(cancelled.GetProperty("cancelled").GetBoolean());
        Assert.Equal(DownloadState.Cancelled, (await WaitForEnd(downloads, model.Id)).State);
    }

    [Fact]
    public async Task TheCatalogSaysWhichEntryIsDownloadingRightNow()
    {
        byte[] body = Body(6 * 1024 * 1024);
        using var server = new RangeServer(body) { DelayPerChunkMs = 25 };
        ModelStore store = Store("i");
        using var downloads = new ModelDownloadManager(store, NullLogger.Instance);
        CatalogModel model = Entry("m", server);

        using LoopbackServer loopback = Own(new LoopbackServer(NullLogger.Instance));
        loopback.MapAgent(
            new[] { model }, store,
            new ConversationStore(Path.Combine(_root, "i-conv")),
            new SettingsStore(Path.Combine(_root, "i-settings.json")),
            downloads: downloads);
        loopback.Start();

        using var client = new HttpClient { BaseAddress = new Uri(loopback.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={loopback.Token}");

        // Before: no download, so nothing to say about one.
        JsonElement before = JsonSerializer.Deserialize<JsonElement>(await client.GetStringAsync("/api/agent/catalog"));
        Assert.Equal(JsonValueKind.Null, before.GetProperty("models")[0].GetProperty("download").ValueKind);

        downloads.Start(model);
        await WaitUntil(() => downloads.StatusOf(model.Id)?.Progress.BytesReceived > 0);

        JsonElement during = JsonSerializer.Deserialize<JsonElement>(await client.GetStringAsync("/api/agent/catalog"));
        JsonElement download = during.GetProperty("models")[0].GetProperty("download");
        Assert.True(download.GetProperty("running").GetBoolean());
        // "Partly downloaded" and "downloading right now" look identical on disk, and
        // only one of them means the user should wait rather than tap.
        Assert.Equal("Partial", during.GetProperty("models")[0].GetProperty("state").GetString());

        downloads.Cancel(model.Id);
        await WaitForEnd(downloads, model.Id);
    }

    // ---- waiting -----------------------------------------------------------------

    private static async Task<ModelDownloadStatus> WaitForEnd(ModelDownloadManager downloads, string id)
    {
        await WaitUntil(() => downloads.StatusOf(id) is { IsRunning: false });
        return downloads.StatusOf(id)!.Value;
    }

    private static async Task WaitUntil(Func<bool> condition, int seconds = 60)
    {
        DateTime deadline = DateTime.UtcNow.AddSeconds(seconds);
        while (DateTime.UtcNow < deadline)
        {
            if (condition())
                return;
            await Task.Delay(25);
        }
        Assert.Fail($"the condition was still false after {seconds}s");
    }
}
