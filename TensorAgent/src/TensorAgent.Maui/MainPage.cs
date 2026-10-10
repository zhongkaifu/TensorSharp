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
#if IOS
using Foundation;
#endif
using TensorAgent.Core.Localization;
using TensorAgent.Maui.Hosting;
#if IOS
using UserNotifications;
#endif

namespace TensorAgent.Maui;

/// <summary>
/// The chat page: a WebView filling the page with the loopback-served Web UI,
/// under a one-line native bar that shows which backend was picked and whether
/// the native engine link is alive.
/// </summary>
public sealed class MainPage : ContentPage
{
    private static readonly Color BarBackground = Color.FromArgb("#0b1220");
    private static readonly Color BarText = Color.FromArgb("#c9d3ea");
    private static readonly Color BarError = Color.FromArgb("#ff8a8a");

    private readonly LoopbackWebHost _host;
    private readonly Label _status;
    private readonly WebView _webView;
    private Services.Dictation? _dictation;

    /// <summary>
    /// True once the page has told us it finished loading, and false again from the
    /// moment it is reloaded. It is what makes "the page did not answer" mean something:
    /// before the first `ready`, silence is normal.
    /// </summary>
    private bool _pageReady;
    private bool _chatVisible;
    private bool _languageReloadPending;
    private bool _languageReloadRetryScheduled;
#if IOS
    private int _shareNotificationPrompting;
#endif

    /// <summary>What the engine probe found once the host started; null before then.</summary>
    private EngineProbeResult? _probe;

    /// <summary>Why the host did not start, or null while nothing has failed.</summary>
    private string? _startupFailure;

    public MainPage(LoopbackWebHost host)
    {
        _host = host;
        Title = "TensorAgent";
        BackgroundColor = BarBackground;
        Shell.SetNavBarIsVisible(this, false);
        // Keep the status bar and the home indicator out of the WebView; the
        // page's own background paints the insets in the Web UI's dark colour.
        SafeAreaEdges = Microsoft.Maui.SafeAreaEdges.All;

        _status = new Label
        {
            FontSize = 12,
            TextColor = BarText,
            BackgroundColor = BarBackground,
            LineBreakMode = LineBreakMode.TailTruncation,
            VerticalOptions = LayoutOptions.Center,
        };
        ShowStatus();

        _webView = new WebView
        {
            BackgroundColor = BarBackground,
            HorizontalOptions = LayoutOptions.Fill,
            VerticalOptions = LayoutOptions.Fill,
        };

        var grid = new Grid
        {
            BackgroundColor = BarBackground,
            RowDefinitions =
            {
                new RowDefinition(GridLength.Auto),
                new RowDefinition(GridLength.Star),
                new RowDefinition(GridLength.Auto),
            },
        };
        // No native row above the page. It used to carry the status and chips for
        // Chats / Models / Settings, which is a second row of chrome stacked on the
        // page's own -- the one thing a small screen cannot afford. The page has a
        // menu that asks for the same routes through OnPageEvent.
        grid.Add(new ContentView { IsVisible = false, HeightRequest = 0 }, 0, 0);
        grid.Add(_webView, 0, 1);
        grid.Add(BuildAttachmentBar(), 0, 2);
        Content = grid;

        // A file the model's code produced is reached by a link the page renders.
        // A WebView cannot save one: the route serves it as an attachment (it is
        // program-written content and must never render inline), and a WKWebView
        // with no download delegate simply does nothing when you tap it. Hand it to
        // the share sheet instead, which is what "save it to my phone" actually
        // means -- Save to Files, Mail, AirDrop.
        _webView.Navigating += OnNavigatingShareArtifact;
#if DEBUG
        _webView.Navigated += OnNavigatedSendDemoPrompt;
#endif
        // A covered WebView may be suspended. Reload when it is visible again,
        // after giving the page a chance to retain its unsent composer.
        Loc.Changed += () => MainThread.BeginInvokeOnMainThread(async () =>
        {
            ShowStatus();
            _languageReloadPending = true;
            if (_chatVisible)
                await ReloadLanguageAsync();
        });
        StartHost();
    }

    /// <summary>The route every generated file is reached through.</summary>
    private const string ArtifactPrefix = "/api/code/artifacts/";

    /// <summary>
    /// Intercept a tap on a generated file and open it rather than navigating.
    ///
    /// <para>
    /// The BACKSTOP, not the main path. The page claims these clicks itself and asks
    /// through <c>open-file</c> (see <see cref="OnPageEvent"/>), because the links carry
    /// <c>target="_blank"</c> and WebKit routes those to its create-web-view delegate
    /// rather than to this one -- which is exactly how a tap ended up navigating to the
    /// route and rendering its 404 body. This still catches a same-frame navigation to
    /// an artifact URL from anywhere else in the page.
    /// </para>
    /// </summary>
    private async void OnNavigatingShareArtifact(object? sender, WebNavigatingEventArgs e)
    {
        if (!Uri.TryCreate(e.Url, UriKind.Absolute, out Uri? uri) ||
            !uri.AbsolutePath.StartsWith(ArtifactPrefix, StringComparison.Ordinal))
        {
            return;
        }

        // A run listing rather than a file: JSON the page can render, so let it navigate.
        string rest = Uri.UnescapeDataString(uri.AbsolutePath[ArtifactPrefix.Length..]);
        int slash = rest.IndexOf('/');
        if (slash <= 0 || slash == rest.Length - 1)
            return;

        e.Cancel = true;
        await OpenArtifactAsync(uri.AbsolutePath, null);
    }

    /// <summary>
    /// The path part of what the page sent, whether it sent a path or a whole URL. The
    /// page sends <c>location.pathname</c>, but a query string rides along and an
    /// absolute spelling costs nothing to accept.
    /// </summary>
    private static string PathOf(string urlOrPath)
    {
        if (Uri.TryCreate(urlOrPath, UriKind.Absolute, out Uri? absolute))
            return absolute.AbsolutePath;
        int query = urlOrPath.IndexOf('?', StringComparison.Ordinal);
        return query >= 0 ? urlOrPath[..query] : urlOrPath;
    }

    /// <summary>
    /// Open one file the model's code produced, named by its own download URL.
    ///
    /// <para>
    /// Resolved through <see cref="TensorSharp.AgentHost.CodeExec.CodeArtifactStore"/>
    /// rather than by fetching the URL: the confinement check -- the path segment was
    /// chosen by a program a model wrote -- stays in the one place that owns it, and a
    /// file already on disk costs no loopback round trip. The URL is the page's, so its
    /// segments are percent-encoded and have to be decoded one at a time; decoding the
    /// whole path at once would turn an encoded separator inside a file name into a
    /// directory boundary.
    /// </para>
    /// </summary>
    private async Task OpenArtifactAsync(string absolutePath, string? displayName)
    {
        if (!absolutePath.StartsWith(ArtifactPrefix, StringComparison.Ordinal))
            return;

        string[] segments = absolutePath[ArtifactPrefix.Length..]
            .Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length < 2)
            return;

        string runId = Uri.UnescapeDataString(segments[0]);
        string relative = string.Join('/', segments.Skip(1).Select(Uri.UnescapeDataString));

        if (!_host.App.Artifacts.TryResolve(runId, relative, out string? full, out string? error))
        {
            await DisplayAlert(Loc.T("app.openFile.alert.title"), error ?? Loc.T("app.openFile.missing"), Loc.T("common.ok"));
            return;
        }

        string? failure = await Services.FilePresenter.PresentAsync(
            full!, string.IsNullOrWhiteSpace(displayName) ? Path.GetFileName(relative) : displayName);
        if (failure is not null)
            await DisplayAlert(Loc.T("app.openFile.alert.title"), failure, Loc.T("common.ok"));
    }

    /// <summary>Save the full-resolution image already shown in the conversation.</summary>
    private async Task SaveImageAsync(string url)
    {
        string? failure;
        if (!Core.Hosting.ImageDownload.TryResolve(_host.App.Paths.UploadsDirectory, url, _host.EntryUrl, out string? full))
            failure = Loc.T("app.openFile.missing");
        else
            failure = await Services.FilePresenter.SaveAsync(full!);

        if (failure is not null)
            await DisplayAlert(Loc.T("app.openFile.alert.title"), failure, Loc.T("common.ok"));
    }

#if DEBUG
    /// <summary>
    /// Device E2E hook (Debug builds only): start downloading the catalog entry named
    /// by TENSORAGENT_DOWNLOAD and log what the transfer does, then stop it after
    /// TENSORAGENT_DOWNLOAD_SECONDS (60 by default).
    ///
    /// <para>
    /// The half of a download that only a real device can answer is what happens when
    /// the user leaves the app: <see cref="Platforms.iOS.BackgroundDownloads"/> takes a
    /// background-task assertion, and an assertion begun twice or never ended is a
    /// termination rather than a warning. Neither a simulator nor a unit test is ever
    /// suspended, so the only way to see it is to start a transfer on the phone,
    /// foreground something else, and read this log. It is bounded on purpose — the
    /// point is the first megabytes and the state changes around them, not five
    /// gigabytes of someone's data allowance.
    /// </para>
    /// </summary>
    private void DownloadIfAsked()
    {
        string? id = Environment.GetEnvironmentVariable("TENSORAGENT_DOWNLOAD");
        if (string.IsNullOrWhiteSpace(id))
            return;
        if (Core.Catalog.ModelCatalog.Find(id) is not { } model)
        {
            Console.WriteLine($"TensorAgent: download FAIL no catalog entry called '{id}'");
            return;
        }

        int seconds = int.TryParse(Environment.GetEnvironmentVariable("TENSORAGENT_DOWNLOAD_SECONDS"), out int s) ? s : 60;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        long lastReported = -1;
        _host.App.Downloads.BusyChanged += busy =>
            Console.WriteLine($"TensorAgent: download busy={busy} at {clock.Elapsed.TotalSeconds:0.0}s");
        _host.App.Downloads.Changed += status =>
        {
            if (!string.Equals(status.ModelId, model.Id, StringComparison.Ordinal))
                return;
            // One line a megabyte: enough to see it moving while the app is in the
            // background, few enough to read.
            long megabytes = status.Progress.BytesReceived / (1024 * 1024);
            if (status.IsRunning && megabytes == lastReported)
                return;
            lastReported = megabytes;
            Console.WriteLine(
                $"TensorAgent: download {status.State} {megabytes} MB of "
                + $"{status.Progress.TotalBytes / (1024 * 1024)} MB at {clock.Elapsed.TotalSeconds:0.0}s"
                + (status.Error is { Length: > 0 } error ? " · " + error : string.Empty));
        };

        _host.App.Downloads.Start(model, Core.Downloads.ModelDownloadManager.OptionalRolesFor(false));
        Console.WriteLine($"TensorAgent: download started {model.Id}, stopping after {seconds}s");
        _ = Task.Run(async () =>
        {
            await Task.Delay(TimeSpan.FromSeconds(seconds));
            Console.WriteLine("TensorAgent: download cancelling after the budget");
            _host.App.Downloads.Cancel(model.Id);
        });
    }

    /// <summary>
    /// Device E2E hook (Debug builds only, on by default): post a large multipart body
    /// to this app's own <c>/api/upload</c> and report whether the file came out the
    /// other end.
    ///
    /// <para>
    /// It is here rather than in the unit suite because the failure it guards was
    /// invisible everywhere else. The hand-written multipart parser dropped a whole
    /// buffer when a single read returned exactly its 65536 bytes, and which read sizes
    /// a body arrives in is decided by the HTTP client and the socket: the managed
    /// listener hands back the leftover of its own 8 kB header read first, so a browser's
    /// <c>fetch</c> upload never reached the fault, while iOS's NSURLSession — which
    /// writes the whole body as one <c>NSData</c> — reached it every time. Every photo
    /// picked on the phone was answered "no file was uploaded"; nothing on a development
    /// machine reproduced it. So the probe runs the real client, on the real device,
    /// against the real route, and says so in the launch log.
    /// </para>
    /// </summary>
    private async Task RunUploadProbeAsync()
    {
        if (string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_SKIP_UPLOAD_CHECK"), "1", StringComparison.Ordinal))
            return;
        try
        {
            // Comfortably past the parser's 65536-byte buffer, so at least one read has
            // to fill it completely.
            byte[] png = SolidPng(700, 700);
            using var form = new MultipartFormDataContent();
            using var part = new ByteArrayContent(png);
            part.Headers.ContentType = new System.Net.Http.Headers.MediaTypeHeaderValue("image/png");
            // No extension, exactly as the photo picker hands one over.
            form.Add(part, "file", "IMG_PROBE");

            using var client = new HttpClient { BaseAddress = new Uri(_host.BaseUrl) };
            client.DefaultRequestHeaders.Add("Cookie", $"tensoragent_token={_host.Token}");
            using HttpResponseMessage response = await client.PostAsync("/api/upload", form);
            string payload = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
            {
                Console.WriteLine($"TensorAgent: uploadcheck FAIL {png.Length} bytes refused: {payload}");
                return;
            }

            using JsonDocument answer = JsonDocument.Parse(payload);
            string? file = answer.RootElement.TryGetProperty("file", out JsonElement f) ? f.GetString() : null;
            string? media = answer.RootElement.TryGetProperty("mediaType", out JsonElement m) ? m.GetString() : null;
            Console.WriteLine($"TensorAgent: uploadcheck ok {png.Length} bytes -> {file} ({media})");
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: uploadcheck FAIL " + ex.Message);
        }
    }

    /// <summary>
    /// Simulator/device E2E hook for the complete share handoff. It writes the same
    /// durable envelope as the extension, asks the host to import it through the real
    /// upload service, then waits for the real WebView to merge its text and attachment
    /// while the envelope remains durably claimed. It finishes through the authenticated
    /// explicit-discard route and verifies that both copies of the file are reclaimed.
    /// Nothing is sent to a model.
    /// </summary>
    private async Task RunShareProbeAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_SHARE_CHECK"), "1", StringComparison.Ordinal))
            return;

        TensorAgent.Sharing.ShareEnvelopeWriter? writer = null;
        var clock = System.Diagnostics.Stopwatch.StartNew();
        try
        {
            TensorAgent.Sharing.ShareEnvelopeStore? store = _host.App.ShareInbox;
            if (store is null)
            {
                Console.WriteLine("TensorAgent: sharecheck FAIL no App Group inbox");
                return;
            }

            const string marker = "tensoragent-share-e2e-marker";
            writer = store.BeginWrite();
            (string absolute, string relative) = writer.ReserveFile("sharecheck.png");
            byte[] png = SolidPng(64, 64);
            await File.WriteAllBytesAsync(absolute, png);

            string id = writer.Commit(new TensorAgent.Sharing.SharePayload
            {
                SourceApp = "Share E2E probe",
                Prompt = "What can you tell me about this?",
                NewChat = true,
                AutoSend = false,
                Items =
                {
                    TensorAgent.Sharing.ShareItem.ForPage(
                        "https://example.com/tensoragent-share-check",
                        "TensorAgent share check",
                        marker + "\nSecond line preserved."),
                    TensorAgent.Sharing.ShareItem.ForFile(
                        relative, "sharecheck.png", png.LongLength, "public.png", "image/png"),
                },
            });
            writer = null; // Commit transferred ownership to the durable inbox.
            Console.WriteLine($"TensorAgent: sharecheck wrote {id} ({png.Length} byte image)");

            int imported = await _host.App.DrainSharedInboxAsync();
            Console.WriteLine($"TensorAgent: sharecheck host imported {imported}");

            // Keep the exact upload family, not merely its browser attachment count.
            // A successful discard has two independently durable things to reclaim:
            // the App Group envelope and the copy the ordinary upload service staged.
            TensorAgent.Core.Sharing.PendingShare? pending = _host.App.Shares.Peek();
            string? stagedName = null;
            if (pending?.Attachments.Count == 1)
            {
                JsonElement attachment = JsonSerializer.SerializeToElement(pending.Attachments[0]);
                if (attachment.ValueKind == JsonValueKind.Object
                    && attachment.TryGetProperty("file", out JsonElement file)
                    && file.ValueKind == JsonValueKind.String)
                {
                    string? candidate = file.GetString();
                    if (candidate is { Length: > 0 }
                        && string.Equals(candidate, Path.GetFileName(candidate), StringComparison.Ordinal))
                    {
                        stagedName = candidate;
                    }
                }
            }

            string? stagedStem = stagedName is null ? null : Path.GetFileNameWithoutExtension(stagedName);
            string[] stagedFiles = stagedStem is null
                ? Array.Empty<string>()
                : Directory.GetFiles(_host.App.Options.UploadDirectory, stagedStem + "*");
            bool stagedBeforeDiscard = stagedName is not null
                && File.Exists(Path.Combine(_host.App.Options.UploadDirectory, stagedName))
                && stagedFiles.Length > 0;

            bool composerApplied = false;
            bool retained = false;
            string claimed = Path.Combine(store.Root, id + ".claimed");
            string ready = Path.Combine(store.Root, id);
            string acknowledged = Path.Combine(store.Root, id + ".acknowledged");
            for (int attempt = 0; attempt < 120; attempt++)
            {
                await Task.Delay(250);
                if (_pageReady)
                {
                    string? answer = await _webView.EvaluateJavaScriptAsync(
                        "(function(){var t=document.getElementById('text');"
                        + "return !!t&&t.value.indexOf('tensoragent-share-e2e-marker')>=0"
                        + "&&t.value.indexOf('What can you tell me about this?')>=0"
                        + "&&window.TensorAgent.attachmentCount()===1;})()");
                    composerApplied = answer?.Contains("true", StringComparison.OrdinalIgnoreCase) == true;
                }

                retained = _host.App.Shares.PendingCount == 1
                    && Directory.Exists(claimed)
                    && !Directory.Exists(ready);
                if (composerApplied && retained)
                    break;
            }

            clock.Stop();
            bool passed = composerApplied && retained && stagedBeforeDiscard;
            Console.WriteLine(passed
                ? $"TensorAgent: sharecheck PASS composer text + image + durable draft retained in {clock.Elapsed.TotalMilliseconds:0} ms"
                : $"TensorAgent: sharecheck FAIL composer={composerApplied} retained={retained} staged={stagedBeforeDiscard} pending={_host.App.Shares.PendingCount} after {clock.Elapsed.TotalSeconds:0.0}s");
            // The probe deliberately never sends a model turn. Calling the explicit
            // route directly avoids depending on a second WKWebView evaluation after
            // the composer check (which can remain pending while WebKit processes the
            // synthetic click), while still crossing the same authenticated HTTP and
            // host cleanup path as the visible shared-item chip.
            if (passed)
            {
                bool discardAccepted = false;
                string discardDetail = string.Empty;
                try
                {
                    using var client = new HttpClient
                    {
                        BaseAddress = new Uri(_host.BaseUrl),
                        Timeout = TimeSpan.FromSeconds(5),
                    };
                    client.DefaultRequestHeaders.Add("Cookie", $"tensoragent_token={_host.Token}");
                    using var body = new StringContent(
                        JsonSerializer.Serialize(new { id }), System.Text.Encoding.UTF8, "application/json");
                    using HttpResponseMessage response = await client.PostAsync("/api/agent/share/discard", body);
                    string payload = await response.Content.ReadAsStringAsync();
                    using JsonDocument answer = JsonDocument.Parse(payload);
                    discardAccepted = response.IsSuccessStatusCode
                        && answer.RootElement.TryGetProperty("ok", out JsonElement ok)
                        && ok.ValueKind == JsonValueKind.True;
                    discardDetail = $"HTTP {(int)response.StatusCode}";
                }
                catch (Exception ex)
                {
                    discardDetail = ex.GetType().Name + ": " + ex.Message;
                }

                bool cleaned = false;
                bool durableGone = false;
                bool stagedGone = false;
                for (int attempt = 0; attempt < 40; attempt++)
                {
                    await Task.Delay(100);
                    durableGone = !Directory.Exists(ready)
                        && !Directory.Exists(claimed)
                        && !Directory.Exists(acknowledged);
                    stagedGone = stagedFiles.All(path => !File.Exists(path));
                    cleaned = discardAccepted
                        && _host.App.Shares.PendingCount == 0
                        && durableGone
                        && stagedGone;
                    if (cleaned)
                        break;
                }
                Console.WriteLine(cleaned
                    ? $"TensorAgent: sharecheck discard cleanup PASS durable envelope + {stagedFiles.Length} staged file(s) reclaimed"
                    : $"TensorAgent: sharecheck discard cleanup FAIL route={discardAccepted} ({discardDetail}) pending={_host.App.Shares.PendingCount} durableGone={durableGone} stagedBefore={stagedBeforeDiscard} stagedGone={stagedGone}");
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: sharecheck FAIL " + ex);
        }
        finally
        {
            writer?.Abandon();
        }
    }

    /// <summary>
    /// Device E2E hook (Debug builds only): a real generation, on the real engine,
    /// carried across a real background switch.
    ///
    /// <para>
    /// This exists for the one bug that cannot be reproduced anywhere else. Leaving
    /// TensorAgent mid-answer used to fail that answer AND every answer after it,
    /// because iOS refuses GPU work from an app that is not frontmost and ggml-metal
    /// answers a refused command buffer by latching an error flag that only recreating
    /// the backend clears. A simulator cannot show the refusal -- there is no Metal
    /// there -- but it CAN show the gate: the engine's step count must stop moving while
    /// the app is away and start again when it is back. A unit test can show neither,
    /// because the whole mechanism is a UIKit lifecycle notification and a GPU.
    /// </para>
    /// <para>
    /// So this asks the app's own <c>/api/chat</c> for a long answer and reports what
    /// becomes of it through <see cref="Core.Hosting.AgentAppHost.TraceBackground"/> as
    /// well as stdout: stdout on a phone is readable only while a console is attached,
    /// and the interesting part happens after the app has left the screen. Launched
    /// with <c>TENSORAGENT_BACKGROUND_CHECK=1</c>; <c>scripts/verify-background.sh</c>
    /// sends the app away and back while this runs and reads the trace afterwards. One
    /// <c>bgcheck</c> line per event; a heartbeat every few seconds carries the token
    /// count, the engine's step count, and whether the gate is open, which is what
    /// proves the model stopped rather than merely that the page did.
    /// </para>
    /// </summary>
    private async Task RunBackgroundProbeAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_BACKGROUND_CHECK"), "1", StringComparison.Ordinal))
            return;

        Core.Hosting.AgentAppHost app = _host.App;
        void Say(string line)
        {
            Console.WriteLine("TensorAgent: bgcheck " + line);
            app.TraceBackground("bgcheck " + line);
        }
        long Steps() => app.ModelService.EngineHost.TryGetEngine()?.TotalStepsRun ?? -1;
        long Held() => app.ModelService.EngineHost.TryGetEngine()?.StepsHeldByGate ?? -1;

        try
        {
            // Named, so a script reading the trace afterwards can find THIS run's lines
            // without trusting two clocks to agree. The file outlives launches.
            if (Environment.GetEnvironmentVariable("TENSORAGENT_BACKGROUND_RUN") is { Length: > 0 } run)
                Say($"run {run}");

            // The model loads in the background at startup and takes as long as it
            // takes; there is nothing to generate with until it is there.
            for (int i = 0; i < 300 && app.ModelLoad != Core.Hosting.AgentAppHost.ModelLoadState.Loaded; i++)
                await Task.Delay(1000);
            if (app.ModelLoad != Core.Hosting.AgentAppHost.ModelLoadState.Loaded)
            {
                Say($"FAIL no model to generate with (state {app.ModelLoad})");
                return;
            }
            // Not beside the warm-up, whose cancellation would otherwise be the first
            // thing this measures.
            for (int i = 0; i < 90 && !app.PrefixCacheIsWarm; i++)
                await Task.Delay(1000);

            string prompt = Environment.GetEnvironmentVariable("TENSORAGENT_BACKGROUND_PROMPT") is { Length: > 0 } asked
                ? asked
                : "Count from one to three hundred. Write each number in words on its own line, "
                  + "and after each one add a short sentence about that number.";
            int maxTokens = int.TryParse(Environment.GetEnvironmentVariable("TENSORAGENT_BACKGROUND_TOKENS"), out int t) ? t : 4096;

            using var client = new HttpClient
            {
                BaseAddress = new Uri(_host.BaseUrl),
                Timeout = Timeout.InfiniteTimeSpan,
            };
            client.DefaultRequestHeaders.Add("Cookie", $"tensoragent_token={_host.Token}");

            // Through a real session, like the page: the default session is declared a
            // different tool set and would warm nothing for the message after this.
            async Task<string> NewSessionAsync()
            {
                using HttpResponseMessage made = await client.PostAsync("/api/sessions?conversation=new", content: null);
                using JsonDocument answer = JsonDocument.Parse(await made.Content.ReadAsStringAsync());
                return answer.RootElement.GetProperty("sessionId").GetString()!;
            }

            async Task<(int Tokens, string? Error, int Restarts)> AskAsync(string sessionId, string question, int budget, bool heartbeat)
            {
                var body = new
                {
                    sessionId,
                    messages = new[] { new { role = "user", content = question } },
                    maxTokens = budget,
                    think = false,
                };
                using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
                {
                    Content = new StringContent(JsonSerializer.Serialize(body), System.Text.Encoding.UTF8, "application/json"),
                };
                var clock = System.Diagnostics.Stopwatch.StartNew();
                int tokens = 0, restarts = 0;
                string? error = null;
                TimeSpan lastSaid = TimeSpan.Zero;

                using HttpResponseMessage response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
                await using Stream stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);
                while (await reader.ReadLineAsync() is { } line)
                {
                    if (!line.StartsWith("data: ", StringComparison.Ordinal))
                        continue;
                    using JsonDocument frame = JsonDocument.Parse(line[6..]);
                    JsonElement root = frame.RootElement;
                    if (root.TryGetProperty("token", out _))
                        tokens++;
                    if (root.TryGetProperty("restart", out JsonElement restart) && restart.ValueKind == JsonValueKind.String)
                    {
                        restarts++;
                        Say($"restart: {restart.GetString()}");
                    }
                    if (root.TryGetProperty("error", out JsonElement e) && e.ValueKind == JsonValueKind.String)
                        error = e.GetString();

                    // A heartbeat, so the trace shows the answer still moving rather than
                    // only its beginning and its end -- and the ENGINE's step count next to
                    // the token count, because the claim is that the model stops, not
                    // that the page stops reading.
                    if (heartbeat && clock.Elapsed - lastSaid > TimeSpan.FromSeconds(3))
                    {
                        lastSaid = clock.Elapsed;
                        Say($"{tokens} tokens after {clock.Elapsed.TotalSeconds:0}s, engine steps {Steps()}, held {Held()}, gate {(app.Compute.IsOpen ? "open" : "CLOSED")}");
                    }
                    if (root.TryGetProperty("done", out JsonElement done) && done.ValueKind == JsonValueKind.True)
                        break;
                }
                clock.Stop();
                return (tokens, error, restarts);
            }

            string session = await NewSessionAsync();
            Say($"asking for a long answer (engine steps {Steps()})");
            long closuresAtStart = app.Compute.Closures;
            long rebuildsAtStart = app.EngineRebuilds;
            var whole = System.Diagnostics.Stopwatch.StartNew();
            (int tokens, string? error, int restarts) = await AskAsync(session, prompt, maxTokens, heartbeat: true);
            whole.Stop();

            long pauses = app.Compute.Closures - closuresAtStart;
            string away = pauses > 0 ? $", paused {pauses} time(s) for {app.Compute.TotalClosed.TotalSeconds:0.#}s, engine held {Held()} time(s)" : "";
            Say(error is { Length: > 0 }
                ? $"FAIL after {tokens} tokens{away}: {error}"
                : $"ok {tokens} tokens in {whole.Elapsed.TotalSeconds:0.#}s{away}, {restarts} restart(s)");
            Say($"the engine was rebuilt {app.EngineRebuilds - rebuildsAtStart} time(s) during the turn");

            // And then the question the whole fix turns on. A refused command buffer
            // poisons ggml-metal for the rest of the process, so the only proof the app
            // is not quietly finished is a SECOND answer, asked afterwards, that works.
            if (pauses > 0)
            {
                if (app.EngineNeedsReload)
                {
                    Say("the engine is marked for a rebuild; waiting for it");
                    for (int i = 0; i < 120 && app.EngineNeedsReload; i++)
                        await Task.Delay(1000);
                }
                (int secondTokens, string? secondError, _) = await AskAsync(
                    await NewSessionAsync(), "In one short sentence, what is a transistor?", 64, heartbeat: false);
                Say(secondError is { Length: > 0 }
                    ? "FAIL the next answer after coming back failed too: " + secondError
                    : $"the next answer after coming back worked ({secondTokens} tokens)");
            }
            else
            {
                Say("the app was never sent away, so nothing was checked about coming back");
            }
            Say($"done: rebuilt {app.EngineRebuilds} time(s) this launch");
        }
        catch (Exception ex)
        {
            Say("FAIL " + ex.Message);
        }
    }

    /// <summary>A valid PNG of a given size, big enough that its IDAT cannot be tiny.</summary>
    private static byte[] SolidPng(int width, int height)
    {
        // Written by hand rather than through a platform encoder, so the probe tests the
        // upload path and not ImageIO. Noise rather than a flat colour: a solid image
        // deflates to almost nothing, and the point here is the number of bytes.
        var raw = new MemoryStream();
        var random = new Random(20260903);
        var row = new byte[width * 3];
        for (int y = 0; y < height; y++)
        {
            raw.WriteByte(0);
            random.NextBytes(row);
            raw.Write(row);
        }

        var png = new MemoryStream();
        png.Write(new byte[] { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A });
        var header = new MemoryStream();
        header.Write(BigEndian(width));
        header.Write(BigEndian(height));
        header.Write(new byte[] { 8, 2, 0, 0, 0 });
        WriteChunk(png, "IHDR"u8.ToArray(), header.ToArray());

        var deflated = new MemoryStream();
        using (var zlib = new System.IO.Compression.ZLibStream(deflated, System.IO.Compression.CompressionLevel.Fastest, leaveOpen: true))
            zlib.Write(raw.ToArray());
        WriteChunk(png, "IDAT"u8.ToArray(), deflated.ToArray());
        WriteChunk(png, "IEND"u8.ToArray(), Array.Empty<byte>());
        return png.ToArray();

        static byte[] BigEndian(int value) =>
            new[] { (byte)(value >> 24), (byte)(value >> 16), (byte)(value >> 8), (byte)value };

        static void WriteChunk(Stream target, byte[] type, byte[] data)
        {
            target.Write(BigEndian(data.Length));
            target.Write(type);
            target.Write(data);
            var crc = new MemoryStream();
            crc.Write(type);
            crc.Write(data);
            target.Write(BigEndian(unchecked((int)Crc32(crc.ToArray()))));
        }

        static uint Crc32(byte[] bytes)
        {
            uint crc = 0xFFFFFFFF;
            foreach (byte b in bytes)
            {
                crc ^= b;
                for (int i = 0; i < 8; i++)
                    crc = (crc >> 1) ^ (0xEDB88320u & (uint)(-(int)(crc & 1)));
            }
            return crc ^ 0xFFFFFFFF;
        }
    }

    /// <summary>
    /// Device E2E hook (Debug builds only): turn the network switch on the way the
    /// Settings page turns it on, run <c>curl</c> the way the model runs it, and report
    /// what the shell said — then put the switch back where it was.
    ///
    /// <para>
    /// This is the reported bug end to end and it can only be answered here: the switch
    /// and the shell agree perfectly in a unit test, and the thing that was broken was
    /// that nothing carried the change from the settings file into the objects a launch
    /// had already built. Off by default because it reaches the internet.
    /// </para>
    /// </summary>
    private void RunNetworkProbe()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_NETWORK_CHECK"), "1", StringComparison.Ordinal))
            return;

        _ = Task.Run(() =>
        {
            Core.Settings.AppSettings original = _host.App.Settings.Load();
            try
            {
                // BOTH directions, in one launch, because half of it proves nothing. A
                // container whose switch was already on would pass a check that only
                // turned it on, while the bug -- a change that reaches the settings file
                // and nothing else -- would be untouched.
                Console.WriteLine($"TensorAgent: netcheck starting from allowNetwork={original.AllowNetwork}");

                Set(false);
                Console.WriteLine("TensorAgent: netcheck off · " + RunShell(
                    "curl -s https://example.com", Core.Sandbox.ExecutionPolicy.NetworkDisabledMessage,
                    expectFailure: true));

                Set(true);
                Console.WriteLine("TensorAgent: netcheck on · " + RunShell(
                    "curl -s https://example.com", "Example Domain"));
                Console.WriteLine("TensorAgent: netcheck on python · " + RunShell(
                    "python3 -c \"import urllib.request as u; print('got', len(u.urlopen('https://example.com').read()))\"",
                    "got "));
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: netcheck FAIL " + ex.Message);
            }
            finally
            {
                _host.App.Settings.Save(original);
                _host.App.ApplySettings(original);
                Console.WriteLine($"TensorAgent: netcheck restored to allowNetwork={original.AllowNetwork} · {_host.App.DescribeEngineForLog()}");
            }
        });

        void Set(bool allow)
        {
            Core.Settings.AppSettings settings = _host.App.Settings.Load();
            settings.AllowNetwork = allow;
            _host.App.Settings.Save(settings);
            // Exactly what the Settings page does, and the line that used to be missing.
            _host.App.ApplySettings(settings);
            Console.WriteLine($"TensorAgent: netcheck switch -> {allow} · {_host.App.DescribeEngineForLog()}");
        }
    }

    /// <summary>
    /// Run one command through the app's REAL execution terms — the ones the model's
    /// shell tool is launched with — and describe what came back in one line.
    /// </summary>
    private string RunShell(string command, string? expected, bool expectFailure = false)
    {
        string root = Path.Combine(_host.App.Paths.ScratchDirectory, "probe-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            TensorSharp.AgentHost.CodeExec.ConfinedResult result =
                ((TensorSharp.AgentHost.CodeExec.IShellBackend)_host.App.Backend).Run(
                    new TensorSharp.AgentHost.CodeExec.ShellLaunch
                    {
                        // Argv rather than Command: a Command launch is a conversation's
                        // shell and is required to carry the session whose directory and
                        // exports it restores. A probe has neither.
                        Argv = new[] { "sh", "-c", command },
                        WorkingDirectory = root,
                        WriteDirectory = root,
                        ReadOnlyDirectory = root,
                        // The point of the probe: whatever the switch currently says.
                        AllowNetwork = _host.App.CodeExec.AllowNetwork,
                        Timeout = TimeSpan.FromSeconds(30),
                    });

            string output = (result.Stdout + result.Stderr).Trim();
            // With the network off, being refused IS the right answer, and saying the
            // refusal is what the model reads. Both halves have to be checked or the
            // probe passes on a build where nothing is enforced at all.
            bool said = expected is null || output.Contains(expected, StringComparison.Ordinal);
            bool ok = expectFailure ? !result.Ok && said : result.Ok && said;
            string detail = output.Length > 200 ? output[..200] + "…" : output;
            return $"{(ok ? "ok" : "FAIL")} `{command.Split('\n')[0]}` exit={result.ExitCode} · {detail.Replace('\n', ' ')}";
        }
        catch (Exception ex)
        {
            return "FAIL " + ex.Message;
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { /* scratch */ }
        }
    }

    /// <summary>
    /// Simulator E2E hook (Debug builds only): when the app is launched with
    /// TENSORAGENT_DEMO_PROMPT set (simctl passes it as
    /// SIMCTL_CHILD_TENSORAGENT_DEMO_PROMPT), type that prompt into the Web UI
    /// and send it once the page has loaded. simctl has no way to tap or type
    /// into the WebView, so this is how scripts/run-sim.sh drives a visible
    /// round trip for a screenshot without modifying index.html.
    /// </summary>
    private async void OnNavigatedSendDemoPrompt(object? sender, WebNavigatedEventArgs e)
    {
        _webView.Navigated -= OnNavigatedSendDemoPrompt;
        if (e.Result != WebNavigationResult.Success)
            return;

        await RunUiCheckAsync();
        await RunTtftProbeAsync();

        // Screenshot hook: neither simctl nor devicectl can tap, so the menu -- the one
        // surface all of this app's navigation lives on -- can otherwise never appear in
        // a picture of the running app.
        if (string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_OPEN_MENU"), "1", StringComparison.Ordinal))
            await Tell("openMenu");

        string? prompt = Environment.GetEnvironmentVariable("TENSORAGENT_DEMO_PROMPT");
        if (string.IsNullOrWhiteSpace(prompt))
        {
            return;
        }

        try
        {
            // sendMessage() refuses while the page's currentLoadedModel is null, so this
            // has to wait for a model rather than for a fixed delay. It used to sleep
            // 1500 ms, which was enough when a model was already loaded at startup and
            // not when one is being loaded concurrently -- the prompt fired first and
            // was silently refused, which is exactly the bug being tested for, produced
            // by the harness instead of by the product. Polling refreshModel() is both
            // the correct wait and a direct exercise of the fix.
            // The HOST's word first: /api/models says "loaded" the moment the text
            // weights are in, while the projector is still being read and the warm-up
            // has not run. A send in that window is the harness racing the load, not
            // the product; wait for the load to settle, and -- for the page check,
            // whose turn must be the only one -- for the warm-up too.
            Core.Hosting.AgentAppHost app = _host.App;
            for (int i = 0; i < 600 && app.ModelLoad == Core.Hosting.AgentAppHost.ModelLoadState.Loading; i++)
                await Task.Delay(500);
            if (string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_PAGE_BACKGROUND_CHECK"), "1", StringComparison.Ordinal))
            {
                for (int i = 0; i < 180 && app.ModelLoad == Core.Hosting.AgentAppHost.ModelLoadState.Loaded && !app.PrefixCacheIsWarm; i++)
                    await Task.Delay(500);
            }
            bool ready = false;
            for (int i = 0; i < 60 && !ready; i++)
            {
                await Task.Delay(1000);
                await _webView.EvaluateJavaScriptAsync(
                    "window.TensorAgent && window.TensorAgent.refreshModel ? window.TensorAgent.refreshModel() : 0");
                // hasModel() is the synchronous half: refreshModel starts the fetch and
                // this reports what the page ended up believing.
                string? answer = await _webView.EvaluateJavaScriptAsync(
                    "window.TensorAgent && window.TensorAgent.hasModel ? window.TensorAgent.hasModel() : false");
                ready = answer is not null && answer.Contains("true", StringComparison.OrdinalIgnoreCase);
            }
            Console.WriteLine($"TensorAgent: demo prompt sees a loaded model = {ready}");
            if (!ready)
            {
                // Into the trace as well: a device launch captures no stdout, so this
                // used to end as eight minutes of silence and 'no turn started'.
                _host.App.TraceBackground(
                    "pagecheck FAIL the page never saw a loaded model, so nothing was sent"
                    + $" (host says {app.ModelLoad}{(app.ModelLoadError is { Length: > 0 } why ? ": " + why : string.Empty)})");
                return;
            }

            // Through the page's own bridge, not through the desktop page's globals.
            // This used to write into #message-input and call sendMessage(), which are
            // TensorSharp.Server's page; the app has had its own since, so the hook was
            // typing into an element that does not exist and the "demo prompt sent"
            // line was printed for a prompt nobody received.
            // Through the base64 bridge rather than spliced into the script: MAUI wraps
            // the script in a single-quoted literal (see CallBridgeAsync), so a prompt
            // with a newline or an apostrophe in it made eval throw and nothing was
            // sent -- while the line below still said it had been.
            // Emptied first: insertText APPENDS (it is what dictation uses), and a
            // composer holding a draft would send that draft plus the prompt.
            await _webView.EvaluateJavaScriptAsync("document.getElementById('text').value = ''; true");
            string? typed = await CallBridgeAsync("insertText", new { text = prompt });
            if (typed is null || !typed.Contains("ok", StringComparison.Ordinal))
                _host.App.TraceBackground("pagecheck FAIL the page refused the prompt: " + (typed ?? "no answer"));
            // Blurred after sending: insertText leaves the composer focused, which keeps
            // the keyboard -- in the simulator, its accessory bar -- over the transcript
            // in every screenshot this hook exists to take.
            await _webView.EvaluateJavaScriptAsync(
                "window.TensorAgent.send(); if (document.activeElement) document.activeElement.blur(); true");
            Console.WriteLine("TensorAgent: demo prompt sent through the Web UI: " + prompt);
            if (string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_PAGE_BACKGROUND_CHECK"), "1", StringComparison.Ordinal))
            {
                // What the page made of the send, for the trace the check reads.
                await Task.Delay(3000);
                await TracePageDiagnosticsAsync("after the demo prompt");
            }

            // And, if asked, walk away from it while it is being answered.
            await RunNavigationProbeAsync();
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: demo prompt failed: " + ex.Message);
        }
    }

    /// <summary>
    /// Device E2E hook (Debug builds only): measure the time to first token, on this
    /// phone, with the app's own prompt.
    ///
    /// <para>
    /// Launched with <c>TENSORAGENT_TTFT_CHECK=1</c> (and a model, via
    /// <c>TENSORAGENT_USE_MODEL</c> or the remembered choice). The question a user
    /// asks is "why does it take so long before it starts answering", and the answer
    /// is almost never the model's speed — it is how many prompt tokens are forwarded
    /// before the first one comes out, and whether any of that work was already done.
    /// So this asks the app's own <c>/api/chat</c> for four turns and prints, for each,
    /// the first-token time and how much of the prompt the KV cache served: a first
    /// turn (served by the warm-up, or cold), a second turn in the same conversation,
    /// a first turn in a NEW conversation (served by the shared-prefix checkpoint),
    /// and a follow-up in that one. Those numbers say which fault is present; a
    /// guess cannot. One <c>ttft</c> line per turn on stdout, readable with
    /// <c>devicectl device process launch --console</c>.
    /// </para>
    /// </summary>
    private async Task RunTtftProbeAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_TTFT_CHECK"), "1", StringComparison.Ordinal))
            return;

        Core.Hosting.AgentAppHost app = _host.App;
        // To stdout AND to a file: a device console detaches (or will not attach at
        // all when the developer disk image is wedged), and the numbers are the whole
        // point. Pulled back with `devicectl device copy from`.
        string ttftLog = Path.Combine(app.Paths.LogsDirectory, "ttft.log");
        void Say(string line)
        {
            Console.WriteLine("TensorAgent: " + line);
            try { File.AppendAllText(ttftLog, $"{DateTimeOffset.Now:yyyy-MM-dd HH:mm:ss.fff} {line}{Environment.NewLine}"); }
            catch (Exception) { /* diagnostic only */ }
        }
        try
        {
            if (Environment.GetEnvironmentVariable("TENSORAGENT_TTFT_RUN") is { Length: > 0 } run)
                Say($"ttft run {run}");
            for (int i = 0; i < 180 && app.ModelLoad != Core.Hosting.AgentAppHost.ModelLoadState.Loaded; i++)
                await Task.Delay(1000);
            if (app.ModelLoad != Core.Hosting.AgentAppHost.ModelLoadState.Loaded)
            {
                Say($"ttft FAIL no model (state {app.ModelLoad})");
                return;
            }

            using var client = new HttpClient
            {
                BaseAddress = new Uri(_host.BaseUrl),
                Timeout = Timeout.InfiniteTimeSpan,
            };
            client.DefaultRequestHeaders.Add("Cookie", $"tensoragent_token={_host.Token}");

            // Wait for the warm-up the way a user does: by taking a few seconds to
            // type. A turn asked for the instant the weights land CANCELS the warm-up
            // by design, so a probe that fires immediately measures the cold path
            // forever and never sees the thing it exists to measure.
            for (int i = 0; i < 90 && !app.PrefixCacheIsWarm; i++)
                await Task.Delay(1000);
            Say(app.PrefixCacheIsWarm
                ? "ttft asking with the cache already warm"
                : "ttft asking WITHOUT a warm cache (the warm-up did not finish in time)");

            // Through a REAL session, exactly as the page does: a request with no
            // session is served by the default one, which has no workspace and is
            // therefore declared a different set of tools.
            async Task<string> NewSessionAsync()
            {
                using HttpResponseMessage made = await client.PostAsync("/api/sessions?conversation=new", content: null);
                using JsonDocument answer = JsonDocument.Parse(await made.Content.ReadAsStringAsync());
                return answer.RootElement.GetProperty("sessionId").GetString()!;
            }

            // The same thinking mode the warm-up used: on Gemma 4 the two modes share
            // no prompt prefix, so measuring the other one would measure nothing.
            bool think = app.Settings.Load().ThinkByDefault;
            string session = await NewSessionAsync();
            var history = new List<object>();
            async Task AskAsync(string label, string question, bool sameConversation)
            {
                if (!sameConversation)
                {
                    history.Clear();
                    session = await NewSessionAsync();
                }
                history.Add(new { role = "user", content = question });

                var body = new { sessionId = session, messages = history.ToArray(), maxTokens = 24, think };
                using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
                {
                    Content = new StringContent(JsonSerializer.Serialize(body), System.Text.Encoding.UTF8, "application/json"),
                };

                var clock = System.Diagnostics.Stopwatch.StartNew();
                TimeSpan firstToken = TimeSpan.Zero;
                var answer = new System.Text.StringBuilder();
                int prompt = 0, reused = 0;
                string? error = null;

                using HttpResponseMessage response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
                await using Stream stream = await response.Content.ReadAsStreamAsync();
                using var reader = new StreamReader(stream);
                while (await reader.ReadLineAsync() is { } line)
                {
                    if (!line.StartsWith("data: ", StringComparison.Ordinal))
                        continue;
                    JsonElement frame = JsonDocument.Parse(line[6..]).RootElement;
                    bool hasToken = frame.TryGetProperty("token", out JsonElement token);
                    if (hasToken || frame.TryGetProperty("thinking", out _))
                    {
                        if (firstToken == TimeSpan.Zero)
                            firstToken = clock.Elapsed;
                        if (hasToken)
                            answer.Append(token.GetString());
                    }
                    // The whole answer, when text the page showed proved to be reasoning:
                    // the next turn sends back what remains, as the page does.
                    if (frame.TryGetProperty("replace", out JsonElement whole) && whole.ValueKind == JsonValueKind.String)
                        answer.Clear().Append(whole.GetString());
                    if (frame.TryGetProperty("promptTokens", out JsonElement p))
                        prompt = p.GetInt32();
                    if (frame.TryGetProperty("kvReusedTokens", out JsonElement r))
                        reused = r.GetInt32();
                    if (frame.TryGetProperty("error", out JsonElement e) && e.ValueKind == JsonValueKind.String)
                        error = e.GetString();
                }
                clock.Stop();

                history.Add(new { role = "assistant", content = answer.ToString() });
                Say(error is { Length: > 0 }
                    ? $"ttft {label} FAILED: {error}"
                    : $"ttft {label}: first token {firstToken.TotalSeconds:0.0}s, "
                      + $"{prompt} prompt tokens, {reused} reused ({(prompt > 0 ? 100.0 * reused / prompt : 0):0}%), "
                      + $"whole turn {clock.Elapsed.TotalSeconds:0.0}s");
            }

            await AskAsync("turn 1 (first chat)", "Say the single word: apple.", sameConversation: false);
            await AskAsync("turn 2 (same chat)", "Now say: banana.", sameConversation: true);
            await AskAsync("turn 3 (new chat)", "Say the single word: cherry.", sameConversation: false);
            await AskAsync("turn 4 (same chat)", "Now say: date.", sameConversation: true);
            Say("ttft done");
        }
        catch (Exception ex)
        {
            Say("ttft FAIL " + ex.Message);
        }
    }

    /// <summary>
    /// Device E2E hook (Debug builds only): send a prompt, LEAVE the chat while the
    /// model is answering, come back, and report whether the answer carried on.
    ///
    /// <para>
    /// This is the reported bug, and the only place it can be answered. It is not a
    /// question about the server — a unit test can drop a reader and watch the frames
    /// keep coming — but about what iOS does to a WKWebView whose view leaves the
    /// window, which is to suspend its content process mid-stream. Nothing off-device
    /// reproduces that. So the probe does what the user did: opens the model list for
    /// twelve seconds while the model is working, and prints the turn's frame count
    /// before and after, then the length of the answer the page ends up showing.
    /// </para>
    /// <para>
    /// The frame count rising while the chat was off-screen is the whole claim. The
    /// answer being on the page afterwards is the other half: that the page found its
    /// way back to the turn rather than showing the half sentence it walked away from.
    /// </para>
    /// </summary>
    private async Task RunNavigationProbeAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_NAV_CHECK"), "1", StringComparison.Ordinal))
            return;

        Core.Sessions.ChatTurnManager turns = _host.App.Turns;
        int away = int.TryParse(Environment.GetEnvironmentVariable("TENSORAGENT_NAV_SECONDS"), out int seconds) ? seconds : 15;
        try
        {
            // Not merely "a turn exists" but "it is producing". A prompt of any length
            // spends its first stretch in prefill, which is minutes on a simulator with
            // no Metal, and leaving during that would compare nothing to nothing.
            if (!await Until(() => turns.TotalFrames > 0, TimeSpan.FromMinutes(6)))
            {
                Console.WriteLine($"TensorAgent: navcheck FAIL nothing was generated · {turns.Describe()}");
                return;
            }

            int framesBefore = turns.TotalFrames;
            Console.WriteLine($"TensorAgent: navcheck leaving the chat with a turn running · {turns.Describe()}");

            await AppShell.OpenAsync("models");
            await Task.Delay(TimeSpan.FromSeconds(away));
            int framesAway = turns.TotalFrames;
            bool stillRunning = turns.IsBusy;
            Console.WriteLine(
                $"TensorAgent: navcheck {(framesAway > framesBefore ? "ok" : "FAIL")} away for {away}s: "
                + $"frames {framesBefore} -> {framesAway}, still generating={stillRunning} · {turns.Describe()}");

            await AppShell.BackToChatAsync();
            // The page has to be asked to resume, ask the host what is running, open a
            // stream and replay it. Generous, because none of that is instant on a
            // WebView whose content process has just been woken up.
            int shown = 0;
            await Until(async () => (shown = await AnswerLengthAsync()) > 0, TimeSpan.FromSeconds(30));
            Console.WriteLine(
                $"TensorAgent: navcheck {(shown > 0 ? "ok" : "FAIL")} back in the chat, "
                + $"the answer on screen is {shown} characters · {turns.Describe()}");
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: navcheck FAIL " + ex.Message);
        }
    }

    /// <summary>How many characters the page is showing in the last assistant bubble.</summary>
    private async Task<int> AnswerLengthAsync()
    {
        string? shown = await _webView.EvaluateJavaScriptAsync(
            "(function(){var b=document.querySelectorAll('.turn.bot .bubble');"
            + "return b.length ? String(b[b.length-1].textContent.length) : '0';})()");
        return int.TryParse((shown ?? string.Empty).Trim('"'), out int length) ? length : 0;
    }

    /// <summary>Poll until it is true, or give up. False means it never became true.</summary>
    private static Task<bool> Until(Func<bool> condition, TimeSpan budget)
        => Until(() => Task.FromResult(condition()), budget);

    private static async Task<bool> Until(Func<Task<bool>> condition, TimeSpan budget)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (clock.Elapsed < budget)
        {
            if (await condition())
                return true;
            await Task.Delay(500);
        }
        return false;
    }

    /// <summary>
    /// Drive the composer's own gestures inside the real WebView and report what
    /// happened, one <c>uicheck</c> line per assertion.
    ///
    /// <para>
    /// The page's behaviour is the half of this app no unit test can reach: it is
    /// JavaScript in WKWebView, and everything interesting about it — whether a long
    /// press on the message box really turns the composer into the hold-to-talk button,
    /// whether the button that comes back really returns you to typing, whether the
    /// live-progress panel is styled at all — is a question about layout and events on
    /// a real engine. <c>simctl</c> cannot tap, so the events are synthesised here and
    /// the answers go to stdout beside the interpreter self-test, where
    /// <c>verify-sim.sh</c> asserts on them.
    /// </para>
    /// </summary>
    private async Task RunUiCheckAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_UI_CHECK"), "1", StringComparison.Ordinal))
            return;

        try
        {
            await _webView.EvaluateJavaScriptAsync(UiCheckScript);
            for (int i = 0; i < 40; i++)
            {
                await Task.Delay(250);
                string? raw = await _webView.EvaluateJavaScriptAsync("window.__uicheck || ''");
                string result = (raw ?? string.Empty).Trim('"').Replace("\\\"", "\"");
                if (result.Length == 0)
                    continue;
                foreach (string check in result.Split('|', StringSplitOptions.RemoveEmptyEntries))
                    Console.WriteLine("TensorAgent: uicheck " + check.Replace("=", " ").Trim());
                return;
            }
            Console.WriteLine("TensorAgent: uicheck FAIL the page never reported (the script did not finish)");
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: uicheck FAIL " + ex.Message);
        }
    }

    /// <summary>
    /// The check itself. Written as one self-contained expression because that is what
    /// <c>evaluateJavaScript</c> takes, and it parks its answer on
    /// <c>window.__uicheck</c> because that call cannot await a promise and every
    /// gesture here needs real time to pass.
    /// </summary>
    private const string UiCheckScript =
        // ONE LINE, and no `//` comments inside it. MAUI's EvaluateJavaScriptAsync
        // flattens the script before handing it to WKWebView, so a line comment
        // swallows everything after it and the page answers "SyntaxError: Unexpected
        // EOF" -- which is what it did the first time this was written as a block.
        "(function(){var out=[];"
        + "function add(n,ok,d){out.push(n+'='+(ok?'ok':'FAIL '+(d||'')));}"
        + "function shown(e){return !!e&&getComputedStyle(e).display!=='none';}"
        + "var text=document.getElementById('text'),hold=document.getElementById('hold'),"
        + "back=document.getElementById('abc'),wrap=document.getElementById('textwrap');"
        + "function press(t){text.dispatchEvent(new PointerEvent(t,{bubbles:true,cancelable:true,"
        + "clientX:100,clientY:100,pointerId:1,pointerType:'touch'}));}"
        // NOT calling nativeReady() here, deliberately. It used to, and that is exactly
        // why this check passed on a phone where holding the box answered "Voice input
        // is only available in the app": the check handed the page the fact it was
        // supposed to be testing. Now the hold only switches if the app really told the
        // page it was the app, so this fails when that handshake is broken.

        + "add('voice-switch-gone',document.getElementById('voice')===null,'the composer still carries a Voice switch');"
        + "add('reasoning-is-a-setting',document.getElementById('think')===null,'the composer still carries a Reasoning switch');"
        + "add('skills-moved-to-the-menu',!document.getElementById('skills-btn')"
        + "&&!!document.querySelector('#nav-sheet [data-sheet=\"skills-sheet\"]'),'no Skills item in the menu');"
        + "var strip=document.getElementById('activity'),row=document.getElementById('row');"
        + "add('activity-above-the-box',!!strip&&!!row&&strip.parentNode===row.parentNode"
        + "&&(strip.compareDocumentPosition(row)&Node.DOCUMENT_POSITION_FOLLOWING)!==0"
        + "&&getComputedStyle(strip).display==='none',"
        + "'the activity strip is missing, is not above the message box, or is showing with nothing to say');"
        + "press('pointerdown');setTimeout(function(){press('pointerup');setTimeout(function(){"
        + "add('a-tap-still-types',!document.body.classList.contains('voice'),'a short press switched to voice mode');"
        + "press('pointerdown');setTimeout(function(){"
        + "var canDictate=window.TensorAgent.canDictate();"
        + "add('native-handshake',JSON.parse(window.TensorAgent.diagnostics()).native,'the app did not announce its capabilities');"
        + "add(canDictate?'holding-the-box-gives-hold-to-talk':'unsupported-dictation-keeps-the-box',"
        + "canDictate?(document.body.classList.contains('voice')&&shown(hold)&&!shown(wrap))"
        + ":(!document.body.classList.contains('voice')&&!shown(hold)&&shown(wrap)),"
        + "'voice='+document.body.classList.contains('voice')+' hold='+shown(hold)+' box='+shown(wrap));"
        + "back.click();setTimeout(function(){"
        + "add('the-keyboard-button-returns',!document.body.classList.contains('voice')&&shown(wrap),'still in voice mode');"
        // The main menu, measured rather than asserted about. "Opens from the left" is a
        // rectangle -- flush with the left edge, narrower than the window, as tall as it
        // -- and a bottom sheet that had merely been renamed would pass every check that
        // only looked at class names.
        + "document.getElementById('menu').click();setTimeout(function(){"
        + "var menu=document.getElementById('nav-sheet'),box=menu.getBoundingClientRect();"
        + "add('the-menu-comes-from-the-left',menu.classList.contains('drawer')&&box.left<1"
        + "&&box.width<window.innerWidth*0.95&&box.height>window.innerHeight*0.9,"
        + "'left='+Math.round(box.left)+' width='+Math.round(box.width)+' of '+window.innerWidth"
        + "+' height='+Math.round(box.height)+' of '+window.innerHeight);"
        + "var chats=document.getElementById('nav-chats');"
        + "add('the-menu-lists-the-saved-chats',!!chats&&menu.contains(chats)"
        + "&&(chats.querySelectorAll('.navchat').length>0||!!chats.querySelector('.navempty')),"
        + "'the menu has no chat list');"
        + "add('a-turn-can-be-taken-back-up',typeof window.TensorAgent.resumeTurn==='function'"
        + "&&typeof window.TensorAgent.openConversation==='function','the bridge cannot resume a turn');"
        + "var master=document.getElementById('skills-master');"
        + "add('skills-have-a-master-switch',!!master&&master.type==='checkbox'"
        + "&&typeof window.TensorAgent.skillsEnabled==='function',"
        + "'the skills sheet has no on/off switch above its list');"
        + "document.getElementById('sheet-bg').click();"
        + "window.__uicheck=out.join('|');},420);},120);},700);},140);},140);})(); true";
#endif

    /// <summary>
    /// Re-sync the page's idea of which model is loaded, every time the chat is shown.
    ///
    /// <para>
    /// The Web UI reads the loaded model ONCE, at page load, into its own
    /// `currentLoadedModel`. That is correct for the desktop server, where the model
    /// cannot change while the page is open. In the app it can: the user picks one in
    /// the Models list. Without this the header still said "No model configured" after
    /// a model had been chosen AND loaded, and the page's own send guard refused to
    /// send anything -- so the prompt never reached a server that was ready to answer
    /// it. Reported from a phone, twice: repointing the engine was necessary and not
    /// sufficient, because the half the user actually looks at had not been told.
    /// </para>
    /// <para>
    /// Done on appearing rather than as a message from the Models page, so it is right
    /// however the chat is reached -- the flyout, the back gesture, or the automatic
    /// return after choosing a model. It costs one request to a loopback server.
    /// </para>
    /// </summary>
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        _chatVisible = true;
        try
        {
            if (_languageReloadPending && await ReloadLanguageAsync())
                return;
            // First, because everything after it is pointless if the answer is no. iOS
            // suspends a WKWebView's content process the moment its view leaves the
            // window -- which every other screen in this app does -- and a suspended
            // process holding a phone with a multi-gigabyte model resident is the first
            // thing the system reclaims under pressure. When that happens the WebView
            // comes back BLANK and stays blank: MAUI does not implement
            // webViewWebContentProcessDidTerminate, so nothing reloads it. Asking the
            // page whether it is still there costs one round trip and is the difference
            // between a chat and a black rectangle.
            if (_pageReady && !await PageIsAliveAsync())
            {
                Console.WriteLine("TensorAgent: the page stopped answering; reloading it");
                _pageReady = false;
                _webView.Source = new UrlWebViewSource { Url = _host.EntryUrl };
                return;
            }

            // Guarded in JS as well: on the very first appearance the page may not have
            // loaded yet, and there is nothing to refresh until it has.
            await AnnounceNativeReadyAsync();
            await Tell("refreshModel");
            // Settings is a native page and this one outlives it, so a choice made
            // there — "Show reasoning by default", the dictation language — has to be
            // re-read on the way back. It used to be a switch the user could see under
            // the composer; now the only place it lives is the settings file.
            await Tell("refreshSettings");
            // A chat may have been renamed or deleted on the Chats page while this one
            // was covered, and the menu lists them.
            await Tell("refreshChats");
            // And the answer the model went on writing while this page was not being
            // shown -- and therefore not being read. See ChatTurnManager.
            await Tell("resumeTurn");
            await Tell("takeShare");
        }
        catch (Exception ex)
        {
            Console.WriteLine("TensorAgent: refresh on appearing failed: " + ex.Message);
        }
    }

    protected override void OnDisappearing()
    {
        _chatVisible = false;
        base.OnDisappearing();
    }

    private async Task<bool> ReloadLanguageAsync()
    {
        try
        {
            if (_pageReady && await PageIsAliveAsync())
            {
                string? kept = await Tell("prepareLanguageReload");
                if (string.Equals(kept?.Trim('"'), "busy", StringComparison.Ordinal))
                {
                    ScheduleLanguageReload();
                    return false;
                }
                if (!string.Equals(kept?.Trim('"'), "true", StringComparison.OrdinalIgnoreCase))
                {
                    Console.WriteLine("TensorAgent: language reload deferred because the composer could not be retained");
                    return false;
                }
            }
            _languageReloadPending = false;
            _pageReady = false;
            Console.WriteLine($"TensorAgent: reloading the page in {Loc.Language.Tag}");
            _webView.Source = new UrlWebViewSource { Url = _host.EntryUrl };
            return true;
        }
        catch (Exception ex)
        {
            _languageReloadPending = true;
            Console.WriteLine("TensorAgent: the page did not reload in the new language: " + ex.Message);
            return false;
        }
    }

    private void ScheduleLanguageReload()
    {
        if (_languageReloadRetryScheduled)
            return;
        _languageReloadRetryScheduled = true;
        Dispatcher.DispatchDelayed(TimeSpan.FromMilliseconds(500), async () =>
        {
            _languageReloadRetryScheduled = false;
            if (_chatVisible && _languageReloadPending)
                await ReloadLanguageAsync();
        });
    }

    /// <summary>
    /// Call one page-bridge method with an argument, without putting that argument
    /// anywhere a JavaScript parser will look at it.
    ///
    /// <para>
    /// MAUI's EvaluateJavaScriptAsync does not hand WKWebView the script: it wraps it as
    /// <c>try{JSON.stringify(eval('&lt;script&gt;'))}catch(e){'null'};</c>, which makes the
    /// script the body of a single-quoted literal. Serialised JSON spliced in there is
    /// read as literal escapes first and as JSON second, so a <c>\n</c> — which every
    /// text file with two lines produces — becomes a real newline inside a string that
    /// then never closes. eval throws, MAUI's own catch turns the throw into the string
    /// "null", and the caller sees a completed task. That is an upload that succeeded,
    /// an attachment that never appeared, and no message anywhere.
    /// </para>
    /// <para>
    /// Base64's alphabet has no quote, backslash or newline, so it crosses that wrapper
    /// intact. The page answers "ok", which is what makes a failure detectable at all.
    /// </para>
    /// </summary>
    /// <returns>The page's answer: "ok", or something else, or null when it never ran.</returns>
    private async Task<string?> CallBridgeAsync(string method, object argument)
    {
        string json = JsonSerializer.Serialize(argument, Core.Hosting.SseFraming.JsonOptions);
        string payload = Convert.ToBase64String(System.Text.Encoding.UTF8.GetBytes(json));
        string? answer = await _webView.EvaluateJavaScriptAsync(
            "window.TensorAgent && window.TensorAgent.__fromHost ? "
            + "window.TensorAgent.__fromHost('" + method + "','" + payload + "') : 'nobridge'");
        // MAUI answers a broken script with the STRING "null", so silence here is the
        // one thing that must not be read as success.
        if (answer is null || answer.Contains("ok", StringComparison.Ordinal))
            return answer;
        Console.WriteLine($"TensorAgent: the page refused {method}: {answer}");
        return answer;
    }

    /// <summary>Call one method on the page's bridge, if the page has one yet.</summary>
    private Task<string?> Tell(string method) => _webView.EvaluateJavaScriptAsync(
        $"window.TensorAgent && window.TensorAgent.{method} ? window.TensorAgent.{method}() : false");

    private Task<string?> AnnounceNativeReadyAsync() => CallBridgeAsync("nativeReady", new
    {
        dictation = Services.Dictation.IsSupported,
#if WINDOWS
        composerHint = Loc.T("app.composer.placeholderWindows"),
#else
        composerHint = Loc.T("app.composer.placeholder"),
#endif
    });

    /// <summary>
    /// The loopback server had to move to another port (see
    /// <see cref="Core.Hosting.LoopbackServer.EnsureListeningAsync"/>): the page's origin,
    /// and with it the token cookie, are gone. Navigate to the new entry URL, exactly as
    /// a dead page is reloaded. The page comes back to the same chat and the same
    /// running turn through /api/agent/launch, as any reload does.
    /// </summary>
    private void OnEntryUrlChanged(string url)
    {
        MainThread.BeginInvokeOnMainThread(() =>
        {
            Console.WriteLine("TensorAgent: the loopback server moved; reloading the page at " + url);
            _pageReady = false;
            _webView.Source = new UrlWebViewSource { Url = url };
        });
    }

    /// <summary>
    /// The host has checked (and, if it had to, rebuilt) its listener on the way back to
    /// the foreground. Now the page: it fired its own visibilitychange requests at about
    /// the same instant, possibly against a listener that was still being rebuilt and
    /// possibly on connections iOS had reclaimed, so it is nudged once more, a moment
    /// later, when the transport is known to be good. A page whose content process was
    /// reclaimed while the app was away answers nothing and is reloaded, as OnAppearing
    /// does for the same reason. And what the page saw is written into the trace,
    /// because it is the only record of it there will ever be.
    /// </summary>
    private void OnForegroundChecked(Core.Hosting.AgentAppHost.ForegroundReport report)
    {
        // A moved port is a reload (OnEntryUrlChanged), which carries everything below.
        if (report.EntryUrlChanged)
            return;
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            try
            {
                // After DidBecomeActive and after the page's own quarter-second deferral,
                // so this is the second, informed attempt rather than a race with the
                // first.
                await Task.Delay(700);
                if (!_pageReady)
                {
                    // The page never announced itself (its start chain failed, or it is
                    // still loading). Nothing here can be sent to it; say so, because
                    // otherwise a page in that state looks identical to a healthy one.
                    _host.App.TraceBackground("foreground: the page has not announced itself; nothing was nudged");
                    return;
                }
                if (!await PageIsAliveAsync())
                {
                    Console.WriteLine("TensorAgent: the page stopped answering while the app was away; reloading it");
                    _host.App.TraceBackground("foreground: the page did not answer; reloading it");
                    _pageReady = false;
                    _webView.Source = new UrlWebViewSource { Url = _host.EntryUrl };
                    return;
                }
                await Tell("resumeTurn");
                await Tell("takeShare");
                await Task.Delay(1500);
                await TracePageDiagnosticsAsync("foreground");
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: the foreground nudge failed: " + ex.Message);
            }
        });
    }

    /// <summary>
    /// Pull the page's own record of its transport events into logs/background.log.
    /// Bounded and tolerant: a page mid-resume may not answer in time, and that is
    /// worth one line, not a failure.
    /// </summary>
    private async Task TracePageDiagnosticsAsync(string when)
    {
        string line;
        try
        {
            string? answer = await _webView
                .EvaluateJavaScriptAsync(
                    "window.TensorAgent && window.TensorAgent.diagnostics ? window.TensorAgent.diagnostics() : 'nopage'")
                .WaitAsync(TimeSpan.FromSeconds(3));
            line = UnwrapEvaluated(answer);
        }
        catch (TimeoutException)
        {
            line = "(the page did not answer within 3 s)";
        }
        catch (Exception ex)
        {
            line = "(asking the page failed: " + ex.Message + ")";
        }
        // The TAIL. The diary ends with `events`, oldest first, so cutting the end
        // throws away exactly the events being traced -- the ones from the return.
        if (line.Length > 3000)
            line = "…" + line[^3000..];
        _host.App.TraceBackground($"page ({when}): {line}");
    }

    /// <summary>
    /// MAUI hands back the JSON.stringify of what the script returned, so a string
    /// arrives quoted and escaped. Unwrap it when it is one; pass anything else through.
    /// </summary>
    private static string UnwrapEvaluated(string? answer)
    {
        if (answer is null)
            return "null";
        // MAUI hands back the JSON.stringify of what the script returned, and then
        // trims the outer quotes itself -- so a returned STRING arrives with its inner
        // quotes still escaped, which is not JSON and cannot be read by anything. Put
        // the quotes back and decode it properly. Anything that is already valid JSON,
        // or is not a JSON string at all, is passed through untouched.
        string quoted = answer.Length >= 2 && answer[0] == '"' && answer[^1] == '"' ? answer : "\"" + answer + "\"";
        try { return JsonSerializer.Deserialize<string>(quoted) ?? answer; }
        catch (JsonException) { return answer; }
    }

#if DEBUG
    /// <summary>
    /// Device E2E hook (Debug builds only): the reported failure, driven through the
    /// real WebView. TENSORAGENT_DEMO_PROMPT sends a message the way a user does;
    /// scripts/verify-background.sh then sends the app away for long enough for iOS to
    /// suspend it (and, for a long enough absence, to reclaim its sockets) and brings
    /// it back. This traces, one 'pagecheck' line at a time, whether the page carried
    /// on: the characters on screen while the turn runs, whether the host's listener
    /// had to be rebuilt, and — the assertion the whole fix is for — whether the page
    /// shows the finished answer, in ONE bubble, and stops saying "working" within a
    /// minute of the turn ending.
    /// </summary>
    private async Task RunPageBackgroundProbeAsync()
    {
        if (!string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_PAGE_BACKGROUND_CHECK"), "1", StringComparison.Ordinal))
            return;

        Core.Hosting.AgentAppHost app = _host.App;
        void Say(string line)
        {
            Console.WriteLine("TensorAgent: pagecheck " + line);
            app.TraceBackground("pagecheck " + line);
        }

        try
        {
            if (Environment.GetEnvironmentVariable("TENSORAGENT_BACKGROUND_RUN") is { Length: > 0 } run)
                Say($"run {run}");
            if (string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORAGENT_DEMO_PROMPT")))
            {
                Say("FAIL TENSORAGENT_DEMO_PROMPT is not set, so nothing will be sent through the page");
                return;
            }

            // A clean share state, or the check measures the wrong thing: a share left in
            // the App Group inbox by an earlier run is applied the moment the turn ends --
            // as designed, into a NEW chat -- and the answer this watches vanishes from
            // the screen for a reason that has nothing to do with the background.
            try { await app.DrainSharedInboxAsync(); }
            catch (Exception ex) { Say("note: draining the share inbox first failed: " + ex.Message); }
            int discarded = 0;
            while (app.Shares.Peek() is { } head && app.DiscardPendingShare(head.Id))
                discarded++;
            if (discarded > 0)
                Say($"discarded {discarded} pending share(s) left by earlier runs");

            // The demo prompt waits for the model, then sends through the page's own
            // bridge; the turn it starts is the one this watches.
            if (!await Until(() => app.Turns.IsBusy, TimeSpan.FromMinutes(8)))
            {
                Say("FAIL no turn started within 8 minutes · " + app.Turns.Describe());
                return;
            }
            Say("the page's turn started · " + app.Turns.Describe());

            var clock = System.Diagnostics.Stopwatch.StartNew();
            while (app.Turns.IsBusy && clock.Elapsed < TimeSpan.FromMinutes(40))
            {
                await Task.Delay(3000);
                // The page can only be asked while the app is in front; while it is
                // away the answer would be silence, and silence would be the story.
                string page = "not asked (the app is away)";
                if (app.Compute.IsOpen)
                {
                    try
                    {
                        int shown = await AnswerLengthAsync().WaitAsync(TimeSpan.FromSeconds(2));
                        string? generating = await Tell("isGenerating").WaitAsync(TimeSpan.FromSeconds(2));
                        page = $"{shown} chars on screen, page generating={(generating ?? "?").Trim('"')}";
                    }
                    catch (Exception ex) { page = "the page did not answer: " + ex.GetType().Name; }
                }
                Say($"{clock.Elapsed.TotalSeconds:0}s: {page}, host frames {app.Turns.TotalFrames}, gate {(app.Compute.IsOpen ? "open" : "CLOSED")}, listener restarts {app.ListenerRestarts}");
            }
            Say("the turn ended · " + app.Turns.Describe());

            // The page's half: within a minute of the turn ending, the whole answer is on
            // screen in one bubble and the page no longer says it is working.
            int chars = 0, bubbles = 0;
            string idle = "?";
            bool caughtUp = await Until(async () =>
            {
                chars = await AnswerLengthAsync();
                idle = (await Tell("isGenerating") ?? "?").Trim('"');
                string? count = await _webView.EvaluateJavaScriptAsync("String(document.querySelectorAll('.turn.bot').length)");
                bubbles = int.TryParse((count ?? string.Empty).Trim('"'), out int n) ? n : -1;
                return chars > 0 && idle.Contains("false", StringComparison.OrdinalIgnoreCase);
            }, TimeSpan.FromSeconds(60));
            Say(caughtUp && bubbles == 1
                ? $"ok {chars} chars on screen in {bubbles} bubble after the turn ended, page idle"
                : $"FAIL after the turn ended: {chars} chars on screen, {bubbles} assistant bubble(s), page generating={idle}");
            await TracePageDiagnosticsAsync("pagecheck");
            Say($"done: listener restarts {app.ListenerRestarts}, rebuilt engine {app.EngineRebuilds} time(s)");
        }
        catch (Exception ex)
        {
            Say("FAIL " + ex.Message);
        }
    }
#endif

    /// <summary>Bring the persistent chat forward and nudge it to claim a new share.</summary>
    private void OnShareArrived()
    {
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            try
            {
                await AppShell.OpenAsync("main");
                if (_pageReady)
                    await Tell("takeShare");
                await OfferShareNotificationPermissionAsync();
            }
            catch (Exception ex)
            {
                Console.WriteLine("TensorAgent: share nudge failed: " + ex.Message);
            }
        });
    }

    /// <summary>
    /// Ask contextually, after the first share has arrived, whether future shares may
    /// post a one-tap notification. Permission is requested by the containing app,
    /// never by the extension running inside another app. iOS only: the share
    /// extension, and so the reason to ask, exists only there.
    /// </summary>
    private async Task OfferShareNotificationPermissionAsync()
    {
#if !IOS
        await Task.CompletedTask;
#else
        const string askedKey = "TensorAgentAskedForShareNotifications";
        if (string.Equals(Environment.GetEnvironmentVariable("TENSORAGENT_SHARE_CHECK"), "1", StringComparison.Ordinal)
            || NSUserDefaults.StandardUserDefaults.BoolForKey(askedKey))
            return;
        if (Interlocked.Exchange(ref _shareNotificationPrompting, 1) != 0)
            return;

        try
        {
            // Re-check inside the in-process gate: several imported envelopes can all
            // schedule this method before the first settings query finishes.
            if (NSUserDefaults.StandardUserDefaults.BoolForKey(askedKey))
                return;
            UNUserNotificationCenter center = UNUserNotificationCenter.Current;
            UNNotificationSettings settings = await center.GetNotificationSettingsAsync();
            if (settings.AuthorizationStatus != UNAuthorizationStatus.NotDetermined)
                return;

            NSUserDefaults.StandardUserDefaults.SetBool(true, askedKey);
            bool allow = await DisplayAlertAsync(
                Loc.T("app.shareNotifications.alert.title"),
                Loc.T("app.shareNotifications.alert.message"),
                Loc.T("app.shareNotifications.alert.allow"),
                Loc.T("app.shareNotifications.alert.notNow"));
            if (allow)
                await center.RequestAuthorizationAsync(UNAuthorizationOptions.Alert);
        }
        finally
        {
            Volatile.Write(ref _shareNotificationPrompting, 0);
        }
#endif
    }

    /// <summary>
    /// Whether the page is still running.
    ///
    /// <para>
    /// Only ever asked of a page that has ALREADY said it was ready, because a page
    /// which has not finished loading answers exactly the same "no" as one whose content
    /// process was killed — and reloading on the first answer would make a cold launch
    /// reload itself forever.
    /// </para>
    /// </summary>
    private async Task<bool> PageIsAliveAsync()
    {
        try
        {
            // Bounded, and silence counts as ALIVE. The probe runs the instant the chat
            // comes back, which is exactly when the content process is resuming and the
            // page's own visibilitychange handler is re-attaching to a running turn --
            // so an evaluation that has not answered yet is the normal case, not a dead
            // page. Treating "did not answer" as "dead" reloaded a perfectly good page
            // and threw away the turn it was showing. Only a definite 'no', or a hard
            // failure of the evaluation itself, means the page is gone.
            string? answer = await _webView
                .EvaluateJavaScriptAsync("window.TensorAgent ? 'yes' : 'no'")
                .WaitAsync(TimeSpan.FromSeconds(2));
            return answer is null || answer.Contains("yes", StringComparison.Ordinal);
        }
        catch (TimeoutException)
        {
            return true;
        }
        catch (Exception)
        {
            return false;
        }
    }

    /// <summary>
    /// Nothing, and deliberately so.
    ///
    /// <para>
    /// Two rows of native chrome used to live around this WebView: a status line with
    /// Chats / Models / Settings chips above it, and a row of Photo / Camera / Video /
    /// File / Speak chips below. Both said what the page already says — it has a menu
    /// and it has a "+" — while spending the one thing a 6.9-inch screen cannot spare.
    /// The page asks for all of it through <see cref="OnPageEvent"/> now.
    /// </para>
    /// </summary>
    private static View BuildAttachmentBar() => new ContentView { IsVisible = false, HeightRequest = 0 };

    private enum MediaSource { Library, Camera, Video, File }

    /// <summary>
    /// Pick something and give it to the chat service, then tell the page it now has an
    /// attachment.
    ///
    /// <para>
    /// Handed to the service DIRECTLY rather than posted to <c>/api/upload</c>, which is
    /// what this used to do and what produced "IMG_0004.jpeg could not be attached: no
    /// file was uploaded" for every photo. The loop went managed stream -> multipart
    /// body -> iOS's own NSURLSession HTTP stack -> a loopback socket -> a hand-written
    /// multipart parser, to move a file this process already had, to a server inside
    /// this same process; a body that arrives empty anywhere along that path produces
    /// exactly that message and names none of the five places it could have happened.
    /// The route still exists for the page's own paperclip, which is a browser and has
    /// no other way in. Native code does not need one.
    /// </para>
    /// <para>
    /// The pick is spooled to a file first, because a length is not optional — the
    /// service is given one, and the picker's stream is not always able to say. It is
    /// also where an empty pick becomes a sentence about an empty pick instead of a
    /// zero-byte upload the model is later asked to look at.
    /// </para>
    /// </summary>
    private async Task AttachAsync(MediaSource source)
    {
        FileResult? picked;
        try
        {
            picked = source switch
            {
                MediaSource.Library => await MediaPicker.Default.PickPhotoAsync(),
                MediaSource.Camera when MediaPicker.Default.IsCaptureSupported => await MediaPicker.Default.CapturePhotoAsync(),
                MediaSource.Camera => throw new NotSupportedException(Loc.T("app.attach.noCamera")),
                MediaSource.Video => await MediaPicker.Default.PickVideoAsync(),
                MediaSource.File => await FilePicker.Default.PickAsync(),
                _ => null,
            };
        }
        catch (Exception ex)
        {
            // Picker availability and permission errors happen before there is a file
            // to upload. This callback runs through an async UI event, so letting the
            // exception escape can close the app on a desktop without a camera.
            await Notice(Loc.T("app.attach.pickerFailed", ("reason", ex.Message)));
            return;
        }
        if (picked is null)
            return;

        // A pick without a name is "That file", which is a sentence of its own in each message.
        string? named = string.IsNullOrWhiteSpace(picked.FileName) ? null : picked.FileName;
        string spooled = Path.Combine(Path.GetTempPath(), "attach-" + Guid.NewGuid().ToString("N"));
        try
        {
            long length = await SpoolAsync(picked, spooled);
            Console.WriteLine(
                $"TensorAgent: attach {source} name='{picked.FileName}' type='{picked.ContentType}' bytes={length}");
            if (length == 0)
            {
                await Notice(named is null
                    ? Loc.T("app.attach.emptyUnnamed")
                    : Loc.T("app.attach.empty", ("name", named)));
                return;
            }

            // The name the service classifies by, resolved from what the pick actually
            // has: iOS's photo picker strips the extension, and the service decides what
            // an upload IS from that extension alone. See UploadNaming.
            string name = Core.Hosting.UploadNaming.ResolveFileName(picked.FileName, picked.ContentType, spooled);
            await using FileStream content = File.OpenRead(spooled);
            object result = await _host.App.Chat.UploadAsync(content, name, length, CancellationToken.None);

            // The page's own addAttachment takes exactly what /api/upload returns, so the
            // service's answer is forwarded verbatim rather than reshaped here.
            string? attached = await CallBridgeAsync("addAttachment", result);
            if (attached is null || !attached.Contains("ok", StringComparison.Ordinal))
            {
                // Uploaded and then lost between here and the composer. Worth its own
                // sentence: "nothing happened" is what this looked like for every
                // multi-line file before the bridge stopped splicing JSON into source.
                await Notice(named is null
                    ? Loc.T("app.attach.notAttachedUnnamed")
                    : Loc.T("app.attach.notAttached", ("name", named)));
            }
        }
        catch (TensorSharp.Chat.WebUiRequestRejectedException rejected)
        {
            // Every refusal carries a sentence written for a person, and that sentence is
            // the whole value of showing the failure at all.
            await Notice(CouldNotAttach(named, ReasonOf(rejected)));
        }
        catch (Exception ex)
        {
            await Notice(CouldNotAttach(named, ex.Message));
        }
        finally
        {
            try { if (File.Exists(spooled)) File.Delete(spooled); } catch (IOException) { /* temp */ }
        }
    }

    /// <summary>
    /// Copy a pick to <paramref name="destination"/> and say how many bytes it was.
    ///
    /// <para>
    /// Two ways in, because the picker has two. <see cref="FileResult.OpenReadAsync"/>
    /// is the supported one; its full path is the fallback, used only when the stream
    /// produced nothing, because a photo library asset can hand back a stream that is
    /// already at its end while the file behind it is perfectly readable.
    /// </para>
    /// </summary>
    private static async Task<long> SpoolAsync(FileResult picked, string destination)
    {
        long copied = await CopyAsync(async () => await picked.OpenReadAsync(), destination);
        if (copied > 0 || string.IsNullOrEmpty(picked.FullPath) || !File.Exists(picked.FullPath))
            return copied;

        Console.WriteLine("TensorAgent: attach the picker's stream was empty; reading " + picked.FullPath);
        return await CopyAsync(() => Task.FromResult<Stream>(File.OpenRead(picked.FullPath)), destination);
    }

    private static async Task<long> CopyAsync(Func<Task<Stream>> open, string destination)
    {
        try
        {
            await using Stream source = await open();
            await using FileStream target = File.Create(destination);
            await source.CopyToAsync(target);
            return target.Length;
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or NotSupportedException)
        {
            Console.WriteLine("TensorAgent: attach could not read the pick: " + ex.Message);
            return 0;
        }
    }

    /// <summary>The sentence inside a refusal payload, or the exception's own message.</summary>
    private static string ReasonOf(TensorSharp.Chat.WebUiRequestRejectedException rejected)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(
                JsonSerializer.Serialize(rejected.Payload, Core.Hosting.SseFraming.JsonOptions));
            if (document.RootElement.TryGetProperty("error", out JsonElement error)
                && error.GetString() is { Length: > 0 } message)
            {
                return message;
            }
        }
        catch (JsonException) { /* fall through to the exception's own words */ }
        return rejected.Message;
    }

    /// <summary>The notice for a pick that could not be attached, with its name or as "That file".</summary>
    private static string CouldNotAttach(string? name, string reason) => name is null
        ? Loc.T("app.attach.failedUnnamed", ("reason", reason))
        : Loc.T("app.attach.failed", ("name", name), ("reason", reason));

    /// <summary>
    /// Dictation, as a toggle. Speech recognition on iOS is a live session rather than
    /// a request, so the button starts it and the second tap ends it; partial results
    /// are ignored and only the final transcription reaches the composer, because
    /// appending partials would rewrite the user's text as they spoke.
    /// </summary>
    /// <summary>
    /// Start listening, because the user is holding the button down.
    ///
    /// <para>
    /// Hold-to-talk rather than a toggle: holding the message box turns the composer
    /// into one large button, and a press that lasts exactly as long as the speech is
    /// what a phone user expects from it. The session ends in
    /// <see cref="StopDictation"/> when the finger lifts, so nothing here waits for a
    /// result -- the transcription is delivered to the composer whenever it arrives.
    /// </para>
    /// </summary>
    private async Task StartDictationAsync()
    {
        if (_dictation is not null)
            return;
        // Cleared here, at the very start, because the finger that asked for this may
        // lift before the session exists.
        _dictationStopRequested = false;

        if (!Services.Dictation.IsSupported)
        {
            await Notice(Services.Dictation.UnsupportedMessage);
            await _webView.EvaluateJavaScriptAsync("window.TensorAgent.dictationEnded()");
            return;
        }
        if (await Services.Dictation.RequestPermissionsAsync() is { } refused)
        {
            // A permission iOS has already stored a "no" for cannot be asked for
            // again, so telling the user to try harder is useless: the only way back
            // is Settings, and the app can open it for them.
            bool permanent = refused.Contains(Services.Dictation.DeniedMarker, StringComparison.Ordinal);
            string message = refused.Replace(Services.Dictation.DeniedMarker, string.Empty).Trim();
            if (permanent)
                await NoticeWithSettings(message);
            else
                await Notice(message);
            await _webView.EvaluateJavaScriptAsync("window.TensorAgent.dictationEnded()");
            return;
        }

        _dictation = new Services.Dictation(_host.App.Settings.Load().SpeechLanguage);
        // The finger is very often already gone. On the first ever hold, iOS puts two
        // permission dialogs in front of the user, and tapping Allow means letting go of
        // the message box -- so `dictate-stop` arrives while this method is still inside
        // the await above, at which point StopDictation has nothing to stop. Starting
        // anyway would open the microphone and leave it open, because the release that
        // would have closed it has already happened.
        if (_dictationStopRequested)
        {
            _dictation.Dispose();
            _dictation = null;
            await _webView.EvaluateJavaScriptAsync("window.TensorAgent.dictationEnded()");
            return;
        }
        try
        {
            string text = await _dictation.ListenAsync();
            if (!string.IsNullOrWhiteSpace(text))
            {
                await CallBridgeAsync("insertText", new { text });
            }
        }
        catch (Exception ex)
        {
            await Notice(Loc.T("app.dictation.failed", ("reason", ex.Message)));
        }
        finally
        {
            _dictation?.Dispose();
            _dictation = null;
            await _webView.EvaluateJavaScriptAsync("window.TensorAgent.dictationEnded()");
        }
    }

    /// <summary>End the session the finger was holding open, or refuse the one that has
    /// not started yet.</summary>
    private void StopDictation()
    {
        _dictationStopRequested = true;
        try { _dictation?.Stop(); }
        catch (Exception ex) { Console.WriteLine("TensorAgent: stop dictation failed: " + ex.Message); }
    }

    /// <summary>Set when a release arrived before there was a session to release.</summary>
    private bool _dictationStopRequested;

    /// <summary>
    /// Show a message inside the page rather than as a native alert, so it looks the
    /// same as everything else the chat says.
    /// </summary>
    private async Task Notice(string text) => await CallBridgeAsync("notice", new { text, kind = "error" });

    /// <summary>
    /// The same notice, with a button that opens this app's page in iOS Settings.
    /// Used for a permission the user has already refused, where nothing the app does
    /// can ask again.
    /// </summary>
    private async Task NoticeWithSettings(string text) => await CallBridgeAsync("noticeWithSettings", new { text });

    /// <summary>Open Settings › TensorAgent, which is the only place these grants live.</summary>
    private static void OpenAppSettings()
    {
        try
        {
#if IOS || MACCATALYST
            var url = new Foundation.NSUrl(UIKit.UIApplication.OpenSettingsUrlString);
            UIKit.UIApplication.SharedApplication.OpenUrl(url, new UIKit.UIApplicationOpenUrlOptions(), null);
#else
            AppInfo.Current.ShowSettingsUI();
#endif
        }
        catch (Exception ex) { Console.WriteLine("TensorAgent: open settings failed: " + ex.Message); }
    }

    /// <summary>
    /// Point the page at a saved conversation, or at a brand new one.
    ///
    /// <para>
    /// A call into the page, NOT a navigation. It used to reload the WebView with the
    /// conversation in the query string, on the reasoning that the Web UI builds its
    /// whole state at load and reusing a loaded page would leave the previous chat's
    /// session and attachments behind. The page now rebuilds exactly those three things
    /// itself, and the reload was paying for that tidiness with everything else in the
    /// page: every request in flight, which on a phone includes the answer the model is
    /// halfway through writing.
    /// </para>
    /// </summary>
    public void OpenConversation(string? conversationId)
    {
        string argument = conversationId is { Length: > 0 } id ? JsonSerializer.Serialize(id) : "null";
        MainThread.BeginInvokeOnMainThread(async () =>
        {
            try
            {
                await _webView.EvaluateJavaScriptAsync(
                    "window.TensorAgent && window.TensorAgent.openConversation ? "
                    + "window.TensorAgent.openConversation(" + argument + ") : false");
            }
            catch (Exception ex) { Console.WriteLine("TensorAgent: open conversation failed: " + ex.Message); }
        });
    }

    /// <summary>
    /// What the page asks the app to do.
    ///
    /// <para>
    /// The page owns the chrome now, so the things only native code can do -- the
    /// camera, the photo library, the document picker, dictation, and leaving the
    /// chat for another route -- are reached by the page ASKING for them over the
    /// transport that already exists, rather than by a second row of native buttons
    /// duplicating the page's own "+". A WKWebView message handler would be the
    /// platform way; this keeps the client free of any iOS-specific API and is
    /// testable over plain HTTP.
    /// </para>
    /// </summary>
    private void OnPageEvent(string kind, System.Text.Json.JsonElement message)
    {
        switch (kind)
        {
            // The page has finished loading and is asking for nothing. This is where it
            // is told it is inside the app, and it has to be HERE rather than in
            // OnAppearing: on a cold launch the page appears before it exists, so the
            // call landed on nothing and `state.native` stayed false — which is a
            // composer whose long press answers "Voice input is only available in the
            // app" while running in the app, and a "+" that opens a browser file input
            // instead of the photo picker. The same hole reopened every time the WebView
            // was navigated to a saved conversation. The page emits this on every load,
            // so answering it is the one place that cannot be too early or too late.
            case "ready":
                _pageReady = true;
                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    try
                    {
                        await AnnounceNativeReadyAsync();
                        await Tell("takeShare");
                    }
                    catch (Exception ex) { Console.WriteLine("TensorAgent: nativeReady failed: " + ex.Message); }
                });
                return;

            case "open-models":
            case "open-route":
                string route = message.TryGetProperty("route", out System.Text.Json.JsonElement r)
                    ? r.GetString() ?? "models" : "models";
                MainThread.BeginInvokeOnMainThread(async () => await AppShell.OpenAsync(route));
                return;

            case "pick":
                string what = message.TryGetProperty("what", out System.Text.Json.JsonElement w)
                    ? w.GetString() ?? string.Empty : string.Empty;
                MediaSource? source = what switch
                {
                    "photo" => MediaSource.Library,
                    "camera" => MediaSource.Camera,
                    "video" => MediaSource.Video,
                    "file" => MediaSource.File,
                    _ => null,
                };
                if (source is { } picked)
                    MainThread.BeginInvokeOnMainThread(async () => await AttachAsync(picked));
                return;

            case "dictate-start":
                MainThread.BeginInvokeOnMainThread(async () => await StartDictationAsync());
                return;

            case "dictate-stop":
                MainThread.BeginInvokeOnMainThread(StopDictation);
                return;

            case "open-settings":
                MainThread.BeginInvokeOnMainThread(OpenAppSettings);
                return;

            // A tap on a file the model's code produced. The page cannot open one
            // itself: the route serves it as an attachment and a WKWebView with no
            // download delegate drops attachments silently, which is what left "here
            // is your PDF" leading nowhere.
            case "open-file":
                string fileUrl = message.TryGetProperty("url", out System.Text.Json.JsonElement u)
                    ? u.GetString() ?? string.Empty : string.Empty;
                string fileName = message.TryGetProperty("name", out System.Text.Json.JsonElement fn)
                    ? fn.GetString() ?? string.Empty : string.Empty;
                if (fileUrl.Length == 0)
                    return;
                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    try { await OpenArtifactAsync(PathOf(fileUrl), fileName); }
                    catch (Exception ex) { Console.WriteLine("TensorAgent: open-file failed: " + ex.Message); }
                });
                return;

            // Image results live under /uploads rather than the code artifact store.
            // A WebView cannot honor their download link, so present the native save UI.
            case "save-image":
                string imageUrl = message.TryGetProperty("url", out System.Text.Json.JsonElement image)
                    ? image.GetString() ?? string.Empty : string.Empty;
                MainThread.BeginInvokeOnMainThread(async () =>
                {
                    try { await SaveImageAsync(imageUrl); }
                    catch (Exception ex)
                    {
                        Console.WriteLine("TensorAgent: save-image failed: " + ex.Message);
                        await DisplayAlert(Loc.T("app.openFile.alert.title"), ex.Message, Loc.T("common.ok"));
                    }
                });
                return;
        }
    }

    private void StartHost()
    {
        try
        {
            // Subscribe before Start: LoopbackWebHost starts the durable share drain,
            // and a small text/URL envelope can arrive before Start returns.
            _host.App.PageEvent += OnPageEvent;
            _host.App.Shares.Arrived += OnShareArrived;
            // The host checks its own listener as the app comes back to the front;
            // this page owns the WebView, so the page-facing half is here.
            _host.App.ForegroundChecked += OnForegroundChecked;
            _host.App.EntryUrlChanged += OnEntryUrlChanged;
            _host.Start();
            EngineProbeResult probe = EngineProbe.Run();
            Console.WriteLine("TensorAgent: engine probe " + JsonSerializer.Serialize(probe, new JsonSerializerOptions(JsonSerializerDefaults.Web)));
#if DEBUG
            // Debug only: lets the simulator E2E harness curl the API from the Mac
            // (X-TensorAgent-Token). A release build keeps the token in-process.
            Console.WriteLine($"TensorAgent: entry URL {_host.EntryUrl}");
#endif
            _probe = probe;
            ShowStatus();
            // The page says when the model starts and stops working; the display is
            // held awake for exactly that stretch, because on iOS the screen sleeping
            // suspends the app and stops the generation partway.
            _webView.Source = new UrlWebViewSource { Url = _host.EntryUrl };
#if DEBUG
            // What the launch log cannot tell you otherwise: whether the interpreters
            // that linked can actually run, and whether the sandbox refuses what it
            // must. Debug only, off the UI thread, into a throwaway directory.
            _ = Task.Run(() =>
            {
                foreach (Core.Hosting.SelfTestResult check in _host.App.SelfTest())
                    Console.WriteLine("TensorAgent: selftest " + check);
            });
            _ = RunUploadProbeAsync();
            _ = RunShareProbeAsync();
            _ = RunBackgroundProbeAsync();
            _ = RunPageBackgroundProbeAsync();
            RunNetworkProbe();
            DownloadIfAsked();
#endif
        }
        catch (Exception ex)
        {
            // Say so once, in the UI and on stdout; there is no useful fallback
            // page without the loopback host.
            Console.WriteLine("TensorAgent: startup failed: " + ex);
            _startupFailure = ex.Message;
            ShowStatus();
        }
    }

    /// <summary>
    /// The status line, in the interface's language: written at construction, again once
    /// the host has started or failed to, and again whenever the language changes.
    /// </summary>
    private void ShowStatus()
    {
        if (_startupFailure is not null)
        {
            _status.TextColor = BarError;
            _status.LineBreakMode = LineBreakMode.WordWrap;
            _status.Text = Loc.T("app.status.startupFailed", ("reason", _startupFailure));
            return;
        }
        if (_probe is not { } probe)
        {
            _status.Text = Loc.T("app.status.starting");
            return;
        }

#if IOS || MACCATALYST
        string detail = probe.GpuName ?? Loc.T("app.status.noMetalDevice");
#else
        string detail = probe.Reason;
#endif
#if IOS
        _status.Text = probe.MainProgramHandleResolved
            ? Loc.T("app.status.engineLinked", ("backend", probe.Backend), ("detail", detail), ("port", _host.Port))
            : Loc.T("app.status.engineNotLinked", ("backend", probe.Backend), ("detail", detail), ("port", _host.Port));
#else
        _status.Text = probe.NativeLibraryLoaded
            ? Loc.T("app.status.engineLoaded", ("backend", probe.Backend), ("detail", detail), ("port", _host.Port))
            : Loc.T("app.status.engineNotLoaded", ("backend", probe.Backend), ("detail", detail), ("port", _host.Port));
#endif
    }
}
