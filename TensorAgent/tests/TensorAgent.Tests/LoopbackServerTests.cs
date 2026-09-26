using System.Globalization;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TensorAgent.Core.Hosting;

namespace TensorAgent.Tests;

public sealed class LoopbackServerTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "tensoragent-loop-" + Guid.NewGuid().ToString("N"));

    public LoopbackServerTests() => Directory.CreateDirectory(_dir);
    public void Dispose() { try { Directory.Delete(_dir, true); } catch { } }

    private static HttpClient Client() => new(new HttpClientHandler { UseCookies = true, CookieContainer = new CookieContainer() });

    [Fact]
    public async Task ServesStaticFilesJsonRoutesAndSse_WithTheTokenHandshake()
    {
        File.WriteAllText(Path.Combine(_dir, "index.html"), "<html>ui</html>");
        Directory.CreateDirectory(Path.Combine(_dir, "images"));
        File.WriteAllBytes(Path.Combine(_dir, "images", "logo.png"), new byte[] { 1, 2, 3 });
        using var server = new LoopbackServer { StaticRoot = _dir };
        server.MapGet("/api/models", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { loaded = (string?)null, models = Array.Empty<string>() })));
        server.MapGet("/api/sessions/{id}", (r, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { id = r.RouteValues["id"] })));
        server.MapGet("/api/code/artifacts/{run}/{*path}", (r, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Text(r.RouteValues["run"] + "|" + r.RouteValues["path"])));
        server.MapPost("/api/chat", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(Frames())));
        server.Start();

        using var http = Client();
        // No token, no cookie -> 403 even for static files.
        var denied = await http.GetAsync(server.BaseUrl + "/api/models");
        Assert.Equal(HttpStatusCode.Forbidden, denied.StatusCode);

        var entry = await http.GetAsync(server.EntryUrl);
        Assert.Equal(HttpStatusCode.OK, entry.StatusCode);
        // The page is served whole, with the app's companion script appended, so the
        // markup file itself carries no script and all behaviour stays in tensoragent.js.
        string served = await entry.Content.ReadAsStringAsync();
        Assert.StartsWith("<html>ui</html>", served, StringComparison.Ordinal);
        Assert.Contains("<script src=\"/tensoragent.js\"></script>", served, StringComparison.Ordinal);
        Assert.Contains("text/html", entry.Content.Headers.ContentType!.ToString());

        // Cookie now carries the token.
        string models = await http.GetStringAsync(server.BaseUrl + "/api/models");
        Assert.Equal("{\"loaded\":null,\"models\":[]}", models);
        var png = await http.GetAsync(server.BaseUrl + "/images/logo.png");
        Assert.Equal("image/png", png.Content.Headers.ContentType!.MediaType);
        Assert.Equal(new byte[] { 1, 2, 3 }, await png.Content.ReadAsByteArrayAsync());
        Assert.Equal("{\"id\":\"abc\"}", await http.GetStringAsync(server.BaseUrl + "/api/sessions/abc"));
        Assert.Equal("r1|a/b/c.txt", await http.GetStringAsync(server.BaseUrl + "/api/code/artifacts/r1/a/b/c.txt"));
        Assert.Equal(HttpStatusCode.NotFound, (await http.GetAsync(server.BaseUrl + "/nope")).StatusCode);
        Assert.Equal(HttpStatusCode.NotFound, (await http.GetAsync(server.BaseUrl + "/../etc/passwd")).StatusCode);

        // SSE: 'data: {json}\n\n' frames, streamed.
        using var chat = await http.SendAsync(new HttpRequestMessage(HttpMethod.Post, server.BaseUrl + "/api/chat") { Content = new StringContent("{}", Encoding.UTF8, "application/json") }, HttpCompletionOption.ResponseHeadersRead);
        Assert.Equal("text/event-stream", chat.Content.Headers.ContentType!.MediaType);
        string body = await chat.Content.ReadAsStringAsync();
        Assert.Equal("data: {\"token\":\"Hel\"}\n\ndata: {\"token\":\"lo\"}\n\ndata: {\"done\":true,\"error\":null}\n\n", body);
    }

    private static async IAsyncEnumerable<object> Frames()
    {
        yield return new { token = "Hel" };
        await Task.Delay(10);
        yield return new { token = "lo" };
        yield return new { done = true, error = (string?)null };
    }

    [Fact]
    public async Task HandlerExceptionsBecomeStatusPayloads()
    {
        using var server = new LoopbackServer { RequireToken = false };
        server.MapGet("/api/rejected", (_, _) => throw new LoopbackHttpException(404, new { error = "Session 'x' not found." }));
        server.MapGet("/api/boom", (_, _) => throw new InvalidOperationException("bad"));
        server.Start();
        using var http = Client();
        var rejected = await http.GetAsync(server.BaseUrl + "/api/rejected");
        Assert.Equal(HttpStatusCode.NotFound, rejected.StatusCode);
        using var doc = JsonDocument.Parse(await rejected.Content.ReadAsStringAsync());
        Assert.Equal("Session 'x' not found.", doc.RootElement.GetProperty("error").GetString());
        var boom = await http.GetAsync(server.BaseUrl + "/api/boom");
        Assert.Equal(HttpStatusCode.InternalServerError, boom.StatusCode);
    }

    [Fact]
    public async Task ParsesMultipartUploadsLikeTheWebUiSends()
    {
        using var server = new LoopbackServer { RequireToken = false };
        server.MapPost("/api/upload", async (r, ct) =>
        {
            Assert.True(r.HasFormContentType);
            using MultipartForm form = await r.ReadFormAsync(ct);
            MultipartFile file = Assert.Single(form.Files);
            byte[] bytes = await File.ReadAllBytesAsync(file.TempPath, ct);
            return LoopbackResponse.Json(new { file.FieldName, file.FileName, file.ContentType, file.Length, sha = Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(bytes)), overwrite = form["overwrite"] });
        });
        server.Start();

        byte[] payload = new byte[300_000];
        new Random(7).NextBytes(payload);
        // Make sure the body contains boundary-like sequences to exercise the search.
        Encoding.ASCII.GetBytes("\r\n--").CopyTo(payload, 1000);
        using var http = Client();
        using var content = new MultipartFormDataContent("----TensorAgentBoundary");
        content.Add(new StringContent("true"), "overwrite");
        var fileContent = new ByteArrayContent(payload);
        fileContent.Headers.ContentType = new MediaTypeHeaderValue("video/mp4");
        content.Add(fileContent, "file", "clip.mp4");
        var response = await http.PostAsync(server.BaseUrl + "/api/upload", content);
        Assert.Equal(HttpStatusCode.OK, response.StatusCode);
        using var doc = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.Equal("file", doc.RootElement.GetProperty("FieldName").GetString());
        Assert.Equal("clip.mp4", doc.RootElement.GetProperty("FileName").GetString());
        Assert.Equal("video/mp4", doc.RootElement.GetProperty("ContentType").GetString());
        Assert.Equal(payload.Length, doc.RootElement.GetProperty("Length").GetInt64());
        Assert.Equal(Convert.ToHexStringLower(System.Security.Cryptography.SHA256.HashData(payload)), doc.RootElement.GetProperty("sha").GetString());
        Assert.Equal("true", doc.RootElement.GetProperty("overwrite").GetString());
    }

    [Fact]
    public void RoutePatternsMatchMinimalApiShapes()
    {
        Assert.True(new RoutePattern("/api/sessions/{id}").TryMatch("/api/sessions/abc", out var v) && v["id"] == "abc");
        Assert.False(new RoutePattern("/api/sessions/{id}").TryMatch("/api/sessions", out _));
        Assert.False(new RoutePattern("/api/sessions/{id}").TryMatch("/api/sessions/a/b", out _));
        Assert.True(new RoutePattern("/api/skills/{name}/files/{*path}").TryMatch("/api/skills/pdf/files/scripts/a.py", out v) && v["path"] == "scripts/a.py");
        Assert.True(new RoutePattern("/").TryMatch("/", out _));
        Assert.False(new RoutePattern("/").TryMatch("/x", out _));
    }

    // ---- a reader that goes away ----------------------------------------------------

    [Fact]
    public async Task AClientThatDisconnectsStopsTheGenerationItStarted()
    {
        // Nobody is listening and the model generates to the end of its budget anyway:
        // minutes of a phone's battery and all of its memory bandwidth spent on an
        // answer no one will ever read. A failed write is the only place a disconnect
        // surfaces at all, so that is where the producer has to be cancelled.
        //
        // Asynchronous continuations, because this is completed inside the server's own
        // Cancel() call: completed synchronously there, it would run the rest of this
        // test on the request's own thread and the request would never end.
        var cancelled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var server = new LoopbackServer { RequireToken = false };
        server.MapGet("/api/chat-stream", (request, ct) =>
            Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(Generation(ct), request.Cancellation)));
        server.Start();

        using var reader = await RawStream.OpenAsync(server, "/api/chat-stream");
        string opening = await reader.ReadUntilAsync(body => body.Contains("data: ", StringComparison.Ordinal));
        Assert.Contains("text/event-stream", reader.Raw, StringComparison.Ordinal);
        Assert.Contains("data: ", opening, StringComparison.Ordinal);

        reader.Drop();

        await Completes(cancelled.Task,
            "the client is gone and the producer is still running: a dropped connection has to cancel the request it started");

        // A generation that keeps talking, so the disconnect is found by a frame that
        // cannot be written rather than by a keep-alive. The frame that already arrived
        // is what says it had started.
        async IAsyncEnumerable<object> Generation(CancellationToken ct)
        {
            using CancellationTokenRegistration stop = ct.Register(() => cancelled.TrySetResult());
            for (int i = 0; i < 100_000; i++)
            {
                yield return new { token = new string('t', 256) };
                await Task.Delay(5, ct);
            }
        }
    }

    [Fact]
    public async Task AStreamThatGoesQuietSaysSoWithCommentsRatherThanWithFrames()
    {
        // A prefill can hold a stream silent for minutes, and silence is the one state
        // in which a reader that left cannot be noticed: nothing is written, so nothing
        // fails. The comment exists to give the stream something to write.
        using var server = new LoopbackServer { RequireToken = false };
        server.MapGet("/api/chat-stream", (request, ct) =>
            Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                OneFrameThenSilence(ct), request.Cancellation, keepAlive: TimeSpan.FromMilliseconds(120))));
        server.Start();

        using var reader = await RawStream.OpenAsync(server, "/api/chat-stream");
        string stream = await reader.ReadUntilAsync(text => Occurrences(text, ": keep-alive") >= 3);

        Assert.True(Occurrences(stream, ": keep-alive") >= 3,
            $"a stream quiet for three keep-alive periods wrote no keep-alive:\n{stream}");

        // And what the page reads is unchanged. Its reader takes 'data: ' lines and
        // ignores everything else, so a keep-alive that arrived as a frame would be
        // parsed as JSON and land in the answer.
        string[] frames = stream.Split('\n')
            .Where(line => line.StartsWith("data: ", StringComparison.Ordinal))
            .Select(line => line.TrimEnd('\r'))
            .ToArray();
        Assert.Equal(new[] { "data: {\"token\":\"first\"}" }, frames);

        // Every stream above passes its own interval, so the one the app actually runs
        // with is asserted here: a keep-alive quietly disabled in production would
        // leave all of these green.
        Assert.InRange(LoopbackResponse.DefaultKeepAlive, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(15));
    }

    [Fact]
    public async Task AClientThatLeavesDuringALongPrefillStopsTheGenerationRatherThanWaitingForATokenToFail()
    {
        // The same disconnect as the first test, in the state where the stream has
        // nothing to say. This is the case that hangs without a keep-alive: the first
        // token can be minutes away, and until something is written nothing tells the
        // server that the page it is generating for has closed.
        //
        // See the first test for why the continuations are asynchronous: completed on
        // the request's own thread, these would keep the request from ever finishing.
        var cancelled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var prefillOver = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var log = new RecordingLogger();
        using var server = new LoopbackServer(log) { RequireToken = false };
        server.MapGet("/api/chat-stream", (request, ct) =>
            Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                Prefill(ct), request.Cancellation, keepAlive: TimeSpan.FromMilliseconds(120))));
        server.Start();

        using var reader = await RawStream.OpenAsync(server, "/api/chat-stream");
        string quiet = await reader.ReadUntilAsync(text => text.Contains(": keep-alive", StringComparison.Ordinal));
        Assert.Contains(": keep-alive", quiet, StringComparison.Ordinal);

        reader.Drop();

        await Completes(cancelled.Task,
            "a reader that left while the stream was quiet was never noticed: the phone is still generating for it");

        // And it ended as what it is, rather than as a server error. The prefill is let
        // go first and the server disposed second, because disposing waits for the
        // request: by the time this reads the log, everything that request had to say
        // has been said.
        prefillOver.TrySetResult();
        server.Dispose();
        Assert.True(log.Entries.Count == 0,
            "a reader that walked away was reported as a failed request: " + string.Join(" | ", log.Entries));

        async IAsyncEnumerable<object> Prefill(CancellationToken ct)
        {
            using CancellationTokenRegistration stop = ct.Register(() => cancelled.TrySetResult());
            yield return new { token = "first" };
            // Ended by the test, and not by the cancellation, on purpose. A prefill is
            // the work that notices a cancellation only when it next looks up — between
            // tokens, or when the graph compute it is inside returns — so the stream is
            // torn down here with a pull still running, which is the hard case. It also
            // means no token can arrive during the assertion above and pass it for the
            // wrong reason.
            await prefillOver.Task;
            ct.ThrowIfCancellationRequested();
            yield return new { done = true, error = (string?)null };
        }
    }

    /// <summary>One frame, then the silence of a model reading a long prompt.</summary>
    private static async IAsyncEnumerable<object> OneFrameThenSilence(CancellationToken ct)
    {
        yield return new { token = "first" };
        await Task.Delay(Timeout.InfiniteTimeSpan, ct);
        yield return new { done = true, error = (string?)null };
    }

    /// <summary>Wait for something that must happen, and say what it means when it does not.</summary>
    private static Task Completes(Task task, string because) => Completes(task, TimeSpan.FromSeconds(20), because);

    private static async Task Completes(Task task, TimeSpan within, string because)
    {
        Task first = await Task.WhenAny(task, Task.Delay(within));
        Assert.True(ReferenceEquals(first, task), because);
        await task;
    }

    /// <summary>Every wait on the network in the suspension tests is bounded by this, so a regression fails in seconds.</summary>
    private static readonly TimeSpan Bound = TimeSpan.FromSeconds(10);

    private static int Occurrences(string text, string needle)
    {
        int count = 0;
        for (int at = text.IndexOf(needle, StringComparison.Ordinal); at >= 0;
             at = text.IndexOf(needle, at + needle.Length, StringComparison.Ordinal))
        {
            count++;
        }
        return count;
    }

    /// <summary>
    /// Whatever the server thought worth a warning or worse.
    ///
    /// <para>
    /// A reader that goes away is an ordinary ending, and the difference between
    /// handling one and merely surviving it shows up nowhere else: the producer is
    /// cancelled either way, and only the log says whether the request that was
    /// serving it ended as a client leaving or as a server error.
    /// </para>
    /// </summary>
    private sealed class RecordingLogger : ILogger
    {
        public List<string> Entries { get; } = new();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel level) => level >= LogLevel.Warning;

        public void Log<TState>(
            LogLevel level, EventId id, TState state, Exception? error, Func<TState, Exception?, string> format)
        {
            if (!IsEnabled(level))
                return;
            lock (Entries)
                Entries.Add($"{level}: {format(state, error)}{(error is null ? string.Empty : " / " + error.GetType().Name)}");
        }
    }

    /// <summary>
    /// One connection, driven by hand.
    ///
    /// <para>
    /// <see cref="HttpClient"/> is the wrong instrument for these tests: it decides for
    /// itself when a connection is really finished with, and what is under test is what
    /// the server does the instant it is not. This sends the request, reads the event
    /// stream the way the page does, and can vanish mid-stream the way a WebView does
    /// when the app is suspended — abruptly, so that the server's next write fails now
    /// rather than whenever the socket would otherwise time out.
    /// </para>
    /// </summary>
    private sealed class RawStream : IDisposable
    {
        private readonly TcpClient _tcp;
        private readonly StringBuilder _received = new();

        private RawStream(TcpClient tcp) => _tcp = tcp;

        public static async Task<RawStream> OpenAsync(LoopbackServer server, string path)
        {
            var tcp = new TcpClient();
            await tcp.ConnectAsync(IPAddress.Loopback, server.Port);
            byte[] request = Encoding.ASCII.GetBytes(
                $"GET {path} HTTP/1.1\r\nHost: 127.0.0.1:{server.Port}\r\nAccept: text/event-stream\r\n\r\n");
            await tcp.GetStream().WriteAsync(request);
            return new RawStream(tcp);
        }

        /// <summary>Everything that arrived, headers and chunk framing included.</summary>
        public string Raw => _received.ToString();

        /// <summary>
        /// Read until <paramref name="enough"/> is satisfied by the body, then hand the
        /// body back — including when it never was, so the assertion can print what the
        /// server sent instead.
        /// </summary>
        public async Task<string> ReadUntilAsync(Func<string, bool> enough)
        {
            byte[] buffer = new byte[4096];
            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(20));
            try
            {
                while (!enough(Body()))
                {
                    int read = await _tcp.GetStream().ReadAsync(buffer, deadline.Token);
                    if (read == 0)
                        break;
                    _received.Append(Encoding.UTF8.GetString(buffer, 0, read));
                }
            }
            catch (OperationCanceledException)
            {
                // Out of time; the caller asserts on what did arrive.
            }
            return Body();
        }

        /// <summary>
        /// The bytes a reader of the event stream sees: past the headers and with the
        /// chunk framing removed.
        ///
        /// <para>
        /// It has to be undone by hand because the framing is an artefact of the
        /// transport, not of the protocol under test — the server flushes each piece of
        /// a frame separately, so 'data: ', the JSON and the blank line arrive as three
        /// chunks with a length line between them, and a test that asserted on those
        /// would be asserting on chunk boundaries. Counting characters as bytes is safe
        /// here and only here: every frame these tests send is ASCII.
        /// </para>
        /// </summary>
        private string Body()
        {
            string raw = _received.ToString();
            int headers = raw.IndexOf("\r\n\r\n", StringComparison.Ordinal);
            if (headers < 0)
                return string.Empty;

            var body = new StringBuilder();
            int at = headers + 4;
            while (at < raw.Length)
            {
                int eol = raw.IndexOf("\r\n", at, StringComparison.Ordinal);
                if (eol < 0)
                    break;
                if (!int.TryParse(raw.AsSpan(at, eol - at), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out int size)
                    || size == 0)
                {
                    break;
                }
                int from = eol + 2;
                if (from + size > raw.Length)
                    break;
                body.Append(raw, from, size);
                at = from + size + 2;
            }
            return body.ToString();
        }

        /// <summary>
        /// Vanish. The zero linger makes it a reset rather than a polite close, which
        /// is what the server sees when the page holding the connection is gone.
        /// </summary>
        public void Drop()
        {
            _tcp.Client.LingerState = new LingerOption(true, 0);
            _tcp.Close();
        }

        /// <summary>
        /// Wait for the SERVER to end the connection: a read that returns nothing, or
        /// a reset. True when it did within <paramref name="within"/>; false when the
        /// stream is still open, which is the failure the restart tests look for.
        /// </summary>
        public async Task<bool> EndsAsync(TimeSpan within)
        {
            byte[] buffer = new byte[4096];
            using var deadline = new CancellationTokenSource(within);
            try
            {
                while (true)
                {
                    int read = await _tcp.GetStream().ReadAsync(buffer, deadline.Token);
                    if (read == 0)
                        return true;
                    _received.Append(Encoding.UTF8.GetString(buffer, 0, read));
                }
            }
            catch (OperationCanceledException)
            {
                return false;
            }
            catch (Exception ex) when (ex is IOException or SocketException or ObjectDisposedException)
            {
                return true;
            }
        }

        public void Dispose() => _tcp.Dispose();
    }

    [Fact]
    public void SseFramingMatchesTheServersSseWriter()
    {
        // The Server writes: "data: " + JsonSerializer.Serialize(payload) + "\n\n" with default options.
        object frame = new { token = "hi", n = (int?)null };
        Assert.Equal("data: " + JsonSerializer.Serialize(frame) + "\n\n", SseFraming.Format(frame));
    }

    // ---- surviving a suspension ------------------------------------------------------
    //
    // iOS reclaims a suspended app's sockets, the listening one included, and the
    // managed HttpListener shows nothing of it afterwards: IsListening stays true,
    // GetContextAsync never returns, nothing is logged, and every connect from the
    // page is refused. These do to the listening socket exactly what the kernel does
    // (ListeningSockets.Kill: the socket object is closed under the listener, which is
    // NOT disposed) and check that the server notices, rebuilds itself on the port it
    // had, ends what the dead incarnation was still serving, and moves — saying so —
    // only when the old port cannot be had again.

    [Fact]
    public async Task AHealthyListenerAnswersTheProbeAndEnsureListeningIsANoOp()
    {
        // The check runs on every return to the foreground, so on a healthy listener
        // it has to be free of side effects: no restart, no port change, no warning,
        // and a stream that was open across it is the same stream afterwards.
        var log = new RecordingLogger();
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var server = new LoopbackServer(log) { RequireToken = false };
        server.MapGet("/api/ping", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { pong = true })));
        server.MapGet("/api/chat-stream", (request, ct) =>
            Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                FirstThenWhenReleased(ct), request.Cancellation, keepAlive: TimeSpan.FromMilliseconds(120))));
        server.Start();
        int port = server.Port;

        LoopbackServer.ListenerProbe probe = await server.ProbeAsync(TimeSpan.FromSeconds(2));
        Assert.True(probe.Alive, "a listener that has just started did not answer its own probe: " + probe.Detail);
        Assert.Equal("answered", probe.Detail);

        using var reader = await RawStream.OpenAsync(server, "/api/chat-stream");
        string opening = await reader.ReadUntilAsync(body => body.Contains("data: {\"token\":\"first\"}", StringComparison.Ordinal));
        Assert.Contains("data: {\"token\":\"first\"}", opening, StringComparison.Ordinal);

        LoopbackServer.ListenerReport report = await server.EnsureListeningAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(LoopbackServer.ListenerHealth.Alive, report.Health);
        Assert.StartsWith("answered in ", report.Detail, StringComparison.Ordinal);
        Assert.Equal(0, server.Restarts);
        Assert.Equal(port, server.Port);
        Assert.Equal(port, report.Port);

        // The stream that was open across the check still breathes, and the frame
        // released after it arrives on the same connection.
        string quiet = await reader.ReadUntilAsync(body => body.Contains(": keep-alive", StringComparison.Ordinal));
        Assert.Contains(": keep-alive", quiet, StringComparison.Ordinal);
        release.TrySetResult();
        string rest = await reader.ReadUntilAsync(body => body.Contains("data: {\"token\":\"second\"}", StringComparison.Ordinal));
        Assert.Contains("data: {\"token\":\"second\"}", rest, StringComparison.Ordinal);

        using HttpClient http = Client();
        Assert.Equal("{\"pong\":true}", await http.GetStringAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound));
        Assert.True(log.Entries.Count == 0, "a healthy listener was warned about: " + string.Join(" | ", log.Entries));

        async IAsyncEnumerable<object> FirstThenWhenReleased(CancellationToken ct)
        {
            yield return new { token = "first" };
            await release.Task.WaitAsync(ct);
            yield return new { token = "second" };
        }
    }

    [SkippableFact]
    public async Task AListenerWhoseSocketWasReclaimedIsReboundOnTheSamePortAndServesAgain()
    {
        Skip.If(!ListeningSockets.ManagedHttpListenerInUse, ListeningSockets.WhyNotManaged);
        var log = new RecordingLogger();
        using var server = new LoopbackServer(log) { RequireToken = false };
        server.MapGet("/api/ping", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { pong = true })));
        server.Start();
        int port = server.Port;

        using (HttpClient before = Client())
            Assert.Equal("{\"pong\":true}", await before.GetStringAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound));

        ListeningSockets.Kill(port);

        // What the page sees from here on: every connect refused, and the listener
        // itself none the wiser.
        using (var tcp = new TcpClient())
        using (var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(2)))
        {
            SocketException refused = await Assert.ThrowsAsync<SocketException>(
                () => tcp.ConnectAsync(IPAddress.Loopback, port, deadline.Token).AsTask());
            Assert.Equal(SocketError.ConnectionRefused, refused.SocketErrorCode);
        }

        LoopbackServer.ListenerProbe probe = await server.ProbeAsync(TimeSpan.FromSeconds(2));
        Assert.False(probe.Alive, "the probe was answered on a port whose listening socket is closed: " + probe.Detail);
        Assert.Contains("ConnectionRefused", probe.Detail, StringComparison.Ordinal);

        LoopbackServer.ListenerReport report = await server.EnsureListeningAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(LoopbackServer.ListenerHealth.Restarted, report.Health);
        Assert.Equal(port, server.Port);
        Assert.Equal(port, report.Port);
        Assert.Equal(1, server.Restarts);
        Assert.Contains("ConnectionRefused", report.Detail, StringComparison.Ordinal);
        Assert.Contains($"rebound on port {port}", report.Detail, StringComparison.Ordinal);

        // A new client, because the old one's pooled connection belonged to the
        // incarnation that was just retired.
        using (HttpClient after = Client())
            Assert.Equal("{\"pong\":true}", await after.GetStringAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound));

        LoopbackServer.ListenerProbe again = await server.ProbeAsync(TimeSpan.FromSeconds(2));
        Assert.True(again.Alive, "the rebuilt listener does not answer its probe: " + again.Detail);

        string warning = Assert.Single(log.Entries);
        Assert.StartsWith("Warning: ", warning, StringComparison.Ordinal);
        Assert.Contains("loopback listener restarted", warning, StringComparison.Ordinal);
        Assert.Contains($"rebound on port {port}", warning, StringComparison.Ordinal);
    }

    [SkippableFact]
    public async Task AnInFlightStreamOfADeadListenerIsEndedByTheRestartNotLeftRunning()
    {
        Skip.If(!ListeningSockets.ManagedHttpListenerInUse, ListeningSockets.WhyNotManaged);
        // A request the dead incarnation accepted is still running when the listener
        // is rebuilt; the managed listener would let it write to its closed response
        // stream, silently, for as long as its producer ran. The restart retires the
        // epoch instead, so the producer is cancelled now and the reader sees the end.
        var cancelled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        using var server = new LoopbackServer { RequireToken = false };
        server.MapGet("/api/chat-stream", (request, ct) =>
            Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(
                Prefill(ct), request.Cancellation, keepAlive: TimeSpan.FromMilliseconds(120))));
        server.Start();

        using var reader = await RawStream.OpenAsync(server, "/api/chat-stream");
        string opening = await reader.ReadUntilAsync(body => body.Contains("data: ", StringComparison.Ordinal));
        Assert.Contains("data: ", opening, StringComparison.Ordinal);

        // The listening socket goes; the established socket under the stream is a
        // different one and stays, exactly as after a suspension.
        ListeningSockets.Kill(server.Port);
        Assert.False(cancelled.Task.IsCompleted, "losing the listening socket alone must not end an established stream");

        LoopbackServer.ListenerReport report = server.Restart("test");
        Assert.Equal(LoopbackServer.ListenerHealth.Restarted, report.Health);
        Assert.Equal(1, server.Restarts);

        await Completes(cancelled.Task, TimeSpan.FromSeconds(5),
            "the listener was rebuilt and the producer of the stream it was serving is still running: the retired epoch was not cancelled");
        Assert.True(await reader.EndsAsync(TimeSpan.FromSeconds(5)),
            "the stream of the retired listener is still open to its reader after the restart");

        async IAsyncEnumerable<object> Prefill(CancellationToken ct)
        {
            using CancellationTokenRegistration stop = ct.Register(() => cancelled.TrySetResult());
            yield return new { token = "first" };
            await Task.Delay(Timeout.InfiniteTimeSpan, ct);
            yield return new { done = true, error = (string?)null };
        }
    }

    [SkippableFact]
    public async Task WhenTheOldPortCannotBeReboundTheServerMovesAndSaysSo()
    {
        Skip.If(!ListeningSockets.ManagedHttpListenerInUse, ListeningSockets.WhyNotManaged);
        var log = new RecordingLogger();
        using var server = new LoopbackServer(log);
        server.MapGet("/", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Text("ui")));
        server.MapGet("/api/ping", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { pong = true })));
        server.Start();
        int oldPort = server.Port;
        string token = server.Token;
        var relisted = new List<int>();
        server.Relisted += relisted.Add;

        ListeningSockets.Kill(oldPort);
        // Somebody else has the port by the time the app is back: Stop()+Start() on
        // the same instance and a fresh listener on that port both fail with
        // address-in-use, so the server has to move.
        var squatter = new TcpListener(IPAddress.Loopback, oldPort);
        squatter.Start();
        try
        {
            LoopbackServer.ListenerReport report = await server.EnsureListeningAsync(TimeSpan.FromSeconds(1));
            Assert.Equal(LoopbackServer.ListenerHealth.Relisted, report.Health);
            Assert.NotEqual(oldPort, server.Port);
            Assert.Equal(server.Port, report.Port);
            Assert.Equal(new[] { server.Port }, relisted);
            Assert.Equal(1, server.Restarts);
            Assert.Contains($"moved from port {oldPort} to {server.Port}", report.Detail, StringComparison.Ordinal);
            Assert.Equal($"http://127.0.0.1:{server.Port}", server.BaseUrl);
            Assert.Equal($"http://127.0.0.1:{server.Port}/?token={token}", server.EntryUrl);
            Assert.Equal(token, server.Token);

            // The new origin is served, with the same handshake as the first one.
            using HttpClient http = Client();
            Assert.Equal(HttpStatusCode.Forbidden, (await http.GetAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound)).StatusCode);
            Assert.Equal(HttpStatusCode.OK, (await http.GetAsync(server.EntryUrl).WaitAsync(Bound)).StatusCode);
            Assert.Equal("{\"pong\":true}", await http.GetStringAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound));

            string warning = Assert.Single(log.Entries);
            Assert.Contains($"moved from port {oldPort} to {server.Port}", warning, StringComparison.Ordinal);
        }
        finally
        {
            squatter.Dispose();
        }
    }

    [Fact]
    public async Task ARestartedServerStillRefusesRequestsWithoutTheToken()
    {
        // The rebuilt listener is the same server: the per-launch secret is unchanged,
        // a request without it is still refused, and the entry URL still sets the cookie.
        using var server = new LoopbackServer();
        server.MapGet("/", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Text("ui")));
        server.MapGet("/api/ping", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { pong = true })));
        server.Start();
        int port = server.Port;
        string token = server.Token;

        LoopbackServer.ListenerReport report = server.Restart("test");
        Assert.Equal(LoopbackServer.ListenerHealth.Restarted, report.Health);
        Assert.Equal(port, server.Port);
        Assert.Equal(token, server.Token);

        using HttpClient http = Client();
        HttpResponseMessage denied = await http.GetAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound);
        Assert.Equal(HttpStatusCode.Forbidden, denied.StatusCode);

        using (var wrong = new HttpRequestMessage(HttpMethod.Get, server.BaseUrl + "/api/ping"))
        {
            wrong.Headers.Add("Cookie", LoopbackServer.TokenCookie + "=not-the-token");
            Assert.Equal(HttpStatusCode.Forbidden, (await http.SendAsync(wrong).WaitAsync(Bound)).StatusCode);
        }

        HttpResponseMessage entry = await http.GetAsync(server.EntryUrl).WaitAsync(Bound);
        Assert.Equal(HttpStatusCode.OK, entry.StatusCode);
        Assert.Contains(entry.Headers.GetValues("Set-Cookie"),
            cookie => cookie.StartsWith(LoopbackServer.TokenCookie + "=" + token, StringComparison.Ordinal));
        Assert.Equal("{\"pong\":true}", await http.GetStringAsync(server.BaseUrl + "/api/ping").WaitAsync(Bound));
    }

    /// <summary>
    /// The phone's own trace, 2026-09-09 19:39: the listener was rebuilt on the way
    /// back to the foreground, the page re-attached to the running turn, received the
    /// frames buffered so far, and then nothing more -- while the host kept producing.
    /// The server side of that is pinned here: a stream opened AFTER a restart must keep
    /// delivering frames that are produced after it was opened, not only the backlog.
    /// </summary>
    [SkippableFact]
    public async Task AStreamOpenedAfterARestartKeepsDeliveringNewFrames()
    {
        Skip.IfNot(ListeningSockets.ManagedHttpListenerInUse, "the managed HttpListener is what iOS uses");
        var log = new RecordingLogger();
        using var server = new LoopbackServer(log) { RequireToken = false };
        var produce = new SemaphoreSlim(0);
        server.MapGet("/api/stream", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Sse(Slowly(produce))));
        server.Start();

        ListeningSockets.Kill(server.Port);
        LoopbackServer.ListenerReport report = server.Restart("test");
        Assert.Equal(LoopbackServer.ListenerHealth.Restarted, report.Health);

        using RawStream stream = await RawStream.OpenAsync(server, "/api/stream");
        produce.Release(3);
        string backlog = await stream.ReadUntilAsync(body => Occurrences(body, "data:") >= 3);
        Assert.Equal(3, Occurrences(backlog, "data:"));

        // Frames produced AFTER the stream was opened on the rebuilt listener.
        produce.Release(4);
        string more = await stream.ReadUntilAsync(body => Occurrences(body, "data:") >= 7);
        Assert.True(Occurrences(more, "data:") >= 7, "the stream opened after the restart stopped delivering: " + more);
        Assert.DoesNotContain(log.Entries, e => e.StartsWith("Error", StringComparison.Ordinal));

        static async IAsyncEnumerable<object> Slowly(SemaphoreSlim produce, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken ct = default)
        {
            for (int i = 0; i < 7; i++)
            {
                await produce.WaitAsync(ct);
                yield return new { token = "t" + i };
            }
            yield return new { done = true, error = (string?)null };
        }
    }

}
