// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using System.Net;
using System.Net.Sockets;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

namespace TensorAgent.Core.Hosting;

/// <summary>Everything a route handler needs about one request.</summary>
public sealed class LoopbackRequest
{
    internal LoopbackRequest(HttpListenerContext context, string path, IReadOnlyDictionary<string, string> routeValues)
    {
        Context = context;
        Path = path;
        RouteValues = routeValues;
    }

    public HttpListenerContext Context { get; }
    public HttpListenerRequest Raw => Context.Request;
    public string Method => Context.Request.HttpMethod;
    /// <summary>Decoded absolute path without the query string.</summary>
    public string Path { get; }
    public IReadOnlyDictionary<string, string> RouteValues { get; }
    public string? Query(string name) => Context.Request.QueryString[name];
    public CancellationToken Aborted { get; internal set; }

    /// <summary>
    /// This request's own cancellation source. A streaming handler passes it to
    /// <see cref="LoopbackResponse.Sse"/> so the producer is stopped the moment a
    /// write shows the client has gone.
    /// </summary>
    public CancellationTokenSource? Cancellation { get; internal set; }

    public async Task<JsonElement> ReadJsonAsync(CancellationToken ct)
    {
        using var doc = await JsonDocument.ParseAsync(Context.Request.InputStream, cancellationToken: ct).ConfigureAwait(false);
        return doc.RootElement.Clone();
    }

    public async Task<string> ReadTextAsync(CancellationToken ct)
    {
        using var reader = new StreamReader(Context.Request.InputStream, Encoding.UTF8);
        return await reader.ReadToEndAsync(ct).ConfigureAwait(false);
    }

    public bool HasFormContentType =>
        Context.Request.ContentType?.StartsWith("multipart/form-data", StringComparison.OrdinalIgnoreCase) == true;

    public Task<MultipartForm> ReadFormAsync(CancellationToken ct) =>
        MultipartFormReader.ReadAsync(Context.Request.InputStream, Context.Request.ContentType ?? string.Empty, ct);
}

/// <summary>What a handler returns. Static helpers mirror ASP.NET's Results.* so route code
/// reads the same as the Server's endpoints.</summary>
public abstract class LoopbackResponse
{
    public abstract Task WriteAsync(HttpListenerResponse response, CancellationToken ct);

    public static LoopbackResponse Json(object payload, int status = 200) => new JsonResponse(payload, status);
    public static LoopbackResponse Text(string text, int status = 200, string contentType = "text/plain; charset=utf-8") => new TextResponse(text, status, contentType);

    /// <summary>Exact bytes, for content that must not be re-encoded on the way out.</summary>
    public static LoopbackResponse Bytes(byte[] body, string contentType, int status = 200) => new BytesResponse(body, contentType, status);
    public static LoopbackResponse File(string path, string contentType, bool attachment = false, string? downloadName = null) => new FileResponse(path, contentType, attachment, downloadName);
    public static LoopbackResponse Status(int status) => new TextResponse(string.Empty, status, "text/plain");
    public static LoopbackResponse NotFound(object? payload = null) => payload is null ? Status(404) : Json(payload, 404);
    /// <summary>A Server-Sent-Events stream: each yielded object becomes one 'data: {json}' frame.</summary>
    /// <param name="frames">The frames to write, pulled one at a time.</param>
    /// <param name="clientGone">
    /// Cancelled when a write fails, so the producer stops as soon as nobody is
    /// listening. Pass the request's own source; null leaves a dropped client
    /// generating to the end of its budget.
    /// </param>
    /// <param name="keepAlive">
    /// How often a comment goes out while the producer has nothing to say, or null for
    /// <see cref="DefaultKeepAlive"/>. It is a parameter because the alternative is a
    /// test that spends five seconds per assertion waiting for a heartbeat, which is a
    /// test nobody runs; nothing in the app passes one.
    /// </param>
    /// <param name="headers">
    /// Extra response headers, written with the rest. It exists so a stream can name
    /// the thing it is a view of — the turn id, which a page needs in order to stop or
    /// re-attach to a generation it no longer owns.
    /// </param>
    public static LoopbackResponse Sse(
        IAsyncEnumerable<object> frames,
        CancellationTokenSource? clientGone = null,
        TimeSpan? keepAlive = null,
        IReadOnlyDictionary<string, string>? headers = null)
        => new SseResponse(frames, clientGone, keepAlive ?? DefaultKeepAlive, headers);

    /// <summary>
    /// How long a stream may say nothing before it writes a comment instead. Cheap
    /// enough to leave on for the life of every stream, and short enough that a page
    /// which went away during a minutes-long prefill is noticed while stopping the
    /// generation still saves something.
    /// </summary>
    internal static readonly TimeSpan DefaultKeepAlive = TimeSpan.FromSeconds(5);

    private sealed class JsonResponse(object payload, int status) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            byte[] body = JsonSerializer.SerializeToUtf8Bytes(payload, SseFraming.JsonOptions);
            response.StatusCode = status;
            response.ContentType = "application/json; charset=utf-8";
            response.ContentLength64 = body.Length;
            await response.OutputStream.WriteAsync(body, ct).ConfigureAwait(false);
        }
    }

    private sealed class TextResponse(string text, int status, string contentType) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            byte[] body = Encoding.UTF8.GetBytes(text);
            response.StatusCode = status;
            response.ContentType = contentType;
            response.ContentLength64 = body.Length;
            await response.OutputStream.WriteAsync(body, ct).ConfigureAwait(false);
        }
    }

    private sealed class BytesResponse(byte[] body, string contentType, int status) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            response.StatusCode = status;
            response.ContentType = contentType;
            response.ContentLength64 = body.Length;
            await response.OutputStream.WriteAsync(body, ct).ConfigureAwait(false);
        }
    }

    private sealed class FileResponse(string path, string contentType, bool attachment, string? downloadName) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            var info = new FileInfo(path);
            response.StatusCode = 200;
            response.ContentType = contentType;
            response.ContentLength64 = info.Length;
            response.Headers["X-Content-Type-Options"] = "nosniff";
            response.Headers["Cache-Control"] = "no-cache";
            if (attachment)
                response.Headers["Content-Disposition"] = "attachment; filename=\"" + (downloadName ?? info.Name).Replace("\"", "") + "\"";
            await using var stream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: true);
            await stream.CopyToAsync(response.OutputStream, ct).ConfigureAwait(false);
        }
    }

    private sealed class SseResponse(
        IAsyncEnumerable<object> frames,
        CancellationTokenSource? clientGone,
        TimeSpan keepAlive,
        IReadOnlyDictionary<string, string>? headers) : LoopbackResponse
    {
        public override async Task WriteAsync(HttpListenerResponse response, CancellationToken ct)
        {
            // The first frame is pulled BEFORE any header goes out, which is the only
            // window in which a status code is still available. The chat service
            // refuses a bad request from its first MoveNextAsync — no model loaded, an
            // unknown session — and that refusal has to reach the page as a 400 with a
            // JSON body, not as a 200 event stream that ends immediately. The desktop
            // adapter does exactly this; a stream that opened its headers first would
            // turn every rejection into a silent empty reply.
            await using IAsyncEnumerator<object> enumerator = frames.GetAsyncEnumerator(ct);
            if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
            {
                Headers(response);
                response.SendChunked = true;
                return;
            }

            Headers(response);
            response.SendChunked = true;
            while (true)
            {
                if (!await WriteAsync(response, enumerator.Current, ct).ConfigureAwait(false))
                    return;

                // Waiting for the next frame is where a disconnect hides. A failed
                // write is the only way HttpListener reveals one, and during a long
                // prefill there is nothing to write for minutes — so a comment goes
                // out every few seconds instead. The page's reader ignores any line
                // that is not `data:`, and the alternative is a Stop button that
                // appears to work while the model keeps going.
                // AsTask may be called only once on a ValueTask, so it is converted
                // here and waited on as a Task from then on.
                Task<bool> next = enumerator.MoveNextAsync().AsTask();
                while (!next.IsCompleted && !ct.IsCancellationRequested)
                {
                    // Waited on with its own token rather than the request's: a
                    // cancelled Task.Delay completes immediately, and looping on that
                    // would spin writing keep-alives as fast as the socket allows.
                    using var tick = new CancellationTokenSource(keepAlive);
                    try { await next.WaitAsync(tick.Token).ConfigureAwait(false); }
                    catch (OperationCanceledException) { }

                    if (next.IsCompleted)
                        break;
                    if (!await WriteAsync(response, null, ct).ConfigureAwait(false))
                    {
                        // The reader is gone and the producer has just been cancelled,
                        // but its pull is still in flight — and an async iterator
                        // refuses to be disposed while it is running, with a
                        // NotSupportedException that would surface as a failed request
                        // rather than as a client that left. So the last pull is waited
                        // for; the cancellation is what ends it.
                        await Finish(next).ConfigureAwait(false);
                        return;
                    }
                }
                if (!await next.ConfigureAwait(false))
                    return;
            }
        }

        private void Headers(HttpListenerResponse response)
        {
            SseFraming.ApplyHeaders(response);
            if (headers is null)
                return;
            foreach (KeyValuePair<string, string> header in headers)
                response.Headers[header.Key] = header.Value;
        }

        /// <summary>
        /// Let a cancelled producer come to a stop, and drop whatever it says on the
        /// way out. Nobody is listening any more, so a frame it still had, or the
        /// cancellation it throws, is not this request's business.
        /// </summary>
        private static async Task Finish(Task<bool> pull)
        {
            try { await pull.ConfigureAwait(false); }
            catch (Exception) { /* the reader left; whatever ends the producer ends it */ }
        }

        /// <summary>
        /// Write one frame, or a keep-alive comment when <paramref name="frame"/> is
        /// null. False means the reader has gone, which is a normal ending rather
        /// than an error: the producer is cancelled and the rest is dropped.
        /// </summary>
        private async Task<bool> WriteAsync(HttpListenerResponse response, object? frame, CancellationToken ct)
        {
            try
            {
                if (frame is null)
                    await SseFraming.WriteCommentAsync(response.OutputStream, ct).ConfigureAwait(false);
                else
                    await SseFraming.WriteFrameAsync(response.OutputStream, frame, ct).ConfigureAwait(false);
                return true;
            }
            catch (Exception ex) when (ex is HttpListenerException or IOException or ObjectDisposedException)
            {
                clientGone?.Cancel();
                return false;
            }
        }
    }
}

/// <summary>The exact wire format TensorSharp.Server's SseWriter produces: 'data: ' + JSON +
/// a blank line, flushed per frame, default serializer options (nulls emitted, property
/// names verbatim). The Web UI keys on those property names.</summary>
public static class SseFraming
{
    /// <summary>
    /// How every JSON body and every stream frame is written.
    ///
    /// <para>
    /// Nulls are WRITTEN, deliberately: this has to stay byte-identical to
    /// TensorSharp.Server's SSE writer, because both Web UI pages read the same frame
    /// format and LoopbackServerTests pins the parity. Omitting them
    /// here looked like the tidy fix for a saved conversation's
    /// <c>"imagePaths": null</c> coming back into the page's history and stopping the
    /// next request in the parser -- but the parser is where that belongs
    /// (ChatMessageParser.StringList), and it is fixed there. A body a page sends must
    /// not be able to depend on what a route chose to omit.
    /// </para>
    /// </summary>
    public static readonly JsonSerializerOptions JsonOptions = new(JsonSerializerDefaults.General);
    private static readonly byte[] Prefix = "data: "u8.ToArray();
    private static readonly byte[] Suffix = "\n\n"u8.ToArray();

    public static void ApplyHeaders(HttpListenerResponse response)
    {
        response.StatusCode = 200;
        response.ContentType = "text/event-stream";
        response.Headers["Cache-Control"] = "no-cache";
        response.Headers["Connection"] = "keep-alive";
        response.Headers["X-Accel-Buffering"] = "no";
    }

    public static async Task WriteFrameAsync(Stream output, object payload, CancellationToken ct)
    {
        byte[] body = JsonSerializer.SerializeToUtf8Bytes(payload, JsonOptions);
        await output.WriteAsync(Prefix, ct).ConfigureAwait(false);
        await output.WriteAsync(body, ct).ConfigureAwait(false);
        await output.WriteAsync(Suffix, ct).ConfigureAwait(false);
        await output.FlushAsync(ct).ConfigureAwait(false);
    }

    /// <summary>
    /// A comment line, which every server-sent-event reader ignores. It exists to
    /// give a stream something to write while it has nothing to say, so that a reader
    /// which has gone away is noticed instead of being generated for.
    /// </summary>
    public static async Task WriteCommentAsync(Stream output, CancellationToken ct)
    {
        await output.WriteAsync(Comment, ct).ConfigureAwait(false);
        await output.FlushAsync(ct).ConfigureAwait(false);
    }

    private static readonly byte[] Comment = ": keep-alive\n\n"u8.ToArray();

    public static string Format(object payload) => "data: " + JsonSerializer.Serialize(payload, JsonOptions) + "\n\n";
}

/// <summary>A handler for one route. Return null to fall through to the next route.</summary>
public delegate Task<LoopbackResponse?> LoopbackHandler(LoopbackRequest request, CancellationToken ct);

/// <summary>
/// A tiny HTTP server on 127.0.0.1 for the WebView to talk to, built on the managed
/// <see cref="HttpListener"/> (iOS has no ASP.NET Core). It knows how to do exactly what
/// the Web UI needs: JSON in/out, multipart uploads, static files, SSE streams, and a
/// per-launch secret so another app on the device cannot drive the model through the
/// loopback port.
///
/// <para>
/// The secret travels as a cookie set by the first <c>GET /?token=…</c> the app itself
/// navigates the WebView to; every <c>/api</c> request without it is refused with 403.
/// Static files and <c>/uploads</c> are served with the same check, because the page's
/// image previews carry the cookie automatically.
/// </para>
/// </summary>
public sealed class LoopbackServer : IDisposable
{
    private HttpListener _listener = new();
    private readonly List<(string method, RoutePattern pattern, LoopbackHandler handler)> _routes = new();
    private readonly CancellationTokenSource _cts = new();
    private readonly ILogger _log;
    /// <summary>Guards the listener's incarnation: <see cref="_listener"/>, <see cref="_epoch"/>, <see cref="_port"/>.</summary>
    private readonly object _lifecycle = new();
    /// <summary>
    /// Cancelled when the listener is restarted, so every request the retired
    /// incarnation was still serving ends then rather than when its stream next
    /// notices. Linked to <see cref="_cts"/>, which is the server's whole life.
    /// </summary>
    private CancellationTokenSource _epoch;
    /// <summary>Bumped on every (re)start; an accept loop that no longer matches it leaves.</summary>
    private int _generation;
    private int _port;
    private Task? _loop;

    public const string TokenCookie = "tensoragent_token";

    public LoopbackServer(ILogger? log = null, int port = 0)
    {
        _log = log ?? NullLogger.Instance;
        _port = port == 0 ? FreePort() : port;
        _epoch = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token);
        Token = Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(24));
        _listener.Prefixes.Add($"http://127.0.0.1:{Port}/");
        // Answered, not merely exempt from the token. It is what ProbeAsync asks for,
        // and a probe that had to read a 404 as "alive" would be one bad route away
        // from reading a dead listener the same way.
        MapGet("/health", (_, _) => Task.FromResult<LoopbackResponse?>(LoopbackResponse.Json(new { ok = true })));
        // Write failures must reach the handler. A failed write is the ONLY way this
        // transport learns that a reader has gone — there is no disconnect callback —
        // and an event stream that never learns it keeps a model generating for a page
        // that closed. Told to ignore them, the listener swallows the exception and
        // reports a successful write to a socket that has been reset, which turns the
        // per-request cancellation and the keep-alive below into dead code.
        _listener.IgnoreWriteExceptions = false;
    }

    /// <summary>
    /// The port the WebView talks to. Fixed for the life of the server except in one
    /// case: a restart after iOS reclaimed the listening socket that could not rebind
    /// it (see <see cref="EnsureListeningAsync"/>), when <see cref="Relisted"/> says so.
    /// </summary>
    public int Port => Volatile.Read(ref _port);
    public string Token { get; }
    /// <summary>How many times the listener has been rebuilt after a probe found it dead. Zero on a healthy run.</summary>
    public int Restarts { get; private set; }
    /// <summary>
    /// Raised, with the new port, when a restart could not keep the old one. The page's
    /// origin has changed, so whoever owns the WebView has to navigate it to
    /// <see cref="EntryUrl"/> again; nothing else about the server is different.
    /// </summary>
    public event Action<int>? Relisted;
    public string BaseUrl => $"http://127.0.0.1:{Port}";
    /// <summary>The URL the app points the WebView at: carries the token once.</summary>
    public string EntryUrl => $"{BaseUrl}/?token={Token}";
    public bool RequireToken { get; set; } = true;
    /// <summary>Directory served for GET requests that no route claims (the Web UI bundle).</summary>
    public string? StaticRoot { get; set; }
    /// <summary>Paths (prefixes) that are exempt from the token check, e.g. "/health".</summary>
    public HashSet<string> Public { get; } = new(StringComparer.Ordinal) { "/health" };

    public void Map(string method, string pattern, LoopbackHandler handler) =>
        _routes.Add((method.ToUpperInvariant(), new RoutePattern(pattern), handler));

    public void MapGet(string pattern, LoopbackHandler handler) => Map("GET", pattern, handler);
    public void MapPost(string pattern, LoopbackHandler handler) => Map("POST", pattern, handler);
    public void MapDelete(string pattern, LoopbackHandler handler) => Map("DELETE", pattern, handler);

    public void Start()
    {
        lock (_lifecycle)
            StartListenerLocked(_listener);
        _log.LogInformation("TensorAgent loopback server listening on {Url}", BaseUrl);
    }

    /// <summary>Start listening on <paramref name="listener"/> and give it an accept loop of its own. Under <see cref="_lifecycle"/>.</summary>
    private void StartListenerLocked(HttpListener listener)
    {
        listener.Start();
        int generation = Interlocked.Increment(ref _generation);
        CancellationToken epoch = _epoch.Token;
        _loop = Task.Run(() => AcceptLoopAsync(listener, generation, epoch));
    }

    /// <summary>
    /// Accept for one incarnation of the listener.
    ///
    /// <para>
    /// The managed HttpListener never faults a pending GetContextAsync for anything
    /// that happens to its listening socket — an accept error is dropped and re-armed,
    /// and a socket that stopped signalling leaves the accept parked for good — so
    /// this loop cannot be the thing that notices a dead listener; <see cref="ProbeAsync"/>
    /// is. What it CAN do is stop being a hazard: leave when its incarnation is
    /// retired, and back off rather than spin if the runtime ever does surface a
    /// failure repeatedly.
    /// </para>
    /// </summary>
    private async Task AcceptLoopAsync(HttpListener listener, int generation, CancellationToken epoch)
    {
        int failures = 0;
        while (!_cts.IsCancellationRequested && !epoch.IsCancellationRequested)
        {
            HttpListenerContext ctx;
            try
            {
                ctx = await listener.GetContextAsync().ConfigureAwait(false);
                failures = 0;
            }
            catch (Exception) when (
                _cts.IsCancellationRequested || epoch.IsCancellationRequested
                || Volatile.Read(ref _generation) != generation)
            {
                // Stopped on purpose: the server is being disposed, or this listener was
                // retired by a restart and a newer loop owns the port now.
                return;
            }
            catch (Exception ex)
            {
                failures++;
                if (failures == 1 || failures % 50 == 0)
                    _log.LogWarning(ex, "loopback accept failed ({Count} in a row)", failures);
                try { await Task.Delay(Math.Min(50 * failures, 1000), _cts.Token).ConfigureAwait(false); }
                catch (OperationCanceledException) { return; }
                continue;
            }
            // Checked again, after the accept: a restart can complete between the loop
            // condition and here, and a request served with a cancelled epoch would be
            // refused for no reason the caller could see. The new loop owns the socket.
            if (epoch.IsCancellationRequested || Volatile.Read(ref _generation) != generation)
            {
                try { ctx.Response.Abort(); } catch (Exception) { /* the restart closed it already */ }
                return;
            }
            _ = Task.Run(() => HandleAsync(ctx, epoch));
        }
    }

    private int _inFlight;

    private async Task HandleAsync(HttpListenerContext ctx, CancellationToken epoch)
    {
        var response = ctx.Response;
        Interlocked.Increment(ref _inFlight);
        try
        {
            string path = Uri.UnescapeDataString(ctx.Request.Url?.AbsolutePath ?? "/");
            string method = ctx.Request.HttpMethod.ToUpperInvariant();

            // Token handshake: the entry URL carries it once, after which the cookie does.
            string? queryToken = ctx.Request.QueryString["token"];
            bool authorised = !RequireToken || IsPublic(path)
                || string.Equals(queryToken, Token, StringComparison.Ordinal)
                || string.Equals(ctx.Request.Cookies[TokenCookie]?.Value, Token, StringComparison.Ordinal);
            if (queryToken is not null && string.Equals(queryToken, Token, StringComparison.Ordinal))
            {
                var cookie = new Cookie(TokenCookie, Token, "/") { HttpOnly = true };
                response.AppendCookie(cookie);
                // HttpListener emits 'Set-Cookie: name=value; Path=/' via AppendCookie; add SameSite explicitly.
                response.Headers.Add("Set-Cookie", $"{TokenCookie}={Token}; Path=/; HttpOnly; SameSite=Strict");
            }
            if (!authorised)
            {
                await LoopbackResponse.Json(new { error = "forbidden" }, 403).WriteAsync(response, _cts.Token).ConfigureAwait(false);
                return;
            }

            LoopbackResponse? result = null;
            foreach (var (m, pattern, handler) in _routes)
            {
                if (m != method || !pattern.TryMatch(path, out var values))
                    continue;
                // One source per request, linked to the server's. A handler that
                // streams hands this to LoopbackResponse.Sse so that a reader walking
                // away stops the work being done for it.
                using var perRequest = CancellationTokenSource.CreateLinkedTokenSource(epoch);
                var request = new LoopbackRequest(ctx, path, values) { Aborted = perRequest.Token, Cancellation = perRequest };
                result = await handler(request, perRequest.Token).ConfigureAwait(false);
                if (result is not null)
                {
                    await result.WriteAsync(response, perRequest.Token).ConfigureAwait(false);
                    return;
                }
            }

            result ??= TryStatic(method, path);
            result ??= LoopbackResponse.Json(new { error = "not found" }, 404);
            await result.WriteAsync(response, _cts.Token).ConfigureAwait(false);
        }
        catch (LoopbackHttpException ex)
        {
            try { await LoopbackResponse.Json(ex.Payload, ex.StatusCode).WriteAsync(response, _cts.Token).ConfigureAwait(false); }
            catch { /* client gone */ }
        }
        catch (OperationCanceledException)
        {
            // client disconnected or server stopping
        }
        catch (Exception ex) when (ex is HttpListenerException or IOException)
        {
            // The client disconnected mid-write: an aborted SSE stream, or a page that
            // went away while a reply was going out. Now that write exceptions are not
            // ignored (see the constructor) this is the ordinary shape of a reader
            // leaving, and logging it as a failed request would bury the real ones.
        }
        catch (ObjectDisposedException) when (
            epoch.IsCancellationRequested || _cts.IsCancellationRequested)
        {
            // The response this request was writing to was closed underneath it by a
            // restart or by shutdown. Also an ordinary ending -- but ONLY then: an
            // ObjectDisposedException from a handler's own state (a disposed engine,
            // a session torn down mid-request) is a server fault and has to be
            // reported as one, not silently turned into the failure this whole change
            // exists to stop the page seeing.
        }
        catch (Exception ex)
        {
            _log.LogError(ex, "loopback request {Method} {Path} failed", ctx.Request.HttpMethod, ctx.Request.Url?.AbsolutePath);
            try
            {
                if (!response.SendChunked && response.ContentLength64 == 0)
                    await LoopbackResponse.Json(new { error = "The server failed to handle the request." }, 500).WriteAsync(response, _cts.Token).ConfigureAwait(false);
            }
            catch { /* headers already sent */ }
        }
        finally
        {
            try { response.Close(); } catch { }
            Interlocked.Decrement(ref _inFlight);
        }
    }


    // ---- surviving a suspension ---------------------------------------------
    //
    // iOS reclaims ("defuncts") the sockets of a suspended process — TN2277 says so
    // for listening sockets in particular, and the kernel's pid_shutdown_sockets makes
    // no exception for 127.0.0.1. The app is suspended about thirty seconds after it
    // leaves the screen; come back minutes later and the listening socket under
    // _listener may be dead. The managed HttpListener hides that completely: the
    // pending GetContextAsync never returns, IsListening stays true, nothing is
    // thrown and nothing is logged, while every connect from the WebView is refused.
    // To the page that is "Load failed" on every request, forever, and the only cure
    // used to be force-quitting the app.
    //
    // So the listener is PROBED on every return to the foreground — a real TCP connect
    // and a GET /health, which is the one thing a defunct socket cannot fake — and
    // rebuilt when the probe fails: Stop() + Start() on the same instance keeps the
    // same port (XNU hands a just-closed listening port straight back, connections
    // still open on it notwithstanding -- LoopbackServerTests proves it), so the
    // page's origin, its token cookie and its composer are untouched. Only if the port cannot be had again does
    // the server move, and then it says so through Relisted.

    /// <summary>What <see cref="ProbeAsync"/> found: whether the listener answered, how it failed if not, and how long it took.</summary>
    public readonly record struct ListenerProbe(bool Alive, string Detail, TimeSpan Elapsed);

    /// <summary>The outcome of <see cref="EnsureListeningAsync"/>.</summary>
    public enum ListenerHealth
    {
        /// <summary>The probe was answered; nothing was touched.</summary>
        Alive,
        /// <summary>The probe failed and the listener was rebuilt on the SAME port.</summary>
        Restarted,
        /// <summary>The probe failed and the listener had to move to a new port; <see cref="Relisted"/> was raised.</summary>
        Relisted,
        /// <summary>The probe failed and no listener could be started. The page cannot be served.</summary>
        Failed,
    }

    /// <summary>The report of one <see cref="EnsureListeningAsync"/>; <see cref="Port"/> is the port in force afterwards.</summary>
    public sealed record ListenerReport(ListenerHealth Health, string Detail, int Port, TimeSpan Elapsed)
    {
        public override string ToString() => $"{Health}: {Detail} (port {Port}, {Elapsed.TotalMilliseconds:0} ms)";
    }

    /// <summary>
    /// Ask the listener, over a fresh TCP connection, whether it is alive. A connect
    /// that is refused, a connection that is accepted by the kernel's backlog but
    /// never answered, and a listener that answers anything but 200 all count as dead.
    /// </summary>
    public async Task<ListenerProbe> ProbeAsync(TimeSpan timeout, CancellationToken ct = default)
    {
        var clock = Stopwatch.StartNew();
        int port = Port;
        using var deadline = CancellationTokenSource.CreateLinkedTokenSource(ct);
        deadline.CancelAfter(timeout);
        try
        {
            using var tcp = new TcpClient();
            await tcp.ConnectAsync(IPAddress.Loopback, port, deadline.Token).ConfigureAwait(false);
            NetworkStream stream = tcp.GetStream();
            byte[] request = Encoding.ASCII.GetBytes(
                $"GET /health HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n");
            await stream.WriteAsync(request, deadline.Token).ConfigureAwait(false);
            byte[] buffer = new byte[512];
            int read = await stream.ReadAsync(buffer, deadline.Token).ConfigureAwait(false);
            string head = Encoding.ASCII.GetString(buffer, 0, read);
            string statusLine = head.Split('\r', '\n')[0];
            bool alive = read > 0 && statusLine.StartsWith("HTTP/1.1 200", StringComparison.Ordinal);
            return new ListenerProbe(alive, alive ? "answered" : "answered '" + statusLine + "'", clock.Elapsed);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return new ListenerProbe(false, $"no answer within {timeout.TotalMilliseconds:0} ms", clock.Elapsed);
        }
        catch (SocketException ex)
        {
            return new ListenerProbe(false, ex.SocketErrorCode.ToString(), clock.Elapsed);
        }
        catch (Exception ex) when (ex is IOException or ObjectDisposedException or InvalidOperationException)
        {
            return new ListenerProbe(false, ex.GetType().Name + ": " + ex.Message, clock.Elapsed);
        }
    }

    /// <summary>
    /// Make sure the listener is serving: probe it, and rebuild it if it is not. Safe
    /// to call on every return to the foreground; on a healthy listener it costs one
    /// loopback round trip and changes nothing — an open event stream is left alone.
    /// </summary>
    public async Task<ListenerReport> EnsureListeningAsync(TimeSpan probeTimeout, CancellationToken ct = default)
    {
        var clock = Stopwatch.StartNew();
        if (_cts.IsCancellationRequested)
            return new ListenerReport(ListenerHealth.Failed, "the server has been disposed", Port, clock.Elapsed);
        ListenerProbe probe = await ProbeAsync(probeTimeout, ct).ConfigureAwait(false);
        if (probe.Alive)
            return new ListenerReport(ListenerHealth.Alive, $"answered in {probe.Elapsed.TotalMilliseconds:0} ms", Port, clock.Elapsed);

        // A refused connection is proof; anything else might be this instant rather
        // than this listener. The probe runs as the app wakes, when every thread the
        // answer needs is waking too, so a slow round trip is a real possibility --
        // and a restart on a healthy listener would cancel work that was going to
        // finish. Ask once more before deciding.
        if (!probe.Detail.Contains(nameof(SocketError.ConnectionRefused), StringComparison.Ordinal)
            && !ct.IsCancellationRequested)
        {
            ListenerProbe again = await ProbeAsync(probeTimeout, ct).ConfigureAwait(false);
            if (again.Alive)
            {
                _log.LogInformation(
                    "the loopback listener missed one probe ({First}) and answered the next in {Ms} ms",
                    probe.Detail, again.Elapsed.TotalMilliseconds);
                return new ListenerReport(ListenerHealth.Alive,
                    $"answered the second probe in {again.Elapsed.TotalMilliseconds:0} ms (the first said {probe.Detail})",
                    Port, clock.Elapsed);
            }
            probe = again;
        }
        return Restart("the listener did not answer a probe (" + probe.Detail + ")", clock);
    }

    /// <summary>
    /// Rebuild the listener without asking it first. <see cref="EnsureListeningAsync"/>
    /// is the ordinary entry; this exists for a caller that already knows.
    /// </summary>
    public ListenerReport Restart(string why) => Restart(why, Stopwatch.StartNew());

    private ListenerReport Restart(string why, Stopwatch clock)
    {
        ListenerReport report;
        int? movedTo = null;
        lock (_lifecycle)
        {
            if (_cts.IsCancellationRequested)
                return new ListenerReport(ListenerHealth.Failed, "the server has been disposed", Port, clock.Elapsed);

            int oldPort = _port;
            // Retire the current incarnation before touching the listener: the accept
            // loop reads the generation to tell "stopped by a restart" from "failed",
            // and the requests it accepted are ended now rather than left to discover a
            // closed response stream (which the managed listener lets them write to,
            // silently, for as long as their producer runs).
            Interlocked.Increment(ref _generation);
            CancellationTokenSource retired = _epoch;
            _epoch = CancellationTokenSource.CreateLinkedTokenSource(_cts.Token);
            try { retired.Cancel(); } catch (Exception) { /* a registered callback threw; the epoch is still cancelled */ }

            try { _listener.Stop(); }
            catch (Exception ex) { _log.LogDebug(ex, "stopping the dead loopback listener threw"); }

            string how;
            try
            {
                // Same instance, same prefix, same port. Stop -> Start is allowed on
                // HttpListener; only Close/Abort/Dispose retire an instance for good.
                StartListenerLocked(_listener);
                how = $"rebound on port {oldPort}";
            }
            catch (Exception sameInstance)
            {
                // A Start() that throws closes the instance permanently, so from here it
                // is a new listener: on the old port if the OS will give it back, else on
                // whatever port is free -- which changes the page's origin and is why
                // Relisted exists.
                try { _listener.Close(); } catch (Exception) { }
                _listener = NewListener(oldPort);
                try
                {
                    StartListenerLocked(_listener);
                    how = $"replaced on port {oldPort} ({sameInstance.Message})";
                }
                catch (Exception samePort)
                {
                    try { _listener.Close(); } catch (Exception) { }
                    int port = FreePort();
                    _listener = NewListener(port);
                    try
                    {
                        StartListenerLocked(_listener);
                    }
                    catch (Exception anyPort)
                    {
                        try { _listener.Close(); } catch (Exception) { }
                        _log.LogError(anyPort, "loopback listener could not be restarted on port {Old} or {New}", oldPort, port);
                        return new ListenerReport(ListenerHealth.Failed,
                            $"{why}; port {oldPort} could not be rebound ({samePort.Message}) and port {port} failed too ({anyPort.Message})",
                            Port, clock.Elapsed);
                    }
                    Volatile.Write(ref _port, port);
                    movedTo = port;
                    how = $"moved from port {oldPort} to {port} ({samePort.Message})";
                }
            }
            Restarts++;
            report = new ListenerReport(movedTo is null ? ListenerHealth.Restarted : ListenerHealth.Relisted,
                why + "; " + how, _port, clock.Elapsed);
        }
        _log.LogWarning("loopback listener restarted: {Why}; {How}", why, report.Detail[(why.Length + 2)..]);
        if (movedTo is { } newPort)
            Relisted?.Invoke(newPort);
        return report;
    }

    private static HttpListener NewListener(int port)
    {
        var listener = new HttpListener { IgnoreWriteExceptions = false };
        listener.Prefixes.Add($"http://127.0.0.1:{port}/");
        return listener;
    }

    private bool IsPublic(string path)
    {
        foreach (string p in Public)
            if (path.Equals(p, StringComparison.Ordinal) || path.StartsWith(p.TrimEnd('/') + "/", StringComparison.Ordinal))
                return true;
        return false;
    }

    private LoopbackResponse? TryStatic(string method, string path)
    {
        if (method != "GET" || StaticRoot is null)
            return null;
        string relative = path == "/" ? "index.html" : path.TrimStart('/');
        if (relative.Contains("..", StringComparison.Ordinal))
            return null;
        string full = Path.GetFullPath(Path.Combine(StaticRoot, relative));
        if (!full.StartsWith(Path.GetFullPath(StaticRoot).TrimEnd(Path.DirectorySeparatorChar) + Path.DirectorySeparatorChar, StringComparison.Ordinal))
            return null;
        // The app's own companion script, served from this assembly rather than from
        // the bundle so it can never drift from the code that expects it.
        if (relative == CompanionScriptName)
            return LoopbackResponse.Text(CompanionScript.Value, contentType: "text/javascript; charset=utf-8");

        if (!File.Exists(full))
            return null;

        // index.html is the app's own phone page (TensorAgent.Maui/wwwroot, bundled as
        // webui/), and it carries only markup and styles. Everything the page DOES —
        // the chat, resuming a saved conversation, native attachments, dictated text —
        // is the companion script above, added by appending one script tag on the way
        // out, so the behaviour ships in this assembly with the routes it talks to.
        if (relative.Equals("index.html", StringComparison.OrdinalIgnoreCase))
            return LoopbackResponse.Bytes(WithCompanionScript(full), "text/html; charset=utf-8");

        return LoopbackResponse.File(full, ContentTypes.For(full));
    }

    private const string CompanionScriptName = "tensoragent.js";

    private static readonly Lazy<string> CompanionScript = new(() =>
    {
        using Stream? stream = typeof(LoopbackServer).Assembly
            .GetManifestResourceStream("TensorAgent.Core.WebUi.tensoragent.js");
        if (stream is null)
        {
            // A build that lost the resource would produce a page with no session
            // resume and no attachments, and nothing would say why. Fail loudly.
            throw new InvalidOperationException(
                "TensorAgent.Core.WebUi.tensoragent.js is not embedded in the assembly; "
                + "check the EmbeddedResource item in TensorAgent.Core.csproj.");
        }
        using var reader = new StreamReader(stream);
        return reader.ReadToEnd();
    });

    /// <summary>
    /// The page's own bytes with one script tag spliced in before <c>&lt;/body&gt;</c>.
    ///
    /// <para>
    /// Bytes, not text. Reading the file into a string and writing it back re-encodes
    /// it — a byte-order mark is dropped, and any encoding the file uses is
    /// normalised — so the page the WebView receives would no longer be the bundled
    /// file with one tag added, which is all this is meant to do to it.
    /// </para>
    /// </summary>
    private static byte[] WithCompanionScript(string indexPath)
    {
        byte[] html = File.ReadAllBytes(indexPath);
        byte[] tag = Encoding.UTF8.GetBytes("\n<script src=\"/" + CompanionScriptName + "\"></script>\n");
        ReadOnlySpan<byte> close = "</body>"u8;

        int at = html.AsSpan().LastIndexOf(close);
        if (at < 0)
            at = html.Length;

        byte[] page = new byte[html.Length + tag.Length];
        html.AsSpan(0, at).CopyTo(page);
        tag.CopyTo(page, at);
        html.AsSpan(at).CopyTo(page.AsSpan(at + tag.Length));
        return page;
    }

    public static int FreePort()
    {
        var l = new TcpListener(IPAddress.Loopback, 0);
        l.Start();
        int port = ((IPEndPoint)l.LocalEndpoint).Port;
        l.Stop();
        return port;
    }

    /// <summary>
    /// Stop accepting, cancel what is running, and wait for it to actually stop.
    ///
    /// <para>
    /// The wait is the part that matters. A request in flight is very often inside
    /// the inference engine, and the engine's weights are freed by whatever disposes
    /// the host next. Returning from here while a generation is still running hands
    /// that code a window in which it frees memory the native compute threads are
    /// still reading, and the process dies with a segmentation fault somewhere
    /// unrelated-looking. Cancellation is delivered between tokens, so this is a
    /// short wait in practice; the cap is there so a wedged request cannot stop the
    /// app from closing.
    /// </para>
    /// </summary>
    public void Dispose()
    {
        _cts.Cancel();
        lock (_lifecycle)
        {
            try { _listener.Stop(); } catch { }
            try { _listener.Close(); } catch { }
        }

        var deadline = Stopwatch.StartNew();
        while (Volatile.Read(ref _inFlight) > 0 && deadline.Elapsed < DrainTimeout)
            Thread.Sleep(20);
        if (Volatile.Read(ref _inFlight) > 0)
            _log.LogWarning("loopback shut down with {Count} request(s) still running", Volatile.Read(ref _inFlight));
    }

    /// <summary>How long <see cref="Dispose"/> waits for running requests before giving up on them.</summary>
    public static TimeSpan DrainTimeout { get; set; } = TimeSpan.FromSeconds(20);
}

/// <summary>A handler may throw this to answer with a status + JSON payload (the same
/// shape the Server's ApiExceptionMiddleware produces).</summary>
public sealed class LoopbackHttpException(int statusCode, object payload) : Exception($"HTTP {statusCode}")
{
    public int StatusCode { get; } = statusCode;
    public object Payload { get; } = payload;
}

/// <summary>Path templates with {name} and {*rest} segments, like Minimal APIs.</summary>
public sealed class RoutePattern
{
    private readonly string[] _segments;
    private readonly bool _catchAll;

    public RoutePattern(string pattern)
    {
        _segments = pattern.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        _catchAll = _segments.Length > 0 && _segments[^1].StartsWith("{*", StringComparison.Ordinal);
    }

    public bool TryMatch(string path, out IReadOnlyDictionary<string, string> values)
    {
        var dict = new Dictionary<string, string>(StringComparer.Ordinal);
        values = dict;
        string[] parts = path.Trim('/').Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (_segments.Length == 0)
            return parts.Length == 0;
        for (int i = 0; i < _segments.Length; i++)
        {
            string seg = _segments[i];
            if (_catchAll && i == _segments.Length - 1)
            {
                if (i >= parts.Length)
                    return false;
                dict[seg[2..^1]] = string.Join('/', parts.Skip(i));
                return true;
            }
            if (i >= parts.Length)
                return false;
            if (seg.StartsWith('{') && seg.EndsWith('}'))
                dict[seg[1..^1]] = parts[i];
            else if (!string.Equals(seg, parts[i], StringComparison.Ordinal))
                return false;
        }
        return parts.Length == _segments.Length;
    }
}

/// <summary>Content types for the Web UI bundle and served uploads; unknown extensions are
/// never served as HTML (the same rule the Server's upload policy applies).</summary>
public static class ContentTypes
{
    private static readonly Dictionary<string, string> Map = new(StringComparer.OrdinalIgnoreCase)
    {
        [".html"] = "text/html; charset=utf-8", [".htm"] = "text/html; charset=utf-8",
        [".js"] = "text/javascript; charset=utf-8", [".css"] = "text/css; charset=utf-8",
        [".json"] = "application/json; charset=utf-8", [".txt"] = "text/plain; charset=utf-8",
        [".md"] = "text/plain; charset=utf-8", [".csv"] = "text/plain; charset=utf-8",
        [".svg"] = "image/svg+xml", [".png"] = "image/png", [".jpg"] = "image/jpeg", [".jpeg"] = "image/jpeg",
        [".gif"] = "image/gif", [".webp"] = "image/webp", [".bmp"] = "image/bmp", [".ico"] = "image/x-icon",
        [".heic"] = "image/heic", [".heif"] = "image/heif",
        [".mp4"] = "video/mp4", [".mov"] = "video/quicktime", [".m4v"] = "video/x-m4v", [".webm"] = "video/webm",
        [".mp3"] = "audio/mpeg", [".wav"] = "audio/wav", [".m4a"] = "audio/mp4", [".aac"] = "audio/aac",
        [".ogg"] = "audio/ogg", [".flac"] = "audio/flac", [".caf"] = "audio/x-caf",
        [".pdf"] = "application/pdf", [".woff"] = "font/woff", [".woff2"] = "font/woff2",
    };

    public static string For(string path) =>
        Map.TryGetValue(Path.GetExtension(path), out string? ct) ? ct : "application/octet-stream";
}
