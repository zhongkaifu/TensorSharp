// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Logging;

/// <summary>
/// Lightweight HTTP request logger that emits a "request started" entry on the
/// way in (with method, path, remote IP, content-type/length) and a
/// "request completed" entry on the way out (with status code and duration).
/// Failed requests are surfaced at the <c>Error</c> level with the captured
/// exception attached.
///
/// A unique request id is allocated up-front and pushed into the logging
/// scope so every entry produced by downstream handlers is automatically
/// correlated. The same id is also written to the <c>X-TensorSharp-Request-Id</c>
/// response header so clients can quote it back when reporting issues.
///
/// "Low-noise" paths configured via <see cref="RequestLoggingOptions"/> have
/// their successful start/completion entries demoted to <c>Debug</c>, which
/// keeps the default log readable when the Web UI polls a status endpoint
/// on a tight loop. 4xx/5xx responses and exceptions on those paths still
/// surface at their natural level so genuine failures aren't hidden.
/// </summary>
public sealed class RequestLoggingMiddleware
{
    public const string RequestIdHeader = "X-TensorSharp-Request-Id";

    private readonly RequestDelegate _next;
    private readonly ILogger<RequestLoggingMiddleware> _logger;
    private readonly RequestLoggingOptions _options;

    /// <summary>Backwards-compatible constructor used by tests and ad-hoc consumers; defaults to no low-noise paths.</summary>
    public RequestLoggingMiddleware(RequestDelegate next, ILogger<RequestLoggingMiddleware> logger)
        : this(next, logger, new RequestLoggingOptions())
    {
    }

    public RequestLoggingMiddleware(
        RequestDelegate next,
        ILogger<RequestLoggingMiddleware> logger,
        RequestLoggingOptions options)
    {
        _next = next ?? throw new ArgumentNullException(nameof(next));
        _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    public async Task InvokeAsync(HttpContext context)
    {
        string requestId = ResolveRequestId(context);
        context.Response.Headers[RequestIdHeader] = requestId;

        // A capability path (an /uploads file, a code-execution artifact, a stored Responses
        // API response) is named by an unguessable id that is all that guards an
        // unauthenticated URL, so it is user content. The {Path} arguments below are marked
        // as such: local logs are unchanged and the OpenTelemetry exporter redacts them. Log
        // scopes are not exported (TensorSharpTelemetrySetup.ConfigureLoggerOptions), so the
        // Operation scope below stays as it always was.
        string path = context.Request.Path.Value ?? "/";
        object? logPath = CapabilityPaths.IsCapabilityPath(context.Request.Path) ? LogContent.Of(path) : path;

        var scopeProperties = new Dictionary<string, object>(StringComparer.Ordinal)
        {
            [LogScopeKeys.RequestId] = requestId,
            [LogScopeKeys.Operation] = $"{context.Request.Method} {context.Request.Path}",
        };

        string? clientLabel = ClassifyClient(context.Request.Path);
        if (!string.IsNullOrEmpty(clientLabel))
            scopeProperties[LogScopeKeys.Client] = clientLabel;

        // OpenTelemetry: the request id on the HTTP server span (so a quoted
        // X-TensorSharp-Request-Id finds its trace), and the id and protocol for the
        // generation spans and metrics further down, which do not see the HttpContext.
        // The id may be the client's own header, of any length, so telemetry gets a
        // bounded prefix; the response header and the log scope keep it whole.
        string telemetryRequestId = TruncateForTelemetry(requestId);
        TelemetryContext.RequestId = telemetryRequestId;
        TelemetryContext.Protocol = clientLabel;
        if (ServerSpan.Recorded(context) is { } serverSpan)
        {
            serverSpan.SetTag(TelemetryAttributes.RequestId, telemetryRequestId);
            if (!string.IsNullOrEmpty(clientLabel))
                serverSpan.SetTag(TelemetryAttributes.Protocol, clientLabel);
        }

        using var scope = _logger.BeginScope(scopeProperties);

        bool isLowNoise = _options.IsLowNoisePath(context.Request.Path);

        string remoteIp = context.Connection.RemoteIpAddress?.ToString() ?? "(unknown)";
        string contentType = context.Request.ContentType ?? string.Empty;
        long? contentLength = context.Request.ContentLength;

        _logger.Log(isLowNoise ? LogLevel.Debug : LogLevel.Information, LogEventIds.HttpRequestStarted,
            "HTTP {Method} {Path} from {RemoteIp} (content-type={ContentType}, content-length={ContentLength})",
            context.Request.Method, logPath, remoteIp, contentType,
            contentLength.HasValue ? contentLength.Value.ToString(CultureInfo.InvariantCulture) : "(none)");

        var sw = Stopwatch.StartNew();
        try
        {
            await _next(context).ConfigureAwait(false);
            sw.Stop();

            LogLevel completionLevel = context.Response.StatusCode >= 500
                ? LogLevel.Error
                : context.Response.StatusCode >= 400
                    ? LogLevel.Warning
                    : isLowNoise
                        ? LogLevel.Debug
                        : LogLevel.Information;

            EventId completionEvent = context.Response.StatusCode >= 400
                ? LogEventIds.HttpRequestRejected
                : LogEventIds.HttpRequestCompleted;

            _logger.Log(completionLevel, completionEvent,
                "HTTP {Method} {Path} -> {StatusCode} in {ElapsedMs:F1} ms (response content-type={ResponseContentType})",
                context.Request.Method, logPath,
                context.Response.StatusCode, sw.Elapsed.TotalMilliseconds,
                context.Response.ContentType ?? "(none)");
        }
        catch (Exception ex)
        {
            sw.Stop();
            _logger.Log(LogLevel.Error, LogEventIds.HttpRequestFailed, ex,
                "HTTP {Method} {Path} threw {ExceptionType} after {ElapsedMs:F1} ms",
                context.Request.Method, logPath,
                ex.GetType().Name, sw.Elapsed.TotalMilliseconds);
            throw;
        }
    }

    private static string ResolveRequestId(HttpContext context)
    {
        // Honor an inbound id (e.g. from a load balancer) so cross-service traces
        // line up. Fall back to a generated short id when none is provided.
        if (context.Request.Headers.TryGetValue(RequestIdHeader, out var existing) &&
            existing.Count > 0 &&
            !string.IsNullOrWhiteSpace(existing[0]))
        {
            return existing[0]!;
        }

        return Guid.NewGuid().ToString("N")[..12];
    }

    /// <summary>Longest request id telemetry carries (span attribute, <see cref="TelemetryContext.RequestId"/>).</summary>
    internal const int MaxTelemetryRequestIdLength = 64;

    /// <summary>
    /// <paramref name="requestId"/> cut to <see cref="MaxTelemetryRequestIdLength"/>
    /// characters, without splitting a surrogate pair. A generated id is 12 characters;
    /// only an inbound header can be longer, and it is copied onto every generation span
    /// and exported, so its length is bounded here.
    /// </summary>
    internal static string TruncateForTelemetry(string requestId)
    {
        if (requestId.Length <= MaxTelemetryRequestIdLength)
            return requestId;
        int length = MaxTelemetryRequestIdLength;
        if (char.IsHighSurrogate(requestId[length - 1]))
            length--;
        return requestId[..length];
    }

    internal static string? ClassifyClient(PathString path)
    {
        string p = path.Value ?? string.Empty;
        if (p.StartsWith("/v1/", StringComparison.OrdinalIgnoreCase))
            return "openai";
        // /api/embed is Ollama's embedding route (OllamaEndpoints / EmbeddingEndpoints),
        // not a Web UI one, though it shares the /api/ prefix.
        if (p.Equals("/api/chat/ollama", StringComparison.OrdinalIgnoreCase) ||
            p.StartsWith("/api/embed", StringComparison.OrdinalIgnoreCase) ||
            p.StartsWith("/api/generate", StringComparison.OrdinalIgnoreCase) ||
            p.StartsWith("/api/show", StringComparison.OrdinalIgnoreCase) ||
            p.StartsWith("/api/tags", StringComparison.OrdinalIgnoreCase) ||
            p.StartsWith("/api/version", StringComparison.OrdinalIgnoreCase))
        {
            return "ollama";
        }
        if (p.StartsWith("/api/", StringComparison.OrdinalIgnoreCase))
            return "webui";
        return null;
    }
}

/// <summary>
/// Configuration for <see cref="RequestLoggingMiddleware"/>. Currently only
/// carries the set of "low-noise" paths whose successful entries should be
/// logged at <c>Debug</c>; designed so future knobs (sampling rate, redacted
/// headers, etc.) can be added without reshaping the middleware contract.
/// </summary>
public sealed class RequestLoggingOptions
{
    /// <summary>
    /// Exact request paths whose successful start/completion entries are
    /// emitted at <c>Debug</c> level instead of <c>Information</c>. Path
    /// comparison is case-insensitive. 4xx, 5xx, and exceptions are
    /// unaffected so genuine errors still surface.
    /// </summary>
    public HashSet<string> LowNoisePaths { get; } = new(StringComparer.OrdinalIgnoreCase);

    internal bool IsLowNoisePath(PathString path)
    {
        string? value = path.Value;
        if (string.IsNullOrEmpty(value) || LowNoisePaths.Count == 0)
            return false;
        return LowNoisePaths.Contains(value);
    }
}

public static class RequestLoggingMiddlewareExtensions
{
    /// <summary>
    /// Register <see cref="RequestLoggingOptions"/> in DI so the middleware
    /// resolves a configured instance instead of the empty default. Idempotent;
    /// repeated calls fold their configuration into the same singleton.
    /// </summary>
    public static IServiceCollection AddTensorSharpRequestLogging(
        this IServiceCollection services,
        Action<RequestLoggingOptions>? configure = null)
    {
        services.AddSingleton<RequestLoggingOptions>(sp =>
        {
            var options = new RequestLoggingOptions();
            configure?.Invoke(options);
            return options;
        });
        return services;
    }

    public static IApplicationBuilder UseTensorSharpRequestLogging(this IApplicationBuilder app)
    {
        return app.UseMiddleware<RequestLoggingMiddleware>();
    }
}
