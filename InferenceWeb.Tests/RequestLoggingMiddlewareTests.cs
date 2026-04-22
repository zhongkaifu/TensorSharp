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
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Server.Logging;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for <see cref="RequestLoggingMiddleware"/>. We use an in-memory
/// recording logger because the middleware's value comes from the structured
/// scope + entries it produces; plumbing through ASP.NET TestHost would obscure
/// what is actually being asserted.
/// </summary>
public class RequestLoggingMiddlewareTests
{
    [Fact]
    public async Task Middleware_LogsStartAndCompletion_WithMatchingRequestId()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "POST";
        ctx.Request.Path = "/api/chat";
        ctx.Response.StatusCode = 200;

        await middleware.InvokeAsync(ctx);

        Assert.Equal(2, logger.Entries.Count);
        Assert.Equal(LogLevel.Information, logger.Entries[0].Level);
        Assert.Contains("HTTP POST /api/chat", logger.Entries[0].Message);
        Assert.Equal(LogLevel.Information, logger.Entries[1].Level);
        Assert.Contains("/api/chat -> 200", logger.Entries[1].Message);

        // The middleware sets the request id header so clients can quote it.
        Assert.True(ctx.Response.Headers.ContainsKey(RequestLoggingMiddleware.RequestIdHeader));
        string requestId = ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader].ToString();
        Assert.False(string.IsNullOrEmpty(requestId));

        // Both entries must share the same RequestId scope so downstream tools can correlate.
        foreach (var entry in logger.Entries)
        {
            var scope = entry.Scope as IDictionary<string, object>;
            Assert.NotNull(scope);
            Assert.Equal(requestId, (string)scope[LogScopeKeys.RequestId]);
        }
    }

    [Fact]
    public async Task Middleware_PromotesClientErrorsToWarning()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(ctx =>
        {
            ctx.Response.StatusCode = 404;
            return Task.CompletedTask;
        }, logger);

        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "GET";
        ctx.Request.Path = "/api/sessions/missing";

        await middleware.InvokeAsync(ctx);

        // start = info, end = warning for 4xx
        Assert.Equal(LogLevel.Information, logger.Entries[0].Level);
        Assert.Equal(LogLevel.Warning, logger.Entries[1].Level);
        Assert.Equal(LogEventIds.HttpRequestRejected.Id, logger.Entries[1].EventId.Id);
    }

    [Fact]
    public async Task Middleware_PromotesServerErrorsToError()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(ctx =>
        {
            ctx.Response.StatusCode = 503;
            return Task.CompletedTask;
        }, logger);

        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "POST";
        ctx.Request.Path = "/api/generate";

        await middleware.InvokeAsync(ctx);

        Assert.Equal(LogLevel.Error, logger.Entries[1].Level);
    }

    [Fact]
    public async Task Middleware_ReportsExceptionAsErrorAndRethrows()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(_ => throw new InvalidOperationException("boom"), logger);

        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "POST";
        ctx.Request.Path = "/api/upload";

        await Assert.ThrowsAsync<InvalidOperationException>(() => middleware.InvokeAsync(ctx));

        // The middleware should still emit a "started" entry and an error entry.
        Assert.Equal(2, logger.Entries.Count);
        Assert.Equal(LogLevel.Information, logger.Entries[0].Level);
        Assert.Equal(LogLevel.Error, logger.Entries[1].Level);
        Assert.Equal(LogEventIds.HttpRequestFailed.Id, logger.Entries[1].EventId.Id);
        Assert.NotNull(logger.Entries[1].Exception);
    }

    [Fact]
    public async Task Middleware_HonoursInboundRequestIdHeader()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        var ctx = new DefaultHttpContext();
        ctx.Request.Method = "GET";
        ctx.Request.Path = "/";
        ctx.Request.Headers[RequestLoggingMiddleware.RequestIdHeader] = "client-correlated-id";

        await middleware.InvokeAsync(ctx);

        Assert.Equal("client-correlated-id",
            ctx.Response.Headers[RequestLoggingMiddleware.RequestIdHeader].ToString());

        var scope = (IDictionary<string, object>)logger.Entries[0].Scope!;
        Assert.Equal("client-correlated-id", scope[LogScopeKeys.RequestId]);
    }

    [Fact]
    public async Task Middleware_AttachesClientLabelBasedOnPath()
    {
        var logger = new RecordingLogger<RequestLoggingMiddleware>();
        var middleware = new RequestLoggingMiddleware(_ => Task.CompletedTask, logger);

        async Task Hit(string path)
        {
            var ctx = new DefaultHttpContext();
            ctx.Request.Method = "POST";
            ctx.Request.Path = path;
            await middleware.InvokeAsync(ctx);
        }

        await Hit("/v1/chat/completions");
        await Hit("/api/chat/ollama");
        await Hit("/api/chat");
        await Hit("/");

        var clients = logger.Entries
            .Where(e => e.EventId.Id == LogEventIds.HttpRequestStarted.Id)
            .Select(e => ((IDictionary<string, object>)e.Scope!).TryGetValue(LogScopeKeys.Client, out var c) ? c?.ToString() : null)
            .ToList();

        Assert.Equal(new[] { "openai", "ollama", "webui", null }, clients);
    }

    private sealed class RecordingLogger<T> : ILogger<T>
    {
        public List<RecordedEntry> Entries { get; } = new();

        private object? _activeScope;

        public IDisposable BeginScope<TState>(TState state) where TState : notnull
        {
            _activeScope = state;
            return new ScopeDisposable(this);
        }

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception, Func<TState, Exception?, string> formatter)
        {
            Entries.Add(new RecordedEntry
            {
                Level = logLevel,
                EventId = eventId,
                Message = formatter != null ? formatter(state, exception) : state?.ToString() ?? string.Empty,
                Exception = exception,
                Scope = _activeScope,
            });
        }

        private sealed class ScopeDisposable : IDisposable
        {
            private readonly RecordingLogger<T> _owner;
            public ScopeDisposable(RecordingLogger<T> owner) => _owner = owner;
            public void Dispose() { _owner._activeScope = null; }
        }

        public sealed class RecordedEntry
        {
            public LogLevel Level { get; init; }
            public EventId EventId { get; init; }
            public string Message { get; init; } = string.Empty;
            public Exception? Exception { get; init; }
            public object? Scope { get; init; }
        }
    }
}
