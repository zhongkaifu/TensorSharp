// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.Extensions.Logging;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Host.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// What the server says at startup about options it accepts but cannot honour as asked.
/// Every case here used to be either silent or worded as the opposite of what happened:
/// the paged-KV flags were logged as "configured" for a cache the server never builds, a
/// larger --upload-max-mb could never apply past a fixed 500 MB request-body limit (and
/// raising it for every route let each JSON request buffer that much), an
/// incomplete --width/--height pair was dropped without a word, and the backend line said
/// "Falling back" on every launch without --backend and right before a refused load.
/// </summary>
public class ServerStartupWarningsTests : IDisposable
{
    private readonly string _baseDir =
        Path.Combine(Path.GetTempPath(), "ts-server-warnings-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new();

    public ServerStartupWarningsTests()
    {
        Directory.CreateDirectory(_baseDir);
        _env.ClearSpeculationVars();
        _env.Set(ServerOptionsBuilder.QwenImageWidthEnvVar, null);
        _env.Set(ServerOptionsBuilder.QwenImageHeightEnvVar, null);
        _env.Set("TS_UPLOAD_MAX_MB", null);
        foreach (string name in new[]
                 {
                     "TS_KV_PAGED_CACHE", "TS_KV_BLOCK_SIZE", "TS_KV_CACHE_MAX_RAM_MB", "TS_KV_CACHE_SSD_DIR",
                     "TS_KV_CACHE_MAX_SSD_MB", "TS_KV_PAGED_QUANT_BITS", "TS_KV_CACHE_REDIS_URL",
                     "TS_KV_CACHE_REDIS_TTL_MINUTES",
                 })
        {
            _env.Set(name, null);
        }
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    // ---- --paged-kv*: accepted, inert, and said so once ----------------------------------

    [Fact]
    public void InertPagedKvWarning_IsNullWithoutAnyPagedKvFlag()
    {
        Assert.Null(ServerOptionsBuilder.DescribeInertPagedKvFlags(
            new[] { "--model", "m.gguf", "--redis-url", "localhost:6379" }, prefixCacheEnabled: true));
        Assert.Null(ServerOptionsBuilder.DescribeInertPagedKvFlags(Array.Empty<string>(), prefixCacheEnabled: true));
    }

    [Fact]
    public void InertPagedKvWarning_NamesEveryFlagOnce_AndWhatServesPrefixReuse()
    {
        string[] args =
        {
            "--paged-kv", "--paged-kv-block-size", "128", "--PAGED-KV-RAM-MB=2048",
            "--paged-kv-redis-url", "localhost:6379", "--paged-kv", "--No-Paged-Kv",
        };
        // Program.cs runs the applier first; the warning must still find every flag after it.
        Assert.True(ServerOptionsBuilder.ApplyPagedKvCacheCliFlags(args));

        string warning = ServerOptionsBuilder.DescribeInertPagedKvFlags(args, prefixCacheEnabled: true);

        Assert.NotNull(warning);
        Assert.StartsWith(
            "--paged-kv, --paged-kv-block-size, --paged-kv-ram-mb, --paged-kv-redis-url, --no-paged-kv have no effect",
            warning, StringComparison.Ordinal);
        Assert.Contains("--paged-bench", warning, StringComparison.Ordinal);
        Assert.Contains("radix prefix cache, which is on", warning, StringComparison.Ordinal);
        // The value of a value flag is never mistaken for a flag of its own.
        Assert.DoesNotContain("localhost", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void InertPagedKvWarning_SaysWhenPrefixReuseIsOffToo()
    {
        string warning = ServerOptionsBuilder.DescribeInertPagedKvFlags(
            new[] { "--paged-kv-quant-bits", "8" }, prefixCacheEnabled: false);

        Assert.StartsWith("--paged-kv-quant-bits has no effect", warning, StringComparison.Ordinal);
        Assert.Contains("reuses no prefix at all", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void Build_StillAcceptsEveryPagedKvSpelling()
    {
        // Inert is not refused: config files and command lines in the wild carry these.
        foreach (string flag in ServerOptionsBuilder.PagedKvSwitchFlags)
            ServerOptionsBuilder.Build(new[] { flag, "--no-skills" }, _baseDir);
        foreach (string flag in ServerOptionsBuilder.PagedKvValueFlags)
            ServerOptionsBuilder.Build(new[] { flag, "1", "--no-skills" }, _baseDir);
    }

    // ---- --upload-max-mb drives the /api/upload request-body limit -----------------------

    [Theory]
    [InlineData(25L, 500L)]      // lowering the per-file cap never shrinks the request-body limit
    [InlineData(500L, 500L)]     // the default deployment is exactly what it was
    [InlineData(2000L, 2000L)]   // raising it finally takes effect
    public void UploadRequestBodyBytes_FollowsTheUploadCap_WithA500MbFloor(long capMb, long expectedMb)
    {
        Assert.Equal(expectedMb * 1024 * 1024, ServerOptionsBuilder.ResolveUploadRequestBodyBytes(capMb * 1024 * 1024));
    }

    [Fact]
    public void UploadRequestBodyBytes_FollowsTheParsedFlag()
    {
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--upload-max-mb", "2000", "--no-skills" }, _baseDir);

        Assert.Equal(2000L * 1024 * 1024, ServerOptionsBuilder.ResolveUploadRequestBodyBytes(options.UploadMaxFileBytes));
    }

    [Fact]
    public void EveryOtherRoute_KeepsThe500MbRequestBodyLimit()
    {
        // The JSON routes buffer the whole body; only the multipart upload route streams.
        Assert.Equal(500L * 1024 * 1024, ServerOptionsBuilder.DefaultMaxRequestBodyBytes);
    }

    [Fact]
    public void UploadRoute_RaisesItsOwnRequestBodyLimit_ToTheCap()
    {
        var context = new Microsoft.AspNetCore.Http.DefaultHttpContext();
        var limit = new FakeBodySizeFeature { MaxRequestBodySize = ServerOptionsBuilder.DefaultMaxRequestBodyBytes };
        context.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>(limit);

        Assert.True(TensorSharp.Server.ProtocolAdapters.WebUiAdapter.RaiseUploadRequestBodyLimit(context, 2000L * 1024 * 1024));
        Assert.Equal(2000L * 1024 * 1024, limit.MaxRequestBodySize);
    }

    [Fact]
    public void UploadRoute_NeverLowersALimit_OrTouchesAReadOnlyOne()
    {
        var context = new Microsoft.AspNetCore.Http.DefaultHttpContext();
        var unlimited = new FakeBodySizeFeature { MaxRequestBodySize = null };
        context.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>(unlimited);
        Assert.False(TensorSharp.Server.ProtocolAdapters.WebUiAdapter.RaiseUploadRequestBodyLimit(context, 2000L * 1024 * 1024));
        Assert.Null(unlimited.MaxRequestBodySize);

        // At the default cap the route's limit is already the server-wide one.
        var atDefault = new FakeBodySizeFeature { MaxRequestBodySize = ServerOptionsBuilder.DefaultMaxRequestBodyBytes };
        context.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>(atDefault);
        Assert.False(TensorSharp.Server.ProtocolAdapters.WebUiAdapter.RaiseUploadRequestBodyLimit(context, 25L * 1024 * 1024));
        Assert.Equal(ServerOptionsBuilder.DefaultMaxRequestBodyBytes, atDefault.MaxRequestBodySize);

        // Once the body has started to be read, Kestrel makes the feature read-only.
        var started = new FakeBodySizeFeature { MaxRequestBodySize = 1, IsReadOnly = true };
        context.Features.Set<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>(started);
        Assert.False(TensorSharp.Server.ProtocolAdapters.WebUiAdapter.RaiseUploadRequestBodyLimit(context, 2000L * 1024 * 1024));
        Assert.Equal(1, started.MaxRequestBodySize);
    }

    private sealed class FakeBodySizeFeature : Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature
    {
        public bool IsReadOnly { get; set; }
        public long? MaxRequestBodySize { get; set; }
    }

    // ---- --width / --height as the Qwen-Image-2.1 default size ---------------------------

    [Theory]
    [InlineData("--model", "m.gguf")]
    [InlineData("--width", "1024", "--height", "768")]
    [InlineData("--video-width", "1000")]   // the video-only spelling sets no image default
    public void QwenImageSizeWarnings_NoneWhenThereIsNothingToSay(params string[] args)
    {
        Assert.Empty(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(args));
        Assert.Empty(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(Array.Empty<string>()));
    }

    [Theory]
    [InlineData("--width", "--height")]
    [InlineData("--height", "--width")]
    public void QwenImageSizeWarnings_OneSideAlone_SaysTheDefaultIsNotSet(string given, string missing)
    {
        string warning = Assert.Single(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(
            new[] { given, "1024" }));

        Assert.StartsWith($"{given} was given without {missing}", warning, StringComparison.Ordinal);
        Assert.Contains("automatic size (a 2048x2048 area)", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void QwenImageSizeWarnings_OtherSideFromTheEnvironment_CompletesThePair()
    {
        // The pipeline reads both env vars, so a half set there is a whole pair.
        _env.Set(ServerOptionsBuilder.QwenImageHeightEnvVar, "768");

        Assert.Empty(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(new[] { "--width", "1024" }));
    }

    [Fact]
    public void QwenImageSizeWarnings_OffGridValue_SaysWhatItIsSnappedTo()
    {
        string warning = Assert.Single(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(
            new[] { "--width=1000", "--height", "1024" }));

        Assert.StartsWith("--width 1000 is not a multiple of 32", warning, StringComparison.Ordinal);
        Assert.Contains("snapped down to a multiple of 32 (992)", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void QwenImageSizeWarnings_BelowOneGridStep_SaysItIsRaisedToTheMinimum()
    {
        // The pipeline never snaps below one 32-pixel tile, so "down to 0" would be false.
        string warning = Assert.Single(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(
            new[] { "--width", "1024", "--height", "20" }));

        Assert.StartsWith("--height 20 is not a multiple of 32", warning, StringComparison.Ordinal);
        Assert.Contains("raised to 32", warning, StringComparison.Ordinal);
        Assert.DoesNotContain("(0)", warning, StringComparison.Ordinal);
    }

    [Fact]
    public void QwenImageSizeWarnings_LastValueWins_LikeTheApplier()
    {
        Assert.Empty(ServerOptionsBuilder.DescribeQwenImageSizeDefaultWarnings(
            new[] { "--width", "1000", "--height", "1024", "--width", "1024" }));
    }

    // ---- the backend line ------------------------------------------------------------------

    [Fact]
    public void BackendFallback_NoRequest_OnThePlatformDefault_SaysNothing()
    {
        // This fired on EVERY launch without --backend, naming an empty request.
        string platform = ServerOptionsBuilder.PlatformDefaultBackend;
        var logger = new RecordingLogger();

        StartupBanner.EmitBackendFallback(logger, Options(platform, "m.gguf", platform), requestedBackendInput: null);

        Assert.Empty(logger.Entries);
    }

    [Fact]
    public void BackendFallback_NoRequest_PlatformDefaultMissing_NamesIt()
    {
        // "cpu" is the managed backend, never the platform default on any OS.
        var logger = new RecordingLogger();

        StartupBanner.EmitBackendFallback(logger, Options("cpu", "m.gguf", "cpu"), requestedBackendInput: null);

        RecordingLogger.Entry entry = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Contains($"'{ServerOptionsBuilder.PlatformDefaultBackend}' is unavailable", entry.Message, StringComparison.Ordinal);
        Assert.Contains("Using 'cpu'", entry.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BackendFallback_ExplicitUnavailable_WithAStartupModel_SaysTheLoadIsRefused()
    {
        // StartupModelLoader resolves the requested backend itself and refuses the load
        // (exit 2), so "Falling back" right before that refusal was the opposite of the truth.
        var logger = new RecordingLogger();

        StartupBanner.EmitBackendFallback(logger, Options("ggml_cpu", "m.gguf", "ggml_cpu"), "ggml_cuda");

        RecordingLogger.Entry entry = Assert.Single(logger.Entries);
        Assert.Equal(LogLevel.Warning, entry.Level);
        Assert.Contains("'ggml_cuda' is not available", entry.Message, StringComparison.Ordinal);
        Assert.Contains("available: ggml_cpu", entry.Message, StringComparison.Ordinal);
        Assert.Contains("refused", entry.Message, StringComparison.Ordinal);
        Assert.DoesNotContain("Falling back", entry.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BackendFallback_ExplicitUnavailable_ModelLess_ReallyFallsBack()
    {
        var logger = new RecordingLogger();

        StartupBanner.EmitBackendFallback(logger, Options("ggml_cpu", null, "ggml_cpu"), "ggml_cuda");

        RecordingLogger.Entry entry = Assert.Single(logger.Entries);
        Assert.Contains("Falling back to 'ggml_cpu'", entry.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void BackendFallback_ExplicitAvailable_InAnySpelling_SaysNothing()
    {
        var logger = new RecordingLogger();

        StartupBanner.EmitBackendFallback(logger, Options("ggml_cuda", "m.gguf", "ggml_cpu", "ggml_cuda"), "GGML-CUDA");

        Assert.Empty(logger.Entries);
    }

    [Fact]
    public void EmbeddingBackend_ExplicitUnavailable_IsRefusedWithAMessageNotAnException()
    {
        // The host reports this as one line and exit 2, like the chat startup load; it used
        // to escape as an unhandled ArgumentException from service registration.
        ServerHostingOptions options = Options("ggml_cpu", "encoder.gguf", "ggml_cpu");

        Assert.False(EmbeddingHosting.TryResolveBackend(options, "ggml_cuda", out string backend, out string error));
        Assert.Null(backend);
        Assert.Contains("not supported", error, StringComparison.Ordinal);
    }

    [Fact]
    public void EmbeddingBackend_NoRequest_UsesTheResolvedDefault()
    {
        Assert.True(EmbeddingHosting.TryResolveBackend(
            EmbeddingEndpointTests.Options(), null, out string backend, out string error));
        Assert.Equal("GGML_CPU", backend);
        Assert.Null(error);
    }

    private static ServerHostingOptions Options(string defaultBackend, string model, params string[] supported) => new(
        startupModelPath: model, startupMmProjPath: null,
        defaultBackend: defaultBackend,
        supportedBackends: supported.Select(b => new BackendOption(b, b)).ToArray(),
        defaultMaxTokens: 100, maxTokensPinned: false, defaultVideoFrames: 0, defaultVideoFps: 0,
        defaultVideoWidth: 0, defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
        uploadDirectory: Path.GetTempPath(), logDirectory: Path.GetTempPath(),
        fileLoggingEnabled: false, samplingDefaults: null);

    private sealed class RecordingLogger : ILogger
    {
        public sealed record Entry(LogLevel Level, string Message);

        public List<Entry> Entries { get; } = new();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
            => Entries.Add(new Entry(logLevel, formatter(state, exception)));
    }
}
