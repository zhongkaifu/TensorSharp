// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Reflection;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Runtime;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// A refused model load, with fakes standing in for the model factory: the refusal is
/// told apart from a bug, it is logged as one reason rather than a stack trace, the
/// service keeps (or restores) the model it had, a reload through the API answers with
/// an error instead of escaping, and the host's report is one line after the cleanup.
/// The real processes are covered by <see cref="HostLoadRefusalProcessTests"/>.
/// </summary>
public class ModelLoadRefusalTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public ModelLoadRefusalTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-load-refusal-unit-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
        _env.Set("TS_DSV4_DSPARK", null);
    }

    public void Dispose()
    {
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- classification ---------------------------------------------------------

    public static IEnumerable<object[]> Refusals() => new[]
    {
        new object[] { new ModelLoadRefusedException("[dsv4] not enough VRAM: re-run with --n-cpu-moe 2") },
        new object[] { new NotSupportedException("KV_CACHE_DTYPE=q8_0 is not supported by DeepSeek V4.1 Flash.") },
        new object[] { new FileNotFoundException("DeepSeek V4.1 requires its Engram lookup sidecar.", "deepseek41.engram.bin") },
        new object[] { new IOException("model.gguf is incomplete: re-download this file.") },
        new object[] { new InvalidDataException("Not a GGUF file (magic: 0x20746F6E)") },
        new object[] { new UnauthorizedAccessException("Access to the path is denied.") },
    };

    [Theory]
    [MemberData(nameof(Refusals))]
    public void Refusals_AreClassifiedWithTheirMessage(Exception refusal)
    {
        Assert.True(ModelLoadRefusal.TryDescribe(refusal, out string reason));
        Assert.Equal(refusal.Message, reason);
    }

    public static IEnumerable<object[]> Bugs() => new[]
    {
        new object[] { new InvalidOperationException("Sequence contains no elements") },
        new object[] { new NullReferenceException() },
        new object[] { new IndexOutOfRangeException() },
        new object[] { new ArgumentOutOfRangeException("rank") },
        new object[] { new OutOfMemoryException() },
    };

    [Theory]
    [MemberData(nameof(Bugs))]
    public void UnexpectedExceptions_AreNotRefusals(Exception bug)
    {
        Assert.False(ModelLoadRefusal.TryDescribe(bug, out string reason));
        Assert.Null(reason);
    }

    [Fact]
    public void ARefusedLoadExceptionIsStillAnInvalidOperationException()
    {
        // Callers that already catch InvalidOperationException keep working.
        Assert.IsAssignableFrom<InvalidOperationException>(new ModelLoadRefusedException("x"));
    }

    [Fact]
    public void WrappersAreSeenThrough()
    {
        var inner = new NotSupportedException("--tp 2 cannot split 64 heads");
        Assert.True(ModelLoadRefusal.TryDescribe(new TargetInvocationException(inner), out string a));
        Assert.Equal(inner.Message, a);
        Assert.True(ModelLoadRefusal.TryDescribe(new AggregateException(inner), out string b));
        Assert.Equal(inner.Message, b);
        // Two different failures are not one refusal.
        Assert.False(ModelLoadRefusal.TryDescribe(new AggregateException(inner, new NullReferenceException()), out _));
    }

    [Fact]
    public void TheErrorLineIsOneLine()
    {
        string line = ModelLoadRefusal.FormatErrorLine("first line\r\nsecond\tline\n\n");
        Assert.Equal("error: model load refused: first line second line", line);
        Assert.DoesNotContain('\n', line);
    }

    [Fact]
    public void ExitCodes_AreTheDocumentedNumbers()
    {
        Assert.Equal(0, HostExitCodes.Success);
        Assert.Equal(1, HostExitCodes.ConfigurationError);
        Assert.Equal(2, HostExitCodes.ModelLoadRefused);
    }

    // ---- lifecycle ----------------------------------------------------------------

    [Fact]
    public void Refusal_IsLoggedAsItsReason_AndTheStackTraceOnlyAtDebug()
    {
        var logger = new RecordingLogger();
        var lifecycle = new ModelLifecycleService(logger,
            (_, _, _, _) => throw new NotSupportedException("KV_CACHE_DTYPE=q8_0 is not supported by DeepSeek V4.1 Flash."));

        Assert.Throws<NotSupportedException>(() => lifecycle.LoadModel(WriteMinimalGguf("v41.gguf"), null, "cpu"));

        RecordingLogger.Entry error = Assert.Single(logger.Entries, e => e.Level == LogLevel.Error);
        Assert.Null(error.Exception);
        Assert.Contains("KV_CACHE_DTYPE=q8_0 is not supported", error.Message);
        Assert.Contains(logger.Entries, e => e.Level == LogLevel.Debug && e.Exception is NotSupportedException);
        Assert.False(lifecycle.IsLoaded);
    }

    [Fact]
    public void UnexpectedFailure_KeepsItsStackTraceAtError()
    {
        var logger = new RecordingLogger();
        var lifecycle = new ModelLifecycleService(logger,
            (_, _, _, _) => throw new NullReferenceException("a bug"));

        Assert.Throws<NullReferenceException>(() => lifecycle.LoadModel(WriteMinimalGguf("bug.gguf"), null, "cpu"));

        RecordingLogger.Entry error = Assert.Single(logger.Entries, e => e.Level == LogLevel.Error);
        Assert.IsType<NullReferenceException>(error.Exception);
    }

    [Fact]
    public void Refusal_DuringAReload_RestoresThePreviousModel()
    {
        string pathA = WriteMinimalGguf("model-a.gguf");
        string pathB = WriteMinimalGguf("model-b.gguf");
        var lifecycle = new ModelLifecycleService(NullLogger.Instance, (path, _, _, _) =>
            path == pathB
                ? throw new ModelLoadRefusedException("[glm] not enough VRAM for --tp 3")
                : new FakeModel(path));
        lifecycle.LoadModel(pathA, null, "cpu");

        Assert.Throws<ModelLoadRefusedException>(() => lifecycle.LoadModel(pathB, null, "cpu"));

        Assert.True(lifecycle.IsLoaded);
        Assert.Equal("model-a.gguf", lifecycle.LoadedModelName);
        lifecycle.Dispose();
    }

    [Fact]
    public void Refusal_ReleasesTheTensorParallelGroupBuiltForIt()
    {
        var groups = new List<RecordingTpGroup>();
        var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (_, _, _, _) => throw new ModelLoadRefusedException("[glm] --tp 4 needs 4 GPUs; only 2 are visible"))
        {
            TensorParallelGroupFactory = _ =>
            {
                var group = new RecordingTpGroup();
                groups.Add(group);
                return group;
            },
        };

        Assert.Throws<ModelLoadRefusedException>(() => lifecycle.LoadModel(WriteMinimalGguf("tp.gguf"), null, "cpu"));

        RecordingTpGroup built = Assert.Single(groups);
        Assert.True(built.Disposed, "a group nobody owns any more must be released with the refused load");
    }

    // ---- API-triggered reloads ----------------------------------------------------

    [Fact]
    public void HostedModelReload_Refused_ReturnsAnErrorAndKeepsServing()
    {
        string pathA = WriteMinimalGguf("model-a.gguf");
        string hosted = WriteMinimalGguf("hosted.gguf");
        using var svc = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) =>
            path == hosted
                ? throw new NotSupportedException("KV_CACHE_DTYPE=q8_0 is not supported by DeepSeek V4.1 Flash.")
                : new FakeModel(path));
        svc.LoadModel(pathA, null, "cpu");

        bool ok = HostedModelGuard.TryEnsureHostedModelLoaded(svc, "hosted.gguf", hosted, null, "cpu", out string error);

        Assert.False(ok);
        Assert.Contains("hosted.gguf", error);
        Assert.Contains("KV_CACHE_DTYPE=q8_0 is not supported", error);
        Assert.True(svc.IsLoaded);
        Assert.Equal("model-a.gguf", svc.LoadedModelName);
    }

    [Fact]
    public void HostedModelReload_UnexpectedFailure_StillPropagates()
    {
        string hosted = WriteMinimalGguf("hosted.gguf");
        using var svc = new ModelService(NullLogger<ModelService>.Instance,
            (_, _, _, _) => throw new NullReferenceException("a bug"));

        Assert.Throws<NullReferenceException>(
            () => HostedModelGuard.TryEnsureHostedModelLoaded(svc, "hosted.gguf", hosted, null, "cpu", out _));
    }

    [Fact]
    public async Task WebUiLoad_Refused_Is500WithTheReasonAndTheModelStillLoaded()
    {
        string pathA = WriteMinimalGguf("model-a.gguf");
        string hosted = WriteMinimalGguf("hosted.gguf");
        using var svc = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) =>
            path == hosted
                ? throw new ModelLoadRefusedException("[dsv4] not enough VRAM: re-run with --n-cpu-moe 2")
                : new FakeModel(path));
        svc.LoadModel(pathA, null, "ggml_cpu");
        var service = new WebUiChatService(
            svc, new SessionManager(), Options(hosted), new UploadStoragePolicy(_dir),
            new SkillRegistry(new SkillRegistryOptions()),
            codeRunner: null, workspaces: null, codeArtifacts: null,
            NullLoggerFactory.Instance);

        var ex = await Assert.ThrowsAsync<WebUiRequestRejectedException>(() => service.LoadModelAsync(
            JsonSerializer.Deserialize<JsonElement>("""{"model":"hosted.gguf"}"""), CancellationToken.None));

        Assert.Equal(500, ex.StatusCode);
        using JsonDocument doc = JsonDocument.Parse(JsonSerializer.Serialize(ex.Payload));
        Assert.False(doc.RootElement.GetProperty("ok").GetBoolean());
        Assert.True(doc.RootElement.GetProperty("refused").GetBoolean());
        Assert.Contains("--n-cpu-moe 2", doc.RootElement.GetProperty("error").GetString());
        Assert.Equal("model-a.gguf", doc.RootElement.GetProperty("loadedModel").GetString());
        Assert.Equal("model-a.gguf", svc.LoadedModelName);
    }

    // ---- startup --------------------------------------------------------------------

    [Fact]
    public void StartupLoad_Refused_ThrowsARefusalAndHoldsNoModel()
    {
        string model = WriteMinimalGguf("startup.gguf");
        using var svc = new ModelService(NullLogger<ModelService>.Instance,
            (_, _, _, _) => throw new NotSupportedException("--tp 3 is not supported"));

        Exception ex = Record.Exception(() =>
            StartupModelLoader.LoadIfConfigured(Options(model), svc, "ggml_cpu", NullLogger.Instance));

        Assert.True(ModelLoadRefusal.TryDescribe(ex, out string reason), ex?.ToString());
        Assert.Equal("--tp 3 is not supported", reason);
        Assert.False(svc.IsLoaded);
    }

    [Fact]
    public void StartupLoad_UnsupportedBackend_IsARefusal()
    {
        string model = WriteMinimalGguf("startup.gguf");
        using var svc = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) => new FakeModel(path));

        Exception ex = Record.Exception(() =>
            StartupModelLoader.LoadIfConfigured(Options(model), svc, "ggml_vulkan", NullLogger.Instance));

        Assert.IsType<ModelLoadRefusedException>(ex);
    }

    [Fact]
    public void ReportRefusal_ReleasesFirst_ThenWritesOneLine_AndReturnsTheRefusalCode()
    {
        var order = new List<string>();
        var stderr = new RecordingWriter(order);

        int code = StartupModelLoader.ReportRefusal(
            new NotSupportedException("x"), "not enough VRAM\nre-run with --n-cpu-moe 2", stderr, NullLogger.Instance,
            () => order.Add("release"));

        Assert.Equal(HostExitCodes.ModelLoadRefused, code);
        Assert.Equal(new[] { "release", "error: model load refused: not enough VRAM re-run with --n-cpu-moe 2" }, order);
    }

    [Fact]
    public void ReportRefusal_AFailedRelease_DoesNotSwallowTheRefusal()
    {
        var order = new List<string>();
        int code = StartupModelLoader.ReportRefusal(
            new NotSupportedException("x"), "the reason", new RecordingWriter(order), NullLogger.Instance,
            () => throw new InvalidOperationException("teardown failed"));

        Assert.Equal(HostExitCodes.ModelLoadRefused, code);
        Assert.Equal(2, order.Count);
        Assert.Contains("teardown failed", order[0]);
        Assert.Equal("error: model load refused: the reason", order[^1]);
    }

    // ---- documentation ----------------------------------------------------------------

    [Theory]
    [InlineData("USAGE.md")]
    [InlineData("USAGE_zh-cn.md")]
    public void UsagePages_DocumentEveryExitCodeAndTheErrorLine(string page)
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        if (dir == null)
            return;

        string text = File.ReadAllText(Path.Combine(dir.FullName, page));
        foreach (int code in new[] { HostExitCodes.Success, HostExitCodes.ConfigurationError, HostExitCodes.ModelLoadRefused })
            Assert.Contains($"| `{code}` |", text);
        Assert.Contains(ModelLoadRefusal.FormatErrorLine("<"), text);
        Assert.Contains("\"refused\": true", text);
    }

    // ---- helpers ----------------------------------------------------------------------

    private ServerHostingOptions Options(string startupModelPath) => new(
        startupModelPath: startupModelPath,
        startupMmProjPath: null,
        defaultBackend: "ggml_cpu",
        supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
        defaultMaxTokens: 100,
        maxTokensPinned: false,
        defaultVideoFrames: 0,
        defaultVideoFps: 0,
        defaultVideoWidth: 0,
        defaultVideoHeight: 0,
        defaultVideoSteps: 0,
        defaultVideoMode: null,
        uploadDirectory: _dir,
        logDirectory: Path.Combine(_dir, "logs"),
        fileLoggingEnabled: false,
        samplingDefaults: null,
        prefixCacheEnabled: false);

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_dir, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version
        bw.Write(0UL);         // tensor count
        bw.Write(0UL);         // metadata count
        bw.Write(new byte[8]);
        return path;
    }

    private sealed class FakeModel : ModelBase
    {
        public FakeModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu)
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }
    }

    private sealed class RecordingTpGroup : ITensorParallelGroup
    {
        public bool Disposed { get; private set; }
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() => Disposed = true;
    }

    private sealed class RecordingLogger : ILogger
    {
        public sealed record Entry(LogLevel Level, string Message, Exception? Exception);

        public List<Entry> Entries { get; } = new();

        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

        public bool IsEnabled(LogLevel logLevel) => true;

        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter)
            => Entries.Add(new Entry(logLevel, formatter(state, exception), exception));
    }

    private sealed class RecordingWriter : StringWriter
    {
        private readonly List<string> _lines;

        public RecordingWriter(List<string> lines) => _lines = lines;

        public override void WriteLine(string? value) => _lines.Add(value ?? string.Empty);
    }
}
