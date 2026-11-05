using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using TensorSharp.Cuda;
using Microsoft.Extensions.Logging;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class Qwen35ServingTraceTests
{
    private const BindingFlags PrivateInstance = BindingFlags.Instance | BindingFlags.NonPublic;
    private static Q35CudaEngine BareEngine(Q35CudaEngine.ServingTraceLog? trace = null)
    {
        var engine = (Q35CudaEngine)RuntimeHelpers.GetUninitializedObject(typeof(Q35CudaEngine));
        typeof(Q35CudaEngine).GetField("_servingTrace", PrivateInstance)!.SetValue(engine, trace);
        typeof(Q35CudaEngine).GetField("_m", PrivateInstance)!.SetValue(engine,
            new Q35CudaEngine.ModelDesc { NUbatch = 1 });
        return engine; // Only pure trace hooks or the original pre-CUDA row-cap rejection are used.
    }
    private static JsonElement[] Events(StringWriter writer) => writer.ToString()
        .Split('\n', StringSplitOptions.RemoveEmptyEntries)
        .Select(line => JsonDocument.Parse(line["[q35-serving-trace] ".Length..]).RootElement.Clone()).ToArray();
    private static Q35CudaEngine.RowGroup Group(Q35CudaEngine.Slot[] slots, int rows,
        bool verify = false, int[]? counts = null, bool decode = false) =>
        new(slots, rows, verify, counts, new[] { 19088743 }, decode: decode);

    [Fact]
    public void ExactOptInIsSnapshottedAndOffDoesNotCreateALog()
    {
        string? original = Environment.GetEnvironmentVariable("TS_Q35_SERVING_TRACE");
        try
        {
            foreach (string? value in new string?[] { null, "", "0", "true", "01", " 1", "1 " })
            {
                Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", value);
                Assert.Null(Q35CudaEngine.ServingTraceLog.FromEnvironment());
            }
            var writer = new StringWriter();
            var enabled = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", "0");
            enabled.Bind("chat-a", new Q35CudaEngine.Slot());
            Assert.Single(Events(writer)); // Later environment changes do not disable an existing log.
        }
        finally { Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", original); }
    }

    [Fact]
    public void DisabledRealEngineHooksDoNotAllocateOrCreateAWriterLedger()
    {
        var engine = BareEngine();
        var slot = new Q35CudaEngine.Slot();
        for (int i = 0; i < 100; i++) engine.TraceBindSequence("chat-a", slot);
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++) engine.TraceBindSequence("chat-a", slot);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - before);
        Assert.False(engine.ServingTraceEnabled);
        Assert.Equal(0, engine.ServingTraceDroppedEvents);
        Assert.Null(typeof(Q35CudaEngine).GetField("_servingTrace", PrivateInstance)!.GetValue(engine));
    }

    [Fact]
    public void RealRunGroupsRejectsTheSameOverflowBeforeCudaWithTraceOffOrOn()
    {
        string? originalMessage = null;
        foreach (bool enabled in new[] { false, true })
        {
            var writer = new StringWriter();
            var engine = BareEngine(Q35CudaEngine.ServingTraceLog.Create(enabled ? "1" : "0", writer));
            var groups = new[] { Group(new[] { new Q35CudaEngine.Slot() }, 100) };
            var error = Assert.Throws<TargetInvocationException>(() => typeof(Q35CudaEngine)
                .GetMethod("RunGroups", PrivateInstance)!.Invoke(engine, new object[] { 0, groups }));
            var inner = Assert.IsType<ArgumentException>(error.InnerException);
            if (!enabled) originalMessage = inner.Message;
            Assert.Equal(originalMessage, inner.Message);
            if (enabled)
            {
                var metadata = Assert.Single(Events(writer)).GetProperty("metadata");
                Assert.Equal("failed", metadata.GetProperty("outcome").GetString());
                Assert.False(metadata.GetProperty("layer_branch_observed").GetBoolean());
                Assert.Equal("NOT_QUERIED_BY_TRACE", metadata.GetProperty("device_completion").GetString());
            }
            else Assert.Empty(Events(writer));
        }
    }

    [Fact]
    public void SlotIdentitySurvivesRebindingAndPrimaryUnmappedIsExplicit()
    {
        var writer = new StringWriter();
        var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        var engine = BareEngine(trace);
        var slot = new Q35CudaEngine.Slot { NPast = 7, RopeDelta = -2, Version = 3 };
        engine.TraceBindSequence("chat-a", slot);
        engine.TraceBindSequence("chat-a", slot);
        engine.TraceBindSequence("chat-b", slot);
        engine.TraceBindSequence(null!, slot);
        var rows = Events(writer);
        Assert.Equal(3, rows.Length);
        Assert.Single(rows.Select(r => r.GetProperty("metadata").GetProperty("slot_id").GetInt64()).Distinct());
        Assert.Equal(new long[] { 1, 2, 3 }, rows.Select(r => r.GetProperty("metadata").GetProperty("binding_generation").GetInt64()));
        Assert.Equal("primary_unmapped", rows[2].GetProperty("metadata").GetProperty("binding_status").GetString());
        Assert.Equal(JsonValueKind.Null, rows[2].GetProperty("metadata").GetProperty("engine_sequence_id").ValueKind);
        Assert.Equal(7, slot.NPast);
        Assert.Equal(-2, slot.RopeDelta);
        Assert.Equal(3, slot.Version);
    }

    [Fact]
    public void RaggedMixedForwardRecordsRealRowsWindowAndActualObservedBranchOnly()
    {
        var writer = new StringWriter();
        var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        var prompt = new Q35CudaEngine.Slot { NPast = 100 };
        var a = new Q35CudaEngine.Slot { NPast = 200 };
        var b = new Q35CudaEngine.Slot { NPast = 300 };
        trace.Bind("chat-prompt", prompt); trace.Bind("chat-a", a); trace.Bind("chat-b", b);
        var groups = new[] { Group(new[] { prompt }, 4096, counts: new[] { 3991 }), Group(new[] { a, b }, 4, verify: true) };
        var capture = trace.BeginForward(4096, groups, 2);
        trace.LayerBranch("native_default_W4A4"); // Synthetic CPU branch observation, never a GPU dispatch claim.
        prompt.NPast += 3991;
        trace.EndForward(capture, groups, 2, null!);
        var run = Events(writer).Last();
        Assert.Equal(Q35CudaEngine.ServingTraceLog.Schema, run.GetProperty("schema").GetString());
        var meta = run.GetProperty("metadata");
        Assert.Equal(4104, meta.GetProperty("total_rows").GetInt32());
        Assert.True(meta.GetProperty("layer_branch_observed").GetBoolean());
        Assert.Equal("native_default_W4A4", meta.GetProperty("layer_branch").GetString());
        var before = meta.GetProperty("groups_before");
        Assert.Equal(3991, before[0].GetProperty("real_rows").GetInt32());
        Assert.Equal(4096, before[0].GetProperty("padded_rows").GetInt32());
        Assert.Equal(4, before[1].GetProperty("rows_per_sequence").GetInt32());
        Assert.Equal("verify", before[1].GetProperty("kind").GetString());
        Assert.Equal("chat-b", before[1].GetProperty("slots")[1].GetProperty("engine_sequence_id").GetString());
        Assert.Equal(100, before[0].GetProperty("slots")[0].GetProperty("npast").GetInt32());
        Assert.Equal(4091, meta.GetProperty("groups_after")[0].GetProperty("slots")[0].GetProperty("npast").GetInt32());
        Assert.DoesNotContain("19088743", writer.ToString()); // Never serializes the supplied token array.
    }

    [Fact]
    public void GraphReplayAndFailedForwardDoNotFabricateALayerBranch()
    {
        var writer = new StringWriter();
        var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        var groups = new[] { Group(new[] { new Q35CudaEngine.Slot() }, 8, verify: true) };
        var graph = trace.BeginForward(0, groups, 4);
        trace.EndForward(graph, groups, 5, null!);
        var failed = trace.BeginForward(0, groups, 5);
        trace.EndForward(failed, groups, 5, "ExampleFailure");
        trace.HeadCompleted(1, 100, false, true);
        var events = Events(writer);
        Assert.Equal(1, events[0].GetProperty("metadata").GetProperty("verify_graph_replay_delta").GetInt64());
        Assert.False(events[0].GetProperty("metadata").GetProperty("layer_branch_observed").GetBoolean());
        Assert.Equal("NOT_OBSERVED_ON_HOST", events[0].GetProperty("metadata").GetProperty("layer_branch").GetString());
        Assert.Equal(0, events[2].GetProperty("metadata").GetProperty("forward_id").GetInt64());
    }

    [Fact]
    public void CommitAndHostCopyJoinTheSuccessfulVerifyWithoutReadingValues()
    {
        var writer = new StringWriter();
        var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        var a = new Q35CudaEngine.Slot { NPast = 30 };
        var b = new Q35CudaEngine.Slot { NPast = 40 };
        trace.Bind("chat-a", a); trace.Bind("chat-b", b);
        var slots = new[] { a, b }; var groups = new[] { Group(slots, 8, verify: true) };
        var capture = trace.BeginForward(0, groups, 0);
        trace.LayerBranch("native_forced_Q81_chunks_le8");
        trace.EndForward(capture, groups, 0, null!);
        var kept = new[] { 2, 7 }; a.NPast = 32; b.NPast = 47;
        trace.CommitCompleted(slots, 8, kept, new[] { 30, 40 });
        trace.HeadCompleted(16, 248320, true, false);
        kept[0] = 0;
        var events = Events(writer); var commit = events[^2].GetProperty("metadata");
        Assert.Equal(8, commit.GetProperty("window").GetInt32());
        Assert.Equal(2, commit.GetProperty("slots")[0].GetProperty("kept").GetInt32());
        Assert.Equal(capture.Id, commit.GetProperty("slots")[1].GetProperty("verify_forward_id").GetInt64());
        var head = events[^1].GetProperty("metadata");
        Assert.Equal(capture.Id, head.GetProperty("forward_id").GetInt64());
        Assert.Equal(16L * 248320 * 4, head.GetProperty("logits_bytes_copied").GetInt64());
        Assert.Equal(0, head.GetProperty("argmax_bytes_copied").GetInt64());
    }

    [Fact]
    public void NewSlotWithoutBindingIsUnmappedAndExternalIdsAreBounded()
    {
        var writer = new StringWriter();
        var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        var slot = new Q35CudaEngine.Slot(); var groups = new[] { Group(new[] { slot }, 1, decode: true) };
        var capture = trace.BeginForward(0, groups, 0); trace.EndForward(capture, groups, 0, null!);
        var initial = Events(writer)[0].GetProperty("metadata").GetProperty("groups_before")[0].GetProperty("slots")[0];
        Assert.Equal("unmapped", initial.GetProperty("binding_status").GetString());
        trace.Bind(new string('x', 512), slot);
        var bound = Events(writer)[1].GetProperty("metadata");
        Assert.Equal(128, bound.GetProperty("engine_sequence_id").GetString()!.Length);
        Assert.True(bound.GetProperty("request_id_truncated").GetBoolean());
    }

    [Fact]
    public void DiagnosticWriterFailureIsCountedAndDoesNotEscapeIntoModelOperation()
    {
        var writer = new FailingOnceWriter(); var trace = Q35CudaEngine.ServingTraceLog.Create("1", writer)!;
        trace.Bind("chat-a", new Q35CudaEngine.Slot());
        Assert.Equal(1, trace.DroppedEvents);
        trace.HeadCompleted(1, 10, false, false);
        Assert.Equal(1, Assert.Single(Events(writer)).GetProperty("previously_dropped_events").GetInt64());
    }
    [Fact]
    public void OptionalChatMappingLoggerFailureDoesNotAbortTheRequestPath()
    {
        TensorSharp.Server.ChatGenerationPipeline.LogServingTraceRequestMapping(new ThrowingLogger(), "chat-a");
    }
    [Fact]
    public void DedicatedFileIsAutoFlushedAndClosesExactlyOnceWithTerminalIndices()
    {
        using var files = new TraceFiles();
        string path = files.PathOf("trace.jsonl");
        var trace = Q35CudaEngine.ServingTraceLog.CreateFile(path);
        var slot = new Q35CudaEngine.Slot { NPast = 7 };
        trace.Bind("chat-file", slot);
        Assert.Equal(2, FileEvents(path).Length); // AutoFlush is visible before disposal.
        var groups = new[] { Group(new[] { slot }, 8, verify: true) };
        var capture = trace.BeginForward(0, groups, 0);
        trace.LayerBranch("native_default_W4A8_Q81");
        trace.EndForward(capture, groups, 0, null!);
        trace.HeadCompleted(8, 100, true, false);
        slot.NPast = 11;
        trace.CommitCompleted(new[] { slot }, 8, new[] { 4 }, new[] { 7 });
        trace.Dispose(); trace.Dispose();
        var rows = FileEvents(path);
        Assert.All(rows, row => Assert.Equal(Q35CudaEngine.ServingTraceLog.FileSchema, row.GetProperty("schema").GetString()));
        Assert.Equal("trace_opened", rows[0].GetProperty("kind").GetString());
        Assert.Equal("trace_closed", rows[^1].GetProperty("kind").GetString());
        Assert.Equal(Enumerable.Range(1, rows.Length).Select(x => (long)x), rows.Select(x => x.GetProperty("event_index").GetInt64()));
        var terminal = rows[^1].GetProperty("metadata");
        Assert.Equal(rows.Length - 1L, terminal.GetProperty("last_event_index").GetInt64());
        Assert.Equal(capture.Id, terminal.GetProperty("last_forward_id").GetInt64());
        Assert.Equal(JsonValueKind.Null, terminal.GetProperty("active_forward_id").ValueKind);
        Assert.Equal(0, terminal.GetProperty("dropped_events").GetInt64());
        using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
    }

    [Fact]
    public void DedicatedFileRejectsExistingDanglingAliasedAndUnnormalizedPaths()
    {
        using var files = new TraceFiles();
        string existing = files.PathOf("existing"); File.WriteAllText(existing, "preserve");
        Assert.Throws<IOException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(existing));
        Assert.Equal("preserve", File.ReadAllText(existing));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.ServingTraceLog.CreateFile("relative.jsonl"));
        Assert.Throws<ArgumentException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(files.PathOf("nested/../trace.jsonl")));
        Assert.Throws<DirectoryNotFoundException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(files.PathOf("missing/trace.jsonl")));
        Assert.Throws<IOException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(files.Root));
        string dangling = files.PathOf("dangling");
        File.CreateSymbolicLink(dangling, files.PathOf("not-created"));
        Assert.Throws<IOException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(dangling));
        Assert.False(File.Exists(files.PathOf("not-created")));
        string target = files.PathOf("real"); Directory.CreateDirectory(target);
        string alias = files.PathOf("alias"); Directory.CreateSymbolicLink(alias, target);
        Assert.Throws<IOException>(() => Q35CudaEngine.ServingTraceLog.CreateFile(Path.Combine(alias, "foreign.jsonl")));
        Assert.Empty(Directory.EnumerateFileSystemEntries(target));
    }

    [Fact]
    public void DisabledEnvironmentIgnoresFileConfigurationAndEnabledSinkIsSnapshotted()
    {
        using var files = new TraceFiles();
        string? enabled = Environment.GetEnvironmentVariable("TS_Q35_SERVING_TRACE");
        string? file = Environment.GetEnvironmentVariable("TS_Q35_SERVING_TRACE_FILE");
        string path = files.PathOf("selected.jsonl");
        try
        {
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE_FILE", path);
            foreach (string? value in new string?[] { null, "0", "true", "01", "" })
            {
                Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", value);
                Assert.Null(Q35CudaEngine.ServingTraceLog.FromEnvironment());
                Assert.False(File.Exists(path));
            }
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", "1");
            using var trace = Q35CudaEngine.ServingTraceLog.FromEnvironment()!;
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE_FILE", files.PathOf("not-selected.jsonl"));
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", "0");
            trace.Bind("chat-selected", new Q35CudaEngine.Slot());
            Assert.Equal(2, FileEvents(path).Length);
            Assert.False(File.Exists(files.PathOf("not-selected.jsonl")));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE", enabled);
            Environment.SetEnvironmentVariable("TS_Q35_SERVING_TRACE_FILE", file);
        }
    }

    [Fact]
    public void FileWriteFailureRemainsBestEffortAndLegacyWriterIsNotDisposed()
    {
        using var files = new TraceFiles();
        string path = files.PathOf("failed.jsonl");
        var trace = Q35CudaEngine.ServingTraceLog.CreateFile(path);
        // Force a real stream failure after a successfully flushed opening marker.
        ((TextWriter)typeof(Q35CudaEngine.ServingTraceLog).GetField("_writer", PrivateInstance)!.GetValue(trace)!).Dispose();
        trace.Bind("chat-failure", new Q35CudaEngine.Slot());
        trace.Dispose();
        Assert.True(trace.DroppedEvents > 0);
        Assert.DoesNotContain(FileEvents(path), row => row.GetProperty("kind").GetString() == "trace_closed");
        var inherited = new StringWriter();
        var legacy = Q35CudaEngine.ServingTraceLog.Create("1", inherited)!;
        legacy.Dispose(); inherited.Write("still-open");
        Assert.Equal("still-open", inherited.ToString());
    }

    [Fact]
    public void ConstructorClosesOnlyItsTraceOnTheOriginalPreCudaPolicyRefusal()
    {
        using var files = new TraceFiles();
        string[] keys = { "TS_Q35_SERVING_TRACE", "TS_Q35_SERVING_TRACE_FILE", "TS_Q35_VERIFY_DECODE_Q81", "TS_Q35_VERIFY_GRAPHS" };
        var original = keys.Select(Environment.GetEnvironmentVariable).ToArray();
        string path = files.PathOf("constructor.jsonl");
        try
        {
            Environment.SetEnvironmentVariable(keys[0], "1"); Environment.SetEnvironmentVariable(keys[1], path);
            Environment.SetEnvironmentVariable(keys[2], "1"); Environment.SetEnvironmentVariable(keys[3], "1");
            Assert.Throws<NotSupportedException>(() => new Q35CudaEngine(new Q35CudaEngine.ModelDesc()));
            Assert.False(File.Exists(path)); // Basic parameter refusal does not open a trace.
            Assert.Throws<NotSupportedException>(() => new Q35CudaEngine(new Q35CudaEngine.ModelDesc
                { HeadDim = Q35Kernels.AttnHeadDim, NHead = 4, NKvHead = 1, DConv = 4 }));
            var rows = FileEvents(path);
            Assert.Equal(new[] { "trace_opened", "trace_closed" }, rows.Select(x => x.GetProperty("kind").GetString()));
            Assert.Equal("constructor_failed", rows[^1].GetProperty("metadata").GetProperty("reason").GetString());
            using var exclusive = new FileStream(path, FileMode.Open, FileAccess.ReadWrite, FileShare.None);
        }
        finally { for (int i = 0; i < keys.Length; i++) Environment.SetEnvironmentVariable(keys[i], original[i]); }
    }

    private static JsonElement[] FileEvents(string path) => File.ReadLines(path)
        .Select(line => JsonDocument.Parse(line).RootElement.Clone()).ToArray();
    private sealed class TraceFiles : IDisposable
    {
        public string Root { get; } = Path.Combine(AppContext.BaseDirectory, "trace-files-" + Guid.NewGuid().ToString("N"));
        public TraceFiles() => Directory.CreateDirectory(Root);
        public string PathOf(string name) => Path.Combine(Root, name);
        public void Dispose() => Directory.Delete(Root, recursive: true);
    }

    private sealed class ThrowingLogger : ILogger
    {
        public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
            Func<TState, Exception?, string> formatter) => throw new IOException("diagnostic provider unavailable");
    }
    private sealed class FailingOnceWriter : StringWriter
    {
        private bool _failed;
        public override void WriteLine(string? value)
        {
            if (!_failed) { _failed = true; throw new IOException("diagnostic writer unavailable"); }
            base.WriteLine(value);
        }
    }
}
