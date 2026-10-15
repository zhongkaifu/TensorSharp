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
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;

namespace InferenceWeb.Tests;

/// <summary>
/// Behavioural tests for <see cref="FileLoggerProvider"/>. These cover the key
/// invariants: structured JSON shape, scope and event id propagation, drop
/// behaviour under back-pressure, and rollover on size/date changes.
/// </summary>
public class FileLoggerProviderTests
{
    private static FileLoggerOptions CreateOptions(string directory) => new()
    {
        Directory = directory,
        FilePrefix = "test",
        MinimumLevel = LogLevel.Trace,
        MaxFileSizeBytes = 64 * 1024,
        MaxQueuedEntries = 1024,
        FlushInterval = TimeSpan.FromMilliseconds(20),
    };

    private static string CreateTempDir()
    {
        string path = Path.Combine(Path.GetTempPath(), $"tensorsharp-log-{Guid.NewGuid():N}");
        Directory.CreateDirectory(path);
        return path;
    }

    private static List<JsonElement> ReadAllEntries(string directory)
    {
        var entries = new List<JsonElement>();
        foreach (var file in Directory.GetFiles(directory, "*.jsonl"))
        {
            foreach (var line in File.ReadAllLines(file))
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;
                using var doc = JsonDocument.Parse(line);
                entries.Add(doc.RootElement.Clone());
            }
        }
        return entries;
    }

    [Fact]
    public void Logger_WritesStructuredJsonLineWithLevelMessageAndTemplate()
    {
        string dir = CreateTempDir();
        try
        {
            var provider = new FileLoggerProvider(CreateOptions(dir));
            var logger = provider.CreateLogger("MyCategory");

            logger.LogInformation(new EventId(42, "TestEvent"),
                "Hello {Name}, count={Count}", "Alice", 7);

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)),
                "All entries should flush within the deadline");
            provider.Dispose();

            var entries = ReadAllEntries(dir);
            Assert.Single(entries);
            var entry = entries[0];

            Assert.Equal("Information", entry.GetProperty("level").GetString());
            Assert.Equal("MyCategory", entry.GetProperty("category").GetString());
            Assert.Equal(42, entry.GetProperty("event").GetProperty("id").GetInt32());
            Assert.Equal("TestEvent", entry.GetProperty("event").GetProperty("name").GetString());
            Assert.Equal("Hello Alice, count=7", entry.GetProperty("message").GetString());
            Assert.Equal("Hello {Name}, count={Count}", entry.GetProperty("template").GetString());

            var props = entry.GetProperty("props");
            Assert.Equal("Alice", props.GetProperty("Name").GetString());
            Assert.Equal(7, props.GetProperty("Count").GetInt32());
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Logger_RespectsMinimumLevel_AndDropsLowerEntries()
    {
        string dir = CreateTempDir();
        try
        {
            var options = CreateOptions(dir);
            options.MinimumLevel = LogLevel.Warning;
            var provider = new FileLoggerProvider(options);
            var logger = provider.CreateLogger("Cat");

            logger.LogTrace("trace");
            logger.LogDebug("debug");
            logger.LogInformation("info");
            logger.LogWarning("warning");
            logger.LogError("error");

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();

            var levels = ReadAllEntries(dir).Select(e => e.GetProperty("level").GetString()).ToList();
            Assert.Equal(new[] { "Warning", "Error" }, levels);
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Logger_PreservesScopePropertiesInOuterToInnerOrder()
    {
        string dir = CreateTempDir();
        try
        {
            var provider = new FileLoggerProvider(CreateOptions(dir));
            var logger = provider.CreateLogger("ScopeTests");

            using (logger.BeginScope(new Dictionary<string, object> { ["RequestId"] = "abc" }))
            using (logger.BeginScope(new Dictionary<string, object> { ["SessionId"] = "xyz" }))
            {
                logger.LogInformation("inside scope");
            }
            logger.LogInformation("outside scope");

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();

            var entries = ReadAllEntries(dir);
            Assert.Equal(2, entries.Count);

            var inside = entries[0];
            Assert.True(inside.TryGetProperty("scope", out var scope));
            Assert.Equal("abc", scope.GetProperty("RequestId").GetString());
            Assert.Equal("xyz", scope.GetProperty("SessionId").GetString());

            var outside = entries[1];
            Assert.False(outside.TryGetProperty("scope", out _));
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Logger_EmitsUniqueScopeMembersWithInnermostRequestIdentityAndRestoresParent()
    {
        string dir = CreateTempDir();
        try
        {
            using var provider = new FileLoggerProvider(CreateOptions(dir));
            var logger = provider.CreateLogger("ScopeIdentity");
            using (logger.BeginScope(new Dictionary<string, object>
                { ["RequestId"] = "transport-id", ["ConnectionId"] = "connection" }))
            {
                using (logger.BeginScope(new Dictionary<string, object>
                    { ["RequestId"] = "http-id", ["SessionId"] = "session" }))
                    logger.LogInformation("inner");
                logger.LogInformation("outer");
            }
            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();
            var entries = ReadAllEntries(dir);
            Assert.Equal(2, entries.Count);
            var inner = entries[0].GetProperty("scope");
            Assert.Equal("http-id", inner.GetProperty("RequestId").GetString());
            Assert.Equal("connection", inner.GetProperty("ConnectionId").GetString());
            Assert.Equal("session", inner.GetProperty("SessionId").GetString());
            Assert.Equal(inner.EnumerateObject().Count(), inner.EnumerateObject().Select(p => p.Name).Distinct().Count());
            Assert.Equal("transport-id", entries[1].GetProperty("scope").GetProperty("RequestId").GetString());
        }
        finally { TryCleanup(dir); }
    }

    [Fact]
    public void Logger_RepeatedTemplateKeysAreUnambiguousAndKeepTheLastValue()
    {
        string dir = CreateTempDir();
        try
        {
            using var provider = new FileLoggerProvider(CreateOptions(dir));
            var logger = provider.CreateLogger("RepeatedTemplate");
            logger.LogInformation("Counts {Count}/{Count}", 1, 2);
            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();
            var entry = Assert.Single(ReadAllEntries(dir));
            Assert.Equal("Counts 1/2", entry.GetProperty("message").GetString());
            Assert.Equal("Counts {Count}/{Count}", entry.GetProperty("template").GetString());
            Assert.Single(entry.GetProperty("props").EnumerateObject());
            Assert.Equal(2, entry.GetProperty("props").GetProperty("Count").GetInt32());
        }
        finally { TryCleanup(dir); }
    }

    [Fact]
    public void Logger_RecordsExceptionDetails()
    {
        string dir = CreateTempDir();
        try
        {
            var provider = new FileLoggerProvider(CreateOptions(dir));
            var logger = provider.CreateLogger("ExTests");

            try
            {
                throw new InvalidOperationException("boom");
            }
            catch (Exception ex)
            {
                logger.LogError(ex, "It failed");
            }

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();

            var entries = ReadAllEntries(dir);
            Assert.Single(entries);
            var entry = entries[0];

            Assert.Equal("Error", entry.GetProperty("level").GetString());
            string? exception = entry.GetProperty("exception").GetString();
            Assert.NotNull(exception);
            Assert.Contains("InvalidOperationException", exception);
            Assert.Contains("boom", exception);
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Flush_DoesNotAcknowledgeARecordStillBeingFormatted()
    {
        string dir = CreateTempDir();
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var provider = new FileLoggerProvider(CreateOptions(dir));
        try
        {
            var state = new[] { new KeyValuePair<string, object>("payload", new BlockingLogValue(entered, release)) };
            provider.CreateLogger("BlockedFormat").Log(LogLevel.Information, default, state, null,
                static (_, _) => "already enqueued");
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));

            var clock = System.Diagnostics.Stopwatch.StartNew();
            Assert.False(provider.Flush(TimeSpan.FromMilliseconds(150)));
            Assert.True(clock.Elapsed < TimeSpan.FromSeconds(1), "Flush timeout must not wait for blocked writer I/O or formatting.");
            Assert.Empty(Directory.GetFiles(dir, "*.jsonl"));

            release.Set();
            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            Assert.Equal(1, CountEntriesWhileWriterOpen(dir));
        }
        finally { release.Set(); provider.Dispose(); TryCleanup(dir); }
    }

    [Fact]
    public async Task Flush_CompletionIncludesTheInFlightRecordBeforeDispose()
    {
        string dir = CreateTempDir();
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var provider = new FileLoggerProvider(CreateOptions(dir));
        try
        {
            var state = new[] { new KeyValuePair<string, object>("payload", new BlockingLogValue(entered, release)) };
            provider.CreateLogger("BlockedFormat").Log(LogLevel.Information, default, state, null,
                static (_, _) => "already enqueued");
            Assert.True(entered.Wait(TimeSpan.FromSeconds(5)));
            Task<bool> flush = Task.Run(() => provider.Flush(TimeSpan.FromSeconds(2)));
            Task first = await Task.WhenAny(flush, Task.Delay(TimeSpan.FromMilliseconds(100)));
            Assert.NotSame(flush, first);
            release.Set();
            Assert.True(await flush);
            Assert.Equal(1, CountEntriesWhileWriterOpen(dir));
        }
        finally { release.Set(); provider.Dispose(); TryCleanup(dir); }
    }

    private sealed class BlockingLogValue(ManualResetEventSlim entered, ManualResetEventSlim release)
    {
        public override string ToString()
        {
            entered.Set();
            if (!release.Wait(TimeSpan.FromSeconds(5))) throw new TimeoutException("Blocked log formatter was not released.");
            return "payload";
        }
    }

    private static int CountEntriesWhileWriterOpen(string directory)
    {
        int count = 0;
        foreach (string file in Directory.GetFiles(directory, "*.jsonl"))
        {
            // Windows readers must allow the existing writer's write access.
            using var stream = new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
            using var reader = new StreamReader(stream);
            while (reader.ReadLine() is { } line)
            {
                using var document = JsonDocument.Parse(line);
                ++count;
            }
        }
        return count;
    }

    [Fact]
    public void Logger_RollsFileWhenSizeExceedsCap()
    {
        string dir = CreateTempDir();
        try
        {
            var options = CreateOptions(dir);
            options.MaxFileSizeBytes = 1024;
            var provider = new FileLoggerProvider(options);
            var logger = provider.CreateLogger("Roll");

            string filler = new string('x', 256);
            for (int i = 0; i < 32; i++)
                logger.LogInformation("Entry {Index} {Filler}", i, filler);

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();

            string[] files = Directory.GetFiles(dir, "*.jsonl");
            Assert.True(files.Length >= 2,
                $"Expected at least 2 rolled files, found {files.Length} in {dir}");
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Logger_DropsExcessEntriesWhenQueueOverflows()
    {
        string dir = CreateTempDir();
        try
        {
            var options = CreateOptions(dir);
            options.MaxQueuedEntries = 4;
            // Use a long flush interval so the producer can outpace the writer.
            options.FlushInterval = TimeSpan.FromMilliseconds(500);
            var provider = new FileLoggerProvider(options);
            var logger = provider.CreateLogger("Overflow");

            for (int i = 0; i < 200; i++)
                logger.LogInformation("Pressure {Index}", i);

            // Wait for the writer to drain whatever it can.
            provider.Flush(TimeSpan.FromSeconds(3));
            provider.Dispose();

            // We don't insist on a specific count; we only require that overflow drops
            // were tracked AND that fewer than the producer count made it to disk.
            Assert.True(provider.DroppedCount > 0,
                $"Expected at least one dropped entry under back-pressure (dropped={provider.DroppedCount})");
            int written = ReadAllEntries(dir).Count;
            Assert.True(written < 200,
                $"Expected fewer than 200 entries persisted under back-pressure, found {written}");
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    [Fact]
    public void Logger_TextFormatProducesPlainLineWhenJsonDisabled()
    {
        string dir = CreateTempDir();
        try
        {
            var options = CreateOptions(dir);
            options.UseJsonFormat = false;
            var provider = new FileLoggerProvider(options);
            var logger = provider.CreateLogger("Plain");

            logger.LogInformation("Hello {Name}", "world");

            Assert.True(provider.Flush(TimeSpan.FromSeconds(2)));
            provider.Dispose();

            string[] files = Directory.GetFiles(dir, "*.jsonl");
            Assert.Single(files);
            string content = File.ReadAllText(files[0]!).TrimEnd();
            Assert.Contains("INF", content);
            Assert.Contains("Plain", content);
            Assert.Contains("Hello world", content);
            Assert.DoesNotContain("\"level\":", content);
        }
        finally
        {
            TryCleanup(dir);
        }
    }

    private static void TryCleanup(string dir)
    {
        try
        {
            if (Directory.Exists(dir))
                Directory.Delete(dir, recursive: true);
        }
        catch
        {
            // best-effort
        }
    }
}
