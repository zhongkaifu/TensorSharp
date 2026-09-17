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
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sandbox;
using TensorSharp.AgentHost.CodeExec;

namespace TensorAgent.Tests;

/// <summary>
/// The prefix-cache warm-up: the throwaway request that forwards the prompt every
/// conversation begins with, so the user's first message does not have to.
///
/// <para>
/// None of this needs a model. What is pinned here is the machinery around the
/// generation — that it starts only when the engine is idle, that it goes through a
/// real session, that a turn stops it AND waits for it, and that shutting the host
/// down with a warm-up pending does not start a generation on a dying engine — because
/// every one of those was learned on a phone, and every one is invisible from a green
/// live run.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class PrefixCacheWarmupTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "tensoragent-warmup-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;

    private AgentAppHost Start()
    {
        var paths = new AgentPaths(Path.Combine(_root, "data"), Path.Combine(_root, "cache"));
        _host = new AgentAppHost(paths);
        _host.Server.Start();
        return _host;
    }

    public void Dispose()
    {
        _host?.Dispose();
        CodeEnvironment.Reset();
        try { Directory.Delete(_root, true); } catch { }
    }

    private static async Task WaitUntilAsync(Func<bool> condition, TimeSpan timeout)
    {
        var clock = System.Diagnostics.Stopwatch.StartNew();
        while (!condition() && clock.Elapsed < timeout)
            await Task.Delay(25);
    }

    /// <summary>A stream that records the request it was given and yields one frame.</summary>
    private sealed class RecordingStream
    {
        public JsonElement? Body { get; private set; }
        public int Calls;

        public async IAsyncEnumerable<object> Frames(JsonElement body, [EnumeratorCancellation] CancellationToken ct)
        {
            Body = body.Clone();
            Interlocked.Increment(ref Calls);
            await Task.Yield();
            yield return new { token = "hi" };
            yield return new { done = true };
        }
    }

    [Fact]
    public async Task TheWarmUpSpeaksThroughARealSessionWithOneTokenAndTheUsersThinkingDefault()
    {
        AgentAppHost host = Start();
        var stream = new RecordingStream();
        host.WarmUpFrames = stream.Frames;

        Assert.False(host.PrefixCacheIsWarm);
        host.WarmThePrefixCache();
        Assert.Equal(1, host.PrefixCacheWarmupsStarted);

        await WaitUntilAsync(() => host.PrefixCacheIsWarm, TimeSpan.FromSeconds(15));
        Assert.True(host.PrefixCacheIsWarm, "the warm-up did not finish");
        Assert.Equal(1, stream.Calls);

        JsonElement body = stream.Body!.Value;
        // A request with no session is served by the default session, which has no
        // workspace and is therefore declared a different set of tools -- a warmed
        // prompt no real turn sends. So it must carry a session the host created.
        string sessionId = body.GetProperty("sessionId").GetString()!;
        Assert.False(string.IsNullOrWhiteSpace(sessionId));
        Assert.NotNull(host.Sessions.GetSession(sessionId));
        // The session is never bound to a conversation: nothing is recorded and no
        // chat appears in the user's list.
        Assert.Empty(host.Conversations.List());
        Assert.Equal(1, body.GetProperty("maxTokens").GetInt32());
        Assert.False(body.GetProperty("think").GetBoolean());
        Assert.Single(body.GetProperty("messages").EnumerateArray());
        Assert.False(body.GetProperty("newChat").GetBoolean());
        Assert.False(body.TryGetProperty("skills", out _));
        Assert.False(body.TryGetProperty("tools", out _));
    }

    [Fact]
    public void DisabledRuntimeCacheDoesNotStartWarmup()
    {
        AgentAppHost host = Start();
        host.ModelService.EngineHost.SchedulerConfigOverride = new TensorSharp.Runtime.Scheduling.SchedulerConfig
        {
            EnablePrefixCaching = false,
        };
        var stream = new RecordingStream();
        host.WarmUpFrames = stream.Frames;
        host.WarmThePrefixCache();

        Assert.Equal(0, host.PrefixCacheWarmupsStarted);
        Assert.Equal(0, stream.Calls);
        Assert.False(host.PrefixCacheIsWarm);
    }

    [Fact]
    public async Task TheWarmUpFollowsTheThinkingDefaultBecauseGemmaPutsTheMarkerAtTheTopOfThePrompt()
    {
        AgentAppHost host = Start();
        var settings = host.Settings.Load();
        settings.ThinkByDefault = true;
        host.Settings.Save(settings);

        var stream = new RecordingStream();
        host.WarmUpFrames = stream.Frames;
        host.WarmThePrefixCache();
        await WaitUntilAsync(() => host.PrefixCacheIsWarm, TimeSpan.FromSeconds(15));

        Assert.True(host.PrefixCacheIsWarm);
        Assert.True(stream.Body!.Value.GetProperty("think").GetBoolean());
    }

    [Fact]
    public async Task ItDoesNotStartWhileATurnIsUsingTheEngine()
    {
        AgentAppHost host = Start();
        var stream = new RecordingStream();
        host.WarmUpFrames = stream.Frames;

        // A turn in flight: the manager reports busy until it is stopped.
        var turnRunning = new TaskCompletionSource();
        async IAsyncEnumerable<object> NeverEnding([EnumeratorCancellation] CancellationToken ct)
        {
            turnRunning.TrySetResult();
            await Task.Delay(Timeout.Infinite, ct);
            yield return new { done = true };
        }
        host.Turns.Start("conversation-1", NeverEnding);
        await turnRunning.Task;
        Assert.True(host.Turns.IsBusy);

        host.WarmThePrefixCache();
        Assert.Equal(0, host.PrefixCacheWarmupsStarted);
        await Task.Delay(300);
        Assert.Equal(0, stream.Calls);
        Assert.False(host.PrefixCacheIsWarm);

        host.Turns.StopAll();
    }

    [Fact]
    public async Task ATurnStopsTheWarmUpAndWaitsForItBeforeAskingTheEngineForAnything()
    {
        AgentAppHost host = Start();
        var entered = new TaskCompletionSource();
        var observedCancellation = new TaskCompletionSource<bool>();
        host.WarmUpFrames = Blocking;

        async IAsyncEnumerable<object> Blocking(JsonElement body, [EnumeratorCancellation] CancellationToken ct)
        {
            entered.TrySetResult();
            try
            {
                await Task.Delay(Timeout.Infinite, ct);
            }
            catch (OperationCanceledException)
            {
                observedCancellation.TrySetResult(true);
                throw;
            }
            yield return new { done = true };
        }

        host.WarmThePrefixCache();
        // Past the settling delay and inside the (fake) generation.
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(15));

        // A turn through the same route the page uses. There is no model, so the
        // chat service refuses -- but the gate in front of it runs FIRST, and that
        // is what this proves: by the time the turn's frames are asked for, the
        // warm-up has been cancelled and has actually finished.
        using var client = new HttpClient { BaseAddress = new Uri(host.Server.BaseUrl) };
        client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={host.Server.Token}");
        using var response = await client.PostAsync(
            "/api/chat",
            new StringContent("""{"messages":[{"role":"user","content":"hello"}]}""", Encoding.UTF8, "application/json"));
        string frames = await response.Content.ReadAsStringAsync();

        Assert.True(await observedCancellation.Task.WaitAsync(TimeSpan.FromSeconds(15)),
            "the warm-up did not observe cancellation before the turn ran");
        Assert.False(host.PrefixCacheIsWarm);
        Assert.Contains("No model", frames);
    }

    [Fact]
    public void ShuttingDownWithAWarmUpPendingNeverStartsIt()
    {
        AgentAppHost host = Start();
        var stream = new RecordingStream();
        host.WarmUpFrames = stream.Frames;

        // Still in its settling delay when the host goes away.
        host.WarmThePrefixCache();
        Assert.Equal(1, host.PrefixCacheWarmupsStarted);

        var clock = System.Diagnostics.Stopwatch.StartNew();
        host.Dispose();
        _host = null;

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(10), $"shutdown waited {clock.Elapsed.TotalSeconds:0.0}s on a warm-up");
        Assert.Equal(0, stream.Calls);
        Assert.False(host.PrefixCacheIsWarm);
    }

    [Fact]
    public async Task ASecondWarmUpReplacesTheFirstAndTheCacheIsNotCalledWarmByAStaleOne()
    {
        AgentAppHost host = Start();
        var first = new RecordingStream();
        host.WarmUpFrames = first.Frames;
        host.WarmThePrefixCache();

        var second = new RecordingStream();
        host.WarmUpFrames = second.Frames;
        host.WarmThePrefixCache();
        Assert.Equal(2, host.PrefixCacheWarmupsStarted);

        await WaitUntilAsync(() => host.PrefixCacheIsWarm, TimeSpan.FromSeconds(15));
        Assert.True(host.PrefixCacheIsWarm);
        Assert.Equal(0, first.Calls);
        Assert.Equal(1, second.Calls);
    }
}
