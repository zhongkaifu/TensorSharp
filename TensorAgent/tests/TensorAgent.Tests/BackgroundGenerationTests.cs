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
using System.Net.Http.Json;
using System.Text.Json;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Settings;

namespace TensorAgent.Tests;

/// <summary>
/// What happens to a generation when the app is not in front of the user.
///
/// <para>
/// The bug these are written against: switching away from TensorAgent during an answer
/// made that answer fail, AND every answer after it, until the app was force-quit. The
/// cause is a rule and a reaction. iOS refuses GPU work from an app that is not
/// frontmost — no entitlement grants it and there is no background mode for compute on
/// iPhone — and ggml-metal's response to a refused command buffer is to latch:
/// <c>ggml_metal_synchronize</c> sets <c>ctx-&gt;has_error</c>, and
/// <c>ggml_metal_graph_compute</c> then returns <c>GGML_STATUS_FAILED</c> for the rest
/// of the process, with a comment saying the backend must be recreated to clear it.
/// TensorAgent made this worse than a race rather than better: it deliberately holds a
/// background-task assertion so a turn keeps running after the app leaves, which
/// guaranteed thirty seconds of submissions the GPU was never going to accept.
/// </para>
/// <para>
/// The GPU itself cannot be tested from a terminal. What can be, and is what these do,
/// is the mechanism built around it: that the host's gate reaches the engine, that the
/// app stops PULLING the generation while the gate is closed, that it resumes where it
/// stopped, and that a fault suffered while away is remembered as one the engine cannot
/// recover from on its own. <c>BackgroundGenerationLiveTests</c> puts a real model
/// behind the same gate.
/// </para>
/// </summary>
[Collection(ProcessEnvironmentCollection.Name)]
public sealed class BackgroundGenerationTests : IDisposable
{
    private readonly string _root = Path.Combine(
        Path.GetTempPath(), "tensoragent-background-" + Guid.NewGuid().ToString("N"));
    private AgentAppHost? _host;
    private HttpClient? _client;

    private AgentAppHost Start()
    {
        _host = new AgentAppHost(new AgentPaths(
            Path.Combine(_root, "data"), Path.Combine(_root, "cache")));
        _host.Start();
        _client = new HttpClient
        {
            BaseAddress = new Uri(_host.Server.BaseUrl),
            Timeout = TimeSpan.FromMinutes(2),
        };
        _client.DefaultRequestHeaders.Add("Cookie", $"{LoopbackServer.TokenCookie}={_host.Server.Token}");
        return _host;
    }

    public void Dispose()
    {
        _client?.Dispose();
        _host?.Dispose();
        TensorSharp.AgentHost.CodeExec.CodeEnvironment.Reset();
        try { Directory.Delete(_root, true); } catch (Exception) { /* scratch */ }
    }

    private static object Message(string text) => new
    {
        messages = new[] { new { role = "user", content = text } },
        maxTokens = 16,
        think = false,
    };

    /// <summary>
    /// Send a chat request and return the first thing the server says, or null if it
    /// says nothing within <paramref name="budget"/>.
    ///
    /// <para>
    /// Headers-read, not content-read: with the gate closed the body never completes,
    /// and the point of the test is precisely that. And "the first thing", not "the
    /// first SSE frame", because what a turn with no model answers — an event-stream
    /// error frame, or a refusal with its own status — is not what these are about; the
    /// question they ask is whether the server answered AT ALL while the app was away.
    /// </para>
    /// </summary>
    private async Task<string?> SayAsync(string text, TimeSpan budget)
    {
        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
        {
            Content = JsonContent.Create(Message(text)),
        };
        using var deadline = new CancellationTokenSource(budget);
        try
        {
            using HttpResponseMessage response = await _client!.SendAsync(
                request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
            await using Stream body = await response.Content.ReadAsStreamAsync(deadline.Token);
            using var reader = new StreamReader(body);
            while (await reader.ReadLineAsync(deadline.Token) is { } line)
            {
                if (line.Trim().Length > 0)
                    return line;
            }
            return $"(empty body, HTTP {(int)response.StatusCode})";
        }
        catch (OperationCanceledException)
        {
            return null;
        }
    }

    // =====================================================================================
    // the gate, through the real route
    // =====================================================================================

    [Fact]
    public void TheHostHasAGateAndItIsOpenWhileTheAppIsInFront()
    {
        AgentAppHost host = Start();
        Assert.True(host.Compute.IsOpen);
        Assert.False(host.EngineNeedsReload);
        Assert.Equal(0, host.EngineRebuilds);
    }

    [Fact]
    public void TheGateReachesTheEngineHostSoTheStepLoopWillHonourIt()
    {
        // The engine decodes on its own thread into an unbounded channel. A wrapper
        // that stops pulling stops nothing; the gate has to be the ENGINE's. The engine
        // itself is built lazily with the first model, so what can be checked here is
        // that the host it will be built by already holds the same gate object.
        AgentAppHost host = Start();
        Assert.Same(host.Compute, host.ModelService.EngineHost.ComputeGate);
    }

    [Fact]
    public async Task AClosedGateStopsTheTurnBeforeItTouchesTheEngineAtAll()
    {
        // No model is loaded here, so an ungated turn answers with a refusal
        // immediately. That is exactly what makes this a test of the WIRING: with the
        // gate closed the refusal must not arrive either, because the gate is consulted
        // before the chat stream is pulled for the first time.
        AgentAppHost host = Start();
        host.Compute.Close();

        Assert.Null(await SayAsync("hello", TimeSpan.FromSeconds(2)));
    }

    [Fact]
    public async Task OpeningTheGateLetsTheTurnGoOnFromWhereItStopped()
    {
        AgentAppHost host = Start();
        host.Compute.Close();

        // Started while away, exactly as a notification tap can do.
        Task<string?> pending = SayAsync("hello", TimeSpan.FromSeconds(30));
        await Task.Delay(300);
        Assert.False(pending.IsCompleted, "the turn answered while the gate was closed");

        host.Compute.Open();

        Assert.NotNull(await pending);
    }

    [Fact]
    public async Task ATurnParkedOnTheGateStillEndsWhenItIsStopped()
    {
        // Otherwise Dispose waits forever: WaitForTheEngineToStop cannot return while a
        // turn is parked, and shutting down with the app in the background is the
        // ordinary case rather than the odd one.
        AgentAppHost host = Start();
        host.Compute.Close();

        _ = SayAsync("hello", TimeSpan.FromSeconds(60));
        await Task.Delay(300);

        var clock = Stopwatch.StartNew();
        host.Dispose();
        _host = null;
        clock.Stop();

        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(30),
            $"disposing with a turn parked on the gate took {clock.Elapsed.TotalSeconds:0.#} s");
    }

    // =====================================================================================
    // remembering that the engine may be broken
    // =====================================================================================

    [Fact]
    public async Task ARefusalIsNotMistakenForEngineDamageHoweverLongTheAppWasAway()
    {
        // "No model is loaded" is a decision, not a fault. Reading it as engine damage
        // would throw away a loaded model on the first message after every background
        // switch, to fix nothing.
        AgentAppHost host = Start();
        host.Compute.Close();
        host.Compute.Open();
        Assert.Equal(1, host.Compute.Closures);

        Assert.NotNull(await SayAsync("hello", TimeSpan.FromSeconds(20)));

        Assert.False(host.EngineNeedsReload);
    }

    [Fact]
    public void RecoveryIsAskedForOnlyWhenSomethingActuallyNeedsIt()
    {
        AgentAppHost host = Start();
        // Nothing has failed, so nothing is reloaded — a reload costs seconds of
        // reading weights and must never happen speculatively.
        Assert.False(host.RecoverEngineIfNeeded());
        Assert.False(host.EngineNeedsReload);
        Assert.Equal(0, host.EngineRebuilds);
    }

    [Fact]
    public void WithNoModelSelectedThereIsNoPoisonedEngineToRebuild()
    {
        // The flag is cleared rather than retried forever: with nothing loaded there is
        // no Metal backend in an error state either.
        AgentAppHost host = Start();
        Assert.False(host.RecoverEngineIfNeeded());

        AppSettings settings = host.Settings.Load();
        Assert.True(settings.SelectedModelId is null or { Length: 0 });
    }

    // =====================================================================================
    // telling a dead engine apart from a turn that merely failed
    // =====================================================================================

    /// <summary>
    /// The message a real phone produced, verbatim, when TensorAgent was sent to the
    /// background one token into an answer. It is here rather than paraphrased because
    /// the classifier is matched against wording, and wording is exactly the thing a
    /// paraphrase gets wrong.
    /// </summary>
    private const string WhatTheDeviceSaid =
        "The GGML GgmlMetal backend failed during GPU execution and cannot recover in this "
        + "process — the results of this and any preceding forward are undefined. Restart "
        + "the host. ggml reported: ggml_metal_synchronize: error: command buffer 0 failed "
        + "with status 5 | error: Insufficient Permission (to submit GPU work from background) "
        + "(00000006:kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted)";

    /// <summary>The one the command buffers queued BEHIND the refused one come back with.</summary>
    private const string WhatTheVictimsSaid =
        "ggml_metal_synchronize: error: command buffer 0 failed with status 5 | error: "
        + "Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)";

    private const string WhatTheNextTurnSaid =
        "Native GGML RmsNorm failed. ggml backend graph execution failed. ggml: "
        + "ggml_metal_graph_compute: backend is in error state from a previous command "
        + "buffer failure - recreate the backend to clear it";

    [Theory]
    [InlineData(WhatTheDeviceSaid)]
    [InlineData(WhatTheVictimsSaid)]
    [InlineData(WhatTheNextTurnSaid)]
    [InlineData("Insufficient Permission (to submit GPU work from background)")]
    [InlineData("kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted")]
    public void TheMessagesARefusedGpuActuallyProducesAreReadAsADeadEngine(string error)
    {
        Assert.NotNull(AgentAppHost.ReadsLikeAPoisonedEngine(error));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("No model is loaded.")]
    [InlineData("network access is disabled by the user")]
    [InlineData("The loaded model is not a Qwen-Image-2.1 model.")]
    [InlineData("Bad request: expected an object")]
    [InlineData("The context is full; start a new conversation.")]
    [InlineData("Out of memory allocating the KV cache.")]
    [InlineData("KV cache capacity exceeded: no running sequence can make progress")]
    public void AnOrdinaryFailureIsNotWorthThrowingTheEngineAwayFor(string? error)
    {
        // Getting this wrong in this direction costs the user a full model reload -- and
        // then another on the next message, and the one after that.
        Assert.Null(AgentAppHost.ReadsLikeAPoisonedEngine(error));
    }

    [Fact]
    public void TheFaultIsRecognisedThroughTheStreamFrameAndNotOnlyThroughAThrow()
    {
        // The first device run failed precisely here: the turn died, the wrapper watched
        // for an exception, the chat service had reported the fault as a `done` frame
        // instead, and so the engine stayed marked healthy while every later message
        // failed too. The frames are anonymous types, so the lookup is by property name.
        object frame = new { done = true, tokenCount = 0, error = WhatTheDeviceSaid };
        string? error = frame.GetType().GetProperty("error")?.GetValue(frame) as string;

        Assert.Equal(WhatTheDeviceSaid, AgentAppHost.ReadsLikeAPoisonedEngine(error));
        Assert.Null(AgentAppHost.ReadsLikeAPoisonedEngine(
            new { done = true, tokenCount = 12 }.GetType().GetProperty("error")?.GetValue(new { done = true }) as string));
    }

    // =====================================================================================
    // never two generations on one engine
    // =====================================================================================

    /// <summary>The repository root, found by walking up for a known marker.</summary>
    private static string FindRepoRoot()
    {
        var here = new DirectoryInfo(AppContext.BaseDirectory);
        while (here is not null && !Directory.Exists(Path.Combine(here.FullName, "TensorSharp.Runtime")))
            here = here.Parent;
        return here?.FullName ?? throw new DirectoryNotFoundException("no repository root above " + AppContext.BaseDirectory);
    }

    [Fact]
    public void TheWarmUpIsStoppedBeforeTheRebuildRatherThanAfterIt()
    {
        // A rebuild unloads the model and frees the backend. Doing that while a warm-up
        // is still generating tears the weights out from under a live step, so the order
        // matters: stopping the warm-up AFTER the rebuild protected the turn and left
        // the rebuild itself exposed. Read from the source, because the ordering is the
        // property and no observable state distinguishes the two arrangements.
        string host = File.ReadAllText(Path.Combine(
            FindRepoRoot(), "TensorAgent", "src", "TensorAgent.Core", "Hosting", "AgentAppHost.cs"));
        int loop = host.IndexOf("for (int attempt = 0; ; attempt++)", StringComparison.Ordinal);
        Assert.True(loop > 0, "the retry loop has been renamed");

        int stop = host.IndexOf("StopWarmingThePrefixCacheAndWaitAsync", loop, StringComparison.Ordinal);
        int rebuild = host.IndexOf("RecoverEngineIfNeeded(warmAfterwards: false)", loop, StringComparison.Ordinal);
        Assert.True(stop > 0 && rebuild > 0, "the retry loop no longer stops the warm-up or rebuilds");
        Assert.True(stop < rebuild,
            "the warm-up must be stopped BEFORE the rebuild that would unload the model under it");
    }

    [Fact]
    public void TheRebuildWaitsForTheForegroundBeforeTouchingTheGpu()
    {
        // Loading a model is GPU work too. A rebuild attempted at the moment of
        // backgrounding produces a backend poisoned before its first token, which reads
        // in the trace as a repair that worked and an answer that died anyway.
        string host = File.ReadAllText(Path.Combine(
            FindRepoRoot(), "TensorAgent", "src", "TensorAgent.Core", "Hosting", "AgentAppHost.cs"));
        int check = host.IndexOf("if (EngineNeedsReload)", StringComparison.Ordinal);
        Assert.True(check > 0, "the retry loop no longer checks whether the engine needs a rebuild");
        int wait = host.IndexOf("WaitForTheAppToBeInFrontAsync", check, StringComparison.Ordinal);
        int rebuild = host.IndexOf("RecoverEngineIfNeeded(warmAfterwards: false)", check, StringComparison.Ordinal);
        Assert.True(wait > 0 && wait < rebuild, "the rebuild must wait for the app to be in front first");
    }

    [Fact]
    public void ARecoveryDrivenByATurnDoesNotWarmTheCacheBehindIt()
    {
        // The turn that asked for the rebuild is about to forward the very prompt a
        // warm-up would forward. Warming would save it nothing and run a second
        // generation beside it.
        AgentAppHost host = Start();
        long before = host.PrefixCacheWarmupsStarted;
        Assert.False(host.RecoverEngineIfNeeded(warmAfterwards: false));
        Assert.Equal(before, host.PrefixCacheWarmupsStarted);
    }

    [Fact]
    public async Task AWarmUpThatMeetsADeadBackendHasTheEngineRebuiltNowRatherThanAtTheNextMessage()
    {
        // Measured on the phone: a warm-up that hit a GPU reset marked the engine, and
        // the user's first new chat then paid the rebuild AND the whole prompt -- a
        // 40 s first token. Nothing was running at the time, so the rebuild belongs
        // there, not at the message. With no model selected here, "rebuilding" is the
        // recovery finding nothing to reload onto and clearing the mark; what is
        // proved is that the path runs, from the warm-up, without being asked.
        AgentAppHost host = Start();
        var poisoned = new TaskCompletionSource<string>(TaskCreationOptions.RunContinuationsAsynchronously);
        host.EnginePoisoned += cause => poisoned.TrySetResult(cause);
        host.WarmUpFrames = (_, _) => DeadBackend();

        static async IAsyncEnumerable<object> DeadBackend()
        {
            await Task.Yield();
            yield return new
            {
                done = true,
                error = "ggml_metal_synchronize: error: command buffer 0 failed with status 5 | error: "
                        + "Discarded (victim of GPU error/recovery) (00000005:kIOGPUCommandBufferCallbackErrorInnocentVictim)",
            };
        }

        host.WarmThePrefixCache();
        string cause = await poisoned.Task.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Contains("victim of GPU error", cause, StringComparison.Ordinal);

        // The rebuild ran on its own, and the engine is no longer marked.
        for (int i = 0; i < 200 && (host.RebuildsAfterPoisonedWarmUps == 0 || host.EngineNeedsReload); i++)
            await Task.Delay(50);
        Assert.Equal(1, host.RebuildsAfterPoisonedWarmUps);
        Assert.False(host.EngineNeedsReload);
    }

    [Fact]
    public void TheTurnStopsTheWarmUpAgainAfterARebuildThatMayHaveStartedOne()
    {
        // The rebuild a turn asks for is serialized with the one the resume path, or
        // a poisoned warm-up, may already be running; those warm afterwards, so by the
        // time the turn holds the lock a fresh warm-up can be forwarding. Read from the
        // source: the second stop must come AFTER the recovery call in the retry loop.
        string host = File.ReadAllText(Path.Combine(
            FindRepoRoot(), "TensorAgent", "src", "TensorAgent.Core", "Hosting", "AgentAppHost.cs"));
        int loop = host.IndexOf("for (int attempt = 0; ; attempt++)", StringComparison.Ordinal);
        int rebuild = host.IndexOf("RecoverEngineIfNeeded(warmAfterwards: false)", loop, StringComparison.Ordinal);
        int stopAgain = host.IndexOf("StopWarmingThePrefixCacheAndWaitAsync", rebuild, StringComparison.Ordinal);
        int firstPull = host.IndexOf("ImageTurns.FramesFor(Chat, attemptBody", rebuild, StringComparison.Ordinal);
        Assert.True(rebuild > 0 && stopAgain > 0 && firstPull > 0, "the retry loop has changed shape");
        Assert.True(stopAgain < firstPull, "a warm-up started by another path's rebuild must be stopped before the turn pulls");
    }

    [Fact]
    public async Task AWarmUpWaitsForTheGateRatherThanSubmittingFromTheBackground()
    {
        // The warm-up is GPU work like any other. One started while the app is away is
        // refused exactly as a token would be -- and it is the likeliest such work,
        // because it starts itself a couple of seconds after every load.
        AgentAppHost host = Start();
        host.Compute.Close();
        host.WarmThePrefixCache();
        Assert.Equal(1, host.PrefixCacheWarmupsStarted);

        // Parked, not failed and not finished: it has not touched the engine.
        await Task.Delay(2500);
        Assert.False(host.PrefixCacheIsWarm);

        // And released by the gate opening, whereupon it runs and (with no model) fails
        // honestly rather than staying parked forever.
        host.Compute.Open();
        var clock = Stopwatch.StartNew();
        await host.StopWarmingThePrefixCacheAndWaitAsync();
        Assert.True(clock.Elapsed < TimeSpan.FromSeconds(35));
    }

    // =====================================================================================
    // carrying an answer on rather than writing it again
    // =====================================================================================

    [Fact]
    public void EveryOutputSyntaxTheEngineParsesIsOneAContinuationMustNotSplit()
    {
        // A drift guard, and the reason is a bug that was already in here: this started
        // as a single check for <tool_call>, which is Qwen's syntax. On a Harmony model
        // (gpt-oss channels), a GLM one, or Gemma's <function=, a half-written call
        // passed the check and was handed back to the model to "continue" - asking it to
        // finish a structure it cannot see the start of. If a new syntax appears in
        // OutputParser, this fails until someone has decided whether it can be split.
        string parser = File.ReadAllText(Path.Combine(
            FindRepoRoot(), "TensorSharp.Runtime", "OutputParser.cs"));

        string[] openersInTheEngine = System.Text.RegularExpressions.Regex
            .Matches(parser, "\"(<\\|?[a-z_]+[=|]?>?)\"")
            .Select(m => m.Groups[1].Value)
            .Where(o => !o.StartsWith("</", StringComparison.Ordinal))
            .Distinct(StringComparer.Ordinal)
            .ToArray();

        Assert.NotEmpty(openersInTheEngine);

        string[] known = AgentAppHost.StructureAModelCanBeHalfwayThrough
            .Select(p => p.Open)
            .ToArray();

        // Terminators are not structures a continuation can be halfway through; they end
        // the answer rather than opening anything.
        string[] terminators = ["<|end|>", "<|eom|>", "<|eot|>", "<|return|>", "<|call|>", "<|message|>"];

        string[] unconsidered = openersInTheEngine
            .Where(o => !known.Contains(o, StringComparer.Ordinal))
            .Where(o => !terminators.Contains(o, StringComparer.Ordinal))
            .ToArray();

        Assert.True(unconsidered.Length == 0,
            "OutputParser understands markers a resumed answer has never been checked against: "
            + string.Join(", ", unconsidered));
    }

    [Theory]
    [InlineData("Let me check.\n<|channel>analysis", false)]              // Harmony, mid-channel
    [InlineData("Calling it now: <function=search", false)]                // Gemma
    [InlineData("Thinking about it <think>the user wants", false)]         // reasoning left open
    [InlineData("<|tool_call>{\"name\":\"shell\"", false)]                 // GLM-style opener
    public void AFragmentHalfwayThroughANonQwenSyntaxIsAlsoNotContinued(string soFar, bool expected)
    {
        Assert.Equal(expected, AgentAppHost.CanBeCarriedOn(soFar.PadRight(60, '.')));
    }

    [Theory]
    [InlineData(null, false)]
    [InlineData("", false)]
    [InlineData("Sure!", false)]                                   // nothing to carry on from
    [InlineData("The transistor was invented at Bell Labs in 1947 by three physicists.", true)]
    [InlineData("Let me look that up.\n<tool_call>\n{\"name\": \"shell\", \"argum", false)] // mid tool call
    [InlineData("Done.\n<tool_call>\n{\"name\":\"shell\"}\n</tool_call>\nThat is the whole story so far.", true)]
    public void OnlyAFragmentWorthContinuingIsContinued(string? soFar, bool expected)
    {
        // Getting this wrong costs the user the paragraph they were reading, or hands
        // the model half a tool call and asks it to finish writing JSON it cannot see
        // the beginning of.
        Assert.Equal(expected, AgentAppHost.CanBeCarriedOn(soFar));
    }

    [Fact]
    public void TheAnswerSoFarIsHandedBackAsTheModelsOwnWords()
    {
        JsonElement body = JsonDocument.Parse(
            """
            {"sessionId":"s1","messages":[{"role":"user","content":"Tell me about the transistor."}],"maxTokens":512,"think":false}
            """).RootElement;

        JsonElement resumed = AgentAppHost.WithTheAnswerSoFar(body, "It was invented in 1947");

        // Everything the original asked for is still asked for: a retry that quietly
        // dropped maxTokens, think or the session would answer a different question.
        Assert.Equal("s1", resumed.GetProperty("sessionId").GetString());
        Assert.Equal(512, resumed.GetProperty("maxTokens").GetInt32());
        Assert.False(resumed.GetProperty("think").GetBoolean());

        JsonElement[] messages = resumed.GetProperty("messages").EnumerateArray().ToArray();
        Assert.Equal(3, messages.Length);
        Assert.Equal("user", messages[0].GetProperty("role").GetString());
        Assert.Equal("assistant", messages[1].GetProperty("role").GetString());
        Assert.Equal("It was invented in 1947", messages[1].GetProperty("content").GetString());
        Assert.Equal("user", messages[2].GetProperty("role").GetString());
        Assert.Contains("Continue it from exactly where it stops",
            messages[2].GetProperty("content").GetString()!, StringComparison.Ordinal);
    }

    /// <summary>
    /// A carried-on attempt knows nothing of the text it continues, so the whole-answer
    /// <c>replace</c> it sends (a DiffusionGemma preview, or the answer left once text it
    /// showed proved to be reasoning) covers only its own part. The page kept the earlier
    /// attempt's text, so the gate puts it in front; without it the replace wiped what the
    /// reader had already watched appear.
    /// </summary>
    [Fact]
    public void AReplaceFromACarriedOnAttemptKeepsTheTextItCarriesOn()
    {
        const string carried = "It was invented in 1947 at Bell Labs, ";
        var written = new System.Text.StringBuilder(carried);

        Assert.IsNotType<Dictionary<string, object?>>(AgentAppHost.AsShown(new { token = "Okay, plan" }, carried, written));
        AgentAppHost.AsShown(new { thinking = "Okay, plan" }, carried, written);
        object retraction = AgentAppHost.AsShown(new { replace = "" }, carried, written);
        Assert.Equal("{\"replace\":\"It was invented in 1947 at Bell Labs, \"}", JsonSerializer.Serialize(retraction));
        AgentAppHost.AsShown(new { token = "by Bardeen and Brattain." }, carried, written);
        Assert.Equal(carried + "by Bardeen and Brattain.", written.ToString());

        // Whatever else a replace frame carries rides along unchanged, in order.
        object preview = AgentAppHost.AsShown(
            new { replace = "canvas", diffusionStep = 3, diffusionTotal = 8, preview = true }, carried, written);
        Assert.Equal(
            "{\"replace\":\"It was invented in 1947 at Bell Labs, canvas\",\"diffusionStep\":3,\"diffusionTotal\":8,\"preview\":true}",
            JsonSerializer.Serialize(preview));

        // An attempt that carries nothing on sends its frames as they are.
        var fresh = new System.Text.StringBuilder("stale");
        var plain = new { replace = "Seven." };
        Assert.Same(plain, AgentAppHost.AsShown(plain, string.Empty, fresh));
        Assert.Equal("Seven.", fresh.ToString());
    }

    /// <summary>
    /// The whole gate, over a turn whose first attempt the GPU poisoned. An ordinary
    /// model's half answer is carried on: the next attempt is asked to continue it, and the
    /// <c>replace</c> that attempt sends when text it showed proves to be reasoning keeps
    /// the carried text in front, as the page shows it.
    /// </summary>
    [Fact]
    public async Task APoisonedAttempt_IsCarriedOn_AndItsReplaceKeepsTheCarriedText()
    {
        const string carried = "It was invented in 1947 at Bell Labs, by ";
        AgentAppHost host = StartWith(new FamilyModelService("qwen35", template: null));
        var bodies = new List<JsonElement>();
        host.TurnFrames = (body, _) => PoisonedThenRetracting(bodies, body, carried);

        List<JsonElement> frames = await GatedFrames(host);

        Assert.Equal(2, bodies.Count);
        Assert.True(bodies[1].GetProperty(AgentAppHost.ResumedTurnMarker).GetBoolean());
        Assert.Equal(carried, bodies[1].GetProperty("messages")[1].GetProperty("content").GetString());
        Assert.Contains(frames, f => f.TryGetProperty("restart", out _) && !f.TryGetProperty("replace", out _));
        JsonElement retraction = Assert.Single(frames, f => f.TryGetProperty("replace", out _));
        Assert.Equal(carried, retraction.GetProperty("replace").GetString());
        Assert.Equal(carried + "Bardeen and Brattain.", Shown(frames));
    }

    /// <summary>
    /// Nemotron-H Reasoning-128K shows a thinking-off reply before the <c>&lt;/think&gt;</c>
    /// that may say it was reasoning, so what the page has been shown is not known to be
    /// the answer. Carried on, it was handed to the model as its own answer and kept in
    /// front of every later <c>replace</c>, which put the reasoning back in the answer. The
    /// gate starts such a turn again from the top instead.
    /// </summary>
    [Fact]
    public async Task APoisonedAttempt_OfAModelWhoseShownAnswerMayBeReasoning_StartsAgainFromTheTop()
    {
        const string reasoning = "Okay, the user wants to know how many lines the file has, so I ";
        const string template =
            "{{ '<SPECIAL_10>System\n' }}{% for message in messages %}{{ '\n<SPECIAL_11>Assistant\n' }}{% endfor %}";
        AgentAppHost host = StartWith(new FamilyModelService("nemotron_h", template));
        var bodies = new List<JsonElement>();
        host.TurnFrames = (body, _) => PoisonedThenRetracting(bodies, body, reasoning);
        Assert.True(AgentAppHost.CanBeCarriedOn(reasoning));

        List<JsonElement> frames = await GatedFrames(host);

        Assert.Equal(2, bodies.Count);
        Assert.False(bodies[1].TryGetProperty(AgentAppHost.ResumedTurnMarker, out _));
        Assert.Equal(1, bodies[1].GetProperty("messages").GetArrayLength());
        JsonElement restart = Assert.Single(frames, f => f.TryGetProperty("restart", out _));
        Assert.Equal(string.Empty, restart.GetProperty("replace").GetString());
        Assert.Equal("Bardeen and Brattain.", Shown(frames));
    }

    private AgentAppHost StartWith(TensorSharp.Server.ModelService modelService)
    {
        _host = new AgentAppHost(new AgentPaths(
            Path.Combine(_root, "data"), Path.Combine(_root, "cache")), modelService: modelService);
        _host.Start();
        return _host;
    }

    /// <summary>The first attempt shows <paramref name="first"/> and dies of a poisoned
    /// engine; the next shows a fragment, takes it back as reasoning and answers.</summary>
    private static async IAsyncEnumerable<object> PoisonedThenRetracting(
        List<JsonElement> bodies, JsonElement body, string first)
    {
        bodies.Add(body.Clone());
        await Task.Yield();
        if (bodies.Count == 1)
        {
            yield return new { token = first };
            yield return new { done = true, error = "Metal command buffer 12 failed with status 5" };
            yield break;
        }
        yield return new { token = "Okay, plan" };
        yield return new { thinking = "Okay, plan" };
        yield return new { replace = string.Empty };
        yield return new { token = "Bardeen and Brattain." };
        yield return new { done = true };
    }

    private static async Task<List<JsonElement>> GatedFrames(AgentAppHost host)
    {
        JsonElement body = JsonSerializer.SerializeToElement(new
        {
            sessionId = "s1",
            messages = new[] { new { role = "user", content = "Who invented the transistor?" } },
            think = false,
        });
        var frames = new List<JsonElement>();
        await foreach (object frame in host.GatedChatFrames(body, CancellationToken.None))
            frames.Add(JsonSerializer.SerializeToElement(frame));
        return frames;
    }

    /// <summary>The answer a page is left showing: tokens append, a replace sets it.</summary>
    private static string Shown(IEnumerable<JsonElement> frames)
    {
        string shown = string.Empty;
        foreach (JsonElement frame in frames)
        {
            if (frame.TryGetProperty("replace", out JsonElement whole)) shown = whole.GetString()!;
            if (frame.TryGetProperty("token", out JsonElement token)) shown += token.GetString();
        }
        return shown;
    }

    /// <summary>A model service that reports a family and template and loads nothing.</summary>
    private sealed class FamilyModelService(string architecture, string? template) : TensorSharp.Server.ModelService
    {
        public override string Architecture => architecture;
        public override string ChatTemplate => template!;
    }

    [Fact]
    public void AResumedTurnIsMarkedSoItsScaffoldingStaysOutOfTheTranscript()
    {
        // The two added messages describe how the answer is being produced, not what
        // was said. A user who scrolled back and found an instruction they never typed
        // would be right to call that a bug.
        JsonElement body = JsonDocument.Parse("""{"messages":[]}""").RootElement;
        JsonElement resumed = AgentAppHost.WithTheAnswerSoFar(body, new string('x', 80));

        Assert.True(resumed.GetProperty(AgentAppHost.ResumedTurnMarker).GetBoolean());
        Assert.False(body.TryGetProperty(AgentAppHost.ResumedTurnMarker, out _));

        // And resuming a resumed turn (the second retry) does not stack the marker or
        // the scaffolding twice.
        JsonElement again = AgentAppHost.WithTheAnswerSoFar(resumed, new string('y', 80));
        Assert.Equal(1, again.EnumerateObject().Count(p => p.Name == AgentAppHost.ResumedTurnMarker));
    }

    [Fact]
    public void AResumedTurnIsNotRecordedIntoTheConversation()
    {
        // Through the real hook: the chat service calls OnChatRequest for every request
        // it serves, and the recorder writes the user's message on that call. A resumed
        // request carries the model's own half-answer and an instruction the user never
        // typed; neither may reach the transcript.
        AgentAppHost host = Start();
        object session = host.Chat.CreateSession();
        string sessionId = session.GetType().GetProperty("sessionId")?.GetValue(session) as string
            ?? throw new InvalidOperationException("the session has no id");
        host.Recorder.Bind(sessionId, null);

        JsonElement original = JsonDocument.Parse(
            """{"messages":[{"role":"user","content":"what is a transistor"}]}""").RootElement;
        host.Chat.OnChatRequest!(sessionId, original);
        host.Chat.OnChatRequest!(sessionId, AgentAppHost.WithTheAnswerSoFar(original, new string('x', 80)));

        string conversationId = host.Recorder.ConversationFor(sessionId)
            ?? throw new InvalidOperationException("the session was not bound");
        TensorAgent.Core.Sessions.Conversation saved = host.Conversations.Load(conversationId)!;
        Assert.Single(saved.Messages);
        Assert.Equal("what is a transistor", saved.Messages[0].Content);
    }
}
