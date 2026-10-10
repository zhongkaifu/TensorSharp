// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorAgent.Core.Hosting;
using TensorAgent.Core.Sessions;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace TensorAgent.Tests;

/// <summary>
/// A generation that outlives the request that asked for it.
///
/// <para>
/// The bug these are written against is the one every user of the phone app hit: open
/// the model list, or glance at another app, while the model is answering, and the
/// answer is gone when you come back. iOS suspends a WKWebView's content process the
/// moment its view leaves the window, so the page stops reading — and the server, which
/// treated the reader leaving as "nobody wants this any more", stopped generating. The
/// fix is an ownership change, and what it has to guarantee is exactly three things: a
/// reader that leaves stops nothing, a reader that arrives afterwards is told
/// everything, and the answer is written down whether or not anyone was listening.
/// </para>
/// </summary>
public sealed class ChatTurnTests
{
    // ---- the manager, on its own -------------------------------------------------

    [Fact]
    public async Task AReaderThatWalksAwayStopsNothingAndTheNextOneIsToldEverything()
    {
        using var turns = new ChatTurnManager();
        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();

        string id = turns.Start("chat-1", ct => Slow(released, reachedTheEnd, ct));

        // The first reader takes one frame and leaves, which is what a page being
        // suspended looks like from here.
        released.Release();
        await using (IAsyncEnumerator<object> first = turns.WatchAsync(id, 0, CancellationToken.None).GetAsyncEnumerator())
        {
            Assert.True(await first.MoveNextAsync());
            Assert.Equal("one ", TokenOf(first.Current));
        }

        // Nothing was cancelled by that.
        released.Release();
        released.Release();
        await reachedTheEnd.Task.WaitAsync(TimeSpan.FromSeconds(10));
        await WaitFor(() => !turns.StatusOfKey("chat-1")!.IsRunning);

        // And a reader arriving now gets the whole answer, from the first frame.
        var seen = new List<string>();
        await foreach (object frame in turns.WatchAsync(id, 0, CancellationToken.None))
            if (TokenOf(frame) is { } token)
                seen.Add(token);
        Assert.Equal(new[] { "one ", "two ", "three" }, seen);
        Assert.Equal(ChatTurnState.Completed, turns.StatusOfKey("chat-1")!.State);
    }

    [Fact]
    public async Task CancellingAReaderIsNotCancellingTheTurn()
    {
        using var turns = new ChatTurnManager();
        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();
        string id = turns.Start("chat-1", ct => Slow(released, reachedTheEnd, ct));

        using var reader = new CancellationTokenSource();
        Task reading = Task.Run(async () =>
        {
            await foreach (object _ in turns.WatchAsync(id, 0, reader.Token)) { }
        });
        released.Release();
        reader.Cancel();
        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => reading);

        released.Release();
        released.Release();
        await reachedTheEnd.Task.WaitAsync(TimeSpan.FromSeconds(10));
        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);
        Assert.Equal(ChatTurnState.Completed, turns.StatusOfId(id)!.State);
    }

    [Fact]
    public async Task StoppingIsSomethingSomebodyHasToAskFor()
    {
        using var turns = new ChatTurnManager();
        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();
        string id = turns.Start("chat-1", ct => Slow(released, reachedTheEnd, ct));

        released.Release();
        // Wait until it is genuinely producing before stopping it, so this is a stop and
        // not a race with the start.
        await WaitFor(() => turns.StatusOfId(id)!.FrameCount > 0);
        Assert.True(turns.StopId(id));

        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);
        Assert.Equal(ChatTurnState.Cancelled, turns.StatusOfId(id)!.State);
        Assert.False(reachedTheEnd.Task.IsCompleted);
    }

    [Fact]
    public async Task ASecondTurnOnTheSameConversationEndsTheFirst()
    {
        using var turns = new ChatTurnManager();
        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();
        string first = turns.Start("chat-1", ct => Slow(released, reachedTheEnd, ct));
        released.Release();
        await WaitFor(() => turns.StatusOfId(first)!.FrameCount > 0);

        string second = turns.Start("chat-1", _ => One());
        await WaitFor(() => !turns.StatusOfId(first)!.IsRunning);

        Assert.Equal(ChatTurnState.Cancelled, turns.StatusOfId(first)!.State);
        Assert.Equal(second, turns.StatusOfKey("chat-1")!.Id);
    }

    /// <summary>
    /// A refusal that arrives before any frame keeps its identity, so the route can
    /// still answer with a status code rather than an empty event stream. That is the
    /// behaviour <c>AChatRequestWithNoModelLoadedIsRefusedWithAStatusNotAnEmptyStream</c>
    /// pins from the outside; this is why it survived the turn manager being put in
    /// between.
    /// </summary>
    [Fact]
    public async Task ARefusalBeforeTheFirstFrameIsRethrownToTheReader()
    {
        using var turns = new ChatTurnManager();
        string id = turns.Start("chat-1", _ => Refuse());
        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

        InvalidOperationException thrown = await Assert.ThrowsAsync<InvalidOperationException>(async () =>
        {
            await foreach (object _ in turns.WatchAsync(id, 0, CancellationToken.None)) { }
        });
        Assert.Equal("no model is loaded", thrown.Message);
    }

    /// <summary>
    /// A failure AFTER frames have gone out cannot be a status code any more — the
    /// headers are spent — so it becomes the error frame the page renders.
    /// </summary>
    [Fact]
    public async Task AFailurePartWayThroughArrivesAsAFrameInstead()
    {
        using var turns = new ChatTurnManager();
        string id = turns.Start("chat-1", _ => OneThenThrow());
        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

        var frames = new List<string>();
        await foreach (object frame in turns.WatchAsync(id, 0, CancellationToken.None))
            frames.Add(JsonSerializer.Serialize(frame, SseFraming.JsonOptions));

        Assert.Equal(2, frames.Count);
        Assert.Contains("\"error\":\"the engine gave up\"", frames[1], StringComparison.Ordinal);
        Assert.Equal(ChatTurnState.Failed, turns.StatusOfId(id)!.State);
    }

    /// <summary>
    /// A turn that outruns its buffer keeps every reader's place.
    ///
    /// <para>
    /// Unreachable in the app — the reply cap is a few thousand tokens against a fifty
    /// thousand frame buffer — and worth a test anyway, because the compaction is the
    /// one place where a frame number could quietly come to mean something different.
    /// A reader holding a position across it must not skip frames or see them twice,
    /// and one arriving afterwards must still be able to rebuild the whole answer.
    /// </para>
    /// </summary>
    [Fact]
    public async Task OutrunningTheBufferCompactsItWithoutLosingAReadersPlace()
    {
        using var turns = new ChatTurnManager { MaxBufferedFrames = 8 };
        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();
        string id = turns.Start("chat-1", ct => Counting(released, reachedTheEnd, 40, ct));

        // A reader that has seen the first few and then stops.
        released.Release(3);
        int seen = 0;
        await foreach (object frame in turns.WatchAsync(id, 0, CancellationToken.None))
        {
            if (TokenOf(frame) is not null && ++seen == 3)
                break;
        }

        released.Release(37);
        await reachedTheEnd.Task.WaitAsync(TimeSpan.FromSeconds(10));
        await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

        // Compaction happened: the count is still absolute and past the cap.
        Assert.True(turns.StatusOfId(id)!.FrameCount > 8);

        // A reader starting over rebuilds exactly the answer the model produced, with
        // the collapsed prefix arriving as one `replace`.
        var answer = new System.Text.StringBuilder();
        await foreach (object frame in turns.WatchAsync(id, 0, CancellationToken.None))
        {
            using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(frame, SseFraming.JsonOptions));
            if (document.RootElement.TryGetProperty("replace", out JsonElement whole))
                answer.Clear().Append(whole.GetString());
            else if (document.RootElement.TryGetProperty("token", out JsonElement token))
                answer.Append(token.GetString());
        }
        Assert.Equal(string.Concat(Enumerable.Range(0, 40).Select(i => i + " ")), answer.ToString());

        // And one resuming from where it left off is not handed the frames again.
        var after = new List<string>();
        await foreach (object frame in turns.WatchAsync(id, 3, CancellationToken.None))
            if (TokenOf(frame) is { } token)
                after.Add(token);
        Assert.DoesNotContain("0 ", after);
    }

    /// <summary>
    /// A turn that was replaced does not write its fragment over the answer that
    /// replaced it.
    ///
    /// <para>
    /// The transcript is written by the turn now, which is what makes an answer survive
    /// a page that walked away — and it opens a hole the request-scoped version could
    /// not have: two pumps for one conversation. A cancelled turn can be a long way from
    /// finishing (a shell command is a synchronous wait of up to the tool timeout), so it
    /// commonly unwinds AFTER the answer that superseded it has been saved. The recorder
    /// removes the last assistant message before appending, so writing then would delete
    /// the answer on the user's screen and file its own fragment under the wrong
    /// question.
    /// </para>
    /// </summary>
    [Fact]
    public async Task ASupersededTurnDoesNotWriteOverTheAnswerThatReplacedIt()
    {
        string root = Path.Combine(Path.GetTempPath(), "turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            var conversations = new ConversationStore(Path.Combine(root, "conversations"));
            var recorder = new ConversationRecorder(conversations);
            using var turns = new ChatTurnManager(recorder);
            Conversation conversation = conversations.Create();
            recorder.Bind("session-1", conversation.Id);

            // The first turn gets as far as a few words and is then left hanging.
            var stuck = new SemaphoreSlim(0);
            var unwound = new TaskCompletionSource();
            turns.Start(conversation.Id, _ => Stuck(stuck, unwound, "half an ans", "session-1"));
            await WaitFor(() => turns.FramesFor(conversation.Id) > 0);

            // The user gives up and asks again. The old turn is cancelled but is still
            // inside its tool call.
            string second = turns.Start(conversation.Id, _ => Finished("the real answer", "session-1"));
            await WaitFor(() => !turns.StatusOfId(second)!.IsRunning);
            Assert.Equal("the real answer", Saved(conversations, conversation.Id));

            // Only now does the first one unwind.
            stuck.Release();
            await unwound.Task.WaitAsync(TimeSpan.FromSeconds(10));
            await Task.Delay(200);

            Assert.Equal("the real answer", Saved(conversations, conversation.Id));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }

        static string Saved(ConversationStore store, string id) =>
            store.Load(id)!.Messages.LastOrDefault(m => m.Role == "assistant")?.Content ?? "(nothing)";
    }

    /// <summary>
    /// The background owner, not the page, closes a turn. It must therefore understand
    /// the completion guard's dedicated artifact frame itself: the earlier file reported
    /// by a successful writer call is still provisional and must not survive a reload.
    /// </summary>
    [Fact]
    public async Task BackgroundCompletionPersistsOnlyTheHostVerifiedArtifactAcrossReload()
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-artifacts-" + Guid.NewGuid().ToString("N")[..8]);
        string conversationRoot = Path.Combine(root, "conversations");
        Directory.CreateDirectory(root);
        try
        {
            var store = new ConversationStore(conversationRoot);
            Conversation conversation = store.Create();
            conversation.Messages.Add(new StoredMessage { Role = "user", Content = "make the report" });
            store.Save(conversation);

            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-artifacts", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            string turnId = turns.Start(
                conversation.Id,
                _ => ArtifactFrames("session-artifacts"));
            await WaitFor(() => !turns.StatusOfId(turnId)!.IsRunning);

            // Load through a fresh store instance: this is the same disk/history path a
            // relaunched WebView takes, rather than an assertion on in-memory objects.
            Conversation reloaded = Assert.IsType<Conversation>(
                new ConversationStore(conversationRoot).Load(conversation.Id));
            StoredMessage assistant = Assert.Single(reloaded.Messages, message => message.Role == "assistant");
            Assert.Equal("The report is ready.", assistant.Content);
            StoredArtifact artifact = Assert.Single(assistant.Artifacts!);
            Assert.Equal("apple-m5-m6-verified.pptx", artifact.Name);
            Assert.Equal(15_569, artifact.Bytes);
            Assert.Equal("/api/code/artifacts/verified/apple-m5-m6.pptx", artifact.Url);
            Assert.DoesNotContain(assistant.Artifacts!, item =>
                item.Url.Contains("provisional", StringComparison.Ordinal));
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    public async Task BackgroundCompletionPersistsTerminalStatsAcrossReload(bool aborted, bool truncated)
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-stats-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new ConversationStore(root);
            Conversation conversation = store.Create();
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-stats", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            // No reader attaches: counters must survive the same background completion
            // and fresh-store load as the answer, without counting displayed pieces.
            string id = turns.Start(conversation.Id, _ => StatsFrames(new
            {
                done = true, sessionId = "session-stats", tokenCount = 1200,
                elapsed = 42.25, tokPerSec = 100.5,
                promptTokens = 2000, kvReusedTokens = 1500, kvReusePercent = 75.0,
                aborted, truncated,
            }));
            await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

            Conversation reloaded = Assert.IsType<Conversation>(new ConversationStore(root).Load(conversation.Id));
            StoredMessage assistant = Assert.Single(reloaded.Messages);
            Assert.Equal("The answer.", assistant.Content);
            StoredTurnStats stats = Assert.IsType<StoredTurnStats>(assistant.Stats);
            Assert.Equal(1200, stats.TokenCount);
            Assert.Equal(42.25, stats.Elapsed);
            Assert.Equal(100.5, stats.TokensPerSecond);
            Assert.Equal(2000, stats.PromptTokens);
            Assert.Equal(1500, stats.KvReusedTokens);
            Assert.Equal(75.0, stats.KvReusePercent);
            Assert.Equal(aborted, stats.Aborted);
            Assert.Equal(truncated, stats.Truncated);
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(true, 0)]
    [InlineData(false, 1200)]
    [InlineData(true, 1200)]
    public async Task MediaCompletionStoresOnlyActualLlmCounters(bool video, int tokenCount)
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-media-stats-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new ConversationStore(root);
            Conversation conversation = store.Create();
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-media", conversation.Id);
            using var turns = new ChatTurnManager(recorder);
            string id = turns.Start(conversation.Id, _ => Frames());
            await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

            StoredMessage answer = Assert.Single(new ConversationStore(root).Load(conversation.Id)!.Messages);
            Assert.Equal(video ? "/uploads/result.mp4" : null, answer.VideoUrl);
            Assert.Equal(video ? null : "/uploads/result.png", answer.ImageUrl);
            if (tokenCount == 0)
                Assert.Null(answer.Stats);
            else
                Assert.Equal(tokenCount, Assert.IsType<StoredTurnStats>(answer.Stats).TokenCount);
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }

        async IAsyncEnumerable<object> Frames()
        {
            await Task.Yield();
            if (tokenCount > 0)
                yield return new { token = "The result." };
            // Media services emit the URL before their numeric terminal frame, and
            // successful image/video generations do not emit an assistant text token.
            if (video)
                yield return new { videoUrl = "/uploads/result.mp4", width = 640, height = 480, frames = 24, fps = 24 };
            else
                yield return new { imageUrl = "/uploads/result.png", width = 640, height = 480 };
            yield return new
            {
                done = true, sessionId = "session-media", tokenCount,
                elapsed = 42.25, tokPerSec = tokenCount > 0 ? 100.5 : 0.0, truncated = false,
            };
        }
    }

    [Theory]
    [InlineData("""{"done":true,"error":"failed"}""")]
    [InlineData("""{"done":true,"imageUrl":"/uploads/image.png"}""")]
    [InlineData("""{"done":true,"tokenCount":3,"elapsed":1}""")]
    [InlineData("""{"done":true,"tokenCount":"3","elapsed":1,"tokPerSec":3}""")]
    [InlineData("""{"done":true,"tokenCount":-3,"elapsed":1,"tokPerSec":3}""")]
    [InlineData("""{"done":true,"tokenCount":3,"elapsed":1e999,"tokPerSec":3}""")]
    [InlineData("""{"tokenCount":3,"elapsed":1,"tokPerSec":3}""")]
    public async Task FramesWithoutValidTerminalCountersDoNotInventStoredStats(string frameJson)
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-no-stats-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new ConversationStore(root);
            Conversation conversation = store.Create();
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-stats", conversation.Id);
            using var turns = new ChatTurnManager(recorder);
            JsonElement frame = JsonSerializer.Deserialize<JsonElement>(frameJson);
            string id = turns.Start(conversation.Id, _ => StatsFrames(frame));
            await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);
            Assert.Null(Assert.Single(new ConversationStore(root).Load(conversation.Id)!.Messages).Stats);
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }

    /// <summary>
    /// Text the page showed and took back as reasoning (Nemotron-H Reasoning-128K closing a
    /// block its prompt had closed) arrives as its reasoning and then the whole answer that
    /// is left. The saved turn says what the page shows: the earlier round's text and the
    /// answer, with the reasoning in its own field and nowhere in the answer. The recorder
    /// already applied a <c>replace</c> this way for DiffusionGemma previews; this pins that
    /// it does for the retraction's frames, in the order the Web UI sends them.
    /// </summary>
    [Fact]
    public async Task ARetractedReasoningIsSavedAsReasoningAndNotAsTheAnswer()
    {
        string root = Path.Combine(Path.GetTempPath(), "turn-retract-" + Guid.NewGuid().ToString("N"));
        try
        {
            var store = new ConversationStore(root);
            Conversation conversation = store.Create();
            var recorder = new ConversationRecorder(store);
            recorder.Bind("session-retract", conversation.Id);
            using var turns = new ChatTurnManager(recorder);

            string id = turns.Start(conversation.Id, _ => RetractionFrames());
            await WaitFor(() => !turns.StatusOfId(id)!.IsRunning);

            Conversation reloaded = Assert.IsType<Conversation>(new ConversationStore(root).Load(conversation.Id));
            StoredMessage assistant = Assert.Single(reloaded.Messages);
            Assert.Equal("I will read it.\n\nThe file has 7 lines.", assistant.Content);
            Assert.Equal("Okay, the user wants the count.", assistant.Thinking);
        }
        finally { try { Directory.Delete(root, true); } catch (IOException) { } }
    }

    private static async IAsyncEnumerable<object> RetractionFrames()
    {
        await Task.Yield();
        yield return new { token = "I will read it.\n\n", sessionId = "session-retract" };
        yield return new { token = "Okay, the user " };
        yield return new { token = "wants the count." };
        yield return new { thinking = "Okay, the user wants the count." };
        yield return new { replace = "I will read it.\n\n" };
        yield return new { token = "The file has 7 lines." };
        yield return new { done = true, sessionId = "session-retract", tokenCount = 12, elapsed = 1.0, tokPerSec = 12.0 };
    }

    private static async IAsyncEnumerable<object> StatsFrames(object terminal)
    {
        await Task.Yield();
        yield return new { token = "The answer.", sessionId = "session-stats" };
        yield return terminal;
    }

    [Fact]
    public async Task BusyFollowsTheTurnAndNotTheReader()
    {
        using var turns = new ChatTurnManager();
        var seen = new List<bool>();
        turns.BusyChanged += busy => { lock (seen) seen.Add(busy); };

        var released = new SemaphoreSlim(0);
        var reachedTheEnd = new TaskCompletionSource();
        string id = turns.Start("chat-1", ct => Slow(released, reachedTheEnd, ct));
        Assert.True(turns.IsBusy);

        released.Release(3);
        await reachedTheEnd.Task.WaitAsync(TimeSpan.FromSeconds(10));
        await WaitFor(() => !turns.IsBusy);
        lock (seen)
            Assert.Equal(new[] { true, false }, seen);
    }

    // ---- through the routes the page actually calls -------------------------------

    /// <summary>
    /// The whole reported journey, over HTTP: send a message, lose the reader half way
    /// through, come back, and find the answer complete — and saved.
    /// </summary>
    [Fact]
    public async Task ATurnSurvivesThePageAndIsThereToAttachToWhenItComesBack()
    {
        string root = Path.Combine(Path.GetTempPath(), "turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            using var harness = new RouteHarness(root);
            var released = new SemaphoreSlim(0);
            var reachedTheEnd = new TaskCompletionSource();
            harness.Frames = ct => Slow(released, reachedTheEnd, ct, sessionId: harness.SessionId);

            // Released before sending, because the stream's headers do not go out until
            // its first frame does -- that is what lets a refusal still be a 400 rather
            // than an empty 200.
            released.Release();
            using HttpResponseMessage started = await harness.SendChatAsync("count");

            Assert.Equal("text/event-stream", started.Content.Headers.ContentType!.MediaType);
            string turnId = Assert.Single(started.Headers.GetValues(WebUiRoutes.TurnHeader));

            // Read the first token, then drop the connection: this is the page being
            // suspended, and it used to be the end of the answer.
            await ReadOneFrameAsync(started);
            started.Dispose();

            released.Release();
            released.Release();
            await reachedTheEnd.Task.WaitAsync(TimeSpan.FromSeconds(10));

            // The page comes back and asks what is going on with its conversation.
            JsonElement status = await harness.JsonAsync(
                "/api/agent/turns?conversation=" + harness.ConversationId);
            Assert.Equal(turnId, status.GetProperty("turn").GetProperty("id").GetString());

            // And replays it in full, from the first frame.
            string replayed = await harness.Client.GetStringAsync("/api/agent/turns/" + turnId);
            Assert.Contains("data: {\"token\":\"one \"}", replayed, StringComparison.Ordinal);
            Assert.Contains("data: {\"token\":\"three\"}", replayed, StringComparison.Ordinal);

            // The answer was written down with nobody reading, which is what a relaunch
            // will show.
            Conversation saved = new ConversationStore(harness.ConversationRoot).Load(harness.ConversationId)!;
            StoredMessage answer = Assert.Single(saved.Messages, m => m.Role == "assistant");
            Assert.Equal("one two three", answer.Content);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    /// <summary>
    /// A page that reloaded is told about the turn by the very request that binds its
    /// conversation, so it can attach without a second round trip and without guessing.
    /// </summary>
    [Fact]
    public async Task BindingAConversationReportsTheGenerationStillRunningForIt()
    {
        string root = Path.Combine(Path.GetTempPath(), "turns-" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(root);
        try
        {
            using var harness = new RouteHarness(root);
            var released = new SemaphoreSlim(0);
            var reachedTheEnd = new TaskCompletionSource();
            harness.Frames = ct => Slow(released, reachedTheEnd, ct, sessionId: harness.SessionId);

            released.Release();
            using HttpResponseMessage started = await harness.SendChatAsync("count");
            await ReadOneFrameAsync(started);

            // A brand new session on the same conversation: exactly what the page does
            // after the WebView was reloaded.
            JsonElement bound = await harness.JsonAsync(
                "/api/sessions?conversation=" + harness.ConversationId, post: true);
            JsonElement turn = bound.GetProperty("activeTurn");
            Assert.True(turn.GetProperty("running").GetBoolean());
            Assert.Equal(
                Assert.Single(started.Headers.GetValues(WebUiRoutes.TurnHeader)),
                turn.GetProperty("id").GetString());

            // Stopping is now an explicit act, and the route for it is the one the page
            // has the id for.
            JsonElement stopped = await harness.JsonAsync(
                "/api/agent/turns/" + turn.GetProperty("id").GetString() + "/stop", post: true);
            Assert.True(stopped.GetProperty("stopped").GetBoolean());
            await WaitFor(() => !harness.Turns.StatusOfKey(harness.ConversationId)!.IsRunning);
        }
        finally
        {
            try { Directory.Delete(root, true); } catch (IOException) { }
        }
    }

    // ---- fixtures -----------------------------------------------------------------

    /// <summary>Three tokens, each released by the test, so "half way through" is a place.</summary>
    private static async IAsyncEnumerable<object> Slow(
        SemaphoreSlim released, TaskCompletionSource reachedTheEnd,
        [EnumeratorCancellation] CancellationToken ct, string? sessionId = null)
    {
        foreach (string token in new[] { "one ", "two ", "three" })
        {
            await released.WaitAsync(ct).ConfigureAwait(false);
            yield return new { token };
        }
        if (sessionId is not null)
            yield return new { done = true, sessionId };
        reachedTheEnd.SetResult();
    }

    /// <summary>As many tokens as asked for, one per release.</summary>
    private static async IAsyncEnumerable<object> Counting(
        SemaphoreSlim released, TaskCompletionSource reachedTheEnd, int count,
        [EnumeratorCancellation] CancellationToken ct)
    {
        for (int i = 0; i < count; i++)
        {
            await released.WaitAsync(ct).ConfigureAwait(false);
            yield return new { token = i + " " };
        }
        reachedTheEnd.SetResult();
    }

    /// <summary>Says its piece, then blocks until released — ignoring cancellation, as a
    /// synchronous tool call does.</summary>
    private static async IAsyncEnumerable<object> Stuck(
        SemaphoreSlim release, TaskCompletionSource unwound, string text, string sessionId)
    {
        yield return new { token = text };
        await release.WaitAsync().ConfigureAwait(false);
        yield return new { done = true, sessionId };
        unwound.SetResult();
    }

    private static async IAsyncEnumerable<object> Finished(string text, string sessionId)
    {
        await Task.Yield();
        yield return new { token = text };
        yield return new { done = true, sessionId };
    }

    private static async IAsyncEnumerable<object> ArtifactFrames(string sessionId)
    {
        await Task.Yield();
        yield return new
        {
            skill_step = "skills_run",
            files = new[]
            {
                new
                {
                    name = "apple-m5-m6-provisional.pptx",
                    bytes = 12_000L,
                    url = "/api/code/artifacts/provisional/apple-m5-m6.pptx",
                },
            },
        };
        yield return new
        {
            artifact_verified = false,
            files = new[]
            {
                new
                {
                    name = "also-provisional.pptx",
                    bytes = 13_000L,
                    url = "/api/code/artifacts/provisional/also.pptx",
                },
            },
        };
        yield return new
        {
            artifact_verified = true,
            files = new[]
            {
                new
                {
                    name = "apple-m5-m6-verified.pptx",
                    bytes = 15_569L,
                    url = "/api/code/artifacts/verified/apple-m5-m6.pptx",
                },
            },
        };
        // Duplicate verified frames can be seen by a reconnecting reader, but the pump
        // consumes the source once; retaining this duplicate pins URL de-duplication too.
        yield return new
        {
            artifact_verified = true,
            files = new[]
            {
                new
                {
                    name = "duplicate-name.pptx",
                    bytes = 99_999L,
                    url = "/api/code/artifacts/verified/apple-m5-m6.pptx",
                },
            },
        };
        yield return new { token = "The report is ready." };
        yield return new { done = true, sessionId };
    }

    private static async IAsyncEnumerable<object> One()
    {
        await Task.Yield();
        yield return new { token = "hello" };
    }

    private static async IAsyncEnumerable<object> Refuse()
    {
        await Task.Yield();
        throw new InvalidOperationException("no model is loaded");
#pragma warning disable CS0162 // the compiler needs a yield to make this an iterator
        yield break;
#pragma warning restore CS0162
    }

    private static async IAsyncEnumerable<object> OneThenThrow()
    {
        await Task.Yield();
        yield return new { token = "hel" };
        throw new InvalidOperationException("the engine gave up");
    }

    private static string? TokenOf(object frame)
    {
        using JsonDocument document = JsonDocument.Parse(JsonSerializer.Serialize(frame, SseFraming.JsonOptions));
        return document.RootElement.TryGetProperty("token", out JsonElement token) ? token.GetString() : null;
    }

    private static async Task WaitFor(Func<bool> condition)
    {
        for (int i = 0; i < 400 && !condition(); i++)
            await Task.Delay(25);
        Assert.True(condition(), "the condition never came true");
    }

    /// <summary>Pull exactly one frame off a stream and leave the rest unread.</summary>
    private static async Task ReadOneFrameAsync(HttpResponseMessage response)
    {
        await using Stream body = await response.Content.ReadAsStreamAsync();
        var buffer = new byte[64];
        for (int i = 0; i < 40; i++)
        {
            int n = await body.ReadAsync(buffer);
            if (n > 0)
                return;
        }
        Assert.Fail("the stream produced nothing");
    }

    /// <summary>The real routes over real HTTP, with the frames supplied by the test.</summary>
    private sealed class RouteHarness : IDisposable
    {
        private readonly LoopbackServer _server;
        private readonly ConversationStore _conversations;

        public RouteHarness(string root)
        {
            string uploads = Path.Combine(root, "uploads");
            Directory.CreateDirectory(uploads);
            _conversations = new ConversationStore(Path.Combine(root, "conversations"));
            var recorder = new ConversationRecorder(_conversations);
            Turns = new ChatTurnManager(recorder);

            _server = new LoopbackServer(NullLogger.Instance) { RequireToken = false };
            _server.MapWebUi(
                ChatService(root, uploads), uploads, skills: null, recorder: recorder,
                chatFrames: (_, ct) => Frames!(ct), turns: Turns);
            _server.Start();
            Client = new HttpClient { BaseAddress = new Uri(_server.BaseUrl) };

            JsonElement created = JsonAsync("/api/sessions?conversation=new", post: true).GetAwaiter().GetResult();
            SessionId = created.GetProperty("sessionId").GetString()!;
            ConversationId = created.GetProperty("conversationId").GetString()!;
        }

        public HttpClient Client { get; }
        public ChatTurnManager Turns { get; }
        public string SessionId { get; }
        public string ConversationId { get; }
        public string ConversationRoot => _conversations.Root;
        public Func<CancellationToken, IAsyncEnumerable<object>>? Frames { get; set; }

        /// <summary>
        /// Send a message and get the response back the moment its headers arrive.
        ///
        /// <para>
        /// Headers-only matters: the default buffers the whole response before returning,
        /// and the whole response is an event stream that ends when the model does, so a
        /// test about dropping a reader half way through could never reach half way
        /// through.
        /// </para>
        /// </summary>
        public Task<HttpResponseMessage> SendChatAsync(string prompt)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, "/api/chat")
            {
                Content = new StringContent(
                    $$"""{"sessionId":"{{SessionId}}","messages":[{"role":"user","content":"{{prompt}}"}]}""",
                    Encoding.UTF8, "application/json"),
            };
            return Client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead);
        }

        public async Task<JsonElement> JsonAsync(string path, bool post = false)
        {
            using HttpResponseMessage response = post
                ? await Client.PostAsync(path, null)
                : await Client.GetAsync(path);
            using JsonDocument document = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
            return document.RootElement.Clone();
        }

        public void Dispose()
        {
            Turns.StopAll();
            Client.Dispose();
            _server.Dispose();
            Turns.Dispose();
        }

        /// <summary>
        /// A real chat service with no model behind it. Nothing here reaches the engine —
        /// the frames are the test's — but the routes still need the service they are
        /// bound to, and building the real one is what keeps this a test of the wiring.
        /// </summary>
        private static WebUiChatService ChatService(string root, string uploads)
        {
            var options = new ServerHostingOptions(
                startupModelPath: Path.Combine(root, "models", "none.gguf"),
                startupMmProjPath: null,
                defaultBackend: "ggml_cpu",
                supportedBackends: new[] { new BackendOption("ggml_cpu", "GGML CPU") },
                defaultMaxTokens: 256,
                maxTokensPinned: false,
                defaultVideoFrames: 0, defaultVideoFps: 0, defaultVideoWidth: 0,
                defaultVideoHeight: 0, defaultVideoSteps: 0, defaultVideoMode: null,
                uploadDirectory: uploads,
                logDirectory: Path.Combine(root, "logs"),
                fileLoggingEnabled: false,
                samplingDefaults: null);

            return new WebUiChatService(
                new ModelService(), new SessionManager(), options,
                new UploadStoragePolicy(uploads), new SkillRegistry(new SkillRegistryOptions()),
                codeRunner: null, workspaces: null, codeArtifacts: null,
                NullLoggerFactory.Instance);
        }
    }
}
