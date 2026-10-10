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

namespace TensorAgent.Core.Sessions;

/// <summary>What a turn is doing, or what it did.</summary>
public enum ChatTurnState
{
    /// <summary>The model is still working.</summary>
    Running,
    /// <summary>The frames ran out on their own: the answer is complete.</summary>
    Completed,
    /// <summary>Somebody asked for it to stop.</summary>
    Cancelled,
    /// <summary>It ended with an error, which is the last frame or the stored rejection.</summary>
    Failed,
}

/// <summary>One turn, as a page needs to find it again.</summary>
/// <param name="Id">Opaque id; what a reader re-attaches with.</param>
/// <param name="Key">The conversation (or session) this turn belongs to.</param>
/// <param name="State">Where it got to.</param>
/// <param name="FrameCount">How many frames exist so far, which is also where a fresh reader would start.</param>
public sealed record ChatTurnStatus(string Id, string Key, ChatTurnState State, int FrameCount)
{
    public bool IsRunning => State == ChatTurnState.Running;
}

/// <summary>
/// A generation that belongs to the APP rather than to the request that started it.
///
/// <para>
/// The Web UI's contract is one <c>POST /api/chat</c> whose response IS the answer: the
/// page reads the event stream and the server stops generating when that stream is
/// dropped. On a desktop that is right — the tab does not go away. On a phone it is the
/// difference between an assistant and a toy: the user opens the model list, glances at
/// a message in another app, or the screen dims, and the turn they waited a minute for
/// dies with the connection. Every reported version of "it stopped answering" on this
/// device is a reader that went away.
/// </para>
/// <para>
/// So the turn is started here, runs on its own, and buffers every frame it produces.
/// A reader — the original request, or the page after it comes back — ATTACHES to it,
/// replays what it missed and then follows along; a reader that leaves cancels nothing.
/// The only things that stop a turn are the user asking for it (the Stop button), a new
/// turn on the same conversation, and the app shutting down. It is the same ownership
/// change <see cref="Downloads.ModelDownloadManager"/> made for a five-gigabyte
/// download, for the same reason and with the same shape.
/// </para>
/// <para>
/// The transcript is written here too, from the frames, so an answer survives even when
/// nobody was reading it when it finished.
/// </para>
/// </summary>
public sealed class ChatTurnManager : IDisposable
{
    private readonly ConversationRecorder? _recorder;
    private readonly object _gate = new();
    private readonly Dictionary<string, Turn> _byKey = new(StringComparer.Ordinal);
    private readonly Dictionary<string, Turn> _byId = new(StringComparer.Ordinal);
    private bool _disposed;

    public ChatTurnManager(ConversationRecorder? recorder = null) => _recorder = recorder;

    /// <summary>
    /// Raised when the app goes from generating nothing to generating something, and
    /// back.
    ///
    /// <para>
    /// It is the HOST's answer, not the page's, and that distinction is the point. The
    /// two things a phone has to do while a model is working — keep the display awake,
    /// and ask iOS for time when the user leaves the app — must follow the turn, not
    /// whichever page happens to be watching it. A page that navigated away has stopped
    /// reading and has not stopped anything else.
    /// </para>
    /// </summary>
    public event Action<bool>? BusyChanged;

    /// <summary>True while any turn is still generating.</summary>
    public bool IsBusy
    {
        get
        {
            lock (_gate)
            {
                foreach (Turn turn in _byId.Values)
                    if (turn.Snapshot().IsRunning)
                        return true;
                return false;
            }
        }
    }

    private int _busy;

    /// <summary>Announce a change in <see cref="IsBusy"/>, and only a change.</summary>
    private void RaiseBusy()
    {
        bool busy = IsBusy;
        if (Interlocked.Exchange(ref _busy, busy ? 1 : 0) == (busy ? 1 : 0))
            return;
        try { BusyChanged?.Invoke(busy); }
        catch (Exception) { /* a listener must not break a turn */ }
    }

    /// <summary>
    /// How long a finished turn stays attachable. It exists for the window that matters
    /// most: an answer that completes while the user is somewhere else, and is read when
    /// they come back. After this the saved conversation is the record, which is what a
    /// relaunch reads anyway.
    ///
    /// <para>
    /// An hour rather than ten minutes, because "somewhere else" is routinely longer
    /// than that: the phone's own trace shows returns after 15, 27 and 52 minutes with
    /// the app still alive. A page that comes back to find its turn evicted cannot show
    /// the ending it missed, and the cost of keeping a finished turn is a few thousand
    /// small frames per conversation.
    /// </para>
    /// </summary>
    public TimeSpan RetainFinished { get; init; } = TimeSpan.FromHours(1);

    /// <summary>
    /// How many finished turns are kept at once, whatever <see cref="RetainFinished"/>
    /// says. The hour above is for the one turn a page might come back to; it is not a
    /// reason to hold every answer of a long session's worth of chats in memory, and a
    /// turn's buffer can reach <see cref="MaxBufferedFrames"/>. The oldest go first.
    /// </summary>
    public int MaxRetainedFinished { get; init; } = 8;

    /// <summary>
    /// The most frames one turn keeps for replay. Reached only by a runaway generation —
    /// the app's own cap is a few thousand tokens — and when it is, the buffer is
    /// COMPACTED rather than trimmed: one <c>replace</c> frame carrying the whole answer
    /// so far, which is exactly what a replaying reader would have rebuilt from the
    /// frames it replaced. Dropping the oldest frames instead would hand a re-attaching
    /// page an answer with its beginning missing.
    ///
    /// <para>
    /// Frame numbers stay ABSOLUTE across a compaction — a reader holds a position, and
    /// renumbering underneath it would either skip frames or repeat them — so the turn
    /// remembers how many it collapsed and a reader sitting before that point is moved
    /// forward to the <c>replace</c> that stands for them.
    /// </para>
    /// </summary>
    public int MaxBufferedFrames { get; init; } = 50_000;

    /// <summary>
    /// Start a turn for <paramref name="key"/> and return its id.
    ///
    /// <para>
    /// A turn already running under the same key is cancelled first: one conversation
    /// generates one answer at a time, and a page that sent a second request without
    /// stopping the first means the first is no longer wanted.
    /// </para>
    /// </summary>
    /// <param name="key">The conversation this belongs to, or the session when there is no conversation.</param>
    /// <param name="frames">The generation, started with the TURN's token — never a request's.</param>
    public string Start(string key, Func<CancellationToken, IAsyncEnumerable<object>> frames)
    {
        ArgumentException.ThrowIfNullOrEmpty(key);
        ArgumentNullException.ThrowIfNull(frames);

        var turn = new Turn(Guid.NewGuid().ToString("N"), key);
        Turn? previous;
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            EvictExpired();
            _byKey.TryGetValue(key, out previous);
            _byKey[key] = turn;
            _byId[turn.Id] = turn;
        }

        // Outside the lock: cancelling runs the producer's continuations.
        previous?.Cancel();

        RaiseBusy();

        // On its own thread, and deliberately not awaited. The first pull of a
        // generation blocks for as long as a prefill takes, and the request that asked
        // for this turn must be free to start streaming it long before then.
        _ = Task.Run(() => PumpAsync(turn, frames));
        return turn.Id;
    }

    /// <summary>The turn for a conversation, running or recently finished, or null.</summary>
    public ChatTurnStatus? StatusOfKey(string? key)
    {
        if (string.IsNullOrEmpty(key))
            return null;
        lock (_gate)
        {
            EvictExpired();
            return _byKey.TryGetValue(key, out Turn? turn) ? turn.Snapshot() : null;
        }
    }

    /// <summary>The turn with this id, running or recently finished, or null.</summary>
    public ChatTurnStatus? StatusOfId(string? id)
    {
        if (string.IsNullOrEmpty(id))
            return null;
        lock (_gate)
        {
            EvictExpired();
            return _byId.TryGetValue(id, out Turn? turn) ? turn.Snapshot() : null;
        }
    }

    /// <summary>
    /// One line naming every turn this launch still knows about: what it belongs to,
    /// what it is doing, and how much it has produced.
    ///
    /// <para>
    /// For the launch log and for the device probe, which is the only place the claim
    /// this class exists to make can actually be checked: that the frame count keeps
    /// rising while the page that asked for the answer is not on screen.
    /// </para>
    /// </summary>
    public string Describe()
    {
        lock (_gate)
        {
            if (_byId.Count == 0)
                return "no turns";
            return string.Join(", ", _byId.Values
                .Select(turn => turn.Snapshot())
                .Select(turn => $"{turn.Key}={turn.State} {turn.FrameCount} frames"));
        }
    }

    /// <summary>How many frames the turn for a conversation has produced, or -1 for none.</summary>
    public int FramesFor(string? key) => StatusOfKey(key)?.FrameCount ?? -1;

    /// <summary>
    /// Frames produced by every turn this launch still knows about. A number that goes
    /// up while no page is reading is the evidence that a turn belongs to the app, and
    /// it is what the device probe watches.
    /// </summary>
    public int TotalFrames
    {
        get
        {
            lock (_gate)
            {
                int frames = 0;
                foreach (Turn turn in _byId.Values)
                    frames += turn.Snapshot().FrameCount;
                return frames;
            }
        }
    }

    /// <summary>Ask the turn for a conversation to stop. True when there was one running.</summary>
    public bool StopKey(string? key)
    {
        Turn? turn;
        lock (_gate)
        {
            if (key is null || !_byKey.TryGetValue(key, out turn))
                return false;
        }
        return turn.Cancel();
    }

    /// <summary>Ask one turn to stop, by id. True when it was running.</summary>
    public bool StopId(string? id)
    {
        Turn? turn;
        lock (_gate)
        {
            if (id is null || !_byId.TryGetValue(id, out turn))
                return false;
        }
        return turn.Cancel();
    }

    /// <summary>Stop everything, which is what shutting the app down means.</summary>
    public void StopAll()
    {
        Turn[] all;
        lock (_gate)
            all = _byId.Values.ToArray();
        foreach (Turn turn in all)
            turn.Cancel();
    }

    /// <summary>
    /// Follow a turn from <paramref name="from"/> onwards: the frames it has already
    /// produced, then the ones it produces next, ending when it does.
    ///
    /// <para>
    /// <paramref name="ct"/> belongs to the READER. Cancelling it ends this enumeration
    /// and nothing else — which is the whole point: the request that started the turn
    /// can go away without taking the answer with it.
    /// </para>
    /// <para>
    /// A turn that was refused before it produced anything (no model loaded, an unknown
    /// session) rethrows that refusal here, before the first frame, so the route can
    /// still answer with a status code instead of an empty event stream.
    /// </para>
    /// </summary>
    public async IAsyncEnumerable<object> WatchAsync(
        string id, int from, [EnumeratorCancellation] CancellationToken ct)
    {
        Turn? turn;
        lock (_gate)
            _byId.TryGetValue(id, out turn);
        if (turn is null)
            throw new KeyNotFoundException($"there is no turn called {id}");

        int index = Math.Max(0, from);
        while (true)
        {
            object[] batch;
            Task? wait;
            bool finished;
            lock (turn.Sync)
            {
                (batch, wait, finished) = turn.Read(ref index);
                if (finished && turn.Rejection is { } rejection && index == 0)
                    throw rejection;
            }

            foreach (object frame in batch)
                yield return frame;
            if (finished)
                yield break;
            if (wait is not null)
                await wait.WaitAsync(ct).ConfigureAwait(false);
        }
    }

    /// <summary>
    /// Run the generation to its end, buffering as it goes, and write the answer down.
    ///
    /// <para>
    /// The transcript is assembled from the very frames the page reads — the same
    /// <c>token</c>, <c>replace</c> and <c>thinking</c> events — because those are the
    /// only place the reply exists on this side. Doing it here rather than in the route
    /// is what makes an answer survive the page that asked for it: the request may be
    /// long gone by the last token.
    /// </para>
    /// </summary>
    private async Task PumpAsync(Turn turn, Func<CancellationToken, IAsyncEnumerable<object>> frames)
    {
        var content = new StringBuilder();
        var thinking = new StringBuilder();
        var artifacts = new List<StoredArtifact>();
        var artifactUrls = new HashSet<string>(StringComparer.Ordinal);
        string? sessionId = null;
        string? imageUrl = null, videoUrl = null, audioUrl = null;
        ImageTurnRecord? image = null;
        StoredTurnStats? stats = null;
        ChatTurnState state;

        try
        {
            await foreach (object frame in frames(turn.Token).WithCancellation(turn.Token).ConfigureAwait(false))
            {
                turn.Append(frame, content, MaxBufferedFrames);
                ReadInto(frame, content, thinking, artifacts, artifactUrls, ref sessionId, ref imageUrl, ref videoUrl, ref audioUrl, ref image, ref stats);
            }
            state = ChatTurnState.Completed;
        }
        catch (OperationCanceledException)
        {
            state = ChatTurnState.Cancelled;
        }
        catch (Exception ex)
        {
            // A refusal that arrived before any frame keeps its identity, so the route
            // can still turn it into a status code. Once frames have gone out the status
            // is spent, and the honest thing left is an error frame the page renders.
            if (turn.IsEmpty)
            {
                turn.Reject(ex);
            }
            else
            {
                turn.Append(new { error = ex.Message }, content, MaxBufferedFrames);
            }
            state = ChatTurnState.Failed;
        }

        // Only the turn that still OWNS the conversation writes the transcript.
        //
        // A turn that was cancelled because a NEW one replaced it can take a long time
        // to unwind -- a shell command is a synchronous wait of up to the tool timeout --
        // and by then the replacement has finished and saved its answer. Recorder.Complete
        // removes the last assistant message before appending, so the stale turn would
        // delete the answer the user is looking at and put its own fragment under the
        // wrong question. A turn the USER stopped is still saved: it is still the turn
        // this conversation has, and its partial answer is what is on the screen.
        if (sessionId is not null && StillOwns(turn))
        {
            try
            {
                _recorder?.Complete(
                    sessionId,
                    content.ToString(),
                    thinking.ToString(),
                    artifacts.Count == 0 ? null : artifacts,
                    imageUrl,
                    videoUrl,
                    audioUrl,
                    stats,
                    image);
            }
            catch (Exception) { /* a lost transcript must not be a crash on a background thread */ }
        }

        turn.Finish(state);
        RaiseBusy();
    }

    /// <summary>Whether this turn is still the one its conversation is generating.</summary>
    private bool StillOwns(Turn turn)
    {
        lock (_gate)
            return _byKey.TryGetValue(turn.Key, out Turn? current) && ReferenceEquals(current, turn);
    }

    /// <summary>
    /// Read one frame the way the page reads it. The frames are anonymous objects, so
    /// JSON is the only way in — the same round trip the route's recorder used to do,
    /// and the only way to stay honest about what was actually sent.
    /// </summary>
    private static void ReadInto(
        object frame,
        StringBuilder content,
        StringBuilder thinking,
        List<StoredArtifact> artifacts,
        HashSet<string> artifactUrls,
        ref string? sessionId,
        ref string? imageUrl,
        ref string? videoUrl,
        ref string? audioUrl,
        ref ImageTurnRecord? image,
        ref StoredTurnStats? stats)
    {
        try
        {
            using JsonDocument document = JsonDocument.Parse(
                JsonSerializer.Serialize(frame, Hosting.SseFraming.JsonOptions));
            JsonElement root = document.RootElement;
            if (root.ValueKind != JsonValueKind.Object)
                return;
            if (root.TryGetProperty("done", out JsonElement done) && done.ValueKind == JsonValueKind.True)
                stats = StoredTurnStats.FromDoneFrame(root);
            if (root.TryGetProperty("token", out JsonElement token) && token.GetString() is { } piece)
                content.Append(piece);
            else if (root.TryGetProperty("replace", out JsonElement replace) && replace.GetString() is { } whole)
                content.Clear().Append(whole);
            else if (root.TryGetProperty("thinking", out JsonElement thought) && thought.GetString() is { } reasoning)
                thinking.Append(reasoning);
            // The host is answering again after a GPU fault (see AgentAppHost
            // .GatedChatFrames). The page drops the reasoning box on this frame, and the
            // transcript must say what the page showed: the reasoning of the attempt
            // that died is not part of the answer that replaced it. (An empty `replace`
            // rides on the same frame when the answer itself starts over.)
            if (root.TryGetProperty("restart", out _))
            {
                thinking.Clear();
                stats = null;
            }

            // A tool's ordinary `files` field is provisional: a guarded workflow can
            // produce a syntactically valid-looking file and then reject it for stale
            // contents, unsafe package relationships, or a missing required step. Only
            // the dedicated frame emitted after the host completion proof belongs in
            // durable history. This also mirrors the page's URL-based de-duplication
            // when a replay contains the same verified frame more than once.
            if (root.TryGetProperty("artifact_verified", out JsonElement verified)
                && verified.ValueKind == JsonValueKind.True
                && root.TryGetProperty("files", out JsonElement files)
                && files.ValueKind == JsonValueKind.Array)
            {
                foreach (JsonElement file in files.EnumerateArray())
                {
                    if (file.ValueKind != JsonValueKind.Object
                        || !file.TryGetProperty("url", out JsonElement urlElement)
                        || urlElement.GetString() is not { Length: > 0 } url
                        || !artifactUrls.Add(url))
                    {
                        continue;
                    }

                    string name = file.TryGetProperty("name", out JsonElement nameElement)
                        && nameElement.GetString() is { Length: > 0 } suppliedName
                            ? suppliedName
                            : url;
                    long bytes = file.TryGetProperty("bytes", out JsonElement bytesElement)
                        && bytesElement.TryGetInt64(out long suppliedBytes)
                        && suppliedBytes > 0
                            ? suppliedBytes
                            : 0;
                    artifacts.Add(new StoredArtifact { Name = name, Bytes = bytes, Url = url });
                }
            }
            if (root.TryGetProperty("sessionId", out JsonElement id) && id.GetString() is { Length: > 0 } value)
                sessionId = value;
            // The picture an image model's turn made, which may be all it made (ImageTurns).
            if (root.TryGetProperty("imageUrl", out JsonElement picture)
                && picture.ValueKind == JsonValueKind.String
                && picture.GetString() is { Length: > 0 } made)
                imageUrl = made;
            // And what that picture was made from, or the readings a turn offered instead of
            // making one: the next image turn plans from these (ImageTurnRecord).
            if (ImageTurnRecord.FromFrame(root) is { } recorded)
                image = recorded;
            // Likewise the clip a video model's turn made, and its soundtrack when that is a
            // file of its own (VideoTurns); both arrive on the same frame.
            if (root.TryGetProperty("videoUrl", out JsonElement clip)
                && clip.ValueKind == JsonValueKind.String
                && clip.GetString() is { Length: > 0 } filmed)
            {
                videoUrl = filmed;
                audioUrl = root.TryGetProperty("audioUrl", out JsonElement sound)
                    && sound.ValueKind == JsonValueKind.String
                    && sound.GetString() is { Length: > 0 } heard
                        ? heard
                        : null;
            }
        }
        catch (JsonException)
        {
            // A frame that will not serialise is one the page could not have read either.
        }
    }

    /// <summary>Forget turns nobody can still be waiting for. Called under <see cref="_gate"/>.</summary>
    private void EvictExpired()
    {
        EvictSurplusFinished();
        DateTimeOffset cutoff = DateTimeOffset.UtcNow - RetainFinished;
        List<string>? expired = null;
        foreach (KeyValuePair<string, Turn> entry in _byId)
        {
            if (entry.Value.FinishedBefore(cutoff))
                (expired ??= new List<string>()).Add(entry.Key);
        }
        if (expired is null)
            return;
        foreach (string id in expired)
            Forget(id);
    }

    /// <summary>
    /// Keep the newest <see cref="MaxRetainedFinished"/> finished turns and let the
    /// rest go. Under <see cref="_gate"/>, called beside <see cref="EvictExpired"/>:
    /// an hour is a long time on a phone, and a session that answers steadily would
    /// otherwise hold every one of those answers' frames until it ended.
    /// </summary>
    private void EvictSurplusFinished()
    {
        List<KeyValuePair<string, DateTimeOffset>>? finished = null;
        foreach (KeyValuePair<string, Turn> entry in _byId)
        {
            if (entry.Value.FinishedAt is { } at)
                (finished ??= new List<KeyValuePair<string, DateTimeOffset>>()).Add(new(entry.Key, at));
        }
        if (finished is null || finished.Count <= MaxRetainedFinished)
            return;
        finished.Sort(static (a, b) => a.Value.CompareTo(b.Value));
        for (int i = 0; i < finished.Count - MaxRetainedFinished; i++)
            Forget(finished[i].Key);
    }

    /// <summary>Drop one turn from both indexes. Under <see cref="_gate"/>.</summary>
    private void Forget(string id)
    {
        if (!_byId.Remove(id, out Turn? turn))
            return;
        if (_byKey.TryGetValue(turn.Key, out Turn? current) && ReferenceEquals(current, turn))
            _byKey.Remove(turn.Key);
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed)
                return;
            _disposed = true;
        }
        StopAll();
    }

    /// <summary>The buffer, the state and the one signal readers wait on.</summary>
    private sealed class Turn(string id, string key)
    {
        private readonly List<object> _frames = new();
        /// <summary>Absolute number of frames a compaction collapsed; <c>_frames[0]</c> is that one.</summary>
        private int _dropped;
        private readonly CancellationTokenSource _cancel = new();
        private TaskCompletionSource _changed = new(TaskCreationOptions.RunContinuationsAsynchronously);
        private ChatTurnState _state = ChatTurnState.Running;
        private DateTimeOffset? _finishedAt;
        private Exception? _rejection;

        /// <summary>Guards everything below; also what a reader waits under.</summary>
        public object Sync { get; } = new();

        public string Id { get; } = id;
        public string Key { get; } = key;
        public CancellationToken Token => _cancel.Token;
        public Exception? Rejection { get { lock (Sync) return _rejection; } }
        public bool IsEmpty { get { lock (Sync) return _dropped + _frames.Count == 0; } }

        public ChatTurnStatus Snapshot()
        {
            lock (Sync)
                return new ChatTurnStatus(Id, Key, _state, _dropped + _frames.Count);
        }

        /// <summary>True when this actually stopped something that was running.</summary>
        public bool Cancel()
        {
            lock (Sync)
            {
                if (_state != ChatTurnState.Running)
                    return false;
            }
            try { _cancel.Cancel(); return true; }
            catch (ObjectDisposedException) { return false; }
        }

        public void Append(object frame, StringBuilder answerSoFar, int maxFrames)
        {
            lock (Sync)
            {
                if (_frames.Count >= maxFrames)
                {
                    // Compaction, not truncation: one frame that sets the whole answer is
                    // what a reader replaying everything before it would have ended up
                    // with. It takes the absolute number of the LAST frame it stands for,
                    // so every position a reader can be holding is still meaningful.
                    _dropped += _frames.Count - 1;
                    _frames.Clear();
                    _frames.Add(new { replace = answerSoFar.ToString() });
                }
                _frames.Add(frame);
                Signal();
            }
        }

        public void Reject(Exception ex)
        {
            lock (Sync)
                _rejection = ex;
        }

        public void Finish(ChatTurnState state)
        {
            lock (Sync)
            {
                _state = state;
                _finishedAt = DateTimeOffset.UtcNow;
                Signal();
            }
        }

        /// <summary>
        /// What a reader sitting at <paramref name="index"/> should do next: the frames it
        /// has not seen, or the task to wait on, or nothing because the turn is over.
        /// Called under <see cref="Sync"/>.
        /// </summary>
        public (object[] Batch, Task? Wait, bool Finished) Read(ref int index)
        {
            // A reader sitting inside what a compaction collapsed is moved to the frame
            // that stands for all of it; one asking for more than exists (a stale `from`,
            // or a client that made a number up) is moved to the end rather than reading
            // off it.
            int total = _dropped + _frames.Count;
            if (index < _dropped)
                index = _dropped;
            if (index > total)
                index = total;

            if (index < total)
            {
                int at = index - _dropped;
                object[] batch = _frames.GetRange(at, _frames.Count - at).ToArray();
                index = total;
                return (batch, null, false);
            }
            if (_state != ChatTurnState.Running)
                return (Array.Empty<object>(), null, true);
            return (Array.Empty<object>(), _changed.Task, false);
        }

        public bool FinishedBefore(DateTimeOffset cutoff)
        {
            lock (Sync)
                return _finishedAt is { } finished && finished < cutoff;
        }

        /// <summary>When this turn ended, or null while it is still running.</summary>
        public DateTimeOffset? FinishedAt { get { lock (Sync) return _finishedAt; } }

        /// <summary>Wake every reader. Called under <see cref="Sync"/>.</summary>
        private void Signal()
        {
            TaskCompletionSource previous = _changed;
            _changed = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            previous.TrySetResult();
        }
    }
}
