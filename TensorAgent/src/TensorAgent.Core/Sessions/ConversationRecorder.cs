// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Concurrent;
using System.Text.Json;

namespace TensorAgent.Core.Sessions;

/// <summary>
/// Keeps the saved transcript in step with what the page is doing.
///
/// <para>
/// The Web UI holds its history in the page and nowhere else: reload it and the
/// conversation is gone. On a desktop that is fine, because the tab stays open. On a
/// phone the app is suspended and killed constantly, and a chat the user had this
/// morning has to still be there this afternoon, so the transcript is written on this
/// side instead.
/// </para>
/// <para>
/// The binding is the awkward part. The engine's session id and the app's
/// conversation id are different things with different lifetimes — a resumed
/// conversation gets a brand-new engine session, because the model's KV cache did not
/// survive the app being killed — so the two are joined when the session is created
/// and the join is what lets a chat request, which knows only its session, be filed
/// under the right conversation.
/// </para>
/// </summary>
public sealed class ConversationRecorder
{
    private readonly ConversationStore _store;
    private readonly ConcurrentDictionary<string, string> _sessionToConversation = new(StringComparer.Ordinal);
    private int _binds;
    private string? _current;

    public ConversationRecorder(ConversationStore store)
        => _store = store ?? throw new ArgumentNullException(nameof(store));

    /// <summary>
    /// Bind an engine session to a conversation, creating the conversation when the
    /// page asked for a new one. Returns the conversation now backing that session.
    /// </summary>
    /// <param name="sessionId">The engine session the page just created.</param>
    /// <param name="requested">The conversation id the page asked to resume, or "new"/null for a fresh one.</param>
    public Conversation Bind(string sessionId, string? requested)
    {
        Conversation? conversation = requested is { Length: > 0 } id && !string.Equals(id, "new", StringComparison.Ordinal)
            ? _store.Load(id)
            : null;
        // An empty conversation is one nobody has typed into yet, so a second session
        // asking for a new chat gets that one rather than another beside it. Without
        // this, every launch of the app leaves a row behind.
        conversation ??= _store.MostRecentEmpty();
        conversation ??= _store.Create();
        _sessionToConversation[sessionId] = conversation.Id;
        Volatile.Write(ref _current, conversation.Id);
        Interlocked.Increment(ref _binds);
        return conversation;
    }

    /// <summary>
    /// True until the first chat of this app launch has been opened.
    ///
    /// <para>
    /// The page cannot tell its own first load apart from its fourth, and the two want
    /// opposite things. Opening the app should show a clean composer; a page that is
    /// merely coming back -- WebKit kills the content process of a WebView whose view
    /// left the window, and the reload lands in an app that never stopped, sometimes
    /// with a turn still generating on this side -- must return to the chat it was in.
    /// This side does know, because it is the app: a launch is a new process, and a
    /// new process has bound nothing yet.
    /// </para>
    /// <para>
    /// Counted rather than derived from <see cref="_sessionToConversation"/>, which
    /// empties again as sessions are disposed and would make a long-running app look
    /// freshly launched every time the user closed a chat.
    /// </para>
    /// </summary>
    public bool IsColdLaunch => Volatile.Read(ref _binds) == 0;

    /// <summary>
    /// The chat the page is in right now, or null before it has opened one.
    ///
    /// <para>
    /// What a reloading page comes back to. The obvious substitute — the newest saved
    /// conversation — is wrong twice over. It is not necessarily the one the user was
    /// reading, because opening an older chat from the menu saves nothing and so moves
    /// nothing to the top; and it can never be the empty chat a launch just opened,
    /// because <see cref="ConversationStore.List"/> omits conversations with no
    /// messages. That second case is the ordinary one: launch the app, go and look at
    /// the model list, come back to find yesterday's chat instead of the clean one you
    /// were given.
    /// </para>
    /// <para>
    /// Every route into a chat goes through <see cref="Bind"/> — the new-chat button,
    /// the menu rows, and the native Chats page — so this follows the page wherever it
    /// goes without the page having to report it.
    /// </para>
    /// </summary>
    public string? CurrentConversationId => Volatile.Read(ref _current);

    /// <summary>Forget a session that has been disposed. The conversation itself is untouched.</summary>
    public void Release(string sessionId) => _sessionToConversation.TryRemove(sessionId, out _);

    /// <summary>The conversation a session is filed under, or null when it was never bound.</summary>
    public string? ConversationFor(string sessionId)
        => _sessionToConversation.TryGetValue(sessionId, out string? id) ? id : null;

    /// <summary>
    /// Record the answer a turn produced, once it has finished.
    ///
    /// <para>
    /// <see cref="Record"/> alone is not enough, and the gap is the one that hurts on
    /// a phone. It runs when a request arrives, so it saves the history the page sent
    /// — which does not yet contain the reply that request is about to generate. The
    /// reply is only written down when the NEXT request carries it, so a user who
    /// asks a question, reads the answer and switches away loses exactly the answer
    /// they were reading. This closes the turn instead of waiting for another one.
    /// </para>
    /// </summary>
    /// <param name="sessionId">The session the answer belongs to.</param>
    /// <param name="content">The assistant's text, as the page assembled it.</param>
    /// <param name="thinking">Its reasoning, when the model produced any.</param>
    /// <param name="artifacts">Files the turn's tools produced, as the page's chips name them.</param>
    /// <param name="imageUrl">The picture an image model made this turn, which may be all it made.</param>
    /// <param name="videoUrl">The clip a video model made this turn, likewise.</param>
    /// <param name="audioUrl">That clip's soundtrack, when it is a separate file rather than inside the clip.</param>
    /// <param name="stats">The model's terminal counters, when supplied by a text turn.</param>
    /// <param name="image">What the picture was made from, or the readings an image turn offered
    /// instead of making one (see <see cref="ImageTurnRecord"/>).</param>
    public void Complete(
        string sessionId, string content, string? thinking = null,
        IReadOnlyList<StoredArtifact>? artifacts = null, string? imageUrl = null,
        string? videoUrl = null, string? audioUrl = null, StoredTurnStats? stats = null,
        ImageTurnRecord? image = null)
    {
        // Image/video completion frames carry zero-token placeholder counters. Keep
        // actual LLM counters when a text turn also produced media through a tool.
        if (stats?.TokenCount == 0 && (!string.IsNullOrEmpty(imageUrl) || !string.IsNullOrEmpty(videoUrl)))
            stats = null;
        if (string.IsNullOrEmpty(content) && string.IsNullOrEmpty(thinking) && artifacts is not { Count: > 0 }
            && string.IsNullOrEmpty(imageUrl) && string.IsNullOrEmpty(videoUrl))
            return;
        try
        {
            string? conversationId = ConversationFor(sessionId);
            if (conversationId is null || _store.Load(conversationId) is not { } conversation)
                return;

            // An answer that arrives twice for the same turn replaces the first: a
            // regenerated reply is a correction, not a second message.
            if (conversation.Messages.Count > 0 && conversation.Messages[^1].Role == "assistant")
                conversation.Messages.RemoveAt(conversation.Messages.Count - 1);

            var answer = new StoredMessage
            {
                Role = "assistant",
                Content = content,
                Thinking = string.IsNullOrEmpty(thinking) ? null : thinking,
                Stats = stats,
                // The file a turn produced is usually the whole point of the turn — the
                // PDF, the spreadsheet, the clip. It arrives on the frames rather than in
                // the answer (a small model repeats a link erratically), so it is written
                // down here or it is lost the moment the user opens another chat.
                Artifacts = artifacts is { Count: > 0 } ? new List<StoredArtifact>(artifacts) : null,
                ImageUrl = string.IsNullOrEmpty(imageUrl) ? null : imageUrl,
                VideoUrl = string.IsNullOrEmpty(videoUrl) ? null : videoUrl,
                AudioUrl = string.IsNullOrEmpty(videoUrl) || string.IsNullOrEmpty(audioUrl) ? null : audioUrl,
            };
            image?.ApplyTo(answer);
            conversation.Messages.Add(answer);
            _store.Save(conversation);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            // Same reasoning as Record: a lost transcript must not cost the answer.
        }
    }

    /// <summary>
    /// Record one accepted chat turn.
    ///
    /// <para>
    /// The whole <c>messages</c> array is written rather than only the newest message,
    /// because the page is the authority on what the conversation contains: it may
    /// have edited a message, dropped one, or started from a resumed transcript. Taking
    /// the array wholesale means the saved copy is whatever the page believes, which is
    /// what the user will see when they come back to it.
    /// </para>
    /// <para>
    /// Failures here are swallowed. This runs on the request path of a turn the user
    /// is waiting for, and losing a saved transcript is a smaller harm than failing the
    /// generation that produced it.
    /// </para>
    /// </summary>
    public void Record(string sessionId, JsonElement body)
    {
        try
        {
            string? conversationId = ConversationFor(sessionId);
            if (conversationId is null)
                return;
            Conversation? conversation = _store.Load(conversationId);
            if (conversation is null)
                return;

            if (body.TryGetProperty("messages", out JsonElement messages) && messages.ValueKind == JsonValueKind.Array)
            {
                // What each picture was made from is written down when the picture is (see
                // Complete), and the next image turn plans from it. The page copies it into
                // its history too, but a copy that lacks it -- a page from before it existed,
                // or one that rebuilt an entry -- must not erase it here, or the picture would
                // read as one drawn from the words before it. Only a picture with no record
                // at all is left to that reading.
                var madeFrom = new Dictionary<string, ImageTurnRecord>(StringComparer.Ordinal);
                foreach (StoredMessage known in conversation.Messages)
                {
                    if (known.ImageUrl is { Length: > 0 } url && ImageTurnRecord.MadeFrom(known) is { } record)
                        madeFrom[url] = record;
                }
                var replacement = new List<StoredMessage>(messages.GetArrayLength());
                foreach (JsonElement message in messages.EnumerateArray())
                {
                    if (message.Deserialize<StoredMessage>(ConversationStore.Json) is { } stored)
                    {
                        if (stored.ImagePlan is null && stored.ImageUrl is { Length: > 0 } picture
                            && madeFrom.TryGetValue(picture, out ImageTurnRecord? record))
                            record.ApplyTo(stored);
                        replacement.Add(stored);
                    }
                }
                conversation.Messages = replacement;
            }
            if (body.TryGetProperty("model", out JsonElement model) && model.GetString() is { Length: > 0 } name)
                conversation.ModelId = name;
            if (body.TryGetProperty("think", out JsonElement think) && think.ValueKind is JsonValueKind.True or JsonValueKind.False)
                conversation.Think = think.GetBoolean();
            if (body.TryGetProperty("skills", out JsonElement skills) && skills.ValueKind == JsonValueKind.Array)
            {
                conversation.Skills.Clear();
                foreach (JsonElement skill in skills.EnumerateArray())
                    if (skill.GetString() is { Length: > 0 } skillName)
                        conversation.Skills.Add(skillName);
                conversation.SkillsExplicit = true;
            }
            else
            {
                // The page omits `skills` only for an untouched selection, where the
                // host is free to infer a route. Do not turn an inferred skill into a
                // sticky user choice when this conversation is reopened.
                conversation.Skills.Clear();
                conversation.SkillsExplicit = false;
            }

            _store.Save(conversation);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or JsonException)
        {
            // See the summary: a lost transcript must not cost the user their answer.
        }
    }
}
