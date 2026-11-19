// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;

namespace TensorSharp.Server.Responses;

/// <summary>
/// The spoken replies this server sent (OpenAI's <c>message.audio.id</c>, the web chat's <c>audioId</c>), kept for an hour
/// (<c>expires_at</c>) so a client that sends one back in a later request's history
/// (<c>{"role":"assistant","audio":{"id":...}}</c>, or the web chat's <c>{"role":"assistant","audioId":...}</c>) gets that
/// turn rendered: spliced back as generation decoded it (its tokens, each frame's row from its codes) where the turn is
/// known, else as its transcript. In memory, per process. It lives beside the chat service rather than the HTTP server
/// because the web chat it also serves runs inside TensorAgent too.
/// </summary>
public sealed class SpeechReplyStore
{
    /// <summary>How long a reply is kept (OpenAI: an hour).</summary>
    public static readonly TimeSpan Lifetime = TimeSpan.FromHours(1);

    /// <summary>Replies kept at most (the oldest go first).</summary>
    private const int Capacity = 4096;

    private readonly ConcurrentDictionary<string, Entry> _entries = new(StringComparer.Ordinal);

    /// <param name="Turn">The reply as generation decoded it (its tokens and every decoded frame's codes), which a later
    /// turn splices back with the frames' rows; null where it is not known (the turn then renders as its transcript).</param>
    /// <param name="Epoch">The model load that generated it (<c>ModelLifecycleService.LoadEpoch</c>), for a caller that
    /// splices only under the same load: token ids are the vocabulary of the model that drew them. 0 when not tracked.</param>
    public sealed record Entry(string Id, string Transcript, IReadOnlyList<int[]> Frames, DateTimeOffset ExpiresAt,
        TensorSharp.Runtime.SpokenTurnRecord Turn = null, long Epoch = 0);

    /// <summary>Keep a reply; returns its entry (id and expiry).</summary>
    public Entry Add(string transcript, IReadOnlyList<int[]> frames, string id = null, TensorSharp.Runtime.SpokenTurnRecord turn = null,
        long epoch = 0)
    {
        Purge();
        var entry = new Entry(id ?? NewId(), transcript ?? string.Empty, frames ?? Array.Empty<int[]>(),
            DateTimeOffset.UtcNow + Lifetime, turn, epoch);
        _entries[entry.Id] = entry;
        return entry;
    }

    /// <summary>The reply's transcript, or null when the id is unknown or expired.</summary>
    public string TranscriptOf(string id)
        => id != null && _entries.TryGetValue(id, out Entry e) && e.ExpiresAt > DateTimeOffset.UtcNow ? e.Transcript : null;

    public bool TryGet(string id, out Entry entry)
        => _entries.TryGetValue(id ?? string.Empty, out entry) && entry.ExpiresAt > DateTimeOffset.UtcNow;

    public static string NewId() => "audio_" + Guid.NewGuid().ToString("N");

    private void Purge()
    {
        DateTimeOffset now = DateTimeOffset.UtcNow;
        foreach (var (key, e) in _entries)
            if (e.ExpiresAt <= now) _entries.TryRemove(key, out _);
        if (_entries.Count < Capacity) return;
        var oldest = new List<Entry>(_entries.Values);
        oldest.Sort((a, b) => a.ExpiresAt.CompareTo(b.ExpiresAt));
        for (int i = 0; i < oldest.Count - Capacity / 2; i++) _entries.TryRemove(oldest[i].Id, out _);
    }
}
