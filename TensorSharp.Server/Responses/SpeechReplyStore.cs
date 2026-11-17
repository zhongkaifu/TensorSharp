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
/// The spoken replies this server sent (OpenAI's <c>message.audio.id</c>), kept for an hour (<c>expires_at</c>) so a client
/// that sends one back in a later request's history (<c>{"role":"assistant","audio":{"id":...}}</c>) gets that turn
/// rendered: as its transcript, and the frames' codes are kept with it. In memory, per process.
/// </summary>
public sealed class SpeechReplyStore
{
    /// <summary>How long a reply is kept (OpenAI: an hour).</summary>
    public static readonly TimeSpan Lifetime = TimeSpan.FromHours(1);

    /// <summary>Replies kept at most (the oldest go first).</summary>
    private const int Capacity = 4096;

    private readonly ConcurrentDictionary<string, Entry> _entries = new(StringComparer.Ordinal);

    public sealed record Entry(string Id, string Transcript, IReadOnlyList<int[]> Frames, DateTimeOffset ExpiresAt);

    /// <summary>Keep a reply; returns its entry (id and expiry).</summary>
    public Entry Add(string transcript, IReadOnlyList<int[]> frames, string id = null)
    {
        Purge();
        var entry = new Entry(id ?? NewId(), transcript ?? string.Empty, frames ?? Array.Empty<int[]>(),
            DateTimeOffset.UtcNow + Lifetime);
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
