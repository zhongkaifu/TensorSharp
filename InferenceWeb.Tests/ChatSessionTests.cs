// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for <see cref="ChatSession"/>: ensures each session owns an isolated
/// <see cref="KVCache"/> and tracked-history buffer, and that disposing the session
/// fully clears that state.
/// </summary>
public class ChatSessionTests
{
    [Fact]
    public void NewSession_HasUniqueIdAndEmptyState()
    {
        var a = new ChatSession();
        var b = new ChatSession();

        Assert.False(string.IsNullOrEmpty(a.Id));
        Assert.False(string.IsNullOrEmpty(b.Id));
        Assert.NotEqual(a.Id, b.Id);
        Assert.True(a.KVCache.IsEmpty);
        Assert.True(b.KVCache.IsEmpty);
        Assert.Empty(a.TrackedHistory);
        Assert.Empty(b.TrackedHistory);
        Assert.False(a.IsDisposed);
    }

    [Fact]
    public void SessionsHaveIndependentState()
    {
        // Mutating one session's cache / history must not leak into another.
        var a = new ChatSession();
        var b = new ChatSession();

        a.KVCache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.1f });
        a.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "hello-a" });

        Assert.Equal(3, a.KVCache.Count);
        Assert.Single(a.TrackedHistory);
        Assert.True(b.KVCache.IsEmpty);
        Assert.Empty(b.TrackedHistory);
    }

    [Fact]
    public void Dispose_ClearsTokensAndHistory()
    {
        var session = new ChatSession();
        session.KVCache.RecordAppend(new[] { 42, 43 }, new float[] { 0.5f });
        session.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "hi" });

        session.Dispose();

        Assert.True(session.IsDisposed);
        Assert.True(session.KVCache.IsEmpty);
        Assert.Empty(session.TrackedHistory);
    }

    [Fact]
    public void Dispose_IsIdempotent()
    {
        var session = new ChatSession();
        session.Dispose();
        session.Dispose();
        Assert.True(session.IsDisposed);
    }

    [Fact]
    public void LastUsedAt_DefaultsToCreatedAt()
    {
        var session = new ChatSession();
        Assert.Equal(session.CreatedAt, session.LastUsedAt);
    }
}
