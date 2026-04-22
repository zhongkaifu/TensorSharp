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
/// Unit tests for <see cref="SessionManager"/>: creation, lookup, removal, and the
/// invariant that the built-in default session is always present and never removable.
/// </summary>
public class SessionManagerTests
{
    [Fact]
    public void NewManager_HasDefaultSessionOnly()
    {
        var mgr = new SessionManager();

        Assert.Equal(1, mgr.SessionCount);
        Assert.NotNull(mgr.DefaultSession);
        Assert.Equal(SessionManager.DefaultSessionId, mgr.DefaultSession.Id);
    }

    [Fact]
    public void CreateSession_ProducesFreshSessionWithUniqueId()
    {
        var mgr = new SessionManager();
        var a = mgr.CreateSession();
        var b = mgr.CreateSession();

        Assert.NotEqual(a.Id, b.Id);
        Assert.Equal(3, mgr.SessionCount); // default + a + b
    }

    [Fact]
    public void GetSession_ByIdReturnsExactInstance()
    {
        var mgr = new SessionManager();
        var created = mgr.CreateSession();

        var looked = mgr.GetSession(created.Id);

        Assert.Same(created, looked);
    }

    [Fact]
    public void GetSession_EmptyIdReturnsDefault()
    {
        var mgr = new SessionManager();
        Assert.Same(mgr.DefaultSession, mgr.GetSession(null));
        Assert.Same(mgr.DefaultSession, mgr.GetSession(""));
        Assert.Same(mgr.DefaultSession, mgr.GetSession("   "));
    }

    [Fact]
    public void GetSession_UnknownIdReturnsNull()
    {
        var mgr = new SessionManager();
        Assert.Null(mgr.GetSession("does-not-exist-123"));
    }

    [Fact]
    public void TryRemove_ReturnsSessionAndRemovesFromRegistry()
    {
        var mgr = new SessionManager();
        var created = mgr.CreateSession();
        string id = created.Id;

        var removed = mgr.TryRemove(id);

        Assert.Same(created, removed);
        Assert.Null(mgr.GetSession(id));
    }

    [Fact]
    public void TryRemove_NonexistentIdReturnsNull()
    {
        var mgr = new SessionManager();
        Assert.Null(mgr.TryRemove("no-such-session"));
    }

    [Fact]
    public void TryRemove_DefaultSessionIdReturnsNullAndKeepsSession()
    {
        // The default session must survive for the lifetime of the server since the
        // Ollama / OpenAI endpoints rely on it for cache reuse.
        var mgr = new SessionManager();
        var before = mgr.DefaultSession;

        var result = mgr.TryRemove(SessionManager.DefaultSessionId);

        Assert.Null(result);
        Assert.Same(before, mgr.DefaultSession);
        Assert.Equal(1, mgr.SessionCount);
    }

    [Fact]
    public void CreatedSessions_HaveIsolatedState()
    {
        var mgr = new SessionManager();
        var a = mgr.CreateSession();
        var b = mgr.CreateSession();

        a.KVCache.RecordAppend(new[] { 10, 20, 30 }, new float[] { 1.0f });
        a.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "A's secret" });

        // Session B must not see any of A's state.
        Assert.True(b.KVCache.IsEmpty);
        Assert.Empty(b.TrackedHistory);
        Assert.Equal(3, a.KVCache.Count);
    }

    [Fact]
    public void TryRemove_SessionIsNotAutoDisposed()
    {
        // TryRemove only unregisters; disposal is the caller's responsibility so that
        // ModelService can reset the model's K/V tensors before the bookkeeping is
        // torn down. The returned session should still be usable for inspection.
        var mgr = new SessionManager();
        var created = mgr.CreateSession();
        created.KVCache.RecordAppend(7, new float[] { 0.1f });

        var removed = mgr.TryRemove(created.Id);

        Assert.False(removed!.IsDisposed);
        Assert.Equal(1, removed.KVCache.Count);
    }
}
