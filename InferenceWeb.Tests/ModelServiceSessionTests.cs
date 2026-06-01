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
/// Unit tests for <see cref="ModelService"/> session-lifecycle operations. These
/// do not load a real model; server sessions now only own tracked history.
/// </summary>
public class ModelServiceSessionTests
{
    [Fact]
    public void ResetSession_ClearsHistoryForTheGivenSession()
    {
        var svc = new ModelService();
        var session = new ChatSession();
        session.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "hi" });

        svc.ResetSession(session);

        Assert.Empty(session.TrackedHistory);
        Assert.False(session.IsDisposed);
    }

    [Fact]
    public void ResetSession_LeavesOtherSessionsUntouched()
    {
        var svc = new ModelService();
        var sessA = new ChatSession();
        var sessB = new ChatSession();

        sessA.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "a" });
        sessB.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "b" });

        svc.ResetSession(sessA);

        Assert.Empty(sessA.TrackedHistory);
        Assert.Single(sessB.TrackedHistory);
    }

    [Fact]
    public void ResetSession_NullIsNoOp()
    {
        var svc = new ModelService();
        svc.ResetSession(null); // must not throw
    }

    [Fact]
    public void DisposeSession_MarksSessionDisposedAndFreesState()
    {
        var svc = new ModelService();
        var session = new ChatSession();
        session.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "x" });

        svc.DisposeSession(session);

        Assert.True(session.IsDisposed);
        Assert.Empty(session.TrackedHistory);
    }

    [Fact]
    public void DisposeSession_DoesNotAffectOtherSessions()
    {
        // Disposing session A must not release any bookkeeping attached to session B.
        var svc = new ModelService();
        var sessA = new ChatSession();
        var sessB = new ChatSession();

        sessB.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "keep" });

        svc.DisposeSession(sessA);

        Assert.False(sessB.IsDisposed);
        Assert.Single(sessB.TrackedHistory);
    }

    [Fact]
    public void DisposeSession_NullIsNoOp()
    {
        var svc = new ModelService();
        svc.DisposeSession(null);
    }

    [Fact]
    public void DisposeSession_TwiceIsIdempotent()
    {
        var svc = new ModelService();
        var session = new ChatSession();

        svc.DisposeSession(session);
        svc.DisposeSession(session); // already disposed -> no-op

        Assert.True(session.IsDisposed);
    }

    [Fact]
    public void InvalidateKVCache_DoesNotThrowWhenNoModelLoaded()
    {
        var svc = new ModelService();
        svc.InvalidateKVCache();
        Assert.False(svc.IsLoaded);
    }

    [Fact]
    public void KVCache_ReturnsIsolatedCompatibilityShim()
    {
        var svc = new ModelService();
        var legacyView = svc.KVCache;
        legacyView.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });

        Assert.True(svc.KVCache.IsEmpty);
    }

    [Fact]
    public void ActiveSession_StartsNullUntilInferenceActivatesOne()
    {
        var svc = new ModelService();
        Assert.Null(svc.ActiveSession);
    }
}
