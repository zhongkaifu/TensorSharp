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
/// Unit tests for <see cref="ModelService"/> session-lifecycle operations. These do
/// not load a real model (they exercise only the cache bookkeeping), so they are
/// safe to run on any machine. The expensive K/V tensor reset is a no-op when no
/// model is loaded.
/// </summary>
public class ModelServiceSessionTests
{
    [Fact]
    public void ResetSession_ClearsCacheAndHistoryForTheGivenSession()
    {
        var svc = new ModelService();
        var session = new ChatSession();
        session.KVCache.RecordAppend(new[] { 1, 2, 3 }, new float[] { 0.5f });
        session.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "hi" });

        svc.ResetSession(session);

        Assert.True(session.KVCache.IsEmpty);
        Assert.Empty(session.TrackedHistory);
        Assert.False(session.IsDisposed);
    }

    [Fact]
    public void ResetSession_LeavesOtherSessionsUntouched()
    {
        var svc = new ModelService();
        var sessA = new ChatSession();
        var sessB = new ChatSession();

        sessA.KVCache.RecordAppend(new[] { 1, 2 }, new float[] { 0.1f });
        sessA.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "a" });
        sessB.KVCache.RecordAppend(new[] { 9, 8 }, new float[] { 0.2f });
        sessB.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "b" });

        svc.ResetSession(sessA);

        Assert.True(sessA.KVCache.IsEmpty);
        Assert.Empty(sessA.TrackedHistory);
        Assert.Equal(2, sessB.KVCache.Count);
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
        session.KVCache.RecordAppend(new[] { 10, 20 }, new float[] { 0.3f });
        session.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "x" });

        svc.DisposeSession(session);

        Assert.True(session.IsDisposed);
        Assert.True(session.KVCache.IsEmpty);
        Assert.Empty(session.TrackedHistory);
    }

    [Fact]
    public void DisposeSession_DoesNotAffectOtherSessions()
    {
        // Disposing session A must not release any bookkeeping attached to session B.
        var svc = new ModelService();
        var sessA = new ChatSession();
        var sessB = new ChatSession();

        sessB.KVCache.RecordAppend(new[] { 77 }, new float[] { 0.9f });
        sessB.TrackedHistory.Add(new ChatMessage { Role = "user", Content = "keep" });

        svc.DisposeSession(sessA);

        Assert.False(sessB.IsDisposed);
        Assert.Equal(1, sessB.KVCache.Count);
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
    public void ActiveSession_StartsNullUntilInferenceActivatesOne()
    {
        var svc = new ModelService();
        Assert.Null(svc.ActiveSession);
    }
}
