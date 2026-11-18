// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: where recorded steps bind slots (the CUDA executor), a step kept to advance is checked against the binding
// versions of the recurrent states it binds, not against one counter of every state's changes: a new conversation's first
// verify window (its states' verify buffers allocated) no longer drops the kept steps of the requests already running, which
// on CUDA disposed their recordings and recorded them again. A change of a running request's own states still drops its
// steps. The managed CPU executor runs the rule through the model's seam (SlotBoundPoliciesForTests), and keeps the global
// rule without it, as every other executor does; the CUDA test runs it on the device.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class RecurrentBindingStampTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-binding-stamp-" + Guid.NewGuid().ToString("N"));

    public RecurrentBindingStampTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private Qwen35RuntimeModel Create(BackendType backend = BackendType.Cpu, bool slotBound = true)
    {
        using var dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(backend == BackendType.Cpu ? KvCacheDtype.F32 : KvCacheDtype.F16);
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));
        var model = new Qwen35RuntimeModel(path, backend, 1, null, 1);
        if (backend == BackendType.Cpu) model.SlotBoundPoliciesForTests = slotBound;
        return model;
    }

    private static int ArgMax(ReadOnlySpan<float> row)
    {
        int best = 0;
        for (int i = 1; i < row.Length; i++)
            if (row[i] > row[best]) best = i;
        return best;
    }

    /// <summary>Requests that run together: each one's length and next token.</summary>
    private sealed class Running
    {
        public readonly string[] Ids;
        public readonly int[] Lengths, Next;
        public Running(params string[] ids)
        {
            Ids = ids;
            Lengths = new int[ids.Length];
            Next = new int[ids.Length];
        }
    }

    /// <summary>Each request's prompt (its own: the requests differ).</summary>
    private static void Prompt(Qwen35RuntimeModel model, Running running)
    {
        var unified = (IUnifiedBatchModel)model;
        for (int s = 0; s < running.Ids.Length; s++)
        {
            int[] prompt = Enumerable.Range(0, 16 + s).Select(i => 40 + (i * (s + 3)) % 17).ToArray();
            Assert.Equal(0, unified.EnsureSequenceSlot(running.Ids[s], 320));
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { running.Ids[s] }, new[] { prompt }, new[] { 0 }, logits));
            running.Lengths[s] = prompt.Length;
            running.Next[s] = ArgMax(logits[0]);
        }
    }

    /// <summary>One decode step of every request of <paramref name="running"/>, together.</summary>
    private static void Decode(Qwen35RuntimeModel model, Running running)
    {
        int n = running.Ids.Length;
        var logits = new float[n][];
        Assert.True(((IUnifiedBatchModel)model).TryForwardUnified(running.Ids, running.Next.Select(t => new[] { t }).ToArray(),
            (int[])running.Lengths.Clone(), logits));
        for (int s = 0; s < n; s++)
        {
            running.Next[s] = ArgMax(logits[s]);
            running.Lengths[s]++;
        }
    }

    /// <summary>A two-row verify window of request <paramref name="s"/>, keeping its first row.</summary>
    private static void Window(Qwen35RuntimeModel model, Running running, int s)
    {
        var batched = (IBatchedSpeculativeTarget)model;
        int vocab = model.Config.VocabSize;
        var logits = new float[2 * vocab];
        string[] id = { running.Ids[s] };
        int[] at = { running.Lengths[s] };
        Assert.True(batched.TryBatchedSpecVerifyLogits(id, at, 2, new[] { running.Next[s], running.Next[s] }, logits, null));
        batched.CommitBatchedSpecVerify(id, at, new[] { 1 });
        running.Next[s] = ArgMax(logits.AsSpan(0, vocab));
        running.Lengths[s]++;
    }

    /// <summary>
    /// Running requests decode as their kept step advanced (one request, and two together); a new request's first verify
    /// window allocates its states' verify buffers. Where recorded steps bind slots, the running requests' next step still
    /// advances; elsewhere (the global rule every other executor keeps) it is traced again. A verify window of a running
    /// request's own states then drops that request's kept step under either rule.
    /// </summary>
    [Theory]
    [InlineData(1, true)]
    [InlineData(2, true)]
    [InlineData(1, false)]
    public void ANewRequestsBufferChangeKeepsTheRunningRequestsKeptSteps(int runningCount, bool slotBound)
    {
        using Qwen35RuntimeModel model = Create(slotBound: slotBound);
        AssertBufferChangeKeepsRunningSteps(model, runningCount, expectAdvance: slotBound, recordings: false);
    }

    /// <summary>The same on direct CUDA, whose kept steps are recorded: the running requests' recordings replay after the new
    /// request's first window, none recorded again.</summary>
    [CudaTheory]
    [InlineData(1)]
    [InlineData(2)]
    public void ANewRequestsBufferChangeKeepsTheRunningRequestsRecordingsOnDirectCuda(int runningCount)
    {
        using Qwen35RuntimeModel model = Create(BackendType.Cuda);
        AssertBufferChangeKeepsRunningSteps(model, runningCount, expectAdvance: true, recordings: true);
    }

    private static void AssertBufferChangeKeepsRunningSteps(Qwen35RuntimeModel model, int runningCount, bool expectAdvance, bool recordings)
    {
        var running = new Running(Enumerable.Range(0, runningCount).Select(i => $"run{i}").ToArray());
        Prompt(model, running);
        for (int t = 0; t < 4; t++) Decode(model, running);
        int advances = model.DecodeAdvances;
        Decode(model, running);
        Assert.Equal(advances + 1, model.DecodeAdvances);   // the running requests' step advances

        // A new request: its prompt, then its first verify window (no plain decode step: that is the running step's shape).
        var newcomer = new Running("new");
        Prompt(model, newcomer);
        long version = StateVersions(model, "new");
        Window(model, newcomer, 0);
        Assert.True(StateVersions(model, "new") > version, "the new request's first window moved none of its states' buffers");

        advances = model.DecodeAdvances;
        int recorded = model.DecodeRecordings;
        Decode(model, running);
        if (expectAdvance)
        {
            Assert.True(advances + 1 == model.DecodeAdvances, "the running requests' kept step was dropped by another request's buffer change");
            if (recordings) Assert.Equal(recorded, model.DecodeRecordings);
        }
        else
        {
            Assert.Equal(advances, model.DecodeAdvances);
        }

        // A running request's own first window: its states' buffers move, and the running requests' kept step goes.
        Decode(model, running);
        Window(model, running, 0);
        advances = model.DecodeAdvances;
        Decode(model, running);
        Assert.Equal(advances, model.DecodeAdvances);
    }

    /// <summary>The binding versions of every state of a request's slot, summed.</summary>
    private static long StateVersions(Qwen35RuntimeModel model, string requestId)
        => model.SlotStatesForTests(requestId).SelectMany(l => l ?? Array.Empty<StandardDecoderModel.RecurrentState>())
            .Sum(s => s?.BindingVersion ?? 0);

    /// <summary>A state's binding version moves with each change of the buffers graphs bind (allocated, grown, freed) and with
    /// nothing else: a retention that keeps an in-place window's buffers leaves it.</summary>
    [Fact]
    public void AStatesBindingVersionMovesWithItsBuffersOnly()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        var executor = new TensorSharp.Models.Graph.ManagedGraphExecutor(allocator);
        using var state = new Qwen35RuntimeModel.DeviceGdnState(allocator, executor, _ => IntPtr.Zero,
            convRows: 3, channels: 48, vHeads: 2, dv: 8, dk: 8, backingRows: 8);
        using var other = new Qwen35RuntimeModel.DeviceGdnState(allocator, executor, _ => IntPtr.Zero,
            convRows: 3, channels: 48, vHeads: 2, dv: 8, dk: 8, backingRows: 8);
        Assert.Equal(0, state.BindingVersion);
        Assert.True(state.EnsureSpecBuffers(4));   // allocated
        Assert.Equal(1, state.BindingVersion);
        Assert.False(state.EnsureSpecBuffers(4));
        state.ReleaseSpecBuffers(keepInPlaceBuffers: true);   // an in-place window's buffers stay
        Assert.Equal(1, state.BindingVersion);
        Assert.True(state.EnsureSpecBuffers(16));  // grown, and a per-row buffer
        Assert.Equal(2, state.BindingVersion);
        state.ReleaseSpecBuffers(keepInPlaceBuffers: true);   // the per-row buffer goes
        Assert.Equal(3, state.BindingVersion);
        state.ReleaseSpecBuffers(keepInPlaceBuffers: false);
        Assert.Equal(4, state.BindingVersion);
        state.ReleaseSpecBuffers(keepInPlaceBuffers: false);  // nothing left
        Assert.Equal(4, state.BindingVersion);
        Assert.Equal(0, other.BindingVersion);
    }
}
