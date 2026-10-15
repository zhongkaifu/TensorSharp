// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

/// <summary>Exercise the real GLM model adapter over a fake physical executor.
/// These tests cover ownership and invalidation, not native device execution.</summary>
public sealed class GlmUnifiedRetainedPrefixTests
{
    private const BindingFlags Private = BindingFlags.Instance | BindingFlags.NonPublic;
    private static void Set(GlmDsaModel model, string field, object? value)
        => typeof(GlmDsaModel).GetField(field, Private)!.SetValue(model, value);

    private static (GlmDsaModel Model, Executor Native, Sink Sink) Model()
    {
        var model = (GlmDsaModel)RuntimeHelpers.GetUninitializedObject(typeof(GlmDsaModel));
        var native = new Executor();
        var sink = new Sink();
        Set(model, "_nativeSync", new object()); Set(model, "_exec", native);
        Set(model, "_maxContextLength", 1024); Set(model, "_primarySlot", 0);
        Set(model, "_retainedSlots", new OrderedDictionary<string, SlotRetention.RetainedSlot>(StringComparer.Ordinal));
        Set(model, "_unifiedPrefixCapture", (Func<int, IDisposable?>)(_ => new Scope()));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Config))!.SetValue(model, new ModelConfig { Architecture = "glm5-next" });
        model.AttachPrefixCache(sink);
        return (model, native, sink);
    }

    private static void Retain(GlmDsaModel model, Executor native, string key, int slot, bool copy)
    {
        var retained = (OrderedDictionary<string, SlotRetention.RetainedSlot>)typeof(GlmDsaModel)
            .GetField("_retainedSlots", Private)!.GetValue(model)!;
        retained.Add(key, new(slot, 64, IsCopy: copy)); native.Heads.Add(slot, 64);
    }

    [Fact]
    public void RoutineTrimPreservesPublicCopyAndFreesOnlyTerminalOwner()
    {
        var (model, native, sink) = Model();
        Retain(model, native, "public", 1, copy: true); Retain(model, native, "terminal", 2, copy: false);
        model.TrimUnifiedRetainedSlots(terminalOnly: true);
        Assert.Equal(new[] { 2 }, native.FreeCalls);
        Assert.Equal(new[] { "public" }, model.RetainedPayloadKeys);
        Assert.True(model.CanMaterialize("public", 64, 64));
        Assert.Equal(Executor.Bytes, model.MeasureEndState("public").Bytes.NativeSlot);
        Assert.Equal(new[] { ("terminal", InvalidationReason.NativeSlotReclaimed) }, sink.Invalidated);
        Assert.Equal(64, native.Heads[1]); Assert.Equal(0, native.Active);
    }

    [Fact]
    public void RefusedTerminalFreeKeepsPhysicalOwnerLookupAndMetadata()
    {
        var (model, native, sink) = Model();
        Retain(model, native, "terminal", 1, copy: false); native.RefuseFree.Add(1);
        model.TrimUnifiedRetainedSlots(terminalOnly: true);
        Assert.Equal(new[] { "terminal" }, model.RetainedPayloadKeys);
        Assert.True(model.CanMaterialize("terminal", 64, 64));
        Assert.Equal(Executor.Bytes, model.MeasureEndState("terminal").Bytes.NativeSlot);
        Assert.Equal(64, native.Heads[1]); Assert.Empty(sink.Invalidated);
    }

    [Fact]
    public void EmergencyTrimIncludesCopiesButInvalidatesOnlySuccessfulPhysicalFrees()
    {
        var (model, native, sink) = Model();
        Retain(model, native, "refused-public", 1, copy: true); Retain(model, native, "terminal", 2, copy: false);
        Retain(model, native, "freed-public", 3, copy: true); native.RefuseFree.Add(1);
        model.TrimUnifiedRetainedSlots();
        Assert.Equal(new[] { 1, 2, 3 }, native.FreeCalls);
        Assert.Equal(new[] { "refused-public" }, model.RetainedPayloadKeys);
        Assert.True(model.CanMaterialize("refused-public", 64, 64));
        Assert.Equal(new[] { "terminal", "freed-public" }, sink.Invalidated.Select(entry => entry.Key));
        Assert.True(native.Heads.ContainsKey(1)); Assert.False(native.Heads.ContainsKey(2)); Assert.False(native.Heads.ContainsKey(3));
    }

    [Fact]
    public void LaterFreeExceptionDoesNotLoseEarlierSuccessfulInvalidation()
    {
        var (model, native, sink) = Model();
        Retain(model, native, "freed", 1, copy: false); Retain(model, native, "throws", 2, copy: false);
        native.ThrowFree.Add(2);
        Assert.Throws<InvalidOperationException>(() => model.TrimUnifiedRetainedSlots(terminalOnly: true));
        Assert.Equal(new[] { "freed" }, sink.Invalidated.Select(entry => entry.Key));
        Assert.Equal(new[] { "throws" }, model.RetainedPayloadKeys);
        Assert.True(model.CanMaterialize("throws", 64, 64)); Assert.Equal(64, native.Heads[2]);
    }

    [Fact]
    public void CapturePublishesActualBytesAndRemainsReusableAfterRoutineTrim()
    {
        var (model, native, sink) = Model();
        Assert.True(model.TryCaptureCopy("request", "public", out var footprint));
        Assert.Equal(Executor.Bytes, footprint.Bytes.NativeSlot);
        Assert.Equal(64, footprint.Tokens);
        Assert.Equal(new[] { (0, 1, 64) }, native.CopyCalls);
        Assert.Equal(0, native.Active); Assert.Equal(64, native.NPast);
        model.TrimUnifiedRetainedSlots(terminalOnly: true);
        Assert.True(model.CanMaterialize("public", 64, 64));
        Assert.Empty(native.FreeCalls); Assert.Empty(sink.Invalidated);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void CaptureRefusalDoesNotPublishOrChangeSourceAndDiagnosticsAreBounded(bool refuseAllocation)
    {
        var (model, native, sink) = Model();
        native.RefuseAllocation = refuseAllocation; native.RefuseCopy = !refuseAllocation;
        for (int attempt = 0; attempt < 12; ++attempt)
            Assert.False(model.TryCaptureCopy("request", "missing-" + attempt, out _));
        Assert.Empty(model.RetainedPayloadKeys); Assert.Empty(sink.Invalidated);
        Assert.Equal(0, native.Active); Assert.Equal(64, native.NPast); Assert.Single(native.Heads);
        Assert.Equal(refuseAllocation ? 0 : 12, native.FreeCalls.Count);
        Assert.Equal(8, (int)typeof(GlmDsaModel).GetField("_unifiedPrefixCaptureDiagnostics", Private)!.GetValue(model)!);
    }

    private sealed class Scope : IDisposable { public void Dispose() { } }
    private sealed class Sink : IPrefixPayloadSink
    {
        internal readonly List<(string Key, InvalidationReason Reason)> Invalidated = [];
        public void OnPayloadInvalidated(string key, InvalidationReason reason) => Invalidated.Add((key, reason));
    }
    private sealed class Executor : IGlmExecutor
    {
        internal const long Bytes = 706_281_472;
        internal readonly Dictionary<int, int> Heads = new() { [0] = 64 };
        internal readonly List<int> FreeCalls = [];
        internal readonly List<(int Source, int Destination, int Head)> CopyCalls = [];
        internal readonly HashSet<int> RefuseFree = [], ThrowFree = [];
        internal bool RefuseAllocation, RefuseCopy;
        internal int Active;
        private int _next = 1;
        public string Kind => "test";
        public int VocabSize => 8;
        public int ContextSize => 1024;
        public int NPast => Heads[Active];
        public bool HasDraftHead => false;
        public bool SupportsSlotCopy => true;
        public int SlotAlloc()
        {
            if (RefuseAllocation) return -1;
            int slot = _next++; Heads.Add(slot, 0); return slot;
        }
        public bool SlotCopyInto(int source, int destination, int expectedHead)
        {
            CopyCalls.Add((source, destination, expectedHead));
            if (RefuseCopy || Heads[source] != expectedHead) return false;
            Heads[destination] = expectedHead; return true;
        }
        public long SlotBytes(int slot) => Heads.ContainsKey(slot) ? Bytes : 0;
        public bool SlotFree(int slot)
        {
            FreeCalls.Add(slot);
            if (ThrowFree.Contains(slot)) throw new InvalidOperationException("injected physical free failure");
            return slot != Active && !RefuseFree.Contains(slot) && Heads.Remove(slot);
        }
        public bool SetActiveSlot(int slot) { if (!Heads.ContainsKey(slot)) return false; Active = slot; return true; }
        public bool ResetChecked() { Heads[Active] = 0; return true; }
        public bool Rewind(int nPast) => nPast == NPast;
        public bool SetReclaimableSlots(int[] slots) => true;
        public int TakeReclaimedSlots(int[] buffer) => 0;
        public void Dispose() { }
        public bool Forward(int[] tokens, float[] logitsOut) => throw new NotSupportedException();
        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits) => throw new NotSupportedException();
        public bool QueueVisionRows(float[] rows, int nRows, int index) => throw new NotSupportedException();
        public void ClearVisionRows() => throw new NotSupportedException();
        public bool SpecForward(int[] tokens, float[] hOut, float[] logitsOut, bool allLogitsRows) => throw new NotSupportedException();
        public bool DraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut) => throw new NotSupportedException();
        public bool DraftCatchUp(int[] tokens, float[] hRows, int startPos) => throw new NotSupportedException();
        public bool KdaStateCapture() => throw new NotSupportedException();
        public int KdaStateRestore() => throw new NotSupportedException();
    }
}
