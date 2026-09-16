using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// Real cache allocation/copy/serialization code with tiny CPU tensors, without
/// loading weights or running inference. The allocator records storage lifetime;
/// injected failures are one-shot so cleanup and a subsequent retry can proceed.
/// This does not qualify GPU graph lifetime or model-level speculative accuracy.
/// </summary>
public class PrefixCheckpointOwnershipTests
{
    static PrefixCheckpointOwnershipTests()
    {
        // OpRegistry matches storage types exactly. Register only this private
        // subtype and delegate to the real CPU fill implementation; all normal
        // storage dispatch remains unchanged. This test assembly is serialized.
        var cpu = new CpuFillCopyOps();
        OpRegistry.Register("fill", args =>
        {
            cpu.Fill((Tensor)args[0]!, (float)args[1]!);
            return null;
        }, new OpConstraint[] { new ArgCountConstraint(2), new ArgStorageTypeConstraint(0, typeof(TrackedStorage), allowNull: false) });
    }

    public static IEnumerable<object[]> AllocationCases()
    {
        foreach (bool qwen in new[] { false, true })
            for (int allocation = 1; allocation <= 6; allocation++)
                yield return new object[] { qwen, allocation };
    }

    [Theory]
    [MemberData(nameof(AllocationCases))]
    public void CheckpointAllocationFailure_ReclaimsPartialCacheAndPreservesSource(bool qwen, int allocation)
    {
        using var f = new Fixture(qwen);
        byte[] before = f.State(f.Source);
        var live = f.Allocator.LiveSet();
        f.Allocator.FailAllocation = f.Allocator.Allocations + allocation;

        Assert.Throws<OutOfMemoryException>(() => f.Checkpoint("retry"));
        Assert.False(f.Retained.Contains("retry"));
        Assert.Equal(before, f.State(f.Source));
        f.Allocator.AssertLiveSet(live);

        Assert.True(f.Checkpoint("retry"));
        f.AssertIndependent(f.Source, f.Retained["retry"]!);
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, 3)]
    [InlineData(false, 5)]
    [InlineData(true, 0)]
    [InlineData(true, 3)]
    [InlineData(true, 5)]
    public void CloneReadFailure_ReclaimsCompletedAllocationAndPreservesRetainedOwner(bool qwen, int storage)
    {
        using var f = new Fixture(qwen);
        Assert.True(f.Checkpoint("source"));
        object retained = f.Retained["source"]!;
        byte[] before = f.State(retained);
        var live = f.Allocator.LiveSet();
        ((TrackedStorage)f.Tensors(retained)[storage].Storage).FailNextRead = true;

        Assert.Throws<IOException>(() => f.Clone("source", "retry"));
        Assert.False(f.Active.Contains("retry"));
        Assert.Same(retained, f.Retained["source"]);
        Assert.Equal(before, f.State(retained));
        f.Allocator.AssertLiveSet(live);

        Assert.True(f.Clone("source", "retry"));
        f.AssertIndependent(retained, f.Active["retry"]!);
    }

    [Theory]
    [MemberData(nameof(AllocationCases))]
    public void ImportAllocationFailure_PublishesNothingAndValidRetryKeepsSource(bool qwen, int allocation)
    {
        using var f = new Fixture(qwen);
        Assert.True(f.Checkpoint("source"));
        byte[] encoded = f.Export("source");
        byte[] before = f.State(f.Source);
        var live = f.Allocator.LiveSet();
        f.Allocator.FailAllocation = f.Allocator.Allocations + allocation;

        Assert.Throws<OutOfMemoryException>(() => f.Import("retry", encoded));
        Assert.False(f.Retained.Contains("retry"));
        Assert.Equal(before, f.State(f.Source));
        f.Allocator.AssertLiveSet(live);

        Assert.True(f.Import("retry", encoded));
        Assert.Equal(encoded, f.Export("retry"));
        f.AssertIndependent(f.Retained["source"]!, f.Retained["retry"]!);
    }

    [Theory]
    [InlineData(false, 1)]
    [InlineData(false, 73)]
    [InlineData(true, 1)]
    [InlineData(true, 73)]
    public void TruncatedImport_ReclaimsBuffersAfterPartialPayloadAndAllowsRetry(bool qwen, int missingBytes)
    {
        using var f = new Fixture(qwen);
        Assert.True(f.Checkpoint("source"));
        byte[] encoded = f.Export("source");
        byte[] before = f.State(f.Source);
        var live = f.Allocator.LiveSet();
        int allocations = f.Allocator.Allocations;

        Assert.False(f.Import("retry", encoded[..^missingBytes]));
        Assert.Equal(6, f.Allocator.Allocations - allocations);
        Assert.False(f.Retained.Contains("retry"));
        Assert.Equal(before, f.State(f.Source));
        f.Allocator.AssertLiveSet(live);

        Assert.True(f.Import("retry", encoded));
        Assert.Equal(encoded, f.Export("retry"));
    }

    [Fact]
    public void QwenCheckpointFailure_PreservesSpeculativeSnapshotAndIndependentCloneRollback()
    {
        using var f = new Fixture(qwen: true);
        var model = (Qwen35Model)f.Model;
        byte[] original = f.State(f.Source);
        model.SpecSnapshotRecurrentState();
        f.MutateRecurrent(f.Source);
        byte[] afterVerify = f.State(f.Source);
        Assert.NotEqual(original, afterVerify);
        var live = f.Allocator.LiveSet();
        f.Allocator.FailAllocation = f.Allocator.Allocations + 5;

        Assert.Throws<OutOfMemoryException>(() => f.Checkpoint("retry"));
        Assert.Equal(afterVerify, f.State(f.Source));
        f.Allocator.AssertLiveSet(live);
        model.SpecRestoreRecurrentState();
        Assert.Equal(original, f.State(f.Source));

        Assert.True(f.Checkpoint("retry"));
        Assert.True(f.Clone("retry", "branch"));
        object branch = f.Active["branch"]!;
        f.AssertIndependent(f.Source, branch);
        f.Load(branch);
        byte[] branchBefore = f.State(branch);
        model.SpecSnapshotRecurrentState();
        f.MutateRecurrent(branch);
        Assert.Equal(original, f.State(f.Source));
        byte[] branchAfterVerify = f.State(branch);
        byte[] encoded = f.Export("retry");
        live = f.Allocator.LiveSet();

        Assert.False(f.Import("failed-import", encoded[..^1]));
        Assert.Equal(branchAfterVerify, f.State(branch));
        f.Allocator.AssertLiveSet(live);
        model.SpecRestoreRecurrentState();
        Assert.Equal(branchBefore, f.State(branch));
        Assert.Equal(original, f.State(f.Source));
        Assert.True(f.Import("failed-import", encoded));
        Assert.Equal(encoded, f.Export("failed-import"));
    }

    [GemmaTraceTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void GemmaThrowingPostCommitTrace_DoesNotReportFailureOrLosePublishedOwner(bool clone)
    {
        using var f = new Fixture(qwen: false);
        Assert.True((bool)typeof(Gemma4Model).GetField("_cbDebug", BindingFlags.Static | BindingFlags.NonPublic)!.GetValue(null)!,
            "TS_CB_DEBUG=1 must be set before the testhost loads Gemma4Model.");
        if (clone) Assert.True(f.Checkpoint("source"));
        int live = f.Allocator.LiveSet().Length;
        byte[] before = f.State(f.Source);
        TextWriter saved = Console.Error;
        var throwing = new ThrowingTraceWriter();
        bool result;
        try
        {
            Console.SetError(throwing);
            result = clone ? f.Clone("source", "committed") : f.Checkpoint("committed");
        }
        finally { Console.SetError(saved); }

        Assert.Equal(1, throwing.Attempts);
        Assert.True(result);
        object committed = (clone ? f.Active : f.Retained)["committed"]!;
        Assert.NotNull(committed);
        Assert.Equal(live + 6, f.Allocator.LiveSet().Length);
        Assert.Equal(before, f.State(f.Source));
        f.AssertIndependent(f.Source, committed);
    }

    public sealed class GemmaTraceTheoryAttribute : TheoryAttribute
    {
        public GemmaTraceTheoryAttribute()
        {
            if (Environment.GetEnvironmentVariable("TS_CB_DEBUG") != "1")
                Skip = "Requires TS_CB_DEBUG=1 before testhost startup to exercise the actual post-commit trace.";
        }
    }

    private sealed class ThrowingTraceWriter : TextWriter
    {
        public override System.Text.Encoding Encoding => System.Text.Encoding.UTF8;
        internal int Attempts;
        public override void WriteLine(string? value)
        {
            Attempts++;
            throw new IOException("injected post-commit trace failure");
        }
    }

    private sealed class Fixture : IDisposable
    {
        internal readonly ModelBase Model;
        internal readonly CountingAllocator Allocator = new();
        internal readonly object Source;
        internal readonly IDictionary Retained;
        internal readonly IDictionary Active;
        private readonly Type modelType;
        private readonly bool qwen;
        private readonly List<object> owners = new();

        internal Fixture(bool qwen)
        {
            this.qwen = qwen;
            modelType = qwen ? typeof(Qwen35Model) : typeof(Gemma4Model);
            Model = (ModelBase)RuntimeHelpers.GetUninitializedObject(modelType);
            Set(typeof(ModelBase), Model, "<Config>k__BackingField", new ModelConfig
            {
                Architecture = qwen ? "qwen35" : "gemma4", NumLayers = 4,
                HiddenSize = 8, NumHeads = 2, NumKVHeads = 2,
            });
            // Public checkpoint APIs are enabled for the GGML CPU execution plan;
            // storage and all arithmetic here are actual .NET CPU allocations.
            Set(typeof(ModelBase), Model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(BackendType.GgmlCpu));
            Set(typeof(ModelBase), Model, "_backend", BackendType.GgmlCpu);
            Set(typeof(ModelBase), Model, "_allocator", Allocator);
            Set(typeof(ModelBase), Model, "_maxContextLength", 16);
            Set(typeof(ModelBase), Model, "_kvCacheDtype", KvCacheDtype.F32);
            Set(typeof(ModelBase), Model, "_cacheSeqLen", 9); // local Gemma ring has wrapped
            Type holderType = modelType.GetNestedType(qwen ? "Qwen35KvCacheHolder" : "Gemma4KvCacheHolder", BindingFlags.NonPublic)!;
            Retained = (IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), holderType))!;
            Active = (IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), holderType))!;
            Field("_retainedFusedHolders", Retained);
            Field("_fusedHolders", Active);
            if (qwen)
            {
                Field("_kvCacheK", new Tensor[4]);
                Field("_isRecurrent", new[] { false, true, false, true });
                Field("_headKDim", 2); Field("_headVDim", 2);
                Field("_numKHeads", 1); Field("_numVHeads", 2); Field("_convKernel", 4);
                Source = Invoke(Model, "AllocateHolder", 16)!;
                Set(Source.GetType(), Source, "CacheSeqLen", 9);
            }
            else
            {
                Field("_kvDonorMap", new Dictionary<int, int> { [3] = 1 });
                Field("_slidingWindowPattern", new[] { true, false, true, false });
                Field("_slidingWindow", 4); Field("_localHeadDim", 4);
                Field("_globalHeadDim", 4); Field("_numGlobalKVHeads", 2);
                Field("_initialGlobalCacheLength", 16);
                Source = Invoke(Model, "CreateFreshHolder")!;
                Set(Source.GetType(), Source, "SeqLen", 9);
            }
            owners.Add(Source);
            int storageIndex = 0;
            foreach (Tensor tensor in Tensors(Source))
            {
                float[] values = Enumerable.Range(0, checked((int)tensor.ElementCount()))
                    .Select(i => (float)(++storageIndex * 0.125 + i * 0.0625)).ToArray();
                tensor.SetElementsAsFloat(values);
            }
            if (qwen)
            {
                var conv = (float[][])Get(Source, "ConvState")!;
                var indexes = (int[])Get(Source, "ConvWriteIdx")!;
                for (int l = 0; l < conv.Length; l++)
                    if (conv[l] != null)
                    {
                        for (int i = 0; i < conv[l].Length; i++) conv[l][i] = l * 100 + i * .25f;
                        indexes[l] = l % 3;
                    }
            }
            Load(Source);
            Assert.Equal(6, Allocator.LiveSet().Length);
        }

        private void Field(string name, object value) => Set(modelType, Model, name, value);
        internal bool Checkpoint(string key) => (bool)Invoke(Model, "TryCheckpointActiveCache", key)!;
        internal bool Clone(string key, string request) => (bool)Invoke(Model, "TryCloneRetainedCache", key, request)!;
        internal bool Import(string key, byte[] bytes)
        {
            using var input = new MemoryStream(bytes, writable: false);
            return (bool)Invoke(Model, "TryImportRetainedCache", key, input)!;
        }
        internal byte[] Export(string key)
        {
            using var output = new MemoryStream();
            Assert.True((bool)Invoke(Model, "TryExportRetainedCache", key, output)!);
            return output.ToArray();
        }
        internal void Load(object holder) => Invoke(Model, "LoadCacheHolder", holder);
        internal Tensor[] Tensors(object holder)
        {
            var result = new List<Tensor>();
            foreach (string field in qwen ? new[] { "K", "V", "DeltaState" } : new[] { "K", "V" })
                foreach (Tensor? tensor in (Tensor[])Get(holder, field)!)
                    if (tensor != null && !result.Contains(tensor)) result.Add(tensor);
            return result.ToArray();
        }
        internal byte[] State(object holder, bool liveRowsOnly = false)
        {
            using var output = new MemoryStream();
            using var writer = new BinaryWriter(output);
            writer.Write((int)Get(holder, qwen ? "CacheSeqLen" : "SeqLen")!);
            foreach (Tensor tensor in Tensors(holder))
            {
                if (liveRowsOnly && tensor.Sizes.Length == 3)
                {
                    // Global padding is deliberately zero in a copy. Compare
                    // every written row of every head; local rings and these
                    // tiny recurrent tensors are smaller than the live head.
                    long capacity = tensor.Sizes[1];
                    long rowBytes = tensor.Storage.ByteLength / (tensor.Sizes[0] * capacity);
                    byte[] row = new byte[checked((int)(Math.Min(9, capacity) * rowBytes))];
                    for (long head = 0; head < tensor.Sizes[0]; head++)
                    {
                        Marshal.Copy(new IntPtr(tensor.Storage.PtrAtElement(0).ToInt64() + head * capacity * rowBytes), row, 0, row.Length);
                        writer.Write(row);
                    }
                }
                else
                {
                    byte[] bytes = new byte[checked((int)tensor.Storage.ByteLength)];
                    Marshal.Copy(tensor.Storage.PtrAtElement(0), bytes, 0, bytes.Length);
                    writer.Write(bytes);
                }
            }
            if (qwen)
            {
                foreach (float[]? row in (float[][])Get(holder, "ConvState")!)
                    if (row != null) foreach (float value in row) writer.Write(value);
                foreach (int index in (int[])Get(holder, "ConvWriteIdx")!) writer.Write(index);
            }
            return output.ToArray();
        }
        internal void AssertIndependent(object source, object copy)
        {
            Assert.Equal(State(source, liveRowsOnly: true), State(copy, liveRowsOnly: true));
            var src = Tensors(source); var dst = Tensors(copy);
            Assert.Equal(src.Length, dst.Length);
            for (int i = 0; i < src.Length; i++)
            {
                Assert.NotSame(src[i], dst[i]);
                Assert.NotEqual(src[i].Storage.PtrAtElement(0), dst[i].Storage.PtrAtElement(0));
            }
            if (!qwen)
            {
                var k = (Tensor[])Get(copy, "K")!;
                var v = (Tensor[])Get(copy, "V")!;
                Assert.Same(k[1], k[3]); Assert.Same(v[1], v[3]);
            }
            else
            {
                var from = (float[][])Get(source, "ConvState")!;
                var to = (float[][])Get(copy, "ConvState")!;
                for (int l = 0; l < from.Length; l++) if (from[l] != null) Assert.NotSame(from[l], to[l]);
                Assert.NotEqual((IntPtr)Get(source, "ConvScratch")!, (IntPtr)Get(copy, "ConvScratch")!);
            }
            byte[] before = State(source);
            dst[0].SetElementAsFloat(-9876.5f, 0, 0, 0);
            Assert.Equal(before, State(source));
        }
        internal void MutateRecurrent(object holder)
        {
            var conv = (float[][])Get(holder, "ConvState")!;
            var indexes = (int[])Get(holder, "ConvWriteIdx")!;
            var delta = (Tensor[])Get(holder, "DeltaState")!;
            for (int l = 0; l < conv.Length; l++) if (conv[l] != null)
            {
                conv[l][0] += 700;
                indexes[l] = (indexes[l] + 1) % 3;
                delta[l].Storage.SetElementAsFloat(0, delta[l].Storage.GetElementAsFloat(0) + 900);
            }
        }
        public void Dispose()
        {
            Allocator.FailAllocation = -1;
            foreach (var storage in Allocator.All) storage.FailNextRead = false;
            foreach (DictionaryEntry entry in Retained) if (!owners.Contains(entry.Value!)) owners.Add(entry.Value!);
            foreach (DictionaryEntry entry in Active) if (!owners.Contains(entry.Value!)) owners.Add(entry.Value!);
            foreach (object holder in owners) Invoke(Model, "DisposeHolder", holder);
            // On the preserved red implementation unpublished holders are leaked.
            // Clean their recorded storage only AFTER the lifetime assertion fails.
            foreach (TrackedStorage storage in Allocator.All) if (!storage.Freed) storage.Release();
            GC.SuppressFinalize(Model); // the GGUF constructor was deliberately skipped
        }
    }

    private sealed class CountingAllocator : IAllocator
    {
        public BlasEnum BlasEnum => BlasEnum.DotNet;
        public int DeviceId => 0;
        public float GetAllocatedMemoryRatio() => 0;
        internal readonly List<TrackedStorage> All = new();
        internal int Allocations;
        internal int FailAllocation = -1;
        public Storage Allocate(DType type, long count)
        {
            if (++Allocations == FailAllocation) throw new OutOfMemoryException("injected cache allocation failure");
            var storage = new TrackedStorage(this, type, count);
            All.Add(storage);
            return storage;
        }
        internal TrackedStorage[] LiveSet() => All.Where(s => !s.Freed).ToArray();
        internal void AssertLiveSet(TrackedStorage[] before) => Assert.Equal(before, LiveSet());
    }
    private sealed class TrackedStorage : CpuStorage
    {
        internal bool Freed;
        internal bool FailNextRead;
        internal TrackedStorage(IAllocator allocator, DType type, long count) : base(allocator, type, count) { }
        public override void EnsureHostReadable()
        {
            if (FailNextRead)
            {
                FailNextRead = false;
                throw new IOException("injected source cache read failure");
            }
        }
        protected override void Destroy()
        {
            base.Destroy();
            Freed = true;
        }
    }
    private static object? Get(object target, string field) => target.GetType().GetField(field,
        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.GetValue(target);
    private static void Set(Type type, object target, string field, object value) => type.GetField(field,
        BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.SetValue(target, value);
    private static object? Invoke(object target, string method, params object[] args)
    {
        try
        {
            return target.GetType().GetMethod(method,
                BindingFlags.Instance | BindingFlags.Public | BindingFlags.NonPublic)!.Invoke(target, args);
        }
        catch (TargetInvocationException e) when (e.InnerException != null)
        {
            ExceptionDispatchInfo.Capture(e.InnerException).Throw();
            throw;
        }
    }
}
