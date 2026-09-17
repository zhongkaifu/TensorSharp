// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// M0a (radix design DEC-37): the engine is bound to a model OBJECT, not only to a
// fingerprint. A fingerprint names a cache shape, so a reload of the same checkpoint -
// or a different checkpoint with the same geometry - reports the same string. Before
// this, InferenceEngineHost.TryGetEngine handed back the engine built on the previous
// (disposed) model object whenever the fingerprints matched, so its scheduler, block
// pool and executor kept driving a model that no longer existed, and every piece of
// reuse state (live cache length, retained holders, pooled block hashes) still
// described the old one.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class InferenceEngineHostRebuildTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-engine-host-" + Guid.NewGuid().ToString("N"));

    public InferenceEngineHostRebuildTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void SameFingerprint_DifferentModelObject_BuildsANewEngineOnTheNewModel()
    {
        var created = new List<SnapshotFakeModel>();
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) =>
            {
                var model = new SnapshotFakeModel(path, "same-shape");
                created.Add(model);
                return model;
            });
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance)
        {
            SchedulerConfigOverride = SmallConfig(),
        };

        lifecycle.LoadModel(WriteMinimalGguf("a.gguf"), null, "cpu");
        InferenceEngine first = host.TryGetEngine();
        Assert.NotNull(first);
        Assert.Same(created[0], first.Model);
        Assert.Equal(PrefixCacheMode.Tree, first.PrefixCacheMode);
        Assert.Equal(1, created[0].PrefixCacheAttachments);
        // Unchanged model: the standing engine is reused.
        Assert.Same(first, host.TryGetEngine());

        // Reload under the SAME fingerprint (here: a second checkpoint of the same
        // shape; the server does the same for a reload of the same file).
        lifecycle.LoadModel(WriteMinimalGguf("b.gguf"), null, "cpu");
        Assert.Equal(2, created.Count);
        Assert.True(created[0].Disposed);
        Assert.Equal(created[0].KVStateFingerprint, created[1].KVStateFingerprint);

        InferenceEngine second = host.TryGetEngine();
        Assert.NotNull(second);
        Assert.NotSame(first, second);
        Assert.Same(created[1], second.Model);
        Assert.Same(second, host.TryGetEngine());
        Assert.Equal(PrefixCacheMode.Tree, second.PrefixCacheMode);
        Assert.Equal(1, created[1].PrefixCacheAttachments);
    }

    [Fact]
    public void DifferentFingerprint_SameModelObject_StillRebuilds()
    {
        // The pre-existing rule stays: a fingerprint change alone rebuilds.
        SnapshotFakeModel model = null;
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance,
            (path, backend, tp, draft) => model = new SnapshotFakeModel(path, "shape-1"));
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance)
        {
            SchedulerConfigOverride = SmallConfig(),
        };

        lifecycle.LoadModel(WriteMinimalGguf("a.gguf"), null, "cpu");
        InferenceEngine first = host.TryGetEngine();
        model.Fingerprint = "shape-2";
        InferenceEngine second = host.TryGetEngine();

        Assert.NotSame(first, second);
        Assert.Same(model, second.Model);
    }

    private static SchedulerConfig SmallConfig() => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 2,
        MaxPrefillChunkSize = 32,
        NumBlocks = 8,
        BlockSize = 16,
        EnablePrefixCaching = true,
    };

    private string WriteMinimalGguf(string name)
    {
        // Smallest file GgufFile accepts: header with zero tensors and zero metadata
        // entries, padded out to the 32-byte data alignment.
        string path = Path.Combine(_dir, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version
        bw.Write(0UL);         // tensor count
        bw.Write(0UL);         // metadata count
        bw.Write(new byte[8]);
        return path;
    }

    /// <summary>A snapshot-capable model with no weights: enough for the host to build
    /// an engine on it, never enough to run a request.</summary>
    private sealed class SnapshotFakeModel : ModelBase, IPageOnlyPrefixCacheModel
    {
        public bool Disposed;
        public string Fingerprint;
        public int PrefixCacheAttachments;

        public SnapshotFakeModel(string ggufPath, string fingerprint)
            : base(ggufPath, BackendType.Cpu)
        {
            Fingerprint = fingerprint;
            Config = new ModelConfig { Architecture = "snapshot-fake", NumLayers = 1, VocabSize = 8 };
        }

        public override bool SupportsKVStateSnapshot => true;
        public override string KVStateFingerprint => Fingerprint;
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public PrefixCacheCapabilities GetPrefixCacheCapabilities()
            => PageFamilyCapabilities(FamilyClass.P, TruncationKind.Any);
        public void AttachPrefixCache(IPrefixPayloadSink sink) => PrefixCacheAttachments++;
        public long QuerySpareBytes(ResourceClass resourceClass) => -1;

        protected override float[] ForwardCore(int[] tokens) => new float[8];

        protected override void ResetKVCacheCore()
        {
        }

        public override void Dispose()
        {
            Disposed = true;
            base.Dispose();
        }
    }
}
