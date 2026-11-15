using System.Collections.Generic;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// A runtime model reload that fails must not leave the server modelless: bad
/// files are rejected before the current model is unloaded, and a failure
/// during the load itself restores the previous model.
/// </summary>
public class ModelReloadRollbackTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public ModelReloadRollbackTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"ts-reload-rollback-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    [Fact]
    public void GenericDraftModelEnvironment_ReachesFactoryTimeDrafterLoad()
    {
        string modelPath = WriteMinimalGguf("model.gguf");
        string draftPath = WriteMinimalGguf("draft.gguf");
        _env.Set(SpeculationEnvVars.DraftModel, draftPath);
        var factory = new ScriptedFactory();
        factory.Enqueue(path => new FakeModel(path));

        using var svc = NewService(factory);
        svc.LoadModel(modelPath, null, "cpu");

        Assert.Equal(draftPath, Assert.Single(factory.DraftPaths));
    }

    [Fact]
    public void MissingFile_IsRejectedWithCurrentModelStillLoaded()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var modelA = (FakeModel)svc.Model;

        Assert.Throws<FileNotFoundException>(
            () => svc.LoadModel(Path.Combine(_dir, "missing.gguf"), null, "cpu"));

        Assert.True(svc.IsLoaded);
        Assert.Same(modelA, svc.Model);
        Assert.False(modelA.Disposed);
        Assert.Single(factory.Calls);
    }

    [Fact]
    public void NonGgufFile_IsRejectedWithCurrentModelStillLoaded()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var modelA = (FakeModel)svc.Model;

        string junk = Path.Combine(_dir, "junk.gguf");
        File.WriteAllText(junk, "not a gguf file at all");
        Assert.Throws<InvalidDataException>(() => svc.LoadModel(junk, null, "cpu"));

        Assert.Same(modelA, svc.Model);
        Assert.False(modelA.Disposed);
        Assert.Single(factory.Calls);
    }

    [Fact]
    public void TruncatedGguf_IsRejectedWithCurrentModelStillLoaded()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var modelA = (FakeModel)svc.Model;

        string truncated = WriteTruncatedGguf("truncated.gguf");
        Assert.Throws<IOException>(() => svc.LoadModel(truncated, null, "cpu"));

        Assert.Same(modelA, svc.Model);
        Assert.False(modelA.Disposed);
        Assert.Single(factory.Calls);
    }

    [Fact]
    public void TruncatedMmProj_IsRejectedWithCurrentModelStillLoaded()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var modelA = (FakeModel)svc.Model;

        string pathB = WriteMinimalGguf("model-b.gguf");
        string badMmProj = WriteTruncatedGguf("mmproj-bad.gguf");
        Assert.Throws<IOException>(() => svc.LoadModel(pathB, badMmProj, "cpu"));

        Assert.Same(modelA, svc.Model);
        Assert.False(modelA.Disposed);
        Assert.Single(factory.Calls);
    }

    [Fact]
    public void FailedLoad_RestoresPreviousModel()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        string pathB = WriteMinimalGguf("model-b.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var firstModelA = (FakeModel)svc.Model;

        factory.Enqueue(_ => throw new InvalidOperationException("backend init failed"));
        factory.Enqueue(p => new FakeModel(p));
        var thrown = Assert.Throws<InvalidOperationException>(() => svc.LoadModel(pathB, null, "cpu"));
        Assert.Equal("backend init failed", thrown.Message);

        Assert.True(svc.IsLoaded);
        Assert.Equal("model-a.gguf", svc.LoadedModelName);
        Assert.True(firstModelA.Disposed);
        Assert.NotSame(firstModelA, svc.Model);
        Assert.Equal(3, factory.Calls.Count);
        Assert.Equal((pathA, BackendType.Cpu), factory.Calls[2]);
    }

    [Fact]
    public void FailedLoadAndFailedRollback_LeavesNoModel()
    {
        var factory = new ScriptedFactory();
        var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        string pathB = WriteMinimalGguf("model-b.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");

        factory.Enqueue(_ => throw new InvalidOperationException("load failed"));
        factory.Enqueue(_ => throw new OutOfMemoryException("rollback failed"));
        var thrown = Assert.Throws<InvalidOperationException>(() => svc.LoadModel(pathB, null, "cpu"));
        Assert.Equal("load failed", thrown.Message);

        Assert.False(svc.IsLoaded);
        Assert.Null(svc.Model);
        Assert.Null(svc.LoadedModelName);
        Assert.Equal(3, factory.Calls.Count);
    }

    [Fact]
    public void FailedFirstLoad_DoesNotAttemptRollback()
    {
        var factory = new ScriptedFactory();
        var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");

        factory.Enqueue(_ => throw new InvalidOperationException("load failed"));
        Assert.Throws<InvalidOperationException>(() => svc.LoadModel(pathA, null, "cpu"));

        Assert.False(svc.IsLoaded);
        Assert.Single(factory.Calls);
    }

    [Fact]
    public void SuccessfulReload_SwapsAndDisposesPreviousModel()
    {
        var factory = new ScriptedFactory();
        using var svc = NewService(factory);
        string pathA = WriteMinimalGguf("model-a.gguf");
        string pathB = WriteMinimalGguf("model-b.gguf");
        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathA, null, "cpu");
        var modelA = (FakeModel)svc.Model;

        factory.Enqueue(p => new FakeModel(p));
        svc.LoadModel(pathB, null, "cpu");

        Assert.True(modelA.Disposed);
        Assert.Equal("model-b.gguf", svc.LoadedModelName);
        Assert.Equal(2, factory.Calls.Count);
    }

    private static ModelLifecycleService NewService(ScriptedFactory factory)
    {
        return new ModelLifecycleService(NullLogger.Instance, factory.Create);
    }

    private string WriteMinimalGguf(string name)
    {
        // Smallest file GgufFile accepts: header with zero tensors and zero
        // metadata entries, padded out to the 32-byte data alignment.
        string path = Path.Combine(_dir, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version
        bw.Write(0UL);         // tensor count
        bw.Write(0UL);         // metadata count
        bw.Write(new byte[8]);
        return path;
    }

    private string WriteTruncatedGguf(string name)
    {
        // The tensor table declares one F32 tensor of 1024 elements but the
        // file ends right after the header, so ThrowIfTruncated fires.
        string path = Path.Combine(_dir, name);
        using var bw = new BinaryWriter(File.Create(path));
        bw.Write(0x46554747u); // "GGUF"
        bw.Write(3u);          // version
        bw.Write(1UL);         // tensor count
        bw.Write(0UL);         // metadata count
        bw.Write(1UL);         // tensor name length
        bw.Write((byte)'t');
        bw.Write(1u);          // dims
        bw.Write(1024UL);      // shape[0]
        bw.Write(0u);          // GgmlTensorType.F32
        bw.Write(0UL);         // data offset
        return path;
    }

    private sealed class ScriptedFactory
    {
        public readonly List<(string Path, BackendType Backend)> Calls = new();
        public readonly List<string> DraftPaths = new();
        private readonly Queue<Func<string, ModelBase>> _steps = new();

        public void Enqueue(Func<string, ModelBase> step) => _steps.Enqueue(step);

        public ModelBase Create(string path, BackendType backend, ITensorParallelGroup tpGroup, string draftPath)
        {
            Calls.Add((path, backend));
            DraftPaths.Add(draftPath);
            return _steps.Dequeue()(path);
        }
    }

    private sealed class FakeModel : ModelBase
    {
        public bool Disposed;

        public FakeModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu)
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

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
