// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using TensorSharp;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

public sealed class SpeculativeDraftHeadLoaderTests : IDisposable
{
    private readonly string _dir;
    private readonly EnvScope _env = new();

    public SpeculativeDraftHeadLoaderTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), $"ts-draft-loader-{Guid.NewGuid():N}");
        Directory.CreateDirectory(_dir);
        _env.ClearSpeculationVars();
    }

    public void Dispose()
    {
        _env.Dispose();
        Directory.Delete(_dir, recursive: true);
    }

    [Theory]
    [InlineData(SpeculationEnvVars.DraftModel)]
    [InlineData(SpeculationEnvVars.LegacyDraftModel)]
    public void ConfiguredDraftHeadPath_TrimsEitherEnvironmentSpelling(string variable)
    {
        string path = Path.Combine(_dir, "draft model.gguf");
        _env.Set(variable, $"  {path}  ");

        Assert.Equal(path, SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath());
    }

    [Fact]
    public void TryAttachConfiguredDraftHead_FactoryLoadedBlockHeadIsAlreadyComplete()
    {
        string targetPath = WriteMinimalGguf("target.gguf");
        string draftPath = Path.Combine(_dir, "dspark.gguf");
        // Deliberately absent. Once the factory has made the head resident, the
        // attach-after-load phase must neither reopen nor reclassify its source
        // file (which may have been moved after loading).
        _env.Set(SpeculationEnvVars.DraftModel, draftPath);
        using var model = new FakeBlockModel(targetPath);

        bool attached = SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(
            model, out string error);

        Assert.True(attached);
        Assert.Null(error);
    }

    /// <summary>
    /// Qwen 3.5/3.8 and Muse-Glimmer decline a DFlash drafter in their
    /// constructors under --tp N (it borrows the sharded LM head and embedding).
    /// The attach-after-load phase then saw HasDFlash == false and loaded the
    /// drafter anyway, undoing that decline. It must refuse with a reason.
    /// </summary>
    [Fact]
    public void TryAttachConfiguredDraftHead_DeclinesADFlashDrafterUnderTensorParallelism()
    {
        string targetPath = WriteMinimalGguf("target.gguf");
        string draftPath = WriteArchitectureGguf("drafter-dflash.gguf", DFlashConfig.ArchName);
        _env.Set(SpeculationEnvVars.DraftModel, draftPath);
        using var model = new FakeTensorParallelModel(targetPath);

        bool attached = SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(model, out string error);

        Assert.False(attached);
        Assert.NotNull(error);
        Assert.Contains("drafter-dflash.gguf", error);
        Assert.Contains("not supported under tensor parallelism", error);
        // Declined before any load was attempted, not after one failed.
        Assert.DoesNotContain("Failed to load", error);
        Assert.False(model.HasDFlash);

        // The model's own decline: no other draft file could attach here, so the server's
        // fail-fast message must say to drop the flag (or --tp), never to find a "matching"
        // draft GGUF by its embedding_length_out.
        Assert.False(SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(model, out string again, out bool refusedByModel));
        Assert.Equal(error, again);
        Assert.True(refusedByModel);
        string fatal = TensorSharp.Server.Hosting.SpeculationStartupValidation.GetFatalActivationError(again, refusedByModel);
        Assert.Contains("run without --tp", fatal);
        Assert.Contains("drop --draft-model", fatal);
        Assert.DoesNotContain("embedding_length_out", fatal);
    }

    private string WriteArchitectureGguf(string name, string architecture)
    {
        string path = Path.Combine(_dir, name);
        using var writer = new BinaryWriter(File.Create(path));
        void WriteString(string value)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(value);
            writer.Write((ulong)bytes.Length);
            writer.Write(bytes);
        }
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write(0UL);         // tensor count
        writer.Write(1UL);         // metadata count
        WriteString("general.architecture");
        writer.Write(8u);          // GGUF string value
        WriteString(architecture);
        long pad = (32 - writer.BaseStream.Position % 32) % 32;
        writer.Write(new byte[pad]); // 32-byte alignment
        return path;
    }

    private string WriteMinimalGguf(string name)
    {
        string path = Path.Combine(_dir, name);
        using var writer = new BinaryWriter(File.Create(path));
        writer.Write(0x46554747u); // "GGUF"
        writer.Write(3u);          // version
        writer.Write(0UL);         // tensor count
        writer.Write(0UL);         // metadata count
        writer.Write(new byte[8]); // 32-byte alignment
        return path;
    }

    private sealed class FakeTensorParallelModel : ModelBase
    {
        public FakeTensorParallelModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu, 2, new StubTpGroup())
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }
    }

    /// <summary>A live two-rank group; nothing here issues a collective.</summary>
    private sealed class StubTpGroup : ITensorParallelGroup
    {
        public int Degree => 2;
        public bool IsActive => true;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() { }
        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Dispose() { }
    }

    private sealed class FakeBlockModel : ModelBase, IDraftHead
    {
        public FakeBlockModel(string ggufPath)
            : base(ggufPath, BackendType.Cpu)
        {
        }

        public DraftHeadKind DraftHeadKind => DraftHeadKind.Block;

        public void DraftCatchUp(int[] tokens, float[] hRows, int startPos)
        {
        }

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }
    }
}
