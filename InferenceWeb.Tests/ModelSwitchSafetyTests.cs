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
using TensorSharp.GGML;
using static InferenceWeb.Tests.TranscriptTestHelper;

namespace InferenceWeb.Tests;

/// <summary>
/// What a model switch must not do to the model it replaces. Found in the Mac app by
/// switching models while a request was still using the old one, and by measuring the
/// footprint across switches: an image encode that runs on the request's thread kept
/// going while the model under it was disposed; the old model's raw token ids were handed
/// to the next model in the same chat; and every unload kept the old model's pooled host
/// memory for the life of the process.
/// </summary>
public sealed class ModelSwitchSafetyTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"ts-model-switch-{Guid.NewGuid():N}");

    public ModelSwitchSafetyTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    /// <summary>
    /// The unload waits for an encode that is inside the model, and the encode stops at
    /// its next yield of the GPU lock instead of running another block on a model that is
    /// being freed. The model is disposed only after the encoder has left it.
    /// </summary>
    [Fact]
    public void AnUnloadStopsAYieldingEncoderAndDisposesOnlyAfterItHasLeft()
    {
        var models = new List<FakeModel>();
        var lifecycle = new ModelLifecycleService(null, (path, backend, _, _) =>
        {
            var created = new FakeModel(path, backend);
            lock (models) models.Add(created);
            return created;
        });
        lifecycle.LoadModel(WriteMinimalGguf("a.gguf"), null, "cpu");
        FakeModel outgoing = models[0];

        using var inside = new ManualResetEventSlim();
        Exception? encoderStop = null;
        bool disposedUnderTheEncoder = false;
        bool disposedBeforeItLeft = false;
        int blocks = 0;
        var encoder = new Thread(() =>
        {
            Assert.True(outgoing.TryEnterUse());
            try
            {
                lock (outgoing.GpuComputeLock)
                {
                    inside.Set();
                    // An encoder's per-block loop: work, then let the engine in.
                    for (int block = 0; block < 2000; block++)
                    {
                        disposedUnderTheEncoder |= outgoing.Disposed;
                        blocks++;
                        Thread.Sleep(5);
                        outgoing.YieldGpuComputeLock();
                    }
                }
            }
            catch (ModelUnloadedException ex)
            {
                encoderStop = ex;
                // What an encoder does on its way out (its finally blocks free intermediates
                // into this model's pool): the model must still be there for it.
                Thread.Sleep(50);
                disposedBeforeItLeft = outgoing.Disposed;
            }
            finally
            {
                outgoing.ExitUse();
            }
        }) { IsBackground = true };
        encoder.Start();
        Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));

        lifecycle.LoadModel(WriteMinimalGguf("b.gguf"), null, "cpu");
        Assert.True(encoder.Join(TimeSpan.FromSeconds(10)));

        Assert.IsType<ModelUnloadedException>(encoderStop);
        Assert.False(disposedUnderTheEncoder);
        Assert.False(disposedBeforeItLeft, "the model was disposed while the stopped encoder was still unwinding");
        Assert.True(blocks < 2000, "the encoder ran to the end instead of stopping at a yield");
        Assert.True(outgoing.Disposed);
        Assert.Same(models[1], lifecycle.Model);
        Assert.False(models[1].IsRetiring);
    }

    /// <summary>
    /// A use that starts after the unload began is refused, and an encoder that does not
    /// yield is waited for: the dispose happens after its last block, never during it.
    /// </summary>
    [Fact]
    public void AnUnloadWaitsForAnEncoderThatNeverYieldsAndRefusesNewUses()
    {
        FakeModel? first = null;
        var lifecycle = new ModelLifecycleService(null, (path, backend, _, _) =>
        {
            var created = new FakeModel(path, backend);
            first ??= created;
            return created;
        });
        lifecycle.LoadModel(WriteMinimalGguf("a.gguf"), null, "cpu");

        using var inside = new ManualResetEventSlim();
        using var finish = new ManualResetEventSlim();
        bool disposedUnderTheEncoder = false;
        var encoder = new Thread(() =>
        {
            Assert.True(first!.TryEnterUse());
            try
            {
                lock (first.GpuComputeLock)
                {
                    inside.Set();
                    finish.Wait(TimeSpan.FromSeconds(30));
                    disposedUnderTheEncoder = first.Disposed;
                }
            }
            finally
            {
                first.ExitUse();
            }
        }) { IsBackground = true };
        encoder.Start();
        Assert.True(inside.Wait(TimeSpan.FromSeconds(10)));

        var loader = new Thread(() => lifecycle.LoadModel(WriteMinimalGguf("b.gguf"), null, "cpu")) { IsBackground = true };
        loader.Start();
        Assert.True(SpinWait.SpinUntil(() => first!.IsRetiring, TimeSpan.FromSeconds(10)));
        Assert.False(first!.TryEnterUse());
        Assert.Null(lifecycle.Model);              // unpublished while it drains
        Assert.False(loader.Join(TimeSpan.FromMilliseconds(300)));
        Assert.False(first.Disposed);

        finish.Set();
        Assert.True(encoder.Join(TimeSpan.FromSeconds(10)));
        Assert.True(loader.Join(TimeSpan.FromSeconds(10)));
        Assert.False(disposedUnderTheEncoder);
        Assert.True(first.Disposed);
    }

    /// <summary>
    /// The production encode path registers its use: a retiring model refuses to start an
    /// encode, and a finished or failed encode leaves no use behind (a leaked one would make
    /// every later unload of that model wait out the whole drain timeout).
    /// </summary>
    [Fact]
    public void TheMediaEncodeRegistersItsUseAndAlwaysReleasesIt()
    {
        var model = new FakeModel(WriteMinimalGguf("a.gguf"), BackendType.Cpu);
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hi" } };

        model.MultimodalInjector.ProcessPromptTokens(history, new List<int> { 1, 2, 3 });
        Assert.True(model.WaitForUsesToDrain(TimeSpan.Zero));

        var withImage = new List<ChatMessage>
        {
            new() { Role = "user", Content = "what is this", ImagePaths = new List<string> { Path.Combine(_dir, "missing.png") } },
        };
        Record.Exception(() => model.MultimodalInjector.ProcessPromptTokens(withImage, new List<int> { 1, 2, 3 }));
        Assert.True(model.WaitForUsesToDrain(TimeSpan.Zero));

        model.BeginRetirement();
        Assert.Throws<ModelUnloadedException>(() => model.MultimodalInjector.ProcessPromptTokens(history, new List<int> { 1, 2, 3 }));
        Assert.True(model.WaitForUsesToDrain(TimeSpan.Zero));
        model.Dispose();
    }

    /// <summary>
    /// A conversation's recorded raw tokens are spliced back only for the model load that
    /// produced them. After a switch the same chat stays open, and the old model's ids
    /// would otherwise reach the new model: garbage in range, an embedding read past the
    /// table out of it.
    /// </summary>
    [Fact]
    public void RecordedTokensAreNotSplicedIntoAnotherModelLoad()
    {
        var store = new ConversationTranscriptStore(maxChains: 64, maxTokens: 100_000);
        var raw = new List<int> { 248_069, 7, 8 };
        var question = new List<ChatMessage> { new() { Role = "user", Content = "Q1" } };
        store.Record(question, Generated("RAW1", raw), Emitted("PARSED1", rawText: "RAW1"), scope: "s", epoch: 3);

        var followUp = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
        };

        TranscriptAugmentation sameLoad = store.Augment(followUp, epoch: 3);
        Assert.Same(raw, sameLoad.History[1].RawOutputTokens);
        Assert.Equal("s", sameLoad.InheritedScope);

        TranscriptAugmentation nextLoad = store.Augment(followUp, epoch: 4);
        Assert.Null(nextLoad.History[1].RawOutputTokens);
        Assert.Equal("PARSED1", nextLoad.History[1].Content);
        Assert.Null(nextLoad.InheritedScope);
        Assert.Equal(0, nextLoad.SplicedTurns);
    }

    /// <summary>Every unload advances the epoch a request reads before it reads the model.</summary>
    [Fact]
    public void EachUnloadAdvancesTheLoadEpoch()
    {
        var lifecycle = new ModelLifecycleService(null, (path, backend, _, _) => new FakeModel(path, backend));
        long before = lifecycle.LoadEpoch;
        lifecycle.LoadModel(WriteMinimalGguf("a.gguf"), null, "cpu");
        Assert.Equal(before, lifecycle.LoadEpoch);

        lifecycle.LoadModel(WriteMinimalGguf("b.gguf"), null, "cpu");
        Assert.Equal(before + 1, lifecycle.LoadEpoch);

        lifecycle.Unload();
        Assert.Equal(before + 2, lifecycle.LoadEpoch);
    }

    /// <summary>
    /// A model's pool is released with the model, and keeps nothing freed afterwards.
    /// Before, nothing could reach it once the model was gone: whatever it held stayed
    /// mapped for the life of the process, on every switch.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void ADisposedModelsPoolReturnsItsBlocksAndKeepsNoneFreedLater()
    {
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);
        context.ReleasePooledMemory();   // the initial blocks, so what follows is the freed tensor
        var kept = new Tensor(allocator, DType.Float32, 1024, 1024);
        new Tensor(allocator, DType.Float32, 1024, 1024).Dispose();

        long released = context.ReleasePoolForDisposal();
        Assert.True(released >= 4L * 1024 * 1024, $"released only {released} bytes");

        // Freed after its model: unmapped at once rather than pooled where nothing can
        // ever allocate it again.
        kept.Dispose();
        Assert.Equal(0, context.ReleasePooledMemory());
    }

    /// <summary>
    /// Disposing a model gives its pool back: the model owns the context it created, and
    /// nothing else can ever reach that pool again.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void DisposingAModelReleasesThePoolItOwns()
    {
        var model = new FakeModel(WriteMinimalGguf("a.gguf"), BackendType.GgmlCpu);
        GgmlContext context = ((GgmlAllocator)model.Allocator).Context;
        new Tensor(model.Allocator, DType.Float32, 1024, 1024).Dispose();
        var kept = new Tensor(model.Allocator, DType.Float32, 1024, 1024);

        model.Dispose();

        // Nothing left pooled, and a tensor freed after its model is unmapped, not kept.
        Assert.Equal(0, context.ReleasePooledMemory());
        kept.Dispose();
        Assert.Equal(0, context.ReleasePooledMemory());
    }

    /// <summary>A pool that is still in use keeps its freed blocks, as before.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void ALivePoolStillKeepsFreedBlocksForReuse()
    {
        var context = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
        var allocator = new GgmlAllocator(context, 0);
        context.ReleasePooledMemory();
        new Tensor(allocator, DType.Float32, 1024, 1024).Dispose();
        Assert.True(context.ReleasePooledMemory() >= 4L * 1024 * 1024);
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

    private sealed class FakeModel : ModelBase
    {
        private volatile bool _disposed;

        public FakeModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
        }

        public bool Disposed => _disposed;

        public IAllocator Allocator => _allocator;

        protected override float[] ForwardCore(int[] tokens) => Array.Empty<float>();

        protected override void ResetKVCacheCore()
        {
        }

        public override void Dispose()
        {
            _disposed = true;
            base.Dispose();
        }
    }
}
