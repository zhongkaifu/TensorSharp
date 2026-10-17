// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// MiniMaxH3Model as an ILoraPluginHost, on the tiny fixture transformer (MiniMaxH3LoraFixtures)
// with empty companion files beside it (the constructor only resolves their paths):
// TS_LORAS at load (applied, skipped when foreign, refused as a load refusal), SetLoras
// swapping and removing sets and disposing the one it replaces, the denoiser handing a packed
// set's device copies back on a residency release, and an unload that arrives while SetLoras
// is loading a set. Packed sets use the process's pinned GGML backend (TS_TEST_GGML_BACKEND,
// GgmlCpu by default); direct sets use BackendType.Cpu.
using System.Reflection;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using Adapter = InferenceWeb.Tests.MiniMaxH3LoraFixtures.Adapter;

namespace InferenceWeb.Tests;

public sealed class MiniMaxH3TurboModelTests : IDisposable
{
    private const string W8 = MiniMaxH3LoraFixtures.Weights8, W4 = MiniMaxH3LoraFixtures.Weights4;

    private readonly MiniMaxH3LoraFixtures _fx = new();
    private readonly EnvScope _env = new();
    private readonly string _model;

    public MiniMaxH3TurboModelTests()
    {
        // The model reads these; a developer's shell must not leak into the fixture.
        foreach (string name in new[]
                 {
                     LoraCliFlags.EnvironmentVariable, "TS_VIDEO_TEXT_ENCODER", "TS_VIDEO_VAE", "TS_VIDEO_AUDIO_VAE",
                     "TS_H3_DIT_LAYERS",
                 })
            _env.Set(name, null);
        _model = _fx.WriteTransformer();
        WriteCompanions();
    }

    public void Dispose()
    {
        MiniMaxH3Model.LoraLoadedHookForTest = null;
        MiniMaxH3LoraSet.InvalidateHookForTest = null;
        _env.Dispose();
        _fx.Dispose();
    }

    /// <summary>The companions' preferred names, so the constructor never scans the temp folder.</summary>
    private void WriteCompanions()
    {
        foreach (string name in new[]
                 {
                     "qwen3vl_32b_minimax_h3-Q4_K_M.gguf", "minimax_h3_video_vae_fp16.safetensors",
                     "minimax_h3_audio_vae_fp32.safetensors",
                 })
            _fx.WriteText(name, "");
    }

    private string Plugin(string manifest = "turbo-8step.json", string weights = W8, int steps = 8, int seed = 1)
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4, seed: seed);
        _fx.WriteAdapter(adapter, weights);
        return _fx.WriteManifest(manifest, weights, steps: steps, name: $"Test Turbo {steps}");
    }

    private string Qwen() =>
        _fx.WriteText("qwen-plugin.json", "{ \"type\": \"qwen-image-2.1-lora\", \"name\": \"q\", \"weights\": \"q.safetensors\" }");

    private MiniMaxH3Model Create(BackendType backend, string path = null) => new(path ?? _model, backend);

    private void Publish(params LoraSpec[] specs) => _env.Set(LoraCliFlags.EnvironmentVariable, LoraCliFlags.ToJson(specs));

    private static readonly MiniMaxH3LoraTarget Q = MiniMaxH3LoraTarget.Q;

    // ---- TS_LORAS at load --------------------------------------------------------------------

    [Fact]
    public void TheConstructorAppliesTheTurboPlugInFromTsLoras_AndSkipsAnotherFamilysPlugIn()
    {
        string manifest = Plugin();
        Publish(new LoraSpec(Qwen()), new LoraSpec(manifest));
        using var model = Create(BackendType.Cpu);

        Assert.Equal(LoraCliFlags.MiniMaxH3PluginType, model.LoraPluginType);
        Assert.NotNull(model.Loras);
        Assert.True(model.Loras.IsDirect);
        var spec = Assert.Single(model.LoraSpecs);
        Assert.Equal(Path.Combine(_fx.Dir, W8), spec.Path);
        Assert.Equal(manifest, spec.ConfigPath);
        Assert.Equal(8, model.SamplingRecipe.Steps);
        var contract = ((IVideoGenerationModel)model).SamplingContract;
        Assert.Equal("minimax-h3-lora:turbo-8step", contract.Id);
        Assert.Equal((8, 6f, 3f, "linspace", "euler", 1f),
            (contract.Steps, contract.VideoShift, contract.AudioShift, contract.Schedule, contract.Sampler, contract.CfgScale));
    }

    [Fact]
    public void TheConstructorTakesTheHostsResolvedForm()
    {
        // What the CLI and the server publish: the weights, with the plug-in config as --lora-config.
        string manifest = Plugin();
        Publish(LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) }).ToArray());
        using var model = Create(BackendType.Cpu);
        Assert.Equal(Path.Combine(_fx.Dir, W8), Assert.Single(model.LoraSpecs).Path);
        Assert.Equal(8, model.SamplingRecipe.Steps);
    }

    [Fact]
    public void WithoutTsLoras_TheModelSamplesItsOwnSchedule()
    {
        using var model = Create(BackendType.Cpu);
        Assert.Null(model.Loras);
        Assert.Null(model.SamplingRecipe);
        Assert.Null(((IVideoGenerationModel)model).SamplingContract);
        Assert.Empty(model.LoraSpecs);
    }

    [Theory]
    [InlineData("missing")]
    [InlineData("two")]
    [InlineData("ref2va")]
    [InlineData("negative")]
    [InlineData("misfit")]
    public void APlugInTheModelCannotApply_IsALoadRefusal(string problem)
    {
        string manifest = Plugin();
        string model = _model;
        string expected;
        switch (problem)
        {
            case "missing":
                string missing = Path.Combine(_fx.Dir, "no-such-plugin.json");
                Publish(new LoraSpec(missing));
                expected = $"--lora file not found: {missing}";
                break;
            case "two":
                string other = Plugin("turbo-4step.json", W4, 4, seed: 2);
                Publish(new LoraSpec(manifest), new LoraSpec(other));
                expected = "Two MiniMax-H3 plug-ins (turbo-8step.json, turbo-4step.json) each fix the sampling schedule; use one.";
                break;
            case "ref2va":
                model = _fx.WriteTransformer(MiniMaxH3Partition.Reference);
                Publish(new LoraSpec(manifest));
                expected = $"LoRA plug-in turbo-8step.json is for the MiniMax-H3 FL2VA checkpoint; {MiniMaxH3LoraFixtures.Ref2vaModel} " +
                    "is the Ref2VA checkpoint, and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.";
                break;
            case "negative":
                Publish(new LoraSpec(manifest, -1f));
                expected = "Test Turbo 8 (" + W8 + "): strength -1 is negative; the Turbo schedule is distilled at strength 1 " +
                    "(0 keeps the adapter's terms with zero up factors).";
                break;
            default:
                // An adapter for a wider model: refused against this transformer at load.
                var wide = new Adapter();
                wide.Add(Adapter.Key(false, 0, Q, 'A'), [4, 12], new float[48]);
                wide.Add(Adapter.Key(false, 0, Q, 'B'), [MiniMaxH3LoraFixtures.Inner, 4], new float[4 * MiniMaxH3LoraFixtures.Inner]);
                _fx.WriteAdapter(wide, W8);
                Publish(new LoraSpec(manifest));
                expected = $"LoRA '{W8}' does not fit MiniMax-H3 (1 problem(s))";
                break;
        }

        var e = Assert.Throws<TensorSharp.Runtime.ModelLoadRefusedException>(() => Create(BackendType.Cpu, model));
        Assert.StartsWith("LoRA plug-in refused: ", e.Message);
        Assert.Contains(expected, e.Message);
    }

    // ---- SetLoras ------------------------------------------------------------------------------

    [Fact]
    public void SetLoras_SwapsAndDisposesTheReplacedSet_AnEmptyListRemovesIt_ARefusalKeepsIt()
    {
        string m8 = Plugin();
        string m4 = Plugin("turbo-4step.json", W4, 4, seed: 2);
        using var model = Create(BackendType.Cpu);

        model.SetLoras(new[] { new LoraSpec(m8) });
        var first = model.Loras;
        Assert.Equal(8, model.SamplingRecipe.Steps);

        model.SetLoras(new[] { new LoraSpec(m4) });
        var second = model.Loras;
        Assert.NotSame(first, second);
        Assert.Equal("minimax-h3-lora:turbo-4step", ((IVideoGenerationModel)model).SamplingContract.Id);
        Assert.Equal(Path.Combine(_fx.Dir, W4), Assert.Single(model.LoraSpecs).Path);
        Assert.Throws<ObjectDisposedException>(() => first.ReadFactors(false, 0, Q));

        // Bare weights and another family's plug-in are refused from their configs, and the set stays.
        string bare = Path.Combine(_fx.Dir, W8);
        Assert.Equal(
            $"--lora {W8}: MiniMax-H3 takes LoRA plug-ins only as a minimax-h3-lora config (for example " +
            "config/lora/minimax-h3-fl2v-turbo-8step.json), which declares the checkpoint partition and the sampling contract. " +
            "Pass that config instead of the bare weights.",
            Assert.Throws<ArgumentException>(() => model.SetLoras(new[] { new LoraSpec(bare) })).Message);
        Assert.Equal(
            $"--lora qwen-plugin.json is a qwen-image-2.1-lora plug-in; {MiniMaxH3LoraFixtures.Fl2vaModel} (minimax-h3) " +
            "takes minimax-h3-lora plug-ins only.",
            Assert.Throws<ArgumentException>(() => model.SetLoras(new[] { new LoraSpec(Qwen()) })).Message);
        Assert.Throws<ArgumentException>(() => model.SetLoras(new[] { new LoraSpec(m8), new LoraSpec(m4) }));
        Assert.Same(second, model.Loras);
        Assert.Equal(4, model.SamplingRecipe.Steps);
        Assert.NotNull(second.ReadFactors(false, 0, Q));

        model.SetLoras(Array.Empty<LoraSpec>());
        Assert.Null(model.Loras);
        Assert.Null(model.SamplingRecipe);
        Assert.Null(((IVideoGenerationModel)model).SamplingContract);
        Assert.Empty(model.LoraSpecs);
        Assert.Throws<ObjectDisposedException>(() => second.ReadFactors(false, 0, Q));
    }

    [Fact]
    public void ContractFor_ReadsTheConfigsAlone()
    {
        string m4 = Plugin("turbo-4step.json", W4, 4, seed: 2);
        using var model = Create(BackendType.Cpu);
        var contract = model.ContractFor(new[] { new LoraSpec(m4) });
        Assert.Equal(("minimax-h3-lora:turbo-4step", 4), (contract.Id, contract.Steps));
        Assert.Null(model.ContractFor(Array.Empty<LoraSpec>()));
        Assert.Null(model.Loras);
        Assert.Throws<ArgumentException>(() => model.ContractFor(new[] { new LoraSpec(Qwen()) }));
    }

    [Fact]
    public void AcceptsLora_TakesOnlyMiniMaxH3PlugIns()
    {
        string manifest = Plugin();
        using var model = Create(BackendType.Cpu);
        Assert.True(model.AcceptsLora(new LoraSpec(manifest)));
        Assert.True(model.AcceptsLora(new LoraSpec(Path.Combine(_fx.Dir, W8), null, manifest)));
        Assert.False(model.AcceptsLora(new LoraSpec(Path.Combine(_fx.Dir, W8))));
        Assert.False(model.AcceptsLora(new LoraSpec(Qwen())));
        Assert.Throws<FileNotFoundException>(() => model.AcceptsLora(new LoraSpec(Path.Combine(_fx.Dir, "gone.json"))));
    }

    [Fact]
    public async Task SetLoras_WaitsForAClipInProgress()
    {
        string manifest = Plugin();
        using var model = Create(BackendType.Cpu);
        object gate = typeof(MiniMaxH3Model).GetField("_generationLock", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(model)!;
        Task pending;
        // GenerateVideo holds this lock for the whole clip.
        lock (gate)
        {
            pending = Task.Run(() => model.SetLoras(new[] { new LoraSpec(manifest) }));
            Assert.False(SpinWait.SpinUntil(() => pending.IsCompleted, 200), "SetLoras swapped the plug-in under a running clip.");
            Assert.Null(model.Loras);
        }
        await pending.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.NotNull(model.Loras);
    }

    // ---- unload ------------------------------------------------------------------------------

    [Fact]
    public void AfterAnUnload_SetLorasAndGenerateRefuse()
    {
        string manifest = Plugin();
        var model = Create(BackendType.Cpu);
        model.SetLoras(new[] { new LoraSpec(manifest) });
        var set = model.Loras;
        model.Dispose();
        Assert.Throws<ObjectDisposedException>(() => set.ReadFactors(false, 0, Q));
        Assert.Null(model.Loras);
        Assert.Equal(MiniMaxH3Model.UnloadedText,
            Assert.Throws<ObjectDisposedException>(() => model.SetLoras(new[] { new LoraSpec(manifest) })).Message);
        Assert.Throws<ObjectDisposedException>(() => model.GenerateVideo("a cat"));
    }

    [Fact]
    public async Task AnUnloadWhileSetLorasLoads_DisposesTheNewSetOnce_AndTheOldOneOnce()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        string m8 = Plugin();
        string m4 = Plugin("turbo-4step.json", W4, 4, seed: 2);
        Publish(new LoraSpec(m8));
        var model = Create(backend);
        var old = model.Loras;
        Assert.NotNull(old);
        Assert.False(old.IsDirect);

        using var loaded = new ManualResetEventSlim();
        using var proceed = new ManualResetEventSlim();
        MiniMaxH3LoraSet incoming = null;
        MiniMaxH3Model.LoraLoadedHookForTest = set =>
        {
            incoming = set;
            loaded.Set();
            Assert.True(proceed.Wait(TimeSpan.FromSeconds(30)));
        };
        var setting = Task.Run(() => model.SetLoras(new[] { new LoraSpec(m4) }));
        Assert.True(loaded.Wait(TimeSpan.FromSeconds(30)), "SetLoras never finished loading the set.");

        var mine = TinyH3Step.FactorPointers(incoming);
        mine.UnionWith(TinyH3Step.FactorPointers(old));
        var calls = new Dictionary<IntPtr, int>();
        MiniMaxH3LoraSet.InvalidateHookForTest = p =>
        {
            if (!mine.Contains(p))
            {
                GgmlBasicOps.InvalidateHostBuffer(p);
                return;
            }
            lock (calls) calls[p] = calls.TryGetValue(p, out int n) ? n + 1 : 1;
        };

        // The unload marks the model, then waits for the lock SetLoras holds.
        var unloading = Task.Run(model.Dispose);
        Assert.False(SpinWait.SpinUntil(() => unloading.IsCompleted, 200), "Dispose ran while SetLoras held the model.");
        proceed.Set();

        var unloaded = await Assert.ThrowsAsync<ObjectDisposedException>(() => setting.WaitAsync(TimeSpan.FromSeconds(30)));
        Assert.Equal(MiniMaxH3Model.UnloadedText, unloaded.Message);
        await unloading.WaitAsync(TimeSpan.FromSeconds(30));

        // Both sets were disposed (each invalidates its factors once, before freeing them) and
        // nothing kept either one.
        Assert.Equal(IntPtr.Zero, incoming.NativeAdapter);
        Assert.Equal(IntPtr.Zero, old.NativeAdapter);
        Assert.Equal(mine.Count, calls.Count);
        Assert.All(calls, c => Assert.Equal(1, c.Value));
        Assert.Null(model.Loras);
        Assert.Empty(model.LoraSpecs);
        incoming.Dispose();
        old.Dispose();
        Assert.All(calls, c => Assert.Equal(1, c.Value));
    }

    // ---- the denoiser's hold on a packed set ---------------------------------------------------

    [Fact]
    public void TheDenoiserHandsAPackedSetsDeviceCopiesBack()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        string m8 = Plugin();
        string m4 = Plugin("turbo-4step.json", W4, 4, seed: 2);
        var cfg = MiniMaxH3LoraFixtures.ConfigOf(_model);
        using var first = MiniMaxH3LoraSet.Load(new[] { new LoraSpec(m8) }, cfg, MiniMaxH3Partition.FirstLastFrame, backend, direct: false);
        using var second = MiniMaxH3LoraSet.Load(new[] { new LoraSpec(m4) }, cfg, MiniMaxH3Partition.FirstLastFrame, backend, direct: false);
        var firstPointers = TinyH3Step.FactorPointers(first);
        var secondPointers = TinyH3Step.FactorPointers(second);
        var calls = new Dictionary<IntPtr, int>();
        MiniMaxH3LoraSet.InvalidateHookForTest = p =>
        {
            if (!firstPointers.Contains(p) && !secondPointers.Contains(p))
            {
                GgmlBasicOps.InvalidateHostBuffer(p);
                return;
            }
            calls[p] = calls.TryGetValue(p, out int n) ? n + 1 : 1;
        };
        int Count(HashSet<IntPtr> pointers) => pointers.Sum(p => calls.TryGetValue(p, out int n) ? n : 0);

        var dit = new MiniMaxH3DiT(_model, backend, null);
        try
        {
            TinyH3Step.Run(dit, first);
            TinyH3Step.Run(dit, first);
            Assert.Empty(calls);

            // A residency release drops the bound set's copies with the weights'.
            dit.ReleaseDeviceResidency();
            Assert.Equal(firstPointers.Count, Count(firstPointers));
            Assert.All(calls, c => Assert.Equal(1, c.Value));

            // Switching sets hands the previous one's copies back; so does running without one.
            calls.Clear();
            TinyH3Step.Run(dit, second);
            Assert.Equal(firstPointers.Count, Count(firstPointers));
            Assert.Equal(0, Count(secondPointers));
            calls.Clear();
            TinyH3Step.Run(dit, null);
            Assert.Equal(0, Count(firstPointers));
            Assert.Equal(secondPointers.Count, Count(secondPointers));

            // Dispose releases the set bound last.
            TinyH3Step.Run(dit, first);
            calls.Clear();
            dit.Dispose();
            Assert.Equal(firstPointers.Count, Count(firstPointers));
            Assert.Equal(0, Count(secondPointers));
        }
        finally
        {
            MiniMaxH3LoraSet.InvalidateHookForTest = null;
            dit.Dispose();
        }
    }
}
