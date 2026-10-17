// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using Adapter = InferenceWeb.Tests.MiniMaxH3LoraFixtures.Adapter;

namespace InferenceWeb.Tests;

/// <summary>
/// End-to-end tests of <see cref="MiniMaxH3LoraSet.Load"/> on synthetic files
/// (<see cref="MiniMaxH3LoraFixtures"/>): a tiny H3 transformer GGUF with two trunk blocks
/// and one token-refiner block, Diffusers-format adapters and their plug-in configs. The
/// packed factors are read back through the <see cref="H3DitAdapter"/> descriptors and the
/// reconstructed update (up · down) is compared with strength · alpha / rank · B · A in the
/// checkpoint's row order. Packing is managed code only: no GGML backend is constructed, so
/// the Metal, CUDA and Vulkan packing rules are checked in any lane.
/// </summary>
public sealed class MiniMaxH3LoraSetTests : IDisposable
{
    private const string W8 = MiniMaxH3LoraFixtures.Weights8, W4 = MiniMaxH3LoraFixtures.Weights4;
    private const string Manifest = "turbo-8step.json";
    private const int Hidden = MiniMaxH3LoraFixtures.Hidden, Inner = MiniMaxH3LoraFixtures.Inner, Ffn = MiniMaxH3LoraFixtures.Ffn;
    private const int F32 = 0, F16 = 1;

    private static readonly MiniMaxH3LoraTarget[] Targets =
    {
        MiniMaxH3LoraTarget.Q, MiniMaxH3LoraTarget.K, MiniMaxH3LoraTarget.V,
        MiniMaxH3LoraTarget.Out, MiniMaxH3LoraTarget.Fc1, MiniMaxH3LoraTarget.Fc2,
    };

    private readonly MiniMaxH3LoraFixtures _fx = new();
    private readonly MiniMaxH3Config _cfg;

    public MiniMaxH3LoraSetTests() => _cfg = MiniMaxH3LoraFixtures.ConfigOf(_fx.WriteTransformer());

    public void Dispose() => _fx.Dispose();

    // ---- helpers ------------------------------------------------------------------------

    /// <summary>Write the adapter and a plug-in config naming it; returns the config's path.</summary>
    private string Plugin(Adapter adapter, float? alpha = 8f, float? scale = 1f, string weights = W8)
    {
        _fx.WriteAdapter(adapter, weights);
        return _fx.WriteManifest(Manifest, weights, alpha: alpha, scale: scale);
    }

    private MiniMaxH3LoraSet Load(LoraSpec spec, BackendType backend = BackendType.GgmlCpu, bool direct = false,
        MiniMaxH3Partition partition = MiniMaxH3Partition.FirstLastFrame) =>
        MiniMaxH3LoraSet.Load(new[] { spec }, _cfg, partition, backend, direct);

    private MiniMaxH3LoraSet Load(string manifest, BackendType backend = BackendType.GgmlCpu, bool direct = false) =>
        Load(new LoraSpec(manifest), backend, direct);

    private static int Padded(int rank, int multiple) => (rank + multiple - 1) / multiple * multiple;

    private static H3DitAdapter AdapterOf(MiniMaxH3LoraSet set) => Marshal.PtrToStructure<H3DitAdapter>(set.NativeAdapter);

    private static H3BlockLora BlockOf(MiniMaxH3LoraSet set, bool refiner, int block)
    {
        var adapter = AdapterOf(set);
        return Marshal.PtrToStructure<H3BlockLora>((refiner ? adapter.Refiner : adapter.Blocks) + block * Marshal.SizeOf<H3BlockLora>());
    }

    private static H3Lora PartOf(H3BlockLora b, MiniMaxH3LoraTarget target) => target switch
    {
        MiniMaxH3LoraTarget.Q => b.Q,
        MiniMaxH3LoraTarget.K => b.K,
        MiniMaxH3LoraTarget.V => b.V,
        MiniMaxH3LoraTarget.Out => b.Out,
        MiniMaxH3LoraTarget.Fc1 => b.Fc1,
        _ => b.Fc2,
    };

    private static H3Lora PartOf(MiniMaxH3LoraSet set, bool refiner, int block, MiniMaxH3LoraTarget target) =>
        PartOf(BlockOf(set, refiner, block), target);

    private static float[] ReadValues(IntPtr p, long count, int type)
    {
        if (type == F16)
        {
            var raw = new short[count];
            Marshal.Copy(p, raw, 0, (int)count);
            return raw.Select(s => (float)BitConverter.Int16BitsToHalf(s)).ToArray();
        }
        var values = new float[count];
        Marshal.Copy(p, values, 0, (int)count);
        return values;
    }

    /// <summary>down as ggml [ne0 = in, ne1 = padded rank], read row-major [padded][in].</summary>
    private static float[] DownOf(H3Lora l) => ReadValues(l.Down, l.Rank * l.In, l.Type);

    /// <summary>up as ggml [ne0 = padded rank, ne1 = out], read row-major [out][padded].</summary>
    private static float[] UpOf(H3Lora l) => ReadValues(l.Up, l.Out * l.Rank, l.Type);

    /// <summary>up · down, as the native graph applies it, row-major [out][in].</summary>
    private static float[] Delta(H3Lora l) =>
        MiniMaxH3LoraFixtures.Product(UpOf(l), DownOf(l), (int)l.Out, l.Rank, (int)l.In);

    private static void AssertClose(float[] expected, float[] actual, string what, float relative)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = expected.Max(v => Math.Abs(v));
        float tolerance = relative * Math.Max(max, 1e-6f);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]} (tolerance {tolerance})");
    }

    private static void AssertEmpty(H3Lora l)
    {
        Assert.Equal(0, l.Rank);
        Assert.Equal(IntPtr.Zero, l.Down);
        Assert.Equal(IntPtr.Zero, l.Up);
    }

    private static void AssertBits(float[] expected, float[] actual, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                $"{what}[{i}]: expected {expected[i]:R}, got {actual[i]:R}");
    }

    private static float[] Scaled(float[] values, float scale) => values.Select(v => v * scale).ToArray();

    /// <summary>The distinct factor pointers of a packed set: one down and one up per term.</summary>
    private HashSet<IntPtr> FactorPointers(MiniMaxH3LoraSet set)
    {
        var pointers = new HashSet<IntPtr>();
        foreach (bool refiner in new[] { false, true })
            for (int b = 0; b < (refiner ? _cfg.TokenRefinerLayers : _cfg.NumLayers); b++)
                foreach (var target in Targets)
                {
                    var l = PartOf(set, refiner, b, target);
                    if (l.Rank == 0) continue;
                    pointers.Add(l.Down);
                    pointers.Add(l.Up);
                }
        return pointers;
    }

    // ---- the update ---------------------------------------------------------------------

    [Theory]
    [InlineData(BackendType.GgmlCpu, 1e-5f)]
    [InlineData(BackendType.GgmlMetal, 5e-3f)]
    public void EveryTarget_PacksTheScaledUpdate_InTheCheckpointsRowOrder(BackendType backend, float tolerance)
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4);
        using var set = Load(Plugin(adapter), backend);

        // alpha 8 at rank 4: 2 x B · A, fc1's halves swapped into [gate; value].
        foreach (var (refiner, block, target) in adapter.Targets)
        {
            var l = PartOf(set, refiner, block, target);
            var (input, output) = MiniMaxH3LoraFixtures.Shape(target);
            Assert.Equal(input, l.In);
            Assert.Equal(output, l.Out);
            AssertClose(adapter.Delta(refiner, block, target, 2f), Delta(l), $"{(refiner ? "refiner " : "")}block {block} {target}", tolerance);
        }
        Assert.Equal(18, set.TermCount);
        Assert.Equal(36, set.TensorCount);
    }

    [Fact]
    public void Load_ReportsTheSetAndItsDescriptor()
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4);
        string manifest = Plugin(adapter);
        using var set = Load(manifest);

        Assert.False(set.IsDirect);
        Assert.Equal(Path.Combine(_fx.Dir, W8), set.WeightsPath);
        Assert.Equal(MiniMaxH3Partition.FirstLastFrame, set.Partition);
        Assert.Equal(8, set.Recipe.Steps);
        Assert.Equal(1f, set.Strength);
        Assert.Equal(8f, set.Alpha);
        Assert.Equal(new[] { 4 }, set.Ranks);
        Assert.Equal(new[] { 2f }, set.Scales);
        // F32 on GgmlCpu, ranks padded to 16: (16 x in + out x 16) x 4 bytes per term.
        long perBlock = Targets.Sum(t => (16L * MiniMaxH3LoraFixtures.Shape(t).In + MiniMaxH3LoraFixtures.Shape(t).Out * 16L) * 4);
        Assert.Equal(3 * perBlock, set.FactorBytes);
        Assert.Equal(
            $"  [lora] {W8} (Test Turbo 8; {Manifest}): 18 low-rank update(s) in 2 transformer and 1 token-refiner block(s), " +
            "rank 4, alpha 8, strength 1, scale 2; F32 factors (GgmlCpu keeps F32), rank padded to a multiple of 16, 0.00 GB",
            set.Summary);

        var descriptor = AdapterOf(set);
        Assert.Equal(Marshal.SizeOf<H3DitAdapter>(), descriptor.StructBytes);
        Assert.Equal(32, descriptor.StructBytes);
        Assert.Equal(2, descriptor.NumBlocks);
        Assert.Equal(1, descriptor.NumRefinerBlocks);
        Assert.Equal(0, descriptor.Reserved);
        Assert.NotEqual(IntPtr.Zero, descriptor.Blocks);
        Assert.NotEqual(IntPtr.Zero, descriptor.Refiner);
    }

    [Fact]
    public void ProjectionsWithoutAnUpdate_HaveEmptyDescriptors()
    {
        var adapter = new Adapter();
        adapter.Term(false, 1, MiniMaxH3LoraTarget.K, 4, seed: 3);
        adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc2, 4, seed: 4);
        using var set = Load(Plugin(adapter));

        foreach (var target in Targets)
        {
            AssertEmpty(PartOf(set, false, 0, target));
            AssertEmpty(PartOf(set, true, 0, target));
            if (target is not (MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.Fc2)) AssertEmpty(PartOf(set, false, 1, target));
        }
        Assert.Equal(2, set.TermCount);
        Assert.Contains("2 low-rank update(s) in 1 transformer and 0 token-refiner block(s)", set.Summary);
    }

    [Theory]
    [InlineData(null, null, 1f)]
    [InlineData(null, .5f, .5f)]
    [InlineData(.25f, .5f, .25f)]
    [InlineData(1.5f, null, 1.5f)]
    public void Strength_TimesAlphaOverRank_IsFoldedIntoUp(float? specScale, float? manifestScale, float strength)
    {
        var adapter = new Adapter();
        var (a, b) = adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 16, seed: 5, dtype: "F32");
        using var set = Load(new LoraSpec(Plugin(adapter, scale: manifestScale), specScale), BackendType.Cpu, direct: true);

        float scale = strength * 8f / 16;
        Assert.Equal(strength, set.Strength);
        Assert.Equal(new[] { scale }, set.Scales);
        var factors = set.ReadFactors(false, 0, MiniMaxH3LoraTarget.Q)!.Value;
        Assert.Equal((16, (long)Hidden, (long)Inner), (factors.Rank, factors.In, factors.Out));
        AssertBits(a, factors.Down, "down");
        AssertBits(Scaled(b, scale), factors.Up, "up");
    }

    [Fact]
    public void AStrengthBeyondTheFloatRange_IsRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 1, seed: 5, dtype: "F32");
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(new LoraSpec(manifest, 1e38f)));
        Assert.Equal($"LoRA '{W8}': strength 1E+38 scales {Adapter.Key(false, 0, MiniMaxH3LoraTarget.Q, 'B')} beyond the float range.", e.Message);
    }

    [Theory]
    [InlineData("BF16")]
    [InlineData("F16")]
    [InlineData("F32")]
    public void EveryFactorDtype_ReadsExactly(string dtype)
    {
        var adapter = new Adapter();
        var (a, b) = adapter.Term(true, 0, MiniMaxH3LoraTarget.Out, 8, seed: 6, dtype: dtype);
        using var set = Load(Plugin(adapter), BackendType.Cpu, direct: true);

        var factors = set.ReadFactors(true, 0, MiniMaxH3LoraTarget.Out)!.Value;
        AssertBits(a, factors.Down, "down");
        AssertBits(Scaled(b, 1f), factors.Up, "up");
    }

    // ---- alpha and key_format -------------------------------------------------------------

    [Theory]
    [InlineData(8f, null, 8f)]
    [InlineData(null, 8f, 8f)]
    [InlineData(16f, 16f, 16f)]
    public void Alpha_ComesFromTheMetadataOrTheConfig(float? metadata, float? manifest, float alpha)
    {
        var adapter = new Adapter(alpha: metadata);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.V, 4, seed: 7);
        using var set = Load(Plugin(adapter, alpha: manifest), BackendType.Cpu, direct: true);

        Assert.Equal(alpha, set.Alpha);
        Assert.Equal(new[] { alpha / 4 }, set.Scales);
    }

    [Fact]
    public void Alpha_DeclaredNowhere_IsRefused()
    {
        var adapter = new Adapter(alpha: null);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.V, 4, seed: 7);
        string manifest = Plugin(adapter, alpha: null);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Equal($"{W8} declares no alpha (safetensors __metadata__ \"alpha\") and {Manifest} sets none; " +
                     "TensorSharp never assumes alpha = rank.", e.Message);
    }

    [Fact]
    public void Alpha_ConfigAndMetadataDisagree_IsRefused()
    {
        var adapter = new Adapter(alpha: 16f);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.V, 4, seed: 7);
        string manifest = Plugin(adapter, alpha: 8f);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Equal($"{Manifest} sets alpha 8, but {W8}'s metadata says 16.", e.Message);
    }

    [Theory]
    [InlineData("eight")]
    [InlineData("0")]
    [InlineData("-8")]
    [InlineData("NaN")]
    public void Alpha_MetadataNotAPositiveNumber_IsRefused(string raw)
    {
        var adapter = new Adapter();
        adapter.Metadata["alpha"] = raw;
        adapter.Term(false, 0, MiniMaxH3LoraTarget.V, 4, seed: 7);
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Equal($"{W8}: __metadata__ \"alpha\" is '{raw}', not a positive number.", e.Message);
    }

    [Theory]
    [InlineData("comfyui")]
    [InlineData("minimax-h3-comfyui")]
    public void KeyFormat_OtherThanDiffusers_IsRefused(string keyFormat)
    {
        var adapter = new Adapter(keyFormat: keyFormat);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 8);
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Equal($"{W8} has key_format '{keyFormat}'; {Manifest} expects 'minimax-h3-diffusers' " +
                     "(the ComfyUI-format *_comfyui_bf16.safetensors files are not supported).", e.Message);
    }

    [Fact]
    public void ComfyUiTwin_DeclaredByItsTargetFormat_IsRefused()
    {
        // The publisher's *_comfyui_bf16.safetensors carry target_format, not key_format.
        var adapter = new Adapter(keyFormat: null);
        adapter.Metadata["target_format"] = "ComfyUI generic LoRA";
        adapter.Add("diffusion_model.blocks.0.mlp.fc1.lora_B.weight", [2 * Ffn, 4], new float[2 * Ffn * 4], "BF16");
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.StartsWith($"{W8} has key_format 'ComfyUI generic LoRA'; {Manifest} expects 'minimax-h3-diffusers'", e.Message);
    }

    [Fact]
    public void KeyFormat_Undeclared_IsAcceptedOnTheKeysAlone()
    {
        var adapter = new Adapter(keyFormat: null);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 8, slot: null);
        using var set = Load(Plugin(adapter));
        Assert.Equal(1, set.TermCount);
    }

    // ---- accounting: everything used or refused -------------------------------------------

    [Fact]
    public void EveryProblem_IsReportedTogether()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 9);
        adapter.Add("transformer_blocks.0.attn.to_q.alpha", [], new[] { 8f });
        adapter.Add("transformer_blocks.0.adaln_proj.linear.lora_A.default.weight", [4, 4], new float[16]);
        adapter.Add("transformer_blocks.2.attn.to_k.lora_A.default.weight", [4, Hidden], new float[4 * Hidden]);
        adapter.Add(Adapter.Key(false, 1, MiniMaxH3LoraTarget.K, 'A'), [4, Hidden], new float[4 * Hidden]);
        adapter.Add(Adapter.Key(false, 1, MiniMaxH3LoraTarget.V, 'A'), [4, 7], new float[28]);
        adapter.Add(Adapter.Key(false, 1, MiniMaxH3LoraTarget.V, 'B'), [Inner, 4], new float[Inner * 4]);
        adapter.Add(Adapter.Key(false, 1, MiniMaxH3LoraTarget.Out, 'A'), [4, Inner], new float[4 * Inner], "F64");
        adapter.Add(Adapter.Key(false, 1, MiniMaxH3LoraTarget.Out, 'B'), [Hidden, 4], new float[Hidden * 4]);
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.StartsWith($"LoRA '{W8}' does not fit MiniMax-H3 (6 problem(s)): ", e.Message);
        Assert.Contains("transformer_blocks.0.attn.to_q.alpha: a per-module .alpha tensor", e.Message);
        Assert.Contains("transformer_blocks.0.adaln_proj.linear.lora_A.default.weight: 'adaln_proj.linear' (an AdaLN modulation parameter) is not a LoRA target", e.Message);
        Assert.Contains("transformer_blocks.2.attn.to_k.lora_A.default.weight: block 2 is out of range (the checkpoint has 2 transformer blocks)", e.Message);
        Assert.Contains("transformer_blocks.1.attn.to_k: has only lora_A; a term needs both factors", e.Message);
        Assert.Contains("transformer_blocks.1.attn.to_v: lora_A [4,7] / lora_B [16,4], but blocks.1.attn.qkv_proj rows [32, 48) is 8 -> 16 " +
                        "(lora_A [rank, in], lora_B [out, rank]; a LoRA made for a different model)", e.Message);
        Assert.Contains($"transformer_blocks.1.attn.to_out.0: {Adapter.Key(false, 1, MiniMaxH3LoraTarget.Out, 'A')} is F64; " +
                        "LoRA factors must be BF16, F16 or F32", e.Message);
        Assert.EndsWith(". Nothing is skipped silently: a partly applied LoRA would not be the adapter you asked for.", e.Message);
    }

    [Fact]
    public void ManyProblems_AreCountedBeyondTheOnesShown()
    {
        var adapter = new Adapter();
        for (int i = 0; i < 9; i++)
            adapter.Add($"transformer_blocks.0.norm{i}.lora_A.default.weight", [4, Hidden], new float[4 * Hidden]);
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Contains("(9 problem(s))", e.Message);
        Assert.Contains("; ... and 3 more. Nothing is skipped silently", e.Message);
    }

    [Fact]
    public void TwoAdapterSlots_AreRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 10);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 11, slot: "other");
        string manifest = Plugin(adapter);

        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Contains("several PEFT adapter slots (default, other); export the one to use on its own", e.Message);
        Assert.Contains("a second lora_A for transformer_blocks.0.attn.to_q", e.Message);
    }

    [Fact]
    public void AnAdapterWithoutTensors_IsRefused()
    {
        string manifest = Plugin(new Adapter());
        var e = Assert.Throws<InvalidDataException>(() => Load(manifest));
        Assert.Equal($"LoRA '{W8}' contains no tensors.", e.Message);
    }

    [Theory]
    [InlineData(float.NaN)]
    [InlineData(float.PositiveInfinity)]
    [InlineData(float.NegativeInfinity)]
    public void NonFiniteFactor_IsRefusedNamingTheTensor(float bad)
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 12);
        var (a, b) = (MiniMaxH3LoraFixtures.Rand(4 * Ffn, 13), MiniMaxH3LoraFixtures.Rand(Hidden * 4, 14));
        b[5] = bad;
        adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc2, 4, a, b);
        string manifest = Plugin(adapter);
        string tensor = Adapter.Key(false, 1, MiniMaxH3LoraTarget.Fc2, 'B');
        string expected = $"LoRA '{W8}': {tensor} holds a non-finite value " +
                          $"({bad.ToString(System.Globalization.CultureInfo.InvariantCulture)}) at element 5; the adapter file is damaged.";

        Assert.Equal(expected, Assert.Throws<InvalidDataException>(() => Load(manifest)).Message);

        // Direct mode reads lazily: the load passes, the read of that term fails.
        using var set = Load(manifest, BackendType.Cpu, direct: true);
        Assert.NotNull(set.ReadFactors(false, 0, MiniMaxH3LoraTarget.Q));
        Assert.Equal(expected, Assert.Throws<InvalidDataException>(() => set.ReadFactors(false, 1, MiniMaxH3LoraTarget.Fc2)).Message);
    }

    // ---- the spec and its plug-in config ------------------------------------------------------

    [Fact]
    public void EverySpecForm_LoadsTheSamePlugin()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Fc1, 4, seed: 15);
        string manifest = Plugin(adapter, scale: .5f);
        string weights = Path.Combine(_fx.Dir, W8);
        var forms = new[]
        {
            new LoraSpec(manifest),                                   // --lora config.json, unresolved
            LoraCliFlags.Resolve(new[] { new LoraSpec(manifest) })[0], // after the host's Resolve
            new LoraSpec(weights, null, manifest),                    // --lora weights --lora-config config.json
        };
        float[] expected = null;
        foreach (var spec in forms)
        {
            using var set = Load(spec, BackendType.Cpu, direct: true);
            Assert.Equal(weights, set.WeightsPath);
            Assert.Equal(.5f, set.Strength);
            float[] up = set.ReadFactors(false, 0, MiniMaxH3LoraTarget.Fc1)!.Value.Up;
            if (expected == null) expected = up;
            else AssertBits(expected, up, spec.ToString());
        }
    }

    [Fact]
    public void APluginConfig_WithAConfigAfterIt_IsRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 15);
        string manifest = Plugin(adapter);
        string other = _fx.WriteManifest("other.json", W8);

        var e = Assert.Throws<ArgumentException>(() => Load(new LoraSpec(manifest, null, other)));
        Assert.Equal($"--lora {manifest} is a plug-in config already; drop the --lora-config after it.", e.Message);
    }

    [Fact]
    public void BareWeights_AreRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 16);
        string weights = _fx.WriteAdapter(adapter);
        string peft = _fx.WriteText("adapter_config.json", """{ "r": 4, "lora_alpha": 8 }""");
        string expected = MiniMaxH3LoraConfig.BareWeightsRefusal(W8);

        Assert.Equal(expected, Assert.Throws<ArgumentException>(() => Load(new LoraSpec(weights))).Message);
        Assert.Equal(expected, Assert.Throws<ArgumentException>(() => Load(new LoraSpec(weights, null, peft))).Message);
        Assert.Equal(
            $"--lora {W8}: MiniMax-H3 takes LoRA plug-ins only as a minimax-h3-lora config (for example " +
            "config/lora/minimax-h3-fl2v-turbo-8step.json), which declares the checkpoint partition and the sampling contract. " +
            "Pass that config instead of the bare weights.", expected);
    }

    [Fact]
    public void AQwenImagePlugin_IsRefused()
    {
        string qwen = _fx.WriteText("qwen.json", """{ "type": "qwen-image-2.1-lora", "weights": "x.safetensors" }""");
        var e = Assert.Throws<ArgumentException>(() => Load(new LoraSpec(qwen)));
        Assert.Equal("--lora qwen.json is a qwen-image-2.1-lora plug-in; MiniMax-H3 (minimax-h3) takes minimax-h3-lora plug-ins only.", e.Message);
    }

    [Fact]
    public void MissingFiles_AreReportedAsMissing()
    {
        string missing = Path.Combine(_fx.Dir, "typo.json");
        Assert.Equal($"--lora file not found: {missing}",
            Assert.Throws<FileNotFoundException>(() => Load(new LoraSpec(missing))).Message);

        string weights = _fx.WriteAdapter(new Adapter());
        Assert.Equal($"--lora-config file not found: {missing}",
            Assert.Throws<FileNotFoundException>(() => Load(new LoraSpec(weights, null, missing))).Message);

        // A config whose weights are not there yet (not resolved, so not downloaded).
        string manifest = _fx.WriteManifest("turbo-4step.json", W4, steps: 4, name: "Test Turbo 4");
        Assert.Equal($"--lora file not found: {Path.Combine(_fx.Dir, W4)}",
            Assert.Throws<FileNotFoundException>(() => Load(new LoraSpec(manifest))).Message);
    }

    [Fact]
    public void FourStepWeights_WithTheEightStepConfig_AreRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 17);
        string weights4 = _fx.WriteAdapter(adapter, W4);
        string manifest8 = _fx.WriteManifest(Manifest, W8);

        var e = Assert.Throws<ArgumentException>(() => Load(new LoraSpec(weights4, null, manifest8)));
        Assert.Equal($"--lora-config {Manifest} is the Test Turbo 8 plug-in for {W8}; --lora names {W4}. " +
                     "Pass the plug-in alone with --lora (it names its own weights), or the weights it was written for.", e.Message);
    }

    [Fact]
    public void NegativeStrength_IsRefused_AndZeroIsLegal()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 18);
        string manifest = Plugin(adapter);

        var e = Assert.Throws<ArgumentException>(() => Load(new LoraSpec(manifest, -.5f)));
        Assert.Equal($"Test Turbo 8 ({W8}): strength -0.5 is negative; the Turbo schedule is distilled at strength 1 " +
                     "(0 keeps the adapter's terms with zero up factors).", e.Message);
        using var zero = Load(new LoraSpec(manifest, 0f));
        Assert.Equal(0f, zero.Strength);
    }

    [Fact]
    public void TwoPlugins_AreRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 19);
        string manifest = Plugin(adapter);
        var specs = new[] { new LoraSpec(manifest), new LoraSpec(manifest) };

        var e = Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraSet.Load(specs, _cfg, MiniMaxH3Partition.FirstLastFrame, BackendType.GgmlCpu, direct: false));
        Assert.Equal($"Two MiniMax-H3 plug-ins ({Manifest}, {Manifest}) each fix the sampling schedule; use one.", e.Message);
        Assert.Throws<ArgumentException>(() =>
            MiniMaxH3LoraSet.Load(Array.Empty<LoraSpec>(), _cfg, MiniMaxH3Partition.FirstLastFrame, BackendType.GgmlCpu, direct: false));
    }

    [Fact]
    public void TheRef2vaCheckpoint_IsRefused()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 20);
        string manifest = Plugin(adapter);
        string ref2va = _fx.WriteTransformer(MiniMaxH3Partition.Reference);
        var partition = MiniMaxH3Config.PartitionFromFileName(ref2va);
        Assert.Equal(MiniMaxH3Partition.Reference, partition);

        // Same shapes as FL2VA: only the file name tells the checkpoints apart.
        var e = Assert.Throws<ArgumentException>(() => MiniMaxH3LoraSet.Load(new[] { new LoraSpec(manifest) },
            MiniMaxH3LoraFixtures.ConfigOf(ref2va), partition, BackendType.GgmlCpu, direct: false));
        Assert.Equal($"LoRA plug-in {Manifest} is for the MiniMax-H3 FL2VA checkpoint; the loaded model is the Ref2VA checkpoint, " +
                     "and Ref2VA Turbo is not supported. Load minimax_h3_fl2va_pruned-*.gguf or drop the plug-in.", e.Message);
    }

    // ---- packing --------------------------------------------------------------------------

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Fc1_UpHalves_AreSwappedIntoGateValueOrder(bool gateOnly)
    {
        // diffusers ff.net.0.proj rows: [value; gate]. Zero one half to see where it lands.
        const int rank = 2;
        float[] a = MiniMaxH3LoraFixtures.Rand(rank * Hidden, 21);
        float[] b = MiniMaxH3LoraFixtures.Rand(2 * Ffn * rank, 22);
        int zeroFirst = gateOnly ? 0 : Ffn;   // value rows are [0, ffn) in the file
        Array.Clear(b, zeroFirst * rank, Ffn * rank);
        var adapter = new Adapter(alpha: 2f);
        adapter.Term(false, 1, MiniMaxH3LoraTarget.Fc1, rank, a, b);
        string manifest = Plugin(adapter, alpha: 2f);   // scale 1

        // Packed: fc1 rows [0, ffn) are the gate, [ffn, 2 ffn) the value.
        using (var set = Load(manifest))
        {
            float[] up = set.PackedUpForTest(false, 1, MiniMaxH3LoraTarget.Fc1);
            Assert.Equal(2 * Ffn * 16, up.Length);
            bool RowsZero(int first) => up.Skip(first * 16).Take(Ffn * 16).All(v => v == 0f);
            Assert.False(RowsZero(gateOnly ? 0 : Ffn));
            Assert.True(RowsZero(gateOnly ? Ffn : 0));
            AssertClose(adapter.Delta(false, 1, MiniMaxH3LoraTarget.Fc1, 1f), Delta(PartOf(set, false, 1, MiniMaxH3LoraTarget.Fc1)), "fc1", 1e-5f);
        }

        // Direct: up rows are the file's rows with the halves exchanged, exactly.
        using (var set = Load(manifest, BackendType.Cpu, direct: true))
        {
            float[] up = set.ReadFactors(false, 1, MiniMaxH3LoraTarget.Fc1)!.Value.Up;
            AssertBits(b.Skip(Ffn * rank).Take(Ffn * rank).ToArray(), up.Take(Ffn * rank).ToArray(), "gate rows");
            AssertBits(b.Take(Ffn * rank).ToArray(), up.Skip(Ffn * rank).ToArray(), "value rows");
        }
    }

    [Theory]
    [InlineData(BackendType.GgmlMetal, 3, 64)]
    [InlineData(BackendType.GgmlMetal, 64, 64)]
    [InlineData(BackendType.GgmlMetal, 65, 128)]
    [InlineData(BackendType.GgmlCpu, 3, 16)]
    [InlineData(BackendType.GgmlCuda, 17, 32)]
    [InlineData(BackendType.GgmlVulkan, 16, 16)]
    public void Ranks_ArePaddedWithZeros_To64OnMetalAnd16Elsewhere(BackendType backend, int rank, int padded)
    {
        var adapter = new Adapter(alpha: rank);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, rank, seed: 23);
        adapter.Term(true, 0, MiniMaxH3LoraTarget.Out, rank, seed: 24);
        using var set = Load(Plugin(adapter, alpha: rank), backend);

        foreach (var (refiner, target) in new[] { (false, MiniMaxH3LoraTarget.Q), (true, MiniMaxH3LoraTarget.Out) })
        {
            var l = PartOf(set, refiner, 0, target);
            Assert.Equal(padded, l.Rank);
            float[] up = UpOf(l), down = DownOf(l);
            for (long o = 0; o < l.Out; o++)
                for (int r = rank; r < padded; r++)
                    Assert.Equal(0f, up[o * padded + r]);
            Assert.All(down.Skip(rank * (int)l.In), v => Assert.Equal(0f, v));
            Assert.Equal(up, set.PackedUpForTest(refiner, 0, target));
            AssertClose(adapter.Delta(refiner, 0, target, 1f), Delta(l), $"{target}", backend == BackendType.GgmlCpu ? 1e-5f : 5e-3f);
        }
        Assert.Contains($"rank padded to a multiple of {(backend == BackendType.GgmlMetal ? 64 : 16)}", set.Summary);
    }

    [Theory]
    [InlineData(BackendType.GgmlCpu, 4)]
    [InlineData(BackendType.GgmlMetal, 2)]
    public void QkvDowns_SitBackToBack_InOneAlignedAllocation(BackendType backend, int elem)
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 25);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.K, 20, seed: 26);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.V, 4, seed: 27);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Out, 4, seed: 28);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Fc1, 4, seed: 29);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Fc2, 4, seed: 30);
        // Block 1 updates k and v only: they still sit back to back.
        adapter.Term(false, 1, MiniMaxH3LoraTarget.K, 4, seed: 31);
        adapter.Term(false, 1, MiniMaxH3LoraTarget.V, 4, seed: 32);
        using var set = Load(Plugin(adapter), backend);

        var b0 = BlockOf(set, false, 0);
        Assert.Equal(b0.Q.Type, b0.K.Type);
        Assert.Equal(b0.Q.Type, b0.V.Type);
        Assert.Equal(b0.Q.Down + (nint)(b0.Q.Rank * b0.Q.In * elem), b0.K.Down);
        Assert.Equal(b0.K.Down + (nint)(b0.K.Rank * b0.K.In * elem), b0.V.Down);
        foreach (var p in new[] { b0.Q.Down, b0.Q.Up, b0.K.Up, b0.V.Up, b0.Out.Down, b0.Out.Up, b0.Fc1.Down, b0.Fc1.Up, b0.Fc2.Down, b0.Fc2.Up })
            Assert.Equal(0L, (long)p % 16384);

        var b1 = BlockOf(set, false, 1);
        AssertEmpty(b1.Q);
        Assert.Equal(0L, (long)b1.K.Down % 16384);
        Assert.Equal(b1.K.Down + (nint)(b1.K.Rank * b1.K.In * elem), b1.V.Down);
    }

    [Theory]
    [InlineData(BackendType.GgmlCpu, F32, "F32 factors (GgmlCpu keeps F32)")]
    [InlineData(BackendType.GgmlMetal, F16, "F16 factors,")]
    [InlineData(BackendType.GgmlCuda, F16, "F16 factors,")]
    [InlineData(BackendType.GgmlVulkan, F16, "F16 factors,")]
    public void FactorType_IsF32OnGgmlCpu_AndF16OnTheGpuBackends(BackendType backend, int type, string summary)
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4);
        using var set = Load(Plugin(adapter), backend);

        foreach (var (refiner, block, target) in adapter.Targets)
            Assert.Equal(type, PartOf(set, refiner, block, target).Type);
        Assert.Contains(summary, set.Summary);
    }

    [Fact]
    public void AValueAtF16Range_KeepsItsQkvGroupF32()
    {
        // Rank 1 at alpha 1: up and down are balanced to about 1.2e5, beyond F16's range.
        var adapter = new Adapter(alpha: 1f);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 1, Enumerable.Repeat(1e5f, Hidden).ToArray(), Enumerable.Repeat(1e5f, Inner).ToArray());
        adapter.Term(false, 0, MiniMaxH3LoraTarget.K, 1, seed: 33);
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Out, 1, seed: 34);
        using var set = Load(Plugin(adapter, alpha: 1f), BackendType.GgmlMetal);

        var b0 = BlockOf(set, false, 0);
        Assert.Equal(F32, b0.Q.Type);
        Assert.Equal(F32, b0.K.Type);
        Assert.Equal(F16, b0.Out.Type);
        Assert.Equal(b0.Q.Down + (nint)(b0.Q.Rank * b0.Q.In * 4), b0.K.Down);
        Assert.Contains("F16/F32 factors", set.Summary);
        AssertClose(adapter.Delta(false, 0, MiniMaxH3LoraTarget.Q, 1f), Delta(b0.Q), "q", 1e-5f);
    }

    [Fact]
    public void StrengthZero_KeepsEveryTerm_WithZeroUpFactors()
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4);
        string manifest = Plugin(adapter);

        using (var set = Load(new LoraSpec(manifest, 0f), BackendType.GgmlMetal))
        {
            Assert.Equal(18, set.TermCount);
            Assert.Equal(new[] { 0f }, set.Scales);
            foreach (var (refiner, block, target) in adapter.Targets)
            {
                var l = PartOf(set, refiner, block, target);
                Assert.Equal(64, l.Rank);
                Assert.All(UpOf(l), v => Assert.Equal(0f, v));
                Assert.Contains(DownOf(l), v => v != 0f);
            }
            Assert.Contains("strength 0, scale 0;", set.Summary);
            Assert.EndsWith("; the schedule was distilled at strength 1", set.Summary);
        }

        using (var set = Load(new LoraSpec(manifest, 0f), BackendType.Cpu, direct: true))
        {
            foreach (var (refiner, block, target) in adapter.Targets)
            {
                var factors = set.ReadFactors(refiner, block, target)!.Value;
                Assert.All(factors.Up, v => Assert.Equal(0f, v));
                AssertBits(adapter.Factors(refiner, block, target)!.Value.A, factors.Down, $"{target} down");
            }
        }

        using (var set = Load(manifest))
            Assert.DoesNotContain("distilled", set.Summary);
    }

    // ---- lifetime ---------------------------------------------------------------------------

    [Fact]
    public void Dispose_InvalidatesEveryFactorBeforeFreeingIt()
    {
        var adapter = new Adapter().AllTargets(blocks: 2, refinerBlocks: 1, rank: 4);
        var set = Load(Plugin(adapter), BackendType.GgmlMetal);
        var mine = FactorPointers(set);
        Assert.Equal(2 * set.TermCount, mine.Count);
        int calls = 0;
        bool beforeFree = true;
        // Pointers of other sets (tests running in parallel) still reach the backend.
        MiniMaxH3LoraSet.InvalidateHookForTest = p =>
        {
            if (!mine.Contains(p))
            {
                GgmlBasicOps.InvalidateHostBuffer(p);
                return;
            }
            Interlocked.Increment(ref calls);
            beforeFree &= set.NativeAdapter != IntPtr.Zero;
        };
        try
        {
            set.InvalidateDeviceCopies();
            Assert.Equal(mine.Count, calls);
            Assert.NotEqual(IntPtr.Zero, set.NativeAdapter);

            calls = 0;
            set.Dispose();
            Assert.Equal(mine.Count, calls);
            Assert.True(beforeFree, "A factor was invalidated after its memory was freed.");
            Assert.Equal(IntPtr.Zero, set.NativeAdapter);

            calls = 0;
            set.InvalidateDeviceCopies();
            set.Dispose();
            Assert.Equal(0, calls);
            Assert.Throws<ObjectDisposedException>(() => set.PackedUpForTest(false, 0, MiniMaxH3LoraTarget.Q));
        }
        finally
        {
            MiniMaxH3LoraSet.InvalidateHookForTest = null;
            set.Dispose();
        }
    }

    [Fact]
    public void DirectMode_KeepsTheFile_AndHasNoNativeFactors()
    {
        var adapter = new Adapter();
        var (a, _) = adapter.Term(false, 1, MiniMaxH3LoraTarget.Out, 4, seed: 35);
        adapter.Term(true, 0, MiniMaxH3LoraTarget.Fc1, 4, seed: 36);
        var set = Load(Plugin(adapter), BackendType.Cpu, direct: true);
        try
        {
            Assert.True(set.IsDirect);
            Assert.Equal(IntPtr.Zero, set.NativeAdapter);
            Assert.Equal((4L * Inner + Hidden * 4L + 4L * Hidden + 2 * Ffn * 4L) * 4, set.FactorBytes);
            Assert.Contains("exact F32 factors read per block for the direct backend", set.Summary);
            Assert.Null(set.ReadFactors(false, 0, MiniMaxH3LoraTarget.Out));
            AssertBits(a, set.ReadFactors(false, 1, MiniMaxH3LoraTarget.Out)!.Value.Down, "down");
            Assert.Throws<ArgumentOutOfRangeException>(() => set.ReadFactors(false, 2, MiniMaxH3LoraTarget.Out));
            Assert.Throws<ArgumentOutOfRangeException>(() => set.ReadFactors(true, 1, MiniMaxH3LoraTarget.Out));
            Assert.Throws<InvalidOperationException>(() => set.PackedUpForTest(false, 1, MiniMaxH3LoraTarget.Out));

            int calls = 0;
            MiniMaxH3LoraSet.InvalidateHookForTest = p => Interlocked.Increment(ref calls);
            try { set.Dispose(); }
            finally { MiniMaxH3LoraSet.InvalidateHookForTest = null; }
            Assert.Equal(0, calls);
            Assert.Throws<ObjectDisposedException>(() => set.ReadFactors(false, 1, MiniMaxH3LoraTarget.Out));
        }
        finally
        {
            set.Dispose();
        }
    }

    [Fact]
    public void PackedMode_DoesNotServeReadFactors()
    {
        var adapter = new Adapter();
        adapter.Term(false, 0, MiniMaxH3LoraTarget.Q, 4, seed: 37);
        using var set = Load(Plugin(adapter));
        Assert.Throws<InvalidOperationException>(() => set.ReadFactors(false, 0, MiniMaxH3LoraTarget.Q));
        Assert.Throws<InvalidOperationException>(() => set.PackedUpForTest(false, 0, MiniMaxH3LoraTarget.K));
    }
}
