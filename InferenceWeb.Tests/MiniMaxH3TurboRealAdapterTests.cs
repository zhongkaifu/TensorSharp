// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>
/// The published LightX2V Turbo adapters (lightx2v/Minimax-h3-Turbo at a29e57d) against the
/// FL2VA checkpoint's own shapes, on the CPU only, and against an independent oracle: rows of
/// the publisher's ComfyUI twin of each adapter
/// (Fixtures/MiniMaxH3/turbo-comfy-reference-rows.json, captured by
/// eng/validation/minimax-h3-turbo-comfy-rows.py). The twin is already in checkpoint layout
/// (fc1 [gate; value], block-diagonal stacked qkv), so its rows pin the fc1 swap and the q/k/v
/// placement with data neither TensorSharp nor this test produced.
/// </summary>
/// <remarks>
/// Gated: TS_MINIMAX_H3_DIR must hold a minimax_h3_fl2va GGUF (only its tensor table is read),
/// and each adapter must sit where the shipped plug-in config's weights entry resolves
/// (<c>${TENSORSHARP_MODELS:-models}/minimax-h3/loras/</c>). A skipped test proves nothing.
/// The fixture's own consistency is checked without either.
/// </remarks>
public sealed class MiniMaxH3TurboRealAdapterTests
{
    internal const string DirEnv = "TS_MINIMAX_H3_DIR";
    internal const string Fl2vaPattern = "minimax_h3_fl2va";
    internal const string Manifest8 = "minimax-h3-fl2v-turbo-8step.json";
    internal const string Manifest4 = "minimax-h3-fl2v-turbo-4step.json";
    private const string Revision = "a29e57d195f80ead07c89f89e55ecf2911e893c5";
    private const int Rank = 128, ComfyQkvRank = 3 * Rank;
    private const float TurboScale = .0625f;   // alpha 8 / rank 128
    private const int Block = 10;

    private readonly ITestOutputHelper _output;

    public MiniMaxH3TurboRealAdapterTests(ITestOutputHelper output) => _output = output;

    // ---- locating the files -------------------------------------------------------------------

    private static string FindRepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx")))
            dir = dir.Parent;
        return dir?.FullName;
    }

    /// <summary>A shipped plug-in config, or null outside a source checkout.</summary>
    internal static string Shipped(string manifest)
    {
        string root = FindRepoRoot();
        return root == null ? null : Path.Combine(root, "config", "lora", manifest);
    }

    /// <summary>Where the shipped config's weights entry resolves (never downloads).</summary>
    internal static string WeightsOf(string manifest)
    {
        string path = Shipped(manifest);
        return path != null && File.Exists(path) ? ConfigFileArgs.ResolveFileEntryPath(path, "weights") : null;
    }

    private static MiniMaxH3Config Fl2vaConfig()
    {
        string gguf = TestGates.FindGguf(Environment.GetEnvironmentVariable(DirEnv), Fl2vaPattern);
        Assert.NotNull(gguf);
        Assert.Equal(MiniMaxH3Partition.FirstLastFrame, MiniMaxH3Config.PartitionFromFileName(gguf));
        using var file = new GgufFile(gguf);
        return MiniMaxH3Config.Detect(file.Tensors);
    }

    private static MiniMaxH3LoraSet Load(string manifest, BackendType backend, bool direct, MiniMaxH3Config cfg,
        MiniMaxH3Partition partition = MiniMaxH3Partition.FirstLastFrame) =>
        // TensorAgent's form: the weights, with the shipped config as --lora-config (the tie is checked).
        MiniMaxH3LoraSet.Load(new[] { new LoraSpec(WeightsOf(manifest), null, Shipped(manifest)) }, cfg, partition, backend, direct);

    // ---- the ComfyUI reference rows -------------------------------------------------------------

    private sealed record ComfyRow(string Tensor, int Row, long First, long Last, float[] Values);

    private sealed record ComfyFile(string DiffusersFile, string ComfyFileName, long DataStart, JsonElement Metadata,
        JsonElement Tensors, IReadOnlyList<ComfyRow> Rows);

    private static (JsonElement Root, IReadOnlyList<ComfyFile> Files) Fixture()
    {
        using Stream stream = typeof(MiniMaxH3TurboRealAdapterTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MiniMaxH3.turbo-comfy-reference-rows.json");
        Assert.NotNull(stream);
        using var document = JsonDocument.Parse(stream);
        var root = document.RootElement.Clone();
        var files = new List<ComfyFile>();
        foreach (var file in root.GetProperty("files").EnumerateArray())
        {
            var rows = new List<ComfyRow>();
            foreach (var row in file.GetProperty("rows").EnumerateArray())
            {
                byte[] raw = Convert.FromHexString(row.GetProperty("bf16_le_hex").GetString());
                var values = new float[raw.Length / 2];
                for (int i = 0; i < values.Length; i++)
                    values[i] = BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(raw, 2 * i) << 16);
                var range = row.GetProperty("byte_range");
                rows.Add(new ComfyRow(row.GetProperty("tensor").GetString(), row.GetProperty("row").GetInt32(),
                    range[0].GetInt64(), range[1].GetInt64(), values));
            }
            files.Add(new ComfyFile(file.GetProperty("diffusers_file").GetString(), file.GetProperty("comfyui_file").GetString(),
                file.GetProperty("data_start").GetInt64(), file.GetProperty("metadata"), file.GetProperty("tensors"), rows));
        }
        return (root, files);
    }

    private static ComfyRow RowOf(ComfyFile file, string tensor, int row) =>
        file.Rows.Single(r => r.Tensor == tensor && r.Row == row);

    [Fact]
    public void ComfyFixture_IsPinnedAndSelfConsistent()
    {
        var (root, files) = Fixture();
        Assert.Equal("lightx2v/Minimax-h3-Turbo", root.GetProperty("repository").GetString());
        Assert.Equal(Revision, root.GetProperty("revision").GetString());
        Assert.Equal(Block, root.GetProperty("block").GetInt32());
        Assert.Equal(0, root.GetProperty("absent").GetArrayLength());
        Assert.Equal(new[] { MiniMaxH3LoraFixtures.Weights8, MiniMaxH3LoraFixtures.Weights4 }, files.Select(f => f.DiffusersFile));

        foreach (var file in files)
        {
            Assert.Equal(file.DiffusersFile.Replace("_bf16.safetensors", "_comfyui_bf16.safetensors"), file.ComfyFileName);
            Assert.Equal("Diffusers [value;gate] -> ComfyUI [gate;value]", file.Metadata.GetProperty("swi_glu_mapping").GetString());
            Assert.Equal("block diagonal B; concat A; alpha multiplied by 3", file.Metadata.GetProperty("qkv_fusion").GetString());
            Assert.Equal("0.0625", file.Metadata.GetProperty("training_scale").GetString());

            // Both fused tensors fold to the Diffusers adapter's 8 / 128.
            var fc1 = file.Tensors.GetProperty("fc1");
            var qkv = file.Tensors.GetProperty("qkv");
            Assert.Equal(new[] { 28672, Rank }, fc1.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()));
            Assert.Equal(new[] { 21504, ComfyQkvRank }, qkv.GetProperty("shape").EnumerateArray().Select(v => v.GetInt32()));
            Assert.Equal(TurboScale, fc1.GetProperty("alpha").GetSingle() / Rank);
            Assert.Equal(TurboScale, qkv.GetProperty("alpha").GetSingle() / ComfyQkvRank);

            // Every row's byte range is where the header puts it.
            foreach (var row in file.Rows)
            {
                var tensor = file.Tensors.GetProperty(row.Tensor);
                int width = tensor.GetProperty("shape")[1].GetInt32();
                Assert.Equal(width, row.Values.Length);
                Assert.Equal(file.DataStart + tensor.GetProperty("data_offsets")[0].GetInt64() + (long)row.Row * width * 2, row.First);
                Assert.Equal(row.First + width * 2 - 1, row.Last);
                Assert.Contains(row.Values, v => v != 0f);
            }
            Assert.Equal(new[] { 0, 14336 }, file.Rows.Where(r => r.Tensor == "fc1").Select(r => r.Row));
            Assert.Equal(new[] { 0, 7168, 14336 }, file.Rows.Where(r => r.Tensor == "qkv").Select(r => r.Row));

            // qkv_fusion: row i x 7168 is non-zero only in column block i (q, k, v).
            for (int part = 0; part < 3; part++)
            {
                float[] values = RowOf(file, "qkv", part * 7168).Values;
                for (int c = 0; c < ComfyQkvRank; c++)
                    if (c / Rank != part) Assert.Equal(0f, values[c]);
                Assert.Contains(values.Skip(part * Rank).Take(Rank), v => v != 0f);
            }
        }
    }

    // ---- the real adapters -----------------------------------------------------------------------

    [MiniMaxH3TurboAdapterFact(Manifest8)]
    public void EightStep_LoadsCompletely_AndRefusesRef2va() => LoadsCompletely(Manifest8);

    [MiniMaxH3TurboAdapterFact(Manifest4)]
    public void FourStep_LoadsCompletely_AndRefusesRef2va() => LoadsCompletely(Manifest4);

    private void LoadsCompletely(string manifest)
    {
        var cfg = Fl2vaConfig();
        Assert.Equal((50, 2), (cfg.NumLayers, cfg.TokenRefinerLayers));
        using (var set = Load(manifest, BackendType.Cpu, direct: true, cfg))
        {
            _output.WriteLine(set.Summary);
            Assert.Equal(624, set.TensorCount);
            Assert.Equal(312, set.TermCount);
            Assert.Equal(new[] { Rank }, set.Ranks);
            Assert.Equal(new[] { TurboScale }, set.Scales);
            Assert.Equal(8f, set.Alpha);
            Assert.Equal(1f, set.Strength);
            Assert.Equal(manifest == Manifest8 ? 8 : 4, set.Recipe.Steps);
            var fc2 = set.ReadFactors(true, 1, MiniMaxH3LoraTarget.Fc2)!.Value;
            Assert.Equal((Rank, (long)cfg.FfnHiddenSize, (long)cfg.HiddenSize), (fc2.Rank, fc2.In, fc2.Out));
        }

        // Same shapes on the Ref2VA checkpoint; its partition (the file name) refuses the plug-in.
        var e = Assert.Throws<ArgumentException>(() => Load(manifest, BackendType.Cpu, direct: true, cfg, MiniMaxH3Partition.Reference));
        Assert.Contains("is the Ref2VA checkpoint, and Ref2VA Turbo is not supported", e.Message);
    }

    [MiniMaxH3TurboAdapterFact(Manifest8)]
    public void EightStep_MatchesItsComfyUiTwinsRows() => MatchesComfyRows(Manifest8);

    [MiniMaxH3TurboAdapterFact(Manifest4)]
    public void FourStep_MatchesItsComfyUiTwinsRows() => MatchesComfyRows(Manifest4);

    private void MatchesComfyRows(string manifest)
    {
        var cfg = Fl2vaConfig();
        var (_, files) = Fixture();
        var file = files.Single(f => f.DiffusersFile == Path.GetFileName(WeightsOf(manifest)));
        int ffn = cfg.FfnHiddenSize, inner = cfg.AttentionInnerDim;
        Assert.Equal((14336, 7168), (ffn, inner));
        using var set = Load(manifest, BackendType.Cpu, direct: true, cfg);

        // BF16 -> F32 is exact and 0.0625 a power of two, so the comparison is bit for bit.
        float[] Row(float[] up, int row) => up.Skip(row * Rank).Take(Rank).Select(v => v / TurboScale).ToArray();
        void AssertRow(float[] expected, float[] actual, string what)
        {
            for (int i = 0; i < expected.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                    $"{what}[{i}]: ComfyUI {expected[i]:R}, TensorSharp {actual[i]:R}");
        }

        // fc1: the twin's gate row 0 and value row ffn are TensorSharp's swapped rows 0 and ffn.
        float[] fc1 = set.ReadFactors(false, Block, MiniMaxH3LoraTarget.Fc1)!.Value.Up;
        AssertRow(RowOf(file, "fc1", 0).Values, Row(fc1, 0), "fc1 gate row 0");
        AssertRow(RowOf(file, "fc1", 14336).Values, Row(fc1, ffn), "fc1 value row 0");
        // Without the swap they would differ: the oracle can tell the two orders apart.
        Assert.False(RowOf(file, "fc1", 0).Values.SequenceEqual(Row(fc1, ffn)));

        // qkv: the twin's stacked row i x inner, column block i, is row 0 of q, k and v.
        var parts = new[] { MiniMaxH3LoraTarget.Q, MiniMaxH3LoraTarget.K, MiniMaxH3LoraTarget.V };
        for (int part = 0; part < 3; part++)
        {
            float[] up = set.ReadFactors(false, Block, parts[part])!.Value.Up;
            float[] expected = RowOf(file, "qkv", part * inner).Values.Skip(part * Rank).Take(Rank).ToArray();
            AssertRow(expected, Row(up, 0), $"{parts[part]} row 0");
        }
        _output.WriteLine($"{file.ComfyFileName} @ {Revision[..7]}: fc1 rows 0/{ffn} and q/k/v row 0 of block {Block} match bit for bit.");
    }

    [MiniMaxH3TurboAdapterFact(Manifest8)]
    public void EightStep_PacksF32ForGgmlCpu_AndF16ForTheGpuBackends()
    {
        var cfg = Fl2vaConfig();
        (float[] Down, float[] Up, int Rank, long In, long Out) direct;
        using (var set = Load(Manifest8, BackendType.Cpu, direct: true, cfg))
            direct = set.ReadFactors(false, Block, MiniMaxH3LoraTarget.Fc1)!.Value;
        float[] expected = DeltaRow(direct.Up, direct.Down, Rank, direct.In, row: 0);

        // Parameters: per block (3 x (rank x hidden + inner x rank)) + rank x inner + hidden x rank
        // + rank x hidden + 2 ffn x rank + rank x ffn + hidden x rank, over 52 blocks.
        long hidden = cfg.HiddenSize, inner = cfg.AttentionInnerDim, ffn = cfg.FfnHiddenSize;
        long parameters = 52 * Rank * (3 * (hidden + inner) + (inner + hidden) + (hidden + 2 * ffn) + (ffn + hidden));
        foreach (var (backend, type, elem, tolerance) in new[] { (BackendType.GgmlCpu, 0, 4, 1e-5f), (BackendType.GgmlMetal, 1, 2, 5e-3f) })
        {
            var clock = Stopwatch.StartNew();
            using var set = Load(Manifest8, backend, direct: false, cfg);
            clock.Stop();
            _output.WriteLine($"{backend}: packed in {clock.Elapsed.TotalSeconds:F1} s; {set.Summary}");
            Assert.Equal(parameters * elem, set.FactorBytes);

            var adapter = Marshal.PtrToStructure<H3DitAdapter>(set.NativeAdapter);
            for (int b = 0; b < 52; b++)
            {
                bool refiner = b < 2;
                var block = Marshal.PtrToStructure<H3BlockLora>((refiner ? adapter.Refiner : adapter.Blocks) +
                    (refiner ? b : b - 2) * Marshal.SizeOf<H3BlockLora>());
                foreach (var l in new[] { block.Q, block.K, block.V, block.Out, block.Fc1, block.Fc2 })
                {
                    Assert.Equal(type, l.Type);
                    Assert.Equal(Rank, l.Rank);
                }
            }

            // The balanced, packed factors apply the same update as the exact direct ones.
            var fc1 = Marshal.PtrToStructure<H3BlockLora>(adapter.Blocks + Block * Marshal.SizeOf<H3BlockLora>()).Fc1;
            float[] actual = DeltaRow(Read(fc1.Up, Rank, type), Read(fc1.Down, Rank * fc1.In, type), Rank, fc1.In, row: 0);
            float max = expected.Max(v => Math.Abs(v));
            for (int i = 0; i < expected.Length; i++)
                Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * max,
                    $"{backend} fc1 delta row 0 [{i}]: direct {expected[i]}, packed {actual[i]}");
        }
    }

    private static float[] Read(IntPtr p, long count, int type)
    {
        var values = new float[count];
        if (type == 1)
        {
            var raw = new short[count];
            Marshal.Copy(p, raw, 0, (int)count);
            for (long i = 0; i < count; i++) values[i] = (float)BitConverter.Int16BitsToHalf(raw[i]);
        }
        else
        {
            Marshal.Copy(p, values, 0, (int)count);
        }
        return values;
    }

    /// <summary>Row <paramref name="row"/> of up · down, accumulated in double.</summary>
    private static float[] DeltaRow(float[] up, float[] down, int rank, long input, int row)
    {
        var result = new float[input];
        for (long i = 0; i < input; i++)
        {
            double sum = 0;
            for (int r = 0; r < rank; r++) sum += (double)up[row * rank + r] * down[r * input + i];
            result[i] = (float)sum;
        }
        return result;
    }
}

/// <summary>
/// A fact on one of the published Turbo adapters: skips unless TS_MINIMAX_H3_DIR holds the
/// FL2VA GGUF and the adapter sits where the shipped plug-in config resolves its weights.
/// </summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
internal sealed class MiniMaxH3TurboAdapterFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models";

    public MiniMaxH3TurboAdapterFactAttribute(string manifest)
    {
        Skip = TestGates.ModelSkip(MiniMaxH3TurboRealAdapterTests.DirEnv, MiniMaxH3TurboRealAdapterTests.Fl2vaPattern);
        if (Skip != null) return;
        string weights = MiniMaxH3TurboRealAdapterTests.WeightsOf(manifest);
        if (weights == null)
            Skip = $"Requires a source checkout with config/lora/{manifest}.";
        else if (!File.Exists(weights))
            Skip = $"Requires the Turbo adapter {weights} (the weights entry of config/lora/{manifest}).";
    }
}
