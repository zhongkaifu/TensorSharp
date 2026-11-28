// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.QwenImage;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen-Image-2.1's denoiser on the diffusion runtime on a small synthetic transformer (and a diffusers-keyed LoRA over it),
/// against the retired transformer's velocities frozen before it was deleted (QwenImage21Goldens): the native graph
/// (TSGgml_QwenImage21Forward) on each GGML lane that froze them, the pure-C# transformer on the host lane. Step for step,
/// through the extract and cached graphs of the prefix cache.
/// </summary>
public sealed class QwenImage21DenoiserTests : IDisposable
{
    internal const int Channels = 16, TextDim = 48, Heads = 2, HeadDim = 32, Dim = Heads * HeadDim, Layers = 2, Ff = 96;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qi21-denoiser-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public QwenImage21DenoiserTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    internal static float[] Rand(int n, int seed, float scale = 1f, float offset = 0f)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = offset + scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>A minimal GGUF v3: no metadata, F32 tensors, 32-byte alignment.</summary>
    internal static void WriteGguf(string path, IReadOnlyList<(string Name, long[] Dims, float[] Data)> tensors) =>
        WriteGguf(path, tensors.Select(t =>
        {
            var raw = new byte[t.Data.Length * 4];
            Buffer.BlockCopy(t.Data, 0, raw, 0, raw.Length);
            return (t.Name, t.Dims, 0, raw);
        }).ToList());

    /// <summary>A minimal GGUF v3 of raw tensors of any GGML type: no metadata, 32-byte alignment.</summary>
    internal static void WriteGguf(string path, IReadOnlyList<(string Name, long[] Dims, int Type, byte[] Data)> tensors)
    {
        const int Alignment = 32;
        static long Pad(long n) => (n + Alignment - 1) / Alignment * Alignment;
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs, Encoding.UTF8);
        w.Write(0x46554747u);
        w.Write(3u);
        w.Write((ulong)tensors.Count);
        w.Write(0UL);
        long offset = 0;
        foreach (var (name, dims, type, data) in tensors)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(name);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
            w.Write((uint)dims.Length);
            foreach (long d in dims) w.Write((ulong)d);
            w.Write((uint)type);
            w.Write((ulong)offset);
            offset += Pad(data.Length);
        }
        w.Flush();
        w.Write(new byte[Pad(fs.Position) - fs.Position]);
        foreach (var (_, _, _, raw) in tensors)
        {
            w.Write(raw);
            w.Write(new byte[Pad(raw.Length) - raw.Length]);
        }
    }

    /// <summary>A two-block transformer: block 0 with separate gate and up projections, block 1 with a fused gate_up.</summary>
    /// <param name="fused">null: block 0 separate, block 1 fused; true / false: every block fused / separate.</param>
    internal static string WriteTransformer(string dir, bool? fused = null)
    {
        var tensors = new List<(string, long[], float[])>();
        int seed = 1;
        void Add(string name, long input, long output, float scale = 0.3f) =>
            tensors.Add((name + ".weight", new[] { input, output }, Rand((int)(input * output), seed++, scale)));
        void Gain(string name, int count) => tensors.Add((name + ".weight", new long[] { count }, Rand(count, seed++, 0.25f, 1f)));
        Add("img_in", Channels, Dim);
        Add("txt_in.in_layer", TextDim, Dim);
        Add("txt_in.out_layer", Dim, Dim);
        Gain("txt_in.text_norm", TextDim);
        Add("time_text_embed.timestep_embedder.linear_1", 256, Dim, 0.1f);
        Add("time_text_embed.timestep_embedder.linear_2", Dim, Dim);
        Add("modulation.1", Dim, 4 * Dim, 0.2f);
        Add("norm_out.linear", Dim, Dim, 0.2f);
        Add("proj_out", Dim, Channels);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"transformer_blocks.{i}.";
            Add(p + "attn.to_q", Dim, Dim);
            Add(p + "attn.to_k", Dim, Dim);
            Add(p + "attn.to_v", Dim, Dim);
            Add(p + "attn.to_out.0", Dim, Dim);
            if (fused == false || (fused == null && i == 0))
            {
                Add(p + "img_mlp.gate_layer", Dim, Ff);
                Add(p + "img_mlp.proj", Dim, Ff);
            }
            else Add(p + "img_mlp.gate_up", Dim, 2 * Ff);
            Add(p + "img_mlp.out", Ff, Dim);
            Gain(p + "attn.norm_q", HeadDim);
            Gain(p + "attn.norm_k", HeadDim);
        }
        string path = Path.Combine(dir, $"tiny-qwen-image-2.1-{(fused == null ? "mixed" : fused.Value ? "fused" : "separate")}.gguf");
        WriteGguf(path, tensors);
        return path;
    }

    /// <summary>One request's inputs: the conditioning (with vision slots for <paramref name="references"/> 2x4 references),
    /// the references' rows and the 4x6 target.</summary>
    internal static (ConditioningSet Text, LatentRef[] References, QwenImage21Layout Layout, float[] Target) Request(int references, int seed)
    {
        var slots = new List<int> { 0, 0, 0 };
        for (int r = 0; r < references; r++)
        {
            slots.Add(r + 1);
            slots.Add(r + 1);
            slots.Add(0);
        }
        slots.Add(0);
        int textRows = slots.Count;
        var text = new ConditioningSet(Rand(textRows * TextDim, seed, 1f), textRows, TextDim, slots.ToArray());
        var refs = new LatentRef[references];
        var shapes = new List<(int, int)>();
        for (int r = 0; r < references; r++)
        {
            refs[r] = new LatentRef(r, new LatentShape(Channels, 1, 2, 4), Rand(8 * Channels, seed + 10 + r));
            shapes.Add((2, 4));
        }
        shapes.Add((4, 6));
        QwenImage21Layout layout = QwenImage21Layout.Build(textRows, references > 0 ? slots.ToArray() : null, shapes, HeadDim);
        return (text, refs, layout, Rand(24 * Channels, seed + 99));
    }

    /// <summary>The managed tensor executor (the cpu backend) runs the same program: its velocities agree with the GGML CPU
    /// executor's to float rounding, through the prefix cache.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(2)]
    public void Managed_executor_runs_the_program_through_the_prefix_cache(int references)
    {
        string path = WriteTransformer(_dir);
        using var ggml = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend);
        // The machine's free memory decides whether a real cache fits; this one is a few KiB, so it never should be declined.
        using var managed = new QwenImage21Denoiser(path, BackendType.Cpu) { FreeMemory = () => -1 };
        var (text, refs, layout, target) = Request(references, 11);
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in new[] { (1f, 0), (0.6f, 1) })
        {
            float[] expected = (float[])ggml.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
            float[] actual = (float[])managed.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
            Assert.Equal(step == 0 ? QwenImage21Denoiser.Mode.Extract : QwenImage21Denoiser.Mode.Cached, managed.LastMode);
            double err = 0, norm = 0;
            for (int i = 0; i < expected.Length; i++)
            {
                err += (expected[i] - actual[i]) * (double)(expected[i] - actual[i]);
                norm += expected[i] * (double)expected[i];
            }
            // ggml-cpu attends F32 keys and values as the managed executor does; the GPU backends attend F16 copies (8e-4 on
            // an RTX 4000 Ada).
            double bound = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : 2e-3;
            Assert.True(Math.Sqrt(err / norm) < bound, $"step {step}: relative L2 {Math.Sqrt(err / norm):E2}, bound {bound:E0}");
        }
        ggml.FinishRequest();
        managed.FinishRequest();
    }

    /// <summary>A diffusers-keyed LoRA over the tiny transformer: q, k and v (one stacked shrink), to_out, both MLP halves of
    /// block 0 (separate projections) and of block 1 (the checkpoint's fused gate_up, split into row views), img_mlp.out,
    /// and five projections outside the blocks. F32 factors.</summary>
    internal static string WriteLora(string dir)
    {
        var tensors = new List<(string Name, long[] Shape, float[] Data)>();
        int seed = 500;
        void Factors(string module, int rank, int input, int output)
        {
            tensors.Add(($"transformer.{module}.lora_A.weight", new long[] { rank, input }, Rand(rank * input, seed++, 0.3f)));
            tensors.Add(($"transformer.{module}.lora_B.weight", new long[] { output, rank }, Rand(output * rank, seed++, 0.3f)));
        }
        foreach (int b in new[] { 0, 1 })
        {
            Factors($"transformer_blocks.{b}.attn.to_q", 4, Dim, Dim);
            Factors($"transformer_blocks.{b}.attn.to_k", 2, Dim, Dim);
            Factors($"transformer_blocks.{b}.attn.to_v", 4, Dim, Dim);
            Factors($"transformer_blocks.{b}.attn.to_out.0", 4, Dim, Dim);
            Factors($"transformer_blocks.{b}.img_mlp.gate_layer", 4, Dim, Ff);
            Factors($"transformer_blocks.{b}.img_mlp.proj", 4, Dim, Ff);
            Factors($"transformer_blocks.{b}.img_mlp.out", 2, Ff, Dim);
        }
        Factors("img_in", 4, Channels, Dim);
        Factors("txt_in.in_layer", 4, TextDim, Dim);
        Factors("modulation.1", 2, Dim, 4 * Dim);
        Factors("time_text_embed.timestep_embedder.linear_2", 4, Dim, Dim);
        Factors("proj_out", 4, Dim, Channels);
        string path = Path.Combine(dir, "tiny-lora.safetensors");
        var header = new JsonObject();
        long offset = 0;
        var blobs = new List<byte[]>();
        foreach (var (name, shape, data) in tensors)
        {
            var raw = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            var dims = new JsonArray();
            foreach (long d in shape) dims.Add(d);
            header[name] = new JsonObject { ["dtype"] = "F32", ["shape"] = dims, ["data_offsets"] = new JsonArray(offset, offset + raw.Length) };
            offset += raw.Length;
            blobs.Add(raw);
        }
        byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)json.Length));
        fs.Write(json);
        foreach (byte[] blob in blobs) fs.Write(blob);
        return path;
    }

    // ---- the retired engines' outputs, frozen (QwenImage21Goldens)

    /// <summary>The frozen cases: (name, transformer layout, LoRA, references).</summary>
    internal static readonly (string Name, bool? Fused, bool Lora, int References)[] GoldenCases =
    {
        ("mixed/refs0", null, false, 0), ("mixed/refs1", null, false, 1), ("mixed/refs2", null, false, 2),
        ("lora-fused/refs0", true, true, 0), ("lora-fused/refs1", true, true, 1), ("lora-separate/refs1", false, true, 1),
    };

    private (string Dit, string Lora, string Sha) Fixture(bool? fused, bool lora)
    {
        string dit = WriteTransformer(_dir, fused);
        string loraPath = lora ? WriteLora(_dir) : null;
        return (dit, loraPath, lora ? QwenImage21Goldens.FixtureSha(dit, loraPath) : QwenImage21Goldens.FixtureSha(dit));
    }

    /// <summary>The runtime's program on the pinned GGML backend against the retired native graph's frozen velocities: the bytes
    /// on the lane that froze them, within 2e-3 elsewhere (another CPU's or GPU's kernels).</summary>
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)] [InlineData(3)] [InlineData(4)] [InlineData(5)]
    public void Program_matches_the_frozen_legacy_graph(int index)
    {
        var (name, fused, withLora, references) = GoldenCases[index];
        var (dit, loraPath, sha) = Fixture(fused, withLora);
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var denoiser = new QwenImage21Denoiser(dit, backend) { FreeMemory = () => -1 };
        using var set = withLora ? QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, denoiser.Weights.Gguf, "", backend, 1) : null;
        if (set != null) Assert.Equal(fused, set.SplitGateUp[1]);
        var (text, refs, layout, target) = Request(references, 31);
        var got = new List<float[]>();
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            float[] v = (float[])denoiser.Predict(x, text, refs, layout, sigma, step, false, set).Clone();
            // The first step stores the text and reference prefix's K/V; the later ones compute only the target.
            Assert.Equal(step == 0 ? QwenImage21Denoiser.Mode.Extract : QwenImage21Denoiser.Mode.Cached, denoiser.LastMode);
            got.Add(v);
            QwenImage21Goldens.Advance(x, v, sigma);
        }
        if (set != null)
        {
            // The plug-in changes the velocity, so the comparison is not vacuous.
            float[] plain = (float[])denoiser.Predict(target, text, refs, layout, 1f, 0, false, null).Clone();
            float[] adapted = (float[])denoiser.Predict(target, text, refs, layout, 1f, 0, false, set).Clone();
            Assert.NotEqual(plain, adapted);
        }
        denoiser.FinishRequest();
        QwenImage21Goldens.Check("denoiser/" + name, MiniMaxH3ImageGoldens.LaneKey(backend), sha, got, bound: 2e-3);
    }

    /// <summary>A prefix cache belongs to the request's own conditioning and reference objects (the K/V encode them, and a
    /// caller may mutate an equal copy independently): a copy, another order or another count of the references extracts its
    /// own; two are kept (a CFG pair), the least recently used goes first, and FinishRequest drops them all.</summary>
    [Fact]
    public void Prefix_caches_follow_the_requests_own_conditioning_and_references()
    {
        string path = WriteTransformer(_dir);
        using var denoiser = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend) { FreeMemory = () => -1 };
        var (text, refs, layout, target) = Request(2, 41);
        QwenImage21Denoiser.Mode Step(ConditioningSet c, LatentRef[] r)
        {
            denoiser.Predict(target, c, r, layout, 0.5f, 1, false, null);
            return denoiser.LastMode;
        }
        var copy = new ConditioningSet((float[])text.Hidden.Clone(), text.Rows, text.Width, (int[])text.Modalities.Clone());
        Assert.Equal(QwenImage21Denoiser.Mode.Extract, Step(text, refs));
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, Step(text, refs));
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, Step(text, new[] { refs[0], refs[1] }));
        Assert.Equal(QwenImage21Denoiser.Mode.Extract, Step(copy, refs));
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, Step(text, refs));
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, Step(copy, refs));
        // A third prefix evicts the least recently used (text): the copy's stays.
        Assert.Equal(QwenImage21Denoiser.Mode.Extract, Step(text, new[] { refs[1], refs[0] }));
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, Step(copy, refs));
        Assert.Equal(QwenImage21Denoiser.Mode.Extract, Step(text, refs));
        denoiser.FinishRequest();
        Assert.Equal(QwenImage21Denoiser.Mode.Extract, Step(copy, refs));
        denoiser.FinishRequest();
    }

    /// <summary>A cache larger than half the free memory is declined: every step recomputes the prefix ("full"), with the
    /// velocities the cached steps give. On ggml-cpu, to float rounding. A GPU backend picks its kernels, and how they split
    /// a product, by the rows the product computes, and the cached graph computes fewer: the retired engine's whole-sequence
    /// steps were 2.3e-3 to 4.6e-3 from its cached ones on ggml-cuda (Turbo Q4_K_M at 512 x 512, RTX 4000 Ada), and this
    /// fixture's are 8.4e-4 apart there.</summary>
    [Fact]
    public void A_declined_prefix_cache_recomputes_the_prefix_every_step()
    {
        string path = WriteTransformer(_dir);
        double bound = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-5 : 2e-3;
        using var cached = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend) { FreeMemory = () => -1 };
        using var declined = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend) { FreeMemory = () => 1024 };
        var (text, refs, layout, target) = Request(1, 43);
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            float[] want = (float[])cached.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
            float[] got = (float[])declined.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
            Assert.Equal(QwenImage21Denoiser.Mode.Full, declined.LastMode);
            double err = 0, norm = 0;
            for (int i = 0; i < want.Length; i++) { err += (got[i] - want[i]) * (double)(got[i] - want[i]); norm += want[i] * (double)want[i]; }
            Assert.True(Math.Sqrt(err / norm) < bound, $"step {step}: relative L2 {Math.Sqrt(err / norm):E3} between the full and cached graphs");
            QwenImage21Goldens.Advance(x, want, sigma);
        }
        cached.FinishRequest();
        declined.FinishRequest();
    }

    /// <summary>A per-step head bundle (a VideoX-Fun PDD LoRA) replaces proj_out with each step's own head: the program
    /// projects step s through head s, so with head 1 twice head 0 the second step's velocity is exactly twice the first's for
    /// the same input (scaling by two is exact), and a step the bundle has no head for is refused. The prefix cache is declined
    /// so both steps run the same whole-sequence graph on every backend; on the GGML executor and the managed one.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void A_per_step_head_bundle_projects_each_step_through_its_own_head(bool managed)
    {
        string path = WriteTransformer(_dir);
        int headValues = QwenImage21Weights.ReleaseChannels * QwenImage21Weights.ReleaseDim;
        float[] first = Rand(headValues, 700, 0.05f);
        var heads = new float[2 * headValues];
        for (int i = 0; i < headValues; i++) { heads[i] = first[i]; heads[headValues + i] = 2 * first[i]; }
        string bundle = WritePddBundle(_dir, heads, steps: 2);

        BackendType backend = managed ? BackendType.Cpu : TestGates.PinnedGgmlBackend;
        using var denoiser = new QwenImage21Denoiser(path, backend) { FreeMemory = () => 1024 };
        using var set = QwenImage21LoraSet.Load(new[] { new LoraSpec(bundle) }, denoiser.Weights.Gguf, "", backend, 1);
        Assert.Equal(2, set.OutputHeads.Length);
        var (text, refs, layout, target) = Request(0, 47);
        float[] v0 = (float[])denoiser.Predict(target, text, refs, layout, 0.5f, 0, false, set).Clone();
        Assert.Equal(QwenImage21Denoiser.Mode.Full, denoiser.LastMode);
        float[] v1 = (float[])denoiser.Predict(target, text, refs, layout, 0.5f, 1, false, set).Clone();
        Assert.Contains(v0, v => v != 0);
        Assert.Equal(v0.Select(v => 2 * v).ToArray(), v1);
        var ex = Assert.Throws<ArgumentOutOfRangeException>(() => denoiser.Predict(target, text, refs, layout, 0.5f, 2, false, set));
        Assert.Contains("2 step heads; step 3 has none", ex.Message, StringComparison.Ordinal);
        denoiser.FinishRequest();
    }

    /// <summary>A VideoX-Fun PDD bundle holding only per-step output heads ([steps, channels, dim] at the released
    /// checkpoint's width), with the pdd_config.json beside it that declares the format.</summary>
    private static string WritePddBundle(string dir, float[] heads, int steps)
    {
        string folder = Path.Combine(dir, "pdd-heads-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(folder);
        string sigmas = string.Join(", ", Enumerable.Range(0, steps + 1)
            .Select(i => (1.0 - (double)i / steps).ToString(System.Globalization.CultureInfo.InvariantCulture)));
        File.WriteAllText(Path.Combine(folder, "pdd_config.json"),
            $$"""{ "pdd_num_steps": {{steps}}, "pdd_block_size": 1, "pdd_sigmas": [{{sigmas}}], "pdd_full_parameters": "['proj_out.weight']", "lora_alpha": 8 }""");
        var raw = new byte[heads.Length * 4];
        Buffer.BlockCopy(heads, 0, raw, 0, raw.Length);
        var header = new JsonObject
        {
            ["__metadata__"] = new JsonObject { ["format"] = "qwenimage21_extracted_prefused_v1" },
            ["proj_out.weight"] = new JsonObject
            {
                ["dtype"] = "F32",
                ["shape"] = new JsonArray(steps, QwenImage21Weights.ReleaseChannels, QwenImage21Weights.ReleaseDim),
                ["data_offsets"] = new JsonArray(0, raw.Length),
            },
        };
        string path = Path.Combine(folder, "bundle.safetensors");
        byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)json.Length));
        fs.Write(json);
        fs.Write(raw);
        return path;
    }

    /// <summary>The managed tensor executor (the cpu backend) against the retired pure-C# transformer's frozen velocities.</summary>
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void Managed_program_matches_the_frozen_pure_csharp_transformer(int index)
    {
        var (name, fused, _, references) = GoldenCases[index];
        var (dit, _, sha) = Fixture(fused, false);
        using var denoiser = new QwenImage21Denoiser(dit, BackendType.Cpu) { FreeMemory = () => -1 };
        var (text, refs, layout, target) = Request(references, 31);
        var got = new List<float[]>();
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            float[] v = (float[])denoiser.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
            got.Add(v);
            QwenImage21Goldens.Advance(x, v, sigma);
        }
        denoiser.FinishRequest();
        QwenImage21Goldens.Check("managed/" + name, MiniMaxH3ImageGoldens.HostLaneKey(), sha, got, bound: 1e-4);
    }

    /// <summary>The frozen cases on <see cref="QwenImage21WideFixture"/> (the released head and conditioning widths): the plain
    /// transformer, and a plug-in with DoRA magnitudes and replaced norm gains over a fused and a separate MLP.</summary>
    internal static readonly (string Name, bool Fused, bool Lora, int References)[] WideCases =
    {
        ("wide-plain/refs1", true, false, 1), ("wide-lora-fused/refs1", true, true, 1), ("wide-lora-separate/refs0", false, true, 0),
    };

    private List<float[]> Trajectory(QwenImage21Denoiser denoiser, (ConditioningSet Text, LatentRef[] References, QwenImage21Layout Layout, float[] Target) request,
        QwenImage21LoraSet set)
    {
        var got = new List<float[]>();
        float[] x = (float[])request.Target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            float[] v = (float[])denoiser.Predict(x, request.Text, request.References, request.Layout, sigma, step, false, set).Clone();
            Assert.Equal(step == 0 ? QwenImage21Denoiser.Mode.Extract : QwenImage21Denoiser.Mode.Cached, denoiser.LastMode);
            got.Add(v);
            QwenImage21Goldens.Advance(x, v, sigma);
        }
        denoiser.FinishRequest();
        return got;
    }

    /// <summary>The runtime's program with DoRA row scales and norm replacements against the retired native graph.</summary>
    [Theory]
    [InlineData(0)] [InlineData(1)] [InlineData(2)]
    public void Program_matches_the_frozen_legacy_graph_on_the_wide_transformer(int index)
    {
        var (name, fused, withLora, references) = WideCases[index];
        string dit = QwenImage21WideFixture.WriteTransformer(_dir, fused);
        string loraPath = withLora ? QwenImage21WideFixture.WriteLora(_dir) : null;
        string sha = withLora ? QwenImage21Goldens.FixtureSha(dit, loraPath) : QwenImage21Goldens.FixtureSha(dit);
        BackendType backend = TestGates.PinnedGgmlBackend;
        using var denoiser = new QwenImage21Denoiser(dit, backend) { FreeMemory = () => -1 };
        using var set = withLora ? QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, denoiser.Weights.Gguf, "", backend, 1) : null;
        if (set != null)
        {
            Assert.NotEqual(IntPtr.Zero, set.TextNorm);
            Assert.NotEqual(IntPtr.Zero, set.NormQ[0]);
            Assert.NotEqual(IntPtr.Zero, set.NormK[1]);
            Assert.Contains("2 DoRA magnitude(s)", set.Summary, StringComparison.Ordinal);
        }
        var got = Trajectory(denoiser, QwenImage21WideFixture.Request(references, 31), set);
        QwenImage21Goldens.Check("denoiser/" + name, MiniMaxH3ImageGoldens.LaneKey(backend), sha, got, bound: 2e-3);
    }

    /// <summary>The managed tensor executor (the cpu backend) with LoRA plug-ins (F32 factors, as the cpu backend loads
    /// them) against the retired pure-C# transformer's frozen velocities.</summary>
    [Theory]
    [InlineData("lora-fused/refs1")] [InlineData("lora-separate/refs1")]
    [InlineData("wide-plain/refs1")] [InlineData("wide-lora-fused/refs1")] [InlineData("wide-lora-separate/refs0")]
    public void Managed_program_matches_the_frozen_pure_csharp_transformer_with_plug_ins(string name)
    {
        bool wide = name.StartsWith("wide-", StringComparison.Ordinal);
        bool withLora = name.Contains("lora", StringComparison.Ordinal);
        bool fused = !name.Contains("separate", StringComparison.Ordinal);
        int references = name.EndsWith("refs1", StringComparison.Ordinal) ? 1 : 0;
        string dit = wide ? QwenImage21WideFixture.WriteTransformer(_dir, fused) : WriteTransformer(_dir, fused);
        string loraPath = !withLora ? null : wide ? QwenImage21WideFixture.WriteLora(_dir) : WriteLora(_dir);
        string sha = withLora ? QwenImage21Goldens.FixtureSha(dit, loraPath) : QwenImage21Goldens.FixtureSha(dit);
        using var denoiser = new QwenImage21Denoiser(dit, BackendType.Cpu) { FreeMemory = () => -1 };
        using var set = withLora ? QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, denoiser.Weights.Gguf, "", BackendType.Cpu, 1) : null;
        var got = Trajectory(denoiser, wide ? QwenImage21WideFixture.Request(references, 31) : Request(references, 31), set);
        QwenImage21Goldens.Check("managed/" + name, MiniMaxH3ImageGoldens.HostLaneKey(), sha, got, bound: 1e-4);
    }

    /// <summary>On a fused gate_up checkpoint a split LoRA (separate gate and up factors) describes the projection's first half
    /// at its own address: a denoiser that ran without the plug-in, then with it, then without it again, gives each step a
    /// fresh denoiser's velocities, on the managed executor (which binds a leaf by its weight object's shape) and on GGML.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void Switching_a_split_LoRA_on_a_fused_checkpoint_rebinds_the_projection(bool managed)
    {
        string dit = WriteTransformer(_dir, fused: true);
        string loraPath = WriteLora(_dir);
        BackendType backend = managed ? BackendType.Cpu : TestGates.PinnedGgmlBackend;
        using var switching = new QwenImage21Denoiser(dit, backend) { FreeMemory = () => -1 };
        using var set = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, switching.Weights.Gguf, "", backend, 1);
        Assert.True(set.SplitGateUp[0]);
        var (text, refs, layout, target) = Request(1, 47);
        float[] Fresh(QwenImage21LoraSet lora)
        {
            using var fresh = new QwenImage21Denoiser(dit, backend) { FreeMemory = () => -1 };
            using var own = lora == null ? null : QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, fresh.Weights.Gguf, "", backend, 1);
            float[] v = (float[])fresh.Predict(target, text, refs, layout, 1f, 0, false, own).Clone();
            fresh.FinishRequest();
            return v;
        }
        float[] Step(QwenImage21LoraSet lora)
        {
            float[] v = (float[])switching.Predict(target, text, refs, layout, 1f, 0, false, lora).Clone();
            switching.FinishRequest();
            return v;
        }
        float[] plain = Fresh(null), adapted = Fresh(set);
        Assert.NotEqual(plain, adapted);
        Assert.Equal(plain, Step(null));
        Assert.Equal(adapted, Step(set));
        Assert.Equal(plain, Step(null));
    }

    // ---- whole-block stages (--layer-split N)

    /// <summary>A CFG pair over one request's references and target: its own conditioning, and a negative prompt's with three
    /// more text rows (another geometry, so the branches compile programs of their own), each with its layout.</summary>
    private static (ConditioningSet Text, QwenImage21Layout Layout)[] CfgBranches(
        (ConditioningSet Text, LatentRef[] References, QwenImage21Layout Layout, float[] Target) request, int headDim, int seed)
    {
        ConditioningSet text = request.Text;
        var slots = new List<int>(text.Modalities);
        slots.AddRange(new[] { 0, 0, 0 });
        var negative = new ConditioningSet(Rand(slots.Count * text.Width, seed, 1f), slots.Count, text.Width, slots.ToArray());
        var shapes = new List<(int, int)>();
        foreach (LatentRef r in request.References) shapes.Add((r.Shape.Height, r.Shape.Width));
        shapes.Add((4, 6));
        var layout = QwenImage21Layout.Build(slots.Count, request.References.Length > 0 ? slots.ToArray() : null, shapes, headDim);
        return new[] { (text, request.Layout), (negative, layout) };
    }

    /// <summary>A guided request's predictions: both branches at every step of QwenImage21Goldens.Steps (the state moved by
    /// the CFG 4 velocity), then the request finished; the modes the predictions ran beside them.</summary>
    private static (List<float[]> Velocities, List<QwenImage21Denoiser.Mode> Modes) CfgTrajectory(QwenImage21Denoiser denoiser,
        (ConditioningSet Text, QwenImage21Layout Layout)[] branches, LatentRef[] references, float[] target, QwenImage21LoraSet set)
    {
        var velocities = new List<float[]>();
        var modes = new List<QwenImage21Denoiser.Mode>();
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            var pair = new List<float[]>();
            foreach (var (text, layout) in branches)
            {
                pair.Add((float[])denoiser.Predict(x, text, references, layout, sigma, step, false, set).Clone());
                modes.Add(denoiser.LastMode);
            }
            velocities.AddRange(pair);
            var guided = new float[x.Length];
            for (int k = 0; k < guided.Length; k++) guided[k] = pair[1][k] + 4f * (pair[0][k] - pair[1][k]);
            QwenImage21Goldens.Advance(x, guided, sigma);
        }
        denoiser.FinishRequest();
        return (velocities, modes);
    }

    private static void AssertBitExact(List<float[]> want, List<float[]> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int i = 0; i < want.Count; i++)
        {
            Assert.Equal(want[i].Length, got[i].Length);
            int differ = 0;
            for (int k = 0; k < want[i].Length; k++)
                if (BitConverter.SingleToInt32Bits(want[i][k]) != BitConverter.SingleToInt32Bits(got[i][k])) differ++;
            Assert.True(differ == 0, $"{what} prediction {i + 1} (step {i / 2 + 1}, branch {i % 2}): {differ} of {want[i].Length} values differ");
        }
    }

    /// <summary>The largest error relative to the largest value, and the relative L2 error.</summary>
    private static (double Rel, double RelL2) Compare(float[] want, float[] got)
    {
        double scale = want.Max(v => Math.Abs((double)v)), worst = 0, l2 = 0, n2 = 0;
        for (int k = 0; k < want.Length; k++)
        {
            Assert.True(float.IsFinite(got[k]), $"value {k} is {got[k]}");
            double d = got[k] - (double)want[k];
            worst = Math.Max(worst, Math.Abs(d));
            l2 += d * d;
            n2 += (double)want[k] * want[k];
        }
        return (worst / scale, Math.Sqrt(l2 / n2));
    }

    /// <summary>
    /// The transformer's blocks over two host "devices" as whole-block stages (DenoiserStages: one program a stage, the packed
    /// stream, the shared modulation and the time embedding handed on as F32 rows), on the managed executor: every prediction
    /// of a guided request (both CFG branches, through the prefix cache's extract and cached programs, or with the cache
    /// declined) equals one device's bit for bit, with and without a LoRA plug-in (each stage binds its own blocks' factors;
    /// the first the input, time and modulation ones, the last proj_out's), and after the plug-in is unbound again. Each stage
    /// compiles what one device compiles, once a program: two programs a branch with the cache, one without.
    /// </summary>
    [Theory]
    [InlineData(false, 2, false)]
    [InlineData(true, 1, false)]
    [InlineData(false, 0, true)]
    public void WholeBlockStagesMatchOneDeviceOnTheManagedExecutor(bool wide, int references, bool declined)
    {
        // The small plug-in's gate and up factors over a fused checkpoint (split into row views) and over a separate one.
        string dit = wide ? QwenImage21WideFixture.WriteTransformer(_dir, fused: true) : WriteTransformer(_dir, fused: !declined);
        string loraPath = wide ? QwenImage21WideFixture.WriteLora(_dir) : WriteLora(_dir);
        Func<long> free = declined ? () => 1024 : () => -1;
        using var one = new QwenImage21Denoiser(dit, BackendType.Cpu) { FreeMemory = free };
        var devices = new IAllocator[] { new CpuAllocator(BlasEnum.DotNet), new CpuAllocator(BlasEnum.DotNet) };
        using var staged = new QwenImage21Denoiser(dit, BackendType.Cpu, devices) { FreeMemory = free };
        Assert.Equal(2, staged.Stages.Count);
        Assert.Equal((0, 1), (staged.Stages.First(0), staged.Stages.End(0)));
        Assert.Equal((1, 2), (staged.Stages.First(1), staged.Stages.End(1)));
        // The stages bind every leaf once between them, and a device holds the larger share.
        var shares = new[] { staged.Weights.WeightsOf(staged.Stages.Span(0)), staged.Weights.WeightsOf(staged.Stages.Span(1)) };
        Assert.Equal(staged.Weights.All.OrderBy(w => w.Data).Select(w => w.Data), shares.SelectMany(s => s).OrderBy(w => w.Data).Select(w => w.Data));
        Assert.Equal(shares.Max(s => s.Sum(w => w.Bytes)), staged.WeightBytes);
        Assert.True(staged.WeightBytes < one.WeightBytes);

        var request = wide ? QwenImage21WideFixture.Request(references, 31) : Request(references, 31);
        var branches = CfgBranches(request, wide ? QwenImage21WideFixture.HeadDim : HeadDim, 77);
        var (want, wantModes) = CfgTrajectory(one, branches, request.References, request.Target, null);
        var (got, gotModes) = CfgTrajectory(staged, branches, request.References, request.Target, null);
        var expectedModes = QwenImage21Goldens.Steps.SelectMany(s => Enumerable.Repeat(declined ? QwenImage21Denoiser.Mode.Full
            : s.Step == 0 ? QwenImage21Denoiser.Mode.Extract : QwenImage21Denoiser.Mode.Cached, 2));
        Assert.Equal(expectedModes, wantModes);
        Assert.Equal(wantModes, gotModes);
        AssertBitExact(want, got, "stages, no LoRA");
        // Once a program: with the cache each branch's extract and cached programs, without it each branch's one (two retained
        // plans a stage, as one device keeps, so the alternating branches replay theirs).
        int programs = declined ? 2 : 4;
        Assert.Equal(programs, one.Compiles);
        Assert.Equal((programs, programs), (staged.StageCompiles(0), staged.StageCompiles(1)));
        Assert.Equal(2 * programs, staged.Compiles);

        using var oneSet = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, one.Weights.Gguf, "", BackendType.Cpu, 1);
        using var stagedSet = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, staged.Weights.Gguf, "", BackendType.Cpu, 1);
        var (wantLora, _) = CfgTrajectory(one, branches, request.References, request.Target, oneSet);
        var (gotLora, _) = CfgTrajectory(staged, branches, request.References, request.Target, stagedSet);
        Assert.NotEqual(want[0], wantLora[0]);   // the plug-in moves the velocity: the comparison is not vacuous
        AssertBitExact(wantLora, gotLora, "stages, LoRA");
        // Unbound again: no stage keeps the set's plans or factors.
        var (wantAfter, _) = CfgTrajectory(one, branches, request.References, request.Target, null);
        var (gotAfter, _) = CfgTrajectory(staged, branches, request.References, request.Target, null);
        AssertBitExact(want, gotAfter, "stages after the LoRA");
        AssertBitExact(wantAfter, gotAfter, "stages after the LoRA (one device)");
        Assert.Equal((one.Compiles, one.Compiles), (staged.StageCompiles(0), staged.StageCompiles(1)));
    }

    /// <summary>
    /// The same over two GPUs of one ggml_cuda context (the layer split's: collectives off), each stage's programs compiled and
    /// run with its GPU active and its weights preloaded there: every prediction of a guided request through the prefix cache
    /// (F16 keys and values, each stage's own layers on its own GPU) equals one GPU's within 1e-5 of the largest value, plain
    /// and with a LoRA plug-in, on the small transformer (explicit attention: ggml-cuda's flash attention takes no 32-wide
    /// head) and the wide one (flash attention, 128-wide heads, DoRA and norm replacements). The measured errors are logged.
    /// </summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData(false)]
    [InlineData(true)]
    public void WholeBlockStagesOnTwoGpusMatchOneGpu(bool wide)
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        string dit = wide ? QwenImage21WideFixture.WriteTransformer(_dir, fused: true) : WriteTransformer(_dir, fused: true);
        string loraPath = wide ? QwenImage21WideFixture.WriteLora(_dir) : WriteLora(_dir);
        var request = wide ? QwenImage21WideFixture.Request(1, 31) : Request(1, 31);
        var branches = CfgBranches(request, wide ? QwenImage21WideFixture.HeadDim : HeadDim, 77);
        List<float[]> want, wantLora;
        using (var one = new QwenImage21Denoiser(dit, BackendType.GgmlCuda) { FreeMemory = () => -1 })
        using (var set = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, one.Weights.Gguf, "", BackendType.GgmlCuda, 1))
        {
            want = CfgTrajectory(one, branches, request.References, request.Target, null).Velocities;
            wantLora = CfgTrajectory(one, branches, request.References, request.Target, set).Velocities;
        }
        var context = new GgmlContext(new[] { 0, 1 }, GgmlBackendType.Cuda, enableCollectives: false);   // as a layer-split model makes it
        var devices = new IAllocator[] { new GgmlAllocator(context, 0), new GgmlAllocator(context, 1) };
        using var staged = new QwenImage21Denoiser(dit, BackendType.GgmlCuda, devices) { FreeMemory = () => -1 };
        Assert.Equal(new[] { 0, 1 }, staged.Stages.Devices);
        Assert.True(staged.Preload() > 0);
        using var stagedSet = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, staged.Weights.Gguf, "", BackendType.GgmlCuda, 1);
        var (got, modes) = CfgTrajectory(staged, branches, request.References, request.Target, null);
        Assert.Equal(QwenImage21Denoiser.Mode.Cached, modes[^1]);
        List<float[]> gotLora = CfgTrajectory(staged, branches, request.References, request.Target, stagedSet).Velocities;
        for (int i = 0; i < want.Count; i++)
        {
            var (rel, relL2) = Compare(want[i], got[i]);
            var (relLora, relL2Lora) = Compare(wantLora[i], gotLora[i]);
            _output.WriteLine($"{(wide ? "wide" : "small")} step {i / 2 + 1} branch {i % 2}: stages vs one GPU max {rel:E3} (L2 {relL2:E3}); " +
                              $"with LoRA max {relLora:E3} (L2 {relL2Lora:E3})");
            Assert.True(rel <= 1e-5 && relLora <= 1e-5, $"prediction {i + 1}: {rel:E3} / {relLora:E3}");
        }
    }
}
