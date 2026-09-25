// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

/// <summary>
/// End-to-end tests of <see cref="QwenImage21LoraSet.Load"/> on synthetic files: a tiny
/// transformer GGUF (8-wide projections, the real module names) and hand-written LoRA
/// safetensors. The packed native factors are read back from the
/// <see cref="QwenImage21Adapter"/> descriptors and the reconstructed update
/// (up · down) is compared with the intended strength * alpha / rank * B · A.
/// </summary>
public sealed class QwenImage21LoraSetTests : IDisposable
{
    private const string Prefix = "model.diffusion_model.";
    private const int D = 8;          // hidden width of the synthetic transformer
    private const int Ff = 12;        // MLP width (gate and up halves)
    private const int ModOut = 32;    // modulation.1 output
    private const int HeadDim = QwenImage21DiT.HeadDim;
    private const int F16 = 1, F32 = 0;

    private readonly string _dir;
    private readonly Dictionary<string, float[]> _base = new(StringComparer.Ordinal);

    public QwenImage21LoraSetTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-lora-set-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ---- synthetic files ------------------------------------------------------------------

    private static float[] Rand(int n, int seed, float scale = 1f, float offset = 0f)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = offset + scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    /// <summary>A minimal GGUF v3: no metadata, F32 tensors, 32-byte alignment.</summary>
    private static void WriteGguf(string path, IReadOnlyList<(string Name, long[] Dims, float[] Data)> tensors)
    {
        const int Alignment = 32;
        static long Pad(long n) => (n + Alignment - 1) / Alignment * Alignment;
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs, Encoding.UTF8);
        w.Write(0x46554747u);            // "GGUF"
        w.Write(3u);
        w.Write((ulong)tensors.Count);
        w.Write(0UL);                    // no key/value pairs
        long offset = 0;
        foreach (var (name, dims, data) in tensors)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(name);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
            w.Write((uint)dims.Length);
            foreach (long d in dims) w.Write((ulong)d);
            w.Write(0u);                 // GGML_TYPE_F32
            w.Write((ulong)offset);
            offset += Pad(data.Length * 4L);
        }
        w.Flush();
        w.Write(new byte[Pad(fs.Position) - fs.Position]);
        foreach (var (_, _, data) in tensors)
        {
            var raw = new byte[data.Length * 4];
            Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
            w.Write(raw);
            w.Write(new byte[Pad(raw.Length) - raw.Length]);
        }
    }

    /// <summary>
    /// The synthetic transformer: block 0 has q/k/v/out [8 -> 8], the MLP (fused gate_up
    /// [8 -> 24], gate rows first, or separate gate_layer/proj [8 -> 12]) and out [12 -> 8],
    /// the q/k norm gains; globally modulation.1 [8 -> 32].
    /// </summary>
    private GgufFile Transformer(bool fused = true)
    {
        var tensors = new List<(string, long[], float[])>();
        void Add(string module, long input, long output, int seed)
        {
            var data = Rand((int)(input * output), seed, 0.5f);
            _base[module] = data;
            tensors.Add((Prefix + module + ".weight", new[] { input, output }, data));
        }
        Add("transformer_blocks.0.attn.to_q", D, D, 101);
        Add("transformer_blocks.0.attn.to_k", D, D, 102);
        Add("transformer_blocks.0.attn.to_v", D, D, 103);
        Add("transformer_blocks.0.attn.to_out.0", D, D, 104);
        if (fused)
        {
            Add("transformer_blocks.0.img_mlp.gate_up", D, 2 * Ff, 105);
        }
        else
        {
            Add("transformer_blocks.0.img_mlp.gate_layer", D, Ff, 106);
            Add("transformer_blocks.0.img_mlp.proj", D, Ff, 107);
        }
        Add("transformer_blocks.0.img_mlp.out", Ff, D, 108);
        Add("modulation.1", D, ModOut, 109);
        Add("proj_out", D, Channels, 112);
        foreach (var (norm, seed) in new[] { ("transformer_blocks.0.attn.norm_q", 110), ("transformer_blocks.0.attn.norm_k", 111) })
        {
            var gain = Rand(HeadDim, seed, 0.25f, 1f);
            _base[norm] = gain;
            tensors.Add((Prefix + norm + ".weight", new long[] { HeadDim }, gain));
        }
        string path = Path.Combine(_dir, (fused ? "fused" : "unfused") + "-dit.gguf");
        WriteGguf(path, tensors);
        return new GgufFile(path);
    }

    private sealed class LoraFile
    {
        internal readonly List<(string Name, long[] Shape, float[] Data, string Dtype)> Tensors = new();
        internal readonly Dictionary<string, string> Metadata = new(StringComparer.Ordinal);

        internal LoraFile Add(string name, long[] shape, float[] data, string dtype = "F32")
        {
            Tensors.Add((name, shape, data, dtype));
            return this;
        }

        /// <summary>Adds lora_A [rank, in] / lora_B [out, rank] under the given key spellings; returns (A, B).</summary>
        internal (float[] A, float[] B) Factors(string downKey, string upKey, int rank, int input, int output, int seed)
        {
            var a = Rand(rank * input, seed);
            var b = Rand(output * rank, seed + 1000);
            Add(downKey, new long[] { rank, input }, a);
            Add(upKey, new long[] { output, rank }, b);
            return (a, b);
        }

        internal LoraFile Alpha(string key, float value) => Add(key, Array.Empty<long>(), new[] { value });
    }

    private static byte[] Encode(float[] data, string dtype)
    {
        switch (dtype)
        {
            case "F32":
            {
                var raw = new byte[data.Length * 4];
                Buffer.BlockCopy(data, 0, raw, 0, raw.Length);
                return raw;
            }
            case "BF16":
            {
                var raw = new byte[data.Length * 2];
                for (int i = 0; i < data.Length; i++)
                    BitConverter.TryWriteBytes(raw.AsSpan(i * 2), (ushort)(BitConverter.SingleToUInt32Bits(data[i]) >> 16));
                return raw;
            }
            case "F16":
            {
                var raw = new byte[data.Length * 2];
                for (int i = 0; i < data.Length; i++)
                    BitConverter.TryWriteBytes(raw.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((Half)data[i]));
                return raw;
            }
            default: throw new ArgumentException(dtype);
        }
    }

    private string Save(LoraFile file, string name)
    {
        string path = Path.Combine(_dir, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var header = new JsonObject();
        if (file.Metadata.Count > 0)
        {
            var meta = new JsonObject();
            foreach (var (k, v) in file.Metadata) meta[k] = v;
            header["__metadata__"] = meta;
        }
        var blobs = new List<byte[]>();
        long offset = 0;
        foreach (var (tensor, shape, data, dtype) in file.Tensors)
        {
            byte[] blob = Encode(data, dtype);
            var dims = new JsonArray();
            foreach (long d in shape) dims.Add(d);
            header[tensor] = new JsonObject
            {
                ["dtype"] = dtype,
                ["shape"] = dims,
                ["data_offsets"] = new JsonArray(offset, offset + blob.Length),
            };
            offset += blob.Length;
            blobs.Add(blob);
        }
        byte[] json = Encoding.UTF8.GetBytes(header.ToJsonString());
        using var fs = File.Create(path);
        fs.Write(BitConverter.GetBytes((ulong)json.Length));
        fs.Write(json);
        foreach (var blob in blobs) fs.Write(blob);
        return path;
    }

    private string WriteText(string name, string text)
    {
        string path = Path.Combine(_dir, name);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
        return path;
    }

    // ---- reading the packed native factors back ----------------------------------------------

    private static QwenImage21Adapter AdapterOf(QwenImage21LoraSet set, int rank = 0) =>
        Marshal.PtrToStructure<QwenImage21Adapter>(set.AdapterFor(rank));

    private static QwenImage21BlockLora BlockOf(QwenImage21LoraSet set, int block, int rank = 0) =>
        Marshal.PtrToStructure<QwenImage21BlockLora>(AdapterOf(set, rank).Blocks + block * Marshal.SizeOf<QwenImage21BlockLora>());

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

    private static float[] DownOf(QwenImage21Lora l) => ReadValues(l.Down, l.Rank * l.In, l.Type);
    private static float[] UpOf(QwenImage21Lora l) => ReadValues(l.Up, l.Out * l.Rank, l.Type);
    private static float[] RowScaleOf(QwenImage21Lora l) => ReadValues(l.RowScale, l.Out, F32);

    /// <summary>up [out, rank] · down [rank, in], as the native graph applies it.</summary>
    private static float[] Delta(QwenImage21Lora l)
    {
        var down = DownOf(l);
        var up = UpOf(l);
        var result = new float[l.Out * l.In];
        for (long o = 0; o < l.Out; o++)
            for (int r = 0; r < l.Rank; r++)
            {
                double u = up[o * l.Rank + r];
                for (long i = 0; i < l.In; i++) result[o * l.In + i] += (float)(u * down[r * l.In + i]);
            }
        return result;
    }

    /// <summary>scale * B [out, rank] · A [rank, in].</summary>
    private static float[] Product(float[] b, float[] a, int output, int rank, int input, float scale = 1f)
    {
        var result = new float[output * input];
        for (int o = 0; o < output; o++)
            for (int r = 0; r < rank; r++)
            {
                double bv = b[o * rank + r];
                for (int i = 0; i < input; i++) result[o * input + i] += (float)(bv * a[r * input + i]);
            }
        for (int i = 0; i < result.Length; i++) result[i] *= scale;
        return result;
    }

    private static float[] Add(float[] x, float[] y) => x.Zip(y, (p, q) => p + q).ToArray();

    private static float[] ScaleRows(float[] m, int input, float[] rows) =>
        m.Select((v, idx) => v * rows[idx / input]).ToArray();

    private static float[] Rows(float[] m, int input, int first, int count) =>
        m.Skip(first * input).Take(count * input).ToArray();

    private static float[] Columns(float[] m, int input, int first, int count)
    {
        int rows = m.Length / input;
        var result = new float[rows * count];
        for (int o = 0; o < rows; o++)
            Array.Copy(m, o * input + first, result, o * count, count);
        return result;
    }

    private static void AssertClose(float[] expected, float[] actual, string what, float relative = 5e-3f)
    {
        Assert.Equal(expected.Length, actual.Length);
        float max = expected.Max(v => Math.Abs(v));
        float tolerance = relative * Math.Max(max, 1e-6f);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance,
                $"{what}[{i}]: expected {expected[i]}, got {actual[i]} (tolerance {tolerance})");
    }

    private static void AssertNear(float expected, float actual, string what, float tolerance = 1e-5f) =>
        Assert.True(Math.Abs(expected - actual) <= tolerance * Math.Max(1f, Math.Abs(expected)),
            $"{what}: expected {expected}, got {actual}");

    private static void AssertEmpty(QwenImage21Lora l)
    {
        Assert.Equal(0, l.Rank);
        Assert.Equal(IntPtr.Zero, l.Down);
        Assert.Equal(IntPtr.Zero, l.Up);
        Assert.Equal(IntPtr.Zero, l.RowScale);
    }

    private static float[] RowNorms(float[] w, int input) =>
        Enumerable.Range(0, w.Length / input)
            .Select(o => (float)Math.Sqrt(w.Skip(o * input).Take(input).Sum(v => (double)v * v))).ToArray();

    private static QwenImage21LoraSet Load(GgufFile dit, int ranks, BackendType backend, params LoraSpec[] specs) =>
        QwenImage21LoraSet.Load(specs, dit, Prefix, backend, ranks);

    private static QwenImage21LoraSet Load(GgufFile dit, params LoraSpec[] specs) =>
        Load(dit, 1, BackendType.GgmlCpu, specs);

    // ---- (a) diffusers: q/k/v stacked, gate_layer / proj on a fused checkpoint ----------------

    [Fact]
    public void Diffusers_SplitMlpHalvesOnAFusedCheckpoint_DescribeTheHalvesSeparately()
    {
        using var dit = Transformer(fused: true);
        var file = new LoraFile();
        var q = file.Factors("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 1);
        var k = file.Factors("transformer.transformer_blocks.0.attn.to_k.lora_A.weight", "transformer.transformer_blocks.0.attn.to_k.lora_B.weight", 2, D, D, 2);
        var v = file.Factors("transformer.transformer_blocks.0.attn.to_v.lora_A.weight", "transformer.transformer_blocks.0.attn.to_v.lora_B.weight", 4, D, D, 3);
        var gate = file.Factors("transformer.transformer_blocks.0.img_mlp.gate_layer.lora_A.weight", "transformer.transformer_blocks.0.img_mlp.gate_layer.lora_B.weight", 4, D, Ff, 4);
        var up = file.Factors("transformer.transformer_blocks.0.img_mlp.proj.lora_A.weight", "transformer.transformer_blocks.0.img_mlp.proj.lora_B.weight", 4, D, Ff, 5);
        string path = Save(file, "diffusers.safetensors");

        using var set = Load(dit, new LoraSpec(path));
        var adapter = AdapterOf(set);
        var block = BlockOf(set, 0);

        Assert.Equal(Marshal.SizeOf<QwenImage21Adapter>(), adapter.StructBytes);
        Assert.Equal(QwenImage21DiT.Layers, adapter.NumLayers);
        Assert.True(set.SplitGateUp[0]);
        Assert.Equal(new bool[QwenImage21DiT.Layers - 1], set.SplitGateUp.Skip(1));

        // q, k, v: one contiguous down allocation (the graph's stacked shrink).
        foreach (var l in new[] { block.Q, block.K, block.V })
        {
            Assert.Equal(F16, l.Type);
            Assert.Equal(16, l.Rank);
            Assert.Equal(D, l.In);
            Assert.Equal(D, l.Out);
            Assert.Equal(IntPtr.Zero, l.RowScale);
        }
        Assert.Equal(block.Q.Down + 16 * D * 2, block.K.Down);
        Assert.Equal(block.K.Down + 16 * D * 2, block.V.Down);
        AssertClose(Product(q.B, q.A, D, 4, D), Delta(block.Q), "q");
        AssertClose(Product(k.B, k.A, D, 2, D), Delta(block.K), "k");
        AssertClose(Product(v.B, v.A, D, 4, D), Delta(block.V), "v");

        // gate and up: each its own 12-row update; their downs are stacked, not shared.
        Assert.Equal(Ff, block.Gate.Out);
        Assert.Equal(Ff, block.Up.Out);
        Assert.Equal(D, block.Gate.In);
        Assert.Equal(block.Gate.Down + block.Gate.Rank * D * 2, block.Up.Down);
        AssertClose(Product(gate.B, gate.A, Ff, 4, D), Delta(block.Gate), "gate");
        AssertClose(Product(up.B, up.A, Ff, 4, D), Delta(block.Up), "up");

        AssertEmpty(block.Out);
        AssertEmpty(block.Down);
        for (int b = 1; b < QwenImage21DiT.Layers; b++)
        {
            var other = BlockOf(set, b);
            foreach (var l in new[] { other.Q, other.K, other.V, other.Out, other.Gate, other.Up, other.Down }) AssertEmpty(l);
        }
        foreach (var g in new[] { adapter.ImageIn, adapter.TextIn, adapter.TextOut, adapter.TimeIn, adapter.TimeOut, adapter.Modulation, adapter.NormOut, adapter.ProjOut })
            AssertEmpty(g);
        Assert.Equal(IntPtr.Zero, adapter.OutputHead);
        Assert.Null(set.Recipe);
        Assert.Empty(set.OutputHeads);
        Assert.True(set.FactorBytes > 0);
        Assert.Contains("diffusers.safetensors: 5 low-rank update(s), rank 2/4", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Diffusers_SplitMlpHalvesOnAnUnfusedCheckpoint_NeedNoSplit()
    {
        using var dit = Transformer(fused: false);
        var file = new LoraFile();
        var gate = file.Factors("transformer_blocks.0.img_mlp.gate_layer.lora_A.weight", "transformer_blocks.0.img_mlp.gate_layer.lora_B.weight", 4, D, Ff, 4);
        var up = file.Factors("transformer_blocks.0.img_mlp.proj.lora_A.weight", "transformer_blocks.0.img_mlp.proj.lora_B.weight", 4, D, Ff, 5);

        using var set = Load(dit, new LoraSpec(Save(file, "unfused.safetensors")));
        var block = BlockOf(set, 0);

        Assert.False(set.SplitGateUp[0]);
        AssertClose(Product(gate.B, gate.A, Ff, 4, D), Delta(block.Gate), "gate");
        AssertClose(Product(up.B, up.A, Ff, 4, D), Delta(block.Up), "up");
        Assert.Equal(block.Gate.Down + block.Gate.Rank * D * 2, block.Up.Down);
    }

    // ---- (b) ComfyUI fused gate_up ----------------------------------------------------------

    private (string Path, float[] Expected) FusedGateUpLora(string name = "comfy.safetensors")
    {
        var file = new LoraFile();
        var f = file.Factors("diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_down.weight",
            "diffusion_model.transformer_blocks.0.img_mlp.gate_up.lora_up.weight", 4, D, 2 * Ff, 7);
        file.Alpha("diffusion_model.transformer_blocks.0.img_mlp.gate_up.alpha", 2f);   // scale 2 / 4
        return (Save(file, name), Product(f.B, f.A, 2 * Ff, 4, D, 0.5f));
    }

    [Fact]
    public void Comfy_FusedGateUpOnAFusedCheckpoint_StaysOneUpdateOnOneDevice()
    {
        using var dit = Transformer(fused: true);
        var (path, expected) = FusedGateUpLora();

        using var set = Load(dit, new LoraSpec(path));
        var block = BlockOf(set, 0);

        Assert.False(set.SplitGateUp[0]);
        Assert.Equal(2 * Ff, block.Gate.Out);
        Assert.Equal(D, block.Gate.In);
        Assert.Equal(16, block.Gate.Rank);
        AssertClose(expected, Delta(block.Gate), "gate_up");
        AssertEmpty(block.Up);
        Assert.Contains("scale 0.5", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Comfy_FusedGateUpUnderTensorParallelism_SplitsIntoHalvesSlicedPerRank()
    {
        using var dit = Transformer(fused: true);
        var (path, expected) = FusedGateUpLora();

        using var set = Load(dit, 2, BackendType.GgmlCpu, new LoraSpec(path));

        Assert.True(set.SplitGateUp[0]);
        var rank0 = BlockOf(set, 0, 0);
        const int local = Ff / 2;
        for (int r = 0; r < 2; r++)
        {
            var block = BlockOf(set, 0, r);
            Assert.Equal(local, block.Gate.Out);
            Assert.Equal(local, block.Up.Out);
            // Column-parallel: rows of up are sliced in place; the halves share one down.
            Assert.Equal(rank0.Gate.Up + r * local * block.Gate.Rank * 2, block.Gate.Up);
            Assert.Equal(rank0.Up.Up + r * local * block.Up.Rank * 2, block.Up.Up);
            Assert.Equal(block.Gate.Down, block.Up.Down);
            Assert.Equal(rank0.Gate.Down, block.Gate.Down);
            AssertClose(Rows(expected, D, r * local, local), Delta(block.Gate), $"rank {r} gate");
            AssertClose(Rows(expected, D, Ff + r * local, local), Delta(block.Up), $"rank {r} up");
        }
    }

    [Fact]
    public void Comfy_FusedGateUpOnAnUnfusedCheckpoint_SharesOneDownBetweenTheHalves()
    {
        using var dit = Transformer(fused: false);
        var (path, expected) = FusedGateUpLora();

        using var set = Load(dit, new LoraSpec(path));
        var block = BlockOf(set, 0);

        Assert.False(set.SplitGateUp[0]);
        Assert.Equal(Ff, block.Gate.Out);
        Assert.Equal(Ff, block.Up.Out);
        Assert.Equal(block.Gate.Down, block.Up.Down);
        AssertClose(Rows(expected, D, 0, Ff), Delta(block.Gate), "gate");
        AssertClose(Rows(expected, D, Ff, Ff), Delta(block.Up), "up");
    }

    // ---- (c) scales: PEFT metadata, alpha tensors, configs, strength --------------------------

    private (string Path, float[] Basis) ToQ(Action<LoraFile> customize = null, string name = "to_q.safetensors", int rank = 4)
    {
        var file = new LoraFile();
        var f = file.Factors("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_B.weight", rank, D, D, 11);
        customize?.Invoke(file);
        return (Save(file, name), Product(f.B, f.A, D, rank, D));
    }

    private static float MeasuredScale(float[] basis, float[] delta)
    {
        double num = 0, den = 0;
        for (int i = 0; i < basis.Length; i++) { num += (double)basis[i] * delta[i]; den += (double)basis[i] * basis[i]; }
        return (float)(num / den);
    }

    private void AssertScale(float expected, QwenImage21LoraSet set, float[] basis)
    {
        var delta = Delta(BlockOf(set, 0).Q);
        float measured = MeasuredScale(basis, delta);
        Assert.True(Math.Abs(measured - expected) <= 2e-3f * Math.Abs(expected), $"scale {measured}, expected {expected}");
        AssertClose(basis.Select(v => v * expected).ToArray(), delta, "q");
    }

    private const string PeftMetadata = """{"transformer.lora_alpha": 8, "transformer.r": 4, "text_encoder.lora_alpha": 64}""";

    [Fact]
    public void Scale_DefaultsToAlphaEqualsRank()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ();

        using var set = Load(dit, new LoraSpec(path));

        AssertScale(1f, set, basis);
    }

    [Fact]
    public void Scale_PeftMetadataAlphaDoublesTheDelta()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(f => f.Metadata["lora_adapter_metadata"] = PeftMetadata);

        using var set = Load(dit, new LoraSpec(path));

        AssertScale(2f, set, basis);
        Assert.Contains("alpha from the file's PEFT metadata", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Scale_AnExplicitAlphaTensorWinsOverMetadata()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(f =>
        {
            f.Metadata["lora_adapter_metadata"] = PeftMetadata;
            f.Metadata["ss_network_alpha"] = "16";
            f.Alpha("transformer.transformer_blocks.0.attn.to_q.alpha", 1f);
        });

        using var set = Load(dit, new LoraSpec(path));

        AssertScale(0.25f, set, basis);
    }

    [Fact]
    public void Scale_StrengthMultipliesAlphaOverRank()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(f => f.Metadata["lora_adapter_metadata"] = PeftMetadata);

        using var set = Load(dit, new LoraSpec(path, 3f));

        AssertScale(6f, set, basis);
    }

    [Fact]
    public void Scale_NegativeStrengthSubtractsTheAdapter()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ();

        using var set = Load(dit, new LoraSpec(path, -0.5f));

        AssertScale(-0.5f, set, basis);
    }

    [Fact]
    public void Scale_KohyaNetworkAlphaMetadata_IsIgnoredWithoutAlphaTensors()
    {
        // ss_network_alpha is training metadata: kohya files carry a per-module .alpha, and a
        // converter that dropped those folded alpha into the factors, so applying it again
        // would double-scale. ComfyUI, diffusers and sd-scripts ignore it; so does TensorSharp.
        using var dit = Transformer();
        var (path, basis) = ToQ(f => f.Metadata["ss_network_alpha"] = "2");

        using var set = Load(dit, new LoraSpec(path));

        AssertScale(1f, set, basis);
    }

    [Fact]
    public void Scale_AnExplicitConfigWinsOverEmbeddedMetadata()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(f => f.Metadata["lora_adapter_metadata"] = PeftMetadata);
        string config = WriteText("adapter_config.json", """{ "peft_type": "LORA", "r": 4, "lora_alpha": 16 }""");

        using var set = Load(dit, new LoraSpec(path, null, config));

        AssertScale(4f, set, basis);
        Assert.Contains("config adapter_config.json (PEFT adapter_config.json)", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Scale_RsLoraDividesBySqrtRank()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ();
        string config = WriteText("rs/adapter_config.json", """{ "lora_alpha": 8, "use_rslora": true }""");

        using var set = Load(dit, new LoraSpec(path, null, config));

        AssertScale(4f, set, basis);   // 8 / sqrt(4)
    }

    [Fact]
    public void Scale_PeftAlphaPatternAppliesPerModule()
    {
        using var dit = Transformer();
        var file = new LoraFile();
        var q = file.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 21);
        var k = file.Factors("transformer_blocks.0.attn.to_k.lora_A.weight", "transformer_blocks.0.attn.to_k.lora_B.weight", 4, D, D, 22);
        string path = Save(file, "pattern.safetensors");
        string config = WriteText("pattern/adapter_config.json", """{ "lora_alpha": 4, "alpha_pattern": { "attn.to_k": 8 } }""");

        using var set = Load(dit, new LoraSpec(path, null, config));
        var block = BlockOf(set, 0);

        AssertClose(Product(q.B, q.A, D, 4, D, 1f), Delta(block.Q), "q");
        AssertClose(Product(k.B, k.A, D, 4, D, 2f), Delta(block.K), "k");
    }

    [Fact]
    public void Scale_PeftFolderConfigIsFoundNextToAdapterModel()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(name: "peft/adapter_model.safetensors");
        WriteText("peft/adapter_config.json", """{ "peft_type": "LORA", "lora_alpha": 12, "r": 4 }""");

        using var set = Load(dit, new LoraSpec(path));

        AssertScale(3f, set, basis);
        Assert.Contains("(PEFT folder)", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Scale_TensorSharpConfigSetsTheDefaultStrength_TheCommandLineOverridesIt()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ();
        string config = WriteText("plugin.json", """{ "type": "qwen-image-2.1-lora", "scale": 0.5, "alpha": 8 }""");

        using (var set = Load(dit, new LoraSpec(path, null, config)))
            AssertScale(1f, set, basis);        // 0.5 * 8 / 4
        using (var set = Load(dit, new LoraSpec(path, 0.25f, config)))
            AssertScale(0.5f, set, basis);      // 0.25 * 8 / 4
    }

    [Fact]
    public void Recipe_ComesFromTheConfigThatCarriesIt()
    {
        using var dit = Transformer();
        var (path, _) = ToQ();
        string config = WriteText("turbo.json", """
        { "type": "qwen-image-2.1-lora", "sampling": { "steps": 3, "sigmas": [1.0, 0.6, 0.3], "cfg": 1.0 } }
        """);
        var (style, _) = ToQ(name: "style.safetensors");

        using var set = Load(dit, new LoraSpec(path, null, config), new LoraSpec(style, 0.5f));

        Assert.NotNull(set.Recipe);
        Assert.Equal(3, set.Recipe.DefaultSteps);
        Assert.Equal(new[] { 1f, 0.6f, 0.3f, 0f }, set.Recipe.Sigmas(3, 4096));
    }

    // ---- stacking ------------------------------------------------------------------------------

    [Fact]
    public void Stacking_TwoLorasOnOneProjection_ConcatenateAlongTheRank()
    {
        using var dit = Transformer();
        var first = new LoraFile();
        var a = first.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 31);
        var second = new LoraFile();
        var b = second.Factors("lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight", "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight", 8, D, D, 32);
        second.Alpha("lora_unet_transformer_blocks_0_attn_to_q.alpha", 4f);

        using var set = Load(dit, new LoraSpec(Save(first, "one.safetensors"), 2f), new LoraSpec(Save(second, "two.safetensors")));
        var q = BlockOf(set, 0).Q;

        Assert.Equal(16, q.Rank);   // 4 + 8 = 12, padded to 16
        AssertClose(Add(Product(a.B, a.A, D, 4, D, 2f), Product(b.B, b.A, D, 8, D, 0.5f)), Delta(q), "q");
    }

    // ---- (d) rank padding and storage type ---------------------------------------------------

    [Theory]
    [InlineData(BackendType.GgmlCpu, 4, 16)]
    [InlineData(BackendType.GgmlMetal, 4, 64)]
    [InlineData(BackendType.GgmlCuda, 20, 32)]
    [InlineData(BackendType.GgmlMetal, 70, 128)]
    public void Padding_RoundsTheRankUpWithZeroComponents(BackendType backend, int rank, int padded)
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(rank: rank);

        using var set = Load(dit, 1, backend, new LoraSpec(path));
        var q = BlockOf(set, 0).Q;

        Assert.Equal(padded, q.Rank);
        var down = DownOf(q);
        var up = UpOf(q);
        for (int r = rank; r < padded; r++)
        {
            for (int i = 0; i < D; i++) Assert.Equal(0f, down[r * D + i]);
            for (int o = 0; o < D; o++) Assert.Equal(0f, up[o * padded + r]);
        }
        Assert.Contains(down.Take(rank * D), x => x != 0f);
        AssertClose(basis, Delta(q), "q");
    }

    [Fact]
    public void Storage_ValuesBeyondF16StayF32()
    {
        using var dit = Transformer();
        var file = new LoraFile();
        var a = Enumerable.Range(0, D).Select(i => 1e5f * (i + 1) / D).ToArray();
        var b = Enumerable.Range(0, D).Select(i => -1e5f * (D - i) / D).ToArray();
        file.Add("transformer_blocks.0.attn.to_v.lora_A.weight", new long[] { 1, D }, a);
        file.Add("transformer_blocks.0.attn.to_v.lora_B.weight", new long[] { D, 1 }, b);

        using var set = Load(dit, new LoraSpec(Save(file, "huge.safetensors")));
        var v = BlockOf(set, 0).V;

        Assert.Equal(F32, v.Type);
        AssertClose(Product(b, a, D, 1, D), Delta(v), "v", 1e-5f);
    }

    [Fact]
    public void Storage_Bf16AndF16FilesAreRead()
    {
        using var dit = Transformer();
        var a = Rand(4 * D, 41);
        var b = Rand(D * 4, 42);
        var file = new LoraFile()
            .Add("transformer_blocks.0.attn.to_k.lora_A.weight", new long[] { 4, D }, a, "BF16")
            .Add("transformer_blocks.0.attn.to_k.lora_B.weight", new long[] { D, 4 }, b, "F16");

        using var set = Load(dit, new LoraSpec(Save(file, "half.safetensors")));

        var aBf16 = a.Select(x => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(x) & 0xFFFF0000u)).ToArray();
        var bF16 = b.Select(x => (float)(Half)x).ToArray();
        AssertClose(Product(bF16, aBf16, D, 4, D), Delta(BlockOf(set, 0).K), "k");
    }

    // ---- globals ---------------------------------------------------------------------------

    [Fact]
    public void Globals_KohyaModulationLora()
    {
        using var dit = Transformer();
        var file = new LoraFile();
        var m = file.Factors("lora_unet_modulation_1.lora_down.weight", "lora_unet_modulation_1.lora_up.weight", 4, D, ModOut, 51);
        file.Alpha("lora_unet_modulation_1.alpha", 8f);

        using var set = Load(dit, 2, BackendType.GgmlCpu, new LoraSpec(Save(file, "mod.safetensors")));
        var rank0 = AdapterOf(set, 0).Modulation;
        var rank1 = AdapterOf(set, 1).Modulation;

        Assert.Equal(D, rank0.In);
        Assert.Equal(ModOut, rank0.Out);
        AssertClose(Product(m.B, m.A, ModOut, 4, D, 2f), Delta(rank0), "modulation");
        // Outside the blocks everything is replicated across tensor-parallel ranks.
        Assert.Equal(rank0, rank1);
    }

    // ---- (e) DoRA --------------------------------------------------------------------------

    private static float[] DoraGain(float[] magnitude, float[] norms) =>
        magnitude.Zip(norms, (m, n) => m / (n + 1.1920929e-7f)).ToArray();

    [Theory]
    [InlineData(1f)]
    [InlineData(0.5f)]
    public void Dora_ScalesTheBaseRowsAndTheLowRankTermLikeComfyUI(float strength)
    {
        using var dit = Transformer();
        var file = new LoraFile();
        var f = file.Factors("diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight", "diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight", 4, D, D, 61);
        var norms = RowNorms(_base["transformer_blocks.0.attn.to_q"], D);
        var magnitude = norms.Select((n, o) => n * (0.5f + 0.1f * o)).ToArray();
        file.Add("diffusion_model.transformer_blocks.0.attn.to_q.dora_scale", new long[] { D, 1 }, magnitude);

        using var set = Load(dit, new LoraSpec(Save(file, "dora.safetensors"), strength));
        var q = BlockOf(set, 0).Q;

        var g = DoraGain(magnitude, norms);
        Assert.NotEqual(IntPtr.Zero, q.RowScale);
        var rowScale = RowScaleOf(q);
        for (int o = 0; o < D; o++)
            AssertNear(1f + strength * (g[o] - 1f), rowScale[o], $"row scale {o}");
        AssertClose(ScaleRows(Product(f.B, f.A, D, 4, D, strength), D, g), Delta(q), "q");
        Assert.Contains("1 DoRA magnitude(s)", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Dora_OnTheUpHalfOfAFusedCheckpoint_NormalizesByTheUpRows()
    {
        using var dit = Transformer(fused: true);
        var file = new LoraFile();
        var f = file.Factors("transformer.transformer_blocks.0.img_mlp.proj.lora_A.weight", "transformer.transformer_blocks.0.img_mlp.proj.lora_B.weight", 4, D, Ff, 62);
        var norms = RowNorms(Rows(_base["transformer_blocks.0.img_mlp.gate_up"], D, Ff, Ff), D);
        var magnitude = norms.Select((n, o) => n * (1.5f - 0.05f * o)).ToArray();
        file.Add("transformer.transformer_blocks.0.img_mlp.proj.dora_scale", new long[] { Ff, 1 }, magnitude);

        using var set = Load(dit, new LoraSpec(Save(file, "dora-up.safetensors")));
        var block = BlockOf(set, 0);

        Assert.True(set.SplitGateUp[0]);
        var g = DoraGain(magnitude, norms);
        var rowScale = RowScaleOf(block.Up);
        for (int o = 0; o < Ff; o++) AssertNear(g[o], rowScale[o], $"row scale {o}");
        AssertClose(ScaleRows(Product(f.B, f.A, Ff, 4, D), D, g), Delta(block.Up), "up");
        AssertEmpty(block.Gate);
    }

    [Fact]
    public void Dora_MagnitudeOnly_IsARowScaleWithoutFactors()
    {
        using var dit = Transformer();
        var norms = RowNorms(_base["transformer_blocks.0.attn.to_k"], D);
        var magnitude = norms.Select((n, o) => n * (0.8f + 0.05f * o)).ToArray();
        var file = new LoraFile().Add("transformer_blocks.0.attn.to_k.dora_scale", new long[] { D }, magnitude);

        using var set = Load(dit, new LoraSpec(Save(file, "magnitude.safetensors")));
        var k = BlockOf(set, 0).K;

        Assert.Equal(0, k.Rank);
        Assert.Equal(IntPtr.Zero, k.Down);
        Assert.Equal(IntPtr.Zero, k.Up);
        var rowScale = RowScaleOf(k);
        var g = DoraGain(magnitude, norms);
        for (int o = 0; o < D; o++) AssertNear(g[o], rowScale[o], $"row scale {o}");
    }

    [Fact]
    public void Dora_AfterAnEarlierLora_RescalesItToo()
    {
        using var dit = Transformer();
        var first = new LoraFile();
        var a = first.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 71);
        var second = new LoraFile();
        var b = second.Factors("diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight", "diffusion_model.transformer_blocks.0.attn.to_q.lora_up.weight", 4, D, D, 72);
        var norms = RowNorms(_base["transformer_blocks.0.attn.to_q"], D);
        var magnitude = norms.Select((n, o) => n * (0.7f + 0.08f * o)).ToArray();
        second.Add("diffusion_model.transformer_blocks.0.attn.to_q.dora_scale", new long[] { D, 1 }, magnitude);

        using var set = Load(dit, new LoraSpec(Save(first, "plain.safetensors")), new LoraSpec(Save(second, "dora.safetensors")));
        var q = BlockOf(set, 0).Q;

        var g = DoraGain(magnitude, norms);
        AssertClose(ScaleRows(Add(Product(a.B, a.A, D, 4, D), Product(b.B, b.A, D, 4, D)), D, g), Delta(q), "q");
        var rowScale = RowScaleOf(q);
        for (int o = 0; o < D; o++) AssertNear(g[o], rowScale[o], $"row scale {o}");
    }

    // ---- 1-D diffs ---------------------------------------------------------------------------

    [Fact]
    public void Diff_ShiftsANormGain()
    {
        using var dit = Transformer();
        var diff = Rand(HeadDim, 81, 0.01f);
        var file = new LoraFile().Add("diffusion_model.transformer_blocks.0.attn.norm_k.diff", new long[] { HeadDim }, diff);

        using var set = Load(dit, new LoraSpec(Save(file, "diff.safetensors")));

        Assert.NotEqual(IntPtr.Zero, set.NormK[0]);
        Assert.Equal(IntPtr.Zero, set.NormQ[0]);
        Assert.Equal(IntPtr.Zero, set.TextNorm);
        Assert.Equal(Add(_base["transformer_blocks.0.attn.norm_k"], diff), ReadValues(set.NormK[0], HeadDim, F32));
    }

    // ---- VideoX-Fun PDD bundles ----------------------------------------------------------------

    private const string PddFormat = "qwenimage21_extracted_prefused_v1";
    private const int Channels = QwenImage21DiT.Channels, Dim = QwenImage21DiT.HiddenSize, TextDim = QwenImage21DiT.TextDim;

    private (string Path, float[] Heads, float[] NormQ, float[] TextNorm, (float[] A, float[] B) Q) PddBundle(
        string folder, int steps = 2, bool withHeads = true, bool withConfig = true, string fullParameters = null)
    {
        var file = new LoraFile();
        file.Metadata["format"] = PddFormat;
        // Multiples of 1/64 in [-2, 2): exact in BF16, so the read-back is exact.
        var heads = Enumerable.Range(0, steps * Channels * Dim).Select(i => ((i * 37) % 256 - 128) / 64f).ToArray();
        if (withHeads) file.Add("proj_out.weight", new long[] { steps, Channels, Dim }, heads, "BF16");
        var normQ = Rand(HeadDim, 91, 0.1f, 1f);
        file.Add("transformer_blocks.0.attn.norm_q.weight", new long[] { HeadDim }, normQ);
        var textNorm = Rand(TextDim, 92, 0.1f, 1f);
        file.Add("txt_in.text_norm.weight", new long[] { TextDim }, textNorm);
        var q = file.Factors("transformer_blocks.0.attn.to_q.lora_down", "transformer_blocks.0.attn.to_q.lora_up", 4, D, D, 93);
        string path = Save(file, Path.Combine(folder, "bundle.safetensors"));
        if (withConfig)
        {
            string sigmas = string.Join(", ", Enumerable.Range(0, steps + 1).Select(i => (1.0 - (double)i / steps).ToString(System.Globalization.CultureInfo.InvariantCulture)));
            fullParameters ??= "\"['proj_out.weight', 'transformer_blocks.0.attn.norm_q.weight', 'txt_in.text_norm.weight']\"";
            WriteText(Path.Combine(folder, "pdd_config.json"),
                $$"""{ "pdd_num_steps": {{steps}}, "pdd_block_size": 1, "pdd_sigmas": [{{sigmas}}], "pdd_full_parameters": {{fullParameters}}, "lora_alpha": 8 }""");
        }
        return (path, heads, normQ, textNorm, q);
    }

    [Fact]
    public void Pdd_BundleReplacesHeadsAndNormsAndCarriesItsRecipe()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd");

        using var set = Load(dit, 2, BackendType.GgmlCpu, new LoraSpec(bundle.Path));

        Assert.Equal(2, set.OutputHeads.Length);
        for (int h = 0; h < 2; h++)
            Assert.Equal(bundle.Heads.Skip(h * Channels * Dim).Take(Channels * Dim).ToArray(), ReadValues(set.OutputHeads[h], Channels * Dim, F32));
        Assert.Equal(bundle.NormQ, ReadValues(set.NormQ[0], HeadDim, F32));
        Assert.Equal(bundle.TextNorm, ReadValues(set.TextNorm, TextDim, F32));
        Assert.Equal(IntPtr.Zero, set.NormK[0]);
        Assert.Equal(IntPtr.Zero, set.NormQ[1]);

        // alpha 8 over rank 4; q is column-parallel, so each of the two ranks holds half its rows.
        var qExpected = Product(bundle.Q.B, bundle.Q.A, D, 4, D, 2f);
        for (int r = 0; r < 2; r++)
            AssertClose(Rows(qExpected, D, r * D / 2, D / 2), Delta(BlockOf(set, 0, r).Q), $"rank {r} q");

        var recipe = set.Recipe;
        Assert.NotNull(recipe);
        Assert.Equal(2, recipe.DefaultSteps);
        Assert.True(recipe.TimestepBf16);
        Assert.Equal(1f, recipe.Cfg);
        Assert.Equal(new[] { 1f, 0.5f, 0f }, recipe.Sigmas(2, 4096));

        // Step heads are selected per call on every rank's descriptor.
        for (int r = 0; r < 2; r++)
        {
            Assert.Equal(set.OutputHeads[0], AdapterOf(set, r).OutputHead);
            Assert.Equal(F32, AdapterOf(set, r).OutputHeadType);
        }
        set.SelectOutputHead(1);
        for (int r = 0; r < 2; r++) Assert.Equal(set.OutputHeads[1], AdapterOf(set, r).OutputHead);
        Assert.Throws<ArgumentOutOfRangeException>(() => set.SelectOutputHead(2));
        Assert.Throws<ArgumentOutOfRangeException>(() => set.SelectOutputHead(-1));
        Assert.Contains("2 per-step output heads", set.Summary, StringComparison.Ordinal);
        Assert.Contains("3 replaced parameter(s)", set.Summary, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_ConfigMayBePassedExplicitly()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-explicit");
        string config = Path.Combine(_dir, "pdd-explicit", "pdd_config.json");
        string moved = Path.Combine(_dir, "elsewhere-pdd_config.json");
        File.Move(config, moved);

        using var set = Load(dit, new LoraSpec(bundle.Path, null, moved));

        Assert.Equal(2, set.OutputHeads.Length);
        Assert.NotNull(set.Recipe);
    }

    [Fact]
    public void Pdd_WithoutItsConfig_IsRefused()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-no-config", withConfig: false);

        var ex = Assert.Throws<FileNotFoundException>(() => Load(dit, new LoraSpec(bundle.Path)));
        Assert.Contains("pdd_config.json", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_WithoutHeads_IsRefused()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-no-heads", withHeads: false);

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(bundle.Path)));
        Assert.Contains("without its per-step proj_out heads", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_HeadCountMustMatchTheTrainedSteps()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-mismatch");
        WriteText(Path.Combine("pdd-mismatch", "pdd_config.json"),
            """{ "pdd_num_steps": 3, "pdd_sigmas": [1.0, 0.6, 0.3, 0.0], "pdd_full_parameters": [], "lora_alpha": 8 }""");

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(bundle.Path)));
        Assert.Contains("2 step heads", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_ReplacingAParameterTheConfigDoesNotList_IsRefused()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-unlisted", fullParameters: "[\"proj_out.weight\", \"txt_in.text_norm.weight\"]");

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(bundle.Path)));
        Assert.Contains("transformer_blocks.0.attn.norm_q", ex.Message, StringComparison.Ordinal);
        Assert.Contains("pdd_full_parameters", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_BundleWithANonPddConfig_IsRefused()
    {
        using var dit = Transformer();
        var bundle = PddBundle("pdd-wrong-config");
        string config = WriteText("peft-config.json", """{ "lora_alpha": 8 }""");

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(bundle.Path, null, config)));
        Assert.Contains("is a PDD bundle", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Pdd_ConfigForAFileThatIsNotABundle_IsRefused()
    {
        using var dit = Transformer();
        var (path, _) = ToQ();
        string config = WriteText("lonely/pdd_config.json", """{ "pdd_num_steps": 1, "pdd_sigmas": [1.0, 0.0] }""");

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(path, null, config)));
        Assert.Contains("is a PDD config", ex.Message, StringComparison.Ordinal);
    }

    // ---- (f) failures ----------------------------------------------------------------------------

    private void AssertRefused<TException>(LoraFile file, string fragment, bool fused = true) where TException : Exception
    {
        using var dit = Transformer(fused);
        string path = Save(file, "bad-" + Guid.NewGuid().ToString("N") + ".safetensors");
        var ex = Assert.Throws<TException>(() => Load(dit, new LoraSpec(path)));
        Assert.Contains(fragment, ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Refuse_AnUnknownModule_NamingTheKeyAndTheReason()
    {
        var file = new LoraFile();
        file.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 1);
        file.Factors("transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight", "transformer.transformer_blocks.0.attn.add_q_proj.lora_B.weight", 4, D, D, 2);

        AssertRefused<InvalidDataException>(file, "transformer.transformer_blocks.0.attn.add_q_proj.lora_A.weight: a module");
        AssertRefused<InvalidDataException>(file, "Nothing is skipped silently");
    }

    [Fact]
    public void Refuse_ALoneFactor()
    {
        var file = new LoraFile().Add("transformer_blocks.0.attn.to_q.lora_A.weight", new long[] { 4, D }, Rand(4 * D, 1));
        AssertRefused<InvalidDataException>(file, "only one of its two factors");
    }

    [Fact]
    public void Refuse_AnAlphaWithoutFactors()
    {
        var file = new LoraFile().Alpha("transformer_blocks.0.attn.to_q.alpha", 4f);
        AssertRefused<InvalidDataException>(file, "has an alpha but no factors");
    }

    [Theory]
    [InlineData(4, 16, D, 4)]   // down reads 16 inputs; the projection has 8
    [InlineData(4, D, 16, 4)]   // up writes 16 outputs
    [InlineData(4, D, D, 3)]    // up's rank differs from down's
    public void Refuse_FactorsThatDoNotMatchTheProjection(int rank, int input, int output, int upRank)
    {
        var file = new LoraFile()
            .Add("transformer_blocks.0.attn.to_q.lora_A.weight", new long[] { rank, input }, Rand(rank * input, 1))
            .Add("transformer_blocks.0.attn.to_q.lora_B.weight", new long[] { output, upRank }, Rand(output * upRank, 2));
        AssertRefused<InvalidDataException>(file, "made for a different model");
    }

    [Fact]
    public void Refuse_ConvolutionFactors()
    {
        var file = new LoraFile()
            .Add("transformer_blocks.0.attn.to_q.lora_down.weight", new long[] { 4, D, 1, 1 }, Rand(4 * D, 1))
            .Add("transformer_blocks.0.attn.to_q.lora_up.weight", new long[] { D, 4, 1, 1 }, Rand(D * 4, 2));
        AssertRefused<NotSupportedException>(file, "not 2-D");
    }

    [Fact]
    public void Refuse_AFullValueOutsideAPddBundle()
    {
        var file = new LoraFile().Add("transformer_blocks.0.attn.norm_q.weight", new long[] { HeadDim }, Rand(HeadDim, 1));
        AssertRefused<InvalidDataException>(file, "only VideoX-Fun PDD bundles");
    }

    [Fact]
    public void Refuse_AFullValueMixedWithFactors()
    {
        var file = new LoraFile().Add("transformer_blocks.0.attn.norm_q.weight", new long[] { HeadDim }, Rand(HeadDim, 1));
        file.Factors("transformer_blocks.0.attn.norm_q.lora_A.weight", "transformer_blocks.0.attn.norm_q.lora_B.weight", 4, HeadDim, HeadDim, 3);
        AssertRefused<InvalidDataException>(file, "mixes a full value with low-rank factors");
    }

    [Fact]
    public void Refuse_ADiffOfTheWrongLength()
    {
        var file = new LoraFile().Add("transformer_blocks.0.attn.norm_k.diff", new long[] { 64 }, Rand(64, 1));
        AssertRefused<InvalidDataException>(file, "does not replace transformer_blocks.0.attn.norm_k");
    }

    [Fact]
    public void Refuse_AModuleTheCheckpointDoesNotHave()
    {
        var file = new LoraFile();
        file.Factors("img_in.lora_A.weight", "img_in.lora_B.weight", 4, D, D, 1);
        AssertRefused<InvalidDataException>(file, "targets img_in, which the transformer GGUF does not have");
    }

    [Fact]
    public void Refuse_TwoTensorsForOneFactor()
    {
        var file = new LoraFile();
        file.Factors("transformer.transformer_blocks.0.attn.to_q.lora_A.weight", "transformer.transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 1);
        file.Add("diffusion_model.transformer_blocks.0.attn.to_q.lora_down.weight", new long[] { 4, D }, Rand(4 * D, 2));
        AssertRefused<InvalidDataException>(file, "a second Down tensor for transformer_blocks.0.attn.to_q");
    }

    [Fact]
    public void Refuse_APeftFileWithTwoAdapterSlots()
    {
        var file = new LoraFile();
        file.Factors("transformer_blocks.0.attn.to_q.lora_A.default.weight", "transformer_blocks.0.attn.to_q.lora_B.default.weight", 4, D, D, 1);
        file.Factors("transformer_blocks.0.attn.to_k.lora_A.style.weight", "transformer_blocks.0.attn.to_k.lora_B.style.weight", 4, D, D, 2);
        AssertRefused<InvalidDataException>(file, "several PEFT adapters");
    }

    [Fact]
    public void Refuse_AnInputAxisDoraMagnitude()
    {
        var file = new LoraFile();
        file.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 1);
        file.Add("transformer_blocks.0.attn.to_q.dora_scale", new long[] { 1, D + 1 }, Rand(D + 1, 2, 0.1f, 1f));
        AssertRefused<NotSupportedException>(file, "only output-axis DoRA magnitudes");
    }

    [Fact]
    public void Refuse_AFileWithoutTensors()
    {
        var file = new LoraFile();
        file.Metadata["note"] = "empty";
        AssertRefused<InvalidDataException>(file, "contains no tensors");
    }

    [Fact]
    public void Refuse_AMissingFile()
    {
        using var dit = Transformer();
        Assert.Throws<FileNotFoundException>(() => Load(dit, new LoraSpec(Path.Combine(_dir, "missing.safetensors"))));
    }

    [Fact]
    public void Refuse_ANonFiniteStrength()
    {
        using var dit = Transformer();
        var (path, _) = ToQ();
        var ex = Assert.Throws<ArgumentException>(() => Load(dit, new LoraSpec(path, float.PositiveInfinity)));
        Assert.Contains("finite", ex.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void Refuse_TwoPluginsWithSamplingRecipes()
    {
        using var dit = Transformer();
        var (first, _) = ToQ(name: "first.safetensors");
        var (second, _) = ToQ(name: "second.safetensors");
        string recipeA = WriteText("a.json", """{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 4, "sigmas": [1, 0.75, 0.5, 0.25] } }""");
        string recipeB = WriteText("b.json", """{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 2, "sigmas": [1, 0.5] } }""");

        var ex = Assert.Throws<ArgumentException>(() => Load(dit, new LoraSpec(first, null, recipeA), new LoraSpec(second, null, recipeB)));
        Assert.Contains("Two LoRA plug-ins define a sampling recipe", ex.Message, StringComparison.Ordinal);
    }

    // ---- (g) tensor parallelism: row-parallel projections -------------------------------------

    [Fact]
    public void TensorParallel_RowParallelProjectionsCopyTheirDownColumnsPerRank()
    {
        using var dit = Transformer();
        var file = new LoraFile();
        var o = file.Factors("transformer_blocks.0.attn.to_out.0.lora_A.weight", "transformer_blocks.0.attn.to_out.0.lora_B.weight", 4, D, D, 111);
        var m = file.Factors("transformer_blocks.0.img_mlp.out.lora_A.weight", "transformer_blocks.0.img_mlp.out.lora_B.weight", 4, Ff, D, 112);
        var q = file.Factors("transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight", 4, D, D, 113);

        using var set = Load(dit, 2, BackendType.GgmlCpu, new LoraSpec(Save(file, "tp.safetensors")));

        var outExpected = Product(o.B, o.A, D, 4, D);
        var mlpExpected = Product(m.B, m.A, D, 4, Ff);
        var qExpected = Product(q.B, q.A, D, 4, D);
        var rank0 = BlockOf(set, 0, 0);
        var rank1 = BlockOf(set, 0, 1);
        Assert.NotEqual(rank0.Out.Down, rank1.Out.Down);
        Assert.Equal(rank0.Out.Up, rank1.Out.Up);
        for (int r = 0; r < 2; r++)
        {
            var block = BlockOf(set, 0, r);
            Assert.Equal(D / 2, block.Out.In);
            Assert.Equal(D, block.Out.Out);
            AssertClose(Columns(outExpected, D, r * D / 2, D / 2), Delta(block.Out), $"rank {r} to_out");
            Assert.Equal(Ff / 2, block.Down.In);
            AssertClose(Columns(mlpExpected, Ff, r * Ff / 2, Ff / 2), Delta(block.Down), $"rank {r} img_mlp.out");
            // Column-parallel q: rows sliced in place.
            Assert.Equal(D / 2, block.Q.Out);
            Assert.Equal(rank0.Q.Up + r * (D / 2) * block.Q.Rank * 2, block.Q.Up);
            AssertClose(Rows(qExpected, D, r * D / 2, D / 2), Delta(block.Q), $"rank {r} q");
        }
        // The rank sum of the partial row-parallel updates is the whole update.
        AssertClose(outExpected, Add(
            ScatterColumns(Delta(rank0.Out), D, 0, D), ScatterColumns(Delta(rank1.Out), D, D / 2, D)), "to_out sum");
    }

    private static float[] ScatterColumns(float[] part, int rows, int first, int input)
    {
        int count = part.Length / rows;
        var result = new float[rows * input];
        for (int o = 0; o < rows; o++) Array.Copy(part, o * count, result, o * input + first, count);
        return result;
    }

    [Fact]
    public void Dispose_ReleasesTheDescriptors()
    {
        using var dit = Transformer();
        var (path, _) = ToQ();
        var set = Load(dit, new LoraSpec(path));
        Assert.NotEqual(IntPtr.Zero, set.AdapterFor(0));

        set.Dispose();
        set.Dispose();   // idempotent

        Assert.Throws<IndexOutOfRangeException>(() => set.AdapterFor(0));
    }

    // ---- regressions from the adversarial review --------------------------------------------

    [Fact]
    public void Diff_ScalesWithTheStrength()
    {
        using var dit = Transformer();
        var diff = Rand(HeadDim, 81, 0.01f);
        var file = new LoraFile().Add("diffusion_model.transformer_blocks.0.attn.norm_k.diff", new long[] { HeadDim }, diff);

        using var set = Load(dit, new LoraSpec(Save(file, "diff.safetensors"), 0.5f));

        // ComfyUI's diff patch: weight += strength * diff.
        AssertClose(_base["transformer_blocks.0.attn.norm_k"].Zip(diff, (w, d) => w + 0.5f * d).ToArray(),
            ReadValues(set.NormK[0], HeadDim, F32), "norm_k", 1e-6f);
    }

    [Fact]
    public void Diff_StacksAcrossPlugins()
    {
        using var dit = Transformer();
        var first = Rand(HeadDim, 81, 0.01f);
        var second = Rand(HeadDim, 82, 0.01f);
        string a = Save(new LoraFile().Add("diffusion_model.transformer_blocks.0.attn.norm_k.diff", new long[] { HeadDim }, first), "a.safetensors");
        string b = Save(new LoraFile().Add("diffusion_model.transformer_blocks.0.attn.norm_k.diff", new long[] { HeadDim }, second), "b.safetensors");

        using var set = Load(dit, new LoraSpec(a), new LoraSpec(b));

        AssertClose(Add(Add(_base["transformer_blocks.0.attn.norm_k"], first), second),
            ReadValues(set.NormK[0], HeadDim, F32), "norm_k", 1e-6f);
    }

    [Fact]
    public void FullReplacement_RefusesToDiscardAnEarlierPluginsChange()
    {
        using var dit = Transformer();
        string diff = Save(new LoraFile().Add("diffusion_model.transformer_blocks.0.attn.norm_q.diff", new long[] { HeadDim },
            Rand(HeadDim, 83, 0.01f)), "norm-q-diff.safetensors");
        string folder = Path.Combine(_dir, "pdd-after-diff");
        Directory.CreateDirectory(folder);
        var bundle = PddBundle(folder);

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(diff), new LoraSpec(bundle.Path)));
        Assert.Contains("already changed", ex.Message, StringComparison.Ordinal);

        // The other order applies the diff on top of the bundle's value.
        using var set = Load(dit, new LoraSpec(bundle.Path), new LoraSpec(diff));
        Assert.NotEqual(bundle.NormQ, ReadValues(set.NormQ[0], HeadDim, F32));
    }

    [Fact]
    public void PddHeads_WithAProjOutUpdate_AreRefusedAtLoad()
    {
        using var dit = Transformer();
        string folder = Path.Combine(_dir, "pdd-proj-out");
        Directory.CreateDirectory(folder);
        var bundle = PddBundle(folder);
        var other = new LoraFile();
        other.Factors("transformer.proj_out.lora_A.weight", "transformer.proj_out.lora_B.weight", 4, D, Channels, 84);
        string otherPath = Save(other, "proj-out.safetensors");

        var ex = Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(bundle.Path), new LoraSpec(otherPath)));
        Assert.Contains("proj_out", ex.Message, StringComparison.Ordinal);
        Assert.Throws<InvalidDataException>(() => Load(dit, new LoraSpec(otherPath), new LoraSpec(bundle.Path)));
    }

    [Fact]
    public void RsLora_FromEmbeddedMetadata_SurvivesAConfigThatDoesNotMentionIt()
    {
        using var dit = Transformer();
        var (path, basis) = ToQ(f => f.Metadata["lora_adapter_metadata"] =
            """{ "transformer.lora_alpha": 16, "transformer.r": 4, "transformer.use_rslora": true }""");
        string config = WriteText("recipe.json", """{ "type": "qwen-image-2.1-lora", "scale": 1.0 }""");

        using var set = Load(dit, new LoraSpec(path, null, config));

        AssertScale(16f / 2f, set, basis); // alpha / sqrt(rank)
    }

    [Fact]
    public void PeftFolderAlpha_StillAppliesBesideAnExplicitConfig()
    {
        using var dit = Transformer();
        string folder = Path.Combine(_dir, "peft-folder");
        Directory.CreateDirectory(folder);
        var (path, basis) = ToQ(name: Path.Combine(folder, "adapter_model.safetensors"));
        WriteText(Path.Combine(folder, "adapter_config.json"), """{ "peft_type": "LORA", "r": 4, "lora_alpha": 8 }""");
        string recipe = WriteText("steps-only.json", """{ "type": "qwen-image-2.1-lora", "sampling": { "steps": 6 } }""");

        using var set = Load(dit, new LoraSpec(path, null, recipe));

        AssertScale(2f, set, basis);
        Assert.Contains("PEFT folder", set.Summary, StringComparison.Ordinal);
    }

    // ---- the article's LoRAs against the real checkpoint (weights required) -----------------

    /// <summary>
    /// Every Qwen-Image-2.1 LoRA the plug-ins in config/lora ship, loaded against the real
    /// transformer GGUF: each file's every tensor is consumed (the loader refuses otherwise)
    /// and the expected number of updates, ranks and scales comes out. Point
    /// TENSORSHARP_QWEN21_LORA_DIR at a folder holding the files (searched recursively) and
    /// TENSORSHARP_QWEN21_DIT at the transformer GGUF.
    /// </summary>
    [ModelTheory("TENSORSHARP_QWEN21_LORA_DIR")]
    [InlineData("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r128.safetensors", null, "227 low-rank update(s), rank 128, scale 1")]
    [InlineData("Qwen-Image-2.1-viggle-turbo-v0.2.1-6step-lora-r256.safetensors", null, "227 low-rank update(s), rank 256, scale 1")]
    [InlineData("p_qwen_image_2.1_8step_v0.1.safetensors", null, "224 low-rank update(s), rank 64, scale 2")]
    [InlineData("p_qwen_image_2.1_5step_v0.1.safetensors", null, "224 low-rank update(s), rank 64, scale 2")]
    [InlineData("Qwen-Image-2.1-Fun-Acc-4Step.safetensors", "pdd_config.json", "231 low-rank update(s), rank 64, scale 1, 66 replaced parameter(s), 4 per-step output heads")]
    [InlineData("qwen-image-2.1-fix-1.0-comfy.safetensors", null, "132 low-rank update(s), rank 32, scale 1, 132 DoRA magnitude(s)")]
    [InlineData("filmstills_qwen21.safetensors", null, "192 low-rank update(s), rank 16, scale 1")]
    [InlineData("grainscape_qwen21.safetensors", null, "192 low-rank update(s), rank 16, scale 1")]
    [InlineData("elusarcas-qwen2-1-detailer-v1.safetensors", null, "192 low-rank update(s), rank 16, scale 1")]
    [InlineData("Qwen-Image-2.1-Natural-Exposure-LoRA-4000.safetensors", null, "224 low-rank update(s), rank 16, scale 1")]
    [InlineData("Qwen2.1_Anime_consistency.safetensors", null, "224 low-rank update(s), rank 32, scale 1")]
    [InlineData("Qwen-Image-2.1-Object-Remover-Bbox-turbo-4000.safetensors", null, "224 low-rank update(s), rank 16, scale 1")]
    [InlineData("Qwen-Image-2.1-Object-Mover-Bbox-Preview-5000.safetensors", null, "224 low-rank update(s), rank 16, scale 1")]
    public void RealArticleLoras_LoadCompletelyAgainstTheCheckpoint(string file, string config, string expected)
    {
        string dir = Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_LORA_DIR")!;
        string dit = Environment.GetEnvironmentVariable("TENSORSHARP_QWEN21_DIT");
        Assert.False(string.IsNullOrWhiteSpace(dit), "TENSORSHARP_QWEN21_DIT must name the transformer GGUF as well.");
        string Find(string name) => Directory.EnumerateFiles(dir, name, SearchOption.AllDirectories).FirstOrDefault()
            ?? throw new FileNotFoundException($"{name} is not under {dir}.");
        string weights = Find(file);
        string configPath = config == null ? null : Path.Combine(Path.GetDirectoryName(weights)!, config);

        using var gguf = new GgufFile(dit!);
        string prefix = gguf.Tensors.ContainsKey("img_in.weight") ? "" : "model.diffusion_model.";
        using var set = QwenImage21LoraSet.Load(new[] { new LoraSpec(weights, null, configPath) }, gguf, prefix, BackendType.GgmlMetal, 1);

        Assert.Contains(expected, set.Summary, StringComparison.Ordinal);
        Assert.NotEqual(IntPtr.Zero, set.AdapterFor(0));
        Assert.True(set.FactorBytes > 0);
        if (config != null)
        {
            // The PDD bundle carries its trained 4-step grid and bf16 timesteps.
            Assert.NotNull(set.Recipe);
            Assert.Equal(4, set.Recipe.DefaultSteps);
            Assert.True(set.Recipe.TimestepBf16);
            Assert.Equal(4, set.OutputHeads.Length);
        }
    }
}
