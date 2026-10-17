// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

/// <summary>
/// Synthetic MiniMax-H3 LoRA fixtures, shared by the loader tests and the model-wiring tests:
/// <list type="bullet">
/// <item>a tiny transformer GGUF with the real tensor names (the MiniMaxH3RefinerCacheTests
/// layout: an 8-wide residual stream, 2 heads of 8, ffn 16), any number of trunk and
/// token-refiner blocks, named for the FL2VA or the Ref2VA checkpoint, optionally with LoRA
/// updates merged into its weights (<see cref="MergedDeltas"/>);</item>
/// <item>a Diffusers-format adapter (<c>transformer_blocks.N.attn.to_q.lora_A.default.weight</c>
/// ...) as BF16, F16 or F32 safetensors with a <c>__metadata__</c> block;</item>
/// <item>its <c>minimax-h3-lora</c> plug-in config.</item>
/// </list>
/// Each instance owns one temporary directory and deletes it on dispose. An
/// <see cref="Adapter"/> records the factors it wrote, already rounded to the stored dtype,
/// so a test computes the update the loader must apply from exactly what the file holds.
/// </summary>
internal sealed class MiniMaxH3LoraFixtures : IDisposable
{
    internal const int Hidden = 8, HeadDim = 8, Heads = 2, Inner = Heads * HeadDim, Ffn = 16;
    internal const int TextDim = 6, TimeDim = 4, VideoPatchDim = 96, AudioPatchDim = 32;

    /// <summary>The checkpoint file names <see cref="MiniMaxH3Config.PartitionFromFileName"/> reads.</summary>
    internal const string Fl2vaModel = "minimax_h3_fl2va_pruned-tiny.gguf", Ref2vaModel = "minimax_h3_ref2va_pruned-tiny.gguf";

    /// <summary>The real adapters' file names, which tie a manifest to its weights.</summary>
    internal const string Weights8 = "minimax_h3_fl2v_turbo_8step_v1.0_768p_bf16.safetensors";
    internal const string Weights4 = "minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors";

    internal MiniMaxH3LoraFixtures()
    {
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3-lora-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
    }

    internal string Dir { get; }

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
    }

    internal static float[] Rand(int n, int seed, float scale = 1f)
    {
        var rng = new Random(seed);
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    // ---- the transformer ---------------------------------------------------------------------

    /// <summary>
    /// The tiny transformer's tensors, deterministic for <paramref name="seed"/>. Shapes are in
    /// GGUF ne order, so a linear weight is [in, out] and its data is row-major [out][in]: the
    /// layout of a LoRA update B · A.
    /// </summary>
    internal static List<(string Name, long[] Shape, float[] Data)> TransformerTensors(int blocks = 2, int refinerBlocks = 1, int seed = 317)
    {
        var tensors = new List<(string Name, long[] Shape, float[] Data)>();
        var random = new Random(seed);
        void Add(string name, long[] shape, bool gain = false)
        {
            int count = checked((int)shape.Aggregate(1L, (a, b) => a * b));
            float[] data = Enumerable.Range(0, count)
                .Select(_ => gain ? 1f : (float)(random.NextDouble() - .5) * .3f).ToArray();
            tensors.Add((name, shape, data));
        }
        void Linear(string name, int input, int output, bool bias = false)
        {
            Add(name + ".weight", [input, output]);
            if (bias) Add(name + ".bias", [output]);
        }
        void Block(string prefix, bool modulation)
        {
            foreach (string norm in new[] { "norm1", "norm2" }) Add(prefix + norm + ".weight", [Hidden], true);
            foreach (string norm in new[] { "q_norm", "k_norm" }) Add(prefix + "attn." + norm + ".weight", [HeadDim], true);
            Linear(prefix + "attn.qkv_proj", Hidden, 3 * Inner);
            Linear(prefix + "attn.out_proj", Inner, Hidden);
            Linear(prefix + "mlp.fc1", Hidden, 2 * Ffn);
            Linear(prefix + "mlp.fc2", Ffn, Hidden);
            if (modulation) Linear(prefix + "adaln_proj.linear", TimeDim, 18 * Hidden, true);
        }
        Linear("video_patch_proj", VideoPatchDim, Hidden, true);
        Linear("audio_patch_proj", AudioPatchDim, Hidden, true);
        Linear("condition_proj", TextDim, Hidden, true);
        for (int b = 0; b < refinerBlocks; b++) Block($"token_refiner.blocks.{b}.", false);
        for (int b = 0; b < blocks; b++) Block($"blocks.{b}.", true);
        Add("token_refiner.final_norm.weight", [Hidden], true);
        Add("final_layer.norm.weight", [Hidden], true);
        Linear("final_layer.adaln_proj.linear", TimeDim, 2 * Hidden, true);
        Linear("final_layer.video_out", Hidden, VideoPatchDim, true);
        Linear("final_layer.audio_out", Hidden, AudioPatchDim, true);
        Add("rope.inv_freq", [1]);
        Add("adaln_t_table", [TimeDim, 8]);
        return tensors;
    }

    /// <summary>
    /// Write the tiny transformer and return its path. <paramref name="deltas"/>, keyed by GGUF
    /// weight name (<c>blocks.0.attn.qkv_proj.weight</c>) and laid out [out][in], are added to
    /// the base weights: a checkpoint with a LoRA merged in.
    /// </summary>
    internal string WriteTransformer(MiniMaxH3Partition partition = MiniMaxH3Partition.FirstLastFrame, int blocks = 2,
        int refinerBlocks = 1, IReadOnlyDictionary<string, float[]> deltas = null, string fileName = null)
    {
        var tensors = TransformerTensors(blocks, refinerBlocks);
        if (deltas != null)
        {
            foreach (var (name, delta) in deltas)
            {
                int index = tensors.FindIndex(t => t.Name == name);
                Assert.True(index >= 0, $"The tiny transformer has no {name}.");
                var data = (float[])tensors[index].Data.Clone();
                Assert.Equal(data.Length, delta.Length);
                for (int i = 0; i < data.Length; i++) data[i] += delta[i];
                tensors[index] = (name, tensors[index].Shape, data);
            }
        }
        string path = Path.Combine(Dir, fileName ?? (partition == MiniMaxH3Partition.Reference ? Ref2vaModel : Fl2vaModel));
        WriteGguf(path, tensors);
        return path;
    }

    /// <summary>A minimal GGUF v3: no metadata (like the published H3 files), F32 tensors, 32-byte alignment.</summary>
    internal static void WriteGguf(string path, IReadOnlyList<(string Name, long[] Shape, float[] Data)> tensors)
    {
        static long Align(long value) => (value + 31) / 32 * 32;
        using var stream = File.Create(path);
        using var writer = new BinaryWriter(stream, Encoding.UTF8);
        writer.Write(0x46554747u); writer.Write(3u); writer.Write((ulong)tensors.Count); writer.Write(0UL);
        long offset = 0;
        foreach (var (name, shape, data) in tensors)
        {
            byte[] utf8 = Encoding.UTF8.GetBytes(name);
            writer.Write((ulong)utf8.Length); writer.Write(utf8); writer.Write((uint)shape.Length);
            foreach (long size in shape) writer.Write((ulong)size);
            writer.Write(0u); writer.Write((ulong)offset); offset += Align(data.LongLength * sizeof(float));
        }
        writer.Write(new byte[Align(stream.Position) - stream.Position]);
        foreach (var (_, _, data) in tensors)
        {
            foreach (float value in data) writer.Write(value);
            writer.Write(new byte[Align(data.LongLength * sizeof(float)) - data.LongLength * sizeof(float)]);
        }
    }

    /// <summary>The configuration <see cref="MiniMaxH3Config.Detect"/> derives from a written transformer.</summary>
    internal static MiniMaxH3Config ConfigOf(string ggufPath)
    {
        using var gguf = new GgufFile(ggufPath);
        return MiniMaxH3Config.Detect(gguf.Tensors);
    }

    /// <summary>The input and output width of a target's update in the tiny transformer.</summary>
    internal static (int In, int Out) Shape(MiniMaxH3LoraTarget target) => target switch
    {
        MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V => (Hidden, Inner),
        MiniMaxH3LoraTarget.Out => (Inner, Hidden),
        MiniMaxH3LoraTarget.Fc1 => (Hidden, 2 * Ffn),
        _ => (Ffn, Hidden),
    };

    /// <summary>The GGUF weight a target updates and the first row of its slice: q/k/v are the
    /// stacked thirds of <c>attn.qkv_proj</c>.</summary>
    internal static (string Weight, int FirstRow) GgufWeight(bool refiner, int block, MiniMaxH3LoraTarget target)
    {
        string b = (refiner ? "token_refiner.blocks." : "blocks.") + block + ".";
        return target switch
        {
            MiniMaxH3LoraTarget.Q => (b + "attn.qkv_proj.weight", 0),
            MiniMaxH3LoraTarget.K => (b + "attn.qkv_proj.weight", Inner),
            MiniMaxH3LoraTarget.V => (b + "attn.qkv_proj.weight", 2 * Inner),
            MiniMaxH3LoraTarget.Out => (b + "attn.out_proj.weight", 0),
            MiniMaxH3LoraTarget.Fc1 => (b + "mlp.fc1.weight", 0),
            _ => (b + "mlp.fc2.weight", 0),
        };
    }

    // ---- update arithmetic ---------------------------------------------------------------

    /// <summary><paramref name="scale"/> · B [out, rank] · A [rank, in], row-major [out][in], accumulated in double.</summary>
    internal static float[] Product(float[] b, float[] a, int output, int rank, int input, float scale = 1f)
    {
        var result = new float[output * input];
        for (int o = 0; o < output; o++)
            for (int i = 0; i < input; i++)
            {
                double sum = 0;
                for (int r = 0; r < rank; r++) sum += (double)b[o * rank + r] * a[r * input + i];
                result[o * input + i] = (float)(sum * scale);
            }
        return result;
    }

    /// <summary>A copy of <paramref name="m"/> [<paramref name="rows"/>][<paramref name="cols"/>]
    /// with its two row halves exchanged: diffusers' ff.net.0.proj [value; gate] to fc1's [gate; value].</summary>
    internal static float[] SwapRowHalves(float[] m, int rows, int cols)
    {
        var result = new float[m.Length];
        int half = rows / 2 * cols;
        Array.Copy(m, half, result, 0, half);
        Array.Copy(m, 0, result, half, half);
        return result;
    }

    // ---- adapters ------------------------------------------------------------------------

    /// <summary>A Diffusers-format MiniMax-H3 LoRA file under construction.</summary>
    internal sealed class Adapter
    {
        internal readonly List<(string Name, long[] Shape, float[] Data, string Dtype)> Tensors = new();
        internal readonly Dictionary<string, string> Metadata = new(StringComparer.Ordinal);
        private readonly Dictionary<(bool Refiner, int Block, MiniMaxH3LoraTarget Target), (float[] A, float[] B, int Rank)> _terms = new();

        /// <summary>The real adapters' metadata: floating_dtype, format, key_format and alpha; a
        /// null <paramref name="alpha"/> or <paramref name="keyFormat"/> leaves that key out.</summary>
        internal Adapter(float? alpha = 8f, string keyFormat = MiniMaxH3LoraConfig.KeyFormat)
        {
            Metadata["floating_dtype"] = "bfloat16";
            Metadata["format"] = "pt";
            if (keyFormat != null) Metadata["key_format"] = keyFormat;
            if (alpha is { } a) Metadata["alpha"] = a.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        internal Adapter Add(string name, long[] shape, float[] data, string dtype = "F32")
        {
            Tensors.Add((name, shape, data, dtype));
            return this;
        }

        /// <summary>The PEFT key of one factor, e.g. <c>transformer_blocks.0.attn.to_q.lora_A.default.weight</c>.</summary>
        internal static string Key(bool refiner, int block, MiniMaxH3LoraTarget target, char part, string slot = "default") =>
            MiniMaxH3LoraKeys.LoraModule(refiner, block, target) + ".lora_" + part + (slot != null ? "." + slot : "") + ".weight";

        /// <summary>
        /// Add lora_A [rank, in] and lora_B [out, rank] for one target, with values in
        /// ±<paramref name="magnitude"/> rounded to <paramref name="dtype"/>; returns them as stored.
        /// </summary>
        internal (float[] A, float[] B) Term(bool refiner, int block, MiniMaxH3LoraTarget target, int rank, int seed,
            string dtype = "BF16", string slot = "default", float magnitude = .5f)
        {
            var (input, output) = Shape(target);
            float[] a = RoundTo(Rand(rank * input, seed, magnitude), dtype);
            float[] b = RoundTo(Rand(output * rank, seed + 1000, magnitude), dtype);
            return Term(refiner, block, target, rank, a, b, dtype, slot);
        }

        /// <summary>Add a term with the given factors (stored as given; F32 keeps them exact).</summary>
        internal (float[] A, float[] B) Term(bool refiner, int block, MiniMaxH3LoraTarget target, int rank, float[] a, float[] b,
            string dtype = "F32", string slot = "default")
        {
            var (input, output) = Shape(target);
            Add(Key(refiner, block, target, 'A', slot), [rank, input], a, dtype);
            Add(Key(refiner, block, target, 'B', slot), [output, rank], b, dtype);
            _terms[(refiner, block, target)] = (a, b, rank);
            return (a, b);
        }

        /// <summary>Every target of every trunk and refiner block, seeded per term.</summary>
        internal Adapter AllTargets(int blocks, int refinerBlocks, int rank, int seed = 1, string dtype = "BF16")
        {
            for (int pass = 0; pass < 2; pass++)
            {
                bool refiner = pass == 1;
                for (int b = 0; b < (refiner ? refinerBlocks : blocks); b++)
                    for (int t = 0; t < MiniMaxH3LoraKeys.TargetCount; t++)
                        Term(refiner, b, (MiniMaxH3LoraTarget)t, rank, seed + 97 * (b + (refiner ? 50 : 0)) + 7 * t, dtype);
            }
            return this;
        }

        /// <summary>The factors written for a target, or null.</summary>
        internal (float[] A, float[] B, int Rank)? Factors(bool refiner, int block, MiniMaxH3LoraTarget target) =>
            _terms.TryGetValue((refiner, block, target), out var f) ? f : null;

        internal IEnumerable<(bool Refiner, int Block, MiniMaxH3LoraTarget Target)> Targets => _terms.Keys;

        /// <summary>
        /// The update of one target as the loader must apply it, [out][in]: <paramref name="scale"/>
        /// · B · A, with fc1's row halves exchanged into the checkpoint's [gate; value] order
        /// unless <paramref name="swapFc1"/> is false (the negative control).
        /// </summary>
        internal float[] Delta(bool refiner, int block, MiniMaxH3LoraTarget target, float scale, bool swapFc1 = true)
        {
            var (a, b, rank) = Factors(refiner, block, target) ?? throw new InvalidOperationException("No such term.");
            var (input, output) = Shape(target);
            var delta = Product(b, a, output, rank, input, scale);
            return target == MiniMaxH3LoraTarget.Fc1 && swapFc1 ? SwapRowHalves(delta, output, input) : delta;
        }

        /// <summary>
        /// Every update merged per GGUF weight ([out][in], q/k/v placed in their qkv_proj
        /// thirds): the deltas <see cref="WriteTransformer"/> adds for a merged checkpoint.
        /// </summary>
        internal Dictionary<string, float[]> MergedDeltas(float scale, bool swapFc1 = true)
        {
            var result = new Dictionary<string, float[]>(StringComparer.Ordinal);
            foreach (var (refiner, block, target) in _terms.Keys)
            {
                var (weight, firstRow) = GgufWeight(refiner, block, target);
                var (input, output) = Shape(target);
                int rows = target is MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V ? 3 * Inner : output;
                if (!result.TryGetValue(weight, out var merged)) result[weight] = merged = new float[rows * input];
                var delta = Delta(refiner, block, target, scale, swapFc1);
                for (int i = 0; i < delta.Length; i++) merged[firstRow * input + i] += delta[i];
            }
            return result;
        }
    }

    /// <summary>Round values to what <paramref name="dtype"/> stores (BF16 by truncation, as <see cref="Encode"/> writes it).</summary>
    internal static float[] RoundTo(float[] values, string dtype) => dtype switch
    {
        "BF16" => values.Select(v => BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(v) & 0xFFFF0000u)).ToArray(),
        "F16" => values.Select(v => (float)(Half)v).ToArray(),
        _ => values,
    };

    internal static byte[] Encode(float[] data, string dtype)
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
            case "F64":
            {
                var raw = new byte[data.Length * 8];
                for (int i = 0; i < data.Length; i++) BitConverter.TryWriteBytes(raw.AsSpan(i * 8), (double)data[i]);
                return raw;
            }
            default: throw new ArgumentException(dtype);
        }
    }

    /// <summary>Write <paramref name="adapter"/> as safetensors in <see cref="Dir"/>; returns its path.</summary>
    internal string WriteAdapter(Adapter adapter, string fileName = Weights8)
    {
        string path = Path.Combine(Dir, fileName);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        var header = new JsonObject();
        if (adapter.Metadata.Count > 0)
        {
            var meta = new JsonObject();
            foreach (var (k, v) in adapter.Metadata) meta[k] = v;
            header["__metadata__"] = meta;
        }
        var blobs = new List<byte[]>();
        long offset = 0;
        foreach (var (tensor, shape, data, dtype) in adapter.Tensors)
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

    // ---- plug-in configs -------------------------------------------------------------------

    /// <summary>
    /// Write a <c>minimax-h3-lora</c> plug-in config naming <paramref name="weightsFileName"/>
    /// beside it (the shipped manifests' schema with a plain weights path); returns its path. A
    /// null <paramref name="alpha"/> or <paramref name="scale"/> leaves that key out.
    /// </summary>
    internal string WriteManifest(string fileName = "turbo-8step.json", string weightsFileName = Weights8, int steps = 8,
        float? alpha = 8f, float? scale = 1f, string name = "Test Turbo 8")
    {
        var root = new JsonObject
        {
            ["type"] = MiniMaxH3LoraConfig.Type,
            ["name"] = name,
            ["weights"] = weightsFileName,
            ["partition"] = "fl2va",
            ["key_format"] = MiniMaxH3LoraConfig.KeyFormat,
        };
        if (alpha is { } a) root["alpha"] = a;
        if (scale is { } s) root["scale"] = s;
        root["sampling"] = JsonNode.Parse(
            $$"""{ "steps": {{steps}}, "schedule": "linspace", "video_shift": 6.0, "audio_shift": 3.0, "sampler": "euler", "cfg": 1.0 }""");
        string path = Path.Combine(Dir, fileName);
        File.WriteAllText(path, root.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        return path;
    }

    /// <summary>Write any text file in <see cref="Dir"/>; returns its path.</summary>
    internal string WriteText(string fileName, string text)
    {
        string path = Path.Combine(Dir, fileName);
        File.WriteAllText(path, text);
        return path;
    }
}
