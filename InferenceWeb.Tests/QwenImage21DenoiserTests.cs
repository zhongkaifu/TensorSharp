// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models.QwenImage;
using TensorSharp.GGML;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using NativeWeight = TensorSharp.GGML.QwenImage21Weight;
using NativeBlock = TensorSharp.GGML.QwenImage21Block;
using NativeSegment = TensorSharp.GGML.QwenImage21Segment;

namespace InferenceWeb.Tests;

/// <summary>
/// Qwen-Image-2.1's denoiser on the diffusion runtime against the retired native forward (TSGgml_QwenImage21Forward) on a small
/// synthetic transformer: the same velocities, step for step, through the extract and cached graphs of the prefix cache.
/// </summary>
public sealed class QwenImage21DenoiserTests : IDisposable
{
    internal const int Channels = 16, TextDim = 48, Heads = 2, HeadDim = 32, Dim = Heads * HeadDim, Layers = 2, Ff = 96;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qi21-denoiser-" + Guid.NewGuid().ToString("N"));

    public QwenImage21DenoiserTests() => Directory.CreateDirectory(_dir);

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
    internal static void WriteGguf(string path, IReadOnlyList<(string Name, long[] Dims, float[] Data)> tensors)
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
        foreach (var (name, dims, data) in tensors)
        {
            byte[] bytes = Encoding.UTF8.GetBytes(name);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
            w.Write((uint)dims.Length);
            foreach (long d in dims) w.Write((ulong)d);
            w.Write(0u);
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

    /// <summary>The retired native forward over the same mapped weights.</summary>
    private sealed class Legacy : IDisposable
    {
        private readonly List<GCHandle> _pins = new();
        private readonly NativeBlock[] _blocks;
        private QwenImage21ForwardArgs _args;

        private IntPtr _adapter;

        public Legacy(QwenImage21Weights w, QwenImage21LoraSet lora = null) : this(w)
        {
            if (lora == null) return;
            unsafe
            {
                var source = (QwenImage21Adapter*)lora.AdapterFor(0);
                var blocks = new QwenImage21BlockLora[w.Layers];
                for (int i = 0; i < w.Layers; i++) blocks[i] = lora.Block(i);
                // The set's split of a fused gate_up into two row views, as the retired transformer described it.
                for (int i = 0; i < w.Layers; i++)
                    if (lora.SplitGateUp[i] && w.Blocks[i].FusedGateUp)
                    {
                        var g = w.Blocks[i].Gate;
                        long rowBytes = g.Bytes / g.Ne1;
                        _blocks[i].Gate = new NativeWeight { Data = g.Data, Type = g.Format, Ne0 = g.Ne0, Ne1 = w.Ff, Bytes = w.Ff * rowBytes };
                        _blocks[i].Up = new NativeWeight { Data = g.Data + (nint)(w.Ff * rowBytes), Type = g.Format, Ne0 = g.Ne0, Ne1 = w.Ff, Bytes = w.Ff * rowBytes };
                    }
                var adapter = *source;
                adapter.NumLayers = w.Layers;
                adapter.Blocks = Pin(blocks);
                _adapter = Pin(new[] { adapter });
            }
        }

        public Legacy(QwenImage21Weights w)
        {
            static NativeWeight W(TensorSharp.Models.QwenImage21.QwenImage21Weight x) =>
                new() { Data = x.Data, Type = x.Format, Ne0 = x.Ne0, Ne1 = x.Ne1, Bytes = x.Bytes };
            _blocks = new NativeBlock[w.Layers];
            for (int i = 0; i < w.Layers; i++)
            {
                var b = w.Blocks[i];
                _blocks[i] = new NativeBlock
                {
                    Q = W(b.Q), K = W(b.K), V = W(b.V), Out = W(b.Out), Gate = W(b.Gate),
                    Up = b.Up == null ? default : W(b.Up), Down = W(b.Down), NormQ = b.NormQ.Data, NormK = b.NormK.Data,
                };
            }
            _args = new QwenImage21ForwardArgs
            {
                ImageIn = W(w.ImageIn), TextIn = W(w.TextIn), TextOut = W(w.TextOut), TimeIn = W(w.TimeIn), TimeOut = W(w.TimeOut),
                Modulation = W(w.Modulation), NormOut = W(w.NormOut), ProjOut = W(w.ProjOut), TextNorm = w.TextNorm.Data,
                StructBytes = Marshal.SizeOf<QwenImage21ForwardArgs>(),
                Dim = w.Dim, Heads = w.Heads, HeadDim = w.HeadDim, Channels = w.Channels, TextDim = w.TextDim,
                NumLayers = w.Layers, Eps = 1e-6f,
            };
        }

        private IntPtr Pin(Array a)
        {
            var h = GCHandle.Alloc(a, GCHandleType.Pinned);
            _pins.Add(h);
            return h.AddrOfPinnedObject();
        }

        public (float[] Output, QwenImage21ForwardPath Path) Forward(float[] target, ConditioningSet text, LatentRef[] references,
            QwenImage21Layout layout, float sigma, ulong cacheKey)
        {
            float[] images = references.SelectMany(r => r.Rows).Concat(target).ToArray();
            var segments = layout.Segments.Select(s => new NativeSegment
            {
                Start = s.Start, End = s.End, SourceStart = s.SourceStart, IsImage = s.IsImage ? 1 : 0,
            }).ToArray();
            var output = new float[target.Length];
            var a = _args;
            a.Images = Pin(images); a.Text = Pin(text.Hidden); a.TimeEmbedding = Pin(QwenImage21Denoiser.TimeEmbedding(sigma, false));
            a.Cos = Pin(layout.Cos); a.Sin = Pin(layout.Sin); a.Output = Pin(output); a.Blocks = Pin(_blocks); a.Segments = Pin(segments);
            a.ImageSeq = images.Length / Channels; a.TextSeq = text.Rows; a.TotalSeq = layout.Rows; a.PrefixSeq = layout.Prefix;
            a.NumSegments = segments.Length; a.PrefixCacheKey = cacheKey; a.Adapter = _adapter;
            QwenImage21ForwardPath path = GgmlBasicOps.QwenImage21Forward(in a);
            return (output, path);
        }

        /// <summary>The retired pure-C# transformer (the cpu backend's) on the same descriptors.</summary>
        public float[] ForwardManaged(float[] target, ConditioningSet text, LatentRef[] references, QwenImage21Layout layout, float sigma,
            ref QwenImage21ManagedPrefix store)
        {
            _managed ??= new QwenImage21ManagedDiT(_args, _blocks);
            float[] images = references.SelectMany(r => r.Rows).Concat(target).ToArray();
            var segments = layout.Segments.Select(s => new NativeSegment
            {
                Start = s.Start, End = s.End, SourceStart = s.SourceStart, IsImage = s.IsImage ? 1 : 0,
            }).ToArray();
            var output = new float[target.Length];
            store ??= new QwenImage21ManagedPrefix(QwenImage21PrefixCacheType.Auto);
            _managed.Forward(images, images.Length / Channels, text.Hidden, text.Rows, QwenImage21Denoiser.TimeEmbedding(sigma, false),
                layout.Cos, layout.Sin, segments, layout.Prefix, layout.Rows, _adapter, store, output);
            return output;
        }

        private QwenImage21ManagedDiT _managed;

        public void Dispose()
        {
            _managed?.Dispose();
            foreach (var h in _pins) h.Free();
        }
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    public void Program_matches_the_retired_native_forward_through_the_prefix_cache(int references)
    {
        string path = WriteTransformer(_dir);
        using var denoiser = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend);
        using var legacy = new Legacy(denoiser.Weights);
        var (text, refs, layout, target) = Request(references, 7);
        ulong key = 0xC0FFEE00UL + (ulong)references;
        try
        {
            float[] x = (float[])target.Clone();
            foreach (var (sigma, step) in new[] { (1f, 0), (0.75f, 1), (0.3f, 2) })
            {
                var (expected, path0) = legacy.Forward(x, text, refs, layout, sigma, key);
                float[] actual = (float[])denoiser.Predict(x, text, refs, layout, sigma, step, false, null).Clone();
                Assert.Equal(step == 0 ? QwenImage21ForwardPath.Extract : QwenImage21ForwardPath.Cached, path0);
                Assert.Equal(step == 0 ? QwenImage21Denoiser.Mode.Extract : QwenImage21Denoiser.Mode.Cached, denoiser.LastMode);
                Assert.Equal(expected.Length, actual.Length);
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"step {step}, value {i}: legacy {expected[i]:R}, runtime {actual[i]:R}");
                for (int i = 0; i < x.Length; i++) x[i] += (sigma - 0.5f) * actual[i] * 0.1f;
            }
        }
        finally
        {
            GgmlBasicOps.QwenImage21ReleasePrefixCache(key);
            denoiser.FinishRequest();
        }
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
            Assert.True(Math.Sqrt(err / norm) < 1e-4, $"step {step}: relative L2 {Math.Sqrt(err / norm):E2}");
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

    /// <summary>The retired native forward with the set's updates (its descriptor cut to the tiny transformer's layers).</summary>
    [Theory]
    [InlineData(0, true)]
    [InlineData(1, true)]
    [InlineData(1, false)]
    public unsafe void Program_with_a_LoRA_matches_the_retired_native_forward(int references, bool fused)
    {
        string path = WriteTransformer(_dir, fused);
        string loraPath = WriteLora(_dir);
        using var denoiser = new QwenImage21Denoiser(path, TestGates.PinnedGgmlBackend);
        using var set = QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, denoiser.Weights.Gguf, "", TestGates.PinnedGgmlBackend, 1);
        Assert.Equal(fused, set.SplitGateUp[1]);
        using var legacy = new Legacy(denoiser.Weights, set);
        var (text, refs, layout, target) = Request(references, 21);
        ulong key = 0xBEEF00UL + (ulong)references;
        try
        {
            float[] x = (float[])target.Clone();
            foreach (var (sigma, step) in new[] { (1f, 0), (0.7f, 1) })
            {
                var (expected, _) = legacy.Forward(x, text, refs, layout, sigma, key);
                float[] actual = (float[])denoiser.Predict(x, text, refs, layout, sigma, step, false, set).Clone();
                for (int i = 0; i < expected.Length; i++)
                    Assert.True(BitConverter.SingleToInt32Bits(expected[i]) == BitConverter.SingleToInt32Bits(actual[i]),
                        $"step {step}, value {i}: legacy {expected[i]:R}, runtime {actual[i]:R}");
                for (int i = 0; i < x.Length; i++) x[i] += (sigma - 0.5f) * actual[i] * 0.1f;
            }
            // The plug-in changes the velocity (the comparison above is not vacuous).
            float[] plain = (float[])denoiser.Predict(target, text, refs, layout, 1f, 0, false, null).Clone();
            var (withLora, _) = legacy.Forward(target, text, refs, layout, 1f, 0);
            Assert.NotEqual(plain, withLora);
        }
        finally
        {
            GgmlBasicOps.QwenImage21ReleasePrefixCache(key);
            denoiser.FinishRequest();
        }
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

    /// <summary>Writes this lane's frozen legacy outputs (TS_QWEN21_FREEZE_DIR): the native graph's on the pinned GGML backend
    /// and the pure-C# transformer's on the host lane, each case's steps driven by the legacy velocities. Recorded once, before
    /// the legacy engines were deleted; a no-op without the variable.</summary>
    [Fact]
    public void Freeze_the_retired_engines_outputs()
    {
        string dir = Environment.GetEnvironmentVariable(QwenImage21Goldens.FreezeVariable);
        if (string.IsNullOrWhiteSpace(dir)) return;
        Directory.CreateDirectory(dir);
        BackendType backend = TestGates.PinnedGgmlBackend;
        var native = new Dictionary<string, (string, bool, IReadOnlyList<float[]>)>();
        var host = new Dictionary<string, (string, bool, IReadOnlyList<float[]>)>();
        foreach (var (name, fused, withLora, references) in GoldenCases)
        {
            var (dit, loraPath, sha) = Fixture(fused, withLora);
            using var denoiser = new QwenImage21Denoiser(dit, backend) { FreeMemory = () => -1 };
            using var set = withLora ? QwenImage21LoraSet.Load(new[] { new LoraSpec(loraPath, 0.8f) }, denoiser.Weights.Gguf, "", backend, 1) : null;
            using var legacy = new Legacy(denoiser.Weights, set);
            var (text, refs, layout, target) = Request(references, 31);
            ulong key = 0xF00D00UL + (ulong)name.GetHashCode();
            var legacyOut = new List<float[]>();
            bool exact = true;
            float[] x = (float[])target.Clone();
            try
            {
                foreach (var (sigma, step) in QwenImage21Goldens.Steps)
                {
                    var (expected, _) = legacy.Forward(x, text, refs, layout, sigma, key);
                    float[] actual = denoiser.Predict(x, text, refs, layout, sigma, step, false, set);
                    exact &= expected.AsSpan().SequenceEqual(actual);
                    legacyOut.Add(expected);
                    QwenImage21Goldens.Advance(x, expected, sigma);
                }
            }
            finally
            {
                GgmlBasicOps.QwenImage21ReleasePrefixCache(key);
                denoiser.FinishRequest();
            }
            native["denoiser/" + name] = (sha, exact, legacyOut);

            // The pure-C# transformer took F32 LoRA factors of its own; its cases are the plain transformer's.
            if (withLora) continue;
            QwenImage21ManagedPrefix store = null;
            var managedOut = new List<float[]>();
            x = (float[])target.Clone();
            foreach (var (sigma, _) in QwenImage21Goldens.Steps)
            {
                float[] expected = legacy.ForwardManaged(x, text, refs, layout, sigma, ref store);
                managedOut.Add(expected);
                QwenImage21Goldens.Advance(x, expected, sigma);
            }
            store?.Dispose();
            host["managed/" + name] = (sha, false, managedOut);
        }
        string ggmlLane = MiniMaxH3ImageGoldens.LaneKey(backend), hostLane = MiniMaxH3ImageGoldens.HostLaneKey();
        File.WriteAllText(Path.Combine(dir, MiniMaxH3ImageGoldens.FileName(ggmlLane)), QwenImage21Goldens.File(ggmlLane, native).ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        File.WriteAllText(Path.Combine(dir, MiniMaxH3ImageGoldens.FileName(hostLane)), QwenImage21Goldens.File(hostLane, host).ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
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
        var (text, refs, layout, target) = Request(references, 31);
        var got = new List<float[]>();
        float[] x = (float[])target.Clone();
        foreach (var (sigma, step) in QwenImage21Goldens.Steps)
        {
            float[] v = (float[])denoiser.Predict(x, text, refs, layout, sigma, step, false, set).Clone();
            got.Add(v);
            QwenImage21Goldens.Advance(x, v, sigma);
        }
        denoiser.FinishRequest();
        QwenImage21Goldens.Check("denoiser/" + name, MiniMaxH3ImageGoldens.LaneKey(backend), sha, got, bound: 2e-3);
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
}
