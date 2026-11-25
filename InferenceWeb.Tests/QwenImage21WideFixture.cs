// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json.Nodes;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

/// <summary>
/// A second synthetic Qwen-Image-2.1 transformer at the released checkpoint's head width (128) and conditioning width
/// (Qwen3-VL-8B's 4096), the widths a LoRA plug-in's norm replacements are checked against, and a plug-in over it with every
/// kind of update the loader builds besides per-step output heads: low-rank factors on every projection, DoRA magnitudes
/// (row scales; on the up half of a fused gate_up too) and replacements of attention and text norm gains. Its frozen legacy
/// outputs are QwenImage21Goldens' "wide-*" cases.
/// </summary>
internal static class QwenImage21WideFixture
{
    internal const int Channels = 16, TextDim = 4096, Heads = 2, HeadDim = 128, Dim = Heads * HeadDim, Layers = 2, Ff = 384;

    private static float[] Rand(int n, int seed, float scale = 1f, float offset = 0f) => QwenImage21DenoiserTests.Rand(n, seed, scale, offset);

    /// <param name="fused">Every block's MLP as one gate_up projection, else separate gate and up projections.</param>
    internal static string WriteTransformer(string dir, bool fused)
    {
        var tensors = new List<(string, long[], float[])>();
        int seed = 900;
        void Add(string name, long input, long output, float scale) =>
            tensors.Add((name + ".weight", new[] { input, output }, Rand((int)(input * output), seed++, scale)));
        void Gain(string name, int count) => tensors.Add((name + ".weight", new long[] { count }, Rand(count, seed++, 0.25f, 1f)));
        // Scales near 1/sqrt(fan-in), so the activations stay of order one through both blocks.
        Add("img_in", Channels, Dim, 0.4f);
        Add("txt_in.in_layer", TextDim, Dim, 0.03f);
        Add("txt_in.out_layer", Dim, Dim, 0.1f);
        Gain("txt_in.text_norm", TextDim);
        Add("time_text_embed.timestep_embedder.linear_1", 256, Dim, 0.1f);
        Add("time_text_embed.timestep_embedder.linear_2", Dim, Dim, 0.1f);
        Add("modulation.1", Dim, 4 * Dim, 0.05f);
        Add("norm_out.linear", Dim, Dim, 0.05f);
        Add("proj_out", Dim, Channels, 0.1f);
        for (int i = 0; i < Layers; i++)
        {
            string p = $"transformer_blocks.{i}.";
            Add(p + "attn.to_q", Dim, Dim, 0.1f);
            Add(p + "attn.to_k", Dim, Dim, 0.1f);
            Add(p + "attn.to_v", Dim, Dim, 0.1f);
            Add(p + "attn.to_out.0", Dim, Dim, 0.1f);
            if (fused) Add(p + "img_mlp.gate_up", Dim, 2 * Ff, 0.1f);
            else
            {
                Add(p + "img_mlp.gate_layer", Dim, Ff, 0.1f);
                Add(p + "img_mlp.proj", Dim, Ff, 0.1f);
            }
            Add(p + "img_mlp.out", Ff, Dim, 0.08f);
            Gain(p + "attn.norm_q", HeadDim);
            Gain(p + "attn.norm_k", HeadDim);
        }
        string path = Path.Combine(dir, $"wide-qwen-image-2.1-{(fused ? "fused" : "separate")}.gguf");
        QwenImage21DenoiserTests.WriteGguf(path, tensors);
        return path;
    }

    /// <summary>One request's inputs at the wide transformer's widths: text with a 2x4 reference's vision slots each.</summary>
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
        var layout = QwenImage21Layout.Build(textRows, references > 0 ? slots.ToArray() : null, shapes, HeadDim);
        return (text, refs, layout, Rand(4 * 6 * Channels, seed + 20));
    }

    /// <summary>A ComfyUI-keyed plug-in (diffusion_model.*.lora_down/lora_up, dora_scale, .diff) over the wide transformer:
    /// factors on every block projection (gate and up by their diffusers names, split from a fused gate_up by the loader)
    /// and on img_in and txt_in; DoRA magnitudes on block 0's q and block 1's up half; replaced norm gains (block 0's q,
    /// block 1's k, the text norm). F32 throughout.</summary>
    internal static string WriteLora(string dir)
    {
        var tensors = new List<(string Name, long[] Shape, float[] Data)>();
        int seed = 1500;
        void Factors(string module, int rank, int input, int output)
        {
            tensors.Add(($"diffusion_model.{module}.lora_down.weight", new long[] { rank, input }, Rand(rank * input, seed++, 0.2f)));
            tensors.Add(($"diffusion_model.{module}.lora_up.weight", new long[] { output, rank }, Rand(output * rank, seed++, 0.2f)));
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
        Factors("txt_in.in_layer", 2, TextDim, Dim);
        tensors.Add(("diffusion_model.transformer_blocks.0.attn.to_q.dora_scale", new long[] { Dim, 1 }, Rand(Dim, seed++, 0.3f, 1.2f)));
        tensors.Add(("diffusion_model.transformer_blocks.1.img_mlp.proj.dora_scale", new long[] { Ff, 1 }, Rand(Ff, seed++, 0.3f, 1.2f)));
        tensors.Add(("diffusion_model.transformer_blocks.0.attn.norm_q.diff", new long[] { HeadDim }, Rand(HeadDim, seed++, 0.05f)));
        tensors.Add(("diffusion_model.transformer_blocks.1.attn.norm_k.diff", new long[] { HeadDim }, Rand(HeadDim, seed++, 0.05f)));
        tensors.Add(("diffusion_model.txt_in.text_norm.diff", new long[] { TextDim }, Rand(TextDim, seed++, 0.05f)));
        string path = Path.Combine(dir, "wide-lora.safetensors");
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
}
