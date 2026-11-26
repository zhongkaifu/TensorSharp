// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;

namespace InferenceWeb.Tests;

/// <summary>
/// A tiny MiniMax-H3 audio-video denoiser written at test time from a seed: MiniMax-H3-Image's synthetic transformer
/// (<see cref="MiniMaxH3ImageFixture"/>: the release's tensor names and dtype mix, 2 blocks and 1 token-refiner block,
/// hidden 256, 4 heads of 64, FFN 512, RoPE on 48 dims) with the audio stream the video release has instead of the image
/// release's PDD head bank: audio_patch_proj [32, hidden], one video head [hidden, 96] and an audio head [hidden, 32]. Its
/// heads of 64 run on every backend's flash attention, so one fixture serves ggml_cpu, ggml_cuda, ggml_metal and the tensor
/// executors. A Q4_K fixture's projections are the pinned bytes of the image fixture's (<see cref="MiniMaxH3ImagePinnedQ4K"/>),
/// so it is the same file on every machine. No metadata, named for the FL2VA checkpoint.
/// </summary>
internal sealed class MiniMaxH3VideoFixture : IDisposable
{
    internal const int Hidden = MiniMaxH3ImageFixture.Hidden, Heads = MiniMaxH3ImageFixture.Heads, HeadDim = MiniMaxH3ImageFixture.HeadDim;
    internal const int Inner = MiniMaxH3ImageFixture.Inner, Ffn = MiniMaxH3ImageFixture.Ffn, TextDim = MiniMaxH3ImageFixture.TextDim;
    internal const int PatchDim = MiniMaxH3ImageFixture.PatchDim, AudioDim = 32;
    internal const int Blocks = MiniMaxH3ImageFixture.Blocks, RefinerBlocks = MiniMaxH3ImageFixture.RefinerBlocks;

    internal MiniMaxH3VideoFixture(int projections, int seed = 2051)
    {
        Projections = projections;
        Dir = Path.Combine(Path.GetTempPath(), "ts-h3vid-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Dir);
        DitPath = Path.Combine(Dir, $"minimax_h3_fl2va_tiny-{MiniMaxH3ImageFixture.Name(projections)}.gguf");
        // A Q4_K fixture takes the reference lane's quantization of the (shared) trunk, so every machine writes the same bytes.
        MiniMaxH3ImageFixture.Write(DitPath, Tensors(projections, seed), projections == MiniMaxH3ImageFixture.Q4_K && seed == 2051
            ? MiniMaxH3ImagePinnedQ4K.Encode : null);
    }

    internal string Dir { get; }
    internal int Projections { get; }
    internal string DitPath { get; }

    public void Dispose()
    {
        try { Directory.Delete(Dir, recursive: true); } catch { /* best effort */ }
    }

    /// <summary>The image fixture's tensors with the video release's heads and audio stream (seeded after the image ones, so
    /// the trunk is the image fixture's bit for bit).</summary>
    internal static List<MiniMaxH3ImageFixture.T> Tensors(int projections, int seed)
    {
        var tensors = MiniMaxH3ImageFixture.Tensors(projections, seed)
            .Where(t => t.Name is not ("final_layer.video_out.weight" or "final_layer.video_out.bias")).ToList();
        var rng = new Random(seed ^ 0x5a5a);
        float[] Values(long n, float scale) =>
            Enumerable.Range(0, checked((int)n)).Select(_ => scale * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        void Add(string name, int type, float scale, params long[] shape) =>
            tensors.Add(new MiniMaxH3ImageFixture.T(name, shape, type, Values(shape.Aggregate(1L, (a, b) => a * b), scale)));
        const int F32 = MiniMaxH3ImageFixture.F32;
        Add("audio_patch_proj.weight", F32, 1.2f / MathF.Sqrt(AudioDim), AudioDim, Hidden);
        Add("audio_patch_proj.bias", F32, 0.05f, Hidden);
        Add("final_layer.video_out.weight", F32, 1.2f / MathF.Sqrt(Hidden), Hidden, PatchDim);
        Add("final_layer.video_out.bias", F32, 0.05f, PatchDim);
        Add("final_layer.audio_out.weight", F32, 1.2f / MathF.Sqrt(Hidden), Hidden, AudioDim);
        Add("final_layer.audio_out.bias", F32, 0.05f, AudioDim);
        return tensors;
    }

    /// <summary>
    /// A Diffusers-format LoRA plug-in for this transformer (every q, k, v, out, fc1 and fc2 of the two blocks and the refiner
    /// block, F32 factors of rank <paramref name="rank"/>, alpha 8) and its <c>minimax-h3-lora</c> config, written beside it;
    /// returns the config's path.
    /// </summary>
    internal string WriteLora(int rank = 4, int seed = 911)
    {
        var adapter = new MiniMaxH3LoraFixtures.Adapter();
        var rng = new Random(seed);
        float[] Rand(long n, float scale) => Enumerable.Range(0, checked((int)n)).Select(_ => scale * (float)(rng.NextDouble() * 2 - 1)).ToArray();
        (int In, int Out) Shape(MiniMaxH3LoraTarget t) => t switch
        {
            MiniMaxH3LoraTarget.Q or MiniMaxH3LoraTarget.K or MiniMaxH3LoraTarget.V => (Hidden, Inner),
            MiniMaxH3LoraTarget.Out => (Inner, Hidden),
            MiniMaxH3LoraTarget.Fc1 => (Hidden, 2 * Ffn),
            _ => (Ffn, Hidden),
        };
        for (int pass = 0; pass < 2; pass++)
        {
            bool refiner = pass == 1;
            for (int b = 0; b < (refiner ? RefinerBlocks : Blocks); b++)
                for (int t = 0; t < MiniMaxH3LoraKeys.TargetCount; t++)
                {
                    var target = (MiniMaxH3LoraTarget)t;
                    var (input, output) = Shape(target);
                    // Factors about 1 / sqrt(width): the update moves a step by tens of percent, as a step-distillation adapter's.
                    adapter.Add(MiniMaxH3LoraFixtures.Adapter.Key(refiner, b, target, 'A'), new long[] { rank, input },
                        Rand((long)rank * input, 1f / MathF.Sqrt(input)));
                    adapter.Add(MiniMaxH3LoraFixtures.Adapter.Key(refiner, b, target, 'B'), new long[] { output, rank },
                        Rand((long)output * rank, 1f / MathF.Sqrt(rank)));
                }
        }
        var writer = new MiniMaxH3LoraFixtures();
        try
        {
            File.Copy(writer.WriteAdapter(adapter, MiniMaxH3LoraFixtures.Weights8), Path.Combine(Dir, MiniMaxH3LoraFixtures.Weights8), overwrite: true);
            string manifest = writer.WriteManifest("tiny-turbo-8step.json");
            string path = Path.Combine(Dir, "tiny-turbo-8step.json");
            File.Copy(manifest, path, overwrite: true);
            return path;
        }
        finally
        {
            writer.Dispose();
        }
    }
}
