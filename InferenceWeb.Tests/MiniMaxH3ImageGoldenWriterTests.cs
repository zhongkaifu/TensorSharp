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
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>Runs only when TS_TEST_FREEZE_H3IMG_GOLDENS names a directory (read, never set, by the test).</summary>
public sealed class GoldenFreezeFactAttribute : FactAttribute
{
    public GoldenFreezeFactAttribute()
    {
        if (string.IsNullOrEmpty(Environment.GetEnvironmentVariable(MiniMaxH3ImageGoldenWriterTests.DirectoryVariable)))
            Skip = $"Freezes the legacy MiniMax-H3-Image outputs only when {MiniMaxH3ImageGoldenWriterTests.DirectoryVariable} names a directory.";
    }
}

/// <summary>
/// The freeze of stage S10 (docs/design/diffusion-runtime.md section 1.6), deleted with the legacy oracle it runs: on the
/// process's pinned GGML backend, every case of <see cref="MiniMaxH3ImageGoldens"/> through the legacy denoiser and decoder
/// (the outputs frozen) and through the runtime (whether it gave their bytes, and how far it was), written as this lane's
/// file, plus the host tables' lane. Category Bench: not part of the CI filter.
/// </summary>
[Trait("Category", "Bench")]
public sealed class MiniMaxH3ImageGoldenWriterTests(ITestOutputHelper output)
{
    internal const string DirectoryVariable = "TS_TEST_FREEZE_H3IMG_GOLDENS";
    private const int F32 = MiniMaxH3ImageFixture.F32, BF16 = MiniMaxH3ImageFixture.BF16, Q8_0 = MiniMaxH3ImageFixture.Q8_0,
        Q4_K = MiniMaxH3ImageFixture.Q4_K;

    internal static readonly (string Request, int Format)[] DenoiserCases =
        new[] { F32, BF16, Q8_0, Q4_K }.SelectMany(f => new[] { ("TextToImage", f), ("WithReferences", f) })
            .Concat(new[] { ("Wide", BF16), ("Wide", Q4_K), ("ManySegments", BF16), ("ManySegments", Q4_K) }).ToArray();

    [GoldenFreezeFact]
    public void FreezeTheLegacyOutputsOfThisLane()
    {
        string dir = Environment.GetEnvironmentVariable(DirectoryVariable)!;
        Directory.CreateDirectory(dir);
        BackendType backend = TestGates.PinnedGgmlBackend;
        GgmlBasicOps.EnsureBackendAvailable(backend switch
        {
            BackendType.GgmlMetal => GgmlBackendType.Metal,
            BackendType.GgmlCuda => GgmlBackendType.Cuda,
            _ => GgmlBackendType.Cpu,
        });
        string lane = MiniMaxH3ImageGoldens.LaneKey(backend);
        var cases = new JsonObject();

        foreach (var (request, format) in DenoiserCases)
        {
            using var fx = new MiniMaxH3ImageFixture(format);
            var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
            using var legacy = new MiniMaxH3ImageLegacyDenoiser(fx.DitPath, backend);
            using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
            var (want, got) = MiniMaxH3ImageGoldens.Run(family, legacy, MiniMaxH3ImageGoldens.Named(request), also: runtime);
            bool exact = want.Zip(got).All(p => MiniMaxH3ImageGoldens.Sha(p.First) == MiniMaxH3ImageGoldens.Sha(p.Second));
            double err = MiniMaxH3ImageGoldens.RelativeError(want, got);
            string name = MiniMaxH3ImageGoldens.DenoiserCase(request, format);
            cases[name] = MiniMaxH3ImageGoldens.CaseJson(MiniMaxH3ImageGoldens.FileSha(fx.DitPath), exact, err, want);
            output.WriteLine($"{lane} {name}: runtime {(exact ? "bit-exact" : $"differs, {err:E3} of the largest value")}");
        }

        foreach (var c in MiniMaxH3ImageGoldens.VaeCases)
        {
            using var fx = new MiniMaxH3ImageVaeFixture(c.Blocks);
            float[] latent = MiniMaxH3ImageGoldens.NormalizedLatent(c.LatentH, c.LatentW, c.Seed);
            int batch = MiniMaxH3ImageGoldens.BatchOn(c, backend);
            RgbImage want;
            using (var legacy = new MiniMaxH3ImageVae(fx.VaePath) { BatchOverride = batch })
                want = legacy.Decode(latent, c.LatentH, c.LatentW);
            using var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend) { BatchLimit = c.Name == "vae/split-4" ? 4 : MiniMaxH3ImageVaeCodec.MaxBatch };
            RgbImage got = codec.Decode(latent, c.LatentH, c.LatentW);
            bool exact = MiniMaxH3ImageGoldens.Sha(want.Pixels) == MiniMaxH3ImageGoldens.Sha(got.Pixels);
            double err = MiniMaxH3ImageGoldens.RelativeError(new[] { want.Pixels }, new[] { got.Pixels });
            cases[c.Name] = MiniMaxH3ImageGoldens.CaseJson(MiniMaxH3ImageGoldens.FileSha(fx.VaePath), exact, err, new[] { want.Pixels });
            output.WriteLine($"{lane} {c.Name} ({want.Width}x{want.Height}, legacy {(batch > 0 ? batch.ToString() : "a row")} a call): runtime {(exact ? "bit-exact" : $"differs, {err:E3}")}");
        }
        Write(dir, lane, cases);

        // The host tables (C#), once a lane: the legacy DiT's AdaLN curve and RoPE tables of a packed request.
        {
            using var fx = new MiniMaxH3ImageFixture(BF16);
            var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
            using var legacy = new MiniMaxH3DiT(fx.DitPath, backend, null);
            using var runtime = new MiniMaxH3ImageDenoiser(fx.DitPath, backend);
            MiniMaxH3PackedLayout layout = MiniMaxH3ImageGoldens.HostTablesLayout(family);
            float[] te = legacy.TimeEmbedding(layout.Timesteps);
            var (cos, sin) = legacy.BuildRope(layout.Positions);
            float[] rte = runtime.TimeEmbedding(layout.Timesteps);
            var (rcos, rsin) = runtime.Rope((float[])layout.Positions.Clone());
            bool exact = MiniMaxH3ImageGoldens.Sha(te) == MiniMaxH3ImageGoldens.Sha(rte) && MiniMaxH3ImageGoldens.Sha(cos) == MiniMaxH3ImageGoldens.Sha(rcos)
                && MiniMaxH3ImageGoldens.Sha(sin) == MiniMaxH3ImageGoldens.Sha(rsin);
            double err = MiniMaxH3ImageGoldens.RelativeError(new[] { te, cos, sin }, new[] { rte, rcos, rsin });
            var host = new JsonObject { ["host/tables"] = MiniMaxH3ImageGoldens.CaseJson(MiniMaxH3ImageGoldens.FileSha(fx.DitPath), exact, err, new[] { te, cos, sin }) };
            Write(dir, MiniMaxH3ImageGoldens.HostLaneKey(), host);
            output.WriteLine($"host tables: runtime {(exact ? "bit-exact" : $"differs, {err:E3}")}");
        }
    }

    private void Write(string dir, string lane, JsonObject cases)
    {
        var doc = new JsonObject
        {
            ["lane"] = lane,
            ["about"] = "The legacy MiniMax-H3-Image denoiser's steps and decoder's pixels on the synthetic fixtures, frozen on this lane " +
                        "before stage S10 deleted the legacy code (InferenceWeb.Tests/MiniMaxH3ImageGoldens.cs reads them).",
            ["ggml"] = "d9fe7bff3d2bf4d077e25c654486d02c321734aa",
            ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
            ["cases"] = cases,
        };
        string path = Path.Combine(dir, MiniMaxH3ImageGoldens.FileName(lane));
        File.WriteAllText(path, doc.ToJsonString(new JsonSerializerOptions { WriteIndented = true }) + "\n");
        output.WriteLine($"wrote {path}");
    }
}
