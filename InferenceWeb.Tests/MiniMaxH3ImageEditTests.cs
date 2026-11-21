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
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image editing (docs/design/diffusion-runtime.md section 4) against independent references, with no weights:
/// Pillow's resize filters byte for byte (Fixtures/MiniMaxH3Image/pillow-resize.json), diffusers' reference sizes, canvas,
/// packed layout and the edit's noise stream through the real DiffusionPipeline (diffusers-edit-golden.json, written by
/// eng/validation/MiniMaxH3ImageEdit/diffusers_edit_golden.py with diffusers d5df05ed), the VAE encoder program against a
/// NumPy oracle (vae-encoder-oracle.json), the presentation of the pictures to the text encoder, and the refusals.
/// </summary>
public sealed class MiniMaxH3ImageEditTests(ITestOutputHelper output)
{
    private static readonly float[] Grid = MiniMaxH3ImageSampling.ReleaseSigmas;

    private static JsonDocument Fixture(string name)
    {
        using Stream s = typeof(MiniMaxH3ImageEditTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.MiniMaxH3Image." + name)
            ?? throw new InvalidOperationException($"missing fixture {name}");
        return JsonDocument.Parse(s);
    }

    private static float Bits(JsonElement e) => BitConverter.Int32BitsToSingle(unchecked((int)e.GetUInt32()));
    private static float[] Floats(JsonElement array) => array.EnumerateArray().Select(Bits).ToArray();

    private static RgbImage FromBytes(byte[] rgb, int width, int height)
    {
        var px = new float[rgb.LongLength];
        for (long i = 0; i < px.LongLength; i++) px[i] = rgb[i] / 255f;
        return new RgbImage(width, height, px);
    }

    private static RgbImage Picture(int width, int height, int seed = 1)
    {
        var px = new float[(long)width * height * 3];
        for (long i = 0; i < px.LongLength; i++) px[i] = (byte)((i * 2654435761L + seed * 97) >> 9) / 255f;
        return new RgbImage(width, height, px);
    }

    // ---- reference resizes

    /// <summary>The references' resizes are Pillow's LANCZOS (the VAE and vision copies, as diffusers' VaeImageProcessor
    /// and ComfyUI resize them) and BICUBIC (the Qwen3-VL processor's) on 8 bits, byte for byte with Pillow 13.0.0.</summary>
    [Fact]
    public void ReferenceResizesArePillowsFiltersByteForByte()
    {
        using JsonDocument doc = Fixture("pillow-resize.json");
        int checkedCount = 0;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32();
            RgbImage source = FromBytes(Convert.FromBase64String(c.GetProperty("rgb").GetString()), w, h);
            foreach (JsonElement r in c.GetProperty("resizes").EnumerateArray())
            {
                int tw = r.GetProperty("width").GetInt32(), th = r.GetProperty("height").GetInt32();
                bool bicubic = r.GetProperty("filter").GetString() == "bicubic";
                byte[] want = Convert.FromBase64String(r.GetProperty("rgb").GetString());
                byte[] got = ImageIO.ToRgb8(MiniMaxH3ImageReferences.PillowResize(source, tw, th, bicubic));
                int differ = want.Where((b, i) => b != got[i]).Count();
                Assert.True(differ == 0, $"{w}x{h} -> {tw}x{th} {(bicubic ? "bicubic" : "lanczos")}: {differ} of {want.Length} bytes differ from Pillow");
                checkedCount++;
            }
        }
        output.WriteLine($"{checkedCount} resizes byte for byte with Pillow {doc.RootElement.GetProperty("pillow").GetString()}");
        Assert.Equal(16, checkedCount);
    }

    /// <summary>bound_short_edge for both copies and the processor's smart_resize, against diffusers' and transformers' own
    /// functions over sizes that keep, shrink, grow and round (Python's round half to even).</summary>
    [Fact]
    public void ReferenceSizesAreDiffusersBoundShortEdgeAndQwenSmartResize()
    {
        using JsonDocument doc = Fixture("diffusers-edit-golden.json");
        foreach (JsonElement s in doc.RootElement.GetProperty("sizes").EnumerateArray())
        {
            int w = s.GetProperty("w").GetInt32(), h = s.GetProperty("h").GetInt32();
            int[] vae = s.GetProperty("vae").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] vision = s.GetProperty("vision").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            int[] smart = s.GetProperty("smart").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal((vae[0], vae[1]), MiniMaxH3ImageReferences.VaeCopySize(w, h));
            Assert.Equal((vision[0], vision[1]), MiniMaxH3ImageReferences.BoundShortEdge(w, h, 1024, 1024));
            Assert.Equal((smart[0], smart[1]), MiniMaxH3ImageReferences.VisionCopySize(w, h));
        }
    }

    /// <summary>The VAE copy keeps a short edge inside 1024-2048 and is cropped at the top left to whole 16-pixel cells; the
    /// vision copy's side that is no multiple of 32 is the processor's bicubic resize; a side under 32 pixels is refused.</summary>
    [Fact]
    public void ReferenceCopiesCropToCellsAndRefuseTinySides()
    {
        var policy = new MiniMaxH3ImageFamily(Grid).References;
        RgbImage source = Picture(1500, 1100);
        RgbImage vae = policy.ForVae(source, 0);
        Assert.Equal((1488, 1088), (vae.Width, vae.Height));
        for (int y = 0; y < 3; y++)
            Assert.Equal(source.Pixels.AsSpan(y * 1500 * 3, 1488 * 3).ToArray(), vae.Pixels.AsSpan(y * 1488 * 3, 1488 * 3).ToArray());
        PresentedImage vision = policy.ForVision(Picture(1500, 1024), 2);
        Assert.Equal((1504, 1024, 2), (vision.Image.Width, vision.Image.Height, vision.Index));
        var refused = Assert.Throws<ArgumentException>(() => policy.ForVae(Picture(31, 400), 4));
        Assert.Contains("<Picture 5>", refused.Message);
        Assert.Contains("at least 32 pixels", refused.Message);
        Assert.Throws<ArgumentException>(() => policy.ForVision(Picture(400, 20), 0));
    }

    // ---- canvas

    [Fact]
    public void EditCanvasIsDiffusersGenerationFloor()
    {
        using JsonDocument doc = Fixture("diffusers-edit-golden.json");
        foreach (JsonElement c in doc.RootElement.GetProperty("canvas").EnumerateArray())
        {
            int h = c.GetProperty("h").GetInt32(), w = c.GetProperty("w").GetInt32();
            int[] want = c.GetProperty("canvas").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            Assert.Equal((want[0], want[1]), MiniMaxH3ImageFamily.EditCanvas(w, h));
        }
    }

    /// <summary>An edit's size: an explicit size (sampled at the floor below 4 MP and returned as asked), keepSourceSize
    /// (<c>&lt;Picture 1&gt;</c>'s exact size), the target area at <c>&lt;Picture 1&gt;</c>'s aspect, or the VAE copy's size;
    /// above 20 MP an explicit size is refused. Text to image keeps its own rule.</summary>
    [Fact]
    public void EditCanvasRules()
    {
        var family = new MiniMaxH3ImageFamily(Grid);
        DiffusionReference[] one = { new(Picture(640, 480)) };
        DiffusionCanvas explicitSize = family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1024, Height = 1024 }, null);
        Assert.Equal(((2048, 2048), (1024, 1024)), (explicitSize.Sampling, explicitSize.Output));
        DiffusionCanvas big = family.ResolveCanvas(new DiffusionRequest { References = one, Width = 4096, Height = 4096 }, null);
        Assert.Equal(((4096, 4096), (4096, 4096)), (big.Sampling, big.Output));
        DiffusionCanvas keep = family.ResolveCanvas(new DiffusionRequest { References = one, KeepSourceSize = true }, null);
        Assert.Equal((MiniMaxH3ImageFamily.EditCanvas(640, 480), (640, 480)), (keep.Sampling, keep.Output));
        DiffusionCanvas area = family.ResolveCanvas(new DiffusionRequest { References = new DiffusionReference[] { new(Picture(1500, 1000)) }, TargetArea = 2048L * 2048 }, null);
        Assert.Equal(((2496, 1664), (2496, 1664)), (area.Sampling, area.Output));
        DiffusionCanvas copy = family.ResolveCanvas(new DiffusionRequest { References = one }, null);
        Assert.Equal((MiniMaxH3ImageFamily.EditCanvas(1376, 1024), (1376, 1024)), (copy.Sampling, copy.Output));
        Assert.Contains("at most", Assert.Throws<ArgumentException>(() =>
            family.ResolveCanvas(new DiffusionRequest { References = one, Width = 5120, Height = 5120 }, null)).Message);
        Assert.Throws<ArgumentException>(() =>
            family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1000, Height = 1024 }, null));
        Assert.Throws<ArgumentException>(() =>
            family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1024, Height = 1024, KeepSourceSize = true }, null));
        DiffusionCanvas t2i = family.ResolveCanvas(new DiffusionRequest { Width = 1024, Height = 1024 }, null);
        Assert.Equal(((1024, 1024), (1024, 1024)), (t2i.Sampling, t2i.Output));
    }

    /// <summary>A speed adapter's contract sets the edit's generation floor (pDMD's 1024 px: the asked size, as its diffusers
    /// pipeline samples); a style adapter, or a contract without one, keeps the family's 4 MP floor. Text to image is
    /// unchanged either way.</summary>
    [Fact]
    public void ASpeedContractsFloorSetsTheEditCanvas()
    {
        var family = new MiniMaxH3ImageFamily(Grid);
        DiffusionReference[] one = { new(Picture(640, 480)) };
        AdapterSet With(long floor) => new(new[]
        {
            new DiffusionAdapter
            {
                Name = "speed", Payload = new object(),
                Contract = new SamplingContract { Name = "speed", Sigmas = new[] { 1f, 12f / 13f, 0f }, Heads = HeadPolicy.ByStep, MinGenerationArea = floor },
            },
        });
        DiffusionCanvas pdmd = family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1024, Height = 1024, Adapters = With(1024) }, null);
        Assert.Equal(((1024, 1024), (1024, 1024)), (pdmd.Sampling, pdmd.Output));
        DiffusionCanvas copy = family.ResolveCanvas(new DiffusionRequest { References = one, Adapters = With(1024) }, null);
        Assert.Equal(((1376, 1024), (1376, 1024)), (copy.Sampling, copy.Output));
        DiffusionCanvas family0 = family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1024, Height = 1024, Adapters = With(0) }, null);
        Assert.Equal(((2048, 2048), (1024, 1024)), (family0.Sampling, family0.Output));
        var style = new AdapterSet(new[] { new DiffusionAdapter { Name = "style", Payload = new object() } });
        DiffusionCanvas styled = family.ResolveCanvas(new DiffusionRequest { References = one, Width = 1024, Height = 1024, Adapters = style }, null);
        Assert.Equal(((2048, 2048), (1024, 1024)), (styled.Sampling, styled.Output));
        DiffusionCanvas t2i = family.ResolveCanvas(new DiffusionRequest { Width = 1024, Height = 1024, Adapters = With(1024) }, null);
        Assert.Equal(((1024, 1024), (1024, 1024)), (t2i.Sampling, t2i.Output));
        // A masked edit's selection samples at the contract's floor too (S6's masks with S7's pDMD): a 1024 x 1024 picture
        // whole, and a 384 x 384 crop of a 2048 x 2048 one at its density, no longer raised to 4 MP.
        RgbImage square = Picture(1024, 1024), big = Picture(2048, 2048);
        var whole = new EditMaskOptions { Mask = RectangleMask(1024, 1024, 10, 10, 200, 200) };
        DiffusionCanvas masked = family.ResolveCanvas(new DiffusionRequest
        {
            References = new[] { new DiffusionReference(square) }, Mask = whole, Adapters = With(1024),
        }, EditMask.Create(whole, square));
        Assert.Equal(new DiffusionCanvas((1024, 1024), (1024, 1024)), masked);
        var cropped = new EditMaskOptions { Mask = RectangleMask(2048, 2048, 512, 512, 744, 768), MaskCrop = true, MaskCropPadding = 64 };
        DiffusionCanvas crop = family.ResolveCanvas(new DiffusionRequest
        {
            References = new[] { new DiffusionReference(big) }, Mask = cropped, Adapters = With(1024),
        }, EditMask.Create(cropped, big));
        Assert.Equal(new DiffusionCanvas((384, 384), (2048, 2048)), crop);
    }

    // ---- layout

    /// <summary>The packed layout of text, references and target against diffusers' build_image_packed_sequence: every
    /// row's float32 position, its modality tag, the reference rows, and its timestep at step 1 (the target and the text at
    /// 1 - sigma, the references at 0.999).</summary>
    [Fact]
    public void PackedLayoutIsDiffusersBuildImagePackedSequence()
    {
        using JsonDocument doc = Fixture("diffusers-edit-golden.json");
        var family = new MiniMaxH3ImageFamily(Grid);
        foreach (JsonElement l in doc.RootElement.GetProperty("layouts").EnumerateArray())
        {
            string name = l.GetProperty("name").GetString();
            int[] tags = l.GetProperty("tags").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var refs = l.GetProperty("references").EnumerateArray().Select((r, k) =>
            {
                int h = r[0].GetInt32(), w = r[1].GetInt32();
                return new LatentRef(k, new LatentShape(24, 1, h, w), new float[(long)(h / 2) * (w / 2) * 96]);
            }).ToArray();
            int th = l.GetProperty("target")[0].GetInt32(), tw = l.GetProperty("target")[1].GetInt32();
            var text = new ConditioningSet(new float[tags.Length * 4], tags.Length, 4, tags);
            DiffusionLayout layout = family.Pack(text, refs, new LatentShape(24, 1, th, tw), new StepState(0, 6, 1f, Grid[1], 0, 1));

            float[] want = Floats(l.GetProperty("positions"));
            Assert.Equal(want.Length, layout.Positions.Length);
            for (int i = 0; i < want.Length; i++)
                Assert.True(BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(layout.Positions[i]),
                    $"{name}: position {i / 3} axis {i % 3} is {layout.Positions[i]:R}, diffusers {want[i]:R}");
            int[] tokenTags = l.GetProperty("tokenTags").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            float[] unique = Floats(l.GetProperty("uniqueTimesteps"));
            int[] index = l.GetProperty("timestepIndex").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var rowTag = new int[layout.Rows];
            var rowT = new float[layout.Rows];
            foreach (DiffusionSegment s in layout.Segments)
                for (int r = s.Start; r < s.Start + s.Length; r++) { rowTag[r] = s.Modality; rowT[r] = layout.Timesteps[s.Timestep]; }
            Assert.Equal(tokenTags, rowTag);
            for (int r = 0; r < layout.Rows; r++)
                Assert.Equal(BitConverter.SingleToInt32Bits(unique[index[r]]), BitConverter.SingleToInt32Bits(rowT[r]));
            Assert.Equal(l.GetProperty("referenceRows").GetInt32(), layout.TargetStart - tags.Length);
            output.WriteLine($"{name}: {layout.Rows} rows bit for bit");
        }
    }

    // ---- the edit's noise stream

    private static LatentPosterior GoldenPosterior(JsonElement reference)
    {
        int h = reference.GetProperty("height").GetInt32(), w = reference.GetProperty("width").GetInt32();
        float[] moments = Floats(reference.GetProperty("moments"));
        long half = 24L * h * w;
        return new LatentPosterior(new LatentShape(24, 1, h, w), moments[..(int)half], moments[(int)half..]);
    }

    private static void AssertBits(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        int differ = 0, first = -1;
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i])) { differ++; if (first < 0) first = i; }
        Assert.True(differ == 0, $"{what}: {differ} of {want.Length} values differ, the first at {first} ({(first >= 0 ? got[first] : 0):R} vs {(first >= 0 ? want[first] : 0):R})");
    }

    /// <summary>Each reference's rows before augmentation: its posterior sample off the stream, rounded to float16, cropped
    /// to whole patches and normalized, bit for bit with diffusers' MiniMaxH3ImageEditVaeEncoderStep.</summary>
    [Fact]
    public void PosteriorSamplesAreDiffusersVaeEncoderStep()
    {
        using JsonDocument doc = Fixture("diffusers-edit-golden.json");
        JsonElement stream = doc.RootElement.GetProperty("stream");
        var family = new MiniMaxH3ImageFamily(Grid);
        INoiseSource noise = family.Noise(stream.GetProperty("seed").GetInt64());
        int k = 0;
        foreach (JsonElement r in stream.GetProperty("references").EnumerateArray())
        {
            LatentRef rows = family.References.FromPosterior(GoldenPosterior(r), k, noise);
            AssertBits(Floats(r.GetProperty("rowsBeforeAugmentation")), rows.Rows, $"reference {k + 1}'s rows");
            k++;
        }
    }

    /// <summary>The whole stream through DiffusionPipeline with MiniMax-H3-Image's family: the references' posterior samples,
    /// then the target noise, then the references' augmentation, from one generator (diffusers' block order), bit for bit
    /// with what diffusers' pipeline hands its first step.</summary>
    [Fact]
    public void TheEditStreamIsDiffusersOrderThroughThePipeline()
    {
        using JsonDocument doc = Fixture("diffusers-edit-golden.json");
        JsonElement stream = doc.RootElement.GetProperty("stream");
        JsonElement[] goldenRefs = stream.GetProperty("references").EnumerateArray().ToArray();
        var family = new StubbedFamily(new MiniMaxH3ImageFamily(Grid))
        {
            Posteriors = goldenRefs.Select(GoldenPosterior).ToArray(),
            Canvas = DiffusionCanvas.Of(stream.GetProperty("target")[1].GetInt32() * 16, stream.GetProperty("target")[0].GetInt32() * 16),
        };
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCpu },
            new ResidencyPlanner(() => null, _ => { }), _ => { });
        pipeline.Run(new DiffusionRequest
        {
            Prompt = "Put <Picture 1> beside <Picture 2> under <Picture 3>.",
            References = goldenRefs.Select(_ => new DiffusionReference(Picture(64, 64))).ToArray(),
            Seed = stream.GetProperty("seed").GetInt64(),
        });
        Assert.True(family.FirstStepValue.HasValue, "the denoiser never ran");
        // The generator's words and their order are exact: every value comes from the same uniforms. Box-Muller's
        // log/sin/cos differ in the last bits between PyTorch's x86 AVX2 kernels (avx_mathfun) and .NET's math library,
        // which TorchCpuRandn's own stored-torch tests allow for (2e-6); the posterior samples above, rounded to float16,
        // are bit for bit.
        AssertClose(Floats(stream.GetProperty("targetNoiseRows")), family.FirstStep.Latent, 2e-6, 0, "the target's noise");
        for (int k = 0; k < goldenRefs.Length; k++)
            AssertClose(Floats(goldenRefs[k].GetProperty("rows")), family.FirstStep.References[k], 2e-9, 1.2e-7, $"reference {k + 1}'s augmented rows");
    }

    private void AssertClose(float[] want, float[] got, double absolute, double relative, string what)
    {
        Assert.Equal(want.Length, got.Length);
        int exact = 0;
        double worst = 0;
        for (int i = 0; i < want.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(want[i]) == BitConverter.SingleToInt32Bits(got[i])) exact++;
            double err = Math.Abs(want[i] - got[i]);
            worst = Math.Max(worst, err);
            Assert.True(err <= absolute + relative * Math.Abs(want[i]), $"{what}: value {i} is {got[i]:R}, diffusers {want[i]:R}");
        }
        output.WriteLine($"{what}: {exact} of {want.Length} bit for bit, max |difference| {worst:E2}");
    }

    // ---- refusals

    [Fact]
    public void EditsAreRefusedBeforeAnyComponentRuns()
    {
        var family = new StubbedFamily(new MiniMaxH3ImageFamily(Grid));
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCpu },
            new ResidencyPlanner(() => null, _ => { }), _ => { });
        DiffusionReference[] Refs(int n) => Enumerable.Range(0, n).Select(i => new DiffusionReference(Picture(64, 64, i))).ToArray();
        string Refused(DiffusionRequest r) => Assert.Throws<ArgumentException>(() => pipeline.Run(r)).Message;

        Assert.Contains("at most 16 reference images; this request has 17", Refused(new DiffusionRequest { Prompt = "x", References = Refs(17) }));
        Assert.Contains("cfg must be 1", Refused(new DiffusionRequest { Prompt = "x", References = Refs(1), Cfg = 2f }));
        Assert.Contains("no negative prompt", Refused(new DiffusionRequest { Prompt = "x", References = Refs(1), NegativePrompt = "blurry" }));
        Assert.Contains("at least 32 pixels", Refused(new DiffusionRequest
        {
            Prompt = "x",
            References = new[] { new DiffusionReference(Picture(64, 64)), new DiffusionReference(Picture(16, 64)) },
        }));
        Assert.Equal(0, family.Steps);
    }

    /// <summary>The row budget: an edit whose denoising step (its weights beside the step's working memory for the
    /// references' rows) does not fit the device's free memory is refused before it runs, with its rows named, by the same
    /// test (and the same exception) that refuses any request the device cannot hold; the same text to image is not.</summary>
    [Fact]
    public void EditsOverTheRowBudgetAreRefusedAndTextToImageIsNot()
    {
        var family = new StubbedFamily(new MiniMaxH3ImageFamily(Grid)) { DenoiserWeights = 8L << 30, BytesPerRow = 512 * 1024 };
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCuda },
            new ResidencyPlanner(() => 10L << 30, _ => { }, freeBytes: () => 11L << 30), _ => { });
        DiffusionReference[] refs = Enumerable.Range(0, 4).Select(i => new DiffusionReference(Picture(64, 64, i))).ToArray();
        string message = Assert.Throws<RequestRefusedException>(() => pipeline.Run(new DiffusionRequest
        {
            Prompt = "x", References = refs, Width = 2048, Height = 2048,
        })).Message;
        output.WriteLine(message);
        Assert.Contains("too large for the device", message);
        Assert.Contains("4,096 rows", message);      // 4 references at 1024x1024: 1,024 rows each
        Assert.Contains("a denoising step would need about", message);
        Assert.Contains("fewer or smaller reference images", message);
        Assert.Equal(0, family.Steps);
        pipeline.Run(new DiffusionRequest { Prompt = "x", Width = 2048, Height = 2048 });
        Assert.Equal(6, family.Steps);
    }

    /// <summary>The row budget and the device check are one test: an edit over the plan's budget (its headroom) but within
    /// the free memory runs, as a text to image does (bf16 at 4096x4096 ran on the A40 with its plan over the budget).</summary>
    [Fact]
    public void AnEditOverTheBudgetButWithinTheFreeMemoryRuns()
    {
        var family = new StubbedFamily(new MiniMaxH3ImageFamily(Grid)) { DenoiserWeights = 8L << 30, BytesPerRow = 512 * 1024 };
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCuda },
            new ResidencyPlanner(() => 12L << 30, _ => { }, freeBytes: () => 16L << 30), _ => { });
        DiffusionReference[] refs = Enumerable.Range(0, 4).Select(i => new DiffusionReference(Picture(64, 64, i))).ToArray();
        pipeline.Run(new DiffusionRequest { Prompt = "x", References = refs, Width = 2048, Height = 2048 });
        Assert.Equal(6, family.Steps);
    }

    // ---- masked edits (docs/design/diffusion-runtime.md section 4.5)

    /// <summary>A decoded PNG's exact bytes: an RGBA picture of random samples, some fully transparent with RGB under them.</summary>
    private static (RgbImage Image, byte[] Png) RandomRgba(int width, int height, int seed)
    {
        var rng = new Random(seed);
        var px = new float[(long)width * height * 3];
        var alpha = new float[(long)width * height];
        for (long i = 0; i < px.LongLength; i++) px[i] = rng.Next(256) / 255f;
        for (long i = 0; i < alpha.LongLength; i++) alpha[i] = (i % 7 == 0 ? 0 : rng.Next(256)) / 255f;
        byte[] png = ImageIO.EncodePng(new RgbImage(width, height, px, alpha));
        return (ImageIO.Decode(png, preserveAlpha: true), png);
    }

    /// <summary>A decoded picture's bytes, RGBA (alpha 255 when it has none).</summary>
    private static byte[] Rgba8(RgbImage image)
    {
        byte[] rgb = ImageIO.ToRgb8(image);
        var rgba = new byte[(long)image.Width * image.Height * 4];
        for (long i = 0; i < (long)image.Width * image.Height; i++)
        {
            rgba[i * 4] = rgb[i * 3]; rgba[i * 4 + 1] = rgb[i * 3 + 1]; rgba[i * 4 + 2] = rgb[i * 3 + 2];
            rgba[i * 4 + 3] = image.Alpha == null ? (byte)255 : (byte)MathF.Round(image.Alpha[i] * 255f);
        }
        return rgba;
    }

    /// <summary>A grayscale mask: a white rectangle, a gray ramp of <paramref name="ramp"/> pixels after it, black elsewhere.</summary>
    private static RgbImage RectangleMask(int width, int height, int x0, int y0, int x1, int y1, int ramp = 24)
    {
        var px = new float[(long)width * height * 3];
        for (int y = y0; y < y1; y++)
            for (int x = x0; x < Math.Min(width, x1 + ramp); x++)
            {
                float v = x < x1 ? 1f : (byte)(255 - (x - x1) * 10) / 255f;
                for (int c = 0; c < 3; c++) px[((long)y * width + x) * 3 + c] = v;
            }
        return new RgbImage(width, height, px);
    }

    /// <summary>
    /// The composite's gate: through the real pipeline (MiniMax-H3-Image's family, canvas, latent mask, reinjection and
    /// composite; stub components), every pixel the mask gives weight 0 comes back as the source PNG's exact bytes, its
    /// alpha (and the RGB under a transparent pixel) included, at the source's size, whatever the mask's mode, inversion,
    /// feather and crop; the selection is the decode's. Weight 0 is read off the mask file here independently: black for a
    /// grayscale mask, white once inverted, opaque for an alpha mask (feathering only lowers weights inward).
    /// </summary>
    [Theory]
    [InlineData("grayscale", false, 0, false)]
    [InlineData("grayscale", true, 0, false)]
    [InlineData("grayscale", false, 12, false)]
    [InlineData("grayscale", false, 0, true)]
    [InlineData("alpha", false, 4, true)]
    public void AMaskedEditKeepsEveryUnselectedPixelsBytes(string mode, bool invert, int feather, bool crop)
    {
        const int W = 1000, H = 750;
        var (source, png) = RandomRgba(W, H, 17);
        RgbImage mask = RectangleMask(W, H, 300, 200, 640, 520);
        if (mode == "alpha")
        {
            var alpha = new float[W * H];
            for (int i = 0; i < alpha.Length; i++) alpha[i] = 1f - mask.Pixels[i * 3];
            mask = new RgbImage(W, H, new float[W * H * 3], alpha);
        }
        var family = new StubbedFamily(new MiniMaxH3ImageFamily(Grid));
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCpu },
            new ResidencyPlanner(() => null, _ => { }), _ => { });
        DiffusionResult result = pipeline.Run(new DiffusionRequest
        {
            Prompt = "rewrite the sign",
            References = new[] { new DiffusionReference(source), new DiffusionReference(Picture(512, 512, 3)) },
            Mask = new EditMaskOptions
            {
                Mask = mask, MaskMode = mode == "alpha" ? EditMaskMode.Alpha : EditMaskMode.Grayscale,
                MaskInvert = invert, MaskFeather = feather, MaskCrop = crop, MaskCropPadding = 48,
            },
            KeepSourceSize = true,
            Seed = 5,
        });
        Assert.Equal(6, family.Steps);
        Assert.Equal((W, H), (result.Image.Width, result.Image.Height));

        byte[] want = Rgba8(ImageIO.Decode(png, preserveAlpha: true));
        byte[] got = Rgba8(ImageIO.Decode(ImageIO.EncodePng(result.Image), preserveAlpha: true));
        int kept = 0, differ = 0, selectedChanged = 0;
        for (int i = 0; i < W * H; i++)
        {
            bool selected = mode == "alpha" ? mask.Alpha[i] < 1f : mask.Pixels[i * 3] > 0f;
            if (invert) selected = mode == "alpha" ? mask.Alpha[i] > 0f : mask.Pixels[i * 3] < 1f;
            bool same = got.AsSpan(i * 4, 4).SequenceEqual(want.AsSpan(i * 4, 4));
            if (!selected) { kept++; if (!same) differ++; }
            else if (!same) selectedChanged++;
        }
        output.WriteLine($"{mode} invert={invert} feather={feather} crop={crop}: {kept} kept pixels, {differ} differ; " +
                         $"{selectedChanged} selected pixels changed; sampled {result.Recipe.Canvas.Sampling}");
        Assert.True(kept > 0);
        Assert.Equal(0, differ);
        Assert.True(selectedChanged > 0);
    }

    /// <summary>
    /// The latent side of a masked edit: the kept cells (a 16x16 cell of weight 0 everywhere; max-pooled, so a cell with any
    /// selected pixel is edited) end exactly on the source's posterior mean, normalized as the references' rows are (F16, then
    /// the VAE's latents_mean/std); the selected cells are the sampler's (here the initial noise, the stub predicting zero
    /// velocity). The source is encoded once more at the sampling canvas when its reference copy is another size.
    /// </summary>
    [Fact]
    public void TheKeptLatentCellsEndOnTheSourcesNormalizedMean()
    {
        const int W = 1024, H = 1024;
        RgbImage source = Picture(W, H, 9);
        RgbImage mask = RectangleMask(W, H, 256, 256, 513, 512, ramp: 0);  // x 512 is one pixel into the next cell column
        var inner = new MiniMaxH3ImageFamily(Grid);
        var canvas = new LatentShape(24, 1, 128, 128);              // the 4 MP floor: 2048x2048
        var mean = new float[canvas.Count];
        for (int i = 0; i < mean.Length; i++) mean[i] = MathF.Sin(i * 0.37f) * 3f;
        var referencePosterior = new LatentPosterior(new LatentShape(24, 1, 64, 64), new float[24 * 64 * 64], new float[24 * 64 * 64]);
        var family = new StubbedFamily(inner) { Posteriors = new[] { referencePosterior, new LatentPosterior(canvas, mean, new float[mean.Length]) } };
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCpu },
            new ResidencyPlanner(() => null, _ => { }), _ => { });
        DiffusionResult result = pipeline.Run(new DiffusionRequest
        {
            Prompt = "x", References = new[] { new DiffusionReference(source) }, Mask = new EditMaskOptions { Mask = mask }, Seed = 3,
        });
        Assert.Equal(((2048, 2048), (1024, 1024)), (result.Recipe.Canvas.Sampling, result.Recipe.Canvas.Output));
        Assert.Equal(new[] { (1024, 1024), (2048, 2048) }, family.Encoded);

        float[] expected = inner.References.SourceLatent(new LatentPosterior(canvas, mean, new float[mean.Length])).Data;
        // The pipeline draws the reference's posterior sample first, then the target noise.
        var stream = new TorchCpuRandn(3);
        stream.Next(referencePosterior.Shape.Count);
        float[] noise = stream.Next(canvas.Count);
        LatentTensor decoded = family.Decoded;
        Assert.Equal(canvas, decoded.Shape);
        int keptCells = 0, editedCells = 0;
        for (int y = 0; y < 128; y++)
            for (int x = 0; x < 128; x++)
            {
                // The 8x8 source pixels of the 1024x1024 mask a latent cell of the 2048 canvas covers; any selected one edits it.
                bool edited = x * 8 + 8 > 256 && x * 8 < 513 && y * 8 + 8 > 256 && y * 8 < 512;
                for (int c = 0; c < 24; c++)
                {
                    long at = ((long)c * 128 + y) * 128 + x;
                    if (edited) Assert.Equal(noise[at], decoded.Data[at]);
                    else Assert.Equal(expected[at], decoded.Data[at]);
                }
                if (edited) editedCells++; else keptCells++;
            }
        output.WriteLine($"{keptCells} kept cells on the source, {editedCells} edited cells the sampler's");
        Assert.True(editedCells > 0 && keptCells > 0);
    }

    /// <summary>
    /// A masked edit's canvas (MiniMaxH3ImageFamily.ResolveCanvas): it returns the source's own size whatever it samples;
    /// the whole picture samples as an unmasked edit's canvas would, the selection's crop at that canvas's density, both
    /// with the 4 MP floor at their own aspect; a crop asked for explicitly over 20 MP is refused, a mask with no picture
    /// to edit too.
    /// </summary>
    [Fact]
    public void AMaskedEditSamplesItsSelectionAndReturnsItsSource()
    {
        var family = new MiniMaxH3ImageFamily(Grid);
        DiffusionCanvas Canvas(RgbImage picture, EditMaskOptions o, int w = 0, int h = 0, bool keep = false) =>
            family.ResolveCanvas(new DiffusionRequest
            {
                References = new[] { new DiffusionReference(picture) }, Width = w, Height = h, KeepSourceSize = keep, Mask = o,
            }, EditMask.Create(o, picture));
        RgbImage square = Picture(1024, 1024), photo = Picture(1000, 750), big = Picture(2048, 2048);
        EditMaskOptions Whole(RgbImage p) => new() { Mask = RectangleMask(p.Width, p.Height, 10, 10, 200, 200) };
        EditMaskOptions Cropped(RgbImage p) => new() { Mask = RectangleMask(p.Width, p.Height, 512, 512, 744, 768), MaskCrop = true, MaskCropPadding = 64 };

        Assert.Equal(new DiffusionCanvas((2048, 2048), (1024, 1024)), Canvas(square, Whole(square)));
        Assert.Equal(new DiffusionCanvas((2368, 1760), (1000, 750)), Canvas(photo, Whole(photo), keep: true));
        Assert.Equal(new DiffusionCanvas((2048, 2048), (1024, 1024)), Canvas(square, Whole(square), 2048, 2048));
        // The crop is 384 x 384 source pixels (the rectangle and its gray ramp, 64 of context each side), 384 x 384 at the
        // whole picture's 2048 x 2048 density, then the floor at its square aspect.
        EditMask crop = EditMask.Create(Cropped(big), big);
        Assert.Equal((384, 384), (crop.Width, crop.Height));
        Assert.Equal(new DiffusionCanvas((2048, 2048), (2048, 2048)), Canvas(big, Cropped(big)));
        // A wide crop of a 1000 x 750 photo samples at the floor at its own aspect.
        EditMask wide = EditMask.Create(new EditMaskOptions { Mask = RectangleMask(1000, 750, 100, 300, 900, 420, ramp: 0), MaskCrop = true, MaskCropPadding = 0 }, photo);
        DiffusionCanvas wideCanvas = family.ResolveCanvas(new DiffusionRequest { References = new[] { new DiffusionReference(photo) }, KeepSourceSize = true }, wide);
        Assert.Equal((1000, 750), wideCanvas.Output);
        Assert.Equal(MiniMaxH3ImageFamily.EditCanvas(800, 128), wideCanvas.Sampling);
        // An explicit size is the whole picture's: its crop at 28672 x 28672 would sample 5376 x 5376.
        Assert.Contains("at most 21.0 MP", Assert.Throws<ArgumentException>(() => Canvas(big, Cropped(big), 4096 * 7, 4096 * 7)).Message);
        Assert.Contains("requires a first reference image", Assert.Throws<ArgumentException>(() => family.ResolveCanvas(
            new DiffusionRequest(), EditMask.Create(Whole(square), square))).Message);
    }

    // ---- presentation

    /// <summary>diffusers' presentation: "Picture i: " (no angle brackets), the vision start marker, one pad a merged
    /// patch, the end marker, then the prompt; labels and prompt tagged text, every vision-block row image; the prompt and
    /// each label cut at 4,096 tokens.</summary>
    [Fact]
    public void PicturesArePresentedAsDiffusersEditTextEncoderStep()
    {
        static List<int> Chars(string s) => s.Select(c => (int)c).ToList();
        var (ids, tags, starts) = MiniMaxH3ImageConditioner.Present(Chars, (-1, -2, -3), new[] { 3, 2 }, "Swap <Picture 1>.");
        var want = Chars("Picture 1: ").Concat(new[] { -1, -2, -2, -2, -3 }).Concat(Chars("Picture 2: ")).Concat(new[] { -1, -2, -2, -3 })
            .Concat(Chars("Swap <Picture 1>.")).ToList();
        Assert.Equal(want, ids);
        var wantTags = Enumerable.Repeat(1, 11).Concat(Enumerable.Repeat(0, 5)).Concat(Enumerable.Repeat(1, 11)).Concat(Enumerable.Repeat(0, 4))
            .Concat(Enumerable.Repeat(1, 17)).ToList();
        Assert.Equal(wantTags, tags);
        Assert.Equal(new[] { 12, 28 }, starts);
        var (longIds, _, _) = MiniMaxH3ImageConditioner.Present(Chars, (-1, -2, -3), new[] { 1 }, new string('a', 5000));
        Assert.Equal(11 + 3 + MiniMaxH3ImageConditioner.MaxSequenceLength, longIds.Count);
    }

    // ---- the VAE encoder

    /// <summary>
    /// The VAE encoder program (one program a tile, replayed) on a synthetic encoder against the NumPy oracle
    /// (eng/validation/MiniMaxH3ImageEdit/vae_encoder_oracle.py, float64): a 320x288 picture's four tiles with the moments
    /// stitched against their raw neighbours, and a one-tile 64x48 picture whose latent is 3 cells wide.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void VaeEncoderMatchesTheNumPyOracleOnGgmlCpu() => VaeEncoderMatchesTheOracle(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void VaeEncoderMatchesTheNumPyOracleOnGgmlMetal() => VaeEncoderMatchesTheOracle(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void VaeEncoderMatchesTheNumPyOracleOnGgmlCuda() => VaeEncoderMatchesTheOracle(BackendType.GgmlCuda);

    private void VaeEncoderMatchesTheOracle(BackendType backend)
    {
        GgmlBasicOps.EnsureBackendAvailable(TestGates.GgmlBackendTypeOf(backend));
        using JsonDocument doc = Fixture("vae-encoder-oracle.json");
        using var fx = new MiniMaxH3ImageVaeFixture(blocks: 1, encoder: true);
        using var codec = new MiniMaxH3ImageVaeCodec(fx.VaePath, backend);
        Assert.True(codec.HasEncoder);
        long decoderBytes = codec.WeightBytes;
        foreach (JsonElement c in doc.RootElement.GetProperty("cases").EnumerateArray())
        {
            int w = c.GetProperty("width").GetInt32(), h = c.GetProperty("height").GetInt32();
            int lh = c.GetProperty("latentHeight").GetInt32(), lw = c.GetProperty("latentWidth").GetInt32();
            float[] want = c.GetProperty("moments").EnumerateArray().Select(e => e.GetSingle()).ToArray();
            LatentPosterior p = codec.Encode(MiniMaxH3ImageVaeFixture.OracleImage(w, h), CancellationToken.None);
            Assert.Equal(new LatentShape(24, 1, lh, lw), p.Shape);
            float[] got = p.Mean.Concat(p.LogVariance).ToArray();
            double maxAbs = 0, maxErr = 0;
            for (int i = 0; i < want.Length; i++)
            {
                maxAbs = Math.Max(maxAbs, Math.Abs(want[i]));
                maxErr = Math.Max(maxErr, Math.Abs(want[i] - got[i]));
            }
            output.WriteLine($"{backend} {w}x{h} ({codec.LastEncodeTiles} tiles): moments [48, {lh}, {lw}], max |error| {maxErr:E2} of max |value| {maxAbs:F3}");
            // The GPU backends' F32 products carry 10-bit mantissas: ggml-cuda runs them as TF32 (cuBLAS TF32 tensor-op
            // math, as diffusers' own CUDA convolutions do by default) and ggml-metal's kernel_mul_mm_f32_f32 stages its
            // tiles in half. 6.5e-4 (A40) and 7.4e-4 (M5 Pro) measured; ggml_cpu stays under 1e-6.
            double tolerance = backend == BackendType.GgmlCpu ? 2e-4 : 2e-3;
            Assert.True(maxErr <= tolerance * Math.Max(1, maxAbs), $"{backend} {w}x{h}: max |error| {maxErr:E3} against the NumPy oracle");
        }
        // Four 256-pixel tiles share one program, the 64x48 tile another; the encoder's weights now count.
        Assert.Equal(2, codec.EncoderCompiles);
        Assert.True(codec.WeightBytes > decoderBytes);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void TextEncoderQueryBlocksMatchItsWholeScoresOnGgmlCpu() => TextEncoderQueryBlocksMatchItsWholeScores(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void TextEncoderQueryBlocksMatchItsWholeScoresOnGgmlMetal() => TextEncoderQueryBlocksMatchItsWholeScores(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void TextEncoderQueryBlocksMatchItsWholeScoresOnGgmlCuda() => TextEncoderQueryBlocksMatchItsWholeScores(BackendType.GgmlCuda);

    /// <summary>The trunk's attention a block of query rows at a time (each block against the keys it sees, as an edit's long
    /// sequence runs it) against the whole [seq, seq, heads] scores, on a tiny two-layer GQA encoder: blocks of one row (each
    /// sees only its own prefix), of five (the last one short) and of all but one row.</summary>
    private void TextEncoderQueryBlocksMatchItsWholeScores(BackendType backend)
    {
        GgmlBasicOps.EnsureBackendAvailable(TestGates.GgmlBackendTypeOf(backend));
        string dir = Path.Combine(Path.GetTempPath(), "ts-h3te-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            const int hidden = 64, heads = 4, kvHeads = 2, headDim = 16, ffn = 96, vocab = 40;
            var rng = new Random(77);
            var tensors = new List<MiniMaxH3ImageFixture.T>();
            void Add(string name, float scale, float offset, params long[] shape) =>
                tensors.Add(new MiniMaxH3ImageFixture.T(name, shape, MiniMaxH3ImageFixture.F32,
                    Enumerable.Range(0, checked((int)shape.Aggregate(1L, (a, b) => a * b)))
                        .Select(_ => offset + scale * (float)(rng.NextDouble() * 2 - 1)).ToArray()));
            Add("model.embed_tokens.weight", 1f, 0f, hidden, vocab);
            for (int l = 0; l < 2; l++)
            {
                string p = $"model.layers.{l}.";
                Add(p + "input_layernorm.weight", 0.2f, 1f, hidden);
                Add(p + "post_attention_layernorm.weight", 0.2f, 1f, hidden);
                Add(p + "self_attn.q_norm.weight", 0.2f, 1f, headDim);
                Add(p + "self_attn.k_norm.weight", 0.2f, 1f, headDim);
                Add(p + "self_attn.q_proj.weight", 0.3f, 0f, hidden, heads * headDim);
                Add(p + "self_attn.k_proj.weight", 0.3f, 0f, hidden, kvHeads * headDim);
                Add(p + "self_attn.v_proj.weight", 0.3f, 0f, hidden, kvHeads * headDim);
                Add(p + "self_attn.o_proj.weight", 0.15f, 0f, heads * headDim, hidden);
                Add(p + "mlp.gate_proj.weight", 0.15f, 0f, hidden, ffn);
                Add(p + "mlp.up_proj.weight", 0.15f, 0f, hidden, ffn);
                Add(p + "mlp.down_proj.weight", 0.1f, 0f, ffn, hidden);
            }
            string path = Path.Combine(dir, "te-tiny.gguf");
            MiniMaxH3ImageFixture.Write(path, tensors);
            File.WriteAllText(Path.Combine(dir, "vocab.json"), "{" + string.Join(",", Enumerable.Range(0, vocab).Select(i => $"\"t{i}\":{i}")) + "}");
            File.WriteAllText(Path.Combine(dir, "merges.txt"), "#version: 0.2\n");
            using var te = new MiniMaxH3TextEncoder(path, dir, backend, null);
            int[] ids = Enumerable.Range(0, 23).Select(i => (i * 7 + 3) % vocab).ToArray();
            float[] whole = te.Encode(ids, null);
            double maxAbs = whole.Max(v => Math.Abs((double)v));
            // ggml-cuda's products of the whole [seq, seq, heads] scores run on cuBLAS as TF32 (10-bit mantissas) while a
            // one-row block's take ggml's F32 vector kernel: 1.4e-3 of 4.0 measured on an A40. A block that saw a key it
            // should not, or lost one, moves the output by its whole scale.
            double tolerance = backend == BackendType.GgmlCuda ? 2e-3 : 1e-5;
            var errors = new List<(int Chunk, double Error)>();
            foreach (int chunk in new[] { 1, 5, ids.Length - 1 })
            {
                float[] blocks = te.Encode(ids, null, queryChunk: chunk);
                double maxErr = 0;
                for (int i = 0; i < whole.Length; i++) maxErr = Math.Max(maxErr, Math.Abs((double)whole[i] - blocks[i]));
                output.WriteLine($"{backend} query blocks of {chunk}: max |error| {maxErr:E2} of max |value| {maxAbs:F3}");
                errors.Add((chunk, maxErr));
            }
            foreach (var (chunk, maxErr) in errors)
                Assert.True(maxErr <= tolerance * Math.Max(1, maxAbs), $"{backend} blocks of {chunk}: max |error| {maxErr:E3}");
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>The conditioner's query block for an edit's sequence: the whole scores while they fit the budget, then as many
    /// rows as a [seq, rows, heads] block fits (H3's 64 heads).</summary>
    [Fact]
    public void EditAttentionBlocksFitTheScoreBudget()
    {
        Assert.Equal(0, MiniMaxH3ImageConditioner.QueryChunkFor(1076, 64));     // one picture: the whole scores (296 MB)
        Assert.Equal(1001, MiniMaxH3ImageConditioner.QueryChunkFor(4189, 64));  // four: 1 GiB / (4189 * 64 * 4)
        Assert.Equal(252, MiniMaxH3ImageConditioner.QueryChunkFor(16635, 64));  // sixteen
        Assert.Equal(32, MiniMaxH3ImageConditioner.QueryChunkFor(200_000, 64)); // never below 32 rows
        Assert.Equal(2L * 16635 * 252 * 64 * 4, MiniMaxH3ImageConditioner.ScoreBytesFor(16635, 64));
    }

    /// <summary>The stitched moments against hand-computed blends: the second tile's first overlapping row is the RAW tile
    /// above's (weight 1 - 0), and the overlap's last row three quarters its own.</summary>
    [Fact]
    public void MomentsStitchAgainstRawNeighbours()
    {
        // Two tiles stacked: 256 px tiles over 320 px with a 192-pixel overlap (12 latent rows).
        var (yIdx, yLen, yOverlap) = MiniMaxH3ImageVaeCodec.SplitTiles(320);
        Assert.Equal(new[] { 0, 64 }, yIdx);
        Assert.Equal(new[] { 192 }, yOverlap);
        float[] Tile(float v) => Enumerable.Repeat(v, 48 * 16 * 1).ToArray();
        float[][][] tiles = { new[] { Tile(1f) }, new[] { Tile(3f) } };
        float[] s = MiniMaxH3ImageVaeCodec.StitchMoments(tiles, yLen, new[] { 16 }, yOverlap, Array.Empty<int>(), out int lh, out int lw);
        Assert.Equal((20, 1), (lh, lw));
        // Rows 0-3 are the first tile's kept rows, rows 4-15 the blend (k = 0..11 of 12), rows 16-19 the second tile's.
        Assert.Equal(1f, s[3]);
        Assert.Equal(1f, s[4]);                                // k = 0: weight 0 on the lower tile
        Assert.Equal(1f * (1f - 6f / 12) + 3f * (6f / 12), s[10]);
        Assert.Equal(3f, s[16]);
    }

    /// <summary>The family over test components: the posteriors a VAE would give, a fixed canvas, and a denoiser that records
    /// its first step and predicts zero.</summary>
    private sealed class StubbedFamily(MiniMaxH3ImageFamily inner) : IDiffusionFamily
    {
        public LatentPosterior[] Posteriors { get; init; }
        public DiffusionCanvas? Canvas { get; init; }
        public long DenoiserWeights { get; init; }
        public long BytesPerRow { get; init; }
        public (float[] Latent, float[][] References)? FirstStepValue;
        public (float[] Latent, float[][] References) FirstStep => FirstStepValue ?? default;
        public int Steps;
        public LatentTensor Decoded;
        public List<(int Width, int Height)> Encoded { get; } = new();
        private int _encodes;

        public string Id => inner.Id;
        public string DisplayName => inner.DisplayName;
        public DiffusionGeometry Geometry => inner.Geometry;
        public DiffusionGuidance Guidance => inner.Guidance;
        public SamplingRecipe DefaultRecipe => inner.DefaultRecipe;
        public IReferencePolicy References => inner.References;
        public IConditioner CreateConditioner(DiffusionLoad load) => new Conditioner();
        public IDenoiser CreateDenoiser(DiffusionLoad load) => new Denoiser(this);
        public IVaeCodec CreateVae(DiffusionLoad load) => new Vae(this);
        public INoiseSource Noise(long seed) => inner.Noise(seed);
        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask) => Canvas ?? inner.ResolveCanvas(request, mask);
        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step) =>
            inner.Pack(text, references, target, step);
        public float[] ToRows(LatentTensor latent) => inner.ToRows(latent);
        public LatentTensor FromRows(float[] rows, LatentShape shape) => inner.FromRows(rows, shape);
        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape) => inner.MaskRows(cellWeights, shape);

        private abstract class Component : IResidentComponent
        {
            public virtual long WeightBytes => 0;
            public virtual long WorkingBytes(DiffusionWorkload workload) => 0;
            public void ReleaseDevice() { }
            public void ReleaseWorkspace() { }
            public void Dispose() { }
        }

        private sealed class Conditioner : Component, IConditioner
        {
            public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct)
            {
                int rows = 5 + pictures.Count * 3;
                var tags = new int[rows];
                for (int i = 0; i < rows; i++) tags[i] = i < 5 ? 1 : 0;
                return new ConditioningSet(new float[rows * 8], rows, 8, tags);
            }
        }

        private sealed class Denoiser(StubbedFamily f) : Component, IDenoiser
        {
            public DenoiserCaps Caps { get; } = new();
            public override long WeightBytes => f.DenoiserWeights;
            public override long WorkingBytes(DiffusionWorkload workload) => f.BytesPerRow * MiniMaxH3ImageDenoiser.RowsOf(workload);
            public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct)
            {
                f.FirstStepValue ??= ((float[])inputs.Latent.Clone(), inputs.References.Select(r => (float[])r.Rows.Clone()).ToArray());
                f.Steps++;
                return new StepOutput(new float[inputs.Latent.Length]);
            }
        }

        private sealed class Vae(StubbedFamily f) : Component, IVaeCodec
        {
            public LatentPosterior Encode(RgbImage image, CancellationToken ct)
            {
                f.Encoded.Add((image.Width, image.Height));
                return f.Posteriors?[f._encodes++] ?? new LatentPosterior(new LatentShape(24, 1, image.Height / 16, image.Width / 16),
                    new float[24L * (image.Height / 16) * (image.Width / 16)], new float[24L * (image.Height / 16) * (image.Width / 16)]);
            }
            public RgbImage Decode(LatentTensor latent, CancellationToken ct)
            {
                f.Decoded = latent;
                return new(latent.Shape.Width * 16, latent.Shape.Height * 16, new float[(long)latent.Shape.Width * 16 * latent.Shape.Height * 16 * 3]);
            }
            public RgbImage Preview(LatentTensor latent, int maxSide) => Decode(latent, CancellationToken.None);
        }
    }
}
