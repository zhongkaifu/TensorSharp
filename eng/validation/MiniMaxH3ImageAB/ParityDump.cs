// The edit parity dump of stage S5 (docs/design/diffusion-runtime.md section 7.3): one MiniMax-H3-Image edit through the
// production pipeline (DiffusionPipeline over MiniMaxH3ImageFamily, its real components and residency plan), with each
// component recorded on the way, up to the first denoising step:
//
//   request.json               the request, the canvas, the shapes, the presentation's token ids and tags, the step
//   vae-copy-<k>.png           reference k's VAE copy (bytes), vision-copy-<k>.png its vision copy
//   moments-<k>.f32            the VAE encoder's posterior of reference k: mean then log-variance, [24][h][w] each
//   cond-hidden.f32            the conditioning, [rows][5120]
//   ref-rows-<k>.f32           reference k's augmented rows as the first step reads them, [rows][96]
//   latent.f32                 the target rows the first step reads (the target's noise), [rows][96]
//   positions.f32              the packed sequence's (t, h, w) positions, [rows][3]
//   velocity.f32               the first step's velocity of the target rows, [rows][96]
//
// eng/validation/MiniMaxH3ImageEdit/diffusers_parity.py compares them with diffusers d5df05ed.
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

internal static class ParityDump
{
    private sealed class StopAfterFirstStep : Exception;

    public static JsonObject Run(BackendType backend, string model, string te, string vae, JsonNode job, string baseDir, int size, string dir)
    {
        Directory.CreateDirectory(dir);
        var inner = MiniMaxH3ImageFamily.ForDenoiser(model);
        var family = new Recording(inner, dir);
        var load = new DiffusionLoad
        {
            Backend = backend,
            Files = new Dictionary<string, string>
            {
                [DiffusionLoad.Denoiser] = model, [DiffusionLoad.TextEncoder] = te, [DiffusionLoad.Vae] = vae,
            },
        };
        string[] paths = ((JsonArray)job["images"]!).Select(x => Path.Combine(baseDir, (string)x!)).ToArray();
        var request = new DiffusionRequest
        {
            Prompt = (string)job["prompt"]!,
            References = paths.Select(p => new DiffusionReference(ImageIO.Load(p, preserveAlpha: true))).ToArray(),
            Width = size > 0 ? size : (int?)job["width"] ?? 0,
            Height = size > 0 ? size : (int?)job["height"] ?? 0,
            Seed = (long)job["seed"]!,
        };
        using var pipeline = new DiffusionPipeline(family, load);
        var clock = System.Diagnostics.Stopwatch.StartNew();
        try { pipeline.Run(request); }
        catch (StopAfterFirstStep) { }
        family.Info["images"] = new JsonArray(paths.Select(p => (JsonNode)p).ToArray());
        family.Info["prompt"] = request.Prompt;
        family.Info["seed"] = request.Seed;
        family.Info["width"] = request.Width;
        family.Info["height"] = request.Height;
        family.Info["seconds"] = clock.Elapsed.TotalSeconds;
        File.WriteAllText(Path.Combine(dir, "request.json"), family.Info.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"[h3img-ab] parity dump: {dir}");
        return family.Info;
    }

    private static void Write(string path, float[] values)
    {
        var bytes = new byte[values.LongLength * 4];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        File.WriteAllBytes(path, bytes);
    }

    private sealed class Recording(MiniMaxH3ImageFamily inner, string dir) : IDiffusionFamily
    {
        public readonly JsonObject Info = new();
        private int _encodes;

        public string Id => inner.Id;
        public string DisplayName => inner.DisplayName;
        public DiffusionGeometry Geometry => inner.Geometry;
        public DiffusionGuidance Guidance => inner.Guidance;
        public SamplingRecipe DefaultRecipe => inner.DefaultRecipe;
        public IReferencePolicy References => new Policy(inner.References, dir);
        public INoiseSource Noise(long seed) => inner.Noise(seed);
        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask)
        {
            DiffusionCanvas c = inner.ResolveCanvas(request, mask);
            Info["sampling"] = $"{c.Sampling.Width}x{c.Sampling.Height}";
            Info["output"] = $"{c.Output.Width}x{c.Output.Height}";
            return c;
        }
        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step) =>
            inner.Pack(text, references, target, step);
        public float[] ToRows(LatentTensor latent) => inner.ToRows(latent);
        public LatentTensor FromRows(float[] rows, LatentShape shape) => inner.FromRows(rows, shape);
        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape) => inner.MaskRows(cellWeights, shape);
        public RgbImage ResizeOutput(RgbImage image, int width, int height) => inner.ResizeOutput(image, width, height);

        public IConditioner CreateConditioner(DiffusionLoad load) => new Conditioner((MiniMaxH3ImageConditioner)inner.CreateConditioner(load), this);
        public IVaeCodec CreateVae(DiffusionLoad load) => new Vae((MiniMaxH3ImageVaeCodec)inner.CreateVae(load), this);
        public IDenoiser CreateDenoiser(DiffusionLoad load) => new Denoiser(inner.CreateDenoiser(load), this);

        private sealed class Policy(IReferencePolicy p, string dir) : IReferencePolicy
        {
            public PresentedImage ForVision(RgbImage image, int index)
            {
                PresentedImage v = p.ForVision(image, index);
                File.WriteAllBytes(Path.Combine(dir, $"vision-copy-{index}.png"), ImageIO.EncodePng(v.Image));
                return v;
            }
            public RgbImage ForVae(RgbImage image, int index)
            {
                RgbImage v = p.ForVae(image, index);
                File.WriteAllBytes(Path.Combine(dir, $"vae-copy-{index}.png"), ImageIO.EncodePng(v));
                return v;
            }
            public LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise) => p.FromPosterior(posterior, index, noise);
            public void Augment(LatentRef reference, INoiseSource noise) => p.Augment(reference, noise);
        }

        private sealed class Conditioner(MiniMaxH3ImageConditioner c, Recording f) : IConditioner
        {
            public long WeightBytes => c.WeightBytes;
            public long WorkingBytes(DiffusionWorkload workload) => c.WorkingBytes(workload);
            public void ReleaseDevice() => c.ReleaseDevice();
            public void ReleaseWorkspace() => c.ReleaseWorkspace();
            public void Dispose() => c.Dispose();
            public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct)
            {
                ConditioningSet set = c.Encode(prompt, pictures, ct);
                Write(Path.Combine(dir, "cond-hidden.f32"), set.Hidden);
                var (ids, tags) = c.LastPresentation;
                f.Info["cond_rows"] = set.Rows;
                f.Info["cond_ids"] = new JsonArray((ids ?? Array.Empty<int>()).Select(i => (JsonNode)i).ToArray());
                f.Info["cond_tags"] = new JsonArray((set.Modalities ?? Array.Empty<int>()).Select(i => (JsonNode)i).ToArray());
                f.Info["vision_seconds"] = set.VisionSeconds;
                return set;
            }
            private string dir => f._dir;
        }

        private string _dir => dir;

        private sealed class Vae(MiniMaxH3ImageVaeCodec v, Recording f) : IVaeCodec
        {
            public long WeightBytes => v.WeightBytes;
            public long WorkingBytes(DiffusionWorkload workload) => v.WorkingBytes(workload);
            public long WorkingBytes(DiffusionWorkload workload, TensorSharp.Runtime.Scheduling.Diffusion.DiffusionPhase phase) => v.WorkingBytes(workload, phase);
            public void ReleaseDevice() => v.ReleaseDevice();
            public void ReleaseWorkspace() => v.ReleaseWorkspace();
            public void Dispose() => v.Dispose();
            public LatentPosterior Encode(RgbImage image, CancellationToken ct)
            {
                LatentPosterior p = v.Encode(image, ct);
                int k = f._encodes++;
                Write(Path.Combine(f._dir, $"moments-{k}.f32"), p.Mean.Concat(p.LogVariance).ToArray());
                f.Info[$"moments_{k}_shape"] = new JsonArray(p.Shape.Height, p.Shape.Width);
                f.Info[$"vae_encode_{k}_tiles"] = v.LastEncodeTiles;
                return p;
            }
            public RgbImage Decode(LatentTensor latent, CancellationToken ct) => v.Decode(latent, ct);
            public RgbImage Preview(LatentTensor latent, int maxSide) => v.Preview(latent, maxSide);
        }

        private sealed class Denoiser(IDenoiser d, Recording f) : IDenoiser
        {
            public DenoiserCaps Caps => d.Caps;
            public long WeightBytes => d.WeightBytes;
            public long WorkingBytes(DiffusionWorkload workload) => d.WorkingBytes(workload);
            public void ReleaseDevice() => d.ReleaseDevice();
            public void ReleaseWorkspace() => d.ReleaseWorkspace();
            public void Dispose() => d.Dispose();
            public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct)
            {
                for (int k = 0; k < inputs.References.Count; k++)
                {
                    Write(Path.Combine(f._dir, $"ref-rows-{k}.f32"), inputs.References[k].Rows);
                    f.Info[$"ref_{k}_latent"] = new JsonArray(inputs.References[k].Shape.Height, inputs.References[k].Shape.Width);
                }
                Write(Path.Combine(f._dir, "latent.f32"), inputs.Latent);
                Write(Path.Combine(f._dir, "positions.f32"), inputs.Layout.Positions);
                f.Info["layout_rows"] = inputs.Layout.Rows;
                f.Info["target_start"] = inputs.Layout.TargetStart;
                f.Info["timesteps"] = new JsonArray(inputs.Layout.Timesteps.Select(t => (JsonNode)t).ToArray());
                f.Info["step"] = new JsonObject
                {
                    ["sigma"] = inputs.Step.Sigma, ["next"] = inputs.Step.Next,
                    ["head_start"] = inputs.Step.HeadStart, ["head_stop"] = inputs.Step.HeadStop,
                };
                var clock = System.Diagnostics.Stopwatch.StartNew();
                StepOutput o = d.Step(inputs, adapters, ct);
                f.Info["step_seconds"] = clock.Elapsed.TotalSeconds;
                Write(Path.Combine(f._dir, "velocity.f32"), o.Velocity);
                throw new StopAfterFirstStep();
            }
        }
    }
}
