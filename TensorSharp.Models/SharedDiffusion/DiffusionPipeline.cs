// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// The generic diffusion runtime: runs a request through a family's components phase by phase
    /// (docs/design/diffusion-runtime.md section 3.4) and owns what no family has to repeat: validation against the
    /// recipe and the adapters' contract, the request's noise stream and its order of draws, residency between phases
    /// and requests (<see cref="ResidencyPlanner"/>), the sampler, guidance, mask reinjection, previews, the final
    /// resize and composite, cancellation and per-phase timings.
    /// <code>
    /// Prepare     validate; resolve the canvas, the references' VAE copies and the mask
    /// Condition   IConditioner.Encode (the vision copies of the references)
    /// EncodeRefs  IVaeCodec.Encode for each reference (posterior samples draw from the stream), and a mask's source
    /// noise       the target noise, then each reference's augmentation, from the same stream
    /// Denoise     for each step: Pack -> IDenoiser.Step (both branches when guided) -> sampler -> mask -> preview
    /// Decode      IVaeCodec.Decode
    /// Composite   resize to the requested size; a masked edit's composite onto its source
    /// </code>
    /// A pipeline serves one loaded model and runs one request at a time (a <see cref="DiffusionJobQueue"/> orders
    /// them). Its components are made on first use (making one maps its weights on the host; nothing reaches the device
    /// until it runs) and stay resident between requests unless the plan releases them.
    /// </summary>
    public sealed class DiffusionPipeline : IDisposable
    {
        /// <summary>The longer side, in pixels, previews are decoded at (about).</summary>
        public const int PreviewMaxSide = 384;

        private readonly IDiffusionFamily _family;
        private readonly DiffusionLoad _load;
        private readonly ResidencyPlanner _planner;
        private readonly Action<string> _log;
        private readonly object _gate = new();
        private readonly HashSet<DiffusionComponent> _resident = new();
        private readonly Dictionary<DiffusionComponent, long> _workspaces = new();
        // The component of the phase running now (null between phases).
        private DiffusionComponent? _running;
        private IConditioner _conditioner;
        private IVaeCodec _vae;
        private IDenoiser _denoiser;
        private bool _disposed;

        /// <param name="planner">The residency planner; null plans against the backend's free device memory less its
        /// headroom (<see cref="GpuMemoryBudget.TryGetReservationSpareBytes(BackendType, out long)"/>: the CUDA and Vulkan
        /// backends' device memory, and Metal's recommended working set, a hard ceiling on unified memory past which command
        /// buffers fail; elsewhere everything stays resident).</param>
        /// <param name="log">Where progress lines go (null: the console).</param>
        public DiffusionPipeline(IDiffusionFamily family, DiffusionLoad load, ResidencyPlanner planner = null, Action<string> log = null)
        {
            _family = family ?? throw new ArgumentNullException(nameof(family));
            _load = load ?? throw new ArgumentNullException(nameof(load));
            _log = log ?? Console.WriteLine;
            family.DefaultRecipe.Validate();
            BackendType backend = load.Backend;
            _planner = planner ?? new ResidencyPlanner(
                () => GpuMemoryBudget.TryGetReservationSpareBytes(backend, out long spare) ? spare : null, _log,
                () => GpuMemoryBudget.TryGetReservationFreeBytes(backend, out long free) ? free : null);
        }

        public IDiffusionFamily Family => _family;

        /// <summary>Whether <paramref name="component"/>'s weights are on the device now (as far as the pipeline knows).</summary>
        public bool IsResident(DiffusionComponent component)
        {
            lock (_gate) return _resident.Contains(component);
        }

        /// <summary>
        /// Run one request. <paramref name="cancellationToken"/> is checked between phases, between steps and (by the
        /// codec) between VAE tile rows; <paramref name="betweenSteps"/> runs before every step after the first (a
        /// queue's checkpoint, which may give the model to waiting compute work). A cancelled or failed run releases
        /// the components' activations and keeps their weights resident, except the weights of the phase that failed
        /// (see <see cref="ReleaseFailedPhase"/>). A request the device cannot hold is refused before any phase runs
        /// (<see cref="RequestRefusedException"/>, <see cref="ResidencyPlan.Refused"/>).
        /// </summary>
        public DiffusionResult Run(DiffusionRequest request, CancellationToken cancellationToken = default, Action betweenSteps = null)
        {
            ArgumentNullException.ThrowIfNull(request);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var total = Stopwatch.StartNew();
                Report(request, new DiffusionProgress(DiffusionPhase.Prepare));
                Prepared job = Prepare(request);
                if (job.Mask?.IsEmpty == true)
                {
                    _log($"{_family.DisplayName}: empty edit mask; returning the original image without inference.");
                    return new DiffusionResult
                    {
                        Image = job.Mask.UnchangedCopy(),
                        Recipe = job.Effective(request),
                        Timings = new DiffusionTimings { Total = total.Elapsed.TotalSeconds },
                    };
                }
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    return Execute(request, job, total, cancellationToken, betweenSteps);
                }
                catch (Exception error)
                {
                    if (error is not OperationCanceledException) ReleaseFailedPhase();
                    ReleaseWorkspaces();
                    throw;
                }
                finally
                {
                    _running = null;
                }
            }
        }

        private DiffusionResult Execute(DiffusionRequest request, Prepared job, Stopwatch total, CancellationToken ct, Action betweenSteps)
        {
            IReferencePolicy policy = _family.References;
            ResidencyPlan plan = PlanResidency(job);
            if (plan.Refused)
                throw new RequestRefusedException(Refusal(job, plan));
            INoiseSource noise = _family.Noise(request.Seed);
            var clock = Stopwatch.StartNew();

            // ---- Condition: text, and the references' vision copies.
            EnterPhase(request, DiffusionPhase.Condition, DiffusionComponent.Conditioner, plan);
            IConditioner conditioner = Conditioner;
            PresentedImage[] pictures = job.Pictures;
            ConditioningSet positive = conditioner.Encode(request.Prompt ?? string.Empty, pictures, ct);
            ConditioningSet negative = job.Guided ? conditioner.Encode(request.NegativePrompt ?? string.Empty, pictures, ct) : null;
            LeavePhase(DiffusionComponent.Conditioner, job.Demand(DiffusionPhase.Condition));
            double conditionSeconds = clock.Elapsed.TotalSeconds, vision = positive.VisionSeconds + (negative?.VisionSeconds ?? 0);
            ct.ThrowIfCancellationRequested();

            // ---- EncodeRefs: each reference's posterior (its sample drawn now, before the target noise), a mask's source.
            clock.Restart();
            var references = new LatentRef[job.Inputs.Length];
            float[] sourceRows = null, maskWeights = null;
            int maskValues = 0;
            if (job.EncodesReferences)
            {
                EnterPhase(request, DiffusionPhase.EncodeReferences, DiffusionComponent.Vae, plan);
                IVaeCodec vae = Vae;
                LatentPosterior first = null;
                for (int i = 0; i < references.Length; i++)
                {
                    ct.ThrowIfCancellationRequested();
                    LatentPosterior posterior = vae.Encode(job.VaeCopies[i], ct);
                    if (i == 0) first = posterior;
                    references[i] = policy.FromPosterior(posterior, i, noise);
                }
                if (job.Mask is { IsFull: false })
                {
                    // The posterior mean at the sampling canvas keeps the blend independent of the stream: the
                    // reference's own encode when its VAE copy is that size, else a separate one.
                    var (w, h) = job.Canvas.Sampling;
                    LatentPosterior source = first != null && job.VaeCopies[0].Width == w && job.VaeCopies[0].Height == h
                        ? first
                        : vae.Encode(ImageIO.Resize(job.Mask.Reference, w, h), ct);
                    if (source.Shape != job.Target)
                        throw new InvalidOperationException($"The mask's source encodes to {source.Shape}, not the target's {job.Target}.");
                    sourceRows = _family.ToRows(new LatentTensor(job.Target, source.Mean));
                    (maskWeights, maskValues) = _family.MaskRows(job.Mask.LatentWeights(job.Target.Width, job.Target.Height), job.Target);
                }
                LeavePhase(DiffusionComponent.Vae, job.Demand(DiffusionPhase.EncodeReferences));
            }
            double encodeSeconds = clock.Elapsed.TotalSeconds;
            ct.ThrowIfCancellationRequested();

            // ---- The target noise, then the references' augmentation, from the same stream.
            float[] eps = _family.ToRows(new LatentTensor(job.Target, noise.Next(job.Target.Count)));
            foreach (LatentRef reference in references) policy.Augment(reference, noise);
            float[] x = (float[])eps.Clone();

            // ---- Denoise.
            clock.Restart();
            EnterPhase(request, DiffusionPhase.Denoise, DiffusionComponent.Denoiser, plan);
            IDenoiser denoiser = Denoiser;
            IDiffusionSampler sampler = job.Recipe.CreateSampler();
            sampler.Reset();
            if (sourceRows != null) EditMask.Reinject(x, sourceRows, eps, maskWeights, maskValues, job.Sigmas[0]);
            int interval = request.PreviewCount > 0 ? Math.Max(1, (job.Steps + request.PreviewCount) / (request.PreviewCount + 1)) : 0;
            var stepSeconds = new double[job.Steps];
            for (int i = 0; i < job.Steps; i++)
            {
                if (i > 0) betweenSteps?.Invoke();
                ct.ThrowIfCancellationRequested();
                var stepClock = Stopwatch.StartNew();
                float sigma = job.Sigmas[i], next = job.Sigmas[i + 1];
                var (headStart, headStop) = job.Recipe.HeadsFor(i, sigma, next);
                var state = new StepState(i, job.Steps, sigma, next, headStart, headStop);
                float[] velocity = Predict(denoiser, x, positive, references, job, state, ct);
                if (negative != null)
                {
                    // A denoiser may hand back its own output buffer, which its next step rewrites.
                    velocity = (float[])velocity.Clone();
                    float[] unconditional = Predict(denoiser, x, negative, references, job, state, ct);
                    for (int k = 0; k < velocity.Length; k++)
                        velocity[k] = unconditional[k] + job.Cfg * (velocity[k] - unconditional[k]);
                }
                sampler.Step(x, velocity, sigma, next);
                if (sourceRows != null) EditMask.Reinject(x, sourceRows, eps, maskWeights, maskValues, next);
                stepSeconds[i] = stepClock.Elapsed.TotalSeconds;
                string heads = job.Recipe.Heads == HeadPolicy.None ? "" : $" (heads {headStart}..{headStop})";
                _log($"  [diffusion] {_family.Id} step {i + 1}/{job.Steps} sigma {sigma:0.#####} -> {next:0.#####}{heads} {stepSeconds[i]:F2}s");
                RgbImage preview = null;
                if (request.Progress != null && interval > 0 && i + 1 < job.Steps && (i + 1) % interval == 0)
                    preview = Preview(x, velocity, next, sourceRows, eps, maskWeights, maskValues, job);
                Report(request, new DiffusionProgress(DiffusionPhase.Denoise, i + 1, job.Steps, preview));
            }
            LeavePhase(DiffusionComponent.Denoiser, job.Demand(DiffusionPhase.Denoise));
            double denoiseSeconds = clock.Elapsed.TotalSeconds;
            ct.ThrowIfCancellationRequested();

            // ---- Decode.
            clock.Restart();
            EnterPhase(request, DiffusionPhase.Decode, DiffusionComponent.Vae, plan);
            RgbImage image = Vae.Decode(_family.FromRows(x, job.Target), ct);
            LeavePhase(DiffusionComponent.Vae, job.Demand(DiffusionPhase.Decode));
            double decodeSeconds = clock.Elapsed.TotalSeconds;

            // ---- Composite: the requested size, a masked edit's source canvas.
            clock.Restart();
            Report(request, new DiffusionProgress(DiffusionPhase.Composite));
            RgbImage output = job.Mask != null ? job.Mask.Composite(image)
                : image.Width == job.Canvas.Output.Width && image.Height == job.Canvas.Output.Height ? image
                : _family.ResizeOutput(image, job.Canvas.Output.Width, job.Canvas.Output.Height);
            double compositeSeconds = clock.Elapsed.TotalSeconds;

            var timings = new DiffusionTimings
            {
                Text = Math.Max(0, conditionSeconds - vision),
                Vision = vision,
                VaeEncode = encodeSeconds,
                Denoise = denoiseSeconds,
                Steps = stepSeconds,
                Decode = decodeSeconds,
                Composite = compositeSeconds,
                Total = total.Elapsed.TotalSeconds,
            };
            _log($"  [diffusion] {_family.Id} {job.Canvas.Sampling.Width}x{job.Canvas.Sampling.Height}, {positive.Rows} conditioning rows, " +
                 $"{references.Length} reference(s): text {timings.Text:F2}s, vision {timings.Vision:F2}s, VAE encode {timings.VaeEncode:F2}s, " +
                 $"denoise {timings.Denoise:F2}s ({timings.Denoise / job.Steps:F2}s/step), decode {timings.Decode:F2}s, total {timings.Total:F2}s");
            return new DiffusionResult { Image = output, Recipe = job.Effective(request), Timings = timings, Residency = plan };
        }

        private float[] Predict(IDenoiser denoiser, float[] x, ConditioningSet conditioning, IReadOnlyList<LatentRef> references,
            Prepared job, StepState state, CancellationToken ct)
        {
            DiffusionLayout layout = _family.Pack(conditioning, references, job.Target, state);
            StepOutput output = denoiser.Step(new DenoiseStepInputs
            {
                Latent = x,
                Conditioning = conditioning,
                References = references,
                Layout = layout,
                Step = state,
                Recipe = job.Recipe,
            }, job.Adapters, ct);
            if (output.Velocity.Length != x.Length)
                throw new InvalidOperationException($"{_family.DisplayName}'s denoiser predicted {output.Velocity.Length} values for {x.Length} latent values.");
            return output.Velocity;
        }

        private RgbImage Preview(float[] x, float[] velocity, float next, float[] sourceRows, float[] eps, float[] maskWeights,
            int maskValues, Prepared job)
        {
            try
            {
                // The state still holds noise at sigma_next; x0 = x - sigma_next * v is the step's clean estimate.
                float[] clean = EulerFlowSampler.Denoised(x, velocity, next);
                if (sourceRows != null) EditMask.Reinject(clean, sourceRows, eps, maskWeights, maskValues, 0f);
                RgbImage preview = Vae.Preview(_family.FromRows(clean, job.Target), PreviewMaxSide);
                _resident.Add(DiffusionComponent.Vae);
                return job.Mask != null ? job.Mask.Composite(preview) : preview;
            }
            catch (Exception error) when (error is not OperationCanceledException)
            {
                _log($"  [diffusion] preview decode skipped: {error.Message}");
                return null;
            }
        }

        // ---- Prepare

        /// <summary>A validated request: what every later phase reads.</summary>
        private sealed class Prepared
        {
            public SamplingRecipe Recipe;
            public int Steps;
            public float[] Sigmas;
            public float Cfg;
            public bool Guided;
            public AdapterSet Adapters;
            public DiffusionCanvas Canvas;
            public LatentShape Target;
            public EditMask Mask;
            public RgbImage[] Inputs;
            public RgbImage[] VaeCopies;
            public PresentedImage[] Pictures = Array.Empty<PresentedImage>();
            public bool EncodesReferences;
            public List<PhaseDemand> Demands;

            public long Demand(DiffusionPhase phase) => Demands.FirstOrDefault(d => d.Phase == phase).WorkingBytes;

            public DiffusionEffectiveRecipe Effective(DiffusionRequest request) => new()
            {
                Steps = Steps,
                Sigmas = Sigmas,
                Cfg = Cfg,
                Seed = request.Seed,
                Canvas = Canvas,
                Recipe = Recipe,
                Adapters = Adapters.Adapters.Select(a => a.Name).ToArray(),
            };
        }

        private Prepared Prepare(DiffusionRequest r)
        {
            string name = _family.DisplayName;
            if (r.Steps < 0 || !float.IsFinite(r.Cfg) || r.Cfg < 0)
                throw new ArgumentException("Steps and CFG must be finite and nonnegative (zero selects the model default).");
            IReadOnlyList<DiffusionReference> references = r.References ?? Array.Empty<DiffusionReference>();
            foreach (DiffusionReference reference in references)
                if (reference?.Image == null) throw new ArgumentException("A reference has no image.");
            DiffusionGeometry geometry = _family.Geometry;
            if (references.Count > geometry.MaxReferences)
                throw new ArgumentException(geometry.MaxReferences == 0
                    ? $"{name} does not edit images; drop the reference images."
                    : $"{name} takes at most {geometry.MaxReferences} reference images; this request has {references.Count}.");

            AdapterSet adapters = r.Adapters ?? AdapterSet.Empty;
            adapters.Contract?.Validate(r.Steps, r.Cfg);
            SamplingRecipe recipe = adapters.Contract?.Apply(_family.DefaultRecipe) ?? _family.DefaultRecipe;
            int steps = recipe.ResolveSteps(r.Steps);
            float cfg = r.Cfg > 0 ? r.Cfg : recipe.Cfg;
            if (_family.Guidance == DiffusionGuidance.None)
            {
                if (cfg > 1f) throw new ArgumentException($"{name} is guidance-distilled: cfg must be 1 (or left unset).");
                if (!string.IsNullOrWhiteSpace(r.NegativePrompt)) throw new ArgumentException($"{name} takes no negative prompt.");
            }
            float[] sigmas = recipe.SigmasFor(steps);

            RgbImage first = references.Count > 0 ? references[0].Image : null;
            EditMask mask = r.Mask == null ? null : EditMask.Create(r.Mask, first);
            var job = new Prepared
            {
                Recipe = recipe, Steps = steps, Sigmas = sigmas, Cfg = cfg,
                Guided = _family.Guidance == DiffusionGuidance.Cfg && cfg > 1f,
                Adapters = adapters, Mask = mask,
                Inputs = references.Select(x => x.Image).ToArray(),
            };
            // The canvas is resolved (and a bad size refused) before an empty selection returns its source unchanged.
            job.Canvas = _family.ResolveCanvas(r, mask);
            if (mask?.IsEmpty == true) return job;
            // A selection's canvas (or crop) is the first picture the encoders see; the others keep their own.
            if (mask != null) job.Inputs[0] = mask.Reference;
            job.Target = geometry.LatentOf(job.Canvas.Sampling.Width, job.Canvas.Sampling.Height);
            // Both copies of every reference are made (and a bad one refused) before any component runs: the vision copy
            // the conditioner reads, and the VAE copy the encoder reads.
            job.VaeCopies = new RgbImage[job.Inputs.Length];
            job.Pictures = new PresentedImage[job.Inputs.Length];
            for (int i = 0; i < job.Inputs.Length; i++)
            {
                job.Pictures[i] = _family.References.ForVision(job.Inputs[i], i);
                job.VaeCopies[i] = _family.References.ForVae(job.Inputs[i], i);
            }
            job.EncodesReferences = job.Inputs.Length > 0 || mask is { IsFull: false };

            var workload = new DiffusionWorkload(job.Target,
                job.VaeCopies.Select(c => new LatentShape(geometry.LatentChannels, 1, c.Height / geometry.VaeFactor, c.Width / geometry.VaeFactor)).ToArray(),
                r.Prompt ?? string.Empty) { Pictures = job.Pictures };
            job.Demands = new List<PhaseDemand> { new(DiffusionPhase.Condition, DiffusionComponent.Conditioner, Conditioner.WorkingBytes(workload, DiffusionPhase.Condition)) };
            if (job.EncodesReferences)
                job.Demands.Add(new(DiffusionPhase.EncodeReferences, DiffusionComponent.Vae, Vae.WorkingBytes(workload, DiffusionPhase.EncodeReferences)));
            job.Demands.Add(new(DiffusionPhase.Denoise, DiffusionComponent.Denoiser, Denoiser.WorkingBytes(workload, DiffusionPhase.Denoise)));
            job.Demands.Add(new(DiffusionPhase.Decode, DiffusionComponent.Vae, Vae.WorkingBytes(workload, DiffusionPhase.Decode)));
            if (job.Inputs.Length > 0) RefuseOverTheRowBudget(job, workload);
            _log($"{name}: {job.Canvas.Sampling.Width}x{job.Canvas.Sampling.Height}" +
                 (job.Canvas.Output != job.Canvas.Sampling ? $" (returned at {job.Canvas.Output.Width}x{job.Canvas.Output.Height})" : "") +
                 $", {recipe.Describe(steps)}, seed {r.Seed}, {job.Inputs.Length} reference(s)" +
                 (mask != null ? ", masked" : "") + (adapters.IsEmpty ? "" : $", adapters {string.Join(", ", adapters.Adapters.Select(a => a.Name))}"));
            return job;
        }

        /// <summary>
        /// An edit whose conditioning or denoising phase alone needs more device memory than the device has (its component's
        /// weights beside the phase's working memory, against the planner's budget) is refused before anything runs, naming
        /// its rows: references add rows to both phases, and fewer or smaller ones are the remedy (decision D11). Text to
        /// image is never refused here; a device with no budget (the CPU backends) refuses nothing.
        /// </summary>
        private void RefuseOverTheRowBudget(Prepared job, DiffusionWorkload workload)
        {
            long held = _resident.Sum(c => Created(c)?.WeightBytes ?? 0) + _workspaces.Values.Sum();
            if (_planner.Budget(held) is not long budget) return;
            DiffusionGeometry g = _family.Geometry;
            long Rows(LatentShape s) => (long)(s.Height / g.Patch) * (s.Width / g.Patch);
            long targetRows = Rows(job.Target), referenceRows = workload.References.Sum(Rows);
            foreach ((DiffusionPhase phase, IResidentComponent component, string what) in new (DiffusionPhase, IResidentComponent, string)[]
                     {
                         (DiffusionPhase.Condition, Conditioner, "the text and vision encoding"),
                         (DiffusionPhase.Denoise, Denoiser, "a denoising step"),
                     })
            {
                long need = component.WeightBytes + job.Demand(phase);
                if (need <= budget) continue;
                throw new ArgumentException(
                    $"{_family.DisplayName}: this edit is too large for the device. Its {job.Inputs.Length} reference image(s) add " +
                    $"{referenceRows:N0} rows to the {targetRows:N0} rows of the {job.Canvas.Sampling.Width}x{job.Canvas.Sampling.Height} " +
                    $"picture (and their vision tokens to the prompt), and {what} would need about {need / 1e9:F1} GB of the " +
                    $"{budget / 1e9:F1} GB the device has. Use fewer or smaller reference images, or a smaller size.");
            }
        }

        // ---- residency

        private ResidencyPlan PlanResidency(Prepared job)
        {
            var weights = new Dictionary<DiffusionComponent, long>
            {
                [DiffusionComponent.Conditioner] = Conditioner.WeightBytes,
                [DiffusionComponent.Vae] = Vae.WeightBytes,
                [DiffusionComponent.Denoiser] = Denoiser.WeightBytes,
            };
            string geometryClass = $"{job.Canvas.Sampling.Width}x{job.Canvas.Sampling.Height}, {job.Inputs.Length} reference(s)";
            return _planner.Plan(job.Demands, weights, _resident.ToArray(), new Dictionary<DiffusionComponent, long>(_workspaces), geometryClass);
        }

        /// <summary>The one-line reason a request the device cannot hold is refused (see <see cref="ResidencyPlan.Refused"/>).</summary>
        private string Refusal(Prepared job, ResidencyPlan plan)
        {
            static string Gb(long bytes) => (bytes / 1e9).ToString("0.0", System.Globalization.CultureInfo.InvariantCulture) + " GB";
            string phase = plan.Limiting.Phase switch
            {
                DiffusionPhase.Condition => "text encoding",
                DiffusionPhase.EncodeReferences => "reference encoding",
                DiffusionPhase.Denoise => "denoise",
                DiffusionPhase.Decode => "VAE decode",
                _ => plan.Limiting.Phase.ToString().ToLowerInvariant(),
            };
            string component = plan.Limiting.Component switch
            {
                DiffusionComponent.Conditioner => "text encoder",
                DiffusionComponent.Denoiser => "denoiser",
                DiffusionComponent.Vae => "VAE",
                _ => plan.Limiting.Component.ToString().ToLowerInvariant(),
            };
            var (w, h) = job.Canvas.Sampling;
            return $"{_family.DisplayName} cannot run {w}x{h} on this device: its {phase} alone needs {Gb(plan.MinimumPeak)} " +
                   $"(the {component}'s {Gb(plan.LimitingWeights)} of weights and {Gb(plan.Limiting.WorkingBytes)} of working memory), " +
                   $"more than the {Gb(plan.Capacity ?? 0)} of device memory free for it. Use a smaller size or a smaller " +
                   "quantization, or free the device.";
        }

        /// <summary>Make the plan's releases before <paramref name="phase"/>; <paramref name="component"/> is resident from
        /// here on (its first use uploads it, whether or not the phase finishes).</summary>
        private void EnterPhase(DiffusionRequest request, DiffusionPhase phase, DiffusionComponent component, ResidencyPlan plan)
        {
            foreach (ResidencyRelease release in plan.Before(phase))
            {
                IResidentComponent released = Created(release.Component);
                if (released == null) continue;
                // A weights release drops the workspace too: its cached plans bind the weights.
                released.ReleaseWorkspace();
                _workspaces.Remove(release.Component);
                if (!release.Weights) continue;
                released.ReleaseDevice();
                _resident.Remove(release.Component);
            }
            _resident.Add(component);
            _running = component;
            Report(request, new DiffusionProgress(phase));
        }

        /// <summary>
        /// A phase failed (a device allocation that failed while its component's weights were being bound or uploaded, a
        /// backend error): which of that component's device copies exist is no longer known, so the plan must not count its
        /// weights as held, or the next request is planned against memory it does not have and fails as well (measured on
        /// the RTX A4500: after the text encoder's upload failed for another process's allocation, the next request kept
        /// "everything resident" against a budget inflated by its 15.5 GB). Its copies are released (the weights stay
        /// mapped; the next request uploads them again) and it is no longer resident.
        /// </summary>
        private void ReleaseFailedPhase()
        {
            if (_running is not DiffusionComponent failed) return;
            _running = null;
            try { Created(failed)?.ReleaseDevice(); }
            catch (Exception error) { _log($"  [diffusion] releasing the {failed.ToString().ToLowerInvariant()} after its phase failed: {error.Message}"); }
            _resident.Remove(failed);
            _workspaces.Remove(failed);
        }

        /// <summary>The phase finished: its component keeps its workspace (cached plans) for the next request.</summary>
        private void LeavePhase(DiffusionComponent component, long workingBytes)
        {
            _workspaces[component] = Math.Max(_workspaces.GetValueOrDefault(component), workingBytes);
            _running = null;
        }

        private void ReleaseWorkspaces()
        {
            foreach (DiffusionComponent c in Enum.GetValues<DiffusionComponent>())
            {
                try { Created(c)?.ReleaseWorkspace(); }
                catch (Exception error) { _log($"  [diffusion] releasing the {c.ToString().ToLowerInvariant()}'s workspace failed: {error.Message}"); }
            }
            _workspaces.Clear();
        }

        private static void Report(DiffusionRequest request, DiffusionProgress progress) => request.Progress?.Invoke(progress);

        // ---- components

        private IConditioner Conditioner => _conditioner ??= _family.CreateConditioner(_load)
            ?? throw new InvalidOperationException($"{_family.DisplayName} made no conditioner.");
        private IVaeCodec Vae => _vae ??= _family.CreateVae(_load)
            ?? throw new InvalidOperationException($"{_family.DisplayName} made no VAE.");
        private IDenoiser Denoiser => _denoiser ??= _family.CreateDenoiser(_load)
            ?? throw new InvalidOperationException($"{_family.DisplayName} made no denoiser.");

        private IResidentComponent Created(DiffusionComponent component) => component switch
        {
            DiffusionComponent.Conditioner => _conditioner,
            DiffusionComponent.Vae => _vae,
            DiffusionComponent.Denoiser => _denoiser,
            _ => null,
        };

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _denoiser?.Dispose(); _denoiser = null;
                _vae?.Dispose(); _vae = null;
                _conditioner?.Dispose(); _conditioner = null;
                _resident.Clear();
                _workspaces.Clear();
            }
        }
    }
}
