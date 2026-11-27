// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion runtime's video pipeline (docs/design/diffusion-runtime.md section 3.9): a clip, and the audio a joint
// audio-video family denoises beside it, through a family's components phase by phase. It is the image pipeline's sibling
// and shares its pieces: the components' residency contract (IResidentComponent) and the ResidencyPlanner, IConditioner and
// IDenoiser (whose StepOutput carries the further named streams), SamplingRecipe and AdapterSet, cancellation and the
// per-phase timings. What a video family differs in (what a request conditions on, which checkpoint of the family denoises
// it, how its streams are packed and converted for the sampler, what its decoders give back) is the family's
// (IVideoDiffusionFamily).
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
    /// <summary>One clip request: what every video family reads, and the family's own request (<see cref="FamilyRequest"/>:
    /// its keyframes, references, sizes and switches), which the runtime hands back to the family untouched.</summary>
    public sealed class DiffusionVideoRequest
    {
        public string Prompt { get; init; } = string.Empty;
        public long Seed { get; init; }

        /// <summary>The adapters bound to the denoiser for this clip (a step-distillation adapter's contract fixes the
        /// sampling; the family checks the request against it).</summary>
        public AdapterSet Adapters { get => _adapters; init => _adapters = value ?? AdapterSet.Empty; }
        private AdapterSet _adapters = AdapterSet.Empty;

        /// <summary>Decode the joint audio stream (a family with one); false skips its decoder.</summary>
        public bool DecodeAudio { get; init; } = true;

        /// <summary>The family's own description of the request.</summary>
        public object FamilyRequest { get; init; }

        /// <summary>Receives each phase and stage entered and each finished step, synchronously, in order.</summary>
        public Action<DiffusionVideoProgress> Progress { get; init; }
    }

    /// <summary>
    /// One progress report of a clip: the phase entered (with <see cref="Stage"/>, the part of it that runs, and
    /// <see cref="Attributes"/>, facts about it for telemetry), or a finished denoising step (<see cref="Step"/> of
    /// <see cref="Total"/>). <see cref="Detail"/> is a human-readable note (a VAE tile).
    /// </summary>
    public readonly record struct DiffusionVideoProgress(DiffusionPhase Phase, int Step = 0, int Total = 0, string Stage = null,
        string Detail = null, IReadOnlyDictionary<string, object> Attributes = null)
    {
        /// <summary>A finished denoising step.</summary>
        public bool IsStep => Phase == DiffusionPhase.Denoise && Step > 0;
    }

    /// <summary>
    /// A validated clip, as a family resolved it (<see cref="IVideoDiffusionFamily.Prepare"/>): the sampling it runs, which
    /// of the family's checkpoints denoises it, what the phases will need on the device, and the streams the sampler steps.
    /// A family keeps its own state (the conditioning frames it encodes, the layout's inputs) in a subclass.
    /// </summary>
    public class DiffusionVideoJob
    {
        /// <summary>The recipe the clip samples with (the family's, or the speed adapter's contract applied to it).</summary>
        public required SamplingRecipe Recipe { get; init; }
        public required int Steps { get; init; }

        /// <summary>The primary stream's sigmas, <see cref="Steps"/> + 1 ending in 0; every stream is stepped on them (a family
        /// whose further streams follow their own schedule converts them for the sampler: <see cref="IVideoDiffusionFamily.SamplerVelocities"/>).
        /// A clip sampled in passes (<see cref="Stages"/>) gives its first pass's here.</summary>
        public required float[] Sigmas { get; init; }

        /// <summary>The clip's sampling passes in order (LTX 2.5's half-size first pass, then its refinement), or null for one
        /// pass over <see cref="Sigmas"/>. <see cref="Steps"/> is their total; between two passes the family makes the next
        /// pass's state (<see cref="IVideoDiffusionFamily.NextStage"/>) while the denoiser stays resident.</summary>
        public IReadOnlyList<DiffusionVideoStage> Stages { get; init; }

        /// <summary>The passes the pipeline runs: <see cref="Stages"/>, or one over <see cref="Sigmas"/>.</summary>
        public IReadOnlyList<DiffusionVideoStage> Passes => Stages ?? new[] { new DiffusionVideoStage(null, Sigmas) };

        /// <summary>The denoiser checkpoint the clip runs on (one of the family's files); another than the loaded one's
        /// replaces the loaded denoiser.</summary>
        public required string DenoiserFile { get; init; }

        /// <summary>The phases the clip runs, in order, with what each needs beside its component's weights: the residency
        /// plan's input.</summary>
        public required IReadOnlyList<PhaseDemand> Demands { get; init; }

        /// <summary>The clip's geometry class in the residency log ("640x384x22").</summary>
        public required string GeometryClass { get; init; }

        /// <summary>One line for the log.</summary>
        public string Describe { get; init; } = "";

        public bool EncodesVideo => Demands.Any(d => d.Phase == DiffusionPhase.EncodeReferences && d.Component == DiffusionComponent.Vae);
        public bool EncodesAudio => Demands.Any(d => d.Phase == DiffusionPhase.EncodeAudio);
        public bool DecodesAudio => Demands.Any(d => d.Phase == DiffusionPhase.DecodeAudio);
    }

    /// <summary>The sampler's state: one array a stream, by name (<see cref="Primary"/> is the stream the denoiser's
    /// <see cref="DenoiseStepInputs.Latent"/> holds).</summary>
    public sealed class DiffusionStreams
    {
        public const string Primary = "video";

        public DiffusionStreams(IReadOnlyDictionary<string, float[]> streams)
        {
            ArgumentNullException.ThrowIfNull(streams);
            if (!streams.ContainsKey(Primary)) throw new ArgumentException($"The streams have no '{Primary}' stream.", nameof(streams));
            Values = streams;
        }

        public IReadOnlyDictionary<string, float[]> Values { get; }
        public float[] this[string name] => Values[name];
    }

    /// <summary>A finished clip: its frames and audio (null without), what ran, the phase timings and the residency plan.</summary>
    public sealed class DiffusionVideoResult
    {
        public required RgbImage[] Frames { get; init; }

        /// <summary>The decoded audio's channel planes, or null.</summary>
        public float[][] Audio { get; init; }
        public int AudioSampleRate { get; init; }
        public required DiffusionVideoJob Job { get; init; }
        public required DiffusionVideoTimings Timings { get; init; }
        public ResidencyPlan Residency { get; init; }
    }

    /// <summary>Seconds each phase of a clip took.</summary>
    public sealed record DiffusionVideoTimings
    {
        public double Condition { get; init; }
        public double Encode { get; init; }
        public double DenoiserLoad { get; init; }
        public double Denoise { get; init; }
        public IReadOnlyList<double> Steps { get; init; } = Array.Empty<double>();
        public double Decode { get; init; }
        public double DecodeAudio { get; init; }
        public double Total { get; init; }
    }

    /// <summary>The components a video family's phases run on; a family's hooks cast them to its own types.</summary>
    public sealed class DiffusionVideoComponents
    {
        internal DiffusionVideoComponents(DiffusionVideoPipeline pipeline) => _pipeline = pipeline;
        private readonly DiffusionVideoPipeline _pipeline;

        public IConditioner Conditioner => _pipeline.ConditionerComponent;
        public IResidentComponent VideoCodec => _pipeline.VideoCodecComponent;

        /// <summary>The audio codec, or null when the family's load has none.</summary>
        public IResidentComponent AudioCodec => _pipeline.AudioCodecComponent;
    }

    /// <summary>
    /// A video family on the shared runtime: its components and the hooks <see cref="DiffusionVideoPipeline"/> runs a clip
    /// through. The runtime owns the phases, residency, the sampler, cancellation, progress and timings.
    /// </summary>
    public interface IVideoDiffusionFamily
    {
        string Id { get; }

        /// <summary>The name refusals use ("MiniMax-H3").</summary>
        string DisplayName { get; }

        IConditioner CreateConditioner(DiffusionLoad load);

        /// <summary>The denoiser of checkpoint <paramref name="file"/> (one of the load's denoiser files).</summary>
        IDenoiser CreateDenoiser(DiffusionLoad load, string file);

        IResidentComponent CreateVideoCodec(DiffusionLoad load);

        /// <summary>The audio codec, or null when the load has none (a silent clip).</summary>
        IResidentComponent CreateAudioCodec(DiffusionLoad load);

        /// <summary>Validate <paramref name="request"/> and resolve what it runs; nothing reaches a device. The components are
        /// made (their weights mapped) so the family can size the phases' working memory.</summary>
        DiffusionVideoJob Prepare(DiffusionVideoRequest request, DiffusionVideoComponents components);

        /// <summary>The text (and vision) conditioning of the clip.</summary>
        ConditioningSet Condition(DiffusionVideoJob job, IConditioner conditioner, CancellationToken cancellationToken);

        /// <summary>Encode the clip's visual conditioning (keyframes, reference pictures and clips) with the video codec.</summary>
        void EncodeVideo(DiffusionVideoJob job, IResidentComponent videoCodec, Action<DiffusionVideoProgress> report, CancellationToken cancellationToken);

        /// <summary>Encode the clip's sound conditioning with the audio codec.</summary>
        void EncodeAudio(DiffusionVideoJob job, IResidentComponent audioCodec, CancellationToken cancellationToken);

        /// <summary>The sampler's state at the first sigma (the clip's noise).</summary>
        DiffusionStreams InitialStreams(DiffusionVideoJob job);

        /// <summary>The denoiser's inputs for <paramref name="step"/> from the sampler's state.</summary>
        DenoiseStepInputs StepInputs(DiffusionVideoJob job, ConditioningSet conditioning, DiffusionStreams state, StepState step);

        /// <summary>The velocity each stream is stepped with, from the denoiser's prediction (a family whose further streams
        /// follow their own schedule converts them to the primary schedule's sampler here). The arrays may be updated in place.</summary>
        DiffusionStreams SamplerVelocities(DiffusionVideoJob job, DiffusionStreams state, DenoiseStepInputs inputs, StepOutput output, StepState step);

        /// <summary>The clip's frames from the sampler's final state.</summary>
        RgbImage[] DecodeVideo(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent videoCodec,
            Action<DiffusionVideoProgress> report, CancellationToken cancellationToken);

        /// <summary>The clip's audio (channel planes and their rate) from the sampler's final state.</summary>
        (float[][] Channels, int SampleRate) DecodeAudio(DiffusionVideoJob job, DiffusionStreams state, IResidentComponent audioCodec,
            CancellationToken cancellationToken);

        /// <summary>The state pass <paramref name="stage"/> (1 or later of <see cref="DiffusionVideoJob.Stages"/>) starts from, made
        /// from the previous pass's final state <paramref name="previous"/> (LTX 2.5: the latent upscaled and mixed with fresh
        /// noise, its guides encoded again at the pass's size). It runs inside the denoise phase, the denoiser resident; the
        /// video codec may run (the pipeline then counts it resident). A one-pass family never sees it.</summary>
        DiffusionStreams NextStage(DiffusionVideoJob job, int stage, DiffusionStreams previous, DiffusionVideoComponents components,
            Action<DiffusionVideoProgress> report, CancellationToken cancellationToken) =>
            throw new NotSupportedException($"{DisplayName} samples in one pass.");

        /// <summary>What a conditioned sampler (<see cref="IConditionedSampler"/>: ancestral Euler, a guide's denoise mask) reads at
        /// <paramref name="step"/> beside each stream's state and velocity, or null for nothing. Called once a step, after the
        /// denoiser's prediction and before any stream is stepped.</summary>
        DiffusionSamplerInputs SamplerInputs(DiffusionVideoJob job, DiffusionStreams state, StepState step) => null;
    }

    /// <summary>
    /// The diffusion runtime's video pipeline: one loaded video family, one clip at a time.
    /// <code>
    /// Prepare      the family validates the clip and resolves its sampling, checkpoint and phases
    /// Condition    the conditioner (text, and the vision tower for pictures)
    /// EncodeRefs   the video codec encodes the visual conditioning (keyframes, references)
    /// EncodeAudio  the audio codec encodes sound conditioning
    /// Denoise      for each step: the family's inputs -> IDenoiser.Step -> the family's sampler velocities -> Euler per stream
    /// Decode       the video codec decodes the frames
    /// DecodeAudio  the audio codec decodes the joint audio stream
    /// </code>
    /// Components are made on first use (their weights mapped on the host) and stay resident between clips unless the
    /// residency plan, or the component itself (<see cref="IResidentComponent.ReleaseAfterPhase"/>), releases them.
    /// </summary>
    public sealed class DiffusionVideoPipeline : IDisposable
    {
        private readonly IVideoDiffusionFamily _family;
        private readonly DiffusionLoad _load;
        private readonly ResidencyPlanner _planner;
        private readonly Action<string> _log;
        private readonly object _gate = new();
        private readonly HashSet<DiffusionComponent> _resident = new();
        private readonly Dictionary<DiffusionComponent, long> _workspaces = new();
        private DiffusionComponent? _running;
        private IConditioner _conditioner;
        private IResidentComponent _videoCodec, _audioCodec;
        private bool _audioCodecMade;
        private IDenoiser _denoiser;
        private string _denoiserFile;
        private int _runSteps;   // the running clip's steps (every report of it carries them)
        private bool _disposed;

        /// <param name="planner">The residency planner; null plans against the backend's free device memory less its headroom
        /// (as <see cref="DiffusionPipeline"/> does).</param>
        /// <param name="log">Where progress lines go (null: the console).</param>
        public DiffusionVideoPipeline(IVideoDiffusionFamily family, DiffusionLoad load, ResidencyPlanner planner = null, Action<string> log = null)
        {
            _family = family ?? throw new ArgumentNullException(nameof(family));
            _load = load ?? throw new ArgumentNullException(nameof(load));
            _log = log ?? Console.WriteLine;
            BackendType backend = load.Backend;
            _planner = planner ?? new ResidencyPlanner(
                () => GpuMemoryBudget.TryGetReservationSpareBytes(backend, out long spare) ? spare : null, _log,
                () => GpuMemoryBudget.TryGetReservationFreeBytes(backend, out long free) ? free : null);
        }

        public IVideoDiffusionFamily Family => _family;

        /// <summary>The loaded denoiser's checkpoint, or null before the first clip.</summary>
        public string DenoiserFile { get { lock (_gate) return _denoiserFile; } }

        /// <summary>Whether <paramref name="component"/>'s weights are on the device now (as far as the pipeline knows).</summary>
        public bool IsResident(DiffusionComponent component)
        {
            lock (_gate) return _resident.Contains(component);
        }

        /// <summary>The loaded denoiser (null before the first clip): a family's adapter changes reach it here.</summary>
        public IDenoiser LoadedDenoiser { get { lock (_gate) return _denoiser; } }

        /// <summary>
        /// Run one clip. <paramref name="cancellationToken"/> is checked between phases and between steps. A cancelled or
        /// failed clip releases the components' workspaces and keeps their weights resident, except the weights of the phase
        /// that failed. A clip the device cannot hold is refused before any phase runs (<see cref="RequestRefusedException"/>).
        /// </summary>
        public DiffusionVideoResult Run(DiffusionVideoRequest request, CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(request);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var total = Stopwatch.StartNew();
                Report(request, new DiffusionVideoProgress(DiffusionPhase.Prepare));
                DiffusionVideoJob job = _family.Prepare(request, new DiffusionVideoComponents(this));
                _runSteps = job.Steps;
                cancellationToken.ThrowIfCancellationRequested();
                try
                {
                    return Execute(request, job, total, cancellationToken);
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
                    _runSteps = 0;
                }
            }
        }

        private DiffusionVideoResult Execute(DiffusionVideoRequest request, DiffusionVideoJob job, Stopwatch total, CancellationToken ct)
        {
            ResidencyPlan plan = PlanResidency(job);
            if (plan.Refused)
                throw new RequestRefusedException(
                    $"{_family.DisplayName} cannot run {job.GeometryClass} on this device: its {plan.Limiting.Phase} alone needs about " +
                    $"{plan.MinimumPeak / 1e9:0.0} GB, more than the {(plan.Capacity ?? 0) / 1e9:0.0} GB of device memory free for it.");
            var clock = Stopwatch.StartNew();
            Action<DiffusionVideoProgress> report = p => Report(request, p);

            // ---- Condition
            EnterPhase(request, DiffusionPhase.Condition, DiffusionComponent.Conditioner, plan);
            ConditioningSet conditioning = _family.Condition(job, Conditioner, ct);
            LeavePhase(request, DiffusionComponent.Conditioner, Demand(job, DiffusionPhase.Condition));
            double conditionSeconds = clock.Elapsed.TotalSeconds;
            ct.ThrowIfCancellationRequested();

            // ---- The denoiser of the clip's checkpoint (made, or replaced, before the encodes: its file is read while they run)
            clock.Restart();
            IDenoiser denoiser = DenoiserFor(request, job);
            double denoiserLoadSeconds = clock.Elapsed.TotalSeconds;

            // ---- EncodeRefs, EncodeAudio
            clock.Restart();
            if (job.EncodesVideo)
            {
                EnterPhase(request, DiffusionPhase.EncodeReferences, DiffusionComponent.Vae, plan);
                _family.EncodeVideo(job, VideoCodec, report, ct);
                LeavePhase(request, DiffusionComponent.Vae, Demand(job, DiffusionPhase.EncodeReferences));
                ct.ThrowIfCancellationRequested();
            }
            if (job.EncodesAudio)
            {
                IResidentComponent audioEncoder = AudioCodec ?? throw new InvalidOperationException($"{_family.DisplayName}'s load has no audio codec.");
                EnterPhase(request, DiffusionPhase.EncodeAudio, DiffusionComponent.AudioVae, plan);
                _family.EncodeAudio(job, audioEncoder, ct);
                LeavePhase(request, DiffusionComponent.AudioVae, Demand(job, DiffusionPhase.EncodeAudio));
                ct.ThrowIfCancellationRequested();
            }
            double encodeSeconds = clock.Elapsed.TotalSeconds;

            // ---- Denoise (each pass in turn; between two, the family makes the next pass's state)
            clock.Restart();
            IReadOnlyList<DiffusionVideoStage> passes = job.Passes;
            int passSteps = 0;
            foreach (DiffusionVideoStage pass in passes)
            {
                if (pass.Sigmas is not { Length: >= 2 } || pass.Sigmas[^1] != 0f)
                    throw new InvalidOperationException($"{_family.DisplayName}'s sampling pass '{pass.Name}' needs sigmas ending in 0.");
                passSteps += pass.Steps;
            }
            if (passSteps != job.Steps)
                throw new InvalidOperationException($"{_family.DisplayName}'s sampling passes take {passSteps} steps, not the clip's {job.Steps}.");
            DiffusionStreams state = _family.InitialStreams(job);
            EnterPhase(request, DiffusionPhase.Denoise, DiffusionComponent.Denoiser, plan);
            IDiffusionSampler sampler = job.Recipe.CreateSampler();
            var conditioned = sampler as IConditionedSampler;
            var stepSeconds = new double[job.Steps];
            int done = 0;
            try
            {
                for (int s = 0; s < passes.Count; s++)
                {
                    DiffusionVideoStage pass = passes[s];
                    if (s > 0)
                    {
                        ct.ThrowIfCancellationRequested();
                        _running = DiffusionComponent.Vae;
                        state = _family.NextStage(job, s, state, new DiffusionVideoComponents(this), report, ct)
                            ?? throw new InvalidOperationException($"{_family.DisplayName} made no state for its pass '{pass.Name}'.");
                        if (_videoCodec != null) _resident.Add(DiffusionComponent.Vae);
                        _running = DiffusionComponent.Denoiser;
                    }
                    sampler.Reset();
                    for (int i = 0; i < pass.Steps; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        var stepClock = Stopwatch.StartNew();
                        float sigma = pass.Sigmas[i], next = pass.Sigmas[i + 1];
                        var step = new StepState(i, pass.Steps, sigma, next, 0, 0) { Stage = s };
                        DenoiseStepInputs inputs = _family.StepInputs(job, conditioning, state, step);
                        StepOutput output = denoiser.Step(inputs, request.Adapters, ct);
                        DiffusionStreams velocities = _family.SamplerVelocities(job, state, inputs, output, step);
                        DiffusionSamplerInputs extra = conditioned != null ? _family.SamplerInputs(job, state, step) : null;
                        foreach (var (name, x) in state.Values)
                        {
                            float[] v = velocities.Values.TryGetValue(name, out float[] found) ? found
                                : throw new InvalidOperationException($"{_family.DisplayName} gave no velocity for its '{name}' stream.");
                            if (conditioned != null) conditioned.Step(x, v, sigma, next, extra?.For(name));
                            else sampler.Step(x, v, sigma, next);
                        }
                        stepSeconds[done] = stepClock.Elapsed.TotalSeconds;
                        done++;
                        _log($"  [diffusion] {_family.Id} step {done}/{job.Steps} sigma {sigma:0.#####} -> {next:0.#####} {stepSeconds[done - 1]:F2}s");
                        Report(request, new DiffusionVideoProgress(DiffusionPhase.Denoise, done, job.Steps, Stage: pass.Name));
                    }
                }
            }
            finally
            {
                denoiser.FinishRequest();
            }
            LeavePhase(request, DiffusionComponent.Denoiser, Demand(job, DiffusionPhase.Denoise));
            double denoiseSeconds = clock.Elapsed.TotalSeconds;
            ct.ThrowIfCancellationRequested();

            // ---- Decode
            clock.Restart();
            EnterPhase(request, DiffusionPhase.Decode, DiffusionComponent.Vae, plan);
            RgbImage[] frames = _family.DecodeVideo(job, state, VideoCodec, report, ct);
            LeavePhase(request, DiffusionComponent.Vae, Demand(job, DiffusionPhase.Decode));
            double decodeSeconds = clock.Elapsed.TotalSeconds;

            // ---- DecodeAudio
            clock.Restart();
            float[][] audio = null;
            int sampleRate = 0;
            if (request.DecodeAudio && job.DecodesAudio && AudioCodec is { } audioCodec)
            {
                ct.ThrowIfCancellationRequested();
                EnterPhase(request, DiffusionPhase.DecodeAudio, DiffusionComponent.AudioVae, plan);
                (audio, sampleRate) = _family.DecodeAudio(job, state, audioCodec, ct);
                LeavePhase(request, DiffusionComponent.AudioVae, Demand(job, DiffusionPhase.DecodeAudio));
            }
            double audioSeconds = clock.Elapsed.TotalSeconds;

            var timings = new DiffusionVideoTimings
            {
                Condition = conditionSeconds,
                Encode = encodeSeconds,
                DenoiserLoad = denoiserLoadSeconds,
                Denoise = denoiseSeconds,
                Steps = stepSeconds,
                Decode = decodeSeconds,
                DecodeAudio = audioSeconds,
                Total = total.Elapsed.TotalSeconds,
            };
            _log($"  [diffusion] {_family.Id} {job.GeometryClass}, {conditioning.Rows} conditioning rows: condition {timings.Condition:F2}s, " +
                 $"denoiser {timings.DenoiserLoad:F2}s, encode {timings.Encode:F2}s, denoise {timings.Denoise:F2}s " +
                 $"({timings.Denoise / Math.Max(1, job.Steps):F2}s/step), decode {timings.Decode:F2}s, audio {timings.DecodeAudio:F2}s, " +
                 $"total {timings.Total:F2}s");
            return new DiffusionVideoResult
            {
                Frames = frames, Audio = audio, AudioSampleRate = sampleRate, Job = job, Timings = timings, Residency = plan,
            };
        }

        /// <summary>The denoiser of <paramref name="job"/>'s checkpoint: the loaded one, or a new one in its place (the old one's
        /// device memory goes first).</summary>
        private IDenoiser DenoiserFor(DiffusionVideoRequest request, DiffusionVideoJob job)
        {
            bool cached = _denoiser != null && string.Equals(_denoiserFile, job.DenoiserFile, StringComparison.Ordinal);
            Report(request, new DiffusionVideoProgress(DiffusionPhase.Condition, Stage: "dit-load",
                Attributes: new Dictionary<string, object> { [TensorSharp.Runtime.Telemetry.TelemetryAttributes.VideoCached] = cached }));
            if (cached) return _denoiser;
            if (_denoiser != null)
            {
                _denoiser.Dispose();
                _denoiser = null;
                _resident.Remove(DiffusionComponent.Denoiser);
                _workspaces.Remove(DiffusionComponent.Denoiser);
            }
            _denoiser = _family.CreateDenoiser(_load, job.DenoiserFile)
                ?? throw new InvalidOperationException($"{_family.DisplayName} made no denoiser.");
            _denoiserFile = job.DenoiserFile;
            return _denoiser;
        }

        private static long Demand(DiffusionVideoJob job, DiffusionPhase phase) =>
            job.Demands.FirstOrDefault(d => d.Phase == phase).WorkingBytes;

        // ---- residency

        private ResidencyPlan PlanResidency(DiffusionVideoJob job)
        {
            var weights = new Dictionary<DiffusionComponent, long>
            {
                [DiffusionComponent.Conditioner] = Conditioner.WeightBytes,
                [DiffusionComponent.Vae] = VideoCodec.WeightBytes,
            };
            if (AudioCodec is { } audio) weights[DiffusionComponent.AudioVae] = audio.WeightBytes;
            // The loaded denoiser's bytes stand for the clip's: the family's checkpoints of one load are one shape.
            weights[DiffusionComponent.Denoiser] = _denoiser?.WeightBytes ?? DenoiserBytesEstimate(job.DenoiserFile);
            return _planner.Plan(job.Demands, weights, _resident.ToArray(), new Dictionary<DiffusionComponent, long>(_workspaces), job.GeometryClass);
        }

        private static long DenoiserBytesEstimate(string file)
        {
            try { return string.IsNullOrEmpty(file) ? 0 : new System.IO.FileInfo(file).Length; }
            catch (Exception e) when (e is System.IO.IOException or UnauthorizedAccessException) { return 0; }
        }

        private void EnterPhase(DiffusionVideoRequest request, DiffusionPhase phase, DiffusionComponent component, ResidencyPlan plan)
        {
            foreach (ResidencyRelease release in plan.Before(phase))
            {
                IResidentComponent released = Created(release.Component);
                if (released == null) continue;
                released.ReleaseWorkspace();
                _workspaces.Remove(release.Component);
                if (!release.Weights) continue;
                if (release.Component == DiffusionComponent.Denoiser)
                    Report(request, new DiffusionVideoProgress(phase, Stage: "denoiser-release",
                        Attributes: new Dictionary<string, object> { ["tensorsharp.video.release_reason"] = "vram" }));
                released.ReleaseDevice();
                _resident.Remove(release.Component);
            }
            _resident.Add(component);
            _running = component;
            Report(request, new DiffusionVideoProgress(phase));
        }

        private void LeavePhase(DiffusionVideoRequest request, DiffusionComponent component, long workingBytes)
        {
            _workspaces[component] = Math.Max(_workspaces.GetValueOrDefault(component), workingBytes);
            _running = null;
            if (Created(component) is { ReleaseAfterPhase: true } released)
            {
                // The denoiser's release is a stage of the decode it makes room for, as a planned one is (EnterPhase); the
                // reason is the backend's when that is why the component releases (Metal's wired zero-copy weights).
                if (component == DiffusionComponent.Denoiser)
                    Report(request, new DiffusionVideoProgress(DiffusionPhase.Decode, Stage: "denoiser-release",
                        Attributes: new Dictionary<string, object>
                        {
                            ["tensorsharp.video.release_reason"] = _load.Backend == BackendType.GgmlMetal ? "metal" : "after-phase",
                        }));
                released.ReleaseWorkspace();
                released.ReleaseDevice();
                _workspaces.Remove(component);
                _resident.Remove(component);
            }
        }

        /// <summary>A phase failed: its component's device copies are no longer known, so they are released (see
        /// <see cref="DiffusionPipeline"/>'s ReleaseFailedPhase).</summary>
        private void ReleaseFailedPhase()
        {
            if (_running is not DiffusionComponent failed) return;
            _running = null;
            try { Created(failed)?.ReleaseDevice(); }
            catch (Exception error) { _log($"  [diffusion] releasing the {failed.ToString().ToLowerInvariant()} after its phase failed: {error.Message}"); }
            _resident.Remove(failed);
            _workspaces.Remove(failed);
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

        /// <summary>Release every component's device memory now (weights and workspaces); their host mappings stay. A host
        /// calls it when another model needs the device.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                foreach (DiffusionComponent c in Enum.GetValues<DiffusionComponent>())
                {
                    IResidentComponent component = Created(c);
                    if (component == null) continue;
                    component.ReleaseWorkspace();
                    component.ReleaseDevice();
                }
                _resident.Clear();
                _workspaces.Clear();
            }
        }

        private void Report(DiffusionVideoRequest request, DiffusionVideoProgress progress) =>
            request.Progress?.Invoke(progress.Total == 0 && _runSteps > 0 ? progress with { Total = _runSteps } : progress);

        // ---- components

        /// <summary>The components made so far (null where none is), without making one: a family's host reads their state
        /// between clips (a prompt cache's hits, whether a network is built).</summary>
        internal IConditioner ConditionerOrNull => Volatile.Read(ref _conditioner);
        internal IResidentComponent VideoCodecOrNull => Volatile.Read(ref _videoCodec);
        internal IResidentComponent AudioCodecOrNull => Volatile.Read(ref _audioCodec);

        internal IConditioner ConditionerComponent { get { lock (_gate) return Conditioner; } }
        internal IResidentComponent VideoCodecComponent { get { lock (_gate) return VideoCodec; } }
        internal IResidentComponent AudioCodecComponent { get { lock (_gate) return AudioCodec; } }

        private IConditioner Conditioner => _conditioner ??= _family.CreateConditioner(_load)
            ?? throw new InvalidOperationException($"{_family.DisplayName} made no conditioner.");
        private IResidentComponent VideoCodec => _videoCodec ??= _family.CreateVideoCodec(_load)
            ?? throw new InvalidOperationException($"{_family.DisplayName} made no video codec.");
        private IResidentComponent AudioCodec
        {
            get
            {
                if (!_audioCodecMade)
                {
                    _audioCodec = _family.CreateAudioCodec(_load);
                    _audioCodecMade = true;
                }
                return _audioCodec;
            }
        }

        private IResidentComponent Created(DiffusionComponent component) => component switch
        {
            DiffusionComponent.Conditioner => _conditioner,
            DiffusionComponent.Vae => _videoCodec,
            DiffusionComponent.AudioVae => _audioCodec,
            DiffusionComponent.Denoiser => _denoiser,
            _ => null,
        };

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _denoiser?.Dispose(); _denoiser = null; _denoiserFile = null;
                _audioCodec?.Dispose(); _audioCodec = null;
                _videoCodec?.Dispose(); _videoCodec = null;
                _conditioner?.Dispose(); _conditioner = null;
                _resident.Clear();
                _workspaces.Clear();
            }
        }
    }
}
