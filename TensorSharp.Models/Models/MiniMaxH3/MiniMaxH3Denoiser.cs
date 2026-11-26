// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3's audio-video denoiser on the diffusion runtime: one step is one GraphProgram (the token refiner, the packed
// [text | conditions | audio | video] sequence through the 50 blocks, and the final layer's video and audio heads), emitted
// through GraphEmitter with the diffusion-transformer regions (GraphDitRegions) and run by the backend's executor: the GGML
// graph executor on ggml_cuda, ggml_metal and ggml_cpu, and the tensor executors on cpu (managed), cuda and mlx. The program
// is the graph of the retired native forward (TSGgml_MiniMaxH3DitForward) node for node, so a backend that lowers it as that
// graph did computes the same velocities; MiniMax-H3-Image's denoiser is the same transformer without its audio stream.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>What one step of the audio-video denoiser reads beside the target's video tokens: the packed layout, the
    /// conditioning frames and sound (in the layout's chunk order) and the two streams' shifts (the audio head's scale).</summary>
    internal sealed class MiniMaxH3StepLayout
    {
        public required MiniMaxH3PackedLayout Layout { get; init; }

        /// <summary>The conditioning video patches [96, ConditionCount] (keyframes or references), or null.</summary>
        public float[] ConditionTokens { get; init; }
        public int ConditionCount { get; init; }

        /// <summary>The conditioning audio latents [32, ConditionAudioCount] (Ref2VA soundtracks), or null.</summary>
        public float[] ConditionAudio { get; init; }
        public int ConditionAudioCount { get; init; }

        /// <summary>The conditioning run's chunks in sequence order (kind 0 video patches, 1 audio latents); null or empty: the
        /// conditioning video patches as one run.</summary>
        public IReadOnlyList<(int Kind, int Count)> Chunks { get; init; }

        public float VideoShift { get; init; } = MiniMaxH3Scheduler.VideoShift;
        public float AudioShift { get; init; } = MiniMaxH3Scheduler.AudioShift;
    }

    /// <summary>
    /// The MiniMax-H3 audio-video DiT as an <see cref="IDenoiser"/>. Making one maps the GGUF and reads its small tensors on
    /// the host; the first step makes the weights resident (<see cref="Preload"/>: MiniMax-H3's uploader on the GGML backends)
    /// and compiles the step's program, which every later step of the same geometry replays with new inputs. The step's video
    /// velocity is negated and its audio prediction scaled by the shift slope, the retired forward's component contract.
    /// </summary>
    public sealed class MiniMaxH3Denoiser : IDenoiser
    {
        /// <summary>The name of the audio stream in <see cref="DenoiseStepInputs.Streams"/> and <see cref="StepOutput.Streams"/>.</summary>
        public const string AudioStream = "audio";

        /// <summary>MiniMax-H3's q8_1 overflow guard around a product of a block-quantized weight (h3_mm's kH3MmScale).</summary>
        internal const float GuardScale = 1024f;

        /// <summary>The managed executor's intermediates kept for reuse (<see cref="RecyclingCpuAllocator"/>): every block of a
        /// step frees and allocates the same ones (about 0.9 GB a block at 2,250 tokens), which as fresh mappings fault their
        /// pages in again each time; returned when a clip's denoising ends.</summary>
        internal const long ManagedRetainedBytes = 2L << 30;

        private readonly MiniMaxH3DenoiserWeights _w;
        private readonly IGraphExecutor _executor;
        private readonly IAllocator _tensorAllocator;   // tensor executors bind weights as objects: their F32 vectors live here
        private readonly bool _ownsAllocator;
        private readonly Dictionary<MiniMaxH3DitWeight, Tensor> _tensors = new();
        private readonly object _gate = new();
        private (float[] Positions, float[] Cos, float[] Sin, float[] SinSigned) _rope;
        private IGraphPlan _plan;
        private (ulong, ulong) _planSignature;
        // The token refiner's program (the condition projection and the refiner blocks, run once a prompt) and its result for
        // the last prompt and LoRA set, as the retired direct DiT cached its refined text between steps and clips.
        private IGraphPlan _textPlan;
        private (ulong, ulong) _textPlanSignature;
        private float[] _refinedFor;      // the text rows the refined rows were made from
        private long _refinedLora = -1;   // the LoRA set's generation they were made with (0: none)
        private float[] _refined;         // [hidden, textCount]
        private bool _preloaded;
        private bool _disposed;
        // The LoRA set the cached plan binds (null: none), and a tensor executor's handles for its factors.
        private MiniMaxH3LoraSet _lora;
        private readonly Dictionary<IntPtr, MiniMaxH3DitWeight> _loraHandles = new();

        /// <summary>The denoiser in <paramref name="path"/> on <paramref name="backend"/>: a GGML backend (its graph executor),
        /// <see cref="BackendType.Cpu"/> (the managed executor) or a backend with a tensor executor for
        /// <paramref name="allocator"/> (<see cref="BackendType.Cuda"/>, <see cref="BackendType.Mlx"/>).</summary>
        public MiniMaxH3Denoiser(string path, BackendType backend, IAllocator allocator = null)
        {
            Backend = backend;
            _w = new MiniMaxH3DenoiserWeights(path, backend);
            try
            {
                (_executor, _tensorAllocator, _ownsAllocator) = Resolve(backend, allocator);
            }
            catch
            {
                _w.Dispose();
                throw;
            }
        }

        private static (IGraphExecutor, IAllocator, bool) Resolve(BackendType backend, IAllocator allocator)
        {
            switch (backend)
            {
                case BackendType.GgmlCuda: return (GgmlGraphExecutor.For(GgmlBackendType.Cuda), null, false);
                case BackendType.GgmlMetal: return (GgmlGraphExecutor.For(GgmlBackendType.Metal), null, false);
                case BackendType.GgmlCpu: return (GgmlGraphExecutor.For(GgmlBackendType.Cpu), null, false);
                case BackendType.Cpu:
                {
                    var cpu = new RecyclingCpuAllocator(BlasEnum.DotNet, ManagedRetainedBytes);
                    return (new ManagedGraphExecutor(cpu) { ServesWideDitRegions = true }, cpu, true);
                }
                default:
                {
                    IGraphExecutor executor = GraphExecutors.For(allocator)
                        ?? throw new NotSupportedException(
                            $"MiniMax-H3's denoiser runs on ggml_cuda, ggml_metal, ggml_cpu, cpu, cuda or mlx, not {Architecture.BackendNames.OptionName(backend)}.");
                    return (executor, allocator, false);
                }
            }
        }

        public BackendType Backend { get; }

        public MiniMaxH3Config Config => _w.Config;

        /// <summary>The checkpoint's file.</summary>
        public string Path => _w.Path;

        /// <summary>The executor the steps run on.</summary>
        internal IGraphExecutor Executor => _executor;

        /// <summary>Whether the steps run on a tensor executor (weights bound as objects, F32 LoRA factors).</summary>
        internal bool OnTensorExecutor => _tensorAllocator != null;

        /// <summary>The plan the last step ran (a test seam).</summary>
        internal IGraphPlan LastPlan => _plan;

        /// <summary>Steps that compiled their program (the rest replayed the cached plan).</summary>
        internal int Compiles { get; private set; }

        /// <summary>Prompts the token refiner ran for (a step whose prompt and LoRA set the last refined rows were made with
        /// takes them as they are).</summary>
        internal int Refinements { get; private set; }

        public DenoiserCaps Caps { get; } = new() { Streams = new[] { AudioStream } };

        public long WeightBytes => _w.Bytes + (AdapterBytes?.Invoke() ?? 0);

        /// <summary>The device bytes of the LoRA factors the next steps bind, for residency.</summary>
        internal Func<long> AdapterBytes { get; set; }

        /// <summary>The LoRA set the cached plan was compiled with (a test seam).</summary>
        internal MiniMaxH3LoraSet BoundLora => _lora;

        /// <summary>A step's activations for <paramref name="workload"/> (its target's and references' rows and the prompt): see
        /// <see cref="WorkingBytesForRows"/>.</summary>
        public long WorkingBytes(DiffusionWorkload workload)
        {
            long rows = (long)workload.Target.Frames * (workload.Target.Height / 2) * (workload.Target.Width / 2);
            if (workload.References != null)
                foreach (LatentShape r in workload.References) rows += (long)r.Frames * (r.Height / 2) * (r.Width / 2);
            rows += Math.Max(1, (workload.Prompt?.Length ?? 0) / 3);
            return WorkingBytesForRows(rows, workload.References is { Count: > 0 });
        }

        /// <summary>
        /// A step's compute buffer for <paramref name="rows"/> packed rows: the measured peaks of MiniMax-H3-Image's denoiser,
        /// the same transformer (MiniMaxH3Image.MiniMaxH3ImageDenoiser.WorkingBytesFor: the fused ggml_cuda step, the unfused one
        /// elsewhere), with a conditioned clip's second timestep (<paramref name="conditioned"/>) as an edit's.
        /// </summary>
        public long WorkingBytesForRows(long rows, bool conditioned)
        {
            bool quantized = false, bf16Route = false;
            foreach (MiniMaxH3DitWeight w in _w.All)
            {
                quantized |= w.Quantized;
                bf16Route |= w.Format == GgmlQ4K;
            }
            return MiniMaxH3Image.MiniMaxH3ImageDenoiser.WorkingBytesFor(_w.Config, rows, fused: Backend == BackendType.GgmlCuda, quantized,
                bf16Route, edit: conditioned);
        }

        /// <summary>GGML_TYPE_Q4_K: the block type the CUDA step's wide products take to cuBLAS BF16.</summary>
        private const int GgmlQ4K = 12;

        /// <summary>Make the weights resident now (MiniMax-H3's uploader on the GGML backends: direct reads from the file
        /// where the device allows, else parallel copies out of the mapping); the first step does it otherwise. Returns how
        /// many weights were copied.</summary>
        public int Preload()
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return PreloadLocked();
            }
        }

        private int PreloadLocked()
        {
            if (_preloaded) return 0;
            int copied = _executor.Capabilities.BindsHostPointers
                ? _executor.Preload(new List<IGraphWeight>(_w.All), "minimax-h3/denoiser")
                : 0;
            _preloaded = true;
            return copied;
        }

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            MiniMaxH3LoraSet lora = LoraOf(adapters);
            cancellationToken.ThrowIfCancellationRequested();
            var step = inputs.Layout.FamilyLayout as MiniMaxH3StepLayout
                ?? throw new ArgumentException("the layout was not packed by MiniMax-H3's family", nameof(inputs));
            if (inputs.Streams == null || !inputs.Streams.TryGetValue(AudioStream, out float[] audio))
                throw new ArgumentException("a MiniMax-H3 step reads its audio stream", nameof(inputs));
            ConditioningSet text = inputs.Conditioning;
            int videoCount = checked((int)(inputs.Latent.LongLength / _w.Config.VideoPatchDim));
            int audioCount = checked((int)(audio.LongLength / _w.Config.AudioLatentChannels));
            var (video, audioVelocity) = Forward(lora, inputs.Latent, videoCount, audio, audioCount, text.Hidden, text.Rows, step.Layout,
                inputs.Step.Sigma, step.ConditionTokens, step.ConditionCount, step.ConditionAudio, step.ConditionAudioCount, step.Chunks,
                step.VideoShift, step.AudioShift);
            return new StepOutput(video, new Dictionary<string, float[]> { [AudioStream] = audioVelocity });
        }

        /// <summary>The packed LoRA set a request's adapters carry (each adapter's payload is the one set the model loaded), or
        /// null for none.</summary>
        private MiniMaxH3LoraSet LoraOf(AdapterSet adapters)
        {
            if (adapters is null or { IsEmpty: true }) return null;
            var set = adapters.Adapters[0].Payload as MiniMaxH3LoraSet
                ?? throw new ArgumentException($"LoRA {adapters.Adapters[0].Name} was not loaded by MiniMax-H3's LoRA loader.", nameof(adapters));
            foreach (DiffusionAdapter a in adapters.Adapters)
                if (!ReferenceEquals(a.Payload, set))
                    throw new ArgumentException("A clip's LoRA adapters are one loaded set: the adapters carry different sets.", nameof(adapters));
            return set;
        }

        /// <summary>Bind <paramref name="lora"/> for the steps that follow: a different set than the cached plan's drops the plan
        /// and the previous set's device copies first, so no step ever runs with factors of a set it was not given.</summary>
        private void BindLora(MiniMaxH3LoraSet lora)
        {
            if (ReferenceEquals(lora, _lora)) return;
            _plan?.Dispose();
            _plan = null;
            DropTextPlan();
            ReleaseLoraDevice();
            _lora = lora;
        }

        /// <summary>Drop the token refiner's plan and its cached rows.</summary>
        private void DropTextPlan()
        {
            _textPlan?.Dispose();
            _textPlan = null;
            _refined = null;
            _refinedFor = null;
            _refinedLora = -1;
        }

        /// <summary>Drop the bound set's device copies and the tensor executor's handles for them.</summary>
        private void ReleaseLoraDevice()
        {
            if (_lora == null) return;
            if (_tensorAllocator == null) _lora.InvalidateDeviceCopies();
            foreach (IntPtr factor in _loraHandles.Keys) ReleaseCopy(factor);
            _loraHandles.Clear();
        }

        /// <summary>The host addresses the device copies of the weights and of the bound LoRA set's factors are keyed by (tests).</summary>
        internal IReadOnlyList<IntPtr> DeviceCopyKeysForTest()
        {
            lock (_gate)
            {
                var keys = new List<IntPtr>();
                foreach (MiniMaxH3DitWeight w in _w.All) keys.Add(w.Data);
                keys.AddRange(_loraHandles.Keys);
                return keys;
            }
        }

        /// <summary>Drop the device copy of the host bytes at <paramref name="data"/>. The GGML graph executor keys it by those
        /// bytes (<see cref="IGraphExecutor.Invalidate"/>); the CUDA and MLX tensor executors keep it in their backends'
        /// resident-weight caches under the same address, which Invalidate does not reach (NetworkWeights releases its weights
        /// the same way, as the retired direct DiT's linears did). A copy left there outlives the denoiser: its device memory
        /// stays taken, and a file mapped later at that address computes with the old values.</summary>
        private void ReleaseCopy(IntPtr data)
        {
            if (data == IntPtr.Zero) return;
            switch (_tensorAllocator)
            {
                case TensorSharp.Cuda.CudaAllocator cuda: TensorSharp.Cuda.CudaQuantizedOps.ReleaseQuantizedWeight(cuda, data); break;
                case TensorSharp.MLX.MlxAllocator mlx: TensorSharp.MLX.MlxQuantizedOps.ReleaseQuantizedWeight(mlx, data); break;
                default: _executor.Invalidate(data); break;
            }
        }

        /// <summary>Forget the bound LoRA set (its device copies and the plan that binds them): the model calls it under its
        /// generation lock before it disposes a set it replaces.</summary>
        internal void ReleaseLora()
        {
            lock (_gate)
            {
                if (_disposed) return;
                BindLora(null);
            }
        }

        /// <summary>One step without a LoRA plug-in, in the retired MiniMaxH3DiT.Forward's argument order (the component
        /// oracles' entry). <paramref name="blockLimit"/> above 0 runs only the first blocks: an oracle of a model slice.</summary>
        internal (float[] Video, float[] Audio) Forward(float[] videoTokens, int videoCount, float[] audioLatent, int audioCount,
            float[] textHidden, int textCount, MiniMaxH3PackedLayout layout, float sigma, float[] conditionTokens = null,
            int conditionCount = 0, float[] conditionAudio = null, int conditionAudioCount = 0,
            IReadOnlyList<(int Kind, int Count)> conditionChunks = null, float videoShift = MiniMaxH3Scheduler.VideoShift,
            float audioShift = MiniMaxH3Scheduler.AudioShift, int blockLimit = 0) =>
            Forward(null, videoTokens, videoCount, audioLatent, audioCount, textHidden, textCount, layout, sigma, conditionTokens,
                conditionCount, conditionAudio, conditionAudioCount, conditionChunks, videoShift, audioShift, blockLimit);

        /// <summary>
        /// One step, as the retired MiniMaxH3DiT.Forward took it: [text | conditions | audio | video] through the trunk, the
        /// target's video velocity (negated) and its audio prediction scaled by <c>-dsigma_audio/dsigma_video</c> at
        /// <paramref name="sigma"/>. The video result is the plan's output buffer, valid until the next step.
        /// </summary>
        internal (float[] Video, float[] Audio) Forward(MiniMaxH3LoraSet lora, float[] videoTokens, int videoCount, float[] audioLatent,
            int audioCount, float[] textHidden, int textCount, MiniMaxH3PackedLayout layout, float sigma, float[] conditionTokens = null,
            int conditionCount = 0, float[] conditionAudio = null, int conditionAudioCount = 0,
            IReadOnlyList<(int Kind, int Count)> chunks = null, float videoShift = MiniMaxH3Scheduler.VideoShift,
            float audioShift = MiniMaxH3Scheduler.AudioShift, int blockLimit = 0)
        {
            ArgumentNullException.ThrowIfNull(videoTokens);
            ArgumentNullException.ThrowIfNull(audioLatent);
            ArgumentNullException.ThrowIfNull(layout);
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                MiniMaxH3Config c = _w.Config;
                if ((long)videoCount * c.VideoPatchDim != videoTokens.LongLength)
                    throw new ArgumentException($"{videoCount} video tokens hold {(long)videoCount * c.VideoPatchDim} values, not {videoTokens.LongLength}.");
                if (audioCount <= 0 || (long)audioCount * c.AudioLatentChannels != audioLatent.LongLength)
                    throw new ArgumentException("a MiniMax-H3 step needs its audio tokens, audioCount latents of the audio width.");
                if (conditionCount > 0 && (conditionTokens is null || conditionTokens.LongLength < (long)conditionCount * c.VideoPatchDim))
                    throw new ArgumentException("the conditioning tokens do not hold conditionCount patches.");
                if (conditionAudioCount > 0 && (conditionAudio is null || conditionAudio.LongLength < (long)conditionAudioCount * c.AudioLatentChannels))
                    throw new ArgumentException("the conditioning audio does not hold conditionAudioCount latents.");
                if (textCount > 0 && (textHidden is null || textHidden.LongLength != (long)textCount * c.TextDim))
                    throw new ArgumentException($"{textCount} text rows of {c.TextDim} need {(long)textCount * c.TextDim} values.");
                if (layout.VideoSpan.Length != videoCount || layout.AudioSpan.Length != audioCount)
                    throw new ArgumentException("the layout does not pack these audio and video rows.");
                if (lora != null && !lora.Fits(c))
                    throw new ArgumentException("The LoRA set was loaded for a MiniMax-H3 checkpoint of another shape.", nameof(lora));
                if (lora != null && lora.ForTensorExecutor != OnTensorExecutor)
                    throw new ArgumentException(lora.ForTensorExecutor
                        ? "This MiniMax-H3 LoRA set was packed for a tensor executor; the GGML graph executor needs its packed factors."
                        : "This MiniMax-H3 LoRA set is packed for the GGML graph executor; a tensor executor reads F32 factors.", nameof(lora));
                BindLora(lora);
                CheckChunks(chunks, conditionCount, conditionAudioCount);

                var step = new StepValues
                {
                    Video = conditionCount > 0 ? Concat(conditionTokens, (long)conditionCount * c.VideoPatchDim, videoTokens) : videoTokens,
                    VideoCount = videoCount,
                    ConditionCount = conditionCount,
                    Audio = audioLatent,
                    AudioCount = audioCount,
                    ConditionAudio = conditionAudioCount > 0 ? conditionAudio : null,
                    ConditionAudioCount = conditionAudioCount,
                    Chunks = chunks,
                    Text = textCount > 0 ? textHidden : null,
                    TextCount = textCount,
                    Layout = layout,
                    Segments = Segments(layout),
                    TimeEmbed = TimeEmbedding(layout.Timesteps),
                    Lora = lora,
                    Blocks = blockLimit > 0 ? Math.Min(blockLimit, _w.Blocks.Length) : _w.Blocks.Length,
                };
                step.Cos = Rope(layout.Positions).Cos;
                step.SinSigned = _rope.SinSigned;
                PreloadLocked();
                if (textCount > 0) step.Refined = RefinedText(step);

                // With a cached plan, the program's signature first (a hash-only trace, whose inputs this step runs with). A
                // geometry the plan was not compiled for (or the clip's first step, with none) records the program and compiles
                // it, after the old plan's memory is released, and that recording runs the step: one emission, not two.
                GraphEmitter runner = null;
                try
                {
                    if (_plan != null)
                    {
                        var trace = new GraphEmitter(_executor, record: false, floatProducts: FloatProducts);
                        Emit(trace, step, text: false);
                        if (_planSignature == trace.Signature) runner = trace;
                        else trace.Dispose();
                    }
                    if (runner == null)
                    {
                        _plan?.Dispose();
                        _plan = null;
                        var clock = System.Diagnostics.Stopwatch.StartNew();
                        var recorded = new GraphEmitter(_executor, record: true, floatProducts: FloatProducts);
                        runner = recorded;
                        Emit(recorded, step, text: false);
                        double emitted = clock.Elapsed.TotalSeconds;
                        _plan = recorded.Compile(replayable: true);
                        _planSignature = recorded.Signature;
                        Compiles++;
                        Console.WriteLine($"  [h3] step program for {layout.TokenCount} tokens: emitted in {emitted:F2}s, " +
                                          $"compiled in {clock.Elapsed.TotalSeconds - emitted:F2}s");
                    }
                    if (!runner.Run(_plan))
                        throw new InvalidOperationException("MiniMax-H3 denoiser step failed: " + _plan.LastError);
                }
                finally
                {
                    runner?.Dispose();
                }
                // The audio head's scale changes every step, so it is applied here rather than in the program (whose signature
                // would change with it): x * s in F32, as the retired graph's scale op multiplies.
                float audioScale = -MiniMaxH3Scheduler.ShiftSlope(MiniMaxH3Scheduler.ClampSigma(sigma), videoShift, audioShift);
                float[] audioRaw = _plan.Outputs[1];
                var audio = new float[audioRaw.LongLength];
                for (long i = 0; i < audio.LongLength; i++) audio[i] = audioRaw[i] * audioScale;
                return (_plan.Outputs[0], audio);
            }
        }

        /// <summary>
        /// The token refiner's rows for the step's prompt: the condition projection and the refiner blocks as a program of their
        /// own, run once a prompt and LoRA set (the retired direct DiT cached them between steps and clips the same way; they
        /// depend on neither the latents nor the timestep), the step program reading them as an input. The values are the ones
        /// the step computed itself: the same ops on the same operands, in an F32 buffer between the two programs.
        /// </summary>
        private float[] RefinedText(StepValues step)
        {
            long generation = step.Lora?.Generation ?? 0;
            if (_refined != null && _refinedLora == generation && _refinedFor != null
                && MemoryMarshal.AsBytes(_refinedFor.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(step.Text.AsSpan())))
                return _refined;
            GraphEmitter runner = null;
            try
            {
                if (_textPlan != null)
                {
                    var trace = new GraphEmitter(_executor, record: false, floatProducts: FloatProducts);
                    Emit(trace, step, text: true);
                    if (_textPlanSignature == trace.Signature) runner = trace;
                    else trace.Dispose();
                }
                if (runner == null)
                {
                    _textPlan?.Dispose();
                    _textPlan = null;
                    var recorded = new GraphEmitter(_executor, record: true, floatProducts: FloatProducts);
                    runner = recorded;
                    Emit(recorded, step, text: true);
                    _textPlan = recorded.Compile(replayable: true);
                    _textPlanSignature = recorded.Signature;
                }
                if (!runner.Run(_textPlan))
                    throw new InvalidOperationException("MiniMax-H3 token refiner failed: " + _textPlan.LastError);
            }
            finally
            {
                runner?.Dispose();
            }
            Refinements++;
            _refined = (float[])_textPlan.Outputs[0].Clone();
            _refinedFor = (float[])step.Text.Clone();
            _refinedLora = generation;
            return _refined;
        }

        /// <summary>Each executor takes the products the retired path took on its backend: on direct CUDA (and MLX) the products
        /// keep their activations in F32, as the retired direct DiT's projections did (on CUDA, bounded F32 bands over the
        /// block formats); the managed executor takes its own products (ManagedQuantizedOps, whose block-format kernels quantize
        /// a few rows' activations), as the retired direct CPU DiT did, and serves the step's attentions and LoRA regions all
        /// the same (<see cref="TensorGraphExecutor.ServesWideDitRegions"/>); the GGML graph executor takes them as the retired
        /// native graph did.</summary>
        private bool FloatProducts => _tensorAllocator != null && _executor is not ManagedGraphExecutor;

        private static void CheckChunks(IReadOnlyList<(int Kind, int Count)> chunks, int conditionCount, int conditionAudioCount)
        {
            if (chunks is not { Count: > 0 })
            {
                if (conditionAudioCount > 0)
                    throw new ArgumentException("conditioning audio needs a chunk list that places it in the sequence.");
                return;
            }
            long video = 0, audio = 0;
            foreach (var (kind, count) in chunks)
            {
                if (kind is not (0 or 1) || count < 0) throw new ArgumentException($"bad conditioning chunk ({kind}, {count}).");
                if (kind == 0) video += count; else audio += count;
            }
            if (video != conditionCount || audio != conditionAudioCount)
                throw new ArgumentException("MiniMax-H3's conditioning chunks do not cover every conditioning row.");
        }

        private static float[] Concat(float[] a, long aCount, float[] b)
        {
            var r = new float[aCount + b.LongLength];
            Array.Copy(a, r, aCount);
            Array.Copy(b, 0, r, aCount, b.LongLength);
            return r;
        }

        // ---- the step's host values

        /// <summary>The layout's modulation spans as the regions' segments: a row timestep * 3 + modality maps to column
        /// base timestep * 18 + modality * 6 of the block projection viewed as [hidden, 18 * timesteps].</summary>
        private static GraphSegment[] Segments(MiniMaxH3PackedLayout layout)
        {
            var segments = new GraphSegment[layout.ModulationSpans.Count];
            for (int i = 0; i < segments.Length; i++)
            {
                MiniMaxH3ModulationSpan span = layout.ModulationSpans[i];
                int ts = span.Row / MiniMaxH3Modality.Count, modality = span.Row % MiniMaxH3Modality.Count;
                segments[i] = new GraphSegment(span.Start, span.End,
                    ts * (MiniMaxH3Modality.VectorsPerRow * MiniMaxH3Modality.Count) + modality * MiniMaxH3Modality.VectorsPerRow);
            }
            return segments;
        }

        /// <summary>The AdaLN curve table interpolated at each timestep (a learned curve stands in for a timestep MLP).</summary>
        internal float[] TimeEmbedding(IReadOnlyList<float> timesteps)
        {
            int dim = _w.Config.TimeEmbedDim, grid = _w.Config.AdaLnCurveGrid;
            float[] table = _w.CurveTable;
            var outp = new float[(long)dim * timesteps.Count];
            for (int t = 0; t < timesteps.Count; t++)
            {
                var (lo, hi, frac) = MiniMaxH3Layout.CurveIndex(timesteps[t], grid);
                long dst = (long)t * dim;
                for (int c = 0; c < dim; c++)
                {
                    float a = table[(long)lo * dim + c];
                    float b = table[(long)hi * dim + c];
                    outp[dst + c] = a + (b - a) * frac;
                }
            }
            return outp;
        }

        /// <summary>The 3-axis partial-RoPE tables of the layout's continuous float positions (3 axes x the learned inverse
        /// frequencies give rot / 2 angles, duplicated so pairs (j, j + half) share one), kept for the request with the signed
        /// sine the program reads: the positions are the same every step.</summary>
        internal (float[] Cos, float[] Sin) Rope(float[] positions)
        {
            ArgumentNullException.ThrowIfNull(positions);
            if (_rope.Positions != null && MemoryMarshal.AsBytes(positions.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(_rope.Positions.AsSpan())))
                return (_rope.Cos, _rope.Sin);
            int nf = _w.Config.RopeInvFreqLength;
            int rot = _w.Config.RopeRotaryDim;
            int half = rot / 2;
            int tokens = positions.Length / 3;
            float[] invFreq = _w.InvFreq;
            var cos = new float[(long)rot * tokens];
            var sin = new float[(long)rot * tokens];
            for (int t = 0; t < tokens; t++)
            {
                long b = (long)t * rot;
                for (int axis = 0; axis < 3; axis++)
                {
                    float pos = positions[t * 3 + axis];
                    for (int k = 0; k < nf; k++)
                    {
                        double a = (double)pos * invFreq[k];
                        int j = axis * nf + k;
                        float co = (float)Math.Cos(a), s = (float)Math.Sin(a);
                        cos[b + j] = co; cos[b + j + half] = co;
                        sin[b + j] = s; sin[b + j + half] = s;
                    }
                }
            }
            // [-sin(first half), sin(second half)] a row: the table GraphDitRegions.SignedSine makes from sin.
            var signed = (float[])sin.Clone();
            for (int t = 0; t < tokens; t++)
                for (int j = 0; j < half; j++) signed[(long)t * rot + j] = -signed[(long)t * rot + j];
            _rope = ((float[])positions.Clone(), cos, sin, signed);
            return (cos, sin);
        }

        // ---- emission

        private sealed class StepValues
        {
            public float[] Video;           // [patch, conditions + video] (conditions first: they share the patch projection)
            public int VideoCount, ConditionCount;
            public float[] Audio;           // [audio, audioCount]
            public int AudioCount;
            public float[] ConditionAudio;  // [audio, conditionAudioCount] or null
            public int ConditionAudioCount;
            public IReadOnlyList<(int Kind, int Count)> Chunks;
            public float[] Text;            // [textDim, textCount] or null
            public int TextCount;
            public float[] Refined;         // the token refiner's rows [hidden, textCount] (the step program's input) or null
            public MiniMaxH3PackedLayout Layout;
            public GraphSegment[] Segments;
            public float[] TimeEmbed;       // [timeEmbedDim, timesteps]
            public float[] Cos, SinSigned;  // [rot, tokens]: cos, and [-sin(first half), sin(second half)]
            public MiniMaxH3LoraSet Lora;   // null: no adapters
            public int Blocks;              // the blocks the step runs (all but in a slice's oracle)
        }

        /// <summary>
        /// The step's program, in the order and with the operands of the retired TSGgml_MiniMaxH3DitForward graph: the q8_1 guard
        /// around every block-quantized product but the AdaLN projections' (h3_mm), the token refiner unmodulated and without
        /// RoPE, the conditioning run rebuilt in its chunks' order from the shared video and audio patch projections, each
        /// block's modulations, attention, gated residuals and SwiGLU as the diffusion-transformer regions (whose ops are that
        /// graph's unfused nodes, and which ggml_cuda serves as its fused nodes in a WideTokenProducts program), and the final
        /// layer's per-timestep shift and scale and its two heads; the video velocity negated. The per-step values (the latents,
        /// the time embedding) are inputs; the segment table, positions and text are fixed for a request, so a geometry's steps
        /// share one plan. With <paramref name="text"/> the token refiner's program instead: the text rows' condition
        /// projection and the refiner blocks, whose result the step program takes as its input (<see cref="RefinedText"/>);
        /// both programs carry WideTokenProducts, so the refiner's attention takes the routes it took inside the step.
        /// </summary>
        private void Emit(GraphEmitter e, StepValues s, bool text)
        {
            MiniMaxH3Config c = _w.Config;
            GraphProgramBuilder b = e.Builder;
            int hidden = c.HiddenSize, heads = c.NumAttentionHeads, hd = c.AttentionHeadDim, inner = c.AttentionInnerDim;
            int rot = c.RopeRotaryDim, ntok = s.Layout.TokenCount, nts = s.Layout.Timesteps.Length;
            float eps = c.NormEps;
            float scale = 1f / MathF.Sqrt(hd);
            var quantized = new HashSet<int>();
            var leaves = new Dictionary<MiniMaxH3DitWeight, int>();

            int Leaf(MiniMaxH3DitWeight w)
            {
                if (w == null) return -1;
                if (leaves.TryGetValue(w, out int node)) return node;
                ReadOnlySpan<long> ne = w.Vector ? stackalloc long[] { w.Ne0 } : stackalloc long[] { w.Ne0, w.Ne1 };
                node = _tensorAllocator == null
                    ? e.HostWeight(w.Format, ne, w.Data, w.Bytes)
                    : e.HostWeight(w.Format, ne, GraphIdentity.Of(w), w.Bytes, Handle(w));
                leaves[w] = node;
                if (w.Quantized) quantized.Add(node);
                return node;
            }
            int Scale(int x, float f) => b.Op(GraphOp.Scale, stackalloc[] { x }, default, stackalloc[] { f });
            // h3_mm: a block-quantized weight's product takes the q8_1 guard on the GGML executor (on ggml_cuda's wide routes,
            // which need none, the executor drops the pair where cuBLAS takes the product: GraphProgram.WideTokenProducts). The
            // tensor executors' products never round activations to q8_1, and the retired direct DiT took them without it.
            int Product(GraphProgramBuilder g, int w, int x) => quantized.Contains(w) && _tensorAllocator == null
                ? Scale(g.Op(GraphOp.MulMat, w, Scale(x, 1f / GuardScale)), GuardScale)
                : g.Op(GraphOp.MulMat, w, x);
            int Linear(MiniMaxH3DitLinear l, int x, bool guard = true)
            {
                int w = Leaf(l.Weight);
                int o = guard ? Product(b, w, x) : b.Op(GraphOp.MulMat, w, x);
                return l.Bias != null ? b.Op(GraphOp.Add, o, Leaf(l.Bias)) : o;
            }
            // A block projection with the LoRA set's unmerged updates (a LoraLinear region: w x plus up (down x) on each part's
            // slot, the guard around w x where Linear takes it), then its bias; exactly Linear where the set has none for it.
            MiniMaxH3LoraSet lora = s.Lora;
            int Factor(MiniMaxH3LoraSet.Part p, bool up)
            {
                IntPtr data = up ? p.Up : p.Down;
                long bytes = up ? p.UpBytes : p.DownBytes;
                ReadOnlySpan<long> ne = up ? stackalloc long[] { p.Rank, p.Out } : stackalloc long[] { p.In, p.Rank };
                if (_tensorAllocator == null) return e.HostWeight(p.Format, ne, data, bytes);
                if (!_loraHandles.TryGetValue(data, out MiniMaxH3DitWeight handle))
                    _loraHandles[data] = handle = new MiniMaxH3DitWeight(p.Format, ne[0], ne[1], data, bytes);
                return e.HostWeight(p.Format, ne, GraphIdentity.Of(handle), bytes, handle);
            }
            int Adapted(MiniMaxH3DitLinear l, int x, long rows, bool refiner, int block, MiniMaxH3LoraSet.Projection projection)
            {
                MiniMaxH3LoraSet.Factors f = lora?.Get(refiner, block, projection);
                if (f == null) return Linear(l, x);
                var parts = new GraphDitRegions.LoraPart[f.Parts.Length];
                for (int i = 0; i < parts.Length; i++)
                    parts[i] = new GraphDitRegions.LoraPart(f.Parts[i].Slot, Factor(f.Parts[i], up: false), Factor(f.Parts[i], up: true));
                int y = GraphDitRegions.LoraLinear(b, x, Leaf(l.Weight), parts, l.Weight.Ne1, f.SlotRows, rows, Product);
                return l.Bias != null ? b.Op(GraphOp.Add, y, Leaf(l.Bias)) : y;
            }
            bool HasLora(bool refiner, int block, MiniMaxH3LoraSet.Projection projection) =>
                lora?.Get(refiner, block, projection) != null;
            int Norm(int x, MiniMaxH3DitWeight w) => b.Op(GraphOp.Mul, b.Op(GraphOp.RmsNorm, stackalloc[] { x }, default, stackalloc[] { eps }), Leaf(w));
            // q, k and v of a stacked [q | k | v] product: [hd, heads, rows] views, rows a product row apart.
            int Third(int qkv, int i, int rows) => b.Op(GraphOp.View, stackalloc[] { qkv },
                stackalloc long[] { 3, hd, heads, rows, 1, hd * 4L, 3L * inner * 4, 0, (long)i * inner * 4 });
            int Rows(int x, long width, int start, int count) => b.Op(GraphOp.View, stackalloc[] { x },
                stackalloc long[] { 2, width, count, 1, 1, width * 4, 0, 0, (long)start * width * 4 });
            int Join(int x, int y) => x < 0 ? y : b.Op(GraphOp.Concat, stackalloc[] { x, y }, stackalloc long[] { 1 });

            e.MarkWideTokenProducts();
            // A plan binds one LoRA set's factors: another set compiles its own even where it reuses the addresses.
            if (lora != null) e.Mix(lora.Generation);

            // ---- text: project, then refine (no modulation, no RoPE; its attention is the retired graph's h3_attend)
            if (text)
            {
                int ctx = Linear(_w.ConditionProj, e.InputF32(s.Text, c.TextDim, s.TextCount));
                for (int ri = 0; ri < _w.Refiner.Length; ri++)
                {
                    MiniMaxH3DitBlock r = _w.Refiner[ri];
                    int n1 = Norm(ctx, r.Norm1);
                    int qkv = Adapted(r.Qkv, n1, s.TextCount, true, ri, MiniMaxH3LoraSet.Projection.Qkv);
                    int q = Third(qkv, 0, s.TextCount), k = Third(qkv, 1, s.TextCount), v = Third(qkv, 2, s.TextCount);
                    q = r.QNorm != null ? Norm(q, r.QNorm) : b.Op(GraphOp.Cont, q);
                    k = r.KNorm != null ? Norm(k, r.KNorm) : b.Op(GraphOp.Cont, k);
                    int merged = GraphDitRegions.Attention(b, q, k, v, hd, heads, s.TextCount, scale, GraphDitRegions.FlashVScale(s.TextCount));
                    ctx = b.Op(GraphOp.Add, ctx, Adapted(r.Out, merged, s.TextCount, true, ri, MiniMaxH3LoraSet.Projection.Out));
                    int n2 = Norm(ctx, r.Norm2);
                    int gv = Adapted(r.Fc1, n2, s.TextCount, true, ri, MiniMaxH3LoraSet.Projection.Fc1);
                    ctx = b.Op(GraphOp.Add, ctx, Adapted(r.Fc2, b.Op(GraphOp.Swiglu, gv), s.TextCount, true, ri,
                        MiniMaxH3LoraSet.Projection.Fc2));
                }
                if (_w.RefinerFinalNorm != null) ctx = Norm(ctx, _w.RefinerFinalNorm);
                e.Output(ctx, (long)hidden * s.TextCount);
                return;
            }

            // ---- inputs
            int vin = e.InputF32(s.Video, c.VideoPatchDim, s.ConditionCount + s.VideoCount);
            int ain = e.InputF32(s.Audio, c.AudioLatentChannels, s.AudioCount);
            int acin = s.ConditionAudioCount > 0 ? e.InputF32(s.ConditionAudio, c.AudioLatentChannels, s.ConditionAudioCount) : -1;
            int temb = e.InputF32(s.TimeEmbed, c.TimeEmbedDim, nts);
            // A projection of the time embedding viewed as [hidden, columns]: the modulation matrix.
            int Modulations(MiniMaxH3DitLinear l, long columns) => b.Op(GraphOp.Reshape, stackalloc[] { Linear(l, temb, guard: false) },
                stackalloc long[] { hidden, columns });
            int cosT = e.InputF32(s.Cos, rot, ntok);
            // The signed sine the rotary regions read, made on the host (negation is exact: GraphDitRegions.SignedSine's values),
            // so an executor serving them sees the tables' values.
            int sinS = e.InputF32(s.SinSigned, rot, ntok);
            // The token refiner's rows (its own program, RefinedText): the packed sequence's first rows.
            int x = s.TextCount > 0 ? e.InputF32(s.Refined, hidden, s.TextCount) : -1;
            GraphDitRegions.CheckSegments(s.Segments, ntok, 18L * nts, 5);
            int table = e.InputI32(GraphDitRegions.SegmentTable(s.Segments), 3, s.Segments.Length);

            // ---- patchify video (conditions and target with one projection) and audio, then pack
            // [text | conditions | audio | video]; a chunk list rebuilds the conditioning run in its order.
            int allVideo = Linear(_w.VideoPatch, vin);
            int audioTok = Linear(_w.AudioPatch, ain);
            int condAudioTok = acin >= 0 ? Linear(_w.AudioPatch, acin) : -1;
            int condTok = -1, videoTok = allVideo;
            if (s.ConditionCount > 0)
            {
                condTok = Rows(allVideo, hidden, 0, s.ConditionCount);
                videoTok = Rows(allVideo, hidden, s.ConditionCount, s.VideoCount);
            }
            if (s.Chunks is { Count: > 0 })
            {
                int rebuilt = -1, videoRow = 0, audioRow = 0;
                foreach (var (kind, count) in s.Chunks)
                {
                    if (count <= 0) continue;
                    int part;
                    if (kind == 0) { part = Rows(condTok, hidden, videoRow, count); videoRow += count; }
                    else { part = Rows(condAudioTok, hidden, audioRow, count); audioRow += count; }
                    rebuilt = Join(rebuilt, part);
                }
                condTok = rebuilt;
            }
            if (condTok >= 0) x = Join(x, condTok);
            x = Join(x, audioTok);
            x = Join(x, videoTok);

            // ---- the blocks
            float vScale = GraphDitRegions.FlashVScale(ntok);
            for (int bi = 0; bi < s.Blocks; bi++)
            {
                MiniMaxH3DitBlock blk = _w.Blocks[bi];
                // The curve-table variant feeds the interpolated embedding straight in (no SiLU), and takes no guard.
                int mods = Modulations(blk.AdaLn, 18L * nts);
                int h = GraphDitRegions.AdaLnModulate(b, x, Leaf(blk.Norm1), mods, table, s.Segments, hidden, 0, 1, eps);
                int qkv = Adapted(blk.Qkv, h, ntok, false, bi, MiniMaxH3LoraSet.Projection.Qkv);
                int merged = GraphDitRegions.TableRopeQkNormAttention(b, Third(qkv, 0, ntok), Third(qkv, 1, ntok), Third(qkv, 2, ntok),
                    Leaf(blk.QNorm), Leaf(blk.KNorm), cosT, sinS, hd, heads, ntok, rot, eps, scale, vScale);
                int attn = Adapted(blk.Out, merged, ntok, false, bi, MiniMaxH3LoraSet.Projection.Out);
                x = GraphDitRegions.GatedResidual(b, x, attn, -1, -1, mods, table, s.Segments, hidden, 2);
                h = GraphDitRegions.AdaLnModulate(b, x, Leaf(blk.Norm2), mods, table, s.Segments, hidden, 3, 4, eps);
                // With an update on fc1 or fc2 the MLP is its two projections around the SwiGLU (a SwigluFfn region holds only
                // its weights' products); without, the region the retired graph fused.
                int m = HasLora(false, bi, MiniMaxH3LoraSet.Projection.Fc1) || HasLora(false, bi, MiniMaxH3LoraSet.Projection.Fc2)
                    ? Adapted(blk.Fc2, b.Op(GraphOp.Swiglu, Adapted(blk.Fc1, h, ntok, false, bi, MiniMaxH3LoraSet.Projection.Fc1)),
                        ntok, false, bi, MiniMaxH3LoraSet.Projection.Fc2)
                    : GraphDitRegions.SwigluFfn(b, h, Leaf(blk.Fc1.Weight), Leaf(blk.Fc2.Weight), -1, -1, Product);
                x = GraphDitRegions.GatedResidual(b, x, m, -1, -1, mods, table, s.Segments, hidden, 5);
            }

            // ---- final layer: shift and scale by TIMESTEP row (timestep * 2), then each stream's head
            int fmods = Modulations(_w.FinalAdaLn, 2L * nts);
            int FinalSlice(MiniMaxH3ModulationSpan span)
            {
                int part = Norm(Rows(x, hidden, span.Start, span.Length), _w.FinalNorm);
                int col = span.Row * 2;
                int sh = Rows(fmods, hidden, col, 1);
                int sc = Rows(fmods, hidden, col + 1, 1);
                return b.Op(GraphOp.Add, b.Op(GraphOp.Add, part, b.Op(GraphOp.Mul, part, sc)), sh);
            }
            int videoOut = Linear(_w.FinalVideoOut, FinalSlice(s.Layout.VideoSpan));
            int audioOut = Linear(_w.FinalAudioOut, FinalSlice(s.Layout.AudioSpan));
            e.Output(Scale(videoOut, -1f), (long)c.VideoPatchDim * s.VideoCount);
            e.Output(audioOut, (long)c.AudioLatentChannels * s.AudioCount);
        }

        /// <summary>A tensor executor's object for a weight leaf: an F32 vector as a tensor, a projection as itself.</summary>
        private object Handle(MiniMaxH3DitWeight w)
        {
            if (!w.Vector) return w;
            if (!_tensors.TryGetValue(w, out Tensor t))
            {
                var values = new float[w.Ne0];
                Marshal.Copy(w.Data, values, 0, values.Length);
                t = new Tensor(_tensorAllocator, DType.Float32, w.Ne0);
                t.SetElementsAsFloat(values);
                _tensors[w] = t;
            }
            return t;
        }

        // ---- residency

        /// <summary>On Metal the denoiser's weights (wired zero-copy wraps of the mapped file) and its step's compute buffer
        /// leave the working set after every denoise, as the retired pipeline released them unconditionally: at 1280x768x107
        /// the compute buffer alone was 23.9 GB, and kept beside the weights it left the 5.2 GB VAE's first command buffer
        /// out of memory. Elsewhere the residency plan decides.</summary>
        public bool ReleaseAfterPhase => Backend == BackendType.GgmlMetal;

        /// <summary>A clip's denoise ended: on the GGML GPU backends the step's compute buffer goes before the decode (the
        /// retired pipeline released the denoise loop's reuse gallocr there: several GB at 20K tokens, of no use to the VAE or
        /// the next clip's 17 GB text encoder); the next clip compiles its step again. On direct CUDA the plans go too and the
        /// step's intermediates the allocator pooled return to the driver (<see cref="TensorSharp.Cuda.CudaAllocator.TrimPool"/>,
        /// as the Wan pipeline hands its denoise over to the VAE): their shapes are of no use to the decode, beside which the
        /// retired pipeline kept them. The managed executor's kept intermediates go back as well. The weights stay.</summary>
        public void FinishRequest()
        {
            if (Backend is BackendType.GgmlCuda or BackendType.GgmlMetal or BackendType.GgmlVulkan || _tensorAllocator is TensorSharp.Cuda.CudaAllocator)
                ReleaseWorkspace();
            else lock (_gate) (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
            if (_tensorAllocator is TensorSharp.Cuda.CudaAllocator cuda)
                lock (_gate) cuda.TrimPool();
        }

        public void ReleaseWorkspace()
        {
            lock (_gate)
            {
                _plan?.Dispose();
                _plan = null;
                // The refined rows are host values: the next clip with this prompt and LoRA set takes them as they are.
                _textPlan?.Dispose();
                _textPlan = null;
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
            }
        }

        /// <summary>Drops the weights' device copies (and the plan, which binds them); the mapping stays, so the next step
        /// uploads them from memory again.</summary>
        public void ReleaseDevice()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _plan?.Dispose();
                _plan = null;
                _textPlan?.Dispose();
                _textPlan = null;
                foreach (MiniMaxH3DitWeight w in _w.All) ReleaseCopy(w.Data);
                // The LoRA factors' copies too; the set stays bound and the next step uploads them again.
                if (_lora != null && _tensorAllocator == null) _lora.InvalidateDeviceCopies();
                foreach (IntPtr factor in _loraHandles.Keys) ReleaseCopy(factor);
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
                _preloaded = false;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _plan?.Dispose();
                _plan = null;
                DropTextPlan();
                ReleaseLoraDevice();
                _lora = null;
                foreach (MiniMaxH3DitWeight w in _w.All) ReleaseCopy(w.Data);
                foreach (Tensor t in _tensors.Values) t.Dispose();
                _tensors.Clear();
                (_tensorAllocator as RecyclingCpuAllocator)?.Trim();
                if (_executor is IDisposable owned && _ownsAllocator) owned.Dispose();
                _w.Dispose();
            }
        }
    }
}
