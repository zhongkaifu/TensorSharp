// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// LTX 2.5's AV denoiser on the diffusion runtime: one step is one GraphProgram (the two streams' patchify projections, the
// nine-way AdaLN of each stream's timesteps, the 48 blocks' self, text and cross-stream attentions and feed-forwards, and the
// output heads), run by a tensor executor this denoiser owns (CUDA, MLX or the managed CPU executor). The program is the op
// sequence of the original Lightricks graph as the direct denoiser computed it (whole-inner Q/K RMS norms, per-head split
// RoPE, 2 * sigmoid head gates, frozen pre-cross-attention stream states), and the executor takes that denoiser's kernels: its
// products as DirectLinear ran them (TensorGraphExecutor.DirectProducts), its norms (GraphFusion.RowNorm), its per-head
// rotary tables (HeadRotary), head gates (HeadGate) and F32 attention (FullAttention). Under --layer-split N the blocks run
// as whole-block stages over several devices (DenoiserStages): each stage's program is its blocks on its own device, the
// video and audio residual streams handed from one stage to the next. Under --tp N every block is sharded over the devices of
// a tensor-parallel group (DenoiserRanks): each rank's program holds its heads of every attention and its slice of every
// feed-forward, with an all-reduce after each row-parallel projection and one for the whole-inner q/k norms' sums of squares.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Threading;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models.Direct;
using TensorSharp.Models.Graph;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.Ltx25;

/// <summary>Batch-one, unmasked AV denoiser inputs. Contexts are already
/// projected to the two checkpoint stream widths. No text encoder or connector
/// is called here. Latents are token-major, not VAE channel-major storage.</summary>
public sealed class Ltx25DenoiserInput
{
    public required float[] Video { get; init; }
    public required int VideoTokens { get; init; }
    public required float[] Audio { get; init; }
    public required int AudioTokens { get; init; }
    public required float[] VideoContext { get; init; }
    public required int VideoContextTokens { get; init; }
    public required float[] AudioContext { get; init; }
    public required int AudioContextTokens { get; init; }
    public required Ltx25PositionGrid VideoPositions { get; init; }
    public required Ltx25PositionGrid AudioPositions { get; init; }
    public required float VideoSigma { get; init; }
    public required float AudioSigma { get; init; }
    public required Ltx25FrequencyPrecision FrequencyPrecision { get; init; }
    public float[] VideoTimesteps { get; init; }
    public float[] AudioTimesteps { get; init; }
    public float[] KeyframeMask { get; init; }
    /// <summary>Sampling metadata: one value per video token, 0 freezes a clean
    /// image and 1 fully denoises it. The transformer receives sigma times this mask.</summary>
    public float[] VideoDenoiseMask { get; init; }
    public float[] CleanVideo { get; init; }
    /// <summary>Enable the separately loaded Ingredients IC-LoRA only for
    /// identity reference generation. Ordinary image/keyframe modes leave it off.</summary>
    public bool UseReferenceLora { get; init; }
    public CancellationToken CancellationToken { get; init; }
}

public sealed record Ltx25DenoiserOutput(float[] VideoVelocity, float[] AudioVelocity, int ExecutedBlocks);

/// <summary>
/// The LTX 2.5 AV transformer as an <see cref="IDenoiser"/>. Making one maps the GGUF; the first step makes the projections
/// resident (on CUDA reading the file ahead in the order the blocks run) and compiles the step's program, which every later
/// step of the same geometry replays with new inputs. The Ingredients IC-LoRA arrives as the step's adapter set
/// (<see cref="Ltx25LoraSet"/>): its unscaled B @ A updates are added to the adapted projections in exact F32.
/// <para>
/// Over several devices (<c>--layer-split N</c>) the 48 blocks are contiguous stages balanced by their projections' bytes
/// (<see cref="DenoiserStages"/>), each stage a program of its own on its device's executor (made as the one device's, with
/// direct products): the first also patchifies the latents and adds the keyframe embedding, the last also runs the output
/// heads, and each hands the video and audio residual streams on as F32 rows, the next stage's inputs. Every stage takes the
/// prompt contexts and the timestep features as its own inputs and computes the eight AdaLN MLPs itself, binds its own
/// blocks' projections and Ingredients factors on its device, and keeps its own copies of the rotary tables and the small
/// tensors. A pass's later steps rerun each stage's plan with only the new timestep features written, as on one device.
/// </para>
/// <para>
/// Over a tensor-parallel group (<c>--tp N</c>) the blocks are sharded Megatron-style (<see cref="DenoiserRanks"/>): every
/// attention by heads (the rank's rows of q, k, v and the head gates' logits with their bias slices, its input columns of
/// to_out), every feed-forward by width (rows of the up projection, columns of the down one), and the rest (patchify, the AdaLN
/// MLPs, the modulation tables, the output heads) whole on every rank, so every rank holds both residual streams whole between
/// the projections and the result is rank 0's. A row-parallel projection's partial products are summed by an all-reduce
/// before its bias and the gated residual; the q and k RMS norms span the whole inner width, so each rank's sums of squares
/// over its heads' columns are summed by one all-reduce an attention (q's rows, then k's) before it normalizes its columns
/// with the global mean and its slice of the norm weight. Each rank's program runs on its own device's executor (made as the
/// one device's), in lockstep from one all-reduce to the next, and a pass's later steps rerun the ranks' plans with only the
/// new timestep features written.
/// </para>
/// </summary>
public sealed class Ltx25Denoiser : IDenoiser
{
    /// <summary>The name of the audio stream in <see cref="DenoiseStepInputs.Streams"/> and <see cref="StepOutput.Streams"/>.</summary>
    public const string AudioStream = "audio";

    /// <summary>Plans kept, by signature: a clip's two passes (and an Ingredients pass) each compile their own.</summary>
    private const int CachedPlans = 4;

    private readonly Ltx25DenoiserWeights _w;
    private readonly IAllocator _allocator;
    private readonly TensorGraphExecutor _executor;   // the one device's, or stage 0's under a split
    // Whole-block stages over several devices (--layer-split N), or null for one device: each stage's executor (this
    // denoiser's, made as the one device's), each stage's live step, and the largest stage's projection bytes.
    private readonly DenoiserStages _stages;
    private readonly TensorGraphExecutor[] _stageExecutors;
    private readonly LiveStep[] _stageLive;
    private readonly long _stageWeightBytes;
    // Tensor parallelism (--tp N), or null: the ranks (each its executor, made as the one device's, joined to the group's
    // collective), each rank's shards of the block projections by name, the column copies the shards own, the largest rank's
    // projection bytes, the ranks' live step, the bound adapter set's factors as each rank binds them (and the copies those
    // own), and ranks 1.. copies of the keyframe embedding.
    private readonly DenoiserRanks _ranks;
    private readonly TensorGraphExecutor[] _rankExecutors;
    private readonly Dictionary<string, Ltx25Weight>[] _rankShards;
    private readonly List<IntPtr> _shardOwned = new();
    private readonly long _rankWeightBytes;
    private RankLive _rankLive;
    private readonly Dictionary<(string, int), (Ltx25Weight Down, Ltx25Weight Up)> _loraShards = new();
    private readonly List<IntPtr> _loraShardOwned = new();
    private Tensor[] _keyframeCopies;
    private readonly object _gate = new();
    private readonly List<(ulong, ulong, IGraphPlan)> _plans = new();
    private readonly HashSet<Ltx25Weight> _warmed = new();
    private Ltx25CudaFilePrefault _prefault;
    private RopeCache _rope;
    private (float[] Mask, Tensor Absolute) _keyframes;
    private Ltx25LoraSet _ownedLora;      // the component surface's adapter (Forward with UseReferenceLora)
    private Ltx25LoraSet _boundLora;      // the set whose factors the cached plans and device copies hold
    private bool _resident;
    private bool _uploading;
    private MappedFileReadAhead _uploadRead;
    private bool _disposed;

    public Ltx25Config Config => _w.Config;

    /// <summary>The checkpoint's file.</summary>
    public string Path => _w.Path;

    /// <summary>Steps that compiled their program (the rest replayed a cached plan); under a split, stage programs; under
    /// tensor parallelism, rank programs (a geometry compiles one a rank).</summary>
    internal int Compiles => _compiles + (_stages?.Compiles ?? 0) + (_ranks?.Compiles ?? 0);

    private int _compiles;

    /// <summary>The executor the steps run on (tests); stage 0's under a split, rank 0's under tensor parallelism.</summary>
    internal TensorGraphExecutor Executor => _executor;

    /// <summary>The whole-block stages, or null for one device (a test seam).</summary>
    internal DenoiserStages Stages => _stages;

    /// <summary>The tensor-parallel ranks, or null (a test seam).</summary>
    internal DenoiserRanks Ranks => _ranks;

    public bool ReferenceLoraLoaded => _ownedLora != null;
    public int ReferenceLoraProjections => _ownedLora?.ProjectionCount ?? 0;
    internal int CachedReferenceLoraProjections => _ownedLora?.LoadedProjectionCount ?? 0;

    /// <summary>The denoiser of <paramref name="checkpoint"/> on <paramref name="allocator"/>'s backend: CUDA, MLX or the CPU,
    /// each on a tensor executor of its own that takes the direct engines' kernels. <paramref name="referenceLoraPath"/>, when
    /// given, is an Ingredients adapter <see cref="Forward"/> applies to inputs that ask for it.</summary>
    /// <param name="devices">The devices of a whole-block split (<c>--layer-split N</c>, <see cref="DenoiserStages"/>), one a
    /// stage, the first <paramref name="allocator"/> itself (the latents, the keyframe embedding and the first blocks live
    /// there); null or one device runs every block on <paramref name="allocator"/>. CUDA devices, or managed CPU host
    /// "devices" (functional placement for tests).</param>
    /// <param name="tensorParallel">A device group the blocks are sharded over (<c>--tp N</c>, <see cref="DenoiserRanks"/>), its
    /// first device <paramref name="allocator"/> itself (the latents, the keyframe embedding and rank 0's program live there);
    /// null or one device runs every block whole. A CUDA tensor-parallel group, or a <see cref="SimulatedDeviceGroup"/>'s host
    /// devices (functional placement for tests). A degree the heads or the row-parallel projections' quant blocks do not divide
    /// is refused, and so is a split with it (tensor-parallel stages are not implemented).</param>
    public Ltx25Denoiser(string checkpoint, IAllocator allocator, string referenceLoraPath = null, IReadOnlyList<IAllocator> devices = null,
        ITensorParallelGroup tensorParallel = null)
    {
        _allocator = allocator ?? throw new ArgumentNullException(nameof(allocator));
        bool split = devices is { Count: > 1 };
        bool sharded = tensorParallel is { Degree: > 1 };
        if (split && sharded)
            throw new NotSupportedException("LTX 2.5 runs --layer-split or --tp, not both: tensor-parallel stages are not implemented.");
        if (split && !ReferenceEquals(devices[0], allocator))
            throw new ArgumentException("A whole-block split's first device is the denoiser's allocator.", nameof(devices));
        if (split && devices.Any(d => d is MlxAllocator))
            throw new NotSupportedException("LTX 2.5's whole-block stages run on CUDA devices; MLX exposes one device.");
        if (sharded && !ReferenceEquals(tensorParallel.GetAllocator(0), allocator))
            throw new ArgumentException("A tensor-parallel group's first device is the denoiser's allocator.", nameof(tensorParallel));
        if (sharded && allocator is MlxAllocator)
            throw new NotSupportedException("LTX 2.5's tensor parallelism runs on CUDA devices; MLX exposes one device.");
        _w = new Ltx25DenoiserWeights(checkpoint, allocator);
        var made = new List<TensorGraphExecutor>();
        try
        {
            if (sharded)
            {
                // The shards first (a degree that does not divide is refused before any device is touched), then each rank's
                // executor, made as the one device's (direct products, the wide DiT regions); this denoiser owns them.
                int n = tensorParallel.Degree;
                _rankShards = _w.Shard(n, _shardOwned);
                _ranks = new DenoiserRanks(tensorParallel, a =>
                {
                    TensorGraphExecutor x = CreateExecutor(a);
                    made.Add(x);
                    return x;
                });
                _rankExecutors = new TensorGraphExecutor[n];
                for (int r = 0; r < n; r++) _rankExecutors[r] = (TensorGraphExecutor)_ranks.Executor(r);
                long replicated = _w.Projections.Where(w => !_rankShards[0].ContainsKey(w.Name)).Sum(w => w.Bytes);
                foreach (Dictionary<string, Ltx25Weight> shards in _rankShards)
                    _rankWeightBytes = Math.Max(_rankWeightBytes, replicated + shards.Values.Sum(w => w.Bytes));
                _executor = _rankExecutors[0];
            }
            else if (split)
            {
                // Each stage's executor is made as the one device's (direct products, the wide DiT regions), so a stage runs
                // the one device's kernels; this denoiser owns them.
                _stages = new DenoiserStages(_w.BlockBytes(), devices, a =>
                {
                    TensorGraphExecutor x = CreateExecutor(a);
                    made.Add(x);
                    return x;
                });
                _stageExecutors = new TensorGraphExecutor[_stages.Count];
                for (int s = 0; s < _stageExecutors.Length; s++) _stageExecutors[s] = (TensorGraphExecutor)_stages.Executor(s);
                _stageLive = new LiveStep[_stages.Count];
                for (int s = 0; s < _stages.Count; s++) _stageWeightBytes = Math.Max(_stageWeightBytes, _w.BytesOf(_stages.Span(s)));
                _executor = _stageExecutors[0];
            }
            else _executor = CreateExecutor(allocator);
            if (referenceLoraPath != null) _ownedLora = Ltx25LoraSet.Open(referenceLoraPath, _w.File, _w.Config);
        }
        catch
        {
            foreach (TensorGraphExecutor x in made) (x as IDisposable)?.Dispose();
            if (made.Count == 0) (_executor as IDisposable)?.Dispose();
            FreeShards();
            _w.Dispose();
            throw;
        }
    }

    /// <summary>The executor a denoiser on <paramref name="allocator"/>'s backend runs its steps on, with direct products and
    /// a wide DiT step's in-place regions (the adapter's LoraLinear updates).</summary>
    internal static TensorGraphExecutor CreateExecutor(IAllocator allocator) => allocator switch
    {
        CudaAllocator cuda => new CudaGraphExecutor(cuda) { DirectProducts = true, ServesWideDitRegions = true },
        MlxAllocator mlx => new MlxGraphExecutor(mlx) { DirectProducts = true, ServesWideDitRegions = true },
        CpuAllocator => new ManagedGraphExecutor(allocator) { DirectProducts = true, ServesWideDitRegions = true },
        _ => throw new NotSupportedException($"LTX 2.5's denoiser runs on CUDA, MLX or the CPU, not {allocator.GetType().Name}."),
    };

    // ---- IDenoiser

    public DenoiserCaps Caps { get; } = new() { Streams = new[] { AudioStream } };

    /// <summary>The projections' device bytes: under a split, the largest stage's share (each device holds its stage's blocks,
    /// the AdaLN MLPs every stage computes, and the first's patchify or the last's output projections); under tensor
    /// parallelism a rank's share (its shards of the block projections and every replicated projection whole). The Ingredients
    /// factors count whole, as on one device (an upper bound for a rank, which holds a slice of one factor of each pair).</summary>
    public long WeightBytes => (_ranks != null ? _rankWeightBytes : _stages == null ? _w.Bytes : _stageWeightBytes) + (AdapterBytes?.Invoke() ?? 0);

    /// <summary>The device bytes of the Ingredients factors the next steps may bind, for residency.</summary>
    internal Func<long> AdapterBytes { get; set; }

    /// <summary>A step's activations for <paramref name="workload"/>'s target (its latent cells are the video tokens) and
    /// references (appended tokens): <see cref="WorkingBytesFor"/>.</summary>
    public long WorkingBytes(DiffusionWorkload workload)
    {
        long tokens = workload.Target.Count / Math.Max(1, workload.Target.Channels);
        if (workload.References != null)
            foreach (LatentShape r in workload.References) tokens += r.Count / Math.Max(1, r.Channels);
        return WorkingBytesFor(tokens);
    }

    /// <summary>
    /// A step's working memory at <paramref name="videoTokens"/> video tokens: the widest intermediates a block holds at once
    /// (the feed-forward's 4x-wide F32 rows and its F16 prefill copies, the q, k, v and attention rows of the video stream), the
    /// per-token AdaLN coefficients of a guided clip, the bounded attention score workspace and the F16 weight scratch of the
    /// direct products.
    /// </summary>
    public long WorkingBytesFor(long videoTokens)
    {
        long dim = _w.Config.VideoDim;
        long rows = Math.Max(1, videoTokens);
        long perToken = dim * sizeof(float) * (4 + 6) + dim * 4 * sizeof(ushort) + 9L * dim * sizeof(float) * 2;
        return rows * perToken + (256L << 20) + 4L * dim * dim * 4 * sizeof(ushort);
    }

    /// <summary>The direct denoiser released its device copies before every decode (the VAE's activations need the room), and
    /// re-uploaded them from the mapping at the next clip's first step: so does this one.</summary>
    public bool ReleaseAfterPhase => true;

    public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        cancellationToken.ThrowIfCancellationRequested();
        var input = inputs.Layout.FamilyLayout as Ltx25DenoiserInput
            ?? throw new ArgumentException("the layout was not packed by LTX 2.5's family", nameof(inputs));
        Ltx25DenoiserOutput output = Run(input, LoraOf(adapters), blockLimit: null, cancellationToken);
        return new StepOutput(output.VideoVelocity, new Dictionary<string, float[]> { [AudioStream] = output.AudioVelocity });
    }

    /// <summary>The Ingredients set a request's adapters carry (each adapter's payload is the one set the model loaded), or
    /// null.</summary>
    private static Ltx25LoraSet LoraOf(AdapterSet adapters)
    {
        if (adapters is null or { IsEmpty: true }) return null;
        var set = adapters.Adapters[0].Payload as Ltx25LoraSet
            ?? throw new ArgumentException($"LoRA {adapters.Adapters[0].Name} was not loaded by LTX 2.5's Ingredients loader.", nameof(adapters));
        foreach (DiffusionAdapter a in adapters.Adapters)
            if (!ReferenceEquals(a.Payload, set))
                throw new ArgumentException("A clip's LoRA adapters are one loaded set: the adapters carry different sets.", nameof(adapters));
        return set;
    }

    /// <summary>One step on the component surface (tests and the A/B tools): the input's own adapter request
    /// (<see cref="Ltx25DenoiserInput.UseReferenceLora"/> applies the adapter given at construction).</summary>
    /// <param name="blockLimit">A diagnostic partial-checkpoint limit. Output
    /// records the actual count. Omit it to execute the complete denoiser.</param>
    public Ltx25DenoiserOutput Forward(Ltx25DenoiserInput input, int? blockLimit = null)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.UseReferenceLora && _ownedLora == null)
            throw new InvalidOperationException("Ingredients reference generation requires the LTX 2.5 Ingredients IC-LoRA adapter.");
        return Run(input, input.UseReferenceLora ? _ownedLora : null, blockLimit, input.CancellationToken);
    }

    private Ltx25DenoiserOutput Run(Ltx25DenoiserInput input, Ltx25LoraSet lora, int? blockLimit, CancellationToken ct)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Validate(input);
            ct.ThrowIfCancellationRequested();
            int blocks = blockLimit ?? Config.Layers;
            if (blocks <= 0 || blocks > Config.Layers) throw new ArgumentOutOfRangeException(nameof(blockLimit));
            if (lora != null && !lora.Fits(Config))
                throw new ArgumentException("The Ingredients adapter was loaded for an LTX checkpoint of another shape.", nameof(lora));
            Bind(lora);
            MakeResident();
            bool ran = false;
            try
            {
                Ltx25DenoiserOutput output = RunStep(input, lora, blocks);
                ran = true;
                return output;
            }
            finally
            {
                EndUpload(ran && blocks == Config.Layers);
            }
        }
    }

    private Ltx25DenoiserOutput RunStep(Ltx25DenoiserInput input, Ltx25LoraSet lora, int blocks)
    {
        if (_ranks != null) return RunRanks(input, lora, blocks);
        if (_stages != null) return RunStages(input, lora, blocks);
        RopeCache rope = Rope(input);
        Tensor keyframes = Keyframes(input);
        float[][] features = StepFeatures(input);
        var span = new DenoiserSpan(0, blocks, IsFirst: true, IsLast: true);
        // A pass's later steps run the program its first step emitted: the latents, contexts, tables and adapter are the
        // same objects (the sampler steps its state in place), so only the timestep features are written again, into the
        // arrays the live trace binds. Anything else new (another pass, a request, a caller's fresh arrays) emits again.
        var key = new LiveKey(input.Video, input.Audio, input.VideoContext, input.AudioContext, rope, keyframes, lora, span,
            features.Select(f => f.Length).ToArray(), null, null);
        if (_live == null || !_live.Key.Equals(key))
        {
            _live?.Dispose();
            _live = null;
            var step = new StepValues(input, span, lora, rope[0], keyframes, features, _allocator, null, null);
            var trace = new GraphEmitter(_executor, record: false);
            try { Emit(trace, step); }
            catch { trace.Dispose(); throw; }
            IGraphPlan plan = PlanFor(trace.Signature);
            GraphEmitter runner = trace;
            if (plan == null)
            {
                trace.Dispose();
                var clock = Stopwatch.StartNew();
                var recorded = new GraphEmitter(_executor, record: true);
                try
                {
                    Emit(recorded, step);
                }
                catch
                {
                    recorded.Dispose();
                    throw;
                }
                runner = recorded;
                double emitted = clock.Elapsed.TotalSeconds;
                plan = recorded.Compile(replayable: true);
                Keep(recorded.Signature, plan);
                _compiles++;
                Console.WriteLine($"  [ltx25] step program for {input.VideoTokens} video + {input.AudioTokens} audio tokens: " +
                    $"emitted in {emitted:F2}s, compiled in {clock.Elapsed.TotalSeconds - emitted:F2}s");
            }
            _live = new LiveStep(key, runner, plan, features);
        }
        else
            for (int i = 0; i < features.Length; i++) Array.Copy(features[i], _live.Features[i], features[i].Length);
        if (!_live.Emitter.Run(_live.Plan))
        {
            string error = _live.Plan.LastError;
            _live.Dispose();
            _live = null;
            throw new InvalidOperationException("LTX denoiser step failed: " + error);
        }
        float[] video = (float[])_live.Plan.Outputs[0].Clone(), audio = (float[])_live.Plan.Outputs[1].Clone();
        foreach (float v in video) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        foreach (float v in audio) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        return new Ltx25DenoiserOutput(video, audio, blocks);
    }

    /// <summary>
    /// One step over the whole-block stages: each stage's program on its device, in block order, the video and audio residual
    /// streams the previous stage's plan wrote (its output arrays, which every run of that plan rewrites) bound as the next
    /// stage's inputs; the last stage's outputs are the velocities. A block limit (a partial-depth diagnostic) clips the
    /// spans: a stage between the first and the last with no block left is skipped, the streams passing it unchanged.
    /// </summary>
    private Ltx25DenoiserOutput RunStages(Ltx25DenoiserInput input, Ltx25LoraSet lora, int blocks)
    {
        RopeCache rope = Rope(input);
        Tensor keyframes = Keyframes(input);
        float[][] features = StepFeatures(input);
        float[][] streams = null;
        for (int st = 0; st < _stages.Count; st++)
        {
            DenoiserSpan whole = _stages.Span(st);
            var span = new DenoiserSpan(Math.Min(whole.First, blocks), Math.Min(whole.End, blocks), whole.IsFirst, whole.IsLast);
            if (!span.IsFirst && !span.IsLast && span.First == span.End) continue;
            streams = RunStage(st, span, input, lora, rope, keyframes, features, streams?[0], streams?[1]);
        }
        float[] video = (float[])streams[0].Clone(), audio = (float[])streams[1].Clone();
        foreach (float v in video) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        foreach (float v in audio) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        return new Ltx25DenoiserOutput(video, audio, blocks);
    }

    /// <summary>
    /// Stage <paramref name="st"/>'s part of a step, the one device's live step per stage: while the stage's inputs are the
    /// arrays its last step bound (the same latents, contexts, tables, adapter, and the previous stage's same plan outputs),
    /// only the new timestep features are written into the arrays its kept trace binds, and its plan runs again. Otherwise the
    /// stages' runner (<see cref="DenoiserStages.RunAll"/>) traces the program, compiles it where the stage's plan was compiled
    /// for another signature, and runs it; a trace of the same program, its inputs pinned, is kept for the later steps.
    /// Returns the stage's outputs (the two residual streams, or the last stage's velocities), valid until it runs again.
    /// </summary>
    private float[][] RunStage(int st, DenoiserSpan span, Ltx25DenoiserInput input, Ltx25LoraSet lora, RopeCache rope, Tensor keyframes,
        float[][] features, float[] carryVideo, float[] carryAudio)
    {
        Tensor stageKeyframes = span.IsFirst ? keyframes : null;
        var key = new LiveKey(input.Video, input.Audio, input.VideoContext, input.AudioContext, rope, stageKeyframes, lora, span,
            features.Select(f => f.Length).ToArray(), carryVideo, carryAudio);
        LiveStep live = _stageLive[st];
        string what = span.End > span.First ? $"blocks {span.First}-{span.End - 1}" : "no blocks";
        if (live != null && live.Key.Equals(key) && ReferenceEquals(live.Plan, _stages.Plan(st)))
        {
            for (int i = 0; i < features.Length; i++) Array.Copy(features[i], live.Features[i], features[i].Length);
            if (!_stages.OnDevice(st, () => live.Emitter.Run(live.Plan)))
            {
                string error = live.Plan.LastError;
                live.Dispose();
                _stageLive[st] = null;
                throw new InvalidOperationException($"LTX denoiser stage {st} ({what}) failed: " + error);
            }
            return live.Plan.Outputs;
        }
        live?.Dispose();
        _stageLive[st] = null;
        var step = new StepValues(input, span, lora, rope[st], stageKeyframes, features, _stages.TensorAllocator(st), carryVideo, carryAudio);
        var trace = new GraphEmitter(_stageExecutors[st], record: false);
        float[][] outputs;
        try
        {
            Emit(trace, step);
            int compiled = _stages.Compiles;
            var clock = Stopwatch.StartNew();
            outputs = _stages.RunAll(st, e => Emit(e, step), "LTX 2.5 denoiser");
            if (_stages.Compiles != compiled)
                Console.WriteLine($"  [ltx25] stage {st} ({what}) program for {input.VideoTokens} video + {input.AudioTokens} audio tokens: " +
                    $"emitted, compiled and run in {clock.Elapsed.TotalSeconds:F2}s");
        }
        catch
        {
            trace.Dispose();
            throw;
        }
        _stageLive[st] = new LiveStep(key, trace, _stages.Plan(st), features);
        return outputs;
    }

    /// <summary>
    /// One step over the tensor-parallel ranks, the one device's live step for all of them: while the step's inputs are the
    /// arrays the last step bound (the same latents, contexts, tables and adapter), only the new timestep features are written
    /// into the arrays the ranks' kept traces bind, and the ranks' plans run again in lockstep. Otherwise the ranks' runner
    /// (<see cref="DenoiserRanks.Run"/>) traces every rank's program, compiles each rank's plan where the combined signature is
    /// new, and runs them; a trace of each rank's program, its inputs pinned, is kept for the later steps. Every rank computes
    /// the whole streams and the output heads; the velocities are rank 0's.
    /// </summary>
    private Ltx25DenoiserOutput RunRanks(Ltx25DenoiserInput input, Ltx25LoraSet lora, int blocks)
    {
        RopeCache rope = Rope(input);
        Tensor keyframes = Keyframes(input);
        float[][] features = StepFeatures(input);
        var span = new DenoiserSpan(0, blocks, IsFirst: true, IsLast: true);
        var key = new LiveKey(input.Video, input.Audio, input.VideoContext, input.AudioContext, rope, keyframes, lora, span,
            features.Select(f => f.Length).ToArray(), null, null);
        float[][] outputs;
        if (_rankLive != null && _rankLive.Key.Equals(key) && _rankLive.Runs(_ranks.Plans))
        {
            for (int i = 0; i < features.Length; i++) Array.Copy(features[i], _rankLive.Features[i], features[i].Length);
            if (!TensorGraphExecutor.RunRanked(_rankLive.Plans, _rankLive.Inputs, out string error))
            {
                _rankLive.Dispose();
                _rankLive = null;
                throw new InvalidOperationException($"LTX denoiser tensor-parallel step over {_ranks.Count} devices failed: {error}");
            }
            outputs = _rankLive.Plans[0].Outputs;
        }
        else
        {
            _rankLive?.Dispose();
            _rankLive = null;
            int n = _ranks.Count;
            var steps = new StepValues[n];
            // A step without keyframes binds none on any rank (the copies of an earlier mask stay for its next pass).
            for (int r = 0; r < n; r++)
                steps[r] = new StepValues(input, span, lora, rope[r], r == 0 || keyframes == null ? keyframes : _keyframeCopies[r], features,
                    _ranks.TensorAllocator(r), null, null) { Rank = r, Ranks = n, Shards = _rankShards[r] };
            var traces = new GraphEmitter[n];
            try
            {
                for (int r = 0; r < n; r++)
                {
                    traces[r] = new GraphEmitter(_rankExecutors[r], record: false);
                    Emit(traces[r], steps[r]);
                }
                int compiled = _ranks.Compiles;
                var clock = Stopwatch.StartNew();
                outputs = _ranks.Run((e, r) => Emit(e, steps[r]), "LTX 2.5 denoiser");
                if (_ranks.Compiles != compiled)
                    Console.WriteLine($"  [ltx25] step programs of {n} tensor-parallel ranks for {input.VideoTokens} video + " +
                        $"{input.AudioTokens} audio tokens: emitted, compiled and run in {clock.Elapsed.TotalSeconds:F2}s");
            }
            catch
            {
                foreach (GraphEmitter t in traces) t?.Dispose();
                throw;
            }
            _rankLive = new RankLive(key, traces, _ranks.Plans.ToArray(), features);
        }
        float[] video = (float[])outputs[0].Clone(), audio = (float[])outputs[1].Clone();
        foreach (float v in video) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        foreach (float v in audio) if (!float.IsFinite(v)) throw new InvalidOperationException("LTX denoiser produced nonfinite velocity.");
        return new Ltx25DenoiserOutput(video, audio, blocks);
    }

    /// <summary>The ranks' last step: each rank's trace (its inputs pinned) and the inputs it binds, the plans it ran, and the
    /// timestep-feature arrays every rank's trace binds.</summary>
    private sealed class RankLive : IDisposable
    {
        public RankLive(LiveKey key, GraphEmitter[] traces, IGraphPlan[] plans, float[][] features)
        {
            Key = key; Traces = traces; Plans = plans; Features = features;
            Inputs = Array.ConvertAll(traces, t => t.RunInputs());
        }

        public LiveKey Key { get; }
        public GraphEmitter[] Traces { get; }
        public IGraphPlan[] Plans { get; }
        public GraphRunInputs[] Inputs { get; }
        public float[][] Features { get; }

        /// <summary>Whether <paramref name="current"/> (the ranks' cached plans) are the plans this step ran.</summary>
        public bool Runs(IReadOnlyList<IGraphPlan> current)
        {
            if (current == null || current.Count != Plans.Length) return false;
            for (int r = 0; r < Plans.Length; r++)
                if (!ReferenceEquals(current[r], Plans[r])) return false;
            return true;
        }

        public void Dispose()
        {
            foreach (GraphEmitter t in Traces) t.Dispose();
        }
    }

    /// <summary>What makes a step's program and bindings the previous step's: the very arrays and tables it binds and its shape
    /// (under a split, also the previous stage's output arrays the stage reads).</summary>
    private sealed record LiveKey(float[] Video, float[] Audio, float[] VideoContext, float[] AudioContext, RopeCache Rope, Tensor Keyframes,
        Ltx25LoraSet Lora, DenoiserSpan Span, int[] FeatureLengths, float[] CarryVideo, float[] CarryAudio)
    {
        public bool Equals(LiveKey other) => other != null && ReferenceEquals(Video, other.Video) && ReferenceEquals(Audio, other.Audio)
            && ReferenceEquals(VideoContext, other.VideoContext) && ReferenceEquals(AudioContext, other.AudioContext)
            && ReferenceEquals(Rope, other.Rope) && ReferenceEquals(Keyframes, other.Keyframes) && ReferenceEquals(Lora, other.Lora)
            && Span == other.Span && FeatureLengths.AsSpan().SequenceEqual(other.FeatureLengths)
            && ReferenceEquals(CarryVideo, other.CarryVideo) && ReferenceEquals(CarryAudio, other.CarryAudio);

        public override int GetHashCode() => System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(Video);
    }

    /// <summary>The last step's trace (its inputs pinned), plan and timestep-feature arrays.</summary>
    private sealed class LiveStep : IDisposable
    {
        public LiveStep(LiveKey key, GraphEmitter emitter, IGraphPlan plan, float[][] features)
        {
            Key = key; Emitter = emitter; Plan = plan; Features = features;
        }

        public LiveKey Key { get; }
        public GraphEmitter Emitter { get; }
        public IGraphPlan Plan { get; }
        public float[][] Features { get; }

        public void Dispose() => Emitter.Dispose();
    }

    private LiveStep _live;

    /// <summary>The step's six timestep-feature rows: each stream's main (and cross scale/shift) timesteps, its prompt sigma and the
    /// other stream's sigma for its cross gate, each through <see cref="TimestepFeatures"/> at its multiplier.</summary>
    private float[][] StepFeatures(Ltx25DenoiserInput input)
    {
        Ltx25Config c = Config;
        return new[]
        {
            TimestepFeatures(input.VideoTimesteps ?? new[] { input.VideoSigma }, c.TimestepScale),
            TimestepFeatures(new[] { input.VideoSigma }, c.TimestepScale),
            TimestepFeatures(new[] { input.AudioSigma }, c.CrossGateTimestepScale),
            TimestepFeatures(input.AudioTimesteps ?? new[] { input.AudioSigma }, c.TimestepScale),
            TimestepFeatures(new[] { input.AudioSigma }, c.TimestepScale),
            TimestepFeatures(new[] { input.VideoSigma }, c.CrossGateTimestepScale),
        };
    }

    private IGraphPlan PlanFor((ulong, ulong) signature)
    {
        for (int i = 0; i < _plans.Count; i++)
        {
            var (a, b, plan) = _plans[i];
            if ((a, b) != signature) continue;
            // Most recently used last.
            _plans.RemoveAt(i);
            _plans.Add((a, b, plan));
            return plan;
        }
        return null;
    }

    private void Keep((ulong, ulong) signature, IGraphPlan plan)
    {
        if (_plans.Count >= CachedPlans)
        {
            _plans[0].Item3.Dispose();
            _plans.RemoveAt(0);
        }
        _plans.Add((signature.Item1, signature.Item2, plan));
    }

    private void DropPlans()
    {
        DropLive();
        foreach (var (_, _, plan) in _plans) plan.Dispose();
        _plans.Clear();
        _stages?.DropPlans();
        _ranks?.DropPlans();
    }

    /// <summary>Unpin the last step's inputs (a request's state arrays) and forget its trace, every stage's under a split and
    /// every rank's under tensor parallelism; the plans stay.</summary>
    private void DropLive()
    {
        _live?.Dispose();
        _live = null;
        _rankLive?.Dispose();
        _rankLive = null;
        if (_stageLive == null) return;
        for (int s = 0; s < _stageLive.Length; s++)
        {
            _stageLive[s]?.Dispose();
            _stageLive[s] = null;
        }
    }

    /// <summary>The devices the steps run on, each with its executor: the one device, every stage's, or every rank's.</summary>
    private IEnumerable<(IAllocator Device, TensorGraphExecutor Executor)> Lanes()
    {
        if (_ranks != null)
        {
            for (int r = 0; r < _ranks.Count; r++) yield return (_ranks.TensorAllocator(r), _rankExecutors[r]);
            yield break;
        }
        if (_stages == null)
        {
            yield return (_allocator, _executor);
            yield break;
        }
        for (int s = 0; s < _stages.Count; s++) yield return (_stages.TensorAllocator(s), _stageExecutors[s]);
    }

    /// <summary>The direct kernels' scratch of every CUDA executor the steps run on goes back to its device.</summary>
    private void ReleaseDirectWorkspaces()
    {
        foreach (var (_, executor) in Lanes())
            if (executor is CudaGraphExecutor cuda) cuda.ReleaseDirectWorkspaces();
    }

    /// <summary>Bind <paramref name="lora"/> for the step: another set than the one bound before drops the old set's device
    /// copies (the plans that bound it differ in signature and age out); under tensor parallelism also the ranks' plans, which
    /// bind the old set's per-rank slices, freed with it.</summary>
    private void Bind(Ltx25LoraSet lora)
    {
        if (lora == null || ReferenceEquals(lora, _boundLora)) return;
        if (_ranks != null && _boundLora != null)
        {
            DropLive();
            _ranks.DropPlans();
        }
        if (_boundLora != null) ReleaseFactors(_boundLora);
        _boundLora = lora;
    }

    /// <summary>Forget the bound adapter set (its factors' device copies and the plans that bind them): the model calls it under
    /// its generation lock before it disposes a set it replaces.</summary>
    internal void ReleaseLora()
    {
        lock (_gate)
        {
            if (_disposed || _boundLora == null) return;
            DropPlans();
            ReleaseFactors(_boundLora);
            _boundLora = null;
        }
    }

    // ---- residency

    /// <summary>
    /// Make the projections resident for the step about to run. On CUDA each one uploads at its first product (the routes cache
    /// it by its host address), as the direct denoiser uploaded each at its first use, so a block's upload overlaps the device
    /// work queued for the blocks before it; just before, its file pages are warmed (<see cref="Ltx25CudaFilePrefault"/>)
    /// and the file's read-ahead moves to its block (the prelude and output tensors first, then one stage per block: the
    /// file stores tensors alphabetically). The step's end (<see cref="EndUpload"/>) stops the read-ahead. On MLX every
    /// projection is converted to its device form before the step. Under a split every stage's executor takes the hook: the
    /// stages run their blocks in block order, so the file is read ahead in the same order, each stage device uploading its
    /// own blocks (and the AdaLN MLPs every stage computes). Under tensor parallelism every rank's executor takes it too: the
    /// ranks run every block in lockstep, each uploading its rows of the column-parallel projections (views of the mapping) and
    /// the replicated ones; the row-parallel shards are host copies made at load, which the file is not read ahead for.
    /// </summary>
    private void MakeResident()
    {
        if (_resident || _uploading) return;
        if (_allocator is CudaAllocator)
        {
            var stages = new List<Ltx25Weight>[Config.Layers + 1];
            for (int i = 0; i < stages.Length; i++) stages[i] = new();
            foreach (Ltx25Weight w in _w.Projections)
            {
                if (_rankShards != null && Ltx25DenoiserWeights.ParallelOf(w.Name) == Ltx25Parallel.RowParallel) continue;
                int block = _w.BlockOf(w.Name);
                stages[block < 0 ? 0 : block + 1].Add(w);
            }
            foreach (List<Ltx25Weight> s in stages) s.Sort((a, b) => a.FileOffset.CompareTo(b.FileOffset));
            _uploadRead = _w.File.IsSplit ? null : MappedFileReadAhead.TryStart(_w.File.FilePaths[0],
                stages.Select(s => s.Select(w => (w.FileOffset, w.Bytes))),
                message => Console.WriteLine("  [ltx25] warning: denoiser " + message));
            _prefault ??= _w.File.IsSplit ? null : new Ltx25CudaFilePrefault(_w.File);
            Action<IGraphWeight> beforeProduct = weight =>
            {
                // The adapter's factors are host copies, not the checkpoint's pages.
                if (weight is not Ltx25Weight w || w.FileOffset < 0) return;
                int block = _w.BlockOf(w.Name);
                if (block >= 0) _uploadRead?.Enter(block + 1);
                if (_prefault != null && _warmed.Add(w)) _prefault.Warm(w.Data, w.FileOffset, w.Bytes);
            };
            foreach (var (_, executor) in Lanes()) executor.BeforeDirectProduct = beforeProduct;
            _uploading = true;
            return;
        }
        if (_allocator is MlxAllocator mlx)
            foreach (Ltx25Weight w in _w.Projections)
                if (MlxQuantizedOps.CanPreloadDirectFloatType(w.Format))
                    MlxQuantizedOps.PreloadDirectFloatWeight(mlx, w.Data, w.Data, w.Format, w.Ne0, w.Ne1, w.Bytes);
        _resident = true;
    }

    /// <summary>The end of a step that uploaded its projections: the hook goes, the read-ahead stops (its summary logged), and
    /// the projections count as resident once a complete step ran (a partial one uploaded only its blocks).</summary>
    private void EndUpload(bool complete)
    {
        if (!_uploading) return;
        foreach (var (_, executor) in Lanes()) executor.BeforeDirectProduct = null;
        _uploading = false;
        if (_uploadRead != null)
        {
            _uploadRead.Stop();
            Console.WriteLine($"  [ltx25] denoiser upload {_uploadRead.Describe()}");
            _uploadRead.Dispose();
            _uploadRead = null;
        }
        _resident = complete;
    }

    /// <summary>Drop the device copy of the host bytes at <paramref name="data"/>: the CUDA and MLX resident-weight caches key
    /// them by that address (a copy left there outlives the denoiser, and a file mapped later at that address would compute
    /// with the old values); the managed executor's F32 copies by the same address. Under a split, every stage device's copy,
    /// and under tensor parallelism every rank device's (a device that never uploaded it has none to drop).</summary>
    private void ReleaseCopy(IntPtr data)
    {
        if (data == IntPtr.Zero) return;
        foreach (var (device, executor) in Lanes())
            switch (device)
            {
                case CudaAllocator cuda: CudaQuantizedOps.ReleaseQuantizedWeight(cuda, data); break;
                case MlxAllocator mlx: MlxQuantizedOps.ReleaseQuantizedWeight(mlx, data); executor.Invalidate(data); break;
                default: executor.Invalidate(data); break;
            }
    }

    /// <summary>Drop the device copies of <paramref name="set"/>'s factors; under tensor parallelism also those of the ranks'
    /// slices of them, whose copies are freed (they are made again, from the set's factors, by the next step that binds it).</summary>
    private void ReleaseFactors(Ltx25LoraSet set)
    {
        foreach (IntPtr factor in set.FactorAddresses) ReleaseCopy(factor);
        foreach (var (down, up) in _loraShards.Values)
        {
            ReleaseCopy(down.Data);
            ReleaseCopy(up.Data);
        }
        _loraShards.Clear();
        foreach (IntPtr owned in _loraShardOwned) QuantizedWeight.FreeBuffer(owned);
        _loraShardOwned.Clear();
    }

    /// <summary>
    /// The factors of adapter <paramref name="set"/>'s update of projection <paramref name="name"/> as rank
    /// <paramref name="rank"/> of <paramref name="ranks"/> binds them, or null where the set does not update it: a
    /// column-parallel projection keeps the down factor whole and takes the rank's output rows of the up factor (a view of the
    /// set's host copy), so its update fills the rank's rows of the output; a row-parallel one takes the rank's input columns of
    /// the down factor (a copy) and keeps the up factor whole, so its update is the rank's partial of the whole update, summed
    /// with the product's partial by the projection's all-reduce. Exact F32 products either way. Made once a rank for the bound
    /// set (<see cref="ReleaseFactors"/> forgets them).
    /// </summary>
    private (Ltx25Weight Down, Ltx25Weight Up)? LoraShard(Ltx25LoraSet set, string name, int rank, int ranks)
    {
        if (_loraShards.TryGetValue((name, rank), out var cached)) return cached;
        if (set.Factors(name) is not { } f) return null;
        (Ltx25Weight Down, Ltx25Weight Up) shard;
        if (Ltx25DenoiserWeights.ParallelOf(name) == Ltx25Parallel.RowParallel)
        {
            DitShards.Slice down = DitShards.Columns(new DitShards.Slice(f.Down.Format, f.Down.Ne0, f.Down.Ne1, f.Down.Data, f.Down.Bytes),
                rank, ranks, _loraShardOwned);
            shard = (f.Down with { Ne0 = down.Ne0, Data = down.Data, Bytes = down.Bytes }, f.Up);
        }
        else
        {
            DitShards.Slice up = DitShards.Rows(new DitShards.Slice(f.Up.Format, f.Up.Ne0, f.Up.Ne1, f.Up.Data, f.Up.Bytes),
                1, rank, ranks, _loraShardOwned);
            shard = (f.Down, f.Up with { Ne1 = up.Ne1, Data = up.Data, Bytes = up.Bytes });
        }
        _loraShards[(name, rank)] = shard;
        return shard;
    }

    /// <summary>A clip's denoising ended: the direct kernels' scratch goes back to the device (every stage's) before the decode.</summary>
    public void FinishRequest()
    {
        lock (_gate)
        {
            if (_disposed) return;
            DropLive();
            ReleaseDirectWorkspaces();
        }
    }

    public void ReleaseWorkspace()
    {
        lock (_gate)
        {
            if (_disposed) return;
            DropPlans();
            ReleaseDirectWorkspaces();
        }
    }

    /// <summary>Drops the projections' and factors' device copies, the position tables and the plans (on every stage device
    /// under a split, every rank device under tensor parallelism, with the copies of the ranks' shards); the mapping, the
    /// shards' host bytes and the small tensors stay, so the next step uploads from memory again.</summary>
    public void ReleaseDevice()
    {
        lock (_gate)
        {
            if (_disposed) return;
            DropPlans();
            _rope?.Dispose();
            _rope = null;
            ForgetKeyframes();
            ReleaseDirectWorkspaces();
            foreach (Ltx25Weight w in _w.Projections) ReleaseCopy(w.Data);
            ReleaseShardCopies();
            if (_boundLora != null) ReleaseFactors(_boundLora);
            _resident = false;
        }
    }

    /// <summary>Drop every rank device's copies of the ranks' shards of the block projections.</summary>
    private void ReleaseShardCopies()
    {
        if (_rankShards == null) return;
        foreach (Dictionary<string, Ltx25Weight> shards in _rankShards)
            foreach (Ltx25Weight w in shards.Values) ReleaseCopy(w.Data);
    }

    /// <summary>Free the column copies the ranks' shards own (their device copies were released first).</summary>
    private void FreeShards()
    {
        foreach (IntPtr owned in _shardOwned) QuantizedWeight.FreeBuffer(owned);
        _shardOwned.Clear();
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            DropPlans();
            _rope?.Dispose();
            _rope = null;
            ForgetKeyframes();
            foreach (Ltx25Weight w in _w.Projections) ReleaseCopy(w.Data);
            ReleaseShardCopies();
            if (_boundLora != null) ReleaseFactors(_boundLora);
            _boundLora = null;
            _ownedLora?.Dispose();
            _ownedLora = null;
            _prefault?.Dispose();
            _stages?.Dispose();
            _ranks?.Dispose();
            foreach (var (_, executor) in Lanes()) (executor as IDisposable)?.Dispose();
            FreeShards();
            _w.Dispose();
            _disposed = true;
        }
    }

    // ---- per-request tables

    /// <summary>The rotary tables of a geometry (video, audio and the video's temporal positions at the audio width), on the
    /// executor's allocator, or under a split one copy on each stage's device (every block reads all three; another GPU's
    /// tensor is not a stage's memory), or under tensor parallelism each rank's heads of them on its device: they do not depend
    /// on sigma, so one request's steps share them. Each table is computed on the host once and copied to every device. The
    /// caller's coordinates are snapshotted, so edits to a reused grid rebuild them.</summary>
    private sealed class RopeCache : IDisposable
    {
        private readonly float[] _video, _audio;
        private readonly Ltx25FrequencyPrecision _precision;
        private readonly RopeTables[] _devices;

        /// <param name="ranks">1: every device gets the whole tables; the tensor-parallel degree: device d is rank d and gets its
        /// heads (the tables are per head, so a rank's slice is exactly the heads it attends).</param>
        public RopeCache(IReadOnlyList<IAllocator> allocators, Ltx25Config c, Ltx25DenoiserInput input, int ranks = 1)
        {
            _video = (float[])input.VideoPositions.Boundaries.Clone();
            _audio = (float[])input.AudioPositions.Boundaries.Clone();
            _precision = input.FrequencyPrecision;
            _devices = new RopeTables[allocators.Count];
            for (int d = 0; d < _devices.Length; d++) _devices[d] = new RopeTables();
            try
            {
                Ltx25Rope video = Ltx25Geometry.BuildRope(input.VideoPositions, c.VideoDim, c.VideoHeads, _precision);
                for (int d = 0; d < _devices.Length; d++) _devices[d].Video = Device(allocators[d], video, d, ranks);
                video = null;
                Ltx25Rope audio = Ltx25Geometry.BuildRope(input.AudioPositions, c.AudioDim, c.AudioHeads, _precision);
                for (int d = 0; d < _devices.Length; d++) _devices[d].Audio = Device(allocators[d], audio, d, ranks);
                audio = null;
                Ltx25Rope cross = Ltx25Geometry.BuildRope(Ltx25Geometry.TemporalPositions(input.VideoPositions),
                    c.AudioDim, c.AudioHeads, _precision);
                for (int d = 0; d < _devices.Length; d++) _devices[d].CrossVideo = Device(allocators[d], cross, d, ranks);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>The tables on device <paramref name="device"/> (0 for one device, else the stage or the rank).</summary>
        public RopeTables this[int device] => _devices[device];

        private static (Tensor, Tensor, int, int, int) Device(IAllocator allocator, Ltx25Rope rope, int rank, int ranks)
        {
            if (ranks > 1) return RankDevice(allocator, rope, rank, ranks);
            var cos = new Tensor(allocator, DType.Float32, rope.Tokens, rope.Heads, rope.HeadDim / 2);
            var sin = new Tensor(allocator, DType.Float32, rope.Tokens, rope.Heads, rope.HeadDim / 2);
            cos.SetElementsAsFloat(rope.Cosine);
            sin.SetElementsAsFloat(rope.Sine);
            return (cos, sin, rope.Tokens, rope.Heads, rope.HeadDim);
        }

        /// <summary>Rank <paramref name="rank"/>'s heads of <paramref name="rope"/>'s tables, [tokens, heads / ranks, head dim /
        /// 2], copied out of each token's row.</summary>
        private static (Tensor, Tensor, int, int, int) RankDevice(IAllocator allocator, Ltx25Rope rope, int rank, int ranks)
        {
            int local = rope.Heads / ranks, half = rope.HeadDim / 2, run = local * half;
            var cosine = new float[(long)rope.Tokens * run];
            var sine = new float[cosine.LongLength];
            for (long t = 0; t < rope.Tokens; t++)
            {
                long from = (t * rope.Heads + (long)rank * local) * half;
                Array.Copy(rope.Cosine, from, cosine, t * run, run);
                Array.Copy(rope.Sine, from, sine, t * run, run);
            }
            var cos = new Tensor(allocator, DType.Float32, rope.Tokens, local, half);
            var sin = new Tensor(allocator, DType.Float32, rope.Tokens, local, half);
            cos.SetElementsAsFloat(cosine);
            sin.SetElementsAsFloat(sine);
            return (cos, sin, rope.Tokens, local, rope.HeadDim);
        }

        public bool Matches(Ltx25DenoiserInput input) => _precision == input.FrequencyPrecision &&
            _video.AsSpan().SequenceEqual(input.VideoPositions.Boundaries) && _audio.AsSpan().SequenceEqual(input.AudioPositions.Boundaries);

        public void Dispose()
        {
            foreach (RopeTables tables in _devices) tables?.Dispose();
        }
    }

    /// <summary>One device's rotary tables: [tokens, heads, head dim / 2] cosines and sines with their geometry.</summary>
    private sealed class RopeTables : IDisposable
    {
        public (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim) Video, Audio, CrossVideo;

        public void Dispose()
        {
            Video.Cos?.Dispose(); Video.Sin?.Dispose();
            Audio.Cos?.Dispose(); Audio.Sin?.Dispose();
            CrossVideo.Cos?.Dispose(); CrossVideo.Sin?.Dispose();
        }
    }

    private RopeCache Rope(Ltx25DenoiserInput input)
    {
        if (_rope?.Matches(input) == true) return _rope;
        // Release the old geometry's tables before making potentially large new ones.
        _rope?.Dispose();
        _rope = null;
        return _rope = new RopeCache(_stages == null && _ranks == null ? new[] { _allocator } : Lanes().Select(l => l.Device).ToArray(),
            Config, input, _ranks?.Count ?? 1);
    }

    /// <summary>The keyframe embedding the marked tokens add ([tokens, dim]: 0 or 1 times the embedding, the products the
    /// direct denoiser took), made once a mask: the same for every step of a pass. Under tensor parallelism every rank adds it
    /// to its whole video stream, so ranks 1.. get copies on their devices (<see cref="_keyframeCopies"/>).</summary>
    private Tensor Keyframes(Ltx25DenoiserInput input)
    {
        if (input.KeyframeMask == null) return null;
        if (!Config.KeyframeEmbedding) throw new ArgumentException("Checkpoint has no keyframe embedding.");
        if (_keyframes.Mask != null && _keyframes.Mask.AsSpan().SequenceEqual(input.KeyframeMask)) return _keyframes.Absolute;
        ForgetKeyframes();
        float[] embedding = DirectOps.DequantTensor(_w.File, Config.Prefix + "keyframes_abs_pos_embedding");
        int dim = Config.VideoDim;
        var values = new float[(long)input.VideoTokens * dim];
        for (int t = 0; t < input.VideoTokens; t++)
        {
            float marked = input.KeyframeMask[t] > 0 ? 1f : 0f;
            for (int c = 0; c < dim; c++) values[(long)t * dim + c] = marked * embedding[c];
        }
        var absolute = new Tensor(_allocator, DType.Float32, input.VideoTokens, dim);
        absolute.SetElementsAsFloat(values);
        _keyframes = ((float[])input.KeyframeMask.Clone(), absolute);
        if (_ranks != null)
        {
            _keyframeCopies = new Tensor[_ranks.Count];
            for (int r = 1; r < _keyframeCopies.Length; r++)
            {
                _keyframeCopies[r] = new Tensor(_ranks.TensorAllocator(r), DType.Float32, input.VideoTokens, dim);
                _keyframeCopies[r].SetElementsAsFloat(values);
            }
        }
        return absolute;
    }

    /// <summary>Dispose the keyframe embedding (and the ranks' copies of it).</summary>
    private void ForgetKeyframes()
    {
        _keyframes.Absolute?.Dispose();
        _keyframes = default;
        if (_keyframeCopies == null) return;
        foreach (Tensor t in _keyframeCopies) t?.Dispose();
        _keyframeCopies = null;
    }

    // ---- emission

    private sealed class StepValues
    {
        public StepValues(Ltx25DenoiserInput input, DenoiserSpan span, Ltx25LoraSet lora, RopeTables rope, Tensor keyframes,
            float[][] features, IAllocator device, float[] carryVideo, float[] carryAudio)
        {
            Input = input; Span = span; Lora = lora; Rope = rope; Keyframes = keyframes; Features = features; Device = device;
            CarryVideo = carryVideo; CarryAudio = carryAudio;
        }

        /// <summary><see cref="StepFeatures"/>' six rows, bound as the program's inputs.</summary>
        public float[][] Features { get; }

        public Ltx25DenoiserInput Input { get; }

        /// <summary>The blocks the program runs, and whether it opens the step (patchify) and closes it (the output heads): every
        /// block, both, on one device.</summary>
        public DenoiserSpan Span { get; }

        public Ltx25LoraSet Lora { get; }

        /// <summary>The rotary tables on the program's device.</summary>
        public RopeTables Rope { get; }

        public Tensor Keyframes { get; }

        /// <summary>Where the program's small tensors live (the executor's allocator, or its stage's device).</summary>
        public IAllocator Device { get; }

        /// <summary>The video and audio residual streams the previous stage handed on ([dim, tokens] F32 rows), where the
        /// program does not open the step.</summary>
        public float[] CarryVideo { get; }
        public float[] CarryAudio { get; }

        /// <summary>The program's tensor-parallel rank of <see cref="Ranks"/> (0 of 1: every block whole).</summary>
        public int Rank { get; init; }
        public int Ranks { get; init; } = 1;

        /// <summary>The rank's shards of the block projections, by name (<see cref="Ltx25DenoiserWeights.Shard"/>), where
        /// <see cref="Ranks"/> is more than one.</summary>
        public IReadOnlyDictionary<string, Ltx25Weight> Shards { get; init; }
    }

    /// <summary>Original cosine-first 256-feature timestep encoding, before
    /// the learned SiLU MLP. Inputs are sigma, scaled once by the given factor.</summary>
    public static float[] TimestepFeatures(ReadOnlySpan<float> timesteps, float multiplier = Ltx25Config.TimestepMultiplier)
    {
        if (timesteps.Length == 0 || !float.IsFinite(multiplier)) throw new ArgumentException("Invalid LTX timesteps.");
        const int half = Ltx25Config.TimestepDimension / 2;
        var output = new float[checked(timesteps.Length * Ltx25Config.TimestepDimension)];
        for (int row = 0; row < timesteps.Length; ++row)
        {
            float time = timesteps[row] * multiplier;
            if (!float.IsFinite(time)) throw new ArgumentException("LTX timestep must be finite.");
            for (int index = 0; index < half; ++index)
            {
                float phase = time * MathF.Exp(-MathF.Log(10000f) * index / half);
                (output[row * 256 + half + index], output[row * 256 + index]) = MathF.SinCos(phase);
            }
        }
        return output;
    }

    /// <summary>
    /// One step's program in the direct denoiser's op order: patchify (and the keyframe embedding), each stream's four AdaLN
    /// MLPs (main, prompt, cross-stream scale/shift and gate), then every block's video and audio self and text attention, the
    /// cross-stream attentions over both streams' norms frozen before either residual moves, and the feed-forwards, then each
    /// stream's layer-normed, modulated output head. The latents, contexts and timestep features are inputs; the rotary
    /// tables and the keyframe embedding are fixed for a request (tensors), so a geometry's steps share one plan.
    /// <para>
    /// A whole-block stage's program is the part of that program its span holds (<see cref="StepValues.Span"/>): where it does
    /// not open the step, the two residual streams the previous stage handed on are its first inputs in place of the patchify
    /// projections; only its blocks run; where it does not close the step, the two streams are its outputs in place of the
    /// heads. The contexts and timestep features are every stage's inputs and every stage computes the eight AdaLN MLPs from
    /// them: exact (the same products of the same inputs), and cheaper than handing on their coefficients, which on a guided
    /// pass are rows of 13 coefficients a video token (about 2.3 GB at 10,752 tokens, through host memory at every stage
    /// boundary each step), against about one block's worth of products to recompute (nothing measurable on an unguided pass,
    /// whose coefficients are one row).
    /// </para>
    /// <para>
    /// A tensor-parallel rank's program (<see cref="StepValues.Rank"/>) is the whole step with the rank's shards of the block
    /// projections in place of them: each attention over the rank's heads (<c>ShardAttention</c>), each sharded projection
    /// through <c>ShardLinear</c> (an all-reduce after every row-parallel one), the rest the one device's ops. A program of one
    /// rank emits exactly the one device's ops.
    /// </para>
    /// </summary>
    private void Emit(GraphEmitter e, StepValues s)
    {
        Ltx25Config c = Config;
        GraphProgramBuilder b = e.Builder;
        Ltx25DenoiserInput input = s.Input;
        DenoiserSpan span = s.Span;
        float eps = Ltx25Config.NormEpsilon;
        var leaves = new Dictionary<object, int>();
        // A tensor-parallel rank's program (StepValues.Rank): its shards of the block projections, the rest whole.
        bool tensorParallel = s.Ranks > 1;
        e.MarkWideTokenProducts();
        // A plan binds one adapter set's factors.
        if (s.Lora != null) e.Mix(s.Lora.Generation);

        int Projection(Ltx25Weight w)
        {
            if (leaves.TryGetValue(w, out int node)) return node;
            node = e.HostWeight(w.Format, stackalloc long[] { w.Ne0, w.Ne1 }, GraphIdentity.Of(w), w.Bytes, w);
            leaves[w] = node;
            return node;
        }
        // A small tensor as a leaf in its ggml extents.
        int Vector(string name)
        {
            if (leaves.TryGetValue(name, out int node)) return node;
            Tensor t = _w.Vector(name, s.Device);
            node = e.TensorWeight(t, _w.Extents(name));
            leaves[name] = node;
            return node;
        }
        int Constant(Tensor t, params long[] ne)
        {
            if (leaves.TryGetValue(t, out int node)) return node;
            node = e.TensorWeight(t, ne);
            leaves[t] = node;
            return node;
        }
        // The direct denoiser's Project: w x, its bias, then the adapter's B (A x) where the adapter updates the projection,
        // added into the projection's output (a LoraLinear region the executors serve in place: an Ingredients step at 10,752
        // tokens held 2.4 GB more as separate ops).
        int Linear(string name, int x)
        {
            if (tensorParallel && s.Shards.TryGetValue(name, out Ltx25Weight shard)) return ShardLinear(name, shard, x);
            Ltx25Weight w = _w.Projection(name);
            bool biased = _w.HasBias(name);
            if (s.Lora?.Factors(name) is { } f)
                return GraphDitRegions.LoraLinear(b, x, Projection(w), new[] { new GraphDitRegions.LoraPart(0, Projection(f.Down), Projection(f.Up)) },
                    w.Ne1, slotRows: 0, rows: 0 /* one slot: no row views */, bias: biased ? Vector(name + ".bias") : -1);
            int y = b.Op(GraphOp.MulMat, Projection(w), x);
            if (biased) y = b.Op(GraphOp.Add, y, Vector(name + ".bias"));
            return y;
        }
        // A rank's slice of a small vector (a column-parallel projection's bias, a whole-inner norm's weight): its heads' values.
        int Slice(string name)
        {
            string key = name + "#slice";
            if (leaves.TryGetValue(key, out int node)) return node;
            Tensor t = _w.VectorSlice(name, s.Rank, s.Ranks, s.Device);
            node = e.TensorWeight(t, t.ElementCount());
            leaves[key] = node;
            return node;
        }
        // A sharded projection on its rank (Ltx25Parallel): a column-parallel one is the rank's rows of the product plus its
        // slice of the bias; a row-parallel one is the rank's partial product over its input columns (with the adapter's
        // partial update), summed over the ranks by an all-reduce, then the bias, whole, once.
        int ShardLinear(string name, Ltx25Weight w, int x)
        {
            bool rowParallel = Ltx25DenoiserWeights.ParallelOf(name) == Ltx25Parallel.RowParallel;
            int bias = !_w.HasBias(name) ? -1 : rowParallel ? Vector(name + ".bias") : Slice(name + ".bias");
            int y;
            if (s.Lora != null && LoraShard(s.Lora, name, s.Rank, s.Ranks) is { } f)
                y = GraphDitRegions.LoraLinear(b, x, Projection(w), new[] { new GraphDitRegions.LoraPart(0, Projection(f.Down), Projection(f.Up)) },
                    w.Ne1, slotRows: 0, rows: 0 /* one slot: no row views */, bias: rowParallel ? -1 : bias);
            else
            {
                y = b.Op(GraphOp.MulMat, Projection(w), x);
                if (!rowParallel && bias >= 0) y = b.Op(GraphOp.Add, y, bias);
            }
            if (!rowParallel) return y;
            y = b.Op(GraphOp.AllReduce, y);
            return bias >= 0 ? b.Op(GraphOp.Add, y, bias) : y;
        }
        int Silu(int x) => b.Op(GraphOp.Silu, x);
        (int Coefficients, int Embedded) Ada(string name, int features)
        {
            int embedded = Linear(name + ".emb.timestep_embedder.linear_2", Silu(Linear(name + ".emb.timestep_embedder.linear_1", features)));
            return (Linear(name + ".linear", Silu(embedded)), embedded);
        }
        // A column of a [coefficients * dim, rows] modulation plus a row of a table [dim, table rows].
        int ModulationRow(int dynamic, int coefficients, int index, long rows, string table, int tableRow, long dim)
        {
            int column = b.Op(GraphOp.Cont, e.View2(dynamic, dim, rows, coefficients * dim * 4L, index * dim * 4L));
            int row = e.View2(Vector(table), dim, 1, dim * 4L, tableRow * dim * 4L);
            return b.Op(GraphOp.Add, column, row);
        }
        int Modulate(int x, int shift, int scale)
        {
            int scalePlusOne = b.Op(GraphOp.ScaleBias, stackalloc[] { scale }, default, stackalloc[] { 1f, 1f });
            return b.Op(GraphOp.Add, b.Op(GraphOp.Mul, x, scalePlusOne), shift);
        }
        int Residual(int target, int update, int gate) => b.Op(GraphOp.Add, target, b.Op(GraphOp.Mul, update, gate));
        int Norm(int x) => GraphDitRegions.RowNorm(b, x, -1, layer: false, eps);
        int Reshape3(int x, long d, long heads, long rows) => b.Op(GraphOp.Reshape, stackalloc[] { x }, stackalloc long[] { d, heads, rows });

        int Attention(string name, int query, long queryRows, int context, long contextRows, int heads, int headDim,
            (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? queryRope = null,
            (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? keyRope = null)
        {
            if (tensorParallel) return ShardAttention(name, query, queryRows, context, contextRows, heads, headDim, queryRope, keyRope);
            // q and k RMS-normed over the whole inner width, then (self and cross-stream attention) rotated per head, as the
            // direct denoiser normed and rotated them in place; the attention gated per head in place before to_out.
            int Prep(int raw, string norm, long rows, (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? rope) => rope is { } r
                ? GraphDitRegions.HeadRotary(b, raw, Constant(r.Cos, headDim / 2, heads, rows), Constant(r.Sin, headDim / 2, heads, rows),
                    headDim, heads, rows, Vector(norm), eps)
                : Reshape3(GraphDitRegions.RowNorm(b, raw, Vector(norm), layer: false, eps), headDim, heads, rows);
            int q3 = Prep(Linear(name + ".to_q", query), name + ".q_norm.weight", queryRows, queryRope);
            int k3 = Prep(Linear(name + ".to_k", context), name + ".k_norm.weight", contextRows, keyRope);
            int v = Linear(name + ".to_v", context);
            int logits = Linear(name + ".to_gate_logits", query);
            int gated = GraphDitRegions.FullAttention(b, q3, k3, Reshape3(v, headDim, heads, contextRows), headDim, heads, queryRows,
                1f / MathF.Sqrt(headDim), logits, 2f);
            return Linear(name + ".to_out.0", gated);
        }

        // The rank's heads of an attention (Attention's ops over heads / ranks heads): q, k, v and the gate logits are the
        // rank's rows (its heads' columns of the inner width), to_out its partial over them, summed by the projection's
        // all-reduce. The q and k RMS norms span the whole inner width, so each rank's sums of squares over its columns are
        // summed over the ranks first (q's rows then k's, one all-reduce), and the rank normalizes its columns with the whole
        // row's mean and its slice of the norm weight, then rotates its heads by its slice of the tables (HeadRotary without
        // a norm of its own).
        int ShardAttention(string name, int query, long queryRows, int context, long contextRows, int heads, int headDim,
            (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? queryRope,
            (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? keyRope)
        {
            int local = heads / s.Ranks;
            float meanScale = 1f / ((long)heads * headDim);
            int q = Linear(name + ".to_q", query), k = Linear(name + ".to_k", context);
            int SumSquares(int x) => b.Op(GraphOp.SumRows, b.Op(GraphOp.Sqr, x));
            int sums = b.Op(GraphOp.AllReduce, b.Op(GraphOp.Concat, stackalloc[] { SumSquares(q), SumSquares(k) }, stackalloc long[] { 1 }));
            int Prep(int raw, string norm, long rows, long first, (Tensor Cos, Tensor Sin, int Tokens, int Heads, int HeadDim)? rope)
            {
                int rowSums = e.View2(sums, 1, rows, sizeof(float), first * sizeof(float));
                int rms = b.Op(GraphOp.Sqrt, b.Op(GraphOp.ScaleBias, stackalloc[] { rowSums }, default, stackalloc[] { meanScale, eps }));
                int normed = b.Op(GraphOp.Mul, b.Op(GraphOp.Div, raw, rms), Slice(norm));
                return rope is { } r
                    ? GraphDitRegions.HeadRotary(b, normed, Constant(r.Cos, headDim / 2, local, rows), Constant(r.Sin, headDim / 2, local, rows),
                        headDim, local, rows)
                    : Reshape3(normed, headDim, local, rows);
            }
            int q3 = Prep(q, name + ".q_norm.weight", queryRows, 0, queryRope);
            int k3 = Prep(k, name + ".k_norm.weight", contextRows, queryRows, keyRope);
            int v = Linear(name + ".to_v", context);
            int logits = Linear(name + ".to_gate_logits", query);
            int gated = GraphDitRegions.FullAttention(b, q3, k3, Reshape3(v, headDim, local, contextRows), headDim, local, queryRows,
                1f / MathF.Sqrt(headDim), logits, 2f);
            return Linear(name + ".to_out.0", gated);
        }

        long nv = input.VideoTokens, na = input.AudioTokens, nvc = input.VideoContextTokens, nac = input.AudioContextTokens;
        long vdim = c.VideoDim, adim = c.AudioDim;
        float[] videoTimes = input.VideoTimesteps ?? new[] { input.VideoSigma }, audioTimes = input.AudioTimesteps ?? new[] { input.AudioSigma };
        long tv = videoTimes.Length, ta = audioTimes.Length;

        // ---- inputs: the latents through their patchify projections (and the keyframe embedding) where the program opens the
        // step, else the two residual streams the previous stage handed on; the contexts in every program
        int video, audio;
        if (span.IsFirst)
        {
            video = Linear("patchify_proj", e.InputF32(input.Video, c.VideoChannels, nv));
            audio = Linear("audio_patchify_proj", e.InputF32(input.Audio, c.AudioChannels, na));
            if (s.Keyframes != null) video = b.Op(GraphOp.Add, video, Constant(s.Keyframes, vdim, nv));
        }
        else
        {
            video = e.InputF32(s.CarryVideo ?? throw new ArgumentException("a later stage needs the video stream", nameof(s)), vdim, nv);
            audio = e.InputF32(s.CarryAudio ?? throw new ArgumentException("a later stage needs the audio stream", nameof(s)), adim, na);
        }
        int videoContext = e.InputF32(input.VideoContext, c.VideoContextDim, nvc);
        int audioContext = e.InputF32(input.AudioContext, c.AudioContextDim, nac);

        // ---- time: per stream its main (per-token where the pass has guides), prompt, cross scale/shift and cross gate AdaLN
        int Features(int i) => e.InputF32(s.Features[i], Ltx25Config.TimestepDimension, s.Features[i].Length / Ltx25Config.TimestepDimension);
        int videoFeatures = Features(0), audioFeatures = Features(3);
        var (videoMain, videoEmbedded) = Ada("adaln_single", videoFeatures);
        int videoPrompt = Ada("prompt_adaln_single", Features(1)).Coefficients;
        int videoCross = Ada("av_ca_video_scale_shift_adaln_single", videoFeatures).Coefficients;
        int videoCrossGate = Ada("av_ca_a2v_gate_adaln_single", Features(2)).Coefficients;
        var (audioMain, audioEmbedded) = Ada("audio_adaln_single", audioFeatures);
        int audioPrompt = Ada("audio_prompt_adaln_single", Features(4)).Coefficients;
        int audioCross = Ada("av_ca_audio_scale_shift_adaln_single", audioFeatures).Coefficients;
        int audioCrossGate = Ada("av_ca_v2a_gate_adaln_single", Features(5)).Coefficients;

        RopeTables rope = s.Rope;
        for (int i = span.First; i < span.End; i++)
        {
            string block = $"transformer_blocks.{i}.";
            // Self and text attention of each stream.
            int SelfAndText(bool isVideo, int state, int context, long rows, long contextRows)
            {
                string p = isVideo ? "" : "audio_";
                long dim = isVideo ? vdim : adim, times = isVideo ? tv : ta;
                int heads = isVideo ? c.VideoHeads : c.AudioHeads, headDim = (int)(dim / heads);
                int main = isVideo ? videoMain : audioMain, prompt = isVideo ? videoPrompt : audioPrompt;
                string table = block + p + "scale_shift_table";
                var streamRope = isVideo ? rope.Video : rope.Audio;
                int shift = ModulationRow(main, 9, 0, times, table, 0, dim), scale = ModulationRow(main, 9, 1, times, table, 1, dim);
                int gate = ModulationRow(main, 9, 2, times, table, 2, dim);
                int modulated = Modulate(Norm(state), shift, scale);
                state = Residual(state, Attention(block + p + "attn1", modulated, rows, modulated, rows, heads, headDim, streamRope, streamRope), gate);
                shift = ModulationRow(main, 9, 6, times, table, 6, dim);
                scale = ModulationRow(main, 9, 7, times, table, 7, dim);
                gate = ModulationRow(main, 9, 8, times, table, 8, dim);
                int query = Modulate(Norm(state), shift, scale);
                string promptTable = block + p + "prompt_scale_shift_table";
                int contextShift = ModulationRow(prompt, 2, 0, 1, promptTable, 0, dim), contextScale = ModulationRow(prompt, 2, 1, 1, promptTable, 1, dim);
                int projectedContext = Modulate(context, contextShift, contextScale);
                return Residual(state, Attention(block + p + "attn2", query, rows, projectedContext, contextRows, heads, headDim), gate);
            }
            video = SelfAndText(true, video, videoContext, nv, nvc);
            audio = SelfAndText(false, audio, audioContext, na, nac);

            // Cross-stream attention: both streams normed BEFORE either residual moves (V2A never sees the updated video).
            int videoNorm = Norm(video), audioNorm = Norm(audio);
            string videoTable = block + "scale_shift_table_a2v_ca_video", audioTable = block + "scale_shift_table_a2v_ca_audio";
            // CrossInput: the scale at row `from`, the shift at `from + 1` of the cross scale/shift modulation.
            int CrossInput(int normed, bool isVideo, int from)
            {
                long dim = isVideo ? vdim : adim, times = isVideo ? tv : ta;
                int cross = isVideo ? videoCross : audioCross;
                string table = isVideo ? videoTable : audioTable;
                int scale = ModulationRow(cross, 4, from, times, table, from, dim);
                int shift = ModulationRow(cross, 4, from + 1, times, table, from + 1, dim);
                return Modulate(normed, shift, scale);
            }
            int a2vGate = ModulationRow(videoCrossGate, 1, 0, 1, videoTable, 4, vdim);
            int a2v = Attention(block + "audio_to_video_attn", CrossInput(videoNorm, true, 0), nv, CrossInput(audioNorm, false, 0), na,
                c.AudioHeads, c.AudioHeadDim, rope.CrossVideo, rope.Audio);
            int v2aGate = ModulationRow(audioCrossGate, 1, 0, 1, audioTable, 4, adim);
            int v2a = Attention(block + "video_to_audio_attn", CrossInput(audioNorm, false, 2), na, CrossInput(videoNorm, true, 2), nv,
                c.AudioHeads, c.AudioHeadDim, rope.Audio, rope.CrossVideo);
            video = Residual(video, a2v, a2vGate);
            audio = Residual(audio, v2a, v2aGate);

            // Feed-forwards.
            int FeedForward(bool isVideo, int state)
            {
                string p = isVideo ? "" : "audio_";
                long dim = isVideo ? vdim : adim, times = isVideo ? tv : ta;
                int main = isVideo ? videoMain : audioMain;
                string table = block + p + "scale_shift_table";
                int shift = ModulationRow(main, 9, 3, times, table, 3, dim), scale = ModulationRow(main, 9, 4, times, table, 4, dim);
                int gate = ModulationRow(main, 9, 5, times, table, 5, dim);
                int modulated = Modulate(Norm(state), shift, scale);
                int update = Linear(block + p + "ff.net.2", b.Op(GraphOp.Gelu, Linear(block + p + "ff.net.0.proj", modulated)));
                return Residual(state, update, gate);
            }
            video = FeedForward(true, video);
            audio = FeedForward(false, audio);
        }

        // A stage that does not close the step hands both residual streams on.
        if (!span.IsLast)
        {
            e.Output(video, nv * vdim);
            e.Output(audio, na * adim);
            return;
        }

        // ---- output heads: the embedded timestep plus the table's two rows, the layer-normed state modulated, then proj_out
        int Output(bool isVideo, int state, int embedded)
        {
            string p = isVideo ? "" : "audio_";
            long dim = isVideo ? vdim : adim;
            string table = p + "scale_shift_table";
            int shift = b.Op(GraphOp.Add, embedded, e.View2(Vector(table), dim, 1, dim * 4L, 0));
            int scale = b.Op(GraphOp.Add, embedded, e.View2(Vector(table), dim, 1, dim * 4L, dim * 4L));
            return Linear(p + "proj_out", Modulate(GraphDitRegions.RowNorm(b, state, -1, layer: true, eps), shift, scale));
        }
        e.Output(Output(true, video, videoEmbedded), nv * c.VideoChannels);
        e.Output(Output(false, audio, audioEmbedded), na * c.AudioChannels);
    }

    private void Validate(Ltx25DenoiserInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        static void Rows(float[] values, int rows, int width)
        {
            if (rows <= 0 || values == null || values.Length != checked(rows * width) ||
                values.Any(value => !float.IsFinite(value))) throw new ArgumentException("Invalid LTX token rows.");
        }
        static void Times(float sigma, float[] values, int tokens)
        {
            if (!float.IsFinite(sigma) || sigma < 0 || sigma > 1 || values != null &&
                ((values.Length != 1 && values.Length != tokens) || values.Any(value => !float.IsFinite(value) || value < 0 || value > 1)))
                throw new ArgumentException("LTX sigma/timesteps must be finite [0,1] scalar or per-token values.");
        }
        Rows(input.Video, input.VideoTokens, Config.VideoChannels); Rows(input.Audio, input.AudioTokens, Config.AudioChannels);
        Rows(input.VideoContext, input.VideoContextTokens, Config.VideoContextDim); Rows(input.AudioContext, input.AudioContextTokens, Config.AudioContextDim);
        Times(input.VideoSigma, input.VideoTimesteps, input.VideoTokens); Times(input.AudioSigma, input.AudioTimesteps, input.AudioTokens);
        if (input.VideoPositions == null || input.VideoPositions.Tokens != input.VideoTokens || input.VideoPositions.Axes != 3 ||
            input.AudioPositions == null || input.AudioPositions.Tokens != input.AudioTokens || input.AudioPositions.Axes != 1 ||
            !Enum.IsDefined(input.FrequencyPrecision)) throw new ArgumentException("LTX latent position grids differ.");
        if (input.KeyframeMask != null && (input.KeyframeMask.Length != input.VideoTokens || input.KeyframeMask.Any(value => !float.IsFinite(value))))
            throw new ArgumentException("Invalid LTX keyframe mask.");
    }
}
