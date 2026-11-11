// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3: joint video + 32 kHz stereo audio generation.
//
// Loaded from the denoiser GGUF; the Qwen3-VL text encoder, video VAE and audio VAE
// are resolved beside it or through TS_VIDEO_TEXT_ENCODER / TS_VIDEO_VAE /
// TS_VIDEO_AUDIO_VAE. Not an autoregressive text model — generation goes through
// GenerateVideo().
//
// Note the checkpoints carry NO GGUF metadata, so this model is identified by the
// joint presence of video_patch_proj and audio_patch_proj rather than by an
// architecture string, and the tokenizer has to come from separate vocab files.
//
// LoRA plug-ins (--lora / TS_LORAS) are minimax-h3-lora configs: the LightX2V Turbo
// step-distillation adapters for the FL2VA checkpoint, applied unmerged, which also
// lock the sampling recipe they were distilled on (MiniMaxH3LoraConfig).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models.Video;
using TensorSharp.Models.WanVideo;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>The MiniMax-H3 audio-video generator.</summary>
    public sealed class MiniMaxH3Model : ModelBase, IVideoGenerationModel, ILoraPluginHost
    {
        /// <summary>Architecture id this model reports. The GGUFs have no metadata,
        /// so nothing reads this back out of a file — it exists for the API and UI.</summary>
        public const string ArchitectureId = "minimax-h3";

        /// <summary>What <see cref="SetLoras"/> throws (an <see cref="ObjectDisposedException"/>)
        /// when the model is unloaded before the plug-in it was loading is in place.</summary>
        internal const string UnloadedText = "MiniMax-H3 was unloaded; its LoRA plug-ins were not applied.";

        /// <summary>Test seam: runs after <see cref="SetLoras"/> has loaded a set and before it
        /// checks for an unload, so a test can unload the model while a set is in hand.</summary>
        internal static Action<MiniMaxH3LoraSet> LoraLoadedHookForTest;

        private readonly string _ditPath;
        private readonly string _firstLastDitPath;
        private readonly string _referenceDitPath;
        private readonly string _tePath;
        private readonly string _vaePath;
        private readonly string _audioVaePath;
        private MiniMaxH3Pipeline _pipeline;

        // One clip at a time, and never across a plug-in swap or an unload: GenerateVideo,
        // SetLoras and Dispose all hold it. Lock order is GpuComputeLock (the hosts'), then
        // this; nothing waits on GpuComputeLock while holding it.
        private readonly object _generationLock = new();
        // Set by Dispose BEFORE it waits for _generationLock, so a SetLoras still loading its
        // plug-in sees the unload when it finishes and releases the set instead of keeping it.
        private volatile bool _disposed;

        /// <summary>The backend this model was loaded for; the pipeline needs it to
        /// make the matching GGML backend the active one.</summary>
        internal BackendType Backend => _backend;

        /// <summary>The allocator the networks are built on (a CudaAllocator on the direct CUDA
        /// backend), for the GPU memory gauges (<see cref="Telemetry.GpuMemoryTelemetry"/>).</summary>
        internal IAllocator DirectAllocator => _allocator;

        public MiniMaxH3Config DitConfig { get; }
        public MiniMaxH3Partition Partition { get; }

        public MiniMaxH3Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            _ditPath = ggufPath;
            string dir = Path.GetDirectoryName(Path.GetFullPath(ggufPath));

            using (var probe = new GgufFile(ggufPath))
            {
                DitConfig = MiniMaxH3Config.Detect(probe.Tensors);
            }
            Partition = MiniMaxH3Config.PartitionFromFileName(ggufPath);
            string alternatePath = ResolveAlternateDenoiser(ggufPath, Partition);
            if (alternatePath != null)
            {
                try
                {
                    using var alternate = new GgufFile(alternatePath);
                    MiniMaxH3Config alternateConfig = MiniMaxH3Config.Detect(alternate.Tensors);
                    if (!SameDenoiserShape(DitConfig, alternateConfig))
                    {
                        Console.WriteLine($"MiniMax-H3 alternate denoiser ignored (incompatible tensor shapes): {alternatePath}");
                        alternatePath = null;
                    }
                }
                catch (Exception e) when (e is IOException or InvalidOperationException or ArgumentException)
                {
                    Console.WriteLine($"MiniMax-H3 alternate denoiser ignored ({e.Message}): {alternatePath}");
                    alternatePath = null;
                }
            }
            _firstLastDitPath = Partition == MiniMaxH3Partition.FirstLastFrame ? ggufPath : alternatePath;
            _referenceDitPath = Partition == MiniMaxH3Partition.Reference ? ggufPath : alternatePath;
            Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = 0 };

            _tePath = ResolveCompanion("TS_VIDEO_TEXT_ENCODER", dir,
                new[] { "qwen3vl_32b_minimax_h3-Q4_K_M.gguf" },
                n => n.Contains("qwen3vl") && n.EndsWith(".gguf"));
            _vaePath = ResolveCompanion("TS_VIDEO_VAE", dir,
                new[] { "minimax_h3_video_vae_fp16.safetensors" },
                n => n.Contains("video_vae") && n.EndsWith(".safetensors"));
            _audioVaePath = ResolveCompanion("TS_VIDEO_AUDIO_VAE", dir,
                new[] { "minimax_h3_audio_vae_fp32.safetensors" },
                n => n.Contains("audio_vae") && n.EndsWith(".safetensors"));

            Console.WriteLine($"MiniMax-H3 ({Partition}): DiT={Path.GetFileName(ggufPath)}"
                + (alternatePath == null ? string.Empty : $", alternate={Path.GetFileName(alternatePath)}"));
            Console.WriteLine($"  {DitConfig}");
            Console.WriteLine($"  text-encoder = {_tePath ?? "<missing>"}");
            Console.WriteLine($"  video VAE    = {_vaePath ?? "<missing>"}");
            Console.WriteLine($"  audio VAE    = {_audioVaePath ?? "<missing> (video only)"}");

            if (_tePath == null || _vaePath == null)
                throw new FileNotFoundException(
                    "MiniMax-H3 needs companion models beside the denoiser GGUF (or via " +
                    "--video-text-encoder / --video-vae): the Qwen3-VL-32B text encoder GGUF " +
                    "(unsloth/MiniMax-H3-GGUF) and minimax_h3_video_vae_fp16.safetensors " +
                    "(Comfy-Org/MiniMax-H3). The text encoder also needs vocab.json, merges.txt " +
                    "and tokenizer_config.json (MiniMaxAI/MiniMax-H3, processor/) beside it, since " +
                    "its GGUF carries no tokenizer; a photo or a reference needs the last of them.");

            // LoRA plug-ins from the host (--lora / TS_LORAS). They are loaded and checked
            // against the transformer now, so a bad adapter or a contradicted recipe fails
            // before any request, and their configuration mistakes are refusals (one line,
            // exit 2), not crashes. TS_LORAS can also carry another family's plug-ins (a
            // Qwen-Image config inherited by this process); the host names what it skips,
            // and this model applies only the ones it accepts.
            try
            {
                var specs = LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable))
                    .Where(AcceptsLora).ToList();
                if (specs.Count > 0) SetLoras(specs);
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or JsonException or FormatException or
                FileNotFoundException or KeyNotFoundException)
            {
                Dispose();
                throw new ModelLoadRefusedException("LoRA plug-in refused: " + e.Message, e);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        private static string ResolveAlternateDenoiser(string primaryPath, MiniMaxH3Partition primaryPartition)
        {
            string directory = Path.GetDirectoryName(Path.GetFullPath(primaryPath));
            string primaryName = Path.GetFileName(primaryPath);
            string marker = primaryPartition == MiniMaxH3Partition.Reference ? "ref2va" : "fl2va";
            int markerAt = primaryName.IndexOf(marker, StringComparison.OrdinalIgnoreCase);
            if (markerAt < 0) return null;
            string alternateMarker = primaryPartition == MiniMaxH3Partition.Reference ? "fl2va" : "ref2va";
            string expectedName = primaryName[..markerAt] + alternateMarker + primaryName[(markerAt + marker.Length)..];
            string expectedPath = Path.Combine(directory, expectedName);
            if (File.Exists(expectedPath)) return expectedPath;

            // Keep custom quantized pairs usable when they follow the published filename
            // pattern, while refusing to guess if a folder contains multiple alternatives.
            string[] candidates = Directory.Exists(directory)
                ? Directory.EnumerateFiles(directory, "*.gguf", SearchOption.TopDirectoryOnly)
                    .Where(path => !string.Equals(Path.GetFullPath(path), Path.GetFullPath(primaryPath), StringComparison.Ordinal)
                        && MiniMaxH3Config.PartitionFromFileName(path) != primaryPartition)
                    .ToArray()
                : Array.Empty<string>();
            return candidates.Length == 1 ? candidates[0] : null;
        }

        private static bool SameDenoiserShape(MiniMaxH3Config a, MiniMaxH3Config b) =>
            a.HiddenSize == b.HiddenSize && a.NumLayers == b.NumLayers
            && a.TokenRefinerLayers == b.TokenRefinerLayers && a.NumAttentionHeads == b.NumAttentionHeads
            && a.AttentionHeadDim == b.AttentionHeadDim && a.FfnHiddenSize == b.FfnHiddenSize
            && a.VideoLatentChannels == b.VideoLatentChannels && a.AudioLatentChannels == b.AudioLatentChannels
            && a.TextDim == b.TextDim && a.RopeInvFreqLength == b.RopeInvFreqLength
            && a.TimeEmbedding == b.TimeEmbedding && a.TimeEmbedDim == b.TimeEmbedDim
            && a.AdaLnCurveGrid == b.AdaLnCurveGrid;

        // ---- LoRA plug-ins (ILoraPluginHost) ----

        /// <inheritdoc/>
        public string LoraPluginType => LoraCliFlags.MiniMaxH3PluginType;

        /// <summary>The plug-in in use, as <see cref="LoraCliFlags.Resolve"/> expands it (empty when none).</summary>
        public IReadOnlyList<LoraSpec> LoraSpecs { get; private set; } = Array.Empty<LoraSpec>();

        /// <summary>A <c>minimax-h3-lora</c> config, as <c>--lora</c> or as the <c>--lora-config</c>
        /// of its weights. Bare weights and other families' plug-ins are not.</summary>
        public bool AcceptsLora(LoraSpec spec) =>
            LoraCliFlags.DeclaredPluginType(spec) == LoraCliFlags.MiniMaxH3PluginType;

        /// <summary>The loaded plug-in, or null; the pipeline applies it to every step.</summary>
        internal MiniMaxH3LoraSet Loras { get; private set; }

        /// <summary>The recipe the loaded plug-in locks (steps, linspace grid, shifts, Euler,
        /// CFG 1), or null when clips sample with the base checkpoint's own schedule.</summary>
        public MiniMaxH3SamplingRecipe SamplingRecipe { get; private set; }

        VideoSamplingContract IVideoGenerationModel.SamplingContract => SamplingRecipe?.ToContract();

        /// <summary>The contract <paramref name="specs"/> would bind this model to, read from the
        /// plug-in configs alone (no weights, no download); null for an empty list.</summary>
        /// <exception cref="ArgumentException">The specs are refused, with the canonical text.</exception>
        /// <exception cref="FileNotFoundException">A named config is missing.</exception>
        public VideoSamplingContract ContractFor(IReadOnlyList<LoraSpec> specs) =>
            MiniMaxH3LoraConfig.RecipeFor(specs, Partition, Path.GetFileName(_ditPath))?.ToContract();

        /// <summary>
        /// Replace the LoRA plug-in applied to every later clip; an empty list removes it. At
        /// most one <c>minimax-h3-lora</c> config, written for this checkpoint's partition. It is
        /// checked from its config before anything is downloaded, then loaded and validated
        /// against the transformer. A failure leaves the previous set in place. Waits for a clip
        /// in progress.
        /// </summary>
        /// <exception cref="ObjectDisposedException">The model was unloaded first, or while the
        /// plug-in loaded (<see cref="UnloadedText"/>); nothing was applied.</exception>
        public void SetLoras(IReadOnlyList<LoraSpec> specs)
        {
            ArgumentNullException.ThrowIfNull(specs);
            lock (_generationLock)
            {
                if (_disposed) throw new ObjectDisposedException(null, UnloadedText);
                // The refusals that need the configs alone (another family's plug-in, bare
                // weights, two plug-ins, the Ref2VA checkpoint, a negative strength, weights the
                // config was not written for) come before Resolve downloads anything.
                MiniMaxH3LoraConfig.RecipeFor(specs, Partition, Path.GetFileName(_ditPath));
                var resolved = LoraCliFlags.Resolve(specs);
                MiniMaxH3LoraSet loaded = null;
                if (resolved.Count > 0)
                {
                    var timer = Stopwatch.StartNew();
                    loaded = MiniMaxH3LoraSet.Load(resolved, DitConfig, Partition, Backend, UsesDirectBackend);
                    try
                    {
                        LoraLoadedHookForTest?.Invoke(loaded);
                        if (_disposed) throw new ObjectDisposedException(null, UnloadedText);
                    }
                    catch
                    {
                        loaded.Dispose();
                        throw;
                    }
                    Console.WriteLine($"MiniMax-H3 LoRA: {Path.GetFileName(loaded.WeightsPath)} applied unmerged " +
                        $"({loaded.FactorBytes / (1024.0 * 1024.0):F0} MiB of factors, loaded in {timer.Elapsed.TotalSeconds:F1}s)");
                    Console.WriteLine(loaded.Summary);
                }
                else if (Loras != null)
                {
                    Console.WriteLine("MiniMax-H3 LoRA: plug-in removed; clips sample with the base checkpoint's schedule.");
                }
                var previous = Loras;
                Loras = loaded;
                SamplingRecipe = loaded?.Recipe;
                LoraSpecs = resolved;
                if (!ReferenceEquals(previous, loaded))
                {
                    _pipeline?.ReleaseLora();
                    previous?.Dispose();
                }
            }
        }

        /// <summary>True when the tensor table looks like a MiniMax-H3 denoiser. Used
        /// to route a GGUF that has no architecture metadata to read.</summary>
        public static bool LooksLikeMiniMaxH3(GgufFile gguf) =>
            gguf != null && MiniMaxH3Config.IsMiniMaxH3(gguf.Tensors);

        internal static string ResolveCompanion(string envVar, string dir,
                                               string[] preferred, Func<string, bool> match)
        {
            string env = Environment.GetEnvironmentVariable(envVar);
            if (!string.IsNullOrWhiteSpace(env) && File.Exists(env)) return env;
            string parent = Path.GetDirectoryName(dir);
            foreach (var d in new[] { dir, parent })
            {
                if (d == null) continue;
                foreach (var name in preferred)
                {
                    string p = Path.Combine(d, name);
                    if (File.Exists(p)) return p;
                }
            }
            foreach (var d in new[] { dir, parent })
            {
                if (d == null || !Directory.Exists(d)) continue;
                foreach (var f in Directory.EnumerateFiles(d, "*", SearchOption.AllDirectories)
                                           .OrderBy(f => f.Length))
                    if (match(Path.GetFileName(f).ToLowerInvariant())) return f;
            }
            return null;
        }

        internal MiniMaxH3TextEncoder CreateTextEncoder() => new(_tePath, null, Backend, _allocator);
        internal MiniMaxH3DiT CreateDiT() => new(_ditPath, Backend, _allocator);
        internal MiniMaxH3DiT CreateDiT(MiniMaxH3Partition partition) => new(DenoiserPathFor(partition), Backend, _allocator);
        internal MiniMaxH3VideoVae CreateVideoVae() => new(_vaePath, Backend, _allocator);

        /// <summary>Whether this installation has the denoiser for a generation mode.</summary>
        public bool SupportsPartition(MiniMaxH3Partition partition) =>
            partition == MiniMaxH3Partition.FirstLastFrame ? _firstLastDitPath != null : _referenceDitPath != null;

        /// <summary>Choose the installed denoiser for the supplied mode. The shared text encoder
        /// and VAEs stay attached to this loaded model; only the denoiser changes between clips.</summary>
        public MiniMaxH3Partition PartitionFor(VideoGenerationParams parameters)
        {
            MiniMaxH3Partition partition = MiniMaxH3ModeResolver.PartitionFor(parameters, Partition);
            if (!SupportsPartition(partition))
            {
                string name = partition == MiniMaxH3Partition.Reference ? "Ref2VA" : "FL2VA";
                string file = partition == MiniMaxH3Partition.Reference ? "minimax_h3_ref2va_pruned-*.gguf" : "minimax_h3_fl2va_pruned-*.gguf";
                throw new InvalidOperationException($"MiniMax-H3 {name} conditioning needs {file} beside the loaded denoiser. Download the complete MiniMax-H3 model entry to enable both modes.");
            }
            return partition;
        }

        /// <summary>The installed denoiser path for the requested conditioning partition.</summary>
        public string DenoiserPathFor(MiniMaxH3Partition partition) => partition switch
        {
            MiniMaxH3Partition.FirstLastFrame => _firstLastDitPath ?? _ditPath,
            MiniMaxH3Partition.Reference => _referenceDitPath ?? _ditPath,
            _ => throw new ArgumentOutOfRangeException(nameof(partition)),
        };

        /// <summary>Pull a weight file through the OS file cache in one sequential pass.
        ///
        /// <para>Weights are bound as pointers into the mmapped GGUF, so the first upload faults
        /// each page in from disk from INSIDE the host-to-device copy. That is the pathological
        /// case for the driver: measured on a 16 GB RTX 3080 Laptop the 10.6 GB denoiser moved
        /// at 0.91 GB/s, against 5.97 GB/s for a pageable copy out of resident memory and 2.7 GB/s
        /// for a plain sequential read of the same file. Reading the file first turns scattered
        /// fault-in into one streaming read, and the copy then runs at memory speed.</para>
        ///
        /// <para>Returns the seconds spent, or -1 when it was skipped. It is skipped when the
        /// file cannot be sized, and when it is larger than the RAM available to hold it - on a
        /// host that cannot cache the file this would evict as it goes and cost a second read
        /// for nothing.</para></summary>
        internal static double PrefaultWeightFile(string path, long availableBytes)
        {
            try
            {
                if (string.IsNullOrEmpty(path)) return -1;
                var fi = new FileInfo(path);
                if (!fi.Exists || fi.Length <= 0) return -1;
                if (availableBytes > 0 && fi.Length > availableBytes) return -1;
                var sw = System.Diagnostics.Stopwatch.StartNew();
                // One stream on macOS, which is the opposite of GgufReader.PrefaultFileCache
                // and is deliberate: this read runs CONCURRENTLY with the encoder teardown and
                // the denoiser upload it is warming, and one stream already saturates an Apple
                // SSD. Splitting the file into one region per thread measured worse on a local
                // disk (640x384, best of three: 1 stream 63.9 s, 4 streams 64.9 s, 16 streams
                // 66.6 s). Linux model storage is often a network volume that needs concurrency
                // instead (an A40 pod's: 0.31 GiB/s for one 8 MiB reader, 2.3 GiB/s for sixteen),
                // so there sixteen readers take consecutive chunks of ONE sweep, staying near the
                // same position, and skip chunks that are already cached. TS_H3_PREFAULT_THREADS
                // overrides the reader count.
                string tenv = Environment.GetEnvironmentVariable("TS_H3_PREFAULT_THREADS");
                int? threads = !string.IsNullOrEmpty(tenv) && int.TryParse(tenv, out int tset) && tset > 0 ? tset : null;
                using var read = Video.MappedFileReadAhead.TryStart(path, Video.MappedFileReadAhead.WholeFile(path),
                    windowBytes: long.MaxValue, workers: threads, honorSwitch: false);
                if (read == null) return -1;
                read.Wait();
                return sw.Elapsed.TotalSeconds;
            }
            catch (IOException) { return -1; }
            catch (UnauthorizedAccessException) { return -1; }
        }

        /// <summary>The denoiser GGUF, for prefaulting.</summary>
        internal string DiTPath => _ditPath;

        /// <summary>The text-encoder GGUF, for reading ahead.</summary>
        internal string TextEncoderPath => _tePath;

        /// <summary>The video VAE file, for prefaulting.</summary>
        internal string VideoVaePath => _vaePath;

        /// <summary>Roughly what the video VAE will occupy on the device once resident, used
        /// to decide whether it can share the card with the still-resident denoiser.
        ///
        /// <para>The safetensors file size is the weight footprint almost exactly - the VAE is
        /// dense F16 with no quantized block overhead - and the extra eighth covers the decode's
        /// own activations, which are bounded because the decoder already runs 5 latent frames
        /// at a time and tiles at 256 px. Returns 0 when the size cannot be read, which the
        /// caller treats as "do not intervene".</para></summary>
        internal long VideoVaeResidentBytesEstimate()
        {
            try
            {
                if (string.IsNullOrEmpty(_vaePath)) return 0;
                var fi = new FileInfo(_vaePath);
                if (!fi.Exists) return 0;
                return fi.Length + fi.Length / 8;
            }
            catch (IOException) { return 0; }
            catch (UnauthorizedAccessException) { return 0; }
        }

        /// <summary>Roughly what the text encoder occupies on the device while it runs, used
        /// to decide whether the networks kept from the previous request can stay beside it.
        ///
        /// <para>The GGUF is uploaded as it is stored, and the encode's own activations are
        /// small beside it, so the file size is the estimate. Returns 0 when the size cannot
        /// be read, which the caller treats as "do not intervene".</para></summary>
        internal long TextEncoderResidentBytesEstimate()
        {
            try
            {
                if (string.IsNullOrEmpty(_tePath)) return 0;
                var fi = new FileInfo(_tePath);
                return fi.Exists ? fi.Length : 0;
            }
            catch (IOException) { return 0; }
            catch (UnauthorizedAccessException) { return 0; }
        }

        internal MiniMaxH3AudioVae CreateAudioVae() =>
            _audioVaePath != null
                ? new MiniMaxH3AudioVae(_audioVaePath, Backend, UsesDirectBackend, _allocator)
                : null;
        /// <summary>True on the backends with no ggml graph to call, which run the
        /// generator through the managed direct implementations instead.</summary>
        internal bool UsesDirectBackend =>
            _allocator != null && Backend is not (BackendType.GgmlCuda or BackendType.GgmlCpu
                or BackendType.GgmlMetal or BackendType.GgmlVulkan);

        internal string AudioVaePath => _audioVaePath;

        /// <summary>Generate a clip with guidance 1.0. Base MiniMax-H3 defaults to
        /// 20 steps; reduced-step runs need their own quality validation. With a LoRA
        /// plug-in loaded (<see cref="SetLoras"/>), the plug-in's recipe fixes the steps,
        /// shift and sampler, and a request that contradicts it is refused.</summary>
        public GeneratedVideo GenerateVideo(string prompt, VideoGenerationParams p = null)
        {
            if (_allocator is MlxAllocator && !MlxWorker.Shared.IsOnWorkerThread)
                return MlxWorker.Shared.Invoke(() => GenerateVideo(prompt, p));
            lock (_generationLock)
            {
                if (_disposed) throw new ObjectDisposedException(nameof(MiniMaxH3Model));
                _pipeline ??= new MiniMaxH3Pipeline(this);
                return _pipeline.Generate(prompt, p ?? new VideoGenerationParams());
            }
        }

        // ---- IVideoGenerationModel ----
        string IVideoGenerationModel.VideoModelFamily => ArchitectureId;
        int IVideoGenerationModel.DefaultPreviewFrames => 22;
        int IVideoGenerationModel.FrameCountMultiple => 17;
        int IVideoGenerationModel.FrameCountOffset => 5;
        int IVideoGenerationModel.DefaultPreviewFps => MiniMaxH3Geometry.Fps;
        int IVideoGenerationModel.DefaultOutputWidth => 640;
        int IVideoGenerationModel.DefaultOutputHeight => 384;
        int IVideoGenerationModel.ResolutionMultiple => 32;
        // The joint audio stream is generated either way; only its decode needs the VAE.
        bool IVideoGenerationModel.SupportsAudio => _audioVaePath != null;
        bool IVideoGenerationModel.SupportsImageConditioning => SupportsPartition(MiniMaxH3Partition.FirstLastFrame);
        bool IVideoGenerationModel.SupportsEndImageConditioning => SupportsPartition(MiniMaxH3Partition.FirstLastFrame);
        bool IVideoGenerationModel.SupportsReferenceConditioning => SupportsPartition(MiniMaxH3Partition.Reference);
        int IVideoGenerationModel.MaxReferenceImages => SupportsPartition(MiniMaxH3Partition.Reference) ? MiniMaxH3Geometry.MaxReferences : 0;

        // ---- not an autoregressive model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException(
                "MiniMaxH3Model is a video-generation model; use GenerateVideo().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        // The pipeline keeps the denoiser and both VAEs between requests, each holding
        // pinned arrays, its own file mapping and the device wraps of its weights.
        // Without this every unload leaked them, and the Metal wraps still registered
        // with the device are what ggml-metal asserts on at process exit. The LoRA set
        // goes after the pipeline, whose denoiser may still hold its device copies.
        public override void Dispose()
        {
            _disposed = true;
            lock (_generationLock)
            {
                _pipeline?.Dispose();
                _pipeline = null;
                Loras?.Dispose();
                Loras = null;
                SamplingRecipe = null;
                LoraSpecs = Array.Empty<LoraSpec>();
                base.Dispose();
            }
        }
    }
}
