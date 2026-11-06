// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server
{
    internal sealed class ModelLifecycleService : IDisposable
    {
        private readonly ILogger _logger;
        private readonly Func<string, BackendType, ITensorParallelGroup, string, ModelBase> _createModel;

        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        public ModelLifecycleService(ILogger logger)
            : this(logger, static (path, backend, tpGroup, draftPath) =>
                ModelBase.Create(path, backend, tpDegree: tpGroup is ILayerSplitGroup ? 1 : tpGroup?.Degree ?? 1,
                    tpGroup: tpGroup, draftModelPath: draftPath,
                    layerSplitDegree: tpGroup is ILayerSplitGroup ? tpGroup.Degree : 1))
        {
        }

        /// <summary>Test seam: <paramref name="createModel"/> stands in for
        /// <see cref="ModelBase.Create(string, BackendType, int, ITensorParallelGroup, string, int)"/>.</summary>
        internal ModelLifecycleService(ILogger logger, Func<string, BackendType, ITensorParallelGroup, string, ModelBase> createModel)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
            _createModel = createModel;
        }

        public bool IsLoaded => _model != null;

        /// <summary>
        /// The operator's <c>--chat-template-file</c>, and the model it was given for
        /// (the host's <c>--model</c>). Applied at every load of that model, before
        /// anything renders a prompt with it; another model loaded later keeps its own
        /// template, and the load says so. Set before <see cref="LoadModel"/>.
        /// </summary>
        public (ChatTemplateFile Template, string ModelPath)? ChatTemplateOverride { get; set; }

        private bool _chatTemplateOverrideSkipReported;

        private void ApplyChatTemplateOverride(string modelPath)
        {
            if (ChatTemplateOverride is not { } chatOverride || _model?.Config == null)
                return;

            if (!SameFile(modelPath, chatOverride.ModelPath))
            {
                if (!_chatTemplateOverrideSkipReported)
                {
                    _chatTemplateOverrideSkipReported = true;
                    _logger.LogWarning(LogEventIds.HostConfiguration,
                        "{Flag} {TemplateFile} applies to {ConfiguredModel} only; {Model} keeps the chat template its GGUF ships.",
                        ChatTemplateFile.Flag, chatOverride.Template.Path, Path.GetFileName(chatOverride.ModelPath),
                        Path.GetFileName(modelPath));
                }
                return;
            }

            string shipped = _model.Config.ChatTemplate;
            _model.Config.ChatTemplate = chatOverride.Template.Template;
            _logger.LogInformation(LogEventIds.HostConfiguration,
                "Chat template: {TemplateFile} ({TemplateChars} chars) replaces {Shipped} for {Model}. " +
                "Output parsing and BOS handling stay the {Architecture} family's.",
                chatOverride.Template.Path, chatOverride.Template.Template.Length,
                string.IsNullOrEmpty(shipped) ? "the built-in renderer (the GGUF ships no template)" : $"the GGUF's template ({shipped.Length} chars)",
                Path.GetFileName(modelPath), _model.Config.Architecture ?? "unknown");
        }

        private static bool SameFile(string a, string b)
        {
            if (string.IsNullOrEmpty(a) || string.IsNullOrEmpty(b)) return false;
            try
            {
                return string.Equals(Path.GetFullPath(a), Path.GetFullPath(b),
                    OperatingSystem.IsLinux() ? StringComparison.Ordinal : StringComparison.OrdinalIgnoreCase);
            }
            catch (Exception ex) when (ex is ArgumentException or NotSupportedException or PathTooLongException)
            {
                return false;
            }
        }

        /// <summary>
        /// Builds the on-node <see cref="ITensorParallelGroup"/> a load shards across, given
        /// the resolved backend; null (the default) means a single-node load. Set by hosts
        /// that carry TensorSharp.Distributed. Read once per load, so it must be set before
        /// <see cref="LoadModel"/>.
        /// </summary>
        public Func<BackendType, ITensorParallelGroup> TensorParallelGroupFactory { get; set; }

        /// <summary>
        /// When the operator explicitly named an MTP draft via
        /// <c>--draft-model</c> (<c>TS_SPEC_DRAFT_MODEL</c>) but it could not be
        /// activated on the loaded target (missing file, wrong architecture, an
        /// incompatible draft, or an incomplete GGUF), this holds a
        /// human-readable reason. It is <c>null</c> when no draft was requested or
        /// when the draft loaded successfully (<c>HasDraftHead</c>). The startup loader
        /// promotes a non-null value to a fail-fast error so an explicit but
        /// unusable draft can't silently leave speculation disabled — matching the
        /// fail-fast contract the rest of startup configuration follows. Runtime
        /// (Web UI) model switches read the warning log but do not fail.
        /// </summary>
        public string DraftHeadActivationError { get; private set; }

        /// <summary>
        /// True when <see cref="DraftHeadActivationError"/> comes from the loaded
        /// model refusing speculation outright
        /// (<see cref="TensorSharp.Runtime.Speculative.ISpeculativeTarget.SpeculationRefusal"/>)
        /// rather than from a missing or mismatched draft file, so no other draft
        /// GGUF could fix it.
        /// </summary>
        public bool DraftHeadRefusedByModel { get; private set; }

        public string LoadedModelName => _loadedModelPath != null ? Path.GetFileName(_loadedModelPath) : null;
        public string LoadedModelPath => _loadedModelPath;
        public string LoadedMmProjName => _loadedMmProjPath != null ? Path.GetFileName(_loadedMmProjPath) : null;
        public string LoadedMmProjPath => _loadedMmProjPath;
        public string LoadedBackend => _model != null ? BackendCatalog.ToBackendValue(_backend) : null;
        public string Architecture => _model?.Config?.Architecture;
        public ModelBase Model => Volatile.Read(ref _model);
        public BackendType Backend => _backend;

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Distributed workers are launched with one fixed checkpoint.
        /// A runtime load cannot restart their warmup/worker loop or change their
        /// weights, so it must be refused before disposing the serving model.</summary>
        internal bool ShouldLoadDistributedModel(string modelPath, string mmProjPath, string backendStr, bool startupLoad)
        {
            bool distributed = TensorParallelGroupFactory != null &&
                (!string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID")) ||
                 !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS")));
            if (!distributed) return true;
            var comparison = OperatingSystem.IsWindows() ? StringComparison.OrdinalIgnoreCase : StringComparison.Ordinal;
            bool SamePath(string left, string right) => string.IsNullOrEmpty(left) || string.IsNullOrEmpty(right)
                ? string.IsNullOrEmpty(left) && string.IsNullOrEmpty(right)
                : string.Equals(Path.GetFullPath(left), Path.GetFullPath(right), comparison);
            if (_model != null && SamePath(modelPath, _loadedModelPath) && SamePath(mmProjPath, _loadedMmProjPath)
                && ResolveBackend(backendStr) == _backend)
                return false;
            if (!startupLoad)
                throw new ModelLoadRefusedException(
                    "Distributed tensor parallelism requires a startup --model. Runtime model loading, backend changes, " +
                    "and model switching are not supported because the worker nodes must load the same checkpoint " +
                    "and warm up together. Restart the driver server with --model and matching CLI workers.");
            return true;
        }

        public void LoadModel(string modelPath, string mmProjPath, string backendStr)
        {
            _logger.LogInformation(LogEventIds.ModelLoadStarted,
                "Loading model {ModelFile} (mmproj={MmProjFile}, backend={Backend}, fullPath={ModelPath}, mmprojPath={MmProjPath})",
                Path.GetFileName(modelPath), Path.GetFileName(mmProjPath ?? string.Empty),
                backendStr ?? "(default)", modelPath, mmProjPath ?? "(none)");

            try
            {
                ValidateModelFiles(modelPath, mmProjPath);
            }
            catch (Exception ex) when (ModelLoadRefusal.TryDescribe(ex, out string reason))
            {
                // A missing, non-GGUF or truncated file: the reason is the whole story,
                // and the stack trace only appears at Debug.
                _logger.LogError(LogEventIds.ModelLoadFailed,
                    "Rejected model load {ModelFile}: {Reason}; current model {CurrentModel} stays loaded",
                    Path.GetFileName(modelPath), reason, LoadedModelName ?? "(none)");
                _logger.LogDebug(LogEventIds.ModelLoadFailed, ex, "Rejected model load {ModelFile}", Path.GetFileName(modelPath));
                throw;
            }
            catch (Exception ex)
            {
                _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Rejected model load {ModelFile}: file validation failed; current model {CurrentModel} stays loaded",
                    Path.GetFileName(modelPath), LoadedModelName ?? "(none)");
                throw;
            }

            string previousModelPath = _loadedModelPath;
            string previousMmProjPath = _loadedMmProjPath;
            string previousBackendValue = LoadedBackend;

            UnloadCurrentModel();

            try
            {
                LoadModelCore(modelPath, mmProjPath, backendStr);
            }
            catch
            {
                // Best-effort rollback so a failed reload doesn't leave the
                // server with no model. The original exception still reaches
                // the caller; the rollback outcome is only logged.
                if (previousModelPath != null)
                {
                    try
                    {
                        LoadModelCore(previousModelPath, previousMmProjPath, previousBackendValue);
                        _logger.LogWarning("Restored previous model {PreviousModel} after failed load of {ModelFile}",
                            Path.GetFileName(previousModelPath), Path.GetFileName(modelPath));
                    }
                    catch (Exception rollbackEx) when (ModelLoadRefusal.TryDescribe(rollbackEx, out string rollbackReason))
                    {
                        // LoadModelCore already logged the refusal's reason; the stack trace
                        // of a refusal is Debug-only here too.
                        _logger.LogError(LogEventIds.ModelLoadFailed,
                            "Could not restore previous model {PreviousModel} after failed load of {ModelFile}: {Reason}; no model is loaded",
                            Path.GetFileName(previousModelPath), Path.GetFileName(modelPath), rollbackReason);
                    }
                    catch (Exception rollbackEx)
                    {
                        _logger.LogError(LogEventIds.ModelLoadFailed, rollbackEx,
                            "Could not restore previous model {PreviousModel} after failed load of {ModelFile}; no model is loaded",
                            Path.GetFileName(previousModelPath), Path.GetFileName(modelPath));
                    }
                }
                throw;
            }
        }

        /// <summary>
        /// Header-only validation of the files a load is about to commit to,
        /// run BEFORE the current model is disposed: a missing, non-GGUF, or
        /// truncated file is rejected while there is still a working model.
        /// </summary>
        private static void ValidateModelFiles(string modelPath, string mmProjPath)
        {
            if (TensorSharp.Runtime.NinferArtifactReader.IsNinferArtifact(modelPath))
            {
                // Images use the Qwen3.8 vision tower from an mmproj GGUF (validated below).
                using var artifact = new TensorSharp.Runtime.NinferArtifactReader(modelPath);
                _ = NinferQwen38TargetPlan.Create(artifact);
                _ = TensorSharp.Runtime.NinferQwen38Frontend.FromReader(artifact);
            }
            else
            {
                using var gguf = new GgufFile(modelPath);
                gguf.ThrowIfTruncated();
            }

            // A missing projector file is skipped by the load itself, but one
            // that exists must parse.
            //
            // A projector may also be a HuggingFace `.safetensors` vision shard rather
            // than an mmproj GGUF: some published checkpoints (diffusiongemma-26B-A4B-it)
            // never shipped an mmproj at all, and Gemma4VisionEncoder loads the tower
            // straight out of the shard. Opening that file as a GGUF here failed the
            // header check and refused the whole model load, so the one accepted form
            // could never be passed. Its structure is validated by the encoder, which
            // reads the safetensors header and names the missing tensor.
            if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath)
                && !IsSafetensorsProjector(mmProjPath))
            {
                using var mmProj = new GgufFile(mmProjPath);
                mmProj.ThrowIfTruncated();
            }
        }

        /// <summary>
        /// Whether a projector path is a HuggingFace safetensors shard rather than an
        /// mmproj GGUF. Extension only: the file's own header is what
        /// <c>Gemma4VisionEncoder</c> validates, and a mis-named file must fail there
        /// with the tensor name it wanted, not here with "not a GGUF".
        /// </summary>
        private static bool IsSafetensorsProjector(string path)
            => !string.IsNullOrEmpty(path)
               && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// Release the loaded model (and its projector and draft head) without loading
        /// another. The engine and diffusion scheduler that were built on top of the
        /// model must already be torn down: <see cref="ModelService.UnloadModel"/> does
        /// that ordering, which is why this is the seam it calls rather than a public API
        /// on its own.
        /// </summary>
        internal void Unload() => UnloadCurrentModel();

        /// <summary>
        /// How long an unload waits for a request that is still encoding an image or a
        /// clip on the outgoing model. An encoder that yields stops at its next block; one
        /// that does not finishes its encode first, and a long clip's frames take a while.
        /// </summary>
        internal static TimeSpan UseDrainTimeout { get; set; } = TimeSpan.FromMinutes(2);

        /// <summary>
        /// Advances every time a model is unloaded, so anything recorded against one load
        /// (a conversation's raw output tokens, see ConversationTranscriptStore) can tell it
        /// is looking at another model's state.
        /// </summary>
        public long LoadEpoch => Interlocked.Read(ref _loadEpoch);
        private long _loadEpoch;

        private void UnloadCurrentModel()
        {
            string previousModel = LoadedModelName;
            ModelBase outgoing = _model;
            // Unpublished FIRST. While the outgoing model drains below, a request arriving
            // now must find no model (and be refused) rather than find this one and build a
            // fresh engine on it that the dispose would then free under its worker.
            Volatile.Write(ref _model, null);
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            DraftHeadActivationError = null;
            DraftHeadRefusedByModel = false;
            if (outgoing != null)
            {
                Interlocked.Increment(ref _loadEpoch);
                // The engine has already been torn down, but a request can still be inside
                // the model: an image or audio encode runs on the request's thread before
                // anything reaches the engine. Stop new uses, let the running ones leave
                // (an encoder stops at its next yield), and dispose holding the GPU lock,
                // so nothing of this model runs while its weights and buffers are freed.
                outgoing.BeginRetirement();
                if (!outgoing.WaitForUsesToDrain(UseDrainTimeout))
                {
                    _logger.LogWarning(LogEventIds.ModelUnloaded,
                        "{Model} was still encoding a request after {Seconds}s; unloading once the GPU lock is free",
                        previousModel, UseDrainTimeout.TotalSeconds);
                }
                lock (outgoing.GpuComputeLock)
                    outgoing.Dispose();
            }

            if (!string.IsNullOrEmpty(previousModel))
            {
                _logger.LogInformation(LogEventIds.ModelUnloaded,
                    "Unloaded previous model {PreviousModel}", previousModel);
            }
        }

        /// <summary>Check header-only distributed support before startup rejects an
        /// unavailable backend or a load creates its GPU group and connects to peers.</summary>
        internal void ValidateDistributedModelSupport(string modelPath, string backendStr)
        {
            if (TensorParallelGroupFactory == null ||
                string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID")) ||
                string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS")))
                return;

            bool layerSelected = !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_LAYER_SPLIT_DEGREE")) &&
                (!int.TryParse(Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE"), out int tpDegree) || tpDegree <= 1);
            BackendType backend = ResolveBackend(backendStr);
            if (layerSelected) ModelBase.ValidateDistributedLayerSplit(modelPath, backend);
            else ModelBase.ValidateDistributedTensorParallelism(modelPath, backend);
        }

        private void LoadModelCore(string modelPath, string mmProjPath, string backendStr)
        {
            _backend = ResolveBackend(backendStr);

            var loadSw = Stopwatch.StartNew();
            ITensorParallelGroup tpGroup = null;
            try
            {
                // Multi-node tensor parallelism is the host's decision, not this
                // library's: the group is built by TensorSharp.Distributed, which
                // references the CUDA backend and so cannot be linked by every host
                // (an iOS app has no peers and no CUDA). The Server and CLI hand in a
                // factory that reads TENSORSHARP_TP_* and builds the on-node group to
                // match the backend; with no factory the load is single-node.
                ValidateDistributedModelSupport(modelPath, backendStr);
                tpGroup = TensorParallelGroupFactory?.Invoke(_backend);

                // Block drafters go to the factory rather than being attached
                // afterwards like Gemma 4's draft head: their weights have to be
                // counted by the layer split and uploaded with the trunk.
                string blockDraftPath = SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath();
                _model = _createModel(modelPath, _backend, tpGroup, blockDraftPath);
                ApplyChatTemplateOverride(modelPath);

                // A worker node (--tp-node-id > 0) spends its life blocked in a
                // mirror loop and cannot also serve HTTP requests, so the server
                // only takes the driver role (node 0). Fail fast with the
                // supported topology instead of hanging on the first inference.
                if (_model.IsDistributedWorker)
                {
                    throw new ModelLoadRefusedException(
                        "TensorSharp.Server cannot run as a distributed WORKER node " +
                        "(--tp-node-id > 0 / TENSORSHARP_TP_NODE_ID > 0). Run this server as node 0 (the driver) " +
                        "and start each worker node with TensorSharp.Cli, e.g.: " +
                        "TensorSharp.Cli --model <same.gguf> --backend <same> <--tp or --layer-split> <localGpus> --tp-node-id <N> --tp-peers <same list>.");
                }

                _loadedModelPath = modelPath;

                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    LoadEncoders(mmProjPath);
                    _loadedMmProjPath = mmProjPath;
                }

                // Speculator weights that ship as their own file (Gemma 4's
                // gemma4-assistant draft head) are attached here, by the same
                // shared loader the CLI uses. DraftHeadActivationError was
                // cleared when the previous model was unloaded.
                if (!SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(_model, out string draftError, out bool draftRefusedByModel))
                {
                    DraftHeadActivationError = draftError;
                    // The model's own decline (a trunk that refuses speculation, a DFlash
                    // drafter under --tp): the startup error then says to drop the flag
                    // rather than to go looking for a matching draft GGUF.
                    DraftHeadRefusedByModel = draftRefusedByModel;
                    _logger.LogWarning("{Error}; speculation disabled.", draftError);
                }
                else if (SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath() is { } attachedDraft)
                {
                    _logger.LogInformation("Loaded draft head {Draft} (HasDraftHead=True)",
                        Path.GetFileName(attachedDraft));
                }

                // --lora plug-ins (TS_LORAS) apply to the models that take them, each model only
                // its own family's (Qwen-Image-2.1, MiniMax-H3), and it loads without the rest;
                // say so instead of letting the adapter vanish silently.
                WarnAboutSkippedLoras(TensorSharp.Runtime.LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(
                    TensorSharp.Runtime.LoraCliFlags.EnvironmentVariable)));
                // So does the checkpoint declaration (--qwen-image-variant), which the CLI refuses
                // for another model outright.
                if (_model is not TensorSharp.Models.QwenImage.QwenImageModel &&
                    Environment.GetEnvironmentVariable(TensorSharp.Runtime.QwenImageVariantFlag.EnvironmentVariable) is { } variant &&
                    !string.IsNullOrWhiteSpace(variant))
                {
                    _logger.LogWarning(
                        "{Flag} / {EnvVar} ({Variant}) applies to Qwen-Image-2.1 models only; {Model} ({Architecture}) is not one and ignores it.",
                        TensorSharp.Runtime.QwenImageVariantFlag.Flag, TensorSharp.Runtime.QwenImageVariantFlag.EnvironmentVariable,
                        variant.Trim(), LoadedModelName, Architecture ?? "unknown architecture");
                }

                loadSw.Stop();
                long modelBytes = _model.ModelFilePaths.Sum(SafeGetFileSize);
                long mmProjBytes = SafeGetFileSize(mmProjPath);
                _logger.LogInformation(LogEventIds.ModelLoadCompleted,
                    "Loaded model {Model} (architecture={Architecture}, backend={Backend}, modelBytes={ModelBytes}, mmproj={MmProjFile}, mmprojBytes={MmProjBytes}) in {ElapsedMs:F1} ms",
                    LoadedModelName, Architecture ?? "(unknown)", LoadedBackend ?? "(unknown)",
                    modelBytes, LoadedMmProjName ?? "(none)", mmProjBytes, loadSw.Elapsed.TotalMilliseconds);
            }
            catch (Exception ex)
            {
                loadSw.Stop();
                if (ModelLoadRefusal.TryDescribe(ex, out string reason))
                {
                    // A refusal is a decision with its reason in the message; the stack
                    // trace of the loader that made it helps nobody, so it is Debug-only
                    // (TENSORSHARP_LOG_LEVEL=Debug). Anything else keeps its stack trace.
                    _logger.LogError(LogEventIds.ModelLoadFailed,
                        "Model load refused: {ModelFile} on backend {Backend} after {ElapsedMs:F1} ms: {Reason}",
                        Path.GetFileName(modelPath), backendStr ?? "(default)", loadSw.Elapsed.TotalMilliseconds, reason);
                    _logger.LogDebug(LogEventIds.ModelLoadFailed, ex,
                        "Model load refused: {ModelFile}", Path.GetFileName(modelPath));
                }
                else
                {
                    _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                        "Failed to load model {ModelFile} on backend {Backend} after {ElapsedMs:F1} ms",
                        Path.GetFileName(modelPath), backendStr ?? "(default)", loadSw.Elapsed.TotalMilliseconds);
                }
                // Drop any partially initialized model so the service holds
                // either a fully loaded model or none at all. A tensor-parallel group
                // built for a model that never came to exist has no other owner
                // (a model disposes its own group; Dispose is idempotent either way).
                _model?.Dispose();
                tpGroup?.Dispose();
                _model = null;
                _loadedModelPath = null;
                _loadedMmProjPath = null;
                DraftHeadActivationError = null;
                DraftHeadRefusedByModel = false;
                throw;
            }
        }

        /// <summary>
        /// Name the plug-ins in <paramref name="loras"/> (inherited as TS_LORAS) that the loaded
        /// model runs without: a model that takes LoRA plug-ins
        /// (<see cref="TensorSharp.Models.ILoraPluginHost"/>) skips the ones it does not accept
        /// (another family's), one warning per declared type, and a model that takes none skips
        /// them all. The texts are the ones the CLI's preflight prints for the same variable
        /// (<see cref="TensorSharp.Models.LoraHostPreflight.EnvForeignTypeWarning"/>,
        /// <see cref="TensorSharp.Models.LoraHostPreflight.EnvNoPluginsWarning"/>); the server
        /// says them here, at load, rather than at startup. A plug-in whose file cannot be read
        /// is named with the reason rather than thrown: the model has loaded, having read the
        /// same variable itself.
        /// </summary>
        private void WarnAboutSkippedLoras(IReadOnlyList<TensorSharp.Runtime.LoraSpec> loras)
        {
            if (loras.Count == 0)
                return;
            string architecture = Architecture ?? "unknown architecture";
            if (_model is not TensorSharp.Models.ILoraPluginHost host)
            {
                _logger.LogWarning("{Warning}", TensorSharp.Models.LoraHostPreflight.EnvNoPluginsWarning(
                    LoraFileNames(loras), LoadedModelName, architecture));
                return;
            }
            var skipped = new List<(string Declared, TensorSharp.Runtime.LoraSpec Spec)>();
            foreach (var spec in loras)
            {
                string declared;
                try
                {
                    if (host.AcceptsLora(spec))
                        continue;
                    declared = TensorSharp.Runtime.LoraCliFlags.DeclaredPluginType(spec);
                }
                catch (Exception ex) when (ex is IOException or UnauthorizedAccessException or ArgumentException)
                {
                    // A missing file or a config that is not JSON: the reason names the path.
                    _logger.LogWarning("{EnvVar} names LoRA plug-in {Lora}, which cannot be read: {Reason}",
                        TensorSharp.Runtime.LoraCliFlags.EnvironmentVariable, Path.GetFileName(spec.Path), ex.Message);
                    continue;
                }
                skipped.Add((declared, spec));
            }
            // Null (bare weights, a PEFT or PDD config) groups like any other declared type.
            foreach (var group in skipped.GroupBy(s => s.Declared, StringComparer.Ordinal))
            {
                _logger.LogWarning("{Warning}", TensorSharp.Models.LoraHostPreflight.EnvForeignTypeWarning(
                    LoraFileNames(group.Select(s => s.Spec)), group.Key, LoadedModelName, architecture, host.LoraPluginType));
            }
        }

        /// <summary>The plug-ins' file names, comma-separated, as the warning texts name them.</summary>
        private static string LoraFileNames(IEnumerable<TensorSharp.Runtime.LoraSpec> loras) =>
            string.Join(", ", loras.Select(l => Path.GetFileName(l.Path)));

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
        }

        private void LoadEncoders(string mmProjPath)
        {
            if (_model == null) return;
            _model.MultimodalInjector.LoadProjectors(
                TensorSharp.Models.Architecture.ModelArchitectureRegistry.ResolveProjectorPath(_model.Config.Architecture, mmProjPath));
        }

        private static BackendType ResolveBackend(string backendStr)
        {
            return BackendCatalog.Canonicalize(backendStr) switch
            {
                "mlx" => BackendType.Mlx,
                "cuda" => BackendType.Cuda,
                "ggml_metal" => BackendType.GgmlMetal,
                "ggml_cpu" => BackendType.GgmlCpu,
                "ggml_cuda" => BackendType.GgmlCuda,
                "ggml_vulkan" => BackendType.GgmlVulkan,
                "cpu" => BackendType.Cpu,
                // No backend named at all: the documented server default.
                null => BackendType.GgmlCpu,
                // A name we do not recognise used to fall back to ggml_cpu in silence,
                // hiding a typo behind a CPU-speed model. Both callers turn this into a
                // clean error (the Web UI's 500 JSON, the startup loader's fail-fast).
                var other => throw new ArgumentException(
                    $"Unrecognised backend '{other}'. Valid backends: mlx, cuda, ggml_metal, ggml_cuda, ggml_vulkan, ggml_cpu, cpu.",
                    nameof(backendStr)),
            };
        }

        private static long SafeGetFileSize(string path)
        {
            if (string.IsNullOrEmpty(path))
                return 0;
            try
            {
                var fi = new FileInfo(path);
                if (fi.LinkTarget is not null && fi.ResolveLinkTarget(true) is FileInfo target)
                    fi = target;
                return fi.Exists ? fi.Length : 0;
            }
            catch
            {
                return 0;
            }
        }
    }
}
