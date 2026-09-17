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
using System.Diagnostics;
using System.IO;
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
                ModelBase.Create(path, backend, tpGroup: tpGroup, draftModelPath: draftPath))
        {
        }

        /// <summary>Test seam: <paramref name="createModel"/> stands in for
        /// <see cref="ModelBase.Create(string, BackendType, int, ITensorParallelGroup, string)"/>.</summary>
        internal ModelLifecycleService(ILogger logger, Func<string, BackendType, ITensorParallelGroup, string, ModelBase> createModel)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
            _createModel = createModel;
        }

        public bool IsLoaded => _model != null;

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
        public ModelBase Model => _model;
        public BackendType Backend => _backend;

        public bool IsModelAlreadyLoaded(string modelName)
        {
            return _model != null && string.Equals(LoadedModelName, modelName, StringComparison.OrdinalIgnoreCase);
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
            using (var gguf = new GgufFile(modelPath))
                gguf.ThrowIfTruncated();

            // A missing projector file is skipped by the load itself, but one
            // that exists must parse.
            if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
            {
                using var mmProj = new GgufFile(mmProjPath);
                mmProj.ThrowIfTruncated();
            }
        }

        /// <summary>
        /// Release the loaded model (and its projector and draft head) without loading
        /// another. The engine and diffusion scheduler that were built on top of the
        /// model must already be torn down: <see cref="ModelService.UnloadModel"/> does
        /// that ordering, which is why this is the seam it calls rather than a public API
        /// on its own.
        /// </summary>
        internal void Unload() => UnloadCurrentModel();

        private void UnloadCurrentModel()
        {
            string previousModel = LoadedModelName;
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            DraftHeadActivationError = null;
            DraftHeadRefusedByModel = false;

            if (!string.IsNullOrEmpty(previousModel))
            {
                _logger.LogInformation(LogEventIds.ModelUnloaded,
                    "Unloaded previous model {PreviousModel}", previousModel);
            }
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
                tpGroup = TensorParallelGroupFactory?.Invoke(_backend);

                // Block drafters go to the factory rather than being attached
                // afterwards like Gemma 4's draft head: their weights have to be
                // counted by the layer split and uploaded with the trunk. The
                // shared --draft-model variables are authoritative; the older
                // DSpark-only variable remains an env compatibility fallback.
                string blockDraftPath = SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath();
                if (string.IsNullOrWhiteSpace(blockDraftPath))
                    blockDraftPath = Environment.GetEnvironmentVariable("TS_DSV4_DSPARK");
                _model = _createModel(modelPath, _backend, tpGroup, blockDraftPath);

                // A worker node (--tp-node-id > 0) spends its life blocked in a
                // mirror loop and cannot also serve HTTP requests, so the server
                // only takes the driver role (node 0). Fail fast with the
                // supported topology instead of hanging on the first inference.
                if (_model.IsDistributedWorker)
                {
                    throw new ModelLoadRefusedException(
                        "TensorSharp.Server cannot run as a distributed tensor-parallel WORKER node " +
                        "(--tp-node-id > 0 / TENSORSHARP_TP_NODE_ID > 0). Run this server as node 0 (the driver) " +
                        "and start each worker node with TensorSharp.Cli, e.g.: " +
                        "TensorSharp.Cli --model <same.gguf> --backend <same> --tp <localGpus> --tp-node-id <N> --tp-peers <same list>.");
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
                if (!SpeculativeDraftHeadLoader.TryAttachConfiguredDraftHead(_model, out string draftError))
                {
                    DraftHeadActivationError = draftError;
                    DraftHeadRefusedByModel =
                        _model is TensorSharp.Runtime.Speculative.ISpeculativeTarget { SpeculationRefusal: not null };
                    _logger.LogWarning("{Error}; speculation disabled.", draftError);
                }
                else if (SpeculativeDraftHeadLoader.ConfiguredDraftHeadPath() is { } attachedDraft)
                {
                    _logger.LogInformation("Loaded draft head {Draft} (HasDraftHead=True)",
                        Path.GetFileName(attachedDraft));
                }

                loadSw.Stop();
                long modelBytes = SafeGetFileSize(modelPath);
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

        public void Dispose()
        {
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
        }

        private void LoadEncoders(string mmProjPath)
        {
            _model?.MultimodalInjector.LoadProjectors(mmProjPath);
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
                return fi.Exists ? fi.Length : 0;
            }
            catch
            {
                return 0;
            }
        }
    }
}
