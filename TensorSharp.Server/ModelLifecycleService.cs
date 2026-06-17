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

        private ModelBase _model;
        private string _loadedModelPath;
        private string _loadedMmProjPath;
        private BackendType _backend;

        public ModelLifecycleService(ILogger logger)
        {
            _logger = logger ?? Microsoft.Extensions.Logging.Abstractions.NullLogger.Instance;
        }

        public bool IsLoaded => _model != null;

        /// <summary>
        /// When the operator explicitly named an MTP draft via
        /// <c>--mtp-draft-model</c> (<c>TS_MTP_DRAFT_MODEL</c>) but it could not be
        /// activated on the loaded target (missing file, wrong architecture, an
        /// incompatible draft, or an incomplete GGUF), this holds a
        /// human-readable reason. It is <c>null</c> when no draft was requested or
        /// when the draft loaded successfully (<c>HasMtp</c>). The startup loader
        /// promotes a non-null value to a fail-fast error so an explicit but
        /// unusable draft can't silently leave speculation disabled — matching the
        /// fail-fast contract the rest of startup configuration follows. Runtime
        /// (Web UI) model switches read the warning log but do not fail.
        /// </summary>
        public string MtpDraftActivationError { get; private set; }

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

            string previousModel = LoadedModelName;
            _model?.Dispose();
            _model = null;
            _loadedModelPath = null;
            _loadedMmProjPath = null;
            MtpDraftActivationError = null;

            if (!string.IsNullOrEmpty(previousModel))
            {
                _logger.LogInformation(LogEventIds.ModelUnloaded,
                    "Unloaded previous model {PreviousModel}", previousModel);
            }

            _backend = ResolveBackend(backendStr);

            var loadSw = Stopwatch.StartNew();
            try
            {
                _model = ModelBase.Create(modelPath, _backend);
                _loadedModelPath = modelPath;

                if (!string.IsNullOrEmpty(mmProjPath) && File.Exists(mmProjPath))
                {
                    LoadEncoders(mmProjPath);
                    _loadedMmProjPath = mmProjPath;
                }

                // Gemma 4 MTP: the draft head ships as a SEPARATE GGUF
                // (gemma4-assistant). Load it onto the target so HasMtp turns on
                // and --mtp-spec engages. (Qwen3.6 embeds its NextN block in the
                // trunk and needs no separate file.) MtpDraftActivationError was
                // already cleared above before the model was (re)created.
                string mtpDraftPath = Environment.GetEnvironmentVariable("TS_MTP_DRAFT_MODEL");
                if (!string.IsNullOrEmpty(mtpDraftPath))
                {
                    if (_model is Gemma4Model g4)
                    {
                        if (!File.Exists(mtpDraftPath))
                        {
                            MtpDraftActivationError = $"MTP draft model file not found: {mtpDraftPath}";
                            _logger.LogWarning("{Error}; speculation disabled.", MtpDraftActivationError);
                        }
                        else
                        {
                            try
                            {
                                g4.LoadMtpDraftWeights(mtpDraftPath);
                                if (g4.HasMtp)
                                {
                                    _logger.LogInformation("Loaded Gemma 4 MTP draft head {Draft} (HasMtp=True)",
                                        Path.GetFileName(mtpDraftPath));
                                }
                                else
                                {
                                    MtpDraftActivationError =
                                        $"MTP draft '{Path.GetFileName(mtpDraftPath)}' loaded but is incomplete (required draft tensors missing).";
                                    _logger.LogWarning("{Error}; speculation disabled.", MtpDraftActivationError);
                                }
                            }
                            catch (Exception mtpEx)
                            {
                                MtpDraftActivationError =
                                    $"Failed to load MTP draft '{Path.GetFileName(mtpDraftPath)}': {mtpEx.Message}";
                                _logger.LogWarning(mtpEx, "Failed to load MTP draft {Path}; speculation disabled.", mtpDraftPath);
                            }
                        }
                    }
                    else
                    {
                        // A draft GGUF was named but the loaded model's architecture
                        // does not consume a separate draft file (e.g. Qwen3.6 embeds
                        // its NextN block in the trunk). Record it so the operator
                        // isn't left wondering why their --mtp-draft-model was ignored.
                        MtpDraftActivationError =
                            $"--mtp-draft-model was given but the loaded model architecture " +
                            $"'{Architecture ?? "unknown"}' does not use a separate MTP draft GGUF.";
                        _logger.LogWarning("{Error}; speculation disabled.", MtpDraftActivationError);
                    }
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
                _logger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Failed to load model {ModelFile} on backend {Backend} after {ElapsedMs:F1} ms",
                    Path.GetFileName(modelPath), backendStr ?? "(default)", loadSw.Elapsed.TotalMilliseconds);
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
                "cpu" => BackendType.Cpu,
                _ => BackendType.GgmlCpu
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
