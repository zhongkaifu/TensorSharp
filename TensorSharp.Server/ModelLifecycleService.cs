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
