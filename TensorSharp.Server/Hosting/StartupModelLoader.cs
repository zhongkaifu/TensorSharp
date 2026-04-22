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
using System.IO;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Pre-loads the model named on the command line into the singleton
    /// <see cref="ModelService"/> before the host starts accepting requests.
    /// Throws when configuration is internally inconsistent (e.g. the requested
    /// backend isn't available) so the process fails fast rather than serving
    /// 4xx/5xx responses for every request.
    /// </summary>
    internal static class StartupModelLoader
    {
        public static void LoadIfConfigured(
            ServerHostingOptions options,
            ModelService modelService,
            string configuredBackendInput,
            ILogger logger)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));
            if (modelService == null) throw new ArgumentNullException(nameof(modelService));
            if (logger == null) throw new ArgumentNullException(nameof(logger));

            if (string.IsNullOrWhiteSpace(options.StartupModelPath))
            {
                logger.LogInformation(LogEventIds.HostConfiguration,
                    "No startup model configured. Launch with --model <path.gguf> --backend <type> [--mmproj <path>] [--max-tokens 20000] to use the Web UI.");
                return;
            }

            if (!BackendSelector.TryResolveSupportedBackend(options, configuredBackendInput, out string startupBackend, out string startupBackendError))
                throw new InvalidOperationException(startupBackendError);

            if (!File.Exists(options.StartupModelPath))
                throw new FileNotFoundException($"Configured model file not found: {options.StartupModelPath}", options.StartupModelPath);

            if (!string.IsNullOrWhiteSpace(options.StartupMmProjPath) && !File.Exists(options.StartupMmProjPath))
                throw new FileNotFoundException($"Configured mmproj file not found: {options.StartupMmProjPath}", options.StartupMmProjPath);

            modelService.LoadModel(options.StartupModelPath, options.StartupMmProjPath, startupBackend);

            logger.LogInformation(LogEventIds.ModelLoadCompleted,
                "Startup model loaded: {Model} architecture={Architecture} backend={Backend} mmproj={MmProj}",
                modelService.LoadedModelName,
                modelService.Architecture ?? "unknown",
                modelService.LoadedBackend,
                modelService.LoadedMmProjName ?? "(none)");
        }
    }
}
