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
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public class StartupModelLoaderTests
{
    [Fact]
    public void LoadIfConfigured_UsesResolvedDefaultBackendWhenConfiguredBackendFallsBack()
    {
        var options = new ServerHostingOptions(
            startupModelPath: Path.Combine(Path.GetTempPath(), "missing-startup-model.gguf"),
            startupMmProjPath: null,
            defaultBackend: "cuda",
            supportedBackends: new[] { new BackendOption("cuda", "CUDA (cuBLAS GPU)") },
            defaultWebMaxTokens: 20000,
            maxTextFileChars: 8000,
            uploadDirectory: Path.GetTempPath(),
            logDirectory: Path.GetTempPath(),
            fileLoggingEnabled: false,
            defaultSamplingConfig: new SamplingConfig());

        using var modelService = new ModelService(NullLogger<ModelService>.Instance);

        var ex = Assert.Throws<FileNotFoundException>(() =>
            StartupModelLoader.LoadIfConfigured(
                options,
                modelService,
                configuredBackendInput: "ggml_cuda",
                NullLogger.Instance));

        Assert.Contains("missing-startup-model.gguf", ex.Message);
    }
}
