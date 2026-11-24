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
using System.Collections.Concurrent;
using System.Diagnostics;
using TensorSharp.Server;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Runtime.Telemetry;
using Xunit;

namespace InferenceWeb.Tests.Telemetry;

public class ModelLoadTelemetryTests
{
    [Theory]
    [InlineData(DraftHeadKind.Block, "models/Qwen3.8-27B-DFlash2-Q4_K_M.gguf", "block", "Qwen3.8-27B-DFlash2-Q4_K_M.gguf")]
    [InlineData(DraftHeadKind.PerToken, null, "per_token", null)]
    [InlineData(DraftHeadKind.None, null, null, null)]
    public void TheModelLoadSpanNamesTheDrafter(DraftHeadKind kind, string? attachedFile, string? expectedKind, string? expectedFile)
    {
        string model = "draft-head-test-" + Guid.NewGuid().ToString("N")[..8] + ".gguf";
        var loads = new ConcurrentQueue<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = s => s.Name == TensorSharpTelemetry.ModelsSourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a =>
            {
                if (a.DisplayName == "model.load" && Equals(a.GetTagItem(TelemetryAttributes.ModelFile), model))
                    loads.Enqueue(a);
            },
        };
        ActivitySource.AddActivityListener(listener);

        using (var telemetry = ModelLoadTelemetry.StartLoad("models/" + model, mmProjPath: null, "ggml_cpu"))
        {
            telemetry.Loaded("qwen35", "ggml_cpu");
            telemetry.DraftHead(kind, attachedFile);
        }

        var span = Assert.Single(loads);
        Assert.Equal(expectedKind, span.GetTagItem("tensorsharp.model.draft_head"));
        Assert.Equal(expectedFile, span.GetTagItem("tensorsharp.model.draft_file"));
    }
}
