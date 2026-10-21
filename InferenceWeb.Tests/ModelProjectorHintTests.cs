// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// A model that places its weights against device memory (qwen4exp's layer split,
// issue #256) has to know, while it is being BUILT, that the host will load a
// vision projector next to it afterwards: the projector's F32 tower lands on GPU 0.
// These pin how that hint travels.
using System;
using System.IO;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class ModelProjectorHintTests
{
    [Fact]
    public void CreateContext_CarriesTheProjectorThroughWith()
    {
        var context = new ModelCreateContext("model.gguf", BackendType.GgmlCuda, probe: null,
            draftModelPath: null, layerSplitDegree: 1, projectorPath: "mmproj.gguf");
        Assert.Equal("mmproj.gguf", context.ProjectorPath);
        var resolved = context.With(1, null, 2);
        Assert.Equal("mmproj.gguf", resolved.ProjectorPath);
        Assert.Equal(2, resolved.LayerSplitDegree);
    }

    [Fact]
    public void CreateContext_DefaultsToNoProjector()
        => Assert.Null(new ModelCreateContext("model.gguf", BackendType.GgmlCuda, probe: null).ProjectorPath);

    [Fact]
    public void ExpectProjector_ScopesNestAndRestore()
    {
        var current = typeof(ModelBase).GetField("s_expectedProjector",
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        Assert.NotNull(current);
        string Read() => ((System.Threading.AsyncLocal<string>)current.GetValue(null)).Value;

        Assert.Null(Read());
        using (ModelBase.ExpectProjector("outer.gguf"))
        {
            Assert.Equal("outer.gguf", Read());
            using (ModelBase.ExpectProjector("  "))
                Assert.Null(Read());
            Assert.Equal("outer.gguf", Read());
            var inner = ModelBase.ExpectProjector("inner.gguf");
            inner.Dispose();
            inner.Dispose(); // idempotent
            Assert.Equal("outer.gguf", Read());
        }
        Assert.Null(Read());
    }

    [Fact]
    public void Cli_ExpectsTheNamedProjectorOrNone()
    {
        string missingModel = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + ".gguf");
        string projector = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + "-mmproj.gguf");
        File.WriteAllBytes(projector, new byte[16]);
        try
        {
            // A named projector FILE is expected even when the model cannot be probed.
            Assert.Equal(projector, TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, projector, false, null, null, null));
            // ...but a name that is no file cannot be priced up front.
            Assert.Null(TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, projector + ".missing", false, null, null, null));
            Assert.Null(TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, null, true, "a.png", null, null));
            // No projector named and no input that needs a vision tower: nothing will load.
            Assert.Null(TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, null, false, null, null, null));
            Assert.Null(TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, null, false, null, "a.wav", null));
            // An unreadable model cannot name its companion; the run still loads it later.
            Assert.Null(TensorSharp.Cli.Program.ExpectedProjectorPath(missingModel, null, false, "a.png", null, null));
        }
        finally
        {
            File.Delete(projector);
        }
    }
}
