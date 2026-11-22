// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>
/// <c>--nvfp4-prompt-activations fp4|q8</c>: one spelling, read by the CLI and the server, captured by a model when it is made
/// and carried by its programs as <see cref="GraphProgram.Q8Nvfp4Activations"/> (a program attribute, part of the
/// signature). The products' arithmetic is held in CudaNvfp4MatmulTests and Lfm3RuntimeTests.
/// </summary>
public sealed class Nvfp4PromptActivationsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("nvfp4-activations").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    [Fact]
    public void TheOptionTakesFp4OrQ8()
    {
        Assert.True(Nvfp4PromptActivationsConfig.TryParse("fp4", out var a) && a == Nvfp4PromptActivations.Fp4);
        Assert.True(Nvfp4PromptActivationsConfig.TryParse("q8", out a) && a == Nvfp4PromptActivations.Q8);
        Assert.True(Nvfp4PromptActivationsConfig.TryParse("Q8", out a) && a == Nvfp4PromptActivations.Q8);
        foreach (string other in new[] { null, "", "int8", "q8_0", "w4a8", "fp8" })
            Assert.False(Nvfp4PromptActivationsConfig.TryParse(other, out _));
        Assert.Equal("q8", Nvfp4PromptActivationsConfig.Spell(Nvfp4PromptActivations.Q8));
        Assert.Equal("fp4", Nvfp4PromptActivationsConfig.Spell(Nvfp4PromptActivations.Fp4));
        Assert.Equal("--nvfp4-prompt-activations", Nvfp4PromptActivationsConfig.OptionName);
    }

    [Fact]
    public void TheServerReadsTheLastOneAndRefusesAnyOtherValue()
    {
        Assert.Null(ServerOptionsBuilder.ReadNvfp4PromptActivationsCliFlag(new[] { "--model", "m.gguf" }));
        Assert.Equal(Nvfp4PromptActivations.Q8,
            ServerOptionsBuilder.ReadNvfp4PromptActivationsCliFlag(new[] { "--nvfp4-prompt-activations", "q8" }));
        // A --config file's tokens come first, the command line's after them: the command line wins.
        Assert.Equal(Nvfp4PromptActivations.Fp4, ServerOptionsBuilder.ReadNvfp4PromptActivationsCliFlag(
            new[] { "--nvfp4-prompt-activations", "q8", "--model", "m.gguf", "--nvfp4-prompt-activations", "fp4" }));
        var refused = Assert.Throws<ArgumentException>(() =>
            ServerOptionsBuilder.ReadNvfp4PromptActivationsCliFlag(new[] { "--nvfp4-prompt-activations", "int8" }));
        Assert.Contains("Valid: fp4, q8", refused.Message);
        // The option parser accepts it (consumed by the earlier pass, not an unknown argument).
        ServerOptionsBuilder.Build(new[] { "--nvfp4-prompt-activations", "q8" }, _dir);
    }

    [Fact]
    public void BothUsagePagesDocumentTheOption()
    {
        var server = new StringWriter();
        TensorSharp.Server.Host.Hosting.ServerUsage.PrintUsage(server);
        Assert.Contains("--nvfp4-prompt-activations", server.ToString());
        var cli = new StringWriter();
        TensorSharp.Cli.CliUsage.PrintUsage(cli);
        Assert.Contains("--nvfp4-prompt-activations", cli.ToString());
    }

    [Fact]
    public void TheAttributeIsPartOfTheProgramAndItsSignature()
    {
        GraphProgram Program(bool q8)
        {
            var g = new GraphProgramBuilder(record: true);
            if (q8) g.MarkQ8Nvfp4Activations();
            int x = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { 4, 2 }, IntPtr.Zero, 32);
            g.Output(g.Op(GraphOp.Scale, stackalloc[] { x }, default, stackalloc[] { 2f }), 8);
            return g.ToProgram(replayable: true, reorder: false);
        }
        GraphProgram plain = Program(false), q8 = Program(true);
        Assert.False(plain.Q8Nvfp4Activations);
        Assert.True(q8.Q8Nvfp4Activations);
        Assert.NotEqual(plain.Signature, q8.Signature);

        var late = new GraphProgramBuilder(record: true);
        late.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { 4 }, IntPtr.Zero, 16);
        Assert.Throws<InvalidOperationException>(late.MarkQ8Nvfp4Activations);
    }

    [Fact]
    public void AModelCapturesTheChoiceWhenItIsMade()
    {
        string path = new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3.gguf"));
        using (ModelBase model = ModelBase.Create(path, BackendType.Cpu))
            Assert.Equal(Nvfp4PromptActivations.Fp4, model.Nvfp4PromptActivations);
        using (Nvfp4PromptActivationsConfig.ForCurrentFlowInTests(Nvfp4PromptActivations.Q8))
        using (ModelBase model = ModelBase.Create(path, BackendType.Cpu))
            Assert.Equal(Nvfp4PromptActivations.Q8, model.Nvfp4PromptActivations);
    }
}
