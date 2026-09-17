// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// bartowski's Mistral-Small-3.1-24B-Instruct-2503 GGUF (the release catalog's
// mistral3 entry) declares general.architecture = llama and keeps every
// hyperparameter under llama.*, because it was converted before llama.cpp had a
// mistral3 architecture. The registry only knew the alias "mistral3", so the file
// died at load with "Unsupported architecture: llama". These tests pin the fix:
// such a file is served by Mistral3Model with its protocol id intact, while every
// other llama-labelled neighbour stays refused with an explanation.
using System;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class LlamaLabelledMistral3RoutingTests : IDisposable
{
    private readonly string _dir;

    public LlamaLabelledMistral3RoutingTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-llama-mistral3-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Write(DenseDecoderSyntheticModelBuilder builder, string name)
        => builder.Write(Path.Combine(_dir, name));

    private static ModelArchitectureDescriptor Resolve(string path)
    {
        using var probe = new GgufFile(path);
        return ModelArchitectureRegistry.Resolve(probe.GetString("general.architecture"), probe);
    }

    [Fact]
    public void LlamaIsNotAnAlias_ItIsClaimedPerFile()
    {
        Assert.False(ModelArchitectureRegistry.TryGet("llama", out _));
    }

    [Fact]
    public void LlamaLabelledTekkenMistral_ResolvesToMistral3()
    {
        string path = Write(new DenseDecoderSyntheticModelBuilder(), "mistral-small-llama.gguf");
        Assert.Equal("mistral3", Resolve(path).Id);
    }

    [Fact]
    public void LlamaLabelledMistral_LoadsAsMistral3_AndMatchesTheMistral3LabelledFile()
    {
        string llamaPath = Write(new DenseDecoderSyntheticModelBuilder(), "llama-labelled.gguf");
        string mistralPath = Write(new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }, "mistral3-labelled.gguf");

        using var relabelled = ModelBase.Create(llamaPath, BackendType.Cpu);
        using var native = ModelBase.Create(mistralPath, BackendType.Cpu);

        Assert.IsType<Mistral3Model>(relabelled);
        // The protocol id, not the file's label: chat rendering, output parsing and
        // capability lookups are all keyed on it.
        Assert.Equal("mistral3", relabelled.Config.Architecture);
        // Hyperparameters were read under llama.*.
        Assert.Equal(DenseDecoderSyntheticModelBuilder.NumBlocks, relabelled.Config.NumLayers);
        Assert.Equal(DenseDecoderSyntheticModelBuilder.Hidden, relabelled.Config.HiddenSize);
        Assert.Equal(DenseDecoderSyntheticModelBuilder.NumKvHeads, relabelled.Config.NumKVHeads);
        Assert.Equal(1_000_000f, relabelled.Config.RopeBase);
        Assert.Equal(4096, relabelled.Config.DeclaredContextLength);
        Assert.NotNull(ChatProtocolRegistry.For(relabelled.Config.Architecture));

        int[] prompt = { 65, 66, 67, 68, 69, 70, 71, 72, 73 };
        float[] a = (float[])relabelled.ForwardRefill(prompt).Clone();
        float[] b = (float[])native.ForwardRefill(prompt).Clone();
        Assert.Equal(b, a);

        float[] a2 = (float[])relabelled.Forward(new[] { 74 }).Clone();
        float[] b2 = (float[])native.Forward(new[] { 74 }).Clone();
        Assert.Equal(b2, a2);
    }

    public static TheoryData<string, DenseDecoderSyntheticModelBuilder> Neighbours => new()
    {
        { "llama3-rope-freqs", new DenseDecoderSyntheticModelBuilder { IncludeRopeFreqs = true } },
        { "sentencepiece-or-other-pre", new DenseDecoderSyntheticModelBuilder { PreTokenizer = "llama-bpe" } },
        { "no-mistral-control-tokens", new DenseDecoderSyntheticModelBuilder { IncludeMistralControlTokens = false } },
        { "mixtral-experts", new DenseDecoderSyntheticModelBuilder { ExpertCount = 8 } },
        { "qk-norm", new DenseDecoderSyntheticModelBuilder { IncludeQkNorms = true } },
        { "llama3-rope-scaling", new DenseDecoderSyntheticModelBuilder { RopeScalingType = "llama3" } },
    };

    [Theory]
    [MemberData(nameof(Neighbours))]
    public void LlamaLabelledNeighbours_AreRefusedWithWhatIsAccepted(string name, DenseDecoderSyntheticModelBuilder builder)
    {
        string path = Write(builder, name + ".gguf");
        var error = Assert.Throws<NotSupportedException>(() => Resolve(path));
        Assert.StartsWith("Unsupported architecture: llama.", error.Message);
        Assert.Contains("mistral3 (", error.Message);
        Assert.Contains("Tekken", error.Message);
    }

    [Fact]
    public void RelabelRecogniserOnlyAnswersForTheLlamaLabel()
    {
        string path = Write(new DenseDecoderSyntheticModelBuilder { Architecture = "not-a-registered-arch" }, "unregistered.gguf");
        using var probe = new GgufFile(path);
        Assert.False(Mistral3Architecture.IsLlamaLabelledMistral3("not-a-registered-arch", probe));
        Assert.Throws<NotSupportedException>(() => ModelArchitectureRegistry.Resolve("not-a-registered-arch", probe));
    }

    [Fact]
    public void Mistral3ProjectorHints_FindBartowskisCompanionName()
    {
        string model = Write(new DenseDecoderSyntheticModelBuilder(), "mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf");
        string mmproj = Path.Combine(_dir, "mmproj-mistralai_Mistral-Small-3.1-24B-Instruct-2503-f16.gguf");
        File.WriteAllBytes(mmproj, new byte[] { 0 });
        Assert.Equal(mmproj, ModelArchitectureRegistry.FindCompanionProjector("mistral3", model));
    }

    [Fact]
    public void RelabelRecogniserWithoutDescription_IsRejectedAtRegistration()
    {
        string id = "test-relabel-" + Guid.NewGuid().ToString("N");
        var bad = new ModelArchitectureDescriptor
        {
            Id = id,
            Aliases = new[] { id },
            Factory = _ => null,
            RecognizeRelabelledFile = (_, _) => false,
        };
        Assert.Throws<InvalidOperationException>(() => ModelArchitectureRegistry.Register(bad));
    }
}
