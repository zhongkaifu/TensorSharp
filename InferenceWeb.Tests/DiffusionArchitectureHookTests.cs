// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// What the architecture layer gives a diffusion family on the shared diffusion runtime: its serving contract
// (ServingImplementation.SharedDiffusionRuntime) and the registry's safetensors detector, which finds the family of a
// model file shipped as safetensors from its header alone (no family special case in the loader or the hosts).
using System.Buffers.Binary;
using System.Text;
using System.Text.Json;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class DiffusionArchitectureHookTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-diffusion-hook-" + Guid.NewGuid().ToString("N"));

    public DiffusionArchitectureHookTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly BackendType[] GgmlImage = { BackendType.GgmlCuda, BackendType.GgmlMetal, BackendType.GgmlCpu };

    [Fact]
    public void ADiffusionContractHasNoBatchingNoExpertsAndTheRuntimesTensorParallelism()
    {
        ServingContract contract = ServingContract.SharedDiffusionRuntime(GgmlImage, tensorParallelBackends: new[] { BackendType.GgmlCuda });
        contract.Validate("diffusion-test");
        Assert.Equal(ServingImplementation.SharedDiffusionRuntime, contract.Implementation);
        Assert.False(contract.Autoregressive);
        Assert.Equal(SupportKind.NotApplicable, contract.ContinuousBatching.Kind);
        Assert.Equal(ServingContract.DiffusionJobsReason, contract.ContinuousBatching.Reason);
        Assert.Equal(SupportKind.NotApplicable, contract.ExpertOffload.Kind);
        Assert.True(contract.TensorParallel.IsSupportedOn(BackendType.GgmlCuda));
        // Layer split and the hybrid are gaps on the multi-device backend; TP is a gap nowhere else it could run.
        Assert.Equal(new[] { "HybridParallel@ggml_cuda", "LayerSplit@ggml_cuda" },
            contract.Gaps().Select(g => g.ToString()).OrderBy(s => s, StringComparer.Ordinal));
    }

    [Fact]
    public void ADiffusionContractWithoutShardableBlocksDeclaresTheTensorParallelGap()
    {
        ServingContract contract = ServingContract.SharedDiffusionRuntime(GgmlImage);
        contract.Validate("diffusion-test");
        Assert.Contains("TensorParallel@ggml_cuda", contract.Gaps().Select(g => g.ToString()));
        Assert.Contains("shardable", contract.TensorParallel.GapReason);
        // On single-device backends alone the placement mechanisms do not apply at all.
        ServingContract mac = ServingContract.SharedDiffusionRuntime(new[] { BackendType.GgmlMetal, BackendType.Mlx });
        mac.Validate("diffusion-test");
        Assert.Equal(SupportKind.NotApplicable, mac.TensorParallel.Kind);
        Assert.Empty(mac.Gaps());
    }

    [Fact]
    public void RegistrationRefusesADiffusionContractThatBatchesOrDecodes()
    {
        ServingContract good = ServingContract.SharedDiffusionRuntime(GgmlImage);
        var batching = new ServingContract
        {
            Implementation = ServingImplementation.SharedDiffusionRuntime, Backends = GgmlImage, Autoregressive = false,
            HasRoutedExperts = false, ContinuousBatching = MechanismSupport.On(GgmlImage), TensorParallel = good.TensorParallel,
            LayerSplit = good.LayerSplit, HybridParallel = good.HybridParallel, ExpertOffload = good.ExpertOffload,
        };
        Assert.Contains("continuous batching must be NotApplicable", Assert.Throws<InvalidOperationException>(() => batching.Validate("x")).Message);
        var decoding = new ServingContract
        {
            Implementation = ServingImplementation.SharedDiffusionRuntime, Backends = GgmlImage, Autoregressive = true,
            HasRoutedExperts = false, ContinuousBatching = MechanismSupport.On(GgmlImage), TensorParallel = good.TensorParallel,
            LayerSplit = good.LayerSplit, HybridParallel = good.HybridParallel, ExpertOffload = good.ExpertOffload,
        };
        Assert.Contains("cannot decode autoregressively", Assert.Throws<InvalidOperationException>(() => decoding.Validate("x")).Message);
    }

    /// <summary>A safetensors file of F32 tensors (zeros) with the given names and metadata.</summary>
    private string WriteSafetensors(string name, IReadOnlyDictionary<string, long[]> tensors, IReadOnlyDictionary<string, string> metadata = null)
    {
        var header = new Dictionary<string, object>();
        if (metadata != null) header["__metadata__"] = metadata;
        long offset = 0;
        foreach (var (tensor, shape) in tensors)
        {
            long bytes = shape.Aggregate(4L, (a, d) => a * d);
            header[tensor] = new Dictionary<string, object> { ["dtype"] = "F32", ["shape"] = shape, ["data_offsets"] = new[] { offset, offset + bytes } };
            offset += bytes;
        }
        byte[] json = Encoding.UTF8.GetBytes(JsonSerializer.Serialize(header));
        string path = Path.Combine(_dir, name);
        using var file = File.Create(path);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(length, (ulong)json.Length);
        file.Write(length);
        file.Write(json);
        file.Write(new byte[offset]);
        return path;
    }

    [Fact]
    public void TheRegistryOffersASafetensorsModelFileToTheFamiliesDetectors()
    {
        string id = "safetensors-detector-" + Guid.NewGuid().ToString("N");
        string marker = id + ".pdd_heads";
        SafetensorsFile seen = null;
        var descriptor = new ModelArchitectureDescriptor
        {
            Id = id,
            Aliases = new[] { id },
            Factory = _ => throw new InvalidOperationException("not constructed by this test"),
            Serving = ServingContract.SharedDiffusionRuntime(GgmlImage),
            DetectSafetensors = file => { seen = file; return file.HasTensor(marker) && file.Metadata.ContainsKey("config"); },
        };
        ModelArchitectureRegistry.Register(descriptor);

        string ours = WriteSafetensors("denoiser.safetensors",
            new Dictionary<string, long[]> { [marker] = new long[] { 6, 4 }, ["blocks.0.w"] = new long[] { 4, 4 } },
            new Dictionary<string, string> { ["config"] = "{}" });
        Assert.Same(descriptor, ModelArchitectureRegistry.ResolveSafetensors(ours));
        Assert.NotNull(seen);
        Assert.Equal(new long[] { 6, 4 }, seen.TensorShape(marker));

        // Another family's file, a file that is not safetensors, a corrupt header and a missing file: no claim.
        string other = WriteSafetensors("vae.safetensors", new Dictionary<string, long[]> { ["decoder.w"] = new long[] { 2 } });
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(other));
        string corrupt = Path.Combine(_dir, "broken.safetensors");
        File.WriteAllBytes(corrupt, new byte[] { 255, 255, 255, 255, 255, 255, 255, 127, (byte)'{' });
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(corrupt));
        string json = Path.Combine(_dir, "garbage.safetensors");
        File.WriteAllBytes(json, Concat(BitConverter.GetBytes(5UL), Encoding.UTF8.GetBytes("{nope")));
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(json));
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(Path.Combine(_dir, "missing.safetensors")));
        Assert.Null(ModelArchitectureRegistry.ResolveSafetensors(Path.ChangeExtension(ours, ".gguf")));
        Assert.True(ModelArchitectureRegistry.IsSafetensorsPath("A.SAFETENSORS"));
        Assert.False(ModelArchitectureRegistry.IsSafetensorsPath("a.gguf"));
    }

    [Fact]
    public void ACreateContextKeepsItsSafetensorsProbeThroughAPlacementChange()
    {
        string path = WriteSafetensors("probe.safetensors", new Dictionary<string, long[]> { ["w"] = new long[] { 2, 2 } });
        using var probe = new SafetensorsFile(path);
        var context = new ModelCreateContext(path, BackendType.GgmlCpu, probe: null) { SafetensorsProbe = probe };
        ModelCreateContext changed = context.With(1, null, 1);
        Assert.Same(probe, changed.SafetensorsProbe);
        Assert.Null(changed.Probe);
    }

    private static byte[] Concat(byte[] a, byte[] b) => a.Concat(b).ToArray();
}
