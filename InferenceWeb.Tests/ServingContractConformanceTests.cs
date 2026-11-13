// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The enforcement half of docs/design/unified-execution.md:
//   * every family declares how it serves (ModelArchitectureDescriptor.Serving, required);
//   * the declared gaps equal the list below, in both directions - a family cannot add a
//     gap (or ship a new one) without this file changing in review, and a closed gap must
//     be deleted here, so the list is always the true support matrix;
//   * a family that is not one of the legacy bespoke implementations must be built on the
//     shared decoder runtime (which provides every mechanism), and must register a fixture
//     in SharedRuntimeConformanceTests.Families (or a dedicated suite in
//     SharedRuntimeConformanceTests.DedicatedSuites) so its claims are proven, not declared.
using TensorSharp.Models.Architecture;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]   // one test sets the process-wide expert-offload policy
public sealed class ServingContractConformanceTests
{
    /// <summary>
    /// Families that implement their mechanisms themselves. CLOSED: a new family goes on the
    /// shared runtime. Moving one of these to the runtime removes it from this list.
    /// </summary>
    internal static readonly HashSet<string> LegacyBespokeFamilies = new(StringComparer.Ordinal)
    {
        "qwen4exp", "gemma4", "gptoss", "nemotron_h", "muse-glimmer", "muse-spark",
        "deepseek4", "deepseek41", "glm-dsa", "glm5next", "minimax-m3",
        "diffusion-gemma", "qwen_image", "wan", "ltx-2.5", "minimax-h3", "minimax-h3-image",
    };

    /// <summary>
    /// Every declared gap, as "family: Mechanism@backend" (Mechanism+batching@backend: the
    /// placement works there, but concurrent requests stop being batched under it). This is
    /// the support matrix of record; shrink it by closing gaps, never grow it.
    /// </summary>
    internal static readonly string[] KnownGaps =
    {
        "deepseek41: ContinuousBatching@cpu",
        "deepseek41: ExpertOffload@cuda",
        "deepseek41: HybridParallel@cuda",
        "deepseek41: HybridParallel@ggml_cuda",
        "deepseek41: TensorParallel@cuda",
        "deepseek41: HybridParallel@ggml_vulkan",
        "deepseek41: LayerSplit@ggml_vulkan",
        "deepseek41: TensorParallel@ggml_vulkan",
        "deepseek4: ContinuousBatching@cpu",
        "deepseek4: ExpertOffload@cuda",
        "deepseek4: HybridParallel@cuda",
        "deepseek4: HybridParallel@ggml_cuda",
        "deepseek4: HybridParallel@ggml_vulkan",
        "deepseek4: TensorParallel@cuda",
        "deepseek4: TensorParallel@ggml_cuda",
        "deepseek4: TensorParallel@ggml_vulkan",
        "diffusion-gemma: ExpertOffload@cuda",
        "diffusion-gemma: ExpertOffload@mlx",
        "diffusion-gemma: HybridParallel@cuda",
        "diffusion-gemma: HybridParallel@ggml_cuda",
        "diffusion-gemma: HybridParallel@ggml_vulkan",
        "diffusion-gemma: LayerSplit@cuda",
        "diffusion-gemma: LayerSplit@ggml_cuda",
        "diffusion-gemma: LayerSplit@ggml_vulkan",
        "diffusion-gemma: TensorParallel@cuda",
        "diffusion-gemma: TensorParallel@ggml_cuda",
        "diffusion-gemma: TensorParallel@ggml_vulkan",
        "gemma4: ExpertOffload@cuda",
        "gemma4: ExpertOffload@mlx",
        "gemma4: HybridParallel@cuda",
        "gemma4: HybridParallel@ggml_cuda",
        "gemma4: HybridParallel@ggml_vulkan",
        "gemma4: LayerSplit@cuda",
        "gemma4: LayerSplit@ggml_cuda",
        "gemma4: LayerSplit@ggml_vulkan",
        "gemma4: TensorParallel+batching@cuda",
        "gemma4: TensorParallel+batching@ggml_cuda",
        "gemma4: TensorParallel+batching@ggml_vulkan",
        "glm-dsa: ContinuousBatching@cpu",
        "glm-dsa: HybridParallel@ggml_cuda",
        "glm-dsa: HybridParallel@ggml_vulkan",
        "glm5next: ContinuousBatching@cpu",
        "glm5next: ContinuousBatching@cuda",
        "glm5next: ExpertOffload@cuda",
        "glm5next: HybridParallel@cuda",
        "glm5next: HybridParallel@ggml_cuda",
        "glm5next: HybridParallel@ggml_vulkan",
        "glm5next: TensorParallel@cuda",
        "gptoss: ExpertOffload@cuda",
        "gptoss: ExpertOffload@mlx",
        "gptoss: HybridParallel@cuda",
        "gptoss: HybridParallel@ggml_cuda",
        "gptoss: HybridParallel@ggml_vulkan",
        "gptoss: LayerSplit@cuda",
        "gptoss: LayerSplit@ggml_cuda",
        "gptoss: LayerSplit@ggml_vulkan",
        "gptoss: TensorParallel+batching@cuda",
        "gptoss: TensorParallel+batching@ggml_cuda",
        "gptoss: TensorParallel+batching@ggml_vulkan",
        "ltx-2.5: HybridParallel@cuda",
        "ltx-2.5: HybridParallel@ggml_cuda",
        "ltx-2.5: LayerSplit@cuda",
        "ltx-2.5: LayerSplit@ggml_cuda",
        "ltx-2.5: TensorParallel@cuda",
        "ltx-2.5: TensorParallel@ggml_cuda",
        "minimax-h3: HybridParallel@cuda",
        "minimax-h3: HybridParallel@ggml_cuda",
        "minimax-h3: LayerSplit@cuda",
        "minimax-h3: LayerSplit@ggml_cuda",
        "minimax-h3: TensorParallel@cuda",
        "minimax-h3: TensorParallel@ggml_cuda",
        "minimax-h3-image: HybridParallel@ggml_cuda",
        "minimax-h3-image: LayerSplit@ggml_cuda",
        "minimax-h3-image: TensorParallel@ggml_cuda",
        "minimax-m3: HybridParallel@ggml_cuda",
        "muse-glimmer: ContinuousBatching@cpu",
        "muse-glimmer: ContinuousBatching@cuda",
        "muse-glimmer: ContinuousBatching@ggml_cpu",
        "muse-glimmer: ContinuousBatching@ggml_cuda",
        "muse-glimmer: ContinuousBatching@ggml_metal",
        "muse-glimmer: ContinuousBatching@ggml_vulkan",
        "muse-glimmer: ContinuousBatching@mlx",
        "muse-glimmer: HybridParallel@cuda",
        "muse-glimmer: HybridParallel@ggml_cuda",
        "muse-glimmer: HybridParallel@ggml_vulkan",
        "muse-glimmer: LayerSplit@cuda",
        "muse-glimmer: LayerSplit@ggml_cuda",
        "muse-glimmer: LayerSplit@ggml_vulkan",
        "muse-glimmer: TensorParallel@cuda",
        "muse-spark: HybridParallel@ggml_cuda",
        "muse-spark: HybridParallel@ggml_vulkan",
        "muse-spark: TensorParallel+batching@ggml_vulkan",
        "nemotron_h: ExpertOffload@cuda",
        "nemotron_h: ExpertOffload@ggml_metal",
        "nemotron_h: ExpertOffload@mlx",
        "nemotron_h: HybridParallel@cuda",
        "nemotron_h: HybridParallel@ggml_cuda",
        "nemotron_h: HybridParallel@ggml_vulkan",
        "nemotron_h: LayerSplit@cuda",
        "nemotron_h: LayerSplit@ggml_cuda",
        "nemotron_h: LayerSplit@ggml_vulkan",
        "nemotron_h: TensorParallel+batching@cuda",
        "nemotron_h: TensorParallel+batching@ggml_cuda",
        "nemotron_h: TensorParallel+batching@ggml_vulkan",
        "qwen4exp: ContinuousBatching@cpu",
        "qwen4exp: ExpertOffload@cuda",
        "qwen4exp: HybridParallel@cuda",
        "qwen4exp: HybridParallel@ggml_cuda",
        "qwen4exp: HybridParallel@ggml_vulkan",
        "qwen4exp: TensorParallel@cuda",
        "qwen_image: HybridParallel@ggml_cuda",
        "qwen_image: HybridParallel@ggml_vulkan",
        "qwen_image: LayerSplit@ggml_cuda",
        "qwen_image: LayerSplit@ggml_vulkan",
        "wan: HybridParallel@ggml_cuda",
        "wan: HybridParallel@ggml_vulkan",
        "wan: LayerSplit@ggml_cuda",
        "wan: LayerSplit@ggml_vulkan",
        "wan: TensorParallel@ggml_cuda",
        "wan: TensorParallel@ggml_vulkan",
    };

    private static IEnumerable<string> ActualGaps() =>
        BuiltInArchitectures.Descriptors.SelectMany(a => a.Serving.Gaps().Select(g => $"{a.Id}: {g}"));

    [Fact]
    public void EveryFamilyDeclaresAServingContract()
    {
        foreach (var arch in BuiltInArchitectures.Descriptors)
        {
            Assert.NotNull(arch.Serving);
            foreach (ServingMechanism mechanism in Enum.GetValues<ServingMechanism>())
                Assert.NotNull(arch.Serving[mechanism]);
        }
    }

    [Fact]
    public void DeclaredGapsAreExactlyTheKnownGaps()
    {
        var actual = ActualGaps().ToHashSet(StringComparer.Ordinal);
        var known = KnownGaps.ToHashSet(StringComparer.Ordinal);
        var added = actual.Except(known).OrderBy(g => g, StringComparer.Ordinal).ToList();
        var closed = known.Except(actual).OrderBy(g => g, StringComparer.Ordinal).ToList();
        Assert.True(added.Count == 0 && closed.Count == 0,
            (added.Count > 0 ? "New gaps (implement the mechanism instead):\n  " + string.Join("\n  ", added) + "\n" : "") +
            (closed.Count > 0 ? "Closed gaps (delete them from KnownGaps):\n  " + string.Join("\n  ", closed) + "\n" : "") +
            "Current matrix:\n" + string.Join("\n", actual.OrderBy(g => g, StringComparer.Ordinal).Select(g => $"        \"{g}\",")));
    }

    [Fact]
    public void NewFamiliesAreBuiltOnTheSharedRuntime()
    {
        foreach (var arch in BuiltInArchitectures.Descriptors)
        {
            if (arch.Serving.Implementation == ServingImplementation.SharedRuntime) continue;
            Assert.True(LegacyBespokeFamilies.Contains(arch.Id),
                $"'{arch.Id}' implements its own serving mechanisms. New families must be built on the shared " +
                "decoder runtime (StandardDecoderModel) and declare ServingContract.SharedRuntime(...).");
        }
    }

    [Fact]
    public void TheLegacyListOnlyShrinks()
    {
        // A family that moved to the shared runtime must leave the legacy list, so the list
        // cannot hide a family that later regresses to a bespoke implementation.
        foreach (string id in LegacyBespokeFamilies)
        {
            Assert.True(ModelArchitectureRegistry.TryGet(id, out var arch), $"'{id}' is not registered; drop it from the legacy list.");
            Assert.Equal(ServingImplementation.Bespoke, arch.Serving.Implementation);
        }
    }

    [Fact]
    public void EverySharedRuntimeFamilyHasAConformanceFixture()
    {
        var withFixtures = SharedRuntimeConformanceTests.FamilyIds.ToHashSet(StringComparer.Ordinal);
        foreach (var arch in BuiltInArchitectures.Descriptors.Where(a => a.Serving.Implementation == ServingImplementation.SharedRuntime))
        {
            if (SharedRuntimeConformanceTests.DedicatedSuites.TryGetValue(arch.Id, out Type suite))
            {
                foreach (string test in SharedRuntimeConformanceTests.MechanismTests)
                    Assert.True(suite.GetMethod(test)?.GetCustomAttributes(typeof(FactAttribute), inherit: true).Length > 0,
                        $"'{arch.Id}' is proven by {suite.Name}, which must declare the test {test}.");
                continue;
            }
            Assert.True(withFixtures.Contains(arch.Id),
                $"'{arch.Id}' claims every mechanism through the shared runtime; add its synthetic fixture to " +
                "SharedRuntimeConformanceTests.Families so the claims are proven.");
        }
    }

    [Fact]
    public void ExpertOffloadIsRefusedWhereAFamilyDeclaresAGap()
    {
        MoeCpuOffloadConfig.Reset();
        try
        {
            MoeCpuOffloadConfig.SetAllLayers();
            Assert.True(ModelArchitectureRegistry.TryGet("gemma4", out var gemma));
            var error = Assert.Throws<NotSupportedException>(() =>
                ModelBase.ValidateExpertOffloadRequest(gemma, BackendType.Mlx, 1));
            Assert.Contains("--n-cpu-moe / --cpu-moe", error.Message, StringComparison.Ordinal);
            Assert.Contains("stacked-expert", error.Message, StringComparison.Ordinal);
            ModelBase.ValidateExpertOffloadRequest(gemma, BackendType.GgmlMetal, 1);

            // Dense families take the option as a no-op with a note, never a refusal.
            Assert.True(ModelArchitectureRegistry.TryGet("mistral3", out var mistral));
            ModelBase.ValidateExpertOffloadRequest(mistral, BackendType.Mlx, 1);
        }
        finally { MoeCpuOffloadConfig.Reset(); }
    }
}
