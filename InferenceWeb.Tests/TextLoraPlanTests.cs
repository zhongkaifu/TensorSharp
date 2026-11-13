// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Lora;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The deny-by-default text-LoRA plan (PR #282 review item 2): a model either applies every
/// target of an adapter on every forward path or refuses the adapter, naming the modules.
/// Model-level refusals (multi-device modes, backends, families, architecture), per-target
/// resolution through projections fused at load (exact composites), the per-forward audit
/// that turns a bypass into an error, and that merely binding an adapter costs base traffic
/// no fused path.
/// </summary>
public sealed class TextLoraPlanTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("ts-lora-plan-").FullName;
    private readonly string _probe;

    public TextLoraPlanTests()
    {
        _probe = Path.Combine(_dir, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_probe);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    // ------------------------------------------------------------------ model-level refusal

    [Fact]
    public void OnlyNemotronMistral3AndHunyuanDense_OptIn_EveryOtherFamilyRefuses()
    {
        var models = typeof(ModelBase).Assembly.GetTypes()
            .Where(t => !t.IsAbstract && t.IsSubclassOf(typeof(ModelBase)))
            .ToList();
        static bool Overrides(Type t, string property)
            => t.GetProperty(property, BindingFlags.Instance | BindingFlags.NonPublic)?.GetMethod?.DeclaringType != typeof(ModelBase);
        static bool OverridesMethod(Type t, string method)
            => t.GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)?.DeclaringType != typeof(ModelBase);

        var optedIn = models.Where(t => Overrides(t, "SupportsTextLora")).Select(t => t.Name).OrderBy(n => n).ToArray();

        Assert.Equal(new[] { "HunyuanDenseModel", "Mistral3Model", "NemotronModel" }, optedIn);
        // Each opted-in family names the weights it applies a delta to; none relies on a default.
        Assert.All(models.Where(t => Overrides(t, "SupportsTextLora")), t => Assert.True(OverridesMethod(t, "TextLoraWeightRefusal"), t.Name));
        // The families the review tested, by name: they never call the hook on every path.
        foreach (string name in new[] { "DeepSeek41Model", "DeepSeek4Model", "Qwen35RuntimeModel", "Qwen35CudaModel", "Gemma4Model", "Qwen4ExpRuntimeModel", "GptOssModel" })
            Assert.DoesNotContain(name, optedIn);
        Assert.Contains(models, t => t.Name == "DeepSeek41Model");
    }

    [Theory]
    [InlineData("mistral3", BackendType.Cpu, false, true, null, null)]
    [InlineData("mistral3", BackendType.GgmlCpu, false, true, null, null)]
    [InlineData("mistral3", BackendType.GgmlCuda, false, true, null, null)]
    [InlineData("mistral3", BackendType.Cuda, false, true, null, null)]
    [InlineData("mistral3", BackendType.GgmlVulkan, false, true, null, "not supported on the GgmlVulkan backend")]
    [InlineData("mistral3", BackendType.GgmlMetal, false, true, null, "not supported on the GgmlMetal backend")]
    [InlineData("mistral3", BackendType.Mlx, false, true, null, "not supported on the Mlx backend")]
    [InlineData("mistral3", BackendType.GgmlCuda, true, true, null, "more than one device or node")]
    [InlineData("deepseek41", BackendType.GgmlCuda, false, false, null, "deepseek41 runs its projections in fused or native graphs")]
    [InlineData("qwen35", BackendType.GgmlCuda, false, false, null, "qwen35 runs its projections in fused or native graphs")]
    [InlineData("gemma4", BackendType.Cpu, false, false, null, "gemma4 runs its projections in fused or native graphs")]
    [InlineData("mistral3", BackendType.Cpu, false, true, "qwen35", "converted for a 'qwen35' model, and the loaded model is 'mistral3'")]
    [InlineData("mistral3", BackendType.Cpu, false, true, "MISTRAL3", null)]
    [InlineData("nemotron_h_moe", BackendType.Cpu, false, true, "nemotron_h", null)]
    public void ModelRefusal_CoversModeBackendFamilyAndArchitecture(
        string arch, BackendType backend, bool multiDevice, bool family, string? adapterArch, string? expected)
    {
        string? refusal = ModelBase.DescribeTextLoraModelRefusal(arch, backend, multiDevice, family, adapterArch);

        if (expected == null)
            Assert.Null(refusal);
        else
            Assert.Contains(expected, refusal);
    }

    /// <summary>
    /// llama.cpp's rule: an adapter labelled with the base GGUF's own
    /// <c>general.architecture</c> fits. A Mistral 3 file labelled <c>llama</c> (served as
    /// mistral3) takes the <c>llama</c> adapter a converter writes for it; on a base whose file
    /// is not labelled <c>llama</c> that adapter is still refused.
    /// </summary>
    [Theory]
    [InlineData("llama", "llama", null)]
    [InlineData("LLAMA", "llama", null)]
    [InlineData("mistral3", "llama", null)]
    [InlineData("llama", null, "converted for a 'llama' model, and the loaded model is 'mistral3'")]
    [InlineData("qwen35", "llama", "converted for a 'qwen35' model, and the loaded model is 'mistral3' (its GGUF is labelled 'llama')")]
    public void ModelRefusal_AcceptsTheBaseFilesOwnArchitectureLabel(string adapterArch, string? fileArch, string? expected)
    {
        string? refusal = ModelBase.DescribeTextLoraModelRefusal("mistral3", BackendType.Cpu, multiDevice: false,
            supportsFamily: true, adapterArch, fileArch);

        if (expected == null)
            Assert.Null(refusal);
        else
            Assert.Contains(expected, refusal);
    }

    /// <summary>
    /// The same through a loaded model: a Mistral 3 GGUF labelled <c>llama</c> takes a
    /// <c>llama</c> adapter of its shapes and refuses one of other shapes per weight, while a
    /// file labelled <c>mistral3</c> refuses the <c>llama</c> adapter as a whole.
    /// </summary>
    [Theory]
    [InlineData("llama", DenseDecoderSyntheticModelBuilder.Hidden, null)]
    [InlineData("llama", DenseDecoderSyntheticModelBuilder.Hidden * 2, "blk.{0}.attn_q.weight")]
    [InlineData("mistral3", DenseDecoderSyntheticModelBuilder.Hidden, "converted for a 'llama' model, and the loaded model is 'mistral3'.")]
    public void ALlamaLabelledMistral3_TakesALlamaAdapterOfItsShapes(string fileArch, int inDim, string? refused)
    {
        string path = new DenseDecoderSyntheticModelBuilder { Architecture = fileArch }
            .Write(Path.Combine(_dir, $"mistral3-{fileArch}-{inDim}.gguf"));
        using var model = ModelBase.Create(path, BackendType.Cpu);
        Assert.IsAssignableFrom<Mistral3Model>(model);
        var shaped = Adapter("labelled", ("blk.0.attn_q.weight", 2, inDim,
            DenseDecoderSyntheticModelBuilder.NumHeads * DenseDecoderSyntheticModelBuilder.HeadDim));
        var adapter = new TextLoraAdapter(shaped.Name, shaped.Rank, shaped.Alpha, shaped.Deltas, shaped.Summary, architecture: "llama");

        var plan = model.PlanTextLora(adapter);

        if (refused == null)
        {
            Assert.True(plan.IsApplicable, plan.RefusalMessage);
            return;
        }
        Assert.False(plan.IsApplicable);
        Assert.Contains(refused, plan.RefusalMessage);
        Assert.Equal(new[] { "blk.{0}.attn_q.weight" }, plan.UnsupportedModules);
    }

    public static TheoryData<string> MultiDeviceModes => new() { "native-tp", "managed-tp", "layer-split" };

    [Theory]
    [MemberData(nameof(MultiDeviceModes))]
    public void EveryMultiDeviceMode_RefusesTheWholeAdapter(string mode)
    {
        using var model = mode switch
        {
            // An executor's own TP creates no managed group; the requested degree still counts.
            "native-tp" => new LoraProbeModel(_probe, new float[12], 4, 3, tpDegree: 2),
            "managed-tp" => new LoraProbeModel(_probe, new float[12], 4, 3, tpGroup: new MuseSparkTensorParallelTests.CpuGroup(2)),
            _ => new LoraProbeModel(_probe, new float[12], 4, 3, layerSplitDegree: 2),
        };
        var adapter = ProbeAdapter("multi", (LoraProbeModel.Target, 2, 3, 4));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Contains("more than one device or node", plan.ModelRefusal);
        Assert.Equal(new[] { "blk.{0}.attn_output.weight" }, plan.UnsupportedModules);
        Assert.False(model.TryBindAdapter(adapter, out _));
    }

    /// <summary>A real Nemotron under --tp 2 (a managed group), and as one node of a two-node
    /// layer split (a group the model never shards for, so only the plan can tell).</summary>
    [Theory]
    [InlineData("tp")]
    [InlineData("distributed-layer-split")]
    public void NemotronOnMoreThanOneDeviceOrNode_IsRefused(string mode)
    {
        string path = new NemotronSyntheticModelBuilder().Write(Path.Combine(_dir, $"nemotron-{mode}.gguf"));
        using var model = mode == "tp"
            ? new NemotronModel(path, BackendType.Cpu, 2, new MuseSparkTensorParallelTests.CpuGroup(2))
            : new NemotronModel(path, BackendType.Cpu, 1, new TwoNodeLayerSplit());
        var adapter = Adapter(mode, ("blk.1.attn_q.weight", 2, NemotronSyntheticModelBuilder.Hidden, NemotronSyntheticModelBuilder.NumHeads * NemotronSyntheticModelBuilder.HeadDim));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Contains("nemotron_h_moe runs on more than one device or node", plan.RefusalMessage);
        Assert.Equal(new[] { "blk.{1}.attn_q.weight" }, plan.UnsupportedModules);
        Assert.False(model.TryBindAdapter(adapter, out _));
    }

    [Fact]
    public void AFamilyThatHasNotOptedIn_RefusesLive()
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "qwen4exp.gguf"), contextLength: 128);
        using var model = new Qwen4ExpRuntimeModel(path, BackendType.Cpu);
        var adapter = Adapter("q", ("blk.3.attn_q.weight", 2, 8, 8), ("output.weight", 2, 8, 8));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Contains("qwen4exp runs its projections in fused or native graphs", plan.RefusalMessage);
        Assert.Equal(new[] { "blk.{3}.attn_q.weight", "output.weight" }, plan.UnsupportedModules);
    }

    // ------------------------------------------------------------------ per-target plan

    [Fact]
    public void FusedProjections_ResolveTheirMembers_AndTheCompositeDeltaIsExact()
    {
        using var model = new FusionProbeModel(_probe);
        const int In = FusionProbeModel.In;
        var targets = new (string Target, int Rank, int Out)[]
        {
            ("blk.0.attn_q.weight", 2, FusionProbeModel.Q),
            ("blk.0.attn_v.weight", 3, FusionProbeModel.V),
            ("blk.0.attn_qkv.weight", 1, FusionProbeModel.QkvRows),   // a delta on the fused weight itself
            ("blk.0.ffn_up.weight", 2, FusionProbeModel.Ff),
        };
        var factors = targets.Select((t, i) => TextLoraFactors.Random(100 + i, t.Rank, In, t.Out)).ToArray();
        float[] scales = { 0.5f, 2f, 1.5f, 3f };
        var adapter = FromFactors("composite", targets.Select((t, i) => (t.Target, factors[i].A, factors[i].B, t.Rank, In, t.Out, scales[i])).ToArray());

        var plan = model.PlanTextLora(adapter);
        Assert.True(plan.IsApplicable, plan.RefusalMessage);
        Assert.Equal(new[] { "blk.0.attn_qkv.weight", "blk.0.ffn_gate_up.weight" }, plan.AppliedWeights);
        Assert.Equal(4, plan.AppliedTargets.Count);

        float[] x = Enumerable.Range(0, 3 * In).Select(i => MathF.Sin(i * 0.7f)).ToArray();
        float[] baseQkv = model.Linear("blk.0.attn_qkv.weight", x, 3);
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);
        model.SelectStepAdapter(adapter);
        float[] qkv = model.Linear("blk.0.attn_qkv.weight", x, 3);
        float[] gateUp = model.Linear("blk.0.ffn_gate_up.weight", x, 3);
        model.SelectStepAdapter(null);
        Assert.Equal(baseQkv, model.Linear("blk.0.attn_qkv.weight", x, 3));

        // Reference: each source weight plus its own delta, concatenated in fused row order,
        // plus the delta named on the fused weight across all of its rows.
        float[] wq = TextLoraFactors.Merge(model.Source("blk.0.attn_q.weight"), factors[0].A, factors[0].B, 2, In, FusionProbeModel.Q, scales[0]);
        float[] wk = model.Source("blk.0.attn_k.weight");
        float[] wv = TextLoraFactors.Merge(model.Source("blk.0.attn_v.weight"), factors[1].A, factors[1].B, 3, In, FusionProbeModel.V, scales[1]);
        float[] wqkv = TextLoraFactors.Merge(wq.Concat(wk).Concat(wv).ToArray(), factors[2].A, factors[2].B, 1, In, FusionProbeModel.QkvRows, scales[2]);
        AssertClose(Naive(x, 3, wqkv, In, FusionProbeModel.QkvRows), qkv);

        float[] wGate = model.Source("blk.0.ffn_gate.weight");
        float[] wUp = TextLoraFactors.Merge(model.Source("blk.0.ffn_up.weight"), factors[3].A, factors[3].B, 2, In, FusionProbeModel.Ff, scales[3]);
        AssertClose(Naive(x, 3, wGate.Concat(wUp).ToArray(), In, 2 * FusionProbeModel.Ff), gateUp);
    }

    [Fact]
    public void Plan_RefusesEveryTargetItCannotApply_WithItsReason_AndNeverSkipsOne()
    {
        using var model = new FusionProbeModel(_probe);
        const int In = FusionProbeModel.In;
        var adapter = Adapter("refused",
            ("blk.0.attn_q.weight", 2, In, FusionProbeModel.Q),                 // applies
            ("token_embd.weight", 2, In, 16),
            ("v.blk.0.attn_qkv.weight", 2, In, 8),
            ("dflash.fc.weight", 2, In, In),
            ("blk.0.ffn_up_exps.weight", 2, In, FusionProbeModel.Ff),
            ("blk.0.attn_k.weight", 2, In, 3),                                   // its rows are [2, 6]
            ("blk.0.attn_output.weight", 2, FusionProbeModel.Q, In),            // present, not run through the hook
            ("blk.0.missing.weight", 2, In, In));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Equal(new[] { "blk.0.attn_q.weight" }, plan.AppliedTargets);
        var reasons = plan.Rejected.ToDictionary(r => r.Target, r => r.Reason);
        Assert.Equal(7, reasons.Count);
        Assert.Contains("not a decoder projection", reasons["token_embd.weight"]);
        Assert.Contains("not a decoder projection", reasons["v.blk.0.attn_qkv.weight"]);
        Assert.Contains("not a decoder projection", reasons["dflash.fc.weight"]);
        Assert.Contains("routed expert", reasons["blk.0.ffn_up_exps.weight"]);
        Assert.Equal("its rows of blk.0.attn_qkv.weight are [2, 6] but the delta is [3, 6]", reasons["blk.0.attn_k.weight"]);
        Assert.Contains("is not run through the LoRA hook", reasons["blk.0.attn_output.weight"]);
        Assert.Equal("not a weight of the loaded model", reasons["blk.0.missing.weight"]);
        Assert.Contains("refused rather than applied in part", plan.RefusalMessage);
        Assert.Equal(7, plan.UnsupportedModules.Count);
        Assert.Contains("blk.{0}.attn_k.weight", plan.UnsupportedModules);
    }

    [Fact]
    public void UnsupportedModules_ListEveryGroup_WhereTheMessageTruncates()
    {
        var targets = Enumerable.Range(0, 10).Select(i => ($"blk.0.p{i}.weight", 1, 1, 1))
            .Concat(Enumerable.Range(0, 28).Select(l => ($"blk.{l}.attn_output.weight", 1, 1, 1)))
            .ToArray();
        using var model = new LoraProbeModel(_probe, new float[1], 1, 1);

        var plan = model.PlanTextLora(Adapter("many", targets));

        Assert.Contains("and 3 more", plan.RefusalMessage);       // 11 groups, 8 shown
        Assert.Equal(11, plan.UnsupportedModules.Count);
        Assert.Contains("blk.{1-27}.attn_output.weight", plan.UnsupportedModules);
    }

    /// <summary>
    /// Target names are the adapter's own text. A layer number past int's range threw an
    /// OverflowException while the refusal was described (and a digit of another script a
    /// FormatException), which POST /v1/loras answered with a 500. Each is a target no model
    /// has: refused, and listed as written, whether the model refuses per target or as a whole.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void LayerNumbersOutOfRange_AreRefusedAndListedAsWritten_NeverThrown(bool supportsLora)
    {
        using var model = new LoraProbeModel(_probe, new float[1], 1, 1, supportsLora);
        var adapter = Adapter("huge",
            ("blk.99999999999.attn_output.weight", 1, 1, 1),
            ("blk.2147483648.attn_output.weight", 1, 1, 1),
            ("blk.2147483647.attn_output.weight", 1, 1, 1),
            ("blk.٣.attn_output.weight", 1, 1, 1));

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        Assert.Equal(new[]
        {
            "blk.{2147483647}.attn_output.weight",
            "blk.2147483648.attn_output.weight",
            "blk.99999999999.attn_output.weight",
            "blk.٣.attn_output.weight",
        }, plan.UnsupportedModules);
        Assert.Contains(supportsLora
            ? "blk.99999999999.attn_output.weight (not a weight of the loaded model)"
            : "nemotron_h, mistral3 and hunyuan-dense", plan.RefusalMessage);
        Assert.False(model.TryBindAdapter(adapter, out string? error));
        Assert.Equal(plan.RefusalMessage, error);
    }

    // ------------------------------------------------------------------ Nemotron

    [Theory]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", false)]
    [InlineData("SAMS", true)]
    public void Nemotron_AppliesEveryProjectionItAdapts_OnEveryForwardPath(string layout, bool latent)
        => RunNemotronAcceptedTargets(BackendType.Cpu, layout, latent);

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", true)]
    public void Nemotron_AppliesEveryProjectionItAdapts_OnEveryGgmlForwardPath(string layout, bool latent)
        => RunNemotronAcceptedTargets(BackendType.GgmlCpu, layout, latent);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", false)]
    [InlineData("SAMS", true)]
    public void Nemotron_AppliesEveryProjectionItAdapts_OnEveryGgmlCudaForwardPath(string layout, bool latent)
        => RunNemotronAcceptedTargets(BackendType.GgmlCuda, layout, latent);

    [CudaTheory]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", false)]
    [InlineData("SAMS", true)]
    public void Nemotron_AppliesEveryProjectionItAdapts_OnEveryCudaForwardPath(string layout, bool latent)
        => RunNemotronAcceptedTargets(BackendType.Cuda, layout, latent);

    /// <summary>
    /// An adapter on every projection Nemotron adapts (Mamba2 in/out, attention, the dense or
    /// shared-expert and latent FFN projections, the router, the LM head) of a Q8_0 hybrid: the
    /// plan accepts all of it, and prefill and decode (the fused decode paths on GGML, the native
    /// Mamba2 decode on ggml_cuda, the managed forward over the CUDA Q8_0 kernel on direct cuda)
    /// apply every planned weight, or the audit throws.
    /// </summary>
    private void RunNemotronAcceptedTargets(BackendType backend, string layout, bool latent)
    {
        string path = new NemotronSyntheticModelBuilder { Layout = layout, Latent = latent }
            .Write(Path.Combine(_dir, $"nemotron-{layout}-{latent}.gguf"));
        using var model = new NemotronModel(path, backend);
        var adapter = NemotronAdapter(path, "accepted", AdaptableNemotronTensors(path));

        var plan = model.PlanTextLora(adapter);
        Assert.True(plan.IsApplicable, plan.RefusalMessage);
        Assert.Contains("blk.1.attn_qkv.weight", plan.AppliedWeights);
        Assert.Contains("blk.0.ssm_in.weight", plan.AppliedWeights);
        Assert.Contains("blk.3.ssm_out.weight", plan.AppliedWeights);
        Assert.Contains(layout.Contains('F') ? "blk.2.ffn_down.weight" : "blk.2.ffn_down_shexp.weight", plan.AppliedWeights);
        if (latent)
        {
            Assert.Contains("blk.2.ffn_latent_in.weight", plan.AppliedWeights);
            Assert.Contains("blk.2.ffn_latent_out.weight", plan.AppliedWeights);
        }

        int[] prompt = NemotronPrompt();
        model.ResetKVCache();
        // Clone: a model may hand back the same logits buffer on every forward.
        float[] baseLogits = (float[])model.ForwardRefill(prompt).Clone();
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);
        model.SelectStepAdapter(adapter);
        try
        {
            model.ResetKVCache();
            float[] logits = model.ForwardRefill(prompt);
            Assert.True(MaxAbsDiff(baseLogits, logits) > 1e-3f, "the adapter must move the logits");
            // Decode steps take the single-token paths (the fused decode residuals on GGML);
            // the audit throws if any planned weight was skipped.
            for (int step = 0; step < 4; step++)
                logits = model.Forward(new[] { ArgMax(logits) });
        }
        finally
        {
            model.SelectStepAdapter(null);
        }
    }

    [Fact]
    public void Nemotron_RefusesRoutedExperts_AndWeightsThatAreNotItsProjections()
    {
        string path = new NemotronSyntheticModelBuilder { Layout = "SAMS", Latent = true }
            .Write(Path.Combine(_dir, "nemotron-refused.gguf"));
        using var model = new NemotronModel(path, BackendType.Cpu);
        var adapter = NemotronAdapter(path, "refused", new[]
        {
            "blk.2.ffn_up_exps.weight", "blk.0.ssm_conv1d.weight", "blk.1.attn_output.weight",
        });

        var plan = model.PlanTextLora(adapter);

        Assert.False(plan.IsApplicable);
        var reasons = plan.Rejected.ToDictionary(r => r.Target, r => r.Reason);
        Assert.Contains("routed expert", reasons["blk.2.ffn_up_exps.weight"]);
        Assert.Contains("not one of the projections Nemotron applies a LoRA delta to", reasons["blk.0.ssm_conv1d.weight"]);
        Assert.False(reasons.ContainsKey("blk.1.attn_output.weight"));
    }

    /// <summary>
    /// The audit against a real bypass: a Nemotron whose fused decode paths no longer decline for
    /// an adapted weight (its <c>HasLoraDelta</c> gate broken) adds attn_output, ssm_out and the
    /// FFN down projections to the residual through FusedMatMulQuantAdd on every GGML decode token,
    /// with no delta. The decode forward must fail instead of returning partly adapted logits.
    /// </summary>
    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", true)]
    public void Audit_CatchesABrokenNemotronDecodeGate(string layout, bool latent) => RunBrokenGate(BackendType.GgmlCpu, layout, latent);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData("SAFS", false)]
    [InlineData("SAMS", false)]
    [InlineData("SAMS", true)]
    public void Audit_CatchesABrokenNemotronDecodeGate_GgmlCuda(string layout, bool latent) => RunBrokenGate(BackendType.GgmlCuda, layout, latent);

    private void RunBrokenGate(BackendType backend, string layout, bool latent)
    {
        string path = new NemotronSyntheticModelBuilder { Layout = layout, Latent = latent }
            .Write(Path.Combine(_dir, $"nemotron-bypass-{layout}-{latent}.gguf"));
        using var model = new GateBreakingNemotron(path, backend);
        var adapter = NemotronAdapter(path, "bypassed", AdaptableNemotronTensors(path));
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);

        int[] prompt = NemotronPrompt();
        model.ResetKVCache();
        model.SelectStepAdapter(adapter);
        try
        {
            float[] logits = model.ForwardRefill(prompt);      // prefill runs LinearForward: applied
            var ex = Assert.Throws<TextLoraNotAppliedException>(() => model.Forward(new[] { ArgMax(logits) }));
            var bypassed = new List<string> { "blk.{1}.attn_output.weight", "blk.{0,3}.ssm_out.weight" };
            if (layout.Contains('F'))
                bypassed.Add("blk.{2}.ffn_down.weight");
            else
                bypassed.Add("blk.{2}.ffn_down_shexp.weight");
            if (latent)
                bypassed.Add("blk.{2}.ffn_latent_out.weight");
            foreach (string weight in bypassed)
                Assert.Contains(weight, ex.Message);
            // The paths that apply the delta themselves are not reported.
            Assert.DoesNotContain("ssm_in", ex.Message);
            Assert.DoesNotContain("attn_qkv", ex.Message);
        }
        finally
        {
            model.SelectStepAdapter(null);
        }
        // Base traffic is untouched by the failed step.
        model.ResetKVCache();
        model.Forward(new[] { ArgMax(model.ForwardRefill(prompt)) });
    }

    /// <summary>
    /// Regression risk 6 for Nemotron: its fused decode paths ask whether a weight is adapted
    /// only while a step runs with an adapter selected, so base traffic (with an adapter loaded
    /// and bound or not) pays nothing; and while one is selected they decline only for the
    /// weights it adapts, keeping FusedMatMulQuantAdd for the rest.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void Nemotron_FusedDecodePaths_AskOnlyWhileAnAdapterIsSelected_AndKeepUnadaptedWeights()
    {
        string path = new NemotronSyntheticModelBuilder { Layout = "SAFS" }.Write(Path.Combine(_dir, "nemotron-gate.gguf"));
        using var model = new GateProbeNemotron(path, BackendType.GgmlCpu);
        var adapter = NemotronAdapter(path, "partial", new[] { "blk.0.ssm_out.weight", "blk.1.attn_q.weight" });
        int[] prompt = Enumerable.Range(0, 9).Select(i => 40 + i * 3).ToArray();
        void Decode()
        {
            model.ResetKVCache();
            model.Forward(new[] { ArgMax(model.ForwardRefill(prompt)) });
        }

        Decode();
        Assert.Empty(model.Asked);
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);
        Decode();
        Assert.Empty(model.Asked);      // bound, not selected

        model.SelectStepAdapter(adapter);
        try
        {
            Decode();
        }
        finally
        {
            model.SelectStepAdapter(null);
        }
        var answers = model.Asked.GroupBy(a => a.Weight).ToDictionary(g => g.Key, g => g.All(a => a.Adapted));
        Assert.True(answers["blk.0.ssm_out.weight"]);           // declines: LinearForward applies the delta
        Assert.False(answers["blk.3.ssm_out.weight"]);          // the rest keep the fused path
        Assert.False(answers["blk.1.attn_output.weight"]);
        Assert.False(answers["blk.2.ffn_down.weight"]);

        model.Asked.Clear();
        Decode();
        Assert.Empty(model.Asked);
    }

    // ------------------------------------------------------------------ forward audit

    [Fact]
    public void Audit_ThrowsWhenAForwardSkipsAPlannedWeight_AndExemptsTheLmHeadWithoutLogits()
    {
        using var model = new AuditProbeModel(_probe);
        var adapter = Adapter("audited",
            (AuditProbeModel.Attn, 1, AuditProbeModel.Width, AuditProbeModel.Width),
            (AuditProbeModel.Down, 1, AuditProbeModel.Width, AuditProbeModel.Width),
            ("output.weight", 1, AuditProbeModel.Width, AuditProbeModel.Width));
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);

        // Without an adapter selected nothing is audited, whatever the forward skips.
        model.SkipDown = true;
        model.Forward(new[] { 1 });

        model.SelectStepAdapter(adapter);
        try
        {
            model.SkipDown = false;
            model.Forward(new[] { 1 });
            // A prompt chunk that produces no logits never runs the LM head, and owes it nothing.
            Assert.Null(model.ForwardRefill(new[] { 1, 2 }));

            model.SkipDown = true;
            var ex = Assert.Throws<TextLoraNotAppliedException>(() => model.Forward(new[] { 1 }));
            Assert.Contains("blk.{0}.ffn_down.weight", ex.Message);
            Assert.DoesNotContain("attn_output", ex.Message);

            // The failed forward closed its audit: the next one is judged on its own.
            model.SkipDown = false;
            model.Forward(new[] { 1 });
        }
        finally
        {
            model.SelectStepAdapter(null);
        }
    }

    // ------------------------------------------------------------------ fused paths for base traffic

    /// <summary>
    /// Regression risk 6: the fused dense FFN graphs decline only while a step runs with an
    /// adapter selected. Loading or binding one must not cost base requests the fused path.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void FusedFfnPaths_StayOnForBaseTraffic_WhileAnAdapterIsOnlyBound() => RunFusedFfnGate(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlCuda)]
    public void FusedFfnPaths_StayOnForBaseTraffic_WhileAnAdapterIsOnlyBound_GgmlCuda() => RunFusedFfnGate(BackendType.GgmlCuda);

    private void RunFusedFfnGate(BackendType backend)
    {
        string path = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", HalfPrecisionLayer = 1 }
            .Write(Path.Combine(_dir, "mistral3-f16.gguf"));
        using var model = new FusedFfnProbe(path, backend);
        var adapter = Adapter("bound", ("blk.0.attn_q.weight", 2, DenseDecoderSyntheticModelBuilder.Hidden,
            DenseDecoderSyntheticModelBuilder.NumHeads * DenseDecoderSyntheticModelBuilder.HeadDim));

        Assert.Equal((true, true), model.FusedFfnTaken(layer: 1));
        Assert.True(model.TryBindAdapter(adapter, out string? error), error);
        Assert.Equal((true, true), model.FusedFfnTaken(layer: 1));      // bound, not selected
        model.SelectStepAdapter(adapter);
        Assert.Equal((false, false), model.FusedFfnTaken(layer: 1));    // selected: the LoRA-aware path
        model.SelectStepAdapter(null);
        Assert.Equal((true, true), model.FusedFfnTaken(layer: 1));
    }

    // ------------------------------------------------------------------ helpers

    private static TextLoraAdapter Adapter(string name, params (string Target, int Rank, int In, int Out)[] targets)
        => FromFactors(name, targets.Select((t, i) =>
        {
            var (a, b) = TextLoraFactors.Random(i + 1, t.Rank, t.In, t.Out);
            return (t.Target, a, b, t.Rank, t.In, t.Out, 1f);
        }).ToArray());

    private static TextLoraAdapter ProbeAdapter(string name, params (string Target, int Rank, int In, int Out)[] targets)
        => Adapter(name, targets);

    /// <summary>An adapter over the targets' shapes as the GGUF at <paramref name="path"/> stores them.</summary>
    private static TextLoraAdapter NemotronAdapter(string path, string name, string[] targets)
    {
        using var gguf = new GgufFile(path);
        return Adapter(name, targets.Select(t =>
        {
            var shape = gguf.Tensors[t].Shape;
            return (t, 2, (int)shape[0], (int)shape[1]);
        }).ToArray());
    }

    private static TextLoraAdapter FromFactors(string name,
        params (string Target, float[] A, float[] B, int Rank, int In, int Out, float Scale)[] targets)
    {
        var deltas = new Dictionary<string, LoraDelta>(StringComparer.Ordinal);
        foreach (var (target, a, b, rank, inDim, outDim, scale) in targets)
        {
            var at = new float[inDim * rank];
            for (int j = 0; j < rank; j++)
                for (int i = 0; i < inDim; i++)
                    at[i * rank + j] = a[j * inDim + i];
            var bt = new float[rank * outDim];
            for (int o = 0; o < outDim; o++)
                for (int j = 0; j < rank; j++)
                    bt[j * outDim + o] = scale * b[o * rank + j];
            deltas[target] = new LoraDelta { In = inDim, Out = outDim, Rank = rank, AT = at, BT = bt };
        }
        return new TextLoraAdapter(name, targets.Max(t => t.Rank), targets.Max(t => t.Rank), deltas, name);
    }

    private static float[] Naive(float[] x, int rows, float[] w, int inDim, int outDim)
    {
        var y = new float[rows * outDim];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outDim; o++)
            {
                double acc = 0;
                for (int i = 0; i < inDim; i++)
                    acc += (double)x[r * inDim + i] * w[o * inDim + i];
                y[r * outDim + o] = (float)acc;
            }
        return y;
    }

    private static void AssertClose(float[] expected, float[] actual, float tolerance = 1e-4f)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Assert.True(Math.Abs(expected[i] - actual[i]) <= tolerance * Math.Max(1f, Math.Abs(expected[i])),
                $"[{i}] expected {expected[i]} got {actual[i]}");
    }

    private static float MaxAbsDiff(float[] a, float[] b) => a.Zip(b, (x, y) => Math.Abs(x - y)).Max();

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }

    /// <summary>A prompt for a Nemotron fixture; see <see cref="TextLoraModelParityTests.NemotronPromptLength"/>.</summary>
    private static int[] NemotronPrompt()
        => Enumerable.Range(0, TextLoraModelParityTests.NemotronPromptLength).Select(i => 40 + i * 3).ToArray();

    private static readonly Regex s_adaptableNemotron = new(
        @"^(output|blk\.\d+\.(attn_[qkv]|attn_output|ffn_up|ffn_down|ssm_in|ssm_out|ffn_gate_inp|ffn_(up|down)_shexp|ffn_latent_(down|up)))\.weight$",
        RegexOptions.CultureInvariant);

    /// <summary>Every projection of a Nemotron GGUF the model adapts, as the GGUF names it.</summary>
    private static string[] AdaptableNemotronTensors(string path)
    {
        using var gguf = new GgufFile(path);
        return gguf.Tensors.Keys.Where(n => s_adaptableNemotron.IsMatch(n)).OrderBy(n => n, StringComparer.Ordinal).ToArray();
    }

    /// <summary>A Nemotron whose fused decode paths never decline for an adapted weight, to give
    /// the audit a real bypass.</summary>
    private sealed class GateBreakingNemotron(string path, BackendType backend) : NemotronModel(path, backend)
    {
        internal override bool HasLoraDelta(string weightName) => false;
    }

    /// <summary>A Nemotron that records what its fused decode paths ask about LoRA, and the answer.</summary>
    private sealed class GateProbeNemotron(string path, BackendType backend) : NemotronModel(path, backend)
    {
        public List<(string Weight, bool Adapted)> Asked { get; } = new();

        internal override bool HasLoraDelta(string weightName)
        {
            bool adapted = base.HasLoraDelta(weightName);
            Asked.Add((weightName, adapted));
            return adapted;
        }
    }

    /// <summary>One node of a two-node layer split, as the model sees it at construction.</summary>
    private sealed class TwoNodeLayerSplit : ILayerSplitGroup
    {
        public int Degree => 1;
        public int GlobalDegree => 2;
        public int GlobalRankOffset => 0;
        public int NodeCount => 2;
        public int NodeId => 0;
        public bool IsActive => false;
        public IAllocator GetAllocator(int rank) => throw new NotSupportedException();
        public void AllReduce(Tensor[] tensors) => throw new NotSupportedException();
        public void Synchronize() => throw new NotSupportedException();
        public void Barrier() => throw new NotSupportedException();
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();
        public void Send(float[] values, int count, int destination) => throw new NotSupportedException();
        public void Receive(float[] values, int count, int source) => throw new NotSupportedException();
        public void Dispose() { }
    }

    /// <summary>A Mistral 3 that reports whether its two fused dense FFN graphs would run for a layer.</summary>
    private sealed class FusedFfnProbe(string path, BackendType backend) : Mistral3Model(path, backend)
    {
        public (bool SwiGluInto, bool Project) FusedFfnTaken(int layer)
        {
            string p = $"blk.{layer}.";
            using var residual = CreateFloatTensor(new float[Config.HiddenSize], 1, Config.HiddenSize);
            bool into = TryFusedDenseSwiGLUFFNInto(residual, p + "ffn_norm.weight", p + "ffn_gate_up.weight", p + "ffn_down.weight");
            using var input = CreateFloatTensor(new float[Config.HiddenSize], 1, Config.HiddenSize);
            using var projected = TryFusedDenseFFNProject(input, p + "ffn_norm.weight", p + "ffn_gate_up.weight", p + "ffn_down.weight", 0);
            return (into, projected != null);
        }
    }
}

/// <summary>
/// A cpu model whose layer 0 holds F32 attn_q/k/v and ffn_gate/up, fused at construction
/// through ModelBase's own fusion helpers, so the plan meets attn_qkv and ffn_gate_up exactly
/// as a family's load leaves them. attn_output is present but (in this probe) not adaptable.
/// </summary>
internal sealed class FusionProbeModel : ModelBase
{
    public const int In = 6, Q = 4, K = 2, V = 2, Ff = 5, QkvRows = Q + K + V;
    private readonly Dictionary<string, float[]> _sources = new(StringComparer.Ordinal);

    public FusionProbeModel(string path) : base(path, BackendType.Cpu)
    {
        Config = new ModelConfig { Architecture = "probe", VocabSize = 16, NumLayers = 1, HiddenSize = In, IntermediateSize = Ff };
        Add("blk.0.attn_q.weight", Q, In);
        Add("blk.0.attn_k.weight", K, In);
        Add("blk.0.attn_v.weight", V, In);
        Add("blk.0.attn_output.weight", In, Q);
        Add("blk.0.ffn_gate.weight", Ff, In);
        Add("blk.0.ffn_up.weight", Ff, In);
        if (!TryFuseQkvWeights(0))
            throw new InvalidOperationException("q/k/v did not fuse");
        FuseGateUpWeights(1);
        if (!_weights.ContainsKey("blk.0.ffn_gate_up.weight"))
            throw new InvalidOperationException("gate/up did not fuse");
    }

    private void Add(string name, int rows, int cols)
    {
        var rng = new Random(name.Length * 31 + name[^8]);
        float[] w = Enumerable.Range(0, rows * cols).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
        _sources[name] = w;
        _weights[name] = CreateFloatTensor(w, rows, cols);
    }

    /// <summary>A source weight as it was before fusion (row-major [out, in]).</summary>
    public float[] Source(string name) => _sources[name];

    protected override bool SupportsTextLora => true;

    protected override string? TextLoraWeightRefusal(string weightName)
        => weightName is "blk.0.attn_qkv.weight" or "blk.0.ffn_gate_up.weight" ? null : "is not run through the LoRA hook by this probe";

    public float[] Linear(string weight, float[] x, int rows)
    {
        using var input = CreateFloatTensor(x, rows, In);
        using var y = LinearForward(input, weight);
        return y.GetElementsAsFloat((int)y.ElementCount());
    }

    protected override float[] ForwardCore(int[] tokens) => new float[16];
    protected override void ResetKVCacheCore() { }
}

/// <summary>
/// A cpu model whose forward runs <see cref="Attn"/>, <see cref="Down"/> (unless
/// <see cref="SkipDown"/>) and the LM head, and whose refill runs the two projections but
/// returns no logits: a stand-in for a family path that bypasses the hook.
/// </summary>
internal sealed class AuditProbeModel : ModelBase
{
    public const string Attn = "blk.0.attn_output.weight", Down = "blk.0.ffn_down.weight";
    public const int Width = 3;

    public AuditProbeModel(string path) : base(path, BackendType.Cpu)
    {
        Config = new ModelConfig { Architecture = "probe", VocabSize = Width, NumLayers = 1 };
        foreach (string name in new[] { Attn, Down, "output.weight" })
            _weights[name] = CreateFloatTensor(Enumerable.Range(0, Width * Width).Select(i => 0.1f * (i % 4)).ToArray(), Width, Width);
    }

    public bool SkipDown { get; set; }

    protected override bool SupportsTextLora => true;
    protected override string? TextLoraWeightRefusal(string weightName) => null;

    private Tensor Project(Tensor x, string weight)
    {
        var y = LinearForward(x, weight);
        x.Dispose();
        return y;
    }

    protected override float[] ForwardCore(int[] tokens)
    {
        Tensor h = CreateFloatTensor(Enumerable.Repeat(1f, Width).ToArray(), 1, Width);
        h = Project(h, Attn);
        if (!SkipDown)
            h = Project(h, Down);
        h = Project(h, "output.weight");
        float[] logits = h.GetElementsAsFloat(Width);
        h.Dispose();
        return logits;
    }

    protected override float[] ForwardRefillCore(int[] tokens)
    {
        Tensor h = CreateFloatTensor(Enumerable.Repeat(1f, Width).ToArray(), 1, Width);
        h = Project(Project(h, Attn), Down);
        h.Dispose();
        return null!;
    }

    protected override void ResetKVCacheCore() { }
}
