// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The loaded model as the probe drives it: prompt chunks through the model's prompt entry (ForwardPrompt, marked as served
// prompt chunks are), scored prompt chunks (every row's logits, StandardDecoderModel.ForwardPromptRowsForTests), decode rows
// (Forward of one token), chat prompts rendered with the model's own template, and the arms: in-process seams of the CUDA
// executor, never process environment variables.
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace LongContextQualityProbe;

/// <summary>One arm of a run: a name the records carry, the prompt attention it runs (default, f16, f32acc, 8bit) and whether every NVFP4
/// product runs on the q8 matvec (<paramref name="Q8Act"/>: prompt rows keep the decode rows' q8 activations).</summary>
internal sealed record Arm(string Name, string PromptAttention, bool Q8Act = false)
{
    public static readonly string[] Settings = { "default", "f16", "f32acc", "8bit" };

    private const string Q8ActSuffix = "+q8act";

    /// <summary>"8bit", "f32acc", "f16", "default", "q8act" (the build's prompt attention with q8 activations), a prompt
    /// attention with "+q8act" ("8bit+q8act", "f16+q8act"), or "name=setting".</summary>
    public static Arm Parse(string spec)
    {
        int eq = spec.IndexOf('=');
        string name = eq < 0 ? spec : spec[..eq], setting = eq < 0 ? spec : spec[(eq + 1)..];
        bool q8act = setting == "q8act" || setting.EndsWith(Q8ActSuffix, StringComparison.Ordinal);
        string attention = setting == "q8act" ? "default" : q8act ? setting[..^Q8ActSuffix.Length] : setting;
        if (!Settings.Contains(attention))
            throw new ArgumentException($"arm '{spec}': the setting is one of {string.Join(", ", Settings)}, q8act, or one of the first with {Q8ActSuffix}");
        if (name.Length == 0 || name.Contains(':') || name.Contains(',')) throw new ArgumentException($"arm '{spec}': a name without ':' or ','");
        return new Arm(name, attention, q8act);
    }

    /// <summary>The setting as <see cref="Parse"/> reads it back.</summary>
    public string Setting => !Q8Act ? PromptAttention : PromptAttention == "default" ? "q8act" : PromptAttention + Q8ActSuffix;
}

internal readonly record struct Counters(long Attn8Launches, long Attn8BandRefusals, long MarkedAttentions, long F32AccLaunches = 0);

internal sealed class ModelHandle
{
    public readonly ModelBase Model;
    public readonly StandardDecoderModel Decoder;
    public readonly CudaGraphExecutor Cuda;
    public readonly TensorGraphExecutor Graph;
    public readonly CudaAllocator CudaAllocator;
    private readonly KVCachePromptRenderer _renderer = new(new GgufPromptRenderer());

    public ModelHandle(ModelBase model)
    {
        Model = model;
        Decoder = model as StandardDecoderModel;
        IGraphExecutor executor = Decoder?.GraphExecutorForTests;
        Graph = executor as TensorGraphExecutor;
        Cuda = executor as CudaGraphExecutor;
        CudaAllocator = model.DirectAllocatorForTelemetry as CudaAllocator;
    }

    public int Vocab => Model.Config.VocabSize;

    /// <summary>Whether the loaded CUDA image carries the 8-bit prompt attention (an sm_120a / sm_121a build on its GPU).</summary>
    public bool Runs8BitPrompt => CudaAllocator?.FlashPrefillKernels?.Runs8BitPrompt(256) == true;

    /// <summary>Whether the loaded CUDA image carries the FP4 tensor-core products (an sm_120a / sm_121a build on its GPU): only
    /// there do NVFP4 prompt rows take the W4A4 GEMMs that the q8act arm moves to the q8 matvec.</summary>
    public bool RunsFp4TensorCores => CudaAllocator?.Fp4Kernels != null;

    public void SetArm(Arm arm)
    {
        if (arm.Setting != "default" && (Cuda == null || CudaAllocator == null))
            throw new InvalidOperationException($"arm {arm.Name}: the {arm.Setting} setting is a seam of the CUDA executor (--backend cuda)");
        if (CudaAllocator != null) CudaAllocator.AllNvfp4RowsOnMatvecForTests = arm.Q8Act;
        if (Cuda != null)
        {
            // The executor's --prompt-attention path (PromptAttentionPath), case by case; "default" keeps the process's own.
            Cuda.PromptAttentionForTests = arm.PromptAttention switch
            {
                "f16" => PromptAttentionPath.F16, "f32acc" => PromptAttentionPath.F32Acc, "8bit" => PromptAttentionPath.Int8, _ => null,
            };
            Cuda.PromptAttention8ForTests = null;
        }
    }

    public Counters Snapshot() => new(Cuda?.PromptAttention8Launches ?? 0, Cuda?.PromptAttention8BandRefusals ?? 0, Graph?.PromptChunkAttentions ?? 0,
        Cuda?.PromptAttentionF32AccLaunches ?? 0);

    public long FreeDeviceBytes() => CudaAllocator?.FreeDeviceBytes() ?? -1;

    /// <summary>A prompt through the model's prompt entry in chunks of <paramref name="chunk"/> rows (the served solo chunk is
    /// 2,048), returning the last row's logits.</summary>
    public float[] Prefill(int[] tokens, int chunk)
    {
        float[] last = null;
        for (int pos = 0; pos < tokens.Length; pos += chunk)
        {
            int len = Math.Min(chunk, tokens.Length - pos);
            bool more = pos + len < tokens.Length;
            last = Model.ForwardPrompt(tokens.AsSpan(pos, len).ToArray(), morePromptFollows: more);
        }
        return last;
    }

    public float[] Decode(int token) => Model.Forward(new[] { token });

    public float[] ScoredPromptChunk(int[] tokens)
        => (Decoder ?? throw new NotSupportedException($"{Model.Config.Architecture}: scored prompt chunks need a shared-decoder model"))
            .ForwardPromptRowsForTests(tokens);

    public int[] Render(List<(string Role, string Content)> messages, bool think)
        => _renderer.RenderToTokens(Model.Tokenizer, Model.Config.ChatTemplate,
            messages.Select(m => new ChatMessage { Role = m.Role, Content = m.Content }).ToList(), Model.Config.Architecture,
            addGenerationPrompt: true, enableThinking: think).ToArray();
}
