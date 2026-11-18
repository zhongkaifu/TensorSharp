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

/// <summary>One arm of a run: a name the records carry and the prompt-attention setting it runs with.</summary>
internal sealed record Arm(string Name, string PromptAttention)
{
    public static readonly string[] Settings = { "default", "f16", "8bit" };

    /// <summary>"8bit", "f16", "default" or "name=setting".</summary>
    public static Arm Parse(string spec)
    {
        int eq = spec.IndexOf('=');
        string name = eq < 0 ? spec : spec[..eq], setting = eq < 0 ? spec : spec[(eq + 1)..];
        if (!Settings.Contains(setting)) throw new ArgumentException($"arm '{spec}': the setting is one of {string.Join(", ", Settings)}");
        if (name.Length == 0 || name.Contains(':') || name.Contains(',')) throw new ArgumentException($"arm '{spec}': a name without ':' or ','");
        return new Arm(name, setting);
    }
}

internal readonly record struct Counters(long Attn8Launches, long Attn8BandRefusals, long MarkedAttentions);

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

    public void SetArm(Arm arm)
    {
        if (arm.PromptAttention == "default") { if (Cuda != null) Cuda.PromptAttention8ForTests = null; return; }
        if (Cuda == null) throw new InvalidOperationException($"arm {arm.Name}: the prompt-attention setting is a seam of the CUDA executor (--backend cuda)");
        Cuda.PromptAttention8ForTests = arm.PromptAttention == "8bit";
    }

    public Counters Snapshot() => new(Cuda?.PromptAttention8Launches ?? 0, Cuda?.PromptAttention8BandRefusals ?? 0, Graph?.PromptChunkAttentions ?? 0);

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
