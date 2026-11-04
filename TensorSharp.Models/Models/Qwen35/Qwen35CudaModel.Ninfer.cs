// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

public sealed partial class Qwen35CudaModel
{
    private readonly string? _ninferCacheIdentity;

    /// <summary>Loads the known NInfer v2 text target on the same sequence/cache runtime as GGUF.</summary>
    public static Qwen35CudaModel FromNinfer(string artifactPath)
    {
        NinferQwen38ModelSource.ValidateExecutionRoute(BackendType.Cuda, 1, null, 1, null);
        using var source = NinferQwen38ModelSource.Open(artifactPath);
        return new Qwen35CudaModel(source);
    }

    private Qwen35CudaModel(NinferQwen38ModelSource source)
        : base(source.Reader.FullPath, BackendType.Cpu, openGguf: false)
    {
        try
        {
            _ninferCacheIdentity = source.CacheIdentity;
            Config = NinferQwen38ModelSource.CreateConfig(source.Frontend);
            Tokenizer = source.Frontend.Tokenizer;
            int context = ResolveNinferContext(Environment.GetEnvironmentVariable("MAX_CONTEXT"), source.Frontend.MaximumContextTokens);
            int nUbatch = ParseEnvInt("TS_Q35_UBATCH", 2048);
            _exec = new Qwen35CudaExecutor(source, context, nUbatch);
            InitializeEngineRuntime();
            Console.WriteLine($"Model: NInfer Qwen3.8 target on the direct-CUDA engine, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Vocab={Config.VocabSize}, n_ctx={_maxContextLength}, drafter={(_exec.Engine.HasNinferDrafter ? (_exec.Engine.NinferDraftUsesShortlist ? "original DFlash2 K7 Q4 shortlist" : "original DFlash2 K7 full head") : "none")}");
        }
        catch
        {
            Dispose();
            throw;
        }
    }

    internal static int ResolveNinferContext(string? requested, int declared)
    {
        if (declared <= 0) throw new ArgumentOutOfRangeException(nameof(declared));
        if (string.IsNullOrWhiteSpace(requested)) return Math.Min(declared, 65536);
        if (!int.TryParse(requested, out int context) || context <= 0 || context > declared)
            throw new ArgumentException($"MAX_CONTEXT must be between 1 and the NInfer declared context {declared}.", nameof(requested));
        return context;
    }

    private void InitializeEngineRuntime()
    {
        _exec.Engine.ReclaimMemory = ReclaimDeviceMemory;
        _hostBudgetBytes = ResolveHostBudgetBytes();
        // The engine owns the same retained-slot and pinned-pool lifetime for both formats.
        _exec.Engine.ConfigureHostPool(_hostBudgetBytes, _hostBudgetBytes);
        _primarySlot = _exec.ActiveSlot;
        _maxContextLength = _exec.ContextSize;
        Config.VocabSize = _exec.VocabSize;
        _logitsBuffer = new float[Config.VocabSize];
    }
}
