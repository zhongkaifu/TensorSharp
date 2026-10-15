// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace TensorSharp.Server.Hosting;

/// <summary>Chooses what a startup request can prepare from the loaded
/// engine's effective prefix-cache contract, independently of model family.</summary>
public static class StartupWarmupPolicy
{
    public const string SinglePreferredVariantReason = "single-public-checkpoint-budget";

    /// <summary>Use the Web UI's default thinking mode unless an operator
    /// explicitly selects a strict boolean value. Whitespace is not ignored.</summary>
    public static bool ResolvePreferredThinking(string? setting)
        => setting switch
        {
            null => true,
            "0" => false,
            "1" => true,
            _ when string.Equals(setting, "true", StringComparison.OrdinalIgnoreCase) => true,
            _ when string.Equals(setting, "false", StringComparison.OrdinalIgnoreCase) => false,
            _ => throw new FormatException("TS_STARTUP_WARMUP_THINKING must be 0, 1, true, or false."),
        };

    /// <summary>A single effective public-copy slot cannot keep both thinking
    /// variants. Warm only the preferred variant, independently of family.</summary>
    public static bool UseSinglePreferredVariant(bool unifiedMemoryEnabled,
        bool prefixCacheActive, PrefixCacheCapabilities? capabilities,
        int effectivePublicCheckpointBudget)
        => unifiedMemoryEnabled && prefixCacheActive && capabilities?.CanCaptureCopy == true
            && effectivePublicCheckpointBudget == 1;

    /// <summary>Unified execution prepares only kernels when it cannot copy a
    /// public prompt checkpoint. Other execution modes retain their existing
    /// startup selection. A successful warmup still does not prove that a
    /// reusable checkpoint was retained.</summary>
    /// <param name="prefixCacheActive">Whether the loaded engine actually
    /// attached its prefix cache, rather than whether an operator requested it.</param>
    /// <param name="capabilities">The loaded model's declared prefix contract.</param>
    /// <param name="effectivePublicCheckpointBudget">The engine's resolved
    /// public checkpoint budget after session configuration. A requested
    /// environment budget is insufficient when an adapter overrides it.</param>
    public static bool UseKernelInitializationOnly(bool unifiedMemoryEnabled,
        bool prefixCacheActive, PrefixCacheCapabilities? capabilities,
        int effectivePublicCheckpointBudget)
        => unifiedMemoryEnabled && !(prefixCacheActive
            && capabilities?.CanCaptureCopy == true && effectivePublicCheckpointBudget > 0);
}
