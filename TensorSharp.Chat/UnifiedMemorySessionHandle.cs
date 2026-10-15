// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Globalization;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server;

/// <summary>A lifecycle-owned session, with a CPU-only test seam for release failures.</summary>
internal sealed class UnifiedMemorySessionHandle
{
    private readonly Func<SchedulerConfig, SchedulerConfig> _configure;
    private readonly Func<string> _featureReport;
    private readonly Action _dispose;

    internal UnifiedMemorySessionHandle(ModelBase model,
        Func<SchedulerConfig, SchedulerConfig> configure, Func<string> featureReport, Action dispose)
    {
        Model = model;
        _configure = configure ?? throw new ArgumentNullException(nameof(configure));
        _featureReport = featureReport ?? throw new ArgumentNullException(nameof(featureReport));
        _dispose = dispose ?? throw new ArgumentNullException(nameof(dispose));
    }

    internal static UnifiedMemorySessionHandle Wrap(IUnifiedMemoryModelSession session)
        => new(session.Model, session.ConfigureScheduler, () => session.FeatureReport, session.Dispose);

    internal static UnifiedMemorySessionHandle CreateOwned(string path, string projector, BackendType backend)
    {
        // Resolve explicit host tuning before any model/native allocation. The
        // same prefill cap must govern load forecasts and the final scheduler;
        // applying only SchedulerConfig would retain the wrong memory plan.
        var options = ResolveHostOptions();
        var session = UnifiedMemoryModelSession.Create(path, projector, backend, options);
        try { return Wrap(session); }
        catch (Exception creation)
        {
            // Creating this managed adapter happens after the model owns real
            // buffers. Even failure here must not orphan the original session.
            try { session.Dispose(); }
            catch (Exception cleanup)
            { throw new UnifiedMemoryModelLoadException(session, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    internal static UnifiedMemoryModelOptions ResolveHostOptions()
    {
        const string chunkVariable = "TS_SCHED_PREFILL_CHUNK";
        // This path is already an explicit --unified-memory opt-in. GLM's
        // matching native planner admits its serialized selected-expert tier;
        // Qwen ignores this architecture-specific option. Public API defaults
        // stay independent of this Host policy.
        var scheduler = SchedulerConfig.FromEnvironment();
        var options = new UnifiedMemoryModelOptions { EnableGlmSelectedExperts = true,
            EnableAdaptiveGlmPrefill = true, AdaptiveGlmPrefillChunkTokens = 1024,
            PrefixCheckpointCount = scheduler.EnablePrefixCaching
                ? (scheduler.ExecutionOptionsOverride ?? ExecutionOptions.FromEnvironment()).PrefixCheckpointBudget : 0 };
        string chunk = Environment.GetEnvironmentVariable(chunkVariable);
        if (chunk != null)
        {
            if (!int.TryParse(chunk, NumberStyles.None, CultureInfo.InvariantCulture, out int rows)
                || rows is < 1 or > 512)
                throw new ArgumentException("--unified-memory requires TS_SCHED_PREFILL_CHUNK (--prefill-chunk-size) to be an integer from 1 through 512.");
            options = options with { PrefillChunkTokens = rows, AdaptiveGlmPrefillChunkTokens = rows };
        }
        foreach (string variable in new[] { "TS_PER_SEQ_FUSED", "TS_BATCHED_FUSED_DECODE" })
        {
            string value = Environment.GetEnvironmentVariable(variable);
            // Match ExecutionOptions' recognized false values. Qualified
            // unified execution needs both paths; refuse an explicit opt-out
            // instead of silently overriding it in ConfigureScheduler.
            if (value == "0" || string.Equals(value, "false", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException($"--unified-memory requires {variable} to remain enabled; remove its explicit disable setting or use the ordinary model-loading mode.");
        }
        return options;
    }

    internal ModelBase Model { get; }
    internal SchedulerConfig ConfigureScheduler(SchedulerConfig config) => _configure(config);
    internal string FeatureReport => _featureReport();
    internal void Dispose() => _dispose();
}
