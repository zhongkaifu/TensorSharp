// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server.Hosting;

/// <summary>Startup-only execution of short, genuinely joint request cohorts.
/// Every request uses the host's ordinary chat stream and a separate owned
/// session/workspace. Completion alone is insufficient: a fresh successful
/// model batch record must prove the exact width. This is an optional warmup,
/// not authorization of a new model/backend or execution capability.</summary>
public static class JointDecodeStartupWarmup
{
    public sealed record Result(int Width, bool RequestsCompleted, bool NativeBatchAccepted,
        bool CleanupCompleted, FusedBatchExecutionEvidence? Evidence, string Detail, TimeSpan Elapsed)
    {
        public bool Warmed => RequestsCompleted && NativeBatchAccepted && CleanupCompleted && !CleanupPending;
        public bool CleanupPending { get; init; }
    }

    public static bool IsQualified(ModelBase? model, bool unified, string backend, InferenceEngine? engine)
        => QualificationDeclineReason(model, unified, backend, engine) == null;

    public static string? QualificationDeclineReason(ModelBase? model, bool unified, string backend, InferenceEngine? engine)
    {
        if (!unified || backend != "ggml_cuda" || model is not Qwen4ExpModel qwen)
            return "startup joint warmup is qualified only for unified Qwen CUDA";
        if (!qwen.SupportsPerSequenceFusedForward || !qwen.SupportsPrefixCheckpoints)
            return "the model lacks the qualified independent-holder and prefix-copy execution path";
        if (qwen.KvCacheDtype != KvCacheDtype.F16)
            return "the qualified Qwen joint graph requires F16 KV caches";
        if (Environment.GetEnvironmentVariable("TS_Q4E_DISABLE_ARENA_DECODE") == "1")
            return "Qwen arena decode is disabled by TS_Q4E_DISABLE_ARENA_DECODE=1";
        if (engine == null || !ReferenceEquals(engine.Model, model))
            return "the standing engine does not belong to the loaded model";
        if (!engine.PrefixCacheActive)
            return "the qualified shared prefix cache is disabled";
        return engine.JointDecodeWarmupDeclineReason(3);
    }

    /// <summary>The host calls before binding its port, after both serial
    /// thinking-on prefix is initialized. Never borrow a lifecycle/custom compute gate.
    /// The temporary engine-only gate lets chat preparation submit all cohort
    /// members before any can finish, then releases the ordinary scheduler.</summary>
    public static async Task<Result> RunAsync(InferenceEngine engine,
        Func<JsonElement, CancellationToken, IAsyncEnumerable<object>> frames,
        Func<string> createSession, Action<string> releaseSession, int width, bool think, ILogger? logger,
        CancellationToken cancellationToken = default, TimeSpan? timeout = null, TimeSpan? drainTimeout = null)
    {
        ArgumentNullException.ThrowIfNull(engine);
        ArgumentNullException.ThrowIfNull(frames);
        ArgumentNullException.ThrowIfNull(createSession);
        ArgumentNullException.ThrowIfNull(releaseSession);
        if (width is < 2 or > 3) throw new ArgumentOutOfRangeException(nameof(width));
        TimeSpan limit = timeout ?? TimeSpan.FromSeconds(120), drain = drainTimeout ?? TimeSpan.FromSeconds(30);
        if (limit <= TimeSpan.Zero || drain <= TimeSpan.Zero) throw new ArgumentOutOfRangeException(nameof(timeout));
        var clock = Stopwatch.StartNew();
        string? decline = engine.JointDecodeWarmupDeclineReason(width);
        if (decline != null)
            return new(width, false, false, true, null, "joint warmup skipped: " + decline, clock.Elapsed);
        if (engine.Model is not IBatchedPagedModel batched)
            return new(width, false, false, true, null, "joint warmup skipped: model has no fused batch evidence", clock.Elapsed);

        var sessions = new List<string>(width);
        var requests = new List<Task<PrefixCacheWarmup.Result>>(width);
        using var requestCancellation = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        requestCancellation.CancelAfter(limit);
        var gate = new ComputeGate(); gate.Close();
        decline = await InstallStartupGateAsync(engine, gate, requestCancellation.Token,
            TimeSpan.FromSeconds(Math.Min(30, limit.TotalSeconds))).ConfigureAwait(false);
        if (decline != null)
            return new(width, false, false, true, null, "joint warmup skipped: " + decline, clock.Elapsed);
        bool requestsCompleted = false;
        string? detail = null;
        FusedBatchExecutionEvidence? evidence = null;
        InferenceEngine.StartupPrefillCohort? cohort = null;
        ulong previous = batched.LastAcceptedFusedBatch?.NativeAcceptedExecutionId ?? 0;
        long submitted = engine.TotalSubmitted;
        try
        {
            var submissionClock = Stopwatch.StartNew();
            for (int i = 0; i < width; ++i)
            {
                requestCancellation.Token.ThrowIfCancellationRequested();
                string id = createSession();
                if (string.IsNullOrWhiteSpace(id) || id == SessionManager.DefaultSessionId || sessions.Contains(id, StringComparer.Ordinal))
                    throw new InvalidOperationException("Joint warmup requires distinct, newly owned sessions.");
                sessions.Add(id);
                requests.Add(PrefixCacheWarmup.RunAsync(frames, id, think, logger, requestCancellation.Token, maxTokens: 2));
            }
            while (engine.TotalSubmitted - submitted < width)
            {
                requestCancellation.Token.ThrowIfCancellationRequested();
                if (submissionClock.Elapsed >= TimeSpan.FromSeconds(30))
                    throw new TimeoutException("The complete joint warmup cohort was not submitted within 30 seconds.");
                if (requests.Any(t => t.IsCompleted))
                    throw new InvalidOperationException("A joint warmup request ended before the entire cohort was submitted.");
                await Task.Delay(5, requestCancellation.Token).ConfigureAwait(false);
            }
            var ownedSessions = new HashSet<string>(sessions, StringComparer.Ordinal);
            var matchedSessions = new HashSet<string>(StringComparer.Ordinal);
            cohort = engine.BeginStartupPrefillCohort(gate, width,
                sequence => sequence.UserTag is ChatSession session && ownedSessions.Contains(session.Id)
                    && matchedSessions.Add(session.Id));
            gate.Open();
            var completed = await Task.WhenAll(requests).WaitAsync(requestCancellation.Token).ConfigureAwait(false);
            requestsCompleted = completed.All(r => r.Warmed);
            detail = requestsCompleted ? "short startup requests completed" : string.Join("; ", completed.Where(r => !r.Warmed).Select(r => r.Detail));
            evidence = batched.LastAcceptedFusedBatch;
        }
        catch (OperationCanceledException) { detail = "joint warmup cancelled or timed out"; }
        catch (Exception error) { detail = "joint warmup failed: " + error.Message; }
        finally
        {
            // Opening wakes cancellation/release commands as well as execution.
            // Restore only our gate; never overwrite a concurrently installed
            // external lifecycle gate or attempt to force it open.
            requestCancellation.Cancel();
            cohort?.Dispose();
            gate.Open();
            engine.TryReplaceComputeGate(gate, null);
        }

        // If native work is still draining, this task retains the lease until
        // every stream has truly finished and the compute lock observes idle.
        // A bounded caller wait never releases an in-flight workspace early.
        Task<bool> cleanup = DrainAndReleaseAsync(engine, requests, sessions, releaseSession, logger);
        bool cleaned = false;
        try { cleaned = await cleanup.WaitAsync(drain).ConfigureAwait(false); }
        catch (TimeoutException) { detail += "; cleanup still pending; owned sessions/workspaces retained until completion"; }
        bool accepted = evidence is { ActiveSequenceCount: var count, NativeNodeCount: > 0, NativeAcceptedExecutionId: > 0 }
            && count == width && evidence.NativeAcceptedExecutionId > previous && cohort?.DeclineReason == null;
        if (cohort?.DeclineReason is { } cohortDecline) detail += "; " + cohortDecline;
        if (requestsCompleted && !accepted) detail += "; no fresh native fused batch of the requested width was observed";
        if (!cleaned && cleanup.IsCompleted) detail += "; session/workspace cleanup failed";
        var result = new Result(width, requestsCompleted, accepted, cleaned, accepted ? evidence : null,
            detail ?? "joint startup warmup did not complete", clock.Elapsed)
            { CleanupPending = !cleanup.IsCompleted };
        logger?.LogInformation(LogEventIds.HostConfiguration,
            "Scheduled startup joint warmup: width={Width} warmed={Warmed} requests={RequestsCompleted} native={NativeBatchAccepted} cleanup={CleanupCompleted} nativeId={NativeId} nodes={Nodes} elapsedMs={ElapsedMs:F1}; {Detail}",
            width, result.Warmed, requestsCompleted, accepted, cleaned, result.Evidence?.NativeAcceptedExecutionId ?? 0,
            result.Evidence?.NativeNodeCount ?? 0, result.Elapsed.TotalMilliseconds, detail);
        return result;
    }

    private static async Task<string?> InstallStartupGateAsync(InferenceEngine engine, ComputeGate gate,
        CancellationToken cancellation, TimeSpan fenceTimeout)
    {
        var clock = Stopwatch.StartNew();
        while (true)
        {
            if (cancellation.IsCancellationRequested) return "startup fence wait cancelled or timed out";
            if (engine.ComputeGate != null) return "engine already has a lifecycle or custom compute gate";
            if (engine.RunningCount != 0 || engine.WaitingCount != 0
                || engine.PendingRequestCount != 0)
                return "engine has active or pending requests";
            // A done frame can be consumed while the worker still owns the
            // previous step's compute lock. Wait for that physical fence rather
            // than silently skipping the next startup cohort. Installation and
            // the final idle check occur under the same model lock.
            if (Monitor.TryEnter(engine.Model.GpuComputeLock))
            {
                try
                {
                    if (engine.RunningCount != 0 || engine.WaitingCount != 0
                        || engine.PendingRequestCount != 0)
                        return "engine acquired active or pending requests before the startup fence";
                    if (engine.PendingSequenceReleaseCount != 0)
                        return "a previous model sequence release remains quarantined";
                    return engine.TryReplaceComputeGate(null, gate) ? null
                        : "a lifecycle or custom compute gate was installed before the startup fence";
                }
                finally { Monitor.Exit(engine.Model.GpuComputeLock); }
            }
            if (clock.Elapsed >= fenceTimeout) return "previous model work did not reach the startup fence within the timeout";
            try { await Task.Delay(5, cancellation).ConfigureAwait(false); }
            catch (OperationCanceledException) { return "startup fence wait cancelled or timed out"; }
        }
    }

    private static bool IsFencedIdle(InferenceEngine engine)
    {
        object computeLock = engine.Model.GpuComputeLock;
        if (!Monitor.TryEnter(computeLock)) return false;
        try { return engine.RunningCount == 0 && engine.WaitingCount == 0
                && engine.PendingRequestCount == 0 && engine.PendingSequenceReleaseCount == 0; }
        finally { Monitor.Exit(computeLock); }
    }

    private static async Task<bool> DrainAndReleaseAsync(InferenceEngine engine,
        IReadOnlyList<Task<PrefixCacheWarmup.Result>> requests, IReadOnlyList<string> sessions,
        Action<string> release, ILogger? logger)
    {
        try { await Task.WhenAll(requests).ConfigureAwait(false); }
        catch (Exception error) { logger?.LogWarning(LogEventIds.HostConfiguration, "Joint warmup stream drain failed: {Error}", error.Message); }
        while (!IsFencedIdle(engine)) await Task.Delay(5).ConfigureAwait(false);
        bool success = true;
        foreach (string id in sessions)
            try { release(id); }
            catch (Exception error)
            {
                success = false;
                logger?.LogWarning(LogEventIds.HostConfiguration, "Joint warmup cleanup failed for {SessionId}: {Error}", id, error.Message);
            }
        return success;
    }
}
