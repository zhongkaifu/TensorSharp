// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using DirectCudaProbe;
using TensorSharp.Memory;

static class HostBudgetLifetimeTests
{
    sealed class Owner(Action release) : IDisposable { public void Dispose() => release(); }

    public static void Run(Action<bool, string> check)
    {
        Exception Failure(Action action)
        {
            try { action(); } catch (Exception error) { return error; }
            throw new InvalidOperationException("Expected controlled failure.");
        }
        check(HostStagingBudgetOption.Parse(null, false) is null, "Omitted host quota preserves ordinary CUDA mode.");
        check(HostStagingBudgetOption.Parse(null, true) is null, "Omitted host quota creates no budget request.");
        foreach (long bytes in new[] { 0L, 8L << 20, long.MaxValue })
            check(HostStagingBudgetOption.Parse(bytes.ToString(System.Globalization.CultureInfo.InvariantCulture), true) == bytes,
                "Explicit quota accepts exact nonnegative bytes; zero is strict.");
        foreach (string invalid in new[] { "", " ", "-1", "1.5", "8MiB", "9223372036854775808" })
            check(Failure(() => HostStagingBudgetOption.Parse(invalid, true)) is ArgumentException, "Invalid host quota rejected before loading.");
        check(Failure(() => HostStagingBudgetOption.Parse("8388608", false)) is ArgumentException, "Host quota requires GGML.");

        int ordinaryDisposals = 0;
        var ordinary = new ProbeModelLifetime<Owner>(() => new(() => ordinaryDisposals++));
        ordinary.Dispose(); ordinary.Dispose();
        check(ordinaryDisposals == 1, "No-scope ordinary model retains exactly one successful disposal.");

        var ledger = new MemoryBudget([new("host", 8)]);
        var order = new List<string>();
        ProbeLifetimeState last = default;
        var success = new ProbeModelLifetime<Owner>(() =>
        {
            order.Add("create");
            var charge = ledger.Reserve([new("host", 8)]); charge.Commit();
            return new(() => { order.Add("model-dispose"); charge.Dispose(); });
        }, () =>
        {
            order.Add("attach");
            return new Owner(() =>
            {
                check(ledger.Snapshot().Single().Committed == 0, "Detach happens only after the model releases credit.");
                order.Add("detach");
            });
        }, state => { last = state; order.Add("observe"); });
        success.Dispose(); success.Dispose();
        check(order.SequenceEqual(new[] { "attach", "create", "model-dispose", "detach", "observe" }), "Native scope surrounds exact model lifetime.");
        check(last.ModelCreated && last.ModelDisposed && last.ScopeDisposed && last.Error is null, "Successful final ownership observation is complete.");
        check(ledger.HighWatermarks().Single().Committed == 8 && ledger.Snapshot().Single().Committed == 0, "Peak survives complete owner release.");

        var createError = new IOException("controlled create failure");
        int detached = 0;
        Exception observed = Failure(() => new ProbeModelLifetime<Owner>(() => throw createError,
            () => new Owner(() => detached++), state => last = state));
        check(ReferenceEquals(createError, observed) && detached == 1, "Constructor failure detaches empty scope and preserves the original error.");
        check(!last.ModelCreated && !last.ModelDisposed && last.ScopeDisposed && ReferenceEquals(last.Error, createError), "Failed construction cannot masquerade as a released model.");

        var modelError = new IOException("controlled model cleanup failure");
        int modelCalls = 0, scopeCalls = 0;
        var retryModel = new ProbeModelLifetime<Owner>(() => new(() => { if (++modelCalls == 1) throw modelError; }),
            () => new Owner(() => scopeCalls++), state => last = state);
        check(ReferenceEquals(modelError, Failure(retryModel.Dispose)), "Model cleanup error remains visible.");
        check(scopeCalls == 0 && !last.ModelDisposed && !last.ScopeDisposed, "Failed model cleanup keeps callbacks attached.");
        retryModel.Dispose();
        check(modelCalls == 2 && scopeCalls == 1 && last.ModelDisposed && last.ScopeDisposed, "Cleanup retry releases model before scope.");
        check(ReferenceEquals(last.Error, modelError), "Successful retry retains earlier lifecycle error evidence.");

        var detachError = new IOException("controlled detach failure");
        modelCalls = scopeCalls = 0;
        var retryScope = new ProbeModelLifetime<Owner>(() => new(() => modelCalls++),
            () => new Owner(() => { if (++scopeCalls == 1) throw detachError; }), state => last = state);
        check(ReferenceEquals(detachError, Failure(retryScope.Dispose)), "Scope cleanup error remains visible.");
        retryScope.Dispose();
        check(modelCalls == 1 && scopeCalls == 2, "Scope retry does not dispose the successfully released model twice.");

        var reportError = new IOException("controlled report failure");
        var combined = Failure(() => new ProbeModelLifetime<Owner>(() => throw createError,
            () => new Owner(() => throw detachError), _ => throw reportError));
        var errors = (combined as AggregateException)?.Flatten().InnerExceptions;
        check(errors is { Count: 3 } && errors.Contains(createError) && errors.Contains(detachError) && errors.Contains(reportError),
            "Create, detach and evidence-write failures are all retained.");

        int unexpectedCreate = 0;
        check(ReferenceEquals(detachError, Failure(() => new ProbeModelLifetime<Owner>(() =>
        {
            unexpectedCreate++; return new(() => { });
        }, () => throw detachError, state => last = state))), "Attach failure preserves its original error.");
        check(unexpectedCreate == 0 && !last.ModelCreated, "No model is constructed after failed attachment.");
    }
}
