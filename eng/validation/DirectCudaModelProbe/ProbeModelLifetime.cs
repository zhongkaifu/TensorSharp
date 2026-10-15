// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Globalization;
using System.Runtime.ExceptionServices;

namespace DirectCudaProbe;

internal static class HostStagingBudgetOption
{
    public static long? Parse(string? value, bool ggmlBackend)
    {
        if (value is null) return null;
        if (!ggmlBackend || !long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out long bytes) || bytes < 0)
            throw new ArgumentException("--ggml-host-staging-budget-bytes requires a nonnegative byte count and a GGML backend; 0 is a strict zero quota.");
        return bytes;
    }
}

internal readonly record struct ProbeLifetimeState(bool ModelCreated, bool ModelDisposed,
    bool ScopeDisposed, Exception? Error);

// The optional native callback scope must outlive model construction, execution,
// and successful model cleanup. Failed cleanup keeps the scope reachable for a
// retry. The observer persists diagnostics; its errors cannot hide owner errors.
internal sealed class ProbeModelLifetime<T> : IDisposable where T : class, IDisposable
{
    private readonly IDisposable? _scope;
    private readonly Action<ProbeLifetimeState>? _observe;
    private bool _modelCreated, _modelDisposed, _scopeDisposed, _disposed;
    private Exception? _history;
    public T Model { get; private set; } = null!;

    public ProbeModelLifetime(Func<T> create, Func<IDisposable?>? attachScope = null,
        Action<ProbeLifetimeState>? observe = null)
    {
        _observe = observe;
        // This owner exists before attachment publishes native callbacks, so
        // subsequent model-factory failures always reach scope cleanup.
        try { _scope = attachScope?.Invoke(); Model = create(); _modelCreated = true; }
        catch (Exception error)
        {
            Finish(error);
            throw;
        }
    }

    public void Dispose()
    {
        if (!_disposed) Finish(null);
    }

    private void Finish(Exception? initialError)
    {
        Exception? failure = initialError;
        void Record(Exception error)
        {
            failure = failure is null ? error : new AggregateException(failure, error);
        }
        try
        {
            if (_modelCreated && !_modelDisposed) { Model.Dispose(); _modelDisposed = true; }
            // A constructor failure has no returned model to dispose. Native
            // detach still refuses any leaked covered owner rather than hiding it.
            if (!_scopeDisposed) { _scope?.Dispose(); _scopeDisposed = true; }
            _disposed = true;
        }
        catch (Exception error) { Record(error); }
        if (failure != null) _history = _history is null ? failure : new AggregateException(_history, failure);
        try { _observe?.Invoke(new(_modelCreated, _modelDisposed, _scopeDisposed, _history)); }
        catch (Exception error) { Record(error); }
        if (failure != null) ExceptionDispatchInfo.Capture(failure).Throw();
    }
}
