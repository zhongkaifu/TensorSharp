// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using TensorSharp.Memory;

namespace TensorSharp.Runtime.Paged;

/// <summary>Single-engine snapshot adapter. Block ids are recycled only after the
/// pool's final reference is released; each reuse gets a fresh resource epoch.</summary>
internal sealed class TieredKvSnapshots : IDisposable
{
    private readonly string _owner = "kv/" + Guid.NewGuid().ToString("N");
    private readonly Dictionary<int, ResourceKey> _keys = new();
    private readonly HostMemoryBackend _host;
    private readonly BoundedTransfers _transfers;
    private readonly SsdSpillStore _spill;
    private readonly IResourceBuffer _scratch;
    private readonly BudgetReservation _scratchCharge;
    private readonly int _blockBytes;
    private long _epoch;
    private bool _disposed;

    public MemoryBudget Budget { get; }
    public TieredMemoryScheduler Scheduler { get; }

    public TieredKvSnapshots(long blockBytes, KvSnapshotOptions options)
    {
        options.Validate(blockBytes);
        _blockBytes = checked((int)blockBytes);
        Budget = options.SharedBudget ?? new(new[]
        {
            new MemoryCharge(options.RamPool, options.RamBytes),
            new MemoryCharge(options.SsdPool, options.SsdBytes),
        });
        _host = new(options.RamPool);
        _transfers = new(Budget, options.RamPool, options.TransferBytes, 1);
        try
        {
            _spill = new(Budget, options.SsdPool, options.SpillDirectory, _transfers);
            try
            {
                _scratchCharge = Budget.Reserve(_host.GetAllocationCharges(blockBytes));
                try
                {
                    _scratch = _host.AllocateAsync(blockBytes).GetAwaiter().GetResult();
                    _scratchCharge.Commit();
                }
                catch { _scratchCharge.Dispose(); throw; }
            }
            catch { _spill.Dispose(); throw; }
        }
        catch { _transfers.Dispose(); throw; }
        Scheduler = new(Budget, new[] { _host }, _transfers, _spill, _host.Location);
    }

    public KvSnapshotLease Acquire(int blockId, ResourceAccess access)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_keys.TryGetValue(blockId, out var key))
        {
            key = new(_owner, checked(++_epoch), blockId.ToString(System.Globalization.CultureInfo.InvariantCulture));
            Scheduler.Register(new(key, _blockBytes, ResourceKind.KvPage, Mutable: true));
            _keys.Add(blockId, key);
        }
        var lease = Scheduler.AcquireAsync(key, _host.Location, access).AsTask().GetAwaiter().GetResult();
        return new KvSnapshotLease(lease, access);
    }

    public unsafe Span<byte> CaptureScratch(int bytes)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if ((uint)bytes > (uint)_blockBytes) throw new ArgumentOutOfRangeException(nameof(bytes));
        return new Span<byte>((void*)_scratch.Pointer, bytes);
    }

    public Task<bool> TryPrefetch(int blockId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _keys.TryGetValue(blockId, out var key)
            ? Scheduler.TryPrefetchAsync(key, _host.Location).AsTask() : Task.FromResult(false);
    }

    public void Release(int blockId)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (!_keys.TryGetValue(blockId, out var key)) return;
        Scheduler.Unregister(key); // Refuses live spans; retain the key if cleanup fails.
        _keys.Remove(blockId);
    }

    public void Dispose()
    {
        if (_disposed) return;
        Scheduler.DisposeAsync().GetAwaiter().GetResult();
        _scratch.Dispose();
        _scratchCharge.Dispose();
        _spill.Dispose();
        _transfers.Dispose();
        _keys.Clear();
        _disposed = true;
    }
}

/// <summary>Owns a pinned snapshot for one synchronous extract/inject call. Spans
/// must not escape this lease; model APIs must finish using the span before returning.</summary>
public sealed class KvSnapshotLease : IDisposable
{
    private readonly byte[]? _managed;
    private readonly ResourceLease? _resource;
    private readonly ResourceAccess _access;
    private bool _disposed;
    internal KvSnapshotLease(byte[] bytes, ResourceAccess access) { _managed = bytes; _access = access; }
    internal KvSnapshotLease(ResourceLease resource, ResourceAccess access) { _resource = resource; _access = access; }
    private unsafe Span<byte> Bytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _resource == null ? _managed.AsSpan() : new Span<byte>((void*)_resource.Pointer, checked((int)_resource.ByteLength));
        }
    }
    public ReadOnlySpan<byte> ReadOnlySpan => Bytes;
    public Span<byte> Span => _access == ResourceAccess.Write ? Bytes
        : throw new InvalidOperationException("A writable snapshot lease is required.");
    public void Dispose()
    {
        if (_disposed) return;
        _resource?.Dispose();
        _disposed = true;
    }
}
