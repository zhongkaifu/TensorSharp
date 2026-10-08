// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections.Concurrent;
using System.Security.Cryptography;

namespace TensorSharp.Memory;

internal interface IChecksummedSource { string Sha256 { get; } }

/// <summary>Reserves the staging escape space up front. Transfer parallelism and RAM
/// are bounded independently of model size, number of tensors and number of requests.</summary>
public sealed class BoundedTransfers : IDisposable
{
    private readonly object _gate = new();
    private readonly SemaphoreSlim _available;
    private readonly ConcurrentQueue<TransferMemory> _buffers = new();
    private readonly BudgetReservation _reservation;
    private int _users;
    private bool _disposed;
    private long _bytesCopied;

    public BoundedTransfers(MemoryBudget budget, string hostPool, int chunkBytes = 1 << 20, int concurrency = 2)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(chunkBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(concurrency);
        ChunkBytes = chunkBytes;
        _available = new SemaphoreSlim(concurrency, concurrency);
        _reservation = budget.Reserve(new[] { new MemoryCharge(hostPool, checked(MemoryRange.Align(chunkBytes, 64) * concurrency)) });
        try
        {
            for (int i = 0; i < concurrency; ++i) _buffers.Enqueue(new TransferMemory(chunkBytes));
            _reservation.Commit();
        }
        catch
        {
            while (_buffers.TryDequeue(out var buffer)) ((IDisposable)buffer).Dispose();
            _reservation.Dispose();
            _available.Dispose();
            throw;
        }
    }
    public int ChunkBytes { get; }
    public long BytesCopied => Interlocked.Read(ref _bytesCopied);

    public async ValueTask CopyAsync(IResourceSource source, IResourceBuffer target, CancellationToken cancellationToken = default)
    {
        if (source.ByteLength != target.ByteLength) throw new ArgumentException("Source and target lengths differ.");
        using var slot = await RentAsync(cancellationToken).ConfigureAwait(false);
        using var hash = source is IChecksummedSource ? IncrementalHash.CreateHash(HashAlgorithmName.SHA256) : null;
        for (long offset = 0; offset < source.ByteLength;)
        {
            int count = (int)Math.Min(ChunkBytes, source.ByteLength - offset);
            var memory = slot.Memory[..count];
            await source.ReadAsync(offset, memory, cancellationToken).ConfigureAwait(false);
            hash?.AppendData(memory.Span);
            await target.WriteAsync(offset, memory, cancellationToken).ConfigureAwait(false);
            Interlocked.Add(ref _bytesCopied, count);
            offset += count;
        }
        if (hash != null && !StringComparer.Ordinal.Equals(Convert.ToHexString(hash.GetHashAndReset()), ((IChecksummedSource)source).Sha256))
            throw new InvalidDataException("Spilled resource checksum mismatch; the destination must not be published.");
    }

    internal async ValueTask<Slot> RentAsync(CancellationToken cancellationToken)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _users++;
        }
        try { await _available.WaitAsync(cancellationToken).ConfigureAwait(false); }
        catch { lock (_gate) _users--; throw; }
        if (!_buffers.TryDequeue(out var buffer)) throw new InvalidOperationException("Transfer-slot accounting failed.");
        return new Slot(this, buffer);
    }

    internal sealed class Slot(BoundedTransfers owner, TransferMemory buffer) : IDisposable
    {
        private BoundedTransfers? _owner = owner;
        internal Memory<byte> Memory => buffer.Memory;
        public void Dispose()
        {
            var current = Interlocked.Exchange(ref _owner, null);
            if (current == null) return;
            current._buffers.Enqueue(buffer);
            current._available.Release();
            lock (current._gate) current._users--;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            if (_users != 0) throw new InvalidOperationException("Drain all transfers before disposing their staging pool.");
            _disposed = true;
            while (_buffers.TryDequeue(out var buffer)) ((IDisposable)buffer).Dispose();
            _available.Dispose();
            _reservation.Dispose();
        }
    }
}
