// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using TensorSharp.Memory;

namespace TensorSharp.Models;

/// <summary>Optional immutable range copies. Callers copy into their existing
/// staging before using a pointer, so trim never invalidates a pending CUDA or
/// file transfer. Source identity is by reference and belongs to one model load.
/// Misses do not evict useful ranges: sequential dense scans larger than RAM
/// would otherwise replace every entry before its next use. Pressure trimming
/// releases least-recently-used entries. This is not a globally optimal policy.</summary>
internal sealed class StreamingWeightReadCache : IDisposable
{
    private readonly object _gate = new();
    private readonly WeightStreamingOptions _options;
    private readonly Dictionary<Key, LinkedListNode<Entry>> _entries = new();
    private readonly LinkedList<Entry> _lru = new();
    private bool _disposed;
    private long _bytes, _peak, _hitBytes, _hits, _evicted;

    internal StreamingWeightReadCache(WeightStreamingOptions options) => _options = options;

    internal (long Bytes, long Peak, long HitBytes, long Hits, long Evicted) Statistics
    { get { lock (_gate) return (_bytes, _peak, _hitBytes, _hits, _evicted); } }

    internal bool TryCopy(IResourceSource source, long offset, Span<byte> destination)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_entries.TryGetValue(new(source, offset, destination.Length), out var entry)) return false;
            entry.Value.Buffer.GetSpan().CopyTo(destination);
            _lru.Remove(entry); _lru.AddLast(entry);
            _hits++; _hitBytes = checked(_hitBytes + destination.Length);
            return true;
        }
    }

    internal void Store(IResourceSource source, long offset, ReadOnlySpan<byte> data)
    {
        if (_options.HostCacheBytes == 0 || data.IsEmpty) return;
        ArgumentNullException.ThrowIfNull(source);
        if (offset < 0 || offset > source.ByteLength || data.Length > source.ByteLength - offset)
            throw new ArgumentOutOfRangeException(nameof(offset));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var key = new Key(source, offset, data.Length);
            // Bound managed index overhead too; that overhead is covered by
            // runtime headroom, not the exact aligned-payload charge below.
            if (_entries.ContainsKey(key) || _entries.Count >= 4096) return;
            long allocated = checked(((long)data.Length + 63) / 64 * 64);
            long available = _options.Budget.Snapshot().Single(p => p.Pool == _options.HostPool).Available;
            if (allocated > _options.HostCacheBytes - _bytes || allocated > available - _options.HostCacheReserveBytes)
                return;
            StreamingHostBuffer buffer = null;
            try
            {
                buffer = new StreamingHostBuffer(_options, data.Length);
                data.CopyTo(buffer.GetSpan());
                var entry = new Entry(key, buffer);
                var node = _lru.AddLast(entry);
                try { _entries.Add(key, node); }
                catch { _lru.Remove(node); throw; }
                _bytes = checked(_bytes + buffer.AllocatedBytes);
                _peak = Math.Max(_peak, _bytes);
                buffer = null; // Published and now owned by the cache.
            }
            catch (MemoryPressureException) { /* Another owner won; the original staging is still valid. */ }
            catch (OutOfMemoryException) { /* Optional retention must not fail an already completed read. */ }
            finally { ((IDisposable)buffer)?.Dispose(); }
        }
    }

    internal long TrimTo(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        lock (_gate)
        {
            long before = _bytes;
            while (_bytes > bytes && _lru.First is { } first)
            {
                ((IDisposable)first.Value.Buffer).Dispose();
                _bytes -= first.Value.Buffer.AllocatedBytes;
                _evicted += first.Value.Buffer.AllocatedBytes;
                _entries.Remove(first.Value.Key); _lru.RemoveFirst();
            }
            return before - _bytes;
        }
    }

    internal void LeaveAvailable(long requiredBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(requiredBytes);
        lock (_gate)
        {
            long available = _options.Budget.Snapshot().Single(p => p.Pool == _options.HostPool).Available;
            if (requiredBytes > available) TrimTo(Math.Max(0, _bytes - (requiredBytes - available)));
        }
    }

    public void Dispose()
    {
        lock (_gate) { if (_disposed) return; TrimTo(0); _disposed = true; }
    }

    private sealed record Entry(Key Key, StreamingHostBuffer Buffer);
    private readonly struct Key(IResourceSource source, long offset, int length) : IEquatable<Key>
    {
        private readonly IResourceSource _source = source;
        private readonly long _offset = offset;
        private readonly int _length = length;
        public bool Equals(Key other) => ReferenceEquals(_source, other._source) && _offset == other._offset && _length == other._length;
        public override bool Equals(object obj) => obj is Key other && Equals(other);
        public override int GetHashCode() => HashCode.Combine(RuntimeHelpers.GetHashCode(_source), _offset, _length);
    }
}
