// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;

namespace TensorSharp.Models;

internal sealed partial class WeightStreamingExecutor
{
    // Access is serialized by the executor gate, including use, trim and reset.
    // Keys are model-owned immutable weight identities, never reusable addresses.
    private readonly Dictionary<QuantizedWeight, LinkedListNode<RetainedWeight>> _deviceWeights = new(ReferenceEqualityComparer.Instance);
    private readonly LinkedList<RetainedWeight> _deviceLru = new();
    private long _deviceCacheBytes, _peakDeviceCacheBytes, _deviceCacheHits, _deviceCacheHitBytes;
    private long _deviceCacheEvictedBytes, _weightUploadBytes, _peakDeviceOwnedBytes;
    private bool _deviceCacheFaulted;

    private sealed record RetainedWeight(QuantizedWeight Weight, int Rank, GgmlResidentWeightSession Session);

    private void ObserveDevice(long workspace)
    {
        _peakDevice = Math.Max(_peakDevice, workspace);
        _peakDeviceOwnedBytes = Math.Max(_peakDeviceOwnedBytes, checked(_deviceCacheBytes + _workspaceBytes + workspace));
    }

    private unsafe bool TryRetainedLinear(QuantizedWeight weight, IntPtr input, IntPtr output, int tokens, int rank)
    {
        if (_options.DeviceCacheBytes == 0 || _options.DevicePools.Contains(_options.HostPool)
            || (_arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda && tokens != 1)) return false;

        bool loaded = false;
        if (!_deviceWeights.TryGetValue(weight, out var node))
        {
            if (_deviceWeights.Count >= 256 || weight.Ne1 > int.MaxValue
                || weight.RawBytes > _options.DeviceCacheBytes - _deviceCacheBytes) return false;
            // Bound metadata and reject oversized matrices before asking the
            // native complete-matrix API; row streaming can support larger shapes.
            int capacity = _arithmetic == GgmlWeightStreamingArithmetic.FullPrecision ? Math.Min(65535, _options.TokenTileRows) : 1;
            int rows = checked((int)weight.Ne1);
            if (!FitsResidentIndexing(weight.Ne0, rows, capacity, capacity, weight.GgmlType, _arithmetic)
                || (_arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda && weight.GgmlType == 1 && rows > int.MaxValue - 31))
                return false; // Complete retention must not narrow row-streaming indexing support.
            long payload = GgmlResidentWeightSession.GetPayloadBytes(rank, weight.GgmlType, weight.Ne0, rows, capacity, _arithmetic);
            var available = _options.Budget.Snapshot().ToDictionary(p => p.Pool, p => p.Available, StringComparer.Ordinal);
            long workspaceHoldout = Math.Max(0, WorkspaceRetentionLimit(available) - _workspaceBytes);
            // Scan resistance: a miss cannot replace a useful retained matrix.
            // Pressure trim below is LRU, and never revokes an in-use address.
            if (payload > _options.DeviceCacheBytes - _deviceCacheBytes
                || _options.DevicePools.Any(p => payload > (decimal)available[p] - _options.DeviceCacheReserveBytes - workspaceHoldout)) return false;
            try
            {
                try
                {
                    _activeResidentSession = new(_options.Budget, _options.DevicePools, rank,
                        weight.GgmlType, weight.Ne0, rows, capacity, _arithmetic);
                }
                catch (GgmlResidentWeightAllocationException failure)
                { _activeResidentSession = failure.UnreleasedSession; throw; }
                catch (MemoryPressureException) { return false; } // Another owner won before allocation.
                _sessionCreations++;
                ObserveDevice(_activeResidentSession.PayloadBytes);
                int tileRows = checked((int)Math.Min(weight.Ne1, _options.TileBytes / weight.StreamingRowBytes));
                foreach (var tile in ReadTiles(weight, tileRows))
                {
                    _activeResidentSession.UploadWeightRows(tile.Pointer, checked((int)tile.FirstRow), tile.Rows);
                    _weightUploadBytes = checked(_weightUploadBytes + tile.Rows * weight.StreamingRowBytes);
                    _linearTiles++;
                }
                var entry = new RetainedWeight(weight, rank, _activeResidentSession);
                node = _deviceLru.AddLast(entry);
                try { _deviceWeights.Add(weight, node); }
                catch { _deviceLru.Remove(node); throw; }
                _deviceCacheBytes = checked(_deviceCacheBytes + payload);
                _peakDeviceCacheBytes = Math.Max(_peakDeviceCacheBytes, _deviceCacheBytes);
                _activeResidentSession = null; // Published only after every weight row arrived.
                loaded = true;
                // These entries serve all token counts, so RAM should retain
                // other sources instead of duplicating the same original bytes.
                if (_arithmetic == GgmlWeightStreamingArithmetic.FullPrecision) _hostCache?.RemoveSource(weight.FileSource);
            }
            catch (Exception original)
            {
                try { ReleaseSession(); }
                catch (Exception cleanup) { throw new AggregateException(original, cleanup); }
                throw;
            }
        }
        if (node.Value.Rank != rank) throw new InvalidOperationException("A retained weight belongs to a different CUDA rank.");
        var session = node.Value.Session;
        try
        {
            // Output downloads stay bounded even for a complete large device
            // matrix. No full-size host weight or output staging is introduced.
            var (tileRows, tokenRows) = SelectTileLayout(_options, weight.Ne0, weight.Ne1,
                Math.Min(tokens, session.TokenCount), (_, _, _) => 0, weight.GgmlType, _arithmetic);
            using var tileOutput = new StreamingHostBuffer(_options, checked(tileRows * tokenRows * sizeof(float)));
            _peakHost = Math.Max(_peakHost, checked(HostReadBytes + tileOutput.AllocatedBytes));
            for (int token = 0; token < tokens;)
            {
                int count = Math.Min(tokenRows, tokens - token);
                session.UploadInputTokens(input + checked((nint)((long)token * weight.Ne0 * sizeof(float))), 0, count);
                _inputUploads++;
                session.Project(count); _matrixProjections++;
                for (int row = 0; row < session.OutputRows;)
                {
                    int rows = Math.Min(tileRows, session.OutputRows - row);
                    session.Download(tileOutput.Pointer, 0, count, row, rows);
                    for (int t = 0; t < count; t++)
                    {
                        long destination = checked(((long)(token + t) * weight.Ne1 + row) * sizeof(float));
                        long source = checked((long)t * rows * sizeof(float));
                        long bytes = checked((long)rows * sizeof(float));
                        Buffer.MemoryCopy((byte*)tileOutput.Pointer + source, (byte*)output + destination, bytes, bytes);
                    }
                    row = checked(row + rows);
                }
                if (!loaded) { _deviceCacheHits++; _deviceCacheHitBytes = checked(_deviceCacheHitBytes + weight.RawBytes); }
                loaded = false;
                token = checked(token + count);
            }
            _deviceLru.Remove(node); _deviceLru.AddLast(node);
            return true;
        }
        catch { _deviceCacheFaulted = true; throw; } // Reset retires failed owners before replay.
    }

    private long TrimDeviceCache(long target)
    {
        long before = _deviceCacheBytes;
        try
        {
            while (_deviceCacheBytes > target && _deviceLru.First is { } node)
            {
                var entry = node.Value;
                entry.Session.Dispose(); // Drain and free before refund/removal; failure preserves this entry.
                _deviceCacheBytes -= entry.Session.PayloadBytes;
                _deviceCacheEvictedBytes = checked(_deviceCacheEvictedBytes + entry.Session.PayloadBytes);
                _deviceWeights.Remove(entry.Weight); _deviceLru.RemoveFirst();
            }
        }
        catch { _deviceCacheFaulted = true; throw; }
        return before - _deviceCacheBytes;
    }

    private void LeaveDeviceAvailable(long required)
    {
        while (_deviceCacheBytes > 0 || _workspaceBytes > 0)
        {
            var available = _options.Budget.Snapshot().ToDictionary(p => p.Pool, p => p.Available, StringComparer.Ordinal);
            long shortage = _options.DevicePools.Max(p => Math.Max(0, required - available[p]));
            if (shortage == 0) return;
            if (_workspaceBytes > 0)
            {
                TrimWorkspaces(Math.Max(0, _workspaceBytes - shortage));
                continue;
            }
            TrimDeviceCache(Math.Max(0, _deviceCacheBytes - shortage));
        }
    }
}
