// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.GGML;

namespace TensorSharp.Models;

internal sealed partial class WeightStreamingExecutor
{
    // Only idle, successfully completed sessions enter this list. The executor
    // gate serializes borrowing, use, return and physical release.
    private readonly LinkedList<Workspace> _workspaces = new();
    private long _workspaceBytes, _peakWorkspaceBytes, _workspaceReuses, _workspaceEvictedBytes;
    private long _workspaceEpoch;
    private bool _workspaceFaulted;
    private sealed record Workspace(int Rank, int LogicalTokens, long LogicalRows, long Epoch, GgmlWeightStreamingSession Session);

    private long WorkspaceRetentionLimit(IReadOnlyDictionary<string, long> available, long active = 0)
    {
        if (_options.DeviceWorkspaceCacheBytes == 0 || _options.DevicePools.Contains(_options.HostPool)) return 0;
        // Split discretionary capacity after other owners and execution headroom.
        // Include our existing optional owners so the limit does not oscillate
        // merely because an arena moves between active and idle states.
        decimal slack = _options.DevicePools.Min(pool => (decimal)available[pool])
            + _deviceCacheBytes + _workspaceBytes + active
            - Math.Max(_options.DeviceCacheReserveBytes, _options.DeviceWorkspaceCacheReserveBytes);
        return (long)Math.Min(_options.DeviceWorkspaceCacheBytes, Math.Max(0, slack) / 4);
    }

    private void BorrowWorkspace(QuantizedWeight weight, int rank, int logicalTokens, int rows, int tokens)
    {
        LinkedListNode<Workspace> best = null;
        for (var node = _workspaces.First; node != null; node = node.Next)
        {
            var entry = node.Value;
            var session = entry.Session;
            if (entry.Rank != rank || session.WeightType != weight.GgmlType || session.InputWidth != weight.Ne0
                || session.MaxTileRows < rows || session.MaxTokenCount < tokens
                || (_arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda
                    && (entry.LogicalTokens != logicalTokens || entry.LogicalRows != weight.Ne1))) continue;
            if (best == null || session.PayloadBytes < best.Value.Session.PayloadBytes) best = node;
        }
        if (best == null) return;
        _activeSession = best.Value.Session;
        _workspaceBytes -= _activeSession.PayloadBytes;
        _workspaces.Remove(best);
        _workspaceReuses++;
    }

    private void ReturnWorkspace(int rank, int logicalTokens, long logicalRows)
    {
        var session = _activeSession;
        if (session == null) return;
        if (_options.DeviceWorkspaceCacheBytes > 0 && !_options.DevicePools.Contains(_options.HostPool))
        {
            var available = _options.Budget.Snapshot().ToDictionary(p => p.Pool, p => p.Available, StringComparer.Ordinal);
            long limit = WorkspaceRetentionLimit(available, session.PayloadBytes);
            if (_workspaces.Count < 16 && session.PayloadBytes <= limit - _workspaceBytes
                && _options.DevicePools.All(pool => available[pool] >= _options.DeviceWorkspaceCacheReserveBytes))
            {
                try
                {
                    _workspaces.AddLast(new Workspace(rank, logicalTokens, logicalRows, _workspaceEpoch, session));
                    _workspaceBytes += session.PayloadBytes;
                    _peakWorkspaceBytes = Math.Max(_peakWorkspaceBytes, _workspaceBytes);
                    _activeSession = null;
                    return;
                }
                catch (OutOfMemoryException) { /* Optional metadata must not fail a completed projection. */ }
            }
        }
        ReleaseSession();
    }

    private void AgeWorkspaces()
    {
        try
        {
            for (var node = _workspaces.First; node != null;)
            {
                var next = node.Next;
                if (node.Value.Epoch < _workspaceEpoch) ReleaseWorkspace(node);
                node = next;
            }
            var available = _options.Budget.Snapshot().ToDictionary(p => p.Pool, p => p.Available, StringComparer.Ordinal);
            TrimWorkspaces(WorkspaceRetentionLimit(available));
            _workspaceEpoch = checked(_workspaceEpoch + 1);
        }
        catch { _workspaceFaulted = true; throw; }
    }

    private void ReleaseWorkspace(LinkedListNode<Workspace> node)
    {
        node.Value.Session.Dispose(); // Failure retains the entry and charged ownership.
        _workspaceBytes -= node.Value.Session.PayloadBytes;
        _workspaceEvictedBytes = checked(_workspaceEvictedBytes + node.Value.Session.PayloadBytes);
        _workspaces.Remove(node);
    }

    private long TrimWorkspaces(long target)
    {
        long before = _workspaceBytes;
        try
        {
            while (_workspaceBytes > target && _workspaces.First is { } node)
            {
                ReleaseWorkspace(node);
            }
        }
        catch { _workspaceFaulted = true; throw; }
        return before - _workspaceBytes;
    }
}
