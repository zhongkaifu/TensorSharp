// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Diagnostics;
using System.Threading;

namespace TensorSharp.Runtime.Scheduling;

/// <summary>Opt-in aggregate timings for snapshot swapping, in stopwatch ticks.
/// Ownership includes its nested phases; phase totals must not be added to it.</summary>
public sealed class KvSwapMetrics
{
    internal enum Phase { Ownership, Extract, Store, Acquire, Prefetch, Inject, Count }
    private readonly long[] _ticks = new long[(int)Phase.Count];
    private readonly long[] _calls = new long[(int)Phase.Count];
    internal readonly bool Enabled = Environment.GetEnvironmentVariable("TS_PROFILE_KV_SWAP") == "1";
    public object Snapshot() => new
    {
        Enabled, Stopwatch.Frequency,
        Phases = Array.ConvertAll(Enum.GetValues<Phase>()[..(int)Phase.Count], p => new
        {
            Phase = p.ToString(), Calls = Interlocked.Read(ref _calls[(int)p]),
            Seconds = Interlocked.Read(ref _ticks[(int)p]) / (double)Stopwatch.Frequency,
        }),
    };
    internal Scope Measure(Phase phase) => Enabled ? new(this, phase) : default;
    internal readonly struct Scope : IDisposable
    {
        private readonly KvSwapMetrics? _owner;
        private readonly Phase _phase;
        private readonly long _start;
        internal Scope(KvSwapMetrics owner, Phase phase)
        { _owner = owner; _phase = phase; _start = Stopwatch.GetTimestamp(); }
        public void Dispose()
        {
            if (_owner == null) return;
            Interlocked.Add(ref _owner._ticks[(int)_phase], Stopwatch.GetTimestamp() - _start);
            Interlocked.Increment(ref _owner._calls[(int)_phase]);
        }
    }
}
