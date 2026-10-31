// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    /// <summary>
    /// How the direct-CUDA engine reads DeepSeek V4.1's host-mapped Engram tables: each token selects
    /// <c>HashColumns</c> scattered rows per table (24 on the 1017), and a row whose page is not cached is a storage
    /// round trip (about 1.3-1.6 ms on the VMs' MooseFS mount).
    /// <list type="bullet">
    /// <item><b>Gather.</b> The rows of one table for one ubatch are dequantized in parallel over (token, column), so a
    /// decode step's 24 rows are read about 16 at a time instead of one after another. The reads run on this reader's
    /// own threads plus the calling thread, never on the .NET thread pool, which a busy server's request threads could
    /// starve. Every row goes through the same <see cref="ManagedQuantizedOps.DequantizeRowToFloat32"/> into the same
    /// place, so the output is the same bytes whatever the parallelism.</item>
    /// <item><b>Prefetch.</b> When a forward starts, the same threads read one byte of every page every row of the
    /// forward will need, in the order the layers will need them, while the host issues the layers and the GPUs run
    /// them. A gather always goes first: a thread that is reading ahead joins it before its next row. So this reader
    /// never has more than <see cref="Threads"/> rows in flight, the gather's included. It only reads: the gather still
    /// dequantizes every row itself.</item>
    /// <item><b>Advice.</b> The tables are mapped with MADV_RANDOM, so a fault reads one page instead of a read-around
    /// window (55-60 KB per row read was measured without it).</item>
    /// </list>
    /// <para>Settings (read once per load; none changes a result):</para>
    /// <list type="bullet">
    /// <item><c>TS_DSV41_ENGRAM_THREADS</c> (1-32, default min(16, cores)): rows read at once, the calling thread
    /// included (the reader starts <see cref="PoolThreads"/> threads of its own). <c>1</c> restores the S9 gather
    /// exactly (tokens in parallel on the .NET pool, each token's columns one after another) and turns the advice off
    /// unless it is asked for.</item>
    /// <item><c>TS_DSV41_ENGRAM_PREFETCH</c> (0/1; unset: off). Opt-in: what it gains on cold network storage, and
    /// what it costs a cgroup that cannot hold every table, have not been measured on the full model.</item>
    /// <item><c>TS_DSV41_ENGRAM_RANDOM</c> (0/1; unset: on when the threads are more than 1), the native loader's
    /// variable with its meaning. Linux only; a no-op elsewhere.</item>
    /// </list>
    /// </summary>
    internal sealed unsafe class Dsv41EngramReader : IDisposable
    {
        public const int MaxThreads = 32;

        /// <summary>One host-mapped (or received) table.</summary>
        public readonly struct Table
        {
            public readonly byte* Base;
            public readonly int Type;
            public readonly long RowBytes;
            public readonly long Rows;

            public Table(byte* @base, int type, long rowBytes, long rows)
            {
                Base = @base;
                Type = type;
                RowBytes = rowBytes;
                Rows = rows;
            }
        }

        /// <summary>A run of row ids of one table to prefetch: <paramref name="Count"/> ids of
        /// <paramref name="Ids"/> from <paramref name="Offset"/>.</summary>
        public readonly record struct Segment(Table Table, int[] Ids, int Offset, int Count);

        public int Threads { get; }
        public bool Prefetch { get; }
        public bool RandomAdvice { get; }

        /// <summary>The S9 gather (tokens in parallel, columns serial).</summary>
        public bool Legacy => Threads == 1;

        /// <summary>The threads this reader starts for itself (on first use): <see cref="Threads"/> less the gather's
        /// calling thread, and at least one, which the prefetch needs.</summary>
        public int PoolThreads => Math.Max(1, Threads - 1);

        /// <summary>Tests only: called on the reading thread just before each row is read (0: read ahead, 1: gathered,
        /// 2: copied raw for another node).</summary>
        internal Action<int> ReadHookForTests { get; set; }

        public Dsv41EngramReader(int threads, bool prefetch, bool randomAdvice)
        {
            if (threads < 1 || threads > MaxThreads)
                throw new ArgumentOutOfRangeException(nameof(threads), $"TS_DSV41_ENGRAM_THREADS must be in [1, {MaxThreads}]");
            Threads = threads;
            Prefetch = prefetch;
            RandomAdvice = randomAdvice;
        }

        /// <summary>The settings from the environment (or <paramref name="read"/>), refusing a malformed value rather
        /// than guessing.</summary>
        public static Dsv41EngramReader FromEnvironment(Func<string, string> read = null)
        {
            read ??= Environment.GetEnvironmentVariable;
            int threads = Math.Min(16, Math.Max(1, Environment.ProcessorCount));
            string raw = read("TS_DSV41_ENGRAM_THREADS");
            if (raw != null)
            {
                if (!int.TryParse(raw.Trim(), out threads) || threads < 1 || threads > MaxThreads)
                    throw new NotSupportedException($"TS_DSV41_ENGRAM_THREADS must be in [1, {MaxThreads}], got '{raw}'.");
            }
            bool Switch(string name, bool unset)
            {
                string value = read(name);
                return value switch
                {
                    null => unset,
                    "0" => false,
                    "1" => true,
                    _ => throw new NotSupportedException($"{name} must be 0 or 1 (unset selects the default), got '{value}'."),
                };
            }
            return new Dsv41EngramReader(threads, Switch("TS_DSV41_ENGRAM_PREFETCH", false), Switch("TS_DSV41_ENGRAM_RANDOM", threads > 1));
        }

        public string Describe()
            => Legacy
                ? $"S9 gather (tokens in parallel), prefetch {(Prefetch ? "on" : "off")}, random advice {(RandomAdvice ? "on" : "off")}"
                : $"{Threads} reads at once over (token, column) on the reader's own threads, prefetch {(Prefetch ? "on" : "off")}, " +
                  $"random advice {(RandomAdvice ? "on" : "off")}";

        // ------------------------------------------------------------------ gather

        /// <summary>
        /// Dequantizes <paramref name="count"/> tokens' rows: token t's column c is row <c>ids[t * columns + c]</c> of
        /// <paramref name="table"/>, written to <c>dst + (t * columns + c) * headDim</c>.
        /// </summary>
        public void Gather(Table table, int* ids, int count, int columns, int headDim, float* dst)
        {
            if (count <= 0)
                return;
            if (Legacy)
            {
                // The S9 gather, verbatim.
                int rowValues = columns * headDim;
                Parallel.For(0, count, t =>
                {
                    float* rows = dst + (long)t * rowValues;
                    int* mine = ids + (long)t * columns;
                    for (int c = 0; c < columns; c++)
                    {
                        int row = mine[c];
                        if ((uint)row >= (uint)table.Rows)
                            throw new InvalidOperationException("[dsv4-cuda] Engram lookup is out of bounds");
                        ManagedQuantizedOps.DequantizeRowToFloat32(
                            table.Type, (IntPtr)(table.Base + row * table.RowBytes),
                            rows + c * headDim, headDim);
                    }
                });
                return;
            }
            Action<int> hook = ReadHookForTests;
            Run(checked(count * columns), item =>
            {
                hook?.Invoke(1);
                GatherOne(table, ids, dst, headDim, item);
            });
        }

        private static void GatherOne(Table table, int* ids, float* dst, int headDim, int item)
        {
            int row = ids[item];
            if ((uint)row >= (uint)table.Rows)
                throw new InvalidOperationException("[dsv4-cuda] Engram lookup is out of bounds");
            ManagedQuantizedOps.DequantizeRowToFloat32(
                table.Type, (IntPtr)(table.Base + row * table.RowBytes), dst + (long)item * headDim, headDim);
        }

        /// <summary>Copies the raw (quantized) rows <c>ids[0 .. items)</c> of <paramref name="table"/> to
        /// <paramref name="dst"/>, back to back: what an owning node sends a reading node.</summary>
        public void CopyRaw(Table table, int* ids, int items, byte* dst)
        {
            if (items <= 0)
                return;
            long rowBytes = table.RowBytes;
            Action<int> hook = ReadHookForTests;
            void One(int i)
            {
                hook?.Invoke(2);
                int row = ids[i];
                if ((uint)row >= (uint)table.Rows)
                    throw new InvalidOperationException("[dsv4-cuda] Engram lookup is out of bounds");
                Buffer.MemoryCopy(table.Base + row * rowBytes, dst + i * rowBytes, rowBytes, rowBytes);
            }
            if (Legacy)
            {
                for (int i = 0; i < items; i++)
                    One(i);
                return;
            }
            Run(items, One);
        }

        // ------------------------------------------------------------------ one call's reads, on the reader's threads

        /// <summary>The items of one gather or raw copy. Threads claim items from <see cref="Next"/>; the call returns
        /// when <see cref="Finished"/> reaches <see cref="Items"/>.</summary>
        private sealed class Job
        {
            public readonly Action<int> Body;
            public readonly int Items;
            public int Next;
            public int Finished;
            public volatile bool Failed;
            public List<Exception> Errors;   // under lock (this)
            public bool Complete;            // under lock (this)

            public Job(Action<int> body, int items)
            {
                Body = body;
                Items = items;
            }

            /// <summary>Whether an item is still unclaimed.</summary>
            public bool Open => Volatile.Read(ref Next) < Items;
        }

        // One call at a time per reader (a forward's gathers come from one thread); the threads help whichever is posted.
        private readonly object _callGate = new object();
        private volatile Job _job;

        /// <summary>
        /// Runs <paramref name="body"/> for every item in [0, <paramref name="items"/>) on the reader's threads and the
        /// calling thread, which reads too (so the call starts at once, however busy the threads are), and returns when
        /// every item has run. A failed item fails the call: the remaining ones are skipped, then one
        /// <see cref="AggregateException"/> carries every failure, as <see cref="Parallel.For(int, int, Action{int})"/>'s
        /// did.
        /// </summary>
        private void Run(int items, Action<int> body)
        {
            if (items == 1)
            {
                body(0);
                return;
            }
            lock (_callGate)
            {
                var job = new Job(body, items);
                lock (_lock)
                {
                    // A disposed reader has no threads: the caller does every item itself.
                    if (!_stopping)
                    {
                        _workers ??= StartWorkers();
                        _job = job;
                        Monitor.PulseAll(_lock);
                    }
                }
                Help(job);
                lock (job)
                {
                    while (!job.Complete)
                        Monitor.Wait(job);
                }
                _job = null;
                if (job.Errors != null)
                    throw new AggregateException(job.Errors);
            }
        }

        /// <summary>Claims and runs items of <paramref name="job"/> until none is left unclaimed.</summary>
        private static void Help(Job job)
        {
            while (true)
            {
                int item = Interlocked.Increment(ref job.Next) - 1;
                if (item >= job.Items)
                    return;
                if (!job.Failed)
                {
                    try
                    {
                        job.Body(item);
                    }
                    catch (Exception ex)
                    {
                        lock (job)
                            (job.Errors ??= new List<Exception>()).Add(ex);
                        job.Failed = true;
                    }
                }
                if (Interlocked.Increment(ref job.Finished) == job.Items)
                {
                    lock (job)
                    {
                        job.Complete = true;
                        Monitor.PulseAll(job);
                    }
                }
            }
        }

        // ------------------------------------------------------------------ prefetch

        private readonly object _lock = new object();
        private Thread[] _workers;
        private Batch _batch;
        private bool _stopping;
        private int _reading;   // threads inside a read-ahead batch (under _lock)
        private long _rowsTouched, _batches;

        /// <summary>Rows the prefetch has touched, and the forwards it was handed (tests and logs).</summary>
        public long RowsTouched => Interlocked.Read(ref _rowsTouched);

        public long Batches => Interlocked.Read(ref _batches);

        /// <summary>Tests only: the threads still reading ahead.</summary>
        internal int ReadingAheadThreads
        {
            get
            {
                lock (_lock)
                    return _reading;
            }
        }

        private sealed class Batch
        {
            public Segment[] Segments;
            public long[] Ends;
            public long Total;
            public long Cursor;
            public int Chunk;
            public volatile bool Abandoned;
        }

        /// <summary>
        /// Hands the threads this forward's rows, in the order given, abandoning whatever the last forward left unread.
        /// Returns at once.
        /// </summary>
        public void PrefetchRows(IReadOnlyList<Segment> segments)
        {
            if (!Prefetch || segments == null || segments.Count == 0)
                return;
            var batch = new Batch { Segments = new Segment[segments.Count], Ends = new long[segments.Count] };
            long total = 0;
            for (int i = 0; i < segments.Count; i++)
            {
                batch.Segments[i] = segments[i];
                total += segments[i].Count;
                batch.Ends[i] = total;
            }
            if (total == 0)
                return;
            batch.Total = total;
            batch.Chunk = (int)Math.Clamp(total / (PoolThreads * 4L), 1, 64);
            lock (_lock)
            {
                if (_stopping)
                    return;
                if (_batch != null)
                    _batch.Abandoned = true;
                _batch = batch;
                _workers ??= StartWorkers();
                Monitor.PulseAll(_lock);
            }
            Interlocked.Increment(ref _batches);
        }

        /// <summary>
        /// Stops reading ahead for the last forward: a reset, a forward that failed, or a page-cache eviction, which must
        /// not see the pages it drops read straight back. With <paramref name="wait"/>, returns only once no thread is
        /// still reading that forward's rows (each finishes the row it is on).
        /// </summary>
        public void CancelPrefetch(bool wait = false)
        {
            lock (_lock)
            {
                if (_batch != null)
                    _batch.Abandoned = true;
                _batch = null;
                if (wait)
                {
                    while (_reading > 0)
                        Monitor.Wait(_lock);
                }
            }
        }

        private Thread[] StartWorkers()
        {
            var workers = new Thread[PoolThreads];
            for (int w = 0; w < workers.Length; w++)
            {
                workers[w] = new Thread(Work) { IsBackground = true, Name = $"dsv41-engram-read-{w}" };
                workers[w].Start();
            }
            return workers;
        }

        private static bool HasRows(Batch batch)
            => batch != null && !batch.Abandoned && Interlocked.Read(ref batch.Cursor) < batch.Total;

        private void Work()
        {
            while (true)
            {
                Job job = null;
                Batch batch = null;
                lock (_lock)
                {
                    while (!_stopping && _job is not { Open: true } && !HasRows(_batch))
                        Monitor.Wait(_lock);
                    if (_stopping)
                        return;
                    // Read both again: another thread may have claimed the job's last item since the wait ended.
                    if (_job is { Open: true } open)
                    {
                        job = open;
                    }
                    else if (HasRows(_batch))
                    {
                        batch = _batch;
                        _reading++;
                    }
                }
                if (job != null)
                {
                    Help(job);
                    continue;
                }
                if (batch == null)
                    continue;   // the work that woke this thread is already taken
                try
                {
                    ReadAhead(batch);
                }
                finally
                {
                    lock (_lock)
                    {
                        _reading--;
                        Monitor.PulseAll(_lock);   // a CancelPrefetch(wait: true) may be waiting
                    }
                }
            }
        }

        private void ReadAhead(Batch batch)
        {
            int sink = 0;
            Action<int> hook = ReadHookForTests;
            while (!batch.Abandoned)
            {
                long start = Interlocked.Add(ref batch.Cursor, batch.Chunk) - batch.Chunk;
                if (start >= batch.Total)
                    break;
                long end = Math.Min(batch.Total, start + batch.Chunk);
                // Ends are the segments' cumulative (exclusive) ends: row k lies in the first segment whose end
                // exceeds k.
                int segment = Array.BinarySearch(batch.Ends, start);
                segment = segment >= 0 ? segment + 1 : ~segment;
                long touched = 0;
                for (long k = start; k < end && !batch.Abandoned; k++)
                {
                    // A gather (or a raw copy) is on the forward's critical path: join it first.
                    if (_job is { Open: true } job)
                        Help(job);
                    while (batch.Ends[segment] <= k)
                        segment++;
                    Segment seg = batch.Segments[segment];
                    long first = segment == 0 ? 0 : batch.Ends[segment - 1];
                    int row = seg.Ids[seg.Offset + (int)(k - first)];
                    if ((uint)row >= (uint)seg.Table.Rows)
                        continue;   // the gather reports it
                    hook?.Invoke(0);
                    // A row can straddle a page boundary: touch its first and its last byte.
                    byte* p = seg.Table.Base + row * seg.Table.RowBytes;
                    sink += Volatile.Read(ref *p);
                    sink += Volatile.Read(ref p[seg.Table.RowBytes - 1]);
                    touched++;
                }
                Interlocked.Add(ref _rowsTouched, touched);
            }
            _sink = sink;
        }

        private int _sink;

        /// <summary>Stops the threads and waits for every one to leave, so no thread reads a mapping after its owner
        /// releases it. Call before the tables are unmapped.</summary>
        public void Dispose()
        {
            Thread[] workers;
            lock (_lock)
            {
                _stopping = true;
                if (_batch != null)
                    _batch.Abandoned = true;
                _batch = null;
                workers = _workers;
                Monitor.PulseAll(_lock);
            }
            if (workers == null)
                return;
            foreach (Thread worker in workers)
                worker.Join();
        }
    }
}
