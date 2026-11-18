using System;
using System.Threading;

namespace TensorSharp.Models
{
    /// <summary>
    /// Persistent spin-then-park worker pool for the pure-C# CPU backend.
    ///
    /// Why not <see cref="System.Threading.Tasks.Parallel"/>: a decoded token
    /// issues a few hundred matmuls, each only ~1 ms of work spread over every
    /// core, so each one is its own fork/join. On the ThreadPool that means
    /// waking parked threads through the kernel and parking them again, and the
    /// wakeup latency for a large pool is the same order as the work itself.
    /// Measured on a 122-core allocation, a 4096x4096 q8_0 matvec went
    /// 2.3 -> 11.7 GB/s from 1 to 16 threads and then FLATLINED at ~13 through
    /// 122, while q4_0 went backwards (4.8 at 16 -> 3.3 at 122). End-to-end
    /// decode showed the same ceiling: 0.5 tok/s at 1 thread, 2.3 by 8, and no
    /// further gain from the remaining 114 cores.
    ///
    /// This pool keeps its workers alive across jobs and hands them work through
    /// a generation counter they spin on, so submitting costs an interlocked
    /// write instead of a kernel round trip. Blocks are claimed with an atomic
    /// counter, which also balances the tail without extra tasks. Workers spin
    /// for a bounded interval and only then park on a monitor, so an idle
    /// process does not burn cores, and the park happens under the same lock the
    /// submitter pulses so a wakeup can never be missed.
    /// </summary>
    internal sealed class CpuWorkerPool : IDisposable
    {
        private static readonly Lazy<CpuWorkerPool> SharedPool = new(() => new CpuWorkerPool(DefaultThreadCount()));
        public static CpuWorkerPool Shared => SharedPool.Value;

        // Waking a parked worker is expensive at this width, whatever the
        // primitive, so the design spins long enough that the steady state never
        // parks at all. Measured on a 122-core allocation, decode tok/s:
        //
        //     spin 4096, monitor park : 7.0   <- this
        //     spin 256,  monitor park : 0.1
        //     spin 512,  event park   : 0.3
        //
        // Parking early loses because a job has to wake every worker, and 121
        // kernel wakeups cost far more than the ~60 us of work being handed out.
        // Spinning is not free either: spinners burn CPU quota that any work still
        // on the ThreadPool needs. When this was measured, attention and the Core
        // elementwise ops ran there; they have since moved onto this pool too (the
        // Core kernels through CpuParallelBinding, the pure-C# model kernels
        // through CpuWorkers), and what remains on the ThreadPool is mostly model
        // glue and the pre-existing per-model loops. ThreadCount is still why the
        // pool deliberately does NOT take every core (see DefaultThreadCountFor).
        private static readonly int SpinsBeforePark = EnvInt("TS_CPU_SPIN", 4096);

        /// <summary>
        /// Stack reserved for each worker. Smaller than a ThreadPool thread's, and running out of
        /// stack aborts the process, so a kernel never sizes a stackalloc by its input here: a
        /// buffer that grows with the context, the heads or the batch goes through
        /// <see cref="CpuScratch"/>, which keeps it on the stack only up to a quarter of this.
        /// </summary>
        internal const int WorkerStackBytes = 512 * 1024;

        private static int EnvInt(string name, int fallback)
            => int.TryParse(Environment.GetEnvironmentVariable(name), out int v) && v >= 0
                ? v
                : fallback;

        // Set on ANY thread that is inside a job - the submitter included, since
        // Monitor is recursive and a nested For() from the submitting thread would
        // otherwise re-enter _submitLock and clobber the job in flight.
        [ThreadStatic] private static bool _inJob;

        private readonly object _submitLock = new object();
        private readonly object _sleepLock = new object();
        private readonly int _workers;
        private readonly Thread[] _threads;

        // The job in flight, published under a sequence lock (_jobSeq odd while the
        // submitter rewrites it): its body and its range [_jobBase, _jobEnd) of the
        // claim counter. _next (the next unclaimed block) and _done (blocks finished)
        // only ever grow, across all jobs: a job's range starts where the previous
        // job's ended. So a block index belongs to exactly one job, a worker that read
        // a job's fields can claim (CAS) only an index of that job while the job is
        // still open, and a worker that is late - descheduled, or still parked - holds
        // nothing up: the submitter waits for the blocks that were CLAIMED, not for
        // every worker to look in. (Waiting for every worker made a 12-18 thread pool
        // collapse whenever another process held one core: each of a step's few
        // hundred jobs waited out that worker's time slice.)
        private Action<int> _body;
        private long _jobBase;
        private long _jobEnd;
        private long _jobSeq;
        private long _next;
        private long _done;
        private int _parked;
        private long _generation;
        private Exception _error;
        private volatile bool _shutdown;

        internal CpuWorkerPool(int totalThreads)
        {
            if (totalThreads < 1 || totalThreads > 512) throw new ArgumentOutOfRangeException(nameof(totalThreads));
            _workers = Math.Max(0, totalThreads - 1);
            _threads = new Thread[_workers];
            for (int i = 0; i < _workers; i++)
            {
                var t = new Thread(WorkerLoop, WorkerStackBytes)
                {
                    IsBackground = true,
                    Name = "ts-cpu-" + i,
                };
                _threads[i] = t;
                try { t.Start(); }
                catch
                {
                    _shutdown = true;
                    lock (_sleepLock) Monitor.PulseAll(_sleepLock);
                    for (int started = 0; started < i; ++started) _threads[started].Join();
                    throw;
                }
            }
            // NB: the workers are NOT waited for here. WorkerLoop touches this
            // type's statics, so a thread that reaches them while the type
            // initializer is still running blocks on it - and the initializer is
            // what runs this constructor. Waiting here deadlocked every worker
            // against the cctor. Nothing waits for them later either: a job
            // completes when its claimed blocks do, so a worker that starts late
            // joins whatever job is in flight then.
        }

        private static int DefaultThreadCount()
        {
            string s = Environment.GetEnvironmentVariable("TS_CPU_THREADS");
            if (int.TryParse(s, out int n) && n > 0) return n;
            return DefaultThreadCountFor(Environment.ProcessorCount);
        }

        /// <summary>
        /// How wide the pool should be. NOT every core. The rationale dates from
        /// when this pool ran only the quantized matmuls and the rest of the CPU
        /// path used the ThreadPool: workers spin between jobs, and taking every
        /// core made the spinners starve that other work and regressed prefill.
        /// It also bought nothing - past about a quarter of a 122-core machine the
        /// matmuls stop scaling, so the extra threads only add spin. The Core
        /// kernels (GEMM, elementwise, norms) and the pure-C# model kernels now
        /// run here as well, which the default has not been re-tuned for: above 8
        /// logical processors it is half of them (one per core on an SMT part such
        /// as the 8-core/16-thread i7-11800H). TS_CPU_THREADS overrides it.
        /// </summary>
        internal static int DefaultThreadCountFor(int processors)
            => processors <= 8 ? Math.Max(1, processors) : Math.Max(8, processors / 2);

        /// <summary>Threads that will run a job, including the submitting one.</summary>
        public int ThreadCount => _workers + 1;

        /// <summary>
        /// Run <paramref name="body"/> for every block index in [0, blockCount)
        /// across this thread and the pool's, returning once all have completed.
        /// </summary>
        public void For(int blockCount, Action<int> body)
        {
            ObjectDisposedException.ThrowIf(_shutdown, this);
            if (blockCount <= 0) return;

            // Nested calls, and a job already in flight, both run inline: this is
            // a flat one-job-at-a-time pool, and recursing into it would either
            // deadlock or oversubscribe.
            if (blockCount == 1 || _workers == 0 || _inJob || !Monitor.TryEnter(_submitLock))
            {
                bool wasInJob = _inJob;
                _inJob = true;
                try { for (int i = 0; i < blockCount; i++) body(i); }
                finally { _inJob = wasInJob; }
                return;
            }

            _inJob = true;
            try
            {
                ObjectDisposedException.ThrowIf(_shutdown, this);
                // Every earlier job is complete, so _next == _done == the new job's base.
                long start = Volatile.Read(ref _next);
                long end = start + blockCount;
                _error = null;
                Interlocked.Increment(ref _jobSeq);           // odd: fields being written
                _body = body;
                _jobBase = start;
                _jobEnd = end;
                Interlocked.Increment(ref _jobSeq);           // even: published (full fence)

                // Full fence: workers that observe the new generation see the job.
                Interlocked.Increment(ref _generation);

                // Only pay for a wakeup when someone is actually parked. The
                // increment above is a full fence and a worker publishes _parked
                // before re-checking the generation, so a worker this misses is
                // guaranteed to see the new generation and not wait.
                if (Volatile.Read(ref _parked) > 0)
                {
                    lock (_sleepLock) Monitor.PulseAll(_sleepLock);
                }

                RunBlocks(body, start, end);

                var spin = new SpinWait();
                while (Volatile.Read(ref _done) != end) spin.SpinOnce(-1);

                // Drop the delegate (and what it captured) until the next job.
                Interlocked.Increment(ref _jobSeq);
                _body = null;
                Interlocked.Increment(ref _jobSeq);

                Exception error = _error;
                if (error != null)
                {
                    _error = null;
                    throw new AggregateException(error);
                }
            }
            finally
            {
                _inJob = false;
                Monitor.Exit(_submitLock);
            }
        }

        /// <summary>Cancellation is observed between blocks and propagated with
        /// the original token after all active workers have returned.</summary>
        public void For(int blockCount, Action<int> body, CancellationToken cancellationToken)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (!cancellationToken.CanBeCanceled) { For(blockCount, body); return; }
            try
            {
                For(blockCount, block => { cancellationToken.ThrowIfCancellationRequested(); body(block); });
            }
            catch (AggregateException error) when (cancellationToken.IsCancellationRequested &&
                error.InnerExceptions.Count == 1 && error.InnerExceptions[0] is OperationCanceledException canceled &&
                canceled.CancellationToken == cancellationToken)
            {
                throw new OperationCanceledException(cancellationToken);
            }
            cancellationToken.ThrowIfCancellationRequested();
        }

        public void Dispose()
        {
            if (_inJob) throw new InvalidOperationException("Cannot dispose a CPU worker pool from a running pool job.");
            lock (_submitLock)
            {
                if (_shutdown) return;
                _shutdown = true;
                lock (_sleepLock) Monitor.PulseAll(_sleepLock);
                foreach (var thread in _threads) thread.Join();
            }
        }

        /// <summary>Claim and run blocks of the job [start, end) until none is left. A block index is claimed by a
        /// compare-exchange that succeeds only while the index is still the next unclaimed one, so it can never take a
        /// block of a later job.</summary>
        private void RunBlocks(Action<int> body, long start, long end)
        {
            try
            {
                while (true)
                {
                    long i = Volatile.Read(ref _next);
                    if (i < start || i >= end) return;
                    if (Interlocked.CompareExchange(ref _next, i + 1, i) != i) continue;
                    try { body((int)(i - start)); }
                    finally { Interlocked.Increment(ref _done); }
                }
            }
            catch (Exception ex)
            {
                // Claim the rest of the range so the other threads wind down (and count
                // it done), and hand the first failure back to the submitter.
                Interlocked.CompareExchange(ref _error, ex, null);
                while (true)
                {
                    long i = Volatile.Read(ref _next);
                    if (i < start || i >= end) return;
                    if (Interlocked.CompareExchange(ref _next, end, i) == i)
                    {
                        Interlocked.Add(ref _done, end - i);
                        return;
                    }
                }
            }
        }

        /// <summary>The job in flight, read consistently (retried while the submitter rewrites it).</summary>
        private bool TryReadJob(out Action<int> body, out long start, out long end)
        {
            var spin = new SpinWait();
            while (true)
            {
                long seq = Volatile.Read(ref _jobSeq);
                if ((seq & 1) == 0)
                {
                    body = Volatile.Read(ref _body);
                    start = Volatile.Read(ref _jobBase);
                    end = Volatile.Read(ref _jobEnd);
                    if (Volatile.Read(ref _jobSeq) == seq)
                        return body != null;
                }
                if (_shutdown)
                {
                    body = null; start = end = 0;
                    return false;
                }
                spin.SpinOnce(-1);
            }
        }

        private void WorkerLoop()
        {
            // No generation seen yet: a worker that starts while a job is in flight joins it.
            long seen = -1;
            _inJob = true;
            int spins = 0;

            while (!_shutdown)
            {
                long gen = Volatile.Read(ref _generation);
                if (gen != seen)
                {
                    seen = gen;
                    if (TryReadJob(out Action<int> body, out long start, out long end))
                        RunBlocks(body, start, end);
                    spins = 0;
                    continue;
                }

                if (++spins < SpinsBeforePark)
                {
                    Thread.SpinWait(64);
                    continue;
                }

                // Park under the lock the submitter pulses, so check-then-wait
                // cannot race with a job being published.
                Interlocked.Increment(ref _parked);
                lock (_sleepLock)
                {
                    if (Volatile.Read(ref _generation) == seen && !_shutdown)
                        Monitor.Wait(_sleepLock);
                }
                Interlocked.Decrement(ref _parked);
                spins = 0;
            }
        }
    }
}
