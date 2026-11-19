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
    internal sealed partial class CpuWorkerPool : IDisposable
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
        /// Completion polls a submitter takes, a pause instruction apart, before <see cref="SpinWait"/>'s yields: after its
        /// own blocks the submitter waits for the other threads' last ones, and a job of a decode step lasts tens of
        /// microseconds. SpinWait yields the thread from its eleventh round on (a <c>swtch_pri</c> system call on macOS),
        /// which on its own measured 30-40 % of a decoding submitter's samples, and returns well after the job is done; 2,000
        /// polls is some 0.1 ms, past which a long job's wait yields as before. LFM3-1.7B Q4_K_M on an M5 Pro, 16-thread
        /// decode, tg128 at depth 0 / 1,024: 138.9 / 128.2 tok/s yielding at once, 142.5 / 130.6 polling (4 interleaved rounds).
        /// </summary>
        private const int JoinPolls = 2000;

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

        // The workers that take a job's blocks: worker i does while i < _activeWorkers (the submitter always does). A
        // worker past them that sees a job parks on _narrowLock until the width grows (SetActiveThreads), so a narrowed
        // pool leaves its cores to other threads - a server's - instead of spinning on them.
        private int _activeWorkers;
        private readonly object _narrowLock = new object();

        internal CpuWorkerPool(int totalThreads)
        {
            if (totalThreads < 1 || totalThreads > 512) throw new ArgumentOutOfRangeException(nameof(totalThreads));
            _workers = Math.Max(0, totalThreads - 1);
            _activeWorkers = _workers;
            _threads = new Thread[_workers];
            for (int i = 0; i < _workers; i++)
            {
                int index = i;
                var t = new Thread(() => WorkerLoop(index), WorkerStackBytes)
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
                    lock (_narrowLock) Monitor.PulseAll(_narrowLock);
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
            return DefaultThreadCountFor(Environment.ProcessorCount, AppleSilicon);
        }

        /// <summary>Apple silicon: no SMT, every logical processor is a core of its own.</summary>
        private static bool AppleSilicon =>
            (OperatingSystem.IsMacOS() || OperatingSystem.IsIOS() || OperatingSystem.IsMacCatalyst())
            && System.Runtime.InteropServices.RuntimeInformation.ProcessArchitecture == System.Runtime.InteropServices.Architecture.Arm64;

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
        ///
        /// Apple silicon takes every core (<paramref name="oneThreadPerCore"/>): it has no SMT, and on an M5 Pro (6 + 12
        /// cores) LFM3-1.7B Q4_K_M ran 596 / 526 / 426 tok/s prompts and 116 / 114 / 97 tok/s decode at 18 / 12 / 9
        /// threads (once a job stopped waiting for every worker to look in: a descheduled worker no longer stalls it).
        /// </summary>
        internal static int DefaultThreadCountFor(int processors, bool oneThreadPerCore = false)
            => processors <= 8 || oneThreadPerCore ? Math.Max(1, processors) : Math.Max(8, processors / 2);

        /// <summary>Rows of a step's weight products up to which the step is decode-sized (one sequence's token, or a few
        /// sequences' tokens in one step), as ggml_cpu splits its prompt and decode pools.</summary>
        internal const int DecodeRows = 4;

        private static readonly Lazy<int> DecodeThreadsValue = new(() =>
        {
            int pool = Shared.ThreadCount;
            int.TryParse(Environment.GetEnvironmentVariable("TS_CPU_DECODE_THREADS"), out int forced);
            bool poolForced = int.TryParse(Environment.GetEnvironmentVariable("TS_CPU_THREADS"), out int n) && n > 0;
            (int perf, int eff) = HybridMacCores();
            return DecodeThreadCountFor(pool, forced, poolForced, perf, eff);
        });

        /// <summary>Width of the shared pool on decode-sized steps (<see cref="DecodeThreadCountFor"/>).</summary>
        internal static int DecodeThreads => DecodeThreadsValue.Value;

        /// <summary>
        /// The shared pool's width on decode-sized steps: <paramref name="forced"/> (TS_CPU_DECODE_THREADS) when set, else the
        /// pool's width when TS_CPU_THREADS sets it (as <c>llama-bench -t</c> sets both counts), else on a Mac with
        /// performance and efficiency cores (<c>hw.perflevel0</c> / <c>hw.perflevel1</c>) every core but two, never fewer
        /// than the performance cores, else the pool's width. A decode step is a few hundred small jobs whose last block
        /// ends each one, so a pool worker on every core loses a core to every other runnable thread of the process at
        /// every join - a server's request, streaming and scheduling threads.
        /// </summary>
        internal static int DecodeThreadCountFor(int poolThreads, int forced, bool poolForced, int perfCores, int effCores)
        {
            if (forced > 0) return Math.Min(forced, poolThreads);
            if (poolForced) return poolThreads;
            if (perfCores > 0 && effCores > 0) return Math.Clamp(poolThreads - 2, Math.Min(perfCores, poolThreads), poolThreads);
            return poolThreads;
        }

        /// <summary>Performance and efficiency core counts of a Mac (macOS or Mac Catalyst), (0, 0) elsewhere or when
        /// the system does not report them.</summary>
        private static (int Perf, int Eff) HybridMacCores()
        {
            if (!(OperatingSystem.IsMacOS() || OperatingSystem.IsMacCatalyst())) return (0, 0);
            try { return (SysctlInt("hw.perflevel0.physicalcpu"), SysctlInt("hw.perflevel1.physicalcpu")); }
            catch (Exception ex) when (ex is DllNotFoundException or EntryPointNotFoundException) { return (0, 0); }
        }

        private static unsafe int SysctlInt(string name)
        {
            int value = 0;
            nuint length = sizeof(int);
            return sysctlbyname(name, &value, &length, null, 0) == 0 ? value : 0;
        }

        [System.Runtime.InteropServices.LibraryImport("libc", EntryPoint = "sysctlbyname",
            StringMarshalling = System.Runtime.InteropServices.StringMarshalling.Utf8)]
        private static unsafe partial int sysctlbyname(string name, void* value, nuint* length, void* newValue, nuint newLength);

        /// <summary>
        /// Run the shared pool at its decode width (<see cref="DecodeThreads"/>) or its full width from the next job on: an
        /// executor calls it once a step, at the step's first weight product (<paramref name="productRows"/> rows, decode-sized
        /// up to <see cref="DecodeRows"/>). Nothing under TS_CPU_POOL=0 (no pool to narrow).
        /// </summary>
        internal static void SelectWidthForStep(long productRows)
        {
            if (!CpuWorkers.PoolEnabled) return;
            CpuWorkerPool pool = Shared;
            pool.SetActiveThreads(productRows <= DecodeRows ? DecodeThreads : pool.ThreadCount);
        }

        /// <summary>Threads the pool has, the submitting one included: what kernels size their task splits from. A
        /// narrowed pool (<see cref="ActiveThreads"/>) keeps this figure. The ARM64 products' shrinking column blocks
        /// (ManagedQuantizedOps.GuidedBounds) follow <see cref="ActiveThreads"/> instead; they move only which thread
        /// computes a column, never how, so no result depends on the width a step runs at.</summary>
        public int ThreadCount => _workers + 1;

        /// <summary>Threads that run a job's blocks now, the submitting one included (<see cref="SetActiveThreads"/>).</summary>
        public int ActiveThreads => Volatile.Read(ref _activeWorkers) + 1;

        /// <summary>
        /// Run the following jobs on <paramref name="threads"/> threads (the submitting one included; clamped to
        /// [1, <see cref="ThreadCount"/>]). The workers past the width park as soon as they see a job and stay parked until
        /// a wider setting wakes them, so their cores are free for other threads; a worker that is mid-block finishes it.
        /// The pool runs the blocks it is given at any width - only which threads claim them changes; a kernel that sizes
        /// its blocks from <see cref="ActiveThreads"/> keeps each output's arithmetic independent of its block. Meant
        /// to change at step boundaries (a decode step narrower than a prompt step), not per job: widening wakes parked
        /// threads through the kernel.
        /// </summary>
        public void SetActiveThreads(int threads)
        {
            int workers = Math.Clamp(threads, 1, _workers + 1) - 1;
            int previous = Interlocked.Exchange(ref _activeWorkers, workers);
            if (workers > previous)
            {
                lock (_narrowLock) Monitor.PulseAll(_narrowLock);
            }
        }

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
            if (blockCount == 1 || Volatile.Read(ref _activeWorkers) == 0 || _inJob || !Monitor.TryEnter(_submitLock))
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

                for (int poll = 0; poll < JoinPolls && Volatile.Read(ref _done) != end; poll++) Thread.SpinWait(1);
                var spin = new SpinWait();
                while (Volatile.Read(ref _done) != end) spin.SpinOnce(-1);

                // Drop the delegate (and what it captured) until the next job.
                Interlocked.Increment(ref _jobSeq);
                _body = null;
                Interlocked.Increment(ref _jobSeq);

                // A block's failure is recorded before the block is counted (Fail), so once _done == end every failure
                // of this job is visible here, and none can land after this job returns.
                Exception error = Volatile.Read(ref _error);
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
                lock (_narrowLock) Monitor.PulseAll(_narrowLock);
                foreach (var thread in _threads) thread.Join();
            }
        }

        /// <summary>Claim and run blocks of the job [start, end) until none is left. A block index is claimed by a
        /// compare-exchange that succeeds only while the index is still the next unclaimed one, so it can never take a
        /// block of a later job.</summary>
        private void RunBlocks(Action<int> body, long start, long end)
        {
            while (true)
            {
                long i = Volatile.Read(ref _next);
                if (i < start || i >= end) return;
                if (Interlocked.CompareExchange(ref _next, i + 1, i) != i) continue;
                try
                {
                    body((int)(i - start));
                }
                catch (Exception ex)
                {
                    Fail(ex, start, end);
                    return;
                }
                Interlocked.Increment(ref _done);
            }
        }

        /// <summary>
        /// A block of the job [start, end) threw: record the first failure, claim the rest of the range so the other
        /// threads wind down, and only then count the failed block (and the claimed rest) done. The order is the point:
        /// the submitter returns once every block is counted, so a failure recorded after its block was counted (by a
        /// <c>finally</c> that counted it, which runs before the <c>catch</c> that recorded it) was lost when that block
        /// finished last, or surfaced in the NEXT job after the submitter had cleared <see cref="_error"/> for it.
        /// </summary>
        private void Fail(Exception ex, long start, long end)
        {
            Interlocked.CompareExchange(ref _error, ex, null);   // a full fence: visible before _done moves
            long counted = 1;                                   // the block that threw
            while (true)
            {
                long i = Volatile.Read(ref _next);
                if (i < start || i >= end) break;
                if (Interlocked.CompareExchange(ref _next, end, i) == i)
                {
                    counted += end - i;
                    break;
                }
            }
            Interlocked.Add(ref _done, counted);
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

        private void WorkerLoop(int index)
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
                    // Past the active width (SetActiveThreads): park until the width grows. The job is read FIRST and the
                    // width after it: the submitter writes the width before it publishes a job (an interlocked write, then
                    // the job's sequence increments, full fences), so a worker that read a job published after a narrowing
                    // reads the narrowing too and leaves that job's blocks to the others. (Read before the job, the width
                    // could predate a narrowing whose job was published after the generation this worker saw: one job's
                    // blocks then went to a parked-to-be worker.)
                    bool haveJob = TryReadJob(out Action<int> body, out long start, out long end);
                    if (index >= Volatile.Read(ref _activeWorkers))
                    {
                        body = null;   // a parked worker holds no job's delegate (or what it captured)
                        lock (_narrowLock)
                        {
                            while (index >= Volatile.Read(ref _activeWorkers) && !_shutdown)
                                Monitor.Wait(_narrowLock);
                        }
                        seen = -1;   // join whatever job is in flight when it wakes
                        spins = 0;
                        continue;
                    }
                    if (haveJob)
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
