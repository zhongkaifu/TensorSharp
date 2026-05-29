using System;
using System.Collections.Concurrent;
using System.Threading;

namespace TensorSharp.MLX
{
    public sealed class MlxWorker : IDisposable
    {
        private readonly BlockingCollection<IWorkItem> queue = new();
        private readonly Thread thread;
        private int workerThreadId;
        private int disposed;

        // Diagnostic counters: every Invoke/Dispatch increments
        // _dispatchCount; the queue-hopping (cross-thread) subset increments
        // _queueCount. The gap between the two indicates how often
        // re-entrant inline execution short-circuits the queue — i.e., how
        // effective an outer Invoke wrapper is at batching its sub-calls.
        private static long _dispatchCount;
        private static long _queueCount;
        public static long DispatchCount => Volatile.Read(ref _dispatchCount);
        public static long QueueCount => Volatile.Read(ref _queueCount);
        public static void ResetDispatchCount()
        {
            Volatile.Write(ref _dispatchCount, 0);
            Volatile.Write(ref _queueCount, 0);
        }

        public static MlxWorker Shared { get; } = new MlxWorker();

        private MlxWorker()
        {
            thread = new Thread(Run)
            {
                IsBackground = true,
                Name = "TensorSharp MLX worker"
            };
            thread.Start();
        }

        // Returns true when called from inside an Invoke/Dispatch on the worker
        // thread itself — i.e. from a re-entrant context like an mlx_compile
        // trace callback. In that case, nesting another Invoke would deadlock
        // (worker thread is busy running us). Callers can detect this and run
        // the work inline.
        public bool IsOnWorkerThread => Thread.CurrentThread.ManagedThreadId == Volatile.Read(ref workerThreadId);

        public T Invoke<T>(Func<T> func)
        {
            if (func == null)
                throw new ArgumentNullException(nameof(func));
            ThrowIfDisposed();
            Interlocked.Increment(ref _dispatchCount);

            // Re-entrant: we're already on the worker. Run inline, otherwise
            // we'd block waiting for ourselves.
            if (IsOnWorkerThread)
                return func();

            Interlocked.Increment(ref _queueCount);
            var item = new WorkItem<T>(func);
            queue.Add(item);
            return item.GetResult();
        }

        public void Invoke(Action action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            Invoke(() =>
            {
                action();
                return 0;
            });
        }

        // Fire-and-forget: enqueue work without waiting for completion or a
        // result. The worker is FIFO so ordering is preserved against any later
        // Invoke calls. Exceptions thrown by `action` are swallowed — only use
        // this for side-effect-only ops that never raise a meaningful error
        // (e.g. mlx_array_free, mlx_async_eval). Skipping the signal/wait round
        // trip is worth ~1-2 microseconds per call, which adds up over the
        // 10^5-10^6 MLX ops issued per benchmark run.
        public void Dispatch(Action action)
        {
            if (action == null)
                throw new ArgumentNullException(nameof(action));
            ThrowIfDisposed();
            Interlocked.Increment(ref _dispatchCount);

            // Re-entrant: run inline to preserve ordering with the synchronous
            // ops surrounding us on the worker thread.
            if (IsOnWorkerThread)
            {
                try { action(); }
                catch { /* fire-and-forget swallows errors */ }
                return;
            }

            Interlocked.Increment(ref _queueCount);
            queue.Add(new FireAndForgetItem(action));
        }

        private void Run()
        {
            Volatile.Write(ref workerThreadId, Thread.CurrentThread.ManagedThreadId);
            foreach (IWorkItem item in queue.GetConsumingEnumerable())
                item.Execute();
        }

        private void ThrowIfDisposed()
        {
            if (Volatile.Read(ref disposed) != 0)
                throw new ObjectDisposedException(nameof(MlxWorker));
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref disposed, 1) != 0)
                return;

            queue.CompleteAdding();
        }

        private interface IWorkItem
        {
            void Execute();
        }

        private sealed class WorkItem<T> : IWorkItem
        {
            private readonly Func<T> func;
            private readonly ManualResetEventSlim completed = new(false);
            private T result;
            private Exception exception;

            public WorkItem(Func<T> func)
            {
                this.func = func;
            }

            public void Execute()
            {
                try
                {
                    result = func();
                }
                catch (Exception ex)
                {
                    exception = ex;
                }
                finally
                {
                    completed.Set();
                }
            }

            public T GetResult()
            {
                completed.Wait();
                if (exception != null)
                    throw exception;
                return result;
            }
        }

        private sealed class FireAndForgetItem : IWorkItem
        {
            private readonly Action action;

            public FireAndForgetItem(Action action)
            {
                this.action = action;
            }

            public void Execute()
            {
                try
                {
                    action();
                }
                catch
                {
                    // Errors from fire-and-forget ops are intentionally
                    // swallowed; callers must not rely on this path for
                    // anything that can raise.
                }
            }
        }
    }
}
