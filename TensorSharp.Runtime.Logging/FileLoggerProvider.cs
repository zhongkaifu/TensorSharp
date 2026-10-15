// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// <see cref="ILoggerProvider"/> that writes each log entry to a local file as a
    /// JSON line (or human-readable text), one file per UTC day, with a soft size
    /// cap that triggers numeric rollover within the day.
    ///
    /// Producers (the <see cref="FileLogger"/> instances) are decoupled from disk
    /// I/O via <see cref="FileLogQueue"/>: they enqueue plain CLR objects and a
    /// dedicated background thread drains the queue, formats records, and writes
    /// to disk. This keeps the request-handling hot path free of blocking I/O even
    /// when log volume spikes.
    /// </summary>
    public sealed class FileLoggerProvider : ILoggerProvider
    {
        private readonly FileLoggerOptions _options;
        private readonly FileLoggerScopeProvider _scopes = new();
        private readonly FileLogQueue _queue;
        private readonly ConcurrentDictionary<string, FileLogger> _loggers = new(StringComparer.Ordinal);
        private readonly Thread _writer;
        private readonly CancellationTokenSource _shutdown = new();
        private readonly object _writeLock = new();
        private readonly object _flushLifecycleLock = new();
        private readonly ConcurrentQueue<TaskCompletionSource<bool>> _flushRequests = new();
        private readonly SemaphoreSlim _flushSlots;
        private int _activeWrites;

        private DateTime _currentDateUtc;
        private int _rolloverSuffix;
        private string? _currentFilePath;
        private long _currentFileBytes;
        private FileStream? _currentFile;
        private bool _disposed;

        public FileLoggerProvider(FileLoggerOptions options)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _options.Validate();

            _queue = new FileLogQueue(_options.MaxQueuedEntries);
            _flushSlots = new SemaphoreSlim(_options.MaxQueuedEntries, _options.MaxQueuedEntries);
            _writer = new Thread(WriterLoop)
            {
                IsBackground = true,
                Name = "TensorSharp.FileLogger",
            };
            _writer.Start();
        }

        /// <summary>Total entries observed by the queue (including dropped).</summary>
        public long EnqueuedCount => _queue.Enqueued;

        /// <summary>Number of entries dropped because the in-memory queue was full.</summary>
        public long DroppedCount => _queue.Dropped;

        public ILogger CreateLogger(string categoryName)
        {
            return _loggers.GetOrAdd(categoryName ?? string.Empty, name =>
                new FileLogger(name, _queue, _scopes, _options.MinimumLevel));
        }

        /// <summary>Wait for the writer to finish pending records and durably flush
        /// their file. Returns false on timeout or after disposal starts. A timeout
        /// does not cancel disk I/O already submitted by the writer.</summary>
        public bool Flush(TimeSpan timeout)
        {
            if (timeout < TimeSpan.Zero && timeout != Timeout.InfiniteTimeSpan)
                throw new ArgumentOutOfRangeException(nameof(timeout));
            if (_disposed)
                return false; // Disposal may still be completing its final durable I/O.

            var timer = Stopwatch.StartNew();
            TimeSpan Remaining() => timeout == Timeout.InfiniteTimeSpan
                ? Timeout.InfiniteTimeSpan : TimeSpan.FromTicks(Math.Max(0, timeout.Ticks - timer.Elapsed.Ticks));
            var completion = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
            if (!_flushSlots.Wait(Remaining()))
                return false;
            lock (_flushLifecycleLock)
            {
                // Disposal cannot retire the writer between this registration
                // and its wakeup, including callers waiting for a bounded slot.
                if (_disposed)
                {
                    _flushSlots.Release();
                    return false;
                }
                _flushRequests.Enqueue(completion);
                _queue.SignalShutdown(); // Wake the writer without a queued log.
            }
            try
            {
                if (!completion.Task.Wait(Remaining()))
                {
                    completion.TrySetCanceled();
                    return false;
                }
            }
            catch (AggregateException)
            {
                // Preserve the original I/O exception rather than changing the
                // public Flush contract to an AggregateException.
                return completion.Task.GetAwaiter().GetResult();
            }
            return completion.Task.GetAwaiter().GetResult();
        }

        public void Dispose()
        {
            lock (_flushLifecycleLock)
            {
                if (_disposed) return;
                _disposed = true;
            }

            try { _shutdown.Cancel(); }
            catch { /* nothing useful we can do here */ }

            _queue.SignalShutdown();
            try { _writer.Join(TimeSpan.FromSeconds(5)); }
            catch { /* shutting down anyway */ }

            DrainQueue();
            CompleteFlushRequests();
            while (_flushRequests.TryDequeue(out var abandoned))
            {
                _flushSlots.Release();
                abandoned.TrySetResult(false);
            }

            lock (_writeLock)
            {
                try { _currentFile?.Flush(true); } catch { }
                try { _currentFile?.Dispose(); } catch { }
                _currentFile = null;
            }

            _shutdown.Dispose();
            _queue.DisposeSignal();
        }

        private void WriterLoop()
        {
            var token = _shutdown.Token;
            while (!token.IsCancellationRequested)
            {
                _queue.WaitForEntry(_options.FlushInterval, token);
                DrainQueue();
                CompleteFlushRequests();
            }

            DrainQueue();
            CompleteFlushRequests();
        }

        private void DrainQueue()
        {
            while (true)
            {
                // Increment before dequeue: an empty queue must not make Flush
                // overlook a record currently being formatted outside _writeLock.
                Interlocked.Increment(ref _activeWrites);
                try
                {
                    if (!_queue.TryDequeue(out var entry))
                        break;
                    if (entry == null) continue;
                    try
                    {
                        WriteEntry(entry);
                    }
                    catch
                    {
                        // Logging must never throw out of an application's hot path.
                        // The next file-rotation pass attempts a fresh handle.
                        try
                        {
                            lock (_writeLock)
                            {
                                try { _currentFile?.Dispose(); } catch { }
                                _currentFile = null;
                                _currentFilePath = null;
                            }
                        }
                        catch { /* give up */ }
                    }
                }
                finally { Interlocked.Decrement(ref _activeWrites); }
            }
        }

        private void CompleteFlushRequests()
        {
            if (_queue.Pending != 0 || Volatile.Read(ref _activeWrites) != 0)
                return;
            while (_queue.Pending == 0 && Volatile.Read(ref _activeWrites) == 0
                && _flushRequests.TryDequeue(out var completion))
            {
                _flushSlots.Release();
                if (completion.Task.IsCompleted) continue;
                try
                {
                    lock (_writeLock) { _currentFile?.Flush(true); }
                    completion.TrySetResult(true);
                }
                catch (Exception error) { completion.TrySetException(error); }
            }
        }

        private void WriteEntry(LogEntry entry)
        {
            string formatted = _options.UseJsonFormat
                ? LogEntryFormatter.ToJsonLine(entry, _options.IncludeScopes)
                : LogEntryFormatter.ToTextLine(entry, _options.IncludeScopes);

            byte[] bytes = Encoding.UTF8.GetBytes(formatted + "\n");
            lock (_writeLock)
            {
                EnsureFile(bytes.Length);
                _currentFile!.Write(bytes, 0, bytes.Length);
                _currentFileBytes += bytes.Length;
                _currentFile.Flush();
            }
        }

        private void EnsureFile(int incomingBytes)
        {
            DateTime today = DateTime.UtcNow.Date;
            bool dateChanged = _currentFile != null && today != _currentDateUtc;
            bool needRotate = _currentFile != null &&
                              _currentFileBytes + incomingBytes > _options.MaxFileSizeBytes;

            if (_currentFile != null && !dateChanged && !needRotate)
                return;

            if (_currentFile != null)
            {
                try { _currentFile.Flush(true); } catch { }
                try { _currentFile.Dispose(); } catch { }
                _currentFile = null;
                _currentFilePath = null;
            }

            if (dateChanged)
            {
                _currentDateUtc = today;
                _rolloverSuffix = 0;
            }
            else if (needRotate)
            {
                _rolloverSuffix++;
            }
            else
            {
                _currentDateUtc = today;
                _rolloverSuffix = 0;
            }

            Directory.CreateDirectory(_options.Directory);
            _currentFilePath = ResolveTargetPath();
            // If we are starting fresh on a new day with index 0 and the file
            // already exists from a previous run, advance the suffix until we find
            // a non-existent file. This protects against multiple processes writing
            // to the same directory.
            while (File.Exists(_currentFilePath) && new FileInfo(_currentFilePath).Length >= _options.MaxFileSizeBytes)
            {
                _rolloverSuffix++;
                _currentFilePath = ResolveTargetPath();
            }

            _currentFile = new FileStream(_currentFilePath, FileMode.Append, FileAccess.Write, FileShare.Read);
            _currentFileBytes = _currentFile.Length;
        }

        private string ResolveTargetPath()
        {
            string suffix = _rolloverSuffix == 0 ? string.Empty : $"-{_rolloverSuffix:D4}";
            string fileName = $"{_options.FilePrefix}-{_currentDateUtc:yyyyMMdd}{suffix}.jsonl";
            return Path.Combine(_options.Directory, fileName);
        }
    }
}
