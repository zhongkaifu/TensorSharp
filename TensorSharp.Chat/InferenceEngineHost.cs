// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Server
{
    /// <summary>
    /// Owner of the per-model <see cref="InferenceEngine"/>. Lifecycle-bound to
    /// <see cref="ModelLifecycleService"/>: the engine is constructed lazily on
    /// first access (after a model has been loaded) and rebuilt whenever the
    /// model object or its KV-state fingerprint changes (i.e. on model swap,
    /// including a reload of a checkpoint with the same fingerprint). Disposing
    /// this service tears down the engine, which joins its worker thread and
    /// frees the paged KV block pool.
    ///
    /// This service is the public substitute for the legacy
    /// <see cref="InferenceQueue"/>: submission is non-blocking, multiple
    /// requests run concurrently (with iteration-level fairness), and the
    /// paged KV pool / continuous-batching scheduler / per-block prefix cache
    /// all live behind this single entry point. Adapters that haven't yet
    /// dropped queue-status chunks still take <see cref="InferenceQueue"/>
    /// tickets, but those tickets grant immediately so the engine remains the
    /// only real concurrency boundary.
    /// </summary>
    public sealed class InferenceEngineHost : IDisposable
    {
        private readonly ModelLifecycleService _lifecycle;
        private readonly ILogger _logger;
        private readonly object _gate = new();
        private InferenceEngine _engine;
        private string _fingerprint;
        // The model object the standing engine was built on. The fingerprint alone is
        // not an identity: it names a cache SHAPE, so two loads of the same checkpoint
        // (or two checkpoints that share a geometry) report the same string, and an
        // engine built on the first would keep driving a model that has since been
        // disposed and replaced (DEC-37).
        private object _engineModel;
        private bool _disposed;

        /// <summary>
        /// Engine sizing supplied by the host; null means
        /// <see cref="SchedulerConfig.FromEnvironment"/>. Read when the engine is
        /// (re)built, which the log line below records so an operator can tell which
        /// source sized the KV pool.
        /// </summary>
        public SchedulerConfig SchedulerConfigOverride { get; set; }

        private ComputeGate _computeGate;

        /// <summary>
        /// The gate every engine this host builds runs behind, or null for none.
        ///
        /// <para>
        /// Set once by a host whose platform can take the GPU away — the iOS app, while
        /// it is not frontmost — and handed to each engine as it is built, AND to the
        /// one already standing, because the engine is rebuilt on every model swap and
        /// a gate that only reached the first one would silently stop working the
        /// first time the user changed models. See <see cref="ComputeGate"/>.
        /// </para>
        /// </summary>
        public ComputeGate ComputeGate
        {
            get { lock (_gate) return _computeGate; }
            set
            {
                lock (_gate)
                {
                    _computeGate = value;
                    if (_engine != null)
                        _engine.ComputeGate = value;
                }
            }
        }

        /// <summary>
        /// Where the engine keeps shared-prefix checkpoints between processes, or null
        /// for nowhere. Handed to every engine this host builds and to the one standing,
        /// for the same reason as <see cref="ComputeGate"/>: the engine is rebuilt on
        /// every model swap, and the host sets this per model.
        /// </summary>
        public IPrefixCheckpointStore PrefixCheckpointStore
        {
            get { lock (_gate) return _checkpointStore; }
            set
            {
                lock (_gate)
                {
                    _checkpointStore = value;
                    if (_engine != null)
                        _engine.PrefixCheckpointStore = value;
                }
            }
        }

        private IPrefixCheckpointStore _checkpointStore;

        internal InferenceEngineHost(ModelLifecycleService lifecycle, ILogger logger)
        {
            _lifecycle = lifecycle ?? throw new ArgumentNullException(nameof(lifecycle));
            _logger = logger ?? NullLogger.Instance;
        }

        /// <summary>Get the engine for the currently-loaded model, constructing
        /// it if it hasn't been built yet (or rebuilding it if the model has
        /// changed). Returns null when no model is loaded or when the model
        /// supports neither the KV-state snapshot contract nor the batched
        /// paged-attention contract.
        ///
        /// Models that implement <see cref="IBatchedPagedModel"/> serve
        /// parallel requests via <c>ForwardBatch</c> and don't need to swap
        /// KV state between sequences, so they qualify even when
        /// <see cref="ModelBase.SupportsKVStateSnapshot"/> reports false.</summary>
        public InferenceEngine TryGetEngine()
        {
            var model = _lifecycle.Model;
            if (model == null) return null;
            if (!model.SupportsKVStateSnapshot && model is not IBatchedPagedModel) return null;

            string fp = model.KVStateFingerprint ?? string.Empty;
            lock (_gate)
            {
                if (_disposed) return null;
                if (_engine != null
                    && ReferenceEquals(_engineModel, model)
                    && string.Equals(_fingerprint, fp, StringComparison.Ordinal))
                    return _engine;

                _engine?.Dispose();
                SchedulerConfig cfg = SchedulerConfigOverride;
                string cfgSource = cfg != null ? "host" : "environment";
                cfg ??= SchedulerConfig.FromEnvironment();
                _engine = new InferenceEngine(model, cfg, _logger)
                {
                    ComputeGate = _computeGate,
                    PrefixCheckpointStore = _checkpointStore,
                };
                _fingerprint = fp;
                _engineModel = model;
                // The most recent switch, in case it was written after this engine's
                // configuration was read from the environment.
                if (_pendingSpeculation is { } pending)
                    _engine.UpdateSpeculation(pending);
                var poolStats = _engine.PoolStats;
                _logger.LogInformation(
                    "InferenceEngine constructed for fingerprint {Fingerprint} (blocks={NumBlocks}, blockSize={BlockSize}, kvCapacityTokens={KvCapacity}, maxBatched={MaxBatched}, config={ConfigSource})",
                    fp, poolStats.totalBlocks, poolStats.blockSize,
                    (long)poolStats.totalBlocks * poolStats.blockSize, cfg.MaxNumBatchedTokens, cfgSource);
                return _engine;
            }
        }

        /// <summary>Peek at the live concurrency counters of the already-built
        /// engine without constructing one. Returns false when no engine exists
        /// yet (no model loaded, the model can't use the engine, or no request
        /// has been submitted yet) - in that case the out values are zero.
        ///
        /// This is deliberately side-effect free: a status poll must never
        /// trigger lazy engine construction (and the matching block-pool
        /// allocation) the way <see cref="TryGetEngine"/> does.</summary>
        public bool TryGetLiveStats(out int processing, out int waiting, out long totalCompleted)
        {
            processing = 0;
            waiting = 0;
            totalCompleted = 0;
            lock (_gate)
            {
                if (_disposed || _engine == null)
                    return false;
                processing = _engine.RunningCount;
                waiting = _engine.WaitingCount;
                totalCompleted = _engine.TotalCompleted;
                return true;
            }
        }

        /// <summary>
        /// Ask the standing engine, if there is one, to release what only speeds up the
        /// next request. Never constructs an engine: a memory warning must not be what
        /// allocates a block pool. Returns false when nothing was there to ask.
        /// </summary>
        public bool TrimIdleMemory()
        {
            // Never WAITS for the gate. Reset and Dispose hold it while they join the
            // engine's worker thread for the rest of an in-flight step, which can be tens
            // of seconds -- and a memory warning arrives on the UI thread. A load or
            // unload in progress is also exactly the moment there is nothing sensible to
            // trim: the engine is being torn down or has not been built yet.
            if (!System.Threading.Monitor.TryEnter(_gate, 0))
                return false;
            try
            {
                if (_disposed || _engine == null)
                    return false;
                _engine.TrimIdleMemory();
                return true;
            }
            finally
            {
                System.Threading.Monitor.Exit(_gate);
            }
        }

        /// <summary>Drop the engine (if any). Called by <see cref="ModelLifecycleService"/>
        /// when the model is unloaded so we don't hold onto a stale block pool.</summary>
        /// <summary>
        /// Hand a new speculation policy to the standing engine, if there is one. An
        /// engine built later reads the policy from its configuration (the environment,
        /// which the host keeps in step), so nothing is remembered here. Never waits for
        /// the gate: a settings switch must not block behind a load or unload.
        /// </summary>
        public bool UpdateSpeculation(SpeculationOptions options)
        {
            options ??= SpeculationOptions.Disabled;
            // Remembered as well as applied: a switch that lands while TryGetEngine is
            // building the engine (the gate held for the block-pool allocation, the
            // environment possibly read before the switch was written) is handed to
            // that engine as soon as it exists, instead of being dropped.
            _pendingSpeculation = options;
            if (!System.Threading.Monitor.TryEnter(_gate, 0))
                return false;
            try
            {
                if (_disposed || _engine == null)
                    return false;
                _engine.UpdateSpeculation(options);
                return true;
            }
            finally
            {
                System.Threading.Monitor.Exit(_gate);
            }
        }

        private volatile SpeculationOptions _pendingSpeculation;

        public void Reset()
        {
            lock (_gate)
            {
                _engine?.Dispose();
                _engine = null;
                _fingerprint = null;
                _engineModel = null;
            }
        }

        public void Dispose()
        {
            lock (_gate)
            {
                if (_disposed) return;
                _disposed = true;
                _engine?.Dispose();
                _engine = null;
                _engineModel = null;
            }
        }
    }
}
