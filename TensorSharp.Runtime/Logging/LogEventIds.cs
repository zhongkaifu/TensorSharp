// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Stable <see cref="EventId"/> values used by TensorSharp host components. Using
    /// fixed numeric ids makes log alerting and dashboards resilient to message text
    /// changes; the symbolic name remains in code for readability while consumers can
    /// pivot on the int.
    ///
    /// Ranges:
    ///   1000-1099  generic host lifecycle (startup, shutdown, config)
    ///   1100-1199  HTTP request pipeline
    ///   1200-1299  inference queue
    ///   1300-1399  session lifecycle
    ///   1400-1499  model load / unload
    ///   1500-1599  chat / generation operations
    ///   1600-1699  uploads / media
    ///   1700-1799  CLI commands
    /// </summary>
    public static class LogEventIds
    {
        // Host lifecycle ----------------------------------------------------
        public static readonly EventId HostStarting = new(1000, nameof(HostStarting));
        public static readonly EventId HostStarted = new(1001, nameof(HostStarted));
        public static readonly EventId HostStopping = new(1002, nameof(HostStopping));
        public static readonly EventId HostStopped = new(1003, nameof(HostStopped));
        public static readonly EventId HostConfiguration = new(1010, nameof(HostConfiguration));
        public static readonly EventId BackendDetected = new(1020, nameof(BackendDetected));
        public static readonly EventId BackendUnavailable = new(1021, nameof(BackendUnavailable));
        public static readonly EventId LoggingInitialized = new(1030, nameof(LoggingInitialized));

        // HTTP -------------------------------------------------------------
        public static readonly EventId HttpRequestStarted = new(1100, nameof(HttpRequestStarted));
        public static readonly EventId HttpRequestCompleted = new(1101, nameof(HttpRequestCompleted));
        public static readonly EventId HttpRequestFailed = new(1102, nameof(HttpRequestFailed));
        public static readonly EventId HttpRequestRejected = new(1103, nameof(HttpRequestRejected));

        // Inference queue --------------------------------------------------
        public static readonly EventId QueueEnqueued = new(1200, nameof(QueueEnqueued));
        public static readonly EventId QueueReady = new(1201, nameof(QueueReady));
        public static readonly EventId QueueReleased = new(1202, nameof(QueueReleased));
        public static readonly EventId QueueCancelled = new(1203, nameof(QueueCancelled));

        // Session lifecycle ------------------------------------------------
        public static readonly EventId SessionCreated = new(1300, nameof(SessionCreated));
        public static readonly EventId SessionRemoved = new(1301, nameof(SessionRemoved));
        public static readonly EventId SessionReset = new(1302, nameof(SessionReset));
        public static readonly EventId SessionDisposed = new(1303, nameof(SessionDisposed));
        public static readonly EventId SessionActivated = new(1304, nameof(SessionActivated));

        // Model lifecycle --------------------------------------------------
        public static readonly EventId ModelLoadStarted = new(1400, nameof(ModelLoadStarted));
        public static readonly EventId ModelLoadCompleted = new(1401, nameof(ModelLoadCompleted));
        public static readonly EventId ModelLoadFailed = new(1402, nameof(ModelLoadFailed));
        public static readonly EventId ModelUnloaded = new(1403, nameof(ModelUnloaded));

        // Chat / generation ------------------------------------------------
        public static readonly EventId ChatStarted = new(1500, nameof(ChatStarted));
        public static readonly EventId ChatCompleted = new(1501, nameof(ChatCompleted));
        public static readonly EventId ChatFailed = new(1502, nameof(ChatFailed));
        public static readonly EventId ChatAborted = new(1503, nameof(ChatAborted));
        public static readonly EventId KvCacheReusePlan = new(1510, nameof(KvCacheReusePlan));
        public static readonly EventId PromptChunking = new(1511, nameof(PromptChunking));
        public static readonly EventId PromptTruncated = new(1512, nameof(PromptTruncated));
        public static readonly EventId VideoFrameDownsample = new(1513, nameof(VideoFrameDownsample));
        public static readonly EventId GenerationProgress = new(1520, nameof(GenerationProgress));

        // Uploads / media --------------------------------------------------
        public static readonly EventId UploadReceived = new(1600, nameof(UploadReceived));
        public static readonly EventId UploadRejected = new(1601, nameof(UploadRejected));

        // CLI --------------------------------------------------------------
        public static readonly EventId CliStarted = new(1700, nameof(CliStarted));
        public static readonly EventId CliCompleted = new(1701, nameof(CliCompleted));
        public static readonly EventId CliFailed = new(1702, nameof(CliFailed));
        public static readonly EventId CliBenchmark = new(1710, nameof(CliBenchmark));
        public static readonly EventId CliBatchProgress = new(1711, nameof(CliBatchProgress));
    }
}
