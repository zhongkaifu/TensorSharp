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
    ///   1600-1699  uploads / media / agent skills / code execution
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
        public static readonly EventId RequestContentDropped = new(1104, nameof(RequestContentDropped));

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

        // Paged KV cache ---------------------------------------------------
        public static readonly EventId PagedKvCacheTierInit = new(1530, nameof(PagedKvCacheTierInit));
        public static readonly EventId PagedKvCacheCapture = new(1531, nameof(PagedKvCacheCapture));
        public static readonly EventId PagedKvCacheCaptureSkip = new(1532, nameof(PagedKvCacheCaptureSkip));
        public static readonly EventId PagedKvCacheRestore = new(1533, nameof(PagedKvCacheRestore));
        public static readonly EventId PagedKvCacheRestoreSkip = new(1534, nameof(PagedKvCacheRestoreSkip));

        // Uploads / media --------------------------------------------------
        public static readonly EventId UploadReceived = new(1600, nameof(UploadReceived));
        public static readonly EventId UploadRejected = new(1601, nameof(UploadRejected));
        public static readonly EventId UploadCleanup = new(1602, nameof(UploadCleanup));

        // Agent skills -----------------------------------------------------
        // Skills are content a user supplies, so their lifecycle is audited on the
        // same footing as uploads: what was registered, what was refused and why,
        // and every file the model was given out of one.
        public static readonly EventId SkillsScanned = new(1610, nameof(SkillsScanned));
        public static readonly EventId SkillRejected = new(1611, nameof(SkillRejected));
        public static readonly EventId SkillInstalled = new(1612, nameof(SkillInstalled));
        public static readonly EventId SkillRemoved = new(1613, nameof(SkillRemoved));
        public static readonly EventId SkillSelected = new(1614, nameof(SkillSelected));
        public static readonly EventId SkillToolInvoked = new(1615, nameof(SkillToolInvoked));
        public static readonly EventId SkillLoopCapped = new(1616, nameof(SkillLoopCapped));
        public static readonly EventId SkillScriptExecuted = new(1617, nameof(SkillScriptExecuted));

        // Code execution ---------------------------------------------------
        // Its own ids rather than borrowing the skills script one, because these two
        // are the pair an operator most needs to tell apart: a skill script is a file
        // they put on disk and can read beforehand, and a shell command is text a model
        // wrote during the request. An alert that cannot distinguish them is an alert
        // about nothing. What is recorded is metadata only — never the command itself,
        // which is conversation content.
        public static readonly EventId CodeExecRan = new(1620, nameof(CodeExecRan));
        public static readonly EventId CodeExecRefused = new(1621, nameof(CodeExecRefused));
        public static readonly EventId CodeExecArtifacts = new(1622, nameof(CodeExecArtifacts));
        public static readonly EventId CodeExecEgressDenied = new(1623, nameof(CodeExecEgressDenied));
        public static readonly EventId CodeExecPatched = new(1624, nameof(CodeExecPatched));
        public static readonly EventId CodeExecBackgroundJob = new(1625, nameof(CodeExecBackgroundJob));

        // The file surface. Separate ids rather than one "edited" event, because the
        // question these exist to answer is which of the three the model actually reached
        // for — the whole change is a bet about that, and a bet with no counter is a
        // belief. codeexec.rewrote is the one that matters most: it counts the times a
        // file was re-typed to change a little of it, which had never been measurable.
        public static readonly EventId CodeExecRead = new(1626, nameof(CodeExecRead));
        public static readonly EventId CodeExecEdited = new(1627, nameof(CodeExecEdited));
        public static readonly EventId CodeExecRewrote = new(1628, nameof(CodeExecRewrote));

        // A fork of this host that wedged before exec and had to be reaped. The recovery
        // is invisible to the model by design, so this is the only place it is countable —
        // and the count is the difference between a known platform hazard being handled
        // and it quietly costing seconds on every tool call.
        public static readonly EventId CodeExecForkWedged = new(1629, nameof(CodeExecForkWedged));

        // Media generation: their own ids so a query (or an exported log) can tell a clip's
        // start, phase changes, completion and refusal apart.
        public static readonly EventId VideoGenerateStarted = new(1640, nameof(VideoGenerateStarted));
        public static readonly EventId VideoGenerateCompleted = new(1641, nameof(VideoGenerateCompleted));
        public static readonly EventId VideoGenerateRejected = new(1642, nameof(VideoGenerateRejected));
        public static readonly EventId VideoGeneratePhase = new(1643, nameof(VideoGeneratePhase));

        // CLI --------------------------------------------------------------
        public static readonly EventId CliStarted = new(1700, nameof(CliStarted));
        public static readonly EventId CliCompleted = new(1701, nameof(CliCompleted));
        public static readonly EventId CliFailed = new(1702, nameof(CliFailed));
        public static readonly EventId CliBenchmark = new(1710, nameof(CliBenchmark));
        public static readonly EventId CliBatchProgress = new(1711, nameof(CliBatchProgress));
    }
}
