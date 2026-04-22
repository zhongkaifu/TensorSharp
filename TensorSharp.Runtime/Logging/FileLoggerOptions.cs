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
using Microsoft.Extensions.Logging;

namespace TensorSharp.Runtime.Logging
{
    /// <summary>
    /// Tunable knobs for <see cref="FileLoggerProvider"/>. Defaults are deliberately
    /// modest so the file sink remains useful out-of-the-box without requiring callers
    /// to think about sizing or rotation.
    /// </summary>
    public sealed class FileLoggerOptions
    {
        /// <summary>
        /// Directory the logger writes to. The directory is created on first write
        /// if it does not already exist.
        /// </summary>
        public string Directory { get; set; } = "logs";

        /// <summary>
        /// Filename prefix; the rotated file becomes
        /// <c>{Prefix}-yyyyMMdd[-NNNN].jsonl</c>.
        /// </summary>
        public string FilePrefix { get; set; } = "tensorsharp";

        /// <summary>
        /// Drop log entries below this level before they reach the file. Use
        /// <see cref="LogLevel.None"/> to disable file logging entirely.
        /// </summary>
        public LogLevel MinimumLevel { get; set; } = LogLevel.Information;

        /// <summary>
        /// Soft cap on the size of any single log file. Once exceeded, a new file with
        /// an incremented numeric suffix is created. Must be positive.
        /// </summary>
        public long MaxFileSizeBytes { get; set; } = 25 * 1024 * 1024;

        /// <summary>
        /// Maximum number of buffered entries before producers begin to drop the
        /// oldest pending records to keep memory bounded. Must be positive.
        /// </summary>
        public int MaxQueuedEntries { get; set; } = 16 * 1024;

        /// <summary>
        /// Background flush cadence. Each flush drains the buffered queue to disk.
        /// Set lower for snappier crash recovery, higher to reduce I/O syscalls.
        /// </summary>
        public TimeSpan FlushInterval { get; set; } = TimeSpan.FromMilliseconds(500);

        /// <summary>
        /// When true (default), each entry is rendered as a single JSON object on its
        /// own line. When false, a more human-readable plain-text format is used.
        /// </summary>
        public bool UseJsonFormat { get; set; } = true;

        /// <summary>
        /// When true (the default), include scope state in each emitted log line.
        /// </summary>
        public bool IncludeScopes { get; set; } = true;

        internal void Validate()
        {
            if (string.IsNullOrWhiteSpace(Directory))
                throw new ArgumentException("FileLoggerOptions.Directory must be a non-empty path.", nameof(Directory));
            if (string.IsNullOrWhiteSpace(FilePrefix))
                throw new ArgumentException("FileLoggerOptions.FilePrefix must be a non-empty string.", nameof(FilePrefix));
            if (MaxFileSizeBytes <= 0)
                throw new ArgumentException("FileLoggerOptions.MaxFileSizeBytes must be positive.", nameof(MaxFileSizeBytes));
            if (MaxQueuedEntries <= 0)
                throw new ArgumentException("FileLoggerOptions.MaxQueuedEntries must be positive.", nameof(MaxQueuedEntries));
            if (FlushInterval <= TimeSpan.Zero)
                throw new ArgumentException("FileLoggerOptions.FlushInterval must be positive.", nameof(FlushInterval));
        }
    }
}
