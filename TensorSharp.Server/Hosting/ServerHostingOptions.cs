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
using System.Collections.Generic;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Immutable bag of values resolved at process start-up from CLI arguments
    /// and environment variables. Registered as a DI singleton so every
    /// endpoint, adapter, and helper can pull the same view of "what is hosted
    /// on this server" without each one re-parsing argv.
    /// </summary>
    internal sealed class ServerHostingOptions
    {
        public ServerHostingOptions(
            string startupModelPath,
            string startupMmProjPath,
            string defaultBackend,
            IReadOnlyList<BackendOption> supportedBackends,
            int defaultWebMaxTokens,
            int maxTextFileChars,
            string uploadDirectory,
            string logDirectory,
            bool fileLoggingEnabled)
        {
            StartupModelPath = startupModelPath;
            StartupMmProjPath = startupMmProjPath;
            DefaultBackend = defaultBackend;
            SupportedBackends = supportedBackends ?? Array.Empty<BackendOption>();
            SupportedBackendValues = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            for (int i = 0; i < SupportedBackends.Count; i++)
                SupportedBackendValues.Add(SupportedBackends[i].Value);
            DefaultWebMaxTokens = defaultWebMaxTokens;
            MaxTextFileChars = maxTextFileChars;
            UploadDirectory = uploadDirectory;
            LogDirectory = logDirectory;
            FileLoggingEnabled = fileLoggingEnabled;
        }

        /// <summary>Absolute path of the model the server was launched with, or null when no model is hosted.</summary>
        public string StartupModelPath { get; }

        /// <summary>Absolute path of the projector the server was launched with, or null when none is hosted.</summary>
        public string StartupMmProjPath { get; }

        /// <summary>Canonical name of the backend chosen at startup (e.g. <c>ggml_metal</c>).</summary>
        public string DefaultBackend { get; }

        /// <summary>Backends actually supported by this host (after probing the GGML runtime).</summary>
        internal IReadOnlyList<BackendOption> SupportedBackends { get; }

        /// <summary>Fast lookup over <see cref="SupportedBackends"/>.</summary>
        internal HashSet<string> SupportedBackendValues { get; }

        /// <summary>Default token budget for the Web UI's chat endpoint.</summary>
        public int DefaultWebMaxTokens { get; }

        /// <summary>Character cap for text uploads when no model tokenizer is available.</summary>
        public int MaxTextFileChars { get; }

        /// <summary>Absolute path to the directory used for user uploads.</summary>
        public string UploadDirectory { get; }

        /// <summary>Resolved log directory (used by the file logger when it is enabled).</summary>
        public string LogDirectory { get; }

        /// <summary>True when the file logger should be wired in.</summary>
        public bool FileLoggingEnabled { get; }
    }
}
