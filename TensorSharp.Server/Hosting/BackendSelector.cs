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

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Resolves a client-requested backend (or null for "use the server's
    /// default") into a backend that this host actually supports. Returns a
    /// human-readable error when neither the requested backend nor any default
    /// is available.
    /// </summary>
    internal static class BackendSelector
    {
        public static bool TryResolveSupportedBackend(
            ServerHostingOptions options,
            string requestedBackend,
            out string resolvedBackend,
            out string error)
        {
            if (options == null) throw new ArgumentNullException(nameof(options));

            resolvedBackend = string.IsNullOrWhiteSpace(requestedBackend)
                ? options.DefaultBackend
                : BackendCatalog.Canonicalize(requestedBackend);

            if (string.IsNullOrWhiteSpace(resolvedBackend) ||
                !options.SupportedBackendValues.Contains(resolvedBackend))
            {
                error = options.SupportedBackends.Count == 0
                    ? "No supported backend is available on this machine."
                    : $"Backend '{requestedBackend ?? options.DefaultBackend}' is not supported on this machine.";
                resolvedBackend = null;
                return false;
            }

            error = null;
            return true;
        }
    }
}
