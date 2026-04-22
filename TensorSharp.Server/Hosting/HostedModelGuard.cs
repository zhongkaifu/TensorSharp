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
using System.IO;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Enforces the "one hosted model per process" invariant for every protocol
    /// adapter. The server is launched against a single <c>--model</c> /
    /// <c>--mmproj</c> pair and clients can only reference that exact pair by
    /// file name, model id, or absolute path. Switching to a different file
    /// requires restarting the process.
    /// </summary>
    internal static class HostedModelGuard
    {
        /// <summary>
        /// Validate that <paramref name="requestedModel"/> refers to the model
        /// this server was launched with and return its absolute path.
        /// </summary>
        public static bool TryResolveHostedModelRequest(
            string requestedModel,
            string hostedModelPath,
            out string resolvedModelPath,
            out string error)
        {
            resolvedModelPath = null;

            if (string.IsNullOrWhiteSpace(hostedModelPath))
            {
                error = "No model is hosted by this server. Restart TensorSharp.Server with --model <path.gguf>.";
                return false;
            }

            if (string.IsNullOrWhiteSpace(requestedModel))
            {
                error = "model is required";
                return false;
            }

            if (!MatchesHostedFileRequest(requestedModel, hostedModelPath, allowBareModelId: true))
            {
                error = $"model '{requestedModel}' is not hosted by this server. Restart TensorSharp.Server with --model <path.gguf> to change it.";
                return false;
            }

            resolvedModelPath = hostedModelPath;
            error = null;
            return true;
        }

        /// <summary>
        /// Validate that the optional projector named in a request matches the
        /// one (if any) the server was launched with. <c>null</c> means "the
        /// request did not mention a projector" and is always accepted.
        /// </summary>
        public static bool TryValidateHostedMmProjRequest(
            string requestedMmProj,
            string hostedMmProjPath,
            out string error)
        {
            if (requestedMmProj == null)
            {
                error = null;
                return true;
            }

            bool requestedNoMmProj = string.IsNullOrWhiteSpace(requestedMmProj) ||
                string.Equals(requestedMmProj, "none", StringComparison.OrdinalIgnoreCase);

            if (requestedNoMmProj)
            {
                if (string.IsNullOrWhiteSpace(hostedMmProjPath))
                {
                    error = null;
                    return true;
                }

                error = $"This server was started with mmproj '{Path.GetFileName(hostedMmProjPath)}'. Restart TensorSharp.Server without --mmproj to host no projector.";
                return false;
            }

            if (string.IsNullOrWhiteSpace(hostedMmProjPath))
            {
                error = "This server was started without --mmproj. Restart TensorSharp.Server with --mmproj <path.gguf> to host a projector.";
                return false;
            }

            if (!MatchesHostedFileRequest(requestedMmProj, hostedMmProjPath, allowBareModelId: false))
            {
                error = $"mmproj '{requestedMmProj}' is not hosted by this server. Restart TensorSharp.Server with --mmproj <path.gguf> to change it.";
                return false;
            }

            error = null;
            return true;
        }

        /// <summary>
        /// Resolve the request, then ensure the <see cref="ModelService"/> has
        /// the correct model + projector + backend loaded. A no-op when the
        /// service already reflects the desired state, otherwise loads it.
        /// </summary>
        public static bool TryEnsureHostedModelLoaded(
            ModelService svc,
            string requestedModel,
            string hostedModelPath,
            string hostedMmProjPath,
            string backend,
            out string error)
        {
            error = null;

            if (!TryResolveHostedModelRequest(requestedModel, hostedModelPath, out string resolvedModelPath, out error))
                return false;

            string canonicalBackend = BackendCatalog.Canonicalize(backend);
            if (svc.IsLoaded &&
                string.Equals(svc.LoadedModelPath, resolvedModelPath, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(svc.LoadedMmProjPath ?? string.Empty, hostedMmProjPath ?? string.Empty, StringComparison.OrdinalIgnoreCase) &&
                string.Equals(svc.LoadedBackend, canonicalBackend, StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            svc.LoadModel(resolvedModelPath, hostedMmProjPath, backend);
            if (!svc.IsLoaded)
            {
                error = $"Failed to load hosted model '{Path.GetFileName(resolvedModelPath)}'.";
                return false;
            }

            return true;
        }

        /// <summary>
        /// Compare a client-supplied identifier against a hosted file path.
        /// Matches happen on file name (always), bare id without extension
        /// (only for the model itself), and full absolute path.
        /// </summary>
        public static bool MatchesHostedFileRequest(string requestedValue, string hostedPath, bool allowBareModelId)
        {
            if (string.IsNullOrWhiteSpace(requestedValue) || string.IsNullOrWhiteSpace(hostedPath))
                return false;

            string trimmed = requestedValue.Trim();
            string hostedFileName = Path.GetFileName(hostedPath);
            if (string.Equals(trimmed, hostedFileName, StringComparison.OrdinalIgnoreCase))
                return true;

            if (allowBareModelId &&
                string.Equals(trimmed, Path.GetFileNameWithoutExtension(hostedPath), StringComparison.OrdinalIgnoreCase))
            {
                return true;
            }

            if (trimmed.IndexOf(Path.DirectorySeparatorChar) >= 0 ||
                trimmed.IndexOf(Path.AltDirectorySeparatorChar) >= 0 ||
                Path.IsPathRooted(trimmed))
            {
                return string.Equals(Path.GetFullPath(trimmed), Path.GetFullPath(hostedPath), StringComparison.OrdinalIgnoreCase);
            }

            return false;
        }
    }
}
