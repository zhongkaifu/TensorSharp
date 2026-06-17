// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Startup-time validation for the optional MTP speculative-decoding draft.
    /// When an operator explicitly names a draft GGUF with <c>--mtp-draft-model</c>,
    /// a load that doesn't activate speculation used to be swallowed as a warning,
    /// leaving the server running with MTP silently off — the operator had to dig
    /// through the log to discover their command did nothing. This helper turns
    /// that reason into a fail-fast startup error with a remediation hint, the same
    /// contract <c>--mmproj</c>/backend/model-path misconfiguration already follow.
    /// Kept as a tiny pure function so it can be unit-tested without loading a model.
    /// </summary>
    internal static class MtpStartupValidation
    {
        /// <summary>
        /// Returns the operator-facing fatal error message when an explicitly
        /// requested MTP draft could not be activated, or <c>null</c> when there is
        /// nothing to fail on (no draft requested, or the draft activated cleanly).
        /// </summary>
        /// <param name="activationError">
        /// The per-load reason recorded by the model lifecycle when an explicit
        /// <c>--mtp-draft-model</c> could not be activated; <c>null</c> when the
        /// draft loaded successfully or no draft was requested.
        /// </param>
        public static string GetFatalActivationError(string activationError)
        {
            if (string.IsNullOrEmpty(activationError))
                return null;

            return "MTP speculative decoding was requested via --mtp-draft-model but the draft head " +
                   "could not be activated: " + activationError + " " +
                   "Use the draft GGUF that matches this target model (the draft's embedding_length_out " +
                   "must equal the target's hidden size — e.g. pair the 12B target with its 12B draft, not the " +
                   "26B-A4B draft), or drop --mtp-draft-model / --mtp-spec to run without speculation.";
        }
    }
}
