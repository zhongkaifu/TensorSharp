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
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    /// <summary>
    /// Attaches SPECULATOR WEIGHTS that ship as their own file to the target
    /// model that was just loaded — layer 3 of the speculative stack, and the
    /// one part that is unavoidably model-specific.
    ///
    /// Most checkpoints carry their drafter inside the trunk GGUF (Qwen 3.6 and
    /// GLM-5.2 embed a NextN block), so nothing happens here. Gemma 4's draft
    /// head ships separately as <c>gemma4-assistant</c> and has to be loaded
    /// onto the target before <see cref="IDraftHead.HasDraftHead"/> turns on.
    ///
    /// Shared by the CLI and the server so a <c>--draft-model</c> means the
    /// same thing in both. It used to live only in the server, which made the
    /// CLI accept the flag and silently ignore it.
    /// </summary>
    public static class SpeculativeDraftHeadLoader
    {
        /// <summary>
        /// Path of the separate draft-head GGUF the operator configured, or null.
        /// </summary>
        public static string ConfiguredDraftHeadPath()
        {
            string path = Environment.GetEnvironmentVariable(SpeculationEnvVars.DraftModel);
            if (string.IsNullOrWhiteSpace(path))
                path = Environment.GetEnvironmentVariable(SpeculationEnvVars.LegacyDraftModel);
            return string.IsNullOrWhiteSpace(path) ? null : path.Trim();
        }

        /// <summary>
        /// Load the configured draft-head GGUF onto <paramref name="model"/>.
        /// Returns true when there was nothing to do or the head attached
        /// successfully; false with <paramref name="error"/> set to an
        /// operator-facing explanation otherwise. Never throws: a drafter is an
        /// optimization, and failing to attach one must degrade to plain
        /// decoding rather than fail the load.
        /// </summary>
        public static bool TryAttachConfiguredDraftHead(ModelBase model, out string error)
        {
            error = null;
            string draftPath = ConfiguredDraftHeadPath();
            if (draftPath == null)
                return true;

            // A trunk that refuses speculation for correctness refuses every
            // drafter too; say so instead of loading weights nothing may use.
            if (model is ISpeculativeTarget { SpeculationRefusal: { } refusal })
            {
                error = $"--draft-model '{Path.GetFileName(draftPath)}' is not attached: {refusal}";
                return false;
            }

            // Block drafters have to participate in model construction so their
            // weights are included in device placement/layer splitting. If the
            // factory already produced a usable one (DSpark or DFlash), the
            // attach-after-load phase is complete. Without this guard a valid
            // DSpark is misrouted as a Gemma-style head and logged as disabled.
            if (model is IDraftHead
                {
                    HasDraftHead: true,
                    DraftHeadKind: DraftHeadKind.Block,
                })
            {
                return true;
            }

            if (!File.Exists(draftPath))
            {
                error = $"Draft-head model file not found: {draftPath}";
                return false;
            }

            if (model is Qwen4ExpModel qwen4Exp)
            {
                try
                {
                    qwen4Exp.LoadMtpDraftWeights(draftPath);
                    return qwen4Exp.HasDraftHead;
                }
                catch (Exception ex)
                {
                    error = $"Failed to load Qwen4Exp shared MTP head '{Path.GetFileName(draftPath)}': {ex.Message}";
                    return false;
                }
            }

            // A DFlash / DFlash2 drafter is architecture-agnostic on this side: any
            // target that can tap the residuals its encoder reads can host one, and
            // the file says which it is. --draft-model may already have attached it
            // during construction, in which case there is nothing to do.
            if (IsDFlashDrafter(draftPath))
            {
                if (model == null)
                {
                    error = "No model is loaded to attach a DFlash drafter to.";
                    return false;
                }
                if (model.HasDFlash)
                    return true;
                try
                {
                    model.LoadDFlashDraftWeights(draftPath);
                }
                catch (Exception ex)
                {
                    error = $"Failed to load DFlash drafter '{Path.GetFileName(draftPath)}': {ex.Message}";
                    return false;
                }
                if (!model.HasDFlash)
                {
                    error = $"DFlash drafter '{Path.GetFileName(draftPath)}' loaded but is incomplete "
                            + "(required draft tensors missing).";
                    return false;
                }
                return true;
            }

            if (model is not Gemma4Model gemma4)
            {
                // A draft GGUF was named but this architecture does not consume a
                // separate draft file (Qwen 3.6 embeds its NextN block in the
                // trunk). Say so rather than leave the operator wondering why
                // their flag was ignored.
                error = $"--draft-model was given but the loaded model architecture "
                        + $"'{model?.Config?.Architecture ?? "unknown"}' does not use a separate draft GGUF.";
                return false;
            }

            try
            {
                gemma4.LoadMtpDraftWeights(draftPath);
            }
            catch (Exception ex)
            {
                error = $"Failed to load draft head '{Path.GetFileName(draftPath)}': {ex.Message}";
                return false;
            }

            if (!gemma4.HasDraftHead)
            {
                error = $"Draft head '{Path.GetFileName(draftPath)}' loaded but is incomplete "
                        + "(required draft tensors missing).";
                return false;
            }
            return true;
        }

        /// <summary>True when the file at <paramref name="path"/> declares itself a
        /// DFlash drafter. Read from the GGUF rather than inferred from the name:
        /// the same flag also names MTP-only assistant files.</summary>
        private static bool IsDFlashDrafter(string path)
        {
            try
            {
                using var probe = new GgufFile(path);
                return string.Equals(probe.GetString("general.architecture"),
                    DFlashConfig.ArchName, StringComparison.Ordinal);
            }
            catch (Exception ex)
            {
                // Routing is unchanged (per-token draft head), but say the probe
                // failed: a truncated/unreadable DFlash file would otherwise be
                // misclassified without a trace. Once per path.
                bool firstForPath;
                lock (_probeFailureWarned)
                    firstForPath = _probeFailureWarned.Add(path);
                if (firstForPath)
                {
                    Console.Error.WriteLine(
                        $"WARNING: Could not read general.architecture from draft model {path} " +
                        $"({ex.Message}); treating it as a per-token draft head. Reported once.");
                }
                return false;
            }
        }

        private static readonly System.Collections.Generic.HashSet<string> _probeFailureWarned = new(StringComparer.Ordinal);
    }
}
