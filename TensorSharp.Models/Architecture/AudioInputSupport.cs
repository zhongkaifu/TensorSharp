// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// Which families refuse audio input, and with what words. One table so the
    /// OpenAI chat and Responses parsers, the Web UI, the shared chat pipeline,
    /// the CLI's <c>--audio</c> / <c>/audio</c> and the multimodal injector all
    /// give the same answer before an upload is written or a prompt rendered.
    /// Keyed on the GGUF's <c>general.architecture</c> as the model reports it
    /// and resolved through <see cref="ModelArchitectureRegistry"/>, so every
    /// alias of a family (Nemotron's <c>nemotron_h</c> / <c>nemotron_h_moe</c> /
    /// <c>nemotron_h_omni</c>) is covered without a second list.
    /// </summary>
    public static class AudioInputSupport
    {
        /// <summary>DeepSeek V4.1 Flash has no audio tower at all.</summary>
        public const string DeepSeek41Message =
            "DeepSeek V4.1 Flash does not support audio input. Remove audio attachments or use a model with an audio encoder.";

        /// <summary>
        /// The refusal <paramref name="architecture"/> gives audio input, or null
        /// when the family can consume audio or is not one this table knows (an
        /// unknown family is left to its own multimodal contract).
        /// </summary>
        public static string UnsupportedReasonFor(string architecture)
        {
            if (string.IsNullOrEmpty(architecture))
                return null;
            if (string.Equals(architecture, "deepseek41", StringComparison.OrdinalIgnoreCase))
                return DeepSeek41Message;
            if (ModelArchitectureRegistry.TryGet(architecture, out ModelArchitectureDescriptor descriptor) &&
                string.Equals(descriptor.Id, "nemotron_h", StringComparison.OrdinalIgnoreCase))
                return NemotronModel.AudioInputUnsupportedMessage;
            return null;
        }
    }
}
