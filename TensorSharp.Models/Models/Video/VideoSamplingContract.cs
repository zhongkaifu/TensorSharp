// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The sampling contract a video model is bound to when something other than its own
// defaults fixes the schedule: a step-distillation LoRA plug-in, for one. Hosts show it
// (Studio's capability, the recipe a job was admitted for) and compare it by Id; the
// model enforces it on every request.
namespace TensorSharp.Models.Video
{
    /// <summary>
    /// A fixed sampling recipe: the step count, guidance, sigma schedule, sampler and flow
    /// shifts a video model samples with. A request may leave each value unset (0 / null)
    /// or repeat it; any other value contradicts the recipe and is refused, never
    /// overridden.
    /// </summary>
    /// <param name="Id">Stable identity a host binds a job to, e.g.
    /// <c>minimax-h3-lora:minimax-h3-fl2v-turbo-8step</c>; <see cref="NativeId"/> for the
    /// model's own schedule.</param>
    /// <param name="DisplayName">Human-readable name, e.g. the plug-in's name.</param>
    /// <param name="Source">Where the recipe comes from (the plug-in config's path).</param>
    /// <param name="Steps">The exact number of denoising steps.</param>
    /// <param name="CfgScale">The guidance scale (1 = no negative pass).</param>
    /// <param name="Schedule">The sigma grid, e.g. <c>linspace</c>.</param>
    /// <param name="Sampler">The sampler, e.g. <c>euler</c>.</param>
    /// <param name="VideoShift">The video stream's flow shift.</param>
    /// <param name="AudioShift">The audio stream's flow shift (0 for a model without audio).</param>
    public sealed record VideoSamplingContract(string Id, string DisplayName, string Source, int Steps, float CfgScale,
        string Schedule, string Sampler, float VideoShift, float AudioShift)
    {
        /// <summary>The id of a model's own built-in schedule, when no recipe binds it.</summary>
        public const string NativeId = "native";
    }
}
