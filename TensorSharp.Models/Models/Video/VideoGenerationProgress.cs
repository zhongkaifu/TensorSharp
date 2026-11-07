// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Progress reporting shared by every video-generation model.
using System.Collections.Generic;

namespace TensorSharp.Models.Video
{
    /// <summary>A progress report from a running video generation.</summary>
    public class VideoGenerationProgress
    {
        /// <summary>"text-encode", "image-encode", "denoise", "vae-decode",
        /// "audio-decode" or "done".</summary>
        public string Phase { get; init; }
        /// <summary>A sub-stage within <see cref="Phase"/>, e.g. LTX 2.5's "first-pass" and
        /// "refinement" inside "denoise", or MiniMax-H3's "dit-load" inside "text-encode";
        /// null = none. A model states it on every report made while the stage runs, so a
        /// report without one ends the stage. Hosts that only follow phases can ignore it:
        /// a stage never changes <see cref="Phase"/>.</summary>
        public string Stage { get; init; }
        /// <summary>Facts about the current stage (whether the denoiser was reused, how long
        /// the encoder took to open), keyed by OpenTelemetry attribute name
        /// (<c>tensorsharp.video.*</c>), for telemetry. Never user content; null = none.</summary>
        public IReadOnlyDictionary<string, object> Attributes { get; init; }
        /// <summary>Completed denoising steps (0-based count of finished steps).</summary>
        public int Step { get; init; }
        /// <summary>Total denoising steps for this request.</summary>
        public int TotalSteps { get; init; }
        /// <summary>Human-readable detail, e.g. "cond pass" or "band 3/7".</summary>
        public string Detail { get; init; }
        /// <summary>Seconds since generation started.</summary>
        public double ElapsedSeconds { get; init; }
        /// <summary>Estimated seconds still to run, or -1 before the first pass finishes.</summary>
        public double EtaSeconds { get; init; } = -1;
        /// <summary>True when this report is a periodic "still working" tick rather
        /// than a phase transition.</summary>
        public bool Heartbeat { get; init; }
    }
}
