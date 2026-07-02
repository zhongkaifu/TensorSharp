// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// EasyCache: whole-step denoise caching for the DiT loop.
//
// Port of stable-diffusion.cpp's EasyCache (src/runtime/easycache.hpp), the
// runtime accelerator sd.cpp ships for exactly this workload. Adjacent
// FlowMatch-Euler steps produce very similar model outputs; instead of running
// the transformer on every step, this cache:
//
//   1. tracks how much the model INPUT (the evolving generated latent) moved
//      since the last step that actually computed,
//   2. converts that input change into a predicted OUTPUT change via the
//      empirical input->output "transformation rate" measured on computed steps,
//   3. accumulates the predicted change across candidate steps; while the
//      accumulated prediction stays below a threshold the step is SKIPPED
//      entirely and each CFG branch's output is reconstructed as
//          output = input + cached(output - input)
//      (the cached diff comes from the last computed step of that branch).
//
// Compared to the First-Block-Cache (QwenImageDiT.Cache.cs) this skips the
// WHOLE forward for BOTH CFG branches — no block-0 probe, no prelude/postlude,
// no device work at all on a skipped step — and its decision costs one host
// pass over the [seq,64] latent (~1 MB). On a compute-bound GPU (the DiT
// forward saturates the power limit) skipping forwards outright is the only
// remaining large lever, which is why this is the default cache mode.
//
// The error is bounded the same way as the reference: the accumulator carries
// across consecutive skips, so drift forces a real compute; the first
// start-fraction of steps (structure formation) and the last steps (final
// detail) always compute.
// ============================================================================
using System;

namespace TensorSharp.Models.QwenImage
{
    /// <summary>
    /// Whole-step denoise cache (EasyCache port). One instance per generation;
    /// the pipeline consults <see cref="TrySkip"/> before running the DiT and
    /// feeds computed outputs back via <see cref="AfterCompute"/>.
    /// </summary>
    internal sealed class QwenImageStepCache
    {
        // Cache-mode selection shared with the First-Block-Cache:
        //   easycache (default) - this whole-step cache, FBC off
        //   fbc                 - the block-level First-Block-Cache only
        //   both                - stack them (more speed, compounding approximation)
        //   off                 - no caching
        // TS_QWEN_DIT_CACHE=0 (legacy knob) still disables everything.
        internal static string Mode =>
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_CACHE") == "0"
                ? "off"
                : (Environment.GetEnvironmentVariable("TS_QWEN_DIT_CACHE_MODE") ?? "easycache").ToLowerInvariant();

        // Accumulated-predicted-change threshold below which a step is skipped.
        // sd.cpp's default (0.2). Lower = fewer skips / closer to no-cache.
        private static readonly float Threshold =
            EnvFloat("TS_QWEN_DIT_EASYCACHE_THRESHOLD", 0.2f);
        // Fraction of steps at the start that always compute (structure forms early;
        // caching there costs the most quality). sd.cpp default 0.15.
        private static readonly float StartPercent =
            EnvFloat("TS_QWEN_DIT_EASYCACHE_START", 0.15f);
        // Fraction of the schedule after which steps always compute again (final
        // detail refinement). sd.cpp default 0.95.
        private static readonly float EndPercent =
            EnvFloat("TS_QWEN_DIT_EASYCACHE_END", 0.95f);

        private static readonly bool Debug =
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_CACHE_DEBUG") == "1";

        // Below this step count the cache stays off: few-step (Lightning-distilled)
        // schedules give every step a large, deliberate role — approximating one skips
        // a meaningful fraction of the whole trajectory for almost no time saved.
        private static readonly int MinSteps = (int)EnvFloat("TS_QWEN_DIT_EASYCACHE_MIN_STEPS", 10);

        private readonly bool _enabled;
        private readonly int _startStep, _endStep;

        // Anchor state (the conditional branch, mirroring sd.cpp's anchor_condition):
        // input/output of the last computed step and the derived predictor state.
        private float[] _prevInput;             // gen latent at last computed step
        private float[] _prevOutput;            // cond velocity at last computed step
        private float _outputPrevNorm;          // mean|prevOutput|
        private float _transformRate;           // mean|Δoutput| / mean|Δinput| (last measured)
        private bool _hasRate;
        private float _cumulativeChange;        // accumulated predicted relative output change
        private float _lastInputChange;         // mean|input - prevInput| of the current decision
        private bool _hasLastInputChange;

        // Cached per-branch diffs (output - input) from the last computed step.
        private float[] _diffCond, _diffNeg;

        private int _skipped, _computed;

        public QwenImageStepCache(int totalSteps)
        {
            _enabled = Mode is "easycache" or "both" && totalSteps >= MinSteps;
            // sd.cpp maps start/end percents through sigma-space; with the (monotone)
            // shifted-linspace FlowMatch schedule that is equivalent to step-index
            // fractions, which is what we use directly.
            _startStep = (int)Math.Ceiling(StartPercent * totalSteps);
            _endStep = (int)(EndPercent * totalSteps);
        }

        public bool Enabled => _enabled;

        /// <summary>
        /// Decide whether this step can be served from the cache. On a hit, fills
        /// <paramref name="vCond"/> (and <paramref name="vNeg"/> when non-null) with the
        /// reconstructed per-branch outputs (gen region, [seq*64]) and returns true.
        /// </summary>
        public bool TrySkip(int step, float[] input, float[] vCond, float[] vNeg)
        {
            if (!_enabled) return false;
            bool active = step >= _startStep && step < _endStep;
            if (!active || _prevInput == null || _prevOutput == null || _diffCond == null ||
                (vNeg != null && _diffNeg == null) || _prevInput.Length != input.Length)
            {
                if (Debug && _enabled) Console.WriteLine($"  [step-cache] step {step}: compute ({(active ? "no prior state" : "outside window")})");
                return false;
            }

            // Mean absolute input change since the last computed step.
            double change = 0;
            for (int i = 0; i < input.Length; i++) change += Math.Abs(input[i] - _prevInput[i]);
            _lastInputChange = (float)(change / input.Length);
            _hasLastInputChange = true;

            if (_outputPrevNorm > 0f && _hasRate && _lastInputChange > 0f)
            {
                float predicted = _transformRate * _lastInputChange / _outputPrevNorm;
                _cumulativeChange += predicted;
                if (_cumulativeChange < Threshold)
                {
                    if (Debug) Console.WriteLine($"  [step-cache] step {step}: SKIP (cum predicted change {_cumulativeChange:F4} < {Threshold:F3})");
                    Apply(_diffCond, input, vCond);
                    if (vNeg != null) Apply(_diffNeg, input, vNeg);
                    _skipped++;
                    return true;
                }
                if (Debug) Console.WriteLine($"  [step-cache] step {step}: compute (cum predicted change {_cumulativeChange:F4} >= {Threshold:F3})");
                _cumulativeChange = 0f;
            }
            else if (Debug)
            {
                Console.WriteLine($"  [step-cache] step {step}: compute (rate not established)");
            }
            return false;
        }

        /// <summary>
        /// Record a computed step: refresh the per-branch diffs and the change-rate
        /// predictor. <paramref name="vCond"/>/<paramref name="vNeg"/> are the PRE-CFG
        /// per-branch outputs over the gen region ([seq*64]); <paramref name="vNeg"/> may
        /// be null when CFG is off.
        /// </summary>
        public void AfterCompute(int step, float[] input, float[] vCond, float[] vNeg)
        {
            _computed++;
            if (!_enabled) return;
            // Like the reference, only track state inside the active window: warmup steps
            // change too fast to seed a useful predictor.
            if (step < _startStep - 1 || step >= _endStep) return;

            _diffCond = Diff(vCond, input, _diffCond);
            if (vNeg != null) _diffNeg = Diff(vNeg, input, _diffNeg);

            double outputChange = 0;
            bool hadPrev = _prevOutput != null && _prevOutput.Length == vCond.Length;
            if (hadPrev)
            {
                for (int i = 0; i < vCond.Length; i++) outputChange += Math.Abs(vCond[i] - _prevOutput[i]);
                outputChange /= vCond.Length;
            }

            _prevInput = CopyInto(_prevInput, input);
            _prevOutput = CopyInto(_prevOutput, vCond);

            double meanAbs = 0;
            for (int i = 0; i < vCond.Length; i++) meanAbs += Math.Abs(vCond[i]);
            _outputPrevNorm = (float)(meanAbs / Math.Max(1, vCond.Length));

            if (_hasLastInputChange && _lastInputChange > 0f && outputChange > 0)
            {
                float rate = (float)(outputChange / _lastInputChange);
                if (float.IsFinite(rate)) { _transformRate = rate; _hasRate = true; }
            }
            _cumulativeChange = 0f;
            _hasLastInputChange = false;
        }

        public string Stats() =>
            $"{_skipped}/{_skipped + _computed} steps skipped (whole-forward), threshold {Threshold:F2}";

        public int SkippedSteps => _skipped;

        private static void Apply(float[] diff, float[] input, float[] output)
        {
            int n = Math.Min(output.Length, Math.Min(diff.Length, input.Length));
            for (int i = 0; i < n; i++) output[i] = input[i] + diff[i];
        }

        private static float[] Diff(float[] output, float[] input, float[] reuse)
        {
            int n = Math.Min(output.Length, input.Length);
            var d = reuse != null && reuse.Length == n ? reuse : new float[n];
            for (int i = 0; i < n; i++) d[i] = output[i] - input[i];
            return d;
        }

        private static float[] CopyInto(float[] dst, float[] src)
        {
            if (dst == null || dst.Length != src.Length) dst = new float[src.Length];
            Array.Copy(src, dst, src.Length);
            return dst;
        }

        private static float EnvFloat(string name, float dflt)
        {
            var v = Environment.GetEnvironmentVariable(name);
            return v != null && float.TryParse(v, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var f) ? f : dflt;
        }
    }
}
