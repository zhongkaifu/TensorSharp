// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// First-Block-Cache (a.k.a. DBCache Fn=1 / TeaCache-family) for the DiT denoise loop.
//
// The denoise loop runs the full 60-block transformer once (or twice, with CFG)
// per step. Adjacent FlowMatch-Euler steps produce very similar velocity, so the
// contribution of blocks 1..N-1 changes little from step to step. This cache:
//   1. always computes block 0 (cheap relative to the whole transformer),
//   2. measures the relative-L1 change of block 0's output residual vs the last
//      step it actually computed,
//   3. if that change is below a threshold, REUSES the cached blocks-1..N-1
//      residual (img_final = img_after_block0 + cached_residual) and skips the
//      other 59 blocks entirely.
//
// This is the canonical Qwen-Image denoise accelerator (SGLang cache-dit /
// ParaAttention First-Block-Cache, the original TeaCache paper arXiv:2411.14324).
// It reduces the NUMBER of expensive block computations rather than the cost of
// each one, which is the lever the prior fusion/capture/flash-attn work did not
// touch. Default-on for the native GGML path; TS_QWEN_DIT_CACHE=0 disables it.
//
// Each CFG branch (conditional / unconditional) keeps an independent cache so the
// two interleaved forwards do not corrupt one another.
// ============================================================================
using System;

namespace TensorSharp.Models.QwenImage
{
    internal sealed partial class QwenImageDiT
    {
        // Per-CFG-branch cache state.
        private sealed class FbCacheState
        {
            // Block-0 output residual (generated region) of the last step we fully computed;
            // the baseline the next step's residual is compared against.
            public float[] PrevFirstResidual;
            // Cached contribution of blocks 1..N-1 to the img stream (full imgSeq*Dim),
            // captured on the last full computation and re-added on a cache hit.
            public float[] RemainingResidual;
            // Consecutive cached (skipped) steps, capped by MaxContinuousSkip.
            public int ContinuousCached;
        }

        // up to 2 branches: 0 = conditional, 1 = unconditional (CFG negative).
        private readonly FbCacheState[] _cache = { new FbCacheState(), new FbCacheState() };
        private int _cacheTotalSteps;
        private int _cacheComputed, _cacheSkipped;   // stats over a generation

        // First-Block-Cache is only meaningful on the native per-block path (it skips
        // native block calls). Default-on there; TS_QWEN_DIT_CACHE=0 forces every step
        // to compute all blocks (the original behavior).
        internal bool CacheEnabled =>
            NativeBlockOn && Environment.GetEnvironmentVariable("TS_QWEN_DIT_CACHE") != "0";

        // Relative-L1 threshold on the block-0 residual change. Below it, the step is
        // cached. Lower = more conservative (closer to no-cache); higher = more skips.
        // 0.08 is a perceptually-equivalent default for Qwen-Image (validated: same
        // composition/edit vs no-cache, ~1.5x denoise); it is conservative relative to
        // the references (SGLang cache-dit 0.24, ParaAttention 0.12). Raise toward
        // 0.10-0.12 for more speed, lower to 0.05 to stay closer to the no-cache result.
        private static readonly float CacheThreshold =
            EnvFloat("TS_QWEN_DIT_CACHE_THRESHOLD", 0.08f);

        // Always fully compute the first N steps (in addition to step 0 and the last
        // step, which always compute) before any caching is allowed.
        private static readonly int CacheWarmup =
            EnvInt("TS_QWEN_DIT_CACHE_WARMUP", 2);

        // Cap on consecutive cached steps; forces a recompute to re-anchor the cache.
        private static readonly int MaxContinuousSkip =
            EnvInt("TS_QWEN_DIT_CACHE_MAXSKIP", 3);

        // Per-step relative-L1 / decision trace for tuning the threshold.
        private static readonly bool CacheDebug =
            Environment.GetEnvironmentVariable("TS_QWEN_DIT_CACHE_DEBUG") == "1";

        private static float EnvFloat(string name, float dflt)
        {
            var v = Environment.GetEnvironmentVariable(name);
            return v != null && float.TryParse(v, System.Globalization.NumberStyles.Float,
                System.Globalization.CultureInfo.InvariantCulture, out var f) ? f : dflt;
        }
        private static int EnvInt(string name, int dflt)
        {
            var v = Environment.GetEnvironmentVariable(name);
            return v != null && int.TryParse(v, out var i) ? i : dflt;
        }

        /// <summary>Reset the cache at the start of a generation (called by the pipeline).</summary>
        public void ResetCache(int totalSteps)
        {
            _cacheTotalSteps = totalSteps;
            _cacheComputed = _cacheSkipped = 0;
            foreach (var s in _cache)
            {
                s.PrevFirstResidual = null;
                s.RemainingResidual = null;
                s.ContinuousCached = 0;
            }
        }

        /// <summary>Per-generation cache stats string (computed vs skipped block-loop runs).</summary>
        public string CacheStats()
        {
            int total = _cacheComputed + _cacheSkipped;
            if (total == 0) return "cache off";
            return $"{_cacheSkipped}/{total} forwards cached (skipped {_cacheSkipped} full block-loops)";
        }

        // Decide whether the current step (after block 0 has run) can reuse the cached
        // blocks-1..N-1 residual. `state` is advanced as a side effect: on a compute the
        // baseline residual is replaced; on a cache hit only the counter advances.
        private bool DecideUseCache(FbCacheState state, int stepIndex, float[] firstResidualGen)
        {
            bool isBoundary = stepIndex <= 0 || stepIndex >= _cacheTotalSteps - 1;
            bool mustCompute =
                isBoundary ||
                stepIndex < CacheWarmup ||
                state.PrevFirstResidual == null ||
                state.RemainingResidual == null ||
                state.ContinuousCached >= MaxContinuousSkip;

            float relDbg = -1f;
            if (!mustCompute)
            {
                float rel = relDbg = RelL1(firstResidualGen, state.PrevFirstResidual);
                if (rel < CacheThreshold)
                {
                    if (CacheDebug) Console.WriteLine($"  [dit-cache] step {stepIndex} branch={(state == _cache[0] ? 0 : 1)} relL1={rel:F4} < {CacheThreshold:F3} -> SKIP");
                    state.ContinuousCached++;   // cache hit: keep the baseline anchored
                    return true;
                }
            }
            if (CacheDebug && stepIndex > 0)
                Console.WriteLine($"  [dit-cache] step {stepIndex} branch={(state == _cache[0] ? 0 : 1)} relL1={relDbg:F4} (thr {CacheThreshold:F3}) -> compute{(mustCompute ? " (forced)" : "")}");

            // Compute path: re-anchor the baseline to this step's first-block residual.
            state.PrevFirstResidual = firstResidualGen;
            state.ContinuousCached = 0;
            return false;
        }

        // mean(|cur - prev|) / mean(|prev|)  (the TeaCache / cache-dit relative-L1 metric).
        private static float RelL1(float[] cur, float[] prev)
        {
            int n = Math.Min(cur.Length, prev.Length);
            double diff = 0, mag = 0;
            for (int i = 0; i < n; i++)
            {
                diff += Math.Abs(cur[i] - prev[i]);
                mag += Math.Abs(prev[i]);
            }
            return mag > 1e-12 ? (float)(diff / mag) : float.MaxValue;
        }
    }
}
