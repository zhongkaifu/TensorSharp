// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Models
{
    /// <summary>
    /// Helpers for <see cref="ModelBase.KVStateFingerprint"/> overrides.
    ///
    /// <para>A fingerprint names the SHAPE of the state a model resumes from: the engine
    /// is rebuilt when it changes, persisted checkpoints are keyed by it, and the paged
    /// prefix cache salts every block hash with it. It must therefore be non-empty for
    /// every engine-served family (two models reporting <c>""</c> would share an engine
    /// and each other's reuse state) and stable for a model's whole life (the engine host
    /// reads it on every request, so a value that changed mid-life would tear the engine
    /// down under running requests).</para>
    /// </summary>
    internal static class KvStateFingerprints
    {
        /// <summary>Per-layer layout (which layers are recurrent, which carry PLE, their
        /// compression ratios...) folded into 8 hex characters, so two checkpoints with
        /// the same counts but a different arrangement still differ. FNV-1a, stable across
        /// processes and runtimes (never <see cref="object.GetHashCode"/>).</summary>
        internal static string Layout(bool[] flags)
        {
            if (flags == null) return "-";
            uint h = 2166136261;
            h = Mix(h, flags.Length);
            for (int i = 0; i < flags.Length; i++)
                h = Mix(h, flags[i] ? 1 : 0);
            return h.ToString("x8");
        }

        /// <inheritdoc cref="Layout(bool[])"/>
        internal static string Layout(int[] values)
        {
            if (values == null) return "-";
            uint h = 2166136261;
            h = Mix(h, values.Length);
            for (int i = 0; i < values.Length; i++)
                h = Mix(h, values[i]);
            return h.ToString("x8");
        }

        private static uint Mix(uint h, int value)
        {
            unchecked
            {
                for (int b = 0; b < 4; b++)
                {
                    h ^= (byte)(value >> (8 * b));
                    h *= 16777619;
                }
                return h;
            }
        }
    }
}
