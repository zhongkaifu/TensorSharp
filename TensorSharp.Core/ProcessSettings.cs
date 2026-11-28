// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp
{
    /// <summary>
    /// Settings a native library takes once per process (D04: the process scope). Some initialization choices cannot be
    /// isolated per model or per backend session, however the managed options are shaped: ggml keeps one backend type per
    /// process, MLX latches its F32 GEMM precision (<c>MLX_ENABLE_TF32</c>) on its first GEMM, ggml-cuda latches graph
    /// capture at backend init. Whoever needs such a setting claims it here before the library takes it: the first claim
    /// fixes the value for the process, the same value claimed again is fine, and a different value is a conflict that is
    /// reported (<see cref="ProcessSettingConflictException"/>) instead of silently running with the first claimant's choice.
    /// </summary>
    public static class ProcessSettings
    {
        private static readonly object s_gate = new();
        private static readonly Dictionary<string, (string Value, string Owner)> s_claims = new(StringComparer.Ordinal);

        /// <summary>Claims <paramref name="setting"/> = <paramref name="value"/> for the process on behalf of
        /// <paramref name="owner"/> (a model family, a backend). Throws <see cref="ProcessSettingConflictException"/> when the
        /// process already took another value.</summary>
        public static void Claim(string setting, string value, string owner)
        {
            if (!TryClaim(setting, value, owner, out ProcessSettingConflictException? conflict))
                throw conflict!;
        }

        /// <summary><see cref="Claim"/> that returns the conflict instead of throwing it (a caller that reports it as a
        /// warning, where refusing would break a workflow that ran before).</summary>
        public static bool TryClaim(string setting, string value, string owner, out ProcessSettingConflictException? conflict)
        {
            ArgumentException.ThrowIfNullOrEmpty(setting);
            ArgumentNullException.ThrowIfNull(value);
            ArgumentException.ThrowIfNullOrEmpty(owner);
            lock (s_gate)
            {
                if (s_claims.TryGetValue(setting, out var held))
                {
                    if (string.Equals(held.Value, value, StringComparison.Ordinal))
                    {
                        conflict = null;
                        return true;
                    }
                    conflict = new ProcessSettingConflictException(setting, held.Value, held.Owner, value, owner);
                    return false;
                }
                s_claims[setting] = (value, owner);
                conflict = null;
                return true;
            }
        }

        /// <summary>The value the process took for <paramref name="setting"/>, and who claimed it.</summary>
        public static bool TryGet(string setting, out string value, out string owner)
        {
            lock (s_gate)
            {
                if (s_claims.TryGetValue(setting, out var held))
                {
                    (value, owner) = held;
                    return true;
                }
            }
            value = owner = null!;
            return false;
        }

        /// <summary>Forgets a claim: only for a setting whose library really was torn down (tests that recreate a backend).</summary>
        public static void ReleaseForTests(string setting)
        {
            lock (s_gate) s_claims.Remove(setting);
        }
    }

    /// <summary>A process-scope setting was claimed with two values (<see cref="ProcessSettings"/>).</summary>
    public sealed class ProcessSettingConflictException : InvalidOperationException
    {
        public ProcessSettingConflictException(string setting, string heldValue, string heldOwner, string wantedValue, string wantedOwner)
            : base($"{wantedOwner} needs the process setting {setting}={wantedValue}, but {heldOwner} already initialized it as "
                   + $"{setting}={heldValue}; it is taken once per process and cannot be isolated per model. Run them in separate "
                   + $"processes, or set {setting} at launch.")
        {
            Setting = setting;
            HeldValue = heldValue;
            HeldOwner = heldOwner;
            WantedValue = wantedValue;
            WantedOwner = wantedOwner;
        }

        public string Setting { get; }
        public string HeldValue { get; }
        public string HeldOwner { get; }
        public string WantedValue { get; }
        public string WantedOwner { get; }
    }
}
