// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The Q8_0 hyper-connection read (Strata's default, STRATA_HC_Q8): Qwen3.8-Flash-Next stores the down and up
// matrices of its 97 hyper-connection mixers in BF16, 1.26 GB that every token reads. As Q8_0 they are 0.67 GB.
// Strata measured mean KL 0.0007 against the BF16 read (top-1 98.9%, perplexity unchanged). Opt-in here
// (TS_Q4E_HC_Q8=1): it changes the arithmetic.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        private const int GgmlTypeBf16 = WeightRequant.GgmlTypeBf16, GgmlTypeQ8_0 = WeightRequant.GgmlTypeQ8_0;
        internal static readonly bool HcQ8 = Environment.GetEnvironmentVariable("TS_Q4E_HC_Q8") == "1";
        private readonly Dictionary<string, (IntPtr Ptr, long Bytes)> _hcQ8 = new();
        private bool _hcQ8Declined;

        internal static bool IsHcMatrix(string name)
            => name.EndsWith(".hc_attn_down.weight", StringComparison.Ordinal) || name.EndsWith(".hc_attn_up.weight", StringComparison.Ordinal)
            || name.EndsWith(".hc_ffn_down.weight", StringComparison.Ordinal) || name.EndsWith(".hc_ffn_up.weight", StringComparison.Ordinal)
            || name == "output_hc_down.weight" || name == "output_hc_up.weight";

        /// <summary>A BF16 hyper-connection matrix as Q8_0 under TS_Q4E_HC_Q8=1, converted once; false otherwise.</summary>
        private unsafe bool TryHcQ8(string name, QuantizedWeight qw, out IntPtr ptr, out long bytes)
        {
            ptr = IntPtr.Zero;
            bytes = 0;
            if (!HcQ8 || qw.GgmlType != GgmlTypeBf16 || !IsHcMatrix(name))
                return false;
            if (!qw.HasHostData || qw.Ne0 % 32 != 0)
            {
                if (!_hcQ8Declined)
                {
                    _hcQ8Declined = true;
                    Console.Error.WriteLine($"[qwen4exp] TS_Q4E_HC_Q8=1: {name} stays BF16 ({(qw.HasHostData ? $"its rows of {qw.Ne0} are not whole Q8_0 blocks" : "its host bytes were released")}); such matrices keep the exact BF16 read.");
                }
                return false;
            }
            lock (_hcQ8)
            {
                if (_hcQ8.TryGetValue(name, out var c))
                {
                    (ptr, bytes) = c;
                    return true;
                }
                bytes = WeightRequant.Q8_0Bytes(qw.Ne1, qw.Ne0);
                IntPtr dst = (IntPtr)NativeMemory.AlignedAlloc((nuint)bytes, 64);
                WeightRequant.Bf16ToQ8_0((ushort*)qw.Data, (byte*)dst, qw.Ne1, qw.Ne0);
                _hcQ8[name] = (dst, bytes);
                ptr = dst;
            }
            return true;
        }

        private unsafe void FreeHcQ8()
        {
            lock (_hcQ8)
            {
                foreach (var (p, _) in _hcQ8.Values)
                {
                    // The resident cache is keyed by the host address: drop it before the address can be reused.
                    if (IsGgmlBackend)
                        GgmlBasicOps.InvalidateHostBuffer(p);
                    NativeMemory.AlignedFree((void*)p);
                }
                _hcQ8.Clear();
            }
        }
    }
}
