// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.Cpu;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// The Gated DeltaNet on MLX: the state lives in MLX arrays and every step runs MLX's fused
    /// kernels (conv + gates preprocessing, the delta-rule scan, gated norm), so the recurrence joins
    /// the step's lazy MLX graph instead of round-tripping through the host (as Qwen35Model does).
    /// </summary>
    public sealed partial class Qwen35RuntimeModel
    {
        /// <summary>The recurrence runs on MLX (one rank: the kernels take whole head groups).</summary>
        private bool MlxRecurrence => _backend == BackendType.Mlx && ShardDegree == 1;

        private static CpuAllocator s_hostAllocator;

        private static CpuAllocator HostAllocator => s_hostAllocator ??= new CpuAllocator(BlasEnum.DotNet);

        /// <summary>One sequence's state for one layer, in MLX arrays (immutable: a copy shares them).
        /// Host mirrors exist only for checkpoint files.</summary>
        private sealed class MlxGdnState : RecurrentState
        {
            private readonly int _convRows, _channels, _vHeads, _dv, _dk;
            private Tensor _hostConv, _hostSsm;

            public MlxGdnState(int convRows, int channels, int vHeads, int dv, int dk)
            {
                _convRows = convRows;
                _channels = channels;
                _vHeads = vHeads;
                _dv = dv;
                _dk = dk;
            }

            private MlxFusedOps.GatedDeltaNetCache _beforeVerify;

            public MlxFusedOps.GatedDeltaNetCache Cache { get; } = new();

            public override void Reset()
            {
                Cache.Reset();
                DropVerify();
            }

            protected internal override bool SupportsVerify => true;

            /// <summary>The window runs its rows one scan step at a time and keeps each row's state
            /// (MLX arrays are immutable, so the state before it is a reference).</summary>
            protected internal override void BeginVerify()
            {
                DropVerify();
                _beforeVerify = Cache.CloneState();
                Cache.KeepRowStates = true;
            }

            protected internal override void CommitVerify(int kept)
            {
                Cache.KeepRowStates = false;
                if (kept == 0)
                {
                    if (_beforeVerify != null) Cache.RestoreFrom(_beforeVerify);
                }
                else if (Cache.RowStates > 0 && !Cache.CommitRow(kept - 1))
                {
                    throw new InvalidOperationException($"No recurrent state after row {kept - 1} of a {Cache.RowStates}-row verify.");
                }
                Cache.ClearRows();
                DropVerify();
            }

            private void DropVerify()
            {
                _beforeVerify?.Dispose();
                _beforeVerify = null;
            }

            protected internal override bool CanCopy => true;

            protected internal override void CopyFrom(RecurrentState source) => Cache.RestoreFrom(((MlxGdnState)source).Cache);

            /// <summary>The state exported to host tensors ([convRows, channels] oldest row first and
            /// [vHeads, dv, dk]); a file import writes them and <see cref="OnHostWritten"/> loads them.</summary>
            protected internal override IEnumerable<Tensor> CheckpointTensors
            {
                get
                {
                    _hostConv ??= new Tensor(HostAllocator, DType.Float32, Math.Max(1, _convRows), _channels);
                    _hostSsm ??= new Tensor(HostAllocator, DType.Float32, _vHeads, _dv, _dk);
                    unsafe
                    {
                        Cache.ExportState((float*)TensorComputePrimitives.GetStoragePointer(_hostConv), _hostConv.ElementCount(),
                            (float*)TensorComputePrimitives.GetStoragePointer(_hostSsm), _hostSsm.ElementCount());
                    }
                    return new[] { _hostConv, _hostSsm };
                }
            }

            protected internal override void OnHostWritten()
            {
                if (_hostConv == null || _hostSsm == null) return;
                unsafe
                {
                    Cache.ImportState((float*)TensorComputePrimitives.GetStoragePointer(_hostConv), _convRows, _channels,
                        (float*)TensorComputePrimitives.GetStoragePointer(_hostSsm), _vHeads, _dv, _dk);
                }
            }

            public override void Dispose()
            {
                DropVerify();
                Cache.Dispose();
                _hostConv?.Dispose();
                _hostSsm?.Dispose();
            }
        }

        private Tensor MlxGatedDeltaNet(RankScope scope, Tensor normed, int layer, MlxGdnState state)
        {
            string p = $"blk.{layer}.";
            int tokens = (int)normed.Sizes[0];
            int qkvDim = 2 * QkDim + VDim;
            Tensor conv = _weights[p + "ssm_conv1d.weight"], dt = _weights[p + "ssm_dt.bias"];
            Tensor a = _weights[p + "ssm_a"], norm = _weights[p + "ssm_norm.weight"];
            using var gated = new Tensor(scope.Allocator, DType.Float32, tokens, VDim);
            bool ran;
            if (HasWeight(InProjName(layer)))
            {
                using Tensor packed = Linear(scope, normed, InProjName(layer));
                ran = state.Cache.TryRunQwen35Packed(gated, packed, conv, dt, a, norm, tokens, (int)packed.Sizes[1], qkvDim,
                    QkDim, VDim, _numKHeads, _numVHeads, _headKDim, _headVDim, _convKernel, Config.Eps);
            }
            else
            {
                using Tensor qkv = Linear(scope, normed, p + "attn_qkv.weight");
                using Tensor z = Linear(scope, normed, p + "attn_gate.weight");
                using Tensor beta = Linear(scope, normed, p + "ssm_beta.weight");
                using Tensor alpha = Linear(scope, normed, p + "ssm_alpha.weight");
                ran = state.Cache.TryRunQwen35(gated, qkv, z, beta, alpha, conv, dt, a, norm, tokens, qkvDim,
                    QkDim, VDim, _numKHeads, _numVHeads, _headKDim, _headVDim, _convKernel, Config.Eps);
            }
            if (!ran)
                throw new InvalidOperationException($"{Config.Architecture}: MLX's Gated DeltaNet kernels declined layer {layer}.");
            return Linear(scope, gated, p + "ssm_out.weight");
        }
    }
}
