// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

using TensorSharp.GGML;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    /// <summary>
    /// Gated DeltaNet on the GGML backends: each rank's block is one native graph
    /// (<c>TSGgml_Qwen35GdnLayerTP</c>: input RMSNorm, the packed in-projection, ssm_conv,
    /// SiLU, q/k L2-norm, ggml_gated_delta_net, the gated RMSNorm and the SiLU(z) gate) on the
    /// rank's device, with the conv window and delta state kept there across calls. The same
    /// kernel serves one device, every stage of a layer split and every rank of tensor
    /// parallelism; the runtime finishes the block with the row-parallel ssm_out.
    /// </summary>
    public partial class Qwen35RuntimeModel
    {
        private bool DeviceRecurrence => IsGgmlBackend;

        /// <summary>The recurrence runs as program nodes over tensor-held states (every backend with a
        /// graph executor on one shard; GGML's per-rank native block under tensor parallelism).</summary>
        private bool TensorRecurrence => DeviceRecurrence || (ShardDegree == 1 && GraphExecutor != null);

        protected override bool RecurrentMixerTakesRawInput => TensorRecurrence;

        private Tensor[][] _deviceConvW;   // [layer][rank] [channels, kernel] (one filter per channel)
        private Tensor[][] _deviceDtBias;  // [layer][rank] [localV]
        private Tensor[][] _deviceA;       // [layer][rank] [localV]
        private Tensor[] _deviceSsmNorm;   // [layer] [headVDim]

        private static string InProjName(int layer) => $"blk.{layer}.ssm_in_proj.weight";

        private static string[] InProjSources(int layer)
            => new[] { $"blk.{layer}.attn_qkv.weight", $"blk.{layer}.attn_gate.weight", $"blk.{layer}.ssm_beta.weight", $"blk.{layer}.ssm_alpha.weight" };

        /// <summary>The four in-projection sources are quantized alike (so packing copies rows).</summary>
        private bool InProjectionSharesType(int layer)
        {
            int type = -1;
            foreach (string name in InProjSources(layer))
            {
                if (!_quantWeights.TryGetValue(name, out QuantizedWeight q)) return false;
                if (type >= 0 && q.GgmlType != type) return false;
                type = q.GgmlType;
            }
            return true;
        }

        /// <summary>
        /// Pack the four input projections into the native block's in-projection,
        /// <c>[qkv | z | beta | alpha]</c> rows in the rank's channel order: rows move as they are
        /// when all four share one quantized type; otherwise they are packed as F32 on one
        /// device and re-encoded as Q8_0 per rank under tensor parallelism (as Qwen35Model does).
        /// The four sources are dropped.
        /// </summary>
        private void PackInProjection(int layer, int ranks, int rankOffset, int degree, Func<int, IAllocator> allocatorOf)
        {
            string p = $"blk.{layer}.";
            string[] sources = { p + "attn_qkv.weight", p + "attn_gate.weight", p + "ssm_beta.weight", p + "ssm_alpha.weight" };
            Func<int, int[]>[] rowsOf =
            {
                r => QkvChannelsOf(r, degree), r => ValueChannelsOf(r, degree),
                r => ValueHeadsOf(r, degree), r => ValueHeadsOf(r, degree),
            };
            int hidden = Config.HiddenSize;
            var quant = new QuantizedWeight[sources.Length];
            var dense = new Tensor[sources.Length];
            int sharedType = -1;
            bool sameQuant = true;
            for (int i = 0; i < sources.Length; i++)
            {
                if (_quantWeights.TryGetValue(sources[i], out quant[i]))
                {
                    if (sharedType < 0) sharedType = quant[i].GgmlType;
                    sameQuant &= quant[i].GgmlType == sharedType;
                }
                else if (_weights.TryGetValue(sources[i], out dense[i])) sameQuant = false;
                else throw new InvalidOperationException($"{Config.Architecture}: '{sources[i]}' is missing.");
            }

            var packedQuant = new QuantizedWeight[ranks];
            var packedDense = new Tensor[ranks];
            for (int r = 0; r < ranks; r++)
            {
                int[][] rows = new int[sources.Length][];
                int total = 0;
                for (int i = 0; i < sources.Length; i++) { rows[i] = rowsOf[i](rankOffset + r); total += rows[i].Length; }
                if (sameQuant)
                    packedQuant[r] = CopyQuantRows(quant, rows, total);
                else if (ranks > 1 && hidden % 32 == 0)
                    packedQuant[r] = RequantRows(quant, dense, rows, total, hidden);
                else
                    packedDense[r] = DenseRows(quant, dense, rows, total, hidden, allocatorOf(r));
            }

            string name = InProjName(layer);
            if (ranks == 1)
            {
                if (packedQuant[0] != null) _quantWeights[name] = packedQuant[0];
                else _weights[name] = packedDense[0];
            }
            else if (packedQuant[0] != null) _tpQuantWeights[name] = packedQuant;
            else _tpWeights[name] = packedDense;

            for (int i = 0; i < sources.Length; i++)
            {
                if (_quantWeights.Remove(sources[i], out var q)) q.Dispose();
                if (_weights.Remove(sources[i], out var d)) d.Dispose();
            }
        }

        private static unsafe QuantizedWeight CopyQuantRows(QuantizedWeight[] sources, int[][] rows, int total)
        {
            QuantizedWeight first = sources[0];
            long rowBytes = NativeDequant.RowSize(first.GgmlType, first.Ne0);
            long bytes = total * rowBytes;
            IntPtr dst = QuantizedWeight.AllocateBuffer(bytes);
            byte* d = (byte*)dst.ToPointer();
            long at = 0;
            for (int i = 0; i < sources.Length; i++)
            {
                byte* src = (byte*)sources[i].Data.ToPointer();
                foreach (int row in rows[i])
                {
                    Buffer.MemoryCopy(src + row * rowBytes, d + at * rowBytes, rowBytes, rowBytes);
                    at++;
                }
            }
            return new QuantizedWeight(dst, bytes, first.GgmlType, first.Ne0, total);
        }

        /// <summary>Row <paramref name="row"/> of source i as F32.</summary>
        private unsafe void ReadRow(QuantizedWeight q, Tensor d, int row, int hidden, float* dst)
        {
            if (q != null)
            {
                long rowBytes = NativeDequant.RowSize(q.GgmlType, q.Ne0);
                NativeDequant.DequantizeToFloat32Native(q.GgmlType, (IntPtr)((byte*)q.Data.ToPointer() + row * rowBytes), (IntPtr)dst, hidden);
                if (q.Scale != 1.0f)
                    for (int c = 0; c < hidden; c++) dst[c] *= q.Scale;
            }
            else
            {
                float* src = GetFloatPtr(d) + (long)row * hidden;
                Buffer.MemoryCopy(src, dst, hidden * sizeof(float), hidden * sizeof(float));
            }
        }

        private unsafe QuantizedWeight RequantRows(QuantizedWeight[] quant, Tensor[] dense, int[][] rows, int total, int hidden)
        {
            const int q8 = (int)GgmlTensorType.Q8_0;
            long rowBytes = NativeDequant.RowSize(q8, hidden);
            long bytes = total * rowBytes;
            IntPtr dst = QuantizedWeight.AllocateBuffer(bytes);
            var row = new float[hidden];
            long at = 0;
            fixed (float* f = row)
            {
                for (int i = 0; i < rows.Length; i++)
                    foreach (int r in rows[i])
                    {
                        ReadRow(quant[i], dense[i], r, hidden, f);
                        ManagedQuantizedOps.QuantizeRowFromFloat32(q8, f, (IntPtr)((byte*)dst.ToPointer() + at * rowBytes), hidden);
                        at++;
                    }
            }
            return new QuantizedWeight(dst, bytes, q8, hidden, total);
        }

        private unsafe Tensor DenseRows(QuantizedWeight[] quant, Tensor[] dense, int[][] rows, int total, int hidden, IAllocator allocator)
        {
            var packed = new Tensor(allocator, DType.Float32, total, hidden);
            float* d = GetFloatPtr(packed);
            long at = 0;
            for (int i = 0; i < rows.Length; i++)
                foreach (int r in rows[i])
                    ReadRow(quant[i], dense[i], r, hidden, d + (at++) * hidden);
            HostWritten(packed);
            return packed;
        }

        /// <summary>The block's small F32 inputs per (layer, rank), as host-backed GGML tensors on
        /// the rank's device; the conv filters stay one per channel, as the kernel reads them.</summary>
        private void BuildDeviceConstants()
        {
            int layers = Config.NumLayers;
            int ranks = Mesh.TensorParallel;
            int qkvDim = 2 * QkDim + VDim;
            _deviceConvW = new Tensor[layers][];
            _deviceDtBias = new Tensor[layers][];
            _deviceA = new Tensor[layers][];
            _deviceSsmNorm = new Tensor[layers];
            for (int l = 0; l < layers; l++)
            {
                if (!_isRecurrent[l] || !OwnsLayer(l)) continue;
                string p = $"blk.{l}.";
                int stage = Mesh.StageOf(l);
                float[] conv = _weights[p + "ssm_conv1d.weight"].GetElementsAsFloat(qkvDim * _convKernel);
                _deviceSsmNorm[l] = Upload(Mesh.Allocator(stage, 0), _ssmNorm[l]);
                _deviceConvW[l] = new Tensor[ranks];
                _deviceDtBias[l] = new Tensor[ranks];
                _deviceA[l] = new Tensor[ranks];
                for (int r = 0; r < ranks; r++)
                {
                    IAllocator alloc = Mesh.Allocator(stage, r);
                    int[] channels = QkvChannelsOf(ShardRankOffset + r, ShardDegree);
                    var w = new float[channels.Length * _convKernel];
                    for (int c = 0; c < channels.Length; c++)
                        Array.Copy(conv, (long)channels[c] * _convKernel, w, (long)c * _convKernel, _convKernel);
                    _deviceConvW[l][r] = Upload(alloc, w);
                    _deviceDtBias[l][r] = Upload(alloc, _dtBias[l][r]);
                    _deviceA[l][r] = Upload(alloc, _ssmA[l][r]);
                }
            }
        }

        private Tensor Upload(IAllocator allocator, float[] values)
        {
            var t = new Tensor(allocator, DType.Float32, values.Length);
            t.SetElementsAsFloat(values);
            HostWritten(t);
            return t;
        }

        /// <summary>A rank's state as tensors on its device: the conv window [channels, kernel-1] (oldest
        /// first per channel) and the delta state [vHeads, Dv, Dk]. Programs bind them as state leaves.
        /// Where the executor binds host bytes (GGML), the tensors are host-backed and the device copy is
        /// authoritative: a reset zeroes the host bytes and drops the device copy; elsewhere they are the
        /// backend's own tensors.</summary>
        protected internal class DeviceGdnState : RecurrentState
        {
            protected readonly Func<Tensor, IntPtr> _hostPointer;
            protected readonly IGraphExecutor _executor;
            protected readonly bool _host;

            public DeviceGdnState(IAllocator allocator, IGraphExecutor executor, Func<Tensor, IntPtr> hostPointer,
                int convRows, int channels, int vHeads, int dv, int dk)
            {
                _hostPointer = hostPointer;
                _executor = executor;
                _host = executor?.Capabilities.BindsHostPointers ?? true;
                _allocator = allocator;
                ConvRows = Math.Max(1, convRows);
                Channels = channels;
                StateFloats = (long)vHeads * dv * dk;
                Conv = new Tensor(allocator, DType.Float32, channels, Math.Max(1, convRows));
                // [one token's outputs | state]: a decode-row graph writes the recurrence's result
                // into Backing, so its state part lands on Ssm in place (no state copy per token).
                RowFloats = (long)vHeads * dv;
                Backing = new Tensor(allocator, DType.Float32, RowFloats + (long)vHeads * dv * dk);
                using (Tensor tail = Backing.Narrow(0, RowFloats, (long)vHeads * dv * dk))
                    Ssm = tail.View(vHeads, dv, dk);
                Reset();
            }

            private readonly IAllocator _allocator;

            public Tensor Conv { get; }
            public Tensor Ssm { get; }
            public int ConvRows { get; }
            public int Channels { get; }
            public long StateFloats { get; }

            /// <summary>A verify window's recurrence result, written there in place: [outputs of its rows |
            /// state after the last row | state 1 row before it | ...], room for SnapshotCapacity rows.</summary>
            public Tensor Verify { get; private set; }
            /// <summary>The conv inputs of a verify window, [channels, ConvRows + capacity]: columns
            /// [k, k + ConvRows) are the window after k of its rows.</summary>
            public Tensor ConvHistory { get; private set; }
            public int SnapshotCapacity { get; private set; }

            /// <summary>Room for a <paramref name="rows"/>-row verify; true when the buffers moved.</summary>
            public bool EnsureSpecBuffers(int rows)
            {
                if (rows <= SnapshotCapacity) return false;
                int capacity = 4;
                while (capacity < rows) capacity *= 2;
                Verify?.Dispose();
                ConvHistory?.Dispose();
                Verify = new Tensor(_allocator, DType.Float32, capacity * (RowFloats + StateFloats));
                ConvHistory = new Tensor(_allocator, DType.Float32, Channels, ConvRows + capacity);
                SnapshotCapacity = capacity;
                return true;
            }
            /// <summary>One token's recurrence outputs followed by the state (Ssm views its tail).</summary>
            public Tensor Backing { get; }
            public long RowFloats { get; }

            public override void Reset()
            {
                Clear(Conv);
                Clear(Backing);
                // The state is also bound through its own view.
                if (_host) Invalidate(Ssm);
            }

            protected internal override IEnumerable<Tensor> HostTensors => new[] { Conv, Ssm, Backing };

            protected internal override bool CanCopy => true;

            protected internal override IEnumerable<Tensor> CheckpointTensors => new[] { Conv, Backing };

            protected internal override unsafe void CopyFrom(RecurrentState source)
            {
                var other = (DeviceGdnState)source;
                foreach (var (from, to) in new[] { (other.Conv, Conv), (other.Backing, Backing) })
                {
                    if (!_host)
                    {
                        Ops.Copy(to, from);
                        continue;
                    }
                    long bytes = to.ElementCount() * sizeof(float);
                    Buffer.MemoryCopy((void*)_hostPointer(from), (void*)_hostPointer(to), bytes, bytes);
                }
                OnHostWritten();
            }

            /// <summary>Every key the state is bound through: the conv window, the backing and the state view.</summary>
            protected internal override void OnHostWritten()
            {
                if (!_host) return;
                foreach (Tensor t in new[] { Conv, Backing, Ssm }) Invalidate(t);
            }

            protected void Invalidate(Tensor t)
            {
                _executor?.Invalidate(_hostPointer(t));
                GgmlBasicOps.InvalidateTensorHostBuffer(t);
            }

            public IntPtr ConvPointer => _hostPointer(Conv);

            /// <summary>Before GGML's per-rank native block binds the state view: whatever the graphs left
            /// in a device copy of the backing is already on the host (SyncToHost); make the view re-upload.</summary>
            public void BeforePerOp() => _executor?.Invalidate(_hostPointer(Ssm));

            /// <summary>After it: bring its state back to the host and make the backing re-upload.</summary>
            public void AfterPerOp()
            {
                _executor?.SyncToHost(_hostPointer(Ssm), Ssm.ElementCount() * sizeof(float));
                _executor?.Invalidate(_hostPointer(Backing));
            }

            public IntPtr SsmPointer => _hostPointer(Ssm);

            protected unsafe void Clear(Tensor t)
            {
                if (!_host)
                {
                    Ops.Fill(t, 0);
                    return;
                }
                IntPtr p = _hostPointer(t);
                new Span<float>((float*)p, checked((int)t.ElementCount())).Clear();
                Invalidate(t);
            }

            public override void Dispose()
            {
                Conv.Dispose();
                Ssm.Dispose();
                Backing.Dispose();
                Verify?.Dispose();
                ConvHistory?.Dispose();
            }
        }

        private Tensor DeviceGatedDeltaNet(RankScope scope, Tensor raw, int layer, DeviceGdnState state)
        {
            string p = $"blk.{layer}.";
            int tokens = (int)raw.Sizes[0];
            int kHeads = _numKHeads / ShardDegree;
            int vHeads = _numVHeads / ShardDegree;
            int qk = kHeads * _headKDim;
            int v = vHeads * _headVDim;
            int qkv = 2 * qk + v;
            int packed = qkv + v + 2 * vHeads;

            // One device keeps in-projections of mixed types unpacked (packing them would mean a
            // dense or requantized copy): the family's graph lowering runs them as quantized.
            // One shard runs the family's mixer program (every backend's executor); GGML's native
            // block serves tensor parallelism. On one device it also keeps in-projections of mixed
            // types unpacked (packing them would mean a dense or requantized copy).
            string name = InProjName(layer);
            if (scope.Degree == 1 && (!IsGgmlBackend || (!_quantWeights.ContainsKey(name) && !_weights.ContainsKey(name))))
                return RunRecurrentMixerGraph(scope, raw, layer, state);
            IntPtr w; int type; long ne0, ne1, bytes;
            QuantizedWeight q = scope.Degree == 1
                ? (_quantWeights.TryGetValue(name, out var single) ? single : null)
                : (_tpQuantWeights.TryGetValue(name, out var shards) ? shards[scope.Rank] : null);
            if (q != null)
            {
                w = q.CacheKey; type = q.GgmlType; ne0 = q.Ne0; ne1 = q.Ne1; bytes = q.RawBytes;
            }
            else
            {
                Tensor d = scope.Degree == 1 ? _weights[name] : _tpWeights[name][scope.Rank];
                unsafe { w = (IntPtr)GetFloatPtr(d); }
                type = 0; ne0 = d.Sizes[1]; ne1 = d.Sizes[0]; bytes = d.ElementCount() * sizeof(float);
            }

            var gated = new Tensor(scope.Allocator, DType.Float32, tokens, v);
            try
            {
                // The graphs may bind the state through the backing buffer; this native block binds the
                // state view. Hand the bytes over through the host in both directions.
                state.BeforePerOp();
                unsafe
                {
                    GgmlBasicOps.Qwen35GdnLayerTP(
                        (IntPtr)GetFloatPtr(raw), Config.HiddenSize, tokens,
                        (IntPtr)GetFloatPtr(Replica(_weights[p + "attn_norm.weight"], scope.Stage, scope.Rank)),
                        w, type, ne0, ne1, bytes,
                        (IntPtr)GetFloatPtr(_deviceConvW[layer][scope.Rank]),
                        (IntPtr)GetFloatPtr(_deviceDtBias[layer][scope.Rank]),
                        (IntPtr)GetFloatPtr(_deviceA[layer][scope.Rank]),
                        (IntPtr)GetFloatPtr(_deviceSsmNorm[layer]),
                        state.ConvPointer,
                        state.SsmPointer,
                        (IntPtr)GetFloatPtr(gated),
                        packed, qkv, qk, v, kHeads, vHeads, _headKDim, _headVDim, _convKernel, Config.Eps);
                }
                state.AfterPerOp();
                HostWritten(gated);
                return Linear(scope, gated, p + "ssm_out.weight");
            }
            finally
            {
                gated.Dispose();
            }
        }
    }
}
