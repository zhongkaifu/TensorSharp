// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Opt-in Q8_1 projections for verify/decode rows above the native GEMV row limit.
// Mixed steps retain the original all-row A4 projections before replacing only
// verify/decode rows. Prefill-only steps and the default policy are unchanged.
using System;
using System.Collections.Generic;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        private readonly bool _verifyDecodeQ81 = Environment.GetEnvironmentVariable("TS_Q35_VERIFY_DECODE_Q81") == "1";
        public bool VerifyDecodeQ81Enabled => _verifyDecodeQ81;
        // Projection policy changes recurrent/KV state. Derive this from the engine's
        // immutable option, rather than rereading an environment variable at reuse.
        public string KVStateNamespaceSuffix => (_verifyDecodeQ81 ? "|verify-decode=q8_1-v1" : string.Empty) + OutputW8NamespaceSuffix + (_m.Ninfer == null ? string.Empty : "|body=ninfer-v2-bf16-kv-calibrated-v1:" + _ninfer.ImageSha256 + ":" + _ninferAttention.ImageSha256) + NinferA16NamespaceSuffix + NinferDecodeGraphsNamespaceSuffix + NinferDraftNamespaceSuffix + NinferHostEmbeddingNamespaceSuffix + NinferExactVerifyNamespaceSuffix + NinferFlashInferNamespaceSuffix + (_ninferAttention?.PrefillNamespaceSuffix ?? string.Empty) + NinferVerify8GemmNamespaceSuffix + NinferResidualNormNamespaceSuffix;
        public long ForcedQ81ProjectionCalls { get; private set; }
        public long ForcedQ81ProjectionRows { get; private set; }
        private readonly bool _projectionTrace = Environment.GetEnvironmentVariable("TS_Q35_PROJECTION_TRACE") == "1";
        private Dictionary<(string Role, string DType, int FirstRow, int Rows), long> _projectionDispatches;
        public readonly record struct ProjectionDispatchStat(string Role, string DType, int FirstRow, int Rows, long Calls);
        public ProjectionDispatchStat[] ProjectionDispatchStats
        {
            get
            {
                if (_projectionDispatches == null || _projectionDispatches.Count == 0)
                    return Array.Empty<ProjectionDispatchStat>();
                var result = new ProjectionDispatchStat[_projectionDispatches.Count];
                int i = 0;
                foreach (var entry in _projectionDispatches)
                    result[i++] = new(entry.Key.Role, entry.Key.DType, entry.Key.FirstRow, entry.Key.Rows, entry.Value);
                return result;
            }
        }
        internal void ResetProjectionDispatchStats() => _projectionDispatches?.Clear();
        internal void RecordProjectionDispatch(string role, string dtype, int firstRow, int rows)
        {
            if (!_projectionTrace) return;
            _projectionDispatches ??= new();
            var key = (role, dtype, firstRow, rows);
            _projectionDispatches.TryGetValue(key, out long count);
            _projectionDispatches[key] = checked(count + 1);
        }
        internal void RecordForcedQ81Projection(string role, int firstRow, int rows)
        {
            if (!_projectionTrace) return;
            RecordProjectionDispatch(role, "NVFP4_W4A8_Q8_1", firstRow, rows);
            ForcedQ81ProjectionCalls++;
            ForcedQ81ProjectionRows += rows;
        }
        private void ProjectObserved(string role, in Planes w, IntPtr input, IntPtr output, int rows)
        {
            Project(w, input, output, rows);
            RecordProjectionDispatch(role, rows <= Q35Kernels.GemvMaxRows ? "NVFP4_W4A8_Q8_1" : "NVFP4_W4A4", 0, rows);
        }
        private void GemmObserved(string role, in Planes w, IntPtr output, int rows)
        {
            GemmQuantized(w, output, rows);
            RecordProjectionDispatch(role, "NVFP4_W4A4", 0, rows);
        }

        internal readonly record struct Q81RowRange(int Start, int Rows);
        internal sealed record Q81RowPlan(int TotalRows, int ForcedRows, Q81RowRange[] ForcedRanges)
        {
            public bool AllForced => TotalRows > 0 && ForcedRows == TotalRows;
            public bool NeedsOverride => TotalRows > Q35Kernels.GemvMaxRows && ForcedRows > 0;
        }
        internal enum Q81GateStage { ProjectQ81ToProj, OriginalA4FusedToAct, SwiGluQ81ToAct }
        private static readonly Q81GateStage[] AllQ81GateStages = { Q81GateStage.ProjectQ81ToProj, Q81GateStage.SwiGluQ81ToAct };
        private static readonly Q81GateStage[] MixedQ81GateStages = { Q81GateStage.ProjectQ81ToProj, Q81GateStage.OriginalA4FusedToAct, Q81GateStage.SwiGluQ81ToAct };

        internal static IReadOnlyList<Q81GateStage> Q81GateStages(bool allForced) => allForced ? AllQ81GateStages : MixedQ81GateStages;

        // This policy has no graph capture/replay contract; reject the combination
        // before creating a CUDA context, including when diagnostic tracing is on.
        internal static void ValidateVerifyDecodeQ81Graphs(bool enabled, bool verifyGraphs, bool draftGraphs)
        {
            if (enabled && (verifyGraphs || draftGraphs))
                throw new NotSupportedException("Q81 verify/decode projections and projection tracing require both CUDA graph opt-ins OFF.");
        }

        internal static Q81RowPlan PlanVerifyDecodeQ81(int[] rows, bool[] force)
        {
            if (rows == null || force == null || rows.Length != force.Length)
                throw new ArgumentException("One force flag is required per row group.");
            int total = 0, forced = 0;
            var ranges = new List<Q81RowRange>();
            for (int i = 0; i < rows.Length; i++)
            {
                if (rows[i] < 0) throw new ArgumentOutOfRangeException(nameof(rows));
                if (force[i] && rows[i] > 0)
                {
                    ranges.Add(new Q81RowRange(total, rows[i]));
                    forced = checked(forced + rows[i]);
                }
                total = checked(total + rows[i]);
            }
            return new Q81RowPlan(total, forced, ranges.ToArray());
        }

        internal static Q81RowRange[] Q81Chunks(Q81RowRange range)
        {
            if (range.Start < 0 || range.Rows < 0) throw new ArgumentOutOfRangeException(nameof(range));
            _ = checked(range.Start + range.Rows);
            var chunks = new List<Q81RowRange>();
            for (int done = 0; done < range.Rows;)
            {
                int rows = Math.Min(Q35Kernels.GemvMaxRows, range.Rows - done);
                chunks.Add(new Q81RowRange(checked(range.Start + done), rows));
                done = checked(done + rows);
            }
            return chunks.ToArray();
        }

        internal static long ProjectionRowByteOffset(int row, int width)
        {
            if (row < 0 || width < 0) throw new ArgumentOutOfRangeException();
            long bytes = checked((long)row * width * sizeof(float));
            _ = checked((nint)bytes);
            return bytes;
        }

        private static IntPtr ProjectionRow(IntPtr pointer, int row, int width)
            => (IntPtr)checked(pointer.ToInt64() + ProjectionRowByteOffset(row, width));

        private void ProjectQ81Ranges(string role, in Planes w, IntPtr input, IntPtr output, Q81RowPlan plan)
        {
            foreach (var range in plan.ForcedRanges)
                for (int done = 0; done < range.Rows;)
                {
                    int rows = Math.Min(Q35Kernels.GemvMaxRows, range.Rows - done);
                    int firstRow = checked(range.Start + done);
                    // Reuse the original <=8 implementation, including its Q8_1 quantizer,
                    // weight planes, arithmetic, alpha and output layout. Quant buffers are
                    // independent of the shared phase slab and reused on the same stream.
                    Project(w, ProjectionRow(input, firstRow, w.K), ProjectionRow(output, firstRow, w.Rows), rows);
                    RecordForcedQ81Projection(role, firstRow, rows);
                    done = checked(done + rows);
                }
        }

        private void RenormalizeForcedRows(IntPtr weights, Q81RowPlan plan)
        {
            foreach (var range in plan.ForcedRanges)
                // The original fused norm already committed x += pending. Do not add it
                // twice. Both native norms use the same 512-thread per-row reduction.
                _k.AddRmsNorm(ProjectionRow(_x, range.Start, _m.NEmbd), IntPtr.Zero, weights,
                    ProjectionRow(_xn, range.Start, _m.NEmbd), range.Rows, _m.NEmbd, _m.Eps, Stream);
        }

        private bool TryRunVerifyDecodeQ81(int n, int tapFirst, RowGroup[] groups, int[] maxKeys)
        {
            if (!_verifyDecodeQ81 || n <= Q35Kernels.GemvMaxRows) return false;
            var rows = new int[groups.Length];
            var force = new bool[groups.Length];
            for (int i = 0; i < groups.Length; i++)
            {
                rows[i] = groups[i].Rows;
                force[i] = groups[i].Verify || groups[i].Decode;
            }
            var plan = PlanVerifyDecodeQ81(rows, force);
            if (plan.TotalRows != n) throw new InvalidOperationException("Q81 projection range mismatch.");
            if (!plan.NeedsOverride) return false;
            RunVerifyDecodeQ81Layers(n, tapFirst, groups, maxKeys, plan);
            return true;
        }

        private void RunVerifyDecodeQ81Layers(int n, int tapFirst, RowGroup[] groups, int[] maxKeys, Q81RowPlan plan)
        {
            int e = _m.NEmbd;
            bool keepOriginalA4 = !plan.AllForced;
            _servingTrace?.LayerBranch(keepOriginalA4 ? "native_mixed_W4A4_plus_Q81_overwrite" : "native_forced_Q81_chunks_le8");
            IntPtr pending = IntPtr.Zero;
            for (int il = 0; il < _m.NLayer; il++)
            {
                var L = _layers[il];
                if (keepOriginalA4)
                {
                    // Preserve the original all-n A4 geometry, BN, split-K and epilogue
                    // for prefill. Mixed verify rows are then overwritten with Q8_1.
                    _k.AddRmsNormNvfp4(_x, pending, L.AttnNorm, _m.Eps, _fp4c, _fp4s, _fp4r, e, n, Stream);
                    RenormalizeForcedRows(L.AttnNorm, plan);
                }
                else
                    _k.AddRmsNorm(_x, pending, L.AttnNorm, _xn, n, e, _m.Eps, Stream);
                if (_featureSlot != null && _featureSlot[il] >= 0)
                    for (int gi = 0, r0 = 0; gi < groups.Length; r0 += groups[gi].Rows, gi++)
                        if (groups[gi].Verify)
                            _k.CopyRows(ProjectionRow(_x, r0, e), e, _features + (nint)((long)_featureSlot[il] * e * 4), FeatureWidth, e, groups[gi].Rows, Stream);
                if (_d != null) DraftTap(il, tapFirst, n);
                if (keepOriginalA4) GemmObserved("InProj", L.InProj, _proj, n);
                ProjectQ81Ranges("InProj", L.InProj, _xn, _proj, plan);
                for (int gi = 0, r0 = 0, q0 = 0; gi < groups.Length; r0 += groups[gi].Rows, q0 += groups[gi].Slots.Length, gi++)
                {
                    if (L.Recurrent) GdnCore(L, groups[gi], r0, q0);
                    else AttnCore(L, groups[gi], r0, q0, maxKeys[gi]);
                }
                IntPtr outInput;
                if (L.Recurrent)
                {
                    _k.GdnOut(_gdnO, _proj, GdnPackedWidth, ConvDim, n, _m.GdnVHeads, L.SsmNorm, _m.Eps, _gdnY, Stream);
                    outInput = _gdnY;
                }
                else outInput = _attn;
                if (keepOriginalA4) ProjectObserved("OutProj", L.OutProj, outInput, _block, n);
                ProjectQ81Ranges("OutProj", L.OutProj, outInput, _block, plan);
                // All out-projection reads finish before _xn can alias/overwrite any
                // attention or GDN phase input. Commands remain on the original stream.
                if (keepOriginalA4)
                {
                    _k.AddRmsNormNvfp4(_x, _block, L.PostNorm, _m.Eps, _fp4c, _fp4s, _fp4r, e, n, Stream);
                    RenormalizeForcedRows(L.PostNorm, plan);
                }
                else
                    _k.AddRmsNorm(_x, _block, L.PostNorm, _xn, n, e, _m.Eps, Stream);
                foreach (var stage in Q81GateStages(plan.AllForced))
                {
                    if (stage == Q81GateStage.ProjectQ81ToProj)
                        // Consume every _xn slice before any _act write: shared scratch
                        // allows _xn and _act to overlap, including across different rows.
                        ProjectQ81Ranges("GateUp", L.GateUp, _xn, _proj, plan);
                    else if (stage == Q81GateStage.OriginalA4FusedToAct)
                    {
                        _k.GemmW4A4(L.GateUp.Codes, L.GateUp.Scales, L.GateUp.Rows, L.GateUp.K, _fp4c, _fp4s, _fp4r, n, 1.0f, _act,
                            _partial, _counters, _sms, 1, Stream, swiglu: true);
                        RecordProjectionDispatch("GateUp", "NVFP4_W4A4", 0, n);
                    }
                    else
                        foreach (var range in plan.ForcedRanges)
                            _k.SwiGlu(ProjectionRow(_proj, range.Start, L.GateUp.Rows), ProjectionRow(_act, range.Start, _m.NFf), _m.NFf, range.Rows, Stream, interleaved: true);
                }
                if (keepOriginalA4)
                {
                    _k.QuantizeNvfp4(_act, _fp4c, _fp4s, _fp4r, _m.NFf, n, Stream);
                    GemmObserved("Down", L.Down, _block, n);
                }
                ProjectQ81Ranges("Down", L.Down, _act, _block, plan);
                pending = _block;
            }
        }
    }
}
