// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using TensorSharp.Cuda.Interop;
using TensorSharp.Graph;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// A prompt chunk's MoE layers with their routes kept on the device: chunks wider than the doorbell takes and shorter
    /// than the staged-ahead path's (CudaGraphExecutor.PromptPrefetch). Such a chunk's layer read its routes back to the
    /// host, which grouped them and chose the experts to stage, then queued the fills, copies and kernels, so the device
    /// waited for the host once a MoE layer. On an A40, Qwen3.8 Flash Next UD-Q2_K_XL's ~150-row prompt (one chunk)
    /// prefilled in 205 ms against the direct engine's 173 ms: the device was busy for less (153.8 against 160.4 ms a step)
    /// but idle for 21.9 ms a step against 0.5, in 47 gaps of about 0.28 ms between a layer's router and its first fill (one
    /// a MoE layer) and 15 of about 0.46 ms between fills.
    /// Here, as the direct engines stage a prompt's experts (CudaMoe.Experts with a stage), a kernel gives each expert the
    /// chunk selects outside the tier a staging slot and points the layer's tables there, the copy kernels bring its bytes
    /// over the link from the model file's page-locked, device-mapped pages, the grouped MoE kernels group the routes and
    /// run over the tables, and a last kernel points the tables back at their resting entries. A counting kernel keeps the
    /// routes for the expert host (the routing profile and the adaptive tiers), handed over at the next step's start. The
    /// products and the weighted sum are the host-planned path's kernels on the same bytes, so the answers are the same.
    /// The slots are the shorter chunks' arena where it holds every expert the layer could stage, else the long chunks' two
    /// halves as one (idle while a shorter chunk runs). The shorter chunks' arena is sized at load from the first MoE layer's
    /// experts, and Qwen3.8 Flash Next UD-Q2_K_XL mixes expert layouts across its layers: on an A40 its 299 MiB held fewer
    /// than the host experts (up to 168) of most layers, and 36 of a short prompt's 48 MoE layers still read their routes
    /// back; the halves (374 MiB each, a quarter over the most host experts a layer has) hold any layer's.
    /// </summary>
    public sealed partial class CudaGraphExecutor
    {
        /// <summary>Prompt-chunk MoE layers whose routes stayed on the device (<see cref="TryDeviceStagedExperts"/>), and
        /// layers whose routes came back to the host to plan the staging (<see cref="TryStagedExperts"/>'s own path); tests
        /// check the route.</summary>
        internal long DeviceStagedLayers { get; private set; }

        /// <inheritdoc cref="DeviceStagedLayers"/>
        internal long HostPlannedLayers { get; private set; }

        /// <summary>Of <see cref="DeviceStagedLayers"/>, the layers staged through the long chunks' arena halves (the shorter
        /// chunks' arena too small for the layer's host experts).</summary>
        internal long DeviceStagedInHalves { get; private set; }

        /// <summary>Tests: prompt chunks take the host-planned staging (the routes read back) instead of the device's.</summary>
        internal bool HostPlannedStagingForTests { get; set; }

        /// <summary>Tests: the most experts the shorter chunks' staging arena holds (0: as many as the device's room allows),
        /// set before the first step, as a device with less free memory would size it.</summary>
        internal int ShortChunkArenaSlotsForTests { get; set; }

        /// <summary>Tests: the long chunks' arena is never allocated (as on a device without room for its two halves), set
        /// before the first step.</summary>
        internal bool PrefetchArenaRefusedForTests { get; set; }

        /// <summary>The device plan's buffers (<see cref="CudaExpertStage"/>): StageOf [experts] (-1 between calls),
        /// StageList [experts], StageCount and StageFrom.</summary>
        private Tensor _promptStagePlan;
        private int _promptStageExperts;
        private readonly CudaExpertStage _promptStage = new();

        /// <summary>
        /// A prompt chunk's routed experts with the routes on the device: the tier's experts read from their resident copies,
        /// the others planned into staging slots and copied there by kernels from their mapped host pages, the grouped MoE
        /// kernels over the layer's tables, and the routes counted for the expert host. False, with nothing launched, where
        /// the device cannot serve the layer: a host expert whose pages are not mapped into the device's address space, no
        /// arena with a slot for every expert the chunk could select outside the tier (at most its routes, at most the
        /// layer's host experts: the plan cannot run out of slots), or a layout the grouped kernels do not take.
        /// </summary>
        private bool TryDeviceStagedExperts(Tensor x, Value selected, Value weightValue, IGraphHostedExperts gate, IGraphExpertWeights up,
            IGraphExpertWeights down, int layer, int rows, int k, out Tensor result)
        {
            result = null;
            var cuda = (CudaAllocator)Allocator;
            Tensor sel = selected.Tensor, w = weightValue.Tensor;
            if (HostPlannedStagingForTests || Kernels == null || CudaGraphCapture.IsCapturing(cuda) || x.Storage is not CudaStorage
                || x.ElementType != DType.Float32 || !x.IsContiguous() || sel?.Storage is not CudaStorage || w?.Storage is not CudaStorage
                || sel.ElementType != DType.Int32 || w.ElementType != DType.Float32 || !sel.IsContiguous() || !w.IsContiguous()
                || w.ElementCount() != (long)rows * k || gate.Format != up.Format)
                return false;
            if (MoeKernels is not { HasExpertTables: true })
                return false;
            int hidden = (int)gate.Ne0, ff = (int)gate.Ne1, experts = (int)gate.Ne2;
            HostedTables t = Tables(gate, up, down, layer);
            if (t.Sets == null || t.ExpertBytes == null || t.IsHost == null || !GroupedTakes(t.Sets, rows, hidden, ff))
                return false;
            int host = MappedHostExperts(t);
            if (host < 0)
                return false;
            int need = (int)Math.Min((long)rows * k, host);
            long stride = TradeStride(t.ExpertBytes);
            IntPtr slots = IntPtr.Zero;
            bool halves = false;
            if (need > 0)
            {
                // An adaptive round borrows both arenas for its copies; one in flight finishes first (HostExperts already
                // flushed it, and a round begins only at a decode-size step, its copies after the work queued before it).
                FlushTrades();
                (slots, halves) = PromptStageSlots(stride, need);
                if (slots == IntPtr.Zero)
                    return false;
                EnsurePromptStagePlan(experts);
            }

            CudaExpertStage stage = null;
            CudaExpertStageLayer stageLayer = default;
            if (need > 0)
            {
                foreach (Tensor table in new[] { t.IsHost, t.Gate, t.Up, t.Down, t.RestGate, t.RestUp, t.RestDown, t.HomeGate, t.HomeUp, t.HomeDown })
                    ((CudaStorage)table.Storage).EnsureDeviceCurrent();
                _promptStage.Staging = slots;
                _promptStage.Stride = stride;
                _promptStage.Capacity = need;
                stage = _promptStage;
                long offUp = Align256(t.ExpertBytes[0]);
                stageLayer = new CudaExpertStageLayer
                {
                    IsHost = Ptr(t.IsHost),
                    TabGate = Ptr(t.Gate), TabUp = Ptr(t.Up), TabDown = Ptr(t.Down),
                    // The copies read the experts' own entries (their mapped host pages); the tables go back to their resting
                    // entries (the zeroed expert where the layer has one) once the kernels ran.
                    SrcTabGate = Ptr(t.RestGate), SrcTabUp = Ptr(t.RestUp), SrcTabDown = Ptr(t.RestDown),
                    RestTabGate = Ptr(t.HomeGate), RestTabUp = Ptr(t.HomeUp), RestTabDown = Ptr(t.HomeDown),
                    BytesGate = t.ExpertBytes[0], BytesUp = t.ExpertBytes[1], BytesDown = t.ExpertBytes[2],
                    OffUp = offUp, OffDown = offUp + Align256(t.ExpertBytes[1]), Staged = true,
                };
            }
            using Tensor x2 = x.View(rows, hidden);
            var output = new Tensor(Allocator, DType.Float32, rows, hidden);
            Ops.Fill(output, 0f);
            Tensor[] tables = { t.Gate, t.Up, t.Down };
            if (halves)
                HalvesTaken();
            bool ran = TryGroupedMoe(x2, output, sel, w, t.Sets, tables, rows, k, hidden, ff, experts, stage, stageLayer);
            if (halves)
                HalvesReleased();
            if (!ran)
            {
                output.Dispose();
                return false;
            }
            if (stage != null)
                foreach (Tensor table in tables) ((CudaStorage)table.Storage).MarkDeviceModified();
            CountPromptRoutes(t, sel, rows, k);
            DeviceStagedLayers++;
            if (halves) DeviceStagedInHalves++;
            result = output;
            return true;
        }

        /// <summary>
        /// Staging slots of <paramref name="stride"/> bytes for <paramref name="need"/> experts: the shorter chunks' arena where
        /// it holds them, else the long chunks' two halves as one region (Halves true), else the shorter chunks' arena grown
        /// where the device's memory allows; zero where none holds them.
        /// </summary>
        private (IntPtr Slots, bool Halves) PromptStageSlots(long stride, int need)
        {
            long bytes = need * stride;
            if (_stagingArena != IntPtr.Zero && _stagingBytes >= bytes) return (_stagingArena, false);
            if (_prefetchArena != IntPtr.Zero && 2 * _halfBytes >= bytes) return (_prefetchArena, true);
            EnsureStagingArena(stride, need);
            return _stagingArena != IntPtr.Zero && _stagingBytes >= bytes ? (_stagingArena, false) : (IntPtr.Zero, false);
        }

        /// <summary>The halves taken for a shorter chunk's staging: the step's stream first waits on the device for copies into
        /// them still in flight on the long chunks' copy stream (a layer staged ahead that never took them).</summary>
        private void HalvesTaken()
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            for (int h = 0; h < 2; h++)
                if (_halfReady[h] != IntPtr.Zero) CudaDriverApi.cuStreamWaitEvent(cuda.Stream.Handle, _halfReady[h], 0).ThrowOnError();
        }

        /// <summary>The halves given back once the layer's kernels are queued: a long chunk's copies into them wait for those
        /// kernels (StageLayer waits on the halves' free events), and no layer is staged ahead in them any more.</summary>
        private void HalvesReleased()
        {
            var cuda = (CudaAllocator)Allocator;
            cuda.Context.MakeCurrent();
            for (int h = 0; h < 2; h++)
                if (_halfFree[h] != IntPtr.Zero) CudaDriverApi.cuEventRecord(_halfFree[h], cuda.Stream.Handle).ThrowOnError();
            _halfLayer[0] = _halfLayer[1] = -1;
        }

        /// <summary>
        /// The layer's experts outside its tier, or -1 where one of them cannot be read by the copy kernels where it lies (its
        /// bytes not all page-locked and mapped into the device's address space: <see cref="MapHostExpert"/>). Experts whose
        /// pages the load locked since the tables were built are mapped first, once a change of the locked total.
        /// </summary>
        private int MappedHostExperts(HostedTables t)
        {
            int host = 0, unmapped = 0;
            foreach (byte m in t.Mask)
            {
                if (m != 1) host++;
                if (m == 0) unmapped++;
            }
            long pins = CudaHostPins.PinnedBytes;
            if (unmapped > 0 && t.MappedAtPins != pins)
            {
                t.MappedAtPins = pins;
                bool changed = false;
                for (int e = 0; e < t.Experts; e++)
                {
                    if (t.Mask[e] != 0) continue;
                    MapHostExpert(t.Sets, t.Copies, e, t.Words, t.Mask);
                    if (t.Mask[e] == 0) continue;
                    changed = true;
                    unmapped--;
                }
                if (changed) UploadTables(t);
            }
            return unmapped == 0 ? host : -1;
        }

        /// <summary>The device plan's buffers for layers of up to <paramref name="experts"/> experts (StageOf at -1, the count
        /// at 0: the plan's unstage leaves them so after every call).</summary>
        private void EnsurePromptStagePlan(int experts)
        {
            if (_promptStagePlan != null && _promptStageExperts >= experts) return;
            // A smaller plan's last readers are queued on the stream: its memory returns to the pool for later work there.
            _promptStagePlan?.Dispose();
            var init = new int[2 * experts + 2];
            Array.Fill(init, -1, 0, experts);
            var plan = new Tensor(Allocator, DType.Int32, init.Length);
            plan.SetElementsAsInt(init);
            ((CudaStorage)plan.Storage).EnsureDeviceCurrent();
            _promptStagePlan = plan;
            _promptStageExperts = experts;
            IntPtr p = Ptr(plan);
            _promptStage.StageOf = p;
            _promptStage.StageList = p + 4 * experts;
            _promptStage.StageCount = p + 8 * experts;
            _promptStage.StageFrom = p + 8 * experts + 4;
        }
    }
}
