// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp.Graph;

namespace TensorSharp.Models
{
    public abstract partial class StandardDecoderModel
    {
        /// <summary>Tests: a long prompt over local whole-layer stages runs pipelined on the tensor executors too (their
        /// plans run in lockstep: one after another on the managed executor), so the wavefronts' bits can be held to the
        /// stages' one chunk after another.</summary>
        internal bool PipelinedPromptForTests { get; set; }

        /// <summary>Tests: prompts over local whole-layer stages run one chunk after another even where they would pipeline.</summary>
        internal bool PipelinedPromptDisabledForTests { get; set; }

        /// <summary><c>TS_PIPELINED_PROMPT=0</c>: prompts over whole-layer stages run one chunk after another (an A/B switch).</summary>
        private static readonly bool PipelinedPromptEnabled = Environment.GetEnvironmentVariable("TS_PIPELINED_PROMPT") != "0";

        private bool _pipelinedNoted;

        /// <summary>Wavefronts the pipelined prompts ran (a test seam: 0 when every prompt ran chunk after chunk).</summary>
        internal int PipelinedWavefronts { get; private set; }

        /// <summary>
        /// Whether the plain prompt rows [0, <paramref name="count"/>) of a refill run pipelined over the whole-layer stages
        /// (<see cref="TryRunPipelinedPrompt"/>): local stages on one executor that runs several devices' plans together (the
        /// GGML executors: their staged chunks otherwise run one device at a time, each run waiting for every device), a plain
        /// prompt (no media, no position table, no draft or DFlash ring following it, no step adapter), no experts on the host
        /// (a ranked run takes no host-expert pause), and two chunks at least. Direct CUDA's stages pipeline too (its tensor
        /// executor's runs return with their device work queued).
        /// </summary>
        private bool PipelinesPrompt(int count)
        {
            if (!PipelinedPromptEnabled || PipelinedPromptDisabledForTests || _mesh == null || _mesh.Stages < 2 || _rankGraphs || _mesh.TensorParallel > 1) return false;
            if (LayerSplitGroup?.NodeCount > 1 || HasPendingMedia || _pendingPositionTable != null || _followPlainPrompt) return false;
            if (StepAdapter != null || AnyExpertLayerOffloaded()) return false;
            if (count <= PipelinedChunkTokens || PipelinedChunkTokens > _maxSequenceStepRows) return false;
            if (GraphExecutor is not { } executor || GraphUnsupportedReason() != null) return false;
            bool ggml = executor is IRankedGraphExecutor && _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
            bool cuda = _backend == BackendType.Cuda && executor is TensorGraphExecutor;
            return ggml || cuda || (PipelinedPromptForTests && executor is TensorGraphExecutor);
        }

        /// <summary>Whether any layer's routed experts run on the host (a stage's graph then pauses for them).</summary>
        private bool AnyExpertLayerOffloaded()
        {
            if (!_anyMoe) return false;
            for (int l = 0; l < TotalLayers; l++)
                if (_moeLayer[l] && IsExpertLayerOffloaded(l)) return true;
            return false;
        }

        /// <summary>
        /// The prompt rows <paramref name="tokens"/>[0, <paramref name="count"/>) of the active sequence over the whole-layer
        /// stages as a pipeline of chunks: wavefront t runs stage s on chunk t - s, every stage of a wavefront submitted together
        /// (<see cref="RunWavefront"/>), so stage s + 1 works on chunk k while stage s works on chunk k + 1. Each stage's step is the one the chunk's sequential step emits
        /// (its layers, caches and states, the chunk's tokens, positions and start): a stage reads only its own layers' caches,
        /// and the rows it hands on are the previous wavefront's output of the stage before it, so the result is the chunks'
        /// one after another, bit for bit. Two stages take max(stage) a wavefront where one chunk after another took their sum
        /// (Qwen3.5-9B Q8_0 on two RTX 2000 Ada: an 8K refill at 1,893 tok/s over two GGML stages against 1,924 on one GPU,
        /// 2,911 pipelined in 2,048-row chunks and 3,348 in 512-row ones; direct CUDA's refill chunks, whose runs return with
        /// their device work queued, had overlapped across chunks: 2,794).
        /// </summary>
        private void TryRunPipelinedPrompt(int[] tokens, int count)
        {
            SettlePendingVerify();
            int baseStart = _cacheSeqLen;
            EnsureCacheCapacity(checked(baseStart + count));
            RunPipelinedChunks(tokens, count, baseStart, bound: null);
            _cacheDeviceDirty = true;
            _cacheSeqLen = baseStart + count;
        }

        /// <summary>Lone prompt steps of a request's slot that ran pipelined (<see cref="TryForwardSlotPromptPipelined"/>; a test seam).</summary>
        internal int PipelinedSlotPrompts { get; private set; }

        /// <summary>
        /// A lone request's plain prompt step over local whole-layer stages (the engine's solo prefill: the prompt, up to
        /// SchedulerConfig.SoloPrefillChunkSize tokens, as one step) as <see cref="RunPipelinedChunks"/>'s wavefronts of
        /// <see cref="PipelinedChunkTokens"/> rows, its last row then a one-row step for the logits: the bits of those chunks one
        /// after another. As one step, each stage had waited for the whole step to pass the stage before it (Qwen3.5-9B Q8_0
        /// over two GGML stages: a 5,131-token chat prompt, a 4,096-row and a 1,035-row step, in 3.07 s, against 1.86 s in
        /// 256-row chunks and 3.04 s on one GPU). False (nothing run) where the step does not pipeline (<see cref="PipelinesPrompt"/>, rows
        /// handed in or out, drafter features, a prompt a draft layer or DFlash ring follows).
        /// </summary>
        private bool TryForwardSlotPromptPipelined(IReadOnlyList<string> requestIds, IReadOnlyList<int[]> chunks, int[] starts, float[][] outLogits,
            float[][] features, SlotRowIO rowIO)
        {
            if (requestIds.Count != 1 || chunks.Count != 1 || starts.Length != 1 || outLogits.Length != 1 || rowIO != null) return false;
            if (features != null && Array.Exists(features, f => f != null)) return false;
            if (chunks[0] is not { Length: > 1 } chunk || !PipelinesPrompt(chunk.Length - 1)) return false;
            if (!SupportsPerSequenceFusedForward || _slots == null || _followPromptRequests?.ContainsKey(requestIds[0]) == true) return false;
            RestorePrimaryCache();
            if (!_slots.TryGetValue(requestIds[0], out Slot slot) || slot.K == null || slot.Length != starts[0]) return false;
            int start = starts[0], count = chunk.Length - 1, required = checked(start + chunk.Length);
            EnsureSlotCapacity(slot, required, wanted: WantedCells(requestIds[0], required));
            RunPipelinedChunks(chunk, count, start, slot);
            slot.DeviceDirty = true;
            slot.Length = start + count;
            PipelinedSlotPrompts++;
            // The slot continues at the last row's position, so the one-row step passes every check the whole step did.
            if (!TryForwardSlots(requestIds, new[] { new[] { chunk[^1] } }, new[] { start + count }, outLogits, features, rowIO))
                throw new InvalidOperationException($"{Config.Architecture}: the pipelined prompt's last row declined after its chunks ran.");
            return true;
        }

        /// <summary>
        /// <see cref="TryRunPipelinedPrompt"/>'s wavefronts over rows <paramref name="tokens"/>[0, <paramref name="count"/>) at
        /// positions from <paramref name="baseStart"/>: of the active fields (<paramref name="bound"/> null: their rotary delta
        /// settled chunk by chunk, BeginRotaryPositions) or of a request's slot (its own delta), capacity ensured by the caller,
        /// which sets the length and the device-written flag once every wavefront ran.
        /// </summary>
        private void RunPipelinedChunks(int[] tokens, int count, int baseStart, Slot bound)
        {
            int chunk = PipelinedChunkTokens, chunks = (count + chunk - 1) / chunk, stages = _mesh.Stages;
            if (!_pipelinedNoted)
            {
                _pipelinedNoted = true;
                Console.WriteLine($"  {Config.Architecture}: long prompts pipeline over the {stages} whole-layer stages in {chunk}-row chunks (TS_PIPELINED_CHUNK; TS_PIPELINED_PROMPT=0 runs them one after another).");
            }
            PrepareHeadLayout();
            MarkDrafterWeights();
            RoomForRecordings();
            // One sequence's prompt, no drafter: the run context of every stage's steps, restored when the prompt ends.
            GraphRunScope runScope = GraphExecutor.EnterRun(new GraphRunContext(Drafting: false, StepSequences: 1));
            // The rows (and the family's carry) each stage handed on in the last wavefront, for the stage after it.
            var handed = new float[stages][];
            var handedCarry = new float[stages][];
            _forwardSw.Start();
            try
            {
                for (int t = 0; t < chunks + stages - 1; t++)
                {
                    int first = Math.Max(0, t - chunks + 1), last = Math.Min(stages - 1, t);
                    int n = last - first + 1;
                    var steps = new GraphStep[n];
                    var plans = new CachedGraph[n];
                    var outputs = new StepOutputs[n];
                    var inputs = new Tensor[n];
                    var carries = new Tensor[n];
                    var starts = new int[n][];
                    var counts = new int[n][];
                    try
                    {
                        for (int i = 0; i < n; i++)
                        {
                            int s = first + i, k = t - s, from0 = k * chunk, rows = Math.Min(chunk, count - from0), start = baseStart + from0;
                            int[] chunkTokens = tokens.AsSpan(from0, rows).ToArray();
                            EnterStage(s);
                            if (s == 0)
                            {
                                if (bound == null) BeginRotaryPositions(start, rows);
                                long t0 = Stopwatch.GetTimestamp();
                                inputs[i] = EmbedStep(chunkTokens);
                                _embTicks += Stopwatch.GetTimestamp() - t0;
                            }
                            else
                            {
                                StepTokens = chunkTokens;
                                inputs[i] = CreateFloatTensor(handed[s - 1], rows, StreamWidth);
                                if (handedCarry[s - 1] != null) carries[i] = CreateFloatTensor(handedCarry[s - 1], rows, StageCarryWidth);
                            }
                            Slot slot = bound ?? SnapshotActiveFields();
                            var slots = new[] { slot };
                            starts[i] = new[] { start };
                            counts[i] = new[] { rows };
                            var request = new GraphRequest { LastRowLogits = false, PromptChunks = MarksPromptChunks(true, counts[i]) };
                            PrepareWrites(slots, starts[i]);
                            (int from, int to) = StageLayers(s);
                            steps[i] = new GraphStep(this, rows, slots, new[] { 0 }, counts[i], starts[i], false, false, record: true);
                            outputs[i] = EmitStep(steps[i], inputs[i], request, from, to, s == stages - 1, carries[i]);
                            plans[i] = steps[i].Build(replayable: false, reorder: false);
                            GraphBuilds++;
                        }
                        RunWavefront(steps, plans, first);
                        GraphRuns += n;
                        PipelinedWavefronts++;
                        for (int i = 0; i < n; i++)
                        {
                            int s = first + i;
                            EnterStage(s);
                            CountDeviceRoutes(plans[i], outputs[i]);
                            HandOverDeviceRoutes(plans[i], outputs[i], starts[i], counts[i]);
                            if (s == stages - 1) continue;
                            handed[s] = (float[])plans[i].Outputs[outputs[i].Hidden].Clone();
                            handedCarry[s] = outputs[i].CarryAt > 0 ? (float[])plans[i].Outputs[outputs[i].CarryAt - 1].Clone() : null;
                        }
                    }
                    finally
                    {
                        for (int i = 0; i < n; i++)
                        {
                            steps[i]?.Dispose();
                            plans[i]?.Dispose();
                            inputs[i]?.Dispose();
                            carries[i]?.Dispose();
                        }
                    }
                }
            }
            finally
            {
                _forwardSw.Stop();
                runScope.Dispose();
                EnterStage(0);
            }
        }

        /// <summary>One wavefront's plans, the stages first .. first + n - 1: the GGML executor queues every device's graph
        /// before it waits for any (<see cref="IRankedGraphExecutor.RunRanked"/>); the tensor executors run them one after
        /// another, the last stage first: a direct-CUDA run returns with its device work queued unless it reads rows back, which
        /// every stage but the last does for the stage after it, so each earlier stage runs while the later ones' work is
        /// queued (the first stage first, the second had started only once the first's rows were back).</summary>
        private void RunWavefront(GraphStep[] steps, CachedGraph[] plans, int first)
        {
            string error = null;
            bool ok = true;
            if (plans.Length == 1 || GraphExecutor is not IRankedGraphExecutor ranked || GraphExecutor is TensorGraphExecutor)
            {
                for (int i = plans.Length - 1; i >= 0 && ok; i--)
                {
                    EnterStage(first + i);
                    ok = steps[i].Run(plans[i].Plan);
                    if (!ok) error = $"stage {first + i}: {plans[i].Plan.LastError}";
                }
            }
            else
            {
                var runInputs = Array.ConvertAll(steps, step => step.RunInputs());
                var compiled = Array.ConvertAll(plans, p => p.Plan);
                EnterStage(first);
                long started = Stopwatch.GetTimestamp();
                ok = ranked.RunRanked(compiled, runInputs, exactSums: false, out error);
                long ticks = Stopwatch.GetTimestamp() - started;
                for (int i = 0; i < steps.Length; i++) steps[i].CountRun(compiled[i], ticks / steps.Length);
            }
            if (!ok)
                throw new InvalidOperationException($"{Config.Architecture} pipelined prompt wavefront (stages {first}-{first + plans.Length - 1}) failed: {error}");
        }
    }
}
