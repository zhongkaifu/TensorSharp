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

        /// <summary>Wavefronts the pipelined prompts ran (a test seam: 0 when every prompt ran chunk after chunk).</summary>
        internal int PipelinedWavefronts { get; private set; }

        /// <summary>
        /// Whether the plain prompt rows [0, <paramref name="count"/>) of a refill run pipelined over the whole-layer stages
        /// (<see cref="TryRunPipelinedPrompt"/>): local stages on one executor that runs several devices' plans together (the
        /// GGML executors: their staged chunks otherwise run one device at a time, each run waiting for every device), a plain
        /// prompt (no media, no position table, no draft or DFlash ring following it, no step adapter), no experts on the host
        /// (a ranked run takes no host-expert pause), and two chunks at least.
        /// </summary>
        private bool PipelinesPrompt(int count)
        {
            if (!PipelinedPromptEnabled || PipelinedPromptDisabledForTests || _mesh == null || _mesh.Stages < 2 || _rankGraphs || _mesh.TensorParallel > 1) return false;
            if (LayerSplitGroup?.NodeCount > 1 || HasPendingMedia || _pendingPositionTable != null || _followPlainPrompt) return false;
            if (StepAdapter != null || AnyExpertLayerOffloaded()) return false;
            if (count < 2 * PromptChunkTokens || PromptChunkTokens > _maxSequenceStepRows) return false;
            if (GraphExecutor is not { } executor || GraphUnsupportedReason() != null) return false;
            bool ggml = executor is IRankedGraphExecutor && _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
            return ggml || (PipelinedPromptForTests && executor is TensorGraphExecutor);
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
        /// (<see cref="IRankedGraphExecutor.RunRanked"/>: each device's graph queued before any is waited for), so stage s + 1
        /// works on chunk k while stage s works on chunk k + 1. Each stage's step is the one the chunk's sequential step emits
        /// (its layers, caches and states, the chunk's tokens, positions and start): a stage reads only its own layers' caches,
        /// and the rows it hands on are the previous wavefront's output of the stage before it, so the result is the chunks'
        /// one after another, bit for bit. Two stages take max(stage) a wavefront where one chunk after another took their sum
        /// (Qwen3.5-9B Q8_0 on two RTX 2000 Ada: an 8K prompt at 1,893 tok/s over two GGML stages against 1,924 on one GPU;
        /// direct CUDA's stages, whose runs return with their device work queued, already overlapped: 2,794).
        /// </summary>
        private void TryRunPipelinedPrompt(int[] tokens, int count)
        {
            SettlePendingVerify();
            int chunk = PromptChunkTokens, chunks = (count + chunk - 1) / chunk, stages = _mesh.Stages;
            int baseStart = _cacheSeqLen;
            EnsureCacheCapacity(checked(baseStart + count));
            PrepareHeadLayout();
            MarkDrafterWeights();
            RoomForRecordings();
            IGraphExecutor executor = GraphExecutor;
            bool drafting = executor.Drafting;
            int sequences = executor.StepSequences;
            executor.Drafting = false;
            executor.StepSequences = 1;
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
                                PrepareWrite(_cacheK, start);
                                BeginRotaryPositions(start, rows);
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
                            Slot slot = SnapshotActiveFields();
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
                _cacheDeviceDirty = true;
                _cacheSeqLen = baseStart + count;
            }
            finally
            {
                _forwardSw.Stop();
                executor.Drafting = drafting;
                executor.StepSequences = sequences;
                EnterStage(0);
            }
        }

        /// <summary>One wavefront's plans, the stages first .. first + n - 1 in order: one plan runs alone; several run together
        /// (the GGML executor queues every device's graph before it waits for any; the tensor executors run them in lockstep).</summary>
        private void RunWavefront(GraphStep[] steps, CachedGraph[] plans, int first)
        {
            string error;
            bool ok;
            if (plans.Length == 1)
            {
                EnterStage(first);
                ok = steps[0].Run(plans[0].Plan);
                error = plans[0].Plan.LastError;
            }
            else
            {
                var runInputs = Array.ConvertAll(steps, step => step.RunInputs());
                var compiled = Array.ConvertAll(plans, p => p.Plan);
                EnterStage(first);
                long started = Stopwatch.GetTimestamp();
                ok = GraphExecutor is IRankedGraphExecutor ranked && GraphExecutor is not TensorGraphExecutor
                    ? ranked.RunRanked(compiled, runInputs, exactSums: false, out error)
                    : TensorGraphExecutor.RunRanked(compiled, runInputs, out error);
                long ticks = Stopwatch.GetTimestamp() - started;
                for (int i = 0; i < steps.Length; i++) steps[i].CountRun(compiled[i], ticks / steps.Length);
            }
            if (!ok)
                throw new InvalidOperationException($"{Config.Architecture} pipelined prompt wavefront (stages {first}-{first + plans.Length - 1}) failed: {error}");
        }
    }
}
