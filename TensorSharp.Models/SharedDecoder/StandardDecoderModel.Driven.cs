// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;

using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    /// <summary>
    /// One request whose slots a step driver chooses, run without the engine (a library caller's speech synthesis, the
    /// CLI's): its own slot, the prompt in chunks, then the driver's slots, each step under the model's compute lock, so an
    /// engine serving other requests on the same model runs its steps between them. A host with an engine submits such a
    /// request to it instead (<see cref="SequenceState.StepDriver"/>), where it batches with every other request.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        /// <summary>The prompt chunk of a driven request run alone.</summary>
        private const int DrivenPromptChunk = 512;

        /// <summary>
        /// Run <paramref name="prompt"/> then <paramref name="driver"/>'s slots (at most <paramref name="maxSlots"/>) on a slot
        /// of their own; text slots draw with <paramref name="sampler"/>. <paramref name="afterSlot"/> runs after each slot
        /// outside the lock (a consumer taking the driver's output). Returns the slots decoded; the reason the request ended
        /// ("eos" from the driver, "max_tokens" at the limit or the context's end) is <paramref name="finishReason"/>.
        /// </summary>
        protected internal int RunDrivenRequest(IReadOnlyList<int> prompt, ISequenceStepDriver driver, int maxSlots, TokenSampler sampler,
            Action afterSlot, CancellationToken cancellationToken, out string finishReason)
        {
            ArgumentNullException.ThrowIfNull(prompt);
            ArgumentNullException.ThrowIfNull(driver);
            ArgumentNullException.ThrowIfNull(sampler);
            if (prompt.Count == 0) throw new ArgumentException("an empty prompt", nameof(prompt));
            if (!SupportsUnifiedRowIO)
                throw new NotSupportedException($"{Config.Architecture}: this load runs no slot steps (a multi-node pipeline), which a driven request needs.");
            string id = $"driven-{Guid.NewGuid():N}";
            int width = Config.HiddenSize;
            int room = MaxContextLength > 0 ? MaxContextLength - prompt.Count : maxSlots;
            if (room <= 0)
                throw new ContextLengthExceededException(prompt.Count, MaxContextLength);
            maxSlots = Math.Min(maxSlots, room);
            var output = new List<int>();
            float[] logits = null;
            float[] hidden = new float[width];
            finishReason = "max_tokens";
            if (!TryEnterUse()) throw new ModelUnloadedException();
            try
            {
                lock (GpuComputeLock) EnsureSequenceSlot(id, prompt.Count + maxSlots);
                try
                {
                    for (int pos = 0; pos < prompt.Count; pos += DrivenPromptChunk)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        int len = Math.Min(DrivenPromptChunk, prompt.Count - pos);
                        var chunk = new int[len];
                        for (int i = 0; i < len; i++) chunk[i] = prompt[pos + i];
                        bool last = pos + len == prompt.Count;
                        lock (GpuComputeLock)
                            logits = ForwardDriven(id, chunk, pos, null, last && driver.NextNeedsLogits, last && driver.NextNeedsHidden ? hidden : null);
                    }
                    int position = prompt.Count;
                    for (int n = 0; n < maxSlots; n++)
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        lock (GpuComputeLock)
                        {
                            float[] stepLogits = logits;
                            DriverSlot slot = driver.Next(logits, hidden, () => sampler.Sample(stepLogits, output));
                            if (slot.IsStop)
                            {
                                finishReason = slot.StopReason;
                                break;
                            }
                            output.Add(slot.Token);
                            bool wantLogits = driver.NextNeedsLogits, wantHidden = driver.NextNeedsHidden;
                            logits = ForwardDriven(id, new[] { slot.Token }, position, slot.Row, wantLogits, wantHidden ? hidden : null);
                            position++;
                        }
                        afterSlot?.Invoke();
                    }
                }
                finally
                {
                    lock (GpuComputeLock) OnSequenceReleased(id);
                }
            }
            finally
            {
                ExitUse();
            }
            afterSlot?.Invoke();
            return output.Count;
        }

        /// <summary>One forward of a driven request's slot; the last row's logits (when asked for, else null).</summary>
        private float[] ForwardDriven(string id, int[] tokens, int start, float[] row, bool wantLogits, float[] hidden)
        {
            var logits = new float[1][];
            if (!TryForwardUnifiedRows(new[] { id }, new[] { tokens }, new[] { start }, new[] { row }, new[] { wantLogits }, logits,
                    new[] { hidden }))
                throw new InvalidOperationException($"{Config.Architecture}: the slot of a driven request declined its step at {start}.");
            if (wantLogits && logits[0] == null)
                throw new InvalidOperationException($"{Config.Architecture}: a driven step returned no logits.");
            return wantLogits ? (float[])logits[0].Clone() : null;
        }
    }
}
