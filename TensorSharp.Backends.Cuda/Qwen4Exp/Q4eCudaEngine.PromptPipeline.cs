// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Prompt chunks in flight under a layer split. A prompt's ubatch runs the first device's layers, hands its streams over
// through pinned memory, then runs the last device's layers. Inside one forward the ubatches already overlap: the host
// issues the next ubatch's first-device work as soon as the last device has taken the previous ubatch's streams
// (Handoff's events), so each device computes one ubatch while the other computes its neighbour. But every forward ended
// with the LM head and a wait for it, although a prompt chunk that more of the prompt follows has no reader for its
// logits: the scheduler feeds a prompt in chunks of one or two ubatches, so the first device idled while the last one
// finished each chunk, and the last one idled while the first started the next. On two H100 NVL (Q8_0, 2048-row
// ubatches, 4096-token chunks) a chunk took 0.89 s, of which the last ~0.2 s was the wait for the last device.
//
// ForwardPromptDeferred issues a chunk without the head and returns while the last device still runs it. The next chunk
// of the same slot follows it exactly as a second ubatch of one call would: the same kernels on the same streams in the
// same order, so every row computes what it computed before (a test holds the prompt's logits and the decode after it
// to the waited-for prompt bit for bit). What the host rewrites per ubatch is guarded as before (the pinned tokens by
// the first device's stream, the PLE rows by the PLE device's stream, the boundary slice by Handoff's events). Any other
// call - another slot's forward, a decode step, a decode lane, a verify, a slot freed or rewound - waits for the chunk
// first (DrainPrompt, through DrainDecodeLanes). TS_Q4E_PROMPT_PIPELINE=0 waits for every chunk.
//
// Measured on those two H100 NVL: a 40K-token prompt read at 6,190 tok/s instead of 4,570 (5,910 before the PLE rows
// were gathered on several threads), a chunk in 0.63-0.65 s, with the same greedy answers. What bounds it now is the
// host: a device's command queue fills (plain launches then block for about a millisecond each), so the one issuing
// thread spends ~0.16 s on each device's half of an ubatch while the other device runs its queue dry, where a device
// needs ~0.28 s for its half. Issuing the two devices' halves from two threads, or interleaved, is what is left.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q4eCudaEngine
    {
        private readonly bool _promptPipeline = Environment.GetEnvironmentVariable("TS_Q4E_PROMPT_PIPELINE") != "0";
        // The next Forward is ForwardPromptDeferred's.
        private bool _deferPromptCall;
        // A deferred chunk may still be running: its slot's next prompt chunk follows it, everything else drains it.
        private bool _promptInFlight;
        private Slot _promptInFlightSlot;

        // TS_Q4E_PERF's split of a forward's host time: waiting for a device before its pinned inputs are rewritten, and
        // gathering the PLE rows.
        private long _fwdWaitTicks, _fwdPleTicks;
        private static double TicksMs(long ticks) => ticks * 1000.0 / System.Diagnostics.Stopwatch.Frequency;

        /// <summary>Prompt chunks issued without waiting for them (tests, diagnostics).</summary>
        internal long PromptChunksDeferred { get; private set; }

        /// <summary>
        /// Whether a prompt chunk of <paramref name="rows"/> rows that more of the prompt follows can be left in flight
        /// (<see cref="ForwardPromptDeferred"/>): a layer split, whose devices then overlap consecutive chunks, and a
        /// prompt-size chunk (a decode-size one runs the captured step, which waits for every device anyway).
        /// </summary>
        public bool DefersPromptChunk(int rows) => _promptPipeline && _devs.Length > 1 && rows > PerSlotMaxRows;

        /// <summary>
        /// A prompt chunk whose logits nobody reads (more of the prompt follows): the active slot advances over
        /// <paramref name="tokens"/> as in <see cref="Forward(int[], float[])"/>, without the LM head, and the call returns
        /// once the chunk is issued. The slot's next prompt chunk may be forwarded at once; any other call finishes this
        /// one first. A failure of the chunk on the device surfaces at that later call.
        /// </summary>
        public void ForwardPromptDeferred(int[] tokens)
        {
            if (tokens == null || !DefersPromptChunk(tokens.Length))
            {
                Forward(tokens, null, null);
                return;
            }
            _deferPromptCall = true;
            try { Forward(tokens, null, null); }
            finally { _deferPromptCall = false; }
        }

        /// <summary>Wait for a prompt chunk left in flight, on every device.</summary>
        private void DrainPrompt()
        {
            if (!_promptInFlight)
                return;
            _promptInFlight = false;
            _promptInFlightSlot = null;
            long start = System.Diagnostics.Stopwatch.GetTimestamp();
            foreach (var dev in _devs)
            {
                dev.MakeCurrent();
                CudaDriverApi.cuStreamSynchronize(dev.Stream).ThrowOnError();
            }
            if (_perf > 0)
                Console.Error.WriteLine($"[q4e-cuda] waited {System.Diagnostics.Stopwatch.GetElapsedTime(start).TotalMilliseconds:F1}ms for the prompt chunk in flight");
        }
    }
}
