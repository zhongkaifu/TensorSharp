// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Scheduling
{
    /// <summary>The next decode row of a request whose slots a step driver chooses (<see cref="ISequenceStepDriver"/>).</summary>
    public readonly struct DriverSlot
    {
        private DriverSlot(int token, float[]? row, string? stopReason)
        {
            Token = token;
            Row = row;
            StopReason = stopReason;
        }

        /// <summary>The row's token: a text token the driver drew, or its placeholder for an embedding row (what the request's
        /// output records at that position, and what its token stream carries for it).</summary>
        public int Token { get; }

        /// <summary>The embedding row decoded in place of <see cref="Token"/>'s own embedding, or null (a text token).</summary>
        public float[]? Row { get; }

        /// <summary>Non-null: the request ends here, with this finish reason, and nothing more is decoded.</summary>
        public string? StopReason { get; }

        public bool IsStop => StopReason != null;

        /// <summary>Decode text token <paramref name="token"/>.</summary>
        public static DriverSlot Text(int token) => new(token, null, null);

        /// <summary>Decode <paramref name="row"/> (the model's embedding width) at the next position; the request's output
        /// records <paramref name="placeholderToken"/> there.</summary>
        public static DriverSlot Embedding(int placeholderToken, float[] row)
            => new(placeholderToken, row ?? throw new ArgumentNullException(nameof(row)), null);

        /// <summary>End the request (<paramref name="reason"/>: its finish reason, "eos" for a natural end).</summary>
        public static DriverSlot Stop(string reason = "eos") => new(-1, null, reason ?? "eos");
    }

    /// <summary>
    /// A request whose decode slots are not all text tokens: each slot's input is a token the request's sampler draws or an
    /// embedding row the driver makes, and a forward returns the last row's logits, its final hidden row (after the final
    /// norm, before the LM head), or both, as the driver's next slot needs. LFM3-Omni's speech output is one: an audio slot
    /// turns the hidden row into a frame of speech codes and decodes the frame's embedding row next.
    /// <para>
    /// The engine runs such a request on the unified runtime (<see cref="IUnifiedBatchModel.TryForwardUnifiedRows"/>),
    /// batched with every other request; it calls <see cref="Next"/> on its thread, inside the step and the model's compute
    /// lock, so the driver may run device work of its own there. The request's token stream carries every slot's
    /// <see cref="DriverSlot.Token"/> (a consumer tells placeholders and control tokens apart). Speculation, the device-greedy
    /// shortcut and end-state retention are off for it; it is refused off the unified runtime.
    /// </para>
    /// The creator owns the driver and disposes it once the request has ended; <see cref="Next"/> after that returns a stop.
    /// </summary>
    public interface ISequenceStepDriver : IDisposable
    {
        /// <summary>The next forward of the request (the prompt's last chunk, then each decode row) must return its last
        /// row's logits: the next slot draws a text token.</summary>
        bool NextNeedsLogits { get; }

        /// <summary>The next forward must return its last row's final hidden row: the next slot is made from it.</summary>
        bool NextNeedsHidden { get; }

        /// <summary>The next slot, from the last forward's outputs (<paramref name="logits"/> when
        /// <see cref="NextNeedsLogits"/> was true, <paramref name="hidden"/> when <see cref="NextNeedsHidden"/> was).
        /// <paramref name="drawToken"/> draws a text token from the logits with the request's sampler.</summary>
        DriverSlot Next(float[]? logits, float[]? hidden, Func<int> drawToken);
    }
}
