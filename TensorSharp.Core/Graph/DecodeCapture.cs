// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Graph
{
    /// <summary>
    /// A host input's copy at a fixed device address (a captured decode step's): a plan binds <see cref="Device"/> as the
    /// input's tensor and still reads the host values it is given (a mask's pattern, positions), as for any host input.
    /// </summary>
    public sealed class StableInput
    {
        public StableInput(Tensor device) => Device = device ?? throw new ArgumentNullException(nameof(device));

        public Tensor Device { get; }

        /// <summary>The values change from launch to launch in a way the recording cannot follow (a family's refilled
        /// input): a plan binds no host values for it, so an op that would read them on the host aborts the recording
        /// rather than replaying the recorded launch's values.</summary>
        public bool Varies { get; init; }
    }

    /// <summary>An executor that can record a decode step's device work once and replay it as one launch.</summary>
    public interface IDecodeCaptureExecutor
    {
        /// <summary>A capture for one kept decode step (null where this executor cannot record).</summary>
        IDecodeCapture OpenDecodeCapture();
    }

    /// <summary>
    /// One decode step recorded on the device and replayed a position on at a time. The step's inputs are uploaded to
    /// <see cref="Stable"/> tensors before each launch, the attention length and the cache write row are read on the device,
    /// and every other launch argument stays what the recorded step had (the same slots, caches and window).
    /// </summary>
    public interface IDecodeCapture : IDisposable
    {
        /// <summary>The step is recorded: <see cref="Launch"/> replays it.</summary>
        bool Captured { get; }

        /// <summary>The tensor input <paramref name="slot"/> is bound to at every launch (allocated once). A
        /// <paramref name="varies"/> input's host copy is stale once uploaded, so a host read of it inside the recording
        /// aborts it (see <see cref="StableInput.Varies"/>).</summary>
        Tensor Stable(int slot, DType type, long[] sizes, bool varies = false);

        /// <summary>
        /// Record <paramref name="run"/> (device work only: a host read of device data aborts the recording) and launch it.
        /// <paramref name="written"/> are the tensors the step writes in place (caches, states): every launch leaves them
        /// device-current. False when the recording failed: nothing ran, so the caller runs the step itself.
        /// </summary>
        bool CaptureAndLaunch(Func<bool> run, int attendLength, int writeRow, IReadOnlyList<Tensor> written);

        /// <summary>Replay the recorded step with this position's attention length and cache write row.</summary>
        void Launch(int attendLength, int writeRow);

        /// <summary>A launch rewrote these tensors on the device (the recorded step's outputs): reads go to the device.</summary>
        void MarkWritten(IReadOnlyList<Tensor> tensors);
    }
}
