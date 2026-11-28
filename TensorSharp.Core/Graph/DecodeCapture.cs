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

        /// <summary>Whether the device keeps room for another recording beside what it holds (a recording keeps the memory its
        /// step used); false defers it.</summary>
        bool HasRoomToRecord() => true;

        /// <summary>The sequences one recorded step may batch, each reading its own attention length and write row on the
        /// device.</summary>
        int MaxRecordedSequences => 1;

        /// <summary>A capture of one kept tensor-parallel decode step over <paramref name="ranks"/> (this executor rank 0, each
        /// rank's executor on its own device), recorded as one launch for every rank; null where the ranks cannot record so.</summary>
        IRankedDecodeCapture OpenRankedDecodeCapture(IReadOnlyList<IDecodeCaptureExecutor> ranks) => null;
    }

    /// <summary>
    /// A tensor-parallel decode step's ranks recorded as one launch (their all-reduces ordered on the devices inside it) and
    /// replayed a position on at a time: each rank binds its inputs to its own <see cref="Rank"/>'s stable tensors, and the
    /// recording and every replay run every rank's work together.
    /// </summary>
    public interface IRankedDecodeCapture : IDisposable
    {
        /// <summary>The step is recorded: <see cref="TryLaunch"/> replays it.</summary>
        bool Captured { get; }

        /// <summary>Rank <paramref name="rank"/>'s stable inputs (<see cref="IDecodeCapture.Stable"/>) and the tensors a launch
        /// rewrote (<see cref="IDecodeCapture.MarkWritten"/>); the group records and launches.</summary>
        IDecodeCapture Rank(int rank);

        /// <summary>Record <paramref name="run"/>, which runs every rank's plan (device work only), and launch it. Each rank's
        /// sequences and written tensors are given as <see cref="IDecodeCapture.CaptureAndLaunch"/> takes them.</summary>
        bool CaptureAndLaunch(Func<bool> run, ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows,
            IReadOnlyList<IReadOnlyList<Tensor>>[] sequences, IReadOnlyList<Tensor>[] written);

        /// <summary>Replay the recorded step with this position's lengths and write rows (see
        /// <see cref="IDecodeCapture.TryLaunch"/>).</summary>
        bool TryLaunch(ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows);
    }

    /// <summary>
    /// One decode step recorded on the device and replayed a position on at a time. The step's inputs are uploaded to
    /// <see cref="Stable"/> tensors before each launch, each sequence's attention length and cache write row are read on the
    /// device, and every other launch argument stays what the recorded step had (the same slots, caches and windows).
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
        /// Sequence s attends <paramref name="attendLengths"/>[s] keys and writes its rows from <paramref name="writeRows"/>[s];
        /// <paramref name="sequences"/>[s] are the tensors its launches name it by (its caches). <paramref name="written"/> are
        /// the tensors the step writes in place (caches, states): every launch leaves them device-current. False when the
        /// recording failed: nothing ran, so the caller runs the step itself.
        /// </summary>
        bool CaptureAndLaunch(Func<bool> run, ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows,
            IReadOnlyList<IReadOnlyList<Tensor>> sequences, IReadOnlyList<Tensor> written);

        /// <summary>Replay the recorded step with this position's attention lengths and cache write rows, one a sequence. False
        /// when the device could not launch it (out of memory): nothing ran, so the caller drops recordings and runs the step
        /// itself.</summary>
        bool TryLaunch(ReadOnlySpan<int> attendLengths, ReadOnlySpan<int> writeRows);

        /// <summary>A launch rewrote these tensors on the device (the recorded step's outputs): reads go to the device.</summary>
        void MarkWritten(IReadOnlyList<Tensor> tensors);

        /// <summary>The step's state leaves (caches, recurrent states and their buffers) in the order its plan binds them,
        /// given before <see cref="CaptureAndLaunch"/>: a recording that knows them can move to another slot's
        /// (<see cref="TryRelocate"/>).</summary>
        void NoteStates(IReadOnlyList<Tensor> states) { }

        /// <summary>
        /// Move the recording to <paramref name="states"/> - the state leaves of a step with the same plan structure over
        /// another slot, in the same order, each the size and at the offset of its counterpart - and to
        /// <paramref name="written"/>, the tensors that step writes in place. False when it cannot move: nothing changed
        /// unless <see cref="Captured"/> turned false (the recording broke part way and must be dropped).
        /// </summary>
        bool TryRelocate(IReadOnlyList<Tensor> states, IReadOnlyList<Tensor> written) => false;
    }
}
