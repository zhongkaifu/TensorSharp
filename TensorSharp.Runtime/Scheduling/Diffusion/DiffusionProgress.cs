// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
namespace TensorSharp.Runtime.Scheduling.Diffusion;

/// <summary>The phases a diffusion job runs through, in order (docs/design/diffusion-runtime.md section 3.4).</summary>
public enum DiffusionPhase
{
    /// <summary>Waiting in the model's queue.</summary>
    Queued,
    /// <summary>Validating the request against the family and its sampling contract; resolving canvas, references, mask.</summary>
    Prepare,
    /// <summary>The text (and vision) conditioning.</summary>
    Condition,
    /// <summary>The VAE encode of the references (and of a mask's source).</summary>
    EncodeReferences,
    /// <summary>The denoising loop.</summary>
    Denoise,
    /// <summary>The VAE decode.</summary>
    Decode,
    /// <summary>Resizing to the requested size and compositing a masked edit.</summary>
    Composite,
    /// <summary>A joint audio-video family's audio encode of its sound conditioning (after <see cref="EncodeReferences"/>).</summary>
    EncodeAudio,
    /// <summary>A joint audio-video family's audio decode (after <see cref="Decode"/>).</summary>
    DecodeAudio,
}

/// <summary>
/// One progress report of a diffusion job: the phase it entered (<see cref="Step"/> 0), or a finished denoising step
/// (<see cref="Step"/> of <see cref="Total"/>, 1-based) with an optional decoded <see cref="Preview"/> (an image of the
/// model's own type, such as an RgbImage; null on the steps between previews). Reports arrive on the job's compute
/// thread, synchronously, in order.
/// </summary>
public readonly record struct DiffusionProgress(DiffusionPhase Phase, int Step = 0, int Total = 0, object? Preview = null)
{
    /// <summary>A finished denoising step (what the hosts' per-step callbacks and SSE step frames report).</summary>
    public bool IsStep => Phase == DiffusionPhase.Denoise && Step > 0;
}
