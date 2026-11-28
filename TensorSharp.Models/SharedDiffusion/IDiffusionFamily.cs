// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Threading;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace TensorSharp.Models.SharedDiffusion
{
    /// <summary>
    /// A diffusion family on the shared runtime (docs/design/diffusion-runtime.md section 3.2): its geometry, its
    /// components, its default recipe and the layout of its packed sequence. <see cref="DiffusionPipeline"/> owns the
    /// rest: the phases, residency, sampling, guidance, masks and the final composite.
    /// </summary>
    public interface IDiffusionFamily
    {
        /// <summary>The architecture id (logs, refusals).</summary>
        string Id { get; }

        /// <summary>The name refusals use ("MiniMax-H3-Image").</summary>
        string DisplayName { get; }

        DiffusionGeometry Geometry { get; }
        DiffusionGuidance Guidance { get; }
        SamplingRecipe DefaultRecipe { get; }
        IReferencePolicy References { get; }

        IConditioner CreateConditioner(DiffusionLoad load);

        /// <summary>The denoiser: emits one step's program(s) and runs them.</summary>
        IDenoiser CreateDenoiser(DiffusionLoad load);

        /// <summary>The VAE: encode and decode programs and their tiling.</summary>
        IVaeCodec CreateVae(DiffusionLoad load);

        /// <summary>The request's noise stream at <paramref name="seed"/> (the CPU torch generator, Philox, ...).</summary>
        INoiseSource Noise(long seed);

        /// <summary>The noise stream <paramref name="request"/> draws from, made after its refusals: <see cref="Noise(long)"/>
        /// at its seed, unless the family's stream also depends on what else the request holds (Qwen-Image-2.1's edits
        /// draw on a stream keyed to their reference pictures as the caller passed them).</summary>
        INoiseSource Noise(DiffusionRequest request) => Noise(request.Seed);

        /// <summary>Where <paramref name="request"/> samples and the size it returns (text to image:
        /// <see cref="DiffusionCanvas.TextToImage"/>; an edit: the family's reference rules). A masked edit returns its
        /// source canvas (<paramref name="mask"/>, null when unmasked) and samples its selection
        /// (<see cref="EditMask.SamplingDimensions"/>).</summary>
        DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask);

        /// <summary>The packed sequence of one step: text, then references, then the target (segments, positions,
        /// row timesteps).</summary>
        DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step);

        /// <summary>A latent (<c>[C, F, H, W]</c>) as the denoiser's token rows.</summary>
        float[] ToRows(LatentTensor latent);

        /// <summary>Token rows back to a latent of <paramref name="shape"/>.</summary>
        LatentTensor FromRows(float[] rows, LatentShape shape);

        /// <summary>A mask's per-cell weights (<see cref="EditMask.LatentWeights"/>, row-major [H, W]) in the token rows'
        /// layout: the weights and how many contiguous row values each covers (1: one weight a value).</summary>
        (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape);

        /// <summary>The family's upstream resize filter: the decode to the size the request returns (a generation floor, a
        /// kept source size, a masked edit's selection), and a masked edit's source to the sampling canvas when it is
        /// encoded on its own. The default is <see cref="QwenImage.ImageIO.Resize"/>.</summary>
        RgbImage ResizeOutput(RgbImage image, int width, int height) => QwenImage.ImageIO.Resize(image, width, height);
    }

    /// <summary>
    /// A component the runtime may keep on the device between phases and requests, or release
    /// (<see cref="ResidencyPlanner"/>). Making a component maps its weights on the host; nothing reaches the device
    /// until its first use, so the runtime can make every component a request needs before it plans. Releasing drops
    /// the device copies only: the host mapping of the weights stays, so the next use re-uploads from memory, not from
    /// disk.
    /// </summary>
    public interface IResidentComponent : IDisposable
    {
        /// <summary>Device bytes the component's weights take while resident.</summary>
        long WeightBytes { get; }

        /// <summary>Device bytes the component's activations and compute buffers take for <paramref name="workload"/>.</summary>
        long WorkingBytes(DiffusionWorkload workload);

        /// <summary>The working memory of <paramref name="phase"/>, for a component that runs in more than one (the VAE
        /// encodes references and decodes the result). The default is <see cref="WorkingBytes(DiffusionWorkload)"/>.</summary>
        long WorkingBytes(DiffusionWorkload workload, DiffusionPhase phase) => WorkingBytes(workload);

        /// <summary>Drop the weights' device copies (<c>IGraphExecutor.Invalidate</c> on their leaves).</summary>
        void ReleaseDevice();

        /// <summary>Drop the activations and compute buffers (cached plans); the weights stay resident.</summary>
        void ReleaseWorkspace();

        /// <summary>The component releases its device memory (weights and workspace) after every phase it runs, whatever the
        /// residency plan would keep: a family whose components took turns on the device before it moved to the runtime
        /// (Qwen-Image-2.1's), and whose working memory is not calibrated on every device yet.</summary>
        bool ReleaseAfterPhase => false;

        /// <summary>
        /// The fewest of the component's weight bytes <paramref name="phase"/> must hold on the device at once when it streams
        /// its weights through the device a group at a time (a group's device copies go before the next group's are made), or
        /// 0 when the phase holds every weight. The residency plan streams a phase only when its component's weights and
        /// working memory do not fit with every other component released (<see cref="ResidencyStream"/>). The default: no
        /// streaming.
        /// </summary>
        long StreamedWeightBytes(DiffusionWorkload workload, DiffusionPhase phase) => 0;

        /// <summary>Hold at most <paramref name="weightBytes"/> of the weights on the device at once in the next phase,
        /// streaming the rest through it a group at a time, or every weight (null: the default). The pipeline sets it before
        /// every phase the component runs, and releases a streamed component's device copies and workspace after the phase.</summary>
        void StreamWithin(long? weightBytes) { }
    }

    /// <summary>Text (and, for references, the vision tower) to hidden rows with modality tags.</summary>
    public interface IConditioner : IResidentComponent
    {
        ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken cancellationToken);
    }

    /// <summary>One step's prediction from the packed sequence.</summary>
    public interface IDenoiser : IResidentComponent
    {
        DenoiserCaps Caps { get; }

        /// <summary>The step's prediction. <see cref="DenoiseStepInputs.Latent"/> is the sampler's state, read only;
        /// the output's arrays may be the denoiser's own buffers, valid until its next step.</summary>
        StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken cancellationToken);

        /// <summary>A request's denoising ended (finished, failed or cancelled): drop what the denoiser kept for that request
        /// alone (Qwen-Image-2.1's prefix keys and values), before the decode needs the room. Weights and cached plans stay.</summary>
        void FinishRequest() { }
    }

    /// <summary>The VAE: tiled encode to a posterior, tiled and blended decode, and a cheap preview.</summary>
    public interface IVaeCodec : IResidentComponent
    {
        LatentPosterior Encode(RgbImage image, CancellationToken cancellationToken);

        RgbImage Decode(LatentTensor latent, CancellationToken cancellationToken);

        /// <summary>A reduced decode whose longer side is at most about <paramref name="maxSide"/> pixels.</summary>
        RgbImage Preview(LatentTensor latent, int maxSide);
    }

    /// <summary>
    /// How a family conditions on reference pictures: the copy each consumer sees (resize rules), the latent rows a
    /// posterior gives, and the augmentation, each drawing from the request's noise stream in the family's order
    /// (diffusers' MiniMax-H3-Image: every posterior sample, then the target noise, then every augmentation).
    /// </summary>
    public interface IReferencePolicy
    {
        /// <summary>The vision tower's copy of reference <paramref name="index"/>.</summary>
        PresentedImage ForVision(RgbImage image, int index);

        /// <summary>The VAE's copy of reference <paramref name="index"/>.</summary>
        RgbImage ForVae(RgbImage image, int index);

        /// <summary>The vision tower's copy of reference <paramref name="index"/> for a request that samples on
        /// <paramref name="canvas"/> (a family whose references follow the output's size: Qwen-Image-2.1's are conditioned at
        /// about 1 MP, or the output's area if smaller); the default is <see cref="ForVision(RgbImage, int)"/>.</summary>
        PresentedImage ForVision(RgbImage image, int index, DiffusionCanvas canvas) => ForVision(image, index);

        /// <summary>The VAE's copy of reference <paramref name="index"/> for a request that samples on <paramref name="canvas"/>;
        /// the default is <see cref="ForVae(RgbImage, int)"/>.</summary>
        RgbImage ForVae(RgbImage image, int index, DiffusionCanvas canvas) => ForVae(image, index);

        /// <summary>The reference's token rows from its posterior, drawing the posterior sample from
        /// <paramref name="noise"/> (before the target noise).</summary>
        LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise);

        /// <summary>Augment a reference's rows in place, drawing from <paramref name="noise"/> (after the target noise).</summary>
        void Augment(LatentRef reference, INoiseSource noise);

        /// <summary>A masked edit's source latent (docs/design/diffusion-runtime.md section 4.5): the posterior's mean in the
        /// space the denoiser works in, so that the kept cells end on it exactly. A family whose VAE hands back normalized
        /// moments needs nothing more (the default); one that normalizes its references' samples itself (MiniMax-H3-Image's
        /// latent statistics) normalizes the mean the same way.</summary>
        LatentTensor SourceLatent(LatentPosterior posterior) => new(posterior.Shape, posterior.Mean);
    }
}
