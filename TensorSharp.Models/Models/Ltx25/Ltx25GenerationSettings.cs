using System;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.Ltx25;

/// <summary>The distilled text/image-to-video/audio recipe. Width and height
/// are the output size. Like the publisher's DistilledPipeline, every mode except
/// Ingredients identity references samples eight steps at half that size, then
/// the x2 latent upscaler and three refinement steps at full size; see
/// <see cref="TwoStage"/>. Ingredients references keep their single full-size pass.</summary>
public sealed record Ltx25GenerationSettings(int Width, int Height, int Frames, int Fps,
    int LatentWidth, int LatentHeight, int LatentFrames, int AudioTokens, long Seed)
{
    public const int Steps = 8;
    /// <summary>The publisher's DistilledPipeline samples LTX 2.5+ checkpoints with ancestral Euler.</summary>
    public const string AncestralSampler = "euler_ancestral";
    /// <summary>Deterministic Euler: an explicit diagnostic, not the publisher's 2.5 recipe.</summary>
    public const string EulerSampler = "euler";
    /// <summary>ANCESTRAL_NOISE_SEED_OFFSET: the stage's step noise is seeded apart from the initial latent.</summary>
    public const long AncestralNoiseSeedOffset = 10000;
    /// <summary>ANCESTRAL_STAGE_2_NOISE_SEED_OFFSET: the refinement pass draws its own step noise.</summary>
    public const long AncestralStage2NoiseSeedOffset = 20000;
    /// <summary>The refinement pass's first sigma, which also scales the noise mixed into the upsampled latent.</summary>
    public const float RefinementNoiseScale = .909375f;
    public int VideoTokens => checked(LatentWidth * LatentHeight * LatentFrames);
    /// <summary>True for the publisher's ancestral recipe; false only when euler was requested.</summary>
    public bool Ancestral { get; init; } = true;
    /// <summary>Half-size first pass, x2 latent upscale and refinement (every mode but Ingredients references).</summary>
    public bool TwoStage { get; init; }
    /// <summary>Total denoising steps of the recipe: eight, plus three refinement steps for two stages.</summary>
    public int TotalSteps => TwoStage ? Steps + RefinementSteps : Steps;
    public const int RefinementSteps = 3;
    /// <summary>Default output size when a request leaves width and height unset.</summary>
    public const int DefaultSize = 1024;
    /// <summary>The Ingredients IC-LoRA's trained bucket ("768×448, 121 frames, 24 fps"), the identity default size.</summary>
    public const int ReferenceWidth = 768, ReferenceHeight = 448;

    public static Ltx25GenerationSettings Resolve(VideoGenerationParams parameters, bool supportsIdentityReferences = false)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        if (parameters.ReferenceAudioPaths is { Count: > 0 } || parameters.ReferenceVideoAudioPaths is { Count: > 0 })
            throw new NotSupportedException("LTX audio-reference workflows are not implemented by this distilled preset.");
        if (parameters.ReferenceVideoPaths is { Count: > 0 } && parameters.Mode != "ref")
            throw new ArgumentException("LTX full-video references require Ingredients ref mode.");
        if (parameters.ReferenceVideoPaths is { Count: > Ltx25ReferenceConditioning.MaxVideos })
            throw new ArgumentException($"LTX Ingredients accepts at most {Ltx25ReferenceConditioning.MaxVideos} reference video.");
        bool identity = parameters.Mode == "ref";
        if (identity && !supportsIdentityReferences)
            throw new NotSupportedException("LTX identity/appearance references require the separate Ingredients IC-LoRA. This preset accepts temporal image guides in i2v, fl2v or keyframes mode.");
        if (!string.IsNullOrWhiteSpace(parameters.Mode) && parameters.Mode is not ("t2v" or "t2va" or "i2v" or "fl2v" or "keyframes" or "ref"))
            throw new NotSupportedException("LTX 2.5 generation mode must be t2v, t2va, i2v, fl2v, keyframes or Ingredients ref.");
        if (parameters.Steps is not (0 or Steps))
            throw new ArgumentException("LTX 2.5 Distilled requires its published eight-step schedule.");
        if (parameters.CfgScale is not (0 or 1))
            throw new ArgumentException("LTX 2.5 Distilled uses CFG 1; classifier-free guidance is not enabled.");
        bool ancestral = string.IsNullOrWhiteSpace(parameters.Sampler) ||
            parameters.Sampler.Equals(AncestralSampler, StringComparison.OrdinalIgnoreCase);
        if (!ancestral && !parameters.Sampler.Equals(EulerSampler, StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"LTX 2.5 Distilled samples with {AncestralSampler} (the publisher's recipe); " +
                $"{EulerSampler} selects deterministic Euler for diagnostics.");
        if (parameters.FlowShift != 0)
            throw new ArgumentException("LTX 2.5 Distilled uses fixed sigmas and does not accept a flow-shift override.");
        int width = parameters.Width == 0 ? identity ? ReferenceWidth : DefaultSize : parameters.Width;
        int height = parameters.Height == 0 ? identity ? ReferenceHeight : DefaultSize : parameters.Height;
        // The half-size first pass keeps the 32-pixel latent grid only for 64-pixel multiples.
        if (!identity && (width <= 0 || height <= 0 || width % 64 != 0 || height % 64 != 0))
            throw new ArgumentException("LTX 2.5 generates at half size before upscaling, so width and height must be positive multiples of 64.");
        int frames = parameters.Frames == 0 ? identity ? 121 : 17 : parameters.Frames;
        int fps = parameters.Fps == 0 ? 24 : parameters.Fps;
        if (fps <= 0 || fps > 240) throw new ArgumentOutOfRangeException(nameof(parameters.Fps));
        var latent = Ltx25Geometry.VideoLatentShape(frames, height, width);
        if (identity)
        {
            if (frames < Ltx25ReferenceConditioning.MinimumFrames)
                throw new ArgumentException("Ingredients identity generation requires at least 121 frames on the 8n+1 temporal grid.");
            _ = Ltx25ReferenceConditioning.ResolveSources(parameters);
            _ = Ltx25ReferenceConditioning.ResolveFirstFrameGuide(parameters);
        }
        else _ = Ltx25ImageConditioning.Resolve(parameters, frames);
        // Original AudioLatentShape.from_duration uses Python's ties-to-even
        // round, rather than SD.cpp's ceil. Keep the original latent geometry.
        int audioTokens = Math.Max(1, checked((int)Math.Round(frames / (double)fps * 25, MidpointRounding.ToEven)));
        long seed = parameters.Seed < 0 ? Random.Shared.NextInt64(long.MaxValue) : parameters.Seed;
        return new(width, height, frames, fps, latent.Width, latent.Height, latent.Frames, audioTokens, seed)
        { Ancestral = ancestral, TwoStage = !identity };
    }

    /// <summary>The half-size first pass of the two-stage recipe.</summary>
    public Ltx25GenerationSettings HalfResolution()
    {
        if (!TwoStage || Width % 64 != 0 || Height % 64 != 0)
            throw new InvalidOperationException("Only a two-stage request with 64-pixel multiples has a half-size first pass.");
        var latent = Ltx25Geometry.VideoLatentShape(Frames, Height / 2, Width / 2);
        return this with { Width = Width / 2, Height = Height / 2, LatentWidth = latent.Width, LatentHeight = latent.Height };
    }

    /// <summary>Step <paramref name="step"/>'s ancestral noise: one Philox draw per
    /// step (word 0 = step) keyed by the seed plus <see cref="AncestralNoiseSeedOffset"/>,
    /// video tokens first and audio after, as the original draws them from one
    /// generator. Bit parity with Torch's generator is not claimed.</summary>
    public Ltx25StepNoise AncestralNoise(int step, int videoLength, int audioLength, long seedOffset = AncestralNoiseSeedOffset)
    {
        if (step < 0 || videoLength <= 0 || audioLength <= 0) throw new ArgumentOutOfRangeException(nameof(step));
        float[] packed = PhiloxNoise.Draw(checked(videoLength + audioLength),
            unchecked(Seed + seedOffset), 0, (uint)step);
        return new(packed.AsSpan(0, videoLength).ToArray(), packed.AsSpan(videoLength).ToArray());
    }

    /// <summary>CUDA Philox noise in the baseline's packed channel-major storage.
    /// One complete padded AV draw preserves its counter offsets, then the two
    /// streams are rearranged into the denoiser's token-major layout.</summary>
    public (float[] Video, float[] Audio) InitialNoise(int extraVideoTokens = 0, ulong stream = 0)
    {
        if (extraVideoTokens < 0) throw new ArgumentOutOfRangeException(nameof(extraVideoTokens));
        int tokens = checked(VideoTokens + extraVideoTokens);
        const int videoChannels = 128, audioChannels = 8, audioBins = 16;
        int videoLength = checked(tokens * videoChannels);
        int audioLength = checked(AudioTokens * audioChannels * audioBins);
        int extraChannels = checked((audioLength + tokens - 1) / tokens);
        float[] packed = PhiloxNoise.Draw(checked(tokens * (videoChannels + extraChannels)), Seed, stream);
        float[] video = Ltx25Geometry.PackVideo(packed.AsSpan(0, videoLength), tokens, videoChannels);
        float[] audio = Ltx25Geometry.PackAudio(packed.AsSpan(videoLength, audioLength), AudioTokens, audioChannels, audioBins);
        return (video, audio);
    }
}
