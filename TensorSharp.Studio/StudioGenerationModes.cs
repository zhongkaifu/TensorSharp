// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
namespace TensorSharp.Studio;

/// <summary>What a request's images MEAN. The same picture is the animated first frame
/// in i2v, the pinned last frame (optionally with a pinned first frame) in fl2v, a timed
/// guide in keyframes, and an identity/appearance reference for a new scene in ref. Each
/// mode accepts exactly its own inputs, so Studio never reinterprets an image the user
/// placed for another purpose.
/// <para>First and last frames are each optional, as on MiniMax-H3's FL2VA checkpoint
/// ("zero, one or two images"; sglang's fl2va signatures [0], [-1], [0,-1]): none is t2v,
/// a first frame alone is i2v, and a last frame, alone or with a first, is fl2v. No mode
/// substitutes a blank image for a missing frame; the frame is simply not pinned.</para></summary>
public static class StudioGenerationModes
{
    /// <summary>Text only.</summary>
    public const string TextToVideo = "t2v";
    /// <summary>The image is the first frame and is animated.</summary>
    public const string ImageToVideo = "i2v";
    /// <summary>The last frame is pinned, and the first frame too when one is given.</summary>
    public const string FirstLastFrame = "fl2v";
    /// <summary>Images guide explicit output frames (optionally with pinned first/last frames).</summary>
    public const string Keyframes = "keyframes";
    /// <summary>The images are identity/appearance references for a new scene, not frames.</summary>
    public const string Reference = "ref";

    public static readonly IReadOnlyList<string> All = [TextToVideo, ImageToVideo, FirstLastFrame, Keyframes, Reference];

    /// <summary>The modes a checkpoint can run, from its conditioning capabilities.</summary>
    public static string[] For(StudioModelCapability capability)
    {
        var modes = new List<string>(All.Count);
        if (capability.TextToVideo) modes.Add(TextToVideo);
        if (capability.ImageConditioning) modes.Add(ImageToVideo);
        // A checkpoint that pins no frames (MiniMax-H3 Ref2VA) has no last frame to pin either.
        if (capability.ImageConditioning && capability.LastFrameConditioning) modes.Add(FirstLastFrame);
        if (capability.TemporalGuidance) modes.Add(Keyframes);
        if (capability.IdentityReferences && capability.MaxReferences > 0) modes.Add(Reference);
        return modes.ToArray();
    }

    /// <summary>The mode a request without an explicit one implies. Requests saved before
    /// modes were explicit carry only their inputs; this matches how they ran then.</summary>
    public static string Infer(StudioGenerationRequest request) =>
        request.ReferenceAssetIds.Count > 0 ? Reference
        : request.TemporalConditions.Count > 0 ? Keyframes
        : request.EndConditioningAssetId != null ? FirstLastFrame
        : request.ConditioningAssetId != null ? ImageToVideo
        : TextToVideo;

    /// <summary>Normalizes an explicit mode in place and returns the effective mode after
    /// checking that the request supplies exactly that mode's inputs. Model-independent;
    /// admission separately checks the mode against the loaded checkpoint.</summary>
    public static string Resolve(StudioGenerationRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        if (request.Mode != null)
        {
            request.Mode = request.Mode.Trim().ToLowerInvariant();
            if (!All.Contains(request.Mode))
                throw new StudioException("invalid_mode", "Generation mode must be t2v, i2v, fl2v, keyframes or ref.");
        }
        string mode = request.Mode ?? Infer(request);
        bool first = request.ConditioningAssetId != null, last = request.EndConditioningAssetId != null;
        int guides = request.TemporalConditions.Count, references = request.ReferenceAssetIds.Count;
        string? error = mode switch
        {
            TextToVideo when first || last || guides > 0 || references > 0 =>
                "Text-to-video (t2v) takes no images. Choose i2v, fl2v, keyframes or ref to use them.",
            ImageToVideo when !first => "Image-to-video (i2v) needs a first-frame image to animate.",
            ImageToVideo when last || guides > 0 || references > 0 =>
                "Image-to-video (i2v) uses only a first-frame image. Choose fl2v to pin the last frame, or ref for identity references.",
            FirstLastFrame when !last => "First-and-last-frame (fl2v) pins a last-frame image; with only a first frame, choose i2v.",
            FirstLastFrame when guides > 0 || references > 0 =>
                "First-and-last-frame (fl2v) pins only the first and last frames. Choose keyframes for timed guides, or ref for identity references.",
            Keyframes when guides == 0 => "Keyframes mode needs at least one timed image guide.",
            Keyframes when references > 0 => "Identity references cannot be combined with temporal image guidance.",
            Reference when references == 0 => "Reference mode (ref) needs at least one identity/appearance reference image.",
            Reference when first || last || guides > 0 =>
                "Reference images are not frames: ref mode cannot also pin a first, last or timed frame.",
            _ => null,
        };
        if (error != null) throw new StudioException("invalid_mode_inputs", error);
        if (request.ReferenceAssetIds.Distinct(StringComparer.Ordinal).Count() != references)
            throw new StudioException("invalid_mode_inputs", "Each reference image can be selected once.");
        return mode;
    }
}
