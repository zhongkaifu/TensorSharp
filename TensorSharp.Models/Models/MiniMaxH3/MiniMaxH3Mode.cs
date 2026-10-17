// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3 conditioning modes.
//
// The distinction that matters to a user is what an image MEANS:
//
//   * i2v  — "animate this photo". The image IS the first frame; the clip starts
//            exactly there and the prompt drives the motion.
//   * fl2v — "go from photo A to photo B". First and last frame are both pinned.
//   * ref  — "use this person/product, new scene". The image is an identity and
//            appearance reference; the first frame need NOT resemble it.
//
// i2v/fl2v need the FL2VA checkpoint and ref needs Ref2VA — they are separate
// files, not settings, so choosing the wrong one has to be reported rather than
// silently producing something the user did not ask for.
using System;
using TensorSharp.Models.Video;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>How MiniMax-H3 should interpret the supplied images.</summary>
    public enum MiniMaxH3Mode
    {
        /// <summary>Text only.</summary>
        TextToVideo,
        /// <summary>The image is the first frame and is animated.</summary>
        ImageToVideo,
        /// <summary>First and last frame are both pinned.</summary>
        FirstLastFrame,
        /// <summary>The images/videos/audio are references for a new scene.</summary>
        Reference,
    }

    /// <summary>Picks the conditioning mode from the request, or honours an explicit one.</summary>
    public static class MiniMaxH3ModeResolver
    {
        /// <summary>Parse an explicit mode string. Accepts the short forms users type
        /// as well as the names the model card uses.</summary>
        public static MiniMaxH3Mode? Parse(string mode)
        {
            if (string.IsNullOrWhiteSpace(mode)) return null;
            return mode.Trim().ToLowerInvariant() switch
            {
                "t2v" or "t2va" or "text" or "text-to-video" => MiniMaxH3Mode.TextToVideo,
                "i2v" or "i2va" or "image" or "image-to-video" or "first-frame"
                    => MiniMaxH3Mode.ImageToVideo,
                "fl2v" or "fl2va" or "firstlast" or "first-last" or "first-last-frame"
                    => MiniMaxH3Mode.FirstLastFrame,
                "ref" or "ref2v" or "ref2va" or "reference" => MiniMaxH3Mode.Reference,
                _ => throw new ArgumentException(
                    $"unknown video mode '{mode}'. Use t2v, i2v, fl2v or ref.", nameof(mode)),
            };
        }

        /// <summary>Resolve the mode for a request, validating it against the loaded
        /// checkpoint. An explicit mode wins; otherwise it is inferred from what was
        /// supplied.</summary>
        public static MiniMaxH3Mode Resolve(VideoGenerationParams p, MiniMaxH3Partition partition)
        {
            bool hasFirst = p.Image != null || p.ImageBytes is { Length: > 0 }
                            || !string.IsNullOrWhiteSpace(p.ImagePath);
            bool hasLast = p.EndImage != null || !string.IsNullOrWhiteSpace(p.EndImagePath);
            bool hasRefs = (p.ReferenceImagePaths?.Count ?? 0) > 0
                           || (p.ReferenceVideoPaths?.Count ?? 0) > 0
                           || (p.ReferenceAudioPaths?.Count ?? 0) > 0;

            // A plain image on the Ref2VA checkpoint means a reference, not a first
            // frame. Clients that only know how to attach "an image" — the Web UI
            // among them — would otherwise be told to load a different model, when
            // the operator already chose the one they wanted.
            MiniMaxH3Mode? explicitMode = Parse(p.Mode);
            bool imageIsReference = partition == MiniMaxH3Partition.Reference
                                    && explicitMode == null && hasFirst && !hasLast && !hasRefs;

            MiniMaxH3Mode mode = explicitMode
                ?? (hasRefs || imageIsReference ? MiniMaxH3Mode.Reference
                    : hasLast ? MiniMaxH3Mode.FirstLastFrame
                    : hasFirst ? MiniMaxH3Mode.ImageToVideo
                    : MiniMaxH3Mode.TextToVideo);

            Validate(mode, partition, hasFirst, hasLast,
                     hasRefs || imageIsReference, namedReferences: hasRefs);
            return mode;
        }

        /// <param name="hasRefs">Anything is being treated as a reference, including a
        /// plain image on the Ref2VA checkpoint.</param>
        /// <param name="namedReferences">The request actually NAMED reference inputs.
        /// Distinct from <paramref name="hasRefs"/> because a bare --image becoming a
        /// reference is an accommodation, while --image alongside --ref-image is a
        /// contradiction.</param>
        private static void Validate(MiniMaxH3Mode mode, MiniMaxH3Partition partition,
                                     bool hasFirst, bool hasLast, bool hasRefs,
                                     bool namedReferences)
        {
            bool wantsReference = mode == MiniMaxH3Mode.Reference;
            bool haveReference = partition == MiniMaxH3Partition.Reference;
            if (wantsReference && !haveReference)
                throw new InvalidOperationException(
                    "reference conditioning needs the Ref2VA checkpoint " +
                    "(minimax_h3_ref2va_pruned-*.gguf); the loaded model is FL2VA. The two are " +
                    "separate checkpoints, not settings — load the other file, or drop the " +
                    "reference inputs to animate the image as a first frame instead (--video-mode i2v).");
            if (!wantsReference && haveReference
                && (mode is MiniMaxH3Mode.ImageToVideo or MiniMaxH3Mode.FirstLastFrame
                    || hasFirst || hasLast))
                throw new InvalidOperationException(
                    "keyframe conditioning needs the FL2VA checkpoint " +
                    "(minimax_h3_fl2va_pruned-*.gguf); the loaded model is Ref2VA, which treats " +
                    "images as references rather than frames. Load the other file, or pass the " +
                    "image with --ref-image to use it as a reference (--video-mode ref).");

            if (mode == MiniMaxH3Mode.ImageToVideo && !hasFirst)
                throw new InvalidOperationException(
                    "--video-mode i2v needs a conditioning image. Upload an image or pass --image.");
            // i2v pins only the first frame; an end image would otherwise be pinned too, unasked.
            if (mode == MiniMaxH3Mode.ImageToVideo && hasLast)
                throw new InvalidOperationException(
                    "--video-mode i2v animates only a first frame; use --video-mode fl2v to also " +
                    "(or only) pin the last frame with --end-image.");
            if (mode == MiniMaxH3Mode.FirstLastFrame && !(hasFirst || hasLast))
                throw new InvalidOperationException(
                    "--video-mode fl2v needs a first and/or last conditioning image. " +
                    "Upload an image or pass --image and/or --end-image.");
            if (mode == MiniMaxH3Mode.Reference && !hasRefs)
                throw new InvalidOperationException(
                    "--video-mode ref needs at least one --ref-image / --ref-video / --ref-audio.");
            // The reference implementation refuses this outright rather than picking a
            // winner: "Ref2VA cannot be combined with --init-img or --end-img in one
            // request." Quietly dropping the keyframe is the worse failure, because the
            // clip that comes back looks like the request was honoured.
            if (mode == MiniMaxH3Mode.Reference && namedReferences && (hasFirst || hasLast))
                throw new InvalidOperationException(
                    "MiniMax-H3 cannot take keyframes and references in one request: " +
                    "--image/--end-image pin a frame the clip must reproduce, and a reference " +
                    "is deliberately not that. Drop the keyframe to keep the references, or " +
                    "drop the references and load the FL2VA checkpoint to animate the image.");
            if (mode is MiniMaxH3Mode.TextToVideo && (hasFirst || hasLast || hasRefs))
                throw new InvalidOperationException(
                    "--video-mode t2v was requested but images were supplied; drop them or pick " +
                    "i2v (animate the image as the first frame) or ref (use it as a reference).");
        }
    }
}
