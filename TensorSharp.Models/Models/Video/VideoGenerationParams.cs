// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Model-agnostic sampling parameters for video generation. Every video model in
// TensorSharp takes these; a model ignores the knobs that do not apply to it (Wan
// has no audio stream, MiniMax-H3 has no second expert) and documents its own
// defaults for the ones set to 0/null.
using System;
using System.Collections.Generic;
using TensorSharp.Models.QwenImage;

namespace TensorSharp.Models.Video
{
    /// <summary>Sampling parameters for a video-generation request.</summary>
    public class VideoGenerationParams
    {
        /// <summary>Cooperative cancellation for models that support it. Passed to
        /// encoder/decoder work so cancellation follows a dedicated backend worker.</summary>
        [System.Text.Json.Serialization.JsonIgnore]
        public System.Threading.CancellationToken CancellationToken { get; set; }

        /// <summary>Output width in pixels, rounded to the model's grid (16 px for most
        /// Wan checkpoints, 32 for Wan2.2-TI2V and MiniMax-H3). 0 = the model's default,
        /// or the conditioning image's aspect for image-to-video.</summary>
        public int Width { get; set; }

        /// <summary>Output height in pixels, rounded to the model's grid. 0 = the model's
        /// default, or the conditioning image's aspect for image-to-video.</summary>
        public int Height { get; set; }

        /// <summary>How the canvas is sized from a keyframe when <see cref="Width"/> and
        /// <see cref="Height"/> are both 0. A host chooses this, not a request: it trades
        /// keyframe detail against time and memory. Models without keyframe sizing ignore it.</summary>
        public VideoCanvasPolicy CanvasPolicy { get; set; }

        /// <summary>Number of output frames, snapped to the model's temporal grid (4k+1 for
        /// Wan, 17k+5 for MiniMax-H3). 0 = the model's default. 1 = a still image, where
        /// the model supports it.</summary>
        public int Frames { get; set; }

        /// <summary>Denoising steps. 0 = the model's default (30 for Wan 2.1, 40 for
        /// Wan2.2-A14B, 50 for Wan2.2-TI2V, 20 for MiniMax-H3; step-distilled checkpoints use
        /// far fewer). A step-distillation LoRA plug-in fixes the count (8 or 4 for MiniMax-H3's
        /// Turbo plug-ins, <see cref="IVideoGenerationModel.SamplingContract"/>): leave it 0 or
        /// pass that count; any other value is refused.</summary>
        public int Steps { get; set; }

        /// <summary>Classifier-free guidance scale. 0 = the model's default; 1.0 disables the
        /// negative pass entirely. CFG-distilled models (MiniMax-H3) require 1.0.</summary>
        public float CfgScale { get; set; }

        /// <summary>Guidance scale for the second expert on dual-expert models
        /// (Wan2.2-A14B's low-noise phase). 0 = the model's default. Ignored by
        /// single-expert models.</summary>
        public float CfgScale2 { get; set; }

        /// <summary>RNG seed; &lt; 0 picks a random seed.</summary>
        public long Seed { get; set; } = -1;

        /// <summary>FlowMatch timestep shift. 0 = the model's default (Wan 2.1's 3.0/5.0/8.0
        /// recipe, 5.0 or 12.0 for Wan 2.2, 12.0 for MiniMax-H3's video stream). On models
        /// with a separate audio stream this shifts the video stream only. A LoRA plug-in's
        /// sampling recipe fixes it (6.0 under MiniMax-H3's Turbo plug-ins): leave it 0 or pass
        /// that value; any other value is refused.</summary>
        public float FlowShift { get; set; }

        /// <summary>Negative prompt; null = the model's default negative prompt. Unused when
        /// the guidance scale is 1.0, since no negative pass runs.</summary>
        public string NegativePrompt { get; set; }

        /// <summary>Sampler name, e.g. "unipc", "euler" or LTX 2.5's "euler_ancestral". null = the model's default.
        /// A LoRA plug-in's sampling recipe fixes it (Euler under MiniMax-H3's Turbo plug-ins): leave it null
        /// or pass that sampler; any other is refused.</summary>
        public string Sampler { get; set; }

        /// <summary>Playback frame rate for the saved video. 0 = the model's default. Models
        /// trained at a fixed rate (MiniMax-H3 at 24 fps) override any other value.</summary>
        public int Fps { get; set; }

        /// <summary>Conditioning image for image-to-video: becomes the first frame of the
        /// generated video, with the text prompt controlling motion, camera and scene
        /// evolution. Highest precedence of the three image inputs.</summary>
        public RgbImage Image { get; set; }

        /// <summary>Conditioning image as an encoded file (PNG/JPEG/WebP/...).</summary>
        public byte[] ImageBytes { get; set; }

        /// <summary>Conditioning image as a file path.</summary>
        public string ImagePath { get; set; }

        /// <summary>Last-frame conditioning image, on models that accept one (LTX or
        /// MiniMax-H3 FL2VA). Alone it pins only the last frame; combined with
        /// <see cref="Image"/> the generation is steered to start and end on the two
        /// supplied frames. A missing frame is never replaced by a blank image. Wan
        /// refuses it.</summary>
        public RgbImage EndImage { get; set; }

        /// <summary>Last-frame conditioning image as a file path.</summary>
        public string EndImagePath { get; set; }

        /// <summary>Images anchored at explicit output pixel-frame indices. LTX uses
        /// frame zero as a first-frame latent and other frames as appended temporal
        /// guides. An index of -1 selects the last output frame.</summary>
        public IList<VideoImageCondition> ImageConditions { get; set; }

        /// <summary>Reference images for reference-conditioned models. MiniMax-H3
        /// refers to them positionally ("the cat from &lt;Picture 1&gt;"). LTX treats
        /// this list as ordered temporal image guides, evenly spaced first to last;
        /// use ImageConditions to select their exact positions and strengths.</summary>
        public IList<string> ReferenceImagePaths { get; set; }

        /// <summary>Reference video clips for reference-conditioned models.</summary>
        public IList<string> ReferenceVideoPaths { get; set; }

        /// <summary>Reference audio clips for reference-conditioned models.</summary>
        public IList<string> ReferenceAudioPaths { get; set; }

        /// <summary>Soundtracks paired BY INDEX with <see cref="ReferenceVideoPaths"/>.
        /// A clip and its audio arrive separately because a video container's audio
        /// track is not readable through the frame decoder; entry i is the soundtrack
        /// of reference video i, and a null or missing entry means that clip is silent.</summary>
        public IList<string> ReferenceVideoAudioPaths { get; set; }

        /// <summary>How to interpret the supplied images, for models that offer more
        /// than one conditioning mode. null = infer from what was supplied.
        /// <para>MiniMax-H3 accepts: <c>t2v</c> (text only), <c>i2v</c> (the image is
        /// the FIRST FRAME and is animated), <c>fl2v</c> (the last frame, optionally with
        /// the first frame), and
        /// <c>ref</c> (the images are identity/appearance references for a new scene,
        /// not the first frame).</para>
        /// <para>The distinction matters: "animate this photo" is <c>i2v</c>, while
        /// "put this person somewhere new" is <c>ref</c>, and they use different
        /// checkpoints.</para>
        /// <para>LTX accepts t2v/t2va, i2v, fl2v and keyframes. Its stock distilled
        /// preset uses temporal image guides; identity mode ref needs a separate
        /// Ingredients IC-LoRA workflow.</para></summary>
        public string Mode { get; set; }

        /// <summary>Decode the audio stream on models that generate audio jointly with video.
        /// Ignored by video-only models. Setting this false skips the audio VAE, which saves
        /// time and memory when only the picture is wanted.</summary>
        public bool GenerateAudio { get; set; } = true;

        /// <summary>High/low-noise expert boundary as a fraction of the 1000-step timestep
        /// range, on dual-expert models (Wan2.2-A14B). 0 = the model's default.</summary>
        public float BoundaryRatio { get; set; }

        /// <summary>
        /// Classifier-free guidance cache stride. Every step normally costs TWO diffusion
        /// passes (conditional + unconditional). Writing the update as
        /// <c>v = v_cond + (cfg-1) * d</c> with <c>d = v_cond - v_uncond</c> isolates
        /// the guidance direction <c>d</c>, which changes far more slowly across the
        /// schedule than <c>v</c> itself. With a stride of N the unconditional pass
        /// runs on one step in N (plus a warm-up and the final step) and the cached
        /// <c>d</c> covers the rest. At the 50-step TI2V recipe, stride 2 runs 77 of
        /// the 100 passes (1.30x faster) and stride 3 runs 70 (1.43x).
        /// <para>0 or 1 = off (default): every step runs both passes exactly as the
        /// reference pipelines do. This is an approximation — leave it off when
        /// matching a reference sample matters. It has no effect at guidance 1.0,
        /// where there is no unconditional pass to skip.</para>
        /// </summary>
        public int CfgCacheStride { get; set; }

        /// <summary>Per-step progress callback: (stepIndex, totalSteps).</summary>
        public Action<int, int> OnStep { get; set; }

        /// <summary>
        /// Fine-grained progress. Fires on every phase transition AND on a periodic
        /// heartbeat while a single diffusion pass is running, so a caller can show that
        /// work is advancing during the minutes one 720p/121-frame pass takes.
        /// </summary>
        public Action<VideoGenerationProgress> OnProgress { get; set; }

        /// <summary>Resolve the conditioning image from whichever input was supplied.</summary>
        internal RgbImage ResolveImage()
        {
            if (Image != null) return Image;
            if (ImageBytes is { Length: > 0 }) return QwenImage.ImageIO.Decode(ImageBytes);
            if (!string.IsNullOrWhiteSpace(ImagePath)) return QwenImage.ImageIO.Load(ImagePath);
            return null;
        }

        /// <summary>Resolve the last-frame conditioning image, if one was supplied.</summary>
        internal RgbImage ResolveEndImage()
        {
            if (EndImage != null) return EndImage;
            if (!string.IsNullOrWhiteSpace(EndImagePath)) return QwenImage.ImageIO.Load(EndImagePath);
            return null;
        }
    }

    /// <summary>How an image-anchored video model sizes its canvas when the request gives
    /// no width or height.
    ///
    /// <para>First- and last-frame modes keep the keyframe's framing, so the canvas decides
    /// how many pixels a face or a hand in that frame gets. MiniMax-H3's video VAE downsamples
    /// 16x in each direction: a full-body photo fitted to 640x384 leaves a face about 60 px
    /// tall, and an encode then decode alone already smears it, so no number of denoising
    /// steps can bring it back. The same photo survives at MiniMax's released 768-pixel
    /// recipe.</para></summary>
    public enum VideoCanvasPolicy
    {
        /// <summary>The model's released recipe. MiniMax-H3 i2v/fl2v: the first keyframe's
        /// aspect at a 768-pixel short edge
        /// (<see cref="MiniMaxH3.MiniMaxH3Geometry.KeyframeCanvas"/>).</summary>
        ModelRecipe = 0,

        /// <summary>A smaller canvas for hosts whose memory or time budget has not been
        /// measured at the recipe size. MiniMax-H3 i2v/fl2v: a 640x384 area at the
        /// keyframe's aspect. Faces that are small in the keyframe do not survive it.</summary>
        CompactPreview = 1,
    }

    /// <summary>A temporal image guide, with strength in [0,1].</summary>
    public sealed class VideoImageCondition
    {
        public RgbImage Image { get; set; }
        public byte[] ImageBytes { get; set; }
        public string ImagePath { get; set; }
        public int FrameIndex { get; set; }
        public float Strength { get; set; } = 1;

        internal RgbImage ResolveImage(bool preserveAlpha = false) => Image ??
            (ImageBytes is { Length: > 0 } ? QwenImage.ImageIO.Decode(ImageBytes, preserveAlpha) :
                !string.IsNullOrWhiteSpace(ImagePath) ? QwenImage.ImageIO.Load(ImagePath, preserveAlpha) :
                throw new ArgumentException("Each image condition requires an image."));
    }
}
