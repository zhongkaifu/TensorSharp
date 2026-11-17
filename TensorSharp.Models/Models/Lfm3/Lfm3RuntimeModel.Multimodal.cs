// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;

using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// LFM3-Omni's media input on the shared runtime: the vision tower (<c>lfm3vl</c>) and the audio tower (<c>lfm3a</c>)
    /// load from the ONE Omni mmproj (the injector hands the same path to both loaders; the speech heads in that file are
    /// the speech output's), their rows enter the forward through <see cref="StandardDecoderModel.QueueMedia"/>, and the
    /// injector lays the prompt out (<see cref="ModelMultimodalInjector.ProcessLfm3History"/>, llama.cpp's mtmd layout).
    /// LFM3-1.7B, or Omni without its mmproj, has neither tower: an image, a video or an audio clip is refused with a
    /// message naming the projector, never answered without.
    /// <para>Encoding keeps the model's GPU lock only for device work: decoding a file, resizing frames and computing
    /// audio features run with it released (<see cref="ModelBase.RunOutsideGpuComputeLock{T}"/>), and a video yields it
    /// between frame pairs, a long clip between its 30 s audio chunks, so concurrent requests keep stepping.</para>
    /// </summary>
    public partial class Lfm3RuntimeModel : IVisionCapableModel, IAudioCapableModel, IAudioEncoderLoader, IMultimodalPromptExpander
    {
        /// <summary>The refusal of an image or a video without the vision tower.</summary>
        public const string VisionInputUnsupportedMessage =
            "This LFM3 model has no vision encoder loaded: image and video input need LFM3-Omni-2B with its projector " +
            "(--mmproj mmproj-LFM3-Omni-2B-Q8_0.gguf). LFM3-1.7B is text only. Remove the attachment or load LFM3-Omni " +
            "with its mmproj.";

        private static int s_soundtrackWarnings;

        /// <summary>The vision tower, or null (no mmproj, or an mmproj without one).</summary>
        public Lfm3VisionEncoder VisionEncoder { get; private set; }

        /// <summary>The audio tower, or null (no mmproj, or an mmproj without one).</summary>
        public Lfm3AudioEncoder AudioEncoder { get; private set; }

        bool IVisionCapableModel.IsVisionEncoderLoaded => VisionEncoder != null;

        /// <summary>Where media rows are built (the towers' and the decoder's allocator).</summary>
        internal IAllocator MediaAllocator => _allocator;

        /// <summary>Whether the audio tower is loaded (<see cref="AudioInputSupport"/> refuses audio otherwise).</summary>
        public bool IsAudioEncoderLoaded => AudioEncoder != null;

        /// <summary>Load the vision tower of an Omni mmproj. A projector that carries no vision tower (an audio-only one)
        /// leaves images refused, and says so; one of another family's type is refused by the encoder.</summary>
        public void LoadVisionEncoder(string mmProjPath)
        {
            ArgumentException.ThrowIfNullOrEmpty(mmProjPath);
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            using var gguf = new GgufFile(mmProjPath);
            if (!gguf.GetBool("clip.has_vision_encoder"))
            {
                Console.WriteLine($"lfm3: '{Path.GetFileName(mmProjPath)}' has no vision tower; images and video are refused.");
                return;
            }
            VisionEncoder = new Lfm3VisionEncoder(gguf, _allocator, _backend);
            Console.WriteLine($"lfm3: vision tower from '{Path.GetFileName(mmProjPath)}': {VisionEncoder.BlockCount} layers of " +
                $"{VisionEncoder.EmbeddingLength}, up to {VisionEncoder.ImageMaxPixels / (Lfm3ImageProcessor.Factor * Lfm3ImageProcessor.Factor)} rows an image, " +
                $"video at {Lfm3Video.SampleFps} fps up to {VisionEncoder.MaxVideoFrames} frames");
        }

        /// <summary>Load the audio tower of an Omni mmproj (null when the file has none: audio stays refused).</summary>
        public void LoadAudioEncoder(string mmProjPath)
        {
            ArgumentException.ThrowIfNullOrEmpty(mmProjPath);
            AudioEncoder?.Dispose();
            AudioEncoder = null;
            AudioEncoder = Lfm3AudioEncoder.Load(mmProjPath, _allocator, _backend);
            Console.WriteLine(AudioEncoder == null
                ? $"lfm3: '{Path.GetFileName(mmProjPath)}' has no audio tower; audio input is refused."
                : $"lfm3: audio tower from '{Path.GetFileName(mmProjPath)}': {AudioEncoder.LayerCount} layers of {AudioEncoder.Width}, " +
                  "one row per 80 ms; videos are heard with their soundtrack");
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        public void SetAudioEmbeddings(Tensor embeddings, int insertPosition) => QueueMedia(embeddings, insertPosition);

        List<int> IMultimodalPromptExpander.ExpandMultimodalPrompt(
            ModelMultimodalInjector injector, List<ChatMessage> history, List<int> inputTokens)
            => injector.ProcessLfm3History(this, history, inputTokens);

        private void DisposeMediaTowers()
        {
            VisionEncoder?.Dispose();
            VisionEncoder = null;
            AudioEncoder?.Dispose();
            AudioEncoder = null;
        }

        // ================================================================ encoding (the injector's calls)

        private Lfm3VisionEncoder Vision => VisionEncoder ?? throw new NotSupportedException(VisionInputUnsupportedMessage);

        private Lfm3AudioEncoder Audio => AudioEncoder ?? throw new NotSupportedException(AudioInputSupport.Lfm3Message);

        /// <summary>An image file's rows [rows, hidden]: decoded and resized off the GPU lock, encoded under it.</summary>
        internal Tensor EncodeImageFile(string path)
        {
            Lfm3VisionEncoder vision = Vision;
            Lfm3RgbImage frame = RunOutsideGpuComputeLock(() =>
            {
                Lfm3RgbImage image = Lfm3ImageProcessor.LoadImage(path);
                (int w, int h) = vision.ImageSize(image.Width, image.Height);
                return Lfm3ImageProcessor.Resize(image, w, h);
            });
            return vision.EncodeFrame(frame).ToTensor(_allocator);
        }

        /// <summary>An audio file's rows [rows, hidden] (16 kHz mono features computed off the GPU lock).</summary>
        internal Tensor EncodeAudioFile(string path)
        {
            Lfm3AudioEncoder audio = Audio;
            (float[] features, int frames) = RunOutsideGpuComputeLock(() =>
            {
                float[] samples = Lfm3AudioPreprocessor.LoadAudio(path);
                float[] f = Lfm3AudioPreprocessor.ComputeFeatures(samples, out int n);
                return (f, n);
            });
            if (frames == 0)
                throw new InvalidDataException($"lfm3 audio: '{Path.GetFileName(path)}' holds no samples");
            return audio.ToTensor(audio.EncodeFeatures(features, frames, YieldGpuComputeLock));
        }

        /// <summary>
        /// A video file read the model's way: <see cref="Lfm3Video"/>'s 2 fps sampling (llama.cpp / HF), frames decoded and
        /// resized off the GPU lock, one program a frame pair with the lock yielded between pairs; and, with the audio tower
        /// loaded, its soundtrack encoded once and split after each pair (<see cref="Lfm3AudioPrompt.SplitSoundtrack"/>).
        /// A clip without an audio stream is its frames alone; a soundtrack that cannot be decoded is warned about (the
        /// model then sees the frames without hearing them).
        /// </summary>
        internal Lfm3EncodedVideo EncodeVideoFile(string path)
        {
            Lfm3VisionEncoder vision = Vision;
            Lfm3AudioEncoder audio = AudioEncoder;
            int maxFrames = vision.MaxVideoFrames;
            var read = RunOutsideGpuComputeLock(() =>
            {
                (Lfm3VideoPlan plan, Lfm3RgbImage[] frames) = Lfm3Video.Read(path, maxFrames);
                float[] features = null;
                int featureFrames = 0;
                IReadOnlyList<Lfm3AudioPrompt.SoundtrackPiece> pieces = Array.Empty<Lfm3AudioPrompt.SoundtrackPiece>();
                float[] track = audio != null ? ReadSoundtrack(path) : null;
                if (track != null)
                {
                    pieces = Lfm3AudioPrompt.SplitSoundtrack(plan.FrameIndices, plan.SourceFps, plan.SourceFrames, track.Length, out long trimmed);
                    if (trimmed > 0)
                        features = Lfm3AudioPreprocessor.ComputeFeatures(track.AsSpan(0, (int)trimmed), out featureFrames);
                }
                return (plan, frames, features, featureFrames, pieces);
            });
            Lfm3VideoEncoding encoding = vision.EncodeVideo(read.plan, read.frames, YieldGpuComputeLock);
            float[] rows = null;
            if (read.features != null && read.featureFrames > 0)
            {
                YieldGpuComputeLock();
                rows = audio.EncodeFeatures(read.features, read.featureFrames, YieldGpuComputeLock);
            }
            return new Lfm3EncodedVideo(encoding, rows != null ? read.pieces : Array.Empty<Lfm3AudioPrompt.SoundtrackPiece>(), rows,
                audio?.OutputWidth ?? 0);
        }

        /// <summary>
        /// A video of frames sampled by the host (an OpenAI <c>video_url</c> sampled at the client's rate, frames extracted
        /// at <c>VIDEO_SAMPLE_FPS</c>): the files decoded off the GPU lock, paired by <see cref="Lfm3Video.PlanFrames"/>
        /// (consecutive frames at most <see cref="QwenVideoFrames.MaxPairedFrameGapSeconds"/> apart share a temporal patch,
        /// labelled with their mean time; sparser frames each fill one) and resized together. No soundtrack: the frames carry
        /// none. Not the model's own sampling, which needs the file (<see cref="EncodeVideoFile"/>).
        /// </summary>
        internal Lfm3VideoEncoding EncodeVideoFrames(IReadOnlyList<string> framePaths, IReadOnlyList<double> seconds)
        {
            ArgumentNullException.ThrowIfNull(framePaths);
            ArgumentNullException.ThrowIfNull(seconds);
            if (framePaths.Count == 0 || framePaths.Count != seconds.Count)
                throw new ArgumentException($"lfm3 video: {framePaths.Count} frames with {seconds.Count} times");
            Lfm3VisionEncoder vision = Vision;
            var (plan, frames) = RunOutsideGpuComputeLock(() =>
            {
                var decoded = new Lfm3RgbImage[framePaths.Count];
                for (int i = 0; i < decoded.Length; i++)
                    decoded[i] = Lfm3ImageProcessor.LoadImage(framePaths[i]);
                Lfm3VideoPlan p = Lfm3Video.PlanFrames(decoded[0].Width, decoded[0].Height, seconds, QwenVideoFrames.MaxPairedFrameGapSeconds);
                for (int i = 0; i < decoded.Length; i++)
                    decoded[i] = Lfm3ImageProcessor.Resize(decoded[i], p.Width, p.Height);
                return (p, decoded);
            });
            return vision.EncodeVideo(plan, frames, YieldGpuComputeLock);
        }

        /// <summary>A video's soundtrack as 16 kHz mono, or null: no audio stream (the clip is silent), or a track the
        /// platform cannot decode (warned: the frames are seen, the sound is not heard).</summary>
        private static float[] ReadSoundtrack(string path)
        {
            try
            {
                return Lfm3AudioPreprocessor.LoadAudio(path);
            }
            catch (InvalidDataException ex) when (ex.Message.Contains("no audio stream", StringComparison.Ordinal)
                                                  || ex.Message.Contains("decodes to no audio samples", StringComparison.Ordinal))
            {
                return null;
            }
            catch (Exception ex) when (ex is NotSupportedException or InvalidDataException or TimeoutException or IOException)
            {
                if (System.Threading.Interlocked.Increment(ref s_soundtrackWarnings) <= 8)
                    Console.Error.WriteLine($"lfm3 video: the soundtrack of '{Path.GetFileName(path)}' could not be decoded ({ex.Message}); " +
                        "the model sees the video's frames without hearing it.");
                return null;
            }
        }
    }

    /// <summary>A video file's encoding for the prompt: the frame pairs, and the soundtrack's rows (null without one) with
    /// the piece that follows each pair.</summary>
    internal sealed class Lfm3EncodedVideo
    {
        public Lfm3EncodedVideo(Lfm3VideoEncoding video, IReadOnlyList<Lfm3AudioPrompt.SoundtrackPiece> pieces, float[] soundtrackRows, int rowWidth)
        {
            Video = video;
            Pieces = pieces;
            SoundtrackRows = soundtrackRows;
            RowWidth = rowWidth;
        }

        public Lfm3VideoEncoding Video { get; }

        public IReadOnlyList<Lfm3AudioPrompt.SoundtrackPiece> Pieces { get; }

        /// <summary>[rows][<see cref="RowWidth"/>] of the whole (trimmed) track, or null.</summary>
        public float[] SoundtrackRows { get; }

        public int RowWidth { get; }
    }
}
