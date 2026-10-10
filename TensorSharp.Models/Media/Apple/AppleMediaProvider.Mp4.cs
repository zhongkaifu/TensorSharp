// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MP4 writing for generated video, through AVAssetWriter + VideoToolbox.
//
// There is no preference ladder here, unlike the desktop provider's ffmpeg -> OpenCV avc1 ->
// OpenCV mp4v: two of those rungs cannot exist on a phone (no spawned processes, no
// ios-arm64 OpenCV) and the third is the same hardware encoder this goes to. So it is one
// path that either produces browser-playable H.264 or throws.
#if IOS || MACCATALYST
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using AVFoundation;
using CoreMedia;
using CoreVideo;
using Foundation;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.Video;

#nullable enable

namespace TensorSharp.Models.Media.Apple;

public sealed partial class AppleMediaProvider
{
    /// <summary>Bits per pixel per second asked of the encoder. VideoToolbox has no CRF mode,
    /// so quality has to be requested as a rate; 0.6 bpp is generous for the smooth, synthetic
    /// frames a diffusion model produces (a phone camera records 1080p30 at about 0.28 bpp) and
    /// is the closest this can come to the desktop path's <c>-crf 17</c>.</summary>
    private const double BitsPerPixel = 0.6;

    /// <summary>Floor and ceiling for the derived bit rate: enough that a postage-stamp clip is
    /// not wrecked by a tiny budget, capped so a large one does not ask for a rate the encoder
    /// will refuse.</summary>
    private const int MinBitRate = 1_000_000;
    private const int MaxBitRate = 60_000_000;

    /// <summary>How long to wait for the encoder to drain before giving up on a frame. The
    /// input goes not-ready when VideoToolbox's queue is full; on a phone that clears in
    /// milliseconds, so a minute means something is genuinely wrong.</summary>
    private static readonly TimeSpan AppendTimeout = TimeSpan.FromSeconds(60);

    /// <summary>AAC rate for the soundtrack of a generated clip. 64 kbit/s per channel is
    /// transparent for the 32 kHz audio a video model produces and is well inside what the
    /// encoder accepts at that sample rate.</summary>
    private const int AacBitRatePerChannel = 64_000;

    /// <summary>Write frames as an H.264 MP4 and return <c>"h264"</c>. <paramref name="path"/>
    /// is a full path whose directory exists (<see cref="TensorSharp.Models.WanVideo.VideoIO.SaveMp4(string, RgbImage[], int)"/>
    /// sees to both).</summary>
    public string SaveMp4(string path, RgbImage[] frames, int fps) => Write(path, frames, fps, null);

    /// <summary>Write frames as an H.264 MP4 with the soundtrack as an AAC track in the same
    /// file, so a player plays the clip with its sound and a shared file keeps it.
    ///
    /// <para>Should the audio half fail, the clip is written again without it rather than
    /// lost: <paramref name="audioMuxed"/> is then false, the reason goes to stderr, and the
    /// caller's sidecar WAV still carries the sound.</para></summary>
    public string SaveMp4(string path, RgbImage[] frames, int fps, GeneratedVideoAudio? audio, out bool audioMuxed)
    {
        audioMuxed = false;
        if (audio is not { ChannelCount: 1 or 2, SampleCount: > 0, SampleRate: > 0 })
            return Write(path, frames, fps, null);
        // Frames that cannot be encoded at all fail here, before the retry below could
        // blame the soundtrack for them.
        CheckEncodable(path, frames);
        try
        {
            string codec = Write(path, frames, fps, audio);
            audioMuxed = true;
            return codec;
        }
        catch (Exception ex) when (ex is InvalidOperationException or TimeoutException)
        {
            Console.Error.WriteLine(
                $"[video] could not put the soundtrack inside '{Path.GetFileName(path)}' ({ex.Message}); " +
                "writing the frames alone.");
            return Write(path, frames, fps, null);
        }
    }

    /// <summary>Refuse frames no MP4 can hold, and return their size.</summary>
    private static (int Width, int Height) CheckEncodable(string path, RgbImage[] frames)
    {
        if (string.IsNullOrWhiteSpace(path)) throw new ArgumentNullException(nameof(path));
        if (frames == null || frames.Length == 0)
            throw new ArgumentException("no frames to save", nameof(frames));

        int width = frames[0].Width;
        int height = frames[0].Height;
        for (int i = 1; i < frames.Length; i++)
        {
            if (frames[i] == null || frames[i].Width != width || frames[i].Height != height)
            {
                throw new ArgumentException(
                    $"frame {i} is {frames[i]?.Width ?? 0}x{frames[i]?.Height ?? 0}, not {width}x{height}; " +
                    "an MP4 track has one frame size", nameof(frames));
            }
        }

        // H.264 in yuv420p subsamples chroma 2x2, so an odd dimension has no valid encoding.
        // Saying so beats letting VideoToolbox fail with a numeric OSStatus.
        if ((width & 1) != 0 || (height & 1) != 0)
        {
            throw new InvalidOperationException(
                $"Cannot encode a {width}x{height} MP4: H.264 4:2:0 needs even dimensions. Pad or crop the frames " +
                "to an even size before saving.");
        }
        return (width, height);
    }

    /// <summary>The smallest multiple of 600 (AVFoundation's default) that one frame at <paramref name="fps"/> divides.</summary>
    internal static int TrackTimeScale(int fps)
    {
        if (fps <= 0)
            return 600;
        int a = 600, b = fps;
        while (b != 0) (a, b) = (b, a % b);
        return 600 / a * fps;
    }

    private string Write(string path, RgbImage[] frames, int fps, GeneratedVideoAudio? audio)
    {
        var (width, height) = CheckEncodable(path, frames);
        if (fps <= 0) fps = 16;

        // AVAssetWriter refuses to start when the output URL already exists.
        if (File.Exists(path))
            File.Delete(path);

        // The binding hands the UTI back as an NSString that is nullable in theory only; the
        // literal is the documented value of AVFileTypeMPEG4 and keeps the call non-nullable.
        string fileType = (string?)AVFileTypes.Mpeg4.GetConstant() ?? "public.mpeg-4";
        string mediaType = (string?)AVMediaTypes.Video.GetConstant() ?? "vide";
        string audioType = (string?)AVMediaTypes.Audio.GetConstant() ?? "soun";

        using NSUrl url = NSUrl.FromFilename(path);
        AVAssetWriter? writer = AVAssetWriter.FromUrl(url, fileType, out NSError error);
        if (writer == null || error != null)
        {
            throw new InvalidOperationException(
                $"Could not create an MP4 writer at '{path}': {error?.LocalizedDescription ?? "unknown error"}");
        }

        // Built outside the try so the finally can always reach them.
        AVAssetWriterInput? audioInput = null;
        AVAudioFormat? pcm = null;
        try
        {
            // The index ('moov') goes at the front of the file instead of after the media, so
            // a web view can start a clip before the whole file has arrived and can seek
            // with range requests. The desktop encoder asks the same of ffmpeg (+faststart).
            writer.ShouldOptimizeForNetworkUse = true;

            int bitRate = (int)Math.Clamp((long)(width * (double)height * fps * BitsPerPixel), MinBitRate, MaxBitRate);
            var settings = new AVVideoSettingsCompressed
            {
                CodecType = AVVideoCodecType.H264,
                Width = width,
                Height = height,
                CodecSettings = new AVVideoCodecSettings
                {
                    AverageBitRate = bitRate,
                    // A keyframe a second keeps seeking usable in a browser without spending
                    // much of the budget on intra frames.
                    MaxKeyFrameInterval = fps,
                    ProfileLevelH264 = AVVideoProfileLevelH264.HighAutoLevel,
                },
            };

            using var input = new AVAssetWriterInput(mediaType, settings)
            {
                // Offline encode: let the writer apply back-pressure instead of dropping.
                ExpectsMediaDataInRealTime = false,
                // A track timescale the frame duration divides exactly. The default is
                // 600, which holds 1/24 s but not 1/16 s: Wan's 16 fps frames were stored
                // as 37/600 s each and the clip played at 16.216 fps, shorter than the turn
                // reported. lcm(fps, 600) keeps 600 for the rates it already suited.
                MediaTimeScale = TrackTimeScale(fps),
            };
            using var adaptor = new AVAssetWriterInputPixelBufferAdaptor(
                input,
                new CVPixelBufferAttributes
                {
                    PixelFormatType = CVPixelFormatType.CV32BGRA,
                    Width = (nint)width,
                    Height = (nint)height,
                });

            if (!writer.CanAddInput(input))
                throw new InvalidOperationException($"AVAssetWriter refused an H.264 {width}x{height} input.");
            writer.AddInput(input);

            if (audio != null)
            {
                audioInput = new AVAssetWriterInput(audioType, new AudioSettings
                {
                    Format = AudioToolbox.AudioFormatType.MPEG4AAC,
                    SampleRate = audio.SampleRate,
                    NumberChannels = audio.ChannelCount,
                    EncoderBitRate = AacBitRatePerChannel * audio.ChannelCount,
                })
                {
                    ExpectsMediaDataInRealTime = false,
                };
                if (!writer.CanAddInput(audioInput))
                {
                    throw new InvalidOperationException(
                        $"AVAssetWriter refused an AAC {audio.SampleRate} Hz x{audio.ChannelCount} input.");
                }
                writer.AddInput(audioInput);
                // What the PCM handed to the encoder is: the WAV writer's 16-bit samples,
                // interleaved, so the track inside the file and the sidecar beside it agree.
                pcm = new AVAudioFormat(AVAudioCommonFormat.PCMInt16, audio.SampleRate, (uint)audio.ChannelCount, true);
            }

            if (!writer.StartWriting())
            {
                throw new InvalidOperationException(
                    $"AVAssetWriter could not start writing '{path}': {writer.Error?.LocalizedDescription ?? writer.Status.ToString()}");
            }
            writer.StartSessionAtSourceTime(CMTime.Zero);

            long audioWritten = 0;
            for (int i = 0; i < frames.Length; i++)
            {
                // Interleaved as it plays: the sound up to the end of this frame, then the
                // frame. Offline, the writer holds back whichever input runs too far ahead
                // of the other, so appending all of one track first would stall it.
                if (audioInput != null && audio != null && pcm != null)
                {
                    long until = Math.Min(audio.SampleCount, (long)Math.Round((i + 1) * (double)audio.SampleRate / fps));
                    audioWritten = AppendAudio(audioInput, writer, pcm, audio, audioWritten, until);
                }
                WaitForInput(input, writer, $"frame {i}");

                // A fresh buffer per frame: the adaptor hands it to VideoToolbox and may still
                // be holding it when the next append is made, so reuse would be a data race.
                using CVPixelBuffer pixelBuffer = CreateBgraBuffer(frames[i], width, height);
                if (!adaptor.AppendPixelBufferWithPresentationTime(pixelBuffer, new CMTime(i, fps)))
                {
                    throw new InvalidOperationException(
                        $"AVAssetWriter rejected frame {i} of '{path}': {writer.Error?.LocalizedDescription ?? writer.Status.ToString()}");
                }
            }

            if (audioInput != null && audio != null && pcm != null)
            {
                AppendAudio(audioInput, writer, pcm, audio, audioWritten, audio.SampleCount);
                audioInput.MarkAsFinished();
            }
            input.MarkAsFinished();
            writer.EndSessionAtSourceTime(new CMTime(frames.Length, fps));
            FinishWriting(writer, path);
            return "h264";
        }
        catch
        {
            if (writer.Status == AVAssetWriterStatus.Writing)
                writer.CancelWriting();
            // A half-written file plays as nothing; do not leave one behind pretending to be output.
            try { if (File.Exists(path)) File.Delete(path); } catch { /* best effort */ }
            throw;
        }
        finally
        {
            audioInput?.Dispose();
            pcm?.Dispose();
            writer.Dispose();
        }
    }

    /// <summary>Append samples <paramref name="from"/>..<paramref name="until"/> of the
    /// soundtrack as one 16-bit interleaved PCM buffer, for the encoder to turn into AAC.
    /// Returns where the next append starts.</summary>
    private static long AppendAudio(
        AVAssetWriterInput input, AVAssetWriter writer, AVAudioFormat pcm,
        GeneratedVideoAudio audio, long from, long until)
    {
        int count = (int)(until - from);
        if (count <= 0)
            return from;

        int channels = audio.ChannelCount;
        byte[] bytes = new byte[count * channels * 2];
        int offset = 0;
        for (long i = from; i < until; i++)
        {
            for (int c = 0; c < channels; c++)
            {
                // The WavWriter expression exactly: clamp first, because a model overshoots
                // [-1, 1] now and then and a wrapped int16 is a loud click.
                float v = Math.Clamp(audio.Channels[c][i], -1f, 1f);
                short s = (short)Math.Round(v * short.MaxValue, MidpointRounding.AwayFromZero);
                bytes[offset++] = (byte)(s & 0xFF);
                bytes[offset++] = (byte)((s >> 8) & 0xFF);
            }
        }

        // Memory CoreMedia allocates and owns, with the samples copied in. The encoder reads a
        // buffer after AppendSampleBuffer has returned, on its own thread, and the convenience
        // overload that wraps a managed array lets the array go when its wrapper is disposed:
        // MEASURED, the AAC track then held a 5 kHz whine at a fifth of the tone's level, a
        // different one on every run, instead of the samples that were handed over.
        using CMBlockBuffer? block = CMBlockBuffer.FromMemoryBlock(
            IntPtr.Zero, (nuint)bytes.Length, null, 0, (nuint)bytes.Length,
            CMBlockBufferFlags.AssureMemoryNow, out CMBlockBufferError blockError);
        if (block == null || blockError != CMBlockBufferError.None)
            throw new InvalidOperationException($"CMBlockBufferCreateWithMemoryBlock failed ({blockError}) for the soundtrack.");
        CMBlockBufferError copyError = block.ReplaceDataBytes(bytes, 0);
        if (copyError != CMBlockBufferError.None)
            throw new InvalidOperationException($"CMBlockBufferReplaceDataBytes failed ({copyError}) for the soundtrack.");
        using CMSampleBuffer? buffer = CMSampleBuffer.CreateReadyWithPacketDescriptions(
            block, pcm.FormatDescription, count, new CMTime(from, audio.SampleRate), null, out CMSampleBufferError bufferError);
        if (buffer == null || bufferError != CMSampleBufferError.None)
            throw new InvalidOperationException($"CMAudioSampleBufferCreate failed ({bufferError}) for the soundtrack.");

        WaitForInput(input, writer, $"the soundtrack at sample {from}");
        if (!input.AppendSampleBuffer(buffer))
        {
            throw new InvalidOperationException(
                $"AVAssetWriter rejected the soundtrack at sample {from}: {writer.Error?.LocalizedDescription ?? writer.Status.ToString()}");
        }
        return until;
    }

    /// <summary>
    /// Flush the encoder and close the container, then block until it is done.
    ///
    /// <para>The completion-handler form, not the synchronous <c>finishWriting</c> that has
    /// been deprecated since iOS 6 (it can block the caller for the length of the flush with no
    /// way to observe failure). AVFoundation runs the handler on its own queue, so waiting on
    /// it here cannot deadlock even when a caller happens to be on the UI thread — and
    /// <see cref="SaveMp4"/> is a synchronous API, so somebody has to wait.</para>
    /// </summary>
    private static void FinishWriting(AVAssetWriter writer, string path)
    {
        using var done = new ManualResetEventSlim(false);
        writer.FinishWriting(() => done.Set());
        if (!done.Wait(AppendTimeout))
            throw new TimeoutException($"The H.264 encoder did not finish '{path}' within {AppendTimeout.TotalSeconds:0} s.");

        if (writer.Status != AVAssetWriterStatus.Completed)
        {
            throw new InvalidOperationException(
                $"AVAssetWriter could not finish '{path}': {writer.Error?.LocalizedDescription ?? writer.Status.ToString()}");
        }
    }

    /// <summary>Block until the encoder will take another frame. Polling rather than
    /// <c>RequestMediaData</c> because <see cref="SaveMp4"/> is a synchronous API called from
    /// a pipeline thread that has nothing else to do; the callback form would need a dispatch
    /// queue and a completion handshake to end up in exactly the same place.</summary>
    private static void WaitForInput(AVAssetWriterInput input, AVAssetWriter writer, string what)
    {
        DateTime deadline = DateTime.UtcNow + AppendTimeout;
        while (!input.ReadyForMoreMediaData)
        {
            if (writer.Status is AVAssetWriterStatus.Failed or AVAssetWriterStatus.Cancelled)
            {
                throw new InvalidOperationException(
                    $"AVAssetWriter stopped at {what}: {writer.Error?.LocalizedDescription ?? writer.Status.ToString()}");
            }
            if (DateTime.UtcNow > deadline)
                throw new TimeoutException($"The encoder did not accept {what} within {AppendTimeout.TotalSeconds:0} s.");
            Thread.Sleep(1);
        }
    }

    /// <summary>
    /// Turn one HWC RGB float frame into a BGRA CoreVideo buffer.
    ///
    /// <para>The float-to-byte rounding is deliberately the same expression the desktop
    /// provider uses (<c>(int)(v * 255 + 0.5)</c>, clamped), so the two encoders are fed
    /// identical pixels and any difference in the resulting file is the encoder's, not a
    /// quantisation difference nobody would think to look for.</para>
    /// </summary>
    private static CVPixelBuffer CreateBgraBuffer(RgbImage frame, int width, int height)
    {
        var pixelBuffer = new CVPixelBuffer(
            (nint)width, (nint)height, CVPixelFormatType.CV32BGRA,
            new CVPixelBufferAttributes
            {
                // IOSurface-backed is what VideoToolbox wants; without it the encoder copies
                // every frame through a staging buffer.
                AllocateWithIOSurface = true,
                CGImageCompatibility = true,
                CGBitmapContextCompatibility = true,
            });

        if (pixelBuffer.Handle == IntPtr.Zero)
        {
            pixelBuffer.Dispose();
            throw new InvalidOperationException($"CVPixelBufferCreate failed for a {width}x{height} BGRA frame.");
        }

        CVReturn locked = pixelBuffer.Lock(CVPixelBufferLock.None);
        if (locked != CVReturn.Success)
        {
            pixelBuffer.Dispose();
            throw new InvalidOperationException($"CVPixelBufferLockBaseAddress failed ({locked}) while writing a frame.");
        }

        try
        {
            int stride = (int)pixelBuffer.BytesPerRow;
            IntPtr baseAddress = pixelBuffer.BaseAddress;
            if (baseAddress == IntPtr.Zero || stride < width * 4)
                throw new InvalidOperationException($"CVPixelBuffer gave an unusable base address / stride {stride}.");

            float[] pixels = frame.Pixels;
            byte[] row = new byte[width * 4];
            for (int y = 0; y < height; y++)
            {
                int src = y * width * 3;
                for (int x = 0; x < width; x++)
                {
                    row[x * 4 + 0] = ToByte(pixels[src + x * 3 + 2]);   // B
                    row[x * 4 + 1] = ToByte(pixels[src + x * 3 + 1]);   // G
                    row[x * 4 + 2] = ToByte(pixels[src + x * 3 + 0]);   // R
                    row[x * 4 + 3] = 255;
                }
                Marshal.Copy(row, 0, baseAddress + y * stride, row.Length);
            }
        }
        catch
        {
            pixelBuffer.Unlock(CVPixelBufferLock.None);
            pixelBuffer.Dispose();
            throw;
        }

        pixelBuffer.Unlock(CVPixelBufferLock.None);
        return pixelBuffer;
    }

    private static byte ToByte(float value)
    {
        int i = (int)(value * 255f + 0.5f);
        return (byte)(i < 0 ? 0 : i > 255 ? 255 : i);
    }
}
#endif
