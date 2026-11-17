// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The audio model families on the command line:
//
//   speech recognition   --model whisper.gguf --audio talk.mp3 [--language zh] [--translate]
//                        [--word-timestamps] [--output talk.srt|.vtt|.json|.txt]
//   speech synthesis     --model kokoro.gguf --prompt "text" [--voice zf_xiaoxiao] [--speed 1.0]
//                        [--output speech.wav]
//                        --model LFM3-Omni-2B.gguf --mmproj mmproj-LFM3-Omni-2B.gguf --tts --prompt "text"
//                        [--voice female|male] [--language en] [--seed 1234] [--speech-temperature 0.4]
//                        [--output speech.wav]   (a chat model that speaks needs --tts to synthesize)
//   spoken chat reply    --model LFM3-Omni-2B.gguf --mmproj ... --input question.txt --speech-out reply.wav
//                        [--speech-mode interleaved|audio] [--voice female|male]   (Program.RunInference)
//   music generation     --model ace-step.gguf --prompt "style tags" [--lyrics lyrics.txt|"[verse] ..."]
//                        [--duration 30] [--bpm 90] [--key "C major"] [--time-signature 4]
//                        [--language en] [--diffusion-steps 8] [--diffusion-seed 0] [--output song.wav]
using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Video;

namespace TensorSharp.Cli;

internal sealed class AudioCliOptions
{
    internal string Language { get; private set; }
    internal bool Translate { get; private set; }
    internal bool WordTimestamps { get; private set; }
    internal string Voice { get; private set; }
    internal float Speed { get; private set; } = 1f;
    internal string Lyrics { get; private set; }
    internal double Duration { get; private set; } = 30;
    internal int Bpm { get; private set; }
    internal string KeyScale { get; private set; }
    internal string TimeSignature { get; private set; }
    internal string Phonemes { get; private set; }
    internal bool NoAudioCodes { get; private set; }

    /// <summary>--tts: speak the prompt with the model's TTS (a chat model that speaks, LFM3-Omni, needs it).</summary>
    internal bool Tts { get; private set; }

    /// <summary>--speech-out FILE: a chat reply is spoken and its speech written to FILE (WAV).</summary>
    internal string SpeechOut { get; private set; }

    /// <summary>--speech-mode: interleaved (text and speech; default) or audio (speech only).</summary>
    internal TensorSharp.Models.Audio.SpeechReplyMode SpeechMode { get; private set; } = TensorSharp.Models.Audio.SpeechReplyMode.Interleaved;

    /// <summary>--speech-temperature: the speech sampler's temperature (the model's default when unset).</summary>
    internal float? SpeechTemperature { get; private set; }

    internal static bool Handles(string flag) => flag is "--language" or "--translate" or "--word-timestamps" or "--voice"
        or "--speed" or "--lyrics" or "--duration" or "--bpm" or "--key" or "--time-signature" or "--phonemes"
        or "--no-audio-codes" or "--tts" or "--speech-out" or "--speech-mode" or "--speech-temperature";

    internal void Read(string[] args, ref int i)
    {
        string flag = args[i];
        if (flag == "--translate") { Translate = true; return; }
        if (flag == "--word-timestamps") { WordTimestamps = true; return; }
        if (flag == "--no-audio-codes") { NoAudioCodes = true; return; }
        if (flag == "--tts") { Tts = true; return; }
        if (i + 1 >= args.Length)
            throw new ArgumentException($"{flag} requires a value.");
        string value = args[++i];
        switch (flag)
        {
            case "--language": Language = value; break;
            case "--voice": Voice = value; break;
            case "--speed": Speed = float.Parse(value, CultureInfo.InvariantCulture); break;
            case "--lyrics": Lyrics = File.Exists(value) ? File.ReadAllText(value) : value; break;
            case "--duration": Duration = double.Parse(value, CultureInfo.InvariantCulture); break;
            case "--bpm": Bpm = int.Parse(value, CultureInfo.InvariantCulture); break;
            case "--key": KeyScale = value; break;
            case "--time-signature": TimeSignature = value; break;
            case "--phonemes": Phonemes = value; break;
            case "--speech-out": SpeechOut = value; break;
            case "--speech-mode":
                SpeechMode = value.ToLowerInvariant() switch
                {
                    "interleaved" => TensorSharp.Models.Audio.SpeechReplyMode.Interleaved,
                    "audio" => TensorSharp.Models.Audio.SpeechReplyMode.Audio,
                    _ => throw new ArgumentException($"--speech-mode takes interleaved or audio, not '{value}'."),
                };
                break;
            case "--speech-temperature":
                float t = float.Parse(value, CultureInfo.InvariantCulture);
                if (!(t >= 0f) || float.IsInfinity(t))
                    throw new ArgumentException($"--speech-temperature takes a number of 0 (argmax) or more, not '{value}'.");
                SpeechTemperature = t;
                break;
        }
    }
}

/// <summary>A single-prompt chat run's spoken reply (--speech-out): where its WAV goes and how it speaks.</summary>
internal sealed record CliSpeechRequest(string OutputPath, TensorSharp.Models.Audio.SpeechReplyOptions Options);

internal static class AudioCli
{
    /// <summary>Run <paramref name="model"/> when it is an audio model; false leaves it to the other routes. A chat model
    /// that also speaks (LFM3-Omni) synthesizes only with --tts; without it the prompt is a chat turn.
    /// <paramref name="speechSeed"/> is --seed for a model that samples its speech.</summary>
    internal static bool TryRun(ModelBase model, AudioCliOptions o, string prompt, string audioPath, string outputFile,
        int steps, long seed, uint? speechSeed = null)
    {
        if (o.Tts && model is not ISpeechSynthesisModel)
            throw new ArgumentException($"--tts speaks with a speech synthesis model (Kokoro, or LFM3-Omni with its mmproj); " +
                $"the loaded model ({model.Config?.Architecture}) is not one.");
        if (o.Tts && o.SpeechOut != null)
            throw new ArgumentException("--tts writes its speech to --output; --speech-out is a chat reply's speech. Use one.");
        switch (model)
        {
            case ISpeechRecognitionModel asr:
                Transcribe(asr, o, prompt, audioPath, outputFile, seed);
                return true;
            case ISpeechSynthesisModel tts when o.Tts || model is not ISpeechOutputModel:
                Speak(tts, o, prompt, outputFile, speechSeed);
                return true;
            case IMusicGenerationModel music:
                Compose(music, o, prompt, outputFile, steps, seed);
                return true;
            default:
                return false;
        }
    }

    private static void Transcribe(ISpeechRecognitionModel asr, AudioCliOptions o, string prompt, string audioPath,
        string outputFile, long seed)
    {
        if (string.IsNullOrEmpty(audioPath) || !File.Exists(audioPath))
            throw new ArgumentException("Speech recognition needs --audio <file> (WAV, MP3, Ogg, or anything ffmpeg reads).");
        DecodedAudio audio = AudioIO.Decode(audioPath, 16000, 1);
        var options = new TranscriptionOptions
        {
            Language = o.Language,
            Translate = o.Translate,
            Prompt = prompt,
            WordTimestamps = o.WordTimestamps,
            Seed = unchecked((int)seed),
            OnSegment = s => Console.WriteLine(
                $"[{Clock(s.Start)} --> {Clock(s.End)}] {s.Text.Trim()}"),
        };
        var result = asr.Transcribe(audio, options);
        if (outputFile != null)
        {
            string format = TranscriptFormats.FromExtension(outputFile);
            File.WriteAllText(outputFile, TranscriptFormats.Render(result, format, o.WordTimestamps));
            Console.WriteLine($"Wrote {format} transcript to {outputFile}");
        }
        else
        {
            Console.WriteLine();
            Console.WriteLine(result.Text.Trim());
        }
    }

    private static void Speak(ISpeechSynthesisModel tts, AudioCliOptions o, string text, string outputFile, uint? seed)
    {
        if (string.IsNullOrWhiteSpace(text) && string.IsNullOrWhiteSpace(o.Phonemes))
            throw new ArgumentException("Speech synthesis needs --prompt \"text\" (or --input text.txt, or --phonemes).");
        if (!tts.SpeechSynthesisAvailable)
            throw new ArgumentException(tts.SpeechSynthesisUnavailableReason ?? "The loaded model cannot synthesize speech.");
        if (!tts.SamplesSpeech && o.SpeechTemperature.HasValue)
            throw new ArgumentException("--speech-temperature applies to a model that samples its speech (LFM3-Omni), not this one.");
        if (tts is ISpeechOutputModel speaker && o.Voice != null && !speaker.IsSpeechVoice(o.Voice))
            throw new ArgumentException($"--voice '{o.Voice}' is not one of the model's voices ({string.Join(", ", speaker.SpeechVoices)}).");
        var clock = Stopwatch.StartNew();
        double firstAudio = -1;
        var options = new SpeechSynthesisOptions
        {
            Voice = o.Voice, Speed = o.Speed, Language = o.Language, Phonemes = o.Phonemes,
            Seed = tts.SamplesSpeech ? seed : null, Temperature = o.SpeechTemperature,
        };
        var speech = tts.SynthesizeStream(text ?? string.Empty, options, _ =>
        {
            if (firstAudio < 0) firstAudio = clock.Elapsed.TotalMilliseconds;
        });
        string path = outputFile ?? "speech.wav";
        if (tts.SamplesSpeech)
            File.WriteAllBytes(path, TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Wav(speech.Channels[0], speech.SampleRate));
        else
            WavWriter.Write(path, speech.Channels, speech.SampleRate);
        Console.WriteLine($"Wrote {speech.DurationSeconds:F2}s of speech to {path} in {clock.Elapsed.TotalSeconds:F2}s " +
                          $"({speech.DurationSeconds / Math.Max(1e-9, clock.Elapsed.TotalSeconds):F1}x real time" +
                          (firstAudio >= 0 && tts.SamplesSpeech ? $", first audio after {firstAudio:F0} ms)" : ")"));
    }

    private static void Compose(IMusicGenerationModel music, AudioCliOptions o, string tags, string outputFile, int steps, long seed)
    {
        var clock = Stopwatch.StartNew();
        var song = music.GenerateMusic(new MusicGenerationRequest
        {
            Tags = tags ?? string.Empty,
            Lyrics = string.IsNullOrWhiteSpace(o.Lyrics) ? "[Instrumental]" : o.Lyrics,
            DurationSeconds = o.Duration,
            Bpm = o.Bpm,
            KeyScale = o.KeyScale,
            TimeSignature = o.TimeSignature,
            Language = o.Language ?? "en",
            Seed = seed,
            Steps = steps,
            GenerateAudioCodes = !o.NoAudioCodes,
            OnStep = (i, n) => Console.WriteLine($"  step {i}/{n}"),
        });
        string path = outputFile ?? "song.wav";
        WavWriter.Write(path, song.Channels, song.SampleRate);
        Console.WriteLine($"Wrote {song.DurationSeconds:F1}s of music to {path} in {clock.Elapsed.TotalSeconds:F1}s");
    }

    private static string Clock(double seconds)
    {
        var t = TimeSpan.FromSeconds(Math.Max(0, seconds));
        return t.TotalHours >= 1 ? t.ToString(@"h\:mm\:ss\.ff", CultureInfo.InvariantCulture)
                                 : t.ToString(@"mm\:ss\.ff", CultureInfo.InvariantCulture);
    }
}
