// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ACE-Step 1.5 text to music (Apache-2.0), from a GGUF written by eng/audio-models/acestep_to_gguf.py,
// as ComfyUI's template runs it (TextEncodeAceStepAudio1.5 -> KSampler euler/simple, ModelSamplingAuraFlow
// shift 3, cfg 1 -> VAEDecodeAudio):
//
//   1. the planner LM (Qwen3, CFG 2 against an empty chain of thought, temperature 0.85, top-p 0.9)
//      writes 5 audio codes per second from the caption, lyrics and metadata;
//   2. Qwen3-0.6B encodes the caption and metadata; the lyrics enter as its token embeddings;
//   3. the DiT denoises a 25 Hz, 64-channel latent in 8 Euler steps, guided by the codes' hints;
//   4. the Oobleck VAE decodes it to 48 kHz stereo, and the result is scaled down when its standard
//      deviation is above 0.2 (VAEDecodeAudio's normalisation).
//
//   TensorSharp.Cli --model ace-step-1.5-turbo.gguf --backend cuda --prompt "lofi hip hop, piano"
//       --lyrics "[Instrumental]" --duration 30 --bpm 90 --key "C major" --time-signature 4 --output song.wav
//
// The server exposes it as POST /v1/audio/music.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Direct;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.Video;
using TensorSharp.Runtime;

namespace TensorSharp.Models.AceStep
{
    public sealed class AceStepModel : ModelBase, IMusicGenerationModel
    {
        public const string ArchitectureId = "ace-step-1.5";
        public const int SampleRate = 48000;
        private const int LatentHop = 1920;
        private const int AudioStart = 151669, AudioEnd = 215669, PadToken = 151643;

        private readonly GgufFile _file;
        private readonly DirectContext _ctx;
        private readonly BpeTokenizer _tokenizer;
        private readonly float[] _silence;           // [64, 750]
        private readonly float _shift;
        private readonly int _defaultSteps;
        private readonly object _lock = new();
        private AceQwen3 _te, _lm;
        private DirectLinear _lmHead;
        private AceStepDiT _dit;
        private AceStepVae _vae;
        private bool _disposed;

        /// <summary>Seconds of the last song: planner, encoders, diffusion, VAE.</summary>
        public (double Planner, double Encode, double Diffusion, double Vae, int Codes, int Steps) LastTimings { get; private set; }

        /// <summary>The audio codes of the last song (to replay it, or compare against another implementation).</summary>
        public IReadOnlyList<int> LastAudioCodes { get; private set; } = Array.Empty<int>();

        /// <summary>When set, the next song uses these codes instead of sampling the planner (validation).</summary>
        public IReadOnlyList<int> AudioCodesOverride { get; set; }

        public AceStepModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend, openGguf: false)
        {
            if (backend is not (BackendType.Cuda or BackendType.Cpu))
                throw new ModelLoadRefusedException(
                    $"ACE-Step 1.5 runs on --backend cuda (or cpu, slowly), not {BackendNames.OptionName(backend)}.");
            _file = new GgufFile(ggufPath);
            try
            {
                _file.ThrowIfTruncated();
                string[] vocab = _file.GetStringArray("tokenizer.ggml.tokens")
                    ?? throw new InvalidDataException("the ACE-Step GGUF has no tokenizer.ggml.tokens.");
                int[] types = _file.GetInt32Array("tokenizer.ggml.token_type");
                string[] merges = _file.GetStringArray("tokenizer.ggml.merges") ?? Array.Empty<string>();
                _tokenizer = new BpeTokenizer(vocab, types, merges, PadToken, new[] { PadToken }, addBos: false, addEos: false);
                _silence = DirectOps.DequantTensor(_file, "ace.silence_latent");
                _shift = _file.GetFloat32("ace.shift", 3f);
                _defaultSteps = (int)_file.GetUint32("ace.steps", 8);
                Config = new ModelConfig { Architecture = ArchitectureId, VocabSize = vocab.Length };
                _ctx = new DirectContext(_allocator);
                Console.WriteLine($"ACE-Step 1.5: shift {_shift}, {_defaultSteps} steps by default, {vocab.Length}-token vocabulary");
            }
            catch
            {
                _file.Dispose();
                throw;
            }
        }

        // ---- IMusicGenerationModel ----
        public string MusicModelFamily => ArchitectureId;
        public int OutputSampleRate => SampleRate;

        public DecodedAudio GenerateMusic(MusicGenerationRequest r)
        {
            ArgumentNullException.ThrowIfNull(r);
            if (r.DurationSeconds is <= 0 or > 600)
                throw new ArgumentException("duration must be between 0 and 600 seconds.");
            lock (_lock)
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                return Generate(r);
            }
        }

        private void EnsureLoaded()
        {
            var clock = Stopwatch.StartNew();
            _te ??= new AceQwen3(_ctx, _file, "te.");
            _dit ??= new AceStepDiT(_ctx, _file);
            _vae ??= new AceStepVae(_file);
            if (clock.Elapsed.TotalSeconds > 0.5)
                Console.WriteLine($"  [ace-step] parts bound in {clock.Elapsed.TotalSeconds:F1}s " +
                                  $"(caption encoder {_te.Layers} layers, DiT {_dit.LayerCount} layers)");
        }

        /// <summary>The planner LM, bound on first use (a song replayed from its codes never needs it).</summary>
        private void EnsurePlanner()
        {
            _lm ??= new AceQwen3(_ctx, _file, "lm.");
            _lmHead ??= DirectLinear.FromGgufRows(_ctx, _file, _lm.EmbedName, AudioStart, AudioEnd - AudioStart);
        }

        private DecodedAudio Generate(MusicGenerationRequest r)
        {
            EnsureLoaded();
            var total = Stopwatch.StartNew();
            int duration = (int)Math.Ceiling(r.DurationSeconds);
            int frames = (int)Math.Round(r.DurationSeconds * SampleRate / LatentHop, MidpointRounding.ToEven);
            string language = string.IsNullOrWhiteSpace(r.Language) ? "en" : r.Language.Trim();
            string tags = (r.Tags ?? string.Empty).Trim();
            string lyrics = r.Lyrics ?? string.Empty;
            var prompts = AceStepPrompts.Build(tags, lyrics, duration, r.Bpm, r.KeyScale, r.TimeSignature, language);

            // ---- 1. planner ----
            var clock = Stopwatch.StartNew();
            List<int> codes;
            // Validation hooks: TS_ACESTEP_CODES names a JSON list of planner codes to use instead of sampling;
            // TS_ACESTEP_DUMP names a directory that receives the caption states, hints and final latent (F32).
            string codesFile = Environment.GetEnvironmentVariable("TS_ACESTEP_CODES");
            if (AudioCodesOverride == null && !string.IsNullOrEmpty(codesFile))
                AudioCodesOverride = System.Text.Json.JsonSerializer.Deserialize<List<int>>(File.ReadAllText(codesFile));
            if (AudioCodesOverride != null) codes = AudioCodesOverride.ToList();
            else if (r.GenerateAudioCodes) codes = SampleCodes(prompts.LmPrompt, prompts.LmNegative, duration * 5, r.Seed);
            else throw new NotSupportedException(
                "ACE-Step without planner codes needs reference audio, which this build does not take; leave GenerateAudioCodes on.");
            double planner = clock.Elapsed.TotalSeconds;
            LastAudioCodes = codes;

            // ---- 2. caption, lyrics, timbre, hints ----
            clock.Restart();
            var capIds = _tokenizer.Encode(prompts.Caption, addSpecial: false);
            float[] caption;
            using (var cache = _te.NewCache(capIds.Count))
            using (var hidden = _te.Forward(capIds, 0, ComfyMask(capIds), cache))
                caption = DirectOps.ToArray(hidden);
            var lyricIds = _tokenizer.Encode(prompts.Lyrics, addSpecial: false);
            float[] lyricEmb = _te.Embed(lyricIds);
            var timbre = new float[(long)frames * 64];
            for (int t = 0; t < frames; t++)
                for (int c = 0; c < 64; c++)
                    timbre[(long)t * 64 + c] = _silence[(long)c * 750 + Math.Min(t, 749)];
            _dit.BindCondition(caption, capIds.Count, lyricEmb, lyricIds.Count, timbre, frames);
            float[] hints = _dit.Hints(codes, frames);
            double encode = clock.Elapsed.TotalSeconds;
            string dump = Environment.GetEnvironmentVariable("TS_ACESTEP_DUMP");
            Dump(dump, "caption", caption);
            Dump(dump, "lyrics", lyricEmb);
            Dump(dump, "hints", hints);

            // ---- 3. diffusion: Euler on the shifted flow schedule ----
            clock.Restart();
            int steps = r.Steps > 0 ? r.Steps : _defaultSteps;
            float[] sigmas = Sigmas(steps, _shift);
            float[] noise = TorchCpuRandn.Normal(64L * frames, unchecked((ulong)r.Seed));
            var x = new float[(long)frames * 64];
            for (int c = 0; c < 64; c++)
                for (int t = 0; t < frames; t++)
                    x[(long)t * 64 + c] = noise[(long)c * frames + t] * sigmas[0];
            for (int i = 0; i < steps; i++)
            {
                float sigma = sigmas[i], next = sigmas[i + 1];
                float[] v = _dit.Velocity(x, hints, frames, sigma);
                for (long k = 0; k < x.LongLength; k++)
                {
                    float denoised = x[k] - v[k] * sigma;
                    float d = (x[k] - denoised) / sigma;
                    x[k] = next == 0f ? denoised : x[k] + d * (next - sigma);
                }
                r.OnStep?.Invoke(i + 1, steps);
            }
            double diffusion = clock.Elapsed.TotalSeconds;

            Dump(dump, "latent", x);

            // ---- 4. decode ----
            clock.Restart();
            float[][] audio = _vae.DecodeTiled(x, frames);
            Normalize(audio);
            double vae = clock.Elapsed.TotalSeconds;
            LastTimings = (planner, encode, diffusion, vae, codes.Count, steps);
            Console.WriteLine($"  [ace-step] {r.DurationSeconds:F1}s song, {codes.Count} codes, {steps} steps: planner {planner:F1}s, " +
                              $"encoders {encode:F1}s, diffusion {diffusion:F1}s ({diffusion / steps:F2}s/step), VAE {vae:F1}s, " +
                              $"total {total.Elapsed.TotalSeconds:F1}s");
            return new DecodedAudio { Channels = audio, SampleRate = SampleRate };
        }

        private static void Dump(string dir, string name, float[] values)
        {
            if (string.IsNullOrEmpty(dir)) return;
            File.WriteAllBytes(Path.Combine(dir, name + ".f32"), System.Runtime.InteropServices.MemoryMarshal.AsBytes<float>(values).ToArray());
        }

        /// <summary>ModelSamplingDiscreteFlow's 1000 shifted sigmas, picked by the "simple" scheduler, then 0.</summary>
        internal static float[] Sigmas(int steps, float shift)
        {
            var table = new float[1000];
            for (int i = 0; i < 1000; i++)
            {
                float t = (i + 1) / 1000f;
                table[i] = shift * t / (1f + (shift - 1f) * t);
            }
            var s = new float[steps + 1];
            double ss = 1000.0 / steps;
            for (int x = 0; x < steps; x++) s[x] = table[999 - (int)(x * ss)];
            s[steps] = 0f;
            return s;
        }

        /// <summary>VAEDecodeAudio: divide by 5 x the (unbiased) standard deviation when that is above 1.</summary>
        private static void Normalize(float[][] audio)
        {
            double sum = 0, sq = 0;
            long n = 0;
            foreach (var ch in audio) { foreach (float v in ch) { sum += v; sq += (double)v * v; } n += ch.Length; }
            double mean = sum / n;
            double std = Math.Sqrt(Math.Max(0, (sq - n * mean * mean) / Math.Max(1, n - 1))) * 5.0;
            if (std < 1.0) return;
            float inv = (float)(1.0 / std);
            foreach (var ch in audio)
                for (int i = 0; i < ch.Length; i++) ch[i] *= inv;
        }

        // ------------------------------------------------------------------ planner sampling
        private List<int> SampleCodes(string positive, string negative, int count, long seed)
        {
            EnsurePlanner();
            var pos = _tokenizer.Encode(positive, addSpecial: false);
            var neg = _tokenizer.Encode(negative, addSpecial: false);
            int len = Math.Max(pos.Count, neg.Count);
            var seqs = new[] { LeftPad(pos, len), LeftPad(neg, len) };
            var valid = new[] { Valid(pos.Count, len, count), Valid(neg.Count, len, count) };
            const float cfg = 2.0f, temperature = 0.85f, topP = 0.9f;
            var rng = new Random(unchecked((int)seed));
            var caches = new[] { _lm.NewCache(len + count), _lm.NewCache(len + count) };
            var codes = new List<int>(count);
            try
            {
                float[][] logits = new float[2][];
                for (int s = 0; s < 2; s++)
                {
                    using var h = _lm.Forward(seqs[s], 0, valid[s].AsSpan(0, len).ToArray(), caches[s]);
                    using var last = h.Narrow(0, len - 1, 1);
                    using var l = _lmHead.Forward(last);
                    logits[s] = DirectOps.ToArray(l);
                }
                string dump = Environment.GetEnvironmentVariable("TS_ACESTEP_DUMP");
                Dump(dump, "lm_cond0", logits[0]);
                Dump(dump, "lm_uncond0", logits[1]);
                Console.WriteLine($"  [ace-step] planner prompts: {pos.Count} and {neg.Count} tokens, {count} codes to write");
                for (int step = 0; step < count; step++)
                {
                    int code = SampleCfg(logits[0], logits[1], cfg, temperature, topP, rng);
                    codes.Add(code);
                    if (step + 1 == count) break;
                    int token = AudioStart + code;
                    for (int s = 0; s < 2; s++)
                    {
                        int position = len + step;
                        using var h = _lm.Forward(new[] { token }, position, valid[s].AsSpan(0, position + 1).ToArray(), caches[s]);
                        using var l = _lmHead.Forward(h);
                        logits[s] = DirectOps.ToArray(l);
                    }
                }
            }
            finally
            {
                foreach (var c in caches) c.Dispose();
            }
            return codes;
        }

        /// <summary>
        /// sd1_clip.SDClipModel.process_tokens' attention mask for a tokenizer whose pad token (151643,
        /// &lt;|endoftext|&gt;) doubles as its end: leading pads are masked, and so is the first later pad and
        /// everything after it. The caption prompt ends in two of them, so its tail attends to the prompt only.
        /// </summary>
        internal static bool[] ComfyMask(IReadOnlyList<int> ids)
        {
            var mask = new bool[ids.Count];
            bool eos = false, leftPad = false;
            for (int i = 0; i < ids.Count; i++)
            {
                int t = ids[i];
                if (i == 0 && t == PadToken) leftPad = true;
                if (eos || (leftPad && t == PadToken)) mask[i] = false;
                else { mask[i] = true; leftPad = false; }
                if (!eos && t == PadToken && !leftPad)
                {
                    mask[i] = false;
                    eos = true;
                }
            }
            return mask;
        }

        private static List<int> LeftPad(List<int> ids, int len)
        {
            var r = new List<int>(len);
            for (int i = ids.Count; i < len; i++) r.Add(PadToken);
            r.AddRange(ids);
            return r;
        }

        private static bool[] Valid(int real, int len, int extra)
        {
            var v = new bool[len + extra];
            for (int i = 0; i < v.Length; i++) v[i] = i >= len - real;
            return v;
        }

        /// <summary>_sample_audio_token: CFG over the audio-code logits, top-p, temperature, a multinomial draw.</summary>
        internal static int SampleCfg(float[] cond, float[] uncond, float cfg, float temperature, float topP, Random rng)
        {
            int n = cond.Length;
            var l = new float[n];
            for (int i = 0; i < n; i++) l[i] = uncond[i] + cfg * (cond[i] - uncond[i]);
            if (topP < 1f)
            {
                var order = Enumerable.Range(0, n).OrderByDescending(i => l[i]).ToArray();
                double max = l[order[0]], z = 0;
                for (int i = 0; i < n; i++) z += Math.Exp(l[i] - max);
                double cum = 0;
                bool removeNext = false;
                foreach (int i in order)
                {
                    bool remove = removeNext;          // shifted right by one: the first token past top_p stays
                    cum += Math.Exp(l[i] - max) / z;
                    removeNext = cum > topP;
                    if (remove) l[i] = float.NegativeInfinity;
                }
            }
            if (temperature <= 0f)
            {
                int best = 0;
                for (int i = 1; i < n; i++) if (l[i] > l[best]) best = i;
                return best;
            }
            double m = double.NegativeInfinity;
            foreach (float v in l) if (v > m) m = v;
            var p = new double[n];
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                p[i] = float.IsNegativeInfinity(l[i]) ? 0 : Math.Exp((l[i] - m) / temperature);
                sum += p[i];
            }
            double u = rng.NextDouble() * sum, acc = 0;
            for (int i = 0; i < n; i++)
            {
                acc += p[i];
                if (p[i] > 0 && acc >= u) return i;
            }
            return Array.IndexOf(l, (float)m);
        }

        // ---- not a chat model ----
        protected override float[] ForwardCore(int[] tokens) =>
            throw new NotSupportedException("ACE-Step is a music generation model; use GenerateMusic().");

        protected override void ResetKVCacheCore() { }

        public override void WarmUpKernels() { }

        public override void Dispose()
        {
            lock (_lock)
            {
                if (_disposed) return;
                _disposed = true;
                _dit?.Dispose();
                _te?.Dispose();
                _lm?.Dispose();
                _ctx?.Dispose();
                _file?.Dispose();
                base.Dispose();
            }
        }
    }

    /// <summary>The four prompts ACE15Tokenizer.tokenize_with_weights builds.</summary>
    internal static class AceStepPrompts
    {
        private const string LmTemplate =
            "<|im_start|>system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n<|im_end|>\n" +
            "<|im_start|>user\n# Caption\n{0}\n\n# Lyric\n{1}\n<|im_end|>\n<|im_start|>assistant\n{2}\n\n<|im_end|>\n";
        private const string LyricsTemplate = "# Languages\n{0}\n\n# Lyric\n{1}<|endoftext|><|endoftext|>";
        private const string CaptionTemplate =
            "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n# Caption\n{0}\n\n# Metas\n{1}\n<|endoftext|>\n<|endoftext|>";

        public sealed record Set(string LmPrompt, string LmNegative, string Lyrics, string Caption);

        public static Set Build(string tags, string lyrics, int duration, int bpm, string keyScale, string timeSignature, string language)
        {
            string ts = timeSignature?.Trim();
            if (ts != null && ts.EndsWith("/4", StringComparison.Ordinal)) ts = ts[..^2];
            string key = string.IsNullOrWhiteSpace(keyScale) ? null : keyScale.Trim();

            // <think> chain of thought: the metadata as YAML, keys sorted (yaml.dump(sort_keys=True)).
            var metas = new SortedDictionary<string, string>(StringComparer.Ordinal) { ["duration"] = duration.ToString(CultureInfo.InvariantCulture) };
            if (bpm > 0) metas["bpm"] = bpm.ToString(CultureInfo.InvariantCulture);
            if (key != null) metas["keyscale"] = Yaml(key);
            if (!string.IsNullOrEmpty(ts)) metas["timesignature"] = ts.All(char.IsDigit) ? ts : Yaml(ts);
            string cot = "<think>\n" + string.Join("\n", metas.Select(kv => $"{kv.Key}: {kv.Value}")) + "\n</think>";
            const string cotNegative = "<think>\n\n</think>";

            string cap = string.Join("\n",
                $"- bpm: {(bpm > 0 ? bpm.ToString(CultureInfo.InvariantCulture) : "N/A")}",
                $"- timesignature: {(string.IsNullOrEmpty(ts) ? "N/A" : ts)}",
                $"- keyscale: {key ?? "N/A"}",
                $"- duration: {duration} seconds");

            return new Set(
                string.Format(CultureInfo.InvariantCulture, LmTemplate, tags, lyrics.Trim(), cot),
                string.Format(CultureInfo.InvariantCulture, LmTemplate, tags, lyrics.Trim(), cotNegative),
                string.Format(CultureInfo.InvariantCulture, LyricsTemplate, language, lyrics),
                string.Format(CultureInfo.InvariantCulture, CaptionTemplate, tags, cap));
        }

        /// <summary>A YAML plain scalar where PyYAML would write one, else single-quoted.</summary>
        private static string Yaml(string s)
        {
            bool plain = s.Length > 0 && !"-?:,[]{}#&*!|>'\"%@`".Contains(s[0]) && !s.Contains(": ") && !s.Contains(" #")
                         && !char.IsWhiteSpace(s[0]) && !char.IsWhiteSpace(s[^1])
                         && !double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out _)
                         && s is not ("true" or "false" or "yes" or "no" or "null" or "on" or "off" or "~");
            return plain ? s : "'" + s.Replace("'", "''") + "'";
        }
    }
}
