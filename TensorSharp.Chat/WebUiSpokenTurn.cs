// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Server;
using TensorSharp.Server.ResponseSerializers;

namespace TensorSharp.Chat
{
    /// <summary>
    /// The speech of a web chat turn that is spoken from its text: the text turn of the same request runs as it would
    /// (reasoning, skills, the host's tools, knowledge, every round), and this speaks what the page shows as the answer
    /// while it is written (<see cref="ReplySpeaker"/>). Fed from the turn's answer frames
    /// (<see cref="WebUiChatService.AnswerFrames"/>); the round's end (a tool call, a tool's progress) speaks the text
    /// before it at once. A host's note in the answer (<c>_(...)_</c>, e.g. "the output started repeating itself") is
    /// not spoken. Its <c>{audio, sampleRate}</c> frames are interleaved with the turn's own as they are made
    /// (<see cref="AudioFramesUntil"/>), and every sample is kept for the turn's WAV.
    /// </summary>
    internal sealed class WebUiSpokenTurn : IDisposable
    {
        private readonly List<float> _pcm = new();

        public WebUiSpokenTurn(ReplySpeaker speaker, string language)
        {
            Speaker = speaker ?? throw new ArgumentNullException(nameof(speaker));
            Language = language;
        }

        public ReplySpeaker Speaker { get; }

        /// <summary>The reply's language (the request's, else the user's), null for the model's default.</summary>
        public string Language { get; }

        /// <summary>Every sample sent in an audio frame, in order (the turn's WAV).</summary>
        public List<float> Pcm => _pcm;

        /// <summary>The sample rate of the audio sent (0 before the first frame).</summary>
        public int SampleRate { get; private set; }

        /// <summary>Audio frames sent.</summary>
        public int AudioFrames { get; private set; }

        /// <summary>Answer text the page shows.</summary>
        public void Say(string content)
        {
            if (string.IsNullOrEmpty(content) || IsHostNote(content)) return;
            Speaker.Say(content);
        }

        /// <summary>Answer text the page takes back (it was reasoning).</summary>
        public void Retract(string retracted) => Speaker.Retract(retracted);

        /// <summary>The round's text has ended (a tool call is being written or run).</summary>
        public void EndOfRound() => Speaker.Pause();

        /// <summary>
        /// A note the host writes into the answer, whole, in one update: italic and in parentheses (the skill loop's
        /// repetition and completion notes). Not the model's words, so not spoken.
        /// </summary>
        internal static bool IsHostNote(string content)
        {
            string t = content.Trim();
            return t.Length > 4 && t.StartsWith("_(", StringComparison.Ordinal) && t.EndsWith(")_", StringComparison.Ordinal);
        }

        /// <summary>The audio frames made so far and not yet sent.</summary>
        public IEnumerable<object> TakeAudioFrames()
        {
            while (Speaker.Audio.TryRead(out ChatStreamUpdate update))
            {
                if (update.Audio is not { Length: > 0 } samples) continue;
                _pcm.AddRange(samples);
                if (update.AudioSampleRate > 0) SampleRate = update.AudioSampleRate;
                AudioFrames++;
                yield return WebUiSseEvents.Audio(TensorSharp.Models.Lfm3.Lfm3SpeechAudio.Pcm16(samples), update.AudioSampleRate);
            }
        }

        /// <summary>
        /// The audio frames made while <paramref name="until"/> runs (the turn's next update, a tool round, the end of the
        /// speech), as they are made; ends when it has completed (it is never awaited here, so its error stays its
        /// awaiter's).
        /// </summary>
        public async IAsyncEnumerable<object> AudioFramesUntil(Task until, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            while (true)
            {
                foreach (object frame in TakeAudioFrames())
                    yield return frame;
                if (until.IsCompleted) yield break;
                Task audio = Speaker.WhenAudioOrEnd();
                if (audio.IsCompleted && Speaker.Audio.Completion.IsCompleted)
                {
                    // The speech has ended: nothing more to send while waiting.
                    await Quietly(until).ConfigureAwait(false);
                    continue;
                }
                await Task.WhenAny(until, audio).ConfigureAwait(false);
            }
        }

        /// <summary>Waits for <paramref name="task"/> without observing its outcome.</summary>
        private static Task Quietly(Task task)
            => task.ContinueWith(static _ => { }, CancellationToken.None, TaskContinuationOptions.ExecuteSynchronously, TaskScheduler.Default);

        public void Dispose() => Speaker.Dispose();
    }
}
