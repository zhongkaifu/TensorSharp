// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Text.Json;

namespace TensorAgent.Tests;

/// <summary>
/// The page's half of a spoken reply (a model that speaks: LFM3-Omni with its projector), run: the switch appears only
/// for such a model, a turn sent with it on asks for <c>speech</c>, the reply's chunks are scheduled one after another on
/// an AudioContext as they arrive, the finished reply leaves a player and a Save link, and the history the next request
/// sends keeps its audio id and WAV -- in a reopened chat too, where Save goes through the app's picker.
/// </summary>
public sealed partial class WebUiPageTests
{
    private const string SpeakingModel = """
        R['/api/models'] = { loaded: 'LFM3-Omni-2B-Q8_0.gguf', architecture: 'lfm3', loadedBackend: 'ggml_metal',
                             visionReady: true, speechOutput: { available: true, voices: ['female', 'male'], reason: null } };
        """;

    /// <summary>An AudioContext that records what the page schedules on it.</summary>
    private const string FakeAudio = """
        var scheduled = [];
        window.AudioContext = function () { this.state = 'running'; this.currentTime = 0; this.destination = {}; };
        window.AudioContext.prototype.createBuffer = function (channels, length, rate) {
          var data = new Float32Array(length);
          return { length: length, sampleRate: rate, duration: length / rate, getChannelData: function () { return data; } };
        };
        window.AudioContext.prototype.createBufferSource = function () {
          var source = { connect: function () {}, stop: function () { source.stopped = true; } };
          source.start = function (at) {
            var data = source.buffer.getChannelData(0);
            scheduled.push({ at: at, length: source.buffer.length, rate: source.buffer.sampleRate,
              samples: Array.prototype.slice.call(data) });
          };
          return source;
        };
        """;

    // Two chunks of PCM16 LE: [0, 16384, -16384, 32767] and [-32768, 8192].
    private const string SpokenFrames = """
        { thinking: 'She asked for a greeting.' },
        { token: 'Hello, ' }, { tool_progress: 'writing', tool: 'shell', text: 'echo hi', seconds: 0 },
        { tool_progress: 'running', tool: 'shell', text: null, seconds: 0, detail: 'echo hi' },
        { audio: 'AAAAQADA/38=', sampleRate: 24000 },
        { tool_progress: 'finished', tool: 'shell', text: null, seconds: 0.1 },
        { token: 'Anna!' }, { audio: 'AIAAIA==', sampleRate: 24000 },
        { notice: 'The reply\'s speech stopped, so only part of it was spoken: engine busy', code: 'speech_failed', detail: 'engine busy' },
        { done: true, sessionId: 's1', tokenCount: 9, elapsed: 1.5, tokPerSec: 6, promptTokens: 40, kvReusedTokens: 0,
          kvReusePercent: 0, audioId: 'audio_abc', audioUrl: '/uploads/speech-20261119-101500-0123456789ab.wav', audioSeconds: 0.24 }
        """;

    [WebJavaScriptFact]
    public void ASpokenReplyPlaysAsItArrivesAndKeepsAPlayerASaveLinkAndItsIdForTheNextTurn()
    {
        JsonElement result = Run(SpeakingModel + "R['/api/chat'] = { __sse: [" + SpokenFrames + "] };", FakeAudio + """
            return settle(20).then(function () {
              var shown = __page.byId['speechbar'].style.display;
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              __page.byId['text'].value = 'Say hello to Anna.';
              __page.byId['send'].dispatch('click');
              return settle(30).then(function () {
                var box = __page.byId['chat'].querySelector('.spoken-reply');
                var save = box && box.querySelector('.speech-download');
                var history = JSON.parse(JSON.stringify(window.TensorAgent.history()));
                var heard = scheduled.slice();
                __page.byId['text'].value = 'And to Ben?';
                __page.byId['send'].dispatch('click');
                return settle(30).then(function () { return {
                  shown: shown,
                  voiceShown: __page.byId['speech-voice'].style.display,
                  sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                  scheduled: heard,
                  scheduledAfterTwoTurns: scheduled.length,
                  player: box && box.querySelector('audio') ? box.querySelector('audio').src : null,
                  saveLabel: save ? save.textContent : null,
                  saveName: save ? save.download : null,
                  notes: __page.byId['chat'].querySelectorAll('.turn-note').map(function (n) { return n.textContent; }),
                  answer: history[1],
                  errors: __page.errorNotices()
                }; });
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal("", result.GetProperty("shown").GetString());
        Assert.Equal("", result.GetProperty("voiceShown").GetString());

        JsonElement first = result.GetProperty("sent")[0];
        Assert.Equal(JsonValueKind.Object, first.GetProperty("speech").ValueKind);
        Assert.Equal("female", first.GetProperty("speech").GetProperty("voice").GetString());

        // Both chunks, back to back, at 24 kHz, decoded from PCM16.
        JsonElement[] scheduled = result.GetProperty("scheduled").EnumerateArray().ToArray();
        Assert.Equal(2, scheduled.Length);
        Assert.Equal(new[] { 0, 0.5, -0.5, 32767 / 32768.0 }, scheduled[0].GetProperty("samples").EnumerateArray().Select(v => v.GetDouble()));
        Assert.Equal(new[] { -1.0, 0.25 }, scheduled[1].GetProperty("samples").EnumerateArray().Select(v => v.GetDouble()));
        Assert.All(scheduled, s => Assert.Equal(24000, s.GetProperty("rate").GetInt32()));
        Assert.Equal(scheduled[0].GetProperty("at").GetDouble() + 4 / 24000.0, scheduled[1].GetProperty("at").GetDouble(), 9);
        // The second (also spoken) reply plays its own two chunks.
        Assert.Equal(4, result.GetProperty("scheduledAfterTwoTurns").GetInt32());

        Assert.Equal("/uploads/speech-20261119-101500-0123456789ab.wav", result.GetProperty("player").GetString());
        Assert.Equal("Save audio (0.2 s)", result.GetProperty("saveLabel").GetString());
        Assert.Equal("speech-20261119-101500-0123456789ab.wav", result.GetProperty("saveName").GetString());
        // A spoken turn's tool round and reasoning render as a text turn's; its audio plays across the round. A notice is
        // said once in each turn, in the interface language (the route answers both turns with the same frames).
        Assert.Equal(Enumerable.Repeat("The reply's speech stopped, so only part of it was spoken: engine busy", 2),
            result.GetProperty("notes").EnumerateArray().Select(n => n.GetString()));

        JsonElement answer = result.GetProperty("answer");
        Assert.Equal("Hello, Anna!", answer.GetProperty("content").GetString());
        Assert.Equal("audio_abc", answer.GetProperty("audioId").GetString());
        Assert.Equal("/uploads/speech-20261119-101500-0123456789ab.wav", answer.GetProperty("audioUrl").GetString());
        Assert.Equal(0.24, answer.GetProperty("audioSeconds").GetDouble());

        // The next request carries the spoken turn back by its id, so the host splices it.
        JsonElement next = result.GetProperty("sent")[1];
        JsonElement back = next.GetProperty("messages")[1];
        Assert.Equal("audio_abc", back.GetProperty("audioId").GetString());
        Assert.Equal("/uploads/speech-20261119-101500-0123456789ab.wav", back.GetProperty("audioUrl").GetString());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }

    /// <summary>
    /// The live chunks play as playback, not as ambient sound: WebKit plays a page's Web Audio under the ambient
    /// audio-session category, which an iPhone's Silent mode mutes, while the finished reply's &lt;audio&gt; player plays as
    /// media -- the reported "the speech can only be heard from the player after the whole answer". The page asks for
    /// playback through the Audio Session API from the send and with each chunk. Playback does not mix (it pauses the
    /// user's music), so the switch alone asks for nothing, and once the reply's audio has played the page gives the
    /// choice back and lets its AudioContext rest, so the other audio comes back after every reply.
    /// </summary>
    [WebJavaScriptFact]
    public void ASpokenReplyPlaysAsPlaybackAndGivesTheAudioBackOnceItHasPlayed()
    {
        JsonElement result = Run(SpeakingModel + "R['/api/chat'] = { __sse: [" + SpokenFrames + "] };", FakeAudio + """
            var types = [], sources = [], suspends = 0;
            navigator.audioSession = {};
            var current = 'auto';
            Object.defineProperty(navigator.audioSession, 'type', {
              get: function () { return current; },
              set: function (v) { current = v; types.push(v); }
            });
            var create = window.AudioContext.prototype.createBufferSource;
            window.AudioContext.prototype.createBufferSource = function () { var s = create.call(this); sources.push(s); return s; };
            window.AudioContext.prototype.suspend = function () { suspends++; this.state = 'suspended'; return Promise.resolve(); };
            window.AudioContext.prototype.resume = function () { this.state = 'running'; return Promise.resolve(); };
            return settle(20).then(function () {
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              var afterSwitch = navigator.audioSession.type;
              __page.byId['text'].value = 'Say hello to Anna.';
              __page.byId['send'].dispatch('click');
              var afterSend = navigator.audioSession.type;
              return settle(30).then(function () {
                // The text is done and the reply's audio is still playing.
                var turnOver = navigator.audioSession.type, suspendsWhilePlaying = suspends;
                sources[0].onended();
                var oneLeft = navigator.audioSession.type;
                sources[1].onended();
                return { afterSwitch: afterSwitch, afterSend: afterSend, turnOver: turnOver, oneLeft: oneLeft,
                  drained: navigator.audioSession.type, types: types, suspendsWhilePlaying: suspendsWhilePlaying,
                  suspends: suspends, scheduled: scheduled.length, errors: __page.errorNotices() };
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal("auto", result.GetProperty("afterSwitch").GetString());
        Assert.Equal("playback", result.GetProperty("afterSend").GetString());
        Assert.Equal("playback", result.GetProperty("turnOver").GetString());
        Assert.Equal("playback", result.GetProperty("oneLeft").GetString());
        Assert.Equal("auto", result.GetProperty("drained").GetString());
        // Set when it changes, not on every chunk.
        Assert.Equal(new[] { "playback", "auto" }, result.GetProperty("types").EnumerateArray().Select(t => t.GetString()));
        Assert.Equal(0, result.GetProperty("suspendsWhilePlaying").GetInt32());
        Assert.Equal(1, result.GetProperty("suspends").GetInt32());
        Assert.Equal(2, result.GetProperty("scheduled").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }

    /// <summary>A spoken turn that made no speech (it failed, or the reply was empty) gives the audio back when it
    /// ends; so does Stop speaking, at once.</summary>
    [WebJavaScriptFact]
    public void ATurnWithoutSpeechAndStopSpeakingGiveTheAudioBack()
    {
        JsonElement result = Run(SpeakingModel + """
            var calls = 0;
            R['/api/chat'] = function () {
              calls++;
              return calls === 1
                ? { __sse: [{ token: 'Hi.' }, { done: true, sessionId: 's1', tokenCount: 1 }] }
                : { __sse: [{ token: 'Hello, ' }, { audio: 'AAAAQADA/38=', sampleRate: 24000 }, { token: 'Anna!' },
                    { done: true, sessionId: 's1', tokenCount: 2, audioId: 'audio_x' }] };
            };
            """, FakeAudio + """
            navigator.audioSession = { type: 'auto' };
            return settle(20).then(function () {
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              __page.byId['text'].value = 'hi';
              __page.byId['send'].dispatch('click');
              return settle(30).then(function () {
                var noSpeech = navigator.audioSession.type;
                __page.byId['text'].value = 'Say hello to Anna.';
                __page.byId['send'].dispatch('click');
                return settle(30).then(function () {
                  var playing = navigator.audioSession.type;
                  var stop = __page.byId['chat'].querySelector('.speech-stop');
                  if (stop) stop.dispatch('click');
                  return { noSpeech: noSpeech, playing: playing, stopped: navigator.audioSession.type, hadStop: !!stop };
                });
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal("auto", result.GetProperty("noSpeech").GetString());
        Assert.Equal("playback", result.GetProperty("playing").GetString());
        Assert.True(result.GetProperty("hadStop").GetBoolean());
        Assert.Equal("auto", result.GetProperty("stopped").GetString());
    }

    /// <summary>
    /// A reply outlives the tap that sent it. WebKit interrupts the AudioContext for a call, Siri, dictation or the app
    /// going to the background; a chunk scheduled on a context that is not running is never heard, so a chunk that
    /// arrives then resumes it (the app's WebView lets audio start without a gesture), and so does the page coming back
    /// while the reply is still being spoken.
    /// </summary>
    [WebJavaScriptFact]
    public void AChunkThatArrivesWhileTheAudioIsInterruptedResumesIt()
    {
        JsonElement result = Run(SpeakingModel + "R['/api/chat'] = { __sse: [" + SpokenFrames + "] };", FakeAudio + """
            var resumes = [];
            window.AudioContext.prototype.resume = function () {
              resumes.push(this.state);
              if (this.state !== 'interrupted') this.state = 'running';
              return Promise.resolve();
            };
            return settle(20).then(function () {
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              var primed = resumes.length;
              // A call comes in as the first chunk is scheduled: the context is interrupted for the rest of the reply.
              var create = window.AudioContext.prototype.createBufferSource;
              window.AudioContext.prototype.createBufferSource = function () {
                var source = create.call(this), context = this, start = source.start;
                source.start = function (at) { start.call(source, at); context.state = 'interrupted'; };
                return source;
              };
              __page.byId['text'].value = 'Say hello to Anna.';
              __page.byId['send'].dispatch('click');
              return settle(30).then(function () {
                var duringReply = resumes.slice(primed);
                // The app comes back to the front with the reply still scheduled.
                document.visibilityState = 'hidden'; document.dispatch('visibilitychange');
                document.visibilityState = 'visible'; document.dispatch('visibilitychange');
                return settle(10).then(function () { return {
                  primed: primed, duringReply: duringReply, afterReturn: resumes.length - primed - duringReply.length,
                  scheduled: scheduled.length, errors: __page.errorNotices()
                }; });
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        // The fake context starts running, so the switch has nothing to resume.
        Assert.Equal(0, result.GetProperty("primed").GetInt32());
        // The chunk that arrived while it was interrupted asked for it back, and was still scheduled.
        Assert.Equal(new[] { "interrupted" }, result.GetProperty("duringReply").EnumerateArray().Select(s => s.GetString()));
        Assert.Equal(2, result.GetProperty("scheduled").GetInt32());
        Assert.Equal(1, result.GetProperty("afterReturn").GetInt32());
        Assert.Empty(result.GetProperty("errors").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void AModelThatDoesNotSpeakShowsNoSwitchAndAsksForNoSpeech()
    {
        JsonElement result = Run("R['/api/chat'] = { __sse: [{ token: 'Hi.' }, { done: true, sessionId: 's1' }] };", """
            return settle(20).then(function () {
              // Even a switch somebody flips cannot ask a model that does not speak.
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              __page.byId['text'].value = 'hi';
              __page.byId['send'].dispatch('click');
              return settle(30).then(function () { return {
                shown: __page.byId['speechbar'].style.display,
                sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                players: __page.byId['chat'].querySelectorAll('.spoken-reply').length
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal("none", result.GetProperty("shown").GetString());
        Assert.False(result.GetProperty("sent")[0].TryGetProperty("speech", out _));
        Assert.Equal(0, result.GetProperty("players").GetInt32());
    }

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void AReopenedChatReplaysAndSavesItsSpokenReply(bool native)
    {
        JsonElement result = Run(SpeakingModel + """
            R['/api/agent/conversations'] = { conversations: [
              { id: 'saved', title: 'Hello', updatedAt: '2026-11-19T10:00:00Z', messageCount: 2 }
            ] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'Say hello.' },
                { role: 'assistant', content: 'Hello!', audioId: 'audio_old', audioUrl: '/uploads/speech-a.wav', audioSeconds: 1.25 }
              ]
            };
            """, $$"""
            var native = {{JsonSerializer.Serialize(native)}};
            return settle(30).then(function () {
              if (native) window.TensorAgent.nativeReady();
              var box = __page.byId['chat'].querySelector('.spoken-reply');
              var save = box && box.querySelector('.speech-download');
              if (!save) return { found: false };
              var event = { target: save, button: 0, defaultPrevented: false,
                preventDefault: function () { this.defaultPrevented = true; }, stopPropagation: function () {} };
              save.dispatch('click', event);
              return settle(10).then(function () { return {
                found: true,
                player: box.querySelector('audio').src,
                label: save.textContent,
                href: save.href,
                prevented: event.defaultPrevented,
                saves: __page.requests('/api/agent/events').filter(function (r) {
                  return r.body && r.body.type === 'save-audio';
                }).map(function (r) { return r.body; })
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.True(result.GetProperty("found").GetBoolean(), "the reopened spoken reply has no Save link");
        Assert.Equal("/uploads/speech-a.wav", result.GetProperty("player").GetString());
        Assert.Equal("Save audio (1.3 s)", result.GetProperty("label").GetString());
        Assert.Equal("/uploads/speech-a.wav", result.GetProperty("href").GetString());
        Assert.Equal(native, result.GetProperty("prevented").GetBoolean());
        JsonElement[] saves = result.GetProperty("saves").EnumerateArray().ToArray();
        if (native)
        {
            JsonElement save = Assert.Single(saves);
            Assert.Equal("/uploads/speech-a.wav", save.GetProperty("url").GetString());
            Assert.Equal("speech-a.wav", save.GetProperty("name").GetString());
        }
        else
        {
            Assert.Empty(saves);
        }
    }
}
