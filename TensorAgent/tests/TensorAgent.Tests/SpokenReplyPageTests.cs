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
        { notice: 'This reply is spoken, so it runs without tools.', code: 'speech_direct', without: ['tools'] },
        { token: 'Hello, ' }, { audio: 'AAAAQADA/38=', sampleRate: 24000 },
        { token: 'Anna!' }, { audio: 'AIAAIA==', sampleRate: 24000 },
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
        // Once in each spoken turn, in the interface language (the route answers both turns with the same frames).
        Assert.Equal(Enumerable.Repeat("This reply is spoken, so it runs without reasoning, tools or skills.", 2),
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
