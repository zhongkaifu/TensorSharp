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
/// An answer as it streams, run: what a paint costs as the answer grows, that a spoken reply's chunks are scheduled
/// as they are read rather than after a paint, and that the chat follows the newest line only while the user is at
/// the bottom. Reported from the phone: a long spoken reply's speech stopped after a few sentences while the text was
/// still coming, and the chat shook and sprang back when scrolled up during an answer.
/// </summary>
public sealed partial class WebUiPageTests
{
    /// <summary>A long answer, a token at a time: ten paragraphs of a dozen sentences, a code block, a table.</summary>
    private static string LongAnswerFrames(out string answer)
    {
        var pieces = new List<string>();
        for (int p = 0; p < 10; p++)
        {
            for (int s = 0; s < 12; s++)
                foreach (string word in $"Paragraph {p} sentence {s} has **some** words in it.".Split(' '))
                    pieces.Add(word + " ");
            pieces.Add(p == 4 ? "\n\n```js\nvar x = 1;\n\nvar y = 2;\n```\n\n" : p == 7 ? "\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n" : "\n\n");
        }
        pieces.Add("The end.");
        answer = string.Concat(pieces);
        return string.Join(",\n", pieces.Select(piece => "{ token: " + JsonSerializer.Serialize(piece) + " }"));
    }

    [WebJavaScriptFact]
    public void ALongAnswersPaintsCostTheBlockBeingWrittenNotTheWholeAnswer()
    {
        string frames = LongAnswerFrames(out string answer);
        // A real millisecond per read, so the page paints between reads as it does on the phone.
        JsonElement result = Run("R['/api/chat'] = { __sse: [" + frames + ", { done: true, sessionId: 's1', tokenCount: 1 }], __delay: 1 };", """
            var writes = [], recording = false;
            __page.onHtml = function (element, v) { if (recording && v) writes.push(v.length); };
            return settle(20).then(function () {
              recording = true;
              __page.byId['text'].value = 'Write a lot.';
              __page.byId['send'].dispatch('click');
              function done(left) {
                if (left <= 0 || (!window.TensorAgent.isGenerating() && window.TensorAgent.history().length >= 2)) return null;
                return wait(50).then(function () { return done(left - 1); });
              }
              return wait(50).then(function () { return done(400); }).then(function () {
                recording = false;
                var turns = __page.transcript(), html = turns[turns.length - 1].html;
                var sum = 0, max = 0;
                writes.forEach(function (n) { sum += n; if (n > max) max = n; });
                return {
                  writes: writes.length, longest: max, total: sum, htmlLength: html.length,
                  shows: [
                    html.indexOf('<p>Paragraph 0 sentence 0 has <strong>some</strong> words in it.') >= 0,
                    html.indexOf('<pre><code>var x = 1;\n\nvar y = 2;\n</code></pre>') >= 0,
                    html.indexOf('<table><thead><tr><th>a</th><th>b</th></tr></thead>') >= 0,
                    html.indexOf('<p>The end.</p>') >= 0,
                  ],
                  content: window.TensorAgent.history()[1].content,
                };
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.True(result.GetProperty("writes").GetInt32() > 100, "the answer was not painted as it streamed");
        int htmlLength = result.GetProperty("htmlLength").GetInt32();
        // What the bubble shows is the answer rendered as one piece would be: paragraphs, the code block with its blank
        // line, the table, the last line.
        Assert.All(result.GetProperty("shows").EnumerateArray(), shown => Assert.True(shown.GetBoolean()));
        Assert.Equal(answer, result.GetProperty("content").GetString());
        // No paint writes more than about a paragraph, however long the answer has grown: a paragraph here renders to
        // ~800 characters, the whole answer to ~8,000. Rendering the whole answer at every read wrote all of it each
        // time (~1,100 reads: over four million characters); now each settled block is written once, and only the block
        // still being written again.
        int longestPaint = result.GetProperty("longest").GetInt32(), total = result.GetProperty("total").GetInt32();
        Assert.True(longestPaint < 1500, $"one paint wrote {longestPaint} characters");
        Assert.True(total < 80 * htmlLength, $"the stream wrote {total} characters for a {htmlLength}-character answer");
    }

    [WebJavaScriptFact]
    public void ASpokenRepliesChunksAreScheduledAsTheyAreReadNotAfterThePaint()
    {
        JsonElement result = Run(SpeakingModel + """
            R['/api/chat'] = { __sse: [
              { token: 'Hello, ' }, { audio: 'AAAAQADA/38=', sampleRate: 24000 },
              { token: 'Anna! ' }, { audio: 'AIAAIA==', sampleRate: 24000 },
              { token: 'Bye.' }, { done: true, sessionId: 's1', tokenCount: 3, audioId: 'audio_x' }
            ] };
            """, FakeAudio + """
            var order = [];
            // The answer's words, not the question's ("Say hello to Anna.").
            __page.onHtml = function (element, v) { if (/Hello,|Bye/.test(v)) order.push('paint'); };
            var create = window.AudioContext.prototype.createBufferSource;
            window.AudioContext.prototype.createBufferSource = function () {
              var source = create.call(this), start = source.start;
              source.start = function (at) { order.push('audio'); start.call(source, at); };
              return source;
            };
            return settle(20).then(function () {
              __page.byId['speech-toggle'].checked = true;
              __page.byId['speech-toggle'].dispatch('change');
              __page.byId['text'].value = 'Say hello to Anna.';
              __page.byId['send'].dispatch('click');
              return settle(40).then(function () { return { order: order, scheduled: scheduled.length }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        string[] order = result.GetProperty("order").EnumerateArray().Select(o => o.GetString()!).ToArray();
        Assert.Equal(2, result.GetProperty("scheduled").GetInt32());
        // The reads arrive back to back: both chunks are scheduled before the text they came with is painted, which
        // happens once, on the next frame, and once more as the finished answer.
        Assert.Equal(new[] { "audio", "audio" }, order.Take(2));
        Assert.Contains("paint", order);
    }

    /// <summary>
    /// The chat follows the answer while the user is at the bottom. A drag upward (even one of a few pixels, which
    /// used to count as "at the bottom" and was taken back on the next paint) stops it: the position holds for the rest
    /// of the stream, the arrow back appears, and the arrow (or coming back to the bottom) resumes following.
    /// </summary>
    [WebJavaScriptTheory]
    [InlineData(300)]
    [InlineData(40)]
    public void ScrollingUpDuringAnAnswerHoldsThePositionUntilTheUserComesBack(int up)
    {
        string frames = string.Join(",\n", Enumerable.Range(0, 80).Select(i => "{ token: 'word" + i + " ' }"));
        JsonElement result = Run("R['/api/chat'] = { __sse: [" + frames + ", { done: true, sessionId: 's1', tokenCount: 80 }], __delay: 4 };", $$"""
            var up = {{up}};
            var chat = __page.byId['chat'];
            chat.clientHeight = 500; chat.scrollHeight = 2000; chat.scrollTop = 1500;
            // Every paint of the answer makes the chat taller; where it scrolls is up to the page.
            __page.onHtml = function (element, v) { if (/word/.test(v)) chat.scrollHeight += 20; };
            function sample() { return [chat.scrollTop, chat.scrollHeight]; }
            var out = {};
            return settle(20).then(function () {
              __page.byId['text'].value = 'Write.';
              __page.byId['send'].dispatch('click');
              return wait(60);
            }).then(function () {
              out.following = sample();
              // The user drags the chat up, mid-answer.
              chat.dispatch('touchstart', {});
              chat.dispatch('touchmove', {});
              chat.scrollTop = chat.scrollHeight - chat.clientHeight - up;
              out.held = chat.scrollTop;
              chat.dispatch('scroll', {});
              chat.dispatch('touchend', {});
              return wait(40);
            }).then(function () { out.held1 = sample(); return wait(40);
            }).then(function () { out.held2 = sample();
              var arrow = __page.byId['composer'].querySelector('.jump-latest');
              out.shown = arrow ? !arrow.hidden : null;
              if (arrow) arrow.dispatch('click');
              out.afterClick = sample();
              out.hiddenAfterClick = arrow ? arrow.hidden : null;
              return wait(40);
            }).then(function () { out.back = sample(); return settle(200);
            }).then(function () { out.end = sample(); return out; });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        int[] At(string name) => result.GetProperty(name).EnumerateArray().Select(v => v.GetInt32()).ToArray();
        // Before the drag the chat followed the answer down.
        Assert.Equal(At("following")[1], At("following")[0]);
        int held = result.GetProperty("held").GetInt32();
        // While scrolled up the answer kept growing below, and the position stayed where the user put it.
        Assert.True(At("held2")[1] > At("held1")[1] && At("held1")[1] > At("following")[1], "the answer did not keep streaming");
        Assert.Equal(held, At("held1")[0]);
        Assert.Equal(held, At("held2")[0]);
        Assert.True(result.GetProperty("shown").GetBoolean(), "no way back to the newest line was offered");
        // The arrow takes the user back, goes away, and the chat follows again to the end of the answer.
        Assert.Equal(At("afterClick")[1], At("afterClick")[0]);
        Assert.True(result.GetProperty("hiddenAfterClick").GetBoolean());
        Assert.True(At("back")[1] > At("afterClick")[1], "the answer did not keep streaming");
        Assert.Equal(At("back")[1], At("back")[0]);
        Assert.Equal(At("end")[1], At("end")[0]);
    }
}
