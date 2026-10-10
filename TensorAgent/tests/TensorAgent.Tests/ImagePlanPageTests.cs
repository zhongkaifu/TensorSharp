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
/// The page's half of a picture turn read against the conversation (ImageTurns on the host):
/// it says under each picture what it was made from, keeps that in the history the host
/// plans the next turn from, and lets the newest turn be answered the other way -- in place
/// of the turn it corrects, never beside it.
/// </summary>
public sealed partial class WebUiPageTests
{
    /// <summary>A chat with one picture the model drew from words, reopened.</summary>
    private const string ChatWithAPicture = """
        R['/api/agent/conversations'] = { conversations: [
          { id: 'saved', title: 'A dog', updatedAt: '2026-10-01T10:00:00Z', messageCount: 2 }
        ] };
        R['/api/sessions?conversation=saved'] = {
          sessionId: 's9', conversationId: 'saved', think: false, skills: [],
          messages: [
            { role: 'user', content: 'a dog' },
            { role: 'assistant', content: '', imageUrl: '/uploads/dog.png',
              imagePlan: 'new', imageSources: [], imagePrompt: 'a dog', imageSeed: 0 }
          ]
        };
        """;

    /// <summary>The host's frames for a follow-up it read as a change to the dog picture.</summary>
    private const string ChangedTheDog = """
        { image_plan: 'edit', image_sources: ['/uploads/dog.png'], image_prompt: 'with a hat', image_plan_reason: 'model' },
        { image_step: 1, image_steps: 40 },
        { imageUrl: '/uploads/hat.png', width: 1024, height: 1024,
          imagePlan: 'edit', imagePlanReason: 'model', imageSources: ['dog.png'], imagePrompt: 'with a hat', imageSeed: 0, imageMask: null },
        { done: true, sessionId: 's9' }
        """;

    /// <summary>The host's frames once told to draw the words as a new picture.</summary>
    private const string DrewItNew = """
        { image_plan: 'new', image_sources: [], image_prompt: 'with a hat', image_plan_reason: 'asked' },
        { imageUrl: '/uploads/fresh.png', width: 1024, height: 1024,
          imagePlan: 'new', imageSources: [], imagePrompt: 'with a hat', imageSeed: 0, imageMask: null },
        { done: true, sessionId: 's9' }
        """;

    private const string SendFollowUp = """
        __page.byId['text'].value = 'with a hat';
        __page.byId['send'].dispatch('click');
        """;

    [WebJavaScriptFact]
    public void APictureSaysWhatItWasMadeFromAndTheHistoryKeepsItForTheNextTurn()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + "R['/api/chat'] = { __sse: [" + ChangedTheDog + "] };",
            SendFollowUp + """
            return settle(30).then(function () {
              var plans = __page.byId['chat'].querySelectorAll('.image-plan');
              var newest = plans[plans.length - 1];
              var history = window.TensorAgent.history();
              var made = JSON.parse(JSON.stringify(history[history.length - 1]));
              var offered = newest.querySelectorAll('button').map(function (b) { return b.textContent; });
              __page.byId['text'].value = 'make it blue';
              __page.byId['send'].dispatch('click');
              return settle(30).then(function () { return {
                captions: plans.map(function (p) { return p.children[0].textContent; }),
                thumb: newest.querySelector('img').src,
                offered: offered,
                offeredAfterNextSend: newest.querySelectorAll('button').length,
                progress: __page.progress(),
                made: made,
                sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                errors: __page.errorNotices()
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(new[] { "New picture", "Changed the picture above" }, Strings(result, "captions"));
        Assert.Equal("/uploads/dog.png", result.GetProperty("thumb").GetString());
        Assert.Equal(new[] { "Make a new picture instead" }, Strings(result, "offered"));
        // Only the newest turn can be answered another way.
        Assert.Equal(0, result.GetProperty("offeredAfterNextSend").GetInt32());
        Assert.Contains("Changing the picture…", Strings(result, "progress"));

        JsonElement made = result.GetProperty("made");
        Assert.Equal("/uploads/hat.png", made.GetProperty("imageUrl").GetString());
        Assert.Equal("edit", made.GetProperty("imagePlan").GetString());
        Assert.Equal("model", made.GetProperty("imagePlanReason").GetString());
        Assert.Equal(new[] { "dog.png" }, Strings(made, "imageSources"));
        Assert.Equal("with a hat", made.GetProperty("imagePrompt").GetString());
        Assert.False(made.TryGetProperty("imageMask", out _));

        // The next request carries it back to the host, which plans from it.
        JsonElement next = result.GetProperty("sent")[1].GetProperty("messages")[3];
        Assert.Equal("/uploads/hat.png", next.GetProperty("imageUrl").GetString());
        Assert.Equal("edit", next.GetProperty("imagePlan").GetString());
        Assert.Equal(new[] { "dog.png" }, Strings(next, "imageSources"));
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// The app's GPU gate runs a turn again after a fault, and the second run plans again:
    /// the plan that holds is the last one, shown once.
    /// </summary>
    [WebJavaScriptFact]
    public void TheLastPlanIsTheOneThatHolds()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + """
            R['/api/chat'] = { __sse: [
              { image_plan: 'edit', image_sources: ['/uploads/dog.png'], image_prompt: 'with a hat', image_plan_reason: 'unavailable' },
              { image_step: 1, image_steps: 40 },
              { restart: 'The GPU was taken away while the app was in the background. Starting this answer again.', replace: '' },
            """ + DrewItNew + """
            ] };
            """, SendFollowUp + """
            return settle(30).then(function () {
              var history = window.TensorAgent.history();
              return {
                captions: __page.byId['chat'].querySelectorAll('.image-plan').map(function (p) { return p.children[0].textContent; }),
                plan: history[history.length - 1].imagePlan
              };
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(new[] { "New picture", "New picture" }, Strings(result, "captions"));
        Assert.Equal("new", result.GetProperty("plan").GetString());
    }

    /// <summary>
    /// The other reading sends the same words again with the reading spelled out, and the
    /// turn it corrects is gone from the screen and from the history the host saves.
    /// </summary>
    [WebJavaScriptFact]
    public void TheOtherReadingAnswersTheSameWordsInPlaceOfTheTurn()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + """
            var chatCalls = 0;
            R['/api/chat'] = function () {
              chatCalls++;
              return chatCalls === 1 ? { __sse: [
            """ + ChangedTheDog + """
              ] } : { __sse: [
            """ + DrewItNew + """
              ] };
            };
            """, SendFollowUp + """
            return settle(30).then(function () {
              var override = __page.byId['chat'].querySelector('.image-redo').querySelector('button');
              override.dispatch('click');
              return settle(30).then(function () { return {
                sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                transcript: __page.transcript(),
                history: window.TensorAgent.history().map(function (m) { return m.imageUrl || m.content; }),
                errors: __page.errorNotices()
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        JsonElement redo = result.GetProperty("sent")[1];
        JsonElement[] messages = redo.GetProperty("messages").EnumerateArray().ToArray();
        Assert.Equal(3, messages.Length);
        Assert.Equal("with a hat", messages[2].GetProperty("content").GetString());
        Assert.Equal("new", messages[2].GetProperty("imageIntent").GetString());
        Assert.False(messages[2].TryGetProperty("imageSource", out _));
        Assert.DoesNotContain("hat.png", redo.ToString(), StringComparison.Ordinal);

        Assert.Equal(new[] { "a dog", "/uploads/dog.png", "with a hat", "/uploads/fresh.png" }, Strings(result, "history"));
        JsonElement[] turns = result.GetProperty("transcript").EnumerateArray().Where(t => t.GetProperty("role").GetString() is not null).ToArray();
        Assert.Equal(new[] { "user", "assistant", "user", "assistant" }, turns.Select(t => t.GetProperty("role").GetString()));
        Assert.DoesNotContain("hat.png", result.GetProperty("transcript").ToString(), StringComparison.Ordinal);
        Assert.Equal("/uploads/fresh.png", turns[3].GetProperty("media")[0].GetProperty("src").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// A wrong reading is cheapest to correct before the picture is finished: the other
    /// reading is offered as soon as the plan arrives, and taking it replaces the turn that
    /// is still drawing.
    /// </summary>
    [WebJavaScriptFact]
    public void TheOtherReadingReplacesATurnThatIsStillDrawing()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + """
            var chatCalls = 0;
            R['/api/chat'] = function () {
              chatCalls++;
              return chatCalls === 1 ? { __sse: [
                { image_plan: 'edit', image_sources: ['/uploads/dog.png'], image_prompt: 'with a hat', image_plan_reason: 'model' },
                { image_step: 1, image_steps: 40 }
              ], __then: 'hang' } : { __sse: [
            """ + DrewItNew + """
              ] };
            };
            """, SendFollowUp + """
            return settle(20).then(function () {
              var drawing = __page.byId['chat'].querySelector('.image-redo').querySelector('button');
              drawing.dispatch('click');
              return settle(30).then(function () { return {
                sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                roles: __page.transcript().map(function (t) { return t.role; }),
                history: window.TensorAgent.history().map(function (m) { return m.imageUrl || m.content; }),
                sendLabel: __page.byId['send'].textContent,
                errors: __page.errorNotices()
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        JsonElement[] sent = result.GetProperty("sent").EnumerateArray().ToArray();
        Assert.Equal(2, sent.Length);
        Assert.Equal("new", sent[1].GetProperty("messages")[2].GetProperty("imageIntent").GetString());
        Assert.Equal(new[] { "user", "assistant", "user", "assistant" }, Strings(result, "roles"));
        Assert.Equal(new[] { "a dog", "/uploads/dog.png", "with a hat", "/uploads/fresh.png" }, Strings(result, "history"));
        Assert.Equal("➤", result.GetProperty("sendLabel").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// A turn too unsure to guess asks, with one button per reading; a tap sends the words
    /// again with that reading in place of the question, and a reopened chat still offers them.
    /// </summary>
    [WebJavaScriptFact]
    public void AnUnsureTurnOffersItsReadingsAndATapAnswersInPlaceOfTheQuestion()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + """
            var chatCalls = 0;
            R['/api/chat'] = function () {
              chatCalls++;
              return chatCalls === 1 ? { __sse: [
                { token: "I'm not sure what you meant. Choose one:" },
                { image_choice: [ { intent: 'edit', source: 'dog.png' }, { intent: 'new', source: null }, { intent: 'again', source: 'dog.png' } ] },
                { done: true, sessionId: 's9' }
              ] } : { __sse: [
            """ + ChangedTheDog + """
              ] };
            };
            """, SendFollowUp + """
            return settle(30).then(function () {
              var choice = __page.byId['chat'].querySelector('.image-choice');
              var labels = choice.querySelectorAll('button').map(function (b) { return b.textContent; });
              var thumbs = choice.querySelectorAll('img').map(function (i) { return i.src; });
              var history = window.TensorAgent.history();
              var asked = JSON.parse(JSON.stringify(history[history.length - 1]));
              choice.querySelectorAll('button')[0].dispatch('click');
              return settle(30).then(function () { return {
                labels: labels, thumbs: thumbs, asked: asked,
                sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                text: __page.transcript().map(function (t) { return t.html; }).join(' '),
                history: window.TensorAgent.history().map(function (m) { return m.imageUrl || m.content; }),
                errors: __page.errorNotices()
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(new[] { "Change the picture above", "Make a new picture", "Make another version" }, Strings(result, "labels"));
        Assert.Equal(new[] { "/uploads/dog.png", "/uploads/dog.png" }, Strings(result, "thumbs"));
        JsonElement asked = result.GetProperty("asked");
        Assert.Equal("I'm not sure what you meant. Choose one:", asked.GetProperty("content").GetString());
        Assert.Equal(3, asked.GetProperty("imageChoices").GetArrayLength());

        JsonElement[] messages = result.GetProperty("sent")[1].GetProperty("messages").EnumerateArray().ToArray();
        Assert.Equal(3, messages.Length);
        Assert.Equal("edit", messages[2].GetProperty("imageIntent").GetString());
        Assert.Equal("dog.png", messages[2].GetProperty("imageSource").GetString());
        Assert.DoesNotContain("not sure", result.GetProperty("text").GetString(), StringComparison.Ordinal);
        Assert.Equal(new[] { "a dog", "/uploads/dog.png", "with a hat", "/uploads/hat.png" }, Strings(result, "history"));
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// A reopened change of an earlier picture compares with, and edits again from, the picture
    /// it was made from -- not from whatever the newest message attached, which for a
    /// follow-up is nothing.
    /// </summary>
    [WebJavaScriptFact]
    public void AReopenedChangeComparesWithAndEditsAgainThePictureItWasMadeFrom()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/agent/conversations'] = { conversations: [
              { id: 'saved', title: 'A dog', updatedAt: '2026-10-01T10:00:00Z', messageCount: 4 }
            ] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'a dog' },
                { role: 'assistant', content: '', imageUrl: '/uploads/dog.png',
                  imagePlan: 'new', imageSources: [], imagePrompt: 'a dog', imageSeed: 0 },
                { role: 'user', content: 'with a hat' },
                { role: 'assistant', content: '', imageUrl: '/uploads/hat.png',
                  imagePlan: 'edit', imageSources: ['dog.png'], imagePrompt: 'with a hat', imageSeed: 0 }
              ]
            };
            """, """
            return settle(30).then(function () {
              var plans = __page.byId['chat'].querySelectorAll('.image-plan');
              var actions = __page.byId['chat'].querySelectorAll('.image-edit-actions');
              var newest = actions[actions.length - 1];
              var picture = newest.parentNode.querySelector('img');
              var buttons = newest.querySelectorAll('button');
              buttons[0].dispatch('click');
              var original = picture.src;
              buttons[0].dispatch('click');
              var shown = picture.src;
              // An empty composer, as the WebView's is: Edit again fills it.
              __page.byId['text'].value = '';
              buttons[1].dispatch('click');
              return settle(5).then(function () { return {
                captions: plans.map(function (p) { return p.children[0].textContent; }),
                offered: plans.map(function (p) { return p.querySelectorAll('button').length; }),
                labels: buttons.map(function (b) { return b.textContent; }),
                original: original, shown: shown,
                prompt: __page.byId['text'].value,
                chips: __page.byId['chips'].querySelectorAll('.nm').map(function (n) { return n.textContent; }),
                errors: __page.errorNotices()
              }; });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(new[] { "New picture", "Changed the picture above" }, Strings(result, "captions"));
        Assert.Equal(new[] { 0, 1 }, result.GetProperty("offered").EnumerateArray().Select(n => n.GetInt32()));
        Assert.Equal(new[] { "Compare original", "Edit again" }, Strings(result, "labels").Take(2));
        Assert.Equal("/uploads/dog.png", result.GetProperty("original").GetString());
        Assert.Equal("/uploads/hat.png", result.GetProperty("shown").GetString());
        Assert.Equal("with a hat", result.GetProperty("prompt").GetString());
        Assert.Equal(new[] { "Generated image" }, Strings(result, "chips"));
        Assert.Empty(Strings(result, "errors"));
    }

    /// <summary>
    /// A turn that failed or was stopped after its plan arrived made nothing, so it left
    /// nothing in the history -- but its request is still the newest message there, and the
    /// other reading is exactly what the user is likely to want next. The chip answers it.
    /// </summary>
    [WebJavaScriptTheory]
    [InlineData("failed")]
    [InlineData("stopped")]
    public void ATurnThatMadeNothingCanStillBeAnsweredTheOtherWay(string ending)
    {
        string firstTurn = ending == "failed"
            ? """
              { __sse: [
                { image_plan: 'edit', image_sources: ['/uploads/dog.png'], image_prompt: 'with a hat', image_plan_reason: 'model' },
                { done: true, error: 'The picture could not be made.', sessionId: 's9' }
              ] }
              """
            : """
              { __sse: [
                { image_plan: 'edit', image_sources: ['/uploads/dog.png'], image_prompt: 'with a hat', image_plan_reason: 'model' },
                { image_step: 1, image_steps: 40 }
              ], __then: 'hang' }
              """;
        string end = ending == "failed" ? "" : "window.TensorAgent.stop();";
        JsonElement result = Run(ImageModel + ChatWithAPicture + """
            R['/api/agent/turns'] = { turn: null };
            var chatCalls = 0;
            R['/api/chat'] = function () {
              chatCalls++;
              return chatCalls === 1 ?
            """ + firstTurn + """
              : { __sse: [
            """ + DrewItNew + """
              ] };
            };
            """, SendFollowUp + """
            return settle(20).then(function () {
            """ + end + """
              return settle(10).then(function () {
                var before = window.TensorAgent.history().length;
                var chip = __page.byId['chat'].querySelector('.image-redo').querySelector('button');
                chip.dispatch('click');
                return settle(30).then(function () { return {
                  before: before,
                  sent: __page.requests('/api/chat').map(function (r) { return r.body; }),
                  // The turns, not the notices between them: a failure's notice says what
                  // happened to the first attempt, and stays.
                  roles: __page.byId['chat'].children
                    .filter(function (c) { return c.className.indexOf('notice') < 0; })
                    .map(function (c) { return /(^|\s)me(\s|$)/.test(c.className) ? 'user' : 'assistant'; }),
                  history: window.TensorAgent.history().map(function (m) { return m.imageUrl || m.content; })
                }; });
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(3, result.GetProperty("before").GetInt32());
        JsonElement[] sent = result.GetProperty("sent").EnumerateArray().ToArray();
        Assert.Equal(2, sent.Length);
        JsonElement[] messages = sent[1].GetProperty("messages").EnumerateArray().ToArray();
        Assert.Equal(3, messages.Length);
        Assert.Equal("with a hat", messages[2].GetProperty("content").GetString());
        Assert.Equal("new", messages[2].GetProperty("imageIntent").GetString());
        Assert.Equal(new[] { "a dog", "/uploads/dog.png", "with a hat", "/uploads/fresh.png" }, Strings(result, "history"));
        Assert.Equal(new[] { "user", "assistant", "user", "assistant" }, Strings(result, "roles"));
    }

    /// <summary>
    /// The other readings re-send the request for the image model to read. With another model
    /// loaded they would send it to that one, which would answer the words as prose in place
    /// of the picture, so they are taken away when the model changes, and a button kept from
    /// before does nothing.
    /// </summary>
    [WebJavaScriptFact]
    public void AnotherModelTakesTheOtherReadingsAway()
    {
        JsonElement result = Run(ImageModel + ChatWithAPicture + "R['/api/chat'] = { __sse: [" + ChangedTheDog + "] };",
            SendFollowUp + """
            return settle(30).then(function () {
              var offered = __page.byId['chat'].querySelectorAll('.image-redo').length;
              var chip = __page.byId['chat'].querySelector('.image-redo').querySelector('button');
              R['/api/models'] = { loaded: 'gemma.gguf', architecture: 'gemma3', loadedBackend: 'ggml_metal', visionReady: true };
              window.TensorAgent.refreshModel();
              return settle(10).then(function () {
                chip.dispatch('click');
                return settle(10).then(function () { return {
                  offered: offered,
                  left: __page.byId['chat'].querySelectorAll('.image-redo').length,
                  sent: __page.requests('/api/chat').length,
                  history: window.TensorAgent.history().map(function (m) { return m.imageUrl || m.content; })
                }; });
              });
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.Equal(1, result.GetProperty("offered").GetInt32());
        Assert.Equal(0, result.GetProperty("left").GetInt32());
        Assert.Equal(1, result.GetProperty("sent").GetInt32());
        Assert.Equal(new[] { "a dog", "/uploads/dog.png", "with a hat", "/uploads/hat.png" }, Strings(result, "history"));
    }

    /// <summary>A picture changed on a guess, because the image model could not be asked, still says so when the chat is reopened.</summary>
    [WebJavaScriptFact]
    public void AGuessIsStillSaidWhenTheChatIsReopened()
    {
        JsonElement result = Run(ImageModel + """
            R['/api/agent/conversations'] = { conversations: [
              { id: 'saved', title: 'A dog', updatedAt: '2026-10-01T10:00:00Z', messageCount: 4 }
            ] };
            R['/api/sessions?conversation=saved'] = {
              sessionId: 's9', conversationId: 'saved', think: false, skills: [],
              messages: [
                { role: 'user', content: 'a dog' },
                { role: 'assistant', content: '', imageUrl: '/uploads/dog.png',
                  imagePlan: 'new', imagePlanReason: 'first', imageSources: [], imagePrompt: 'a dog', imageSeed: 0 },
                { role: 'user', content: 'with a hat' },
                { role: 'assistant', content: '', imageUrl: '/uploads/hat.png',
                  imagePlan: 'edit', imagePlanReason: 'unavailable', imageSources: ['dog.png'], imagePrompt: 'with a hat', imageSeed: 0 }
              ]
            };
            """, """
            return settle(30).then(function () {
              return { plans: __page.byId['chat'].querySelectorAll('.image-plan').map(function (p) { return p.textContent; }) };
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        string[] plans = Strings(result, "plans");
        Assert.Equal(2, plans.Length);
        Assert.DoesNotContain("Couldn't check what you meant", plans[0], StringComparison.Ordinal);
        Assert.Contains("Couldn't check what you meant", plans[1], StringComparison.Ordinal);
    }
}
