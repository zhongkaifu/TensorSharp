using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    private const string ModelLoadClock = """
        var modelLoadIntervals = {}, modelLoadIntervalId = 0;
        setInterval = function (callback) {
          modelLoadIntervals[++modelLoadIntervalId] = callback;
          return modelLoadIntervalId;
        };
        clearInterval = function (id) { delete modelLoadIntervals[id]; };
        function tickModelLoad() {
          Object.keys(modelLoadIntervals).forEach(function (id) {
            if (modelLoadIntervals[id]) modelLoadIntervals[id]();
          });
        }
        """;

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void ReturningDuringModelLoadingUpdatesImageCapabilitiesWhenQwenFinishes(bool previousModelStillVisible)
    {
        JsonElement result = Run(ModelLoadClock + (previousModelStillVisible ? "" : "R['/api/models'] = {};"), """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image' });
            __page.byId['text'].value = 'edit just this area';
            R['/api/agent/engine'] = { model: { id: 'qwen-image', name: 'Qwen-Image', state: 'Loading', loading: true } };
            // MainPage calls this once when returning from Models. The load is still
            // running, and /api/models may continue to report the previous model.
            window.TensorAgent.refreshModel();
            return settle(20).then(function () {
              var before = __page.byId['chips'].querySelectorAll('.filechip').length;
              var beforeHint = __page.byId['text'].placeholder;
              tickModelLoad();
              return settle(20).then(function () {
                var whileLoading = Object.keys(modelLoadIntervals).length;
                // Complete as the next poll reads the load status. Capabilities must
                // be fetched AFTER this status, so the terminal poll cannot leave the
                // old model's controls on screen and then stop polling.
                R['/api/agent/engine'] = function () {
                  R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image' };
                  return { model: { id: 'qwen-image', name: 'Qwen-Image', state: 'Loaded', loading: false } };
                };
                tickModelLoad();
                return settle(20).then(function () {
                  var reads = __page.requests('/api/models').length;
                  tickModelLoad();
                  return settle(20).then(function () { return {
                    before: before, beforeHint: beforeHint, whileLoading: whileLoading,
                    after: __page.byId['chips'].querySelectorAll('.filechip').map(function (b) { return b.textContent; }),
                    afterHint: __page.byId['text'].placeholder, sendEnabled: !__page.byId['send'].disabled,
                    stopped: Object.keys(modelLoadIntervals).length === 0,
                    noMoreReads: reads === __page.requests('/api/models').length,
                    draft: __page.byId['text'].value, attached: window.TensorAgent.attachmentCount(),
                    errors: __page.errorNotices()
                  }; });
                });
              });
            });
            """);
        Assert.Equal(1, result.GetProperty("before").GetInt32());
        Assert.DoesNotContain("Describe a picture", result.GetProperty("beforeHint").GetString());
        Assert.Equal(1, result.GetProperty("whileLoading").GetInt32());
        Assert.Equal("Select area", Assert.Single(Strings(result, "after")));
        Assert.Contains("Describe a picture", result.GetProperty("afterHint").GetString());
        Assert.True(result.GetProperty("sendEnabled").GetBoolean());
        Assert.True(result.GetProperty("stopped").GetBoolean());
        Assert.True(result.GetProperty("noMoreReads").GetBoolean());
        Assert.Equal("edit just this area", result.GetProperty("draft").GetString());
        Assert.Equal(1, result.GetProperty("attached").GetInt32());
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void MaskedEditWaitsForTheChosenLorasBeforeSending()
    {
        JsonElement result = Run(MaskImageModel + TwoStyles + Held + """
            R['/api/chat'] = { __sse: [
              { image_step: 1, image_steps: 6, image_loras: ['Film Stills'], preview: 'data:image/png;base64,AAAA' },
              { imageUrl: '/uploads/result.png' }, { done: true, sessionId: 's1' }
            ] };
            """, """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'mask.png' });
            __page.byId['text'].value = 'make only the selected area blue';
            return openLoras().then(function () {
              hold('/api/agent/loras/choice');
              var film = loraRow('film').querySelectorAll('INPUT')[0];
              film.checked = true; film.dispatch('change');
              __page.byId['sheet-bg'].dispatch('click');
              __page.byId['send'].dispatch('click');
              var early = __page.requests('/api/chat').length;
              var draft = __page.byId['text'].value;
              return settle(20).then(function () {
                var pending = heldFor('/api/agent/loras/choice')[0];
                pending.answer(loraRows(pending.call.body.loras));
                return settle(20).then(function () {
                  __page.byId['send'].dispatch('click');
                  return settle(30).then(function () { return {
                    early: early, draft: draft, sent: __page.requests('/api/chat'), progress: __page.progress(),
                    media: __page.transcript().slice(-1)[0].media, errors: __page.errorNotices()
                  }; });
                });
              });
            });
            """);
        Assert.Equal(0, result.GetProperty("early").GetInt32());
        Assert.Equal("make only the selected area blue", result.GetProperty("draft").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal("mask.png", message.GetProperty("maskPath").GetString());
        Assert.Equal(new[] { "source.png" }, Strings(message, "stillImagePaths"));
        Assert.Contains("Drawing with Film Stills… step 1 of 6", Strings(result, "progress"));
        JsonElement picture = Assert.Single(result.GetProperty("media").EnumerateArray(), m => m.GetProperty("tag").GetString() == "IMG");
        Assert.Equal("/uploads/result.png", picture.GetProperty("src").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    private const string MaskImageModel = """
        R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image', visionReady: true };
        R['/api/upload'] = { ok: true, file: 'selection.png', mediaType: 'image' };
        R['/api/chat'] = { __sse: [{ imageUrl: '/uploads/result.png' }, { done: true, sessionId: 's1' }] };
        """;

    [WebJavaScriptFact]
    public void FirstPhotoOffersSelectionBeforeAndAfterSwitchingToAnImageModel()
    {
        JsonElement result = Run("", """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image' });
            var before = __page.byId['chips'].querySelectorAll('.filechip').length;
            R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image' };
            window.TensorAgent.refreshModel();
            return settle(20).then(function () { return {
              before: before, after: __page.byId['chips'].querySelectorAll('.filechip').map(function (b) { return b.textContent; })
            }; });
            """);
        Assert.Equal(1, result.GetProperty("before").GetInt32());
        Assert.Equal("Select area", Assert.Single(Strings(result, "after")));
    }

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void SelectionCanBePreparedBeforeAnImageModelAndSurvivesARefusedSend(bool chatModelLoaded)
    {
        JsonElement result = Run((chatModelLoaded ? "" : "R['/api/models'] = {};") + """
            R['/api/upload'] = { ok: true, file: 'selection.png', mediaType: 'image' };
            R['/api/chat'] = { __sse: [{ imageUrl: '/uploads/result.png' }, { done: true, sessionId: 's1' }] };
            """, """
            var editorSource;
            window.TensorSharpMaskEditor = { open: function (options) {
              editorSource = options.sourceUrl;
              return Promise.resolve({ blob: { png: true }, maskFeather: 4, maskCrop: true });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image' });
            __page.byId['text'].value = 'make only this area blue';
            __page.byId['chips'].querySelector('.mask-select').dispatch('click');
            return settle(20).then(function () {
              var hint = __page.byId['chips'].querySelector('.image-selection-hint');
              var hintText = hint && hint.textContent;
              hint.querySelector('.notice-action').dispatch('click');
              // The bridge can be called even when no model keeps the Send button
              // disabled. Neither case may consume or send an unsupported selection.
              window.TensorAgent.send();
              return settle(20).then(function () {
                var refused = {
                  sent: __page.requests('/api/chat').length, draft: __page.byId['text'].value,
                  attached: window.TensorAgent.attachmentCount(), history: window.TensorAgent.history().length,
                  label: __page.byId['chips'].querySelector('.mask-select').textContent
                };
                R['/api/models'] = { loaded: 'qwen-image.gguf', architecture: 'qwen_image' };
                window.TensorAgent.refreshModel();
                return settle(20).then(function () {
                  var hintGone = !__page.byId['chips'].querySelector('.image-selection-hint');
                  __page.byId['send'].dispatch('click');
                  return settle(20).then(function () { return {
                    source: editorSource, hint: hintText, hintGone: hintGone, refused: refused,
                    events: __page.requests('/api/agent/events').map(function (call) { return call.body; }),
                    sent: __page.requests('/api/chat'), uploads: __page.requests('/api/upload').length,
                    errors: __page.errorNotices()
                  }; });
                });
              });
            });
            """);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal("/uploads/source.png", result.GetProperty("source").GetString());
        Assert.Contains("Load Qwen-Image 2.1", result.GetProperty("hint").GetString());
        Assert.True(result.GetProperty("hintGone").GetBoolean());
        JsonElement refused = result.GetProperty("refused");
        Assert.Equal(0, refused.GetProperty("sent").GetInt32());
        Assert.Equal("make only this area blue", refused.GetProperty("draft").GetString());
        Assert.Equal(1, refused.GetProperty("attached").GetInt32());
        Assert.Equal(0, refused.GetProperty("history").GetInt32());
        Assert.Equal("Selection saved · Adjust", refused.GetProperty("label").GetString());
        Assert.Contains(result.GetProperty("events").EnumerateArray(), item =>
            item.TryGetProperty("type", out JsonElement type) && type.GetString() == "open-route"
            && item.TryGetProperty("route", out JsonElement route) && route.GetString() == "models");
        JsonElement sent = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "source.png" }, Strings(sent, "stillImagePaths"));
        Assert.Equal("selection.png", sent.GetProperty("maskPath").GetString());
        Assert.Equal("grayscale", sent.GetProperty("maskMode").GetString());
        Assert.Equal(4, sent.GetProperty("maskFeather").GetInt32());
        Assert.True(sent.GetProperty("maskCrop").GetBoolean());
        Assert.Equal("selection.png", sent.GetProperty("attachments")[0].GetProperty("maskPath").GetString());
        Assert.Equal(1, result.GetProperty("uploads").GetInt32());
        Assert.Contains("Load Qwen-Image 2.1 before sending", Assert.Single(Strings(result, "errors")));
    }

    [WebJavaScriptFact]
    public void SelectedAreaIsUploadedAndSentWithOnlyTheFirstSourcePhoto()
    {
        JsonElement result = Run(MaskImageModel, """
            var editorOptions;
            window.TensorSharpMaskEditor = { open: function (options) {
              editorOptions = options;
              return Promise.resolve({ blob: { png: true }, maskFeather: 3, maskCrop: true });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', url: '/uploads/source.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image', url: '/uploads/reference.png' });
            var controls = __page.byId['chips'].querySelectorAll('.filechip');
            var controlCount = controls.length;
            controls[0].dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'make the scarf red'; __page.byId['send'].dispatch('click');
              return settle(20).then(function () {
                return { controlCount: controlCount, options: editorOptions,
                  uploads: __page.requests('/api/upload'), sent: __page.requests('/api/chat'),
                  errors: __page.errorNotices() };
              });
            });
            """);
        Assert.Equal(2, result.GetProperty("controlCount").GetInt32());
        Assert.Equal("/uploads/source.png", result.GetProperty("options").GetProperty("sourceUrl").GetString());
        Assert.Equal("selection.png", Assert.Single(result.GetProperty("uploads").EnumerateArray()).GetProperty("parts")[0].GetProperty("fileName").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "source.png", "reference.png" }, Strings(message, "stillImagePaths"));
        Assert.Equal("selection.png", message.GetProperty("maskPath").GetString());
        Assert.Equal("grayscale", message.GetProperty("maskMode").GetString());
        Assert.Equal(3, message.GetProperty("maskFeather").GetInt32());
        Assert.True(message.GetProperty("maskCrop").GetBoolean());
        Assert.Equal("selection.png", message.GetProperty("attachments")[0].GetProperty("maskPath").GetString());
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void FailedSelectionUploadKeepsThePreviousSelectionAndDraft()
    {
        JsonElement result = Run(MaskImageModel + """
            R['/api/upload'] = { __status: 500, body: { error: 'disk full' } };
            """, """
            window.TensorSharpMaskEditor = { open: function () {
              return Promise.resolve({ blob: {}, maskFeather: 2, maskCrop: false });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'previous.png' });
            __page.byId['text'].value = 'change only this area';
            __page.byId['chips'].querySelector('.filechip').dispatch('click');
            __page.byId['send'].dispatch('click');
            var during = __page.requests('/api/chat').length;
            return settle(20).then(function () {
              var draft = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return {
                during: during, draft: draft, sent: __page.requests('/api/chat'), errors: __page.errorNotices()
              }; });
            });
            """);
        Assert.Equal(0, result.GetProperty("during").GetInt32());
        Assert.Equal("change only this area", result.GetProperty("draft").GetString());
        Assert.Equal("previous.png", Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0].GetProperty("maskPath").GetString());
        Assert.Contains("disk full", Assert.Single(Strings(result, "errors")), StringComparison.Ordinal);
    }

    [WebJavaScriptFact]
    public void RemovingSelectionRestoresWholeImageEditing()
    {
        JsonElement result = Run(MaskImageModel, """
            window.TensorSharpMaskEditor = { open: function () { return Promise.resolve({ remove: true }); } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'previous.png' });
            __page.byId['chips'].querySelector('.filechip').dispatch('click');
            return settle(20).then(function () {
              __page.byId['text'].value = 'change the style'; __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return { sent: __page.requests('/api/chat'), uploads: __page.requests('/api/upload') }; });
            });
            """);
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.False(message.TryGetProperty("maskPath", out _));
        Assert.Empty(result.GetProperty("uploads").EnumerateArray());
    }

    [WebJavaScriptFact]
    public void ResultCanCompareAndRepeatTheOriginalSelectionAndPrompt()
    {
        JsonElement result = Run(MaskImageModel, """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'mask.png', maskFeather: 5, maskCrop: true });
            __page.byId['text'].value = 'make the scarf red'; __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              var actions = __page.byId['chat'].querySelector('.image-edit-actions');
              var buttons = actions.querySelectorAll('button');
              buttons[0].dispatch('click');
              var original = __page.transcript().slice(-1)[0].media[0].src;
              buttons[0].dispatch('click');
              var edited = __page.transcript().slice(-1)[0].media[0].src;
              buttons[1].dispatch('click');
              var prompt = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return { original: original, edited: edited, prompt: prompt, sent: __page.requests('/api/chat') }; });
            });
            """);
        Assert.Equal("/uploads/source.png", result.GetProperty("original").GetString());
        Assert.Equal("/uploads/result.png", result.GetProperty("edited").GetString());
        Assert.Equal("make the scarf red", result.GetProperty("prompt").GetString());
        JsonElement message = result.GetProperty("sent")[1].GetProperty("body").GetProperty("messages")[2];
        Assert.Equal("source.png", Assert.Single(Strings(message, "stillImagePaths")));
        Assert.Equal("mask.png", message.GetProperty("maskPath").GetString());
        Assert.Equal(5, message.GetProperty("maskFeather").GetInt32());
        Assert.True(message.GetProperty("maskCrop").GetBoolean());
    }

    private const string PhotoDraftHelpers = """
        function photoChip(file) {
          return __page.byId['chips'].querySelectorAll('.editable-image').filter(function (chip) {
            return chip.querySelector('.nm').textContent === file;
          })[0];
        }
        function choosePhoto(file) { photoChip(file).querySelector('.mask-select').dispatch('click'); }
        function draftOrder() {
          return __page.byId['chips'].querySelectorAll('.chip').map(function (chip) { return chip.querySelector('.nm').textContent; });
        }
        function draftTarget() {
          return __page.byId['chips'].querySelectorAll('.editable-image').filter(function (chip) {
            return !!chip.querySelector('.image-edit-source');
          })[0].querySelector('.nm').textContent;
        }
        """;

    private static void AssertActiveSelection(JsonElement message, string? mask)
    {
        if (mask is null) Assert.False(message.TryGetProperty("maskPath", out _));
        else Assert.Equal(mask, message.GetProperty("maskPath").GetString());
        string source = message.GetProperty("stillImagePaths")[0].GetString()!;
        foreach (JsonElement attachment in message.GetProperty("attachments").EnumerateArray())
        {
            Assert.False(attachment.TryGetProperty("_maskActive", out _));
            if (mask is not null && attachment.GetProperty("file").GetString() == source)
                Assert.Equal(mask, attachment.GetProperty("maskPath").GetString());
            else
                Assert.DoesNotContain(attachment.EnumerateObject(), property => property.Name.StartsWith("mask", StringComparison.Ordinal));
        }
    }

    [WebJavaScriptTheory]
    [InlineData("second.png")]
    [InlineData("third.png")]
    public void EveryPhotoCanBecomeTheEditTargetWithoutDroppingMixedAttachments(string selected)
    {
        JsonElement result = Run(MaskImageModel, "var selected = " + JsonSerializer.Serialize(selected) + ";" + PhotoDraftHelpers + """
            var opened;
            window.TensorSharpMaskEditor = { open: function (options) {
              opened = options; return Promise.resolve({ blob: {}, maskFeather: 6, maskCrop: true });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'notes.csv', mediaType: 'text', fileBacked: true });
            window.TensorAgent.addAttachment({ ok: true, file: 'first.png', mediaType: 'image', maskPath: 'first-mask.png', maskInvert: true, maskCropPadding: 91 });
            window.TensorAgent.addAttachment({ ok: true, file: 'scanned.pdf', mediaType: 'pdf', frames: ['page.png'] });
            window.TensorAgent.addAttachment({ ok: true, file: 'second.png', mediaType: 'image', maskPath: 'second-mask.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'third.png', mediaType: 'image' });
            var controls = __page.byId['chips'].querySelectorAll('.mask-select').map(function (button) {
              return { tag: button.tagName, type: button.type, disabled: button.disabled, label: button.getAttribute('aria-label') };
            });
            __page.byId['text'].value = 'only change the selected area';
            choosePhoto(selected);
            return settle(20).then(function () {
              var order = draftOrder(), target = draftTarget(), draft = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return {
                opened: opened, controls: controls, order: order, target: target, draft: draft,
                sent: __page.requests('/api/chat'), errors: __page.errorNotices()
              }; });
            });
            """);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        JsonElement controls = result.GetProperty("controls");
        Assert.Equal(3, controls.GetArrayLength());
        Assert.Equal(new[] { "Adjust selection for first.png", "Adjust selection for second.png", "Select area in third.png" },
            controls.EnumerateArray().Select(button => button.GetProperty("label").GetString()));
        Assert.All(controls.EnumerateArray(), button =>
        {
            Assert.Equal("BUTTON", button.GetProperty("tag").GetString());
            Assert.Equal("button", button.GetProperty("type").GetString());
            Assert.False(button.GetProperty("disabled").GetBoolean());
        });
        Assert.Equal("/uploads/" + selected, result.GetProperty("opened").GetProperty("sourceUrl").GetString());
        Assert.Equal(selected == "second.png" ? "/uploads/second-mask.png" : null,
            result.GetProperty("opened").GetProperty("maskUrl").GetString());
        string other = selected == "second.png" ? "third.png" : "second.png";
        string[] order = [selected, "notes.csv", "first.png", "scanned.pdf", other];
        Assert.Equal(order, Strings(result, "order"));
        Assert.Equal(selected, result.GetProperty("target").GetString());
        Assert.Equal("only change the selected area", result.GetProperty("draft").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(order, message.GetProperty("attachments").EnumerateArray().Select(a => a.GetProperty("file").GetString()));
        Assert.Equal(new[] { selected, "first.png", other }, Strings(message, "stillImagePaths"));
        Assert.Equal(new[] { selected, "first.png", "page.png", other }, Strings(message, "imagePaths"));
        Assert.Equal(new[] { "notes.csv", "scanned.pdf" }, Strings(message, "textFilePaths"));
        AssertActiveSelection(message, "selection.png");
        Assert.Equal(6, message.GetProperty("maskFeather").GetInt32());
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptTheory]
    [InlineData("cancel")]
    [InlineData("editor-failure")]
    [InlineData("upload-failure")]
    [InlineData("remove")]
    public void AnUnsuccessfulReferenceSelectionKeepsThePreviousTargetAndDraft(string outcome)
    {
        JsonElement result = Run(MaskImageModel, "var outcome = " + JsonSerializer.Serialize(outcome) + ";" + PhotoDraftHelpers + """
            var reopened;
            window.TensorSharpMaskEditor = { open: function () {
              if (outcome === 'cancel') return Promise.resolve(null);
              if (outcome === 'editor-failure') return Promise.reject(new Error('decode failed'));
              if (outcome === 'remove') return Promise.resolve({ remove: true });
              R['/api/upload'] = { __status: 500, body: { error: 'disk full' } };
              return Promise.resolve({ blob: {}, maskFeather: 1, maskCrop: false });
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'source-mask.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'notes.csv', mediaType: 'text', fileBacked: true });
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image', maskPath: 'reference-mask.png' });
            __page.byId['text'].value = 'keep this draft';
            choosePhoto('reference.png');
            return settle(20).then(function () {
              window.TensorSharpMaskEditor.open = function (options) { reopened = options; return Promise.resolve(null); };
              choosePhoto('reference.png');
              return settle(20).then(function () {
                var order = draftOrder(), target = draftTarget(), draft = __page.byId['text'].value;
                __page.byId['send'].dispatch('click');
                return settle(20).then(function () { return {
                  reopened: reopened, order: order, target: target, draft: draft,
                  sent: __page.requests('/api/chat'), errors: __page.errorNotices()
                }; });
              });
            });
            """);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal(new[] { "source.png", "notes.csv", "reference.png" }, Strings(result, "order"));
        Assert.Equal("source.png", result.GetProperty("target").GetString());
        Assert.Equal("keep this draft", result.GetProperty("draft").GetString());
        Assert.Equal(outcome == "remove" ? null : "/uploads/reference-mask.png", result.GetProperty("reopened").GetProperty("maskUrl").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "source.png", "reference.png" }, Strings(message, "stillImagePaths"));
        AssertActiveSelection(message, "source-mask.png");
        if (outcome.EndsWith("failure", StringComparison.Ordinal)) Assert.Single(Strings(result, "errors"));
        else Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void SwitchingTargetsRetainsEachPhotosOwnSelectionUntilTheDraftIsSent()
    {
        JsonElement result = Run(MaskImageModel, PhotoDraftHelpers + """
            var opened = [], uploadIndex = 0;
            R['/api/upload'] = function () { return { ok: true, file: 'saved-' + (++uploadIndex) + '.png', mediaType: 'image' }; };
            window.TensorSharpMaskEditor = { open: function (options) {
              opened.push(options);
              return Promise.resolve(opened.length === 1 || opened.length === 3 ? { blob: {}, maskFeather: 2, maskCrop: false } : null);
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'a.png', mediaType: 'image', maskPath: 'mask-a.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'b.png', mediaType: 'image', maskPath: 'mask-b.png' });
            __page.byId['text'].value = 'preserve both selections';
            choosePhoto('b.png');
            return settle(20).then(function () {
              choosePhoto('a.png');
              return settle(20).then(function () {
                var afterCancel = draftTarget();
                choosePhoto('a.png');
                return settle(20).then(function () {
                  choosePhoto('b.png');
                  return settle(20).then(function () {
                    var beforeSend = draftTarget();
                    __page.byId['send'].dispatch('click');
                    return settle(20).then(function () { return {
                      opened: opened, afterCancel: afterCancel, beforeSend: beforeSend,
                      sent: __page.requests('/api/chat'), errors: __page.errorNotices()
                    }; });
                  });
                });
              });
            });
            """);
        Assert.Equal(new[] { "/uploads/b.png", "/uploads/a.png", "/uploads/a.png", "/uploads/b.png" },
            result.GetProperty("opened").EnumerateArray().Select(item => item.GetProperty("sourceUrl").GetString()));
        Assert.Equal(new[] { "/uploads/mask-b.png", "/uploads/mask-a.png", "/uploads/mask-a.png", "/uploads/saved-1.png" },
            result.GetProperty("opened").EnumerateArray().Select(item => item.GetProperty("maskUrl").GetString()));
        Assert.Equal("b.png", result.GetProperty("afterCancel").GetString());
        Assert.Equal("a.png", result.GetProperty("beforeSend").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "a.png", "b.png" }, Strings(message, "stillImagePaths"));
        AssertActiveSelection(message, "saved-2.png");
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void RemovingTheTargetKeepsReferenceSelectionsDormantUntilExplicitlySaved(bool restore)
    {
        JsonElement result = Run(MaskImageModel, "var restore = " + (restore ? "true;" : "false;") + PhotoDraftHelpers + """
            var reopened;
            window.TensorSharpMaskEditor = { open: function () { return Promise.resolve({ blob: {}, maskFeather: 1, maskCrop: false }); } };
            window.TensorAgent.addAttachment({ ok: true, file: 'a.png', mediaType: 'image', maskPath: 'mask-a.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'b.png', mediaType: 'image', maskPath: 'mask-b.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'c.png', mediaType: 'image' });
            __page.byId['text'].value = 'keep the remaining photos';
            choosePhoto('b.png');
            return settle(20).then(function () {
              photoChip('b.png').querySelector('.x').dispatch('click');
              window.TensorSharpMaskEditor.open = function (options) {
                reopened = options; return Promise.resolve(restore ? { blob: {}, maskFeather: 7, maskCrop: true } : null);
              };
              choosePhoto('a.png');
              return settle(20).then(function () {
                var order = draftOrder(), target = draftTarget(), draft = __page.byId['text'].value;
                __page.byId['send'].dispatch('click');
                return settle(20).then(function () { return {
                  reopened: reopened, order: order, target: target, draft: draft,
                  sent: __page.requests('/api/chat'), errors: __page.errorNotices()
                }; });
              });
            });
            """);
        Assert.Equal(new[] { "a.png", "c.png" }, Strings(result, "order"));
        Assert.Equal("a.png", result.GetProperty("target").GetString());
        Assert.Equal("keep the remaining photos", result.GetProperty("draft").GetString());
        Assert.Equal("/uploads/mask-a.png", result.GetProperty("reopened").GetProperty("maskUrl").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal(new[] { "a.png", "c.png" }, Strings(message, "stillImagePaths"));
        AssertActiveSelection(message, restore ? "selection.png" : null);
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void EditAgainRestoresTheChosenSourceAndAllSentAttachments()
    {
        JsonElement result = Run(MaskImageModel, PhotoDraftHelpers + """
            window.TensorSharpMaskEditor = { open: function () { return Promise.resolve({ blob: {}, maskFeather: 4, maskCrop: true }); } };
            window.TensorAgent.addAttachment({ ok: true, file: 'notes.csv', mediaType: 'text', fileBacked: true });
            window.TensorAgent.addAttachment({ ok: true, file: 'first.png', mediaType: 'image', maskPath: 'old-mask.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'chosen.png', mediaType: 'image' });
            __page.byId['text'].value = 'only make this area blue';
            choosePhoto('chosen.png');
            return settle(20).then(function () {
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () {
                var buttons = __page.byId['chat'].querySelector('.image-edit-actions').querySelectorAll('button');
                buttons[0].dispatch('click');
                var original = __page.transcript().slice(-1)[0].media[0].src;
                buttons[1].dispatch('click');
                var order = draftOrder(), target = draftTarget(), prompt = __page.byId['text'].value;
                var labels = __page.byId['chips'].querySelectorAll('.mask-select').map(function (button) { return button.textContent; });
                __page.byId['send'].dispatch('click');
                return settle(20).then(function () { return {
                  original: original, order: order, target: target, prompt: prompt, labels: labels,
                  sent: __page.requests('/api/chat'), errors: __page.errorNotices()
                }; });
              });
            });
            """);
        Assert.Equal("/uploads/chosen.png", result.GetProperty("original").GetString());
        Assert.Equal(new[] { "chosen.png", "notes.csv", "first.png" }, Strings(result, "order"));
        Assert.Equal("chosen.png", result.GetProperty("target").GetString());
        Assert.Equal("only make this area blue", result.GetProperty("prompt").GetString());
        Assert.Equal(new[] { "Selection saved · Adjust", "Select area" }, Strings(result, "labels"));
        JsonElement first = result.GetProperty("sent")[0].GetProperty("body").GetProperty("messages")[0];
        JsonElement repeated = result.GetProperty("sent")[1].GetProperty("body").GetProperty("messages")[2];
        Assert.Equal(first.ToString(), repeated.ToString());
        AssertActiveSelection(repeated, "selection.png");
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void EditAgainDoesNotActivateDormantMasksFromOldAttachmentMetadata()
    {
        JsonElement result = Run(MaskImageModel, PhotoDraftHelpers + """
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image' });
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image' });
            __page.byId['text'].value = 'change the overall style';
            __page.byId['send'].dispatch('click');
            return settle(20).then(function () {
              // Earlier versions persisted per-photo masks in chips even when the
              // actual completed request had no top-level active mask.
              var previous = window.TensorAgent.history()[0];
              previous.attachments[0].maskPath = 'dormant-source.png';
              previous.attachments[1].maskPath = 'dormant-reference.png';
              __page.byId['chat'].querySelector('.image-edit-actions').querySelectorAll('button')[1].dispatch('click');
              var labels = __page.byId['chips'].querySelectorAll('.mask-select').map(function (button) { return button.textContent; });
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return { labels: labels, sent: __page.requests('/api/chat'), errors: __page.errorNotices() }; });
            });
            """);
        Assert.Equal(new[] { "Select area", "Select area" }, Strings(result, "labels"));
        JsonElement repeated = result.GetProperty("sent")[1].GetProperty("body").GetProperty("messages")[2];
        Assert.Equal(new[] { "source.png", "reference.png" }, Strings(repeated, "stillImagePaths"));
        AssertActiveSelection(repeated, null);
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void DiscardingASharedSourceDoesNotActivateAReferenceSelection()
    {
        JsonElement result = Run(MaskImageModel + """
            R['/api/agent/share/claim'] = function () {
              return { share: { id: 'shared-source', title: 'Shared photo', attachments: [
                { ok: true, file: 'source.png', fileName: 'source.png', mediaType: 'image', maskPath: 'source-mask.png' }
              ] } };
            };
            R['/api/agent/share/discard'] = { ok: true };
            """, PhotoDraftHelpers + """
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image', maskPath: 'reference-mask.png' });
            __page.byId['text'].value = 'keep this draft';
            __page.byId['chips'].querySelector('.shared').querySelector('button').dispatch('click');
            return settle(20).then(function () {
              var order = draftOrder(), target = draftTarget(), label = photoChip('reference.png').querySelector('.mask-select').textContent;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return {
                order: order, target: target, label: label, sent: __page.requests('/api/chat'), errors: __page.errorNotices()
              }; });
            });
            """);
        Assert.Equal(new[] { "reference.png" }, Strings(result, "order"));
        Assert.Equal("reference.png", result.GetProperty("target").GetString());
        Assert.Equal("Selection saved · Adjust", result.GetProperty("label").GetString());
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        Assert.Equal("keep this draft", message.GetProperty("content").GetString());
        AssertActiveSelection(message, null);
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptTheory]
    [InlineData("camera.heic")]
    [InlineData("camera.HEIF")]
    public void HeicSelectionAndRepeatUseTheFullResolutionConversionWhileChipsUseThePreview(string file)
    {
        JsonElement result = Run(MaskImageModel, "var cameraFile = " + JsonSerializer.Serialize(file) + ";" + PhotoDraftHelpers + """
            var opened = [];
            window.TensorSharpMaskEditor = { open: function (options) {
              opened.push(options);
              return Promise.resolve(opened.length === 1 ? { blob: {}, maskFeather: 2, maskCrop: true } : null);
            } };
            window.TensorAgent.addAttachment({ ok: true, file: 'reference.png', mediaType: 'image' });
            window.TensorAgent.addAttachment({ ok: true, file: cameraFile, mediaType: 'image',
              previewUrl: '/uploads/camera-preview.png', editUrl: '/uploads/camera-full.png' });
            var thumbnail = photoChip(cameraFile).querySelector('img').src;
            __page.byId['text'].value = 'change just this area';
            choosePhoto(cameraFile);
            return settle(20).then(function () {
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () {
                var buttons = __page.byId['chat'].querySelector('.image-edit-actions').querySelectorAll('button');
                buttons[0].dispatch('click');
                var compared = __page.transcript().slice(-1)[0].media[0].src;
                buttons[1].dispatch('click');
                var restoredThumbnail = photoChip(cameraFile).querySelector('img').src;
                choosePhoto(cameraFile);
                return settle(20).then(function () {
                  __page.byId['send'].dispatch('click');
                  return settle(20).then(function () { return {
                    thumbnail: thumbnail, restoredThumbnail: restoredThumbnail, compared: compared,
                    opened: opened, sent: __page.requests('/api/chat'), errors: __page.errorNotices()
                  }; });
                });
              });
            });
            """);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal("/uploads/camera-preview.png", result.GetProperty("thumbnail").GetString());
        Assert.Equal("/uploads/camera-preview.png", result.GetProperty("restoredThumbnail").GetString());
        Assert.Equal("/uploads/camera-full.png", result.GetProperty("compared").GetString());
        Assert.Equal(new[] { "/uploads/camera-full.png", "/uploads/camera-full.png" },
            result.GetProperty("opened").EnumerateArray().Select(options => options.GetProperty("sourceUrl").GetString()));
        Assert.Equal("/uploads/selection.png", result.GetProperty("opened")[1].GetProperty("maskUrl").GetString());
        JsonElement first = result.GetProperty("sent")[0].GetProperty("body").GetProperty("messages")[0];
        JsonElement repeated = result.GetProperty("sent")[1].GetProperty("body").GetProperty("messages")[2];
        Assert.Equal(new[] { file, "reference.png" }, Strings(first, "stillImagePaths"));
        Assert.Equal("camera-full.png", first.GetProperty("attachments")[0].GetProperty("editFile").GetString());
        Assert.Equal("camera-preview.png", first.GetProperty("attachments")[0].GetProperty("previewFile").GetString());
        Assert.False(first.GetProperty("attachments")[0].TryGetProperty("editUrl", out _));
        Assert.Equal(first.ToString(), repeated.ToString());
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptTheory]
    [InlineData(null)]
    [InlineData("This image exceeds the editor's size limit.")]
    [InlineData("The full-resolution HEIC conversion failed.")]
    public void AnUnavailableHeicEditNeverFallsBackToTheThumbnailOrChangesTheTarget(string? reason)
    {
        JsonElement result = Run(MaskImageModel, "var reason = " + JsonSerializer.Serialize(reason) + ";" + PhotoDraftHelpers + """
            var opened = 0;
            window.TensorSharpMaskEditor = { open: function () { opened++; return Promise.resolve(null); } };
            window.TensorAgent.addAttachment({ ok: true, file: 'source.png', mediaType: 'image', maskPath: 'source-mask.png' });
            window.TensorAgent.addAttachment({ ok: true, file: 'camera.heic', mediaType: 'image',
              previewUrl: '/uploads/small-preview.png', editUnavailableReason: reason });
            __page.byId['text'].value = 'keep this draft';
            choosePhoto('camera.heic');
            return settle(20).then(function () {
              var order = draftOrder(), target = draftTarget(), draft = __page.byId['text'].value;
              __page.byId['send'].dispatch('click');
              return settle(20).then(function () { return {
                opened: opened, order: order, target: target, draft: draft,
                uploads: __page.requests('/api/upload').length, sent: __page.requests('/api/chat'), errors: __page.errorNotices()
              }; });
            });
            """);
        Assert.False(result.TryGetProperty("error", out _), result.ToString());
        Assert.Equal(0, result.GetProperty("opened").GetInt32());
        Assert.Equal(0, result.GetProperty("uploads").GetInt32());
        Assert.Equal(new[] { "source.png", "camera.heic" }, Strings(result, "order"));
        Assert.Equal("source.png", result.GetProperty("target").GetString());
        Assert.Equal("keep this draft", result.GetProperty("draft").GetString());
        string error = Assert.Single(Strings(result, "errors"));
        if (reason is null) Assert.Contains("Reattach this HEIC or HEIF photo", error);
        else Assert.Equal(reason, error);
        JsonElement message = Assert.Single(result.GetProperty("sent").EnumerateArray()).GetProperty("body").GetProperty("messages")[0];
        AssertActiveSelection(message, "source-mask.png");
        if (reason is not null) Assert.Equal(reason, message.GetProperty("attachments")[1].GetProperty("editUnavailableReason").GetString());
    }
}
