using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptFact]
    public void CompatibleLorasAreOneTapFromTheComposerAndSearchLocally()
    {
        JsonElement result = Run(ImageModel + TwoStyles, """
            var button = __page.byId['composer-loras'];
            var visible = button.style.display !== 'none';
            button.dispatch('click');
            return settle(20).then(function () {
              var reads = __page.requests('/api/agent/loras').length;
              __page.byId['lora-search'].value = 'film';
              __page.byId['lora-search'].dispatch('input');
              var rows = __page.byId['lora-list'].querySelectorAll('.lorarow');
              var result = { visible: visible, rows: rows.length,
                name: rows[0].getAttribute('data-lora'),
                context: __page.byId['lora-context'].textContent,
                label: rows[0].querySelector('INPUT').getAttribute('aria-label'),
                readCount: __page.requests('/api/agent/loras').length - reads };
              __page.byId['lora-close'].dispatch('click');
              result.closed = !__page.byId['lora-sheet'].classList.contains('on');
              return result;
            });
            """);
        Assert.True(result.GetProperty("visible").GetBoolean());
        Assert.Equal(1, result.GetProperty("rows").GetInt32());
        Assert.Equal("film", result.GetProperty("name").GetString());
        Assert.Equal("Compatible with Qwen-Image 2.1", result.GetProperty("context").GetString());
        Assert.Equal("Use Film Stills", result.GetProperty("label").GetString());
        Assert.Equal(0, result.GetProperty("readCount").GetInt32());
        Assert.True(result.GetProperty("closed").GetBoolean());
    }

    [WebJavaScriptTheory]
    [InlineData("{}", "{ supportsLoras: true, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'gemma.gguf', architecture: 'gemma3' }", "{ supportsLoras: true, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: false, loras: [{ id: 'film' }] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: true, loras: [] }")]
    [InlineData("{ loaded: 'qwen.gguf', architecture: 'qwen_image' }", "{ supportsLoras: true, loadedModelFile: 'different.gguf', loras: [{ id: 'film' }] }")]
    // MiniMax-H3 References: a video model the host offers no plug-ins.
    [InlineData("{ loaded: 'h3-ref.gguf', architecture: 'minimax-h3', video: { family: 'minimax-h3', supportsReferenceConditioning: true, maxReferenceImages: 9 } }",
        "{ supportsLoras: false, loras: [] }")]
    public void UnsupportedModelsHaveNoLoraEntry(string model, string loras)
    {
        JsonElement result = Run($$"""
            R['/api/models'] = {{model}};
            R['/api/agent/loras'] = {{loras}};
            """, """
            __page.byId['model'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['composer-loras'].dispatch('click');
              return {
                composer: __page.byId['composer-loras'].style.display,
                details: __page.byId['open-loras'].style.display,
                sheetOpen: __page.byId['lora-sheet'].classList.contains('on')
              };
            });
            """);
        Assert.Equal("none", result.GetProperty("composer").GetString());
        Assert.Equal("none", result.GetProperty("details").GetString());
        Assert.False(result.GetProperty("sheetOpen").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SwitchingModelsHidesLorasAndIgnoresThePreviousModelsLateResponse()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            hold('/api/agent/loras');
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              R['/api/models'] = { loaded: 'gemma.gguf', architecture: 'gemma3' };
              document.dispatch('visibilitychange');
              return wait(400);
            }).then(function () {
              heldFor('/api/agent/loras').forEach(function (h) { h.answer(loraRows([])); });
              return settle(20);
            }).then(function () {
              return { composer: __page.byId['composer-loras'].style.display,
                details: __page.byId['open-loras'].style.display,
                open: __page.byId['lora-sheet'].classList.contains('on') };
            });
            """);
        Assert.Equal("none", result.GetProperty("composer").GetString());
        Assert.Equal("none", result.GetProperty("details").GetString());
        Assert.False(result.GetProperty("open").GetBoolean());
    }

    [WebJavaScriptFact]
    public void SwitchingBetweenImageModelsWhileSavingRefreshesTheNewModelsLoras()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              hold('/api/agent/loras/choice');
              var box = loraRow('film').querySelector('INPUT');
              box.checked = true;
              box.dispatch('change');
              return settle(20);
            }).then(function () {
              R['/api/models'] = { loaded: 'turbo.gguf', architecture: 'qwen_image' };
              R['/api/agent/loras'] = Object.assign(loraRows([]), { loadedModelFile: 'turbo.gguf' });
              document.dispatch('visibilitychange');
              return wait(400);
            }).then(function () {
              heldFor('/api/agent/loras/choice')[0].answer(loraRows([{ id: 'film', strength: 0.7 }]));
              return settle(20);
            }).then(function () {
              return { composer: __page.byId['composer-loras'].style.display,
                hint: __page.byId['composer-loras-hint'].textContent };
            });
            """);
        Assert.NotEqual("none", result.GetProperty("composer").GetString());
        Assert.Equal("Choose a style or edit", result.GetProperty("hint").GetString());
    }

    /// <summary>The LoRA sheet of MiniMax-H3's keyframes checkpoint: its two Turbo speed plug-ins,
    /// one of the clip's own at a time, beside a Qwen-Image speed plug-in chosen for pictures.</summary>
    private const string TurboSheet = """
        function turboRows(chosen) {
          function on(id) { return chosen.some(function (c) { return c.id === id; }); }
          function turbo(id, name, steps) {
            return { id: id, name: name, baseModel: 'minimax-h3-fl2va-q4k', kind: 'Speed', steps: steps, purpose: 'Clips in ' + steps + ' steps.',
              license: 'Apache-2.0', totalBytes: 1383677808, state: 'Installed', installedBytes: 1383677808, defaultStrength: 1,
              strengthAdjustable: false, chosen: on(id), strength: 1, download: null };
          }
          return { loadedModel: 'minimax-h3-fl2va-q4k', loadedModelFile: 'minimax-h3-q4k.gguf', loadedModelName: 'MiniMax-H3',
            supportsLoras: true, minStrength: 0.1, maxStrength: 1.5, chosen: chosen,
            loras: [turbo('turbo-8', 'LightX2V Turbo 8-Step', 8), turbo('turbo-4', 'LightX2V Turbo 4-Step', 4)] };
        }
        R['/api/agent/loras'] = turboRows([{ id: 'viggle', strength: 1 }]);
        R['/api/agent/loras/choice'] = function (call) { return turboRows(call.body.loras); };
        """;

    /// <summary>
    /// With a video model loaded the LoRA entry speaks of clips and speed, not of styles and
    /// edits, and turning one Turbo on turns the other off while the picture model's speed
    /// plug-in stays chosen: two schedules conflict only on the same model.
    /// </summary>
    [WebJavaScriptFact]
    public void AVideoModelsLoraSheetOffersItsSpeedPlugInsInAClipsWords()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + TurboSheet, """
            function row(id) {
              return __page.byId['lora-list'].querySelectorAll('.lorarow').filter(function (r) { return r.getAttribute('data-lora') === id; })[0];
            }
            var out = {
              shown: __page.byId['composer-loras'].style.display !== 'none',
              composer: __page.byId['composer-loras-hint'].textContent,
              details: __page.byId['loras-hint'].textContent
            };
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              out.sheetHint = __page.byId['lora-hint'].textContent;
              out.context = __page.byId['lora-context'].textContent;
              var eight = row('turbo-8').querySelector('INPUT');
              eight.checked = true;
              eight.dispatch('change');
              return settle(20);
            }).then(function () {
              var four = row('turbo-4').querySelector('INPUT');
              four.checked = true;
              four.dispatch('change');
              return settle(20);
            }).then(function () {
              out.sent = __page.requests('/api/agent/loras/choice').map(function (c) { return c.body.loras.map(function (l) { return l.id; }).join('+'); });
              out.after = __page.byId['composer-loras-hint'].textContent;
              out.names = __page.byId['lora-list'].querySelectorAll('.nm').map(function (n) { return n.textContent; });
              return out;
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.True(result.GetProperty("shown").GetBoolean());
        Assert.Equal("Choose a speed plug-in", result.GetProperty("composer").GetString());
        Assert.Equal("speed", result.GetProperty("details").GetString());
        Assert.Equal("They apply from the next clip. One speed plug-in at a time.", result.GetProperty("sheetHint").GetString());
        Assert.Equal("Compatible with MiniMax-H3", result.GetProperty("context").GetString());
        Assert.Equal(new[] { "viggle+turbo-8", "viggle+turbo-4" }, Strings(result, "sent"));
        Assert.Equal("LightX2V Turbo 4-Step", result.GetProperty("after").GetString());
        Assert.Equal(new[] { "LightX2V Turbo 8-Step · 8 steps", "● LightX2V Turbo 4-Step · 4 steps" }, Strings(result, "names"));
    }

    /// <summary>
    /// With LoRA plug-ins on, the host names them on every step of a clip, and the strip says
    /// what the clip is filmed with; the other stages read as they always did.
    /// </summary>
    [WebJavaScriptFact]
    public void AClipsProgressNamesTheLoraPlugInsItFilmsWith()
    {
        JsonElement result = Run(VideoModel(KeyframesVideo) + """
            R['/api/chat'] = { __sse: [
              { video_step: 0, video_steps: 8, video_phase: 'text-encode', elapsed: 2.1, eta: -1, video_loras: ['LightX2V Turbo 8-Step'] },
              { video_step: 1, video_steps: 8, video_phase: 'denoise', elapsed: 30.0, eta: 210.5, video_loras: ['LightX2V Turbo 8-Step'] },
              { video_step: 2, video_steps: 8, video_phase: 'denoise', elapsed: 60.0, eta: -1, video_loras: ['LightX2V Turbo 8-Step'] },
              { video_step: 8, video_steps: 8, video_phase: 'vae-decode', elapsed: 250.0, eta: -1, video_loras: ['LightX2V Turbo 8-Step'] },
              { videoUrl: '/uploads/video-1.mp4', width: 640, height: 384, frames: 22, fps: 24, seed: 1, hasAudio: true },
              { done: true, sessionId: 's1', tokenCount: 0, elapsed: 260, tokPerSec: 0.0, truncated: false }
            ] };
            """, """
            var before = __page.progress().length;
            __page.byId['text'].value = 'a lighthouse at dusk';
            __page.byId['send'].dispatch('click');
            return settle(40).then(function () { return { shown: __page.progress().slice(before), errors: __page.errorNotices() }; });
            """);

        Assert.Equal(new[]
        {
            "Filming…",
            "Reading the description…",
            "Filming with LightX2V Turbo 8-Step… step 1 of 8 · about 4 min left",
            "Filming with LightX2V Turbo 8-Step… step 2 of 8",
            "Developing the frames…",
        }, Strings(result, "shown"));
        Assert.Empty(Strings(result, "errors"));
    }

    [WebJavaScriptFact]
    public void ClosingLorasBeforeADownloadResponseDoesNotRestartPolling()
    {
        JsonElement result = Run(ImageModel + TwoStyles + Held, """
            hold('/api/agent/loras');
            __page.byId['composer-loras'].dispatch('click');
            return settle(20).then(function () {
              __page.byId['lora-close'].dispatch('click');
              var d = loraRows([]);
              d.loras[0].download = { running: true, progress: { fraction: 0.25 } };
              heldFor('/api/agent/loras').forEach(function (h) { h.answer(d); });
              return settle(20);
            }).then(function () {
              var reads = __page.requests('/api/agent/loras').length;
              return wait(1200).then(function () {
                return { addedReads: __page.requests('/api/agent/loras').length - reads,
                  open: __page.byId['lora-sheet'].classList.contains('on') };
              });
            });
            """);
        Assert.Equal(0, result.GetProperty("addedReads").GetInt32());
        Assert.False(result.GetProperty("open").GetBoolean());
    }
}
