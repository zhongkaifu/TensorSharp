// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace InferenceWeb.Tests;

/// <summary>
/// The bundled Web UI's spoken replies, read from the asset the server ships (the page has no JavaScript test harness;
/// TensorAgent's page, which has one, runs its half): the toggle is offered only when <c>/api/models</c> says the model
/// speaks, the request carries <c>speech</c> only then, the audio frames play through an AudioContext primed inside the
/// Send click, and the finished reply leaves a player and a download in the bubble and its id and WAV in the history.
/// </summary>
public class WebUiSpokenReplySourceTests
{
    private static string ReadWebUi()
    {
        string path = Path.Combine(AppContext.BaseDirectory, "wwwroot", "index.html");
        Assert.True(File.Exists(path), $"The TensorSharp.Server Web UI was not copied to the test output: {path}");
        return File.ReadAllText(path);
    }

    private static string Between(string source, string startMarker, string endMarker)
    {
        int start = source.IndexOf(startMarker, StringComparison.Ordinal);
        Assert.True(start >= 0, $"Could not find Web UI marker: {startMarker}");
        int end = source.IndexOf(endMarker, start + startMarker.Length, StringComparison.Ordinal);
        Assert.True(end > start, $"Could not find Web UI marker after {startMarker}: {endMarker}");
        return source.Substring(start, end - start);
    }

    [Fact]
    public void TheToggleIsHiddenUntilTheModelCanSpeak()
    {
        string html = ReadWebUi();
        Assert.Contains("<label class=\"reasoning-toggle\" id=\"speech-toggle-label\" style=\"display:none\"", html, StringComparison.Ordinal);
        Assert.Contains("<input type=\"checkbox\" id=\"speech-toggle\" onchange=\"onSpeechToggle()\">", html, StringComparison.Ordinal);
        Assert.Contains("<select id=\"speech-voice\"", html, StringComparison.Ordinal);
        // The capability comes from /api/models and nothing else.
        Assert.Contains("currentSpeechCaps = (data && data.speechOutput && data.speechOutput.available === true) ? data.speechOutput : null;",
            html, StringComparison.Ordinal);
        string paint = Between(html, "function updateSpeechControl()", "function speechRequested()");
        Assert.Contains("label.style.display = 'none';", paint, StringComparison.Ordinal);
        Assert.Contains("toggle.checked = false;", paint, StringComparison.Ordinal);
    }

    [Fact]
    public void OnlyASpokenTurnAsksForSpeech()
    {
        string html = ReadWebUi();
        string request = Between(html, "async function requestAssistantResponse()", "function revertFrom(");
        Assert.Contains("if (spokenTurn) chatPayload.speech = speechRequestField();", request, StringComparison.Ordinal);
        Assert.Contains("const spokenTurn = speechRequested()", request, StringComparison.Ordinal);
        string field = Between(html, "function speechRequestField()", "let speechAudioContext");
        Assert.Contains("{ voice: voice.value } : true", field, StringComparison.Ordinal);
    }

    [Fact]
    public void AudioPlaysAsItArrivesThroughAContextPrimedInTheClick()
    {
        string html = ReadWebUi();
        // Primed in the click's synchronous part, before any await, and from the toggle.
        string send = Between(html, "async function sendMessage()", "const userContent = text || describeAttachments();");
        Assert.Contains("primeSpeechPlayback();", send, StringComparison.Ordinal);
        Assert.DoesNotContain("await", send, StringComparison.Ordinal);
        Assert.Contains("function onSpeechToggle() {\n  primeSpeechPlayback();", html, StringComparison.Ordinal);
        string prime = Between(html, "function primeSpeechPlayback()", "function onSpeechToggle()");
        Assert.Contains("window.AudioContext || window.webkitAudioContext", prime, StringComparison.Ordinal);
        Assert.Contains("if (speechAudioContext.state !== 'running') speechAudioContext.resume();", prime, StringComparison.Ordinal);

        // Each chunk is a buffer scheduled right after the previous one.
        string player = Between(html, "function createLiveSpeech(", "function spokenReplyFileName(");
        Assert.Contains("context.createBuffer(1, samples.length, sampleRate)", player, StringComparison.Ordinal);
        Assert.Contains("source.start(nextAt);", player, StringComparison.Ordinal);
        Assert.Contains("nextAt += buffer.duration;", player, StringComparison.Ordinal);
        Assert.Contains("stop()", player, StringComparison.Ordinal);

        string request = Between(html, "async function requestAssistantResponse()", "function revertFrom(");
        Assert.Contains("if (liveSpeech) liveSpeech.push(data.audio, data.sampleRate);", request, StringComparison.Ordinal);
        // A stop of the reply (or a steer, or a new chat) silences it.
        Assert.Contains("if (liveSpeech) liveSpeech.stop();", request, StringComparison.Ordinal);
        Assert.Contains("'\\u25A0 Stop speaking'", request, StringComparison.Ordinal);
        // A spoken turn's notice is shown in the bubble.
        Assert.Contains("appendTurnNotice(speechBubble, data.notice);", request, StringComparison.Ordinal);
    }

    [Fact]
    public void TheFinishedReplyKeepsAPlayerAndADownloadAndTheHistoryKeepsItsIdAndWav()
    {
        string html = ReadWebUi();
        string keep = Between(html, "function appendSpokenReply(", "function appendTurnNotice(");
        Assert.Contains("document.createElement('audio')", keep, StringComparison.Ordinal);
        Assert.Contains("audio.controls = true;", keep, StringComparison.Ordinal);
        Assert.Contains("download.setAttribute('download',", keep, StringComparison.Ordinal);
        Assert.Contains("download.href = message.audioUrl;", keep, StringComparison.Ordinal);

        string request = Between(html, "async function requestAssistantResponse()", "function revertFrom(");
        Assert.Contains("spokenReply = { audioId: data.audioId };", request, StringComparison.Ordinal);
        Assert.Contains("appendSpokenReply(speechBubble, spokenReply);", request, StringComparison.Ordinal);
        Assert.Contains("if (spokenReply) Object.assign(assistantMessage, spokenReply);", request, StringComparison.Ordinal);
        // The player sits outside .bubble-text, which the final render replaces.
        string bubble = Between(html, "function addAssistantBubble()", "function toggleThinking(");
        Assert.Contains("<div class=\"bubble-text\"></div>", bubble, StringComparison.Ordinal);
        Assert.Contains("bubble.insertBefore(box, stats || null);", keep, StringComparison.Ordinal);
    }
}
