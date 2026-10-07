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

public sealed partial class WebUiPageTests
{
    [WebJavaScriptTheory]
    [InlineData(false, false, false)]
    [InlineData(false, false, true)]
    [InlineData(false, true, false)]
    [InlineData(false, true, true)]
    [InlineData(true, false, false)]
    [InlineData(true, false, true)]
    [InlineData(true, true, false)]
    [InlineData(true, true, true)]
    public void ImageResultsCanBeDownloadedInLiveAndReopenedChats(bool reopened, bool edited, bool native)
    {
        const string resultUrl = "/uploads/result%20image.png";
        string input = edited
            ? "{ role: 'user', content: 'make it night', stillImagePaths: ['original.png'], "
                + "attachments: [{ file: 'original.png', fileName: 'original.png', mediaType: 'image' }] }"
            : "{ role: 'user', content: 'a lighthouse at dusk' }";
        string routes = reopened
            ? $$"""
                R['/api/agent/conversations'] = { conversations: [
                  { id: 'saved', title: 'A picture', updatedAt: '2026-09-01T10:00:00Z', messageCount: 2 }
                ] };
                R['/api/sessions?conversation=saved'] = {
                  sessionId: 's9', conversationId: 'saved', think: false, skills: [],
                  messages: [{{input}}, { role: 'assistant', content: 'Here it is.', imageUrl: '{{resultUrl}}' }]
                };
                """
            : $$"""
                R['/api/chat'] = { __sse: [
                  { image_step: 1, image_steps: 1, preview: 'data:image/png;base64,AAAA' },
                  { token: 'Here it is.' },
                  { imageUrl: '{{resultUrl}}', width: 1024, height: 768 },
                  { done: true, sessionId: 's1', truncated: false }
                ] };
                """;

        JsonElement result = Run(ImageModel + routes, $$"""
            var reopened = {{JsonSerializer.Serialize(reopened)}};
            var edited = {{JsonSerializer.Serialize(edited)}};
            var native = {{JsonSerializer.Serialize(native)}};
            if (!reopened) {
              if (edited) window.TensorAgent.addAttachment({ ok: true, file: 'original.png',
                fileName: 'original.png', mediaType: 'image', url: '/uploads/original.png' });
              __page.byId['text'].value = edited ? 'make it night' : 'a lighthouse at dusk';
              __page.byId['send'].dispatch('click');
            }
            return settle(30).then(function () {
              // The host can become ready after the result has already rendered.
              if (native) window.TensorAgent.nativeReady();
              var actions = __page.byId['chat'].querySelector('.image-edit-actions');
              var anchor = actions && actions.querySelector('.image-download');
              if (!anchor) return { found: false };
              var picture = actions.parentNode.querySelector('img');
              // Saving an edit must keep selecting the result even while the user
              // is comparing it with the original photograph.
              if (edited) actions.querySelectorAll('button')[0].dispatch('click');
              var event = { target: anchor, button: 0, defaultPrevented: false,
                preventDefault: function () { this.defaultPrevented = true; },
                stopPropagation: function () {} };
              anchor.dispatch('click', event);
              return {
                found: true,
                count: __page.byId['chat'].querySelectorAll('.image-download').length,
                tag: anchor.tagName,
                label: anchor.textContent,
                href: anchor.href || anchor.getAttribute('href'),
                filename: anchor.download || anchor.getAttribute('download'),
                displayed: picture.src,
                prevented: event.defaultPrevented,
                saves: __page.requests('/api/agent/events').filter(function (r) {
                  return r.body && r.body.type === 'save-image';
                }).map(function (r) { return { method: r.method, body: r.body }; })
              };
            });
            """);

        Assert.False(result.TryGetProperty("error", out JsonElement failure), failure.ToString());
        Assert.True(result.GetProperty("found").GetBoolean(), "the finished image has no download action");
        Assert.Equal(1, result.GetProperty("count").GetInt32());
        Assert.Equal("A", result.GetProperty("tag").GetString());
        Assert.Equal("Download", result.GetProperty("label").GetString());
        Assert.Equal(resultUrl, result.GetProperty("href").GetString());
        Assert.Equal("result image.png", result.GetProperty("filename").GetString());
        Assert.Equal(edited ? "/uploads/original.png" : resultUrl, result.GetProperty("displayed").GetString());
        Assert.Equal(native, result.GetProperty("prevented").GetBoolean());
        JsonElement[] saves = result.GetProperty("saves").EnumerateArray().ToArray();
        if (native)
        {
            JsonElement save = Assert.Single(saves);
            Assert.Equal("POST", save.GetProperty("method").GetString());
            Assert.Equal(resultUrl, save.GetProperty("body").GetProperty("url").GetString());
            Assert.Equal("result image.png", save.GetProperty("body").GetProperty("name").GetString());
        }
        else
        {
            Assert.Empty(saves);
        }
    }
}
