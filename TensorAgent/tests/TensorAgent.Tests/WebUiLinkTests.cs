using System.Text.Json;

namespace TensorAgent.Tests;

public sealed partial class WebUiPageTests
{
    [WebJavaScriptTheory]
    [InlineData("https://example.com/article.html?q=one&lang=en#details", "https://example.com/article.html?q=one&lang=en#details", "")]
    [InlineData("http://example.com/article", "http://example.com/article", "_blank")]
    [InlineData("/help.html", "http://127.0.0.1/help.html", "")]
    [InlineData("//example.com/article", "http://example.com/article", "_blank")]
    public void AWebLinkOpensInTheNativePopupAndKeepsTheConversation(string href, string resolvedUrl, string target)
    {
        JsonElement result = Run("""
            R['/api/sessions?conversation=new'] = { sessionId: 's1', conversationId: 'c1', messages: [], think: false, skills: [] };
            R['/api/chat'] = { __sse: [{ token: 'Here is the article.' }, { done: true, truncated: false }] };
            """, $$"""
            window.TensorAgent.nativeReady();
            __page.byId['text'].value = 'Find an article';
            __page.byId['send'].dispatch('click');
            return settle(30).then(function () {
              var before = JSON.stringify(__page.transcript());
              var sessions = __page.calls.filter(function (r) { return r.path === '/api/sessions'; }).length;
              __page.byId['text'].value = 'Keep my unsent draft';
              var link = document.createElement('a');
              link.setAttribute('href', {{JsonSerializer.Serialize(href)}});
              // A real browser resolves HTMLAnchorElement.href against the page URL.
              link.href = {{JsonSerializer.Serialize(resolvedUrl)}};
              link.setAttribute('target', {{JsonSerializer.Serialize(target)}});
              var label = document.createElement('strong'); label.textContent = 'Read article';
              link.appendChild(label);
              var event = { target: label, button: 0, defaultPrevented: false,
                preventDefault: function () { this.defaultPrevented = true; } };
              document.dispatch('click', event);
              return { prevented: event.defaultPrevented,
                opened: __page.requests('/api/agent/events').filter(function (r) { return r.body.type === 'open-link'; }).map(function (r) { return r.body; }),
                draft: __page.byId['text'].value, location: window.location.href,
                sameTranscript: before === JSON.stringify(__page.transcript()),
                sameSession: sessions === __page.calls.filter(function (r) { return r.path === '/api/sessions'; }).length };
            });
            """);

        Assert.True(result.GetProperty("prevented").GetBoolean());
        JsonElement opened = Assert.Single(result.GetProperty("opened").EnumerateArray());
        Assert.Equal(resolvedUrl, opened.GetProperty("url").GetString());
        Assert.Equal("Read article", opened.GetProperty("name").GetString());
        Assert.Equal("Keep my unsent draft", result.GetProperty("draft").GetString());
        Assert.Equal("http://127.0.0.1/", result.GetProperty("location").GetString());
        Assert.True(result.GetProperty("sameTranscript").GetBoolean());
        Assert.True(result.GetProperty("sameSession").GetBoolean());
    }

    [WebJavaScriptFact]
    public void AWebLinkKeepsNormalBrowserHandlingOutsideTheApp()
    {
        JsonElement result = Run("", """
            var link = document.createElement('a'); link.href = 'https://example.com/article.html';
            link.setAttribute('target', '_blank');
            var event = { target: link, button: 0, defaultPrevented: false,
              preventDefault: function () { this.defaultPrevented = true; } };
            document.dispatch('click', event);
            return { prevented: event.defaultPrevented,
              opened: __page.requests('/api/agent/events').filter(function (r) { return r.body.type === 'open-link'; }).length };
            """);

        Assert.False(result.GetProperty("prevented").GetBoolean());
        Assert.Equal(0, result.GetProperty("opened").GetInt32());
    }

    [WebJavaScriptFact]
    public void DownloadsFragmentsAndApplicationLinksDoNotOpenTheWebPopup()
    {
        JsonElement result = Run("", """
            window.TensorAgent.nativeReady();
            function click(href, resolved, download, prevented, button) {
              var link = document.createElement('a'); link.setAttribute('href', href); link.href = resolved;
              if (download) link.setAttribute('download', 'report.html');
              var event = { target: link, button: button || 0, defaultPrevented: !!prevented,
                preventDefault: function () { this.defaultPrevented = true; } };
              document.dispatch('click', event);
              return event.defaultPrevented;
            }
            var prevented = [
              click('#details', 'http://127.0.0.1/#details'),
              click('mailto:reader@example.com', 'mailto:reader@example.com'),
              click('tel:123456', 'tel:123456'),
              click('javascript:void(0)', 'javascript:void(0)'),
              click('https://example.com/report.html', 'https://example.com/report.html', true),
              click('https://example.com/article', 'https://example.com/article', false, true),
              click('https://example.com/article', 'https://example.com/article', false, false, 1)
            ];
            return { prevented: prevented,
              opened: __page.requests('/api/agent/events').filter(function (r) { return r.body.type === 'open-link'; }).length };
            """);

        Assert.Equal(new[] { false, false, false, false, false, true, false },
            result.GetProperty("prevented").EnumerateArray().Select(x => x.GetBoolean()));
        Assert.Equal(0, result.GetProperty("opened").GetInt32());
    }

    [WebJavaScriptFact]
    public void AGeneratedHtmlFileStillUsesTheFilePresenter()
    {
        JsonElement result = Run("", """
            window.TensorAgent.nativeReady();
            var link = document.createElement('a');
            link.href = 'http://127.0.0.1/api/code/artifacts/run1/report.html';
            link.textContent = 'report.html';
            var event = { target: link, defaultPrevented: false,
              preventDefault: function () { this.defaultPrevented = true; } };
            document.dispatch('click', event);
            return { prevented: event.defaultPrevented,
              opened: __page.requests('/api/agent/events').filter(function (r) {
                return r.body.type === 'open-file' || r.body.type === 'open-link';
              }).map(function (r) { return r.body; }) };
            """);

        Assert.True(result.GetProperty("prevented").GetBoolean());
        JsonElement opened = Assert.Single(result.GetProperty("opened").EnumerateArray());
        Assert.Equal("open-file", opened.GetProperty("type").GetString());
        Assert.Equal("/api/code/artifacts/run1/report.html", opened.GetProperty("url").GetString());
    }
}
