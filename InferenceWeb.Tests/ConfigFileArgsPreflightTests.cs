// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Net;
using System.Net.Sockets;
using System.Text.Json;

namespace InferenceWeb.Tests;

/// <summary>
/// <see cref="ConfigFileArgs.Expand(string[], Action{ConfigExpansionPreview})"/>: a host's
/// preflight sees the expansion before anything is downloaded, so a contradictory command line
/// (a locked sampling flag beside a LoRA plug-in, say) is refused without first fetching tens of
/// gigabytes. A local HTTP server counts every request, so "nothing was downloaded" is measured,
/// not assumed. Also <see cref="ConfigFileArgs.ResolveFileEntryPath"/>, the no-download resolver
/// plug-in configs use.
/// </summary>
public sealed class ConfigFileArgsPreflightTests : IDisposable
{
    private readonly string _dir;

    public ConfigFileArgsPreflightTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-config-preflight-tests-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { /* best effort */ }
    }

    private string WriteConfig(string json, string name = "config.json")
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, json);
        return path;
    }

    private string Cached(string name)
    {
        string path = Path.Combine(_dir, name);
        File.WriteAllText(path, "already here");
        return path;
    }

    private static string Q(string value) => JsonSerializer.Serialize(value);

    private static int Count(string text, string fragment)
    {
        int count = 0;
        for (int i = text.IndexOf(fragment, StringComparison.Ordinal); i >= 0; i = text.IndexOf(fragment, i + fragment.Length, StringComparison.Ordinal))
            count++;
        return count;
    }

    // ---- the preview and the preflight ---------------------------------------------------------

    [Fact]
    public void Preflight_SeesTheRealTokensAndPendingDownloads_BeforeAnyRequest()
    {
        using var server = new CountingServer();
        server.AddFile("/model.gguf", new byte[] { 1, 2, 3, 4 });
        string model = Path.Combine(_dir, "downloads", "model.gguf");
        string projector = Cached("mmproj.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": { "path": {{Q(model)}}, "urls": [ {{Q(server.UrlFor("/model.gguf"))}} ] },
          "mmproj": { "path": {{Q(projector)}}, "urls": [ {{Q(server.UrlFor("/never.gguf"))}} ] },
          "max-tokens": 64,
          "stop": ["</s>"]
        }
        """);
        var sink = new StringWriter();
        ConfigExpansionPreview seen = null;
        int requestsDuringPreflight = -1;
        string logDuringPreflight = null;
        bool fileDuringPreflight = true;

        string[] result = ConfigFileArgs.Expand(new[] { "--config", cfg, "--temperature", "0.5" }, sink, interactiveProgress: false,
            preview =>
            {
                seen = preview;
                requestsDuringPreflight = server.Requests;
                logDuringPreflight = sink.ToString();
                fileDuringPreflight = File.Exists(model);
            });

        Assert.NotNull(seen);
        Assert.Equal(result, seen.Tokens);
        Assert.Equal(new[]
        {
            "--model", Path.GetFullPath(model), "--mmproj", Path.GetFullPath(projector), "--max-tokens", "64",
            "--stop", "</s>", "--temperature", "0.5",
        }, result);
        // Nothing was fetched or logged while the preflight ran ...
        Assert.Equal(0, requestsDuringPreflight);
        Assert.DoesNotContain("[model-download]", logDuringPreflight, StringComparison.Ordinal);
        Assert.False(fileDuringPreflight);
        // ... it was told what the real pass would fetch ...
        Assert.Equal(new[] { Path.GetFullPath(model) }, seen.PendingDownloads);
        Assert.True(seen.IsPending(model));
        Assert.True(seen.IsPending(Path.Combine(_dir, "downloads", ".", "model.gguf")));
        Assert.False(seen.IsPending(projector));
        Assert.False(seen.IsPending(""));
        // ... and the real pass, which it let continue, downloaded it.
        Assert.Equal(1, server.RequestCount("/model.gguf"));
        Assert.Equal(0, server.RequestCount("/never.gguf"));
        Assert.True(File.Exists(model));
        Assert.Contains("[model-download] model:", sink.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Preflight_ThatThrows_StopsStartupWithZeroRequests()
    {
        using var server = new CountingServer();
        server.AddFile("/model.gguf", new byte[] { 1, 2, 3, 4 });
        string model = Path.Combine(_dir, "model.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{Q(model)}}, "urls": [ {{Q(server.UrlFor("/model.gguf"))}} ] }, "diffusion-steps": 20 }
        """);
        var sink = new StringWriter();

        var ex = Assert.Throws<ArgumentException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }, sink,
            interactiveProgress: false, _ => throw new ArgumentException("--diffusion-steps 20 contradicts the plug-in.")));

        Assert.Equal("--diffusion-steps 20 contradicts the plug-in.", ex.Message);
        Assert.Equal(0, server.Requests);
        Assert.False(File.Exists(model));
        Assert.DoesNotContain("[model-download]", sink.ToString(), StringComparison.Ordinal);
    }

    [Fact]
    public void Preview_ListsEveryShardOfADownloadGroupButEmitsOnlyTheMainPath()
    {
        using var server = new CountingServer();
        string first = Path.Combine(_dir, "split-00001-of-00002.gguf");
        string second = Path.Combine(_dir, "split-00002-of-00002.gguf");
        string cfg = WriteConfig($$"""
        {
          "model": {
            "path": {{Q(first)}}, "urls": [ {{Q(server.UrlFor("/1.gguf"))}} ],
            "files": [ { "path": {{Q(second)}}, "urls": [ {{Q(server.UrlFor("/2.gguf"))}} ] } ]
          }
        }
        """);
        ConfigExpansionPreview seen = null;

        Assert.Throws<InvalidOperationException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null,
            interactiveProgress: false, preview => { seen = preview; throw new InvalidOperationException("stop"); }));

        Assert.Equal(new[] { "--model", Path.GetFullPath(first) }, seen.Tokens);
        Assert.Equal(2, seen.PendingDownloads.Count);
        Assert.True(seen.IsPending(first));
        Assert.True(seen.IsPending(second));
        Assert.Equal(0, server.Requests);
    }

    [Fact]
    public void Preview_SkipsOverriddenEntries_AndTheirNoticeIsReportedOnce()
    {
        using var server = new CountingServer();
        server.AddFile("/vision.gguf", new byte[] { 9 });
        string vision = Path.Combine(_dir, "vision.gguf");
        string cfg = WriteConfig($$"""
        { "model": "text.gguf", "mmproj": { "path": {{Q(vision)}}, "urls": [ {{Q(server.UrlFor("/vision.gguf"))}} ] } }
        """);
        var sink = new StringWriter();
        ConfigExpansionPreview seen = null;

        string[] result = ConfigFileArgs.Expand(new[] { "--config", cfg, "--mmproj", "none" }, sink, interactiveProgress: false,
            preview => seen = preview);

        Assert.Equal(new[] { "--model", "text.gguf", "--mmproj", "none" }, result);
        Assert.Equal(result, seen.Tokens);
        Assert.Empty(seen.PendingDownloads);
        Assert.Equal(0, server.Requests);
        Assert.Equal(1, Count(sink.ToString(), "[config] --mmproj is set on the command line"));
    }

    [Fact]
    public void Preview_AMissingFileWithoutUrl_IsRefusedBeforeThePreflight()
    {
        string cfg = WriteConfig($$"""{ "model": { "path": {{Q(Path.Combine(_dir, "missing.gguf"))}} } }""");
        bool called = false;

        Assert.Throws<FileNotFoundException>(() => ConfigFileArgs.Expand(new[] { "--config", cfg }, TextWriter.Null,
            interactiveProgress: false, _ => called = true));

        Assert.False(called);
    }

    [Fact]
    public void Preflight_IsNotCalledWithoutAConfigFile()
    {
        bool called = false;
        string[] args = { "--model", "m.gguf", "--lora", "x.json" };

        Assert.Same(args, ConfigFileArgs.Expand(args, TextWriter.Null, interactiveProgress: false, _ => called = true));
        Assert.False(called);
    }

    [Fact]
    public void NullPreflight_LeavesTheExpansionAndItsLogUnchanged()
    {
        string model = Cached("cached.gguf");
        string cfg = WriteConfig($$"""
        { "model": { "path": {{Q(model)}}, "urls": [ "http://127.0.0.1:9/never" ] }, "max-tokens": 64 }
        """);
        string[] args = { "--config", cfg, "--max-tokens", "8" };
        var withoutHook = new StringWriter();
        var withNull = new StringWriter();

        string[] expected = ConfigFileArgs.Expand(args, withoutHook, interactiveProgress: false);
        string[] actual = ConfigFileArgs.Expand(args, withNull, interactiveProgress: false, preflight: null);

        Assert.Equal(expected, actual);
        Assert.Equal(withoutHook.ToString(), withNull.ToString());
        Assert.Contains("using cached file", withNull.ToString(), StringComparison.Ordinal);
        Assert.Equal(1, Count(withNull.ToString(), "[config] --max-tokens is set on the command line"));
    }

    // ---- ResolveFileEntryPath -----------------------------------------------------------------

    [Fact]
    public void ResolveFileEntryPath_SubstitutesVariablesAndNeverDownloads()
    {
        using var server = new CountingServer();
        server.AddFile("/w.safetensors", new byte[] { 1 });
        string doc = WriteConfig($$"""
        {
          // a plug-in config's conventions: variables, a path relative to the document, a download entry
          "variables": { "root": "sub" },
          "weights": { "path": "${root}/w.safetensors", "urls": [ {{Q(server.UrlFor("/w.safetensors"))}} ] },
          "plain": "${root}/plain.safetensors",
          "bad": 5,
        }
        """, "plugin.json");
        string expected = Path.GetFullPath(Path.Combine(_dir, "sub", "w.safetensors"));

        Assert.Equal(expected, ConfigFileArgs.ResolveFileEntryPath(doc, "weights"));
        Assert.Equal(Path.GetFullPath(Path.Combine(_dir, "sub", "plain.safetensors")), ConfigFileArgs.ResolveFileEntryPath(doc, "plain"));
        Assert.Null(ConfigFileArgs.ResolveFileEntryPath(doc, "absent"));
        Assert.Throws<ArgumentException>(() => ConfigFileArgs.ResolveFileEntryPath(doc, "bad"));
        Assert.Equal(0, server.Requests);
        Assert.False(File.Exists(expected));
    }

    /// <summary>A loopback HTTP server that serves registered payloads and counts every request,
    /// including those for paths it does not know.</summary>
    private sealed class CountingServer : IDisposable
    {
        private readonly HttpListener _listener = new();
        private readonly string _prefix;
        private readonly Dictionary<string, byte[]> _files = new();
        private readonly Dictionary<string, int> _hits = new();
        private readonly object _gate = new();
        private int _requests;

        public CountingServer()
        {
            var probe = new TcpListener(IPAddress.Loopback, 0);
            probe.Start();
            int port = ((IPEndPoint)probe.LocalEndpoint).Port;
            probe.Stop();
            _prefix = $"http://127.0.0.1:{port}/";
            _listener.Prefixes.Add(_prefix);
            _listener.Start();
            Task.Run(Loop);
        }

        public int Requests
        {
            get { lock (_gate) return _requests; }
        }

        public void AddFile(string path, byte[] content)
        {
            lock (_gate) _files[path] = content;
        }

        public string UrlFor(string path) => _prefix.TrimEnd('/') + path;

        public int RequestCount(string path)
        {
            lock (_gate) return _hits.TryGetValue(path, out int n) ? n : 0;
        }

        private void Loop()
        {
            while (_listener.IsListening)
            {
                HttpListenerContext ctx;
                try { ctx = _listener.GetContext(); }
                catch { return; }

                string path = ctx.Request.Url!.AbsolutePath;
                byte[] body;
                lock (_gate)
                {
                    _requests++;
                    _hits[path] = (_hits.TryGetValue(path, out int n) ? n : 0) + 1;
                    _files.TryGetValue(path, out body);
                }
                try
                {
                    if (body == null)
                    {
                        ctx.Response.StatusCode = 404;
                        ctx.Response.Close();
                        continue;
                    }
                    ctx.Response.StatusCode = 200;
                    ctx.Response.ContentLength64 = body.Length;
                    ctx.Response.OutputStream.Write(body, 0, body.Length);
                    ctx.Response.OutputStream.Close();
                }
                catch { /* the client went away */ }
            }
        }

        public void Dispose()
        {
            try { _listener.Stop(); } catch { }
            try { _listener.Close(); } catch { }
        }
    }
}
