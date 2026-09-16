using System.Net;
using TensorSharp.AgentHost.CodeExec;

namespace RemoteAgentBench;

/// <summary>Serve only guarded immutable artifacts on loopback.</summary>
internal sealed class ArtifactServer(CodeArtifactStore store, int port) : IAsyncDisposable
{
    private readonly HttpListener _listener = new();
    private readonly CancellationTokenSource _stop = new();
    private Task? _serve;
    public string Prefix => $"http://127.0.0.1:{port}/api/code/artifacts";

    public void Start()
    {
        _listener.Prefixes.Add(Prefix + "/");
        _listener.Start();
        _serve = Serve();
    }

    private async Task Serve()
    {
        while (!_stop.IsCancellationRequested)
        {
            HttpListenerContext context;
            try { context = await _listener.GetContextAsync().WaitAsync(_stop.Token); }
            catch (OperationCanceledException) { break; }
            catch (HttpListenerException) when (_stop.IsCancellationRequested) { break; }
            try
            {
                string route = context.Request.Url!.AbsolutePath;
                const string prefix = "/api/code/artifacts/";
                string[] pieces = route.StartsWith(prefix, StringComparison.Ordinal)
                    ? route[prefix.Length..].Split('/', 2) : [];
                if (context.Request.HttpMethod != "GET" || pieces.Length != 2 ||
                    !store.TryResolve(Uri.UnescapeDataString(pieces[0]), Uri.UnescapeDataString(pieces[1]), out var path, out _))
                {
                    context.Response.StatusCode = 404;
                    continue;
                }
                using var file = File.OpenRead(path!);
                context.Response.ContentType = "application/octet-stream";
                context.Response.ContentLength64 = file.Length;
                await file.CopyToAsync(context.Response.OutputStream, _stop.Token);
            }
            catch (IOException) { context.Response.Abort(); }
            finally { context.Response.Close(); }
        }
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        _listener.Stop();
        if (_serve != null) await _serve;
        _listener.Close();
        _stop.Dispose();
    }
}
