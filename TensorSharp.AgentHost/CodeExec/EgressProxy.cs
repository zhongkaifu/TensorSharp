// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.AgentHost.CodeExec
{
    /// <summary>
    /// A loopback HTTP CONNECT proxy with a host allowlist — the only way out of a
    /// sandboxed install.
    ///
    /// <para>
    /// The install phase needs a package registry, and "needs the registry" used to
    /// be granted as "gets the whole internet". The shape used by Anthropic's
    /// sandbox-runtime closes that: the sandbox profile admits exactly ONE loopback
    /// TCP port, this proxy listens on it on the host side, and pip/npm are pointed
    /// at it via <c>HTTPS_PROXY</c>. The installer can then reach the hosts on the
    /// allowlist and nothing else — not because pip is trusted to behave, but
    /// because every other destination is unreachable at the OS level and this
    /// proxy refuses every host not on the list.
    /// </para>
    /// <para>
    /// CONNECT-only on purpose: pip and npm speak HTTPS exclusively, a CONNECT
    /// tunnel never sees the plaintext (no TLS termination, nothing to get wrong
    /// about certificates), and refusing plain-HTTP proxying costs nothing.
    /// </para>
    /// </summary>
    public sealed class EgressProxy : IDisposable
    {
        private readonly TcpListener _listener;
        private readonly IReadOnlyList<string> _allowedHosts;
        private readonly ILogger _logger;
        private readonly CancellationTokenSource _stop = new();
        private readonly ConcurrentQueue<string> _denied = new();
        private int _deniedCount;

        /// <param name="allowedHosts">
        /// Host patterns the proxy will tunnel to: an exact name (<c>pypi.org</c>) or
        /// a wildcard (<c>*.pythonhosted.org</c>). Ports are fixed at 443.
        /// </param>
        public EgressProxy(IReadOnlyList<string> allowedHosts, ILogger? logger = null)
        {
            _allowedHosts = allowedHosts ?? throw new ArgumentNullException(nameof(allowedHosts));
            _logger = logger ?? NullLogger.Instance;
            _listener = new TcpListener(IPAddress.Loopback, 0);
            _listener.Start();
            Port = ((IPEndPoint)_listener.LocalEndpoint).Port;
            _ = AcceptLoop();
        }

        /// <summary>The loopback port the sandbox profile admits.</summary>
        public int Port { get; }

        /// <summary>
        /// Hosts refused since the last call, oldest first — appended to a failed
        /// install's output so "connection refused" carries its reason.
        /// </summary>
        public IReadOnlyList<string> DrainDeniedHosts()
        {
            var drained = new List<string>();
            while (_denied.TryDequeue(out string? host))
                drained.Add(host);
            return drained;
        }

        /// <summary>True when <paramref name="host"/> matches the allowlist.</summary>
        internal static bool HostAllowed(string host, IReadOnlyList<string> allowedHosts)
        {
            if (string.IsNullOrWhiteSpace(host))
                return false;
            foreach (string pattern in allowedHosts)
            {
                if (string.IsNullOrWhiteSpace(pattern))
                    continue;
                if (pattern.StartsWith("*.", StringComparison.Ordinal))
                {
                    // *.example.com matches a.example.com and a.b.example.com but
                    // never example.com itself and never notexample.com.
                    if (host.EndsWith(pattern.Substring(1), StringComparison.OrdinalIgnoreCase))
                        return true;
                }
                else if (string.Equals(host, pattern, StringComparison.OrdinalIgnoreCase))
                {
                    return true;
                }
            }
            return false;
        }

        private async Task AcceptLoop()
        {
            try
            {
                while (!_stop.IsCancellationRequested)
                {
                    TcpClient client = await _listener.AcceptTcpClientAsync(_stop.Token).ConfigureAwait(false);
                    _ = Serve(client);
                }
            }
            catch (OperationCanceledException) { /* disposed */ }
            catch (ObjectDisposedException) { /* disposed */ }
        }

        private async Task Serve(TcpClient client)
        {
            using (client)
            {
                client.NoDelay = true;
                NetworkStream stream = client.GetStream();
                string? requestLine;
                try
                {
                    requestLine = await ReadHead(stream).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is IOException or SocketException or OperationCanceledException)
                {
                    return;
                }
                if (requestLine == null)
                    return;

                // "CONNECT host:port HTTP/1.1"
                string[] parts = requestLine.Split(' ', 3, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length < 2 || !string.Equals(parts[0], "CONNECT", StringComparison.OrdinalIgnoreCase))
                {
                    await Refuse(stream, "405 Method Not Allowed",
                        "only CONNECT tunneling is served here").ConfigureAwait(false);
                    return;
                }

                string target = parts[1];
                int colon = target.LastIndexOf(':');
                string host = colon > 0 ? target.Substring(0, colon) : target;
                int port = 443;
                if (colon > 0 && !int.TryParse(target.AsSpan(colon + 1), out port))
                    port = -1;

                if (port != 443 || !HostAllowed(host, _allowedHosts))
                {
                    RecordDenied(target);
                    await Refuse(stream, "403 Forbidden",
                        $"'{target}' is not on this install's registry allowlist").ConfigureAwait(false);
                    return;
                }

                try
                {
                    using var upstream = new TcpClient();
                    await upstream.ConnectAsync(host, port, _stop.Token).ConfigureAwait(false);
                    upstream.NoDelay = true;

                    byte[] ok = Encoding.ASCII.GetBytes("HTTP/1.1 200 Connection Established\r\n\r\n");
                    await stream.WriteAsync(ok, _stop.Token).ConfigureAwait(false);

                    NetworkStream up = upstream.GetStream();
                    Task a = stream.CopyToAsync(up, _stop.Token);
                    Task b = up.CopyToAsync(stream, _stop.Token);
                    await Task.WhenAny(a, b).ConfigureAwait(false);
                }
                catch (Exception ex) when (ex is IOException or SocketException or OperationCanceledException)
                {
                    // A dropped tunnel is the installer's problem to retry; nothing
                    // to report beyond the connection ending.
                }
            }
        }

        /// <summary>Read up to the blank line ending the request head; return its first line.</summary>
        private static async Task<string?> ReadHead(NetworkStream stream)
        {
            var buffer = new byte[8192];
            int filled = 0;
            while (filled < buffer.Length)
            {
                int n = await stream.ReadAsync(buffer.AsMemory(filled)).ConfigureAwait(false);
                if (n <= 0)
                    return null;
                filled += n;
                string head = Encoding.ASCII.GetString(buffer, 0, filled);
                int end = head.IndexOf("\r\n\r\n", StringComparison.Ordinal);
                if (end >= 0)
                {
                    int lineEnd = head.IndexOf("\r\n", StringComparison.Ordinal);
                    return lineEnd > 0 ? head.Substring(0, lineEnd) : null;
                }
            }
            return null;   // head too large — not a request an installer sends
        }

        private void RecordDenied(string target)
        {
            // Bounded: a runaway loop of refused connects must not grow a queue
            // without limit. The count still tells the story past the cap.
            if (Interlocked.Increment(ref _deniedCount) <= 32)
                _denied.Enqueue(target);
            _logger.LogWarning(LogEventIds.SkillScriptExecuted,
                "codeexec.egress.denied target={Target} (not on the install registry allowlist)", LogContent.Of(target));
        }

        private static async Task Refuse(NetworkStream stream, string status, string reason)
        {
            try
            {
                byte[] payload = Encoding.ASCII.GetBytes(
                    $"HTTP/1.1 {status}\r\nConnection: close\r\nContent-Length: {reason.Length}\r\n\r\n{reason}");
                await stream.WriteAsync(payload).ConfigureAwait(false);
            }
            catch (Exception ex) when (ex is IOException or SocketException) { /* peer gone */ }
        }

        public void Dispose()
        {
            _stop.Cancel();
            try { _listener.Stop(); }
            catch (SocketException) { /* already down */ }
            _stop.Dispose();
        }
    }
}
