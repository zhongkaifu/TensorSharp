// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Chat
{
    /// <summary>
    /// The Agent Skills management surface, host-neutral: list what is installed, read
    /// one skill's instructions or a bundled file, install a bundle, remove one.
    ///
    /// <para>
    /// Two shapes are served from one service because they answer the same questions
    /// for different audiences — the <c>/v1/skills</c> list for API clients that want
    /// to know what they may name in a <c>skills</c> field, the <c>/api/skills</c>
    /// shape for the Web UI, which additionally needs the load errors and the
    /// install/remove operations. The error envelopes differ by audience for the same
    /// reason the Server's exception middleware distinguishes them: <c>/v1</c> gets
    /// <c>{error:{message,type}}</c>, <c>/api</c> gets a flat <c>{error:"..."}</c>.
    /// Refusals are thrown as <see cref="WebUiRequestRejectedException"/> carrying that
    /// envelope and the status, so every host answers identically.
    /// </para>
    /// </summary>
    public sealed class SkillsService
    {
        private readonly SkillRegistry _skills;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly ILoggerFactory _loggerFactory;

        public SkillsService(
            SkillRegistry skills,
            ServerHostingOptions options,
            UploadStoragePolicy uploads,
            ILoggerFactory loggerFactory)
        {
            _skills = skills ?? throw new ArgumentNullException(nameof(skills));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _uploads = uploads ?? throw new ArgumentNullException(nameof(uploads));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        }

        /// <summary>Whether this host accepts skill uploads at all (an install directory is configured).</summary>
        public bool CanInstall => _skills.CanInstall;

        // ---- /v1 ---------------------------------------------------------------

        /// <summary>
        /// <c>GET /v1/skills</c> — the OpenAI-shaped list, so a client can discover what
        /// it may put in a request's <c>skills</c> array.
        /// </summary>
        public object ListV1() => new
        {
            @object = "list",
            data = _skills.Skills.Select(s => Describe(s)).ToArray(),
        };

        /// <summary><c>GET /v1/skills/{name}</c> — one skill, including its instructions; 404 in the <c>/v1</c> envelope.</summary>
        public object GetV1(string name)
        {
            if (!_skills.TryGet(name, out Skill skill))
            {
                throw new WebUiRequestRejectedException(404,
                    new { error = new { message = $"No skill called '{name}' is installed.", type = "invalid_request_error" } });
            }
            return Describe(skill, includeInstructions: true);
        }

        // ---- /api --------------------------------------------------------------

        /// <summary>
        /// <c>GET /api/skills</c> — everything the Web UI needs in one round trip:
        /// the roster, whether uploads are possible, and the directories that failed to
        /// load. The errors matter: without them a skill with a broken <c>SKILL.md</c>
        /// is simply absent from the list, which is the hardest kind of problem for an
        /// author to diagnose.
        /// </summary>
        public object ListForUi() => new
        {
            enabled = _options.SkillsEnabled,
            installable = _skills.CanInstall,
            allowScripts = _options.SkillsAllowScripts,
            discovery = _options.SkillsDiscovery,
            roots = _skills.Roots,
            skills = _skills.Skills.Select(s => Describe(s)).ToArray(),
            errors = _skills.Errors.Select(e => new { path = e.Path, message = e.Message }).ToArray(),
        };

        /// <summary><c>GET /api/skills/{name}</c> — one skill with its instructions; 404 <c>{error}</c> when unknown.</summary>
        public object GetForUi(string name)
        {
            if (!_skills.TryGet(name, out Skill skill))
                throw new WebUiRequestRejectedException(404, new { error = $"No skill called '{name}' is installed." });
            return Describe(skill, includeInstructions: true);
        }

        /// <summary>
        /// <c>GET /api/skills/{name}/files/{*path}</c> — one bundled file as text, so the
        /// Web UI can show what a skill actually ships.
        ///
        /// <para>
        /// Read through the same <see cref="SkillPathGuard"/> the model's own reads go
        /// through. The transport must serve it as <c>text/plain</c> with <c>nosniff</c>:
        /// a skill may ship an <c>.html</c> or <c>.js</c> file, and serving it with its
        /// real type would execute uploaded content in the host's own origin.
        /// </para>
        /// </summary>
        /// <returns>False with a 404 <c>{error}</c> payload in <paramref name="error"/> for an unknown skill or an unreadable path.</returns>
        public bool TryGetFile(string name, string path, out string text, out object error)
        {
            text = null;
            error = null;

            if (!_skills.TryGet(name, out Skill skill))
            {
                error = new { error = $"No skill called '{name}' is installed." };
                return false;
            }

            if (!skill.TryReadResource(path, MaxUiReadBytes, 0, out SkillResourceContent content, out string readError))
            {
                error = new { error = readError };
                return false;
            }

            text = content.Text;
            return true;
        }

        /// <summary>
        /// The 403 an install gets on a host with no install directory. Separate from
        /// <see cref="Install"/> so a transport can refuse BEFORE it reads a multipart
        /// body it would otherwise have to buffer in full first.
        /// </summary>
        public void EnsureInstallable()
        {
            if (!_skills.CanInstall)
                throw new WebUiRequestRejectedException(403, new { error = "This server does not accept skill uploads." });
        }

        /// <summary>
        /// <c>POST /api/skills</c> — install a skill from a ZIP.
        ///
        /// <para>
        /// The sequence mirrors <see cref="WebUiChatService.UploadAsync"/> exactly, and
        /// for the same reasons: the storage budget is RESERVED before a byte is written
        /// so a full disk fails in milliseconds rather than after a long upload, and every
        /// failure path releases the reservation. What it adds is what a ZIP needs and a
        /// plain upload does not — <see cref="SkillArchive"/> resolves every entry through
        /// the path guard (an entry name is attacker-controlled text, and
        /// <c>../../authorized_keys</c> is the classic form), enforces a decompressed-size
        /// budget the compressed length cannot bound, and caps the entry count.
        /// </para>
        /// </summary>
        /// <param name="zip">The archive bytes; null counts as "no file was uploaded".</param>
        /// <param name="fileName">The client's file name; only its extension is consulted.</param>
        /// <param name="length">The archive's byte length, reserved against the upload budget before extraction.</param>
        /// <param name="overwrite">Replace an installed skill of the same id.</param>
        /// <returns>The installed skill's description (the transport answers 201 with it).</returns>
        /// <summary>
        /// <c>POST /api/skills/from-url</c> — install from a link, or from a list of them.
        ///
        /// <para>
        /// A link is how a skill is actually shared: someone sends a URL, and a phone has
        /// nowhere convenient to put a downloaded file first. The body may name one
        /// archive, or a plain-text list of archives one per line -- a collection is the
        /// other way skills travel. Anything that is not a ZIP and does not read as a
        /// list of http(s) URLs is refused by NAME, because "that did not work" over a
        /// download the user cannot see is the least useful answer available.
        /// </para>
        /// <para>
        /// Everything downloaded goes through <see cref="Install"/>, so the archive is
        /// held to exactly the same limits as an uploaded one: the path guard on every
        /// entry, the decompressed-size budget and the entry-count cap. A URL is a
        /// different way to arrive, not a different level of trust.
        /// </para>
        /// </summary>
        public async Task<object> InstallFromUrlAsync(string url, bool overwrite, CancellationToken cancellationToken)
        {
            EnsureInstallable();

            if (!Uri.TryCreate(url, UriKind.Absolute, out Uri? parsed)
                || (parsed.Scheme != Uri.UriSchemeHttp && parsed.Scheme != Uri.UriSchemeHttps))
            {
                throw new WebUiRequestRejectedException(400,
                    new { error = "A skill link must be an http:// or https:// URL." });
            }

            byte[] payload = await DownloadAsync(parsed, cancellationToken);

            // A ZIP starts "PK\x03\x04". Anything else is treated as a list.
            bool isZip = payload.Length > 4 && payload[0] == 0x50 && payload[1] == 0x4B
                && payload[2] == 0x03 && payload[3] == 0x04;
            if (isZip)
            {
                using var single = new MemoryStream(payload, writable: false);
                return Install(single, NameFor(parsed), payload.Length, overwrite);
            }

            string[] links;
            try { links = ParseLinkList(System.Text.Encoding.UTF8.GetString(payload)); }
            catch (Exception) { links = Array.Empty<string>(); }

            if (links.Length == 0)
            {
                throw new WebUiRequestRejectedException(400, new
                {
                    error = "That link is neither a .zip skill nor a list of skill links "
                        + "(one http(s) URL per line).",
                });
            }

            var installed = new List<object>();
            var failed = new List<object>();
            foreach (string link in links)
            {
                try { installed.Add(await InstallFromUrlAsync(link, overwrite, cancellationToken)); }
                catch (WebUiRequestRejectedException ex) { failed.Add(new { url = link, error = ex.Payload }); }
                catch (Exception ex) { failed.Add(new { url = link, error = ex.Message }); }
            }

            // Partial success is reported as such rather than as one verdict: a list of
            // ten where two are broken should still install eight, and say which two.
            return new { installed, failed, count = installed.Count };
        }

        private static string NameFor(Uri url)
        {
            string last = url.Segments.Length > 0 ? url.Segments[^1].Trim('/') : string.Empty;
            return last.EndsWith(".zip", StringComparison.OrdinalIgnoreCase) ? last : "skill.zip";
        }

        private static string[] ParseLinkList(string text) => text
            .Split('\n', '\r')
            .Select(line => line.Trim())
            .Where(line => line.StartsWith("http://", StringComparison.OrdinalIgnoreCase)
                        || line.StartsWith("https://", StringComparison.OrdinalIgnoreCase))
            .Distinct(StringComparer.OrdinalIgnoreCase)
            .Take(50)
            .ToArray();

        /// <summary>
        /// Fetch, with a ceiling. The size is capped while READING rather than trusting
        /// Content-Length, which a server chooses and can lie about; the limit is the
        /// same archive budget an upload is held to.
        /// </summary>
        private static async Task<byte[]> DownloadAsync(Uri url, CancellationToken cancellationToken)
        {
            const int MaxBytes = 64 * 1024 * 1024;
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(2) };
            http.DefaultRequestHeaders.UserAgent.ParseAdd("TensorAgent");
            try
            {
                using HttpResponseMessage response = await http.GetAsync(
                    url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                if (!response.IsSuccessStatusCode)
                {
                    throw new WebUiRequestRejectedException(400,
                        new { error = $"{url} answered {(int)response.StatusCode}." });
                }

                await using Stream body = await response.Content.ReadAsStreamAsync(cancellationToken);
                using var buffer = new MemoryStream();
                byte[] chunk = new byte[81920];
                int read;
                while ((read = await body.ReadAsync(chunk, cancellationToken)) > 0)
                {
                    if (buffer.Length + read > MaxBytes)
                    {
                        throw new WebUiRequestRejectedException(413,
                            new { error = "That skill is larger than 64 MB." });
                    }
                    buffer.Write(chunk, 0, read);
                }
                return buffer.ToArray();
            }
            catch (WebUiRequestRejectedException) { throw; }
            catch (Exception ex)
            {
                throw new WebUiRequestRejectedException(400,
                    new { error = $"{url} could not be downloaded: {ex.Message}" });
            }
        }

        public object Install(Stream zip, string? fileName, long length, bool overwrite)
        {
            ILogger logger = _loggerFactory.CreateLogger("TensorSharp.Server.Skills");

            EnsureInstallable();

            if (zip == null || length <= 0)
                throw new WebUiRequestRejectedException(400, new { error = "No file was uploaded." });

            string extension = Path.GetExtension(fileName ?? string.Empty);
            if (!string.Equals(extension, ".zip", StringComparison.OrdinalIgnoreCase))
            {
                // The client's own file name, so it is content.
                logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.upload.rejected reason=extension name={FileName}", LogContent.Of(fileName));
                throw new WebUiRequestRejectedException(400,
                    new { error = "A skill must be uploaded as a .zip containing its SKILL.md." });
            }

            // Reserve against the SAME budget the upload directory uses, before writing.
            // A skill tree is not stored under the upload root — the upload quota scan
            // and the TTL sweep are both non-recursive, so a skills subtree there would
            // be invisible to the tally and would be deleted by --upload-ttl-hours — but
            // the bytes are still client-originated writes and belong in the same
            // accounting.
            if (!_uploads.TryReserveClientWrite(length, out string limitError, out int limitStatus))
            {
                logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.upload.rejected reason=quota bytes={Bytes} status={Status}", length, limitStatus);
                throw new WebUiRequestRejectedException(limitStatus, new { error = limitError });
            }

            try
            {
                Skill installed = _skills.InstallFromZip(zip, overwrite, new SkillArchiveLimits
                {
                    MaxTotalBytes = Math.Min(MaxInstalledSkillBytes, _options.UploadMaxFileBytes * 8),
                });

                // Release the whole reservation. It existed to bound the UPLOAD — the
                // same per-file cap and quota check every client write goes through —
                // but the extracted tree lives under the skills root, not the upload
                // root, and the upload policy's tally is seeded by a non-recursive scan
                // of that one directory. Leaving the bytes charged would drift the
                // quota upward permanently with nothing to reconcile it. The tree's own
                // ceiling is SkillRegistryOptions.MaxSkillBytes, enforced during
                // extraction.
                _uploads.Release(length);

                logger.LogInformation(LogEventIds.SkillInstalled,
                    "skills.upload.installed id={SkillId} files={FileCount} bytes={Bytes}",
                    installed.Id, installed.Files.Count, installed.TotalBytes);
                return Describe(installed);
            }
            catch (SkillInstallException ex)
            {
                _uploads.Release(length);
                logger.LogWarning(LogEventIds.SkillRejected, "skills.upload.rejected reason={Reason}", ex.Message);
                throw new WebUiRequestRejectedException(400,
                    new { error = "The skill could not be installed: " + ex.Message });
            }
            catch (InvalidOperationException ex)
            {
                _uploads.Release(length);
                throw new WebUiRequestRejectedException(403, new { error = ex.Message });
            }
            catch
            {
                _uploads.Release(length);
                throw;
            }
        }

        /// <summary>
        /// <c>DELETE /api/skills/{name}</c> — remove an installed skill.
        ///
        /// <para>
        /// Refused (403) for a skill discovered under an operator-configured root: that
        /// is the operator's own file tree, and a management API must not delete out of it.
        /// </para>
        /// </summary>
        public object Remove(string name)
        {
            ILogger logger = _loggerFactory.CreateLogger("TensorSharp.Server.Skills");
            try
            {
                if (!_skills.Remove(name))
                    throw new WebUiRequestRejectedException(404, new { error = $"No skill called '{name}' is installed." });

                logger.LogInformation(LogEventIds.SkillRemoved, "skills.removed id={SkillId}", name);
                return new { removed = true, name };
            }
            catch (InvalidOperationException ex)
            {
                throw new WebUiRequestRejectedException(403, new { error = ex.Message });
            }
        }

        /// <summary><c>POST /api/skills/rescan</c> — pick up changes made on disk without a restart; answers with the <see cref="ListForUi"/> shape.</summary>
        public object Rescan()
        {
            SkillScanResult result = _skills.Refresh();
            _loggerFactory.CreateLogger("TensorSharp.Server.Skills").LogInformation(
                LogEventIds.SkillsScanned,
                "skills.rescan loaded={SkillCount} errors={ErrorCount}", result.Skills.Count, result.Errors.Count);
            return ListForUi();
        }

        /// <summary>
        /// The capability block folded into <c>GET /api/models</c>, so the Web UI can
        /// hide the whole Skills control in one round trip rather than discovering
        /// after a failed fetch that the host has no skills.
        /// </summary>
        public object DescribeCapabilities() => new
        {
            enabled = _options.SkillsEnabled,
            installable = _skills.CanInstall,
            allowScripts = _options.SkillsAllowScripts,
            count = _skills.Skills.Count,
        };

        // ---- shaping -----------------------------------------------------------

        /// <summary>Largest file the Web UI's viewer will fetch in one go.</summary>
        private const int MaxUiReadBytes = 512 * 1024;

        /// <summary>Hard ceiling on one installed skill, independent of the upload cap.</summary>
        private const long MaxInstalledSkillBytes = 256L * 1024 * 1024;

        /// <summary>The wire shape of one skill, shared by every listing and both envelopes.</summary>
        public static object Describe(Skill skill, bool includeInstructions = false)
        {
            if (skill == null) throw new ArgumentNullException(nameof(skill));

            var shaped = new Dictionary<string, object>(StringComparer.Ordinal)
            {
                ["id"] = skill.Id,
                ["object"] = "skill",
                ["name"] = skill.Manifest.Name,
                ["description"] = skill.Description,
                ["license"] = skill.Manifest.License,
                ["compatibility"] = skill.Manifest.Compatibility,
                ["metadata"] = skill.Manifest.Metadata,
                ["allowed_tools"] = skill.Manifest.AllowedTools,
                ["bytes"] = skill.TotalBytes,
                ["origin"] = skill.Origin == SkillOrigin.Installed ? "installed" : "discovered",
                ["warnings"] = skill.Manifest.Warnings,
                ["modified"] = skill.ModifiedUtc,
                ["files"] = skill.BundledFiles.Select(f => new
                {
                    path = f.Path,
                    bytes = f.Bytes,
                    kind = f.Kind.ToString().ToLowerInvariant(),
                    text = f.IsText,
                }).ToArray(),
            };

            // The instructions are the skill's whole body and are the largest thing here,
            // so a listing never carries them — only an explicit GET of one skill does.
            if (includeInstructions)
                shaped["instructions"] = skill.Manifest.Body;

            return shaped;
        }
    }
}
