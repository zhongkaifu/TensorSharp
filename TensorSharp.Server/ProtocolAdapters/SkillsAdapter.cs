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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// The Agent Skills management surface: list what is installed, read one skill's
    /// instructions, install a bundle, remove one.
    ///
    /// <para>
    /// Two prefixes are served from one adapter because they answer the same questions
    /// for different audiences — <c>/v1/skills</c> for API clients that want to know
    /// what they may name in a <c>skills</c> field, <c>/api/skills</c> for the Web UI,
    /// which additionally needs the load errors and the install/remove operations. The
    /// error envelopes differ by prefix, matching the rule
    /// <see cref="Logging.ApiExceptionMiddleware"/> already applies to everything else:
    /// <c>/v1</c> gets <c>{error:{message,type}}</c>, <c>/api</c> gets a flat
    /// <c>{error:"..."}</c>.
    /// </para>
    /// </summary>
    public sealed class SkillsAdapter
    {
        private readonly SkillRegistry _skills;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly ILoggerFactory _loggerFactory;

        public SkillsAdapter(
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

        // ---- /v1 ---------------------------------------------------------------

        /// <summary>
        /// <c>GET /v1/skills</c> — the OpenAI-shaped list, so a client can discover what
        /// it may put in a request's <c>skills</c> array.
        /// </summary>
        public IResult ListV1() => Results.Json(new
        {
            @object = "list",
            data = _skills.Skills.Select(s => Describe(s)).ToArray(),
        });

        /// <summary><c>GET /v1/skills/{name}</c> — one skill, including its instructions.</summary>
        public IResult GetV1(string name)
        {
            if (!_skills.TryGet(name, out Skill skill))
            {
                return Results.Json(
                    new { error = new { message = $"No skill called '{name}' is installed.", type = "invalid_request_error" } },
                    statusCode: StatusCodes.Status404NotFound);
            }
            return Results.Json(Describe(skill, includeInstructions: true));
        }

        // ---- /api --------------------------------------------------------------

        /// <summary>
        /// <c>GET /api/skills</c> — everything the Web UI needs in one round trip:
        /// the roster, whether uploads are possible, and the directories that failed to
        /// load. The errors matter: without them a skill with a broken <c>SKILL.md</c>
        /// is simply absent from the list, which is the hardest kind of problem for an
        /// author to diagnose.
        /// </summary>
        public IResult ListForUi() => Results.Json(new
        {
            enabled = _options.SkillsEnabled,
            installable = _skills.CanInstall,
            allowScripts = _options.SkillsAllowScripts,
            discovery = _options.SkillsDiscovery,
            roots = _skills.Roots,
            skills = _skills.Skills.Select(s => Describe(s)).ToArray(),
            errors = _skills.Errors.Select(e => new { path = e.Path, message = e.Message }).ToArray(),
        });

        /// <summary><c>GET /api/skills/{name}</c> — one skill with its instructions.</summary>
        public IResult GetForUi(string name)
        {
            if (!_skills.TryGet(name, out Skill skill))
                return Results.Json(new { error = $"No skill called '{name}' is installed." }, statusCode: StatusCodes.Status404NotFound);
            return Results.Json(Describe(skill, includeInstructions: true));
        }

        /// <summary>
        /// <c>GET /api/skills/{name}/files/{*path}</c> — one bundled file as plain text,
        /// so the Web UI can show what a skill actually ships.
        ///
        /// <para>
        /// Served through the same <see cref="SkillPathGuard"/> the model's own reads go
        /// through, and always as <c>text/plain</c> with <c>nosniff</c>: a skill may ship
        /// an <c>.html</c> or <c>.js</c> file, and serving it with its real type would
        /// execute uploaded content in the server's own origin.
        /// </para>
        /// </summary>
        public IResult GetFile(string name, string path)
        {
            if (!_skills.TryGet(name, out Skill skill))
                return Results.Json(new { error = $"No skill called '{name}' is installed." }, statusCode: StatusCodes.Status404NotFound);

            if (!skill.TryReadResource(path, MaxUiReadBytes, 0, out SkillResourceContent content, out string error))
                return Results.Json(new { error }, statusCode: StatusCodes.Status404NotFound);

            return Results.Text(content.Text, "text/plain; charset=utf-8");
        }

        /// <summary>
        /// <c>POST /api/skills</c> — install a skill from a ZIP.
        ///
        /// <para>
        /// The sequence mirrors <see cref="WebUiAdapter.UploadAsync"/> exactly, and for
        /// the same reasons: the form is checked before anything is read, the storage
        /// budget is RESERVED before a byte is written so a full disk fails in
        /// milliseconds rather than after a long upload, and every failure path releases
        /// the reservation. What it adds is what a ZIP needs and a plain upload does not
        /// — <see cref="SkillArchive"/> resolves every entry through the path guard
        /// (an entry name is attacker-controlled text, and <c>../../authorized_keys</c>
        /// is the classic form), enforces a decompressed-size budget the compressed
        /// length cannot bound, and caps the entry count.
        /// </para>
        /// </summary>
        public async Task<IResult> InstallAsync(HttpRequest request)
        {
            ILogger logger = _loggerFactory.CreateLogger("TensorSharp.Server.Skills");

            if (!_skills.CanInstall)
                return Results.Json(new { error = "This server does not accept skill uploads." }, statusCode: StatusCodes.Status403Forbidden);

            if (!request.HasFormContentType)
                return Results.Json(new { error = "Expected a multipart/form-data upload." }, statusCode: StatusCodes.Status400BadRequest);

            IFormCollection form = await request.ReadFormAsync();
            IFormFile file = form.Files.FirstOrDefault();
            if (file == null || file.Length == 0)
                return Results.Json(new { error = "No file was uploaded." }, statusCode: StatusCodes.Status400BadRequest);

            string extension = Path.GetExtension(file.FileName ?? string.Empty);
            if (!string.Equals(extension, ".zip", StringComparison.OrdinalIgnoreCase))
            {
                logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.upload.rejected reason=extension name={FileName}", file.FileName);
                return Results.Json(
                    new { error = "A skill must be uploaded as a .zip containing its SKILL.md." },
                    statusCode: StatusCodes.Status400BadRequest);
            }

            // Reserve against the SAME budget the upload directory uses, before writing.
            // A skill tree is not stored under the upload root — the upload quota scan
            // and the TTL sweep are both non-recursive, so a skills subtree there would
            // be invisible to the tally and would be deleted by --upload-ttl-hours — but
            // the bytes are still client-originated writes and belong in the same
            // accounting.
            if (!_uploads.TryReserveClientWrite(file.Length, out string limitError, out int limitStatus))
            {
                logger.LogWarning(LogEventIds.SkillRejected,
                    "skills.upload.rejected reason=quota bytes={Bytes} status={Status}", file.Length, limitStatus);
                return Results.Json(new { error = limitError }, statusCode: limitStatus);
            }

            bool overwrite = ReadBool(form, "overwrite");
            try
            {
                await using Stream stream = file.OpenReadStream();
                Skill installed = _skills.InstallFromZip(stream, overwrite, new SkillArchiveLimits
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
                _uploads.Release(file.Length);

                logger.LogInformation(LogEventIds.SkillInstalled,
                    "skills.upload.installed id={SkillId} files={FileCount} bytes={Bytes}",
                    installed.Id, installed.Files.Count, installed.TotalBytes);
                return Results.Json(Describe(installed), statusCode: StatusCodes.Status201Created);
            }
            catch (SkillInstallException ex)
            {
                _uploads.Release(file.Length);
                logger.LogWarning(LogEventIds.SkillRejected, "skills.upload.rejected reason={Reason}", ex.Message);
                return Results.Json(new { error = "The skill could not be installed: " + ex.Message },
                    statusCode: StatusCodes.Status400BadRequest);
            }
            catch (InvalidOperationException ex)
            {
                _uploads.Release(file.Length);
                return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status403Forbidden);
            }
            catch
            {
                _uploads.Release(file.Length);
                throw;
            }
        }

        /// <summary>
        /// <c>DELETE /api/skills/{name}</c> — remove an installed skill.
        ///
        /// <para>
        /// Refused for a skill discovered under an operator-configured root: that is the
        /// operator's own file tree, and a management API must not delete out of it.
        /// </para>
        /// </summary>
        public IResult Remove(string name)
        {
            ILogger logger = _loggerFactory.CreateLogger("TensorSharp.Server.Skills");
            try
            {
                if (!_skills.Remove(name))
                    return Results.Json(new { error = $"No skill called '{name}' is installed." }, statusCode: StatusCodes.Status404NotFound);

                logger.LogInformation(LogEventIds.SkillRemoved, "skills.removed id={SkillId}", name);
                return Results.Json(new { removed = true, name });
            }
            catch (InvalidOperationException ex)
            {
                return Results.Json(new { error = ex.Message }, statusCode: StatusCodes.Status403Forbidden);
            }
        }

        /// <summary><c>POST /api/skills/rescan</c> — pick up changes made on disk without a restart.</summary>
        public IResult Rescan()
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
        /// after a failed fetch that the server has no skills.
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

        private static object Describe(Skill skill, bool includeInstructions = false)
        {
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

        private static bool ReadBool(IFormCollection form, string key) =>
            form.TryGetValue(key, out Microsoft.Extensions.Primitives.StringValues values)
            && values.Count > 0
            && (string.Equals(values[0], "true", StringComparison.OrdinalIgnoreCase)
                || string.Equals(values[0], "1", StringComparison.Ordinal));
    }
}
