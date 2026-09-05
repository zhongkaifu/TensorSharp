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
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.AgentHost.CodeExec;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Serves the files a <c>shell</c> command produced.
    ///
    /// <para>
    /// The model is handed these URLs in its tool result and repeats them to the user, so
    /// they are the whole answer to "the code made a PDF, now what". The route is the
    /// prefix <see cref="CodeExecOptions.ArtifactUriPrefix"/> is set to, which is why the
    /// two must agree.
    /// </para>
    /// <para>
    /// Everything served here was written by a program a model wrote, so it is served
    /// defensively: always as an attachment, never with a guessed content type that a
    /// browser might execute, and only from inside the store —
    /// <see cref="CodeArtifactStore.TryResolve"/> re-checks confinement rather than
    /// trusting the route to have done it.
    /// </para>
    /// </summary>
    public static class CodeArtifactEndpoints
    {
        /// <summary>The route artifacts are served under. Must match the runner's prefix.</summary>
        public const string RoutePrefix = "/api/code/artifacts";

        public static IEndpointRouteBuilder MapCodeArtifactEndpoints(this IEndpointRouteBuilder endpoints)
        {
            // List what one run left behind, for a UI that wants to show them.
            endpoints.MapGet(RoutePrefix + "/{runId}",
                (CodeArtifactStore? store, string runId) =>
                {
                    if (store == null)
                        return Results.NotFound(new { error = "code execution is not enabled on this server" });

                    IReadOnlyList<CodeArtifact> artifacts =
                        store.List(runId, (id, rel, _) => $"{RoutePrefix}/{id}/{rel}");

                    return artifacts.Count == 0
                        ? Results.NotFound(new { error = "no files are held for that run" })
                        : Results.Json(new
                        {
                            runId,
                            files = artifacts.Select(a => new { path = a.Path, bytes = a.Bytes, url = a.Pointer }),
                        });
                });

            // A catch-all so a nested path such as out/report.pdf binds whole.
            endpoints.MapGet(RoutePrefix + "/{runId}/{*path}",
                (HttpContext ctx, CodeArtifactStore? store, string runId, string path) =>
                {
                    if (store == null)
                        return Results.NotFound(new { error = "code execution is not enabled on this server" });

                    if (!store.TryResolve(runId, path, out string? full, out string? error))
                        return Results.NotFound(new { error = error ?? "not found" });

                    // This content came out of model-written code. Never let a browser
                    // sniff it into something executable, and never render it inline.
                    ctx.Response.Headers["X-Content-Type-Options"] = "nosniff";
                    ctx.Response.Headers["Content-Disposition"] =
                        "attachment; filename=\"" + SafeFileName(path) + "\"";

                    return Results.File(full!, ContentTypeFor(path), enableRangeProcessing: true);
                });

            return endpoints;
        }

        /// <summary>
        /// A filename safe to put in a header.
        ///
        /// <para>
        /// The program chose this name. A quote or a newline in it would let it write its
        /// own header fields, so only a conservative set of characters survives.
        /// </para>
        /// </summary>
        private static string SafeFileName(string path)
        {
            string name = Path.GetFileName(path);
            if (string.IsNullOrEmpty(name))
                return "download";

            var clean = new string(name
                .Where(c => char.IsAsciiLetterOrDigit(c) || c is '.' or '-' or '_')
                .ToArray());

            return clean.Length == 0 ? "download" : clean;
        }

        /// <summary>
        /// A content type for the common outputs, and a deliberately inert default.
        ///
        /// <para>
        /// The list is short on purpose: these are the types a model actually generates
        /// when asked for a document, and anything unrecognised is served as
        /// <c>application/octet-stream</c> so it downloads rather than renders. In
        /// particular nothing here maps to <c>text/html</c> — that is the one type that
        /// would turn "download the file the code produced" into stored XSS against the
        /// person who asked for it.
        /// </para>
        /// </summary>
        private static string ContentTypeFor(string path) =>
            Path.GetExtension(path).ToLowerInvariant() switch
            {
                ".pdf" => "application/pdf",
                ".csv" => "text/csv",
                ".txt" or ".md" or ".log" => "text/plain",
                ".json" => "application/json",
                ".xml" => "application/xml",
                ".png" => "image/png",
                ".jpg" or ".jpeg" => "image/jpeg",
                ".gif" => "image/gif",
                ".svg" => "image/svg+xml",
                ".webp" => "image/webp",
                ".zip" => "application/zip",
                ".xlsx" => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                ".xls" => "application/vnd.ms-excel",
                ".docx" => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ".doc" => "application/msword",
                ".pptx" => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ".wav" => "audio/wav",
                ".mp3" => "audio/mpeg",
                ".mp4" => "video/mp4",
                _ => "application/octet-stream",
            };
    }
}
