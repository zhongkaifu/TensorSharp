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
using Microsoft.AspNetCore.Http;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;

namespace TensorSharp.Server.Logging;

/// <summary>
/// Request paths that are bearer capabilities: the host has no authentication, so for these
/// routes an unguessable segment of the path is the only thing that guards what it serves.
/// Anyone who reads the path can fetch the content, so the path is user content for
/// telemetry: request log lines mark it with <see cref="TensorSharp.Runtime.Logging.LogContent"/>
/// and the HTTP server span gets a placeholder (unless content capture is on).
/// <list type="bullet">
///   <item><c>/uploads/&lt;file&gt;</c>: uploads and generated media, named by a random id.</item>
///   <item><c>/api/code/artifacts/&lt;run&gt;[/&lt;file&gt;]</c>: files a code-execution run produced;
///         the run listing returns every file's download URL.</item>
///   <item><c>/v1/responses/&lt;id&gt;[/...]</c>: a stored Responses API response, returned in full
///         to any caller who has its id.</item>
/// </list>
/// </summary>
public static class CapabilityPaths
{
    private const string ResponsesPrefix = "/v1/responses";

    /// <summary>
    /// The path with its capability segments replaced (<c>/uploads/{file}</c>,
    /// <c>/api/code/artifacts/{run}/{path}</c>, <c>/v1/responses/{id}</c>), or null when the path
    /// is not a capability path.
    /// </summary>
    public static string? Mask(PathString path)
    {
        if (UploadStaticFiles.IsUploadFilePath(path))
            return UploadStaticFiles.RedactedFilePath;

        if (path.StartsWithSegments(CodeArtifactEndpoints.RoutePrefix, StringComparison.OrdinalIgnoreCase, out PathString artifactRest)
            && artifactRest.HasValue && artifactRest.Value!.Length > 1)
        {
            int next = artifactRest.Value.IndexOf('/', 1);
            return next < 0 || next == artifactRest.Value.Length - 1
                ? CodeArtifactEndpoints.RoutePrefix + "/{run}"
                : CodeArtifactEndpoints.RoutePrefix + "/{run}/{path}";
        }

        if (path.StartsWithSegments(ResponsesPrefix, StringComparison.OrdinalIgnoreCase, out PathString responseRest)
            && responseRest.HasValue && responseRest.Value!.Length > 1)
        {
            // Keep a fixed sub-route such as /input_items; only the id is the capability.
            int next = responseRest.Value.IndexOf('/', 1);
            return next < 0
                ? ResponsesPrefix + "/{id}"
                : ResponsesPrefix + "/{id}" + responseRest.Value[next..];
        }

        return null;
    }

    /// <summary>Whether <paramref name="path"/> is one of the capability paths.</summary>
    public static bool IsCapabilityPath(PathString path) => Mask(path) is not null;
}
