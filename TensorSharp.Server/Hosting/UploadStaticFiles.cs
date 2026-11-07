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
using System.Linq;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.FileProviders;

namespace TensorSharp.Server.Hosting;

/// <summary>
/// The ASP.NET Core half of <see cref="UploadContentPolicy"/>: mounts the upload
/// directory at <c>/uploads</c> with the policy's content-type table. Kept apart
/// from the table itself because the table is shared with hosts that have no
/// static-file middleware at all (TensorSharp.Chat carries it), while
/// <see cref="StaticFileOptions"/> only exists here.
/// </summary>
public static class UploadStaticFiles
{
    /// <summary>Where the upload directory is served.</summary>
    public const string MountPath = "/uploads";

    /// <summary>
    /// What leaves the process in place of an upload's path: span names, <c>url.path</c>
    /// and the request logger's exported scope. The unguessable file name is the only
    /// thing guarding an unauthenticated <c>/uploads</c> URL, so it is user content.
    /// </summary>
    public const string RedactedFilePath = MountPath + "/{file}";

    /// <summary>
    /// True for a request for a file under <c>/uploads/</c> (not <c>/uploads</c> itself),
    /// matched the way the static-file middleware matches it: by segment, ignoring case.
    /// </summary>
    public static bool IsUploadFilePath(PathString path) =>
        path.StartsWithSegments(MountPath, out PathString remaining) && remaining.Value is { Length: > 1 };

    internal static IContentTypeProvider BuildServeContentTypes() =>
        new FileExtensionContentTypeProvider(UploadContentPolicy.ServeContentTypes.ToDictionary(
            kv => kv.Key, kv => kv.Value, StringComparer.OrdinalIgnoreCase));

    public static StaticFileOptions BuildStaticFileOptions(string uploadDirectory) => new()
    {
        FileProvider = new PhysicalFileProvider(uploadDirectory),
        RequestPath = MountPath,
        ContentTypeProvider = BuildServeContentTypes(),
        OnPrepareResponse = ctx =>
            ctx.Context.Response.Headers["X-Content-Type-Options"] = "nosniff",
    };
}
