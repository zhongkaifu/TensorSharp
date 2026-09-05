// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Routes for the Agent Skills management surface.
    ///
    /// <para>
    /// Mapped only when skills are enabled, so <c>--no-skills</c> means the endpoints
    /// are genuinely absent rather than answering with an empty list — a client can
    /// then tell "this server has no skills feature" from "this server has no skills
    /// installed", which is exactly what the Web UI's capability probe and the C#
    /// client's delivery-mode probe rely on.
    /// </para>
    /// </summary>
    public static class SkillEndpoints
    {
        public static IEndpointRouteBuilder MapSkillEndpoints(this IEndpointRouteBuilder endpoints)
        {
            // OpenAI-shaped discovery: what may be named in a request's `skills` array.
            endpoints.MapGet("/v1/skills",
                (SkillsAdapter adapter) => adapter.ListV1());
            endpoints.MapGet("/v1/skills/{name}",
                (SkillsAdapter adapter, string name) => adapter.GetV1(name));

            // Management surface for the Web UI.
            endpoints.MapGet("/api/skills",
                (SkillsAdapter adapter) => adapter.ListForUi());
            endpoints.MapGet("/api/skills/{name}",
                (SkillsAdapter adapter, string name) => adapter.GetForUi(name));
            // A catch-all so a nested path such as references/api.md binds whole.
            endpoints.MapGet("/api/skills/{name}/files/{*path}",
                (SkillsAdapter adapter, string name, string path) => adapter.GetFile(name, path));
            // Extraction walks the whole archive and can take a while for a large
            // bundle, so this route opts out of the request timeout the same way the
            // media-generation routes do.
            endpoints.MapPost("/api/skills",
                (HttpRequest req, SkillsAdapter adapter) => adapter.InstallAsync(req))
                .DisableRequestTimeout();
            endpoints.MapPost("/api/skills/rescan",
                (SkillsAdapter adapter) => adapter.Rescan());
            endpoints.MapDelete("/api/skills/{name}",
                (SkillsAdapter adapter, string name) => adapter.Remove(name));

            return endpoints;
        }
    }
}
