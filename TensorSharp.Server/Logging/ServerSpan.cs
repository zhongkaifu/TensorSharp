// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Http.Features;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Server.Logging;

/// <summary>
/// The ASP.NET Core server span of an HTTP request (App Insights "requests"), for the
/// facts only the protocol layer knows: the request id, the protocol, the OpenAI
/// response id.
/// </summary>
internal static class ServerSpan
{
    /// <summary>
    /// The request's server span when an exporter records it; otherwise null.
    ///
    /// <para>
    /// Read through <see cref="IHttpActivityFeature"/> rather than
    /// <see cref="Activity.Current"/>, which a child span or an async iterator resumed
    /// under another context could have replaced by the time a handler asks. A context
    /// that has no hosting feature (a test's <c>DefaultHttpContext</c>, an embedder) falls
    /// back to <see cref="Activity.Current"/>.
    /// </para>
    /// <para>
    /// Only a <see cref="ActivityKind.Server"/> activity counts: that is what the
    /// <c>Microsoft.AspNetCore</c> source creates for a listener. With telemetry off,
    /// ASP.NET Core still creates a plain activity for log correlation, which reports
    /// <see cref="Activity.IsAllDataRequested"/> as true; tagging it would be work
    /// nobody reads.
    /// </para>
    /// </summary>
    internal static Activity? Recorded(HttpContext context)
    {
        Activity? activity = context.Features.Get<IHttpActivityFeature>()?.Activity ?? Activity.Current;
        return activity is { IsAllDataRequested: true, Kind: ActivityKind.Server } ? activity : null;
    }

    /// <summary>
    /// <c>gen_ai.response.id</c> on the server span: the protocol's own id
    /// (<c>chatcmpl-…</c>, <c>resp_…</c>), so a client that quotes it finds the trace.
    /// The generation span below does not get it: the chat pipeline that creates that
    /// span never sees the id.
    /// </summary>
    internal static void SetResponseId(HttpContext context, string? responseId)
    {
        if (!string.IsNullOrEmpty(responseId))
            Recorded(context)?.SetTag(TelemetryAttributes.GenAiResponseId, responseId);
    }
}
