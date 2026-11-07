// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Collections.Generic;
using OpenTelemetry;
using OpenTelemetry.Logs;
using TensorSharp.Runtime.Telemetry;

namespace TensorSharp.Telemetry
{
    /// <summary>
    /// Adds <c>tensorsharp.request_id</c> and <c>tensorsharp.protocol</c> to every exported log
    /// record written during an HTTP request.
    ///
    /// <para>
    /// The request logger's scope already carries the id as <c>RequestId</c>, but ASP.NET Core's
    /// own hosting scope uses the same key for <c>HttpContext.TraceIdentifier</c> and comes first,
    /// so in the exported record (and in Application Insights, which keeps the first value of a
    /// key) <c>RequestId</c> is not the <c>X-TensorSharp-Request-Id</c> a client quotes. A scope
    /// cannot be rewritten by a processor; this adds the TensorSharp id under its own name
    /// instead. It reads <see cref="TelemetryContext"/>, which is set for the request's async
    /// flow, and log processors run on the thread that wrote the record.
    /// </para>
    /// </summary>
    public sealed class LogContextEnrichmentProcessor : BaseProcessor<LogRecord>
    {
        public override void OnEnd(LogRecord data)
        {
            string? requestId = TelemetryContext.RequestId;
            string? protocol = TelemetryContext.Protocol;
            if (string.IsNullOrEmpty(requestId) && string.IsNullOrEmpty(protocol))
                return;

            IReadOnlyList<KeyValuePair<string, object?>>? attributes = data.Attributes;
            int count = attributes?.Count ?? 0;
            var enriched = new List<KeyValuePair<string, object?>>(count + 2);
            if (attributes != null)
            {
                for (int i = 0; i < count; i++)
                    enriched.Add(attributes[i]);
            }
            if (!string.IsNullOrEmpty(requestId))
                enriched.Add(new KeyValuePair<string, object?>(TelemetryAttributes.RequestId, requestId));
            if (!string.IsNullOrEmpty(protocol))
                enriched.Add(new KeyValuePair<string, object?>(TelemetryAttributes.Protocol, protocol));
            data.Attributes = enriched;
        }
    }
}
