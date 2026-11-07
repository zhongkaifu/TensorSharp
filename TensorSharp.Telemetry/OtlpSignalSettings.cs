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
using System.Globalization;
using OpenTelemetry.Exporter;

namespace TensorSharp.Telemetry
{
    /// <summary>
    /// One signal's OTLP overrides, from the per-signal variables the OpenTelemetry
    /// specification defines (<c>OTEL_EXPORTER_OTLP_TRACES_ENDPOINT</c> and so on).
    ///
    /// <para>
    /// TensorSharp registers one OTLP exporter per signal (so a signal can go to OTLP, Azure
    /// Monitor, both or neither), and the .NET SDK's per-signal <c>AddOtlpExporter</c> reads
    /// only the generic <c>OTEL_EXPORTER_OTLP_*</c> variables; the cross-cutting
    /// <c>UseOtlpExporter</c> that reads the per-signal ones always turns every signal on and
    /// cannot be combined with anything else. So the per-signal values are applied here.
    /// A per-signal endpoint is used exactly as given, as the specification says; the
    /// generic endpoint keeps the SDK's own handling (it appends <c>/v1/traces</c> and so on
    /// for <c>http/protobuf</c>).
    /// </para>
    /// </summary>
    public sealed class OtlpSignalSettings
    {
        public static readonly OtlpSignalSettings Empty = new(null, null, null, null);

        private OtlpSignalSettings(Uri? endpoint, string? headers, OtlpExportProtocol? protocol, int? timeoutMilliseconds)
        {
            Endpoint = endpoint;
            Headers = headers;
            Protocol = protocol;
            TimeoutMilliseconds = timeoutMilliseconds;
        }

        public Uri? Endpoint { get; }
        public string? Headers { get; }
        public OtlpExportProtocol? Protocol { get; }
        public int? TimeoutMilliseconds { get; }

        public bool IsEmpty => Endpoint is null && Headers is null && Protocol is null && TimeoutMilliseconds is null;

        /// <summary>Apply to an exporter's options; fields left null keep the SDK's own value.</summary>
        public void ApplyTo(OtlpExporterOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            if (Protocol is { } protocol) options.Protocol = protocol;
            if (Endpoint is { } endpoint) options.Endpoint = endpoint;
            if (Headers is { } headers) options.Headers = headers;
            if (TimeoutMilliseconds is { } timeout) options.TimeoutMilliseconds = timeout;
        }

        internal static OtlpSignalSettings Resolve(string signal, Func<string, string?> get, string? endpoint)
        {
            string prefix = "OTEL_EXPORTER_OTLP_" + signal + "_";
            OtlpExportProtocol? protocol = null;
            if (get(prefix + "PROTOCOL") is { } rawProtocol)
            {
                protocol = rawProtocol.Trim().ToLowerInvariant() switch
                {
                    "grpc" => OtlpExportProtocol.Grpc,
                    "http/protobuf" => OtlpExportProtocol.HttpProtobuf,
                    _ => throw new ArgumentException(
                        $"{prefix}PROTOCOL={rawProtocol} is not supported; use grpc or http/protobuf."),
                };
            }

            int? timeout = null;
            if (get(prefix + "TIMEOUT") is { } rawTimeout)
            {
                if (!int.TryParse(rawTimeout, NumberStyles.Integer, CultureInfo.InvariantCulture, out int ms) || ms <= 0)
                    throw new ArgumentException($"{prefix}TIMEOUT={rawTimeout} is not a positive number of milliseconds.");
                timeout = ms;
            }

            Uri? uri = endpoint is null ? null : new Uri(endpoint, UriKind.Absolute);
            string? headers = get(prefix + "HEADERS");
            if (uri is null && headers is null && protocol is null && timeout is null)
                return Empty;
            return new OtlpSignalSettings(uri, headers, protocol, timeout);
        }
    }
}
