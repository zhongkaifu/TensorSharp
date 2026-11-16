// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using TensorSharp.Server.ProtocolAdapters;

namespace TensorSharp.Server.Endpoints;

/// <summary>
/// OpenAI's audio routes (transcriptions, translations, speech) and a music route, served by
/// whichever audio model is loaded (Whisper, Kokoro, ACE-Step).
/// </summary>
public static class AudioEndpoints
{
    public static IEndpointRouteBuilder MapAudioEndpoints(this IEndpointRouteBuilder endpoints)
    {
        endpoints.MapPost("/v1/audio/transcriptions",
            (HttpRequest req, AudioAdapter adapter) => adapter.TranscriptionsAsync(req, translate: false))
            .DisableRequestTimeout();
        endpoints.MapPost("/v1/audio/translations",
            (HttpRequest req, AudioAdapter adapter) => adapter.TranscriptionsAsync(req, translate: true))
            .DisableRequestTimeout();
        endpoints.MapPost("/v1/audio/speech",
            (HttpRequest req, AudioAdapter adapter) => adapter.SpeechAsync(req))
            .DisableRequestTimeout();
        endpoints.MapPost("/v1/audio/music",
            (HttpRequest req, AudioAdapter adapter) => adapter.MusicAsync(req))
            .DisableRequestTimeout();
        return endpoints;
    }
}
