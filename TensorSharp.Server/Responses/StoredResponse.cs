// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;

namespace TensorSharp.Server.Responses
{
    /// <summary>
    /// A completed Responses API object as returned to callers, cached so
    /// <c>GET /v1/responses/{id}</c> can serve it back. Stores the pre-serialised
    /// JSON body rather than the anonymous object graph so retrieval is just an
    /// echo of exactly what the client originally received.
    /// </summary>
    public sealed class StoredResponse
    {
        public required string Id { get; init; }
        public required string Json { get; init; }
        public DateTimeOffset CreatedAt { get; init; } = DateTimeOffset.UtcNow;
    }
}
