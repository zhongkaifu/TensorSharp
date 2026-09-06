// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Globalization;
using Microsoft.Extensions.Caching.Memory;

namespace TensorSharp.Server.Responses
{
    /// <summary>
    /// Process-lifetime <see cref="IResponsesStore"/> backed by
    /// <see cref="MemoryCache"/> (the framework's own bounded in-memory cache),
    /// rather than a hand-rolled unbounded dictionary. Responses vanish on
    /// restart and there is no cross-request conversation chaining
    /// (<c>previous_response_id</c> is rejected by the adapter); this only
    /// backs <c>GET /v1/responses/{id}</c> for as long as an entry survives.
    ///
    /// Bounded two ways so a client that always sends <c>store=true</c> (the
    /// API default) can't grow this without limit: a per-entry TTL
    /// (<c>TS_RESPONSES_STORE_TTL_MINUTES</c>, default 60) and a max entry
    /// count (<c>TS_RESPONSES_STORE_MAX_ENTRIES</c>, default 1000) enforced via
    /// <see cref="MemoryCacheOptions.SizeLimit"/>, which evicts under memory
    /// pressure once the limit is hit.
    /// </summary>
    public sealed class InMemoryResponsesStore : IResponsesStore, IDisposable
    {
        private const int DefaultMaxEntries = 1000;
        private static readonly TimeSpan DefaultTtl = TimeSpan.FromHours(1);

        private readonly MemoryCache _cache;
        private readonly TimeSpan _ttl;

        public InMemoryResponsesStore() : this(ResolveMaxEntries(), ResolveTtl())
        {
        }

        internal InMemoryResponsesStore(int maxEntries, TimeSpan ttl)
        {
            _ttl = ttl;
            _cache = new MemoryCache(new MemoryCacheOptions { SizeLimit = maxEntries });
        }

        public void Store(StoredResponse response)
        {
            _cache.Set(response.Id, response, new MemoryCacheEntryOptions
            {
                Size = 1,
                AbsoluteExpirationRelativeToNow = _ttl,
            });
        }

        public bool TryGet(string id, out StoredResponse response)
        {
            return _cache.TryGetValue(id, out response);
        }

        public void Dispose() => _cache.Dispose();

        private static int ResolveMaxEntries()
        {
            string raw = Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_MAX_ENTRIES")?.Trim();
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n) && n > 0
                ? n
                : DefaultMaxEntries;
        }

        private static TimeSpan ResolveTtl()
        {
            string raw = Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_TTL_MINUTES")?.Trim();
            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double minutes) && minutes > 0
                ? TimeSpan.FromMinutes(minutes)
                : DefaultTtl;
        }
    }
}
