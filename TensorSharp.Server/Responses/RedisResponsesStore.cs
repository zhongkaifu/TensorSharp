// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Globalization;
using System.Text;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Redis;

namespace TensorSharp.Server.Responses
{
    /// <summary>
    /// Redis-backed <see cref="IResponsesStore"/> that persists completed
    /// <c>/v1/responses</c> objects in a shared Redis instance, enabling
    /// <c>GET /v1/responses/{id}</c> to survive process restarts and work
    /// across multiple server instances behind a load balancer.
    ///
    /// Each entry is stored as the pre-serialised JSON body under the key
    /// <c>tsresp:{id}</c> with a configurable TTL
    /// (<c>TS_RESPONSES_STORE_TTL_MINUTES</c>, default 60). The max-entry
    /// bound enforced by <see cref="InMemoryResponsesStore"/> is replaced by
    /// Redis's own <c>maxmemory</c> eviction policy.
    /// </summary>
    public sealed class RedisResponsesStore : IResponsesStore, IDisposable
    {
        private const string KeyPrefix = "tsresp:";
        private static readonly TimeSpan DefaultTtl = TimeSpan.FromHours(1);

        private readonly IRedisKeyValueStore _redis;
        private readonly TimeSpan _ttl;
        private readonly ILogger _logger;

        public RedisResponsesStore(IRedisKeyValueStore redis, ILogger logger = null)
            : this(redis, ResolveTtl(), logger)
        {
        }

        internal RedisResponsesStore(IRedisKeyValueStore redis, TimeSpan ttl, ILogger logger = null)
        {
            _redis = redis ?? throw new ArgumentNullException(nameof(redis));
            _ttl = ttl;
            _logger = logger ?? NullLogger.Instance;
        }

        public void Store(StoredResponse response)
        {
            string key = KeyPrefix + response.Id;
            byte[] json = Encoding.UTF8.GetBytes(response.Json);
            if (!_redis.StringSet(key, json, _ttl))
            {
                _logger.LogWarning("Failed to store response {ResponseId} in Redis", response.Id);
            }
        }

        public bool TryGet(string id, out StoredResponse response)
        {
            string key = KeyPrefix + id;
            byte[] json = _redis.StringGet(key);
            if (json == null)
            {
                response = null;
                return false;
            }

            response = new StoredResponse
            {
                Id = id,
                Json = Encoding.UTF8.GetString(json),
            };
            return true;
        }

        public void Dispose() => _redis.Dispose();

        private static TimeSpan ResolveTtl()
        {
            string raw = Environment.GetEnvironmentVariable("TS_RESPONSES_STORE_TTL_MINUTES")?.Trim();
            return double.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out double minutes) && minutes > 0
                ? TimeSpan.FromMinutes(minutes)
                : DefaultTtl;
        }
    }
}
