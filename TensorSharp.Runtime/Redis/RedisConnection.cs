// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using StackExchange.Redis;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Runtime.Redis
{
    /// <summary>
    /// <see cref="IRedisKeyValueStore"/> backed by a StackExchange.Redis
    /// <see cref="ConnectionMultiplexer"/>. The connection is established
    /// eagerly in the constructor and reconnects automatically (the
    /// multiplexer's default behaviour). All operations are synchronous
    /// from the caller's perspective; the multiplexer pipelines them
    /// internally.
    /// </summary>
    public sealed class RedisConnection : IRedisKeyValueStore
    {
        private readonly ConnectionMultiplexer _mux;
        private readonly ILogger _logger;

        public RedisConnection(string connectionString, ILogger? logger = null)
        {
            if (string.IsNullOrWhiteSpace(connectionString))
                throw new ArgumentException("Redis connection string must be non-empty.", nameof(connectionString));
            _logger = logger ?? NullLogger.Instance;
            _mux = ConnectionMultiplexer.Connect(connectionString);
        }

        internal RedisConnection(ConnectionMultiplexer mux, ILogger? logger = null)
        {
            _mux = mux ?? throw new ArgumentNullException(nameof(mux));
            _logger = logger ?? NullLogger.Instance;
        }

        public bool IsConnected => _mux.IsConnected;

        private IDatabase Db => _mux.GetDatabase();

        public bool StringSet(string key, byte[] value, TimeSpan? expiry)
        {
            try
            {
                return Db.StringSet(key, value, expiry);
            }
            catch (RedisException ex)
            {
                _logger.LogWarning(ex, "Redis SET failed for key {Key}", LogContent.Of(key));
                return false;
            }
        }

        public byte[]? StringGet(string key)
        {
            try
            {
                RedisValue value = Db.StringGet(key);
                return value.HasValue ? (byte[]?)value : null;
            }
            catch (RedisException ex)
            {
                // A key can name a stored response (tsresp:<id>), whose id is a bearer key to it.
                _logger.LogWarning(ex, "Redis GET failed for key {Key}", LogContent.Of(key));
                return null;
            }
        }

        public void KeyDelete(string[] keys)
        {
            if (keys == null || keys.Length == 0)
                return;
            try
            {
                Db.KeyDelete(keys.Select(k => (RedisKey)k).ToArray());
            }
            catch (RedisException ex)
            {
                _logger.LogWarning(ex, "Redis DEL failed for {Count} keys", keys.Length);
            }
        }

        public string[] ScanKeys(string pattern)
        {
            var result = new List<string>();
            try
            {
                var server = _mux.GetServer(_mux.GetEndPoints().First());
                foreach (var key in server.Keys(pattern: pattern))
                    result.Add(key.ToString());
            }
            catch (RedisException ex)
            {
                _logger.LogWarning(ex, "Redis SCAN failed for pattern {Pattern}", pattern);
            }
            return result.ToArray();
        }

        public void Dispose() => _mux.Dispose();
    }
}
