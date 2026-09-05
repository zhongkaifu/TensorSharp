// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

namespace TensorSharp.Server.Responses
{
    /// <summary>
    /// Persistence boundary for completed <c>/v1/responses</c> objects, keyed
    /// by their <c>resp_...</c> id. The only implementation today is
    /// <see cref="InMemoryResponsesStore"/> (process-lifetime only, no
    /// <c>previous_response_id</c> chaining), but callers only depend on this
    /// interface so a durable store can be swapped in later via DI without
    /// touching the adapter.
    /// </summary>
    public interface IResponsesStore
    {
        void Store(StoredResponse response);

        bool TryGet(string id, out StoredResponse response);
    }
}
