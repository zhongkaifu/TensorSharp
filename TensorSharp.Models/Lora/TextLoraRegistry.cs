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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models.Lora
{
    /// <summary>
    /// Owns loaded <see cref="TextLoraAdapter"/> instances independently of any model, so
    /// adapters can be loaded and unloaded at runtime without reloading the base model.
    ///
    /// <para>
    /// Adapters are immutable after load, so the read path is lock-free: <see cref="Get"/>
    /// and <see cref="List"/> only observe the <see cref="ConcurrentDictionary{TKey, TValue}"/>.
    /// <see cref="Load"/> replaces an existing entry under the same name (reload semantics),
    /// and <see cref="Unload"/> removes one. A caller must not detach or replace an adapter
    /// that a model currently holds active; the model's <c>SetActiveLora</c> grabs a strong
    /// reference, so an unloaded adapter that is still active stays alive by that reference.
    /// </para>
    /// </summary>
    public sealed class TextLoraRegistry : IDisposable
    {
        private readonly ConcurrentDictionary<string, TextLoraAdapter> _adapters = new(StringComparer.Ordinal);

        /// <summary>
        /// Load (or reload) an adapter under <paramref name="name"/> and register it.
        /// </summary>
        /// <exception cref="InvalidDataException">The file or config is malformed.</exception>
        /// <exception cref="FileNotFoundException">A named file is missing.</exception>
        public TextLoraAdapter Load(string name, string safetensorsPath, string? configPath = null, float? scaleOverride = null)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw new ArgumentException("A LoRA adapter name is required.", nameof(name));

            var adapter = TextLoraLoader.Load(name, safetensorsPath, configPath, scaleOverride);
            _adapters[name] = adapter;
            return adapter;
        }

        /// <summary>Unload and stop advertising an adapter by name. Returns true when it was loaded.</summary>
        public bool Unload(string name) => _adapters.TryRemove(name, out _);

        /// <summary>The adapter registered under <paramref name="name"/>, or null.</summary>
        public TextLoraAdapter? Get(string name) => _adapters.TryGetValue(name, out var adapter) ? adapter : null;

        /// <summary>True when an adapter with this name is loaded.</summary>
        public bool Contains(string name) => _adapters.ContainsKey(name);

        /// <summary>Snapshot of loaded adapters, ordered by name for a stable listing.</summary>
        public IReadOnlyList<LoraAdapterInfo> List() =>
            _adapters.Values
                .OrderBy(a => a.Name, StringComparer.Ordinal)
                .Select(a => new LoraAdapterInfo(a.Name, a.Rank, a.Alpha, a.Deltas.Count))
                .ToList();

        public void Dispose()
        {
            // The registry holds no unmanaged resources: the adapters own only managed
            // float[] buffers, so clearing the dictionary releases them once no active
            // model reference remains.
            _adapters.Clear();
        }
    }
}