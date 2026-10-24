// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;

namespace TensorSharp.Models.Moe
{
    /// <summary>
    /// Owns loaded <see cref="MoeModule"/> instances independently of any model, so
    /// expert modules can be loaded, unloaded, and registered at runtime without
    /// reloading the base model. Modules are immutable after load, so reads are
    /// lock-free; <see cref="Load"/> replaces an existing id and
    /// <see cref="RegisterInProcess"/> adopts an already-built module (the learning
    /// use case).
    ///
    /// <para>
    /// <see cref="ActiveSelection"/> is pure data — which module maps to which layer.
    /// Wiring that selection into a running model's expert dispatch is not part of
    /// this registry; that integration is the forward-path work tracked separately.
    /// </para>
    /// </summary>
    public sealed class MoeRegistry : IDisposable
    {
        private readonly ConcurrentDictionary<string, MoeModule> _modules = new(StringComparer.Ordinal);

        /// <summary>Load a module from a safetensors file and register it under <paramref name="id"/>.</summary>
        /// <exception cref="InvalidDataException">The file is malformed.</exception>
        /// <exception cref="FileNotFoundException">The file is missing.</exception>
        public MoeModule Load(string id, string safetensorsPath)
        {
            if (string.IsNullOrWhiteSpace(id))
                throw new ArgumentException("A MoE module id is required.", nameof(id));

            var module = MoeModuleLoader.Load(id, safetensorsPath);
            _modules[id] = module;
            return module;
        }

        /// <summary>
        /// Register a module whose weights were created in-process (e.g. by a learning
        /// process). The caller retains ownership of the buffers; the registry only
        /// holds a reference and does not copy.
        /// </summary>
        public MoeModule RegisterInProcess(MoeModule module)
        {
            if (module == null)
                throw new ArgumentNullException(nameof(module));
            _modules[module.Id] = module;
            return module;
        }

        /// <summary>Unload a module by id. Returns true when it was loaded.</summary>
        public bool Unload(string id) => _modules.TryRemove(id, out _);

        /// <summary>The module registered under <paramref name="id"/>, or null.</summary>
        public MoeModule? Get(string id) => _modules.TryGetValue(id, out var module) ? module : null;

        /// <summary>True when a module with this id is loaded.</summary>
        public bool Contains(string id) => _modules.ContainsKey(id);

        /// <summary>
        /// The module selection currently in force: per-layer overrides keyed by id.
        /// Null means the base model's original experts on every layer.
        /// </summary>
        public MoeSelection? ActiveSelection { get; set; }

        /// <summary>Snapshot of loaded modules, ordered by id for a stable listing.</summary>
        public IReadOnlyList<MoeModuleInfo> List() =>
            _modules.Values
                .OrderBy(m => m.Id, StringComparer.Ordinal)
                .Select(m => new MoeModuleInfo(
                    m.Id,
                    m.Architecture,
                    m.ExpertType,
                    m.Experts.Count,
                    m.Experts.Count == 0 ? 0 : m.Experts.Values.First().NumExperts,
                    m.Experts.Values.Any(e => e.HasRouter)))
                .ToList();

        public void Dispose()
        {
            // Modules own only managed float[] buffers; clearing releases them once no
            // active model reference remains.
            _modules.Clear();
        }
    }
}