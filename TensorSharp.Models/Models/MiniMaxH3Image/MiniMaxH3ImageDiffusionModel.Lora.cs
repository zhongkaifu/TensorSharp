// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// MiniMax-H3-Image's LoRA plug-ins (ILoraPluginHost): the hosts' --lora / --lora-scale / --lora-config (published to the
// model as TS_LORAS), a request's LoRA list (WebUiChatService's ApplyLoras) and SetLoras all land here. The set is loaded
// and validated against the denoiser before it replaces the previous one, then swapped under the generation lock: the
// denoiser forgets the old set's plan and device copies first, so nothing of one set reaches a request that runs with
// another (or with none).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3Image
{
    public sealed partial class MiniMaxH3ImageDiffusionModel : ILoraPluginHost
    {
        private MiniMaxH3ImageLoraSet _lora;
        private AdapterSet _adapters = AdapterSet.Empty;

        /// <inheritdoc/>
        public string LoraPluginType => LoraCliFlags.MiniMaxH3ImagePluginType;

        /// <summary>The LoRA plug-ins in use, as <see cref="LoraCliFlags.Resolve"/> expanded them (empty when none).</summary>
        public IReadOnlyList<LoraSpec> LoraSpecs { get; private set; } = Array.Empty<LoraSpec>();

        /// <summary>A <c>minimax-h3-image-lora</c> plug-in config, or bare weights (a style adapter); another family's plug-in
        /// is not. Reads the config only.</summary>
        public bool AcceptsLora(LoraSpec spec)
        {
            string declared = LoraCliFlags.DeclaredPluginType(spec);
            return declared == null || declared == LoraCliFlags.MiniMaxH3ImagePluginType;
        }

        /// <summary>The speed adapter's sampling contract in force, or null.</summary>
        public SamplingContract SamplingContract => _adapters.Contract;

        /// <summary>The loaded set (a test seam).</summary>
        internal MiniMaxH3ImageLoraSet Loras => _lora;

        /// <summary>The adapters every later request runs with.</summary>
        internal AdapterSet Adapters => _adapters;

        /// <summary>
        /// Replace the LoRA plug-ins applied to every later request (an empty list removes them): any number of style
        /// adapters, each with its own strength, and at most one speed adapter. The set is loaded and checked against the
        /// denoiser before anything changes; a failure leaves the previous set in place.
        /// </summary>
        public void SetLoras(IReadOnlyList<LoraSpec> specs)
        {
            ArgumentNullException.ThrowIfNull(specs);
            ObjectDisposedException.ThrowIf(_disposed, this);
            List<LoraSpec> resolved = LoraCliFlags.Resolve(specs);
            MiniMaxH3ImageLoraSet loaded = null;
            if (resolved.Count > 0)
            {
                var timer = Stopwatch.StartNew();
                loaded = MiniMaxH3ImageLoraSet.Load(resolved, DitConfig, _backend);
                try
                {
                    CheckContract(loaded.Contract);
                }
                catch
                {
                    loaded.Dispose();
                    throw;
                }
                Console.WriteLine($"MiniMax-H3-Image LoRA: {resolved.Count} plug-in(s), applied unmerged " +
                                  $"({loaded.FactorBytes / (1024.0 * 1024.0):F0} MiB of factors, loaded in {timer.Elapsed.TotalSeconds:F1}s)");
                Console.WriteLine(loaded.Summary);
            }
            lock (_generationLock)
            {
                if (_disposed)
                {
                    loaded?.Dispose();
                    throw new ObjectDisposedException(nameof(MiniMaxH3ImageDiffusionModel));
                }
                // The denoiser drops the plan and the device copies that bind the old set before it is freed.
                _family.Denoiser?.ReleaseLora();
                MiniMaxH3ImageLoraSet old = _lora;
                _lora = loaded;
                _adapters = loaded?.AdapterSet ?? AdapterSet.Empty;
                _family.Lora = loaded;
                LoraSpecs = resolved;
                old?.Dispose();
            }
        }

        /// <summary>A speed adapter's contract against the checkpoint: by-step heads need a head of their own for every step.</summary>
        private void CheckContract(SamplingContract contract)
        {
            if (contract == null) return;
            int heads = _family.Grid.Count - 1;
            if (contract.Heads == HeadPolicy.ByStep && contract.Steps > heads)
                throw new ArgumentException(
                    $"{contract.Name} samples {contract.Steps} steps with a head of their own each (by-step), but this checkpoint has {heads} heads.");
            contract.Apply(_family.DefaultRecipe);   // the structural checks a run makes
        }

        /// <summary>The plug-ins the host published (TS_LORAS), the ones this model takes. Configuration mistakes are load
        /// refusals (one line), not crashes.</summary>
        private void ApplyInheritedLoras()
        {
            try
            {
                // TS_LORAS may also carry another family's plug-ins; the host names what this model skips.
                var specs = LoraCliFlags.FromJson(Environment.GetEnvironmentVariable(LoraCliFlags.EnvironmentVariable))
                    .Where(AcceptsLora).ToList();
                if (specs.Count > 0) SetLoras(specs);
            }
            catch (Exception e) when (e is ArgumentException or System.Text.Json.JsonException or FormatException or
                KeyNotFoundException or System.IO.InvalidDataException or System.IO.FileNotFoundException)
            {
                _pipeline.Dispose();
                throw new ModelLoadRefusedException("LoRA plug-in refused: " + e.Message, e);
            }
        }

        /// <summary>Free the set (the model is disposing; its pipeline is gone).</summary>
        private void DisposeLoras()
        {
            _lora?.Dispose();
            _lora = null;
            _adapters = AdapterSet.Empty;
            _family.Lora = null;
        }
    }
}
