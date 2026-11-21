// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>
    /// The set of architecture plug-ins this process can load, keyed by every
    /// <c>general.architecture</c> string that selects them.
    ///
    /// Built-ins register themselves through <see cref="BuiltInArchitectures"/> on
    /// first use. <see cref="Register"/> is public so a host application can add an
    /// architecture without forking TensorSharp.
    /// </summary>
    public static class ModelArchitectureRegistry
    {
        private static readonly object Gate = new();
        private static readonly Dictionary<string, ModelArchitectureDescriptor> ByAlias =
            new(StringComparer.OrdinalIgnoreCase);
        private static readonly List<ModelArchitectureDescriptor> Ordered = new();
        private static bool _builtInsLoaded;

        /// <summary>Add an architecture plug-in. Idempotent for an identical descriptor
        /// instance; conflicting aliases are rejected loudly rather than silently
        /// shadowing an existing family.</summary>
        public static void Register(ModelArchitectureDescriptor descriptor)
        {
            ArgumentNullException.ThrowIfNull(descriptor);
            descriptor.Validate();

            lock (Gate)
            {
                foreach (string alias in descriptor.Aliases)
                {
                    if (ByAlias.TryGetValue(alias, out var existing) && !ReferenceEquals(existing, descriptor))
                    {
                        throw new InvalidOperationException(
                            $"Architecture alias '{alias}' is already registered by '{existing.Id}'; " +
                            $"'{descriptor.Id}' cannot claim it too.");
                    }
                }

                if (Ordered.Contains(descriptor))
                    return;

                foreach (string alias in descriptor.Aliases)
                    ByAlias[alias] = descriptor;
                Ordered.Add(descriptor);
            }
        }

        /// <summary>All registered architectures, in registration order.</summary>
        public static IReadOnlyList<ModelArchitectureDescriptor> All
        {
            get
            {
                EnsureBuiltIns();
                lock (Gate) return Ordered.ToArray();
            }
        }

        /// <summary>Every architecture id a model file may declare.</summary>
        public static IReadOnlyCollection<string> KnownAliases
        {
            get
            {
                EnsureBuiltIns();
                lock (Gate) return ByAlias.Keys.ToArray();
            }
        }

        public static bool TryGet(string architecture, out ModelArchitectureDescriptor descriptor)
        {
            EnsureBuiltIns();
            lock (Gate)
                return ByAlias.TryGetValue(architecture ?? string.Empty, out descriptor);
        }

        /// <summary>
        /// Pick the plug-in for a model file. <paramref name="architecture"/> is the
        /// GGUF's <c>general.architecture</c>, which may be null: a file with no
        /// metadata at all is offered to every registered detector first. If no
        /// detector claims it, loading fails because guessing a model architecture
        /// from an unlabelled tensor file is unsafe.
        /// </summary>
        public static ModelArchitectureDescriptor Resolve(string architecture, GgufFile probe)
        {
            EnsureBuiltIns();

            if (!string.IsNullOrEmpty(architecture))
            {
                if (TryGet(architecture, out var byName))
                    return byName;

                // A generic label (llama) is never an alias: it is claimed per file, by
                // the one family whose own recogniser matches the file's contents.
                var relabelClaimants = All.Where(d => d.RecognizeRelabelledFile != null).ToArray();
                if (probe != null)
                {
                    foreach (var candidate in relabelClaimants)
                    {
                        if (candidate.RecognizeRelabelledFile(architecture, probe))
                            return candidate;
                    }
                }

                string relabelHint = relabelClaimants.Length == 0
                    ? string.Empty
                    : " No family that also accepts a differently labelled file recognised this one: " +
                      string.Join("; ", relabelClaimants.Select(d => $"{d.Id} ({d.RelabelledFileDescription})")) + ".";
                throw new NotSupportedException(
                    $"Unsupported architecture: {architecture}. Registered: " +
                    string.Join(", ", All.Select(d => d.Id).OrderBy(x => x, StringComparer.Ordinal)) + "." +
                    relabelHint);
            }

            if (probe != null)
            {
                foreach (var candidate in All)
                {
                    if (candidate.DetectFromTensors != null && candidate.DetectFromTensors(probe))
                        return candidate;
                }
            }

            throw new NotSupportedException(
                "The model declares no architecture and no registered tensor-layout detector recognized it.");
        }

        /// <summary>Whether <paramref name="path"/> names a safetensors model file (by its extension), which the loader
        /// offers to the families' <see cref="ModelArchitectureDescriptor.DetectSafetensors"/> instead of opening it as
        /// a GGUF.</summary>
        public static bool IsSafetensorsPath(string path) =>
            path != null && path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase);

        /// <summary>
        /// The family that claims the safetensors model file <paramref name="path"/>, offered to every registered
        /// <see cref="ModelArchitectureDescriptor.DetectSafetensors"/> in registration order (the first claim wins), or
        /// null when the path is not a safetensors file, cannot be read as one, or no family claims it. Reads the
        /// header only; nothing is mapped.
        /// </summary>
        public static ModelArchitectureDescriptor ResolveSafetensors(string path)
        {
            if (!IsSafetensorsPath(path) || !System.IO.File.Exists(path))
                return null;
            var detectors = All.Where(d => d.DetectSafetensors != null).ToArray();
            if (detectors.Length == 0)
                return null;
            SafetensorsFile file;
            try
            {
                file = new SafetensorsFile(path);
            }
            catch (Exception e) when (e is System.IO.IOException or System.IO.InvalidDataException or NotSupportedException
                or UnauthorizedAccessException or System.Text.Json.JsonException or KeyNotFoundException
                or InvalidOperationException or FormatException or OverflowException)
            {
                // Not a readable safetensors header: no family claims it, and the caller reports the file as it would
                // any unrecognised model.
                return null;
            }
            using (file)
            {
                foreach (var candidate in detectors)
                {
                    if (candidate.DetectSafetensors(file))
                        return candidate;
                }
                return null;
            }
        }

        /// <summary>
        /// Locate the mmproj companion for a loaded model when the operator did not
        /// name one, using the architecture's own <see
        /// cref="ModelArchitectureDescriptor.ProjectorFileHints"/>. Returns null when
        /// the family declares no hints or nothing beside the model matches.
        ///
        /// Hosts call this only after deciding the model can actually use a projector
        /// (it implements a multimodal capability), so this never has to know which
        /// families are multimodal.
        /// </summary>
        public static string FindCompanionProjector(string architecture, string modelPath)
            => string.IsNullOrEmpty(modelPath) ? null : FindProjectorIn(architecture, System.IO.Path.GetDirectoryName(modelPath));

        /// <summary>
        /// The projector an <c>--mmproj</c> argument names. A file is used as given. A directory is
        /// resolved to the family's companion inside it (the <see cref="ModelArchitectureDescriptor.ProjectorFileHints"/>
        /// the beside-the-model lookup uses): the DeepSeek V4.1 CLI opened
        /// <c>--mmproj /workspace/models/deepseek/</c> as a GGUF, failed on the magic bytes and aborted with a
        /// core dump. A directory with no such file throws <see cref="System.IO.FileNotFoundException"/>, a
        /// load refusal, naming what it looked for.
        /// </summary>
        public static string ResolveProjectorPath(string architecture, string mmProjPath)
        {
            if (string.IsNullOrWhiteSpace(mmProjPath) || !System.IO.Directory.Exists(mmProjPath))
                return mmProjPath;
            string found = FindProjectorIn(architecture, mmProjPath);
            if (found != null)
                return found;
            string hints = TryGet(architecture, out var descriptor) && descriptor.ProjectorFileHints.Count > 0
                ? string.Join(", ", descriptor.ProjectorFileHints) : "(this architecture names no projector file)";
            throw new System.IO.FileNotFoundException(
                $"--mmproj names the directory {mmProjPath}, which holds no {architecture} projector ({hints}); " +
                "pass the projector file.", mmProjPath);
        }

        private static string FindProjectorIn(string architecture, string directory)
        {
            if (!TryGet(architecture, out var descriptor) || descriptor.ProjectorFileHints.Count == 0)
                return null;
            if (string.IsNullOrEmpty(directory) || !System.IO.Directory.Exists(directory))
                return null;

            foreach (string hint in descriptor.ProjectorFileHints)
            {
                if (!hint.Contains('*'))
                {
                    string exact = System.IO.Path.Combine(directory, hint);
                    if (System.IO.File.Exists(exact))
                        return exact;
                    continue;
                }

                // Ordered so the pick is stable when several companions sit side by side.
                string[] matches = System.IO.Directory.GetFiles(directory, hint);
                if (matches.Length > 0)
                {
                    Array.Sort(matches, StringComparer.OrdinalIgnoreCase);
                    return matches[0];
                }
            }

            return null;
        }

        private static void EnsureBuiltIns()
        {
            lock (Gate)
            {
                if (_builtInsLoaded)
                    return;
                // Register() takes this same reentrant lock. Publish completion
                // only after every built-in is visible to concurrent resolvers.
                BuiltInArchitectures.RegisterAll();
                _builtInsLoaded = true;
            }
        }
    }
}
